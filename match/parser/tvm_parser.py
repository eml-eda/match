# TVM imports
import numpy as np
from match.dim.dim import MatchDim
from match.node.node import MatchNode
from match.tensor.tensor import MatchTensor
import tvm
import tvm.relay
from tvm.relay.transform import InferType
# utils imports
from typing import List
from match.target.exec_module import ExecModule

def get_depth_arr_pattern(pattern_inst_):
    if isinstance(pattern_inst_,tvm.relay.dataflow_pattern.AttrPattern):
        pattern_inst_=pattern_inst_.pattern
    if isinstance(pattern_inst_,tvm.relay.dataflow_pattern.AltPattern):
        return [get_depth_arr_pattern(pattern_inst_=pattern_inst_.left),get_depth_arr_pattern(pattern_inst_=pattern_inst_.right)]
    elif isinstance(pattern_inst_,tvm.relay.dataflow_pattern.CallPattern):
        return [(pattern_inst_.op.expr.name,get_depth_arr_pattern(arg_)) for arg_ in pattern_inst_.args]
    else:
        return 0

LAYOUTS_RESHAPES_AND_TRANSFORMS_OF_CONSTANTS_OPS = ("reshape","transpose","expand_dims","squeeze","cast","reshape_like","transpose_like","layout_transform")

class MatchTVMParser:
    def __init__(self, node:tvm.ir.IRModule, args_list:List=[],
                 exec_module:ExecModule=None, pattern_name:str="",
                 partitioned:bool=False, pattern_inst=None,
                 match_node:MatchNode=None
                ):
        self.exec_module=exec_module
        self.args_list=args_list
        self.pattern_name=pattern_name
        self.node = node
        self.match_node = match_node
        self.index = 0
        self.callsindexes = dict()
        self.calllist = list()
        self.node_vars = dict()
        self.node_consts = dict()
        self.energy = 0
        self.latency = 0
        self.pattern_inst=pattern_inst
        self.occurrences = dict()
        self.calls_to_name = dict()
        self.name_to_calls = dict()
        self.calls_tensors = dict()
        self.node_all_dims = dict()
        self.visit_router = dict()
        self.tensor_name_mapping = dict()
        self.original_node = node
        # partition graph and infer type
        if not partitioned:
            mod = tvm.ir.IRModule()
            self.node=InferType()(mod.from_expr(pattern_inst.pattern().partition(node)))
            self.args_list = self.node["main"].body.args
            params_ = self.node["main"].body.op.params
            new_args = []
            for idx,arg in enumerate(self.args_list):
                new_arg = arg
                # its actually a function
                if isinstance(new_arg, tvm.relay.Call) and not hasattr(new_arg.op, "name"):
                    new_arg = params_[idx]
                # its actually a function
                elif isinstance(new_arg, tvm.relay.Call) and isinstance(new_arg, tvm.relay.Function):
                    new_arg = params_[idx]
                elif isinstance(new_arg, tvm.relay.Call):
                    while isinstance(new_arg, tvm.relay.Call) and new_arg.op.name in LAYOUTS_RESHAPES_AND_TRANSFORMS_OF_CONSTANTS_OPS:
                        new_arg = new_arg.args[0]
                        if isinstance(new_arg, tvm.relay.Var) or isinstance(new_arg, tvm.relay.Constant) or (isinstance(new_arg, tvm.relay.Call) and not hasattr(new_arg.op, "name")):
                            new_arg = params_[idx]
                            break
                if not isinstance(new_arg, tvm.relay.Var) and not isinstance(new_arg, tvm.relay.Constant):
                    new_arg = params_[idx]
                new_args.append(new_arg)
            self.args_list = new_args
            self.node=self.node["main"].body.op.body

    def get_name_and_tensor_of_arg(self,call,arg,arg_idx:int=0):
        if isinstance(arg, tvm.relay.Var):
            if self.tensor_name_mapping[arg.name_hint] in self.node_vars:
                return arg.name_hint, self.node_vars[self.tensor_name_mapping[arg.name_hint]],"var"
            else:
                return arg.name_hint, self.node_consts[self.tensor_name_mapping[arg.name_hint]],"const"
        elif isinstance(arg, tvm.relay.Constant):
            return f"{self.calls_to_name[call]}_arg_{arg_idx}", self.node_consts[f"{self.calls_to_name[call]}_arg_{arg_idx}"],"var"
        else:
            return self.calls_to_name[arg], self.calls_tensors[self.calls_to_name[arg]],"call"

    def update_all_dim_names_occurrences_with(self,old_dim_name,new_dim_name):
        for k in self.node_all_dims.keys():
            if old_dim_name == self.node_all_dims[k].name:
                self.node_all_dims[k].name=new_dim_name

    def update_match_node(self,op=None,call=None,name:str="conv2d_3"):
        self.match_node.ops[name] = op
        self.match_node.calls[name] = call
        self.match_node.ops_occurrences[self.op_name(call)] = [name] if self.op_name(call) not in self.match_node.ops_occurrences else [*self.match_node.ops_occurrences[self.op_name(call)], name]

    def op_name(self,call):
        return "_".join([n for n in call.op.name.split(".") if n not in {"nn","op","relay"}])

    def adjust_name(self,name):
        return "_".join([n for n in name.split(".") if n not in {"nn","op","relay"}])

    def get_unique_name_and_update_occs(self,call):
        unique_name=self.op_name(call)
        if unique_name in self.occurrences:
            self.occurrences[unique_name]+=1
            unique_name+=f"_{self.occurrences[unique_name]}"
        else:
            self.occurrences[unique_name]=1
        return unique_name

    def check_broadcasting_and_get_out_dims(self, inp_tensor, w_tensor):
        w_tensor.layout = inp_tensor.layout
        # check if broadcasted
        broadcasted_tensor = None
        dim_idxs_to_remove = []
        axeses = []
        dim_idx = 0
        for inp_dim, w_dim in zip(inp_tensor.dims, w_tensor.dims):
            if inp_dim.size!=w_dim.size and inp_dim.size!=1 and w_dim.size!=1:
                raise RuntimeError(f"[TVM PARSER] Trying to do broadcast an operation which violates constraints,\
                                    shape A {[dim.size for dim in inp_tensor.dims]} shape B {[dim.size for dim in w_tensor.dims]}")
            if inp_dim.size!=w_dim.size:
                if broadcasted_tensor is None and w_dim.size==1:
                    broadcasted_tensor = w_tensor
                elif broadcasted_tensor is None and inp_dim.size==1:
                    broadcasted_tensor = inp_tensor
                if broadcasted_tensor==w_tensor and w_dim.size==1:
                    dim_idxs_to_remove.append(dim_idx)
                    self.update_all_dim_names_occurrences_with(old_dim_name=w_dim.name, new_dim_name=inp_dim.name)
                elif broadcasted_tensor==inp_tensor and inp_tensor.size==1:
                    dim_idxs_to_remove.append(dim_idx)
                    self.update_all_dim_names_occurrences_with(old_dim_name=inp_dim.name, new_dim_name=w_dim.name)
            else:
                axeses.append(dim_idx)
            dim_idx += 1

        other_tensor = inp_tensor if broadcasted_tensor is None or broadcasted_tensor==w_tensor else w_tensor
        if broadcasted_tensor is None:
            for dim_idx in range(len(inp_tensor.dims)):
                self.update_all_dim_names_occurrences_with(old_dim_name=w_tensor.dims[dim_idx].name, new_dim_name=inp_tensor.dims[dim_idx].name)
        else:
            sum_sizes_other_tensor = sum([dim.size for dim in other_tensor.dims])
            for dim_idx in range(len(broadcasted_tensor.dims)):
                if sum_sizes_other_tensor>1 and broadcasted_tensor.dims[dim_idx].size==1 and dim_idx not in dim_idxs_to_remove:
                    dim_idxs_to_remove.append(dim_idx)
                    axeses.remove(dim_idx)
                    self.update_all_dim_names_occurrences_with(old_dim_name=broadcasted_tensor.dims[dim_idx].name, new_dim_name=other_tensor.dims[dim_idx].name)
                else:
                    self.update_all_dim_names_occurrences_with(old_dim_name=broadcasted_tensor.dims[dim_idx].name, new_dim_name=other_tensor.dims[dim_idx].name)
            broadcasted_tensor.dims = [broadcasted_tensor.dims[dim_idx] for dim_idx in range(len(broadcasted_tensor.dims)) if dim_idx not in dim_idxs_to_remove]
            if broadcasted_tensor.tensor_type=="const":
                new_broadcasted_tensor_shape = tuple([dim.size for dim in broadcasted_tensor.dims])
                broadcasted_tensor.data = broadcasted_tensor.data.reshape(new_broadcasted_tensor_shape)
                broadcasted_tensor.num_dims = len(broadcasted_tensor.dims)
        if len(axeses) not in (1, len(other_tensor.dims)):
            raise RuntimeError(f"[TVM PARSER] Trying to do broadcast an operation which violates constraints,\
                                    shape A {[dim.size for dim in inp_tensor.dims]} shape B {[dim.size for dim in w_tensor.dims]}")
        return other_tensor.dims, axeses

    def get_io_from_layout(self, layout, data, dims):
        # conv2d and other 4 dims operators
        if len(data)==4:
            if layout=="NHWC":
                # layout is nhwc
                n = (int(data[0]), dims[0])
                c = (int(data[3]), dims[3])
                h = (int(data[1]), dims[1])
                w = (int(data[2]), dims[2])
            elif layout=="HWIO":
                n = (int(data[3]), dims[3])
                c = (int(data[2]), dims[2])
                h = (int(data[0]), dims[0])
                w = (int(data[1]), dims[1])
            elif layout=="NCHW" or layout=="OIHW":
                n = (int(data[0]), dims[0])
                c = (int(data[1]), dims[1])
                h = (int(data[2]), dims[2])
                w = (int(data[3]), dims[3])
            elif layout=="OIHW":
                n = (int(data[0]), dims[0])
                c = (int(data[1]), dims[1])
                h = (int(data[2]), dims[2])
                w = (int(data[3]), dims[3])
            elif layout=="OHWI":
                n = (int(data[0]), dims[0])
                c = (int(data[3]), dims[3])
                h = (int(data[1]), dims[1])
                w = (int(data[2]), dims[2])
            else:
                print(f"[PARSER]: Warning, layout {layout} not recognized, interpreting as NCHW")
                #layout is nchw
                n = (int(data[0]), dims[0])
                c = (int(data[1]), dims[1])
                h = (int(data[2]), dims[2])
                w = (int(data[3]), dims[3])
            return n, c, h, w
        # conv1d and other 3 dims operators
        elif len(data)==3:
            n = (int(data[0]), dims[0])
            c = (int(data[1]), dims[1])
            spat = (int(data[2]), dims[2])
            return n, c, spat

    def get_dim_arr_from_layout_and_nchw_arr(self,layout,nchw_arr):
        if layout=="NHWC":
            return [nchw_arr[0],nchw_arr[2],nchw_arr[3],nchw_arr[1]]
        elif layout=="NCHW":
            return nchw_arr
        else:
            print(f"[PARSER]: Warning, layout {layout} not recognized, interpreting as NCHW")
            #layout is nchw
            return nchw_arr
    
    def get_type(self, dtype):
        if dtype[:3] == "int":
            return "int"
        elif dtype[:4] == "uint":
            return "uint"
        # put here other cases
        return "int"

    def get_bits(self, dtype):
        if dtype[:3] == "int":
            return int(dtype[3:])
        elif dtype[:4] == "uint":
            return int(dtype[4:])
        # put here other cases
        return 8

    def get_bits_type(self, dtype):
        if dtype[:3] == "int":
            return int(dtype[3:]), "int"
        elif dtype[:4] == "uint":
            return int(dtype[4:]), "uint"
        # put here other cases
        return 8, "int"

    def visit_calls_with_depth(self,call,depth_limit=[]):
        if not isinstance(depth_limit,list):
            return
        if not isinstance(call, tvm.relay.Call):
            return
        # the call is just a function pre partitioned
        elif not isinstance(call.op,tvm.ir.Op):
            return
        self.calllist.append(call)
        for idx_arg_,arg_ in enumerate(call.args):
            self.visit_calls_with_depth(arg_,depth_limit=depth_limit[idx_arg_])
    
    def visit_calls(self,call):
        if isinstance(call,tvm.relay.Function):
            return self.visit_calls(call.body)
        if not isinstance(call, tvm.relay.Call):
            return
        # the call is just a function pre partitioned
        elif not isinstance(call.op,tvm.ir.Op):
            return
        self.calllist.append(call)
        for arg_ in call.args:
            self.visit_calls(arg_)
    
    def visit(self):
        call = self.node
        self.calllist=[]
        #self.visit_calls_with_depth(call,[self.depth_limits])
        self.visit_calls(call)
        self.calllist.reverse()
        var_and_consts_not_unrolled = dict()
        var_and_consts_unrolled = dict()
        for c in self.calllist:
            unique_name=self.get_unique_name_and_update_occs(c)
            # fit variables and constants
            for idx_arg,a in enumerate(c.args):
                if isinstance(a, tvm.relay.Var):
                    if len(self.args_list)>len(var_and_consts_not_unrolled):
                        ## this can be either a constant or the input still so let's check the real type
                        if isinstance(
                            self.args_list[len(var_and_consts_not_unrolled)], tvm.relay.Var
                        ):
                            v_name = "input_"+str(len(self.node_vars))
                            self.tensor_name_mapping[a.name_hint] = v_name
                            var_and_consts_not_unrolled[v_name] = self.args_list[
                                len(var_and_consts_not_unrolled)
                            ]
                            var_ = self.args_list[
                                len(var_and_consts_not_unrolled)-1
                            ]
                            shape = [int(v) if isinstance(v,tvm.tir.IntImm) else -1 for v in var_.checked_type.shape]
                            dtype = var_.checked_type.dtype
                            var_dims=[MatchDim(name=v_name+f"_dim_{idx}",size=shape[idx],is_dynamic=shape[idx]!=-1) for idx in range(len(shape))]
                            for dim in var_dims:
                                self.node_all_dims[dim.name]=dim
                            self.node_vars[v_name] = MatchTensor(name=v_name,
                                                                dims=var_dims,
                                                                dtype=np.dtype(dtype),tensor_type="var")
                        else:
                            v_name=unique_name + f"_arg_{idx_arg}"
                            self.tensor_name_mapping[a.name_hint] = v_name
                            var_and_consts_not_unrolled[v_name] = self.args_list[len(var_and_consts_not_unrolled)]
                            const_  = self.args_list[len(var_and_consts_not_unrolled)-1]
                            if isinstance(const_.checked_type, tvm.ir.type.TupleType):
                                breakpoint()
                            shape = [int(v) if isinstance(v,tvm.tir.IntImm) else -1 for v in const_.checked_type.shape]
                            dtype = const_.checked_type.dtype
                            const_dims=[MatchDim(name=v_name+f"_dim_{idx}",size=shape[idx],is_dynamic=shape[idx]!=-1) for idx in range(len(shape))]
                            for dim in const_dims:
                                self.node_all_dims[dim.name]=dim
                            self.node_consts[v_name] = MatchTensor(name=v_name,
                                                                dims=const_dims,
                                                                dtype=np.dtype(dtype),tensor_type="const",
                                                                data = const_.data.numpy())
                    else:
                        v_name = "input_"+str(len(self.node_vars))
                        self.tensor_name_mapping[a.name_hint] = v_name
                        var_and_consts_not_unrolled[v_name] = a
                        shape = [int(v) if isinstance(v,tvm.tir.IntImm) else -1 for v in a.checked_type.shape]
                        dtype = a.checked_type.dtype
                        var_dims=[MatchDim(name=v_name+f"_dim_{idx}",size=shape[idx],is_dynamic=shape[idx]!=-1) for idx in range(len(shape))]
                        for dim in var_dims:
                            self.node_all_dims[dim.name]=dim
                        self.node_vars[v_name] = MatchTensor(name=v_name,
                                                                dims=var_dims,
                                                                dtype=np.dtype(dtype),tensor_type="var")
                elif isinstance(a, tvm.relay.Constant):
                    v_name=unique_name + f"_arg_{idx_arg}"
                    self.tensor_name_mapping[v_name] = v_name
                    var_and_consts_unrolled[v_name] = a
                    shape = [int(v) if isinstance(v,tvm.tir.IntImm) else -1 for v in a.checked_type.shape]
                    dtype = a.checked_type.dtype
                    const_dims=[MatchDim(name=v_name+f"_dim_{idx}",size=shape[idx],is_dynamic=shape[idx]!=-1) for idx in range(len(shape))]
                    for dim in const_dims:
                        self.node_all_dims[dim.name]=dim
                    self.node_consts[v_name] = MatchTensor(name=v_name,
                                                            dims=const_dims,
                                                            dtype=np.dtype(dtype),tensor_type="const",
                                                            data = a.data.numpy())
            self.index += 1
            #self.ops_names.append(c.op.name)
            self.calls_to_name[c] = unique_name
            self.visit_call(c,unique_name)
            self.name_to_calls[unique_name] = c
            self.callsindexes[c] = self.index
        
        # save vars
        self.match_node.var_tensors = self.node_vars
        self.match_node.const_tensors = self.node_consts
        # add out to name of call tensors
        for call_tensor in self.calls_tensors.values():
            call_tensor.name = call_tensor.name + "_out"
        
        self.match_node.output_tensors = {t_name+"_out":t_value for t_name,t_value in self.calls_tensors.items() if t_value.tensor_type=="output"}
        self.match_node.intermediate_tensors = {t_name+"_out":t_value for t_name,t_value in self.calls_tensors.items() if t_value.tensor_type=="intermediate"}
        # remove useless dims
        # Find duplicate dims with different names but same properties     
        # Replace dims in tensors with duplicates removed
        for tensor_dict in [self.node_vars, self.node_consts, self.calls_tensors]:
            for _, tensor in tensor_dict.items():
                updated_dims = []
                for dim in tensor.dims:
                    if dim.original_name!=dim.name:
                        # Use the canonical dim instead
                        canonical_dim = self.node_all_dims[dim.name]
                        updated_dims.append(canonical_dim)
                    else:
                        updated_dims.append(dim)
                tensor.dims = updated_dims

        # Remove duplicate dims from node_all_dims
        all_dims_names = [d for d in self.node_all_dims.keys()]
        for dim_name in all_dims_names:
            if dim_name!=self.node_all_dims[dim_name].name:
                del self.node_all_dims[dim_name]
        
        self.match_node.dims = self.node_all_dims
        self.exec_module.update_constants(self.match_node, self.pattern_name)

    def visit_call(self, call, unique_name):
        if call.op.name not in self.visit_router:
            # skip if not supported for testing purposes
            print("[MATCH TVM PARSER] The operator",call.op.name,"is not supported yet")
            raise NotImplementedError(
                f"Currently the operator {call.op.name} is not supported."
            )
        else:
            self.visit_router[call.op.name](call, call.attrs, unique_name)