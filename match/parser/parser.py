# TVM imports
import numpy as np
from match.dim.dim import MatchDim
from match.node.node import MatchNode
from match.tensor.tensor import MatchTensor
import tvm
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
    
class MatchParser:
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
        # partition graph and infer type
        if not partitioned:
            mod = tvm.ir.IRModule()
            self.node=InferType()(mod.from_expr(pattern_inst.pattern().partition(node)))
            self.node=self.node["main"].body.op.body

    def get_name_and_tensor_of_arg(self,arg,arg_idx:int=0):
        if isinstance(arg, tvm.relay.Var):
            if arg.name_hint in self.node_vars:
                return arg.name_hint, self.node_vars[arg.name_hint],"var"
            else:
                return arg.name_hint, self.node_consts[arg.name_hint],"const"
        elif isinstance(arg, tvm.relay.Constant):
            return f"{self.op_name(arg)}_arg_{arg_idx}", self.node_consts[f"{self.op_name(arg)}_arg_{arg_idx}"],"var"
        else:
            return self.calls_to_name[arg], self.calls_tensors[self.calls_to_name[arg]],"call"

    def update_all_dim_names_occurrences_with(self,old_dim_name,new_dim_name):
        for k in self.node_all_dims.keys():
            if old_dim_name == self.node_all_dims[k].name:
                self.node_all_dims[k].name=new_dim_name

    def update_match_node(self,op=None,call=None,name:str="conv2d_3"):
        self.match_node.ops[name] = op
        self.match_node.calls[name] = call
        self.match_node.ops_occurrences[self.op_name(call)] = [name] if self.op_name(call) not in self.match_node.ops_occurrences else self.match_node.ops_occurrences[self.op_name(call)].append(name)

    def op_name(self,call):
        return "_".join([n for n in call.op.name.split(".") if n not in {"nn","op","relay"}])

    def get_unique_name_and_update_occs(self,call):
        unique_name=self.op_name(call)
        if unique_name in self.occurrences:
            unique_name+=f"_{self.occurrences[unique_name]}"
            self.occurrences[unique_name]+=1
        else:
            self.occurrences[unique_name]=1
        return unique_name

    def get_io_from_layout(self,layout, data, dims):
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
        else:
            print(f"[PARSER]: Warning, layout {layout} not recognized, interpreting as NCHW")
            #layout is nchw
            n = (int(data[0]), dims[0])
            c = (int(data[1]), dims[1])
            h = (int(data[2]), dims[2])
            w = (int(data[3]), dims[3])
        return n, c, h, w

    def get_dim_arr_from_layout_and_nchw_arr(self,layout,nchw_arr):
        if layout=="NHWC":
            return [nchw_arr[0],nchw_arr[3],nchw_arr[1],nchw_arr[2]]
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
            # fit variables and constants
            for idx_arg,a in enumerate(c.args):
                if isinstance(a, tvm.relay.Var):
                    if len(self.args_list)>len(var_and_consts_not_unrolled):
                        ## this can be either a constant or the input still so let's check the real type
                        if isinstance(
                            self.args_list[len(var_and_consts_not_unrolled)], tvm.relay.Var
                        ):
                            var_and_consts_not_unrolled[a.name_hint] = self.args_list[
                                len(var_and_consts_not_unrolled)
                            ]
                            var_ = self.args_list[
                                len(var_and_consts_not_unrolled)-1
                            ]
                            shape = [int(v) if isinstance(v,tvm.tir.IntImm) else -1 for v in var_.checked_type.shape]
                            dtype = var_.checked_type.dtype
                            var_dims=[MatchDim(name=a.name_hint+f"_dim_{idx}",size=shape[idx],is_dynamic=shape[idx]!=-1) for idx in range(len(shape))]
                            for dim in var_dims:
                                self.node_all_dims[dim.name]=dim
                            self.node_vars[a.name_hint] = MatchTensor(name=a.name_hint,
                                                                dims=var_dims,
                                                                dtype=np.dtype(dtype),tensor_type="var")
                        else:
                            old_idx = sum([c.op.name in k for k in var_and_consts_not_unrolled.keys()])
                            v_name=a.name_hint
                            var_and_consts_not_unrolled[v_name] = self.args_list[len(var_and_consts_not_unrolled)]
                            const_  = self.args_list[len(var_and_consts_not_unrolled)-1]
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
                        var_and_consts_not_unrolled[
                            a.name_hint
                        ] = a
                        shape = [int(v) if isinstance(v,tvm.tir.IntImm) else -1 for v in a.checked_type.shape]
                        dtype = a.checked_type.dtype
                        var_dims=[MatchDim(name=a.name_hint+f"_dim_{idx}",size=shape[idx],is_dynamic=shape[idx]!=-1) for idx in range(len(shape))]
                        for dim in var_dims:
                            self.node_all_dims[dim.name]=dim
                        self.node_vars[a.name_hint] = MatchTensor(name=a.name_hint,
                                                                dims=var_dims,
                                                                dtype=np.dtype(dtype),tensor_type="var")
                elif isinstance(a, tvm.relay.Constant):
                    old_idx = sum([c.op.name in k for k in var_and_consts_not_unrolled.keys()])
                    v_name=c.op.name+ f"_arg_{idx_arg}"
                    var_and_consts_unrolled[v_name] = a
                    shape = [int(v) if isinstance(v,tvm.tir.IntImm) else -1 for v in a.checked_type.shape]
                    dtype = a.checked_type.dtype
                    const_dims=[MatchDim(name=v_name+f"_dim_{idx}",size=shape[idx],is_dynamic=shape[idx]!=-1) for idx in range(len(shape))]
                    for dim in const_dims:
                        self.node_all_dims[dim.name]=dim
                    self.node_consts[a.name_hint] = MatchTensor(name=v_name,
                                                            dims=const_dims,
                                                            dtype=np.dtype(dtype),tensor_type="const",
                                                            data = a.data.numpy())
            self.index += 1
            #self.ops_names.append(c.op.name)
            unique_name=self.get_unique_name_and_update_occs(c)
            self.calls_to_name[c] = unique_name
            self.visit_call(c,unique_name)
            self.name_to_calls[unique_name] = c
            self.callsindexes[c] = self.index
        
        # save vars
        self.match_node.var_tensors = self.node_vars
        self.match_node.const_tensors = self.node_consts
        self.match_node.output_tensors = {t_name:t_value for t_name,t_value in self.calls_tensors.items() if t_value.tensor_type=="output"}
        self.match_node.intermediate_tensors = {t_name:t_value for t_name,t_value in self.calls_tensors.items() if t_value.tensor_type=="intermediate"}
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
        
        # breakpoint()
        self.match_node.dims = self.node_all_dims

    def visit_call(self, call, unique_name):
        if call.op.name not in self.visit_router:
            # skip if not supported for testing purposes
            print("[PARSER] The operator",call.op.name,"is not supported yet")
            raise NotImplementedError(
                f"Currently the operator {call.op.name} is not supported."
            )
        else:
            self.visit_router[call.op.name](call, call.attrs, unique_name)