    
from typing import List
from match.dim.dim import MatchDim
from match.node.node import MatchNode
from match.ops.conv1d import MatchOpConv1D
from match.ops.conv2d import MatchOpConv2D
from match.ops.dense import MatchOpDense
from match.tensor.tensor import MatchTensor


class MatchNodeToZigZagParser:
    def __init__(self, match_node: MatchNode=None, pattern_name: str="conv2d"):
        self.match_node = match_node
        self.pattern_name = pattern_name
        # prec
        self.o_intermediate_prec = 32
        self.o_prec = 8
        self.first_inp_prec = 8
        self.second_inp_prec = 8
        # attrs
        self.strides = (1,1)
        self.dilations = (1,1)
        self.padding = (0,0,0,0)
        self.kernel_size = (1,1)

        self.loop_dim_size = {
            "B":1,
            "C":1,
            "K":1,
            "OY":1,
            "OX":1,
            "FY":1,
            "FX":1,
        }

        self.num_vars = len([t for t in self.match_node.tensors_arr if t.tensor_type == "var"])
        self.num_outs = len([t for t in self.match_node.tensors_arr if t.tensor_type=="output"])
        self.num_consts = len([t for t in self.match_node.tensors_arr if t.tensor_type == "const"])
        self.vars = [t for t in self.match_node.tensors_arr if t.tensor_type == "var"]
        self.outs = [t for t in self.match_node.tensors_arr if t.tensor_type=="output"]
        self.consts = [t for t in self.match_node.tensors_arr if t.tensor_type == "const"]
        # tensors
        self.i_tensor = self.vars[0]
        self.x_tensor = self.i_tensor
        self.y_tensor = None if self.num_vars<2 else self.vars[1]
        self.o_tensor = self.outs[0]
        self.w_tensor = None if self.num_vars>1 or self.num_consts==0 else self.consts[0]

        self.pr_loop_dim_size = {"IY":1, "IX":1} if self.w_tensor is not None else {}
        self.operand_source = {"W": [], "I": []} if self.w_tensor is not None else {"X":[], "Y":[]}
        self.constant_operands = ["W"] if self.w_tensor is not None else []
        self.operand_source_dimension_mapping = dict()
        if self.num_vars>1:
            if self.x_tensor is not None:
                self.first_inp_prec = self.x_tensor.bits
                self.operand_source_dimension_mapping["X"] = {"IX": "OX", "IY": "OY", "C": "K"}
            if self.y_tensor is not None:
                self.second_inp_prec = self.y_tensor.bits
                self.operand_source_dimension_mapping["Y"] = {"IX": "OX", "IY": "OY", "C": "K"}
        else:
            if self.i_tensor is not None:
                self.first_inp_prec = self.i_tensor.bits
                self.operand_source_dimension_mapping["I"] = {"IX": "OX", "IY": "OY"}
            if self.w_tensor is not None:
                self.second_inp_prec = self.w_tensor.bits
        self.o_prec = self.o_tensor.bits
        self.operand_precision = {
            "O": self.o_intermediate_prec,
            "O_final": self.o_prec,
            "I" if self.w_tensor is not None else "X": self.first_inp_prec,
            "W" if self.w_tensor is not None else "Y": self.second_inp_prec,
        }
        # let this dimensions at the end, dont tile them
        self.spatially_unrolled_dimensions = ["FX","FY","C"]

    def get_dim_name_by_name(self,name):
        def get_io_from_layout(dims: List[MatchDim]=[], layout: str="NCHW", tensor: MatchTensor=None, key: str="K"):
            # conv2d and other 4 dims operators
            if len(dims)==5:
                if layout=="NCHWc16":
                    n = dims[0]
                    c = dims[1]
                    h = dims[2]
                    w = dims[3]
                    c16 = dims[4]
                if tensor.tensor_type=="const":
                    if key=="C":
                        return c
                    if key=="K":
                        return n
                    if key=="FY":
                        return h
                    if key=="FX":
                        return w
                if tensor.tensor_type=="output":
                    if key=="B":
                        return n
                    if key=="K":
                        return c
                    if key=="OY":
                        return h
                    if key=="OX":
                        return w
                if tensor.tensor_type=="var":
                    if key=="B":
                        return n
                    if key=="C":
                        return c
                    if key=="IY":
                        return h
                    if key=="IX":
                        return w
            if len(dims)==4:
                if layout=="NHWC":
                    # layout is nhwc
                    n = dims[0]
                    c = dims[3]
                    h = dims[1]
                    w = dims[2]
                elif layout=="HWIO":
                    n = dims[3]
                    c = dims[2]
                    h = dims[0]
                    w = dims[1]
                elif layout=="NCHW" or layout=="OIHW":
                    n = dims[0]
                    c = dims[1]
                    h = dims[2]
                    w = dims[3]
                elif layout=="OIHW":
                    n = dims[0]
                    c = dims[1]
                    h = dims[2]
                    w = dims[3]
                elif layout=="OHWI":
                    n = dims[0]
                    c = dims[3]
                    h = dims[1]
                    w = dims[2]
                else:
                    #layout is nchw
                    n = dims[0]
                    c = dims[1]
                    h = dims[2]
                    w = dims[3]
                if tensor.tensor_type=="const":
                    if key=="C":
                        return c
                    if key=="K":
                        return n
                    if key=="FY":
                        return h
                    if key=="FX":
                        return w
                if tensor.tensor_type=="output":
                    if key=="B":
                        return n
                    if key=="K":
                        return c
                    if key=="OY":
                        return h
                    if key=="OX":
                        return w
                if tensor.tensor_type=="var":
                    if key=="B":
                        return n
                    if key=="C":
                        return c
                    if key=="IY":
                        return h
                    if key=="IX":
                        return w
            # conv1d and other 3 dims operators
            elif len(dims)==3:
                n = dims[0]
                c = dims[1]
                spat = dims[2]
                if tensor.tensor_type=="const":
                    if key=="C":
                        return c
                    if key=="K":
                        return n
                    if key=="H":
                        return spat
                if tensor.tensor_type=="output":
                    if key=="B":
                        return n
                    if key=="K":
                        return c
                    if key=="OY":
                        return spat
                if tensor.tensor_type=="var":
                    if key=="B":
                        return n
                    if key=="C":
                        return c
                    if key=="IY":
                        return spat
            elif len(dims)==2:
                if key=="N":
                    return dims[0]
                if key in ["C","K"]:
                    return dims[1]
            elif len(dims)==1:
                if key=="N":
                    return dims[0]
            return self.match_node.default_dim
        
        tensor = self.o_tensor
        if name=="C":
            tensor = self.i_tensor
        if name=="FY":
            tensor = self.w_tensor
        if name=="FX":
            tensor = self.w_tensor
        if name=="IY":
            tensor = self.i_tensor
        if name=="IX":
            tensor = self.i_tensor

        if tensor is None:
            return self.match_node.default_dim
        
        found_dim = get_io_from_layout(dims=tensor.dims, layout=tensor.layout, tensor=tensor, key=name)
        if found_dim==self.match_node.default_dim:
            error_dim = False
            if name in ["IX","FX","OX"] and len(tensor.dims)>=4:
                error_dim = True
            elif name in ["IY","FY","OY"] and len(tensor.dims)>=3:
                error_dim = True
            elif name in ["C","K"] and len(tensor.dims)>=2:
                error_dim = True
            elif name in ["N"] and len(tensor.dims)>=1:
                error_dim = True
            
            if error_dim:
                print(f"[ZIGZAG ENGINE] Error during dimension parsing, trying to get {name} from dims {[dim.name for dim in tensor.dims]} in tensor {tensor.name}")
        return found_dim
    
    def get_operands(self):
        return ["I","W","O"] if self.w_tensor is not None else ["X","Y","O"]

    def get_spatially_unrolled_dimensions(self):
        return self.spatially_unrolled_dimensions
    
    def generate_workload(self): 
        return {
            1: {
                "match_node": self.match_node,
                "operator_type": self.pattern_name,
                "equation": self.equation,
                "dimension_relations": [
                    f"ix={self.strides[1]}*ox+{self.dilations[1]}*fx",
                    f"iy={self.strides[0]}*oy+{self.dilations[0]}*fy",
                ],
                "loop_dim_size": self.loop_dim_size,
                "operand_precision": self.operand_precision,
                "pr_loop_dim_size": self.pr_loop_dim_size,
                "padding": {"IY":(self.padding[0],self.padding[2]),"IX":(self.padding[1],self.padding[3])},
                "strides": self.strides,
                "operand_source": self.operand_source,
                "constant_operands": self.constant_operands,
                "operand_source_dimension_mapping": self.operand_source_dimension_mapping,
            }
        }
    
    def parse(self):
        # TODO: currently its a sort of priority queue of operations, should be done better
        if "conv2d" in self.match_node.ops_occurrences:
            self.visit_conv2d()
        elif "conv1d" in self.match_node.ops_occurrences:
            self.visit_conv1d()
        elif "dense" in self.match_node.ops_occurrences:
            self.visit_dense()
        elif "maxpool2d" in self.match_node.ops_occurrences:
            self.visit_maxpool2d()
        elif "add" in self.match_node.ops_occurrences:
            self.visit_add()
        else:
            print("[ZIGZAG PARSER] Warning, no operator found to tile, continuing with default workload")

    def visit_maxpool2d(self):
        pass

    def visit_conv1d(self):
        # get data
        conv1d_node: MatchOpConv1D = self.match_node.ops["conv1d"]
        self.strides = conv1d_node.strides + (1,)
        self.dilations = conv1d_node.dilation + (1,)
        self.padding = conv1d_node.padding
        self.kernel_size = conv1d_node.kernel_size + (1,)
        # as if it was height only
        self.padding = (self.padding[0], 0, self.padding[1], 0)
        # get sizes
        o_n, o_c, o_h = [self.get_dim_name_by_name(key).size for key in ["B","K","OY"]]
        i_h = self.get_dim_name_by_name("IY").size
        w_cin = self.get_dim_name_by_name("C").size
        
        self.equation = "O[b][k][oy][ox]+=W[k][c][fy]*I[b][c][iy]"
        self.loop_dim_size["B"] = o_n
        self.loop_dim_size["K"] = o_c
        self.loop_dim_size["C"] = w_cin
        self.loop_dim_size["OY"] = o_h
        self.loop_dim_size["FY"] = self.kernel_size[0]
        # dependencies dims
        self.pr_loop_dim_size["IY"] = i_h
        if conv1d_node.depthwise:
            self.operand_source_dimension_mapping["I"]["C"]="K"
            self.equation = "O[b][k][oy][ox]+=W[k][c][fy]*I[b][k][iy]"
        
    def visit_conv2d(self):
        # get conv attrs
        conv2d_node: MatchOpConv2D = self.match_node.ops["conv2d"]
        self.strides = conv2d_node.strides
        self.dilations = conv2d_node.dilation
        self.padding = conv2d_node.padding
        self.kernel_size = conv2d_node.kernel_size
        # get sizes
        o_n, o_c, o_h, o_w = [self.get_dim_name_by_name(key).size for key in ["B","K","OY","OX"]]
        i_h,i_w = [self.get_dim_name_by_name(key).size for key in ["IY","IX"]]
        w_cin = self.get_dim_name_by_name("C").size
        
        self.equation = "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]"
        self.loop_dim_size["B"] = o_n
        self.loop_dim_size["K"] = o_c
        self.loop_dim_size["C"] = w_cin
        self.loop_dim_size["OY"] = o_h
        self.loop_dim_size["OX"] = o_w
        self.loop_dim_size["FY"] = self.kernel_size[0]
        self.loop_dim_size["FX"] = self.kernel_size[1]
        # dependencies dims
        self.pr_loop_dim_size["IY"] = i_h
        self.pr_loop_dim_size["IX"] = i_w
        if conv2d_node.depthwise:
            self.operand_source_dimension_mapping["I"]["C"]="K"
            self.equation = "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][k][iy][ix]"
    
    def visit_dense(self):
        dense_node: MatchOpDense = self.match_node.ops["dense"]

        self.loop_dim_size["C"] = dense_node.inp_features
        self.loop_dim_size["K"] = dense_node.out_features
        self.equation = "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][k][iy][ix]"
        

    def visit_add(self):
        o_n, o_c, o_h, o_w = [self.get_dim_name_by_name(key).size for key in ["B","K","OY","OX"]]
        
        self.equation = "O[b][k][oy][ox]+=X[b][k][oy][ox]*Y[b][k][oy][ox]"
        self.loop_dim_size["B"] = o_n
        self.loop_dim_size["K"] = o_c
        self.loop_dim_size["OY"] = o_h
        self.loop_dim_size["OX"] = o_w
        self.spatially_unrolled_dimensions = list()