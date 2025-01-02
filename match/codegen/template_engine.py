from typing import Any, Dict, List
from match.dim.dim import MatchDim, MatchTiledDim
from match.node.node import MatchNode
from match.tensor.tensor import MatchTensor,MatchTensorTile
from match.ops.bias_add import MatchOpBiasAdd
from match.ops.conv2d import MatchOpConv2D
from match.ops.relu import MatchOpReLU
from match.schedule.mem_transfer import MatchMemTransfer
from match.schedule.loop import MatchLoop
import tvm
from mako.template import Template
from mako import exceptions
import pathlib
import os
from match.target.exec_module import ExecModule

class TemplateEngine:
    def __init__(self,mod:tvm.ir.IRModule,exec_module:ExecModule=None,pattern_name:str="",template_data:Dict[str,Any]={},template_engine:str="mako"):
        self.mod=mod
        self.exec_module=exec_module
        self.pattern_name=pattern_name
        self.template_data=template_data
        self.template_engine=template_engine
        self.debug_template=False

    def get_code(self) -> str:
        # layer file
        temp = Template(filename=os.path.dirname(__file__)+"/lib/match_template.c")
        error_codegen=False
        code=""
        try:
            code=temp.render(**self.template_data)
        except:
            code=exceptions.html_error_template().render()
            error_codegen=True
        if self.debug_template:
            with open(f"./output/last_out.{'html' if error_codegen else 'c'}",
                "wb" if error_codegen else "w") as fw:
                fw.write(code)
        return code,error_codegen

def pattern_test():
    # layer file
    # template_filename = os.path.dirname(__file__)+"/lib/pattern_template.c"
    template_filename = os.environ["MATCH_PATH"]+"/match/codegen/template/lib/pattern_template.c"
    output_filename= os.environ["MATCH_PATH"]+"/match/codegen/template/tmp_output"
    temp = Template(filename=template_filename)
    error_codegen=False
    code=""
    independent_dims = [
        MatchDim(name="channels"),
        MatchDim(name="height"),
        MatchDim(name="width"),
    ]
    tensors = [
        # just for testing 4 inputs and one out
        MatchTensor(name="inp_K",dims=independent_dims,tensor_type="var",idx=0),
        MatchTensor(name="inp_Q",dims=independent_dims,tensor_type="var",idx=1),
        MatchTensor(name="inp_V",dims=independent_dims,tensor_type="var",idx=2),
        MatchTensor(name="inp_W",dims=independent_dims,tensor_type="var",idx=3),
        # out
        MatchTensor(name="logits",dims=independent_dims,tensor_type="output",idx=0),
        MatchTensor(name="present",dims=independent_dims,tensor_type="output",idx=1),
    ]
    for_loops = [
        MatchLoop(dim=independent_dims[0],name="ch_outer",size=4,loads=[MatchMemTransfer(tensor=tensors[0])]),
        MatchLoop(dim=independent_dims[0],name="w_outer",size=4,loads=[MatchMemTransfer(tensor=tensors[1]),MatchMemTransfer(tensor=tensors[2]),MatchMemTransfer(tensor=tensors[3])]),
        MatchLoop(dim=independent_dims[0],name="height",size=32,loads=[MatchMemTransfer(tensor=tensors[4]),MatchMemTransfer(tensor=tensors[5])]),
        MatchLoop(dim=independent_dims[0],name="channel",size=8,loads=[]),
        MatchLoop(dim=independent_dims[0],name="width",size=8,loads=[]),
    ]
    dependent_dims = {
        MatchDim(name="inp_height"):{
            independent_dims[1]:2,
        }
    }
    template_data = {
        "pattern_family":"conv2d",
        "specific_pattern":"conv2d_biasadd_req",
        "ctx_extension":"void",
        "tensors":tensors,
        "const_tensors":[tensor for tensor in tensors if tensor.is_constant],
        "var_tensors":[tensor for tensor in tensors if tensor.is_var],
        "output_tensors":[tensor for tensor in tensors if tensor.is_output],
        "include_list":["stdio.h","stdlib.h"],
        "independent_dims":independent_dims,
        "for_loops":for_loops,
        "dims_dependencies":dependent_dims,
        "latency":0,
        "energy":0,
        "name":"testing_llm_longlivematch",
    }
    try:
        code=temp.render(**template_data)
    except:
        code=exceptions.html_error_template().render()
        error_codegen=True
    
    output_extension = ".html" if error_codegen else ".c"
    with open(output_filename+output_extension,
        "wb" if error_codegen else "w") as fw:
        fw.write(code)


def pattern_params_test():
    template_filename = os.environ["MATCH_PATH"]+"/match/codegen/template/lib/pattern_params"
    output_filename= os.environ["MATCH_PATH"]+"/match/codegen/template/tmp_params_output"
    error_codegen=False
    code=""
    independent_dims = [
        MatchDim(name="channels",size=32),
        MatchDim(name="height",size=32),
        MatchDim(name="width",size=32),
    ]
    default_tile = MatchTensorTile([MatchTiledDim(dim=dim,size=dim.size) for dim in independent_dims])
    vars = [
        # just for testing 4 inputs
        MatchTensor(name="inp_K",dims=independent_dims,tensor_type="var",tiles=[default_tile,default_tile,default_tile]),
        MatchTensor(name="inp_Q",dims=independent_dims,tensor_type="var",tiles=[default_tile,default_tile,default_tile]),
        MatchTensor(name="inp_V",dims=independent_dims,tensor_type="var",tiles=[default_tile,default_tile,default_tile]),
        MatchTensor(name="inp_W",dims=independent_dims,tensor_type="var",tiles=[default_tile,default_tile,default_tile]),
    ]
    # out
    outs = [
        MatchTensor(name="logits",dims=independent_dims,tensor_type="output",tiles=[default_tile,default_tile,default_tile]),
        MatchTensor(name="present",dims=independent_dims,tensor_type="output",tiles=[default_tile,default_tile,default_tile]),
    ]
    
    ops = [
        MatchOpConv2D(name="conv2d",attrs={"padding":[1,1,1,1],"strides":[1,1],"dilation":[1,1]}),
        MatchOpBiasAdd(name="bias_add",attrs={"axis":-1}),
        MatchOpReLU(name="relu",attrs={}),
    ]
    consts = []
    template_data = {
        "pattern_family":"conv2d",
        "pattern_name":"conv2d_biasadd_req",
        "ctx_extension":"void",
        "tensors":vars+consts+outs,
        "vars":vars,
        "consts":consts,
        "outputs":outs,
        "ops":ops,
        "include_list":["stdio.h","stdlib.h"],
        "dims":independent_dims,
        "latency":0,
        "energy":0,
        "name":"testing_llm_longlivematch",
    }

    # C file
    temp = Template(filename=template_filename+".c")
    try:
        code=temp.render(**template_data)
    except:
        code=exceptions.html_error_template().render()
        error_codegen=True
    
    output_extension = "_c_.html" if error_codegen else ".c"
    with open(output_filename+output_extension,
        "wb" if error_codegen else "w") as fw:
        fw.write(code)

    # H file
    temp = Template(filename=template_filename+".h")
    try:
        code=temp.render(**template_data)
    except:
        code=exceptions.html_error_template().render()
        error_codegen=True
    
    output_extension = "_h_.html" if error_codegen else ".h"
    with open(output_filename+output_extension,
        "wb" if error_codegen else "w") as fw:
        fw.write(code)

if __name__=="__main__":
    pattern_params_test()