from typing import Any, Dict, List
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

class Dim:
    def __init__(self, name: str="width") -> None:
        self.name = name

class Operand:
    def __init__(self,is_var: bool=True, is_constant: bool=False, is_output: bool=False, name: str="inp_A", idx: int=0, dims: List[Dim]=[]) -> None:
        self.is_var = is_var
        self.is_constant = is_constant
        self.is_output = is_output
        self.name = name
        self.idx = idx
        self.dims = dims

class Load:
    def __init__(self, operand: Operand=Operand()) -> None:
        self.operand = operand


class ForLoop:
    def __init__(self, dim: Dim, size: int=0, name: str="width", loads: List[Load]=[]) -> None:
        self.name = ""
        self.size = size
        self.name = name
        self.dim = dim
        self.loads = loads


if __name__=="__main__":
    # layer file
    # template_filename = os.path.dirname(__file__)+"/lib/pattern_template.c"
    template_filename = os.environ("MATCH_PATH")+"/match/codegen/template/lib/pattern_template.c"
    output_filename= os.environ("MATCH_PATH")+"/match/codegen/template/tmp_output"
    temp = Template(filename=template_filename)
    error_codegen=False
    code=""
    independent_dims = [
        Dim(name="channels"),
        Dim(name="height"),
        Dim(name="width"),
    ]
    operands = [
        # just for testing 4 inputs and one out
        Operand(is_var=True,name="inp_K",dims=independent_dims,idx=0),
        Operand(is_var=True,name="inp_Q",dims=independent_dims,idx=1),
        Operand(is_var=True,name="inp_V",dims=independent_dims,idx=2),
        Operand(is_var=True,name="inp_W",dims=independent_dims,idx=3),
        # out
        Operand(is_output=True,name="out",dims=independent_dims,idx=0),
    ]
    for_loops = [
        ForLoop(dim=independent_dims[0],name="ch_outer",size=4,loads=[Load(operand=operands[0])]),
        ForLoop(dim=independent_dims[0],name="w_outer",size=4,loads=[Load(operand=operands[1]),Load(operand=operands[2]),Load(operand=operands[3])]),
        ForLoop(dim=independent_dims[0],name="height",size=32,loads=[Load(operand=operands[4])]),
        ForLoop(dim=independent_dims[0],name="channel",size=8,loads=[]),
        ForLoop(dim=independent_dims[0],name="width",size=8,loads=[]),
    ]
    dependent_dims = {
        Dim(name="inp_height"):{
            independent_dims[1]:2,
        }
    }
    template_data = {
        "pattern_family":"conv2d",
        "specific_pattern":"conv2d_biasadd_req",
        "ctx_extension":"void",
        "operands":operands,
        "include_list":["stdio.h","stdlib.h"],
        "independent_dims":independent_dims,
        "for_loops":for_loops,
        "dims_dependencies":dependent_dims,
        "latency":0,
        "energy":0,
        "func_name":"testing_llm_longlivematch",
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