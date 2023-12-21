from typing import Any, Dict
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
        with open(f"./outputs/last_out.{'html' if error_codegen else 'c'}",
            "wb" if error_codegen else "w") as fw:
            fw.write(code)
        return code,error_codegen