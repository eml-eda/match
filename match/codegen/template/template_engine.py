from typing import Any, Dict
import tvm
from mako.template import Template
from mako import exceptions

class TemplateEngine:
    def __init__(self,mod:tvm.ir.IRModule,hwmodel:Dict[str,Any]="",pattern_name:str="",template_data:Dict[str,Any]={},template_engine:str="mako"):
        self.mod=mod
        self.hwmodel=hwmodel
        self.pattern_name=pattern_name
        self.template_data=template_data
        self.template_engine=template_engine
    
    def get_code(self) -> str:
        # layer file
        temp = Template(filename="./lib/match_template.c")
        return temp.render(**self.temp_data), False, self.latency, self.cme