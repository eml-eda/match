

import os
from match.node.node import MatchNode
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ComputationalApis, ExecModule, MemoryApis, PlatformApis, SyncApis
from mako.template import Template
from mako import exceptions
from pathlib import Path
from match.target.target import MatchTarget
from match.utils.utils import get_output_path,numpy_dtype_to_c_type,c_friendly_npvalue
import tvm

class TemplateWriter:
    def __init__(self,mod: tvm.ir.IRModule,target: MatchTarget,exec_module: ExecModule,
                 pattern_name:str="fused_quant_conv2d",schedule: MatchSchedule=None,match_node: MatchNode=None,
                 latency: float=0.0,energy: float=0.0):
        self.mod=mod
        self.target=target
        self.exec_module=exec_module
        self.pattern_name=pattern_name
        self.schedule=schedule
        self.match_node=match_node
        self.latency=latency
        self.energy=energy
        self.template_data=dict()
    
    def get_template_data(self):
        self.template_data["c_dtype"]=numpy_dtype_to_c_type
        self.template_data["c_np_array"]=c_friendly_npvalue
        #self.template_data["pattern_name"]=self.exec_module.specific_pattern
        self.template_data["pattern_name"]=self.pattern_name
        self.template_data["c_ident"]=lambda x: "\t"*x
        self.template_data["pattern_family"]=self.pattern_name
        self.template_data["latency"]=self.latency
        self.template_data["energy"]=self.energy
        self.template_data["exec_module"]=self.exec_module
        self.template_data["target"]=self.target
        self.template_data["schedule"]=self.schedule
        self.template_data["mem_levels"]=[]
        self.template_data["match_node"]=self.match_node
        self.template_data["mod"] = self.mod
        self.template_data["model_name"] = self.mod.attrs.global_symbol.split("_")[1]
        self.template_data["node_name"] = "_".join(self.mod.attrs.global_symbol.split("_")[3:])
        self.template_data["name"] = self.template_data["node_name"]
        self.template_data["fullname"] = self.mod.attrs.global_symbol
        self.template_data["node_fullname"] = self.template_data["fullname"]
        self.template_data["node_idx"] = int(self.template_data["fullname"].split("_")[::-1][0])
        self.template_data["async_computation"] = False
        self.template_data["top_level_memory_vars"] = "L2_CACHE"
        self.template_data["top_level_memory_out"] = "L2_CACHE"
        self.template_data["mem_apis"] = MemoryApis()
        self.template_data["sync_apis"] = SyncApis()
        self.template_data["platform_apis"] = PlatformApis()
        self.template_data["comp_apis"] = ComputationalApis()

    def write_layer_files(self):
        # write layer files
        out_path = get_output_path()
        node_path = "nodes/"+self.template_data["model_name"]
        Path(out_path).mkdir(parents=True,exist_ok=True)
        Path(out_path+"/src").mkdir(parents=True,exist_ok=True)
        Path(out_path+"/src/nodes").mkdir(parents=True,exist_ok=True)
        Path(out_path+"/src/"+node_path).mkdir(parents=True,exist_ok=True)
        Path(out_path+"/include").mkdir(parents=True,exist_ok=True)
        Path(out_path+"/include/nodes").mkdir(parents=True,exist_ok=True)
        Path(out_path+"/include/"+node_path).mkdir(parents=True,exist_ok=True)
        node_code = "#include <stdio.h>\n"
        for base_dir in ["src", "include", "metadata"]:
            template_dir = os.path.dirname(__file__) + "/../libs/c/mako/node/" + base_dir
            node_base_path = "" if base_dir=="metadata" else base_dir+"/"+node_path
            for filename in os.listdir(template_dir):
                filename_without_dots = filename.split(".")
                filename_without_ext = ".".join(filename_without_dots[:-1])
                ext = filename_without_dots[-1]
                if ext in {"c","h","json"}:
                    try:
                        template = Template(filename = os.path.join(template_dir, filename))
                        rendered_content = template.render(**self.template_data)
                        if filename=="node.c":
                            node_code = rendered_content
                            continue
                        output_path = os.path.join(out_path, node_base_path, filename.replace("node", self.template_data["node_name"]))
                        with open(output_path, "w") as output_file:
                            output_file.write(rendered_content)
                    except:
                        print(f"[TEMPLATE WRITER] Error processing template: {filename}")
                        output_path = os.path.join(out_path, node_base_path, filename_without_ext.replace("node",self.template_data["node_name"])+".html")
                        with open(output_path, "wb") as output_file:
                            output_file.write(exceptions.html_error_template().render())
        return node_code

    def get_code(self) -> str:
        # layer file
        self.get_template_data()
        return self.write_layer_files()