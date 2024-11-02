

import os
from pathlib import Path
import subprocess
from typing import Dict
from match.utils import save_all_relay,add_save_relay,reset_relay_list,reset_output_path,set_output_path,reset_schedules,save_all_schedules
from match.driver.driver import driver

class MatchModel:

    def __init__(self,relay_mod,relay_params,dynamic:bool=False,dyn_is_max:bool=False,dynamic_sizes:Dict={}):
        self.relay_mod = relay_mod
        self.relay_params = relay_params
        self.dynamic = dynamic
        self.dynamic_sizes = dynamic_sizes
        self.dyn_is_max = dyn_is_max
        self.name = "default" if (not self.dynamic or self.dyn_is_max) else "_".join(f"{dyn_name.replace(' ', '_')}_{dyn_val}" for dyn_name, dyn_val in dynamic_sizes.items())
    

    def get_path(self,out_path):
        return str(Path(out_path).absolute())+"/"+self.name+"_buildtmp"


    def compile_model(self,target,out_path):
        reset_output_path()
        reset_relay_list()
        reset_schedules()
        model_path = self.get_path(out_path=out_path)
        set_output_path(model_path)
        #breakpoint()
        driver(self.relay_mod, self.relay_params, target=target,output_path=model_path,mod_name=self.name)
        #self.rename_lib(out_path=out_path)
        save_all_relay()
        save_all_schedules()

    def rename_lib(self,out_path):
        pass

    def remove_static_app(self):
        pass

    def move_static_app_to(self,out_path):
        build_dir = self.get_path(out_path=out_path)
        abs_out_path = str(Path(out_path).absolute())
        
        # create codegen if it doesn't exist
        if not Path(abs_out_path+"/codegen").is_dir():
            subprocess.getoutput(f"mkdir {abs_out_path}/codegen")
        if not Path(abs_out_path+"/models").is_dir():
            subprocess.getoutput(f"mkdir {abs_out_path}/models")
        
        def create_mod_dir_and_mv(rm_dirlist):
            subprocess.getoutput(f"mkdir {abs_out_path}/models/{self.name}")
            for ext_type in ["relay","log"]:
                subprocess.getoutput(f"mkdir {abs_out_path}/models/{self.name}/match_{ext_type}")
                subprocess.getoutput(f"cp {build_dir}/*.{ext_type} {abs_out_path}/models/{self.name}/match_{ext_type}")
                subprocess.getoutput(f"rm {build_dir}/*.{ext_type}")
            # parameters
            subprocess.getoutput(f"mv {build_dir}/parameters {abs_out_path}/models/{self.name}/parameters")
            # codegen
            subprocess.getoutput(f"mv {build_dir}/codegen/host {abs_out_path}/codegen/{self.name}")
            # rm stuff now
            for rm_dir in rm_dirlist:
                subprocess.getoutput(f"rm {build_dir}/{rm_dir} -r")
            subprocess.getoutput(f"mkdir {abs_out_path}/models/{self.name}/other_metadata")
            subprocess.getoutput(f"cp {build_dir}/* {abs_out_path}/models/{self.name}/other_metadata")
            subprocess.getoutput(f"rm {build_dir} -r")

        def move_final_relay():
            subprocess.getoutput(f"mv {build_dir}/src/{self.name}.relay {build_dir}/{self.name}.relay")
        
        if (not self.dynamic) or (self.dyn_is_max):
            # move app
            move_final_relay()
            # include src runtime
            for static_dir in ["include","src","runtime"]:
                subprocess.getoutput(f"mv {build_dir}/{static_dir} {abs_out_path}/{static_dir}")
            create_mod_dir_and_mv(rm_dirlist=["codegen","templates"])
        else:
            # remove all the static stuff and move the codegen in the overall build folder
            move_final_relay()
            create_mod_dir_and_mv(rm_dirlist=["codegen","templates","include","src","runtime"])
