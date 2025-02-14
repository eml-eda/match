

import json
from math import prod
import os
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

import mako
import numpy as np
from numpy import typing as npt
from match.compile.c_graph import MatchCompilerCGraph
from match.model.dynamic_dim import DynamicDim
from match.utils import save_all_relay,add_save_relay,reset_relay_list,reset_output_path,set_output_path,reset_schedules,save_all_schedules
from match.compile.c_aot import MatchCompilerCAoT
from mako.template import Template

from match.utils.utils import c_friendly_npvalue, get_random_np_array, numpy_dtype_to_c_type
import tvm
from tvm import relay

class MatchMemoryTensor:
    def __init__(self,name:str="p1",is_intermediate:bool=False,is_constant:bool=False,
                 is_output:bool=False,is_input:bool=False,
                 constant_val:npt.ArrayLike=np.array([1]),shape:Tuple[int]=(1,),
                 dtype:npt.DTypeLike=np.dtype("uint8"),node_id:int=0):
        self.name=name
        self.is_intermediate=is_intermediate
        self.is_constant=is_constant
        self.is_output=is_output
        self.is_input=is_input
        if sum([self.is_intermediate,self.is_constant,self.is_output,self.is_input])!=1:
            raise Exception(f"Match Memory Tensor can only be one option between(intermediate,constant,output,input)")
        self.constant_val=constant_val
        self.shape=shape
        self.dtype=dtype
        self.node_id=node_id
        self.last_usage=node_id
        self.soc_memory_offset = -1
        self.stored_in_external_memory = False

    @property
    def lifetime(self):
        return (self.node_id,self.last_usage)
    
    @property
    def elems(self):
        return prod(self.shape)

    def update_last_usage(self,new_ending_idx):
        self.last_usage=new_ending_idx

class MatchMemoryPlanner:
    def __init__(self,mem_tensors:List[MatchMemoryTensor],available_soc_bytes:int=1024):
        self.mem_tensors = mem_tensors
        self.available_soc_bytes = available_soc_bytes
        self.last_idx = mem_tensors[-1]
        self.intermediate_memory_usage = [0] * (self.last_idx.last_usage + 1)
        self.overall_intermediate_memory_usage = [0] * (self.last_idx.last_usage + 1)
        self.input_memory_usage = 0
        self.constant_memory_usage = 0
        self.output_memory_usage = 0
        for tensor in self.mem_tensors:
            if not tensor.is_input and not tensor.is_output and not tensor.is_constant:
                for time in range(tensor.node_id, tensor.last_usage + 1):
                    self.intermediate_memory_usage[time] += tensor.elems * tensor.dtype.itemsize
                    self.overall_intermediate_memory_usage[time] += tensor.elems * tensor.dtype.itemsize
            elif tensor.is_constant:
                self.constant_memory_usage += tensor.elems * tensor.dtype.itemsize
                self.overall_intermediate_memory_usage[time] += tensor.elems * tensor.dtype.itemsize
            elif tensor.is_input:
                self.input_memory_usage += tensor.elems * tensor.dtype.itemsize
            elif tensor.is_output:
                self.output_memory_usage += tensor.elems * tensor.dtype.itemsize
        self.total_memory_needed_bytes = self.input_memory_usage + self.output_memory_usage + self.constant_memory_usage + max(self.intermediate_memory_usage)
        self.total_memory_needed_bytes_w_consts = self.input_memory_usage + self.output_memory_usage + max(self.overall_intermediate_memory_usage)
    
    @property
    def external_memory_needed(self):
        return self.total_memory_needed_bytes > self.available_soc_bytes
    
    def run_memory_planner(self):
        # use only SoC memory

        if not self.external_memory_needed:
            offsets_time = [0] * (self.last_idx.last_usage + 1)
            for tensor in self.mem_tensors:
                if not tensor.is_input and not tensor.is_output and not tensor.is_constant:
                    for time in range(tensor.node_id, tensor.last_usage + 1):
                        if time==tensor.node_id:
                            tensor.soc_memory_offset = offsets_time[time]
                        offsets_time[time] += tensor.elems * tensor.dtype.itemsize
        elif self.total_memory_needed_bytes_w_consts > self.available_soc_bytes:
            offsets_time = [0] * (self.last_idx.last_usage + 1)
            for tensor in self.mem_tensors:
                if not tensor.is_input and not tensor.is_output:
                    for time in range(tensor.node_id, tensor.last_usage + 1):
                        if time==tensor.node_id:
                            if tensor.is_constant:
                                tensor.stored_in_external_memory = True
                            tensor.soc_memory_offset = offsets_time[time]
                        offsets_time[time] += tensor.elems * tensor.dtype.itemsize
        else:
            raise Exception("Not enough SoC memory available")
    

    

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


    def compile_model(self,target,out_path,executor):
        reset_output_path()
        reset_relay_list()
        reset_schedules()
        model_path = self.get_path(out_path=out_path)
        set_output_path(model_path)
        #breakpoint()
        if executor=="graph":
            compiler = MatchCompilerCGraph(self.relay_mod, self.relay_params,
                            target=target,
                            build_dir=model_path,
                            mod_name=self.name,)
        else:
            compiler = MatchCompilerCAoT(self.relay_mod, self.relay_params,
                            target=target,
                            build_dir=model_path,
                            mod_name=self.name,)
        compiler.tvm_compile(fusion=True)
        #self.rename_lib(out_path=out_path)
        save_all_relay()
        save_all_schedules()

    def rename_lib(self,out_path):
        pass

    def remove_static_app(self):
        pass

    def generate_model_graph_runtime(self,target,mod_info,params):
        nodes = []
        tensor_map = {}
        mem_tensors = []
        dtypes = mod_info["attrs"]["dltype"][1]
        shapes = mod_info["attrs"]["shape"][1]
        for node_id,node in enumerate(mod_info["nodes"]):
            if node["op"]=="null":
                # input or parameter
                # param
                if node["name"] in params:
                    param = params[node["name"]]
                    # store this into the weights to store
                    mem_tensor = MatchMemoryTensor(name=node["name"],constant=True,
                                                   constant_val=param.data.numpy(),shape=param.shape,
                                                   dtype=np.dtype(param.dtype),node_id=node_id)
                    mem_tensors[node["name"]] = mem_tensor
                else:
                    mem_tensor = MatchMemoryTensor(name=node["name"],intermediate=False,constant=False,
                                                   shape=tuple(shapes[node_id]),dtype=np.dtype(dtypes[node_id]),
                                                   node_id=node_id)
                    mem_tensors[node["name"]] = mem_tensor
            else:
                inputs = [(inp_node_idx,mod_info["nodes"][inp_node_idx],mod_info["nodes"][inp_node_idx]["name"]+("" if mod_info["nodes"][inp_node_idx]["op"]=="null" else "_out"))
                                              for inp_node_idx in [inp_node_idxs[0] for inp_node_idxs in node["inputs"]]]
                for inp_idx,inp,inp_name in inputs:
                    mem_tensors[inp_name].update_last_usage(node_id)
                MatchGraphRuntimeNodeCall(inputs=inputs,name=node["name"])
                mem_tensor = MatchMemoryTensor(name=node["name"]+"_out",intermediate=True,constant=False,
                                                shape=tuple(shapes[node_id]),dtype=np.dtype(dtypes[node_id]),
                                                node_id=node_id)
        

    def move_static_app_to(self,target,out_path,executor:str="graph"):
        build_dir = self.get_path(out_path=out_path)
        abs_out_path = str(Path(out_path).absolute())
        if executor=="graph":
            subprocess.getoutput(f"mkdir {build_dir}/codegen")
            subprocess.getoutput(f"mkdir {build_dir}/codegen/host")
            subprocess.getoutput(f"mkdir {build_dir}/codegen/host/include")
            subprocess.getoutput(f"mkdir {build_dir}/codegen/host/src")
            subprocess.getoutput(f"tar -xvf {build_dir}/mod.tar -C {build_dir}/codegen/host/src")
            # read the json of the mod and the params to build the runtime
            mod_info = {}
            with open(f"{build_dir}/mod.json","r") as mod_file:
                mod_info = json.load(mod_file)
            if len(mod_info)==0:
                raise FileNotFoundError()
            params_bytes=bytes("","utf8")
            params = None
            with open(f"{build_dir}/mod.params","rb") as params_file:
                params_bytes=params_file.read()
            params=relay.load_param_dict(params_bytes)
            self.generate_model_graph_runtime(target,mod_info,params)
            subprocess.getoutput(f"rm {build_dir}/mod.tar")
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
        
        if self.name=="default":
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



def build_runtime(target=None,static_models:List[MatchModel]=[],dynamic_dims:Dict[str,DynamicDim]={},
                  match_inputs=None,match_outputs=None,runtime:str="default",out_path:str="/build",benchmarking:bool=True):
    abs_out_path = str(Path(out_path).absolute())
    models_ = {model.name:model for model in static_models}
    temp_args = {
        "generative_models":models_,
        "models":models_,
        "dynamic_dims":dynamic_dims,
        "outputs":match_outputs,
        "inputs":match_inputs,
        "benchmarking":benchmarking,
        "golden_cpu_model":"golden_cpu_model" in models_,
        "runtime":runtime,
        "target":target,
    }
    with open(abs_out_path+"/src/match/runtime.c","w") as run_file:
        run_file.write(Template(filename=os.path.dirname(__file__)+"/../libs/c/mako/match/src/runtime.c").render(**temp_args))
    with open(abs_out_path+"/include/match/runtime.h","w") as run_file:
        run_file.write(Template(filename=os.path.dirname(__file__)+"/../libs/c/mako/match/include/runtime.h").render(**temp_args))

def get_match_inputs_and_outputs(static_model:MatchModel=None):
    mod_checked_types = tvm.relay.transform.InferType()(static_model.relay_mod)
    func=mod_checked_types["main"]
    relay_inputs=func.params
    if isinstance(func.checked_type.ret_type,tvm.relay.TupleType):
        relay_out_types = [ret_type for ret_type in func.checked_type.ret_type.fields]
    else:
        relay_out_types=[func.checked_type.ret_type]
    np.random.seed(0)
    match_inputs={inp_.name_hint:{
        "name":inp_.name_hint,
        "c_arr_size":int(prod(inp_.type_annotation.shape)),
        "c_type":numpy_dtype_to_c_type(inp_.type_annotation.dtype),
        "prod_shape":int(prod(inp_.type_annotation.shape)),
        "shape":[int(sh) for sh in inp_.type_annotation.shape],
        "dims":[int(sh) for sh in inp_.type_annotation.shape],
        "c_arr_values":c_friendly_npvalue(get_random_np_array(dtype=inp_.type_annotation.dtype,shape=tuple([int(v) for v in inp_.type_annotation.shape]))),
    } for inp_ in relay_inputs}
    match_outputs = {f"output{idx if len(relay_out_types)>1 else ''}":{
        "name":f"output{idx if len(relay_out_types)>1 else ''}",
        "c_arr_size":int(prod(out.shape)),
        "c_type":numpy_dtype_to_c_type(out.dtype),
        "prod_shape":int(prod(out.shape)),
        "shape":[int(sh) for sh in out.shape],
        "dims":[int(sh) for sh in out.shape],
    } for idx,out in enumerate(relay_out_types)}
    return match_inputs,match_outputs