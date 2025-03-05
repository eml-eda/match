from math import prod
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from numpy import typing as npt
import mako

from match.target.target import MatchTarget
from match.utils.utils import numpy_dtype_to_c_type


class MatchMemoryTensor:
    def __init__(self,name:str="p1",is_intermediate:bool=False,is_constant:bool=False,
                 is_output:bool=False,is_input:bool=False,
                 constant_val:npt.ArrayLike=np.array([1]),shape:Tuple[int]=(1,),
                 dtype:npt.DTypeLike=np.dtype("uint8"),node_id:int=0,
                 node_info={}):
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
        self.load_from_ext_mem_at = set()
        self.c_type = numpy_dtype_to_c_type(self.dtype)
        self.node_info = node_info

    @property
    def lifetime(self):
        return (self.node_id,self.last_usage)
    
    @property
    def elems(self):
        return prod(self.shape)

    def update_last_usage(self,new_ending_idx):
        self.last_usage=new_ending_idx

class MatchMemoryPlanner:
    def __init__(self, mem_tensors:List[MatchMemoryTensor], available_soc_bytes:int=1024, algorithm: str="match"):
        self.mem_tensors = mem_tensors
        self.available_soc_bytes = available_soc_bytes
        self.algorithm = algorithm
        self.last_timestep = max([tens.last_usage for tens in self.mem_tensors])
        self.intermediate_memory_usage = [0] * (self.last_timestep + 1)
        self.overall_intermediate_memory_usage = [0] * (self.last_timestep + 1)
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
                for time in range(tensor.node_id, tensor.last_usage + 1):
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
    
    def generate(self):
        # use only SoC memory
        breakpoint()
        if not self.external_memory_needed:
            offsets_time = [0] * (self.last_timestep + 1)
            for tensor in self.mem_tensors:
                if not tensor.is_input and not tensor.is_output and not tensor.is_constant:
                    for time in range(tensor.node_id, tensor.last_usage + 1):
                        if time==tensor.node_id:
                            tensor.soc_memory_offset = offsets_time[time]
                        offsets_time[time] += tensor.elems * tensor.dtype.itemsize
            return max(self.intermediate_memory_usage), 0
        elif self.total_memory_needed_bytes_w_consts <= self.available_soc_bytes:
        # elif True:
            offsets_time = [0] * (self.last_timestep + 1)
            ext_mem_needed = 0
            for tensor in self.mem_tensors:
                if not tensor.is_input and not tensor.is_output:
                    for time in range(tensor.node_id, tensor.last_usage + 1):
                        if time==tensor.node_id:
                            if tensor.is_constant:
                                tensor.stored_in_external_memory = True
                                tensor.load_from_ext_mem_at.add(tensor.node_id)
                                ext_mem_needed += tensor.elems * tensor.dtype.itemsize
                            tensor.soc_memory_offset = offsets_time[time]
                        offsets_time[time] += tensor.elems * tensor.dtype.itemsize
            return max(self.overall_intermediate_memory_usage), ext_mem_needed
        else:
            raise Exception("Not enough SoC memory available")
    
class MatchGraphRuntimeNodeCall:
    def __init__(self, inputs=None, outputs=None, fn_name="default_lib_1", name="default_lib_1", node_info={}, node_id: int=0):
        self.inputs = inputs
        self.outputs = outputs
        self.fn_name = fn_name
        self.name = name
        self.node_info = node_info
        self.fallback = "match" not in self.fn_name
        self.node_id = node_id


class MatchTVMGraphRuntime:
    def __init__(self, target: MatchTarget, mod_info: Dict[str,Any], params=None, model_name: str="default", out_path: str="model_out"):
        self.target = target
        self.mod_info = mod_info
        self.params = params
        self.model_name = model_name
        self.out_path = out_path
        self.mem_planner = None
        self.ext_mem_needed_bytes = 0
        self.soc_mem_needed_bytes = 0

    def generate(self):
        tensor_map = {}
        nodes_map = {}
        mem_tensors = []
        nodes = []
        dtypes = self.mod_info["attrs"]["dltype"][1]
        shapes = self.mod_info["attrs"]["shape"][1]
        heads = [head[0] for head in self.mod_info["heads"]]
        # breakpoint()
        for node_id,node in enumerate(self.mod_info["nodes"]):
            if node["op"]=="null":
                # input or parameter
                # param
                if node["name"] in self.params:
                    param = self.params[node["name"]]
                    # store this into the weights to store
                    mem_tensor = MatchMemoryTensor(name=node["name"],is_constant=True,
                                                   constant_val=param.numpy(),shape=param.shape,
                                                   dtype=np.dtype(param.dtype),node_id=node_id,
                                                   node_info=node)
                    mem_tensors.append(mem_tensor)
                    tensor_map[node["name"]] = mem_tensor
                else:
                    mem_tensor = MatchMemoryTensor(name=node["name"],is_input=True,
                                                   shape=tuple(shapes[node_id]),dtype=np.dtype(dtypes[node_id]),
                                                   node_id=node_id, node_info=node)
                    mem_tensors.append(mem_tensor)
                    tensor_map[node["name"]] = mem_tensor
            else:
                inputs = [mem_tensors[inp_node_idx] for inp_node_idx in [inp_node_idxs[0] for inp_node_idxs in node["inputs"]]]
                for inp in inputs:
                    inp.update_last_usage(node_id)
                id_out = -1
                tens_name = self.model_name+"_node_"+str(node_id)+"_out"
                for head_idx,head in enumerate(heads):
                    if head==node_id:
                        id_out = head_idx
                        tens_name = self.model_name+"_out_"+str(id_out)
                        break
                mem_tensor = MatchMemoryTensor(name=tens_name,is_output=id_out!=-1,
                                               is_intermediate=id_out==-1,
                                                shape=tuple(shapes[node_id]),dtype=np.dtype(dtypes[node_id]),
                                                node_id=node_id)
                mem_tensors.append(mem_tensor)
                tensor_map[node["name"]+"_out"] = mem_tensor
                outputs = [mem_tensor]
                call_node = MatchGraphRuntimeNodeCall(inputs=inputs, outputs=outputs,
                                                      name=self.model_name+"_node_"+str(node_id),
                                                      fn_name=node["name"], node_info=node,
                                                      node_id=node_id)
                nodes.append(call_node)
                nodes_map[node["name"]] = call_node
        
        
        # set memory planner and run it
        self.mem_planner = MatchMemoryPlanner(mem_tensors=mem_tensors, available_soc_bytes=self.target.soc_memory_bytes,algorithm="match")
        self.soc_mem_needed_bytes, self.ext_mem_needed_bytes = self.mem_planner.generate()

        inputs = [tens for tens in mem_tensors if tens.is_input]
        outputs = [tens for tens in mem_tensors if tens.is_output]
        if not Path(self.out_path+"/parameters").absolute().is_dir():
            Path(self.out_path+"/parameters").absolute().mkdir()
        for mem_tensor in mem_tensors:
            if mem_tensor.stored_in_external_memory and mem_tensor.is_constant:
                mem_tensor.constant_val.flatten().astype("uint8").tofile(Path(self.out_path+f"/parameters/{self.model_name}_{mem_tensor.name}_data.hex"))
        # breakpoint()
        template_data = {
            "target": self.target,
            "mem_tensors": mem_tensors,
            "ext_mem_needed_bytes": self.ext_mem_needed_bytes,
            "soc_mem_needed_bytes": self.soc_mem_needed_bytes,
            "nodes": nodes,
            "model_name": self.model_name,
            "rt_inputs": inputs,
            "rt_outputs": outputs,
        }
        return template_data
