from math import prod
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from numpy import typing as npt
import mako

from match.node.node import MatchNode
from match.target.target import MatchTarget
from match.utils.utils import c_friendly_npvalue, get_fname_node_schedule, numpy_dtype_to_c_type


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
        self.mem_offset = -1
        self.stored_in_external_memory = False
        self.move_temp_to_ext_mem = dict()
        self.load_from_ext_mem_at = list()
        self.c_type = numpy_dtype_to_c_type(self.dtype)
        self.c_value = "{}" if not self.is_constant else c_friendly_npvalue(self.constant_val)
        self.prod_shape = prod(self.shape) 
        self.node_info = node_info
        self.start_usage = -1 if (int(self.is_intermediate)+int(self.is_output))==0 else self.node_id
        self.used_at = list()
        self.mem_offset_at = dict()
        self.used_by_tvm = False

    @property
    def lifetime(self):
        return (self.start_usage,self.last_usage)
    
    @property
    def lifetime_span(self):
        return self.last_usage-self.start_usage

    @property
    def elems(self):
        return prod(self.shape)

    def update_last_usage(self,new_ending_idx):
        if self.start_usage==-1:
            self.start_usage=new_ending_idx
        self.used_at.append(new_ending_idx)
        self.last_usage=new_ending_idx

class MatchMemoryPlanner:
    def __init__(self, mem_tensors:List[MatchMemoryTensor], available_soc_bytes:int=1024, calls_idxs: List=[], nodes: List=[], algorithm: str="match"):
        self.mem_tensors = mem_tensors
        self.available_soc_bytes = available_soc_bytes
        self.algorithm = algorithm
        self.last_timestep = max([tens.last_usage for tens in self.mem_tensors])
        self.intermediate_memory_usage = [0] * (self.last_timestep + 1)
        self.overall_intermediate_memory_usage = [0] * (self.last_timestep + 1)
        self.input_memory_usage = 0
        self.constant_memory_usage = 0
        self.output_memory_usage = 0
        self.calls_idxs = calls_idxs
        self.nodes = nodes
        for tensor in self.mem_tensors:
            if not tensor.is_input and not tensor.is_output and not tensor.is_constant:
                for time in range(tensor.node_id, tensor.last_usage + 1):
                    if time in self.calls_idxs:
                        self.intermediate_memory_usage[time] += tensor.elems * tensor.dtype.itemsize
                        self.overall_intermediate_memory_usage[time] += tensor.elems * tensor.dtype.itemsize
            elif tensor.is_constant:
                self.constant_memory_usage += tensor.elems * tensor.dtype.itemsize
                for time in range(tensor.node_id, tensor.last_usage + 1):
                    if time in self.calls_idxs:
                        self.overall_intermediate_memory_usage[time] += tensor.elems * tensor.dtype.itemsize
            elif tensor.is_input:
                self.input_memory_usage += tensor.elems * tensor.dtype.itemsize
            elif tensor.is_output:
                self.output_memory_usage += tensor.elems * tensor.dtype.itemsize
        self.total_memory_needed_bytes = self.input_memory_usage + self.output_memory_usage + self.constant_memory_usage + max(self.intermediate_memory_usage)
        self.total_memory_needed_bytes_w_consts = self.input_memory_usage + self.output_memory_usage + max(self.overall_intermediate_memory_usage)
        self.available_soc_bytes -= (self.input_memory_usage + self.output_memory_usage)
    
    @property
    def external_memory_needed(self):
        return self.total_memory_needed_bytes > self.available_soc_bytes

    def match_mem_planner_impl(self):
        sorted_mem_tensors = sorted([m_t for m_t in self.mem_tensors if m_t.is_intermediate or m_t.is_constant],
                                    key=lambda m_t:(-(m_t.prod_shape*m_t.dtype.itemsize),-m_t.lifetime_span))
        ext_mem_needed = 0
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}

        def check_if_any_valid_allocation(
            time: int=0,
            tens_size: int=1,
        ):
            if tens_size>self.available_soc_bytes:
                return False, 0
            len_at_time = len(tensors_allocated_at_time[time])
            if len_at_time>1:
                for tens_a,tens_b in zip(tensors_allocated_at_time[time][:len_at_time-1],tensors_allocated_at_time[time][1:]):
                    end_tens_a = tens_a.mem_offset_at[time]+tens_a.prod_shape*tens_a.dtype.itemsize
                    start_tens_b = tens_b.mem_offset_at[time]
                    if start_tens_b-end_tens_a >= tens_size:
                        return True, end_tens_a

                # check if theres enough space from the last tensor to the end of the memory
                tens_ = tensors_allocated_at_time[time][-1]
                end_tens = tens_.mem_offset_at[time]+tens_.prod_shape*tens_.dtype.itemsize
                if self.available_soc_bytes-end_tens >= tens_size:
                    return True, end_tens
            elif len_at_time==1:
                # check if tensor can be allocated before
                tens = tensors_allocated_at_time[time][-1]
                end_tens = tens.mem_offset_at[time] + tens.prod_shape*tens.dtype.itemsize
                if tens.mem_offset_at[time]>=tens_size:
                    return True, 0
                elif self.available_soc_bytes-end_tens >= tens_size:
                    return True, end_tens
            else:
                return True, 0
            
            return False, 0
        
        def check_if_valid_allocation(
            time: int=0,
            offset: int=0,
            tens_size: int=1,
        ):
            if tens_size>self.available_soc_bytes:
                return False
            len_at_time = len(tensors_allocated_at_time[time])
            if len_at_time>1:
                for tens_a,tens_b in zip(tensors_allocated_at_time[time][:len_at_time-1],tensors_allocated_at_time[time][1:]):
                    end_tens_a = tens_a.mem_offset_at[time]+tens_a.prod_shape*tens_a.dtype.itemsize
                    start_tens_b = tens_b.mem_offset_at[time]
                    if start_tens_b-end_tens_a >= tens_size:
                        if start_tens_b>=offset>=end_tens_a and start_tens_b>=offset+tens_size>=end_tens_a:
                            return True

                # check if theres enough space from the last tensor to the end of the memory
                tens_ = tensors_allocated_at_time[time][-1]
                end_tens = tens_.mem_offset_at[time]+tens_.prod_shape*tens_.dtype.itemsize
                if self.available_soc_bytes-end_tens >= tens_size:
                    if self.available_soc_bytes>=offset>=end_tens and self.available_soc_bytes>=offset+tens_size>=end_tens:
                        return True
            elif len_at_time==1:
                # check if tensor can be allocated before
                tens = tensors_allocated_at_time[time][-1]
                start_tens = tens.mem_offset_at[time]
                end_tens = tens.mem_offset_at[time] + tens.prod_shape*tens.dtype.itemsize
                if tens.mem_offset_at[time]>=tens_size:
                    if offset<=start_tens and offset+tens_size<=start_tens:
                        return True
                if self.available_soc_bytes>=offset>=end_tens and self.available_soc_bytes>=offset+tens_size>=end_tens:
                    return True
            else:
                return True

            return False
        
        def try_to_allocate(
            tensor: MatchMemoryTensor=None,
            allocation_time: int=0,
            offset: int=0,
            intermediate_store_fine: bool=False,
            separeted_intermediate_fine: bool=False,
            tens_size: int=1,
        ):
            allocated = False
            valid_for_all = True
            valid_for_all_used = True
            valid_for_separeted_intermediate = True
            separeted_last_pt = offset
            allocation_at = {allocation_time: offset}
            space_between_idx = 0
            external_mem_needed = 0
            for time in range(tensor.start_usage, tensor.last_usage+1):
                if time in self.calls_idxs and time!=allocation_time:
                    if not check_if_valid_allocation(time=time, offset=offset, tens_size=tens_size):
                        if time in tensor.used_at:
                            valid_for_all_used = False
                        valid_for_all = False
                    if time in tensor.used_at and valid_for_separeted_intermediate:
                        if space_between_idx==0:
                            if not check_if_valid_allocation(time=time, offset=separeted_last_pt, tens_size=tens_size):
                                valid_for_separeted_intermediate = False
                            space_between_idx+=1
                        else:
                            found, new_off = check_if_any_valid_allocation(time=time, tens_size=tens_size) 
                            if valid_for_separeted_intermediate and found:
                                allocation_at[time] = new_off
                                separeted_last_pt = new_off
                            space_between_idx=0
            if valid_for_all:
                for time in range(tensor.start_usage, tensor.last_usage+1):
                    if time in self.calls_idxs:
                        tensor.mem_offset_at[time] = offset
                        tensors_allocated_at_time[time] = sorted(tensors_allocated_at_time[time]+[tensor], key=lambda m_t:m_t.mem_offset_at[time])
                allocated=True
            
            elif valid_for_all_used and intermediate_store_fine:
                in_ext_mem = tensor.is_constant
                if tensor.is_constant:
                    tensor.stored_in_external_memory = True
                for time in range(tensor.start_usage, tensor.last_usage+1):
                    if time in self.calls_idxs:
                        if time in tensor.used_at:
                            if in_ext_mem:
                                tensor.load_from_ext_mem_at.append(time)
                                in_ext_mem = False
                            tensor.mem_offset_at[time] = offset
                            tensors_allocated_at_time[time] = sorted(tensors_allocated_at_time[time]+[tensor], key=lambda m_t:m_t.mem_offset_at[time])
                        else:
                            tensor.move_temp_to_ext_mem.append(time)
                            in_ext_mem = True
                allocated=True
            
            elif valid_for_separeted_intermediate and separeted_intermediate_fine:
                in_ext_mem = tensor.is_constant
                if tensor.is_constant:
                    tensor.stored_in_external_memory = True
                for time in range(tensor.start_usage, tensor.last_usage+1):
                    if time in self.calls_idxs:
                        if time in tensor.used_at:
                            if in_ext_mem:
                                tensor.load_from_ext_mem_at.append(time)
                                in_ext_mem = False
                            tensor.mem_offset_at[time] = allocation_at[time]
                            tensors_allocated_at_time[time] = sorted(tensors_allocated_at_time[time]+[tensor], key=lambda m_t:m_t.mem_offset_at[time])
                        else:
                            tensor.move_temp_to_ext_mem.append(time)
                            in_ext_mem = True
                allocated=True
            return allocated

        def try_allocate_congested(
            tensor: MatchMemoryTensor=None,
            intermediate_store_fine: bool=False,
            separeted_intermediate_fine: bool=False,
            tens_size: int=1, time: int=0
        ):
            # allocate a single time
            allocated = False
            for tens_a,tens_b in zip(tensors_allocated_at_time[time][:len(tensors_allocated_at_time[time])-1],tensors_allocated_at_time[time][1:]):
                end_tens_a = tens_a.mem_offset_at[time]+tens_a.prod_shape*tens_a.dtype.itemsize
                start_tens_b = tens_b.mem_offset_at[time]
                # theres an empty cut big enough
                if start_tens_b-end_tens_a >= tens_size:
                    allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=end_tens_a,
                                                intermediate_store_fine=intermediate_store_fine,
                                                separeted_intermediate_fine=separeted_intermediate_fine,
                                                tens_size=tens_size)
                    if allocated:
                        break

            if not allocated:
                # check if theres enough space from the last tensor to the end of the memory
                tens_ = tensors_allocated_at_time[time][-1]
                end_tens = tens_.mem_offset_at[time]+tens_.prod_shape*tens_.dtype.itemsize
                while self.available_soc_bytes-end_tens >= tens_size:
                    allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=end_tens,
                                                intermediate_store_fine=intermediate_store_fine,
                                                separeted_intermediate_fine=separeted_intermediate_fine,
                                                tens_size=tens_size)
                    if allocated:
                        break
                    else:
                        end_tens+=tens_size
            return allocated
        
        def try_allocate_buffer(
            tensor: MatchMemoryTensor=None,
            intermediate_store_fine: bool=False,
            separeted_intermediate_fine: bool=False,
            tens_size: int=1, time: int=0
        ):
            # allocate a single time
            allocated = False
            # check if theres enough space from the last tensor to the end of the memory
            tens_ = tensors_allocated_at_time[time][-1]
            end_tens = tens_.mem_offset_at[time]+tens_.prod_shape*tens_.dtype.itemsize
            if tens_.mem_offset_at[time]>=tens_size:
                allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=0,
                                            intermediate_store_fine=intermediate_store_fine,
                                            separeted_intermediate_fine=separeted_intermediate_fine,
                                            tens_size=tens_size)
            if not allocated:
                if self.available_soc_bytes-end_tens >= tens_size:
                    allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=end_tens,
                                                intermediate_store_fine=intermediate_store_fine,
                                                separeted_intermediate_fine=separeted_intermediate_fine,
                                                tens_size=tens_size)
            return allocated
        
        def try_allocate_easy(
            tensor: MatchMemoryTensor=None,
            intermediate_store_fine: bool=False,
            separeted_intermediate_fine: bool=False,
            tens_size: int=1, time: int=0
        ):
            # allocate a single time
            allocated = False
            # check if theres enough space from the last tensor to the end of the memory
            if self.available_soc_bytes>=tens_size:
                allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=0,
                                            intermediate_store_fine=intermediate_store_fine,
                                            separeted_intermediate_fine=separeted_intermediate_fine,
                                            tens_size=tens_size)
            return allocated

        def allocate_tensor(
            tensor: MatchMemoryTensor=None):
            tens_size = tensor.prod_shape * tensor.dtype.itemsize
            max_allocated_tensors_at = -1
            less_free_mem_at = -1
            for time in range(tensor.start_usage, tensor.last_usage+1):
                if time in self.calls_idxs:
                    if max_allocated_tensors_at==-1 or len(tensors_allocated_at_time[time])>len(tensors_allocated_at_time[max_allocated_tensors_at]):
                        max_allocated_tensors_at = time
                    if less_free_mem_at==-1 or free_size_at_time[time]<free_size_at_time[less_free_mem_at]:
                        less_free_mem_at = time
            # found the most congested spot
            allocated = False
            if max_allocated_tensors_at!=-1:
                if len(tensors_allocated_at_time[max_allocated_tensors_at])>1:
                    allocated = try_allocate_congested(tensor=tensor, tens_size=tens_size, time=max_allocated_tensors_at)
                    if not allocated:
                        allocated = try_allocate_congested(tensor=tensor, intermediate_store_fine=True, tens_size=tens_size,
                                                           time=max_allocated_tensors_at)
                    # last try moving around the offset if possible also
                    if not allocated:
                        allocated = try_allocate_congested(tensor=tensor, intermediate_store_fine=True,
                                                           separeted_intermediate_fine=True, tens_size=tens_size,
                                                           time=max_allocated_tensors_at)
                
                elif len(tensors_allocated_at_time[max_allocated_tensors_at])==1:
                    allocated = try_allocate_buffer(tensor=tensor, tens_size=tens_size, time=max_allocated_tensors_at)
                    if not allocated:
                        allocated = try_allocate_buffer(tensor=tensor, intermediate_store_fine=True, tens_size=tens_size,
                                                        time=max_allocated_tensors_at)
                    # last try moving around the offset if possible also
                    if not allocated:
                        allocated = try_allocate_buffer(tensor=tensor, intermediate_store_fine=True,
                                                        separeted_intermediate_fine=True, tens_size=tens_size,
                                                        time=max_allocated_tensors_at)
                else:
                    # try to allocate at 0
                    allocated = try_allocate_easy(tensor=tensor, tens_size=tens_size, time=max_allocated_tensors_at)
                    if not allocated:
                        allocated = try_allocate_easy(tensor=tensor, intermediate_store_fine=True, tens_size=tens_size,
                                                      time=max_allocated_tensors_at)
                    # last try moving around the offset if possible also
                    if not allocated:
                        allocated = try_allocate_easy(tensor=tensor, intermediate_store_fine=True,
                                                      separeted_intermediate_fine=True, tens_size=tens_size,
                                                      time=max_allocated_tensors_at)
            if not allocated:
                print(f"[MEMORY PLANNER] Couldnt allocate all the tensors, tensor {tensor.name} allocation was not successfull")
                raise Exception(f"[MEMORY PLANNER] Couldnt allocate all the tensors, tensor {tensor.name} allocation was not successfull")
        
            for time_ in tensor.mem_offset_at:
                free_size_at_time[time_] -= tensor.prod_shape * tensor.dtype.itemsize
            
        for tensor in sorted_mem_tensors:
            allocate_tensor(tensor=tensor)
        
        # now with everything setupped lets check which constants cannot be stored in the soc memory
        for tensor in sorted_mem_tensors[::-1]:
            if tensor.is_constant and not tensor.stored_in_external_memory:
                keep_in_main_mem = True
                tens_size = tensor.prod_shape*tensor.dtype.itemsize
                for time in self.calls_idxs:
                    if time not in tensor.mem_offset_at:
                        if free_size_at_time[time]-tens_size<0:
                            keep_in_main_mem = False
                            break
                if keep_in_main_mem:
                    for time in self.calls_idxs:
                        if time not in tensor.mem_offset_at:
                            free_size_at_time[time]-=tens_size
                else:
                    tensor.stored_in_external_memory=True
                    tensor.load_from_ext_mem_at.append(tensor.used_at[0])
        # remove constants allocated in the SoC already
        for tensor in sorted_mem_tensors:
            if tensor.is_constant and not tensor.stored_in_external_memory:
                self.available_soc_bytes -= tensor.prod_shape*tensor.dtype.itemsize
            tensor.mem_offset_at = dict()
        sorted_mem_tensors = [m_t for m_t in sorted_mem_tensors if m_t.is_intermediate or (m_t.is_constant and m_t.stored_in_external_memory)]
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}
        # reallocate these intermediate and tensors who may have to stay in SoC memory
        for tensor in sorted_mem_tensors:
            tensor.stored_in_external_memory = False
            allocate_tensor(tensor=tensor)

        # compute external memory needed
        for tensor in sorted_mem_tensors:
            if len(tensor.load_from_ext_mem_at)>0:
                ext_mem_needed+=tensor.prod_shape*tensor.dtype.itemsize
            tensor.mem_offset = tensor.mem_offset_at[tensor.used_at[0]]

        soc_mem_needed = self.available_soc_bytes-min([val for val in free_size_at_time.values()])
        return soc_mem_needed, ext_mem_needed

    def generate(self):
        # use only SoC memory if possible, otherwise use external memory
        try:
            return self.match_mem_planner_impl()
        except Exception as exc:
            print(f"[MEMORY PLANNER] Error during memory planner {exc}")
            raise Exception("Not enough SoC memory available")
    
class MatchGraphRuntimeNodeCall:
    def __init__(self, inputs=None, outputs=None, fn_name="default_lib_1",
                 name="default_lib_1", node_info={}, node_id: int=0, 
                 node_name: str=None, schedule=None, match_node=None):
        self.inputs = inputs
        self.outputs = outputs
        self.fn_name = fn_name
        self.name = name
        self.node_info = node_info
        self.fallback = "match" not in self.fn_name
        self.node_id = node_id
        self.node_name = node_name
        self.schedule = schedule
        self.match_node = match_node


class MatchTVMGraphRuntime:
    def __init__(self, target: MatchTarget, mod_info: Dict[str,Any], params=None, model_name: str="default", out_path: str="model_out"):
        self.target = target
        self.mod_info = mod_info
        self.params = params
        self.model_name = model_name
        self.out_path = out_path
        self.mem_planner = None
        self.ext_mem_needed_bytes = 0
        self.mem_needed_bytes = 0

    def generate(self):
        tensor_map = {}
        nodes_map = {}
        mem_tensors = []
        nodes = []
        dtypes = self.mod_info["attrs"]["dltype"][1]
        shapes = self.mod_info["attrs"]["shape"][1]
        heads = [head[0] for head in self.mod_info["heads"]]
        nop_maps = dict()
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
                inputs = []
                for inp_node_idx in [inp_node_idxs[0] for inp_node_idxs in node["inputs"]]:
                    if self.mod_info["nodes"][inp_node_idx]["op"]!="null" and "_nop" in self.mod_info["nodes"][inp_node_idx]["name"]\
                        and self.mod_info["nodes"][inp_node_idx]["name"] in nop_maps:
                        # inputs is a nop skip
                        inputs.append(nop_maps[self.mod_info["nodes"][inp_node_idx]["name"]])
                    else:
                        name_tens = self.mod_info["nodes"][inp_node_idx]["name"]
                        if self.mod_info["nodes"][inp_node_idx]["op"]!="null":
                            name_tens = name_tens+"_out"
                        inputs.append(tensor_map[name_tens])
                if "_nop" in node["name"]:
                    if len(inputs)==1:
                        nop_maps[node["name"]] = inputs[0]
                    continue
                
                match_node, schedule, match_node_name = (None, None, None)
                if "match" in node["name"]:
                    match_node, schedule, match_node_name = get_fname_node_schedule(node["name"])
                    for w_tensor in match_node.const_tensors.values():
                        if w_tensor.name in schedule.tensors:
                            w_tensor = schedule.tensors[w_tensor.name]
                            mem_tensor = MatchMemoryTensor(name=match_node_name+"_"+w_tensor.name,is_constant=True,
                                                           constant_val=w_tensor.data,shape=w_tensor.data.shape,
                                                           dtype=w_tensor.dtype,node_id=node_id,
                                                           node_info=node)
                            mem_tensors.append(mem_tensor)
                            tensor_map[w_tensor.name] = mem_tensor
                            inputs.append(mem_tensor)
                # inputs = [mem_tensors[inp_node_idx] for inp_node_idx in [inp_node_idxs[0] for inp_node_idxs in node["inputs"]]]
                for inp in inputs:
                    if "match" not in node["name"]:
                        inp.used_by_tvm = True
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
                if "match" not in node["name"]:
                    mem_tensor.used_by_tvm = True
                mem_tensors.append(mem_tensor)
                tensor_map[node["name"]+"_out"] = mem_tensor
                outputs = [mem_tensor]
                call_node = MatchGraphRuntimeNodeCall(inputs=inputs, outputs=outputs,
                                                      name=self.model_name+"_node_"+str(node_id),
                                                      fn_name=node["name"], node_info=node,
                                                      node_id=node_id, node_name=match_node_name,
                                                      schedule=schedule, match_node=match_node)
                nodes.append(call_node)
                nodes_map[node["name"]] = call_node
        
        
        # set memory planner and run it
        self.mem_planner = MatchMemoryPlanner(mem_tensors=mem_tensors, available_soc_bytes=self.target.soc_memory_bytes,
                                              calls_idxs=[node.node_id for node in nodes], nodes=nodes, algorithm="match")
        self.mem_needed_bytes, self.ext_mem_needed_bytes = self.mem_planner.generate()

        inputs = [tens for tens in mem_tensors if tens.is_input]
        outputs = [tens for tens in mem_tensors if tens.is_output]
        if not Path(self.out_path+"/parameters").absolute().is_dir():
            Path(self.out_path+"/parameters").absolute().mkdir()
        for mem_tensor in mem_tensors:
            if mem_tensor.stored_in_external_memory and mem_tensor.is_constant:
                np.frombuffer(mem_tensor.constant_val.flatten().tobytes(),dtype="uint8").tofile(Path(self.out_path+f"/parameters/{self.model_name}_{mem_tensor.name}_data.hex"))
        template_data = {
            "target": self.target,
            "mem_tensors": mem_tensors,
            "ext_mem_needed_bytes": self.ext_mem_needed_bytes,
            "mem_needed_bytes": self.mem_needed_bytes,
            "nodes": nodes,
            "model_name": self.model_name,
            "rt_inputs": inputs,
            "rt_outputs": outputs,
        }
        return template_data
