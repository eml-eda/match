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
import tvm

class MatchMemoryTensor:
    def __init__(
            self, name: str="p1", is_intermediate: bool=False,
            is_constant: bool=False, is_output: bool=False,
            is_input: bool=False,
            constant_val: npt.ArrayLike=np.array([1]),
            original_constant_val: npt.ArrayLike=np.array(1),
            shape: Tuple[int]=(1,),
            dtype: npt.DTypeLike=np.dtype("uint8"),
            node_id: int=0,
            node_info={}
        ):
        self.name=name
        self.is_intermediate=is_intermediate
        self.is_constant=is_constant
        self.is_output=is_output
        self.is_input=is_input
        if sum([self.is_intermediate,self.is_constant,self.is_output,self.is_input])!=1:
            raise Exception(f"Match Memory Tensor can only be one option between(intermediate,constant,output,input)")
        self.constant_val=constant_val
        self.original_constant_val=original_constant_val
        self.shape=shape
        self.dtype=dtype
        self.node_id=node_id
        self.last_usage=node_id
        self.mem_offset = -1
        self.stored_in_external_memory = False
        self.move_temp_to_ext_mem = list()
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

    @property
    def num_bytes(self):
        return self.prod_shape * self.dtype.itemsize
    
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
    
    @property
    def external_memory_needed(self):
        return self.total_memory_needed_bytes > self.available_soc_bytes

    def match_mem_planner_impl(self):
        sorted_mem_tensors = sorted([m_t for m_t in self.mem_tensors],
                                    key=lambda m_t:((m_t.num_bytes),m_t.lifetime_span))
        ext_mem_needed = 0
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}
        original_start_usage_tensors = {tensor:tensor.start_usage for tensor in sorted_mem_tensors}
        original_last_usage_tensors = {tensor:tensor.last_usage for tensor in sorted_mem_tensors}

        for tensor in sorted_mem_tensors:
            original_last_usage_tensors[tensor.name] = tensor.last_usage
            original_start_usage_tensors[tensor.name] = tensor.start_usage
            if tensor.is_input or tensor.is_output:
                tensor.start_usage = 0
                tensor.last_usage = self.last_timestep
    
        def check_if_any_valid_allocation(
            time: int=0,
            tens_size: int=1,
        ):
            if tens_size>self.available_soc_bytes:
                return False, 0
            len_at_time = len(tensors_allocated_at_time[time])
            if len_at_time>1:
                for tens_a,tens_b in zip(tensors_allocated_at_time[time][:len_at_time-1],tensors_allocated_at_time[time][1:]):
                    end_tens_a = tens_a.mem_offset_at[time]+tens_a.num_bytes
                    start_tens_b = tens_b.mem_offset_at[time]
                    if start_tens_b-end_tens_a >= tens_size:
                        return True, end_tens_a

                # check if theres enough space from the last tensor to the end of the memory
                tens_ = tensors_allocated_at_time[time][-1]
                end_tens = tens_.mem_offset_at[time]+tens_.num_bytes
                if self.available_soc_bytes-end_tens >= tens_size:
                    return True, end_tens
            elif len_at_time==1:
                # check if tensor can be allocated before
                tens = tensors_allocated_at_time[time][-1]
                end_tens = tens.mem_offset_at[time] + tens.num_bytes
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
                    end_tens_a = tens_a.mem_offset_at[time]+tens_a.num_bytes
                    start_tens_b = tens_b.mem_offset_at[time]
                    if start_tens_b-end_tens_a >= tens_size:
                        if start_tens_b>=offset>=end_tens_a and start_tens_b>=offset+tens_size>=end_tens_a:
                            return True

                # check if theres enough space from the last tensor to the end of the memory
                tens_ = tensors_allocated_at_time[time][-1]
                end_tens = tens_.mem_offset_at[time]+tens_.num_bytes
                if self.available_soc_bytes-end_tens >= tens_size:
                    if self.available_soc_bytes>=offset>=end_tens and self.available_soc_bytes>=offset+tens_size>=end_tens:
                        return True
            elif len_at_time==1:
                # check if tensor can be allocated before
                tens = tensors_allocated_at_time[time][-1]
                start_tens = tens.mem_offset_at[time]
                end_tens = tens.mem_offset_at[time] + tens.num_bytes
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
            separeted_intermediate_fine: bool=False,
            tens_size: int=1,
        ):
            allocated = False
            valid_for_all = True
            valid_for_separeted_intermediate = True
            separeted_last_pt = offset
            allocation_at = {allocation_time: offset}
            contiguous_area = False
            for time in range(tensor.start_usage, tensor.last_usage+1):
                if time in self.calls_idxs and time!=allocation_time:
                    if not check_if_valid_allocation(time=time, offset=offset, tens_size=tens_size):
                        valid_for_all = False
                    if time in tensor.used_at and valid_for_separeted_intermediate:
                        if contiguous_area:
                            if not check_if_valid_allocation(time=time, offset=separeted_last_pt, tens_size=tens_size):
                                valid_for_separeted_intermediate = False
                            else:
                                allocation_at[time] = separeted_last_pt
                        else:
                            found, new_off = check_if_any_valid_allocation(time=time, tens_size=tens_size) 
                            if valid_for_separeted_intermediate and found:
                                allocation_at[time] = new_off
                                separeted_last_pt = new_off
                            else:
                                valid_for_separeted_intermediate = False
                        contiguous_area = True
                    if time not in tensor.used_at:
                        contiguous_area = False
            if valid_for_all:
                for time in range(tensor.start_usage, tensor.last_usage+1):
                    if time in self.calls_idxs:
                        tensor.mem_offset_at[time] = offset
                        tensors_allocated_at_time[time] = sorted(tensors_allocated_at_time[time]+[tensor], key=lambda m_t:m_t.mem_offset_at[time])
                allocated=True
            
            elif valid_for_separeted_intermediate and separeted_intermediate_fine:
                in_ext_mem = tensor.is_input or tensor.is_constant
                loaded_first_time = False
                tensor.load_from_ext_mem_at = list()
                tensor.move_temp_to_ext_mem = list()
                if in_ext_mem:
                    tensor.stored_in_external_memory = True
                for time in range(tensor.start_usage, tensor.last_usage+1):
                    if time in self.calls_idxs:
                        if time in tensor.used_at:
                            if in_ext_mem:
                                tensor.load_from_ext_mem_at.append(time)
                                in_ext_mem = False
                                if not loaded_first_time:
                                    loaded_first_time = True
                            tensor.mem_offset_at[time] = allocation_at[time]
                            tensors_allocated_at_time[time] = sorted(tensors_allocated_at_time[time]+[tensor], key=lambda m_t:m_t.mem_offset_at[time])
                        else:
                            if loaded_first_time:
                                tensor.move_temp_to_ext_mem.append(time)
                            in_ext_mem = True
                allocated=True
            return allocated

        def try_allocate_congested(
            tensor: MatchMemoryTensor=None,
            separeted_intermediate_fine: bool=False,
            tens_size: int=1, time: int=0
        ):
            # allocate a single time
            allocated = False
            for tens_a,tens_b in zip(tensors_allocated_at_time[time][:len(tensors_allocated_at_time[time])-1],tensors_allocated_at_time[time][1:]):
                end_tens_a = tens_a.mem_offset_at[time]+tens_a.num_bytes
                start_tens_b = tens_b.mem_offset_at[time]
                # theres an empty cut big enough
                if start_tens_b-end_tens_a >= tens_size:
                    allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=end_tens_a,
                                                separeted_intermediate_fine=separeted_intermediate_fine,
                                                tens_size=tens_size)
                    if allocated:
                        break

            if not allocated:
                # check if theres enough space from the last tensor to the end of the memory
                tens_ = tensors_allocated_at_time[time][-1]
                end_tens = tens_.mem_offset_at[time]+tens_.num_bytes
                while self.available_soc_bytes-end_tens >= tens_size:
                    allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=end_tens,
                                                separeted_intermediate_fine=separeted_intermediate_fine,
                                                tens_size=tens_size)
                    if allocated:
                        break
                    else:
                        end_tens+=tens_size
            return allocated
        
        def try_allocate_buffer(
            tensor: MatchMemoryTensor=None,
            separeted_intermediate_fine: bool=False,
            tens_size: int=1, time: int=0
        ):
            # allocate a single time
            allocated = False
            # check if theres enough space from the last tensor to the end of the memory
            tens_ = tensors_allocated_at_time[time][-1]
            end_tens = tens_.mem_offset_at[time]+tens_.num_bytes
            if tens_.mem_offset_at[time]>=tens_size:
                allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=0,
                                            separeted_intermediate_fine=separeted_intermediate_fine,
                                            tens_size=tens_size)
            if not allocated:
                while self.available_soc_bytes-end_tens >= tens_size:
                    allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=end_tens,
                                                separeted_intermediate_fine=separeted_intermediate_fine,
                                                tens_size=tens_size)
                    if allocated:
                        break
                    else:
                        end_tens+=tens_size
            return allocated
        
        def try_allocate_easy(
            tensor: MatchMemoryTensor=None,
            separeted_intermediate_fine: bool=False,
            tens_size: int=1, time: int=0
        ):
            # allocate a single time
            allocated = False
            # check if theres enough space from the last tensor to the end of the memory
            if self.available_soc_bytes>=tens_size:
                end_tens = 0
                while self.available_soc_bytes-end_tens >= tens_size:
                    allocated = try_to_allocate(tensor=tensor,allocation_time=time,offset=end_tens,
                                                separeted_intermediate_fine=separeted_intermediate_fine,
                                                tens_size=tens_size)
                    if allocated:
                        break
                    else:
                        end_tens+=tens_size
            return allocated

        def allocate_tensor(
            tensor: MatchMemoryTensor=None):
            tens_size = tensor.num_bytes
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
                        allocated = try_allocate_congested(tensor=tensor,
                                                           separeted_intermediate_fine=True, tens_size=tens_size,
                                                           time=max_allocated_tensors_at)
                
                elif len(tensors_allocated_at_time[max_allocated_tensors_at])==1:
                    allocated = try_allocate_buffer(tensor=tensor, tens_size=tens_size, time=max_allocated_tensors_at)
                    if not allocated:
                        allocated = try_allocate_buffer(tensor=tensor,
                                                        separeted_intermediate_fine=True, tens_size=tens_size,
                                                        time=max_allocated_tensors_at)
                else:
                    # try to allocate at 0
                    allocated = try_allocate_easy(tensor=tensor, tens_size=tens_size, time=max_allocated_tensors_at)
                    if not allocated:
                        allocated = try_allocate_easy(tensor=tensor, 
                                                      separeted_intermediate_fine=True, tens_size=tens_size,
                                                      time=max_allocated_tensors_at)
            if not allocated:
                print(f"[MEMORY PLANNER] Couldnt allocate all the tensors, tensor {tensor.name} allocation was not successfull")
                print(f"[MEMORY PLANNER] Node at {max_allocated_tensors_at} cannot fit SoC memory")
                # TODO: add list of nodes to run from external memory
                raise Exception(f"[MEMORY PLANNER] Couldnt allocate all the tensors, tensor {tensor.name} allocation was not successfull")
        
            for time_ in tensor.mem_offset_at:
                free_size_at_time[time_] -= tensor.num_bytes
        
        print(f"[MEMORY PLANNER] Allocating tensors with {self.available_soc_bytes} bytes of on-chip memory")

        for tensor in sorted_mem_tensors:
            allocate_tensor(tensor=tensor)
        
        print(f"[MEMORY PLANNER] All tensors allocated")
        # remove constants allocated in the SoC already
        real_constant_tensors = list()
        for tensor in sorted_mem_tensors:
            if tensor.is_constant and not tensor.stored_in_external_memory:
                store_as_constant = True
                for time in free_size_at_time:
                    if time not in tensor.used_at and free_size_at_time[time]<=tensor.num_bytes:
                        store_as_constant = False
                        break
                if store_as_constant:
                    self.available_soc_bytes -= tensor.num_bytes
                    for time in free_size_at_time:
                        if time not in tensor.used_at:
                            free_size_at_time[time] -= tensor.num_bytes
                    real_constant_tensors.append(tensor.name)
                else:
                    print(f"[MEMORY PLANNER] Constant tensor {tensor.name} will be stored in external memory")
            tensor.mem_offset_at = dict()
        sorted_mem_tensors = [m_t for m_t in sorted_mem_tensors if m_t.name not in real_constant_tensors]
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}
        print(f"[MEM PLANNER] Moved actual constants to on-chip memory, now there are {self.available_soc_bytes} bytes of available on-chip memory")
        # reallocate these intermediate and tensors who may have to stay in SoC memory
        for tensor in sorted_mem_tensors:
            tensor.stored_in_external_memory = False
            allocate_tensor(tensor=tensor)
        
        print("[MEMORY PLANNER] Moved to on-chip memory all possible constants and reallocated other tensors")
        # now check if inputs and outputs can stay always in SoC memory
        
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}
        # remove constants allocated in the SoC already
        real_constant_tensors = list()
        for tensor in sorted_mem_tensors:
            if tensor.is_constant and not tensor.stored_in_external_memory:
                store_as_constant = True
                for time in free_size_at_time:
                    if time not in tensor.used_at and free_size_at_time[time]<=tensor.num_bytes:
                        store_as_constant = False
                        break
                if store_as_constant:
                    self.available_soc_bytes -= tensor.num_bytes
                    for time in free_size_at_time:
                        if time not in tensor.used_at:
                            free_size_at_time[time] -= tensor.num_bytes
                    real_constant_tensors.append(tensor.name)
                else:
                    print(f"[MEMORY PLANNER] Constant tensor {tensor.name} will be stored in external memory")
            tensor.mem_offset_at = dict()

        for tensor in sorted_mem_tensors:
            if (tensor.is_input or tensor.is_output) and not tensor.stored_in_external_memory:
                if self.available_soc_bytes<=tensor.num_bytes:
                    self.available_soc_bytes -= tensor.num_bytes
                else:
                    self.stored_in_external_memory = True
            
        sorted_mem_tensors = [m_t for m_t in sorted_mem_tensors if m_t.name not in real_constant_tensors and not ((tensor.is_input or tensor.is_output) and not tensor.stored_in_external_memory)]
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}

        print(f"[MEM PLANNER] Moved on-chip inputs and outputs to on-chip memory, now there are {self.available_soc_bytes} bytes of available on-chip memory")
        # reallocate again
        for tensor in sorted_mem_tensors:
            tensor.stored_in_external_memory = False
            allocate_tensor(tensor=tensor)

        print("[MEMORY PLANNER] Moved to on-chip memory all possible inputs and outputs and reallocated other tensors")
        # compute external memory needed
        for tensor in sorted_mem_tensors:
            # ext mem for inputs and outputs doesnt count here...
            if len(tensor.load_from_ext_mem_at)>0 or tensor.stored_in_external_memory:
                ext_mem_needed+=tensor.num_bytes
            tensor.mem_offset = tensor.mem_offset_at[tensor.used_at[0]] if len(tensor.used_at)>0 else self.available_soc_bytes
            if tensor.is_output or tensor.is_input:
                tensor.start_usage = original_start_usage_tensors[tensor.name]
                tensor.last_usage = original_last_usage_tensors[tensor.name]

        soc_mem_needed = max([sum([m_t.num_bytes for m_t in tensors]) for tensors in tensors_allocated_at_time.values()])
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
    def __init__(self, target: MatchTarget, mod_info: Dict[str,Any], params=None,
                 model_name: str="default", out_path: str="model_out", match_inputs=None,
                 host_module=None):
        self.target = target
        self.mod_info = mod_info
        self.params = params
        self.model_name = model_name
        self.out_path = out_path
        self.mem_planner = None
        self.ext_mem_needed_bytes = 0
        self.mem_needed_bytes = 0
        self.match_inputs = match_inputs
        self.host_module = host_module
        self.dev = tvm.cpu(0)

    def generate(self):
        tensor_map = {}
        nodes_map = {}
        map_names = dict()
        mem_tensors = []
        nodes = []
        dtypes = self.mod_info["attrs"]["dltype"][1]
        shapes = self.mod_info["attrs"]["shape"][1]
        heads = [head[0] for head in self.mod_info["heads"]]
        nop_maps = dict()
        activations = dict()
        for match_inp in self.match_inputs.values():
            activations[match_inp["name"]] = match_inp["np_values"]
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
                    map_names[node["name"]] = (mem_tensor.name, mem_tensor.name, mem_tensor.name)
                else:
                    mem_tensor = MatchMemoryTensor(name=node["name"],is_input=True,
                                                   shape=tuple(shapes[node_id]),dtype=np.dtype(dtypes[node_id]),
                                                   node_id=node_id, node_info=node)
                    mem_tensors.append(mem_tensor)
                    tensor_map[node["name"]] = mem_tensor
                    map_names[node["name"]] = (mem_tensor.name, mem_tensor.name, mem_tensor.name)
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
                host_lib = None
                if "match" in node["name"]:
                    match_node, schedule, match_node_name, cpu_only_c_lib, host_lib = get_fname_node_schedule(node["name"])
                    if match_node is not None and schedule is not None:
                        for w_tensor in match_node.const_tensors.values():
                            if w_tensor.name in schedule.tensors:
                                w_tensor = schedule.tensors[w_tensor.name]
                                mem_tensor = MatchMemoryTensor(name=match_node_name+"_"+w_tensor.name,is_constant=True,
                                                            constant_val=w_tensor.data,original_constant_val=w_tensor.original_data,
                                                            shape=w_tensor.data.shape,
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
                node_activations = list()
                for tens_inp in inputs:
                    if tens_inp.is_input or tens_inp.is_intermediate:
                        node_activations.append(tvm.nd.array(activations[tens_inp.name]))
                    elif tens_inp.is_constant:
                        node_activations.append(tvm.nd.array(tens_inp.original_constant_val))
                # breakpoint()
                if "match" not in node["name"]:
                    mem_tensor.used_by_tvm = True
                    output_nd = tvm.nd.empty(shape=mem_tensor.shape, dtype=mem_tensor.dtype)
                    self.host_module[node["attrs"]["func_name"]](*node_activations, output_nd)
                    activations[mem_tensor.name] = output_nd.numpy()
                else:
                    module = tvm.contrib.graph_executor.GraphModule(host_lib["default"](self.dev))
                    for tens_inp, param in zip(node_activations, host_lib.ir_mod["main"].params):
                        module.set_input(param.name_hint, tens_inp)
                    module.run()
                    output_np = module.get_output(0).numpy()
                    activations[mem_tensor.name] = output_np
                mem_tensors.append(mem_tensor)
                tensor_map[node["name"]+"_out"] = mem_tensor
                outputs = [mem_tensor]
                call_node = MatchGraphRuntimeNodeCall(inputs=inputs, outputs=outputs,
                                                      name=self.model_name+"_node_"+str(node_id),
                                                      fn_name=node["attrs"]["func_name"], node_info=node,
                                                      node_id=node_id, node_name=match_node_name,
                                                      schedule=schedule, match_node=match_node)
                nodes.append(call_node)
                nodes_map[node["name"]] = call_node
                map_names[tens_name] = (call_node.name, node["name"]+"_out", node["name"])
        
        # set memory planner and run it
        self.mem_planner = MatchMemoryPlanner(mem_tensors=mem_tensors, available_soc_bytes=self.target.soc_memory_bytes,
                                              calls_idxs=[node.node_id for node in nodes], nodes=nodes, algorithm="match")
        self.mem_needed_bytes, self.ext_mem_needed_bytes = self.mem_planner.generate()

        inputs = [tens for tens in mem_tensors if tens.is_input]
        outputs = [tens for tens in mem_tensors if tens.is_output]
        if not Path(self.out_path+"/parameters").absolute().is_dir():
            Path(self.out_path+"/parameters").absolute().mkdir()
        if not Path(self.out_path+"/golden").absolute().is_dir():
            Path(self.out_path+"/golden").absolute().mkdir()
        for mem_tensor in mem_tensors:
            if mem_tensor.stored_in_external_memory and mem_tensor.is_constant:
                np.frombuffer(mem_tensor.constant_val.flatten().tobytes(),dtype="uint8").tofile(Path(self.out_path+f"/parameters/{self.model_name}_{mem_tensor.name}_data.hex"))
        for activation_name, activation in activations.items():
            mem_tensor_ = None
            for m_t in mem_tensors:
                if m_t.name==activation_name:
                    if not m_t.is_input:
                        mem_tensor_ = m_t
                    break
            if mem_tensor_ is not None:
                np.frombuffer(activation.flatten().tobytes(),dtype="uint8").tofile(Path(self.out_path+f"/golden/{self.model_name}_{activation_name}_data.hex"))
        template_data = {
            "target": self.target,
            "mem_tensors": mem_tensors,
            "ext_mem_needed_bytes": self.ext_mem_needed_bytes,
            "mem_needed_bytes": self.mem_needed_bytes,
            "nodes": nodes,
            "model_name": self.model_name,
            "tensor_map": tensor_map,
            "nodes_map": nodes_map,
            "rt_inputs": inputs,
            "rt_outputs": outputs,
            "activations": activations,
            "map_names": map_names,
            "checksums": {activation_name: np.frombuffer(activation.flatten().tobytes(),dtype="uint8").sum() for activation_name, activation in activations.items()},
        }
        return template_data
