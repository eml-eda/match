
from typing import List
from match.runtime.graph.tensor import MatchMemoryTensor
from match.runtime.graph.alloc import allocate_tensor
from match.runtime.graph.utils import save_memory_allocation_graph

class MatchMemoryPlanner:
    def __init__(
        self,
        mem_tensors:List[MatchMemoryTensor],
        available_soc_bytes:int=1024,
        calls_idxs: List=[],
        nodes: List=[],
        out_path: str="output_path",
        algorithm: str="match"
    ):
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
        self.out_path = out_path
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

    def match_mem_planner_impl(self, tensor_fixed_to_ext_mem:List[str]=[]):
        sorted_mem_tensors = sorted([m_t for m_t in self.mem_tensors],
                                    key=lambda m_t:(-(m_t.num_bytes),-m_t.lifetime_span))
        ext_mem_needed = 0
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}
        original_start_usage_tensors = {tensor:tensor.start_usage for tensor in sorted_mem_tensors}
        original_last_usage_tensors = {tensor:tensor.last_usage for tensor in sorted_mem_tensors}

        for tensor in sorted_mem_tensors:
            original_last_usage_tensors[tensor.name] = tensor.last_usage
            original_start_usage_tensors[tensor.name] = tensor.start_usage
            if (tensor.is_input or tensor.is_output) and tensor.name not in tensor_fixed_to_ext_mem:
                tensor.start_usage = 0
                tensor.last_usage = self.last_timestep
        
        print(f"[MEMORY PLANNER] Allocating tensors with {self.available_soc_bytes} bytes of on-chip memory")
        for tensor in sorted_mem_tensors:
            allocate_tensor(
                calls_idxs=self.calls_idxs,
                free_size_at_time=free_size_at_time,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=self.available_soc_bytes,
                tensor_fixed_to_ext_mem = tensor.name in tensor_fixed_to_ext_mem,
                tensor=tensor
            )
        
        print(f"[MEMORY PLANNER] All tensors allocated")
        save_memory_allocation_graph(
            sorted_mem_tensors,
            available_soc_bytes=self.available_soc_bytes,
            output_file=self.out_path+"/memory_plan_stage_1.png"
        )
        # remove constants allocated in the SoC already
        real_constant_tensors = list()
        for tensor in sorted_mem_tensors:
            if tensor.is_constant and not tensor.stored_in_external_memory:
                store_as_constant = True
                for time in free_size_at_time:
                    if time not in tensor.mem_offset_at and free_size_at_time[time]<tensor.num_bytes:
                        store_as_constant = False
                        break
                if store_as_constant:
                    self.available_soc_bytes -= tensor.num_bytes
                    for time in free_size_at_time:
                        if time not in tensor.mem_offset_at:
                            free_size_at_time[time] -= tensor.num_bytes
                    real_constant_tensors.append(tensor.name)
                else:
                    print(f"[MEMORY PLANNER] Constant tensor {tensor.name} will be stored in external memory")
            tensor.mem_offset_at = dict()
            tensor.stored_in_external_memory = False
            tensor.move_temp_to_ext_mem = list()
            tensor.load_from_ext_mem_at = list()
        sorted_mem_tensors = [m_t for m_t in sorted_mem_tensors if m_t.name not in real_constant_tensors]
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}
        
        print(f"[MEM PLANNER] Moved actual constants to on-chip memory, now there are {self.available_soc_bytes} bytes of available on-chip memory")
        # reallocate these intermediate and tensors who may have to stay in SoC memory
        
        for tensor in sorted_mem_tensors:
            allocate_tensor(
                calls_idxs=self.calls_idxs,
                free_size_at_time=free_size_at_time,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=self.available_soc_bytes,
                tensor_fixed_to_ext_mem = tensor.name in tensor_fixed_to_ext_mem,
                tensor=tensor
            )
        
        print("[MEMORY PLANNER] Moved to on-chip memory all possible constants and reallocated other tensors")
        save_memory_allocation_graph(
            sorted_mem_tensors,
            available_soc_bytes=self.available_soc_bytes,
            output_file=self.out_path+"/memory_plan_stage_2.png"
        )
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
            if (tensor.is_input or tensor.is_output) and len(tensor.load_from_ext_mem_at)==0:
                if self.available_soc_bytes<=tensor.num_bytes:
                    self.available_soc_bytes -= tensor.num_bytes
                    real_constant_tensors.append(tensor.name)
            
        sorted_mem_tensors = [m_t for m_t in sorted_mem_tensors if m_t.name not in real_constant_tensors]
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}

        print(f"[MEM PLANNER] Moved on-chip inputs and outputs to on-chip memory, now there are {self.available_soc_bytes} bytes of available on-chip memory")
        # reallocate again
        for tensor in sorted_mem_tensors:
            tensor.stored_in_external_memory = False
            tensor.move_temp_to_ext_mem = list()
            tensor.load_from_ext_mem_at = list()
        
        for tensor in sorted_mem_tensors:
            allocate_tensor(
                calls_idxs=self.calls_idxs,
                free_size_at_time=free_size_at_time,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=self.available_soc_bytes,
                tensor_fixed_to_ext_mem = tensor.name in tensor_fixed_to_ext_mem,
                tensor=tensor
            )

        print("[MEMORY PLANNER] Moved to on-chip memory all possible inputs and outputs and reallocated other tensors")
        # compute external memory needed
        for tensor in sorted_mem_tensors:
            # ext mem for inputs and outputs doesnt count here...
            if ((len(tensor.load_from_ext_mem_at)>0 or len(tensor.move_temp_to_ext_mem)>0) or tensor.stored_in_external_memory) and not (tensor.is_input or tensor.is_output):
                ext_mem_needed+=tensor.num_bytes
            tensor.mem_offset = tensor.mem_offset_at[tensor.used_at[0]] if len(tensor.used_at)>0 else self.available_soc_bytes
            if tensor.is_output or tensor.is_input:
                tensor.start_usage = original_start_usage_tensors[tensor.name]
                tensor.last_usage = original_last_usage_tensors[tensor.name]

        soc_mem_needed = 0
        for time_,tensors in tensors_allocated_at_time.items():
            max_tensor = None if len(tensors)==0 else tensors[-1]
            if max_tensor is not None and (max_tensor.mem_offset_at[time_]+max_tensor.num_bytes)>soc_mem_needed:
                soc_mem_needed = max_tensor.mem_offset_at[time_]+max_tensor.num_bytes
        
        save_memory_allocation_graph(
            sorted_mem_tensors,
            available_soc_bytes=self.available_soc_bytes,
            output_file=self.out_path+"/memory_plan.png"
        )
        return soc_mem_needed, ext_mem_needed

    def generate(self):
        # use only SoC memory if possible, otherwise use external memory
        try:
            if self.algorithm=="match":
                return self.match_mem_planner_impl(
                    tensor_fixed_to_ext_mem=[tensor.name for tensor in self.mem_tensors if tensor.is_output or tensor.is_input]
                )
            else:
                raise Exception(f"[MEMORY PLANNER] Algorithm {self.algorithm} not implemented")
        except Exception as exc:
            print(f"[MEMORY PLANNER] Error during memory planner {exc}")
            raise Exception(f"[MEMORY PLANNER] Error during memory planner {exc}")