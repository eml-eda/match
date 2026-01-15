
import json
from typing import Any, List, Dict
from match.runtime.graph.tensor import MatchMemoryTensor
from match.runtime.graph.alloc import allocate_tensor
from match.runtime.graph.utils import save_memory_allocation_graph, save_memory_allocation_graph_nodes_buffers

class MatchMemoryPlanner:
    def __init__(
        self,
        mem_tensors:List[MatchMemoryTensor]=[],
        extra_dynamic_buffers:List[MatchMemoryTensor]=[],
        max_extra_dynamic_mem:int=0,
        fallback_kernel_extra_dynamic_mem:Dict[str, Any] = {},
        available_soc_bytes:int=1024,
        calls_idxs: List=[],
        nodes: List=[],
        out_path: str="output_path",
        algorithm: str="match",
        fix_io_tensors_in_ext_mem: bool=True
    ):
        self.mem_tensors = mem_tensors
        self.extra_dynamic_buffers = extra_dynamic_buffers
        self.max_extra_dynamic_mem = max_extra_dynamic_mem
        self.fallback_kernel_extra_dynamic_mem = fallback_kernel_extra_dynamic_mem
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
        self.on_chip_constants = []
        self.on_chip_io = []
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
        self.fix_io_tensors_in_ext_mem = fix_io_tensors_in_ext_mem
        
    @property
    def external_memory_needed(self):
        return self.total_memory_needed_bytes > self.available_soc_bytes

    def match_mem_planner_impl(self, tensor_fixed_to_ext_mem:List[str]=[]):
        sorted_mem_tensors = sorted(
            [m_t for m_t in self.mem_tensors+self.extra_dynamic_buffers if m_t.lifetime!=(-1,-1)],
            key=lambda m_t:(-(m_t.num_bytes),-m_t.lifetime_span)
        )
        ext_mem_needed = 0
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}
        original_start_usage_tensors = {tensor.name:tensor.start_usage for tensor in sorted_mem_tensors}
        original_last_usage_tensors = {tensor.name:tensor.last_usage for tensor in sorted_mem_tensors}

        def set_io_lifetime_as_inf(tensor: MatchMemoryTensor):
            tensor.start_usage = 0
            tensor.last_usage = self.last_timestep

        on_chip_mem_needed_w_io_and_consts_off_chip = 0
        tmp_tensors_fixed_to_ext_mem = [tensor.name for tensor in sorted_mem_tensors if (tensor.name in tensor_fixed_to_ext_mem) or tensor.is_constant or tensor.is_input or tensor.is_output]
        for tensor in sorted_mem_tensors:
            allocate_tensor(
                calls_idxs=self.calls_idxs,
                free_size_at_time=free_size_at_time,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=self.available_soc_bytes,
                tensor_fixed_to_ext_mem = tensor.name in tmp_tensors_fixed_to_ext_mem,
                tensor=tensor
            )

        for time_,tensors in tensors_allocated_at_time.items():
            max_tensor = None if len(tensors)==0 else tensors[-1]
            if max_tensor is not None and (max_tensor.mem_offset_at[time_]+max_tensor.num_bytes)>on_chip_mem_needed_w_io_and_consts_off_chip:
                on_chip_mem_needed_w_io_and_consts_off_chip = max_tensor.mem_offset_at[time_]+max_tensor.num_bytes
        # reset everything and now try to allocate everything again
        for tensor in sorted_mem_tensors:
            tensor.reset()
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}

        print(f"[MEM PLANNER] Allocating tensors with {self.available_soc_bytes} bytes of on-chip memory")
        for tensor in sorted_mem_tensors:
            allocate_tensor(
                calls_idxs=self.calls_idxs,
                free_size_at_time=free_size_at_time,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=self.available_soc_bytes,
                tensor_fixed_to_ext_mem = tensor.name in tensor_fixed_to_ext_mem,
                tensor=tensor
            )
        
        print(f"[MEM PLANNER] All tensors allocated")
        save_memory_allocation_graph(
            sorted_mem_tensors,
            graph_output_file=self.out_path+"/memory_plan_stage_1.png",
            metadata_output_file=self.out_path+"/memory_plan_stage_1_metadata.json"
        )
        # remove constants allocated in the SoC already
        real_constant_tensors = list()
        for tensor in sorted_mem_tensors:
            if tensor.is_constant and not tensor.stored_in_external_memory:
                if self.available_soc_bytes - tensor.num_bytes >= on_chip_mem_needed_w_io_and_consts_off_chip:
                    self.available_soc_bytes -= tensor.num_bytes
                    real_constant_tensors.append(tensor.name)
                    self.on_chip_constants.append(tensor)
                else:
                    tensor_fixed_to_ext_mem.append(tensor.name)
                    print(f"[MEM PLANNER] Constant tensor {tensor.name} will be stored in external memory")
            tensor.reset()
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
            graph_output_file=self.out_path+"/memory_plan_stage_2.png",
            metadata_output_file=self.out_path+"/memory_plan_stage_2_metadata.json"
        )
        # now check if inputs and outputs can stay always in SoC memory
        
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}
        # remove constants allocated in the SoC already
        real_constant_tensors = list()

        for tensor in sorted_mem_tensors:
            if (tensor.is_input or tensor.is_output) and len(tensor.load_from_ext_mem_at)==0 and tensor.name not in tensor_fixed_to_ext_mem:
                # if self.available_soc_bytes - tensor.num_bytes >= on_chip_mem_needed_w_io_and_consts_off_chip:
                if tensor.is_input:
                    self.available_soc_bytes -= tensor.num_bytes
                real_constant_tensors.append(tensor.name)
                self.on_chip_io.append(tensor)
                set_io_lifetime_as_inf(tensor)
                print(f"[MEM PLANNER] Input/Output tensor {tensor.name} will be stored in on-chip memory")
            
        sorted_mem_tensors = [m_t for m_t in sorted_mem_tensors if m_t.name not in real_constant_tensors]
        tensors_allocated_at_time = {key:[] for key in self.calls_idxs}
        free_size_at_time = {key:self.available_soc_bytes for key in self.calls_idxs}
        print(f"[MEM PLANNER] Moved on-chip inputs and outputs to on-chip memory, now there are {self.available_soc_bytes} bytes of available on-chip memory")
        # reallocate again
        for tensor in sorted_mem_tensors:
            tensor.reset()
        
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
        io_ext_mem_needed = 0
        inp_ext_mem_needed = 0
        constant_ext_mem_needed = 0
        # compute external memory needed
        for tensor in sorted_mem_tensors:
            # ext mem for inputs and outputs doesnt count here...
            if ((len(tensor.load_from_ext_mem_at)>0 or len(tensor.move_temp_to_ext_mem)>0) or tensor.stored_in_external_memory) and not (tensor.is_input or tensor.is_output):
                ext_mem_needed+=tensor.num_bytes
                if tensor.is_constant:
                    constant_ext_mem_needed += tensor.num_bytes
            tensor.mem_offset = tensor.mem_offset_at[tensor.used_at[0]] if len(tensor.used_at)>0 else self.available_soc_bytes
            if tensor.is_output or tensor.is_input:
                if ((len(tensor.load_from_ext_mem_at)>0 or len(tensor.move_temp_to_ext_mem)>0) or tensor.stored_in_external_memory):
                    io_ext_mem_needed += tensor.num_bytes
                    if tensor.is_input:
                        inp_ext_mem_needed += tensor.num_bytes
                tensor.start_usage = original_start_usage_tensors[tensor.name]
                tensor.last_usage = original_last_usage_tensors[tensor.name]

        soc_mem_needed = 0
        for time_,tensors in tensors_allocated_at_time.items():
            max_tensor = None if len(tensors)==0 else tensors[-1]
            if max_tensor is not None and (max_tensor.mem_offset_at[time_]+max_tensor.num_bytes)>soc_mem_needed:
                soc_mem_needed = max_tensor.mem_offset_at[time_]+max_tensor.num_bytes

        save_memory_allocation_graph(
            sorted_mem_tensors,
            graph_output_file=self.out_path+"/memory_plan.png",
            metadata_output_file=self.out_path+"/memory_plan_metadata.json"
        )
        io_on_chip_needed = sum([tensor.num_bytes for tensor in self.on_chip_io])
        constants_on_chip_needed = sum([tensor.num_bytes for tensor in self.on_chip_constants])
        metadata_soc_mem_ext_mem = {
            "total_on_chip": soc_mem_needed + constants_on_chip_needed + io_on_chip_needed,
            "static_on_chip": constants_on_chip_needed + io_on_chip_needed,
            "constants_on_chip": constants_on_chip_needed,
            "io_on_chip": io_on_chip_needed,
            "dynamic_on_chip": soc_mem_needed,
            "tvm_buffers_extra_dynamic": self.max_extra_dynamic_mem,
            "total_off_chip": ext_mem_needed + io_ext_mem_needed,
            "dynamic_off_chip": ext_mem_needed,
            "io_off_chip": io_ext_mem_needed,
            "inp_off_chip_files": inp_ext_mem_needed,
            "constants_off_chip": constant_ext_mem_needed,
        }
        json.dump(metadata_soc_mem_ext_mem, open(self.out_path+"/memory_plan_on_off_chip_summary.json", "w"), indent=4)
        nodes_buffers = save_memory_allocation_graph_nodes_buffers(
            mem_tensors_at=tensors_allocated_at_time,
            calls_idxs=self.calls_idxs,
            match_mem_size=soc_mem_needed,
            output_file=self.out_path+"/memory_plan_node_buffers.json"
        )
        for node in self.nodes:
            node.free_buffers = nodes_buffers[node.node_id]["empty_areas"]
        return soc_mem_needed, ext_mem_needed

    def generate(self):
        # use only SoC memory if possible, otherwise use external memory
        try:
            if self.algorithm=="match":
                return self.match_mem_planner_impl(
                    # tensor_fixed_to_ext_mem=[tensor.name for tensor in self.mem_tensors if\
                    #                          tensor.is_output\
                    #                             or tensor.is_input
                    # ]
                )
            else:
                raise Exception(f"[MEM PLANNER] Algorithm {self.algorithm} not implemented")
        except Exception as exc:
            print(f"[MEM PLANNER] Error during memory planner {exc}")
            raise Exception(f"[MEM PLANNER] Error during memory planner {exc}")