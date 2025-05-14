from typing import Dict, List
from match.runtime.graph.tensor import MatchMemoryTensor


def check_if_any_valid_allocation(
    available_soc_bytes: int=0,
    tensor_fixed_to_ext_mem: bool=False,
    tensors_allocated_at_time: Dict={},
    time: int=0,
    tens_size: int=1,
):
    if tens_size>available_soc_bytes:
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
        if available_soc_bytes-end_tens >= tens_size:
            return True, end_tens
    elif len_at_time==1:
        # check if tensor can be allocated before
        tens = tensors_allocated_at_time[time][-1]
        end_tens = tens.mem_offset_at[time] + tens.num_bytes
        if tens.mem_offset_at[time]>=tens_size:
            return True, 0
        elif available_soc_bytes-end_tens >= tens_size:
            return True, end_tens
    else:
        return True, 0
    
    return False, 0

def check_if_valid_allocation(
    available_soc_bytes: int=0,
    tensor_fixed_to_ext_mem: bool=False,
    tensors_allocated_at_time: Dict={},
    time: int=0,
    offset: int=0,
    tens_size: int=1,
):
    if tens_size>available_soc_bytes:
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
        if available_soc_bytes-end_tens >= tens_size:
            if available_soc_bytes>=offset>=end_tens and available_soc_bytes>=offset+tens_size>=end_tens:
                return True
    elif len_at_time==1:
        # check if tensor can be allocated before
        tens = tensors_allocated_at_time[time][-1]
        start_tens = tens.mem_offset_at[time]
        end_tens = tens.mem_offset_at[time] + tens.num_bytes
        if tens.mem_offset_at[time]>=tens_size:
            if offset<=start_tens and offset+tens_size<=start_tens:
                return True
        if available_soc_bytes>=offset>=end_tens and available_soc_bytes>=offset+tens_size>=end_tens:
            return True
    else:
        return True

    return False

def try_to_allocate(
    calls_idxs: List=[],
    tensors_allocated_at_time: Dict={},
    available_soc_bytes: int=0,
    tensor_fixed_to_ext_mem: bool=False,
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
        if time in calls_idxs and time!=allocation_time:
            if not check_if_valid_allocation(
                available_soc_bytes=available_soc_bytes,
                tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                tensors_allocated_at_time=tensors_allocated_at_time,
                time=time,
                offset=offset,
                tens_size=tens_size
            ):
                valid_for_all = False
            if time in tensor.used_at and valid_for_separeted_intermediate:
                if contiguous_area:
                    if not check_if_valid_allocation(
                        available_soc_bytes=available_soc_bytes,
                        tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                        tensors_allocated_at_time=tensors_allocated_at_time,
                        time=time,
                        offset=separeted_last_pt,
                        tens_size=tens_size
                    ):
                        valid_for_separeted_intermediate = False
                    else:
                        allocation_at[time] = separeted_last_pt
                else:
                    found, new_off = check_if_any_valid_allocation(
                        available_soc_bytes=available_soc_bytes,
                        tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                        tensors_allocated_at_time=tensors_allocated_at_time,
                        time=time,
                        tens_size=tens_size
                    ) 
                    if valid_for_separeted_intermediate and found:
                        allocation_at[time] = new_off
                        separeted_last_pt = new_off
                    else:
                        valid_for_separeted_intermediate = False
                contiguous_area = True
            if time not in tensor.used_at:
                contiguous_area = False
    if valid_for_all:
        if tensor_fixed_to_ext_mem:
            tensor.stored_in_external_memory = True
            if tensor.is_constant or tensor.is_input:
                tensor.load_from_ext_mem_at.append(tensor.used_at[0])
            else:
                last_usage = tensor.used_at[-1]
                for call_idx in calls_idxs:
                    if call_idx>last_usage:
                        tensor.move_temp_to_ext_mem.append(call_idx)
                        break
                if len(tensor.move_temp_to_ext_mem)==0:
                    tensor.move_temp_to_ext_mem.append(-1)
        for time in range(tensor.start_usage, tensor.last_usage+1):
            if time in calls_idxs:
                tensor.mem_offset_at[time] = offset
                tensors_allocated_at_time[time] = sorted(tensors_allocated_at_time[time]+[tensor], key=lambda m_t:m_t.mem_offset_at[time])
        allocated=True
    
    elif valid_for_separeted_intermediate and separeted_intermediate_fine:
        in_ext_mem = tensor.is_input or tensor.is_constant
        loaded_first_time = tensor.is_output or tensor.is_intermediate
        tensor.load_from_ext_mem_at = list()
        tensor.move_temp_to_ext_mem = list()
        tensor.stored_in_external_memory = True
        if tensor.is_output:
            print(f"[MEMORY PLANNER] Tensor {tensor.name} is an output tensor")
        for time in range(tensor.start_usage, tensor.last_usage+1):
            if time in calls_idxs:
                if time in tensor.used_at:
                    if not loaded_first_time:
                        loaded_first_time = True
                    if in_ext_mem:
                        tensor.load_from_ext_mem_at.append(time)
                        in_ext_mem = False
                    tensor.mem_offset_at[time] = allocation_at[time]
                    tensors_allocated_at_time[time] = sorted(tensors_allocated_at_time[time]+[tensor], key=lambda m_t:m_t.mem_offset_at[time])
                elif not in_ext_mem:
                    if loaded_first_time and any([time>use_time for use_time in tensor.used_at]):
                        tensor.move_temp_to_ext_mem.append(time)
                    in_ext_mem = True
        if tensor.is_output and tensor_fixed_to_ext_mem:
            last_usage = tensor.used_at[-1]
            if not any([move_time>last_usage for move_time in tensor.move_temp_to_ext_mem]):
                for call_idx in calls_idxs:
                    if call_idx>last_usage:
                        tensor.move_temp_to_ext_mem.append(call_idx)
                        break
                if len(tensor.move_temp_to_ext_mem)==0:
                    tensor.move_temp_to_ext_mem.append(-1)
        allocated=True
        if tensor.is_output:
            print(f"[MEMORY PLANNER] Tensor {tensor.name} is an output tensor, it will be moved to external memory at {tensor.move_temp_to_ext_mem}")
    return allocated

def try_allocate_congested(
    calls_idxs: List=[],
    tensors_allocated_at_time: Dict={},
    available_soc_bytes: int=0,
    tensor_fixed_to_ext_mem: bool=False,
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
            allocated = try_to_allocate(
                calls_idxs=calls_idxs,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=available_soc_bytes,
                tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                tensor=tensor,
                allocation_time=time,
                offset=end_tens_a,
                separeted_intermediate_fine=separeted_intermediate_fine,
                tens_size=tens_size
            )
            if allocated:
                break

    if not allocated:
        # check if theres enough space from the last tensor to the end of the memory
        tens_ = tensors_allocated_at_time[time][-1]
        end_tens = tens_.mem_offset_at[time]+tens_.num_bytes
        while available_soc_bytes-end_tens >= tens_size:
            allocated = try_to_allocate(
                calls_idxs=calls_idxs,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=available_soc_bytes,
                tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                tensor=tensor,
                allocation_time=time,offset=end_tens,
                separeted_intermediate_fine=separeted_intermediate_fine,
                tens_size=tens_size
            )
            if allocated:
                break
            else:
                end_tens+=tens_size
    return allocated

def try_allocate_buffer(
    calls_idxs: List=[],
    tensors_allocated_at_time: Dict={},
    available_soc_bytes: int=0,
    tensor_fixed_to_ext_mem: bool=False,
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
        allocated = try_to_allocate(
            calls_idxs=calls_idxs,
            tensors_allocated_at_time=tensors_allocated_at_time,
            available_soc_bytes=available_soc_bytes,
            tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
            tensor=tensor,
            allocation_time=time,
            offset=0,
            separeted_intermediate_fine=separeted_intermediate_fine,
            tens_size=tens_size
        )
    if not allocated:
        while available_soc_bytes-end_tens >= tens_size:
            allocated = try_to_allocate(
                calls_idxs=calls_idxs,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=available_soc_bytes,
                tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                tensor=tensor,
                allocation_time=time,
                offset=end_tens,
                separeted_intermediate_fine=separeted_intermediate_fine,
                tens_size=tens_size
            )
            if allocated:
                break
            else:
                end_tens+=tens_size
    return allocated

def try_allocate_easy(
    calls_idxs: List=[],
    tensors_allocated_at_time: Dict={},
    available_soc_bytes: int=0,
    tensor_fixed_to_ext_mem: bool=False,
    tensor: MatchMemoryTensor=None,
    separeted_intermediate_fine: bool=False,
    tens_size: int=1, time: int=0
):
    # allocate a single time
    allocated = False
    # check if theres enough space from the last tensor to the end of the memory
    if available_soc_bytes>=tens_size:
        end_tens = 0
        while available_soc_bytes-end_tens >= tens_size:
            allocated = try_to_allocate(
                calls_idxs=calls_idxs,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=available_soc_bytes,
                tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                tensor=tensor,
                allocation_time=time,
                offset=end_tens,
                separeted_intermediate_fine=separeted_intermediate_fine,
                tens_size=tens_size
            )
            if allocated:
                break
            else:
                end_tens+=tens_size
    return allocated

def allocate_tensor(
    calls_idxs: List=[],
    tensors_allocated_at_time: Dict={},
    available_soc_bytes: int=0,
    free_size_at_time: Dict={},
    tensor_fixed_to_ext_mem: bool=False,
    tensor: MatchMemoryTensor=None
):
    tens_size = tensor.num_bytes
    max_allocated_tensors_at = -1
    less_free_mem_at = -1
    for time in range(tensor.start_usage, tensor.last_usage+1):
        if time in calls_idxs and time in tensor.used_at:
            if max_allocated_tensors_at==-1 or len(tensors_allocated_at_time[time])>len(tensors_allocated_at_time[max_allocated_tensors_at]):
                max_allocated_tensors_at = time
            if less_free_mem_at==-1 or free_size_at_time[time]<free_size_at_time[less_free_mem_at]:
                less_free_mem_at = time
    # found the most congested spot
    allocated = False
    if max_allocated_tensors_at!=-1:
        if len(tensors_allocated_at_time[max_allocated_tensors_at])>1:
            allocated = try_allocate_congested(
                calls_idxs=calls_idxs,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=available_soc_bytes,
                tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                tensor=tensor,
                tens_size=tens_size,
                time=max_allocated_tensors_at
            )
            if not allocated:
                allocated = try_allocate_congested(
                    calls_idxs=calls_idxs,
                    tensors_allocated_at_time=tensors_allocated_at_time,
                    available_soc_bytes=available_soc_bytes,
                    tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                    tensor=tensor,
                    separeted_intermediate_fine=True,
                    tens_size=tens_size,
                    time=max_allocated_tensors_at
                )
        
        elif len(tensors_allocated_at_time[max_allocated_tensors_at])==1:
            allocated = try_allocate_buffer(
                calls_idxs=calls_idxs,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=available_soc_bytes,
                tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                tensor=tensor,
                tens_size=tens_size,
                time=max_allocated_tensors_at
            )
            if not allocated:
                allocated = try_allocate_buffer(
                    calls_idxs=calls_idxs,
                    tensors_allocated_at_time=tensors_allocated_at_time,
                    available_soc_bytes=available_soc_bytes,
                    tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                    tensor=tensor,
                    separeted_intermediate_fine=True,
                    tens_size=tens_size,
                    time=max_allocated_tensors_at
                )
        else:
            # try to allocate at 0
            allocated = try_allocate_easy(
                calls_idxs=calls_idxs,
                tensors_allocated_at_time=tensors_allocated_at_time,
                available_soc_bytes=available_soc_bytes,
                tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                tensor=tensor,
                tens_size=tens_size,
                time=max_allocated_tensors_at
            )
            if not allocated:
                allocated = try_allocate_easy(
                    calls_idxs=calls_idxs,
                    tensors_allocated_at_time=tensors_allocated_at_time,
                    available_soc_bytes=available_soc_bytes,
                    tensor_fixed_to_ext_mem=tensor_fixed_to_ext_mem,
                    tensor=tensor,
                    separeted_intermediate_fine=True,
                    tens_size=tens_size,
                    time=max_allocated_tensors_at
                )
    if not allocated:
        print(f"[MEMORY PLANNER] Couldnt allocate all the tensors, tensor {tensor.name} allocation was not successfull")
        print(f"[MEMORY PLANNER] Node at {max_allocated_tensors_at} cannot fit SoC memory")
        # TODO: add list of nodes to run from external memory
        raise Exception(f"[MEMORY PLANNER] Couldnt allocate all the tensors, tensor {tensor.name} with size {tens_size} allocation was not successfull")

    for time_ in tensor.mem_offset_at:
        free_size_at_time[time_] -= tensor.num_bytes