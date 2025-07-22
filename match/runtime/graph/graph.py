from pathlib import Path
from typing import Any, Dict
import numpy as np
from match.runtime.graph.memplan import MatchMemoryPlanner
from match.runtime.graph.tensor import MatchMemoryTensor
from match.target.target import MatchTarget
from match.utils.utils import get_fname_node_schedule
import tvm

class MatchGraphRuntimeNodeCall:
    def __init__(
        self,
        inputs=None,
        outputs=None,fn_name="default_lib_1",
        name="default_lib_1",
        node_info={},
        node_id: int=0,
        node_name: str=None,
        schedule=None,
        match_node=None,
        dtype_output_node='int8',
    ):
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
        self.dtype_output_node = dtype_output_node
        self.free_buffers = []
    


class MatchTVMGraphRuntime:
    def __init__(
        self,
        target: MatchTarget,
        mod_info: Dict[str,Any],
        params=None,
        model_name: str="default",
        out_path: str="model_out",
        match_inputs=None,
        host_module=None
    ):
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
        self.metadata = dict()
        self.dev = tvm.cpu(0)
        self.fallback_kernel_extra_dynamic_mem = dict()
        self.max_extra_dynamic_mem = 0

    def parse_host_lib_for_extra_dynamic_mem(self, host_lib_path: str="match/runtime/graph/host_lib.c"):
        func_name = ""
        with open(host_lib_path, "r") as f:
            for line in f:
                if "TVM_DLL int32_t" in line and line[-2]=="{":
                    # this is a function that allocates dynamic memory
                    # get the function name
                    func_name = line.strip().split()[2].split("(")[0]
                if func_name!="" and "#ifdef __cplusplus" in line:
                    func_name = ""
                if func_name!="" and "TVMBackendAllocWorkspace" in line:
                    splitted_line = line.strip().split()
                    buffer_name = splitted_line[1]
                    buffer_size = int(splitted_line[5].split(")")[1][:-1])
                    if func_name not in self.fallback_kernel_extra_dynamic_mem:
                        self.fallback_kernel_extra_dynamic_mem[func_name] = {"total": 0, "buffers": []}
                    self.fallback_kernel_extra_dynamic_mem[func_name]["buffers"].append((buffer_name, buffer_size))
                    self.fallback_kernel_extra_dynamic_mem[func_name]["total"] += buffer_size
        self.max_extra_dynamic_mem = max(
            [self.fallback_kernel_extra_dynamic_mem[func_name]["total"] for func_name in self.fallback_kernel_extra_dynamic_mem]
        )
        print(f"[MEM PLANNER] Found {len(self.fallback_kernel_extra_dynamic_mem)} functions with extra dynamic memory allocations")
        print(f"[MEM PLANNER] Maximum extra dynamic memory needed: {self.max_extra_dynamic_mem} bytes")

    def generate(self):
        tensor_map = {}
        out_tensor_map = {}
        nodes_map = {}
        map_names = dict()
        mem_tensors = []
        nodes = []
        dtypes = self.mod_info["attrs"]["dltype"][1]
        shapes = self.mod_info["attrs"]["shape"][1]
        storage_ids = self.mod_info["attrs"]["storage_id"][1]
        heads = [head[0] for head in self.mod_info["heads"]]    # list of the output nodes
        nop_maps = dict()
        activations = dict()
        dtype_activations = dict()
        storage_ids_size = dict()
        for i, storage_id in enumerate(storage_ids):
            tensor_size = np.prod(shapes[i]) * np.dtype(dtypes[i]).itemsize
            if storage_id not in storage_ids_size or tensor_size > storage_ids_size[storage_id]:
                storage_ids_size[storage_id] = tensor_size
        extra_dynamic_buffer_id = 0
        extra_dynamic_buffers = []
        for match_inp in self.match_inputs.values():
            activations[match_inp["name"]] = match_inp["np_values"]
            dtype_activations[match_inp["name"]] = match_inp["np_values"].dtype
        #iterate over the nodes
        for node_id,node in enumerate(self.mod_info["nodes"]):
            if node["op"]=="null":
                # input or parameter
                # param
                if node["name"] in self.params:
                    param = self.params[node["name"]]
                    # store this into the weights to store
                    const_val = param.numpy()
                    mem_tensor = MatchMemoryTensor(
                        name=node["name"],
                        is_constant=True,
                        is_output=node_id in heads,
                        constant_val=const_val,
                        original_constant_val=const_val,
                        shape=param.shape,
                        dtype=np.dtype(param.dtype),
                        node_id=node_id,
                        node_info=node,
                        tvm_memplan_storage_id=storage_ids[node_id]
                    )
                    mem_tensors.append(mem_tensor)
                    tensor_map[node["name"]] = mem_tensor
                    out_tensor_map[(node_id,0)] = mem_tensor
                    map_names[node["name"]] = (mem_tensor.name, mem_tensor.name, mem_tensor.name)
                else:
                    mem_tensor = MatchMemoryTensor(
                        name=node["name"],is_input=True,
                        is_output=node_id in heads,
                        shape=tuple(shapes[node_id]),
                        dtype=np.dtype(dtypes[node_id]),
                        node_id=node_id, node_info=node,
                        tvm_memplan_storage_id=storage_ids[node_id]
                    )
                    mem_tensors.append(mem_tensor)
                    tensor_map[node["name"]] = mem_tensor
                    out_tensor_map[(node_id,0)] = mem_tensor
                    map_names[node["name"]] = (mem_tensor.name, mem_tensor.name, mem_tensor.name)
                    if node_id in heads:
                        mem_tensor_out = MatchMemoryTensor(
                            name=node["name"]+"_out",is_input=False,
                            is_output=node_id in heads,
                            shape=tuple(shapes[node_id]),dtype=np.dtype(dtypes[node_id]),
                            node_id=-1, node_info=node,
                            tvm_memplan_storage_id=storage_ids[node_id]
                        )
                        mem_tensors.append(mem_tensor_out)
                        out_tensor_map[(node_id,0)] = mem_tensor_out
            else:
                inputs = []
                for inp_node_idx in [inp_node_idxs[0] for inp_node_idxs in node["inputs"]]:
                    if self.mod_info["nodes"][inp_node_idx]["op"]!="null" and "_nop" in self.mod_info["nodes"][inp_node_idx]["name"]\
                        and self.mod_info["nodes"][inp_node_idx]["name"]+'_'+str(inp_node_idx) in nop_maps:
                        # inputs is a nop skip
                        inputs.append(nop_maps[self.mod_info["nodes"][inp_node_idx]["name"]+'_'+str(inp_node_idx)])
                    else:
                        name_tens = self.mod_info["nodes"][inp_node_idx]["name"]
                        if self.mod_info["nodes"][inp_node_idx]["op"]!="null":
                            name_tens = name_tens+"_out"
                        inputs.append(tensor_map[name_tens])
                if "_nop" in node["name"]:
                    if len(inputs)==1:
                        nop_maps[node["name"]+'_'+str(node_id)] = inputs[0]
                        input_ptr = inputs[0]
                        # give also the right name to the tensor
                        id_out = -1
                        tens_name = None
                        for head_idx,head in enumerate(heads):
                            if head==node_id:
                                id_out = head_idx
                                tens_name = self.model_name+"_out_"+str(head_idx)
                                break
                        if id_out>=0:
                            if out_tensor_map[(input_ptr.node_id,0)].is_output:
                                name_out = self.model_name+"_out_"+str(id_out)
                                mem_tensor = MatchMemoryTensor(
                                    name=name_out,
                                    is_input=False,
                                    is_output=True,
                                    is_intermediate=False,
                                    shape=tuple(shapes[node_id]),
                                    dtype=np.dtype(dtypes[node_id]),
                                    node_id=-1,
                                    node_info=node,
                                    tvm_memplan_storage_id=storage_ids[node_id]
                                )
                                out_tensor_map[(node_id,0)] = mem_tensor
                                prev_tens_name = input_ptr.name
                                activations[name_out] = activations[prev_tens_name]
                                dtype_activations[name_out] = dtype_activations[prev_tens_name]
                                map_names[name_out] = map_names[prev_tens_name]
                            else:
                                input_ptr.is_output = True
                                mem_tensor = out_tensor_map[(input_ptr.node_id,0)]
                                del out_tensor_map[(input_ptr.node_id,0)]
                                out_tensor_map[(node_id,0)] = mem_tensor
                                #rename tensor name and activation name
                                prev_tens_name = input_ptr.name
                                input_ptr.name = tens_name
                                activations[input_ptr.name] = activations[prev_tens_name]
                                del activations[prev_tens_name]
                                dtype_activations[input_ptr.name] = dtype_activations[prev_tens_name]
                                del dtype_activations[prev_tens_name]
                                map_names[input_ptr.name] = map_names[prev_tens_name]
                                del map_names[prev_tens_name]
                            cnt_ = 0
                            for head_idx,head in [(head_idx, head) for head_idx, head in enumerate(heads)][id_out+1:]:
                                if head==node_id:
                                    cnt_ += 1
                                    out_tensor_map[(node_id,cnt_)] = MatchMemoryTensor(
                                        name=self.model_name+"_out_"+str(head_idx),
                                        is_input=False,
                                        is_output=True,
                                        is_intermediate=False,
                                        shape=tuple(shapes[node_id]),dtype=np.dtype(dtypes[node_id]),
                                        node_id=-1,
                                        node_info=node,
                                        tvm_memplan_storage_id=storage_ids[node_id]
                                    )
                    continue
                
                match_node, schedule, match_node_name = (None, None, None)
                host_lib = None
                if "match" in node["name"]:
                    match_node, schedule, match_node_name, cpu_only_c_lib, host_lib = get_fname_node_schedule(node["name"])
                    if match_node is not None and schedule is not None:
                        for w_tensor in match_node.const_tensors.values():
                            if w_tensor.name in schedule.tensors:
                                w_tensor = schedule.tensors[w_tensor.name]
                                mem_tensor = MatchMemoryTensor(
                                    name=match_node_name+"_"+w_tensor.name,
                                    is_constant=True,
                                    constant_val=w_tensor.data,
                                    original_constant_val=w_tensor.original_data,
                                    shape=w_tensor.data.shape,
                                    dtype=w_tensor.dtype,node_id=node_id,
                                    node_info=node,
                                    tvm_memplan_storage_id=storage_ids[node_id]
                                )
                                mem_tensors.append(mem_tensor)
                                tensor_map[w_tensor.name] = mem_tensor
                                out_tensor_map[(node_id,0)] = mem_tensor
                                inputs.append(mem_tensor)
                # inputs = [mem_tensors[inp_node_idx] for inp_node_idx in [inp_node_idxs[0] for inp_node_idxs in node["inputs"]]]
                for inp in inputs:
                    if "match" not in node["name"]:
                        inp.used_by_tvm = True
                    # every node that is input to another node and it is not an graph input is also an intermediate
                    if not inp.is_input and not inp.is_constant:
                        inp.is_intermediate = True
                    inp.update_last_usage(node_id)
                id_out = -1
                tens_name = self.model_name+"_node_"+str(node_id)+"_out"
                for head_idx,head in enumerate(heads):
                    if head==node_id:
                        id_out = head_idx
                        tens_name = self.model_name+"_out_"+str(id_out)
                        break
                mem_tensor = MatchMemoryTensor(
                    name=tens_name,is_output=id_out!=-1,
                    is_intermediate=id_out==-1,
                    shape=tuple(shapes[node_id]),dtype=np.dtype(dtypes[node_id]),
                    node_id=node_id,
                    node_info=node,
                    tvm_memplan_storage_id=storage_ids[node_id]
                )
                out_tensor_map[(node_id,0)] = mem_tensor
                if mem_tensor.is_output:
                    cnt_ = 0
                    for head_idx,head in [(head_idx, head) for head_idx, head in enumerate(heads)][id_out+1:]:
                        if head==node_id:
                            cnt_ += 1
                            out_tensor_map[(node_id,cnt_)] = MatchMemoryTensor(
                                name=self.model_name+"_out_"+str(head_idx),
                                is_input=False,
                                is_output=True,
                                is_intermediate=False,
                                shape=tuple(shapes[node_id]),dtype=np.dtype(dtypes[node_id]),
                                node_id=-1,
                                node_info=node,
                                tvm_memplan_storage_id=storage_ids[node_id]
                            )
                mem_tensor.update_last_usage(node_id)
                # get the activations values for debugging purposes
                node_activations = list()
                for tens_inp in inputs:
                    if tens_inp.is_input or tens_inp.is_intermediate: # or (len(tens_inp.used_at)>0 and tens_inp.is_output):
                        node_activations.append(tvm.nd.array(activations[tens_inp.name]))
                    elif tens_inp.is_constant:
                        node_activations.append(tvm.nd.array(tens_inp.original_constant_val))
                if "match" not in node["name"]:
                    # run the fallback function to get the debugging values
                    mem_tensor.used_by_tvm = True
                    output_nd = tvm.nd.empty(shape=mem_tensor.shape, dtype=mem_tensor.dtype)
                    self.host_module[node["attrs"]["func_name"]](*node_activations, output_nd)
                    activations[mem_tensor.name] = output_nd.numpy()
                else:
                    # run the CPU version of the MATCH-node to get the debugging values
                    module = tvm.contrib.graph_executor.GraphModule(host_lib["default"](self.dev))
                    for tens_inp, param in zip(node_activations, host_lib.ir_mod["main"].params):
                        module.set_input(param.name_hint, tens_inp)
                    module.run()
                    output_np = module.get_output(0).numpy()
                    activations[mem_tensor.name] = output_np
                
                # type of the activation output
                dtype_activations[mem_tensor.name] = mem_tensor.dtype
                mem_tensors.append(mem_tensor)
                tensor_map[node["name"]+"_out"] = mem_tensor
                outputs = [mem_tensor]
                call_node = MatchGraphRuntimeNodeCall(
                    inputs=inputs, outputs=outputs,
                    name=self.model_name+"_node_"+str(node_id),
                    fn_name=node["attrs"]["func_name"], node_info=node,
                    node_id=node_id, node_name=match_node_name,
                    schedule=schedule, match_node=match_node,
                    dtype_output_node=dtypes[node_id]
                )
                nodes.append(call_node)
                nodes_map[node["name"]] = call_node
                map_names[tens_name] = (call_node.name, node["name"]+"_out", node["name"])

                if "match" not in node["name"] and call_node.fn_name in self.fallback_kernel_extra_dynamic_mem:
                    for buffer_name, buffer_size in self.fallback_kernel_extra_dynamic_mem[call_node.fn_name]["buffers"]:
                        mem_tensor_extra = MatchMemoryTensor(
                            name=f"TVM_EXTRA_DYNAMIC_BUFFER_{extra_dynamic_buffer_id}_{buffer_name}",
                            is_intermediate=True,
                            is_extra_dynamic=True,
                            extra_dynamic_buffer_id=extra_dynamic_buffer_id,
                            shape=(buffer_size,),
                            dtype=np.dtype("uint8"),
                            node_id=node_id,
                        )
                        extra_dynamic_buffers.append(mem_tensor_extra)
                        mem_tensor_extra.update_last_usage(node_id)
                        extra_dynamic_buffer_id += 1
        
        # set memory planner and run it
        self.mem_planner = MatchMemoryPlanner(
            mem_tensors=mem_tensors,
            extra_dynamic_buffers=extra_dynamic_buffers,
            available_soc_bytes=self.target.soc_memory_bytes,
            max_extra_dynamic_mem=self.max_extra_dynamic_mem,
            fallback_kernel_extra_dynamic_mem=self.fallback_kernel_extra_dynamic_mem,
            calls_idxs=[node.node_id for node in nodes],
            nodes=nodes,
            out_path=self.out_path,
            algorithm="match"
        )
        self.mem_needed_bytes, self.ext_mem_needed_bytes = self.mem_planner.generate()
        inputs = [tens for tens in mem_tensors if tens.is_input]
        outputs = list()
        found_outputs_cnt = dict()
        for head in heads:
            if head not in found_outputs_cnt:
                found_outputs_cnt[head] = 0
            else:
                found_outputs_cnt[head] += 1
            outputs.append(out_tensor_map[(head, found_outputs_cnt[head])])
        if not Path(self.out_path+"/parameters").absolute().is_dir():
            Path(self.out_path+"/parameters").absolute().mkdir()
        if not Path(self.out_path+"/golden").absolute().is_dir():
            Path(self.out_path+"/golden").absolute().mkdir()
        for mem_tensor in mem_tensors:
            if mem_tensor.stored_in_external_memory and mem_tensor.is_constant:
                np.frombuffer(mem_tensor.constant_val.flatten().tobytes(),dtype="uint8").tofile(Path(self.out_path+f"/parameters/{self.model_name}_{mem_tensor.name}_data.hex"))
            elif mem_tensor.stored_in_external_memory and mem_tensor.is_input:
                np.frombuffer(activations[mem_tensor.name].flatten().tobytes(),dtype="uint8").tofile(Path(self.out_path+f"/parameters/{self.model_name}_{mem_tensor.name}_data.hex"))
        for activation_name, activation in activations.items():
            mem_tensor_ = None
            for m_t in mem_tensors:
                if m_t.name==activation_name:
                    if not m_t.is_input:
                        mem_tensor_ = m_t
                    break
            if mem_tensor_ is not None:
                np.frombuffer(activation.flatten().tobytes(),dtype=mem_tensor_.dtype).tofile(Path(self.out_path+f"/golden/{self.model_name}_{activation_name}_data.hex"))
        
        # compute the checksums in the correct data type            
        checksums = {}
        for activation_name, activation in activations.items():
            # print(activation_name, dtype_activations[activation_name])
            if np.issubdtype(dtype_activations[activation_name], np.floating):
                checksums[activation_name] = np.frombuffer(activation.flatten(), dtype="float32").astype(np.float64).sum()
            else: # FIXME 
                checksums[activation_name] = np.frombuffer(activation.flatten().tobytes(),dtype="uint8").sum()
        
        # template data for code generation
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
            "checksums": checksums,
            "tvm_memplan_storage_ids_size": storage_ids_size,
            "total_tvm_memplan_storage_size": sum(storage_ids_size.values())
        }
        return template_data
