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
        match_node=None
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
                    const_val = param.numpy()
                    mem_tensor = MatchMemoryTensor(
                        name=node["name"],
                        is_constant=True,
                        constant_val=const_val,
                        original_constant_val=const_val,
                        shape=param.shape,
                        dtype=np.dtype(param.dtype),
                        node_id=node_id,
                        node_info=node
                    )
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
                                mem_tensor = MatchMemoryTensor(
                                    name=match_node_name+"_"+w_tensor.name,
                                    is_constant=True,
                                    constant_val=w_tensor.data,
                                    original_constant_val=w_tensor.original_data,
                                    shape=w_tensor.data.shape,
                                    dtype=w_tensor.dtype,node_id=node_id,
                                    node_info=node
                                )
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
                # get the activations values for debugging purposes
                node_activations = list()
                for tens_inp in inputs:
                    if tens_inp.is_input or tens_inp.is_intermediate:
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
        self.mem_planner = MatchMemoryPlanner(
            mem_tensors=mem_tensors,
            available_soc_bytes=self.target.soc_memory_bytes,
            calls_idxs=[node.node_id for node in nodes],
            nodes=nodes,
            out_path=self.out_path,
            algorithm="match"
        )
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
