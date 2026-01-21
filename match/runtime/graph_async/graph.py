from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tvm

from match.target.target import MatchTarget
from match.utils.utils import get_fname_node_schedule

from .node import MatchGraphRuntimeNodeCall
from .tensor import MatchMemoryTensor
from .memplan import MatchMemoryPlanner


class MatchTVMGraphRuntime:
    """
    Aync graph runtime class for MATCH models.
    Handles runtime graph generation from TVM graph, memory planning and activation extraction for debug.
    For each node, we keep track of dependencies and device allocation.
    A topological order is generated.
    At runtime we can execute nodes in parallel, respecting dependencies.
    """
    def __init__(
        self,
        target: MatchTarget,
        mod_info: Dict[str, Any],
        params: Optional[dict] = None,
        model_name: str = "default",
        out_path: str = "model_out",
        match_inputs: Optional[dict] = None,
        host_module: Optional[dict] = None,
    ):
        self.target = target
        self.mod_info = mod_info
        self.params = params or {}
        self.model_name = model_name
        self.out_path = out_path
        self.mem_planner = None
        self.ext_mem_needed_bytes = 0
        self.mem_needed_bytes = 0
        self.match_inputs = match_inputs or {}
        self.host_module = host_module or {}
        self.dev = tvm.cpu(0)

    def generate(self) -> dict:
        """
        Generate tensors, nodes and memory plan and scheduling for the runtime graph.
        """
        mem_tensors = [] # List of MatchMemoryTensor objects
        tensor_name_to_id = {} # Maps tensor names to their IDs
        tensor_map = {} # Maps tensor names to MatchMemoryTensor objects
        idx_tensor_map = {} # Maps (node_id, output_index) to MatchMemoryTensor objects
        map_names = {} # Maps tensor names to (call_name, out_name, node_name)
        
        
        nodes_map = {} # Maps node names to MatchGraphRuntimeNodeCall objects
        nodes_ids = {} # Maps node names to node IDs
        nodes = [] # List of MatchGraphRuntimeNodeCall objects
        
        dtypes = self.mod_info["attrs"]["dltype"][1]
        shapes = self.mod_info["attrs"]["shape"][1]
        heads = [head[0] for head in self.mod_info["heads"]] 
        
        nop_maps = {}
        activations = {inp["name"]: inp["np_values"] for inp in self.match_inputs.values()}

        # Pass 1: Create memory tensors for inputs and parameters
        for node_id, node in enumerate(self.mod_info["nodes"]):
            if node["op"] == "null":
                name = node["name"]
                is_output = node_id in heads
                dtype = np.dtype(dtypes[node_id])
                shape = tuple(shapes[node_id])
                
                
                if name in self.params:
                    # Parameter node: weight tensor
                    param = self.params[name]
                    const_val = param.numpy()
                    mem_tensor = MatchMemoryTensor(
                        id = len(mem_tensors),
                        name = name,
                        is_constant = True,
                        is_output = is_output,
                        constant_val = const_val,
                        original_constant_val = const_val,
                        shape = param.shape,
                        dtype = np.dtype(param.dtype),
                        node_id = node_id,
                        node_info = node,
                    )
                else:
                    # Input node: variable tensor
                    mem_tensor = MatchMemoryTensor(
                        id = len(mem_tensors),
                        name = name,
                        is_input = True,
                        is_output = is_output,
                        shape = shape,
                        dtype = dtype,
                        node_id = node_id,
                        node_info = node,
                    )
                    
                mem_tensors.append(mem_tensor)
                tensor_name_to_id[name] = mem_tensor.id
                tensor_map[name] = mem_tensor
                idx_tensor_map[(node_id, 0)] = mem_tensor
                map_names[name] = (name, name, name)
                
                # For output input nodes, create an explicit output memory tensor too
                if name not in self.params and is_output:
                    mem_tensor_out = MatchMemoryTensor(
                        name = f"{name}_out",
                        is_input = False,
                        is_output = True,
                        shape = shape,
                        dtype = dtype,
                        node_id = -1,
                        node_info = node,
                    )
                    mem_tensors.append(mem_tensor_out)
                    idx_tensor_map[(node_id, 0)] = mem_tensor_out
                    tensor_name_to_id[mem_tensor_out.name] = mem_tensor_out.id
        
        # Pass 2: Create compute and intermediate tensors, nodes
        for node_id, node in enumerate(self.mod_info["nodes"]):
            if node["op"] == "null":
                continue
            
            input_tensors = []
            
            # Build input list for this node
            for inp_idxs in node["inputs"]:
                inp_node = self.mod_info["nodes"][inp_idxs[0]]
                inp_name = inp_node["name"]
                inp_tensor_name = (inp_name + "_out") if inp_node["op"] != "null" else inp_name
                
                if inp_node["op"] != "null" and "_nop" in inp_name and inp_name in nop_maps:
                    # NOP input node, bypass to NOP input node input
                    input_tensors.append(nop_maps[inp_name])
                else:
                    input_tensors.append(tensor_map[inp_tensor_name])
                    
            # NOP nodes: map and skip
            if "_nop" in node["name"]:
                if len(input_tensors) == 1:
                    nop_maps[node["name"]] = input_tensors[0]
                continue
            
            # MATCH nodes: get schedule and add constant tensors
            match_node = schedule = match_node_name = host_lib = None
            if "match" in node["name"]:
                match_node, schedule, match_node_name, _, host_lib = get_fname_node_schedule(node["name"])
                if match_node and schedule:
                    for w_tensor in match_node.const_tensors.values():
                        if w_tensor.name in schedule.tensors:
                            s_tensor = schedule.tensors[w_tensor.name]
                            mem_tensor = MatchMemoryTensor(
                                id = len(mem_tensors),
                                name = f"{match_node_name}_{s_tensor.name}",
                                is_constant = True,
                                constant_val = s_tensor.data,
                                original_constant_val = s_tensor.original_data,
                                shape = s_tensor.data.shape,
                                dtype = s_tensor.dtype,
                                node_id = node_id,
                                node_info = node,
                            )
                            mem_tensors.append(mem_tensor)
                            tensor_map[s_tensor.name] = mem_tensor
                            idx_tensor_map[(node_id, 0)] = mem_tensor
                            input_tensors.append(mem_tensor)
                            tensor_name_to_id[mem_tensor.name] = mem_tensor.id

            # Mark tensor usage info for memory planning
            for inp in input_tensors:
                if "match" not in node["name"]:
                    inp.used_by_tvm = True
                if inp.is_output:
                    inp.is_intermediate = True
                #inp.update_last_usage(node_id)
                inp.update_last_usage(len(self.mod_info["nodes"]) - 1)
            
            # Build output tensor for this node
            id_out = -1
            tens_name = f"{self.model_name}_node_{node_id}_out"
            for head_idx, head in enumerate(heads):
                if head == node_id:
                    id_out = head_idx
                    tens_name = f"{self.model_name}_out_{id_out}"
                    break
                
            is_output = (id_out != -1)
            is_intermediate = (id_out == -1)
                
            mem_tensor = MatchMemoryTensor(
                id = len(mem_tensors),
                name = tens_name,
                is_output = is_output,
                is_intermediate = is_intermediate,
                shape = tuple(shapes[node_id]),
                dtype = np.dtype(dtypes[node_id]),
                node_id = node_id,
            )
            
            idx_tensor_map[(node_id, 0)] = mem_tensor
            if is_output:
                cnt = 0
                for head_idx, head in enumerate(heads[id_out + 1:], id_out + 1):
                    if head == node_id:
                        cnt += 1
                        idx_tensor_map[(node_id, cnt)] = MatchMemoryTensor(
                            name = f"{self.model_name}_out_{head_idx}",
                            is_input = False,
                            is_output = True,
                            is_intermediate = False,
                            shape = tuple(shapes[node_id]),
                            dtype = np.dtype(dtypes[node_id]),
                            node_id = -1,
                        )
            #mem_tensor.update_last_usage(node_id)
            mem_tensor.update_last_usage(len(self.mod_info["nodes"]) - 1)
            
            # Get activations for each input for debugging purposes
            node_activations = []
            for tens_inp in input_tensors:
                if tens_inp.is_input or tens_inp.is_intermediate:
                    node_activations.append(tvm.nd.array(activations[tens_inp.name]))
                elif tens_inp.is_constant:
                    node_activations.append(tvm.nd.array(tens_inp.original_constant_val))

            # Run the TVM fallback node function to get debugging activations
            if "match" not in node["name"]:
                mem_tensor.used_by_tvm = True
                output_nd = tvm.nd.empty(shape=mem_tensor.shape, dtype=mem_tensor.dtype)
                self.host_module[node["attrs"]["func_name"]](*node_activations, output_nd)
                activations[mem_tensor.name] = output_nd.numpy()
            # Run the CPU version of the MATCH-node to get the debugging values
            else:
                module = tvm.contrib.graph_executor.GraphModule(host_lib["default"](self.dev))
                for tens_inp, param in zip(node_activations, host_lib.ir_mod["main"].params):
                    module.set_input(param.name_hint, tens_inp)
                module.run()
                output_np = module.get_output(0).numpy()
                activations[mem_tensor.name] = output_np

            mem_tensors.append(mem_tensor)
            tensor_map[node["name"] + "_out"] = mem_tensor
            tensor_name_to_id[mem_tensor.name] = mem_tensor.id
            outputs = [mem_tensor]
            
            call_node = MatchGraphRuntimeNodeCall(
                inputs = input_tensors,
                outputs = outputs,
                name = f"{self.model_name}_node_{node_id}",
                fn_name = node["attrs"]["func_name"],
                node_info = node,
                node_id = node_id,
                node_name = match_node_name,
                schedule = schedule,
                match_node = match_node,
                inp_tensor_ids= [tensor_name_to_id[tens.name] for tens in input_tensors],
                out_tensor_ids= [tensor_name_to_id[tens.name] for tens in outputs],
            )
            nodes.append(call_node)
            nodes_map[node["name"]] = call_node
            nodes_ids[call_node.name] = len(nodes) - 1
            map_names[tens_name] = (call_node.name, node["name"] + "_out", node["name"])
            
        
        # Gather parents and children for async execution
        for node_id, call_node in enumerate(nodes):
            for inp in call_node.inputs:
                if not inp.is_constant:
                    parent_node_name = inp.name.split("_out")[0]
                    if parent_node_name in nodes_ids:
                        parent_node_id = nodes_ids[parent_node_name]
                        nodes[parent_node_id].children.append(node_id)
                        nodes[node_id].num_parents += 1
                
                
        print("[GRAPH ASYNC] Tensors to be planned:")
        print(f"  (ID) {'Name':<30} {'Shape':<20} {'dtype':<10} {'Const':<7} {'Input':<7} {'Output':<7}")
        for tendsor_id, tensor in enumerate(mem_tensors):
            print(f"  ({tendsor_id:02d}) {tensor.name:<30} {str(tensor.shape):<20} {str(tensor.dtype):<10} {'C' if tensor.is_constant else '':<7} {'I' if tensor.is_input else '':<7} {'O' if tensor.is_output else '':<7}")

        print("[GRAPH ASYNC] Nodes to be executed:")
        print(f"  (ID) {'Name':<20} {'Function Name':<50} {'Type':<6} {'Input Tensors':<20} {'Output Tensors':<20} {'Num Parents':<12} {'Children':<15} {'Dev ID':<6}")
        for node_id, node in enumerate(nodes):
            print(f"  ({node_id:02d}) {node.name:<20} {node.fn_name:<50} {'TVM' if node.fallback else 'MATCH':<6} {str(node.inp_tensor_ids):<20} {str(node.out_tensor_ids):<20} {node.num_parents:<12} {str(node.children):<15} {node.device_id:<6}")

        # Memory planning
        self.mem_planner = MatchMemoryPlanner(
            mem_tensors = mem_tensors,
            available_soc_bytes = self.target.soc_memory_bytes,
            calls_idxs = [node.node_id for node in nodes],
            nodes = nodes,
            out_path = self.out_path,
            algorithm = "match",
            fix_io_tensors_in_ext_mem = self.target.fix_io_tensors_in_ext_mem,
        )
        self.mem_needed_bytes, self.ext_mem_needed_bytes = self.mem_planner.generate()

        # Gather inputs and outputs
        input_tensors = [tens for tens in mem_tensors if tens.is_input]
        outputs = []
        found_outputs_cnt = {}
        for head in heads:
            cnt = found_outputs_cnt.get(head, 0)
            outputs.append(idx_tensor_map[(head, cnt)])
            found_outputs_cnt[head] = cnt + 1

        # Ensure output directories exist
        Path(self.out_path, "parameters").absolute().mkdir(parents=True, exist_ok=True)
        Path(self.out_path, "golden").absolute().mkdir(parents=True, exist_ok=True)

        # Save parameters and activations
        for mem_tensor in mem_tensors:
            if mem_tensor.stored_in_external_memory and mem_tensor.is_constant:
                arr = np.frombuffer(mem_tensor.constant_val.flatten().tobytes(), dtype="uint8")
                arr.tofile(Path(self.out_path, f"parameters/{self.model_name}_{mem_tensor.name}_data.hex"))
            elif mem_tensor.stored_in_external_memory and mem_tensor.is_input:
                arr = np.frombuffer(activations[mem_tensor.name].flatten().tobytes(), dtype="uint8")
                arr.tofile(Path(self.out_path, f"parameters/{self.model_name}_{mem_tensor.name}_data.hex"))
        for activation_name, activation in activations.items():
            mem_tensor_ = next((m_t for m_t in mem_tensors if m_t.name == activation_name and not m_t.is_input), None)
            if mem_tensor_ is not None:
                arr = np.frombuffer(activation.flatten().tobytes(), dtype="uint8")
                arr.tofile(Path(self.out_path, f"golden/{self.model_name}_{activation_name}_data.hex"))
                    
        # Prepare template data for codegen
        template_data = {
            "async": True,
            "target": self.target,
            "mem_tensors": mem_tensors,
            "ext_mem_needed_bytes": self.ext_mem_needed_bytes,
            "mem_needed_bytes": self.mem_needed_bytes,
            "nodes": nodes,
            "model_name": self.model_name,
            "tensor_map": tensor_map,
            "nodes_map": nodes_map,
            "rt_inputs": input_tensors,
            "rt_outputs": outputs,
            "activations": activations,
            "map_names": map_names,
            "checksums": {
                activation_name: np.frombuffer(activation.flatten().tobytes(), dtype="uint8").sum()
                for activation_name, activation in activations.items()
            },
        }
        return template_data
