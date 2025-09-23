
import math
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict


import numpy as np
import tvm

from match.target.target import MatchTarget
from match.utils.utils import get_fname_node_schedule

from .node import RuntimeNode
from .tensor import RuntimeTensor, TensorType
from .optimize import optimize
from .viz import plot_optimization_result


class RuntimeGraph:
    def __init__(
        self, 
        mod_info: Dict[str, Any], 
        mod_params: Optional[dict], 
        match_inputs: dict = {},
        model_name: str = "default",
        host_module: Optional[dict] = None,
    ):
        # gid = relay global node id
        # tid = tensor id
        # nid = node (layer) id
        # sid = super-node id
        self.mod_info = mod_info
        self.model_name = model_name
        self.host_module = host_module or {}
        self.mod_params = mod_params or {}
        self.match_inputs = match_inputs
        self.tvm_device = tvm.cpu(0)

        self.super_nodes = []
        self.nid_to_sid = defaultdict(list)
        
        self.tensors = []  # List of MatchMemoryTensor objects
        tensor_name_to_tid = {}  # Maps tensor names to their IDs
        self.tensor_map = {}  # Maps tensor names to MatchMemoryTensor objects
        self.idx_tensor_map = {}  # Maps (node_id, output_index) to MatchMemoryTensor objects
        self.map_names = {}  # Maps tensor names to (call_name, out_name, node_name)

        self.nodes_map = {}  # Maps node names to MatchGraphRuntimeNodeCall objects
        nodes_ids = {}  # Maps node names to node IDs
        self.nodes = []  # List of MatchGraphRuntimeNodeCall objects

        dtypes = mod_info["attrs"]["dltype"][1]
        shapes = mod_info["attrs"]["shape"][1]
        self.heads = [head[0] for head in mod_info["heads"]]

        nop_maps = {}
        self.activations = {inp["name"]: inp["np_values"] for inp in match_inputs.values()}
        
        def estimate_latency_tvm(func_name, inp_tids, out_tids):
            if "conv2d" in func_name:
                filter_shape = self.tensors[inp_tids[1]].shape
                out_shape = self.tensors[out_tids[0]].shape
                # Estimate latency based on input and output shapes
                return filter_shape[0] * filter_shape[1] * out_shape[2] * out_shape[3]
            elif "dense" in func_name:
                in_shape = self.tensors[inp_tids[0]].shape
                out_shape = self.tensors[out_tids[0]].shape
                # Estimate latency based on input and output shapes
                return in_shape[1] * out_shape[1]
            else:
                out_shape = self.tensors[out_tids[0]].shape
                return math.prod(out_shape) // 10 # Default estimate based on output shape size

        # Pass 1: Create memory tensors for inputs and parameters
        for node_id, node in enumerate(mod_info["nodes"]):
            if node["op"] == "null":
                name = node["name"]
                is_output = node_id in self.heads
                dtype = np.dtype(dtypes[node_id])
                shape = tuple(shapes[node_id])

                if name in mod_params:
                    # Parameter node: weight tensor
                    param = mod_params[name]
                    const_val = param.numpy()
                    mem_tensor = RuntimeTensor(
                        id=len(self.tensors),
                        name=name,
                        type=TensorType.CONST,
                        constant_val=const_val,
                        original_constant_val=const_val,
                        shape=param.shape,
                        dtype=np.dtype(param.dtype),
                        node_id=node_id,
                        node_info=node,
                    )
                else:
                    # Input node: variable tensor
                    mem_tensor = RuntimeTensor(
                        id=len(self.tensors),
                        name=name,
                        type=TensorType.INOUT if is_output else TensorType.INPUT,
                        shape=shape,
                        dtype=dtype,
                        node_id=node_id,
                        node_info=node,
                    )

                self.tensors.append(mem_tensor)
                tensor_name_to_tid[name] = mem_tensor.id
                self.tensor_map[name] = mem_tensor
                self.idx_tensor_map[(node_id, 0)] = mem_tensor
                self.map_names[name] = (name, name, name)

                # For output input nodes, create an explicit output memory tensor too
                if name not in mod_params and is_output:
                    mem_tensor_out = RuntimeTensor(
                        id=len(self.tensors),
                        name=f"{name}_out",
                        type=TensorType.OUTPUT,
                        shape=shape,
                        dtype=dtype,
                        node_id=-1,
                        node_info=node,
                    )
                    self.tensors.append(mem_tensor_out)
                    self.tensor_map[f"{name}_out"] = mem_tensor_out
                    self.idx_tensor_map[(node_id, 0)] = mem_tensor_out
                    tensor_name_to_tid[mem_tensor_out.name] = mem_tensor_out.id

        # Pass 2: Create compute and intermediate tensors, nodes
        for node_id, node in enumerate(mod_info["nodes"]):
            if node["op"] == "null":
                continue

            input_tensors = []

            # Build input list for this node
            for inp_idxs in node["inputs"]:
                inp_node = mod_info["nodes"][inp_idxs[0]]
                inp_name = inp_node["name"]
                inp_tensor_name = (inp_name + "_out") if inp_node["op"] != "null" else inp_name

                if inp_node["op"] != "null" and "_nop" in inp_name and inp_name in nop_maps:
                    # NOP input node, bypass to NOP input node input
                    input_tensors.append(nop_maps[inp_name])
                else:
                    input_tensors.append(self.tensor_map[inp_tensor_name])

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
                            mem_tensor = RuntimeTensor(
                                id=len(self.tensors),
                                name=f"{match_node_name}_{s_tensor.name}",
                                type=TensorType.CONST,
                                constant_val=s_tensor.data,
                                original_constant_val=s_tensor.original_data,
                                shape=s_tensor.data.shape,
                                dtype=s_tensor.dtype,
                                node_id=node_id,
                                node_info=node,
                            )
                            self.tensors.append(mem_tensor)
                            self.tensor_map[s_tensor.name] = mem_tensor
                            self.idx_tensor_map[(node_id, 0)] = mem_tensor
                            input_tensors.append(mem_tensor)
                            tensor_name_to_tid[mem_tensor.name] = mem_tensor.id

            # Mark tensor usage info for memory planning
            for inp in input_tensors:
                if "match" not in node["name"]:
                    inp.used_by_tvm = True
                if inp.is_output:
                    inp.is_intermediate = True

            # Build output tensor for this node
            id_out = -1
            tens_name = f"{self.model_name}_node_{node_id}_out"
            for head_idx, head in enumerate(self.heads):
                if head == node_id:
                    id_out = head_idx
                    tens_name = f"{self.model_name}_out_{id_out}"
                    break

            is_output = id_out != -1
            is_intermediate = id_out == -1

            mem_tensor = RuntimeTensor(
                id=len(self.tensors),
                name=tens_name,
                type=TensorType.OUTPUT if is_output else TensorType.INTERMEDIATE,
                shape=tuple(shapes[node_id]),
                dtype=np.dtype(dtypes[node_id]),
                node_id=node_id,
            )

            self.idx_tensor_map[(node_id, 0)] = mem_tensor
            if is_output:
                cnt = 0
                for head_idx, head in enumerate(self.heads[id_out + 1 :], id_out + 1):
                    if head == node_id:
                        cnt += 1
                        self.idx_tensor_map[(node_id, cnt)] = RuntimeTensor(
                            name=f"{self.model_name}_out_{head_idx}",
                            type=TensorType.OUTPUT,
                            shape=tuple(shapes[node_id]),
                            dtype=np.dtype(dtypes[node_id]),
                            node_id=-1,
                        )

            # Get self.activations for each input for debugging purposes
            node_activations = []
            for tens_inp in input_tensors:
                if tens_inp.is_input or tens_inp.is_intermediate:
                    node_activations.append(tvm.nd.array(self.activations[tens_inp.name]))
                elif tens_inp.is_constant:
                    node_activations.append(tvm.nd.array(tens_inp.original_constant_val))

            # Run the TVM fallback node function to get debugging self.activations
            if "match" not in node["name"]:
                mem_tensor.used_by_tvm = True
                output_nd = tvm.nd.empty(shape=mem_tensor.shape, dtype=mem_tensor.dtype)
                self.host_module[node["attrs"]["func_name"]](*node_activations, output_nd)
                self.activations[mem_tensor.name] = output_nd.numpy()
            # Run the CPU version of the MATCH-node to get the debugging values
            else:
                module = tvm.contrib.graph_executor.GraphModule(host_lib["default"](self.tvm_device))
                for tens_inp, param in zip(node_activations, host_lib.ir_mod["main"].params):
                    module.set_input(param.name_hint, tens_inp)
                module.run()
                output_np = module.get_output(0).numpy()
                self.activations[mem_tensor.name] = output_np

            self.tensors.append(mem_tensor)
            self.tensor_map[node["name"] + "_out"] = mem_tensor
            tensor_name_to_tid[mem_tensor.name] = mem_tensor.id
            outputs = [mem_tensor]

            inp_tids = [tensor_name_to_tid[tens.name] for tens in input_tensors]
            out_tids = [tensor_name_to_tid[tens.name] for tens in outputs]
            call_node = RuntimeNode(
                id=len(self.nodes),
                inputs=input_tensors,
                outputs=outputs,
                name=f"{self.model_name}_node_{node_id}",
                fn_name=node["attrs"]["func_name"],
                node_info=node,
                node_id=node_id,
                node_name=match_node_name,
                mapping=schedule,
                duration= int(schedule.latency) if schedule else estimate_latency_tvm(node["attrs"]["func_name"], inp_tids, out_tids),
                match_node=match_node,
                inp_tids=inp_tids,
                out_tids=out_tids,
                device_id = 0 if schedule is None else (schedule.exec_module.id + 1)
            )
            self.nodes.append(call_node)
            self.nodes_map[node["name"]] = call_node
            nodes_ids[call_node.name] = len(self.nodes) - 1
            self.map_names[tens_name] = (call_node.name, node["name"] + "_out", node["name"])

        # Gather parents and children for async execution
        for node_id, call_node in enumerate(self.nodes):
            for inp in call_node.inputs:
                if not inp.is_constant:
                    parent_node_name = inp.name.split("_out")[0]
                    if parent_node_name in nodes_ids:
                        parent_node_id = nodes_ids[parent_node_name]
                        self.nodes[parent_node_id].children_nids.append(node_id)
                        self.nodes[node_id].num_parents += 1
                        
        # Update tensor to node ids
        self.tensor_to_nids = {t.id: [] for t in self.tensors}
        for node in self.nodes:
            for tensor_id in node.inp_tids + node.out_tids:
                self.tensor_to_nids[tensor_id].append(node.id)
                
        # Empty pattern list
        self.patterns = []


class Runtime:
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
        self.ext_mem_pool_size = 0
        self.soc_mem_pool_size = 0
        self.match_inputs = match_inputs or {}
        self.host_module = host_module or {}

    def generate(self) -> dict:
        """
        Generate tensors, nodes and memory plan and scheduling for the runtime graph.
        """
        
        runtime_graph = RuntimeGraph(self.mod_info, self.params, self.match_inputs, self.model_name, self.host_module)

        print("[GRAPH ASYNC] Tensors to be planned:")
        print(f"  (ID) {'Name':<30} {'Shape':<20} {'dtype':<10} {'Type':<7}")
        for tendsor_id, tensor in enumerate(runtime_graph.tensors):
            print(
                f"  ({tendsor_id:02d}) {tensor.name:<30} {str(tensor.shape):<20} {str(tensor.dtype):<10} {'C'*tensor.is_constant+'I'*tensor.is_input+'O'*tensor.is_output:<7}"
            )

        print("[GRAPH ASYNC] Nodes to be executed:")
        print(
            f"  (ID) {'Name':<20} {'Function Name':<50} {'Type':<6} {'Input Tensors':<20} {'Output Tensors':<20} {'Num Parents':<12} {'Children':<15} {'Dev ID':<6}"
        )
        for node_id, node in enumerate(runtime_graph.nodes):
            print(
                f"  ({node_id:02d}) {node.name:<20} {node.fn_name:<50} {'TVM' if node.fallback else 'MATCH':<6} {str(node.inp_tids):<20} {str(node.out_tids):<20} {node.num_parents:<12} {str(node.children_nids):<15} {node.device_id:<6}"
            )

        # Optimize again
        print("[OPTIMIZER] Optimizing scheduling and memory planning.")

        model, solver, solution = optimize(
            runtime_graph,
            devices=list(range(len(self.target.exec_modules) + 1)),
            l2_size=self.target.soc_memory_bytes,
            l3_size=4_000_000,
            bandwidth=4,
            dtype_size=2,
            scale_time=True,
            scale_addr=False,
            tiling=False
        )

        plot_path = Path(self.out_path, "optimization_result.png").absolute()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_optimization_result(solution, filename=plot_path)

        self.soc_mem_pool_size, self.ext_mem_pool_size = solution['l2_peak_usage']*solution['addr_scale'], solution['l3_peak_usage']*solution['addr_scale']
        
        # Update tensors with solution
        for t, tensor_sol in enumerate(solution['tensors']):
            def scale(x):
                return x * solution['addr_scale']
            runtime_graph.tensors[t].soc_mem_offsets = list(map(scale, tensor_sol['l2_offsets']))
            runtime_graph.tensors[t].ext_mem_offsets = list(map(scale, tensor_sol['l3_offsets']))
            runtime_graph.tensors[t].static_in_soc_mem = tensor_sol['static_in_l2']
            runtime_graph.tensors[t].static_in_ext_mem = tensor_sol['static_in_l3']
            
        # Update Nodes with solution
        for n, node_sol in enumerate(solution['nodes']):
            runtime_graph.nodes[n].tensor_soc_segments_ids = node_sol['tensors_segments']
            
        # Build priority queue for each device
        device_queues = {d: list(n['node_id'] for n in sorted(filter(lambda n: n['device'] == d, solution['nodes']), key=lambda n: n['start']))
                         for d in range(len(self.target.exec_modules) + 1)}

        # Check partitioning is ok 
        assert set(sum(device_queues.values(), [])) == set(n.id for n in runtime_graph.nodes)
        print("  Device queues: ", device_queues)
        
        # Add extra node dependencies between nodes sharing the same pool address space
        def addr_overlap(start1, size1, start2, size2):
            return start1 < start2 + size2 and start2 < start1 + size1
        for n, node in enumerate(runtime_graph.nodes):
            for n_, future_node in enumerate(runtime_graph.nodes):
                if solution['nodes'][n]['end'] <= solution['nodes'][n_]['start'] and n != n_:
                    for tid, segment in solution['nodes'][n]['tensors_segments'].items():
                        if solution['tensors'][tid]['static_in_l2']:
                            continue
                        t_offset = solution['tensors'][tid]['l2_offsets'][segment]
                        t_size = runtime_graph.tensors[tid].size
                        for tid_, segment_ in solution['nodes'][n_]['tensors_segments'].items():
                            if solution['tensors'][tid_]['static_in_l2']:
                                continue
                            t_offset_ = solution['tensors'][tid_]['l2_offsets'][segment_]
                            t_size_ = runtime_graph.tensors[tid_].size
                            if addr_overlap(t_offset, t_size, t_offset_, t_size_):
                                # Add dependency if segments overlap
                                if n_ not in node.children_nids:
                                    runtime_graph.nodes[n].children_nids.append(n_)
                                    runtime_graph.nodes[n_].num_parents += 1
                                    #print(f"  Adding dependency from node {node.name} to {future_node.name} due to overlapping tensor segments in L2.")
                                    assert n not in runtime_graph.nodes[n_].children_nids, f"Cycle detected between nodes ({n}) {node.name} and ({n_}) {future_node.name}!"
                                    
        # Add extra node dependencies to enforce execution order in each device
        for d, queue in device_queues.items():
            if len(queue) < 2:
                continue
            for i in range(len(queue) - 1):
                node_id = queue[i]
                next_node_id = queue[i + 1]
                if next_node_id not in runtime_graph.nodes[node_id].children_nids:
                    runtime_graph.nodes[node_id].children_nids.append(next_node_id)
                    runtime_graph.nodes[next_node_id].num_parents += 1
                    #print(f"  Adding dependency from node {runtime_graph.nodes[node_id].name} to {runtime_graph.nodes[next_node_id].name} to enforce execution order in device {d}.")

        # Gather inputs and outputs
        input_tensors = [tens for tens in runtime_graph.tensors if tens.is_input]
        outputs = []
        found_outputs_cnt = {}
        for head in runtime_graph.heads:
            cnt = found_outputs_cnt.get(head, 0)
            outputs.append(runtime_graph.idx_tensor_map[(head, cnt)])
            found_outputs_cnt[head] = cnt + 1

        # Ensure output directories exist
        Path(self.out_path, "parameters").absolute().mkdir(parents=True, exist_ok=True)
        Path(self.out_path, "golden").absolute().mkdir(parents=True, exist_ok=True)

        # Save parameters and activations
        for mem_tensor in runtime_graph.tensors:
            if mem_tensor.static_in_ext_mem and mem_tensor.is_constant:
                arr = np.frombuffer(mem_tensor.constant_val.flatten().tobytes(), dtype="uint8")
                arr.tofile(Path(self.out_path, f"parameters/{self.model_name}_{mem_tensor.name}_data.hex"))
            elif mem_tensor.static_in_ext_mem and mem_tensor.is_input:
                arr = np.frombuffer(runtime_graph.activations[mem_tensor.name].flatten().tobytes(), dtype="uint8")
                arr.tofile(Path(self.out_path, f"parameters/{self.model_name}_{mem_tensor.name}_data.hex"))
        for activation_name, activation in runtime_graph.activations.items():
            mem_tensor_ = next((m_t for m_t in runtime_graph.tensors if m_t.name == activation_name and not m_t.is_input), None)
            if mem_tensor_ is not None:
                arr = np.frombuffer(activation.flatten().tobytes(), dtype="uint8")
                arr.tofile(Path(self.out_path, f"golden/{self.model_name}_{activation_name}_data.hex"))
                np.savetxt(Path(self.out_path, f"golden/{self.model_name}_{activation_name}_debug.txt"), activation.flatten(), delimiter=", ")
        
        # Calculate checksums for activations
        checksums = {
            activation_name: np.frombuffer(activation.flatten().tobytes(), dtype="uint8").sum()
            for activation_name, activation in runtime_graph.activations.items()
        }
        
        with open("matcha.activations.json", "w") as f:
            import json
            json.dump(
                {k: v.tolist() for k, v in runtime_graph.activations.items()},
                f,
                indent=4,
                sort_keys=True,
            )

        # Prepare template data for codegen
        template_data = {
            "async": True,
            "target": self.target,
            "nodes": runtime_graph.nodes,
            "tensors": runtime_graph.tensors,
            "ext_mem_pool_size": self.ext_mem_pool_size,
            "soc_mem_pool_size": self.soc_mem_pool_size,
            "model_name": self.model_name,
            "tensor_map": runtime_graph.tensor_map,
            "nodes_map": runtime_graph.nodes_map,
            "rt_inputs": input_tensors,
            "rt_outputs": outputs,
            "activations": runtime_graph.activations,
            "map_names": runtime_graph.map_names,
            "checksums": checksums,
        }
        return template_data
