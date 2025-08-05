import math
from copy import deepcopy
from typing import List

import networkx as nx

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback, rewrite

from match.target import MatchTargetPattern

from .tensor import Tensor, TensorType
from .node import Node, SuperNode
    
    
class PatternCollector(DFPatternCallback):
    def __init__(self, match_pattern : MatchTargetPattern):
        super().__init__(require_type=False)
        self.pattern = match_pattern.pattern()
        self.check = match_pattern.additional_checks
        self.matches = []
        
    def callback(self, pre, post, node_map):
        if not self.check(pre):
            return post
        match = []
        
        def recursive_collect(pattern):
            if pattern in node_map:
                if isinstance(pattern, relay.dataflow_pattern.CallPattern):
                    if node_map[pattern][0] not in match:
                        match.append(node_map[pattern][0])
                    for arg in pattern.args:
                        recursive_collect(arg)
                elif isinstance(pattern, relay.dataflow_pattern.AltPattern):
                    if pattern.left in node_map:
                        recursive_collect(pattern.left)
                    if pattern.right in node_map:
                        recursive_collect(pattern.right)
                
        recursive_collect(self.pattern)
        self.matches.append(match)
        return post


class Graph():
    def __init__(
        self, 
        mod : tvm.ir.IRModule, 
        patterns : List[MatchTargetPattern]
    ):
        # gid = relay global node id
        # tid = tensor id
        # nid = node (layer) id
        # sid = super-node id
        
        self.device_speeds = [1, 8, 8]
        self.mod = mod
        self.patterns = patterns
        
        self._relay_to_dict(mod)
        
        self._build_graph()
                    
        self._extract_tensors()

        self._extract_nodes()
        
        self._match_patterns()
        
        print(f"Graph has {len(self.super_nodes)} possible super-nodes and {len(self.nodes)} nodes.")
        
    
    def _relay_to_dict(self, mod : tvm.ir.IRModule):
        self.relay_to_gid = {}
        self.gid_to_relay = {}
        
        nodes_not_annotated = []
        
        def _visit_relay_node(node):
            if node in self.relay_to_gid or node in nodes_not_annotated:
                return
            if isinstance(node, tvm.ir.Op):
                return 
            if hasattr(node, 'span') and node.span.source_name.name == "GID":
                gid = node.span.line
                self.gid_to_relay[gid] = node
                self.relay_to_gid[node] = gid
            else:
                nodes_not_annotated.append(node)
        
        relay.analysis.post_order_visit(mod['main'], _visit_relay_node)
        
        print(f"  Found {len(self.relay_to_gid)} relay nodes with GID annotation.")
        print(f"  Found {len(nodes_not_annotated)} relay nodes without GID annotation. These will be assigned a new GID.")
        
        for node in nodes_not_annotated:
            self.gid_to_relay[len(self.relay_to_gid)] = node
            self.relay_to_gid[node] = len(self.relay_to_gid)
            
        
    def _build_graph(self):
        self.graph = nx.DiGraph()
        for node, node_idx in self.relay_to_gid.items():
            if isinstance(node, relay.Var):
                self.graph.add_node(node_idx, label=f'{tuple(node.type_annotation.shape)}', shape='box', type="input", op="None", num_ops=0, tensor_shape=tuple(node.type_annotation.shape))
            elif isinstance(node, relay.Constant):
                self.graph.add_node(node_idx, label=f'{tuple(node.data.shape)}', shape='box', type="const", op="None", num_ops=0, tensor_shape=tuple(node.data.shape))
            elif isinstance(node, relay.Call):
                args = [self.relay_to_gid[arg] for arg in node.args]
                num_ops = Graph._call_num_ops_estimator(node, node.op.name)
                self.graph.add_node(node_idx, label=f'{node.op.name}', shape='box', type="call", op=f"{node.op.name}", num_ops=num_ops, tensor_shape=tuple(node.checked_type.shape))
                for arg in args:
                    self.graph.add_edge(arg, node_idx)
            elif isinstance(node, relay.Function):
                self.graph.nodes[self.relay_to_gid[node.body]]['type'] = "output"
            else:
                print("Unknown node type:", type(node))
                #raise RuntimeError(f'Unknown node type. node_idx: {node_idx}, node: {type(node)}')
            
    def _extract_tensors(self):
        tensors, gid_to_tid = [], {}
        for gid in self.graph.nodes:
            node = self.graph.nodes[gid]
            tensor_id = len(tensors)
            if node['type'] == "input":
                gid_to_tid[gid] = tensor_id
                tensors.append(Tensor(id=tensor_id, shape=tuple(map(int, node['tensor_shape'])), type=TensorType.INPUT))
            elif node['type'] == "const":
                gid_to_tid[gid] = tensor_id
                tensors.append(Tensor(id=tensor_id, shape=tuple(map(int, node['tensor_shape'])), type=TensorType.CONST))
            elif node['type'] == "call":
                gid_to_tid[gid] = tensor_id
                tensors.append(Tensor(id=tensor_id, shape=tuple(map(int, node['tensor_shape'])), type=TensorType.INTERMEDIATE))
            elif node['type'] == "output":
                gid_to_tid[gid] = tensor_id
                tensors.append(Tensor(id=tensor_id, shape=tuple(map(int, node['tensor_shape'])), type=TensorType.OUTPUT))
            else:
                pass
        self.tensors = tensors
        self.gid_to_tid = gid_to_tid
    
    def _extract_nodes(self):
        nodes, gid_to_nid, nid_to_gid = [], {}, {}
        for gid in self.graph.nodes:
            node = self.graph.nodes[gid]
            if node['type'] in {"call", "output"}:
                nid = len(nodes)
                gid_to_nid[gid] = nid 
                nid_to_gid[nid] = gid
            
                out_tid = self.gid_to_tid[gid]
                if len(self.tensors[out_tid].shape) == 4:
                    self.tensors[out_tid].tiling_dim = 2 # H
                else:
                    self.tensors[out_tid].tiling_dim = 1 # K
                
                nodes.append(Node(
                    id=nid,
                    inp_tids=[self.gid_to_tid[inp] for inp in self.graph.predecessors(gid) if inp in self.gid_to_tid],
                    out_tids=[out_tid],
                    children_nids=list(self.graph.successors(gid)),
                    duration=int(node['num_ops']),
                    device_id=0,
                    chunks = self.tensors[out_tid].chunks,
                ))
                
        for i in range(len(nodes)):
            nodes[i].children_nids = [gid_to_nid[child] for child in nodes[i].children_nids if child in gid_to_nid]
        
        self.nodes = nodes
        self.gid_to_nid = gid_to_nid
        self.nid_to_gid = nid_to_gid
    
        
    def _match_patterns(self):
        self.super_nodes = []
        self.nid_to_sid = [[] for _ in range(len(self.nodes))]
        self.child_nid_to_sid = [[] for _ in range(len(self.nodes))]
        
        for pat_id, pattern in enumerate(self.patterns):
            collector = PatternCollector(pattern)
            rewrite(collector, self.mod["main"])
            for mat_id, match in enumerate(collector.matches):
                #print(f"Match: {[self.gid_to_nid[self.relay_to_gid[node]] for node in match]}")
                
                super_node_id = len(self.super_nodes) + len(self.nodes)
                
                matched_gids, matched_nids = [], []
                for node in match:
                    if node in self.relay_to_gid:
                        g_node = self.relay_to_gid[node]
                        if g_node in self.gid_to_nid:
                            matched_gids.append(g_node)
                            matched_nids.append(self.gid_to_nid[g_node])
                
                first_nids, last_nids = [], []
                for nid, gid in zip(matched_nids, matched_gids):
                    if not any(parent in matched_gids for parent in self.graph.predecessors(gid)):
                        first_nids.append(nid)
                    if not any(child in matched_gids for child in self.graph.successors(gid)):
                        last_nids.append(nid)
                assert first_nids and last_nids, "Pattern match must have a first and last node."
                
                inp_tids = list(set(inp_id for nid in matched_nids for inp_id in self.nodes[nid].inp_tids if self.tensors[inp_id].type != TensorType.INTERMEDIATE or nid in first_nids))
                out_tids = list(set(out_id for nid in matched_nids for out_id in self.nodes[nid].out_tids if nid in last_nids))
                children_nids = []
                for gid in matched_gids:
                    if gid in self.gid_to_nid:
                        nid = self.gid_to_nid[gid]
                        if nid in last_nids:
                            children_nids.extend(self.nodes[nid].children_nids)
                duration = int(int(sum(self.nodes[nid].duration for nid in matched_nids)) / self.device_speeds[pattern.exec_module.id + 1])
                
                self.super_nodes.append(SuperNode(
                    id = super_node_id,
                    inp_tids = inp_tids, # all input tensors of contained nodes
                    out_tids = out_tids, # last output tensor of contained nodes
                    children_nids = children_nids,
                    duration = duration,
                    device_id = pattern.exec_module.id + 1,
                     chunks = self.tensors[out_tids[0]].chunks,
                    pattern_id = pat_id,
                    match_id = mat_id,
                    sub_nids = list(nid for nid in matched_nids),
                ))
                
                for nid in first_nids:
                    self.child_nid_to_sid[nid].append(super_node_id)
                for nid in matched_nids:
                    self.nid_to_sid[nid].append(super_node_id)
                
        # Update super-node children ids   
        for i in range(len(self.nodes)):
            if self.nodes[i].children_nids is None:
                self.nodes[i].children_nids = []
            children_nids = deepcopy(self.nodes[i].children_nids)
            for child_id in children_nids:
                self.nodes[i].children_nids.extend(self.child_nid_to_sid[child_id])
            self.nodes[i].children_nids = list(set(self.nodes[i].children_nids))
        
        for i in range(len(self.super_nodes)):
            if self.super_nodes[i].children_nids is None:
                self.super_nodes[i].children_nids = []
            children_nids = deepcopy(self.super_nodes[i].children_nids)
            for child_id in children_nids:
                self.super_nodes[i].children_nids.extend(self.child_nid_to_sid[child_id])
            self.super_nodes[i].children_nids = list(set(self.super_nodes[i].children_nids))
            
        # Update tensor to node ids
        self.tensor_to_nids = {t.id: [] for t in self.tensors}
        for node in self.nodes:
            for tensor_id in node.inp_tids + node.out_tids:
                self.tensor_to_nids[tensor_id].append(node.id)
        for super_node in self.super_nodes:
            for tensor_id in super_node.inp_tids + super_node.out_tids:
                self.tensor_to_nids[tensor_id].append(super_node.id)
        

        
    @staticmethod
    def _call_num_ops_estimator(call, op_name):
        if op_name in ["nn.conv2d"]:
            batch, out_channels, out_h, out_w = [int(i) for i in call.checked_type.shape]
            kernel_h, kernel_w = call.attrs.kernel_size
            in_channels = int(call.args[1].checked_type.shape[1])
            ops = batch * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w
            return ops
        elif op_name in ["nn.max_pool2d", "nn.avg_pool2d"]:
            batch, out_channels, out_h, out_w = [int(i) for i in call.checked_type.shape]
            kernel_h, kernel_w = call.attrs.pool_size
            in_channels = int(call.args[0].checked_type.shape[1])
            ops = batch * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w
            return ops
        elif op_name in ["nn.dense"]:
            # Extract info for dense
            batch, out_dim = [int(i) for i in call.checked_type.shape]
            in_dim = int(call.args[1].checked_type.shape[1])
            ops = batch * in_dim * out_dim
            return ops
        elif op_name in ["nn.relu", "reshape", "nn.bias_add", "multiply", "add", "transpose"]:
            # Assume one op for each tensor element
            dims = [int(i) for i in call.checked_type.shape]
            return math.prod(dims)
        else:
            return 0  # Fallback for other ops

  