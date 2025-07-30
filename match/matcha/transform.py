import json
from functools import partial

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback, rewrite

from match.target import MatchTarget, MatchTargetPattern

from .graph import Graph, PatternAnnotator
from .globals import set_optimization_result
from .optimize import optimize
from .viz import plot_optimization_result


    
@tvm.ir.transform.module_pass(opt_level=0)
class MatchOptimizer:
    def __init__(self, target : MatchTarget):
        super().__init__()
        self.target = target

    def transform_module(self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext) -> tvm.ir.IRModule:
        
        patterns = [m_pt for m_pt in self.target.match_patterns if m_pt.exec_module.name not in self.target.disabled_exec_modules]
        
        
        graph = Graph(mod, patterns)
        
        
        devices = list(range(len(self.target.exec_modules) + 1))
        l2_size = self.target.soc_memory_bytes # 256_000
        l3_size = 4_000_000
        bandwidth = 4
        dtype_size = 2
        
        model, solver, solution, matched_patterns = optimize(
            graph, 
            devices, 
            l2_size, 
            l3_size, 
            bandwidth, 
            dtype_size, 
            scale_time=True, 
            scale_addr=False
        )
        
        with open("matcha.result.json", "w") as f:
            json.dump(solution, f, indent=4)
            
        print("Tensors:")
        for tensor in graph.tensors:
            print(f"ID: {tensor.id:>5}, Size: {tensor.size:>20}, Type: {tensor.type.name:>20}, Shape: {str(tensor.shape):>20}")
            
        print("Nodes:")
        for node in graph.nodes:
            print(f"ID: {node.id:>5}, Duration: {node.duration:>15}, Inputs: {str(node.inp_tids):>30}, Outputs: {str(node.out_tids):>10}, Children: {str([c for c in node.children_nids if c < len(graph.nodes)]):>10}, Super-children: {str([c for c in node.children_nids if c >= len(graph.nodes)]):>15}, Chunks: {str(node.chunks):>5}")
        
        print("Super Nodes:")
        for node in graph.super_nodes:
            print(f"ID: {node.id:>5}, Duration: {node.duration:>15}, Inputs: {str(node.inp_tids):>30}, Outputs: {str(node.out_tids):>10}, Children: {str([c for c in node.children_nids if c < len(graph.nodes)]):>10}, Super-children: {str([c for c in node.children_nids if c >= len(graph.nodes)]):>15}, Chunks: {str(node.chunks):>5}, Sub-nodes: {str(node.sub_nids):>20}")
        
        print("Node to super_node ids:")
        print(graph.nid_to_sid)
        
        plot_optimization_result(solution)
        
        set_optimization_result(solution)
    
        
        for pattern in patterns:
            annotator = PatternAnnotator(pattern)
            mod['main'] = rewrite(annotator, mod['main'])
            print(f"Pattern matched {annotator.num_matches} times.")
        
        def merge_check(node, p):
            if isinstance(node, relay.Call):
                return node.span.line in matched_patterns[p]
            return False
        
        merging_rules = [
            (f"match.{pattern.name}", pattern.pattern(), partial(merge_check, p=p)) for p, pattern in enumerate(patterns)
        ]
        
        mod = relay.transform.MergeComposite(merging_rules)(mod)

        return mod

    def __call__(self, mod):
        return self.transform_module(mod)

