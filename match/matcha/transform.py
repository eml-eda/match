import json
from functools import partial

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import rewrite

from match.target import MatchTarget

from .graph import Graph
from .globals import set_optimization_result, set_starting_graph
from .optimize import optimize
from .splitter import NodeSplitter
from .viz import plot_optimization_result

def get_shape(expr):
    # Wrap expr in a dummy function/module to run InferType
    free_vars = relay.analysis.free_vars(expr)
    func = relay.Function(free_vars, expr)
    mod = tvm.IRModule({"main": func})
    mod = relay.transform.InferType()(mod)
    # For a function, the body is expr
    shape = [int(x) for x in mod["main"].body.checked_type.shape]
    return shape

class SplitGroupedConvMutator(relay.ExprMutator):
    def visit_call(self, call):
        new_call = super().visit_call(call)
        if isinstance(new_call.op, tvm.ir.Op) and new_call.op.name == "nn.conv2d":
            attrs = new_call.attrs
            groups = attrs.groups
            if groups > 1 and groups != new_call.args[1].checked_type.shape[-1]: # Split if not depthwise
                data, weight = new_call.args
                data_shape = get_shape(data)
                weight_shape = get_shape(weight)
                
                print("fm_shape: ", data_shape, " weight_shape: ", weight_shape, " groups: ", groups)

                # Input: (N, H, W, CI), Weight: (KH, KW, CI//groups, CO) TODO - get data_layout and weight_layout from attrs
                in_channels = data_shape[-1]
                out_channels = weight_shape[-1]
                in_step = in_channels // groups
                out_step = out_channels // groups
                
                print("in_channels: ", in_channels, " out_channels: ", out_channels, " in_step: ", in_step, " out_step: ", out_step)

                out_convs = []
                for g in range(groups):
                    data_slice = relay.strided_slice(
                        data,
                        begin=[0, 0, 0, g * in_step],
                        end=[data_shape[0], data_shape[1], data_shape[2], (g + 1) * in_step]
                    )
                    weight_slice = relay.strided_slice(
                        weight,
                        begin=[0, 0, 0, g * out_step],
                        end=[weight_shape[0], weight_shape[1], weight_shape[2], (g + 1) * out_step]
                    )
                    # Set groups=1 for split conv
                    new_attrs = dict(attrs)
                    new_attrs['groups'] = 1
                    new_attrs['channels'] = in_step
                    
                    conv_op = relay.nn.conv2d(data_slice, weight_slice, **new_attrs)
                    out_convs.append(conv_op)

                # Concatenate along channel axis
                print("Done splitting grouped conv with groups=", groups)
                concat = relay.concatenate(out_convs, axis=3)
                return concat
        return new_call

class NodeAnnotator(relay.ExprMutator):
    # Just yet another workaround to uniquely identify nodes across transformations in TVM :)
    def __init__(self):
        super().__init__()
        self.node_id = 0
        self.var_map = {}

    def _make_span(self):
        span = tvm.ir.Span(tvm.ir.SourceName("GID"), self.node_id, 0, 0, 0)
        self.node_id += 1
        return span

    def visit_var(self, var):
        if var not in self.var_map:
            new_var = relay.Var(var.name_hint, var.type_annotation, span=self._make_span())
            self.var_map[var] = new_var
        return self.var_map[var]

    def visit_constant(self, const):
        return relay.Constant(const.data, span=self._make_span())

    def visit_call(self, call):
        new_op = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return relay.Call(new_op, new_args, call.attrs, call.type_args, span=self._make_span())

    def visit_function(self, fn):
        new_params = [self.visit_var(p) for p in fn.params]
        new_body = self.visit(fn.body)
        return relay.Function(new_params, new_body, fn.ret_type, fn.type_params, span=self._make_span())




@tvm.ir.transform.module_pass(opt_level=0)
class MatchOptimizer:
    def __init__(self, target : MatchTarget):
        super().__init__()
        self.target = target

    def transform_module(self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext) -> tvm.ir.IRModule:
        
        patterns = [m_pt for m_pt in self.target.match_patterns if m_pt.exec_module.name not in self.target.disabled_exec_modules]
        
        func = mod['main']
        
        if False:
            grouped_conv_splitter = SplitGroupedConvMutator()
            mod['main'] = grouped_conv_splitter.visit(func)
            mod = relay.transform.FoldConstant()(mod)
            func = mod['main']
        
        node_annotator = NodeAnnotator()
        mod['main'] = node_annotator.visit(func)
          
        mod = relay.transform.InferType()(mod)
            
        graph = Graph(mod, patterns)
        set_starting_graph(graph)
        
        devices = list(range(len(self.target.exec_modules) + 1))
        l2_size = self.target.soc_memory_bytes # 256_000
        #l2_size = 512_000
        print(l2_size)
        #quit()
        #l3_size = 4_000_000
        l3_size = 0
        bandwidth = 4
        dtype_size = 2
        
        model, solver, solution = optimize(
            graph, 
            devices, 
            l2_size, 
            l3_size, 
            bandwidth, 
            dtype_size, 
            scale_time=True, 
            scale_addr=False,
            tiling=self.target.enable_device_parallelism,
            optimize_space=False
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
        
        mod = relay.transform.InferType()(mod)
        
        # Split matched nodes if needed
        
        # TODO

        print("\nORIGINAL")
        print(mod)
        matched_patterns_chunks = solution['matched_patterns_chunks']
        for p, pattern in enumerate(patterns):
            if p in matched_patterns_chunks:
                print(f"Splitting pattern {p} ({pattern.name})")
                splitter = NodeSplitter(pattern, p, matched_patterns_chunks[p])
                mod['main'] = rewrite(splitter, mod['main'])
                mod = relay.transform.InferType()(mod)

        print("\nSPLITTED")
        print(mod)
        
        print("\nSHAPE INFERENCE")
        #mod = relay.transform.AnnotateSpans()(mod)
        mod = relay.transform.InferType()(mod)
        print(mod)
        
        
        # Merge chosen pattern matchings
        matched_patterns = solution['matched_patterns_gids']
        print("Matched Patterns Nodes: ", matched_patterns)
        
        def merge_check(node, p, device_id):
            if p not in matched_patterns:
                return False
            if hasattr(node, "span") and node.span and node.span.source_name.name == "GID":
                gid = node.span.line
                return gid in matched_patterns[p] and node.span.column == device_id
            return False
        
        merging_rules = [
            (f"match.{pattern.name}", pattern.pattern(), partial(merge_check, p=p, device_id=pattern.exec_module.id+1)) for p, pattern in enumerate(patterns)
        ]
        
        mod = relay.transform.MergeComposite(merging_rules)(mod)
        mod = relay.transform.FoldConstant()(mod)

        return mod

    def __call__(self, mod):
        return self.transform_module(mod)

