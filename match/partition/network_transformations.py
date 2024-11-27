import json
from math import prod
import re

import numpy as np
from match.relay import CompiledModule
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.dataflow_pattern import DFPatternCallback, is_op, rewrite, wildcard
from tvm.relay import transform
from tvm import relay
import tvm
from match.utils import add_save_relay

class RewriteOnnxBiasesCallback(DFPatternCallback):
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.conv2d = is_op("nn.conv2d")(wildcard(),wildcard())
        self.bias = wildcard()
        self.pattern = is_op("nn.bias_add")(self.conv2d,self.bias)

    def callback(self, pre, post, node_map):
        conv2d = node_map[self.conv2d][0]
        out_dtype=conv2d.attrs["out_dtype"]
        bias = node_map[self.bias][0]
        return relay.op.nn.bias_add(relay.op.cast(conv2d, out_dtype),bias)


@tvm.ir.transform.module_pass(opt_level=0)
class RewriteOnnxBiases:
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        global_var=mod.get_global_var("main")
        func=mod.functions[global_var]
        func = rewrite(RewriteOnnxBiasesCallback(), func)
        mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)

@transform.function_pass(opt_level=0)
class MatchOnnxBiasAdd(ExprMutator):
    """Cast linear layers in graph to integers and insert the necessary cast operations (from MATCH ONNX file)
    """

    def __init__(self):
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        """Rewrite ops
        """
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        # Default case
        new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
        span_name=""
        if hasattr(call,"span") and hasattr(call.span,"source_name"):
            span_name = call.span.source_name.name

        if span_name=="match_onnx_bias_cast":
            new_args_attrs=dict(new_args[0].attrs)
            new_args_attrs["out_dtype"]=call.attrs.dtype
            new_args[0]=relay.frontend.common.get_relay_op(new_args[0].op.name)(new_args[0].args[0], new_args[0].args[1], **new_args_attrs)
            new_args[0]=relay.frontend.common.set_span(new_args[0],"match_onnx_bias_cast_removed_added_out_dtype")
            new_call=new_args[0]
        
        if span_name=="match_onnx_bias_add":
            new_call=relay.op.nn.bias_add(new_args[0],relay.const(new_args[1].data.numpy().flatten()))
            new_call=relay.frontend.common.set_span(new_call,"match_onnx_bias_add")
        return new_call
    
@transform.function_pass(opt_level=0)
class MatchOnnxBiasAddRemoveFromMain(ExprMutator):
    """Cast linear layers in graph to integers and insert the necessary cast operations (from MATCH ONNX file)
    """

    def __init__(self):
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(mod.functions[mod.get_global_var("main")])

    def visit_call(self, call):
        """Rewrite ops
        """
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        # Default case
        new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
        span_name=""
        if hasattr(call,"span") and hasattr(call.span,"source_name"):
            span_name = call.span.source_name.name

        if span_name=="match_onnx_bias_add":
            new_call=relay.op.nn.bias_add(relay.frontend.common.set_span(relay.op.cast(
                new_args[0],new_args[0].attrs["out_dtype"]),"match_onnx_bias_cast")
                ,new_args[1])
        return new_call
    
class AddCastInMainPatternCallback(DFPatternCallback):
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.conv2d = is_op("nn.conv2d")(wildcard(),wildcard())
        self.multiply = wildcard()
        self.pattern = is_op("multiply")(self.conv2d,self.multiply)

    def callback(self, pre, post, node_map):
        conv2d = node_map[self.conv2d][0]
        mult = node_map[self.multiply][0]
        return relay.op.multiply(relay.op.cast(conv2d, "int32"),mult)


@tvm.ir.transform.module_pass(opt_level=0)
class MatchAddCastInMain:
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        global_var=mod.get_global_var("main")
        func=mod.functions[global_var]
        func = rewrite(AddCastInMainPatternCallback(), func)
        mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)

@tvm.ir.transform.module_pass(opt_level=0)
class MatchSaveModule:
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        global_var=mod.get_global_var("main")
        func=mod.functions[global_var]
        relay_inputs=func.params
        #match_inputs=[{"name":inp_.name_hint,"size":prod(inp_.type_annotation.shape[1:]),"type":inp_.type_annotation.dtype,"prec":np.dtype(inp_.type_annotation.dtype).itemsize,"shape":[int(sh) for sh in inp_.type_annotation.shape]} for inp_ in relay_inputs]
        #match_output={"size":prod(func.ret_type.shape[1:]),"prec":np.dtype(func.ret_type.dtype).itemsize,"type":func.ret_type.dtype,"shape":[int(sh) for sh in func.ret_type.shape]}
        match_inputs = None
        match_output = None
        CompiledModule.define_compiled_module(mod=mod,match_inputs=match_inputs,match_output=match_output)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)
    
@tvm.ir.transform.module_pass(opt_level=0)
class MatchSaveRelay:

    def __init__(self,prefix_relay:str=""):
        super().__init__()
        self.prefix_relay=prefix_relay

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        add_save_relay(prefix=self.prefix_relay,mod=mod)
        return mod
    
    def __call__(self, mod):
        return self.transform_module(mod)
    
