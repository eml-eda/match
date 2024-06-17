import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.dataflow_pattern import DFPatternCallback, rewrite, wildcard, is_op, is_constant


class Gap9ClusterOnnxDigitalRequantRewriter(DFPatternCallback):
    """Rewriter for digital requant pattern
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)

        self.x = wildcard()
        self.div1 = is_constant()
        self.div2 = is_constant()
        self.maximum = is_constant()
        self.minimum = is_constant()

        cast = is_op("cast")(self.x)
        div1 = is_op("divide")(cast, self.div1)
        div2 = is_op("divide")(div1, self.div2)
        floor = is_op("floor")(div2)
        maximum = is_op("maximum")(floor, self.maximum)
        minimum = is_op("minimum")(maximum, self.minimum)
        self.pattern = is_op("cast")(minimum)

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        div1 = node_map[self.div1][0]
        div2 = node_map[self.div2][0]
        maximum = node_map[self.maximum][0]
        minimum = node_map[self.minimum][0]

        shift_factor = int(np.log2(div1.data.numpy() * div2.data.numpy()))

        x = relay.op.right_shift(x, relay.const(shift_factor))
        x = relay.op.clip(x, a_min=int(maximum.data.numpy()), a_max=int(minimum.data.numpy()))
        return relay.op.cast(x, 'uint8')

@tvm.ir.transform.module_pass(opt_level=0)
class Gap9ClusterOnnxRequantTransform:
    """ Find and rewrite MATCH ONNX requant to requant for internal use:
        div->div->floor->max->min to
        right_shift->clip->cast
    """
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(Gap9ClusterOnnxDigitalRequantRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)


@transform.function_pass(opt_level=0)
class Gap9ClusterOnnxIntegerize(ExprMutator):
    """Cast linear layers in graph to integers and insert the necessary cast operations (from MATCH ONNX file)
    """

    def __init__(self, dtype):
        self.dtype = dtype
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

        if call.op.name == 'nn.conv2d' and new_args[1].data.dtype.startswith('int'):
            # ensure that the output of the conv2d op is int32
            w = new_args[1]
            new_call = relay.op.nn.conv2d(new_args[0], w,
                                          strides=call.attrs.strides,
                                          padding=call.attrs.padding,
                                          dilation=call.attrs.dilation,
                                          groups=call.attrs.groups,
                                          out_dtype='int32',
                                          kernel_size=w.data.shape[-2:])

        elif call.op.name == 'nn.dense' and new_args[1].data.dtype.startswith('int'):
            # ensure that the output of the dense op is int32
            new_call = relay.op.nn.dense(new_args[0], new_args[1], out_dtype='int32')

        elif call.op.name == 'nn.bias_add' or call.op.name == 'add':
            # ensure bias data type matches the data type of previous operation's output type
            # make sure to eliminate element-wise add, so check if rhs is constant
            new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
            if isinstance(new_args[1], relay.Constant):
                if isinstance(new_args[0],relay.Var):
                    dtype=new_args[0].checked_type.dtype
                    #dtype="uint8"
                else:
                    dtype = new_args[0].attrs.out_dtype if new_args[0].attrs is not None and hasattr(new_args[0].attrs,"out_dtype") else "int32"
                new_args[1] = relay.const(new_args[1].data.numpy().astype(dtype if dtype!="" else new_args[1].checked_type.dtype))
                new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        elif call.op.name == 'multiply' or call.op.name == "add":
            new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
            if isinstance(new_args[1], relay.Constant):
                if isinstance(new_args[0],relay.Var):
                    dtype=new_args[0].checked_type.dtype
                    #dtype="uint8"
                else:
                    dtype = new_args[0].attrs.out_dtype if new_args[0].attrs is not None and hasattr(new_args[0].attrs,"out_dtype") else "int32"
                new_args[1] = relay.const(new_args[1].data.numpy().astype(dtype if dtype!="" else new_args[1].checked_type.dtype))
                new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        elif call.op.name == 'divide':
            # a divide operation with division factor > 1 that is a power of two, is assumed to be a dequant op
            # put cast before this op in that case
            x = new_args[0]
            div = new_args[1].data.numpy().item()
            if div >= 1 and np.log2(div).is_integer():
                x = relay.cast(x, 'float')
            new_call = relay.divide(x, new_args[1])

        elif call.op.name == 'minimum':
            # test if this is the last layer of the quantize sequence, if so, put cast after this op
            new_call = relay.minimum(new_args[0], new_args[1])
            if new_args[0].op.name == "maximum" and \
               new_args[0].args[0].op.name == "floor" and \
               new_args[0].args[0].args[0].op.name == "divide":
                new_call = relay.cast(new_call, self.dtype)

        return new_call

    def visit_function(self, fn):
        """Rewrite function arguments
        """
        new_params = []
        binds = {}

        for param in fn.params:
            # Get the parameter's type annotation.
            var_type = param.type_annotation

            # bias params are int32
            if param.name_hint.endswith('bias'):
                dtype = 'int32'
            else:
                dtype = self.dtype

            # Generate new variable.
            new_param = relay.var(param.name_hint, shape=var_type.shape, dtype=dtype)

            new_params.append(new_param)
            binds[param] = new_param

        new_body = self.visit(fn.body)
        # Rewrite the body to use new parameters.
        new_body = relay.bind(new_body, binds)

        # Construct the updated function and return.
        return relay.Function(
            new_params,
            new_body,
            # You could change the return type, if you use None it will re-infer.
            None,
            type_params=fn.type_params,
            attrs=fn.attrs,
        )


class FindLayoutTransformShape(ExprVisitor):
    """Convert relay graph to dory graph
    """
    def __init__(self):
        super().__init__()
        self.shapes = []

    def visit_call(self, call):
        """Extract parameters and construct dory graph"""
        self.visit(call.op)
        for a in call.args:
            self.visit(a)

        if isinstance(call.op, tvm.ir.Op) and not isinstance(call.args[0], relay.Constant):
            # we don't want to insert transformations on constants like weights and biases
            if call.op.name == 'annotation.compiler_begin' and call.attrs.compiler == 'match':
                self.shapes.append(call.args[0].checked_type.shape)

            elif call.op.name == 'annotation.compiler_end' and call.attrs.compiler == 'match':
                self.shapes.append(call.args[0].checked_type.shape)

class DivFloorPlinioOnnx(DFPatternCallback):
    """Rewriter for digital requant pattern
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)

        self.div = is_op("divide")(wildcard(),is_constant())
        self.floor = is_op("floor")(self.div)
        self.clip = is_op("clip")(self.floor)
        self.cast = is_op("cast")(self.clip)
        self.pattern = self.cast

    def callback(self, pre, post, node_map):
        div = node_map[self.div][0]
        cast = node_map[self.cast][0]

        shift_factor = int(np.log2(abs(int(div.args[1].data.numpy()))))

        x = relay.op.right_shift(div.args[0], relay.const(shift_factor))
        x = relay.op.clip(x, a_min=int(0), a_max=int(255))
        return relay.op.cast(x, cast.attrs["dtype"])
    
class DivReqPlinioOnnx(DFPatternCallback):
    """Rewriter for digital requant pattern
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)

        self.div = is_op("divide")(wildcard(),is_constant())
        self.clip = is_op("clip")(self.div)
        self.cast = is_op("cast")(self.clip)
        self.pattern = self.cast

    def callback(self, pre, post, node_map):
        div = node_map[self.div][0]
        cast = node_map[self.cast][0]

        shift_factor = int(np.log2(abs(int(div.args[1].data.numpy()))))

        x = relay.op.right_shift(div.args[0], relay.const(shift_factor))
        x = relay.op.clip(x, a_min=int(0), a_max=int(255))
        return relay.op.cast(x, cast.attrs["dtype"])

@tvm.ir.transform.module_pass(opt_level=0)
class RequantRewriterPlinioOnnx:
    """ Find and rewrite MATCH ONNX requant to requant for internal use:
        div->div->floor->max->min to
        right_shift->clip->cast
    """
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(DivFloorPlinioOnnx(), func)
            func = rewrite(DivReqPlinioOnnx(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)


class FindLayoutTransformShape(ExprVisitor):
    """Convert relay graph to graph
    """
    def __init__(self):
        super().__init__()
        self.shapes = []

    def visit_call(self, call):
        """Extract parameters and construct graph"""
        self.visit(call.op)
        for a in call.args:
            self.visit(a)

        if isinstance(call.op, tvm.ir.Op) and not isinstance(call.args[0], relay.Constant):
            # we don't want to insert transformations on constants like weights and biases
            if call.op.name == 'annotation.compiler_begin' and call.attrs.compiler == 'default':
                self.shapes.append(call.args[0].checked_type.shape)

            elif call.op.name == 'annotation.compiler_end' and call.attrs.compiler == 'default':
                self.shapes.append(call.args[0].checked_type.shape)

@transform.function_pass(opt_level=0)
class GapLayoutTransform(ExprMutator):
    """Insert match specific layout transform before and after each 'match' annotated relay Function
    TODO: make this smart to avoid unnecessary transformations
    """

    def transform_function(self, func, mod, ctx):
        self.f = FindLayoutTransformShape()
        self.f.visit(func)

        return self.visit(func)

    def create_transform(self, x, shape, end, flatten:bool=False):
        """NHWC to NCHW transformations
        """
        if not end and len(shape)==4 and (shape[2]>1 or shape[3]>1):
            #sono in realtà in HWC Ma per TVM In CHW 
            x = relay.reshape(x,(shape[0],shape[2],shape[3],shape[1]))
            x = relay.op.transpose(x,(0,3,1,2))
        if end and len(shape)==4 and (shape[2]>1 or shape[3]>1):
            # sono in CHW sia per me che per TVM ma voglio ripassare a HWC
            #x = relay.reshape(x,(shape[0],shape[1],shape[3],shape[1]))
            x = relay.op.transpose(x,(0,2,3,1))
            x = relay.reshape(x,(shape[0],shape[1],shape[2],shape[3]))
            
        return x

    def visit_call(self, call):
        """Rewrite ops
        """
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        if isinstance(call.op, tvm.ir.Op) and not isinstance(call.args[0], relay.Constant):
            # we don't want to insert transformations on constants like weights and biases
            if call.op.name == 'annotation.compiler_begin' and call.attrs.compiler == 'default':
                # insert transformation before this op
                shape = self.f.shapes.pop(0)
                if len(call.args)>0 and isinstance(call.args[0],tvm.relay.Call) and call.args[0].op.name=="annotation.compiler_end" and call.args[0].attrs.compiler=="match":
                    x = self.create_transform(new_args[0], shape, False)
                    #new_call = relay.op.annotation.compiler_begin(x, 'default-reshape')
                else:
                    x = new_args[0]
                new_call = relay.op.annotation.compiler_begin(x, 'default')

            elif call.op.name == 'annotation.compiler_end' and call.attrs.compiler == 'default':
                # insert transformation after this op
                shape = self.f.shapes.pop(0)
                new_call = self.create_transform(new_call, shape, True)

        return new_call

class Gap9ClusterFlattenRewriter(DFPatternCallback):
    """Rewriter for digital requant pattern
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.x = wildcard()
        #self.maxpool2d = is_op("nn.max_pool2d")(wildcard())
        transpose = is_op("transpose")(self.x)
        reshape = is_op("reshape")(transpose)
        comp = is_op("annotation.compiler_begin")(reshape)
        self.pattern = is_op("annotation.compiler_end")(is_op("nn.batch_flatten")(comp))
        

    def callback(self, pre, post, node_map):
        #max_pool = node_map[self.maxpool2d][0]
        x = node_map[self.x][0]
        return relay.nn.batch_flatten(x)

@tvm.ir.transform.module_pass(opt_level=0)
class Gap9BatchFlattenTransform:
    """ Find and rewrite MATCH ONNX requant to requant for internal use:
        div->div->floor->max->min to
        right_shift->clip->cast
    """
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(Gap9ClusterFlattenRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)


def network_transformations(opts):
    pipeline=[]
    #pipeline.append(transform.ConvertLayout({'nn.conv2d': ['NHWC']}))
    pipeline.append(RequantRewriterPlinioOnnx())  
    #if 'requant_transform' not in opts or opts['requant_transform'] != '0':
    #    pipeline.append(Gap9ClusterOnnxRequantTransform())   
    pipeline.append(Gap9ClusterOnnxIntegerize('uint8'))
    return pipeline

def adjust_network(opts):
    pipeline=[]
    pipeline.append(GapLayoutTransform())
    pipeline.append(Gap9BatchFlattenTransform())
    pipeline.append(transform.InferType())
    return pipeline
