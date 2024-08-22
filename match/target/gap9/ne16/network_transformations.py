import ctypes
import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.dataflow_pattern import DFPatternCallback, rewrite, wildcard, is_op, is_constant


class Gap9NE16OnnxDigitalRequantRewriter(DFPatternCallback):
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
class Gap9NE16OnnxRequantTransform:
    """ Find and rewrite MATCH ONNX requant to requant for internal use:
        div->div->floor->max->min to
        right_shift->clip->cast
    """
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(Gap9NE16OnnxDigitalRequantRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)


@transform.function_pass(opt_level=0)
class Gap9NE16OnnxIntegerize(ExprMutator):
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


def network_transformations(opts):
    pipeline=[]
    pipeline.append(RequantRewriterPlinioOnnx())  
    #if 'requant_transform' not in opts or opts['requant_transform'] != '0':
    #    pipeline.append(Gap9ClusterOnnxRequantTransform())   
    #pipeline.append(Gap9ClusterOnnxIntegerize('uint8'))
    return pipeline

def np_to_tvm_arr(np_arr, dtype: str):
    """ Convert a numpy array to a TVM array with datatype `dtype`.
    Although such a function exists in TVM, it does not support creating TVM arrays with dtypes
    that are not supported in numpy, like 'int4' or 'int2'.
    :param np_arr: the given numpy array
    :param dtype:  the resulting data type of the TVM array
    :return: the TVM array
    """
    assert np_arr.flags["C_CONTIGUOUS"]

    arr = tvm.nd.empty(np_arr.shape, dtype)
    data = np_arr.ctypes.data_as(ctypes.c_void_p)
    nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
    tvm.nd.check_call(tvm.nd._LIB.TVMArrayCopyFromBytes(arr.handle, data, nbytes))

    return arr

@transform.function_pass(opt_level=0)
class GapPadTransform(ExprMutator):
    """Insert match specific layout transform before and after each 'match' annotated relay Function
    TODO: make this smart to avoid unnecessary transformations
    """

    MAP_OLD_TO_PADDED = dict()
    def transform_function(self, func, mod, ctx):
        return self.visit(func)


    def visit_pad_func_body(self,call,params,is_dw):
        # Generate new variable.
        if isinstance(call, relay.Constant):
            return self.visit(call)
        elif call.op.name=="nn.conv2d":
            #breakpoint()
            new_fn = self.visit(call.op)
            new_args = params
            return relay.op.nn.conv2d(new_args[0], new_args[1],
                                          strides=call.attrs.strides,
                                          padding=call.attrs.padding,
                                          dilation=call.attrs.dilation,
                                          groups=int(new_args[0].type_annotation.shape[1]) if is_dw else call.attrs.groups,
                                          out_dtype='int32',
                                          kernel_size=new_args[1].type_annotation.shape[-2:])
            #return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
        else:
            new_fn = self.visit(call.op)
            #breakpoint()
            if is_dw and call.op.name!="right_shift":
                new_args = list()
                for arg in call.args:
                    if isinstance(arg,relay.Constant):
                        if len(arg.checked_type.shape)>1:
                            new_args.append(relay.const(np_to_tvm_arr(np.pad(arg.data.numpy(),((0,0),(0,16-(int(arg.checked_type.shape[1])%16)),(0,0),(0,0))),dtype=arg.checked_type.dtype), dtype=arg.checked_type.dtype))
                        else:
                            new_args.append(relay.const(np_to_tvm_arr(np.pad(arg.data.numpy(),((0,16-(int(arg.checked_type.shape[0])%16)))),dtype=arg.checked_type.dtype), dtype=arg.checked_type.dtype))
                    else:
                        new_args.append(self.visit_pad_func_body(arg,params=params,is_dw=is_dw))
            else:
                new_args = [self.visit_pad_func_body(arg,params=params,is_dw=is_dw) for arg in call.args]
            return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

    def visit_pad_func(self,fn,is_dw):
        """Rewrite function arguments
        """
        new_params = []
        binds = {}
        for idx,param in enumerate(fn.params):
            # Get the parameter's type annotation.
            var_type = param.type_annotation

            # Generate new variable.
            if idx==1 and is_dw:
                new_param = relay.var(param.name_hint, shape=var_type.shape if len(var_type.shape)!=4 or var_type.shape[0]%16==0 else [var_type.shape[0]+16-(var_type.shape[0]%16),var_type.shape[1],var_type.shape[2],var_type.shape[3]], dtype=var_type.dtype)
            else:
                new_param = relay.var(param.name_hint, shape=var_type.shape if len(var_type.shape)!=4 or var_type.shape[1]%16==0 else [var_type.shape[0],var_type.shape[1]+16-(var_type.shape[1]%16),var_type.shape[2],var_type.shape[3]], dtype=var_type.dtype)

            new_params.append(new_param)
            binds[param] = new_param

        new_body = self.visit_pad_func_body(fn.body,new_params,is_dw)
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
    
    def visit_pad_call_args(self,arg,old_arg,is_dw,idx):
        # Generate new variable.
        #breakpoint()
        if idx==0:
            if int(old_arg.checked_type.shape[1])%16==0:
                return arg
            else:
                #if isinstance(old_arg,relay.Call) and old_arg.op.name == 'annotation.compiler_begin' and isinstance(old_arg.args[0],relay.Var):
                #    param = old_arg.args[0]
                #    var_type = old_arg.args[0].type_annotation
                #    if is_dw:
                #        self.NEW_VAR_ARGS.append(relay.var(param.name_hint, shape=var_type.shape if len(var_type.shape)!=4 or var_type.shape[0]%16==0 else [var_type.shape[0]+16-(var_type.shape[0]%16),var_type.shape[1],var_type.shape[2],var_type.shape[3]], dtype=var_type.dtype))
                #    else:
                #        self.NEW_VAR_ARGS.append(relay.var(param.name_hint, shape=var_type.shape if len(var_type.shape)!=4 or var_type.shape[1]%16==0 else [var_type.shape[0],var_type.shape[1]+16-(var_type.shape[1]%16),var_type.shape[2],var_type.shape[3]], dtype=var_type.dtype))
                #else:    
                new_arg=relay.reshape(relay.nn.pad(relay.reshape(arg.args[0],(old_arg.checked_type.shape[0],old_arg.checked_type.shape[2],old_arg.checked_type.shape[3],old_arg.checked_type.shape[1])),((0,0),(0,0),(0,0),(0,16-(int(old_arg.checked_type.shape[1])%16)))),(old_arg.checked_type.shape[0],old_arg.checked_type.shape[1]+16-(int(old_arg.checked_type.shape[1])%16),old_arg.checked_type.shape[2],old_arg.checked_type.shape[3]))
                new_fn = self.visit(arg.op)
                return relay.Call(new_fn,[new_arg] , arg.attrs)
        if isinstance(arg, relay.Call) and isinstance(arg.args[0],relay.Constant):
            if int(old_arg.args[0].checked_type.shape[0 if is_dw else 1])%16==0:
                return arg
            else:
                new_arg = relay.const(np_to_tvm_arr(np.pad(arg.args[0].data.numpy(),((0,16-(int(old_arg.checked_type.shape[0])%16)),(0,0),(0,0),(0,0)) if is_dw else ((0,0),(0,16-(int(old_arg.checked_type.shape[1])%16)),(0,0),(0,0))),dtype=old_arg.checked_type.dtype), dtype=old_arg.checked_type.dtype)
                new_fn = self.visit(arg.op)
                return relay.Call(new_fn,[new_arg] , arg.attrs)
        else:
            return self.visit(arg)

    def is_dw(self,call):
        conv = call
        while conv.op.name!="nn.conv2d":
            conv = conv.args[0]
        return conv.attrs.groups==conv.args[0].checked_type.shape[1] and conv.attrs.groups > 1


    def visit_pad_call(self,call,old_call):
        #breakpoint()
        if old_call in self.MAP_OLD_TO_PADDED:
            return self.MAP_OLD_TO_PADDED[old_call]
        #breakpoint()
        is_dw = self.is_dw(call.op.body)
        #if is_dw:
        #    return call,(None,None,None)
        new_fn = self.visit_pad_func(call.op,is_dw)
        new_args = [self.visit_pad_call_args(arg,old_call.args[idx],is_dw,idx) for idx,arg in enumerate(call.args)]
        final_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
        original_out_shape=None
        padded_shape=None
        strided_shape=None
        if is_dw:
            original_out_shape=[int(x) for x in old_call.checked_type.shape]
            padded_shape=[original_out_shape[0],original_out_shape[2],original_out_shape[3],original_out_shape[1]+(16-(original_out_shape[1]%16))]
            strided_shape=[original_out_shape[0],original_out_shape[2],original_out_shape[3],original_out_shape[1]]
        self.MAP_OLD_TO_PADDED[old_call]=(final_call,(original_out_shape,padded_shape,strided_shape))
        return self.MAP_OLD_TO_PADDED[old_call]


    def visit_call(self, call):
        """Rewrite ops
        """
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        if isinstance(call.op, tvm.ir.Op) and not isinstance(call.args[0], relay.Constant)\
            and call.op.name == 'annotation.compiler_end' and call.attrs.compiler == 'match'\
                and call.args[0].op.attrs["Composite"].split(".")[2]=="NE16"\
                    and len(call.args[0].args[0].checked_type.shape)==4 and (call.args[0].args[0].checked_type.shape[1]%16)!=0:
            # insert transformation after this op
            new_args_ = []
            new_arg,(orig,padd,strid)=self.visit_pad_call(new_args[0],call.args[0])
            new_args_.append(new_arg)
            new_args_+=new_args[1:]
            new_call = relay.Call(new_fn, new_args_, call.attrs, call.type_args, call.span)
            if orig is not None:
                new_call = relay.reshape(relay.strided_slice(relay.reshape(new_call,tuple(padd)),
                                            begin=(0,0,0,0),
                                            end=tuple(strid)),
                                            tuple(orig))
        return new_call
    


def adjust_network(opts):
    pipeline=[]
    pipeline.append(transform.InferType())
    pipeline.append(GapPadTransform())
    pipeline.append(transform.InferType())
    # TODO: add a transformation that after the padding one deletes useless padding on the inputs, if not
    # used elsewhere not padded and then changes directly the input, getting then the same thing as statically padding
    #pipeline.append(GapRemovePadOnInputs())
    #pipeline.append(transform.InferType())
    return pipeline
