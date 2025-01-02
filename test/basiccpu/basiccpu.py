
import os
import match
from match.model.model import MatchModel
from match.target.exec_module import ExecModule
from match.target.memory_inst import MemoryInst
from match.target.target import MatchTarget
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import wildcard, is_op, is_var, is_constant
from match.partition.partitioning_pattern import PartitioningPattern
import ctypes
from pathlib import Path
import numpy as np
from numpy import typing as npt
from typing import Dict, Optional, Tuple


class BasicCpu(ExecModule):
    def __init__(self):
        super(BasicCpu, self).__init__(name="basiccpu",
                                          specific_patterns=[
                                              "vec_dense",
                                              "vec_conv"
                                          ],
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include")
    
    def partitioning_patterns(self):

        def vec_conv_pt():
            #Create pattern for a 2D Conv block, with bias and ReLU.
            conv2d = is_op("nn.conv2d")(
                wildcard(), wildcard()
            )
            bias_add = is_op("nn.bias_add")(conv2d, wildcard())
            relu = is_op("nn.relu")(bias_add)
            return relu
        
        def vec_dense_pt():
            """Create pattern for conv2D with optional fused relu."""
            dense = is_op("nn.dense")(
                wildcard(), wildcard()
            )
            bias_add = is_op("nn.bias_add")(dense, wildcard())
            relu = is_op("nn.relu")(bias_add)
            return relu
        
        return [
            PartitioningPattern(name="vec_dense",pattern=vec_dense_pt,ordered_operation="nn.dense"),
            PartitioningPattern(name="vec_conv",pattern=vec_conv_pt,ordered_operation="nn.conv2d"),
        ]

    def memories_def(self, pattern_name, operands):
        return [
            # from lower level to higher level memories
            MemoryInst(name="L1_CACHE",operands=["I","W","O"],k_bytes=128),
            MemoryInst(name="L2_CACHE",operands=["I","W","O"],k_bytes=4196),
            #MemoryInst(name="NEOPTEX_L3_CACHE",k_bytes=128),
            #MemoryInst(name="NEOPTEX_DRAM",k_bytes=4196),
        ]
    
class BasicCpuTarget(MatchTarget):
    def __init__(self):
        super(BasicCpuTarget,self).__init__([
            BasicCpu(),
        ],name="basiccpu_example")
        self.cpu_type = "arm_cpu"
        self.static_mem_plan = False
        self.static_mem_plan_algorithm = "hill_climb"

def numpy_to_array(np_arr: npt.NDArray, dtype: str):
    """ CPulpTargetnvert a numpy array to a TVM array with datatype `dtype`.
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

def create_random_array(shape: Tuple[int, ...], dtype: str) -> tvm.nd.array:
    """
    Generate random interger weights with numpy and converts them to a TVMArray with requested dtype.
    :param shape: tuple of ints that indicates size of array
    :param dtype: datatype that indicates the data type of the array
    :return: array which was loaded or created

    NOTE: The random data is integer, uniformely distributed and ranges from
    minimum to maximum depending on the data type:
    E.g. in8 --> [-128, 127]
    """
    def get_dtype_range():
        try:
            dtype_min = np.iinfo(dtype).min
            dtype_max = np.iinfo(dtype).max
        except ValueError:
            range_map = {
                'int4': (-8, 7),
                'int2': (-1, 1)     # technically this should be (-2, 1), but we prefer to not use -2
            }
            try:
                dtype_min, dtype_max = range_map[dtype]
            except KeyError:
                raise ValueError(f"Creating an array of dtype {dtype} is not supported")

        return dtype_min, dtype_max

    dtype_min, dtype_max = get_dtype_range()
    np_dtype = dtype
    if dtype in ['int4', 'int2']:
        np_dtype = 'int8'
    np_array = np.random.randint(low=dtype_min, high=dtype_max+1,
                                 size=shape, dtype=np_dtype)
    return numpy_to_array(np_array, dtype)

def create_vec_conv_ex(inp_shape:Tuple=(32,32),fil_shape:Tuple=(1,1),
                       padding:Tuple=(0,0,0,0),strides:Tuple=(1,1),
                       groups:int=1,out_ch:int=1,inp_ch:int=3):
    np.random.seed(0)
    x = relay.var("input_0", relay.TensorType((1,inp_ch)+inp_shape, "int8"))
    # Get or generate weight_values
    weights = create_random_array((out_ch,inp_ch)+fil_shape,"int8")
    # Get or generate bias values
    bias = create_random_array((out_ch,), "int32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "vec_conv_weights"
    bias_name = "vec_conv_bias"

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights.shape, weights.dtype))

    # define weights and bias values in params
    params = {weights_name: weights, bias_name: bias}

    # define operations
    x = relay.op.nn.conv2d(x, w,
                           strides=strides,
                           padding=padding,
                           groups=groups,
                           kernel_size=fil_shape,
                           out_dtype="int32",
                           )
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=0)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params
    

def create_vec_dense_ex(inp_features:int=256,out_features:int=128):
    """Generate a small network in TVM Relay IR that performs a requantized convolution
    """
    np.random.seed(0)
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType((1,inp_features), "int8"))
    # Get or generate weight_values
    weights = create_random_array((out_features,inp_features),"int8")
    # Get or generate bias values
    bias = create_random_array((out_features,), "int32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "vec_dense_weights"
    bias_name = "vec_dense_bias"

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights.shape, weights.dtype))

    # define weights and bias values in params
    params = {weights_name: weights, bias_name: bias}

    # define operations
    x = relay.op.nn.dense(x, w, out_dtype=bias.dtype)
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=-1)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def save_model_and_params(mod,params):
    if not Path("./model_save").is_dir():
        Path("./model_save").mkdir()
    with open(str(Path("./model_save/model_graph.relay").absolute()),"w") as mod_file:
        mod_file.write(relay.astext(mod))
    with open(str(Path("./model_save/model_params.txt").absolute()),"wb") as par_file:
        par_file.write(relay.save_param_dict(params=params))

def run_model(conv_ex:bool=False,output_path:str="./model_build"):
    if conv_ex:
        mod,params = create_vec_conv_ex(strides=(2,2),padding=(1,1,1,1),fil_shape=(3,3))
    else:
        mod,params = create_vec_dense_ex()
    save_model_and_params(mod=mod,params=params)
    #define HW Target inside match
    target=BasicCpuTarget()
    match.match(models_to_compile=[MatchModel(relay_mod=mod,relay_params=params)],target=target,output_path=output_path)

def run_relay_saved_model_at(mod_file,params_file,output_path):
    #define HW Target inside match
    target=BasicCpuTarget()
    match.match(input_type="relay",filename=mod_file,params_filename=params_file,target=target,output_path=output_path)

if __name__=="__main__":
    # if Path("./model_save").is_dir() and \
    #     Path("./model_save/model_graph.relay").exists() and \
    #         Path("./model_save/model_params.txt").exists():
    #     run_relay_saved_model_at(mod_file=str(Path("./model_save/model_graph.relay").absolute()),
    #                              params_file=str(Path("./model_save/model_params.txt").absolute()),
    #                              output_path=str(Path("./model_build").absolute()))
    # else:
    #     run_model(output_path=str(Path("./model_build").absolute()))
    run_model(conv_ex=True,output_path=str(Path("./model_build").absolute()))