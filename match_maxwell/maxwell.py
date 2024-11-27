
import os
from typing import Dict
import match
from match.target.exec_module import ComputationalApis, ExecModule
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

class MaxwellMemAcc(ExecModule):
    def __init__(self):
        super(MaxwellMemAcc, self).__init__(name="maxwell",
                                          specific_patterns=[
                                              "conv2d_biasadd_relu"
                                          ],
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include")
        self.NUM_BANKS = 2
        self.module_options["MAXWELL_NUM_BANKS"] = self.NUM_BANKS


    def optimal_spatial_mapping_def(self, pattern_name: str = "gap9cluster_conv2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        # k out channels
        # c in channels
        # oy out height
        # ox out width
        return [
            ("K",32), ("OY",self.NUM_BANKS)
        ]
    
    
    def partitioning_patterns(self):


        def conv2d_pattern():
            """Create pattern for conv2D with optional fused relu."""
            conv2d = is_op("nn.conv2d")(
                wildcard(), wildcard()
            )
            bias_add = is_op("nn.bias_add")(conv2d, wildcard())
            relu = is_op("nn.relu")(bias_add)
            return relu
        
        return [
            PartitioningPattern(name="conv2d_biasadd_relu",pattern=conv2d_pattern,ordered_operation="nn.conv2d"),
        ]


    def def_include_list(self,patter_name):
        return ["maxwell.h"]


    def comp_apis_def(self,comp_apis: ComputationalApis=ComputationalApis()):
        comp_apis.innermost_computation="maxwell_computation"
        return comp_apis

    def memories_def(self, pattern_name, operands):
        return [
            # from lower level to higher level memories
            MemoryInst(name="COMPUTE_MEMORY",k_bytes=self.NUM_BANKS*4,operands=operands),
            MemoryInst(name="INTER_LAYER_MEMORY",k_bytes=128,operands=operands,r_ports=1,w_ports=1,rw_ports=0),
            #MemoryInst(name="EXT_MEM",k_bytes=2406,operands=operands)
        ]
    
    def mem_apis_def(self, memory_apis = ...):
        memory_apis.mem_transfer["I"]="maxwell_load_activations"
        memory_apis.copy_out_curr_computation="maxwell_compute_and_store"
        return memory_apis


class MaxwellTarget(MatchTarget):
    def __init__(self):
        super(MaxwellTarget,self).__init__([
            MaxwellMemAcc(),
        ],name="maxwell")

def numpy_to_array(np_arr: npt.NDArray, dtype: str):
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

def relay_conv2d(input_tensor: relay.Var, layer_name: str,
                      w_value: tvm.nd.array,
                      b_value: tvm.nd.array,
                      strides: Tuple[int, ...] = (1, 1),
                      padding: Tuple[int, ...] = (0, 0),
                      groups: int = 1,
                      shift_bits: int = 0,
                      ) -> Tuple[relay.Var,
                                                    Dict[relay.Expr,
                                                         tvm.nd.array]]:
    # define weights and bias variables
    weights_name = layer_name + '_weights'
    bias_name = layer_name + '_bias'

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(w_value.shape, w_value.dtype))

    # define weights and bias values in params
    params = {weights_name: w_value}

    # define operations
    x = relay.op.nn.conv2d(input_tensor, w,
                           strides=strides,
                           padding=padding,
                           groups=groups,
                           kernel_size=(w_value.shape[2],w_value.shape[3]),
                           #out_dtype="int32" if b_value is not None else b_value.dtype
                           )
    params = {weights_name: w_value, bias_name: b_value}
    b = relay.var(bias_name, relay.TensorType(b_value.shape, b_value.dtype))
    x = relay.op.nn.bias_add(x, b)
    return x, params


def create_model_conv_2d(prec_bits: int = 16,
                 input_shape: Tuple[int, ...] = (1, 3, 16, 16),
                 weights_shape: Tuple[int, ...] = (3, 3, 1, 1),
                 weights_values: Optional[npt.NDArray] = None,
                 bias_values: Optional[npt.NDArray] = None,
                 padding: Tuple[int, int] = (0, 0),
                 strides: Tuple[int, int] = (1, 1),
                 depthwise: bool = False,
                 relu: bool = True,
                 ):
    """Generate a small network in TVM Relay IR that performs a requantized convolution
    """
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType(input_shape, f'int{prec_bits}'))
    # Get or generate weight_values
    if weights_values is None:
        weights = create_random_array(weights_shape, 
                                            f'int{prec_bits}')
    else:
        weights = numpy_to_array(weights_values,weights_values.dtype.name)
    # Get or generate bias values
    if bias_values is None:
        bias = create_random_array(weights_shape[0], f"int{prec_bits}")
    else:
        bias = numpy_to_array(bias_values,bias_values.dtype.name)
    # Generate the conv2d call
    x, params1 = relay_conv2d(x, 'conv1', weights, bias, 
                                         padding=padding, 
                                         strides=strides,
                                         groups=weights_shape[0] if depthwise else 1,
                                         )
    if relu:
        x = relay.op.nn.relu(x)
    #if input_pad is not None:
    #    x= relay.reshape(x,(16))
    params = params1
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

def run_model(output_path):
    mod,params = create_model_conv_2d()
    save_model_and_params(mod=mod,params=params)
    #define HW Target inside match
    target=MaxwellTarget()
    res=match.match(relay_mod=mod,relay_params=params,target=target,output_path=output_path)

def run_relay_saved_model_at(mod_file,params_file,output_path):
    #define HW Target inside match
    target=MaxwellTarget()
    res=match.match(input_type="relay",filename=mod_file,params_filename=params_file,target=target,output_path=output_path)

if __name__=="__main__":
    if Path("./model_save").is_dir() and \
        Path("./model_save/model_graph.relay").exists() and \
            Path("./model_save/model_params.txt").exists():
        run_relay_saved_model_at(mod_file=str(Path("./model_save/model_graph.relay").absolute()),
                                 params_file=str(Path("./model_save/model_params.txt").absolute()),
                                 output_path=str(Path("./model_build").absolute()))
    else:
        run_model(output_path=str(Path("./model_build").absolute()))