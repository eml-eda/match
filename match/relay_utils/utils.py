import pathlib
import tarfile
import shutil
import ctypes
import re
import os
import subprocess
import argparse
import tvm
import tvm.relay as relay
import numpy as np
from tvm.driver.tvmc.compiler import compile_model
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.backend import Executor, Runtime
import dory

from typing import Tuple, Dict, Optional, Union
import numpy.typing as npt
from dory.Hardware_targets.PULP.Backend_Kernels.BackendKernelsAdapter import PulpNNAdapter
from match.target.target import MatchTarget


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


def relay_gap9_layout_transform(x, shape):
    """
    Creates a relay layout transform that reverses chunks of four bytes
    """

    x = relay.reshape(x, (np.prod(shape) // 4, 4))
    x = relay.reverse(x, axis=1)
    x = relay.reshape(x, shape)

    return x


def simple_basic_type_checker(x,weights_shape):
    strides=[1,1]
    call=x
    while call is not None:
        if isinstance(call,tvm.relay.Var) or isinstance(call,relay.Constant):
            starting_shape=call.type_annotation.shape
            call = None
        elif hasattr(call,'args') and len(call.args) > 0:
            if hasattr(call.attrs,'strides'):
                strides[0]*=call.attrs.strides[0]
                strides[1]*=call.attrs.strides[1]
            call = call.args[0]
        else:
            starting_shape=call.type_annotation.shape
            call = None
    return [1,int(weights_shape[0]),int(starting_shape[2]//strides[0]),int(starting_shape[3]//strides[1])]

def relay_gap9_conv2d(input_tensor: relay.Var, layer_name: str,
                      w_value: tvm.nd.array,
                      b_value: tvm.nd.array,
                      strides: Tuple[int, ...] = (1, 1),
                      padding: Tuple[int, ...] = (0, 0),
                      groups: int = 1,
                      act: bool = False,
                      shift_bits: int = 0,
                      batchnorm = False) -> Tuple[relay.Var,
                                                    Dict[relay.Expr,
                                                         tvm.nd.array]]:
    '''
    Creates a relay conv2d op which is GAP9 compatible
    This means it can be offloaded to the accelerator.
    :param input_tensor: relay.Var for input
    :param layer_name: string that determines relay variable naming
    :param w_value: int8 tensor that contains weight values
    :param b_value: int32 tensor that contains bias values
    :param strides: tuple describing convolution stride (x,y)
    :param padding: tuple describing convolution padding
    :param act: bool that toggles extra ReLU to be added (see below)
    :shift_bits: int that sets amount of bits to shift right. Value must be between [0,31]
    :return: tuple that contains relay param dictionary and relay
        expr for the subgraph.

    The Relay code for one of these layers looks like this:
    ```
        %0 = qnn.conv2d(%input, %conv1.weights,...,out_dtype="int32");
        %1 = nn.bias_add(%0, %conv1.bias);
        %2 = right_shift(%1, 4);
        %3 = clip(%2, a_min=-128f, a_max=127f);
        %4 = cast(%3, dtype="int8");
    ```
    If `act` is set to `True` an additional ReLU-like clip is added
    ```
        %5 = clip(%4, a_min=0f, a_max=127f);
    ```

    This function returns the relay expression for this graph,
    along with a parameter dictionary
    '''
    # define weights and bias variables
    weights_name = layer_name + '.weights'
    bias_name = layer_name + '.bias'

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(w_value.shape, w_value.dtype))

    # define weights and bias values in params
    params = {weights_name: w_value}

    # define operations
    x = relay.op.nn.conv2d(input_tensor, w,
                           strides=strides,
                           padding=padding,
                           groups=groups,
                           out_dtype='int32')
    if batchnorm:
        input_shape=simple_basic_type_checker(input_tensor,w_value.shape)
        #input_shape = [int(x) for x in input_tensor.type_annotation.shape]
        lambda_name = layer_name + '.lambda'
        k_name = layer_name + '.k' 
        lambda_var=relay.var(lambda_name, relay.TensorType(input_shape, 'int32'))
        k_var=relay.var(k_name,  relay.TensorType(input_shape, 'int32'))
        params[lambda_name]=numpy_to_array(np.zeros(input_shape, dtype = np.int32), 'int32')
        params[k_name]=numpy_to_array(np.ones(input_shape, dtype=  np.int32), 'int32')
        #x = relay.op.nn.batch_norm()
        x = relay.op.multiply(x, k_var)
        x = relay.op.add(x, lambda_var)
    else:
        params = {weights_name: w_value, bias_name: b_value}
        b = relay.var(bias_name, relay.TensorType(b_value.shape, b_value.dtype))
        x = relay.op.nn.bias_add(x, b)
    x = relay.op.right_shift(x, relay.const(shift_bits))
    # Optional: ReLU
    if act:
        x = relay.op.clip(x, a_min=0, a_max=255)
    else:
        x = relay.op.clip(x, a_min=0, a_max=255)
    x = relay.op.cast(x, 'uint8')
    return x, params


def relay_gap9_dense(input_tensor: relay.Var, layer_name: str,
                     w_value: tvm.nd.array,
                     b_value: tvm.nd.array,
                     act: bool = False,
                     shift_bits: int = 0,
                     batchnorm: bool = False):
    """
    Creates a relay dense op which is gap9 compatible
    :param input_tensor: relay.Var for input
    :param layer_name: string that determines relay variable naming
    :param w_value: int8 tensor that contains weight values, must be of shape (num_inputs, num_outputs, 1, 1)
    :param b_value: int32 tensor that contains bias values
    :param act: bool that toggles extra ReLU to be added (see below)
    :shift_bits: int that sets amount of bits to shift right. Value must be between [0,31]
    """
    # define weights and bias variables
    weights_name = layer_name + '.weights'
    bias_name = layer_name + '.bias'
    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(w_value.shape, w_value.dtype))
    b = relay.var(bias_name, relay.TensorType(b_value.shape, b_value.dtype))

    # define weights and bias values in params
    params = {weights_name: w_value, bias_name: b_value}

    # define operations
    x = relay.op.nn.dense(input_tensor, w, out_dtype=b_value.dtype)
    # x = relay.op.nn.bias_add(x, b)
    if batchnorm:
        input_shape=simple_basic_type_checker(input_tensor,w_value.shape)
        #input_shape = [int(x) for x in input_tensor.type_annotation.shape]
        lambda_name = layer_name + '.lambda'
        k_name = layer_name + '.k' 
        lambda_var=relay.var(lambda_name, relay.TensorType(input_shape, 'int32'))
        k_var=relay.var(k_name,  relay.TensorType(input_shape, 'int32'))
        params[lambda_name]=numpy_to_array(np.zeros(input_shape, dtype = np.int32), 'int32')
        params[k_name]=numpy_to_array(np.ones(input_shape, dtype=  np.int32), 'int32')
        x = relay.op.multiply(x, k_var)
        x = relay.op.add(x, lambda_var)
    else:
        params = {weights_name: w_value, bias_name: b_value}
        b = relay.var(bias_name, relay.TensorType(b_value.shape, b_value.dtype))
        x = relay.op.nn.bias_add(x, b)

    x = relay.op.right_shift(x, relay.const(shift_bits))
    # Optional: ReLU
    if act:
        x = relay.op.clip(x, a_min=0, a_max=255)
    else:
        x = relay.op.clip(x, a_min=0, a_max=255)
    x = relay.op.cast(x, 'uint8')
    return x, params


def relay_gap9_add(input_tensor_a: relay.Var,
                   input_tensor_b: relay.Var,
                   layer_name: str,
                   shift_bits: int = 0):
    """
    Creates a relay element-wise-add op which is gap9 compatible
    :param input_tensor_a: relay.Var for input tensor A
    :param input_tensor_b: relay.Var for input tensor B
    :param layer_name: string that determines relay variable naming
    :shift_bits: int that sets amount of bits to shift right. Value must be between [0,31]
    """
    # define operations
    a = relay.op.cast(input_tensor_a, 'int32')
    b = relay.op.cast(input_tensor_b, 'int32')
    x = relay.op.add(a, b)
    x = relay.op.right_shift(x, relay.const(shift_bits))
    x = relay.op.clip(x, a_min=0, a_max=255)
    x = relay.op.cast(x, 'uint8')

    return x


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


def tvmc_wrapper(model: TVMCModel, target: str = "gap9, c",
                 fuse_layers: bool = True, 
                 package_path: pathlib.Path = pathlib.Path("model.tar")):
    '''
    Utility wrapper for TVMC that sets supported
    :param model: TVMC model that you wish to compile
    :param target: Can be "gap9, c" if you want to offload all possible
        computations to accelerator, and can be "c" for golden model checking.
    :param fuse_layers: sets relay.FuseOps.max_depth parameter to 1
        if set to False. This tells relay to not fuse operations.
        This can be useful when debuggin the TVM-generated c code kernels.
    '''
    # Check arguments
    #assert ((target == "gap9, c") or (target == "c"))
    # Add -device=arm_cpu as default device for TVM C codegen
    # This will use the arm_cpu relay strategy as opposed to the x86 one.
    target += " -device=arm_cpu"
    # This has to be set by default to use the C runtime
    pass_context_configs = ['tir.disable_vectorize=1']
    if not fuse_layers:
        pass_context_configs.append('relay.FuseOps.max_depth=1')
    compile_model(tvmc_model=model,
                  target=target,
                  executor=Executor("aot",
                                    {"interface-api": "c",
                                     "unpacked-api": 1}
                                    ),
                  runtime=Runtime("crt"),
                  output_format="mlf",
                  package_path=package_path,
                  pass_context_configs=pass_context_configs,
                  )


def tvmc_compile_and_unpack(model: TVMCModel, target: str = "gap9, c",
                            fuse_layers: bool = True,
                            build_path: str = "./build",
                            byoc_path: str = ".",
                            device="pulp"):
    '''
    Utility function that calls tvmc_wrapper and extracts output mlf
    (= TVM model library format) file.
    :param model: TVMC model that you wish to compile
    :param target: Can be "gap9, c" if you want to offload all possible
        computations to accelerator, and can be "c" for golden model checking.
    :param fuse_layers: sets relay.FuseOps.max_depth parameter to 1
        if set to False. This tells relay to not fuse operations.
        This can be useful when debuggin the TVM-generated c code kernels.
    :param build_path: path to export mlf file output to
    '''
    # Compile new model
    mlf_path = os.path.join(build_path, "model.tar")
    tvmc_wrapper(model, target, fuse_layers, mlf_path)
    # extract mlf file
    mlf = tarfile.TarFile(mlf_path)
    mlf.extractall(build_path)
    # remove the archive
    os.remove(mlf_path)

from mako.template import Template
def create_build_dir(build_path: str = "./build",
                     match_lib_path: str = "./lib",
                     target: MatchTarget=None):
    """
    param byoc_path: path to import Makefiles and C dependencies from
    """
    build_path = pathlib.Path(build_path)
    match_lib_path = pathlib.Path(match_lib_path)
    # check if build folder exists
    if build_path.is_dir():
        # remove build folder and all contents
        shutil.rmtree(build_path)
        # make the build folder again
        build_path.mkdir(parents=True)
    if not build_path.is_dir():
        # If no build folder exists create one
        build_path.mkdir(parents=True)
    # Copy over other necessary files
    src_dir = pathlib.Path("src")
    include_dir = pathlib.Path("include")
    # Copy over src, include and dory folders
    shutil.copytree(src=match_lib_path / src_dir, 
                    dst=build_path / src_dir, dirs_exist_ok=True)
    shutil.copytree(src=match_lib_path / include_dir, 
                    dst=build_path / include_dir, dirs_exist_ok=True)
    match_target_params_template = Template(filename=f"{match_lib_path}/match_target_params_template.h")
    possible_operands_combs=[["O","I","W"],["O","X","Y"]]
    memory_names={ex_mod.name:[] for ex_mod in target.exec_modules}
    for exec_module in target.exec_modules:
        for operands in possible_operands_combs:
            exec_module.memories_def(operands)
            for mem in exec_module.platform_memories:
                if mem.name not in memory_names[exec_module.name]:
                    memory_names[exec_module.name].append(mem.name)
    
    temp_data={"exec_modules":target.exec_modules,"memory_names":memory_names}
    match_target_params=match_target_params_template.render(**temp_data)
    with open(f"{build_path}/include/match_target_params.h", "w") as fw:
        fw.write(match_target_params)
