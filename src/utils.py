
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


def create_build_dir(byoc_path: str = ".",
                     build_path: str = "./build",
                     device: str = "pulp"):
    """
    param byoc_path: path to import Makefiles and C dependencies from
    """
    build_path = pathlib.Path(build_path)
    byoc_path = pathlib.Path(byoc_path)
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
    if device == "gap9":
        makefile_pulprt = pathlib.Path("Makefile.pulprt")
        shutil.copyfile(src=byoc_path / makefile_pulprt, 
                        dst=build_path / makefile_pulprt)
    elif device == "x86":
        makefile_x86 = pathlib.Path("Makefile.x86")
        shutil.copyfile(src=byoc_path / makefile_x86, 
                        dst=build_path / makefile_x86)
    else:
        raise NotImplementedError
    src_dir = pathlib.Path("src")
    include_dir = pathlib.Path("include")
    # Copy over src, include and dory folders
    shutil.copytree(src=byoc_path / src_dir, 
                    dst=build_path / src_dir, dirs_exist_ok=True)
    shutil.copytree(src=byoc_path / include_dir, 
                    dst=build_path / include_dir, dirs_exist_ok=True)


def copy_dory_files(dory_path: str = "",
                    build_path: str = "./build"):
    """
    Function that copies dory library files on dory_path to a build_path
    """
    dory_path = os.path.dirname(os.path.abspath(dory.__file__))
    # Use the new adapter from DORY - gets the pulp-nn files
    adapter = PulpNNAdapter(dirname = "pulp-nn", node = None, constant_bits = 32)
    kernel_src_files= adapter._get_src_files()
    kernel_include_files = adapter._get_inc_files()

    # Get all utils files, flattened
    utils_src_files = list(pathlib.Path(os.path.join(dory_path,  "Hardware_targets/PULP/GAP9/Utils_files")).rglob("*.c"))
    utils_include_files = list(pathlib.Path(os.path.join(dory_path,  "Hardware_targets/PULP/GAP9/Utils_files")).rglob("*.h"))

    # Create the new dirs
    dory_src_dir = pathlib.Path(build_path) / "dory/src"
    dory_src_dir.mkdir(parents=True)
    dory_inc_dir = pathlib.Path(build_path) / "dory/include"
    dory_inc_dir.mkdir(parents=True)


    for src in kernel_src_files:
        fname = os.path.basename(src)
        shutil.copyfile(src, dory_src_dir/fname)
    for inc in kernel_include_files:
        fname = os.path.basename(inc)
        shutil.copyfile(inc, dory_inc_dir/fname)
    for src in utils_src_files:
        fname = os.path.basename(src)
        shutil.copyfile(src, dory_src_dir/fname)
    for inc in utils_include_files:
        fname = os.path.basename(inc)
        shutil.copyfile(inc, dory_inc_dir/fname)
    
def copy_tvmzigzag_files(tvmzigzag_path: str = "./templates/gap9",
                    build_path: str = "./build"):
    """
    Function that copies tvm+zigzag library files on build path
    """
    arch_dir = pathlib.Path("gap9")
    shutil.copytree(src=tvmzigzag_path,dst=build_path/arch_dir,dirs_exist_ok=True)

def create_demo_file(mod: tvm.ir.IRModule, directory: str = "build", 
                     init_value: int = 1, indefinite: bool = False, 
                     no_of_inputs: int = 1,
                     ):
    '''
    Function that creates a demo file in which inputs and outputs of the
    right size are allocated and setup automatically. Based on:

    https://discuss.tvm.apache.org/t/
    how-to-get-the-input-and-output-of-relay-call-node/8743

    Note, the no_of_inputs argument currently sets extra inputs.
    This is for example necessary in the testing of add layers (2 inputs)
    Beware that this does not include any sanity checking!
    It just takes the two most upfront inferred types!
    '''
    ones=False
    ones=True
    init_value="1" if ones else "i%121"
    directory = pathlib.Path(directory)
    def get_c_type(dtype):
        if dtype == "int8":
            return "int8_t"
        elif dtype == "float32":
            return "float"
        elif dtype == "uint8":
            return "uint8_t"
        else:
            raise NotImplementedError
    # Before you can get the input and output types of a relay node
    # you first have to run the InferType Relay pass
    # otherwise checked_type will return a ValueError
    print("Creating demo file: Inferring shapes and types...")
    mod = relay.transform.InferType()(mod)
    # Assuming the first arguments are the user-supplied input
    # Convert from TVM runtime datatype to numpy array
    input_shapes = []
    input_dtypes = []
    for i in range(no_of_inputs):
        input_shapes.append(np.array(mod["main"].checked_type.arg_types[i].shape))
        input_dtypes.append(mod["main"].checked_type.arg_types[i].dtype)
    type_decls_in = [get_c_type(dtype) for dtype in input_dtypes]
    # Assuming there is only output to this Relay IRMod
    # Convert from TVM runtime datatype to numpy array
    output_shape = np.array(mod["main"].checked_type.ret_type.shape)
    output_dtype = mod["main"].checked_type.ret_type.dtype
    create_demo_gdb_scripts(output_dtype, directory=directory)
    type_decl_out = get_c_type(output_dtype)
    # Additional GAP9 code, functions are empty for x86
    # TODO: It would make more sense not to add these depending on the target
    #? Similar to DIANA Analog code
    hardware_init_code = "gap9_cluster_init();"
    hardware_stop_code = "gap9_cluster_close();"

    print("Creating demo file: Inferred shapes:")
    for i in range(no_of_inputs):
        print(f"\tinput ({input_dtypes[i]}):")
        print(f"\t {input_shapes[i]}")
    print(f"\toutput ({output_dtype}):")
    print(f"\t {output_shape}")
    mallocs = ""
    frees = ""
    sizes = ""
    for i, type_d in enumerate(type_decls_in):
        mallocs += f"  {type_d} *input_{i} = ({type_d}*)malloc_wrapper(input_{i}_size * sizeof({type_d}));\n"
        frees += f"  free_wrapper(input_{i});\n"
    mallocs += f"  {type_decl_out} *output = ({type_decl_out}*)malloc_wrapper(output_size * sizeof({type_decl_out}));\n\n"
    frees += "    free_wrapper(output);\n"
    inits = f"  // Fill input with {init_value}\n"
    for i, input_shape in enumerate(input_shapes):
        sizes += f"  uint32_t input_{i}_size = {np.prod(input_shape)};\n"
        inits += "  for (uint32_t i = 0; i < input_"+str(i)+"_size; i++){\n"
        inits += f"    input_{i}[i] = {init_value};\n"
        inits += "  }\n"
    sizes += f"  uint32_t output_size = {np.prod(output_shape)};\n\n"
    call = "  struct tvmgen_default_outputs outputs = { .output = output, };\n"
    call += "  struct tvmgen_default_inputs inputs = {\n"
    for i in range(len(input_shapes)):
        call += f"    .input_{i} = input_{i},\n"
    call += "  };\n\n"
    # Now produce the final c code
    c_code = "#include <stdio.h>\n"  +\
             "#include <stdint.h>\n" +\
             "#include \"tvmgen_default.h\"\n" +\
             "#include <tvm_runtime.h>\n" +\
             "#include <malloc_wrapper.h>\n" +\
             "#include <gdb_anchor.h>\n" +\
    "\n" +\
    "int abs(int v) {return v * ((v > 0) - (v < 0)); }\n\n" +\
    "int main(int argc, char** argv) {\n" +\
    hardware_init_code +\
    "  // Sizes automatically added by utils.create_demo_file\n" +\
    'printf("{\\\"latencies\\\":{");\n' +\
    sizes + \
    mallocs + \
    inits + "\n" +\
    call + \
    "  int32_t status = 0;\n" + \
    ("  while (status == 0){\n   " if indefinite else "") + \
    "  status = tvmgen_default_run(&inputs, &outputs);\n" + \
    ("}\n" if indefinite else "") + \
    "  gdb_anchor();" + \
    '\nprintf("},\\\"output\\\":[");' +\
    "\nfor(int k=0;k<output_size;k++) {printf(\"%d\",(uint8_t)output[k]);if(k!=output_size-1) printf(\", \");}\n"+\
    '\nprintf("]}");' +\
    frees + \
    "  if(status != 0){\n" +\
    "    abort();\n" +\
    "  }\n" +\
    hardware_stop_code +\
    "  return 0;\n" +\
    "}\n"
    with open(directory/ "src/demo.c", "w") as file:
        file.writelines(c_code)

def create_demo_file_x86(mod: tvm.ir.IRModule, directory: str = "build", 
                     init_value: int = 1, indefinite: bool = False, 
                     no_of_inputs: int = 1,
                     ):
    '''
    Function that creates a demo file in which inputs and outputs of the
    right size are allocated and setup automatically. Based on:

    https://discuss.tvm.apache.org/t/
    how-to-get-the-input-and-output-of-relay-call-node/8743

    Note, the no_of_inputs argument currently sets extra inputs.
    This is for example necessary in the testing of add layers (2 inputs)
    Beware that this does not include any sanity checking!
    It just takes the two most upfront inferred types!
    '''
    ones=False
    ones=True
    init_value="1" if ones else "i%121"
    directory = pathlib.Path(directory)
    def get_c_type(dtype):
        if dtype == "int8":
            return "int8_t"
        elif dtype == "float32":
            return "float"
        else:
            raise NotImplementedError
    # Before you can get the input and output types of a relay node
    # you first have to run the InferType Relay pass
    # otherwise checked_type will return a ValueError
    print("Creating demo file: Inferring shapes and types...")
    mod = relay.transform.InferType()(mod)
    # Assuming the first arguments are the user-supplied input
    # Convert from TVM runtime datatype to numpy array
    input_shapes = []
    input_dtypes = []
    for i in range(no_of_inputs):
        input_shapes.append(np.array(mod["main"].checked_type.arg_types[i].shape))
        input_dtypes.append(mod["main"].checked_type.arg_types[i].dtype)
    type_decls_in = [get_c_type(dtype) for dtype in input_dtypes]
    # Assuming there is only output to this Relay IRMod
    # Convert from TVM runtime datatype to numpy array
    output_shape = np.array(mod["main"].checked_type.ret_type.shape)
    output_dtype = mod["main"].checked_type.ret_type.dtype
    create_demo_gdb_scripts(output_dtype, directory=directory)
    type_decl_out = get_c_type(output_dtype)
    # Additional GAP9 code, functions are empty for x86
    # TODO: It would make more sense not to add these depending on the target
    #? Similar to DIANA Analog code

    print("Creating demo file: Inferred shapes:")
    for i in range(no_of_inputs):
        print(f"\tinput ({input_dtypes[i]}):")
        print(f"\t {input_shapes[i]}")
    print(f"\toutput ({output_dtype}):")
    print(f"\t {output_shape}")
    mallocs = ""
    frees = ""
    sizes = ""
    for i, type_d in enumerate(type_decls_in):
        mallocs += f"  {type_d} *input_{i} = ({type_d}*)malloc_wrapper(input_{i}_size * sizeof({type_d}));\n"
        frees += f"  free_wrapper(input_{i});\n"
    mallocs += f"  {type_decl_out} *output = ({type_decl_out}*)malloc_wrapper(output_size * sizeof({type_decl_out}));\n\n"
    frees += "    free_wrapper(output);\n"
    inits = f"  // Fill input with {init_value}\n"
    for i, input_shape in enumerate(input_shapes):
        sizes += f"  uint32_t input_{i}_size = {np.prod(input_shape)};\n"
        inits += "  for (uint32_t i = 0; i < input_"+str(i)+"_size; i++){\n"
        inits += f"    input_{i}[i] = {init_value};\n"
        inits += "  }\n"
    sizes += f"  uint32_t output_size = {np.prod(output_shape)};\n\n"
    call = "  struct tvmgen_default_outputs outputs = { .output = output, };\n"
    call += "  struct tvmgen_default_inputs inputs = {\n"
    for i in range(len(input_shapes)):
        call += f"    .input_{i} = input_{i},\n"
    call += "  };\n\n"
    # Now produce the final c code
    c_code = "#include <stdio.h>\n"  +\
             "#include <stdint.h>\n" +\
             "#include \"tvmgen_default.h\"\n" +\
             "#include <tvm_runtime.h>\n" +\
             "#include <malloc_wrapper.h>\n" +\
             "#include <gdb_anchor.h>\n" +\
    "\n" +\
    "int abs(int v) {return v * ((v > 0) - (v < 0)); }\n\n" +\
    "int main(int argc, char** argv) {\n" +\
    "  // Sizes automatically added by utils.create_demo_file\n" +\
    sizes + \
    mallocs + \
    inits + "\n" +\
    call + \
    "  int32_t status = 0;\n" + \
    ("  while (status == 0){\n   " if indefinite else "") + \
    "  status = tvmgen_default_run(&inputs, &outputs);\n" + \
    ("}\n" if indefinite else "") + \
    "  gdb_anchor();" + \
    frees + \
    "  if(status != 0){\n" +\
    "    abort();\n" +\
    "  }\n" +\
    "  return 0;\n" +\
    "}\n"
    with open(directory/ "src/demo.c", "w") as file:
        file.writelines(c_code)

def adapt_gcc_opt(makefile_path: str, opt_level: int):
    '''
    Adapts this line in a file to change OPT_LEVEL:
        OPT_LEVEL = 3
    typically used for makefiles.

    NOTE: Raises runtime error if no alterations were made
    '''
    regex = r"(OPT_LEVEL =) (\d)"
    subst = f"\\1 {opt_level}"
    with open(makefile_path, "r+") as makefile:
        makefile_string = makefile.read()
        replaced_string, subs = re.subn(regex, subst, makefile_string,
                                        0, re.MULTILINE)
        if subs != 1:
            raise RuntimeError("Could not alter makefile opt level")
        makefile.seek(0)
        makefile.write(replaced_string)
        makefile.truncate()
        print(f"Changed opt_level to {opt_level} @ {makefile_path}")

def make(device: str = "pulp", make_dir: str = ".", verbose: bool = False):
    '''
    Invokes make from the current directory in a new subprocess

    :param device: select which device to call make for, "x86" or "pulp"
    :param verbose: print make output
    '''
    print(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"].replace("/tvm-zigzag/gap9/docker/zigzag/zigzag", "")
    print(f"Make: Invoking make for device: '{device}'")
    if device == "x86":
        makefile = "Makefile.x86"
    elif device == "gap9":
        makefile = "Makefile.pulprt"
    else:
        raise ValueError(f"Device: '{device}' not supported")
    output = subprocess.run(["make", "-f", makefile, 
                                         "all"], cwd=make_dir,
                                         check=True,
                                         stderr=subprocess.STDOUT,
                                         universal_newlines=True)
    if verbose:
        print(output)
    print(f"Make: Built for '{device}'")

def create_demo_gdb_scripts(dtype : str = "uint8", directory: str = "."):
    def get_gdb_type(dtype):
        if dtype == "int8":
            return "/d"
        elif dtype == "float32":
            return "/f"
        elif dtype == "uint8":
            return "/d"
        else:
            raise NotImplementedError()
    directory = pathlib.Path(directory)
    preamble =\
    'set print elements 0\n' +\
    'set print repeats 0\n' +\
    'set pagination off\n'  +\
    'set max-value-size unlimited\n'
    # These parts are not common
    x86 = preamble +\
    'break gdb_anchor\n' +\
    'run\n' +\
    'n\n' +\
    'n\n' +\
    'n\n' +\
    f'set logging file {directory.resolve()}/demo_x86.txt\n'
    pulp = preamble +\
    'target remote localhost:12345\n' +\
    'load\n' +\
    'break gdb_anchor\n' +\
    'c\n' +\
    'n\n' +\
    'n\n' +\
    f'set logging file {directory.resolve()}/demo.txt\n'
    # These parts are common again
    common =\
    'set logging on\n' +\
    f'print {get_gdb_type(dtype)} *output@output_size\n' +\
    'set logging off\n'
    with open(directory / "gdb_demo_x86.sh", "w") as gdb_script:
        gdb_script.write(x86 + common)
        print(f"Made gdb_demo_x86.sh for {dtype}")
    with open(directory / "gdb_demo.sh", "w") as gdb_script:
        gdb_script.write(pulp + common)
        print(f"Made gdb_demo.sh for {dtype}")

def gdb(device: str, binary: str = None,
        directory: pathlib.Path = pathlib.Path("."),
        gdb_script: str = None, 
        verbose : bool = False) -> np.typing.NDArray:
    """
    Calls gdb run (batch mode) for binary with gdb_script on specified device
    If verbose is set, output is printed

    returns the parsed gdb output in a numpy float array
    """
    directory = pathlib.Path(directory)
    def print_error(log_file, gdb_output):
        print(f"Could not open {log_file} -> gdb output was:")
        print("============================================")
        print(gdb_output)
    if device == "x86":
        log = directory / "demo_x86.txt"
        # Remove previous log before proceeding
        log.unlink(missing_ok=True)
        if binary is None:
            binary = "demo" 
        if gdb_script is None:
            gdb_script = "gdb_demo_x86.sh"
        print(f"GDB: Running '{gdb_script}' on '{device}'...")
        out = gdb_x86(directory/gdb_script, directory/binary, verbose)
        print("GDB: Run on x86 finished")
        try:
            result = get_gdb_output(log)
        except FileNotFoundError as e:
            print_error(log, out)
            return None
        return result
    elif device == "gap9":
        log = directory / "demo.txt"
        # Remove previous log before proceeding
        log.unlink(missing_ok=True)
        if binary is None:
            binary = "gap9/demo/demo"
        if gdb_script is None:
            gdb_script = "gdb_demo.sh"
        print(f"GDB: Running '{gdb_script}' on '{device}'...")
        gdb_working=False
        if gdb_working:
            out = gdb_pulp(directory/gdb_script, directory/binary, verbose)
            print("GDB: Run on PULP finished")
            try:
                result = get_gdb_output(log)
            except FileNotFoundError as e:
                print_error(log, out)
                result = [1,1,1,1]
        else:
            result = run_gvsoc(directory,verbose)
        return result
    else:
        raise ValueError(f"Device: '{device}' not supported")

def run_gvsoc(directory: str, verbose: bool = False):
    from subprocess import Popen,PIPE
    output={"latencies":{0:0,"total":0},"output":[1,1,1,1]}
    with open(directory/"tmplog.txt","wb") as logfile:
        output1 = Popen(["make","-f","Makefile.pulprt","run"],cwd=directory,stdout=PIPE)
        output2 = Popen(["grep","}"],stdin=output1.stdout,stdout=logfile)
        output2.wait()
    with open(directory/"tmplog.txt") as logfile:
        import json
        output=json.load(logfile)
        output["latencies"]["total"]=sum([lat for layer,lat in output["latencies"].items()])
    Popen(["rm","tmplog.txt"],cwd=directory)
    return output

def gdb_x86(gdb_script: str, binary: str, verbose: bool = False) -> str:
    output = subprocess.check_output(["gdb", binary, "-x", gdb_script, 
                                      "-batch"],
                                     stderr=subprocess.STDOUT,
                                     timeout=120,
                                     universal_newlines=True) 
    if verbose:
        print(output)
    return output


def gdb_pulp(gdb_script: str, binary: str, verbose: bool = False) -> str: 
    import os
    path_toolchain = os.environ["GAP_RISCV_GCC_TOOLCHAIN"]
    riscv_gdb = os.path.join(path_toolchain, "bin", "riscv32-unknown-elf-gdb")
    """
    NOTE for some reason this program exits with zero even after errors?
    https://sourceware.org/bugzilla/show_bug.cgi?id=13000
    (Bug was fixed in 2018)
    """
    timeout=200
    #try:
    output = subprocess.check_output([riscv_gdb, binary, "-x", gdb_script,
                                        "-batch"],
                                        stderr=subprocess.STDOUT,
                                        timeout=timeout,
                                        universal_newlines=True)
    #except subprocess.TimeoutExpired as e:
    #    print(f"GDB timed out after {timeout} seconds! --> Output:")
    #    print("==================================================")
    #    print(e.stdout.decode())
    #    return None
    if verbose:
        print(output)
    return output
    
def size_pulp(binary: str, verbose: bool = False) -> Dict[str,int]: 
    riscv_size = "/pulp-riscv-gnu-toolchain/bin/riscv32-unknown-elf-size"
    output = subprocess.check_output([riscv_size, binary],
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True) 
    if verbose:
        print(output)
    out = [int(match.group()) for match in re.finditer("(\d+)", output,
                                                       re.MULTILINE)]
    return {"text": out[0],
            "data": out[1],
            "bss": out[2],
            "total": out[3]}



def get_gdb_output(gdb_log_path="debug/gdb.txt"):
    """
    Following lines use the logging output of gdb to match test results with model results

    logging is set by:
        (gdb) set logging on       --> log to gdb.txt
        (gdb) print some_variable
        $1 = \032
        (gdb) set logging off

    In the code below we use regex to match an output created by gdb for an array:
        (gdb) print *my_array@array_size
        $2 = { 69, 420, ... , 42}  --> will be logged to gdb.txt

    After some string manipulation this array is converted to a numpy array.
    This array is checked for a complete match with np.ma.allequal()

    raises FileNotFoundError in case the log file can not be opened
    """
    with open(gdb_log_path) as log:
        data = ""
        for line in log.readlines():
            data += line.strip()
        # Find the right string in gdb log
        matcher = re.compile(r"{.*}",flags=re.DOTALL)
        result = matcher.search(data)
        string = result.group(0)
        # "{ ... , ... }" --> "... , ..."
        string = string.replace("{","")
        string = string.replace("}","")
        # makes a list of numbers in string format
        list_numbers = string.split(",")
        # convert strings to integers
        values = [float(number) for number in list_numbers]
    values_from_test = np.array(values, dtype="float")
    # Values are returned from GDB in one big one-dimensional tensor
    # Reshaping here such that it matches the output
    return values_from_test


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