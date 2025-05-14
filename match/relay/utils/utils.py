import os
import shutil
import ctypes
import pathlib
from typing import Tuple, Dict

import tvm
import tvm.relay as relay

import numpy as np
import numpy.typing as npt

from mako.template import Template
from mako import exceptions

from match.target.target import DefaultMatchTarget, MatchTarget
from match.utils.utils import format_c_code


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

def relay_layout_transform(x, shape):
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

def relay_conv2d_uint8_requant(input_tensor: relay.Var, layer_name: str,
                      w_value: tvm.nd.array,
                      b_value: tvm.nd.array,
                      strides: Tuple[int, ...] = (1, 1),
                      padding: Tuple[int, ...] = (0, 0),
                      groups: int = 1,
                      act: bool = False,
                      shift_bits: int = 0,
                      batchnorm = True) -> Tuple[relay.Var,
                                                    Dict[relay.Expr,
                                                         tvm.nd.array]]:
    '''
    Creates a relay conv2d op
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
                           out_dtype="int32" if b_value is not None else b_value.dtype
                           )
    if batchnorm:
        input_shape=simple_basic_type_checker(input_tensor,w_value.shape)
        #input_shape = [int(x) for x in input_tensor.type_annotation.shape]
        lambda_name = layer_name + '_lambda'
        k_name = layer_name + '_k' 
        lambda_var=relay.var(lambda_name, relay.TensorType((1,w_value.shape[0],1,1), 'int32'))
        k_var=relay.var(k_name,  relay.TensorType((1,w_value.shape[0],1,1), 'int32'))
        params[lambda_name]=numpy_to_array(np.ones((1,w_value.shape[0],1,1), dtype = np.int32), 'int32')
        params[k_name]=numpy_to_array(np.ones((1,w_value.shape[0],1,1), dtype=  np.int32), 'int32')
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


def relay_dense_uint8_requant(input_tensor: relay.Var, layer_name: str,
                     w_value: tvm.nd.array,
                     b_value: tvm.nd.array,
                     act: bool = False,
                     shift_bits: int = 0,
                     batchnorm: bool = False):
    """
    Creates a relay dense op
    :param input_tensor: relay.Var for input
    :param layer_name: string that determines relay variable naming
    :param w_value: int8 tensor that contains weight values, must be of shape (num_inputs, num_outputs, 1, 1)
    :param b_value: int32 tensor that contains bias values
    :param act: bool that toggles extra ReLU to be added (see below)
    :shift_bits: int that sets amount of bits to shift right. Value must be between [0,31]
    """
    # define weights and bias variables
    weights_name = layer_name + '_weights'
    bias_name = layer_name + '_bias'
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
        lambda_name = layer_name + '_lambda'
        k_name = layer_name + '_k' 
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


def relay_add_uint8_requant(input_tensor_a: relay.Var,
                   input_tensor_b: relay.Var,
                   layer_name: str,
                   shift_bits: int = 0):
    """
    Creates a relay element-wise-add op
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


def create_random_array(shape: Tuple[int, ...], dtype: str, min_val=None, max_val=None) -> tvm.nd.array:
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
            if min_val is not None and isinstance(min_val, (float,int)):
                dtype_min = min_val
            else:
                dtype_min = np.iinfo(dtype).min
            if max_val is not None and isinstance(max_val, (float,int)):
                dtype_max = max_val
            else:
                dtype_max = np.iinfo(dtype).max
        except ValueError:
            range_map = {
                "int4": (-8, 7),
                "int2": (-1, 1),     # technically this should be (-2, 1), but we prefer to not use -2
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
    shape = [int(i) for i in shape]
    np_array = np.random.randint(low=dtype_min, high=dtype_max+1,
                                 size=shape, dtype=np_dtype)
    return numpy_to_array(np_array, dtype)

def create_build_dir(build_path: str = "./build",
                     match_lib_path: str = "./lib",
                     target: MatchTarget=DefaultMatchTarget()):
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
    pathlib.Path(build_path / "src").mkdir(parents=True,exist_ok=True)
    pathlib.Path(build_path / "include").mkdir(parents=True,exist_ok=True)

    shutil.copytree(src=match_lib_path / "static" / "match" /"src",
                    dst=build_path / "src" / "match", dirs_exist_ok=True)
    shutil.copytree(src=match_lib_path / "static" / "match" /"include",
                    dst=build_path / "include" / "match", dirs_exist_ok=True)

    libs_found = set()  
    for ex_mod in target.exec_modules:
        for lib_name, lib in ex_mod.libs_required.items():
            if lib_name in libs_found:
                print(f"[UTILS:: create_build_dir] Skipping {lib.name} as it has already been processed")
                continue
            libs_found.add(lib_name)
            # Copy over src, include folders
            if os.path.isdir(lib.src_path):
                shutil.copytree(src=pathlib.Path(lib.src_path), 
                                dst=build_path / "src" / lib_name, dirs_exist_ok=True)
            elif os.path.isdir(lib.base_path+"/src"):
                shutil.copytree(src=pathlib.Path(lib.base_path) / "src",
                                dst=build_path / "src" / lib_name, dirs_exist_ok=True)
            else:
                print(f"[UTILS:: create_build_dir] Could not find src path for {lib_name}")
            if os.path.isdir(lib.src_path):
                shutil.copytree(src=pathlib.Path(lib.inc_path), 
                                dst=build_path / "include" / lib_name, dirs_exist_ok=True)
            elif os.path.isdir(lib.base_path+"/include"):
                shutil.copytree(src=pathlib.Path(lib.base_path) / "include",
                                dst=build_path / "include" / lib_name, dirs_exist_ok=True)
            else:
                print(f"[UTILS:: create_build_dir] Could not find include path for {lib_name}")
    patterns_set=set()
    for exec_module in target.exec_modules:
        patterns_set.update([pt.name for pt in exec_module.partitioning_patterns()])
    template_data={"exec_modules":target.exec_modules,"patterns_list":list(patterns_set)}

    def translate_filename_and_get_template_data(filename, target):
        if filename == "exec_module":
            return {exec_module.name:{"exec_module":exec_module, "exec_module_id":exec_module_id, "target":target} for exec_module_id,exec_module in enumerate(target.exec_modules)}
        if filename == "target":
            return {target.name:{"target":target}}
        return {filename:template_data}
        
    for base_dir in ["src", "include"]:
        template_dir = match_lib_path / "mako" / "create" / base_dir
        if not template_dir.is_dir():
            continue
        for filename in os.listdir(template_dir):
            filename_without_dots = filename.split(".")
            filename_without_ext = ".".join(filename_without_dots[:-1])
            ext = filename_without_dots[-1]
            if ext in {"c","h","json"}:
                for build_filename,build_template_data in translate_filename_and_get_template_data(filename_without_ext, target).items():
                    try:
                        template = Template(filename = os.path.join(template_dir, filename))
                        rendered_content = template.render(**build_template_data)
                        if ext in {"c", 'h'}:
                            rendered_content = format_c_code(rendered_content)
                        output_path = os.path.join(build_path, base_dir, build_filename + "." + ext)
                        with open(output_path, "w") as output_file:
                            output_file.write(rendered_content)
                    except Exception as exc:
                        print(f"[UTILS:: create_build_dir] Error processing template: {filename}")
                        with open(os.path.join(build_path, base_dir, filename_without_ext+".html"), "wb") as output_file:
                            output_file.write(exceptions.html_error_template().render())