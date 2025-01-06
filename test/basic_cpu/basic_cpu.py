
import argparse
import json
import os
import match
from match.relay.utils.utils import create_random_array
from match.target.exec_module import ExecModule
from match.target.memory_inst import MemoryInst
from match.target.target import MatchTarget
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import wildcard, is_op
from match.partition.partitioning_pattern import PartitioningPattern
from pathlib import Path
import numpy as np
from typing import Tuple


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

def create_vec_conv_ex(inp_shape:Tuple=(32,32),fil_shape:Tuple=(1,1),
                       padding:Tuple=(0,0,0,0),strides:Tuple=(1,1),
                       groups:int=1,out_ch:int=1,inp_ch:int=3,**kwargs):
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
    

def create_vec_dense_ex(inp_features:int=256,out_features:int=128,**kwargs):
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
    if not Path("./models").is_dir():
        Path("./models").mkdir()
    if not Path("./models/last_model").is_dir():
        Path("./models/last_model").mkdir()
    with open(str(Path("./models/last_model/model_graph.relay").absolute()),"w") as mod_file:
        mod_file.write(relay.astext(mod))
    with open(str(Path("./models/last_model/model_params.txt").absolute()),"wb") as par_file:
        par_file.write(relay.save_param_dict(params=params))

def run_model(conv_ex:bool=False,output_path:str="./builds/last_build"):
    config = dict()
    if Path("./node_config.json").is_file():
        with open("./node_config.json") as config_file:
            config = json.load(config_file)
            for config_key in config:
                if isinstance(config[config_key],list):
                    config[config_key] = tuple(config[config_key])
    if conv_ex:
        mod,params = create_vec_conv_ex(**config)
    else:
        mod,params = create_vec_dense_ex(**config)
    save_model_and_params(mod=mod,params=params)
    #define HW Target inside match
    target=BasicCpuTarget()
    match.match(relay_mod=mod,relay_params=params,
                target=target,output_path=output_path,
                golden_default_cpu_model=True)

def run_relay_saved_model_at(mod_file,params_file,output_path):
    #define HW Target inside match
    target=BasicCpuTarget()
    match.match(input_type="relay",filename=mod_file,params_filename=params_file,target=target,output_path=output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        "--last_model",
        dest="last_model",
        action="store_true",
        help="compile the last model saved",
    )
    parser.add_argument(
        "-c",
        "--conv_example",
        dest="conv_example",
        action="store_true",
        help="Compile a simple 2d Convolution example node, that contains a Conv2d, a bias add and a ReLU operation",
    )

    parser.add_argument(
        "-d",
        "--dense_example",
        dest="dense_example",
        action="store_true",
        help="Compile a simple Dense example node, that contains a Dense, a bias add and a ReLU operation",
    )
    args = parser.parse_args()

    if not Path("./builds").is_dir():
        Path("./builds").mkdir()
    if args.last_model and Path("./models/last_model").is_dir() and \
        Path("./models/last_model/model_graph.relay").exists() and \
            Path("./models/last_model/model_params.txt").exists():
        run_relay_saved_model_at(mod_file=str(Path("./models/last_model/model_graph.relay").absolute()),
                                 params_file=str(Path("./models/last_model/model_params.txt").absolute()),
                                 output_path=str(Path("./builds/last_build").absolute()))
    else:
        run_model(conv_ex=args.conv_example,output_path=str(Path("./builds/last_build").absolute()))