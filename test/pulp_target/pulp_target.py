
import argparse
import json
import math
import os

import cartoonnyx
import match
from match.model.model import MatchModel
from match.relay.utils.utils import create_random_array,numpy_to_array
from match.target.exec_module import ExecModule
from match.target.gap9.cluster.cluster_cost_model import PulpClusterCostModel
from match.target.memory_inst import MemoryInst
from match.target.target import MatchTarget
from match.transform.layout import MatchLayoutNCHWtoNHWC
from match.transform.requant import MatchRequantRewriter
import tvm
from tvm import relay
import tvm.relay
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, is_var, is_expr, DFPatternCallback, rewrite
from match.partition.partitioning_pattern import PartitioningPattern
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
from match.utils.utils import get_random_np_array

class PulpCluster(ExecModule):
    def __init__(self):
        super(PulpCluster, self).__init__(name="pulp_cluster",
                                          specific_patterns=[
                                              "dense_requant",
                                              "conv_requant"
                                          ],
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include")
        self.top_memory = "L2_SHARED_MEM"

    def zigzag_cost_model(self):
        return PulpClusterCostModel

    def network_transformations(self, opts):
        return [
            ("requant",MatchRequantRewriter()),
            ("layout",MatchLayoutNCHWtoNHWC()),
        ]

    def update_constants(self, match_node, pattern_name):
        breakpoint()
        for w_tensor in match_node.const_tensors.values():
            if "dense" in w_tensor.name:
                w_tensor.data = w_tensor.data.transpose(1,0)
            elif "conv2d" in w_tensor.name:
                w_tensor.data = w_tensor.data.transpose(0,2,3,1)

    def platform_apis_def(self, platform_apis = ...):
        platform_apis.init_platform = "offload_to_pulp_cluster"
        platform_apis.init_module = "cluster_lib_init"
        return platform_apis
    
    def mem_apis_def(self, memory_apis = ...):
        memory_apis.mem_transfer = "handle_dma_transfer"
        memory_apis.init_memory["L1_SCRATCHPAD"] = "init_l1_scratchpad_memory"
        memory_apis.free_memory["L1_SCRATCHPAD"] = "free_l1_scrachpad_memory"
        return memory_apis
    
    def sync_apis_def(self, sync_apis = ...):
        sync_apis.wait_load = "wait_l1_dma_transfers"
        sync_apis.wait_store = "wait_l1_dma_transfers"
        sync_apis.wait_tile_computation = "wait_pulp_nn_computation"
        sync_apis.must_sync_after_store = True
        sync_apis.must_sync_after_computation = True
        return sync_apis
    
    def comp_apis_def(self, computational_apis = ...):
        computational_apis.compute_tile = "pulp_nn_wrapper"
        return computational_apis
    
    def specific_pattern_def(self, match_node=None, pattern_name = "conv_2d"):
        return "dense" if pattern_name=="dense" else pattern_name
    
    def partitioning_patterns(self):
        
        def conv_pt_requant():
            #Create pattern for a 2D Conv block, with bias and ReLU.
            conv2d = is_op("nn.conv2d")(
                wildcard(), wildcard()
            )
            conv2d = is_op("cast")(conv2d) | conv2d
            bias_add = is_op("nn.bias_add")(conv2d, wildcard())
            scale = is_op("multiply")(conv2d, wildcard()) | is_op("multiply")(wildcard(), conv2d)
            bias = is_op("add")(scale, wildcard()) | is_op("add")(wildcard(), scale)
            right_shift = is_op("right_shift")(bias_add | bias, is_constant())
            clip = is_op("clip")(right_shift)
            cast = is_op("cast")(clip)
            return cast

        
        def dense_pt_requant():
            """Create pattern for conv2D with optional fused relu."""
            dense = is_op("nn.dense")(
                wildcard(), wildcard()
            )
            dense = is_op("cast")(dense) | dense
            bias_add = is_op("nn.bias_add")(dense, wildcard())
            scale = is_op("multiply")(dense, wildcard()) | is_op("multiply")(wildcard(), dense)
            bias = is_op("add")(scale, wildcard()) | is_op("add")(wildcard(), scale)
            right_shift = is_op("right_shift")(bias_add | bias, is_constant())
            clip = is_op("clip")(right_shift)
            cast = is_op("cast")(clip)
            return cast
        
        def match_everything():
            return wildcard()

        def only_conv_pt():
            conv2d = is_op("nn.conv2d")(
                wildcard(), wildcard()
            )
            return conv2d

        return [
            # PartitioningPattern(name="only_conv",pattern=only_conv_pt),
            # PartitioningPattern(name="wildcard_pt",pattern=match_everything),
            PartitioningPattern(name="dense_requant",pattern=dense_pt_requant),
            # PartitioningPattern(name="conv_requant",pattern=conv_pt_requant),
        ]

    def memories_def(self, pattern_name, operands):
        return [
            # from lower level to higher level memories
            MemoryInst(name="L1_SCRATCHPAD",operands=["I","W","O"],k_bytes=128,sw_controlled=True),
            MemoryInst(name="L2_SHARED_MEM",operands=["I","W","O"],k_bytes=1496),
            # MemoryInst(name="L3_RAM",k_bytes=8912,sw_controlled=True),
        ]
    
class PulpPlatform(MatchTarget):
    def __init__(self):
        super(PulpPlatform,self).__init__([
            PulpCluster(),
        ],name="pulp_platform")
        # self.cpu_type = "arm_cpu"
        self.static_mem_plan = False
        self.static_mem_plan_algorithm = "hill_climb"
        self.makefile_path = os.path.dirname(__file__)+"/lib/Makefile"
        self.tvm_runtime_include_path = os.path.dirname(__file__)+"/lib/tvm_runtime.h"
        self.tvm_runtime_src_path = os.path.dirname(__file__)+"/lib/tvm_runtime.c"
        self.crt_config_path = os.path.dirname(__file__)+"/lib/crt_config.h"
        self.timestamp_to_ms = ""
        self.timestamp_type = "int"
        self.start_get_timestamp_api = "start_match_perf_counter"
        self.end_get_timestamp_api = "stop_match_perf_counter"
        self.init_funcs = ["pulp_cluster_init"]
        self.clean_funcs = ["pulp_cluster_close"]
        self.include_list = ["pulp_cluster/pulp_rt_profiler_wrapper","pmsis",
                             "pulp_cluster/cluster_dev","pulp_cluster/dory_dma",
                             "pulp_cluster/cluster_lib"]
        self.alloc_fn = "malloc_wrapper"
        self.free_fn = "free_wrapper"
        self.allocate_ext_mem = "pulp_init_ram"
        self.load_file_to_ext_mem_fn = "pulp_load_file"
        self.load_to_ext_mem_fn = "pulp_memcpy_to_ram"
        self.load_from_ext_mem_fn = "pulp_memcpy_from_ram"
        self.free_external_mem = "pulp_shutdown_ram"
        self.soc_memory_bytes = 12428

def create_dense_conv_dense_ex(inp_features:int=256,out_features:int=128,
                            inp_shape:Tuple=(32,32),fil_shape:Tuple=(1,1),
                            padding:Tuple=(0,0,0,0),strides:Tuple=(1,1),
                            groups:int=1,requant_pattern:bool=False,
                            right_shift:int=1,**kwargs):
    np.random.seed(0)
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType((1,inp_features), "uint8"))
    # Get or generate weight_values
    weights_1 = create_random_array((out_features*math.prod(inp_shape),inp_features),"int8")
    weights_2 = create_random_array((inp_features,out_features)+fil_shape,"int8")
    weights_3 = create_random_array((out_features,inp_features*math.prod([int(inp_shape[idx]/strides[idx]) for idx in range(len(inp_shape))])), "int8")
    # Get or generate bias values
    bias_1 = create_random_array((out_features*math.prod(inp_shape),), "int32")
    bias_2 = create_random_array((inp_features,), "int32")
    bias_3 = create_random_array((out_features,), "int32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_1_name = "dense_1_weights"
    bias_1_name = "dense_1_bias"
    weights_2_name = "conv_weights"
    bias_2_name = "conv_bias"
    weights_3_name = "dense_2_weights"
    bias_3_name = "dense_2_bias"

    # define relay input vars
    w_1 = relay.var(weights_1_name, relay.TensorType(weights_1.shape, weights_1.dtype))
    w_2 = relay.var(weights_2_name, relay.TensorType(weights_2.shape, weights_2.dtype))
    w_3 = relay.var(weights_3_name, relay.TensorType(weights_3.shape, weights_3.dtype))
    b_1 = relay.var(bias_1_name, relay.TensorType(bias_1.shape, bias_1.dtype))
    b_2 = relay.var(bias_2_name, relay.TensorType(bias_2.shape, bias_2.dtype))
    b_3 = relay.var(bias_3_name, relay.TensorType(bias_3.shape, bias_3.dtype))

    # define weights and bias values in params
    params = {weights_1_name: weights_1, bias_1_name: bias_1,
              weights_2_name: weights_2, bias_2_name: bias_2,
              weights_3_name: weights_3, bias_3_name: bias_3}

    # define operations
    x = relay.op.nn.dense(x, w_1, out_dtype=bias_1.dtype)
    x = relay.op.nn.bias_add(x, b_1, axis=-1)
    x = relay.op.nn.relu(x)
    x = relay.op.cast(x, "uint8")
    x = relay.op.reshape(x, (1, out_features)+inp_shape)
    x = relay.op.nn.conv2d(x, w_2,
                           strides=strides,
                           padding=padding,
                           groups=groups,
                           kernel_size=fil_shape,
                           out_dtype="int32",
                           )
    x = relay.op.nn.bias_add(x, b_2, axis=1)
    x = relay.op.nn.relu(x)
    x = relay.op.reshape(x, (1, inp_features*math.prod([int(inp_shape[idx]/strides[idx]) for idx in range(len(inp_shape))])))
    x = relay.op.nn.dense(x, w_3, out_dtype=bias_3.dtype)
    x = relay.op.nn.bias_add(x, b_3, axis=-1)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_conv_ex(inp_shape:Tuple=(32,32),fil_shape:Tuple=(1,1),
                       padding:Tuple=(0,0,0,0),strides:Tuple=(1,1),
                       groups:int=1,out_ch:int=1,inp_ch:int=3,
                       requant_pattern:bool=False,
                       right_shift:int=1,**kwargs):
    np.random.seed(0)
    x = relay.var("input_0", relay.TensorType((1,inp_ch)+inp_shape, "int8"))
    # Get or generate weight_values
    weights = create_random_array((out_ch,inp_ch)+fil_shape,"int8")
    # Get or generate bias values
    bias = create_random_array((out_ch,), "int32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "conv_weights"
    bias_name = "conv_bias"

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
                        #    data_layout="NHWC",
                        #    kernel_layout="HWIO",
                           out_dtype="int32",
                           )
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=1)
    if requant_pattern:
        x = relay.op.right_shift(x, relay.const(right_shift))
        x = relay.op.clip(x, a_min=0, a_max=255)
        x = relay.op.cast(x, "int8")
    else:
        x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params
    
def create_fp_conv_ex(inp_shape:Tuple=(32,32),fil_shape:Tuple=(1,1),
                       padding:Tuple=(0,0,0,0),strides:Tuple=(1,1),
                       groups:int=1,out_ch:int=1,inp_ch:int=3,**kwargs):
    np.random.seed(0)
    x = relay.var("input_0", relay.TensorType((1,inp_ch)+inp_shape))
    # Get or generate weight_values
    weights = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_ch,inp_ch)+fil_shape),dtype="float32")
    # Get or generate bias values
    bias = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_ch,)),dtype="float32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "pulp_conv_weights"
    bias_name = "pulp_conv_bias"

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
                           )
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=1)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_globalavgpool_ex(inp_shape:Tuple=(1,32,32),**kwargs):
    np.random.seed(0)
    x = relay.var("input_0", relay.TensorType((1,)+inp_shape))

    # define weights and bias values in params
    params = {}

    # define operations
    x = relay.op.nn.global_avg_pool2d(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_add_ex(inp_shape:Tuple=(1,32,32),**kwargs):
    np.random.seed(0)
    x = relay.var("input_0", relay.TensorType((1,)+inp_shape))
    y = relay.var("input_1", relay.TensorType((1,)+inp_shape))
    # define params
    params = {}
    # define operations
    x = relay.op.add(x,y)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_dense_ex(inp_features:int=256,out_features:int=128,**kwargs):
    np.random.seed(0)
    x = relay.var("input_0", relay.TensorType((1,inp_features)))
    # Get or generate weight_values
    weights = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_features,inp_features)),dtype="float32")
    # Get or generate bias values
    bias = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_features,)),dtype="float32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "pulp_dense_weights"
    bias_name = "pulp_dense_bias"

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights.shape, weights.dtype))

    # define weights and bias values in params
    params = {weights_name: weights, bias_name: bias}

    # define operations
    x = relay.op.nn.dense(x, w)
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=1)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params


def create_dense_ex(inp_features:int=256,out_features:int=128,
                        activation:bool=True,
                        requant_pattern:bool=False,
                        right_shift:int=1,**kwargs):
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
    weights_name = "dense_weights"
    bias_name = "dense_bias"

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights.shape, weights.dtype))

    # define weights and bias values in params
    params = {weights_name: weights, bias_name: bias}

    # define operations
    x = relay.op.nn.dense(x, w, out_dtype=bias.dtype)
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=-1)
    if activation:
        if requant_pattern:
            x = relay.op.right_shift(x, relay.const(right_shift))
            x = relay.op.clip(x, a_min=0, a_max=255)
            x = relay.op.cast(x, "int8")
        else:
            x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def write_config_file_in_build(config,output_path):
    with open(output_path+"/include/pulp_target/node_config.h","w") as config_file:
        config_file.write("\n".join(["#ifndef __MATCH_NODE_PULP_TARGET_CONFIG_TEST_H__",
                                    "#define __MATCH_NODE_PULP_TARGET_CONFIG_TEST_H__",
                                    # output dims
                                    f"#define NODE_OUT_CH {1 if 'out_ch' not in config else config['out_ch']}",
                                    f"#define NODE_OUT_H {32 if 'inp_shape' not in config else int(config['inp_shape'][0]/(1 if 'strides' not in config else config['strides'][0]))}",
                                    f"#define NODE_OUT_W {32 if 'inp_shape' not in config else int(config['inp_shape'][1]/(1 if 'strides' not in config else config['strides'][1]))}",
                                    # input dims
                                    f"#define NODE_INP_CH {3 if 'inp_ch' not in config else config['out_ch']}",
                                    f"#define NODE_INP_H {32 if 'inp_shape' not in config else config['inp_shape'][0]}",
                                    f"#define NODE_INP_W {32 if 'inp_shape' not in config else config['inp_shape'][1]}",
                                    # filter
                                    f"#define NODE_FIL_H {1 if 'fil_shape' not in config else config['fil_shape'][0]}",
                                    f"#define NODE_FIL_W {1 if 'fil_shape' not in config else config['fil_shape'][1]}",
                                    # strides
                                    f"#define NODE_STRIDE_H {1 if 'strides' not in config else config['strides'][0]}",
                                    f"#define NODE_STRIDE_W {1 if 'strides' not in config else config['strides'][1]}",
                                    # pad
                                    f"#define NODE_PAD_TOP {0 if 'padding' not in config else config['padding'][0]}",
                                    f"#define NODE_PAD_BOTTOM {0 if 'padding' not in config else config['padding'][1]}",
                                    f"#define NODE_PAD_LEFT {0 if 'padding' not in config else config['padding'][2]}",
                                    f"#define NODE_PAD_RIGHT {0 if 'padding' not in config else config['padding'][3]}",
                                    # shift
                                    f"#define NODE_RIGHT_SHIFT {1 if 'right_shift' not in config else config['right_shift']}",
                                    "#endif"
                                    ]))

def save_model_and_params(mod,params):
    if not Path("./models").is_dir():
        Path("./models").mkdir()
    if not Path("./models/last_model").is_dir():
        Path("./models/last_model").mkdir()
    with open(str(Path("./models/last_model/model_graph.relay").absolute()),"w") as mod_file:
        mod_file.write(relay.astext(mod))
    with open(str(Path("./models/last_model/model_params.txt").absolute()),"wb") as par_file:
        par_file.write(relay.save_param_dict(params=params))

MICROBENCH_MAPPER = {
    "conv":create_conv_ex,
    "dense":create_dense_ex,
    "dense_conv_dense":create_dense_conv_dense_ex,
    "avg_pool": create_fp_globalavgpool_ex,
}

def run_relay_saved_model_at(mod_file,params_file,output_path):
    #define HW Target inside match
    target=PulpPlatform()
    match.match(input_type="relay",filename=mod_file,params_filename=params_file,target=target,output_path=output_path)
    write_config_file_in_build({},output_path)




def run_microbench(microbench: str="conv", output_path: str="./builds/last_build", executor: str="aot"):
    config = dict()
    if Path("./node_config.json").is_file():
        with open("./node_config.json") as config_file:
            config = json.load(config_file)
            for config_key in config:
                if isinstance(config[config_key],list):
                    config[config_key] = tuple(config[config_key])
    mod,params = MICROBENCH_MAPPER[microbench](**config)
    save_model_and_params(mod=mod,params=params)
    #define HW Target inside match
    target=PulpPlatform()
    match.match(
        model=MatchModel(
           relay_mod=mod, relay_params=params,
           model_name=microbench, executor=executor,
           golden_cpu_model=False,
        ),
        target=target,
        output_path=output_path
    )

def run_model(model: str="keyword_spotting", output_path: str="./builds/last_build", executor: str="aot"):
    #define HW Target inside match
    target=PulpPlatform()
    cartoonnyx.cartoonnyx.plinio.sanitize_to_mps(os.path.dirname(__file__)+"/../models/"+model+".onnx")
    match.match(
        model=MatchModel(
           filename=os.path.dirname(__file__)+"/../models/"+model+".onnx",
           model_type="onnx",
           model_name=model, executor=executor
        ),
        target=target,
        output_path=output_path
    )


def run_relay_saved_model_at(mod_file, params_file, output_path, executor: str="aot"):
    #define HW Target inside match
    target=PulpPlatform()
    match.match(
        model=MatchModel(
           filename=mod_file, params_filename=params_file,
           model_type="relay",
           model_name="default", executor="graph"
        ),
        target=target,
        output_path=output_path
    )
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
        "--microbench",
        dest="microbench",
        type=str,
        default="conv",
        help="Compile a simple 2d Convolution example node, that contains a Conv2d, a bias add and a ReLU operation",
    )

    parser.add_argument(
        "--model",
        dest="model",
        default="",
        type=str,
        help="Model name to compile, models are available in the test/models directory."
    )

    parser.add_argument(
        "--executor",
        dest="executor",
        default="aot",
        type=str,
        choices=["aot","graph"],
        help="Choose the executor to use within MATCH",
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
        if args.model!="":
            run_model(model=args.model, output_path=str(Path("./builds/last_build").absolute()), executor=args.executor)
        else:
            run_microbench(microbench=args.microbench,output_path=str(Path("./builds/last_build").absolute()), executor=args.executor)