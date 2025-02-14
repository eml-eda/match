
import argparse
import json
import math
import os
import match
from match.relay.utils.utils import create_random_array,numpy_to_array
from match.target.exec_module import ExecModule
from match.target.gap9.cluster.cluster_cost_model import PulpClusterCostModel
from match.target.memory_inst import MemoryInst
from match.target.target import MatchTarget
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, is_var, is_expr
from match.partition.partitioning_pattern import PartitioningPattern
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
from match.utils.utils import get_random_np_array

class PulpTarget(ExecModule):
    def __init__(self):
        super(PulpTarget, self).__init__(name="pulp_target",
                                          specific_patterns=[
                                              "vec_dense",
                                              "vec_conv"
                                          ],
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include")
        self.top_memory = "L2_SHARED_MEM"

    def cost_model(self):
        return PulpClusterCostModel

    def platform_apis_def(self, platform_apis = ...):
        platform_apis.init_platform = "offload_to_pulp_cluster"
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
        return "dense" if pattern_name=="vec_dense" else pattern_name
    
    def partitioning_patterns(self):

        def vec_conv_pt():
            #Create pattern for a 2D Conv block, with bias and ReLU.
            conv2d = is_op("nn.conv2d")(
                wildcard(), wildcard()
            )
            bias_add = is_op("nn.bias_add")(conv2d, wildcard())
            relu = is_op("nn.relu")(bias_add)
            return relu
        
        def vec_conv_pt_requant():
            #Create pattern for a 2D Conv block, with bias and ReLU.
            conv2d = is_op("nn.conv2d")(
                wildcard(), wildcard()
            )
            bias_add = is_op("nn.bias_add")(conv2d, wildcard())
            right_shift = is_op("right_shift")(bias_add, is_constant())
            clip = is_op("clip")(right_shift)
            cast = is_op("cast")(clip)
            return cast

        def vec_dense_pt():
            """Create pattern for conv2D with optional fused relu."""
            dense = is_op("nn.dense")(
                wildcard(), wildcard()
            )
            bias_add = is_op("nn.bias_add")(dense, wildcard())
            relu = is_op("nn.relu")(bias_add) | bias_add
            return relu
        
        def vec_dense_pt_requant():
            """Create pattern for conv2D with optional fused relu."""
            dense = is_op("nn.dense")(
                wildcard(), wildcard()
            )
            bias_add = is_op("nn.bias_add")(dense, wildcard())
            right_shift = is_op("right_shift")(bias_add, is_constant())
            clip = is_op("clip")(right_shift)
            cast = is_op("cast")(clip)
            return cast
        
        def match_everything():
            return wildcard()


        return [
            # PartitioningPattern(name="wildcard_pt",pattern=match_everything),
            PartitioningPattern(name="vec_dense",pattern=vec_dense_pt),
            # PartitioningPattern(name="vec_conv",pattern=vec_conv_pt),
            # PartitioningPattern(name="vec_dense_requant",pattern=vec_dense_pt_requant),
            # PartitioningPattern(name="vec_conv_requant",pattern=vec_conv_pt_requant),
        ]

    def memories_def(self, pattern_name, operands):
        return [
            # from lower level to higher level memories
            MemoryInst(name="L1_SCRATCHPAD",operands=["I","W","O"],k_bytes=128,sw_controlled=True),
            MemoryInst(name="L2_SHARED_MEM",operands=["I","W","O"],k_bytes=1496),
            #MemoryInst(name="NEOPTEX_L3_CACHE",k_bytes=128),
            #MemoryInst(name="NEOPTEX_DRAM",k_bytes=4196),
        ]
    
class PulpPlatform(MatchTarget):
    def __init__(self):
        super(PulpPlatform,self).__init__([
            PulpTarget(),
        ],name="pulp_platform")
        # self.cpu_type = "arm_cpu"
        # self.static_mem_plan = False
        self.static_mem_plan_algorithm = "hill_climb"
        self.makefile_path = os.path.dirname(__file__)+"/lib/Makefile"
        self.tvm_runtime_include_path = os.path.dirname(__file__)+"/lib/tvm_runtime.h"
        self.tvm_runtime_src_path = os.path.dirname(__file__)+"/lib/tvm_runtime.c"
        self.crt_config_path = os.path.dirname(__file__)+"/lib/crt_config.h"
        self.timestamp_to_ms = ""
        self.timestamp_type = "int"
        self.start_get_timestamp_api = "start_match_perf_counter"
        self.end_get_timestamp_api = "stop_match_perf_counter"
        self.init_funcs = ["gap9_cluster_init"]
        self.clean_funcs = ["gap9_cluster_close"]
        self.include_list = ["pulp_target/pulp_rt_profiler_wrapper","pmsis","pulp_target/gap9_cluster","pulp_target/dory_dma"]
        self.alloc_fn = "malloc_wrapper"
        self.free_fn = "free_wrapper"

def create_vec_conv_ex(inp_shape:Tuple=(32,32),fil_shape:Tuple=(1,1),
                       padding:Tuple=(0,0,0,0),strides:Tuple=(1,1),
                       groups:int=1,out_ch:int=1,inp_ch:int=3,
                       requant_pattern:bool=False,
                       right_shift:int=1,**kwargs):
    np.random.seed(0)
    x = relay.var("input_0", relay.TensorType((1,)+inp_shape+(inp_ch,), "int8"))
    # Get or generate weight_values
    weights = create_random_array(fil_shape+(inp_ch,out_ch),"int8")
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
                           data_layout="NHWC",
                           kernel_layout="HWIO",
                           out_dtype="int32",
                           )
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=-1)
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


def create_vec_dense_ex(inp_features:int=256,out_features:int=128,
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

def run_model(model:str="keyword_spotting",output_path:str="./builds/last_build"):
    #define HW Target inside match
    target=PulpPlatform()
    match.match(filename=os.path.dirname(__file__)+"/../models/"+model+".onnx",
                target=target,output_path=output_path,golden_default_cpu_model=len(target.partitioning_patterns())>0)

MICROBENCH_MAPPER = {
    "conv":create_vec_conv_ex,
    "dense":create_vec_dense_ex,
}

def run_microbench(microbench:str="conv",output_path:str="./builds/last_build"):
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
    match.match(relay_mod=mod,relay_params=params,
                target=target,output_path=output_path,
                golden_default_cpu_model=True)
    write_config_file_in_build(config,output_path)

def run_relay_saved_model_at(mod_file,params_file,output_path):
    #define HW Target inside match
    target=PulpPlatform()
    match.match(input_type="relay",filename=mod_file,params_filename=params_file,target=target,output_path=output_path)
    write_config_file_in_build({},output_path)

def compile_single_layers(layers:List=[],base_path:str="./builds",
                          layer_function_mapper:Dict={},target:MatchTarget=PulpPlatform()):
    for layer_name in set(layers):
        layer_name_split = layer_name.split("/")
        model_name = layer_name_split[0]
        layer_type = layer_name_split[1]
        config = dict()
        for layer_config in layer_name_split[2:]:
            layer_config_split = layer_config.split("-")
            key = layer_config_split[0]
            value = layer_config_split[1]
            if len(value.split("x"))>1:
                value = tuple([int(v) for v in value.split("x")])
            elif value.isnumeric():
                value = int(value)
            config[key] = value
        if not Path(f"{base_path}/{model_name}").is_dir():
            Path(f"{base_path}/{model_name}").mkdir(exist_ok=True)
        mod,params = layer_function_mapper[layer_type](**config)
        output_path = str(Path(f"{base_path}/{model_name}/{'-'.join(layer_name_split[1:])}").absolute())
        match.match(relay_mod=mod,relay_params=params,
                    output_path=output_path,
                    target=target,)
        write_config_file_in_build(config=config,output_path=output_path)

def compile_resnet_18_fp_layer_by_layer(base_path:str="./builds"):
    #define HW Target inside match
    target=PulpPlatform()
    layers = [
        # tail
        "resnet_18/conv/out_ch-64/inp_ch-3/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
        # basic block no exp conv 1
        "resnet_18/conv/out_ch-64/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
        # basic block no exp conv 2
        "resnet_18/conv/out_ch-64/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
        # basic block no exp conv 3
        "resnet_18/conv/out_ch-64/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
        # basic block no exp conv 4
        "resnet_18/conv/out_ch-64/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
        # basic block expand conv 1
        "resnet_18/conv/out_ch-128/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-2x2/padding-1x1x1x1",
        # basic block expand conv 2
        "resnet_18/conv/out_ch-128/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-1x1/padding-1x1x1x1",
        # basic block expand shortcut
        "resnet_18/conv/out_ch-128/inp_ch-64/fil_shape-1x1/inp_shape-32x32/strides-2x2/padding-0x0x0x0",
        # basic block expand conv 3
        "resnet_18/conv/out_ch-128/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-1x1/padding-1x1x1x1",
        # basic block expand conv 4
        "resnet_18/conv/out_ch-128/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-1x1/padding-1x1x1x1",
        # basic block expand 2 conv 1
        "resnet_18/conv/out_ch-256/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-2x2/padding-1x1x1x1",
        # basic block expand 2 conv 2
        "resnet_18/conv/out_ch-256/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-1x1/padding-1x1x1x1",
        # basic block expand 2 shortcut
        "resnet_18/conv/out_ch-256/inp_ch-128/fil_shape-1x1/inp_shape-16x16/strides-2x2/padding-0x0x0x0",
        # basic block expand 2 conv 3
        "resnet_18/conv/out_ch-256/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-1x1/padding-1x1x1x1",
        # basic block expand 2 conv 4
        "resnet_18/conv/out_ch-256/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-1x1/padding-1x1x1x1",
        # basic block expand 3 conv 1
        "resnet_18/conv/out_ch-512/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-2x2/padding-1x1x1x1",
        # basic block expand 3 conv 2
        "resnet_18/conv/out_ch-512/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",
        # basic block expand 3 shortcut
        "resnet_18/conv/out_ch-512/inp_ch-256/fil_shape-1x1/inp_shape-8x8/strides-2x2/padding-0x0x0x0",
        # basic block expand 3 conv 3
        "resnet_18/conv/out_ch-512/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",
        # basic block expand 3 conv 4
        "resnet_18/conv/out_ch-512/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",
        # head globalavgpool
        "resnet_18/global_avgpool/inp_shape-512x4x4",
        # head dense
        "resnet_18/dense/inp_features-512/out_features-100",
        # add layers
        "resnet_18/add/inp_shape-64x32x32",
        "resnet_18/add/inp_shape-128x16x16",
        "resnet_18/add/inp_shape-256x8x8",
        "resnet_18/add/inp_shape-512x4x4",
    ]
    MATCH_OP_TO_CREATE_LAYER_FUNCTION ={
        "conv":create_fp_conv_ex,
        "dense":create_fp_dense_ex,
        "global_avgpool":create_fp_globalavgpool_ex,
        "add":create_fp_add_ex,
    }
    compile_single_layers(layers=layers,base_path=base_path,
                          layer_function_mapper=MATCH_OP_TO_CREATE_LAYER_FUNCTION,
                          target=target,)
    print("Iterations for each unique layer:")
    for layer in sorted(list(set(layers))):
        print(sum([layer==layer_ for layer_ in layers]),":",layer)
    
def compile_resnet_18_fp_fix_div2_layer_by_layer(base_path:str="./builds"):
    #define HW Target inside match
    target=PulpPlatform()
    layers = [
        "resnet_18_fix_div2/conv/out_ch-256/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",# fix for "resnet_18/conv/out_ch-512/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",
        "resnet_18_fix_div2/conv/out_ch-256/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-2x2/padding-1x1x1x1",# fix for "resnet_18/conv/out_ch-512/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-2x2/padding-1x1x1x1",
        "resnet_18_fix_div2/conv/out_ch-128/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-1x1/padding-1x1x1x1",# fix for "resnet_18/conv/out_ch-256/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-1x1/padding-1x1x1x1",
        "resnet_18_fix_div2/conv/out_ch-128/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-2x2/padding-1x1x1x1",# fix for "resnet_18/conv/out_ch-256/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-2x2/padding-1x1x1x1",
    ]
    MATCH_OP_TO_CREATE_LAYER_FUNCTION ={
        "conv":create_fp_conv_ex,
        "dense":create_fp_dense_ex,
        "global_avgpool":create_fp_globalavgpool_ex,
        "add":create_fp_add_ex,
    }
    compile_single_layers(layers=layers,base_path=base_path,
                          layer_function_mapper=MATCH_OP_TO_CREATE_LAYER_FUNCTION,
                          target=target,)
    
def compile_resnet_18_fp_fix_div4_layer_by_layer(base_path:str="./builds"):
    #define HW Target inside match
    target=PulpPlatform()
    layers = [
        "resnet_18_fix_div4/conv/out_ch-128/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",# fix for "resnet_18/conv/out_ch-512/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",
        "resnet_18_fix_div4/conv/out_ch-128/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-2x2/padding-1x1x1x1",# fix for "resnet_18/conv/out_ch-512/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-2x2/padding-1x1x1x1",
    ]
    MATCH_OP_TO_CREATE_LAYER_FUNCTION ={
        "conv":create_fp_conv_ex,
        "dense":create_fp_dense_ex,
        "global_avgpool":create_fp_globalavgpool_ex,
        "add":create_fp_add_ex,
    }
    compile_single_layers(layers=layers,base_path=base_path,
                          layer_function_mapper=MATCH_OP_TO_CREATE_LAYER_FUNCTION,
                          target=target,)

def compile_resnet_18_fp_fix_div8_layer_by_layer(base_path:str="./builds"):
    #define HW Target inside match
    target=PulpPlatform()
    layers = [
        "resnet_18_fix_div8/conv/out_ch-64/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",# fix for "resnet_18/conv/out_ch-512/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",
    ]
    MATCH_OP_TO_CREATE_LAYER_FUNCTION ={
        "conv":create_fp_conv_ex,
        "dense":create_fp_dense_ex,
        "global_avgpool":create_fp_globalavgpool_ex,
        "add":create_fp_add_ex,
    }
    compile_single_layers(layers=layers,base_path=base_path,
                          layer_function_mapper=MATCH_OP_TO_CREATE_LAYER_FUNCTION,
                          target=target,)

def create_fp_conv_vit(div_out_chs_by:int=1, out_ch:int=384):
    np.random.seed(0)
    inp_shape = (224,224)
    inp_ch = 3
    fil_shape = (16,16)
    strides = (16,16)
    out_ch = out_ch//div_out_chs_by
    padding = (0,0,0,0)
    groups = 1
    x = relay.var("input_0", relay.TensorType((1,inp_ch)+inp_shape))
    # Get or generate weight_values
    weights_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_ch,inp_ch)+fil_shape),dtype="float32")
    # Get or generate bias values
    bias_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_ch,)),dtype="float32")
    concat_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,out_ch)),dtype="float32")
    add_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,197,out_ch)),dtype="float32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "pulp_conv_weights"
    bias_name = "pulp_conv_bias"
    concat_name = "pulp_conv_concat"
    add_name = "pulp_conv_add"
    # define weights and bias values in params
    params = {weights_name: weights_c, bias_name: bias_c,
              concat_name: concat_c, add_name: add_c}

    # define operations
    weights_v = relay.var(weights_name, relay.TensorType(weights_c.shape, weights_c.dtype))
    x = relay.op.nn.conv2d(x, weights_v,
                           strides=strides,
                           padding=padding,
                           groups=groups,
                           kernel_size=fil_shape,
                           )
    bias_v = relay.var(bias_name, relay.TensorType(bias_c.shape, bias_c.dtype))
    x = relay.op.nn.bias_add(x, bias_v, axis=1)
    x = relay.op.reshape(x, newshape=(1,out_ch,196))
    x = relay.op.transpose(x,axes=(0,2,1))
    concat_v = relay.var(concat_name, relay.TensorType(concat_c.shape,concat_c.dtype))
    x = relay.op.concatenate(relay.Tuple([x,concat_v]), axis=1)
    add_v = relay.var(add_name, relay.TensorType(add_c.shape,add_c.dtype))
    x = relay.op.add(x,add_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_layer_norm_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384):
    np.random.seed(0)
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((1,out_ch)+inp_shape))
    # Get or generate weight_values
    beta_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=inp_shape),dtype="float32")
    gamma_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=inp_shape),dtype="float32")
    # Generate the conv2d call
    # define weights and bias variables
    beta_name = "pulp_layer_norm_beta"
    gamma_name = "pulp_layer_norm_gamma"
    # define weights and bias values in params
    params = {beta_name: beta_c, gamma_name: gamma_c}

    # define operations
    beta_v = relay.var(beta_name, relay.TensorType(beta_c.shape, beta_c.dtype))
    gamma_v = relay.var(gamma_name, relay.TensorType(gamma_c.shape, gamma_c.dtype))
    x = relay.op.nn.layer_norm(x,gamma=gamma_v,beta=beta_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_proj_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, projections:int=3):
    np.random.seed(0)
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((out_ch,)+inp_shape))
    # Get or generate weight_values
    weights_name = "pulp_proj_weights"
    add_name = "pulp_proj_add"
    weights_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(inp_size*projections,inp_size)),dtype="float32")
    add_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,inp_size*projections)),dtype="float32")
    params = {weights_name: weights_c, add_name: add_c}
    weights_v = relay.var(weights_name, relay.TensorType(weights_c.shape, weights_c.dtype))
    x = relay.op.nn.dense(x,weights_v)
    add_v = relay.var(add_name, relay.TensorType(add_c.shape, add_c.dtype))
    x = relay.op.add(x,add_v)
    x = relay.op.reshape(x,newshape=(1,out_ch,projections,inp_size//64,64))
    x = relay.op.transpose(x,axes=(2,0,3,1,4))
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_attention_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384):
    np.random.seed(0)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((3,1,inp_size//64,out_ch,64)))
    # Get or generate weight_values
    weights_name = "pulp_attention_weights"
    add_name = "pulp_attention_add"
    sqrt_1_name = "pulp_attention_sqrt_1"
    sqrt_2_name = "pulp_attention_sqrt_2"
    weights_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(inp_size,inp_size)),dtype="float32")
    add_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,inp_size)),dtype="float32")
    sqrt_1_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,)),dtype="float32")
    sqrt_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,)),dtype="float32")
    params = {sqrt_1_name: sqrt_1_c, sqrt_2_name: sqrt_2_c, add_name: add_c, weights_name: weights_c}
    x = relay.op.split(x,indices_or_sections=(1,2))
    q = relay.op.reshape(x[0], newshape=(1,1,inp_size//64,out_ch,64))
    k = relay.op.reshape(x[1], newshape=(1,1,inp_size//64,out_ch,64))
    v = relay.op.reshape(x[2], newshape=(1,1,inp_size//64,out_ch,64))
    q = relay.op.squeeze(q,axis=0)
    sqrt_1_v = relay.var(sqrt_1_name, relay.TensorType(sqrt_1_c.shape, sqrt_1_c.dtype))
    q = relay.op.multiply(q,sqrt_1_v)
    k = relay.op.squeeze(k,axis=0)
    k = relay.op.transpose(k,axes=(0,1,3,2))
    sqrt_2_v = relay.var(sqrt_2_name, relay.TensorType(sqrt_2_c.shape, sqrt_2_c.dtype))
    k = relay.op.multiply(k,sqrt_2_v)
    k = relay.op.reshape(k,newshape=(-1,64,out_ch))
    k = relay.op.transpose(k, axes=(0,2,1))
    q = relay.op.reshape(q,newshape=(-1,out_ch,64))
    qk = relay.op.nn.batch_matmul(q,k)
    qk = relay.op.nn.softmax(qk,axis=-1)
    v = relay.op.squeeze(v,axis=0)
    v = relay.op.reshape(v,newshape=(-1,out_ch,64))
    v = relay.op.transpose(v, axes=(0,2,1))
    qk = relay.op.reshape(qk,newshape=(-1,out_ch,out_ch))
    x = relay.op.nn.batch_matmul(qk,v)
    x = relay.op.reshape(x,newshape=(1,inp_size//64,out_ch,64))
    x = relay.op.transpose(x,axes=(0,2,1,3))
    x = relay.op.reshape(x,newshape=(out_ch,inp_size))
    weights_v = relay.var(weights_name, relay.TensorType(weights_c.shape, weights_c.dtype))
    x = relay.op.nn.dense(x,weights_v)
    add_v = relay.var(add_name, relay.TensorType(add_c.shape, add_c.dtype))
    x = relay.op.add(x,add_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_addlayernorm_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384):
    np.random.seed(0)
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((1,out_ch)+inp_shape))
    y = relay.var("input_1", relay.TensorType((1,out_ch)+inp_shape))
    # Get or generate weight_values
    beta_name = "pulp_layer_norm_beta"
    gamma_name = "pulp_layer_norm_gamma"
    beta_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=inp_shape),dtype="float32")
    gamma_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=inp_shape),dtype="float32")
    params = {beta_name: beta_c, gamma_name: gamma_c}
    x = relay.op.add(x,y)
    beta_v = relay.var(beta_name, relay.TensorType(beta_c.shape, beta_c.dtype))
    gamma_v = relay.var(gamma_name, relay.TensorType(gamma_c.shape, gamma_c.dtype))
    x = relay.op.nn.layer_norm(x,gamma=gamma_v,beta=beta_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_add_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384):
    np.random.seed(0)
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((1,out_ch)+inp_shape))
    y = relay.var("input_1", relay.TensorType((1,out_ch)+inp_shape))
    # Get or generate weight_values
    params = {}
    x = relay.op.add(x,y)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_classifier_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, out_neurons:int=10):
    np.random.seed(0)
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((1,out_ch)+inp_shape))
    # Get or generate weight_values
    weights_name = "pulp_classifier_weights"
    add_name = "pulp_classifier_add"
    weights_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_neurons,inp_size)),dtype="float32")
    add_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,out_neurons)),dtype="float32")
    params = {weights_name: weights_c, add_name: add_c}
    x = relay.op.take(x,relay.const(0),axis=1)
    weights_v = relay.var(weights_name, relay.TensorType(weights_c.shape, weights_c.dtype))
    x = relay.op.nn.dense(x,weights_v)
    add_v = relay.var(add_name, relay.TensorType(add_c.shape, add_c.dtype))
    x = relay.op.add(x,add_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def gelu_helper(data):
    const1 = relay.const(math.sqrt(2.0))
    const2 = relay.const(1.0)
    const3 = relay.const(0.5)
    divisor = relay.op.divide(data, const1)
    val_erf = relay.op.erf(divisor)
    added_erf = relay.op.add(val_erf, const2)
    mul1 = relay.op.multiply(data, added_erf)
    out = relay.op.multiply(mul1, const3)
    return out

def create_densegelu_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, projections:int=3):
    np.random.seed(0)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((out_ch,inp_size)))
    # Get or generate weight_values
    add_1_name = "pulp_mlp_add_1"
    add_2_name = "pulp_mlp_add_2"
    weights_1_name = "pulp_mlp_weights_1"
    weights_2_name = "pulp_mlp_weights_2"
    add_1_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=((1,1,(inp_size+128)*projections))),dtype="float32")
    add_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,inp_size)),dtype="float32")
    weights_1_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=((inp_size+128)*projections,inp_size)),dtype="float32")
    weights_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(inp_size,(inp_size+128)*projections)),dtype="float32")
    params = {add_1_name: add_1_c, add_2_name: add_2_c, weights_1_name: weights_1_c, weights_2_name: weights_2_c}
    weights_1_v = relay.var(weights_1_name, relay.TensorType(weights_1_c.shape, weights_1_c.dtype))
    x = relay.op.nn.dense(x,weights_1_v)
    add_1_v = relay.var(add_1_name, relay.TensorType(add_1_c.shape, add_1_c.dtype))
    x = relay.op.add(x,add_1_v)
    x = gelu_helper(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_densemlp_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, projections:int=3):
    np.random.seed(0)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((out_ch,(inp_size+128)*projections)))
    # Get or generate weight_values
    add_2_name = "pulp_mlp_add_2"
    weights_2_name = "pulp_mlp_weights_2"
    add_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,inp_size)),dtype="float32")
    weights_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(inp_size,(inp_size+128)*projections)),dtype="float32")
    params = {add_2_name: add_2_c, weights_2_name: weights_2_c}
    weights_2_v = relay.var(weights_2_name, relay.TensorType(weights_2_c.shape, weights_2_c.dtype))
    x = relay.op.nn.dense(x,weights_2_v)
    add_2_v = relay.var(add_2_name, relay.TensorType(add_2_c.shape, add_2_c.dtype))
    x = relay.op.add(x,add_2_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def compile_vit_fp_layer_by_layer(base_path:str="./builds"):
    #define HW Target inside match
    target=PulpPlatform()
    layers = [
        # "vit_fp_build_by_8/conv/div_out_chs_by-8", # DONE WITH 8
        # "vit_fp_build/layernorm", # DONE WITH 1
        # "vit_fp_build/proj/projections-1", # DONE WITH 1 PROJS 1
        # "vit_fp_build_by_3/attention/div_out_chs_by-3/out_ch-198", # DONE WITH 3
        # "vit_fp_build/addlayernorm", # DONE WITH 1
        # "vit_fp_build_by_2/densegelu/projections-1/div_out_chs_by-2/out_ch-198", 
        "vit_fp_build_by_2/densemlp/projections-1/div_out_chs_by-2/out_ch-198", 
        # "vit_fp_build/add", # DONE WITH 1
        # "vit_fp_build/classifier", # DONE WITH 1
    ]
    MATCH_OP_TO_CREATE_LAYER_FUNCTION ={
        "conv":create_fp_conv_vit,
        "layernorm":create_fp_layer_norm_vit,
        "proj":create_fp_proj_vit,
        "attention":create_attention_fp_vit,
        "addlayernorm":create_addlayernorm_fp_vit,
        "densegelu":create_densegelu_fp_vit,
        "densemlp":create_densemlp_fp_vit,
        "add":create_add_fp_vit,
        "classifier":create_classifier_fp_vit,
    }
    compile_single_layers(layers=layers,base_path=base_path,
                          layer_function_mapper=MATCH_OP_TO_CREATE_LAYER_FUNCTION,
                          target=target,)
    #print("Iterations for each unique layer:")
    #for layer in sorted(list(set(layers))):
    #    print(sum([layer==layer_ for layer_ in layers]),":",layer)


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
            run_model(model=args.model, output_path=str(Path("./builds/last_build").absolute()))
        else:
            run_microbench(microbench=args.microbench,output_path=str(Path("./builds/last_build").absolute()))