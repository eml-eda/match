import json
import os
from pathlib import Path

import numpy as np

from networks import RESNET_18_MAPPER, RESNET_18_NODES, VIT_MAPPER, VIT_NODES

from quants import create_conv_ex, create_dense_ex, create_simple_sum_ex, create_sumnet_ex
from mini import create_dense_conv_dense_ex, create_easy_dense_int32_ex


MICROBENCH_MAPPER = {
    "conv":create_conv_ex,
    "dense":create_dense_ex,
    "simple_sum":create_simple_sum_ex,
    "dense_conv_dense":create_dense_conv_dense_ex,
    "easy_dense_int32":create_easy_dense_int32_ex,
    "sumnet":create_sumnet_ex,
}

NETWORK_MAPPER = {
    "resnet_18": (RESNET_18_MAPPER, RESNET_18_NODES),
    "vit": (VIT_MAPPER, VIT_NODES)
}

def get_network_single_nodes(network_name: str="resnet_18"):
    if network_name not in NETWORK_MAPPER:
        raise Exception(f"{network_name} network is not available, the networks tests available are {[k for k in NETWORK_MAPPER.keys()]}")
    mapper, nodes = NETWORK_MAPPER[network_name]
    single_node_mod_params = []
    for node_name in set(nodes):
        node_name_split = node_name.split("/")
        node_type = node_name_split[1]
        config = dict()
        # use default config
        for layer_config in node_name_split[2:]:
            layer_config_split = layer_config.split("-")
            key = layer_config_split[0]
            value = layer_config_split[1]
            if len(value.split("x"))>1:
                value = tuple([int(v) for v in value.split("x")])
            elif value.isnumeric():
                value = int(value)
            config[key] = value
        # add configs of node configs files if available
        if Path(os.path.dirname(__file__)+f"/node_config_{node_type}.json").is_file():
            with open(os.path.dirname(__file__)+f"/node_config_{node_type}.json") as config_file:
                config = json.load(config_file)
                for config_key in config:
                    if isinstance(config[config_key],list):
                        config[config_key] = tuple(config[config_key])
        # set random to 0 for weights and constants generation...
        np.random.seed(0)
        mod,params = mapper[node_type](**config)
        single_node_mod_params.append((node_type, mod, params))

def get_microbench_mod(microbench_name: str="conv"):
    if microbench_name not in MICROBENCH_MAPPER:
        raise Exception(f"{microbench_name} microbench is not available, the microbench tests available are {[k for k in MICROBENCH_MAPPER.keys()]}")
    config = dict()
    if Path(os.path.dirname(__file__)+"/node_config.json").is_file():
        with open(os.path.dirname(__file__)+"/node_config.json") as config_file:
            config = json.load(config_file)
            for config_key in config:
                if isinstance(config[config_key],list):
                    config[config_key] = tuple(config[config_key])
    # set random to 0 for weights and constants generation...
    np.random.seed(0)
    return MICROBENCH_MAPPER[microbench_name](**config)