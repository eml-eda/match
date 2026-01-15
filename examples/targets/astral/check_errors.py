import numpy as np
import json
import torch

memory_plan_metadata = dict()
mod_metadata = dict()

def get_qyolo_layer_data(num_layer, plinio_num_layer):
    plinio_filename = f"./models/qyolo_features/features/out_layer{plinio_num_layer}.txt"
    match_filename = f"./qyolo_graph_host/models/qyolo/golden/qyolo_qyolo_node_{num_layer}_out_data.hex"
    quant_data = np.loadtxt(plinio_filename, delimiter=',', dtype=np.dtype(qyolo_layer_dtype[num_layer]), usecols=[0])
    bytes__ = bytes()
    with open(match_filename, "rb") as f_:  bytes__ = f_.read()
    my_data = np.frombuffer(bytes__, qyolo_layer_dtype[num_layer])
    plinio_layer_name = ""
    with open(plinio_filename, "r") as f__:
        plinio_layer_name = f__.readline().strip().split()[1]
    return my_data, quant_data, plinio_layer_name

def get_qyolo_out_data(num_output):
    # plinio_filename = f"./models/qyolo_features/features/output_{num_output}.txt"
    plinio_filename = f"./models/qyolo_inp_oup/oup_net{num_output}.pt"
    match_filename = f"./qyolo_graph_host/models/qyolo/golden/qyolo_qyolo_out_{num_output}_data.hex"
    # quant_data = np.loadtxt(plinio_filename, delimiter=',', dtype=np.float32, usecols=[0])
    quant_data = torch.load(plinio_filename).flatten()
    bytes__ = bytes()
    with open(match_filename, "rb") as f_:  bytes__ = f_.read()
    my_data = np.frombuffer(bytes__, np.float32)
    plinio_layer_name = ""
    return my_data, quant_data, plinio_layer_name

def get_plinio_layer_name(plinio_num_layer):
    plinio_filename = f"./models/qyolo_features/features/out_layer{plinio_num_layer}.txt"
    plinio_layer_name = ""
    with open(plinio_filename, "r") as f__:
        plinio_layer_name = f__.readline().strip().split()[1]
    return plinio_layer_name

with open("./qyolo_graph_host/models/qyolo/metadata/memory_plan_metadata.json", "r") as f:
    memory_plan_metadata = json.load(f)

with open("./qyolo_graph_host/models/qyolo/metadata/mod.json", "r") as f:
    mod_metadata = json.load(f)

qyolo_num_layers = list()
qyolo_layer_dtype = dict()
for node_name, node in memory_plan_metadata.items():
    if "qyolo_node_" in node["name"] and "out" in node["name"]:
        num_layer = node["name"].split("qyolo_node_")[1].split("_out")[0]
        num_layer = int(num_layer)
        used_at_plinio_layers = False
        # for node_idx in node["used_at"]:
            # used_at_node = mod_metadata["nodes"][node_idx]
            # if "conv2d" in used_at_node["name"] or "add" in used_at_node["name"]:
                # used_at_plinio_layers = True
                # break
        used_at_node = mod_metadata["nodes"][num_layer]
        if "conv2d" in used_at_node["name"] or "add" in used_at_node["name"]:
            used_at_plinio_layers = True
        if used_at_plinio_layers:
            qyolo_num_layers.append(num_layer)
            qyolo_layer_dtype[num_layer] = node["dtype"]

qyolo_num_layers = sorted(qyolo_num_layers)
plinio_num_layer = 0

match_plinio_map = dict()
plinio_name_map = dict()

for num_layer in range(96):
    plinio_layer_name = get_plinio_layer_name(plinio_num_layer)
    if "conv" in plinio_layer_name or "add" in plinio_layer_name:
        plinio_name_map[plinio_num_layer] = plinio_layer_name
    plinio_num_layer += 1

# plinio_ordered_layers = 

# plinio_num_layer = 0

""" for num_layer in qyolo_num_layers:
    my_data, quant_data, plinio_layer_name = get_qyolo_layer_data(num_layer, plinio_num_layer)
    if my_data.shape != quant_data.shape:
        print(f"Shape mismatch at plinio layer {plinio_num_layer} / qyolo layer {num_layer} ❌")
        print(f"  my_data shape: {my_data.shape}, quant_data shape: {quant_data.shape}")
        plinio_num_layer += 1
        continue
    diffs = [(i,my_data[i],quant_data[i]) for i in np.where(my_data!=quant_data)[0]]
    if len(diffs) > 0:
        print(f"Layer mismatch at plinio layer {plinio_num_layer} / qyolo layer {num_layer} ❌")
        # print("quant_data [DUMP FEATURES]", quant_data.shape, quant_data.dtype)
        # print("my_data [MATCH RES]", my_data.shape, my_data.dtype)

        print(len(diffs), "differences")
        # print(f"First 100 differences (index, my_data, quant_data): {diffs[:100]}")
        # break
    else:
        print(f"Layer match at plinio layer {plinio_num_layer} / qyolo layer {num_layer} ✅")
    plinio_num_layer += 1 """

print("Starting plinio to QYolo layer matching...")
print("QYolo layers:", qyolo_num_layers)
print("Plinio layers:", list(plinio_name_map.keys()))

idx_plinio_layer = 0
for plinio_num_layer in sorted(plinio_name_map.keys()):
    found_correct_layer = False
    num_qyolo_layer = -1
    print(f"Checking plinio layer {plinio_num_layer} ({plinio_name_map[plinio_num_layer]})")
    for num_layer in qyolo_num_layers:
        if num_layer in match_plinio_map.values():
            continue
        my_data, quant_data, plinio_layer_name = get_qyolo_layer_data(num_layer, plinio_num_layer)
        if my_data.shape != quant_data.shape:
            # print(f"Shape mismatch at plinio layer {plinio_num_layer} / qyolo layer {num_layer} ❌")
            # print(f"  my_data shape: {my_data.shape}, quant_data shape: {quant_data.shape}")
            # plinio_num_layer += 1
            # print(num_layer , "shape mismatch", my_data.shape, quant_data.shape)
            continue
        diffs = [(i,my_data[i],quant_data[i]) for i in np.where(my_data!=quant_data)[0]]
        if len(diffs) > 0:
            # print(f"Layer mismatch at plinio layer {plinio_num_layer} / qyolo layer {num_layer} ❌")
            # print("quant_data [DUMP FEATURES]", quant_data.shape, quant_data.dtype)
            # print("my_data [MATCH RES]", my_data.shape, my_data.dtype)

            # print(num_layer , len(diffs), "differences")
            # print(f"First 100 differences (index, my_data, quant_data): {diffs[:100]}")
            # break
            continue
        else:
            found_correct_layer = True
            num_qyolo_layer = num_layer
            break
    if found_correct_layer:
        match_plinio_map[plinio_num_layer] = num_qyolo_layer
        print(f"Found matching QYolo layer {num_qyolo_layer} for Plinio layer {plinio_num_layer} ({plinio_name_map[plinio_num_layer]}) ✅")
    else:
        print(f"No matching QYolo layer found for Plinio layer {plinio_num_layer} ({plinio_name_map[plinio_num_layer]}) ❌")
    print("")
    idx_plinio_layer += 1

print("Plinio to QYolo layer mapping:")
for plinio_layer, qyolo_layer in match_plinio_map.items():
    print(f"Plinio layer {plinio_layer} ({plinio_name_map[plinio_layer]}) -> QYolo layer {qyolo_layer}")

for output_idx in range(4):
    my_data, quant_data, plinio_layer_name = get_qyolo_out_data(output_idx)
    if my_data.shape != quant_data.shape:
        print(f"Shape mismatch at output {output_idx} ❌")
        print(f"  my_data shape: {my_data.shape}, quant_data shape: {quant_data.shape}")
        continue
    else:
        print(f"Output {output_idx} shape match: {my_data.shape}")
    diffs = [(i,my_data[i],quant_data[i]) for i in np.where(my_data!=quant_data)[0]]
    if len(diffs) > 0:
        print(f"Output mismatch at output {output_idx} ❌")
        print(len(diffs), "differences")
    else:
        print(f"Output match at output {output_idx} ✅")