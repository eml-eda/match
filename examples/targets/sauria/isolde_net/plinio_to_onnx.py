import onnx
import os
import numpy as np
import copy
from collections import Counter
import argparse
from onnx import helper, numpy_helper, shape_inference

ACCELERATED_NODES = ("MatMul", "Gemm", "Conv", "Add", "ConvTranspose")
CONST_ACCELERATED_OPS = ("Add", "Mul", "Div")
OUT_DTYPE_NODES = ("MatMul", "Gemm", "Conv", "ConvTranspose")

def lift_onnx_1d_to_2d(input_path, output_path, mode="height"):
    """
    Lifts 1D ONNX model to 2D.
    mode="height": Transforms [N, C, W] -> [N, C, W_old, 1]
    mode="width":  Transforms [N, C, W] -> [N, C, 1, W_old]
    """
    model = onnx.load(input_path)
    graph = model.graph
    is_height = (mode == "height")

    def reshape_tensor(proto):
        arr = numpy_helper.to_array(proto)
        if len(arr.shape) == 3: # Weight: [O, I, W]
            new_shape = (arr.shape[0], arr.shape[1], arr.shape[2], 1) if is_height else \
                        (arr.shape[0], arr.shape[1], 1, arr.shape[2])
            new_proto = numpy_helper.from_array(arr.reshape(new_shape), name=proto.name)
            proto.CopyFrom(new_proto)

    # 1. Update Initializers & Constant Nodes
    for init in graph.initializer:
        reshape_tensor(init)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    reshape_tensor(attr.t)

    # 2. Update Node Attributes (Conv, Pool, etc.)
    for node in graph.node:
        if node.op_type in ["Conv", "MaxPool", "AveragePool"]:
            for attr in node.attribute:
                if attr.name in ["kernel_shape", "strides", "dilations"]:
                    v = list(attr.ints) # [W]
                    attr.ints[:] = [v[0], 1] if is_height else [1, v[0]]
                elif attr.name == "pads":
                    v = list(attr.ints) # [L, R]
                    # Format: [H_begin, W_begin, H_end, W_end]
                    attr.ints[:] = [v[0], 0, v[1], 0] if is_height else [0, v[0], 0, v[1]]
        # Pad Node (Handles explicit Pad operators)
        elif node.op_type == "Pad":
            for attr in node.attribute:
                if attr.name == "pads":
                    v = list(attr.ints)
                    # For a 3D input [N, C, W], pads is [n_begin, c_begin, w_begin, n_end, c_end, w_end]
                    if len(v) == 6:
                        nb, cb, wb, ne, ce, we = v
                        # Target 4D: [N, C, H, W]
                        if is_height:
                            # Map W to H, set W padding to 0
                            attr.ints[:] = [nb, cb, wb, 0, ne, ce, we, 0]
                        else:
                            # Map W to W, set H padding to 0
                            attr.ints[:] = [nb, cb, 0, wb, ne, ce, 0, we]
    # 3. Update Value Info (Inputs, Outputs, and Intermediates)
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        shape = vi.type.tensor_type.shape
        if len(shape.dim) == 3: # [N, C, W]
            d0, d1, d2 = shape.dim[0], shape.dim[1], shape.dim[2]
            # Clear and rebuild as 4D
            new_dims = [d0, d1, d2, 1] if is_height else [d0, d1, 1, d2]
            while len(shape.dim) > 0: shape.dim.pop()
            for d in new_dims:
                new_d = shape.dim.add()
                if hasattr(d, 'dim_value') and d.dim_value > 0: new_d.dim_value = d.dim_value
                elif hasattr(d, 'dim_param'): new_d.dim_param = d.dim_param
                else: new_d.dim_value = d # integer case

    # 4. Global Shape Inference
    # Broadcasting for Add/Mul/Normalization nodes is handled automatically here.
    inferred_model = shape_inference.infer_shapes(model)
    onnx.save(inferred_model, output_path)
    print(f"Model lifted to 2D ({mode}-major) and saved to {output_path}")

def remove_plinio_attrs(model_path, new_model_path):
    onnx_model = onnx.load(model_path)
    new_nodes = list()
    for node_idx,node in enumerate(onnx_model.graph.node):
        new_node = copy.deepcopy(node)
        onnx_update_node_attributes(new_node)
        new_nodes.append(new_node)
    for node_idx in range(len(onnx_model.graph.node)):
        # remove the original node
        onnx_model.graph.node.pop(0)
    onnx_model.graph.node.extend(new_nodes)
    with open(new_model_path, "wb") as new_f:
        onnx.save(onnx_model,new_f)

def remove_first_unsqueeze(model_path, new_model_path):
    onnx_model = onnx.load(model_path)
    new_nodes = list()
    if onnx_model.graph.node[0].op_type != "Unsqueeze":
        print("The first node is not an Unsqueeze, nothing to do")
        return
    input_name = onnx_model.graph.node[0].input[0]
    unsqueeze_out_name = onnx_model.graph.node[0].output[0]
    axes = onnx_model.graph.node[0].attribute[0].ints
    new_inputs = list()
    for input__ in onnx_model.graph.input:
        if input__.name == input_name:
            shape = [d__.dim_value for d__ in input__.type.tensor_type.shape.dim]
            for axis in axes:
                shape.insert(axis, 1)
            new_inputs.append(
                onnx.helper.make_tensor_value_info(
                    name="inp",
                    elem_type=input__.type.tensor_type.elem_type,
                    shape=shape,
                )
            )
        else:
            new_inputs.append(input__)
    for _ in range(len(onnx_model.graph.input)):
        onnx_model.graph.input.pop(0)
    onnx_model.graph.input.extend(new_inputs)
    for node_idx, node in enumerate(onnx_model.graph.node):
        if node_idx == 0:
            # skip the first unsqueeze
            continue
        for input_idx, input__ in enumerate(node.input):
            if input__ == unsqueeze_out_name:
                # replace the input with the new one
                node.input[input_idx] = "inp"
        new_nodes.append(node)
    for node_idx in range(len(onnx_model.graph.node)):
        # remove the original node
        onnx_model.graph.node.pop(0)
    # add the new nodes
    onnx_model.graph.node.extend(new_nodes)
    with open(new_model_path, "wb") as new_f:
        onnx.save(onnx_model,new_f)

def merge_pad_with_next_op(model_path, new_model_path):
    onnx_model = onnx.load(model_path)
    prev_output = None
    prev_input = None
    prev_pad_pads = None
    prev_pad_mode = None
    prev_pad_value = None
    new_nodes = list()
    for node_idx, node in enumerate(onnx_model.graph.node):
        if node.op_type == "Pad":
            prev_output = node.output[0]
            prev_input = node.input[0]
            for attr__ in node.attribute:
                if attr__.name == "pads":
                    prev_pad_pads = attr__.ints
                    if len(prev_pad_pads) == 6:
                        prev_pad_pads = (prev_pad_pads[2], prev_pad_pads[5])
                    elif len(prev_pad_pads) > 6:
                        if len(prev_pad_pads) % 2 != 0:
                            raise ValueError(f"The pads attribute of the Pad node {node.name} should have an even number of elements, got {len(prev_pad_pads)}")
                        num_dims = len(prev_pad_pads) // 2
                        diff_dims = num_dims - 3
                        prev_pad_pads = prev_pad_pads[diff_dims:diff_dims+3] + prev_pad_pads[num_dims+diff_dims:num_dims+diff_dims+3]
                elif attr__.name == "mode":
                    prev_pad_mode = attr__.s.decode("utf-8")
                elif attr__.name == "value":
                    prev_pad_value = attr__.f if attr__.type == onnx.TensorProto.FLOAT else None
        else:
            if prev_output is not None:
                new_attributes = list()
                new_attributes.append(
                    onnx.helper.make_attribute(
                        key="pads",
                        value=prev_pad_pads if prev_pad_pads is not None else [0, 0, 0, 0, 0, 0]
                    )
                )
                for attr__ in node.attribute:
                    if attr__.name != "auto_pad" and attr__.name != "pads":
                        new_attributes.append(attr__)
                for _ in range(len(node.attribute)):
                    node.attribute.pop(0)
                node.attribute.extend(new_attributes)
                for input_idx, input__ in enumerate(node.input):
                    if input__ == prev_output:
                        # replace the input with the previous output
                        node.input[input_idx] = prev_input
                # reset the previous output
                prev_output = None
                prev_input = None
                prev_pad_pads = None
                prev_pad_mode = None
                prev_pad_value = None
            new_nodes.append(node)
    for node_idx in range(len(onnx_model.graph.node)):
        # remove the original node
        onnx_model.graph.node.pop(0)
    # add the new nodes
    onnx_model.graph.node.extend(new_nodes)
    with open(new_model_path, "wb") as new_f:
        onnx.save(onnx_model,new_f)

def onnx_dtype_from_np_dtype(np_dtype):
    return onnx.helper.np_dtype_to_tensor_dtype(np_dtype)

def onnx_is_weights(onnx_model, weights_name, new_weights_name):
    for init__ in onnx_model.graph.initializer:
        if init__.name == weights_name or init__.name == new_weights_name:
            return True
    return False

def onnx_update_weight_to(onnx_model, weights_name, new_weights_name, dtype, ndims=4, broadcast_dim_idx=3):
    for init__ in onnx_model.graph.initializer:
        if init__.name == weights_name:
            init__.name = new_weights_name
            init__.raw_data = (
                np.frombuffer(
                    init__.raw_data,
                    onnx.helper.tensor_dtype_to_np_dtype(
                        init__.data_type
                    ),
                )
                .astype(dtype)
                .tobytes()
            )
            if len(init__.dims)!=ndims:
                if len(init__.dims)!=1:
                    raise ValueError(f"The weight {new_weights_name} should be 1D or {ndims}D")
                broadcast_dim_idx = broadcast_dim_idx if broadcast_dim_idx>0 else ndims-1
                broadcast_dim_size = init__.dims.pop(0)
                for fake_shape_idx in range(ndims):
                    init__.dims.append(1 if fake_shape_idx!=broadcast_dim_idx else broadcast_dim_size)
            init__.data_type = onnx_dtype_from_np_dtype(dtype)
            return True
    return False

def onnx_update_inputs_to(onnx_model, dtype):
    for input__ in onnx_model.graph.input:
        input__.type.tensor_type.elem_type = onnx_dtype_from_np_dtype(dtype)

def onnx_update_outputs_to(onnx_model, dtype):
    for output__ in onnx_model.graph.output:
        output__.type.tensor_type.elem_type = onnx_dtype_from_np_dtype(dtype)

def onnx_update_node_attributes(node):
    original_attributes = [attr__ for attr__ in node.attribute]
    for attribute in original_attributes:
        # Plinio-MATCH exporter add some attributes that are not standard-ONNX so they should be removed
        if not ("_" in attribute.name and attribute.name.split("_")[1]=="bits"):
            node.attribute.append(attribute)
        node.attribute.pop(0)

def add_cast_as(onnx_model, new_nodes, node_idx, node, cast_type, cast_name: str="cast"):
    new_nodes.append(
        onnx.helper.make_node(
            name=cast_name,
            op_type="Cast",
            inputs=[node.output[0]],
            outputs=[node.output[0] + "_" + cast_name],
            to=onnx.helper.np_dtype_to_tensor_dtype(cast_type),
        ),
    )
    for other_node in onnx_model.graph.node[node_idx+1:]:
        for i in range(len(other_node.input)):
            if other_node.input[i] == node.output[0]:
                other_node.input[i] = node.output[0] + "_" + cast_name

def add_cast_as_of_input(new_nodes, cast_type, input_name: str="input_0", cast_name: str="cast"):
    new_nodes.append(
        onnx.helper.make_node(
            name=cast_name,
            op_type="Cast",
            inputs=[input_name],
            outputs=[input_name + "_" + cast_name],
            to=onnx.helper.np_dtype_to_tensor_dtype(cast_type),
        ),
    )

def onnx_update_accelerated_node(onnx_model, new_nodes, node_idx, node, dtype):
    # it may that the input is detached due to a bug about naming in Plinio-MATCH exporter
    integerize = np.issubdtype(dtype, np.integer)
    weights_dtype = dtype if not integerize else dtype if dtype.kind == 'i' else np.dtype('i' + str(dtype.itemsize))
    const_dtype = np.dtype("int32") if integerize else np.dtype("float32")
    ndims = 5 if node.op_type in ("Conv", "ConvTranspose") else 2
    broadcast_dim_idx = 1
    # if integerize:
    #     node.op_type = node.op_type+"Integer"
    original_inputs = [str(inp__) for inp__ in node.input]
    other_constants = list()
    for input_idx, input__ in enumerate(original_inputs):
        if "::" in input__:
            new_input_name = "_".join(input__.split("::"))
        else:
            new_input_name = input__
        
        if input_idx<2:
            if onnx_is_weights(onnx_model, input__, new_input_name):
                onnx_update_weight_to(onnx_model, input__, new_input_name, weights_dtype, ndims, broadcast_dim_idx)
            if node.op_type not in OUT_DTYPE_NODES:
                add_cast_as_of_input(new_nodes, const_dtype, new_input_name, "FAKE_CAST_TVM_IN_DTYPE")
                node.input.append(new_input_name + "_FAKE_CAST_TVM_IN_DTYPE")
            else:
                node.input.append(new_input_name)
        else:
            if onnx_is_weights(onnx_model, input__, new_input_name):
                onnx_update_weight_to(onnx_model, input__, new_input_name, const_dtype, ndims, broadcast_dim_idx)
            other_constants.append((new_input_name, "Add"))
        node.input.pop(0)
    new_nodes.append(node)
    if dtype.itemsize!=4:
        cast_name = "accumulator"
        if node.op_type in OUT_DTYPE_NODES:
            cast_name = "FAKE_CAST_TVM_OUT_DTYPE"
            add_cast_as(onnx_model, new_nodes, node_idx, node, const_dtype, cast_name)
    return other_constants

def onnx_update_constant_to(node, dtype):
    # read input and reinterpret it as int32
    node.attribute[0].t.raw_data = (
        np.frombuffer(
            node.attribute[0].t.raw_data,
            onnx.helper.tensor_dtype_to_np_dtype(
                node.attribute[0].t.data_type
            ),
        )
        .astype(dtype)
        .tobytes()
    )
    node.attribute[0].t.data_type = onnx_dtype_from_np_dtype(dtype)

def onnx_update_nodes(onnx_model, dtype):
    new_nodes = list()
    floating = np.issubdtype(dtype, np.floating)
    single_prec_constant_dtype = np.dtype("float32") if floating else np.dtype("int32")
    constants_ops_to_add = list()
    in_fused_node = False
    for node_idx,node in enumerate(onnx_model.graph.node):
        new_node = copy.deepcopy(node)
        onnx_update_node_attributes(new_node)
        if node.op_type in ACCELERATED_NODES and not in_fused_node:
            constants_ops_to_add.extend(onnx_update_accelerated_node(onnx_model, new_nodes, node_idx, new_node, dtype))
            in_fused_node = True
        elif node.op_type in CONST_ACCELERATED_OPS:
            new_nodes.append(new_node)
            new_output = new_node.output[0]
            for const_op_idx,(const_name, op_type) in enumerate(constants_ops_to_add):
                new_output = new_node.output[0] + "_const_" + str(const_op_idx)
                new_nodes.append(
                    onnx.helper.make_node(
                        name="ONNX_BIAS_SCALE_CONST",
                        op_type=op_type,
                        inputs=[node.output[0], const_name],
                        outputs=[new_output],
                    ),
                )
            for const_op_idx in range(len(constants_ops_to_add)):
                constants_ops_to_add.pop(0)
            if new_output != new_node.output[0]:
                for other_node in onnx_model.graph.node[node_idx+1:]:
                    for i in range(len(other_node.input)):
                        if other_node.input[i] == new_node.output[0]:
                            other_node.input[i] = new_output
        elif node.op_type == "Constant":
            onnx_update_constant_to(new_node, single_prec_constant_dtype)
            new_nodes.append(new_node)
        elif node.op_type == "Clip":
            new_nodes.append(new_node)
            add_cast_as(onnx_model, new_nodes, node_idx, node, dtype)
            in_fused_node = False
        elif node.op_type not in CONST_ACCELERATED_OPS and len(constants_ops_to_add)>0:
            new_output = new_node.output[0]
            new_input = new_node.input[0]
            for const_op_idx,(const_name, op_type) in enumerate(constants_ops_to_add):
                new_output = new_node.output[0] + "_const_" + str(const_op_idx)
                new_nodes.append(
                    onnx.helper.make_node(
                        name="ONNX_BIAS_SCALE_CONST",
                        op_type=op_type,
                        inputs=[new_input, const_name],
                        outputs=[new_output],
                    ),
                )
                new_input = new_output
            for const_op_idx in range(len(constants_ops_to_add)):
                constants_ops_to_add.pop(0)
            new_node.input[0] = new_input
            new_nodes.append(new_node)
        else:
            in_fused_node = False
            new_nodes.append(new_node)
    for node_idx in range(len(onnx_model.graph.node)):
        # remove the original node
        onnx_model.graph.node.pop(0)
    onnx_model.graph.node.extend(new_nodes)
    node_outputs_names = [node.output[0] for node in new_nodes]
    for node in new_nodes:
        node_outputs_names += node.input
    key_counts = Counter(node_outputs_names)
    outputs__ = dict(key_counts)
    weights_names = [init__.name for init__ in onnx_model.graph.initializer]
    input_names = [input__.name for input__ in onnx_model.graph.input]
    outputs__ = {k: v for k, v in outputs__.items() if v == 1 and k not in weights_names and k not in input_names}
    for output__, output_name in zip(onnx_model.graph.output, outputs__):
        output__.name = output_name
    return


def sanitize_model_to_int32(model_path, new_model_path):
    onnx_model = onnx.load(model_path)
    
    onnx_update_inputs_to(onnx_model, np.dtype("int32"))
    onnx_update_outputs_to(onnx_model, np.dtype("int32"))
    onnx_update_nodes(onnx_model, np.dtype("int32"))
    
    with open(new_model_path, "wb") as new_f:
        onnx.save(onnx_model,new_f)

def sanitize_model_to_uint8(model_path, new_model_path):
    onnx_model = onnx.load(model_path)
    
    onnx_update_inputs_to(onnx_model, np.dtype("uint8"))
    onnx_update_outputs_to(onnx_model, np.dtype("int32"))
    onnx_update_nodes(onnx_model, np.dtype("uint8"))
    
    with open(new_model_path, "wb") as new_f:
        onnx.save(onnx_model,new_f)

def sanitize_model_multiple_outputs(model_path, new_model_path):
    onnx_model = onnx.load(model_path)
    new_nodes = list()
    for node_idx,node in enumerate(onnx_model.graph.node):
        new_node = copy.deepcopy(node)
        onnx_update_node_attributes(new_node)
        new_nodes.append(new_node)
    for node_idx in range(len(onnx_model.graph.node)):
        # remove the original node
        onnx_model.graph.node.pop(0)
    onnx_model.graph.node.extend(new_nodes)
    node_outputs_names = [node.output[0] for node in new_nodes]
    for node in new_nodes:
        node_outputs_names += node.input
    key_counts = Counter(node_outputs_names)
    outputs__ = dict(key_counts)
    weights_names = [init__.name for init__ in onnx_model.graph.initializer]
    input_names = [input__.name for input__ in onnx_model.graph.input]
    outputs__ = {k: v for k, v in outputs__.items() if v == 1 and k not in weights_names and k not in input_names}
    for output__, output_name in zip(onnx_model.graph.output, outputs__):
        output__.name = output_name
    with open(new_model_path, "wb") as new_f:
        onnx.save(onnx_model,new_f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model to int32 or uint8")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--new_model_name", type=str, help="Name of the new model")
    parser.add_argument("--uint8", action="store_true", help="Convert to uint8")
    parser.add_argument("--fix_multiple_outputs", action="store_true", help="Fix only the multiple outputs issue")
    parser.add_argument("--remove_first_unsqueeze", action="store_true", help="Remove the first Unsqueeze node if present")
    parser.add_argument("--merge_pad_with_next_op", action="store_true", help="Merge Pad with next op if present")
    parser.add_argument("--remove_plinio_attrs", action="store_true", help="Remove Plinio-MATCH specific attributes")
    parser.add_argument("--lift_1d_to_2d_height", action="store_true", help="Lift 1D tensors to 2D (Height-major) format")
    parser.add_argument("--lift_1d_to_2d_width", action="store_true", help="Lift 1D tensors to 2D (Width-major) format")
    args = parser.parse_args()

    if args.remove_plinio_attrs:
        remove_plinio_attrs(os.path.dirname(__file__)+"/"+args.model_name+".onnx", os.path.dirname(__file__)+"/"+args.new_model_name+".onnx")
    elif args.merge_pad_with_next_op:
        merge_pad_with_next_op(os.path.dirname(__file__)+"/"+args.model_name+".onnx", os.path.dirname(__file__)+"/"+args.new_model_name+".onnx")
    elif args.remove_first_unsqueeze:
        remove_first_unsqueeze(os.path.dirname(__file__)+"/"+args.model_name+".onnx", os.path.dirname(__file__)+"/"+args.new_model_name+".onnx")
    elif args.fix_multiple_outputs:
        sanitize_model_multiple_outputs(os.path.dirname(__file__)+"/"+args.model_name+".onnx", os.path.dirname(__file__)+"/"+args.new_model_name+".onnx")
    elif args.uint8:
        sanitize_model_to_uint8(os.path.dirname(__file__)+"/"+args.model_name+".onnx", os.path.dirname(__file__)+"/"+args.new_model_name+".onnx")
    elif args.lift_1d_to_2d_height:
        lift_onnx_1d_to_2d(os.path.dirname(__file__)+"/"+args.model_name+".onnx", os.path.dirname(__file__)+"/"+args.new_model_name+".onnx", mode="height")
    elif args.lift_1d_to_2d_width:
        lift_onnx_1d_to_2d(os.path.dirname(__file__)+"/"+args.model_name+".onnx", os.path.dirname(__file__)+"/"+args.new_model_name+".onnx", mode="width")
    else:
        sanitize_model_to_int32(os.path.dirname(__file__)+"/"+args.model_name+".onnx", os.path.dirname(__file__)+"/"+args.new_model_name+".onnx")