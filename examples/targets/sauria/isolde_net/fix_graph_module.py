import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx import shape_inference


def get_static_dimension(graph, tensor_name, axis):
    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        if value_info.name == tensor_name:
            dimensions = value_info.type.tensor_type.shape.dim
            if axis < len(dimensions) and dimensions[axis].dim_value:
                return dimensions[axis].dim_value
    return None

def extract_conv3d_weights(model_graph, weight_name, bias_name=None):
    """Retrieves weight and bias numpy arrays from graph initializers."""
    weights = None
    bias = None
    initializers_to_remove = set()

    for init in model_graph.initializer:
        if init.name == weight_name:
            weights = numpy_helper.to_array(init)
            initializers_to_remove.add(init.name)
        elif bias_name and init.name == bias_name:
            bias = numpy_helper.to_array(init)
            initializers_to_remove.add(init.name)

    # Clean up old initializers
    if initializers_to_remove:
        kept_initializers = [
            i for i in model_graph.initializer if i.name not in initializers_to_remove
        ]
        del model_graph.initializer[:]
        model_graph.initializer.extend(kept_initializers)
    
    return weights, bias, initializers_to_remove

def convert_conv3d_to_conv2d_with_pad(onnx_model_path, output_model_path):
    model = onnx.load(onnx_model_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    transformed_nodes = []
    new_initializers = []
    relocated_cast_outputs = set()
    
    node_id = 0

    for node in graph.node:
        if node.op_type != "Conv":
            if node.op_type == "Cast" and node.output[0] in relocated_cast_outputs:
                continue
            transformed_nodes.append(node)
            continue

        # Extract attributes
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        strides = attrs.get("strides", [1, 1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0, 0, 0])  # [D_start, H_start, W_start, D_end, H_end, W_end]
        dilations = attrs.get("dilations", [1, 1, 1])

        # Identify Conv3D nodes (5D input/kernel shape profile)
        input_name = node.input[0]
        weight_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else None
        output_name = node.output[0]

        original_cast = next(
            (
                candidate
                for candidate in graph.node
                if candidate.op_type == "Cast"
                and candidate.input
                and candidate.input[0] == output_name
            ),
            None,
        )
        cast_name = "FAKE_CAST_TVM_OUT_DTYPE"
        cast_type = TensorProto.INT32
        final_output_name = output_name
        if original_cast is not None:
            relocated_cast_outputs.add(original_cast.output[0])
            final_output_name = original_cast.output[0]
            for attribute in original_cast.attribute:
                if attribute.name == "to":
                    cast_type = attribute.i
                    break

        # Retrieve weight array from initializers
        weights, bias, _ = extract_conv3d_weights(graph, weight_name, bias_name)
        if weights is None or weights.ndim != 5:
            # Skip if weights are not 5D (i.e. already Conv2D or dynamically provided)
            transformed_nodes.append(node)
            continue

        replacement_nodes = []
        
        # Conv3D weights format: (Out_channels, In_channels, Kd, Kh, Kw)
        out_c, in_c, kd, kh, kw = weights.shape
        stride_d, stride_h, stride_w = strides
        pad_d_start, pad_h_start, pad_w_start, pad_d_end, pad_h_end, pad_w_end = pads
        dil_d, dil_h, dil_w = dilations

        # ------------------------------------------------------------------
        # Step 1: Explicit 2D Padding via Pad Node
        # Separating H/W spatial padding out of Conv and into an explicit Pad node
        # ------------------------------------------------------------------
        padded_input_name = f"{input_name}_explicit_padded_{node_id}"
        
        # ONNX Pad format for 5D tensor (N, C, D, H, W):
        # [N_start, C_start, D_start, H_start, W_start, N_end, C_end, D_end, H_end, W_end]
        pad_paddings = [0, 0, pad_d_start, pad_h_start, pad_w_start, 
                        0, 0, pad_d_end, pad_h_end, pad_w_end]
        
        pad_node = helper.make_node(
            "Pad",
            inputs=[input_name],
            outputs=[padded_input_name],
            pads=pad_paddings,
            value=0.0,
            mode="constant",
            name=f"ExplicitPad_{node_id}"
        )
        replacement_nodes.append(pad_node)

        # ------------------------------------------------------------------
        # Step 2: Reconstruct every output-depth position with Conv2D slices.
        # ------------------------------------------------------------------
        input_depth = get_static_dimension(graph, input_name, 2)
        if input_depth is None:
            raise ValueError(f"Conv3D input depth is dynamic: {input_name}")
        padded_depth = input_depth + pad_d_start + pad_d_end
        output_depth = (
            (padded_depth - dil_d * (kd - 1) - 1) // stride_d
        ) + 1
        depth_outputs = []

        for output_depth_index in range(output_depth):
            kd_outputs = []
            for k in range(kd):
                w2d_slice = weights[:, :, k, :, :]
                w2d_name = f"{weight_name}_slice_{k}_{node_id}"
                if not any(init.name == w2d_name for init in new_initializers):
                    new_initializers.append(numpy_helper.from_array(w2d_slice, name=w2d_name))

                sliced_input_name = (
                    f"{padded_input_name}_out{output_depth_index}_k{k}"
                )
                start = output_depth_index * stride_d + k * dil_d
                start_init = helper.make_tensor(
                    f"start_{node_id}_{output_depth_index}_{k}",
                    TensorProto.INT64,
                    [1],
                    [start],
                )
                end_init = helper.make_tensor(
                    f"end_{node_id}_{output_depth_index}_{k}",
                    TensorProto.INT64,
                    [1],
                    [start + 1],
                )
                axes_init = helper.make_tensor(
                    f"axes_{node_id}_{output_depth_index}_{k}",
                    TensorProto.INT64,
                    [1],
                    [2],
                )
                new_initializers.extend([start_init, end_init, axes_init])
                replacement_nodes.append(helper.make_node(
                    "Slice",
                    inputs=[padded_input_name, start_init.name, end_init.name, axes_init.name],
                    outputs=[sliced_input_name],
                    name=f"Slice_D_{node_id}_{output_depth_index}_{k}",
                ))

                squeezed_input_name = f"{sliced_input_name}_4d"
                replacement_nodes.append(helper.make_node(
                    "Squeeze",
                    inputs=[sliced_input_name],
                    outputs=[squeezed_input_name],
                    axes=[2],
                    name=f"Squeeze_D_{node_id}_{output_depth_index}_{k}",
                ))

                conv2d_out_name = f"conv2d_out_{node_id}_{output_depth_index}_{k}"
                replacement_nodes.append(helper.make_node(
                    "Conv",
                    inputs=[squeezed_input_name, w2d_name],
                    outputs=[conv2d_out_name],
                    strides=[stride_h, stride_w],
                    pads=[0, 0, 0, 0],
                    dilations=[dil_h, dil_w],
                    name=f"Conv2D_{node_id}_{output_depth_index}_{k}",
                ))

                casted_conv2d_out_name = f"{conv2d_out_name}_{cast_name}"
                replacement_nodes.append(helper.make_node(
                    "Cast",
                    inputs=[conv2d_out_name],
                    outputs=[casted_conv2d_out_name],
                    to=cast_type,
                    name=f"{cast_name}_{node_id}_{output_depth_index}_{k}",
                ))
                kd_outputs.append(casted_conv2d_out_name)

            summed_conv_name = f"conv_summed_{node_id}_{output_depth_index}"
            if len(kd_outputs) == 1:
                summed_conv_name = kd_outputs[0]
            else:
                replacement_nodes.append(helper.make_node(
                    "Sum",
                    inputs=kd_outputs,
                    outputs=[summed_conv_name],
                    name=f"Sum2D_Slices_{node_id}_{output_depth_index}",
                ))

            if bias is not None:
                bias_2d_name = f"{bias_name}_reshaped_{node_id}"
                if not any(init.name == bias_2d_name for init in new_initializers):
                    bias_init = numpy_helper.from_array(
                        bias.reshape(1, -1, 1, 1), name=bias_2d_name
                    )
                    new_initializers.append(bias_init)
                biased_output_name = f"{summed_conv_name}_biased"
                replacement_nodes.append(helper.make_node(
                    "Add",
                    inputs=[summed_conv_name, bias_2d_name],
                    outputs=[biased_output_name],
                    name=f"AddBias_{node_id}_{output_depth_index}",
                ))
            else:
                biased_output_name = summed_conv_name

            plane_name = f"{biased_output_name}_5d"
            replacement_nodes.append(helper.make_node(
                "Unsqueeze",
                inputs=[biased_output_name],
                outputs=[plane_name],
                axes=[2],
                name=f"Unsqueeze_D_{node_id}_{output_depth_index}",
            ))
            depth_outputs.append(plane_name)

        if len(depth_outputs) == 1:
            replacement_nodes.append(helper.make_node(
                "Identity",
                inputs=[depth_outputs[0]],
                outputs=[final_output_name],
                name=f"Conv3DToConv2DOutput_{node_id}",
            ))
        else:
            replacement_nodes.append(helper.make_node(
                "Concat",
                inputs=depth_outputs,
                outputs=[final_output_name],
                axis=2,
                name=f"ConcatDepth_{node_id}",
            ))

        transformed_nodes.extend(replacement_nodes)
        node_id += 1

    # Update Graph Nodes & Initializers
    del graph.node[:]
    graph.node.extend(transformed_nodes)

    graph.initializer.extend(new_initializers)

    # Save and validate updated ONNX model
    onnx.checker.check_model(model)
    onnx.save(model, output_model_path)
    print(f"Successfully transformed model saved to: {output_model_path}")

if __name__ == "__main__":
    convert_conv3d_to_conv2d_with_pad("GraphModule.onnx", "model_2d_padded.onnx")