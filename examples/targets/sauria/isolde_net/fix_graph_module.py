import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np

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
    graph = model.graph

    transformed_nodes = []
    new_initializers = []
    
    node_id = 0

    for node in graph.node:
        if node.op_type != "Conv":
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
        # Step 2: Slice along Depth (D) and Apply 2D Convolutions
        # ------------------------------------------------------------------
        kd_outputs = []

        for k in range(kd):
            # Extract 2D weight slice at index k -> Shape: (out_c, in_c, kh, kw)
            w2d_slice = weights[:, :, k, :, :]
            w2d_name = f"{weight_name}_slice_{k}_{node_id}"
            w2d_init = numpy_helper.from_array(w2d_slice, name=w2d_name)
            new_initializers.append(w2d_init)

            # Slice the 5D input across depth for kernel position k
            sliced_input_name = f"{padded_input_name}_slice_k{k}_{node_id}"

            # Slice in this model uses opset-10 form: starts/ends/axes are tensor inputs.
            start_init = helper.make_tensor(
                f"start_{k}_{node_id}", TensorProto.INT64, [1], [k * dil_d]
            )
            end_init = helper.make_tensor(
                f"end_{k}_{node_id}", TensorProto.INT64, [1], [(k * dil_d) + 1]
            )
            axes_init = helper.make_tensor(
                f"axes_{k}_{node_id}", TensorProto.INT64, [1], [2]
            )
            new_initializers.extend([start_init, end_init, axes_init])

            slice_node = helper.make_node(
                "Slice",
                inputs=[padded_input_name, start_init.name, end_init.name, axes_init.name],
                outputs=[sliced_input_name],
                name=f"Slice_D_{k}_{node_id}"
            )
            replacement_nodes.append(slice_node)

            # Reshape 5D (N, C, 1, H, W) -> 4D (N, C, H, W) for Conv2D execution
            squeezed_input_name = f"{sliced_input_name}_4d"
            
            squeeze_node = helper.make_node(
                "Squeeze",
                inputs=[sliced_input_name],
                outputs=[squeezed_input_name],
                axes=[2],
                name=f"Squeeze_D_{k}_{node_id}"
            )
            replacement_nodes.append(squeeze_node)

            # Apply Conv2D with spatial padding set to 0 (handled by explicit Pad node above)
            conv2d_out_name = f"conv2d_out_k{k}_{node_id}"
            conv2d_node = helper.make_node(
                "Conv",
                inputs=[squeezed_input_name, w2d_name],  # No bias added here (added at end)
                outputs=[conv2d_out_name],
                strides=[stride_h, stride_w],
                pads=[0, 0, 0, 0],  # 2D Padding zeroed out
                dilations=[dil_h, dil_w],
                name=f"Conv2D_k{k}_{node_id}"
            )
            replacement_nodes.append(conv2d_node)
            kd_outputs.append(conv2d_out_name)

        # ------------------------------------------------------------------
        # Step 3: Sum sub-convolution results and add Bias
        # ------------------------------------------------------------------
        summed_conv_name = f"conv_summed_{node_id}"
        if len(kd_outputs) == 1:
            summed_conv_name = kd_outputs[0]
        else:
            sum_node = helper.make_node(
                "Sum",
                inputs=kd_outputs,
                outputs=[summed_conv_name],
                name=f"Sum2D_Slices_{node_id}"
            )
            replacement_nodes.append(sum_node)

        # Re-unsqueeze back to 5D (N, C, 1, H, W)
        unsqueezed_out_name = f"{summed_conv_name}_5d"
        unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=[summed_conv_name],
            outputs=[unsqueezed_out_name if bias is not None else output_name],
            axes=[2],
            name=f"Unsqueeze_D_{node_id}"
        )
        replacement_nodes.append(unsqueeze_node)

        # Add optional bias back to tensor
        if bias is not None:
            bias_2d_name = f"{bias_name}_reshaped_{node_id}"
            # Reshape bias to (1, C, 1, 1, 1) for broadcasting over 5D output
            bias_reshaped = bias.reshape(1, -1, 1, 1, 1)
            bias_init = numpy_helper.from_array(bias_reshaped, name=bias_2d_name)
            new_initializers.append(bias_init)

            bias_add_node = helper.make_node(
                "Add",
                inputs=[unsqueezed_out_name, bias_2d_name],
                outputs=[output_name],
                name=f"AddBias_{node_id}"
            )
            replacement_nodes.append(bias_add_node)

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

# Run conversion
convert_conv3d_to_conv2d_with_pad("GraphModule.onnx", "model_2d_padded.onnx")