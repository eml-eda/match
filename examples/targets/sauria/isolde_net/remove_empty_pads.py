"""Remove the Pad nodes of an ONNX model that don't pad anything (all pads == 0)."""

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper


def get_pads(node, constants):
	"""Returns the pads of a Pad node as a list, None if they aren't static."""
	# opset < 11: pads is an attribute
	for attr in node.attribute:
		if attr.name == "pads":
			return list(attr.ints)
	# opset >= 11: pads is the second input (initializer or Constant node)
	if len(node.input) < 2 or node.input[1] == "":
		return None
	return constants.get(node.input[1])


def collect_constants(graph):
	"""Maps every initializer / Constant node output to its values as a list."""
	constants = {init.name: numpy_helper.to_array(init).flatten().tolist() for init in graph.initializer}
	for node in graph.node:
		if node.op_type == "Constant":
			for attr in node.attribute:
				if attr.name == "value":
					constants[node.output[0]] = numpy_helper.to_array(attr.t).flatten().tolist()
	return constants


def remove_empty_pads(input_model_path, output_model_path):
	model = onnx.load(input_model_path)
	graph = model.graph
	constants = collect_constants(graph)
	graph_outputs = {out.name for out in graph.output}

	removed = []
	pad_params = set()  # pads / constant_value / axes inputs of the removed nodes
	for node in list(graph.node):
		if node.op_type != "Pad":
			continue
		pads = get_pads(node, constants)
		if pads is None:
			print(f"[remove_empty_pads] Skipping {node.name}: pads are not static")
			continue
		if any(p != 0 for p in pads):
			continue

		pad_in, pad_out = node.input[0], node.output[0]
		node_idx = list(graph.node).index(node)
		graph.node.remove(node)
		removed.append(node.name)
		pad_params |= {name for name in node.input[1:] if name}

		has_producer = any(pad_in in other.output for other in graph.node)
		if pad_out in graph_outputs and not has_producer:
			# Graph input -> graph output: nothing to rename, keep an Identity
			graph.node.insert(node_idx, onnx.helper.make_node(
				"Identity", inputs=[pad_in], outputs=[pad_out], name=f"{node.name}_identity"
			))
			continue
		if pad_out in graph_outputs:
			# Keep the graph output name: let the producer of pad_in write pad_out directly
			for other in graph.node:
				for i, name in enumerate(other.output):
					if name == pad_in:
						other.output[i] = pad_out
				for i, name in enumerate(other.input):
					if name == pad_in:
						other.input[i] = pad_out
			old = pad_in
		else:
			for other in graph.node:
				for i, name in enumerate(other.input):
					if name == pad_out:
						other.input[i] = pad_in
			old = pad_out

		# Drop the value_info of the tensor that disappeared
		for vi in list(graph.value_info):
			if vi.name == old:
				graph.value_info.remove(vi)

	# Drop the pads / constant_value / axes tensors no longer used by any node
	used = {name for node in graph.node for name in node.input}
	used |= graph_outputs
	unused = pad_params - used
	for init in list(graph.initializer):
		if init.name in unused:
			graph.initializer.remove(init)
	for node in list(graph.node):
		if node.op_type == "Constant" and node.output[0] in unused:
			graph.node.remove(node)

	print(f"[remove_empty_pads] Removed {len(removed)} empty Pad nodes: {removed}")
	onnx.checker.check_model(model)
	onnx.save(model, output_model_path)


def main():
	parser = argparse.ArgumentParser(
		description="Remove the Pad nodes of an ONNX model that don't pad anything."
	)
	parser.add_argument("input_model", help="Path to the input ONNX model.")
	parser.add_argument("output_model", help="Path for the output ONNX model.")
	args = parser.parse_args()

	if not Path(args.input_model).is_file():
		parser.error(f"input model does not exist: {args.input_model}")

	remove_empty_pads(args.input_model, args.output_model)


if __name__ == "__main__":
	main()
