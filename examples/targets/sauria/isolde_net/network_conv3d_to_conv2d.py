"""Convert Conv3D operators in an ONNX model to a Conv2D-based graph."""

import argparse
from pathlib import Path

from fix_graph_module import convert_conv3d_to_conv2d_with_pad


def main():
	parser = argparse.ArgumentParser(
		description="Convert 5D ONNX Conv operators to a Conv2D-based graph."
	)
	parser.add_argument(
		"input_model",
		nargs="?",
		default="GraphModule.onnx",
		help="Path to the input ONNX model (default: GraphModule.onnx).",
	)
	parser.add_argument(
		"output_model",
		nargs="?",
		default="isolde_2d.onnx",
		help="Path for the converted ONNX model (default: isolde_2d.onnx).",
	)
	args = parser.parse_args()

	input_model = Path(args.input_model)
	if not input_model.is_file():
		parser.error(f"input model does not exist: {input_model}")

	convert_conv3d_to_conv2d_with_pad(args.input_model, args.output_model)


if __name__ == "__main__":
	main()
