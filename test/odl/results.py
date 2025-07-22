import argparse
import json
import os
from pathlib import Path
import pandas as pd
import xlsxwriter

BUILDS_BASE_DIR = os.path.dirname(__file__) + "/../builds"

def get_results(name_exp, target, base_dir, directions, models, results_file):
    workbook = xlsxwriter.Workbook(results_file)
    results_main_sheet = workbook.add_worksheet("results")
    main_sheet_headers = [
        "name_exp", "model", "direction",
        "target", "node_cycles",
        "mem_transfer_cycles", "peak_dynamic_on_chip",
        "binary_l2_size",
        # on chip off chip summary metadata
        "total_on_chip", "static_on_chip",
        "constants_on_chip", "io_on_chip",
        "dynamic_on_chip", "tvm_buffers_extra_dynamic",
        "total_off_chip", "dynamic_off_chip",
        "io_off_chip", "constants_off_chip"
    ]
    nodes_headers = ["name_node", "cycles"]
    mem_transfer_headers = ["name_transfer", "transfer_type", "bytes", "cycles"]
    compilation_headers = ["memory_name", "used", "total", "used_percentage"]
    for idx, header in enumerate(main_sheet_headers):
        results_main_sheet.write(0, idx, header)
    row_number = 1
    for model in models:
        for direction in directions:
            model_dir = base_dir / model / direction / target
            print(f"Processing model: {model}, direction: {direction}, target: {target} for experiment: {name_exp} at {model_dir}")
            if not model_dir.exists():
                print(f"Directory {model_dir} does not exist. Skipping.")
                continue
            compilation_log = model_dir / "compilation.log"
            run_log = model_dir / "run.log"
            if not compilation_log.exists() or not run_log.exists():
                print(f"Log files for {model_dir} are missing. Skipping.")
                continue
            compilation_metadata = {}
            with open(compilation_log, 'r') as f:
                reading_memory = False
                for line in f:
                    if reading_memory:
                        splitted = line.strip().split()
                        if len(splitted) != 6:
                            break
                        code_size_mem_name = splitted[0][:-1]
                        compilation_metadata[code_size_mem_name] = {}
                        compilation_metadata[code_size_mem_name]["used"] = int(splitted[1]) * (1 if splitted[2] == "B" else 1024 if splitted[2] == "KB" else 1024 * 1024 if splitted[2] == "MB" else 1)
                        compilation_metadata[code_size_mem_name]["total"] = int(splitted[3]) * (1 if splitted[4] == "B" else 1024 if splitted[4] == "KB" else 1024 * 1024 if splitted[4] == "MB" else 1)
                        compilation_metadata[code_size_mem_name]["used_percentage"] = float(splitted[5].strip('%'))
                    if "Memory region" in line:
                        reading_memory = True
            run_metadata = {}
            run_metadata["nodes"] = {}
            run_metadata["mem_transfers"] = {}
            run_metadata["max_on_chip_memory"] = 0
            with open(run_log, 'r') as f:
                reading_nodes = False
                reading_mem_transfer = False
                for line in f:
                    if reading_nodes:
                        splitted = line.strip().split()
                        if len(splitted) != 2:
                            reading_nodes = False
                            continue
                        name_node = splitted[0][1:-1]
                        run_metadata["nodes"][name_node] = {}
                        run_metadata["nodes"][name_node]["cycles"] = int(splitted[1])
                    if reading_mem_transfer:
                        splitted = line.strip().split()
                        if len(splitted) not in [6, 7]:
                            reading_mem_transfer = False
                            continue
                        name_transfer = "_".join(splitted[:3])[1:-1]
                        run_metadata["mem_transfers"][name_transfer] = {}
                        run_metadata["mem_transfers"][name_transfer]["bytes"] = int(splitted[-3]) if len(splitted)==7 else int(splitted[-2].replace("Cycles:", ""))
                        run_metadata["mem_transfers"][name_transfer]["transfer_type"] = splitted[2][:-1]
                        run_metadata["mem_transfers"][name_transfer]["cycles"] = int(splitted[-1])
                    if "Node	Cycle" in line:
                        reading_nodes = True
                    if "Profiling Mem Transfers Performance" in line:
                        reading_mem_transfer = True
                    if "Peak dynamic memory allocated" in line:
                        run_metadata["peak_dynamic_on_chip"] = int(line.strip().split()[-2])
                        break
            run_metadata["node_cycles"] = sum(node["cycles"] for node in run_metadata["nodes"].values())
            run_metadata["mem_transfer_cycles"] = sum(transfer["cycles"] for transfer in run_metadata["mem_transfers"].values())
            # Store results in a dictionary for each model/direction/target
            result_entry = {
                "name_exp": name_exp,
                "model": model,
                "direction": direction,
                "target": target,
                "compilation_metadata": compilation_metadata,
                "run_metadata": run_metadata
            }
            # Save to Excel and JSON with a sheet per model-direction-target
            json_file = f"{name_exp}_results_{model}_{direction}_{target}.json"
            json.dump(result_entry, open(json_file, "w"), indent=4)
            on_chip_off_chip_metadata_path = model_dir / "models/test_bp_tvm/metadata/memory_plan_on_off_chip_summary.json"
            on_chip_off_chip_metadata = dict()
            with open(on_chip_off_chip_metadata,"r") as mod_file:
                on_chip_off_chip_metadata = json.load(mod_file)
            # Flatten the result_entry for DataFrame
            flat_entry = {
                "name_exp": name_exp,
                "model": model,
                "direction": direction,
                "target": target,
                "constant_off_chip_loading": run_metadata.get("constant_off_chip_loading", 0),
                "io_off_chip_loading": run_metadata.get("io_off_chip_loading", 0),
                "node_cycles": run_metadata.get("node_cycles", 0),
                "mem_transfer_cycles": run_metadata.get("mem_transfer_cycles", 0),
                "peak_dynamic_on_chip": run_metadata.get("peak_dynamic_on_chip", 0),
                "binary_l2_size": compilation_metadata.get("L2_shared", {}).get("used", 0)
            }
            flat_entry = {**flat_entry, **on_chip_off_chip_metadata}
            row_val=[flat_entry[header] for header in main_sheet_headers]
            for idx,row_it in enumerate(row_val):    
                results_main_sheet.write(row_number,idx,row_it)
            row_number+=1
            comp_sheet_name = f"{model}_{direction}_{target}_compilation"
            comp_sheet = workbook.add_worksheet(comp_sheet_name)
            for idx, header in enumerate(compilation_headers):
                comp_sheet.write(0, idx, header)
            for idx, (memory_name, memory_data) in enumerate(compilation_metadata.items()):
                comp_sheet.write(idx + 1, 0, memory_name)
                comp_sheet.write(idx + 1, 1, memory_data["used"])
                comp_sheet.write(idx + 1, 2, memory_data["total"])
                comp_sheet.write(idx + 1, 3, memory_data["used_percentage"])
            nodes_sheet_name = f"{model}_{direction}_{target}_nodes"
            nodes_sheet = workbook.add_worksheet(nodes_sheet_name)
            for idx, header in enumerate(nodes_headers):
                nodes_sheet.write(0, idx, header)
            for idx, (node_name, node_data) in enumerate(run_metadata["nodes"].items()):
                nodes_sheet.write(idx + 1, 0, node_name)
                nodes_sheet.write(idx + 1, 1, node_data["cycles"])
            mem_transfer_sheet_name = f"{model}_{direction}_{target}_mem_transfers"
            mem_transfer_sheet = workbook.add_worksheet(mem_transfer_sheet_name)
            for idx, header in enumerate(mem_transfer_headers):
                mem_transfer_sheet.write(0, idx, header)
            for idx, (transfer_name, transfer_data) in enumerate(run_metadata["mem_transfers"].items()):
                mem_transfer_sheet.write(idx + 1, 0, transfer_name)
                mem_transfer_sheet.write(idx + 1, 1, transfer_data["transfer_type"])
                mem_transfer_sheet.write(idx + 1, 2, transfer_data["bytes"])
                mem_transfer_sheet.write(idx + 1, 3, transfer_data["cycles"])
    workbook.close()
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="gap",
        help="Target HW for the ODL experiments.",
    )
    parser.add_argument(
        "--name_exp",
        type=str,
        default="odl_results",
        help="Base directory for ODL results.",
    )
    parser.add_argument(
        "--directions",
        dest="directions",
        type=str,
        nargs='+',
        default=["fw", "bw"],
        help="List of directions for the ODL experiments(can be fw, bw or fw and bw).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=["kws", "imcls", "vww"],
        help="List of models to use for the ODL experiments.",
    )
    args = parser.parse_args()
    
    if len(args.directions) not in [1, 2]:
        raise ValueError("At least one direction must be specified.")
    if len(args.directions) != len(set(args.directions)):
        raise ValueError("Directions must be unique.")
    if any(dir not in ["fw", "bw"] for dir in args.directions):
        raise ValueError("Directions must be either 'fw', 'bw' or both.")
    if len(args.models) not in [1, 2, 3]:
        raise ValueError("At least one model must be specified.")
    if any(model not in ["kws", "imcls", "vww"] for model in args.models):
        raise ValueError("Models must be one of 'kws', 'imcls', or 'vww'.")
    if len(args.models) != len(set(args.models)):
        raise ValueError("Models must be unique.")
    base_dir = Path(BUILDS_BASE_DIR) / args.name_exp
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} does not exist.")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"{base_dir} is not a directory.")
    results_file = f"{args.name_exp}_results.xlsx"
    get_results(
        name_exp=args.name_exp,
        target=args.target,
        base_dir=base_dir,
        directions=args.directions,
        models=args.models,
        results_file=results_file
    )