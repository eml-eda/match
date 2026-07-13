import json
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import cycle

def abbreviate_name(name):
    """Abbreviates tensor names for simplicity."""
    return "".join(["_" if len(part) == 0 else part[0].upper() for part in name.split("_")])


def save_memory_allocation_graph(
    mem_tensors,
    graph_output_file="memory_allocation.png",
    metadata_output_file="memory_allocation_metadata.json",
):
    """
    Generates and saves a graph of memory allocation over time, including tensor sizes.
    Optimized for performance by merging contiguous timesteps into single patches.

    Args:
        mem_tensors (List[MatchMemoryTensor]): List of memory tensors with allocation details.
        graph_output_file (str): Path to save the generated graph.
        metadata_output_file (str): Path to save the metadata JSON.
    """
    # 1. Output Metadata quickly
    metadata = {}
    for tensor in mem_tensors:
        metadata[tensor.name] = {
            "num_bytes": tensor.num_bytes,
            "last_usage": tensor.last_usage,
            "mem_offset": tensor.mem_offset,
            "mem_offset_at": tensor.mem_offset_at,
            "load_from_ext_mem_at": tensor.load_from_ext_mem_at,
            "move_temp_to_ext_mem": tensor.move_temp_to_ext_mem,
            "used_at": tensor.used_at,
            "abbreviated_name": abbreviate_name(tensor.name),
            "is_intermediate": tensor.is_intermediate,
            "is_constant": tensor.is_constant,
            "stored_in_external_memory": getattr(tensor, "stored_in_external_memory", False),
            "name": tensor.name,
            "tvm_memplan_storage_id": getattr(tensor, "tvm_memplan_storage_id", None),
            "ext_mem_offset": getattr(tensor, "ext_mem_offset", None),
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "is_input": tensor.is_input,
            "is_output": tensor.is_output,
            "node_id": tensor.node_id,
        }

    with open(metadata_output_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[MEM PLANNER] Memory allocation metadata saved to {metadata_output_file}")

    if not mem_tensors:
        return

    # 2. Render optimized figure
    # Start with a small placeholder; we will resize dynamically once we know
    # max_time/max_offset and number of tensors.
    fig, ax = plt.subplots(figsize=(12, 8))

    color_palette = cycle([
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
        "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6"
    ])

    legend_handles_dict = {}
    max_time = 0
    max_offset = 0

    for tensor in mem_tensors:
        color = next(color_palette)
        abbrev_name = abbreviate_name(tensor.name)

        if abbrev_name not in legend_handles_dict:
            legend_handles_dict[abbrev_name] = mpatches.Patch(color=color, label=f"{abbrev_name}: {tensor.name}")

        if not tensor.mem_offset_at:
            continue

        sorted_times = sorted(tensor.mem_offset_at.keys())
        if sorted_times[-1] > max_time:
            max_time = sorted_times[-1]

        height = tensor.num_bytes

        # Fast contiguous block detection (avoids drawing thousands of 1-width patches)
        blocks = []
        start_t = sorted_times[0]
        prev_t = start_t
        curr_offset = tensor.mem_offset_at[start_t]

        for t in sorted_times[1:]:
            if t == prev_t + 1 and tensor.mem_offset_at[t] == curr_offset:
                prev_t = t
            else:
                blocks.append((start_t, prev_t, curr_offset))
                start_t = t
                prev_t = t
                curr_offset = tensor.mem_offset_at[t]
        blocks.append((start_t, prev_t, curr_offset))

        # Render single blocks
        for (st, et, off) in blocks:
            width = et - st + 1
            if off + height > max_offset:
                max_offset = off + height

            rect = mpatches.Rectangle(
                (st, off), width, height, color=color, alpha=0.7, edgecolor="black", linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(
                st + width / 2.0, off + height / 2.0, abbrev_name,
                fontsize=8, fontweight="bold", ha='center', va='center', color="white"
            )

        # Plot load/move arrows based directly on times found avoiding per-rect conditions
        load_times = set(tensor.load_from_ext_mem_at).intersection(sorted_times)
        for t in load_times:
            off = tensor.mem_offset_at[t]
            ax.annotate(
                '', xy=(t + 0.5, off + height), xytext=(t + 0.5, off + height + 0.1),
                arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8)
            )

        move_times = set(tensor.move_temp_to_ext_mem).intersection(sorted_times)
        for t in move_times:
            off = tensor.mem_offset_at[t]
            ax.annotate(
                '', xy=(t + 0.5, off + height + 0.1), xytext=(t + 0.5, off + height),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8)
            )

    ax.set_title("Memory Allocation")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Memory (Bytes)")
    ax.set_xlim(0, max_time + 2)
    ax.set_ylim(0, max_offset + 32)
    ax.grid(True)

    # Dynamic figure sizing based on graph complexity
    num_nodes = max_time + 1 if max_time > 0 else 1
    num_tensors = len(mem_tensors)
    fig_width = min(48, max(12, 8 + num_nodes * 0.45))
    fig_height = min(28, max(8, 5 + num_tensors * 0.22))
    fig.set_size_inches(fig_width, fig_height)

    legend_handles = list(legend_handles_dict.values())
    legend_ncols = max(1, min(len(legend_handles), int(fig_width // 2.5)))

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22, 1.0, 0.2),
        mode="expand",
        ncol=legend_ncols,
        title="Tensors",
        fontsize=8, title_fontsize=10, frameon=True
    )

    # Reserve bottom area for the expanded legend strip.
    plt.tight_layout(rect=[0, 0.14, 1, 1])
    plt.savefig(graph_output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[MEM PLANNER] Memory allocation graph saved to {graph_output_file}")


def save_memory_allocation_graph_nodes_buffers(
    mem_tensors_at: Dict = None,
    calls_idxs: List = None,
    match_mem_size: int = 0,
    save_every_empty_areas: bool = False,
    output_file="memory_allocation.json"
):
    """
    Generates and saves a graph of memory allocation over time, including tensor sizes.
    """
    if calls_idxs is None: calls_idxs = []
    if mem_tensors_at is None: mem_tensors_at = {}

    metadata = {}
    for call_idx in calls_idxs:
        tensors = mem_tensors_at.get(call_idx, [])
        allocs = []
        empty_areas = []

        for tensor in tensors:
            tensor_pt = tensor.mem_offset_at[call_idx]
            if getattr(tensor, 'is_extra_dynamic', False):
                empty_areas.append((tensor_pt, tensor.num_bytes, tensor.name))
            allocs.append((tensor_pt, tensor_pt + tensor.num_bytes, tensor.num_bytes))
            
        allocs.sort(key=lambda x: x[0])

        if save_every_empty_areas:
            last_alloc = 0
            for alloc_idx, alloc in enumerate(allocs):
                if alloc[0] > last_alloc:
                    empty_areas.append((last_alloc, alloc[0] - last_alloc, f"EMPTY_AREA_NODE_{call_idx}_{alloc_idx}"))
                last_alloc = alloc[1]
            if last_alloc < match_mem_size:
                empty_areas.append((last_alloc, match_mem_size - last_alloc, f"EMPTY_AREA_NODE_{call_idx}_LAST"))

        metadata[call_idx] = {
            "allocs": allocs,
            "empty_areas": empty_areas
        }
        
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[MEM PLANNER] Memory allocation nodes buffers metadata saved to {output_file}")
    return metadata
            