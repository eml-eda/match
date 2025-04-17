import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import cycle

def save_memory_allocation_graph(
    mem_tensors,
    available_soc_bytes: int = 0,
    output_file="memory_allocation.png"
):
    """
    Generates and saves a graph of memory allocation over time, including tensor sizes.

    Args:
        mem_tensors (List[MatchMemoryTensor]): List of memory tensors with allocation details.
        output_file (str): Path to save the generated graph.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Expanded color palette
    color_palette = cycle([
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
        "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6"
    ])
    tensor_colors = {tensor.name: next(color_palette) for tensor in mem_tensors}

    def abbreviate_name(name):
        """Abbreviates tensor names for simplicity."""
        return "".join([part[0].upper() for part in name.split("_")])

    # Create a mapping of abbreviated names to full names
    name_mapping = {abbreviate_name(tensor.name): tensor.name for tensor in mem_tensors}

    # Create legend handles for the dictionary
    legend_handles = []

    for tensor in mem_tensors:
        color = tensor_colors[tensor.name]
        if abbreviate_name(tensor.name) not in [handle.get_label() for handle in legend_handles]:
            legend_handles.append(
                mpatches.Patch(color=color, label=f"{abbreviate_name(tensor.name)}: {tensor.name}")
            )
        for time, offset in tensor.mem_offset_at.items():
            height = tensor.num_bytes  # Size of the tensor
            rect = mpatches.Rectangle(
                (time, offset), 1, height, color=color, alpha=0.7, edgecolor="black", linewidth=1.5
            )
            ax.add_patch(rect)
            # Abbreviate the tensor name for display
            abbreviated_name = abbreviate_name(tensor.name)
            ax.text(
                time + 0.5, offset + height / 2, abbreviated_name,
                fontsize=8, fontweight="bold", ha='center', va='center', color="white"
            )

            # Add green arrows for loading from external memory
            if time in tensor.load_from_ext_mem_at:
                ax.annotate(
                    '', xy=(time + 0.5, offset + height), xytext=(time + 0.5, offset + height + 0.1),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8)
                )

            # Add red arrows for moving to external memory
            if time in tensor.move_temp_to_ext_mem:
                ax.annotate(
                    '', xy=(time + 0.5, offset + height + 0.1), xytext=(time + 0.5, offset + height),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8)
                )

    ax.set_title("Memory Allocation")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Memory")
    ax.set_xlim(0, max([tensor.last_usage for tensor in mem_tensors]+[0]) + 1)
    ax.set_ylim(0, max([max(list(tensor.mem_offset_at.values())+[0]) + tensor.num_bytes \
                       for tensor in mem_tensors]+[0]) + 32)
    ax.grid(True)

    # Add the legend box for the dictionary
    ax.legend(
        handles=legend_handles,
        loc="upper right",  # Position the legend in the upper right corner
        title="Tensors",
        fontsize=8, title_fontsize=10, frameon=True
    )

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")  # Ensure the legend is included in the saved image
    plt.close(fig)  # Close the figure to prevent it from popping up in Jupyter
    print(f"[MEM PLANNER] Memory allocation graph saved")