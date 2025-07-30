import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

from copy import deepcopy

# Define hatch patterns for transfers
hatch_patterns = {
    'load': '///',    # Forward diagonal lines
    'store': '\\\\\\'  # Backward diagonal lines
}


def plot_optimization_result(data, filename='optimization_result.png'):
    """
    Plot optimization results with 3 subplots:
    1. Gantt chart of node execution and communication
    2. L2 tensor 2D bin packing
    3. L3 tensor 2D bin packing
    """
    
    # Set up the figure with a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                        gridspec_kw={'height_ratios': [1, 2, 2]})
    
    # Color palettes
    device_colors = ['#2E86AB', '#A23B72', '#C73E1D', '#0A8754']
    comm_colors = {'load': '#4CAF50', 'store': '#FF9800'}
    tensor_type_colors = {
        'const': '#9C27B0',
        'input': '#2196F3', 
        'intermediate': '#FF5722',
        'output': '#4CAF50'
    }
    
    max_time = max(
        max([node['end'] for node in data['nodes']] + [0]),
        max([comm['end'] for comm in data['comm']] + [0]),
        max([t['end'] for t in data['tensor_l2']] + [0]),
        max([t['end'] for t in data['tensor_l3']] + [0])
    )
    
    addr_scale = data['addr_scale']
    time_scale = data['time_scale']
    max_time *= time_scale
    
    # Plot 1: Gantt Chart
    plot_gantt_chart(ax1, deepcopy(data), time_scale, device_colors, comm_colors)
    
    # Plot 2: L2 Memory Layout
    plot_memory_layout(ax2, deepcopy(data['tensor_l2']), deepcopy(data['tensor_static_l2']), deepcopy(data['comm']), time_scale, addr_scale, tensor_type_colors, 'L2 Memory Layout', max_time)
    
    # Plot 3: L3 Memory Layout  
    plot_memory_layout(ax3, deepcopy(data['tensor_l3']), deepcopy(data['tensor_static_l3']), deepcopy(data['comm']), time_scale, addr_scale, tensor_type_colors, 'L3 Memory Layout', max_time)
    
    # Synchronize x-axes for time
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, max_time)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

def plot_gantt_chart(ax, data, time_scale, device_colors, comm_colors):
    """Plot Gantt chart for node execution and communication operations"""
    
    # Determine y-axis layout
    num_devices = len(set(node['device'] for node in data['nodes']))
    y_positions = {}
    y_labels = []
    
    # Device positions
    for i in range(num_devices):
        y_positions[f'device_{i}'] = i
        y_labels.append(f'Device {i}')
    
    # Single DMA position for both load and store operations
    dma_y_pos = num_devices
    y_positions['load'] = dma_y_pos
    y_positions['store'] = dma_y_pos
    y_labels.append('DMA')
    
    # Rescale
    for i in range(len(data['nodes'])):
        data['nodes'][i]['start'] *= time_scale
        data['nodes'][i]['end'] *= time_scale
    for i in range(len(data['comm'])):
        data['comm'][i]['start'] *= time_scale
        data['comm'][i]['end'] *= time_scale
    
    # Plot nodes
    for node in data['nodes']:
        y_pos = y_positions[f'device_{node["device"]}']
        duration = node['end'] - node['start']
        
        rect = Rectangle(
            (node['start'], y_pos - 0.5), 
            duration, 1,
            facecolor=device_colors[node['device'] % len(device_colors)],
            edgecolor='black',
            linewidth=1,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        node_text = node['name']
        if node['chunks'] != node['total_chunks']:
            node_text += f" ({node['chunks']}/{node['total_chunks']})"
            
        # Add node label
        ax.text(
            node['start'] + duration/2, y_pos,
            node_text,
            ha='center', va='center',
            fontweight='bold',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3),
            color='white'
        )
    
    # Plot communication operations on the same DMA line
    for comm in data['comm']:
        y_pos = y_positions[comm['type']]
        duration = comm['end'] - comm['start']
        
        rect = Rectangle(
            (comm['start'], y_pos - 0.5),
            duration, 1,
            facecolor=comm_colors[comm['type']],
            hatch=hatch_patterns[comm['type']],
            edgecolor='black',
            linewidth=1,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add tensor label
        ax.text(
            comm['start'] + duration/2, y_pos,
            comm['name'],
            ha='center', va='center',
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3),
            color='white'
        )
    
    # Styling
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_title('Execution Schedule', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend for node devices and DMA operations
    device_patches = [patches.Patch(color=device_colors[i % len(device_colors)], 
                                  label=f'Device {i}') for i in range(num_devices)]
    comm_patches = [patches.Patch(color=comm_colors[op], label=f'DMA {op.capitalize()}') 
                   for op in comm_colors.keys()]
    
    ax.legend(handles=device_patches + comm_patches, 
             bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add devices utilization info
    max_time = max(
        max([node['end'] for node in data['nodes']] + [0]),
        max([comm['end'] for comm in data['comm']] + [0])
    )
    total_usage_time = sum(
        node['end'] - node['start'] for node in data['nodes']
    ) # + sum(
    #    comm['end'] - comm['start'] for comm in data['comm']
    #)
    avg_utilization = total_usage_time / (max_time * num_devices) * 100 if max_time > 0 else 0
        
    ax.text(1.05, 0.05, f'Avg Utilization: {avg_utilization:.1f}%',
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
           fontsize=9)

def plot_memory_layout(ax, tensor_data, static_tensor_data, comm_data, time_scale, addr_scale, tensor_type_colors, title, max_time):
    """Plot 2D bin packing visualization for memory layout with transfer overlays"""
    
    if not tensor_data and not static_tensor_data:
        ax.text(0.5, 0.5, 'No tensor allocations', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, style='italic')
        ax.set_title(title, fontsize=12, fontweight='bold')
        return
    
    # Add static tensor data
    static_tensor_data = sorted(static_tensor_data, key=lambda x: x['size'])
    static_tensor_data_size = sum(t['size'] for t in static_tensor_data)
    l2_max_time = max(t['end'] for t in tensor_data)
    for i in range(len(tensor_data)):
        tensor_data[i]['addr_start'] += static_tensor_data_size
    current_offset = 0
    for i in range(len(static_tensor_data)):
        tensor_data.append({
            'start': 0,
            'end': l2_max_time,
            'addr_start': current_offset,
            'size': static_tensor_data[i]['size'],
            'tensor_id': static_tensor_data[i]['tensor_id'],
            'type': static_tensor_data[i]['type'],
            'chunks': static_tensor_data[i]['chunks'],
            'total_chunks': static_tensor_data[i]['total_chunks']
        })
        current_offset += static_tensor_data[i]['size']
            
    # Rescale
    for i in range(len(tensor_data)):
        tensor_data[i]['start'] *= time_scale
        tensor_data[i]['end'] *= time_scale
        tensor_data[i]['addr_start'] *= addr_scale
        tensor_data[i]['size'] *= addr_scale
    for i in range(len(comm_data)):
        comm_data[i]['start'] *= time_scale
        comm_data[i]['end'] *= time_scale
        
    
    
    # Plot tensor rectangles first
    for i, tensor in enumerate(tensor_data):
        duration = tensor['end'] - tensor['start']
        size = tensor['chunks'] * (tensor['size'] // tensor['total_chunks'])
        
        if duration == 0 or size == 0:
            continue
        
        rect = Rectangle(
            (tensor['start'], tensor['addr_start']),
            duration, size,
            facecolor=tensor_type_colors.get(tensor['type'], '#CCCCCC'),
            edgecolor='black',
            linewidth=1,
            alpha=0.8,
            zorder=1  # Base layer
        )
        ax.add_patch(rect)
        
        # Add tensor ID with better positioning
        label_x = tensor['start'] + duration/2
        label_y = tensor['addr_start'] + size /2
        
        tensor_text = f"T{tensor['tensor_id']:02d}"
        if tensor['chunks'] != tensor['total_chunks']:
            tensor_text += f" ({tensor['chunks']}/{tensor['total_chunks']})"
            
        ax.text(
            label_x, label_y,
            tensor_text,
            ha='center', va='center',
            fontsize=8,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3),
            zorder=2
        )
    
    # Create dictionaries to track tensor addresses and sizes for matching transfers
    tensor_addr_map = {}
    tensor_size_map = {}
    for tensor in tensor_data:
        tensor_id = tensor['tensor_id']
        if tensor_id not in tensor_addr_map:
            tensor_addr_map[tensor_id] = []
            tensor_size_map[tensor_id] = tensor['chunks'] * (tensor['size'] // tensor['total_chunks'])
        tensor_addr_map[tensor_id].append({
            'start_time': tensor['start'],
            'end_time': tensor['end'],
            'addr_start': tensor['addr_start']
        })
    
    # Plot transfer overlays with hatching
    for comm in comm_data:
        tensor_id = comm['tensor_id']
        transfer_type = comm['type']
        transfer_start = comm['start']
        transfer_end = comm['end']
        
        # Find matching tensor allocation for this transfer
        if tensor_id in tensor_addr_map:
            tensor_size = tensor_size_map[tensor_id]
            
            # Find the tensor allocation that overlaps with this transfer
            for tensor_alloc in tensor_addr_map[tensor_id]:
                alloc_start = tensor_alloc['start_time']
                alloc_end = tensor_alloc['end_time']
                addr_start = tensor_alloc['addr_start']
                
                # Check if transfer overlaps with this allocation
                if (transfer_start < alloc_end and transfer_end > alloc_start):
                    # Determine the memory type based on title
                    memory_type = 'L2' if 'L2' in title else 'L3'
                    
                    # For loads to L2 or stores from L3, show the hatching during transfer
                    if ((transfer_type == 'load' and memory_type == 'L2') or 
                        (transfer_type == 'store' and memory_type == 'L3')):
                        
                        # Calculate the overlap period
                        overlap_start = max(transfer_start, alloc_start)
                        overlap_end = min(transfer_end, alloc_end)
                        
                        if overlap_end > overlap_start:
                            # Create hatched rectangle overlay
                            hatch_rect = Rectangle(
                                (overlap_start, addr_start),
                                overlap_end - overlap_start,
                                tensor_size,
                                facecolor='none',
                                edgecolor='black',
                                linewidth=1,
                                hatch=hatch_patterns[transfer_type],
                                alpha=0.6,
                                zorder=10  # Top layer
                            )
                            ax.add_patch(hatch_rect)
                    
                    # For stores from L2 or loads to L3, show hatching during transfer
                    elif ((transfer_type == 'store' and memory_type == 'L2') or 
                          (transfer_type == 'load' and memory_type == 'L3')):
                        
                        # Calculate the overlap period
                        overlap_start = max(transfer_start, alloc_start)
                        overlap_end = min(transfer_end, alloc_end)
                        
                        if overlap_end > overlap_start:
                            # Create hatched rectangle overlay
                            hatch_rect = Rectangle(
                                (overlap_start, addr_start),
                                overlap_end - overlap_start,
                                tensor_size,
                                facecolor='none',
                                edgecolor='black',
                                linewidth=1,
                                hatch=hatch_patterns[transfer_type],
                                alpha=0.6,
                                zorder=10  # Top layer
                            )
                            ax.add_patch(hatch_rect)
    
    # Styling
    max_addr = max([t['addr_start'] + t['size'] for t in tensor_data]) if tensor_data else 1
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'0x{int(y):08X}'))
    ax.set_ylim(0, max_addr + 1)
    ax.set_ylabel('Memory Address', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Add legend for tensor types and transfer patterns
    type_patches = [patches.Patch(color=tensor_type_colors[ttype], label=ttype.capitalize()) 
                   for ttype in set(t['type'] for t in tensor_data)]
    
    # Add hatch pattern legend
    hatch_patches = [
        patches.Patch(facecolor='none', edgecolor='gray', hatch='///', 
                     label='Load Transfer', linewidth=1),
        patches.Patch(facecolor='none', edgecolor='gray', hatch='\\\\\\', 
                     label='Store Transfer', linewidth=1)
    ]
    
    all_patches = type_patches + hatch_patches
    ax.legend(handles=all_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add memory utilization info
    total_memory_time = sum((t['end'] - t['start']) * t['size'] for t in tensor_data)
    #max_time = max(t['end'] for t in tensor_data) if tensor_data else 1
    avg_utilization = total_memory_time / (max_time * max_addr) * 100 if max_addr > 0 else 0
    
    ax.text(1.05, 0.05, f'Avg Utilization: {avg_utilization:.1f}%',
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
           fontsize=9)