import math

from .graph import Tensor, TensorType

from ortools.sat.python import cp_model as cp
import networkx as nx



def optimize(graph, devices, l2_size, l3_size, bandwidth, dtype_size, scale_time=True, scale_addr=False, tiling=False):
    model = cp.CpModel()
    
    nodes = graph.nodes
    super_nodes = graph.super_nodes
    tensors = graph.tensors
    
    # Scale
    addr_scale = 1
    time_scale = 1
    
    bandwidth = bandwidth // dtype_size
    l2_size = l2_size // dtype_size
    l3_size = l3_size // dtype_size
    
    if scale_time:
        time_scale = math.gcd(bandwidth, *(node.duration for node in graph.nodes + graph.super_nodes))
        for n in range(len(graph.nodes)):
            nodes[n].duration = int(graph.nodes[n].duration // time_scale)
        for n in range(len(graph.super_nodes)):
            super_nodes[n].duration = int(graph.super_nodes[n].duration // time_scale)
        bandwidth = bandwidth // time_scale
      
    if scale_addr:      
        # rescale tensor sizes with respect to smallest size
        addr_scale = math.gcd(*(tensor.size for tensor in graph.tensors))
        for t in range(len(graph.tensors)):
            tensors[t].size = int(graph.tensors[t].size // addr_scale)
        
        l2_size = l2_size // addr_scale
        l3_size = l3_size // addr_scale
        
    addr_scale *= dtype_size
    
    # Max makespan
    horizon = sum(node.duration for node in nodes)
    
    # Node execution variables
    node_vars = []
    node_active_vars = []
    for n, node in enumerate(nodes + super_nodes):
        start = model.new_int_var(0, horizon, f"node_{n}_start")
        end   = model.new_int_var(0, horizon, f"node_{n}_end")
        
        duration = model.new_int_var(0, horizon, f"node_{n}_duration")
        chunks = model.new_int_var(0, node.chunks, f"node_{n}_chunks")
        model.add(duration == node.chunk_duration * chunks)
        
        active = model.new_bool_var(f"node_{n}_active")
        model.add(chunks == 0).only_enforce_if(active.Not())  # Ensure node is active if it has chunks
        model.add(chunks > 0).only_enforce_if(active)  # Ensure node is active if it has chunks
        
        interval = model.new_optional_interval_var(start, duration, end, active, f"node_{n}_interval")  
        
        node_vars.append((start, end, duration, interval, active, chunks))

        node_active_vars.append(active)
    
    # Node conflicts
    for n, node in enumerate(nodes):
        chunks = 0
        for sid in graph.nid_to_sid[n]:
            chunks += node_vars[sid][5]
        chunks += node_vars[n][5]
        model.add(chunks == node.chunks)
        
        if not tiling:
            actives = sum(node_active_vars[sid] for sid in graph.nid_to_sid[n]) + node_active_vars[n]
            model.add(actives == 1)  # Only one node can be active at a time in the same pattern
    
    # Makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [var[1] for var in node_vars])
    
    # L2 available space for non permanent tensors
    l2_size_var = model.new_int_var(0, l2_size, "l2_size")

    # Add dependency constraints
    for node in nodes + super_nodes:
        for child_id in node.children_nids:
            model.add(node_vars[node.id][1] < node_vars[child_id][0]).only_enforce_if([node_active_vars[node.id], node_active_vars[child_id]])

    # Add non-overlapping node execution in devices
    for d in range(0, len(devices)):
        model.add_no_overlap([interval for (_, _, _, interval, _, _), node in zip(node_vars, nodes + super_nodes) if node.device_id == d])

    # Add tensor intervals
    tensor_lives_l2, tensor_lives_l3 = [], []
    tensor_addrs_l2, tensor_addrs_l3 = [], []
    tensor_loads, tensor_stores = [], []
    tensor_chunks = []
    tensor_static_in_l2, tensor_static_in_l3 = [], []

    num_tensor_lifetimes = 2

    for t, tensor in enumerate(tensors):
        t_lives_l2, t_addrs_l2 = [], []
        t_lives_l3, t_addrs_l3 = [], []
        t_loads, t_stores = [], []
        
        t_chunks = None
        t_static_in_l2 = model.new_bool_var(f"tensor_{t}_static_in_l2")
        t_static_in_l3 = model.new_bool_var(f"tensor_{t}_static_in_l3")

        if tensor.type == TensorType.INPUT or tensor.type == TensorType.OUTPUT:
            # Assume input tensors are in L2 at start
            # Assume output tensors are stored in L2 at end
            active = model.new_bool_var(f"tensor_{t}_active")
            
            start, end, duration, interval, _ = gen_opt_interval(model, 0, horizon, f"tensor_{t}_l2_life_0", active = active)
            t_lives_l2 = [(start, end, duration, interval, active)]
            
            addr_start, addr_end, addr_dur, addr_int, _ = gen_opt_interval(model, 0, l2_size, f"tensor_{t}_l2_addr_0", size=tensor.size, active=active)
            model.add(addr_end <= l2_size_var)
            t_addrs_l2 = [(addr_start, addr_end, addr_dur, addr_int, active)]
            
            model.add(active == t_static_in_l2.Not())
            model.add(t_static_in_l3 == 0)
            
            if tensor.type == TensorType.INPUT:
                model.add(start == 0)
            else:
                model.add(end == makespan)
                
            # For now make output and input static in l2
            model.add(t_static_in_l2 == 1)
                
        elif tensor.type == TensorType.INTERMEDIATE:
            # Assume intermediate can be in L2 two times, intermediate are stored and loaded from L3
            t_chunks = model.new_int_var(0, tensor.chunks, f"tensor_{t}_chunks")
            
            t_size_expr = t_chunks * (tensor.size // tensor.chunks)
            t_size = model.new_int_var(0, tensor.size, f"tensor_{t}_l2_tensor_size")
            model.add(t_size == t_size_expr)

            for i in range(num_tensor_lifetimes):
                segment_active = model.new_bool_var(f"tensor_{t}_l2_{i}_active")
                # An intermediate tensor borns in L2, can be stored in L3, and loaded back to L2 and so on

                if i > 0:
                    load_duration_expr = t_chunks * (tensor_load_time(tensor, bandwidth) // tensor.chunks)
                    load_duration = model.new_int_var(0,  tensor_load_time(tensor, bandwidth), f"tensor_{t}_load_duration_{i}")
                    model.add(load_duration == load_duration_expr)
                    load_start, load_end, load_duration, load_interval, _ = gen_opt_interval(
                        model, 0, horizon, f"tensor_{t}_load_{i}", size = load_duration, active = segment_active
                    )
                    t_load_interval = (load_start, load_end, load_duration, load_interval, segment_active)
                    t_loads.append(t_load_interval)

                l2_start, l2_end, l2_duration, l2_interval, _ = gen_opt_interval(
                    model, 0, horizon, f"tensor_{t}_l2_life_{i}", active=segment_active
                )
                t_interval_l2 = (l2_start, l2_end, l2_duration, l2_interval, segment_active)
                
                l2_addr_start, l2_addr_end, l2_tensor_size, l2_addr_interval, _ = gen_opt_interval(
                    model, 0, l2_size, f"tensor_{t}_l2_addr_{i}", size=t_size, active=segment_active
                )
                model.add(l2_addr_end <= l2_size_var)
                t_addr_l2 = (l2_addr_start, l2_addr_end, l2_tensor_size, l2_addr_interval, segment_active)

                if i == 0:
                    #model.add(segment_active == 1)  # Can exist in L2 at start after parent layer
                    model.add(segment_active <= sum(node_active_vars[n] for n in graph.tensor_to_nids[t]))
                else:
                    model.add(l2_start == load_start)
                    model.add(l2_end >= load_end)

                t_lives_l2.append(t_interval_l2)
                t_addrs_l2.append(t_addr_l2)

            for i in range(num_tensor_lifetimes - 1):
                segment_active = model.new_bool_var(f"tensor_{t}_l3_{i}_active")

                # Each L3 segment start with a store
                store_duration_expr = t_chunks * (tensor_store_time(tensor, bandwidth) // tensor.chunks)
                store_duration = model.new_int_var(0,  tensor_store_time(tensor, bandwidth), f"tensor_{t}_store_duration_{i}")
                model.add(store_duration == store_duration_expr)
                
                store_start, store_end, store_duration, store_interval, _ = gen_opt_interval(
                    model, 0, horizon, f"tensor_{t}_store_{i}", size=store_duration, active=segment_active
                )
                t_store_interval = (store_start, store_end, store_duration, store_interval, segment_active)

                t_stores.append(t_store_interval)

                l3_start, l3_end, l3_duration, l3_interval, _ = gen_opt_interval(
                    model, 0, horizon, f"tensor_{t}_l3_life_{i}", active=segment_active
                )
                t_interval_l3 = (l3_start, l3_end, l3_duration, l3_interval, segment_active)
                
                l3_addr_start, l3_addr_end, l3_tensor_size, l3_addr_interval, _ = gen_opt_interval(
                    model, 0, l3_size, f"tensor_{t}_l3_addr_{i}", size=t_size, active=segment_active
                )
                t_addr_l3 = (l3_addr_start, l3_addr_end, l3_tensor_size, l3_addr_interval, segment_active)

                t_lives_l3.append(t_interval_l3)
                t_addrs_l3.append(t_addr_l3)

                model.add(l3_start == store_start)  # L3 segment starts after store
                model.add(l3_end >= store_end)

            # For each load enforce it happening during at least one L3 segment
            for i in range(num_tensor_lifetimes - 1):
                load_during_l3_segment = 0
                for j in range(num_tensor_lifetimes - 1):
                    load_start, load_end, load_duration, load_interval, l2_segment_active = t_loads[i]
                    l3_start, l3_end, l3_duration, l3_interval, l3_segment_active = t_lives_l3[j]
                    overlapping = model.new_bool_var(f"load_{i}_during_l3_{j}")
                    model.add(load_start >= l3_start).only_enforce_if(overlapping)
                    model.add(load_end <= l3_end).only_enforce_if(overlapping)
                    model.add(overlapping <= l3_segment_active)
                    load_during_l3_segment += overlapping
                model.add(load_during_l3_segment >= l2_segment_active)

            # For each store enforce it happening during at least one L2 segment
            for i in range(num_tensor_lifetimes - 1):
                store_during_l2_segment = 0
                for j in range(num_tensor_lifetimes):
                    store_start, store_end, store_duration, store_interval, l3_segment_active = t_stores[i]
                    l2_start, l2_end, l2_duration, l2_interval, l2_segment_active = t_lives_l2[j]
                    overlapping = model.new_bool_var(f"store_{i}_during_l2_{j}")
                    model.add(store_start >= l2_start).only_enforce_if(overlapping)
                    model.add(store_end <= l2_end).only_enforce_if(overlapping)
                    model.add(overlapping <= l2_segment_active)
                    store_during_l2_segment += overlapping
                model.add(store_during_l2_segment >= l3_segment_active)
                
            model.add(t_static_in_l2 == 0)
            model.add(t_static_in_l3 == 0)

        elif tensor.type == TensorType.CONST:
            # Const tensor must be in L2 from start to end or in L3 from start to end
            # In the latter case, it can be loaded one time to L2 and used in a node


            # L2 life can start with a load or be the only tensor life
        
            load_start, load_end, load_duration, load_interval, l2_active = gen_opt_interval(
                model, 0, horizon, f"tensor_{t}_load_interval", size=tensor_load_time(tensor, bandwidth)
            )
            t_load_interval = (load_start, load_end, load_duration, load_interval, l2_active)
            t_loads.append(t_load_interval)

            l2_start, l2_end, l2_duration, l2_interval, l2_active = gen_opt_interval(
                model, 0, horizon, f"tensor_{t}_l2_interval", active=l2_active
            )
            t_interval_l2 = (l2_start, l2_end, l2_duration, l2_interval, l2_active)
            
            l2_addr_start, l2_addr_end, l2_tensor_size, l2_addr_interval, _ = gen_opt_interval(
                model, 0, l2_size, f"tensor_{t}_l2_address", size=tensor.size, active=l2_active
            )
            model.add(l2_addr_end <= l2_size_var)
            t_addr_l2 = (l2_addr_start, l2_addr_end, l2_tensor_size, l2_addr_interval, l2_active)
            
            model.add(t_static_in_l2 == l2_active.Not())
            model.add(l2_active == t_static_in_l3)  # Cannot load if l3 not active
            
            #model.add(l2_start == 0).only_enforce_if(l2_active.Not())
            #model.add(l2_end == makespan).only_enforce_if(l2_active.Not())

            model.add(l2_start == load_start).only_enforce_if(l2_active)
            model.add(l2_end >= load_end).only_enforce_if(l2_active)

            # Cannot load if l2 not active
            t_lives_l2.append(t_interval_l2)
            t_addrs_l2.append(t_addr_l2)

            
        tensor_lives_l2.append(t_lives_l2)
        tensor_addrs_l2.append(t_addrs_l2)
        
        tensor_lives_l3.append(t_lives_l3)
        tensor_addrs_l3.append(t_addrs_l3)
        
        tensor_loads.append(t_loads)
        tensor_stores.append(t_stores)
        
        tensor_chunks.append(t_chunks)
        tensor_static_in_l2.append(t_static_in_l2)
        tensor_static_in_l3.append(t_static_in_l3)
        
            
    # Tensor 2D bin packing in L2 and L3
    
    t_lives_l2flat = [interval[3] for sublist in tensor_lives_l2 for interval in sublist]
    t_addrs_l2flat = [addr[3] for sublist in tensor_addrs_l2 for addr in sublist]
    model.add_no_overlap_2d(t_lives_l2flat, t_addrs_l2flat)
    
    t_lives_l3flat = [interval[3] for sublist in tensor_lives_l3 for interval in sublist]
    t_addrs_l3flat = [addr[3] for sublist in tensor_addrs_l3 for addr in sublist]
    model.add_no_overlap_2d(t_lives_l3flat, t_addrs_l3flat)

    # Node and supernode - Tensor relation constraints
    node_needed_tensors_segment_ids = []
    
    for n, node in enumerate(nodes + super_nodes):
        node_needed_tensors_segment_ids.append({})
        for tensor_id in (node.inp_tids + node.out_tids):
            # Make sure that while execution a node, needed tensors are alive in L2
            tensor = tensors[tensor_id]
            node_needed_tensors_segment_ids[-1][tensor_id] = []
            
            contained_in_at_least_one = 0
            node_start, node_end, _, _, _, _ = node_vars[n]
            for j, (l2_start, l2_end, l2_duration, l2_interval, active) in enumerate(tensor_lives_l2[tensor_id]):
                contained = model.new_bool_var(f"node_{n}_tensor_{tensor_id}_l2_contained_in_{j}")
                model.add(contained <= active)
                if tensor_id in node.out_tids and j == 0:
                    model.add(node_start >= l2_start).only_enforce_if(node_active_vars[n])
                    model.add(node_end <= l2_end).only_enforce_if(node_active_vars[n])
                    model.add(contained == tensor_static_in_l2[tensor_id].Not()).only_enforce_if(node_active_vars[n])
                else:
                    model.add(node_start >= l2_start).only_enforce_if(contained)
                    model.add(node_end <= l2_end).only_enforce_if(contained)
                contained_in_at_least_one += contained
                
                node_needed_tensors_segment_ids[-1][tensor_id].append(contained)
                
            contained_in_at_least_one += tensor_static_in_l2[tensor_id]
            model.add(contained_in_at_least_one >= node_active_vars[n])
            
        # Avoid overlapping layer compute with tensor loads and stores of relevant tensors
        node_intervals_ = [node_vars[n][3]]
        load_intervals_ = [interval for tensor_id in (node.inp_tids + node.out_tids) for _,_,_,interval,_ in tensor_loads[tensor_id] ]
        store_intervals_ = [interval for tensor_id in (node.inp_tids + node.out_tids) for _,_,_,interval,_ in tensor_stores[tensor_id] ]
        model.add_no_overlap(node_intervals_ + load_intervals_ + store_intervals_)

    
    # Tensor chunks for output tensors
    for t, tensor in enumerate(tensors):
        chunks_needed = 0
        if tensor_chunks[t] is not None:
            for n, node in enumerate(nodes + super_nodes):
                if tensor.id in node.out_tids:
                    chunks_needed += node_vars[n][5]
            model.add(tensor_chunks[t] == chunks_needed)
    

    # Make sure store and loads are not overlapping, ideally we could consider a maximum amount of concurrent transfers
    
    load_intervals_flat = [interval[3] for sublist in tensor_loads for interval in sublist]
    store_intervals_flat = [interval[3] for sublist in tensor_stores for interval in sublist]
    model.add_no_overlap(load_intervals_flat + store_intervals_flat)
    
    # No lifetime after makespan
    for t, tensor in enumerate(tensors):
        for i, (l2_start, l2_end, l2_duration, l2_interval, active) in enumerate(tensor_lives_l2[t]):
            model.add(l2_end <= makespan)
        for i, (l3_start, l3_end, l3_duration, l3_interval, active) in enumerate(tensor_lives_l3[t]):
            model.add(l3_end <= makespan)
    # No transfers after makespan
    for t, tensor in enumerate(tensors):
        for i, (load_start, load_end, load_duration, load_interval, active) in enumerate(tensor_loads[t]):
            model.add(load_end <= makespan)
        for i, (store_start, store_end, store_duration, store_interval, active) in enumerate(tensor_stores[t]):
            model.add(store_end <= makespan)
            
    # L2 Size constraint removing const tensor occupation
    model.add(l2_size_var == (l2_size - sum(tensor_static_in_l2[tensor_id]*tensor.size for t, tensor in enumerate(tensors) if tensor.type == TensorType.CONST)))
    
    # Objective
    
    num_loads = sum(tensor_loads[t][i][4] for t in range(len(tensors)) for i in range(len(tensor_loads[t])))
    num_stores = sum(tensor_stores[t][i][4] for t in range(len(tensors)) for i in range(len(tensor_stores[t])))
    
    #model.add(makespan >= 589300)
    model.minimize(makespan + num_loads + num_stores)
    
    # Solve
    
    solver = cp.CpSolver()
    solver.parameters.log_search_progress = False
    solver.parameters.num_workers = 32
    
    print("  Optimizing...")
    
    status = solver.solve(model)
    
    if status != cp.OPTIMAL and status != cp.FEASIBLE:
        raise Exception("No feasible solution found")

    print(f"  {solver.status_name(status)} solution found.")
    
    # Fix makespan minimize l2_usage
    if True:
        print("Now minimizing L2 and L3 usage...")
        model.add(makespan <= solver.value(makespan))  # fix previous objective
        peak_l2_usage = model.new_int_var(0, l2_size, "peak_l2_usage")
        peak_l3_usage = model.new_int_var(0, l3_size, "peak_l3_usage")
        # Calculate peak L2 usage
        for t, tensor in enumerate(tensors):
            for i, (l2_start, l2_end, l2_duration, l2_interval, active) in enumerate(tensor_addrs_l2[t]):
                model.add(peak_l2_usage >= l2_end).only_enforce_if(active)
            for i, (l3_start, l3_end, l3_duration, l3_interval, active) in enumerate(tensor_addrs_l3[t]):
                model.add(peak_l3_usage >= l3_end).only_enforce_if(active)
        model.minimize(10*peak_l3_usage + peak_l2_usage + num_loads + num_stores)  # optimize for second objective
        status = solver.solve(model)
        assert status == cp.OPTIMAL
    
    # Extract solution
    print("  Extracting solution...")
    used_tensor_segments = {tid: set() for tid in range(len(tensors))}
    
    node_data = []
    for n, node in enumerate(nodes+super_nodes):
        node_start, node_end, node_duration, node_interval, node_active, chunks = node_vars[n]
        if solver.boolean_value(node_active):
            tensor_segments = {}
            for tid, segments in node_needed_tensors_segment_ids[n].items():
                for j, active in enumerate(segments):
                    if solver.boolean_value(active):
                        tensor_segments[tid] = j
                if solver.boolean_value(tensor_static_in_l2[tid]):
                    tensor_segments[tid] = 0
            for tid, j in tensor_segments.items():
                if not solver.boolean_value(tensor_static_in_l2[tid]):
                    used_tensor_segments[tid].add(j)
            node_data.append({
                'node_id': node.id,
                'name': f"L{node.id:02d}",
                'start': solver.value(node_start),
                'end': solver.value(node_end),
                'device': node.device_id,
                'super': False,
                'chunks': solver.value(chunks) if chunks is not None else node.chunks,
                'total_chunks': node.chunks,
                'tensors_segments': tensor_segments
            })        
            
    comm_data = []
    for t, tensor in enumerate(tensors):
        # Load operations
        for i, (load_start, load_end, load_duration, load_interval, active) in enumerate(tensor_loads[t]):
            if solver.boolean_value(active):
                comm_data.append({
                    'name': f"T{t:02d}",
                    'start': solver.value(load_start),
                    'end': solver.value(load_end),
                    'type': 'load',
                    'tensor_id': t
                })
        
        # Store operations
        for i, (store_start, store_end, store_duration, store_interval, active) in enumerate(tensor_stores[t]):
            if solver.boolean_value(active):
                comm_data.append({
                    'name': f"T{t:02d}",
                    'start': solver.value(store_start),
                    'end': solver.value(store_end),
                    'type': 'store',
                    'tensor_id': t
                })
                
    # L2 tensor allocations
    tensor_l2_data = []
    for t, tensor in enumerate(tensors):
        for i, (l2_start, l2_end, l2_duration, l2_interval, active) in enumerate(tensor_lives_l2[t]):
            if solver.boolean_value(active):
                tensor_l2_data.append({
                    'tensor_id': t,
                    'start': solver.value(l2_start),
                    'end': solver.value(l2_end),
                    'addr_start': solver.value(tensor_addrs_l2[t][i][0]),
                    'size': tensor.size,
                    'type': tensor.type.value,
                    'chunks': solver.value(tensor_chunks[t]) if tensor_chunks[t] is not None else tensor.chunks,
                    'total_chunks': tensor.chunks
                })
    # L2 static allocations
    tensor_static_l2_data = []
    for t, tensor in enumerate(tensors):
        if solver.boolean_value(tensor_static_in_l2[t]):
            tensor_static_l2_data.append({
                'tensor_id': t,
                'size': tensor.size,
                'type': tensor.type.value,
                'chunks': solver.value(tensor_chunks[t]) if tensor_chunks[t] is not None else tensor.chunks,
                'total_chunks': tensor.chunks
            })
    # L3 tensor allocations
    tensor_l3_data = []
    for t, tensor in enumerate(tensors):
        for i, (l3_start, l3_end, l3_duration, l3_interval, active) in enumerate(tensor_lives_l3[t]):
            if solver.boolean_value(active):
                tensor_l3_data.append({
                    'tensor_id': t,
                    'start': solver.value(l3_start),
                    'end': solver.value(l3_end),
                    'addr_start': solver.value(tensor_addrs_l3[t][i][0]),
                    'size': tensor.size,
                    'type': tensor.type.value,
                    'chunks': solver.value(tensor_chunks[t]) if tensor_chunks[t] is not None else tensor.chunks,
                    'total_chunks': tensor.chunks
                })
    # L3 static allocations
    tensor_static_l3_data = []
    for t, tensor in enumerate(tensors):
        if solver.boolean_value(tensor_static_in_l3[t]):
            tensor_static_l3_data.append({
                'tensor_id': t,
                'size': tensor.size,
                'type': tensor.type.value,
                'chunks': tensor.chunks,
                'total_chunks': tensor.chunks
            })
        
    # Tensors Data
    tensors_data = [
        {
            'id': tensors[t].id,
            'static_in_l2': solver.boolean_value(tensor_static_in_l2[t]),
            'static_in_l3': solver.boolean_value(tensor_static_in_l3[t]),
            'l2_offsets': [solver.value(tensor_addrs_l2[t][i][0]) for i, l2 in enumerate(tensor_lives_l2[t]) if i in used_tensor_segments[t] and not solver.boolean_value(tensor_static_in_l2[t])],
            'l3_offsets': [solver.value(tensor_addrs_l3[t][i][0]) for i, l3 in enumerate(tensor_lives_l3[t]) if solver.boolean_value(l3[4])]
        }
        for t in range(len(tensors))
    ]
            
    # Graph Data
    graph_data = {
        'nodes': [
            {
                'id': node.id,
                'total_chunks': node.chunks,
                'children': node.children_nids,
                'inp_tensors': node.inp_tids,
                'out_tensors': node.out_tids,
            }
            for node in nodes
        ],
        'super_nodes': [
            {
                'id': node.id,
                'total_chunks': node.chunks,
                'children': node.children_nids,
                'inp_tensors': node.inp_tids,
                'out_tensors': node.out_tids,
            }
            for node in super_nodes
        ],
        'tensors': [
            {
                'id': tensors[t].id,
                'shape': tensors[t].shape,
                'size': tensors[t].size,
                'chunks': tensors[t].chunks,
                'node_ids': graph.tensor_to_nids[t],
            }
            for t in range(len(tensors))
        ]
    }
    
    matched_patterns_gids = {}
    matched_patterns_chunks = {}
    for n, node in enumerate(super_nodes, start=len(nodes)):
        if solver.boolean_value(node_active_vars[n]):
            matched_patterns_gids.setdefault(node.pattern_id, []).extend([graph.nid_to_gid[n] for n in node.sub_nids])
            for n_ in node.sub_nids:
                matched_patterns_chunks.setdefault(node.pattern_id, {})[graph.nid_to_gid[n_]] = solver.value(node_vars[n][5])
                
     
    print("  Solution extracted successfully")
    solution = {
        'nodes': node_data,
        'comm': comm_data,
        'tensors': tensors_data,
        'tensor_l2': tensor_l2_data,
        'tensor_static_l2': tensor_static_l2_data,
        'tensor_l3': tensor_l3_data,
        'tensor_static_l3': tensor_static_l3_data,
        'addr_scale': addr_scale,
        'time_scale': time_scale,
        'graph': graph_data,
        'l2_peak_usage': solver.value(peak_l2_usage),
        'l3_peak_usage': solver.value(peak_l3_usage),
        'matched_patterns_gids': matched_patterns_gids,
        'matched_patterns_chunks': matched_patterns_chunks
    }
    
    
    
    
    return model, solver, solution


def gen_opt_interval(model, mn, mx, name, size=None, active=None):
    start = model.new_int_var(mn, mx, f"{name}_start")
    end = model.new_int_var(mn, mx, f"{name}_end")
    if size is None:
        size = model.new_int_var(0, mx-mn, f"{name}_dur")
    if active is None:
        active = model.new_bool_var(f"{name}_active")
    interval = model.new_optional_interval_var(start, size, end, active, f"{name}_interval")
    return start, end, size, interval, active


def tensor_load_time(tensor : Tensor, bandwidth : int = 8):
    return int(tensor.size / bandwidth)


def tensor_store_time(tensor : Tensor, bandwidth : int = 8):
    return int(tensor.size / bandwidth)