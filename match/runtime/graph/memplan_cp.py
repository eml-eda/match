import json
import os
from typing import Dict, List, Tuple

from match.runtime.graph.utils import save_memory_allocation_graph, save_memory_allocation_graph_nodes_buffers
from match.runtime.graph.waits import compute_async_waits

def cp_mem_planner_impl(planner, tensor_fixed_to_ext_mem: List[str] = []) -> Tuple[int, int, List[object]]:
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        raise ImportError(f"[MEM PLANNER] Please install ortools to use the CP-based memory planner: {e}")

    sorted_mem_tensors = sorted(
        [m_t for m_t in planner.mem_tensors + planner.extra_dynamic_buffers if m_t.lifetime != (-1, -1)],
        key=lambda m_t: (m_t.start_usage, -m_t.num_bytes),
    )
    calls = list(planner.calls_idxs)
    call_to_pos = {c: i for i, c in enumerate(calls)}

    model = cp_model.CpModel()
    max_mem = planner.available_soc_bytes

    target = getattr(planner, "target", None)
    async_dma = bool(getattr(target, "async_off_chip_to_on_chip_dm", False))
    # Bound async prefetch horizon to keep CP-SAT tractable on larger graphs.
    # A small lookback still enables real prefetching while avoiding combinatorial blow-up.
    max_prefetch_lookback = int(getattr(target, "max_async_prefetch_lookback", 8)) if async_dma else 0
    # Bound async write-back lookahead separately (defaults to same value as prefetch lookback).
    max_writeback_lookahead = int(getattr(target, "max_async_writeback_lookahead", max_prefetch_lookback)) if async_dma else 0

    def _alignment_of(tensor) -> int:
        if hasattr(tensor, "dtype") and hasattr(tensor.dtype, "itemsize") and tensor.dtype.itemsize:
            return int(tensor.dtype.itemsize)
        return 4

    def _new_aligned_offset(base_name: str, size: int, alignment: int):
        start = model.NewIntVar(0, max(0, max_mem - size), f"start_{base_name}")
        mult = model.NewIntVar(0, max_mem // max(1, alignment), f"mult_{base_name}")
        model.Add(start == mult * max(1, alignment))
        end = model.NewIntVar(size, max_mem, f"end_{base_name}")
        model.Add(end == start + size)
        return start, end

    tensor_vars: Dict[object, Dict[str, object]] = {}
    for tensor in sorted_mem_tensors:
        alignment = _alignment_of(tensor)
        on_chip = model.NewBoolVar(f"on_chip_{tensor.name}")
        off_chip = model.NewBoolVar(f"off_chip_{tensor.name}")
        model.Add(on_chip + off_chip == 1)

        if tensor.name in tensor_fixed_to_ext_mem or tensor.num_bytes > max_mem:
            model.Add(on_chip == 0)

        start, end = _new_aligned_offset(tensor.name, tensor.num_bytes, alignment)
        tensor_vars[tensor] = {
            "on_chip": on_chip,
            "off_chip": off_chip,
            "start": start,
            "end": end,
        }

    constant_tensors = [t for t in sorted_mem_tensors if getattr(t, "is_constant", False)]
    const_on_chip_bytes = sum(tensor_vars[t]["on_chip"] * t.num_bytes for t in constant_tensors)

    event_vars: Dict[tuple, Dict[str, object]] = {}
    writeback_event_vars: Dict[tuple, Dict[str, object]] = {}
    output_stage_vars: Dict[object, Dict[str, object]] = {}
    load_alive_by_time: Dict[int, List[Tuple[object, object, tuple]]] = {c: [] for c in calls}
    writeback_alive_by_time: Dict[int, List[Tuple[object, object, tuple]]] = {c: [] for c in calls}
    transfer_lateness_terms = []

    # Off-chip outputs still need a valid on-chip staging buffer when produced.
    # Model that staging window so its address participates in overlap/capacity
    # constraints, even for outputs whose tensor.node_id is not a valid call id
    # (e.g., duplicated heads with node_id = -1).
    for tensor in sorted_mem_tensors:
        if not getattr(tensor, "is_output", False):
            continue

        produce_time = None
        if tensor.node_id in call_to_pos:
            produce_time = tensor.node_id
        else:
            uses = sorted(set(getattr(tensor, "used_at", []) or []))
            valid_uses = [u for u in uses if u in call_to_pos]
            if valid_uses:
                produce_time = valid_uses[0]

        if produce_time is None:
            continue

        alignment = _alignment_of(tensor)
        stage_start, stage_end = _new_aligned_offset(f"{tensor.name}_out_stage_{produce_time}", tensor.num_bytes, alignment)
        output_stage_vars[tensor] = {
            "start": stage_start,
            "end": stage_end,
            "produce_time": produce_time,
        }

    for tensor in sorted_mem_tensors:
        if getattr(tensor, "is_output", False):
            continue

        uses = sorted(set(getattr(tensor, "used_at", []) or []))
        if len(uses) == 0:
            uses = [tensor.node_id]

        if getattr(tensor, "is_input", False) or getattr(tensor, "is_constant", False):
            uses = [uses[0]]

        use_positions = [call_to_pos[u] for u in uses if u in call_to_pos]
        if not use_positions:
            continue

        for event_idx, use_pos in enumerate(use_positions):
            use_time = calls[use_pos]
            key = (tensor, use_time, event_idx)

            alignment = _alignment_of(tensor)
            event_start, event_end = _new_aligned_offset(f"{tensor.name}_u{use_time}", tensor.num_bytes, alignment)

            if not async_dma:
                earliest = use_pos
            else:
                earliest_allowed = max(0, use_pos - max_prefetch_lookback)
                earliest = earliest_allowed if event_idx == 0 else max(earliest_allowed, use_positions[event_idx - 1] + 1)
                earliest = min(earliest, use_pos)

            choices = []
            for p in range(earliest, use_pos + 1):
                b = model.NewBoolVar(f"prefetch_{tensor.name}_u{use_time}_at_{calls[p]}")
                choices.append((p, b))

            model.Add(sum(b for _, b in choices) == tensor_vars[tensor]["off_chip"])

            alive_at = {}
            for pos in range(earliest, use_pos + 1):
                alive = model.NewBoolVar(f"alive_{tensor.name}_u{use_time}_at_{calls[pos]}")
                model.Add(alive == sum(b for p, b in choices if p <= pos))
                time_idx = calls[pos]
                alive_at[time_idx] = alive
                load_alive_by_time[time_idx].append((alive, tensor, key))

            transfer_lateness_terms.append(sum((p - earliest) * b for p, b in choices))

            event_vars[key] = {
                "start": event_start,
                "end": event_end,
                "choices": choices,
                "earliest": earliest,
                "use_pos": use_pos,
                "use_time": use_time,
                "alive_at": alive_at,
            }

            # In async mode, model write-back to external memory as a first-class
            # transfer event so the source buffer occupancy is tracked until the
            # selected move time.
            if async_dma and not (getattr(tensor, "is_input", False) or getattr(tensor, "is_constant", False)):
                next_use_pos = use_positions[event_idx + 1] if event_idx + 1 < len(use_positions) else None
                earliest_move = use_pos + 1
                # Only model a write-back when there is a *gap* between uses.
                # This matches the async allocator behavior: buffers are kept
                # on-chip across consecutive uses and written back only when
                # there is a gap (i.e., a time not in used_at).
                if next_use_pos is None or next_use_pos <= use_pos + 1:
                    latest_move = earliest_move - 1
                else:
                    # Keep write-back strictly before the next use time.
                    # Also cap how far ahead a write-back can be chosen to
                    # limit combinatorial blow-up via `max_writeback_lookahead`.
                    latest_move = min(next_use_pos - 1, use_pos + max_writeback_lookahead)

                move_choices = []
                alive_at_writeback = {}
                if earliest_move <= latest_move:
                    for p in range(earliest_move, latest_move + 1):
                        b = model.NewBoolVar(f"writeback_{tensor.name}_u{use_time}_at_{calls[p]}")
                        move_choices.append((p, b))

                    model.Add(sum(b for _, b in move_choices) == tensor_vars[tensor]["off_chip"])

                    for pos in range(earliest_move, latest_move + 1):
                        alive = model.NewBoolVar(f"wb_alive_{tensor.name}_u{use_time}_at_{calls[pos]}")
                        model.Add(alive == sum(b for p, b in move_choices if p >= pos))
                        time_idx = calls[pos]
                        alive_at_writeback[time_idx] = alive
                        writeback_alive_by_time[time_idx].append((alive, tensor, key))

                    transfer_lateness_terms.append(sum((p - earliest_move) * b for p, b in move_choices))

                next_key = None
                if next_use_pos is not None:
                    next_key = (tensor, calls[next_use_pos], event_idx + 1)

                writeback_event_vars[key] = {
                    "choices": move_choices,
                    "earliest": earliest_move,
                    "latest": latest_move,
                    "use_pos": use_pos,
                    "use_time": use_time,
                    "alive_at": alive_at_writeback,
                    "next_key": next_key,
                }

    max_mem_used = model.NewIntVar(0, max_mem, "max_mem_used")
    # Reserve SoC space for on-chip constants (separate from match_mem).
    if constant_tensors:
        model.Add(max_mem_used + const_on_chip_bytes <= max_mem)
    for time_idx in calls:
        intervals = []
        cap_terms = []

        for tensor in sorted_mem_tensors:
            if getattr(tensor, "is_constant", False):
                # Constants are stored in a separate on-chip region when on-chip;
                # do not allocate match_mem offsets for them.
                continue
            on_chip = tensor_vars[tensor]["on_chip"]
            start = tensor_vars[tensor]["start"]
            end = tensor_vars[tensor]["end"]

            full_lifetime = (
                getattr(tensor, "is_input", False)
                or getattr(tensor, "is_output", False)
                or getattr(tensor, "is_constant", False)
            )
            active = full_lifetime or (tensor.start_usage <= time_idx <= tensor.last_usage)
            if not active:
                continue

            i_end = model.NewIntVar(tensor.num_bytes, max_mem, f"iend_{tensor.name}_{time_idx}")
            interval = model.NewOptionalIntervalVar(start, tensor.num_bytes, i_end, on_chip, f"iv_{tensor.name}_{time_idx}")
            intervals.append(interval)
            cap_terms.append(on_chip * tensor.num_bytes)
            model.Add(max_mem_used >= end - max_mem * (1 - on_chip))

        for alive, tensor, key in load_alive_by_time.get(time_idx, []):
            ev = event_vars[key]
            e_start = ev["start"]
            e_end = ev["end"]
            i_end = model.NewIntVar(tensor.num_bytes, max_mem, f"eend_{tensor.name}_{time_idx}_{ev['use_time']}")
            interval = model.NewOptionalIntervalVar(
                e_start,
                tensor.num_bytes,
                i_end,
                alive,
                f"eiv_{tensor.name}_{time_idx}_{ev['use_time']}",
            )
            intervals.append(interval)
            cap_terms.append(alive * tensor.num_bytes)
            model.Add(max_mem_used >= e_end - max_mem * (1 - alive))

        for alive, tensor, key in writeback_alive_by_time.get(time_idx, []):
            ev = event_vars[key]
            e_start = ev["start"]
            e_end = ev["end"]
            i_end = model.NewIntVar(tensor.num_bytes, max_mem, f"wb_eend_{tensor.name}_{time_idx}_{ev['use_time']}")
            interval = model.NewOptionalIntervalVar(
                e_start,
                tensor.num_bytes,
                i_end,
                alive,
                f"wb_eiv_{tensor.name}_{time_idx}_{ev['use_time']}",
            )
            intervals.append(interval)
            cap_terms.append(alive * tensor.num_bytes)
            model.Add(max_mem_used >= e_end - max_mem * (1 - alive))

        for tensor, out_ev in output_stage_vars.items():
            if out_ev["produce_time"] != time_idx:
                continue

            off_chip = tensor_vars[tensor]["off_chip"]
            e_start = out_ev["start"]
            e_end = out_ev["end"]
            i_end = model.NewIntVar(tensor.num_bytes, max_mem, f"out_eend_{tensor.name}_{time_idx}")
            interval = model.NewOptionalIntervalVar(
                e_start,
                tensor.num_bytes,
                i_end,
                off_chip,
                f"out_eiv_{tensor.name}_{time_idx}",
            )
            intervals.append(interval)
            cap_terms.append(off_chip * tensor.num_bytes)
            model.Add(max_mem_used >= e_end - max_mem * (1 - off_chip))

        if len(intervals) > 1:
            model.AddNoOverlap(intervals)
        if cap_terms:
            model.Add(sum(cap_terms) <= max_mem)

    ext_mem_candidates = [
        t
        for t in sorted_mem_tensors
        if not (getattr(t, "is_input", False) or getattr(t, "is_output", False) or getattr(t, "is_constant", False))
    ]
    ext_mem_used = model.NewIntVar(0, sum(t.num_bytes for t in ext_mem_candidates), "ext_mem_used")
    model.Add(ext_mem_used == sum(tensor_vars[t]["off_chip"] * t.num_bytes for t in ext_mem_candidates))

    compact_terms = []
    for tensor in sorted_mem_tensors:
        if getattr(tensor, "is_constant", False):
            continue
        start = tensor_vars[tensor]["start"]
        on_chip = tensor_vars[tensor]["on_chip"]
        start_on_chip = model.NewIntVar(0, max_mem, f"start_on_chip_{tensor.name}")
        model.Add(start_on_chip <= start)
        model.Add(start_on_chip <= max_mem * on_chip)
        model.Add(start_on_chip >= start - max_mem * (1 - on_chip))
        compact_terms.append(start_on_chip)

    for key, ev in event_vars.items():
        tensor = key[0]
        off_chip = tensor_vars[tensor]["off_chip"]
        e_start = ev["start"]
        e_start_off_chip = model.NewIntVar(0, max_mem, f"estart_offchip_{tensor.name}_{ev['use_time']}")
        model.Add(e_start_off_chip <= e_start)
        model.Add(e_start_off_chip <= max_mem * off_chip)
        model.Add(e_start_off_chip >= e_start - max_mem * (1 - off_chip))
        compact_terms.append(e_start_off_chip)

    sum_compact = sum(compact_terms) if compact_terms else 0
    sum_lateness = sum(transfer_lateness_terms) if transfer_lateness_terms else 0

    ext_mem_upper = sum(t.num_bytes for t in ext_mem_candidates)
    compact_upper = max_mem * max(1, len(compact_terms))
    lateness_upper = max(1, len(transfer_lateness_terms)) * max(1, len(calls))

    if async_dma:
        # Async mode: keep external-memory pressure low first, then aggressively
        # prefer *earlier* transfers (prefetch + write-back lateness), then peak on-chip usage,
        # and finally compactness.
        # Use bounded coefficients to avoid CP-SAT MODEL_INVALID on large graphs.
        # Priority order is preserved in practice by scale separation.
        w_ext = 10**9
        w_late = 10**6
        w_peak = 10**3
        model.Minimize(ext_mem_used * w_ext + sum_lateness * w_late + max_mem_used * w_peak + sum_compact)
    else:
        # Non-async mode: aggressively minimize on-chip peak first.
        # Use safe, bounded coefficients to avoid large-number issues in CP-SAT objective.
        # Lexicographic intent:
        #   1) minimize max_mem_used
        #   2) minimize ext_mem_used
        # Since ext_mem_used ∈ [0, ext_mem_upper], choosing w_peak = ext_mem_upper + 1
        # guarantees any 1-byte peak reduction dominates the full ext_mem_used range.
        w_peak = ext_mem_upper + 1
        model.Minimize(max_mem_used * w_peak + ext_mem_used)

    solver = cp_model.CpSolver()
    # Increase async solver time limit for harder graphs; keep sync smaller.
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers = 8

    class CustomSolutionCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.solutions_found = 0
            self.best_obj = None

        def on_solution_callback(self):
            self.solutions_found += 1
            self.best_obj = self.ObjectiveValue()

    import threading
    import time
    import sys

    stop_pbar = False
    status = None
    cb = CustomSolutionCallback()

    def update_pbar():
        start_time = time.time()
        max_t = 60.0
        bar_len = 30
        while not stop_pbar:
            elapsed = time.time() - start_time
            if elapsed > max_t:
                elapsed = max_t

            progress = elapsed / max_t
            filled_len = int(bar_len * progress)
            bar = "█" * filled_len + "-" * (bar_len - filled_len)

            best_obj = f"{cb.best_obj:.0f}" if cb.best_obj is not None else "N/A"
            msg = f"\r[MEM PLANNER] CP Solver |{bar}| {elapsed:.1f}/{max_t}s [sols: {cb.solutions_found}, obj: {best_obj}]   "

            sys.stdout.write(msg)
            sys.stdout.flush()

            if elapsed >= max_t:
                break

            time.sleep(0.1)

    pbar_thread = threading.Thread(target=update_pbar)
    pbar_thread.start()

    try:
        status = solver.Solve(model, cb)
    finally:
        stop_pbar = True
        pbar_thread.join()
        sys.stdout.write("\r" + " " * 110 + "\r")
        sys.stdout.flush()

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        status_name = solver.StatusName(status)
        # Collect solver diagnostics to help reproduce infeasible cases.
        try:
            stats = solver.ResponseStats()
        except Exception:
            stats = "<no response stats available>"
        try:
            sols = cb.solutions_found
        except Exception:
            sols = 0
        msg = (
            f"[MEM PLANNER] CP Planner could not find a feasible allocation (status: {status_name}).\n"
            f"Solver solutions found: {sols}\n"
            f"Solver stats: {stats}\n"
        )
        # Write diagnostics to planner out_path for offline inspection.
        try:
            diag_file = f"{planner.out_path}/memory_plan_cp_solver_diagnostics.txt"
            with open(diag_file, "w") as df:
                df.write(msg)
        except Exception:
            pass
        raise Exception(msg)

    actual_peak_mem = solver.Value(max_mem_used)
    ext_mem_needed = solver.Value(ext_mem_used)
    print(f"[MEM PLANNER] CP Planner found a solution! Peak Memory Used: {actual_peak_mem} | Off-chip: {ext_mem_needed}")

    for tensor in sorted_mem_tensors:
        tensor.load_from_ext_mem_at = []
        tensor.move_temp_to_ext_mem = []
        tensor.mem_offset_at = {}
        tensor.wait_async_off_chip_transfer_at = []

        is_on_chip = solver.Value(tensor_vars[tensor]["on_chip"]) == 1
        if is_on_chip:
            if getattr(tensor, "is_constant", False):
                # On-chip constants live in the params area, not match_mem.
                tensor.stored_in_external_memory = False
                tensor.mem_offset = -1
                continue
            offset = solver.Value(tensor_vars[tensor]["start"])
            tensor.mem_offset = offset
            tensor.stored_in_external_memory = False

            full_lifetime = (
                getattr(tensor, "is_input", False)
                or getattr(tensor, "is_output", False)
                or getattr(tensor, "is_constant", False)
            )
            start = 0 if full_lifetime else tensor.start_usage
            end = planner.last_timestep if full_lifetime else tensor.last_usage
            tensor.start_usage, tensor.last_usage = start, end
            for c in calls:
                if start <= c <= end:
                    tensor.mem_offset_at[c] = offset
            continue

        tensor.stored_in_external_memory = True
        tensor.mem_offset = solver.Value(tensor_vars[tensor]["start"])

        if getattr(tensor, "is_output", False):
            stage_offset = None
            if tensor in output_stage_vars:
                stage_offset = solver.Value(output_stage_vars[tensor]["start"])
                produce_time = output_stage_vars[tensor]["produce_time"]
                tensor.mem_offset_at[produce_time] = stage_offset
                tensor.mem_offset = stage_offset

            uses = sorted(set(getattr(tensor, "used_at", []) or []))
            if len(uses) == 0:
                uses = [tensor.node_id]
            for use_time in uses:
                if use_time in call_to_pos:
                    tensor.mem_offset_at[use_time] = stage_offset if stage_offset is not None else tensor.mem_offset
            last_usage = uses[-1]
            move_time = -1
            for c in calls:
                if c > last_usage:
                    move_time = c
                    break
            tensor.move_temp_to_ext_mem.append(move_time)
            continue

        uses = sorted(set(getattr(tensor, "used_at", []) or []))
        if len(uses) == 0:
            uses = [tensor.node_id]
        if getattr(tensor, "is_input", False) or getattr(tensor, "is_constant", False):
            uses = [uses[0]]

        for event_idx, use_time in enumerate([u for u in uses if u in call_to_pos]):
            key = (tensor, use_time, event_idx)
            if key not in event_vars:
                continue
            ev = event_vars[key]

            chosen_pos = None
            for p, b in ev["choices"]:
                if solver.Value(b) == 1:
                    chosen_pos = p
                    break
            if chosen_pos is None:
                continue

            load_time = calls[chosen_pos]
            use_pos = ev["use_pos"]
            use_time = ev["use_time"]
            event_offset = solver.Value(ev["start"])

            tensor.load_from_ext_mem_at.append(load_time)
            for pos in range(chosen_pos, use_pos + 1):
                tensor.mem_offset_at[calls[pos]] = event_offset

            if async_dma:
                tensor.wait_async_off_chip_transfer_at.append(use_time)

            if not (getattr(tensor, "is_input", False) or getattr(tensor, "is_constant", False)):
                if async_dma and key in writeback_event_vars:
                    wb = writeback_event_vars[key]
                    chosen_move_pos = None
                    for p, b in wb["choices"]:
                        if solver.Value(b) == 1:
                            chosen_move_pos = p
                            break

                    if chosen_move_pos is None:
                        tensor.move_temp_to_ext_mem.append(-1)
                    else:
                        move_time = calls[chosen_move_pos]
                        tensor.move_temp_to_ext_mem.append(move_time)
                        for pos in range(use_pos + 1, chosen_move_pos + 1):
                            tensor.mem_offset_at.setdefault(calls[pos], event_offset)
                else:
                    next_call = -1
                    for c in calls:
                        if c > use_time:
                            next_call = c
                            break
                    tensor.move_temp_to_ext_mem.append(next_call)

        if tensor.wait_async_off_chip_transfer_at:
            tensor.wait_async_off_chip_transfer_at = sorted(set(tensor.wait_async_off_chip_transfer_at))

        if tensor.mem_offset_at:
            first_time = min(tensor.mem_offset_at.keys())
            tensor.mem_offset = tensor.mem_offset_at[first_time]

    # Safety repair: ensure off-chip outputs never overlap at emitted timesteps.
    # This is a conservative post-solve guard for corner cases in graph/output
    # aliasing flows; it only adjusts conflicting output offsets.
    def _find_first_fit_offset(intervals, size: int, alignment: int) -> int:
        align = max(1, alignment)
        current = 0
        for start, end in sorted(intervals, key=lambda x: x[0]):
            if current % align != 0:
                current += align - (current % align)
            if start - current >= size:
                return current
            if end > current:
                current = end
        if current % align != 0:
            current += align - (current % align)
        if current + size <= max_mem:
            return current
        return -1

    for tensor in sorted_mem_tensors:
        if not (getattr(tensor, "is_output", False) and getattr(tensor, "stored_in_external_memory", False)):
            continue
        if not tensor.mem_offset_at:
            continue

        alignment = _alignment_of(tensor)
        for t in sorted(tensor.mem_offset_at.keys()):
            cur = tensor.mem_offset_at[t]
            cur_end = cur + tensor.num_bytes

            others = []
            for other in sorted_mem_tensors:
                if other is tensor:
                    continue
                if t not in getattr(other, "mem_offset_at", {}):
                    continue
                o_start = other.mem_offset_at[t]
                o_end = o_start + other.num_bytes
                others.append((o_start, o_end))

            overlaps = any(not (cur_end <= o_start or o_end <= cur) for o_start, o_end in others)
            if not overlaps:
                continue

            repaired = _find_first_fit_offset(others, tensor.num_bytes, alignment)
            if repaired < 0:
                raise Exception(
                    f"[MEM PLANNER] Could not repair overlap for output tensor {tensor.name} at time {t}."
                )
            tensor.mem_offset_at[t] = repaired

        first_time = min(tensor.mem_offset_at.keys())
        tensor.mem_offset = tensor.mem_offset_at[first_time]

    tensors_allocated_at_time = {key: [] for key in calls}
    for time_idx in calls:
        active_tensors = [
            t
            for t in sorted_mem_tensors
            if (
                (
                    (getattr(t, "is_input", False) or getattr(t, "is_output", False) or getattr(t, "is_constant", False))
                    or (t.start_usage <= time_idx <= t.last_usage)
                    or (time_idx in t.mem_offset_at)
                )
                and (time_idx in t.mem_offset_at)
            )
        ]
        active_tensors.sort(key=lambda tensor: tensor.mem_offset_at.get(time_idx, tensor.mem_offset))
        tensors_allocated_at_time[time_idx] = active_tensors

    __metadata_output_file__ = f"{planner.out_path}/memory_plan_cp_metadata.json"
    save_memory_allocation_graph(
        sorted_mem_tensors,
        graph_output_file=f"{planner.out_path}/memory_plan_cp.png",
        metadata_output_file=__metadata_output_file__,
    )
    __metadata__ = dict()
    wait_points = []
    if os.path.exists(__metadata_output_file__):
        with open(__metadata_output_file__) as f:
            __metadata__ = json.load(f)
        wait_points = compute_async_waits(__metadata__, verbose=False)
    wait_points_output = [
        {
            "node_id":                wp.node_id,
            "async_wait_transfer_id": wp.transfer_id,
            "kind":                   wp.kind.name,
            "tracking_tensor":        wp.tracking_tensor,
            "covered_transfer_ids":   wp.covered_transfers,
        }
        for wp in wait_points
    ]
    wait_points_output_file = f"{planner.out_path}/memory_plan_cp_async_wait_points.json"
    with open(wait_points_output_file, "w") as f:
        json.dump(wait_points_output, f, indent=2)
    print(f"[MEM PLAN] Schedule written to {wait_points_output_file}")
    
    nodes_buffers = save_memory_allocation_graph_nodes_buffers(
        mem_tensors_at=tensors_allocated_at_time,
        calls_idxs=calls,
        match_mem_size=actual_peak_mem,
        output_file=f"{planner.out_path}/memory_plan_cp_node_buffers.json",
    )
    for node in planner.nodes:
        if node.node_id in nodes_buffers:
            node.free_buffers = nodes_buffers[node.node_id]["empty_areas"]

    return actual_peak_mem, ext_mem_needed, wait_points
