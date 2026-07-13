from math import prod
from typing import Dict, List
import sys

from match.dim.dim import MatchTiledDim
from match.node.node import MatchNode
from match.opt.engine import ScheduleEngine
from match.schedule.block import MatchBlock
from match.schedule.loop import MatchLoop
from match.schedule.mem_transfer import MatchMemTransfer
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ExecModule
from match.target.memory_inst import MemoryInst
from match.tensor.tensor import MatchTensorTile


def _divisors_desc(n: int) -> List[int]:
    if n <= 0:
        return [1]
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs, reverse=True)


class ConstraintProgrammingEngine(ScheduleEngine):
    """Constraint-programming style scheduler based on finite-domain search.

    The solver searches tile sizes over the divisor domain of each independent
    dimension, enforcing memory-capacity constraints and maximizing tile volume.
    """

    def __init__(
        self,
        target=None,
        exec_module: ExecModule = None,
        pattern_name: str = "",
        match_node: MatchNode = None,
    ):
        super(ConstraintProgrammingEngine, self).__init__(
            target=target,
            exec_module=exec_module,
            pattern_name=pattern_name,
            match_node=match_node,
        )
        self.mem_hierarchy = self.target.memory_hierarchy_for_pt(
            exec_module=self.exec_module,
            pattern_name=self.pattern_name,
        )
        self.mem_hierarchy_dict = {
            mem.name: mem
            for mem in set([mem_ for _, v in self.mem_hierarchy.items() for mem_ in v])
        }
        self._buffer_bytes_cache = {}

    def transform_schedule_for_engine(self):
        pass

    def _get_num_buffers_for_computation(self, tensors) -> int:
        # Enable double buffering if any compute-relevant lowest-level memory supports it.
        for tensor in tensors.values():
            if tensor.tensor_type not in ["var", "output", "intermediate"]:
                continue
            memories = self.mem_hierarchy[tensor.tensor_type]
            if memories and memories[0].double_buffering_support:
                return 2
        return 1

    def _dependent_dim_size(self, tensor, dim_idx: int, dims_sizes: Dict[str, int]) -> int:
        size = 0
        for (ind_dim, mult) in tensor.dims[dim_idx].dim_dependency.size_dependencies:
            size += mult * (ind_dim if not hasattr(ind_dim, "name") else dims_sizes[ind_dim.name])
        return int(size)

    def _tensor_tile_size_bytes(self, tensor, dims_sizes: Dict[str, int]) -> int:
        return int(
            prod(
                [
                    self._dependent_dim_size(tensor, dim_idx, dims_sizes)
                    if dim.dim_dependency
                    else dims_sizes[dim.name]
                    for dim_idx, dim in enumerate(tensor.dims)
                ]
                + [tensor.bits // 8]
            )
        )

    def _is_feasible(self, tensors, dims_sizes: Dict[str, int], num_buffers_for_computation: int) -> bool:
        per_mem_usage = dict()
        per_mem_capacity = dict()

        # Capacities from tensor memory hierarchy
        for mems in self.mem_hierarchy.values():
            for mem in mems:
                per_mem_usage.setdefault(mem.name, 0)
                per_mem_capacity.setdefault(mem.name, mem.k_bytes * 1024)

        # Capacities from exec-module declared memories (for explicit buffer mem_name accounting)
        for mem in self.exec_module.module_memories():
            per_mem_usage.setdefault(mem.name, 0)
            per_mem_capacity.setdefault(mem.name, mem.k_bytes * 1024)

        for tensor in tensors.values():
            if len(tensor.dims) == 0:
                continue
            mem: MemoryInst = self.mem_hierarchy[tensor.tensor_type][0]
            tile_size = self._tensor_tile_size_bytes(tensor, dims_sizes)
            multiplier = 1
            if (
                num_buffers_for_computation > 1
                and mem.double_buffering_support
                and tensor.tensor_type in ["var", "output", "intermediate"]
            ):
                # A tensor is tiled if any of its dependent dims are tiled
                tensor_deps = set()
                for dim in tensor.dims:
                    if getattr(dim, "dim_dependency", None):
                        for (ind_dim, _) in dim.dim_dependency.size_dependencies:
                            if hasattr(ind_dim, "name"):
                                tensor_deps.add(ind_dim.name)
                    else:
                        if hasattr(dim, "name"):
                            tensor_deps.add(dim.name)
                
                tensor_is_tiled = False
                for dep in tensor_deps:
                    # If this dimension exists in dims_sizes and is smaller than its max_size, it's tiled.
                    # We can find max_size from the dimensions of the node.
                    node_dim = self.match_node.dims.get(dep)
                    if node_dim and dep in dims_sizes and dims_sizes[dep] != node_dim.size:
                        tensor_is_tiled = True
                        break
                        
                if tensor_is_tiled:
                    multiplier = num_buffers_for_computation
            per_mem_usage[mem.name] += tile_size * multiplier
            if per_mem_usage[mem.name] > per_mem_capacity[mem.name]:
                return False

        # Extra constraint: all generated buffers must fit together with tiled tensors.
        buffers_bytes_per_mem = self._estimate_buffers_per_memory(tensors, dims_sizes)
        for mem_name, buffer_bytes in buffers_bytes_per_mem.items():
            if mem_name not in per_mem_capacity:
                # Unknown memory for this engine: don't fail hard here, as the target may
                # expose only a subset of memories to this scheduler path.
                continue
            per_mem_usage[mem_name] += buffer_bytes
            if per_mem_usage[mem_name] > per_mem_capacity[mem_name]:
                return False

        return True

    def _build_tensor_tiles(self, tensors, dims_sizes: Dict[str, int]) -> Dict[str, List[MatchTensorTile]]:
        tensor_tiles = dict()
        for tensor_name, tensor in tensors.items():
            memories = self.mem_hierarchy[tensor.tensor_type]
            tensor_tiles[tensor_name] = [
                MatchTensorTile(
                    tensor=tensor,
                    tiled_dims=[
                        MatchTiledDim(dim=dim, size=dim.size, max_size=dim.max_size)
                        for dim in tensor.dims
                    ],
                )
                for _ in memories
            ]

            for dim_idx in range(len(tensor.dims)):
                if tensor.dims[dim_idx].dim_dependency:
                    new_size = self._dependent_dim_size(tensor, dim_idx, dims_sizes)
                    tensor_tiles[tensor_name][-1].tiled_dims[dim_idx].max_size = new_size
                    if new_size > tensor.dims[dim_idx].size:
                        new_size = tensor.dims[dim_idx].size
                    tensor_tiles[tensor_name][-1].tiled_dims[dim_idx].size = new_size
                else:
                    tensor_tiles[tensor_name][-1].tiled_dims[dim_idx].max_size = dims_sizes[
                        tensor.dims[dim_idx].name
                    ]
                    tensor_tiles[tensor_name][-1].tiled_dims[dim_idx].size = dims_sizes[
                        tensor.dims[dim_idx].name
                    ]

        return tensor_tiles

    def _estimate_buffers_per_memory(self, tensors, dims_sizes: Dict[str, int]) -> Dict[str, int]:
        cache_key = tuple(sorted(dims_sizes.items()))
        if cache_key in self._buffer_bytes_cache:
            return self._buffer_bytes_cache[cache_key]

        tmp_schedule = MatchSchedule(
            blocks=[],
            tensors=tensors,
            tensor_tiles=self._build_tensor_tiles(tensors, dims_sizes),
            buffers=[],
            init_instrs=[],
            instrs=[],
            exec_module=self.exec_module,
        )

        tmp_schedule.buffers = []
        self.exec_module.set_buffers_for_schedule(
            match_node=self.match_node,
            schedule=tmp_schedule,
            pattern_name=self.pattern_name,
            engine="CP",
        )

        buffers_bytes_per_mem = dict()
        for buf in tmp_schedule.buffers:
            buffers_bytes_per_mem.setdefault(buf.mem_name, 0)
            buffers_bytes_per_mem[buf.mem_name] += buf.num_bytes

        self._buffer_bytes_cache[cache_key] = buffers_bytes_per_mem
        return buffers_bytes_per_mem

    def _inner_loop_mem_usage(self, tensors, dims_sizes: Dict[str, int], num_buffers_for_computation: int) -> Dict[str, int]:
        per_mem_usage = {}
        for mems in self.mem_hierarchy.values():
            for mem in mems:
                per_mem_usage.setdefault(mem.name, 0)

        for tensor in tensors.values():
            if len(tensor.dims) == 0:
                continue
            mem: MemoryInst = self.mem_hierarchy[tensor.tensor_type][0]
            tile_size = self._tensor_tile_size_bytes(tensor, dims_sizes)
            multiplier = 1
            if (
                num_buffers_for_computation > 1
                and mem.double_buffering_support
                and tensor.tensor_type in ["var", "output", "intermediate"]
            ):
                tensor_deps = set()
                for dim in tensor.dims:
                    if getattr(dim, "dim_dependency", None):
                        for (ind_dim, _) in dim.dim_dependency.size_dependencies:
                            if hasattr(ind_dim, "name"):
                                tensor_deps.add(ind_dim.name)
                    else:
                        if hasattr(dim, "name"):
                            tensor_deps.add(dim.name)
                tensor_is_tiled = False
                for dep in tensor_deps:
                    node_dim = self.match_node.dims.get(dep)
                    if node_dim and dep in dims_sizes and dims_sizes[dep] != node_dim.size:
                        tensor_is_tiled = True
                        break
                if tensor_is_tiled:
                    multiplier = num_buffers_for_computation
            per_mem_usage[mem.name] += tile_size * multiplier

        buffers_bytes_per_mem = self._estimate_buffers_per_memory(tensors, dims_sizes)
        for mem_name, buffer_bytes in buffers_bytes_per_mem.items():
            per_mem_usage.setdefault(mem_name, 0)
            per_mem_usage[mem_name] += buffer_bytes

        return per_mem_usage

    def _fixed_no_tile_dim_names(self, dims) -> set:
        """Identify dimensions that must not be tiled.

        We lock:
        - input channels dimension of Conv2D input activation
        - input channels dimension of Conv3D input activation
        - kernel height/width dimensions of Conv2D weights

        Detection is semantic (op/tensor/layout based), not name based, so it works
        for generated names like `input_0_dim_3`, `const_0_dim_0`, etc.
        """
        fixed_dim_names = set()

        # Prefer semantic extraction from parsed Conv2D/Conv3D ops.
        conv_ops = [
            op
            for op in self.match_node.ops.values()
            if getattr(op, "op", "") in {"Conv2D", "Conv3D"}
        ]

        for conv in conv_ops:
            op_name = str(getattr(conv, "op", ""))
            data_layout = str(getattr(conv, "data_layout", "NCHW"))
            kernel_layout = str(getattr(conv, "kernel_layout", "OIHW"))

            # Input channels dim from activation tensor.
            if getattr(conv, "vars", None):
                inp_tensor = conv.vars[0]
                if op_name == "Conv3D":
                    inp_c_idx = 1 if data_layout == "NCDHW" else 4
                else:
                    inp_c_idx = 1 if data_layout == "NCHW" else 3
                if inp_c_idx < len(inp_tensor.dims):
                    fixed_dim_names.add(inp_tensor.dims[inp_c_idx].name)

            # Kernel H/W dims from weight tensor.
            if getattr(conv, "consts", None):
                w_tensor = conv.consts[0]
                if kernel_layout == "OIHW":
                    kh_idx, kw_idx = 2, 3
                elif kernel_layout == "HWIO":
                    kh_idx, kw_idx = 0, 1
                elif kernel_layout == "OHWI":
                    kh_idx, kw_idx = 1, 2
                else:
                    kh_idx, kw_idx = None, None

                if kh_idx is not None and kw_idx is not None:
                    if kh_idx < len(w_tensor.dims):
                        fixed_dim_names.add(w_tensor.dims[kh_idx].name)
                    if kw_idx < len(w_tensor.dims):
                        fixed_dim_names.add(w_tensor.dims[kw_idx].name)

                # Extra guard: infer by kernel_size values if layout mapping is unknown.
                if (kh_idx is None or kw_idx is None) and hasattr(conv, "kernel_size") and len(conv.kernel_size) == 2:
                    k_h, k_w = int(conv.kernel_size[0]), int(conv.kernel_size[1])
                    for dim in w_tensor.dims:
                        if dim.size == k_h or dim.size == k_w:
                            fixed_dim_names.add(dim.name)

        # Conservative fallback for non-conv graphs or missing metadata.
        if not fixed_dim_names:
            exact_tokens = {
                "c", "ci", "ic", "in_c", "in_channels", "input_channels",
                "kh", "kw", "kernel_h", "kernel_w", "kernel_height", "kernel_width",
            }
            contains_tokens = ("input_channel", "in_channel", "kernel_h", "kernel_w", "kh", "kw")
            for dim in dims:
                normalized = str(dim.name).lower()
                if normalized in exact_tokens or any(token in normalized for token in contains_tokens):
                    fixed_dim_names.add(dim.name)

        # Keep only currently-optimizable independent dims.
        dim_names = {d.name for d in dims}
        return fixed_dim_names & dim_names

    def _search_best_dims_cp(self, dims, tensors, num_buffers_for_computation: int) -> Dict[str, int]:
        # CP model via python-constraint.
        from constraint import Problem  # type: ignore[import-not-found]

        domains = {dim.name: _divisors_desc(dim.size) for dim in dims}
        fixed_dims = self._fixed_no_tile_dim_names(dims)
        for dim in dims:
            if dim.name in fixed_dims:
                domains[dim.name] = [dim.size]

        dim_names = [dim.name for dim in dims]

        problem = Problem()
        for name in dim_names:
            problem.addVariable(name, domains[name])

        def memory_feasible(*values):
            assignment = {name: values[idx] for idx, name in enumerate(dim_names)}
            return self._is_feasible(tensors, assignment, num_buffers_for_computation)

        problem.addConstraint(memory_feasible, dim_names)

        total_candidates = prod([len(domains[name]) for name in dim_names]) if dim_names else 0
        visited_candidates = 0
        best_solution = None
        best_score = (-1, -1)

        def update_progress_bar():
            if total_candidates <= 0:
                return
            progress = min(1.0, visited_candidates / total_candidates)
            bar_len = 30
            filled_len = int(bar_len * progress)
            bar = "█" * filled_len + "-" * (bar_len - filled_len)
            msg = f"\r[CP TILE] Constraint search |{bar}| {visited_candidates}/{total_candidates}"
            sys.stdout.write(msg)
            sys.stdout.flush()

        for sol in problem.getSolutionIter():
            visited_candidates += 1
            if visited_candidates % 128 == 0:
                update_progress_bar()

            score = (prod([sol[d.name] for d in dims]), sum(sol[d.name] for d in dims))
            if score > best_score:
                best_score = score
                best_solution = sol

        if total_candidates > 0:
            if visited_candidates < total_candidates:
                visited_candidates = total_candidates
            update_progress_bar()
            sys.stdout.write("\n")
            sys.stdout.flush()

        if not best_solution:
            return None

        return best_solution

    def generate_schedule(self):
        self._buffer_bytes_cache = {}
        tensors = {
            tens_name: tens
            for tens_name, tens in self.match_node.tensors.items()
            if tens.tensor_type != "intermediate"
        }
        dims = self.match_node.independent_dims
        original_dim_sizes = {dim.name: dim.size for dim in dims}

        num_buffers_for_computation = self._get_num_buffers_for_computation(tensors)
        dims_sizes = self._search_best_dims_cp(dims, tensors, num_buffers_for_computation)
        if dims_sizes is None:
            raise Exception("[ConstraintProgrammingEngine] No valid schedule found")

        loops = []

        tiled_dim_names = set()
        for dim in dims:
            dim_name = dim.name
            dim_size = dims_sizes[dim_name]
            if dim_size != original_dim_sizes[dim_name]:
                tiled_dim_names.add(dim_name)
                loops.append(
                    MatchLoop(
                        name=f"loop_{dim_name}",
                        dim=dim,
                        size=original_dim_sizes[dim_name] // dim_size,
                        step=dim_size,
                        mem_transfers=[],
                        init_instrs=[],
                        instrs=[],
                    )
                )

        inner_loop = MatchLoop(
            name="nop_loop",
            dim=self.match_node.default_dim,
            size=1,
            step=1,
            mem_transfers=[],
            init_instrs=[],
            instrs=[],
        )
        loops.append(inner_loop)

        tensor_tiles = self._build_tensor_tiles(tensors, dims_sizes)
        for tensor in tensors.values():
            memories = self.mem_hierarchy[tensor.tensor_type]
            if len(tensor.dims) == 0 or len(memories) <= 1:
                continue

            tensor_deps = set()
            for dim in tensor.dims:
                if getattr(dim, "dim_dependency", None):
                    for (ind_dim, _) in dim.dim_dependency.size_dependencies:
                        if hasattr(ind_dim, "name"):
                            tensor_deps.add(ind_dim.name)
                else:
                    if hasattr(dim, "name"):
                        tensor_deps.add(dim.name)

            tensor_is_tiled = bool(tensor_deps.intersection(tiled_dim_names))
            num_buffers = num_buffers_for_computation if tensor_is_tiled else 1

            inner_loop.mem_transfers.append(
                MatchMemTransfer(
                    tensor=tensor,
                    top_mem=memories[-1].name,
                    mem=memories[0].name,
                    sw_controlled=memories[0].sw_controlled,
                    num_buffers=num_buffers,
                )
            )

        block = MatchBlock(
            loops=loops,
            backend="ConstraintProgrammingEngine",
            init_instrs=[],
            instrs=[],
            parallel_execution=False,
            num_tasks=1,
            num_buffers_for_computation=num_buffers_for_computation,
        )
        self.schedule = MatchSchedule(
            blocks=[block],
            tensors=tensors,
            tensor_tiles=tensor_tiles,
            buffers=[],
            init_instrs=[],
            instrs=[],
            exec_module=self.exec_module,
        )
        self.schedule.buffers = []

        inner_mem_usage = self._inner_loop_mem_usage(tensors, dims_sizes, num_buffers_for_computation)
        if inner_mem_usage:
            print(f"[CP TILE] Inner-loop memory usage (bytes): {inner_mem_usage}")

        # Final guarantee: buffers are computed from the selected tile sizes.
        # This keeps emitted schedule buffers aligned with what feasibility checked.
        self.exec_module.set_buffers_for_schedule(
            match_node=self.match_node,
            schedule=self.schedule,
            pattern_name=self.pattern_name,
            engine="CP",
        )
        buffer_bytes = sum([buff.num_bytes for buff in self.schedule.buffers])
        tiles_bytes = sum([prod([tile.size for tile in val[1].tiled_dims])*2 for val in tensor_tiles.values()])
        print("Buffer bytes", buffer_bytes, "tiles bytes", tiles_bytes, "sum", buffer_bytes+tiles_bytes)

    def transform_schedule(self):
        pass