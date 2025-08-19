from copy import deepcopy
from math import prod
from match.dim.dim import MatchTiledDim
from match.node.node import MatchNode
from match.opt.engine import ScheduleEngine
from match.schedule.block import MatchBlock
from match.schedule.loop import MatchLoop
from match.schedule.mem_transfer import MatchMemTransfer
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ExecModule
from match.tensor.tensor import MatchTensorTile

def smallest_factor(n):
    """
    Finds the smallest factor (greater than 1) of a given number.

    Args:
        n: The input number (an integer).

    Returns:
        The smallest factor of n, or n itself if it's a prime number.
    """
    if n <= 1:
        return n  # Or raise an exception, as 1 and below don't have factors > 1

    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i
    return n  # If no smaller factor is found, the number is prime

class EasyTileEngine(ScheduleEngine):
    """
    EasyTileEngine is a ScheduleEngine that generates a schedule for tiling tensors
    based on the memory hierarchy of the target and the dimensions of the tensors.
    It creates loops for each independent dimension and tiles tensors according to
    the available memory.
    It doesn't consider yet additional extra buffers, also its reference architecture
    is one where each operand has a 2-level memory hierarchy.
    It tries to fit the tensors by reducing iteratively the size of the biggest
    independent dimension until it fits in the lowest memory of the hierarchy.
    If it cannot fit, it raises an exception.
    Additionally, it expects the node to be fused, i.e., it does not
    support multiple blocks
    
    Args:
        target: The target for which the schedule is generated.
        exec_module: The execution module to be used.
        pattern_name: The name of the pattern being matched.
        match_node: The MatchNode containing the tensors and dimensions to be tiled.
    """
    def __init__(
        self, target=None, exec_module: ExecModule=None,
        pattern_name: str="", match_node: MatchNode=None
    ):
        super(EasyTileEngine, self).__init__(
            target=target, exec_module=exec_module,
            pattern_name=pattern_name, match_node=match_node
        )
        self.mem_hierarchy = self.target.memory_hierarchy_for_pt(exec_module=self.exec_module, pattern_name=self.pattern_name)
        self.mem_hierarchy_dict = {mem.name:mem for mem in set([mem_ for k,v in self.mem_hierarchy.items() for mem_ in v])}
    
    def transform_schedule_for_engine(self):
        pass

    def generate_schedule(self):
        tensors = {tens_name:tens for tens_name,tens in self.match_node.tensors.items() if tens.tensor_type!="intermediate"}
        loops = list()
        inner_loop = MatchLoop(
            name = "nop_loop",
            dim=self.match_node.default_dim,
            size=1,
            step=1,
            mem_transfers=[],
            init_instrs=[],
            instrs=[],
        )
        dims = self.match_node.independent_dims
        original_dim_sizes = {dim.name:dim.size for dim in dims}
        dims_sizes = deepcopy(original_dim_sizes)
        original_memories_sizes = dict()

        def get_dependent_dim_size(tensor, dim_idx):
            size = 0
            for (ind_dim,mult) in tensor.dims[dim_idx].dim_dependency.size_dependencies:
                size += (mult*(ind_dim if not hasattr(ind_dim,"name") else dims_sizes[ind_dim.name]))
            return int(size)
        
        for mems in self.mem_hierarchy.values():
            for mem in mems:
                if mem.name not in original_memories_sizes:
                    original_memories_sizes[mem.name] = mem.k_bytes * 1024  # Convert kB to bytes
        memories_sizes = deepcopy(original_memories_sizes)
        fit_lowest_memory = False
        while not fit_lowest_memory:
            fit_lowest_memory = True
            for tensor_name, tensor in tensors.items():
                if len(tensor.dims)>0:
                    memories = self.mem_hierarchy[tensor.tensor_type]
                    tile_size = prod(
                        [get_dependent_dim_size(tensor, dim_idx) if dim.dim_dependency else dims_sizes[dim.name] for dim_idx,dim in enumerate(tensor.dims)]
                        + [tensor.bits//8]  # Convert bits to bytes
                    )
                    memories_sizes[memories[0].name] -= tile_size
                    if memories_sizes[memories[0].name] < 0:
                        fit_lowest_memory = False
                        break
            if not fit_lowest_memory:
                for mem_name in memories_sizes.keys():
                    memories_sizes[mem_name] = original_memories_sizes[mem_name]
                if all([dims_sizes[dim.name] == 1 for dim in dims]):
                    break
                max_dim_ = max([(dim_size, dim_name) for dim_name, dim_size in dims_sizes.items()], key=lambda x: x[0])
                dims_sizes[max_dim_[1]] = int(max_dim_[0]//smallest_factor(max_dim_[0]))
    
        if not fit_lowest_memory:
            raise Exception(f"[EasyTileEngine] No valid schedule found")
        for dim_name, dim_size in dims_sizes.items():
            if dim_size != original_dim_sizes[dim_name]:
                loops.append(
                    MatchLoop(
                        name = f"loop_{dim_name}",
                        dim = self.match_node.dims[dim_name],
                        size = original_dim_sizes[dim_name] // dim_size,
                        step = dim_size,
                        mem_transfers = [],
                        init_instrs = [],
                        instrs = [],
                    )
                )
        loops.append(inner_loop)
            
        tensor_tiles = dict()
        for tensor_name, tensor in tensors.items():
            if len(tensor.dims)>0:
                memories = self.mem_hierarchy[tensor.tensor_type]
                if len(memories)>1:
                    sw_controlled = memories[0].sw_controlled
                    inner_loop.mem_transfers.append(
                        MatchMemTransfer(
                            tensor= tensor,
                            top_mem = memories[-1].name,
                            mem = memories[0].name,
                            sw_controlled = sw_controlled,
                        )
                    )
            tensor_tiles[tensor_name] = [
                MatchTensorTile(
                    tensor=tensor,
                    tiled_dims=[
                        MatchTiledDim(
                            dim=dim,
                            size=dim.size,
                            max_size=dim.max_size
                        ) for dim in tensor.dims
                    ]
                ) for _ in memories
            ]
            for dim_idx in range(len(tensor.dims)):
                if tensor.dims[dim_idx].dim_dependency:
                    new_size = get_dependent_dim_size(tensor, dim_idx)
                    tensor_tiles[tensor_name][-1].tiled_dims[dim_idx].max_size = new_size
                    if new_size>tensor.dims[dim_idx].size:
                        new_size = tensor.dims[dim_idx].size
                    tensor_tiles[tensor_name][-1].tiled_dims[dim_idx].size = new_size
                else:
                    tensor_tiles[tensor_name][-1].tiled_dims[dim_idx].max_size = dims_sizes[tensor.dims[dim_idx].name]
                    tensor_tiles[tensor_name][-1].tiled_dims[dim_idx].size = dims_sizes[tensor.dims[dim_idx].name]
        block = MatchBlock(
            loops = loops,
            backend = "EasyTileEngine",
            init_instrs = [],
            instrs = [],
            parallel_execution = False,
            num_tasks = 1,
            num_buffers_for_computation = 1,
        )
        self.schedule = MatchSchedule(
            blocks = [block],
            tensors = tensors,
            tensor_tiles = tensor_tiles,
            buffers = [],
            init_instrs=[],
            instrs=[],
            exec_module=self.exec_module,
        )
    
    def transform_schedule(self):
        pass
