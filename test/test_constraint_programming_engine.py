import numpy as np

from match.dim.dim import MatchDim
from match.node.node import MatchNode
from match.opt.cp import ConstraintProgrammingEngine
from match.opt.easy_tile import EasyTileEngine
from match.opt.generator import get_schedule_engine
from match.target.memory_inst import MemoryInst
from match.tensor.tensor import MatchTensor


class _MockTarget:
    def __init__(self, memories):
        self._memories = memories

    def memory_hierarchy_for_pt(self, exec_module=None, pattern_name=""):
        host = MemoryInst(name="HOST_MEM", k_bytes=1024, tensor_types=["var", "const", "output", "intermediate"])
        return {
            "var": [self._memories[0], host],
            "const": [self._memories[0], host],
            "output": [self._memories[0], host],
            "intermediate": [self._memories[0], host],
        }


class _MockExecModule:
    pass


def _build_node():
    node = MatchNode()
    h = MatchDim(name="H", size=8)
    w = MatchDim(name="W", size=8)
    node.dims = {"H": h, "W": w}
    node.var_tensors = {
        "inp": MatchTensor(name="inp", dims=[h, w], dtype=np.dtype("int8"), tensor_type="var")
    }
    node.output_tensors = {
        "out": MatchTensor(name="out", dims=[h, w], dtype=np.dtype("int8"), tensor_type="output")
    }
    return node


def test_schedule_engine_registration_for_constraint_programming():
    assert get_schedule_engine("ConstraintProgramming") is ConstraintProgrammingEngine
    assert get_schedule_engine("CP") is ConstraintProgrammingEngine


def test_easy_tile_respects_double_buffering_capacity():
    node = _build_node()
    l1 = MemoryInst(name="L1", k_bytes=0.2, double_buffering_support=True, tensor_types=["var", "const", "output", "intermediate"])
    target = _MockTarget([l1])
    engine = EasyTileEngine(target=target, exec_module=_MockExecModule(), pattern_name="test", match_node=node)

    engine.generate_schedule()
    schedule = engine.get_schedule()

    assert schedule.blocks[0].num_buffers_for_computation == 2
    # 8x8 var + 8x8 output with double buffering does not fit in ~0.2KB -> at least one tiled loop is required.
    assert any(lp.name.startswith("loop_") for lp in schedule.blocks[0].loops)


def test_cp_engine_generates_double_buffered_schedule():
    node = _build_node()
    l1 = MemoryInst(name="L1", k_bytes=0.2, double_buffering_support=True, tensor_types=["var", "const", "output", "intermediate"])
    target = _MockTarget([l1])
    engine = ConstraintProgrammingEngine(target=target, exec_module=_MockExecModule(), pattern_name="test", match_node=node)

    engine.generate_schedule()
    schedule = engine.get_schedule()

    assert schedule.blocks[0].num_buffers_for_computation == 2
    assert schedule.blocks[0].backend == "ConstraintProgrammingEngine"
    assert any(lp.name.startswith("loop_") for lp in schedule.blocks[0].loops)
