import numpy as np
import pytest

from match.runtime.graph.tensor import MatchMemoryTensor
from match.runtime.graph import memplan_cp


pytest.importorskip("ortools")


class _Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.free_buffers = []


class _Target:
    def __init__(self, async_off_chip_to_on_chip_dm: bool):
        self.async_off_chip_to_on_chip_dm = async_off_chip_to_on_chip_dm


class _Planner:
    def __init__(self, tensor, async_dma: bool, out_path: str, calls_idxs=None):
        calls_idxs = calls_idxs or [0, 1, 2]
        self.mem_tensors = [tensor]
        self.extra_dynamic_buffers = []
        self.available_soc_bytes = 16
        self.calls_idxs = calls_idxs
        self.last_timestep = max(self.calls_idxs)
        self.nodes = [_Node(node_id) for node_id in self.calls_idxs]
        self.out_path = out_path
        self.target = _Target(async_dma)


@pytest.fixture
def offchip_intermediate_tensor():
    t = MatchMemoryTensor(
        name="tmp",
        is_intermediate=True,
        shape=(4,),
        dtype=np.dtype("uint8"),
        node_id=2,
    )
    t.update_last_usage(2)
    return t


@pytest.fixture
def offchip_early_use_tensor():
    t = MatchMemoryTensor(
        name="tmp_early",
        is_intermediate=True,
        shape=(4,),
        dtype=np.dtype("uint8"),
        node_id=1,
    )
    t.update_last_usage(1)
    return t


def _stub_graph(*args, **kwargs):
    return None


def _stub_nodes_buffers(mem_tensors_at=None, calls_idxs=None, **kwargs):
    calls_idxs = calls_idxs or []
    return {c: {"empty_areas": []} for c in calls_idxs}


def test_cp_memplan_sync_load_happens_at_use(monkeypatch, tmp_path, offchip_intermediate_tensor):
    monkeypatch.setattr(memplan_cp, "save_memory_allocation_graph", _stub_graph)
    monkeypatch.setattr(memplan_cp, "save_memory_allocation_graph_nodes_buffers", _stub_nodes_buffers)

    planner = _Planner(offchip_intermediate_tensor, async_dma=False, out_path=str(tmp_path))
    _, ext_mem, _ = memplan_cp.cp_mem_planner_impl(planner, tensor_fixed_to_ext_mem=["tmp"])

    assert ext_mem == offchip_intermediate_tensor.num_bytes
    assert offchip_intermediate_tensor.load_from_ext_mem_at == [2]
    assert offchip_intermediate_tensor.move_temp_to_ext_mem == [-1]
    assert 2 in offchip_intermediate_tensor.mem_offset_at


def test_cp_memplan_async_can_prefetch_earlier(monkeypatch, tmp_path, offchip_intermediate_tensor):
    monkeypatch.setattr(memplan_cp, "save_memory_allocation_graph", _stub_graph)
    monkeypatch.setattr(memplan_cp, "save_memory_allocation_graph_nodes_buffers", _stub_nodes_buffers)

    planner = _Planner(offchip_intermediate_tensor, async_dma=True, out_path=str(tmp_path))
    _, ext_mem, _ = memplan_cp.cp_mem_planner_impl(planner, tensor_fixed_to_ext_mem=["tmp"])

    assert ext_mem == offchip_intermediate_tensor.num_bytes
    assert len(offchip_intermediate_tensor.load_from_ext_mem_at) == 1
    # In async mode, prefetch is allowed before use if feasible.
    assert offchip_intermediate_tensor.load_from_ext_mem_at[0] <= 2
    assert 2 in offchip_intermediate_tensor.mem_offset_at


def test_cp_memplan_async_writeback_has_tracked_window(monkeypatch, tmp_path, offchip_early_use_tensor):
    monkeypatch.setattr(memplan_cp, "save_memory_allocation_graph", _stub_graph)
    monkeypatch.setattr(memplan_cp, "save_memory_allocation_graph_nodes_buffers", _stub_nodes_buffers)

    planner = _Planner(
        offchip_early_use_tensor,
        async_dma=True,
        out_path=str(tmp_path),
        calls_idxs=[0, 1, 2, 3],
    )
    _, ext_mem, _ = memplan_cp.cp_mem_planner_impl(planner, tensor_fixed_to_ext_mem=["tmp_early"])

    assert ext_mem == offchip_early_use_tensor.num_bytes
    assert offchip_early_use_tensor.move_temp_to_ext_mem

    move_time = offchip_early_use_tensor.move_temp_to_ext_mem[0]
    assert move_time in [2, 3]
    # Write-back now has an explicit modeled window in async mode, so the tensor
    # must keep a valid on-chip offset at the selected move timestep.
    assert move_time in offchip_early_use_tensor.mem_offset_at
