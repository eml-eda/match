"""
compute_async_waits.py
======================
Computes the minimal set of async-DMA wait points for a TVM-compiled graph,
covering both:

  1. Loads   (load_from_ext_mem_at)  – tensor must arrive before first use
  2. Writebacks (move_temp_to_ext_mem) – buffer must be flushed before the
     next tenant of the same on-chip storage region starts writing into it

Key assumptions
---------------
* All transfers (loads AND writebacks) share a single in-order DMA queue.
  Transfers are issued in the order tensors appear in the flat mem_tensors
  list (approximated here as dict-insertion order), broken into groups per
  node.  Within a node, writebacks are issued before loads (flush-before-fill).
* Because the queue is strictly in-order, waiting on transfer #k guarantees
  transfers #0 ... #k-1 are also complete.  So for a group of transfers all
  required before the same node, only the highest-numbered one needs an
  explicit wait.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class TransferKind(Enum):
    LOAD      = auto()   # ext-mem -> on-chip
    WRITEBACK = auto()   # on-chip -> ext-mem


@dataclass
class Transfer:
    transfer_id:    int
    kind:           TransferKind
    tensor_name:    str
    issue_node:     int   # Node that issues this DMA
    wait_node:      int   # Earliest node that *must* see it complete


@dataclass
class WaitPoint:
    node_id:            int
    transfer_id:        int
    tracking_tensor:    str
    kind:               TransferKind
    covered_transfers:  List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_geq(values: List[int], threshold: int) -> Optional[int]:
    """Return the smallest value in `values` that is >= threshold, or None."""
    return next((v for v in sorted(values) if v >= threshold), None)


def _build_storage_map(metadata: dict) -> Dict[int, List[str]]:
    """Map tvm_memplan_storage_id -> list of tensor names sharing that buffer."""
    storage: Dict[int, List[str]] = defaultdict(list)
    for name, info in metadata.items():
        sid = info.get("tvm_memplan_storage_id")
        if sid is not None:
            storage[sid].append(name)
    return storage


def _first_buffer_access_after(
    tensor_name: str,
    after_node:  int,
    metadata:    dict,
    storage_map: Dict[int, List[str]],
) -> Optional[int]:
    """
    Return the earliest node (strictly > after_node) at which any tensor
    sharing the same on-chip storage buffer as `tensor_name` is first accessed.

    'Accessed' means the node appears in `used_at` OR `load_from_ext_mem_at`
    (a load also writes into the buffer).  We exclude `move_temp_to_ext_mem`
    itself — that is the source of the hazard, not a new use.
    """
    info = metadata[tensor_name]
    sid  = info.get("tvm_memplan_storage_id")
    if sid is None:
        return None

    earliest: Optional[int] = None
    for peer_name in storage_map.get(sid, []):
        if peer_name == tensor_name:
            continue
        peer = metadata[peer_name]
        access_nodes = sorted(
            set(peer.get("used_at", []))
            | set(peer.get("load_from_ext_mem_at", []))
        )
        candidate = _first_geq(access_nodes, after_node + 1)
        if candidate is not None:
            earliest = candidate if earliest is None else min(earliest, candidate)

    return earliest


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def compute_async_waits(metadata: dict, verbose: bool = False) -> List[WaitPoint]:
    """
    Parameters
    ----------
    metadata : dict   Full metadata dict from metadata.json.
    verbose  : bool   Print intermediate steps when True.

    Returns
    -------
    Sorted list of WaitPoint (one per node that needs at least one wait).
    """

    storage_map = _build_storage_map(metadata)
    tensor_pos: Dict[str, int] = {name: i for i, name in enumerate(metadata)}

    # ------------------------------------------------------------------
    # Step 1 - Enumerate all transfers with (issue_node, wait_node)
    # ------------------------------------------------------------------
    # Each entry: (issue_node, intra_node_priority, kind, name, wait_node, pos)
    # intra_node_priority: 0=writeback, 1=load  (WBs issued before loads)
    pending = []

    for name, info in metadata.items():
        pos = tensor_pos[name]

        # ---- WRITEBACKS ----
        for issue_node in info.get("move_temp_to_ext_mem", []):
            wait_node = _first_buffer_access_after(name, issue_node, metadata, storage_map)
            if wait_node is None:
                if verbose:
                    print(f"  [SKIP WB] {name}: writeback at node {issue_node} "
                          f"- no subsequent buffer tenant found.")
                continue
            pending.append((issue_node, 0, TransferKind.WRITEBACK, name, wait_node, pos))

        # ---- LOADS ----
        for issue_node in info.get("load_from_ext_mem_at", []):
            used_at = sorted(info.get("used_at", []))
            wait_node = _first_geq(used_at, issue_node)
            if wait_node is None:
                if verbose:
                    print(f"  [SKIP LD] {name}: loaded at node {issue_node} "
                          f"- never used afterwards.")
                continue
            pending.append((issue_node, 1, TransferKind.LOAD, name, wait_node, pos))

    # Sort: (issue_node, intra_node_priority, tensor_position_in_mem_list)
    pending.sort(key=lambda x: (x[0], x[1], x[5]))

    transfers: List[Transfer] = []
    for tid, (issue_node, _prio, kind, name, wait_node, _pos) in enumerate(pending):
        transfers.append(Transfer(
            transfer_id=tid,
            kind=kind,
            tensor_name=name,
            issue_node=issue_node,
            wait_node=wait_node,
        ))

    if verbose:
        print("=== All DMA transfers (global issue order) ===")
        for t in transfers:
            tag = "LOAD" if t.kind == TransferKind.LOAD else "  WB"
            print(f"  [{tag}] id={t.transfer_id:3d}  tensor={t.tensor_name:<40s}"
                  f"  issued@node={t.issue_node}  wait_needed@node={t.wait_node}")
        print()

    # ------------------------------------------------------------------
    # Step 2 - Group by wait_node; keep only the last transfer_id per group
    # ------------------------------------------------------------------
    by_wait_node: Dict[int, List[Transfer]] = defaultdict(list)
    for t in transfers:
        by_wait_node[t.wait_node].append(t)

    wait_points: List[WaitPoint] = []
    for node_id in sorted(by_wait_node):
        group   = by_wait_node[node_id]
        last    = max(group, key=lambda t: t.transfer_id)
        covered = sorted(t.transfer_id for t in group)
        wait_points.append(WaitPoint(
            node_id=node_id,
            transfer_id=last.transfer_id,
            tracking_tensor=last.tensor_name,
            kind=last.kind,
            covered_transfers=covered,
        ))

    return wait_points


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_wait_schedule(wait_points: List[WaitPoint]) -> None:
    KIND = {TransferKind.LOAD: "LOAD", TransferKind.WRITEBACK: "  WB"}
    print("=== Async-DMA wait schedule ===")
    print(f"{'Node':>6}  {'async_wait(id)':>16}  {'kind':>4}  "
          f"{'tracking tensor':<40}  covered ids")
    print("-" * 105)
    for wp in wait_points:
        cov = str(wp.covered_transfers) if len(wp.covered_transfers) > 1 else ""
        print(f"  {wp.node_id:>4}  {'async_wait(' + str(wp.transfer_id) + ')':>16}  "
              f"{KIND[wp.kind]:>4}  {wp.tracking_tensor:<40}  {cov}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("metadata.json")
    with open(path) as f:
        metadata = json.load(f)

    wait_points = compute_async_waits(metadata, verbose=True)
    print_wait_schedule(wait_points)

    output = [
        {
            "node_id":                wp.node_id,
            "async_wait_transfer_id": wp.transfer_id,
            "kind":                   wp.kind.name,
            "tracking_tensor":        wp.tracking_tensor,
            "covered_transfer_ids":   wp.covered_transfers,
        }
        for wp in wait_points
    ]
    out_path = path.with_name("wait_schedule.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSchedule written to {out_path}")