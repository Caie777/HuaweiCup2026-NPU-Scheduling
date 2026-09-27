"""Reusable Task-local L1/UB liveness and cache-risk estimates.

The estimate follows an operation order and releases a tensor immediately
after its last local use. It is a ranking signal, not a legality test or a
replacement for the official Step2 spill allocator.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import os
import sys


def _position(tensor):
    pos = tensor.get("pos", "UB")
    return "UB" if pos == "DDR" else pos


def _simulate(ops, tensors, edges, order, capacity):
    op_by_id = {op["id"]: op for op in ops}
    tensor_by_id = {t["id"]: t for t in tensors}
    inputs = defaultdict(set); outputs = defaultdict(set)
    for edge in edges:
        a, b = edge["source"], edge["target"]
        if a in op_by_id and b in tensor_by_id:
            outputs[a].add(b)
        elif a in tensor_by_id and b in op_by_id:
            inputs[b].add(a)
    order = [op for op in order if op in op_by_id]
    if len(order) != len(op_by_id) or len(set(order)) != len(order):
        raise ValueError("Task 内操作序列没有恰好覆盖所有 Op")
    index = {op: i for i, op in enumerate(order)}
    last_use = {}
    first_use = {}
    local_consumers = defaultdict(list)
    for op in order:
        for tid in inputs[op]:
            local_consumers[tid].append(op)
            first_use[tid] = min(first_use.get(tid, index[op]), index[op])
            last_use[tid] = max(last_use.get(tid, index[op]), index[op])
        for tid in outputs[op]:
            first_use[tid] = min(first_use.get(tid, index[op]), index[op])
            last_use[tid] = max(last_use.get(tid, index[op]), index[op])
    live = set(); current = defaultdict(int); peak = defaultdict(int)
    peak_tensors = {"L1": [], "UB": []}
    for i, op in enumerate(order):
        for tid in sorted(inputs[op] | outputs[op]):
            if tid not in live:
                live.add(tid)
                pos = _position(tensor_by_id[tid])
                if pos in ("L1", "UB"):
                    current[pos] += tensor_by_id[tid].get("size", 0)
        for pos in ("L1", "UB"):
            if current[pos] > peak[pos]:
                peak[pos] = current[pos]
                peak_tensors[pos] = sorted(t for t in live if _position(tensor_by_id[t]) == pos)
        for tid in sorted(inputs[op] | outputs[op]):
            if last_use.get(tid) == i and tid in live:
                live.remove(tid)
                pos = _position(tensor_by_id[tid])
                if pos in ("L1", "UB"):
                    current[pos] -= tensor_by_id[tid].get("size", 0)
    lifetimes = []
    for tid, start in first_use.items():
        tensor = tensor_by_id[tid]
        pos = _position(tensor)
        if pos not in ("L1", "UB"):
            continue
        span = last_use[tid] - start + 1
        lifetimes.append({"tensor_id": tid, "pos": pos, "size": tensor.get("size", 0),
                          "first_op_index": start, "last_op_index": last_use[tid],
                          "live_span_ops": span, "local_consumer_count": len(local_consumers[tid]),
                          "weighted_live_bytes": tensor.get("size", 0) * span})
    lifetimes.sort(key=lambda r: (-r["weighted_live_bytes"], -r["size"], r["tensor_id"]))
    peak_live = {pos: peak.get(pos, 0) for pos in ("L1", "UB")}
    overflow = {pos: max(0, peak_live[pos] - capacity.get(pos, 0)) for pos in ("L1", "UB")}
    return {"peak_live_bytes": peak_live, "overflow_bytes": overflow,
            "peak_tensors": peak_tensors,
            "high_risk_tensors": lifetimes[:20],
            "shared_input_reuse_count": sum(max(0, len(uses) - 1)
                                            for uses in local_consumers.values()),
            "tensor_lifetime_count": len(lifetimes), "op_count": len(order)}


def estimate_task(graph, members, capacity, order=None):
    """Estimate pre-spill pressure for a proposed Task in an original graph."""
    members = set(members)
    if not members:
        return {"peak_live_bytes": {"L1": 0, "UB": 0},
                "overflow_bytes": {"L1": 0, "UB": 0}, "high_risk_tensors": []}
    local_order = [op for op in (order or graph.compute_order) if op in members]
    touched = set()
    for op in members:
        touched.update(graph.op_inputs[op]); touched.update(graph.op_outputs[op])
    edges = []
    for op in members:
        edges.extend({"source": tid, "target": op} for tid in graph.op_inputs[op])
        edges.extend({"source": op, "target": tid} for tid in graph.op_outputs[op])
    return _simulate([graph.ops[op] for op in members],
                     [graph.tensors[tid] for tid in touched], edges, local_order, capacity)


def estimate_official_task_graph(task_graph, step1_order, capacity):
    """Calibrate against official Step1 order on its expanded Task graph."""
    return _simulate(task_graph["ops"], task_graph["tensors"],
                     task_graph["edges"], step1_order, capacity)


def index_compute_edges(graph):
    """Index raw edges by incident compute Op, retaining original edge order."""
    index = {op: [] for op in graph.compute_ids}
    for position, edge in enumerate(graph.raw["edges"]):
        source, target = edge["source"], edge["target"]
        if source in index:
            index[source].append(position)
        if target in index and target != source:
            index[target].append(position)
    return index


@lru_cache(maxsize=1)
def _official_step1():
    from official_input import resolve_official_root
    official = resolve_official_root(os.environ.get("MATH_MODEL_OFFICIAL_ROOT"))
    sys.path.insert(0, str(official / "code"))
    from schedule_step1 import step1_schedule
    return step1_schedule


def estimate_task_step1(graph, members, capacity, *, edge_index=None):
    """Use official Step1 ordering on the Task's original induced graph.

    Boundary COPY insertion is omitted for speed. This still reproduces the
    Op ordering responsible for long-lived L1 tensors in case_083; calibration
    against exact expanded Task graphs is reported separately.
    """
    members = set(members)
    if not members:
        return estimate_task(graph, members, capacity)
    touched = set()
    for op in members:
        touched.update(graph.op_inputs[op]); touched.update(graph.op_outputs[op])
    source_edges = (graph.raw["edges"] if edge_index is None else
                    (graph.raw["edges"][i] for i in sorted({position
                     for op in members for position in edge_index[op]})))
    edges = [edge for edge in source_edges
             if ((edge["source"] in members and edge["target"] in touched)
                 or (edge["source"] in touched and edge["target"] in members)
                 or (edge["source"] in members and edge["target"] in members))]
    local = {"ops": [graph.ops[o] for o in sorted(members)],
             "tensors": [graph.tensors[t] for t in sorted(touched)],
             "edges": edges}
    order = _official_step1()(local)
    return _simulate(local["ops"], local["tensors"], local["edges"], order, capacity)
