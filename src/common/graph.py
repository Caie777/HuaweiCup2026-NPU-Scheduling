"""完整读取官方计算图，并构造可复用的结构视图（仅标准库）。"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


COPY_TYPES = {"COPY_IN", "COPY_OUT"}


def _topological(nodes, adjacency):
    indegree = {n: 0 for n in nodes}
    for src in nodes:
        for dst in adjacency[src]:
            indegree[dst] += 1
    ready = deque(sorted(n for n, d in indegree.items() if d == 0))
    order = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for dst in sorted(adjacency[node]):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
    if len(order) != len(nodes):
        raise ValueError("原始计算图存在环")
    return order


@dataclass
class GraphModel:
    raw: dict[str, Any]
    ops: dict[int, dict]
    tensors: dict[int, dict]
    op_inputs: dict[int, set[int]]
    op_outputs: dict[int, set[int]]
    tensor_producers: dict[int, set[int]]
    tensor_consumers: dict[int, set[int]]
    full_succ: dict[int, set[int]]
    full_pred: dict[int, set[int]]
    compute_ids: list[int]
    compute_pred: dict[int, set[int]]
    compute_succ: dict[int, set[int]]
    topological_order: list[int]
    compute_order: list[int]
    levels: dict[int, int]
    longest_path: dict[int, int]
    pipe_work: dict[str, int]
    boundary_tensors: dict[tuple[int, int], set[int]]

    @classmethod
    def parse(cls, graph: dict[str, Any]) -> "GraphModel":
        if not isinstance(graph, dict) or not all(isinstance(graph.get(k), list)
                                                  for k in ("ops", "tensors", "edges")):
            raise ValueError("原图必须含 ops、tensors、edges 列表")
        ops = {o["id"]: o for o in graph["ops"]}
        tensors = {t["id"]: t for t in graph["tensors"]}
        if len(ops) != len(graph["ops"]) or len(tensors) != len(graph["tensors"]):
            raise ValueError("Op/Tensor ID 重复")
        overlap = set(ops) & set(tensors)
        if overlap:
            raise ValueError(f"Op 与 Tensor ID 空间冲突: {min(overlap)}")
        all_nodes = set(ops) | set(tensors)
        full_succ = {n: set() for n in all_nodes}
        full_pred = {n: set() for n in all_nodes}
        op_inputs = {i: set() for i in ops}
        op_outputs = {i: set() for i in ops}
        producers = {i: set() for i in tensors}
        consumers = {i: set() for i in tensors}
        edge_seen = set()
        for e in graph["edges"]:
            a, b = e["source"], e["target"]
            if a not in all_nodes or b not in all_nodes:
                raise ValueError(f"边引用未知节点: {a}->{b}")
            if (a, b) in edge_seen:
                raise ValueError(f"重复边: {a}->{b}")
            edge_seen.add((a, b))
            if a in tensors and b in tensors:
                raise ValueError(f"不支持 Tensor→Tensor 边: {a}->{b}")
            full_succ[a].add(b)
            full_pred[b].add(a)
            if a in ops and b in tensors:
                op_outputs[a].add(b)
                producers[b].add(a)
            elif a in tensors and b in ops:
                op_inputs[b].add(a)
                consumers[a].add(b)
        full_order = _topological(all_nodes, full_succ)
        compute_ids = sorted(i for i, op in ops.items() if op.get("op") not in COPY_TYPES)

        # 在完整图上穿过 COPY 和 Tensor 节点，遇到计算 Op 即建立数据流依赖。
        compute_set = set(compute_ids)
        c_succ = {i: set() for i in compute_ids}
        c_pred = {i: set() for i in compute_ids}
        for source in compute_ids:
            seen, stack = set(), list(full_succ[source])
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                if node in compute_set:
                    if node != source:
                        c_succ[source].add(node)
                        c_pred[node].add(source)
                else:
                    stack.extend(full_succ[node])
        c_order = _topological(compute_ids, c_succ)
        levels, longest = {}, {}
        for op_id in c_order:
            levels[op_id] = max((levels[p] + 1 for p in c_pred[op_id]), default=0)
            longest[op_id] = ops[op_id].get("cycles", 0) + max(
                (longest[p] for p in c_pred[op_id]), default=0)
        pipes = sorted({op.get("pipe", "UNKNOWN") for op in ops.values()})
        pipe_work = {p: sum(o.get("cycles", 0) for o in ops.values()
                            if o.get("pipe", "UNKNOWN") == p and o.get("op") not in COPY_TYPES)
                     for p in pipes}
        edge_tensors = defaultdict(set)
        for tid in tensors:
            ps = producers[tid] & compute_set
            cs = consumers[tid] & compute_set
            for a in ps:
                for b in cs:
                    if a != b:
                        edge_tensors[(a, b)].add(tid)
        return cls(graph, ops, tensors, op_inputs, op_outputs, producers, consumers,
                   full_succ, full_pred, compute_ids, c_pred, c_succ, full_order,
                   c_order, levels, longest, pipe_work, dict(edge_tensors))

    def summary(self) -> dict[str, Any]:
        types = defaultdict(int)
        for op_id in self.compute_ids:
            types[self.ops[op_id].get("op", "UNKNOWN")] += 1
        shared = {str(t): sorted(self.tensor_consumers[t]) for t in self.tensors
                  if len(self.tensor_consumers[t] & set(self.compute_ids)) > 1}
        forks = sum(len(self.compute_succ[i]) > 1 for i in self.compute_ids)
        joins = sum(len(self.compute_pred[i]) > 1 for i in self.compute_ids)
        work = sum(self.ops[i].get("cycles", 0) for i in self.compute_ids)
        path = max(self.longest_path.values(), default=0)
        layer_histogram = defaultdict(int)
        for level in self.levels.values():
            layer_histogram[level] += 1
        return {
            "op_count_all": len(self.ops), "tensor_count": len(self.tensors),
            "edge_count": len(self.raw["edges"]), "compute_op_count": len(self.compute_ids),
            "copy_op_count": len(self.ops) - len(self.compute_ids),
            "op_types": dict(sorted(types.items())), "shared_input_tensors": shared,
            "fork_op_count": forks, "join_op_count": joins,
            "topological_layer_count": len(layer_histogram),
            "ops_per_topological_layer": {str(k): layer_histogram[k] for k in sorted(layer_histogram)},
            "critical_path_cycles_estimate": path,
            "total_compute_cycles": work,
            "theoretical_parallelism_cycles_ratio": (work / path if path else 0.0),
            "pipe_work_cycles": self.pipe_work,
        }
