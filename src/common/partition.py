"""Shared partition coverage and quotient-DAG checks."""
from __future__ import annotations
import heapq
from collections import deque


def validate_partition(graph, components):
    flat = [op for group in components for op in group]
    if sorted(flat) != sorted(graph.compute_ids) or len(flat) != len(set(flat)):
        raise ValueError("分组必须恰好覆盖所有非 COPY Op 一次")
    owner = {op: i for i, group in enumerate(components) for op in group}
    succ = {i: set() for i in range(len(components))}
    for a in graph.compute_ids:
        for b in graph.compute_succ[a]:
            if owner[a] != owner[b]:
                succ[owner[a]].add(owner[b])
    indeg = {i: 0 for i in succ}
    for a in succ:
        for b in succ[a]: indeg[b] += 1
    q = deque(i for i, d in indeg.items() if d == 0)
    count = 0
    while q:
        a = q.popleft(); count += 1
        for b in succ[a]:
            indeg[b] -= 1
            if indeg[b] == 0: q.append(b)
    if count != len(succ):
        raise ValueError("聚合后子图依赖图存在环")
    return succ


def quotient_graph(graph, groups):
    owner = {op: i for i, group in enumerate(groups) for op in group}
    succ = {i: set() for i in range(len(groups))}
    pred = {i: set() for i in range(len(groups))}
    for src in graph.compute_ids:
        for dst in graph.compute_succ[src]:
            a, b = owner[src], owner[dst]
            if a != b:
                succ[a].add(b)
                pred[b].add(a)
    indegree = {i: len(pred[i]) for i in pred}
    ready = [i for i in indegree if indegree[i] == 0]
    heapq.heapify(ready)
    topo = []
    while ready:
        node = heapq.heappop(ready)
        topo.append(node)
        for child in sorted(succ[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topo) != len(groups):
        raise ValueError("聚合后的 Task 商图存在环")
    return owner, succ, pred, topo
