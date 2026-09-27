"""自底向上结构聚合；策略接口可替换为其他确定性聚合器。"""
from __future__ import annotations

import heapq
import statistics
from collections import defaultdict, deque


class GreedyStructuralAggregator:
    def __init__(self, graph, cache_bytes=655360, merge_penalty=0.15):
        self.g = graph
        self.cache_bytes = max(1, cache_bytes)
        self.merge_penalty = merge_penalty
        sizes = [t.get("size", 0) for t in graph.tensors.values() if t.get("size", 0) > 0]
        self.reference_bytes = max(1, statistics.median(sizes) if sizes else 1)

    def _component_graph(self, components, owner):
        succ = {i: set() for i in components}
        for a in self.g.compute_ids:
            ca = owner[a]
            for b in self.g.compute_succ[a]:
                cb = owner[b]
                if ca != cb:
                    succ[ca].add(cb)
        indeg = {i: 0 for i in succ}
        for a in succ:
            for b in succ[a]:
                indeg[b] += 1
        q = deque(i for i in sorted(indeg) if indeg[i] == 0)
        count = 0
        while q:
            a = q.popleft(); count += 1
            for b in succ[a]:
                indeg[b] -= 1
                if indeg[b] == 0:
                    q.append(b)
        return succ, count == len(components)

    def _features(self, members):
        pipe = defaultdict(int)
        tids = set()
        for op_id in members:
            op = self.g.ops[op_id]
            pipe[op.get("pipe", "UNKNOWN")] += op.get("cycles", 0)
            tids.update(self.g.op_inputs[op_id]); tids.update(self.g.op_outputs[op_id])
        return dict(pipe), sum(self.g.tensors[t].get("size", 0) for t in tids)

    def run(self):
        # 每轮合并都验证完整商图无环，不依赖相邻拓扑区间的假设。
        comps = {i: {i} for i in self.g.compute_ids}
        owner = {i: i for i in self.g.compute_ids}
        features = {i: self._features({i}) for i in comps}
        heap = []

        def push_pair(a, b):
            if a == b:
                return
            a, b = sorted((a, b))
            crossing = set()
            for (u, v), tids in self.g.boundary_tensors.items():
                if (owner[u] == a and owner[v] == b) or (owner[u] == b and owner[v] == a):
                    crossing.update(tids)
            # 直接 Op→Op 依赖也形成可合并的结构边，但不会虚构 Tensor 节省。
            adjacent = any(owner[v] == b for u in comps[a] for v in self.g.compute_succ[u]) or \
                       any(owner[v] == a for u in comps[b] for v in self.g.compute_succ[u])
            if not adjacent:
                return
            saved_bytes = sum(self.g.tensors[t].get("size", 0) for t in crossing)
            if saved_bytes == 0:
                # 管线邻接且内部边界没有 Tensor 时，给予一个小的融合收益。
                saved_bytes = 1
            pa, ba = features[a]; pb, bb = features[b]
            work_a, work_b = sum(pa.values()), sum(pb.values())
            parallel_penalty = self.merge_penalty * min(work_a, work_b) / max(1, work_a + work_b)
            cache_pressure = max(0.0, (ba + bb) / self.cache_bytes - 1.0)
            score = saved_bytes / self.reference_bytes - parallel_penalty - 0.05 * cache_pressure
            # 小型、同一流水线的相邻 Op 能省 Task 边界启动开销。
            if set(pa) == set(pb):
                score += 0.02
            heapq.heappush(heap, (-score, a, b))

        for a in sorted(comps):
            for b in sorted(self.g.compute_succ[a]):
                push_pair(a, b)
        while heap:
            neg_score, a, b = heapq.heappop(heap)
            score = -neg_score
            if a not in comps or b not in comps or score <= 0:
                continue
            members = comps[a] | comps[b]
            trial_components = {k: v for k, v in comps.items() if k not in (a, b)}
            new_id = min(members)
            trial_components[new_id] = members
            trial_owner = {op: cid for cid, ops in trial_components.items() for op in ops}
            _, acyclic = self._component_graph(trial_components, trial_owner)
            if not acyclic:
                continue
            del comps[a]; del comps[b]
            comps[new_id] = members
            owner = trial_owner
            features.pop(a, None); features.pop(b, None)
            features[new_id] = self._features(members)
            # 仅新组件的邻居对需要加入候选队列；陈旧候选由 component ID 校验丢弃。
            neighbors = set()
            for op in members:
                neighbors.update(owner[v] for v in self.g.compute_succ[op] if owner[v] != new_id)
                neighbors.update(owner[v] for v in self.g.compute_pred[op] if owner[v] != new_id)
            for other in sorted(neighbors):
                push_pair(new_id, other)
        return sorted((sorted(m) for m in comps.values()), key=lambda m: min(m))


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
