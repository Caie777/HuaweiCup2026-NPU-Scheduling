"""Structural, raw-Op macro partitions for OJO.

Every group is a contiguous interval of a computed topological order.  This
gives a DAG quotient while allowing boundaries to cross natural modules.
The final OJO partition and schedule validators still check every proposal.
"""
from __future__ import annotations

import bisect
import heapq
import math
from collections import defaultdict


def structural_orders(graph, work):
    """Return stage, critical-chain and Tensor-locality topological orders."""
    ops = graph.compute_order
    remaining_path = {}
    for op in reversed(ops):
        remaining_path[op] = work[op] + max(
            (remaining_path[n] for n in graph.compute_succ[op]), default=0)
    shared = defaultdict(int)
    compute_set = set(graph.compute_ids)
    for tid, consumers in graph.tensor_consumers.items():
        local = consumers & compute_set
        if len(local) < 2:
            continue
        size = int(graph.tensors[tid].get("size", 0))
        # Divide by fanout so one very wide input does not dominate all orderings.
        contribution = size / math.sqrt(len(local))
        for op in local:
            shared[op] += contribution
    modes = {
        "chain": lambda op: (-remaining_path[op], -graph.levels[op], op),
        "stage": lambda op: (graph.levels[op], -remaining_path[op], op),
        "tensor": lambda op: (-shared[op], graph.levels[op], -remaining_path[op], op),
    }
    result = {}
    for name, priority in modes.items():
        pending = {op: len(graph.compute_pred[op]) for op in ops}
        ready = [(priority(op), op) for op in ops if pending[op] == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            _, op = heapq.heappop(ready)
            order.append(op)
            for child in graph.compute_succ[op]:
                pending[child] -= 1
                if pending[child] == 0:
                    heapq.heappush(ready, (priority(child), child))
        if len(order) != len(ops):
            raise ValueError("raw compute DAG is cyclic")
        result[name] = order
    return result


def tensor_boundary_cost(graph, order):
    """Approximate bytes live across each possible cut, once per Tensor.

    External shared inputs span their first and last consumer. Internal Tensors
    span their earliest producer/use and latest consumer. This is a ranking
    heuristic, never an official communication-cost claim.
    """
    position = {op: i for i, op in enumerate(order)}
    delta = [0] * (len(order) + 1)
    for tid, tensor in graph.tensors.items():
        users = [position[o] for o in graph.tensor_consumers[tid] if o in position]
        producers = [position[o] for o in graph.tensor_producers[tid] if o in position]
        if not users:
            continue
        first = min(producers + users)
        last = max(users)
        if first < last:
            size = int(tensor.get("size", 0))
            delta[first + 1] += size
            delta[last + 1] -= size
    active = 0
    cost = [0] * (len(order) + 1)
    for cut in range(1, len(order)):
        active += delta[cut]
        cost[cut] = active
    return cost


def structural_cuts(graph, order, work, target, *, boundary_cost=None):
    """Choose work-balanced cuts near low Tensor traffic or level boundaries."""
    n = len(order)
    target = min(max(1, target), n)
    if target == 1:
        return [order]
    prefix = [0]
    for op in order:
        prefix.append(prefix[-1] + max(1, work[op]))
    cost = boundary_cost if boundary_cost is not None else tensor_boundary_cost(graph, order)
    max_cost = max(cost) or 1
    positions = [0]
    for part in range(1, target):
        lo = positions[-1] + 1
        hi = n - (target - part)
        if lo > hi:
            break
        ideal = bisect.bisect_left(prefix, prefix[-1] * part / target)
        radius = max(4, min(256, n // (target * 3)))
        left, right = max(lo, ideal - radius), min(hi, ideal + radius)
        if left > right:
            left = right = min(hi, max(lo, ideal))
        scale = max(1, prefix[-1] / target)
        cut = min(range(left, right + 1), key=lambda i: (
            abs(prefix[i] - prefix[-1] * part / target) / scale
            + 0.32 * cost[i] / max_cost
            - (0.08 if graph.levels[order[i - 1]] != graph.levels[order[i]] else 0),
            i))
        positions.append(cut)
    positions.append(n)
    return [order[a:b] for a, b in zip(positions, positions[1:]) if a < b]


def macro_partitions(graph, work, cores, *, max_candidates=15):
    """Bounded global partition family over raw Op dependencies and Tensors."""
    n = len(graph.compute_ids)
    if n < 2:
        return []
    targets = sorted({min(n, max(2, int(cores) * multiplier))
                      for multiplier in (2, 8, 32, 96)})
    orders = structural_orders(graph, work)
    proposals = []
    seen = set()
    for mode, order in orders.items():
        cost = tensor_boundary_cost(graph, order)
        for target in targets:
            groups = structural_cuts(graph, order, work, target, boundary_cost=cost)
            identity = tuple(tuple(sorted(g)) for g in groups)
            if identity in seen:
                continue
            seen.add(identity)
            proposals.append((groups, f"macro_{mode}_{target}_tasks"))
            if len(proposals) >= max_candidates:
                break
    # A quotient of level-homogeneous groups is always acyclic: every raw
    # dependency increases the level. Splitting each level retains parallel
    # branches that a single chain of topological intervals would serialize.
    by_level = defaultdict(list)
    for op in graph.compute_order:
        by_level[graph.levels[op]].append(op)
    for lanes in (2, 4):
        groups = []
        for level in sorted(by_level):
            row = sorted(by_level[level], key=lambda op: (-work[op], op))
            bins = [[] for _ in range(min(lanes, len(row)))]
            loads = [0] * len(bins)
            for op in row:
                chosen = min(range(len(bins)), key=lambda i: (loads[i], i))
                bins[chosen].append(op)
                loads[chosen] += work[op]
            groups.extend(bin for bin in bins if bin)
        identity = tuple(tuple(sorted(g)) for g in groups)
        if identity not in seen and len(groups) <= 8000:
            seen.add(identity)
            proposals.append((groups, f"macro_level_{lanes}_lanes"))
    # Weak raw-Op components have no inter-component dependency. Small ones
    # may be packed together; large ones are split internally at several
    # scales rather than treated as indivisible natural modules.
    visited = set()
    components = []
    for op in graph.compute_order:
        if op in visited:
            continue
        todo = [op]
        visited.add(op)
        members = set()
        while todo:
            current = todo.pop()
            members.add(current)
            for nxt in graph.compute_pred[current] | graph.compute_succ[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    todo.append(nxt)
        components.append(members)
    if len(components) > 1:
        position = {op: i for i, op in enumerate(graph.compute_order)}
        for cap in (64, 256, 1024):
            large = []
            small = []
            for members in components:
                order = sorted(members, key=position.__getitem__)
                if len(order) > cap:
                    large.extend(order[i:i + cap] for i in range(0, len(order), cap))
                else:
                    small.append(order)
            bins = [[] for _ in range(min(max(2, cores * 4), len(small)))]
            loads = [0] * len(bins)
            for component in sorted(small, key=lambda row: -sum(work[op] for op in row)):
                chosen = min(range(len(bins)), key=lambda i: (loads[i], i))
                bins[chosen].extend(component)
                loads[chosen] += sum(work[op] for op in component)
            groups = large + [b for b in bins if b]
            identity = tuple(tuple(sorted(g)) for g in groups)
            if identity not in seen:
                seen.add(identity)
                proposals.append((groups, f"macro_component_{cap}_op_chunks"))
    # Contract unbranched producer-consumer runs directly on the raw DAG.
    # This retains independent lanes in a deep graph and internalizes the
    # traffic within each run. The full quotient check remains mandatory.
    for cap in (4, 8, 16):
        assigned = set()
        groups = []
        for op in graph.compute_order:
            if op in assigned:
                continue
            group = [op]
            assigned.add(op)
            while len(group) < cap and len(graph.compute_succ[group[-1]]) == 1:
                nxt = next(iter(graph.compute_succ[group[-1]]))
                if nxt in assigned or len(graph.compute_pred[nxt]) != 1:
                    break
                group.append(nxt)
                assigned.add(nxt)
            groups.append(group)
        identity = tuple(tuple(sorted(g)) for g in groups)
        if identity not in seen:
            seen.add(identity)
            proposals.append((groups, f"macro_unary_{cap}_op_runs"))
    return proposals
