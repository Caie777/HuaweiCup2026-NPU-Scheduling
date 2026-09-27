"""Fast level-one module aggregation and resource-aware level-two coarsening.

V1 remains untouched. The first class preserves V1's deterministic scoring but
uses a local alternate-path contraction check instead of rebuilding the whole
quotient DAG after every candidate merge.
"""
from __future__ import annotations

import heapq
import statistics
from collections import defaultdict, deque

from aggregate import validate_partition
from schedule import make_plan


def _reachable(start_nodes, target, adjacency):
    """Whether target can be reached from any start without visiting source."""
    stack = list(start_nodes)
    seen = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return False


class FastV1ModuleAggregator:
    """V1-equivalent module discovery with incremental quotient adjacency."""

    def __init__(self, graph, cache_bytes=655360, merge_penalty=0.15):
        self.g = graph
        self.cache_bytes = max(1, cache_bytes)
        sizes = [t.get("size", 0) for t in graph.tensors.values()
                 if t.get("size", 0) > 0]
        self.reference_bytes = max(1, statistics.median(sizes) if sizes else 1)

    def _features(self, members):
        pipe = defaultdict(int)
        tids = set()
        for op_id in members:
            op = self.g.ops[op_id]
            pipe[op.get("pipe", "UNKNOWN")] += op.get("cycles", 0)
            tids.update(self.g.op_inputs[op_id])
            tids.update(self.g.op_outputs[op_id])
        return dict(pipe), sum(self.g.tensors[t].get("size", 0) for t in tids)

    def run(self):
        comps = {op_id: {op_id} for op_id in self.g.compute_ids}
        owner = {op_id: op_id for op_id in self.g.compute_ids}
        succ = {op_id: set(self.g.compute_succ[op_id]) for op_id in self.g.compute_ids}
        pred = {op_id: set(self.g.compute_pred[op_id]) for op_id in self.g.compute_ids}
        # Dense reachability bitsets are fast on small graphs but have O(C^2)
        # worst-case bits. Large graphs use exact bounded-start DFS instead.
        use_bitsets = len(self.g.compute_ids) <= 4096
        member_mask, component_descendants = {}, {}
        if use_bitsets:
            dense = {op_id: index for index, op_id in enumerate(self.g.compute_order)}
            member_mask = {op_id: 1 << dense[op_id] for op_id in self.g.compute_ids}
            descendant_bits = {}
            for op_id in reversed(self.g.compute_order):
                reachable = 0
                for child in self.g.compute_succ[op_id]:
                    reachable |= descendant_bits[child] | member_mask[child]
                descendant_bits[op_id] = reachable
            component_descendants = dict(descendant_bits)
        features = {op_id: self._features({op_id}) for op_id in comps}
        component_tensors = {
            op_id: set(self.g.op_inputs[op_id]) | set(self.g.op_outputs[op_id])
            for op_id in comps
        }
        edge_tensors = {
            (src, dst): set(self.g.boundary_tensors.get((src, dst), ()))
            for src in self.g.compute_ids for dst in self.g.compute_succ[src]
        }
        heap = []

        def push_pair(a, b):
            if a == b or a not in comps or b not in comps:
                return
            a, b = sorted((a, b))
            adjacent = b in succ[a] or a in succ[b]
            if not adjacent:
                return
            crossing = (edge_tensors.get((a, b), set())
                        | edge_tensors.get((b, a), set()))
            saved_bytes = sum(self.g.tensors[t].get("size", 0) for t in crossing)
            if saved_bytes == 0:
                saved_bytes = 1
            pa, ba = features[a]
            pb, bb = features[b]
            work_a, work_b = sum(pa.values()), sum(pb.values())
            penalty = 0.15 * min(work_a, work_b) / max(1, work_a + work_b)
            pressure = max(0.0, (ba + bb) / self.cache_bytes - 1.0)
            score = saved_bytes / self.reference_bytes - penalty - 0.05 * pressure
            if set(pa) == set(pb):
                score += 0.02
            heapq.heappush(heap, (-score, a, b))

        for a in sorted(comps):
            for b in sorted(succ[a]):
                push_pair(a, b)

        while heap:
            neg_score, a, b = heapq.heappop(heap)
            if a not in comps or b not in comps or -neg_score <= 0:
                continue
            # For adjacent A->B, an alternate path exists exactly when some
            # other immediate predecessor P of B is reachable from A. Dense
            # integer bitsets make this check a few wordwise operations.
            reverse_path = b in succ[a]
            alternate = False
            if reverse_path:
                if use_bitsets:
                    alternate = any(component_descendants[a] & member_mask[p]
                                    for p in pred[b] if p != a)
                else:
                    alternate = _reachable(succ[a] - {b}, b, succ)
            elif a in succ[b]:
                if use_bitsets:
                    alternate = any(component_descendants[b] & member_mask[p]
                                    for p in pred[a] if p != b)
                else:
                    alternate = _reachable(succ[b] - {a}, a, succ)
            if alternate:
                continue

            members = comps[a] | comps[b]
            new_id = min(members)
            predecessors = (pred[a] | pred[b]) - {a, b}
            successors = (succ[a] | succ[b]) - {a, b}

            # Keep the larger Tensor set and add the smaller set. This makes
            # repeated feature updates amortized instead of rescanning a large
            # merged component for every accepted edge.
            tids_a, tids_b = component_tensors[a], component_tensors[b]
            if len(tids_a) < len(tids_b):
                tids_a, tids_b = tids_b, tids_a
                feature_a, feature_b = features[b], features[a]
            else:
                feature_a, feature_b = features[a], features[b]
            overlap_bytes = sum(self.g.tensors[tid].get("size", 0)
                                for tid in tids_b if tid in tids_a)
            tids_a.update(tids_b)
            merged_mask = member_mask[a] | member_mask[b] if use_bitsets else None
            merged_descendants = (component_descendants[a] | component_descendants[b]
                                  if use_bitsets else None)
            pipe_work = defaultdict(int, feature_a[0])
            for pipe, amount in feature_b[0].items():
                pipe_work[pipe] += amount
            merged_bytes = feature_a[1] + feature_b[1] - overlap_bytes

            for node in predecessors:
                incoming = (edge_tensors.pop((node, a), set())
                            | edge_tensors.pop((node, b), set()))
                edge_tensors[(node, new_id)] = incoming
            for node in successors:
                outgoing = (edge_tensors.pop((a, node), set())
                            | edge_tensors.pop((b, node), set()))
                edge_tensors[(new_id, node)] = outgoing
            edge_tensors.pop((a, b), None)
            edge_tensors.pop((b, a), None)

            for node in predecessors:
                succ[node].discard(a)
                succ[node].discard(b)
                succ[node].add(new_id)
            for node in successors:
                pred[node].discard(a)
                pred[node].discard(b)
                pred[node].add(new_id)

            for node in (a, b):
                succ.pop(node, None)
                pred.pop(node, None)
                comps.pop(node, None)
                features.pop(node, None)
                component_tensors.pop(node, None)
                member_mask.pop(node, None)
                component_descendants.pop(node, None)
            comps[new_id] = members
            succ[new_id] = set(successors)
            pred[new_id] = set(predecessors)
            features[new_id] = (dict(pipe_work), merged_bytes)
            component_tensors[new_id] = tids_a
            if use_bitsets:
                member_mask[new_id] = merged_mask
                component_descendants[new_id] = merged_descendants
            for op_id in members:
                owner[op_id] = new_id

            neighbors = predecessors | successors
            for other in sorted(neighbors):
                push_pair(new_id, other)

        return sorted((sorted(members) for members in comps.values()),
                      key=lambda members: members[0])


def _topological_modules(adjacency):
    indegree = {node: 0 for node in adjacency}
    for src in adjacency:
        for dst in adjacency[src]:
            indegree[dst] += 1
    ready = [node for node, value in indegree.items() if value == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for dst in sorted(adjacency[node]):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                heapq.heappush(ready, dst)
    if len(order) != len(adjacency):
        raise ValueError("第一级模块依赖图存在环")
    return order


class ResourceAwareAggregator:
    """Select a legal level-two partition from workload-derived candidates.

    Candidate Task counts start at the requested core count and double until
    the level-one module count is reached. Each candidate is load-balanced by
    Pipe work, then scored by the existing deterministic HEFT proxy, shared
    Tensor boundary traffic, same-core waits, and estimated cache overflow.
    No module-per-Task constant or case identity is used.
    """

    COMMUNICATION_WEIGHT = 0.25

    def __init__(self, graph, modules, num_cores, *, bandwidth,
                 cross_wait, same_wait, capacity):
        self.g = graph
        self.modules = [sorted(set(members)) for members in modules]
        self.num_cores = num_cores
        self.bandwidth = bandwidth
        self.cross_wait = cross_wait
        self.same_wait = same_wait
        self.capacity = dict(capacity)
        if num_cores < 1:
            raise ValueError("核心数必须为正")
        flattened = [op for members in self.modules for op in members]
        if sorted(flattened) != graph.compute_ids or len(flattened) != len(set(flattened)):
            raise ValueError("第一级模块必须恰好覆盖所有计算 Op 一次")
        self.module_of = {op: mid for mid, members in enumerate(self.modules)
                          for op in members}
        self.module_succ = {mid: set() for mid in range(len(self.modules))}
        self.module_pred = {mid: set() for mid in range(len(self.modules))}
        for op in graph.compute_ids:
            a = self.module_of[op]
            for child in graph.compute_succ[op]:
                b = self.module_of[child]
                if a != b:
                    self.module_succ[a].add(b)
                    self.module_pred[b].add(a)
        self.module_topo = _topological_modules(self.module_succ)
        self.module_features = [self._module_features(mid, members)
                                for mid, members in enumerate(self.modules)]
        self.tensor_modules = self._build_tensor_module_views()

    def _module_features(self, module_id, members):
        pipe = defaultdict(int)
        touched = set()
        for op_id in members:
            op = self.g.ops[op_id]
            pipe[op.get("pipe", "UNKNOWN")] += op.get("cycles", 0)
            touched.update(self.g.op_inputs[op_id])
            touched.update(self.g.op_outputs[op_id])
        by_pos = defaultdict(int)
        for tid in touched:
            pos = self.g.tensors[tid].get("pos", "UNKNOWN")
            by_pos[pos] += self.g.tensors[tid].get("size", 0)
        return {
            "module_id": module_id,
            "ops": set(members),
            "pipe": dict(pipe),
            "cycles": sum(pipe.values()),
            "critical_path": max((self.g.longest_path[op] for op in members), default=0),
            "touched_tensors": touched,
            "working_set_by_pos": dict(by_pos),
        }

    def _build_tensor_module_views(self):
        rows = {}
        compute = set(self.g.compute_ids)
        for tid in self.g.tensors:
            producer_ops = self.g.tensor_producers[tid] & compute
            consumer_ops = self.g.tensor_consumers[tid] & compute
            rows[tid] = {
                "producers": {self.module_of[op] for op in producer_ops},
                "consumers": {self.module_of[op] for op in consumer_ops},
                "original_copy_in": sum(
                    self.g.ops[op].get("op") == "COPY_IN"
                    for op in self.g.tensor_producers[tid]),
                "original_copy_out": any(
                    self.g.ops[op].get("op") == "COPY_OUT"
                    for op in self.g.tensor_consumers[tid]),
            }
        return rows

    def _candidate_task_counts(self):
        count = len(self.modules)
        if count == 0:
            return [0]
        candidates = {count}
        task_count = min(count, self.num_cores)
        while task_count < count:
            candidates.add(task_count)
            task_count = min(count, task_count * 2)
        return sorted(candidates)

    def _boundary_bytes(self, module_groups):
        owner = {module_id: task_id for task_id, members in enumerate(module_groups)
                 for module_id in members}
        boundary_bytes = 0
        per_task_bytes = [0] * len(module_groups)
        for tid, row in self.tensor_modules.items():
            producer_tasks = {owner[mid] for mid in row["producers"]}
            consumer_tasks = {owner[mid] for mid in row["consumers"]}
            for task_id in consumer_tasks:
                if task_id not in producer_tasks:
                    size = self.g.tensors[tid].get("size", 0)
                    boundary_bytes += size
                    per_task_bytes[task_id] += size
            for task_id in producer_tasks:
                external_consumer = bool(row["original_copy_out"])
                has_any_compute_consumer = bool(row["consumers"])
                remote_compute_consumer = any(other != task_id for other in consumer_tasks)
                if external_consumer or not has_any_compute_consumer or remote_compute_consumer:
                    size = self.g.tensors[tid].get("size", 0)
                    boundary_bytes += size
                    per_task_bytes[task_id] += size
        return boundary_bytes, per_task_bytes

    def _cache_peaks(self, module_groups):
        """Conservative module-liveness estimate, avoiding summing independent scratch."""
        peaks = []
        for mids in module_groups:
            mid_set = set(mids)
            order = [mid for mid in self.module_topo if mid in mid_set]
            consumers_by_tensor = {
                tid: row["consumers"] & mid_set
                for tid, row in self.tensor_modules.items()
            }
            live = set()
            peak = defaultdict(int)
            for mid in order:
                local_tensors = self.module_features[mid]["touched_tensors"]
                live_before = live | local_tensors
                totals = defaultdict(int)
                for tid in live_before:
                    pos = self.g.tensors[tid].get("pos", "UNKNOWN")
                    totals[pos] += self.g.tensors[tid].get("size", 0)
                for pos, amount in totals.items():
                    peak[pos] = max(peak[pos], amount)
                for tid, row in self.tensor_modules.items():
                    if mid in row["producers"] and consumers_by_tensor[tid]:
                        live.add(tid)
                for tid in list(live):
                    if consumers_by_tensor[tid] and max(
                            self.module_topo.index(m) for m in consumers_by_tensor[tid]) == \
                            self.module_topo.index(mid):
                        live.discard(tid)
            peaks.append(dict(peak))
        return peaks

    def _make_balanced_groups(self, task_count):
        if task_count <= 0:
            return [], []
        bins = [[] for _ in range(task_count)]
        bin_pipe = [defaultdict(int) for _ in bins]
        bin_work = [0] * task_count
        bin_modules = [set() for _ in bins]
        module_order = sorted(
            range(len(self.modules)),
            key=lambda mid: (-self.module_features[mid]["critical_path"],
                             -max(self.module_features[mid]["pipe"].values(), default=0),
                             -self.module_features[mid]["cycles"], mid))
        for mid in module_order:
            feature = self.module_features[mid]
            choices = []
            for task_id in range(task_count):
                pipe_after = dict(bin_pipe[task_id])
                for name, amount in feature["pipe"].items():
                    pipe_after[name] = pipe_after.get(name, 0) + amount
                duration_proxy = max(pipe_after.values(), default=0)
                module_cache = max(feature["working_set_by_pos"].values(), default=0)
                overflow = sum(max(0, feature["working_set_by_pos"].get(pos, 0)
                                   - self.capacity.get(pos, 0))
                               for pos in ("L1", "UB"))
                choices.append((bin_work[task_id] + duration_proxy + overflow,
                                bin_work[task_id], len(bins[task_id]), task_id))
            _, _, _, chosen = min(choices)
            bins[chosen].append(mid)
            bin_modules[chosen].add(mid)
            bin_work[chosen] += feature["cycles"]
            for name, amount in feature["pipe"].items():
                bin_pipe[chosen][name] += amount
        groups = [sorted(bucket) for bucket in bins]
        if self._module_partition_acyclic(groups):
            return groups, []
        return self._topological_fallback(task_count), ["lpt_assignment_repaired_to_topological_ranges"]

    def _module_partition_acyclic(self, groups):
        owner = {mid: tid for tid, group in enumerate(groups) for mid in group}
        succ = {tid: set() for tid in range(len(groups))}
        indegree = {tid: 0 for tid in range(len(groups))}
        for src, children in self.module_succ.items():
            for dst in children:
                a, b = owner[src], owner[dst]
                if a != b:
                    succ[a].add(b)
        for children in succ.values():
            for dst in children:
                indegree[dst] += 1
        ready = deque(tid for tid, degree in indegree.items() if degree == 0)
        count = 0
        while ready:
            tid = ready.popleft()
            count += 1
            for dst in succ[tid]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
        return count == len(groups)

    def _topological_fallback(self, task_count):
        """Partition topo order into work-balanced consecutive convex ranges."""
        if task_count >= len(self.modules):
            return [[mid] for mid in self.module_topo]
        total = sum(feature["cycles"] for feature in self.module_features)
        groups, current, current_work = [], [], 0
        remaining_work = total
        remaining_bins = task_count
        for index, mid in enumerate(self.module_topo):
            work = self.module_features[mid]["cycles"]
            if current and remaining_bins > 1 and current_work + work > remaining_work / remaining_bins:
                groups.append(current)
                remaining_work -= current_work
                remaining_bins -= 1
                current, current_work = [], 0
            current.append(mid)
            current_work += work
        if current:
            groups.append(current)
        while len(groups) > task_count:
            # Merge the adjacent pair with least combined estimated work.
            costs = [sum(self.module_features[mid]["cycles"] for mid in groups[i] + groups[i + 1])
                     for i in range(len(groups) - 1)]
            at = min(range(len(costs)), key=lambda i: (costs[i], i))
            groups[at:at + 2] = [groups[at] + groups[at + 1]]
        return groups

    def _objective(self, groups):
        op_groups = [[op for mid in mids for op in self.modules[mid]] for mids in groups]
        deps = validate_partition(self.g, op_groups)
        plan, estimate = make_plan(
            self.g, op_groups, self.num_cores, bandwidth=self.bandwidth,
            cross_wait=self.cross_wait, same_wait=self.same_wait)
        estimated_makespan = max(estimate["task_estimated_finish"].values(), default=0)
        boundary_bytes, task_boundary_bytes = self._boundary_bytes(groups)
        original_bytes = 0
        for op_id, op in self.g.ops.items():
            if op.get("op") == "COPY_IN":
                original_bytes += sum(self.g.tensors[t].get("size", 0)
                                      for t in self.g.op_outputs[op_id])
            elif op.get("op") == "COPY_OUT":
                original_bytes += sum(self.g.tensors[t].get("size", 0)
                                      for t in self.g.op_inputs[op_id])
        added_bytes = max(0, boundary_bytes - original_bytes)
        peaks = self._cache_peaks(groups)
        overflow_bytes = sum(
            max(0, peak.get(pos, 0) - self.capacity.get(pos, 0))
            for peak in peaks for pos in ("L1", "UB"))
        cache_penalty_cycles = overflow_bytes / max(1, self.bandwidth)
        communication_penalty_cycles = (
            self.COMMUNICATION_WEIGHT * added_bytes / max(1, self.bandwidth))
        objective = estimated_makespan + communication_penalty_cycles + cache_penalty_cycles
        return {
            "objective_cycles": objective,
            "estimated_makespan_cycles": estimated_makespan,
            "estimated_added_boundary_bytes": added_bytes,
            "estimated_boundary_bytes": boundary_bytes,
            "communication_penalty_cycles": communication_penalty_cycles,
            "cache_peak_by_task": peaks,
            "cache_overflow_bytes": overflow_bytes,
            "cache_penalty_cycles": cache_penalty_cycles,
            "task_pipe_work": estimate["task_estimated_duration"],
            "task_boundary_bytes_proxy": estimate["task_boundary_bytes_proxy"],
            "dependency_edges": sum(map(len, deps.values())),
            "plan": plan,
        }

    def run(self):
        if not self.modules:
            return [], {"selected_task_count": 0, "candidates": []}
        candidates = []
        for task_count in self._candidate_task_counts():
            module_groups, repairs = self._make_balanced_groups(task_count)
            op_groups = [[op for mid in mids for op in self.modules[mid]]
                         for mids in module_groups]
            validate_partition(self.g, op_groups)
            objective = self._objective(module_groups)
            candidates.append({
                "task_count": len(module_groups),
                "requested_task_count": task_count,
                "repair_notes": repairs,
                "objective_cycles": objective["objective_cycles"],
                "estimated_makespan_cycles": objective["estimated_makespan_cycles"],
                "estimated_added_boundary_bytes": objective["estimated_added_boundary_bytes"],
                "communication_penalty_cycles": objective["communication_penalty_cycles"],
                "cache_overflow_bytes": objective["cache_overflow_bytes"],
                "cache_penalty_cycles": objective["cache_penalty_cycles"],
                "dependency_edges": objective["dependency_edges"],
            })
        chosen = min(candidates, key=lambda row: (
            row["objective_cycles"], row["estimated_makespan_cycles"],
            row["estimated_added_boundary_bytes"], row["task_count"]))
        groups, repairs = self._make_balanced_groups(chosen["requested_task_count"])
        op_groups = [[op for mid in mids for op in self.modules[mid]] for mids in groups]
        validate_partition(self.g, op_groups)
        details = {
            "selected_task_count": len(op_groups),
            "modules_per_task_min": min(map(len, groups)),
            "modules_per_task_max": max(map(len, groups)),
            "modules_per_task_mean": sum(map(len, groups)) / len(groups),
            "selected_candidate": chosen,
            "candidate_objectives": candidates,
            "repair_notes": repairs,
            "core_count": self.num_cores,
            "wait_cycles": {"same_core": self.same_wait, "cross_core": self.cross_wait},
            "bandwidth_bytes_per_cycle": self.bandwidth,
            "capacity_bytes": self.capacity,
            "objective_definition": (
                "HEFT proxy makespan + 0.25 * estimated added boundary bytes / DDR bandwidth "
                "+ cache-overflow bytes / DDR bandwidth"),
        }
        return op_groups, details
