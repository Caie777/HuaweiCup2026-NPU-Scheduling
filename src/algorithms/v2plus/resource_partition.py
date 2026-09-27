"""Resource-aware Task partition base used by V2plus."""
from __future__ import annotations
import heapq
from collections import defaultdict, deque
from common.partition import validate_partition
from common.schedule import make_plan


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
