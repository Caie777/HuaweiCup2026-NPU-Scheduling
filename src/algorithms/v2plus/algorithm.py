"""V2plus: deterministic two-level aggregation with Step1-aware memory risk.

This route decides a small set of Task partitions before multi-core scheduling.
It deliberately does not perform schedule-driven merge/move/reorder rounds.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

from common.partition import validate_partition
from algorithms.v2plus.resource_partition import ResourceAwareAggregator
from common.memory import estimate_task, estimate_task_step1, index_compute_edges
from common.schedule import make_plan


class V2PlusAggregator(ResourceAwareAggregator):
    def __init__(self, graph, modules, num_cores, *, bandwidth, cross_wait,
                 same_wait, capacity):
        super().__init__(graph, modules, num_cores, bandwidth=bandwidth,
                         cross_wait=cross_wait, same_wait=same_wait, capacity=capacity)
        self._risk_cache = {}
        self._step1_edge_index = index_compute_edges(graph)

    def _counts(self):
        m, p = len(self.modules), self.num_cores
        if m <= 1:
            return [1]
        counts = {m, min(m, p)}
        k = min(m, p)
        while k < m:
            counts.add(k)
            counts.add(min(m, max(k + 1, math.ceil(1.5 * k))))
            k = min(m, 2 * k)
        return sorted(counts)

    def _affinity_groups(self, task_count):
        """Place natural modules using load plus shared Tensor reuse."""
        if task_count >= len(self.modules):
            return [[mid] for mid in self.module_topo]
        bins = [[] for _ in range(task_count)]
        pipe = [defaultdict(int) for _ in bins]
        touched = [set() for _ in bins]
        order = sorted(self.module_topo, key=lambda mid: (
            -self.module_features[mid]["critical_path"],
            -self.module_features[mid]["cycles"], mid))
        for mid in order:
            feature = self.module_features[mid]
            choices = []
            for task in range(task_count):
                after = dict(pipe[task])
                for name, work in feature["pipe"].items():
                    after[name] = after.get(name, 0) + work
                work_after = max(after.values(), default=0)
                common = feature["touched_tensors"] & touched[task]
                shared_bytes = sum(self.g.tensors[t].get("size", 0) for t in common)
                # The load term keeps unrelated branches balanced; the shared
                # term discounts a Tensor already loaded inside this Task.
                score = (work_after - 0.25 * shared_bytes / self.bandwidth,
                         sum(pipe[task].values()), len(bins[task]), task)
                choices.append(score)
            chosen = min(range(task_count), key=lambda t: choices[t])
            bins[chosen].append(mid)
            touched[chosen].update(feature["touched_tensors"])
            for name, work in feature["pipe"].items():
                pipe[chosen][name] += work
        groups = [sorted(b) for b in bins if b]
        if self._module_partition_acyclic(groups):
            return groups
        return self._topological_fallback(task_count)

    def _split_large_natural_module(self, count):
        """A convex topological cut is a safe fallback for one huge module."""
        order = list(self.g.compute_order)
        if count <= 1 or count >= len(order):
            return [[op] for op in order] if count >= len(order) else [order]
        total = sum(self.g.ops[op].get("cycles", 0) for op in order)
        target = total / count
        groups = []; current = []; work = 0
        for index, op in enumerate(order):
            if current and len(groups) < count - 1 and work >= target:
                groups.append(current); current = []; work = 0
            current.append(op); work += self.g.ops[op].get("cycles", 0)
        if current:
            groups.append(current)
        while len(groups) > count:
            groups[-2].extend(groups[-1]); groups.pop()
        return groups

    def _risk(self, group):
        key = tuple(sorted(group))
        if key not in self._risk_cache:
            try:
                risk = estimate_task_step1(self.g, group, self.capacity,
                                           edge_index=self._step1_edge_index)
                source = "official_step1_induced_graph"
            except Exception as exc:
                risk = estimate_task(self.g, group, self.capacity)
                source = f"topological_fallback:{type(exc).__name__}"
            self._risk_cache[key] = (risk, source)
        return self._risk_cache[key]

    def _score(self, groups, source):
        validate_partition(self.g, groups)
        plan, schedule = make_plan(
            self.g, groups, self.num_cores, bandwidth=self.bandwidth,
            cross_wait=self.cross_wait, same_wait=self.same_wait)
        proxy = max(schedule["task_estimated_finish"].values(), default=0)
        # Count boundary copies per distinct Task, including shared Tensor
        # fan-out. This is exact for partition COPY in scene A.
        module_groups = []
        for group in groups:
            module_groups.append(sorted({self.module_of[op] for op in group}))
        if len(self.modules) == 1 and len(groups) > 1:
            added_bytes = 0
            # The single-module fallback is scored from its Task boundary
            # directly, since one natural module now spans several Tasks.
            owner = {op: task for task, group in enumerate(groups) for op in group}
            for tid, row in self.tensor_modules.items():
                producers = {owner[o] for o in self.g.tensor_producers[tid] if o in owner}
                consumers = {owner[o] for o in self.g.tensor_consumers[tid] if o in owner}
                copies = sum(task not in producers for task in consumers)
                copies += sum(row["original_copy_out"] or not consumers or
                              any(c != task for c in consumers) for task in producers)
                copies -= row["original_copy_in"] + int(row["original_copy_out"])
                added_bytes += copies * self.g.tensors[tid].get("size", 0)
        else:
            boundary, _ = self._boundary_bytes(module_groups)
            original = 0
            for tid, row in self.tensor_modules.items():
                original += self.g.tensors[tid].get("size", 0) * (
                    row["original_copy_in"] + int(row["original_copy_out"]))
            added_bytes = boundary - original
        risks = []; potential_spill = 0
        for group in groups:
            risk, risk_source = self._risk(group)
            overflow = sum(risk["overflow_bytes"].values())
            # Long Task residency multiplies refill cost. The scale depends
            # on Op count, never case identity or a fixed modules/Task size.
            repeat = min(12.0, 2.0 + len(group) / 256.0)
            potential_spill += overflow * repeat
            risks.append({"peak": risk["peak_live_bytes"],
                          "overflow": risk["overflow_bytes"],
                          "source": risk_source,
                          "high_risk_tensors": risk.get("high_risk_tensors", [])[:5]})
        objective = proxy + 0.25 * max(0, added_bytes) / self.bandwidth
        objective += potential_spill / self.bandwidth
        return {"groups": [list(g) for g in groups], "plan": plan,
                "source": source, "task_count": len(groups),
                "proxy_makespan": proxy,
                "added_copy_bytes_proxy": max(0, added_bytes),
                "potential_spill_bytes": potential_spill,
                "task_risks": risks, "objective": objective}

    def run(self):
        started = time.monotonic()
        proposals = {}
        def add(groups, source):
            if not groups or any(not group for group in groups):
                return
            key = tuple(sorted(tuple(sorted(group)) for group in groups))
            proposals.setdefault(key, (groups, source))
        if len(self.modules) == 1:
            add([list(self.modules[0])], "natural_module")
            for count in sorted({self.num_cores, 2 * self.num_cores}):
                if count > 1:
                    add(self._split_large_natural_module(count), f"single_module_split_{count}")
        else:
            for count in self._counts():
                for kind, module_groups in (
                    ("work_balanced", self._make_balanced_groups(count)[0]),
                    ("tensor_affinity", self._affinity_groups(count)),
                    ("convex_topological", self._topological_fallback(count))):
                    groups = [[op for mid in mids for op in self.modules[mid]]
                              for mids in module_groups]
                    add(groups, f"{kind}_{count}")
        scored = []
        for groups, source in proposals.values():
            try:
                scored.append(self._score(groups, source))
            except Exception as exc:
                scored.append({"source": source, "task_count": len(groups),
                               "error": f"{type(exc).__name__}: {exc}"})
        valid = [r for r in scored if "error" not in r]
        if not valid:
            groups = [list(module) for module in self.modules]
            valid = [self._score(groups, "safe_natural_modules")]
        valid.sort(key=lambda r: (r["objective"], r["proxy_makespan"],
                                  r["added_copy_bytes_proxy"], r["task_count"], r["source"]))
        return valid[0]["groups"], {"route": "V2+", "selected": valid[0],
                                  "ranked_candidates": valid,
                                  "failed_candidates": [r for r in scored if "error" in r],
                                  "candidate_count": len(scored),
                                  "algorithm_seconds": time.monotonic() - started}
