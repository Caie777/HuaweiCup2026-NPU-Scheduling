"""Deterministic resource-aware multilevel partitioning and scheduling.

RAMP-DAG keeps the V1 natural-module partition as its starting level, explores
several workload-independent coarsening profiles, and can split a coarse task
at a topological cut during refinement. All cost estimates in this module are
proxies; official problem-1 evaluation remains authoritative.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from bisect import bisect_left
from collections import defaultdict, deque

from aggregate import validate_partition
from aggregate_v2 import FastV1ModuleAggregator


PROFILES = {
    "balanced": {"comm": 1.0, "parallel": 0.20, "memory": 1.0, "complement": 0.35},
    "communication_heavy": {"comm": 2.0, "parallel": 0.0001, "memory": 0.8, "complement": 0.20},
    "parallelism_heavy": {"comm": 0.65, "parallel": 0.55, "memory": 0.8, "complement": 0.65},
    "memory_safe": {"comm": 0.8, "parallel": 0.20, "memory": 2.5, "complement": 0.25},
}


def canonical_partition(groups):
    return tuple(sorted(tuple(sorted(set(g))) for g in groups if g))


def partition_hash(groups):
    payload = json.dumps(canonical_partition(groups), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha_plan(plan):
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


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


class _Coarsener:
    """Bounded-candidate multilevel contractions over natural modules."""

    def __init__(self, graph, modules, profile_name, profile, bandwidth, capacity, num_cores,
                 ablations=None):
        self.g, self.modules = graph, modules
        self.profile_name, self.w = profile_name, profile
        self.bandwidth, self.capacity = bandwidth, capacity
        self.num_cores = num_cores
        self.ablations = ablations or {}
        if self.ablations.get("memory_penalty_disabled"):
            self.w = {**profile, "memory": 0.0}
        self.module_of = {op: mid for mid, group in enumerate(modules) for op in group}
        self.msucc = {i: set() for i in range(len(modules))}
        self.mpred = {i: set() for i in range(len(modules))}
        for src in graph.compute_ids:
            a = self.module_of[src]
            for dst in graph.compute_succ[src]:
                b = self.module_of[dst]
                if a != b:
                    self.msucc[a].add(b)
                    self.mpred[b].add(a)
        self.mod_topo = self._topo(self.msucc)
        self.mod_pos = {m: i for i, m in enumerate(self.mod_topo)}
        self.features = [self._feature(g) for g in modules]
        self.tensor_modules = self._tensor_views()

    @staticmethod
    def _topo(succ):
        indeg = {n: 0 for n in succ}
        for children in succ.values():
            for c in children:
                indeg[c] += 1
        q = [n for n, d in indeg.items() if d == 0]
        heapq.heapify(q)
        out = []
        while q:
            n = heapq.heappop(q); out.append(n)
            for c in sorted(succ[n]):
                indeg[c] -= 1
                if indeg[c] == 0:
                    heapq.heappush(q, c)
        if len(out) != len(succ):
            raise ValueError("自然模块依赖图存在环")
        return out

    def _feature(self, ops):
        pipe = defaultdict(int); tensors = set()
        for opid in ops:
            op = self.g.ops[opid]
            pipe[op.get("pipe", "UNKNOWN")] += op.get("cycles", 0)
            tensors.update(self.g.op_inputs[opid]); tensors.update(self.g.op_outputs[opid])
        by_pos = defaultdict(int)
        for tid in tensors:
            pos = self.g.tensors[tid].get("pos", "UNKNOWN")
            by_pos["UB" if pos == "DDR" else pos] += self.g.tensors[tid].get("size", 0)
        return {"pipe": dict(pipe), "work": sum(pipe.values()), "tensors": tensors,
                "cache": dict(by_pos), "ops": len(ops)}

    def _tensor_views(self):
        comp = set(self.g.compute_ids); rows = {}
        for tid in self.g.tensors:
            ps = self.g.tensor_producers[tid] & comp
            cs = self.g.tensor_consumers[tid] & comp
            rows[tid] = {
                "producers": {self.module_of[o] for o in ps},
                "consumers": {self.module_of[o] for o in cs},
                "bytes": self.g.tensors[tid].get("size", 0),
                "external_in": not ps and bool(cs),
                "copy_out": any(self.g.ops[o].get("op") == "COPY_OUT"
                                for o in self.g.tensor_consumers[tid]),
            }
        return rows

    def _candidate_pairs(self, clusters, owner):
        pairs = set()
        active = set(clusters)
        if not self.ablations.get("dependency_only"):
            for a in active:
                for b in self.msucc[a]:
                    x, y = owner[a], owner[b]
                    if x != y:
                        pairs.add(tuple(sorted((x, y))))
        else:
            for a in active:
                for b in self.msucc[a]:
                    x, y = owner[a], owner[b]
                    if x != y:
                        pairs.add(tuple(sorted((x, y))))

        if not self.ablations.get("shared_tensor") and not self.ablations.get("dependency_only"):
            # Treat fan-out as a hyperedge. Candidate generation is linear in
            # fan-out after sorting; it never materializes all O(fanout^2) pairs.
            for row in self.tensor_modules.values():
                consumers = sorted({owner[m] for m in row["consumers"]},
                                   key=lambda c: (min(self.mod_pos[m] for m in clusters[c]), c))
                if len(consumers) > 1:
                    for a, b in zip(consumers, consumers[1:]):
                        if a != b:
                            pairs.add(tuple(sorted((a, b))))
                producers = {owner[m] for m in row["producers"]}
                # Producer/consumer candidates are already covered by the
                # module dependency edges above. Bound additional candidates
                # for a pathological many-producer tensor.
                for p in sorted(producers)[:8]:
                    for c in consumers[:8]:
                        if p != c:
                            pairs.add(tuple(sorted((p, c))))

        if not self.ablations.get("dependency_only"):
            # Add a constant-size neighborhood of work/Pipe-complement pairs.
            ids = sorted(active)
            feature = {a: self._cluster_feature(clusters[a]) for a in ids}
            by_work = sorted(ids, key=lambda a: (feature[a]["work"], a))
            work_values = [feature[a]["work"] for a in by_work]
            for a in ids:
                at = bisect_left(work_values, feature[a]["work"])
                for pos in (at - 2, at - 1, at, at + 1, at + 2):
                    if 0 <= pos < len(by_work) and by_work[pos] != a:
                        pairs.add(tuple(sorted((a, by_work[pos]))))
            # Add the nearest workload candidate with a complementary dominant
            # Pipe using sorted buckets, keeping the candidate count O(m).
            dominant = {a: max(feature[a]["pipe"], key=feature[a]["pipe"].get, default="")
                        for a in ids}
            buckets = defaultdict(list)
            for a in ids:
                buckets[dominant[a]].append(a)
            for bucket in buckets.values():
                bucket.sort(key=lambda a: (feature[a]["work"], a))
            for a in ids:
                alternatives = [bucket for p, bucket in buckets.items()
                                if p and p != dominant[a]]
                if alternatives:
                    best = None
                    best_key = None
                    for bucket in alternatives:
                        values = [feature[b]["work"] for b in bucket]
                        pos = bisect_left(values, feature[a]["work"])
                        for at in (pos - 1, pos):
                            if 0 <= at < len(bucket):
                                b = bucket[at]
                                key = (abs(math.log((feature[a]["work"] + 1) /
                                                    (feature[b]["work"] + 1))), b)
                                if best_key is None or key < best_key:
                                    best, best_key = b, key
                    if best is not None:
                        pairs.add(tuple(sorted((a, best))))
        return pairs

    def _cluster_feature(self, mids):
        pipes = defaultdict(int); cache = defaultdict(int); ops = work = 0
        for m in mids:
            f = self.features[m]; work += f["work"]; ops += f["ops"]
            for p, v in f["pipe"].items(): pipes[p] += v
            # Module working sets are not simultaneously live by default.
            # Use a max as a cheap coarsening pre-screen; full Task liveness is
            # recomputed in _task_metrics for every retained partition.
            for p, v in f["cache"].items(): cache[p] = max(cache[p], v)
        return {"pipe": dict(pipes), "cache": dict(cache), "work": work, "ops": ops}

    def _merge_score(self, a, b, clusters, owner):
        fa, fb = self._cluster_feature(clusters[a]), self._cluster_feature(clusters[b])
        shared_bytes = self._pair_savings.get((min(a, b), max(a, b)), 0)
        internalized_edges = 0
        shared_saving_cycles = shared_bytes / max(1, self.bandwidth)
        boundary_saving_cycles = internalized_edges / max(1, self.bandwidth)
        b_members = set(clusters[b])
        dep_edges = sum(1 for x in clusters[a] for y in self.msucc[x]
                        if y in b_members) + sum(1 for x in clusters[b] for y in self.msucc[x]
                                                 if y in clusters[a])
        wait_saving = dep_edges * 100.0
        combined_work = fa["work"] + fb["work"]
        lost_parallel = min(fa["work"], fb["work"])
        pipes_a, pipes_b = set(fa["pipe"]), set(fb["pipe"])
        complementary = bool(pipes_a and pipes_b and pipes_a.isdisjoint(pipes_b))
        cache_after = defaultdict(int)
        for pos in set(fa["cache"]) | set(fb["cache"]):
            cache_after[pos] = max(fa["cache"].get(pos, 0), fb["cache"].get(pos, 0))
        mem_over = sum(max(0, cache_after[pos] - self.capacity.get(pos, 0))
                       for pos in ("L1", "UB"))
        mem_before = sum(max(0, f["cache"].get(pos, 0) - self.capacity.get(pos, 0))
                         for f in (fa, fb) for pos in ("L1", "UB"))
        memory_saving = (mem_before - mem_over) / max(1, self.bandwidth)
        score = (self.w["comm"] * (shared_saving_cycles + boundary_saving_cycles)
                 + wait_saving
                 + self.w["parallel"] * 100.0 * (1.0 - 1.0 / max(1, self.num_cores))
                 + self.w["complement"] * (0.08 * min(fa["work"], fb["work"])
                                             if complementary else 0)
                 - self.w["parallel"] * 0.12 * lost_parallel
                 - self.w["memory"] * mem_over / max(1, self.bandwidth)
                 + 0.05 * memory_saving)
        return score

    @staticmethod
    def _reachable(start, target, succ, forbidden):
        todo = list(start); seen = set()
        while todo:
            n = todo.pop()
            if n == target:
                return True
            if n in seen or n == forbidden:
                continue
            seen.add(n); todo.extend(succ.get(n, ()))
        return False

    def _can_merge(self, a, b, succ):
        # Contracting A/B creates a cycle iff there is an alternate directed
        # path between them with at least one third component in its interior.
        if b in succ[a]:
            if self._reachable((x for x in succ[a] if x != b), b, succ, forbidden=a):
                return False
        elif a in succ[b]:
            if self._reachable((x for x in succ[b] if x != a), a, succ, forbidden=b):
                return False
        else:
            # A non-edge merge is legal only if neither node is connected by a path.
            if self._reachable(succ[a], b, succ, forbidden=a):
                return False
            if self._reachable(succ[b], a, succ, forbidden=b):
                return False
        return True

    @staticmethod
    def _contract(a, b, clusters, owner, succ, pred):
        rep = min(a, b); other = max(a, b)
        members = clusters[rep] | clusters[other]
        incoming = (pred[rep] | pred[other]) - {rep, other}
        outgoing = (succ[rep] | succ[other]) - {rep, other}
        for x in incoming:
            succ[x].discard(rep); succ[x].discard(other); succ[x].add(rep)
        for x in outgoing:
            pred[x].discard(rep); pred[x].discard(other); pred[x].add(rep)
        for x in (rep, other):
            succ.pop(x, None); pred.pop(x, None); clusters.pop(x, None)
        clusters[rep] = members; succ[rep] = outgoing; pred[rep] = incoming
        for m in members:
            owner[m] = rep
        return rep

    def run(self, deadline=None):
        clusters = {i: {i} for i in range(len(self.modules))}
        owner = {i: i for i in clusters}
        succ = {i: set(self.msucc[i]) for i in clusters}
        pred = {i: set(self.mpred[i]) for i in clusters}
        snapshots = [{"profile": self.profile_name, "level": 0,
                      "module_groups": [list(x) for _, x in sorted(clusters.items())],
                      "merge_log": []}]
        level = 0
        while len(clusters) > 1:
            if deadline and time.monotonic() >= deadline:
                break
            pairs = self._candidate_pairs(clusters, owner)
            # Tensor savings are accumulated by Task-level hyperedge fan-out,
            # not consumer Op count, and only for the bounded candidate pairs.
            pair_savings = defaultdict(int)
            for row in self.tensor_modules.values():
                consumers = sorted({owner[m] for m in row["consumers"]},
                                   key=lambda c: (min(self.mod_pos[m] for m in clusters[c]), c))
                for a, b in zip(consumers, consumers[1:]):
                    pair = tuple(sorted((a, b)))
                    if pair in pairs and not ({a, b} & {owner[m] for m in row["producers"]}):
                        pair_savings[pair] += row["bytes"]
                consumer_modules = row["consumers"]
                for pmod in row["producers"]:
                    for cmod in self.msucc[pmod] & consumer_modules:
                        p, c = owner[pmod], owner[cmod]
                        pair = tuple(sorted((p, c)))
                        if p != c and pair in pairs:
                            pair_savings[pair] += row["bytes"]
            self._pair_savings = pair_savings
            ranked = sorted(((self._merge_score(a, b, clusters, owner), a, b)
                             for a, b in pairs if a in clusters and b in clusters),
                            key=lambda x: (-x[0], x[1], x[2]))
            merged = set(); logs = []
            for score, a, b in ranked:
                if a in merged or b in merged or a not in clusters or b not in clusters:
                    continue
                if score <= 0 or not self._can_merge(a, b, succ):
                    continue
                fa, fb = self._cluster_feature(clusters[a]), self._cluster_feature(clusters[b])
                rep = self._contract(a, b, clusters, owner, succ, pred)
                merged.add(rep)
                logs.append({"left": sorted(clusters[rep]), "score": score,
                             "shared_bytes_saved_est": max(0, score),
                             "combined_work": fa["work"] + fb["work"],
                             "reason": "dependency/shared-Tensor/Pipe-complement candidate"})
            if not logs:
                break
            level += 1
            groups = [sorted(x) for _, x in sorted(clusters.items())]
            snapshots.append({"profile": self.profile_name, "level": level,
                              "module_groups": groups, "merge_log": logs[:30]})
            # Defensive quotient validation after each coarsening level.
            self._validate_module_groups(groups)
        return snapshots

    def _validate_module_groups(self, groups):
        owner = {m: i for i, group in enumerate(groups) for m in group}
        succ = {i: set() for i in range(len(groups))}
        for a, children in self.msucc.items():
            for b in children:
                x, y = owner[a], owner[b]
                if x != y:
                    succ[x].add(y)
        self._topo(succ)


class RAMPDAG:
    """Multilevel partition search, deterministic HEFT-style placement and refinement."""

    def __init__(self, graph, num_cores, *, bandwidth=60, same_wait=100,
                 cross_wait=1000, capacity=None, ablations=None):
        if num_cores < 1:
            raise ValueError("核心数必须为正")
        self.g, self.k = graph, num_cores
        self.bandwidth = max(1, bandwidth)
        self.same_wait, self.cross_wait = same_wait, cross_wait
        self.capacity = dict(capacity or {"L1": 524288, "UB": 131072})
        self.ablations = dict(ablations or {})
        self.started = time.monotonic()
        phase_started = time.monotonic()
        self.modules = FastV1ModuleAggregator(graph).run()
        self.phase_seconds = {"natural_module_aggregation": time.monotonic() - phase_started}
        self.module_of = {op: m for m, group in enumerate(self.modules) for op in group}
        self.op_pos = {op: i for i, op in enumerate(graph.compute_order)}
        self.module_order = sorted(range(len(self.modules)),
                                   key=lambda m: min(self.op_pos[o] for o in self.modules[m]))
        self.module_order_pos = {m: i for i, m in enumerate(self.module_order)}
        phase_started = time.monotonic()
        self.tensor_views = self._tensor_views()
        self.phase_seconds["tensor_view_construction"] = time.monotonic() - phase_started

    def _tensor_views(self):
        comp = set(self.g.compute_ids); rows = {}
        for tid, t in self.g.tensors.items():
            ps = self.g.tensor_producers[tid] & comp
            cs = self.g.tensor_consumers[tid] & comp
            rows[tid] = {"producers": ps, "consumers": cs,
                         "producer_modules": {self.module_of[x] for x in ps},
                         "consumer_modules": {self.module_of[x] for x in cs},
                         "bytes": t.get("size", 0), "pos": t.get("pos", "UNKNOWN"),
                         "external_in": not ps and bool(cs),
                         "copy_out": any(self.g.ops[x].get("op") == "COPY_OUT"
                                         for x in self.g.tensor_consumers[tid])}
        return rows

    @staticmethod
    def _module_partition_acyclic(coarsener, module_groups):
        owner = {m: t for t, group in enumerate(module_groups) for m in group}
        succ = {t: set() for t in range(len(module_groups))}
        indeg = {t: 0 for t in range(len(module_groups))}
        for a, children in coarsener.msucc.items():
            for b in children:
                x, y = owner[a], owner[b]
                if x != y:
                    succ[x].add(y)
        for children in succ.values():
            for child in children:
                indeg[child] += 1
        ready = [t for t, d in indeg.items() if d == 0]
        heapq.heapify(ready); seen = 0
        while ready:
            t = heapq.heappop(ready); seen += 1
            for child in succ[t]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    heapq.heappush(ready, child)
        return seen == len(module_groups)

    def _rebalance_modules(self, coarsener, requested_count):
        """Split coarse clusters at natural-module cuts, then LPT-pack them.

        The requested number is the number of clusters in a coarsening level,
        not the core count. A full quotient check guards the re-packing; cyclic
        proposals fall back to convex topological ranges.
        """
        nmods = len(self.modules)
        count = max(1, min(requested_count, nmods))
        feature = coarsener.features
        order = sorted(range(nmods), key=lambda m: (
            -max(feature[m]["pipe"].values(), default=0), -feature[m]["work"], m))
        bins = [[] for _ in range(count)]
        bin_pipe = [defaultdict(int) for _ in bins]
        bin_work = [0] * count
        for mid in order:
            f = feature[mid]
            choices = []
            for task in range(count):
                next_pipe = dict(bin_pipe[task])
                for pipe, cycles in f["pipe"].items():
                    next_pipe[pipe] = next_pipe.get(pipe, 0) + cycles
                duration = max(next_pipe.values(), default=0)
                choices.append((duration, bin_work[task], len(bins[task]), task))
            _, _, _, chosen = min(choices)
            bins[chosen].append(mid)
            bin_work[chosen] += f["work"]
            for pipe, cycles in f["pipe"].items():
                bin_pipe[chosen][pipe] += cycles
        bins = [sorted(b) for b in bins]
        if self._module_partition_acyclic(coarsener, bins):
            return bins, "lpt-module-rebalance"

        # Consecutive ranges of a topological order induce a DAG quotient.
        topo = coarsener.mod_topo
        remaining = sum(feature[m]["work"] for m in topo)
        groups, current, work = [], [], 0
        bins_left = count
        for mid in topo:
            amount = feature[mid]["work"]
            if current and bins_left > 1 and work + amount > remaining / bins_left:
                groups.append(current); remaining -= work; bins_left -= 1
                current, work = [], 0
            current.append(mid); work += amount
        if current:
            groups.append(current)
        while len(groups) > count:
            idx = min(range(len(groups) - 1), key=lambda i: (
                sum(feature[m]["work"] for m in groups[i] + groups[i + 1]), i))
            groups[idx:idx + 2] = [groups[idx] + groups[idx + 1]]
        return groups, "convex-topological-fallback"

    def _task_metrics(self, groups):
        owner, succ, pred, topo = quotient_graph(self.g, groups)
        metrics = []
        for task_id, group in enumerate(groups):
            members = set(group); pipe = defaultdict(int)
            local_order = sorted(members, key=self.op_pos.__getitem__)
            cp = {}
            for op in local_order:
                cp[op] = self.g.ops[op].get("cycles", 0) + max(
                    (cp[p] for p in self.g.compute_pred[op] if p in members), default=0)
                pipe[self.g.ops[op].get("pipe", "UNKNOWN")] += self.g.ops[op].get("cycles", 0)
            in_tids, out_tids = set(), set()
            touched = set()
            for op in members:
                touched.update(self.g.op_inputs[op]); touched.update(self.g.op_outputs[op])
            for tid in touched:
                row = self.tensor_views[tid]
                if row["consumers"] & members and not row["producers"] & members:
                    in_tids.add(tid)
                if row["producers"] & members and (row["copy_out"] or
                        not row["consumers"] or row["consumers"] - members):
                    out_tids.add(tid)
            input_bytes = sum(self.g.tensors[t].get("size", 0) for t in in_tids)
            output_bytes = sum(self.g.tensors[t].get("size", 0) for t in out_tids)
            peak = self._liveness_peak(members)
            overflow = sum(max(0, peak.get(pos, 0) - self.capacity.get(pos, 0))
                           for pos in ("L1", "UB"))
            work_by_pipe = dict(pipe)
            comp_cost = max(max(work_by_pipe.values(), default=0), max(cp.values(), default=0))
            mte2 = input_bytes / self.bandwidth
            mte3 = output_bytes / self.bandwidth
            spill_est = 2 * overflow
            duration = max(comp_cost, mte2, mte3) + spill_est / self.bandwidth
            metrics.append({"task": task_id, "ops": len(members), "pipe_work": work_by_pipe,
                            "compute_cycles_lb": comp_cost, "critical_path_cycles": max(cp.values(), default=0),
                            "copy_in_bytes": input_bytes, "copy_out_bytes": output_bytes,
                            "copy_in_cycles": mte2, "copy_out_cycles": mte3,
                            "memory_peak_est": peak, "spill_bytes_est": spill_est,
                            "duration_proxy": duration, "members": sorted(members)})
        return owner, succ, pred, topo, metrics

    def _liveness_peak(self, members):
        # Tensor lifetime under a valid internal topological order; each
        # external input enters local UB once, regardless of consumer Op count.
        live = set(); remaining = {}
        touched = set()
        for op in members:
            touched.update(self.g.op_inputs[op]); touched.update(self.g.op_outputs[op])
        for tid in touched:
            row = self.tensor_views[tid]
            local_consumers = row["consumers"] & members
            copyout_use = bool(row["producers"] & members) and row["copy_out"]
            use_count = len(local_consumers) + int(copyout_use)
            if use_count:
                remaining[tid] = use_count
        cur = defaultdict(int); peak = defaultdict(int)
        def add(tid):
            pos = self.g.tensors[tid].get("pos", "UNKNOWN")
            pos = "UB" if pos == "DDR" else pos
            cur[pos] += self.g.tensors[tid].get("size", 0)
        state = {}; postorder = []
        roots = sorted((o for o in members if not (self.g.compute_pred[o] & members)),
                       key=lambda o: (-self.g.longest_path[o], o))
        for root in roots:
            if state.get(root) == 2:
                continue
            stack = [(root, False)]
            while stack:
                node, exiting = stack.pop()
                if exiting:
                    state[node] = 2; postorder.append(node)
                    continue
                if state.get(node) == 2 or state.get(node) == 1:
                    continue
                state[node] = 1
                stack.append((node, True))
                children = sorted((c for c in self.g.compute_succ[node]
                                   if c in members and state.get(c) != 2),
                                  key=lambda c: (self.g.longest_path[c], c), reverse=True)
                for child in children:
                    if state.get(child) != 2:
                        stack.append((child, False))
        memory_order = list(reversed(postorder))
        if len(memory_order) != len(members):
            memory_order = sorted(members, key=self.op_pos.__getitem__)
        for op in memory_order:
            # Inputs become live at their first local read, not all at Task
            # start; otherwise independent weights look falsely simultaneous.
            for tid in self.g.op_inputs[op]:
                if tid in remaining and tid not in live:
                    live.add(tid); add(tid)
            # A produced activation overlaps its consumer's live inputs.
            for tid in self.g.op_outputs[op]:
                if tid in remaining and (self.tensor_views[tid]["consumers"] & members or
                                         self.tensor_views[tid]["copy_out"]) and tid not in live:
                    live.add(tid); add(tid)
            for pos, n in cur.items():
                peak[pos] = max(peak[pos], n)
            for tid in self.g.op_inputs[op]:
                if tid in remaining:
                    remaining[tid] -= 1
                    if remaining[tid] == 0 and tid in live:
                        live.remove(tid)
                        pos = self.g.tensors[tid].get("pos", "UNKNOWN")
                        pos = "UB" if pos == "DDR" else pos
                        cur[pos] -= self.g.tensors[tid].get("size", 0)
            # COPY_OUT has no compute Op consumer inside GraphModel; model its
            # one local read as immediately following the producing Op.
            for tid in self.g.op_outputs[op]:
                row = self.tensor_views[tid]
                if row["copy_out"] and not (row["consumers"] & members) and tid in remaining:
                    remaining[tid] -= 1
                    if remaining[tid] == 0 and tid in live:
                        live.remove(tid)
                        pos = self.g.tensors[tid].get("pos", "UNKNOWN")
                        pos = "UB" if pos == "DDR" else pos
                        cur[pos] -= self.g.tensors[tid].get("size", 0)
        for pos, n in cur.items():
            peak[pos] = max(peak[pos], n)
        return dict(peak)

    def _schedule(self, groups, mode="heft"):
        owner, succ, pred, topo, metrics = self._task_metrics(groups)
        rank = {}
        for t in reversed(topo):
            extra = max((self.cross_wait + rank[c] for c in succ[t]), default=0)
            rank[t] = metrics[t]["duration_proxy"] + extra
        if self.ablations.get("critical_path"):
            priority = lambda t: (t, -t)
        elif mode == "work":
            priority = lambda t: (sum(metrics[t]["pipe_work"].values()), -t)
        else:
            priority = lambda t: (rank[t], -t)
        pending = {t: len(pred[t]) for t in pred}
        ready = {t for t, d in pending.items() if d == 0}
        finish, placement = {}, {}
        schedules = [[] for _ in range(self.k)]
        core_ready = [0.0] * self.k
        starts = {}
        while ready:
            if mode == "topological" or self.ablations.get("critical_path"):
                task = min(ready)
            else:
                task = min(ready, key=lambda t: (-priority(t)[0], t))
            ready.remove(task)
            options = []
            for core in range(self.k):
                dep_ready = max((finish[p] + (0 if placement[p] == core else self.cross_wait)
                                 for p in pred[task]), default=0)
                serial_ready = core_ready[core] + (self.same_wait if schedules[core] else 0)
                start = max(dep_ready, serial_ready)
                options.append((start + metrics[task]["duration_proxy"], start, core))
            end, start, core = min(options)
            starts[task] = start; finish[task] = end; placement[task] = core
            schedules[core].append(task); core_ready[core] = end
            for child in succ[task]:
                pending[child] -= 1
                if pending[child] == 0:
                    ready.add(child)
        total_ddr = sum(m["copy_in_bytes"] + m["copy_out_bytes"] + m["spill_bytes_est"]
                        for m in metrics)
        base_finish = max(finish.values(), default=0)
        # The shared DDR lower bound captures aggregate contention. Per-task
        # MTE occupancy is already represented in duration_proxy.
        ddr_lb = total_ddr / self.bandwidth
        proxy = max(base_finish, ddr_lb)
        added_bytes = self._added_copy_bytes(groups)
        spill_est = sum(m["spill_bytes_est"] for m in metrics)
        memory_term = (0 if self.ablations.get("memory_penalty_disabled")
                       else spill_est / self.bandwidth)
        objective = proxy + 0.25 * added_bytes / self.bandwidth + memory_term
        plan = {"node_to_subgraph": {str(op): owner[op] for op in sorted(owner)},
                "core_schedules": schedules}
        details = {"task_metrics": metrics, "task_predecessors": {str(t): sorted(pred[t]) for t in pred},
                   "task_successors": {str(t): sorted(succ[t]) for t in succ},
                   "task_core": placement, "task_start_proxy": starts,
                   "task_finish_proxy": finish, "estimated_makespan_proxy": proxy,
                   "global_ddr_lower_bound_cycles": ddr_lb, "added_copy_bytes_proxy": added_bytes,
                   "spill_bytes_proxy": spill_est, "objective_proxy": objective,
                   "critical_rank": rank}
        return plan, details

    def _added_copy_bytes(self, groups):
        owner = {op: i for i, group in enumerate(groups) for op in group}
        total = 0
        for tid, row in self.tensor_views.items():
            producers = {owner[x] for x in row["producers"] if x in owner}
            consumers = {owner[x] for x in row["consumers"] if x in owner}
            for task in consumers:
                if task not in producers:
                    total += row["bytes"]
            for task in producers:
                if row["copy_out"] or not consumers or any(c != task for c in consumers):
                    total += row["bytes"]
        original = 0
        for opid, op in self.g.ops.items():
            if op.get("op") == "COPY_IN":
                original += sum(self.g.tensors[t].get("size", 0) for t in self.g.op_outputs[opid])
            elif op.get("op") == "COPY_OUT":
                original += sum(self.g.tensors[t].get("size", 0) for t in self.g.op_inputs[opid])
        return max(0, total - original)

    def _split_neighbors(self, groups, limit=3):
        # Split the largest few Tasks first. A contiguous topological cut is
        # convex, so its quotient remains acyclic; large natural modules may
        # be split back at Op boundaries.
        candidates = []
        ordered = sorted(enumerate(groups), key=lambda x: (-len(x[1]), x[0]))[:limit]
        for task_id, group in ordered:
            if len(group) < 2:
                continue
            task_modules = sorted({self.module_of[op] for op in group},
                                  key=lambda m: (self.module_order_pos[m], m))
            if len(task_modules) >= 2:
                work_by_module = {m: sum(self.g.ops[o].get("cycles", 0)
                                         for o in self.modules[m]) for m in task_modules}
                total_module_work = sum(work_by_module.values()); prefix = 0
                best_module_cut, best_gap = None, None
                for i, mid in enumerate(task_modules[:-1], 1):
                    prefix += work_by_module[mid]
                    gap = abs(total_module_work - 2 * prefix)
                    if best_gap is None or gap < best_gap:
                        best_gap, best_module_cut = gap, i
                left_modules, right_modules = task_modules[:best_module_cut], task_modules[best_module_cut:]
                left = [op for m in left_modules for op in self.modules[m]]
                right = [op for m in right_modules for op in self.modules[m]]
                proposal = [list(g) for g in groups]
                proposal[task_id:task_id + 1] = [left, right]
                try:
                    validate_partition(self.g, proposal)
                except ValueError:
                    pass
                else:
                    candidates.append((proposal, {"operation": "split_at_natural_module",
                        "task_id": task_id, "module_cut": best_module_cut,
                        "reason": "work-balanced natural-module boundary"}))
                    continue
            topo_ops = [op for op in self.g.compute_order if op in set(group)]
            total = sum(self.g.ops[o].get("cycles", 0) for o in topo_ops)
            prefix = 0; cut = None; best = None
            for i, op in enumerate(topo_ops[:-1], 1):
                prefix += self.g.ops[op].get("cycles", 0)
                key = abs(total - 2 * prefix)
                if best is None or key < best:
                    best, cut = key, i
            if cut is None:
                continue
            left, right = topo_ops[:cut], topo_ops[cut:]
            proposal = [list(g) for g in groups]
            proposal[task_id:task_id + 1] = [left, right]
            try:
                validate_partition(self.g, proposal)
            except ValueError:
                continue
            candidates.append((proposal, {"operation": "split", "task_id": task_id,
                                          "cut_ops": [left[-1], right[0]],
                                          "reason": "topological work-balanced cut"}))
        return candidates

    def candidates(self, deadline=None):
        out = {}
        def add(groups, label, log=None):
            key = partition_hash(groups)
            if key not in out:
                out[key] = {"groups": groups, "label": label, "decisions": log or []}
        add([list(x) for x in self.modules], "natural_modules")
        for name, profile in PROFILES.items():
            coarse = _Coarsener(self.g, self.modules, name, profile, self.bandwidth,
                                self.capacity, self.k, self.ablations)
            for snap in coarse.run(deadline=deadline):
                groups = [[op for m in mids for op in self.modules[m]]
                          for mids in snap["module_groups"]]
                add(groups, f"coarse:{name}:L{snap['level']}", snap["merge_log"])
                balanced, reason = self._rebalance_modules(coarse, len(snap["module_groups"]))
                balanced_ops = [[op for m in mids for op in self.modules[m]] for mids in balanced]
                add(balanced_ops, f"coarse:{name}:L{snap['level']}:{reason}", snap["merge_log"])
                level_count = len(snap["module_groups"])
                if level_count <= max(8, 2 * self.k + 2):
                    for neighbor_count in (level_count - 1, level_count + 1):
                        if 1 <= neighbor_count <= len(self.modules):
                            neighbor, why = self._rebalance_modules(coarse, neighbor_count)
                            neighbor_ops = [[op for m in mids for op in self.modules[m]]
                                            for mids in neighbor]
                            add(neighbor_ops,
                                f"coarse:{name}:L{snap['level']}:split-rebalance:{why}:{neighbor_count}",
                                snap["merge_log"] + [{"operation": "split_or_merge_rebalance",
                                    "from_tasks": level_count, "to_tasks": neighbor_count}])
                if deadline and time.monotonic() >= deadline:
                    break
        # One deterministic split neighborhood from each of the best proxy
        # partitions; the caller schedules and ranks all candidates.
        base = list(out.values())
        for row in base:
            if deadline and time.monotonic() >= deadline:
                break
            if self.ablations.get("no_refinement"):
                continue
            for groups, decision in self._split_neighbors(row["groups"]):
                add(groups, row["label"] + ":split", row["decisions"] + [decision])
        return list(out.values())

    def _evaluate_fixed_schedule(self, groups, plan):
        owner, succ, pred, _, metrics = self._task_metrics(groups)
        schedules = [list(core) for core in plan["core_schedules"]]
        if len(schedules) != self.k:
            return None
        flattened = [task for core in schedules for task in core]
        if sorted(flattened) != list(range(len(groups))) or len(flattened) != len(set(flattened)):
            return None
        core_of, previous = {}, {}
        for core, order in enumerate(schedules):
            for i, task in enumerate(order):
                core_of[task] = core
                if i:
                    previous[task] = order[i - 1]
        combined_pred = {t: set(pred[t]) for t in pred}
        combined_succ = {t: set(succ[t]) for t in succ}
        for task, prev in previous.items():
            if task != prev:
                combined_pred[task].add(prev)
                combined_succ[prev].add(task)
        indegree = {t: len(combined_pred[t]) for t in combined_pred}
        ready = [t for t, d in indegree.items() if d == 0]
        heapq.heapify(ready); topo = []
        while ready:
            task = heapq.heappop(ready); topo.append(task)
            for child in combined_succ[task]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(topo) != len(groups):
            return None
        finish, starts = {}, {}
        for task in topo:
            dep_ready = max((finish[p] + (0 if core_of[p] == core_of[task]
                                           else self.cross_wait)
                             for p in pred[task]), default=0)
            serial_ready = (finish[previous[task]] + self.same_wait
                            if task in previous else 0)
            starts[task] = max(dep_ready, serial_ready)
            finish[task] = starts[task] + metrics[task]["duration_proxy"]
        total_ddr = sum(m["copy_in_bytes"] + m["copy_out_bytes"] + m["spill_bytes_est"]
                        for m in metrics)
        ddr_lb = total_ddr / self.bandwidth
        proxy = max(max(finish.values(), default=0), ddr_lb)
        added_bytes = self._added_copy_bytes(groups)
        spill_est = sum(m["spill_bytes_est"] for m in metrics)
        memory_term = (0 if self.ablations.get("memory_penalty_disabled")
                       else spill_est / self.bandwidth)
        objective = proxy + 0.25 * added_bytes / self.bandwidth + memory_term
        return {"task_metrics": metrics, "task_predecessors": pred,
                "task_successors": succ, "task_core": core_of,
                "task_start_proxy": starts, "task_finish_proxy": finish,
                "estimated_makespan_proxy": proxy,
                "global_ddr_lower_bound_cycles": ddr_lb,
                "added_copy_bytes_proxy": added_bytes,
                "spill_bytes_proxy": spill_est, "objective_proxy": objective,
                "schedule_validation": "acyclic"}

    def _schedule_neighborhoods(self, row):
        """One deterministic move and adjacent-reorder neighborhood pass."""
        if self.ablations.get("no_refinement"):
            return []
        groups, base_plan = row["groups"], row["plan"]
        schedules = [list(core) for core in base_plan["core_schedules"]]
        _, _, _, _, metrics = self._task_metrics(groups)
        loads = [sum(metrics[t]["duration_proxy"] for t in order) for order in schedules]
        source = max(range(self.k), key=lambda c: (loads[c], len(schedules[c]), -c))
        neighbors = []
        if schedules[source]:
            task = schedules[source][-1]
            for target in range(self.k):
                if target == source:
                    continue
                proposal = [list(order) for order in schedules]
                proposal[source].remove(task); proposal[target].append(task)
                estimate = self._evaluate_fixed_schedule(groups, {
                    "node_to_subgraph": base_plan["node_to_subgraph"],
                    "core_schedules": proposal})
                if estimate:
                    plan = {"node_to_subgraph": dict(base_plan["node_to_subgraph"]),
                            "core_schedules": proposal}
                    neighbors.append({"groups": groups, "plan": plan, "estimate": estimate,
                        "label": row["label"] + f":move:{task}:{source}->{target}",
                        "decisions": row.get("decisions", []) + [{"operation": "move",
                            "task": task, "from_core": source, "to_core": target}],
                        "hash": sha_plan(plan),
                        "task_count": len(groups)})
        swap_count = 0
        for core, order in enumerate(schedules):
            for index in range(len(order) - 1):
                if swap_count >= 3:
                    break
                proposal = [list(items) for items in schedules]
                proposal[core][index], proposal[core][index + 1] = \
                    proposal[core][index + 1], proposal[core][index]
                estimate = self._evaluate_fixed_schedule(groups, {
                    "node_to_subgraph": base_plan["node_to_subgraph"],
                    "core_schedules": proposal})
                if estimate:
                    plan = {"node_to_subgraph": dict(base_plan["node_to_subgraph"]),
                            "core_schedules": proposal}
                    a, b = order[index], order[index + 1]
                    neighbors.append({"groups": groups, "plan": plan, "estimate": estimate,
                        "label": row["label"] + f":reorder:{core}:{a}<->{b}",
                        "decisions": row.get("decisions", []) + [{"operation": "reorder",
                            "core": core, "tasks": [a, b]}],
                        "hash": sha_plan(plan),
                        "task_count": len(groups)})
                swap_count += 1
            if swap_count >= 3:
                break
        return neighbors

    def solve(self, *, time_budget=300, top_k=5):
        deadline = self.started + max(0.1, time_budget)
        candidate_rows = []
        schedule_modes = ("heft", "topological", "work")
        phase_started = time.monotonic()
        partitions = self.candidates(deadline=deadline)
        self.phase_seconds["partition_coarsening"] = time.monotonic() - phase_started
        phase_started = time.monotonic()
        for candidate in partitions:
            if time.monotonic() >= deadline:
                break
            groups = candidate["groups"]
            try:
                validate_partition(self.g, groups)
                modes = ("topological",) if self.ablations.get("critical_path") else schedule_modes
                for mode in modes:
                    plan, estimate = self._schedule(groups, mode=mode)
                    candidate_rows.append({**candidate, "label": candidate["label"] + ":" + mode,
                                           "plan": plan, "estimate": estimate,
                                           "hash": sha_plan(plan),
                                           "task_count": len(groups)})
            except Exception as exc:
                candidate_rows.append({**candidate, "task_count": len(groups),
                                       "error": f"{type(exc).__name__}: {exc}"})
        if not candidate_rows:
            # Guaranteed fallback: singleton compute Op Tasks.
            groups = [[op] for op in self.g.compute_order]
            plan, estimate = self._schedule(groups)
            candidate_rows.append({"label": "singleton_fallback", "groups": groups,
                                   "plan": plan, "estimate": estimate,
                                   "hash": partition_hash(groups), "task_count": len(groups)})
        valid = [r for r in candidate_rows if "error" not in r]
        valid.sort(key=lambda r: (r["estimate"]["objective_proxy"],
                                  r["estimate"]["estimated_makespan_proxy"],
                                  r["estimate"]["added_copy_bytes_proxy"], r["task_count"], r["hash"]))
        self.phase_seconds["schedule_construction"] = time.monotonic() - phase_started
        phase_started = time.monotonic()
        if not self.ablations.get("no_refinement"):
            for base in valid[:min(5, max(2, top_k))]:
                candidate_rows.extend(self._schedule_neighborhoods(base))
            valid = [r for r in candidate_rows if "error" not in r]
            valid.sort(key=lambda r: (r["estimate"]["objective_proxy"],
                                      r["estimate"]["estimated_makespan_proxy"],
                                      r["estimate"]["added_copy_bytes_proxy"], r["task_count"], r["hash"]))
        self.phase_seconds["schedule_neighborhoods"] = time.monotonic() - phase_started
        coarse = [r for r in valid if not any(token in r["label"]
                                               for token in (":split", ":move:", ":reorder:"))]
        return {"selected": valid[0], "top_candidates": valid[:top_k],
                "best_initial_coarse": coarse[0] if coarse else valid[0],
                "candidate_count": len(candidate_rows), "elapsed_seconds": time.monotonic() - self.started,
                "natural_module_count": len(self.modules),
                "timed_out_during_search": time.monotonic() >= deadline,
                "phase_seconds": dict(self.phase_seconds),
                "all_candidate_errors": [r for r in candidate_rows if "error" in r]}
