"""OJO-LNS: raw-Op joint partitioning, core placement, and ordering.

This is an independent search route. It starts from singleton compute Ops and
never consumes V1/V2/RAMP partitions unless an explicit warm-start ablation is
requested by the caller. Proxy scores rank candidates; the official evaluator
remains the only source of reported makespans.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import random
import time
from collections import defaultdict

from aggregate import validate_partition
from ojo_macro import macro_partitions
from ramp_dag import quotient_graph
from scene_cost import score_plan


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class OJOLNS:
    def __init__(self, graph, cores, *, bandwidth, same_wait, cross_wait,
                 capacity=None, seed=20260924, cpsat=False,
                 problem=1, scene_settings=None,
                 communication_boundary=False):
        self.g, self.k = graph, int(cores)
        self.problem, self.scene_settings = problem, scene_settings
        self.bandwidth = bandwidth
        self.same_wait = same_wait if problem == 1 else 0
        self.cross_wait = (cross_wait if problem == 1 else
                           scene_settings["scene_b"]["cross_core_copy_delay_cycles"])
        self.capacity = capacity or {}
        self.seed, self.use_cpsat = seed, cpsat
        self.communication_boundary = communication_boundary
        self.ops = list(graph.compute_order)
        self.op_set = set(self.ops)
        self.work = {o: max(0, int(graph.ops[o].get("cycles", 0))) for o in self.ops}
        self._eval_cache = {}
        self._feature_cache = {}
        self._candidate_cache = {}
        self._candidate_cache_hits = 0
        self._proxy_seconds = 0.0
        self._critical_ranked = sorted(self.ops, key=lambda o: (
            -graph.longest_path[o], -self.work[o], o))
        self._shared_pools = [sorted(cs & self.op_set)
                              for cs in graph.tensor_consumers.values()
                              if len(cs & self.op_set) > 1]
        self._shared_counts = defaultdict(int)
        for pool in self._shared_pools:
            for op in pool:
                self._shared_counts[op] += 1
        self._fork_pivots = [op for op in self.ops
                             if len(graph.compute_pred[op]) > 1
                             or len(graph.compute_succ[op]) > 1]
        self._communication_tensor_ids = sorted(
            (tid for tid in graph.tensors
             if graph.tensor_consumers[tid] & self.op_set
             and (graph.tensor_producers[tid] & self.op_set or
                  len(graph.tensor_consumers[tid] & self.op_set) > 1)),
            key=lambda tid: (
                -int(graph.tensors[tid].get("size", 0)) * max(
                    1, len(graph.tensor_consumers[tid] & self.op_set)), tid))[:256]

    def _communication_boundaries(self, cand):
        """Bounded, scene-specific boundary proxy; count each Tensor once."""
        owner = {int(op): task for op, task in cand["plan"]["node_to_subgraph"].items()}
        core_of = {task: core for core, row in enumerate(
            cand["plan"]["core_schedules"]) for task in row}
        rows = []
        for tid in self._communication_tensor_ids:
            producers = self.g.tensor_producers[tid] & self.op_set
            consumers = self.g.tensor_consumers[tid] & self.op_set
            if not consumers:
                continue
            producer_tasks = {owner[op] for op in producers}
            consumer_tasks = {owner[op] for op in consumers}
            size = int(self.g.tensors[tid].get("size", 0))
            if self.problem == 1:
                copies = (sum(task not in producer_tasks for task in consumer_tasks)
                          + sum(any(task != p for task in consumer_tasks)
                                for p in producer_tasks) if producer_tasks else
                          max(0, len(consumer_tasks) - 1))
            else:
                producer_cores = {core_of[task] for task in producer_tasks}
                consumer_cores = {core_of[task] for task in consumer_tasks}
                copies = (2 * len(consumer_cores - producer_cores)
                          if producer_cores else max(0, len(consumer_cores) - 1))
            if copies:
                rows.append({"tensor_id": tid, "estimated_bytes": size * copies,
                             "producer_ops": sorted(producers),
                             "consumer_ops": sorted(consumers),
                             "distinct_consumer_tasks": len(consumer_tasks)})
        rows.sort(key=lambda row: (-row["estimated_bytes"],
                                   -row["distinct_consumer_tasks"],
                                   row["tensor_id"]))
        return rows

    def _groups(self, owner_groups):
        return [sorted(set(g)) for g in owner_groups if g]

    def _valid(self, groups):
        try:
            validate_partition(self.g, groups)
            quotient_graph(self.g, groups)
            return True
        except (ValueError, KeyError):
            return False

    def _partial_valid(self, groups):
        """Check unique assigned compute Ops and the induced quotient DAG.

        Unassigned Ops are intentionally allowed here. This is never used as
        the final candidate check.
        """
        owner = {}
        for tid, group in enumerate(groups):
            if not group:
                return False
            for op in group:
                if op not in self.op_set or op in owner:
                    return False
                owner[op] = tid
        succ = [set() for _ in groups]
        indegree = [0] * len(groups)
        for op, src in owner.items():
            for nxt in self.g.compute_succ[op]:
                dst = owner.get(nxt)
                if dst is not None and dst != src and dst not in succ[src]:
                    succ[src].add(dst)
                    indegree[dst] += 1
        ready = [i for i, degree in enumerate(indegree) if degree == 0]
        visited = 0
        while ready:
            src = ready.pop()
            visited += 1
            for dst in succ[src]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
        return visited == len(groups)

    def _features(self, group):
        key = frozenset(group)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        pipe = defaultdict(int)
        touched = set()
        for op in group:
            pipe[self.g.ops[op].get("pipe", "UNKNOWN")] += self.work[op]
            touched.update(self.g.op_inputs[op])
            touched.update(self.g.op_outputs[op])
        local = set(group)
        boundary_bytes = 0
        live_bytes = 0
        boundary_count = 0
        for tid in touched:
            ps, cs = self.g.tensor_producers[tid], self.g.tensor_consumers[tid]
            lp, lc = ps & local, cs & local
            copy_out = any(self.g.ops[p].get("op") == "COPY_OUT" for p in ps)
            if (lc and not lp) or (lp and (copy_out or bool((cs & self.op_set) - local) or not (cs & self.op_set))):
                boundary_bytes += self.g.tensors[tid].get("size", 0)
                boundary_count += 1
            live_bytes += self.g.tensors[tid].get("size", 0)
        cap = sum(v for v in self.capacity.values() if isinstance(v, (int, float)))
        spill = max(0, live_bytes - cap) if cap else 0
        result = (max(pipe.values(), default=0), boundary_bytes, boundary_count, spill)
        self._feature_cache[key] = result
        return result

    def _schedule(self, groups, mode="critical", return_proxy=False):
        """Deterministic dynamic ready-list scheduling of the quotient DAG."""
        owner = {op: t for t, group in enumerate(groups) for op in group}
        _, succ, pred, topo = quotient_graph(self.g, groups)
        weight, boundary, spill_bytes = {}, {}, 0
        for t, group in enumerate(groups):
            pipe_cycles, transfer_bytes, count, spill = self._features(group)
            weight[t] = pipe_cycles + (transfer_bytes / max(1, self.bandwidth)
                                       if self.problem == 1 else 0)
            boundary[t] = count
            spill_bytes += spill
        cp = {}
        for t in reversed(topo):
            cp[t] = weight[t] + max((cp[v] + self.cross_wait for v in succ[t]), default=0)
        rem = {t: len(pred[t]) for t in pred}; ready = {t for t in pred if rem[t] == 0}
        core_end = [0.0] * self.k; orders = [[] for _ in range(self.k)]
        finish, placed = {}, {}
        while ready:
            if mode == "random":
                # Seeded tie breaking only; stable deterministic replay.
                task = min(ready, key=lambda t: (-cp[t], t))
            elif mode == "shared":
                task = min(ready, key=lambda t: (-boundary[t], -cp[t], t))
            else:
                task = min(ready, key=lambda t: (-cp[t], -weight[t], t))
            ready.remove(task); options = []
            for c in range(self.k):
                dep = max((finish[p] + (0 if placed[p] == c else self.cross_wait) for p in pred[task]), default=0)
                serial = core_end[c] + (self.same_wait if orders[c] else 0)
                st = max(dep, serial)
                options.append((st + weight[task], st, len(orders[c]), c))
            if mode == "core_first":
                c = min(range(self.k), key=lambda x:(core_end[x], len(orders[x]), x))
                st = max(core_end[c] + (self.same_wait if orders[c] else 0),
                         max((finish[p] + (0 if placed[p] == c else self.cross_wait) for p in pred[task]), default=0))
                options = [(st + weight[task], st, len(orders[c]), c)]
            _, st, _, c = min(options)
            placed[task] = c; finish[task] = st + weight[task]
            orders[c].append(task); core_end[c] = finish[task]
            for v in succ[task]:
                rem[v] -= 1
                if rem[v] == 0: ready.add(v)
        plan = {"node_to_subgraph": {str(op): owner[op] for op in sorted(owner)},
                "core_schedules": orders}
        # Pipe work and boundary traffic enter each Task duration. The greedy
        # finish time includes quotient dependencies, core serialization and
        # same/cross-core waits. Add a conservative memory-pressure surcharge.
        proxy = max(core_end, default=0) + spill_bytes / max(1, self.bandwidth)
        return (plan, proxy) if return_proxy else plan

    def _score_plan(self, groups, plan):
        """Validate Task dependencies plus core orders and score this exact plan.

        Returns None for any incomplete assignment, mismatched partition or
        cycle in the union graph. No default scheduler is called here.
        """
        n = len(groups)
        if len(plan.get("core_schedules", [])) != self.k:
            return None
        expected = {str(op): tid for tid, group in enumerate(groups) for op in group}
        if plan.get("node_to_subgraph") != expected:
            return None
        orders = plan["core_schedules"]
        flat = [tid for row in orders for tid in row]
        if len(flat) != n or sorted(flat) != list(range(n)):
            return None
        _, task_succ, task_pred, _ = quotient_graph(self.g, groups)
        succ = [set(task_succ[t]) for t in range(n)]
        pred = [set(task_pred[t]) for t in range(n)]
        core_of = {}
        prior = {}
        for core, row in enumerate(orders):
            for index, tid in enumerate(row):
                core_of[tid] = core
                prior[tid] = row[index-1] if index else None
            for a, b in zip(row, row[1:]):
                succ[a].add(b)
                pred[b].add(a)
        indegree = [len(pred[t]) for t in range(n)]
        ready = [t for t in range(n) if indegree[t] == 0]
        heapq.heapify(ready)
        finish = [0.0] * n
        while ready:
            tid = heapq.heappop(ready)
            pipe_cycles, boundary_bytes, _, spill = self._features(groups[tid])
            duration = pipe_cycles + (boundary_bytes + spill) / max(1, self.bandwidth)
            dep_ready = max((finish[p] + (0 if core_of[p] == core_of[tid] else self.cross_wait)
                             for p in task_pred[tid]), default=0.0)
            previous = prior[tid]
            core_ready = finish[previous] + self.same_wait if previous is not None else 0.0
            finish[tid] = max(dep_ready, core_ready) + duration
            for nxt in succ[tid]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(ready,nxt)
        if any(indegree):
            return None
        proxy = max(finish, default=0.0)
        if self.problem in (2, 3):
            # Scene B merges every core's subgraphs into one Task. The
            # Scene-A per-subgraph timeline above only checks plan legality;
            # scene-specific communication, residency and L2 terms rank it.
            proxy = score_plan(self.g, plan, self.problem,
                               self.scene_settings)["objective"]
        return {"proxy_makespan": proxy,
                "task_finish": finish, "core_finish": [finish[row[-1]] if row else 0.0 for row in orders]}

    def _candidate(self, groups, label, plan=None):
        groups = self._groups(groups)
        mode = "core_first" if label.startswith("core_first") else ("shared" if "shared" in label else "critical")
        group_key = tuple(tuple(group) for group in groups)
        cache_key = (group_key, "scheduled", mode) if plan is None else (
            group_key, "fixed", plan_hash(plan))
        if cache_key in self._candidate_cache:
            self._candidate_cache_hits += 1
            cached = self._candidate_cache[cache_key]
            return {**cached, "label": label} if cached is not None else None
        if not self._valid(groups):
            self._candidate_cache[cache_key] = None
            return None
        proxy_started = time.monotonic()
        if plan is None:
            plan = self._schedule(groups, mode)
        scored = self._score_plan(groups, plan)
        self._proxy_seconds += time.monotonic()-proxy_started
        if scored is None:
            self._candidate_cache[cache_key] = None
            return None
        proxy = scored["proxy_makespan"]
        ph = hashlib.sha256(json.dumps(sorted(tuple(sorted(g)) for g in groups),separators=(",",":")).encode()).hexdigest()
        result = {"groups": groups, "plan": plan, "label": label,
                  "hash": plan_hash(plan), "partition_hash": ph,
                  "task_count": len(groups), "proxy_makespan": proxy}
        self._candidate_cache[cache_key] = result
        return dict(result)

    def _schedule_neighbors(self, cand, max_neighbors=24):
        """Bounded fixed-partition move, reorder and cross-core swap search."""
        groups, plan = cand["groups"], cand["plan"]
        profile = self._score_plan(groups, plan)
        if profile is None:
            return [], 0
        orders = plan["core_schedules"]
        n = len(groups)
        if n < 2:
            return [], 0
        _, _, pred, _ = quotient_graph(self.g, groups)
        critical_core = max(range(self.k), key=lambda c: profile["core_finish"][c])
        ranked = sorted(range(n), key=lambda t:(
            -(profile["task_finish"][t] if t in orders[critical_core] else 0),
            -self._features(groups[t])[0], t))
        chosen = ranked[:min(4,n)]
        out, seen = [], set()
        attempts = 0
        def add(new_orders, operation):
            nonlocal attempts
            attempts += 1
            new_plan = {"node_to_subgraph":plan["node_to_subgraph"],"core_schedules":new_orders}
            key = plan_hash(new_plan)
            if key == cand["hash"] or key in seen:
                return
            seen.add(key)
            candidate = self._candidate(groups, f"schedule_{operation}", new_plan)
            if candidate is not None:
                out.append(candidate)
        for tid in chosen:
            source = next(c for c,row in enumerate(orders) if tid in row)
            for target in range(self.k):
                if target == source: continue
                row = orders[target]
                # Endpoints, center, dependency-aware earliest placement and
                # estimated earliest-start insertion are all considered.
                after_pred = max((row.index(p)+1 for p in pred[tid] if p in row),default=0)
                earliest = min(range(len(row)+1),key=lambda i:(
                    profile["task_finish"][row[i-1]] if i else 0.0, abs(i-after_pred)))
                for pos in sorted({0,len(row),len(row)//2,after_pred,earliest}):
                    new_orders=[list(x) for x in orders]
                    new_orders[source].remove(tid)
                    new_orders[target].insert(pos,tid)
                    add(new_orders,"move")
            row = orders[source]
            at = row.index(tid)
            for other in sorted({max(0,at-1),min(len(row)-1,at+1),0,len(row)-1}):
                if other == at: continue
                new_orders=[list(x) for x in orders]
                new_orders[source][at],new_orders[source][other]=new_orders[source][other],new_orders[source][at]
                add(new_orders,"reorder")
            for target in range(self.k):
                if target == source or not orders[target]: continue
                for other in (orders[target][0],orders[target][-1]):
                    new_orders=[list(x) for x in orders]
                    where=new_orders[target].index(other)
                    new_orders[source][at],new_orders[target][where]=other,tid
                    add(new_orders,"swap")
        out.sort(key=lambda x:(x["proxy_makespan"],x["hash"]))
        return out[:max_neighbors], attempts

    def construct(self, *, macro=False):
        """Four non-module-based seeds over raw compute Ops."""
        out = []
        # Op-level ready list: one Op per Task, hence no forced aggregation.
        out.append(self._candidate([[o] for o in self.ops], "raw_op_list_singletons"))
        # Core/workload-first bins, then dependency-safe quotient validation.
        groups = [[o] for o in self.ops]
        out.append(self._candidate(groups, "core_first_singleton_tasks"))
        # Critical-path-first ordering affects scheduling and tie resolution.
        out.append(self._candidate([[o] for o in sorted(self.ops, key=lambda x: (-self.g.longest_path[x], -self.work[x], x))], "critical_path_first"))
        # Shared-Tensor consumers are grouped first, across all natural boundaries.
        consumers = defaultdict(set)
        for tid, cs in self.g.tensor_consumers.items():
            cc = cs & self.op_set
            if len(cc) > 1: consumers[tid].update(cc)
        shared_sets = sorted(consumers.values(), key=lambda s: (-len(s), -sum(self.work[o] for o in s), min(s)))
        used, shared_groups = set(), []
        for ss in shared_sets:
            candidate = sorted(ss - used)
            if candidate and self._partial_valid(shared_groups + [candidate]):
                shared_groups.append(candidate); used.update(candidate)
        shared_groups.extend([[o] for o in self.ops if o not in used])
        out.append(self._candidate(shared_groups, "shared_tensor_first"))
        # Coarse raw-topological chunks followed by repeated binary splitting.
        order = list(self.g.compute_order); chunks = [order[i:i+max(2, (len(order)+self.k-1)//self.k)] for i in range(0, len(order), max(2, (len(order)+self.k-1)//self.k))]
        out.append(self._candidate(chunks, "coarse_initial"))
        while any(len(g) > 1 for g in chunks):
            nxt = []
            for g in chunks:
                if len(g) < 2: nxt.append(g); continue
                mid = len(g)//2; nxt.extend((g[:mid], g[mid:]))
            chunks = nxt
            out.append(self._candidate(chunks, f"coarse_split_{len(chunks)}"))
        targets=sorted({max(1,min(len(order),self.k*m)) for m in (1,2,4,8,12,16,24,32,48,64,80,100,128,160,256)},
                       key=lambda x:(min(abs(x-24*self.k),abs(x-100*self.k)),x))
        # A deepest-ready Kahn traversal follows independent raw dependency
        # chains before switching branches. It can form useful groups such as
        # short producer-consumer chains without importing any natural modules.
        import heapq
        rem={o:len(self.g.compute_pred[o]) for o in self.ops}
        ready=[(-self.g.levels[o],-self.g.longest_path[o],o) for o in self.ops if rem[o]==0]
        heapq.heapify(ready); chain_order=[]
        while ready:
            _,_,op=heapq.heappop(ready); chain_order.append(op)
            for v in self.g.compute_succ[op]:
                rem[v]-=1
                if rem[v]==0: heapq.heappush(ready,(-self.g.levels[v],-self.g.longest_path[v],v))
        def append_cuts(cut_order,prefix,selected_targets=None):
            use_targets=targets if selected_targets is None else selected_targets
            total=sum(self.work[o] for o in cut_order)
            for target in use_targets:
                width=max(1,(len(cut_order)+target-1)//target)
                count_groups=[cut_order[i:i+width] for i in range(0,len(cut_order),width)]
                out.append(self._candidate(count_groups,f"{prefix}_count_{target}_tasks"))
                work_groups=[]; current=[]; prefix_work=0
                for ix,op in enumerate(cut_order):
                    current.append(op); prefix_work+=self.work[op]
                    remaining_groups=target-len(work_groups)
                    remaining_ops=len(cut_order)-ix-1
                    if remaining_groups>1 and remaining_ops>=remaining_groups-1 and prefix_work >= total/target*(len(work_groups)+1):
                        work_groups.append(current); current=[]
                if current: work_groups.append(current)
                if len(work_groups)>target: work_groups=work_groups[:target-1]+[[o for g in work_groups[target-1:] for o in g]]
                out.append(self._candidate(work_groups,f"{prefix}_work_{target}_tasks"))
        # Prioritize chain-following cuts because they tend to internalize true
        # raw dataflow edges; independent Op-level candidates remain below.
        append_cuts(chain_order,"deep_chain")
        # Multi-resolution contiguous ordinary topological cuts remain a
        # structurally different family. Work-balanced boundaries account for
        # heterogeneous operation cycles.
        append_cuts(order,"topological")
        if macro:
            for groups, label in macro_partitions(self.g, self.work, self.k):
                out.append(self._candidate(groups, label))
        return [x for x in out if x]

    def _destroy(self, cand, kind, size, rng):
        groups = [list(g) for g in cand["groups"]]; owner = {o:t for t,g in enumerate(groups) for o in g}
        selected = set()
        boundary_info = None
        if kind == "critical_path":
            ranked = self._critical_ranked
            seed = ranked[rng.randrange(min(len(ranked), max(1, size)))] if ranked else None
            if seed is not None:
                selected.add(seed); frontier = [seed]
                while frontier and len(selected) < size:
                    x = frontier.pop(0); ns = sorted(self.g.compute_pred[x] | self.g.compute_succ[x], key=lambda o:(-self.g.longest_path[o], o))
                    for y in ns:
                        if y not in selected: selected.add(y); frontier.append(y)
                        if len(selected) >= size: break
        elif kind == "shared_tensor":
            pools = self._shared_pools
            if pools:
                pool = pools[rng.randrange(min(6, len(pools)))] if len(pools) <= 6 else sorted(pools, key=lambda x:-len(x))[rng.randrange(6)]
                selected.update(pool[:size])
        elif kind == "core_segment":
            core = max(range(self.k), key=lambda c: sum(self._features(groups[t])[0] for t in cand["plan"]["core_schedules"][c]))
            order = cand["plan"]["core_schedules"][core]
            if order:
                at = rng.randrange(len(order))
                for tid in order[at:]:
                    selected.update(groups[tid])
                    if len(selected)>=size: break
        elif kind == "communication_boundary":
            boundaries = self._communication_boundaries(cand)
            if boundaries:
                boundary_info = boundaries[rng.randrange(min(4, len(boundaries)))]
                anchors = boundary_info["producer_ops"] + boundary_info["consumer_ops"]
                for op in dict.fromkeys(anchors):
                    selected.add(op)
                    if len(selected) >= size:
                        break
                frontier = list(selected)
                for op in frontier:
                    for near in sorted(self.g.compute_pred[op] |
                                       self.g.compute_succ[op]):
                        selected.add(near)
                        if len(selected) >= size:
                            break
                    if len(selected) >= size:
                        break
        else:  # fork/join dependency region
            pivots = self._fork_pivots
            if pivots:
                pivot = pivots[rng.randrange(len(pivots))]
                selected.add(pivot); selected.update(self.g.compute_pred[pivot]); selected.update(self.g.compute_succ[pivot])
                for x in list(selected): selected.update(self.g.compute_pred[x] & set(pivots)); selected.update(self.g.compute_succ[x] & set(pivots))
        if len(selected) < size:
            remaining = [o for o in self.ops if o not in selected]
            rng.shuffle(remaining); selected.update(remaining[:size-len(selected)])
        selected = set(sorted(selected, key=lambda o:(-self.g.longest_path[o],o))[:max(1,size)])
        touched_tasks = len({owner[o] for o in selected})
        # Split touched Tasks and release only selected raw Ops. This keeps the
        # local search bounded even when a seed has a giant Task.
        order_index = {op:i for i,op in enumerate(self.ops)}
        fixed = []
        touched_remainder = []
        for group in groups:
            if not any(o in selected for o in group):
                fixed.append(group)
                continue
            released_positions = sorted(order_index[o] for o in group if o in selected)
            keep = sorted((o for o in group if o not in selected),key=order_index.get)
            segments = []
            for op in keep:
                if (segments and any(order_index[segments[-1][-1]] < pos < order_index[op]
                                     for pos in released_positions)):
                    segments.append([])
                if not segments: segments.append([])
                segments[-1].append(op)
            touched_remainder.extend(segments)
        fixed.extend(touched_remainder)
        released = sorted(selected)
        if not self._valid(fixed+[[o] for o in released]):
            fixed = [g for g in fixed if g not in touched_remainder]
            fixed.extend([[o] for g in touched_remainder for o in g])
        self._last_destroy_info = {"target_released_ops":size,
                                   "actual_released_ops":len(released),
                                   "touched_tasks":touched_tasks,"neighborhood_type":kind}
        if boundary_info:
            self._last_destroy_info.update(
                tensor_id=boundary_info["tensor_id"],
                estimated_boundary_bytes=boundary_info["estimated_bytes"])
        return fixed, released

    def _repair(self, fixed, released, rng, mode):
        groups = [list(g) for g in fixed]
        owner = {op:i for i,g in enumerate(groups) for op in g}
        order = sorted(released, key=lambda o:(self.g.levels[o], -self.work[o], o))
        if mode == "shared":
            order.sort(key=lambda o:(-self._shared_counts[o], self.g.levels[o], o))
        for op in order:
            options = []
            neighboring = set()
            for v in self.g.compute_pred[op] | self.g.compute_succ[op]:
                if v in owner: neighboring.add(owner[v])
            for tid in self.g.op_inputs[op] | self.g.op_outputs[op]:
                for v in (self.g.tensor_producers[tid] | self.g.tensor_consumers[tid]):
                    if v != op and v in owner: neighboring.add(owner[v])
            # Bound repair cost on large graphs: test structural neighbors and a few deterministic
            # workload-balanced alternatives, instead of rescanning every Task for every Op.
            if len(neighboring) < 3 and groups:
                neighboring.update(range(max(0,len(groups)-3),len(groups)))
            for idx in sorted(i for i in neighboring if i >= 0):
                group = groups[idx]
                trial = [*groups]; trial[idx] = sorted(group + [op])
                if self._partial_valid(trial):
                    # Merge preference: boundary bytes saved with a small serialization penalty.
                    cross = sum(self.g.tensors[t].get("size",0) for v in group for t in self.g.boundary_tensors.get((v,op),set()) | self.g.boundary_tensors.get((op,v),set()))
                    score = ((cross if self.problem == 1 else 0.25 * cross)
                             - 0.15 * min(sum(self.work[x] for x in group), self.work[op]))
                    options.append((-score, idx, trial))
            options.append((0, len(groups), groups + [[op]]))
            options.sort(key=lambda x:(x[0], x[1]))
            choice = options[0][2]
            # Small seeded perturbation enables controlled non-monotone diversity.
            if len(options) > 1 and rng.random() < 0.08: choice = options[1][2]
            groups = choice
            owner[op] = next(i for i,g in enumerate(groups) if op in g)
        return self._candidate(groups, f"repair_{mode}")

    def solve(self, *, iterations=120, neighborhood_ops=24, seed_count=12,
              allow_uphill=True, cpsat_seconds=2.0, time_budget=None, incumbent=None,
              schedule_search=True, multiscale=True, macro=False,
              feedback_only=False):
        started = time.monotonic(); local_deadline = started + time_budget if time_budget is not None else float("inf")
        rng = random.Random(self.seed)
        proxy_before = self._proxy_seconds
        cache_hits_before = self._candidate_cache_hits
        if feedback_only and incumbent:
            starting = self._candidate(incumbent["groups"],
                                       "official_feedback_incumbent",
                                       incumbent.get("plan"))
            seeds = [starting] if starting else self.construct(macro=False)
        else:
            seeds = self.construct(macro=macro)
        pool = {x["hash"]:x for x in seeds}
        legacy_seeds = [x for x in seeds if not x["label"].startswith("macro_")]
        macro_seeds = [x for x in seeds if x["label"].startswith("macro_")]
        construction_seconds = time.monotonic() - started
        phase_seconds = defaultdict(float)
        released_counts = []
        destroy_stats = []
        valid_repairs = 0
        schedule_attempts = schedule_valid = 0
        # Optional prior-route warm start is injected only by explicit ablation.
        # Default multi-start state favors a moderate raw-Op granularity. Other
        # constructors remain in the candidate pool and official evaluation.
        state = min(legacy_seeds,key=lambda x:(x["proxy_makespan"],x["hash"]))
        official_seed = None
        if incumbent:
            candidate=self._candidate(incumbent["groups"],"official_feedback_incumbent",
                                      incumbent.get("plan"))
            if candidate:
                pool[candidate["hash"]]=candidate; state=candidate
                official_seed = candidate
        best = state; legacy_best = None; accepted = 0; attempts = 0; cpsat_status = "disabled"
        if self.use_cpsat:
            try:
                import ortools  # noqa: F401
                cpsat_status = "available"
            except ImportError:
                cpsat_status = "unavailable_install_with_pip_install_ortools"
        neighborhoods = ["critical_path", "shared_tensor", "core_segment", "fork_join"]
        if self.communication_boundary:
            neighborhoods.append("communication_boundary")
        def consider(candidate):
            nonlocal state,best,accepted
            if candidate is None or candidate["hash"] in pool:
                return
            pool[candidate["hash"]] = candidate
            score = candidate["proxy_makespan"]
            current = state["proxy_makespan"]
            if score <= current or (allow_uphill and score <= 1.15*current and rng.random() < 0.22):
                state = candidate
                accepted += 1
            if (score,candidate["hash"]) < (best["proxy_makespan"],best["hash"]):
                best = candidate
        total_iterations = iterations * 2 if macro and macro_seeds else iterations
        for it in range(total_iterations):
            if time.monotonic() >= local_deadline: break
            if macro and it == iterations and macro_seeds:
                legacy_best = best
                state = min(macro_seeds,key=lambda x:(x["proxy_makespan"],x["hash"]))
                if (state["proxy_makespan"],state["hash"]) < (best["proxy_makespan"],best["hash"]):
                    best = state
            active_seeds = legacy_seeds if not macro or it < iterations else macro_seeds
            attempts += 1
            base = state if rng.random() < 0.65 else (best if rng.random() < 0.6 else active_seeds[rng.randrange(len(active_seeds))])
            kind = neighborhoods[it % len(neighborhoods)]
            scale = "small"
            if multiscale and (it % 4 == 3 or
                               (kind == "communication_boundary" and
                                (it // len(neighborhoods)) % 2 == 1)):
                scale = "medium"
                size = min(len(self.ops),max(32,min(64,neighborhood_ops*8)))
            elif multiscale:
                size = min(len(self.ops),max(8,min(24,neighborhood_ops*2)))
            else:
                size = min(len(self.ops), max(2, neighborhood_ops * (1 + (it//len(neighborhoods))%3)//2))
            t0 = time.monotonic()
            fixed, released = self._destroy(base, kind, size, rng)
            destroy_elapsed = time.monotonic()-t0
            phase_seconds["destroy"] += destroy_elapsed
            if kind == "communication_boundary":
                phase_seconds["communication_boundary_destroy"] += destroy_elapsed
            released_counts.append(len(released))
            destroy_stats.append({**self._last_destroy_info,"scale":scale})
            t0 = time.monotonic()
            cand = self._repair(fixed, released, rng, "shared" if kind == "shared_tensor" else "critical")
            repair_elapsed = time.monotonic()-t0
            phase_seconds["repair_and_proxy"] += repair_elapsed
            if kind == "communication_boundary":
                phase_seconds["communication_boundary_repair_and_proxy"] += repair_elapsed
            destroy_stats[-1]["repair_seconds"] = repair_elapsed
            destroy_stats[-1]["valid_repair"] = cand is not None
            if cand is not None:
                valid_repairs += 1
                cand["label"] = f"repair_{scale}_{kind}"
                cand["destroy_info"] = {
                    key: value for key, value in destroy_stats[-1].items()
                    if key != "repair_seconds"}
            if self.use_cpsat and it % 4 == 0:
                rebuilt=cpsat_rebuild(self.g,fixed,released,self.k,bandwidth=self.bandwidth,
                    same_wait=self.same_wait,cross_wait=self.cross_wait,time_limit=cpsat_seconds,
                    base_groups=base["groups"],base_plan=base["plan"])
                if rebuilt:
                    try:
                        if self._valid(rebuilt["groups"]):
                            flat=[t for core in rebuilt["plan"]["core_schedules"] for t in core]
                            if sorted(flat)==list(range(len(rebuilt["groups"]))) and len(flat)==len(set(flat)):
                                ph=hashlib.sha256(json.dumps(sorted(tuple(sorted(g)) for g in rebuilt["groups"]),separators=(",",":")).encode()).hexdigest()
                                cp_candidate=self._candidate(rebuilt["groups"],f"local_cpsat_{rebuilt['status']}",rebuilt["plan"])
                                if cp_candidate: cp_candidate["cpsat"]={k:rebuilt[k] for k in ("status","proxy_makespan","wall_seconds","objective")}
                                if cp_candidate and (cand is None or cp_candidate["hash"] not in pool): cand=cp_candidate
                    except (ValueError,KeyError):
                        pass
            consider(cand)
            if schedule_search and it % 10 == 0 and time.monotonic() < local_deadline:
                source = official_seed if official_seed and it % 8 == 0 and len(official_seed["groups"])>1 else (cand or state)
                if len(source["groups"])<2:
                    source = next((s for s in seeds if len(s["groups"])>1),source)
                t0 = time.monotonic()
                neighbors, tried = self._schedule_neighbors(source,max_neighbors=6)
                phase_seconds["schedule_neighborhood"] += time.monotonic()-t0
                schedule_attempts += tried
                schedule_valid += len(neighbors)
                for neighbor in neighbors:
                    consider(neighbor)
        distinct = sorted(pool.values(), key=lambda x:(x["proxy_makespan"], x["partition_hash"], x["hash"]))
        selected=[]; seen_hashes=set(); partition_counts=defaultdict(int)
        def select(c, limit=seed_count):
            if c is None or len(selected)>=limit or c["hash"] in seen_hashes:
                return
            if partition_counts[c["partition_hash"]] >= 3:
                return
            selected.append(c); seen_hashes.add(c["hash"])
            partition_counts[c["partition_hash"]] += 1
        select(official_seed)
        select(legacy_best)
        select(best)
        if incumbent:
            # Reserve feedback pool space for raw-Op mechanisms before macro
            # constructor seeds fill it. A proxy error in one destroy kind
            # must not make every official repair evaluation the same kind.
            for scale in ("small", "medium"):
                for kind in neighborhoods:
                    matches = (c for c in distinct
                               if c["label"] == f"repair_{scale}_{kind}")
                    select(next(matches, None))
        for target in (self.k,4*self.k):
            select(min(seeds,key=lambda x:(abs(x["task_count"]-target),x["proxy_makespan"],x["hash"])))
        if macro:
            macro_seeds = sorted((c for c in seeds if c["label"].startswith("macro_")),
                                 key=lambda x:(x["proxy_makespan"],x["hash"]))
            # First cover different structural mechanisms, then different
            # scales. A proxy misrank cannot exclude every macro family.
            quota = min(8 if iterations == 0 else 5, max(1, seed_count - 3))
            chosen_counts = set()
            for family in ("unary", "component", "level", "chain", "tensor", "stage"):
                family_seeds = [c for c in macro_seeds
                                if c["label"].startswith(f"macro_{family}_")]
                if family_seeds and len(chosen_counts) < quota:
                    candidate = family_seeds[0]
                    select(candidate)
                    chosen_counts.add(candidate["task_count"])
            for c in macro_seeds:
                if len(chosen_counts) >= quota:
                    break
                if c["task_count"] not in chosen_counts:
                    select(c)
                    chosen_counts.add(c["task_count"])
        categories=(
            [c for c in distinct if c["label"].startswith("schedule_")],
            [c for c in distinct if c["label"].startswith("repair_small_")],
            [c for c in distinct if c["label"].startswith("repair_medium_")],
            [c for c in sorted(seeds,key=lambda x:(x["proxy_makespan"],x["hash"]))],
        )
        for category in categories:
            for c in category[:2]: select(c)
        for c in distinct:
            select(c)
        elapsed = time.monotonic()-started
        return {"seeds": seeds, "candidates": selected[:seed_count], "candidate_count": len(pool),
                "candidate_cache_hits": self._candidate_cache_hits - cache_hits_before,
                "attempts": attempts, "accepted_transitions": accepted,
                "search_seconds": elapsed, "cpsat_status": cpsat_status,
                "best_proxy_candidate": best, "current_candidate": state,
                "official_best_candidate": official_seed, "seed": self.seed,
                "phase_seconds": {"construction":construction_seconds,**dict(phase_seconds),
                                  "proxy_total":self._proxy_seconds-proxy_before},
                "released_op_counts": released_counts, "valid_repairs":valid_repairs,
                "destroy_stats":destroy_stats,"schedule_attempts":schedule_attempts,
                "schedule_valid_candidates":schedule_valid,
                "valid_candidates_per_second":valid_repairs/max(0.001,elapsed-construction_seconds)}


def lower_bounds(graph, cores):
    work = sum(graph.ops[o].get("cycles", 0) for o in graph.compute_ids)
    path = max(graph.longest_path.values(), default=0)
    pipe = max(graph.pipe_work.values(), default=0)
    # DDR bound is a transparent traffic lower bound: each distinct original DDR
    # boundary tensor is transferred once at configured bandwidth (no spill term).
    ddr = sum(t.get("size", 0) for tid,t in graph.tensors.items() if t.get("pos") == "DDR" and
              ((graph.tensor_producers[tid] & set(graph.compute_ids)) or (graph.tensor_consumers[tid] & set(graph.compute_ids))))
    return {"compute_cores_bound": work/max(1,cores), "critical_path_bound": path,
            "pipe_serial_bound": pipe, "ddr_bytes_lower_bound": ddr}


def cpsat_rebuild(graph, fixed_groups, released_ops, cores, *, bandwidth, same_wait,
                  cross_wait, time_limit=2.0, max_ops=12, max_tasks=24,
                  base_groups=None, base_plan=None):
    """Joint local partition/core/order/earliest-finish CP-SAT rebuild.

    Returns None when OR-Tools is absent, the neighborhood exceeds its explicit
    size caps, or CP-SAT has no feasible incumbent. The caller must retain its
    incumbent in all such cases. Durations and boundary transfer penalties are
    ranking proxies; every returned plan still requires official evaluation.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None
    released = sorted(set(released_ops))
    fixed_groups = [sorted(set(g)) for g in fixed_groups if g]
    if len(released) > max_ops or len(fixed_groups) + min(len(released), 8) > max_tasks:
        return None
    local_slots = min(max(1, len(released)), 8)
    nt = len(fixed_groups) + local_slots
    if nt > max_tasks: return None
    model = cp_model.CpModel(); f = len(fixed_groups)
    fixed_owner = {o:t for t,g in enumerate(fixed_groups) for o in g}
    local_x = {(o,t):model.NewBoolVar(f"x_{o}_{t}") for o in released for t in range(f,nt)}
    active = [model.NewConstant(1) if t < f else model.NewBoolVar(f"active_{t}") for t in range(nt)]
    for o in released:
        model.Add(sum(local_x[o,t] for t in range(f,nt)) == 1)
    for t in range(f,nt):
        assigned = sum(local_x[o,t] for o in released)
        model.Add(assigned >= active[t]); model.Add(assigned <= len(released)*active[t])
    # Active local slots form a prefix, eliminating empty-slot symmetry.
    for t in range(f+1,nt): model.Add(active[t] <= active[t-1])
    place = [[model.NewBoolVar(f"core_{t}_{c}") for c in range(cores)] for t in range(nt)]
    for t in range(nt): model.Add(sum(place[t]) == active[t])
    rank = [model.NewIntVar(0,nt-1,f"rank_{t}") for t in range(nt)]
    # Freeze every outside Task's core and its relative position on that core.
    # The released region alone may receive new core/order decisions.
    if base_groups is not None and base_plan is not None:
        original_id={tuple(sorted(g)):i for i,g in enumerate(base_groups)}
        original_core={tid:c for c,row in enumerate(base_plan["core_schedules"]) for tid in row}
        slot_by_original={original_id[tuple(sorted(g))]:slot for slot,g in enumerate(fixed_groups)}
        fixed_orders=[[] for _ in range(cores)]
        for c,row in enumerate(base_plan["core_schedules"]):
            for old_id in row:
                if old_id in slot_by_original:
                    slot=slot_by_original[old_id]
                    model.Add(place[slot][c] == 1)
                    fixed_orders[c].append(slot)
        for row in fixed_orders:
            for a,b in zip(row,row[1:]): model.Add(rank[a]+1 <= rank[b])
    horizon = sum(int(graph.ops[o].get("cycles",0)) for o in graph.compute_ids) + (nt+1)*(int(same_wait)+int(cross_wait)+1)
    starts=[model.NewIntVar(0,horizon,f"start_{t}") for t in range(nt)]
    ends=[model.NewIntVar(0,horizon,f"end_{t}") for t in range(nt)]
    durations=[model.NewIntVar(0,horizon,f"dur_{t}") for t in range(nt)]
    def pipe_work_fixed(group):
        out=defaultdict(int)
        for o in group: out[graph.ops[o].get("pipe","UNKNOWN")]+=int(graph.ops[o].get("cycles",0))
        return out
    pipe_names=sorted({graph.ops[o].get("pipe","UNKNOWN") for o in graph.compute_ids})
    for t,g in enumerate(fixed_groups):
        model.Add(durations[t] == max(pipe_work_fixed(g).values(),default=0))
    for t in range(f,nt):
        by_pipe=[]
        for pipe in pipe_names:
            w=model.NewIntVar(0,horizon,f"pw_{t}_{pipe}")
            model.Add(w == sum(int(graph.ops[o].get("cycles",0))*local_x[o,t] for o in released if graph.ops[o].get("pipe","UNKNOWN")==pipe))
            by_pipe.append(w)
        model.AddMaxEquality(durations[t],by_pipe)
    for t in range(nt):
        model.Add(ends[t] == starts[t] + durations[t]).OnlyEnforceIf(active[t])
        model.Add(starts[t] == 0).OnlyEnforceIf(active[t].Not())
        model.Add(ends[t] == 0).OnlyEnforceIf(active[t].Not())
        model.Add(rank[t] == 0).OnlyEnforceIf(active[t].Not())
    # Optional intervals jointly choose each Task's core and avoid overlap.
    intervals=[[] for _ in range(cores)]
    for t in range(nt):
        for c in range(cores):
            intervals[c].append(model.NewOptionalIntervalVar(starts[t],durations[t],ends[t],place[t][c],f"iv_{t}_{c}"))
    for c in range(cores): model.AddNoOverlap(intervals[c])
    for t in range(nt):
        for u in range(t+1,nt):
            both=[active[t],active[u]]
            model.Add(rank[t] != rank[u]).OnlyEnforceIf(both)
            before=model.NewBoolVar(f"ord_{t}_{u}")
            model.Add(rank[t] < rank[u]).OnlyEnforceIf(before)
            model.Add(rank[u] < rank[t]).OnlyEnforceIf(before.Not())
            for c in range(cores):
                model.Add(starts[u] >= ends[t] + int(same_wait)).OnlyEnforceIf([before,place[t][c],place[u][c]])
                model.Add(starts[t] >= ends[u] + int(same_wait)).OnlyEnforceIf([before.Not(),place[t][c],place[u][c]])

    def slots_for(op):
        return [(fixed_owner[op],None)] if op in fixed_owner else [(t,local_x[op,t]) for t in range(f,nt)]
    # Preserve all quotient arcs. Local assignments can choose entirely new
    # groups; reified slot-pair precedences reject any quotient cycle.
    cut_terms=[]; seen_fixed_arcs=set()
    for a in graph.compute_ids:
        for b in graph.compute_succ[a]:
            aa,bb=slots_for(a),slots_for(b)
            for t,xa in aa:
                for u,xb in bb:
                    if t==u: continue
                    lits=[v for v in (xa,xb) if v is not None]
                    if not lits:
                        seen_fixed_arcs.add((t,u)); continue
                    model.Add(rank[t]+1 <= rank[u]).OnlyEnforceIf(lits)
                    for c in range(cores):
                        for d in range(cores):
                            delay=0 if c==d else int(cross_wait)
                            model.Add(starts[u] >= ends[t]+delay).OnlyEnforceIf(lits+[place[t][c],place[u][d]])
                    if xa is not None and xb is not None:
                        cut=model.NewBoolVar(f"cut_{a}_{b}_{t}_{u}")
                        model.AddBoolAnd([xa,xb]).OnlyEnforceIf(cut)
                        model.AddBoolOr([xa.Not(),xb.Not(),cut])
                        cut_terms.append(cut)
    for t,u in seen_fixed_arcs: model.Add(rank[t]+1 <= rank[u])
    makespan=model.NewIntVar(0,horizon,"proxy_makespan")
    model.AddMaxEquality(makespan,ends)
    objective=1000*makespan + sum(active) + sum(cut_terms)
    model.Minimize(objective)
    solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=max(0.05,float(time_limit)); solver.parameters.num_search_workers=1
    status=solver.Solve(model)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return None
    groups=[list(g) for g in fixed_groups]
    for t in range(f,nt):
        if solver.Value(active[t]): groups.append(sorted(o for o in released if solver.Value(local_x[o,t])))
    plan={"node_to_subgraph":{str(o):t for t,g in enumerate(groups) for o in g},"core_schedules":[[] for _ in range(cores)]}
    # Compaction can change slot IDs; recover core/rank from the solved slots.
    for t in range(nt):
        if not solver.Value(active[t]): continue
        ops=fixed_groups[t] if t<f else [o for o in released if solver.Value(local_x[o,t])]
        compact_id=next(i for i,g in enumerate(groups) if g==ops)
        c=next(c for c in range(cores) if solver.Value(place[t][c]))
        plan["core_schedules"][c].append((solver.Value(rank[t]),compact_id))
    plan["core_schedules"]=[[tid for _,tid in sorted(row)] for row in plan["core_schedules"]]
    return {"groups":groups,"plan":plan,"proxy_makespan":solver.Value(makespan),
            "status":"OPTIMAL" if status==cp_model.OPTIMAL else "FEASIBLE",
            "wall_seconds":solver.WallTime(),"objective":"local CP-SAT proxy; official validation required"}
