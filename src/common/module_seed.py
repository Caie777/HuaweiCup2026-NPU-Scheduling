"""Natural module discovery shared by V2plus and RAMPplus."""
from __future__ import annotations
import heapq
import statistics
from collections import defaultdict


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


class NaturalModuleAggregator:
    """Natural module discovery with incremental quotient adjacency."""

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
