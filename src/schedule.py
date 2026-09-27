"""确定性 HEFT 风格任务优先级与核心分配。"""
from __future__ import annotations

from collections import defaultdict, deque


def make_plan(graph, groups, num_cores, bandwidth=60, cross_wait=1000, same_wait=100):
    if num_cores < 1:
        raise ValueError("核心数必须为正")
    op_to_task = {op: tid for tid, group in enumerate(groups) for op in group}
    n = len(groups)
    pred = {i: set() for i in range(n)}
    succ = {i: set() for i in range(n)}
    for a in graph.compute_ids:
        for b in graph.compute_succ[a]:
            x, y = op_to_task[a], op_to_task[b]
            if x != y:
                succ[x].add(y); pred[y].add(x)
    indeg = {i: len(pred[i]) for i in range(n)}
    q = deque(i for i in range(n) if indeg[i] == 0)
    topo = []
    while q:
        a = q.popleft(); topo.append(a)
        for b in sorted(succ[a]):
            indeg[b] -= 1
            if indeg[b] == 0: q.append(b)
    if len(topo) != n:
        raise ValueError("任务依赖图存在环")

    # 时间估算：每 Pipe 内串行，Pipe 间并行；对边界字节按共享 DDR 带宽估计。
    duration, task_bytes = {}, {}
    compute_set = set(graph.compute_ids)
    for tid, members in enumerate(groups):
        pipe_work = defaultdict(int)
        member_set = set(members)
        touched = set()
        for op_id in members:
            op = graph.ops[op_id]
            pipe_work[op.get("pipe", "UNKNOWN")] += op.get("cycles", 0)
            touched.update(graph.op_inputs[op_id]); touched.update(graph.op_outputs[op_id])
        boundary = set()
        for tensor_id in touched:
            ps = graph.tensor_producers[tensor_id]
            cs = graph.tensor_consumers[tensor_id]
            local_p = ps & member_set; local_c = cs & member_set
            compute_c = cs & compute_set
            copy_out = any(graph.ops[p].get("op") == "COPY_OUT" for p in ps)
            input_boundary = bool(local_c) and not bool(local_p)
            output_boundary = bool(local_p) and (
                copy_out or bool(compute_c - member_set) or not compute_c)
            if input_boundary or output_boundary:
                boundary.add(tensor_id)
        b = sum(graph.tensors[t].get("size", 0) for t in boundary)
        task_bytes[tid] = b
        duration[tid] = max(pipe_work.values(), default=0) + b / max(1, bandwidth)

    rank = {}
    for tid in reversed(topo):
        rank[tid] = duration[tid] + max(
            (rank[v] + task_bytes[v] / max(1, bandwidth) for v in succ[tid]), default=0)

    # 动态 ready 集合：每次只放置依赖已放置的任务，所得同核序天然不会和 DAG 构成等待环。
    remaining = {i: len(pred[i]) for i in range(n)}
    ready = {i for i in range(n) if remaining[i] == 0}
    finish, placement = {}, {}
    core_orders = [[] for _ in range(num_cores)]
    core_ready = [0.0] * num_cores
    while ready:
        task = min(ready, key=lambda i: (-rank[i], i))
        ready.remove(task)
        choices = []
        for core in range(num_cores):
            pred_ready = 0.0
            for p in pred[task]:
                delay = 0 if placement[p] == core else cross_wait
                pred_ready = max(pred_ready, finish[p] + delay)
            serial_ready = core_ready[core] + (same_wait if core_orders[core] else 0)
            start = max(pred_ready, serial_ready)
            choices.append((start + duration[task], start, core))
        end, start, core = min(choices)
        placement[task] = core; finish[task] = end
        core_orders[core].append(task); core_ready[core] = end
        for child in succ[task]:
            remaining[child] -= 1
            if remaining[child] == 0: ready.add(child)
    mapping = {str(op): op_to_task[op] for op in sorted(op_to_task)}
    return {"node_to_subgraph": mapping, "core_schedules": core_orders}, {
        "task_estimated_duration": duration, "task_boundary_bytes_proxy": task_bytes,
        "task_estimated_finish": finish, "task_core": placement,
        "task_predecessors": {k: sorted(v) for k, v in pred.items()},
    }
