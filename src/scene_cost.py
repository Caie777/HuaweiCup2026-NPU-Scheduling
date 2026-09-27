"""Cheap, scene-specific candidate ranking; official Makespan is authoritative."""
from __future__ import annotations

from collections import defaultdict


def score_plan(graph, plan, problem, settings):
    """Rank legal Plans using core load, communication, memory and L2 reuse.

    This is deliberately a proxy. In problems 2/3 all subgraphs on a core
    share one Task, so only cross-core Tensor traffic is charged. The L2 term
    estimates bounded reuse of read-only input Tensors; it is not a blanket
    DDR bandwidth discount and cannot model FIFO timing exactly.
    """
    if problem not in (1, 2, 3):
        raise ValueError(problem)
    schedules = plan["core_schedules"]
    owner = {int(op): int(task) for op, task in plan["node_to_subgraph"].items()}
    core_of = {task: core for core, row in enumerate(schedules) for task in row}
    pipes = [defaultdict(int) for _ in schedules]
    touched = [set() for _ in schedules]
    for op in graph.compute_ids:
        core = core_of[owner[op]]
        pipes[core][graph.ops[op].get("pipe", "UNKNOWN")] += int(graph.ops[op].get("cycles", 0))
        touched[core].update(graph.op_inputs[op])
        touched[core].update(graph.op_outputs[op])
    loads = [max(row.values(), default=0) for row in pipes]
    bandwidth = max(1, settings["bandwidth"])
    copy_bytes = 0
    shared_input_bytes = 0
    l2_reuse_candidates = []
    cross_edges = set()
    for tid, tensor in graph.tensors.items():
        producer_cores = {core_of[owner[o]] for o in graph.tensor_producers[tid] if o in owner}
        consumer_cores = {core_of[owner[o]] for o in graph.tensor_consumers[tid] if o in owner}
        size = int(tensor.get("size", 0))
        if problem == 1:
            producers = {owner[o] for o in graph.tensor_producers[tid] if o in owner}
            consumers = {owner[o] for o in graph.tensor_consumers[tid] if o in owner}
            copy_bytes += size * (sum(task not in producers for task in consumers)
                                  + sum(any(c != task for c in consumers) for task in producers))
        elif producer_cores:
            receivers = consumer_cores - producer_cores
            copy_bytes += 2 * size * len(receivers)
            cross_edges.update((src, dst) for src in producer_cores for dst in receivers)
        else:
            # Original DDR inputs may be read separately by several cores.
            repeated = max(0, len(consumer_cores) - 1)
            shared_input_bytes += repeated * size
            if repeated and size:
                l2_reuse_candidates.append((size, repeated))
    capacity = settings["capacity"]
    pressure = 0
    if problem in (2, 3):
        for core, tids in enumerate(touched):
            by_pos = defaultdict(int)
            for tid in tids:
                pos = graph.tensors[tid].get("pos", "DDR")
                if pos in capacity:
                    by_pos[pos] += int(graph.tensors[tid].get("size", 0))
            # Total touched bytes upper-bound live residency. Charge a modest
            # fraction, and always charge a single Tensor larger than capacity.
            for pos, total in by_pos.items():
                largest = max((int(graph.tensors[t].get("size", 0)) for t in tids
                               if graph.tensors[t].get("pos") == pos), default=0)
                pressure += max(0, largest - capacity[pos])
                pressure += 0.15 * max(0, total - 4 * capacity[pos])
    l2_potential_cycles = 0.0
    if problem == 3:
        cache = settings["cache"]
        cap = cache["cache_capacity_bytes"]
        rate = cache["cache_bandwidth_bytes_per_cycle"]
        admitted = 0
        for size, repeated in sorted(l2_reuse_candidates, key=lambda x: (-x[1], x[0])):
            if size <= cap and admitted + size <= cap:
                admitted += size
                l2_potential_cycles += repeated * size * (1 / bandwidth - 1 / rate)
    wait = (settings["scene_a"]["task_cross_core_wait_cycles"]
            if problem == 1 else settings["scene_b"]["cross_core_copy_delay_cycles"])
    communication_cycles = (copy_bytes + shared_input_bytes + pressure) / bandwidth
    sync_cycles = wait * min(len(cross_edges), max(1, len(schedules) * 4))
    # A bounded cache potential reflects uncertain FIFO occupancy and access
    # timing. It is only a tie/ranking hint before the actual official run.
    objective = max(loads, default=0) + 0.35 * communication_cycles + sync_cycles
    if problem == 3:
        objective -= min(0.2 * objective, 0.25 * l2_potential_cycles)
    return {"objective": objective, "core_pipe_loads": loads,
            "cross_core_copy_bytes_proxy": copy_bytes,
            "repeated_input_bytes_proxy": shared_input_bytes,
            "memory_pressure_bytes_proxy": pressure,
            "l2_reuse_cycles_potential": l2_potential_cycles,
            "cross_core_pairs": len(cross_edges)}
