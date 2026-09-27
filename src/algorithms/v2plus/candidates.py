"""V2plus candidate construction from natural modules."""
from common.module_seed import NaturalModuleAggregator
from algorithms.v2plus.algorithm import V2PlusAggregator


def construct_rows(graph, cores, settings, common):
    rows = []
    modules = NaturalModuleAggregator(
        graph, cache_bytes=sum(settings["capacity"].values())).run()
    solved = V2PlusAggregator(graph, modules, cores, **common).run()[1]
    for item in solved["ranked_candidates"]:
        rows.append({"plan": item["plan"], "groups": item["groups"],
                     "label": item["source"], "source": "V2plus",
                     "search_stage": "constructor"})
    return rows
