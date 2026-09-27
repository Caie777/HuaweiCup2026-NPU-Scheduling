#!/usr/bin/env python3
"""Export one officially verified experiment Plan in the official file format."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.evaluation import atomic_json, plan_digest
from common.evaluation import resolve_official_root

OFFICIAL = None


def load_official(value):
    global OFFICIAL, derive_multicore_plan
    OFFICIAL = resolve_official_root(value)
    sys.path.insert(0, str(OFFICIAL / "code"))
    from stub_multicore_cut_and_schedule import derive_multicore_plan


def export(summary_path: Path, *, case: str, problem: int, cores: int,
           algorithm: str, output_path: Path) -> dict:
    if OFFICIAL is None:
        raise ValueError("Provide the official attachment with --official-root or MATH_MODEL_OFFICIAL_ROOT")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))["rows"]
    matches = [row for row in rows if (row["case"], row["problem"],
                row["cores"], row["algorithm"]) ==
               (case, problem, cores, algorithm)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching experiment unit")
    row = matches[0]
    verified_statuses = {"PASS", "PARTIAL_PASS_BUDGET",
                         "PARTIAL_PASS_INTERRUPTED", "PARTIAL_PASS_ERROR"}
    if row["status"] not in verified_statuses:
        raise ValueError(f"Unit has no verified result: {row['status']}")
    if not row.get("best_plan_path") or not row.get("best_result_path"):
        raise ValueError("Unit has no verified Plan and official Result")
    plan = json.loads(Path(row["best_plan_path"]).read_text(encoding="utf-8"))
    result = json.loads(Path(row["best_result_path"]).read_text(encoding="utf-8"))
    prefix = f"p{problem}_{plan_digest(plan)}"
    if (Path(row["best_plan_path"]).name != f"{prefix}_plan.json" or
            Path(row["best_result_path"]).name != f"{prefix}_result.json"):
        raise ValueError("Plan and official Result are not the same evaluated candidate")
    graph_path = OFFICIAL / "data" / f"{case}.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    view = derive_multicore_plan(graph, plan)
    if view["num_cores"] != cores:
        raise ValueError("Verified Plan core count differs from selected unit")
    scene = "A" if problem == 1 else "B"
    if (result.get("input_graph") != graph_path.name or
            result.get("input_plan") != Path(row["best_plan_path"]).name or
            result.get("scene") != scene or result.get("num_cores") != cores or
            result.get("makespan") != row["official_makespan"] or
            (problem == 3 and result.get("problem") != 3) or
            (problem != 3 and result.get("problem") == 3)):
        raise ValueError("Official Result disagrees with selected unit")
    atomic_json(output_path, plan)
    return {"output": str(output_path), "case": case, "problem": problem,
            "cores": cores, "algorithm": algorithm,
            "official_makespan": result["makespan"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path,
                        help="local contest attachment directory; alternatively set MATH_MODEL_OFFICIAL_ROOT")
    parser.add_argument("--summary", required=True, type=Path,
                        help="runner summary.json (not summary.csv)")
    parser.add_argument("--case", required=True, help="case_001 or 001")
    parser.add_argument("--problem", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--cores", required=True, type=int)
    parser.add_argument("--algorithm", required=True,
                        choices=("V2plus", "RAMPplus", "OJOmacro"))
    parser.add_argument("--output", required=True, type=Path,
                        help="Official <case>_multicore_res.json destination")
    args = parser.parse_args()
    try:
        load_official(args.official_root)
    except ValueError as exc:
        parser.error(str(exc))
    case = args.case if args.case.startswith("case_") else f"case_{int(args.case):03d}"
    if args.output.name != f"{case}_multicore_res.json":
        parser.error("output filename must be <case>_multicore_res.json")
    print(json.dumps(export(args.summary, case=case, problem=args.problem,
                            cores=args.cores, algorithm=args.algorithm,
                            output_path=args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
