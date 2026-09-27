"""Locate a user supplied copy of the contest attachment at runtime."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_official_root(value: Path | str | None = None) -> Path:
    supplied = value or os.environ.get("MATH_MODEL_OFFICIAL_ROOT")
    if not supplied:
        raise ValueError(
            "Provide --official-root or set MATH_MODEL_OFFICIAL_ROOT to your "
            "local contest attachment directory (containing data/ and code/)."
        )
    root = Path(supplied).expanduser().resolve()
    required = (
        root / "data" / "config.txt",
        root / "code" / "stub_multicore_cut_and_schedule.py",
        root / "code" / "evaluation_validation.py",
        root / "code" / "multicore_cut_evaluate_problem_1.py",
        root / "code" / "multicore_cut_evaluate_problem_2.py",
        root / "code" / "multicore_cut_evaluate_problem_3.py",
        root / "code" / "contest_io.py",
        root / "code" / "schedule_step1.py",
        root / "code" / "schedule_step2.py",
        root / "code" / "schedule_step3.py",
        root / "code" / "singlecore_evaluate.py",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete official attachment at {root}: missing {', '.join(missing)}")
    return root
