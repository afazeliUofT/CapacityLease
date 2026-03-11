#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "tools" / "flexible_solver_audit.py"
BACKUP = REPO_ROOT / "tools" / "flexible_solver_audit.py.pre_stable_cases.bak"


REPLACEMENT = """    case_builders: list[tuple[str, Any]] = [
        (
            "parallel_baseline",
            replace(
                base_config.solver,
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
        (
            "tight_baseline",
            replace(
                base_config.solver,
                root_xtol=min(base_config.solver.root_xtol, 1e-12),
                root_rtol=min(base_config.solver.root_rtol, 1e-12),
                max_root_iter=max(base_config.solver.max_root_iter, 400),
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
        (
            "dense_81",
            replace(
                base_config.solver,
                flexible_nM_search_points=max(base_config.solver.flexible_nM_search_points, 81),
                flexible_nV_search_points=max(base_config.solver.flexible_nV_search_points, 81),
                refine_levels=max(base_config.solver.refine_levels, 3),
                top_k_seeds=max(base_config.solver.top_k_seeds, 6),
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
        (
            "tight_dense_81",
            replace(
                base_config.solver,
                root_xtol=min(base_config.solver.root_xtol, 1e-12),
                root_rtol=min(base_config.solver.root_rtol, 1e-12),
                max_root_iter=max(base_config.solver.max_root_iter, 400),
                flexible_nM_search_points=max(base_config.solver.flexible_nM_search_points, 81),
                flexible_nV_search_points=max(base_config.solver.flexible_nV_search_points, 81),
                refine_levels=max(base_config.solver.refine_levels, 3),
                top_k_seeds=max(base_config.solver.top_k_seeds, 6),
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
        (
            "dense_81_fast_bvn_off",
            replace(
                base_config.solver,
                use_fast_bvn_if_available=False,
                flexible_nM_search_points=max(base_config.solver.flexible_nM_search_points, 81),
                flexible_nV_search_points=max(base_config.solver.flexible_nV_search_points, 81),
                refine_levels=max(base_config.solver.refine_levels, 3),
                top_k_seeds=max(base_config.solver.top_k_seeds, 6),
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
    ]

"""


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Target file not found: {TARGET}")

    text = TARGET.read_text()

    if "dense_81_fast_bvn_off" in text and "tight_baseline" in text:
        print("Audit case-builder patch already present. No changes made.")
        return

    shutil.copy2(TARGET, BACKUP)

    pattern = re.escape("    case_builders: list[tuple[str, Any]] = [") + r".*?(?=    save_json\()"
    new_text, count = re.subn(pattern, REPLACEMENT, text, count=1, flags=re.S)

    if count != 1:
        raise RuntimeError(
            "Could not patch case_builders block in tools/flexible_solver_audit.py.\n"
            "Expected one block starting with:\n"
            "    case_builders: list[tuple[str, Any]] = [\n"
            "and ending just before:\n"
            "    save_json(\n"
        )

    TARGET.write_text(new_text)
    print(f"Patched: {TARGET}")
    print(f"Backup written to: {BACKUP}")


if __name__ == "__main__":
    main()
