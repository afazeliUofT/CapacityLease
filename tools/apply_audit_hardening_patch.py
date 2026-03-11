#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "tools" / "flexible_solver_audit.py"
BACKUP = REPO_ROOT / "tools" / "flexible_solver_audit.py.pre_audit_hardening.bak"


def replace_full(text: str, begin: str, end: str, replacement: str, label: str) -> str:
    pattern = re.escape(begin) + r".*?" + re.escape(end)
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(
            f"Could not patch {label}. Expected exactly one block.\n"
            f"BEGIN ANCHOR:\n{begin}\n\nEND ANCHOR:\n{end}\n"
        )
    return new_text


def replace_up_to(text: str, begin: str, end: str, replacement: str, label: str) -> str:
    pattern = re.escape(begin) + r".*?(?=" + re.escape(end) + r")"
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(
            f"Could not patch {label}. Expected exactly one block.\n"
            f"BEGIN ANCHOR:\n{begin}\n\nEND ANCHOR:\n{end}\n"
        )
    return new_text


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Target file not found: {TARGET}")

    text = TARGET.read_text()

    if "_MONOPOLY_FIXED_POINT_TOL" in text and "missing_best_row" in text and "exact_inf_match" in text:
        print("tools/flexible_solver_audit.py already appears patched. No changes made.")
        return

    shutil.copy2(TARGET, BACKUP)

    # ------------------------------------------------------------------
    # Patch 1: add explicit audit constants
    # Begin anchor: _NUMERIC_COMPARE_ATOL = 1e-6
    # End anchor:   _RESIDUAL_TOL = 1e-5
    # ------------------------------------------------------------------
    text = replace_full(
        text,
        "_NUMERIC_COMPARE_ATOL = 1e-6",
        "_RESIDUAL_TOL = 1e-5",
        """_NUMERIC_COMPARE_ATOL = 1e-6
_NUMERIC_COMPARE_RTOL = 1e-6
_RESIDUAL_TOL = 1e-5
_MONOPOLY_FIXED_POINT_TOL = 1e-7
_SENSITIVITY_BEST_CAPACITY_SHIFT_TOL_MBPS = 5.0
_SENSITIVITY_BEST_REVENUE_REL_TOL = 1e-3
""",
        "constants block",
    )

    # ------------------------------------------------------------------
    # Patch 2: treat matching infinities as equal in CSV regression
    # Begin anchor:
    #     if gn.notna().any() or cn.notna().any() or pd.api.types.is_numeric_dtype(gs) or pd.api.types.is_numeric_dtype(cs):
    # End anchor:
    #     else:
    # Replace everything from begin anchor up to (but not including) the end anchor.
    # ------------------------------------------------------------------
    text = replace_up_to(
        text,
        "        if gn.notna().any() or cn.notna().any() or pd.api.types.is_numeric_dtype(gs) or pd.api.types.is_numeric_dtype(cs):",
        "        else:",
        """        if gn.notna().any() or cn.notna().any() or pd.api.types.is_numeric_dtype(gs) or pd.api.types.is_numeric_dtype(cs):
            both_nan = gn.isna() & cn.isna()
            both_pos_inf = np.isposinf(gn) & np.isposinf(cn)
            both_neg_inf = np.isneginf(gn) & np.isneginf(cn)
            exact_inf_match = both_pos_inf | both_neg_inf
            exact_equal_mask = both_nan | exact_inf_match

            diff = (gn - cn).abs()
            diff = diff.where(~exact_equal_mask, 0.0)

            denom = np.maximum(np.maximum(gn.abs(), cn.abs()), 1.0)
            rel = diff / denom
            rel = rel.where(~exact_equal_mask, 0.0)

            bad = (
                ~((diff <= _NUMERIC_COMPARE_ATOL) | (rel <= _NUMERIC_COMPARE_RTOL))
            ) & ~exact_equal_mask

            finite_diff = diff.where(np.isfinite(diff))
            finite_rel = rel.where(np.isfinite(rel))

            status = "ok" if int(bad.sum()) == 0 else "numeric_mismatch"

            rows.append(
                {
                    "artifact": golden_path.name,
                    "column": column,
                    "status": status,
                    "details": "",
                    "max_abs_diff": _max_abs(finite_diff),
                    "max_rel_diff": _max_abs(finite_rel),
                    "n_bad": int(bad.sum()),
                    "golden_rows": golden_rows,
                    "candidate_rows": candidate_rows,
                }
            )
""",
        "numeric CSV comparison block",
    )

    # ------------------------------------------------------------------
    # Patch 3: record missing-best-row information in sensitivity summary
    # Begin anchor:     best_base = _best_row(base)
    # End anchor:       return out
    # Replace everything from begin anchor up to (but not including) the end anchor.
    # ------------------------------------------------------------------
    text = replace_up_to(
        text,
        "    best_base = _best_row(base)",
        "    return out",
        """    best_base = _best_row(base)

    best_cand = _best_row(cand)

    out["baseline_has_best_row"] = bool(best_base is not None)
    out["candidate_has_best_row"] = bool(best_cand is not None)
    out["missing_best_row"] = bool(best_base is None or best_cand is None)

    if best_base is None or best_cand is None:
        out["best_capacity_shift_mbps"] = np.nan
        out["best_revenue_abs_diff"] = np.nan
        out["best_revenue_rel_diff"] = np.nan

        out["best_capacity_baseline"] = (
            _safe_float(best_base["capacity_leased_mbps"]) if best_base is not None else np.nan
        )
        out["best_capacity_candidate"] = (
            _safe_float(best_cand["capacity_leased_mbps"]) if best_cand is not None else np.nan
        )

        out["best_revenue_baseline"] = (
            _safe_float(best_base["revenue_mno_total"]) if best_base is not None else np.nan
        )
        out["best_revenue_candidate"] = (
            _safe_float(best_cand["revenue_mno_total"]) if best_cand is not None else np.nan
        )

    else:
        best_capacity_shift = abs(
            _safe_float(best_base["capacity_leased_mbps"])
            - _safe_float(best_cand["capacity_leased_mbps"])
        )

        best_revenue_abs_diff = abs(
            _safe_float(best_base["revenue_mno_total"])
            - _safe_float(best_cand["revenue_mno_total"])
        )

        denom = max(
            abs(_safe_float(best_base["revenue_mno_total"])),
            abs(_safe_float(best_cand["revenue_mno_total"])),
            1.0,
        )

        out["best_capacity_shift_mbps"] = float(best_capacity_shift)
        out["best_revenue_abs_diff"] = float(best_revenue_abs_diff)
        out["best_revenue_rel_diff"] = float(best_revenue_abs_diff / denom)

        out["best_capacity_baseline"] = _safe_float(best_base["capacity_leased_mbps"])
        out["best_capacity_candidate"] = _safe_float(best_cand["capacity_leased_mbps"])
        out["best_revenue_baseline"] = _safe_float(best_base["revenue_mno_total"])
        out["best_revenue_candidate"] = _safe_float(best_cand["revenue_mno_total"])

""",
        "best-row sensitivity summary block",
    )

    # ------------------------------------------------------------------
    # Patch 4: tighten the verdict logic so it tells the truth
    # Begin anchor:     regression_pass = bool(
    # End anchor:       sensitivity_pass = len(material_shift_cases) == 0
    # Replace the full block including the end anchor line.
    # ------------------------------------------------------------------
    text = replace_full(
        text,
        "    regression_pass = bool(",
        "    sensitivity_pass = len(material_shift_cases) == 0",
        """    regression_pass = bool(
        regression_df[~regression_df["status"].eq("ok")].empty
        and summary_compare_df[~summary_compare_df["status"].eq("ok")].empty
    )

    monopoly_pass = bool(
        _max_abs(monopoly_diag["fixed_point_residual"]) <= _MONOPOLY_FIXED_POINT_TOL
        and _max_abs(monopoly_diag["rate_identity_error"]) <= 1e-10
        and _max_abs(monopoly_diag["revenue_identity_error"]) <= 1e-10
        and _max_abs(monopoly_diag["max_acceptance_column_error"]) <= 1e-8
    )

    market_pass = bool(
        _max_abs(mc_capacity_diag["pV_formula_error"]) <= 1e-10
        and _max_abs(mc_capacity_diag["pM_root_residual"]) <= 1e-6
        and _max_abs(mc_capacity_diag["revenue_identity_error"]) <= 1e-8
        and _max_abs(mc_nm_diag["pV_formula_error"]) <= 1e-10
        and _max_abs(mc_nm_diag["pM_root_residual"]) <= 1e-6
        and _max_abs(mc_nm_diag["revenue_identity_error"]) <= 1e-8
        and bool(mc_capacity_diag.loc[mc_capacity_diag["valid"], "interval_contains"].all())
        and bool(mc_nm_diag.loc[mc_nm_diag["valid"], "interval_contains"].all())
    )

    flexible_pass = bool(
        _max_abs(flex_capacity_diag["revenue_identity_error"]) <= 1e-8
        and _max_abs(flex_nm_diag["revenue_identity_error"]) <= 1e-8
        and _max_abs(flex_capacity_diag["mvno_retained_identity_error"]) <= 1e-8
        and _max_abs(flex_nm_diag["mvno_retained_identity_error"]) <= 1e-8
        and _max_abs(flex_capacity_diag["residual_norm2"]) <= _RESIDUAL_TOL
        and _max_abs(flex_nm_diag["residual_norm2"]) <= _RESIDUAL_TOL
    )

    parallel_row = sensitivity_df.loc[sensitivity_df["case"] == "parallel_baseline"]

    parallel_pass = True

    if not parallel_row.empty:
        pr = parallel_row.iloc[0]
        raw_parallel_missing_best = pr.get("missing_best_row", np.nan)
        parallel_missing_best = bool(raw_parallel_missing_best) if pd.notna(raw_parallel_missing_best) else True

        parallel_pass = bool(
            pr.get("capacity_grid_identical", False)
            and _safe_float(pr.get("best_capacity_shift_mbps")) <= 0.0 + 1e-12
            and _safe_float(pr.get("best_revenue_abs_diff")) <= 1e-10
            and _safe_float(pr.get("max_abs_diff_revenue_mno_total")) <= 1e-10
            and _safe_float(pr.get("max_abs_diff_nM")) <= 1e-10
            and _safe_float(pr.get("max_abs_diff_nV")) <= 1e-10
            and int(pr.get("valid_status_flip_count", 0)) == 0
            and not parallel_missing_best
        )

    material_shift_cases = []

    for _, row in sensitivity_df.iterrows():
        case_name = str(row["case"])
        if case_name == "parallel_baseline":
            continue

        raw_flip_count = row.get("valid_status_flip_count", np.nan)
        valid_flip_count = int(raw_flip_count) if pd.notna(raw_flip_count) else 0

        raw_missing_best = row.get("missing_best_row", np.nan)
        missing_best_row = bool(raw_missing_best) if pd.notna(raw_missing_best) else True

        best_capacity_shift = _safe_float(row.get("best_capacity_shift_mbps"))
        best_revenue_rel_diff = _safe_float(row.get("best_revenue_rel_diff"))

        case_has_material_shift = (
            valid_flip_count != 0
            or missing_best_row
            or not np.isfinite(best_capacity_shift)
            or not np.isfinite(best_revenue_rel_diff)
            or best_capacity_shift > _SENSITIVITY_BEST_CAPACITY_SHIFT_TOL_MBPS + 1e-12
            or best_revenue_rel_diff > _SENSITIVITY_BEST_REVENUE_REL_TOL
        )

        if case_has_material_shift:
            material_shift_cases.append(case_name)

    sensitivity_pass = len(material_shift_cases) == 0
""",
        "verdict block",
    )

    TARGET.write_text(text)
    print(f"Patched: {TARGET}")
    print(f"Backup written to: {BACKUP}")


if __name__ == "__main__":
    main()
