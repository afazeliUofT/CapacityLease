#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy


def _pin_blas_threads() -> None:
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ.setdefault(name, "1")


def _ensure_repo_on_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate deterministic outputs, compare them against frozen goldens, "
            "audit fixed-point/root residuals, and run flexible-solver search-grid "
            "sensitivity checks."
        )
    )
    parser.add_argument("--repo-root", default=".", help="CapacityLease repo root.")
    parser.add_argument(
        "--config",
        default="configs/paper_literal.json",
        help="Config JSON relative to repo root.",
    )
    parser.add_argument(
        "--golden-dir",
        default="tests/golden/paper_literal",
        help=(
            "Frozen golden directory relative to repo root. It must contain data/*.csv "
            "and summary.json."
        ),
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where the audit bundle will be written.",
    )
    parser.add_argument(
        "--baseline-workers",
        type=int,
        default=1,
        help="Workers for the serial baseline regeneration used for regression and diagnostics.",
    )
    parser.add_argument(
        "--sensitivity-workers",
        type=int,
        default=32,
        help="Workers for parallel invariance and sensitivity cases.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="ProcessPoolExecutor chunksize for regenerated runs.",
    )
    parser.add_argument(
        "--pin-blas-threads",
        action="store_true",
        help="Set BLAS/OpenMP thread counts to 1 before importing the package.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with status 1 when the overall verdict is false.",
    )
    return parser.parse_args()


FILES_AND_KEYS: dict[str, str] = {
    "monopoly_curve.csv": "price",
    "market_clearing_capacity_curve.csv": "capacity_leased_mbps",
    "flexible_capacity_curve.csv": "capacity_leased_mbps",
    "market_clearing_nM_curve.csv": "nM",
    "flexible_nM_curve.csv": "nM",
}


def _flatten_dict(data: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_dict(value, next_prefix))
        return out
    if isinstance(data, list):
        for idx, value in enumerate(data):
            next_prefix = f"{prefix}[{idx}]"
            out.update(_flatten_dict(value, next_prefix))
        return out
    out[prefix] = data
    return out


_IGNORE_SUMMARY_KEYS = {
    "config.solver.workers",
    "config.solver.chunksize",
}


_NUMERIC_COMPARE_ATOL = 1e-6
_NUMERIC_COMPARE_RTOL = 1e-6
_RESIDUAL_TOL = 1e-5


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")



def _max_abs(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))



def _align_frame(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key in df.columns:
        return df.sort_values(key).reset_index(drop=True)
    return df.reset_index(drop=True)



def _series_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")



def compare_csvs(golden_path: Path, candidate_path: Path, key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not golden_path.exists():
        rows.append(
            {
                "artifact": golden_path.name,
                "column": "__file__",
                "status": "missing_golden",
                "details": str(golden_path),
                "max_abs_diff": np.nan,
                "max_rel_diff": np.nan,
                "n_bad": np.nan,
                "golden_rows": np.nan,
                "candidate_rows": np.nan,
            }
        )
        return pd.DataFrame(rows)
    if not candidate_path.exists():
        rows.append(
            {
                "artifact": golden_path.name,
                "column": "__file__",
                "status": "missing_candidate",
                "details": str(candidate_path),
                "max_abs_diff": np.nan,
                "max_rel_diff": np.nan,
                "n_bad": np.nan,
                "golden_rows": np.nan,
                "candidate_rows": np.nan,
            }
        )
        return pd.DataFrame(rows)

    g = _align_frame(pd.read_csv(golden_path), key)
    c = _align_frame(pd.read_csv(candidate_path), key)

    golden_rows = int(len(g))
    candidate_rows = int(len(c))
    all_columns = sorted(set(g.columns) | set(c.columns))

    if golden_rows != candidate_rows:
        rows.append(
            {
                "artifact": golden_path.name,
                "column": "__row_count__",
                "status": "row_count_mismatch",
                "details": f"golden={golden_rows}, candidate={candidate_rows}",
                "max_abs_diff": np.nan,
                "max_rel_diff": np.nan,
                "n_bad": abs(golden_rows - candidate_rows),
                "golden_rows": golden_rows,
                "candidate_rows": candidate_rows,
            }
        )

    n = min(golden_rows, candidate_rows)
    g = g.iloc[:n].copy()
    c = c.iloc[:n].copy()

    for column in all_columns:
        if column not in g.columns:
            rows.append(
                {
                    "artifact": golden_path.name,
                    "column": column,
                    "status": "missing_from_golden",
                    "details": "column missing from golden",
                    "max_abs_diff": np.nan,
                    "max_rel_diff": np.nan,
                    "n_bad": n,
                    "golden_rows": golden_rows,
                    "candidate_rows": candidate_rows,
                }
            )
            continue
        if column not in c.columns:
            rows.append(
                {
                    "artifact": golden_path.name,
                    "column": column,
                    "status": "missing_from_candidate",
                    "details": "column missing from candidate",
                    "max_abs_diff": np.nan,
                    "max_rel_diff": np.nan,
                    "n_bad": n,
                    "golden_rows": golden_rows,
                    "candidate_rows": candidate_rows,
                }
            )
            continue

        gs = g[column]
        cs = c[column]
        gn = _series_numeric(gs)
        cn = _series_numeric(cs)

        if gn.notna().any() or cn.notna().any() or pd.api.types.is_numeric_dtype(gs) or pd.api.types.is_numeric_dtype(cs):
            diff = (gn - cn).abs()
            denom = np.maximum(np.maximum(gn.abs(), cn.abs()), 1.0)
            rel = diff / denom
            bad = (~((diff <= _NUMERIC_COMPARE_ATOL) | (rel <= _NUMERIC_COMPARE_RTOL))) & ~(gn.isna() & cn.isna())
            status = "ok" if int(bad.sum()) == 0 else "numeric_mismatch"
            rows.append(
                {
                    "artifact": golden_path.name,
                    "column": column,
                    "status": status,
                    "details": "",
                    "max_abs_diff": _max_abs(diff),
                    "max_rel_diff": _max_abs(rel),
                    "n_bad": int(bad.sum()),
                    "golden_rows": golden_rows,
                    "candidate_rows": candidate_rows,
                }
            )
        else:
            gtxt = gs.astype(str).where(~gs.isna(), "<NA>")
            ctxt = cs.astype(str).where(~cs.isna(), "<NA>")
            bad = gtxt != ctxt
            status = "ok" if int(bad.sum()) == 0 else "value_mismatch"
            rows.append(
                {
                    "artifact": golden_path.name,
                    "column": column,
                    "status": status,
                    "details": "",
                    "max_abs_diff": np.nan,
                    "max_rel_diff": np.nan,
                    "n_bad": int(bad.sum()),
                    "golden_rows": golden_rows,
                    "candidate_rows": candidate_rows,
                }
            )

    return pd.DataFrame(rows)



def compare_summary(golden_path: Path, candidate_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not golden_path.exists() or not candidate_path.exists():
        rows.append(
            {
                "key": "__file__",
                "status": "missing_file",
                "details": f"golden_exists={golden_path.exists()}, candidate_exists={candidate_path.exists()}",
                "max_abs_diff": np.nan,
                "max_rel_diff": np.nan,
            }
        )
        return pd.DataFrame(rows)

    g = _flatten_dict(json.loads(golden_path.read_text()))
    c = _flatten_dict(json.loads(candidate_path.read_text()))
    all_keys = sorted((set(g) | set(c)) - _IGNORE_SUMMARY_KEYS)

    for key in all_keys:
        if key not in g:
            rows.append(
                {
                    "key": key,
                    "status": "missing_from_golden",
                    "details": "",
                    "max_abs_diff": np.nan,
                    "max_rel_diff": np.nan,
                }
            )
            continue
        if key not in c:
            rows.append(
                {
                    "key": key,
                    "status": "missing_from_candidate",
                    "details": "",
                    "max_abs_diff": np.nan,
                    "max_rel_diff": np.nan,
                }
            )
            continue

        gv = g[key]
        cv = c[key]
        try:
            gf = float(gv)
            cf = float(cv)
            if math.isnan(gf) and math.isnan(cf):
                status = "ok"
                max_abs = 0.0
                max_rel = 0.0
            else:
                diff = abs(gf - cf)
                denom = max(abs(gf), abs(cf), 1.0)
                rel = diff / denom
                status = "ok" if diff <= _NUMERIC_COMPARE_ATOL or rel <= _NUMERIC_COMPARE_RTOL else "numeric_mismatch"
                max_abs = diff
                max_rel = rel
        except Exception:
            status = "ok" if gv == cv else "value_mismatch"
            max_abs = np.nan
            max_rel = np.nan
        rows.append(
            {
                "key": key,
                "status": status,
                "details": "",
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
            }
        )
    return pd.DataFrame(rows)



def build_summary(config: Any, monopoly: Any, mc_curve: pd.DataFrame, flex_curve: pd.DataFrame) -> dict[str, Any]:
    def _best_row(df: pd.DataFrame) -> dict[str, Any] | None:
        if df.empty:
            return None
        if "revenue_mno_total" not in df.columns:
            return None
        series = pd.to_numeric(df["revenue_mno_total"], errors="coerce").dropna()
        if series.empty:
            return None
        idx = int(series.idxmax())
        return df.loc[idx].to_dict()

    return {
        "config": {
            "model": {
                "group_sizes": config.model.group_sizes.tolist(),
                "alpha": config.model.alpha.tolist(),
                "beta": config.model.beta.tolist(),
                "monopoly_noise_mean": config.model.monopoly_noise_mean.tolist(),
                "monopoly_noise_sd": config.model.monopoly_noise_sd.tolist(),
                "mno_noise_mean": config.model.mno_noise_mean.tolist(),
                "mno_noise_sd": config.model.mno_noise_sd.tolist(),
                "mvno_noise_mean": config.model.mvno_noise_mean.tolist(),
                "mvno_noise_sd": config.model.mvno_noise_sd.tolist(),
                "capacity_mbps": config.model.capacity_mbps,
                "delta": config.model.delta,
                "zeta": config.model.zeta,
                "lambda_retention": config.model.lambda_retention,
                "participation_fee": config.model.participation_fee,
            },
            "solver": asdict(config.solver),
        },
        "monopoly": {
            "critical_price": monopoly.critical_price,
            "optimal_price": monopoly.optimal_price,
            "optimal_subscribers": monopoly.optimal_subscribers,
            "optimal_rate": monopoly.optimal_rate,
            "optimal_revenue": monopoly.optimal_revenue,
        },
        "market_clearing_best_capacity": _best_row(mc_curve),
        "flexible_best_capacity": _best_row(flex_curve),
    }



def generate_data_only(repo_root: Path, config: Any, outdir: Path) -> dict[str, Any]:
    from capacity_lease.flexible import solve_flexible_capacity_curve, solve_flexible_nm_curve
    from capacity_lease.market_clearing import solve_market_clearing_capacity_curve, solve_market_clearing_nm_curve
    from capacity_lease.monopoly import solve_monopoly_problem
    from capacity_lease.utils import save_dataframe, save_json

    data_dir = outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    monopoly = solve_monopoly_problem(config.model, config.solver)
    save_dataframe(monopoly.curve, data_dir / "monopoly_curve.csv")

    mc_curve = solve_market_clearing_capacity_curve(config.model, config.solver)
    save_dataframe(mc_curve, data_dir / "market_clearing_capacity_curve.csv")

    flex_curve = solve_flexible_capacity_curve(config.model, config.solver, monopoly.optimal_revenue)
    save_dataframe(flex_curve, data_dir / "flexible_capacity_curve.csv")

    mc_nm = solve_market_clearing_nm_curve(
        config.model,
        config.solver,
        config.solver.reported_market_clearing_capacity_mbps,
    )
    save_dataframe(mc_nm, data_dir / "market_clearing_nM_curve.csv")

    flex_nm = solve_flexible_nm_curve(
        config.model,
        config.solver,
        monopoly.optimal_revenue,
        config.solver.reported_flexible_capacity_mbps,
    )
    save_dataframe(flex_nm, data_dir / "flexible_nM_curve.csv")

    summary = build_summary(config, monopoly, mc_curve, flex_curve)
    save_json(summary, outdir / "summary.json")

    return {
        "monopoly": monopoly,
        "mc_curve": mc_curve,
        "flex_curve": flex_curve,
        "mc_nm": mc_nm,
        "flex_nm": flex_nm,
        "summary": summary,
    }



def monopoly_diagnostics(model: Any, solver: Any, monopoly_curve: pd.DataFrame, critical_price_value: float) -> pd.DataFrame:
    from capacity_lease.probability import normal_cdf, normal_sf

    rows: list[dict[str, Any]] = []
    for _, row in monopoly_curve.iterrows():
        price = _safe_float(row.get("price"))
        feasible = bool(row.get("feasible", False))
        subscribers = _safe_float(row.get("subscribers"))
        rate = _safe_float(row.get("rate_mbps"))
        revenue = _safe_float(row.get("revenue"))

        fixed_point_residual = np.nan
        rate_identity_error = np.nan
        revenue_identity_error = np.nan
        max_acceptance_column_error = np.nan
        acceptance_prob_bounds_ok = False
        price_bounds_ok = bool(np.isfinite(price) and 0.0 <= price <= critical_price_value + 1e-9)

        if feasible and np.isfinite(subscribers) and subscribers > 0.0:
            const = model.beta * price - model.alpha * np.log(model.capacity_mbps / model.delta)
            threshold_subscribers = const + model.alpha * np.log(subscribers)
            cdfs = normal_cdf(threshold_subscribers, model.monopoly_noise_mean, model.monopoly_noise_sd)
            fixed_point_residual = float(model.N - np.dot(model.group_sizes, cdfs) - subscribers)

            rate_expected = model.capacity_mbps / (model.delta * subscribers)
            rate_identity_error = float(rate - rate_expected)
            revenue_identity_error = float(revenue - price * subscribers)

            threshold_rate = model.beta * price - model.alpha * np.log(rate_expected)
            acceptance = np.asarray(
                normal_sf(threshold_rate, model.monopoly_noise_mean, model.monopoly_noise_sd),
                dtype=float,
            )
            acceptance_prob_bounds_ok = bool(np.all((acceptance >= -1e-12) & (acceptance <= 1.0 + 1e-12)))
            col_errors: list[float] = []
            for g in range(model.G):
                column_name = f"A_group_{g + 1}"
                if column_name in row.index:
                    col_errors.append(abs(_safe_float(row[column_name]) - float(acceptance[g])))
            max_acceptance_column_error = float(max(col_errors)) if col_errors else 0.0

        rows.append(
            {
                "price": price,
                "feasible": feasible,
                "subscribers": subscribers,
                "rate_mbps": rate,
                "revenue": revenue,
                "price_bounds_ok": price_bounds_ok,
                "fixed_point_residual": fixed_point_residual,
                "rate_identity_error": rate_identity_error,
                "revenue_identity_error": revenue_identity_error,
                "max_acceptance_column_error": max_acceptance_column_error,
                "acceptance_prob_bounds_ok": acceptance_prob_bounds_ok,
            }
        )
    return pd.DataFrame(rows)



def market_clearing_diagnostics(model: Any, solver: Any, df: pd.DataFrame) -> pd.DataFrame:
    from capacity_lease.market_clearing import profitability_interval, mvno_price
    from capacity_lease.probability import normal_cdf

    rows: list[dict[str, Any]] = []
    mu_diff = model.mno_noise_mean - model.mvno_noise_mean
    sd_diff = np.sqrt(model.mno_noise_sd**2 + model.mvno_noise_sd**2)

    for _, row in df.iterrows():
        CV = _safe_float(row.get("capacity_leased_mbps"))
        nV = _safe_float(row.get("nV"))
        nM = _safe_float(row.get("nM"))
        pV = _safe_float(row.get("pV"))
        pM = _safe_float(row.get("pM"))
        valid = bool(row.get("valid", False))
        interval = profitability_interval(model, solver, CV) if np.isfinite(CV) else None

        interval_contains = False
        pV_formula_error = np.nan
        pM_root_residual = np.nan
        revenue_identity_error = np.nan

        if valid and all(np.isfinite(v) for v in [CV, nV, nM, pV, pM]) and nV > 0.0 and nM > 0.0:
            if interval is not None:
                interval_contains = bool(interval[0] - 1e-9 <= nV <= interval[1] + 1e-9)
            pV_formula = mvno_price(model, solver, CV, nV)
            pV_formula_error = float(pV - pV_formula)
            const = model.alpha * np.log((nM * CV) / ((model.capacity_mbps - CV) * nV)) - model.beta * pV
            cdfs = normal_cdf(model.beta * pM + const, mu_diff, sd_diff)
            pM_root_residual = float(np.dot(model.group_sizes, cdfs) - nV)
            revenue_expected = nM * pM + (1.0 - model.lambda_retention) * nV * pV - model.participation_fee
            revenue_identity_error = float(_safe_float(row.get("revenue_mno_total")) - revenue_expected)

        rows.append(
            {
                "capacity_leased_mbps": CV,
                "nM": nM,
                "nV": nV,
                "valid": valid,
                "interval_contains": interval_contains,
                "pV_formula_error": pV_formula_error,
                "pM_root_residual": pM_root_residual,
                "revenue_identity_error": revenue_identity_error,
            }
        )
    return pd.DataFrame(rows)



def flexible_diagnostics(model: Any, solver: Any, monopoly_revenue: float, df: pd.DataFrame) -> pd.DataFrame:
    from capacity_lease.flexible import FlexibleSolverLocal

    local = FlexibleSolverLocal(model, solver, monopoly_revenue)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        CV = _safe_float(row.get("capacity_leased_mbps"))
        nM = _safe_float(row.get("nM"))
        nV = _safe_float(row.get("nV"))
        pM = _safe_float(row.get("pM"))
        pV = _safe_float(row.get("pV"))
        valid = bool(row.get("valid", False))

        theta_m_error = np.nan
        theta_v_error = np.nan
        residual_norm2 = np.nan
        revenue_identity_error = np.nan
        mvno_retained_identity_error = np.nan
        monopoly_margin = np.nan
        residual_ok = False

        if valid and all(np.isfinite(v) for v in [CV, nM, nV, pM, pV]) and nM > 0.0 and nV > 0.0:
            theta_m = local.kernel.theta_m(pM, pV, CV, nM, nV)
            theta_v = local.kernel.theta_v(pM, pV, CV, nM, nV)
            theta_m_error = float(theta_m - nM)
            theta_v_error = float(theta_v - nV)
            residual_norm2 = float(math.hypot(theta_m_error, theta_v_error))
            revenue_expected = nM * pM + (1.0 - model.lambda_retention) * nV * pV - model.participation_fee
            revenue_identity_error = float(_safe_float(row.get("revenue_mno_total")) - revenue_expected)
            if "revenue_mvno_retained" in row.index:
                mvno_retained_identity_error = float(
                    _safe_float(row.get("revenue_mvno_retained")) - model.lambda_retention * nV * pV
                )
            monopoly_margin = float(_safe_float(row.get("revenue_mno_total")) - monopoly_revenue)
            residual_ok = bool(residual_norm2 <= _RESIDUAL_TOL)

        rows.append(
            {
                "capacity_leased_mbps": CV,
                "nM": nM,
                "nV": nV,
                "valid": valid,
                "theta_m_error": theta_m_error,
                "theta_v_error": theta_v_error,
                "residual_norm2": residual_norm2,
                "residual_ok": residual_ok,
                "revenue_identity_error": revenue_identity_error,
                "mvno_retained_identity_error": mvno_retained_identity_error,
                "monopoly_margin": monopoly_margin,
            }
        )
    return pd.DataFrame(rows)



def _best_row(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    if "revenue_mno_total" not in df.columns:
        return None
    series = pd.to_numeric(df["revenue_mno_total"], errors="coerce").dropna()
    if series.empty:
        return None
    idx = int(series.idxmax())
    return df.loc[idx].to_dict()



def flexible_case_curve(model: Any, solver: Any, monopoly_revenue: float) -> pd.DataFrame:
    from capacity_lease.flexible import solve_flexible_capacity_curve

    return solve_flexible_capacity_curve(model, solver, monopoly_revenue)



def flexible_local_probe(model: Any, solver: Any, monopoly_revenue: float, capacities: list[float]) -> pd.DataFrame:
    from capacity_lease.flexible import FlexibleSolverLocal

    local = FlexibleSolverLocal(model, solver, monopoly_revenue)
    rows: list[dict[str, Any]] = []
    for CV in capacities:
        row = local.mno_best_response_for_capacity(float(CV))
        row = dict(row)
        row["capacity_leased_mbps"] = float(CV)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("capacity_leased_mbps").reset_index(drop=True)



def compare_flexible_case_to_baseline(case_name: str, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    base = _align_frame(baseline, "capacity_leased_mbps")
    cand = _align_frame(candidate, "capacity_leased_mbps")
    merged = base.merge(
        cand,
        how="outer",
        on="capacity_leased_mbps",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
    )

    out: dict[str, Any] = {
        "case": case_name,
        "merged_rows": int(len(merged)),
        "capacity_grid_identical": bool((merged["_merge"] == "both").all()),
    }

    for field in ["revenue_mno_total", "revenue_mvno_gross", "nM", "nV", "pM", "pV"]:
        left = pd.to_numeric(merged.get(f"{field}_baseline"), errors="coerce")
        right = pd.to_numeric(merged.get(f"{field}_candidate"), errors="coerce")
        diff = (left - right).abs()
        valid = diff[np.isfinite(diff)]
        out[f"max_abs_diff_{field}"] = float(valid.max()) if not valid.empty else 0.0

    if "valid_baseline" in merged.columns and "valid_candidate" in merged.columns:
        vb = merged["valid_baseline"].fillna(False).astype(bool)
        vc = merged["valid_candidate"].fillna(False).astype(bool)
        out["valid_status_flip_count"] = int((vb != vc).sum())
    else:
        out["valid_status_flip_count"] = np.nan

    best_base = _best_row(base)
    best_cand = _best_row(cand)
    if best_base is None or best_cand is None:
        out["best_capacity_shift_mbps"] = np.nan
        out["best_revenue_abs_diff"] = np.nan
        out["best_revenue_rel_diff"] = np.nan
    else:
        best_capacity_shift = abs(_safe_float(best_base["capacity_leased_mbps"]) - _safe_float(best_cand["capacity_leased_mbps"]))
        best_revenue_abs_diff = abs(_safe_float(best_base["revenue_mno_total"]) - _safe_float(best_cand["revenue_mno_total"]))
        denom = max(abs(_safe_float(best_base["revenue_mno_total"])), abs(_safe_float(best_cand["revenue_mno_total"])), 1.0)
        out["best_capacity_shift_mbps"] = float(best_capacity_shift)
        out["best_revenue_abs_diff"] = float(best_revenue_abs_diff)
        out["best_revenue_rel_diff"] = float(best_revenue_abs_diff / denom)
        out["best_capacity_baseline"] = _safe_float(best_base["capacity_leased_mbps"])
        out["best_capacity_candidate"] = _safe_float(best_cand["capacity_leased_mbps"])
        out["best_revenue_baseline"] = _safe_float(best_base["revenue_mno_total"])
        out["best_revenue_candidate"] = _safe_float(best_cand["revenue_mno_total"])

    return out



def write_environment(outdir: Path) -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "env": {
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            ]
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "environment.json").write_text(json.dumps(payload, indent=2, sort_keys=True))



def main() -> int:
    args = _parse_args()
    if args.pin_blas_threads:
        _pin_blas_threads()

    repo_root = Path(args.repo_root).resolve()
    _ensure_repo_on_path(repo_root)

    from capacity_lease.config import load_config
    from capacity_lease.utils import save_dataframe, save_json

    config_path = repo_root / args.config
    golden_dir = repo_root / args.golden_dir
    outdir = Path(args.outdir).resolve()
    regenerated_dir = outdir / "regenerated_serial"
    diagnostics_dir = outdir / "diagnostics"
    regression_dir = outdir / "regression"
    sensitivity_dir = outdir / "sensitivity"

    outdir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    regression_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    write_environment(outdir)

    base_config = load_config(config_path)
    serial_solver = replace(
        base_config.solver,
        workers=int(args.baseline_workers),
        chunksize=int(args.chunksize),
    )
    serial_config = replace(base_config, solver=serial_solver)

    regenerated = generate_data_only(repo_root, serial_config, regenerated_dir)

    regression_frames: list[pd.DataFrame] = []
    golden_data_dir = golden_dir / "data"
    candidate_data_dir = regenerated_dir / "data"
    for filename, key in FILES_AND_KEYS.items():
        regression_frames.append(compare_csvs(golden_data_dir / filename, candidate_data_dir / filename, key))
    regression_df = pd.concat(regression_frames, ignore_index=True)
    save_dataframe(regression_df, regression_dir / "csv_regression_compare.csv")

    summary_compare_df = compare_summary(golden_dir / "summary.json", regenerated_dir / "summary.json")
    save_dataframe(summary_compare_df, regression_dir / "summary_regression_compare.csv")

    monopoly_diag = monopoly_diagnostics(
        serial_config.model,
        serial_config.solver,
        regenerated["monopoly"].curve,
        regenerated["monopoly"].critical_price,
    )
    save_dataframe(monopoly_diag, diagnostics_dir / "monopoly_diagnostics.csv")

    mc_capacity_diag = market_clearing_diagnostics(serial_config.model, serial_config.solver, regenerated["mc_curve"])
    save_dataframe(mc_capacity_diag, diagnostics_dir / "market_clearing_capacity_diagnostics.csv")

    mc_nm_diag = market_clearing_diagnostics(serial_config.model, serial_config.solver, regenerated["mc_nm"])
    save_dataframe(mc_nm_diag, diagnostics_dir / "market_clearing_nM_diagnostics.csv")

    flex_capacity_diag = flexible_diagnostics(
        serial_config.model,
        serial_config.solver,
        regenerated["monopoly"].optimal_revenue,
        regenerated["flex_curve"],
    )
    save_dataframe(flex_capacity_diag, diagnostics_dir / "flexible_capacity_diagnostics.csv")

    flex_nm_diag = flexible_diagnostics(
        serial_config.model,
        serial_config.solver,
        regenerated["monopoly"].optimal_revenue,
        regenerated["flex_nm"],
    )
    save_dataframe(flex_nm_diag, diagnostics_dir / "flexible_nM_diagnostics.csv")

    baseline_best = _best_row(regenerated["flex_curve"])
    baseline_best_capacity = (
        _safe_float(baseline_best["capacity_leased_mbps"])
        if baseline_best is not None
        else _safe_float(serial_config.solver.reported_flexible_capacity_mbps)
    )
    probe_capacities = sorted(
        {
            float(serial_config.solver.reported_flexible_capacity_mbps),
            *[
                float(cv)
                for cv in [
                    baseline_best_capacity - 20.0,
                    baseline_best_capacity - 10.0,
                    baseline_best_capacity - 5.0,
                    baseline_best_capacity,
                    baseline_best_capacity + 5.0,
                    baseline_best_capacity + 10.0,
                    baseline_best_capacity + 20.0,
                ]
                if 0.0 < cv < serial_config.model.capacity_mbps
            ],
        }
    )

    case_builders: list[tuple[str, Any]] = [
        (
            "parallel_baseline",
            replace(
                base_config.solver,
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
        (
            "dense_41",
            replace(
                base_config.solver,
                flexible_nM_search_points=max(base_config.solver.flexible_nM_search_points, 41),
                flexible_nV_search_points=max(base_config.solver.flexible_nV_search_points, 41),
                refine_levels=max(base_config.solver.refine_levels, 3),
                top_k_seeds=max(base_config.solver.top_k_seeds, 3),
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
        (
            "dense_61",
            replace(
                base_config.solver,
                flexible_nM_search_points=max(base_config.solver.flexible_nM_search_points, 61),
                flexible_nV_search_points=max(base_config.solver.flexible_nV_search_points, 61),
                refine_levels=max(base_config.solver.refine_levels, 3),
                top_k_seeds=max(base_config.solver.top_k_seeds, 4),
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
        (
            "tight_dense_61",
            replace(
                base_config.solver,
                root_xtol=min(base_config.solver.root_xtol, 1e-12),
                root_rtol=min(base_config.solver.root_rtol, 1e-12),
                max_root_iter=max(base_config.solver.max_root_iter, 400),
                flexible_nM_search_points=max(base_config.solver.flexible_nM_search_points, 61),
                flexible_nV_search_points=max(base_config.solver.flexible_nV_search_points, 61),
                refine_levels=max(base_config.solver.refine_levels, 3),
                top_k_seeds=max(base_config.solver.top_k_seeds, 4),
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
        (
            "dense_41_fast_bvn_off",
            replace(
                base_config.solver,
                use_fast_bvn_if_available=False,
                flexible_nM_search_points=max(base_config.solver.flexible_nM_search_points, 41),
                flexible_nV_search_points=max(base_config.solver.flexible_nV_search_points, 41),
                refine_levels=max(base_config.solver.refine_levels, 3),
                top_k_seeds=max(base_config.solver.top_k_seeds, 3),
                workers=int(args.sensitivity_workers),
                chunksize=int(args.chunksize),
            ),
        ),
    ]

    save_json(
        {name: asdict(solver_case) for name, solver_case in case_builders},
        sensitivity_dir / "case_definitions.json",
    )

    baseline_curve = regenerated["flex_curve"].copy()
    sensitivity_rows: list[dict[str, Any]] = []
    probe_frames: list[pd.DataFrame] = []

    for case_name, case_solver in case_builders:
        case_dir = sensitivity_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_curve = flexible_case_curve(serial_config.model, case_solver, regenerated["monopoly"].optimal_revenue)
        save_dataframe(case_curve, case_dir / "flexible_capacity_curve.csv")
        case_probe = flexible_local_probe(serial_config.model, case_solver, regenerated["monopoly"].optimal_revenue, probe_capacities)
        case_probe.insert(0, "case", case_name)
        probe_frames.append(case_probe)
        save_dataframe(case_probe, case_dir / "local_probes.csv")
        sensitivity_rows.append(compare_flexible_case_to_baseline(case_name, baseline_curve, case_curve))

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    save_dataframe(sensitivity_df, sensitivity_dir / "sensitivity_summary.csv")
    if probe_frames:
        save_dataframe(pd.concat(probe_frames, ignore_index=True), sensitivity_dir / "all_local_probes.csv")

    regression_pass = bool(
        regression_df[~regression_df["status"].eq("ok")].empty
        and summary_compare_df[~summary_compare_df["status"].eq("ok")].empty
    )

    monopoly_pass = bool(
        _max_abs(monopoly_diag["fixed_point_residual"]) <= 1e-8
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
        parallel_pass = bool(
            pr.get("capacity_grid_identical", False)
            and _safe_float(pr.get("best_capacity_shift_mbps")) <= 0.0 + 1e-12
            and _safe_float(pr.get("best_revenue_abs_diff")) <= 1e-10
            and _safe_float(pr.get("max_abs_diff_revenue_mno_total")) <= 1e-10
            and _safe_float(pr.get("max_abs_diff_nM")) <= 1e-10
            and _safe_float(pr.get("max_abs_diff_nV")) <= 1e-10
            and int(pr.get("valid_status_flip_count", 0)) == 0
        )

    material_shift_cases = []
    for _, row in sensitivity_df.iterrows():
        if str(row["case"]) == "parallel_baseline":
            continue
        best_capacity_shift = _safe_float(row.get("best_capacity_shift_mbps"))
        best_revenue_rel_diff = _safe_float(row.get("best_revenue_rel_diff"))
        if (
            np.isfinite(best_capacity_shift)
            and best_capacity_shift > float(serial_config.solver.capacity_step_mbps) + 1e-12
        ) or (
            np.isfinite(best_revenue_rel_diff)
            and best_revenue_rel_diff > 1e-3
        ):
            material_shift_cases.append(str(row["case"]))

    sensitivity_pass = len(material_shift_cases) == 0

    verdict = {
        "regression_pass": regression_pass,
        "monopoly_pass": monopoly_pass,
        "market_clearing_pass": market_pass,
        "flexible_pass": flexible_pass,
        "parallel_invariance_pass": parallel_pass,
        "sensitivity_pass": sensitivity_pass,
        "material_shift_cases": material_shift_cases,
        "overall_pass": bool(
            regression_pass
            and monopoly_pass
            and market_pass
            and flexible_pass
            and parallel_pass
            and sensitivity_pass
        ),
        "thresholds": {
            "numeric_compare_atol": _NUMERIC_COMPARE_ATOL,
            "numeric_compare_rtol": _NUMERIC_COMPARE_RTOL,
            "flexible_residual_tol": _RESIDUAL_TOL,
            "sensitivity_best_capacity_shift_tol_mbps": serial_config.solver.capacity_step_mbps,
            "sensitivity_best_revenue_rel_tol": 1e-3,
        },
    }
    save_json(verdict, outdir / "overall_verdict.json")

    print(json.dumps(verdict, indent=2, sort_keys=True))
    if args.fail_on_issues and not verdict["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
