from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ModelParams, SolverConfig
from .numerics import grid_refine_maximize_scalar, safe_brentq
from .probability import lambertw_real, normal_cdf, normal_ppf
from .utils import parallel_map


@dataclass(frozen=True)
class MarketClearingState:
    capacity_leased_mbps: float
    nV: float
    nM: float
    pV: float
    pM: float
    rV_mbps: float
    rM_mbps: float
    revenue_mno_total: float
    revenue_mvno_gross: float
    valid: bool



def _quantile_level(model: ModelParams, solver: SolverConfig) -> float:
    return model.zeta if solver.market_clearing_quantile_tail == "lower" else 1.0 - model.zeta



def _kappa_tau(model: ModelParams, solver: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    q = _quantile_level(model, solver)
    kappa = normal_ppf(q, model.mvno_noise_mean, model.mvno_noise_sd) / model.beta
    tau = model.alpha / model.beta
    return np.asarray(kappa, dtype=float), np.asarray(tau, dtype=float)



def profitability_interval(model: ModelParams, solver: SolverConfig, capacity_leased_mbps: float) -> tuple[float, float] | None:
    CV = float(capacity_leased_mbps)
    if CV <= 0 or CV >= model.capacity_mbps:
        return None
    kappa, tau = _kappa_tau(model, solver)
    TV = float(np.min(kappa / tau))
    lo = 1.0 / model.delta
    hi = min(model.N - 1.0 / model.delta, CV / model.delta * np.exp(TV))
    if TV < -np.log(CV) or hi < lo:
        return None

    target = model.participation_fee / (1.0 - model.lambda_retention)
    for g in range(model.G):
        A = CV / model.delta * np.exp(kappa[g] / tau[g])
        n_peak = A / np.e
        f_max = n_peak * tau[g]
        if target > f_max + 1e-12:
            return None
        if abs(target - f_max) <= 1e-12:
            lo = max(lo, n_peak)
            hi = min(hi, n_peak)
        else:
            B = target / tau[g]
            z = -B / A
            if not (-1.0 / np.e <= z < 0.0):
                return None
            n_small = -B / lambertw_real(z, -1)
            n_large = -B / lambertw_real(z, 0)
            lo = max(lo, n_small)
            hi = min(hi, n_large)
        if lo > hi:
            return None
    return float(lo), float(hi)



def mvno_price(model: ModelParams, solver: SolverConfig, capacity_leased_mbps: float, nV: float) -> float:
    kappa, tau = _kappa_tau(model, solver)
    return float(np.min(kappa + tau * np.log(capacity_leased_mbps / (model.delta * nV))))



def mno_price(model: ModelParams, solver: SolverConfig, capacity_leased_mbps: float, nV: float) -> float | None:
    CV = float(capacity_leased_mbps)
    if CV <= 0 or CV >= model.capacity_mbps:
        return None
    nM = model.N - nV
    if nM <= 0:
        return None
    pV = mvno_price(model, solver, CV, nV)
    mu_diff = model.mno_noise_mean - model.mvno_noise_mean
    sd_diff = np.sqrt(model.mno_noise_sd**2 + model.mvno_noise_sd**2)
    const = model.alpha * np.log((nM * CV) / ((model.capacity_mbps - CV) * nV)) - model.beta * pV

    def func(price: float) -> float:
        cdfs = normal_cdf(model.beta * price + const, mu_diff, sd_diff)
        return float(np.dot(model.group_sizes, cdfs) - nV)

    if func(0.0) > 0.0:
        return None
    hi = 1.0
    while func(hi) < 0.0 and hi < 1e5:
        hi *= 2.0
    if func(hi) < 0.0:
        return None
    return safe_brentq(func, 0.0, hi, xtol=solver.root_xtol, rtol=solver.root_rtol, maxiter=solver.max_root_iter)



def market_clearing_state(model: ModelParams, solver: SolverConfig, capacity_leased_mbps: float, nV: float) -> MarketClearingState:
    interval = profitability_interval(model, solver, capacity_leased_mbps)
    if interval is None or not (interval[0] - 1e-9 <= nV <= interval[1] + 1e-9):
        return MarketClearingState(capacity_leased_mbps, nV, model.N - nV, np.nan, np.nan, np.nan, np.nan, -np.inf, np.nan, False)
    pV = mvno_price(model, solver, capacity_leased_mbps, nV)
    if pV < 0.0:
        return MarketClearingState(capacity_leased_mbps, nV, model.N - nV, pV, np.nan, np.nan, np.nan, -np.inf, np.nan, False)
    pM = mno_price(model, solver, capacity_leased_mbps, nV)
    if pM is None or pM < 0.0:
        return MarketClearingState(capacity_leased_mbps, nV, model.N - nV, pV, np.nan, np.nan, np.nan, -np.inf, np.nan, False)
    nM = model.N - nV
    rV = capacity_leased_mbps / (model.delta * nV)
    rM = (model.capacity_mbps - capacity_leased_mbps) / (model.delta * nM)
    revenue_v = nV * pV
    revenue_m_total = nM * pM + (1.0 - model.lambda_retention) * revenue_v - model.participation_fee
    return MarketClearingState(capacity_leased_mbps, nV, nM, pV, pM, rV, rM, revenue_m_total, revenue_v, True)



def optimize_market_clearing_for_capacity(model: ModelParams, solver: SolverConfig, capacity_leased_mbps: float) -> dict[str, Any]:
    interval = profitability_interval(model, solver, capacity_leased_mbps)
    if interval is None:
        return {
            "capacity_leased_mbps": float(capacity_leased_mbps),
            "valid": False,
            "revenue_mno_total": np.nan,
            "revenue_mvno_gross": np.nan,
            "nV": np.nan,
            "nM": np.nan,
            "pV": np.nan,
            "pM": np.nan,
            "rV_mbps": np.nan,
            "rM_mbps": np.nan,
            "nV_lo": np.nan,
            "nV_hi": np.nan,
        }

    def objective(nV: float) -> tuple[float, MarketClearingState]:
        state = market_clearing_state(model, solver, capacity_leased_mbps, nV)
        return float(state.revenue_mno_total), state

    optimum = grid_refine_maximize_scalar(
        objective,
        interval[0],
        interval[1],
        points=solver.market_clearing_search_points,
        refine_levels=solver.refine_levels,
        top_k=solver.top_k_seeds,
    )
    if optimum is None or optimum.payload is None:
        return {
            "capacity_leased_mbps": float(capacity_leased_mbps),
            "valid": False,
            "revenue_mno_total": np.nan,
            "revenue_mvno_gross": np.nan,
            "nV": np.nan,
            "nM": np.nan,
            "pV": np.nan,
            "pM": np.nan,
            "rV_mbps": np.nan,
            "rM_mbps": np.nan,
            "nV_lo": float(interval[0]),
            "nV_hi": float(interval[1]),
        }
    state = optimum.payload
    return {
        "capacity_leased_mbps": float(capacity_leased_mbps),
        "valid": bool(state.valid),
        "revenue_mno_total": float(state.revenue_mno_total),
        "revenue_mvno_gross": float(state.revenue_mvno_gross),
        "nV": float(state.nV),
        "nM": float(state.nM),
        "pV": float(state.pV),
        "pM": float(state.pM),
        "rV_mbps": float(state.rV_mbps),
        "rM_mbps": float(state.rM_mbps),
        "nV_lo": float(interval[0]),
        "nV_hi": float(interval[1]),
    }



def _market_clearing_task(payload: tuple[ModelParams, SolverConfig, float]) -> dict[str, Any]:
    model, solver, CV = payload
    return optimize_market_clearing_for_capacity(model, solver, CV)



def solve_market_clearing_capacity_curve(model: ModelParams, solver: SolverConfig) -> pd.DataFrame:
    capacities = np.arange(solver.capacity_step_mbps, model.capacity_mbps + 1e-12, solver.capacity_step_mbps)
    payloads = [(model, solver, float(CV)) for CV in capacities]
    rows = parallel_map(_market_clearing_task, payloads, workers=solver.workers, chunksize=solver.chunksize)
    df = pd.DataFrame(rows).sort_values("capacity_leased_mbps").reset_index(drop=True)
    return df



def solve_market_clearing_nm_curve(model: ModelParams, solver: SolverConfig, capacity_leased_mbps: float, nM_points: int | None = None) -> pd.DataFrame:
    interval = profitability_interval(model, solver, capacity_leased_mbps)
    if interval is None:
        return pd.DataFrame(
            columns=[
                "nM", "nV", "capacity_leased_mbps", "valid", "revenue_mno_total", "revenue_mvno_gross", "pM", "pV", "rM_mbps", "rV_mbps"
            ]
        )
    points = solver.nM_curve_points if nM_points is None else int(nM_points)
    nM_lo = model.N - interval[1]
    nM_hi = model.N - interval[0]
    nM_grid = np.linspace(nM_lo, nM_hi, points)
    rows: list[dict[str, Any]] = []
    for nM in nM_grid:
        nV = model.N - nM
        state = market_clearing_state(model, solver, capacity_leased_mbps, nV)
        rows.append(
            {
                "nM": float(nM),
                "nV": float(nV),
                "capacity_leased_mbps": float(capacity_leased_mbps),
                "valid": bool(state.valid),
                "revenue_mno_total": float(state.revenue_mno_total),
                "revenue_mvno_gross": float(state.revenue_mvno_gross),
                "pM": float(state.pM),
                "pV": float(state.pV),
                "rM_mbps": float(state.rM_mbps),
                "rV_mbps": float(state.rV_mbps),
            }
        )
    return pd.DataFrame(rows)
