from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ModelParams, SolverConfig
from .numerics import grid_refine_maximize_scalar, safe_brentq
from .probability import normal_cdf, normal_sf


@dataclass(frozen=True)
class MonopolyResult:
    critical_price: float
    optimal_price: float
    optimal_subscribers: float
    optimal_rate: float
    optimal_revenue: float
    curve: pd.DataFrame



def critical_price(model: ModelParams, solver: SolverConfig) -> float:
    target = model.N - 1.0 / model.delta

    def func(price: float) -> float:
        threshold = model.beta * price - model.alpha * np.log(model.capacity_mbps)
        cdfs = normal_cdf(threshold, model.monopoly_noise_mean, model.monopoly_noise_sd)
        return float(np.dot(model.group_sizes, cdfs) - target)

    lo, hi = -1.0, 1.0
    flo, fhi = func(lo), func(hi)
    while flo > 0:
        lo *= 2.0
        flo = func(lo)
    while fhi < 0:
        hi *= 2.0
        fhi = func(hi)
    return safe_brentq(func, lo, hi, xtol=solver.root_xtol, rtol=solver.root_rtol, maxiter=solver.max_root_iter)



def subscribers_at_price(model: ModelParams, solver: SolverConfig, price: float, price_critical: float | None = None) -> float | None:
    pc = critical_price(model, solver) if price_critical is None else price_critical
    if price < 0 or price > pc + 1e-12:
        return None

    lo = 1.0 / model.delta
    hi = model.N
    const = model.beta * price - model.alpha * np.log(model.capacity_mbps / model.delta)

    def func(subscribers: float) -> float:
        threshold = const + model.alpha * np.log(subscribers)
        cdfs = normal_cdf(threshold, model.monopoly_noise_mean, model.monopoly_noise_sd)
        return float(model.N - np.dot(model.group_sizes, cdfs) - subscribers)

    return safe_brentq(func, lo, hi, xtol=solver.root_xtol, rtol=solver.root_rtol, maxiter=solver.max_root_iter)



def monopoly_state(model: ModelParams, solver: SolverConfig, price: float, price_critical: float | None = None) -> dict[str, Any]:
    pc = critical_price(model, solver) if price_critical is None else price_critical
    subscribers = subscribers_at_price(model, solver, price, price_critical=pc)
    if subscribers is None:
        return {
            "price": float(price),
            "feasible": False,
            "subscribers": 0.0,
            "rate_mbps": np.nan,
            "revenue": 0.0,
            **{f"A_group_{g+1}": np.nan for g in range(model.G)},
        }
    rate = model.capacity_mbps / (model.delta * subscribers)
    threshold = model.beta * price - model.alpha * np.log(rate)
    acceptance = normal_sf(threshold, model.monopoly_noise_mean, model.monopoly_noise_sd)
    state = {
        "price": float(price),
        "feasible": True,
        "subscribers": float(subscribers),
        "rate_mbps": float(rate),
        "revenue": float(price * subscribers),
    }
    for g in range(model.G):
        state[f"A_group_{g+1}"] = float(acceptance[g])
    return state



def solve_monopoly_problem(model: ModelParams, solver: SolverConfig) -> MonopolyResult:
    pc = critical_price(model, solver)

    def objective(price: float) -> tuple[float, dict[str, Any]]:
        state = monopoly_state(model, solver, price, price_critical=pc)
        return float(state["revenue"]), state

    optimum = grid_refine_maximize_scalar(
        objective,
        0.0,
        pc,
        points=max(solver.monopoly_price_points // 8, 101),
        refine_levels=2,
        top_k=4,
    )
    if optimum is None:
        raise RuntimeError("Failed to optimize the monopoly problem.")
    curve_prices = np.linspace(0.0, max(pc, solver.monopoly_plot_price_max), solver.monopoly_price_points)
    curve = pd.DataFrame([monopoly_state(model, solver, float(p), price_critical=pc) for p in curve_prices])
    best_state = optimum.payload
    return MonopolyResult(
        critical_price=float(pc),
        optimal_price=float(best_state["price"]),
        optimal_subscribers=float(best_state["subscribers"]),
        optimal_rate=float(best_state["rate_mbps"]),
        optimal_revenue=float(best_state["revenue"]),
        curve=curve,
    )
