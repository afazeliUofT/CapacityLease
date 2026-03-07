from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelParams:
    group_sizes: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    monopoly_noise_mean: np.ndarray
    monopoly_noise_sd: np.ndarray
    mno_noise_mean: np.ndarray
    mno_noise_sd: np.ndarray
    mvno_noise_mean: np.ndarray
    mvno_noise_sd: np.ndarray
    capacity_mbps: float
    delta: float
    zeta: float
    lambda_retention: float
    participation_fee: float

    def __post_init__(self) -> None:
        array_fields = [
            "group_sizes",
            "alpha",
            "beta",
            "monopoly_noise_mean",
            "monopoly_noise_sd",
            "mno_noise_mean",
            "mno_noise_sd",
            "mvno_noise_mean",
            "mvno_noise_sd",
        ]
        for name in array_fields:
            value = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, value)

        lengths = {len(getattr(self, name)) for name in array_fields}
        if len(lengths) != 1:
            raise ValueError(f"All group-level arrays must have the same length, got {lengths}.")
        if np.any(self.group_sizes <= 0):
            raise ValueError("All group sizes must be strictly positive.")
        if np.any(self.alpha <= 0) or np.any(self.beta <= 0):
            raise ValueError("All alpha and beta values must be strictly positive.")
        if np.any(self.monopoly_noise_sd <= 0) or np.any(self.mno_noise_sd <= 0) or np.any(self.mvno_noise_sd <= 0):
            raise ValueError("All Gaussian standard deviations must be strictly positive.")
        if self.capacity_mbps <= 0:
            raise ValueError("capacity_mbps must be positive.")
        if not (0 < self.delta <= 1):
            raise ValueError("delta must lie in (0, 1].")
        if not (0 < self.zeta < 1):
            raise ValueError("zeta must lie in (0, 1).")
        if not (0 <= self.lambda_retention < 1):
            raise ValueError("lambda_retention must lie in [0, 1).")
        if self.participation_fee < 0:
            raise ValueError("participation_fee must be nonnegative.")

    @property
    def G(self) -> int:
        return int(self.group_sizes.size)

    @property
    def N(self) -> float:
        return float(np.sum(self.group_sizes))


@dataclass(frozen=True)
class SolverConfig:
    root_xtol: float = 1e-10
    root_rtol: float = 1e-10
    max_root_iter: int = 200
    use_fast_bvn_if_available: bool = True
    market_clearing_quantile_tail: str = "lower"
    monopoly_price_points: int = 1501
    monopoly_plot_price_max: float = 150.0
    capacity_step_mbps: float = 5.0
    market_clearing_search_points: int = 161
    flexible_nM_search_points: int = 25
    flexible_nV_search_points: int = 25
    refine_levels: int = 2
    top_k_seeds: int = 2
    nM_curve_points: int = 401
    workers: int = 1
    chunksize: int = 1
    reported_market_clearing_capacity_mbps: float = 100.0
    reported_flexible_capacity_mbps: float = 350.0

    def __post_init__(self) -> None:
        if self.root_xtol <= 0 or self.root_rtol <= 0:
            raise ValueError("Root tolerances must be strictly positive.")
        if self.max_root_iter < 1:
            raise ValueError("max_root_iter must be at least 1.")
        if self.market_clearing_quantile_tail not in {"lower", "upper"}:
            raise ValueError("market_clearing_quantile_tail must be 'lower' or 'upper'.")
        positive_ints = [
            "monopoly_price_points",
            "market_clearing_search_points",
            "flexible_nM_search_points",
            "flexible_nV_search_points",
            "refine_levels",
            "top_k_seeds",
            "nM_curve_points",
            "workers",
            "chunksize",
        ]
        for name in positive_ints:
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1.")
        if self.capacity_step_mbps <= 0:
            raise ValueError("capacity_step_mbps must be strictly positive.")
        if self.monopoly_plot_price_max <= 0:
            raise ValueError("monopoly_plot_price_max must be strictly positive.")
        if self.reported_market_clearing_capacity_mbps < 0 or self.reported_flexible_capacity_mbps < 0:
            raise ValueError("Reported paper capacities must be nonnegative.")


@dataclass(frozen=True)
class Config:
    model: ModelParams
    solver: SolverConfig



def _get_section(raw: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = raw
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur



def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = json.loads(path.read_text())

    model = ModelParams(
        group_sizes=_get_section(raw, "model", "group_sizes"),
        alpha=_get_section(raw, "model", "alpha"),
        beta=_get_section(raw, "model", "beta"),
        monopoly_noise_mean=_get_section(raw, "model", "monopoly_noise_mean"),
        monopoly_noise_sd=_get_section(raw, "model", "monopoly_noise_sd"),
        mno_noise_mean=_get_section(raw, "model", "mno_noise_mean"),
        mno_noise_sd=_get_section(raw, "model", "mno_noise_sd"),
        mvno_noise_mean=_get_section(raw, "model", "mvno_noise_mean"),
        mvno_noise_sd=_get_section(raw, "model", "mvno_noise_sd"),
        capacity_mbps=float(_get_section(raw, "model", "capacity_mbps")),
        delta=float(_get_section(raw, "model", "delta")),
        zeta=float(_get_section(raw, "model", "zeta")),
        lambda_retention=float(_get_section(raw, "model", "lambda_retention")),
        participation_fee=float(_get_section(raw, "model", "participation_fee")),
    )

    solver_raw = raw.get("solver", {})
    solver = SolverConfig(
        root_xtol=float(solver_raw.get("root_xtol", SolverConfig.root_xtol)),
        root_rtol=float(solver_raw.get("root_rtol", SolverConfig.root_rtol)),
        max_root_iter=int(solver_raw.get("max_root_iter", SolverConfig.max_root_iter)),
        use_fast_bvn_if_available=bool(
            solver_raw.get("use_fast_bvn_if_available", SolverConfig.use_fast_bvn_if_available)
        ),
        market_clearing_quantile_tail=str(
            solver_raw.get("market_clearing_quantile_tail", SolverConfig.market_clearing_quantile_tail)
        ),
        monopoly_price_points=int(solver_raw.get("monopoly_price_points", SolverConfig.monopoly_price_points)),
        monopoly_plot_price_max=float(
            solver_raw.get("monopoly_plot_price_max", SolverConfig.monopoly_plot_price_max)
        ),
        capacity_step_mbps=float(solver_raw.get("capacity_step_mbps", SolverConfig.capacity_step_mbps)),
        market_clearing_search_points=int(
            solver_raw.get("market_clearing_search_points", SolverConfig.market_clearing_search_points)
        ),
        flexible_nM_search_points=int(
            solver_raw.get("flexible_nM_search_points", SolverConfig.flexible_nM_search_points)
        ),
        flexible_nV_search_points=int(
            solver_raw.get("flexible_nV_search_points", SolverConfig.flexible_nV_search_points)
        ),
        refine_levels=int(solver_raw.get("refine_levels", SolverConfig.refine_levels)),
        top_k_seeds=int(solver_raw.get("top_k_seeds", SolverConfig.top_k_seeds)),
        nM_curve_points=int(solver_raw.get("nM_curve_points", SolverConfig.nM_curve_points)),
        workers=int(solver_raw.get("workers", SolverConfig.workers)),
        chunksize=int(solver_raw.get("chunksize", SolverConfig.chunksize)),
        reported_market_clearing_capacity_mbps=float(
            solver_raw.get(
                "reported_market_clearing_capacity_mbps",
                SolverConfig.reported_market_clearing_capacity_mbps,
            )
        ),
        reported_flexible_capacity_mbps=float(
            solver_raw.get(
                "reported_flexible_capacity_mbps",
                SolverConfig.reported_flexible_capacity_mbps,
            )
        ),
    )
    return Config(model=model, solver=solver)
