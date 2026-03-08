from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import math
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .config import ModelParams, SolverConfig
from .numerics import grid_refine_maximize_scalar, safe_brentq
from .probability import BivariateUpperHelper
from .utils import parallel_map


@dataclass(frozen=True)
class FlexibleState:
    capacity_leased_mbps: float
    nM: float
    nV: float
    pM: float
    pV: float
    rM_mbps: float
    rV_mbps: float
    revenue_mno_total: float
    revenue_mvno_gross: float
    revenue_mvno_retained: float
    valid: bool


class CompetitiveKernel:
    def __init__(self, model: ModelParams, solver: SolverConfig) -> None:
        self.model = model
        self.solver = solver
        self.sig_x = np.sqrt(model.mno_noise_sd**2 + model.mvno_noise_sd**2)
        self.rho_v = -model.mvno_noise_sd / self.sig_x
        self.rho_m = -model.mno_noise_sd / self.sig_x
        self.mu_x_v = model.mno_noise_mean - model.mvno_noise_mean
        self.mu_x_m = model.mvno_noise_mean - model.mno_noise_mean
        self.helper_v = [BivariateUpperHelper(float(rho), use_fast=solver.use_fast_bvn_if_available) for rho in self.rho_v]
        self.helper_m = [BivariateUpperHelper(float(rho), use_fast=solver.use_fast_bvn_if_available) for rho in self.rho_m]

    def mvno_acceptance(self, g: int, pM: float, pV: float, CV: float, nM: float, nV: float) -> float:
        a = self.model.beta[g] * (pM - pV) + self.model.alpha[g] * math.log((nM * CV) / (nV * (self.model.capacity_mbps - CV)))
        b = self.model.beta[g] * pV - self.model.alpha[g] * math.log(CV / (self.model.delta * nV))
        zx = (a - self.mu_x_v[g]) / self.sig_x[g]
        zy = (b - self.model.mvno_noise_mean[g]) / self.model.mvno_noise_sd[g]
        return self.helper_v[g].prob_x_le_y_ge(float(zx), float(zy))

    def mno_acceptance(self, g: int, pM: float, pV: float, CV: float, nM: float, nV: float) -> float:
        a = self.model.beta[g] * (pV - pM) + self.model.alpha[g] * math.log((nV * (self.model.capacity_mbps - CV)) / (nM * CV))
        b = self.model.beta[g] * pM - self.model.alpha[g] * math.log((self.model.capacity_mbps - CV) / (self.model.delta * nM))
        zx = (a - self.mu_x_m[g]) / self.sig_x[g]
        zy = (b - self.model.mno_noise_mean[g]) / self.model.mno_noise_sd[g]
        return self.helper_m[g].prob_x_le_y_ge(float(zx), float(zy))

    def theta_m(self, pM: float, pV: float, CV: float, nM: float, nV: float) -> float:
        return float(sum(self.model.group_sizes[g] * self.mno_acceptance(g, pM, pV, CV, nM, nV) for g in range(self.model.G)))

    def theta_v(self, pM: float, pV: float, CV: float, nM: float, nV: float) -> float:
        return float(sum(self.model.group_sizes[g] * self.mvno_acceptance(g, pM, pV, CV, nM, nV) for g in range(self.model.G)))


class FlexibleSolverLocal:
    def __init__(self, model: ModelParams, solver: SolverConfig, monopoly_revenue: float) -> None:
        self.model = model
        self.solver = solver
        self.monopoly_revenue = float(monopoly_revenue)
        self.kernel = CompetitiveKernel(model, solver)
        self._price_cache: dict[tuple[float, float, float], tuple[float, float] | None] = {}
        self._best_response_cache: dict[tuple[float, float], dict[str, Any] | None] = {}

    @staticmethod
    def _round_key(*values: float) -> tuple[float, ...]:
        return tuple(round(float(v), 10) for v in values)

    def _solve_pM_given_pV(self, CV: float, nM: float, nV: float, pV: float) -> float | None:
        def func(pM: float) -> float:
            return self.kernel.theta_m(pM, pV, CV, nM, nV) - nM

        if func(0.0) < 0.0:
            return None
        hi = 1.0
        while func(hi) > 0.0 and hi < 1e5:
            hi *= 2.0
        if func(hi) > 0.0:
            return None
        return safe_brentq(
            func,
            0.0,
            hi,
            xtol=self.solver.root_xtol,
            rtol=self.solver.root_rtol,
            maxiter=self.solver.max_root_iter,
        )

    def _solve_prices_nested(self, CV: float, nM: float, nV: float) -> tuple[float, float] | None:
        key = self._round_key(CV, nM, nV)
        if key in self._price_cache:
            return self._price_cache[key]

        if not (0.0 < CV < self.model.capacity_mbps):
            self._price_cache[key] = None
            return None
        if not (1.0 / self.model.delta <= nM <= self.model.N - 1.0 / self.model.delta + 1e-9):
            self._price_cache[key] = None
            return None
        if not (1.0 / self.model.delta <= nV <= self.model.N - nM + 1e-9):
            self._price_cache[key] = None
            return None

        def h(pV: float) -> float:
            return self.kernel.theta_m(0.0, pV, CV, nM, nV) - nM

        if h(0.0) >= 0.0:
            pV_lo = 0.0
        else:
            hi = 1.0
            while h(hi) < 0.0 and hi < 1e5:
                hi *= 2.0
            if h(hi) < 0.0:
                self._price_cache[key] = None
                return None
            pV_lo = safe_brentq(
                h,
                0.0,
                hi,
                xtol=self.solver.root_xtol,
                rtol=self.solver.root_rtol,
                maxiter=self.solver.max_root_iter,
            )

        def g(pV: float) -> float:
            pM = self._solve_pM_given_pV(CV, nM, nV, pV)
            if pM is None:
                return np.nan
            return self.kernel.theta_v(pM, pV, CV, nM, nV) - nV

        g_lo = g(pV_lo)
        if np.isnan(g_lo) or g_lo < 0.0:
            self._price_cache[key] = None
            return None
        hi = max(1.0, pV_lo + 1.0)
        while True:
            g_hi = g(hi)
            if not np.isnan(g_hi) and g_hi <= 0.0:
                break
            hi *= 2.0
            if hi >= 1e5:
                self._price_cache[key] = None
                return None
        pV = safe_brentq(
            g,
            pV_lo,
            hi,
            xtol=self.solver.root_xtol,
            rtol=self.solver.root_rtol,
            maxiter=self.solver.max_root_iter,
        )
        pM = self._solve_pM_given_pV(CV, nM, nV, pV)
        result = None if pM is None else (float(pM), float(pV))
        self._price_cache[key] = result
        return result

    def _solve_prices_fallback(self, CV: float, nM: float, nV: float) -> tuple[float, float] | None:
        seeds = np.array(
            [
                [1.0, 1.0],
                [5.0, 1.0],
                [1.0, 5.0],
                [10.0, 10.0],
                [20.0, 5.0],
                [50.0, 10.0],
            ],
            dtype=float,
        )
        best: tuple[float, float] | None = None
        best_norm = math.inf

        def residual(x: np.ndarray) -> np.ndarray:
            pM, pV = float(x[0]), float(x[1])
            return np.array(
                [
                    self.kernel.theta_m(pM, pV, CV, nM, nV) - nM,
                    self.kernel.theta_v(pM, pV, CV, nM, nV) - nV,
                ],
                dtype=float,
            )

        for seed in seeds:
            try:
                res = least_squares(
                    residual,
                    seed,
                    bounds=(0.0, np.inf),
                    xtol=self.solver.root_xtol,
                    ftol=self.solver.root_xtol,
                    gtol=self.solver.root_xtol,
                    max_nfev=max(200, self.solver.max_root_iter * 2),
                )
            except Exception:
                continue
            norm2 = float(np.linalg.norm(res.fun, ord=2))
            if res.success and norm2 < best_norm and norm2 <= 1e-5:
                best_norm = norm2
                best = (float(res.x[0]), float(res.x[1]))
        return best

    def solve_prices(self, CV: float, nM: float, nV: float) -> tuple[float, float] | None:
        """Solve the flexible-participation price system.

        The nested monotone solver is tried first because it is much faster for
        the paper's parameter ranges than the general 2-D least-squares fallback.
        Unlike the original version, this method now validates the returned price
        pair by re-evaluating both fixed-point equations and only accepts a
        candidate when the residual norm is numerically tight.
        """

        def residual_norm(pair: tuple[float, float] | None) -> float:
            if pair is None:
                return math.inf
            pM, pV = float(pair[0]), float(pair[1])
            if not np.isfinite(pM) or not np.isfinite(pV):
                return math.inf
            if pM < 0.0 or pV < 0.0:
                return math.inf
            residual = np.array(
                [
                    self.kernel.theta_m(pM, pV, CV, nM, nV) - nM,
                    self.kernel.theta_v(pM, pV, CV, nM, nV) - nV,
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(residual)):
                return math.inf
            return float(np.linalg.norm(residual, ord=2))

        tolerance = max(1e-5, 100.0 * self.solver.root_xtol)

        nested = self._solve_prices_nested(CV, nM, nV)
        nested_norm = residual_norm(nested)
        if nested is not None and nested_norm <= tolerance:
            return float(nested[0]), float(nested[1])

        fallback = self._solve_prices_fallback(CV, nM, nV)
        fallback_norm = residual_norm(fallback)

        best_pair: tuple[float, float] | None = None
        best_norm = math.inf
        for pair, norm in ((nested, nested_norm), (fallback, fallback_norm)):
            if pair is None:
                continue
            if norm < best_norm:
                best_pair = pair
                best_norm = norm

        if best_pair is None or not np.isfinite(best_norm) or best_norm > tolerance:
            return None

        return float(best_pair[0]), float(best_pair[1])

    def candidate_state(self, CV: float, nM: float, nV: float) -> FlexibleState:
        prices = self.solve_prices(CV, nM, nV)
        if prices is None:
            return FlexibleState(CV, nM, nV, np.nan, np.nan, np.nan, np.nan, -np.inf, -np.inf, -np.inf, False)
        pM, pV = prices
        if pM < 0.0 or pV < 0.0:
            return FlexibleState(CV, nM, nV, pM, pV, np.nan, np.nan, -np.inf, -np.inf, -np.inf, False)
        rM = (self.model.capacity_mbps - CV) / (self.model.delta * nM)
        rV = CV / (self.model.delta * nV)
        revenue_v_gross = nV * pV
        revenue_v_retained = self.model.lambda_retention * revenue_v_gross
        revenue_m_total = nM * pM + (1.0 - self.model.lambda_retention) * revenue_v_gross - self.model.participation_fee
        valid = bool(revenue_m_total >= self.monopoly_revenue)
        return FlexibleState(CV, nM, nV, pM, pV, rM, rV, revenue_m_total if valid else -np.inf, revenue_v_gross, revenue_v_retained, valid)

    def mvno_best_response(self, CV: float, nM: float) -> dict[str, Any] | None:
        key = self._round_key(CV, nM)
        if key in self._best_response_cache:
            return self._best_response_cache[key]
        if not (0.0 < CV < self.model.capacity_mbps):
            self._best_response_cache[key] = None
            return None
        lo = 1.0 / self.model.delta
        hi = self.model.N - nM
        if hi < lo:
            self._best_response_cache[key] = None
            return None

        cache: dict[float, FlexibleState] = {}

        def objective(nV: float) -> tuple[float, FlexibleState]:
            nV = float(nV)
            state = cache.get(nV)
            if state is None:
                state = self.candidate_state(CV, nM, nV)
                cache[nV] = state
            value = state.revenue_mvno_gross if state.valid else -np.inf
            return float(value), state

        optimum = grid_refine_maximize_scalar(
            objective,
            lo,
            hi,
            points=self.solver.flexible_nV_search_points,
            refine_levels=self.solver.refine_levels,
            top_k=self.solver.top_k_seeds,
        )
        if optimum is None or optimum.payload is None or not optimum.payload.valid:
            self._best_response_cache[key] = None
            return None
        state: FlexibleState = optimum.payload
        result = {
            "capacity_leased_mbps": float(CV),
            "nM": float(state.nM),
            "nV": float(state.nV),
            "pM": float(state.pM),
            "pV": float(state.pV),
            "rM_mbps": float(state.rM_mbps),
            "rV_mbps": float(state.rV_mbps),
            "revenue_mno_total": float(state.revenue_mno_total),
            "revenue_mvno_gross": float(state.revenue_mvno_gross),
            "revenue_mvno_retained": float(state.revenue_mvno_retained),
            "valid": True,
        }
        self._best_response_cache[key] = result
        return result

    def mno_best_response_for_capacity(self, CV: float) -> dict[str, Any]:
        lo = 1.0 / self.model.delta
        hi = self.model.N - 1.0 / self.model.delta
        response_cache: dict[float, dict[str, Any] | None] = {}

        def objective(nM: float) -> tuple[float, dict[str, Any] | None]:
            nM = float(nM)
            result = response_cache.get(nM)
            if result is None:
                result = self.mvno_best_response(CV, nM)
                response_cache[nM] = result
            if result is None or not result.get("valid", False):
                return -np.inf, result
            return float(result["revenue_mno_total"]), result

        optimum = grid_refine_maximize_scalar(
            objective,
            lo,
            hi,
            points=self.solver.flexible_nM_search_points,
            refine_levels=self.solver.refine_levels,
            top_k=self.solver.top_k_seeds,
        )
        if optimum is None or optimum.payload is None:
            return {
                "capacity_leased_mbps": float(CV),
                "valid": False,
                "revenue_mno_total": np.nan,
                "revenue_mvno_gross": np.nan,
                "nM": np.nan,
                "nV": np.nan,
                "pM": np.nan,
                "pV": np.nan,
                "rM_mbps": np.nan,
                "rV_mbps": np.nan,
            }
        out = dict(optimum.payload)
        out["capacity_leased_mbps"] = float(CV)
        return out



def _flex_capacity_task(payload: tuple[ModelParams, SolverConfig, float, float]) -> dict[str, Any]:
    model, solver, monopoly_revenue, CV = payload
    local = FlexibleSolverLocal(model, solver, monopoly_revenue)
    return local.mno_best_response_for_capacity(CV)



def solve_flexible_capacity_curve(model: ModelParams, solver: SolverConfig, monopoly_revenue: float) -> pd.DataFrame:
    capacities = np.arange(solver.capacity_step_mbps, model.capacity_mbps + 1e-12, solver.capacity_step_mbps)
    payloads = [(model, solver, float(monopoly_revenue), float(CV)) for CV in capacities]
    rows = parallel_map(_flex_capacity_task, payloads, workers=solver.workers, chunksize=solver.chunksize)
    return pd.DataFrame(rows).sort_values("capacity_leased_mbps").reset_index(drop=True)



def _fixed_nm_task(payload: tuple[ModelParams, SolverConfig, float, float, float]) -> dict[str, Any]:
    model, solver, monopoly_revenue, CV, nM = payload
    local = FlexibleSolverLocal(model, solver, monopoly_revenue)
    response = local.mvno_best_response(CV, nM)
    if response is None:
        return {
            "capacity_leased_mbps": float(CV),
            "nM": float(nM),
            "nV": np.nan,
            "pM": np.nan,
            "pV": np.nan,
            "rM_mbps": np.nan,
            "rV_mbps": np.nan,
            "revenue_mno_total": np.nan,
            "revenue_mvno_gross": np.nan,
            "revenue_mvno_retained": np.nan,
            "valid": False,
        }
    return response



def solve_flexible_nm_curve(model: ModelParams, solver: SolverConfig, monopoly_revenue: float, capacity_leased_mbps: float, nM_points: int | None = None) -> pd.DataFrame:
    points = solver.nM_curve_points if nM_points is None else int(nM_points)
    nM_grid = np.linspace(1.0 / model.delta, model.N - 1.0 / model.delta, points)
    payloads = [
        (model, solver, float(monopoly_revenue), float(capacity_leased_mbps), float(nM))
        for nM in nM_grid
    ]
    rows = parallel_map(_fixed_nm_task, payloads, workers=solver.workers, chunksize=solver.chunksize)
    return pd.DataFrame(rows).sort_values("nM").reset_index(drop=True)
