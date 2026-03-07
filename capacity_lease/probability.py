from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import lambertw, ndtr, ndtri

try:  # pragma: no cover - availability depends on SciPy build
    from scipy.stats._qmvnt import _bvnu as _fast_bvnu
except Exception:  # pragma: no cover - fallback handled below
    _fast_bvnu = None

from scipy.stats import multivariate_normal



def clip_probability(value: float) -> float:
    if value <= 0.0:
        return 0.0 if value > -1e-15 else max(0.0, value)
    if value >= 1.0:
        return 1.0 if value < 1.0 + 1e-15 else min(1.0, value)
    return float(value)



def normal_cdf_standard(z: float | np.ndarray) -> float | np.ndarray:
    return ndtr(z)



def normal_sf_standard(z: float | np.ndarray) -> float | np.ndarray:
    return ndtr(-np.asarray(z))



def normal_cdf(x: float | np.ndarray, mean: float | np.ndarray, sd: float | np.ndarray) -> float | np.ndarray:
    return ndtr((np.asarray(x) - mean) / sd)



def normal_sf(x: float | np.ndarray, mean: float | np.ndarray, sd: float | np.ndarray) -> float | np.ndarray:
    return ndtr(-(np.asarray(x) - mean) / sd)



def normal_ppf(q: float | np.ndarray, mean: float | np.ndarray, sd: float | np.ndarray) -> float | np.ndarray:
    return mean + sd * ndtri(q)



def lambertw_real(x: float, branch: int) -> float:
    value = lambertw(x, branch)
    if abs(value.imag) > 1e-12:
        raise ValueError(f"Lambert W returned a complex value for x={x}, branch={branch}: {value}")
    return float(value.real)


@dataclass(frozen=True)
class BivariateUpperHelper:
    """Fast probability helper for P(X <= h, Y >= k) in standardized coordinates.

    If the private SciPy fast path is not available, it falls back to the public
    multivariate_normal CDF.
    """

    rho: float
    use_fast: bool = True
    _fallback: Optional[object] = None

    def __post_init__(self) -> None:
        use_fast = bool(self.use_fast and _fast_bvnu is not None)
        object.__setattr__(self, "use_fast", use_fast)
        if not use_fast:
            object.__setattr__(
                self,
                "_fallback",
                multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, self.rho], [self.rho, 1.0]]),
            )

    def prob_x_le_y_ge(self, zx: float, zy: float) -> float:
        if self.use_fast:
            value = float(normal_sf_standard(zy) - _fast_bvnu(float(zx), float(zy), float(self.rho)))
            return clip_probability(value)
        assert self._fallback is not None
        value = float(normal_cdf_standard(zx) - self._fallback.cdf([float(zx), float(zy)]))
        return clip_probability(value)
