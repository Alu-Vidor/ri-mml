from __future__ import annotations

import mpmath as mp
import numpy as np
from scipy import special

try:
    from pymittagleffler import mittag_leffler as _pymittagleffler
except Exception:
    _pymittagleffler = None


def _inverse_gamma(value: float) -> float:
    gamma_value = special.gamma(value)
    if not np.isfinite(gamma_value):
        return 0.0
    return 1.0 / float(gamma_value)


def _series_scalar(
    alpha: float,
    beta: float,
    z: complex,
    tol: float,
    max_terms: int,
) -> complex:
    term = complex(_inverse_gamma(beta))
    result = term
    if abs(term) <= tol:
        return result

    for n in range(max_terms - 1):
        numerator = special.gammaln(alpha * n + beta)
        denominator = special.gammaln(alpha * (n + 1) + beta)
        ratio = z * np.exp(numerator - denominator)
        term *= ratio
        result += term
        if abs(term) <= tol * max(1.0, abs(result)):
            return result
    raise RuntimeError("Power series for the Mittag-Leffler function did not converge")


def _asymptotic_scalar(
    alpha: float,
    beta: float,
    z: complex,
    tol: float,
    max_terms: int,
) -> complex:
    result = 0.0j
    for k in range(1, max_terms + 1):
        term = -(z ** (-k)) * _inverse_gamma(beta - alpha * k)
        result += term
        if k >= 4 and abs(term) <= tol * max(1.0, abs(result)):
            return result
    return result


def _mpmath_fallback(alpha: float, beta: float, z: complex, tol: float) -> complex:
    mp.mp.dps = max(50, int(-np.log10(tol)) + 20)
    z_mp = mp.mpc(z.real, z.imag)
    result = mp.nsum(
        lambda n: z_mp**n / mp.gamma(alpha * n + beta),
        [0, mp.inf],
    )
    return complex(result)


def _mittag_leffler_scalar(
    alpha: float,
    beta: float,
    z: complex,
    tol: float,
    max_terms: int,
    asymptotic_cutoff: float,
) -> complex:
    if z == 0:
        return complex(_inverse_gamma(beta))

    if z.imag == 0.0 and z.real < 0.0 and abs(z) >= asymptotic_cutoff:
        result = _asymptotic_scalar(alpha, beta, z, tol, min(max_terms, 64))
        if np.isfinite(result.real) and np.isfinite(result.imag):
            return result

    try:
        return _series_scalar(alpha, beta, z, tol, max_terms)
    except RuntimeError:
        return _mpmath_fallback(alpha, beta, z, tol)


def mittag_leffler(
    alpha: float,
    beta: float,
    z: float | complex | np.ndarray,
    *,
    tol: float = 1e-13,
    max_terms: int = 2000,
    asymptotic_cutoff: float = 5.0,
) -> np.ndarray | float | complex:
    """Evaluate E_{alpha,beta}(z) for scalar or array-like z."""

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")

    if _pymittagleffler is not None:
        try:
            result = _pymittagleffler(z, alpha=alpha, beta=beta)
            if np.isscalar(z):
                if np.isrealobj(z):
                    return float(np.real(result))
                return complex(result)
            result_array = np.asarray(result)
            if np.isrealobj(z):
                return np.asarray(np.real(result_array), dtype=float)
            return result_array
        except Exception:
            pass

    array = np.asarray(z, dtype=complex)
    flat = array.reshape(-1)
    values = np.empty_like(flat, dtype=complex)
    for index, value in enumerate(flat):
        values[index] = _mittag_leffler_scalar(
            alpha=alpha,
            beta=beta,
            z=complex(value),
            tol=tol,
            max_terms=max_terms,
            asymptotic_cutoff=asymptotic_cutoff,
        )

    result = values.reshape(array.shape)
    if np.isrealobj(z):
        result = result.real
    if np.isscalar(z):
        return result.reshape(()).item()
    return result
