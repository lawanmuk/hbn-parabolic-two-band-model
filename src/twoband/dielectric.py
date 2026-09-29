"""Imaginary part of the dielectric function for parabolic two-band models."""

import numpy as np


def im_eps_2d(
    omega: np.ndarray,
    e_gap: float,
    m_r: float,
    p_vc: float,
    eta: float = 0.05 / 27.21138386,
    k_max: float = 2.0,
    n_k: int = 4000,
) -> np.ndarray:
    """Im[eps(omega)] for a 2D parabolic two-band model.

    The energy-conserving delta function is replaced by a Lorentzian of width ``eta``
    and the k integral is done on a radial grid. For eta -> 0 the result approaches
    the analytic step (2 pi / omega)^2 * 2/(2 pi)^2 * |p_vc|^2 * 2 pi m_r above the gap.
    """
    omega = np.asarray(omega, dtype=float)
    if np.any(omega <= 0):
        raise ValueError("omega must be positive")
    k = np.linspace(0.0, k_max, n_k)
    dk = k[1] - k[0]
    transition = e_gap + k**2 / (2.0 * m_r)
    lorentz = (eta / np.pi) / ((transition[None, :] - omega[:, None]) ** 2 + eta**2)
    integral = np.sum(abs(p_vc) ** 2 * lorentz * 2.0 * np.pi * k[None, :], axis=1) * dk
    return (2.0 * np.pi / omega) ** 2 * 2.0 / (2.0 * np.pi) ** 2 * integral


def im_eps_3d(omega: np.ndarray, e_gap: float, m_r: float, p_vc: float) -> np.ndarray:
    """Im[eps(omega)] for a 3D parabolic two-band model (square-root absorption edge)."""
    omega = np.asarray(omega, dtype=float)
    above = omega > e_gap
    result = np.zeros_like(omega)
    result[above] = (
        2.0
        * (2.0 * m_r) ** 1.5
        * abs(p_vc) ** 2
        * np.sqrt(omega[above] - e_gap)
        / omega[above] ** 2
    )
    return result
