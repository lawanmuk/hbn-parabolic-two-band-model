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


def im_eps_1d(omega: np.ndarray, e_gap: float, m_r: float, p_vc: float) -> np.ndarray:
    """Im[eps(omega)] for a 1D parabolic two-band model (inverse square-root edge)."""
    omega = np.asarray(omega, dtype=float)
    above = omega > e_gap
    result = np.zeros_like(omega)
    k = np.sqrt(2.0 * m_r * (omega[above] - e_gap))
    jdos = 2.0 * m_r / k  # two roots +-k, each with |d(transition)/dk| = k / m_r
    result[above] = (2.0 * np.pi / omega[above]) ** 2 * 2.0 / (2.0 * np.pi) * abs(p_vc) ** 2 * jdos
    return result


def re_sigma_1d_broadened(
    energy: np.ndarray,
    e_gap: float,
    m_r: float,
    p_vc: float,
    gamma: float,
    e_max: float | None = None,
    n_u: int = 20000,
) -> np.ndarray:
    """Re[sigma] = omega Im[eps_1d] / (4 pi), convolved with a Lorentzian of half-width gamma.

    This is what a damped Fourier transform with damping ``gamma`` extracts from a weak-probe
    simulation of the 1D model. The inverse square-root edge is integrated with the
    substitution omega = e_gap + u^2, which turns the integrand into the smooth function
    2 sqrt(2 m_r) |p_vc|^2 / omega.
    """
    energy = np.asarray(energy, dtype=float)
    e_max = e_gap + 2.0 if e_max is None else e_max
    u = np.linspace(0.0, np.sqrt(e_max - e_gap), n_u)
    omega = e_gap + u**2
    smooth = 2.0 * np.sqrt(2.0 * m_r) * abs(p_vc) ** 2 / omega
    result = np.empty_like(energy)
    flat_energy, flat_result = energy.ravel(), result.ravel()
    for start in range(0, flat_energy.size, 50):
        e = flat_energy[start : start + 50, None]
        lorentz = (gamma / np.pi) / ((omega[None, :] - e) ** 2 + gamma**2)
        flat_result[start : start + 50] = np.trapezoid(smooth[None, :] * lorentz, u, axis=1)
    return flat_result.reshape(energy.shape)
