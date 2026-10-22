"""Density-matrix propagation of the two-band model with dephasing.

Each k point carries a 2x2 density matrix rho, with rho = c c^dagger for the pure state
(c_v, c_c) of the simulator. The coherent part of the evolution is the same SU(2) step,
U = [[alpha, beta], [-beta*, alpha*]], applied as rho -> U rho U^dagger. Dephasing with time
T2 multiplies the off-diagonal elements by exp(-t / T2). The populations are untouched, so
there is no relaxation to the valence band, only loss of coherence.

The step is split symmetrically: half the decay, the unitary step, then half the decay. This
second-order splitting is exact for the decay alone, and with T2 -> infinity it reduces to
the pure-state propagator.

The current is the same expression as in ``twoband.simulate``, written in terms of rho:
    J = -2 * sum_k w_k [ rho_cc (v_c - v_v) + 2 p_vc Re(rho_cv) ] - D A / c.
"""

import numpy as np

from twoband.constants import C_AU
from twoband.simulate import TwoBandModel, _propagation_points, diamagnetic_coefficient


def _evolve(model, time, vector_potential, t2, rho0=None):
    """Evolve the density matrices. Returns ``(current, rho)`` with rho of shape (n_pts, 2, 2)."""
    time = np.asarray(time, dtype=float)
    a = np.asarray(vector_potential, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be 1D with at least two points")
    if a.shape != time.shape:
        raise ValueError("vector_potential and time have different lengths")
    if t2 <= 0:
        raise ValueError("t2 must be positive")
    dt = time[1] - time[0]
    decay_half = np.exp(-0.5 * dt / t2)  # 0 < t2 <= inf; for t2 = inf this is exactly 1

    kx, ky2, mult, measure = _propagation_points(model)
    n_pts = kx.size
    inv_2mr = 0.5 * (1.0 / model.m_c + 1.0 / model.m_v)
    inv_mr = 2.0 * inv_2mr
    ky_term = -0.5 * inv_2mr * ky2
    a_mid = 0.5 * (a[:-1] + a[1:]) / C_AU
    a_next = a[1:] / C_AU
    weight = -2.0 * measure

    if rho0 is None:
        rho = np.zeros((n_pts, 2, 2), dtype=complex)
        rho[:, 0, 0] = 1.0  # every k starts in the valence band
    else:
        rho = np.array(rho0, dtype=complex, copy=True)
    current = np.zeros(time.size)
    scale = np.array([[1.0, decay_half], [decay_half, 1.0]])

    for n in range(time.size - 1):
        hx = a_mid[n] * model.p_vc
        kk = kx + a_mid[n]
        hz = -0.5 * inv_2mr * kk**2 + ky_term - 0.5 * model.e_gap
        w = np.sqrt(hz**2 + hx**2)
        cos = np.cos(w * dt)
        sin_over = np.sin(w * dt) / w
        alpha = cos - 1j * sin_over * hz
        beta = -1j * sin_over * hx
        unitary = np.empty((n_pts, 2, 2), dtype=complex)
        unitary[:, 0, 0] = alpha
        unitary[:, 0, 1] = beta
        unitary[:, 1, 0] = -np.conj(beta)
        unitary[:, 1, 1] = np.conj(alpha)

        rho *= scale  # half decay of the coherences
        rho = unitary @ rho @ np.conj(np.transpose(unitary, (0, 2, 1)))
        rho *= scale  # second half

        kk = kx + a_next[n]
        rho_cc = rho[:, 1, 1].real
        rho_cv = rho[:, 1, 0].real
        current[n + 1] = weight * (
            np.sum(mult * rho_cc * kk) * inv_mr + 2.0 * model.p_vc * np.sum(mult * rho_cv)
        )

    current -= diamagnetic_coefficient(model) * a / C_AU
    return current, rho


def propagate_dephased(
    model: TwoBandModel, time: np.ndarray, vector_potential: np.ndarray, t2: float
) -> np.ndarray:
    """Current J(t) of the two-band model with coherence lost at rate 1/t2 (atomic units).

    ``t2 = np.inf`` gives the pure-state result of :func:`twoband.propagate`. Only a single run
    is supported.
    """
    current, _ = _evolve(model, time, vector_potential, t2)
    return current
