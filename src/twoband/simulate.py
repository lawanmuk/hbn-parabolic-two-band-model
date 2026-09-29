"""Real-time simulation of the 1D parabolic two-band model.

Model (velocity gauge, atomic units, electron charge -1):

    H_k(t) = [[eps_v(k + A/c), (A/c) p_vc],
              [(A/c) p_vc,     eps_c(k + A/c)]]

with eps_v(k) = -k^2 / (2 m_v) and eps_c(k) = e_gap + k^2 / (2 m_c). Every k point is an
independent two-level system that starts in the valence band. The 2x2 propagator is applied
exactly at each time step, using the vector potential at the midpoint of the step.

The current density is

    J(t) = -2 * sum_k dk/(2 pi) [ |c_c|^2 (v_c - v_v) + 2 Re(c_v* c_c) p_vc ],

where v_n = d eps_n / dk at k + A/c. The factor 2 is spin. The contribution of the filled
valence band is removed, as it carries no current in a real crystal.

Truncating to two bands breaks the f-sum rule, which leaves a spurious 1/omega term in the
conductivity. It is removed with a diamagnetic term -D A/c, where D is fixed by requiring
that the static (omega -> 0) linear response vanishes, as it must for an insulator:

    D = 2 * sum_k dk/(2 pi) * 2 p_vc^2 / (eps_c(k) - eps_v(k)).
"""

from dataclasses import dataclass

import numpy as np

from twoband.constants import C_AU


@dataclass(frozen=True)
class TwoBandModel:
    """Parameters of the parabolic two-band model (atomic units)."""

    e_gap: float
    m_c: float
    m_v: float
    p_vc: float
    k_max: float = 0.8
    n_k: int = 3201

    def __post_init__(self):
        if self.m_c <= 0 or self.m_v <= 0:
            raise ValueError("effective masses must be positive")
        if self.e_gap <= 0:
            raise ValueError("e_gap must be positive")
        if self.n_k < 3:
            raise ValueError("n_k must be at least 3")

    @property
    def k(self) -> np.ndarray:
        return np.linspace(-self.k_max, self.k_max, self.n_k)


def propagate(model: TwoBandModel, time: np.ndarray, vector_potential: np.ndarray) -> np.ndarray:
    """Evolve the model under A(t) and return the current density J(t).

    ``vector_potential`` can be 1D (one run, shape (n_t,)) or 2D (several independent runs,
    shape (n_runs, n_t)); the result has the same shape.
    """
    time = np.asarray(time, dtype=float)
    a_all = np.asarray(vector_potential, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be 1D with at least two points")
    if a_all.shape[-1] != time.size:
        raise ValueError("vector_potential and time have different lengths")
    single = a_all.ndim == 1
    a_all = np.atleast_2d(a_all)
    dt = time[1] - time[0]

    k = model.k[None, :]
    dk = k[0, 1] - k[0, 0]
    weight = -2.0 * dk / (2.0 * np.pi)
    n_runs = a_all.shape[0]

    c_v = np.ones((n_runs, model.n_k), dtype=complex)
    c_c = np.zeros((n_runs, model.n_k), dtype=complex)
    current = np.zeros((n_runs, time.size))

    for n in range(time.size - 1):
        a_mid = 0.5 * (a_all[:, n] + a_all[:, n + 1])[:, None] / C_AU
        kk = k + a_mid
        e_v = -(kk**2) / (2.0 * model.m_v)
        e_c = model.e_gap + kk**2 / (2.0 * model.m_c)
        h0 = 0.5 * (e_v + e_c)
        hz = 0.5 * (e_v - e_c)
        hx = a_mid * model.p_vc
        omega = np.sqrt(hz**2 + hx**2)
        cos = np.cos(omega * dt)
        sin_over = np.sin(omega * dt) / omega
        phase = np.exp(-1j * h0 * dt)
        new_v = phase * ((cos - 1j * sin_over * hz) * c_v - 1j * sin_over * hx * c_c)
        new_c = phase * (-1j * sin_over * hx * c_v + (cos + 1j * sin_over * hz) * c_c)
        c_v, c_c = new_v, new_c

        kk = k + a_all[:, n + 1][:, None] / C_AU
        v_diff = kk / model.m_c + kk / model.m_v
        current[:, n + 1] = weight * np.sum(
            np.abs(c_c) ** 2 * v_diff + 2.0 * np.real(np.conj(c_v) * c_c) * model.p_vc, axis=1
        )

    current -= diamagnetic_coefficient(model) * a_all / C_AU
    return current[0] if single else current


def diamagnetic_coefficient(model: TwoBandModel) -> float:
    """Sum-rule coefficient D of the diamagnetic current -D A/c (see module docstring)."""
    k = model.k
    dk = k[1] - k[0]
    transition = model.e_gap + k**2 / (2.0 * model.m_c) + k**2 / (2.0 * model.m_v)
    return float(2.0 * dk / (2.0 * np.pi) * np.sum(2.0 * model.p_vc**2 / transition))
