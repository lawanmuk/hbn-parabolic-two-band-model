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
    """Parameters of the parabolic two-band model (atomic units).

    With the default ``n_ky = 1`` the model is one-dimensional. With ``n_ky > 1`` the k grid
    gets a second direction, ky, perpendicular to the field. The field and the dipole stay
    along x, so ky does not couple to the vector potential: it only adds ky^2 to the band
    energies and to the k-space area element.
    """

    e_gap: float
    m_c: float
    m_v: float
    p_vc: float
    k_max: float = 0.8
    n_k: int = 3201
    ky_max: float = 0.0
    n_ky: int = 1

    def __post_init__(self):
        if self.m_c <= 0 or self.m_v <= 0:
            raise ValueError("effective masses must be positive")
        if self.e_gap <= 0:
            raise ValueError("e_gap must be positive")
        if self.n_k < 3:
            raise ValueError("n_k must be at least 3")
        if self.n_ky < 1:
            raise ValueError("n_ky must be at least 1")
        if self.n_ky > 1 and (self.n_ky < 3 or self.ky_max <= 0):
            raise ValueError("a ky grid needs n_ky >= 3 and ky_max > 0")

    @property
    def k(self) -> np.ndarray:
        """The kx grid (the field direction)."""
        return np.linspace(-self.k_max, self.k_max, self.n_k)

    @property
    def ky(self) -> np.ndarray:
        """The ky grid, or the single point ky = 0 in one dimension."""
        if self.n_ky == 1:
            return np.zeros(1)
        return np.linspace(-self.ky_max, self.ky_max, self.n_ky)

    def grid(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Flattened k points as ``(kx, ky^2, measure)``.

        ``measure`` is the k-space area element per point divided by (2 pi)^d, with d the
        dimension, so that a sum over the points times 2 gives the spin-summed integral.
        """
        kx = self.k
        ky = self.ky
        dk = kx[1] - kx[0]
        if self.n_ky == 1:
            return kx.copy(), np.zeros(kx.size), dk / (2.0 * np.pi)
        dky = ky[1] - ky[0]
        kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
        measure = dk * dky / (2.0 * np.pi) ** 2
        return kxg.ravel(), (kyg**2).ravel(), measure


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

    kx, ky2, measure = model.grid()
    k = kx[None, :]
    weight = -2.0 * measure
    n_runs = a_all.shape[0]
    inv_2mr = 0.5 * (1.0 / model.m_c + 1.0 / model.m_v)  # 1 / (2 m_r)
    inv_mr = 2.0 * inv_2mr
    ky_term = (-0.5 * inv_2mr * ky2)[None, :]  # the ky part of hz, independent of the field
    a_mid_all = 0.5 * (a_all[:, :-1] + a_all[:, 1:]) / C_AU
    a_next_all = a_all[:, 1:] / C_AU

    # Amplitudes c_v = vr + i vi and c_c = cr + i ci, kept as real arrays and updated in place.
    shape = (n_runs, kx.size)
    vr, vi = np.ones(shape), np.zeros(shape)
    cr, ci = np.zeros(shape), np.zeros(shape)
    current = np.zeros((n_runs, time.size))

    # Work buffers, reused at every step to avoid allocating temporaries.
    kk, hz, omega, cos, sin_over, b, c = (np.empty(shape) for _ in range(7))
    t1, t2, t3, t4 = (np.empty(shape) for _ in range(4))

    # The common phase exp(-i (eps_v + eps_c) dt / 2) multiplies both amplitudes equally and
    # cancels in every observable (|c_c|^2 and Re(c_v* c_c)), so it is left out. What remains
    # is the exact SU(2) step
    #   c_v' = cos c_v - i (b c_v + c c_c),   c_c' = cos c_c + i (b c_c - c c_v),
    # with b = sin(w dt) hz / w, c = sin(w dt) hx / w and w = sqrt(hz^2 + hx^2).
    for n in range(time.size - 1):
        a_mid = a_mid_all[:, n : n + 1]
        hx = a_mid * model.p_vc  # (n_runs, 1): interband coupling
        np.add(k, a_mid, out=kk)
        np.multiply(kk, kk, out=hz)
        hz *= -0.5 * inv_2mr
        hz += ky_term
        hz -= 0.5 * model.e_gap  # hz = (eps_v - eps_c) / 2
        np.multiply(hz, hz, out=omega)
        omega += hx * hx
        np.sqrt(omega, out=omega)
        np.multiply(omega, dt, out=t1)
        np.cos(t1, out=cos)
        np.sin(t1, out=sin_over)
        sin_over /= omega
        np.multiply(sin_over, hz, out=b)
        np.multiply(sin_over, hx, out=c)

        # y = b c_v + c c_c  ->  c_v' = cos c_v + Im(y) - i Re(y)
        np.multiply(b, vr, out=t1)
        t1 += c * cr  # Re(y)
        np.multiply(b, vi, out=t2)
        t2 += c * ci  # Im(y)
        # x = b c_c - c c_v  ->  c_c' = cos c_c - Im(x) + i Re(x)
        np.multiply(b, cr, out=t3)
        t3 -= c * vr  # Re(x)
        np.multiply(b, ci, out=t4)
        t4 -= c * vi  # Im(x)
        vr *= cos
        vr += t2
        vi *= cos
        vi -= t1
        cr *= cos
        cr -= t4
        ci *= cos
        ci += t3

        np.add(k, a_next_all[:, n : n + 1], out=kk)
        np.multiply(cr, cr, out=t1)
        t1 += ci * ci  # |c_c|^2
        np.multiply(vr, cr, out=t2)
        t2 += vi * ci  # Re(c_v* c_c)
        current[:, n + 1] = weight * (
            np.einsum("ij,ij->i", t1, kk) * inv_mr + 2.0 * model.p_vc * t2.sum(axis=1)
        )

    current -= diamagnetic_coefficient(model) * a_all / C_AU
    return current[0] if single else current


def diamagnetic_coefficient(model: TwoBandModel) -> float:
    """Sum-rule coefficient D of the diamagnetic current -D A/c (see module docstring)."""
    kx, ky2, measure = model.grid()
    inv_2mr = 0.5 * (1.0 / model.m_c + 1.0 / model.m_v)
    transition = model.e_gap + kx**2 * inv_2mr + ky2 * inv_2mr
    return float(2.0 * measure * np.sum(2.0 * model.p_vc**2 / transition))
