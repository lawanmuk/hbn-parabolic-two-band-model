"""Parabolic band dispersions (atomic units)."""

import numpy as np


def valence_band(k: np.ndarray, m_v: float) -> np.ndarray:
    """Valence band energy: -k^2 / (2 m_v)."""
    if m_v <= 0:
        raise ValueError("m_v must be positive")
    return -(np.asarray(k) ** 2) / (2.0 * m_v)


def conduction_band(k: np.ndarray, m_c: float, e_gap: float) -> np.ndarray:
    """Conduction band energy: e_gap + k^2 / (2 m_c)."""
    if m_c <= 0:
        raise ValueError("m_c must be positive")
    return e_gap + np.asarray(k) ** 2 / (2.0 * m_c)


def reduced_mass(m_c: float, m_v: float) -> float:
    """Reduced electron-hole mass m_c m_v / (m_c + m_v)."""
    if m_c <= 0 or m_v <= 0:
        raise ValueError("masses must be positive")
    return m_c * m_v / (m_c + m_v)
