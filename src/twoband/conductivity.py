"""Optical conductivity and transient absorption from current and field traces."""

import numpy as np

from twoband.fourier import damped_ft


def optical_conductivity(
    time: np.ndarray,
    current: np.ndarray,
    efield: np.ndarray,
    energy: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """sigma(w) = J(w) / E(w), both from the same damped Fourier transform.

    Works for one trace (1D arrays) or several delays at once (2D, one row per delay).
    """
    j_w = damped_ft(time, current, energy, gamma)
    e_w = damped_ft(time, efield, energy, gamma)
    return j_w / e_w


def split_delays(data: np.ndarray, block_length: int) -> np.ndarray:
    """Split a concatenated delay scan into blocks of ``block_length`` rows.

    Returns an array of shape (n_delays, block_length, n_columns). Raises if the
    data does not divide evenly, instead of silently producing ragged blocks.
    """
    data = np.asarray(data)
    n_rows = data.shape[0]
    if n_rows % block_length != 0:
        raise ValueError(f"{n_rows} rows do not split evenly into blocks of {block_length}")
    return data.reshape(n_rows // block_length, block_length, *data.shape[1:])


def subtract_reference(pump_probe: np.ndarray, pump_only: np.ndarray) -> np.ndarray:
    """Probe signal = pump-probe signal - pump-only signal.

    ``pump_probe`` may be 1D (one delay) or 2D (one row per delay); ``pump_only`` is 1D.
    """
    pump_probe = np.asarray(pump_probe)
    pump_only = np.asarray(pump_only)
    if pump_probe.shape[-1] != pump_only.shape[-1]:
        raise ValueError(
            f"length mismatch: pump-probe {pump_probe.shape[-1]}, pump {pump_only.shape[-1]}"
        )
    return pump_probe - pump_only


def transient_absorption(sigma_transient: np.ndarray, sigma_equilibrium: np.ndarray) -> np.ndarray:
    """Delta sigma(w, T) = sigma(w, T) - sigma(w), one row per delay T.

    Both inputs must be on the same energy grid.
    """
    sigma_transient = np.atleast_2d(sigma_transient)
    sigma_equilibrium = np.asarray(sigma_equilibrium)
    if sigma_transient.shape[-1] != sigma_equilibrium.shape[-1]:
        raise ValueError("transient and equilibrium spectra are on different energy grids")
    return sigma_transient - sigma_equilibrium[None, :]
