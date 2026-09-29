"""Damped Fourier transform used for all spectra in this project."""

import numpy as np


def damped_ft(
    time: np.ndarray,
    signal: np.ndarray,
    energy: np.ndarray,
    gamma: float,
    block_size: int = 50,
) -> np.ndarray:
    """Compute F(w) = sum_t signal(t) exp(i w t - gamma t) dt.

    ``signal`` can be 1D (one trace, shape (n_t,)) or 2D (several traces, shape
    (n_traces, n_t)). The input array is never modified.

    The energy axis is processed ``block_size`` points at a time, so memory stays
    small even for long traces (50001 time steps x 50 energies is about 40 MB).

    Returns shape (n_energy,) for 1D input or (n_traces, n_energy) for 2D input.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal)
    energy = np.asarray(energy, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must be 1D with at least two points")
    if signal.shape[-1] != time.size:
        raise ValueError(f"signal has {signal.shape[-1]} time points but time has {time.size}")
    dt = time[1] - time[0]
    damped = signal * np.exp(-gamma * time)
    result = np.empty((*signal.shape[:-1], energy.size), dtype=complex)
    for start in range(0, energy.size, block_size):
        block = energy[start : start + block_size]
        phase = np.exp(1j * np.outer(time, block))  # (n_t, len(block))
        result[..., start : start + block_size] = damped @ phase
    return result * dt
