"""Laser pulses in atomic units."""

import numpy as np

from twoband.constants import C_AU

ATOMIC_UNIT_OF_INTENSITY = 3.50944758e16
"""Intensity in W/cm^2 that corresponds to a field amplitude of 1 atomic unit."""


def intensity_to_field(intensity_w_cm2: float) -> float:
    """Peak electric field (a.u.) of a laser with the given peak intensity in W/cm^2."""
    if intensity_w_cm2 < 0:
        raise ValueError("intensity must be non-negative")
    return float(np.sqrt(intensity_w_cm2 / ATOMIC_UNIT_OF_INTENSITY))


def sin4_field(
    time: np.ndarray, amplitude: float, omega: float, duration: float, center: float
) -> np.ndarray:
    """E(t) = amplitude * sin^4(pi (t - t_start) / duration) * cos(omega (t - center)).

    The envelope is non-zero only for |t - center| < duration / 2.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    time = np.asarray(time, dtype=float)
    start = center - duration / 2.0
    inside = (time > start) & (time < start + duration)
    envelope = np.where(inside, np.sin(np.pi * (time - start) / duration) ** 4, 0.0)
    return amplitude * envelope * np.cos(omega * (time - center))


def vector_potential(time: np.ndarray, field: np.ndarray) -> np.ndarray:
    """A(t) = -c * integral of E from time[0] to t (trapezoid rule), so E = -(1/c) dA/dt."""
    time = np.asarray(time, dtype=float)
    field = np.asarray(field, dtype=float)
    dt = np.diff(time)
    steps = 0.5 * (field[..., 1:] + field[..., :-1]) * dt
    integral = np.concatenate(
        [np.zeros(field.shape[:-1] + (1,)), np.cumsum(steps, axis=-1)], axis=-1
    )
    return -C_AU * integral


def field_from_vector_potential(time: np.ndarray, vector_potential: np.ndarray) -> np.ndarray:
    """E(t) = -(1/c) dA/dt, by central differences."""
    time = np.asarray(time, dtype=float)
    return -np.gradient(np.asarray(vector_potential, dtype=float), time, axis=-1) / C_AU
