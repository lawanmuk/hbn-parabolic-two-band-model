import numpy as np
import pytest

from twoband import (
    optical_conductivity,
    split_delays,
    subtract_reference,
    transient_absorption,
)


@pytest.fixture
def traces():
    time = np.arange(0.0, 200.0, 0.1)
    efield = np.exp(-(((time - 20.0) / 3.0) ** 2)) * np.cos(0.2 * time)
    return time, efield


def test_ohmic_current_gives_constant_conductivity(traces):
    time, efield = traces
    energy = np.linspace(0.1, 0.3, 30)
    sigma = optical_conductivity(time, 0.7 * efield, efield, energy, gamma=0.01)
    np.testing.assert_allclose(sigma, 0.7)


def test_conductivity_for_several_delays(traces):
    time, efield = traces
    energy = np.linspace(0.1, 0.3, 30)
    currents = np.vstack([0.5 * efield, 1.5 * efield])
    fields = np.vstack([efield, efield])
    sigma = optical_conductivity(time, currents, fields, energy, gamma=0.01)
    assert sigma.shape == (2, 30)
    np.testing.assert_allclose(sigma[0], 0.5)
    np.testing.assert_allclose(sigma[1], 1.5)


def test_split_delays_shape_and_order():
    data = np.arange(30 * 8).reshape(30, 8)
    blocks = split_delays(data, 10)
    assert blocks.shape == (3, 10, 8)
    np.testing.assert_array_equal(blocks[1, 0], data[10])


def test_split_delays_rejects_ragged_data():
    with pytest.raises(ValueError, match="do not split evenly"):
        split_delays(np.zeros((31, 8)), 10)


def test_subtract_reference_for_each_delay():
    pump = np.array([1.0, 2.0, 3.0])
    pump_probe = np.array([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])
    probe = subtract_reference(pump_probe, pump)
    np.testing.assert_array_equal(probe, [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])


def test_subtract_reference_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        subtract_reference(np.zeros(10), np.zeros(9))


def test_transient_absorption_is_zero_without_pump_effect():
    sigma_eq = np.linspace(0, 1, 50) + 0.1j
    delta = transient_absorption(np.vstack([sigma_eq, sigma_eq]), sigma_eq)
    assert delta.shape == (2, 50)
    np.testing.assert_array_equal(delta, 0)


def test_transient_absorption_rejects_different_grids():
    with pytest.raises(ValueError, match="energy grids"):
        transient_absorption(np.zeros((2, 50)), np.zeros(49))
