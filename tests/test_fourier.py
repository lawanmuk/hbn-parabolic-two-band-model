import numpy as np
import pytest

from twoband import HARTREE_TO_EV, damped_ft


def sin4_pulse(time, carrier, width):
    """sin^4 envelope pulse, zero after ``width``."""
    envelope = np.sin(np.pi * time / width) ** 4 * (time < width)
    return envelope * np.cos(carrier * time)


@pytest.fixture
def pulse():
    time = np.arange(0.0, 400.0, 0.1)
    carrier = 5.0 / HARTREE_TO_EV
    return time, sin4_pulse(time, carrier, 100.0), carrier


def test_peak_at_carrier_frequency(pulse):
    time, signal, carrier = pulse
    energy = np.linspace(1.0, 10.0, 901) / HARTREE_TO_EV
    spectrum = damped_ft(time, signal, energy, gamma=0.01)
    peak = energy[np.argmax(np.abs(spectrum))]
    assert peak == pytest.approx(carrier, abs=0.02 / HARTREE_TO_EV)


def test_matches_direct_sum(pulse):
    time, signal, _ = pulse
    energy = np.linspace(1.0, 10.0, 37) / HARTREE_TO_EV
    gamma = 0.01
    dt = time[1] - time[0]
    expected = np.array(
        [np.sum(signal * np.exp(1j * w * time - gamma * time)) * dt for w in energy]
    )
    np.testing.assert_allclose(damped_ft(time, signal, energy, gamma), expected, rtol=1e-10)


def test_result_does_not_depend_on_block_size(pulse):
    time, signal, _ = pulse
    energy = np.linspace(1.0, 10.0, 123) / HARTREE_TO_EV
    a = damped_ft(time, signal, energy, 0.01, block_size=7)
    b = damped_ft(time, signal, energy, 0.01, block_size=500)
    np.testing.assert_allclose(a, b)


def test_input_is_not_modified(pulse):
    time, signal, _ = pulse
    original = signal.copy()
    damped_ft(time, signal, np.linspace(0.1, 0.3, 10), gamma=0.01)
    np.testing.assert_array_equal(signal, original)


def test_2d_input_gives_one_row_per_trace(pulse):
    time, signal, _ = pulse
    energy = np.linspace(0.1, 0.3, 20)
    stacked = np.vstack([signal, 2 * signal, 3 * signal])
    result = damped_ft(time, stacked, energy, gamma=0.01)
    assert result.shape == (3, 20)
    single = damped_ft(time, signal, energy, gamma=0.01)
    np.testing.assert_allclose(result[2], 3 * single)


def test_length_mismatch_raises(pulse):
    time, signal, _ = pulse
    with pytest.raises(ValueError, match="time points"):
        damped_ft(time, signal[:-1], np.array([0.1]), gamma=0.01)
