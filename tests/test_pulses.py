import numpy as np
import pytest

import twoband as tb


def test_intensity_conversion():
    assert tb.intensity_to_field(3.50944758e16) == pytest.approx(1.0)
    assert tb.intensity_to_field(4 * 3.50944758e16) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        tb.intensity_to_field(-1.0)


def test_sin4_field_is_zero_outside_the_pulse():
    time = np.linspace(0, 100, 1001)
    field = tb.sin4_field(time, amplitude=1.0, omega=0.5, duration=40.0, center=50.0)
    assert np.all(field[time <= 30.0] == 0.0)
    assert np.all(field[time >= 70.0] == 0.0)
    assert field[500] == pytest.approx(1.0)  # peak at the center, cos(0) = 1


def test_vector_potential_round_trip():
    time = np.linspace(0, 200, 20001)
    field = tb.sin4_field(time, amplitude=1e-3, omega=0.3, duration=80.0, center=100.0)
    a = tb.vector_potential(time, field)
    np.testing.assert_allclose(tb.field_from_vector_potential(time, a), field, atol=1e-8)


def test_vector_potential_handles_several_pulses():
    time = np.linspace(0, 100, 1001)
    fields = np.vstack([np.ones(1001), 2 * np.ones(1001)])
    a = tb.vector_potential(time, fields)
    assert a.shape == (2, 1001)
    assert a[1, -1] == pytest.approx(-tb.C_AU * 200.0)
