import numpy as np
import pytest

from twoband import conduction_band, reduced_mass, valence_band


def test_valence_band_divides_by_mass():
    # This was the original notebook bug: -k**2/2*m multiplies by m.
    assert valence_band(np.array([1.0]), m_v=0.5)[0] == pytest.approx(-1.0)


def test_conduction_band_starts_at_gap():
    assert conduction_band(np.array([0.0]), m_c=0.3, e_gap=0.16)[0] == pytest.approx(0.16)


def test_transition_energy_uses_reduced_mass():
    k = np.linspace(-1, 1, 11)
    m_c, m_v, gap = 0.3, 0.5, 0.16
    transition = conduction_band(k, m_c, gap) - valence_band(k, m_v)
    np.testing.assert_allclose(transition, gap + k**2 / (2 * reduced_mass(m_c, m_v)))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_mass_raises(bad):
    with pytest.raises(ValueError):
        valence_band(np.array([1.0]), m_v=bad)
    with pytest.raises(ValueError):
        conduction_band(np.array([1.0]), m_c=bad, e_gap=0.1)
