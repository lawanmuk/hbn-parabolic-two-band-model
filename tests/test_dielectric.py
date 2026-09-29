import numpy as np
import pytest

from twoband import HARTREE_TO_EV, im_eps_2d, im_eps_3d

GAP = 4.45 / HARTREE_TO_EV


def test_2d_matches_analytic_step_well_above_gap():
    m_r, p_vc = 0.1875, 0.96
    omega = np.array([8.0, 12.0]) / HARTREE_TO_EV
    numeric = im_eps_2d(omega, GAP, m_r, p_vc)
    analytic = (2 * np.pi / omega) ** 2 * 2 / (2 * np.pi) ** 2 * p_vc**2 * 2 * np.pi * m_r
    np.testing.assert_allclose(numeric, analytic, rtol=0.01)


def test_2d_absorption_is_small_below_gap():
    # Compare omega^2 * Im[eps] so the 1/omega^2 prefactor does not hide the step.
    omega = np.array([2.0, 12.0]) / HARTREE_TO_EV
    below, above = im_eps_2d(omega, GAP, 0.1875, 0.96) * omega**2
    assert below < 0.01 * above


def test_2d_rejects_non_positive_frequency():
    with pytest.raises(ValueError):
        im_eps_2d(np.array([0.0, 0.1]), GAP, 0.1875, 0.96)


def test_3d_is_zero_below_gap():
    omega = np.linspace(0.01, GAP, 50)
    assert np.all(im_eps_3d(omega, GAP, 0.067, 0.9) == 0.0)


def test_3d_square_root_onset():
    # Just above the gap, Im[eps] * omega^2 grows like sqrt(omega - gap).
    d1, d2 = 1e-4, 4e-4
    omega = np.array([GAP + d1, GAP + d2])
    scaled = im_eps_3d(omega, GAP, 0.067, 0.9) * omega**2
    assert scaled[1] / scaled[0] == pytest.approx(np.sqrt(d2 / d1), rel=1e-6)


def test_3d_uses_three_halves_power():
    # The notebook bug computed (2 m)^3 / 2 instead of (2 m)^1.5.
    omega = np.array([GAP + 0.01])
    ratio = im_eps_3d(omega, GAP, 0.2, 1.0) / im_eps_3d(omega, GAP, 0.1, 1.0)
    assert ratio[0] == pytest.approx(2**1.5)
