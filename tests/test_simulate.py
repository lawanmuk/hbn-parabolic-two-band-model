import numpy as np
import pytest

import twoband as tb

GAP = 4.45 / tb.HARTREE_TO_EV


@pytest.fixture(scope="module")
def model():
    return tb.TwoBandModel(e_gap=GAP, m_c=0.3, m_v=0.5, p_vc=0.96, k_max=0.5, n_k=1201)


@pytest.fixture(scope="module")
def weak_probe():
    time = np.arange(0.0, 800.0, 0.05)
    field = tb.sin4_field(time, amplitude=1e-5, omega=0.4, duration=40.0, center=200.0)
    return time, tb.vector_potential(time, field)


def test_no_field_gives_no_current(model, weak_probe):
    time, _ = weak_probe
    current = tb.propagate(model, time[:2000], np.zeros(2000))
    np.testing.assert_array_equal(current, 0.0)


def test_weak_field_response_is_linear(model, weak_probe):
    time, a = weak_probe
    j1 = tb.propagate(model, time[:6000], a[:6000])
    j2 = tb.propagate(model, time[:6000], 2 * a[:6000])
    np.testing.assert_allclose(j2, 2 * j1, rtol=0, atol=1e-5 * np.abs(j1).max())


def test_several_runs_match_separate_runs(model, weak_probe):
    time, a = weak_probe
    time, a = time[:3000], a[:3000]
    both = tb.propagate(model, time, np.vstack([a, 3 * a]))
    np.testing.assert_allclose(both[0], tb.propagate(model, time, a))
    np.testing.assert_allclose(both[1], tb.propagate(model, time, 3 * a))


def test_linear_response_matches_analytic_absorption(model, weak_probe):
    # Weak-probe Re[sigma] must equal omega Im[eps] / (4 pi) of the analytic 1D model,
    # broadened by the Lorentzian that the damped Fourier transform introduces.
    time, a = weak_probe
    current = tb.propagate(model, time, a)
    field = tb.field_from_vector_potential(time, a)
    gamma = 0.25 / tb.HARTREE_TO_EV
    energy = np.array([6.0, 8.0, 12.0]) / tb.HARTREE_TO_EV
    sigma = tb.optical_conductivity(time, current, field, energy, gamma)

    m_r = tb.reduced_mass(model.m_c, model.m_v)
    expected = tb.re_sigma_1d_broadened(energy, GAP, m_r, model.p_vc, gamma)

    np.testing.assert_allclose(sigma.real, expected, rtol=0.02)


def test_sum_rule_removes_static_response(model):
    # A slowly switched-on constant A must not drive a lasting current in an insulator.
    time = np.arange(0.0, 3000.0, 0.5)
    ramp = (
        1e-4 * tb.C_AU * np.clip(time / 2000.0, 0, 1) ** 2 * (3 - 2 * np.clip(time / 2000.0, 0, 1))
    )
    current = tb.propagate(model, time, ramp)
    assert abs(current[-1]) < 1e-3 * tb.diamagnetic_coefficient(model) * 1e-4


def test_invalid_model_parameters():
    with pytest.raises(ValueError):
        tb.TwoBandModel(e_gap=0.1, m_c=-0.3, m_v=0.5, p_vc=1.0)
    with pytest.raises(ValueError):
        tb.TwoBandModel(e_gap=0.0, m_c=0.3, m_v=0.5, p_vc=1.0)


def test_length_mismatch_raises(model):
    with pytest.raises(ValueError, match="different lengths"):
        tb.propagate(model, np.arange(10.0), np.zeros(9))


def _reference_propagate(model, time, a):
    """Straightforward complex-valued propagator, including the global phase."""
    dt = time[1] - time[0]
    k = model.k
    dk = k[1] - k[0]
    c_v = np.ones(k.size, dtype=complex)
    c_c = np.zeros(k.size, dtype=complex)
    current = np.zeros(time.size)
    for n in range(time.size - 1):
        a_mid = 0.5 * (a[n] + a[n + 1]) / tb.C_AU
        kk = k + a_mid
        e_v = -(kk**2) / (2 * model.m_v)
        e_c = model.e_gap + kk**2 / (2 * model.m_c)
        h0, hz, hx = 0.5 * (e_v + e_c), 0.5 * (e_v - e_c), a_mid * model.p_vc
        w = np.sqrt(hz**2 + hx**2)
        cos, sin_over, phase = np.cos(w * dt), np.sin(w * dt) / w, np.exp(-1j * h0 * dt)
        c_v, c_c = (
            phase * ((cos - 1j * sin_over * hz) * c_v - 1j * sin_over * hx * c_c),
            phase * (-1j * sin_over * hx * c_v + (cos + 1j * sin_over * hz) * c_c),
        )
        kk = k + a[n + 1] / tb.C_AU
        current[n + 1] = (
            -2
            * dk
            / (2 * np.pi)
            * np.sum(
                np.abs(c_c) ** 2 * (kk / model.m_c + kk / model.m_v)
                + 2 * np.real(np.conj(c_v) * c_c) * model.p_vc
            )
        )
    return current - tb.diamagnetic_coefficient(model) * a / tb.C_AU


def test_fast_propagator_matches_reference_in_strong_field(model):
    # Strong enough to excite real carriers, so populations, coherences and the
    # intraband current all matter. The global phase dropped by the fast loop must not.
    time = np.arange(0.0, 300.0, 0.05)
    field = tb.sin4_field(time, amplitude=5e-3, omega=0.057, duration=250.0, center=150.0)
    a = tb.vector_potential(time, field)
    fast = tb.propagate(model, time, a)
    reference = _reference_propagate(model, time, a)
    np.testing.assert_allclose(fast, reference, rtol=0, atol=1e-10 * np.abs(reference).max())
