import numpy as np
import pytest

import twoband as tb

GAP = 4.45 / tb.HARTREE_TO_EV


def _model_2d():
    return tb.TwoBandModel(
        e_gap=GAP, m_c=0.3, m_v=0.5, p_vc=0.96, k_max=0.5, n_k=201, ky_max=0.5, n_ky=7
    )


def _reference_2d(model, time, a):
    """Complex-valued propagator over the full (kx, ky) grid, written out from the energies."""
    kx, ky2, _ = model.grid()
    dkx = model.k[1] - model.k[0]
    dky = model.ky[1] - model.ky[0]
    weight = -2.0 * dkx * dky / (2.0 * np.pi) ** 2
    dt = time[1] - time[0]
    c_v = np.ones(kx.size, dtype=complex)
    c_c = np.zeros(kx.size, dtype=complex)
    current = np.zeros(time.size)
    for n in range(time.size - 1):
        a_mid = 0.5 * (a[n] + a[n + 1]) / tb.C_AU
        kk = kx + a_mid
        e_v = -(kk**2 + ky2) / (2 * model.m_v)
        e_c = model.e_gap + (kk**2 + ky2) / (2 * model.m_c)
        h0, hz, hx = 0.5 * (e_v + e_c), 0.5 * (e_v - e_c), a_mid * model.p_vc
        w = np.sqrt(hz**2 + hx**2)
        cos, sin_over, phase = np.cos(w * dt), np.sin(w * dt) / w, np.exp(-1j * h0 * dt)
        c_v, c_c = (
            phase * ((cos - 1j * sin_over * hz) * c_v - 1j * sin_over * hx * c_c),
            phase * (-1j * sin_over * hx * c_v + (cos + 1j * sin_over * hz) * c_c),
        )
        kk = kx + a[n + 1] / tb.C_AU
        current[n + 1] = weight * np.sum(
            np.abs(c_c) ** 2 * (kk / model.m_c + kk / model.m_v)
            + 2 * np.real(np.conj(c_v) * c_c) * model.p_vc
        )
    return current - tb.diamagnetic_coefficient(model) * a / tb.C_AU


def test_one_dimensional_grid_is_unchanged():
    model = tb.TwoBandModel(e_gap=GAP, m_c=0.3, m_v=0.5, p_vc=0.96, k_max=0.5, n_k=1201)
    kx, ky2, measure = model.grid()
    np.testing.assert_array_equal(kx, model.k)
    np.testing.assert_array_equal(ky2, 0.0)
    assert measure == pytest.approx((model.k[1] - model.k[0]) / (2 * np.pi))


def test_2d_grid_covers_the_rectangle_with_the_right_area():
    model = _model_2d()
    kx, ky2, measure = model.grid()
    assert kx.size == model.n_k * model.n_ky
    dk = model.k[1] - model.k[0]
    dky = model.ky[1] - model.ky[0]
    assert measure == pytest.approx(dk * dky / (2 * np.pi) ** 2)
    assert ky2.max() == pytest.approx(model.ky_max**2)


def test_ky_grid_validation():
    with pytest.raises(ValueError, match="ky grid"):
        tb.TwoBandModel(e_gap=GAP, m_c=0.3, m_v=0.5, p_vc=1.0, n_ky=2, ky_max=0.5)
    with pytest.raises(ValueError, match="ky grid"):
        tb.TwoBandModel(e_gap=GAP, m_c=0.3, m_v=0.5, p_vc=1.0, n_ky=5, ky_max=0.0)
    with pytest.raises(ValueError, match="n_ky"):
        tb.TwoBandModel(e_gap=GAP, m_c=0.3, m_v=0.5, p_vc=1.0, n_ky=0)


def test_2d_no_field_gives_no_current():
    model = _model_2d()
    time = np.arange(0.0, 50.0, 0.1)
    np.testing.assert_array_equal(tb.propagate(model, time, np.zeros(time.size)), 0.0)


def test_2d_fast_propagator_matches_reference_in_strong_field():
    model = _model_2d()
    time = np.arange(0.0, 300.0, 0.05)
    field = tb.sin4_field(time, amplitude=5e-3, omega=0.057, duration=250.0, center=150.0)
    a = tb.vector_potential(time, field)
    fast = tb.propagate(model, time, a)
    reference = _reference_2d(model, time, a)
    np.testing.assert_allclose(fast, reference, rtol=0, atol=1e-10 * np.abs(reference).max())


def test_2d_weak_field_response_is_linear():
    model = _model_2d()
    time = np.arange(0.0, 400.0, 0.05)
    field = tb.sin4_field(time, amplitude=1e-5, omega=0.4, duration=40.0, center=150.0)
    a = tb.vector_potential(time, field)
    j1 = tb.propagate(model, time, a)
    j2 = tb.propagate(model, time, 2 * a)
    np.testing.assert_allclose(j2, 2 * j1, rtol=0, atol=1e-5 * np.abs(j1).max())


def test_2d_static_vector_potential_gives_no_lasting_current():
    # A slowly switched-on constant A must not drive a lasting current in an insulator, in
    # two dimensions as in one. The diamagnetic term is what cancels it.
    model = _model_2d()
    time = np.arange(0.0, 3000.0, 0.5)
    s = np.clip(time / 2000.0, 0, 1)
    ramp = 1e-4 * tb.C_AU * s**2 * (3 - 2 * s)
    current = tb.propagate(model, time, ramp)
    assert abs(current[-1]) < 1e-3 * tb.diamagnetic_coefficient(model) * 1e-4


def test_2d_absorption_matches_analytic_im_eps_2d():
    # Weak-probe Re[sigma] of the 2D simulator against omega Im[eps_2d] / (4 pi), with the
    # analytic Lorentzian broadening set equal to the damping of the Fourier transform.
    # Above the gap the step is reproduced to a few per cent on this grid.
    m_c, m_v, p_vc = 0.3, 0.5, 0.96
    m_r = tb.reduced_mass(m_c, m_v)
    gamma = 0.5 / tb.HARTREE_TO_EV
    model = tb.TwoBandModel(
        e_gap=GAP, m_c=m_c, m_v=m_v, p_vc=p_vc, k_max=0.5, n_k=101, ky_max=0.5, n_ky=101
    )
    time = np.arange(0.0, 800.0, 0.05)
    field = tb.sin4_field(time, amplitude=1e-5, omega=0.4, duration=40.0, center=200.0)
    a = tb.vector_potential(time, field)
    current = tb.propagate(model, time, a)
    energy = np.array([6.0, 7.0, 8.0, 10.0, 12.0]) / tb.HARTREE_TO_EV
    sigma = tb.optical_conductivity(
        time, current, tb.field_from_vector_potential(time, a), energy, gamma
    )
    expected = energy * tb.im_eps_2d(energy, GAP, m_r, p_vc, eta=gamma) / (4 * np.pi)
    np.testing.assert_allclose(sigma.real, expected, rtol=0.03)


@pytest.mark.parametrize("n_ky", [5, 6])
def test_reduced_points_reproduce_the_full_grid_sum(n_ky):
    # Only ky >= 0 is propagated, with multiplicities. For any function of ky^2 the weighted
    # sum over the reduced points must equal the sum over the full grid, odd or even n_ky.
    from twoband.simulate import _propagation_points

    model = tb.TwoBandModel(
        e_gap=GAP, m_c=0.3, m_v=0.5, p_vc=0.96, k_max=0.5, n_k=21, ky_max=0.5, n_ky=n_ky
    )
    kx_full, ky2_full, measure_full = model.grid()
    kx, ky2, mult, measure = _propagation_points(model)
    assert measure == pytest.approx(measure_full)
    assert kx.size == model.n_k * ((n_ky + 1) // 2)
    for g in (lambda q: q, lambda q: np.exp(-q), lambda q: 1.0 / (1.0 + q)):
        full = np.sum(g(ky2_full) * np.cos(kx_full))
        reduced = np.sum(mult * g(ky2) * np.cos(kx))
        assert reduced == pytest.approx(full, rel=1e-12)
