import numpy as np
import pytest

import twoband as tb
from twoband.density import _evolve

GAP = 4.45 / tb.HARTREE_TO_EV


@pytest.fixture(scope="module")
def model():
    return tb.TwoBandModel(e_gap=GAP, m_c=0.3, m_v=0.5, p_vc=0.96, k_max=0.5, n_k=401)


@pytest.fixture(scope="module")
def strong_pulse():
    # Strong enough to excite real carriers, so the coherences matter.
    time = np.arange(0.0, 300.0, 0.05)
    field = tb.sin4_field(time, amplitude=0.02, omega=0.2, duration=150.0, center=150.0)
    return time, tb.vector_potential(time, field)


def test_infinite_t2_is_the_pure_state_propagator(model, strong_pulse):
    time, a = strong_pulse
    pure = tb.propagate(model, time, a)
    dephased = tb.propagate_dephased(model, time, a, np.inf)
    np.testing.assert_allclose(dephased, pure, rtol=0, atol=1e-9 * np.abs(pure).max())


def test_density_matrix_stays_a_valid_state(model, strong_pulse):
    time, a = strong_pulse
    _, rho = _evolve(model, time, a, t2=5.0)
    trace = rho[:, 0, 0].real + rho[:, 1, 1].real
    np.testing.assert_allclose(trace, 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(rho, np.conj(np.transpose(rho, (0, 2, 1))), atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(rho)
    assert eigenvalues.min() > -1e-9


def test_free_coherence_decays_as_exp_t_over_t2(model):
    # With no field the Hamiltonian is diagonal, so the energy phase does not change |rho_vc|.
    # Only the dephasing acts, and |rho_vc| must fall as exp(-t / T2) exactly.
    t2 = 50.0
    time = np.arange(0.0, 200.0, 0.05)
    rho0 = np.full((model.n_k, 2, 2), 0.5, dtype=complex)
    _, rho = _evolve(model, time, np.zeros_like(time), t2, rho0=rho0)
    elapsed = time[-1] - time[0]
    np.testing.assert_allclose(np.abs(rho[:, 0, 1]), 0.5 * np.exp(-elapsed / t2), rtol=1e-9)
    np.testing.assert_allclose(rho[:, 0, 0].real, 0.5, rtol=1e-12)


def test_time_step_converges(model, strong_pulse):
    time, a = strong_pulse
    coarse = tb.propagate_dephased(model, time[::2], a[::2], 5.0)
    fine = tb.propagate_dephased(model, time, a, 5.0)[::2]
    np.testing.assert_allclose(coarse, fine, rtol=0, atol=1e-3 * np.abs(fine).max())


def test_invalid_t2_raises(model, strong_pulse):
    time, a = strong_pulse
    with pytest.raises(ValueError, match="t2"):
        tb.propagate_dephased(model, time, a, 0.0)
