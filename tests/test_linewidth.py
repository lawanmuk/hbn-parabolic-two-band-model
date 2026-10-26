import numpy as np
import pytest

import twoband as tb

GAP = 4.45 / tb.HARTREE_TO_EV


def _half_width(energy, sigma):
    """Half width at half maximum of a single peak, by linear interpolation on both sides."""
    i = int(np.argmax(sigma))
    half = sigma[i] / 2
    left = int(np.where(sigma[:i] <= half)[0][-1])
    right = i + int(np.where(sigma[i:] <= half)[0][0])

    def cross(i0, i1):
        return energy[i0] + (half - sigma[i0]) * (energy[i1] - energy[i0]) / (sigma[i1] - sigma[i0])

    return 0.5 * (cross(right - 1, right) - cross(left, left + 1))


@pytest.mark.parametrize("t2", [100.0, 200.0])
def test_homogeneous_linewidth_is_one_over_t2(t2):
    # A tiny k range makes every transition sit at the gap, so the width is homogeneous. The
    # dephased current decays as exp(-t / T2) by itself. Fourier transforming without any
    # damping (gamma = 0) then gives a Lorentzian with half width 1 / T2.
    model = tb.TwoBandModel(e_gap=GAP, m_c=0.3, m_v=0.5, p_vc=0.96, k_max=0.01, n_k=21)
    time = np.arange(0.0, 10 * t2 + 200.0, 0.05)
    field = tb.sin4_field(time, amplitude=1e-5, omega=GAP, duration=40.0, center=100.0)
    a = tb.vector_potential(time, field)
    current = tb.propagate_dephased(model, time, a, t2)
    energy = GAP + np.linspace(-0.06, 0.06, 1201)
    sigma = tb.optical_conductivity(
        time, current, tb.field_from_vector_potential(time, a), energy, 0.0
    ).real
    assert _half_width(energy, sigma) == pytest.approx(1.0 / t2, rel=0.02)
