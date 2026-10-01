"""Analysis tools for the parabolic two-band model.

All quantities use atomic units (e = m_e = hbar = 1) unless a name says otherwise.
"""

from twoband.bands import conduction_band, reduced_mass, valence_band
from twoband.conductivity import (
    optical_conductivity,
    split_delays,
    subtract_reference,
    transient_absorption,
)
from twoband.constants import AU_TO_FS, C_AU, HARTREE_TO_EV
from twoband.dataset import DatasetConfig, generate_dataset, load_or_generate
from twoband.dielectric import im_eps_1d, im_eps_2d, im_eps_3d, re_sigma_1d_broadened
from twoband.fourier import damped_ft
from twoband.pulses import (
    field_from_vector_potential,
    intensity_to_field,
    sin4_field,
    vector_potential,
)
from twoband.simulate import TwoBandModel, diamagnetic_coefficient, propagate

__all__ = [
    "AU_TO_FS",
    "C_AU",
    "HARTREE_TO_EV",
    "DatasetConfig",
    "TwoBandModel",
    "conduction_band",
    "damped_ft",
    "diamagnetic_coefficient",
    "field_from_vector_potential",
    "generate_dataset",
    "im_eps_1d",
    "im_eps_2d",
    "im_eps_3d",
    "intensity_to_field",
    "load_or_generate",
    "optical_conductivity",
    "propagate",
    "re_sigma_1d_broadened",
    "reduced_mass",
    "sin4_field",
    "split_delays",
    "subtract_reference",
    "transient_absorption",
    "valence_band",
    "vector_potential",
]

__version__ = "0.2.1"
