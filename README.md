# Parabolic Two-Band Model: Optical Response Analysis

[![CI](https://github.com/lawanmuk/hbn-parabolic-two-band-model/actions/workflows/ci.yml/badge.svg)](https://github.com/lawanmuk/hbn-parabolic-two-band-model/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Analysis tools for the **parabolic two-band model** of a gapped semiconductor (such as monolayer hBN) driven by pump and probe laser pulses. The `twoband` package turns simulated current and field traces into optical conductivity, dielectric functions and transient absorption spectra (TAS), used to study the Dynamical Franz-Keldysh Effect (DFKE).

## Overview

The time propagation of the two-band model runs in a separate Fortran code. It writes the vector potential A(t), current J(t) and electric field E(t) for three kinds of runs: probe only, pump only, and pump-probe at a series of delays. This repository contains:

- **`twoband`**, a tested Python package with the analysis steps (damped Fourier transform, conductivity, reference subtraction, delay splitting, TAS)
- **Jupyter notebooks** that apply the package to simulation output and produce the figures

## Theoretical background

Parabolic valence and conduction bands (atomic units, $\hbar = e = m_e = 1$):

$$\epsilon_v(k) = -\frac{k^2}{2 m_v}, \qquad \epsilon_c(k) = \epsilon_g + \frac{k^2}{2 m_c}$$

The optical conductivity follows from the damped Fourier transforms of current and field:

$$\sigma(\omega) = \frac{J(\omega)}{E(\omega)}, \qquad F(\omega) = \int F(t)\, e^{i\omega t - \gamma t}\, dt$$

In a pump-probe run, the probe response is isolated by subtracting the pump-only run,

$$J_{\text{probe}}(t) = J_{\text{pump-probe}}(t) - J_{\text{pump}}(t),$$

and the transient absorption at delay $T$ is

$$\Delta\sigma(\omega, T) = \sigma(\omega, T) - \sigma(\omega).$$

The dielectric function is related by $\epsilon(\omega) = 1 + 4\pi i\,\sigma(\omega)/\omega$. Analytic reference results for $\mathrm{Im}\,\epsilon(\omega)$ are included for 2D (a step at the gap) and 3D (a square-root edge).

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/lawanmuk/hbn-parabolic-two-band-model.git
cd hbn-parabolic-two-band-model
pip install -e .
```

For development (tests and linting): `pip install -e ".[dev]"`
For the notebooks: `pip install -e ".[notebooks]"`

## Usage

```python
import numpy as np
import twoband as tb

# Load simulation output: columns 0 = time, 1 = A(t), 4 = J(t), 7 = E(t)
probe = np.loadtxt("data/Act_jt_3.6d8_probe.out")
pump = np.loadtxt("data/Act_jt_3.6d10_pump.out")
scan = np.loadtxt("data/Act_jt_3d10_3d8_pp.out")

energy = np.linspace(2, 20, 500) / tb.HARTREE_TO_EV
gamma = 0.25 / tb.HARTREE_TO_EV

# Equilibrium conductivity
sigma_eq = tb.optical_conductivity(probe[:, 0], probe[:, 4], probe[:, 7], energy, gamma)

# Transient absorption over all delays
blocks = tb.split_delays(scan, 50001)
j_probe = tb.subtract_reference(blocks[:, :, 4], pump[:, 4])
e_probe = tb.subtract_reference(blocks[:, :, 7], pump[:, 7])
sigma_t = tb.optical_conductivity(blocks[0, :, 0], j_probe, e_probe, energy, gamma)
tas = tb.transient_absorption(sigma_t, sigma_eq)   # shape (n_delays, n_energy)
```

## Project structure

```
src/twoband/
    constants.py      unit conversions (Hartree to eV, a.u. to fs)
    bands.py          parabolic dispersions and reduced mass
    fourier.py        damped Fourier transform (1D or many traces at once)
    conductivity.py   sigma(w), delay splitting, reference subtraction, TAS
    dielectric.py     Im[eps(w)] for 2D and 3D parabolic bands
tests/                pytest suite
*.ipynb               analysis notebooks
```

| Notebook | Purpose |
|---|---|
| `Analysis_of_TBModel.ipynb` | Band structure, equilibrium and transient conductivity, TAS map |
| `FT_analysis_current.ipynb` | Conductivity and dielectric function from laser and current traces |
| `Fourier_Analysis.ipynb` | sin^4 pulse construction and spectrum (direct FT vs FFT) |
| `Im_3D_DEF.ipynb` | Analytic Im[eps(w)] for 3D parabolic bands |

## Data

Simulation output is not included in the repository because of its size. The notebooks expect these files in a `data/` folder:

| File | Content |
|---|---|
| `Act_jt_3.6d8_probe.out` | probe only (equilibrium reference) |
| `Act_jt_3.6d10_pump.out` | pump only |
| `Act_jt_3d10_3d8_pp.out` | pump-probe, one block of 50001 rows per delay |

Columns: 0 = time (a.u.), 1 = A(t), 4 = J(t), 7 = E(t).

The package itself does not need these files; the tests use synthetic signals.

## Testing

```bash
pytest            # run the test suite
ruff check .      # lint
ruff format .     # format
```

CI runs linting, the tests on Python 3.10 to 3.13, and a dependency audit on every push and pull request.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Reference

S. A. Sato et al., "Nonlinear optical responses of solids: first-principles simulations and the dynamical Franz-Keldysh effect", *Applied Sciences* 8(10), 1777 (2018). https://www.mdpi.com/2076-3417/8/10/1777

## License

MIT, see [LICENSE](LICENSE).
