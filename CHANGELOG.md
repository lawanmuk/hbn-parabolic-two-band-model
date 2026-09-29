# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).



## [Unreleased]

## [0.2.0] - 2026-09-29

### Added
- Real-time simulator of the 1D parabolic two-band model (`twoband.simulate`) with f-sum-rule diamagnetic correction.
- sin^4 pulses and intensity conversion (`twoband.pulses`).
- Dataset generator with metadata and the `twoband-generate` command (`twoband.dataset`); notebooks generate data on first run.
- `im_eps_1d` and `re_sigma_1d_broadened` analytic references; tests validating the simulator against them.

### Changed
- Notebooks use generated data and plot Re[sigma] with the physical sign.

## [0.1.0] - 2026-09-29

### Added
- `twoband` Python package with a `pyproject.toml` build definition.
- Damped Fourier transform that handles one or many traces and works in energy blocks to limit memory use.
- Optical conductivity, delay splitting, pump reference subtraction and transient absorption functions with input checks.
- Im[eps(w)] for 2D (Lorentzian-broadened k integral) and 3D parabolic bands.
- pytest suite, ruff linting, GitHub Actions CI on Python 3.10 to 3.13, pip-audit and Dependabot.


### Fixed
- Band dispersion multiplied by the effective mass instead of dividing by it.
- 3D dielectric function used (2m)^3/2 instead of (2m)^(3/2).
- Fourier transform modified the input signal in place.

### Added
- `equilibrium_spectrum_tbm.ipynb`: equilibrium, single-delay and delay-scan TAS analysis built on the `twoband` package.

### Fixed
- Single-delay Fourier transform used the equilibrium time and energy arrays instead of the pump-probe ones.
- Equilibrium and transient spectra were computed on different energy grids before being subtracted.

### Changed
- Replaced the five original notebooks with four notebooks built on the `twoband` package: `band_structure_and_dielectric`, `pulse_spectrum`, `equilibrium_spectrum_tbm` and `conductivity_from_current`.
- The single-delay analysis detects how many delays the pump-probe file holds.

### Removed
- `Copy_of_Analysis_of_TBModel.ipynb` and the anonfiles.com downloads, which no longer work.
