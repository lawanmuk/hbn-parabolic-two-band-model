# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-10-15

### Added
- Two-dimensional k grid in the simulator: `TwoBandModel` takes `ky_max` and `n_ky`, with the field and the dipole along x. The default `n_ky = 1` keeps the one-dimensional model unchanged. `TwoBandModel.grid()` gives the flattened points and the area element, and the diamagnetic coefficient uses the same grid.
- Tests for the 2D grid: a complex-valued reference over the full (kx, ky) grid, weak-field linearity, and the static sum rule.
- Check of the 2D simulator against the analytic `im_eps_2d`: the weak-probe Re[sigma] agrees to 3 % from 6 to 12 eV, with the Lorentzian broadening equal to the damping.

### Changed
- 2D propagation is about 1.8x faster on a 101 x 101 grid (9.2 s to 5.2 s for 16000 steps). The field and dipole are along x, so ky and -ky points are identical: only ky >= 0 is propagated, with each ky > 0 point counted twice. Results are unchanged to 1e-15 relative.
- Two-dimensional k grid in the simulator: `TwoBandModel` takes `ky_max` and `n_ky`, with the field and the dipole along x. The default `n_ky = 1` keeps the one-dimensional model unchanged. `TwoBandModel.grid()` gives the flattened points and the area element, and the diamagnetic coefficient uses the same grid.
- Tests for the 2D grid: a complex-valued reference over the full (kx, ky) grid, weak-field linearity, and the static sum rule.

## [0.2.1] - 2026-10-01

### Added
- Pinned `requirements.txt` (generated with `uv pip compile --universal`) for reproducible installs on Python 3.10 and newer.
- Regression test comparing the fast propagator with a straightforward reference implementation.

### Changed
- Propagator about 2.4x faster: the global phase, which cancels in every observable, is no longer computed, amplitudes are updated in place as real arrays, and the generator runs all 15 simulations in one pass. Results are unchanged to machine precision.

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
