# Parabolic Two-Band Model Calculation, Simulation and Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

This repository contains numerical tools and Python simulations for a **parabolic two-band model** in condensed matter physics. The code simulates quantum dynamics, non-equilibrium carrier distributions, and time-dependent current responses in semiconductor systems subjected to external optical fields (e.g., pump-probe laser pulses).

The framework uses an effective mass approximation to model electron-hole excitations across a direct band gap, providing insights into time-resolved optical observables such as transient absorption and probe-induced currents to evaluate the optical conductivity.

---

## 🔬 Theoretical Background

The system is modeled using a simplified two-band Hamiltonian consisting of a parabolic valence band ($v$) and conduction band ($c$):

$$E_c(k) = E_g + \frac{\hbar^2 k^2}{2 m_c}$$

$$E_v(k) = -\frac{\hbar^2 k^2}{2 m_v}$$

Where:
* $E_g$ is the direct energy band gap.
* $m_c$ and $m_v$ are the effective masses of electrons and holes, respectively.
* $k$ is the crystal momentum in $k$-space.

The light-matter interaction is introduced via minimum coupling $\mathbf{p} \rightarrow \mathbf{p} - e\mathbf{A}(t)$ or dipole coupling $\mathbf{r} \cdot \mathbf{E}(t)$, where the time-dependent vector potential $\mathbf{A}(t)$ represents pump and probe laser pulses:

$$\mathbf{A}(t) = \mathbf{A}_{\text{pump}}(t) + \mathbf{A}_{\text{probe}}(t - \tau)$$

The time evolution of the system is evaluated numerically to calculate the time-dependent current density $\mathbf{J}(t)$ and microscopic polarization.

---

## 🚀 Key Features

* **$k$-Space Band Dispersions:** Calculation of 1D/2D parabolic band structures and density of states (DOS).
* **Time-Dependent Field Coupling:** Custom pulse configurations (envelope, frequency, polarization, and pump-probe delay $\tau$).
* **Quantum Dynamics Integrator:** Solves the time-dependent Schrödinger equation (TDSE) or Semiconductor Bloch Equations (SBE).
* **Observables Output:** Computes time-resolved photocurrents, interband/intraband polarization, and transient absorption spectra.

---

## 🛠️ Installation & Prerequisites

### Dependencies

Ensure you have the following packages installed:

```bash
pip install numpy scipy matplotlib 
