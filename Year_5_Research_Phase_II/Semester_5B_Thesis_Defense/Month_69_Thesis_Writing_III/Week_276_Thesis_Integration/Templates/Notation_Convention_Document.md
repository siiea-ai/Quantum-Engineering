# Notation Convention Document

## Thesis-Wide Symbol and Convention Reference

---

## Purpose

This document establishes the notation conventions used throughout the thesis. All symbols, operators, and mathematical conventions should conform to this guide to ensure consistency and clarity.

---

## Part I: General Conventions

### 1.1 Mathematical Typography

| Type | Convention | Examples |
|------|-----------|----------|
| Scalars | Italic lowercase or Greek | $a$, $x$, $\alpha$, $\omega$ |
| Vectors | Bold italic | $\mathbf{r}$, $\mathbf{k}$, $\boldsymbol{\mu}$ |
| Matrices | Bold capital or calligraphic | $\mathbf{M}$, $\mathcal{M}$ |
| Operators | Hat notation | $\hat{H}$, $\hat{a}$, $\hat{\sigma}$ |
| Sets | Blackboard bold | $\mathbb{R}$, $\mathbb{C}$, $\mathbb{Z}$ |
| Special functions | Roman upright | $\sin$, $\exp$, $\mathrm{Tr}$ |

### 1.2 Bracket Notation

| Notation | Meaning | Usage |
|----------|---------|-------|
| $\langle \cdot \rangle$ | Expectation value | $\langle A \rangle = \mathrm{Tr}[\rho A]$ |
| $\| \psi \rangle$ | Ket (state vector) | Quantum state |
| $\langle \psi \|$ | Bra (dual vector) | Dual of quantum state |
| $\langle \phi \| \psi \rangle$ | Inner product | Overlap of states |
| $\| \psi \rangle \langle \phi \|$ | Outer product | Projection operator |
| $[A, B]$ | Commutator | $[A, B] = AB - BA$ |
| $\{A, B\}$ | Anticommutator | $\{A, B\} = AB + BA$ |

### 1.3 Derivative Notation

| Notation | Meaning | Usage |
|----------|---------|-------|
| $\frac{d}{dt}$ | Total time derivative | Closed system dynamics |
| $\partial_t$ | Partial time derivative | When other variables present |
| $\dot{x}$ | Newton's notation | $\dot{x} = dx/dt$ |
| $\nabla$ | Gradient operator | Spatial derivatives |
| $\partial_x$ | Partial derivative w.r.t. $x$ | Shorthand for $\partial/\partial x$ |

---

## Part II: Physical Constants

| Symbol | Name | Value | Notes |
|--------|------|-------|-------|
| $\hbar$ | Reduced Planck constant | $1.055 \times 10^{-34}$ J·s | Often set to 1 |
| $k_B$ | Boltzmann constant | $1.381 \times 10^{-23}$ J/K | Often set to 1 |
| $e$ | Elementary charge | $1.602 \times 10^{-19}$ C | |
| $c$ | Speed of light | $2.998 \times 10^8$ m/s | |
| $\mu_0$ | Vacuum permeability | $4\pi \times 10^{-7}$ H/m | |
| $\epsilon_0$ | Vacuum permittivity | $8.854 \times 10^{-12}$ F/m | |

---

## Part III: Quantum Mechanics Symbols

### 3.1 States and Operators

| Symbol | Meaning | Definition Location |
|--------|---------|---------------------|
| $\|\psi\rangle$ | Pure quantum state | Ch. 2, Sec. 2.1 |
| $\rho$ | Density matrix | Ch. 2, Sec. 2.2 |
| $\hat{H}$ | Hamiltonian operator | Ch. 2, Sec. 2.1 |
| $\hat{U}$ | Unitary evolution operator | Ch. 2, Sec. 2.3 |
| $\hat{a}$, $\hat{a}^\dagger$ | Annihilation, creation operators | Ch. 2, Sec. 2.4 |
| $\hat{\sigma}_x, \hat{\sigma}_y, \hat{\sigma}_z$ | Pauli matrices | Ch. 2, Sec. 2.1 |
| $\hat{\sigma}_+, \hat{\sigma}_-$ | Raising/lowering operators | Ch. 2, Sec. 2.1 |
| $\|0\rangle, \|1\rangle$ | Computational basis states | Ch. 2, Sec. 2.1 |
| $\|+\rangle, \|-\rangle$ | Superposition states | Ch. 2, Sec. 2.1 |

### 3.2 Hamiltonians

| Symbol | Meaning | Context |
|--------|---------|---------|
| $H_0$ | Bare system Hamiltonian | Undriven system |
| $H_S$ | System Hamiltonian | Open system context |
| $H_E$ | Environment Hamiltonian | Open system context |
| $H_{SE}$ | System-environment interaction | Open system context |
| $H_c(t)$ | Control Hamiltonian | Driven system |
| $H_{\text{eff}}$ | Effective Hamiltonian | Approximated dynamics |

### 3.3 Open Quantum Systems

| Symbol | Meaning | Definition Location |
|--------|---------|---------------------|
| $\mathcal{L}$ | Lindblad superoperator | Ch. 2, Sec. 2.5 |
| $L_k$ | Lindblad (jump) operators | Ch. 2, Sec. 2.5 |
| $\gamma_k$ | Decay rates | Ch. 2, Sec. 2.5 |
| $\kappa$ | Cavity decay rate | Ch. 3, Sec. 3.2 |
| $\Gamma$ | Total decoherence rate | Ch. 3, Sec. 3.3 |
| $S(\omega)$ | Noise power spectral density | Ch. 3, Sec. 3.4 |
| $F(\omega)$ | Filter function | Ch. 3, Sec. 3.5 |

---

## Part IV: Coherence and Relaxation

### 4.1 Time Scales

| Symbol | Meaning | Typical Units |
|--------|---------|---------------|
| $T_1$ | Longitudinal relaxation time | μs to ms |
| $T_2$ | Transverse coherence time | μs to ms |
| $T_2^*$ | Inhomogeneous dephasing time | μs |
| $T_\phi$ | Pure dephasing time | μs to ms |
| $\tau_c$ | Noise correlation time | ns to μs |
| $\tau$ | Generic time interval | Context-dependent |
| $t_g$ | Gate time | ns |

**Relationships:**
$$\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$$

$$\frac{1}{T_2^*} = \frac{1}{T_2} + \frac{1}{T_{\text{inhom}}}$$

### 4.2 Coherence Measures

| Symbol | Meaning | Range |
|--------|---------|-------|
| $C(t)$ | Coherence function | [0, 1] |
| $\mathcal{F}$ | Fidelity | [0, 1] |
| $P(t)$ | Population | [0, 1] |
| $\chi$ | Process matrix | — |
| $W(t)$ | Decay envelope | [0, 1] |

---

## Part V: Control and Pulses

### 5.1 Pulse Parameters

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\Omega$ | Rabi frequency / pulse amplitude | rad/s or MHz |
| $\Omega_0$ | Nominal Rabi frequency | rad/s or MHz |
| $\phi$ | Pulse phase | rad |
| $\tau_p$ | Pulse duration | ns |
| $t_p$ | Pulse time (center) | ns |
| $N$ | Number of pulses | — |

### 5.2 Pulse Sequences

| Symbol | Meaning | Definition |
|--------|---------|------------|
| $\Pi(t)$ | Control protocol | Time-dependent pulse function |
| $\tau_{\text{free}}$ | Free evolution period | Time between pulses |
| $\mathbf{n}$ | Rotation axis | Unit vector on Bloch sphere |
| $\theta$ | Rotation angle | rad |

### 5.3 Common Sequences

| Name | Description | Notation |
|------|-------------|----------|
| Hahn Echo | $\pi$ pulse at midpoint | HE |
| CPMG | Periodic $\pi$ pulses | CPMG-$N$ |
| XY-4 | Phase-alternating sequence | XY4 |
| UDD | Uhrig dynamical decoupling | UDD-$N$ |

---

## Part VI: Experimental Parameters

### 6.1 System Parameters

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| $\omega_q$ | Qubit transition frequency | 4-8 GHz |
| $\omega_r$ | Resonator frequency | 6-10 GHz |
| $g$ | Coupling strength | 50-200 MHz |
| $E_C$ | Charging energy | 200-400 MHz |
| $E_J$ | Josephson energy | 10-50 GHz |
| $\chi$ | Dispersive shift | 1-10 MHz |
| $\alpha$ | Anharmonicity | 200-400 MHz |

### 6.2 Environmental Parameters

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| $T$ | Temperature | 10-50 mK |
| $n_{\text{th}}$ | Thermal occupation | $<$ 0.01 |
| $Q$ | Quality factor | $10^3 - 10^6$ |
| $\eta$ | Detection efficiency | 0.3-0.9 |

---

## Part VII: Statistical Notation

| Symbol | Meaning |
|--------|---------|
| $\bar{x}$ | Sample mean |
| $\sigma$ | Standard deviation |
| $\sigma_x$ | Standard error of $x$ |
| $\chi^2$ | Chi-squared statistic |
| $N$ | Number of samples (when not pulse count) |
| $p$ | p-value |
| $r$ | Correlation coefficient |
| $\mathcal{N}(\mu, \sigma^2)$ | Normal distribution |

---

## Part VIII: Subscripts and Superscripts

### 8.1 Common Subscripts

| Subscript | Meaning |
|-----------|---------|
| $_0$ | Initial / unperturbed / reference |
| $_1$, $_2$ | First, second (modes, states, etc.) |
| $_q$ | Qubit |
| $_r$ | Resonator |
| $_c$ | Control / cavity |
| $_{\text{eff}}$ | Effective |
| $_{\text{opt}}$ | Optimal |
| $_{\text{exp}}$ | Experimental |
| $_{\text{th}}$ | Theoretical / thermal |

### 8.2 Common Superscripts

| Superscript | Meaning |
|-------------|---------|
| $^\dagger$ | Hermitian conjugate |
| $^*$ | Complex conjugate |
| $^{-1}$ | Inverse |
| $^T$ | Transpose |
| $^{\text{R1}}$, $^{\text{R2}}$ | Research Project 1, 2 |

---

## Part IX: Chapter-Specific Notation

### Research Project 1 Specific

| Symbol | Meaning | Defined in |
|--------|---------|------------|
| | | Sec. ___ |
| | | Sec. ___ |
| | | Sec. ___ |

### Research Project 2 Specific

| Symbol | Meaning | Defined in |
|--------|---------|------------|
| | | Sec. ___ |
| | | Sec. ___ |
| | | Sec. ___ |

---

## Part X: Notation Changes Log

| Symbol | Previous Use | Thesis Standard | Chapters Affected |
|--------|--------------|-----------------|-------------------|
| | | | |
| | | | |
| | | | |

---

## LaTeX Macros

For consistency, define macros in preamble:

```latex
% States
\newcommand{\ket}[1]{\left| #1 \right\rangle}
\newcommand{\bra}[1]{\left\langle #1 \right|}
\newcommand{\braket}[2]{\left\langle #1 | #2 \right\rangle}
\newcommand{\ketbra}[2]{\left| #1 \right\rangle\left\langle #2 \right|}
\newcommand{\expval}[1]{\left\langle #1 \right\rangle}

% Operators
\newcommand{\op}[1]{\hat{#1}}
\newcommand{\comm}[2]{\left[ #1, #2 \right]}
\newcommand{\acomm}[2]{\left\{ #1, #2 \right\}}

% Common symbols
\newcommand{\hbar}{\hslash}
\newcommand{\Tr}{\mathrm{Tr}}
\newcommand{\Tone}{T_1}
\newcommand{\Ttwo}{T_2}
\newcommand{\Tphi}{T_\phi}

% Units (with siunitx package)
\newcommand{\us}{\si{\micro\second}}
\newcommand{\MHz}{\si{\mega\hertz}}
\newcommand{\GHz}{\si{\giga\hertz}}
```

---

*Notation Convention Document Template | Week 276 | Thesis Writing III*
