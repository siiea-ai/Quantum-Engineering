# Week 146: Measurement and Dynamics — Review Guide

## Introduction

This week covers the operational heart of quantum mechanics: how we extract information from quantum systems through measurement, and how quantum states evolve in time. These topics form the foundation for understanding all quantum phenomena and are extensively tested on PhD qualifying exams.

---

## 1. The Postulates of Quantum Mechanics

### Postulate 1: State Space
The state of a quantum system is completely described by a vector $$|\psi\rangle$$ in a Hilbert space $$\mathcal{H}$$.

**Key Points:**
- States are normalized: $$\langle\psi|\psi\rangle = 1$$
- Physical states are rays (equivalent up to phase)
- Superposition principle follows from linearity

### Postulate 2: Observables
Every physical observable $$A$$ is represented by a Hermitian operator $$\hat{A}$$ on $$\mathcal{H}$$.

**Key Points:**
- Hermiticity ensures real eigenvalues
- Eigenstates form a complete basis
- Not all operators are observables

### Postulate 3: Measurement (Born Rule)
Measurement of observable $$\hat{A}$$ on state $$|\psi\rangle$$ yields eigenvalue $$a_n$$ with probability:

$$\boxed{P(a_n) = |\langle a_n | \psi \rangle|^2}$$

For degenerate eigenvalue with eigenspace $$\mathcal{E}_n$$:
$$P(a_n) = \sum_{k \in \mathcal{E}_n} |\langle a_n^{(k)} | \psi \rangle|^2 = \langle\psi|\hat{P}_n|\psi\rangle$$

### Postulate 4: State Collapse
After measuring eigenvalue $$a_n$$, the state collapses to:

$$|\psi\rangle \to \frac{\hat{P}_n|\psi\rangle}{\sqrt{\langle\psi|\hat{P}_n|\psi\rangle}}$$

where $$\hat{P}_n = |a_n\rangle\langle a_n|$$ (non-degenerate case).

### Postulate 5: Time Evolution
The state evolves according to the Schrödinger equation:

$$\boxed{i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle}$$

---

## 2. Measurement Theory

### 2.1 Expectation Values

The expectation value of observable $$\hat{A}$$ in state $$|\psi\rangle$$:

$$\langle \hat{A} \rangle = \langle\psi|\hat{A}|\psi\rangle = \sum_n a_n P(a_n)$$

For a general operator (not necessarily Hermitian):
$$\langle \hat{A} \rangle = \text{Tr}(\hat{\rho}\hat{A})$$

where $$\hat{\rho} = |\psi\rangle\langle\psi|$$ is the density operator.

### 2.2 Variance and Uncertainty

$$(\Delta A)^2 = \langle \hat{A}^2 \rangle - \langle \hat{A} \rangle^2 = \langle (\hat{A} - \langle\hat{A}\rangle)^2 \rangle$$

The variance vanishes if and only if $$|\psi\rangle$$ is an eigenstate of $$\hat{A}$$.

### 2.3 Repeated Measurements

If we immediately remeasure the same observable:
- We get the same result with probability 1
- The state is already the eigenstate

For sequential measurements of different observables:
- Order matters if $$[\hat{A}, \hat{B}] \neq 0$$
- First measurement destroys information about the other

### 2.4 Compatible Observables

Observables $$\hat{A}$$ and $$\hat{B}$$ are **compatible** if $$[\hat{A}, \hat{B}] = 0$$.

**Consequences:**
- They share a complete set of eigenstates
- Can be measured simultaneously with arbitrary precision
- Measuring one doesn't disturb the other

**Incompatible observables** ($$[\hat{A}, \hat{B}] \neq 0$$):
- No simultaneous eigenstates (in general)
- Subject to uncertainty principle
- Measurement of one disturbs the other

---

## 3. Continuous Spectra

### 3.1 Position Measurement

For the position operator $$\hat{x}$$:
$$P(x \in [a,b]) = \int_a^b |\psi(x)|^2 \, dx$$

The probability density is $$\rho(x) = |\psi(x)|^2$$.

After measuring position $$x_0$$:
$$\psi(x) \to \delta(x - x_0)$$

### 3.2 Momentum Measurement

For momentum:
$$P(p \in [p_1, p_2]) = \int_{p_1}^{p_2} |\tilde{\psi}(p)|^2 \, dp$$

where $$\tilde{\psi}(p) = \langle p | \psi \rangle$$ is the momentum-space wavefunction.

### 3.3 Generalized Measurements

For an observable with both discrete and continuous spectra:
$$\sum_n P(a_n) + \int P(a) \, da = 1$$

---

## 4. The Schrödinger Equation

### 4.1 Time-Dependent Form

$$i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle$$

In position representation:
$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi + V(\mathbf{r})\psi$$

### 4.2 Time-Independent Form

For a time-independent Hamiltonian, separation of variables gives:
$$\hat{H}|\phi_n\rangle = E_n|\phi_n\rangle$$

**Stationary states:**
$$|\psi_n(t)\rangle = e^{-iE_nt/\hbar}|\phi_n\rangle$$

Key property: probability density $$|\psi_n(\mathbf{r},t)|^2$$ is time-independent.

### 4.3 General Solution

For time-independent $$\hat{H}$$:
$$|\psi(t)\rangle = \sum_n c_n e^{-iE_nt/\hbar}|\phi_n\rangle$$

where $$c_n = \langle\phi_n|\psi(0)\rangle$$.

### 4.4 Properties of Solutions

1. **Normalization preserved:** $$\langle\psi(t)|\psi(t)\rangle = \langle\psi(0)|\psi(0)\rangle$$

2. **Probability current conservation:**
   $$\frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{j} = 0$$
   where $$\mathbf{j} = \frac{\hbar}{2mi}(\psi^*\nabla\psi - \psi\nabla\psi^*)$$

3. **Superposition principle:** If $$\psi_1$$ and $$\psi_2$$ are solutions, so is $$c_1\psi_1 + c_2\psi_2$$

---

## 5. The Time-Evolution Operator

### 5.1 Definition

The **time-evolution operator** $$\hat{U}(t,t_0)$$ satisfies:
$$|\psi(t)\rangle = \hat{U}(t,t_0)|\psi(t_0)\rangle$$

### 5.2 Properties

1. **Unitarity:** $$\hat{U}^\dagger(t,t_0)\hat{U}(t,t_0) = \hat{1}$$

2. **Composition:** $$\hat{U}(t_2,t_0) = \hat{U}(t_2,t_1)\hat{U}(t_1,t_0)$$

3. **Initial condition:** $$\hat{U}(t_0,t_0) = \hat{1}$$

4. **Inverse:** $$\hat{U}^{-1}(t,t_0) = \hat{U}(t_0,t)$$

### 5.3 For Time-Independent Hamiltonians

$$\boxed{\hat{U}(t) = e^{-i\hat{H}t/\hbar}}$$

Using spectral decomposition:
$$\hat{U}(t) = \sum_n e^{-iE_nt/\hbar}|\phi_n\rangle\langle\phi_n|$$

### 5.4 For Time-Dependent Hamiltonians

$$\hat{U}(t,t_0) = \mathcal{T}\exp\left(-\frac{i}{\hbar}\int_{t_0}^t \hat{H}(t') \, dt'\right)$$

where $$\mathcal{T}$$ is the time-ordering operator.

---

## 6. The Propagator

### 6.1 Definition

The **propagator** (or Green's function) is:
$$K(x,t;x',t') = \langle x | \hat{U}(t,t') | x' \rangle$$

It propagates the wavefunction:
$$\psi(x,t) = \int K(x,t;x',t')\psi(x',t') \, dx'$$

### 6.2 Properties

1. **Initial condition:** $$K(x,t;x',t) = \delta(x-x')$$

2. **Composition:** $$K(x,t;x',t') = \int K(x,t;x'',t'')K(x'',t'';x',t') \, dx''$$

3. **Satisfies Schrödinger equation:**
   $$i\hbar\frac{\partial K}{\partial t} = \hat{H}_x K$$

### 6.3 Free Particle Propagator

For $$\hat{H} = \frac{\hat{p}^2}{2m}$$:

$$\boxed{K_0(x,t;x',0) = \sqrt{\frac{m}{2\pi i\hbar t}}\exp\left[\frac{im(x-x')^2}{2\hbar t}\right]}$$

**Derivation sketch:**
1. Insert completeness in momentum: $$K = \int \langle x|p\rangle e^{-ip^2t/(2m\hbar)} \langle p|x'\rangle \, dp$$
2. Substitute $$\langle x|p\rangle = e^{ipx/\hbar}/\sqrt{2\pi\hbar}$$
3. Complete the Gaussian integral

### 6.4 Harmonic Oscillator Propagator

$$K(x,t;x',0) = \sqrt{\frac{m\omega}{2\pi i\hbar\sin(\omega t)}}\exp\left[\frac{im\omega}{2\hbar\sin(\omega t)}((x^2+x'^2)\cos(\omega t) - 2xx')\right]$$

---

## 7. Pictures of Quantum Mechanics

### 7.1 Schrödinger Picture

- States evolve: $$|\psi_S(t)\rangle = \hat{U}(t)|\psi_S(0)\rangle$$
- Operators fixed: $$\hat{A}_S = \hat{A}$$
- Most common formulation

### 7.2 Heisenberg Picture

- States fixed: $$|\psi_H\rangle = |\psi_S(0)\rangle$$
- Operators evolve: $$\hat{A}_H(t) = \hat{U}^\dagger(t)\hat{A}_S\hat{U}(t)$$

**Heisenberg equation of motion:**
$$\boxed{\frac{d\hat{A}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{A}_H] + \left(\frac{\partial \hat{A}}{\partial t}\right)_H}$$

### 7.3 Interaction Picture

For $$\hat{H} = \hat{H}_0 + \hat{V}(t)$$:
- States evolve with $$\hat{V}$$: $$|\psi_I(t)\rangle = e^{i\hat{H}_0t/\hbar}|\psi_S(t)\rangle$$
- Operators evolve with $$\hat{H}_0$$: $$\hat{A}_I(t) = e^{i\hat{H}_0t/\hbar}\hat{A}_Se^{-i\hat{H}_0t/\hbar}$$

Useful for perturbation theory and time-dependent problems.

### 7.4 Equivalence

All pictures give the same physics:
$$\langle\psi_S(t)|\hat{A}_S|\psi_S(t)\rangle = \langle\psi_H|\hat{A}_H(t)|\psi_H\rangle = \langle\psi_I(t)|\hat{A}_I(t)|\psi_I(t)\rangle$$

---

## 8. Ehrenfest's Theorem

### 8.1 Statement

For any observable $$\hat{A}$$:
$$\boxed{\frac{d}{dt}\langle \hat{A} \rangle = \frac{i}{\hbar}\langle [\hat{H}, \hat{A}] \rangle + \left\langle \frac{\partial \hat{A}}{\partial t} \right\rangle}$$

### 8.2 Derivation

$$\frac{d}{dt}\langle\hat{A}\rangle = \frac{d}{dt}\langle\psi|\hat{A}|\psi\rangle$$
$$= \left\langle\frac{d\psi}{dt}\right|\hat{A}|\psi\rangle + \langle\psi|\frac{\partial\hat{A}}{\partial t}|\psi\rangle + \langle\psi|\hat{A}\left|\frac{d\psi}{dt}\right\rangle$$

Using $$i\hbar\frac{d|\psi\rangle}{dt} = \hat{H}|\psi\rangle$$:
$$= \frac{1}{i\hbar}\langle\psi|\hat{H}\hat{A}|\psi\rangle + \frac{1}{-i\hbar}\langle\psi|\hat{A}\hat{H}|\psi\rangle + \left\langle\frac{\partial\hat{A}}{\partial t}\right\rangle$$
$$= \frac{i}{\hbar}\langle[\hat{H},\hat{A}]\rangle + \left\langle\frac{\partial\hat{A}}{\partial t}\right\rangle$$

### 8.3 Position and Momentum

For $$\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})$$:

$$\frac{d\langle\hat{x}\rangle}{dt} = \frac{\langle\hat{p}\rangle}{m}$$

$$\frac{d\langle\hat{p}\rangle}{dt} = -\left\langle\frac{dV}{d\hat{x}}\right\rangle$$

These resemble Newton's equations!

### 8.4 Classical Limit

Ehrenfest's theorem shows that expectation values follow classical trajectories IF:
- The potential is slowly varying
- Wave packet remains narrow
- $$\langle dV/dx \rangle \approx dV/dx|_{x=\langle x \rangle}$$

---

## 9. Symmetries and Conservation Laws

### 9.1 Symmetries as Unitary Operators

A symmetry is a unitary transformation $$\hat{U}$$ such that:
$$\hat{U}^\dagger\hat{H}\hat{U} = \hat{H}$$

or equivalently: $$[\hat{H}, \hat{U}] = 0$$

### 9.2 Generators and Conserved Quantities

If $$\hat{U} = e^{-i\hat{G}\theta}$$ and $$[\hat{H}, \hat{G}] = 0$$, then $$\langle\hat{G}\rangle$$ is conserved.

**Examples:**
| Symmetry | Generator | Conserved Quantity |
|----------|-----------|-------------------|
| Time translation | $$\hat{H}$$ | Energy |
| Space translation | $$\hat{p}$$ | Momentum |
| Rotation | $$\hat{L}$$ | Angular momentum |
| Phase rotation | $$\hat{N}$$ | Particle number |

### 9.3 Noether's Theorem (Quantum Version)

Continuous symmetries correspond to conservation laws:
$$[\hat{H}, \hat{G}] = 0 \Rightarrow \frac{d\langle\hat{G}\rangle}{dt} = 0$$

---

## 10. Key Results Summary

### Measurement
- Born rule: $$P(a) = |\langle a|\psi\rangle|^2$$
- Collapse: $$|\psi\rangle \to |a\rangle$$ (normalized)
- Expectation: $$\langle\hat{A}\rangle = \langle\psi|\hat{A}|\psi\rangle$$

### Dynamics
- Schrödinger: $$i\hbar\partial_t|\psi\rangle = \hat{H}|\psi\rangle$$
- Evolution: $$|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle$$
- Heisenberg: $$\frac{d\hat{A}}{dt} = \frac{i}{\hbar}[\hat{H},\hat{A}]$$

### Propagator
- $$\psi(x,t) = \int K(x,t;x',0)\psi(x',0)dx'$$
- Free particle: $$K_0 \propto \exp[im(x-x')^2/(2\hbar t)]$$

---

## 11. Common Qualifying Exam Problem Types

1. **Calculate measurement probabilities** for a given state
2. **Find the state after measurement** (including normalization)
3. **Compute time evolution** of specific states
4. **Derive the propagator** for simple systems
5. **Apply Ehrenfest's theorem** to various Hamiltonians
6. **Compare Schrödinger and Heisenberg** pictures
7. **Identify conserved quantities** from Hamiltonian symmetries

---

*Review Guide for Week 146 — Measurement and Dynamics*
*Month 37: QM Foundations Review I*
