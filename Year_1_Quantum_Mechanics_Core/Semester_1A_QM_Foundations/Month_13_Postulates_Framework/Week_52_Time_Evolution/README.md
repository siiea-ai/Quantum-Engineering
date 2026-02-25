# Week 52: Time Evolution

## Overview

**Days:** 358-364 (7 days)
**Position:** Year 1, Month 13, Week 4
**Theme:** Dynamics of Quantum Systems

This week addresses the fifth and final postulate: how quantum states evolve in time. We develop the Schrodinger equation, the time evolution operator, and the three fundamental pictures of quantum mechanics (Schrodinger, Heisenberg, Interaction). This completes our study of the quantum mechanical postulates.

---

## Learning Objectives

By the end of Week 52, you will be able to:

1. Derive and interpret the time-dependent Schrodinger equation
2. Construct the time evolution operator U(t) and verify its properties
3. Distinguish stationary states from time-dependent superpositions
4. Transform between Schrodinger and Heisenberg pictures
5. Apply the Heisenberg equation of motion to compute operator dynamics
6. Use the Interaction picture for time-dependent perturbation problems
7. Synthesize all five quantum postulates into a unified framework

---

## Daily Schedule

| Day | Date | Topic | Shankar | Sakurai |
|-----|------|-------|---------|---------|
| **358** | Mon | The Schrodinger Equation | Ch. 4.1-4.2 | Ch. 2.1 |
| **359** | Tue | Time Evolution Operator | Ch. 4.3 | Ch. 2.1 |
| **360** | Wed | Stationary States | Ch. 4.4-4.5 | Ch. 2.1 |
| **361** | Thu | Schrodinger Picture | Ch. 4.6 | Ch. 2.2 |
| **362** | Fri | Heisenberg Picture | Ch. 4.7 | Ch. 2.2 |
| **363** | Sat | Interaction Picture | Ch. 4.8 | Ch. 2.3 |
| **364** | Sun | Month 13 Capstone | — | — |

---

## Key Concepts

### 1. The Schrodinger Equation

The fundamental dynamical equation of quantum mechanics:

$$\boxed{i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle}$$

This is a first-order differential equation in time, ensuring unique evolution from initial conditions.

### 2. Time Evolution Operator

For time-independent Hamiltonians:

$$\boxed{\hat{U}(t) = e^{-i\hat{H}t/\hbar}}$$

Properties:
- **Unitarity:** $\hat{U}^\dagger(t)\hat{U}(t) = \hat{I}$
- **Composition:** $\hat{U}(t_1)\hat{U}(t_2) = \hat{U}(t_1 + t_2)$
- **Inverse:** $\hat{U}^{-1}(t) = \hat{U}(-t) = \hat{U}^\dagger(t)$

### 3. Stationary States

Energy eigenstates have simple time evolution:

$$|\psi(t)\rangle = e^{-iEt/\hbar}|E\rangle$$

The probability density $|\langle x|E\rangle|^2$ is time-independent.

### 4. The Three Pictures

| Picture | States | Operators | Use Case |
|---------|--------|-----------|----------|
| Schrodinger | Evolve: $\|\psi_S(t)\rangle = \hat{U}(t)\|\psi(0)\rangle$ | Fixed | Time-dependent problems |
| Heisenberg | Fixed: $\|\psi_H\rangle$ | Evolve: $\hat{A}_H(t) = \hat{U}^\dagger(t)\hat{A}_S\hat{U}(t)$ | Operator algebra |
| Interaction | Partial: $\|\psi_I(t)\rangle$ | Partial: $\hat{A}_I(t)$ | Perturbation theory |

### 5. Heisenberg Equation of Motion

$$\boxed{\frac{d\hat{A}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{A}_H] + \frac{\partial \hat{A}_S}{\partial t}}$$

This is the quantum analog of Hamilton's equation $\dot{A} = \{A, H\}$.

---

## Essential Formulas

### Time Evolution
$$|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle$$
$$|\psi(t)\rangle = \sum_n c_n e^{-iE_n t/\hbar}|E_n\rangle$$

### Expectation Value Dynamics
$$\frac{d}{dt}\langle \hat{A} \rangle = \frac{i}{\hbar}\langle[\hat{H}, \hat{A}]\rangle + \left\langle\frac{\partial \hat{A}}{\partial t}\right\rangle$$

### Picture Transformations
$$\hat{A}_H(t) = \hat{U}^\dagger(t)\hat{A}_S\hat{U}(t)$$
$$|\psi_I(t)\rangle = e^{i\hat{H}_0 t/\hbar}|\psi_S(t)\rangle$$

### Interaction Picture Evolution
$$i\hbar\frac{\partial}{\partial t}|\psi_I(t)\rangle = \hat{V}_I(t)|\psi_I(t)\rangle$$

---

## Connections to Year 0

| Year 0 Topic | Week 52 Application |
|--------------|---------------------|
| Classical Mechanics (Month 6) | Hamilton's equations → Heisenberg equation |
| Poisson brackets | $\{A, H\} \to \frac{1}{i\hbar}[\hat{A}, \hat{H}]$ |
| Differential Equations (Month 5) | Schrodinger equation as evolution equation |
| Matrix Exponentials | Time evolution operator $e^{-i\hat{H}t/\hbar}$ |
| Unitary Operators (Month 4-5) | Conservation of probability |

---

## Quantum Computing Connection

Time evolution is fundamental to quantum computing:

| QM Concept | Quantum Computing |
|------------|-------------------|
| Time evolution | Gate operations |
| $\hat{U}(t) = e^{-i\hat{H}t}$ | Hamiltonian simulation |
| Stationary states | Eigenstates of problem Hamiltonians |
| Heisenberg picture | Operator tracking in circuits |
| Interaction picture | Control pulse design |

**Gate Implementation:** The CNOT gate can be understood as time evolution under:
$$\hat{H}_{\text{CNOT}} = \frac{\pi}{4}(I - Z_1)(I - X_2)$$

---

## Problem Set Topics

1. Solve the Schrodinger equation for specific systems
2. Verify properties of the time evolution operator
3. Compute time evolution of superposition states
4. Transform observables between pictures
5. Apply Heisenberg equation to harmonic oscillator
6. Use Interaction picture for time-dependent perturbations
7. Prove Ehrenfest's theorem

---

## Computational Lab (Day 364)

**Topics:**
- Numerical solution of time-dependent Schrodinger equation
- Animation of wave packet evolution
- Comparison of pictures for expectation values
- Simulation of Rabi oscillations

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Time evolution operator for two-level system
def U(t, H, hbar=1):
    """Compute time evolution operator."""
    return expm(-1j * H * t / hbar)

# Example: Spin-1/2 in magnetic field
omega = 1.0  # Larmor frequency
H = omega/2 * np.array([[1, 0], [0, -1]])  # sigma_z Hamiltonian

# Time evolution of |+x> state
psi_0 = np.array([[1], [1]]) / np.sqrt(2)  # |+x>
t = np.pi / omega  # Time for pi rotation
psi_t = U(t, H) @ psi_0
```

---

## Self-Assessment Checklist

After Week 52, you should be able to:

- [ ] Write down the time-dependent Schrodinger equation
- [ ] Compute the time evolution operator for a given Hamiltonian
- [ ] Evolve superposition states and compute oscillation frequencies
- [ ] Identify conservation laws from $[\hat{H}, \hat{A}] = 0$
- [ ] Transform between Schrodinger and Heisenberg pictures
- [ ] Derive the Heisenberg equation of motion
- [ ] Set up problems in the Interaction picture
- [ ] Connect classical evolution to quantum evolution

---

## Preview: Month 14

With the complete postulate framework established, Month 14 applies these tools to one-dimensional quantum systems: the free particle, wave packets, infinite and finite square wells, the quantum harmonic oscillator, and tunneling through barriers.

---

**Next:** [Day_358_Monday.md](Day_358_Monday.md) — The Schrodinger Equation
