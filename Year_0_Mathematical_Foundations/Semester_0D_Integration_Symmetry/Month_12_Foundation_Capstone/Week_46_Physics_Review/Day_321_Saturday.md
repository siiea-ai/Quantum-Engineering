# Day 321: Classical-Quantum Bridge

## Overview

**Month 12, Week 46, Day 6 — Saturday**

Today we build the bridge from classical to quantum mechanics: canonical quantization, the correspondence principle, and why classical mechanics emerges as the $\hbar \to 0$ limit.

## Learning Objectives

1. Understand canonical quantization
2. Apply the correspondence principle
3. See classical mechanics as QM limit
4. Prepare for Year 1 quantum mechanics

---

## 1. Canonical Quantization

### The Prescription

Replace Poisson brackets with commutators:

$$\boxed{\{A, B\} \to \frac{1}{i\hbar}[\hat{A}, \hat{B}]}$$

### Fundamental Commutators

$$[\hat{q}, \hat{p}] = i\hbar$$
$$[\hat{x}_i, \hat{p}_j] = i\hbar\delta_{ij}$$

### Position Representation

$$\hat{x} \to x, \quad \hat{p} \to -i\hbar\frac{\partial}{\partial x}$$

---

## 2. The Correspondence Principle

### Bohr's Statement

Quantum mechanics must reduce to classical mechanics for large quantum numbers.

### Ehrenfest's Theorem

$$\frac{d\langle \hat{x} \rangle}{dt} = \frac{\langle \hat{p} \rangle}{m}$$
$$\frac{d\langle \hat{p} \rangle}{dt} = -\langle \nabla V \rangle$$

For narrow wave packets: classical trajectories emerge.

---

## 3. Classical Limit ($\hbar \to 0$)

### WKB Approximation

$$\psi \approx A(x)e^{iS(x)/\hbar}$$

As $\hbar \to 0$: $S$ satisfies Hamilton-Jacobi equation.

### Coherent States

Minimum uncertainty wave packets that follow classical trajectories.

---

## 4. The Parallel Structure

| Classical | Quantum |
|-----------|---------|
| Phase space $(q, p)$ | Hilbert space $\mathcal{H}$ |
| Observable $f(q, p)$ | Operator $\hat{f}$ |
| Poisson bracket | $\frac{1}{i\hbar}$ Commutator |
| Hamilton's equations | Schrödinger equation |
| State $(q_0, p_0)$ | State $\|\psi\rangle$ |
| Probability on phase space | $\|\psi\|^2$ |

---

## 5. Key Quantum Features

### Uncertainty Principle

$$\Delta x \Delta p \geq \frac{\hbar}{2}$$

From $[\hat{x}, \hat{p}] = i\hbar$.

### Quantization of Energy

Discrete spectrum for bound states.

### Superposition

Linear combinations of states.

### Interference

No classical analog for amplitudes.

---

## 6. Computational Lab

```python
"""
Day 321: Classical-Quantum Bridge
"""

import numpy as np
import matplotlib.pyplot as plt

def coherent_state():
    """Show coherent state following classical trajectory."""
    # Harmonic oscillator parameters
    omega = 1
    hbar = 0.5  # Use small hbar to show classical-like behavior

    # Classical trajectory
    t = np.linspace(0, 4*np.pi, 200)
    x_classical = np.cos(omega * t)
    p_classical = -np.sin(omega * t)

    # Quantum expectation values follow same trajectory
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, x_classical, 'b-', label='⟨x⟩')
    plt.plot(t, p_classical, 'r-', label='⟨p⟩')
    plt.xlabel('t')
    plt.ylabel('Expectation value')
    plt.title('Coherent State: Classical Trajectory')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_classical, p_classical, 'g-')
    plt.xlabel('⟨x⟩')
    plt.ylabel('⟨p⟩')
    plt.title('Phase Space Trajectory')
    plt.axis('equal')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('coherent_state.png', dpi=150)
    plt.close()
    print("Saved: coherent_state.png")


def uncertainty_principle():
    """Visualize uncertainty principle."""
    # Minimum uncertainty state
    x = np.linspace(-5, 5, 500)
    sigma = 1

    # Position space wavefunction
    psi_x = np.exp(-x**2 / (2*sigma**2)) / (np.pi*sigma**2)**0.25

    # Momentum space (Fourier transform)
    p = np.linspace(-5, 5, 500)
    sigma_p = 0.5 / sigma  # hbar = 1
    psi_p = np.exp(-p**2 / (2*sigma_p**2)) / (np.pi*sigma_p**2)**0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, np.abs(psi_x)**2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('|ψ(x)|²')
    axes[0].set_title(f'Position: Δx = {sigma:.2f}')

    axes[1].plot(p, np.abs(psi_p)**2)
    axes[1].set_xlabel('p')
    axes[1].set_ylabel('|φ(p)|²')
    axes[1].set_title(f'Momentum: Δp = {sigma_p:.2f}')

    plt.suptitle(f'Uncertainty: Δx·Δp = {sigma * sigma_p:.2f} ≥ ℏ/2')
    plt.tight_layout()
    plt.savefig('uncertainty.png', dpi=150)
    plt.close()
    print("Saved: uncertainty.png")


if __name__ == "__main__":
    coherent_state()
    uncertainty_principle()
```

---

## Summary

### The Bridge

$$\boxed{\text{Classical: } \{q, p\} = 1 \quad \longrightarrow \quad \text{Quantum: } [\hat{q}, \hat{p}] = i\hbar}$$

### Ready for Quantum Mechanics!

Year 0 has prepared you with:
- Linear algebra for Hilbert spaces
- Differential equations for Schrödinger equation
- Complex analysis for wave functions
- Group theory for symmetries
- Classical mechanics for correspondence

---

## Preview: Day 322

Tomorrow: **Physics Integration Exam** — comprehensive assessment of all Year 0 physics.
