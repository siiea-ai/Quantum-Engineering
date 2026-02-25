# Day 318: Hamiltonian Mechanics Review

## Overview

**Month 12, Week 46, Day 3 — Wednesday**

Today we review Hamiltonian mechanics: the phase space formulation that directly maps to quantum mechanics via canonical quantization.

## Learning Objectives

1. Master the Legendre transform
2. Work with Hamilton's equations
3. Understand phase space structure
4. Connect to quantum mechanics

---

## 1. The Hamiltonian

### Legendre Transform

$$H(q_i, p_i, t) = \sum_i p_i \dot{q}_i - L$$

### Hamilton's Equations

$$\boxed{\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}}$$

---

## 2. Phase Space

### Definition

The 2n-dimensional space of $(q_1, ..., q_n, p_1, ..., p_n)$.

### Symplectic Structure

$$\omega = \sum_i dp_i \wedge dq_i$$

### Liouville's Theorem

Phase space volume is conserved under Hamiltonian flow.

---

## 3. Poisson Brackets

$$\{f, g\} = \sum_i \left(\frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i}\right)$$

### Fundamental Brackets

$$\{q_i, q_j\} = 0, \quad \{p_i, p_j\} = 0, \quad \{q_i, p_j\} = \delta_{ij}$$

### Time Evolution

$$\frac{df}{dt} = \{f, H\} + \frac{\partial f}{\partial t}$$

---

## 4. Canonical Transformations

Preserve the form of Hamilton's equations.

### Generating Functions

Transform $(q, p) \to (Q, P)$ using $F_1(q, Q)$, $F_2(q, P)$, etc.

---

## 5. The QM Connection

### Canonical Quantization

$$\{q_i, p_j\} = \delta_{ij} \quad \to \quad [\hat{q}_i, \hat{p}_j] = i\hbar\delta_{ij}$$

### Heisenberg Picture

$$\frac{d\hat{A}}{dt} = \frac{1}{i\hbar}[\hat{A}, \hat{H}]$$

---

## 6. Computational Lab

```python
"""
Day 318: Hamiltonian Mechanics
"""

import numpy as np
import matplotlib.pyplot as plt

def phase_space_harmonic():
    """Phase space trajectories for harmonic oscillator."""
    # H = p²/2m + mω²q²/2
    omega = 1

    q = np.linspace(-2, 2, 20)
    p = np.linspace(-2, 2, 20)
    Q, P = np.meshgrid(q, p)

    # Hamilton's equations: dq/dt = p, dp/dt = -ω²q
    dQ = P
    dP = -omega**2 * Q

    # Energy contours
    H = P**2/2 + omega**2 * Q**2/2

    plt.figure(figsize=(8, 8))
    plt.contour(Q, P, H, levels=20)
    plt.quiver(Q, P, dQ, dP, alpha=0.7)
    plt.xlabel('q')
    plt.ylabel('p')
    plt.title('Phase Space: Harmonic Oscillator')
    plt.axis('equal')
    plt.savefig('phase_space.png', dpi=150)
    plt.close()
    print("Saved: phase_space.png")

if __name__ == "__main__":
    phase_space_harmonic()
```

---

## Summary

### The Hamiltonian Framework

$$\boxed{\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}}$$

$$\boxed{\{q_i, p_j\} = \delta_{ij} \quad \to \quad [\hat{q}_i, \hat{p}_j] = i\hbar\delta_{ij}}$$

---

## Preview: Day 319

Tomorrow: **Symmetry and Conservation** — Noether's theorem in depth.
