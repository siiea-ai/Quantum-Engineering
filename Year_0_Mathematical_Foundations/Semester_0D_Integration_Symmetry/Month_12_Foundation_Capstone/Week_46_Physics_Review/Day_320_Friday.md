# Day 320: Phase Space and Poisson Brackets

## Overview

**Month 12, Week 46, Day 5 — Friday**

Today we explore the geometric structure of Hamiltonian mechanics: phase space, symplectic geometry, and the Poisson bracket algebra that becomes commutators in quantum mechanics.

## Learning Objectives

1. Master phase space geometry
2. Calculate Poisson brackets fluently
3. Understand symplectic structure
4. Connect to quantum commutators

---

## 1. Phase Space Structure

### Definition

Phase space: $(q_1,...,q_n,p_1,...,p_n)$ — a $2n$-dimensional manifold.

### Symplectic Form

$$\omega = \sum_i dp_i \wedge dq_i$$

### Hamiltonian Flow

$$\dot{z} = J\nabla H, \quad J = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$$

---

## 2. Poisson Brackets

### Definition

$$\{f, g\} = \sum_i \left(\frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i}\right)$$

### Properties

- Antisymmetry: $\{f, g\} = -\{g, f\}$
- Linearity: $\{af + bg, h\} = a\{f, h\} + b\{g, h\}$
- Leibniz: $\{fg, h\} = f\{g, h\} + g\{f, h\}$
- Jacobi: $\{\{f, g\}, h\} + \text{cyclic} = 0$

### Fundamental Brackets

$$\{q_i, q_j\} = 0, \quad \{p_i, p_j\} = 0, \quad \{q_i, p_j\} = \delta_{ij}$$

---

## 3. Time Evolution

$$\frac{df}{dt} = \{f, H\} + \frac{\partial f}{\partial t}$$

Conservation: $\{f, H\} = 0 \implies f$ constant.

---

## 4. Angular Momentum Algebra

$$\{L_i, L_j\} = \epsilon_{ijk}L_k$$

Compare to QM: $[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$

---

## 5. Canonical Quantization

$$\boxed{\{A, B\} \to \frac{1}{i\hbar}[\hat{A}, \hat{B}]}$$

This is the bridge from classical to quantum mechanics.

---

## 6. Computational Lab

```python
"""
Day 320: Phase Space and Poisson Brackets
"""

import numpy as np
import sympy as sp

def poisson_bracket_algebra():
    """Demonstrate Poisson bracket calculations."""
    print("=" * 50)
    print("POISSON BRACKET ALGEBRA")
    print("=" * 50)

    q, p = sp.symbols('q p')

    def poisson(f, g):
        return sp.diff(f, q)*sp.diff(g, p) - sp.diff(f, p)*sp.diff(g, q)

    # Fundamental brackets
    print(f"\n{{q, p}} = {poisson(q, p)}")
    print(f"{{q, q}} = {poisson(q, q)}")
    print(f"{{p, p}} = {poisson(p, p)}")

    # Harmonic oscillator
    H = p**2/2 + q**2/2
    print(f"\nHarmonic oscillator H = p²/2 + q²/2")
    print(f"{{q, H}} = {poisson(q, H)}")
    print(f"{{p, H}} = {poisson(p, H)}")

    # Angular momentum (2D)
    x, y, px, py = sp.symbols('x y p_x p_y')
    Lz = x*py - y*px

    def poisson_2d(f, g):
        return (sp.diff(f, x)*sp.diff(g, px) - sp.diff(f, px)*sp.diff(g, x) +
                sp.diff(f, y)*sp.diff(g, py) - sp.diff(f, py)*sp.diff(g, y))

    print(f"\n{{x, Lz}} = {poisson_2d(x, Lz)}")
    print(f"{{y, Lz}} = {poisson_2d(y, Lz)}")

if __name__ == "__main__":
    poisson_bracket_algebra()
```

---

## Summary

### The Poisson-Commutator Correspondence

$$\boxed{\{q, p\} = 1 \quad \leftrightarrow \quad [\hat{q}, \hat{p}] = i\hbar}$$

---

## Preview: Day 321

Tomorrow: **Classical-Quantum Bridge** — canonical quantization and the correspondence principle.
