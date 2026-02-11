# Day 313: Functional Analysis Synthesis

## Overview

**Month 12, Week 45, Day 5 — Friday**

Today we synthesize functional analysis: Hilbert spaces, linear operators, spectral theory. This is the mathematical language of quantum mechanics — states are vectors in Hilbert space, observables are self-adjoint operators.

## Learning Objectives

1. Master Hilbert space structure
2. Understand operator theory fundamentals
3. Connect to quantum mechanics formalism
4. Prepare for rigorous QM treatment

---

## 1. Hilbert Spaces

### Definition

A complete inner product space over $\mathbb{C}$.

### Key Properties

- Inner product: $\langle f|g \rangle$
- Norm: $\|f\| = \sqrt{\langle f|f\rangle}$
- Completeness: Cauchy sequences converge

### Examples

| Space | Inner Product |
|-------|---------------|
| $\mathbb{C}^n$ | $\langle u\|v\rangle = \sum_i u_i^* v_i$ |
| $L^2[a,b]$ | $\langle f\|g\rangle = \int_a^b f^*(x)g(x)dx$ |
| $\ell^2$ | $\langle a\|b\rangle = \sum_n a_n^* b_n$ |

---

## 2. Operators

### Bounded Operators

$$\|Af\| \leq M\|f\| \quad \forall f$$

### Adjoint

$$\langle f|Ag\rangle = \langle A^\dagger f|g\rangle$$

### Self-Adjoint (Hermitian)

$$A = A^\dagger$$

**Properties:**
- Real eigenvalues
- Orthogonal eigenvectors
- Spectral decomposition

### Unitary

$$U^\dagger U = UU^\dagger = I$$

Preserves inner product: $\langle Uf|Ug\rangle = \langle f|g\rangle$

---

## 3. Spectral Theory

### Spectral Theorem

For self-adjoint $A$:
$$A = \int \lambda \, dE_\lambda$$

where $E_\lambda$ is the spectral family.

### Discrete Spectrum

$$A = \sum_n \lambda_n |n\rangle\langle n|$$

### Continuous Spectrum

Example: Position operator $\hat{x}$
$$\hat{x} = \int x |x\rangle\langle x| dx$$

---

## 4. Quantum Mechanics Connection

### Postulate 1: States

States are vectors $|\psi\rangle$ in Hilbert space $\mathcal{H}$.

### Postulate 2: Observables

Observables are self-adjoint operators.

### Postulate 3: Measurement

Measuring $A$ on $|\psi\rangle$ gives eigenvalue $a$ with probability $|\langle a|\psi\rangle|^2$.

### Postulate 4: Evolution

$$|\psi(t)\rangle = e^{-iHt/\hbar}|\psi(0)\rangle$$

---

## 5. Computational Lab

```python
"""
Day 313: Functional Analysis Synthesis
"""

import numpy as np
from scipy.linalg import eigh

def hilbert_space_demo():
    """Demonstrate Hilbert space concepts."""
    print("=" * 50)
    print("HILBERT SPACE CONCEPTS")
    print("=" * 50)

    # L² inner product (discretized)
    N = 1000
    x = np.linspace(0, np.pi, N)
    dx = x[1] - x[0]

    # Orthonormal functions: sin(nx)/√(π/2)
    def phi(n, x):
        return np.sin(n * x) * np.sqrt(2/np.pi)

    # Verify orthonormality
    print("\nOrthonormality check:")
    for n in range(1, 4):
        for m in range(1, 4):
            inner = np.sum(phi(n, x) * phi(m, x)) * dx
            print(f"  ⟨φ_{n}|φ_{m}⟩ = {inner:.4f}")


def spectral_decomposition():
    """Demonstrate spectral decomposition."""
    print("\n" + "=" * 50)
    print("SPECTRAL DECOMPOSITION")
    print("=" * 50)

    # Hermitian matrix
    H = np.array([[2, 1, 0],
                  [1, 3, 1],
                  [0, 1, 2]])

    eigenvalues, eigenvectors = eigh(H)

    print(f"\nH = \n{H}")
    print(f"\nEigenvalues: {eigenvalues}")

    # Spectral decomposition: H = Σ λ_n |n⟩⟨n|
    H_reconstructed = np.zeros_like(H, dtype=float)
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        projector = np.outer(v, v)
        H_reconstructed += lam * projector

    print(f"\nH reconstructed = \n{H_reconstructed.round(6)}")
    print(f"Match: {np.allclose(H, H_reconstructed)}")


# Main
if __name__ == "__main__":
    hilbert_space_demo()
    spectral_decomposition()
```

---

## Summary

### Functional Analysis → QM Dictionary

$$\boxed{\text{Hilbert Space} \leftrightarrow \text{State Space}}$$
$$\boxed{\text{Self-Adjoint Operator} \leftrightarrow \text{Observable}}$$
$$\boxed{\text{Eigenvalue} \leftrightarrow \text{Measurement Outcome}}$$

---

## Preview: Day 314

Tomorrow: **Group Theory Synthesis** — symmetries and quantum mechanics.
