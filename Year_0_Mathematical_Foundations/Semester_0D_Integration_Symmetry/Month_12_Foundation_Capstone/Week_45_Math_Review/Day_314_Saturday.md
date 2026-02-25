# Day 314: Group Theory Synthesis

## Overview

**Month 12, Week 45, Day 6 — Saturday**

Today we synthesize group theory: from abstract groups to Lie algebras to quantum mechanical angular momentum. Symmetry is the organizing principle of modern physics.

## Learning Objectives

1. Review abstract group theory
2. Connect representations to quantum states
3. Master Lie groups and algebras
4. Apply to angular momentum

---

## 1. Abstract Groups

### Axioms

- Closure: $a \cdot b \in G$
- Associativity: $(ab)c = a(bc)$
- Identity: $ea = ae = a$
- Inverse: $aa^{-1} = e$

### Key Results

**Lagrange:** $|H|$ divides $|G|$

**First Isomorphism:** $G/\ker\phi \cong \text{im}\phi$

---

## 2. Representations

### Definition

Homomorphism $D: G \to GL(V)$

$$D(gh) = D(g)D(h)$$

### Characters

$$\chi(g) = \text{Tr}(D(g))$$

### Orthogonality

$$\sum_g \chi^{(\alpha)*}(g)\chi^{(\beta)}(g) = |G|\delta_{\alpha\beta}$$

---

## 3. Lie Groups and Algebras

### Lie Group

Continuous group with smooth structure.

### Lie Algebra

Tangent space at identity with bracket $[X, Y]$.

### Exponential Map

$$g = e^X$$

### Example: SU(2)

Algebra: $[T_a, T_b] = i\epsilon_{abc}T_c$

Representations: Spin $j = 0, 1/2, 1, 3/2, ...$

---

## 4. Angular Momentum

### Commutation Relations

$$[J_i, J_j] = i\hbar\epsilon_{ijk}J_k$$

### Eigenvalues

$$J^2|j,m\rangle = \hbar^2 j(j+1)|j,m\rangle$$
$$J_z|j,m\rangle = \hbar m|j,m\rangle$$

### Addition

$$j_1 \otimes j_2 = |j_1 - j_2| \oplus \cdots \oplus (j_1 + j_2)$$

---

## 5. Physics Applications

### Selection Rules

From Wigner-Eckart theorem:
$$\langle j', m'|T^{(k)}_q|j, m\rangle = \langle j'\|T^{(k)}\|j\rangle C^{j'm'}_{jm;kq}$$

### Atomic Structure

- Orbital angular momentum: $Y_\ell^m$
- Spin: Pauli matrices
- Coupling: Clebsch-Gordan

---

## 6. Computational Lab

```python
"""
Day 314: Group Theory Synthesis
"""

import numpy as np

def su2_representations():
    """Generate SU(2) representations."""
    print("=" * 50)
    print("SU(2) REPRESENTATIONS")
    print("=" * 50)

    for j in [0.5, 1, 1.5, 2]:
        dim = int(2*j + 1)
        m_values = np.arange(j, -j-1, -1)

        # Jz
        Jz = np.diag(m_values)

        # Casimir
        casimir = j * (j + 1)

        print(f"\nj = {j}: dim = {dim}, J² = {casimir}")
        print(f"   m values: {m_values}")


def clebsch_gordan_example():
    """Two spin-1/2 addition."""
    print("\n" + "=" * 50)
    print("TWO SPIN-1/2 ADDITION")
    print("=" * 50)

    print("\nj₁ = 1/2 ⊗ j₂ = 1/2 = 0 ⊕ 1")
    print("\nSinglet (j=0):")
    print("  |0,0⟩ = (|↑↓⟩ - |↓↑⟩)/√2")
    print("\nTriplet (j=1):")
    print("  |1,1⟩ = |↑↑⟩")
    print("  |1,0⟩ = (|↑↓⟩ + |↓↑⟩)/√2")
    print("  |1,-1⟩ = |↓↓⟩")


# Main
if __name__ == "__main__":
    su2_representations()
    clebsch_gordan_example()
```

---

## Summary

### The Symmetry → QM Pipeline

$$\boxed{\text{Group} \to \text{Algebra} \to \text{Generators} \to \text{Observables}}$$

### Key Formula

$$[J_i, J_j] = i\hbar\epsilon_{ijk}J_k$$

---

## Preview: Day 315

Tomorrow: **Mathematics Integration Exam** — comprehensive assessment.
