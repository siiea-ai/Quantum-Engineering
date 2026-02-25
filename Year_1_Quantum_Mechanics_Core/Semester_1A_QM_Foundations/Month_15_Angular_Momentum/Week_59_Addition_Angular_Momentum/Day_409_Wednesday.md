# Day 409: Clebsch-Gordan Coefficients

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: CG coefficients |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab |

---

## Learning Objectives

By the end of Day 409, you will be able to:

1. Define Clebsch-Gordan coefficients mathematically
2. Apply the triangle rule and selection rules
3. Calculate simple CG coefficients
4. Use tables and software for complex cases
5. Understand orthogonality and completeness

---

## Core Content

### 1. Definition

The Clebsch-Gordan (CG) coefficients relate the coupled and uncoupled bases:

$$\boxed{|j, m\rangle = \sum_{m_1, m_2} \langle j_1, m_1; j_2, m_2 | j, m\rangle |j_1, m_1; j_2, m_2\rangle}$$

The CG coefficient is: ⟨j₁,m₁;j₂,m₂|j,m⟩

Alternative notation: C^{j,m}_{j₁m₁,j₂m₂} or (j₁m₁j₂m₂|jm)

### 2. Selection Rules

CG coefficients vanish unless:

1. **Triangle rule:** |j₁ - j₂| ≤ j ≤ j₁ + j₂
2. **Magnetic quantum number:** m = m₁ + m₂
3. **Integer constraint:** j₁ + j₂ + j is an integer

### 3. Orthogonality Relations

**Orthogonality in j, m:**
$$\sum_{m_1, m_2} \langle j_1, m_1; j_2, m_2 | j, m\rangle \langle j_1, m_1; j_2, m_2 | j', m'\rangle = \delta_{jj'}\delta_{mm'}$$

**Completeness:**
$$\sum_{j,m} \langle j_1, m_1; j_2, m_2 | j, m\rangle \langle j_1, m'_1; j_2, m'_2 | j, m\rangle = \delta_{m_1m'_1}\delta_{m_2m'_2}$$

### 4. Symmetry Properties

$$\langle j_1, m_1; j_2, m_2 | j, m\rangle = (-1)^{j_1+j_2-j}\langle j_2, m_2; j_1, m_1 | j, m\rangle$$

$$\langle j_1, m_1; j_2, m_2 | j, m\rangle = (-1)^{j_1+j_2-j}\langle j_1, -m_1; j_2, -m_2 | j, -m\rangle$$

### 5. Recursion Relations

Starting from the highest weight state:
$$|j, j\rangle = |j_1, j_1; j_2, j_2\rangle \text{ when } j = j_1 + j_2$$

Apply Ĵ₋ = Ĵ₁₋ + Ĵ₂₋ to generate lower m states.

### 6. Two Spin-1/2 CG Coefficients

| m₁ | m₂ | j=1, m | j=0, m=0 |
|----|----|----|------|
| +1/2 | +1/2 | 1 (m=1) | — |
| +1/2 | -1/2 | 1/√2 (m=0) | 1/√2 |
| -1/2 | +1/2 | 1/√2 (m=0) | -1/√2 |
| -1/2 | -1/2 | 1 (m=-1) | — |

---

## Quantum Computing Connection

CG coefficients appear in:
- **Quantum error correction:** Symmetry-based codes
- **Variational algorithms:** State preparation
- **Quantum simulation:** Angular momentum conservation

---

## Worked Examples

### Example 1: Construct |1,0⟩ for Two Spin-1/2

**Problem:** Find |j=1, m=0⟩.

**Solution:**
Start from |1,1⟩ = |↑↑⟩ and apply Ĵ₋ = Ĵ₁₋ + Ĵ₂₋:

$$\hat{J}_-|1,1\rangle = \hbar\sqrt{2}|1,0\rangle$$

$$(\hat{J}_{1-} + \hat{J}_{2-})|{↑↑}\rangle = \hbar(|{↓↑}\rangle + |{↑↓}\rangle)$$

Therefore:
$$|1,0\rangle = \frac{1}{\sqrt{2}}(|{↑↓}\rangle + |{↓↑}\rangle)$$

### Example 2: Construct |0,0⟩ (Singlet)

**Problem:** Find |j=0, m=0⟩.

**Solution:**
|0,0⟩ must be orthogonal to |1,0⟩ and in the m=0 subspace:

$$|0,0\rangle = \frac{1}{\sqrt{2}}(|{↑↓}\rangle - |{↓↑}\rangle)$$

This is the **singlet state** = **Bell state** |Ψ⁻⟩!

### Example 3: CG for j₁=1, j₂=1/2, j=3/2

**Problem:** Find ⟨1,1; 1/2,1/2 | 3/2,3/2⟩.

**Solution:**
The maximum m = 3/2 requires m₁=1, m₂=1/2.

Since this is the only way to get m=3/2:
$$\langle 1,1; 1/2,1/2 | 3/2,3/2\rangle = 1$$

---

## Practice Problems

### Direct Application

1. Using the table, find ⟨1/2,1/2; 1/2,-1/2 | 1,0⟩.

2. What is ⟨1/2,1/2; 1/2,-1/2 | 0,0⟩?

3. For j₁=1, j₂=1, what are the allowed j values?

### Intermediate

4. Calculate all CG coefficients for j₁=1, j₂=1/2, j=3/2.

5. Verify orthogonality: show that |1,0⟩ and |0,0⟩ are orthogonal.

6. Express |↑↓⟩ in the coupled basis.

### Challenging

7. Derive the CG coefficient ⟨1,0; 1,0|2,0⟩ using the recursion relation.

8. Prove that the singlet |0,0⟩ is antisymmetric under particle exchange.

---

## Computational Lab

```python
"""
Day 409 Computational Lab: Clebsch-Gordan Coefficients
"""

import numpy as np
from sympy.physics.quantum.cg import CG
from sympy import sqrt, Rational, N

def cg_coefficient(j1, m1, j2, m2, j, m):
    """Calculate CG coefficient using SymPy."""
    cg = CG(j1, m1, j2, m2, j, m)
    return cg.doit()

def print_cg_table(j1, j2):
    """Print table of CG coefficients."""
    print(f"\nClebsch-Gordan Coefficients for j₁={j1}, j₂={j2}")
    print("=" * 60)

    # Triangle rule
    j_min = abs(j1 - j2)
    j_max = j1 + j2

    print(f"Allowed j values: {j_min} to {j_max}")

    for j in np.arange(j_max, j_min - 0.5, -1):
        print(f"\nj = {j}:")
        for m in np.arange(j, -j-0.5, -1):
            print(f"  m = {m}:")
            for m1 in np.arange(j1, -j1-0.5, -1):
                m2 = m - m1
                if abs(m2) <= j2:
                    cg = cg_coefficient(Rational(j1), Rational(m1),
                                       Rational(j2), Rational(m2),
                                       Rational(j), Rational(m))
                    if cg != 0:
                        print(f"    ⟨{j1},{m1};{j2},{m2}|{j},{m}⟩ = {cg}")

def two_spin_half_cg():
    """Complete CG table for two spin-1/2."""
    print("\nTwo Spin-1/2 CG Coefficients")
    print("=" * 60)

    j1, j2 = Rational(1,2), Rational(1,2)

    # j=1 triplet
    print("\nTriplet (j=1):")
    print("  |1,+1⟩ = |↑↑⟩")

    cg_pp = cg_coefficient(j1, Rational(1,2), j2, Rational(-1,2),
                           Rational(1), Rational(0))
    cg_mp = cg_coefficient(j1, Rational(-1,2), j2, Rational(1,2),
                           Rational(1), Rational(0))
    print(f"  |1, 0⟩ = {cg_pp}|↑↓⟩ + {cg_mp}|↓↑⟩")

    print("  |1,-1⟩ = |↓↓⟩")

    # j=0 singlet
    print("\nSinglet (j=0):")
    cg_pp_s = cg_coefficient(j1, Rational(1,2), j2, Rational(-1,2),
                             Rational(0), Rational(0))
    cg_mp_s = cg_coefficient(j1, Rational(-1,2), j2, Rational(1,2),
                             Rational(0), Rational(0))
    print(f"  |0, 0⟩ = {cg_pp_s}|↑↓⟩ + {cg_mp_s}|↓↑⟩")

def verify_orthonormality():
    """Verify orthonormality of CG expansion."""
    print("\nVerifying Orthonormality")
    print("=" * 60)

    j1, j2 = Rational(1,2), Rational(1,2)

    # Get |1,0⟩ coefficients
    c1_pp = float(N(cg_coefficient(j1, Rational(1,2), j2, Rational(-1,2),
                                    Rational(1), Rational(0))))
    c1_mp = float(N(cg_coefficient(j1, Rational(-1,2), j2, Rational(1,2),
                                    Rational(1), Rational(0))))

    # Get |0,0⟩ coefficients
    c0_pp = float(N(cg_coefficient(j1, Rational(1,2), j2, Rational(-1,2),
                                    Rational(0), Rational(0))))
    c0_mp = float(N(cg_coefficient(j1, Rational(-1,2), j2, Rational(1,2),
                                    Rational(0), Rational(0))))

    # Orthogonality: <1,0|0,0>
    overlap = c1_pp * c0_pp + c1_mp * c0_mp
    print(f"⟨1,0|0,0⟩ = {overlap:.6f} (should be 0)")

    # Normalization
    norm_1 = c1_pp**2 + c1_mp**2
    norm_0 = c0_pp**2 + c0_mp**2
    print(f"⟨1,0|1,0⟩ = {norm_1:.6f} (should be 1)")
    print(f"⟨0,0|0,0⟩ = {norm_0:.6f} (should be 1)")

def inverse_transformation():
    """Express uncoupled states in coupled basis."""
    print("\nInverse Transformation (Uncoupled → Coupled)")
    print("=" * 60)

    print("\n|↑↑⟩ = |1,+1⟩")
    print("|↓↓⟩ = |1,-1⟩")
    print("|↑↓⟩ = (|1,0⟩ + |0,0⟩)/√2")
    print("|↓↑⟩ = (|1,0⟩ - |0,0⟩)/√2")

    print("\nNote: |↑↓⟩ and |↓↑⟩ are NOT eigenstates of J²!")

if __name__ == "__main__":
    print("Day 409: Clebsch-Gordan Coefficients")
    print("=" * 60)

    # Two spin-1/2
    two_spin_half_cg()

    # Verify orthonormality
    verify_orthonormality()

    # Inverse transformation
    inverse_transformation()

    # General table
    print_cg_table(Rational(1), Rational(1,2))

    print("\nLab complete!")
```

---

## Summary

| Concept | Formula |
|---------|---------|
| CG definition | \|j,m⟩ = Σ ⟨j₁,m₁;j₂,m₂\|j,m⟩\|j₁,m₁;j₂,m₂⟩ |
| Selection rules | m = m₁ + m₂, \|j₁-j₂\| ≤ j ≤ j₁+j₂ |
| Orthogonality | Σ CG² = 1 |
| Two spin-1/2 | Triplet (j=1) + Singlet (j=0) |

---

## Daily Checklist

- [ ] I understand CG coefficient definition
- [ ] I can apply selection rules
- [ ] I can calculate simple CG coefficients
- [ ] I know the two spin-1/2 table
- [ ] I completed the computational lab

---

## Preview: Day 410

Tomorrow we apply angular momentum addition to atoms: spin-orbit coupling and the fine structure of hydrogen.

---

**Next:** [Day_410_Thursday.md](Day_410_Thursday.md) — Spin-Orbit Coupling
