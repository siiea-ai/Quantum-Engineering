# Day 306: Clebsch-Gordan Coefficients

## Overview

**Month 11, Week 44, Day 5 — Friday**

Today we master Clebsch-Gordan (CG) coefficients — the unitary transformation connecting uncoupled and coupled angular momentum bases. These coefficients appear throughout atomic physics, nuclear physics, and quantum chemistry. We develop systematic methods for their calculation.

## Learning Objectives

1. Define and interpret Clebsch-Gordan coefficients
2. Derive symmetry relations and selection rules
3. Calculate coefficients using recursion relations
4. Apply to physical problems
5. Use standard tables and computational tools

---

## 1. Definition and Notation

### The Transformation

$$\boxed{|j_1, j_2; j, m\rangle = \sum_{m_1, m_2} C^{jm}_{j_1 m_1; j_2 m_2} |j_1, m_1; j_2, m_2\rangle}$$

The Clebsch-Gordan coefficient:
$$C^{jm}_{j_1 m_1; j_2 m_2} = \langle j_1, m_1; j_2, m_2 | j_1, j_2; j, m \rangle$$

### Alternative Notations

$$\langle j_1 m_1 j_2 m_2 | j m \rangle$$
$$(j_1 m_1 j_2 m_2 | j m)$$
$$C(j_1 j_2 j; m_1 m_2)$$

### Inverse Transformation

$$|j_1, m_1; j_2, m_2\rangle = \sum_{j, m} C^{jm}_{j_1 m_1; j_2 m_2} |j_1, j_2; j, m\rangle$$

---

## 2. Selection Rules

### The $m$-Rule

$$\boxed{C^{jm}_{j_1 m_1; j_2 m_2} = 0 \quad \text{unless} \quad m = m_1 + m_2}$$

This follows from $[J_z, J_{1z} + J_{2z}] = 0$.

### The Triangle Rule

$$\boxed{C^{jm}_{j_1 m_1; j_2 m_2} = 0 \quad \text{unless} \quad |j_1 - j_2| \leq j \leq j_1 + j_2}$$

Equivalently: $j_1, j_2, j$ must form a valid "triangle" (each $\leq$ sum of other two).

### Integer Constraint

$$j_1 + j_2 + j = \text{integer}$$

---

## 3. Symmetry Relations

### Exchange of $j_1$ and $j_2$

$$C^{jm}_{j_2 m_2; j_1 m_1} = (-1)^{j_1 + j_2 - j} C^{jm}_{j_1 m_1; j_2 m_2}$$

### Reversal of $m$ Values

$$C^{j,-m}_{j_1, -m_1; j_2, -m_2} = (-1)^{j_1 + j_2 - j} C^{jm}_{j_1 m_1; j_2 m_2}$$

### Sign Conventions (Condon-Shortley)

1. All CG coefficients are real
2. $C^{j,j}_{j_1,j_1;j_2,j-j_1} > 0$ (highest weight state coefficient positive)

---

## 4. Orthogonality Relations

### First Orthogonality

$$\sum_{m_1, m_2} C^{jm}_{j_1 m_1; j_2 m_2} C^{j'm'}_{j_1 m_1; j_2 m_2} = \delta_{jj'}\delta_{mm'}$$

### Second Orthogonality

$$\sum_{j, m} C^{jm}_{j_1 m_1; j_2 m_2} C^{jm}_{j_1 m_1'; j_2 m_2'} = \delta_{m_1 m_1'}\delta_{m_2 m_2'}$$

---

## 5. Recursion Relations

### Starting Point

The highest weight state:
$$|j_1, j_2; j_1 + j_2, j_1 + j_2\rangle = |j_1, j_1; j_2, j_2\rangle$$

Thus:
$$C^{j_1+j_2, j_1+j_2}_{j_1, j_1; j_2, j_2} = 1$$

### Lowering Recursion

Apply $J_- = J_{1-} + J_{2-}$:

$$\sqrt{j(j+1) - m(m-1)} \, C^{j,m-1}_{j_1 m_1; j_2 m_2}$$
$$= \sqrt{j_1(j_1+1) - m_1(m_1+1)} \, C^{jm}_{j_1, m_1+1; j_2 m_2}$$
$$+ \sqrt{j_2(j_2+1) - m_2(m_2+1)} \, C^{jm}_{j_1 m_1; j_2, m_2+1}$$

### Raising Recursion

$$\sqrt{j(j+1) - m(m+1)} \, C^{j,m+1}_{j_1 m_1; j_2 m_2}$$
$$= \sqrt{j_1(j_1+1) - m_1(m_1-1)} \, C^{jm}_{j_1, m_1-1; j_2 m_2}$$
$$+ \sqrt{j_2(j_2+1) - m_2(m_2-1)} \, C^{jm}_{j_1 m_1; j_2, m_2-1}$$

---

## 6. Special Cases

### $j_2 = 1/2$ (Adding Spin-1/2)

$$C^{j_1+1/2,m}_{j_1, m-1/2; 1/2, 1/2} = \sqrt{\frac{j_1 + m + 1/2}{2j_1 + 1}}$$

$$C^{j_1+1/2,m}_{j_1, m+1/2; 1/2, -1/2} = \sqrt{\frac{j_1 - m + 1/2}{2j_1 + 1}}$$

$$C^{j_1-1/2,m}_{j_1, m-1/2; 1/2, 1/2} = -\sqrt{\frac{j_1 - m + 1/2}{2j_1 + 1}}$$

$$C^{j_1-1/2,m}_{j_1, m+1/2; 1/2, -1/2} = \sqrt{\frac{j_1 + m + 1/2}{2j_1 + 1}}$$

### $j_2 = 1$ (Adding Spin-1)

For $j = j_1 + 1, j_1, j_1 - 1$, explicit formulas exist but are more complex.

### Two Spin-1/2 Particles

| $j$ | $m$ | Coefficient | State |
|-----|-----|-------------|-------|
| 1 | 1 | $C^{1,1}_{1/2,1/2;1/2,1/2} = 1$ | $\|\uparrow\uparrow\rangle$ |
| 1 | 0 | $C^{1,0}_{1/2,1/2;1/2,-1/2} = 1/\sqrt{2}$ | |
| 1 | 0 | $C^{1,0}_{1/2,-1/2;1/2,1/2} = 1/\sqrt{2}$ | |
| 1 | -1 | $C^{1,-1}_{1/2,-1/2;1/2,-1/2} = 1$ | $\|\downarrow\downarrow\rangle$ |
| 0 | 0 | $C^{0,0}_{1/2,1/2;1/2,-1/2} = 1/\sqrt{2}$ | |
| 0 | 0 | $C^{0,0}_{1/2,-1/2;1/2,1/2} = -1/\sqrt{2}$ | |

---

## 7. The Wigner 3j Symbol

### Definition

$$\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix} = \frac{(-1)^{j_1 - j_2 - m_3}}{\sqrt{2j_3 + 1}} C^{j_3, -m_3}_{j_1 m_1; j_2 m_2}$$

### Advantages

More symmetric properties:
$$\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix} = 0 \quad \text{unless} \quad m_1 + m_2 + m_3 = 0$$

### Cyclic Symmetry

$$\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix} = \begin{pmatrix} j_2 & j_3 & j_1 \\ m_2 & m_3 & m_1 \end{pmatrix} = \begin{pmatrix} j_3 & j_1 & j_2 \\ m_3 & m_1 & m_2 \end{pmatrix}$$

### Exchange Symmetry

$$\begin{pmatrix} j_2 & j_1 & j_3 \\ m_2 & m_1 & m_3 \end{pmatrix} = (-1)^{j_1 + j_2 + j_3} \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}$$

---

## 8. Explicit Formula (Racah)

### The General Formula

$$C^{jm}_{j_1 m_1; j_2 m_2} = \delta_{m, m_1+m_2} \sqrt{2j+1} \Delta(j_1 j_2 j)$$
$$\times \sqrt{(j_1+m_1)!(j_1-m_1)!(j_2+m_2)!(j_2-m_2)!(j+m)!(j-m)!}$$
$$\times \sum_k \frac{(-1)^k}{k!(j_1+j_2-j-k)!(j_1-m_1-k)!(j_2+m_2-k)!(j-j_2+m_1+k)!(j-j_1-m_2+k)!}$$

where:
$$\Delta(j_1 j_2 j) = \sqrt{\frac{(j_1+j_2-j)!(j_1-j_2+j)!(-j_1+j_2+j)!}{(j_1+j_2+j+1)!}}$$

The sum runs over all $k$ giving non-negative factorials.

---

## 9. Computational Lab

```python
"""
Day 306: Clebsch-Gordan Coefficients
"""

import numpy as np
from scipy.special import factorial
from fractions import Fraction
import sympy as sp
from sympy.physics.quantum.cg import CG, cg_simp
from sympy.physics.wigner import wigner_3j

def clebsch_gordan(j1, m1, j2, m2, j, m):
    """
    Compute Clebsch-Gordan coefficient <j1 m1; j2 m2|j m>.
    Uses Racah formula.
    """
    # Selection rules
    if m != m1 + m2:
        return 0.0
    if not (abs(j1 - j2) <= j <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m) > j:
        return 0.0

    # Triangle coefficient
    def delta(j1, j2, j):
        num = factorial(j1+j2-j) * factorial(j1-j2+j) * factorial(-j1+j2+j)
        den = factorial(j1+j2+j+1)
        return np.sqrt(num / den)

    # Prefactor
    prefactor = np.sqrt(2*j + 1) * delta(j1, j2, j)
    prefactor *= np.sqrt(
        factorial(j1+m1) * factorial(j1-m1) *
        factorial(j2+m2) * factorial(j2-m2) *
        factorial(j+m) * factorial(j-m)
    )

    # Sum over k
    total = 0.0
    for k in range(100):  # Large enough range
        denom_args = [
            k,
            j1+j2-j-k,
            j1-m1-k,
            j2+m2-k,
            j-j2+m1+k,
            j-j1-m2+k
        ]
        # Check all arguments are non-negative integers
        if all(arg >= 0 and arg == int(arg) for arg in denom_args):
            denom = 1.0
            for arg in denom_args:
                denom *= factorial(int(arg))
            total += (-1)**k / denom

    return prefactor * total


def generate_cg_table(j1, j2):
    """Generate complete CG coefficient table for given j1, j2."""
    j_values = np.arange(abs(j1-j2), j1+j2+1)

    print(f"Clebsch-Gordan coefficients for j1={j1}, j2={j2}")
    print("=" * 60)

    for j in j_values:
        print(f"\n--- j = {j} ---")
        for m in np.arange(j, -j-1, -1):
            print(f"\n|{j},{m}⟩ =")
            for m1 in np.arange(j1, -j1-1, -1):
                m2 = m - m1
                if abs(m2) <= j2:
                    coeff = clebsch_gordan(j1, m1, j2, m2, j, m)
                    if abs(coeff) > 1e-10:
                        # Express as fraction if close to simple form
                        frac = Fraction(coeff**2).limit_denominator(100)
                        sign = '+' if coeff > 0 else '-'
                        print(f"  {sign} √({frac.numerator}/{frac.denominator}) |{m1},{m2}⟩")


def verify_orthogonality(j1, j2):
    """Verify orthogonality relations."""
    print("\n" + "=" * 60)
    print("ORTHOGONALITY VERIFICATION")
    print("=" * 60)

    j_values = np.arange(abs(j1-j2), j1+j2+1)

    # First orthogonality: sum over m1, m2
    print("\nFirst orthogonality (sum over m1, m2):")
    for j in j_values:
        for jp in j_values:
            for m in np.arange(j, -j-1, -1):
                for mp in np.arange(jp, -jp-1, -1):
                    total = 0.0
                    for m1 in np.arange(j1, -j1-1, -1):
                        m2 = m - m1
                        m2p = mp - m1
                        if abs(m2) <= j2 and abs(m2p) <= j2:
                            total += (clebsch_gordan(j1, m1, j2, m2, j, m) *
                                    clebsch_gordan(j1, m1, j2, m2p, jp, mp))

                    expected = 1.0 if (j == jp and m == mp) else 0.0
                    if abs(total - expected) > 1e-10:
                        print(f"  FAIL: j={j},m={m}; j'={jp},m'={mp}: "
                              f"got {total:.6f}, expected {expected}")

    print("  All first orthogonality relations verified ✓")


def recursion_example():
    """Demonstrate recursion relation."""
    print("\n" + "=" * 60)
    print("RECURSION RELATION EXAMPLE")
    print("=" * 60)

    j1, j2 = 1, 0.5

    print(f"\nFor j1={j1}, j2={j2}:")
    print("\nStarting from |j1+j2, j1+j2⟩ = |j1,j1;j2,j2⟩:")
    print("  |3/2, 3/2⟩ = |1,1;1/2,1/2⟩")
    print(f"  C = {clebsch_gordan(j1, j1, j2, j2, j1+j2, j1+j2):.4f}")

    print("\nApplying J- to get |3/2, 1/2⟩:")
    m = 0.5
    j = 1.5
    for m1 in np.arange(j1, -j1-1, -1):
        m2 = m - m1
        if abs(m2) <= j2:
            coeff = clebsch_gordan(j1, m1, j2, m2, j, m)
            if abs(coeff) > 1e-10:
                print(f"  C({j1},{m1};{j2},{m2}|{j},{m}) = {coeff:.4f}")


def sympy_cg_demo():
    """Demonstrate SymPy CG calculations."""
    print("\n" + "=" * 60)
    print("SYMPY CLEBSCH-GORDAN")
    print("=" * 60)

    # Define symbolic j values
    j1, m1 = sp.Rational(1), sp.Rational(0)
    j2, m2 = sp.Rational(1,2), sp.Rational(1,2)
    j, m = sp.Rational(3,2), sp.Rational(1,2)

    cg = CG(j1, m1, j2, m2, j, m)
    print(f"\n<1,0; 1/2,1/2 | 3/2,1/2> = {cg.doit()}")

    # Wigner 3j symbol
    threej = wigner_3j(1, sp.Rational(1,2), sp.Rational(3,2),
                       0, sp.Rational(1,2), sp.Rational(-1,2))
    print(f"\n3j symbol (1 1/2 3/2; 0 1/2 -1/2) = {threej}")


def physical_application():
    """Apply CG coefficients to physical problem."""
    print("\n" + "=" * 60)
    print("PHYSICAL APPLICATION: p-ELECTRON STATES")
    print("=" * 60)

    # p-electron: l=1, s=1/2
    # Coupled states: j=3/2 or j=1/2

    print("\nCoupled states |l,s;j,mj⟩ in terms of |ml,ms⟩:")

    # j = 3/2 states
    print("\n--- j = 3/2 (quartet) ---")
    for mj in [1.5, 0.5, -0.5, -1.5]:
        print(f"|3/2, {mj:+.1f}⟩ = ", end="")
        terms = []
        for ml in [1, 0, -1]:
            ms = mj - ml
            if abs(ms) <= 0.5:
                coeff = clebsch_gordan(1, ml, 0.5, ms, 1.5, mj)
                if abs(coeff) > 1e-10:
                    sign = "+" if coeff > 0 and terms else ""
                    frac = Fraction(abs(coeff)**2).limit_denominator(10)
                    sqrt_str = f"√({frac.numerator}/{frac.denominator})" if frac != 1 else "1"
                    if coeff < 0:
                        sqrt_str = "-" + sqrt_str
                    terms.append(f"{sign}{sqrt_str}|{ml},{'+' if ms > 0 else ''}{ms:.1f}⟩")
        print(" ".join(terms))

    # j = 1/2 states
    print("\n--- j = 1/2 (doublet) ---")
    for mj in [0.5, -0.5]:
        print(f"|1/2, {mj:+.1f}⟩ = ", end="")
        terms = []
        for ml in [1, 0, -1]:
            ms = mj - ml
            if abs(ms) <= 0.5:
                coeff = clebsch_gordan(1, ml, 0.5, ms, 0.5, mj)
                if abs(coeff) > 1e-10:
                    sign = "+" if coeff > 0 and terms else ""
                    frac = Fraction(abs(coeff)**2).limit_denominator(10)
                    sqrt_str = f"√({frac.numerator}/{frac.denominator})"
                    if coeff < 0:
                        sqrt_str = "-" + sqrt_str
                    terms.append(f"{sign}{sqrt_str}|{ml},{'+' if ms > 0 else ''}{ms:.1f}⟩")
        print(" ".join(terms))


# Main execution
if __name__ == "__main__":
    # Two spin-1/2 table
    generate_cg_table(0.5, 0.5)

    # Spin-1 plus spin-1/2
    generate_cg_table(1, 0.5)

    # Orthogonality verification
    verify_orthogonality(0.5, 0.5)

    # Recursion example
    recursion_example()

    # Physical application
    physical_application()

    # SymPy demo
    try:
        sympy_cg_demo()
    except ImportError:
        print("SymPy physics module not available")
```

---

## 10. Practice Problems

### Problem 1: Direct Calculation

Calculate $C^{1,0}_{1,1;1,-1}$ using the recursion relation, starting from $C^{2,2}_{1,1;1,1} = 1$.

### Problem 2: Orthogonality

Verify that $\sum_{m_1+m_2=0} |C^{1,0}_{1/2,m_1;1/2,m_2}|^2 = 1$.

### Problem 3: Symmetry

Using the symmetry relation, express $C^{j,m}_{j_2,m_2;j_1,m_1}$ in terms of $C^{j,m}_{j_1,m_1;j_2,m_2}$.

### Problem 4: 3j Symbol

Convert the CG coefficient $\langle 1,1;1,-1|2,0\rangle$ to a 3j symbol.

### Problem 5: d-Electron

For a d-electron ($\ell = 2$, $s = 1/2$), write $|j=5/2, m_j=5/2\rangle$ in the uncoupled basis.

---

## Summary

### Clebsch-Gordan Coefficients

$$\boxed{|j_1, j_2; j, m\rangle = \sum_{m_1, m_2} C^{jm}_{j_1 m_1; j_2 m_2} |j_1, m_1; j_2, m_2\rangle}$$

### Key Properties

| Property | Formula |
|----------|---------|
| Selection rule | $m = m_1 + m_2$ |
| Triangle rule | $\|j_1 - j_2\| \leq j \leq j_1 + j_2$ |
| Reality | All CG coefficients are real |
| Orthogonality | $\sum_{m_1,m_2} C^{jm}_{j_1m_1;j_2m_2} C^{j'm'}_{j_1m_1;j_2m_2} = \delta_{jj'}\delta_{mm'}$ |

### Standard Sources

- PDG (Particle Data Group): Table of CG coefficients
- Condon & Shortley: Theory of Atomic Spectra
- Varshalovich et al.: Quantum Theory of Angular Momentum

---

## Preview: Day 307

Tomorrow we apply angular momentum theory to **physical systems**: selection rules in atomic spectra, the Wigner-Eckart theorem, and matrix element calculations.
