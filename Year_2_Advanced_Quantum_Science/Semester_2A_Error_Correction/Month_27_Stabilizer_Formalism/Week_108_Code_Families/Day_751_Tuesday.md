# Day 751: Reed-Muller Quantum Codes

## Overview

**Day:** 751 of 1008
**Week:** 108 (Code Families & Construction Techniques)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Quantum Reed-Muller Codes and Transversal T

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Classical RM codes |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Quantum RM and transversal T |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Construct** classical Reed-Muller codes RM(r, m)
2. **Compute** RM code parameters and dual relationships
3. **Build** quantum codes from Reed-Muller codes
4. **Explain** why certain RM codes have transversal T gates
5. **Apply** RM codes to magic state distillation
6. **Compare** RM codes with other CSS constructions

---

## Classical Reed-Muller Codes

### Definition

The **Reed-Muller code RM(r, m)** is defined by Boolean polynomials of degree ≤ r in m variables.

**Parameters:**
$$\boxed{RM(r, m) = [2^m, \sum_{i=0}^r \binom{m}{i}, 2^{m-r}]}$$

- Block length: $n = 2^m$
- Dimension: $k = \sum_{i=0}^r \binom{m}{i}$
- Distance: $d = 2^{m-r}$

### Generator Matrix Construction

**Evaluation vectors:** For each monomial of degree ≤ r, the codeword is its evaluation over all $2^m$ points in $\mathbb{F}_2^m$.

**Example: RM(1, 3) = [8, 4, 4]**

Variables: $x_1, x_2, x_3$
Monomials of degree ≤ 1: $\{1, x_1, x_2, x_3\}$

Evaluation at all 8 points:
| Point | 1 | $x_1$ | $x_2$ | $x_3$ |
|-------|---|-------|-------|-------|
| 000 | 1 | 0 | 0 | 0 |
| 001 | 1 | 0 | 0 | 1 |
| 010 | 1 | 0 | 1 | 0 |
| 011 | 1 | 0 | 1 | 1 |
| 100 | 1 | 1 | 0 | 0 |
| 101 | 1 | 1 | 0 | 1 |
| 110 | 1 | 1 | 1 | 0 |
| 111 | 1 | 1 | 1 | 1 |

Generator matrix:
$$G = \begin{pmatrix}
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 \\
0 & 1 & 0 & 1 & 0 & 1 & 0 & 1
\end{pmatrix}$$

### Dual Code Relationship

**Key Property:**
$$\boxed{RM(r, m)^\perp = RM(m-r-1, m)}$$

**Examples:**
- $RM(0, 3)^\perp = RM(2, 3)$
- $RM(1, 3)^\perp = RM(1, 3)$ (self-dual!)
- $RM(1, 4)^\perp = RM(2, 4)$

### Self-Orthogonality

RM(r, m) is self-orthogonal when:
$$RM(m-r-1, m) \subseteq RM(r, m)$$

This holds when $m - r - 1 \leq r$, i.e., $r \geq (m-1)/2$.

---

## Quantum Reed-Muller Codes

### CSS Construction

Using the CSS framework with C₁ = C₂ = RM(r, m):

If RM(r, m) is self-orthogonal:
$$CSS(RM(r,m), RM(r,m)) = [[2^m, k_{CSS}, d_{CSS}]]$$

**Dimension:**
$$k_{CSS} = 2 \cdot \sum_{i=0}^r \binom{m}{i} - 2^m$$

### Example: [[8, 0, 4]] from RM(1, 3)

RM(1, 3) = [8, 4, 4] is self-dual.

CSS(RM(1,3), RM(1,3)):
- n = 8
- k = 2·4 - 8 = 0

No logical qubits! The code is a stabilizer state.

### The [[15, 1, 3]] Reed-Muller Code

**Construction:** Use RM(1, 4) and RM(2, 4).

RM(1, 4) = [16, 5, 8]
RM(2, 4) = [16, 11, 4]

$RM(1, 4)^\perp = RM(2, 4)$

**Punctured code:** Remove one coordinate to get [15, ...].

The punctured construction gives the **[[15, 1, 3]]** quantum Reed-Muller code.

---

## Transversal T Gate

### The Remarkable Property

**Theorem:** The [[15, 1, 3]] quantum Reed-Muller code has a **transversal T gate**.

$$\bar{T} = T^{\otimes 15}$$

This is exceptional! Most codes (including Steane and surface codes) do NOT have transversal T.

### Why T is Transversal

The T gate transforms stabilizers as:
$$T X T^\dagger = \frac{1}{\sqrt{2}}(X + Y) = e^{i\pi/4} P_+ X$$
$$T Z T^\dagger = Z$$

where $P_+ = \frac{1}{2}(I + Z)$ is projection onto |0⟩.

For $T^{\otimes n}$ to preserve stabilizers:

**X stabilizers:** $X^{\mathbf{a}} \to e^{i\theta} X^{\mathbf{a}} Z^{\mathbf{b}}$

The phase $e^{i\theta}$ must be consistent, and $Z^{\mathbf{b}}$ must be a stabilizer.

**Condition:** The code must have special **doubly-even** structure.

### Doubly-Even Property

A code is **doubly-even** if all codeword weights are divisible by 4.

RM(1, m) codes have this property for m ≥ 3.

For doubly-even CSS codes with appropriate structure, $T^{\otimes n}$ can be transversal.

---

## Magic State Distillation

### T States

The **T state** (or magic state):
$$|T\rangle = T|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$

With T states and Clifford operations, we can implement T gates on encoded qubits.

### 15-to-1 Distillation

**Protocol:**
1. Prepare 15 noisy |T⟩ states with error rate ε
2. Encode into [[15, 1, 3]] RM code
3. Measure X and Z stabilizers
4. If all measurements pass, decode

**Error improvement:** $\epsilon_{\text{out}} = O(\epsilon^3)$

**Overhead:** 15 input states → 1 output state

### Connection to RM Codes

The [[15, 1, 3]] code is specifically designed for T-state distillation:
- Transversal T means T gates don't spread errors
- Distance 3 catches weight-1 errors
- Triorthogonal structure enables the protocol

---

## Code Comparison

### RM-Based Quantum Codes

| Code | Parameters | Transversal Gates | Use |
|------|------------|-------------------|-----|
| [[8, 3, 2]] | From RM(1,3) | Clifford | Encoding |
| [[15, 1, 3]] | Punctured RM | Clifford + T | Distillation |
| [[16, 0, 4]] | RM(1,4) CSS | Stabilizer state | - |
| [[127, 1, 15]] | Higher RM | Clifford + T | High-fidelity |

### Steane vs 15-qubit RM

| Property | Steane [[7,1,3]] | RM [[15,1,3]] |
|----------|------------------|---------------|
| Physical qubits | 7 | 15 |
| Distance | 3 | 3 |
| Transversal H | ✓ | ✓ |
| Transversal T | ✗ | ✓ |
| Use case | General | T distillation |

---

## Worked Examples

### Example 1: Compute RM Parameters

**Problem:** Find parameters of RM(2, 4).

**Solution:**

$n = 2^4 = 16$

$k = \sum_{i=0}^2 \binom{4}{i} = \binom{4}{0} + \binom{4}{1} + \binom{4}{2} = 1 + 4 + 6 = 11$

$d = 2^{4-2} = 4$

**Result:** RM(2, 4) = [16, 11, 4]

### Example 2: Verify Dual Relationship

**Problem:** Show $RM(1, 4)^\perp = RM(2, 4)$.

**Solution:**

RM(1, 4) = [16, 5, 8]
- Dimension: k₁ = 5

Dual dimension: 16 - 5 = 11

RM(2, 4) = [16, 11, 4]
- Dimension: k₂ = 11 ✓

By the RM duality theorem: $RM(r, m)^\perp = RM(m-r-1, m)$

$RM(1, 4)^\perp = RM(4-1-1, 4) = RM(2, 4)$ ✓

### Example 3: CSS from RM Codes

**Problem:** Construct CSS code from RM(2, 4) and RM(1, 4).

**Solution:**

Check containment: $RM(1, 4)^\perp = RM(2, 4) \subseteq RM(2, 4)$ ✓

CSS parameters:
- n = 16
- k = k(RM(2,4)) + k(RM(1,4)) - 16 = 11 + 5 - 16 = 0

No logical qubits—this is a stabilizer state.

---

## Practice Problems

### Level 1: Direct Application

**P1.1** Compute parameters of RM(1, 5) and RM(3, 5).

**P1.2** Write the generator matrix for RM(0, 3) (the repetition code).

**P1.3** Verify that RM(1, 4) is self-dual by checking dimensions.

### Level 2: Intermediate

**P2.1** Prove that $RM(0, m) = $ repetition code [2^m, 1, 2^m].

**P2.2** For the [[15, 1, 3]] code:
a) How many X stabilizers?
b) How many Z stabilizers?
c) What is the total number of stabilizer generators?

**P2.3** Show that RM(r, m) is doubly-even when r ≥ 1 and m ≥ 3.

### Level 3: Challenging

**P3.1** Prove the duality relation $RM(r, m)^\perp = RM(m-r-1, m)$.

**P3.2** Construct the [[15, 1, 3]] code explicitly by puncturing RM codes.

**P3.3** Analyze why $T^{\otimes 15}$ is transversal for [[15, 1, 3]] using the doubly-even property.

---

## Computational Lab

```python
"""
Day 751: Reed-Muller Quantum Codes
==================================

Implementing Reed-Muller code construction and analysis.
"""

import numpy as np
from typing import List, Tuple
from itertools import combinations
from math import comb


def rm_parameters(r: int, m: int) -> Tuple[int, int, int]:
    """
    Compute Reed-Muller code parameters.

    Returns (n, k, d) for RM(r, m).
    """
    n = 2 ** m
    k = sum(comb(m, i) for i in range(r + 1))
    d = 2 ** (m - r)
    return n, k, d


def rm_generator_matrix(r: int, m: int) -> np.ndarray:
    """
    Construct generator matrix for RM(r, m).

    Uses evaluation of monomials at all 2^m points.
    """
    n = 2 ** m
    k = sum(comb(m, i) for i in range(r + 1))

    G = []

    # Enumerate all monomials of degree <= r
    for degree in range(r + 1):
        for variables in combinations(range(m), degree):
            row = np.zeros(n, dtype=int)
            for point in range(n):
                # Evaluate monomial at this point
                bits = [(point >> i) & 1 for i in range(m)]
                value = 1
                for var in variables:
                    value *= bits[var]
                row[point] = value
            G.append(row)

    return np.array(G)


def is_doubly_even(G: np.ndarray) -> bool:
    """Check if all codewords have weight ≡ 0 mod 4."""
    k, n = G.shape
    if k > 20:
        return None  # Too large

    for i in range(2**k):
        coeffs = np.array([(i >> j) & 1 for j in range(k)])
        codeword = (coeffs @ G) % 2
        weight = np.sum(codeword)
        if weight % 4 != 0:
            return False
    return True


def check_self_orthogonal(G: np.ndarray) -> bool:
    """Check if code generated by G is self-orthogonal."""
    return np.all((G @ G.T) % 2 == 0)


class ReedMullerCode:
    """Reed-Muller code analysis."""

    def __init__(self, r: int, m: int):
        self.r = r
        self.m = m
        self.n, self.k, self.d = rm_parameters(r, m)
        self.G = rm_generator_matrix(r, m)

    def parity_check(self) -> np.ndarray:
        """Compute parity check matrix."""
        # H is generator of dual code RM(m-r-1, m)
        r_dual = self.m - self.r - 1
        if r_dual < 0:
            return np.array([]).reshape(0, self.n)
        return rm_generator_matrix(r_dual, self.m)

    def is_self_dual(self) -> bool:
        """Check if RM(r, m) = RM(r, m)^⊥."""
        # Self-dual when 2r + 1 = m
        return 2 * self.r + 1 == self.m

    def is_self_orthogonal(self) -> bool:
        """Check if RM(r, m)^⊥ ⊆ RM(r, m)."""
        # Self-orthogonal when r >= (m-1)/2
        return self.r >= (self.m - 1) / 2

    def dual_parameters(self) -> Tuple[int, int, int]:
        """Parameters of dual code."""
        r_dual = self.m - self.r - 1
        if r_dual < 0:
            return self.n, 0, self.n  # Trivial dual
        return rm_parameters(r_dual, self.m)

    def __repr__(self) -> str:
        return f"RM({self.r}, {self.m}) = [{self.n}, {self.k}, {self.d}]"


def quantum_rm_code(r: int, m: int) -> dict:
    """
    Construct quantum CSS code from RM(r, m).

    Returns code parameters and properties.
    """
    rm = ReedMullerCode(r, m)

    if not rm.is_self_orthogonal():
        return {'valid': False, 'reason': 'Not self-orthogonal'}

    n = rm.n
    k_classical = rm.k
    k_quantum = 2 * k_classical - n

    return {
        'valid': True,
        'n': n,
        'k': k_quantum,
        'classical': (r, m),
        'self_dual': rm.is_self_dual(),
        'doubly_even': is_doubly_even(rm.G)
    }


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 751: Reed-Muller Quantum Codes")
    print("=" * 60)

    # Example 1: RM code parameters
    print("\n1. Reed-Muller Code Parameters")
    print("-" * 40)

    for m in range(3, 6):
        print(f"\nm = {m}:")
        for r in range(m + 1):
            n, k, d = rm_parameters(r, m)
            rm = ReedMullerCode(r, m)
            dual_n, dual_k, dual_d = rm.dual_parameters()
            print(f"  RM({r},{m}) = [{n},{k},{d}], "
                  f"dual = RM({m-r-1},{m}) = [{dual_n},{dual_k},{dual_d}]")

    # Example 2: Generator matrices
    print("\n2. Generator Matrix Examples")
    print("-" * 40)

    for (r, m) in [(0, 3), (1, 3), (1, 4)]:
        rm = ReedMullerCode(r, m)
        print(f"\n{rm}")
        print(f"Generator matrix ({rm.k} × {rm.n}):")
        print(rm.G)

    # Example 3: Self-orthogonality
    print("\n3. Self-Orthogonality Check")
    print("-" * 40)

    for m in range(3, 6):
        for r in range(m + 1):
            rm = ReedMullerCode(r, m)
            so = rm.is_self_orthogonal()
            sd = rm.is_self_dual()
            status = "self-dual" if sd else ("self-orth" if so else "—")
            print(f"RM({r},{m}): {status}")

    # Example 4: Quantum codes
    print("\n4. Quantum Reed-Muller Codes")
    print("-" * 40)

    for (r, m) in [(1, 3), (1, 4), (2, 4), (2, 5)]:
        qrm = quantum_rm_code(r, m)
        if qrm['valid']:
            print(f"RM({r},{m}) → [[{qrm['n']}, {qrm['k']}]] "
                  f"(doubly-even: {qrm['doubly_even']})")
        else:
            print(f"RM({r},{m}) → {qrm['reason']}")

    # Example 5: Doubly-even check
    print("\n5. Doubly-Even Property")
    print("-" * 40)

    for (r, m) in [(1, 3), (1, 4), (2, 4)]:
        rm = ReedMullerCode(r, m)
        de = is_doubly_even(rm.G)
        print(f"RM({r},{m}): doubly-even = {de}")

    # Example 6: The [[15,1,3]] code
    print("\n6. The [[15,1,3]] RM Code")
    print("-" * 40)

    print("This code comes from punctured RM codes.")
    print("Key properties:")
    print("  - Transversal T gate: ✓")
    print("  - Used for magic state distillation")
    print("  - 15 input noisy T states → 1 high-fidelity T state")

    print("\n" + "=" * 60)
    print("Reed-Muller codes: transversal T gates!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| RM parameters | $[2^m, \sum_{i=0}^r \binom{m}{i}, 2^{m-r}]$ |
| Dual relationship | $RM(r,m)^\perp = RM(m-r-1, m)$ |
| Self-orthogonal | $r \geq (m-1)/2$ |
| Self-dual | $2r + 1 = m$ |
| Quantum RM | $k_{CSS} = 2k_{classical} - 2^m$ |

### Main Takeaways

1. **Reed-Muller codes** RM(r, m) have elegant algebraic structure
2. **Dual relationship** connects RM(r, m) to RM(m-r-1, m)
3. **[[15, 1, 3]]** RM code has transversal T gate
4. **Doubly-even** property enables transversal T
5. RM codes are used for **magic state distillation**

---

## Daily Checklist

- [ ] I can compute RM(r, m) parameters
- [ ] I understand the dual code relationship
- [ ] I know when RM codes are self-orthogonal
- [ ] I can explain why [[15,1,3]] has transversal T
- [ ] I understand magic state distillation
- [ ] I can compare RM codes with other CSS codes

---

## Preview: Day 752

Tomorrow we explore **triorthogonal codes**:

- The triorthogonality condition
- Connection to magic state distillation
- Designing codes for T-state preparation
- Beyond Reed-Muller constructions

Triorthogonality is the key algebraic property for non-Clifford fault tolerance!
