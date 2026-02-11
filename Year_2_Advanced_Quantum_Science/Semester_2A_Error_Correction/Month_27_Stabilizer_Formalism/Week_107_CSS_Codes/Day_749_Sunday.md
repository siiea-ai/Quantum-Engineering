# Day 749: Week 107 Synthesis

## Overview

**Day:** 749 of 1008
**Week:** 107 (CSS Codes & Related Constructions) — Final Day
**Month:** 27 (Stabilizer Formalism)
**Topic:** Comprehensive Review and Integration

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Week review |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Integration problems |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Preparation for Week 108 |

---

## Week 107 Concept Map

```
                        CSS CODES
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
      CONSTRUCTION      EXAMPLES         APPLICATIONS
           │                │                │
      ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
      ▼         ▼      ▼         ▼      ▼         ▼
   C₂⊥⊆C₁   Parameters  Steane   Surface  Transversal  HP
   Condition    k,d      Code     Code      Gates     Codes
      │         │         │         │         │         │
      └─────────┴─────────┴─────────┴─────────┴─────────┘
                            │
                     FAULT TOLERANCE
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
       Transversal      Eastin-Knill      Magic
         CNOT           Theorem          States
```

---

## Week 107 Daily Summary

| Day | Topic | Key Results |
|-----|-------|-------------|
| 743 | CSS Introduction | $C_2^\perp \subseteq C_1$, CSS(C₁,C₂) construction |
| 744 | Dual Containment | Self-orthogonal codes, $H \cdot H^T = 0$ test |
| 745 | CSS Examples | Steane [[7,1,3]], Shor [[9,1,3]], RM codes |
| 746 | Surface Codes | Toric [[2L²,2,L]], planar codes, boundaries |
| 747 | Hypergraph Product | HP: [[n₁n₂+m₁m₂, k₁k₂, min(d₁,d₂)]] |
| 748 | Transversal Gates | X, Z, H, CNOT transversal; Eastin-Knill |

---

## Master Formula Sheet

### CSS Construction

$$\boxed{CSS(C_1, C_2): \quad C_2^\perp \subseteq C_1}$$

$$\boxed{[[n, k_1 + k_2 - n, \min(d(C_1), d(C_2^\perp))]]}$$

### CSS Stabilizers

$$\boxed{S_X = \langle X^{\mathbf{h}} : \mathbf{h} \in C_2^\perp \rangle}$$

$$\boxed{S_Z = \langle Z^{\mathbf{g}} : \mathbf{g} \in C_1^\perp \rangle}$$

### CSS Parity Check

$$\boxed{H = \begin{pmatrix} H_X & 0 \\ 0 & H_Z \end{pmatrix} = \begin{pmatrix} H_{C_2} & 0 \\ 0 & H_{C_1} \end{pmatrix}}$$

### Self-Orthogonal Condition

$$\boxed{C^\perp \subseteq C \Leftrightarrow H \cdot H^T = 0 \pmod 2}$$

### Surface Code (Toric)

$$\boxed{[[2L^2, 2, L]]}$$

$$\boxed{A_v = \prod_{e \ni v} Z_e, \quad B_p = \prod_{e \in \partial p} X_e}$$

### Hypergraph Product

$$\boxed{HP(C_1, C_2) = [[n_1 n_2 + m_1 m_2, k_1 k_2, \min(d_1, d_2)]]}$$

### Transversal Gates

$$\boxed{\bar{X} = X^{\otimes n}, \quad \bar{Z} = Z^{\otimes n}}$$

$$\boxed{\overline{CNOT} = CNOT^{\otimes n}}$$

---

## Comprehensive Problem Set

### Part A: CSS Construction (Days 743-744)

**A1.** Given classical codes C₁ = [9, 5, 3] and C₂ = [9, 4, 4]:
a) Verify that CSS(C₁, C₂) can be valid (dimension check)
b) What is the expected number of logical qubits?
c) What classical condition must be checked?

**A2.** Prove that for a self-orthogonal [n, k, d] code C:
$$CSS(C, C) = [[n, 2k-n, d']]$$
where d' ≥ d(C⊥).

**A3.** Design a CSS code with [[n, 1, 5]] by identifying appropriate classical codes.

### Part B: CSS Examples (Day 745)

**B1.** For the Steane [[7, 1, 3]] code:
a) Write all 6 stabilizer generators
b) Verify they pairwise commute
c) Find two independent logical operators

**B2.** Compare the [[7,1,3]] Steane and [[9,1,3]] Shor codes:
a) Number of stabilizer generators
b) Syndrome bits for X vs Z errors
c) Encoding circuit complexity

**B3.** Can you construct a [[5, 1, 3]] CSS code? Explain why or why not.

### Part C: Surface Codes (Day 746)

**C1.** For a 4×4 toric code:
a) How many qubits?
b) How many X stabilizers? Z stabilizers?
c) What is the code distance?

**C2.** Draw the stabilizers for a 3×3 planar code with:
- Rough boundaries (top/bottom)
- Smooth boundaries (left/right)

**C3.** Prove that the minimum weight logical Z operator in toric code has weight L.

### Part D: Advanced Constructions (Days 747-748)

**D1.** Compute HP parameters for:
a) [7, 4, 3] × [7, 4, 3]
b) [15, 11, 3] × [7, 4, 3]

**D2.** For the Steane code, verify:
a) X^⊗7 is a valid logical X
b) H^⊗7 is a valid logical H
c) Why T^⊗7 is NOT a valid logical T

**D3.** Design a fault-tolerant T gate implementation using magic state injection for the Steane code.

### Part E: Integration

**E1.** Show that every surface code is a CSS code by identifying C₁ and C₂.

**E2.** Prove that the hypergraph product of two self-orthogonal codes gives a valid CSS code.

**E3.** Design a complete fault-tolerant protocol for:
- Encoding a qubit in Steane code
- Performing a logical CNOT with another Steane block
- Measuring the logical Z operator

---

## Solutions to Selected Problems

### Solution A1

a) **Dimension check:**
For CSS(C₁, C₂): k = k₁ + k₂ - n = 5 + 4 - 9 = 0

This gives 0 logical qubits—a valid code but not useful for encoding.

b) **Logical qubits:** k = 0

c) **Condition:** Must verify $C_2^\perp \subseteq C_1$.
- $C_2^\perp$ has dimension 9 - 4 = 5
- $C_1$ has dimension 5
- Need $C_2^\perp = C_1$ or $C_2^\perp \subsetneq C_1$

### Solution B1

**Steane code stabilizers:**

X stabilizers (from [7,3,4] dual code):
- $S_X^{(1)} = IIIXXXX$
- $S_X^{(2)} = IXXIIXX$
- $S_X^{(3)} = XIXIXIX$

Z stabilizers (same pattern):
- $S_Z^{(1)} = IIIZZZZ$
- $S_Z^{(2)} = IZZIIZZ$
- $S_Z^{(3)} = ZIZIZIZ$

**Commutation:** X-X always commute, Z-Z always commute.
X-Z: Check overlap parity. Each pair shares even number of qubits. ✓

**Logical operators:**
- $\bar{X} = X^{\otimes 7}$
- $\bar{Z} = Z^{\otimes 7}$

### Solution C1

**4×4 toric code:**

a) Qubits: n = 2L² = 2(16) = 32

b) Stabilizers:
- Vertices: 16 (but product = I, so 15 independent)
- Faces: 16 (but product = I, so 15 independent)
- Total independent: 30

c) Distance: d = L = 4

**Code:** [[32, 2, 4]]

### Solution D2

**Steane transversal gates:**

a) **X^⊗7 is logical X:**
- Commutes with Z stabilizers (all have even weight 4)
- Anticommutes with Z^⊗7 (overlap = 7, odd)
- ✓ Valid logical X

b) **H^⊗7 is logical H:**
- H swaps X ↔ Z
- X stabilizers → Z stabilizers (same pattern)
- Code is self-dual CSS, so structure preserved
- ✓ Valid logical H

c) **T^⊗7 is NOT logical T:**
- T: X → (X+Y)/√2 (up to phase)
- T^⊗7 doesn't preserve X stabilizer eigenspaces
- Introduces phases that don't cancel
- ✗ Not a valid logical gate

---

## Self-Assessment

### Mastery Checklist

| Skill | Day | Confidence (1-5) |
|-------|-----|------------------|
| CSS construction from classical codes | 743 | ___ |
| Verify dual containment | 744 | ___ |
| Self-orthogonal code recognition | 744 | ___ |
| Steane code stabilizers | 745 | ___ |
| Surface code from lattice | 746 | ___ |
| Toric vs planar distinction | 746 | ___ |
| HP parameter computation | 747 | ___ |
| HP rate analysis | 747 | ___ |
| Transversal gate identification | 748 | ___ |
| Eastin-Knill implications | 748 | ___ |
| Magic state distillation concept | 748 | ___ |

---

## Connections to Future Topics

### Week 108: Code Families

**From CSS Codes:**
- Color codes as CSS codes on colored lattices
- Reed-Muller quantum codes with transversal T
- Triorthogonal codes for magic state distillation

**Key Connection:**
$$\text{CSS structure} \to \text{Specialized constructions} \to \text{Better parameters}$$

### Month 28: Advanced Topics

- Fault-tolerant computation circuits
- Threshold theorems
- Practical implementations

---

## Preparation for Week 108

### Coming Next: Code Families & Construction Techniques

**Topics:**
- Color codes and topological structure
- Reed-Muller quantum codes
- Triorthogonal codes
- Good qLDPC constructions
- Gottesman-Knill theorem

### Prerequisites Check

Ensure mastery of:
- [ ] CSS code construction (Week 107)
- [ ] Dual containment condition
- [ ] Transversal gate analysis
- [ ] Hypergraph products
- [ ] Surface code structure

### Preview Questions

1. What is a color code?
2. Why do Reed-Muller codes have transversal T gates?
3. What are triorthogonal codes used for?
4. What makes a "good" qLDPC code?

---

## Week 107 Complete!

### Summary of Achievements

This week you learned:

1. **CSS Construction:** Building quantum codes from classical codes
2. **Dual Containment:** The key condition $C_2^\perp \subseteq C_1$
3. **Famous Examples:** Steane, Shor, surface codes
4. **Surface Codes:** Toric and planar variants from lattices
5. **Hypergraph Products:** Systematic code construction with constant rate
6. **Transversal Gates:** Fault-tolerant operations and their limits

### Key Insight

CSS codes transform **classical coding theory** into **quantum error correction**:
- Classical codes → Quantum codes
- Classical decoding → Quantum syndrome decoding
- Self-orthogonality → Symmetric CSS codes
- LDPC property → Efficient quantum codes

### Progress

- **Week 107:** 100% complete (7/7 days)
- **Month 27:** 75% complete (21/28 days)
- **Next:** Week 108 — Code Families & Construction Techniques

---

## Computational Synthesis

```python
"""
Week 107 Synthesis: Complete CSS Toolkit
========================================
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def gf2_rank(M: np.ndarray) -> int:
    """Matrix rank over GF(2)."""
    M = M.copy() % 2
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank


class CSSCode:
    """Complete CSS code analysis toolkit."""

    def __init__(self, H_X: np.ndarray, H_Z: np.ndarray, name: str = "CSS"):
        """
        Initialize CSS code with X and Z parity checks.

        Parameters:
        -----------
        H_X : np.ndarray
            X stabilizer check matrix (detects Z errors)
        H_Z : np.ndarray
            Z stabilizer check matrix (detects X errors)
        """
        self.name = name
        self.H_X = np.array(H_X) % 2
        self.H_Z = np.array(H_Z) % 2
        self.n = self.H_X.shape[1]

        # Compute parameters
        rank_X = gf2_rank(self.H_X)
        rank_Z = gf2_rank(self.H_Z)
        self.k = self.n - rank_X - rank_Z

    @classmethod
    def from_classical(cls, H1: np.ndarray, H2: np.ndarray,
                       name: str = "CSS") -> 'CSSCode':
        """Construct CSS from classical parity checks."""
        return cls(H2, H1, name)

    @classmethod
    def steane(cls) -> 'CSSCode':
        """Construct [[7,1,3]] Steane code."""
        H = np.array([
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1]
        ])
        return cls(H, H, "Steane")

    @classmethod
    def shor(cls) -> 'CSSCode':
        """Construct [[9,1,3]] Shor code."""
        H_Z = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1]
        ])
        H_X = np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1]
        ])
        return cls(H_X, H_Z, "Shor")

    @classmethod
    def toric(cls, L: int) -> 'CSSCode':
        """Construct [[2L²,2,L]] toric code."""
        n = 2 * L * L

        # Build star (Z) and plaquette (X) operators
        H_Z = []  # Z stabilizers
        H_X = []  # X stabilizers

        def edge_idx(x, y, d):
            x, y = x % L, y % L
            if d == 'h':
                return y * L + x
            else:
                return L*L + y * L + x

        for y in range(L):
            for x in range(L):
                # Star at (x,y)
                star = np.zeros(n, dtype=int)
                for e in [edge_idx(x,y,'h'), edge_idx(x-1,y,'h'),
                         edge_idx(x,y,'v'), edge_idx(x,y-1,'v')]:
                    star[e] = 1
                H_Z.append(star)

                # Plaquette at (x,y)
                plaq = np.zeros(n, dtype=int)
                for e in [edge_idx(x,y,'h'), edge_idx(x,y+1,'h'),
                         edge_idx(x,y,'v'), edge_idx(x+1,y,'v')]:
                    plaq[e] = 1
                H_X.append(plaq)

        return cls(np.array(H_X), np.array(H_Z), f"Toric-{L}")

    def verify_css(self) -> bool:
        """Verify H_X · H_Z^T = 0."""
        return np.all((self.H_X @ self.H_Z.T) % 2 == 0)

    def x_stabilizers(self) -> List[str]:
        """Return X stabilizers as Pauli strings."""
        return [''.join('X' if b else 'I' for b in row) for row in self.H_X]

    def z_stabilizers(self) -> List[str]:
        """Return Z stabilizers as Pauli strings."""
        return [''.join('Z' if b else 'I' for b in row) for row in self.H_Z]

    def syndrome(self, x_err: np.ndarray, z_err: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute syndromes for X and Z errors."""
        syn_x = (self.H_Z @ x_err) % 2  # X errors detected by Z stabs
        syn_z = (self.H_X @ z_err) % 2  # Z errors detected by X stabs
        return syn_x, syn_z

    def is_transversal_valid(self, gate: str) -> bool:
        """Check if gate^⊗n preserves stabilizers."""
        if gate == 'X':
            # X^⊗n commutes with Z stabs iff all have even weight
            return all(np.sum(row) % 2 == 0 for row in self.H_Z)
        elif gate == 'Z':
            return all(np.sum(row) % 2 == 0 for row in self.H_X)
        elif gate == 'H':
            # H^⊗n valid iff self-dual (H_X ↔ H_Z)
            return (self.H_X.shape == self.H_Z.shape and
                    np.array_equal(np.sort(self.H_X, axis=0),
                                  np.sort(self.H_Z, axis=0)))
        return False

    def __repr__(self) -> str:
        return f"{self.name} [[{self.n}, {self.k}]] CSS code"


class HypergraphProduct:
    """Hypergraph product construction."""

    def __init__(self, H1: np.ndarray, H2: np.ndarray):
        self.H1 = np.array(H1) % 2
        self.H2 = np.array(H2) % 2
        self.m1, self.n1 = self.H1.shape
        self.m2, self.n2 = self.H2.shape

        self.k1 = self.n1 - gf2_rank(self.H1)
        self.k2 = self.n2 - gf2_rank(self.H2)

        self.n = self.n1 * self.n2 + self.m1 * self.m2
        self.k = self.k1 * self.k2

    def rate(self) -> float:
        return self.k / self.n if self.n > 0 else 0

    def __repr__(self) -> str:
        return f"HP [[{self.n}, {self.k}, ?]]"


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Week 107 Synthesis: CSS Code Toolkit")
    print("=" * 60)

    # Famous codes
    codes = [CSSCode.steane(), CSSCode.shor(), CSSCode.toric(3)]

    print("\n1. CSS Code Gallery")
    print("-" * 40)
    for code in codes:
        print(f"{code}")
        print(f"  CSS valid: {code.verify_css()}")
        print(f"  Transversal X: {code.is_transversal_valid('X')}")
        print(f"  Transversal H: {code.is_transversal_valid('H')}")

    # Hypergraph products
    print("\n2. Hypergraph Products")
    print("-" * 40)

    H_ham = np.array([[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]])
    hp = HypergraphProduct(H_ham, H_ham)
    print(f"Hamming × Hamming: {hp}")
    print(f"Rate: {hp.rate():.3f}")

    # Syndrome example
    print("\n3. Syndrome Computation")
    print("-" * 40)

    steane = CSSCode.steane()
    x_err = np.array([0, 0, 1, 0, 0, 0, 0])  # X on qubit 3
    z_err = np.array([0, 0, 0, 0, 1, 0, 0])  # Z on qubit 5

    syn_x, syn_z = steane.syndrome(x_err, z_err)
    print(f"X error on qubit 3: syndrome = {syn_x}")
    print(f"Z error on qubit 5: syndrome = {syn_z}")

    print("\n" + "=" * 60)
    print("Week 107 Complete! CSS codes mastered.")
    print("=" * 60)
```

---

**Week 107 Complete! Next: Week 108 — Code Families & Construction Techniques**
