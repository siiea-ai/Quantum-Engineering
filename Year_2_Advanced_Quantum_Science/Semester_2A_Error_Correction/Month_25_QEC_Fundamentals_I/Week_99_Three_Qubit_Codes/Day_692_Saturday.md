# Day 692: CSS Code Construction

## Overview

**Week:** 99 (Three-Qubit Codes and Beyond)
**Day:** Saturday
**Date:** Year 2, Month 25, Day 692
**Topic:** Calderbank-Shor-Steane (CSS) Code Construction
**Hours:** 7 (3.5 theory + 2.5 problems + 1 computational lab)

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | CSS construction theorem, classical code pairs |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Code design, distance properties |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | CSS code generator implementation |

---

## Prerequisites

From Days 687-691:
- Stabilizer formalism and Pauli group
- Binary symplectic representation
- Knill-Laflamme conditions
- Steane [[7,1,3]] code structure

---

## Learning Objectives

By the end of this day, you will be able to:

1. **State** the CSS code construction theorem and its requirements
2. **Define** dual-containing classical codes and verify containment
3. **Construct** CSS codes from arbitrary classical code pairs $(C_1, C_2)$
4. **Calculate** the parameters $[[n, k, d]]$ from classical code parameters
5. **Design** CSS codes with specific distance properties
6. **Prove** that CSS codes satisfy the Knill-Laflamme conditions

---

## Core Content

### 1. Historical Context

#### The CSS Discovery

In 1996, independently:
- **Calderbank and Shor** (Bell Labs/AT&T)
- **Steane** (Oxford)

discovered a powerful construction for quantum error correcting codes using pairs of classical linear codes.

**Key Insight:** The structure of classical linear codes can be leveraged to build quantum codes where X and Z errors are corrected independently.

#### Why CSS Codes Matter

1. **Systematic Construction:** Transform classical coding theory expertise to quantum
2. **Structural Simplicity:** X and Z sectors decouple
3. **Fault Tolerance:** Natural transversal gate implementations
4. **Universality:** Many important codes are CSS (Steane, surface, color codes)

---

### 2. Classical Linear Codes Review

#### Linear Code Definition

A classical linear code $C$ over $\mathbb{F}_2$ is a $k$-dimensional subspace of $\mathbb{F}_2^n$.

**Parameters:** $[n, k, d]$
- $n$: Block length
- $k$: Dimension (information bits)
- $d$: Minimum distance

**Generator Matrix:** $G \in \mathbb{F}_2^{k \times n}$ where $C = \{xG : x \in \mathbb{F}_2^k\}$

**Parity-Check Matrix:** $H \in \mathbb{F}_2^{(n-k) \times n}$ where $C = \{c : Hc^T = 0\}$

#### Dual Code

The dual code $C^\perp$ consists of all vectors orthogonal to $C$:

$$C^\perp = \{v \in \mathbb{F}_2^n : \langle v, c \rangle = 0 \text{ for all } c \in C\}$$

**Properties:**
- $\dim(C^\perp) = n - k$
- The generator of $C^\perp$ is the parity-check of $C$
- $(C^\perp)^\perp = C$

#### Dual-Containing Codes

A code $C$ is **dual-containing** (or **weakly self-dual**) if:

$$\boxed{C^\perp \subseteq C}$$

**Equivalent conditions:**
- $HH^T = 0$ (rows of H are orthogonal)
- Every generator of $C^\perp$ is a codeword of $C$
- $C$ contains its dual as a subcode

---

### 3. CSS Code Construction Theorem

#### The Main Theorem

**Theorem (Calderbank-Shor-Steane):**
Let $C_1$ and $C_2$ be classical linear codes with parameters $[n, k_1, d_1]$ and $[n, k_2, d_2]$ respectively, such that $C_2^\perp \subseteq C_1$.

Then there exists a quantum code $CSS(C_1, C_2)$ with parameters:

$$\boxed{[[n, k_1 + k_2 - n, \min(d_1, d_2^\perp)]]}$$

where $d_2^\perp$ is the minimum distance of $C_2^\perp$.

#### Construction Details

**Stabilizer Generators:**

1. **X-type stabilizers:** For each $h \in H_1$ (parity-check rows of $C_1$):
   $$g_X = X^h = X_1^{h_1} X_2^{h_2} \cdots X_n^{h_n}$$

2. **Z-type stabilizers:** For each $g \in G_2$ (generator rows of $C_2$, equivalently $H_2^\perp$):
   $$g_Z = Z^g = Z_1^{g_1} Z_2^{g_2} \cdots Z_n^{g_n}$$

**Why They Commute:**

For $X^h$ and $Z^g$:
$$X^h Z^g = (-1)^{\langle h, g \rangle} Z^g X^h$$

Commutation requires $\langle h, g \rangle = 0 \pmod{2}$.

Since $h \in C_1^\perp$ and $g \in C_2$, and $C_2^\perp \subseteq C_1$ implies $C_1^\perp \subseteq C_2$, we have:
$$\langle h, g \rangle = 0 \quad \text{for all } h \in C_1^\perp, g \in C_2$$

---

### 4. Logical Operators

#### Logical X Operators

Logical X operators are elements of $C_1 \setminus C_2^\perp$ (in $C_1$ but not in the Z-stabilizer generators):

$$\bar{X}_j \in C_1 / C_2^\perp$$

The number of independent logical X operators is $k_1 - (n - k_2) = k_1 + k_2 - n$.

#### Logical Z Operators

Logical Z operators are elements of $C_2 \setminus C_1^\perp$:

$$\bar{Z}_j \in C_2 / C_1^\perp$$

#### Anticommutation

For proper logical operators, we need:
$$\{\bar{X}_j, \bar{Z}_j\} = 0, \quad [\bar{X}_j, \bar{Z}_k] = 0 \text{ for } j \neq k$$

This is guaranteed by the quotient structure.

---

### 5. Special Case: Symmetric CSS Codes

#### Self-Orthogonal Construction

When $C_1 = C_2 = C$ with $C^\perp \subseteq C$ (dual-containing):

$$CSS(C, C) = [[n, 2k - n, d]]$$

**The Steane Code:** Uses the [7,4,3] Hamming code with $C^\perp \subset C$:
$$CSS(Hamming, Hamming) = [[7, 2 \cdot 4 - 7, 3]] = [[7, 1, 3]]$$

#### Self-Dual Codes

If $C = C^\perp$ (self-dual), then $k = n/2$ and:
$$CSS(C, C) = [[n, 0, d]]$$

These codes encode **zero** logical qubits — they're degenerate codes used for other purposes (e.g., magic state preparation).

---

### 6. CSS Code Distance

#### Distance Formula

The distance of a CSS code is:

$$d = \min(d_X, d_Z)$$

where:
- $d_X = $ minimum weight of $C_1 \setminus C_2^\perp$ (distance for Z errors)
- $d_Z = $ minimum weight of $C_2 \setminus C_1^\perp$ (distance for X errors)

#### Asymmetric CSS Codes

We can have $d_X \neq d_Z$, creating codes optimized for asymmetric noise:

$$[[n, k, d_X / d_Z]]$$

Notation: $d_X / d_Z$ means different X and Z distances.

#### Example: Asymmetric Code

Using $C_1 = [7,4,3]$ Hamming and $C_2 = [7,3,4]$ simplex:
- $d_X = 3$ (from Hamming)
- $d_Z = 4$ (from simplex)

Results in $[[7, 0, 3/4]]$ — better Z protection.

---

### 7. CSS Code Examples

#### Example 1: Repetition Code as CSS

Classical repetition code: $C = [n, 1, n]$ (all-zeros and all-ones)

Dual: $C^\perp = [n, n-1, 2]$ (even-weight vectors)

Since $C^\perp \not\subseteq C$, we can't use $CSS(C, C)$.

But with $C_1 = [n, n-1, 2]$ and $C_2 = [n, 1, n]$ where $C_2^\perp = C_1$:
$$CSS(C_1, C_2) = [[n, (n-1) + 1 - n, \min(2, n)]] = [[n, 0, 2]]$$

This is a **quantum repetition code** variant.

#### Example 2: [[15, 7, 3]] Quantum Hamming

Using the $[15, 11, 3]$ Hamming code:
- $C_1 = [15, 11, 3]$
- $C_2^\perp = [15, 4, 8]$ (dual Hamming)

If $C_2 = C_1$, we get:
$$CSS(Hamming_{15}, Hamming_{15}) = [[15, 2 \cdot 11 - 15, 3]] = [[15, 7, 3]]$$

This code encodes 7 logical qubits!

#### Example 3: Surface Code (Preview)

The surface code can be viewed as a CSS code where:
- $C_1$ and $C_2$ come from homology of a torus
- Stabilizers are local (nearest-neighbor)
- Distance scales with lattice size: $d = L$

---

### 8. Encoding CSS Codes

#### Encoding Procedure

For CSS$(C_1, C_2)$ with $k$ logical qubits:

1. **Prepare** logical $|0\rangle^{\otimes k}$: Superposition over $C_2^\perp$ cosets
2. **Apply** encoding unitary based on $C_1$ structure

#### Logical Basis States

$$|x_L\rangle = \frac{1}{\sqrt{|C_2^\perp|}} \sum_{c \in C_2^\perp} |c \oplus x\rangle$$

where $x$ represents the logical information (a coset representative of $C_1 / C_2^\perp$).

#### Circuit Construction

For CSS codes, encoding typically requires:
- Hadamard gates (for X-type superposition)
- CNOT gates (for spreading correlations)
- Complexity: $O(n^2)$ gates in general

---

### 9. Error Correction in CSS Codes

#### Decoupled Correction

**Key Property:** X and Z errors can be corrected independently!

1. **Z error syndrome:** Measure X-stabilizers
   - Syndrome $\rightarrow$ classical decoding in $C_2$
   - Apply correction Z operator

2. **X error syndrome:** Measure Z-stabilizers
   - Syndrome $\rightarrow$ classical decoding in $C_1$
   - Apply correction X operator

#### Why Independence Works

For a CSS code:
- X errors anticommute only with Z-stabilizers
- Z errors anticommute only with X-stabilizers

Y errors produce combined syndromes: $\text{syn}(Y) = \text{syn}(X) \oplus \text{syn}(Z)$

---

## Quantum Mechanics Connection

### Classical-Quantum Bridge

The CSS construction reveals a fundamental bridge:

| Classical | Quantum |
|-----------|---------|
| Linear code $C$ | Stabilizer code |
| Parity-check $H$ | Stabilizer generators |
| Codeword | Code space basis state |
| Dual $C^\perp$ | Logical operator constraints |
| Syndrome decoding | Quantum error correction |

### Topological Connection

CSS codes with local stabilizers lead to:
- **Toric codes** (Kitaev)
- **Surface codes** (generalized)
- **Color codes** (2D and 3D)

These have topological protection — errors must create macroscopic loops to be undetectable.

---

## Worked Examples

### Example 1: Verify Steane as CSS

**Problem:** Show that the Steane code is $CSS(C, C)$ with $C = [7,4,3]$ Hamming.

**Solution:**

1. **Dual containment:** The [7,4,3] Hamming code has dual [7,3,4].

   We need to verify $C^\perp \subseteq C$.

   The dual codewords are generated by the parity-check matrix $H$.

   All rows of $H$ (which generate $C^\perp$) are codewords of $C$ because the Hamming code contains all weight-3 and weight-4 vectors that satisfy its parity constraints.

2. **Parameter calculation:**
   $$[[n, 2k - n, d]] = [[7, 2(4) - 7, 3]] = [[7, 1, 3]]$$ ✓

3. **Stabilizers:**
   - X-stabilizers from $H$: weight-4 X operators
   - Z-stabilizers from $H$: weight-4 Z operators

**Conclusion:** Steane code is $CSS([7,4,3], [7,4,3])$.

---

### Example 2: Design a [[15,1,3]] CSS Code

**Problem:** Construct a CSS code with $n=15$, $k=1$, $d \geq 3$.

**Solution:**

Need classical codes with:
- $k_1 + k_2 - n = 1 \Rightarrow k_1 + k_2 = 16$
- $C_2^\perp \subseteq C_1$
- $\min(d_1, d_2^\perp) \geq 3$

**Option 1:** Use $C_1 = C_2 = C$ (symmetric).
- Need $2k - 15 = 1 \Rightarrow k = 8$
- Need $[15, 8, d]$ code with $C^\perp \subseteq C$

The Reed-Muller code $RM(1, 4) = [16, 5, 8]$ shortened to [15, 4, 8] doesn't work.

**Option 2:** Punctured Reed-Muller
- $RM(2, 4)^* = [15, 11, 3]$ and $RM(1, 4)^* = [15, 5, 7]$
- $C_1 = [15, 11, 3]$, $C_2 = [15, 11, 3]$ (same code)
- Parameters: $[[15, 2(11) - 15, 3]] = [[15, 7, 3]]$

For $k = 1$, we need different approach...

**Option 3:** Concatenation (conceptual)
- Use distance-3 outer code
- Results in overhead

---

### Example 3: Syndrome Calculation

**Problem:** For $CSS(C_1, C_2)$, show that a Z error at position $j$ produces syndrome equal to column $j$ of $H_1$.

**Solution:**

The X-stabilizers are $\{X^{h_i}\}$ where $h_i$ is row $i$ of $H_1$.

For error $Z_j$:
$$X^{h_i} Z_j = (-1)^{h_{i,j}} Z_j X^{h_i}$$

The syndrome bit is $h_{i,j}$, the $(i,j)$ entry of $H_1$.

Collecting all syndrome bits: $\text{syndrome} = H_1 e_j^T = $ column $j$ of $H_1$.

**Interpretation:** Classical syndrome decoding directly applies!

---

## Practice Problems

### Level 1: Direct Application

1. **Parameter Calculation:**
   Given $[8, 4, 4]$ extended Hamming code with $C^\perp = C$ (self-dual), calculate the CSS code parameters.

2. **Dual Containment Check:**
   The [5, 2, 3] code has generator $G = \begin{pmatrix} 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 & 1 \end{pmatrix}$. Check if $C^\perp \subseteq C$.

3. **Stabilizer Count:**
   A CSS code uses $C_1 = [n, k_1]$ and $C_2 = [n, k_2]$. How many X-stabilizers? How many Z-stabilizers?

### Level 2: Intermediate

4. **Asymmetric Design:**
   Design a CSS code with $d_X = 5$ and $d_Z = 3$ using appropriate classical codes.

5. **Encoding Rate:**
   Compare the encoding rates of $[[7,1,3]]$, $[[15,7,3]]$, and $[[23,1,7]]$ CSS codes.

6. **Syndrome Mapping:**
   For the Steane code, show that the syndrome for $X_3 Z_5$ equals $\text{syn}(X_3) \oplus \text{syn}(Z_5)$.

### Level 3: Challenging

7. **Self-Orthogonality Proof:**
   Prove that if $C$ is a $[2^m - 1, 2^m - 1 - m, 3]$ Hamming code, then $C^\perp \subset C$.

8. **CSS from BCH:**
   The [15, 5, 7] BCH code has dual [15, 10, 4]. Can you construct a valid CSS code from these? Calculate parameters.

9. **Distance Bound:**
   Prove that for any CSS$(C_1, C_2)$ code: $d \leq d_1$ and $d \leq d_2^\perp$.

---

## Computational Lab

### CSS Code Construction Framework

```python
"""
Day 692 Computational Lab: CSS Code Construction
General framework for building CSS codes from classical linear codes
"""

import numpy as np
from typing import Tuple, List, Optional
from itertools import combinations
import matplotlib.pyplot as plt

class ClassicalLinearCode:
    """Represents a classical linear code over F_2."""

    def __init__(self, generator_matrix: np.ndarray):
        """
        Initialize from generator matrix G.

        Args:
            generator_matrix: k x n binary matrix
        """
        self.G = np.array(generator_matrix, dtype=int) % 2
        self.k = self.G.shape[0]
        self.n = self.G.shape[1]
        self.H = self._compute_parity_check()
        self._codewords = None
        self._distance = None

    def _compute_parity_check(self) -> np.ndarray:
        """Compute parity-check matrix H such that GH^T = 0."""
        # Use Gaussian elimination to find null space
        # For simplicity, assume G is in systematic form [I_k | P]
        # Then H = [-P^T | I_{n-k}] = [P^T | I_{n-k}] over F_2

        # General method: find basis of null space of G
        augmented = np.hstack([self.G.T, np.eye(self.n, dtype=int)])

        # Gaussian elimination
        pivot_row = 0
        for col in range(self.k):
            # Find pivot
            for row in range(pivot_row, self.n):
                if augmented[row, col] == 1:
                    # Swap rows
                    augmented[[pivot_row, row]] = augmented[[row, pivot_row]]
                    break
            else:
                continue

            # Eliminate
            for row in range(self.n):
                if row != pivot_row and augmented[row, col] == 1:
                    augmented[row] = (augmented[row] + augmented[pivot_row]) % 2

            pivot_row += 1

        # Null space generators are rows where first k columns are zero
        null_space_rows = []
        for row in range(self.n):
            if np.sum(augmented[row, :self.k]) == 0:
                null_space_rows.append(augmented[row, self.k:])

        if len(null_space_rows) == 0:
            return np.zeros((0, self.n), dtype=int)

        return np.array(null_space_rows, dtype=int) % 2

    @property
    def codewords(self) -> np.ndarray:
        """Generate all 2^k codewords."""
        if self._codewords is None:
            codewords = []
            for i in range(2**self.k):
                info_bits = np.array([int(b) for b in format(i, f'0{self.k}b')], dtype=int)
                codeword = np.dot(info_bits, self.G) % 2
                codewords.append(codeword)
            self._codewords = np.array(codewords)
        return self._codewords

    @property
    def distance(self) -> int:
        """Compute minimum distance."""
        if self._distance is None:
            min_weight = self.n + 1
            for cw in self.codewords:
                weight = np.sum(cw)
                if 0 < weight < min_weight:
                    min_weight = weight
            self._distance = min_weight if min_weight <= self.n else 0
        return self._distance

    def dual(self) -> 'ClassicalLinearCode':
        """Return the dual code C^⊥."""
        return ClassicalLinearCode(self.H)

    def contains(self, other: 'ClassicalLinearCode') -> bool:
        """Check if this code contains the other code as a subcode."""
        # other ⊆ self iff every codeword of other is in self
        for cw in other.codewords:
            # Check if cw satisfies self's parity checks
            syndrome = np.dot(self.H, cw) % 2
            if np.any(syndrome != 0):
                return False
        return True

    def __repr__(self):
        return f"[{self.n}, {self.k}, {self.distance}] code"


class CSSCode:
    """CSS quantum code constructed from two classical codes."""

    def __init__(self, C1: ClassicalLinearCode, C2: ClassicalLinearCode):
        """
        Construct CSS(C1, C2).

        Requirement: C2^⊥ ⊆ C1

        Args:
            C1: Classical code for X-stabilizers
            C2: Classical code for Z-stabilizers
        """
        self.C1 = C1
        self.C2 = C2

        # Verify CSS condition
        C2_dual = C2.dual()
        if not C1.contains(C2_dual):
            raise ValueError("CSS condition violated: C2^⊥ ⊄ C1")

        self.n = C1.n
        self.k = C1.k + C2.k - self.n

        # Compute distance (minimum over X and Z)
        self._compute_distance()

    def _compute_distance(self):
        """Compute X and Z distances."""
        # d_Z = min weight of C2 \ C1^⊥
        C1_dual = self.C1.dual()
        C1_dual_codewords = set(tuple(cw) for cw in C1_dual.codewords)

        min_dz = self.n + 1
        for cw in self.C2.codewords:
            if tuple(cw) not in C1_dual_codewords:
                weight = np.sum(cw)
                if 0 < weight < min_dz:
                    min_dz = weight

        # d_X = min weight of C1 \ C2^⊥
        C2_dual = self.C2.dual()
        C2_dual_codewords = set(tuple(cw) for cw in C2_dual.codewords)

        min_dx = self.n + 1
        for cw in self.C1.codewords:
            if tuple(cw) not in C2_dual_codewords:
                weight = np.sum(cw)
                if 0 < weight < min_dx:
                    min_dx = weight

        self.d_X = min_dx if min_dx <= self.n else None
        self.d_Z = min_dz if min_dz <= self.n else None
        self.d = min(self.d_X or self.n + 1, self.d_Z or self.n + 1)

    @property
    def X_stabilizers(self) -> List[np.ndarray]:
        """Return X-type stabilizer generators."""
        return [row for row in self.C1.H]

    @property
    def Z_stabilizers(self) -> List[np.ndarray]:
        """Return Z-type stabilizer generators."""
        return [row for row in self.C2.G]

    def parameters_string(self) -> str:
        """Return code parameters as string."""
        if self.k < 0:
            return f"Invalid: k = {self.k}"
        if self.d_X != self.d_Z:
            return f"[[{self.n}, {self.k}, {self.d_X}/{self.d_Z}]]"
        return f"[[{self.n}, {self.k}, {self.d}]]"

    def print_stabilizers(self):
        """Print stabilizer generators in Pauli notation."""
        print("\nX-type stabilizers (from H1):")
        for i, stab in enumerate(self.X_stabilizers):
            pauli_str = ''.join(['X' if v else 'I' for v in stab])
            print(f"  g_X{i+1} = {pauli_str}")

        print("\nZ-type stabilizers (from G2):")
        for i, stab in enumerate(self.Z_stabilizers):
            pauli_str = ''.join(['Z' if v else 'I' for v in stab])
            print(f"  g_Z{i+1} = {pauli_str}")

    def verify_commutation(self) -> bool:
        """Verify all stabilizers commute."""
        for x_stab in self.X_stabilizers:
            for z_stab in self.Z_stabilizers:
                overlap = np.sum(x_stab * z_stab) % 2
                if overlap != 0:
                    return False
        return True

    def calculate_syndrome(self, error_x: np.ndarray, error_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate syndrome for given X and Z errors.

        Args:
            error_x: Binary vector indicating X error positions
            error_z: Binary vector indicating Z error positions

        Returns:
            (x_syndrome, z_syndrome)
        """
        # X errors detected by Z-stabilizers
        x_syndrome = np.dot(np.array(self.Z_stabilizers), error_x) % 2

        # Z errors detected by X-stabilizers
        z_syndrome = np.dot(np.array(self.X_stabilizers), error_z) % 2

        return x_syndrome, z_syndrome


def create_hamming_7_4() -> ClassicalLinearCode:
    """Create the [7,4,3] Hamming code."""
    G = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1],
    ], dtype=int)
    return ClassicalLinearCode(G)


def create_repetition(n: int) -> ClassicalLinearCode:
    """Create the [n,1,n] repetition code."""
    G = np.ones((1, n), dtype=int)
    return ClassicalLinearCode(G)


def create_hamming_15_11() -> ClassicalLinearCode:
    """Create the [15,11,3] Hamming code."""
    # Parity check matrix (4x15)
    H = np.array([
        [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    ], dtype=int)

    # Construct generator from null space of H
    # G = [I_11 | P^T] where H = [A | I_4]
    # Simpler: use known generator matrix
    G = np.zeros((11, 15), dtype=int)

    # Systematic part
    for i in range(11):
        G[i, i] = 1

    # Parity part (from H relationship)
    G[0, 11:] = [1, 1, 0, 0]
    G[1, 11:] = [0, 1, 1, 0]
    G[2, 11:] = [0, 0, 1, 1]
    G[3, 11:] = [1, 1, 0, 1]
    G[4, 11:] = [1, 0, 1, 0]
    G[5, 11:] = [0, 1, 0, 1]
    G[6, 11:] = [1, 1, 1, 0]
    G[7, 11:] = [0, 1, 1, 1]
    G[8, 11:] = [1, 1, 1, 1]
    G[9, 11:] = [1, 0, 1, 1]
    G[10, 11:] = [1, 0, 0, 1]

    return ClassicalLinearCode(G)


def demonstrate_css_construction():
    """Demonstrate CSS code construction with various classical codes."""

    print("=" * 70)
    print("CSS CODE CONSTRUCTION DEMONSTRATION")
    print("=" * 70)

    # Example 1: Steane code from Hamming
    print("\n" + "-" * 50)
    print("EXAMPLE 1: STEANE CODE FROM [7,4,3] HAMMING")
    print("-" * 50)

    hamming_7 = create_hamming_7_4()
    print(f"\nClassical Hamming code: {hamming_7}")
    print(f"Generator matrix G:\n{hamming_7.G}")
    print(f"Parity-check matrix H:\n{hamming_7.H}")

    # Check dual containment
    hamming_dual = hamming_7.dual()
    print(f"\nDual code: {hamming_dual}")

    is_contained = hamming_7.contains(hamming_dual)
    print(f"C^⊥ ⊆ C: {is_contained}")

    if is_contained:
        steane = CSSCode(hamming_7, hamming_7)
        print(f"\nSteane code parameters: {steane.parameters_string()}")
        steane.print_stabilizers()
        print(f"\nStabilizers commute: {steane.verify_commutation()}")

    # Example 2: Larger Hamming-based CSS
    print("\n" + "-" * 50)
    print("EXAMPLE 2: [[15,7,3]] FROM [15,11,3] HAMMING")
    print("-" * 50)

    hamming_15 = create_hamming_15_11()
    print(f"\nClassical Hamming code: {hamming_15}")

    hamming_15_dual = hamming_15.dual()
    print(f"Dual code: {hamming_15_dual}")

    is_contained_15 = hamming_15.contains(hamming_15_dual)
    print(f"C^⊥ ⊆ C: {is_contained_15}")

    if is_contained_15:
        css_15 = CSSCode(hamming_15, hamming_15)
        print(f"\nCSS code parameters: {css_15.parameters_string()}")
        print(f"Stabilizers commute: {css_15.verify_commutation()}")


def compare_css_codes():
    """Compare parameters of different CSS codes."""

    print("\n" + "=" * 70)
    print("CSS CODE COMPARISON")
    print("=" * 70)

    codes_data = []

    # Steane code
    hamming_7 = create_hamming_7_4()
    try:
        steane = CSSCode(hamming_7, hamming_7)
        codes_data.append({
            'name': 'Steane',
            'params': steane.parameters_string(),
            'n': steane.n,
            'k': steane.k,
            'd': steane.d,
            'rate': steane.k / steane.n,
            'overhead': steane.n / max(steane.k, 1)
        })
    except ValueError:
        pass

    # [[15,7,3]]
    hamming_15 = create_hamming_15_11()
    try:
        css_15 = CSSCode(hamming_15, hamming_15)
        codes_data.append({
            'name': 'Hamming-15',
            'params': css_15.parameters_string(),
            'n': css_15.n,
            'k': css_15.k,
            'd': css_15.d,
            'rate': css_15.k / css_15.n,
            'overhead': css_15.n / max(css_15.k, 1)
        })
    except ValueError:
        pass

    print("\n| Code | Parameters | Rate | Overhead |")
    print("|------|------------|------|----------|")
    for code in codes_data:
        print(f"| {code['name']:12s} | {code['params']:12s} | {code['rate']:.3f} | {code['overhead']:.2f}x |")


def syndrome_decoding_example():
    """Demonstrate syndrome decoding in CSS codes."""

    print("\n" + "=" * 70)
    print("CSS SYNDROME DECODING")
    print("=" * 70)

    hamming_7 = create_hamming_7_4()
    steane = CSSCode(hamming_7, hamming_7)

    print("\nSteane code syndrome table:")
    print("\nError   | X-synd | Z-synd | Combined")
    print("-" * 42)

    for error_type in ['I', 'X', 'Z', 'Y']:
        for qubit in range(7):
            if error_type == 'I' and qubit > 0:
                continue

            error_x = np.zeros(7, dtype=int)
            error_z = np.zeros(7, dtype=int)

            if error_type in ['X', 'Y']:
                error_x[qubit] = 1
            if error_type in ['Z', 'Y']:
                error_z[qubit] = 1

            x_syn, z_syn = steane.calculate_syndrome(error_x, error_z)

            x_str = ''.join(map(str, x_syn))
            z_str = ''.join(map(str, z_syn))

            if error_type == 'I':
                print(f"  I     | {x_str}   | {z_str}   | {x_str}{z_str}")
            else:
                print(f"  {error_type}{qubit+1}    | {x_str}   | {z_str}   | {x_str}{z_str}")


def plot_css_code_space():
    """Visualize the code space structure of CSS codes."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Venn diagram of CSS structure
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # Draw circles for C1 and C2^⊥
    circle1 = plt.Circle((4, 5), 3, fill=False, color='blue', linewidth=2, label='$C_1$')
    circle2 = plt.Circle((6, 5), 2, fill=False, color='red', linewidth=2, label='$C_2^\\perp$')

    ax1.add_patch(circle1)
    ax1.add_patch(circle2)

    ax1.text(2.5, 5, '$C_1 \\setminus C_2^\\perp$\n(Logical X)', ha='center', fontsize=10)
    ax1.text(5.5, 5, '$C_2^\\perp$\n(Z-stab)', ha='center', fontsize=10)

    ax1.set_title('CSS Code Structure: C₂⊥ ⊆ C₁', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.axis('off')

    # Plot 2: Parameter comparison
    ax2 = axes[1]

    codes = ['[[5,1,3]]', '[[7,1,3]]', '[[9,1,3]]', '[[15,7,3]]', '[[23,1,7]]']
    rates = [0.2, 0.143, 0.111, 0.467, 0.043]
    distances = [3, 3, 3, 3, 7]

    x = np.arange(len(codes))
    width = 0.35

    bars1 = ax2.bar(x - width/2, rates, width, label='Rate k/n', color='steelblue')
    bars2 = ax2.bar(x + width/2, [d/10 for d in distances], width, label='Distance/10', color='coral')

    ax2.set_ylabel('Value')
    ax2.set_title('CSS Code Parameters Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(codes, rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('css_code_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: css_code_analysis.png")


if __name__ == "__main__":
    demonstrate_css_construction()
    compare_css_codes()
    syndrome_decoding_example()

    print("\n" + "=" * 70)
    print("Generating visualization...")
    print("=" * 70)
    plot_css_code_space()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| CSS condition | $C_2^\perp \subseteq C_1$ |
| Code parameters | $[[n, k_1 + k_2 - n, \min(d_1, d_2^\perp)]]$ |
| X-stabilizers | $\{X^h : h \in H_1\}$ |
| Z-stabilizers | $\{Z^g : g \in G_2\}$ |
| Logical qubits | $k = k_1 + k_2 - n$ |
| Symmetric CSS | $CSS(C,C) = [[n, 2k-n, d]]$ |

### Main Takeaways

1. **Powerful Bridge:** CSS construction transforms classical coding theory to quantum
2. **Dual Containment:** The key requirement $C_2^\perp \subseteq C_1$ ensures stabilizer commutation
3. **Decoupled Correction:** X and Z errors are corrected independently using classical decoders
4. **Flexibility:** Can design asymmetric codes with different X/Z distances
5. **Foundation:** Surface codes, color codes, and many modern codes are CSS

---

## Daily Checklist

- [ ] Can state the CSS construction theorem
- [ ] Understand dual-containing code requirement
- [ ] Can calculate CSS code parameters from classical codes
- [ ] Know how to find X and Z stabilizers
- [ ] Understand decoupled error correction
- [ ] Can verify the Steane code is CSS
- [ ] Know examples beyond Steane (e.g., [[15,7,3]])

---

## Preview: Day 693

Tomorrow's synthesis day will bring together all concepts from Week 99:

- Complete stabilizer formalism toolkit
- Unified view of Shor, Steane, and general CSS codes
- Advanced topics: Reed-Muller codes, code families
- Preparation for Week 100: QEC Conditions deep dive

We'll consolidate the mathematical foundations before moving to more abstract QEC theory.

---

*"The CSS construction is one of the most elegant bridges between classical and quantum information theory."*
— Peter Shor
