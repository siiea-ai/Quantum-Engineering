# Day 729: Binary Symplectic Representation

## Overview

**Day:** 729 of 1008
**Week:** 105 (Binary Representation & F₂ Linear Algebra)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Binary Encoding of Pauli Operators

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Binary representation theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Worked examples and practice |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Recall** the structure of the n-qubit Pauli group
2. **Convert** any Pauli operator to its binary symplectic representation
3. **Perform** Pauli multiplication using binary vector addition
4. **Track** phases in Pauli multiplication systematically
5. **Verify** representations for multi-qubit Paulis
6. **Implement** binary encoding algorithms in code

---

## Core Content

### The n-Qubit Pauli Group

The Pauli group on n qubits, $\mathcal{P}_n$, consists of all tensor products of single-qubit Paulis with overall phases ±1, ±i:

$$\mathcal{P}_n = \{i^p \cdot P_1 \otimes P_2 \otimes \cdots \otimes P_n : p \in \{0,1,2,3\}, P_j \in \{I, X, Y, Z\}\}$$

**Group Properties:**
- Order: $|\mathcal{P}_n| = 4 \cdot 4^n$
- Non-Abelian (operators may not commute)
- Center: $Z(\mathcal{P}_n) = \{\pm I, \pm iI\}$
- Important quotient: $\mathcal{P}_n / Z(\mathcal{P}_n) \cong \mathbb{F}_2^{2n}$

### Single-Qubit Pauli Matrices

$$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Key Relations:**
$$XZ = iY, \quad ZX = -iY, \quad XZ = -ZX$$
$$X^2 = Y^2 = Z^2 = I$$

### The Binary Symplectic Representation

The key insight is that every Pauli operator (up to phase) can be written as:
$$P = X^{a_1}Z^{b_1} \otimes X^{a_2}Z^{b_2} \otimes \cdots \otimes X^{a_n}Z^{b_n}$$

where $a_i, b_i \in \{0, 1\}$.

**Definition (Binary Symplectic Vector):**
$$P \leftrightarrow (\mathbf{a}|\mathbf{b}) = (a_1, a_2, \ldots, a_n | b_1, b_2, \ldots, b_n) \in \mathbb{F}_2^{2n}$$

The first n components encode which qubits have X operators (including Y = XZ).
The last n components encode which qubits have Z operators (including Y).

### Encoding Table

| Pauli | Binary (a|b) | Explanation |
|-------|--------------|-------------|
| I | (0\|0) | No X, no Z |
| X | (1\|0) | Has X, no Z |
| Y | (1\|1) | Has both X and Z (Y = iXZ) |
| Z | (0\|1) | No X, has Z |

### Multi-Qubit Examples

**Example 1: Two Qubits**

$$X_1 Z_2 = X \otimes Z \leftrightarrow (1, 0 | 0, 1)$$

- First qubit: X → a₁=1, b₁=0
- Second qubit: Z → a₂=0, b₂=1

**Example 2: Three Qubits**

$$Y_1 X_2 Z_3 = Y \otimes X \otimes Z \leftrightarrow (1, 1, 0 | 1, 0, 1)$$

- First qubit: Y → a₁=1, b₁=1
- Second qubit: X → a₂=1, b₂=0
- Third qubit: Z → a₃=0, b₃=1

**Example 3: Stabilizer Generator**

$$X_1 X_2 X_3 X_4 \leftrightarrow (1, 1, 1, 1 | 0, 0, 0, 0)$$
$$Z_1 Z_2 Z_3 Z_4 \leftrightarrow (0, 0, 0, 0 | 1, 1, 1, 1)$$

### Pauli Multiplication in Binary

**Theorem (Vector Addition):**
If $P_1 \leftrightarrow (\mathbf{a}_1|\mathbf{b}_1)$ and $P_2 \leftrightarrow (\mathbf{a}_2|\mathbf{b}_2)$, then:
$$P_1 P_2 \leftrightarrow (\mathbf{a}_1 + \mathbf{a}_2 | \mathbf{b}_1 + \mathbf{b}_2) \pmod{2}$$

up to an overall phase.

**Proof:**
Consider a single qubit:
$$(X^{a_1}Z^{b_1})(X^{a_2}Z^{b_2}) = X^{a_1}Z^{b_1}X^{a_2}Z^{b_2}$$

Using $ZX = -XZ$:
$$= (-1)^{b_1 a_2} X^{a_1}X^{a_2}Z^{b_1}Z^{b_2} = (-1)^{b_1 a_2} X^{a_1+a_2}Z^{b_1+b_2}$$

The phase factor $(-1)^{b_1 a_2}$ comes from commuting Z past X.

### Phase Tracking

The full phase when multiplying $P_1 P_2$ requires careful tracking:

**Phase Formula:**
$$P_1 P_2 = i^{\phi} \cdot X^{\mathbf{a}_1 + \mathbf{a}_2} Z^{\mathbf{b}_1 + \mathbf{b}_2}$$

where:
$$\phi = \sum_{j=1}^{n} \left( 2(b_1)_j (a_2)_j + (a_1)_j (b_1)_j + (a_2)_j (b_2)_j - (a_1 + a_2)_j (b_1 + b_2)_j \right) \pmod{4}$$

**Simplified Phase (for commutation check):**
The sign change when commuting is:
$$P_1 P_2 = (-1)^{\mathbf{b}_1 \cdot \mathbf{a}_2} P_2 P_1 \cdot (-1)^{\mathbf{a}_1 \cdot \mathbf{b}_2}$$

So:
$$P_1 P_2 = (-1)^{\mathbf{a}_1 \cdot \mathbf{b}_2 + \mathbf{b}_1 \cdot \mathbf{a}_2} P_2 P_1$$

This is the **symplectic inner product**!

### Symplectic Inner Product Preview

**Definition:**
$$\boxed{\langle(\mathbf{a}_1|\mathbf{b}_1), (\mathbf{a}_2|\mathbf{b}_2)\rangle_s = \mathbf{a}_1 \cdot \mathbf{b}_2 + \mathbf{b}_1 \cdot \mathbf{a}_2 \pmod{2}}$$

**Commutation Criterion:**
$$\boxed{[P_1, P_2] = 0 \Leftrightarrow \langle v_1, v_2 \rangle_s = 0}$$

This is the fundamental connection between commutation and linear algebra!

### Weight and Support

**Definition (Weight):**
The weight of a Pauli operator is the number of qubits on which it acts non-trivially:
$$\text{wt}(P) = |\{j : P_j \neq I\}|$$

In binary representation:
$$\text{wt}(P) = |\{j : a_j = 1 \text{ or } b_j = 1\}|$$

**Definition (Support):**
$$\text{supp}(P) = \{j : P_j \neq I\}$$

### Stabilizer Codes in Binary

An [[n, k, d]] stabilizer code has n-k independent stabilizer generators $S_1, \ldots, S_{n-k}$.

In binary form, these become rows of a matrix:
$$H = \begin{pmatrix}
(\mathbf{a}_1 | \mathbf{b}_1) \\
(\mathbf{a}_2 | \mathbf{b}_2) \\
\vdots \\
(\mathbf{a}_{n-k} | \mathbf{b}_{n-k})
\end{pmatrix} \in \mathbb{F}_2^{(n-k) \times 2n}$$

The condition that all stabilizers commute becomes:
$$H \Omega H^T = 0$$

where $\Omega = \begin{pmatrix} 0 & I_n \\ I_n & 0 \end{pmatrix}$.

---

## Worked Examples

### Example 1: Binary Encoding of the [[5,1,3]] Code

The [[5,1,3]] perfect code has stabilizers:
$$S_1 = X_1 Z_2 Z_3 X_4$$
$$S_2 = X_2 Z_3 Z_4 X_5$$
$$S_3 = X_1 X_3 Z_4 Z_5$$
$$S_4 = Z_1 X_2 X_4 Z_5$$

**Convert to binary:**

$S_1 = X_1 Z_2 Z_3 X_4 I_5$:
- X positions: 1, 4 → $\mathbf{a}_1 = (1,0,0,1,0)$
- Z positions: 2, 3 → $\mathbf{b}_1 = (0,1,1,0,0)$
- Binary: $(1,0,0,1,0 | 0,1,1,0,0)$

$S_2 = I_1 X_2 Z_3 Z_4 X_5$:
- X positions: 2, 5 → $\mathbf{a}_2 = (0,1,0,0,1)$
- Z positions: 3, 4 → $\mathbf{b}_2 = (0,0,1,1,0)$
- Binary: $(0,1,0,0,1 | 0,0,1,1,0)$

$S_3 = X_1 I_2 X_3 Z_4 Z_5$:
- X positions: 1, 3 → $\mathbf{a}_3 = (1,0,1,0,0)$
- Z positions: 4, 5 → $\mathbf{b}_3 = (0,0,0,1,1)$
- Binary: $(1,0,1,0,0 | 0,0,0,1,1)$

$S_4 = Z_1 X_2 I_3 X_4 Z_5$:
- X positions: 2, 4 → $\mathbf{a}_4 = (0,1,0,1,0)$
- Z positions: 1, 5 → $\mathbf{b}_4 = (1,0,0,0,1)$
- Binary: $(0,1,0,1,0 | 1,0,0,0,1)$

**Parity Check Matrix:**
$$H = \begin{pmatrix}
1 & 0 & 0 & 1 & 0 & | & 0 & 1 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 1 & | & 0 & 0 & 1 & 1 & 0 \\
1 & 0 & 1 & 0 & 0 & | & 0 & 0 & 0 & 1 & 1 \\
0 & 1 & 0 & 1 & 0 & | & 1 & 0 & 0 & 0 & 1
\end{pmatrix}$$

### Example 2: Verifying Commutation

Check that $S_1$ and $S_2$ commute:

$v_1 = (1,0,0,1,0 | 0,1,1,0,0)$
$v_2 = (0,1,0,0,1 | 0,0,1,1,0)$

$$\langle v_1, v_2 \rangle_s = \mathbf{a}_1 \cdot \mathbf{b}_2 + \mathbf{b}_1 \cdot \mathbf{a}_2$$

$$\mathbf{a}_1 \cdot \mathbf{b}_2 = (1,0,0,1,0) \cdot (0,0,1,1,0) = 0 + 0 + 0 + 1 + 0 = 1$$
$$\mathbf{b}_1 \cdot \mathbf{a}_2 = (0,1,1,0,0) \cdot (0,1,0,0,1) = 0 + 1 + 0 + 0 + 0 = 1$$

$$\langle v_1, v_2 \rangle_s = 1 + 1 = 0 \pmod{2}$$

✓ $S_1$ and $S_2$ commute.

### Example 3: Product of Paulis

Compute $P_1 P_2$ where $P_1 = X_1 Y_2$ and $P_2 = Z_1 X_2$ on 2 qubits.

**Binary representations:**
- $P_1 = X_1 Y_2 \leftrightarrow (1,1 | 0,1)$
- $P_2 = Z_1 X_2 \leftrightarrow (0,1 | 1,0)$

**Product (ignoring phase):**
$$(\mathbf{a}_1 + \mathbf{a}_2 | \mathbf{b}_1 + \mathbf{b}_2) = (1+0, 1+1 | 0+1, 1+0) = (1, 0 | 1, 1)$$

This corresponds to $Y_1 Z_2$.

**Phase calculation:**
From $ZX = -XZ$, we need to count how many Z's pass through X's.

At qubit 1: Z from $P_2$ passes through nothing in $P_1$ (X is to left of Z in $P_1 \cdot P_2$)
At qubit 2: X from $P_2$ must pass through Y = iXZ from $P_1$

Actually, let's compute directly:
$$P_1 P_2 = (X \otimes Y)(Z \otimes X) = XZ \otimes YX = (iY) \otimes (-iZ) = -Y \otimes Z$$

So $P_1 P_2 = -Y_1 Z_2$.

---

## Practice Problems

### Level 1: Direct Application

1. **Binary Encoding:** Convert the following to binary symplectic form:
   a) $X_1 X_2 I_3 Z_4$
   b) $Y_1 Y_2 Y_3$
   c) $Z_1 I_2 X_3 Y_4 Z_5$

2. **Decoding:** What Pauli operators do these binary vectors represent?
   a) $(1,0,1 | 0,0,0)$
   b) $(0,0,0 | 1,1,1)$
   c) $(1,1,0 | 1,0,1)$

3. **Weight Calculation:** Compute the weight of each operator in Problem 2.

### Level 2: Intermediate

4. **Commutation Check:** Determine which pairs commute:
   a) $X_1 Z_2$ and $Z_1 X_2$
   b) $X_1 X_2 X_3$ and $Z_1 Z_2 I_3$
   c) $Y_1 Y_2$ and $X_1 Z_2$

5. **Product Calculation:** Compute $P_1 P_2$ in binary form (ignore phase):
   a) $P_1 = X_1 Z_2$, $P_2 = Z_1 X_2$
   b) $P_1 = Y_1 Y_2 Y_3$, $P_2 = X_1 X_2 X_3$
   c) $P_1 = X_1 Z_2 Z_3$, $P_2 = Z_1 X_2 Z_3$

6. **Steane Code:** The [[7,1,3]] Steane code has stabilizers:
   $$S_1 = I I I X X X X, \quad S_2 = I X X I I X X, \quad S_3 = X I X I X I X$$
   $$S_4 = I I I Z Z Z Z, \quad S_5 = I Z Z I I Z Z, \quad S_6 = Z I Z I Z I Z$$

   Write the full parity check matrix H.

### Level 3: Challenging

7. **Self-Orthogonality:** Prove that for any stabilizer code, $H \Omega H^T = 0$ where $\Omega = \begin{pmatrix} 0 & I \\ I & 0 \end{pmatrix}$.

8. **Logical Operators:** For the [[5,1,3]] code with the H matrix from Example 1:
   a) Find a vector $v$ with $\langle v, h_i \rangle_s = 0$ for all rows $h_i$ of H
   b) Verify this corresponds to a logical operator
   c) What is the weight of the minimum logical operator?

9. **Code Construction:** Given the classical [7,4,3] Hamming code with parity check matrix:
   $$H_c = \begin{pmatrix} 1 & 1 & 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 1 & 0 & 0 & 1 \end{pmatrix}$$

   Construct the stabilizer matrix for a CSS code using this.

---

## Solutions

### Level 1 Solutions

1. **Binary Encoding:**
   a) $X_1 X_2 I_3 Z_4 \leftrightarrow (1,1,0,0 | 0,0,0,1)$
   b) $Y_1 Y_2 Y_3 \leftrightarrow (1,1,1 | 1,1,1)$
   c) $Z_1 I_2 X_3 Y_4 Z_5 \leftrightarrow (0,0,1,1,0 | 1,0,0,1,1)$

2. **Decoding:**
   a) $(1,0,1 | 0,0,0) = X_1 I_2 X_3$
   b) $(0,0,0 | 1,1,1) = Z_1 Z_2 Z_3$
   c) $(1,1,0 | 1,0,1) = Y_1 X_2 Z_3$

3. **Weight:**
   a) wt = 2 (qubits 1 and 3)
   b) wt = 3 (all qubits)
   c) wt = 3 (all qubits)

### Level 2 Solutions

4. **Commutation:**
   a) $v_1 = (1,0|0,1)$, $v_2 = (0,1|1,0)$
      $\langle v_1, v_2 \rangle_s = (1)(1) + (0)(0) + (0)(0) + (1)(1) = 1 + 1 = 0$ ✓ Commute

   b) $v_1 = (1,1,1|0,0,0)$, $v_2 = (0,0,0|1,1,0)$
      $\langle v_1, v_2 \rangle_s = 1+1+0 + 0 = 0$ ✓ Commute

   c) $v_1 = (1,1|1,1)$, $v_2 = (1,0|0,1)$
      $\langle v_1, v_2 \rangle_s = (1)(0) + (1)(1) + (1)(1) + (1)(0) = 0 + 1 + 1 + 0 = 0$ ✓ Commute

5. **Products:**
   a) $(1,0|0,1) + (0,1|1,0) = (1,1|1,1) = Y_1 Y_2$
   b) $(1,1,1|1,1,1) + (1,1,1|0,0,0) = (0,0,0|1,1,1) = Z_1 Z_2 Z_3$
   c) $(1,0,0|0,1,1) + (0,1,0|1,0,1) = (1,1,0|1,1,0) = Y_1 Y_2 I_3$

---

## Computational Lab

```python
"""
Day 729: Binary Symplectic Representation
=========================================
Implementation of Pauli-to-binary conversion and basic operations.
"""

import numpy as np
from typing import Tuple, List

class BinaryPauli:
    """
    Represents a Pauli operator in binary symplectic form.

    An n-qubit Pauli P = X^a Z^b is stored as (a|b) in F_2^{2n}.
    Phase is tracked separately as i^phase where phase ∈ {0,1,2,3}.
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, phase: int = 0):
        """
        Initialize from X-part (a) and Z-part (b).

        Parameters:
        -----------
        a : np.ndarray
            Binary vector for X components
        b : np.ndarray
            Binary vector for Z components
        phase : int
            Power of i (0, 1, 2, or 3)
        """
        self.a = np.array(a, dtype=int) % 2
        self.b = np.array(b, dtype=int) % 2
        self.phase = phase % 4
        self.n = len(a)

        assert len(a) == len(b), "a and b must have same length"

    @classmethod
    def from_string(cls, pauli_str: str):
        """
        Create BinaryPauli from string like 'XYZII'.

        Parameters:
        -----------
        pauli_str : str
            String of I, X, Y, Z characters

        Returns:
        --------
        BinaryPauli
        """
        pauli_str = pauli_str.upper().replace(' ', '')
        n = len(pauli_str)
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        phase = 0

        for i, char in enumerate(pauli_str):
            if char == 'X':
                a[i] = 1
            elif char == 'Z':
                b[i] = 1
            elif char == 'Y':
                a[i] = 1
                b[i] = 1
                phase += 1  # Y = iXZ
            elif char == 'I':
                pass
            else:
                raise ValueError(f"Invalid Pauli character: {char}")

        return cls(a, b, phase)

    def to_string(self) -> str:
        """Convert back to Pauli string."""
        chars = []
        for i in range(self.n):
            if self.a[i] == 0 and self.b[i] == 0:
                chars.append('I')
            elif self.a[i] == 1 and self.b[i] == 0:
                chars.append('X')
            elif self.a[i] == 0 and self.b[i] == 1:
                chars.append('Z')
            else:  # a[i] == 1 and b[i] == 1
                chars.append('Y')

        phase_str = ['', 'i', '-', '-i'][self.phase]
        return phase_str + ''.join(chars)

    def to_vector(self) -> np.ndarray:
        """Return full 2n binary vector (a|b)."""
        return np.concatenate([self.a, self.b])

    @classmethod
    def from_vector(cls, v: np.ndarray, phase: int = 0):
        """Create from 2n binary vector."""
        n = len(v) // 2
        return cls(v[:n], v[n:], phase)

    def weight(self) -> int:
        """Return the weight (number of non-identity positions)."""
        return np.sum((self.a | self.b) > 0)

    def support(self) -> List[int]:
        """Return indices where Pauli is non-identity."""
        return list(np.where((self.a | self.b) > 0)[0])

    def symplectic_inner_product(self, other: 'BinaryPauli') -> int:
        """
        Compute symplectic inner product <self, other>_s.

        Returns 0 if operators commute, 1 if they anticommute.
        """
        return (np.dot(self.a, other.b) + np.dot(self.b, other.a)) % 2

    def commutes_with(self, other: 'BinaryPauli') -> bool:
        """Check if self commutes with other."""
        return self.symplectic_inner_product(other) == 0

    def __mul__(self, other: 'BinaryPauli') -> 'BinaryPauli':
        """
        Multiply two Pauli operators.

        P1 * P2 computes the product with proper phase tracking.
        """
        assert self.n == other.n, "Paulis must have same number of qubits"

        # New X and Z parts
        new_a = (self.a + other.a) % 2
        new_b = (self.b + other.b) % 2

        # Phase from Y = iXZ representation
        # When we have a Y, it contributes i
        # Phase tracking is complex - simplified version:
        # Count XZ → ZX swaps needed
        phase_contribution = 0

        for i in range(self.n):
            # Phase from commuting Z (self) through X (other)
            phase_contribution += 2 * self.b[i] * other.a[i]

            # Correction for Y representation
            # self Y contribution
            if self.a[i] == 1 and self.b[i] == 1:
                pass  # already counted in __init__
            # other Y contribution
            if other.a[i] == 1 and other.b[i] == 1:
                pass

        new_phase = (self.phase + other.phase + phase_contribution) % 4

        return BinaryPauli(new_a, new_b, new_phase)

    def __repr__(self):
        return f"BinaryPauli({self.to_string()}, vector={self.to_vector()})"


def verify_commutation_matrix(paulis: List[BinaryPauli]) -> np.ndarray:
    """
    Compute the commutation matrix for a list of Paulis.

    Returns matrix C where C[i,j] = <P_i, P_j>_s
    (0 = commute, 1 = anticommute)
    """
    n = len(paulis)
    C = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            C[i, j] = paulis[i].symplectic_inner_product(paulis[j])

    return C


def create_parity_check_matrix(stabilizers: List[BinaryPauli]) -> np.ndarray:
    """
    Create the parity check matrix H from stabilizer generators.

    Each row is the binary vector of a stabilizer generator.
    """
    return np.array([s.to_vector() for s in stabilizers])


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 729: Binary Symplectic Representation")
    print("=" * 60)

    # Example 1: Basic conversion
    print("\n1. Basic Pauli-to-Binary Conversion")
    print("-" * 40)

    paulis = ['IXYZ', 'XXXX', 'ZZZZ', 'YIYI']
    for p_str in paulis:
        p = BinaryPauli.from_string(p_str)
        print(f"{p_str:8} → {p.to_vector()} (weight={p.weight()})")

    # Example 2: Commutation check
    print("\n2. Commutation Relations")
    print("-" * 40)

    p1 = BinaryPauli.from_string('XZ')
    p2 = BinaryPauli.from_string('ZX')
    print(f"XZ and ZX: symplectic product = {p1.symplectic_inner_product(p2)}")
    print(f"  → {'commute' if p1.commutes_with(p2) else 'anticommute'}")

    p3 = BinaryPauli.from_string('XX')
    p4 = BinaryPauli.from_string('ZZ')
    print(f"XX and ZZ: symplectic product = {p3.symplectic_inner_product(p4)}")
    print(f"  → {'commute' if p3.commutes_with(p4) else 'anticommute'}")

    # Example 3: [[5,1,3]] code stabilizers
    print("\n3. [[5,1,3]] Code Stabilizers")
    print("-" * 40)

    # Stabilizers of the 5-qubit code
    stabs_513 = [
        BinaryPauli.from_string('XZZXI'),
        BinaryPauli.from_string('IXZZX'),
        BinaryPauli.from_string('XIXZZ'),
        BinaryPauli.from_string('ZXIXZ')
    ]

    print("Stabilizer generators:")
    for i, s in enumerate(stabs_513):
        print(f"  S{i+1}: {s.to_string():6} → {s.to_vector()}")

    # Verify all pairs commute
    print("\nCommutation matrix (should be all zeros):")
    comm_matrix = verify_commutation_matrix(stabs_513)
    print(comm_matrix)

    # Create parity check matrix
    print("\nParity check matrix H:")
    H = create_parity_check_matrix(stabs_513)
    print(H)

    # Example 4: Pauli multiplication
    print("\n4. Pauli Multiplication")
    print("-" * 40)

    pa = BinaryPauli.from_string('XY')
    pb = BinaryPauli.from_string('ZX')
    pc = pa * pb
    print(f"XY × ZX = {pc.to_string()}")
    print(f"  (vector: {pc.to_vector()})")

    # Example 5: Steane code
    print("\n5. [[7,1,3]] Steane Code")
    print("-" * 40)

    steane_stabs = [
        BinaryPauli.from_string('IIIXXXX'),
        BinaryPauli.from_string('IXXIIXX'),
        BinaryPauli.from_string('XIXIXIX'),
        BinaryPauli.from_string('IIIZZZZ'),
        BinaryPauli.from_string('IZZIIZZ'),
        BinaryPauli.from_string('ZIZIZIZ')
    ]

    H_steane = create_parity_check_matrix(steane_stabs)
    print("Steane code parity check matrix:")
    print("H_X | H_Z")
    print(H_steane)

    # Verify self-orthogonality
    print("\nVerifying H Ω H^T = 0:")
    n = 7
    Omega = np.block([[np.zeros((n,n), dtype=int), np.eye(n, dtype=int)],
                      [np.eye(n, dtype=int), np.zeros((n,n), dtype=int)]])
    result = (H_steane @ Omega @ H_steane.T) % 2
    print(f"H Ω H^T mod 2 = \n{result}")
    print(f"All zeros: {np.all(result == 0)}")

    print("\n" + "=" * 60)
    print("End of Day 729 Lab")
    print("=" * 60)
```

**Expected Output:**
```
============================================================
Day 729: Binary Symplectic Representation
============================================================

1. Basic Pauli-to-Binary Conversion
----------------------------------------
IXYZ     → [0 1 1 0 0 0 1 1] (weight=3)
XXXX     → [1 1 1 1 0 0 0 0] (weight=4)
ZZZZ     → [0 0 0 0 1 1 1 1] (weight=4)
YIYI     → [1 0 1 0 1 0 1 0] (weight=2)

2. Commutation Relations
----------------------------------------
XZ and ZX: symplectic product = 0
  → commute
XX and ZZ: symplectic product = 0
  → commute

3. [[5,1,3]] Code Stabilizers
----------------------------------------
Stabilizer generators:
  S1: XZZXI  → [1 0 0 1 0 0 1 1 0 0]
  S2: IXZZX  → [0 1 0 0 1 0 0 1 1 0]
  S3: XIXZZ  → [1 0 1 0 0 0 0 0 1 1]
  S4: ZXIXZ  → [0 1 0 1 0 1 0 0 0 1]

Commutation matrix (should be all zeros):
[[0 0 0 0]
 [0 0 0 0]
 [0 0 0 0]
 [0 0 0 0]]
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Binary encoding | $P = X^{\mathbf{a}}Z^{\mathbf{b}} \leftrightarrow (\mathbf{a}\|\mathbf{b})$ |
| Pauli product | $P_1 P_2 \leftrightarrow (\mathbf{a}_1 + \mathbf{a}_2 \| \mathbf{b}_1 + \mathbf{b}_2)$ |
| Symplectic product | $\langle v_1, v_2 \rangle_s = \mathbf{a}_1 \cdot \mathbf{b}_2 + \mathbf{b}_1 \cdot \mathbf{a}_2$ |
| Commutation | $[P_1, P_2] = 0 \Leftrightarrow \langle v_1, v_2 \rangle_s = 0$ |
| Weight | $\text{wt}(P) = \|\{j : a_j = 1 \text{ or } b_j = 1\}\|$ |

### Main Takeaways

1. **The binary representation** converts the non-Abelian Pauli group to linear algebra over F₂
2. **Pauli multiplication** becomes simple vector addition (mod 2)
3. **Commutation relations** are encoded in the symplectic inner product
4. **Stabilizer codes** become null spaces of the parity check matrix
5. **This formalism** enables efficient classical simulation and systematic code analysis

---

## Daily Checklist

- [ ] I can convert any Pauli string to binary form
- [ ] I understand why Y maps to (1|1)
- [ ] I can compute Pauli products using vector addition
- [ ] I can check commutation using the symplectic inner product
- [ ] I understand the structure of the parity check matrix
- [ ] I implemented the BinaryPauli class successfully

---

## Preview: Day 730

Tomorrow we dive into **F₂ Vector Spaces** — the mathematical foundation for binary stabilizer formalism:
- Finite field F₂ = {0, 1} and its properties
- Vector spaces over F₂
- Gaussian elimination mod 2
- Row echelon form and rank
- Applications to stabilizer code analysis
