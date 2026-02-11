# Day 703: Symplectic Representation of Clifford Gates

## Overview

**Date:** Day 703 of 1008
**Week:** 101 (Advanced Stabilizer Theory)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Binary Symplectic Representation of Clifford Operations

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Symplectic vector spaces and Pauli representations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Clifford gates as symplectic matrices |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Represent Pauli operators** as vectors in $\mathbb{F}_2^{2n}$ (binary field)
2. **Construct the symplectic inner product** and understand its physical meaning
3. **Express Clifford gates** as $2n \times 2n$ symplectic matrices over $\mathbb{F}_2$
4. **Derive symplectic matrices** for H, S, CNOT, and composite gates
5. **Verify symplectic conditions** and compose transformations
6. **Connect symplectic structure** to Pauli commutation relations

---

## Core Content

### 1. Binary Representation of Pauli Operators

#### The Vector Space $\mathbb{F}_2^{2n}$

For an $n$-qubit system, we represent Pauli operators (up to phase) as binary vectors.

**Single Qubit:**

$$P = i^\gamma X^a Z^b \quad \longleftrightarrow \quad (a|b) \in \mathbb{F}_2^2$$

| Pauli | $X^a Z^b$ | Vector $(a|b)$ |
|-------|-----------|----------------|
| $I$ | $X^0 Z^0$ | $(0|0)$ |
| $X$ | $X^1 Z^0$ | $(1|0)$ |
| $Z$ | $X^0 Z^1$ | $(0|1)$ |
| $Y$ | $X^1 Z^1$ | $(1|1)$ |

Note: $Y = iXZ$, so $Y \sim X^1 Z^1$ ignoring phase.

**Multi-Qubit Extension:**

For $n$ qubits, a Pauli operator $P = P_1 \otimes P_2 \otimes \cdots \otimes P_n$ maps to:

$$P \longleftrightarrow (\mathbf{a}|\mathbf{b}) = (a_1, a_2, \ldots, a_n | b_1, b_2, \ldots, b_n) \in \mathbb{F}_2^{2n}$$

where $P_j = X_j^{a_j} Z_j^{b_j}$ (up to phase).

#### Example: Two-Qubit Paulis

$$X_1 Z_2 = X \otimes Z \longleftrightarrow (1,0|0,1)$$
$$Y_1 Y_2 = (XZ) \otimes (XZ) \longleftrightarrow (1,1|1,1)$$

---

### 2. The Symplectic Inner Product

#### Definition

For two vectors $\mathbf{v} = (\mathbf{a}|\mathbf{b})$ and $\mathbf{v}' = (\mathbf{a}'|\mathbf{b}')$ in $\mathbb{F}_2^{2n}$:

$$\boxed{\langle \mathbf{v}, \mathbf{v}' \rangle_s = \mathbf{a} \cdot \mathbf{b}' + \mathbf{a}' \cdot \mathbf{b} \pmod{2}}$$

This is the **symplectic inner product** (also called symplectic form).

#### Matrix Form

Using the symplectic matrix:

$$\Lambda = \begin{pmatrix} 0 & I_n \\ I_n & 0 \end{pmatrix}$$

The symplectic inner product is:

$$\langle \mathbf{v}, \mathbf{v}' \rangle_s = \mathbf{v}^T \Lambda \mathbf{v}' \pmod{2}$$

#### Physical Meaning: Commutation Relations

**Key Theorem:** Two Pauli operators $P, P'$ corresponding to $\mathbf{v}, \mathbf{v}'$:

$$\boxed{PP' = (-1)^{\langle \mathbf{v}, \mathbf{v}' \rangle_s} P'P}$$

- $\langle \mathbf{v}, \mathbf{v}' \rangle_s = 0$: Paulis **commute**
- $\langle \mathbf{v}, \mathbf{v}' \rangle_s = 1$: Paulis **anticommute**

**Proof:**
$$XZ = -ZX \Rightarrow X^a Z^b \cdot X^{a'} Z^{b'} = (-1)^{ab' + a'b} X^{a'} Z^{b'} \cdot X^a Z^b$$

---

### 3. Symplectic Matrices and the Symplectic Group

#### Symplectic Group $Sp(2n, \mathbb{F}_2)$

A matrix $M \in GL(2n, \mathbb{F}_2)$ is **symplectic** if it preserves the symplectic form:

$$\boxed{M^T \Lambda M = \Lambda}$$

Equivalently, for all $\mathbf{v}, \mathbf{v}'$:
$$\langle M\mathbf{v}, M\mathbf{v}' \rangle_s = \langle \mathbf{v}, \mathbf{v}' \rangle_s$$

The set of all such matrices forms the **symplectic group** $Sp(2n, \mathbb{F}_2)$.

#### Properties of Symplectic Matrices

1. **Closure:** $M_1, M_2 \in Sp \Rightarrow M_1 M_2 \in Sp$
2. **Inverse:** $M \in Sp \Rightarrow M^{-1} \in Sp$, with $M^{-1} = \Lambda M^T \Lambda$
3. **Determinant:** $\det(M) = 1$ (over $\mathbb{F}_2$, this means odd)
4. **Size:** $|Sp(2n, \mathbb{F}_2)| = 2^{n^2} \prod_{j=1}^{n}(4^j - 1)$

For $n=1$: $|Sp(2, \mathbb{F}_2)| = 6$

---

### 4. Clifford Gates as Symplectic Matrices

#### The Isomorphism (up to phases)

**Theorem:** The Clifford group modulo phases is isomorphic to the symplectic group:

$$\boxed{\mathcal{C}_n / \mathcal{P}_n \cong Sp(2n, \mathbb{F}_2) \ltimes \mathbb{F}_2^{2n}}$$

where the $\mathbb{F}_2^{2n}$ factor accounts for Pauli translations.

#### Action of Clifford Gates

If Clifford gate $C$ maps Pauli $P_{\mathbf{v}}$ to Pauli $P_{\mathbf{v}'}$ (up to phase):

$$C P_{\mathbf{v}} C^\dagger = \pm P_{\mathbf{v}'} \quad \Rightarrow \quad \mathbf{v}' = M_C \mathbf{v}$$

where $M_C$ is the symplectic matrix for $C$.

---

### 5. Symplectic Matrices for Standard Clifford Gates

#### Hadamard Gate $H$

Transformation:
- $H X H^\dagger = Z$: $(1|0) \to (0|1)$
- $H Z H^\dagger = X$: $(0|1) \to (1|0)$

$$\boxed{M_H = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}}$$

**Verification:** $M_H^T \Lambda M_H = \Lambda$ ✓

#### Phase Gate $S$

Transformation:
- $S X S^\dagger = Y = iXZ$: $(1|0) \to (1|1)$
- $S Z S^\dagger = Z$: $(0|1) \to (0|1)$

$$\boxed{M_S = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}}$$

#### CNOT Gate

For two qubits (control=1, target=2):
- $\text{CNOT}: X_1 \to X_1 X_2$: $(10|00) \to (11|00)$
- $\text{CNOT}: Z_1 \to Z_1$: $(00|10) \to (00|10)$
- $\text{CNOT}: X_2 \to X_2$: $(01|00) \to (01|00)$
- $\text{CNOT}: Z_2 \to Z_1 Z_2$: $(00|01) \to (00|11)$

$$\boxed{M_{\text{CNOT}} = \begin{pmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1
\end{pmatrix}}$$

In block form with $A, B, C, D$ as $n \times n$ blocks:

$$M_{\text{CNOT}} = \begin{pmatrix} A & B \\ C & D \end{pmatrix} = \begin{pmatrix}
\begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix} & \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix} \\
\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix} & \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
\end{pmatrix}$$

---

### 6. Composing Clifford Gates

#### Matrix Multiplication

For sequential gates $C_1$ then $C_2$:

$$M_{C_2 C_1} = M_{C_2} M_{C_1}$$

**Example:** $HS$ (S then H)

$$M_{HS} = M_H M_S = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$$

Verify: This is the symplectic matrix for the gate $HS = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -i \\ 1 & i \end{pmatrix}$.

---

### 7. From Symplectic Matrix to Clifford Circuit

#### Decomposition Theorem

Any symplectic matrix $M \in Sp(2n, \mathbb{F}_2)$ can be written as a product of symplectic matrices corresponding to:
- **Hadamard gates** $H_i$ on single qubits
- **Phase gates** $S_i$ on single qubits
- **CNOT gates** $\text{CNOT}_{ij}$ between pairs

**Constructive Algorithm:**

1. Use Gaussian elimination over $\mathbb{F}_2$
2. Preserve symplectic structure at each step
3. Express row operations as gate compositions

This gives a circuit of $O(n^2)$ gates for any Clifford.

---

### 8. Stabilizer Codes in Symplectic Language

#### Isotropic Subspaces

A subspace $S \subset \mathbb{F}_2^{2n}$ is **isotropic** if:

$$\forall \mathbf{v}, \mathbf{v}' \in S: \quad \langle \mathbf{v}, \mathbf{v}' \rangle_s = 0$$

Equivalently: all corresponding Paulis commute.

**Stabilizer codes:** An $[[n, k]]$ code has stabilizer group corresponding to an isotropic subspace of dimension $n-k$.

#### Self-Orthogonality Condition

For CSS codes from classical codes $C_1, C_2$:

$$C_2^\perp \subseteq C_1 \quad \Leftrightarrow \quad \text{Symplectic subspace is isotropic}$$

---

## Worked Examples

### Example 1: Verify Symplectic Property of CNOT

**Problem:** Show that $M_{\text{CNOT}}^T \Lambda M_{\text{CNOT}} = \Lambda$.

**Solution:**

With $\Lambda = \begin{pmatrix} 0 & I_2 \\ I_2 & 0 \end{pmatrix}$ and $M = M_{\text{CNOT}}$:

$$M^T = \begin{pmatrix}
1 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 1 & 1
\end{pmatrix}$$

Compute $M^T \Lambda$:
$$M^T \Lambda = \begin{pmatrix}
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0
\end{pmatrix}$$

Compute $M^T \Lambda M$:
$$M^T \Lambda M = \begin{pmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{pmatrix} = \Lambda \quad \checkmark$$

---

### Example 2: Symplectic Matrix for CZ Gate

**Problem:** Derive the symplectic matrix for the controlled-Z (CZ) gate.

**Solution:**

CZ gate action:
- $CZ: X_1 \to X_1 Z_2$: $(10|00) \to (10|01)$
- $CZ: Z_1 \to Z_1$: $(00|10) \to (00|10)$
- $CZ: X_2 \to Z_1 X_2$: $(01|00) \to (01|10)$
- $CZ: Z_2 \to Z_2$: $(00|01) \to (00|01)$

$$M_{CZ} = \begin{pmatrix}
1 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}$$

**Verification:** $CZ = (I \otimes H) \cdot \text{CNOT} \cdot (I \otimes H)$

$$M_{CZ} = M_{I \otimes H} M_{\text{CNOT}} M_{I \otimes H}$$

where $M_{I \otimes H} = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0
\end{pmatrix}$

---

### Example 3: Commutation from Symplectic Product

**Problem:** Do $X_1 Z_2$ and $Y_1 X_2$ commute?

**Solution:**

Binary vectors:
- $X_1 Z_2 \leftrightarrow \mathbf{v} = (1,0|0,1)$
- $Y_1 X_2 = (XZ)_1 X_2 \leftrightarrow \mathbf{v}' = (1,1|1,0)$

Symplectic product:
$$\langle \mathbf{v}, \mathbf{v}' \rangle_s = (1)(1) + (0)(0) + (1)(0) + (0)(1) = 1 + 0 + 0 + 0 = 1$$

Since $\langle \mathbf{v}, \mathbf{v}' \rangle_s = 1$, they **anticommute**.

**Check:** $X_1 Z_2 \cdot Y_1 X_2 = (XY)_1 (ZX)_2 = (iZ)_1 (-iY)_2 = -Z_1 Y_2$

$Y_1 X_2 \cdot X_1 Z_2 = (YX)_1 (XZ)_2 = (-iZ)_1 (iY)_2 = +Z_1 Y_2$

Indeed: $X_1 Z_2 \cdot Y_1 X_2 = -(Y_1 X_2 \cdot X_1 Z_2)$ ✓

---

## Practice Problems

### Direct Application

1. **Problem 1:** Find the binary representation for $Z_1 X_2 Y_3$ on 3 qubits.

2. **Problem 2:** Compute the symplectic matrix for the gate sequence $S \cdot H \cdot S$.

3. **Problem 3:** Given $\mathbf{v} = (1,0,1|0,1,0)$ and $\mathbf{v}' = (0,1,1|1,0,1)$, compute $\langle \mathbf{v}, \mathbf{v}' \rangle_s$.

### Intermediate

4. **Problem 4:** Derive the symplectic matrix for SWAP gate on 2 qubits.

5. **Problem 5:** Show that the symplectic matrices form a group under multiplication.

6. **Problem 6:** For the 3-qubit bit-flip code, express the stabilizer generators $Z_1 Z_2$ and $Z_2 Z_3$ as binary vectors and verify they are orthogonal under $\langle \cdot, \cdot \rangle_s$.

### Challenging

7. **Problem 7:** Prove that $|Sp(2, \mathbb{F}_2)| = 6$ by enumeration and verify these correspond to the single-qubit Cliffords.

8. **Problem 8:** Given a random symplectic matrix, design an algorithm to decompose it into H, S, CNOT gates.

9. **Problem 9:** Show that the stabilizer of the $[[5,1,3]]$ code forms a 4-dimensional isotropic subspace in $\mathbb{F}_2^{10}$.

---

## Computational Lab

```python
"""
Day 703: Symplectic Representation of Clifford Gates
Week 101: Advanced Stabilizer Theory

Implements binary symplectic representation and Clifford gate operations.
"""

import numpy as np
from typing import Tuple, List
from itertools import product

class SymplecticAlgebra:
    """Symplectic algebra over F_2 for Clifford representation."""

    def __init__(self, n_qubits: int):
        """Initialize for n-qubit system."""
        self.n = n_qubits
        self.dim = 2 * n_qubits

        # Construct symplectic form matrix
        self.Lambda = self._build_symplectic_form()

    def _build_symplectic_form(self) -> np.ndarray:
        """Build the symplectic form matrix Lambda."""
        n = self.n
        Lambda = np.zeros((2*n, 2*n), dtype=int)
        Lambda[:n, n:] = np.eye(n, dtype=int)
        Lambda[n:, :n] = np.eye(n, dtype=int)
        return Lambda

    def pauli_to_binary(self, pauli_string: str) -> np.ndarray:
        """
        Convert Pauli string to binary vector.

        Args:
            pauli_string: e.g., "XZY" for X⊗Z⊗Y

        Returns:
            Binary vector (a1,...,an|b1,...,bn)
        """
        n = len(pauli_string)
        assert n == self.n, f"String length {n} != {self.n} qubits"

        vec = np.zeros(2*n, dtype=int)

        for i, p in enumerate(pauli_string):
            if p == 'X':
                vec[i] = 1      # a_i = 1, b_i = 0
            elif p == 'Z':
                vec[n + i] = 1  # a_i = 0, b_i = 1
            elif p == 'Y':
                vec[i] = 1      # a_i = 1
                vec[n + i] = 1  # b_i = 1
            elif p == 'I':
                pass            # a_i = 0, b_i = 0
            else:
                raise ValueError(f"Invalid Pauli: {p}")

        return vec

    def binary_to_pauli(self, vec: np.ndarray) -> str:
        """Convert binary vector back to Pauli string."""
        n = self.n
        result = []

        for i in range(n):
            a_i = vec[i] % 2
            b_i = vec[n + i] % 2

            if a_i == 0 and b_i == 0:
                result.append('I')
            elif a_i == 1 and b_i == 0:
                result.append('X')
            elif a_i == 0 and b_i == 1:
                result.append('Z')
            else:  # a_i == 1 and b_i == 1
                result.append('Y')

        return ''.join(result)

    def symplectic_inner_product(self, v1: np.ndarray, v2: np.ndarray) -> int:
        """Compute symplectic inner product <v1, v2>_s mod 2."""
        return (v1 @ self.Lambda @ v2) % 2

    def commutes(self, p1: str, p2: str) -> bool:
        """Check if two Paulis commute."""
        v1 = self.pauli_to_binary(p1)
        v2 = self.pauli_to_binary(p2)
        return self.symplectic_inner_product(v1, v2) == 0

    def is_symplectic(self, M: np.ndarray) -> bool:
        """Check if matrix M is symplectic (M^T Lambda M = Lambda mod 2)."""
        product = (M.T @ self.Lambda @ M) % 2
        return np.array_equal(product, self.Lambda)

    def apply_clifford(self, M: np.ndarray, pauli: str) -> str:
        """Apply Clifford (symplectic matrix) to Pauli."""
        vec = self.pauli_to_binary(pauli)
        new_vec = (M @ vec) % 2
        return self.binary_to_pauli(new_vec)


class CliffordGates:
    """Standard Clifford gates as symplectic matrices."""

    @staticmethod
    def H(qubit: int, n_qubits: int) -> np.ndarray:
        """Hadamard gate on specified qubit."""
        n = n_qubits
        M = np.eye(2*n, dtype=int)

        # Swap X and Z for this qubit
        M[qubit, qubit] = 0
        M[qubit, n + qubit] = 1
        M[n + qubit, qubit] = 1
        M[n + qubit, n + qubit] = 0

        return M

    @staticmethod
    def S(qubit: int, n_qubits: int) -> np.ndarray:
        """Phase gate on specified qubit."""
        n = n_qubits
        M = np.eye(2*n, dtype=int)

        # X -> XZ (Y), Z -> Z
        M[n + qubit, qubit] = 1

        return M

    @staticmethod
    def CNOT(control: int, target: int, n_qubits: int) -> np.ndarray:
        """CNOT with specified control and target qubits."""
        n = n_qubits
        M = np.eye(2*n, dtype=int)

        # X_c -> X_c X_t
        M[target, control] = 1

        # Z_t -> Z_c Z_t
        M[n + control, n + target] = 1

        return M

    @staticmethod
    def CZ(qubit1: int, qubit2: int, n_qubits: int) -> np.ndarray:
        """Controlled-Z gate."""
        n = n_qubits
        M = np.eye(2*n, dtype=int)

        # X_1 -> X_1 Z_2
        M[n + qubit2, qubit1] = 1

        # X_2 -> Z_1 X_2
        M[n + qubit1, qubit2] = 1

        return M

    @staticmethod
    def SWAP(qubit1: int, qubit2: int, n_qubits: int) -> np.ndarray:
        """SWAP gate."""
        n = n_qubits
        M = np.eye(2*n, dtype=int)

        # Swap X parts
        M[qubit1, qubit1] = 0
        M[qubit2, qubit2] = 0
        M[qubit1, qubit2] = 1
        M[qubit2, qubit1] = 1

        # Swap Z parts
        M[n + qubit1, n + qubit1] = 0
        M[n + qubit2, n + qubit2] = 0
        M[n + qubit1, n + qubit2] = 1
        M[n + qubit2, n + qubit1] = 1

        return M


def demonstrate_symplectic():
    """Demonstrate symplectic representation."""

    print("=" * 70)
    print("SYMPLECTIC REPRESENTATION OF CLIFFORD GATES")
    print("=" * 70)

    # Single qubit
    print("\n1. SINGLE-QUBIT BINARY REPRESENTATION")
    print("-" * 50)

    sa = SymplecticAlgebra(1)

    for pauli in ['I', 'X', 'Y', 'Z']:
        vec = sa.pauli_to_binary(pauli)
        print(f"  {pauli} -> ({vec[0]}|{vec[1]})")

    # Symplectic form
    print("\n2. SYMPLECTIC FORM MATRIX (n=2)")
    print("-" * 50)

    sa2 = SymplecticAlgebra(2)
    print(f"  Λ =\n{sa2.Lambda}")

    # Commutation check
    print("\n3. COMMUTATION VIA SYMPLECTIC PRODUCT")
    print("-" * 50)

    test_pairs = [('XX', 'ZZ'), ('XZ', 'ZX'), ('XY', 'YX'), ('XI', 'IX')]

    for p1, p2 in test_pairs:
        v1 = sa2.pauli_to_binary(p1)
        v2 = sa2.pauli_to_binary(p2)
        sip = sa2.symplectic_inner_product(v1, v2)
        comm = "commute" if sip == 0 else "anticommute"
        print(f"  <{p1}, {p2}>_s = {sip}  =>  {comm}")

    # Standard gate matrices
    print("\n4. SYMPLECTIC MATRICES FOR STANDARD GATES")
    print("-" * 50)

    gates = CliffordGates()

    print("  Hadamard (1 qubit):")
    M_H = gates.H(0, 1)
    print(f"    {M_H.tolist()}")

    print("\n  Phase S (1 qubit):")
    M_S = gates.S(0, 1)
    print(f"    {M_S.tolist()}")

    print("\n  CNOT (2 qubits, control=0, target=1):")
    M_CNOT = gates.CNOT(0, 1, 2)
    print(f"    {M_CNOT.tolist()}")

    # Verify symplectic property
    print("\n5. VERIFYING SYMPLECTIC PROPERTY")
    print("-" * 50)

    for name, M, n in [("H", gates.H(0, 1), 1),
                        ("S", gates.S(0, 1), 1),
                        ("CNOT", gates.CNOT(0, 1, 2), 2),
                        ("CZ", gates.CZ(0, 1, 2), 2)]:
        sa_test = SymplecticAlgebra(n)
        is_symp = sa_test.is_symplectic(M)
        print(f"  {name}: M^T Λ M = Λ ? {is_symp}")

    # Apply Cliffords to Paulis
    print("\n6. CLIFFORD ACTION ON PAULIS")
    print("-" * 50)

    print("  Single-qubit H gate:")
    for p in ['X', 'Y', 'Z']:
        new_p = sa.apply_clifford(gates.H(0, 1), p)
        print(f"    H {p} H† = {new_p}")

    print("\n  Single-qubit S gate:")
    for p in ['X', 'Y', 'Z']:
        new_p = sa.apply_clifford(gates.S(0, 1), p)
        print(f"    S {p} S† = {new_p}")

    print("\n  CNOT gate on 2 qubits:")
    for p in ['XI', 'IX', 'ZI', 'IZ']:
        new_p = sa2.apply_clifford(gates.CNOT(0, 1, 2), p)
        print(f"    CNOT {p} CNOT† = {new_p}")

    # Gate composition
    print("\n7. GATE COMPOSITION")
    print("-" * 50)

    # HS gate (S then H)
    M_HS = (gates.H(0, 1) @ gates.S(0, 1)) % 2
    print(f"  M_H · M_S (HS gate):\n    {M_HS.tolist()}")

    # Verify it's symplectic
    print(f"  Is symplectic? {sa.is_symplectic(M_HS)}")

    # Action on Paulis
    print("  HS action:")
    for p in ['X', 'Y', 'Z']:
        new_p = sa.apply_clifford(M_HS, p)
        print(f"    (HS) {p} (HS)† = {new_p}")

    # Count Sp(2, F_2)
    print("\n8. SIZE OF Sp(2, F_2)")
    print("-" * 50)

    count = 0
    symplectic_matrices = []

    for entries in product([0, 1], repeat=4):
        M = np.array(entries, dtype=int).reshape(2, 2)
        if sa.is_symplectic(M):
            count += 1
            symplectic_matrices.append(M)

    print(f"  Found {count} symplectic 2×2 matrices over F_2")
    print("  (Expected: 6, corresponding to single-qubit Cliffords / phases)")

    print("\n  The 6 matrices:")
    for i, M in enumerate(symplectic_matrices):
        print(f"    M_{i+1} = {M.flatten().tolist()}")

    # Stabilizer code example
    print("\n9. STABILIZER CODE IN SYMPLECTIC REPRESENTATION")
    print("-" * 50)

    print("  3-qubit bit-flip code stabilizers: ZZI, IZZ")

    sa3 = SymplecticAlgebra(3)
    g1 = sa3.pauli_to_binary("ZZI")
    g2 = sa3.pauli_to_binary("IZZ")

    print(f"    ZZI -> {g1}")
    print(f"    IZZ -> {g2}")

    sip = sa3.symplectic_inner_product(g1, g2)
    print(f"\n  <ZZI, IZZ>_s = {sip}  (0 = commute = valid stabilizer)")

    # Check orthogonality with logical operators
    print("\n  Logical operators: X_L = XXX, Z_L = ZII")
    xl = sa3.pauli_to_binary("XXX")
    zl = sa3.pauli_to_binary("ZII")

    print(f"    <ZZI, XXX>_s = {sa3.symplectic_inner_product(g1, xl)}")
    print(f"    <IZZ, XXX>_s = {sa3.symplectic_inner_product(g2, xl)}")
    print(f"    <XXX, ZII>_s = {sa3.symplectic_inner_product(xl, zl)}")
    print("  (All 0: logical ops commute with stabilizers but anticommute with each other)")


if __name__ == "__main__":
    demonstrate_symplectic()
```

**Expected Output:**
```
======================================================================
SYMPLECTIC REPRESENTATION OF CLIFFORD GATES
======================================================================

1. SINGLE-QUBIT BINARY REPRESENTATION
--------------------------------------------------
  I -> (0|0)
  X -> (1|0)
  Y -> (1|1)
  Z -> (0|1)

2. SYMPLECTIC FORM MATRIX (n=2)
--------------------------------------------------
  Λ =
[[0 0 1 0]
 [0 0 0 1]
 [1 0 0 0]
 [0 1 0 0]]

3. COMMUTATION VIA SYMPLECTIC PRODUCT
--------------------------------------------------
  <XX, ZZ>_s = 0  =>  commute
  <XZ, ZX>_s = 0  =>  commute
  <XY, YX>_s = 0  =>  commute
  <XI, IX>_s = 0  =>  commute

4. SYMPLECTIC MATRICES FOR STANDARD GATES
--------------------------------------------------
  Hadamard (1 qubit):
    [[0, 1], [1, 0]]

  Phase S (1 qubit):
    [[1, 0], [1, 1]]

  CNOT (2 qubits, control=0, target=1):
    [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]

5. VERIFYING SYMPLECTIC PROPERTY
--------------------------------------------------
  H: M^T Λ M = Λ ? True
  S: M^T Λ M = Λ ? True
  CNOT: M^T Λ M = Λ ? True
  CZ: M^T Λ M = Λ ? True

6. CLIFFORD ACTION ON PAULIS
--------------------------------------------------
  Single-qubit H gate:
    H X H† = Z
    H Y H† = Y
    H Z H† = X

  Single-qubit S gate:
    S X S† = Y
    S Y S† = X
    S Z S† = Z

  CNOT gate on 2 qubits:
    CNOT XI CNOT† = XX
    CNOT IX CNOT† = IX
    CNOT ZI CNOT† = ZI
    CNOT IZ CNOT† = ZZ
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| **Binary representation** | $P = X^{\mathbf{a}} Z^{\mathbf{b}} \leftrightarrow (\mathbf{a}\|\mathbf{b}) \in \mathbb{F}_2^{2n}$ |
| **Symplectic form** | $\Lambda = \begin{pmatrix} 0 & I_n \\ I_n & 0 \end{pmatrix}$ |
| **Symplectic product** | $\langle \mathbf{v}, \mathbf{v}' \rangle_s = \mathbf{a} \cdot \mathbf{b}' + \mathbf{a}' \cdot \mathbf{b} \pmod{2}$ |
| **Commutation** | $PP' = (-1)^{\langle \mathbf{v}, \mathbf{v}' \rangle_s} P'P$ |
| **Symplectic condition** | $M^T \Lambda M = \Lambda$ |
| **Clifford isomorphism** | $\mathcal{C}_n / \mathcal{P}_n \cong Sp(2n, \mathbb{F}_2) \ltimes \mathbb{F}_2^{2n}$ |

### Main Takeaways

1. **Binary encoding** reduces Pauli operators to vectors in $\mathbb{F}_2^{2n}$
2. **Symplectic product** captures commutation: 0 = commute, 1 = anticommute
3. **Clifford gates** act as symplectic matrices preserving the symplectic form
4. **Gate composition** is matrix multiplication over $\mathbb{F}_2$
5. **Stabilizer codes** correspond to isotropic subspaces

---

## Daily Checklist

- [ ] Understand binary representation of Pauli operators
- [ ] Master the symplectic inner product and its physical meaning
- [ ] Derive symplectic matrices for H, S, CNOT
- [ ] Verify symplectic condition for standard gates
- [ ] Implement symplectic algebra computationally
- [ ] Connect to stabilizer codes via isotropic subspaces

---

## Preview: Day 704

Tomorrow we explore **Clifford Circuits and Classical Simulation**, where the symplectic representation enables efficient classical simulation via the Gottesman-Knill theorem. We'll see how:
- Stabilizer tableaux track quantum states classically
- $O(n^2)$ operations suffice per gate
- Measurements reduce to linear algebra over $\mathbb{F}_2$
