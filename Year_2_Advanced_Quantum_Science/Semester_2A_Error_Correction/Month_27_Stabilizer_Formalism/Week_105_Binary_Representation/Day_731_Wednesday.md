# Day 731: Symplectic Inner Product

## Overview

**Day:** 731 of 1008
**Week:** 105 (Binary Representation & F₂ Linear Algebra)
**Month:** 27 (Stabilizer Formalism)
**Topic:** The Symplectic Form and Commutation Relations

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Symplectic form theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Lagrangian subspaces |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define** the symplectic inner product on F₂^{2n}
2. **Prove** the connection between symplectic orthogonality and commutation
3. **Characterize** isotropic and Lagrangian subspaces
4. **Explain** why stabilizer groups are isotropic
5. **Understand** the symplectic group Sp(2n, F₂)
6. **Connect** Clifford operations to symplectic transformations

---

## Core Content

### The Symplectic Form

**Definition (Symplectic Inner Product):**
For vectors $v = (\mathbf{a}|\mathbf{b})$ and $w = (\mathbf{c}|\mathbf{d})$ in F₂^{2n}:

$$\boxed{\langle v, w \rangle_s = \mathbf{a} \cdot \mathbf{d} + \mathbf{b} \cdot \mathbf{c} = \sum_{i=1}^n (a_i d_i + b_i c_i) \pmod{2}}$$

**Matrix Form:**
$$\langle v, w \rangle_s = v^T \Omega w$$

where the **symplectic matrix** is:
$$\Omega = \begin{pmatrix} 0 & I_n \\ I_n & 0 \end{pmatrix}$$

### Properties of the Symplectic Form

**Bilinearity:**
$$\langle v_1 + v_2, w \rangle_s = \langle v_1, w \rangle_s + \langle v_2, w \rangle_s$$
$$\langle v, w_1 + w_2 \rangle_s = \langle v, w_1 \rangle_s + \langle v, w_2 \rangle_s$$

**Antisymmetry:**
$$\langle v, w \rangle_s = \langle w, v \rangle_s$$

Note: Over F₂, antisymmetry is the same as symmetry since -1 = 1.

**Non-degeneracy:**
If $\langle v, w \rangle_s = 0$ for all $w$, then $v = 0$.

**Not Positive Definite:**
$$\langle v, v \rangle_s = 2 \mathbf{a} \cdot \mathbf{b} = 0 \pmod{2}$$

Every vector is self-orthogonal under the symplectic form!

### The Fundamental Theorem

**Theorem (Commutation-Symplectic Correspondence):**
For Pauli operators $P_1 \leftrightarrow v_1$ and $P_2 \leftrightarrow v_2$:

$$\boxed{P_1 P_2 = (-1)^{\langle v_1, v_2 \rangle_s} P_2 P_1}$$

Therefore:
- $\langle v_1, v_2 \rangle_s = 0$ ⟺ $P_1$ and $P_2$ commute
- $\langle v_1, v_2 \rangle_s = 1$ ⟺ $P_1$ and $P_2$ anticommute

**Proof:**
Consider single qubits. For $P_1 = X^{a_1}Z^{b_1}$ and $P_2 = X^{a_2}Z^{b_2}$:
$$P_1 P_2 = X^{a_1}Z^{b_1}X^{a_2}Z^{b_2}$$

Moving $Z^{b_1}$ past $X^{a_2}$:
$$ZX = -XZ \implies Z^{b_1}X^{a_2} = (-1)^{b_1 a_2} X^{a_2}Z^{b_1}$$

Similarly, moving $X^{a_1}$ past $Z^{b_2}$:
$$P_2 P_1 = X^{a_2}Z^{b_2}X^{a_1}Z^{b_1} = (-1)^{a_1 b_2} X^{a_1}X^{a_2}Z^{b_1}Z^{b_2}$$

Comparing:
$$P_1 P_2 = (-1)^{b_1 a_2 + a_1 b_2} P_2 P_1 = (-1)^{\langle v_1, v_2 \rangle_s} P_2 P_1$$

The tensor product extends this to n qubits. □

### Isotropic Subspaces

**Definition (Isotropic Subspace):**
A subspace V ⊆ F₂^{2n} is **isotropic** if:
$$\langle v, w \rangle_s = 0 \text{ for all } v, w \in V$$

Equivalently: all pairs of vectors in V are symplectically orthogonal.

**Physical Meaning:** An isotropic subspace corresponds to a set of mutually commuting Pauli operators.

**Dimension Bound:**
For an isotropic subspace V ⊆ F₂^{2n}:
$$\dim(V) \leq n$$

**Proof:** Consider the map $\phi: V \to V^*$ given by $\phi(v)(w) = \langle v, w \rangle_s$. For isotropic V, this map is zero, so V ⊆ V^{⊥_s}$ (symplectic orthogonal complement). Since dim(V^{⊥_s}) = 2n - dim(V), we have:
$$\dim(V) \leq 2n - \dim(V) \implies \dim(V) \leq n$$

### Lagrangian Subspaces

**Definition (Lagrangian Subspace):**
A subspace L ⊆ F₂^{2n} is **Lagrangian** if:
1. L is isotropic
2. dim(L) = n (maximal dimension)

Equivalently: $L = L^{\perp_s}$ (self-dual under symplectic form).

**Importance:** Lagrangian subspaces are in bijection with stabilizer states!

**Theorem (Stabilizer-Lagrangian Correspondence):**
An [[n, 0, d]] stabilizer code (encoding 0 qubits, i.e., a single state) has stabilizer group S whose binary representation forms a Lagrangian subspace.

### Stabilizer Codes and Isotropy

**Theorem (Stabilizer Codes are Isotropic):**
The binary representation of stabilizer generators for any [[n, k, d]] code forms an isotropic subspace of dimension n-k.

**Proof:**
- Stabilizer generators must mutually commute: $[S_i, S_j] = 0$
- By the fundamental theorem: $\langle v_i, v_j \rangle_s = 0$
- There are n-k independent generators
- Their span is isotropic of dimension n-k

### Symplectic Orthogonal Complement

**Definition:**
For subspace V ⊆ F₂^{2n}:
$$V^{\perp_s} = \{w \in \mathbb{F}_2^{2n} : \langle v, w \rangle_s = 0 \text{ for all } v \in V\}$$

**Dimension Formula:**
$$\dim(V) + \dim(V^{\perp_s}) = 2n$$

**Characterization of Logical Operators:**
For a stabilizer code with stabilizer space S:
- Centralizer: $C(S) = S^{\perp_s}$ (all operators commuting with S)
- Logical operators: $C(S) \setminus S$

### The Symplectic Group

**Definition:**
The symplectic group Sp(2n, F₂) consists of matrices M ∈ GL(2n, F₂) that preserve the symplectic form:
$$M^T \Omega M = \Omega$$

**Properties:**
- |Sp(2n, F₂)| = 2^{n²} ∏_{i=1}^n (4^i - 1)
- Sp(2, F₂) ≅ S₃ (symmetric group on 3 elements)
- Sp(4, F₂) ≅ S₆

### Clifford Group and Symplectic Group

**Theorem (Clifford-Symplectic Correspondence):**
The Clifford group C_n modulo phases acts on the Pauli group by conjugation. This action corresponds to the symplectic group:
$$\mathcal{C}_n / \{\text{phases}\} \cong \text{Sp}(2n, \mathbb{F}_2)$$

**Explicitly:** For Clifford unitary U and Pauli P with binary vector v:
$$U P U^\dagger = \pm P' \text{ where } P' \leftrightarrow M_U v$$

and $M_U \in \text{Sp}(2n, \mathbb{F}_2)$.

### Clifford Generators in Symplectic Form

**Hadamard on qubit j:**
$$M_H = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}_j \text{ (swaps X and Z)}$$

**Phase gate S on qubit j:**
$$M_S = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}_j \text{ (X → Y = XZ)}$$

**CNOT from i to j:**
$$M_{CNOT} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}_{X} \otimes \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}_{Z}$$

---

## Worked Examples

### Example 1: Verifying Commutation via Symplectic Product

Check if $P_1 = X_1 Y_2 Z_3$ and $P_2 = Z_1 X_2 Y_3$ commute.

**Binary vectors:**
- $P_1 = X_1 Y_2 Z_3 \leftrightarrow v_1 = (1, 1, 0 | 0, 1, 1)$
- $P_2 = Z_1 X_2 Y_3 \leftrightarrow v_2 = (0, 1, 1 | 1, 0, 1)$

**Symplectic product:**
$$\langle v_1, v_2 \rangle_s = \mathbf{a}_1 \cdot \mathbf{b}_2 + \mathbf{b}_1 \cdot \mathbf{a}_2$$
$$= (1, 1, 0) \cdot (1, 0, 1) + (0, 1, 1) \cdot (0, 1, 1)$$
$$= (1 + 0 + 0) + (0 + 1 + 1) = 1 + 0 = 1$$

Since $\langle v_1, v_2 \rangle_s = 1$, the operators **anticommute**.

**Verification:**
$P_1 P_2 = (X \otimes Y \otimes Z)(Z \otimes X \otimes Y)$
$= XZ \otimes YX \otimes ZY$
$= (iY) \otimes (-iZ) \otimes (-iX)$
$= -i^3 Y \otimes Z \otimes X = -(-i) YZX = i YZX$

$P_2 P_1 = (Z \otimes X \otimes Y)(X \otimes Y \otimes Z)$
$= ZX \otimes XY \otimes YZ$
$= (-iY) \otimes (iZ) \otimes (iX)$
$= -i^3 YZX = i YZX$

Wait, let me recalculate more carefully...

$XZ = iY$, $ZX = -iY$ → $XZ = -ZX$
$YX = -iZ$, $XY = iZ$ → $YX = -XY$
$ZY = -iX$, $YZ = iX$ → $ZY = -YZ$

$P_1 P_2 = (XZ)(YX)(ZY) = (iY)(-iZ)(-iX) = -i Y Z X$
$P_2 P_1 = (ZX)(XY)(YZ) = (-iY)(iZ)(iX) = i Y Z X$

$P_1 P_2 = -P_2 P_1$ ✓ Anticommute confirmed!

### Example 2: Finding the Symplectic Complement

Let V = span{(1,0,0|0,0,1), (0,1,0|0,1,0)} in F₂^6 (3 qubits).

These correspond to $X_1 Z_3$ and $X_2 Z_2 = Y_2$.

**Find V^{⊥_s}:**

A vector $w = (c_1, c_2, c_3 | d_1, d_2, d_3)$ is in V^{⊥_s} iff:
$$\langle (1,0,0|0,0,1), w \rangle_s = 0$$
$$\langle (0,1,0|0,1,0), w \rangle_s = 0$$

First condition:
$$1 \cdot d_1 + 0 \cdot d_2 + 0 \cdot d_3 + 0 \cdot c_1 + 0 \cdot c_2 + 1 \cdot c_3 = 0$$
$$d_1 + c_3 = 0 \pmod{2}$$

Second condition:
$$0 \cdot d_1 + 1 \cdot d_2 + 0 \cdot d_3 + 0 \cdot c_1 + 1 \cdot c_2 + 0 \cdot c_3 = 0$$
$$d_2 + c_2 = 0 \pmod{2}$$

So $d_1 = c_3$ and $d_2 = c_2$. The complement is:
$$V^{\perp_s} = \{(c_1, c_2, c_3 | c_3, c_2, d_3) : c_1, c_2, c_3, d_3 \in \mathbb{F}_2\}$$

Basis: {(1,0,0|0,0,0), (0,1,0|0,1,0), (0,0,1|1,0,0), (0,0,0|0,0,1)}

dim(V^{⊥_s}) = 4 = 6 - 2 = 2n - dim(V) ✓

### Example 3: Hadamard in Symplectic Form

The Hadamard transforms: $H X H^\dagger = Z$, $H Z H^\dagger = X$.

**Binary representation:**
- $X \leftrightarrow (1|0)$
- $Z \leftrightarrow (0|1)$

Hadamard's symplectic matrix:
$$M_H = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**Verification:**
$$M_H \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ ✓ (X → Z)

$$M_H \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ ✓ (Z → X)

**Check symplectic:**
$$M_H^T \Omega M_H = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$
$$= \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \Omega$$ ✓

---

## Practice Problems

### Level 1: Direct Application

1. **Symplectic Product:** Compute $\langle v_1, v_2 \rangle_s$ for:
   a) $v_1 = (1,1|0,0)$, $v_2 = (0,0|1,1)$
   b) $v_1 = (1,0,1|0,1,0)$, $v_2 = (0,1,0|1,0,1)$
   c) $v_1 = (1,1,1,1|0,0,0,0)$, $v_2 = (0,0,0,0|1,1,1,1)$

2. **Isotropy Check:** Determine if the following sets are isotropic:
   a) {(1,0|0,1), (0,1|1,0)}
   b) {(1,1|0,0), (0,0|1,1), (1,1|1,1)}
   c) The stabilizers of the [[4,2,2]] code: {XXXX, ZZZZ}

3. **Self-Orthogonality:** Show that every vector in F₂^{2n} is symplectically self-orthogonal.

### Level 2: Intermediate

4. **Symplectic Complement:** Find a basis for V^{⊥_s} where:
   a) V = span{(1,0,0,0|0,0,0,1)} in F₂^8 (n=4)
   b) V = span{(1,1|1,1)} in F₂^4

5. **Lagrangian Check:** Is the following a Lagrangian subspace?
   $$L = \text{span}\{(1,0|0,0), (0,1|0,0), (0,0|1,0), (0,0|0,1)\}$$

6. **CNOT Symplectic Matrix:** Write the 4×4 symplectic matrix for CNOT from qubit 1 to qubit 2.

### Level 3: Challenging

7. **Stabilizer Code Logicals:** For the [[5,1,3]] code with stabilizers:
   - $S_1 = XZZXI$, $S_2 = IXZZX$, $S_3 = XIXZZ$, $S_4 = ZXIXZ$

   a) Verify the stabilizers form an isotropic subspace.
   b) Find $S^{\perp_s}$.
   c) Identify logical $\bar{X}$ and $\bar{Z}$ operators.

8. **Symplectic Group Size:** Compute |Sp(4, F₂)| using the formula:
   $$|\text{Sp}(2n, \mathbb{F}_2)| = 2^{n^2} \prod_{i=1}^n (4^i - 1)$$

9. **Clifford Generators:** Prove that the symplectic matrices for H, S, and CNOT generate Sp(2n, F₂) for any n.

---

## Solutions

### Level 1 Solutions

1. **Symplectic Products:**
   a) $\langle(1,1|0,0), (0,0|1,1)\rangle_s = (1,1)\cdot(1,1) + (0,0)\cdot(0,0) = 0$
   b) $(1,0,1)\cdot(1,0,1) + (0,1,0)\cdot(0,1,0) = 0 + 1 = 1$
   c) $(1,1,1,1)\cdot(1,1,1,1) + 0 = 0$ (even weight)

2. **Isotropy:**
   a) $\langle(1,0|0,1), (0,1|1,0)\rangle_s = 1\cdot1 + 0\cdot0 + 0\cdot1 + 1\cdot0 = 1$ → Not isotropic
   b) Check all pairs:
      - $\langle(1,1|0,0), (0,0|1,1)\rangle_s = 0$
      - $\langle(1,1|0,0), (1,1|1,1)\rangle_s = 0$
      - $\langle(0,0|1,1), (1,1|1,1)\rangle_s = 0$
      → Isotropic!
   c) XXXX = (1,1,1,1|0,0,0,0), ZZZZ = (0,0,0,0|1,1,1,1)
      $\langle \cdot, \cdot \rangle_s = 0$ → Isotropic ✓

3. **Self-orthogonality:**
   $\langle(a|b), (a|b)\rangle_s = a\cdot b + b\cdot a = 2(a\cdot b) = 0 \pmod{2}$ ✓

---

## Computational Lab

```python
"""
Day 731: Symplectic Inner Product and Isotropic Subspaces
==========================================================
Implementation of symplectic geometry over F₂.
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations

def mod2(M: np.ndarray) -> np.ndarray:
    """Reduce mod 2."""
    return M % 2

def symplectic_matrix(n: int) -> np.ndarray:
    """
    Create the 2n × 2n symplectic form matrix Ω.

    Ω = [[0, I_n], [I_n, 0]]
    """
    return np.block([
        [np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
        [np.eye(n, dtype=int), np.zeros((n, n), dtype=int)]
    ])

def symplectic_inner_product(v1: np.ndarray, v2: np.ndarray) -> int:
    """
    Compute symplectic inner product <v1, v2>_s.

    <v1, v2>_s = a1·b2 + b1·a2 (mod 2)
    where v1 = (a1|b1), v2 = (a2|b2)
    """
    n = len(v1) // 2
    a1, b1 = v1[:n], v1[n:]
    a2, b2 = v2[:n], v2[n:]
    return (np.dot(a1, b2) + np.dot(b1, a2)) % 2

def symplectic_inner_product_matrix(v1: np.ndarray, v2: np.ndarray,
                                     Omega: np.ndarray) -> int:
    """Compute <v1, v2>_s using matrix form: v1^T Ω v2."""
    return int(mod2(v1 @ Omega @ v2))

def is_isotropic(vectors: np.ndarray) -> bool:
    """
    Check if set of vectors forms an isotropic subspace.

    All pairs must have symplectic inner product 0.
    """
    n_vecs = vectors.shape[0]

    for i in range(n_vecs):
        for j in range(i, n_vecs):  # Include self for completeness
            if symplectic_inner_product(vectors[i], vectors[j]) != 0:
                return False
    return True

def is_lagrangian(vectors: np.ndarray) -> bool:
    """
    Check if vectors span a Lagrangian subspace.

    Must be isotropic and have dimension n (where space is F_2^{2n}).
    """
    dim_space = vectors.shape[1]
    n = dim_space // 2

    if not is_isotropic(vectors):
        return False

    # Check dimension equals n
    # Compute rank
    from Day_730_Tuesday import rank_f2
    rank = rank_f2(vectors)
    return rank == n

def symplectic_complement(V: np.ndarray) -> np.ndarray:
    """
    Compute symplectic complement V^{⊥_s}.

    V^{⊥_s} = {w : <v, w>_s = 0 for all v in V}
    = ker(V Ω)^T where Ω is symplectic matrix
    """
    dim_space = V.shape[1]
    n = dim_space // 2
    Omega = symplectic_matrix(n)

    # V^{⊥_s} = ker((V Ω)^T) = ker(Ω^T V^T) = ker(Ω V^T)
    # Since Ω^T = Ω for our convention
    M = mod2(V @ Omega)

    from Day_730_Tuesday import null_space_f2
    return null_space_f2(M.T)

def is_symplectic_matrix(M: np.ndarray) -> bool:
    """
    Check if M is a symplectic matrix: M^T Ω M = Ω.
    """
    dim = M.shape[0]
    n = dim // 2
    Omega = symplectic_matrix(n)
    result = mod2(M.T @ Omega @ M)
    return np.array_equal(result, Omega)

def hadamard_symplectic(qubit: int, n_qubits: int) -> np.ndarray:
    """
    Symplectic matrix for Hadamard on specified qubit.

    H: X ↔ Z (swaps X and Z parts)
    """
    M = np.eye(2 * n_qubits, dtype=int)

    # Swap rows/columns for X and Z parts
    # For qubit j: swap position j with position j+n
    M[qubit, qubit] = 0
    M[qubit, qubit + n_qubits] = 1
    M[qubit + n_qubits, qubit + n_qubits] = 0
    M[qubit + n_qubits, qubit] = 1

    return M

def phase_symplectic(qubit: int, n_qubits: int) -> np.ndarray:
    """
    Symplectic matrix for Phase gate S on specified qubit.

    S: X → Y = XZ (adds Z), Z → Z
    In binary: (1|0) → (1|1), (0|1) → (0|1)
    """
    M = np.eye(2 * n_qubits, dtype=int)

    # X part stays, but Z part gains X part
    # z_new = z_old + x_old
    M[qubit + n_qubits, qubit] = 1

    return M

def cnot_symplectic(control: int, target: int, n_qubits: int) -> np.ndarray:
    """
    Symplectic matrix for CNOT from control to target.

    CNOT: X_c → X_c X_t, Z_t → Z_c Z_t
    """
    M = np.eye(2 * n_qubits, dtype=int)

    # X part: x_t_new = x_t_old + x_c_old
    M[target, control] = 1

    # Z part: z_c_new = z_c_old + z_t_old
    M[control + n_qubits, target + n_qubits] = 1

    return M

def apply_symplectic(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply symplectic transformation M to vector v."""
    return mod2(M @ v)

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 731: Symplectic Inner Product")
    print("=" * 60)

    # Example 1: Basic symplectic inner products
    print("\n1. Symplectic Inner Products")
    print("-" * 40)

    test_pairs = [
        ("X⊗X", "Z⊗Z", np.array([1,1,0,0]), np.array([0,0,1,1])),
        ("X⊗Z", "Z⊗X", np.array([1,0,0,1]), np.array([0,1,1,0])),
        ("Y⊗Y", "X⊗X", np.array([1,1,1,1]), np.array([1,1,0,0])),
    ]

    for name1, name2, v1, v2 in test_pairs:
        sip = symplectic_inner_product(v1, v2)
        comm_status = "commute" if sip == 0 else "anticommute"
        print(f"<{name1}, {name2}>_s = {sip} → {comm_status}")

    # Example 2: Isotropy check
    print("\n2. Isotropy Check for Stabilizer Codes")
    print("-" * 40)

    # [[4,2,2]] code stabilizers
    H_422 = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],  # XXXX
        [0, 0, 0, 0, 1, 1, 1, 1]   # ZZZZ
    ])

    print("[[4,2,2]] code stabilizers: XXXX, ZZZZ")
    print(f"Is isotropic: {is_isotropic(H_422)}")

    # [[5,1,3]] code
    H_513 = np.array([
        [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],  # XZZXI
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # IXZZX
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],  # XIXZZ
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]   # ZXIXZ
    ])

    print("\n[[5,1,3]] code stabilizers:")
    print(f"Is isotropic: {is_isotropic(H_513)}")

    # Example 3: Symplectic complement
    print("\n3. Symplectic Complement")
    print("-" * 40)

    V = np.array([[1, 0, 0, 1]])  # XZ on 2 qubits
    print(f"V = span of {V}")

    V_perp = symplectic_complement(V)
    print(f"V^⊥_s has dimension {len(V_perp)}")
    print(f"V^⊥_s basis:")
    print(V_perp)

    # Verify orthogonality
    print("\nVerification:")
    for w in V_perp:
        sip = symplectic_inner_product(V[0], w)
        print(f"  <{V[0]}, {w}>_s = {sip}")

    # Example 4: Clifford gates as symplectic matrices
    print("\n4. Clifford Gates as Symplectic Matrices")
    print("-" * 40)

    n = 2  # 2 qubits

    # Hadamard on qubit 0
    M_H = hadamard_symplectic(0, n)
    print("Hadamard on qubit 0:")
    print(M_H)
    print(f"Is symplectic: {is_symplectic_matrix(M_H)}")

    # Test: H transforms X → Z
    x0 = np.array([1, 0, 0, 0])  # X on qubit 0
    z0 = np.array([0, 0, 1, 0])  # Z on qubit 0
    print(f"\nH transforms X_0: {x0} → {apply_symplectic(M_H, x0)}")
    print(f"Expected Z_0: {z0}")

    # Phase gate on qubit 0
    M_S = phase_symplectic(0, n)
    print("\nPhase S on qubit 0:")
    print(M_S)
    print(f"Is symplectic: {is_symplectic_matrix(M_S)}")

    # Test: S transforms X → Y (binary: (1|0) → (1|1))
    print(f"\nS transforms X_0: {x0} → {apply_symplectic(M_S, x0)}")
    print(f"Expected Y_0 = (1,0|1,0): [1 0 1 0]")

    # CNOT from 0 to 1
    M_CNOT = cnot_symplectic(0, 1, n)
    print("\nCNOT from qubit 0 to qubit 1:")
    print(M_CNOT)
    print(f"Is symplectic: {is_symplectic_matrix(M_CNOT)}")

    # Test CNOT transformations
    x0 = np.array([1, 0, 0, 0])  # X_0
    z1 = np.array([0, 0, 0, 1])  # Z_1
    print(f"\nCNOT transforms X_0: {x0} → {apply_symplectic(M_CNOT, x0)}")
    print(f"Expected X_0 X_1: [1 1 0 0]")
    print(f"CNOT transforms Z_1: {z1} → {apply_symplectic(M_CNOT, z1)}")
    print(f"Expected Z_0 Z_1: [0 0 1 1]")

    # Example 5: Lagrangian subspace check
    print("\n5. Lagrangian Subspace Check")
    print("-" * 40)

    # Standard Lagrangian: all X operators
    L_X = np.array([
        [1, 0, 0, 0],  # X_0
        [0, 1, 0, 0]   # X_1
    ])
    print("L_X = span{X_0, X_1}:")
    print(f"  Is isotropic: {is_isotropic(L_X)}")
    print(f"  Dimension: {np.linalg.matrix_rank(L_X)}")
    print(f"  Is Lagrangian: {is_isotropic(L_X) and np.linalg.matrix_rank(L_X) == 2}")

    # Example 6: Commutation matrix for code
    print("\n6. Commutation Matrix for [[5,1,3]] Code")
    print("-" * 40)

    n_stabs = H_513.shape[0]
    comm_matrix = np.zeros((n_stabs, n_stabs), dtype=int)
    for i in range(n_stabs):
        for j in range(n_stabs):
            comm_matrix[i, j] = symplectic_inner_product(H_513[i], H_513[j])

    print("Commutation matrix (0 = commute):")
    print(comm_matrix)

    print("\n" + "=" * 60)
    print("End of Day 731 Lab")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Symplectic product | $\langle v, w \rangle_s = \mathbf{a} \cdot \mathbf{d} + \mathbf{b} \cdot \mathbf{c}$ |
| Matrix form | $\langle v, w \rangle_s = v^T \Omega w$ |
| Commutation | $P_1 P_2 = (-1)^{\langle v_1, v_2 \rangle_s} P_2 P_1$ |
| Isotropic | $\dim(V) \leq n$ for isotropic $V \subseteq \mathbb{F}_2^{2n}$ |
| Lagrangian | $L = L^{\perp_s}$ and $\dim(L) = n$ |

### Main Takeaways

1. **The symplectic inner product** encodes commutation relations
2. **Stabilizer codes** correspond to isotropic subspaces
3. **Stabilizer states** correspond to Lagrangian subspaces
4. **The Clifford group** acts as the symplectic group on binary vectors
5. **Logical operators** live in the symplectic complement minus stabilizers

---

## Daily Checklist

- [ ] I can compute the symplectic inner product
- [ ] I understand why $\langle v, v \rangle_s = 0$ always
- [ ] I can check if a subspace is isotropic
- [ ] I can characterize Lagrangian subspaces
- [ ] I understand the Clifford-symplectic correspondence
- [ ] I can write symplectic matrices for basic Clifford gates

---

## Preview: Day 732

Tomorrow we explore the **GF(4) Representation**:
- The field GF(4) = {0, 1, ω, ω̄}
- Representing single-qubit Paulis as GF(4) elements
- Trace function and Hermitian form
- Connection to quantum codes over GF(4)
- Additive vs multiplicative code theory
