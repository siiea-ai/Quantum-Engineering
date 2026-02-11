# Day 289: Reducible and Irreducible Representations

## Overview

**Month 11, Week 42, Day 2 — Tuesday**

Today we explore the fundamental distinction between **reducible** and **irreducible** representations. A representation is irreducible if it cannot be decomposed into smaller pieces—these are the "atoms" of representation theory. Understanding this decomposition is crucial for quantum mechanics, where irreducible representations correspond to multiplets of degenerate states.

## Prerequisites

From Day 288:
- Definition of group representations
- Matrix representations and equivalence
- Basic examples of representations

## Learning Objectives

By the end of today, you will be able to:

1. Define invariant subspaces and identify them
2. Distinguish reducible from irreducible representations
3. State and apply Maschke's theorem
4. Decompose representations into irreducible components
5. Connect irreducibility to quantum mechanical degeneracy
6. Implement computational tests for reducibility

---

## 1. Invariant Subspaces

### Definition

**Definition:** A subspace $W \subseteq V$ is **invariant** under representation $\rho: G \to GL(V)$ if:

$$\boxed{\rho(g)(W) \subseteq W \quad \forall g \in G}$$

Equivalently: if $w \in W$, then $\rho(g)w \in W$ for all $g$.

### Examples of Invariant Subspaces

**Example 1:** The zero subspace $\{0\}$ and the full space $V$ are always invariant (trivially).

**Example 2:** For the permutation representation of $S_n$ on $\mathbb{C}^n$:
- The "diagonal" subspace $\mathbb{C}(1, 1, \ldots, 1)$ is invariant
- The "zero-sum" subspace $\{(x_1, \ldots, x_n) : \sum x_i = 0\}$ is invariant

**Example 3:** For any representation, eigenspaces of commuting operators are invariant.

### Finding Invariant Subspaces

If $W$ is invariant, then in a basis where $\{e_1, \ldots, e_k\}$ span $W$:

$$D(g) = \begin{pmatrix} A(g) & B(g) \\ 0 & C(g) \end{pmatrix}$$

The representation restricted to $W$ has matrices $A(g)$.

---

## 2. Reducible and Irreducible Representations

### Definitions

**Definition:** A representation $\rho: G \to GL(V)$ is:

- **Reducible** if there exists a proper invariant subspace $W$ (i.e., $\{0\} \subsetneq W \subsetneq V$)
- **Irreducible** (or **irrep**) if no such subspace exists

**Definition:** A representation is **completely reducible** (or **semisimple**) if it decomposes as a direct sum of irreducible representations:

$$V = V_1 \oplus V_2 \oplus \cdots \oplus V_k$$

where each $V_i$ is an irreducible invariant subspace.

### Block Diagonal Form

A completely reducible representation can be put in block diagonal form:

$$D(g) = \begin{pmatrix} D_1(g) & 0 & \cdots & 0 \\ 0 & D_2(g) & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & D_k(g) \end{pmatrix}$$

where each block $D_i$ is an irreducible representation.

---

## 3. Maschke's Theorem

### Statement

**Theorem (Maschke):** Every representation of a finite group over $\mathbb{C}$ (or any field of characteristic not dividing $|G|$) is completely reducible.

### Proof Idea

**Key Insight:** If $W$ is an invariant subspace, we can find an invariant complement.

*Proof Sketch:*
1. Let $W \subseteq V$ be an invariant subspace
2. Choose any complement $W'$ (not necessarily invariant)
3. Define the projection $P: V \to W$ along $W'$
4. Average over the group: $\tilde{P} = \frac{1}{|G|} \sum_{g \in G} \rho(g) P \rho(g)^{-1}$
5. $\tilde{P}$ is still a projection onto $W$
6. $\ker(\tilde{P})$ is an invariant complement to $W$ ∎

### Significance

Maschke's theorem guarantees that understanding irreducible representations is sufficient—every representation decomposes into irreps.

**Note:** Maschke fails for infinite groups and for fields of characteristic dividing $|G|$ (modular representation theory).

---

## 4. Counting Irreducible Representations

### For Finite Groups

**Theorem:** The number of inequivalent irreducible representations of a finite group $G$ equals the number of conjugacy classes of $G$.

**Theorem:** If $d_1, d_2, \ldots, d_r$ are the dimensions of the irreps, then:

$$\boxed{\sum_{i=1}^{r} d_i^2 = |G|}$$

### Examples

**$S_3$:** 3 conjugacy classes → 3 irreps.
Dimensions satisfy $d_1^2 + d_2^2 + d_3^2 = 6$.
Solution: $1^2 + 1^2 + 2^2 = 6$ ✓

**$\mathbb{Z}_n$:** $n$ conjugacy classes (each element is its own class).
All irreps are 1-dimensional: $1^2 \times n = n$ ✓

**$D_4$:** 5 conjugacy classes → 5 irreps.
$|D_4| = 8$, so $d_1^2 + \cdots + d_5^2 = 8$.
Solution: $1^2 + 1^2 + 1^2 + 1^2 + 2^2 = 8$ ✓

---

## 5. Finding Irreducible Components

### The General Strategy

Given a reducible representation:
1. Find invariant subspaces
2. Decompose into direct sum
3. Repeat until all components are irreducible

### Using Characters (Preview)

The decomposition of a representation into irreps can be computed using characters:

$$\rho \cong \bigoplus_{\alpha} n_\alpha \rho_\alpha$$

where $n_\alpha$ is the **multiplicity** of irrep $\rho_\alpha$:

$$n_\alpha = \frac{1}{|G|} \sum_{g \in G} \overline{\chi_\alpha(g)} \chi(g)$$

### Projection Operators

The projection onto the irreducible component $\alpha$ is:

$$P_\alpha = \frac{d_\alpha}{|G|} \sum_{g \in G} \overline{\chi_\alpha(g)} \rho(g)$$

where $d_\alpha = \chi_\alpha(e)$ is the dimension of irrep $\alpha$.

---

## 6. Quantum Mechanics Connection

### Irreps and Degeneracy

**Fundamental Principle:** If Hamiltonian $H$ commutes with all representation matrices, $[H, D(g)] = 0$, then:

1. Each irreducible subspace is an eigenspace of $H$
2. States in the same irrep have the same energy
3. Dimension of irrep = degree of degeneracy

### Multiplets

In atomic physics, electrons in an atom form **multiplets** under rotational symmetry:

| Orbital | Irrep of SO(3) | Dimension | Degeneracy |
|---------|---------------|-----------|------------|
| s | $\ell = 0$ | 1 | 1-fold |
| p | $\ell = 1$ | 3 | 3-fold |
| d | $\ell = 2$ | 5 | 5-fold |
| f | $\ell = 3$ | 7 | 7-fold |

### Selection Rules

Matrix elements $\langle \psi' | \hat{A} | \psi \rangle$ vanish unless the representations satisfy certain conditions—this is where representation theory gives **selection rules**.

### Symmetry Breaking

When a perturbation breaks symmetry:
- Full symmetry group $G$ → subgroup $H$
- Irreps of $G$ decompose into irreps of $H$
- Degeneracies split according to this decomposition

---

## 7. Worked Examples

### Example 1: Decompose the 3D Permutation Rep of $S_3$

The permutation representation of $S_3$ acts on $\mathbb{C}^3$ by permuting coordinates.

**Finding invariant subspaces:**

1. The vector $(1, 1, 1)$ is fixed by all permutations.
   $W_1 = \text{span}\{(1,1,1)\}$ is 1-dim invariant, gives trivial rep.

2. The orthogonal complement: $W_2 = \{(x,y,z) : x + y + z = 0\}$ is also invariant.
   $W_2$ is 2-dimensional.

**Check if $W_2$ is irreducible:**

In $W_2$, use basis $\{(1,-1,0), (0,1,-1)\}$.

For transposition $(12)$: $(x,y,z) \mapsto (y,x,z)$
- $(1,-1,0) \mapsto (-1,1,0) = -(1,-1,0)$
- $(0,1,-1) \mapsto (1,0,-1) = (1,-1,0) + (0,1,-1)$

Matrix: $D((12)) = \begin{pmatrix} -1 & 1 \\ 0 & 1 \end{pmatrix}$

Similarly work out other elements. This 2-dim representation is irreducible!

**Decomposition:**
$$\text{perm}_{S_3} \cong \text{triv} \oplus \text{standard}$$

### Example 2: Decompose Regular Rep of $\mathbb{Z}_3$

The regular representation of $\mathbb{Z}_3 = \{e, r, r^2\}$ has dimension 3.

Basis: $\{e_e, e_r, e_{r^2}\}$ with action $r \cdot e_g = e_{rg}$.

Matrix for $r$:
$$D(r) = \begin{pmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$$

Eigenvalues: solutions to $\lambda^3 = 1$, i.e., $\lambda = 1, \omega, \omega^2$ where $\omega = e^{2\pi i/3}$.

Eigenvectors:
- $\lambda = 1$: $(1, 1, 1)$
- $\lambda = \omega$: $(1, \omega, \omega^2)$
- $\lambda = \omega^2$: $(1, \omega^2, \omega)$

**Decomposition:** $\text{reg}_{\mathbb{Z}_3} \cong \rho_0 \oplus \rho_1 \oplus \rho_2$

where $\rho_k$ is the 1-dim rep with $\rho_k(r) = \omega^k$.

### Example 3: Test Irreducibility of 2-dim Rep

Given representation of $\mathbb{Z}_4$ on $\mathbb{C}^2$:
$$D(r) = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

Is this irreducible?

Eigenvalues of $D(r)$: $\det(D(r) - \lambda I) = \lambda^2 + 1 = 0$, so $\lambda = \pm i$.

Eigenvectors: For $\lambda = i$: $(1, -i)^T$. For $\lambda = -i$: $(1, i)^T$.

These eigenspaces are 1-dimensional and invariant under $D(r)$.

In the eigenbasis: $D(r) \to \begin{pmatrix} i & 0 \\ 0 & -i \end{pmatrix}$

This decomposes as $\rho_1 \oplus \rho_3$ where $\rho_k(r) = i^k$. **Reducible!**

---

## 8. Computational Lab

```python
"""
Day 289: Reducible and Irreducible Representations
Testing reducibility and finding decompositions
"""

import numpy as np
from numpy.linalg import eig, matrix_rank, solve
from typing import List, Dict, Tuple, Optional
from itertools import combinations

class RepAnalyzer:
    """
    Analyze reducibility of representations.
    """

    def __init__(self, matrices: Dict, name: str = "ρ"):
        """
        Initialize with representation matrices.

        Parameters:
            matrices: Dict mapping group elements to numpy matrices
        """
        self.matrices = matrices
        self.elements = list(matrices.keys())
        self.dim = matrices[self.elements[0]].shape[0]
        self.name = name

    def is_invariant_subspace(self, basis_vectors: List[np.ndarray]) -> bool:
        """
        Check if the span of basis_vectors is invariant.
        """
        if len(basis_vectors) == 0:
            return True

        # Build matrix with columns = basis vectors
        V = np.column_stack(basis_vectors)
        rank_V = matrix_rank(V)

        for g in self.elements:
            D = self.matrices[g]
            # D @ V should have columns in span(V)
            DV = D @ V
            # Check if columns of DV are in column space of V
            augmented = np.column_stack([V, DV])
            if matrix_rank(augmented) > rank_V:
                return False

        return True

    def find_invariant_subspaces(self, dim: int = 1) -> List[np.ndarray]:
        """
        Find invariant subspaces of specified dimension.
        Returns list of basis vectors for each invariant subspace found.
        """
        invariant = []

        if dim == 1:
            # Look for common eigenvectors
            # Find intersection of eigenspaces
            common = self._find_common_eigenvectors()
            for v in common:
                if self.is_invariant_subspace([v]):
                    invariant.append(v)

        return invariant

    def _find_common_eigenvectors(self) -> List[np.ndarray]:
        """Find vectors that are eigenvectors for all matrices."""
        # Start with eigenvectors of first non-identity element
        common = []

        for g in self.elements:
            D = self.matrices[g]
            if not np.allclose(D, np.eye(self.dim)):
                eigenvalues, eigenvectors = eig(D)
                for i in range(self.dim):
                    v = eigenvectors[:, i]
                    # Check if this is eigenvector for all
                    is_common = True
                    for h in self.elements:
                        Dh = self.matrices[h]
                        Dv = Dh @ v
                        # Check if Dv is parallel to v
                        if np.linalg.norm(v) > 1e-10:
                            ratio = Dv / (v + 1e-10)
                            if not np.allclose(ratio, ratio[0]):
                                is_common = False
                                break
                    if is_common:
                        common.append(v)
                break

        return common

    def test_irreducibility(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Test if representation is irreducible.

        Returns:
            (is_irreducible, basis_change_matrix if reducible)
        """
        # Look for 1-dim invariant subspaces
        for g in self.elements:
            D = self.matrices[g]
            if np.allclose(D, np.eye(self.dim)):
                continue

            eigenvalues, eigenvectors = eig(D)

            for i in range(self.dim):
                v = eigenvectors[:, i]
                if self.is_invariant_subspace([v]):
                    # Found invariant subspace!
                    return False, v

        # More thorough check using Schur's lemma approach
        # (Will be refined with characters)
        return True, None

    def decompose(self) -> List['RepAnalyzer']:
        """
        Decompose into irreducible components.
        Returns list of irreducible representations.
        """
        components = []

        # Try to find invariant subspace
        is_irred, v = self.test_irreducibility()

        if is_irred:
            return [self]

        # Found reducible - decompose
        # This is a simplified version; full decomposition requires
        # finding complement and recursing

        print(f"Found invariant subspace spanned by {v}")
        return components  # Placeholder

    def character(self, g) -> complex:
        """Compute character."""
        return np.trace(self.matrices[g])

    def character_inner_product(self, other: 'RepAnalyzer') -> complex:
        """
        Compute <χ, χ'> = (1/|G|) Σ χ(g)* χ'(g)
        """
        result = 0
        for g in self.elements:
            result += np.conj(self.character(g)) * other.character(g)
        return result / len(self.elements)

    def multiplicity_of_irrep(self, irrep: 'RepAnalyzer') -> float:
        """
        Compute multiplicity of irrep in this representation.
        """
        return self.character_inner_product(irrep).real


def create_permutation_rep_S3() -> RepAnalyzer:
    """3-dim permutation representation of S_3."""
    from itertools import permutations

    elements = list(permutations(range(3)))
    matrices = {}

    for p in elements:
        D = np.zeros((3, 3))
        for i in range(3):
            D[p[i], i] = 1.0
        matrices[p] = D

    return RepAnalyzer(matrices, "perm")


def create_regular_rep_Z3() -> RepAnalyzer:
    """Regular representation of Z_3."""
    # r acts as cyclic shift
    D_e = np.eye(3)
    D_r = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    D_r2 = D_r @ D_r

    matrices = {0: D_e, 1: D_r, 2: D_r2}
    return RepAnalyzer(matrices, "reg")


def create_2d_rep_Z4() -> RepAnalyzer:
    """2-dim representation of Z_4."""
    D = {}
    theta = np.pi / 2
    D[0] = np.eye(2)
    D[1] = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    D[2] = D[1] @ D[1]
    D[3] = D[2] @ D[1]

    return RepAnalyzer(D, "2d")


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("REDUCIBLE AND IRREDUCIBLE REPRESENTATIONS")
    print("=" * 60)

    # Example 1: Permutation rep of S_3
    print("\n1. PERMUTATION REPRESENTATION OF S_3")
    print("-" * 40)

    perm = create_permutation_rep_S3()
    print(f"Dimension: {perm.dim}")

    # Test invariant subspaces
    v1 = np.array([1, 1, 1]) / np.sqrt(3)
    print(f"\nIs span{{(1,1,1)}} invariant? {perm.is_invariant_subspace([v1])}")

    v2 = np.array([1, -1, 0]) / np.sqrt(2)
    v3 = np.array([1, 1, -2]) / np.sqrt(6)
    print(f"Is span{{(1,-1,0), (1,1,-2)}} invariant? {perm.is_invariant_subspace([v2, v3])}")

    # Characters
    print("\nCharacters:")
    for g in perm.elements:
        print(f"  χ({g}) = {perm.character(g):.2f}")

    is_irred, v = perm.test_irreducibility()
    print(f"\nIrreducible? {is_irred}")
    if not is_irred:
        print(f"Found invariant vector: {v}")

    # Example 2: Regular rep of Z_3
    print("\n2. REGULAR REPRESENTATION OF Z_3")
    print("-" * 40)

    reg = create_regular_rep_Z3()
    print(f"Dimension: {reg.dim}")

    # Find eigenvalues of D(r)
    D_r = reg.matrices[1]
    eigenvalues, eigenvectors = eig(D_r)
    print(f"\nEigenvalues of D(r): {eigenvalues}")

    is_irred, v = reg.test_irreducibility()
    print(f"Irreducible? {is_irred}")

    # Check each eigenspace
    print("\nInvariant subspaces from eigenvectors:")
    for i in range(3):
        v = eigenvectors[:, i]
        inv = reg.is_invariant_subspace([v])
        print(f"  Eigenspace for λ={eigenvalues[i]:.4f}: invariant = {inv}")

    # Example 3: 2d rep of Z_4
    print("\n3. 2-DIM REPRESENTATION OF Z_4")
    print("-" * 40)

    rep = create_2d_rep_Z4()
    print(f"Dimension: {rep.dim}")

    print("\nMatrices:")
    for g, D in rep.matrices.items():
        print(f"  D({g}) =\n{D}")

    is_irred, v = rep.test_irreducibility()
    print(f"\nOver R: Irreducible? {is_irred}")

    # Over C, this decomposes
    D_r = rep.matrices[1]
    eigenvalues, eigenvectors = eig(D_r)
    print(f"\nOver C, eigenvalues of D(r): {eigenvalues}")
    print("This decomposes into two 1-dim reps over C")

    # Example 4: Verify Maschke's theorem
    print("\n4. MASCHKE'S THEOREM VERIFICATION")
    print("-" * 40)

    print("Finding invariant complement for permutation rep...")

    # Invariant subspace: W = span{(1,1,1)}
    # Need to find invariant complement

    # Project onto orthogonal complement
    v_fixed = np.array([1, 1, 1]) / np.sqrt(3)
    P_W = np.outer(v_fixed, v_fixed)  # Projection onto W

    # Average projection
    P_avg = np.zeros((3, 3))
    for g in perm.elements:
        D = perm.matrices[g]
        P_avg += D @ P_W @ np.linalg.inv(D)
    P_avg /= len(perm.elements)

    print(f"Averaged projection matrix:\n{P_avg}")
    print(f"Rank: {matrix_rank(P_avg)}")

    W_perp = np.eye(3) - P_avg
    print(f"\nComplement projector:\n{W_perp}")
    print(f"This projects onto the 2-dim standard rep subspace")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Irreducible reps are the "atoms" of representation theory
    2. Maschke: Every rep of finite group completely reduces
    3. Invariant subspaces ↔ block diagonal form
    4. Number of irreps = number of conjugacy classes
    5. Σ dᵢ² = |G| (dimension formula)
    6. In QM: irreps ↔ multiplets with same energy
    """)
```

---

## 9. Practice Problems

### Problem Set A: Invariant Subspaces

**A1.** For the 2-dim representation of $\mathbb{Z}_2$ with $D(a) = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, find all invariant subspaces.

**A2.** Show that the "zero-sum" subspace $\{x \in \mathbb{C}^n : \sum x_i = 0\}$ is invariant under the permutation representation of $S_n$.

**A3.** If $W$ is invariant under $\rho$, prove that $W^\perp$ is invariant under $\rho^*$ (dual representation).

### Problem Set B: Decomposition

**B1.** Decompose the regular representation of $\mathbb{Z}_4$ into irreducible components.

**B2.** Show that the permutation representation of $S_4$ on $\mathbb{C}^4$ decomposes as $\text{triv} \oplus \text{standard}_{S_4}$.

**B3.** Find the irreducible decomposition of $\rho_1 \otimes \rho_2$ for 1-dim reps of $\mathbb{Z}_n$.

### Problem Set C: Quantum Applications

**C1.** A quantum system has $D_3$ symmetry. How many distinct energy levels can there be, at most?

**C2.** **(Spin)** The tensor product of two spin-1/2 representations decomposes as $\frac{1}{2} \otimes \frac{1}{2} = 0 \oplus 1$. Verify the dimensions match.

**C3.** If $H$ commutes with a 3-dim irreducible representation, what can you say about the spectrum of $H$ restricted to that subspace?

---

## 10. Summary

### Key Concepts

| Concept | Definition | Significance |
|---------|------------|--------------|
| Invariant subspace | $\rho(g)W \subseteq W$ | Building block for reduction |
| Irreducible | No proper invariant subspaces | Fundamental unit |
| Completely reducible | Direct sum of irreps | What Maschke guarantees |
| Multiplicity | How many times irrep appears | Degeneracy count |

### Key Theorems

$$\boxed{\text{Maschke: Every rep of finite } G \text{ over } \mathbb{C} \text{ is completely reducible}}$$

$$\boxed{\#\text{irreps} = \#\text{conjugacy classes}}$$

$$\boxed{\sum_i d_i^2 = |G|}$$

---

## 11. Preview: Day 290

Tomorrow we study **Schur's Lemma**, one of the most powerful results in representation theory:
- Statement of Schur's lemma
- Consequences for intertwining operators
- Application to selection rules
- Orthogonality of matrix elements

---

*"The problem of decomposing representations is central to both pure mathematics and physics." — Israel Gelfand*
