# Day 288: Group Representations — Groups Acting on Vector Spaces

## Overview

**Month 11, Week 42, Day 1 — Monday**

Today we begin **representation theory**, the study of how groups act on vector spaces through linear transformations. This is where abstract algebra meets linear algebra and becomes directly applicable to physics. In quantum mechanics, states live in vector spaces (Hilbert spaces), and symmetry transformations act as linear operators. Representation theory tells us how to systematically study these actions.

## Prerequisites

From Week 41:
- Group theory fundamentals
- Homomorphisms and isomorphisms
- Matrix groups ($GL_n$, $SL_n$, etc.)

From linear algebra:
- Vector spaces, bases, dimension
- Linear transformations and matrices
- Change of basis

## Learning Objectives

By the end of today, you will be able to:

1. Define group representations on vector spaces
2. Construct matrix representations in a given basis
3. Understand equivalent representations
4. Identify the trivial, regular, and permutation representations
5. Connect representations to quantum state transformations
6. Implement representations computationally

---

## 1. Definition of Representation

### The Fundamental Definition

**Definition:** A **representation** of a group $G$ on a vector space $V$ (over field $\mathbb{F}$, usually $\mathbb{C}$) is a homomorphism:

$$\boxed{\rho: G \to GL(V)}$$

where $GL(V)$ is the group of invertible linear transformations on $V$.

Explicitly, this means:
1. Each $g \in G$ is assigned a linear operator $\rho(g): V \to V$
2. $\rho(gh) = \rho(g) \circ \rho(h)$ for all $g, h \in G$
3. $\rho(e) = \text{id}_V$ (identity map)
4. $\rho(g^{-1}) = \rho(g)^{-1}$

### Terminology

- The **dimension** of the representation is $\dim(V)$
- A representation is **faithful** if $\rho$ is injective (one-to-one)
- $V$ is called the **representation space** or **carrier space**

### Matrix Representations

Choosing a basis $\{e_1, \ldots, e_n\}$ for $V$ gives a **matrix representation**:

$$D: G \to GL_n(\mathbb{F}), \quad g \mapsto D(g)$$

where $D(g)$ is the matrix of $\rho(g)$ in the chosen basis:
$$\rho(g)(e_j) = \sum_{i=1}^n D(g)_{ij} e_i$$

The representation condition becomes:
$$D(gh) = D(g) D(h)$$

---

## 2. Basic Examples

### Example 1: The Trivial Representation

For any group $G$, the **trivial representation** is:
$$\rho_{\text{triv}}: G \to GL_1(\mathbb{C}), \quad \rho_{\text{triv}}(g) = 1 \quad \forall g$$

Every group element acts as the identity.

Dimension: 1

### Example 2: The Sign Representation of $S_n$

$$\text{sgn}: S_n \to GL_1(\mathbb{C}), \quad \text{sgn}(\sigma) = \begin{cases} +1 & \sigma \text{ even} \\ -1 & \sigma \text{ odd} \end{cases}$$

This is a 1-dimensional representation.

### Example 3: The Defining Representation

Many groups have a "natural" representation:

**$S_n$ permutation representation:** Act on $\mathbb{C}^n$ by permuting coordinates:
$$\rho(\sigma)(e_i) = e_{\sigma(i)}$$

Matrix form: $D(\sigma)_{ij} = \delta_{i, \sigma(j)}$ (permutation matrices)

**$O(n)$ defining representation:** Act on $\mathbb{R}^n$ by rotation/reflection.

**$SU(2)$ defining representation:** Act on $\mathbb{C}^2$ by matrix multiplication.

### Example 4: The Regular Representation

For finite group $G$, the **regular representation** acts on the vector space with basis $\{e_g : g \in G\}$:

$$\rho_{\text{reg}}(h)(e_g) = e_{hg}$$

Dimension: $|G|$

This representation contains every irreducible representation!

### Example 5: Cyclic Group $\mathbb{Z}_n$

Representations of $\mathbb{Z}_n = \langle r | r^n = e \rangle$:

Since $r^n = e$, we need $\rho(r)^n = I$.

For 1-dimensional representations: $\rho(r) = \omega$ where $\omega^n = 1$.

The $n$ distinct 1-dimensional representations are:
$$\rho_k(r) = e^{2\pi i k/n}, \quad k = 0, 1, \ldots, n-1$$

---

## 3. Equivalent Representations

### Definition

Two representations $\rho: G \to GL(V)$ and $\rho': G \to GL(V')$ are **equivalent** (or isomorphic) if there exists an invertible linear map $T: V \to V'$ such that:

$$\boxed{T \rho(g) = \rho'(g) T \quad \forall g \in G}$$

or equivalently: $\rho'(g) = T \rho(g) T^{-1}$

### In Matrix Terms

Matrix representations $D$ and $D'$ are equivalent if there exists an invertible matrix $S$ such that:
$$D'(g) = S D(g) S^{-1} \quad \forall g \in G$$

This is just a change of basis!

### Why Equivalence Matters

Equivalent representations describe the "same" action of $G$—they differ only in the choice of basis. Representation theory classifies representations up to equivalence.

---

## 4. Operations on Representations

### Direct Sum

Given representations $\rho_1: G \to GL(V_1)$ and $\rho_2: G \to GL(V_2)$:

$$(\rho_1 \oplus \rho_2)(g) = \rho_1(g) \oplus \rho_2(g)$$

acting on $V_1 \oplus V_2$.

In matrices: $D_1 \oplus D_2 = \begin{pmatrix} D_1 & 0 \\ 0 & D_2 \end{pmatrix}$

### Tensor Product

$$(\rho_1 \otimes \rho_2)(g) = \rho_1(g) \otimes \rho_2(g)$$

acting on $V_1 \otimes V_2$.

Dimension: $\dim(V_1) \cdot \dim(V_2)$

### Dual Representation

The **dual** (or contragredient) representation acts on $V^*$:
$$\rho^*(g) = (\rho(g^{-1}))^T$$

### Complex Conjugate

For complex representations, the **conjugate representation**:
$$\bar{\rho}(g) = \overline{\rho(g)}$$

---

## 5. Quantum Mechanics Connection

### States as Representation Spaces

In quantum mechanics:
- States $|\psi\rangle$ live in a Hilbert space $\mathcal{H}$
- Symmetry group $G$ acts via unitary operators: $|\psi\rangle \mapsto U(g)|\psi\rangle$
- This defines a **unitary representation**: $U: G \to U(\mathcal{H})$

### Why Representations Matter in QM

1. **Classifying particles:** Particle types correspond to irreducible representations
2. **Degeneracy:** States in the same irrep have the same energy under symmetric Hamiltonians
3. **Selection rules:** Matrix elements $\langle \psi' | A | \psi \rangle$ constrained by representation theory
4. **Conservation laws:** Symmetry representations determine quantum numbers

### The Fundamental Postulate

**Wigner's Theorem:** Symmetry transformations in quantum mechanics are represented by unitary (or antiunitary) operators.

### Example: Rotations

The rotation group SO(3) acts on quantum states. Different particles transform under different representations:

| Spin | Representation Dimension | Example |
|------|-------------------------|---------|
| 0 | 1 | Pion, Higgs |
| 1/2 | 2 | Electron, quark |
| 1 | 3 | Photon, W/Z bosons |
| 3/2 | 4 | Delta baryon |
| 2 | 5 | Graviton (hypothetical) |

---

## 6. Worked Examples

### Example 1: Representations of $\mathbb{Z}_2$

$\mathbb{Z}_2 = \{e, a\}$ with $a^2 = e$.

**1-dimensional representations:**
Need $D(a)^2 = 1$, so $D(a) = \pm 1$.

- **Trivial:** $D(e) = 1, D(a) = 1$
- **Sign:** $D(e) = 1, D(a) = -1$

**2-dimensional representation:**
$$D(e) = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad D(a) = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

Is this irreducible? The eigenspaces of $D(a)$ are $\mathbb{C}(1,1)^T$ and $\mathbb{C}(1,-1)^T$.

In the eigenbasis: $D(a) \to \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$

This is $\rho_{\text{triv}} \oplus \rho_{\text{sign}}$ — reducible!

### Example 2: Representations of $S_3$

$S_3$ has 6 elements. How many irreps?

**Character theory fact:** Number of irreps = number of conjugacy classes.

Conjugacy classes of $S_3$: $\{e\}$, $\{(12), (13), (23)\}$, $\{(123), (132)\}$

So there are **3 irreps**.

**Dimension constraint:** $\sum_i d_i^2 = |G| = 6$

Only solution: $1^2 + 1^2 + 2^2 = 6$.

The irreps are:
1. **Trivial** (dim 1)
2. **Sign** (dim 1): $D(\sigma) = \text{sgn}(\sigma)$
3. **Standard** (dim 2): Acts on $\{(x,y,z) : x+y+z=0\} \subset \mathbb{C}^3$

### Example 3: Matrix Representation of $D_3$

$D_3$ = symmetries of equilateral triangle. Same group as $S_3$!

2-dimensional representation (vertices at $e^{2\pi i k/3}$):

$$D(r) = \begin{pmatrix} \cos(2\pi/3) & -\sin(2\pi/3) \\ \sin(2\pi/3) & \cos(2\pi/3) \end{pmatrix} = \begin{pmatrix} -1/2 & -\sqrt{3}/2 \\ \sqrt{3}/2 & -1/2 \end{pmatrix}$$

$$D(s) = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

This is the "standard" 2-dimensional irrep of $S_3 \cong D_3$.

---

## 7. Computational Lab

```python
"""
Day 288: Group Representations
Constructing and analyzing representations
"""

import numpy as np
from typing import List, Dict, Callable, Tuple
from numpy.linalg import matrix_power, inv, eig

class Representation:
    """
    A matrix representation of a finite group.
    """

    def __init__(self, group_elements: List, matrices: Dict, name: str = "ρ"):
        """
        Initialize representation.

        Parameters:
            group_elements: List of group elements
            matrices: Dict mapping elements to numpy matrices
            name: Name of representation
        """
        self.elements = group_elements
        self.matrices = matrices
        self.name = name
        self.dim = matrices[group_elements[0]].shape[0]

    def __call__(self, g) -> np.ndarray:
        """Return matrix for element g."""
        return self.matrices[g]

    def verify_homomorphism(self, operation: Callable) -> bool:
        """
        Verify D(gh) = D(g)D(h) for all g, h.

        Parameters:
            operation: Group operation function
        """
        for g in self.elements:
            for h in self.elements:
                gh = operation(g, h)
                lhs = self.matrices[gh]
                rhs = self.matrices[g] @ self.matrices[h]
                if not np.allclose(lhs, rhs):
                    return False
        return True

    def character(self, g) -> complex:
        """Compute character χ(g) = Tr(D(g))."""
        return np.trace(self.matrices[g])

    def character_table_row(self) -> Dict:
        """Return characters for all elements."""
        return {g: self.character(g) for g in self.elements}

    def is_unitary(self) -> bool:
        """Check if representation is unitary (D(g)† D(g) = I)."""
        for g in self.elements:
            D = self.matrices[g]
            if not np.allclose(D.conj().T @ D, np.eye(self.dim)):
                return False
        return True

    def direct_sum(self, other: 'Representation') -> 'Representation':
        """Compute direct sum representation."""
        from scipy.linalg import block_diag

        new_matrices = {}
        for g in self.elements:
            new_matrices[g] = block_diag(self.matrices[g], other.matrices[g])

        return Representation(
            self.elements, new_matrices,
            f"({self.name} ⊕ {other.name})"
        )

    def tensor_product(self, other: 'Representation') -> 'Representation':
        """Compute tensor product representation."""
        new_matrices = {}
        for g in self.elements:
            new_matrices[g] = np.kron(self.matrices[g], other.matrices[g])

        return Representation(
            self.elements, new_matrices,
            f"({self.name} ⊗ {other.name})"
        )


def trivial_rep(elements: List) -> Representation:
    """Construct trivial representation."""
    matrices = {g: np.array([[1.0]]) for g in elements}
    return Representation(elements, matrices, "1")


def sign_rep_Sn(n: int) -> Representation:
    """Construct sign representation of S_n."""
    from itertools import permutations

    def sign(p):
        inv = sum(1 for i in range(n) for j in range(i+1, n) if p[i] > p[j])
        return (-1) ** inv

    elements = list(permutations(range(n)))
    matrices = {p: np.array([[float(sign(p))]]) for p in elements}
    return Representation(elements, matrices, "sgn")


def permutation_rep_Sn(n: int) -> Representation:
    """Construct n-dimensional permutation representation of S_n."""
    from itertools import permutations

    elements = list(permutations(range(n)))
    matrices = {}

    for p in elements:
        D = np.zeros((n, n))
        for i in range(n):
            D[p[i], i] = 1.0
        matrices[p] = D

    return Representation(elements, matrices, "perm")


def regular_rep(elements: List, operation: Callable) -> Representation:
    """Construct regular representation."""
    n = len(elements)
    elem_to_idx = {g: i for i, g in enumerate(elements)}

    matrices = {}
    for h in elements:
        D = np.zeros((n, n))
        for i, g in enumerate(elements):
            hg = operation(h, g)
            j = elem_to_idx[hg]
            D[j, i] = 1.0
        matrices[h] = D

    return Representation(elements, matrices, "reg")


def cyclic_reps(n: int) -> List[Representation]:
    """Construct all 1-dim representations of Z_n."""
    elements = list(range(n))
    reps = []

    for k in range(n):
        omega = np.exp(2j * np.pi * k / n)
        matrices = {m: np.array([[omega ** m]]) for m in elements}
        reps.append(Representation(elements, matrices, f"χ_{k}"))

    return reps


def D3_reps() -> Dict[str, Representation]:
    """Construct all irreps of D_3 ≅ S_3."""
    # Elements: e, r, r^2, s, sr, sr^2 as (rotation, reflection)
    elements = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]

    # Group operation
    def op(g1, g2):
        r1, s1 = g1
        r2, s2 = g2
        if s1 == 0:
            return ((r1 + r2) % 3, s2)
        else:
            return ((r1 - r2) % 3, (s1 + s2) % 2)

    # Trivial representation
    triv = {g: np.array([[1.0]]) for g in elements}

    # Sign representation
    sign_matrices = {g: np.array([[(-1.0) ** g[1]]]) for g in elements}

    # 2-dimensional standard representation
    omega = np.exp(2j * np.pi / 3)
    r_mat = np.array([[omega, 0], [0, omega.conj()]])
    s_mat = np.array([[0, 1], [1, 0]])

    def compute_2d(g):
        r, s = g
        result = np.linalg.matrix_power(r_mat, r)
        if s == 1:
            result = result @ s_mat
        return result

    std_matrices = {g: compute_2d(g) for g in elements}

    return {
        'trivial': Representation(elements, triv, "1"),
        'sign': Representation(elements, sign_matrices, "sgn"),
        'standard': Representation(elements, std_matrices, "std")
    }


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("GROUP REPRESENTATIONS")
    print("=" * 60)

    # Example 1: Representations of Z_4
    print("\n1. ALL 1-DIM REPRESENTATIONS OF Z_4")
    print("-" * 40)

    reps = cyclic_reps(4)
    operation = lambda a, b: (a + b) % 4

    for rep in reps:
        print(f"\nRepresentation {rep.name}:")
        for g in rep.elements:
            val = rep(g)[0, 0]
            print(f"  {rep.name}({g}) = {val:.4f}")

        valid = rep.verify_homomorphism(operation)
        print(f"  Valid homomorphism: {valid}")

    # Example 2: S_3 representations
    print("\n2. REPRESENTATIONS OF S_3")
    print("-" * 40)

    perm_rep = permutation_rep_Sn(3)
    sign_rep = sign_rep_Sn(3)

    print(f"Permutation representation dimension: {perm_rep.dim}")
    print(f"Sign representation dimension: {sign_rep.dim}")

    # Show some matrices
    sigma = (1, 2, 0)  # The 3-cycle (0 1 2)
    print(f"\nFor σ = (0 1 2):")
    print(f"Permutation matrix:\n{perm_rep(sigma)}")
    print(f"Sign: {sign_rep(sigma)[0,0]}")

    # Example 3: D_3 irreps
    print("\n3. IRREDUCIBLE REPRESENTATIONS OF D_3")
    print("-" * 40)

    d3_reps = D3_reps()

    for name, rep in d3_reps.items():
        print(f"\n{name.upper()} (dim = {rep.dim}):")
        for g in rep.elements[:3]:  # Show first few
            D = rep(g)
            if rep.dim == 1:
                print(f"  D{g} = {D[0,0]:.4f}")
            else:
                print(f"  D{g} =\n{D}")

    # Example 4: Characters
    print("\n4. CHARACTER COMPUTATION")
    print("-" * 40)

    print("\nCharacters of D_3 representations:")
    print(f"{'Element':<15} {'Trivial':>10} {'Sign':>10} {'Standard':>10}")
    print("-" * 45)

    for g in d3_reps['trivial'].elements:
        chi_triv = d3_reps['trivial'].character(g)
        chi_sign = d3_reps['sign'].character(g)
        chi_std = d3_reps['standard'].character(g)
        print(f"{str(g):<15} {chi_triv.real:>10.2f} {chi_sign.real:>10.2f} {chi_std.real:>10.2f}")

    # Example 5: Direct sum and tensor product
    print("\n5. COMBINING REPRESENTATIONS")
    print("-" * 40)

    triv = d3_reps['trivial']
    sign = d3_reps['sign']
    std = d3_reps['standard']

    direct = triv.direct_sum(sign)
    tensor = triv.tensor_product(std)

    print(f"Trivial ⊕ Sign: dim = {direct.dim}")
    print(f"Trivial ⊗ Standard: dim = {tensor.dim}")

    # Example 6: Verify unitarity
    print("\n6. UNITARITY CHECK")
    print("-" * 40)

    for name, rep in d3_reps.items():
        unitary = rep.is_unitary()
        print(f"{name}: {'Unitary ✓' if unitary else 'Not unitary'}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Representation = homomorphism G → GL(V)
    2. Matrix rep = choosing a basis for V
    3. Equivalent reps = related by change of basis
    4. Character χ(g) = Tr(D(g)) is basis-independent
    5. In QM: states form representation spaces
    6. Symmetry operations act via representation matrices
    """)
```

---

## 8. Practice Problems

### Problem Set A: Basic Representations

**A1.** Find all 1-dimensional representations of $\mathbb{Z}_6$.

**A2.** Write down the permutation representation matrices for all elements of $S_3$.

**A3.** Verify that the matrices $D(r) = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$, $D(r^2) = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}$ define a representation of $\mathbb{Z}_4$.

### Problem Set B: Equivalence and Operations

**B1.** Show that the representation $D(a) = \begin{pmatrix} -1 & 1 \\ 0 & -1 \end{pmatrix}$ for $\mathbb{Z}_2 = \{e, a\}$ is equivalent to $D'(a) = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}$.

**B2.** Compute the tensor product of two 1-dimensional representations of $\mathbb{Z}_3$.

**B3.** For the 2-dim representation of $D_3$, verify $D(r)^3 = I$ and $D(s)^2 = I$.

### Problem Set C: Quantum Applications

**C1.** A quantum system has $\mathbb{Z}_2$ symmetry with representation $D(a) = \sigma_z$. What are the symmetry eigenstates?

**C2.** **(Spin)** The spin-1/2 representation of rotations is given by $D(\hat{n}, \theta) = e^{-i\theta \hat{n} \cdot \vec{\sigma}/2}$. Verify this is a representation.

**C3.** If a Hamiltonian commutes with representation matrices, $[H, D(g)] = 0$, what can you conclude about energy eigenstates?

---

## 9. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| Representation | $\rho: G \to GL(V)$, a homomorphism |
| Dimension | $\dim(V)$ |
| Faithful | $\rho$ is injective |
| Character | $\chi(g) = \text{Tr}(D(g))$ |
| Equivalent | Related by $S D(g) S^{-1}$ |

### Key Examples

| Group | Important Reps |
|-------|---------------|
| $\mathbb{Z}_n$ | $n$ 1-dim reps: $\rho_k(r) = e^{2\pi ik/n}$ |
| $S_n$ | Trivial, sign, standard, permutation |
| $D_n$ | Trivial, sign, 2-dim rotations |

---

## 10. Preview: Day 289

Tomorrow we study **reducible and irreducible representations**:
- Invariant subspaces
- Definition of irreducibility
- Complete reducibility (Maschke's theorem)
- Finding irreducible components

---

*"Representation theory is a way of taking complex objects and realizing them as matrices. It is the translation of group theory into linear algebra." — Hermann Weyl*
