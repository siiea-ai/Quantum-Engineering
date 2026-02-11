# Day 290: Schur's Lemma — The Fundamental Tool

## Overview

**Month 11, Week 42, Day 3 — Wednesday**

Today we prove and apply **Schur's Lemma**, arguably the single most important result in representation theory. This lemma characterizes maps between irreducible representations and has profound consequences: it leads to orthogonality relations, explains selection rules in quantum mechanics, and provides the foundation for character theory.

## Prerequisites

From Days 288-289:
- Group representations and matrix representations
- Irreducible representations
- Invariant subspaces

## Learning Objectives

By the end of today, you will be able to:

1. State and prove both parts of Schur's lemma
2. Apply Schur's lemma to intertwining operators
3. Derive orthogonality of matrix elements
4. Understand the connection to selection rules
5. Use Schur's lemma to analyze operators commuting with representations
6. Apply the lemma to quantum mechanical problems

---

## 1. Intertwining Operators

### Definition

**Definition:** Let $\rho_1: G \to GL(V_1)$ and $\rho_2: G \to GL(V_2)$ be two representations. A linear map $T: V_1 \to V_2$ is an **intertwining operator** (or **$G$-morphism**) if:

$$\boxed{T \rho_1(g) = \rho_2(g) T \quad \forall g \in G}$$

Equivalently: $T$ "commutes" with the group action.

### Notation

The space of all intertwining operators from $\rho_1$ to $\rho_2$ is denoted:
$$\text{Hom}_G(\rho_1, \rho_2) = \{T: V_1 \to V_2 : T\rho_1(g) = \rho_2(g)T \text{ for all } g\}$$

### Examples

1. If $\rho_1 = \rho_2$, then $\rho(g)$ itself is an intertwining operator
2. Any scalar multiple of identity: $T = cI$
3. Projections onto irreducible subspaces (when properly defined)

---

## 2. Schur's Lemma

### Statement

**Theorem (Schur's Lemma):**

**Part 1:** Let $\rho_1$ and $\rho_2$ be irreducible representations. If $T: V_1 \to V_2$ is an intertwining operator, then either:
- $T = 0$, or
- $T$ is an isomorphism

**Part 2 (over $\mathbb{C}$):** If $\rho$ is an irreducible representation and $T: V \to V$ is an intertwining operator (i.e., $T\rho(g) = \rho(g)T$ for all $g$), then:
$$T = \lambda I$$
for some $\lambda \in \mathbb{C}$.

### Proof

**Proof of Part 1:**

Consider $\ker(T) \subseteq V_1$ and $\text{Im}(T) \subseteq V_2$.

**Claim:** $\ker(T)$ is an invariant subspace of $V_1$.

*Proof:* If $v \in \ker(T)$, then $T(\rho_1(g)v) = \rho_2(g)T(v) = \rho_2(g) \cdot 0 = 0$, so $\rho_1(g)v \in \ker(T)$. ∎

**Claim:** $\text{Im}(T)$ is an invariant subspace of $V_2$.

*Proof:* If $w = T(v) \in \text{Im}(T)$, then $\rho_2(g)w = \rho_2(g)T(v) = T(\rho_1(g)v) \in \text{Im}(T)$. ∎

Since $\rho_1$ is irreducible, $\ker(T) = \{0\}$ or $\ker(T) = V_1$.

Since $\rho_2$ is irreducible, $\text{Im}(T) = \{0\}$ or $\text{Im}(T) = V_2$.

If $\ker(T) = V_1$, then $T = 0$. ✓

If $\ker(T) = \{0\}$ (injective) and $\text{Im}(T) = V_2$ (surjective), then $T$ is an isomorphism. ✓

The only remaining case is $\ker(T) = \{0\}$ and $\text{Im}(T) = \{0\}$, but injectivity implies $T$ is zero or non-trivial image. ∎

**Proof of Part 2 (over $\mathbb{C}$):**

Since we're over $\mathbb{C}$, $T$ has an eigenvalue $\lambda$.

Consider $T' = T - \lambda I$.

$T'$ is still an intertwining operator: $T'\rho(g) = T\rho(g) - \lambda\rho(g) = \rho(g)T - \lambda\rho(g) = \rho(g)T'$.

But $\ker(T') \neq \{0\}$ (contains $\lambda$-eigenvectors).

Since $\rho$ is irreducible and $\ker(T')$ is invariant, $\ker(T') = V$.

Therefore $T' = 0$, so $T = \lambda I$. ∎

---

## 3. Consequences of Schur's Lemma

### Corollary 1: Dimension of Hom Space

For irreducible representations $\rho_1, \rho_2$ over $\mathbb{C}$:

$$\dim \text{Hom}_G(\rho_1, \rho_2) = \begin{cases} 1 & \text{if } \rho_1 \cong \rho_2 \\ 0 & \text{if } \rho_1 \not\cong \rho_2 \end{cases}$$

### Corollary 2: Commutant

If $\rho$ is irreducible, the only operators commuting with all $\rho(g)$ are scalar multiples of identity.

$$\{T : [T, \rho(g)] = 0 \text{ for all } g\} = \mathbb{C} \cdot I$$

### Corollary 3: Center of Group Algebra

The center of the group algebra $\mathbb{C}[G]$ acts by scalars on any irreducible representation.

### Corollary 4: Abelian Groups

For abelian groups, every irreducible representation is 1-dimensional.

*Proof:* For abelian $G$, every $\rho(g)$ commutes with every $\rho(h)$.
By Schur, $\rho(g) = \lambda_g I$ for each $g$.
Any 1-dim subspace is invariant, so if $\dim V > 1$, we'd have proper invariant subspaces.
Hence $\dim V = 1$. ∎

---

## 4. Orthogonality of Matrix Elements

### The Great Orthogonality Theorem (Preview)

**Theorem:** For irreducible representations $D^{(\alpha)}$ and $D^{(\beta)}$ of a finite group $G$:

$$\boxed{\sum_{g \in G} D^{(\alpha)}_{ij}(g)^* D^{(\beta)}_{kl}(g) = \frac{|G|}{d_\alpha} \delta_{\alpha\beta} \delta_{ik} \delta_{jl}}$$

where $d_\alpha = \dim(\rho_\alpha)$.

### Proof Sketch Using Schur

Define the operator:
$$A = \sum_{g \in G} D^{(\beta)}(g) E D^{(\alpha)}(g)^{-1}$$

where $E$ is any linear map from $V_\alpha$ to $V_\beta$.

**Claim:** $A$ is an intertwining operator from $\rho_\alpha$ to $\rho_\beta$.

By Schur's lemma:
- If $\alpha \neq \beta$: $A = 0$
- If $\alpha = \beta$: $A = \lambda I$

Taking traces and specific matrix elements yields the orthogonality relations.

---

## 5. Quantum Mechanics Connection

### Selection Rules

**The Wigner-Eckart Theorem** (preview): Matrix elements of operators between states are constrained by representation theory.

If $|\alpha, i\rangle$ transforms under irrep $\rho_\alpha$ and $|\beta, j\rangle$ under $\rho_\beta$, and operator $\hat{O}$ transforms under $\rho_\gamma$, then:

$$\langle \alpha, i | \hat{O} | \beta, j \rangle = 0$$

unless $\rho_\gamma \otimes \rho_\beta$ contains $\rho_\alpha$.

This is the mathematical origin of **selection rules** in spectroscopy!

### Example: Angular Momentum

For SO(3) rotations:
- States labeled by angular momentum $|l, m\rangle$
- Operators transform under specific representations
- Electric dipole operator ($\ell = 1$) gives selection rule $\Delta l = \pm 1$

### Degeneracy and Commuting Operators

**Theorem:** If $H$ commutes with an irreducible representation, $H$ acts as a scalar on the representation space.

$$[H, \rho(g)] = 0 \text{ for all } g \Rightarrow H|_V = E \cdot I$$

This explains why states in the same multiplet have the same energy!

---

## 6. Worked Examples

### Example 1: Schur's Lemma for $\mathbb{Z}_n$

All irreps of $\mathbb{Z}_n$ are 1-dimensional (abelian group!).

$\rho_k(r) = e^{2\pi ik/n}$ for $k = 0, 1, \ldots, n-1$.

Intertwining operator $T: \mathbb{C} \to \mathbb{C}$ satisfies:
$$T \cdot e^{2\pi ik/n} = e^{2\pi il/n} \cdot T$$

For $k \neq l$: This requires $T = 0$.
For $k = l$: Any $T = \lambda$ works.

Consistent with Schur!

### Example 2: Commuting Operators in $S_3$

Consider the 2-dim standard representation of $S_3$.

What operators $T$ commute with all $D(g)$?

By Schur: $T = \lambda I$ (since the rep is irreducible).

Verify: The matrices don't have any common eigenvector, so no non-trivial $T$ can commute with all of them.

### Example 3: Reducible Representation

Consider $\rho = \rho_1 \oplus \rho_2$ where $\rho_1, \rho_2$ are inequivalent irreps.

Intertwining operators $T: V_1 \oplus V_2 \to V_1 \oplus V_2$:

$$T = \begin{pmatrix} T_{11} & T_{12} \\ T_{21} & T_{22} \end{pmatrix}$$

By Schur:
- $T_{11} = \lambda_1 I_1$ (intertwiner $\rho_1 \to \rho_1$)
- $T_{22} = \lambda_2 I_2$ (intertwiner $\rho_2 \to \rho_2$)
- $T_{12} = 0$ (intertwiner $\rho_2 \to \rho_1$, inequivalent)
- $T_{21} = 0$ (intertwiner $\rho_1 \to \rho_2$, inequivalent)

So $T = \begin{pmatrix} \lambda_1 I_1 & 0 \\ 0 & \lambda_2 I_2 \end{pmatrix}$.

The commutant of a completely reducible representation is block-diagonal!

---

## 7. Computational Lab

```python
"""
Day 290: Schur's Lemma
Verifying Schur's lemma and computing intertwining operators
"""

import numpy as np
from numpy.linalg import eig, solve, matrix_rank, inv
from typing import List, Dict, Tuple, Optional

class SchurAnalyzer:
    """
    Tools for analyzing intertwining operators using Schur's lemma.
    """

    def __init__(self, rep1_matrices: Dict, rep2_matrices: Dict = None):
        """
        Initialize with one or two representations.

        Parameters:
            rep1_matrices: Dict of matrices for first representation
            rep2_matrices: Dict of matrices for second (defaults to rep1)
        """
        self.D1 = rep1_matrices
        self.D2 = rep2_matrices if rep2_matrices else rep1_matrices
        self.elements = list(self.D1.keys())

        self.dim1 = self.D1[self.elements[0]].shape[0]
        self.dim2 = self.D2[self.elements[0]].shape[0]

    def find_intertwiners(self, verbose: bool = False) -> np.ndarray:
        """
        Find basis for space of intertwining operators.

        Solves: T @ D1(g) = D2(g) @ T for all g

        Returns:
            Matrix whose columns form a basis for Hom_G(ρ1, ρ2)
        """
        # Set up linear system
        # T is a dim2 × dim1 matrix, viewed as dim1*dim2 vector

        # Equation: T @ D1(g) - D2(g) @ T = 0
        # In vec form: (D1(g)^T ⊗ I - I ⊗ D2(g)) vec(T) = 0

        equations = []
        for g in self.elements:
            D1g = self.D1[g]
            D2g = self.D2[g]

            # (D1(g)^T ⊗ I_{dim2}) - (I_{dim1} ⊗ D2(g))
            eq = np.kron(D1g.T, np.eye(self.dim2)) - np.kron(np.eye(self.dim1), D2g)
            equations.append(eq)

        A = np.vstack(equations)

        # Find null space
        # Use SVD
        U, S, Vh = np.linalg.svd(A)

        # Null space = rows of Vh corresponding to zero singular values
        tol = 1e-10
        null_mask = np.abs(S) < tol
        rank = np.sum(~null_mask)

        # Null vectors are last rows of Vh (after rank)
        null_dim = A.shape[1] - rank
        null_vectors = Vh[-null_dim:, :].T if null_dim > 0 else np.zeros((A.shape[1], 0))

        if verbose:
            print(f"Dimension of Hom_G(ρ1, ρ2): {null_dim}")

        return null_vectors

    def verify_schur_part1(self) -> Dict:
        """
        Verify Schur's lemma Part 1:
        For irreducible reps, intertwiners are either 0 or isomorphisms.
        """
        basis = self.find_intertwiners()
        dim_hom = basis.shape[1]

        results = {
            'hom_dimension': dim_hom,
            'conclusion': ''
        }

        if dim_hom == 0:
            results['conclusion'] = 'Hom = {0}: reps are inequivalent or one is zero'
        elif dim_hom == 1 and self.dim1 == self.dim2:
            # Check if the intertwiner is an isomorphism
            T = basis[:, 0].reshape(self.dim2, self.dim1)
            det_T = np.linalg.det(T)
            if np.abs(det_T) > 1e-10:
                results['conclusion'] = 'Hom = C⋅T where T is isomorphism: reps are equivalent'
                results['intertwiner'] = T
            else:
                results['conclusion'] = 'Unexpected: dim Hom = 1 but intertwiner not invertible'
        else:
            results['conclusion'] = f'dim Hom = {dim_hom}: rep is reducible or not irreducible'

        return results

    def verify_schur_part2(self) -> Dict:
        """
        Verify Schur's lemma Part 2:
        For irreducible self-intertwiner, T = λI.
        """
        if self.dim1 != self.dim2:
            return {'error': 'Dimensions must match for self-intertwiner'}

        basis = self.find_intertwiners()
        dim_hom = basis.shape[1]

        results = {
            'hom_dimension': dim_hom,
            'is_scalar_multiple': False
        }

        if dim_hom == 1:
            T = basis[:, 0].reshape(self.dim1, self.dim1)
            # Check if T is scalar multiple of identity
            if self.dim1 > 0:
                lambda_guess = T[0, 0]
                is_scalar = np.allclose(T, lambda_guess * np.eye(self.dim1))
                results['is_scalar_multiple'] = is_scalar
                if is_scalar:
                    results['scalar'] = lambda_guess
                results['intertwiner'] = T

        return results

    def commutant_dimension(self) -> int:
        """
        Compute dimension of commutant {T: TD(g) = D(g)T for all g}.
        """
        return self.find_intertwiners().shape[1]


def verify_orthogonality(rep1: Dict, rep2: Dict, group_elements: List) -> np.ndarray:
    """
    Verify orthogonality of matrix elements.

    Computes: Σ_g D^(α)_ij(g)* D^(β)_kl(g)
    """
    d1 = rep1[group_elements[0]].shape[0]
    d2 = rep2[group_elements[0]].shape[0]
    G = len(group_elements)

    result = np.zeros((d1, d1, d2, d2), dtype=complex)

    for g in group_elements:
        D1 = rep1[g]
        D2 = rep2[g]

        for i in range(d1):
            for j in range(d1):
                for k in range(d2):
                    for l in range(d2):
                        result[i, j, k, l] += np.conj(D1[i, j]) * D2[k, l]

    return result


# Create test representations
def create_S3_irreps() -> Dict[str, Dict]:
    """Create all irreps of S_3."""
    from itertools import permutations

    elements = list(permutations(range(3)))

    def sign(p):
        inv = sum(1 for i in range(3) for j in range(i+1, 3) if p[i] > p[j])
        return (-1) ** inv

    # Trivial rep
    triv = {p: np.array([[1.0]]) for p in elements}

    # Sign rep
    sgn = {p: np.array([[float(sign(p))]]) for p in elements}

    # Standard rep (on zero-sum subspace)
    omega = np.exp(2j * np.pi / 3)

    def std_matrix(p):
        # Permutation matrix
        P = np.zeros((3, 3))
        for i in range(3):
            P[p[i], i] = 1
        # Project onto zero-sum subspace
        # Basis: v1 = (1,-1,0)/√2, v2 = (1,1,-2)/√6
        v1 = np.array([1, -1, 0]) / np.sqrt(2)
        v2 = np.array([1, 1, -2]) / np.sqrt(6)
        B = np.column_stack([v1, v2])
        # D = B^T @ P @ B
        return B.T @ P @ B

    std = {p: std_matrix(p) for p in elements}

    return {'trivial': triv, 'sign': sgn, 'standard': std}


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("SCHUR'S LEMMA")
    print("=" * 60)

    irreps = create_S3_irreps()
    elements = list(irreps['trivial'].keys())

    # Example 1: Self-intertwiner of trivial rep
    print("\n1. SELF-INTERTWINER OF TRIVIAL REP")
    print("-" * 40)

    analyzer = SchurAnalyzer(irreps['trivial'])
    results = analyzer.verify_schur_part2()

    print(f"dim Hom(triv, triv) = {results['hom_dimension']}")
    print(f"Is scalar multiple? {results['is_scalar_multiple']}")

    # Example 2: Self-intertwiner of standard rep
    print("\n2. SELF-INTERTWINER OF STANDARD REP (2-dim)")
    print("-" * 40)

    analyzer = SchurAnalyzer(irreps['standard'])
    results = analyzer.verify_schur_part2()

    print(f"dim Hom(std, std) = {results['hom_dimension']}")
    print(f"Is scalar multiple? {results['is_scalar_multiple']}")
    if results['is_scalar_multiple']:
        print(f"Intertwiner = {results['scalar']:.4f} × I")

    # Example 3: Intertwiner between different irreps
    print("\n3. INTERTWINER: TRIVIAL → STANDARD")
    print("-" * 40)

    analyzer = SchurAnalyzer(irreps['trivial'], irreps['standard'])
    results = analyzer.verify_schur_part1()

    print(f"dim Hom(triv, std) = {results['hom_dimension']}")
    print(f"Conclusion: {results['conclusion']}")

    # Example 4: Intertwiner: standard → sign
    print("\n4. INTERTWINER: STANDARD → SIGN")
    print("-" * 40)

    analyzer = SchurAnalyzer(irreps['standard'], irreps['sign'])
    results = analyzer.verify_schur_part1()

    print(f"dim Hom(std, sgn) = {results['hom_dimension']}")
    print(f"Conclusion: {results['conclusion']}")

    # Example 5: Orthogonality of matrix elements
    print("\n5. ORTHOGONALITY OF MATRIX ELEMENTS")
    print("-" * 40)

    # For standard rep (2-dim)
    std = irreps['standard']
    G = len(elements)
    d = 2

    print(f"|G| = {G}, dim = {d}")
    print(f"Expected: Σ_g D*_ij(g) D_kl(g) = |G|/d × δ_ik δ_jl = {G/d:.2f} × δ_ik δ_jl")

    orth = verify_orthogonality(std, std, elements)

    print("\nActual values for (i,j,k,l):")
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    val = orth[i, j, k, l]
                    expected = G/d if (i == k and j == l) else 0
                    if np.abs(val - expected) < 0.01:
                        check = "✓"
                    else:
                        check = "✗"
                    print(f"  ({i},{j},{k},{l}): {val.real:6.2f} (expected {expected:.2f}) {check}")

    # Example 6: Orthogonality between different irreps
    print("\n6. ORTHOGONALITY: TRIVIAL vs STANDARD")
    print("-" * 40)

    orth_cross = verify_orthogonality(irreps['trivial'], irreps['standard'], elements)
    print(f"All matrix element products should be 0:")
    print(f"  Σ_g D^(triv)(g)* D^(std)_00(g) = {orth_cross[0,0,0,0].real:.4f}")
    print(f"  Σ_g D^(triv)(g)* D^(std)_01(g) = {orth_cross[0,0,0,1].real:.4f}")
    print(f"  Σ_g D^(triv)(g)* D^(std)_10(g) = {orth_cross[0,0,1,0].real:.4f}")
    print(f"  Σ_g D^(triv)(g)* D^(std)_11(g) = {orth_cross[0,0,1,1].real:.4f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Schur Part 1: Intertwiner is 0 or isomorphism (for irreps)
    2. Schur Part 2: Self-intertwiner = λI (over ℂ)
    3. Abelian groups have only 1-dim irreps
    4. Commutant of irreducible rep = scalars
    5. Selection rules arise from Schur's lemma
    6. Matrix elements of different irreps are orthogonal
    """)
```

---

## 8. Practice Problems

### Problem Set A: Basic Applications

**A1.** Use Schur's lemma to prove that all irreps of $\mathbb{Z}_n$ are 1-dimensional.

**A2.** Show that if $T$ commutes with all matrices in an irreducible representation, then $T = \lambda I$.

**A3.** For the 2-dim standard rep of $S_3$, verify there are no invariant 1-dim subspaces.

### Problem Set B: Intertwining Operators

**B1.** Find all intertwining operators from the permutation rep of $S_3$ to itself.

**B2.** Prove: If $\rho_1 \not\cong \rho_2$ are irreducible, then $\text{Hom}_G(\rho_1, \rho_2) = \{0\}$.

**B3.** Compute the dimension of $\text{Hom}_G(\rho, \rho \oplus \rho)$ where $\rho$ is irreducible.

### Problem Set C: Quantum Applications

**C1.** A Hamiltonian $H$ commutes with all rotations. Show that eigenstates with the same $\ell$ have the same energy.

**C2.** **(Selection Rules)** Using Schur's lemma, explain why $\langle l', m' | \hat{z} | l, m \rangle = 0$ unless $l' = l \pm 1$.

**C3.** Prove: Two operators that both commute with an irreducible representation must commute with each other.

---

## 9. Summary

### Schur's Lemma

$$\boxed{\text{Part 1: } T: V_1 \to V_2 \text{ intertwiner of irreps} \Rightarrow T = 0 \text{ or } T \text{ iso}}$$

$$\boxed{\text{Part 2 (over } \mathbb{C}\text{): } T \text{ self-intertwiner of irrep} \Rightarrow T = \lambda I}$$

### Key Consequences

| Result | Implication |
|--------|------------|
| $\text{Hom}_G(\rho_1, \rho_2) = 0$ if $\rho_1 \not\cong \rho_2$ | Different irreps "don't talk" |
| Commutant of irrep = $\mathbb{C} \cdot I$ | Only scalars commute |
| Abelian $\Rightarrow$ all irreps 1-dim | Complete diagonalization |

---

## 10. Preview: Day 291

Tomorrow we study **characters**:
- Definition and basic properties
- Character tables
- Class functions
- Using characters to decompose representations

---

*"Schur's lemma is the most basic tool in representation theory. Almost everything else flows from it." — I.M. Gelfand*
