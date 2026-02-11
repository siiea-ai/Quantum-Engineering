# Day 486: Fermionic Creation and Annihilation Operators

## Overview

**Day 486 of 2520 | Week 70, Day 3 | Month 18: Identical Particles & Many-Body Physics**

Today we develop the formalism for fermionic systems, where particles obey the Pauli exclusion principle. The fundamental difference from bosons is the replacement of commutation relations with anticommutation relations. This algebraic change automatically enforces fermionic statistics, making second quantization an elegant framework for describing electrons, quarks, and all spin-1/2 particles. We also preview the Jordan-Wigner transformation, which maps fermions to qubits.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Fermionic Operators: Definition & Motivation | 60 min |
| 10:00 AM | Canonical Anticommutation Relations | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Pauli Exclusion from Algebra | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Multi-Mode Fermionic Systems | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Jordan-Wigner Transformation Preview | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define** fermionic creation ($\hat{c}^\dagger$) and annihilation ($\hat{c}$) operators
2. **Apply** canonical anticommutation relations (CAR): $\{\hat{c}, \hat{c}^\dagger\} = 1$
3. **Derive** the Pauli exclusion principle from anticommutation: $(\hat{c}^\dagger)^2 = 0$
4. **Construct** fermionic Fock states with proper sign conventions
5. **Explain** the ordering convention and its physical significance
6. **Preview** the Jordan-Wigner transformation for mapping fermions to qubits

---

## 1. Fermionic Operators: Definitions

### The Annihilation Operator $\hat{c}$

For a single fermionic mode with occupation $n \in \{0, 1\}$:

$$\boxed{\hat{c}|1\rangle = |0\rangle, \quad \hat{c}|0\rangle = 0}$$

The annihilation operator removes a fermion if present, gives zero if mode is empty.

### The Creation Operator $\hat{c}^\dagger$

$$\boxed{\hat{c}^\dagger|0\rangle = |1\rangle, \quad \hat{c}^\dagger|1\rangle = 0}$$

The creation operator adds a fermion if mode is empty, gives **zero** if already occupied!

### Comparison: Bosons vs Fermions

| Property | Bosons ($\hat{a}$, $\hat{a}^\dagger$) | Fermions ($\hat{c}$, $\hat{c}^\dagger$) |
|----------|---------------------------------------|----------------------------------------|
| Occupation | $n = 0, 1, 2, \ldots$ | $n = 0, 1$ only |
| $\hat{a}^\dagger\|n\rangle$ | $\sqrt{n+1}\|n+1\rangle$ | $0$ if $n=1$ |
| Statistics | Bose-Einstein | Fermi-Dirac |

### Matrix Representation (Single Mode)

In the basis $\{|0\rangle, |1\rangle\}$:

$$\hat{c} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad \hat{c}^\dagger = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$$

$$\hat{n} = \hat{c}^\dagger\hat{c} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

---

## 2. Canonical Anticommutation Relations (CAR)

### Definition of Anticommutator

$$\{A, B\} \equiv AB + BA$$

### The Fundamental CAR

$$\boxed{\{\hat{c}, \hat{c}^\dagger\} = \hat{c}\hat{c}^\dagger + \hat{c}^\dagger\hat{c} = 1}$$

### Verification

Acting on $|0\rangle$:
$$\hat{c}\hat{c}^\dagger|0\rangle = \hat{c}|1\rangle = |0\rangle$$
$$\hat{c}^\dagger\hat{c}|0\rangle = \hat{c}^\dagger \cdot 0 = 0$$
$$\{\hat{c}, \hat{c}^\dagger\}|0\rangle = |0\rangle + 0 = |0\rangle$$ ✓

Acting on $|1\rangle$:
$$\hat{c}\hat{c}^\dagger|1\rangle = \hat{c} \cdot 0 = 0$$
$$\hat{c}^\dagger\hat{c}|1\rangle = \hat{c}^\dagger|0\rangle = |1\rangle$$
$$\{\hat{c}, \hat{c}^\dagger\}|1\rangle = 0 + |1\rangle = |1\rangle$$ ✓

### Other Anticommutation Relations

$$\boxed{\{\hat{c}, \hat{c}\} = 2\hat{c}^2 = 0 \Rightarrow \hat{c}^2 = 0}$$

$$\boxed{\{c^\dagger, \hat{c}^\dagger\} = 2(\hat{c}^\dagger)^2 = 0 \Rightarrow (\hat{c}^\dagger)^2 = 0}$$

### Critical Consequence: Pauli Exclusion!

$$(\hat{c}^\dagger)^2 = 0$$

This means: **You cannot create two fermions in the same state!**

$$\hat{c}^\dagger\hat{c}^\dagger|0\rangle = (\hat{c}^\dagger)^2|0\rangle = 0$$

**The Pauli exclusion principle emerges automatically from the anticommutation algebra!**

---

## 3. Number Operator and Its Properties

### Definition

$$\boxed{\hat{n} = \hat{c}^\dagger\hat{c}}$$

### Eigenvalues

$$\hat{n}|0\rangle = 0|0\rangle, \quad \hat{n}|1\rangle = 1|1\rangle$$

Only eigenvalues 0 and 1 are possible!

### Useful Identity

From the CAR:
$$\hat{c}\hat{c}^\dagger = 1 - \hat{c}^\dagger\hat{c} = 1 - \hat{n}$$

This is the **particle-hole** relationship:
- $\hat{n}$ counts particles
- $1 - \hat{n}$ counts "holes" (empty states)

### Anticommutators with $\hat{n}$

Unlike bosons (where $[\hat{n}, \hat{a}] = -\hat{a}$), for fermions we have:

$$\{\hat{n}, \hat{c}\} = \hat{n}\hat{c} + \hat{c}\hat{n} = \hat{c}^\dagger\hat{c}\hat{c} + \hat{c}\hat{c}^\dagger\hat{c} = 0 + \hat{c}\hat{c}^\dagger\hat{c}$$

Using $\hat{c}^2 = 0$: $= (1-\hat{c}^\dagger\hat{c})\hat{c} = \hat{c}$

But more useful are the **commutators**:

$$[\hat{n}, \hat{c}] = \hat{n}\hat{c} - \hat{c}\hat{n} = -\hat{c}$$
$$[\hat{n}, \hat{c}^\dagger] = \hat{c}^\dagger$$

Same as bosons! This reflects that both $\hat{c}$ and $\hat{a}$ change particle number by 1.

---

## 4. Multi-Mode Fermionic Systems

### Canonical Anticommutation Relations

For modes labeled $i, j$:

$$\boxed{\{\hat{c}_i, \hat{c}_j^\dagger\} = \delta_{ij}}$$
$$\boxed{\{\hat{c}_i, \hat{c}_j\} = 0}$$
$$\boxed{\{\hat{c}_i^\dagger, \hat{c}_j^\dagger\} = 0}$$

### Critical Point: Different Modes Anticommute!

For $i \neq j$:
$$\hat{c}_i \hat{c}_j = -\hat{c}_j \hat{c}_i$$
$$\hat{c}_i^\dagger \hat{c}_j^\dagger = -\hat{c}_j^\dagger \hat{c}_i^\dagger$$

**This is the algebraic origin of antisymmetric wave functions!**

### Building Multi-Mode States

The state with fermions in modes $\alpha_1, \alpha_2, \ldots, \alpha_N$:

$$|\alpha_1, \alpha_2, \ldots, \alpha_N\rangle = \hat{c}_{\alpha_1}^\dagger \hat{c}_{\alpha_2}^\dagger \cdots \hat{c}_{\alpha_N}^\dagger |0\rangle$$

**Order matters!** Swapping two operators introduces a minus sign:

$$\hat{c}_1^\dagger \hat{c}_2^\dagger |0\rangle = -\hat{c}_2^\dagger \hat{c}_1^\dagger |0\rangle$$

### Occupation Number Representation

Standard convention: **Order modes by increasing index**

$$|n_1, n_2, n_3, \ldots\rangle = (\hat{c}_1^\dagger)^{n_1} (\hat{c}_2^\dagger)^{n_2} (\hat{c}_3^\dagger)^{n_3} \cdots |0\rangle$$

where each $n_i \in \{0, 1\}$.

**Example:** $|1, 0, 1, 1, 0\rangle = \hat{c}_1^\dagger \hat{c}_3^\dagger \hat{c}_4^\dagger |0\rangle$

### Sign Convention for Operators

When acting with $\hat{c}_j$ or $\hat{c}_j^\dagger$ on a multi-mode state, we must account for anticommutation with operators to the left:

$$\hat{c}_j |n_1, \ldots, n_j, \ldots\rangle = (-1)^{\sum_{k<j} n_k} n_j |n_1, \ldots, n_j - 1, \ldots\rangle$$

$$\hat{c}_j^\dagger |n_1, \ldots, n_j, \ldots\rangle = (-1)^{\sum_{k<j} n_k} (1 - n_j) |n_1, \ldots, n_j + 1, \ldots\rangle$$

The phase $(-1)^{\sum_{k<j} n_k}$ counts the number of occupied modes to the left.

---

## 5. Connection to Antisymmetric Wave Functions

### Two-Fermion Example

First quantization (Slater determinant):
$$\Psi(x_1, x_2) = \frac{1}{\sqrt{2}}[\phi_1(x_1)\phi_2(x_2) - \phi_2(x_1)\phi_1(x_2)]$$

Second quantization:
$$|1, 1, 0, \ldots\rangle = \hat{c}_1^\dagger \hat{c}_2^\dagger |0\rangle$$

The antisymmetry is encoded in:
$$\hat{c}_1^\dagger \hat{c}_2^\dagger = -\hat{c}_2^\dagger \hat{c}_1^\dagger$$

### General N-Fermion State

The Slater determinant:
$$\Psi = \frac{1}{\sqrt{N!}} \begin{vmatrix} \phi_1(x_1) & \phi_1(x_2) & \cdots \\ \phi_2(x_1) & \phi_2(x_2) & \cdots \\ \vdots & \vdots & \ddots \end{vmatrix}$$

becomes simply:
$$\hat{c}_1^\dagger \hat{c}_2^\dagger \cdots \hat{c}_N^\dagger |0\rangle$$

**No explicit symmetrization needed!** The anticommutation algebra handles it automatically.

---

## 6. Jordan-Wigner Transformation Preview

### The Mapping Problem

Qubits have commuting Pauli operators on different sites:
$$[\sigma_i^\alpha, \sigma_j^\beta] = 0 \text{ for } i \neq j$$

But fermions anticommute:
$$\{\hat{c}_i, \hat{c}_j\} = 0 \text{ for } i \neq j$$

How do we simulate fermions on a quantum computer?

### Jordan-Wigner Transformation

The key insight: include a **string of Z operators** to track the sign:

$$\boxed{\hat{c}_j = \left(\prod_{k<j} Z_k\right) \sigma_j^-}$$

$$\boxed{\hat{c}_j^\dagger = \left(\prod_{k<j} Z_k\right) \sigma_j^+}$$

where $\sigma^- = \frac{1}{2}(X - iY)$ and $\sigma^+ = \frac{1}{2}(X + iY)$.

### Why It Works

The Z-string creates the anticommutation:
- For $i < j$: $\hat{c}_i$ includes $Z_i$, which anticommutes with $\sigma_j^\pm$
- This generates the required $-1$ when swapping operators

### Explicit Form

$$\hat{c}_j = \left(\bigotimes_{k=1}^{j-1} Z_k\right) \otimes \sigma_j^- \otimes \left(\bigotimes_{k=j+1}^{L} I_k\right)$$

### Number Operator (Simple!)

$$\hat{n}_j = \hat{c}_j^\dagger \hat{c}_j = \frac{1 - Z_j}{2}$$

No string needed for the number operator!

### Implications for Quantum Computing

1. **Fermionic simulation is possible** on qubit-based quantum computers
2. **Non-local strings** make some operations costly
3. **Alternative mappings** (Bravyi-Kitaev) reduce overhead
4. **Native fermionic hardware** would eliminate this overhead

---

## 7. Worked Examples

### Example 1: Two-Mode Anticommutation

**Problem:** Show that $\hat{c}_1^\dagger \hat{c}_2^\dagger \hat{c}_2 \hat{c}_1 = \hat{n}_1 \hat{n}_2$.

**Solution:**

Step 1: Move $\hat{c}_2$ past $\hat{c}_2^\dagger$.
Using $\{\hat{c}_2, \hat{c}_2^\dagger\} = 1$:
$$\hat{c}_2^\dagger \hat{c}_2 = 1 - \hat{c}_2 \hat{c}_2^\dagger \Rightarrow \hat{c}_2 \hat{c}_2^\dagger = 1 - \hat{n}_2$$

But we need $\hat{c}_1^\dagger \hat{c}_2^\dagger \hat{c}_2 \hat{c}_1$.

Step 2: Rewrite.
$$\hat{c}_1^\dagger (\hat{c}_2^\dagger \hat{c}_2) \hat{c}_1 = \hat{c}_1^\dagger \hat{n}_2 \hat{c}_1$$

Step 3: Commute $\hat{n}_2$ with $\hat{c}_1^\dagger$.
Since $[\hat{n}_2, \hat{c}_1^\dagger] = 0$ (different modes):
$$= \hat{n}_2 \hat{c}_1^\dagger \hat{c}_1 = \boxed{\hat{n}_1 \hat{n}_2}$$

### Example 2: Fermionic Swap

**Problem:** Calculate $\hat{c}_1^\dagger \hat{c}_2 + \hat{c}_2^\dagger \hat{c}_1$ acting on $|1, 1\rangle$ and $|1, 0\rangle$.

**Solution:**

**State $|1, 1\rangle = \hat{c}_1^\dagger \hat{c}_2^\dagger |0\rangle$:**

$\hat{c}_1^\dagger \hat{c}_2 |1, 1\rangle$:
- $\hat{c}_2 |1, 1\rangle$: Need to move $\hat{c}_2$ past $\hat{c}_1^\dagger$ in the state definition.
- $\hat{c}_2 \hat{c}_1^\dagger \hat{c}_2^\dagger |0\rangle = -\hat{c}_1^\dagger \hat{c}_2 \hat{c}_2^\dagger |0\rangle = -\hat{c}_1^\dagger (1 - \hat{c}_2^\dagger \hat{c}_2)|0\rangle = -\hat{c}_1^\dagger |0\rangle = -|1, 0\rangle$
- Then $\hat{c}_1^\dagger(-|1, 0\rangle) = -\hat{c}_1^\dagger |1, 0\rangle = 0$ (mode 1 already occupied)

$\hat{c}_2^\dagger \hat{c}_1 |1, 1\rangle$:
- $\hat{c}_1 |1, 1\rangle = |0, 1\rangle$ (no sign since $\hat{c}_1$ acts first)
- $\hat{c}_2^\dagger |0, 1\rangle = 0$ (mode 2 already occupied)

Result: $(\hat{c}_1^\dagger \hat{c}_2 + \hat{c}_2^\dagger \hat{c}_1)|1, 1\rangle = \boxed{0}$

**State $|1, 0\rangle = \hat{c}_1^\dagger |0\rangle$:**

$\hat{c}_1^\dagger \hat{c}_2 |1, 0\rangle = \hat{c}_1^\dagger \cdot 0 = 0$

$\hat{c}_2^\dagger \hat{c}_1 |1, 0\rangle = \hat{c}_2^\dagger |0, 0\rangle = |0, 1\rangle$

Result: $(\hat{c}_1^\dagger \hat{c}_2 + \hat{c}_2^\dagger \hat{c}_1)|1, 0\rangle = \boxed{|0, 1\rangle}$

This operator **hops** a fermion between modes!

### Example 3: Jordan-Wigner Verification

**Problem:** Verify $\{\hat{c}_1, \hat{c}_2^\dagger\} = 0$ using the Jordan-Wigner mapping.

**Solution:**

$$\hat{c}_1 = \sigma_1^-, \quad \hat{c}_2^\dagger = Z_1 \sigma_2^+$$

$$\hat{c}_1 \hat{c}_2^\dagger = \sigma_1^- Z_1 \sigma_2^+ = \sigma_1^- Z_1 \otimes \sigma_2^+$$

Since $\sigma^- Z = -Z \sigma^-$ (verify: $\sigma^- = |0\rangle\langle 1|$, $Z|0\rangle = |0\rangle$, $Z|1\rangle = -|1\rangle$):

$$\sigma_1^- Z_1 = -Z_1 \sigma_1^- \cdot (-1) = ... $$

Actually, let's compute directly:
$$\sigma^- Z = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 0 & 0 \end{pmatrix} = -\sigma^-$$

So $\hat{c}_1 \hat{c}_2^\dagger = -Z_1 \sigma_1^- \sigma_2^+$

Similarly:
$$\hat{c}_2^\dagger \hat{c}_1 = Z_1 \sigma_2^+ \sigma_1^- = Z_1 \sigma_1^- \sigma_2^+$$ (since modes commute)

Therefore:
$$\hat{c}_1 \hat{c}_2^\dagger + \hat{c}_2^\dagger \hat{c}_1 = -Z_1 \sigma_1^- \sigma_2^+ + Z_1 \sigma_1^- \sigma_2^+ = \boxed{0}$$ ✓

---

## 8. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate $\hat{c}^\dagger \hat{c}|0\rangle$, $\hat{c}\hat{c}^\dagger|0\rangle$, and verify $\{\hat{c}, \hat{c}^\dagger\} = 1$.

**Problem 1.2:** Show that $(\hat{c}^\dagger)^3 = 0$ using anticommutation.

**Problem 1.3:** For two modes, expand $|0, 1\rangle$ and $|1, 1\rangle$ in terms of creation operators on vacuum.

### Level 2: Intermediate

**Problem 2.1:** Prove $\{\hat{c}_i, \hat{c}_j^\dagger\} = \delta_{ij}$ implies $[\hat{c}_i^\dagger\hat{c}_i, \hat{c}_j^\dagger\hat{c}_j] = 0$ for all $i, j$.

**Problem 2.2:** Calculate the matrix representation of the hopping operator $\hat{c}_1^\dagger\hat{c}_2 + \hat{c}_2^\dagger\hat{c}_1$ in the 2-mode, 1-particle basis $\{|1, 0\rangle, |0, 1\rangle\}$.

**Problem 2.3:** For 3 fermionic modes, how many basis states exist in the 2-particle sector? List them with proper signs.

### Level 3: Challenging

**Problem 3.1:** Show that the fermionic commutation $[\hat{c}_i^\dagger\hat{c}_i, \hat{c}_j^\dagger\hat{c}_k] = \delta_{ij}\hat{c}_i^\dagger\hat{c}_k - \delta_{ik}\hat{c}_j^\dagger\hat{c}_i$ follows from CAR.

**Problem 3.2:** Verify the Jordan-Wigner mapping preserves $\{\hat{c}_i, \hat{c}_j\} = 0$ for arbitrary $i \neq j$.

**Problem 3.3:** The **parity operator** is $\hat{P} = (-1)^{\hat{N}}$ where $\hat{N} = \sum_j \hat{n}_j$. Show that $\hat{P}\hat{c}_j\hat{P}^{-1} = -\hat{c}_j$ and $[\hat{P}, \hat{H}] = 0$ for any Hamiltonian that conserves particle number.

---

## 9. Computational Lab: Fermionic Operators

```python
"""
Day 486 Computational Lab: Fermionic Creation and Annihilation Operators
Implementing fermionic algebra and Jordan-Wigner transformation.
"""

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

class FermionicMode:
    """
    Single fermionic mode with creation/annihilation operators.
    Hilbert space: {|0⟩, |1⟩}
    """

    def __init__(self):
        """Initialize single fermionic mode."""
        self.dim = 2

        # Annihilation: c|1⟩ = |0⟩, c|0⟩ = 0
        self.c = np.array([[0, 1],
                          [0, 0]])

        # Creation: c†|0⟩ = |1⟩, c†|1⟩ = 0
        self.c_dag = np.array([[0, 0],
                               [1, 0]])

        # Number operator
        self.n = self.c_dag @ self.c

        # Identity
        self.I = np.eye(2)

    def vacuum(self):
        """Return |0⟩."""
        return np.array([1, 0])

    def occupied(self):
        """Return |1⟩."""
        return np.array([0, 1])

    def verify_car(self):
        """Verify canonical anticommutation relations."""
        # {c, c†} = 1
        anticomm = self.c @ self.c_dag + self.c_dag @ self.c
        error1 = np.max(np.abs(anticomm - self.I))

        # {c, c} = 0
        anticomm_cc = self.c @ self.c + self.c @ self.c
        error2 = np.max(np.abs(anticomm_cc))

        # {c†, c†} = 0
        anticomm_cdc = self.c_dag @ self.c_dag + self.c_dag @ self.c_dag
        error3 = np.max(np.abs(anticomm_cdc))

        print("CAR Verification:")
        print(f"  {{c, c†}} = 1: error = {error1:.2e}")
        print(f"  {{c, c}} = 0: error = {error2:.2e}")
        print(f"  {{c†, c†}} = 0: error = {error3:.2e}")

        return error1 < 1e-10 and error2 < 1e-10 and error3 < 1e-10


class MultiModeFermions:
    """
    Multi-mode fermionic system.
    Handles proper anticommutation signs.
    """

    def __init__(self, num_modes):
        """
        Initialize multi-mode system.

        Parameters:
        -----------
        num_modes : int
            Number of fermionic modes
        """
        self.num_modes = num_modes
        self.dim = 2**num_modes

        # Build operators
        self._build_operators()

    def _build_operators(self):
        """Construct creation/annihilation operators with proper signs."""

        self.c = []
        self.c_dag = []

        # Pauli matrices
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]])
        sigma_minus = np.array([[0, 1], [0, 0]])
        sigma_plus = np.array([[0, 0], [1, 0]])

        for j in range(self.num_modes):
            # c_j = Z⊗...⊗Z⊗σ⁻⊗I⊗...⊗I (j Z's, then σ⁻, then I's)
            # This is the Jordan-Wigner transformation

            op = np.array([[1]])

            # Z string for modes k < j
            for k in range(j):
                op = np.kron(op, Z)

            # σ⁻ at mode j
            op = np.kron(op, sigma_minus)

            # Identity for modes k > j
            for k in range(j + 1, self.num_modes):
                op = np.kron(op, I)

            self.c.append(op)
            self.c_dag.append(op.T.conj())

        # Number operators (no Jordan-Wigner string needed)
        self.n = [self.c_dag[j] @ self.c[j] for j in range(self.num_modes)]

        # Total number operator
        self.N = sum(self.n)

        # Identity
        self.I = np.eye(self.dim)

    def verify_car(self):
        """Verify all anticommutation relations."""

        print("Multi-mode CAR Verification:")
        max_error = 0

        for i in range(self.num_modes):
            for j in range(self.num_modes):
                # {c_i, c_j†} = δ_ij
                anticomm = self.c[i] @ self.c_dag[j] + self.c_dag[j] @ self.c[i]
                expected = self.I if i == j else np.zeros_like(self.I)
                error = np.max(np.abs(anticomm - expected))
                max_error = max(max_error, error)

                # {c_i, c_j} = 0
                anticomm_cc = self.c[i] @ self.c[j] + self.c[j] @ self.c[i]
                error = np.max(np.abs(anticomm_cc))
                max_error = max(max_error, error)

        print(f"  Maximum error across all relations: {max_error:.2e}")
        return max_error < 1e-10

    def occupation_state(self, occupations):
        """
        Return state |n_1, n_2, ...⟩.

        Parameters:
        -----------
        occupations : list
            List of 0s and 1s
        """
        if len(occupations) != self.num_modes:
            raise ValueError("Wrong number of modes")

        state = np.zeros(self.dim)
        index = sum(n * 2**(self.num_modes - 1 - i)
                   for i, n in enumerate(occupations))
        state[int(index)] = 1.0
        return state

    def vacuum(self):
        """Return vacuum state."""
        return self.occupation_state([0] * self.num_modes)

    def basis_states(self, N_particles):
        """
        Generate all basis states with N particles.

        Returns:
        --------
        list : List of (occupation_tuple, state_vector) pairs
        """
        states = []
        for occupied_modes in combinations(range(self.num_modes), N_particles):
            occupations = [0] * self.num_modes
            for m in occupied_modes:
                occupations[m] = 1
            state = self.occupation_state(occupations)
            states.append((tuple(occupations), state))
        return states


def demonstrate_single_mode():
    """Demonstrate single-mode fermionic operators."""

    print("=" * 60)
    print("SINGLE FERMIONIC MODE")
    print("=" * 60)

    mode = FermionicMode()
    mode.verify_car()

    print("\nOperator matrices:")
    print("c =")
    print(mode.c)
    print("\nc† =")
    print(mode.c_dag)
    print("\nn = c†c =")
    print(mode.n)

    # Demonstrate Pauli exclusion
    print("\n--- Pauli Exclusion ---")
    vac = mode.vacuum()
    occ = mode.occupied()

    state = mode.c_dag @ vac
    print(f"c†|0⟩ = {state} = |1⟩ ✓")

    state = mode.c_dag @ occ
    print(f"c†|1⟩ = {state} = 0 (Pauli exclusion!)")

    print(f"\n(c†)² = ")
    print(mode.c_dag @ mode.c_dag)
    print("= 0 (algebraic Pauli exclusion)")


def demonstrate_multi_mode():
    """Demonstrate multi-mode fermionic system."""

    print("\n" + "=" * 60)
    print("MULTI-MODE FERMIONIC SYSTEM")
    print("=" * 60)

    system = MultiModeFermions(num_modes=3)
    system.verify_car()

    # Show anticommutation between different modes
    print("\n--- Anticommutation between modes ---")
    for i in range(2):
        for j in range(i+1, 3):
            result = system.c[i] @ system.c_dag[j] + system.c_dag[j] @ system.c[i]
            is_zero = np.allclose(result, 0)
            print(f"{{c_{i}, c†_{j}}} = 0: {is_zero}")

    # Build states
    print("\n--- Building Fock States ---")
    vac = system.vacuum()

    # |1,0,0⟩
    state_100 = system.c_dag[0] @ vac
    print(f"c†_0|vac⟩ = |1,0,0⟩")

    # |1,1,0⟩
    state_110 = system.c_dag[1] @ system.c_dag[0] @ vac
    print(f"c†_1 c†_0|vac⟩ = |1,1,0⟩")

    # Verify antisymmetry: c†_0 c†_1 = -c†_1 c†_0
    state_alt = system.c_dag[0] @ system.c_dag[1] @ vac
    print(f"\nc†_0 c†_1|vac⟩ vs c†_1 c†_0|vac⟩:")
    print(f"Sum = {np.linalg.norm(state_110 + state_alt):.6f} (should be 0)")


def demonstrate_jordan_wigner():
    """Demonstrate Jordan-Wigner transformation."""

    print("\n" + "=" * 60)
    print("JORDAN-WIGNER TRANSFORMATION")
    print("=" * 60)

    # Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    sigma_minus = (X - 1j*Y) / 2
    sigma_plus = (X + 1j*Y) / 2

    num_modes = 3

    print("\nJordan-Wigner mapping for 3 modes:")
    print("c_j = (⊗_{k<j} Z_k) ⊗ σ⁻_j ⊗ (⊗_{k>j} I_k)")

    print("\nExplicit:")
    print("c_0 = σ⁻ ⊗ I ⊗ I")
    print("c_1 = Z ⊗ σ⁻ ⊗ I")
    print("c_2 = Z ⊗ Z ⊗ σ⁻")

    # Verify this matches our MultiModeFermions implementation
    system = MultiModeFermions(num_modes=3)

    # Manual construction
    c0_manual = np.kron(np.kron(sigma_minus, I), I)
    c1_manual = np.kron(np.kron(Z, sigma_minus), I)
    c2_manual = np.kron(np.kron(Z, Z), sigma_minus)

    print("\nVerification against manual construction:")
    print(f"c_0 match: {np.allclose(system.c[0], c0_manual)}")
    print(f"c_1 match: {np.allclose(system.c[1], c1_manual)}")
    print(f"c_2 match: {np.allclose(system.c[2], c2_manual)}")

    # Number operator is local (no string)
    print("\nNumber operators (local, no JW string):")
    for j in range(3):
        n_j = (I - Z) / 2 if j == 0 else None
        # n_j = c†_j c_j
        expected = np.kron(np.kron(I if j != 0 else (I-Z)/2,
                                   I if j != 1 else (I-Z)/2),
                          I if j != 2 else (I-Z)/2)
        print(f"n_{j} = (I - Z_{j})/2")


def hopping_hamiltonian():
    """Study the fermionic hopping Hamiltonian."""

    print("\n" + "=" * 60)
    print("FERMIONIC HOPPING (TIGHT-BINDING) HAMILTONIAN")
    print("=" * 60)

    num_sites = 4
    system = MultiModeFermions(num_modes=num_sites)

    # Hopping Hamiltonian: H = -t Σ (c†_i c_{i+1} + h.c.)
    t = 1.0  # Hopping amplitude

    H = np.zeros((system.dim, system.dim), dtype=complex)
    for i in range(num_sites - 1):
        H += -t * (system.c_dag[i] @ system.c[i+1] +
                   system.c_dag[i+1] @ system.c[i])

    # Add periodic boundary condition
    H += -t * (system.c_dag[num_sites-1] @ system.c[0] +
               system.c_dag[0] @ system.c[num_sites-1])

    print(f"\nTight-binding model on {num_sites}-site ring")
    print("H = -t Σ (c†_i c_{i+1} + c†_{i+1} c_i)")

    # Eigenvalues (should match analytic result)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # For 1-particle sector
    one_particle_states = system.basis_states(1)
    dim_1p = len(one_particle_states)

    H_1p = np.zeros((dim_1p, dim_1p), dtype=complex)
    for i, (_, state_i) in enumerate(one_particle_states):
        for j, (_, state_j) in enumerate(one_particle_states):
            H_1p[i, j] = state_i.conj() @ H @ state_j

    E_1p = np.linalg.eigvalsh(H_1p)

    print(f"\n1-particle energies: {np.sort(E_1p)}")
    print(f"Analytic: E_k = -2t cos(2πk/N) for k = 0, 1, ..., N-1")

    k_values = np.arange(num_sites)
    E_analytic = -2 * t * np.cos(2 * np.pi * k_values / num_sites)
    print(f"Expected: {np.sort(E_analytic)}")


def plot_fermionic_spectrum():
    """Plot energy spectrum of fermionic model."""

    num_sites = 6
    system = MultiModeFermions(num_modes=num_sites)

    # Hopping Hamiltonian with PBC
    t = 1.0
    H = np.zeros((system.dim, system.dim), dtype=complex)
    for i in range(num_sites):
        j = (i + 1) % num_sites
        H += -t * (system.c_dag[i] @ system.c[j] +
                   system.c_dag[j] @ system.c[i])

    # Full spectrum
    eigenvalues = np.linalg.eigvalsh(H)

    # Classify by particle number
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, num_sites + 1))

    for N in range(num_sites + 1):
        states = system.basis_states(N)
        dim_N = len(states)

        if dim_N == 0:
            continue

        H_N = np.zeros((dim_N, dim_N), dtype=complex)
        for i, (_, state_i) in enumerate(states):
            for j, (_, state_j) in enumerate(states):
                H_N[i, j] = state_i.conj() @ H @ state_j

        E_N = np.linalg.eigvalsh(H_N)

        # Plot
        for E in E_N:
            ax.plot([N - 0.3, N + 0.3], [E, E], '-',
                   color=colors[N], linewidth=2)
        ax.scatter([N] * len(E_N), E_N, color=colors[N], s=50, zorder=5)

    ax.set_xlabel('Number of Fermions N', fontsize=12)
    ax.set_ylabel('Energy E/t', fontsize=12)
    ax.set_title(f'Fermionic Spectrum: {num_sites}-Site Ring', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(num_sites + 1))

    plt.tight_layout()
    plt.savefig('fermionic_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()


def quantum_computing_applications():
    """Discuss quantum computing applications."""

    print("\n" + "=" * 60)
    print("QUANTUM COMPUTING APPLICATIONS")
    print("=" * 60)

    print("""
    FERMIONS ON QUANTUM COMPUTERS
    =============================

    1. THE MAPPING CHALLENGE:
       - Qubits have local, commuting operators
       - Fermions have non-local anticommutation
       - Must transform fermion operators to qubit operators

    2. JORDAN-WIGNER TRANSFORMATION:
       c_j = (Z_0 ⊗ ... ⊗ Z_{j-1}) ⊗ σ⁻_j ⊗ I_{j+1} ⊗ ...

       Pros:
       - Exact mapping, preserves all physics
       - Local in 1D systems

       Cons:
       - Long Z-strings for 2D/3D systems
       - Non-local operations increase circuit depth

    3. ALTERNATIVE MAPPINGS:
       - Bravyi-Kitaev: O(log N) locality
       - Parity mapping: Different tradeoffs
       - Compact encodings for special cases

    4. APPLICATIONS:
       - Quantum chemistry (VQE for molecules)
       - Condensed matter (Hubbard model, Fermi-Hubbard)
       - High-energy physics (lattice gauge theory)

    5. CURRENT CHALLENGES:
       - Error rates still too high for large molecules
       - Need thousands of qubits for useful chemistry
       - Classical post-processing overhead (measurement)

    6. NEAR-TERM PROSPECTS:
       - Small molecules (H2, LiH, BeH2) demonstrated
       - Variational approaches (VQE, ADAPT-VQE)
       - Hybrid classical-quantum algorithms
    """)


# Main execution
if __name__ == "__main__":
    print("Day 486: Fermionic Creation and Annihilation Operators")
    print("=" * 60)

    demonstrate_single_mode()
    demonstrate_multi_mode()
    demonstrate_jordan_wigner()
    hopping_hamiltonian()
    plot_fermionic_spectrum()
    quantum_computing_applications()
```

---

## 10. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Fermionic operators | $\hat{c}$, $\hat{c}^\dagger$ with occupation $\in \{0, 1\}$ |
| CAR | $\{\hat{c}, \hat{c}^\dagger\} = 1$, $\{\hat{c}, \hat{c}\} = 0$ |
| Pauli exclusion | $(\hat{c}^\dagger)^2 = 0$ from anticommutation |
| Sign conventions | Order matters due to anticommutation |
| Jordan-Wigner | Maps fermions to qubits via Z-strings |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$\{\hat{c}_i, \hat{c}_j^\dagger\} = \delta_{ij}$$ | Canonical anticommutation |
| $$\{\hat{c}_i, \hat{c}_j\} = 0$$ | Fermions anticommute |
| $$(\hat{c}^\dagger)^2 = 0$$ | Pauli exclusion principle |
| $$\hat{c}_j = \left(\prod_{k<j} Z_k\right) \sigma_j^-$$ | Jordan-Wigner transformation |

---

## 11. Daily Checklist

### Conceptual Understanding
- [ ] I can define fermionic creation and annihilation operators
- [ ] I understand anticommutation vs commutation
- [ ] I see how Pauli exclusion emerges algebraically
- [ ] I understand the sign convention for multi-mode states

### Mathematical Skills
- [ ] I can verify CAR algebraically and numerically
- [ ] I can build fermionic Fock states from vacuum
- [ ] I can compute the action of operators with proper signs
- [ ] I understand the Jordan-Wigner transformation

### Computational Skills
- [ ] I implemented single and multi-mode fermionic systems
- [ ] I verified anticommutation numerically
- [ ] I computed the fermionic hopping Hamiltonian spectrum

### Quantum Computing Connection
- [ ] I understand why fermion-to-qubit mapping is needed
- [ ] I know the Jordan-Wigner transformation formula
- [ ] I see the relevance to quantum chemistry simulation

---

## 12. Preview: Day 487

Tomorrow we study **field operators**:

- Position-space operators $\hat{\psi}(r)$, $\hat{\psi}^\dagger(r)$
- Commutation/anticommutation in continuous position space
- Connection between field operators and wave functions
- Particle density operator $\hat{\rho}(r) = \hat{\psi}^\dagger(r)\hat{\psi}(r)$

Field operators bridge second quantization with the familiar wave function picture.

---

## References

1. Fetter, A.L. & Walecka, J.D. (2003). *Quantum Theory of Many-Particle Systems*. Dover, Ch. 1.

2. Jordan, P. & Wigner, E. (1928). "Uber das Paulische Aquivalenzverbot." *Z. Phys.* 47, 631.

3. Nielsen, M.A. & Chuang, I.L. (2010). *Quantum Computation and Quantum Information*, Ch. 9.

4. McArdle, S. et al. (2020). "Quantum computational chemistry." *Rev. Mod. Phys.* 92, 015003.

---

*"The anticommutation relations for fermions encode the Pauli exclusion principle in the most elegant possible way."*
— Philip Anderson

---

**Day 486 Complete.** Tomorrow: Field Operators.
