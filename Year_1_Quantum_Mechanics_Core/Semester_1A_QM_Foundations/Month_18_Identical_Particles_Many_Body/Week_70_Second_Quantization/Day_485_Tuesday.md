# Day 485: Bosonic Creation and Annihilation Operators

## Overview

**Day 485 of 2520 | Week 70, Day 2 | Month 18: Identical Particles & Many-Body Physics**

Today we introduce the bosonic creation and annihilation operators, the fundamental tools for building and manipulating Fock space states. These operators satisfy canonical commutation relations and provide an elegant algebraic framework for many-body quantum mechanics. The connection to the harmonic oscillator ladder operators reveals deep mathematical unity across quantum physics.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Creation & Annihilation Operator Definitions | 60 min |
| 10:00 AM | Canonical Commutation Relations | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Action on Fock States | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Harmonic Oscillator Connection | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Multi-Mode Systems | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define** bosonic creation ($\hat{a}^\dagger$) and annihilation ($\hat{a}$) operators
2. **Derive** and apply the canonical commutation relation $[\hat{a}, \hat{a}^\dagger] = 1$
3. **Calculate** the action of these operators on Fock states
4. **Express** the number operator as $\hat{n} = \hat{a}^\dagger \hat{a}$
5. **Connect** bosonic operators to harmonic oscillator ladder operators
6. **Generalize** to multi-mode systems with commutation relations $[\hat{a}_i, \hat{a}_j^\dagger] = \delta_{ij}$

---

## 1. Defining Creation and Annihilation Operators

### Physical Motivation

Instead of specifying occupation numbers directly, we build states by:
- **Creating** particles: adding one particle to a mode
- **Annihilating** particles: removing one particle from a mode

Starting from the vacuum $|0\rangle$, we can construct any Fock state!

### The Annihilation Operator $\hat{a}$

The annihilation operator removes one particle from a single-mode bosonic state:

$$\boxed{\hat{a}|n\rangle = \sqrt{n}|n-1\rangle}$$

**Key properties:**
- Reduces occupation by 1
- The $\sqrt{n}$ factor ensures proper normalization
- $\hat{a}|0\rangle = 0$ (cannot remove particle from vacuum)

### The Creation Operator $\hat{a}^\dagger$

The creation operator adds one particle:

$$\boxed{\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle}$$

**Key properties:**
- Increases occupation by 1
- The $\sqrt{n+1}$ factor ensures proper normalization
- Can always add particles (no upper limit for bosons)

### Hermitian Conjugate Relationship

$\hat{a}$ and $\hat{a}^\dagger$ are Hermitian conjugates:

$$\langle m | \hat{a} | n \rangle = \langle n | \hat{a}^\dagger | m \rangle^*$$

**Proof:**
$$\langle m | \hat{a} | n \rangle = \sqrt{n} \langle m | n-1 \rangle = \sqrt{n} \delta_{m, n-1}$$
$$\langle n | \hat{a}^\dagger | m \rangle^* = (\sqrt{m+1} \langle n | m+1 \rangle)^* = \sqrt{m+1} \delta_{n, m+1} = \sqrt{n} \delta_{m, n-1}$$ ✓

---

## 2. Canonical Commutation Relations (CCR)

### The Fundamental Commutator

The defining algebraic property of bosonic operators:

$$\boxed{[\hat{a}, \hat{a}^\dagger] \equiv \hat{a}\hat{a}^\dagger - \hat{a}^\dagger\hat{a} = 1}$$

### Derivation

Acting on state $|n\rangle$:

$$\hat{a}\hat{a}^\dagger|n\rangle = \hat{a}(\sqrt{n+1}|n+1\rangle) = \sqrt{n+1}\sqrt{n+1}|n\rangle = (n+1)|n\rangle$$

$$\hat{a}^\dagger\hat{a}|n\rangle = \hat{a}^\dagger(\sqrt{n}|n-1\rangle) = \sqrt{n}\sqrt{n}|n\rangle = n|n\rangle$$

Therefore:
$$[\hat{a}, \hat{a}^\dagger]|n\rangle = (n+1 - n)|n\rangle = |n\rangle$$

Since this holds for all basis states: $[\hat{a}, \hat{a}^\dagger] = \hat{I} = 1$ ✓

### Other Commutators

$$[\hat{a}, \hat{a}] = 0$$
$$[\hat{a}^\dagger, \hat{a}^\dagger] = 0$$

**Interpretation:** Multiple creations or annihilations commute (order doesn't matter for bosons).

### Commutation vs Anticommutation

| Bosons | Fermions |
|--------|----------|
| $[\hat{a}, \hat{a}^\dagger] = 1$ | $\{\hat{c}, \hat{c}^\dagger\} = 1$ |
| Commutation relation | Anticommutation relation |
| Symmetric statistics | Antisymmetric statistics |

---

## 3. The Number Operator

### Definition from Creation/Annihilation

$$\boxed{\hat{n} = \hat{a}^\dagger \hat{a}}$$

### Verification

$$\hat{n}|n\rangle = \hat{a}^\dagger \hat{a}|n\rangle = \hat{a}^\dagger(\sqrt{n}|n-1\rangle) = \sqrt{n}\sqrt{n}|n\rangle = n|n\rangle$$ ✓

### Important Identity

From the CCR:
$$\hat{a}\hat{a}^\dagger = \hat{a}^\dagger\hat{a} + 1 = \hat{n} + 1$$

### Commutation with $\hat{a}$ and $\hat{a}^\dagger$

$$[\hat{n}, \hat{a}] = [\hat{a}^\dagger\hat{a}, \hat{a}] = \hat{a}^\dagger[\hat{a}, \hat{a}] + [\hat{a}^\dagger, \hat{a}]\hat{a} = 0 - \hat{a} = -\hat{a}$$

$$\boxed{[\hat{n}, \hat{a}] = -\hat{a}}$$

Similarly:
$$\boxed{[\hat{n}, \hat{a}^\dagger] = \hat{a}^\dagger}$$

**Physical interpretation:**
- $\hat{a}$ reduces particle number by 1 (eigenvalue shift: $n \to n-1$)
- $\hat{a}^\dagger$ increases particle number by 1 (eigenvalue shift: $n \to n+1$)

---

## 4. Building States from Vacuum

### Constructing Number States

Starting from vacuum $|0\rangle$:

$$|1\rangle = \hat{a}^\dagger|0\rangle$$

$$|2\rangle = \frac{1}{\sqrt{2}}(\hat{a}^\dagger)^2|0\rangle$$

$$|3\rangle = \frac{1}{\sqrt{3!}}(\hat{a}^\dagger)^3|0\rangle$$

**General formula:**

$$\boxed{|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle}$$

### Proof by Induction

**Base case:** $|0\rangle$ is vacuum (given).

**Inductive step:** Assume $|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle$.

$$\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$$

$$\hat{a}^\dagger \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle = \sqrt{n+1}|n+1\rangle$$

$$\frac{(\hat{a}^\dagger)^{n+1}}{\sqrt{n!}}|0\rangle = \sqrt{n+1}|n+1\rangle$$

$$|n+1\rangle = \frac{(\hat{a}^\dagger)^{n+1}}{\sqrt{(n+1)!}}|0\rangle$$ ✓

### The Vacuum Condition

$$\hat{a}|0\rangle = 0$$

This is the fundamental property defining the vacuum:
- No particles to remove
- Ground state of the system
- Lowest eigenstate of $\hat{n}$

---

## 5. Connection to Harmonic Oscillator

### Review: Harmonic Oscillator Ladder Operators

For the quantum harmonic oscillator with $\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$:

$$\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} + \frac{i\hat{p}}{m\omega}\right)$$

$$\hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} - \frac{i\hat{p}}{m\omega}\right)$$

### Hamiltonian in Terms of Number Operator

$$\hat{H} = \hbar\omega\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right) = \hbar\omega\left(\hat{n} + \frac{1}{2}\right)$$

**Energy eigenvalues:**
$$E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad n = 0, 1, 2, \ldots$$

### Reinterpretation: Phonons

The harmonic oscillator can be viewed as:
- **Classical:** Particle oscillating in a potential
- **Quantum (1st quantization):** Wave function $\psi_n(x)$
- **Quantum (2nd quantization):** $n$ **phonons** (quanta of vibration)

**Key insight:** The state $|n\rangle$ represents $n$ phonons, each with energy $\hbar\omega$.

### Position and Momentum Operators

Inverting the definitions:

$$\boxed{\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)}$$

$$\boxed{\hat{p} = i\sqrt{\frac{\hbar m\omega}{2}}(\hat{a}^\dagger - \hat{a})}$$

---

## 6. Multi-Mode Systems

### Multiple Bosonic Modes

For modes labeled by index $\alpha$:

$$\boxed{[\hat{a}_\alpha, \hat{a}_\beta^\dagger] = \delta_{\alpha\beta}}$$

$$[\hat{a}_\alpha, \hat{a}_\beta] = 0$$

$$[\hat{a}_\alpha^\dagger, \hat{a}_\beta^\dagger] = 0$$

### Action on Multi-Mode Fock States

$$\hat{a}_\alpha |n_1, n_2, \ldots, n_\alpha, \ldots\rangle = \sqrt{n_\alpha} |n_1, n_2, \ldots, n_\alpha - 1, \ldots\rangle$$

$$\hat{a}_\alpha^\dagger |n_1, n_2, \ldots, n_\alpha, \ldots\rangle = \sqrt{n_\alpha + 1} |n_1, n_2, \ldots, n_\alpha + 1, \ldots\rangle$$

### Number Operators

$$\hat{n}_\alpha = \hat{a}_\alpha^\dagger \hat{a}_\alpha$$

$$\hat{N} = \sum_\alpha \hat{n}_\alpha = \sum_\alpha \hat{a}_\alpha^\dagger \hat{a}_\alpha$$

### Building Multi-Mode States

$$|n_1, n_2, n_3, \ldots\rangle = \prod_\alpha \frac{(\hat{a}_\alpha^\dagger)^{n_\alpha}}{\sqrt{n_\alpha!}} |0\rangle$$

**Example:** $|2, 1, 0, 3\rangle$:

$$|2, 1, 0, 3\rangle = \frac{(\hat{a}_1^\dagger)^2}{\sqrt{2!}} \cdot \frac{\hat{a}_2^\dagger}{\sqrt{1!}} \cdot \frac{(\hat{a}_4^\dagger)^3}{\sqrt{3!}} |0\rangle$$

---

## 7. Worked Examples

### Example 1: Operator Algebra

**Problem:** Show that $[\hat{a}^2, (\hat{a}^\dagger)^2] = 4\hat{a}^\dagger\hat{a} + 2$.

**Solution:**

Use the identity $[\hat{A}, \hat{B}\hat{C}] = \hat{B}[\hat{A}, \hat{C}] + [\hat{A}, \hat{B}]\hat{C}$.

$$[\hat{a}^2, (\hat{a}^\dagger)^2] = [\hat{a} \cdot \hat{a}, \hat{a}^\dagger \cdot \hat{a}^\dagger]$$

First: $[\hat{a}, (\hat{a}^\dagger)^2] = \hat{a}^\dagger[\hat{a}, \hat{a}^\dagger] + [\hat{a}, \hat{a}^\dagger]\hat{a}^\dagger = \hat{a}^\dagger + \hat{a}^\dagger = 2\hat{a}^\dagger$

Then:
$$[\hat{a}^2, (\hat{a}^\dagger)^2] = \hat{a}[\hat{a}, (\hat{a}^\dagger)^2] + [\hat{a}, (\hat{a}^\dagger)^2]\hat{a}$$
$$= \hat{a}(2\hat{a}^\dagger) + (2\hat{a}^\dagger)\hat{a}$$
$$= 2\hat{a}\hat{a}^\dagger + 2\hat{a}^\dagger\hat{a}$$
$$= 2(\hat{a}^\dagger\hat{a} + 1) + 2\hat{a}^\dagger\hat{a}$$
$$= \boxed{4\hat{a}^\dagger\hat{a} + 2 = 4\hat{n} + 2}$$

### Example 2: Matrix Elements

**Problem:** Calculate $\langle m | (\hat{a} + \hat{a}^\dagger) | n \rangle$ and $\langle m | (\hat{a} + \hat{a}^\dagger)^2 | n \rangle$.

**Solution:**

Part (a):
$$\langle m | (\hat{a} + \hat{a}^\dagger) | n \rangle = \langle m | \hat{a} | n \rangle + \langle m | \hat{a}^\dagger | n \rangle$$
$$= \sqrt{n}\delta_{m, n-1} + \sqrt{n+1}\delta_{m, n+1}$$

$$\boxed{\langle m | (\hat{a} + \hat{a}^\dagger) | n \rangle = \sqrt{n}\delta_{m, n-1} + \sqrt{n+1}\delta_{m, n+1}}$$

Part (b):
$$(\hat{a} + \hat{a}^\dagger)^2 = \hat{a}^2 + \hat{a}\hat{a}^\dagger + \hat{a}^\dagger\hat{a} + (\hat{a}^\dagger)^2$$
$$= \hat{a}^2 + (\hat{n} + 1) + \hat{n} + (\hat{a}^\dagger)^2 = \hat{a}^2 + (\hat{a}^\dagger)^2 + 2\hat{n} + 1$$

$$\langle m | (\hat{a} + \hat{a}^\dagger)^2 | n \rangle = \sqrt{n(n-1)}\delta_{m,n-2} + (2n+1)\delta_{m,n} + \sqrt{(n+1)(n+2)}\delta_{m,n+2}$$

### Example 3: Two-Mode State

**Problem:** A system has two modes in state $|\psi\rangle = \frac{1}{\sqrt{2}}(|1, 0\rangle + |0, 1\rangle)$. Calculate $\langle \hat{a}_1^\dagger \hat{a}_2 \rangle$.

**Solution:**

$$\hat{a}_1^\dagger \hat{a}_2 |1, 0\rangle = \hat{a}_1^\dagger \cdot 0 = 0$$
$$\hat{a}_1^\dagger \hat{a}_2 |0, 1\rangle = \hat{a}_1^\dagger |0, 0\rangle = |1, 0\rangle$$

$$\langle \psi | \hat{a}_1^\dagger \hat{a}_2 | \psi \rangle = \frac{1}{2}\langle 1,0 | + \frac{1}{2}\langle 0,1 | \cdot (0 + |1, 0\rangle)$$
$$= \frac{1}{2}\langle 1,0 | 1,0\rangle = \boxed{\frac{1}{2}}$$

**Interpretation:** This measures correlation between modes 1 and 2.

---

## 8. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate $\hat{a}|5\rangle$, $\hat{a}^\dagger|5\rangle$, and $\hat{n}|5\rangle$.

**Problem 1.2:** Express $|4\rangle$ in terms of creation operators acting on vacuum.

**Problem 1.3:** Verify $[\hat{a}, \hat{a}^\dagger] = 1$ by computing both $\hat{a}\hat{a}^\dagger|3\rangle$ and $\hat{a}^\dagger\hat{a}|3\rangle$.

### Level 2: Intermediate

**Problem 2.1:** Show that $[\hat{n}, \hat{a}^k] = -k\hat{a}^k$ and $[\hat{n}, (\hat{a}^\dagger)^k] = k(\hat{a}^\dagger)^k$.

**Problem 2.2:** Calculate the matrix representation of $\hat{a}$, $\hat{a}^\dagger$, and $\hat{n}$ in the basis $\{|0\rangle, |1\rangle, |2\rangle, |3\rangle\}$.

**Problem 2.3:** For the state $|\psi\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|2\rangle$, calculate:
(a) $\langle \hat{n} \rangle$
(b) $\langle \hat{n}^2 \rangle$ and $\Delta n$
(c) $\langle \hat{a} + \hat{a}^\dagger \rangle$

### Level 3: Challenging

**Problem 3.1:** The **coherent state** is defined as $|\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^\infty \frac{\alpha^n}{\sqrt{n!}}|n\rangle$. Show that:
(a) $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$ (eigenstate of annihilation operator)
(b) $|\alpha\rangle = e^{-|\alpha|^2/2}e^{\alpha\hat{a}^\dagger}|0\rangle$

**Problem 3.2:** Prove the Baker-Campbell-Hausdorff formula for bosons: $e^{\hat{A}}e^{\hat{B}} = e^{\hat{A}+\hat{B}+[\hat{A},\hat{B}]/2}$ when $[\hat{A}, [\hat{A}, \hat{B}]] = [\hat{B}, [\hat{A}, \hat{B}]] = 0$.

**Problem 3.3:** For a two-mode system, define $\hat{J}_+ = \hat{a}_1^\dagger\hat{a}_2$, $\hat{J}_- = \hat{a}_2^\dagger\hat{a}_1$, $\hat{J}_z = \frac{1}{2}(\hat{n}_1 - \hat{n}_2)$. Show these satisfy the SU(2) algebra: $[\hat{J}_z, \hat{J}_\pm] = \pm\hat{J}_\pm$, $[\hat{J}_+, \hat{J}_-] = 2\hat{J}_z$.

---

## 9. Computational Lab: Bosonic Operators

```python
"""
Day 485 Computational Lab: Bosonic Creation and Annihilation Operators
Implementing bosonic algebra and demonstrating operator properties.
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

class BosonicMode:
    """
    Represents a single bosonic mode with creation/annihilation operators.
    Works in truncated Fock space with maximum occupation n_max.
    """

    def __init__(self, n_max=20):
        """
        Initialize bosonic mode.

        Parameters:
        -----------
        n_max : int
            Maximum occupation number (truncation)
        """
        self.n_max = n_max
        self.dim = n_max + 1

        # Build matrix representations
        self._build_operators()

    def _build_operators(self):
        """Construct matrix representations of operators."""

        # Annihilation operator: a|n⟩ = √n|n-1⟩
        self.a = np.zeros((self.dim, self.dim))
        for n in range(1, self.dim):
            self.a[n-1, n] = np.sqrt(n)

        # Creation operator: a†|n⟩ = √(n+1)|n+1⟩
        self.a_dag = self.a.T.copy()

        # Number operator: n = a†a
        self.n_op = self.a_dag @ self.a

        # Identity
        self.identity = np.eye(self.dim)

    def number_state(self, n):
        """Return the Fock state |n⟩ as a vector."""
        if n > self.n_max:
            raise ValueError(f"n={n} exceeds n_max={self.n_max}")
        state = np.zeros(self.dim)
        state[n] = 1.0
        return state

    def vacuum(self):
        """Return the vacuum state |0⟩."""
        return self.number_state(0)

    def coherent_state(self, alpha):
        """
        Return the coherent state |α⟩.
        |α⟩ = e^{-|α|²/2} Σₙ (αⁿ/√n!) |n⟩
        """
        state = np.zeros(self.dim, dtype=complex)
        for n in range(self.dim):
            state[n] = np.exp(-np.abs(alpha)**2 / 2) * (alpha**n) / np.sqrt(np.math.factorial(n))
        return state

    def expectation(self, state, operator):
        """Calculate ⟨ψ|O|ψ⟩."""
        return np.real(np.conj(state) @ operator @ state)

    def variance(self, state, operator):
        """Calculate ⟨O²⟩ - ⟨O⟩²."""
        mean = self.expectation(state, operator)
        mean_sq = self.expectation(state, operator @ operator)
        return mean_sq - mean**2

    def verify_ccr(self):
        """Verify canonical commutation relation [a, a†] = 1."""
        commutator = self.a @ self.a_dag - self.a_dag @ self.a
        error = np.max(np.abs(commutator - self.identity))
        print(f"[a, a†] = 1 verification: max error = {error:.2e}")
        return error < 1e-10


def demonstrate_basic_operations():
    """Demonstrate basic bosonic operator operations."""

    print("=" * 60)
    print("BOSONIC OPERATOR DEMONSTRATION")
    print("=" * 60)

    mode = BosonicMode(n_max=10)

    # Verify CCR
    mode.verify_ccr()

    # Action on number states
    print("\n--- Action on Number States ---")
    for n in [0, 1, 3, 5]:
        state_n = mode.number_state(n)

        # Annihilation
        a_state = mode.a @ state_n
        if n > 0:
            expected_coeff = np.sqrt(n)
            actual_coeff = a_state[n-1]
            print(f"a|{n}⟩ = √{n}|{n-1}⟩ = {expected_coeff:.4f}|{n-1}⟩ (computed: {actual_coeff:.4f})")
        else:
            print(f"a|0⟩ = 0 (norm: {np.linalg.norm(a_state):.4f})")

        # Creation
        a_dag_state = mode.a_dag @ state_n
        expected_coeff = np.sqrt(n + 1)
        actual_coeff = a_dag_state[n+1]
        print(f"a†|{n}⟩ = √{n+1}|{n+1}⟩ = {expected_coeff:.4f}|{n+1}⟩ (computed: {actual_coeff:.4f})")

    # Number operator
    print("\n--- Number Operator ---")
    for n in range(6):
        state_n = mode.number_state(n)
        n_exp = mode.expectation(state_n, mode.n_op)
        print(f"⟨{n}|n̂|{n}⟩ = {n_exp:.4f} (expected: {n})")


def demonstrate_building_states():
    """Show how to build Fock states from vacuum."""

    print("\n" + "=" * 60)
    print("BUILDING STATES FROM VACUUM")
    print("=" * 60)

    mode = BosonicMode(n_max=10)
    vac = mode.vacuum()

    print("\n|n⟩ = (a†)ⁿ/√n! |0⟩")
    print("-" * 40)

    for n in range(6):
        # Build |n⟩ using (a†)^n / √n!
        state_built = vac.copy()
        for _ in range(n):
            state_built = mode.a_dag @ state_built
        state_built = state_built / np.sqrt(np.math.factorial(n))

        # Compare with direct construction
        state_direct = mode.number_state(n)

        # Check overlap
        overlap = np.abs(np.dot(state_built, state_direct))
        print(f"|{n}⟩: overlap = {overlap:.6f}, norm = {np.linalg.norm(state_built):.6f}")


def coherent_state_analysis():
    """Analyze properties of coherent states."""

    print("\n" + "=" * 60)
    print("COHERENT STATE ANALYSIS")
    print("=" * 60)

    mode = BosonicMode(n_max=30)

    alphas = [0.5, 1.0, 2.0, 3.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, alpha in enumerate(alphas):
        ax = axes[idx // 2, idx % 2]

        # Create coherent state
        psi = mode.coherent_state(alpha)

        # Probability distribution
        probs = np.abs(psi)**2
        ns = np.arange(mode.dim)

        # Poisson distribution for comparison
        poisson = np.exp(-np.abs(alpha)**2) * (np.abs(alpha)**(2*ns)) / \
                  np.array([np.math.factorial(n) for n in ns])

        ax.bar(ns[:15], probs[:15], alpha=0.6, label=f'|α={alpha}⟩')
        ax.plot(ns[:15], poisson[:15], 'ro-', markersize=4, label='Poisson')
        ax.set_xlabel('n')
        ax.set_ylabel('P(n)')
        ax.set_title(f'α = {alpha}, ⟨n⟩ = {np.abs(alpha)**2:.2f}')
        ax.legend()

        # Verify eigenvalue equation: a|α⟩ = α|α⟩
        a_psi = mode.a @ psi
        eigenvalue_error = np.linalg.norm(a_psi - alpha * psi)
        print(f"α = {alpha}: a|α⟩ = α|α⟩ error = {eigenvalue_error:.2e}")

        # Statistics
        n_mean = mode.expectation(psi, mode.n_op)
        n_var = mode.variance(psi, mode.n_op)
        print(f"  ⟨n⟩ = {n_mean:.4f} (expected: {np.abs(alpha)**2:.4f})")
        print(f"  Δn = {np.sqrt(n_var):.4f} (expected: {np.abs(alpha):.4f})")

    plt.tight_layout()
    plt.savefig('coherent_states.png', dpi=150, bbox_inches='tight')
    plt.show()


def operator_matrices():
    """Visualize operator matrix representations."""

    print("\n" + "=" * 60)
    print("OPERATOR MATRIX REPRESENTATIONS")
    print("=" * 60)

    mode = BosonicMode(n_max=6)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    operators = [
        (mode.a, 'a (annihilation)'),
        (mode.a_dag, 'a† (creation)'),
        (mode.n_op, 'n̂ (number)'),
        (mode.a + mode.a_dag, 'a + a† (position-like)')
    ]

    for ax, (op, title) in zip(axes, operators):
        im = ax.imshow(np.real(op), cmap='RdBu', vmin=-3, vmax=3)
        ax.set_title(title)
        ax.set_xlabel('Column (n)')
        ax.set_ylabel('Row (m)')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Add text annotations for small matrices
        for i in range(min(5, mode.dim)):
            for j in range(min(5, mode.dim)):
                val = np.real(op[i, j])
                if np.abs(val) > 0.01:
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('operator_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print explicit matrices
    print("\n--- Explicit Matrix Elements (n_max=4) ---")
    mode_small = BosonicMode(n_max=4)

    print("\na (annihilation):")
    print(np.round(mode_small.a, 3))

    print("\na† (creation):")
    print(np.round(mode_small.a_dag, 3))


def harmonic_oscillator_connection():
    """Demonstrate connection to harmonic oscillator."""

    print("\n" + "=" * 60)
    print("HARMONIC OSCILLATOR CONNECTION")
    print("=" * 60)

    mode = BosonicMode(n_max=20)

    # Parameters (in natural units: ℏ = m = ω = 1)
    hbar = 1
    m = 1
    omega = 1

    # Position and momentum operators
    # x = √(ℏ/2mω) (a + a†)
    # p = i√(ℏmω/2) (a† - a)
    x_scale = np.sqrt(hbar / (2 * m * omega))
    p_scale = np.sqrt(hbar * m * omega / 2)

    x_op = x_scale * (mode.a + mode.a_dag)
    p_op = 1j * p_scale * (mode.a_dag - mode.a)

    # Hamiltonian: H = ℏω(n + 1/2)
    H = hbar * omega * (mode.n_op + 0.5 * mode.identity)

    # Verify energy eigenvalues
    print("\nEnergy Eigenvalues E_n = ℏω(n + 1/2):")
    print("-" * 40)
    for n in range(6):
        state = mode.number_state(n)
        E = mode.expectation(state, H)
        E_expected = hbar * omega * (n + 0.5)
        print(f"E_{n} = {E:.4f} (expected: {E_expected:.4f})")

    # Uncertainty relations
    print("\nUncertainty Relations:")
    print("-" * 40)
    for n in range(5):
        state = mode.number_state(n)

        # ⟨x⟩ and ⟨p⟩ should be zero for number states
        x_mean = mode.expectation(state, x_op)
        p_mean = mode.expectation(state, np.real(p_op))

        # Variances
        x_var = mode.variance(state, x_op)
        p_var = mode.variance(state, np.real(p_op))

        # Uncertainty product
        delta_x = np.sqrt(x_var)
        delta_p = np.sqrt(p_var)
        product = delta_x * delta_p

        print(f"|{n}⟩: Δx·Δp = {product:.4f} (≥ ℏ/2 = {hbar/2:.4f})")


def quantum_computing_connection():
    """Demonstrate relevance to quantum computing."""

    print("\n" + "=" * 60)
    print("QUANTUM COMPUTING CONNECTION")
    print("=" * 60)

    print("""
    BOSONIC MODES IN QUANTUM COMPUTING
    ==================================

    1. CAVITY QED / Circuit QED:
       - Microwave photons in superconducting resonators
       - a† creates a photon, a destroys a photon
       - Coherent states |α⟩ are "classical-like" states
       - Fock states |n⟩ are highly non-classical

    2. BOSONIC QUANTUM ERROR CORRECTION:
       - Cat codes: encode qubit in |α⟩ ± |-α⟩
       - GKP codes: use periodic structure in phase space
       - Binomial codes: specific superpositions of Fock states
       - Advantage: single oscillator can encode protected qubit

    3. CONTINUOUS-VARIABLE QC:
       - Qumodes instead of qubits
       - Gates: displacement D(α), squeezing S(r), beamsplitter
       - Universal computation possible with non-Gaussian gates

    4. QUANTUM SIMULATION:
       - Bosonic modes naturally simulate:
         * Phonons in crystals
         * Photons in cavities
         * Magnons in magnetic systems
       - No Jordan-Wigner transformation needed!

    Key Quantum Computing Operators:
    --------------------------------
    Displacement: D(α) = exp(αa† - α*a)
    Squeezing: S(r) = exp(r(a†² - a²)/2)
    Rotation: R(θ) = exp(-iθa†a) = exp(-iθn̂)
    """)

    mode = BosonicMode(n_max=15)

    # Demonstrate displacement operator
    alpha = 2.0

    # D(α)|0⟩ = |α⟩ (coherent state)
    D_matrix = expm(alpha * mode.a_dag - np.conj(alpha) * mode.a)

    vac = mode.vacuum()
    displaced = D_matrix @ vac
    coherent = mode.coherent_state(alpha)

    overlap = np.abs(np.dot(np.conj(displaced), coherent))
    print(f"\nDisplacement verification: ⟨D(α)|0⟩|α⟩ = {overlap:.6f}")


def two_mode_operations():
    """Demonstrate two-mode bosonic operations."""

    print("\n" + "=" * 60)
    print("TWO-MODE BOSONIC OPERATIONS")
    print("=" * 60)

    n_max = 5
    dim = n_max + 1

    # Single mode operators
    a1 = np.zeros((dim, dim))
    for n in range(1, dim):
        a1[n-1, n] = np.sqrt(n)
    a1_dag = a1.T

    a2 = a1.copy()
    a2_dag = a1_dag.copy()

    # Two-mode Hilbert space (tensor product)
    dim2 = dim**2

    # Operators on full space
    I = np.eye(dim)
    A1 = np.kron(a1, I)  # a₁ ⊗ I
    A1_dag = np.kron(a1_dag, I)
    A2 = np.kron(I, a2)  # I ⊗ a₂
    A2_dag = np.kron(I, a2_dag)

    # Verify commutation: [a₁, a₂†] = 0 (different modes commute)
    commutator_12 = A1 @ A2_dag - A2_dag @ A1
    print(f"[a₁, a₂†] = 0 verification: max error = {np.max(np.abs(commutator_12)):.2e}")

    # Beamsplitter operation: BS(θ) = exp(θ(a₁†a₂ - a₂†a₁))
    theta = np.pi / 4  # 50-50 beamsplitter

    BS_generator = theta * (A1_dag @ A2 - A2_dag @ A1)
    BS = expm(BS_generator)

    # Apply to |1,0⟩ (one photon in mode 1)
    state_10 = np.zeros(dim2)
    state_10[1 * dim + 0] = 1.0  # |1⟩⊗|0⟩

    output = BS @ state_10

    print(f"\nBeamsplitter (θ=π/4) on |1,0⟩:")
    print("Output state probabilities:")
    for n1 in range(3):
        for n2 in range(3):
            idx = n1 * dim + n2
            prob = np.abs(output[idx])**2
            if prob > 0.001:
                print(f"  |{n1},{n2}⟩: {prob:.4f}")


# Main execution
if __name__ == "__main__":
    print("Day 485: Bosonic Creation and Annihilation Operators")
    print("=" * 60)

    demonstrate_basic_operations()
    demonstrate_building_states()
    coherent_state_analysis()
    operator_matrices()
    harmonic_oscillator_connection()
    quantum_computing_connection()
    two_mode_operations()
```

---

## 10. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Annihilation $\hat{a}$ | Removes particle: $\hat{a}\|n\rangle = \sqrt{n}\|n-1\rangle$ |
| Creation $\hat{a}^\dagger$ | Adds particle: $\hat{a}^\dagger\|n\rangle = \sqrt{n+1}\|n+1\rangle$ |
| CCR | $[\hat{a}, \hat{a}^\dagger] = 1$ |
| Number operator | $\hat{n} = \hat{a}^\dagger\hat{a}$ |
| Building states | $\|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}\|0\rangle$ |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$[\hat{a}, \hat{a}^\dagger] = 1$$ | Canonical commutation relation |
| $$[\hat{n}, \hat{a}] = -\hat{a}$$ | $\hat{a}$ lowers $n$ by 1 |
| $$[\hat{n}, \hat{a}^\dagger] = \hat{a}^\dagger$$ | $\hat{a}^\dagger$ raises $n$ by 1 |
| $$[\hat{a}_\alpha, \hat{a}_\beta^\dagger] = \delta_{\alpha\beta}$$ | Multi-mode CCR |

---

## 11. Daily Checklist

### Conceptual Understanding
- [ ] I can define creation and annihilation operators physically
- [ ] I understand the canonical commutation relation
- [ ] I can explain why the vacuum satisfies $\hat{a}|0\rangle = 0$
- [ ] I see the connection to harmonic oscillator ladder operators

### Mathematical Skills
- [ ] I can apply $\hat{a}$ and $\hat{a}^\dagger$ to Fock states
- [ ] I can compute commutators involving these operators
- [ ] I can build Fock states from vacuum using $\hat{a}^\dagger$
- [ ] I can work with multi-mode systems

### Computational Skills
- [ ] I implemented bosonic operators as matrices
- [ ] I verified the CCR numerically
- [ ] I analyzed coherent states and their properties
- [ ] I explored two-mode operations

### Quantum Computing Connection
- [ ] I understand coherent states as "classical-like"
- [ ] I see relevance to cavity QED / circuit QED
- [ ] I know bosonic codes exist for quantum error correction

---

## 12. Preview: Day 486

Tomorrow we study **fermionic creation and annihilation operators**:

- Anticommutation relations $\{\hat{c}, \hat{c}^\dagger\} = 1$
- Pauli exclusion from algebra: $(\hat{c}^\dagger)^2 = 0$
- Jordan-Wigner transformation preview
- Mapping fermions to qubits

The fermionic case requires anticommutation instead of commutation, leading to fundamentally different physics.

---

## References

1. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 7.

2. Fetter, A.L. & Walecka, J.D. (2003). *Quantum Theory of Many-Particle Systems*. Dover, Ch. 1.

3. Gerry, C. & Knight, P. (2005). *Introductory Quantum Optics*. Cambridge University Press.

4. Nielsen, M.A. & Chuang, I.L. (2010). *Quantum Computation and Quantum Information*, Ch. 7.

---

*"The creation and annihilation operators are the alphabet of quantum field theory."*
— Steven Weinberg

---

**Day 485 Complete.** Tomorrow: Fermionic Operators.
