# Day 337: Complex Vector Spaces — The Arena of Quantum Mechanics

## Welcome to Year 1: Quantum Mechanics Core

**Congratulations!** You have completed Year 0's mathematical foundations. Today begins your journey into quantum mechanics proper. Everything you learned—linear algebra, complex analysis, functional analysis, classical mechanics—converges here.

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Complex Vector Spaces |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 337, you will be able to:

1. Define a complex vector space and verify the axioms
2. Understand why quantum mechanics requires complex numbers
3. Work with finite and infinite-dimensional vector spaces
4. Define linear independence, span, basis, and dimension
5. Express quantum states as vectors in abstract spaces
6. Connect abstract formalism to wave functions

---

## Core Content

### 1. Why Complex Vector Spaces?

Quantum mechanics fundamentally requires complex numbers. This is not a mathematical convenience—it's a physical necessity.

**Evidence from Nature:**
- Wave functions ψ(x,t) are intrinsically complex
- Interference requires complex amplitudes
- The Schrödinger equation explicitly contains i = √(-1)
- Probability amplitudes, not probabilities, add

**The Key Insight:**

$$\text{Probability} = |\text{amplitude}|^2 = |\psi|^2$$

Amplitudes are complex; probabilities are real. Interference arises because:

$$|c_1 + c_2|^2 = |c_1|^2 + |c_2|^2 + 2\text{Re}(c_1^* c_2) \neq |c_1|^2 + |c_2|^2$$

The cross term creates interference—impossible with real numbers alone.

---

### 2. Definition of a Vector Space

A **complex vector space** V is a set equipped with two operations:
- **Vector addition:** V × V → V
- **Scalar multiplication:** ℂ × V → V

satisfying the following axioms for all |ψ⟩, |φ⟩, |χ⟩ ∈ V and α, β ∈ ℂ:

**Addition Axioms:**
1. **Commutativity:** |ψ⟩ + |φ⟩ = |φ⟩ + |ψ⟩
2. **Associativity:** (|ψ⟩ + |φ⟩) + |χ⟩ = |ψ⟩ + (|φ⟩ + |χ⟩)
3. **Zero vector:** ∃ |0⟩ such that |ψ⟩ + |0⟩ = |ψ⟩
4. **Additive inverse:** ∀|ψ⟩, ∃|-ψ⟩ such that |ψ⟩ + |-ψ⟩ = |0⟩

**Scalar Multiplication Axioms:**
5. **Associativity:** α(β|ψ⟩) = (αβ)|ψ⟩
6. **Identity:** 1|ψ⟩ = |ψ⟩
7. **Distributivity (scalar):** (α + β)|ψ⟩ = α|ψ⟩ + β|ψ⟩
8. **Distributivity (vector):** α(|ψ⟩ + |φ⟩) = α|ψ⟩ + α|φ⟩

---

### 3. Examples of Vector Spaces in Physics

#### Example 1: ℂⁿ (Finite-Dimensional)

The space of n-tuples of complex numbers:

$$|ψ⟩ = \begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{pmatrix}, \quad c_i ∈ ℂ$$

**Physical realization:** An n-level quantum system (e.g., n=2 for a qubit)

#### Example 2: L²(ℝ) (Infinite-Dimensional)

Square-integrable functions on ℝ:

$$L^2(ℝ) = \left\{ ψ: ℝ → ℂ \,\middle|\, \int_{-∞}^{∞} |ψ(x)|^2 dx < ∞ \right\}$$

**Physical realization:** Wave functions of particles in 1D

#### Example 3: ℓ²(ℤ) (Countably Infinite)

Square-summable sequences:

$$ℓ^2 = \left\{ (c_0, c_1, c_2, \ldots) \,\middle|\, \sum_{n=0}^{∞} |c_n|^2 < ∞ \right\}$$

**Physical realization:** States in energy eigenbasis (e.g., harmonic oscillator)

---

### 4. Linear Independence and Span

**Definition (Linear Independence):**
Vectors |v₁⟩, |v₂⟩, ..., |vₙ⟩ are **linearly independent** if:

$$α_1|v_1⟩ + α_2|v_2⟩ + \cdots + α_n|v_n⟩ = |0⟩ \implies α_1 = α_2 = \cdots = α_n = 0$$

**Definition (Span):**
The **span** of vectors {|vᵢ⟩} is the set of all linear combinations:

$$\text{span}\{|v_1⟩, |v_2⟩, \ldots\} = \left\{ \sum_i α_i|v_i⟩ \,\middle|\, α_i ∈ ℂ \right\}$$

**Definition (Basis):**
A **basis** is a linearly independent set that spans V.

**Definition (Dimension):**
The **dimension** of V is the number of vectors in any basis.

---

### 5. The Superposition Principle

The vector space structure directly encodes the **superposition principle**:

$$\boxed{|ψ⟩ = \sum_n c_n |n⟩}$$

If |n⟩ are possible states, then any linear combination is also a valid state. This is quantum mechanics' most distinctive feature.

**Classical Analogy (and its failure):**
- Classical: A particle is at x=3 OR x=5
- Quantum: A particle can be in a superposition of x=3 AND x=5

---

### 6. From Abstract to Concrete: Wave Functions

The abstract state |ψ⟩ connects to the wave function ψ(x) through:

$$ψ(x) = ⟨x|ψ⟩$$

where |x⟩ are the (generalized) position eigenstates.

**Key insight:** ψ(x) is just one *representation* of |ψ⟩. The same state has:
- Position representation: ψ(x) = ⟨x|ψ⟩
- Momentum representation: φ(p) = ⟨p|ψ⟩
- Energy representation: cₙ = ⟨Eₙ|ψ⟩

The abstract |ψ⟩ is representation-independent!

---

### 7. The Quantum State Postulate

**Postulate 1 (State Space):**

> *The state of a quantum system is completely described by a vector |ψ⟩ in a complex Hilbert space ℋ.*

This is the first of the fundamental postulates. We'll develop the full Hilbert space structure (with inner product) tomorrow.

---

## Physical Interpretation

### Why Vectors?

The vector space structure captures three essential quantum features:

1. **Superposition:** Linear combinations are valid states
2. **Interference:** Complex amplitudes interfere
3. **Probability:** |amplitude|² gives probabilities

### The Two-Slit Experiment

Consider a particle passing through two slits:

- State after slit 1: |ψ₁⟩
- State after slit 2: |ψ₂⟩
- Superposition: |ψ⟩ = (|ψ₁⟩ + |ψ₂⟩)/√2

The interference pattern emerges from the superposition, impossible to explain classically.

---

## Worked Examples

### Example 1: Verifying Vector Space Axioms

**Problem:** Show that ℂ² forms a vector space.

**Solution:**

Let |ψ⟩ = (a, b)ᵀ and |φ⟩ = (c, d)ᵀ where a, b, c, d ∈ ℂ.

1. **Closure under addition:**
   $$|ψ⟩ + |φ⟩ = \begin{pmatrix} a+c \\ b+d \end{pmatrix} ∈ ℂ^2 \quad ✓$$

2. **Closure under scalar multiplication:**
   $$α|ψ⟩ = \begin{pmatrix} αa \\ αb \end{pmatrix} ∈ ℂ^2 \quad ✓$$

3. **Zero vector:**
   $$|0⟩ = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

4. **Additive inverse:**
   $$-|ψ⟩ = \begin{pmatrix} -a \\ -b \end{pmatrix}$$

All other axioms follow from properties of complex number arithmetic. ∎

---

### Example 2: Linear Independence

**Problem:** Are the vectors |v₁⟩ = (1, i)ᵀ and |v₂⟩ = (i, -1)ᵀ linearly independent in ℂ²?

**Solution:**

Suppose α|v₁⟩ + β|v₂⟩ = |0⟩:

$$α\begin{pmatrix} 1 \\ i \end{pmatrix} + β\begin{pmatrix} i \\ -1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives:
- α + iβ = 0
- iα - β = 0

From the first equation: α = -iβ

Substituting into the second: i(-iβ) - β = β - β = 0 ✓

This is satisfied for any β! So there exist non-trivial solutions (e.g., β = 1, α = -i).

**Conclusion:** The vectors are **linearly dependent**.

Indeed: |v₂⟩ = i|v₁⟩ ∎

---

### Example 3: Superposition State

**Problem:** A qubit is prepared in state |ψ⟩ = (1/√2)|0⟩ + (i/√2)|1⟩. Express this as a column vector.

**Solution:**

Using the computational basis:
$$|0⟩ = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1⟩ = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

The state is:
$$|ψ⟩ = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 0 \end{pmatrix} + \frac{i}{\sqrt{2}}\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}$$

Verify normalization: |1/√2|² + |i/√2|² = 1/2 + 1/2 = 1 ✓ ∎

---

## Practice Problems

### Level 1: Direct Application

1. Verify that the set of 2×2 complex matrices forms a vector space under matrix addition and scalar multiplication.

2. Show that |+⟩ = (|0⟩ + |1⟩)/√2 and |-⟩ = (|0⟩ - |1⟩)/√2 are linearly independent.

3. Express |0⟩ and |1⟩ in terms of |+⟩ and |-⟩.

### Level 2: Intermediate

4. Prove that if |v₁⟩, |v₂⟩, |v₃⟩ are linearly independent, and |v₄⟩ = α|v₁⟩ + β|v₂⟩, then |v₁⟩, |v₂⟩, |v₃⟩, |v₄⟩ are linearly dependent.

5. Show that the functions {1, x, x², x³, ...} are linearly independent in the space of polynomials.

6. Find a basis for the space of 2×2 Hermitian matrices.

### Level 3: Challenging

7. Prove that an n-dimensional vector space cannot have more than n linearly independent vectors.

8. Show that L²[0,1] is infinite-dimensional by constructing an infinite linearly independent set.

9. **Research:** Why does quantum mechanics require a *complex* vector space? Could we use real numbers only? (Hint: Consider time evolution)

---

## Computational Lab

### Objective
Implement basic vector space operations and verify axioms numerically.

```python
"""
Day 337 Computational Lab: Complex Vector Spaces
Quantum Mechanics Core - Year 1
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Basic Vector Space Operations in ℂ²
# =============================================================================

print("=" * 60)
print("Part 1: Vector Space Operations in ℂ²")
print("=" * 60)

# Define basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

print("\nComputational basis states:")
print(f"|0⟩ = {ket_0.flatten()}")
print(f"|1⟩ = {ket_1.flatten()}")

# Create superposition
psi = (ket_0 + 1j * ket_1) / np.sqrt(2)
print(f"\n|ψ⟩ = (|0⟩ + i|1⟩)/√2 = {psi.flatten()}")

# Verify normalization
norm_sq = np.vdot(psi, psi)  # vdot handles complex conjugation
print(f"⟨ψ|ψ⟩ = {norm_sq.real:.6f} (should be 1.0)")

# =============================================================================
# Part 2: Linear Independence Check
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Linear Independence Check")
print("=" * 60)

def check_linear_independence(vectors):
    """
    Check if a set of vectors is linearly independent.
    Uses the rank of the matrix formed by column vectors.
    """
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    n_vectors = len(vectors)

    print(f"\nMatrix formed by {n_vectors} vectors:")
    print(matrix)
    print(f"Rank: {rank}")
    print(f"Linearly independent: {rank == n_vectors}")

    return rank == n_vectors

# Test 1: Standard basis (should be independent)
v1 = np.array([1, 0], dtype=complex)
v2 = np.array([0, 1], dtype=complex)
print("\nTest 1: Standard basis |0⟩, |1⟩")
check_linear_independence([v1, v2])

# Test 2: Dependent vectors
v3 = np.array([1, 1j], dtype=complex)
v4 = np.array([1j, -1], dtype=complex)  # v4 = i * v3
print("\nTest 2: |v₁⟩ = (1, i) and |v₂⟩ = (i, -1)")
check_linear_independence([v3, v4])

# Verify: v4 = i * v3?
print(f"\ni × v₃ = {1j * v3}")
print(f"v₄ = {v4}")
print(f"Equal: {np.allclose(1j * v3, v4)}")

# =============================================================================
# Part 3: Change of Basis
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Change of Basis (Computational ↔ Hadamard)")
print("=" * 60)

# Hadamard basis
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

print("\nHadamard basis:")
print(f"|+⟩ = {ket_plus.flatten()}")
print(f"|-⟩ = {ket_minus.flatten()}")

# Express |0⟩ in Hadamard basis
# |0⟩ = (|+⟩ + |-⟩)/√2
coeff_plus = np.vdot(ket_plus, ket_0)
coeff_minus = np.vdot(ket_minus, ket_0)

print(f"\n|0⟩ in Hadamard basis:")
print(f"⟨+|0⟩ = {coeff_plus:.4f}")
print(f"⟨-|0⟩ = {coeff_minus:.4f}")
print(f"|0⟩ = {coeff_plus:.4f}|+⟩ + {coeff_minus:.4f}|-⟩")

# Verify reconstruction
reconstructed = coeff_plus * ket_plus + coeff_minus * ket_minus
print(f"\nReconstructed |0⟩: {reconstructed.flatten()}")
print(f"Original |0⟩: {ket_0.flatten()}")
print(f"Match: {np.allclose(reconstructed, ket_0)}")

# =============================================================================
# Part 4: Visualization - Superposition in ℂ²
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Visualizing Quantum States")
print("=" * 60)

def plot_state_amplitudes(state, title="Quantum State"):
    """Visualize the complex amplitudes of a quantum state."""
    n = len(state)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Magnitude (probability amplitude)
    ax1 = axes[0]
    magnitudes = np.abs(state.flatten())
    ax1.bar(range(n), magnitudes**2, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Basis State Index')
    ax1.set_ylabel('Probability |cₙ|²')
    ax1.set_title(f'{title}: Probabilities')
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([f'|{i}⟩' for i in range(n)])

    # Complex amplitudes in polar form
    ax2 = axes[1]
    for i, c in enumerate(state.flatten()):
        mag = np.abs(c)
        phase = np.angle(c)
        ax2.arrow(0, 0, mag * np.cos(phase), mag * np.sin(phase),
                  head_width=0.05, head_length=0.03, fc=f'C{i}', ec=f'C{i}',
                  label=f'c_{i} = {c:.3f}')

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.set_xlabel('Real')
    ax2.set_ylabel('Imaginary')
    ax2.set_title(f'{title}: Complex Amplitudes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('day_337_quantum_state.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved as 'day_337_quantum_state.png'")

# Visualize our superposition state
plot_state_amplitudes(psi, title="|ψ⟩ = (|0⟩ + i|1⟩)/√2")

# =============================================================================
# Part 5: Higher-Dimensional Example
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Three-Level System (Qutrit)")
print("=" * 60)

# Create a qutrit state
ket_qutrit = np.array([[1], [1j], [-1]], dtype=complex) / np.sqrt(3)
print(f"\nQutrit state |ψ⟩ = (|0⟩ + i|1⟩ - |2⟩)/√3")
print(f"|ψ⟩ = {ket_qutrit.flatten()}")

# Verify normalization
norm_qutrit = np.vdot(ket_qutrit, ket_qutrit)
print(f"⟨ψ|ψ⟩ = {norm_qutrit.real:.6f}")

# Probabilities
probs = np.abs(ket_qutrit.flatten())**2
print(f"\nProbabilities:")
for i, p in enumerate(probs):
    print(f"  P(|{i}⟩) = |c_{i}|² = {p:.4f}")
print(f"  Sum = {np.sum(probs):.4f}")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Vector space | V with + and · satisfying 8 axioms |
| Linear combination | α₁\|v₁⟩ + α₂\|v₂⟩ + ⋯ |
| Linear independence | Σαᵢ\|vᵢ⟩ = 0 ⟹ all αᵢ = 0 |
| Basis expansion | \|ψ⟩ = Σₙ cₙ\|n⟩ |
| Wave function | ψ(x) = ⟨x\|ψ⟩ |

### Main Takeaways

1. **Quantum states are vectors** in a complex vector space
2. **Complex numbers are essential** for quantum interference
3. **Superposition** is encoded in the vector space structure
4. **Abstract formalism** unifies different representations
5. **Finite dimensions** → matrix mechanics; **infinite dimensions** → wave mechanics

---

## Daily Checklist

- [ ] Read Shankar Chapter 1.1-1.3
- [ ] Verify vector space axioms for ℂ²
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and understand the computational lab
- [ ] Write a one-paragraph summary of why QM needs complex vector spaces

---

## Preview: Day 338

Tomorrow we introduce **Dirac notation**—the elegant bra-ket formalism that revolutionized quantum mechanics. You'll learn to write ⟨φ|ψ⟩, |ψ⟩⟨φ|, and ⟨φ|Â|ψ⟩ with fluency.

---

*"The superposition principle is the central mystery of quantum mechanics."*
— Richard Feynman

---

**Next:** [Day_338_Tuesday.md](Day_338_Tuesday.md) — Dirac Notation
