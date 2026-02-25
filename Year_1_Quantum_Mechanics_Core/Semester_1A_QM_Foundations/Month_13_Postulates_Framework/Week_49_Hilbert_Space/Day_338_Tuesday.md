# Day 338: Dirac Notation — The Language of Quantum Mechanics

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Bras, Kets, and Brackets |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 338, you will be able to:

1. Write quantum states in Dirac bra-ket notation
2. Compute inner products using bracket notation
3. Construct outer products |ψ⟩⟨φ|
4. Compute matrix elements ⟨φ|Â|ψ⟩
5. Use the dual correspondence between bras and kets
6. Express operators in terms of outer products

---

## Core Content

### 1. The Genius of Dirac Notation

Paul Dirac introduced this notation in 1939. It revolutionized quantum mechanics by:

- Unifying Heisenberg's matrix mechanics and Schrödinger's wave mechanics
- Making the mathematics representation-independent
- Providing an elegant, intuitive symbolic system

**Dirac's insight:** Separate the "bra" ⟨ from the "ket" ⟩ in the "bracket" ⟨|⟩.

---

### 2. Kets: State Vectors

A **ket** |ψ⟩ represents a quantum state vector in Hilbert space.

$$\boxed{|ψ⟩ ∈ \mathcal{H}}$$

**Properties:**
- Kets are vectors (column vectors in matrix representation)
- Kets can be added: |ψ⟩ + |φ⟩
- Kets can be scaled: α|ψ⟩

**Examples:**

| Notation | Meaning |
|----------|---------|
| \|0⟩, \|1⟩ | Qubit basis states |
| \|↑⟩, \|↓⟩ | Spin states |
| \|n⟩ | Energy eigenstate n |
| \|x⟩ | Position eigenstate |
| \|p⟩ | Momentum eigenstate |
| \|ψ⟩ | Generic state |

---

### 3. Bras: Dual Vectors

A **bra** ⟨φ| is the dual (conjugate transpose) of a ket:

$$\boxed{⟨φ| = (|φ⟩)^†}$$

**The dual correspondence:**

$$|ψ⟩ = \begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{pmatrix} \quad \longleftrightarrow \quad ⟨ψ| = \begin{pmatrix} c_1^* & c_2^* & \cdots & c_n^* \end{pmatrix}$$

**Key rule for scalars:**

$$\boxed{α|ψ⟩ \longleftrightarrow α^*⟨ψ|}$$

The scalar comes out with complex conjugation!

---

### 4. Brackets: Inner Products

The **inner product** of |φ⟩ and |ψ⟩ is written as a bracket:

$$\boxed{⟨φ|ψ⟩ = \text{inner product} ∈ ℂ}$$

**Properties:**

1. **Conjugate symmetry:** $⟨φ|ψ⟩ = ⟨ψ|φ⟩^*$

2. **Linearity in second argument:**
   $$⟨φ|α ψ_1 + β ψ_2⟩ = α⟨φ|ψ_1⟩ + β⟨φ|ψ_2⟩$$

3. **Antilinearity in first argument:**
   $$⟨αφ_1 + βφ_2|ψ⟩ = α^*⟨φ_1|ψ⟩ + β^*⟨φ_2|ψ⟩$$

4. **Positivity:** $⟨ψ|ψ⟩ ≥ 0$, with equality iff |ψ⟩ = 0

**Normalization:**
$$⟨ψ|ψ⟩ = 1 \quad \text{(normalized state)}$$

**Orthogonality:**
$$⟨φ|ψ⟩ = 0 \quad \text{(orthogonal states)}$$

**Orthonormality:**
$$⟨m|n⟩ = δ_{mn} \quad \text{(orthonormal basis)}$$

---

### 5. Outer Products: Operators

The **outer product** |ψ⟩⟨φ| creates an operator:

$$\boxed{|ψ⟩⟨φ| = \text{operator}}$$

**Action on a state:**
$$(|ψ⟩⟨φ|)|χ⟩ = |ψ⟩⟨φ|χ⟩ = ⟨φ|χ⟩ · |ψ⟩$$

The operator projects onto |ψ⟩ with amplitude ⟨φ|χ⟩.

**Projection operator:**
$$\boxed{P̂_ψ = |ψ⟩⟨ψ| \quad \text{(for normalized |ψ⟩)}}$$

Properties: $P̂_ψ^2 = P̂_ψ$, $P̂_ψ^† = P̂_ψ$

---

### 6. The Completeness Relation

For an orthonormal basis {|n⟩}, the **completeness relation** is:

$$\boxed{\hat{I} = \sum_n |n⟩⟨n|}$$

This is the resolution of the identity—profoundly useful!

**Inserting identity:**
$$|ψ⟩ = \hat{I}|ψ⟩ = \sum_n |n⟩⟨n|ψ⟩ = \sum_n c_n|n⟩$$

where $c_n = ⟨n|ψ⟩$ are the expansion coefficients.

**For continuous bases:**
$$\hat{I} = \int |x⟩⟨x| \, dx = \int |p⟩⟨p| \, dp$$

---

### 7. Matrix Elements

The **matrix element** of operator Â between states |φ⟩ and |ψ⟩:

$$\boxed{⟨φ|\hat{A}|ψ⟩ = A_{φψ}}$$

**Matrix representation:**
In basis {|n⟩}, the operator Â has matrix elements:
$$A_{mn} = ⟨m|\hat{A}|n⟩$$

**Operator expansion:**
$$\hat{A} = \sum_{m,n} A_{mn}|m⟩⟨n| = \sum_{m,n} ⟨m|\hat{A}|n⟩ |m⟩⟨n|$$

---

### 8. Adjoint in Dirac Notation

For operator Â, the adjoint Â† satisfies:

$$\boxed{⟨φ|\hat{A}^†|ψ⟩ = ⟨ψ|\hat{A}|φ⟩^*}$$

**Useful identities:**
- $(|ψ⟩⟨φ|)^† = |φ⟩⟨ψ|$
- $(\hat{A}\hat{B})^† = \hat{B}^†\hat{A}^†$
- $(α\hat{A})^† = α^*\hat{A}^†$

---

## Physical Interpretation

### Probability Amplitude

The inner product ⟨φ|ψ⟩ is a **probability amplitude**:

$$P(\text{find } |φ⟩ \text{ given } |ψ⟩) = |⟨φ|ψ⟩|^2$$

### Transition Amplitude

For time evolution from |ψᵢ⟩ to |ψ_f⟩:

$$\text{Transition amplitude} = ⟨ψ_f|ψ_i(t)⟩$$

### Expectation Value

$$⟨\hat{A}⟩ = ⟨ψ|\hat{A}|ψ⟩$$

---

## Worked Examples

### Example 1: Inner Product Calculation

**Problem:** Compute ⟨φ|ψ⟩ where |ψ⟩ = (1+i)|0⟩ + 2|1⟩ and |φ⟩ = |0⟩ - i|1⟩.

**Solution:**

First, write the bra:
$$⟨φ| = ⟨0| - (-i)⟨1| = ⟨0| + i⟨1|$$

Note: The coefficient -i becomes +i (complex conjugate)!

Now compute:
$$⟨φ|ψ⟩ = (⟨0| + i⟨1|)((1+i)|0⟩ + 2|1⟩)$$
$$= (1+i)⟨0|0⟩ + 2⟨0|1⟩ + i(1+i)⟨1|0⟩ + 2i⟨1|1⟩$$
$$= (1+i)(1) + 2(0) + i(1+i)(0) + 2i(1)$$
$$= 1 + i + 2i = 1 + 3i$$

**Answer:** ⟨φ|ψ⟩ = 1 + 3i ∎

---

### Example 2: Outer Product as Operator

**Problem:** Given |+⟩ = (|0⟩ + |1⟩)/√2, find the matrix representation of P̂₊ = |+⟩⟨+|.

**Solution:**

First, express in computational basis:
$$|+⟩ = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

The outer product is:
$$P̂_+ = |+⟩⟨+| = \frac{1}{2}\begin{pmatrix} 1 \\ 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**Verify it's a projector:**
$$P̂_+^2 = \frac{1}{4}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \frac{1}{4}\begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = P̂_+$$ ✓ ∎

---

### Example 3: Completeness Relation

**Problem:** Use the completeness relation to show |ψ⟩ = ⟨0|ψ⟩|0⟩ + ⟨1|ψ⟩|1⟩.

**Solution:**

The completeness relation for the computational basis is:
$$\hat{I} = |0⟩⟨0| + |1⟩⟨1|$$

Acting on |ψ⟩:
$$|ψ⟩ = \hat{I}|ψ⟩ = (|0⟩⟨0| + |1⟩⟨1|)|ψ⟩ = |0⟩⟨0|ψ⟩ + |1⟩⟨1|ψ⟩$$

This is exactly |ψ⟩ = c₀|0⟩ + c₁|1⟩ where cₙ = ⟨n|ψ⟩. ∎

---

## Practice Problems

### Level 1: Direct Application

1. If |ψ⟩ = 3|0⟩ - 4i|1⟩, write down ⟨ψ|.

2. Compute ⟨0|+⟩ and ⟨1|+⟩ where |+⟩ = (|0⟩ + |1⟩)/√2.

3. Verify that ⟨+|-⟩ = 0 where |-⟩ = (|0⟩ - |1⟩)/√2.

### Level 2: Intermediate

4. Show that (|ψ⟩⟨φ|)† = |φ⟩⟨ψ|.

5. If P̂ = |ψ⟩⟨ψ| is a projector, prove P̂² = P̂ and P̂† = P̂.

6. Express the Pauli-X operator σ̂ₓ in terms of outer products.

### Level 3: Challenging

7. Prove that Tr(|ψ⟩⟨φ|) = ⟨φ|ψ⟩.

8. Show that {|+⟩, |-⟩} and {|0⟩, |1⟩} are related by a unitary transformation. Find the matrix.

9. Derive the representation ⟨x|p⟩ = (2πℏ)^{-1/2} e^{ipx/ℏ} using the definition p̂ = -iℏ d/dx.

---

## Computational Lab

```python
"""
Day 338 Computational Lab: Dirac Notation Operations
"""

import numpy as np

print("=" * 60)
print("Day 338: Dirac Notation in Practice")
print("=" * 60)

# =============================================================================
# Part 1: Kets and Bras
# =============================================================================

print("\n--- Part 1: Kets and Bras ---")

# Define ket |ψ⟩ = (1+i)|0⟩ + 2|1⟩
psi_ket = np.array([[1 + 1j], [2]], dtype=complex)
print(f"|ψ⟩ = {psi_ket.flatten()}")

# The bra is the conjugate transpose
psi_bra = psi_ket.conj().T
print(f"⟨ψ| = {psi_bra.flatten()}")

# Define |φ⟩ = |0⟩ - i|1⟩
phi_ket = np.array([[1], [-1j]], dtype=complex)
phi_bra = phi_ket.conj().T
print(f"|φ⟩ = {phi_ket.flatten()}")
print(f"⟨φ| = {phi_bra.flatten()}")

# =============================================================================
# Part 2: Inner Products (Brackets)
# =============================================================================

print("\n--- Part 2: Inner Products ---")

# ⟨φ|ψ⟩
inner_phi_psi = phi_bra @ psi_ket
print(f"⟨φ|ψ⟩ = {inner_phi_psi[0,0]}")

# ⟨ψ|φ⟩ should be complex conjugate
inner_psi_phi = psi_bra @ phi_ket
print(f"⟨ψ|φ⟩ = {inner_psi_phi[0,0]}")
print(f"⟨φ|ψ⟩* = {inner_phi_psi[0,0].conj()}")
print(f"Conjugate symmetry verified: {np.isclose(inner_psi_phi[0,0], inner_phi_psi[0,0].conj())}")

# Normalization
norm_psi_sq = (psi_bra @ psi_ket)[0,0]
print(f"\n⟨ψ|ψ⟩ = {norm_psi_sq.real}")

# =============================================================================
# Part 3: Outer Products
# =============================================================================

print("\n--- Part 3: Outer Products ---")

# Define |+⟩ and |-⟩
plus_ket = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
minus_ket = np.array([[1], [-1]], dtype=complex) / np.sqrt(2)

# Projection operator P_+ = |+⟩⟨+|
P_plus = plus_ket @ plus_ket.conj().T
print("P̂₊ = |+⟩⟨+| =")
print(P_plus)

# Verify P² = P
P_plus_sq = P_plus @ P_plus
print("\nP̂₊² =")
print(P_plus_sq)
print(f"P² = P? {np.allclose(P_plus_sq, P_plus)}")

# =============================================================================
# Part 4: Completeness Relation
# =============================================================================

print("\n--- Part 4: Completeness Relation ---")

# Computational basis projectors
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

P_0 = ket_0 @ ket_0.conj().T
P_1 = ket_1 @ ket_1.conj().T

identity_check = P_0 + P_1
print("P̂₀ + P̂₁ =")
print(identity_check)
print(f"Equals identity: {np.allclose(identity_check, np.eye(2))}")

# Expand |+⟩ using completeness
c_0 = (ket_0.conj().T @ plus_ket)[0,0]
c_1 = (ket_1.conj().T @ plus_ket)[0,0]
print(f"\nExpanding |+⟩:")
print(f"⟨0|+⟩ = {c_0}")
print(f"⟨1|+⟩ = {c_1}")

reconstructed = c_0 * ket_0 + c_1 * ket_1
print(f"⟨0|+⟩|0⟩ + ⟨1|+⟩|1⟩ = {reconstructed.flatten()}")
print(f"Original |+⟩ = {plus_ket.flatten()}")

# =============================================================================
# Part 5: Matrix Elements
# =============================================================================

print("\n--- Part 5: Matrix Elements ---")

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Matrix elements in computational basis
print("σ̂ₓ matrix elements:")
for i, bra in enumerate([ket_0.conj().T, ket_1.conj().T]):
    for j, ket in enumerate([ket_0, ket_1]):
        element = (bra @ sigma_x @ ket)[0,0]
        print(f"  ⟨{i}|σ̂ₓ|{j}⟩ = {element}")

# Express σ̂ₓ in outer product form
sigma_x_outer = ket_0 @ ket_1.conj().T + ket_1 @ ket_0.conj().T
print("\nσ̂ₓ = |0⟩⟨1| + |1⟩⟨0| =")
print(sigma_x_outer)
print(f"Matches: {np.allclose(sigma_x_outer, sigma_x)}")

# =============================================================================
# Part 6: Expectation Values
# =============================================================================

print("\n--- Part 6: Expectation Values ---")

# Expectation of σ̂z in state |+⟩
exp_z_plus = (plus_ket.conj().T @ sigma_z @ plus_ket)[0,0]
print(f"⟨+|σ̂ᵤ|+⟩ = {exp_z_plus.real}")

# Expectation of σ̂x in state |+⟩
exp_x_plus = (plus_ket.conj().T @ sigma_x @ plus_ket)[0,0]
print(f"⟨+|σ̂ₓ|+⟩ = {exp_x_plus.real}")

# Expectation of σ̂x in state |0⟩
exp_x_0 = (ket_0.conj().T @ sigma_x @ ket_0)[0,0]
print(f"⟨0|σ̂ₓ|0⟩ = {exp_x_0.real}")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Dirac Notation Dictionary

| Symbol | Name | Type | Example |
|--------|------|------|---------|
| \|ψ⟩ | Ket | Vector | \|0⟩, \|↑⟩ |
| ⟨φ\| | Bra | Dual vector | ⟨0\|, ⟨↑\| |
| ⟨φ\|ψ⟩ | Bracket | Scalar (ℂ) | ⟨0\|1⟩ = 0 |
| \|ψ⟩⟨φ\| | Outer product | Operator | \|0⟩⟨1\| |
| ⟨φ\|Â\|ψ⟩ | Matrix element | Scalar (ℂ) | ⟨0\|σ̂ₓ\|1⟩ |

### Key Formulas

$$\boxed{⟨φ|ψ⟩ = ⟨ψ|φ⟩^*}$$

$$\boxed{\hat{I} = \sum_n |n⟩⟨n|}$$

$$\boxed{|ψ⟩ = \sum_n ⟨n|ψ⟩|n⟩}$$

---

## Daily Checklist

- [ ] Read Shankar Chapter 1.4-1.5
- [ ] Practice converting between bras and kets
- [ ] Compute inner products with complex coefficients
- [ ] Construct outer products and verify projector properties
- [ ] Complete Level 1-2 practice problems
- [ ] Run the computational lab
- [ ] Write down the completeness relation from memory

---

## Preview: Day 339

Tomorrow we study **operators in Hilbert space**—how linear operators act on quantum states, their matrix representations, and the connection between abstract operators and physical observables.

---

*"The bracket notation is not merely a shorthand—it embodies the entire structure of quantum mechanics."*

---

**Next:** [Day_339_Wednesday.md](Day_339_Wednesday.md) — Operators in Hilbert Space
