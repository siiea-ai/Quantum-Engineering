# Day 351: The Commutator — [A,B] = AB - BA

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Commutator Algebra |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 351, you will be able to:

1. Define the commutator and explain its physical significance
2. Calculate commutators of various operator combinations
3. Apply commutator algebraic identities (linearity, product rule)
4. Prove and use the Jacobi identity
5. Connect commutators to simultaneous measurability
6. Compute commutators involving position, momentum, and angular momentum

---

## Core Content

### 1. Motivation: Why Do Operators Not Commute?

In classical mechanics, observables are functions that always commute under multiplication:

$$f \cdot g = g \cdot f$$

In quantum mechanics, observables become operators, and operators generally do **not** commute:

$$\hat{A}\hat{B} \neq \hat{B}\hat{A}$$

**Why does this matter?** The non-commutativity of operators is the mathematical origin of:
- The uncertainty principle
- Incompatible observables
- The fundamentally probabilistic nature of quantum mechanics

---

### 2. Definition of the Commutator

The **commutator** of two operators Â and B̂ is defined as:

$$\boxed{[\hat{A}, \hat{B}] \equiv \hat{A}\hat{B} - \hat{B}\hat{A}}$$

**Key observations:**

1. The commutator is itself an operator
2. If [Â, B̂] = 0, we say Â and B̂ **commute**
3. If [Â, B̂] ≠ 0, we say Â and B̂ are **incompatible** or **non-commuting**

**Simple example:** Consider matrices:

$$A = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}, \quad B = \begin{pmatrix} 1 & 0 \\ 3 & 1 \end{pmatrix}$$

$$AB = \begin{pmatrix} 7 & 2 \\ 3 & 1 \end{pmatrix}, \quad BA = \begin{pmatrix} 1 & 2 \\ 3 & 7 \end{pmatrix}$$

$$[A, B] = AB - BA = \begin{pmatrix} 6 & 0 \\ 0 & -6 \end{pmatrix} \neq 0$$

---

### 3. Fundamental Algebraic Properties

The commutator satisfies several important identities.

#### Property 1: Antisymmetry

$$\boxed{[\hat{A}, \hat{B}] = -[\hat{B}, \hat{A}]}$$

**Proof:**
$$[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A} = -(\hat{B}\hat{A} - \hat{A}\hat{B}) = -[\hat{B}, \hat{A}]$$

**Corollary:** $[\hat{A}, \hat{A}] = 0$ for any operator Â.

#### Property 2: Linearity

$$\boxed{[\hat{A}, \alpha\hat{B} + \beta\hat{C}] = \alpha[\hat{A}, \hat{B}] + \beta[\hat{A}, \hat{C}]}$$

$$\boxed{[\alpha\hat{A} + \beta\hat{B}, \hat{C}] = \alpha[\hat{A}, \hat{C}] + \beta[\hat{B}, \hat{C}]}$$

**Proof of first identity:**
$$[\hat{A}, \alpha\hat{B} + \beta\hat{C}] = \hat{A}(\alpha\hat{B} + \beta\hat{C}) - (\alpha\hat{B} + \beta\hat{C})\hat{A}$$
$$= \alpha(\hat{A}\hat{B} - \hat{B}\hat{A}) + \beta(\hat{A}\hat{C} - \hat{C}\hat{A})$$
$$= \alpha[\hat{A}, \hat{B}] + \beta[\hat{A}, \hat{C}]$$

#### Property 3: Product Rule (Leibniz Rule)

$$\boxed{[\hat{A}, \hat{B}\hat{C}] = [\hat{A}, \hat{B}]\hat{C} + \hat{B}[\hat{A}, \hat{C}]}$$

$$\boxed{[\hat{A}\hat{B}, \hat{C}] = \hat{A}[\hat{B}, \hat{C}] + [\hat{A}, \hat{C}]\hat{B}}$$

**Proof of first identity:**
$$[\hat{A}, \hat{B}\hat{C}] = \hat{A}\hat{B}\hat{C} - \hat{B}\hat{C}\hat{A}$$

Add and subtract $\hat{B}\hat{A}\hat{C}$:
$$= \hat{A}\hat{B}\hat{C} - \hat{B}\hat{A}\hat{C} + \hat{B}\hat{A}\hat{C} - \hat{B}\hat{C}\hat{A}$$
$$= (\hat{A}\hat{B} - \hat{B}\hat{A})\hat{C} + \hat{B}(\hat{A}\hat{C} - \hat{C}\hat{A})$$
$$= [\hat{A}, \hat{B}]\hat{C} + \hat{B}[\hat{A}, \hat{C}]$$

---

### 4. The Jacobi Identity

The **Jacobi identity** is a fundamental constraint on commutators:

$$\boxed{[\hat{A}, [\hat{B}, \hat{C}]] + [\hat{B}, [\hat{C}, \hat{A}]] + [\hat{C}, [\hat{A}, \hat{B}]] = 0}$$

This can be remembered as a **cyclic sum** over all three operators.

**Proof:**

Expand each term:
$$[\hat{A}, [\hat{B}, \hat{C}]] = \hat{A}\hat{B}\hat{C} - \hat{A}\hat{C}\hat{B} - \hat{B}\hat{C}\hat{A} + \hat{C}\hat{B}\hat{A}$$

$$[\hat{B}, [\hat{C}, \hat{A}]] = \hat{B}\hat{C}\hat{A} - \hat{B}\hat{A}\hat{C} - \hat{C}\hat{A}\hat{B} + \hat{A}\hat{C}\hat{B}$$

$$[\hat{C}, [\hat{A}, \hat{B}]] = \hat{C}\hat{A}\hat{B} - \hat{C}\hat{B}\hat{A} - \hat{A}\hat{B}\hat{C} + \hat{B}\hat{A}\hat{C}$$

Adding all terms: every term appears once with + and once with -, giving **zero**.

**Physical significance:** The Jacobi identity is essential for:
- Consistency of quantum mechanics
- Lie algebra structure of observables
- Angular momentum algebra

---

### 5. Commutators with Functions of Operators

For a function f(Â) defined by its power series:

$$[\hat{B}, f(\hat{A})] = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!}[\hat{B}, \hat{A}^n]$$

**Important special case:** If [Â, B̂] = c (a constant), then:

$$\boxed{[\hat{A}, \hat{B}^n] = nc\hat{B}^{n-1}}$$

$$\boxed{[\hat{A}^n, \hat{B}] = nc\hat{A}^{n-1}}$$

**Proof by induction for [Â, B̂ⁿ]:**

Base case (n=1): [Â, B̂] = c ✓

Inductive step: Assume [Â, B̂ⁿ] = ncB̂ⁿ⁻¹

Using the product rule:
$$[\hat{A}, \hat{B}^{n+1}] = [\hat{A}, \hat{B}^n]\hat{B} + \hat{B}^n[\hat{A}, \hat{B}]$$
$$= nc\hat{B}^{n-1}\hat{B} + \hat{B}^n \cdot c$$
$$= nc\hat{B}^n + c\hat{B}^n = (n+1)c\hat{B}^n$$ ✓

---

### 6. Physical Interpretation: Simultaneous Measurability

**Theorem:** Two observables Â and B̂ can be simultaneously measured with arbitrary precision if and only if they commute:

$$[\hat{A}, \hat{B}] = 0 \iff \text{Â and B̂ share a complete set of simultaneous eigenstates}$$

**Proof (⟹):**

If [Â, B̂] = 0 and |a⟩ is an eigenstate of Â with eigenvalue a:
$$\hat{A}|a⟩ = a|a⟩$$

Consider Â(B̂|a⟩):
$$\hat{A}(\hat{B}|a⟩) = \hat{B}(\hat{A}|a⟩) = \hat{B}(a|a⟩) = a(\hat{B}|a⟩)$$

So B̂|a⟩ is also an eigenstate of Â with eigenvalue a.

If a is non-degenerate: B̂|a⟩ ∝ |a⟩, so |a⟩ is an eigenstate of B̂ too.

If a is degenerate: Can choose a basis of the eigenspace that diagonalizes B̂.

**Example: Compatible observables**
- x̂ and ŷ (different position components): [x̂, ŷ] = 0
- Ĥ and L̂² (for central potentials): [Ĥ, L̂²] = 0
- L̂² and L̂z: [L̂², L̂z] = 0

**Example: Incompatible observables**
- x̂ and p̂x: [x̂, p̂x] = iℏ ≠ 0
- L̂x and L̂y: [L̂x, L̂y] = iℏL̂z ≠ 0

---

### 7. The Anticommutator

Alongside the commutator, we define the **anticommutator**:

$$\boxed{\{\hat{A}, \hat{B}\} \equiv \hat{A}\hat{B} + \hat{B}\hat{A}}$$

**Properties:**
- Symmetric: {Â, B̂} = {B̂, Â}
- {Â, Â} = 2Â²

**Product decomposition:**
$$\hat{A}\hat{B} = \frac{1}{2}[\hat{A}, \hat{B}] + \frac{1}{2}\{\hat{A}, \hat{B}\}$$

The anticommutator appears in:
- Fermionic systems (anticommutation relations)
- Uncertainty principle derivation
- Jordan-Wigner transformation

---

### 8. Important Commutator Examples

#### Position and Momentum

$$\boxed{[\hat{x}, \hat{p}] = i\hbar}$$

(We will derive this tomorrow.)

#### Powers of Position and Momentum

$$[\hat{x}^n, \hat{p}] = i\hbar n\hat{x}^{n-1}$$
$$[\hat{x}, \hat{p}^n] = i\hbar n\hat{p}^{n-1}$$

#### Angular Momentum

$$\boxed{[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k}$$

Explicitly:
$$[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$$
$$[\hat{L}_y, \hat{L}_z] = i\hbar\hat{L}_x$$
$$[\hat{L}_z, \hat{L}_x] = i\hbar\hat{L}_y$$

#### L² Commutators

$$[\hat{L}^2, \hat{L}_x] = [\hat{L}^2, \hat{L}_y] = [\hat{L}^2, \hat{L}_z] = 0$$

This is why we can label states by both l and m.

#### Pauli Matrices

$$[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k$$

---

## Physical Interpretation

### The Commutator as a Measure of Incompatibility

The magnitude of the commutator quantifies how "incompatible" two observables are:

1. **[Â, B̂] = 0:** Perfect compatibility. Can measure both simultaneously.

2. **[Â, B̂] = iℏ (constant):** Maximum incompatibility for conjugate variables. The uncertainty product has a fundamental lower bound.

3. **[Â, B̂] = iℏĈ (operator):** State-dependent incompatibility. The uncertainty relation depends on the state.

### Connection to Classical Mechanics

The commutator is the quantum analog of the **Poisson bracket**:

$$[\hat{A}, \hat{B}] \leftrightarrow i\hbar\{A, B\}_{PB}$$

where the Poisson bracket is:
$$\{A, B\}_{PB} = \frac{\partial A}{\partial q}\frac{\partial B}{\partial p} - \frac{\partial A}{\partial p}\frac{\partial B}{\partial q}$$

**Example:** For position and momentum:
$$\{x, p\}_{PB} = 1 \quad \Rightarrow \quad [\hat{x}, \hat{p}] = i\hbar$$

---

## Worked Examples

### Example 1: Commutator of x and p²

**Problem:** Calculate [x̂, p̂²].

**Solution:**

Using the product rule with Â = x̂, B̂ = p̂, Ĉ = p̂:

$$[\hat{x}, \hat{p}^2] = [\hat{x}, \hat{p}\hat{p}] = [\hat{x}, \hat{p}]\hat{p} + \hat{p}[\hat{x}, \hat{p}]$$

Since [x̂, p̂] = iℏ:
$$= i\hbar\hat{p} + \hat{p}(i\hbar) = i\hbar\hat{p} + i\hbar\hat{p}$$

$$\boxed{[\hat{x}, \hat{p}^2] = 2i\hbar\hat{p}}$$

**Verification using the power rule:** [x̂, p̂ⁿ] = iℏnp̂ⁿ⁻¹

For n = 2: [x̂, p̂²] = iℏ(2)p̂¹ = 2iℏp̂ ✓

---

### Example 2: Commutator Involving Angular Momentum

**Problem:** Calculate [L̂x, L̂²] where L̂² = L̂x² + L̂y² + L̂z².

**Solution:**

$$[\hat{L}_x, \hat{L}^2] = [\hat{L}_x, \hat{L}_x^2 + \hat{L}_y^2 + \hat{L}_z^2]$$

By linearity:
$$= [\hat{L}_x, \hat{L}_x^2] + [\hat{L}_x, \hat{L}_y^2] + [\hat{L}_x, \hat{L}_z^2]$$

**First term:** [L̂x, L̂x²] = 0 (anything commutes with functions of itself)

**Second term:** Using the product rule:
$$[\hat{L}_x, \hat{L}_y^2] = [\hat{L}_x, \hat{L}_y]\hat{L}_y + \hat{L}_y[\hat{L}_x, \hat{L}_y]$$
$$= i\hbar\hat{L}_z\hat{L}_y + \hat{L}_y(i\hbar\hat{L}_z) = i\hbar(\hat{L}_z\hat{L}_y + \hat{L}_y\hat{L}_z)$$

**Third term:**
$$[\hat{L}_x, \hat{L}_z^2] = [\hat{L}_x, \hat{L}_z]\hat{L}_z + \hat{L}_z[\hat{L}_x, \hat{L}_z]$$

Since [L̂x, L̂z] = -iℏL̂y:
$$= -i\hbar\hat{L}_y\hat{L}_z - i\hbar\hat{L}_z\hat{L}_y = -i\hbar(\hat{L}_y\hat{L}_z + \hat{L}_z\hat{L}_y)$$

**Sum:**
$$[\hat{L}_x, \hat{L}^2] = 0 + i\hbar(\hat{L}_z\hat{L}_y + \hat{L}_y\hat{L}_z) - i\hbar(\hat{L}_y\hat{L}_z + \hat{L}_z\hat{L}_y) = 0$$

$$\boxed{[\hat{L}_x, \hat{L}^2] = 0}$$

This proves L̂x and L̂² can be simultaneously diagonalized.

---

### Example 3: Commutator from Representations

**Problem:** Verify [x̂, p̂] = iℏ using the position representation where p̂ = -iℏ(d/dx).

**Solution:**

Let [x̂, p̂] act on an arbitrary function ψ(x):

$$[\hat{x}, \hat{p}]\psi(x) = \hat{x}\hat{p}\psi - \hat{p}\hat{x}\psi$$

**First term:**
$$\hat{x}\hat{p}\psi = x \cdot \left(-i\hbar\frac{d\psi}{dx}\right) = -i\hbar x\frac{d\psi}{dx}$$

**Second term:**
$$\hat{p}\hat{x}\psi = -i\hbar\frac{d}{dx}(x\psi) = -i\hbar\left(\psi + x\frac{d\psi}{dx}\right)$$

**Difference:**
$$[\hat{x}, \hat{p}]\psi = -i\hbar x\frac{d\psi}{dx} - \left(-i\hbar\psi - i\hbar x\frac{d\psi}{dx}\right)$$
$$= -i\hbar x\frac{d\psi}{dx} + i\hbar\psi + i\hbar x\frac{d\psi}{dx}$$
$$= i\hbar\psi$$

Since this holds for all ψ:
$$\boxed{[\hat{x}, \hat{p}] = i\hbar}$$

---

## Practice Problems

### Level 1: Direct Application

1. **Basic commutators:** Calculate:
   (a) [x̂², p̂]
   (b) [x̂, p̂³]
   (c) [x̂², p̂²]

2. **Verify antisymmetry:** Show that [Â, B̂] = -[B̂, Â] for Â = σx and B̂ = σy.

3. **Product rule practice:** Given [Â, B̂] = Ĉ, calculate [Â, B̂²] and [Â², B̂].

### Level 2: Intermediate

4. **Angular momentum:** Calculate [L̂y, L̂z²] using the angular momentum commutation relations.

5. **Hamiltonian commutators:** For Ĥ = p̂²/(2m) + V(x̂), calculate:
   (a) [Ĥ, x̂]
   (b) [Ĥ, p̂]

6. **Nested commutators:** Evaluate [L̂x, [L̂y, L̂z]] and verify the Jacobi identity for angular momentum.

### Level 3: Challenging

7. **Exponential operators:** If [Â, B̂] = c (constant), show that:
   $$e^{\hat{A}}\hat{B}e^{-\hat{A}} = \hat{B} + c$$
   (Hint: Consider f(λ) = e^{λÂ}B̂e^{-λÂ} and compute df/dλ.)

8. **Baker-Campbell-Hausdorff:** If [Â, B̂] = c and [[Â, B̂], Â] = [[Â, B̂], B̂] = 0, prove:
   $$e^{\hat{A}}e^{\hat{B}} = e^{\hat{A} + \hat{B} + \frac{1}{2}[\hat{A}, \hat{B}]}$$

9. **Creation and annihilation:** Define â = (x̂ + ip̂/mω)/√(2ℏ/mω) and ↠= (x̂ - ip̂/mω)/√(2ℏ/mω).
   (a) Calculate [â, â†].
   (b) Express x̂ and p̂ in terms of â and â†.
   (c) Calculate [â, â†â] (the number operator commutator).

---

## Computational Lab

### Objective
Implement commutator algebra for matrix operators and verify key identities.

```python
"""
Day 351 Computational Lab: Commutator Algebra
Quantum Mechanics Core - Year 1, Week 51
"""

import numpy as np
from typing import Tuple, Callable
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Basic Commutator Operations
# =============================================================================

print("=" * 70)
print("Part 1: Commutator Algebra with Matrices")
print("=" * 70)

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculate the commutator [A, B] = AB - BA.

    Parameters:
        A, B: Square matrices (numpy arrays)

    Returns:
        Commutator [A, B]
    """
    return A @ B - B @ A

def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculate the anticommutator {A, B} = AB + BA.
    """
    return A @ B + B @ A

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

print("\nPauli matrices:")
print(f"σx =\n{sigma_x}")
print(f"\nσy =\n{sigma_y}")
print(f"\nσz =\n{sigma_z}")

# Calculate Pauli commutators
comm_xy = commutator(sigma_x, sigma_y)
comm_yz = commutator(sigma_y, sigma_z)
comm_zx = commutator(sigma_z, sigma_x)

print("\nPauli matrix commutators:")
print(f"[σx, σy] = \n{comm_xy}")
print(f"Expected: 2i·σz = \n{2j * sigma_z}")
print(f"Match: {np.allclose(comm_xy, 2j * sigma_z)}")

print(f"\n[σy, σz] = \n{comm_yz}")
print(f"Expected: 2i·σx = \n{2j * sigma_x}")
print(f"Match: {np.allclose(comm_yz, 2j * sigma_x)}")

print(f"\n[σz, σx] = \n{comm_zx}")
print(f"Expected: 2i·σy = \n{2j * sigma_y}")
print(f"Match: {np.allclose(comm_zx, 2j * sigma_y)}")

# =============================================================================
# Part 2: Verify Commutator Properties
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Verifying Commutator Properties")
print("=" * 70)

# Property 1: Antisymmetry
print("\n--- Antisymmetry: [A, B] = -[B, A] ---")
comm_AB = commutator(sigma_x, sigma_y)
comm_BA = commutator(sigma_y, sigma_x)
print(f"[σx, σy] = \n{comm_AB}")
print(f"-[σy, σx] = \n{-comm_BA}")
print(f"Antisymmetry verified: {np.allclose(comm_AB, -comm_BA)}")

# Property 2: Linearity
print("\n--- Linearity: [A, αB + βC] = α[A,B] + β[A,C] ---")
alpha, beta = 2, 3
lhs = commutator(sigma_x, alpha * sigma_y + beta * sigma_z)
rhs = alpha * commutator(sigma_x, sigma_y) + beta * commutator(sigma_x, sigma_z)
print(f"LHS: [σx, 2σy + 3σz] =\n{lhs}")
print(f"RHS: 2[σx, σy] + 3[σx, σz] =\n{rhs}")
print(f"Linearity verified: {np.allclose(lhs, rhs)}")

# Property 3: Product Rule (Leibniz)
print("\n--- Product Rule: [A, BC] = [A,B]C + B[A,C] ---")
A, B, C = sigma_x, sigma_y, sigma_z
lhs = commutator(A, B @ C)
rhs = commutator(A, B) @ C + B @ commutator(A, C)
print(f"LHS: [σx, σy·σz] =\n{lhs}")
print(f"RHS: [σx, σy]σz + σy[σx, σz] =\n{rhs}")
print(f"Product rule verified: {np.allclose(lhs, rhs)}")

# =============================================================================
# Part 3: Jacobi Identity
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Jacobi Identity")
print("=" * 70)

def jacobi_sum(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Calculate the Jacobi sum [A,[B,C]] + [B,[C,A]] + [C,[A,B]]."""
    term1 = commutator(A, commutator(B, C))
    term2 = commutator(B, commutator(C, A))
    term3 = commutator(C, commutator(A, B))
    return term1 + term2 + term3

jacobi = jacobi_sum(sigma_x, sigma_y, sigma_z)
print(f"\nJacobi sum for Pauli matrices:")
print(f"[σx, [σy, σz]] + [σy, [σz, σx]] + [σz, [σx, σy]] =")
print(jacobi)
print(f"\nIs zero matrix: {np.allclose(jacobi, np.zeros((2, 2)))}")

# Test with random matrices
np.random.seed(42)
A_rand = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
B_rand = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
C_rand = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)

jacobi_rand = jacobi_sum(A_rand, B_rand, C_rand)
print(f"\nJacobi identity for random 3×3 matrices:")
print(f"Max absolute value: {np.max(np.abs(jacobi_rand)):.2e}")
print(f"Is zero (to machine precision): {np.allclose(jacobi_rand, np.zeros((3, 3)))}")

# =============================================================================
# Part 4: Angular Momentum Algebra
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Angular Momentum Commutators (Spin-1)")
print("=" * 70)

# Spin-1 matrices (3×3)
Lx_s1 = (1/np.sqrt(2)) * np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=complex)

Ly_s1 = (1/np.sqrt(2)) * np.array([
    [0, -1j, 0],
    [1j, 0, -1j],
    [0, 1j, 0]
], dtype=complex)

Lz_s1 = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, -1]
], dtype=complex)

# Note: These are in units of ℏ
hbar = 1  # Natural units

print("Spin-1 matrices (in units of ℏ):")
print(f"Lx =\n{Lx_s1}")
print(f"\nLy =\n{Ly_s1}")
print(f"\nLz =\n{Lz_s1}")

# Verify [Lx, Ly] = i*hbar*Lz
comm_LxLy = commutator(Lx_s1, Ly_s1)
print(f"\n[Lx, Ly] =\n{comm_LxLy}")
print(f"i·ℏ·Lz =\n{1j * hbar * Lz_s1}")
print(f"Match: {np.allclose(comm_LxLy, 1j * hbar * Lz_s1)}")

# Verify [L², Lz] = 0
L_squared = Lx_s1 @ Lx_s1 + Ly_s1 @ Ly_s1 + Lz_s1 @ Lz_s1
comm_L2Lz = commutator(L_squared, Lz_s1)
print(f"\nL² =\n{L_squared}")
print(f"\n[L², Lz] =\n{comm_L2Lz}")
print(f"Is zero: {np.allclose(comm_L2Lz, np.zeros((3, 3)))}")

# =============================================================================
# Part 5: Position and Momentum (Finite Difference Approximation)
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Position-Momentum Commutator (Discrete Approximation)")
print("=" * 70)

def create_position_momentum_matrices(N: int, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create finite-difference approximations for position and momentum operators.

    Parameters:
        N: Number of grid points
        dx: Grid spacing

    Returns:
        X: Position operator (diagonal)
        P: Momentum operator (central difference)
    """
    # Position operator: diagonal with x values
    x_values = np.arange(-(N-1)/2, (N+1)/2) * dx
    X = np.diag(x_values)

    # Momentum operator: central difference (with periodic boundaries)
    P = np.zeros((N, N), dtype=complex)
    for i in range(N):
        P[i, (i+1) % N] = 1
        P[i, (i-1) % N] = -1
    P = -1j / (2 * dx) * P  # -iℏ d/dx with ℏ=1

    return X.astype(complex), P

# Create operators
N = 20
dx = 0.5
X, P = create_position_momentum_matrices(N, dx)

# Calculate [X, P]
comm_XP = commutator(X, P)

print(f"Grid: N = {N}, dx = {dx}")
print(f"\nDiagonal elements of [X, P]:")
diagonal = np.diag(comm_XP)
print(f"First 5: {diagonal[:5]}")
print(f"All approximately equal: {np.allclose(diagonal, diagonal[0])}")
print(f"Value: {diagonal[0]:.4f}")
print(f"Expected (iℏ = i): {1j:.4f}")
print(f"Relative error: {np.abs(diagonal[0] - 1j) / np.abs(1j):.2%}")

# =============================================================================
# Part 6: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Visualization of Commutator Structure")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot Pauli commutators
paulis = [sigma_x, sigma_y, sigma_z]
labels = ['σx', 'σy', 'σz']

for i, (ax, label) in enumerate(zip(axes[0], labels)):
    # Calculate [σi, σ_next]
    j = (i + 1) % 3
    comm = commutator(paulis[i], paulis[j])

    im = ax.imshow(np.abs(comm), cmap='Blues')
    ax.set_title(f'|[{labels[i]}, {labels[j]}]|')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    plt.colorbar(im, ax=ax)

# Plot spin-1 angular momentum commutators
spins = [Lx_s1, Ly_s1, Lz_s1]
spin_labels = ['Lx', 'Ly', 'Lz']

for i, (ax, label) in enumerate(zip(axes[1], spin_labels)):
    j = (i + 1) % 3
    comm = commutator(spins[i], spins[j])

    im = ax.imshow(np.abs(comm), cmap='Reds')
    ax.set_title(f'|[{spin_labels[i]}, {spin_labels[j]}]| (Spin-1)')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('day_351_commutators.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_351_commutators.png'")

# =============================================================================
# Part 7: Commutator Power Series Identity
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Baker-Campbell-Hausdorff Formula Verification")
print("=" * 70)

def matrix_exp(A: np.ndarray, terms: int = 50) -> np.ndarray:
    """Calculate matrix exponential via Taylor series."""
    result = np.eye(A.shape[0], dtype=complex)
    term = np.eye(A.shape[0], dtype=complex)
    for n in range(1, terms):
        term = term @ A / n
        result += term
    return result

# For Pauli matrices: if [A,B] = c and [c,A]=[c,B]=0
# Then e^A e^B = e^(A+B+[A,B]/2)

# Scale down to avoid numerical issues
epsilon = 0.1
A = epsilon * sigma_x
B = epsilon * sigma_y

# Calculate both sides
lhs = matrix_exp(A) @ matrix_exp(B)
comm_AB = commutator(A, B)
rhs = matrix_exp(A + B + comm_AB / 2)

print(f"\nBCH formula test with A = ε·σx, B = ε·σy (ε = {epsilon})")
print(f"\ne^A · e^B =\n{lhs}")
print(f"\ne^(A + B + [A,B]/2) =\n{rhs}")
print(f"\nMatch: {np.allclose(lhs, rhs)}")
print(f"Max difference: {np.max(np.abs(lhs - rhs)):.2e}")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Commutator definition | [Â, B̂] = ÂB̂ - B̂Â |
| Anticommutator | {Â, B̂} = ÂB̂ + B̂Â |
| Antisymmetry | [Â, B̂] = -[B̂, Â] |
| Linearity | [Â, αB̂ + βĈ] = α[Â, B̂] + β[Â, Ĉ] |
| Product rule | [Â, B̂Ĉ] = [Â, B̂]Ĉ + B̂[Â, Ĉ] |
| Jacobi identity | [Â, [B̂, Ĉ]] + [B̂, [Ĉ, Â]] + [Ĉ, [Â, B̂]] = 0 |
| Power rule | [Â, B̂ⁿ] = nc B̂ⁿ⁻¹ if [Â, B̂] = c |

### Main Takeaways

1. **Non-commutativity is fundamental** — Operators in quantum mechanics generally do not commute, unlike classical observables
2. **Commutators quantify incompatibility** — [Â, B̂] = 0 means simultaneous measurement is possible
3. **Algebraic structure** — Commutators satisfy linearity, product rules, and the Jacobi identity
4. **Connection to physics** — The commutator [x̂, p̂] = iℏ encodes the uncertainty principle
5. **Lie algebra** — Angular momentum commutators define the SO(3) Lie algebra

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.2 (The Postulates)
- [ ] Read Sakurai Chapter 1.4 (Measurements and Compatible Observables)
- [ ] Derive [x̂ⁿ, p̂] = iℏnx̂ⁿ⁻¹ using the product rule
- [ ] Verify [L̂x, L̂y] = iℏL̂z using explicit matrix representations
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the computational lab
- [ ] Prove the Jacobi identity for a general case

---

## Preview: Day 352

Tomorrow we derive the fundamental commutation relation [x̂, p̂] = iℏ from first principles. We'll see how this emerges from the Poisson bracket structure of classical mechanics and understand why position and momentum are "canonically conjugate" variables. This single equation contains the essence of quantum mechanics.

---

*"The commutator is to quantum mechanics what the Poisson bracket is to classical mechanics—it is the fundamental algebraic structure that encodes dynamics."*

---

**Next:** [Day_352_Tuesday.md](Day_352_Tuesday.md) — Canonical Commutation Relation
