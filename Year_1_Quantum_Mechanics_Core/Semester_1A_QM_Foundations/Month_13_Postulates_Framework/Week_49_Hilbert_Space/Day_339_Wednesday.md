# Day 339: Operators in Hilbert Space — The Mathematics of Physical Observables

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Linear Operators and Their Properties |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 339, you will be able to:

1. Define linear operators and verify linearity
2. Compute the action of operators on quantum states: Â|ψ⟩
3. Construct matrix representations of operators in a given basis
4. Transform operator matrices under change of basis
5. Define and evaluate functions of operators
6. Perform operator algebra: addition, multiplication, and preview commutators

---

## Core Content

### 1. Linear Operators: Definition and Properties

An **operator** Â is a mapping from one vector space to another (often the same space):

$$\hat{A}: \mathcal{H} \to \mathcal{H}$$

In quantum mechanics, operators act on state vectors to produce new state vectors.

**Definition (Linear Operator):**
An operator Â is **linear** if for all |ψ⟩, |φ⟩ ∈ ℋ and α, β ∈ ℂ:

$$\boxed{\hat{A}(\alpha|ψ⟩ + \beta|φ⟩) = \alpha\hat{A}|ψ⟩ + \beta\hat{A}|φ⟩}$$

This is the defining property: operators distribute over linear combinations and commute with scalar multiplication.

**Examples of Linear Operators:**

| Operator | Action | Physical Meaning |
|----------|--------|------------------|
| Identity Î | Î\|ψ⟩ = \|ψ⟩ | Does nothing |
| Zero operator 0̂ | 0̂\|ψ⟩ = \|0⟩ | Annihilates all states |
| Projection P̂ | P̂\|ψ⟩ = \|φ⟩⟨φ\|ψ⟩ | Projects onto subspace |
| Position x̂ | x̂\|ψ⟩ = x\|ψ⟩ (eigenstate) | Position observable |
| Momentum p̂ | p̂ψ(x) = -iℏ∂ψ/∂x | Momentum observable |

---

### 2. Operator Action on States: Â|ψ⟩

When an operator acts on a ket, it produces a new ket:

$$\hat{A}|ψ⟩ = |φ⟩$$

**Interpretation:** The operator transforms the quantum state. This can represent:
- Time evolution (Û = e^{-iĤt/ℏ})
- Measurement back-action
- Symmetry transformations
- Quantum gates

**Computing Operator Action:**

Given |ψ⟩ in some basis {|n⟩}:
$$|ψ⟩ = \sum_n c_n|n⟩$$

The transformed state is:
$$\hat{A}|ψ⟩ = \sum_n c_n \hat{A}|n⟩$$

If we know how Â acts on each basis vector, we know how it acts on any vector.

---

### 3. Matrix Representation of Operators

The power of Dirac notation becomes clear when representing operators as matrices.

**Definition (Matrix Elements):**
In an orthonormal basis {|n⟩}, the matrix elements of Â are:

$$\boxed{A_{mn} = ⟨m|\hat{A}|n⟩}$$

This forms the (m,n) entry of the matrix representation.

**The Matrix:**

$$[\hat{A}] = \begin{pmatrix} ⟨1|\hat{A}|1⟩ & ⟨1|\hat{A}|2⟩ & \cdots \\ ⟨2|\hat{A}|1⟩ & ⟨2|\hat{A}|2⟩ & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

**Operator-State Correspondence:**

If |ψ⟩ has coefficients $c_n = ⟨n|ψ⟩$ in basis {|n⟩}, and Â|ψ⟩ = |φ⟩ has coefficients $d_m = ⟨m|φ⟩$, then:

$$d_m = \sum_n A_{mn} c_n$$

In matrix form:
$$\begin{pmatrix} d_1 \\ d_2 \\ \vdots \end{pmatrix} = \begin{pmatrix} A_{11} & A_{12} & \cdots \\ A_{21} & A_{22} & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix} \begin{pmatrix} c_1 \\ c_2 \\ \vdots \end{pmatrix}$$

**Proof using Completeness:**

$$d_m = ⟨m|\hat{A}|ψ⟩ = ⟨m|\hat{A}\left(\sum_n |n⟩⟨n|\right)|ψ⟩ = \sum_n ⟨m|\hat{A}|n⟩⟨n|ψ⟩ = \sum_n A_{mn}c_n$$

---

### 4. Operator Expansion in Outer Products

Any operator can be expanded in outer products:

$$\boxed{\hat{A} = \sum_{m,n} A_{mn}|m⟩⟨n| = \sum_{m,n} ⟨m|\hat{A}|n⟩ |m⟩⟨n|}$$

**Verification:** Acting on an arbitrary basis state |k⟩:
$$\hat{A}|k⟩ = \sum_{m,n} A_{mn}|m⟩⟨n|k⟩ = \sum_{m,n} A_{mn}|m⟩δ_{nk} = \sum_m A_{mk}|m⟩$$

This gives the coefficients of Â|k⟩, confirming the expansion.

**Important Special Cases:**

1. **Identity operator:**
   $$\hat{I} = \sum_n |n⟩⟨n|$$

2. **Projection operator onto state |ψ⟩:**
   $$\hat{P}_ψ = |ψ⟩⟨ψ|$$

3. **Pauli-X (bit flip):**
   $$\hat{σ}_x = |0⟩⟨1| + |1⟩⟨0|$$

---

### 5. Change of Basis for Operators

When switching from basis {|n⟩} to basis {|n'⟩}, operator matrix elements transform.

**The Unitary Transformation:**

Let Ŝ be the unitary operator that transforms bases:
$$|n'⟩ = \hat{S}|n⟩$$

The matrix elements of Ŝ are:
$$S_{mn} = ⟨m|n'⟩$$

**Transformation of Operator Matrices:**

If A is the matrix of Â in basis {|n⟩} and A' is the matrix in basis {|n'⟩}:

$$\boxed{A' = S^† A S}$$

**Derivation:**
$$A'_{m'n'} = ⟨m'|\hat{A}|n'⟩$$

Insert completeness twice:
$$= \sum_{k,l} ⟨m'|k⟩⟨k|\hat{A}|l⟩⟨l|n'⟩ = \sum_{k,l} S^*_{km'} A_{kl} S_{ln'}$$

In matrix notation: $A' = S^† A S$.

**Physical Interpretation:**
The same operator has different matrix representations in different bases. The eigenvalues (physical predictions) remain unchanged—only the representation changes.

**Example: Pauli-Z in Different Bases**

In computational basis {|0⟩, |1⟩}:
$$[\hat{σ}_z]_{comp} = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

The Hadamard matrix transforms to the |+⟩, |-⟩ basis:
$$S = H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Then:
$$[\hat{σ}_z]_{+-} = H^†[\hat{σ}_z]_{comp}H = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = [\hat{σ}_x]_{comp}$$

This confirms that σ̂_z in the Hadamard basis looks like σ̂_x in the computational basis!

---

### 6. Functions of Operators

Given a scalar function f(x), we can define f(Â) for operators.

**Definition via Spectral Decomposition:**

If Â has eigenvalues {aᵢ} and eigenstates {|aᵢ⟩}:

$$\hat{A} = \sum_i a_i |a_i⟩⟨a_i|$$

Then the function of the operator is:

$$\boxed{f(\hat{A}) = \sum_i f(a_i) |a_i⟩⟨a_i|}$$

**Key Examples:**

1. **Powers:**
   $$\hat{A}^n = \sum_i a_i^n |a_i⟩⟨a_i|$$

2. **Inverse (if all aᵢ ≠ 0):**
   $$\hat{A}^{-1} = \sum_i \frac{1}{a_i} |a_i⟩⟨a_i|$$

3. **Exponential:**
   $$e^{\hat{A}} = \sum_i e^{a_i} |a_i⟩⟨a_i|$$

4. **Square root (if all aᵢ ≥ 0):**
   $$\sqrt{\hat{A}} = \sum_i \sqrt{a_i} |a_i⟩⟨a_i|$$

**The Exponential Operator:**

Particularly important in quantum mechanics for time evolution:

$$\hat{U}(t) = e^{-i\hat{H}t/ℏ}$$

The exponential can also be defined via Taylor series:

$$e^{\hat{A}} = \sum_{n=0}^{\infty} \frac{\hat{A}^n}{n!} = \hat{I} + \hat{A} + \frac{\hat{A}^2}{2!} + \frac{\hat{A}^3}{3!} + \cdots$$

---

### 7. Operator Algebra

Operators form an algebra under addition, scalar multiplication, and composition (multiplication).

**Addition:**
$$(\hat{A} + \hat{B})|ψ⟩ = \hat{A}|ψ⟩ + \hat{B}|ψ⟩$$

Matrix elements add:
$$(A + B)_{mn} = A_{mn} + B_{mn}$$

**Scalar Multiplication:**
$$(α\hat{A})|ψ⟩ = α(\hat{A}|ψ⟩)$$

**Operator Multiplication (Composition):**
$$(\hat{A}\hat{B})|ψ⟩ = \hat{A}(\hat{B}|ψ⟩)$$

First apply B̂, then apply Â. Matrix elements follow matrix multiplication:
$$(AB)_{mk} = \sum_n A_{mn}B_{nk}$$

**Critical Point: Non-Commutativity**

Unlike numbers, operator multiplication is generally **non-commutative**:

$$\boxed{\hat{A}\hat{B} \neq \hat{B}\hat{A} \quad \text{in general}}$$

This is the mathematical origin of quantum uncertainty!

---

### 8. The Commutator: Preview

The **commutator** quantifies the failure to commute:

$$\boxed{[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}}$$

**Properties:**
- [Â, B̂] = -[B̂, Â] (antisymmetry)
- [Â, Â] = 0
- [Â, B̂ + Ĉ] = [Â, B̂] + [Â, Ĉ] (linearity)
- [Â, B̂Ĉ] = [Â, B̂]Ĉ + B̂[Â, Ĉ] (product rule)

**The Fundamental Commutator:**
$$[\hat{x}, \hat{p}] = iℏ\hat{I}$$

This is the cornerstone of quantum mechanics, leading to the uncertainty principle. We'll explore this deeply in coming days.

---

## Quantum Computing Connection: Operators as Quantum Gates

In quantum computing, operators are **quantum gates** that transform qubit states.

### Single-Qubit Gates as 2×2 Unitary Matrices

| Gate | Matrix | Action |
|------|--------|--------|
| X (NOT) | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | Bit flip: \|0⟩ ↔ \|1⟩ |
| Z | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Phase flip: \|1⟩ → -\|1⟩ |
| H | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | Creates superposition |
| S | $\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ | π/2 phase |
| T | $\begin{pmatrix} 1 & 0 \\ 0 & e^{iπ/4} \end{pmatrix}$ | π/4 phase |

### Gate Composition = Operator Multiplication

Sequential gates multiply in reverse order:
$$|ψ_{final}⟩ = \hat{U}_n \cdots \hat{U}_2 \hat{U}_1 |ψ_{initial}⟩$$

The circuit reads left-to-right, but operator multiplication reads right-to-left.

### Non-Commutativity in Circuits

Gate order matters! In general:
$$\hat{H}\hat{Z} \neq \hat{Z}\hat{H}$$

**Compute:**
$$HZ = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$

$$ZH = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$$

Different results! The commutator [H, Z] ≠ 0.

### Universal Gate Sets

A set of gates is **universal** if any unitary can be approximated by compositions from the set.

**Example universal sets:**
- {H, T, CNOT}
- {H, Toffoli}
- {CNOT, all single-qubit gates}

This connects to the operator algebra: any operator in SU(2ⁿ) can be built from products of these generators.

---

## Worked Examples

### Example 1: Matrix Representation of an Operator

**Problem:** Find the matrix representation of the raising operator σ̂₊ = (σ̂ₓ + iσ̂ᵧ)/2 in the computational basis.

**Solution:**

First, recall the Pauli matrices:
$$\hat{σ}_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \hat{σ}_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

Compute σ̂₊:
$$\hat{σ}_+ = \frac{1}{2}\left[\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} + i\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}\right]$$

$$= \frac{1}{2}\begin{pmatrix} 0 & 1+1 \\ 1-1 & 0 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 0 & 2 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$$

**Verify the action:**
$$\hat{σ}_+|0⟩ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} = 0$$

$$\hat{σ}_+|1⟩ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} = |0⟩$$

The raising operator raises |1⟩ to |0⟩ (in spin language: |↓⟩ → |↑⟩) and annihilates the top state |0⟩. ∎

---

### Example 2: Change of Basis

**Problem:** The operator Â has matrix $A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$ in the computational basis. Find its matrix in the basis |+⟩ = (|0⟩+|1⟩)/√2, |-⟩ = (|0⟩-|1⟩)/√2.

**Solution:**

The transformation matrix from computational to ±-basis is the Hadamard:
$$S = H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Note: S is its own inverse (H² = I), so S† = S.

The transformed matrix is:
$$A' = S^† A S = H A H$$

Compute step by step:
$$AH = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 3 & -1 \\ 3 & 1 \end{pmatrix}$$

$$A' = HAH = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 3 & -1 \\ 3 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 6 & 0 \\ 0 & -2 \end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & -1 \end{pmatrix}$$

**Result:** In the ±-basis, the matrix is diagonal!

$$A' = \begin{pmatrix} 3 & 0 \\ 0 & -1 \end{pmatrix}$$

This reveals that |+⟩ and |-⟩ are the eigenstates of Â with eigenvalues 3 and -1 respectively. ∎

---

### Example 3: Function of an Operator

**Problem:** Compute e^{iπσ̂z/2} where σ̂z = |0⟩⟨0| - |1⟩⟨1|.

**Solution:**

**Method 1: Spectral Decomposition**

The eigenvalues of σ̂z are +1 and -1 with eigenstates |0⟩ and |1⟩:
$$\hat{σ}_z = (+1)|0⟩⟨0| + (-1)|1⟩⟨1|$$

Using f(Â) = Σᵢ f(aᵢ)|aᵢ⟩⟨aᵢ|:
$$e^{iπ\hat{σ}_z/2} = e^{iπ(+1)/2}|0⟩⟨0| + e^{iπ(-1)/2}|1⟩⟨1|$$

$$= e^{iπ/2}|0⟩⟨0| + e^{-iπ/2}|1⟩⟨1| = i|0⟩⟨0| + (-i)|1⟩⟨1|$$

**Matrix form:**
$$e^{iπ\hat{σ}_z/2} = \begin{pmatrix} i & 0 \\ 0 & -i \end{pmatrix} = i\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = i\hat{σ}_z$$

**Method 2: Taylor Series (verification)**

Using σ̂z² = I:
$$e^{iθ\hat{σ}_z} = \cos(θ)\hat{I} + i\sin(θ)\hat{σ}_z$$

With θ = π/2:
$$e^{iπ\hat{σ}_z/2} = \cos(π/2)\hat{I} + i\sin(π/2)\hat{σ}_z = 0 \cdot \hat{I} + i \cdot 1 \cdot \hat{σ}_z = i\hat{σ}_z$$ ✓

**Physical interpretation:** This is the S-gate up to a global phase! ∎

---

## Practice Problems

### Level 1: Direct Application

1. Show that the operator Â|ψ⟩ = 3|ψ⟩ (scalar multiplication) is linear.

2. Compute the matrix representation of σ̂₋ = (σ̂ₓ - iσ̂ᵧ)/2 in the computational basis.

3. Verify that σ̂ₓσ̂ᵧ ≠ σ̂ᵧσ̂ₓ by explicit matrix multiplication.

**Answers:**
1. Â(α|ψ⟩ + β|φ⟩) = 3(α|ψ⟩ + β|φ⟩) = 3α|ψ⟩ + 3β|φ⟩ = αÂ|ψ⟩ + βÂ|φ⟩ ✓
2. σ̂₋ = $\begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$
3. σ̂ₓσ̂ᵧ = iσ̂z, σ̂ᵧσ̂ₓ = -iσ̂z

### Level 2: Intermediate

4. An operator Â satisfies Â|0⟩ = |0⟩ + |1⟩ and Â|1⟩ = |0⟩ - |1⟩. Find the matrix representation of Â. Is Â unitary?

5. Prove that if Ŝ is unitary and Â' = Ŝ†ÂŜ, then Tr(Â') = Tr(Â).

6. Show that [Â, B̂]† = [B̂†, Â†].

**Hints:**
4. Build the matrix column by column; check if A†A = I
5. Use the cyclic property of trace
6. Expand using (ÂB̂)† = B̂†Â†

### Level 3: Challenging

7. Prove that for any Hermitian operator Ĥ, e^{iĤ} is unitary.

8. The operator Â has matrix $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. Compute sin(Â) using spectral decomposition.

9. **Research Connection:** Operators that almost commute: If [Â, B̂] = cÎ for some scalar c, show that e^Â e^B̂ = e^{c/2} e^{Â+B̂}.
   (Hint: This is the Baker-Campbell-Hausdorff formula for a special case.)

---

## Computational Lab

### Objective
Implement operator algebra and basis transformations in Python.

```python
"""
Day 339 Computational Lab: Operators in Hilbert Space
Quantum Mechanics Core - Year 1
"""

import numpy as np
from scipy.linalg import expm, logm, sqrtm
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Pauli Operators and Their Algebra
# =============================================================================

print("=" * 60)
print("Part 1: Pauli Operator Algebra")
print("=" * 60)

# Define Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

print("\nPauli matrices:")
print(f"σₓ =\n{sigma_x}")
print(f"σᵧ =\n{sigma_y}")
print(f"σz =\n{sigma_z}")

# Verify Pauli algebra: σᵢσⱼ = δᵢⱼI + iεᵢⱼₖσₖ
print("\n--- Pauli Algebra Verification ---")
xy = sigma_x @ sigma_y
yx = sigma_y @ sigma_x
print(f"σₓσᵧ =\n{xy}")
print(f"σᵧσₓ =\n{yx}")
print(f"σₓσᵧ = iσz? {np.allclose(xy, 1j * sigma_z)}")
print(f"σᵧσₓ = -iσz? {np.allclose(yx, -1j * sigma_z)}")

# Commutator
commutator_xy = sigma_x @ sigma_y - sigma_y @ sigma_x
print(f"\n[σₓ, σᵧ] = σₓσᵧ - σᵧσₓ =\n{commutator_xy}")
print(f"= 2iσz? {np.allclose(commutator_xy, 2j * sigma_z)}")

# =============================================================================
# Part 2: Raising and Lowering Operators
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Raising and Lowering Operators")
print("=" * 60)

sigma_plus = (sigma_x + 1j * sigma_y) / 2
sigma_minus = (sigma_x - 1j * sigma_y) / 2

print(f"\nσ₊ = (σₓ + iσᵧ)/2 =\n{sigma_plus}")
print(f"σ₋ = (σₓ - iσᵧ)/2 =\n{sigma_minus}")

# Define basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Action on basis states
print("\nAction on basis states:")
print(f"σ₊|0⟩ = {(sigma_plus @ ket_0).flatten()}")
print(f"σ₊|1⟩ = {(sigma_plus @ ket_1).flatten()}")
print(f"σ₋|0⟩ = {(sigma_minus @ ket_0).flatten()}")
print(f"σ₋|1⟩ = {(sigma_minus @ ket_1).flatten()}")

# Verify σ₊σ₋ + σ₋σ₊ = I/2 (anticommutator)
anticomm = sigma_plus @ sigma_minus + sigma_minus @ sigma_plus
print(f"\n{{σ₊, σ₋}} = σ₊σ₋ + σ₋σ₊ =\n{anticomm}")
print(f"= I/2? {np.allclose(anticomm, I/2)}")

# =============================================================================
# Part 3: Change of Basis
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Change of Basis")
print("=" * 60)

# Hadamard transforms computational to ±-basis
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

print("\nHadamard matrix H:")
print(H)

# Transform σz to ±-basis
sigma_z_pm = H.conj().T @ sigma_z @ H
print(f"\nσz in ±-basis (H†σzH):\n{np.round(sigma_z_pm, 10)}")
print(f"Equals σₓ? {np.allclose(sigma_z_pm, sigma_x)}")

# Transform σx to ±-basis
sigma_x_pm = H.conj().T @ sigma_x @ H
print(f"\nσₓ in ±-basis (H†σₓH):\n{np.round(sigma_x_pm, 10)}")
print(f"Equals σz? {np.allclose(sigma_x_pm, sigma_z)}")

# =============================================================================
# Part 4: Functions of Operators
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Functions of Operators (Exponentials)")
print("=" * 60)

# Compute e^{iπσz/2}
theta = np.pi / 2
exp_sigma_z = expm(1j * theta * sigma_z)
print(f"\nexp(iπσz/2) =\n{np.round(exp_sigma_z, 10)}")

# Compare with formula: e^{iθσz} = cos(θ)I + i sin(θ)σz
formula_result = np.cos(theta) * I + 1j * np.sin(theta) * sigma_z
print(f"\nUsing formula cos(θ)I + i sin(θ)σz:\n{np.round(formula_result, 10)}")
print(f"Match: {np.allclose(exp_sigma_z, formula_result)}")

# Compute e^{iπσx/4} (T-gate equivalent on different axis)
theta_x = np.pi / 4
exp_sigma_x = expm(1j * theta_x * sigma_x)
print(f"\nexp(iπσₓ/4) =\n{np.round(exp_sigma_x, 6)}")

# =============================================================================
# Part 5: Operator Products and Non-Commutativity
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Non-Commutativity of Quantum Gates")
print("=" * 60)

# Define common gates
X = sigma_x
Z = sigma_z
Y = sigma_y
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# HZ vs ZH
HZ = H @ Z
ZH = Z @ H

print("\nH·Z (Hadamard then Z):")
print(np.round(HZ, 6))
print("\nZ·H (Z then Hadamard):")
print(np.round(ZH, 6))
print(f"\nHZ = ZH? {np.allclose(HZ, ZH)}")

# Commutator [H, Z]
comm_HZ = H @ Z - Z @ H
print(f"\n[H, Z] = HZ - ZH =\n{np.round(comm_HZ, 6)}")

# =============================================================================
# Part 6: Matrix Representation from Operator Action
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Building Operators from Their Action")
print("=" * 60)

def build_operator_matrix(action_function, dim=2):
    """
    Build the matrix representation of an operator from its action.
    action_function takes a basis state index and returns the output vector.
    """
    basis = [np.zeros((dim, 1), dtype=complex) for _ in range(dim)]
    for i in range(dim):
        basis[i][i, 0] = 1

    columns = [action_function(basis[i]) for i in range(dim)]
    return np.hstack(columns)

# Example: Define an operator by its action
def my_operator_action(ket):
    """Custom operator: A|0⟩ = |0⟩ + |1⟩, A|1⟩ = |0⟩ - |1⟩"""
    if np.allclose(ket, ket_0):
        return ket_0 + ket_1
    elif np.allclose(ket, ket_1):
        return ket_0 - ket_1
    else:
        # Linear extension
        c0 = (ket_0.conj().T @ ket)[0, 0]
        c1 = (ket_1.conj().T @ ket)[0, 0]
        return c0 * (ket_0 + ket_1) + c1 * (ket_0 - ket_1)

# Build the matrix
A_custom = np.hstack([my_operator_action(ket_0), my_operator_action(ket_1)])
print("\nCustom operator A where A|0⟩ = |0⟩+|1⟩, A|1⟩ = |0⟩-|1⟩:")
print(A_custom)

# Check if unitary
is_unitary = np.allclose(A_custom @ A_custom.conj().T, 2 * I)
print(f"\nA†A = {A_custom.conj().T @ A_custom}")
print(f"Is A unitary? {np.allclose(A_custom.conj().T @ A_custom, I)}")
print(f"A = √2 × H? {np.allclose(A_custom, np.sqrt(2) * H)}")

# =============================================================================
# Part 7: Spectral Decomposition
# =============================================================================

print("\n" + "=" * 60)
print("Part 7: Spectral Decomposition")
print("=" * 60)

# Diagonalize a Hermitian operator
A_hermitian = np.array([[2, 1-1j], [1+1j, 3]], dtype=complex)
print("\nHermitian operator A:")
print(A_hermitian)

eigenvalues, eigenvectors = np.linalg.eigh(A_hermitian)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

# Verify spectral decomposition: A = Σ λᵢ |vᵢ⟩⟨vᵢ|
reconstructed = np.zeros_like(A_hermitian)
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i:i+1]  # Column vector
    outer = v @ v.conj().T
    reconstructed += eigenvalues[i] * outer
    print(f"\nλ_{i} = {eigenvalues[i]:.4f}")
    print(f"|v_{i}⟩⟨v_{i}| =\n{outer}")

print(f"\nReconstructed A:\n{reconstructed}")
print(f"Match original: {np.allclose(reconstructed, A_hermitian)}")

# =============================================================================
# Part 8: Visualization - Gate Actions on Bloch Sphere
# =============================================================================

print("\n" + "=" * 60)
print("Part 8: Visualizing Operator Actions")
print("=" * 60)

def state_to_bloch(psi):
    """Convert a qubit state to Bloch sphere coordinates."""
    # Normalize
    psi = psi / np.linalg.norm(psi)
    # Extract components
    alpha = psi[0, 0]
    beta = psi[1, 0]
    # Bloch coordinates
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2
    return np.array([x, y, z])

# Initial state
psi_initial = np.array([[1], [0]], dtype=complex)
print(f"\nInitial state |ψ₀⟩ = |0⟩")
print(f"Bloch coordinates: {state_to_bloch(psi_initial)}")

# Apply sequence of gates
gates = [('H', H), ('T', T), ('H', H)]
psi = psi_initial.copy()

for name, gate in gates:
    psi = gate @ psi
    bloch = state_to_bloch(psi)
    print(f"After {name}: Bloch = [{bloch[0]:.3f}, {bloch[1]:.3f}, {bloch[2]:.3f}]")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Linearity | Â(α\|ψ⟩ + β\|φ⟩) = αÂ\|ψ⟩ + βÂ\|φ⟩ |
| Matrix elements | A_{mn} = ⟨m\|Â\|n⟩ |
| Operator expansion | Â = Σ_{m,n} A_{mn}\|m⟩⟨n\| |
| Change of basis | A' = S†AS |
| Function of operator | f(Â) = Σᵢ f(aᵢ)\|aᵢ⟩⟨aᵢ\| |
| Operator product | (AB)_{mk} = Σₙ A_{mn}B_{nk} |
| Commutator | [Â, B̂] = ÂB̂ - B̂Â |
| Exponential | e^Â = Σₙ Âⁿ/n! |

### Main Takeaways

1. **Operators are linear maps** on Hilbert space—they transform quantum states
2. **Matrix representation** depends on basis choice; eigenvalues are basis-independent
3. **Change of basis** follows A' = S†AS where S is the unitary transformation matrix
4. **Functions of operators** are defined via spectral decomposition
5. **Non-commutativity** is the mathematical origin of quantum uncertainty
6. **Quantum gates** are unitary operators—gate composition is operator multiplication

---

## References

**Primary Reading:**
- Shankar, *Principles of Quantum Mechanics*, Chapter 1.6-1.7
- Sakurai, *Modern Quantum Mechanics*, Chapter 1.3

**Supplementary:**
- Nielsen & Chuang, *Quantum Computation and Quantum Information*, Chapter 2.1
- Preskill Lecture Notes, Chapter 2

---

## Daily Checklist

- [ ] Read Shankar 1.6-1.7 on linear operators
- [ ] Verify matrix element formula using completeness relation
- [ ] Practice computing change of basis transformations
- [ ] Work through all three examples
- [ ] Complete Level 1-2 practice problems
- [ ] Run the computational lab and experiment with different operators
- [ ] Write down the commutator [σ̂ₓ, σ̂ᵧ] = 2iσ̂z from memory

---

## Preview: Day 340

Tomorrow we study **Hermitian and Unitary Operators**—the two most important classes of operators in quantum mechanics. Hermitian operators represent observables (their eigenvalues are measurement outcomes), while unitary operators represent symmetries and time evolution. We'll prove the spectral theorem and understand why these operators are central to the quantum formalism.

---

*"The formalism of quantum mechanics is just linear algebra—but linear algebra applied to the physical world reveals profound truths about nature."*
— Based on Dirac's insights

---

**Next:** [Day_340_Thursday.md](Day_340_Thursday.md) — Hermitian and Unitary Operators
