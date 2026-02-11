# Day 347: Compatible Observables and Simultaneous Measurements

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Commutators and Compatible Observables |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving and Examples |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: CSCO Implementation |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Define compatible observables in terms of the commutator condition [A, B] = 0
2. Prove that commuting operators share a complete set of common eigenstates
3. Construct and verify simultaneous eigenstates for compatible observables
4. Explain the concept of Complete Set of Commuting Observables (CSCO)
5. Identify physical examples of compatible and incompatible observables
6. Connect compatibility to the possibility of simultaneous precise measurements

---

## Required Reading

### Primary Texts
- **Shankar, Chapter 4.2**: Compatible and Incompatible Observables (pp. 164-171)
- **Sakurai, Chapter 1.4**: Compatible Observables (pp. 30-35)
- **Griffiths, Chapter 3.5**: Compatible Observables (pp. 111-115)

### Supplementary Reading
- **Cohen-Tannoudji, Chapter III.D**: Complete Sets of Commuting Observables
- **Nielsen & Chuang, Chapter 2.2.5**: Commutator and Simultaneous Measurement

---

## Core Content: Theory and Concepts

### 1. The Fundamental Question of Simultaneous Measurement

In classical physics, we can simultaneously measure any two observables with arbitrary precision. A particle can have definite position AND definite momentum at the same instant.

**Quantum mechanics is different.** The key question: When can we simultaneously measure two observables and obtain definite values for both?

**Answer:** If and only if the operators commute.

### 2. The Commutator

**Definition:** For two operators $\hat{A}$ and $\hat{B}$, the **commutator** is:

$$\boxed{[\hat{A}, \hat{B}] \equiv \hat{A}\hat{B} - \hat{B}\hat{A}}$$

**Properties of Commutators:**

1. **Antisymmetry:** $[\hat{A}, \hat{B}] = -[\hat{B}, \hat{A}]$

2. **Linearity:** $[\hat{A}, \hat{B} + \hat{C}] = [\hat{A}, \hat{B}] + [\hat{A}, \hat{C}]$

3. **Scalar multiplication:** $[\hat{A}, c\hat{B}] = c[\hat{A}, \hat{B}]$

4. **Product rule (Leibniz):** $[\hat{A}, \hat{B}\hat{C}] = [\hat{A}, \hat{B}]\hat{C} + \hat{B}[\hat{A}, \hat{C}]$

5. **Jacobi identity:** $[\hat{A}, [\hat{B}, \hat{C}]] + [\hat{B}, [\hat{C}, \hat{A}]] + [\hat{C}, [\hat{A}, \hat{B}]] = 0$

### 3. Definition of Compatible Observables

**Definition:** Two observables $A$ and $B$ are **compatible** (or **simultaneously measurable**) if their corresponding operators commute:

$$\boxed{[\hat{A}, \hat{B}] = 0}$$

If $[\hat{A}, \hat{B}] \neq 0$, the observables are **incompatible**.

### 4. The Fundamental Theorem

**Theorem (Compatible Observables):** Two Hermitian operators $\hat{A}$ and $\hat{B}$ commute if and only if they possess a complete set of common eigenvectors.

$$[\hat{A}, \hat{B}] = 0 \iff \exists \text{ common eigenbasis } \{|a_i, b_j\rangle\}$$

**Proof (Necessity, non-degenerate case):**

Suppose $[\hat{A}, \hat{B}] = 0$ and $\hat{A}$ has non-degenerate eigenvalues.

Let $|a\rangle$ be an eigenstate of $\hat{A}$: $\hat{A}|a\rangle = a|a\rangle$

Consider $\hat{A}(\hat{B}|a\rangle)$:

$$\hat{A}(\hat{B}|a\rangle) = \hat{B}(\hat{A}|a\rangle) = \hat{B}(a|a\rangle) = a(\hat{B}|a\rangle)$$

This shows $\hat{B}|a\rangle$ is an eigenvector of $\hat{A}$ with eigenvalue $a$.

Since $a$ is non-degenerate, $\hat{B}|a\rangle$ must be proportional to $|a\rangle$:

$$\hat{B}|a\rangle = b|a\rangle$$

Therefore $|a\rangle$ is also an eigenstate of $\hat{B}$. $\blacksquare$

**Proof (Sufficiency):**

Suppose $\{|a_i, b_j\rangle\}$ is a common eigenbasis:

$$\hat{A}|a_i, b_j\rangle = a_i|a_i, b_j\rangle, \quad \hat{B}|a_i, b_j\rangle = b_j|a_i, b_j\rangle$$

For any state $|\psi\rangle = \sum_{i,j} c_{ij}|a_i, b_j\rangle$:

$$[\hat{A}, \hat{B}]|\psi\rangle = \sum_{i,j} c_{ij}(a_i b_j - b_j a_i)|a_i, b_j\rangle = 0$$

Since this holds for all $|\psi\rangle$: $[\hat{A}, \hat{B}] = 0$ $\blacksquare$

### 5. Physical Interpretation

If $[\hat{A}, \hat{B}] = 0$:

1. **Simultaneous measurement is possible:** We can prepare a state with definite values of both $A$ and $B$.

2. **Order doesn't matter:** Measuring $A$ then $B$ gives the same statistics as measuring $B$ then $A$.

3. **No disturbance:** Measuring $A$ doesn't affect the outcome of a subsequent $B$ measurement (when in a common eigenstate).

If $[\hat{A}, \hat{B}] \neq 0$:

1. **Uncertainty principle applies:** $\Delta A \cdot \Delta B \geq \frac{1}{2}|⟨[\hat{A}, \hat{B}]⟩|$

2. **Measurement disturbs:** Measuring one observable affects the other.

### 6. The Degenerate Case

When $\hat{A}$ has degenerate eigenvalues, the proof requires more care.

**Theorem (Degenerate case):** If $[\hat{A}, \hat{B}] = 0$ and $a$ is a degenerate eigenvalue of $\hat{A}$ with eigenspace $\mathcal{E}_a$, then $\hat{B}$ maps $\mathcal{E}_a$ into itself.

We can then choose a basis of $\mathcal{E}_a$ consisting of eigenvectors of $\hat{B}$.

**Key insight:** Within each degenerate subspace of $\hat{A}$, we can diagonalize $\hat{B}$.

### 7. Complete Set of Commuting Observables (CSCO)

**Definition:** A **Complete Set of Commuting Observables (CSCO)** is a minimal set of mutually commuting Hermitian operators $\{\hat{A}, \hat{B}, \hat{C}, ...\}$ whose common eigenstates are uniquely specified (non-degenerate).

**Properties of CSCO:**
1. All operators in the set commute pairwise
2. Specifying eigenvalues $(a, b, c, ...)$ uniquely determines the state
3. The set is minimal (no operator can be removed while maintaining uniqueness)

**Example: Free particle in 3D**

- $\hat{H} = \frac{\hat{p}^2}{2m}$ alone has degenerate eigenvalues (same energy for different directions)
- CSCO: $\{\hat{H}, \hat{L}^2, \hat{L}_z\}$ or equivalently $\{\hat{p}_x, \hat{p}_y, \hat{p}_z\}$

**Example: Hydrogen atom**

CSCO: $\{\hat{H}, \hat{L}^2, \hat{L}_z, \hat{S}_z\}$

States are labeled $|n, \ell, m_\ell, m_s\rangle$.

### 8. Physical Examples of Compatible Observables

#### Example 1: Energy and Angular Momentum (Central Potential)

For a central potential $V(r)$:

$$[\hat{H}, \hat{L}^2] = 0, \quad [\hat{H}, \hat{L}_z] = 0, \quad [\hat{L}^2, \hat{L}_z] = 0$$

Common eigenstates: $|n, \ell, m\rangle$

$$\hat{H}|n,\ell,m\rangle = E_n|n,\ell,m\rangle$$
$$\hat{L}^2|n,\ell,m\rangle = \ell(\ell+1)\hbar^2|n,\ell,m\rangle$$
$$\hat{L}_z|n,\ell,m\rangle = m\hbar|n,\ell,m\rangle$$

#### Example 2: Momentum Components

$$[\hat{p}_x, \hat{p}_y] = [\hat{p}_y, \hat{p}_z] = [\hat{p}_z, \hat{p}_x] = 0$$

A particle can have definite $p_x$, $p_y$, and $p_z$ simultaneously.

Common eigenstates: plane waves $|p_x, p_y, p_z\rangle$

#### Example 3: Position Components

$$[\hat{x}, \hat{y}] = [\hat{y}, \hat{z}] = [\hat{z}, \hat{x}] = 0$$

A particle can have definite position in all three directions.

### 9. Examples of Incompatible Observables

#### Example 1: Position and Momentum (Same Component)

$$[\hat{x}, \hat{p}_x] = i\hbar \neq 0$$

Cannot simultaneously measure $x$ and $p_x$ with arbitrary precision.

#### Example 2: Angular Momentum Components

$$[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z \neq 0$$

Cannot have definite $L_x$ and $L_y$ simultaneously (unless $L_z = 0$).

#### Example 3: Position and Energy (Bound States)

For a particle in a potential:

$$[\hat{x}, \hat{H}] = \frac{i\hbar}{m}\hat{p}_x \neq 0$$

Energy eigenstates don't have definite position.

---

## Quantum Computing Connection

### Compatible Observables in Quantum Information

In quantum computing, compatible observables correspond to **measurements that can be performed simultaneously**.

**Pauli Matrices:**
$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Commutation relations:**
$$[\sigma_x, \sigma_y] = 2i\sigma_z, \quad [\sigma_y, \sigma_z] = 2i\sigma_x, \quad [\sigma_z, \sigma_x] = 2i\sigma_y$$

These do NOT commute! Measuring in X-basis disturbs Z-basis measurements.

### Stabilizer Formalism

In quantum error correction, we need observables that:
1. Commute with each other (can measure simultaneously)
2. Commute with logical operators (don't disturb encoded information)

**Example: 3-qubit bit-flip code**

Stabilizers: $Z_1Z_2$ and $Z_2Z_3$

$$[Z_1Z_2, Z_2Z_3] = 0$$

These can be measured simultaneously to detect errors.

### Multi-Qubit Compatible Measurements

For $n$ qubits, Pauli operators on DIFFERENT qubits commute:

$$[\sigma_z^{(1)}, \sigma_x^{(2)}] = 0$$

This allows measuring qubit 1 in Z-basis while measuring qubit 2 in X-basis.

---

## Worked Examples

### Example 1: Verifying Compatibility of Operators

**Problem:** Show that $[\hat{L}^2, \hat{L}_z] = 0$.

**Solution:**

Express $\hat{L}^2 = \hat{L}_x^2 + \hat{L}_y^2 + \hat{L}_z^2$

We need: $[\hat{L}_x^2 + \hat{L}_y^2 + \hat{L}_z^2, \hat{L}_z] = 0$

First, $[\hat{L}_z^2, \hat{L}_z] = 0$ (any operator commutes with itself and its powers).

For $[\hat{L}_x^2, \hat{L}_z]$, use the identity:
$$[\hat{A}^2, \hat{B}] = \hat{A}[\hat{A}, \hat{B}] + [\hat{A}, \hat{B}]\hat{A}$$

With $[\hat{L}_x, \hat{L}_z] = -i\hbar\hat{L}_y$:

$$[\hat{L}_x^2, \hat{L}_z] = \hat{L}_x(-i\hbar\hat{L}_y) + (-i\hbar\hat{L}_y)\hat{L}_x = -i\hbar(\hat{L}_x\hat{L}_y + \hat{L}_y\hat{L}_x)$$

Similarly, with $[\hat{L}_y, \hat{L}_z] = i\hbar\hat{L}_x$:

$$[\hat{L}_y^2, \hat{L}_z] = i\hbar(\hat{L}_y\hat{L}_x + \hat{L}_x\hat{L}_y)$$

Adding:
$$[\hat{L}^2, \hat{L}_z] = [\hat{L}_x^2, \hat{L}_z] + [\hat{L}_y^2, \hat{L}_z] = 0$$ $\checkmark$

### Example 2: Constructing Simultaneous Eigenstates

**Problem:** For a spin-1/2 particle, the operators $\hat{S}^2$ and $\hat{S}_z$ commute. Find their common eigenstates.

**Solution:**

For spin-1/2: $s = 1/2$

$\hat{S}^2$ eigenvalue: $s(s+1)\hbar^2 = \frac{3}{4}\hbar^2$

$\hat{S}_z$ eigenvalues: $m_s\hbar$ where $m_s = \pm 1/2$

Common eigenstates:

$$|s, m_s\rangle = |\tfrac{1}{2}, +\tfrac{1}{2}\rangle \equiv |\uparrow\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

$$|s, m_s\rangle = |\tfrac{1}{2}, -\tfrac{1}{2}\rangle \equiv |\downarrow\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

**Verification:**

$$\hat{S}^2|\uparrow\rangle = \frac{3}{4}\hbar^2|\uparrow\rangle, \quad \hat{S}_z|\uparrow\rangle = +\frac{\hbar}{2}|\uparrow\rangle$$

$$\hat{S}^2|\downarrow\rangle = \frac{3}{4}\hbar^2|\downarrow\rangle, \quad \hat{S}_z|\downarrow\rangle = -\frac{\hbar}{2}|\downarrow\rangle$$

Both states are simultaneous eigenstates of $\hat{S}^2$ and $\hat{S}_z$.

### Example 3: CSCO for a 2D Harmonic Oscillator

**Problem:** Find a CSCO for the 2D isotropic harmonic oscillator with Hamiltonian:

$$\hat{H} = \frac{\hat{p}_x^2 + \hat{p}_y^2}{2m} + \frac{1}{2}m\omega^2(\hat{x}^2 + \hat{y}^2)$$

**Solution:**

**Step 1:** Identify degeneracies of $\hat{H}$.

The energy eigenvalues are $E_{n_x, n_y} = \hbar\omega(n_x + n_y + 1)$.

For total quantum number $N = n_x + n_y$, there are $N+1$ degenerate states.

$\hat{H}$ alone is NOT a CSCO.

**Step 2:** Find a commuting operator.

Define $\hat{L}_z = \hat{x}\hat{p}_y - \hat{y}\hat{p}_x$.

Show $[\hat{H}, \hat{L}_z] = 0$:

$$[\hat{H}, \hat{L}_z] = [\frac{\hat{p}_x^2}{2m}, \hat{x}\hat{p}_y] + [\frac{\hat{p}_y^2}{2m}, -\hat{y}\hat{p}_x] + \frac{m\omega^2}{2}[\hat{x}^2, -\hat{y}\hat{p}_x] + \frac{m\omega^2}{2}[\hat{y}^2, \hat{x}\hat{p}_y]$$

Each term vanishes (exercise: verify!).

**Step 3:** The CSCO is $\{\hat{H}, \hat{L}_z\}$.

Common eigenstates: $|N, m\rangle$ where $N = 0, 1, 2, ...$ and $m = -N, -N+2, ..., N-2, N$.

These are uniquely labeled by $(N, m)$.

---

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Compute $[\hat{x}^2, \hat{p}]$.

*Hint: Use $[\hat{A}\hat{B}, \hat{C}] = \hat{A}[\hat{B}, \hat{C}] + [\hat{A}, \hat{C}]\hat{B}$*

**Answer:** $[\hat{x}^2, \hat{p}] = 2i\hbar\hat{x}$

---

**Problem 1.2:** Show that $[\hat{p}_x, \hat{p}_y] = 0$.

*Hint: In position representation, $\hat{p}_x = -i\hbar\partial/\partial x$.*

**Answer:** Partial derivatives commute: $\frac{\partial^2}{\partial x \partial y} = \frac{\partial^2}{\partial y \partial x}$

---

**Problem 1.3:** For the Pauli matrices, verify $[\sigma_x, \sigma_y] = 2i\sigma_z$.

**Answer:** Direct matrix multiplication gives $\sigma_x\sigma_y = i\sigma_z$ and $\sigma_y\sigma_x = -i\sigma_z$.

---

### Level 2: Intermediate

**Problem 2.1:** A system has Hamiltonian $\hat{H}$ with energy eigenstates $|E_1\rangle, |E_2\rangle$ (non-degenerate). Another observable $\hat{A}$ has eigenstates:

$$|a_+\rangle = \frac{1}{\sqrt{2}}(|E_1\rangle + |E_2\rangle), \quad |a_-\rangle = \frac{1}{\sqrt{2}}(|E_1\rangle - |E_2\rangle)$$

Do $\hat{H}$ and $\hat{A}$ commute? Explain.

**Answer:** No, they don't commute. The eigenstates $|E_1\rangle, |E_2\rangle$ of $\hat{H}$ are NOT eigenstates of $\hat{A}$ (they are superpositions of $|a_+\rangle, |a_-\rangle$). Commuting operators must share common eigenstates.

---

**Problem 2.2:** The angular momentum operators for spin-1 satisfy:

$$[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$$

Is it possible to find a state $|\psi\rangle$ such that $\hat{L}_x|\psi\rangle = \hbar|\psi\rangle$ AND $\hat{L}_y|\psi\rangle = 0$?

**Answer:** If such a state existed, it would be a simultaneous eigenstate of $\hat{L}_x$ and $\hat{L}_y$. But these operators don't commute, so no common eigenstates exist (except trivially when $\hat{L}_z = 0$, which isn't the case here). No such state exists.

---

**Problem 2.3:** Show that for any operator $\hat{A}$, $[\hat{A}, \hat{A}^n] = 0$ for all positive integers $n$.

**Answer:** Use induction. Base case: $[\hat{A}, \hat{A}] = 0$. Inductive step: $[\hat{A}, \hat{A}^{n+1}] = [\hat{A}, \hat{A}^n\hat{A}] = [\hat{A}, \hat{A}^n]\hat{A} + \hat{A}^n[\hat{A}, \hat{A}] = 0$ by inductive hypothesis.

---

### Level 3: Challenging

**Problem 3.1:** Prove that if $[\hat{A}, \hat{B}] = 0$ and $[\hat{A}, \hat{C}] = 0$, then $[\hat{A}, \hat{B}\hat{C}] = 0$.

**Solution:**
$$[\hat{A}, \hat{B}\hat{C}] = [\hat{A}, \hat{B}]\hat{C} + \hat{B}[\hat{A}, \hat{C}] = 0 \cdot \hat{C} + \hat{B} \cdot 0 = 0$$ $\checkmark$

---

**Problem 3.2:** Consider the operators:

$$\hat{A} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{pmatrix}, \quad \hat{B} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

(a) Verify that $[\hat{A}, \hat{B}] = 0$.
(b) Find all common eigenvectors.
(c) Is $\{\hat{A}, \hat{B}\}$ a CSCO for $\mathbb{C}^3$?

**Solution:**
(a) Both are diagonal, so $\hat{A}\hat{B} = \hat{B}\hat{A}$. $[\hat{A}, \hat{B}] = 0$ $\checkmark$

(b) Common eigenvectors are the standard basis vectors:
- $|1\rangle = (1,0,0)^T$: $\hat{A}|1\rangle = |1\rangle$, $\hat{B}|1\rangle = |1\rangle$ (eigenvalues $a=1, b=1$)
- $|2\rangle = (0,1,0)^T$: $\hat{A}|2\rangle = |2\rangle$, $\hat{B}|2\rangle = -|2\rangle$ (eigenvalues $a=1, b=-1$)
- $|3\rangle = (0,0,1)^T$: $\hat{A}|3\rangle = -|3\rangle$, $\hat{B}|3\rangle = |3\rangle$ (eigenvalues $a=-1, b=1$)

(c) Yes! Each pair $(a, b)$ uniquely specifies a state. No degeneracy remains.

---

**Problem 3.3:** Let $\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})$ be a 1D Hamiltonian. Under what condition on $V(x)$ does $[\hat{H}, \hat{P}] = 0$ where $\hat{P}$ is the parity operator?

**Solution:** The parity operator satisfies $\hat{P}|\psi(x)\rangle = |\psi(-x)\rangle$.

For $[\hat{H}, \hat{P}] = 0$, we need $\hat{H}\hat{P} = \hat{P}\hat{H}$.

The kinetic term $\hat{p}^2$ always commutes with $\hat{P}$ (momentum squared is even).

For the potential: $\hat{P}V(\hat{x})\hat{P}^{-1} = V(-\hat{x})$.

So $[\hat{H}, \hat{P}] = 0$ if and only if $V(-x) = V(x)$, i.e., the potential is **symmetric (even)**.

---

## Computational Lab: Compatible Observables in Python

```python
"""
Day 347 Computational Lab: Compatible Observables
Topics: Commutator calculations, CSCO, simultaneous eigenstates
"""

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

# ============================================
# Part 1: Computing Commutators
# ============================================

def commutator(A, B):
    """Compute the commutator [A, B] = AB - BA"""
    return A @ B - B @ A

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

print("=" * 50)
print("Part 1: Pauli Matrix Commutators")
print("=" * 50)

# Verify Pauli commutation relations
comm_xy = commutator(sigma_x, sigma_y)
print("\n[sigma_x, sigma_y] = ")
print(comm_xy)
print(f"Expected 2i*sigma_z = {2j * sigma_z}")
print(f"Match: {np.allclose(comm_xy, 2j * sigma_z)}")

comm_yz = commutator(sigma_y, sigma_z)
print(f"\n[sigma_y, sigma_z] = 2i*sigma_x: {np.allclose(comm_yz, 2j * sigma_x)}")

comm_zx = commutator(sigma_z, sigma_x)
print(f"[sigma_z, sigma_x] = 2i*sigma_y: {np.allclose(comm_zx, 2j * sigma_y)}")

# ============================================
# Part 2: Compatible Observables - Finding Common Eigenstates
# ============================================

print("\n" + "=" * 50)
print("Part 2: Finding Common Eigenstates")
print("=" * 50)

# Define two commuting operators
A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, -1]], dtype=complex)

B = np.array([[1, 0, 0],
              [0, -1, 0],
              [0, 0, 1]], dtype=complex)

print(f"\nOperator A:\n{A}")
print(f"\nOperator B:\n{B}")
print(f"\n[A, B] = 0: {np.allclose(commutator(A, B), np.zeros((3,3)))}")

# Find eigenstates of A
eigenvalues_A, eigenvectors_A = la.eig(A)
print(f"\nEigenvalues of A: {eigenvalues_A}")

# For each eigenstate of A, check if it's also an eigenstate of B
print("\nChecking common eigenstates:")
for i, (eval_A, evec) in enumerate(zip(eigenvalues_A, eigenvectors_A.T)):
    B_evec = B @ evec
    # Check if B|v> is proportional to |v>
    if np.allclose(evec, 0):
        continue
    ratio = B_evec / evec
    # If all ratios are the same, it's an eigenstate
    ratio = ratio[~np.isnan(ratio) & ~np.isinf(ratio)]
    if len(ratio) > 0 and np.allclose(ratio, ratio[0]):
        eval_B = ratio[0]
        print(f"  State {i+1}: A-eigenvalue = {eval_A:.1f}, B-eigenvalue = {eval_B:.1f}")

# ============================================
# Part 3: Spin-1 Angular Momentum
# ============================================

print("\n" + "=" * 50)
print("Part 3: Spin-1 Angular Momentum (L=1)")
print("=" * 50)

# L=1 angular momentum matrices
sqrt2 = np.sqrt(2)
Lx = (1/sqrt2) * np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=complex)

Ly = (1/sqrt2) * np.array([[0, -1j, 0],
                           [1j, 0, -1j],
                           [0, 1j, 0]], dtype=complex)

Lz = np.array([[1, 0, 0],
               [0, 0, 0],
               [0, 0, -1]], dtype=complex)

# L^2 = Lx^2 + Ly^2 + Lz^2
L_squared = Lx @ Lx + Ly @ Ly + Lz @ Lz

print(f"\n[Lx, Ly] = i*Lz: {np.allclose(commutator(Lx, Ly), 1j * Lz)}")
print(f"[L^2, Lz] = 0: {np.allclose(commutator(L_squared, Lz), np.zeros((3,3)))}")
print(f"[Lx, Lz] = -i*Ly: {np.allclose(commutator(Lx, Lz), -1j * Ly)}")

# Find common eigenstates of L^2 and Lz
print("\nL^2 eigenvalues (should all be l(l+1) = 2):")
evals_L2, evecs_L2 = la.eig(L_squared)
print(np.round(evals_L2.real, 4))

print("\nLz eigenvalues (m = -1, 0, 1):")
evals_Lz, evecs_Lz = la.eig(Lz)
print(np.round(evals_Lz.real, 4))

# ============================================
# Part 4: Demonstration - Measurement Order Matters
# ============================================

print("\n" + "=" * 50)
print("Part 4: Measurement Order for Non-Commuting Observables")
print("=" * 50)

# For incompatible observables, measurement order matters
# Simulate measuring X then Z vs Z then X

np.random.seed(42)

def measure(state, observable):
    """
    Perform a measurement of observable on state.
    Returns (outcome, post_measurement_state)
    """
    eigenvalues, eigenvectors = la.eig(observable)
    probabilities = np.abs(eigenvectors.conj().T @ state)**2
    probabilities = probabilities.flatten()

    # Choose outcome based on probabilities
    outcome_idx = np.random.choice(len(eigenvalues), p=probabilities)
    outcome = eigenvalues[outcome_idx].real

    # Post-measurement state (collapsed)
    post_state = eigenvectors[:, outcome_idx:outcome_idx+1]

    return outcome, post_state

# Start in |+> state
plus = np.array([[1], [1]], dtype=complex) / np.sqrt(2)

print("\nStarting state: |+> = (|0> + |1>)/sqrt(2)")

# Experiment 1: Measure Z, then X
n_trials = 10000
z_then_x_results = []
x_then_z_results = []

for _ in range(n_trials):
    state = plus.copy()
    z_result, state = measure(state, sigma_z)
    x_result, _ = measure(state, sigma_x)
    z_then_x_results.append((z_result, x_result))

for _ in range(n_trials):
    state = plus.copy()
    x_result, state = measure(state, sigma_x)
    z_result, _ = measure(state, sigma_z)
    x_then_z_results.append((x_result, z_result))

# Analyze results
z_then_x_array = np.array(z_then_x_results)
x_then_z_array = np.array(x_then_z_results)

print(f"\nMeasuring Z then X on |+>:")
print(f"  P(Z=+1) = {np.mean(z_then_x_array[:,0] == 1):.3f}")
print(f"  <X> after Z measurement = {np.mean(z_then_x_array[:,1]):.3f}")

print(f"\nMeasuring X then Z on |+>:")
print(f"  P(X=+1) = {np.mean(x_then_z_array[:,0] == 1):.3f}")
print(f"  <Z> after X measurement = {np.mean(x_then_z_array[:,1]):.3f}")

print("\nNote: For |+>, measuring X first gives X=+1 always!")
print("But measuring Z first randomizes, then X is random too.")

# ============================================
# Part 5: Visualizing CSCO for 2D Harmonic Oscillator
# ============================================

print("\n" + "=" * 50)
print("Part 5: Energy Level Diagram with CSCO Labels")
print("=" * 50)

# Create energy level diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Without CSCO (degenerate)
for N in range(5):
    energy = N + 1  # E = hbar*omega*(N+1), set hbar*omega = 1
    degeneracy = N + 1
    ax1.hlines(energy, 0.2, 0.8, colors='blue', linewidth=2)
    ax1.text(0.82, energy, f'N={N}, deg={degeneracy}', fontsize=10, va='center')

ax1.set_xlim(0, 1.5)
ax1.set_ylim(0, 6)
ax1.set_ylabel('Energy (units of hbar*omega)', fontsize=12)
ax1.set_title('2D Harmonic Oscillator\n(Labeled by H only - degenerate)', fontsize=12)
ax1.set_xticks([])

# With CSCO (H, Lz)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, 10))
for N in range(5):
    energy = N + 1
    m_values = list(range(-N, N+1, 2))
    n_states = len(m_values)

    for i, m in enumerate(m_values):
        x_offset = 0.3 + i * 0.4 / n_states
        ax2.hlines(energy, x_offset, x_offset + 0.3/n_states,
                   colors=colors[m+5], linewidth=2)
        ax2.text(x_offset + 0.15/n_states, energy + 0.15, f'm={m}',
                 fontsize=8, ha='center')

ax2.set_xlim(0, 1.5)
ax2.set_ylim(0, 6)
ax2.set_ylabel('Energy (units of hbar*omega)', fontsize=12)
ax2.set_title('2D Harmonic Oscillator\n(CSCO: H, Lz - non-degenerate)', fontsize=12)
ax2.set_xticks([])

plt.tight_layout()
plt.savefig('csco_energy_levels.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nEnergy level diagram saved to 'csco_energy_levels.png'")

# ============================================
# Part 6: Building a CSCO from scratch
# ============================================

print("\n" + "=" * 50)
print("Part 6: Systematically Finding a CSCO")
print("=" * 50)

def is_degenerate(operator, tol=1e-10):
    """Check if an operator has degenerate eigenvalues"""
    eigenvalues = la.eigvals(operator)
    for i, ev1 in enumerate(eigenvalues):
        for ev2 in eigenvalues[i+1:]:
            if np.abs(ev1 - ev2) < tol:
                return True
    return False

def find_csco(operators, labels):
    """
    Given a list of commuting operators, find a minimal CSCO.
    Returns indices of operators that form the CSCO.
    """
    n = operators[0].shape[0]
    csco_indices = []
    combined = np.zeros((n, n), dtype=complex)

    for i, (op, label) in enumerate(zip(operators, labels)):
        # Check if adding this operator helps resolve degeneracy
        test_combined = combined + (i+1) * op  # Weight by index to distinguish
        if is_degenerate(combined) or len(csco_indices) == 0:
            if not is_degenerate(test_combined) or len(csco_indices) == 0:
                csco_indices.append(i)
                combined = test_combined
                if not is_degenerate(combined):
                    break

    return csco_indices

# Example: 3-level system with H degenerate
H = np.diag([1, 1, 2]).astype(complex)  # First two levels degenerate
A = np.diag([1, -1, 0]).astype(complex)  # Splits the degeneracy

print(f"H eigenvalues: {la.eigvals(H)}")
print(f"H is degenerate: {is_degenerate(H)}")
print(f"\nA eigenvalues: {la.eigvals(A)}")
print(f"[H, A] = 0: {np.allclose(commutator(H, A), 0)}")

# Combined operator H + lambda*A for labeling
combined = H + 0.1 * A
print(f"\nCombined (H + 0.1*A) eigenvalues: {np.round(la.eigvals(combined).real, 3)}")
print(f"Combined is degenerate: {is_degenerate(combined)}")
print("\n{H, A} forms a CSCO for this 3D Hilbert space.")

print("\n" + "=" * 50)
print("Lab Complete!")
print("=" * 50)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Commutator | $[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}$ |
| Compatible condition | $[\hat{A}, \hat{B}] = 0$ |
| Product rule | $[\hat{A}, \hat{B}\hat{C}] = [\hat{A}, \hat{B}]\hat{C} + \hat{B}[\hat{A}, \hat{C}]$ |
| Jacobi identity | $[\hat{A}, [\hat{B}, \hat{C}]] + [\hat{B}, [\hat{C}, \hat{A}]] + [\hat{C}, [\hat{A}, \hat{B}]] = 0$ |
| Uncertainty relation | $\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle[\hat{A}, \hat{B}]\rangle|$ |

### Key Takeaways

1. **Compatible observables** commute and share a complete set of common eigenstates.

2. **The commutator** $[\hat{A}, \hat{B}]$ determines whether simultaneous measurement is possible.

3. **CSCO** provides the minimum set of observables needed to uniquely label quantum states.

4. **Physical examples:** Position components commute; angular momentum components don't.

5. **Quantum computing:** Stabilizer measurements must commute for simultaneous measurement.

---

## Daily Checklist

- [ ] Read Shankar 4.2 and Sakurai 1.4 on compatible observables
- [ ] Derive the commutator product rule (Leibniz identity)
- [ ] Prove that $[\hat{L}^2, \hat{L}_z] = 0$
- [ ] Work through all three examples
- [ ] Complete Level 1 and 2 practice problems
- [ ] Attempt at least one Level 3 problem
- [ ] Run computational lab and interpret results
- [ ] Create flashcards for commutator identities
- [ ] Write summary of CSCO concept in study journal

---

## Preview: Tomorrow's Topics

**Day 348: Position and Momentum Operators**

Tomorrow we dive into the fundamental compatible/incompatible pair:

- Position operator $\hat{x}$ and its eigenstates
- Momentum operator $\hat{p} = -i\hbar\frac{d}{dx}$
- The canonical commutation relation $[\hat{x}, \hat{p}] = i\hbar$
- Position and momentum representations
- Why $\langle x|\psi\rangle = \psi(x)$ is the wave function

**Preparation:** Review the delta function from your mathematical foundations.

---

**References:**
- Shankar, R. (1994). Principles of Quantum Mechanics, Chapter 4
- Sakurai, J.J. (2017). Modern Quantum Mechanics, Chapter 1.4
- Griffiths, D.J. (2018). Introduction to Quantum Mechanics, Chapter 3.5
- Nielsen, M.A. & Chuang, I.L. (2010). Quantum Computation and Quantum Information, Chapter 2
