# Day 341: Eigenvalue Problems — The Mathematics of Measurement

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Eigenvalue Problems in Quantum Mechanics |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 341, you will be able to:

1. Solve the eigenvalue equation for operators in Hilbert space
2. Compute the characteristic polynomial and find eigenvalues via the secular equation
3. Distinguish between algebraic and geometric multiplicity (degeneracy)
4. Construct the spectral decomposition of Hermitian operators
5. Diagonalize quantum mechanical operators
6. Evaluate functions of operators using eigenvalue decomposition
7. Connect eigenvalues to physical measurement outcomes

---

## Core Content

### 1. The Eigenvalue Equation

The central equation of quantum mechanics connects operators to observable quantities:

$$\boxed{\hat{A}|a\rangle = a|a\rangle}$$

**Terminology:**
- $|a\rangle$ is an **eigenstate** (or eigenvector, eigenket) of $\hat{A}$
- $a$ is the corresponding **eigenvalue**
- The set of all eigenvalues is the **spectrum** of $\hat{A}$

**Physical Interpretation:**

When a system is in eigenstate $|a\rangle$ of observable $\hat{A}$:
- Measurement of $A$ yields value $a$ with certainty (probability 1)
- The state remains $|a\rangle$ after measurement (no disturbance)
- $|a\rangle$ is a state of "definite $A$"

**Example:** For the spin-z operator $\hat{S}_z$:
$$\hat{S}_z|\uparrow\rangle = +\frac{\hbar}{2}|\uparrow\rangle, \quad \hat{S}_z|\downarrow\rangle = -\frac{\hbar}{2}|\downarrow\rangle$$

The eigenvalues $\pm\hbar/2$ are the only possible measurement outcomes.

---

### 2. The Characteristic Polynomial and Secular Equation

To find eigenvalues, we solve the **secular equation** (characteristic equation):

$$\boxed{\det(\hat{A} - \lambda \hat{I}) = 0}$$

**Derivation:**

The eigenvalue equation $\hat{A}|a\rangle = a|a\rangle$ can be rewritten as:

$$(\hat{A} - a\hat{I})|a\rangle = 0$$

For a non-trivial solution ($|a\rangle \neq 0$), the operator $(\hat{A} - a\hat{I})$ must be singular:

$$\det(\hat{A} - a\hat{I}) = 0$$

**Characteristic Polynomial:**

For an $n \times n$ matrix, this determinant is a polynomial of degree $n$:

$$p(\lambda) = \det(\hat{A} - \lambda\hat{I}) = (-1)^n\lambda^n + c_{n-1}\lambda^{n-1} + \cdots + c_1\lambda + c_0$$

**Fundamental Theorem of Algebra:** An $n \times n$ matrix has exactly $n$ eigenvalues (counting multiplicity) in $\mathbb{C}$.

**Example: 2x2 Matrix**

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

$$\det(A - \lambda I) = \det\begin{pmatrix} a-\lambda & b \\ c & d-\lambda \end{pmatrix} = (a-\lambda)(d-\lambda) - bc$$

$$= \lambda^2 - (a+d)\lambda + (ad - bc) = \lambda^2 - \text{Tr}(A)\lambda + \det(A) = 0$$

$$\boxed{\lambda = \frac{\text{Tr}(A) \pm \sqrt{\text{Tr}(A)^2 - 4\det(A)}}{2}}$$

---

### 3. Degeneracy: Algebraic vs. Geometric Multiplicity

**Definitions:**

- **Algebraic multiplicity** $m_a$: The number of times eigenvalue $a$ appears as a root of the characteristic polynomial

- **Geometric multiplicity** $m_g$: The dimension of the eigenspace $\mathcal{E}_a = \{|v\rangle : \hat{A}|v\rangle = a|v\rangle\}$

**Key Theorem:**

$$\boxed{1 \leq m_g \leq m_a}$$

**Degeneracy in Quantum Mechanics:**

When $m_a > 1$, we say eigenvalue $a$ is **degenerate** with degeneracy $m_a$.

**Physical Significance:**
- Degenerate eigenvalues mean multiple linearly independent states have the same measurement outcome
- Often arises from symmetries (e.g., angular momentum degeneracy in hydrogen)
- Requires careful handling in perturbation theory

**Example: Hydrogen Atom**

The energy eigenvalues $E_n = -13.6/n^2$ eV have degeneracy $n^2$:
- $n=1$: 1 state (1s)
- $n=2$: 4 states (2s, 2p$_x$, 2p$_y$, 2p$_z$)
- $n=3$: 9 states

---

### 4. Spectral Decomposition (Eigenvalue Decomposition)

For a **Hermitian operator** $\hat{A}$, the spectral theorem guarantees:

$$\boxed{\hat{A} = \sum_a a |a\rangle\langle a| = \sum_a a \hat{P}_a}$$

where $\hat{P}_a = |a\rangle\langle a|$ is the projection onto eigenstate $|a\rangle$.

**For degenerate eigenvalues:**

$$\hat{A} = \sum_a a \hat{P}_a, \quad \hat{P}_a = \sum_{i=1}^{m_a} |a, i\rangle\langle a, i|$$

where $\{|a, i\rangle\}$ is an orthonormal basis for the eigenspace $\mathcal{E}_a$.

**Properties of the Spectral Decomposition:**

1. **Resolution of identity:**
   $$\sum_a \hat{P}_a = \hat{I}$$

2. **Orthogonality of projectors:**
   $$\hat{P}_a \hat{P}_{a'} = \delta_{aa'}\hat{P}_a$$

3. **Idempotence:**
   $$\hat{P}_a^2 = \hat{P}_a$$

**Physical Interpretation:**

The spectral decomposition expresses an observable as a weighted sum of "measurement apparatus settings"---each projector $\hat{P}_a$ corresponds to the detector that clicks when outcome $a$ is obtained.

---

### 5. Diagonalization of Operators

An operator is **diagonalizable** if there exists a basis in which its matrix representation is diagonal.

**Procedure for Diagonalization:**

1. **Find eigenvalues:** Solve $\det(\hat{A} - \lambda\hat{I}) = 0$

2. **Find eigenvectors:** For each eigenvalue $a$, solve $(\hat{A} - a\hat{I})|a\rangle = 0$

3. **Construct transformation matrix:** $\hat{U} = (|a_1\rangle, |a_2\rangle, \ldots, |a_n\rangle)$

4. **Verify:** $\hat{U}^{-1}\hat{A}\hat{U} = \hat{D}$ where $\hat{D}$ is diagonal

**For Hermitian Operators:**

If $\hat{A}^\dagger = \hat{A}$, then:
- All eigenvalues are real
- Eigenvectors for distinct eigenvalues are orthogonal
- $\hat{U}$ is unitary: $\hat{U}^{-1} = \hat{U}^\dagger$

$$\boxed{\hat{U}^\dagger \hat{A} \hat{U} = \text{diag}(a_1, a_2, \ldots, a_n)}$$

**Matrix Form:**

$$A = \begin{pmatrix} a_{11} & a_{12} & \cdots \\ a_{21} & a_{22} & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix} \xrightarrow{\hat{U}^\dagger(\cdot)\hat{U}} D = \begin{pmatrix} \lambda_1 & 0 & \cdots \\ 0 & \lambda_2 & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

---

### 6. Functions of Operators via Eigenvalue Decomposition

Given a function $f: \mathbb{C} \to \mathbb{C}$ and a diagonalizable operator $\hat{A}$:

$$\boxed{f(\hat{A}) = \sum_a f(a) |a\rangle\langle a|}$$

**Key Insight:** The function acts on the eigenvalues, not the operator directly.

**Important Examples:**

1. **Powers:**
   $$\hat{A}^n = \sum_a a^n |a\rangle\langle a|$$

2. **Exponential (crucial for time evolution):**
   $$e^{\hat{A}} = \sum_a e^a |a\rangle\langle a|$$

3. **Square root:**
   $$\sqrt{\hat{A}} = \sum_a \sqrt{a} |a\rangle\langle a|$$

4. **Inverse (if all $a \neq 0$):**
   $$\hat{A}^{-1} = \sum_a a^{-1} |a\rangle\langle a|$$

**Time Evolution Operator:**

The most important application in quantum mechanics:

$$\hat{U}(t) = e^{-i\hat{H}t/\hbar} = \sum_n e^{-iE_n t/\hbar} |E_n\rangle\langle E_n|$$

This is why finding energy eigenstates is so powerful---once we know them, time evolution becomes trivial!

---

### 7. Physical Interpretation: Eigenvalues as Measurement Outcomes

**The Measurement Postulate (Preview):**

When observable $\hat{A}$ is measured on state $|\psi\rangle$:

1. **Possible outcomes:** Only eigenvalues $\{a\}$ can be observed

2. **Probability of outcome $a$:**
   $$P(a) = |\langle a|\psi\rangle|^2 = \langle\psi|\hat{P}_a|\psi\rangle$$

3. **Post-measurement state:** $|a\rangle$ (or projection into $\mathcal{E}_a$ if degenerate)

**Connection to Expectation Value:**

$$\langle\hat{A}\rangle = \langle\psi|\hat{A}|\psi\rangle = \sum_a a |\langle a|\psi\rangle|^2 = \sum_a a \cdot P(a)$$

The expectation value is the probability-weighted average of eigenvalues.

**Uncertainty and Eigenstates:**

If $|\psi\rangle = |a\rangle$ (an eigenstate):
$$\Delta A = \sqrt{\langle\hat{A}^2\rangle - \langle\hat{A}\rangle^2} = 0$$

Eigenstates have zero uncertainty in the corresponding observable.

---

## Connection to the Measurement Postulate

The eigenvalue problem is the mathematical foundation for quantum measurement:

| Mathematical Concept | Physical Meaning |
|---------------------|------------------|
| Eigenvalues $\{a\}$ | Possible measurement outcomes |
| Eigenstates $\{|a\rangle\}$ | States of definite value |
| Spectral decomposition | Resolution into measurement outcomes |
| $|\langle a|\psi\rangle|^2$ | Probability of outcome $a$ |
| $\hat{P}_a = |a\rangle\langle a|$ | "Detector" for outcome $a$ |

Tomorrow's topic on continuous spectra will extend these ideas to position, momentum, and other observables with uncountably many eigenvalues.

---

## Worked Examples

### Example 1: Eigenvalues of Pauli Matrices

**Problem:** Find the eigenvalues and eigenvectors of all three Pauli matrices.

**Solution:**

The Pauli matrices are:

$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**For $\sigma_z$ (already diagonal):**

Eigenvalues: $\lambda_1 = +1$, $\lambda_2 = -1$

Eigenvectors: $|+z\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $|-z\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$

**For $\sigma_x$:**

Characteristic equation:
$$\det\begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix} = \lambda^2 - 1 = 0$$

Eigenvalues: $\lambda = \pm 1$

For $\lambda = +1$: $\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} a \\ b \end{pmatrix} = 0 \Rightarrow a = b$

$$|+x\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

For $\lambda = -1$: $\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} a \\ b \end{pmatrix} = 0 \Rightarrow a = -b$

$$|-x\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

**For $\sigma_y$:**

Characteristic equation:
$$\det\begin{pmatrix} -\lambda & -i \\ i & -\lambda \end{pmatrix} = \lambda^2 - 1 = 0$$

Eigenvalues: $\lambda = \pm 1$

For $\lambda = +1$: $\begin{pmatrix} -1 & -i \\ i & -1 \end{pmatrix}\begin{pmatrix} a \\ b \end{pmatrix} = 0 \Rightarrow a = -ib$

$$|+y\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}$$

For $\lambda = -1$: $a = ib$

$$|-y\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -i \end{pmatrix}$$

**Spectral Decompositions:**

$$\sigma_z = (+1)|+z\rangle\langle+z| + (-1)|-z\rangle\langle-z| = |0\rangle\langle 0| - |1\rangle\langle 1|$$

$$\sigma_x = (+1)|+x\rangle\langle+x| + (-1)|-x\rangle\langle-x| = |+\rangle\langle+| - |-\rangle\langle-|$$

**Key Result:** All Pauli matrices have eigenvalues $\pm 1$. This corresponds to spin measurements yielding $\pm\hbar/2$ (since $\hat{S}_i = \frac{\hbar}{2}\sigma_i$). $\blacksquare$

---

### Example 2: 2D Rotation Matrix

**Problem:** Find the eigenvalues and eigenvectors of the 2D rotation matrix $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.

**Solution:**

**Characteristic equation:**

$$\det(R - \lambda I) = \det\begin{pmatrix} \cos\theta - \lambda & -\sin\theta \\ \sin\theta & \cos\theta - \lambda \end{pmatrix}$$

$$= (\cos\theta - \lambda)^2 + \sin^2\theta = \lambda^2 - 2\lambda\cos\theta + 1 = 0$$

**Eigenvalues:**

$$\lambda = \frac{2\cos\theta \pm \sqrt{4\cos^2\theta - 4}}{2} = \cos\theta \pm i\sin\theta$$

$$\boxed{\lambda_{\pm} = e^{\pm i\theta}}$$

**Key Insight:** Rotation eigenvalues lie on the unit circle! This is expected since rotation preserves lengths (unitary/orthogonal operator).

**Eigenvectors:**

For $\lambda_+ = e^{i\theta}$:

$$\begin{pmatrix} \cos\theta - e^{i\theta} & -\sin\theta \\ \sin\theta & \cos\theta - e^{i\theta} \end{pmatrix}\begin{pmatrix} a \\ b \end{pmatrix} = 0$$

Using $\cos\theta - e^{i\theta} = \cos\theta - \cos\theta - i\sin\theta = -i\sin\theta$:

$$\begin{pmatrix} -i\sin\theta & -\sin\theta \\ \sin\theta & -i\sin\theta \end{pmatrix}\begin{pmatrix} a \\ b \end{pmatrix} = 0$$

From row 1 (assuming $\sin\theta \neq 0$): $-ia - b = 0 \Rightarrow b = -ia$

$$|v_+\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -i \end{pmatrix}$$

Similarly: $|v_-\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}$

**Physical Interpretation:**

The real rotation matrix has complex eigenvalues and eigenvectors! The eigenvectors represent circular polarization states---they rotate without changing direction, only acquiring a phase. $\blacksquare$

---

### Example 3: Simple Harmonic Oscillator Hamiltonian (Truncated)

**Problem:** Consider a truncated harmonic oscillator with basis states $|0\rangle$, $|1\rangle$, $|2\rangle$. The Hamiltonian matrix is:

$$H = \hbar\omega\begin{pmatrix} \frac{1}{2} & 0 & 0 \\ 0 & \frac{3}{2} & 0 \\ 0 & 0 & \frac{5}{2} \end{pmatrix}$$

Find the eigenvalues, eigenstates, and verify the spectral decomposition.

**Solution:**

**Eigenvalues:**

Since $H$ is already diagonal, the eigenvalues are the diagonal elements:

$$E_0 = \frac{1}{2}\hbar\omega, \quad E_1 = \frac{3}{2}\hbar\omega, \quad E_2 = \frac{5}{2}\hbar\omega$$

Or in general: $E_n = (n + \frac{1}{2})\hbar\omega$

**Eigenstates:**

The eigenstates are the standard basis vectors:

$$|E_0\rangle = |0\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \quad |E_1\rangle = |1\rangle = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \quad |E_2\rangle = |2\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$

**Spectral Decomposition:**

$$H = \sum_{n=0}^{2} E_n |n\rangle\langle n| = \frac{\hbar\omega}{2}|0\rangle\langle 0| + \frac{3\hbar\omega}{2}|1\rangle\langle 1| + \frac{5\hbar\omega}{2}|2\rangle\langle 2|$$

**Verification:**

$$|0\rangle\langle 0| = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad |1\rangle\langle 1| = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad |2\rangle\langle 2| = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

$$H = \hbar\omega\left[\frac{1}{2}\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} + \frac{3}{2}\begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix} + \frac{5}{2}\begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}\right] = \hbar\omega\begin{pmatrix} \frac{1}{2} & 0 & 0 \\ 0 & \frac{3}{2} & 0 \\ 0 & 0 & \frac{5}{2} \end{pmatrix}$$ ✓

**Time Evolution:**

A state $|\psi(0)\rangle = c_0|0\rangle + c_1|1\rangle + c_2|2\rangle$ evolves as:

$$|\psi(t)\rangle = c_0 e^{-iE_0 t/\hbar}|0\rangle + c_1 e^{-iE_1 t/\hbar}|1\rangle + c_2 e^{-iE_2 t/\hbar}|2\rangle$$

$$= e^{-i\omega t/2}\left(c_0|0\rangle + c_1 e^{-i\omega t}|1\rangle + c_2 e^{-2i\omega t}|2\rangle\right)$$

The energy eigenbasis makes time evolution transparent! $\blacksquare$

---

## Practice Problems

### Level 1: Direct Application

1. Find the eigenvalues and eigenvectors of $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$. Verify that the eigenvectors are orthogonal.

2. Write the spectral decomposition of $\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

3. If $\hat{A}$ has eigenvalues $\{1, 2, 3\}$ with corresponding eigenstates $\{|1\rangle, |2\rangle, |3\rangle\}$, what are the eigenvalues of $\hat{A}^2$?

### Level 2: Intermediate

4. **Degeneracy Analysis:** The matrix $A = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix}$ has eigenvalue $\lambda = 1$ with algebraic multiplicity 3. Find the geometric multiplicity. Is $A$ diagonalizable?

5. **Function of Operator:** Given $\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, compute $e^{i\theta\sigma_x}$ using the spectral decomposition.

6. **Measurement Probabilities:** A spin-1/2 particle is in state $|\psi\rangle = \frac{1}{\sqrt{3}}|+z\rangle + \sqrt{\frac{2}{3}}|-z\rangle$. Find the probability of measuring $S_x = +\hbar/2$.

### Level 3: Challenging

7. **Simultaneous Eigenstates:** Prove that two Hermitian operators $\hat{A}$ and $\hat{B}$ have a common eigenbasis if and only if $[\hat{A}, \hat{B}] = 0$.

8. **Spectral Theorem Application:** For a Hermitian operator $\hat{A}$ with spectral decomposition $\hat{A} = \sum_a a|a\rangle\langle a|$, prove that $f(\hat{A})g(\hat{A}) = g(\hat{A})f(\hat{A})$ for any functions $f, g$.

9. **Research Problem:** The Hamiltonian $H = \begin{pmatrix} E_0 & V \\ V & E_0 \end{pmatrix}$ describes a two-level system with coupling $V$. Find the energy eigenvalues and eigenstates. Show that when $V \ll E_0$, perturbation theory gives the same result to leading order.

---

## Computational Lab

```python
"""
Day 341 Computational Lab: Eigenvalue Problems in Quantum Mechanics
Year 1 - Quantum Mechanics Core

Topics:
1. Eigenvalue decomposition
2. Spectral decomposition verification
3. Functions of operators
4. Visualization of eigenspaces
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("=" * 70)
print("Day 341: Eigenvalue Problems - Computational Lab")
print("=" * 70)

# =============================================================================
# Part 1: Pauli Matrix Eigenanalysis
# =============================================================================

print("\n--- Part 1: Pauli Matrix Eigenanalysis ---")

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

paulis = {'sigma_x': sigma_x, 'sigma_y': sigma_y, 'sigma_z': sigma_z}

for name, sigma in paulis.items():
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    print(f"\n{name}:")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Eigenvector 1: {eigenvectors[:, 0]}")
    print(f"  Eigenvector 2: {eigenvectors[:, 1]}")

    # Verify spectral decomposition
    P1 = np.outer(eigenvectors[:, 0], eigenvectors[:, 0].conj())
    P2 = np.outer(eigenvectors[:, 1], eigenvectors[:, 1].conj())
    reconstructed = eigenvalues[0] * P1 + eigenvalues[1] * P2

    print(f"  Spectral decomposition verified: {np.allclose(reconstructed, sigma)}")

# =============================================================================
# Part 2: Characteristic Polynomial and Secular Equation
# =============================================================================

print("\n--- Part 2: Characteristic Polynomial ---")

def characteristic_polynomial(A):
    """
    Compute characteristic polynomial coefficients.
    For 2x2: det(A - lambda*I) = lambda^2 - Tr(A)*lambda + det(A)
    """
    n = A.shape[0]
    if n == 2:
        trace = np.trace(A)
        det = np.linalg.det(A)
        return np.array([1, -trace, det])
    else:
        # General case: use numpy's polynomial fitting
        return np.poly(A)

# Example: Generic 2x2 Hermitian matrix
A = np.array([[3, 1-1j], [1+1j, 2]], dtype=complex)
print(f"\nMatrix A = \n{A}")
print(f"Is Hermitian: {np.allclose(A, A.conj().T)}")

coeffs = characteristic_polynomial(A)
print(f"\nCharacteristic polynomial: lambda^2 + ({coeffs[1]:.4f})*lambda + ({coeffs[2]:.4f})")

# Solve using quadratic formula
trace_A = np.trace(A)
det_A = np.linalg.det(A)
lambda_plus = (trace_A + np.sqrt(trace_A**2 - 4*det_A)) / 2
lambda_minus = (trace_A - np.sqrt(trace_A**2 - 4*det_A)) / 2

print(f"\nEigenvalues from secular equation:")
print(f"  lambda_+ = {lambda_plus:.6f}")
print(f"  lambda_- = {lambda_minus:.6f}")

# Verify with numpy
eigenvalues_np, _ = np.linalg.eigh(A)
print(f"\nEigenvalues from numpy: {eigenvalues_np}")

# =============================================================================
# Part 3: Degeneracy Example
# =============================================================================

print("\n--- Part 3: Degeneracy Analysis ---")

# Matrix with degenerate eigenvalues (Jordan block - not diagonalizable)
B_jordan = np.array([[1, 1, 0],
                     [0, 1, 1],
                     [0, 0, 1]], dtype=complex)

# Matrix with degenerate eigenvalues (diagonalizable)
B_diag = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 2]], dtype=complex)

for name, B in [("Jordan block (not diagonalizable)", B_jordan),
                ("Diagonal (diagonalizable)", B_diag)]:
    print(f"\n{name}:")
    eigenvalues, eigenvectors = np.linalg.eig(B)
    print(f"  Eigenvalues: {eigenvalues}")

    # Check algebraic multiplicity
    unique, counts = np.unique(np.round(eigenvalues, 10), return_counts=True)
    print(f"  Algebraic multiplicities: {dict(zip(unique, counts))}")

    # Check if diagonalizable (eigenvectors span space)
    rank = np.linalg.matrix_rank(eigenvectors)
    print(f"  Eigenvector matrix rank: {rank} (need {B.shape[0]} for diagonalizable)")

# =============================================================================
# Part 4: Functions of Operators
# =============================================================================

print("\n--- Part 4: Functions of Operators ---")

def function_of_operator(A, f):
    """
    Compute f(A) using spectral decomposition.
    f(A) = sum_i f(lambda_i) |v_i><v_i|
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    result = np.zeros_like(A)

    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        projector = np.outer(vec, vec.conj())
        result += f(lam) * projector

    return result

# Compute exp(i*theta*sigma_x) for various theta
print("\nExponential of Pauli-X: exp(i*theta*sigma_x)")

theta_test = np.pi / 4
exp_sigma_x = function_of_operator(sigma_x, lambda x: np.exp(1j * theta_test * x))

# Verify using the known formula: exp(i*theta*sigma_x) = cos(theta)*I + i*sin(theta)*sigma_x
expected = np.cos(theta_test) * I + 1j * np.sin(theta_test) * sigma_x

print(f"\nFor theta = pi/4:")
print(f"exp(i*theta*sigma_x) via spectral decomposition:")
print(exp_sigma_x)
print(f"\ncos(theta)*I + i*sin(theta)*sigma_x:")
print(expected)
print(f"\nMatch: {np.allclose(exp_sigma_x, expected)}")

# =============================================================================
# Part 5: Measurement Probabilities
# =============================================================================

print("\n--- Part 5: Measurement Probabilities ---")

# State: |psi> = (1/sqrt(3))|+z> + sqrt(2/3)|-z>
psi = np.array([[1/np.sqrt(3)], [np.sqrt(2/3)]], dtype=complex)

print(f"|psi> = {psi.flatten()}")
print(f"Normalization: {np.vdot(psi, psi).real:.6f}")

# Probability of measuring S_z = +hbar/2 (eigenvalue +1 of sigma_z)
P_plus_z = np.abs(psi[0, 0])**2
P_minus_z = np.abs(psi[1, 0])**2
print(f"\nS_z measurement:")
print(f"  P(+hbar/2) = {P_plus_z:.6f}")
print(f"  P(-hbar/2) = {P_minus_z:.6f}")

# Probability of measuring S_x = +hbar/2
# Need eigenvectors of sigma_x
eigenvalues_x, eigenvectors_x = np.linalg.eigh(sigma_x)
plus_x = eigenvectors_x[:, 1:2]  # eigenvalue +1
minus_x = eigenvectors_x[:, 0:1]  # eigenvalue -1

P_plus_x = np.abs(np.vdot(plus_x, psi))**2
P_minus_x = np.abs(np.vdot(minus_x, psi))**2

print(f"\nS_x measurement:")
print(f"  |+x> = {plus_x.flatten()}")
print(f"  P(+hbar/2) = {P_plus_x:.6f}")
print(f"  P(-hbar/2) = {P_minus_x:.6f}")

# =============================================================================
# Part 6: Visualization - Eigenspaces on Bloch Sphere
# =============================================================================

print("\n--- Part 6: Visualization ---")

def bloch_vector(state):
    """Convert a qubit state to Bloch sphere coordinates."""
    rho = np.outer(state, state.conj())
    x = np.real(np.trace(rho @ sigma_x))
    y = np.real(np.trace(rho @ sigma_y))
    z = np.real(np.trace(rho @ sigma_z))
    return x, y, z

# Create Bloch sphere
fig = plt.figure(figsize=(12, 5))

# Plot 1: Eigenstates of Pauli matrices
ax1 = fig.add_subplot(121, projection='3d')

# Draw sphere wireframe
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

# Plot eigenstates
eigenstates = {
    '|+z>': np.array([1, 0]),
    '|-z>': np.array([0, 1]),
    '|+x>': np.array([1, 1]) / np.sqrt(2),
    '|-x>': np.array([1, -1]) / np.sqrt(2),
    '|+y>': np.array([1, 1j]) / np.sqrt(2),
    '|-y>': np.array([1, -1j]) / np.sqrt(2),
}

colors = {'z': 'blue', 'x': 'red', 'y': 'green'}
for name, state in eigenstates.items():
    bx, by, bz = bloch_vector(state)
    axis = name[2]
    ax1.scatter([bx], [by], [bz], color=colors[axis], s=100)
    ax1.text(bx * 1.2, by * 1.2, bz * 1.2, name, fontsize=8)

# Draw axes
ax1.quiver(0, 0, 0, 1.3, 0, 0, color='red', alpha=0.5, arrow_length_ratio=0.1)
ax1.quiver(0, 0, 0, 0, 1.3, 0, color='green', alpha=0.5, arrow_length_ratio=0.1)
ax1.quiver(0, 0, 0, 0, 0, 1.3, color='blue', alpha=0.5, arrow_length_ratio=0.1)
ax1.text(1.4, 0, 0, 'x', fontsize=10)
ax1.text(0, 1.4, 0, 'y', fontsize=10)
ax1.text(0, 0, 1.4, 'z', fontsize=10)

ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.5, 1.5])
ax1.set_zlim([-1.5, 1.5])
ax1.set_title('Eigenstates of Pauli Matrices on Bloch Sphere')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Plot 2: Eigenvalue spectrum
ax2 = fig.add_subplot(122)

# Eigenvalues of various operators
operators = {
    r'$\sigma_z$': [1, -1],
    r'$\sigma_x$': [1, -1],
    r'$H_{SHO}$ (3-level)': [0.5, 1.5, 2.5],
    r'$J_z$ (j=1)': [-1, 0, 1],
}

y_pos = 0
for name, eigs in operators.items():
    ax2.scatter(eigs, [y_pos] * len(eigs), s=200, marker='|', linewidths=3)
    ax2.text(-2.5, y_pos, name, fontsize=10, va='center')
    y_pos += 1

ax2.set_xlim([-3, 3])
ax2.set_ylim([-0.5, len(operators) - 0.5])
ax2.set_xlabel('Eigenvalue', fontsize=12)
ax2.set_title('Eigenvalue Spectra of Quantum Operators')
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_yticks([])

plt.tight_layout()
plt.savefig('day_341_eigenvalue_problems.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_341_eigenvalue_problems.png'")

# =============================================================================
# Part 7: Time Evolution via Spectral Decomposition
# =============================================================================

print("\n--- Part 7: Time Evolution ---")

# Two-level system Hamiltonian
omega = 1.0  # Natural frequency
H = np.array([[1, 0], [0, -1]], dtype=complex) * omega / 2  # H = (omega/2) * sigma_z

# Initial state: |+x>
psi_0 = np.array([[1], [1]], dtype=complex) / np.sqrt(2)

# Time evolution
def evolve(psi_0, H, t):
    """Evolve state psi_0 under Hamiltonian H for time t."""
    U = function_of_operator(H, lambda E: np.exp(-1j * E * t))
    return U @ psi_0

# Evolve and track Bloch vector
times = np.linspace(0, 4 * np.pi / omega, 100)
bloch_x = []
bloch_y = []
bloch_z = []

for t in times:
    psi_t = evolve(psi_0, H, t)
    bx, by, bz = bloch_vector(psi_t.flatten())
    bloch_x.append(bx)
    bloch_y.append(by)
    bloch_z.append(bz)

# Plot time evolution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bloch vector components vs time
ax1 = axes[0]
ax1.plot(times * omega / (2 * np.pi), bloch_x, 'r-', label=r'$\langle\sigma_x\rangle$')
ax1.plot(times * omega / (2 * np.pi), bloch_y, 'g-', label=r'$\langle\sigma_y\rangle$')
ax1.plot(times * omega / (2 * np.pi), bloch_z, 'b-', label=r'$\langle\sigma_z\rangle$')
ax1.set_xlabel(r'Time ($\omega t / 2\pi$)', fontsize=12)
ax1.set_ylabel('Expectation Value', fontsize=12)
ax1.set_title(r'Time Evolution: $|\psi_0\rangle = |+x\rangle$, $H = \frac{\omega}{2}\sigma_z$')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Probabilities vs time
ax2 = axes[1]
P_plus_z_t = [(1 + bz) / 2 for bz in bloch_z]
P_minus_z_t = [(1 - bz) / 2 for bz in bloch_z]
ax2.plot(times * omega / (2 * np.pi), P_plus_z_t, 'b-', label=r'$P(|+z\rangle)$')
ax2.plot(times * omega / (2 * np.pi), P_minus_z_t, 'r--', label=r'$P(|-z\rangle)$')
ax2.set_xlabel(r'Time ($\omega t / 2\pi$)', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Measurement Probabilities Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('day_341_time_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_341_time_evolution.png'")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Eigenvalue equation | $\hat{A}\|a\rangle = a\|a\rangle$ |
| Secular equation | $\det(\hat{A} - \lambda\hat{I}) = 0$ |
| 2x2 eigenvalues | $\lambda = \frac{\text{Tr}(A) \pm \sqrt{\text{Tr}(A)^2 - 4\det(A)}}{2}$ |
| Spectral decomposition | $\hat{A} = \sum_a a \|a\rangle\langle a\|$ |
| Function of operator | $f(\hat{A}) = \sum_a f(a) \|a\rangle\langle a\|$ |
| Measurement probability | $P(a) = \|\langle a\|\psi\rangle\|^2$ |
| Expectation value | $\langle\hat{A}\rangle = \sum_a a \cdot P(a)$ |

### Main Takeaways

1. **Eigenvalues are measurement outcomes:** The spectrum of an observable determines what values can be measured.

2. **Eigenstates are states of certainty:** In an eigenstate, the corresponding observable has definite value with zero uncertainty.

3. **Spectral decomposition is fundamental:** It connects the abstract operator to its measurable properties and enables computation of operator functions.

4. **Degeneracy reflects symmetry:** When multiple states share an eigenvalue, there's often an underlying symmetry.

5. **Diagonalization simplifies everything:** In the eigenbasis, operators become diagonal and time evolution becomes trivial.

6. **Functions of operators via eigenvalues:** The spectral theorem allows us to compute $f(\hat{A})$ by applying $f$ to each eigenvalue.

---

## References

- **Shankar, R.** *Principles of Quantum Mechanics* (2nd ed.), Chapter 1.9: The Eigenvalue Problem
- **Sakurai, J.J. & Napolitano, J.** *Modern Quantum Mechanics* (2nd ed.), Chapter 1.4: Matrix Representations
- **Cohen-Tannoudji, C.** *Quantum Mechanics*, Vol. 1, Complement A_II

---

## Daily Checklist

- [ ] Read Shankar Chapter 1.9
- [ ] Read Sakurai Chapter 1.4
- [ ] Solve the characteristic equation for at least 3 different 2x2 matrices
- [ ] Verify spectral decomposition for the Pauli matrices
- [ ] Complete Level 1 and Level 2 practice problems
- [ ] Run the computational lab and understand each section
- [ ] Explain in your own words why eigenvalues are measurement outcomes
- [ ] Compute $e^{i\theta\sigma_y}$ using spectral decomposition

---

## Preview: Day 342

Tomorrow we tackle **continuous spectra**---what happens when operators like position $\hat{x}$ and momentum $\hat{p}$ have uncountably many eigenvalues? We'll meet Dirac delta functions, generalized eigenvectors, and the position-momentum uncertainty relation in its full glory.

---

*"The eigenvalues of an observable are the only possible results of a measurement of that observable."*
--- Third Postulate of Quantum Mechanics

---

**Next:** [Day_342_Saturday.md](Day_342_Saturday.md) --- Continuous Spectra
