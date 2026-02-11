# Day 359: The Time Evolution Operator — Propagating Quantum States

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Time Evolution Operator |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 359, you will be able to:

1. Define the time evolution operator and derive its explicit form
2. Prove that the time evolution operator is unitary
3. Verify the composition law for time evolution
4. Distinguish time-independent from time-dependent Hamiltonians
5. Expand the evolution operator in the energy eigenbasis
6. Connect the time evolution operator to quantum gates

---

## Core Content

### 1. The Formal Solution to the Schrodinger Equation

Yesterday we wrote down the Schrodinger equation:
$$i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle$$

For a **time-independent** Hamiltonian, this first-order linear equation has a formal solution:

$$\boxed{|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle}$$

We define the **time evolution operator** (or **propagator**):

$$\boxed{\hat{U}(t) = e^{-i\hat{H}t/\hbar}}$$

such that:
$$|\psi(t)\rangle = \hat{U}(t)|\psi(0)\rangle$$

---

### 2. What Does the Exponential of an Operator Mean?

The exponential of an operator is defined by the Taylor series:

$$e^{\hat{A}} = \sum_{n=0}^{\infty} \frac{\hat{A}^n}{n!} = \hat{I} + \hat{A} + \frac{\hat{A}^2}{2!} + \frac{\hat{A}^3}{3!} + \cdots$$

For the time evolution operator:
$$\hat{U}(t) = e^{-i\hat{H}t/\hbar} = \sum_{n=0}^{\infty} \frac{1}{n!}\left(\frac{-i\hat{H}t}{\hbar}\right)^n$$

**Important Properties of Operator Exponentials:**

1. $e^{\hat{A}}e^{\hat{B}} = e^{\hat{A}+\hat{B}}$ **only if** $[\hat{A}, \hat{B}] = 0$
2. $(e^{\hat{A}})^{-1} = e^{-\hat{A}}$
3. $(e^{\hat{A}})^\dagger = e^{\hat{A}^\dagger}$
4. $\frac{d}{dt}e^{\hat{A}t} = \hat{A}e^{\hat{A}t} = e^{\hat{A}t}\hat{A}$

---

### 3. Verification: $\hat{U}(t)$ Satisfies the Schrodinger Equation

Let us verify that $|\psi(t)\rangle = \hat{U}(t)|\psi(0)\rangle$ satisfies the Schrodinger equation.

**Compute the time derivative:**
$$\frac{\partial}{\partial t}|\psi(t)\rangle = \frac{\partial}{\partial t}\left(e^{-i\hat{H}t/\hbar}|\psi(0)\rangle\right)$$

Using $\frac{d}{dt}e^{\hat{A}t} = \hat{A}e^{\hat{A}t}$:
$$= \left(\frac{-i\hat{H}}{\hbar}\right)e^{-i\hat{H}t/\hbar}|\psi(0)\rangle = \frac{-i\hat{H}}{\hbar}|\psi(t)\rangle$$

**Rearranging:**
$$i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle \quad \checkmark$$

The Schrodinger equation is satisfied.

---

### 4. Unitarity of the Time Evolution Operator

A unitary operator $\hat{U}$ satisfies:
$$\hat{U}^\dagger \hat{U} = \hat{U}\hat{U}^\dagger = \hat{I}$$

**Proof that $\hat{U}(t)$ is unitary:**

Since $\hat{H}$ is Hermitian ($\hat{H}^\dagger = \hat{H}$):
$$\hat{U}^\dagger(t) = \left(e^{-i\hat{H}t/\hbar}\right)^\dagger = e^{i\hat{H}^\dagger t/\hbar} = e^{i\hat{H}t/\hbar}$$

Therefore:
$$\hat{U}^\dagger(t)\hat{U}(t) = e^{i\hat{H}t/\hbar}e^{-i\hat{H}t/\hbar}$$

Since $[\hat{H}, \hat{H}] = 0$, we can combine exponentials:
$$= e^{i\hat{H}t/\hbar - i\hat{H}t/\hbar} = e^{0} = \hat{I}$$

Similarly, $\hat{U}(t)\hat{U}^\dagger(t) = \hat{I}$.

$$\boxed{\hat{U}^\dagger(t)\hat{U}(t) = \hat{I} \implies \text{unitarity}}$$

**Physical Consequence:**
Unitarity implies probability conservation:
$$\langle\psi(t)|\psi(t)\rangle = \langle\psi(0)|\hat{U}^\dagger(t)\hat{U}(t)|\psi(0)\rangle = \langle\psi(0)|\psi(0)\rangle$$

---

### 5. Properties of the Time Evolution Operator

**Property 1: Identity at $t=0$**
$$\hat{U}(0) = e^{0} = \hat{I}$$

**Property 2: Composition Law**
$$\boxed{\hat{U}(t_2)\hat{U}(t_1) = \hat{U}(t_1 + t_2)}$$

*Proof:*
$$\hat{U}(t_2)\hat{U}(t_1) = e^{-i\hat{H}t_2/\hbar}e^{-i\hat{H}t_1/\hbar} = e^{-i\hat{H}(t_1+t_2)/\hbar} = \hat{U}(t_1+t_2)$$

This is called the **group property** — the time evolution operators form a one-parameter group.

**Property 3: Inverse**
$$\hat{U}^{-1}(t) = \hat{U}(-t) = \hat{U}^\dagger(t) = e^{i\hat{H}t/\hbar}$$

**Property 4: Continuity**
$$\lim_{t \to 0} \hat{U}(t) = \hat{I}$$

**Property 5: Time Evolution Between Arbitrary Times**
$$|\psi(t_2)\rangle = \hat{U}(t_2, t_1)|\psi(t_1)\rangle$$

where for time-independent H:
$$\hat{U}(t_2, t_1) = \hat{U}(t_2 - t_1) = e^{-i\hat{H}(t_2-t_1)/\hbar}$$

---

### 6. Expansion in Energy Eigenbasis

The most useful representation of $\hat{U}(t)$ uses the energy eigenstates.

Let $\{|E_n\rangle\}$ be the eigenstates of $\hat{H}$:
$$\hat{H}|E_n\rangle = E_n|E_n\rangle$$

Using the completeness relation $\hat{I} = \sum_n |E_n\rangle\langle E_n|$:

$$\hat{U}(t) = e^{-i\hat{H}t/\hbar} = \sum_n e^{-i\hat{H}t/\hbar}|E_n\rangle\langle E_n|$$

Since $|E_n\rangle$ is an eigenstate of $\hat{H}$:
$$e^{-i\hat{H}t/\hbar}|E_n\rangle = e^{-iE_n t/\hbar}|E_n\rangle$$

Therefore:
$$\boxed{\hat{U}(t) = \sum_n e^{-iE_n t/\hbar}|E_n\rangle\langle E_n|}$$

This is the **spectral decomposition** of the time evolution operator.

**Action on an arbitrary state:**
$$|\psi(t)\rangle = \hat{U}(t)|\psi(0)\rangle = \sum_n e^{-iE_n t/\hbar}|E_n\rangle\langle E_n|\psi(0)\rangle = \sum_n c_n e^{-iE_n t/\hbar}|E_n\rangle$$

where $c_n = \langle E_n|\psi(0)\rangle$.

---

### 7. Time-Dependent Hamiltonians

When $\hat{H} = \hat{H}(t)$ depends on time, the situation is more complex.

**The Simple Exponential Fails:**
$$|\psi(t)\rangle \neq e^{-\frac{i}{\hbar}\int_0^t \hat{H}(t')dt'}|\psi(0)\rangle \quad \text{(generally)}$$

This only works if $[\hat{H}(t_1), \hat{H}(t_2)] = 0$ for all times.

**The Correct Solution: Time-Ordered Exponential**
$$\hat{U}(t) = \mathcal{T}\exp\left(-\frac{i}{\hbar}\int_0^t \hat{H}(t')dt'\right)$$

where $\mathcal{T}$ is the **time-ordering operator** that places later times to the left.

**Dyson Series:**
$$\hat{U}(t) = \hat{I} + \sum_{n=1}^{\infty}\left(\frac{-i}{\hbar}\right)^n \int_0^t dt_1 \int_0^{t_1} dt_2 \cdots \int_0^{t_{n-1}} dt_n \, \hat{H}(t_1)\hat{H}(t_2)\cdots\hat{H}(t_n)$$

This is essential for perturbation theory and the Interaction picture (Day 363).

---

### 8. The Infinitesimal Time Evolution

For small time $\epsilon$:
$$\hat{U}(\epsilon) = e^{-i\hat{H}\epsilon/\hbar} \approx \hat{I} - \frac{i\hat{H}\epsilon}{\hbar} + O(\epsilon^2)$$

This shows that $\hat{H}$ is the **generator of time translations**:
$$\hat{U}(\epsilon) = \hat{I} - \frac{i\epsilon}{\hbar}\hat{H}$$

Compare to spatial translations generated by momentum:
$$\hat{T}(a) = e^{-i\hat{p}a/\hbar} \approx \hat{I} - \frac{ia}{\hbar}\hat{p}$$

---

## Physical Interpretation

### The Hamiltonian as Generator

The relationship between symmetries and conservation laws (Noether's theorem) takes a beautiful form in QM:

| Symmetry | Generator | Conserved Quantity |
|----------|-----------|-------------------|
| Time translation | $\hat{H}$ | Energy |
| Space translation | $\hat{p}$ | Momentum |
| Rotation | $\hat{L}$ | Angular momentum |

The generator of a symmetry transformation is also the observable whose conservation that symmetry implies.

### Unitary Evolution and Determinism

Unitary evolution means:
1. **Probability Conservation:** Total probability remains 1
2. **Reversibility:** $\hat{U}^{-1}$ exists; we can run time backwards
3. **Determinism:** The state at any time determines the state at all times

This contrasts with measurement, which is non-unitary and irreversible.

### The Two Processes in Quantum Mechanics

| Process | Mathematical | Physical |
|---------|--------------|----------|
| Evolution | Unitary: $\hat{U}\|\psi\rangle$ | Deterministic, continuous |
| Measurement | Projection: $\hat{P}_a\|\psi\rangle$ | Probabilistic, discontinuous |

---

## Quantum Computing Connection

### Quantum Gates as Time Evolution Operators

Every quantum gate is a unitary operator, and every unitary can be written as $e^{-i\hat{H}t}$ for some Hermitian $\hat{H}$.

**Example: Pauli-X Gate (NOT)**
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \sigma_x$$

We can write $X = e^{-i\pi\sigma_x/2}$ since:
$$e^{-i\theta\sigma_x/2} = \cos(\theta/2)\hat{I} - i\sin(\theta/2)\sigma_x$$

Setting $\theta = \pi$: $\cos(\pi/2) = 0$, $\sin(\pi/2) = 1$:
$$e^{-i\pi\sigma_x/2} = -i\sigma_x$$

Up to a global phase, this is $\sigma_x = X$.

**Example: Hadamard Gate**
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = e^{i\pi/2}e^{-i\pi(\sigma_x + \sigma_z)/(2\sqrt{2})}$$

**Example: CNOT Gate**
$$\text{CNOT} = e^{-i\frac{\pi}{4}(I - Z) \otimes (I - X)}$$

### Gate Composition

The composition law $\hat{U}(t_2)\hat{U}(t_1) = \hat{U}(t_1 + t_2)$ underlies quantum circuits:

Applying gate $\hat{U}_1$ then $\hat{U}_2$ gives total evolution $\hat{U}_2\hat{U}_1$.

**Note:** In circuit notation, gates are applied left-to-right, but in equations, operators act right-to-left.

### Universal Gate Sets

Any unitary can be approximated by products from a universal gate set (e.g., CNOT + single-qubit rotations). This is the quantum analog of building arbitrary time evolution from simple steps.

---

## Worked Examples

### Example 1: Two-Level System Evolution

**Problem:** For $\hat{H} = \frac{\hbar\omega}{2}\sigma_z$, find $\hat{U}(t)$ explicitly as a 2x2 matrix.

**Solution:**

The Hamiltonian in matrix form:
$$\hat{H} = \frac{\hbar\omega}{2}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

The time evolution operator:
$$\hat{U}(t) = e^{-i\hat{H}t/\hbar} = e^{-i(\omega t/2)\sigma_z}$$

For diagonal matrices, the exponential acts element-wise:
$$\hat{U}(t) = \begin{pmatrix} e^{-i\omega t/2} & 0 \\ 0 & e^{i\omega t/2} \end{pmatrix}$$

**Verify unitarity:**
$$\hat{U}^\dagger(t) = \begin{pmatrix} e^{i\omega t/2} & 0 \\ 0 & e^{-i\omega t/2} \end{pmatrix}$$

$$\hat{U}^\dagger(t)\hat{U}(t) = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \hat{I} \quad \checkmark$$

$$\boxed{\hat{U}(t) = \begin{pmatrix} e^{-i\omega t/2} & 0 \\ 0 & e^{i\omega t/2} \end{pmatrix}}$$ ∎

---

### Example 2: Verifying the Composition Law

**Problem:** Show explicitly that $\hat{U}(t_1)\hat{U}(t_2) = \hat{U}(t_1 + t_2)$ for the two-level system in Example 1.

**Solution:**

$$\hat{U}(t_1) = \begin{pmatrix} e^{-i\omega t_1/2} & 0 \\ 0 & e^{i\omega t_1/2} \end{pmatrix}, \quad \hat{U}(t_2) = \begin{pmatrix} e^{-i\omega t_2/2} & 0 \\ 0 & e^{i\omega t_2/2} \end{pmatrix}$$

$$\hat{U}(t_1)\hat{U}(t_2) = \begin{pmatrix} e^{-i\omega t_1/2}e^{-i\omega t_2/2} & 0 \\ 0 & e^{i\omega t_1/2}e^{i\omega t_2/2} \end{pmatrix}$$

$$= \begin{pmatrix} e^{-i\omega(t_1+t_2)/2} & 0 \\ 0 & e^{i\omega(t_1+t_2)/2} \end{pmatrix} = \hat{U}(t_1 + t_2) \quad \checkmark$$ ∎

---

### Example 3: Evolution of Superposition

**Problem:** Using the spectral decomposition of $\hat{U}(t)$, evolve the initial state $|\psi(0)\rangle = \frac{1}{\sqrt{3}}|E_1\rangle + \sqrt{\frac{2}{3}}|E_2\rangle$ where $E_1 = \hbar\omega$ and $E_2 = 2\hbar\omega$.

**Solution:**

The spectral decomposition of the evolution operator:
$$\hat{U}(t) = e^{-iE_1 t/\hbar}|E_1\rangle\langle E_1| + e^{-iE_2 t/\hbar}|E_2\rangle\langle E_2|$$

Acting on $|\psi(0)\rangle$:
$$|\psi(t)\rangle = e^{-iE_1 t/\hbar}|E_1\rangle\langle E_1|\psi(0)\rangle + e^{-iE_2 t/\hbar}|E_2\rangle\langle E_2|\psi(0)\rangle$$

With $\langle E_1|\psi(0)\rangle = 1/\sqrt{3}$ and $\langle E_2|\psi(0)\rangle = \sqrt{2/3}$:

$$|\psi(t)\rangle = \frac{1}{\sqrt{3}}e^{-i\omega t}|E_1\rangle + \sqrt{\frac{2}{3}}e^{-2i\omega t}|E_2\rangle$$

$$\boxed{|\psi(t)\rangle = \frac{e^{-i\omega t}}{\sqrt{3}}\left(|E_1\rangle + \sqrt{2}e^{-i\omega t}|E_2\rangle\right)}$$

The relative phase oscillates at frequency $\omega$, while a global phase $e^{-i\omega t}$ is physically irrelevant. ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Matrix Exponential:** For $\hat{H} = E_0\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ (proportional to identity), find $\hat{U}(t)$.

2. **Unitarity Check:** Show that $\hat{U} = e^{i\theta\sigma_y}$ is unitary for any real $\theta$.

3. **Composition:** If $\hat{U}_1 = e^{-i\theta_1\sigma_z}$ and $\hat{U}_2 = e^{-i\theta_2\sigma_z}$, find $\hat{U}_1\hat{U}_2$.

### Level 2: Intermediate

4. **Pauli Matrix Exponential:** Use the identity $e^{-i\theta\hat{n}\cdot\vec{\sigma}/2} = \cos(\theta/2)\hat{I} - i\sin(\theta/2)\hat{n}\cdot\vec{\sigma}$ to find $e^{-i\pi\sigma_y/4}$ explicitly.

5. **Energy Basis Evolution:** A system has energies $E_n = n^2 E_0$ for $n = 1, 2, 3, \ldots$. Write $\hat{U}(t)$ in the energy basis and find the shortest time $T > 0$ for which $\hat{U}(T) = \hat{I}$ (if it exists).

6. **Three-Level System:** For $\hat{H} = \hbar\omega\begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 2 \end{pmatrix}$, compute $\hat{U}(t)$ and verify $\hat{U}(2\pi/\omega) = \hat{I}$.

### Level 3: Challenging

7. **Non-Commuting Exponentials:** Show that for general non-commuting operators $\hat{A}$ and $\hat{B}$:
$$e^{\hat{A}}e^{\hat{B}} = e^{\hat{A}+\hat{B}+\frac{1}{2}[\hat{A},\hat{B}]+\cdots}$$
(Baker-Campbell-Hausdorff formula, first terms)

8. **Time-Dependent Case:** For $\hat{H}(t) = \hat{H}_0 + f(t)\hat{V}$ where $[\hat{H}_0, \hat{V}] \neq 0$, show that the simple exponential $e^{-\frac{i}{\hbar}\int_0^t \hat{H}(t')dt'}$ does NOT generally give the correct evolution.

9. **Propagator in Position Space:** For a free particle, show that:
$$\langle x'|\hat{U}(t)|x\rangle = \sqrt{\frac{m}{2\pi i\hbar t}}\exp\left[\frac{im(x'-x)^2}{2\hbar t}\right]$$

---

## Computational Lab

### Objective
Implement the time evolution operator and explore its properties numerically.

```python
"""
Day 359 Computational Lab: Time Evolution Operator
Quantum Mechanics Core - Year 1

This lab explores the time evolution operator U(t) = exp(-iHt/hbar),
its unitarity, composition law, and spectral decomposition.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh

# =============================================================================
# Part 1: Basic Properties of the Time Evolution Operator
# =============================================================================

print("=" * 60)
print("Part 1: Time Evolution Operator for Two-Level System")
print("=" * 60)

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# Hamiltonian: H = (hbar*omega/2) * sigma_z
# Set hbar = 1, omega = 1
hbar = 1.0
omega = 1.0
H = (hbar * omega / 2) * sigma_z

print(f"\nHamiltonian H = (hbar*omega/2) * sigma_z:")
print(H)

def time_evolution_operator(H, t, hbar=1.0):
    """Compute U(t) = exp(-i*H*t/hbar)"""
    return expm(-1j * H * t / hbar)

# Compute U(t) for several times
times = [0, np.pi/(2*omega), np.pi/omega, 2*np.pi/omega]
print("\nTime evolution operators at key times:")
for t in times:
    U = time_evolution_operator(H, t, hbar)
    print(f"\nt = {t:.4f}:")
    print(f"U(t) = ")
    print(np.round(U, 4))

# =============================================================================
# Part 2: Verify Unitarity
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Verify Unitarity: U^dagger * U = I")
print("=" * 60)

t_test = 1.5
U = time_evolution_operator(H, t_test, hbar)
U_dagger = U.conj().T

print(f"\nAt t = {t_test}:")
print("U(t) =")
print(np.round(U, 6))
print("\nU^dagger(t) =")
print(np.round(U_dagger, 6))
print("\nU^dagger * U =")
print(np.round(U_dagger @ U, 10))
print("\nIs unitary:", np.allclose(U_dagger @ U, I))

# =============================================================================
# Part 3: Verify Composition Law: U(t1) * U(t2) = U(t1 + t2)
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Verify Composition Law: U(t1)*U(t2) = U(t1+t2)")
print("=" * 60)

t1, t2 = 0.7, 1.3
U_t1 = time_evolution_operator(H, t1, hbar)
U_t2 = time_evolution_operator(H, t2, hbar)
U_sum = time_evolution_operator(H, t1 + t2, hbar)

print(f"\nt1 = {t1}, t2 = {t2}")
print("\nU(t1) * U(t2) =")
print(np.round(U_t1 @ U_t2, 6))
print("\nU(t1 + t2) =")
print(np.round(U_sum, 6))
print("\nComposition law holds:", np.allclose(U_t1 @ U_t2, U_sum))

# =============================================================================
# Part 4: Spectral Decomposition
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Spectral Decomposition of U(t)")
print("=" * 60)

# Get eigenvalues and eigenvectors of H
eigenvalues, eigenvectors = eigh(H)
print("\nEnergy eigenvalues:", eigenvalues)
print("Eigenvectors (columns):")
print(eigenvectors)

# Construct U(t) from spectral decomposition
def U_spectral(t, eigenvalues, eigenvectors, hbar=1.0):
    """Construct U(t) using spectral decomposition."""
    U = np.zeros((len(eigenvalues), len(eigenvalues)), dtype=complex)
    for n, E_n in enumerate(eigenvalues):
        v_n = eigenvectors[:, n:n+1]  # Column vector
        U += np.exp(-1j * E_n * t / hbar) * (v_n @ v_n.conj().T)
    return U

# Compare with matrix exponential
t_test = 2.5
U_expm = time_evolution_operator(H, t_test, hbar)
U_spec = U_spectral(t_test, eigenvalues, eigenvectors, hbar)

print(f"\nAt t = {t_test}:")
print("\nU(t) from expm:")
print(np.round(U_expm, 6))
print("\nU(t) from spectral decomposition:")
print(np.round(U_spec, 6))
print("\nMethods agree:", np.allclose(U_expm, U_spec))

# =============================================================================
# Part 5: Time Evolution of States
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Time Evolution of Quantum States")
print("=" * 60)

# Initial state: |+x> = (|0> + |1>)/sqrt(2)
psi_0 = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
print("\nInitial state |+x> = (|0> + |1>)/sqrt(2):")
print(psi_0.flatten())

# Evolve and track probabilities
t_values = np.linspace(0, 4*np.pi/omega, 200)
prob_0 = []  # Probability of |0>
prob_1 = []  # Probability of |1>
norms = []

for t in t_values:
    U = time_evolution_operator(H, t, hbar)
    psi_t = U @ psi_0

    prob_0.append(np.abs(psi_t[0, 0])**2)
    prob_1.append(np.abs(psi_t[1, 0])**2)
    norms.append(np.abs(np.vdot(psi_t, psi_t)))

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Probabilities over time
ax1 = axes[0, 0]
ax1.plot(t_values * omega / (2*np.pi), prob_0, 'b-', label='P(|0>)', linewidth=2)
ax1.plot(t_values * omega / (2*np.pi), prob_1, 'r-', label='P(|1>)', linewidth=2)
ax1.set_xlabel('Time (periods: t*omega/2pi)')
ax1.set_ylabel('Probability')
ax1.set_title('Measurement Probabilities vs Time')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 2)

# Plot 2: Norm conservation
ax2 = axes[0, 1]
ax2.plot(t_values * omega / (2*np.pi), norms, 'g-', linewidth=2)
ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (periods)')
ax2.set_ylabel('Norm |<psi|psi>|')
ax2.set_title('Norm Conservation (Unitarity Check)')
ax2.set_ylim(0.99, 1.01)
ax2.grid(True, alpha=0.3)

# =============================================================================
# Part 6: Bloch Sphere Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Bloch Sphere Trajectory")
print("=" * 60)

def bloch_coords(psi):
    """Extract Bloch sphere coordinates from state vector."""
    # psi = cos(theta/2)|0> + e^(i*phi)*sin(theta/2)|1>
    # Bloch vector: (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    alpha, beta = psi.flatten()
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2
    return x, y, z

# Track Bloch vector
bloch_x, bloch_y, bloch_z = [], [], []
for t in t_values:
    U = time_evolution_operator(H, t, hbar)
    psi_t = U @ psi_0
    bx, by, bz = bloch_coords(psi_t)
    bloch_x.append(bx)
    bloch_y.append(by)
    bloch_z.append(bz)

# Plot Bloch trajectory (top-down view)
ax3 = axes[1, 0]
ax3.plot(bloch_x, bloch_y, 'b-', linewidth=1.5, alpha=0.7)
ax3.plot(bloch_x[0], bloch_y[0], 'go', markersize=10, label='Start')
ax3.plot(bloch_x[-1], bloch_y[-1], 'ro', markersize=10, label='End')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
ax3.add_patch(circle)
ax3.set_xlabel('Bloch x')
ax3.set_ylabel('Bloch y')
ax3.set_title('Bloch Sphere Trajectory (Top View: xy-plane)')
ax3.set_xlim(-1.2, 1.2)
ax3.set_ylim(-1.2, 1.2)
ax3.set_aspect('equal')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot z-component (which should stay constant for sigma_z Hamiltonian)
ax4 = axes[1, 1]
ax4.plot(t_values * omega / (2*np.pi), bloch_z, 'm-', linewidth=2)
ax4.set_xlabel('Time (periods)')
ax4.set_ylabel('Bloch z')
ax4.set_title('Bloch z-component vs Time')
ax4.set_ylim(-1.1, 1.1)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_359_time_evolution_operator.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_359_time_evolution_operator.png'")

# =============================================================================
# Part 7: Different Hamiltonians
# =============================================================================

print("\n" + "=" * 60)
print("Part 7: Evolution Under Different Hamiltonians")
print("=" * 60)

# Compare evolution under sigma_x, sigma_y, sigma_z
hamiltonians = {
    'sigma_x': (hbar * omega / 2) * sigma_x,
    'sigma_y': (hbar * omega / 2) * sigma_y,
    'sigma_z': (hbar * omega / 2) * sigma_z
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, H_test) in enumerate(hamiltonians.items()):
    ax = axes[idx]

    bloch_x, bloch_y, bloch_z = [], [], []
    for t in t_values[:100]:  # Shorter time range
        U = time_evolution_operator(H_test, t, hbar)
        psi_t = U @ psi_0
        bx, by, bz = bloch_coords(psi_t)
        bloch_x.append(bx)
        bloch_y.append(by)
        bloch_z.append(bz)

    # 2D projection (x-z plane)
    ax.plot(bloch_x, bloch_z, 'b-', linewidth=1.5)
    ax.plot(bloch_x[0], bloch_z[0], 'go', markersize=10)
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    ax.set_xlabel('Bloch x')
    ax.set_ylabel('Bloch z')
    ax.set_title(f'H ~ {name}')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_359_different_hamiltonians.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_359_different_hamiltonians.png'")

# =============================================================================
# Part 8: Quantum Gates from Evolution
# =============================================================================

print("\n" + "=" * 60)
print("Part 8: Quantum Gates as Time Evolution")
print("=" * 60)

# X gate = exp(-i * pi * sigma_x / 2)
t_x = np.pi / omega  # Time for pi rotation
H_x = (hbar * omega / 2) * sigma_x
X_from_evolution = time_evolution_operator(H_x, t_x, hbar)

print("X gate from evolution exp(-i*pi*sigma_x/2):")
print(np.round(X_from_evolution, 6))
print("\nActual X gate (sigma_x):")
print(sigma_x)
print("\nEqual up to global phase:", np.allclose(np.abs(X_from_evolution), np.abs(sigma_x)))

# Z gate = exp(-i * pi * sigma_z / 2)
H_z = (hbar * omega / 2) * sigma_z
Z_from_evolution = time_evolution_operator(H_z, t_x, hbar)

print("\nZ gate from evolution exp(-i*pi*sigma_z/2):")
print(np.round(Z_from_evolution, 6))
print("\nActual Z gate (sigma_z):")
print(sigma_z)

# Hadamard (more complex - requires combined rotation)
print("\nNote: Hadamard requires rotation around a different axis")
print("H ~ exp(-i*pi*(sigma_x + sigma_z)/(2*sqrt(2)))")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Time evolution operator | $\hat{U}(t) = e^{-i\hat{H}t/\hbar}$ |
| State evolution | $\|\psi(t)\rangle = \hat{U}(t)\|\psi(0)\rangle$ |
| Unitarity | $\hat{U}^\dagger(t)\hat{U}(t) = \hat{I}$ |
| Composition law | $\hat{U}(t_1)\hat{U}(t_2) = \hat{U}(t_1 + t_2)$ |
| Inverse | $\hat{U}^{-1}(t) = \hat{U}(-t) = \hat{U}^\dagger(t)$ |
| Spectral form | $\hat{U}(t) = \sum_n e^{-iE_n t/\hbar}\|E_n\rangle\langle E_n\|$ |
| Infinitesimal | $\hat{U}(\epsilon) \approx \hat{I} - \frac{i\epsilon}{\hbar}\hat{H}$ |

### Main Takeaways

1. **The time evolution operator** $\hat{U}(t) = e^{-i\hat{H}t/\hbar}$ provides the formal solution to the Schrodinger equation
2. **Unitarity** ensures probability conservation and reversibility
3. **Composition law** allows us to build up evolution from small steps
4. **The Hamiltonian generates** time translations — it's the quantum analog of the classical Hamiltonian
5. **Energy eigenstates** simplify the evolution operator via spectral decomposition
6. **Quantum gates** are time evolution operators for specific Hamiltonians and durations

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.3 and Sakurai Chapter 2.1
- [ ] Prove that $\hat{U}(t)$ satisfies the Schrodinger equation
- [ ] Verify unitarity: $\hat{U}^\dagger\hat{U} = \hat{I}$
- [ ] Derive the spectral decomposition of $\hat{U}(t)$
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and understand the computational lab
- [ ] Verify composition law numerically

---

## Preview: Day 360

Tomorrow we focus on **stationary states** — energy eigenstates that have especially simple time evolution. We'll see that their probability distributions are time-independent, even though the state itself acquires a time-dependent phase.

---

*"The notion that all these fragments is separately existent is evidently an illusion, and this illusion cannot do other than lead to endless conflict and confusion."*
— David Bohm

---

**Next:** [Day_360_Wednesday.md](Day_360_Wednesday.md) — Stationary States
