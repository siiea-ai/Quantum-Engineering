# Day 381: Number States |n⟩ — The Fock Space

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Constructing Number States |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 381, you will be able to:

1. Prove the existence of a ground state |0⟩ from the ladder operator algebra
2. Construct all number states |n⟩ using the creation operator
3. Derive the energy spectrum $E_n = \hbar\omega(n + \frac{1}{2})$
4. Prove the ladder operator actions: $\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$
5. Understand zero-point energy as a quantum mechanical necessity
6. Build Fock states computationally and verify orthonormality

---

## Core Content

### 1. Finding the Ground State

From yesterday, we have:
- $[\hat{a}, \hat{a}^\dagger] = 1$
- $\hat{H} = \hbar\omega(\hat{N} + \frac{1}{2})$ where $\hat{N} = \hat{a}^\dagger\hat{a}$

**Key Question:** What are the eigenstates of $\hat{N}$?

#### The Ground State Condition

Since $\hat{N} = \hat{a}^\dagger\hat{a}$, we have:
$$\langle\psi|\hat{N}|\psi\rangle = \langle\psi|\hat{a}^\dagger\hat{a}|\psi\rangle = ||\hat{a}|\psi\rangle||^2 \geq 0$$

**Theorem:** All eigenvalues of $\hat{N}$ are non-negative.

Now, if $|n\rangle$ is an eigenstate with $\hat{N}|n\rangle = n|n\rangle$, then from $[\hat{N}, \hat{a}] = -\hat{a}$:
$$\hat{N}(\hat{a}|n\rangle) = (n-1)(\hat{a}|n\rangle)$$

So $\hat{a}|n\rangle$ is an eigenstate with eigenvalue $n-1$.

**Problem:** If we keep applying $\hat{a}$, we get states with eigenvalues $n, n-1, n-2, \ldots$

Eventually, this would produce negative eigenvalues unless the sequence **terminates**.

#### The Termination Condition

There must exist a state $|0\rangle$ such that:
$$\boxed{\hat{a}|0\rangle = 0}$$

This is the **ground state** — the annihilation operator cannot lower it further.

**Verification:** Check that $|0\rangle$ has eigenvalue 0:
$$\hat{N}|0\rangle = \hat{a}^\dagger\hat{a}|0\rangle = \hat{a}^\dagger \cdot 0 = 0 = 0|0\rangle$$ ✓

---

### 2. Building the Excited States

Starting from $|0\rangle$, we can build all states by repeated application of $\hat{a}^\dagger$:

$$|1\rangle \propto \hat{a}^\dagger|0\rangle$$
$$|2\rangle \propto \hat{a}^\dagger|1\rangle \propto (\hat{a}^\dagger)^2|0\rangle$$
$$|n\rangle \propto (\hat{a}^\dagger)^n|0\rangle$$

#### Finding the Normalization

**Claim:** $\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$

**Proof:** Let $\hat{a}^\dagger|n\rangle = c_{n+1}|n+1\rangle$ for some constant $c_{n+1}$.

The norm squared is:
$$||\ \hat{a}^\dagger|n\rangle||^2 = \langle n|\hat{a}\hat{a}^\dagger|n\rangle$$

Using $\hat{a}\hat{a}^\dagger = \hat{a}^\dagger\hat{a} + 1 = \hat{N} + 1$:
$$= \langle n|(\hat{N} + 1)|n\rangle = n + 1$$

Therefore $|c_{n+1}|^2 = n+1$, so we choose:
$$\boxed{\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle}$$

**Similarly:**
$$\boxed{\hat{a}|n\rangle = \sqrt{n}|n-1\rangle}$$

(Proof: $||\hat{a}|n\rangle||^2 = \langle n|\hat{a}^\dagger\hat{a}|n\rangle = n$)

---

### 3. The Complete Spectrum

#### Energy Eigenvalues

Since $\hat{N}|n\rangle = n|n\rangle$ and $\hat{H} = \hbar\omega(\hat{N} + \frac{1}{2})$:

$$\hat{H}|n\rangle = \hbar\omega\left(n + \frac{1}{2}\right)|n\rangle$$

$$\boxed{E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad n = 0, 1, 2, 3, \ldots}$$

| Quantum Number | Energy | In terms of ground state |
|----------------|--------|--------------------------|
| $n = 0$ | $\frac{1}{2}\hbar\omega$ | $E_0$ |
| $n = 1$ | $\frac{3}{2}\hbar\omega$ | $E_0 + \hbar\omega$ |
| $n = 2$ | $\frac{5}{2}\hbar\omega$ | $E_0 + 2\hbar\omega$ |
| $n = 3$ | $\frac{7}{2}\hbar\omega$ | $E_0 + 3\hbar\omega$ |
| $\vdots$ | $\vdots$ | $\vdots$ |

**Key Features:**
1. **Equally spaced:** $\Delta E = \hbar\omega$ (constant)
2. **Zero-point energy:** $E_0 = \frac{1}{2}\hbar\omega \neq 0$
3. **Discrete:** Only integer multiples of $\hbar\omega$ above ground state

---

### 4. Constructing States Explicitly

The normalized $n$-th excited state is:

$$\boxed{|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle}$$

**Proof by induction:**

Base case: $|0\rangle = \frac{(\hat{a}^\dagger)^0}{\sqrt{0!}}|0\rangle = |0\rangle$ ✓

Inductive step: Assume $|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle$. Then:
$$|n+1\rangle = \frac{\hat{a}^\dagger|n\rangle}{\sqrt{n+1}} = \frac{\hat{a}^\dagger}{\sqrt{n+1}} \cdot \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle = \frac{(\hat{a}^\dagger)^{n+1}}{\sqrt{(n+1)!}}|0\rangle$$ ✓

---

### 5. Orthonormality and Completeness

#### Orthonormality

**Theorem:** $\langle m|n\rangle = \delta_{mn}$

**Proof:** For $m \neq n$, say $m < n$:
$$\langle m|n\rangle = \frac{1}{\sqrt{m!n!}}\langle 0|(\hat{a})^m(\hat{a}^\dagger)^n|0\rangle$$

Using $\hat{a}|0\rangle = 0$, any $\hat{a}$ acting to the right on $|0\rangle$ gives zero after enough applications. Careful counting shows this is zero for $m \neq n$.

For $m = n$: $\langle n|n\rangle = 1$ by construction.

#### Completeness (Resolution of Identity)

The number states form a complete basis:

$$\boxed{\hat{I} = \sum_{n=0}^{\infty}|n\rangle\langle n|}$$

Any state can be expanded:
$$|\psi\rangle = \sum_{n=0}^{\infty}c_n|n\rangle, \quad c_n = \langle n|\psi\rangle$$

---

### 6. Zero-Point Energy: Deep Implications

The ground state energy $E_0 = \frac{1}{2}\hbar\omega$ is **non-zero**. This is not arbitrary — it's required by the uncertainty principle.

#### Heisenberg Uncertainty Argument

For the ground state:
$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

The energy can be written:
$$E = \frac{(\Delta p)^2}{2m} + \frac{1}{2}m\omega^2(\Delta x)^2$$

Using $\Delta p \geq \frac{\hbar}{2\Delta x}$:
$$E \geq \frac{\hbar^2}{8m(\Delta x)^2} + \frac{1}{2}m\omega^2(\Delta x)^2$$

Minimizing over $\Delta x$:
$$\frac{dE}{d(\Delta x)} = 0 \implies \Delta x = \sqrt{\frac{\hbar}{2m\omega}}$$

$$E_{min} = \frac{\hbar\omega}{4} + \frac{\hbar\omega}{4} = \frac{\hbar\omega}{2}$$ ✓

#### Physical Consequences

| Phenomenon | Role of Zero-Point Energy |
|------------|---------------------------|
| **Casimir effect** | Vacuum energy between conductors |
| **Lamb shift** | Atomic energy level corrections |
| **van der Waals forces** | Molecular attraction |
| **Helium superfluidity** | Large zero-point motion prevents solidification |
| **Quantum fluctuations** | Cannot "turn off" the vacuum |

---

### 7. Expectation Values in Number States

Using ladder operators, we can compute expectation values without wave functions.

#### Position and Momentum

$$\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$$

$$\langle n|\hat{x}|n\rangle = \sqrt{\frac{\hbar}{2m\omega}}(\langle n|\hat{a}|n\rangle + \langle n|\hat{a}^\dagger|n\rangle)$$
$$= \sqrt{\frac{\hbar}{2m\omega}}(\sqrt{n}\langle n|n-1\rangle + \sqrt{n+1}\langle n|n+1\rangle)$$
$$= \sqrt{\frac{\hbar}{2m\omega}}(\sqrt{n} \cdot 0 + \sqrt{n+1} \cdot 0) = 0$$

$$\boxed{\langle n|\hat{x}|n\rangle = 0, \quad \langle n|\hat{p}|n\rangle = 0}$$

**Physical interpretation:** The particle oscillates symmetrically — average position and momentum are zero.

#### Squared Quantities

$$\hat{x}^2 = \frac{\hbar}{2m\omega}(\hat{a} + \hat{a}^\dagger)^2 = \frac{\hbar}{2m\omega}(\hat{a}^2 + 2\hat{N} + 1 + (\hat{a}^\dagger)^2)$$

$$\langle n|\hat{x}^2|n\rangle = \frac{\hbar}{2m\omega}(0 + 2n + 1 + 0) = \frac{\hbar}{2m\omega}(2n + 1)$$

$$\boxed{\langle n|\hat{x}^2|n\rangle = \frac{\hbar}{m\omega}\left(n + \frac{1}{2}\right)}$$

Similarly:
$$\boxed{\langle n|\hat{p}^2|n\rangle = m\omega\hbar\left(n + \frac{1}{2}\right)}$$

#### Uncertainty Product

$$\Delta x = \sqrt{\langle\hat{x}^2\rangle - \langle\hat{x}\rangle^2} = \sqrt{\frac{\hbar}{m\omega}\left(n + \frac{1}{2}\right)}$$

$$\Delta p = \sqrt{m\omega\hbar\left(n + \frac{1}{2}\right)}$$

$$\boxed{\Delta x \cdot \Delta p = \hbar\left(n + \frac{1}{2}\right)}$$

For $n = 0$: $\Delta x \cdot \Delta p = \frac{\hbar}{2}$ — the **minimum uncertainty state**!

---

### 8. Quantum Computing Connection: Fock States in Photonics

#### Photon Number States

In quantum optics, $|n\rangle$ represents a state with exactly $n$ photons:

| State | Physical Meaning |
|-------|------------------|
| $|0\rangle$ | Vacuum (no photons) |
| $|1\rangle$ | Single photon (non-classical light) |
| $|2\rangle$ | Two-photon state |
| $|n\rangle$ | n-photon Fock state |

#### Single-Photon Sources

Fock states with $n = 1$ are crucial for:
- **Quantum key distribution:** BB84 protocol
- **Linear optical quantum computing:** KLM scheme
- **Quantum metrology:** Heisenberg-limited sensing

#### Bosonic Quantum Error Correction

Fock states encode quantum information:
- **Binomial codes:** $|0_L\rangle = |0\rangle + |4\rangle$, $|1_L\rangle = |2\rangle$
- **Cat codes:** Superpositions of coherent states
- **GKP codes:** Grid states in phase space

The ladder operators $\hat{a}$, $\hat{a}^\dagger$ are the building blocks for manipulating these encodings!

---

## Worked Examples

### Example 1: Constructing |2⟩ Explicitly

**Problem:** Start from $|0\rangle$ and construct $|2\rangle$ step by step.

**Solution:**

Step 1: Apply $\hat{a}^\dagger$ to $|0\rangle$:
$$\hat{a}^\dagger|0\rangle = \sqrt{1}|1\rangle = |1\rangle$$

Step 2: Apply $\hat{a}^\dagger$ to $|1\rangle$:
$$\hat{a}^\dagger|1\rangle = \sqrt{2}|2\rangle$$

Therefore:
$$|2\rangle = \frac{1}{\sqrt{2}}\hat{a}^\dagger|1\rangle = \frac{1}{\sqrt{2}}\hat{a}^\dagger\hat{a}^\dagger|0\rangle = \frac{(\hat{a}^\dagger)^2}{\sqrt{2!}}|0\rangle$$ ✓

**Energy:** $E_2 = \hbar\omega(2 + \frac{1}{2}) = \frac{5}{2}\hbar\omega$ $\blacksquare$

---

### Example 2: Computing $\langle 1|\hat{x}|2\rangle$

**Problem:** Find the matrix element $\langle 1|\hat{x}|2\rangle$.

**Solution:**

Using $\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$:

$$\langle 1|\hat{x}|2\rangle = \sqrt{\frac{\hbar}{2m\omega}}\left(\langle 1|\hat{a}|2\rangle + \langle 1|\hat{a}^\dagger|2\rangle\right)$$

Compute each term:
- $\hat{a}|2\rangle = \sqrt{2}|1\rangle$, so $\langle 1|\hat{a}|2\rangle = \sqrt{2}$
- $\hat{a}^\dagger|2\rangle = \sqrt{3}|3\rangle$, so $\langle 1|\hat{a}^\dagger|2\rangle = \sqrt{3}\langle 1|3\rangle = 0$

Therefore:
$$\boxed{\langle 1|\hat{x}|2\rangle = \sqrt{\frac{\hbar}{2m\omega}} \cdot \sqrt{2} = \sqrt{\frac{\hbar}{m\omega}}}$$ $\blacksquare$

---

### Example 3: Energy of a Superposition State

**Problem:** A QHO is in the state $|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |2\rangle)$. Find:
(a) $\langle\hat{H}\rangle$
(b) $\langle\hat{H}^2\rangle$
(c) $\Delta E$

**Solution:**

(a) Energy expectation:
$$\langle\hat{H}\rangle = \frac{1}{2}\left(\langle 0|\hat{H}|0\rangle + \langle 0|\hat{H}|2\rangle + \langle 2|\hat{H}|0\rangle + \langle 2|\hat{H}|2\rangle\right)$$

Since $\hat{H}|n\rangle = E_n|n\rangle$ and $\langle m|n\rangle = \delta_{mn}$:
$$= \frac{1}{2}(E_0 + 0 + 0 + E_2) = \frac{1}{2}\left(\frac{\hbar\omega}{2} + \frac{5\hbar\omega}{2}\right) = \frac{3\hbar\omega}{2}$$

$$\boxed{\langle\hat{H}\rangle = \frac{3}{2}\hbar\omega}$$

(b) Energy squared:
$$\langle\hat{H}^2\rangle = \frac{1}{2}(E_0^2 + E_2^2) = \frac{1}{2}\left(\frac{\hbar^2\omega^2}{4} + \frac{25\hbar^2\omega^2}{4}\right) = \frac{26\hbar^2\omega^2}{8} = \frac{13\hbar^2\omega^2}{4}$$

(c) Uncertainty:
$$(\Delta E)^2 = \langle\hat{H}^2\rangle - \langle\hat{H}\rangle^2 = \frac{13\hbar^2\omega^2}{4} - \frac{9\hbar^2\omega^2}{4} = \hbar^2\omega^2$$

$$\boxed{\Delta E = \hbar\omega}$$ $\blacksquare$

---

## Practice Problems

### Level 1: Direct Application

1. **State Construction:** Write $|3\rangle$ and $|4\rangle$ in terms of $\hat{a}^\dagger$ acting on $|0\rangle$.

2. **Ladder Actions:** Compute:
   (a) $\hat{a}|5\rangle$
   (b) $\hat{a}^\dagger|5\rangle$
   (c) $\hat{a}^2|5\rangle$

3. **Matrix Elements:** Find $\langle 2|\hat{a}|3\rangle$ and $\langle 2|\hat{a}^\dagger|3\rangle$.

### Level 2: Intermediate

4. **Number Operator:** Verify that $\langle n|\hat{N}|n\rangle = n$ and $\langle n|\hat{N}^2|n\rangle = n^2$ for a general number state.

5. **Position Matrix:** Compute the matrix elements $\langle m|\hat{x}|n\rangle$ for $m, n = 0, 1, 2, 3$. What pattern do you observe?

6. **Superposition:** For the state $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle + |1\rangle + |2\rangle)$:
   (a) Verify normalization
   (b) Calculate $\langle\hat{N}\rangle$
   (c) Calculate $\Delta N$

### Level 3: Challenging

7. **Baker-Campbell-Hausdorff for QHO:** Prove that:
   $$e^{\alpha\hat{a}^\dagger}|0\rangle = \sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle$$
   This is related to coherent states (Day 383 preview).

8. **Virial Theorem:** For any stationary state of the QHO, prove:
   $$\langle\hat{T}\rangle = \langle\hat{V}\rangle = \frac{E_n}{2}$$
   where $\hat{T} = \frac{\hat{p}^2}{2m}$ and $\hat{V} = \frac{1}{2}m\omega^2\hat{x}^2$.

9. **Thermal State:** At temperature $T$, the oscillator has a thermal density matrix:
   $$\hat{\rho} = \sum_{n=0}^{\infty}P_n|n\rangle\langle n|, \quad P_n = \frac{e^{-E_n/k_BT}}{Z}$$
   (a) Find the partition function $Z$.
   (b) Compute $\langle\hat{N}\rangle$ at temperature $T$.
   (c) Show that $\langle\hat{N}\rangle \to 0$ as $T \to 0$ and $\langle\hat{N}\rangle \to k_BT/\hbar\omega$ as $T \to \infty$.

---

## Computational Lab

### Objective
Build Fock states, verify orthonormality, and compute expectation values using ladder operators.

```python
"""
Day 381 Computational Lab: Number States (Fock States)
Quantum Harmonic Oscillator - Week 55
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# =============================================================================
# Part 1: Building Fock States from Ladder Operators
# =============================================================================

print("=" * 70)
print("Part 1: Constructing Fock States")
print("=" * 70)

def create_ladder_operators(N_dim):
    """Create annihilation and creation operator matrices"""
    a = np.zeros((N_dim, N_dim), dtype=complex)
    a_dag = np.zeros((N_dim, N_dim), dtype=complex)

    for n in range(1, N_dim):
        a[n-1, n] = np.sqrt(n)
    for n in range(N_dim - 1):
        a_dag[n+1, n] = np.sqrt(n + 1)

    return a, a_dag

def fock_state(n, N_dim):
    """Create |n⟩ as a column vector"""
    state = np.zeros(N_dim, dtype=complex)
    state[n] = 1.0
    return state

def construct_fock_from_vacuum(n, a_dag, N_dim):
    """
    Build |n⟩ = (a†)^n / √(n!) |0⟩
    """
    vacuum = fock_state(0, N_dim)

    # Apply (a†)^n
    state = vacuum.copy()
    for _ in range(n):
        state = a_dag @ state

    # Normalize by 1/√(n!)
    state = state / np.sqrt(factorial(n))

    return state

N_dim = 10
a, a_dag = create_ladder_operators(N_dim)

print(f"\nHilbert space dimension: {N_dim}")

# Construct states both ways and compare
print("\nComparing Fock states: direct vs. ladder construction")
for n in range(5):
    direct = fock_state(n, N_dim)
    ladder = construct_fock_from_vacuum(n, a_dag, N_dim)
    match = np.allclose(direct, ladder)
    print(f"|{n}⟩: Direct matches ladder construction: {match}")

# =============================================================================
# Part 2: Verify Ladder Operator Actions
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Verifying â|n⟩ = √n|n-1⟩ and â†|n⟩ = √(n+1)|n+1⟩")
print("=" * 70)

for n in range(1, 5):
    ket_n = fock_state(n, N_dim)

    # Test annihilation
    a_ket_n = a @ ket_n
    expected_a = np.sqrt(n) * fock_state(n-1, N_dim)
    print(f"â|{n}⟩ = √{n}|{n-1}⟩: {np.allclose(a_ket_n, expected_a)}")

    # Test creation
    a_dag_ket_n = a_dag @ ket_n
    expected_a_dag = np.sqrt(n+1) * fock_state(n+1, N_dim)
    print(f"â†|{n}⟩ = √{n+1}|{n+1}⟩: {np.allclose(a_dag_ket_n, expected_a_dag)}")

# Special case: ground state annihilation
ket_0 = fock_state(0, N_dim)
a_ket_0 = a @ ket_0
print(f"\nâ|0⟩ = 0: {np.allclose(a_ket_0, np.zeros(N_dim))}")

# =============================================================================
# Part 3: Orthonormality of Fock States
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Orthonormality ⟨m|n⟩ = δₘₙ")
print("=" * 70)

# Build inner product matrix
inner_product_matrix = np.zeros((N_dim, N_dim), dtype=complex)
for m in range(N_dim):
    for n in range(N_dim):
        ket_m = fock_state(m, N_dim)
        ket_n = fock_state(n, N_dim)
        inner_product_matrix[m, n] = np.vdot(ket_m, ket_n)

print("\nInner product matrix ⟨m|n⟩:")
print(np.real(inner_product_matrix[:6, :6]))
print(f"\nIs identity matrix? {np.allclose(inner_product_matrix, np.eye(N_dim))}")

# =============================================================================
# Part 4: Energy Eigenvalues
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Energy Spectrum E_n = ℏω(n + 1/2)")
print("=" * 70)

N_op = a_dag @ a
hbar_omega = 1.0  # Set ℏω = 1
H = hbar_omega * (N_op + 0.5 * np.eye(N_dim))

# Check eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(H)

print("\nEnergy eigenvalues from diagonalization:")
for n in range(6):
    expected = hbar_omega * (n + 0.5)
    computed = eigenvalues[n]
    print(f"E_{n} = {computed:.4f} (expected: {expected:.4f})")

# =============================================================================
# Part 5: Expectation Values
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Expectation Values in Number States")
print("=" * 70)

# Position and momentum operators (natural units: ℏ = m = ω = 1)
x_op = (a + a_dag) / np.sqrt(2)
p_op = 1j * (a_dag - a) / np.sqrt(2)

def expectation(op, state):
    """Compute ⟨ψ|Ô|ψ⟩"""
    return np.real(np.vdot(state, op @ state))

print("\n<n|x|n> and <n|p|n>:")
for n in range(5):
    ket_n = fock_state(n, N_dim)
    x_exp = expectation(x_op, ket_n)
    p_exp = expectation(p_op, ket_n)
    print(f"n={n}: ⟨x⟩ = {x_exp:.6f}, ⟨p⟩ = {p_exp:.6f}")

print("\n<n|x²|n> and <n|p²|n>:")
x2_op = x_op @ x_op
p2_op = p_op @ p_op

for n in range(5):
    ket_n = fock_state(n, N_dim)
    x2_exp = expectation(x2_op, ket_n)
    p2_exp = expectation(p2_op, ket_n)
    expected = n + 0.5
    print(f"n={n}: ⟨x²⟩ = {x2_exp:.4f}, ⟨p²⟩ = {p2_exp:.4f} (expected: {expected:.4f})")

print("\nUncertainty Product Δx·Δp:")
for n in range(5):
    ket_n = fock_state(n, N_dim)
    delta_x = np.sqrt(expectation(x2_op, ket_n) - expectation(x_op, ket_n)**2)
    delta_p = np.sqrt(expectation(p2_op, ket_n) - expectation(p_op, ket_n)**2)
    product = delta_x * delta_p
    expected = n + 0.5  # In natural units where ℏ = 1
    print(f"n={n}: Δx·Δp = {product:.4f} (expected: {expected:.4f} ℏ)")

# =============================================================================
# Part 6: Visualize Number State Probabilities
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Visualizing Number States and Superpositions")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot individual Fock states
ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))
for n, color in zip(range(5), colors):
    ket_n = fock_state(n, N_dim)
    probs = np.abs(ket_n)**2
    ax.bar(np.arange(N_dim) + n*0.15, probs, width=0.15, color=color,
           label=f'|{n}⟩', alpha=0.8)

ax.set_xlabel('Fock state index m', fontsize=12)
ax.set_ylabel('Probability |⟨m|n⟩|²', fontsize=12)
ax.set_title('Number States |n⟩ in Fock Basis', fontsize=14)
ax.legend()
ax.set_xticks(range(N_dim))

# Superposition state
ax = axes[0, 1]
psi = (fock_state(0, N_dim) + fock_state(2, N_dim)) / np.sqrt(2)
probs = np.abs(psi)**2

ax.bar(range(N_dim), probs, color='purple', alpha=0.7)
ax.set_xlabel('Fock state index n', fontsize=12)
ax.set_ylabel('Probability |c_n|²', fontsize=12)
ax.set_title('Superposition |ψ⟩ = (|0⟩ + |2⟩)/√2', fontsize=14)
ax.set_xticks(range(N_dim))

# Energy levels with population
ax = axes[1, 0]
n_vals = np.arange(8)
E_vals = hbar_omega * (n_vals + 0.5)

ax.barh(E_vals, np.ones(8), height=0.3, color='blue', alpha=0.5)
ax.set_xlabel('Degeneracy', fontsize=12)
ax.set_ylabel('Energy / ℏω', fontsize=12)
ax.set_title('QHO Energy Level Diagram', fontsize=14)
ax.set_yticks(E_vals)
ax.set_yticklabels([f'E_{n} = {E:.1f}' for n, E in enumerate(E_vals)])

# Matrix elements ⟨m|x|n⟩
ax = axes[1, 1]
x_matrix = np.zeros((6, 6))
for m in range(6):
    for n in range(6):
        ket_m = fock_state(m, N_dim)
        ket_n = fock_state(n, N_dim)
        x_matrix[m, n] = np.real(np.vdot(ket_m, x_op @ ket_n))

im = ax.imshow(x_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
ax.set_xlabel('n', fontsize=12)
ax.set_ylabel('m', fontsize=12)
ax.set_title('Matrix Elements ⟨m|x̂|n⟩', fontsize=14)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_xticks(range(6))
ax.set_yticks(range(6))

plt.tight_layout()
plt.savefig('day_381_fock_states.png', dpi=150, bbox_inches='tight')
plt.show()

print("Fock state visualization saved.")

# =============================================================================
# Part 7: Thermal State Distribution
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Thermal State at Various Temperatures")
print("=" * 70)

def thermal_distribution(n_max, temperature, hbar_omega=1.0, k_B=1.0):
    """
    Compute Boltzmann distribution P_n = exp(-E_n/kT) / Z
    """
    if temperature == 0:
        P = np.zeros(n_max)
        P[0] = 1.0
        return P

    E_n = hbar_omega * (np.arange(n_max) + 0.5)
    P = np.exp(-E_n / (k_B * temperature))
    Z = np.sum(P)
    return P / Z

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution at different temperatures
ax = axes[0]
temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(temperatures)))

for T, color in zip(temperatures, colors):
    P = thermal_distribution(15, T)
    ax.plot(range(15), P, 'o-', color=color, label=f'kT/ℏω = {T}', linewidth=2)

ax.set_xlabel('Quantum number n', fontsize=12)
ax.set_ylabel('Probability P_n', fontsize=12)
ax.set_title('Thermal State Distribution', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Average photon number vs temperature
ax = axes[1]
T_range = np.linspace(0.01, 5, 100)
n_avg = []

for T in T_range:
    P = thermal_distribution(50, T)
    n_avg.append(np.sum(np.arange(50) * P))

ax.plot(T_range, n_avg, 'b-', linewidth=2, label='Exact')
ax.plot(T_range, 1/(np.exp(1/T_range) - 1), 'r--', linewidth=2,
        label='Bose-Einstein: 1/(e^{ℏω/kT} - 1)')
ax.set_xlabel('Temperature kT/ℏω', fontsize=12)
ax.set_ylabel('Average photon number ⟨N⟩', fontsize=12)
ax.set_title('Mean Occupation Number vs Temperature', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_381_thermal_state.png', dpi=150, bbox_inches='tight')
plt.show()

print("Thermal state analysis saved.")

# =============================================================================
# Part 8: Zero-Point Energy Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 8: Zero-Point Energy and Uncertainty")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

# Classical potential
x = np.linspace(-3, 3, 200)
V = 0.5 * x**2

ax.plot(x, V, 'k-', linewidth=2, label='V(x) = ½mω²x²')

# Ground state energy level
E_0 = 0.5
ax.axhline(E_0, color='blue', linestyle='-', linewidth=2, label=f'E_0 = ½ℏω')

# Classical turning points
x_turn = np.sqrt(2 * E_0)
ax.plot([-x_turn, x_turn], [E_0, E_0], 'bo', markersize=10)

# Shade classically forbidden region
x_forbidden_left = np.linspace(-3, -x_turn, 100)
x_forbidden_right = np.linspace(x_turn, 3, 100)
ax.fill_between(x_forbidden_left, V[x < -x_turn][:len(x_forbidden_left)], 3,
                alpha=0.3, color='red', label='Classically forbidden')
ax.fill_between(x_forbidden_right, V[x > x_turn][:len(x_forbidden_right)], 3,
                alpha=0.3, color='red')

# Ground state probability (Gaussian)
psi_0 = (1/np.pi)**0.25 * np.exp(-x**2/2)
prob_0 = np.abs(psi_0)**2
ax.fill_between(x, E_0, E_0 + prob_0*1.5, alpha=0.5, color='blue',
                label='|ψ_0(x)|² (scaled)')

# Annotations
ax.annotate('Zero-point energy\n(uncertainty principle)', xy=(0, E_0),
            xytext=(1.5, 1.5), fontsize=11,
            arrowprops=dict(arrowstyle='->', color='blue'))
ax.annotate('Classical turning point', xy=(x_turn, E_0),
            xytext=(2, 0.8), fontsize=11,
            arrowprops=dict(arrowstyle='->', color='black'))

ax.set_xlabel('Position x (natural units)', fontsize=12)
ax.set_ylabel('Energy (ℏω)', fontsize=12)
ax.set_title('QHO Ground State: Zero-Point Energy and Quantum Tunneling', fontsize=14)
ax.legend(loc='upper right')
ax.set_xlim(-3, 3)
ax.set_ylim(0, 3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_381_zero_point_energy.png', dpi=150, bbox_inches='tight')
plt.show()

print("Zero-point energy visualization saved.")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Ground state condition | $\hat{a}|0\rangle = 0$ |
| Excited states | $|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle$ |
| Creation action | $\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$ |
| Annihilation action | $\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$ |
| Energy spectrum | $E_n = \hbar\omega(n + \frac{1}{2})$ |
| Orthonormality | $\langle m|n\rangle = \delta_{mn}$ |
| Completeness | $\hat{I} = \sum_{n=0}^{\infty}|n\rangle\langle n|$ |
| Uncertainty | $\Delta x \cdot \Delta p = \hbar(n + \frac{1}{2})$ |

### Main Takeaways

1. **Ground state exists** because eigenvalues of $\hat{N}$ cannot be negative
2. **States are built algebraically:** $|n\rangle = (\hat{a}^\dagger)^n/\sqrt{n!}|0\rangle$
3. **Spectrum is discrete and equally spaced:** $E_n = \hbar\omega(n + \frac{1}{2})$
4. **Zero-point energy is real:** $E_0 = \frac{1}{2}\hbar\omega$ from uncertainty principle
5. **Ground state is minimum uncertainty:** $\Delta x \cdot \Delta p = \hbar/2$
6. **Fock states are orthonormal:** Form a complete basis for the Hilbert space

---

## Daily Checklist

- [ ] Read Shankar Section 7.3 on the algebraic solution
- [ ] Prove the ground state condition $\hat{a}|0\rangle = 0$ independently
- [ ] Derive the normalization factors $\sqrt{n}$ and $\sqrt{n+1}$
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt the thermal state problem (Level 3, #9)
- [ ] Run and understand the computational lab
- [ ] Contemplate: Why can't we have zero energy classically but not quantum mechanically?

---

## Preview: Day 382

Tomorrow we derive the **wave functions** $\psi_n(x)$ using the position representation. We'll see Hermite polynomials emerge and visualize the probability densities for excited states.

---

*"The algebraic approach does not require us to solve any differential equation, and this is its great charm."*
— R. Shankar

---

**Next:** [Day_382_Thursday.md](Day_382_Thursday.md) — QHO Wave Functions
