# Day 380: Ladder Operators — The Algebraic Method

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Ladder Operators |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 380, you will be able to:

1. Define the annihilation ($\hat{a}$) and creation ($\hat{a}^\dagger$) operators
2. Prove the fundamental commutation relation $[\hat{a}, \hat{a}^\dagger] = 1$
3. Express the Hamiltonian as $\hat{H} = \hbar\omega(\hat{a}^\dagger\hat{a} + \frac{1}{2})$
4. Derive $\hat{x}$ and $\hat{p}$ in terms of ladder operators
5. Understand why this method is called "algebraic" (no differential equations!)
6. Build matrix representations of ladder operators

---

## Core Content

### 1. Motivation: Why Ladder Operators?

Yesterday we set up the Schrodinger equation for the QHO. We could solve it by brute-force differential equation techniques (leading to Hermite polynomials). But there's a more elegant approach.

**The Algebraic Method:**
- Uses only commutator algebra (no calculus on wave functions)
- Reveals the structure of the Hilbert space directly
- Generalizes to angular momentum, spin, and quantum field theory
- Provides computational advantages

**Key insight:** The Hamiltonian $\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$ can be **factored** into simpler operators!

---

### 2. Defining the Ladder Operators

Consider the classical complex combination $a = x + ip$. Inspired by this, we define:

#### Annihilation (Lowering) Operator

$$\boxed{\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} + \frac{i\hat{p}}{m\omega}\right)}$$

#### Creation (Raising) Operator

$$\boxed{\hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} - \frac{i\hat{p}}{m\omega}\right)}$$

**Note:** $\hat{a}^\dagger$ is obtained from $\hat{a}$ by taking the Hermitian conjugate ($\hat{x}^\dagger = \hat{x}$, $\hat{p}^\dagger = \hat{p}$, $i^\dagger = -i$).

#### Why These Names?

- **Annihilation:** $\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$ — removes one quantum of energy
- **Creation:** $\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$ — adds one quantum of energy
- **Ladder:** They move up and down the energy ladder

---

### 3. The Fundamental Commutator

**Theorem:** $[\hat{a}, \hat{a}^\dagger] = 1$

**Proof:**

Start with the canonical commutation relation:
$$[\hat{x}, \hat{p}] = i\hbar$$

Compute:
$$\hat{a}\hat{a}^\dagger = \frac{m\omega}{2\hbar}\left(\hat{x} + \frac{i\hat{p}}{m\omega}\right)\left(\hat{x} - \frac{i\hat{p}}{m\omega}\right)$$

Expanding:
$$\hat{a}\hat{a}^\dagger = \frac{m\omega}{2\hbar}\left(\hat{x}^2 - \frac{i\hat{x}\hat{p}}{m\omega} + \frac{i\hat{p}\hat{x}}{m\omega} + \frac{\hat{p}^2}{m^2\omega^2}\right)$$

$$= \frac{m\omega}{2\hbar}\left(\hat{x}^2 + \frac{\hat{p}^2}{m^2\omega^2} + \frac{i}{m\omega}(\hat{p}\hat{x} - \hat{x}\hat{p})\right)$$

$$= \frac{m\omega}{2\hbar}\left(\hat{x}^2 + \frac{\hat{p}^2}{m^2\omega^2} - \frac{i}{m\omega}[\hat{x}, \hat{p}]\right)$$

Using $[\hat{x}, \hat{p}] = i\hbar$:
$$= \frac{m\omega}{2\hbar}\left(\hat{x}^2 + \frac{\hat{p}^2}{m^2\omega^2} - \frac{i \cdot i\hbar}{m\omega}\right)$$

$$= \frac{m\omega}{2\hbar}\left(\hat{x}^2 + \frac{\hat{p}^2}{m^2\omega^2} + \frac{\hbar}{m\omega}\right)$$

Similarly:
$$\hat{a}^\dagger\hat{a} = \frac{m\omega}{2\hbar}\left(\hat{x}^2 + \frac{\hat{p}^2}{m^2\omega^2} - \frac{\hbar}{m\omega}\right)$$

Therefore:
$$[\hat{a}, \hat{a}^\dagger] = \hat{a}\hat{a}^\dagger - \hat{a}^\dagger\hat{a} = \frac{m\omega}{2\hbar} \cdot 2 \cdot \frac{\hbar}{m\omega} = 1$$

$$\boxed{[\hat{a}, \hat{a}^\dagger] = 1}$$

This is the **bosonic commutation relation**, fundamental to quantum mechanics and quantum field theory. $\blacksquare$

---

### 4. Hamiltonian in Terms of Ladder Operators

From the calculation above:
$$\hat{a}^\dagger\hat{a} = \frac{m\omega}{2\hbar}\hat{x}^2 + \frac{\hat{p}^2}{2m\hbar\omega} - \frac{1}{2}$$

Multiply by $\hbar\omega$:
$$\hbar\omega\hat{a}^\dagger\hat{a} = \frac{1}{2}m\omega^2\hat{x}^2 + \frac{\hat{p}^2}{2m} - \frac{\hbar\omega}{2}$$

Recognizing the Hamiltonian:
$$\hbar\omega\hat{a}^\dagger\hat{a} = \hat{H} - \frac{\hbar\omega}{2}$$

$$\boxed{\hat{H} = \hbar\omega\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right)}$$

#### The Number Operator

Define:
$$\boxed{\hat{N} = \hat{a}^\dagger\hat{a}}$$

Then:
$$\boxed{\hat{H} = \hbar\omega\left(\hat{N} + \frac{1}{2}\right)}$$

The eigenstates of $\hat{N}$ are the energy eigenstates!

---

### 5. Position and Momentum from Ladder Operators

Inverting the definitions:

$$\boxed{\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)}$$

$$\boxed{\hat{p} = i\sqrt{\frac{m\omega\hbar}{2}}(\hat{a}^\dagger - \hat{a})}$$

**Verification:** Add and subtract the definitions of $\hat{a}$ and $\hat{a}^\dagger$:
$$\hat{a} + \hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}} \cdot 2\hat{x} = \sqrt{\frac{2m\omega}{\hbar}}\hat{x}$$

Therefore:
$$\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$$ ✓

---

### 6. Useful Commutator Identities

From $[\hat{a}, \hat{a}^\dagger] = 1$, we can derive:

$$[\hat{N}, \hat{a}] = [\hat{a}^\dagger\hat{a}, \hat{a}] = \hat{a}^\dagger[\hat{a}, \hat{a}] + [\hat{a}^\dagger, \hat{a}]\hat{a} = 0 - \hat{a} = -\hat{a}$$

$$[\hat{N}, \hat{a}^\dagger] = [\hat{a}^\dagger\hat{a}, \hat{a}^\dagger] = \hat{a}^\dagger[\hat{a}, \hat{a}^\dagger] + [\hat{a}^\dagger, \hat{a}^\dagger]\hat{a} = \hat{a}^\dagger + 0 = \hat{a}^\dagger$$

**Summary:**

| Commutator | Value |
|------------|-------|
| $[\hat{a}, \hat{a}^\dagger]$ | $1$ |
| $[\hat{N}, \hat{a}]$ | $-\hat{a}$ |
| $[\hat{N}, \hat{a}^\dagger]$ | $\hat{a}^\dagger$ |
| $[\hat{H}, \hat{a}]$ | $-\hbar\omega\hat{a}$ |
| $[\hat{H}, \hat{a}^\dagger]$ | $\hbar\omega\hat{a}^\dagger$ |

The last two follow from $\hat{H} = \hbar\omega(\hat{N} + \frac{1}{2})$.

---

### 7. Physical Interpretation

#### Why "Ladder"?

If $|E\rangle$ is an eigenstate with energy $E$:
$$\hat{H}|E\rangle = E|E\rangle$$

Then $\hat{a}^\dagger|E\rangle$ is an eigenstate with energy $E + \hbar\omega$:
$$\hat{H}(\hat{a}^\dagger|E\rangle) = \hat{a}^\dagger\hat{H}|E\rangle + [\hat{H}, \hat{a}^\dagger]|E\rangle = E\hat{a}^\dagger|E\rangle + \hbar\omega\hat{a}^\dagger|E\rangle$$
$$= (E + \hbar\omega)(\hat{a}^\dagger|E\rangle)$$

Similarly, $\hat{a}|E\rangle$ has energy $E - \hbar\omega$.

**The operators act as ladder rungs:**
- $\hat{a}^\dagger$ raises energy by $\hbar\omega$
- $\hat{a}$ lowers energy by $\hbar\omega$

#### Photon Interpretation

In quantum optics:
- $|n\rangle$ = state with $n$ photons
- $\hat{a}^\dagger$ creates one photon (adds energy $\hbar\omega$)
- $\hat{a}$ annihilates one photon (removes energy $\hbar\omega$)
- $\hat{N} = \hat{a}^\dagger\hat{a}$ counts photons

This connection makes the QHO central to quantum electrodynamics!

---

### 8. Quantum Computing Connection: Bosonic Qubits

#### Hardware Implementation

Superconducting circuits realize $\hat{a}$ and $\hat{a}^\dagger$ as microwave photon operators:

| Component | QHO Analog |
|-----------|------------|
| Superconducting resonator | Harmonic potential |
| Microwave photons | Excitation quanta |
| Transmon qubit | Nonlinear "anharmonic" oscillator |
| Cavity QED | Coupling oscillators |

#### Bosonic Codes

Information encoded in oscillator states:
- **Fock encoding:** Logical states = $|0\rangle_L = |0\rangle$, $|1\rangle_L = |1\rangle$
- **Cat codes:** Logical states = $|\pm\alpha\rangle$ (coherent states)
- **GKP codes:** Grid states in phase space (error-correctable)

The ladder operators are the fundamental building blocks for these encodings!

---

## Worked Examples

### Example 1: Proving $[\hat{N}, \hat{a}] = -\hat{a}$

**Problem:** Use $[\hat{a}, \hat{a}^\dagger] = 1$ to prove $[\hat{N}, \hat{a}] = -\hat{a}$.

**Solution:**

Using the identity $[AB, C] = A[B, C] + [A, C]B$:

$$[\hat{N}, \hat{a}] = [\hat{a}^\dagger\hat{a}, \hat{a}]$$
$$= \hat{a}^\dagger[\hat{a}, \hat{a}] + [\hat{a}^\dagger, \hat{a}]\hat{a}$$
$$= \hat{a}^\dagger \cdot 0 + (-1)\hat{a}$$
$$= -\hat{a}$$

$$\boxed{[\hat{N}, \hat{a}] = -\hat{a}}$$ $\blacksquare$

---

### Example 2: Computing $\hat{x}^2$ in Terms of Ladder Operators

**Problem:** Express $\hat{x}^2$ using $\hat{a}$ and $\hat{a}^\dagger$.

**Solution:**

From $\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$:

$$\hat{x}^2 = \frac{\hbar}{2m\omega}(\hat{a} + \hat{a}^\dagger)^2$$

Expanding:
$$(\hat{a} + \hat{a}^\dagger)^2 = \hat{a}^2 + \hat{a}\hat{a}^\dagger + \hat{a}^\dagger\hat{a} + (\hat{a}^\dagger)^2$$

Using $\hat{a}\hat{a}^\dagger = \hat{a}^\dagger\hat{a} + 1 = \hat{N} + 1$:
$$= \hat{a}^2 + (\hat{N} + 1) + \hat{N} + (\hat{a}^\dagger)^2$$
$$= \hat{a}^2 + 2\hat{N} + 1 + (\hat{a}^\dagger)^2$$

Therefore:
$$\boxed{\hat{x}^2 = \frac{\hbar}{2m\omega}\left(\hat{a}^2 + 2\hat{N} + 1 + (\hat{a}^\dagger)^2\right)}$$ $\blacksquare$

---

### Example 3: Raising the Energy

**Problem:** If $|n\rangle$ is a normalized eigenstate of $\hat{N}$ with eigenvalue $n$, show that $\hat{a}^\dagger|n\rangle$ is an eigenstate with eigenvalue $n+1$.

**Solution:**

We need to show $\hat{N}(\hat{a}^\dagger|n\rangle) = (n+1)(\hat{a}^\dagger|n\rangle)$.

Using $[\hat{N}, \hat{a}^\dagger] = \hat{a}^\dagger$:
$$\hat{N}\hat{a}^\dagger = \hat{a}^\dagger\hat{N} + \hat{a}^\dagger = \hat{a}^\dagger(\hat{N} + 1)$$

Therefore:
$$\hat{N}(\hat{a}^\dagger|n\rangle) = \hat{a}^\dagger(\hat{N} + 1)|n\rangle = \hat{a}^\dagger(n + 1)|n\rangle = (n+1)\hat{a}^\dagger|n\rangle$$

So $\hat{a}^\dagger|n\rangle$ is an eigenstate of $\hat{N}$ with eigenvalue $n+1$.

**Energy:** Since $\hat{H} = \hbar\omega(\hat{N} + \frac{1}{2})$:
$$E_{n+1} = \hbar\omega\left((n+1) + \frac{1}{2}\right) = \hbar\omega\left(n + \frac{3}{2}\right)$$

This is $\hbar\omega$ higher than $E_n = \hbar\omega(n + \frac{1}{2})$. $\blacksquare$

---

## Practice Problems

### Level 1: Direct Application

1. **Definition Check:** Starting from the definitions of $\hat{a}$ and $\hat{a}^\dagger$, verify that:
   $$\hat{a}^\dagger - \hat{a} = \sqrt{\frac{2m\omega}{\hbar}} \cdot \frac{i\hat{p}}{m\omega} = i\sqrt{\frac{2}{m\omega\hbar}}\hat{p}$$

2. **Commutator Practice:** Compute $[\hat{a}^2, \hat{a}^\dagger]$ using $[\hat{a}, \hat{a}^\dagger] = 1$.
   *Hint:* Use $[AB, C] = A[B, C] + [A, C]B$.

3. **Momentum Squared:** Express $\hat{p}^2$ in terms of ladder operators.

### Level 2: Intermediate

4. **Hamiltonian Verification:** Starting from:
   $$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$$
   and the expressions for $\hat{x}$ and $\hat{p}$ in terms of $\hat{a}$, $\hat{a}^\dagger$, verify that $\hat{H} = \hbar\omega(\hat{N} + \frac{1}{2})$.

5. **Product Rule:** Prove that $[\hat{a}, (\hat{a}^\dagger)^n] = n(\hat{a}^\dagger)^{n-1}$.
   *Hint:* Use induction, starting from $[\hat{a}, \hat{a}^\dagger] = 1$.

6. **Expectation Values:** For a state $|n\rangle$, use ladder operator methods to show:
   $$\langle n|\hat{x}|n\rangle = 0$$
   *Hint:* Express $\hat{x}$ in terms of $\hat{a}$ and $\hat{a}^\dagger$.

### Level 3: Challenging

7. **Baker-Campbell-Hausdorff:** If $[\hat{A}, [\hat{A}, \hat{B}]] = 0$, then:
   $$e^{\hat{A}}\hat{B}e^{-\hat{A}} = \hat{B} + [\hat{A}, \hat{B}]$$
   Use this to show that for $\alpha \in \mathbb{C}$:
   $$e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}\hat{a}e^{-(\alpha\hat{a}^\dagger - \alpha^*\hat{a})} = \hat{a} + \alpha$$

8. **Position-Momentum Uncertainty:** Using ladder operator methods, compute $\langle n|\hat{x}^2|n\rangle$ and $\langle n|\hat{p}^2|n\rangle$ for the state $|n\rangle$. Verify that:
   $$\Delta x \cdot \Delta p = \hbar\left(n + \frac{1}{2}\right)$$

9. **Coherent States Preview:** A coherent state $|\alpha\rangle$ satisfies $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$. Show that $|\alpha\rangle$ is NOT an eigenstate of $\hat{a}^\dagger$. What is $\hat{a}^\dagger|\alpha\rangle$?

---

## Computational Lab

### Objective
Build matrix representations of ladder operators and verify their properties numerically.

```python
"""
Day 380 Computational Lab: Ladder Operators
Quantum Harmonic Oscillator - Week 55
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# =============================================================================
# Part 1: Matrix Representations of Ladder Operators
# =============================================================================

print("=" * 70)
print("Part 1: Building Ladder Operator Matrices")
print("=" * 70)

def create_annihilation_operator(N_dim):
    """
    Create the annihilation operator matrix for a truncated Hilbert space.

    a|n⟩ = √n |n-1⟩

    Matrix elements: ⟨m|a|n⟩ = √n δ_{m,n-1}
    """
    a = np.zeros((N_dim, N_dim), dtype=complex)
    for n in range(1, N_dim):
        a[n-1, n] = np.sqrt(n)
    return a

def create_creation_operator(N_dim):
    """
    Create the creation operator matrix.

    a†|n⟩ = √(n+1) |n+1⟩

    Matrix elements: ⟨m|a†|n⟩ = √(n+1) δ_{m,n+1}
    """
    a_dag = np.zeros((N_dim, N_dim), dtype=complex)
    for n in range(N_dim - 1):
        a_dag[n+1, n] = np.sqrt(n + 1)
    return a_dag

# Create operators for N_dim = 8 (truncated Hilbert space)
N_dim = 8
a = create_annihilation_operator(N_dim)
a_dag = create_creation_operator(N_dim)

print(f"\nHilbert space dimension: {N_dim}")
print(f"\nAnnihilation operator a:")
print(np.real(a))

print(f"\nCreation operator a†:")
print(np.real(a_dag))

# Verify a† = (a)^†
print(f"\nVerify a† = (a)^†: {np.allclose(a_dag, a.conj().T)}")

# =============================================================================
# Part 2: Verify Commutation Relation [a, a†] = 1
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Verifying [a, a†] = 1")
print("=" * 70)

commutator = a @ a_dag - a_dag @ a
identity = np.eye(N_dim)

print(f"\n[a, a†] matrix:")
print(np.real(commutator))

print(f"\nIs [a, a†] = I? {np.allclose(commutator, identity)}")

# Note: Due to truncation, the last diagonal element is off
print(f"\nNote: The (N-1, N-1) element is {commutator[N_dim-1, N_dim-1]:.0f}")
print("This is a truncation artifact - we're missing |N⟩ in our basis.")

# =============================================================================
# Part 3: Number Operator and Hamiltonian
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Number Operator N = a†a and Hamiltonian")
print("=" * 70)

N_op = a_dag @ a

print(f"\nNumber operator N = a†a:")
print(np.real(N_op))

# The number operator should be diagonal with entries 0, 1, 2, ...
eigenvalues_N = np.diag(N_op)
print(f"\nDiagonal of N: {np.real(eigenvalues_N)}")
print(f"Expected: {np.arange(N_dim)}")

# Hamiltonian H = ℏω(N + 1/2) [we set ℏω = 1]
hbar_omega = 1.0
H = hbar_omega * (N_op + 0.5 * identity)

print(f"\nHamiltonian H = ℏω(N + 1/2) with ℏω = 1:")
print(f"Energy eigenvalues: {np.real(np.diag(H))}")
print(f"Expected E_n = n + 1/2: {np.arange(N_dim) + 0.5}")

# =============================================================================
# Part 4: Position and Momentum Operators
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Position and Momentum from Ladder Operators")
print("=" * 70)

# In natural units where ℏ = m = ω = 1:
# x = (a + a†)/√2
# p = i(a† - a)/√2

x_op = (a + a_dag) / np.sqrt(2)
p_op = 1j * (a_dag - a) / np.sqrt(2)

print(f"\nPosition operator x = (a + a†)/√2:")
print(np.real(x_op[:5, :5]))  # Show first 5x5 block

print(f"\nMomentum operator p = i(a† - a)/√2:")
print(np.real(p_op[:5, :5]))  # Show first 5x5 block

# Verify [x, p] = i (in natural units)
xp_comm = x_op @ p_op - p_op @ x_op
print(f"\n[x, p]:")
print(np.real(xp_comm[:5, :5]))

print(f"\nIs [x, p] = i·I? {np.allclose(xp_comm, 1j * identity)}")

# =============================================================================
# Part 5: Action of Ladder Operators on Number States
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Ladder Operators Acting on Number States")
print("=" * 70)

# Create Fock states as column vectors
def fock_state(n, N_dim):
    """Create |n⟩ as a column vector"""
    state = np.zeros(N_dim, dtype=complex)
    state[n] = 1.0
    return state

# Demonstrate a|n⟩ = √n |n-1⟩
n = 3
ket_n = fock_state(n, N_dim)
a_ket_n = a @ ket_n

print(f"\n|{n}⟩ = {ket_n}")
print(f"a|{n}⟩ = {a_ket_n}")
print(f"Expected: √{n} |{n-1}⟩ = {np.sqrt(n)} × {fock_state(n-1, N_dim)}")
print(f"Match: {np.allclose(a_ket_n, np.sqrt(n) * fock_state(n-1, N_dim))}")

# Demonstrate a†|n⟩ = √(n+1) |n+1⟩
a_dag_ket_n = a_dag @ ket_n

print(f"\na†|{n}⟩ = {a_dag_ket_n}")
print(f"Expected: √{n+1} |{n+1}⟩ = {np.sqrt(n+1):.4f} × {fock_state(n+1, N_dim)}")
print(f"Match: {np.allclose(a_dag_ket_n, np.sqrt(n+1) * fock_state(n+1, N_dim))}")

# What happens to ground state?
ket_0 = fock_state(0, N_dim)
a_ket_0 = a @ ket_0
print(f"\na|0⟩ = {a_ket_0}")
print("The ground state is annihilated by a!")

# =============================================================================
# Part 6: Visualization of Ladder Operator Structure
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Visualizing Operator Matrices")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot annihilation operator
ax = axes[0, 0]
im = ax.imshow(np.abs(a), cmap='Blues')
ax.set_title('|a| (Annihilation)', fontsize=12)
ax.set_xlabel('Column (input state)')
ax.set_ylabel('Row (output state)')
plt.colorbar(im, ax=ax, fraction=0.046)

# Plot creation operator
ax = axes[0, 1]
im = ax.imshow(np.abs(a_dag), cmap='Reds')
ax.set_title('|a†| (Creation)', fontsize=12)
ax.set_xlabel('Column (input state)')
ax.set_ylabel('Row (output state)')
plt.colorbar(im, ax=ax, fraction=0.046)

# Plot number operator
ax = axes[0, 2]
im = ax.imshow(np.real(N_op), cmap='Purples')
ax.set_title('N = a†a (Number)', fontsize=12)
ax.set_xlabel('Column (input state)')
ax.set_ylabel('Row (output state)')
plt.colorbar(im, ax=ax, fraction=0.046)

# Plot position operator
ax = axes[1, 0]
im = ax.imshow(np.real(x_op), cmap='RdBu_r', vmin=-2, vmax=2)
ax.set_title('x = (a + a†)/√2', fontsize=12)
ax.set_xlabel('Column (input state)')
ax.set_ylabel('Row (output state)')
plt.colorbar(im, ax=ax, fraction=0.046)

# Plot momentum operator
ax = axes[1, 1]
im = ax.imshow(np.imag(p_op), cmap='RdBu_r', vmin=-2, vmax=2)
ax.set_title('Im(p) = (a† - a)/√2', fontsize=12)
ax.set_xlabel('Column (input state)')
ax.set_ylabel('Row (output state)')
plt.colorbar(im, ax=ax, fraction=0.046)

# Plot Hamiltonian
ax = axes[1, 2]
im = ax.imshow(np.real(H), cmap='viridis')
ax.set_title('H = ℏω(N + 1/2)', fontsize=12)
ax.set_xlabel('Column (input state)')
ax.set_ylabel('Row (output state)')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('day_380_operator_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

print("Operator matrix visualization saved.")

# =============================================================================
# Part 7: Energy Level Diagram
# =============================================================================

print("\n" + "=" * 70)
print("Part 7: Energy Level Diagram with Ladder Transitions")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

# Draw energy levels
for n in range(6):
    E_n = n + 0.5
    ax.hlines(E_n, 0.2, 0.8, linewidth=3, color='blue')
    ax.text(0.85, E_n, f'|{n}⟩, $E_{n}$ = {E_n}ℏω', va='center', fontsize=11)

# Draw creation operator transitions (red arrows going up)
for n in range(5):
    ax.annotate('', xy=(0.35, n + 1.5), xytext=(0.35, n + 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.28, n + 1.0, f'$\\hat{{a}}^\\dagger$\n$\\sqrt{{{n+1}}}$',
            ha='center', va='center', fontsize=9, color='red')

# Draw annihilation operator transitions (green arrows going down)
for n in range(1, 6):
    ax.annotate('', xy=(0.65, n - 0.5), xytext=(0.65, n + 0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.72, n, f'$\\hat{{a}}$\n$\\sqrt{{{n}}}$',
            ha='center', va='center', fontsize=9, color='green')

ax.set_xlim(0, 1.2)
ax.set_ylim(-0.2, 6.5)
ax.set_ylabel('Energy / ℏω', fontsize=12)
ax.set_title('Quantum Harmonic Oscillator: Ladder Operator Transitions', fontsize=14)
ax.set_xticks([])
ax.grid(True, alpha=0.3, axis='y')

# Legend
ax.plot([], [], 'r-', linewidth=2, label='Creation $\\hat{a}^\\dagger$ (energy +ℏω)')
ax.plot([], [], 'g-', linewidth=2, label='Annihilation $\\hat{a}$ (energy -ℏω)')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('day_380_energy_ladder.png', dpi=150, bbox_inches='tight')
plt.show()

print("Energy ladder diagram saved.")

# =============================================================================
# Part 8: Verify Commutator Identities
# =============================================================================

print("\n" + "=" * 70)
print("Part 8: Numerical Verification of Commutator Identities")
print("=" * 70)

def commutator(A, B):
    """Compute [A, B] = AB - BA"""
    return A @ B - B @ A

# [N, a] = -a
comm_Na = commutator(N_op, a)
print(f"[N, a] = -a? {np.allclose(comm_Na, -a)}")

# [N, a†] = a†
comm_Na_dag = commutator(N_op, a_dag)
print(f"[N, a†] = a†? {np.allclose(comm_Na_dag, a_dag)}")

# [H, a] = -ℏω a
comm_Ha = commutator(H, a)
print(f"[H, a] = -ℏω·a? {np.allclose(comm_Ha, -hbar_omega * a)}")

# [H, a†] = ℏω a†
comm_Ha_dag = commutator(H, a_dag)
print(f"[H, a†] = ℏω·a†? {np.allclose(comm_Ha_dag, hbar_omega * a_dag)}")

# [a, (a†)^2] = 2a†
a_dag_sq = a_dag @ a_dag
comm_a_adag2 = commutator(a, a_dag_sq)
print(f"[a, (a†)²] = 2a†? {np.allclose(comm_a_adag2, 2 * a_dag)}")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Annihilation operator | $\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} + \frac{i\hat{p}}{m\omega}\right)$ |
| Creation operator | $\hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} - \frac{i\hat{p}}{m\omega}\right)$ |
| Fundamental commutator | $[\hat{a}, \hat{a}^\dagger] = 1$ |
| Number operator | $\hat{N} = \hat{a}^\dagger\hat{a}$ |
| Hamiltonian | $\hat{H} = \hbar\omega(\hat{N} + \frac{1}{2})$ |
| Position | $\hat{x} = \sqrt{\frac{\hbar}{2m\omega}}(\hat{a} + \hat{a}^\dagger)$ |
| Momentum | $\hat{p} = i\sqrt{\frac{m\omega\hbar}{2}}(\hat{a}^\dagger - \hat{a})$ |

### Main Takeaways

1. **Factorization:** The QHO Hamiltonian factors as $\hat{H} = \hbar\omega(\hat{a}^\dagger\hat{a} + \frac{1}{2})$
2. **Bosonic algebra:** $[\hat{a}, \hat{a}^\dagger] = 1$ is the fundamental commutation relation
3. **Ladder action:** $\hat{a}^\dagger$ raises energy by $\hbar\omega$, $\hat{a}$ lowers it by $\hbar\omega$
4. **No calculus needed:** The algebraic method avoids differential equations
5. **Universal structure:** This algebra appears in quantum field theory (photons, phonons, etc.)

---

## Daily Checklist

- [ ] Read Shankar Section 7.3 on ladder operators
- [ ] Prove $[\hat{a}, \hat{a}^\dagger] = 1$ independently
- [ ] Derive $\hat{H} = \hbar\omega(\hat{N} + \frac{1}{2})$ step by step
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt the BCH identity problem (Level 3, #7)
- [ ] Run and understand the computational lab
- [ ] Build the 8×8 ladder operator matrices by hand for small cases

---

## Preview: Day 381

Tomorrow we use the ladder operators to construct the complete set of **number states** |n⟩. We'll prove that the spectrum is $E_n = \hbar\omega(n + \frac{1}{2})$ for $n = 0, 1, 2, \ldots$ and understand the zero-point energy.

---

*"The operators $\hat{a}$ and $\hat{a}^\dagger$ are the key to the whole problem... They allow us to solve the harmonic oscillator without ever solving a differential equation."*
— R. Shankar

---

**Next:** [Day_381_Wednesday.md](Day_381_Wednesday.md) — Number States |n⟩
