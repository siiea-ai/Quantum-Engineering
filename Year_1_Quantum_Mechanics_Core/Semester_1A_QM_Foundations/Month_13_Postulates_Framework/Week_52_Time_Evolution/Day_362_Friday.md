# Day 362: The Heisenberg Picture — Operators Evolve, States Fixed

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Heisenberg Picture |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 362, you will be able to:

1. Define the Heisenberg picture and relate it to the Schrodinger picture
2. Derive the Heisenberg equation of motion for operators
3. Solve operator equations of motion for simple systems
4. Connect the Heisenberg equation to classical Hamilton equations
5. Prove equivalence of expectation values between pictures
6. Apply the Heisenberg picture to harmonic oscillator dynamics

---

## Core Content

### 1. The Heisenberg Picture: Definition

**In the Heisenberg picture:**

1. **State vectors** are time-independent:
$$|\psi_H\rangle = |\psi_S(0)\rangle = \text{constant}$$

2. **Operators** carry all the time dependence:
$$\hat{A}_H(t) = \hat{U}^\dagger(t)\hat{A}_S\hat{U}(t)$$

3. **Expectation values** are identical to the Schrodinger picture:
$$\langle\hat{A}\rangle = \langle\psi_H|\hat{A}_H(t)|\psi_H\rangle = \langle\psi_S(t)|\hat{A}_S|\psi_S(t)\rangle$$

The subscript "H" denotes Heisenberg picture quantities.

---

### 2. Transformation Between Pictures

The pictures are related by the time evolution operator $\hat{U}(t) = e^{-i\hat{H}t/\hbar}$:

**States:**
$$|\psi_H\rangle = \hat{U}^\dagger(t)|\psi_S(t)\rangle = |\psi_S(0)\rangle$$

**Operators:**
$$\boxed{\hat{A}_H(t) = \hat{U}^\dagger(t)\hat{A}_S\hat{U}(t) = e^{i\hat{H}t/\hbar}\hat{A}_S e^{-i\hat{H}t/\hbar}}$$

**Equivalence of Expectation Values:**
$$\langle\hat{A}\rangle = \langle\psi_S(t)|\hat{A}_S|\psi_S(t)\rangle = \langle\psi_S(0)|\hat{U}^\dagger\hat{A}_S\hat{U}|\psi_S(0)\rangle = \langle\psi_H|\hat{A}_H(t)|\psi_H\rangle$$

Physics is picture-independent!

---

### 3. The Heisenberg Equation of Motion

The fundamental equation of the Heisenberg picture gives the time evolution of operators:

**Derivation:**
$$\frac{d\hat{A}_H}{dt} = \frac{d}{dt}\left(\hat{U}^\dagger\hat{A}_S\hat{U}\right)$$

Using the product rule:
$$= \frac{d\hat{U}^\dagger}{dt}\hat{A}_S\hat{U} + \hat{U}^\dagger\hat{A}_S\frac{d\hat{U}}{dt} + \hat{U}^\dagger\frac{\partial\hat{A}_S}{\partial t}\hat{U}$$

From the Schrodinger equation: $\frac{d\hat{U}}{dt} = \frac{-i}{\hbar}\hat{H}\hat{U}$ and $\frac{d\hat{U}^\dagger}{dt} = \frac{i}{\hbar}\hat{U}^\dagger\hat{H}$

$$= \frac{i}{\hbar}\hat{U}^\dagger\hat{H}\hat{A}_S\hat{U} - \frac{i}{\hbar}\hat{U}^\dagger\hat{A}_S\hat{H}\hat{U} + \hat{U}^\dagger\frac{\partial\hat{A}_S}{\partial t}\hat{U}$$

$$= \frac{i}{\hbar}\hat{U}^\dagger[\hat{H}, \hat{A}_S]\hat{U} + \hat{U}^\dagger\frac{\partial\hat{A}_S}{\partial t}\hat{U}$$

Since $[\hat{H}, \hat{A}_S] = \hat{H}\hat{A}_S - \hat{A}_S\hat{H}$ and $\hat{U}^\dagger\hat{H}\hat{U} = \hat{H}$ (for time-independent $\hat{H}$):

$$\boxed{\frac{d\hat{A}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{A}_H] + \left(\frac{\partial\hat{A}_S}{\partial t}\right)_H}$$

This is the **Heisenberg equation of motion**.

---

### 4. Connection to Classical Mechanics

The Heisenberg equation is the quantum analog of Hamilton's equation with Poisson brackets:

| Classical | Quantum |
|-----------|---------|
| $\frac{dA}{dt} = \{A, H\} + \frac{\partial A}{\partial t}$ | $\frac{d\hat{A}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{A}_H] + \frac{\partial\hat{A}_H}{\partial t}$ |
| $\{q, p\} = 1$ | $[\hat{x}, \hat{p}] = i\hbar$ |

The correspondence is:
$$\boxed{\{A, B\} \longleftrightarrow \frac{1}{i\hbar}[\hat{A}, \hat{B}]}$$

This makes the Heisenberg picture the natural framework for understanding the classical limit.

---

### 5. Equations of Motion for Position and Momentum

**Position:**
$$\frac{d\hat{x}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{x}_H]$$

For $\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})$:
$$[\hat{H}, \hat{x}] = \frac{1}{2m}[\hat{p}^2, \hat{x}] = \frac{-i\hbar}{m}\hat{p}$$

$$\boxed{\frac{d\hat{x}_H}{dt} = \frac{\hat{p}_H}{m}}$$

**Momentum:**
$$\frac{d\hat{p}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{p}_H]$$

$$[\hat{H}, \hat{p}] = [V(\hat{x}), \hat{p}] = i\hbar\frac{\partial V}{\partial \hat{x}}$$

$$\boxed{\frac{d\hat{p}_H}{dt} = -\frac{\partial V}{\partial \hat{x}_H}}$$

These are **operator versions of Newton's equations**!

---

### 6. Conservation of Commutation Relations

A remarkable property: the canonical commutation relation is preserved under time evolution.

**Proof:**
$$[\hat{x}_H(t), \hat{p}_H(t)] = \hat{U}^\dagger[\hat{x}_S, \hat{p}_S]\hat{U} = \hat{U}^\dagger(i\hbar)\hat{U} = i\hbar$$

$$\boxed{[\hat{x}_H(t), \hat{p}_H(t)] = i\hbar \quad \text{for all } t}$$

This is essential for the consistency of quantum mechanics: the uncertainty principle holds at all times.

---

### 7. The Harmonic Oscillator in Heisenberg Picture

For $\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$:

**Equations of motion:**
$$\frac{d\hat{x}_H}{dt} = \frac{\hat{p}_H}{m}$$
$$\frac{d\hat{p}_H}{dt} = -m\omega^2\hat{x}_H$$

**Combine to get:**
$$\frac{d^2\hat{x}_H}{dt^2} = -\omega^2\hat{x}_H$$

**Solution:**
$$\boxed{\hat{x}_H(t) = \hat{x}_H(0)\cos(\omega t) + \frac{\hat{p}_H(0)}{m\omega}\sin(\omega t)}$$

$$\boxed{\hat{p}_H(t) = \hat{p}_H(0)\cos(\omega t) - m\omega\hat{x}_H(0)\sin(\omega t)}$$

The operators oscillate exactly like classical variables!

---

### 8. Creation and Annihilation Operators in Heisenberg Picture

The ladder operators:
$$\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} + \frac{i\hat{p}}{m\omega}\right)$$
$$\hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} - \frac{i\hat{p}}{m\omega}\right)$$

**Heisenberg equation:**
$$\frac{d\hat{a}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{a}_H]$$

Using $\hat{H} = \hbar\omega(\hat{a}^\dagger\hat{a} + \frac{1}{2})$ and $[\hat{a}, \hat{a}^\dagger] = 1$:
$$[\hat{H}, \hat{a}] = \hbar\omega[\hat{a}^\dagger\hat{a}, \hat{a}] = \hbar\omega(\hat{a}^\dagger[\hat{a}, \hat{a}] + [\hat{a}^\dagger, \hat{a}]\hat{a}) = -\hbar\omega\hat{a}$$

$$\frac{d\hat{a}_H}{dt} = -i\omega\hat{a}_H$$

**Solution:**
$$\boxed{\hat{a}_H(t) = \hat{a}_H(0)e^{-i\omega t}}$$
$$\boxed{\hat{a}^\dagger_H(t) = \hat{a}^\dagger_H(0)e^{i\omega t}}$$

The ladder operators rotate in the complex plane with frequency $\omega$.

---

## Physical Interpretation

### Operators as Dynamical Variables

In the Heisenberg picture, operators play the role of classical dynamical variables:
- $\hat{x}_H(t)$ is the "position at time t"
- $\hat{p}_H(t)$ is the "momentum at time t"

The state $|\psi_H\rangle$ encodes initial conditions and statistical properties.

### Classical Limit Made Transparent

The Heisenberg equations are structurally identical to Hamilton's equations:
$$\frac{dx}{dt} = \frac{p}{m}, \quad \frac{dp}{dt} = -\frac{\partial V}{\partial x}$$

The quantum corrections appear through:
1. Non-commutativity of operators
2. The uncertainty principle
3. Quantum fluctuations in expectation values

### Matrix Mechanics Origin

Historically, Heisenberg developed matrix mechanics before Schrodinger's wave mechanics:
- Heisenberg (1925): Observables are matrices that evolve in time
- Schrodinger (1926): States are wave functions that evolve in time
- Dirac (1926): Proved equivalence, introduced transformation theory

---

## Quantum Computing Connection

### Gate Synthesis via Heisenberg Evolution

Designing a quantum gate to implement a target unitary $\hat{U}_{\text{target}}$ can be approached via the Heisenberg picture:

1. Specify how operators should transform: $\hat{A} \to \hat{U}_{\text{target}}^\dagger\hat{A}\hat{U}_{\text{target}}$
2. Find a Hamiltonian $\hat{H}$ and time $t$ such that this transformation occurs

**Example: Hadamard Gate**
$$H: \sigma_z \to \sigma_x, \quad \sigma_x \to \sigma_z$$

### Clifford Gates and Stabilizer Formalism

In the stabilizer formalism:
- States are represented by their stabilizer group
- Clifford gates transform Pauli operators to Pauli operators
- The Heisenberg picture naturally describes how stabilizers evolve

This is efficient classical simulation of Clifford circuits.

### Error Propagation

Understanding how errors propagate through a quantum circuit is naturally done in the Heisenberg picture:
- An error $\hat{E}$ at the beginning becomes $\hat{U}^\dagger\hat{E}\hat{U}$ at the end
- Error correction codes are designed to detect transformed errors

---

## Worked Examples

### Example 1: Free Particle

**Problem:** Find $\hat{x}_H(t)$ and $\hat{p}_H(t)$ for a free particle.

**Solution:**

For $\hat{H} = \frac{\hat{p}^2}{2m}$:

**Momentum equation:**
$$\frac{d\hat{p}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{p}_H] = \frac{i}{2m\hbar}[\hat{p}^2, \hat{p}] = 0$$

$$\boxed{\hat{p}_H(t) = \hat{p}_H(0)}$$

Momentum is conserved (no force).

**Position equation:**
$$\frac{d\hat{x}_H}{dt} = \frac{\hat{p}_H}{m} = \frac{\hat{p}_H(0)}{m}$$

$$\boxed{\hat{x}_H(t) = \hat{x}_H(0) + \frac{\hat{p}_H(0)}{m}t}$$

This is the operator version of $x(t) = x_0 + v_0 t$. ∎

---

### Example 2: Verifying Commutator Preservation

**Problem:** For the harmonic oscillator, verify that $[\hat{x}_H(t), \hat{p}_H(t)] = i\hbar$.

**Solution:**

Using the solutions:
$$\hat{x}_H(t) = \hat{x}(0)\cos(\omega t) + \frac{\hat{p}(0)}{m\omega}\sin(\omega t)$$
$$\hat{p}_H(t) = \hat{p}(0)\cos(\omega t) - m\omega\hat{x}(0)\sin(\omega t)$$

Compute the commutator:
$$[\hat{x}_H(t), \hat{p}_H(t)] = [\hat{x}(0)\cos\omega t + \frac{\hat{p}(0)}{m\omega}\sin\omega t, \hat{p}(0)\cos\omega t - m\omega\hat{x}(0)\sin\omega t]$$

Expanding and using $[\hat{x}(0), \hat{p}(0)] = i\hbar$ and $[\hat{x}(0), \hat{x}(0)] = [\hat{p}(0), \hat{p}(0)] = 0$:

$$= \cos^2\omega t[\hat{x}(0), \hat{p}(0)] - \sin^2\omega t\cdot\frac{m\omega}{m\omega}[\hat{p}(0), \hat{x}(0)]$$
$$= \cos^2\omega t(i\hbar) + \sin^2\omega t(i\hbar)$$
$$= i\hbar(\cos^2\omega t + \sin^2\omega t) = i\hbar$$

$$\boxed{[\hat{x}_H(t), \hat{p}_H(t)] = i\hbar} \quad \checkmark$$ ∎

---

### Example 3: Spin Precession

**Problem:** For a spin-1/2 particle in a magnetic field $\vec{B} = B_0\hat{z}$, with $\hat{H} = \omega_L\hat{S}_z$ where $\omega_L = \gamma B_0$, find the Heisenberg evolution of $\hat{S}_x$, $\hat{S}_y$, $\hat{S}_z$.

**Solution:**

Using $[\hat{S}_z, \hat{S}_x] = i\hbar\hat{S}_y$ and $[\hat{S}_z, \hat{S}_y] = -i\hbar\hat{S}_x$:

**For $\hat{S}_z$:**
$$\frac{d\hat{S}_{z,H}}{dt} = \frac{i}{\hbar}\omega_L[\hat{S}_z, \hat{S}_z] = 0$$
$$\boxed{\hat{S}_{z,H}(t) = \hat{S}_z(0)}$$

**For $\hat{S}_x$:**
$$\frac{d\hat{S}_{x,H}}{dt} = \frac{i\omega_L}{\hbar}[\hat{S}_z, \hat{S}_x] = \frac{i\omega_L}{\hbar}(i\hbar\hat{S}_y) = -\omega_L\hat{S}_{y,H}$$

**For $\hat{S}_y$:**
$$\frac{d\hat{S}_{y,H}}{dt} = \frac{i\omega_L}{\hbar}[\hat{S}_z, \hat{S}_y] = \frac{i\omega_L}{\hbar}(-i\hbar\hat{S}_x) = \omega_L\hat{S}_{x,H}$$

**Combine:** $\frac{d^2\hat{S}_{x,H}}{dt^2} = -\omega_L^2\hat{S}_{x,H}$

**Solutions:**
$$\boxed{\hat{S}_{x,H}(t) = \hat{S}_x(0)\cos(\omega_L t) - \hat{S}_y(0)\sin(\omega_L t)}$$
$$\boxed{\hat{S}_{y,H}(t) = \hat{S}_y(0)\cos(\omega_L t) + \hat{S}_x(0)\sin(\omega_L t)}$$

The spin operators precess around the z-axis with the Larmor frequency $\omega_L$. ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Constant Potential:** For $\hat{H} = \frac{\hat{p}^2}{2m} + V_0$ (constant potential), find $\hat{x}_H(t)$ and $\hat{p}_H(t)$.

2. **Commutator Calculation:** Verify that $[\hat{a}_H(t), \hat{a}^\dagger_H(t)] = 1$ for the harmonic oscillator.

3. **Picture Transformation:** If $\hat{A}_S = \hat{x}^2$ in the Schrodinger picture, write $\hat{A}_H(t)$ for a free particle.

### Level 2: Intermediate

4. **Linear Potential:** For $\hat{H} = \frac{\hat{p}^2}{2m} + mgx$ (uniform gravity), solve for $\hat{x}_H(t)$ and $\hat{p}_H(t)$.

5. **Angular Momentum:** For $\hat{H} = \frac{\hat{L}^2}{2I}$ (free rotor), show that $\hat{L}_H(t) = \hat{L}(0)$ is constant.

6. **Number Operator:** For the harmonic oscillator, show that $\hat{n}_H(t) = \hat{a}^\dagger_H(t)\hat{a}_H(t) = \hat{n}(0)$ (constant).

### Level 3: Challenging

7. **Baker-Campbell-Hausdorff:** Use $e^{\hat{A}}\hat{B}e^{-\hat{A}} = \hat{B} + [\hat{A}, \hat{B}] + \frac{1}{2!}[\hat{A}, [\hat{A}, \hat{B}]] + \cdots$ to derive the Heisenberg operator for $\hat{x}$ in a harmonic oscillator.

8. **Forced Oscillator:** For $\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 - F(t)\hat{x}$, derive the Heisenberg equations and solve for constant $F$.

9. **Coupled Oscillators:** Two oscillators with $\hat{H} = \frac{\hat{p}_1^2 + \hat{p}_2^2}{2m} + \frac{1}{2}m\omega_0^2(\hat{x}_1^2 + \hat{x}_2^2) + \lambda\hat{x}_1\hat{x}_2$. Find the normal mode operators and their Heisenberg evolution.

---

## Computational Lab

### Objective
Implement Heisenberg picture evolution and compare with Schrodinger picture.

```python
"""
Day 362 Computational Lab: Heisenberg Picture
Quantum Mechanics Core - Year 1

This lab implements Heisenberg picture dynamics and compares
with Schrodinger picture for various systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# =============================================================================
# Part 1: Heisenberg Picture Basics
# =============================================================================

print("=" * 60)
print("Part 1: Heisenberg Picture - Two Level System")
print("=" * 60)

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# Hamiltonian: H = (hbar*omega/2) * sigma_z
hbar = 1.0
omega = 1.0
H = (hbar * omega / 2) * sigma_z

def U(t, H, hbar=1.0):
    """Time evolution operator."""
    return expm(-1j * H * t / hbar)

def heisenberg_operator(A_S, t, H, hbar=1.0):
    """Transform Schrodinger operator to Heisenberg picture."""
    Ut = U(t, H, hbar)
    return Ut.conj().T @ A_S @ Ut

# Initial state (for comparison)
psi_H = np.array([[1], [1]], dtype=complex) / np.sqrt(2)  # |+x>

# Time evolution
t_values = np.linspace(0, 4*np.pi/omega, 200)

# Heisenberg picture: evolve operators, fixed state
sigma_x_H = []
sigma_y_H = []
sigma_z_H = []

for t in t_values:
    sx_H = heisenberg_operator(sigma_x, t, H, hbar)
    sy_H = heisenberg_operator(sigma_y, t, H, hbar)
    sz_H = heisenberg_operator(sigma_z, t, H, hbar)

    # Expectation values with fixed state
    sigma_x_H.append(np.real((psi_H.conj().T @ sx_H @ psi_H)[0, 0]))
    sigma_y_H.append(np.real((psi_H.conj().T @ sy_H @ psi_H)[0, 0]))
    sigma_z_H.append(np.real((psi_H.conj().T @ sz_H @ psi_H)[0, 0]))

sigma_x_H = np.array(sigma_x_H)
sigma_y_H = np.array(sigma_y_H)
sigma_z_H = np.array(sigma_z_H)

# Schrodinger picture: evolve state, fixed operators
sigma_x_S = []
sigma_y_S = []
sigma_z_S = []

for t in t_values:
    psi_S = U(t, H, hbar) @ psi_H

    sigma_x_S.append(np.real((psi_S.conj().T @ sigma_x @ psi_S)[0, 0]))
    sigma_y_S.append(np.real((psi_S.conj().T @ sigma_y @ psi_S)[0, 0]))
    sigma_z_S.append(np.real((psi_S.conj().T @ sigma_z @ psi_S)[0, 0]))

sigma_x_S = np.array(sigma_x_S)
sigma_y_S = np.array(sigma_y_S)
sigma_z_S = np.array(sigma_z_S)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.plot(t_values * omega / (2*np.pi), sigma_x_H, 'b-', label='Heisenberg', linewidth=2)
ax1.plot(t_values * omega / (2*np.pi), sigma_x_S, 'r--', label='Schrodinger', linewidth=2)
ax1.set_xlabel('Time (periods)')
ax1.set_ylabel('<sigma_x>')
ax1.set_title('Comparison: <sigma_x>')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(t_values * omega / (2*np.pi), sigma_y_H, 'b-', label='Heisenberg', linewidth=2)
ax2.plot(t_values * omega / (2*np.pi), sigma_y_S, 'r--', label='Schrodinger', linewidth=2)
ax2.set_xlabel('Time (periods)')
ax2.set_ylabel('<sigma_y>')
ax2.set_title('Comparison: <sigma_y>')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Show difference (should be zero)
ax3 = axes[1, 0]
diff_x = sigma_x_H - sigma_x_S
diff_y = sigma_y_H - sigma_y_S
diff_z = sigma_z_H - sigma_z_S
ax3.plot(t_values, diff_x, 'b-', label='sigma_x', linewidth=1)
ax3.plot(t_values, diff_y, 'r-', label='sigma_y', linewidth=1)
ax3.plot(t_values, diff_z, 'g-', label='sigma_z', linewidth=1)
ax3.set_xlabel('Time')
ax3.set_ylabel('Heisenberg - Schrodinger')
ax3.set_title('Difference Between Pictures (Should Be Zero)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-1e-10, 1e-10)

# Operator matrix elements evolution
ax4 = axes[1, 1]
# Track matrix elements of sigma_x(t)
sx_00 = []
sx_01 = []
sx_11 = []

for t in t_values:
    sx_H = heisenberg_operator(sigma_x, t, H, hbar)
    sx_00.append(np.real(sx_H[0, 0]))
    sx_01.append(np.real(sx_H[0, 1]))
    sx_11.append(np.real(sx_H[1, 1]))

ax4.plot(t_values * omega / (2*np.pi), sx_00, 'b-', label='(sigma_x)_00', linewidth=2)
ax4.plot(t_values * omega / (2*np.pi), sx_01, 'r-', label='(sigma_x)_01', linewidth=2)
ax4.plot(t_values * omega / (2*np.pi), sx_11, 'g-', label='(sigma_x)_11', linewidth=2)
ax4.set_xlabel('Time (periods)')
ax4.set_ylabel('Matrix element')
ax4.set_title('Heisenberg: sigma_x(t) Matrix Elements')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_362_heisenberg_picture.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_362_heisenberg_picture.png'")

print(f"\nMax difference between pictures: {max(np.max(np.abs(diff_x)), np.max(np.abs(diff_y)), np.max(np.abs(diff_z))):.2e}")

# =============================================================================
# Part 2: Harmonic Oscillator in Heisenberg Picture
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Harmonic Oscillator - Heisenberg Picture")
print("=" * 60)

# Use truncated Fock space
N_fock = 20  # Number of Fock states

# Creation and annihilation operators
a = np.diag(np.sqrt(np.arange(1, N_fock)), 1)
a_dag = a.conj().T
n_op = a_dag @ a

# Position and momentum operators (scaled units: hbar = m = omega = 1)
x_op = (a + a_dag) / np.sqrt(2)
p_op = 1j * (a_dag - a) / np.sqrt(2)

# Hamiltonian
H_ho = n_op + 0.5 * np.eye(N_fock)

print(f"Fock space dimension: {N_fock}")
print(f"x and p operators defined in scaled units (hbar = m = omega = 1)")

# Initial state: coherent state (displaced ground state)
alpha = 2.0  # Coherent state parameter

# Coherent state |alpha> = exp(-|alpha|^2/2) sum_n (alpha^n/sqrt(n!)) |n>
psi_coherent = np.zeros((N_fock, 1), dtype=complex)
for n in range(N_fock):
    psi_coherent[n] = alpha**n / np.sqrt(np.math.factorial(n))
psi_coherent *= np.exp(-np.abs(alpha)**2 / 2)
psi_coherent = psi_coherent / np.linalg.norm(psi_coherent)

print(f"Initial coherent state with alpha = {alpha}")
print(f"Expected <x>(0) = sqrt(2)*Re(alpha) = {np.sqrt(2)*np.real(alpha):.4f}")

# Time evolution
T_ho = 2 * np.pi  # Period (omega = 1)
t_ho = np.linspace(0, 2*T_ho, 200)

# Heisenberg picture
x_H_expect = []
p_H_expect = []

for t in t_ho:
    x_H = heisenberg_operator(x_op, t, H_ho, hbar=1.0)
    p_H = heisenberg_operator(p_op, t, H_ho, hbar=1.0)

    x_H_expect.append(np.real((psi_coherent.conj().T @ x_H @ psi_coherent)[0, 0]))
    p_H_expect.append(np.real((psi_coherent.conj().T @ p_H @ psi_coherent)[0, 0]))

x_H_expect = np.array(x_H_expect)
p_H_expect = np.array(p_H_expect)

# Analytical solution (scaled units)
x_analytical = np.sqrt(2) * np.real(alpha) * np.cos(t_ho) + np.sqrt(2) * np.imag(alpha) * np.sin(t_ho)
p_analytical = -np.sqrt(2) * np.real(alpha) * np.sin(t_ho) + np.sqrt(2) * np.imag(alpha) * np.cos(t_ho)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.plot(t_ho / T_ho, x_H_expect, 'b-', label='Heisenberg <x_H>', linewidth=2)
ax1.plot(t_ho / T_ho, x_analytical, 'r--', label='Analytical', linewidth=2)
ax1.set_xlabel('Time (periods)')
ax1.set_ylabel('<x>')
ax1.set_title('Position Expectation Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(t_ho / T_ho, p_H_expect, 'b-', label='Heisenberg <p_H>', linewidth=2)
ax2.plot(t_ho / T_ho, p_analytical, 'r--', label='Analytical', linewidth=2)
ax2.set_xlabel('Time (periods)')
ax2.set_ylabel('<p>')
ax2.set_title('Momentum Expectation Value')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Phase space
ax3 = axes[1, 0]
ax3.plot(x_H_expect, p_H_expect, 'b-', label='Quantum', linewidth=2)
ax3.plot(x_analytical, p_analytical, 'r--', label='Classical', linewidth=2)
ax3.set_xlabel('<x>')
ax3.set_ylabel('<p>')
ax3.set_title('Phase Space Trajectory')
ax3.legend()
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# Commutator preservation
commutators = []
for t in t_ho:
    x_H = heisenberg_operator(x_op, t, H_ho, hbar=1.0)
    p_H = heisenberg_operator(p_op, t, H_ho, hbar=1.0)
    comm = x_H @ p_H - p_H @ x_H
    commutators.append(np.trace(comm) / N_fock)  # Should be i*hbar

ax4 = axes[1, 1]
ax4.plot(t_ho / T_ho, np.imag(commutators), 'g-', linewidth=2)
ax4.axhline(y=1.0, color='r', linestyle='--', label='Expected: i*hbar = i')
ax4.set_xlabel('Time (periods)')
ax4.set_ylabel('Im([x_H, p_H]) per state')
ax4.set_title('Commutator Preservation: [x_H(t), p_H(t)] = i*hbar')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_362_harmonic_oscillator_heisenberg.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_362_harmonic_oscillator_heisenberg.png'")

# =============================================================================
# Part 3: Ladder Operator Evolution
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Ladder Operator Evolution a_H(t) = a(0)e^(-i*omega*t)")
print("=" * 60)

# Track a_H(t) matrix elements
a_00 = []
a_01 = []
a_10 = []
phase_expected = []

for t in t_ho:
    a_H = heisenberg_operator(a, t, H_ho, hbar=1.0)
    a_00.append(a_H[0, 0])
    a_01.append(a_H[0, 1])
    a_10.append(a_H[1, 0])
    phase_expected.append(np.exp(-1j * t))

a_01 = np.array(a_01)
phase_expected = np.array(phase_expected)

# a_01 should be 1 * e^(-i*omega*t) = e^(-it) (since omega=1)
print(f"Expected: a_01(t) = 1 * e^(-i*omega*t)")
print(f"At t=0: a_01(0) = {a_01[0]:.4f}, expected = 1")
print(f"At t=T/4: a_01(T/4) = {a_01[len(t_ho)//8]:.4f}, expected = e^(-i*pi/2) = -i")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(t_ho / T_ho, np.real(a_01), 'b-', label='Re(a_01)', linewidth=2)
ax1.plot(t_ho / T_ho, np.imag(a_01), 'r-', label='Im(a_01)', linewidth=2)
ax1.plot(t_ho / T_ho, np.real(phase_expected), 'b--', alpha=0.5, linewidth=1)
ax1.plot(t_ho / T_ho, np.imag(phase_expected), 'r--', alpha=0.5, linewidth=1)
ax1.set_xlabel('Time (periods)')
ax1.set_ylabel('a_01(t) components')
ax1.set_title('Ladder Operator: a_H(t) = a(0)e^{-i*omega*t}')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(np.real(a_01), np.imag(a_01), 'purple', linewidth=2)
ax2.plot(np.real(a_01[0]), np.imag(a_01[0]), 'go', markersize=10, label='t=0')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
ax2.add_patch(circle)
ax2.set_xlabel('Re(a_01)')
ax2.set_ylabel('Im(a_01)')
ax2.set_title('a_01(t) in Complex Plane (Rotates with Frequency omega)')
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_362_ladder_operator_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_362_ladder_operator_evolution.png'")

# =============================================================================
# Part 4: Summary
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Heisenberg Picture Summary")
print("=" * 60)

print("""
Heisenberg Picture Summary:
---------------------------
1. States are fixed: |psi_H> = |psi(0)>
2. Operators evolve: A_H(t) = U^dag(t) A_S U(t)
3. Equation of motion: dA_H/dt = (i/hbar)[H, A_H]
4. Classical correspondence: {A, H} <-> (1/i*hbar)[A, H]
5. Commutators preserved: [x_H(t), p_H(t)] = i*hbar

Key results for harmonic oscillator:
- x_H(t) = x(0)cos(wt) + (p(0)/mw)sin(wt)
- p_H(t) = p(0)cos(wt) - mw*x(0)sin(wt)
- a_H(t) = a(0)e^(-i*w*t)
- a^dag_H(t) = a^dag(0)e^(i*w*t)
""")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Operator transformation | $\hat{A}_H(t) = \hat{U}^\dagger(t)\hat{A}_S\hat{U}(t)$ |
| Heisenberg equation | $\frac{d\hat{A}_H}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{A}_H] + (\frac{\partial\hat{A}}{\partial t})_H$ |
| Position evolution | $\frac{d\hat{x}_H}{dt} = \frac{\hat{p}_H}{m}$ |
| Momentum evolution | $\frac{d\hat{p}_H}{dt} = -\frac{\partial V}{\partial\hat{x}}$ |
| Commutator preservation | $[\hat{x}_H(t), \hat{p}_H(t)] = i\hbar$ |
| Ladder operator | $\hat{a}_H(t) = \hat{a}(0)e^{-i\omega t}$ |

### Main Takeaways

1. **Heisenberg picture:** States are fixed, operators evolve
2. **The Heisenberg equation** is the quantum analog of Hamilton's equation
3. **Commutation relations** are preserved under time evolution
4. **The classical correspondence** is most transparent in this picture
5. **Expectation values** are identical in both pictures
6. **Ladder operators** rotate in the complex plane with frequency $\omega$

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.7 and Sakurai Chapter 2.2
- [ ] Derive the Heisenberg equation of motion
- [ ] Solve for $\hat{x}_H(t)$ and $\hat{p}_H(t)$ for harmonic oscillator
- [ ] Verify commutator preservation
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and understand the computational lab
- [ ] Compare Heisenberg and Schrodinger pictures numerically

---

## Preview: Day 363

Tomorrow we introduce the **Interaction picture**, which combines features of both Schrodinger and Heisenberg pictures. It's essential for time-dependent perturbation theory and quantum field theory.

---

*"In quantum mechanics, it's a waste of time to dispute about which representation is the correct one."*
— Paul Dirac

---

**Next:** [Day_363_Saturday.md](Day_363_Saturday.md) — Interaction Picture
