# Day 363: The Interaction Picture — Split Evolution for Perturbation Theory

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Interaction Picture |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 363, you will be able to:

1. Define the Interaction picture and explain its utility
2. Transform states and operators between all three pictures
3. Derive the equation of motion in the Interaction picture
4. Apply the Interaction picture to time-dependent perturbation problems
5. Understand the Dyson series for the time-evolution operator
6. Connect the Interaction picture to Fermi's Golden Rule

---

## Core Content

### 1. Motivation: Why a Third Picture?

Consider a Hamiltonian with a "simple" part and a "perturbation":
$$\hat{H} = \hat{H}_0 + \hat{V}(t)$$

where:
- $\hat{H}_0$ is the **unperturbed Hamiltonian** (exactly solvable)
- $\hat{V}(t)$ is the **perturbation** (possibly time-dependent)

**The Problem:**
- Schrodinger picture: Hard to solve when $\hat{V}(t)$ is present
- Heisenberg picture: Operator evolution is complicated

**The Solution:** The **Interaction picture** (also called **Dirac picture**):
- Let $\hat{H}_0$ handle part of the evolution
- Focus on the effects of $\hat{V}(t)$

---

### 2. The Interaction Picture: Definition

**In the Interaction picture:**

1. **States** carry the perturbation dynamics:
$$|\psi_I(t)\rangle = e^{i\hat{H}_0 t/\hbar}|\psi_S(t)\rangle = \hat{U}_0^\dagger(t)|\psi_S(t)\rangle$$

2. **Operators** carry the free evolution:
$$\hat{A}_I(t) = e^{i\hat{H}_0 t/\hbar}\hat{A}_S e^{-i\hat{H}_0 t/\hbar} = \hat{U}_0^\dagger(t)\hat{A}_S\hat{U}_0(t)$$

where $\hat{U}_0(t) = e^{-i\hat{H}_0 t/\hbar}$ is the **free evolution operator**.

**Key Insight:** Operators evolve as if only $\hat{H}_0$ were present; states evolve due to $\hat{V}$.

---

### 3. Relations Between Pictures

| Quantity | Schrodinger | Heisenberg | Interaction |
|----------|-------------|------------|-------------|
| State | $\|\psi_S(t)\rangle = \hat{U}(t)\|\psi(0)\rangle$ | $\|\psi_H\rangle = \|\psi(0)\rangle$ | $\|\psi_I(t)\rangle = \hat{U}_0^\dagger(t)\|\psi_S(t)\rangle$ |
| Operator | $\hat{A}_S$ (fixed) | $\hat{A}_H(t) = \hat{U}^\dagger\hat{A}_S\hat{U}$ | $\hat{A}_I(t) = \hat{U}_0^\dagger\hat{A}_S\hat{U}_0$ |

**At $t = 0$:** All three pictures coincide:
$$|\psi_S(0)\rangle = |\psi_H\rangle = |\psi_I(0)\rangle$$
$$\hat{A}_S = \hat{A}_H(0) = \hat{A}_I(0)$$

---

### 4. Equation of Motion for Operators

The Interaction picture operator evolves according to:

$$\frac{d\hat{A}_I}{dt} = \frac{i}{\hbar}[\hat{H}_0, \hat{A}_I] + \left(\frac{\partial\hat{A}_S}{\partial t}\right)_I$$

This is the **Heisenberg equation with $\hat{H}_0$ only** — the perturbation $\hat{V}$ does not appear!

---

### 5. Equation of Motion for States

The crucial equation of the Interaction picture governs the state evolution:

**Derivation:**

Start with $|\psi_I(t)\rangle = \hat{U}_0^\dagger(t)|\psi_S(t)\rangle$

$$\frac{d}{dt}|\psi_I\rangle = \frac{d\hat{U}_0^\dagger}{dt}|\psi_S\rangle + \hat{U}_0^\dagger\frac{d|\psi_S\rangle}{dt}$$

Using $\frac{d\hat{U}_0^\dagger}{dt} = \frac{i}{\hbar}\hat{H}_0\hat{U}_0^\dagger$ and the Schrodinger equation:

$$= \frac{i}{\hbar}\hat{H}_0\hat{U}_0^\dagger|\psi_S\rangle + \hat{U}_0^\dagger\frac{1}{i\hbar}(\hat{H}_0 + \hat{V})|\psi_S\rangle$$

$$= \frac{i}{\hbar}\hat{H}_0|\psi_I\rangle - \frac{i}{\hbar}\hat{U}_0^\dagger\hat{H}_0|\psi_S\rangle - \frac{i}{\hbar}\hat{U}_0^\dagger\hat{V}|\psi_S\rangle$$

The $\hat{H}_0$ terms cancel! We're left with:

$$\boxed{i\hbar\frac{d}{dt}|\psi_I(t)\rangle = \hat{V}_I(t)|\psi_I(t)\rangle}$$

where $\hat{V}_I(t) = \hat{U}_0^\dagger(t)\hat{V}(t)\hat{U}_0(t) = e^{i\hat{H}_0 t/\hbar}\hat{V}(t)e^{-i\hat{H}_0 t/\hbar}$

This is the **Interaction picture Schrodinger equation** — states evolve only due to the perturbation!

---

### 6. The Interaction Picture Evolution Operator

Define $\hat{U}_I(t, t_0)$ such that:
$$|\psi_I(t)\rangle = \hat{U}_I(t, t_0)|\psi_I(t_0)\rangle$$

From the equation of motion:
$$i\hbar\frac{\partial}{\partial t}\hat{U}_I(t, t_0) = \hat{V}_I(t)\hat{U}_I(t, t_0)$$

with initial condition $\hat{U}_I(t_0, t_0) = \hat{I}$.

**Integral Form:**
$$\hat{U}_I(t, t_0) = \hat{I} - \frac{i}{\hbar}\int_{t_0}^t \hat{V}_I(t')\hat{U}_I(t', t_0)dt'$$

---

### 7. The Dyson Series

Iterating the integral equation gives the **Dyson series**:

$$\boxed{\hat{U}_I(t, t_0) = \mathcal{T}\exp\left(-\frac{i}{\hbar}\int_{t_0}^t \hat{V}_I(t')dt'\right)}$$

where $\mathcal{T}$ is the **time-ordering operator**.

**Explicit expansion:**
$$\hat{U}_I(t, t_0) = \hat{I} + \sum_{n=1}^{\infty}\left(\frac{-i}{\hbar}\right)^n \int_{t_0}^t dt_1 \int_{t_0}^{t_1} dt_2 \cdots \int_{t_0}^{t_{n-1}} dt_n \, \hat{V}_I(t_1)\hat{V}_I(t_2)\cdots\hat{V}_I(t_n)$$

**First-order approximation:**
$$\hat{U}_I(t, t_0) \approx \hat{I} - \frac{i}{\hbar}\int_{t_0}^t \hat{V}_I(t')dt'$$

This is the basis for **time-dependent perturbation theory**.

---

### 8. Transition Amplitudes

The transition amplitude from initial state $|i\rangle$ to final state $|f\rangle$ is:

$$c_{f \leftarrow i}(t) = \langle f|\hat{U}_I(t, 0)|i\rangle$$

**To first order:**
$$c_{f \leftarrow i}^{(1)}(t) = -\frac{i}{\hbar}\int_0^t \langle f|\hat{V}_I(t')|i\rangle dt'$$

Using $\hat{V}_I(t) = e^{i\hat{H}_0 t/\hbar}\hat{V}e^{-i\hat{H}_0 t/\hbar}$:

$$= -\frac{i}{\hbar}\int_0^t \langle f|\hat{V}|i\rangle e^{i(E_f - E_i)t'/\hbar} dt'$$

$$\boxed{c_{f \leftarrow i}^{(1)}(t) = -\frac{i}{\hbar}V_{fi}\int_0^t e^{i\omega_{fi}t'} dt'}$$

where $V_{fi} = \langle f|\hat{V}|i\rangle$ and $\omega_{fi} = (E_f - E_i)/\hbar$.

---

### 9. Preview: Fermi's Golden Rule

For a constant perturbation turned on at $t = 0$:

$$c_{f \leftarrow i}^{(1)}(t) = -\frac{i}{\hbar}V_{fi}\frac{e^{i\omega_{fi}t} - 1}{i\omega_{fi}}$$

The transition probability is:
$$P_{f \leftarrow i}(t) = |c_{f \leftarrow i}|^2 = \frac{|V_{fi}|^2}{\hbar^2}\frac{4\sin^2(\omega_{fi}t/2)}{\omega_{fi}^2}$$

For long times ($t \to \infty$), using $\lim_{t\to\infty}\frac{\sin^2(\omega t/2)}{\omega^2 t} = \frac{\pi}{2}\delta(\omega)$:

$$\boxed{\Gamma_{f \leftarrow i} = \frac{2\pi}{\hbar}|V_{fi}|^2\delta(E_f - E_i)}$$

This is **Fermi's Golden Rule** — the transition rate to states of energy $E_f = E_i$.

---

## Physical Interpretation

### Separation of Dynamics

The Interaction picture cleanly separates two types of evolution:

| Component | Evolution | Physical Meaning |
|-----------|-----------|------------------|
| Operators | $\hat{A}_I(t)$ via $\hat{H}_0$ | "Free" motion (known) |
| States | $\|\psi_I(t)\rangle$ via $\hat{V}$ | Effect of perturbation (what we want to calculate) |

### When to Use Each Picture

| Picture | Best For |
|---------|----------|
| Schrodinger | Time-independent problems, visualizing probability |
| Heisenberg | Algebraic manipulations, classical limit |
| Interaction | Time-dependent perturbations, scattering, QFT |

### The Role of Time-Ordering

The time-ordering operator $\mathcal{T}$ is necessary because $[\hat{V}_I(t_1), \hat{V}_I(t_2)] \neq 0$ in general. Time-ordering ensures causality — earlier events affect later ones.

---

## Quantum Computing Connection

### Control Pulses in the Interaction Picture

When controlling qubits with microwave pulses:
- $\hat{H}_0 = \frac{\hbar\omega_0}{2}\sigma_z$ (qubit Hamiltonian)
- $\hat{V}(t) = \hbar\Omega(t)\cos(\omega_d t)\sigma_x$ (drive)

In the Interaction picture (rotating frame):
$$\hat{V}_I(t) = \frac{\hbar\Omega(t)}{2}\left[e^{i(\omega_0 - \omega_d)t}\sigma_+ + e^{-i(\omega_0 - \omega_d)t}\sigma_-\right]$$

When $\omega_d \approx \omega_0$ (resonance), the perturbation becomes nearly constant — this is the **rotating wave approximation**.

### Rabi Oscillations

For a resonant drive:
$$i\hbar\frac{d}{dt}|\psi_I\rangle = \frac{\hbar\Omega}{2}\sigma_x|\psi_I\rangle$$

The state oscillates between $|0\rangle$ and $|1\rangle$ at the **Rabi frequency** $\Omega$.

### Gate Implementation

Single-qubit gates are implemented by:
1. Applying a pulse $\hat{V}(t)$ for time $T$
2. The evolution in the Interaction picture gives $\hat{U}_I(T, 0)$
3. This implements the gate $\hat{U}_{\text{gate}} = \hat{U}_0(T)\hat{U}_I(T, 0)$

---

## Worked Examples

### Example 1: Two-Level System with Harmonic Perturbation

**Problem:** A two-level atom with $\hat{H}_0 = \frac{\hbar\omega_0}{2}\sigma_z$ is subject to a perturbation $\hat{V}(t) = \hbar\Omega\cos(\omega t)\sigma_x$ starting at $t = 0$. Find $\hat{V}_I(t)$.

**Solution:**

First, note that:
$$\hat{U}_0(t) = e^{-i\omega_0 t\sigma_z/2} = \cos(\omega_0 t/2)\hat{I} - i\sin(\omega_0 t/2)\sigma_z = \begin{pmatrix} e^{-i\omega_0 t/2} & 0 \\ 0 & e^{i\omega_0 t/2} \end{pmatrix}$$

Transform $\sigma_x$:
$$\sigma_{x,I}(t) = \hat{U}_0^\dagger(t)\sigma_x\hat{U}_0(t)$$

Using $\hat{U}_0^\dagger \sigma_x \hat{U}_0 = \cos(\omega_0 t)\sigma_x + \sin(\omega_0 t)\sigma_y$:

$$\hat{V}_I(t) = \hbar\Omega\cos(\omega t)[\cos(\omega_0 t)\sigma_x + \sin(\omega_0 t)\sigma_y]$$

Using product-to-sum formulas:
$$= \frac{\hbar\Omega}{2}[(\cos((\omega_0 - \omega)t) + \cos((\omega_0 + \omega)t))\sigma_x + (\sin((\omega_0 + \omega)t) - \sin((\omega_0 - \omega)t))\sigma_y]$$

**Rotating Wave Approximation (RWA):** When $\omega \approx \omega_0$, the $(\omega_0 + \omega)$ terms oscillate rapidly and average to zero. Keeping only the slowly varying terms:

$$\boxed{\hat{V}_I^{\text{RWA}}(t) \approx \frac{\hbar\Omega}{2}[\cos(\Delta t)\sigma_x - \sin(\Delta t)\sigma_y]}$$

where $\Delta = \omega_0 - \omega$ is the **detuning**. ∎

---

### Example 2: First-Order Transition Probability

**Problem:** For the system in Example 1 at resonance ($\omega = \omega_0$), calculate the transition probability $P_{1 \leftarrow 0}(t)$ starting from $|0\rangle$.

**Solution:**

At resonance ($\Delta = 0$), the RWA perturbation is:
$$\hat{V}_I^{\text{RWA}} = \frac{\hbar\Omega}{2}\sigma_x = \frac{\hbar\Omega}{2}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

This is time-independent in the RWA!

The first-order transition amplitude:
$$c_{1 \leftarrow 0}^{(1)}(t) = -\frac{i}{\hbar}\int_0^t \langle 1|\hat{V}_I|0\rangle dt' = -\frac{i}{\hbar}\cdot\frac{\hbar\Omega}{2}\cdot t = -\frac{i\Omega t}{2}$$

$$P_{1 \leftarrow 0}^{(1)}(t) = \left|\frac{\Omega t}{2}\right|^2 = \frac{\Omega^2 t^2}{4}$$

**Note:** This is only valid for short times $\Omega t \ll 1$.

**Exact solution:** The full evolution gives Rabi oscillations:
$$P_{1 \leftarrow 0}(t) = \sin^2\left(\frac{\Omega t}{2}\right)$$

The first-order result matches the small-angle expansion: $\sin^2(\Omega t/2) \approx (\Omega t/2)^2$ for small $\Omega t$. ∎

---

### Example 3: Sudden Turn-On of Perturbation

**Problem:** A system is in eigenstate $|i\rangle$ of $\hat{H}_0$. At $t = 0$, a constant perturbation $\hat{V}$ is suddenly turned on. Find the first-order transition probability to state $|f\rangle$ at time $t$.

**Solution:**

The Interaction picture perturbation is:
$$\hat{V}_I(t) = e^{i\hat{H}_0 t/\hbar}\hat{V}e^{-i\hat{H}_0 t/\hbar}$$

The matrix element:
$$\langle f|\hat{V}_I(t)|i\rangle = e^{i(E_f - E_i)t/\hbar}\langle f|\hat{V}|i\rangle = V_{fi}e^{i\omega_{fi}t}$$

First-order amplitude:
$$c_{f \leftarrow i}^{(1)}(t) = -\frac{i}{\hbar}\int_0^t V_{fi}e^{i\omega_{fi}t'}dt' = -\frac{V_{fi}}{\hbar\omega_{fi}}(e^{i\omega_{fi}t} - 1)$$

$$= -\frac{V_{fi}}{\hbar\omega_{fi}}e^{i\omega_{fi}t/2}(e^{i\omega_{fi}t/2} - e^{-i\omega_{fi}t/2}) = \frac{-2iV_{fi}}{\hbar\omega_{fi}}e^{i\omega_{fi}t/2}\sin(\omega_{fi}t/2)$$

Probability:
$$P_{f \leftarrow i}(t) = |c_{f \leftarrow i}^{(1)}|^2 = \frac{4|V_{fi}|^2}{\hbar^2\omega_{fi}^2}\sin^2\left(\frac{\omega_{fi}t}{2}\right)$$

$$\boxed{P_{f \leftarrow i}(t) = \frac{|V_{fi}|^2}{\hbar^2}\frac{\sin^2((E_f - E_i)t/2\hbar)}{((E_f - E_i)/2\hbar)^2}}$$

This is sharply peaked at $E_f = E_i$ (energy conservation). ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Picture Transformation:** Given $|\psi_S(t)\rangle = e^{-i\omega t/2}|0\rangle + e^{i\omega t/2}|1\rangle$ for $\hat{H}_0 = \frac{\hbar\omega}{2}\sigma_z$, find $|\psi_I(t)\rangle$.

2. **Operator Transformation:** For $\hat{H}_0 = \hbar\omega(\hat{a}^\dagger\hat{a} + \frac{1}{2})$, find $\hat{a}_I(t)$ and $\hat{a}^\dagger_I(t)$.

3. **Constant Perturbation:** For a constant $\hat{V}$ with $V_{fi} \neq 0$, show that the first-order transition probability oscillates in time.

### Level 2: Intermediate

4. **Rotating Frame:** For a two-level system driven at frequency $\omega$, show that the transformation $|\psi_R\rangle = e^{i\omega t\sigma_z/2}|\psi_S\rangle$ leads to a time-independent effective Hamiltonian (in the RWA).

5. **Second-Order Correction:** Derive the second-order term in the Dyson series and apply it to calculate the energy shift of a level due to off-resonant coupling.

6. **Harmonic Oscillator Perturbation:** For $\hat{H}_0 = \hbar\omega\hat{a}^\dagger\hat{a}$ and $\hat{V} = \lambda(\hat{a} + \hat{a}^\dagger)$, find $\hat{V}_I(t)$ and the first-order transition amplitude from $|n\rangle$ to $|n\pm1\rangle$.

### Level 3: Challenging

7. **Fermi's Golden Rule Derivation:** Starting from the first-order transition probability, derive Fermi's Golden Rule for transitions to a continuum of final states with density $\rho(E)$.

8. **Dyson Time-Ordering:** Prove that the time-ordered exponential $\mathcal{T}\exp(-\frac{i}{\hbar}\int_0^t\hat{V}dt')$ satisfies the differential equation $i\hbar\frac{\partial\hat{U}}{\partial t} = \hat{V}(t)\hat{U}$.

9. **Bloch Equations:** Starting from the Interaction picture equation for a two-level atom coupled to a field, derive the optical Bloch equations including phenomenological decay terms.

---

## Computational Lab

### Objective
Implement the Interaction picture and simulate Rabi oscillations.

```python
"""
Day 363 Computational Lab: Interaction Picture
Quantum Mechanics Core - Year 1

This lab implements the Interaction picture and simulates
time-dependent perturbation problems including Rabi oscillations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# =============================================================================
# Part 1: Interaction Picture Basics
# =============================================================================

print("=" * 60)
print("Part 1: Interaction Picture - Two Level System")
print("=" * 60)

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# Parameters
hbar = 1.0
omega_0 = 5.0  # Transition frequency
Omega = 0.5    # Rabi frequency

# H_0 = (hbar * omega_0 / 2) * sigma_z
H_0 = (hbar * omega_0 / 2) * sigma_z

def U_0(t, H_0, hbar=1.0):
    """Free evolution operator."""
    return expm(-1j * H_0 * t / hbar)

def to_interaction_picture_state(psi_S, t, H_0, hbar=1.0):
    """Transform state from Schrodinger to Interaction picture."""
    return U_0(t, H_0, hbar).conj().T @ psi_S

def to_interaction_picture_operator(A_S, t, H_0, hbar=1.0):
    """Transform operator from Schrodinger to Interaction picture."""
    U = U_0(t, H_0, hbar)
    return U.conj().T @ A_S @ U

# Initial state: ground state |0>
psi_0 = np.array([[1], [0]], dtype=complex)

# Verify picture equivalence at various times
print("\nVerifying picture equivalence:")
t_test = 1.0

# Schrodinger picture
psi_S = U_0(t_test, H_0, hbar) @ psi_0
exp_sigma_z_S = np.real((psi_S.conj().T @ sigma_z @ psi_S)[0, 0])

# Interaction picture (with no perturbation, state is constant)
psi_I = psi_0  # No perturbation, so constant
sigma_z_I = to_interaction_picture_operator(sigma_z, t_test, H_0, hbar)
exp_sigma_z_I = np.real((psi_I.conj().T @ sigma_z_I @ psi_I)[0, 0])

print(f"At t = {t_test}:")
print(f"  Schrodinger: <sigma_z> = {exp_sigma_z_S:.6f}")
print(f"  Interaction: <sigma_z> = {exp_sigma_z_I:.6f}")
print(f"  Match: {np.isclose(exp_sigma_z_S, exp_sigma_z_I)}")

# =============================================================================
# Part 2: Rabi Oscillations
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Rabi Oscillations - Resonant Driving")
print("=" * 60)

def rabi_evolution(Omega, omega_0, omega_d, t_max, n_points=500):
    """
    Simulate Rabi oscillations using the Interaction picture.

    Parameters:
    -----------
    Omega : float
        Rabi frequency
    omega_0 : float
        Transition frequency
    omega_d : float
        Drive frequency
    t_max : float
        Maximum simulation time
    n_points : int
        Number of time points
    """
    t_values = np.linspace(0, t_max, n_points)
    dt = t_values[1] - t_values[0]

    # Detuning
    Delta = omega_0 - omega_d

    # Initial state in Interaction picture
    psi_I = np.array([1.0, 0.0], dtype=complex)  # |0>

    # Track probabilities
    P_0 = [np.abs(psi_I[0])**2]
    P_1 = [np.abs(psi_I[1])**2]

    for i in range(1, len(t_values)):
        t = t_values[i]

        # V_I(t) in the rotating wave approximation
        # V_I = (hbar*Omega/2) * [cos(Delta*t)*sigma_x - sin(Delta*t)*sigma_y]
        # Equivalently: V_I = (hbar*Omega/2) * [e^{-i*Delta*t}*sigma_+ + e^{i*Delta*t}*sigma_-]

        # For small dt, use first-order evolution
        # |psi_I(t+dt)> = |psi_I(t)> - (i/hbar)*V_I(t)*|psi_I(t)>*dt

        # Using RWA Hamiltonian in rotating frame
        # H_RWA = (hbar/2) * [Delta*sigma_z + Omega*sigma_x]
        H_RWA = (hbar / 2) * (Delta * sigma_z + Omega * sigma_x)

        # Full evolution (not just first order)
        U_RWA = expm(-1j * H_RWA * dt / hbar)
        psi_I = U_RWA @ psi_I

        P_0.append(np.abs(psi_I[0])**2)
        P_1.append(np.abs(psi_I[1])**2)

    return t_values, np.array(P_0), np.array(P_1)

# Resonant driving (Delta = 0)
t_values, P_0_res, P_1_res = rabi_evolution(Omega, omega_0, omega_0, t_max=30, n_points=1000)

# Off-resonant driving
Delta_off = Omega  # Detuning equal to Rabi frequency
t_values_off, P_0_off, P_1_off = rabi_evolution(Omega, omega_0, omega_0 - Delta_off, t_max=30, n_points=1000)

# Analytical solutions
# Resonant: P_1 = sin^2(Omega*t/2)
P_1_analytical_res = np.sin(Omega * t_values / 2)**2

# Off-resonant: P_1 = (Omega/Omega_eff)^2 * sin^2(Omega_eff*t/2)
Omega_eff = np.sqrt(Omega**2 + Delta_off**2)
P_1_analytical_off = (Omega / Omega_eff)**2 * np.sin(Omega_eff * t_values_off / 2)**2

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.plot(t_values * Omega / (2*np.pi), P_0_res, 'b-', label='P(|0>)', linewidth=2)
ax1.plot(t_values * Omega / (2*np.pi), P_1_res, 'r-', label='P(|1>)', linewidth=2)
ax1.plot(t_values * Omega / (2*np.pi), P_1_analytical_res, 'k--', label='Analytical', linewidth=1)
ax1.set_xlabel('Time (Rabi periods)')
ax1.set_ylabel('Probability')
ax1.set_title(f'Resonant Driving (Delta = 0)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(t_values_off * Omega / (2*np.pi), P_0_off, 'b-', label='P(|0>)', linewidth=2)
ax2.plot(t_values_off * Omega / (2*np.pi), P_1_off, 'r-', label='P(|1>)', linewidth=2)
ax2.plot(t_values_off * Omega / (2*np.pi), P_1_analytical_off, 'k--', label='Analytical', linewidth=1)
ax2.set_xlabel('Time (Rabi periods)')
ax2.set_ylabel('Probability')
ax2.set_title(f'Off-Resonant Driving (Delta = Omega)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Detuning scan (Rabi amplitude vs detuning)
detunings = np.linspace(-3*Omega, 3*Omega, 100)
max_P1 = []

for Delta in detunings:
    Omega_eff = np.sqrt(Omega**2 + Delta**2)
    max_P1.append((Omega / Omega_eff)**2)

ax3 = axes[1, 0]
ax3.plot(detunings / Omega, max_P1, 'purple', linewidth=2)
ax3.set_xlabel('Detuning Delta/Omega')
ax3.set_ylabel('Maximum P(|1>)')
ax3.set_title('Rabi Amplitude vs Detuning (Lorentzian)')
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax3.grid(True, alpha=0.3)

# Bloch sphere trajectory (resonant)
def bloch_coords(psi):
    """Extract Bloch sphere coordinates."""
    rho = np.outer(psi, psi.conj())
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])
    return x, y, z

ax4 = axes[1, 1]

# Compute Bloch trajectory
bloch_x, bloch_y, bloch_z = [], [], []
psi = np.array([1.0, 0.0], dtype=complex)
H_RWA = (hbar / 2) * Omega * sigma_x  # Resonant
dt = t_values[1] - t_values[0]

for t in t_values[:500]:
    bx, by, bz = bloch_coords(psi)
    bloch_x.append(bx)
    bloch_y.append(by)
    bloch_z.append(bz)

    U = expm(-1j * H_RWA * dt / hbar)
    psi = U @ psi

ax4.plot(bloch_x, bloch_z, 'blue', linewidth=2)
ax4.plot(bloch_x[0], bloch_z[0], 'go', markersize=10, label='Start')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
ax4.add_patch(circle)
ax4.set_xlabel('Bloch x')
ax4.set_ylabel('Bloch z')
ax4.set_title('Bloch Sphere: Resonant Rabi (xz-plane)')
ax4.set_xlim(-1.2, 1.2)
ax4.set_ylim(-1.2, 1.2)
ax4.set_aspect('equal')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_363_rabi_oscillations.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_363_rabi_oscillations.png'")

# =============================================================================
# Part 3: First-Order Perturbation Theory
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: First-Order Perturbation Theory")
print("=" * 60)

def first_order_transition_probability(V_fi, omega_fi, t):
    """
    First-order transition probability.

    P = (4 |V_fi|^2 / hbar^2) * sin^2(omega_fi * t / 2) / omega_fi^2
    """
    if np.abs(omega_fi) < 1e-10:
        return (np.abs(V_fi)**2 / hbar**2) * t**2
    else:
        return (4 * np.abs(V_fi)**2 / hbar**2) * np.sin(omega_fi * t / 2)**2 / omega_fi**2

# Parameters for transition
E_i = 0
E_f_values = np.linspace(-2, 2, 200)  # Different final energies
V_fi = 0.1 * hbar

t_short = 5.0
t_long = 50.0

P_short = []
P_long = []

for E_f in E_f_values:
    omega_fi = (E_f - E_i) / hbar
    P_short.append(first_order_transition_probability(V_fi, omega_fi, t_short))
    P_long.append(first_order_transition_probability(V_fi, omega_fi, t_long))

P_short = np.array(P_short)
P_long = np.array(P_long)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(E_f_values, P_short, 'b-', label=f't = {t_short}', linewidth=2)
ax1.plot(E_f_values, P_long, 'r-', label=f't = {t_long}', linewidth=2)
ax1.axvline(x=E_i, color='k', linestyle='--', alpha=0.5, label='E_i (resonance)')
ax1.set_xlabel('Final energy E_f')
ax1.set_ylabel('Transition probability P(f<-i)')
ax1.set_title('First-Order Perturbation: Energy Dependence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Time dependence at resonance
t_resonance = np.linspace(0, 20, 100)
P_resonance = [(np.abs(V_fi)**2 / hbar**2) * t**2 for t in t_resonance]

# Compare with exact (Rabi) at small perturbation
P_exact = [np.sin(np.abs(V_fi) * t / hbar)**2 for t in t_resonance]

ax2 = axes[1]
ax2.plot(t_resonance, P_resonance, 'b-', label='First-order perturbation', linewidth=2)
ax2.plot(t_resonance, P_exact, 'r--', label='Exact (Rabi)', linewidth=2)
ax2.set_xlabel('Time')
ax2.set_ylabel('P(f<-i) at resonance')
ax2.set_title('Perturbation Theory vs Exact (Resonance)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_363_perturbation_theory.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_363_perturbation_theory.png'")

# =============================================================================
# Part 4: Fermi's Golden Rule Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Fermi's Golden Rule")
print("=" * 60)

# The function sin^2(omega*t/2) / omega^2 becomes a delta function as t -> infinity

omega_range = np.linspace(-10, 10, 1000)
times = [1, 5, 20, 100]

fig, ax = plt.subplots(figsize=(10, 6))

for t in times:
    f = np.sin(omega_range * t / 2)**2 / (omega_range**2 + 1e-10) / t
    ax.plot(omega_range, f, label=f't = {t}', linewidth=1.5)

ax.set_xlabel('omega')
ax.set_ylabel('sin^2(omega*t/2) / (omega^2 * t)')
ax.set_title('Transition Probability: Approaching Delta Function')
ax.legend()
ax.set_xlim(-10, 10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_363_fermi_golden_rule.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_363_fermi_golden_rule.png'")

print("\nFermi's Golden Rule: Gamma = (2*pi/hbar) * |V_fi|^2 * delta(E_f - E_i)")
print("For continuum: Gamma = (2*pi/hbar) * |V_fi|^2 * rho(E)")

# =============================================================================
# Part 5: Summary
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Interaction Picture Summary")
print("=" * 60)

print("""
Interaction Picture Summary:
----------------------------
1. States carry perturbation dynamics: |psi_I(t)> = U_0^dag(t)|psi_S(t)>
2. Operators carry free evolution: A_I(t) = U_0^dag(t) A_S U_0(t)
3. Equation of motion: i*hbar*d|psi_I>/dt = V_I(t)|psi_I>
4. First-order amplitude: c_fi^(1) = -(i/hbar) integral V_fi e^{i*omega_fi*t} dt
5. Fermi's Golden Rule: Gamma = (2*pi/hbar)|V_fi|^2 * delta(E_f - E_i)

Key applications:
- Time-dependent perturbation theory
- Rabi oscillations and quantum control
- Scattering theory
- Quantum field theory
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
| State transformation | $\|\psi_I(t)\rangle = e^{i\hat{H}_0 t/\hbar}\|\psi_S(t)\rangle$ |
| Operator transformation | $\hat{A}_I(t) = e^{i\hat{H}_0 t/\hbar}\hat{A}_S e^{-i\hat{H}_0 t/\hbar}$ |
| Interaction picture equation | $i\hbar\frac{d}{dt}\|\psi_I\rangle = \hat{V}_I(t)\|\psi_I\rangle$ |
| First-order amplitude | $c_{fi}^{(1)} = -\frac{i}{\hbar}\int_0^t V_{fi}e^{i\omega_{fi}t'}dt'$ |
| Dyson series | $\hat{U}_I = \mathcal{T}\exp(-\frac{i}{\hbar}\int\hat{V}_I dt)$ |
| Fermi's Golden Rule | $\Gamma = \frac{2\pi}{\hbar}\|V_{fi}\|^2\delta(E_f - E_i)$ |

### Main Takeaways

1. **Interaction picture** separates free and perturbed evolution
2. **Operators evolve** with $\hat{H}_0$; **states evolve** with $\hat{V}$
3. **The Dyson series** provides a systematic perturbative expansion
4. **First-order perturbation** gives transition amplitudes directly
5. **Fermi's Golden Rule** emerges for long-time transitions
6. **Rabi oscillations** are naturally described in this picture

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.8 and Sakurai Chapter 2.3
- [ ] Derive the Interaction picture equation of motion
- [ ] Compute $\hat{V}_I(t)$ for a two-level system
- [ ] Derive the first-order transition amplitude
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and understand the computational lab
- [ ] Simulate Rabi oscillations numerically

---

## Preview: Day 364

Tomorrow is the **Month 13 Capstone** — a comprehensive review and assessment covering all postulates and time evolution concepts. We'll synthesize everything learned and practice with qualifying exam-style problems.

---

*"The interaction picture is particularly useful when we wish to treat the time-dependent part of the Hamiltonian as a perturbation."*
— J. J. Sakurai

---

**Next:** [Day_364_Sunday.md](Day_364_Sunday.md) — Month 13 Capstone
