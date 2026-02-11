# Day 361: The Schrodinger Picture — States Evolve, Operators Fixed

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Schrodinger Picture |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 361, you will be able to:

1. Define the Schrodinger picture and its key characteristics
2. Write the equations of motion for states in this picture
3. Compute time-dependent expectation values
4. Derive Ehrenfest's theorem
5. Show equivalence between pictures for observable quantities
6. Recognize when the Schrodinger picture is most useful

---

## Core Content

### 1. What is a "Picture" in Quantum Mechanics?

In quantum mechanics, physical predictions depend only on **expectation values**:
$$\langle\hat{A}\rangle = \langle\psi(t)|\hat{A}|\psi(t)\rangle$$

There are infinitely many ways to distribute the time dependence between states and operators while keeping expectation values unchanged. The three most useful choices are:

| Picture | States | Operators |
|---------|--------|-----------|
| **Schrodinger** | Evolve | Fixed |
| **Heisenberg** | Fixed | Evolve |
| **Interaction** | Partial | Partial |

Today we formalize the **Schrodinger picture**, which we've been using implicitly throughout this course.

---

### 2. The Schrodinger Picture: Definition

**In the Schrodinger picture:**

1. **State vectors** carry all the time dependence:
$$|\psi_S(t)\rangle = \hat{U}(t)|\psi_S(0)\rangle = e^{-i\hat{H}t/\hbar}|\psi_S(0)\rangle$$

2. **Operators** are time-independent (unless they have explicit time dependence):
$$\hat{A}_S = \hat{A}_S(0)$$

3. **Basis states** are time-independent:
$$|n\rangle_S = |n\rangle$$

The subscript "S" denotes Schrodinger picture quantities.

---

### 3. Equation of Motion for States

The fundamental equation of the Schrodinger picture is the time-dependent Schrodinger equation:

$$\boxed{i\hbar\frac{\partial}{\partial t}|\psi_S(t)\rangle = \hat{H}|\psi_S(t)\rangle}$$

**Key Features:**
- First-order in time
- Linear in the state
- Deterministic evolution

The formal solution is:
$$|\psi_S(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi_S(0)\rangle$$

---

### 4. Time Evolution of Expectation Values

Even though operators are fixed, expectation values can still change because the state evolves:

$$\langle\hat{A}\rangle(t) = \langle\psi_S(t)|\hat{A}_S|\psi_S(t)\rangle$$

**Derivation of time evolution:**

$$\frac{d}{dt}\langle\hat{A}\rangle = \frac{d}{dt}\langle\psi_S(t)|\hat{A}_S|\psi_S(t)\rangle$$

Using the product rule:
$$= \left\langle\frac{\partial\psi_S}{\partial t}\middle|\hat{A}_S\middle|\psi_S\right\rangle + \langle\psi_S|\hat{A}_S\left|\frac{\partial\psi_S}{\partial t}\right\rangle + \left\langle\psi_S\left|\frac{\partial\hat{A}_S}{\partial t}\right|\psi_S\right\rangle$$

From the Schrodinger equation: $\frac{\partial|\psi_S\rangle}{\partial t} = \frac{1}{i\hbar}\hat{H}|\psi_S\rangle$

And its adjoint: $\frac{\partial\langle\psi_S|}{\partial t} = -\frac{1}{i\hbar}\langle\psi_S|\hat{H}$

Substituting:
$$\frac{d}{dt}\langle\hat{A}\rangle = -\frac{1}{i\hbar}\langle\psi_S|\hat{H}\hat{A}_S|\psi_S\rangle + \frac{1}{i\hbar}\langle\psi_S|\hat{A}_S\hat{H}|\psi_S\rangle + \left\langle\frac{\partial\hat{A}_S}{\partial t}\right\rangle$$

$$\boxed{\frac{d}{dt}\langle\hat{A}\rangle = \frac{i}{\hbar}\langle[\hat{H}, \hat{A}_S]\rangle + \left\langle\frac{\partial\hat{A}_S}{\partial t}\right\rangle}$$

This is **Ehrenfest's theorem** in the Schrodinger picture.

---

### 5. Ehrenfest's Theorem

Ehrenfest's theorem connects quantum expectation values to classical equations of motion.

**For position and momentum:**

$$\boxed{\frac{d\langle\hat{x}\rangle}{dt} = \frac{\langle\hat{p}\rangle}{m}}$$

$$\boxed{\frac{d\langle\hat{p}\rangle}{dt} = -\left\langle\frac{\partial V}{\partial x}\right\rangle}$$

**Derivation of the first equation:**

Using $[\hat{H}, \hat{x}] = \frac{1}{2m}[\hat{p}^2, \hat{x}] = \frac{1}{2m}([\hat{p}, \hat{x}]\hat{p} + \hat{p}[\hat{p}, \hat{x}]) = \frac{-i\hbar}{m}\hat{p}$

$$\frac{d\langle\hat{x}\rangle}{dt} = \frac{i}{\hbar}\langle[\hat{H}, \hat{x}]\rangle = \frac{i}{\hbar}\cdot\frac{-i\hbar}{m}\langle\hat{p}\rangle = \frac{\langle\hat{p}\rangle}{m}$$

**Classical Correspondence:**

Compare to Newton's equations:
$$\frac{dx}{dt} = \frac{p}{m}, \quad \frac{dp}{dt} = -\frac{\partial V}{\partial x}$$

The **expectation values** obey the classical equations! This is why macroscopic objects appear classical.

**Important Caveat:**
$$\left\langle\frac{\partial V}{\partial x}\right\rangle \neq \frac{\partial V(\langle x\rangle)}{\partial\langle x\rangle} \quad \text{(in general)}$$

The correspondence is exact only when $V(x)$ is at most quadratic (free particle, harmonic oscillator).

---

### 6. Conservation Laws in the Schrodinger Picture

An observable $\hat{A}$ is **conserved** if $\frac{d}{dt}\langle\hat{A}\rangle = 0$.

From Ehrenfest's theorem, this requires:
1. $[\hat{H}, \hat{A}] = 0$ (commutes with Hamiltonian)
2. $\frac{\partial\hat{A}}{\partial t} = 0$ (no explicit time dependence)

**Examples of Conserved Quantities:**

| System | Conserved Quantity | Why |
|--------|-------------------|-----|
| Any | Energy $\hat{H}$ | $[\hat{H}, \hat{H}] = 0$ |
| Free particle | Momentum $\hat{p}$ | $[\hat{H}, \hat{p}] = 0$ when $V = 0$ |
| Central potential | Angular momentum $\hat{L}^2$, $\hat{L}_z$ | Rotational symmetry |
| Parity symmetric | Parity $\hat{\Pi}$ | $[\hat{H}, \hat{\Pi}] = 0$ |

---

### 7. The Schrodinger Picture Wave Function

In the position representation, the state $|\psi_S(t)\rangle$ is represented by the wave function:

$$\psi_S(x, t) = \langle x|\psi_S(t)\rangle$$

The Schrodinger equation becomes:
$$i\hbar\frac{\partial\psi_S(x, t)}{\partial t} = \hat{H}\psi_S(x, t) = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi_S(x, t)$$

**Boundary Conditions** and **normalization** are imposed on $\psi_S(x, t)$.

---

### 8. Advantages of the Schrodinger Picture

The Schrodinger picture is most useful when:

1. **Solving for wave functions** — the TISE gives energy eigenstates directly
2. **Visualizing dynamics** — probability density $|\psi(x,t)|^2$ shows how the "particle" moves
3. **Initial value problems** — start with $|\psi(0)\rangle$ and propagate forward
4. **Perturbation theory** — both time-independent and time-dependent versions are naturally formulated here

**Limitations:**
- Basis states are fixed, making some symmetries less obvious
- Connection to classical mechanics less transparent than Heisenberg picture

---

## Physical Interpretation

### The State as Physical Reality

In the Schrodinger picture, the state vector $|\psi_S(t)\rangle$ is treated as the fundamental object. It contains complete information about the system.

**Philosophical Perspective:**
- **Copenhagen interpretation:** $|\psi\rangle$ encodes our knowledge/predictions
- **Many-worlds:** $|\psi\rangle$ is the physical reality (deterministically evolving)
- **Pilot wave:** $|\psi\rangle$ guides actual particles

### Classical Limit

Ehrenfest's theorem shows that:
- Expectation values follow classical equations
- The classical limit emerges when:
  - Wave packets are narrow compared to $V(x)$ variations
  - Quantum corrections $\propto \hbar$ are negligible
  - Decoherence destroys superpositions

### Information Flow

The unitary evolution $|\psi(t)\rangle = \hat{U}(t)|\psi(0)\rangle$ preserves information:
- The initial state can be recovered: $|\psi(0)\rangle = \hat{U}^{-1}(t)|\psi(t)\rangle$
- Entropy is constant for isolated systems
- Measurements are the only source of irreversibility

---

## Quantum Computing Connection

### Quantum Circuits in the Schrodinger Picture

In the Schrodinger picture, quantum computation is viewed as:
1. **Initialize** state: $|\psi(0)\rangle = |00\cdots0\rangle$
2. **Apply** sequence of gates: $|\psi\rangle \to \hat{U}_n \cdots \hat{U}_2\hat{U}_1|\psi\rangle$
3. **Measure** in the computational basis

The gates are fixed operations; the state transforms.

### State Tomography

Quantum state tomography reconstructs $|\psi_S(t)\rangle$ by:
1. Preparing many copies of the state
2. Measuring different observables
3. Reconstructing the density matrix

This is inherently a Schrodinger picture concept — we're asking "what is the state?"

### Variational Quantum Eigensolver (VQE)

VQE works by:
1. Parameterizing a state: $|\psi(\theta)\rangle$
2. Computing $\langle\psi(\theta)|\hat{H}|\psi(\theta)\rangle$ (energy expectation)
3. Optimizing $\theta$ to minimize energy

The variational principle is naturally expressed in the Schrodinger picture.

---

## Worked Examples

### Example 1: Ehrenfest's Theorem for Harmonic Oscillator

**Problem:** For a harmonic oscillator with $\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$, derive the equations of motion for $\langle\hat{x}\rangle$ and $\langle\hat{p}\rangle$.

**Solution:**

**For position:**
$$\frac{d\langle\hat{x}\rangle}{dt} = \frac{i}{\hbar}\langle[\hat{H}, \hat{x}]\rangle$$

We need $[\hat{H}, \hat{x}] = \frac{1}{2m}[\hat{p}^2, \hat{x}]$ (the potential term commutes with $\hat{x}$).

Using $[\hat{p}^2, \hat{x}] = \hat{p}[\hat{p}, \hat{x}] + [\hat{p}, \hat{x}]\hat{p} = -2i\hbar\hat{p}$:
$$[\hat{H}, \hat{x}] = \frac{-i\hbar\hat{p}}{m}$$

$$\frac{d\langle\hat{x}\rangle}{dt} = \frac{i}{\hbar}\cdot\frac{-i\hbar}{m}\langle\hat{p}\rangle = \frac{\langle\hat{p}\rangle}{m}$$

**For momentum:**
$$\frac{d\langle\hat{p}\rangle}{dt} = \frac{i}{\hbar}\langle[\hat{H}, \hat{p}]\rangle$$

We need $[\hat{H}, \hat{p}] = \frac{m\omega^2}{2}[\hat{x}^2, \hat{p}]$ (the kinetic term commutes with $\hat{p}$).

Using $[\hat{x}^2, \hat{p}] = \hat{x}[\hat{x}, \hat{p}] + [\hat{x}, \hat{p}]\hat{x} = 2i\hbar\hat{x}$:
$$[\hat{H}, \hat{p}] = i\hbar m\omega^2\hat{x}$$

$$\frac{d\langle\hat{p}\rangle}{dt} = \frac{i}{\hbar}\cdot i\hbar m\omega^2\langle\hat{x}\rangle = -m\omega^2\langle\hat{x}\rangle$$

**Combined:**
$$\boxed{\frac{d^2\langle\hat{x}\rangle}{dt^2} = -\omega^2\langle\hat{x}\rangle}$$

This is exactly the classical harmonic oscillator equation! The expectation value oscillates:
$$\langle\hat{x}\rangle(t) = A\cos(\omega t + \phi)$$ ∎

---

### Example 2: Free Particle Wave Packet Spreading

**Problem:** A free particle Gaussian wave packet has initial width $\sigma_0$. Using the Schrodinger picture, show that the width grows as:
$$\sigma(t) = \sigma_0\sqrt{1 + \left(\frac{\hbar t}{2m\sigma_0^2}\right)^2}$$

**Solution:**

The initial wave function:
$$\psi_S(x, 0) = \frac{1}{(2\pi\sigma_0^2)^{1/4}}e^{-x^2/4\sigma_0^2}e^{ik_0 x}$$

For a free particle, the momentum eigenstates $e^{ipx/\hbar}$ evolve as $e^{ipx/\hbar}e^{-ip^2 t/2m\hbar}$.

The wave packet in momentum space is:
$$\tilde{\psi}(p) \propto e^{-(p-\hbar k_0)^2\sigma_0^2/\hbar^2}$$

After time evolution:
$$\psi_S(x, t) = \int \tilde{\psi}(p)e^{ipx/\hbar}e^{-ip^2 t/2m\hbar}\frac{dp}{2\pi\hbar}$$

Computing the integral (completing the square in the Gaussian):
$$|\psi_S(x, t)|^2 = \frac{1}{\sqrt{2\pi\sigma(t)^2}}e^{-(x - \hbar k_0 t/m)^2/2\sigma(t)^2}$$

where
$$\sigma(t)^2 = \sigma_0^2 + \frac{\hbar^2 t^2}{4m^2\sigma_0^2}$$

$$\boxed{\sigma(t) = \sigma_0\sqrt{1 + \left(\frac{\hbar t}{2m\sigma_0^2}\right)^2}}$$

The spreading timescale is $\tau = 2m\sigma_0^2/\hbar$.

For an electron with $\sigma_0 = 1$ nm: $\tau \approx 10^{-14}$ s (femtoseconds!).
For a dust grain with $\sigma_0 = 1$ nm: $\tau \approx 10^{14}$ s (millions of years). ∎

---

### Example 3: Conservation of Energy

**Problem:** Prove that $\langle\hat{H}\rangle$ is constant in time for a time-independent Hamiltonian.

**Solution:**

From Ehrenfest's theorem:
$$\frac{d\langle\hat{H}\rangle}{dt} = \frac{i}{\hbar}\langle[\hat{H}, \hat{H}]\rangle + \left\langle\frac{\partial\hat{H}}{\partial t}\right\rangle$$

Since $[\hat{H}, \hat{H}] = 0$ and $\frac{\partial\hat{H}}{\partial t} = 0$:

$$\frac{d\langle\hat{H}\rangle}{dt} = 0$$

Therefore, $\langle\hat{H}\rangle$ is constant — **energy is conserved**.

$$\boxed{\langle\hat{H}\rangle = \text{constant}}$$ ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Commutator Calculation:** Compute $[\hat{H}, \hat{p}]$ for $\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})$ and use it to derive $\frac{d\langle\hat{p}\rangle}{dt} = -\langle\frac{\partial V}{\partial x}\rangle$.

2. **Conservation Law:** For a free particle, show that $\langle\hat{p}\rangle$ and $\langle\hat{p}^2\rangle$ are both conserved.

3. **Energy Conservation:** Verify that $\langle\hat{H}\rangle$ is constant for the state $|\psi(t)\rangle = \frac{1}{\sqrt{2}}(e^{-iE_1 t/\hbar}|1\rangle + e^{-iE_2 t/\hbar}|2\rangle)$.

### Level 2: Intermediate

4. **Ehrenfest for Angular Momentum:** For a central potential $V(r)$, show that $\frac{d\langle\hat{L}_z\rangle}{dt} = 0$.

5. **Virial Theorem:** For a power-law potential $V \propto r^n$, use $\frac{d}{dt}\langle\hat{x}\cdot\hat{p}\rangle = 0$ (at equilibrium) to derive $2\langle T \rangle = n\langle V \rangle$.

6. **Classical Limit:** For $V(x) = \frac{1}{24}V_0 x^4$, show that $\frac{d\langle\hat{p}\rangle}{dt} \neq -\frac{\partial V(\langle x\rangle)}{\partial\langle x\rangle}$ and explain why the classical correspondence breaks down.

### Level 3: Challenging

7. **Spreading Rate:** Derive $\frac{d\sigma_x^2}{dt}$ for a free particle in terms of the covariance $\langle\hat{x}\hat{p} + \hat{p}\hat{x}\rangle/2 - \langle\hat{x}\rangle\langle\hat{p}\rangle$.

8. **Heisenberg Inequality from Ehrenfest:** Use $\frac{d}{dt}\langle\hat{x}^2\rangle = \frac{1}{m}\langle\hat{x}\hat{p} + \hat{p}\hat{x}\rangle$ and the uncertainty principle to bound the spreading rate.

9. **Gauge Invariance:** In the presence of a magnetic field, $\hat{H} = \frac{1}{2m}(\hat{p} - e\vec{A})^2 + V$. Show that physical observables depend on $\vec{B} = \nabla \times \vec{A}$, not on $\vec{A}$ directly.

---

## Computational Lab

### Objective
Implement Schrodinger picture evolution and verify Ehrenfest's theorem numerically.

```python
"""
Day 361 Computational Lab: Schrodinger Picture
Quantum Mechanics Core - Year 1

This lab implements Schrodinger picture dynamics and verifies
Ehrenfest's theorem for various potentials.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# =============================================================================
# Part 1: Schrodinger Picture for Two-Level System
# =============================================================================

print("=" * 60)
print("Part 1: Schrodinger Picture - Two Level System")
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

# Initial state: |+x> = (|0> + |1>)/sqrt(2)
psi_0 = np.array([[1], [1]], dtype=complex) / np.sqrt(2)

print(f"Hamiltonian H = (hbar*omega/2) * sigma_z")
print(f"Initial state: |+x> = (|0> + |1>)/sqrt(2)")

def evolve_schrodinger(psi_0, H, t, hbar=1.0):
    """Evolve state in Schrodinger picture."""
    U = expm(-1j * H * t / hbar)
    return U @ psi_0

def expectation_value(psi, A):
    """Compute <psi|A|psi>."""
    return np.real((psi.conj().T @ A @ psi)[0, 0])

# Time evolution
t_values = np.linspace(0, 4*np.pi/omega, 200)

# Track expectation values of sigma_x, sigma_y, sigma_z
exp_x = []
exp_y = []
exp_z = []

for t in t_values:
    psi_t = evolve_schrodinger(psi_0, H, t, hbar)
    exp_x.append(expectation_value(psi_t, sigma_x))
    exp_y.append(expectation_value(psi_t, sigma_y))
    exp_z.append(expectation_value(psi_t, sigma_z))

exp_x = np.array(exp_x)
exp_y = np.array(exp_y)
exp_z = np.array(exp_z)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Expectation values vs time
ax1 = axes[0, 0]
ax1.plot(t_values * omega / (2*np.pi), exp_x, 'b-', label='<sigma_x>', linewidth=2)
ax1.plot(t_values * omega / (2*np.pi), exp_y, 'r-', label='<sigma_y>', linewidth=2)
ax1.plot(t_values * omega / (2*np.pi), exp_z, 'g-', label='<sigma_z>', linewidth=2)
ax1.set_xlabel('Time (periods)')
ax1.set_ylabel('Expectation value')
ax1.set_title('Schrodinger Picture: Expectation Values')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Analytical comparison
# For H ~ sigma_z, state precesses around z-axis
# <sigma_x>(t) = cos(omega*t), <sigma_y>(t) = -sin(omega*t), <sigma_z> = 0
ax2 = axes[0, 1]
ax2.plot(t_values, exp_x, 'b-', label='Numerical', linewidth=2)
ax2.plot(t_values, np.cos(omega * t_values), 'b--', label='cos(omega*t)', linewidth=2)
ax2.plot(t_values, exp_y, 'r-', linewidth=2)
ax2.plot(t_values, -np.sin(omega * t_values), 'r--', linewidth=2)
ax2.set_xlabel('Time')
ax2.set_ylabel('<sigma_x> (blue), <sigma_y> (red)')
ax2.set_title('Comparison with Analytical Solution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Energy conservation
H_expect = [expectation_value(evolve_schrodinger(psi_0, H, t, hbar), H) for t in t_values]
ax3 = axes[1, 0]
ax3.plot(t_values * omega / (2*np.pi), H_expect, 'g-', linewidth=2)
ax3.set_xlabel('Time (periods)')
ax3.set_ylabel('<H>')
ax3.set_title('Energy Conservation: <H> vs Time')
ax3.set_ylim(np.mean(H_expect) - 0.1, np.mean(H_expect) + 0.1)
ax3.grid(True, alpha=0.3)

# Bloch sphere trajectory
ax4 = axes[1, 1]
ax4.plot(exp_x, exp_y, 'purple', linewidth=2)
ax4.plot(exp_x[0], exp_y[0], 'go', markersize=10, label='Start')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
ax4.add_patch(circle)
ax4.set_xlabel('<sigma_x>')
ax4.set_ylabel('<sigma_y>')
ax4.set_title('Bloch Sphere: xy-plane')
ax4.set_xlim(-1.2, 1.2)
ax4.set_ylim(-1.2, 1.2)
ax4.set_aspect('equal')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_361_schrodinger_picture.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_361_schrodinger_picture.png'")

# =============================================================================
# Part 2: Ehrenfest's Theorem for Harmonic Oscillator
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Ehrenfest's Theorem - Harmonic Oscillator")
print("=" * 60)

# Solve Schrodinger equation numerically for a wave packet
# in a harmonic oscillator potential

# Parameters
m = 1.0
omega_osc = 1.0
N_x = 200
x_max = 10.0
x = np.linspace(-x_max, x_max, N_x)
dx = x[1] - x[0]

# Potential
V = 0.5 * m * omega_osc**2 * x**2

# Initial wave packet (displaced Gaussian)
x0 = 2.0  # Initial displacement
sigma0 = 1.0 / np.sqrt(m * omega_osc / hbar)  # Ground state width
k0 = 0.0  # Initial momentum (stationary)

psi_initial = np.exp(-(x - x0)**2 / (4 * sigma0**2)) / (2 * np.pi * sigma0**2)**0.25

# Normalize
norm = np.sqrt(np.trapz(np.abs(psi_initial)**2, x))
psi_initial = psi_initial / norm

print(f"Initial displacement: x0 = {x0}")
print(f"Initial width: sigma0 = {sigma0:.4f}")

# Build Hamiltonian matrix (finite difference)
diag_main = hbar**2 / (m * dx**2) + V
diag_off = -hbar**2 / (2 * m * dx**2) * np.ones(N_x - 1)

H_matrix = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)

# Time evolution using matrix exponential (small steps for accuracy)
def evolve_wave_packet(psi, dt, H_matrix, hbar=1.0):
    """Single time step evolution."""
    return expm(-1j * H_matrix * dt / hbar) @ psi

# Track expectation values
T_period = 2 * np.pi / omega_osc
dt = T_period / 100
N_t = 200
t_values = np.arange(N_t) * dt

x_expect_qm = []
p_expect_qm = []

psi = psi_initial.copy()
for i in range(N_t):
    # Compute expectation values
    prob = np.abs(psi)**2
    x_exp = np.trapz(x * prob, x)

    # Momentum expectation (using gradient)
    dpsi = np.gradient(psi, dx)
    p_exp = np.real(np.trapz(np.conj(psi) * (-1j * hbar * dpsi), x))

    x_expect_qm.append(x_exp)
    p_expect_qm.append(p_exp)

    # Evolve
    psi = evolve_wave_packet(psi, dt, H_matrix, hbar)

x_expect_qm = np.array(x_expect_qm)
p_expect_qm = np.array(p_expect_qm)

# Classical solution
x_classical = x0 * np.cos(omega_osc * t_values)
p_classical = -m * omega_osc * x0 * np.sin(omega_osc * t_values)

# Compute d<x>/dt numerically
dx_dt_numerical = np.gradient(x_expect_qm, dt)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# <x>(t)
ax1 = axes[0, 0]
ax1.plot(t_values / T_period, x_expect_qm, 'b-', label='Quantum <x>', linewidth=2)
ax1.plot(t_values / T_period, x_classical, 'r--', label='Classical x(t)', linewidth=2)
ax1.set_xlabel('Time (periods)')
ax1.set_ylabel('<x>')
ax1.set_title('Position: Quantum vs Classical')
ax1.legend()
ax1.grid(True, alpha=0.3)

# <p>(t)
ax2 = axes[0, 1]
ax2.plot(t_values / T_period, p_expect_qm, 'b-', label='Quantum <p>', linewidth=2)
ax2.plot(t_values / T_period, p_classical, 'r--', label='Classical p(t)', linewidth=2)
ax2.set_xlabel('Time (periods)')
ax2.set_ylabel('<p>')
ax2.set_title('Momentum: Quantum vs Classical')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Ehrenfest: d<x>/dt = <p>/m
ax3 = axes[1, 0]
ax3.plot(t_values / T_period, dx_dt_numerical, 'b-', label='d<x>/dt (numerical)', linewidth=2)
ax3.plot(t_values / T_period, p_expect_qm / m, 'r--', label='<p>/m', linewidth=2)
ax3.set_xlabel('Time (periods)')
ax3.set_ylabel('Velocity')
ax3.set_title('Ehrenfest: d<x>/dt = <p>/m')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Phase space trajectory
ax4 = axes[1, 1]
ax4.plot(x_expect_qm, p_expect_qm, 'b-', label='Quantum', linewidth=2)
ax4.plot(x_classical, p_classical, 'r--', label='Classical', linewidth=2)
ax4.set_xlabel('<x>')
ax4.set_ylabel('<p>')
ax4.set_title('Phase Space Trajectory')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

plt.tight_layout()
plt.savefig('day_361_ehrenfest_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_361_ehrenfest_theorem.png'")

# =============================================================================
# Part 3: Wave Packet Spreading (Free Particle)
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Wave Packet Spreading (Free Particle)")
print("=" * 60)

# Free particle: V = 0
V_free = np.zeros_like(x)

# Initial Gaussian wave packet
sigma0_free = 0.5
x0_free = 0.0
k0_free = 5.0  # Nonzero initial momentum

psi_free = np.exp(-(x - x0_free)**2 / (4 * sigma0_free**2)) * np.exp(1j * k0_free * x)
psi_free = psi_free / np.sqrt(np.trapz(np.abs(psi_free)**2, x))

# Hamiltonian for free particle
H_free = np.diag(hbar**2 / (m * dx**2) * np.ones(N_x)) + \
         np.diag(-hbar**2 / (2 * m * dx**2) * np.ones(N_x - 1), 1) + \
         np.diag(-hbar**2 / (2 * m * dx**2) * np.ones(N_x - 1), -1)

# Time evolution
spreading_time = 2 * m * sigma0_free**2 / hbar
t_max = 3 * spreading_time
dt_free = t_max / 100
t_free = np.arange(0, t_max, dt_free)

widths = []
psi = psi_free.copy()

for t in t_free:
    prob = np.abs(psi)**2

    # Compute width
    x_mean = np.trapz(x * prob, x)
    x2_mean = np.trapz(x**2 * prob, x)
    sigma = np.sqrt(x2_mean - x_mean**2)
    widths.append(sigma)

    # Evolve
    psi = evolve_wave_packet(psi, dt_free, H_free, hbar)

widths = np.array(widths)

# Theoretical width
sigma_theory = sigma0_free * np.sqrt(1 + (hbar * t_free / (2 * m * sigma0_free**2))**2)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(t_free / spreading_time, widths, 'b-', label='Numerical', linewidth=2)
ax1.plot(t_free / spreading_time, sigma_theory, 'r--', label='Theoretical', linewidth=2)
ax1.set_xlabel('Time (spreading times)')
ax1.set_ylabel('Width sigma')
ax1.set_title('Wave Packet Spreading')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Snapshots at different times
ax2 = axes[1]
psi = psi_free.copy()
times_snapshot = [0, 0.5 * spreading_time, spreading_time, 2 * spreading_time]

for t_snap in times_snapshot:
    # Evolve to this time
    psi_temp = psi_free.copy()
    n_steps = int(t_snap / dt_free)
    for _ in range(n_steps):
        psi_temp = evolve_wave_packet(psi_temp, dt_free, H_free, hbar)

    prob = np.abs(psi_temp)**2
    ax2.plot(x, prob, label=f't = {t_snap/spreading_time:.1f} tau', linewidth=1.5)

ax2.set_xlabel('Position')
ax2.set_ylabel('|psi|^2')
ax2.set_title('Wave Packet Spreading Snapshots')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-5, 15)

plt.tight_layout()
plt.savefig('day_361_wave_packet_spreading.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_361_wave_packet_spreading.png'")

print("\n" + "=" * 60)
print("Part 4: Summary of Schrodinger Picture Properties")
print("=" * 60)

print("""
Schrodinger Picture Summary:
----------------------------
1. States evolve: |psi(t)> = U(t)|psi(0)>
2. Operators fixed: A_S = constant
3. Equation of motion: i*hbar * d|psi>/dt = H|psi>
4. Ehrenfest: d<A>/dt = (i/hbar)<[H,A]> + <dA/dt>
5. Energy conservation: d<H>/dt = 0

This picture is ideal for:
- Solving for wave functions
- Visualizing probability dynamics
- Initial value problems
- Time-dependent perturbation theory
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
| State evolution | $\|\psi_S(t)\rangle = e^{-i\hat{H}t/\hbar}\|\psi_S(0)\rangle$ |
| Operators | $\hat{A}_S = \text{constant}$ |
| Ehrenfest's theorem | $\frac{d\langle\hat{A}\rangle}{dt} = \frac{i}{\hbar}\langle[\hat{H}, \hat{A}]\rangle + \langle\frac{\partial\hat{A}}{\partial t}\rangle$ |
| Position dynamics | $\frac{d\langle\hat{x}\rangle}{dt} = \frac{\langle\hat{p}\rangle}{m}$ |
| Momentum dynamics | $\frac{d\langle\hat{p}\rangle}{dt} = -\langle\frac{\partial V}{\partial x}\rangle$ |
| Conservation | $[\hat{H}, \hat{A}] = 0 \Rightarrow \frac{d\langle\hat{A}\rangle}{dt} = 0$ |

### Main Takeaways

1. **Schrodinger picture:** States evolve, operators are fixed
2. **Ehrenfest's theorem** connects quantum expectation values to classical equations
3. **Conservation laws** arise from commutators with the Hamiltonian
4. **The classical limit** emerges for narrow wave packets
5. **Wave packets spread** for free particles due to momentum uncertainty
6. **Energy is conserved** for time-independent Hamiltonians

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.6 and Sakurai Chapter 2.2
- [ ] Derive Ehrenfest's theorem from the Schrodinger equation
- [ ] Verify $d\langle x\rangle/dt = \langle p\rangle/m$ for harmonic oscillator
- [ ] Understand the classical limit and its limitations
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and understand the computational lab
- [ ] Verify Ehrenfest numerically

---

## Preview: Day 362

Tomorrow we explore the **Heisenberg picture**, where operators evolve and states remain fixed. This picture makes the connection to classical mechanics more transparent and is often more elegant for solving problems.

---

*"The Schrodinger picture is like watching the ball move while the basket stays fixed. The Heisenberg picture is like watching the basket move while holding the ball still."*
— Anonymous quantum mechanics instructor

---

**Next:** [Day_362_Friday.md](Day_362_Friday.md) — Heisenberg Picture
