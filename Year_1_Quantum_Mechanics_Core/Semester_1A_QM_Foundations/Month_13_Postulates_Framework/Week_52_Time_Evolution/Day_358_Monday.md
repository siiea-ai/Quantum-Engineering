# Day 358: The Schrodinger Equation — The Fundamental Law of Quantum Dynamics

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: The Schrodinger Equation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 358, you will be able to:

1. State the time-dependent Schrodinger equation and explain each term
2. Derive the Schrodinger equation from classical-quantum correspondence
3. Prove that the Schrodinger equation conserves probability
4. Understand why the equation must be first-order in time
5. Connect the Schrodinger equation to energy conservation
6. Solve the Schrodinger equation for simple time-independent Hamiltonians

---

## Core Content

### 1. The Fifth Postulate: Time Evolution

We now address the final question of quantum mechanics: *How do quantum states change in time?*

**Postulate 5 (Time Evolution):**

> *The time evolution of a quantum state $|\psi(t)\rangle$ is governed by the Schrodinger equation:*
>
> $$\boxed{i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle}$$

where $\hat{H}$ is the Hamiltonian operator of the system.

This equation was first written down by Erwin Schrodinger in 1926 and remains the cornerstone of non-relativistic quantum mechanics.

---

### 2. Anatomy of the Schrodinger Equation

Let us examine each component:

**The Left-Hand Side: $i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle$**

- $i = \sqrt{-1}$: The imaginary unit ensures unitary (probability-conserving) evolution
- $\hbar = h/(2\pi) \approx 1.055 \times 10^{-34}$ J·s: Sets the quantum scale
- $\frac{\partial}{\partial t}$: Time derivative — first order, not second
- $|\psi(t)\rangle$: The state vector as a function of time

**The Right-Hand Side: $\hat{H}|\psi(t)\rangle$**

- $\hat{H}$: The Hamiltonian operator, representing total energy
- For a particle: $\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})$

---

### 3. Why First Order in Time?

The Schrodinger equation is first-order in time, unlike classical mechanics (Newton's second law is second-order). This has profound implications:

**Deterministic Evolution:**
Given $|\psi(t_0)\rangle$, the state at any later time is uniquely determined:
$$|\psi(t_0)\rangle \xrightarrow{\text{Schrodinger}} |\psi(t)\rangle$$

**No Initial Velocity Needed:**
In classical mechanics, you need position AND velocity. In QM, you only need the state.

**Time Reversal:**
Replace $t \to -t$ and $i \to -i$ (complex conjugate): the equation is symmetric under this combined transformation.

---

### 4. Derivation from Classical Mechanics

We can motivate the Schrodinger equation through the classical-quantum correspondence.

**Step 1: Classical Energy**
$$E = \frac{p^2}{2m} + V(x)$$

**Step 2: de Broglie Relations**
$$E = \hbar\omega, \quad p = \hbar k$$

**Step 3: Plane Wave**
A free particle with definite momentum $p$ is described by:
$$\psi(x,t) = e^{i(kx - \omega t)} = e^{i(px - Et)/\hbar}$$

**Step 4: Operator Identification**
Taking derivatives:
$$\frac{\partial \psi}{\partial t} = -\frac{iE}{\hbar}\psi \implies E\psi = i\hbar\frac{\partial \psi}{\partial t}$$
$$\frac{\partial \psi}{\partial x} = \frac{ip}{\hbar}\psi \implies p\psi = -i\hbar\frac{\partial \psi}{\partial x}$$

**Step 5: Operator Substitution**
Replace $E \to i\hbar\frac{\partial}{\partial t}$ and $p \to -i\hbar\frac{\partial}{\partial x}$ in the classical energy:

$$i\hbar\frac{\partial \psi}{\partial t} = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi$$

This is the Schrodinger equation in position representation.

---

### 5. The Schrodinger Equation in Position Representation

In the position basis, $\psi(x,t) = \langle x|\psi(t)\rangle$, the equation becomes:

$$\boxed{i\hbar \frac{\partial \psi(x,t)}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2 \psi(x,t)}{\partial x^2} + V(x)\psi(x,t)}$$

This is a partial differential equation (PDE) — specifically, a linear PDE with complex coefficients.

**Key Properties:**
- Linear: Superposition principle holds
- Homogeneous: If $\psi$ is a solution, so is $c\psi$
- Parabolic: First-order in $t$, second-order in $x$

---

### 6. Conservation of Probability

The Schrodinger equation guarantees that probability is conserved. Let us prove this.

**Total Probability:**
$$P = \langle\psi(t)|\psi(t)\rangle = \int_{-\infty}^{\infty} |\psi(x,t)|^2 dx$$

**Time Derivative:**
$$\frac{dP}{dt} = \frac{d}{dt}\langle\psi|\psi\rangle = \left\langle\frac{\partial\psi}{\partial t}\middle|\psi\right\rangle + \left\langle\psi\middle|\frac{\partial\psi}{\partial t}\right\rangle$$

From the Schrodinger equation:
$$\frac{\partial|\psi\rangle}{\partial t} = \frac{1}{i\hbar}\hat{H}|\psi\rangle$$

Taking the adjoint (using $\hat{H}^\dagger = \hat{H}$):
$$\frac{\partial\langle\psi|}{\partial t} = -\frac{1}{i\hbar}\langle\psi|\hat{H}$$

Substituting:
$$\frac{dP}{dt} = -\frac{1}{i\hbar}\langle\psi|\hat{H}|\psi\rangle + \frac{1}{i\hbar}\langle\psi|\hat{H}|\psi\rangle = 0$$

**Result:** $\boxed{\frac{dP}{dt} = 0}$ — Probability is conserved!

This is why the factor of $i$ is essential: it makes the evolution unitary.

---

### 7. The Continuity Equation

The local form of probability conservation is the continuity equation. Define:

**Probability Density:**
$$\rho(x,t) = |\psi(x,t)|^2 = \psi^*(x,t)\psi(x,t)$$

**Probability Current:**
$$j(x,t) = \frac{\hbar}{2mi}\left[\psi^*\frac{\partial\psi}{\partial x} - \psi\frac{\partial\psi^*}{\partial x}\right] = \frac{\hbar}{m}\text{Im}\left(\psi^*\frac{\partial\psi}{\partial x}\right)$$

**Continuity Equation:**
$$\boxed{\frac{\partial \rho}{\partial t} + \frac{\partial j}{\partial x} = 0}$$

This says: probability cannot be created or destroyed, only flow from place to place.

**Derivation:**
$$\frac{\partial \rho}{\partial t} = \frac{\partial \psi^*}{\partial t}\psi + \psi^*\frac{\partial \psi}{\partial t}$$

Using the Schrodinger equation and its conjugate:
$$\frac{\partial \psi}{\partial t} = \frac{i\hbar}{2m}\frac{\partial^2 \psi}{\partial x^2} - \frac{i}{\hbar}V\psi$$
$$\frac{\partial \psi^*}{\partial t} = -\frac{i\hbar}{2m}\frac{\partial^2 \psi^*}{\partial x^2} + \frac{i}{\hbar}V\psi^*$$

The potential terms cancel (V is real), leaving:
$$\frac{\partial \rho}{\partial t} = \frac{i\hbar}{2m}\left[\psi^*\frac{\partial^2 \psi}{\partial x^2} - \psi\frac{\partial^2 \psi^*}{\partial x^2}\right] = -\frac{\partial j}{\partial x}$$

---

### 8. Energy Conservation

For a time-independent Hamiltonian, energy is conserved in the sense that:

$$\frac{d}{dt}\langle\hat{H}\rangle = \frac{i}{\hbar}\langle[\hat{H}, \hat{H}]\rangle = 0$$

More specifically, if $|\psi(0)\rangle = \sum_n c_n |E_n\rangle$, then the probability of measuring energy $E_n$ is $|c_n|^2$, which is constant in time.

---

## Physical Interpretation

### The Role of the Hamiltonian

The Hamiltonian serves dual roles:
1. **Observable:** $\hat{H}$ is the energy operator; measuring it gives energy eigenvalues
2. **Generator:** $\hat{H}$ generates time translations; it tells states how to evolve

This duality is captured in Noether's theorem: time translation symmetry ↔ energy conservation.

### Quantum vs Classical Evolution

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| State | Point $(q, p)$ | Vector $\|\psi\rangle$ |
| Evolution | Hamilton's equations | Schrodinger equation |
| Order | 2nd order in $t$ | 1st order in $t$ |
| Determinism | Position & momentum | Amplitudes |

### The Imaginary Unit $i$

The factor of $i$ is not optional — it's required for:
1. **Unitarity:** Preserving $\langle\psi|\psi\rangle = 1$
2. **Oscillation:** Complex exponentials $e^{-iEt/\hbar}$ instead of growth/decay
3. **Interference:** Phase accumulation leads to interference

---

## Quantum Computing Connection

### Quantum Gates as Time Evolution

Every quantum gate is a unitary operator, which can be written as:
$$\hat{U} = e^{-i\hat{H}t/\hbar}$$

for some effective Hamiltonian $\hat{H}$ and time $t$.

**Example: The Pauli-X Gate**
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = e^{-i\pi\sigma_x/2}$$

This corresponds to evolution under $\hat{H} = \frac{\pi\hbar}{2t}\sigma_x$.

### Hamiltonian Simulation

A major application of quantum computers is simulating quantum systems, which requires implementing:
$$|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle$$

For complex many-body Hamiltonians, this is exponentially faster on quantum computers than classical ones.

### Continuous-Time Quantum Walks

The Schrodinger equation governs continuous-time quantum walks on graphs:
$$i\frac{d}{dt}|\psi\rangle = \hat{A}|\psi\rangle$$

where $\hat{A}$ is the adjacency matrix — used in quantum search algorithms.

---

## Worked Examples

### Example 1: Free Particle

**Problem:** Solve the Schrodinger equation for a free particle ($V = 0$) with initial condition $\psi(x,0) = e^{ikx}$.

**Solution:**

The Schrodinger equation is:
$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2 \psi}{\partial x^2}$$

For $\psi(x,t) = e^{i(kx - \omega t)}$:
$$\frac{\partial \psi}{\partial t} = -i\omega \psi$$
$$\frac{\partial^2 \psi}{\partial x^2} = -k^2 \psi$$

Substituting:
$$i\hbar(-i\omega)\psi = -\frac{\hbar^2}{2m}(-k^2)\psi$$
$$\hbar\omega = \frac{\hbar^2 k^2}{2m}$$

With $p = \hbar k$:
$$\omega = \frac{\hbar k^2}{2m} = \frac{p^2}{2m\hbar} = \frac{E}{\hbar}$$

**Solution:**
$$\boxed{\psi(x,t) = e^{i(kx - \hbar k^2 t/2m)}}$$

This represents a plane wave traveling with phase velocity $v_p = \omega/k = \hbar k/2m = p/2m$. ∎

---

### Example 2: Verifying Probability Conservation

**Problem:** For $\psi(x,t) = \frac{1}{(2\pi\sigma^2)^{1/4}}e^{-(x-vt)^2/4\sigma^2}e^{imvx/\hbar}e^{-imv^2t/2\hbar}$ (a Gaussian wave packet), verify that $\int|\psi|^2 dx = 1$ is constant.

**Solution:**

Compute $|\psi|^2$:
$$|\psi|^2 = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-vt)^2/2\sigma^2}$$

This is a Gaussian centered at $x = vt$ with width $\sigma$.

$$\int_{-\infty}^{\infty}|\psi|^2 dx = \frac{1}{\sqrt{2\pi\sigma^2}}\int_{-\infty}^{\infty}e^{-(x-vt)^2/2\sigma^2}dx$$

Let $u = (x - vt)/\sqrt{2}\sigma$:
$$= \frac{1}{\sqrt{2\pi\sigma^2}} \cdot \sqrt{2}\sigma \cdot \sqrt{\pi} = 1$$

The result is independent of $t$: $\boxed{\int|\psi|^2 dx = 1}$ for all $t$. ∎

---

### Example 3: Two-Level System (Qubit)

**Problem:** A qubit has Hamiltonian $\hat{H} = \frac{\hbar\omega}{2}\sigma_z = \frac{\hbar\omega}{2}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

If $|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, find $|\psi(t)\rangle$.

**Solution:**

The energy eigenstates are $|0\rangle$ and $|1\rangle$ with energies $E_0 = \hbar\omega/2$ and $E_1 = -\hbar\omega/2$.

The Schrodinger equation gives:
$$|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle$$

Since $|0\rangle$ and $|1\rangle$ are eigenstates:
$$e^{-i\hat{H}t/\hbar}|0\rangle = e^{-iE_0 t/\hbar}|0\rangle = e^{-i\omega t/2}|0\rangle$$
$$e^{-i\hat{H}t/\hbar}|1\rangle = e^{-iE_1 t/\hbar}|1\rangle = e^{i\omega t/2}|1\rangle$$

Therefore:
$$|\psi(t)\rangle = \frac{1}{\sqrt{2}}\left(e^{-i\omega t/2}|0\rangle + e^{i\omega t/2}|1\rangle\right)$$

$$\boxed{|\psi(t)\rangle = \frac{e^{-i\omega t/2}}{\sqrt{2}}\left(|0\rangle + e^{i\omega t}|1\rangle\right)}$$

The relative phase oscillates at frequency $\omega$. ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Normalization Conservation:** Show that if $\langle\psi(0)|\psi(0)\rangle = 1$, then $\langle\psi(t)|\psi(t)\rangle = 1$ for all $t$.

2. **Probability Current:** For $\psi = Ae^{i(kx-\omega t)}$, compute the probability current $j(x,t)$ and interpret the result.

3. **Energy Eigenstates:** If $\hat{H}|E\rangle = E|E\rangle$, show that $|\psi(t)\rangle = e^{-iEt/\hbar}|E\rangle$ satisfies the Schrodinger equation.

### Level 2: Intermediate

4. **Superposition Dynamics:** A particle in an infinite well has initial state $|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|1\rangle + |2\rangle)$ where $|n\rangle$ has energy $E_n = n^2 E_1$. Find $|\psi(t)\rangle$ and compute $|\langle x|\psi(t)\rangle|^2$ — what frequency does the probability oscillate at?

5. **Continuity Equation:** Derive the continuity equation starting from the Schrodinger equation. Show all steps explicitly.

6. **Complex Potential:** Consider $\hat{H} = \hat{H}_0 - i\Gamma$ where $\Gamma > 0$ is real. Show that $\frac{d}{dt}\langle\psi|\psi\rangle = -\frac{2\Gamma}{\hbar}\langle\psi|\psi\rangle$. Interpret this physically (particle decay).

### Level 3: Challenging

7. **Time Reversal:** Define the time-reversal operator $\hat{T}$ by $\hat{T}\psi(x,t) = \psi^*(x,-t)$. Show that if $\psi(x,t)$ satisfies the Schrodinger equation with real $V(x)$, then so does $\hat{T}\psi$.

8. **Ehrenfest's Theorem Preview:** Starting from the Schrodinger equation, prove that:
   $$\frac{d}{dt}\langle\hat{x}\rangle = \frac{\langle\hat{p}\rangle}{m}$$
   (Hint: Use $[\hat{H}, \hat{x}] = -\frac{i\hbar}{m}\hat{p}$)

9. **Spreading Wave Packet:** A free-particle Gaussian wave packet has width $\sigma(t) = \sigma_0\sqrt{1 + (\hbar t/2m\sigma_0^2)^2}$. Show that this follows from the Schrodinger equation and find the spreading timescale for an electron with $\sigma_0 = 1$ nm.

---

## Computational Lab

### Objective
Numerically solve the time-dependent Schrodinger equation and visualize probability evolution.

```python
"""
Day 358 Computational Lab: The Schrodinger Equation
Quantum Mechanics Core - Year 1

This lab numerically solves the time-dependent Schrodinger equation
using the Crank-Nicolson method and visualizes quantum dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_banded

# =============================================================================
# Part 1: Setup the Problem
# =============================================================================

print("=" * 60)
print("Part 1: Time-Dependent Schrodinger Equation Solver")
print("=" * 60)

# Physical parameters (atomic units: hbar = m = 1)
hbar = 1.0
m = 1.0

# Spatial grid
N_x = 500
x_min, x_max = -20, 20
x = np.linspace(x_min, x_max, N_x)
dx = x[1] - x[0]

# Time parameters
dt = 0.01
N_t = 1000
t_final = N_t * dt

print(f"Spatial grid: {N_x} points from {x_min} to {x_max}")
print(f"dx = {dx:.4f}")
print(f"Time steps: {N_t}, dt = {dt}, t_final = {t_final}")

# =============================================================================
# Part 2: Initial Wave Function (Gaussian Wave Packet)
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Initial Wave Packet")
print("=" * 60)

def gaussian_wave_packet(x, x0, sigma, k0):
    """
    Create a Gaussian wave packet.

    Parameters:
    -----------
    x : array
        Position grid
    x0 : float
        Center of the packet
    sigma : float
        Width of the packet
    k0 : float
        Central wave vector (momentum = hbar * k0)

    Returns:
    --------
    psi : complex array
        Normalized wave function
    """
    psi = np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)
    # Normalize
    norm = np.sqrt(np.trapz(np.abs(psi)**2, x))
    return psi / norm

# Initial conditions
x0 = -5.0      # Start position
sigma = 1.0    # Initial width
k0 = 3.0       # Initial momentum (units of hbar)

psi_0 = gaussian_wave_packet(x, x0, sigma, k0)

print(f"Initial position: x0 = {x0}")
print(f"Initial width: sigma = {sigma}")
print(f"Initial momentum: p0 = hbar * k0 = {hbar * k0}")
print(f"Expected velocity: v = p0/m = {hbar * k0 / m}")

# Verify normalization
norm_0 = np.trapz(np.abs(psi_0)**2, x)
print(f"Initial normalization: {norm_0:.10f}")

# =============================================================================
# Part 3: Potential Energy
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Potential Energy")
print("=" * 60)

def potential_free(x):
    """Free particle: V = 0"""
    return np.zeros_like(x)

def potential_barrier(x, V0=5.0, a=0.5):
    """Rectangular barrier centered at x=0"""
    return np.where(np.abs(x) < a, V0, 0)

def potential_harmonic(x, omega=0.5):
    """Harmonic oscillator: V = 0.5 * m * omega^2 * x^2"""
    return 0.5 * m * omega**2 * x**2

# Choose potential (try different ones!)
V = potential_barrier(x, V0=8.0, a=0.5)
# V = potential_free(x)
# V = potential_harmonic(x, omega=0.3)

print(f"Using barrier potential: V0 = 8.0, width = 1.0")

# =============================================================================
# Part 4: Crank-Nicolson Method
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Crank-Nicolson Time Evolution")
print("=" * 60)

def crank_nicolson_step(psi, V, dx, dt, hbar, m):
    """
    Perform one time step using Crank-Nicolson method.

    The TDSE: i*hbar * dpsi/dt = H*psi
    where H = -hbar^2/(2m) * d^2/dx^2 + V

    Crank-Nicolson: (1 + i*dt*H/(2*hbar)) * psi_new = (1 - i*dt*H/(2*hbar)) * psi_old
    """
    N = len(psi)

    # Coefficients for the tridiagonal system
    alpha = 1j * hbar * dt / (4 * m * dx**2)
    beta = 1j * dt / (2 * hbar) * V

    # Build the tridiagonal matrices
    # Main diagonal: 1 + 2*alpha + beta
    # Off-diagonals: -alpha

    # Right-hand side: (1 - i*dt*H/(2*hbar)) * psi
    # Main diagonal: 1 - 2*alpha - beta
    # Off-diagonals: +alpha

    # Compute RHS
    rhs = np.zeros(N, dtype=complex)
    rhs[1:-1] = ((1 - 2*alpha - beta[1:-1]) * psi[1:-1]
                 + alpha * (psi[:-2] + psi[2:]))

    # Boundary conditions (psi = 0 at boundaries)
    rhs[0] = 0
    rhs[-1] = 0

    # Build banded matrix for scipy.linalg.solve_banded
    # Format: ab[u + i - j, j] = a[i,j]  where u = number of upper diagonals
    ab = np.zeros((3, N), dtype=complex)
    ab[0, 1:] = -alpha  # Upper diagonal
    ab[1, :] = 1 + 2*alpha + beta  # Main diagonal
    ab[2, :-1] = -alpha  # Lower diagonal

    # Boundary conditions
    ab[1, 0] = 1
    ab[1, -1] = 1
    ab[0, 1] = 0
    ab[2, -2] = 0

    # Solve the tridiagonal system
    psi_new = solve_banded((1, 1), ab, rhs)

    return psi_new

# Time evolution
psi = psi_0.copy()
psi_history = [psi.copy()]
norms = [np.trapz(np.abs(psi)**2, x)]
times = [0]

for n in range(N_t):
    psi = crank_nicolson_step(psi, V, dx, dt, hbar, m)

    if (n + 1) % 10 == 0:  # Save every 10 steps
        psi_history.append(psi.copy())
        norms.append(np.trapz(np.abs(psi)**2, x))
        times.append((n + 1) * dt)

psi_history = np.array(psi_history)
norms = np.array(norms)
times = np.array(times)

print(f"Evolution complete!")
print(f"Initial norm: {norms[0]:.10f}")
print(f"Final norm: {norms[-1]:.10f}")
print(f"Norm change: {abs(norms[-1] - norms[0]):.2e}")

# =============================================================================
# Part 5: Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Visualization")
print("=" * 60)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Initial and final states
ax1 = axes[0, 0]
ax1.plot(x, np.abs(psi_0)**2, 'b-', label='Initial', linewidth=2)
ax1.plot(x, np.abs(psi_history[-1])**2, 'r-', label='Final', linewidth=2)
ax1.fill_between(x, 0, V/V.max() * np.max(np.abs(psi_0)**2) * 0.5,
                  alpha=0.3, color='gray', label='Potential (scaled)')
ax1.set_xlabel('Position x')
ax1.set_ylabel('Probability density |psi|^2')
ax1.set_title('Initial and Final Probability Densities')
ax1.legend()
ax1.set_xlim(x_min, x_max)
ax1.grid(True, alpha=0.3)

# Plot 2: Norm conservation
ax2 = axes[0, 1]
ax2.plot(times, norms, 'g-', linewidth=2)
ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time t')
ax2.set_ylabel('Total probability')
ax2.set_title('Probability Conservation')
ax2.set_ylim(0.99, 1.01)
ax2.grid(True, alpha=0.3)

# Plot 3: Space-time density plot
ax3 = axes[1, 0]
prob_density = np.abs(psi_history)**2
im = ax3.imshow(prob_density.T, aspect='auto', origin='lower',
                extent=[0, times[-1], x_min, x_max], cmap='hot')
plt.colorbar(im, ax=ax3, label='|psi|^2')
ax3.set_xlabel('Time t')
ax3.set_ylabel('Position x')
ax3.set_title('Space-Time Evolution')

# Plot 4: Expectation values
ax4 = axes[1, 1]
x_expect = [np.trapz(x * np.abs(psi_history[i])**2, x) for i in range(len(times))]
ax4.plot(times, x_expect, 'b-', linewidth=2, label='<x>(t)')

# Classical trajectory for comparison
v_classical = hbar * k0 / m
x_classical = x0 + v_classical * times
ax4.plot(times, x_classical, 'r--', linewidth=2, label='Classical')

ax4.set_xlabel('Time t')
ax4.set_ylabel('Position')
ax4.set_title('Expectation Value vs Classical Trajectory')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_358_schrodinger_equation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_358_schrodinger_equation.png'")

# =============================================================================
# Part 6: Probability Current Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Probability Current")
print("=" * 60)

def probability_current(psi, x, hbar, m):
    """
    Compute the probability current j = (hbar/m) * Im(psi* dpsi/dx)
    """
    dpsi_dx = np.gradient(psi, x)
    return (hbar / m) * np.imag(np.conj(psi) * dpsi_dx)

# Compute current at several times
fig, ax = plt.subplots(figsize=(10, 6))

time_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

for idx, color in zip(time_indices, colors):
    j = probability_current(psi_history[idx], x, hbar, m)
    ax.plot(x, j, color=color, label=f't = {times[idx]:.2f}')

ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel('Position x')
ax.set_ylabel('Probability current j(x,t)')
ax.set_title('Probability Current at Different Times')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(x_min, x_max)

plt.tight_layout()
plt.savefig('day_358_probability_current.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_358_probability_current.png'")

# =============================================================================
# Part 7: Transmission and Reflection
# =============================================================================

print("\n" + "=" * 60)
print("Part 7: Transmission and Reflection Coefficients")
print("=" * 60)

# Define regions for transmission/reflection
x_left = x < -5   # Reflection region (far left)
x_right = x > 5   # Transmission region (far right)

# Compute probabilities in each region at final time
prob_left = np.trapz(np.abs(psi_history[-1])**2 * x_left, x)
prob_right = np.trapz(np.abs(psi_history[-1])**2 * x_right, x)
prob_barrier = 1 - prob_left - prob_right

print(f"Reflection coefficient (approx): R = {prob_left:.4f}")
print(f"Transmission coefficient (approx): T = {prob_right:.4f}")
print(f"In barrier region: {prob_barrier:.4f}")
print(f"R + T + barrier = {prob_left + prob_right + prob_barrier:.4f}")

# Theoretical transmission (for comparison)
E = (hbar * k0)**2 / (2 * m)  # Kinetic energy
V0 = 8.0  # Barrier height
a = 0.5   # Barrier half-width
if E < V0:
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar
    T_theory = 1 / (1 + (V0**2 * np.sinh(2*kappa*a)**2) / (4*E*(V0-E)))
    print(f"\nTheoretical T (rectangular barrier): {T_theory:.4f}")
    print(f"Note: Wave packet is not monoenergetic, so exact match not expected")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Schrodinger equation (abstract) | $i\hbar \frac{\partial}{\partial t}\|\psi\rangle = \hat{H}\|\psi\rangle$ |
| Schrodinger equation (position) | $i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2 \psi}{\partial x^2} + V\psi$ |
| Probability density | $\rho = \|\psi\|^2$ |
| Probability current | $j = \frac{\hbar}{m}\text{Im}(\psi^* \nabla\psi)$ |
| Continuity equation | $\frac{\partial \rho}{\partial t} + \nabla \cdot j = 0$ |
| Conservation of probability | $\frac{d}{dt}\langle\psi\|\psi\rangle = 0$ |

### Main Takeaways

1. **The Schrodinger equation** is the fundamental dynamical law of quantum mechanics
2. **First-order in time** ensures unique, deterministic evolution from initial conditions
3. **The factor of $i$** is essential for probability conservation (unitary evolution)
4. **The Hamiltonian** plays dual roles: energy observable and evolution generator
5. **Probability flows** according to the continuity equation — never created or destroyed
6. **Classical correspondence** motivates the form of the equation

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.1-4.2 and Sakurai Chapter 2.1
- [ ] Derive the Schrodinger equation from de Broglie relations
- [ ] Prove probability conservation mathematically
- [ ] Derive the continuity equation
- [ ] Solve the free particle Schrodinger equation
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and understand the computational lab
- [ ] Verify norm conservation in numerical simulation

---

## Preview: Day 359

Tomorrow we study the **time evolution operator** $\hat{U}(t) = e^{-i\hat{H}t/\hbar}$, the formal solution to the Schrodinger equation. We will prove its unitarity, examine the composition law, and understand how it connects to quantum gates.

---

*"Where did we get that from? Nowhere. It is not possible to derive it from anything you know. It came out of the mind of Schrodinger."*
— Richard Feynman, on the Schrodinger equation

---

**Next:** [Day_359_Tuesday.md](Day_359_Tuesday.md) — Time Evolution Operator
