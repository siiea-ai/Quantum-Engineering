# Day 360: Stationary States — The Timeless Eigenstates of Energy

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Stationary States |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 360, you will be able to:

1. Define stationary states and explain their physical significance
2. Solve the time-independent Schrodinger equation (TISE)
3. Show that stationary states have time-independent probability densities
4. Expand arbitrary states in the energy eigenbasis
5. Calculate oscillation frequencies between energy levels
6. Distinguish stationary states from general time-dependent states

---

## Core Content

### 1. What Are Stationary States?

**Definition:** A **stationary state** is an energy eigenstate — a state that satisfies:

$$\boxed{\hat{H}|E\rangle = E|E\rangle}$$

where $E$ is the energy eigenvalue and $|E\rangle$ is the corresponding eigenstate.

The name "stationary" comes from the fact that the **probability density** does not change in time, even though the state itself evolves.

---

### 2. Time Evolution of Stationary States

For a stationary state $|E\rangle$, the time evolution is remarkably simple:

$$|E, t\rangle = e^{-i\hat{H}t/\hbar}|E\rangle = e^{-iEt/\hbar}|E\rangle$$

The state acquires only a **global phase factor** $e^{-iEt/\hbar}$.

**Why does the eigenvalue come out?**

Using the Taylor series of the exponential:
$$e^{-i\hat{H}t/\hbar}|E\rangle = \sum_{n=0}^{\infty}\frac{1}{n!}\left(\frac{-i\hat{H}t}{\hbar}\right)^n|E\rangle$$

Since $\hat{H}|E\rangle = E|E\rangle$, we have $\hat{H}^n|E\rangle = E^n|E\rangle$:
$$= \sum_{n=0}^{\infty}\frac{1}{n!}\left(\frac{-iEt}{\hbar}\right)^n|E\rangle = e^{-iEt/\hbar}|E\rangle$$

---

### 3. The Time-Independent Schrodinger Equation

To find stationary states, we must solve the **time-independent Schrodinger equation** (TISE):

$$\boxed{\hat{H}|\psi\rangle = E|\psi\rangle}$$

In position representation with $\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})$:

$$\boxed{-\frac{\hbar^2}{2m}\frac{d^2\psi(x)}{dx^2} + V(x)\psi(x) = E\psi(x)}$$

This is an **eigenvalue equation** — a second-order ODE whose solutions depend on the potential $V(x)$.

**Key Properties:**
- $E$ (energy) is the eigenvalue
- $\psi(x)$ (wave function) is the eigenfunction
- Boundary conditions determine which values of $E$ are allowed (quantization)

---

### 4. Why "Stationary"?

Consider a stationary state in position representation:
$$\psi(x, t) = e^{-iEt/\hbar}\psi(x)$$

The probability density is:
$$|\psi(x, t)|^2 = |e^{-iEt/\hbar}|^2 |\psi(x)|^2 = |\psi(x)|^2$$

**The probability density is independent of time!**

Similarly, all expectation values in a stationary state are time-independent:
$$\langle\hat{A}\rangle = \langle E|\hat{A}|E\rangle$$

The state is "stationary" in the sense that all measurable properties remain constant.

**However:** The state vector itself still has time dependence through the phase $e^{-iEt/\hbar}$. This phase becomes physically important when considering superpositions.

---

### 5. General Solutions: Superpositions of Stationary States

Any initial state can be expanded in the energy eigenbasis:
$$|\psi(0)\rangle = \sum_n c_n|E_n\rangle$$

where $c_n = \langle E_n|\psi(0)\rangle$.

The time evolution is:
$$|\psi(t)\rangle = \sum_n c_n e^{-iE_n t/\hbar}|E_n\rangle$$

**Key insight:** Each energy component evolves with its own phase. This creates interference effects that cause the probability density to oscillate.

**In position representation:**
$$\psi(x, t) = \sum_n c_n \psi_n(x) e^{-iE_n t/\hbar}$$

$$|\psi(x, t)|^2 = \sum_{m,n} c_m^* c_n \psi_m^*(x)\psi_n(x) e^{i(E_m - E_n)t/\hbar}$$

The oscillation frequencies are determined by **energy differences**:
$$\omega_{mn} = \frac{E_m - E_n}{\hbar}$$

---

### 6. Energy-Time Relation and Quantum Oscillations

The fundamental formula connecting energy differences to oscillation frequencies:

$$\boxed{\omega = \frac{\Delta E}{\hbar} \quad \Leftrightarrow \quad \nu = \frac{\Delta E}{h}}$$

**Physical Examples:**

| System | Energy Difference | Oscillation Frequency |
|--------|-------------------|----------------------|
| Atomic transition | $\Delta E \sim$ eV | $\nu \sim 10^{15}$ Hz (optical) |
| Nuclear transition | $\Delta E \sim$ MeV | $\nu \sim 10^{20}$ Hz (gamma) |
| Qubit | $\Delta E \sim$ GHz·$h$ | $\nu \sim 10^9$ Hz (microwave) |

---

### 7. The Bohr Frequency Condition

When we have a superposition of two energy states:
$$|\psi(t)\rangle = c_1 e^{-iE_1 t/\hbar}|E_1\rangle + c_2 e^{-iE_2 t/\hbar}|E_2\rangle$$

The probability density oscillates at the **Bohr frequency**:
$$\omega_{\text{Bohr}} = \frac{E_2 - E_1}{\hbar}$$

This is the frequency of light emitted/absorbed in transitions between these levels:
$$\boxed{h\nu = E_2 - E_1}$$

This is the **Bohr frequency condition** — the bridge between quantum mechanics and spectroscopy.

---

### 8. Separation of Variables

The time-independent Schrodinger equation arises from **separation of variables**.

Assume $\Psi(x, t) = \psi(x)T(t)$. Substitute into the TDSE:
$$i\hbar\psi(x)\frac{dT}{dt} = T(t)\hat{H}\psi(x)$$

Divide by $\psi(x)T(t)$:
$$i\hbar\frac{1}{T}\frac{dT}{dt} = \frac{1}{\psi}\hat{H}\psi$$

The left side depends only on $t$; the right side depends only on $x$. Both must equal a constant, which we call $E$:

**Time equation:**
$$i\hbar\frac{dT}{dt} = ET \implies T(t) = e^{-iEt/\hbar}$$

**Space equation:**
$$\hat{H}\psi = E\psi \quad \text{(TISE)}$$

The general solution is a superposition of separable solutions.

---

## Physical Interpretation

### Stationary States vs. Moving States

| Aspect | Stationary State | Superposition |
|--------|------------------|---------------|
| Energy | Definite value $E$ | Uncertain: $\Delta E > 0$ |
| Probability density | Time-independent | Oscillates |
| Expectation values | Constant | Time-varying |
| Example | Ground state atom | Wave packet |

### The Quantum-Classical Correspondence

Classically, particles move in orbits. Quantum mechanically, stationary states don't "move" — their probability density is static.

**Resolution:** Classical motion emerges from superpositions of many energy eigenstates, where the phase factors combine to create localized, moving wave packets.

### Stability of Atoms

A hydrogen atom in its ground state $|1s\rangle$ is stationary. The electron doesn't "orbit" — the probability cloud is static. This explains why atoms don't radiate and collapse (a classical puzzle).

---

## Quantum Computing Connection

### Energy Eigenstates as Computational Basis

In quantum computing, the computational basis states $|0\rangle$ and $|1\rangle$ are often chosen to be energy eigenstates:
- For superconducting qubits: ground and first excited states of a nonlinear oscillator
- For trapped ions: hyperfine ground states
- For spin qubits: spin-up and spin-down in a magnetic field

### Decoherence and Energy Relaxation

When a qubit in superposition $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$ interacts with its environment:
1. **Dephasing:** The relative phase between $|0\rangle$ and $|1\rangle$ becomes random
2. **Relaxation:** The excited state $|1\rangle$ decays to $|0\rangle$ (energy is lost)

Stationary states (especially the ground state) are protected from relaxation.

### Adiabatic Quantum Computing

Adiabatic quantum algorithms work by:
1. Start in the ground state of a simple Hamiltonian
2. Slowly change the Hamiltonian
3. System remains in the ground state (adiabatic theorem)
4. Final ground state encodes the answer

The algorithm exploits stationary states throughout.

---

## Worked Examples

### Example 1: Two-Level System Superposition

**Problem:** A qubit has energy eigenstates $|0\rangle$ (energy 0) and $|1\rangle$ (energy $\hbar\omega$). If the initial state is $|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, find $|\psi(t)\rangle$ and the probability $P(0, t)$ of measuring energy 0.

**Solution:**

Each energy eigenstate evolves with its own phase:
$$|0, t\rangle = e^{-i(0)t/\hbar}|0\rangle = |0\rangle$$
$$|1, t\rangle = e^{-i\hbar\omega t/\hbar}|1\rangle = e^{-i\omega t}|1\rangle$$

The evolved state:
$$|\psi(t)\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + e^{-i\omega t}|1\rangle\right)$$

The probability of measuring energy 0:
$$P(0, t) = |\langle 0|\psi(t)\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$

**Important:** The measurement probabilities for energy are **constant** even though the state is not stationary!

This is because $\{|0\rangle, |1\rangle\}$ are energy eigenstates. Measurement probabilities only oscillate for observables that don't commute with $\hat{H}$. ∎

---

### Example 2: Infinite Square Well Superposition

**Problem:** A particle in an infinite square well (width $L$) is in the state $|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|1\rangle + |2\rangle)$ where $|n\rangle$ has energy $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$. Find the probability density $|\psi(x,t)|^2$ and identify the oscillation frequency.

**Solution:**

The wave functions are:
$$\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)$$

The time-evolved state in position representation:
$$\psi(x, t) = \frac{1}{\sqrt{2}}\left[\psi_1(x)e^{-iE_1 t/\hbar} + \psi_2(x)e^{-iE_2 t/\hbar}\right]$$

The probability density:
$$|\psi(x, t)|^2 = \frac{1}{2}\left[|\psi_1(x)|^2 + |\psi_2(x)|^2 + 2\text{Re}\left(\psi_1^*(x)\psi_2(x)e^{-i(E_2-E_1)t/\hbar}\right)\right]$$

$$= \frac{1}{2}\left[|\psi_1(x)|^2 + |\psi_2(x)|^2 + 2\psi_1(x)\psi_2(x)\cos\left(\frac{(E_2-E_1)t}{\hbar}\right)\right]$$

The oscillation frequency:
$$\omega_{21} = \frac{E_2 - E_1}{\hbar} = \frac{(4-1)\pi^2\hbar}{2mL^2} = \frac{3\pi^2\hbar}{2mL^2}$$

$$\boxed{\omega_{21} = \frac{3\pi^2\hbar}{2mL^2}}$$

The probability density oscillates — the particle "sloshes" back and forth in the well. ∎

---

### Example 3: Time-Dependent Expectation Value

**Problem:** For the state in Example 2, calculate $\langle x \rangle(t)$.

**Solution:**

$$\langle x \rangle = \langle\psi(t)|\hat{x}|\psi(t)\rangle$$

Expanding:
$$\langle x \rangle = \frac{1}{2}\left[\langle 1|\hat{x}|1\rangle + \langle 2|\hat{x}|2\rangle + \langle 1|\hat{x}|2\rangle e^{i(E_1-E_2)t/\hbar} + \langle 2|\hat{x}|1\rangle e^{i(E_2-E_1)t/\hbar}\right]$$

For the infinite well:
$$\langle n|\hat{x}|n\rangle = \int_0^L x|\psi_n(x)|^2 dx = \frac{L}{2} \quad \text{(by symmetry)}$$

$$\langle 1|\hat{x}|2\rangle = \int_0^L x\psi_1(x)\psi_2(x)dx = -\frac{16L}{9\pi^2}$$

Since $\langle 2|\hat{x}|1\rangle = \langle 1|\hat{x}|2\rangle^* = \langle 1|\hat{x}|2\rangle$ (x is real):

$$\langle x \rangle = \frac{L}{2} - \frac{16L}{9\pi^2}\cos(\omega_{21}t)$$

$$\boxed{\langle x \rangle(t) = \frac{L}{2} - \frac{16L}{9\pi^2}\cos\left(\frac{3\pi^2\hbar t}{2mL^2}\right)}$$

The expectation value oscillates about the center of the well! ∎

---

## Practice Problems

### Level 1: Direct Application

1. **Pure Stationary State:** If $|\psi(t)\rangle = e^{-iEt/\hbar}|E\rangle$, show that $\langle\hat{A}\rangle$ is time-independent for any observable $\hat{A}$.

2. **Three-Level System:** A system has energies $E_1 = 0$, $E_2 = \hbar\omega$, $E_3 = 3\hbar\omega$. What are the three Bohr frequencies?

3. **Ground State:** A hydrogen atom is in its ground state. Show that it's stationary and find the phase accumulated after time $T$.

### Level 2: Intermediate

4. **Coherent Superposition:** A particle in a harmonic oscillator has initial state $|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ where $E_n = (n + \frac{1}{2})\hbar\omega$. Find $|\psi(t)\rangle$, the revival time, and $\langle x \rangle(t)$.

5. **Infinite Well Mixture:** For initial state $|\psi\rangle = c_1|1\rangle + c_2|2\rangle + c_3|3\rangle$, how many distinct oscillation frequencies appear in $|\psi(x,t)|^2$?

6. **Energy Uncertainty:** For the state $|\psi\rangle = \frac{1}{\sqrt{2}}(|E_1\rangle + |E_2\rangle)$, compute $\langle\hat{H}\rangle$ and $\Delta H$.

### Level 3: Challenging

7. **Derivation:** Starting from $|\psi(t)\rangle = \sum_n c_n e^{-iE_n t/\hbar}|E_n\rangle$, prove that:
$$\frac{d}{dt}\langle\hat{H}\rangle = 0$$

8. **Quasi-Stationary States:** In nuclear physics, "quasi-bound" states have complex energies $E - i\Gamma/2$. Show that $|\psi(t)|^2 \propto e^{-\Gamma t/\hbar}$ (exponential decay).

9. **Quantum Beat Spectroscopy:** An atom is excited to a superposition of two closely-spaced levels ($\Delta E \ll E$). Derive the beat frequency of the emitted radiation and explain how this is used in precision spectroscopy.

---

## Computational Lab

### Objective
Visualize stationary states and time evolution of superpositions.

```python
"""
Day 360 Computational Lab: Stationary States
Quantum Mechanics Core - Year 1

This lab visualizes stationary states and superposition dynamics
in the infinite square well.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# =============================================================================
# Part 1: Infinite Square Well Eigenstates
# =============================================================================

print("=" * 60)
print("Part 1: Infinite Square Well Stationary States")
print("=" * 60)

# Parameters (natural units: hbar = m = L = 1)
L = 1.0
hbar = 1.0
m = 1.0

# Spatial grid
N_x = 500
x = np.linspace(0, L, N_x)

def energy(n, m=1.0, L=1.0, hbar=1.0):
    """Energy eigenvalue for infinite square well."""
    return (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)

def psi_n(x, n, L=1.0):
    """Energy eigenfunction for infinite square well."""
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

# Plot first few eigenstates
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, n in enumerate([1, 2, 3, 4]):
    ax = axes[idx // 2, idx % 2]
    psi = psi_n(x, n, L)
    prob = np.abs(psi)**2

    ax.fill_between(x, 0, prob, alpha=0.3, color=f'C{n-1}')
    ax.plot(x, prob, f'C{n-1}', linewidth=2, label=f'|psi_{n}|^2')
    ax.plot(x, psi, f'C{n-1}', linewidth=1, linestyle='--', alpha=0.7, label=f'psi_{n}')

    # Add energy level
    E_n = energy(n, m, L, hbar)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xlabel('Position x/L')
    ax.set_ylabel('Wave function / Probability')
    ax.set_title(f'n = {n}, E_{n} = {E_n:.4f}')
    ax.legend()
    ax.set_xlim(0, L)
    ax.grid(True, alpha=0.3)

plt.suptitle('Infinite Square Well: Stationary States', fontsize=14)
plt.tight_layout()
plt.savefig('day_360_stationary_states.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_360_stationary_states.png'")

# =============================================================================
# Part 2: Time Evolution of Superposition
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Time Evolution of Superposition")
print("=" * 60)

def time_evolved_superposition(x, t, coeffs, n_values, L=1.0, hbar=1.0, m=1.0):
    """
    Compute time-evolved superposition state.

    Parameters:
    -----------
    x : array
        Position grid
    t : float
        Time
    coeffs : list
        Complex coefficients c_n
    n_values : list
        Quantum numbers n

    Returns:
    --------
    psi : complex array
        Wave function at time t
    """
    psi = np.zeros_like(x, dtype=complex)
    for c_n, n in zip(coeffs, n_values):
        E_n = energy(n, m, L, hbar)
        phase = np.exp(-1j * E_n * t / hbar)
        psi += c_n * phase * psi_n(x, n, L)
    return psi

# Superposition of n=1 and n=2
n_values = [1, 2]
coeffs = [1/np.sqrt(2), 1/np.sqrt(2)]

# Calculate oscillation period
E1, E2 = energy(1, m, L, hbar), energy(2, m, L, hbar)
omega_21 = (E2 - E1) / hbar
T_period = 2 * np.pi / omega_21

print(f"E_1 = {E1:.4f}")
print(f"E_2 = {E2:.4f}")
print(f"Bohr frequency omega_21 = {omega_21:.4f}")
print(f"Oscillation period T = {T_period:.4f}")

# Time snapshots
times = [0, T_period/4, T_period/2, 3*T_period/4]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, t in enumerate(times):
    ax = axes[idx // 2, idx % 2]

    psi_t = time_evolved_superposition(x, t, coeffs, n_values, L, hbar, m)
    prob = np.abs(psi_t)**2

    ax.fill_between(x, 0, prob, alpha=0.5, color='purple')
    ax.plot(x, prob, 'purple', linewidth=2)

    # Mark expectation value
    x_mean = np.trapz(x * prob, x)
    ax.axvline(x=x_mean, color='red', linestyle='--', linewidth=2, label=f'<x> = {x_mean:.3f}')

    ax.set_xlabel('Position x/L')
    ax.set_ylabel('Probability |psi|^2')
    ax.set_title(f't = {t/T_period:.2f} T')
    ax.legend()
    ax.set_xlim(0, L)
    ax.set_ylim(0, 4)
    ax.grid(True, alpha=0.3)

plt.suptitle('Superposition (|1> + |2>)/sqrt(2): Probability Density Evolution', fontsize=14)
plt.tight_layout()
plt.savefig('day_360_superposition_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_360_superposition_evolution.png'")

# =============================================================================
# Part 3: Expectation Value Dynamics
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Expectation Value Dynamics")
print("=" * 60)

# Track <x> and <x^2> over time
t_values = np.linspace(0, 3*T_period, 200)
x_expect = []
x2_expect = []

for t in t_values:
    psi_t = time_evolved_superposition(x, t, coeffs, n_values, L, hbar, m)
    prob = np.abs(psi_t)**2

    x_expect.append(np.trapz(x * prob, x))
    x2_expect.append(np.trapz(x**2 * prob, x))

x_expect = np.array(x_expect)
x2_expect = np.array(x2_expect)
sigma_x = np.sqrt(x2_expect - x_expect**2)

# Analytical result
x12 = -16 * L / (9 * np.pi**2)  # <1|x|2>
x_analytical = L/2 + 2 * np.real(coeffs[0] * np.conj(coeffs[1]) * x12 * np.exp(-1j * omega_21 * t_values))

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot <x>
ax1 = axes[0]
ax1.plot(t_values/T_period, x_expect, 'b-', linewidth=2, label='Numerical <x>')
ax1.plot(t_values/T_period, x_analytical, 'r--', linewidth=2, label='Analytical')
ax1.axhline(y=L/2, color='k', linestyle=':', alpha=0.5, label='Well center')
ax1.set_xlabel('Time (periods)')
ax1.set_ylabel('<x> / L')
ax1.set_title('Position Expectation Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot uncertainty
ax2 = axes[1]
ax2.plot(t_values/T_period, sigma_x, 'g-', linewidth=2)
ax2.set_xlabel('Time (periods)')
ax2.set_ylabel('sigma_x / L')
ax2.set_title('Position Uncertainty')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_360_expectation_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_360_expectation_dynamics.png'")

# =============================================================================
# Part 4: Multi-Level Superposition
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Multi-Level Superposition (Quantum Beats)")
print("=" * 60)

# Three-level superposition
n_values_3 = [1, 2, 3]
coeffs_3 = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]

# Find all Bohr frequencies
E_levels = [energy(n, m, L, hbar) for n in n_values_3]
print("Energy levels:", E_levels)
print("\nBohr frequencies:")
for i in range(len(n_values_3)):
    for j in range(i+1, len(n_values_3)):
        omega_ij = (E_levels[j] - E_levels[i]) / hbar
        T_ij = 2 * np.pi / omega_ij
        print(f"  omega_{n_values_3[j]}{n_values_3[i]} = {omega_ij:.4f}, T = {T_ij:.4f}")

# Plot probability evolution
fig, ax = plt.subplots(figsize=(12, 6))

# Compute over longer time to see beats
t_max = 5 * T_period
t_dense = np.linspace(0, t_max, 500)

# Track probability at specific points
x_probe = [0.25 * L, 0.5 * L, 0.75 * L]
colors = ['blue', 'green', 'red']

for x_val, color in zip(x_probe, colors):
    prob_at_x = []
    x_idx = np.argmin(np.abs(x - x_val))

    for t in t_dense:
        psi_t = time_evolved_superposition(x, t, coeffs_3, n_values_3, L, hbar, m)
        prob_at_x.append(np.abs(psi_t[x_idx])**2)

    ax.plot(t_dense/T_period, prob_at_x, color=color, linewidth=1.5,
            label=f'x = {x_val/L:.2f}L', alpha=0.8)

ax.set_xlabel('Time (periods of omega_21)')
ax.set_ylabel('Probability density |psi(x,t)|^2')
ax.set_title('Quantum Beats: Three-Level Superposition')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_360_quantum_beats.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_360_quantum_beats.png'")

# =============================================================================
# Part 5: Stationary vs Non-Stationary Comparison
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Stationary vs Non-Stationary States")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time points
times_compare = np.linspace(0, 2*T_period, 50)

# Left: Stationary state (n=2)
ax1 = axes[0]
for t in times_compare[::5]:  # Plot every 5th time
    psi_stationary = psi_n(x, 2, L) * np.exp(-1j * energy(2) * t)
    prob = np.abs(psi_stationary)**2
    ax1.plot(x, prob, 'b-', alpha=0.3)

ax1.set_xlabel('Position x/L')
ax1.set_ylabel('|psi|^2')
ax1.set_title('Stationary State |2>: All times overlap')
ax1.set_xlim(0, L)
ax1.grid(True, alpha=0.3)

# Right: Superposition (changes with time)
ax2 = axes[1]
for idx, t in enumerate(times_compare[::5]):
    psi_super = time_evolved_superposition(x, t, coeffs, n_values, L, hbar, m)
    prob = np.abs(psi_super)**2
    alpha = 0.1 + 0.8 * idx / len(times_compare[::5])
    ax2.plot(x, prob, 'purple', alpha=alpha)

ax2.set_xlabel('Position x/L')
ax2.set_ylabel('|psi|^2')
ax2.set_title('Superposition (|1> + |2>)/sqrt(2): Time evolution')
ax2.set_xlim(0, L)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_360_stationary_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_360_stationary_comparison.png'")

# =============================================================================
# Part 6: Energy Conservation Check
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Energy Conservation")
print("=" * 60)

# For the superposition, compute <H> at different times
# <H> should be constant!

def compute_energy_expectation(psi, x, m=1.0, hbar=1.0, V=None):
    """Compute <H> = <T> + <V> numerically."""
    dx = x[1] - x[0]

    # Kinetic energy: -hbar^2/(2m) * d^2psi/dx^2
    d2psi = np.gradient(np.gradient(psi, dx), dx)
    T_psi = -hbar**2 / (2*m) * d2psi

    # For infinite well, V=0 inside
    if V is None:
        V = np.zeros_like(x)

    H_psi = T_psi + V * psi

    # <H> = integral of psi* H psi
    return np.real(np.trapz(np.conj(psi) * H_psi, x))

H_expect = []
for t in t_values:
    psi_t = time_evolved_superposition(x, t, coeffs, n_values, L, hbar, m)
    H_expect.append(compute_energy_expectation(psi_t, x, m, hbar))

H_expect = np.array(H_expect)

# Theoretical value: <H> = |c1|^2 E1 + |c2|^2 E2
H_theory = np.abs(coeffs[0])**2 * E1 + np.abs(coeffs[1])**2 * E2

print(f"Theoretical <H> = {H_theory:.6f}")
print(f"Numerical <H> (mean) = {np.mean(H_expect):.6f}")
print(f"Numerical <H> (std) = {np.std(H_expect):.6e}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_values/T_period, H_expect, 'b-', linewidth=2)
ax.axhline(y=H_theory, color='r', linestyle='--', linewidth=2, label='Theoretical')
ax.set_xlabel('Time (periods)')
ax.set_ylabel('<H>')
ax.set_title('Energy Conservation: <H> vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_360_energy_conservation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_360_energy_conservation.png'")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| TISE | $\hat{H}\|E\rangle = E\|E\rangle$ |
| Stationary state evolution | $\|E, t\rangle = e^{-iEt/\hbar}\|E\rangle$ |
| General evolution | $\|\psi(t)\rangle = \sum_n c_n e^{-iE_n t/\hbar}\|E_n\rangle$ |
| Bohr frequency | $\omega_{mn} = (E_m - E_n)/\hbar$ |
| Time-independent probability | $\|\langle x\|E\rangle\|^2$ = constant |

### Main Takeaways

1. **Stationary states** are energy eigenstates with time-independent probability densities
2. **The TISE** $\hat{H}|\psi\rangle = E|\psi\rangle$ determines allowed energies and wave functions
3. **General states** are superpositions that evolve with phases $e^{-iE_n t/\hbar}$
4. **Probability densities** of superpositions oscillate at Bohr frequencies
5. **Energy is conserved** — $\langle\hat{H}\rangle$ is constant in time
6. **Quantum beats** arise from multiple frequency components

---

## Daily Checklist

- [ ] Read Shankar Chapter 4.4-4.5 and Griffiths Chapter 2.1
- [ ] Derive the TISE from separation of variables
- [ ] Show stationary states have time-independent probability
- [ ] Calculate oscillation frequencies for two-level superpositions
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and understand the computational lab
- [ ] Visualize quantum beats with multiple energy levels

---

## Preview: Day 361

Tomorrow we formalize the **Schrodinger picture** — the framework where states evolve and operators remain fixed. We'll contrast this with the Heisenberg picture (Day 362) where the roles are reversed.

---

*"In the Schrodinger picture, the state vector is the fundamental entity... In the Heisenberg picture, it is the operators that are fundamental."*
— J. J. Sakurai

---

**Next:** [Day_361_Thursday.md](Day_361_Thursday.md) — Schrodinger Picture
