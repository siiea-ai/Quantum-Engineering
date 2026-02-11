# Day 379: Quantum Harmonic Oscillator — Setup & Motivation

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Classical Review & Quantum Setup |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 379, you will be able to:

1. Review the classical harmonic oscillator and its phase space behavior
2. Write the quantum Hamiltonian for the harmonic oscillator
3. Introduce dimensionless variables to simplify the problem
4. Understand why the QHO is fundamental to physics (Taylor expansion)
5. Recognize the QHO in molecular, optical, and solid-state systems
6. Compare classical and quantum oscillator behavior computationally

---

## Core Content

### 1. The Classical Harmonic Oscillator

The harmonic oscillator describes a particle subject to a restoring force proportional to displacement:

$$F = -kx = -m\omega^2 x$$

where $\omega = \sqrt{k/m}$ is the angular frequency.

#### Classical Equation of Motion

Newton's second law gives:

$$m\ddot{x} = -m\omega^2 x$$

$$\boxed{\ddot{x} + \omega^2 x = 0}$$

**Solution:**
$$x(t) = A\cos(\omega t + \phi)$$

or equivalently:
$$x(t) = C_1 e^{i\omega t} + C_2 e^{-i\omega t}$$

#### Classical Hamiltonian

The potential energy for a spring-like system is:

$$V(x) = \frac{1}{2}kx^2 = \frac{1}{2}m\omega^2 x^2$$

The total energy (Hamiltonian) is:

$$\boxed{H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 x^2}$$

This defines an **ellipse** in phase space $(x, p)$:

$$\frac{x^2}{2E/m\omega^2} + \frac{p^2}{2mE} = 1$$

#### Key Classical Properties

| Property | Value |
|----------|-------|
| Energy | E (continuous, any positive value) |
| Period | $T = 2\pi/\omega$ (independent of amplitude!) |
| Maximum displacement | $x_{max} = \sqrt{2E/m\omega^2}$ |
| Maximum momentum | $p_{max} = \sqrt{2mE}$ |

---

### 2. Why the Harmonic Oscillator Is Ubiquitous

The harmonic oscillator appears everywhere in physics because of **Taylor's theorem**.

#### Taylor Expansion of Any Potential

Consider any smooth potential V(x) with a minimum at $x = x_0$:

$$V(x) = V(x_0) + V'(x_0)(x - x_0) + \frac{1}{2}V''(x_0)(x - x_0)^2 + \mathcal{O}((x-x_0)^3)$$

At a minimum:
- $V'(x_0) = 0$ (definition of minimum)
- $V''(x_0) > 0$ (stability condition)

Dropping the constant and higher-order terms:

$$\boxed{V(x) \approx \frac{1}{2}V''(x_0)(x - x_0)^2 = \frac{1}{2}k_{\text{eff}}(x - x_0)^2}$$

**Any system near a stable equilibrium behaves as a harmonic oscillator!**

#### Physical Examples

| System | "Spring Constant" | Frequency |
|--------|-------------------|-----------|
| Molecule (bond vibration) | $k = V''(r_0)$ | ~$10^{13}$ Hz (infrared) |
| Pendulum (small angle) | $k = mg/L$ | ~1 Hz |
| LC circuit | $k = 1/LC$ | Radio to microwave |
| Electromagnetic field mode | $k = 1$ (per mode) | Any frequency |
| Crystal lattice (phonon) | Interatomic forces | ~$10^{12}$ Hz |

---

### 3. The Quantum Hamiltonian

We now promote classical observables to quantum operators:

$$x \to \hat{x}, \quad p \to \hat{p}$$

with the canonical commutation relation:

$$[\hat{x}, \hat{p}] = i\hbar$$

The **quantum harmonic oscillator Hamiltonian** is:

$$\boxed{\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2}$$

#### The Time-Independent Schrodinger Equation

We seek energy eigenstates:

$$\hat{H}|\psi_n\rangle = E_n|\psi_n\rangle$$

In position representation, $\hat{p} = -i\hbar\frac{d}{dx}$, giving:

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + \frac{1}{2}m\omega^2 x^2 \psi = E\psi$$

This is a second-order ODE with a **Gaussian-like solution** (as we'll see).

---

### 4. Introducing Dimensionless Variables

The QHO has a **natural length scale** and **energy scale**:

#### Characteristic Length

$$\boxed{x_0 = \sqrt{\frac{\hbar}{m\omega}}}$$

This is the ground state width — the scale of quantum fluctuations.

**Physical interpretation:** The uncertainty principle gives $\Delta x \cdot \Delta p \gtrsim \hbar$. For the ground state, $\Delta p \sim m\omega\Delta x$, so:

$$\Delta x \cdot m\omega\Delta x \sim \hbar \implies \Delta x \sim \sqrt{\frac{\hbar}{m\omega}} = x_0$$

#### Characteristic Energy

$$\boxed{E_0 = \hbar\omega}$$

This is the quantum of energy for the oscillator.

#### Dimensionless Position

Define:

$$\boxed{\xi = \frac{x}{x_0} = x\sqrt{\frac{m\omega}{\hbar}}}$$

Then:
$$\frac{d}{dx} = \frac{1}{x_0}\frac{d}{d\xi} = \sqrt{\frac{m\omega}{\hbar}}\frac{d}{d\xi}$$

#### Dimensionless Schrodinger Equation

Substituting into the TISE:

$$-\frac{\hbar^2}{2m} \cdot \frac{m\omega}{\hbar} \frac{d^2\psi}{d\xi^2} + \frac{1}{2}m\omega^2 \cdot \frac{\hbar}{m\omega}\xi^2\psi = E\psi$$

Simplifying:

$$-\frac{\hbar\omega}{2}\frac{d^2\psi}{d\xi^2} + \frac{\hbar\omega}{2}\xi^2\psi = E\psi$$

Let $\varepsilon = E/(\hbar\omega/2) = 2E/\hbar\omega$:

$$\boxed{-\frac{d^2\psi}{d\xi^2} + \xi^2\psi = \varepsilon\psi}$$

This elegant equation has **no parameters** — all physics is encoded in $\varepsilon$!

---

### 5. Asymptotic Behavior

For large $|\xi|$:
$$\frac{d^2\psi}{d\xi^2} \approx \xi^2\psi$$

The solutions behave like:
$$\psi \sim e^{\pm\xi^2/2}$$

We must choose $e^{-\xi^2/2}$ for normalizability (square-integrable).

**Ansatz:** Factor out the asymptotic behavior:

$$\boxed{\psi(\xi) = h(\xi)e^{-\xi^2/2}}$$

where $h(\xi)$ is a polynomial (to be determined).

Substituting gives **Hermite's differential equation** for $h(\xi)$:

$$\frac{d^2h}{d\xi^2} - 2\xi\frac{dh}{d\xi} + (\varepsilon - 1)h = 0$$

The solutions are **Hermite polynomials** when $\varepsilon = 2n + 1$ for non-negative integers $n$.

---

### 6. Preview of Results

The full solution (derived this week) gives:

#### Energy Eigenvalues

$$\boxed{E_n = \hbar\omega\left(n + \frac{1}{2}\right)}, \quad n = 0, 1, 2, 3, \ldots$$

| Level | Energy | Spacing from ground |
|-------|--------|---------------------|
| $n = 0$ | $\frac{1}{2}\hbar\omega$ | 0 |
| $n = 1$ | $\frac{3}{2}\hbar\omega$ | $\hbar\omega$ |
| $n = 2$ | $\frac{5}{2}\hbar\omega$ | $2\hbar\omega$ |
| $n = 3$ | $\frac{7}{2}\hbar\omega$ | $3\hbar\omega$ |

**Key features:**
1. **Equally spaced levels:** $\Delta E = \hbar\omega$ (unlike hydrogen!)
2. **Zero-point energy:** $E_0 = \frac{1}{2}\hbar\omega \neq 0$ (pure quantum effect)
3. **Discrete spectrum:** Energy is quantized (unlike free particle)

#### Physical Origin of Zero-Point Energy

Classical: A particle can sit at rest at $x = 0$ with $E = 0$.

Quantum: The uncertainty principle forbids this!
- If $\Delta x = 0$, then $\Delta p = \infty$, giving infinite kinetic energy.
- Minimizing total energy gives a compromise: finite $\Delta x$ and $\Delta p$.

$$\boxed{E_0 = \frac{\hbar\omega}{2} = \text{zero-point energy (vacuum energy)}}$$

---

### 7. Quantum Computing Connection: Bosonic Qubits

The harmonic oscillator is central to **bosonic quantum computing**:

#### Fock Encoding
The infinite-dimensional Hilbert space $\{|0\rangle, |1\rangle, |2\rangle, \ldots\}$ can encode quantum information:

| Encoding | Basis States |
|----------|--------------|
| Fock qubit | $|0\rangle, |1\rangle$ (photon number) |
| Cat qubit | $|\alpha\rangle + |-\alpha\rangle$, $|\alpha\rangle - |-\alpha\rangle$ |
| GKP qubit | Grid states in phase space |

#### Physical Platforms
- **Superconducting circuits:** Microwave resonators coupled to transmons
- **Trapped ions:** Motional modes of ions in harmonic traps
- **Photonic systems:** Optical cavities, squeezed light

**Advantage:** The large Hilbert space allows hardware-efficient error correction!

---

## Worked Examples

### Example 1: Classical Phase Space Trajectories

**Problem:** A classical oscillator has mass $m = 1$ kg, $\omega = 2\pi$ rad/s, and energy $E = 5$ J. Find:
(a) Maximum displacement
(b) Maximum momentum
(c) The phase space ellipse equation

**Solution:**

(a) Maximum displacement occurs when $p = 0$:
$$E = \frac{1}{2}m\omega^2 x_{max}^2$$
$$x_{max} = \sqrt{\frac{2E}{m\omega^2}} = \sqrt{\frac{2 \times 5}{1 \times (2\pi)^2}} = \sqrt{\frac{10}{4\pi^2}} \approx 0.50 \text{ m}$$

(b) Maximum momentum occurs when $x = 0$:
$$E = \frac{p_{max}^2}{2m}$$
$$p_{max} = \sqrt{2mE} = \sqrt{2 \times 1 \times 5} = \sqrt{10} \approx 3.16 \text{ kg m/s}$$

(c) Phase space ellipse:
$$\frac{x^2}{x_{max}^2} + \frac{p^2}{p_{max}^2} = 1$$
$$\frac{x^2}{0.25} + \frac{p^2}{10} = 1$$

The particle traces this ellipse with period $T = 1$ s. $\blacksquare$

---

### Example 2: Characteristic Scales

**Problem:** For a hydrogen molecule (H$_2$) vibration with $\omega = 8.3 \times 10^{13}$ rad/s and reduced mass $\mu = 8.4 \times 10^{-28}$ kg, calculate:
(a) The characteristic length $x_0$
(b) The zero-point energy in eV
(c) The energy of the first excited state

**Solution:**

(a) Characteristic length:
$$x_0 = \sqrt{\frac{\hbar}{m\omega}} = \sqrt{\frac{1.055 \times 10^{-34}}{8.4 \times 10^{-28} \times 8.3 \times 10^{13}}}$$
$$x_0 = \sqrt{\frac{1.055 \times 10^{-34}}{6.97 \times 10^{-14}}} = \sqrt{1.51 \times 10^{-21}} \approx 1.23 \times 10^{-11} \text{ m}$$
$$\boxed{x_0 \approx 0.12 \text{ \AA}}$$

(b) Zero-point energy:
$$E_0 = \frac{1}{2}\hbar\omega = \frac{1}{2}(1.055 \times 10^{-34})(8.3 \times 10^{13})$$
$$E_0 = 4.38 \times 10^{-21} \text{ J} = \frac{4.38 \times 10^{-21}}{1.6 \times 10^{-19}} \approx 0.027 \text{ eV}$$
$$\boxed{E_0 \approx 27 \text{ meV}}$$

(c) First excited state:
$$E_1 = \frac{3}{2}\hbar\omega = 3E_0 \approx 81 \text{ meV}$$

The transition energy $\Delta E = E_1 - E_0 = \hbar\omega \approx 54$ meV corresponds to infrared light ($\lambda \approx 23\ \mu$m). $\blacksquare$

---

### Example 3: Dimensionless Form

**Problem:** Show that in dimensionless variables, the harmonic oscillator potential becomes $V(\xi) = \frac{\hbar\omega}{2}\xi^2$.

**Solution:**

Starting with:
$$V(x) = \frac{1}{2}m\omega^2 x^2$$

With $x = x_0 \xi = \sqrt{\frac{\hbar}{m\omega}}\xi$:

$$V = \frac{1}{2}m\omega^2 \cdot \frac{\hbar}{m\omega}\xi^2 = \frac{1}{2}\hbar\omega\xi^2$$

$$\boxed{V(\xi) = \frac{\hbar\omega}{2}\xi^2}$$

This shows that all harmonic oscillators are fundamentally the same when expressed in natural units! $\blacksquare$

---

## Practice Problems

### Level 1: Direct Application

1. **Classical Review:** A mass-spring system has $m = 0.5$ kg and $k = 200$ N/m.
   (a) Find the angular frequency $\omega$.
   (b) If released from $x_0 = 0.1$ m with zero velocity, find the energy.
   (c) Write $x(t)$ and $p(t)$.

2. **Natural Units:** For the system in Problem 1, calculate:
   (a) The characteristic quantum length $x_0 = \sqrt{\hbar/m\omega}$
   (b) The zero-point energy
   (c) How many quanta $n$ would be needed to reach the classical energy?

3. **Dimensionless Variables:** Express the position $x = 3 \times 10^{-11}$ m in dimensionless units $\xi$ for an oscillator with $m = 10^{-26}$ kg and $\omega = 10^{14}$ rad/s.

### Level 2: Intermediate

4. **Taylor Expansion:** The Lennard-Jones potential for molecular interactions is:
   $$V(r) = 4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]$$
   (a) Find the equilibrium distance $r_0$ where $V'(r_0) = 0$.
   (b) Compute $V''(r_0)$ and find the effective spring constant.
   (c) Express the vibrational frequency in terms of $\epsilon$, $\sigma$, and $m$.

5. **Dimensional Analysis:** Using only $\hbar$, $m$, and $\omega$, construct:
   (a) A quantity with dimensions of length
   (b) A quantity with dimensions of energy
   (c) A quantity with dimensions of momentum

6. **Phase Space Area:** Show that the area enclosed by a classical phase space trajectory is $A = 2\pi E/\omega$. What is the quantum of area (minimum enclosed area)?

### Level 3: Challenging

7. **Zero-Point Energy Estimation:** Use the uncertainty principle to estimate the ground state energy of the QHO:
   (a) Write $E = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 x^2$ in terms of $\Delta x$ and $\Delta p$.
   (b) Use $\Delta x \cdot \Delta p \geq \hbar/2$ to eliminate $\Delta p$.
   (c) Minimize $E$ with respect to $\Delta x$ to find $E_0$.
   (d) Compare with the exact result $E_0 = \hbar\omega/2$.

8. **Coherence Length:** The thermal de Broglie wavelength is $\lambda_{dB} = h/\sqrt{2\pi m k_B T}$. At what temperature does $\lambda_{dB}$ equal the QHO ground state width $x_0$? This is when quantum effects become important.

9. **Research Problem:** The Casimir effect arises from zero-point energy of electromagnetic field modes. If each mode contributes $\frac{1}{2}\hbar\omega$, explain why the total zero-point energy diverges and how the Casimir force between conducting plates nonetheless emerges as a finite, measurable quantity.

---

## Computational Lab

### Objective
Compare classical and quantum harmonic oscillators, visualize phase space trajectories, and explore the quantum potential.

```python
"""
Day 379 Computational Lab: QHO Setup & Classical Comparison
Quantum Harmonic Oscillator - Week 55
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =============================================================================
# Part 1: Classical Harmonic Oscillator Phase Space
# =============================================================================

print("=" * 70)
print("Part 1: Classical Harmonic Oscillator Phase Space")
print("=" * 70)

# Physical parameters
m = 1.0      # mass (kg)
omega = 2.0  # angular frequency (rad/s)

# Classical equations of motion
def classical_eom(y, t, m, omega):
    """dy/dt for classical harmonic oscillator"""
    x, p = y
    dxdt = p / m
    dpdt = -m * omega**2 * x
    return [dxdt, dpdt]

# Simulate trajectories for different energies
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Phase space plot
ax1 = axes[0]
t = np.linspace(0, 10, 1000)
energies = [0.5, 1.0, 2.0, 3.0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(energies)))

for E, color in zip(energies, colors):
    # Initial condition: all energy in position
    x0 = np.sqrt(2 * E / (m * omega**2))
    p0 = 0

    solution = odeint(classical_eom, [x0, p0], t, args=(m, omega))
    x_traj = solution[:, 0]
    p_traj = solution[:, 1]

    ax1.plot(x_traj, p_traj, color=color, label=f'E = {E:.1f} J')

ax1.set_xlabel('Position x (m)', fontsize=12)
ax1.set_ylabel('Momentum p (kg m/s)', fontsize=12)
ax1.set_title('Classical Phase Space Trajectories', fontsize=14)
ax1.legend()
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Time evolution plot
ax2 = axes[1]
E = 2.0
x0 = np.sqrt(2 * E / (m * omega**2))
solution = odeint(classical_eom, [x0, 0], t, args=(m, omega))

ax2.plot(t, solution[:, 0], 'b-', label='x(t)', linewidth=2)
ax2.plot(t, solution[:, 1], 'r--', label='p(t)', linewidth=2)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('x (m) or p (kg m/s)', fontsize=12)
ax2.set_title(f'Classical Oscillator (E = {E} J)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_379_classical_phase_space.png', dpi=150, bbox_inches='tight')
plt.show()

print("Classical phase space figure saved.")

# =============================================================================
# Part 2: Quantum vs Classical Scales
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Quantum vs Classical Characteristic Scales")
print("=" * 70)

# Constants
hbar = 1.055e-34  # J s
m_electron = 9.11e-31  # kg
m_proton = 1.67e-27  # kg
m_macro = 1.0  # kg

# Function to compute QHO scales
def qho_scales(m, omega, name=""):
    """Compute characteristic scales for QHO"""
    x0 = np.sqrt(hbar / (m * omega))  # characteristic length
    E0 = 0.5 * hbar * omega            # zero-point energy
    p0 = np.sqrt(hbar * m * omega)     # characteristic momentum

    print(f"\n{name}")
    print("-" * 50)
    print(f"  Mass: {m:.2e} kg")
    print(f"  Frequency: {omega:.2e} rad/s = {omega/(2*np.pi):.2e} Hz")
    print(f"  Characteristic length x_0: {x0:.2e} m")
    print(f"  Zero-point energy E_0: {E0:.2e} J = {E0/1.6e-19:.2e} eV")
    print(f"  Characteristic momentum p_0: {p0:.2e} kg m/s")

    return x0, E0, p0

# Example systems
print("\n--- Quantum Systems ---")
qho_scales(m_electron, 1e15, "Electron in quantum dot")
qho_scales(m_proton, 8.3e13, "H2 molecule vibration")
qho_scales(1e-25, 1e12, "Phonon in crystal")

print("\n--- Macroscopic System ---")
qho_scales(1.0, 2*np.pi, "Pendulum (1 kg, 1 Hz)")

# =============================================================================
# Part 3: Dimensionless Potential
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Dimensionless Harmonic Potential")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Dimensional potential
ax1 = axes[0]
x_vals = np.linspace(-3, 3, 500)
for omega_val, label in [(1.0, r'$\omega = 1$'),
                          (2.0, r'$\omega = 2$'),
                          (0.5, r'$\omega = 0.5$')]:
    V = 0.5 * m * omega_val**2 * x_vals**2
    ax1.plot(x_vals, V, label=label, linewidth=2)

ax1.set_xlabel('Position x (m)', fontsize=12)
ax1.set_ylabel('V(x) (J)', fontsize=12)
ax1.set_title('Harmonic Potential (Dimensional)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 5)

# Dimensionless potential with energy levels
ax2 = axes[1]
xi_vals = np.linspace(-5, 5, 500)
V_dimless = xi_vals**2  # In units of hbar*omega/2

ax2.plot(xi_vals, V_dimless, 'k-', linewidth=2, label=r'$V(\xi)/(\hbar\omega/2) = \xi^2$')

# Add quantum energy levels
for n in range(6):
    E_n = 2*n + 1  # dimensionless energy (epsilon = 2E/hbar*omega)
    # Classical turning points
    xi_turn = np.sqrt(E_n)
    ax2.hlines(E_n, -xi_turn, xi_turn, colors=f'C{n}', linewidth=2,
               label=f'n={n}: $\\varepsilon$ = {E_n}' if n < 4 else None)

ax2.set_xlabel(r'Dimensionless position $\xi = x/x_0$', fontsize=12)
ax2.set_ylabel(r'Energy / $(\hbar\omega/2)$', fontsize=12)
ax2.set_title('Dimensionless Potential with Quantum Levels', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 15)
ax2.set_xlim(-5, 5)

plt.tight_layout()
plt.savefig('day_379_dimensionless_potential.png', dpi=150, bbox_inches='tight')
plt.show()

print("Dimensionless potential figure saved.")

# =============================================================================
# Part 4: Classical vs Quantum Energy Spectrum
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Classical vs Quantum Energy Spectrum")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

# Classical: continuous spectrum (shown as gradient)
E_class = np.linspace(0, 6, 100)
for E in E_class:
    ax.axhline(E, xmin=0.05, xmax=0.45, color='blue', alpha=0.1, linewidth=1)

ax.text(0.25, 6.3, 'Classical\n(continuous)', ha='center', fontsize=12, color='blue')

# Quantum: discrete spectrum
hbar_omega = 1.0  # set hbar*omega = 1 for display
for n in range(7):
    E_n = hbar_omega * (n + 0.5)
    ax.hlines(E_n, 0.55, 0.95, colors='red', linewidth=3)
    ax.text(0.97, E_n, f'$n = {n}$', va='center', fontsize=10)

# Zero-point energy annotation
ax.annotate('', xy=(0.75, 0), xytext=(0.75, 0.5),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(0.78, 0.25, r'$E_0 = \frac{1}{2}\hbar\omega$', fontsize=11, color='green')

# Equal spacing annotation
ax.annotate('', xy=(0.65, 1.5), xytext=(0.65, 2.5),
            arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
ax.text(0.67, 2.0, r'$\Delta E = \hbar\omega$', fontsize=11, color='orange')

ax.text(0.75, 6.3, 'Quantum\n(discrete)', ha='center', fontsize=12, color='red')

ax.set_ylabel(r'Energy / $\hbar\omega$', fontsize=12)
ax.set_xlim(0, 1.1)
ax.set_ylim(-0.2, 7)
ax.set_title('Classical vs Quantum Energy Spectrum', fontsize=14)
ax.set_xticks([])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('day_379_energy_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()

print("Energy spectrum comparison figure saved.")

# =============================================================================
# Part 5: Taylor Expansion Example
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Harmonic Approximation to Morse Potential")
print("=" * 70)

def morse_potential(r, D_e, a, r_e):
    """Morse potential for molecular vibration"""
    return D_e * (1 - np.exp(-a * (r - r_e)))**2

def harmonic_approx(r, D_e, a, r_e):
    """Harmonic approximation near minimum"""
    k_eff = 2 * D_e * a**2
    return 0.5 * k_eff * (r - r_e)**2

# Morse parameters (approximate H2 molecule)
D_e = 4.75  # eV (dissociation energy)
a = 1.93    # 1/angstrom
r_e = 0.74  # angstrom (equilibrium)

r = np.linspace(0.4, 3.0, 500)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(r, morse_potential(r, D_e, a, r_e), 'b-', linewidth=2, label='Morse (exact)')
ax.plot(r, harmonic_approx(r, D_e, a, r_e), 'r--', linewidth=2, label='Harmonic approx.')

# Mark equilibrium and dissociation
ax.axhline(D_e, color='gray', linestyle=':', alpha=0.7, label=f'Dissociation: $D_e$ = {D_e} eV')
ax.axvline(r_e, color='green', linestyle=':', alpha=0.7)

# Add energy levels (Morse exact for illustration)
k_eff = 2 * D_e * a**2
omega_e = np.sqrt(k_eff / 1.0)  # reduced mass = 1 for illustration

for n in range(5):
    E_harm = 0.27 * (n + 0.5)  # approximate harmonic energy
    ax.hlines(E_harm, r_e - 0.3/(n+1)**0.5, r_e + 0.3/(n+1)**0.5,
              colors='red', alpha=0.5)

ax.set_xlabel('Internuclear distance r (angstrom)', fontsize=12)
ax.set_ylabel('Potential energy (eV)', fontsize=12)
ax.set_title('Morse Potential vs Harmonic Approximation (H$_2$ molecule)', fontsize=14)
ax.legend(loc='upper right')
ax.set_xlim(0.4, 3.0)
ax.set_ylim(-0.5, 6)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_379_morse_harmonic.png', dpi=150, bbox_inches='tight')
plt.show()

print("Morse vs harmonic potential figure saved.")

# =============================================================================
# Part 6: Quantum Number Estimation
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Classical to Quantum Energy Correspondence")
print("=" * 70)

# Macroscopic pendulum
m_pend = 1.0  # kg
omega_pend = 2 * np.pi  # 1 Hz
E_classical = 0.1  # J (typical energy)

E_quantum = hbar * omega_pend / 2  # zero-point energy
n_quanta = int(E_classical / (hbar * omega_pend))

print(f"\nMacroscopic Pendulum (m = 1 kg, f = 1 Hz)")
print(f"  Classical energy: {E_classical} J")
print(f"  Zero-point energy: {E_quantum:.2e} J")
print(f"  Number of quanta at classical energy: n ~ {n_quanta:.2e}")
print(f"  Relative spacing: Delta E / E ~ {1/n_quanta:.2e}")
print(f"  --> Spectrum appears continuous (classical limit)")

# Molecular vibration
m_mol = 1.67e-27  # proton mass
omega_mol = 8.3e13  # rad/s
E_mol_classical = 0.05 * 1.6e-19  # 50 meV

E_mol_quantum = hbar * omega_mol / 2
n_mol = int(E_mol_classical / (hbar * omega_mol))

print(f"\nMolecular Vibration (H2, omega = 8.3e13 rad/s)")
print(f"  Thermal energy at 300K: ~25 meV")
print(f"  Zero-point energy: {E_mol_quantum/1.6e-19*1000:.1f} meV")
print(f"  Number of quanta at 50 meV: n ~ {n_mol}")
print(f"  --> Quantum effects dominate!")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Classical Hamiltonian | $H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 x^2$ |
| Quantum Hamiltonian | $\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$ |
| Characteristic length | $x_0 = \sqrt{\hbar/m\omega}$ |
| Characteristic energy | $E_0 = \hbar\omega$ |
| Dimensionless position | $\xi = x/x_0 = x\sqrt{m\omega/\hbar}$ |
| Dimensionless TISE | $-\frac{d^2\psi}{d\xi^2} + \xi^2\psi = \varepsilon\psi$ |
| Energy spectrum (preview) | $E_n = \hbar\omega(n + \frac{1}{2})$ |

### Main Takeaways

1. **Universality:** The QHO describes any system near a stable equilibrium (Taylor expansion)
2. **Natural scales:** $x_0 = \sqrt{\hbar/m\omega}$ and $E_0 = \hbar\omega$ set the quantum regime
3. **Dimensionless form:** The TISE becomes parameter-free in natural units
4. **Zero-point energy:** Quantum mechanics forbids $E = 0$ due to uncertainty principle
5. **Applications:** Molecular vibrations, phonons, photons, trapped ions, superconducting qubits

---

## Daily Checklist

- [ ] Read Shankar Chapter 7, Sections 7.1-7.2
- [ ] Review classical harmonic oscillator from Year 0 (Month 6)
- [ ] Work through all three examples
- [ ] Complete Level 1 practice problems
- [ ] Attempt the uncertainty principle derivation (Problem 7)
- [ ] Run and understand the computational lab
- [ ] Identify one physical system where QHO applies in your area of interest

---

## Preview: Day 380

Tomorrow we introduce the **ladder operator method** — an elegant algebraic approach that solves the QHO without solving any differential equations! We'll define $\hat{a}$ and $\hat{a}^\dagger$ and prove the fundamental commutator $[\hat{a}, \hat{a}^\dagger] = 1$.

---

*"Any smooth potential can be approximated near a minimum by a harmonic potential; that is why the simple harmonic oscillator has played such an important role in physics."*
— R. Shankar

---

**Next:** [Day_380_Tuesday.md](Day_380_Tuesday.md) — Ladder Operators
