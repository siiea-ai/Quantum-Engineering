# Day 473: Harmonic Perturbations

## Overview
**Day 473** | Year 1, Month 17, Week 68 | Oscillating Fields and Resonance

Today we analyze harmonic (sinusoidal) perturbations, the foundation for understanding absorption and stimulated emission of radiation.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Harmonic perturbation theory |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Resonance and detuning |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Rabi oscillation simulation |

---

## Learning Objectives

By the end of today, you will be able to:
1. Solve the time-dependent Schrödinger equation for harmonic perturbations
2. Identify absorption and stimulated emission resonances
3. Calculate transition rates near resonance
4. Understand the rotating wave approximation (RWA)
5. Derive Rabi oscillations from exact two-level dynamics
6. Connect to laser-atom interactions

---

## Core Content

### Harmonic Perturbation

Consider a sinusoidally oscillating perturbation:
$$V(t) = V_0 \cos(\omega t) = \frac{V_0}{2}(e^{i\omega t} + e^{-i\omega t})$$

where V₀ is time-independent and ω is the driving frequency.

### First-Order Amplitude

For the transition |i⟩ → |f⟩:
$$c_f^{(1)}(t) = -\frac{i}{\hbar}\int_0^t V_{fi}\cos(\omega t')e^{i\omega_{fi}t'}\,dt'$$

$$= -\frac{iV_{fi}}{2\hbar}\int_0^t \left(e^{i(\omega_{fi}+\omega)t'} + e^{i(\omega_{fi}-\omega)t'}\right)dt'$$

### Two Resonance Terms

$$c_f^{(1)}(t) = -\frac{V_{fi}}{2\hbar}\left[\frac{e^{i(\omega_{fi}+\omega)t}-1}{\omega_{fi}+\omega} + \frac{e^{i(\omega_{fi}-\omega)t}-1}{\omega_{fi}-\omega}\right]$$

### Absorption vs Stimulated Emission

**Absorption:** E_f > E_i (ω_{fi} > 0)
- Resonance when ω ≈ ω_{fi}
- Second term dominates (ω_{fi} - ω ≈ 0)
- System absorbs energy from field

**Stimulated Emission:** E_f < E_i (ω_{fi} < 0)
- Resonance when ω ≈ |ω_{fi}| = -ω_{fi}
- First term dominates (ω_{fi} + ω ≈ 0)
- System releases energy to field

### Near-Resonance Approximation

For absorption near ω ≈ ω_{fi}:
$$c_f^{(1)}(t) \approx -\frac{V_{fi}}{2\hbar}\frac{e^{i(\omega_{fi}-\omega)t}-1}{\omega_{fi}-\omega}$$

$$|c_f^{(1)}(t)|^2 = \frac{|V_{fi}|^2}{\hbar^2}\frac{\sin^2[(\omega_{fi}-\omega)t/2]}{(\omega_{fi}-\omega)^2}$$

### Transition Rate (Fermi's Golden Rule for Harmonic)

At long times:
$$\boxed{W_{i \to f} = \frac{\pi|V_{fi}|^2}{2\hbar^2}\left[\delta(\omega_{fi}-\omega) + \delta(\omega_{fi}+\omega)\right]}$$

The δ-functions enforce:
- Absorption: ℏω = E_f - E_i (photon absorbed)
- Emission: ℏω = E_i - E_f (photon emitted)

---

## The Rotating Wave Approximation (RWA)

### Full Hamiltonian (Two-Level System)

$$H = H_0 + V_0\cos(\omega t)|e\rangle\langle g| + \text{h.c.}$$

In the interaction picture:
$$H_I = \frac{\Omega}{2}(e^{i(\omega_{eg}-\omega)t} + e^{i(\omega_{eg}+\omega)t})|e\rangle\langle g| + \text{h.c.}$$

where Ω = V_{eg}/ℏ is the Rabi frequency.

### RWA: Drop Fast-Rotating Terms

Near resonance (ω ≈ ω_{eg}):
- $e^{i(\omega_{eg}-\omega)t}$ is slow
- $e^{i(\omega_{eg}+\omega)t}$ oscillates at ~2ω (fast)

Drop the fast term:
$$H_{RWA} = \frac{\hbar\Omega}{2}(e^{-i\delta t}|e\rangle\langle g| + e^{i\delta t}|g\rangle\langle e|)$$

where δ = ω - ω_{eg} is the **detuning**.

### Rabi Oscillations (Exact Two-Level)

With RWA, the coupled equations:
$$i\dot{c}_g = \frac{\Omega}{2}e^{i\delta t}c_e$$
$$i\dot{c}_e = \frac{\Omega}{2}e^{-i\delta t}c_g$$

**Solution** (starting in |g⟩):
$$|c_e(t)|^2 = \frac{\Omega^2}{\Omega^2 + \delta^2}\sin^2\left(\frac{\Omega_R t}{2}\right)$$

where the **generalized Rabi frequency**:
$$\boxed{\Omega_R = \sqrt{\Omega^2 + \delta^2}}$$

### On Resonance (δ = 0)

$$|c_e(t)|^2 = \sin^2\left(\frac{\Omega t}{2}\right)$$

Complete population transfer at t = π/Ω (π-pulse).

### Key Results

| Pulse Area | Effect |
|------------|--------|
| π/2-pulse | Creates superposition |
| π-pulse | Complete inversion |
| 2π-pulse | Full cycle, returns to initial |

---

## Quantum Computing Connection

### Single-Qubit Gates via Rabi Oscillations

**X Gate (NOT):** Apply π-pulse
$$|0\rangle \xrightarrow{\pi} |1\rangle, \quad |1\rangle \xrightarrow{\pi} |0\rangle$$

**Hadamard-like:** Apply π/2-pulse
$$|0\rangle \xrightarrow{\pi/2} \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle)$$

### Microwave Control in Superconducting Qubits

Transmon qubits use microwave pulses at ω ≈ ω_{01}:
$$H_{drive} = \Omega(t)\cos(\omega t + \phi)(a + a^\dagger)$$

Gate fidelity depends on:
- Pulse shaping (DRAG pulses)
- Detuning calibration
- T₁, T₂ coherence times

### Trapped Ion Gates

Raman transitions create effective two-level systems:
$$\Omega_{eff} = \frac{\Omega_1 \Omega_2^*}{2\Delta}$$

where Δ = detuning from intermediate state.

---

## Worked Examples

### Example 1: Transition Rate Calculation

**Problem:** A hydrogen atom is exposed to monochromatic light (ω = 2.5×10¹⁵ rad/s). Calculate the absorption rate for the 1s→2p transition.

**Solution:**

Transition frequency:
$$\omega_{2p,1s} = \frac{E_{2p} - E_{1s}}{\hbar} = \frac{13.6(1 - 1/4) \text{ eV}}{\hbar} = 1.55 \times 10^{16} \text{ rad/s}$$

The light is far from resonance (ω ≪ ω_{21}), so the transition rate is negligible.

For resonant light (ω = ω_{21}):
$$W = \frac{\pi|V_{21}|^2}{2\hbar^2}\delta(\omega_{21}-\omega)$$

With the matrix element |V_{21}| = eE₀|⟨2p|z|1s⟩| and using ⟨2p|z|1s⟩ = 0.74a₀:
$$|V_{21}| = 0.74 eE_0 a_0$$

### Example 2: π-Pulse Time

**Problem:** A superconducting qubit has ω_{01} = 5 GHz and coupling Ω = 100 MHz. Find the time for a π-pulse.

**Solution:**

On resonance, Ω_R = Ω = 100 MHz = 2π × 100 × 10⁶ rad/s.

For a π-pulse:
$$t_\pi = \frac{\pi}{\Omega} = \frac{\pi}{2\pi \times 100 \times 10^6} = 5 \text{ ns}$$

### Example 3: Detuned Rabi Oscillations

**Problem:** With Ω = 10 MHz and δ = 8 MHz detuning, find the maximum excitation probability.

**Solution:**

Generalized Rabi frequency:
$$\Omega_R = \sqrt{\Omega^2 + \delta^2} = \sqrt{100 + 64} \text{ MHz} = 12.8 \text{ MHz}$$

Maximum excitation:
$$P_{max} = \frac{\Omega^2}{\Omega_R^2} = \frac{100}{164} = 0.61$$

Only 61% excitation possible (detuning prevents full transfer).

---

## Practice Problems

### Problem Set 68.4

**Direct Application:**
1. Show that for E_f > E_i and ω > 0, the absorption term dominates when ω ≈ ω_{fi}.

2. Calculate the π-pulse time for a trapped ion qubit with Rabi frequency Ω/2π = 50 kHz.

3. If Ω = 20 MHz and δ = 15 MHz, what is:
   - The generalized Rabi frequency?
   - The period of population oscillations?
   - The maximum excited state population?

**Intermediate:**
4. Derive the transition probability including both resonance terms (no RWA). Show corrections are O(Ω/ω).

5. For a three-level Λ system, show that two-photon resonance occurs when ω₁ - ω₂ = ω_{13} even if neither field is resonant.

6. Calculate the power broadening of the absorption line when Ω ≈ γ (natural linewidth).

**Challenging:**
7. Derive the Bloch equations from the density matrix evolution under harmonic driving. Include phenomenological decay terms T₁ and T₂.

8. For a DRAG pulse Ω(t) = Ω₀e^{-(t/τ)²}(1 + iβ d/dt), show it reduces leakage to the |2⟩ state.

9. In the dressed state picture, find the energies of |+⟩ and |−⟩ as functions of detuning. Sketch the avoided crossing.

---

## Computational Lab

```python
"""
Day 473 Lab: Harmonic Perturbations and Rabi Oscillations
Simulates two-level dynamics under oscillating fields
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import expm

# Physical constants
hbar = 1.055e-34  # J·s

def rabi_probability_rwa(t, Omega, delta):
    """
    Excited state probability under RWA.

    Parameters:
    -----------
    t : array - time
    Omega : float - Rabi frequency (rad/s)
    delta : float - detuning (rad/s)

    Returns:
    --------
    P_e : array - probability in excited state
    """
    Omega_R = np.sqrt(Omega**2 + delta**2)
    return (Omega / Omega_R)**2 * np.sin(Omega_R * t / 2)**2

def two_level_equations(y, t, Omega, omega_eg, omega_drive):
    """
    Full two-level system equations (no RWA).

    y = [Re(c_g), Im(c_g), Re(c_e), Im(c_e)]
    """
    cg_r, cg_i, ce_r, ce_i = y
    cg = cg_r + 1j * cg_i
    ce = ce_r + 1j * ce_i

    # Full Hamiltonian (in rotating frame of ground state)
    V = Omega * np.cos(omega_drive * t)

    dcg = -1j * V / 2 * ce
    dce = -1j * V / 2 * cg - 1j * omega_eg * ce

    return [dcg.real, dcg.imag, dce.real, dce.imag]

def simulate_full_dynamics(t, Omega, omega_eg, omega_drive):
    """
    Solve full two-level dynamics without RWA.
    """
    y0 = [1, 0, 0, 0]  # Start in ground state
    solution = odeint(two_level_equations, y0, t,
                      args=(Omega, omega_eg, omega_drive))
    P_e = solution[:, 2]**2 + solution[:, 3]**2
    return P_e

# Visualization 1: Rabi oscillations at different detunings
print("=" * 60)
print("RABI OSCILLATIONS AT DIFFERENT DETUNINGS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

Omega = 1.0  # Rabi frequency (normalized)
detunings = [0, 0.5, 1.0, 2.0]  # In units of Omega
t = np.linspace(0, 6 * np.pi, 1000)

for ax, delta in zip(axes.flat, detunings):
    P_e = rabi_probability_rwa(t, Omega, delta * Omega)

    ax.plot(t / np.pi, P_e, 'b-', linewidth=2)
    ax.axhline(y=Omega**2 / (Omega**2 + (delta * Omega)**2),
               color='r', linestyle='--', alpha=0.7,
               label=f'$P_{{max}} = {Omega**2/(Omega**2 + (delta*Omega)**2):.2f}$')

    ax.set_xlabel('Time (units of π/Ω)', fontsize=12)
    ax.set_ylabel('$|c_e|^2$', fontsize=12)
    ax.set_title(f'δ/Ω = {delta}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.1)

plt.suptitle('Rabi Oscillations: Effect of Detuning', fontsize=14)
plt.tight_layout()
plt.savefig('rabi_oscillations.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 2: Resonance curve
print("\n" + "=" * 60)
print("RESONANCE LINE SHAPE")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

delta_range = np.linspace(-5, 5, 500)  # In units of Omega
t_values = [np.pi, 2*np.pi, 5*np.pi]  # Different pulse times

for t_pulse in t_values:
    P_max = (1 / (1 + (delta_range)**2)) * np.sin(
        np.sqrt(1 + delta_range**2) * t_pulse / 2)**2
    ax.plot(delta_range, P_max, linewidth=2,
            label=f't = {t_pulse/np.pi:.0f}π/Ω')

ax.set_xlabel('Detuning δ/Ω', fontsize=12)
ax.set_ylabel('Excited State Probability', fontsize=12)
ax.set_title('Resonance Line Shape at Different Times', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('resonance_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 3: RWA vs Full dynamics
print("\n" + "=" * 60)
print("RWA vs FULL DYNAMICS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

omega_eg = 10.0  # Transition frequency
Omega = 1.0  # Rabi frequency

# Case 1: Weak driving (RWA valid)
ax = axes[0]
t = np.linspace(0, 20 * np.pi, 2000)
omega_drive = omega_eg  # On resonance

P_rwa = rabi_probability_rwa(t, Omega, 0)
P_full = simulate_full_dynamics(t, Omega, omega_eg, omega_drive)

ax.plot(t / np.pi, P_rwa, 'b-', linewidth=2, label='RWA')
ax.plot(t / np.pi, P_full, 'r--', linewidth=1.5, alpha=0.7, label='Full')
ax.set_xlabel('Time (units of π/Ω)', fontsize=12)
ax.set_ylabel('$|c_e|^2$', fontsize=12)
ax.set_title(f'Weak Driving: Ω/ω = {Omega/omega_eg:.1f}', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)

# Case 2: Strong driving (RWA breaks down)
ax = axes[1]
Omega_strong = 3.0

P_rwa_strong = rabi_probability_rwa(t, Omega_strong, 0)
P_full_strong = simulate_full_dynamics(t, Omega_strong, omega_eg, omega_drive)

ax.plot(t / np.pi, P_rwa_strong, 'b-', linewidth=2, label='RWA')
ax.plot(t / np.pi, P_full_strong, 'r--', linewidth=1.5, alpha=0.7, label='Full')
ax.set_xlabel('Time (units of π/Ω)', fontsize=12)
ax.set_ylabel('$|c_e|^2$', fontsize=12)
ax.set_title(f'Strong Driving: Ω/ω = {Omega_strong/omega_eg:.1f}', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)

plt.tight_layout()
plt.savefig('rwa_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 4: Bloch sphere trajectory
print("\n" + "=" * 60)
print("BLOCH SPHERE DYNAMICS")
print("=" * 60)

from mpl_toolkits.mplot3d import Axes3D

def bloch_components(Omega, delta, t):
    """
    Calculate Bloch vector components during Rabi oscillation.
    """
    Omega_R = np.sqrt(Omega**2 + delta**2)
    sin_half = np.sin(Omega_R * t / 2)
    cos_half = np.cos(Omega_R * t / 2)

    # Bloch vector: (u, v, w) = (2Re(ρ_ge), 2Im(ρ_ge), ρ_ee - ρ_gg)
    u = (Omega / Omega_R) * np.sin(Omega_R * t) * (delta / Omega_R)
    v = (Omega / Omega_R) * np.sin(Omega_R * t)
    w = -1 + 2 * (Omega / Omega_R)**2 * sin_half**2

    return u, v, w

fig = plt.figure(figsize=(12, 5))

# π/2 pulse trajectory
ax1 = fig.add_subplot(121, projection='3d')
t_pi2 = np.linspace(0, np.pi / 2, 100)
u, v, w = bloch_components(1.0, 0, t_pi2)

# Draw Bloch sphere
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
x_sphere = np.sin(theta) * np.cos(phi)
y_sphere = np.sin(theta) * np.sin(phi)
z_sphere = np.cos(theta)
ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

ax1.plot(u, v, w, 'b-', linewidth=3, label='π/2 pulse')
ax1.scatter([0], [0], [-1], color='g', s=100, label='Start |0⟩')
ax1.scatter([u[-1]], [v[-1]], [w[-1]], color='r', s=100, label='End')

ax1.set_xlabel('u')
ax1.set_ylabel('v')
ax1.set_zlabel('w')
ax1.set_title('π/2 Pulse Trajectory', fontsize=14)
ax1.legend()

# π pulse trajectory
ax2 = fig.add_subplot(122, projection='3d')
t_pi = np.linspace(0, np.pi, 100)
u, v, w = bloch_components(1.0, 0, t_pi)

ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
ax2.plot(u, v, w, 'b-', linewidth=3, label='π pulse')
ax2.scatter([0], [0], [-1], color='g', s=100, label='Start |0⟩')
ax2.scatter([u[-1]], [v[-1]], [w[-1]], color='r', s=100, label='End |1⟩')

ax2.set_xlabel('u')
ax2.set_ylabel('v')
ax2.set_zlabel('w')
ax2.set_title('π Pulse Trajectory', fontsize=14)
ax2.legend()

plt.tight_layout()
plt.savefig('bloch_trajectory.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary table
print("\n" + "=" * 60)
print("PULSE CALIBRATION SUMMARY")
print("=" * 60)

pulse_types = ['π/2', 'π', '3π/2', '2π']
angles = [np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]

print(f"{'Pulse Type':<12} {'Time (t·Ω)':<15} {'Final State':<20}")
print("-" * 47)

for name, angle in zip(pulse_types, angles):
    if angle == np.pi/2:
        final = "(|0⟩ - i|1⟩)/√2"
    elif angle == np.pi:
        final = "-i|1⟩"
    elif angle == 3*np.pi/2:
        final = "(-|0⟩ - i|1⟩)/√2"
    else:
        final = "-|0⟩"
    print(f"{name:<12} {angle:.4f}{'':8} {final:<20}")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("""
1. Harmonic perturbations create two resonance conditions
2. Absorption: ℏω = E_f - E_i (ω = ω_fi)
3. Emission: ℏω = E_i - E_f (ω = -ω_fi)
4. RWA valid when Ω ≪ ω (weak driving)
5. Detuning reduces maximum population transfer
6. Pulse area determines qubit rotation angle
""")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Harmonic perturbation | $V(t) = V_0\cos(\omega t)$ |
| Transition rate | $W = \frac{\pi\|V_{fi}\|^2}{2\hbar^2}[\delta(\omega_{fi}-\omega) + \delta(\omega_{fi}+\omega)]$ |
| Rabi frequency | $\Omega = V_{fi}/\hbar$ |
| Generalized Rabi | $\Omega_R = \sqrt{\Omega^2 + \delta^2}$ |
| Excitation probability | $P_e = \frac{\Omega^2}{\Omega_R^2}\sin^2(\Omega_R t/2)$ |
| π-pulse time | $t_\pi = \pi/\Omega$ |

### Main Takeaways

1. **Harmonic perturbations** produce absorption and stimulated emission resonances
2. **Rotating wave approximation** simplifies near-resonance dynamics
3. **Rabi oscillations** give complete population transfer on resonance
4. **Detuning** reduces maximum excitation and speeds oscillations
5. **Pulse area** (Ωt) controls qubit rotation angle

---

## Daily Checklist

- [ ] I can solve for transition amplitudes under harmonic perturbation
- [ ] I understand the RWA and when it's valid
- [ ] I can calculate Rabi oscillation frequency and amplitude
- [ ] I know how detuning affects population dynamics
- [ ] I can relate pulse areas to qubit gates

---

## Preview: Day 474

Tomorrow we study **selection rules** — the symmetry constraints that determine which transitions are allowed.

---

**Next:** [Day_474_Friday.md](Day_474_Friday.md) — Selection Rules
