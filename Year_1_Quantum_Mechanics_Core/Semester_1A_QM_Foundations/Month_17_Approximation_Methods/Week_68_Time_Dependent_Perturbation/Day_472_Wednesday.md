# Day 472: Fermi's Golden Rule

## Overview
**Day 472** | Year 1, Month 17, Week 68 | Transition Rates to Continuum

Today we derive Fermi's Golden Rule—one of the most important results in quantum mechanics, governing transition rates to a continuum of states.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Golden Rule derivation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Applications and examples |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Decay rate simulations |

---

## Learning Objectives

By the end of today, you will be able to:
1. Derive Fermi's Golden Rule from first-order perturbation theory
2. Understand the role of the density of states
3. Calculate decay rates for various systems
4. Apply the golden rule to atomic transitions
5. Connect to spontaneous emission rates
6. Understand the validity conditions for the golden rule

---

## Core Content

### From Discrete to Continuum

In Day 471, we found for a constant perturbation:
$$P_{i \to f}(t) = \frac{4|V_{fi}|^2}{\hbar^2\omega_{fi}^2}\sin^2\left(\frac{\omega_{fi}t}{2}\right)$$

### The Sinc Function Limit

Using the identity:
$$\lim_{t \to \infty} \frac{\sin^2(\omega t/2)}{\omega^2} = \frac{\pi t}{2}\delta(\omega)$$

We get:
$$P_{i \to f}(t) = \frac{2\pi}{\hbar}|V_{fi}|^2 \delta(E_f - E_i) \cdot t$$

### Transition Rate

The transition rate (probability per unit time) to a single final state:
$$W_{i \to f} = \frac{dP_{i \to f}}{dt} = \frac{2\pi}{\hbar}|V_{fi}|^2 \delta(E_f - E_i)$$

### Density of States

When transitions occur to a **continuum** of final states, we sum over them:
$$W_{i \to f} = \frac{2\pi}{\hbar}|V_{fi}|^2 \rho(E_f)$$

where $\rho(E)$ is the **density of states**: number of states per unit energy.

### Fermi's Golden Rule

$$\boxed{W_{i \to f} = \frac{2\pi}{\hbar}|\langle f|V|i\rangle|^2 \rho(E_f)}$$

**Key insight:** The transition rate depends on:
1. Matrix element squared (coupling strength)
2. Density of final states (phase space availability)
3. Energy conservation (δ-function built into ρ)

### Derivation Details

Starting from:
$$c_f^{(1)}(t) = -\frac{V_{fi}}{\hbar\omega_{fi}}(e^{i\omega_{fi}t} - 1)$$

The probability:
$$|c_f^{(1)}(t)|^2 = \frac{|V_{fi}|^2}{\hbar^2\omega_{fi}^2}|e^{i\omega_{fi}t} - 1|^2$$

$$= \frac{|V_{fi}|^2}{\hbar^2\omega_{fi}^2}(2 - 2\cos\omega_{fi}t) = \frac{4|V_{fi}|^2}{\hbar^2\omega_{fi}^2}\sin^2\frac{\omega_{fi}t}{2}$$

Define:
$$f_t(\omega) = \frac{\sin^2(\omega t/2)}{\omega^2}$$

This function:
- Peaks at ω = 0 with height t²/4
- Has width ~ 2π/t
- Area = πt/2
- Approaches (πt/2)δ(ω) as t → ∞

### Summing Over Final States

For a continuum:
$$W_{\text{total}} = \sum_f W_{i \to f} \to \int W_{i \to f} \, \rho(E_f) \, dE_f$$

$$W_{\text{total}} = \frac{2\pi}{\hbar}\int |V_{fi}|^2 \delta(E_f - E_i) \rho(E_f) \, dE_f$$

$$\boxed{W_{\text{total}} = \frac{2\pi}{\hbar}|V_{fi}|^2_{E_f = E_i} \rho(E_i)}$$

### Validity Conditions

Fermi's Golden Rule is valid when:
1. **Weak perturbation:** |V| ≪ |E_n - E_m|
2. **Long time:** t ≫ ℏ/ΔE (energy resolution)
3. **Short time:** P ≪ 1 (perturbation theory valid)
4. **Dense spectrum:** spacing ≪ ℏ/t

This gives a "window" of validity:
$$\frac{\hbar}{\Delta E} \ll t \ll \frac{\hbar}{|V|}$$

---

## Density of States Examples

### Free Particle in 3D

For a particle in a box of volume V:
$$\rho(E) = \frac{V}{2\pi^2}\left(\frac{2m}{\hbar^2}\right)^{3/2}\sqrt{E}$$

### Photon Density of States

For photons in volume V:
$$\rho(\omega) = \frac{V\omega^2}{\pi^2 c^3}$$

This gives the number of modes per unit frequency.

### 1D, 2D, 3D Comparison

| Dimension | ρ(E) |
|-----------|------|
| 1D | ρ(E) ∝ E^(-1/2) |
| 2D | ρ(E) = constant |
| 3D | ρ(E) ∝ E^(1/2) |

---

## Quantum Computing Connection

### Qubit Decay Rates

T₁ relaxation in qubits follows Fermi's Golden Rule:
$$\frac{1}{T_1} = \frac{2\pi}{\hbar}|\langle 0|V|1\rangle|^2 \rho(E_{01})$$

**Engineering longer T₁:**
- Reduce |V| (isolate from environment)
- Reduce ρ(E) (gap engineering, 3D cavity)
- Choose matrix elements wisely

### Purcell Effect

In circuit QED, coupling to a resonator modifies the density of states:
$$\kappa_{\text{Purcell}} = \frac{g^2}{\Delta^2}\kappa$$

where g = coupling, Δ = detuning, κ = cavity decay.

### State Readout

Dispersive readout relies on state-dependent transition rates:
- |0⟩ and |1⟩ states shift cavity frequency differently
- Fermi's Golden Rule determines photon emission rates

---

## Worked Examples

### Example 1: Ionization by Constant Electric Field

**Problem:** An atom in ground state |1s⟩ is exposed to a weak constant electric field. Estimate the ionization rate.

**Solution:**

The perturbation:
$$V = eEz$$

Final states are free particle states |**k**⟩ with:
$$\rho(E) = \frac{V}{2\pi^2}\left(\frac{2m}{\hbar^2}\right)^{3/2}\sqrt{E}$$

The matrix element (for hydrogen):
$$\langle \mathbf{k}|z|1s\rangle = \int \psi^*_k(\mathbf{r}) \, z \, \psi_{1s}(\mathbf{r}) \, d^3r$$

For k ≈ 0 (threshold):
$$|\langle \mathbf{k}|z|1s\rangle|^2 \sim a_0^3$$

The ionization rate:
$$W = \frac{2\pi}{\hbar}(eE)^2 a_0^3 \cdot \frac{V}{2\pi^2}\left(\frac{2m}{\hbar^2}\right)^{3/2}\sqrt{E_i}$$

### Example 2: Nuclear Beta Decay

**Problem:** Apply Fermi's Golden Rule to β-decay: n → p + e⁻ + ν̄ₑ

**Solution:**

The weak interaction matrix element:
$$|V_{fi}|^2 = G_F^2 |M_{nuclear}|^2$$

where G_F = Fermi coupling constant.

The phase space (density of states for electron + neutrino):
$$\rho(E_e) \propto p_e^2 (E_0 - E_e)^2$$

where E₀ = endpoint energy.

The decay rate:
$$W = \frac{G_F^2}{2\pi^3\hbar^7 c^6}|M|^2 \int_0^{E_0} p_e^2(E_0 - E_e)^2 \, dE_e$$

This gives the famous Fermi theory of beta decay!

### Example 3: Photoelectric Effect

**Problem:** Calculate the photoionization cross-section for hydrogen 1s.

**Solution:**

The photon perturbation (dipole approximation):
$$V = \frac{e}{m}\mathbf{A} \cdot \mathbf{p} = \frac{eA_0}{m}p_z e^{-i\omega t}$$

Matrix element:
$$V_{fi} = \frac{eA_0}{m}\langle \mathbf{k}|p_z|1s\rangle$$

Using ⟨f|p|i⟩ = im⟨f|[H₀, z]|i⟩/ℏ = imω_{fi}⟨f|z|i⟩:
$$|V_{fi}|^2 = e^2 A_0^2 \omega^2 |\langle \mathbf{k}|z|1s\rangle|^2$$

The transition rate:
$$W = \frac{2\pi}{\hbar}e^2 A_0^2 \omega^2 |\langle \mathbf{k}|z|1s\rangle|^2 \rho(E_f)$$

The cross-section:
$$\sigma = \frac{\hbar\omega \cdot W}{I} = \frac{4\pi^2 \alpha \omega}{c}|\langle \mathbf{k}|z|1s\rangle|^2 \rho(E_f)$$

For hydrogen:
$$\sigma_{1s} = \frac{2^9 \pi^2}{3} \alpha a_0^2 \left(\frac{E_1}{\hbar\omega}\right)^{7/2}$$

---

## Practice Problems

### Problem Set 68.3

**Direct Application:**
1. Show that $\int_{-\infty}^{\infty} f_t(\omega)\,d\omega = \pi t/2$ where $f_t(\omega) = \sin^2(\omega t/2)/\omega^2$.

2. Calculate the density of states ρ(E) for:
   - Electrons in a 2D quantum well
   - Phonons in a Debye solid

3. A perturbation couples states |i⟩ and |f⟩ with matrix element V = 0.1 eV. If ρ(E_f) = 0.5 states/eV, find the transition rate in s⁻¹.

**Intermediate:**
4. For a hydrogen atom in the n = 2 state exposed to blackbody radiation, estimate which transitions dominate and why.

5. Derive the density of states for a relativistic particle (E² = p²c² + m²c⁴) in 3D.

6. Show that for beta decay, the electron spectrum peaks before the endpoint energy E₀.

**Challenging:**
7. Calculate the photoionization cross-section for hydrogen 2p states. Compare to 1s.

8. In superconducting qubits, the quasiparticle tunneling rate follows Fermi's Golden Rule. If the gap is Δ = 200 μeV and the quasiparticle density is nqp, derive the dependence of T₁ on temperature.

9. Consider a two-level system coupled to a bath of harmonic oscillators (spin-boson model). Derive the relaxation rate using Fermi's Golden Rule. Under what conditions does the golden rule break down?

---

## Computational Lab

```python
"""
Day 472 Lab: Fermi's Golden Rule Visualization
Demonstrates transition probability time evolution and δ-function formation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Physical constants
hbar = 1.055e-34  # J·s
eV = 1.602e-19    # J

def sinc_squared(omega, t):
    """
    The function f_t(ω) = sin²(ωt/2)/ω²
    Approaches (πt/2)δ(ω) as t → ∞
    """
    if np.isscalar(omega):
        if abs(omega) < 1e-10:
            return t**2 / 4
        else:
            return np.sin(omega * t / 2)**2 / omega**2
    else:
        result = np.zeros_like(omega)
        small = np.abs(omega) < 1e-10
        result[small] = t**2 / 4
        result[~small] = np.sin(omega[~small] * t / 2)**2 / omega[~small]**2
        return result

def transition_probability(V_fi, omega_fi, t):
    """
    P(t) = (4|V_fi|²/ℏ²ω²)sin²(ω t/2)

    Parameters:
    -----------
    V_fi : float - matrix element (eV)
    omega_fi : float - transition frequency (rad/s)
    t : float or array - time (s)
    """
    V_fi_J = V_fi * eV
    if abs(omega_fi) < 1e-10:
        return (V_fi_J * t / hbar)**2
    return 4 * V_fi_J**2 / (hbar**2 * omega_fi**2) * np.sin(omega_fi * t / 2)**2

def fermi_golden_rule_rate(V_fi, rho_Ef):
    """
    W = (2π/ℏ)|V_fi|² ρ(E_f)

    Parameters:
    -----------
    V_fi : float - matrix element (eV)
    rho_Ef : float - density of states (1/eV)

    Returns:
    --------
    W : float - transition rate (1/s)
    """
    V_fi_J = V_fi * eV
    rho_J = rho_Ef / eV
    return 2 * np.pi / hbar * V_fi_J**2 * rho_J

# Visualization 1: δ-function formation
print("=" * 60)
print("FERMI'S GOLDEN RULE: δ-FUNCTION FORMATION")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

omega = np.linspace(-10, 10, 1000)
times = [1, 5, 20, 100]

for ax, t in zip(axes.flat, times):
    f_t = sinc_squared(omega, t)

    # Normalize for comparison
    area, _ = quad(lambda w: sinc_squared(w, t), -100, 100)
    f_normalized = f_t / area * np.pi / 2  # Normalize to height 1 at peak

    ax.plot(omega, f_t, 'b-', linewidth=2, label=f't = {t}')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('ω', fontsize=12)
    ax.set_ylabel(f'$f_t(ω) = sin²(ωt/2)/ω²$', fontsize=11)
    ax.set_title(f't = {t} (Area ≈ {area:.2f}, πt/2 = {np.pi*t/2:.2f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 10)

plt.suptitle('Evolution toward δ-function: $f_t(ω) → (πt/2)δ(ω)$', fontsize=14)
plt.tight_layout()
plt.savefig('fermi_delta_formation.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 2: Transition probability vs time
print("\n" + "=" * 60)
print("TRANSITION PROBABILITY TIME EVOLUTION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Case 1: Resonant (ω = 0)
t = np.linspace(0, 100, 1000)
V_values = [0.01, 0.02, 0.05]  # eV

ax = axes[0]
for V in V_values:
    P = (V * eV * t / hbar)**2  # Resonant case
    ax.plot(t, P, linewidth=2, label=f'V = {V*1000:.0f} meV')

ax.set_xlabel('Time (arbitrary units)', fontsize=12)
ax.set_ylabel('Transition Probability P(t)', fontsize=12)
ax.set_title('Resonant Case (E_f = E_i)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Case 2: Off-resonant with oscillations
ax = axes[1]
omega_values = [0.5, 1.0, 2.0]
V = 0.01  # eV

for omega in omega_values:
    P = transition_probability(V, omega, t)
    ax.plot(t, P, linewidth=2, label=f'ω = {omega} rad/s')

ax.set_xlabel('Time (arbitrary units)', fontsize=12)
ax.set_ylabel('Transition Probability P(t)', fontsize=12)
ax.set_title(f'Off-Resonant Case (V = {V*1000:.0f} meV)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transition_probability.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 3: Energy spectrum and density of states
print("\n" + "=" * 60)
print("DENSITY OF STATES IN DIFFERENT DIMENSIONS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

E = np.linspace(0.01, 5, 1000)

# Different dimensional densities of states
rho_1D = 1 / np.sqrt(E)
rho_2D = np.ones_like(E)
rho_3D = np.sqrt(E)

ax.plot(E, rho_1D / rho_1D.max(), 'b-', linewidth=2, label='1D: ρ(E) ∝ E^(-1/2)')
ax.plot(E, rho_2D / rho_2D.max(), 'g-', linewidth=2, label='2D: ρ(E) = const')
ax.plot(E, rho_3D / rho_3D.max(), 'r-', linewidth=2, label='3D: ρ(E) ∝ E^(1/2)')

ax.set_xlabel('Energy E', fontsize=12)
ax.set_ylabel('Density of States ρ(E) (normalized)', fontsize=12)
ax.set_title('Density of States vs Dimension', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.5)

plt.tight_layout()
plt.savefig('density_of_states.png', dpi=150, bbox_inches='tight')
plt.show()

# Numerical calculation example
print("\n" + "=" * 60)
print("NUMERICAL EXAMPLE: QUBIT DECAY RATE")
print("=" * 60)

# Superconducting qubit parameters
V_coupling = 0.001  # eV (1 meV coupling to environment)
rho_env = 10  # states/eV (environmental density of states)

W = fermi_golden_rule_rate(V_coupling, rho_env)
T1 = 1 / W

print(f"Coupling matrix element: |V| = {V_coupling*1000:.1f} meV")
print(f"Environmental DOS: ρ(E) = {rho_env} states/eV")
print(f"Decay rate: W = {W:.2e} s⁻¹")
print(f"T₁ relaxation time: T₁ = {T1*1e6:.2f} μs")

# Phase space factor visualization
print("\n" + "=" * 60)
print("BETA DECAY SPECTRUM (PHASE SPACE)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

E0 = 1.0  # Endpoint energy (normalized)
E_e = np.linspace(0, E0, 1000)

# Electron spectrum: ∝ p_e² (E_0 - E_e)²
# Non-relativistic: p ∝ √E
spectrum = np.sqrt(E_e) * E_e * (E0 - E_e)**2

# Include Fermi function effect (simplified)
Z = 1  # Daughter nucleus charge
# Coulomb correction (simplified)
eta = 0.01 / np.sqrt(E_e + 0.001)  # Avoid division by zero
F_coulomb = 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))
spectrum_corrected = spectrum * F_coulomb

ax.plot(E_e, spectrum / spectrum.max(), 'b-', linewidth=2,
        label='Uncorrected: $p²(E_0-E)²$')
ax.plot(E_e, spectrum_corrected / spectrum_corrected.max(), 'r--', linewidth=2,
        label='With Coulomb correction')

ax.axvline(x=E0, color='k', linestyle=':', label=f'Endpoint E₀ = {E0}')
ax.set_xlabel('Electron Energy E', fontsize=12)
ax.set_ylabel('Electron Spectrum (normalized)', fontsize=12)
ax.set_title('Beta Decay Electron Spectrum from Fermi\'s Golden Rule', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig('beta_decay_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("KEY INSIGHTS FROM FERMI'S GOLDEN RULE")
print("=" * 60)
print("""
1. The δ-function enforces energy conservation
2. Transition rate ∝ |V|² (coupling strength squared)
3. Transition rate ∝ ρ(E) (phase space availability)
4. Valid for weak perturbations and intermediate times
5. Foundation for all decay and scattering calculations
""")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Fermi's Golden Rule | $W = \frac{2\pi}{\hbar}\|\langle f\|V\|i\rangle\|^2 \rho(E_f)$ |
| δ-function formation | $\lim_{t \to \infty} \frac{\sin^2(\omega t/2)}{\omega^2} = \frac{\pi t}{2}\delta(\omega)$ |
| 3D particle DOS | $\rho(E) = \frac{V}{2\pi^2}\left(\frac{2m}{\hbar^2}\right)^{3/2}\sqrt{E}$ |
| Photon DOS | $\rho(\omega) = \frac{V\omega^2}{\pi^2 c^3}$ |
| Validity window | $\frac{\hbar}{\Delta E} \ll t \ll \frac{\hbar}{\|V\|}$ |

### Main Takeaways

1. **Fermi's Golden Rule** gives transition rates to continua
2. **Two factors:** coupling strength and density of states
3. **Energy conservation** emerges from the δ-function
4. **Universal applicability:** atomic transitions, nuclear decay, qubit relaxation
5. **Validity conditions** must be carefully checked

---

## Daily Checklist

- [ ] I can derive Fermi's Golden Rule from first-order perturbation theory
- [ ] I understand how the δ-function emerges in the long-time limit
- [ ] I can calculate density of states for various systems
- [ ] I can apply the golden rule to physical problems
- [ ] I understand the validity conditions

---

## Preview: Day 473

Tomorrow we study **harmonic perturbations** V(t) = V cos(ωt), which model oscillating electromagnetic fields and lead to resonance absorption.

---

**Next:** [Day_473_Thursday.md](Day_473_Thursday.md) — Harmonic Perturbations
