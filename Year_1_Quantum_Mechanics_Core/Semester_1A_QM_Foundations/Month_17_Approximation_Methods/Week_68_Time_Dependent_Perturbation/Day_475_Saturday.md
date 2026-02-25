# Day 475: Spontaneous Emission

## Overview
**Day 475** | Year 1, Month 17, Week 68 | Quantum Theory of Radiation

Today we derive spontaneous emission rates using Fermi's Golden Rule with the quantized electromagnetic field—a cornerstone result bridging quantum mechanics and quantum electrodynamics.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Einstein A coefficient derivation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Lifetime calculations and oscillator strengths |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Atomic lifetime simulation |

---

## Learning Objectives

By the end of today, you will be able to:
1. Derive the spontaneous emission rate using Fermi's Golden Rule
2. Calculate Einstein A coefficients for atomic transitions
3. Understand the role of vacuum fluctuations
4. Compute oscillator strengths and their sum rule
5. Calculate atomic lifetimes from first principles
6. Connect to qubit T₁ relaxation mechanisms

---

## Core Content

### The Puzzle of Spontaneous Emission

**Classical problem:** A stationary excited atom has no time-dependent potential—why does it emit?

**Quantum answer:** The electromagnetic field has zero-point fluctuations that stimulate emission.

### Quantized Electromagnetic Field

In second quantization, the electric field:
$$\hat{\mathbf{E}} = i\sum_{\mathbf{k},\lambda}\sqrt{\frac{\hbar\omega_k}{2\epsilon_0 V}}\left(\hat{a}_{\mathbf{k}\lambda}\boldsymbol{\epsilon}_{\mathbf{k}\lambda}e^{i\mathbf{k}\cdot\mathbf{r}} - \text{h.c.}\right)$$

where:
- $\hat{a}_{\mathbf{k}\lambda}$, $\hat{a}^\dagger_{\mathbf{k}\lambda}$: photon annihilation/creation operators
- $\boldsymbol{\epsilon}_{\mathbf{k}\lambda}$: polarization vectors (λ = 1, 2)

### Vacuum State

The electromagnetic vacuum |0⟩ has:
$$\langle 0|\hat{a}_{\mathbf{k}\lambda}|0\rangle = 0$$
$$\langle 0|\hat{E}^2|0\rangle = \sum_{\mathbf{k},\lambda}\frac{\hbar\omega_k}{2\epsilon_0 V} \neq 0 \quad \text{(zero-point fluctuations)}$$

### Atom-Field Interaction

The dipole interaction:
$$\hat{V} = -\hat{\mathbf{d}} \cdot \hat{\mathbf{E}} = -e\hat{\mathbf{r}} \cdot \hat{\mathbf{E}}$$

### Spontaneous Emission: Initial and Final States

**Initial:** Atom in |e⟩, field in vacuum |0⟩
$$|i\rangle = |e\rangle \otimes |0\rangle$$

**Final:** Atom in |g⟩, one photon in mode (k, λ)
$$|f\rangle = |g\rangle \otimes |1_{\mathbf{k}\lambda}\rangle$$

### Matrix Element

$$\langle f|\hat{V}|i\rangle = -e\langle g|\hat{\mathbf{r}}|e\rangle \cdot \langle 1_{\mathbf{k}\lambda}|\hat{\mathbf{E}}|0\rangle$$

The field matrix element:
$$\langle 1_{\mathbf{k}\lambda}|\hat{\mathbf{E}}|0\rangle = i\sqrt{\frac{\hbar\omega_k}{2\epsilon_0 V}}\boldsymbol{\epsilon}_{\mathbf{k}\lambda}$$

(in dipole approximation: $e^{i\mathbf{k}\cdot\mathbf{r}} \approx 1$)

### Applying Fermi's Golden Rule

Transition rate to mode (k, λ):
$$W_{\mathbf{k}\lambda} = \frac{2\pi}{\hbar}|\langle f|\hat{V}|i\rangle|^2 \delta(E_e - E_g - \hbar\omega_k)$$

$$= \frac{2\pi}{\hbar} \cdot \frac{e^2\hbar\omega}{2\epsilon_0 V}|\mathbf{d}_{eg} \cdot \boldsymbol{\epsilon}_{\mathbf{k}\lambda}|^2 \delta(\hbar\omega_{eg} - \hbar\omega_k)$$

### Sum Over Final States

Total rate (sum over all k, λ):
$$A = \sum_{\mathbf{k},\lambda} W_{\mathbf{k}\lambda}$$

Converting sum to integral:
$$\sum_{\mathbf{k}} \to \frac{V}{(2\pi)^3}\int d^3k = \frac{V}{(2\pi)^3}\int_0^\infty k^2 dk \int d\Omega_k$$

Using $k = \omega/c$:
$$= \frac{V}{(2\pi c)^3}\int_0^\infty \omega^2 d\omega \int d\Omega_k$$

### Polarization Sum

For each direction k, summing over two polarizations:
$$\sum_{\lambda=1,2}|\mathbf{d}_{eg} \cdot \boldsymbol{\epsilon}_{\mathbf{k}\lambda}|^2 = |d_{eg}|^2 - |\mathbf{d}_{eg} \cdot \hat{\mathbf{k}}|^2 = |d_{eg}|^2\sin^2\theta$$

Angular integration:
$$\int \sin^2\theta \, d\Omega = \frac{8\pi}{3}$$

### Einstein A Coefficient

Combining everything:

$$A = \frac{e^2\omega_{eg}}{2\pi\epsilon_0\hbar c^3} \cdot \frac{\omega_{eg}^2}{2\pi^2} \cdot \frac{8\pi}{3}|d_{eg}|^2$$

$$\boxed{A = \frac{\omega_{eg}^3}{3\pi\epsilon_0\hbar c^3}|\mathbf{d}_{eg}|^2 = \frac{e^2\omega_{eg}^3}{3\pi\epsilon_0\hbar c^3}|\langle g|\mathbf{r}|e\rangle|^2}$$

Or in terms of the fine structure constant:
$$\boxed{A = \frac{4\alpha\omega_{eg}^3}{3c^2}|\langle g|\mathbf{r}|e\rangle|^2}$$

### Natural Lifetime

The excited state lifetime:
$$\boxed{\tau = \frac{1}{A} = \frac{3\pi\epsilon_0\hbar c^3}{\omega_{eg}^3|\mathbf{d}_{eg}|^2}}$$

---

## Oscillator Strength

### Definition

The **oscillator strength** is a dimensionless measure of transition probability:
$$\boxed{f_{eg} = \frac{2m_e\omega_{eg}}{3\hbar}|\langle g|\mathbf{r}|e\rangle|^2}$$

### Relation to A Coefficient

$$A = \frac{e^2\omega_{eg}^2}{2\pi\epsilon_0 m_e c^3}f_{eg}$$

### Thomas-Reiche-Kuhn Sum Rule

$$\boxed{\sum_n f_{gn} = Z}$$

where Z = number of electrons. This is a consequence of the commutator [x, p] = iℏ.

---

## Applications

### Hydrogen 2p → 1s Transition

Matrix element:
$$|\langle 1s|r|2p\rangle| = \frac{2^7}{3^5}a_0 \approx 0.74a_0$$

Transition energy:
$$\hbar\omega = 10.2 \text{ eV}$$

Einstein A coefficient:
$$A = \frac{4\alpha \omega^3}{3c^2}(0.74a_0)^2 \approx 6.3 \times 10^8 \text{ s}^{-1}$$

Natural lifetime:
$$\tau = 1/A \approx 1.6 \text{ ns}$$

### Natural Linewidth

The uncertainty principle gives:
$$\Delta E \cdot \tau \geq \hbar/2$$

The natural linewidth:
$$\Gamma = \hbar A = \frac{\hbar}{\tau}$$

For hydrogen 2p: Γ ≈ 100 MHz

---

## Quantum Computing Connection

### Qubit T₁ Relaxation

Spontaneous emission to the environment causes T₁ decay:
$$\frac{1}{T_1} = A_{eff} = \frac{4\alpha\omega_{01}^3}{3c^2}|\langle 0|\hat{d}|1\rangle|^2 \times F$$

where F accounts for the modified density of states (cavity/circuit effects).

### Purcell Enhancement

In a resonant cavity of volume V and quality factor Q:
$$F_P = \frac{3Q\lambda^3}{4\pi^2 V}$$

Enhanced emission rate:
$$A_{cavity} = F_P \cdot A_{free}$$

### Purcell Suppression

For qubits, we want **long T₁** (suppressed emission):
- Detune qubit from cavity: Δ ≫ g
- Use 3D cavities: large V, smaller F_P

### Circuit QED Design

Transmon T₁ limited by:
- Dielectric loss
- Quasiparticle tunneling
- Purcell decay through readout resonator

Typical values: T₁ ~ 50-500 μs

---

## Worked Examples

### Example 1: Hydrogen Lifetime

**Problem:** Calculate the spontaneous emission lifetime for the 2p → 1s transition in hydrogen.

**Solution:**

Transition frequency:
$$\omega = \frac{E_{2p} - E_{1s}}{\hbar} = \frac{10.2 \text{ eV}}{6.58 \times 10^{-16} \text{ eV·s}} = 1.55 \times 10^{16} \text{ rad/s}$$

Matrix element (from tables):
$$|\langle 1s|z|2p,m=0\rangle| = \frac{2^7}{3^5}\sqrt{\frac{1}{3}}a_0 = 0.43a_0$$

For all m states averaged:
$$\overline{|\langle 1s|\mathbf{r}|2p\rangle|^2} = (0.74a_0)^2$$

Einstein A coefficient:
$$A = \frac{4\alpha\omega^3}{3c^2}(0.74a_0)^2$$

$$= \frac{4 \times (1/137) \times (1.55 \times 10^{16})^3}{3 \times (3 \times 10^8)^2}(0.74 \times 5.29 \times 10^{-11})^2$$

$$= 6.27 \times 10^8 \text{ s}^{-1}$$

Lifetime:
$$\tau = 1/A = 1.6 \text{ ns}$$

### Example 2: Scaling with Frequency

**Problem:** How does the spontaneous emission rate scale with transition frequency for fixed matrix element?

**Solution:**

From the formula:
$$A \propto \omega^3$$

Higher frequency transitions decay faster. This is why:
- X-ray transitions: τ ~ fs
- Optical transitions: τ ~ ns
- Microwave transitions: τ ~ ms to s

For superconducting qubits (ω ~ 5 GHz):
$$\frac{A_{qubit}}{A_{optical}} \sim \left(\frac{5 \times 10^9}{5 \times 10^{14}}\right)^3 = 10^{-15}$$

This is why microwave qubits have long intrinsic T₁!

### Example 3: Purcell Effect

**Problem:** A qubit with ω/2π = 5 GHz is coupled to a cavity with Q = 10⁴ and V = (1 cm)³. Calculate the Purcell factor.

**Solution:**

Wavelength:
$$\lambda = c/\nu = (3 \times 10^8)/(5 \times 10^9) = 6 \text{ cm}$$

Purcell factor:
$$F_P = \frac{3Q\lambda^3}{4\pi^2 V} = \frac{3 \times 10^4 \times (0.06)^3}{4\pi^2 \times 10^{-6}}$$

$$= \frac{3 \times 10^4 \times 2.16 \times 10^{-4}}{3.95 \times 10^{-5}} = 164$$

The emission rate is enhanced by ~160× in the cavity!

---

## Practice Problems

### Problem Set 68.6

**Direct Application:**
1. Calculate the oscillator strength for the hydrogen 2p → 1s transition using f = 2mω|⟨r⟩|²/3ℏ.

2. The sodium D line (3p → 3s) has λ = 589 nm and f = 0.98. Calculate the Einstein A coefficient and lifetime.

3. Verify the Thomas-Reiche-Kuhn sum rule for a harmonic oscillator (sum f_{0n} over all n).

**Intermediate:**
4. Derive the relation between Einstein A and B coefficients using detailed balance: A/B = ℏω³/π²c³.

5. Calculate the natural linewidth of the hydrogen 2p state in MHz. Compare to Doppler broadening at room temperature.

6. For a two-level atom in a cavity with Purcell factor F_P = 100, how does the effective lifetime change?

**Challenging:**
7. Derive the spontaneous emission rate including both electric dipole and magnetic dipole contributions. Show M1 is suppressed by α².

8. In circuit QED, the transmon couples to the readout resonator with g/2π = 100 MHz. If the resonator has κ/2π = 1 MHz and the qubit-resonator detuning is Δ/2π = 1 GHz, calculate the Purcell-limited T₁.

9. Derive the modification to spontaneous emission near a perfectly conducting mirror at distance d from the atom. Show the rate oscillates with d.

---

## Computational Lab

```python
"""
Day 475 Lab: Spontaneous Emission and Atomic Lifetimes
Calculates Einstein A coefficients and simulates decay dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import quad

# Physical constants
hbar = 1.055e-34  # J·s
c = 3e8  # m/s
e = 1.602e-19  # C
m_e = 9.109e-31  # kg
epsilon_0 = 8.854e-12  # F/m
a_0 = 5.29e-11  # m (Bohr radius)
alpha = 1/137  # Fine structure constant
eV = 1.602e-19  # J

def einstein_A(omega, d_squared):
    """
    Calculate Einstein A coefficient.

    Parameters:
    -----------
    omega : float - transition frequency (rad/s)
    d_squared : float - |<g|r|e>|² (m²)

    Returns:
    --------
    A : float - spontaneous emission rate (s⁻¹)
    """
    return 4 * alpha * omega**3 / (3 * c**2) * d_squared

def oscillator_strength(omega, d_squared):
    """
    Calculate oscillator strength.

    f = 2mω|<r>|² / 3ℏ
    """
    return 2 * m_e * omega * d_squared / (3 * hbar)

def natural_linewidth(A):
    """
    Natural linewidth Γ = ℏA (in Hz).
    """
    return A / (2 * np.pi)

# Hydrogen radial matrix elements (in units of a_0)
# |<n'l'|r|nl>| for common transitions
hydrogen_matrix_elements = {
    '1s-2p': 0.7449,
    '1s-3p': 0.2892,
    '2s-3p': 3.065,
    '2p-3s': 0.9384,
    '2p-3d': 4.748,
}

# Transition energies in eV
hydrogen_energies = {
    '1s-2p': 10.2,
    '1s-3p': 12.09,
    '2s-3p': 1.89,
    '2p-3s': 1.89,
    '2p-3d': 1.51,
}

# Calculate lifetimes for hydrogen transitions
print("=" * 60)
print("HYDROGEN SPONTANEOUS EMISSION RATES")
print("=" * 60)

print(f"\n{'Transition':<12} {'ΔE (eV)':<10} {'|d| (a₀)':<10} {'A (s⁻¹)':<12} {'τ (ns)':<10} {'Γ (MHz)':<10}")
print("-" * 64)

for trans, d_a0 in hydrogen_matrix_elements.items():
    dE = hydrogen_energies[trans]
    omega = dE * eV / hbar
    d_squared = (d_a0 * a_0)**2

    A = einstein_A(omega, d_squared)
    tau = 1 / A * 1e9  # ns
    Gamma = natural_linewidth(A) / 1e6  # MHz

    print(f"{trans:<12} {dE:<10.2f} {d_a0:<10.4f} {A:<12.2e} {tau:<10.2f} {Gamma:<10.1f}")

# Visualization 1: Lifetime vs frequency scaling
print("\n" + "=" * 60)
print("LIFETIME SCALING WITH TRANSITION FREQUENCY")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

# Fixed dipole moment = 1 a_0
d_fixed = a_0

# Frequency range from microwave to X-ray
freq_Hz = np.logspace(9, 18, 100)  # 1 GHz to 1 EHz
omega = 2 * np.pi * freq_Hz

A = einstein_A(omega, d_fixed**2)
tau = 1 / A

ax.loglog(freq_Hz, tau, 'b-', linewidth=2)

# Mark different spectral regions
regions = [
    (5e9, 'Microwave\n(qubits)', 'green'),
    (5e14, 'Optical\n(atoms)', 'orange'),
    (3e17, 'X-ray', 'red')
]

for freq, label, color in regions:
    idx = np.argmin(np.abs(freq_Hz - freq))
    ax.axvline(freq, color=color, linestyle='--', alpha=0.5)
    ax.scatter([freq], [tau[idx]], color=color, s=100, zorder=5)
    ax.annotate(label, (freq, tau[idx]), textcoords="offset points",
                xytext=(10, 10), fontsize=10)

ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Lifetime τ (s)', fontsize=12)
ax.set_title('Spontaneous Emission Lifetime: τ ∝ ω⁻³', fontsize=14)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(1e9, 1e18)
ax.set_ylim(1e-18, 1e6)

plt.tight_layout()
plt.savefig('lifetime_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 2: Decay dynamics
print("\n" + "=" * 60)
print("EXCITED STATE POPULATION DECAY")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Single exponential decay
ax = axes[0]
A_values = [1e8, 5e8, 2e9]  # Different decay rates
t = np.linspace(0, 20, 1000)  # ns

for A in A_values:
    tau = 1 / A * 1e9  # ns
    P_e = np.exp(-A * t * 1e-9)
    ax.plot(t, P_e, linewidth=2, label=f'τ = {tau:.1f} ns')

ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('Excited State Population', fontsize=12)
ax.set_title('Exponential Decay: P(t) = exp(-t/τ)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.05)

# Right: Photon emission probability distribution
ax = axes[1]
A = 6.27e8  # Hydrogen 2p
tau = 1 / A

t = np.linspace(0, 10e-9, 1000)  # seconds
# Probability distribution for emission time
P_emission = A * np.exp(-A * t)

ax.plot(t * 1e9, P_emission * 1e-9, 'r-', linewidth=2)
ax.fill_between(t * 1e9, P_emission * 1e-9, alpha=0.3)
ax.axvline(tau * 1e9, color='k', linestyle='--', label=f'τ = {tau*1e9:.2f} ns')

ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('Emission Probability Density (ns⁻¹)', fontsize=12)
ax.set_title('Photon Emission Time Distribution', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decay_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 3: Purcell effect
print("\n" + "=" * 60)
print("PURCELL EFFECT: CAVITY-MODIFIED EMISSION")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

# Purcell factor as function of detuning
Delta = np.linspace(-5, 5, 1000)  # Detuning in units of κ
kappa = 1  # Cavity linewidth (normalized)
g = 0.5  # Coupling strength in units of κ

# Purcell decay rate
gamma_purcell = g**2 * kappa / (Delta**2 + (kappa/2)**2)

ax.plot(Delta, gamma_purcell, 'b-', linewidth=2)
ax.axhline(g**2 / (kappa/4), color='r', linestyle='--',
           label=f'On resonance: Γ = 4g²/κ')
ax.axvline(0, color='k', linestyle=':', alpha=0.5)

ax.set_xlabel('Detuning Δ/κ', fontsize=12)
ax.set_ylabel('Purcell Decay Rate (normalized)', fontsize=12)
ax.set_title('Purcell Effect: Cavity-Enhanced/Suppressed Emission', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Add annotations
ax.annotate('Enhanced\n(resonant)', xy=(0, g**2/(kappa/4)),
            xytext=(1.5, g**2/(kappa/4)*1.2),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10)
ax.annotate('Suppressed\n(detuned)', xy=(3, gamma_purcell[800]),
            xytext=(3.5, 0.3),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10)

plt.tight_layout()
plt.savefig('purcell_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 4: Sum rule verification
print("\n" + "=" * 60)
print("THOMAS-REICHE-KUHN SUM RULE")
print("=" * 60)

# For harmonic oscillator, f_{0n} = n δ_{n,1}
# Sum = 1 (one electron)

# For hydrogen, approximate sum
print("\nHydrogen 1s state - oscillator strength sum:")
print("-" * 40)

total_f = 0
print(f"{'Final State':<15} {'f_1s→n':<15}")
print("-" * 30)

# Known oscillator strengths for hydrogen
f_values = {
    '2p': 0.4162,
    '3p': 0.0791,
    '4p': 0.0290,
    '5p': 0.0139,
    'continuum': 0.436  # Integrated continuum contribution
}

for state, f in f_values.items():
    print(f"1s → {state:<10} {f:<15.4f}")
    total_f += f

print("-" * 30)
print(f"{'Sum':<15} {total_f:<15.4f}")
print(f"{'Expected (Z=1)':<15} {'1.0000':<15}")

# Summary
print("\n" + "=" * 60)
print("KEY FORMULAS SUMMARY")
print("=" * 60)
print(f"""
Einstein A coefficient:
  A = (4α ω³)/(3c²) |⟨g|r|e⟩|²
  A = (ω³)/(3πε₀ℏc³) |d|²

Oscillator strength:
  f = (2mω)/(3ℏ) |⟨g|r|e⟩|²

Relations:
  A = (e²ω²)/(2πε₀mc³) f
  τ = 1/A
  Γ = ℏA (natural linewidth)

Sum rule:
  Σ f = Z (number of electrons)

Purcell factor:
  F_P = (3Qλ³)/(4π²V)
""")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("""
1. Spontaneous emission arises from vacuum fluctuations
2. Rate scales as ω³ - higher frequency decays faster
3. Microwave qubits have intrinsically long T₁
4. Cavity QED can enhance OR suppress emission (Purcell)
5. Sum rule constrains total oscillator strength
6. Natural linewidth sets fundamental limit on spectroscopy
""")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Einstein A coefficient | $A = \frac{4\alpha\omega^3}{3c^2}\|\mathbf{d}_{eg}\|^2$ |
| Natural lifetime | $\tau = 1/A$ |
| Natural linewidth | $\Gamma = \hbar A$ |
| Oscillator strength | $f = \frac{2m\omega}{3\hbar}\|\mathbf{d}_{eg}\|^2$ |
| Sum rule | $\sum_n f_{gn} = Z$ |
| Purcell factor | $F_P = \frac{3Q\lambda^3}{4\pi^2 V}$ |

### Main Takeaways

1. **Vacuum fluctuations** cause spontaneous emission
2. **A ∝ ω³** — higher frequencies decay faster
3. **Natural linewidth** limits spectroscopic resolution
4. **Purcell effect** modifies rates in cavities
5. **T₁ in qubits** follows same physics

---

## Daily Checklist

- [ ] I can derive the Einstein A coefficient using Fermi's Golden Rule
- [ ] I understand the role of vacuum fluctuations
- [ ] I can calculate atomic lifetimes from matrix elements
- [ ] I know the Thomas-Reiche-Kuhn sum rule
- [ ] I can explain the Purcell effect

---

## Preview: Day 476

Tomorrow is the **Month 17 Capstone**—a comprehensive review integrating all approximation methods learned this month.

---

**Next:** [Day_476_Sunday.md](Day_476_Sunday.md) — Month 17 Capstone
