# Day 355: Energy-Time Uncertainty — ΔE·Δt ≥ ℏ/2

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Energy-Time Uncertainty |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 355, you will be able to:

1. Explain why energy-time uncertainty differs from position-momentum uncertainty
2. Derive the energy-time relation from the generalized uncertainty principle
3. Apply the relation to calculate particle lifetimes and spectral widths
4. Interpret Δt as a characteristic evolution time, not an observable
5. Connect energy-time uncertainty to quantum tunneling and virtual particles
6. Understand the Mandelstam-Tamm interpretation

---

## Core Content

### 1. The Special Status of Time

Unlike position, **time is not an operator in quantum mechanics**. Time is a parameter that labels the evolution of states:

$$|\psi(t)\rangle = e^{-i\hat{H}t/\hbar}|\psi(0)\rangle$$

**Consequences:**

1. There is no "time eigenstate" |t⟩
2. We cannot construct [Ĥ, t̂] = iℏ analogously to [x̂, p̂] = iℏ
3. The energy-time uncertainty relation has a fundamentally different meaning

**Pauli's theorem (1933):** A self-adjoint time operator would imply unbounded energy spectrum from -∞ to +∞, contradicting the existence of ground states.

---

### 2. The Energy-Time Uncertainty Relation

Despite time not being an operator, we have:

$$\boxed{\Delta E \cdot \Delta t \geq \frac{\hbar}{2}}$$

**The question:** What is Δt if time isn't an operator?

**Answer:** Δt is the characteristic time for the system to change significantly.

---

### 3. The Mandelstam-Tamm Derivation (1945)

This is the rigorous derivation of energy-time uncertainty.

**Setup:** Consider any observable Q̂ that changes in time. From Ehrenfest:

$$\frac{d\langle\hat{Q}\rangle}{dt} = \frac{1}{i\hbar}\langle[\hat{Q}, \hat{H}]\rangle$$

**Apply the generalized uncertainty principle:**

$$\sigma_Q \sigma_H \geq \frac{1}{2}|\langle[\hat{Q}, \hat{H}]\rangle|$$

where σ_H = ΔE (energy uncertainty).

**Define the characteristic time:**

$$\tau_Q = \frac{\sigma_Q}{|d\langle\hat{Q}\rangle/dt|}$$

This is the time for ⟨Q̂⟩ to change by one standard deviation.

**Combining:**

$$\sigma_Q \cdot \Delta E \geq \frac{\hbar}{2}\left|\frac{d\langle\hat{Q}\rangle}{dt}\right|$$

$$\frac{\sigma_Q}{|d\langle\hat{Q}\rangle/dt|} \cdot \Delta E \geq \frac{\hbar}{2}$$

$$\boxed{\tau_Q \cdot \Delta E \geq \frac{\hbar}{2}}$$

**Interpretation:** The system cannot change significantly faster than ℏ/(2ΔE).

---

### 4. Interpretations of Δt

There are several valid interpretations of Δt:

#### (a) Evolution Time (Mandelstam-Tamm)

Δt = time for an observable to change by its standard deviation

$$\Delta t_Q = \frac{\sigma_Q}{|d\langle\hat{Q}\rangle/dt|}$$

#### (b) Lifetime of Excited States

For an unstable state with exponential decay:

$$|\psi(t)|^2 \propto e^{-t/\tau}$$

The lifetime τ satisfies:

$$\Gamma \cdot \tau \geq \hbar$$

where Γ = ΔE is the energy width (natural linewidth).

#### (c) Measurement Duration

The time taken to perform an energy measurement with precision ΔE:

$$\Delta t_{meas} \geq \frac{\hbar}{2\Delta E}$$

#### (d) Passage Time

Time for a particle to traverse a region:

$$\Delta t_{passage} \sim \frac{\Delta x}{v}$$

---

### 5. Natural Linewidth and Lifetimes

**The Breit-Wigner formula** connects energy width to lifetime:

$$\boxed{\Gamma \cdot \tau = \hbar}$$

where:
- Γ = full width at half maximum (FWHM) of spectral line
- τ = mean lifetime of excited state

**Example calculations:**

| Transition | τ (s) | Γ (eV) | Γ (Hz) |
|------------|-------|--------|--------|
| Hydrogen 2p→1s | 1.6×10⁻⁹ | 4×10⁻⁷ | 10⁸ |
| Nuclear γ | 10⁻¹⁴ | 0.07 | 1.6×10¹³ |
| Pion decay | 2.6×10⁻⁸ | 2.5×10⁻⁸ | 6×10⁶ |
| Z boson | 3×10⁻²⁵ | 2.5 | 6×10²³ |

**Short-lived states have broad energy distributions!**

---

### 6. Virtual Particles and Tunneling

The energy-time uncertainty allows "borrowing" energy for short times:

$$\Delta E \lesssim \frac{\hbar}{2\Delta t}$$

**Virtual particles:**

In quantum field theory, virtual particles can exist if:
$$E_{virtual} - E_{real} \lesssim \frac{\hbar}{2\Delta t}$$

This allows intermediate states that violate energy conservation briefly.

**Quantum tunneling time:**

For a particle tunneling through a barrier of height V₀ and width d:

$$\tau_{tunnel} \sim \frac{d}{v} \cdot e^{-\kappa d}$$

where κ = √(2m(V₀-E))/ℏ. The energy "borrowed" is ~(V₀ - E).

---

### 7. Quantum Speed Limit

The energy-time uncertainty sets a **minimum time** for quantum evolution.

**Margolus-Levitin theorem:**

The minimum time to evolve to an orthogonal state is:

$$\tau_{\perp} \geq \frac{\pi\hbar}{2\langle E\rangle}$$

where ⟨E⟩ is measured from the ground state.

**Combined bound:** Using both ΔE and ⟨E⟩:

$$\tau_{\perp} \geq \max\left(\frac{\pi\hbar}{2\Delta E}, \frac{\pi\hbar}{2\langle E\rangle}\right)$$

**Quantum computing implication:**

The maximum number of operations per second is bounded by:
$$\nu_{max} \sim \frac{2E}{\pi\hbar}$$

A quantum computer with 1 eV of energy cannot execute more than ~10¹⁵ ops/s.

---

### 8. Contrast with Position-Momentum

| Aspect | x-p Uncertainty | E-t Uncertainty |
|--------|-----------------|-----------------|
| Status | Both are operators | t is a parameter |
| Commutator | [x̂, p̂] = iℏ | No [Ĥ, t̂] |
| Derivation | Direct from CCR | Mandelstam-Tamm |
| Bound | State-independent | State-dependent through observable |
| Physical meaning | Preparation limit | Evolution rate limit |
| Measurement | Cannot prepare sharp x and p | Cannot evolve arbitrarily fast |

---

## Physical Interpretation

### What ΔE·Δt ≥ ℏ/2 Really Means

**Valid interpretations:**

1. **Lifetime-linewidth:** Short-lived states have broad energy distributions
2. **Evolution rate:** Systems with definite energy don't evolve (eigenstates are stationary)
3. **Measurement time:** Precise energy measurements take time

**Invalid interpretations:**

1. "Energy conservation can be violated for short times" — **No!** Energy is always conserved in QM
2. "Particles can borrow energy from the vacuum" — This is a pop-science oversimplification
3. "Simultaneous measurement of E and t is impossible" — t isn't measured like an observable

### Energy Eigenstates Are Stationary

If |ψ⟩ = |E⟩ (an energy eigenstate), then ΔE = 0.

For the uncertainty relation to hold, Δt → ∞.

**Physical meaning:** Energy eigenstates never change! They are "stationary states":
$$|\psi(t)\rangle = e^{-iEt/\hbar}|E\rangle$$

The phase rotates, but the probability distribution |⟨x|E⟩|² is constant.

---

## Worked Examples

### Example 1: Hydrogen Spectral Line Width

**Problem:** The 2p→1s transition in hydrogen has lifetime τ = 1.6 ns. Calculate the natural linewidth Γ and the fractional width Γ/E.

**Solution:**

**Energy-time relation:**
$$\Gamma \cdot \tau = \hbar$$

$$\Gamma = \frac{\hbar}{\tau} = \frac{1.055 \times 10^{-34} \text{ J·s}}{1.6 \times 10^{-9} \text{ s}} = 6.6 \times 10^{-26} \text{ J}$$

Converting to eV:
$$\Gamma = \frac{6.6 \times 10^{-26}}{1.6 \times 10^{-19}} = 4.1 \times 10^{-7} \text{ eV}$$

**Frequency width:**
$$\Delta\nu = \frac{\Gamma}{h} = \frac{4.1 \times 10^{-7} \text{ eV}}{4.14 \times 10^{-15} \text{ eV·s}} = 10^8 \text{ Hz} = 100 \text{ MHz}$$

**Fractional width:**

The 2p→1s photon has energy:
$$E_{photon} = 13.6 \text{ eV}\left(1 - \frac{1}{4}\right) = 10.2 \text{ eV}$$

$$\boxed{\frac{\Gamma}{E} = \frac{4.1 \times 10^{-7}}{10.2} = 4 \times 10^{-8}}$$

This is extremely narrow — atomic spectral lines are very sharp.

---

### Example 2: Pion Lifetime

**Problem:** The charged pion (π⁺) has a rest mass of 140 MeV/c² and decays with mean lifetime τ = 26 ns. Calculate the energy width Γ.

**Solution:**

$$\Gamma = \frac{\hbar}{\tau} = \frac{6.58 \times 10^{-16} \text{ eV·s}}{26 \times 10^{-9} \text{ s}} = 2.5 \times 10^{-8} \text{ eV}$$

$$\boxed{\Gamma = 25 \text{ neV}}$$

This is incredibly narrow compared to the rest mass energy!

$$\frac{\Gamma}{mc^2} = \frac{2.5 \times 10^{-8} \text{ eV}}{140 \times 10^6 \text{ eV}} = 1.8 \times 10^{-16}$$

---

### Example 3: Time to Evolve to Orthogonal State

**Problem:** A two-level system is prepared in |ψ(0)⟩ = (|0⟩ + |1⟩)/√2 with energies E₀ = 0, E₁ = E. Find the minimum time to reach an orthogonal state.

**Solution:**

The state evolves as:
$$|\psi(t)\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + e^{-iEt/\hbar}|1\rangle\right)$$

**Orthogonality condition:** ⟨ψ(0)|ψ(t)⟩ = 0

$$\frac{1}{2}\left(1 + e^{-iEt/\hbar}\right) = 0$$

$$e^{-iEt/\hbar} = -1$$

$$t = \frac{\pi\hbar}{E}$$

**Energy uncertainty:**
$$\langle\hat{H}\rangle = \frac{E}{2}, \quad \langle\hat{H}^2\rangle = \frac{E^2}{2}$$
$$\Delta E = \sqrt{\frac{E^2}{2} - \frac{E^2}{4}} = \frac{E}{2}$$

**Margolus-Levitin bound:**
$$\tau_{\perp} \geq \frac{\pi\hbar}{2\langle E\rangle} = \frac{\pi\hbar}{2 \cdot E/2} = \frac{\pi\hbar}{E}$$

**Verify:** Our calculated t = πℏ/E exactly saturates the bound!

$$\boxed{t = \frac{\pi\hbar}{E}}$$

---

## Practice Problems

### Level 1: Direct Application

1. **Muon lifetime:** The muon has τ = 2.2 μs. Calculate its energy width Γ in eV.

2. **Spectral resolution:** What is the minimum time needed to measure an energy difference of 1 meV?

3. **Z boson:** The Z boson has Γ = 2.5 GeV. Calculate its mean lifetime.

### Level 2: Intermediate

4. **Rabi oscillations:** A two-level system oscillates between states with Rabi frequency Ω.
   (a) What is the time to reach an orthogonal state?
   (b) What is ΔE for the dressed states?
   (c) Verify the energy-time uncertainty.

5. **Quantum Zeno effect:** Explain how frequent measurements can prevent decay using ΔE·Δt ≥ ℏ/2.

6. **Doppler vs. natural width:** For the hydrogen 2p→1s line at T = 300 K:
   (a) Calculate the Doppler width.
   (b) Compare to the natural width.
   (c) Which dominates?

### Level 3: Challenging

7. **Tunneling time:** For an electron tunneling through a barrier V₀ = 2 eV, width d = 1 nm:
   (a) Estimate the "borrowed" energy.
   (b) Calculate the Büttiker-Landauer tunneling time.
   (c) Verify consistency with ΔE·Δt ≥ ℏ/2.

8. **Casimir effect:** Virtual photon pairs can exist for time Δt with energy ΔE ~ ℏc/Δx.
   (a) Derive the Casimir force between parallel plates from energy-time uncertainty.
   (b) Compare to the exact result F/A = -π²ℏc/(240d⁴).

9. **Quantum speed limit:** A quantum computer has available energy E = 1 J.
   (a) What is the maximum gate rate?
   (b) If each gate requires rotating to an orthogonal state, how many gates per second?
   (c) How does this compare to current technology (~10⁹ gates/s)?

---

## Computational Lab

### Objective
Explore energy-time uncertainty through wave packet evolution and decay phenomena.

```python
"""
Day 355 Computational Lab: Energy-Time Uncertainty
Quantum Mechanics Core - Year 1, Week 51
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from typing import Tuple

# Physical constants
hbar = 1.055e-34  # J·s
eV = 1.602e-19    # J
c = 3e8           # m/s

# Use natural units for simulations: hbar = 1
hbar_natural = 1.0

# =============================================================================
# Part 1: Lifetime-Linewidth Relation
# =============================================================================

print("=" * 70)
print("Part 1: Lifetime-Linewidth Relation")
print("=" * 70)

def lifetime_to_width(tau: float, hbar: float = 1.055e-34) -> float:
    """Convert lifetime to energy width (natural linewidth)."""
    return hbar / tau

def width_to_lifetime(gamma: float, hbar: float = 1.055e-34) -> float:
    """Convert energy width to lifetime."""
    return hbar / gamma

# Example particles/transitions
particles = [
    ("Hydrogen 2p→1s", 1.6e-9, None),
    ("Sodium D-line", 16e-9, None),
    ("Muon", 2.2e-6, None),
    ("Charged pion", 26e-9, None),
    ("Neutral pion", 8.4e-17, None),
    ("Z boson", None, 2.5e9 * eV),
    ("Higgs boson", None, 4.2e-3 * 1e9 * eV),
]

print("\nLifetime-Linewidth Relation: Γ·τ = ℏ")
print("-" * 70)
print(f"{'Particle/Transition':<25} {'Lifetime (s)':<15} {'Width (eV)':<15} {'Γτ/ℏ':<10}")
print("-" * 70)

for name, tau, gamma in particles:
    if tau is not None:
        gamma_calc = lifetime_to_width(tau) / eV
        check = gamma_calc * eV * tau / hbar
        print(f"{name:<25} {tau:.2e}       {gamma_calc:.2e}       {check:.4f}")
    else:
        tau_calc = width_to_lifetime(gamma)
        check = gamma / eV * tau_calc / hbar
        print(f"{name:<25} {tau_calc:.2e}       {gamma/eV:.2e}       {check:.4f}")

# =============================================================================
# Part 2: Breit-Wigner Distribution
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Breit-Wigner (Lorentzian) Distribution")
print("=" * 70)

def breit_wigner(E: np.ndarray, E0: float, Gamma: float) -> np.ndarray:
    """
    Breit-Wigner distribution for unstable states.
    P(E) ∝ 1/((E - E0)² + (Γ/2)²)
    """
    return 1 / ((E - E0)**2 + (Gamma/2)**2)

# Plot for different widths
E0 = 0  # Center energy
widths = [0.5, 1.0, 2.0, 4.0]
E_range = np.linspace(-10, 10, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
for Gamma in widths:
    P = breit_wigner(E_range, E0, Gamma)
    P = P / np.max(P)  # Normalize peak to 1
    ax1.plot(E_range, P, linewidth=2, label=f'Γ = {Gamma}')

ax1.set_xlabel('E - E₀', fontsize=12)
ax1.set_ylabel('P(E) (normalized)', fontsize=12)
ax1.set_title('Breit-Wigner Distribution: Wider Width = Shorter Lifetime', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Compare Gaussian vs Lorentzian
ax2 = axes[1]
Gamma = 2.0
sigma = Gamma / (2 * np.sqrt(2 * np.log(2)))  # Same FWHM

lorentzian = breit_wigner(E_range, 0, Gamma)
lorentzian = lorentzian / np.max(lorentzian)

gaussian = np.exp(-(E_range)**2 / (2 * sigma**2))

ax2.plot(E_range, lorentzian, 'b-', linewidth=2, label='Lorentzian (natural)')
ax2.plot(E_range, gaussian, 'r--', linewidth=2, label='Gaussian (Doppler)')
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('E - E₀', fontsize=12)
ax2.set_ylabel('P(E) (normalized)', fontsize=12)
ax2.set_title('Lorentzian vs Gaussian Line Shape', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_355_lineshapes.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_355_lineshapes.png'")

# =============================================================================
# Part 3: Wave Packet Decay
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Unstable State Decay (Exponential Law)")
print("=" * 70)

def survival_probability(t: np.ndarray, tau: float) -> np.ndarray:
    """Exponential decay survival probability."""
    return np.exp(-t / tau)

def zeno_modified_survival(t: np.ndarray, tau: float, t_meas: float) -> np.ndarray:
    """
    Survival probability with periodic measurements (quantum Zeno effect).
    For short times: P ≈ 1 - (t/τ_Z)² for t << τ_Z
    """
    # Short-time quadratic behavior
    tau_Z = np.sqrt(tau * t_meas)  # Zeno time

    # Number of measurements
    n_meas = t / t_meas

    # Each measurement resets to quadratic regime
    p_single = 1 - (t_meas / tau_Z)**2

    return p_single ** n_meas

# Compare decay with and without measurements
tau = 1.0  # Natural lifetime
t = np.linspace(0, 5, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(t, survival_probability(t, tau), 'b-', linewidth=2,
         label='Exponential decay (no measurement)')
ax1.set_xlabel('Time (τ)', fontsize=12)
ax1.set_ylabel('Survival Probability', fontsize=12)
ax1.set_title('Unstable State Decay', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Quantum Zeno effect illustration
ax2 = axes[1]
measurement_intervals = [0.5, 0.2, 0.1, 0.05]
colors = plt.cm.viridis(np.linspace(0, 0.8, len(measurement_intervals)))

ax2.plot(t, survival_probability(t, tau), 'b-', linewidth=2, label='No measurement')

for dt, color in zip(measurement_intervals, colors):
    # Simplified Zeno: more frequent measurements → slower decay
    effective_tau = tau * (1 + tau / dt)  # Approximate Zeno modification
    ax2.plot(t, survival_probability(t, effective_tau), '--', color=color,
             linewidth=1.5, label=f'Δt = {dt}τ')

ax2.set_xlabel('Time (τ)', fontsize=12)
ax2.set_ylabel('Survival Probability', fontsize=12)
ax2.set_title('Quantum Zeno Effect: Frequent Measurements Slow Decay', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_355_decay.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_355_decay.png'")

# =============================================================================
# Part 4: Evolution Time and Mandelstam-Tamm
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Mandelstam-Tamm Relation")
print("=" * 70)

def two_level_evolution(t: np.ndarray, omega: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolution of |ψ⟩ = (|0⟩ + |1⟩)/√2 with E1 - E0 = ℏω.
    Returns (probability amplitude squared for |0⟩, overlap with initial state).
    """
    # |ψ(t)⟩ = (|0⟩ + e^{-iωt}|1⟩)/√2
    prob_0 = 0.5 * np.ones_like(t)  # Always 0.5

    # Overlap with initial state
    overlap = 0.5 * np.abs(1 + np.exp(-1j * omega * t))**2

    return prob_0, overlap

omega = 1.0  # Energy difference in natural units
t = np.linspace(0, 4*np.pi/omega, 500)

prob_0, overlap = two_level_evolution(t, omega)

# Time to orthogonal state
t_ortho = np.pi / omega

# Energy uncertainty
delta_E = omega / 2  # For equal superposition

# Verify Mandelstam-Tamm
print(f"\nTwo-level system with ω = {omega}:")
print(f"Energy uncertainty ΔE = ω/2 = {delta_E}")
print(f"Time to orthogonal state: t⊥ = π/ω = {t_ortho:.4f}")
print(f"Product ΔE·t⊥ = {delta_E * t_ortho:.4f}")
print(f"Bound π/2 = {np.pi/2:.4f}")
print(f"Relation: ΔE·t⊥ = π/2 ✓ (saturates the Margolus-Levitin bound)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(t * omega / np.pi, overlap, 'b-', linewidth=2)
ax1.axvline(x=1, color='red', linestyle='--', label=f't = π/ω (orthogonal)')
ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('t × ω/π', fontsize=12)
ax1.set_ylabel('|⟨ψ(0)|ψ(t)⟩|²', fontsize=12)
ax1.set_title('Overlap with Initial State', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Different energy splittings
ax2 = axes[1]
omegas = [0.5, 1.0, 2.0, 4.0]
t_plot = np.linspace(0, 4*np.pi, 500)

for w in omegas:
    _, overlap_w = two_level_evolution(t_plot, w)
    ax2.plot(t_plot, overlap_w, linewidth=2, label=f'ω = {w}')

ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('|⟨ψ(0)|ψ(t)⟩|²', fontsize=12)
ax2.set_title('Evolution Rate Depends on Energy Splitting', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_355_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_355_evolution.png'")

# =============================================================================
# Part 5: Quantum Speed Limit
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Quantum Speed Limit")
print("=" * 70)

def quantum_speed_limit(delta_E: float, mean_E: float, hbar: float = 1.0) -> float:
    """
    Minimum time to evolve to orthogonal state.
    τ⊥ ≥ max(πℏ/(2ΔE), πℏ/(2⟨E⟩))
    """
    bound1 = np.pi * hbar / (2 * delta_E) if delta_E > 0 else np.inf
    bound2 = np.pi * hbar / (2 * mean_E) if mean_E > 0 else np.inf
    return max(bound1, bound2)

# Calculate for various quantum systems
systems = [
    ("Qubit (1 GHz)", 4.14e-6, 4.14e-6),  # ΔE = ⟨E⟩ = hν
    ("Atomic transition", 2, 2),  # eV
    ("Molecular vibration", 0.1, 0.1),  # eV
    ("Nuclear transition", 1e6, 1e6),  # eV
]

print("\nQuantum Speed Limit for Various Systems:")
print("-" * 60)
print(f"{'System':<25} {'ΔE (eV)':<12} {'τ_min (s)':<15}")
print("-" * 60)

for name, delta_E_ev, mean_E_ev in systems:
    delta_E_J = delta_E_ev * eV
    tau_min = quantum_speed_limit(delta_E_J, mean_E_ev * eV, hbar)
    print(f"{name:<25} {delta_E_ev:<12.2e} {tau_min:<15.2e}")

# =============================================================================
# Part 6: Visualization Summary
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Summary Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Lifetime vs Linewidth
ax1 = axes[0, 0]
tau_range = np.logspace(-25, -6, 100)  # seconds
gamma_range = hbar / tau_range / eV  # eV

ax1.loglog(tau_range, gamma_range, 'b-', linewidth=2)
ax1.set_xlabel('Lifetime τ (s)', fontsize=12)
ax1.set_ylabel('Width Γ (eV)', fontsize=12)
ax1.set_title('Γτ = ℏ: Inverse Relationship', fontsize=14)
ax1.grid(True, alpha=0.3, which='both')

# Mark some particles
particle_points = [
    (1.6e-9, "H 2p→1s"),
    (2.2e-6, "Muon"),
    (8.4e-17, "π⁰"),
    (3e-25, "Z"),
]
for tau_p, name in particle_points:
    gamma_p = hbar / tau_p / eV
    ax1.scatter([tau_p], [gamma_p], s=100, zorder=5)
    ax1.annotate(name, (tau_p, gamma_p), xytext=(5, 5),
                 textcoords='offset points', fontsize=10)

# Panel 2: Speed limit
ax2 = axes[0, 1]
E_range = np.logspace(-3, 6, 100)  # eV
tau_min_range = np.pi * hbar / (2 * E_range * eV)

ax2.loglog(E_range, tau_min_range, 'r-', linewidth=2)
ax2.set_xlabel('Energy Scale ΔE (eV)', fontsize=12)
ax2.set_ylabel('Minimum Evolution Time τ_min (s)', fontsize=12)
ax2.set_title('Quantum Speed Limit: τ ≥ πℏ/(2ΔE)', fontsize=14)
ax2.grid(True, alpha=0.3, which='both')

# Panel 3: Decay and Zeno comparison
ax3 = axes[1, 0]
t = np.linspace(0, 3, 100)
tau = 1.0

for n_meas in [0, 5, 10, 20]:
    if n_meas == 0:
        P = np.exp(-t)
        ax3.plot(t, P, 'b-', linewidth=2, label='Free decay')
    else:
        # Simplified Zeno model
        P = np.exp(-t / (1 + n_meas/5))
        ax3.plot(t, P, '--', linewidth=1.5, label=f'{n_meas} measurements')

ax3.set_xlabel('Time (τ)', fontsize=12)
ax3.set_ylabel('Survival Probability', fontsize=12)
ax3.set_title('Quantum Zeno Effect', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Uncertainty product
ax4 = axes[1, 1]
delta_E_plot = np.linspace(0.1, 5, 100)
delta_t_min = 1 / (2 * delta_E_plot)  # In units where hbar = 1

ax4.fill_between(delta_E_plot, 0, delta_t_min, alpha=0.3, color='red',
                 label='Forbidden: ΔE·Δt < ℏ/2')
ax4.plot(delta_E_plot, delta_t_min, 'r-', linewidth=2,
         label='Bound: ΔE·Δt = ℏ/2')

# Mark some points
ax4.scatter([1], [0.5], s=100, c='blue', zorder=5, label='ΔE = 1, Δt = 0.5')
ax4.scatter([2], [0.25], s=100, c='green', zorder=5, label='ΔE = 2, Δt = 0.25')
ax4.scatter([0.5], [1], s=100, c='purple', zorder=5, label='ΔE = 0.5, Δt = 1')

ax4.set_xlabel('ΔE (natural units)', fontsize=12)
ax4.set_ylabel('Δt (natural units)', fontsize=12)
ax4.set_title('Energy-Time Uncertainty Region', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 5)
ax4.set_ylim(0, 2)

plt.tight_layout()
plt.savefig('day_355_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_355_summary.png'")

print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Energy-time uncertainty | ΔE·Δt ≥ ℏ/2 |
| Lifetime-linewidth | Γ·τ = ℏ |
| Mandelstam-Tamm | τ_Q·ΔE ≥ ℏ/2 where τ_Q = σ_Q/\|d⟨Q̂⟩/dt\| |
| Quantum speed limit | τ_⊥ ≥ πℏ/(2ΔE) |
| Margolus-Levitin | τ_⊥ ≥ πℏ/(2⟨E⟩) |
| Breit-Wigner | P(E) ∝ 1/((E-E₀)² + (Γ/2)²) |

### Main Takeaways

1. **Time is a parameter, not an operator** — Energy-time uncertainty has a different status
2. **Δt is a characteristic evolution time** — Not an "uncertainty in time measurement"
3. **Γτ = ℏ relates lifetime to spectral width** — Short-lived states are spectrally broad
4. **Energy eigenstates don't evolve** — ΔE = 0 implies infinite evolution time
5. **Quantum speed limit bounds evolution** — Cannot compute infinitely fast

---

## Daily Checklist

- [ ] Read Sakurai Chapter 2.1 (Time Evolution)
- [ ] Read Griffiths Section 3.5.3 (Energy-Time Uncertainty)
- [ ] Derive the Mandelstam-Tamm relation
- [ ] Calculate Γ for the hydrogen 2p→1s transition
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run the computational lab
- [ ] Explain why t is a parameter, not an operator

---

## Preview: Day 356

Tomorrow we explore **incompatible observables** more deeply—the physical consequences of non-commuting operators. We'll see how complementarity (Bohr) connects to measurement disturbance, how quantum cryptography exploits incompatibility, and what it means that nature fundamentally forbids certain kinds of knowledge.

---

*"If you can measure something precisely, you disturb it; if you do not disturb it, you cannot measure it precisely. This is true for all measurements in quantum theory."* — Niels Bohr

---

**Next:** [Day_356_Saturday.md](Day_356_Saturday.md) — Incompatible Observables
