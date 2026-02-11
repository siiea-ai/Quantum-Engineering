# Day 220: Relativistic Mechanics (Energy-Momentum)

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Relativistic Energy and Momentum |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications: Collisions and Decays |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 220, you will be able to:

1. Derive the relativistic energy-momentum relation from first principles
2. Apply mass-energy equivalence ($E = mc^2$) to physical problems
3. Analyze relativistic collisions using 4-momentum conservation
4. Calculate invariant mass and threshold energies
5. Understand the concept of massless particles (photons)
6. Connect relativistic mechanics to particle physics and quantum field theory

---

## Core Content

### 1. Relativistic Momentum

**Classical momentum fails at high speeds.** Newton's second law $\mathbf{F} = d\mathbf{p}/dt$ must be preserved, but we need a new definition of momentum.

From the 4-momentum $p^{\mu} = mu^{\mu} = m\gamma(c, \mathbf{v})$, the spatial components give:

$$\boxed{\mathbf{p} = \gamma m\mathbf{v} = \frac{m\mathbf{v}}{\sqrt{1 - v^2/c^2}}}$$

**Properties:**
- As $v \to 0$: $\mathbf{p} \to m\mathbf{v}$ (classical limit)
- As $v \to c$: $|\mathbf{p}| \to \infty$ (infinite momentum needed to reach $c$)
- Direction: $\mathbf{p} \parallel \mathbf{v}$ (same as classical)

### 2. Relativistic Energy

The time component of 4-momentum is $p^0 = \gamma mc$.

Define **relativistic energy**:
$$\boxed{E = \gamma mc^2 = \frac{mc^2}{\sqrt{1 - v^2/c^2}}}$$

**Taylor expansion for low velocities:**
$$E = mc^2\left(1 + \frac{1}{2}\frac{v^2}{c^2} + \frac{3}{8}\frac{v^4}{c^4} + \cdots\right)$$

$$E \approx mc^2 + \frac{1}{2}mv^2 + \cdots$$

The first term is the **rest energy**, the second is the classical kinetic energy!

### 3. Mass-Energy Equivalence

**Rest energy:**
$$\boxed{E_0 = mc^2}$$

This is Einstein's most famous result: mass is a form of energy.

**Kinetic energy:**
$$\boxed{K = E - E_0 = (\gamma - 1)mc^2}$$

**Examples of mass-energy conversion:**
- Nuclear fission: $\sim 0.1\%$ mass converted to energy
- Nuclear fusion: $\sim 0.7\%$ mass converted to energy
- Matter-antimatter annihilation: $100\%$ mass converted to energy

### 4. The Energy-Momentum Relation

From the invariant $p_{\mu}p^{\mu} = -m^2c^2$:

$$-\frac{E^2}{c^2} + |\mathbf{p}|^2 = -m^2c^2$$

$$\boxed{E^2 = (pc)^2 + (mc^2)^2}$$

This is the **relativistic dispersion relation**.

**Special cases:**

| Particle Type | Condition | Energy |
|---------------|-----------|--------|
| At rest | $p = 0$ | $E = mc^2$ |
| Ultrarelativistic | $pc \gg mc^2$ | $E \approx pc$ |
| Massless | $m = 0$ | $E = pc$ |

### 5. Massless Particles

For $m = 0$:
$$E = pc, \quad \text{and} \quad v = c$$

**Photons** have:
- Energy: $E = h\nu = \hbar\omega$
- Momentum: $p = E/c = h/\lambda = \hbar k$
- 4-momentum: $p^{\mu} = (\hbar\omega/c)(1, \hat{\mathbf{n}})$ where $\hat{\mathbf{n}}$ is the direction

The invariant:
$$p_{\mu}p^{\mu} = -\frac{E^2}{c^2} + p^2 = 0$$

### 6. Relativistic Force

Newton's second law generalizes to:
$$\mathbf{F} = \frac{d\mathbf{p}}{dt} = \frac{d}{dt}(\gamma m\mathbf{v})$$

**For constant mass:**
$$\mathbf{F} = m\frac{d(\gamma\mathbf{v})}{dt} = \gamma m\mathbf{a} + m\mathbf{v}\frac{d\gamma}{dt}$$

The force depends on both acceleration and rate of change of $\gamma$!

**Special case - force parallel to velocity:**
$$F_{\parallel} = \gamma^3 ma_{\parallel}$$

**Special case - force perpendicular to velocity:**
$$F_{\perp} = \gamma ma_{\perp}$$

### 7. Relativistic Collisions

**Conservation laws:**
1. Total 4-momentum is conserved: $\sum_i p_i^{\mu} = \sum_f p_f^{\mu}$
2. This gives 4 equations: energy + 3 momentum components

**Invariant mass of a system:**
$$\left(\sum_i p_i^{\mu}\right)\left(\sum_i p_{i\mu}\right) = -M^2c^2$$

**Center-of-momentum (CM) frame:**
- Total 3-momentum is zero: $\sum_i \mathbf{p}_i = 0$
- Total energy equals invariant mass: $E_{CM} = Mc^2$

### 8. Threshold Energy

To create a particle of mass $M$ in a collision, we need minimum energy when the products are at rest in the CM frame.

**Example: Creating an antiproton**
$$p + p \to p + p + p + \bar{p}$$

Minimum CM energy: $E_{CM} = 4m_pc^2$ (4 proton masses)

In the lab frame (one proton at rest):
$$(E_{beam} + m_pc^2)^2 - p_{beam}^2c^2 = (4m_pc^2)^2$$

Using $E_{beam}^2 = p_{beam}^2c^2 + m_p^2c^4$:
$$2E_{beam}m_pc^2 + 2m_p^2c^4 = 16m_p^2c^4$$

$$E_{beam} = 7m_pc^2$$

The beam proton needs kinetic energy $K = 6m_pc^2 \approx 5.6$ GeV!

### 9. Two-Body Decays

For a particle of mass $M$ decaying into particles with masses $m_1$ and $m_2$:

**In the rest frame of $M$:**
- Energy conservation: $Mc^2 = E_1 + E_2$
- Momentum conservation: $\mathbf{p}_1 = -\mathbf{p}_2$, so $|\mathbf{p}_1| = |\mathbf{p}_2| = p$

From $E_i^2 = p^2c^2 + m_i^2c^4$:
$$E_1 = \frac{M^2 + m_1^2 - m_2^2}{2M}c^2$$
$$E_2 = \frac{M^2 + m_2^2 - m_1^2}{2M}c^2$$
$$p = \frac{c}{2M}\sqrt{[M^2 - (m_1 + m_2)^2][M^2 - (m_1 - m_2)^2]}$$

---

## Quantum Mechanics Connection

### The Dirac Equation and Antimatter

The Dirac equation linearizes the energy-momentum relation:

$$i\hbar\frac{\partial\psi}{\partial t} = (c\boldsymbol{\alpha}\cdot\hat{\mathbf{p}} + \beta mc^2)\psi$$

where $\boldsymbol{\alpha}$ and $\beta$ are $4\times4$ matrices.

This predicts:
1. **Spin-1/2** particles (two spin states)
2. **Antimatter** (negative energy solutions reinterpreted as positrons)
3. **Precise magnetic moment** ($g \approx 2$)

### Spin as a Relativistic Effect

The electron's spin angular momentum emerges naturally from the Dirac equation. In the non-relativistic limit:

$$H = mc^2 + \frac{\mathbf{p}^2}{2m} - \frac{\mathbf{p}^4}{8m^3c^2} + \frac{e\hbar}{2mc}\boldsymbol{\sigma}\cdot\mathbf{B} + \text{spin-orbit}$$

The term $\frac{e\hbar}{2mc}\boldsymbol{\sigma}\cdot\mathbf{B}$ is the Zeeman interaction with $g = 2$ (predicted without being put in by hand).

### Particle Creation and Annihilation

The equation $E = mc^2$ implies that energy can be converted to mass and vice versa:

**Pair creation:** $\gamma \to e^- + e^+$ (requires $E_\gamma \geq 2m_ec^2 = 1.022$ MeV)

**Pair annihilation:** $e^- + e^+ \to 2\gamma$ (usually produces two photons)

This is the foundation of quantum field theory, where particles are excitations of underlying fields.

### Relativistic Quantum Mechanics Summary

| Non-relativistic | Relativistic |
|------------------|--------------|
| Schrödinger equation | Klein-Gordon, Dirac equations |
| $E = p^2/2m$ | $E^2 = (pc)^2 + (mc^2)^2$ |
| Particle number conserved | Particle creation/annihilation |
| Spin is added ad hoc | Spin emerges naturally |
| No antimatter | Antimatter required |

---

## Worked Examples

### Example 1: Relativistic Kinetic Energy

**Problem:** How much kinetic energy is required to accelerate a proton to $v = 0.99c$?

**Solution:**

Rest mass energy: $E_0 = m_pc^2 = 938.3$ MeV

Lorentz factor: $\gamma = 1/\sqrt{1 - 0.99^2} = 7.09$

Total energy: $E = \gamma m_pc^2 = 7.09 \times 938.3 = 6652$ MeV

Kinetic energy: $K = E - E_0 = 6652 - 938 = 5714$ MeV

$$\boxed{K = 5.71 \text{ GeV}}$$

Note: Classical formula would give $K = \frac{1}{2}mv^2 = \frac{1}{2}(0.99)^2 \times 938 = 460$ MeV, a huge underestimate!

### Example 2: Compton Scattering

**Problem:** A photon with wavelength $\lambda = 0.1$ nm scatters off an electron at rest. Find the wavelength after scattering at $\theta = 90°$.

**Solution:**

The Compton formula:
$$\lambda' - \lambda = \frac{h}{m_ec}(1 - \cos\theta)$$

Compton wavelength: $\lambda_C = h/(m_ec) = 2.43 \times 10^{-12}$ m = 0.00243 nm

For $\theta = 90°$: $\cos\theta = 0$

$$\lambda' = \lambda + \lambda_C = 0.1 + 0.00243 = 0.10243 \text{ nm}$$

$$\boxed{\lambda' = 0.1024 \text{ nm}}$$

### Example 3: Invariant Mass Reconstruction

**Problem:** In a collider experiment, a particle decays into two photons with energies $E_1 = 70$ MeV and $E_2 = 70$ MeV, traveling at an angle of $30°$ to each other. What was the mass of the original particle?

**Solution:**

For photons: $p_i = E_i/c$

Total 4-momentum:
$$P^{\mu} = p_1^{\mu} + p_2^{\mu}$$

Total energy: $E = E_1 + E_2 = 140$ MeV

Total momentum (using vector addition):
$$|\mathbf{P}|^2 = p_1^2 + p_2^2 + 2p_1p_2\cos(30°)$$
$$= (70)^2 + (70)^2 + 2(70)(70)(0.866)/c^2$$
$$= 4900 + 4900 + 8492 = 18292 \text{ MeV}^2/c^2$$
$$|\mathbf{P}| = 135.2 \text{ MeV}/c$$

Invariant mass:
$$M^2c^4 = E^2 - |\mathbf{P}|^2c^2 = 140^2 - 135.2^2 = 19600 - 18279 = 1321 \text{ MeV}^2$$
$$M = 36.3 \text{ MeV}/c^2$$

This is close to the $\pi^0$ mass (135 MeV). Let me recalculate...

Actually for a $\pi^0$ the photons would have equal energy in its rest frame. In the lab, with opening angle $\theta$:
$$M^2 = 2E_1E_2(1 - \cos\theta)/c^4$$
$$M^2 = 2(70)(70)(1 - 0.866) = 1313 \text{ MeV}^2$$
$$M = 36 \text{ MeV}/c^2$$

$$\boxed{M = 36 \text{ MeV}/c^2}$$

---

## Practice Problems

### Problem 1: Direct Application
An electron has kinetic energy $K = 2$ MeV. Calculate:
(a) Its total energy
(b) Its momentum
(c) Its velocity

**Answers:** (a) $E = 2.511$ MeV; (b) $p = 2.46$ MeV/c; (c) $v = 0.979c$

### Problem 2: Intermediate
A photon of energy 10 MeV creates an electron-positron pair. What is the maximum kinetic energy each particle can have?

**Answer:** $K_{max} = (10 - 2 \times 0.511)/2 = 4.49$ MeV (when both move in same direction)

### Problem 3: Challenging
In the reaction $\pi^- + p \to K^0 + \Lambda^0$, a negative pion with kinetic energy $K_\pi$ strikes a proton at rest. Find the threshold kinetic energy for this reaction.

Masses: $m_\pi = 140$ MeV/c², $m_p = 938$ MeV/c², $m_K = 498$ MeV/c², $m_\Lambda = 1116$ MeV/c²

**Answer:** $K_{threshold} = 904$ MeV

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

# Physical constants (in natural units where convenient)
c = 299792458  # m/s
m_e = 0.511  # MeV/c^2
m_p = 938.3  # MeV/c^2
m_pi0 = 135.0  # MeV/c^2
m_mu = 105.7  # MeV/c^2

def gamma(v, c=1):
    """Lorentz factor"""
    return 1 / np.sqrt(1 - (v/c)**2)

def relativistic_energy(m, v, c=1):
    """E = gamma * m * c^2"""
    return gamma(v, c) * m * c**2

def relativistic_momentum(m, v, c=1):
    """p = gamma * m * v"""
    return gamma(v, c) * m * v

def energy_from_momentum(m, p, c=1):
    """E = sqrt((pc)^2 + (mc^2)^2)"""
    return np.sqrt((p*c)**2 + (m*c**2)**2)

def velocity_from_momentum(m, p, c=1):
    """v = pc^2/E"""
    E = energy_from_momentum(m, p, c)
    return p * c**2 / E

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Relativistic vs Classical Kinetic Energy ==========
ax1 = axes[0, 0]

v_ratio = np.linspace(0.01, 0.99, 100)  # v/c

# Classical KE: (1/2)mv^2
KE_classical = 0.5 * v_ratio**2  # In units of mc^2

# Relativistic KE: (gamma - 1)mc^2
gamma_values = 1 / np.sqrt(1 - v_ratio**2)
KE_relativistic = gamma_values - 1  # In units of mc^2

ax1.plot(v_ratio, KE_classical, 'b--', linewidth=2, label='Classical: $\\frac{1}{2}mv^2$')
ax1.plot(v_ratio, KE_relativistic, 'r-', linewidth=2, label='Relativistic: $(\\gamma-1)mc^2$')

# Mark where they diverge
ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
ax1.annotate('v = 0.5c', (0.51, 0.5), fontsize=10)

ax1.set_xlabel('v/c', fontsize=12)
ax1.set_ylabel('Kinetic Energy (units of $mc^2$)', fontsize=12)
ax1.set_title('Classical vs Relativistic Kinetic Energy', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 5)

# ========== Plot 2: Energy-Momentum Relation ==========
ax2 = axes[0, 1]

# Different particles
particles = [
    ('Electron', m_e, 'blue'),
    ('Muon', m_mu, 'green'),
    ('Proton', m_p, 'red'),
    ('Photon', 0, 'orange'),
]

p_values = np.linspace(0, 2000, 500)  # MeV/c

for name, mass, color in particles:
    if mass > 0:
        E = np.sqrt(p_values**2 + mass**2)
        ax2.plot(p_values, E, color=color, linewidth=2, label=f'{name} ($m={mass:.1f}$ MeV/c²)')
    else:
        # Photon: E = pc
        ax2.plot(p_values, p_values, color=color, linewidth=2, linestyle='--', label='Photon (m=0)')

ax2.set_xlabel('Momentum p (MeV/c)', fontsize=12)
ax2.set_ylabel('Energy E (MeV)', fontsize=12)
ax2.set_title('Energy-Momentum Relation: $E^2 = (pc)^2 + (mc^2)^2$', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 2000)
ax2.set_ylim(0, 2500)

# ========== Plot 3: Momentum vs Velocity ==========
ax3 = axes[1, 0]

v_ratio = np.linspace(0.01, 0.999, 100)

# Classical momentum: p = mv
p_classical = v_ratio  # In units of mc

# Relativistic momentum: p = gamma*m*v
gamma_values = 1 / np.sqrt(1 - v_ratio**2)
p_relativistic = gamma_values * v_ratio  # In units of mc

ax3.plot(v_ratio, p_classical, 'b--', linewidth=2, label='Classical: $mv$')
ax3.plot(v_ratio, p_relativistic, 'r-', linewidth=2, label='Relativistic: $\\gamma mv$')
ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax3.annotate('p = mc', (0.1, 1.1), fontsize=10)

ax3.set_xlabel('v/c', fontsize=12)
ax3.set_ylabel('Momentum (units of mc)', fontsize=12)
ax3.set_title('Momentum vs Velocity', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 10)

# ========== Plot 4: Threshold Energy Calculation ==========
ax4 = axes[1, 1]

# Threshold energy for pair production: gamma -> e+ e-
# and for antiproton production: p + p -> p + p + p + pbar

# For general reaction with target at rest:
# E_threshold = (sum of final masses)^2 * c^2 / (2 * m_target)

# Plot threshold energy vs mass created
m_created = np.linspace(0.1, 5000, 500)  # MeV

# Assuming proton target at rest
m_target = m_p
# Minimum is when all final products are at rest in CM frame
# s = (E_beam + m_target)^2 - p_beam^2 = (sum m_final)^2
# For creating mass M in addition to original particles:
# E_threshold = (m_target + m_created)^2 / (2*m_target) - m_target/2
# This simplifies for creating just M: E_thresh = (2*m_target + M)^2 / (2*m_target) - m_target

# Let's plot the beam energy needed to create mass M
# Assuming reaction: p + p -> p + p + X (where X has mass M)
M_created = m_created  # mass of new particle(s)
E_cm_min = 2*m_p + M_created  # minimum CM energy
# s = E_cm^2, and s = 2*m_p*(E_beam + m_p) for target at rest
# E_beam = (E_cm_min^2)/(2*m_p) - m_p
E_beam_threshold = (E_cm_min**2)/(2*m_p) - m_p
K_beam_threshold = E_beam_threshold - m_p

ax4.plot(m_created, K_beam_threshold, 'b-', linewidth=2)

# Mark specific thresholds
markers = [
    ('$e^+e^-$', 2*m_e, 'red'),
    ('$\\pi^0$', m_pi0, 'green'),
    ('$\\mu^+\\mu^-$', 2*m_mu, 'purple'),
    ('$p\\bar{p}$', 2*m_p, 'orange'),
]

for label, mass, color in markers:
    E_cm = 2*m_p + mass
    K_th = (E_cm**2)/(2*m_p) - 2*m_p
    ax4.axvline(x=mass, color=color, linestyle='--', alpha=0.5)
    ax4.plot(mass, K_th, 'o', color=color, markersize=10)
    ax4.annotate(f'{label}\n{K_th:.0f} MeV', (mass, K_th),
                 textcoords='offset points', xytext=(10, 10), fontsize=9)

ax4.set_xlabel('Mass Created (MeV/c²)', fontsize=12)
ax4.set_ylabel('Threshold Beam KE (MeV)', fontsize=12)
ax4.set_title('Threshold Energy for Particle Production\n(p + p → p + p + X)', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 3000)
ax4.set_ylim(0, 8000)

plt.tight_layout()
plt.savefig('day_220_relativistic_mechanics.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Numerical Examples ==========
print("=" * 60)
print("RELATIVISTIC MECHANICS CALCULATIONS")
print("=" * 60)

# Example 1: Electron at different energies
print("\n--- Electron Properties at Various Kinetic Energies ---")
print(f"{'KE (MeV)':<12} {'E (MeV)':<12} {'p (MeV/c)':<12} {'v/c':<12} {'gamma':<10}")
print("-" * 60)

KE_values = [0.001, 0.1, 0.511, 1, 10, 100, 1000]
for KE in KE_values:
    E = KE + m_e
    p = np.sqrt(E**2 - m_e**2)
    v_over_c = p / E
    g = E / m_e
    print(f"{KE:<12.3f} {E:<12.3f} {p:<12.3f} {v_over_c:<12.6f} {g:<10.3f}")

# Example 2: Pion decay
print("\n" + "=" * 60)
print("PION DECAY: π⁺ → μ⁺ + ν_μ")
print("=" * 60)

m_pi_plus = 139.6  # MeV
m_muon = 105.7
m_neutrino = 0  # effectively

# In pion rest frame
# Energy conservation: m_pi = E_mu + E_nu
# Momentum conservation: p_mu = p_nu = p

# E_nu = p (since m_nu = 0)
# E_mu^2 = p^2 + m_mu^2

# (m_pi - p)^2 = p^2 + m_mu^2
# m_pi^2 - 2*m_pi*p = m_mu^2
# p = (m_pi^2 - m_mu^2) / (2*m_pi)

p_decay = (m_pi_plus**2 - m_muon**2) / (2 * m_pi_plus)
E_nu = p_decay
E_mu = m_pi_plus - E_nu
v_mu = p_decay / E_mu

print(f"\nMasses: m_π = {m_pi_plus} MeV, m_μ = {m_muon} MeV")
print(f"\nIn pion rest frame:")
print(f"  Muon energy: E_μ = {E_mu:.2f} MeV")
print(f"  Muon momentum: p_μ = {p_decay:.2f} MeV/c")
print(f"  Muon velocity: v_μ = {v_mu:.4f}c")
print(f"  Muon kinetic energy: K_μ = {E_mu - m_muon:.2f} MeV")
print(f"  Neutrino energy: E_ν = {E_nu:.2f} MeV")

# Example 3: Compton scattering
print("\n" + "=" * 60)
print("COMPTON SCATTERING")
print("=" * 60)

lambda_i = 0.01  # nm
angles = [30, 60, 90, 120, 180]
lambda_C = 0.002426  # nm (Compton wavelength)

print(f"\nIncident wavelength: λ = {lambda_i} nm")
print(f"Compton wavelength: λ_C = {lambda_C:.6f} nm")
print(f"\n{'Angle (°)':<12} {'Δλ (nm)':<12} {'λ_f (nm)':<12} {'E_f/E_i':<12}")
print("-" * 50)

for theta in angles:
    delta_lambda = lambda_C * (1 - np.cos(np.radians(theta)))
    lambda_f = lambda_i + delta_lambda
    E_ratio = lambda_i / lambda_f
    print(f"{theta:<12} {delta_lambda:<12.6f} {lambda_f:<12.6f} {E_ratio:<12.4f}")

# Example 4: Invariant mass
print("\n" + "=" * 60)
print("INVARIANT MASS RECONSTRUCTION")
print("=" * 60)

# Two photons from pi0 decay
E1, E2 = 80, 60  # MeV
theta = 25  # degrees opening angle

# Invariant mass squared
M_squared = 2 * E1 * E2 * (1 - np.cos(np.radians(theta)))
M = np.sqrt(M_squared)

print(f"\nTwo photon system:")
print(f"  E_1 = {E1} MeV, E_2 = {E2} MeV")
print(f"  Opening angle: θ = {theta}°")
print(f"  Invariant mass: M = {M:.1f} MeV/c²")
print(f"  (π⁰ mass = {m_pi0} MeV/c²)")

print("\n" + "=" * 60)
print("Day 220: Relativistic Mechanics Complete")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\mathbf{p} = \gamma m\mathbf{v}$ | Relativistic momentum |
| $E = \gamma mc^2$ | Relativistic total energy |
| $E_0 = mc^2$ | Rest energy |
| $K = (\gamma - 1)mc^2$ | Relativistic kinetic energy |
| $E^2 = (pc)^2 + (mc^2)^2$ | Energy-momentum relation |
| $E = pc$ | Massless particle energy |
| $M^2c^4 = (\sum E_i)^2 - |\sum\mathbf{p}_i|^2c^2$ | Invariant mass of system |

### Main Takeaways

1. **Relativistic momentum** $\mathbf{p} = \gamma m\mathbf{v}$ diverges as $v \to c$
2. **Mass-energy equivalence** $E = mc^2$ reveals mass as concentrated energy
3. **Energy-momentum relation** is a Lorentz invariant: $p_\mu p^\mu = -m^2c^2$
4. **Massless particles** travel at speed $c$ and have $E = pc$
5. **4-momentum conservation** governs relativistic collisions and decays
6. **Threshold energy** calculations determine minimum energies for particle creation

---

## Daily Checklist

- [ ] I can derive and apply the relativistic energy-momentum relation
- [ ] I understand mass-energy equivalence and its implications
- [ ] I can analyze relativistic collisions using 4-momentum
- [ ] I can calculate invariant mass and threshold energies
- [ ] I understand the properties of massless particles
- [ ] I can connect to the Dirac equation and antimatter

---

## Preview: Day 221

Tomorrow we explore how **electric and magnetic fields transform** between reference frames. We'll discover that $\mathbf{E}$ and $\mathbf{B}$ are not separate entities but components of a single electromagnetic field.

---

*"It followed from the special theory of relativity that mass and energy are both but different manifestations of the same thing — a somewhat unfamiliar conception for the average mind."*
— Albert Einstein

---

**Next:** Day 221 — Transformation of E&M Fields
