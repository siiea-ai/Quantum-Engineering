# Day 425: Finite Spherical Well

## Overview
**Day 425** | Year 1, Month 16, Week 61 | Bound States with Finite Depth

Today we tackle the finite spherical well, where the potential has finite depth V₀. This allows wavefunction penetration and introduces the concept of a minimum well depth for bound states.

---

## Learning Objectives

By the end of today, you will be able to:
1. Set up matching conditions at r = a for finite wells
2. Derive the transcendental equation for bound states
3. Understand when bound states exist
4. Calculate the critical potential depth for the first bound state
5. Analyze the continuum threshold
6. Connect to nuclear and atomic physics

---

## Core Content

### The Finite Spherical Well

$$V(r) = \begin{cases} -V_0 & r < a \\ 0 & r \geq a \end{cases}$$

with V₀ > 0 (attractive well).

### Interior Region (r < a)

For bound states with E < 0, define:
$$k^2 = \frac{2m(V_0 + E)}{\hbar^2} = \frac{2m(V_0 - |E|)}{\hbar^2}$$

Solution (regular at origin):
$$R_{\text{in}}(r) = A j_l(kr)$$

### Exterior Region (r > a)

Define:
$$\kappa^2 = \frac{2m|E|}{\hbar^2}$$

The solution must decay as r → ∞. The appropriate functions are modified spherical Bessel functions, but for bound states:

$$R_{\text{out}}(r) = B \frac{e^{-\kappa r}}{r} \cdot f_l(\kappa r)$$

For l = 0:
$$R_{\text{out}}(r) = B \frac{e^{-\kappa r}}{r}$$

General form using spherical Hankel function:
$$R_{\text{out}}(r) = B h_l^{(1)}(i\kappa r)$$

### Matching Conditions at r = a

**Continuity of wavefunction:**
$$R_{\text{in}}(a) = R_{\text{out}}(a)$$

**Continuity of logarithmic derivative:**
$$\frac{R'_{\text{in}}(a)}{R_{\text{in}}(a)} = \frac{R'_{\text{out}}(a)}{R_{\text{out}}(a)}$$

### The l = 0 Case (s-waves)

Interior: R_in = A sin(kr)/(kr)
Exterior: R_out = B e^{-κr}/r

Matching logarithmic derivatives:
$$k \cot(ka) - \frac{1}{a} = -\kappa - \frac{1}{a}$$
$$\boxed{k \cot(ka) = -\kappa}$$

### Dimensionless Form

Define:
$$z = ka, \quad z_0 = \frac{a}{\hbar}\sqrt{2mV_0}$$

Then:
$$\kappa a = \sqrt{z_0^2 - z^2}$$

And the quantization condition becomes:
$$\boxed{z \cot z = -\sqrt{z_0^2 - z^2}}$$

### Critical Depth for First Bound State

The first s-wave bound state appears when z₀ = π/2.

$$V_0^{\text{crit}} = \frac{\hbar^2 \pi^2}{8ma^2}$$

For weaker wells, no bound states exist!

### General l Condition

For general l, the critical condition is related to the first zero of j_l.

| l | First zero of j_l | Critical z₀ |
|---|-------------------|-------------|
| 0 | 3.14 (π) | π/2 ≈ 1.57 |
| 1 | 4.49 | ~4.49/2 ≈ 2.25 |
| 2 | 5.76 | ~2.88 |

Higher l requires deeper wells for bound states.

### Number of Bound States

For s-waves, the number of bound states is:
$$n_{\text{bound}} = \left\lfloor \frac{z_0}{\pi} + \frac{1}{2} \right\rfloor$$

---

## Quantum Computing Connection

### Quantum Simulation of Scattering

The finite well is a prototype for:
- **Nuclear scattering:** Neutron-nucleus interactions
- **Molecular binding:** Simplified molecular potentials
- **Quantum chemistry:** Testing VQE algorithms

### Threshold Behavior

The critical depth for bound states connects to:
- Universal behavior near thresholds
- Efimov physics in ultracold atoms

---

## Worked Examples

### Example 1: Critical Depth

**Problem:** A particle of mass m in a well of radius a = 1 fm. What is the minimum V₀ to bind an l = 0 state?

**Solution:**
$$V_0^{\text{crit}} = \frac{\hbar^2 \pi^2}{8ma^2}$$

For a nucleon (m ≈ 940 MeV/c²):
Using ℏc ≈ 197 MeV·fm:
$$V_0^{\text{crit}} = \frac{(197)^2 \pi^2}{8 \times 940 \times 1^2} \approx 51 \text{ MeV}$$

### Example 2: Bound State Energy

**Problem:** For z₀ = 4 (twice critical), find the s-wave bound state energy.

**Solution:**
Solve z cot(z) = -√(16 - z²).

By numerical methods or graphically: z ≈ 2.47

$$E = -V_0 + \frac{\hbar^2 k^2}{2m} = -V_0\left(1 - \frac{z^2}{z_0^2}\right)$$
$$= -V_0\left(1 - \frac{6.1}{16}\right) = -0.62 V_0$$

The bound state lies at E ≈ -0.62 V₀.

### Example 3: Wavefunction Penetration

**Problem:** Find the decay length outside the well for the ground state with z₀ = 4.

**Solution:**
$$\kappa a = \sqrt{z_0^2 - z^2} = \sqrt{16 - 6.1} \approx 3.15$$
$$\kappa = \frac{3.15}{a}$$

Decay length = 1/κ = a/3.15 ≈ 0.32 a

The wavefunction penetrates about 1/3 of the well radius outside.

---

## Practice Problems

### Direct Application
1. What is z₀ for V₀ = 4 V₀^crit?
2. How many s-wave bound states exist for z₀ = 5?
3. Write the exterior wavefunction explicitly for l = 0.

### Intermediate
4. Derive the matching condition for l = 1.
5. Show that the ground state energy approaches -V₀ as z₀ → ∞.
6. Calculate the normalization of the wavefunction including both regions.

### Challenging
7. Prove that there's always at least one bound state for any attractive potential in 1D, but not necessarily in 3D.
8. Analyze the behavior of the wavefunction as E → 0⁻ (threshold).

---

## Computational Lab

```python
"""
Day 425: Finite Spherical Well
Finding bound states through graphical and numerical methods
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import spherical_jn

# Graphical solution for s-wave bound states
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Graphical solution
ax = axes[0, 0]

z = np.linspace(0.01, 4*np.pi, 1000)

# Left-hand side: z cot(z)
lhs = z / np.tan(z)
# Mask discontinuities
lhs[np.abs(np.diff(lhs, prepend=lhs[0])) > 10] = np.nan

ax.plot(z, lhs, 'b-', linewidth=2, label='z cot(z)')

# Right-hand side for different z0
for z0 in [2, 4, 6, 8]:
    rhs = -np.sqrt(z0**2 - z**2)
    rhs[z > z0] = np.nan
    ax.plot(z, rhs, '--', linewidth=2, label=f'z₀ = {z0}')

ax.set_xlabel('z = ka', fontsize=12)
ax.set_ylabel('Function value', fontsize=12)
ax.set_title('Graphical Solution: z cot(z) = -√(z₀² - z²)', fontsize=14)
ax.set_xlim(0, 12)
ax.set_ylim(-15, 5)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Mark critical value
ax.axvline(x=np.pi/2, color='red', linestyle=':', alpha=0.7)
ax.text(np.pi/2 + 0.1, -13, 'z₀ᶜʳⁱᵗ = π/2', fontsize=10, color='red')

# Panel 2: Number of bound states vs z0
ax = axes[0, 1]

z0_range = np.linspace(0.1, 12, 200)
n_bound = []

def count_bound_states(z0, l=0):
    """Count bound states by finding roots"""
    count = 0
    z_prev = 0.01
    for z_upper in np.linspace(0.1, z0 - 0.01, 100):
        try:
            def f(z):
                if z >= z0:
                    return 0
                return z / np.tan(z) + np.sqrt(z0**2 - z**2)

            if f(z_prev) * f(z_upper) < 0:
                root = brentq(f, z_prev, z_upper)
                count += 1
                z_prev = z_upper + 0.01
            else:
                z_prev = z_upper
        except:
            z_prev = z_upper
    return count

for z0 in z0_range:
    try:
        n = count_bound_states(z0)
        n_bound.append(n)
    except:
        n_bound.append(0)

ax.plot(z0_range, n_bound, 'b-', linewidth=2)
ax.axvline(x=np.pi/2, color='r', linestyle='--', label='First bound state')
ax.axvline(x=3*np.pi/2, color='g', linestyle='--', label='Second bound state')
ax.axvline(x=5*np.pi/2, color='orange', linestyle='--', label='Third bound state')
ax.set_xlabel('z₀ = a√(2mV₀)/ℏ', fontsize=12)
ax.set_ylabel('Number of s-wave bound states', fontsize=12)
ax.set_title('Bound State Count vs Well Strength', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Wavefunction for specific z0
ax = axes[1, 0]

z0 = 4.0  # Well strength parameter

def find_bound_states_z(z0):
    """Find bound state z values"""
    roots = []
    z_search = np.linspace(0.01, z0 - 0.01, 500)
    for i in range(len(z_search) - 1):
        z1, z2 = z_search[i], z_search[i+1]
        f1 = z1 / np.tan(z1) + np.sqrt(z0**2 - z1**2)
        f2 = z2 / np.tan(z2) + np.sqrt(z0**2 - z2**2)
        if f1 * f2 < 0:
            try:
                root = brentq(lambda z: z/np.tan(z) + np.sqrt(z0**2 - z**2), z1, z2)
                roots.append(root)
            except:
                pass
    return roots

# Find bound state
z_bound = find_bound_states_z(z0)
print(f"For z₀ = {z0}: Bound state at z = {z_bound}")

if z_bound:
    z_val = z_bound[0]
    kappa_a = np.sqrt(z0**2 - z_val**2)

    # Radial coordinate
    r_in = np.linspace(0.01, 1, 100)  # r/a inside
    r_out = np.linspace(1, 3, 100)    # r/a outside

    # Interior: sin(kr)/kr = sin(z*r/a)/(z*r/a)
    R_in = np.sin(z_val * r_in) / (z_val * r_in)

    # Match at r = a
    R_at_a = np.sin(z_val) / z_val

    # Exterior: A * exp(-kappa*r) / r, matched at a
    R_out = R_at_a * np.exp(-kappa_a * (r_out - 1)) / r_out

    ax.plot(r_in, R_in, 'b-', linewidth=2, label='Interior')
    ax.plot(r_out, R_out, 'r-', linewidth=2, label='Exterior')
    ax.axvline(x=1, color='k', linestyle='--', alpha=0.5, label='r = a')
    ax.fill_between([0, 1], [-0.5, -0.5], [1.2, 1.2], alpha=0.1, color='blue')

ax.set_xlabel('r/a', fontsize=12)
ax.set_ylabel('R(r) (normalized)', fontsize=12)
ax.set_title(f'Bound State Wavefunction (z₀ = {z0})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3)
ax.set_ylim(-0.5, 1.2)

# Panel 4: Binding energy vs well depth
ax = axes[1, 1]

z0_range = np.linspace(np.pi/2 + 0.1, 10, 100)
binding_energies = []

for z0 in z0_range:
    z_bound = find_bound_states_z(z0)
    if z_bound:
        z_val = z_bound[0]
        # E/V0 = -1 + z²/z0²
        E_ratio = -1 + z_val**2 / z0**2
        binding_energies.append(E_ratio)
    else:
        binding_energies.append(0)

ax.plot(z0_range, binding_energies, 'b-', linewidth=2)
ax.axhline(y=-1, color='r', linestyle='--', alpha=0.7, label='Bottom of well')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('z₀', fontsize=12)
ax.set_ylabel('E / V₀', fontsize=12)
ax.set_title('Ground State Energy vs Well Strength', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.1, 0.1)

plt.tight_layout()
plt.savefig('day425_finite_well.png', dpi=150)
plt.show()

# Physical examples
print("\n=== Physical Examples ===")
print("\nDeuterium nucleus (proton + neutron in square well):")
print("  Well radius a ≈ 2.1 fm")
print("  Well depth V₀ ≈ 35 MeV")
print("  Binding energy E_B ≈ 2.2 MeV (weakly bound!)")

hbar_c = 197  # MeV·fm
m_nucleon = 940  # MeV/c²
a_deuteron = 2.1  # fm
V0_deuteron = 35  # MeV

z0_deuteron = a_deuteron * np.sqrt(2 * m_nucleon * V0_deuteron) / hbar_c
print(f"  z₀ = {z0_deuteron:.2f}")

print("\n  This gives only ONE bound state (the deuteron ground state)")
print("  No excited states exist - the deuteron is a barely-bound system!")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Interior wavenumber | k² = 2m(V₀ + E)/ℏ² |
| Exterior decay | κ² = 2m|E|/ℏ² |
| s-wave condition | k cot(ka) = -κ |
| Critical depth | V₀^crit = ℏ²π²/(8ma²) |
| Bound state count | n = ⌊z₀/π + 1/2⌋ |

### Key Insights

1. **Finite wells** don't always have bound states
2. **Critical depth** required: V₀ > ℏ²π²/(8ma²) for first s-wave
3. **Penetration** outside well is exponential decay
4. **Higher l** requires deeper wells to bind
5. **Deuteron** is example of barely-bound system

---

## Daily Checklist

- [ ] I can set up matching conditions at the well boundary
- [ ] I understand the transcendental equation for bound states
- [ ] I can calculate the critical depth for binding
- [ ] I know how to count bound states from z₀
- [ ] I can describe wavefunction penetration
- [ ] I see connections to nuclear physics

---

## Preview: Day 426

Tomorrow we solve the **3D isotropic harmonic oscillator**, a fundamental model for molecular vibrations and quantum optics. The algebraic structure reveals beautiful degeneracies.

---

**Next:** [Day_426_Saturday.md](Day_426_Saturday.md) — 3D Harmonic Oscillator
