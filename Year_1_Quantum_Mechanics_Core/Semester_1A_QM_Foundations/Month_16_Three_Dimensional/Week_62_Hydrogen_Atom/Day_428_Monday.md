# Day 428: Coulomb Problem Setup

## Overview
**Day 428** | Year 1, Month 16, Week 62 | Setting Up the Hydrogen Atom

Today we formulate the quantum mechanical hydrogen atom problem — the two-body Coulomb problem that laid the foundation for our understanding of atomic structure.

---

## Learning Objectives

By the end of today, you will be able to:
1. Write the Hamiltonian for the hydrogen atom
2. Separate center-of-mass and relative motion
3. Define the reduced mass and its importance
4. Express the problem in natural atomic units
5. Identify the connection to the classical Kepler problem
6. Understand why hydrogen is exactly solvable

---

## Core Content

### The Two-Body Problem

**Hydrogen atom:** One electron (mass m_e, charge -e) bound to one proton (mass m_p, charge +e).

**Full Hamiltonian:**
$$\hat{H} = \frac{\hat{\mathbf{p}}_e^2}{2m_e} + \frac{\hat{\mathbf{p}}_p^2}{2m_p} - \frac{e^2}{|\mathbf{r}_e - \mathbf{r}_p|}$$

### Center-of-Mass Separation

Define:
- **Center of mass:** R = (m_e r_e + m_p r_p)/(m_e + m_p)
- **Relative position:** r = r_e - r_p
- **Total mass:** M = m_e + m_p
- **Reduced mass:** μ = m_e m_p/(m_e + m_p)

The Hamiltonian separates:
$$\hat{H} = \underbrace{\frac{\hat{\mathbf{P}}^2}{2M}}_{\text{CM motion}} + \underbrace{\frac{\hat{\mathbf{p}}^2}{2\mu} - \frac{e^2}{r}}_{\text{Relative motion}}$$

### The Reduced Mass

$$\mu = \frac{m_e m_p}{m_e + m_p} = \frac{m_e}{1 + m_e/m_p} \approx m_e\left(1 - \frac{m_e}{m_p}\right)$$

Since m_p/m_e ≈ 1836:
$$\mu \approx 0.99946 \, m_e$$

The reduced mass is very close to the electron mass!

### The Relative Motion Hamiltonian

Ignoring CM motion (free particle), the bound state problem is:
$$\boxed{\hat{H} = \frac{\hat{p}^2}{2\mu} - \frac{e^2}{r}}$$

### Schrödinger Equation

$$\left[-\frac{\hbar^2}{2\mu}\nabla^2 - \frac{e^2}{r}\right]\psi(r, \theta, \phi) = E\psi(r, \theta, \phi)$$

### Atomic Units

Define natural units where ℏ = m_e = e = 1:

| Quantity | Atomic Unit | SI Value |
|----------|-------------|----------|
| Length | a₀ = ℏ²/(m_e e²) | 0.529 Å |
| Energy | E_H = e²/a₀ = m_e e⁴/ℏ² | 27.2 eV |
| Time | ℏ/E_H | 24.2 as |

In atomic units:
$$\hat{H} = -\frac{1}{2}\nabla^2 - \frac{1}{r}$$

### The Bohr Radius

$$\boxed{a_0 = \frac{\hbar^2}{m_e e^2} = \frac{4\pi\varepsilon_0\hbar^2}{m_e e^2} = 0.529 \text{ Å}}$$

This sets the natural length scale for atomic physics.

### The Rydberg Energy

$$\boxed{E_R = \frac{m_e e^4}{2\hbar^2} = \frac{e^2}{2a_0} = 13.6 \text{ eV}}$$

This is the ionization energy of hydrogen from the ground state.

### Separation in Spherical Coordinates

Following Week 61:
$$\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)$$

The radial equation:
$$\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{dR}{dr}\right) + \frac{2\mu}{\hbar^2}\left[E + \frac{e^2}{r} - \frac{\hbar^2 l(l+1)}{2\mu r^2}\right]R = 0$$

---

## Quantum Computing Connection

### VQE for Hydrogen

The hydrogen atom is the simplest test case for:
- **Variational Quantum Eigensolver (VQE)**
- **Quantum Phase Estimation**
- Benchmarking quantum chemistry algorithms

### Atomic Qubits

Hydrogen-like atoms form qubits:
- Ground and first excited state as |0⟩ and |1⟩
- Trapped ions (H-like) use similar energy structure

---

## Worked Examples

### Example 1: Reduced Mass Correction

**Problem:** Calculate the energy correction from using μ instead of m_e.

**Solution:**
$$\frac{\mu}{m_e} = \frac{1}{1 + m_e/m_p} = \frac{1}{1 + 1/1836} = 0.99946$$

Energy scales as mass, so:
$$\frac{E_\mu}{E_{m_e}} = 0.99946$$

Correction: 0.054% reduction in binding energy.

For ground state: ΔE = 13.6 eV × 0.00054 = 7.3 meV

### Example 2: Atomic Units

**Problem:** Express the hydrogen Hamiltonian in atomic units.

**Solution:**
With ℏ = m_e = e = 4πε₀ = 1:

Kinetic energy: p²/(2m_e) → (1/2)∇² in units of E_H
Potential: -e²/(4πε₀r) → -1/r in units of E_H/a₀ × a₀ = E_H

$$\hat{H} = -\frac{1}{2}\nabla^2 - \frac{1}{r}$$

All quantities are dimensionless numbers!

### Example 3: Classical Correspondence

**Problem:** What is the classical orbit for E = -E_R (ground state energy)?

**Solution:**
For a classical Kepler orbit:
$$E = -\frac{e^2}{2a}$$

where a is the semi-major axis.

$$a = \frac{e^2}{2|E|} = \frac{e^2}{2E_R} = \frac{e^2 \cdot 2\hbar^2}{2 m_e e^4} = \frac{\hbar^2}{m_e e^2} = a_0$$

The ground state energy corresponds to a circular orbit at the Bohr radius!

---

## Practice Problems

### Direct Application
1. Calculate the Bohr radius in nm.
2. What is E_R in units of kJ/mol?
3. Express the ground state energy in atomic units.

### Intermediate
4. Calculate the reduced mass for deuterium (proton replaced by deuteron).
5. How does the Bohr radius change for muonic hydrogen (muon instead of electron)?
6. Show that E_R = (1/2)α² m_e c² where α is the fine structure constant.

### Challenging
7. Derive the relation between atomic units and SI units.
8. Show that the virial theorem gives ⟨T⟩ = -E and ⟨V⟩ = 2E for hydrogen.

---

## Computational Lab

```python
"""
Day 428: Coulomb Problem Setup
Physical constants and atomic units
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, e, epsilon_0, physical_constants

# Fundamental constants
print("=== Fundamental Constants for Hydrogen ===\n")

# Bohr radius
a0 = 4 * np.pi * epsilon_0 * hbar**2 / (m_e * e**2)
print(f"Bohr radius: a₀ = {a0*1e10:.4f} Å = {a0*1e9:.4f} nm")

# Rydberg energy
E_R = m_e * e**4 / (2 * (4 * np.pi * epsilon_0)**2 * hbar**2)
E_R_eV = E_R / e
print(f"Rydberg energy: E_R = {E_R_eV:.4f} eV")

# Hartree energy
E_H = 2 * E_R
E_H_eV = E_H / e
print(f"Hartree energy: E_H = {E_H_eV:.4f} eV")

# Fine structure constant
alpha = e**2 / (4 * np.pi * epsilon_0 * hbar * 3e8)
print(f"Fine structure constant: α = {alpha:.6f} = 1/{1/alpha:.2f}")

# Reduced mass
m_p = physical_constants['proton mass'][0]
mu = m_e * m_p / (m_e + m_p)
print(f"\nReduced mass: μ/m_e = {mu/m_e:.6f}")
print(f"Mass ratio: m_p/m_e = {m_p/m_e:.2f}")

# Atomic unit conversion table
print("\n=== Atomic Units Conversion ===")
print("┌─────────────┬────────────────────────────────┐")
print("│ Quantity    │ SI Value                       │")
print("├─────────────┼────────────────────────────────┤")
print(f"│ Length (a₀) │ {a0:.5e} m              │")
print(f"│ Energy (E_H)│ {E_H:.5e} J = {E_H_eV:.3f} eV   │")
print(f"│ Time (ℏ/E_H)│ {hbar/E_H:.5e} s             │")
print(f"│ Velocity    │ {alpha * 3e8:.5e} m/s            │")
print("└─────────────┴────────────────────────────────┘")

# Plot the Coulomb potential
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Coulomb potential and energy levels
ax = axes[0]
r = np.linspace(0.1, 15, 500)  # in units of a0

V_coulomb = -1/r  # in units of E_H

ax.plot(r, V_coulomb, 'b-', linewidth=2, label='V(r) = -1/r')
ax.axhline(y=0, color='k', linewidth=0.5)

# Add energy levels
for n in range(1, 6):
    E_n = -0.5 / n**2  # in Hartree
    r_classical = 2 * n**2  # Classical turning point
    ax.hlines(E_n, 0.5, min(r_classical, 14), colors='red', linewidth=1.5, alpha=0.7)
    ax.text(14.5, E_n, f'n={n}', fontsize=10, va='center')

ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('Energy / E_H', fontsize=12)
ax.set_title('Coulomb Potential and Energy Levels', fontsize=14)
ax.set_xlim(0, 15)
ax.set_ylim(-0.6, 0.1)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Effective potential for different l
ax = axes[1]

for l in range(4):
    V_eff = -1/r + l*(l+1)/(2*r**2)
    ax.plot(r, V_eff, linewidth=2, label=f'l = {l}')

ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('r / a₀', fontsize=12)
ax.set_ylabel('V_eff / E_H', fontsize=12)
ax.set_title('Effective Potential for Different l', fontsize=14)
ax.set_xlim(0, 15)
ax.set_ylim(-0.6, 0.5)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day428_coulomb_setup.png', dpi=150)
plt.show()

# Isotope effects
print("\n=== Isotope Effects ===")

isotopes = [
    ('Hydrogen (H)', m_p),
    ('Deuterium (D)', 2 * m_p),
    ('Tritium (T)', 3 * m_p),
    ('Muonic H', 207 * m_e),  # muon mass ≈ 207 m_e
]

print("\nIsotope         μ/m_e      a₀(isotope)/a₀(H)   E_R(isotope)/E_R(H)")
print("-" * 70)

mu_H = m_e * m_p / (m_e + m_p)

for name, m_nucleus in isotopes:
    mu_iso = m_e * m_nucleus / (m_e + m_nucleus)
    a0_ratio = mu_H / mu_iso  # a₀ ∝ 1/μ
    E_ratio = mu_iso / mu_H   # E ∝ μ

    if 'Muonic' in name:
        # For muonic, replace electron mass with muon mass
        m_muon = 207 * m_e
        mu_muonic = m_muon * m_p / (m_muon + m_p)
        a0_ratio = (m_e / mu_H) / (m_muon / mu_muonic)
        E_ratio = (mu_muonic / m_muon) / (mu_H / m_e)

        print(f"{name:15s} {mu_muonic/m_e:8.4f}      {1/207:.6f}            {207:.1f}")
    else:
        print(f"{name:15s} {mu_iso/m_e:8.6f}   {a0_ratio:.6f}            {E_ratio:.6f}")

print("\nNote: Muonic hydrogen has a much smaller Bohr radius (probes nuclear structure)")
```

---

## Summary

### Key Formulas

| Quantity | Formula | Value |
|----------|---------|-------|
| Bohr radius | a₀ = ℏ²/(m_e e²) | 0.529 Å |
| Rydberg energy | E_R = m_e e⁴/(2ℏ²) | 13.6 eV |
| Reduced mass | μ = m_e m_p/(m_e + m_p) | 0.9995 m_e |
| Hamiltonian | H = p²/(2μ) - e²/r | — |
| Atomic units | ℏ = m_e = e = 1 | — |

### Key Insights

1. **Two-body problem** reduces to one-body with reduced mass
2. **Atomic units** simplify calculations enormously
3. **Bohr radius** sets natural length scale
4. **Rydberg energy** is ionization energy
5. **Reduced mass** gives small isotope effects

---

## Daily Checklist

- [ ] I can write the hydrogen Hamiltonian
- [ ] I understand center-of-mass separation
- [ ] I can calculate the reduced mass
- [ ] I know the Bohr radius and Rydberg energy
- [ ] I can convert to atomic units
- [ ] I see the classical correspondence

---

## Preview: Day 429

Tomorrow we solve the radial equation for hydrogen, introducing the associated Laguerre polynomials and deriving the quantization of energy.

---

**Next:** [Day_429_Tuesday.md](Day_429_Tuesday.md) — Radial Solution
