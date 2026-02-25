# Day 480: Pauli Exclusion Principle

## Overview
**Day 480** | Year 1, Month 18, Week 69 | The Foundation of Atomic Structure

Today we explore the Pauli exclusion principle—one of the most consequential results in physics, responsible for the stability of matter and the periodic table.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Statement and proof |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Physical consequences |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Atomic structure simulation |

---

## Learning Objectives

By the end of today, you will be able to:
1. State the Pauli exclusion principle precisely
2. Derive it from antisymmetry of fermionic wave functions
3. Explain its role in atomic structure
4. Understand degeneracy pressure and stellar evolution
5. Connect to the stability of matter
6. Apply it to qubit systems

---

## Core Content

### The Pauli Exclusion Principle

**Statement:** No two identical fermions can occupy the same quantum state.

**Formal statement:** If |ψ_α⟩ and |ψ_β⟩ are fermionic states with α = β, then:
$$|\Psi\rangle = |\psi_\alpha\rangle|\psi_\alpha\rangle = 0$$

### Derivation from Antisymmetry

For two identical fermions in the same state |α⟩:
$$|\Psi_F\rangle = \frac{1}{\sqrt{2}}(|\alpha\rangle_1|\alpha\rangle_2 - |\alpha\rangle_2|\alpha\rangle_1) = 0$$

The antisymmetric wave function **vanishes identically**.

### For Atoms: Four Quantum Numbers

For electrons in an atom, a quantum state is specified by:
- **n**: principal quantum number
- **ℓ**: orbital angular momentum
- **mℓ**: magnetic quantum number
- **ms**: spin projection (±1/2)

$$\boxed{\text{No two electrons can have the same } (n, \ell, m_\ell, m_s)}$$

### Consequences for Orbital Occupancy

Each spatial orbital (n, ℓ, mℓ) can hold:
- 1 spin-up electron (ms = +1/2)
- 1 spin-down electron (ms = -1/2)
- **Maximum 2 electrons per orbital**

Each subshell (n, ℓ) can hold:
$$2(2\ell + 1) \text{ electrons}$$

| Subshell | ℓ | Orbitals | Max electrons |
|----------|---|----------|---------------|
| s | 0 | 1 | 2 |
| p | 1 | 3 | 6 |
| d | 2 | 5 | 10 |
| f | 3 | 7 | 14 |

---

## Building the Periodic Table

### The Aufbau Principle

Electrons fill orbitals in order of increasing energy:
$$1s < 2s < 2p < 3s < 3p < 4s < 3d < 4p < 5s < 4d < \cdots$$

### Example: First 10 Elements

| Z | Element | Configuration |
|---|---------|---------------|
| 1 | H | 1s¹ |
| 2 | He | 1s² |
| 3 | Li | 1s² 2s¹ |
| 4 | Be | 1s² 2s² |
| 5 | B | 1s² 2s² 2p¹ |
| 6 | C | 1s² 2s² 2p² |
| 7 | N | 1s² 2s² 2p³ |
| 8 | O | 1s² 2s² 2p⁴ |
| 9 | F | 1s² 2s² 2p⁵ |
| 10 | Ne | 1s² 2s² 2p⁶ |

### Chemical Properties

**Key insight:** Chemical properties are determined by **valence electrons** (outermost shell).

The periodic table structure emerges directly from:
1. Pauli exclusion principle
2. Aufbau principle
3. Hund's rules

---

## Degeneracy Pressure

### Fermi Gas

For N fermions in a box, filled states form a **Fermi sea**:
- All states up to Fermi energy E_F occupied
- States above E_F empty

### Fermi Energy

For 3D free electrons:
$$E_F = \frac{\hbar^2}{2m}\left(\frac{3\pi^2 N}{V}\right)^{2/3}$$

### Degeneracy Pressure

Even at T = 0, fermions exert **pressure** from the exclusion principle:
$$P = \frac{2}{5}nE_F = \frac{\hbar^2}{5m}\left(\frac{3\pi^2}{V}\right)^{2/3}N^{5/3}$$

This **does not require temperature**—it's purely quantum mechanical!

### Stellar Physics

**White dwarfs:** Supported by electron degeneracy pressure
- Chandrasekhar limit: M < 1.4 M☉

**Neutron stars:** Supported by neutron degeneracy pressure
- After electron degeneracy fails → supernova → neutron star

---

## Stability of Matter

### Why Matter Doesn't Collapse

Without Pauli exclusion:
- All electrons would fall into lowest energy state
- Atoms would collapse to nuclear size
- Matter would be incredibly dense

### Lieb-Thirring Inequality

The total energy of N electrons is bounded below:
$$E \geq -C \cdot Z^2 N$$

for some constant C, thanks to the exclusion principle.

**Result:** Matter is stable—volume scales with N, not collapsing.

---

## Quantum Computing Connection

### Qubit Distinctiveness

**Standard qubits are distinguishable**—labeled by position/index.

But fermionic systems (molecular simulation) must respect exclusion:
- Can't have two electrons in same spin-orbital
- Occupation restricted to 0 or 1

### Jordan-Wigner Encoding

Maps fermionic operators to qubit operators while preserving exclusion:
$$c_j \to \left(\prod_{k<j} Z_k\right) \sigma_j^-$$

The Z-string enforces anticommutation and exclusion.

### Fermionic Simulation

VQE for molecular chemistry must:
- Preserve particle number
- Respect Pauli exclusion
- Use fermionic ansätze (UCCSD, etc.)

---

## Worked Examples

### Example 1: Maximum Electrons in Shell

**Problem:** How many electrons can the n = 3 shell hold?

**Solution:**

For n = 3: ℓ = 0, 1, 2 (s, p, d)

- 3s (ℓ = 0): 2(2·0 + 1) = 2 electrons
- 3p (ℓ = 1): 2(2·1 + 1) = 6 electrons
- 3d (ℓ = 2): 2(2·2 + 1) = 10 electrons

Total: 2 + 6 + 10 = **18 electrons**

General formula: 2n² electrons per shell.

### Example 2: Ground State of Carbon

**Problem:** Determine the ground state electron configuration of carbon (Z = 6).

**Solution:**

Fill in order: 1s² 2s² 2p²

For the two 2p electrons, Hund's rules give:
- Maximize total S: both electrons have same ms → S = 1
- Maximize total L: put in different mℓ orbitals → L = 1
- Term symbol: ³P

### Example 3: Fermi Energy of Copper

**Problem:** Calculate the Fermi energy of copper (n = 8.5 × 10²⁸ m⁻³).

**Solution:**

$$E_F = \frac{\hbar^2}{2m_e}(3\pi^2 n)^{2/3}$$

$$= \frac{(1.055 \times 10^{-34})^2}{2 \times 9.109 \times 10^{-31}}(3\pi^2 \times 8.5 \times 10^{28})^{2/3}$$

$$= 1.13 \times 10^{-18} \text{ J} = 7.0 \text{ eV}$$

---

## Practice Problems

### Problem Set 69.4

**Direct Application:**
1. Write the electron configuration for Fe (Z = 26).

2. How many electrons can fit in the 4f subshell? The 5g subshell?

3. Calculate the Fermi energy for aluminum (n = 1.8 × 10²⁹ m⁻³).

**Intermediate:**
4. Show that degeneracy pressure P ∝ n^(5/3) for a Fermi gas.

5. Explain why the d-block elements (transition metals) have similar chemistry despite different electron numbers.

6. If electrons were bosons, what would be the ground state of helium?

**Challenging:**
7. Derive the Chandrasekhar mass limit for white dwarfs using dimensional analysis.

8. In a magnetic field, electrons have only ms = -1/2 in the ground state. How does this affect atomic structure?

9. Calculate the degeneracy pressure in a neutron star of density ρ = 10¹⁷ kg/m³.

---

## Computational Lab

```python
"""
Day 480 Lab: Pauli Exclusion and Atomic Structure
Visualizes orbital filling and Fermi statistics
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ORBITAL FILLING
# ============================================================

print("=" * 60)
print("ELECTRON CONFIGURATION GENERATOR")
print("=" * 60)

def get_orbital_order():
    """Return orbitals in Aufbau filling order"""
    # (n, l, name, max_electrons)
    order = [
        (1, 0, '1s', 2), (2, 0, '2s', 2), (2, 1, '2p', 6),
        (3, 0, '3s', 2), (3, 1, '3p', 6), (4, 0, '4s', 2),
        (3, 2, '3d', 10), (4, 1, '4p', 6), (5, 0, '5s', 2),
        (4, 2, '4d', 10), (5, 1, '5p', 6), (6, 0, '6s', 2),
        (4, 3, '4f', 14), (5, 2, '5d', 10), (6, 1, '6p', 6),
        (7, 0, '7s', 2), (5, 3, '5f', 14), (6, 2, '6d', 10),
    ]
    return order

def electron_configuration(Z):
    """Generate electron configuration for element with atomic number Z"""
    orbitals = get_orbital_order()
    config = []
    remaining = Z

    for n, l, name, max_e in orbitals:
        if remaining <= 0:
            break
        electrons = min(remaining, max_e)
        if electrons > 0:
            config.append(f"{name}{electrons if electrons > 1 else ''}")
        remaining -= electrons

    return ' '.join(config)

# Generate configurations for first 36 elements
elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']

print(f"\n{'Z':<4} {'Element':<8} {'Configuration':<40}")
print("-" * 52)

for Z, elem in enumerate(elements, 1):
    config = electron_configuration(Z)
    print(f"{Z:<4} {elem:<8} {config:<40}")

# ============================================================
# ORBITAL ENERGY LEVELS
# ============================================================

print("\n" + "=" * 60)
print("ORBITAL ENERGY LEVEL DIAGRAM")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12, 8))

# Simplified energy levels (not to scale, for illustration)
levels = {
    '1s': -13.6, '2s': -3.4, '2p': -3.0,
    '3s': -1.5, '3p': -1.2, '3d': -0.8,
    '4s': -0.85, '4p': -0.6, '4d': -0.4, '4f': -0.3,
    '5s': -0.45, '5p': -0.3
}

l_positions = {'s': 0, 'p': 1, 'd': 2, 'f': 3}

for name, energy in levels.items():
    n = int(name[0])
    l_letter = name[1]
    l = l_positions[l_letter]

    x = l + 0.5 * (n - 1)
    ax.hlines(energy, x - 0.3, x + 0.3, colors='blue', linewidth=3)
    ax.text(x, energy + 0.15, name, ha='center', fontsize=10)

ax.set_ylabel('Energy (eV, not to scale)', fontsize=12)
ax.set_xlabel('Orbital type', fontsize=12)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['s', 'p', 'd', 'f'])
ax.set_title('Atomic Orbital Energy Levels', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-0.5, 5)

plt.tight_layout()
plt.savefig('orbital_levels.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# FERMI ENERGY AND DEGENERACY PRESSURE
# ============================================================

print("\n" + "=" * 60)
print("FERMI ENERGY IN METALS")
print("=" * 60)

# Physical constants
hbar = 1.055e-34  # J·s
m_e = 9.109e-31   # kg
eV = 1.602e-19    # J

def fermi_energy(n):
    """Calculate Fermi energy given electron density n (m^-3)"""
    return (hbar**2 / (2 * m_e)) * (3 * np.pi**2 * n)**(2/3)

def degeneracy_pressure(n):
    """Calculate electron degeneracy pressure (Pa)"""
    E_F = fermi_energy(n)
    return (2/5) * n * E_F

# Metal electron densities (free electrons per m³)
metals = {
    'Li': 4.7e28,
    'Na': 2.5e28,
    'K': 1.4e28,
    'Cu': 8.5e28,
    'Ag': 5.9e28,
    'Au': 5.9e28,
    'Al': 18.1e28,
}

print(f"\n{'Metal':<8} {'n (10²⁸/m³)':<15} {'E_F (eV)':<12} {'P (GPa)':<12}")
print("-" * 47)

for metal, n in metals.items():
    E_F = fermi_energy(n) / eV
    P = degeneracy_pressure(n) / 1e9
    print(f"{metal:<8} {n/1e28:<15.1f} {E_F:<12.2f} {P:<12.1f}")

# ============================================================
# FERMI-DIRAC DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("FERMI-DIRAC OCCUPATION")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

E_F = 7.0  # Fermi energy in eV
energies = np.linspace(0, 14, 500)

for T_K in [0, 300, 1000, 3000, 10000]:
    if T_K == 0:
        occupation = np.where(energies < E_F, 1.0, 0.0)
        label = 'T = 0 K'
    else:
        k_B = 8.617e-5  # eV/K
        T_eV = k_B * T_K
        x = (energies - E_F) / T_eV
        x = np.clip(x, -500, 500)
        occupation = 1 / (np.exp(x) + 1)
        label = f'T = {T_K} K'

    ax.plot(energies, occupation, linewidth=2, label=label)

ax.axvline(E_F, color='red', linestyle='--', alpha=0.5, label=f'$E_F$ = {E_F} eV')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Energy (eV)', fontsize=12)
ax.set_ylabel('Occupation Probability f(E)', fontsize=12)
ax.set_title('Fermi-Dirac Distribution at Different Temperatures', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 14)
ax.set_ylim(-0.05, 1.1)

plt.tight_layout()
plt.savefig('fermi_dirac.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# DEGENERACY PRESSURE IN STARS
# ============================================================

print("\n" + "=" * 60)
print("STELLAR DEGENERACY PRESSURE")
print("=" * 60)

# For white dwarfs: electron degeneracy
# For neutron stars: neutron degeneracy

def electron_deg_pressure(rho):
    """Electron degeneracy pressure for fully ionized carbon
    rho in kg/m³, returns pressure in Pa"""
    m_p = 1.673e-27  # proton mass
    m_u = m_p  # atomic mass unit ≈ proton mass
    # For carbon: 6 electrons per 12 nucleons → Y_e = 0.5
    Y_e = 0.5
    n_e = (Y_e * rho) / m_u
    return (hbar**2 / (5 * m_e)) * (3 * np.pi**2)**(2/3) * n_e**(5/3)

def neutron_deg_pressure(rho):
    """Neutron degeneracy pressure
    rho in kg/m³, returns pressure in Pa"""
    m_n = 1.675e-27
    n_n = rho / m_n
    return (hbar**2 / (5 * m_n)) * (3 * np.pi**2)**(2/3) * n_n**(5/3)

rho_range = np.logspace(6, 18, 100)  # kg/m³

P_e = electron_deg_pressure(rho_range)
P_n = neutron_deg_pressure(rho_range)

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(rho_range, P_e, 'b-', linewidth=2, label='Electron degeneracy')
ax.loglog(rho_range, P_n, 'r-', linewidth=2, label='Neutron degeneracy')

# Mark typical densities
ax.axvline(1e9, color='blue', linestyle='--', alpha=0.5)
ax.text(1e9, 1e20, 'White dwarf', rotation=90, va='bottom', fontsize=10)
ax.axvline(1e17, color='red', linestyle='--', alpha=0.5)
ax.text(1e17, 1e30, 'Neutron star', rotation=90, va='bottom', fontsize=10)

ax.set_xlabel('Density (kg/m³)', fontsize=12)
ax.set_ylabel('Degeneracy Pressure (Pa)', fontsize=12)
ax.set_title('Stellar Degeneracy Pressure', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('stellar_pressure.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("KEY CONSEQUENCES OF PAULI EXCLUSION")
print("=" * 60)
print("""
1. ATOMIC STRUCTURE
   - Shell structure of atoms
   - Periodic table organization
   - Chemical properties from valence electrons

2. SOLID STATE PHYSICS
   - Fermi energy and Fermi surface
   - Electrical conductivity in metals
   - Band structure in semiconductors

3. ASTROPHYSICS
   - White dwarf stability (electron degeneracy)
   - Neutron star stability (neutron degeneracy)
   - Chandrasekhar limit

4. STABILITY OF MATTER
   - Prevents atomic collapse
   - Volume scales with N (extensive)
   - Matter is stable against compression
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Exclusion principle | No two fermions in same state |
| Electrons per orbital | 2 (spin up + down) |
| Electrons per subshell | 2(2ℓ + 1) |
| Fermi energy | $E_F = \frac{\hbar^2}{2m}(3\pi^2 n)^{2/3}$ |
| Degeneracy pressure | $P = \frac{2}{5}nE_F$ |

### Main Takeaways

1. **Pauli exclusion** follows from fermionic antisymmetry
2. **Atomic structure** emerges from exclusion + Aufbau
3. **Degeneracy pressure** exists even at T = 0
4. **Stellar remnants** are supported by degeneracy pressure
5. **Stability of matter** requires the exclusion principle

---

## Daily Checklist

- [ ] I can derive Pauli exclusion from antisymmetry
- [ ] I can determine electron configurations
- [ ] I understand degeneracy pressure
- [ ] I know why matter is stable
- [ ] I completed the computational lab

---

## Preview: Day 481

Tomorrow we study **Slater determinants**—the systematic way to construct antisymmetric wave functions for many-fermion systems.

---

**Next:** [Day_481_Friday.md](Day_481_Friday.md) — Slater Determinants
