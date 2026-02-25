# Day 478: Bosons and Fermions

## Overview
**Day 478** | Year 1, Month 18, Week 69 | The Two Families of Particles

Today we dive deep into the two fundamental classes of particles in nature: bosons and fermions. Their contrasting behaviors govern all of condensed matter physics and quantum statistics.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Bosonic and fermionic wave functions |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Quantum statistics |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Distribution function simulation |

---

## Learning Objectives

By the end of today, you will be able to:
1. Construct symmetric (bosonic) multi-particle wave functions
2. Construct antisymmetric (fermionic) multi-particle wave functions
3. Explain Bose-Einstein and Fermi-Dirac statistics
4. Calculate occupation numbers for bosonic/fermionic systems
5. Identify examples of bosons and fermions in nature
6. Understand composite particle statistics

---

## Core Content

### Classification of Particles

**Bosons:** Particles with integer spin (0, 1, 2, ...)
- Symmetric wave functions
- Can share quantum states (no exclusion)
- Examples: photons, phonons, gluons, W/Z bosons, Higgs

**Fermions:** Particles with half-integer spin (1/2, 3/2, ...)
- Antisymmetric wave functions
- Cannot share quantum states (Pauli exclusion)
- Examples: electrons, protons, neutrons, quarks, neutrinos

### Two-Particle Wave Functions

Given single-particle states |ψ_a⟩ and |ψ_b⟩:

**Distinguishable particles:**
$$|\Psi\rangle = |\psi_a\rangle_1 |\psi_b\rangle_2$$

**Bosons (symmetric):**
$$|\Psi_B\rangle = \frac{1}{\sqrt{2}}(|\psi_a\rangle_1 |\psi_b\rangle_2 + |\psi_b\rangle_1 |\psi_a\rangle_2)$$

**Fermions (antisymmetric):**
$$|\Psi_F\rangle = \frac{1}{\sqrt{2}}(|\psi_a\rangle_1 |\psi_b\rangle_2 - |\psi_b\rangle_1 |\psi_a\rangle_2)$$

### When Both Particles in Same State

If ψ_a = ψ_b = ψ:

**Bosons:**
$$|\Psi_B\rangle = \frac{1}{\sqrt{2}}(|\psi\psi\rangle + |\psi\psi\rangle) = \sqrt{2}|\psi\psi\rangle$$

Normalizing: $|\Psi_B\rangle = |\psi\rangle_1|\psi\rangle_2$ (allowed!)

**Fermions:**
$$|\Psi_F\rangle = \frac{1}{\sqrt{2}}(|\psi\psi\rangle - |\psi\psi\rangle) = 0$$

$$\boxed{\text{Two fermions cannot occupy the same state}}$$

### Normalization with Identical States

For N bosons, n_i particles in state i:
$$|\Psi_B\rangle = \frac{1}{\sqrt{N!\prod_i n_i!}}\sum_{\text{perms}} |\psi_{\sigma(1)}\cdots\psi_{\sigma(N)}\rangle$$

For N fermions, all n_i ∈ {0, 1}:
$$|\Psi_F\rangle = \frac{1}{\sqrt{N!}}\sum_{\text{perms}} (-1)^\sigma |\psi_{\sigma(1)}\cdots\psi_{\sigma(N)}\rangle$$

---

## Quantum Statistics

### Bose-Einstein Distribution

For non-interacting bosons at temperature T:
$$\bar{n}_i^{BE} = \frac{1}{e^{(\epsilon_i - \mu)/k_BT} - 1}$$

where μ = chemical potential, ε_i = single-particle energy.

**Key features:**
- No upper limit on occupation
- Bose-Einstein condensation at low T
- μ ≤ ε_0 (ground state energy)

### Fermi-Dirac Distribution

For non-interacting fermions at temperature T:
$$\bar{n}_i^{FD} = \frac{1}{e^{(\epsilon_i - \mu)/k_BT} + 1}$$

**Key features:**
- Occupation 0 ≤ n_i ≤ 1
- At T = 0: step function at Fermi energy
- μ = E_F at T = 0

### Maxwell-Boltzmann (Classical Limit)

$$\bar{n}_i^{MB} = e^{-(\epsilon_i - \mu)/k_BT}$$

Valid when n_i ≪ 1 (high T, low density).

### Comparison of Statistics

| Property | Bose-Einstein | Fermi-Dirac | Maxwell-Boltzmann |
|----------|--------------|-------------|-------------------|
| Occupation | 0, 1, 2, ... | 0 or 1 | 0, 1, 2, ... |
| Low T behavior | Condensation | Fermi sea | Classical gas |
| Exchange | +1 | -1 | 1 (no symmetry) |

---

## Composite Particles

### Bosons or Fermions?

A composite particle behaves as:
- **Boson** if total spin is integer
- **Fermion** if total spin is half-integer

**Examples:**

| Composite | Components | Total Spin | Type |
|-----------|------------|------------|------|
| ⁴He atom | 2p + 2n + 2e | 0 | Boson |
| ³He atom | 2p + 1n + 2e | 1/2 | Fermion |
| Hydrogen | 1p + 1e | 0 or 1 | Boson |
| Cooper pair | 2e | 0 | Boson |

### He-4 vs He-3

**⁴He (boson):**
- Superfluid below 2.17 K
- Bose-Einstein condensation
- Zero viscosity flow

**³He (fermion):**
- Fermi liquid above 1 mK
- Superfluid below ~2.5 mK (Cooper pairing)
- Much more complex phase diagram

---

## Quantum Computing Connection

### Bosonic vs Fermionic Qubits

**Standard qubits:** Neither bosonic nor fermionic—they're distinguishable!
- Labeled by position (ion trap, transmon)
- No exchange symmetry required

**Majorana fermions:**
- Non-Abelian anyons (neither boson nor fermion)
- Potential topological qubits
- Braiding encodes quantum information

### Fermionic Simulation

**The challenge:** Map fermions to qubits while preserving anticommutation.

**Jordan-Wigner transformation:**
$$c_j = \left(\prod_{k<j} Z_k\right) \frac{X_j - iY_j}{2}$$

This maps fermionic operators to qubit operators with the correct anticommutation.

### Bosonic Encodings

**Cat codes:** Logical |0⟩, |1⟩ encoded as superpositions of coherent states
$$|0_L\rangle \propto |\alpha\rangle + |-\alpha\rangle$$
$$|1_L\rangle \propto |\alpha\rangle - |-\alpha\rangle$$

---

## Worked Examples

### Example 1: Two Bosons in Harmonic Oscillator

**Problem:** Write the wave function for two identical bosons in states n=0 and n=1 of a harmonic oscillator.

**Solution:**

Single-particle states: $\phi_0(x)$ and $\phi_1(x)$

Two-boson wave function:
$$\Psi_B(x_1, x_2) = \frac{1}{\sqrt{2}}[\phi_0(x_1)\phi_1(x_2) + \phi_0(x_2)\phi_1(x_1)]$$

This is symmetric under x₁ ↔ x₂.

### Example 2: Two Fermions in Harmonic Oscillator

**Problem:** Same setup, but for fermions.

**Solution:**

$$\Psi_F(x_1, x_2) = \frac{1}{\sqrt{2}}[\phi_0(x_1)\phi_1(x_2) - \phi_0(x_2)\phi_1(x_1)]$$

Note: If both were in n=0, $\Psi_F = 0$.

### Example 3: Fermi Energy

**Problem:** N non-interacting electrons in a 1D box of length L at T = 0. Find the Fermi energy.

**Solution:**

Energy levels: $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$

Each level holds 2 electrons (spin up/down).

For N electrons, fill levels up to n = N/2:
$$E_F = \frac{(N/2)^2\pi^2\hbar^2}{2mL^2} = \frac{N^2\pi^2\hbar^2}{8mL^2}$$

---

## Practice Problems

### Problem Set 69.2

**Direct Application:**
1. Write the symmetric and antisymmetric wave functions for two particles in states ψ_1(r) = e^{ikr} and ψ_2(r) = e^{-ikr}.

2. Calculate ⟨n⟩ for photons (bosons) at ℏω = k_BT.

3. What is the probability of finding a fermion state occupied at ε = μ? At ε = μ + 2k_BT?

**Intermediate:**
4. Show that for fermions at T = 0:
   $$\bar{n}(\epsilon) = \begin{cases} 1 & \epsilon < E_F \\ 0 & \epsilon > E_F \end{cases}$$

5. A harmonic oscillator has N identical bosons. How many ways can 3 bosons be distributed among the first 2 energy levels?

6. Why does ⁴He become superfluid while ³He doesn't (until much lower T)?

**Challenging:**
7. Derive the Bose-Einstein distribution using the grand canonical ensemble.

8. Show that at high T, both BE and FD reduce to MB statistics.

9. Calculate the heat capacity of a 2D electron gas at low T.

---

## Computational Lab

```python
"""
Day 478 Lab: Bose-Einstein vs Fermi-Dirac Statistics
Compares quantum statistics with classical Maxwell-Boltzmann
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants (use k_B = 1 units)
k_B = 1

def fermi_dirac(epsilon, mu, T):
    """Fermi-Dirac distribution"""
    if T == 0:
        return np.where(epsilon < mu, 1.0, 0.0)
    x = (epsilon - mu) / (k_B * T)
    # Avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (np.exp(x) + 1)

def bose_einstein(epsilon, mu, T):
    """Bose-Einstein distribution (mu < epsilon_min)"""
    if T == 0:
        return np.where(epsilon == epsilon.min(), np.inf, 0.0)
    x = (epsilon - mu) / (k_B * T)
    x = np.clip(x, -500, 500)
    return 1 / (np.exp(x) - 1)

def maxwell_boltzmann(epsilon, mu, T):
    """Classical Maxwell-Boltzmann distribution"""
    x = (epsilon - mu) / (k_B * T)
    x = np.clip(x, -500, 500)
    return np.exp(-x)

# ============================================================
# COMPARISON OF STATISTICS
# ============================================================

print("=" * 60)
print("QUANTUM STATISTICS COMPARISON")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Fermi-Dirac at different temperatures
ax = axes[0]
epsilon = np.linspace(-2, 4, 500)
mu = 1  # Chemical potential (Fermi level at T=0)

temperatures = [0.001, 0.1, 0.3, 1.0]
colors = ['blue', 'green', 'orange', 'red']

for T, c in zip(temperatures, colors):
    n = fermi_dirac(epsilon, mu, T)
    label = f'T = {T}' if T > 0.01 else 'T → 0'
    ax.plot(epsilon, n, color=c, linewidth=2, label=label)

ax.axvline(mu, color='black', linestyle='--', alpha=0.5, label=f'μ = {mu}')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Energy ε', fontsize=12)
ax.set_ylabel('Occupation ⟨n⟩', fontsize=12)
ax.set_title('Fermi-Dirac Distribution', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 4)
ax.set_ylim(-0.05, 1.1)

# Plot 2: Bose-Einstein at different temperatures
ax = axes[1]
epsilon = np.linspace(0.01, 3, 500)
mu_BE = 0  # For bosons, μ ≤ ε_min

temperatures_BE = [0.5, 1.0, 2.0, 5.0]

for T, c in zip(temperatures_BE, colors):
    n = bose_einstein(epsilon, mu_BE, T)
    n = np.clip(n, 0, 10)  # Clip for visualization
    ax.plot(epsilon, n, color=c, linewidth=2, label=f'T = {T}')

ax.set_xlabel('Energy ε', fontsize=12)
ax.set_ylabel('Occupation ⟨n⟩', fontsize=12)
ax.set_title('Bose-Einstein Distribution', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3)
ax.set_ylim(0, 5)

plt.tight_layout()
plt.savefig('quantum_statistics.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ALL THREE DISTRIBUTIONS COMPARED
# ============================================================

print("\n" + "=" * 60)
print("COMPARING ALL THREE DISTRIBUTIONS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

epsilon = np.linspace(0.5, 5, 500)
mu = 0
T = 1

n_FD = fermi_dirac(epsilon, mu, T)
n_BE = bose_einstein(epsilon, mu - 0.1, T)  # μ < ε_min for bosons
n_MB = maxwell_boltzmann(epsilon, mu, T)

ax.plot(epsilon, n_FD, 'b-', linewidth=2, label='Fermi-Dirac')
ax.plot(epsilon, n_BE, 'r-', linewidth=2, label='Bose-Einstein')
ax.plot(epsilon, n_MB, 'g--', linewidth=2, label='Maxwell-Boltzmann')

ax.set_xlabel('Energy ε / k_B T', fontsize=12)
ax.set_ylabel('Occupation ⟨n⟩', fontsize=12)
ax.set_title('Comparison of Quantum Statistics (μ ≈ 0, T = 1)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 5)
ax.set_ylim(0, 3)

plt.tight_layout()
plt.savefig('statistics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# FERMI SEA VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("FERMI SEA AT T = 0")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Energy level diagram
ax = axes[0]
n_levels = 10
E_F_level = 5  # Fermi level

for n in range(n_levels):
    color = 'blue' if n < E_F_level else 'lightgray'
    ax.hlines(n, 0, 1, colors=color, linewidth=8)

    # Add spin-up and spin-down electrons
    if n < E_F_level:
        ax.annotate('↑', xy=(0.4, n), fontsize=20, ha='center', va='center')
        ax.annotate('↓', xy=(0.6, n), fontsize=20, ha='center', va='center')

ax.axhline(E_F_level - 0.5, color='red', linestyle='--', linewidth=2,
           label=f'Fermi Energy E_F')
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, n_levels + 0.5)
ax.set_ylabel('Energy Level n', fontsize=12)
ax.set_title('Fermi Sea at T = 0', fontsize=14)
ax.set_xticks([])
ax.legend(loc='upper right')

# Right: Occupation vs energy
ax = axes[1]
epsilon = np.linspace(0, 10, 500)
E_F = 5

# T = 0
n_T0 = np.where(epsilon < E_F, 1.0, 0.0)

# Finite T
for T in [0.5, 1.0, 2.0]:
    n = fermi_dirac(epsilon, E_F, T)
    ax.plot(epsilon, n, linewidth=2, label=f'T = {T}')

ax.plot(epsilon, n_T0, 'k-', linewidth=2, label='T = 0')
ax.axvline(E_F, color='red', linestyle='--', alpha=0.5)

ax.set_xlabel('Energy ε', fontsize=12)
ax.set_ylabel('Occupation ⟨n(ε)⟩', fontsize=12)
ax.set_title('Fermi Distribution vs Temperature', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fermi_sea.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# BOSE-EINSTEIN CONDENSATION
# ============================================================

print("\n" + "=" * 60)
print("BOSE-EINSTEIN CONDENSATION")
print("=" * 60)

# Simplified model: occupations in ground state vs excited states
T_c = 1  # Critical temperature (normalized)
temperatures = np.linspace(0.01, 2, 100)

def ground_state_fraction(T, T_c):
    """Fraction of bosons in ground state"""
    if T >= T_c:
        return 0
    else:
        return 1 - (T / T_c)**(3/2)

n0_fraction = [ground_state_fraction(T, T_c) for T in temperatures]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(temperatures / T_c, n0_fraction, 'b-', linewidth=2)
ax.axvline(1, color='red', linestyle='--', label='$T_c$')
ax.fill_between(temperatures / T_c, n0_fraction, alpha=0.3)

ax.set_xlabel('$T / T_c$', fontsize=12)
ax.set_ylabel('Ground State Fraction $N_0 / N$', fontsize=12)
ax.set_title('Bose-Einstein Condensation', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 2)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('bec_condensation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY: BOSONS VS FERMIONS")
print("=" * 60)

print("""
┌─────────────────┬─────────────────────┬─────────────────────┐
│ Property        │ Bosons              │ Fermions            │
├─────────────────┼─────────────────────┼─────────────────────┤
│ Spin            │ Integer (0,1,2,...) │ Half-integer (1/2,.)│
│ Wave function   │ Symmetric           │ Antisymmetric       │
│ Occupation      │ Any number 0,1,2,...│ Only 0 or 1         │
│ Distribution    │ Bose-Einstein       │ Fermi-Dirac         │
│ Low T behavior  │ BEC possible        │ Fermi sea           │
│ Examples        │ Photons, phonons    │ Electrons, quarks   │
│                 │ Higgs, gluons       │ Protons, neutrons   │
└─────────────────┴─────────────────────┴─────────────────────┘
""")
```

---

## Summary

### Key Formulas

| Quantity | Bosons | Fermions |
|----------|--------|----------|
| Wave function | Symmetric | Antisymmetric |
| Same state | Allowed | Forbidden |
| Distribution | $\frac{1}{e^{(\epsilon-\mu)/k_BT}-1}$ | $\frac{1}{e^{(\epsilon-\mu)/k_BT}+1}$ |
| Occupation | 0, 1, 2, ... | 0 or 1 |

### Main Takeaways

1. **Bosons** have integer spin, symmetric wave functions
2. **Fermions** have half-integer spin, antisymmetric wave functions
3. **Pauli exclusion:** No two fermions in the same state
4. **Statistics** determines low-temperature behavior
5. **Composite particles:** Total spin determines statistics

---

## Daily Checklist

- [ ] I can construct bosonic and fermionic wave functions
- [ ] I understand Bose-Einstein and Fermi-Dirac statistics
- [ ] I know why fermions obey the exclusion principle
- [ ] I can identify particles as bosons or fermions
- [ ] I completed the computational lab

---

## Preview: Day 479

Tomorrow we explore the **spin-statistics theorem**—the deep connection between spin and exchange symmetry.

---

**Next:** [Day_479_Wednesday.md](Day_479_Wednesday.md) — Spin-Statistics Theorem
