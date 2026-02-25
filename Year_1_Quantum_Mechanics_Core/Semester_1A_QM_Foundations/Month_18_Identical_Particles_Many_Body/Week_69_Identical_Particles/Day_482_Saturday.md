# Day 482: Exchange Forces

## Overview
**Day 482** | Year 1, Month 18, Week 69 | The Physics of Particle Indistinguishability

Today we explore exchange forces—the effective interactions that arise purely from the quantum mechanical requirement of symmetric or antisymmetric wave functions. These are not "real" forces but have profound physical consequences.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Exchange interaction theory |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Physical consequences |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Exchange energy simulation |

---

## Learning Objectives

By the end of today, you will be able to:
1. Derive the exchange energy for two-particle systems
2. Distinguish direct and exchange integrals
3. Explain ferromagnetism from exchange interactions
4. Calculate exchange splitting in atoms
5. Understand covalent bonding from exchange
6. Connect exchange to quantum computing applications

---

## Core Content

### The Two-Electron Problem

Consider two electrons with Hamiltonian:
$$\hat{H} = \hat{h}_1 + \hat{h}_2 + \frac{e^2}{|\mathbf{r}_1 - \mathbf{r}_2|}$$

where $\hat{h}_i = -\frac{\hbar^2}{2m}\nabla_i^2 + V(\mathbf{r}_i)$

### Spatial Wave Functions

**Symmetric (singlet spin):**
$$\Psi_S(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\psi_a(\mathbf{r}_1)\psi_b(\mathbf{r}_2) + \psi_a(\mathbf{r}_2)\psi_b(\mathbf{r}_1)]$$

**Antisymmetric (triplet spin):**
$$\Psi_A(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\psi_a(\mathbf{r}_1)\psi_b(\mathbf{r}_2) - \psi_a(\mathbf{r}_2)\psi_b(\mathbf{r}_1)]$$

### Energy Expectation Values

$$E_{S/A} = \langle\Psi_{S/A}|\hat{H}|\Psi_{S/A}\rangle = E_a + E_b + J \pm K$$

where:

**Direct (Coulomb) integral:**
$$J = \int |\psi_a(\mathbf{r}_1)|^2 \frac{e^2}{r_{12}} |\psi_b(\mathbf{r}_2)|^2 \, d^3r_1 d^3r_2$$

**Exchange integral:**
$$K = \int \psi_a^*(\mathbf{r}_1)\psi_b^*(\mathbf{r}_2) \frac{e^2}{r_{12}} \psi_a(\mathbf{r}_2)\psi_b(\mathbf{r}_1) \, d^3r_1 d^3r_2$$

### Exchange Splitting

$$\boxed{E_S - E_A = 2K}$$

**Physical meaning:**
- K > 0 (typical): Triplet (parallel spins) lower in energy
- K < 0 (rare): Singlet (antiparallel) lower in energy

### The Exchange Hole

Antisymmetric spatial functions have:
$$|\Psi_A(\mathbf{r}, \mathbf{r})|^2 = 0$$

Fermions with parallel spins **cannot be at the same point**!

This creates an "exchange hole"—a region of reduced probability around each fermion.

---

## Physical Applications

### Hund's First Rule

**Statement:** For a given electron configuration, the term with maximum spin S has lowest energy.

**Explanation:** Parallel spins → antisymmetric spatial → reduced Coulomb repulsion (exchange hole)

### Example: Carbon (2p²)

Configurations with same spatial occupancy:
- ³P: S = 1 (triplet) — lowest energy
- ¹D: S = 0 (singlet) — higher energy
- ¹S: S = 0 (singlet) — highest energy

### Ferromagnetism

**Heisenberg model:** Exchange interaction between localized spins
$$\hat{H} = -\sum_{ij} J_{ij} \mathbf{S}_i \cdot \mathbf{S}_j$$

**If J > 0:** Parallel spins favored → ferromagnetism
**If J < 0:** Antiparallel favored → antiferromagnetism

### Covalent Bonding

**H₂ molecule:** Exchange interaction determines bonding
- Singlet (S = 0): Symmetric spatial → electrons between nuclei → bonding
- Triplet (S = 1): Antisymmetric spatial → node between nuclei → antibonding

---

## Quantum Computing Connection

### Exchange Gates

**SWAP gate:** Exchanges qubit states
$$\text{SWAP}|ij\rangle = |ji\rangle$$

**√SWAP:** Square root of SWAP, useful for entanglement

### Exchange-Based Qubits

**Singlet-triplet qubits:** Encode information in S-T states
- |0⟩ = singlet
- |1⟩ = triplet (m = 0)
- Exchange interaction controls transitions

**Advantages:**
- All-electrical control
- No need for magnetic field gradients

### Exchange in Quantum Dots

Double quantum dot with two electrons:
$$H = J(t)(\mathbf{S}_1 \cdot \mathbf{S}_2 + \frac{1}{4})$$

Tunable J enables single-qubit gates.

---

## Worked Examples

### Example 1: Helium Singlet vs Triplet

**Problem:** For helium in 1s2s configuration, estimate the singlet-triplet splitting.

**Solution:**

The exchange integral K between 1s and 2s orbitals:
$$K_{1s,2s} = \int \psi_{1s}^*(\mathbf{r}_1)\psi_{2s}^*(\mathbf{r}_2) \frac{e^2}{r_{12}} \psi_{1s}(\mathbf{r}_2)\psi_{2s}(\mathbf{r}_1) d^3r_1 d^3r_2$$

Numerical result: K ≈ 0.026 Hartree ≈ 0.7 eV

Splitting: E_S - E_T = 2K ≈ 1.4 eV

The triplet (orthohelium) is lower than singlet (parahelium).

### Example 2: Hund's Rule for Nitrogen

**Problem:** Predict the ground state term for nitrogen (1s² 2s² 2p³).

**Solution:**

Three 2p electrons, maximizing S:
- Put all three with same spin: S = 3/2
- Distribute in different mℓ: -1, 0, +1 → L = 0

Ground term: ⁴S (quartet S-state)

---

## Practice Problems

### Problem Set 69.6

**Direct Application:**
1. Show that K > 0 for hydrogen-like orbitals (attractive exchange).

2. Calculate the exchange splitting for two electrons in the first two states of an infinite square well.

3. Verify that ⟨Ψ_A(r, r)|² = 0 for the antisymmetric wave function.

**Intermediate:**
4. Derive Hund's first rule by showing that parallel spins reduce electron-electron repulsion.

5. For H₂, explain why the singlet is bonding and triplet is antibonding.

6. Calculate the exchange integral J for the Hubbard model in a double quantum dot.

**Challenging:**
7. Derive the Heisenberg Hamiltonian from the Hubbard model in the limit U ≫ t.

8. Explain why antiferromagnetism is common in insulators but ferromagnetism requires metallic bands.

9. Design a √SWAP gate using exchange interaction in a double quantum dot.

---

## Computational Lab

```python
"""
Day 482 Lab: Exchange Forces and Exchange Energy
Calculates exchange splitting for two-particle systems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, tplquad
from scipy.special import factorial

print("=" * 60)
print("EXCHANGE INTERACTION IN TWO-ELECTRON SYSTEMS")
print("=" * 60)

# ============================================================
# EXCHANGE ENERGY FOR 1D HARMONIC OSCILLATOR
# ============================================================

def psi_n(x, n, alpha=1):
    """Harmonic oscillator eigenfunction"""
    H_n = np.polynomial.hermite.hermval(alpha * x, [0]*n + [1])
    norm = (alpha / np.pi)**0.25 / np.sqrt(2**n * factorial(n))
    return norm * H_n * np.exp(-alpha * x**2 / 2)

def direct_integral_1d(n1, n2, alpha=1):
    """Direct (Coulomb) integral J = ∫|ψ_n1|²|ψ_n2|² / |x1-x2| dx1 dx2"""
    # Use softened Coulomb: 1/sqrt((x1-x2)² + ε²)
    eps = 0.1

    def integrand(x1, x2):
        return (psi_n(x1, n1, alpha)**2 * psi_n(x2, n2, alpha)**2
                / np.sqrt((x1 - x2)**2 + eps**2))

    result, _ = dblquad(integrand, -5, 5, -5, 5)
    return result

def exchange_integral_1d(n1, n2, alpha=1):
    """Exchange integral K = ∫ψ*_n1(x1)ψ*_n2(x2) V ψ_n1(x2)ψ_n2(x1) dx1 dx2"""
    eps = 0.1

    def integrand(x1, x2):
        return (psi_n(x1, n1, alpha) * psi_n(x2, n2, alpha)
                * psi_n(x2, n1, alpha) * psi_n(x1, n2, alpha)
                / np.sqrt((x1 - x2)**2 + eps**2))

    result, _ = dblquad(integrand, -5, 5, -5, 5)
    return result

# Calculate for n1=0, n2=1
print("\nTwo electrons in harmonic oscillator (n=0 and n=1):")
J = direct_integral_1d(0, 1)
K = exchange_integral_1d(0, 1)
print(f"Direct integral J = {J:.4f}")
print(f"Exchange integral K = {K:.4f}")
print(f"Singlet-Triplet splitting = 2K = {2*K:.4f}")

# ============================================================
# VISUALIZE SYMMETRIC VS ANTISYMMETRIC
# ============================================================

print("\n" + "=" * 60)
print("SPATIAL WAVE FUNCTIONS: SINGLET VS TRIPLET")
print("=" * 60)

x = np.linspace(-4, 4, 100)
X1, X2 = np.meshgrid(x, x)

# Two particles in states 0 and 1
psi0_1 = psi_n(X1, 0)
psi1_1 = psi_n(X1, 1)
psi0_2 = psi_n(X2, 0)
psi1_2 = psi_n(X2, 1)

# Symmetric (singlet spin → symmetric spatial)
Psi_S = (psi0_1 * psi1_2 + psi0_2 * psi1_1) / np.sqrt(2)

# Antisymmetric (triplet spin → antisymmetric spatial)
Psi_A = (psi0_1 * psi1_2 - psi0_2 * psi1_1) / np.sqrt(2)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Top row: Wave functions
im = axes[0, 0].contourf(X1, X2, Psi_S, levels=50, cmap='RdBu_r')
axes[0, 0].set_title('Symmetric Ψ_S (Singlet spin)', fontsize=12)
axes[0, 0].set_xlabel('x₁')
axes[0, 0].set_ylabel('x₂')
axes[0, 0].plot(x, x, 'k--', alpha=0.5)
plt.colorbar(im, ax=axes[0, 0])

im = axes[0, 1].contourf(X1, X2, Psi_A, levels=50, cmap='RdBu_r')
axes[0, 1].set_title('Antisymmetric Ψ_A (Triplet spin)', fontsize=12)
axes[0, 1].set_xlabel('x₁')
axes[0, 1].set_ylabel('x₂')
axes[0, 1].plot(x, x, 'k--', alpha=0.5)
plt.colorbar(im, ax=axes[0, 1])

# Diagonal comparison
axes[0, 2].plot(x, np.diag(Psi_S), 'b-', linewidth=2, label='Symmetric')
axes[0, 2].plot(x, np.diag(Psi_A), 'r-', linewidth=2, label='Antisymmetric')
axes[0, 2].axhline(0, color='k', linewidth=0.5)
axes[0, 2].set_xlabel('x₁ = x₂')
axes[0, 2].set_ylabel('Ψ')
axes[0, 2].set_title('Wave Function Along Diagonal', fontsize=12)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Bottom row: Probability densities
im = axes[1, 0].contourf(X1, X2, Psi_S**2, levels=50, cmap='viridis')
axes[1, 0].set_title('|Ψ_S|² (electrons can be close)', fontsize=12)
axes[1, 0].set_xlabel('x₁')
axes[1, 0].set_ylabel('x₂')
axes[1, 0].plot(x, x, 'w--', alpha=0.5)
plt.colorbar(im, ax=axes[1, 0])

im = axes[1, 1].contourf(X1, X2, Psi_A**2, levels=50, cmap='viridis')
axes[1, 1].set_title('|Ψ_A|² (exchange hole at x₁=x₂)', fontsize=12)
axes[1, 1].set_xlabel('x₁')
axes[1, 1].set_ylabel('x₂')
axes[1, 1].plot(x, x, 'w--', alpha=0.5)
plt.colorbar(im, ax=axes[1, 1])

# Diagonal probability
axes[1, 2].plot(x, np.diag(Psi_S**2), 'b-', linewidth=2, label='Singlet (close)')
axes[1, 2].plot(x, np.diag(Psi_A**2), 'r-', linewidth=2, label='Triplet (apart)')
axes[1, 2].set_xlabel('x₁ = x₂')
axes[1, 2].set_ylabel('|Ψ|²')
axes[1, 2].set_title('Probability at Same Position', fontsize=12)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exchange_spatial.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# HEISENBERG MODEL: FERROMAGNETISM
# ============================================================

print("\n" + "=" * 60)
print("HEISENBERG MODEL: FERROMAGNETIC vs ANTIFERROMAGNETIC")
print("=" * 60)

def heisenberg_energy(spins, J):
    """Energy of 1D Heisenberg chain with coupling J"""
    N = len(spins)
    E = 0
    for i in range(N - 1):
        E -= J * spins[i] * spins[i + 1]
    return E

# Compare ferromagnetic (J > 0) vs antiferromagnetic (J < 0)
N = 10
spins_ferro = np.ones(N)  # All aligned
spins_antiferro = np.array([(-1)**i for i in range(N)])  # Alternating

J_values = np.linspace(-1, 1, 100)

fig, ax = plt.subplots(figsize=(10, 6))

E_ferro = [heisenberg_energy(spins_ferro, J) for J in J_values]
E_antiferro = [heisenberg_energy(spins_antiferro, J) for J in J_values]

ax.plot(J_values, E_ferro, 'r-', linewidth=2, label='Ferromagnetic (↑↑↑↑)')
ax.plot(J_values, E_antiferro, 'b-', linewidth=2, label='Antiferromagnetic (↑↓↑↓)')

ax.axvline(0, color='k', linestyle='--', alpha=0.5)
ax.axhline(0, color='k', linewidth=0.5)

ax.fill_between(J_values, E_ferro, E_antiferro,
                where=np.array(E_ferro) < np.array(E_antiferro),
                alpha=0.3, color='red', label='Ferro wins')
ax.fill_between(J_values, E_ferro, E_antiferro,
                where=np.array(E_ferro) > np.array(E_antiferro),
                alpha=0.3, color='blue', label='AF wins')

ax.set_xlabel('Exchange coupling J', fontsize=12)
ax.set_ylabel('Energy', fontsize=12)
ax.set_title('Heisenberg Model: Exchange Determines Magnetic Order', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heisenberg_model.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("KEY INSIGHTS: EXCHANGE FORCES")
print("=" * 60)
print("""
1. EXCHANGE SPLITTING: E_S - E_T = 2K
   - K > 0: Triplet (parallel spins) lower in energy
   - This is NOT a magnetic interaction!

2. EXCHANGE HOLE: Antisymmetric spatial → Ψ(r,r) = 0
   - Fermions with parallel spins avoid each other
   - Reduces Coulomb repulsion

3. HUND'S FIRST RULE: Maximize S
   - Exchange hole reduces repulsion
   - Parallel spins favored

4. MAGNETISM:
   - Ferromagnetism: J > 0 (parallel spins)
   - Antiferromagnetism: J < 0 (antiparallel)

5. CHEMICAL BONDING:
   - Singlet → symmetric spatial → bonding
   - Triplet → antisymmetric spatial → antibonding
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Energy | $E_{S/A} = E_a + E_b + J \pm K$ |
| Exchange splitting | $E_S - E_T = 2K$ |
| Direct integral | $J = \int \|\psi_a\|^2 V \|\psi_b\|^2$ |
| Exchange integral | $K = \int \psi_a^*\psi_b^* V \psi_a\psi_b$ |
| Heisenberg model | $H = -\sum J_{ij} \mathbf{S}_i \cdot \mathbf{S}_j$ |

### Main Takeaways

1. **Exchange forces** arise purely from symmetry requirements
2. **Exchange integral K** determines singlet-triplet splitting
3. **Hund's rule:** Parallel spins reduce repulsion via exchange hole
4. **Ferromagnetism** comes from positive exchange coupling
5. **Covalent bonding** involves exchange between atoms

---

## Daily Checklist

- [ ] I can derive exchange energy for two electrons
- [ ] I understand direct vs exchange integrals
- [ ] I can explain Hund's first rule
- [ ] I know how exchange leads to magnetism
- [ ] I completed the computational lab

---

## Preview: Day 483

Tomorrow we consolidate Week 69 with a **comprehensive review** of identical particles, bringing together all the concepts from this week.

---

**Next:** [Day_483_Sunday.md](Day_483_Sunday.md) — Week 69 Review
