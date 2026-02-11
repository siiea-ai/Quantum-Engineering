# Day 481: Slater Determinants

## Overview
**Day 481** | Year 1, Month 18, Week 69 | Constructing Fermionic Wave Functions

Today we learn the systematic method for constructing antisymmetric wave functions for N fermions: the Slater determinant. This elegant mathematical structure automatically enforces Pauli exclusion.

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Slater determinant construction |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Properties and applications |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Determinant computation lab |

---

## Learning Objectives

By the end of today, you will be able to:
1. Construct Slater determinants for N fermions
2. Verify antisymmetry under particle exchange
3. Apply Slater determinants to atomic systems
4. Calculate matrix elements between Slater determinants
5. Understand the connection to Hartree-Fock theory
6. Recognize limitations of single-determinant descriptions

---

## Core Content

### The Two-Particle Case

For two fermions in spin-orbitals χ_a and χ_b:
$$\Psi(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{\sqrt{2}}[\chi_a(\mathbf{x}_1)\chi_b(\mathbf{x}_2) - \chi_a(\mathbf{x}_2)\chi_b(\mathbf{x}_1)]$$

This can be written as a **determinant**:
$$\Psi(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{\sqrt{2}}\begin{vmatrix} \chi_a(\mathbf{x}_1) & \chi_b(\mathbf{x}_1) \\ \chi_a(\mathbf{x}_2) & \chi_b(\mathbf{x}_2) \end{vmatrix}$$

### The Slater Determinant

**Definition:** For N fermions in spin-orbitals χ_1, χ_2, ..., χ_N:

$$\boxed{\Psi(\mathbf{x}_1, \ldots, \mathbf{x}_N) = \frac{1}{\sqrt{N!}}\begin{vmatrix} \chi_1(\mathbf{x}_1) & \chi_2(\mathbf{x}_1) & \cdots & \chi_N(\mathbf{x}_1) \\ \chi_1(\mathbf{x}_2) & \chi_2(\mathbf{x}_2) & \cdots & \chi_N(\mathbf{x}_2) \\ \vdots & \vdots & \ddots & \vdots \\ \chi_1(\mathbf{x}_N) & \chi_2(\mathbf{x}_N) & \cdots & \chi_N(\mathbf{x}_N) \end{vmatrix}}$$

where $\mathbf{x}_i = (\mathbf{r}_i, \sigma_i)$ includes both position and spin.

### Spin-Orbitals

A **spin-orbital** is a product of spatial and spin parts:
$$\chi(\mathbf{x}) = \psi(\mathbf{r})\eta(\sigma)$$

where η(σ) = α (spin up) or β (spin down).

### Properties of Slater Determinants

**1. Antisymmetry:**
Exchanging two particles = exchanging two rows → determinant changes sign.

**2. Pauli Exclusion:**
If two orbitals are the same (χ_a = χ_b), two columns are identical → determinant = 0.

**3. Normalization:**
The factor $1/\sqrt{N!}$ ensures $\langle\Psi|\Psi\rangle = 1$ when orbitals are orthonormal.

### Shorthand Notation

Instead of writing the full determinant:
$$|\Psi\rangle = |\chi_1 \chi_2 \cdots \chi_N\rangle$$

or for specific orbitals:
$$|1s\alpha, 1s\beta, 2s\alpha\rangle \equiv \frac{1}{\sqrt{3!}}\begin{vmatrix} 1s\alpha(1) & 1s\beta(1) & 2s\alpha(1) \\ 1s\alpha(2) & 1s\beta(2) & 2s\alpha(2) \\ 1s\alpha(3) & 1s\beta(3) & 2s\alpha(3) \end{vmatrix}$$

---

## Example: Helium Ground State

### Spatial Part

Ground state: both electrons in 1s orbital
$$\psi_{1s}(\mathbf{r}) = \sqrt{\frac{Z^3}{\pi a_0^3}}e^{-Zr/a_0}$$

### Including Spin

Since spatial parts are identical, spins must differ:
- Electron 1: 1s↑
- Electron 2: 1s↓

### Slater Determinant

$$\Psi(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{\sqrt{2}}\begin{vmatrix} \psi_{1s}(\mathbf{r}_1)\alpha(1) & \psi_{1s}(\mathbf{r}_1)\beta(1) \\ \psi_{1s}(\mathbf{r}_2)\alpha(2) & \psi_{1s}(\mathbf{r}_2)\beta(2) \end{vmatrix}$$

$$= \frac{1}{\sqrt{2}}\psi_{1s}(\mathbf{r}_1)\psi_{1s}(\mathbf{r}_2)[\alpha(1)\beta(2) - \alpha(2)\beta(1)]$$

$$= \psi_{1s}(\mathbf{r}_1)\psi_{1s}(\mathbf{r}_2) \cdot \frac{1}{\sqrt{2}}[\alpha(1)\beta(2) - \beta(1)\alpha(2)]$$

**Result:** Symmetric spatial × antisymmetric spin (singlet)

---

## Matrix Elements

### One-Body Operators

For $\hat{F} = \sum_i \hat{f}(i)$:
$$\langle\Psi|\hat{F}|\Psi\rangle = \sum_{i=1}^{N}\langle\chi_i|\hat{f}|\chi_i\rangle$$

### Two-Body Operators

For $\hat{G} = \sum_{i<j} \hat{g}(i,j)$:
$$\langle\Psi|\hat{G}|\Psi\rangle = \sum_{i<j}[\langle\chi_i\chi_j|\hat{g}|\chi_i\chi_j\rangle - \langle\chi_i\chi_j|\hat{g}|\chi_j\chi_i\rangle]$$

The second term is the **exchange integral**—unique to fermions!

### Slater-Condon Rules

Systematic rules for matrix elements between determinants that differ by 0, 1, or 2 orbitals.

---

## Quantum Computing Connection

### VQE Ansätze

Many VQE ansätze use Slater determinants as reference:
- **UCCSD:** Unitary Coupled Cluster with Singles and Doubles
- Starts from Hartree-Fock Slater determinant
- Applies excitation operators

### Fermionic Encodings

Jordan-Wigner maps Slater determinants to qubit states:
$$|1s\alpha, 1s\beta\rangle \to |1100...0\rangle$$

(occupation number representation)

### Configuration Interaction

Going beyond single determinant:
$$|\Psi\rangle = c_0|\Phi_0\rangle + \sum_{ia} c_i^a|\Phi_i^a\rangle + \sum_{ijab} c_{ij}^{ab}|\Phi_{ij}^{ab}\rangle + \cdots$$

Multiple Slater determinants capture **electron correlation**.

---

## Worked Examples

### Example 1: Lithium Ground State

**Problem:** Write the Slater determinant for Li (Z = 3).

**Solution:**

Configuration: 1s² 2s¹

Spin-orbitals: 1sα, 1sβ, 2sα

$$\Psi = \frac{1}{\sqrt{6}}\begin{vmatrix} 1s\alpha(1) & 1s\beta(1) & 2s\alpha(1) \\ 1s\alpha(2) & 1s\beta(2) & 2s\alpha(2) \\ 1s\alpha(3) & 1s\beta(3) & 2s\alpha(3) \end{vmatrix}$$

### Example 2: Verifying Antisymmetry

**Problem:** Show the helium ground state changes sign under particle exchange.

**Solution:**

$$\Psi(\mathbf{x}_1, \mathbf{x}_2) = \psi_{1s}(r_1)\psi_{1s}(r_2)\frac{1}{\sqrt{2}}[\alpha(1)\beta(2) - \beta(1)\alpha(2)]$$

Exchange 1 ↔ 2:
$$\Psi(\mathbf{x}_2, \mathbf{x}_1) = \psi_{1s}(r_2)\psi_{1s}(r_1)\frac{1}{\sqrt{2}}[\alpha(2)\beta(1) - \beta(2)\alpha(1)]$$
$$= \psi_{1s}(r_1)\psi_{1s}(r_2)\frac{1}{\sqrt{2}}[-\alpha(1)\beta(2) + \beta(1)\alpha(2)]$$
$$= -\Psi(\mathbf{x}_1, \mathbf{x}_2) \checkmark$$

---

## Practice Problems

### Problem Set 69.5

**Direct Application:**
1. Write the Slater determinant for the beryllium ground state (Z = 4).

2. Show that a Slater determinant with two identical spin-orbitals is zero.

3. For helium in the 1s2s configuration (one electron in each), write both singlet and triplet Slater determinants.

**Intermediate:**
4. Calculate ⟨Ψ|Ĥ|Ψ⟩ for helium where Ĥ = h(1) + h(2) + 1/r₁₂, expressing the result in terms of one-electron and two-electron integrals.

5. How many Slater determinants are needed to describe the carbon 2p² configuration exactly?

6. Show that the normalization factor for an N-particle Slater determinant is 1/√N!.

**Challenging:**
7. Derive the Slater-Condon rules for matrix elements between Slater determinants that differ by one orbital.

8. Explain why a single Slater determinant cannot describe the H₂ molecule at large bond distances.

9. Write the Slater determinant for the first excited state of helium (1s2s triplet).

---

## Computational Lab

```python
"""
Day 481 Lab: Slater Determinants
Constructs and analyzes antisymmetric wave functions
"""

import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

# ============================================================
# SLATER DETERMINANT CONSTRUCTION
# ============================================================

print("=" * 60)
print("SLATER DETERMINANT CONSTRUCTION")
print("=" * 60)

def slater_determinant(orbitals, positions):
    """
    Construct Slater determinant matrix.

    Parameters:
    -----------
    orbitals : list of functions
        Single-particle orbitals χ_i(x)
    positions : array
        Particle positions x_1, x_2, ..., x_N

    Returns:
    --------
    Normalized Slater determinant value
    """
    N = len(orbitals)
    matrix = np.zeros((N, N), dtype=complex)

    for i, pos in enumerate(positions):
        for j, orbital in enumerate(orbitals):
            matrix[i, j] = orbital(pos)

    det = np.linalg.det(matrix)
    return det / np.sqrt(np.math.factorial(N))

# Define simple orbitals (1D harmonic oscillator)
def psi_0(x):
    """Ground state of HO"""
    return np.exp(-x**2 / 2) * np.pi**(-0.25)

def psi_1(x):
    """First excited state of HO"""
    return np.sqrt(2) * x * np.exp(-x**2 / 2) * np.pi**(-0.25)

def psi_2(x):
    """Second excited state of HO"""
    return (2*x**2 - 1) / np.sqrt(2) * np.exp(-x**2 / 2) * np.pi**(-0.25)

# Test: Two particles in states 0 and 1
orbitals = [psi_0, psi_1]
positions = [0.5, 1.0]

psi = slater_determinant(orbitals, positions)
print(f"\nΨ(x₁=0.5, x₂=1.0) = {psi:.6f}")

# Verify antisymmetry
psi_exchanged = slater_determinant(orbitals, [1.0, 0.5])
print(f"Ψ(x₁=1.0, x₂=0.5) = {psi_exchanged:.6f}")
print(f"Antisymmetric? {np.isclose(psi, -psi_exchanged)}")

# ============================================================
# VISUALIZATION: 2-PARTICLE WAVE FUNCTION
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZING 2-FERMION SLATER DETERMINANT")
print("=" * 60)

x = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x, x)

# Calculate Slater determinant on grid
Psi = np.zeros_like(X1)
for i in range(len(x)):
    for j in range(len(x)):
        Psi[i, j] = slater_determinant([psi_0, psi_1], [X1[i,j], X2[i,j]])

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Wave function
im0 = axes[0].contourf(X1, X2, Psi, levels=50, cmap='RdBu_r')
axes[0].plot(x, x, 'k--', alpha=0.5)
axes[0].set_xlabel('$x_1$', fontsize=12)
axes[0].set_ylabel('$x_2$', fontsize=12)
axes[0].set_title('Slater Determinant Ψ(x₁, x₂)', fontsize=12)
axes[0].set_aspect('equal')
plt.colorbar(im0, ax=axes[0])

# Probability density
im1 = axes[1].contourf(X1, X2, Psi**2, levels=50, cmap='viridis')
axes[1].plot(x, x, 'w--', alpha=0.5)
axes[1].set_xlabel('$x_1$', fontsize=12)
axes[1].set_ylabel('$x_2$', fontsize=12)
axes[1].set_title('|Ψ|² (Probability Density)', fontsize=12)
axes[1].set_aspect('equal')
plt.colorbar(im1, ax=axes[1])

# Diagonal slice
axes[2].plot(x, Psi[50, :], 'b-', linewidth=2, label='Ψ(x, x₂=0)')
axes[2].plot(x, Psi[:, 50], 'r--', linewidth=2, label='Ψ(x₁=0, x)')
axes[2].axhline(0, color='k', linewidth=0.5)
axes[2].set_xlabel('Position', fontsize=12)
axes[2].set_ylabel('Ψ', fontsize=12)
axes[2].set_title('Cross Sections', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('slater_determinant_2d.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# EXCHANGE HOLE
# ============================================================

print("\n" + "=" * 60)
print("THE EXCHANGE HOLE")
print("=" * 60)

# Fix particle 1 at x₁ = 0, plot probability of particle 2
x1_fixed = 0
prob_x2 = np.array([slater_determinant([psi_0, psi_1], [x1_fixed, x2])**2 for x2 in x])

# Compare to uncorrelated (product) probability
prob_uncorrelated = (psi_0(x)**2 + psi_1(x)**2) / 2  # Average density

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, prob_x2, 'b-', linewidth=2, label='Correlated |Ψ(0, x₂)|²')
ax.plot(x, prob_uncorrelated, 'r--', linewidth=2, label='Uncorrelated')
ax.axvline(x1_fixed, color='green', linestyle=':', label=f'x₁ = {x1_fixed}')

ax.fill_between(x, prob_x2, prob_uncorrelated, where=prob_x2<prob_uncorrelated,
                alpha=0.3, color='gray', label='Exchange hole')

ax.set_xlabel('Position x₂', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Exchange Hole: Fermions Avoid Each Other', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exchange_hole.png', dpi=150, bbox_inches='tight')
plt.show()

print("The exchange hole shows reduced probability of finding")
print("a second fermion near the first - antisymmetry in action!")

# ============================================================
# EXPLICIT DETERMINANT FOR HELIUM
# ============================================================

print("\n" + "=" * 60)
print("HELIUM GROUND STATE")
print("=" * 60)

def helium_1s(r, Z=2):
    """Helium 1s orbital (normalized)"""
    a0 = 1  # Atomic units
    return np.sqrt(Z**3 / np.pi) * np.exp(-Z * r / a0)

# The spatial part is symmetric, spin part is antisymmetric
# Ψ(r₁,s₁,r₂,s₂) = ψ_1s(r₁)ψ_1s(r₂) × (1/√2)[α(s₁)β(s₂) - β(s₁)α(s₂)]

print("""
Helium Ground State (1s²):

Spatial: Ψ_space(r₁,r₂) = ψ_1s(r₁) × ψ_1s(r₂)  [SYMMETRIC]

Spin:    χ_spin(s₁,s₂) = (1/√2)[|↑↓⟩ - |↓↑⟩]   [ANTISYMMETRIC]
                       = Singlet (S=0)

Total:   Ψ_total = Ψ_space × χ_spin            [ANTISYMMETRIC]

The Slater determinant automatically handles this!
""")

# ============================================================
# THREE-PARTICLE EXAMPLE
# ============================================================

print("\n" + "=" * 60)
print("THREE-PARTICLE SLATER DETERMINANT (LITHIUM)")
print("=" * 60)

print("""
Lithium (1s² 2s¹):

            | 1sα(1)  1sβ(1)  2sα(1) |
Ψ = (1/√6) | 1sα(2)  1sβ(2)  2sα(2) |
            | 1sα(3)  1sβ(3)  2sα(3) |

Expanding:
Ψ = (1/√6)[1sα(1)⋅|1sβ(2) 2sα(2)|  - 1sβ(1)⋅|1sα(2) 2sα(2)| + ...]
           |1sβ(3) 2sα(3)|         |1sα(3) 2sα(3)|

This gives 3! = 6 terms, each with coefficient ±1/√6.
""")

# Count terms in general N-particle determinant
for N in range(2, 8):
    terms = np.math.factorial(N)
    print(f"N = {N}: {terms} terms in Slater determinant")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("KEY PROPERTIES OF SLATER DETERMINANTS")
print("=" * 60)
print("""
1. ANTISYMMETRY: Built-in from determinant properties
2. PAULI EXCLUSION: Identical columns → det = 0
3. NORMALIZATION: Factor 1/√N! for orthonormal orbitals
4. SINGLE-PARTICLE: Simplest many-fermion approximation
5. HARTREE-FOCK: Optimal single-determinant wave function
6. CORRELATION: Beyond single determinant → CI, CC methods
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Slater determinant | $\Psi = \frac{1}{\sqrt{N!}}\det[\chi_j(\mathbf{x}_i)]$ |
| Antisymmetry | Exchange rows → sign change |
| Pauli exclusion | Identical columns → det = 0 |
| One-body expectation | $\langle\hat{F}\rangle = \sum_i \langle\chi_i\|\hat{f}\|\chi_i\rangle$ |

### Main Takeaways

1. **Slater determinants** automatically enforce antisymmetry
2. **Pauli exclusion** emerges from determinant properties
3. **Hartree-Fock** finds optimal single-determinant approximation
4. **Electron correlation** requires multiple determinants
5. Essential for **molecular simulation** on quantum computers

---

## Daily Checklist

- [ ] I can construct Slater determinants for N fermions
- [ ] I understand why determinants give antisymmetric functions
- [ ] I can write Slater determinants for atoms
- [ ] I know the limitations of single-determinant descriptions
- [ ] I completed the computational lab

---

## Preview: Day 482

Tomorrow we study **exchange forces**—the effective interaction between identical particles arising from exchange symmetry.

---

**Next:** [Day_482_Saturday.md](Day_482_Saturday.md) — Exchange Forces
