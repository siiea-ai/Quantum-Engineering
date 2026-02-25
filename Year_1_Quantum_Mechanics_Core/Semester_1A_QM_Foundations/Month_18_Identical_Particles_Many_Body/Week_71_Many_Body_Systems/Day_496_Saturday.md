# Day 496: Hartree-Fock Introduction

## Overview

**Day 496 of 2520 | Week 71, Day 6 | Month 18: Identical Particles & Many-Body Physics**

Today we introduce the Hartree-Fock method—the cornerstone of modern computational quantum chemistry. This self-consistent field approach provides a systematic way to find the best single-determinant wave function for a many-electron system. We will derive the Hartree and Fock equations, understand the role of exchange, and connect this classical method to the quantum computing approach of VQE.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | From Central Field to Self-Consistency | 45 min |
| 9:45 AM | The Hartree Approximation | 75 min |
| 11:00 AM | Break | 15 min |
| 11:15 AM | Exchange and the Fock Operator | 90 min |
| 12:45 PM | Lunch | 60 min |
| 1:45 PM | Self-Consistent Field Procedure | 75 min |
| 3:00 PM | Break | 15 min |
| 3:15 PM | Connection to VQE | 60 min |
| 4:15 PM | Computational Lab | 75 min |
| 5:30 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** the variational principle applied to Slater determinants
2. **Derive** the Hartree equations for the mean-field approximation
3. **Understand** how exchange leads to the Fock operator
4. **Describe** the self-consistent field iteration procedure
5. **Connect** Hartree-Fock to quantum computing VQE methods
6. **Identify** the limitations of Hartree-Fock and need for correlation

---

## 1. From Variational Principle to Hartree-Fock

### The Best Single-Determinant Approximation

**Goal:** Find the best possible wave function of the form:

$$\Psi = \frac{1}{\sqrt{N!}} \begin{vmatrix} \chi_1(\mathbf{x}_1) & \chi_2(\mathbf{x}_1) & \cdots & \chi_N(\mathbf{x}_1) \\ \chi_1(\mathbf{x}_2) & \chi_2(\mathbf{x}_2) & \cdots & \chi_N(\mathbf{x}_2) \\ \vdots & \vdots & \ddots & \vdots \\ \chi_1(\mathbf{x}_N) & \chi_2(\mathbf{x}_N) & \cdots & \chi_N(\mathbf{x}_N) \end{vmatrix}$$

where $\chi_i(\mathbf{x}) = \phi_i(\mathbf{r})\sigma(s)$ are **spin orbitals** and $\mathbf{x} = (\mathbf{r}, s)$ includes both space and spin.

### Why Slater Determinant?

1. **Automatically antisymmetric** under particle exchange
2. **Satisfies Pauli exclusion** (determinant = 0 if two orbitals are the same)
3. **Reduces N-body problem** to effective one-body equations

### Variational Optimization

Minimize:
$$E[\{\chi_i\}] = \langle \Psi | \hat{H} | \Psi \rangle$$

subject to orthonormality constraints:
$$\langle \chi_i | \chi_j \rangle = \delta_{ij}$$

Using Lagrange multipliers, this leads to the **Hartree-Fock equations**.

---

## 2. The Hartree Approximation

### Ignoring Exchange (Historical First Step)

Douglas Hartree (1928) proposed using a simple product wave function:

$$\Psi_{\text{Hartree}} = \phi_1(\mathbf{r}_1)\phi_2(\mathbf{r}_2)\cdots\phi_N(\mathbf{r}_N)$$

This ignores antisymmetry but captures the mean-field idea.

### The Hartree Equations

Each electron moves in the average potential of all others:

$$\boxed{\left[-\frac{1}{2}\nabla^2 - \frac{Z}{r} + V_H^{(i)}(\mathbf{r})\right]\phi_i(\mathbf{r}) = \varepsilon_i \phi_i(\mathbf{r})}$$

where the **Hartree potential** is:

$$\boxed{V_H^{(i)}(\mathbf{r}) = \sum_{j \neq i} \int d^3r' \, \frac{|\phi_j(\mathbf{r}')|^2}{|\mathbf{r} - \mathbf{r}'|}}$$

### Physical Interpretation

- Each electron feels the **electrostatic potential** created by the charge density of all other electrons
- The potential $V_H^{(i)}$ depends on the orbitals $\{\phi_j\}$
- The orbitals depend on the potential
- **Self-consistency required!**

### Hartree Energy

$$E_{\text{Hartree}} = \sum_i \varepsilon_i - \frac{1}{2}\sum_{i \neq j} J_{ij}$$

where the **Coulomb integral**:
$$J_{ij} = \int d^3r \int d^3r' \, \frac{|\phi_i(\mathbf{r})|^2 |\phi_j(\mathbf{r}')|^2}{|\mathbf{r} - \mathbf{r}'|}$$

The factor of 1/2 corrects for double-counting in the orbital energies.

---

## 3. Including Exchange: The Fock Operator

### What's Missing in Hartree?

The Hartree approximation:
1. Does **not** enforce antisymmetry (Pauli principle)
2. Misses **exchange energy** (from indistinguishability)
3. Allows unphysical self-interaction

### The Fock Operator

Vladimir Fock (1930) extended Hartree's approach using Slater determinants:

$$\boxed{\hat{f}(1)\chi_i(1) = \varepsilon_i \chi_i(1)}$$

where the **Fock operator** is:

$$\hat{f}(1) = \hat{h}(1) + \sum_j [\hat{J}_j(1) - \hat{K}_j(1)]$$

### Components of the Fock Operator

**One-electron part:**
$$\hat{h}(1) = -\frac{1}{2}\nabla_1^2 - \frac{Z}{r_1}$$

**Coulomb operator** (direct interaction):
$$\hat{J}_j(1)\chi_i(1) = \left[\int d\mathbf{x}_2 \, \chi_j^*(2)\frac{1}{r_{12}}\chi_j(2)\right]\chi_i(1)$$

**Exchange operator** (non-local!):
$$\boxed{\hat{K}_j(1)\chi_i(1) = \left[\int d\mathbf{x}_2 \, \chi_j^*(2)\frac{1}{r_{12}}\chi_i(2)\right]\chi_j(1)}$$

### Exchange: The Key Difference

Note the **exchange of labels** in the exchange operator: electron 2 enters with orbital $\chi_i$ but exits with orbital $\chi_j$.

This makes the exchange operator:
- **Non-local**: Depends on the wave function everywhere
- **Purely quantum**: No classical analog
- **Responsible for**: Exchange energy stabilization of parallel spins

### No Self-Interaction

For $i = j$:
$$\hat{J}_i\chi_i - \hat{K}_i\chi_i = 0$$

The Coulomb and exchange terms cancel for an electron interacting with "itself."

This is a major advantage over simple Hartree!

---

## 4. The Hartree-Fock Equations

### Canonical Form

The Hartree-Fock equations can be written:

$$\boxed{\hat{f}|\chi_i\rangle = \varepsilon_i|\chi_i\rangle}$$

This is an **eigenvalue problem** for the Fock operator.

### Closed-Shell Restricted Hartree-Fock (RHF)

For systems with all electrons paired (closed shell):

$$\hat{f}|\phi_i\rangle = \varepsilon_i|\phi_i\rangle$$

Each spatial orbital $\phi_i$ is doubly occupied (spin up and spin down).

### Hartree-Fock Energy

$$\boxed{E_{HF} = \sum_i \langle\chi_i|\hat{h}|\chi_i\rangle + \frac{1}{2}\sum_{ij}(J_{ij} - K_{ij})}$$

where:
- $J_{ij} = \langle\chi_i\chi_j|\chi_i\chi_j\rangle$ (Coulomb integral)
- $K_{ij} = \langle\chi_i\chi_j|\chi_j\chi_i\rangle$ (Exchange integral)

### Koopman's Theorem

The orbital energies have physical meaning:

$$\boxed{-\varepsilon_i \approx IE_i}$$

The negative of an occupied orbital energy approximates the ionization energy.

For virtual (unoccupied) orbitals: $-\varepsilon_a \approx EA_a$ (electron affinity)

---

## 5. Self-Consistent Field (SCF) Procedure

### The Self-Consistency Problem

The Fock operator depends on the orbitals:
$$\hat{f}[\{\chi\}]|\chi_i\rangle = \varepsilon_i|\chi_i\rangle$$

But we need the orbitals to construct the Fock operator!

**Solution:** Iterate until self-consistent.

### SCF Algorithm

```
1. INITIAL GUESS
   - Start with approximate orbitals (e.g., from simpler calculation)

2. BUILD FOCK MATRIX
   - Compute J and K operators from current orbitals
   - Construct F = h + J - K

3. SOLVE EIGENVALUE PROBLEM
   - Diagonalize: F C = C ε
   - Get new orbitals and energies

4. CHECK CONVERGENCE
   - Compare new density with old
   - If |ρ_new - ρ_old| < threshold: DONE
   - Else: Go to step 2

5. OUTPUT
   - Converged orbitals
   - Total energy
   - Other properties
```

### Convergence Considerations

- SCF may **oscillate** or **diverge** without care
- Techniques to help: **DIIS** (Direct Inversion of Iterative Subspace)
- Typically converges in 10-30 iterations
- More difficult for transition metals and open shells

### Basis Set Expansion

In practice, expand orbitals in a **basis set**:

$$\phi_i(\mathbf{r}) = \sum_\mu c_{\mu i} \chi_\mu(\mathbf{r})$$

Common basis functions:
- **Slater-type orbitals (STOs):** $\chi \propto r^{n-1}e^{-\zeta r}Y_l^m$
- **Gaussian-type orbitals (GTOs):** $\chi \propto r^l e^{-\alpha r^2}Y_l^m$

GTOs are less accurate but much easier to compute integrals!

---

## 6. Hartree-Fock Results and Limitations

### Helium Hartree-Fock Energy

$$E_{HF}(\text{He}) = -2.8617 \text{ Hartree}$$

Compare:
| Method | Energy (Hartree) | Error (eV) |
|--------|------------------|------------|
| Zeroth order | -4.000 | +29.8 |
| First-order pert. | -2.750 | -4.2 |
| Variational $Z_{\text{eff}}$ | -2.848 | -1.5 |
| **Hartree-Fock** | **-2.8617** | **-1.1** |
| Exact | -2.9037 | 0 |

### Correlation Energy

The difference between exact and Hartree-Fock:

$$\boxed{E_{\text{corr}} = E_{\text{exact}} - E_{HF}}$$

For helium: $E_{\text{corr}} = -0.042$ Hartree = -1.1 eV

### Why Does Hartree-Fock Miss Correlation?

Hartree-Fock describes each electron in the **average** field of others.

**Misses:**
- **Dynamic correlation:** Instantaneous electron-electron avoidance
- **Static correlation:** Near-degeneracy effects
- **Dispersion:** Long-range correlation (van der Waals)

### Percentage of Correlation Energy

| System | $E_{HF}$ (Ha) | $E_{\text{corr}}$ (Ha) | % correlation |
|--------|---------------|------------------------|---------------|
| He | -2.862 | -0.042 | 1.5% |
| Be | -14.573 | -0.094 | 0.6% |
| Ne | -128.547 | -0.390 | 0.3% |
| Ar | -526.817 | -0.722 | 0.1% |

Small percentage, but chemically important! Chemical accuracy requires ~0.001 Ha.

---

## 7. Connection to VQE

### Hartree-Fock as Reference

In quantum computing approaches:

1. **Classical HF calculation** provides a reference state
2. **VQE** builds on top with parameterized unitary
3. **Captures correlation** that HF misses

### The UCCSD Ansatz

Unitary Coupled Cluster Singles and Doubles:

$$|\Psi_{VQE}\rangle = e^{\hat{T} - \hat{T}^\dagger}|\Psi_{HF}\rangle$$

where $\hat{T}$ creates excitations from the HF reference:

$$\hat{T} = \sum_{ia} t_i^a \hat{a}^\dagger_a \hat{a}_i + \sum_{ijab} t_{ij}^{ab} \hat{a}^\dagger_a \hat{a}^\dagger_b \hat{a}_j \hat{a}_i$$

### Why HF + VQE?

1. **HF captures ~99%** of the energy
2. **VQE focuses on** the harder 1% (correlation)
3. **Smaller parameter space** than starting from scratch
4. **Chemically motivated** excitation structure

### Quantum vs Classical

**Classical post-HF methods:**
- Configuration Interaction (CI)
- Coupled Cluster (CC)
- Perturbation theory (MPn)

Scaling: $O(N^6)$ to $O(N^7)$ for accurate methods

**Quantum VQE:**
- Polynomial scaling with system size
- But current hardware limitations
- Noise and error issues

---

## 8. Worked Examples

### Example 1: Hartree-Fock Energy Expression

**Problem:** Write the Hartree-Fock energy for a two-electron system (helium).

**Solution:**

For helium with both electrons in orbital $\phi_{1s}$:

$$E_{HF} = 2\langle\phi_{1s}|\hat{h}|\phi_{1s}\rangle + J_{11} - K_{11}$$

But $K_{11} = J_{11}$ for the same orbital, so:

$$E_{HF} = 2\langle\phi_{1s}|\hat{h}|\phi_{1s}\rangle + J_{11}$$

Explicitly:
- $\langle\phi_{1s}|\hat{h}|\phi_{1s}\rangle = $ kinetic + electron-nucleus
- $J_{11} = $ Coulomb self-repulsion of 1s orbital

With optimal $\phi_{1s}$: $E_{HF} = -2.8617$ Ha

$$\boxed{E_{HF}(\text{He}) = 2h_{11} + J_{11} = -2.8617 \text{ Ha}}$$

### Example 2: Exchange Integral

**Problem:** Show that for real orbitals, $K_{ij} \geq 0$.

**Solution:**

$$K_{ij} = \int d^3r_1 \int d^3r_2 \, \phi_i(\mathbf{r}_1)\phi_j(\mathbf{r}_1) \frac{1}{r_{12}} \phi_i(\mathbf{r}_2)\phi_j(\mathbf{r}_2)$$

Define $\rho_{ij}(\mathbf{r}) = \phi_i(\mathbf{r})\phi_j(\mathbf{r})$ (overlap density).

$$K_{ij} = \int d^3r_1 \int d^3r_2 \, \frac{\rho_{ij}(\mathbf{r}_1)\rho_{ij}(\mathbf{r}_2)}{r_{12}}$$

This is the self-repulsion energy of the charge distribution $\rho_{ij}$.

For real orbitals, this integral is always positive!

$$\boxed{K_{ij} \geq 0 \text{ for real orbitals}}$$

### Example 3: Koopmans' Theorem Application

**Problem:** The HOMO energy of water from HF calculation is $\varepsilon_{HOMO} = -0.50$ Ha. Estimate the first ionization energy.

**Solution:**

By Koopmans' theorem:
$$IE_1 \approx -\varepsilon_{HOMO} = 0.50 \text{ Ha} = 13.6 \text{ eV}$$

Experimental value: 12.6 eV

The ~1 eV difference is due to:
- Orbital relaxation (ignored in Koopmans')
- Correlation effects

$$\boxed{IE_1 \approx 13.6 \text{ eV} \text{ (Koopmans')}}$$

---

## 9. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Write the Fock operator explicitly for a two-electron system.

**Problem 1.2:** How many Coulomb integrals $J_{ij}$ and exchange integrals $K_{ij}$ are there for a system with 4 occupied orbitals?

**Problem 1.3:** If the HF energy of an atom is -100.000 Ha and the exact energy is -100.250 Ha, what is the correlation energy in eV?

### Level 2: Intermediate

**Problem 2.1:** Show that the Hartree-Fock equations are invariant under unitary transformation of the occupied orbitals.

**Problem 2.2:** For a closed-shell system, show that the exchange energy contribution is:
$$E_x = -\frac{1}{2}\sum_{ij} K_{ij}$$

**Problem 2.3:** Explain why the SCF procedure might fail to converge for a stretched H₂ molecule.

### Level 3: Challenging

**Problem 3.1:** Derive the Roothaan-Hall equations $\mathbf{FC} = \mathbf{SC}\boldsymbol{\varepsilon}$ from the Hartree-Fock equations using basis set expansion.

**Problem 3.2:** Show that for a uniform electron gas, the exchange energy per electron scales as $\rho^{1/3}$ where $\rho$ is the density.

**Problem 3.3:** Estimate the correlation energy of beryllium using the relationship $E_{\text{corr}} \approx -0.04N$ Hartree, where N is the number of electron pairs.

---

## 10. Computational Lab: Hartree-Fock Implementation

```python
"""
Day 496 Computational Lab: Hartree-Fock Method
Simple demonstration of the self-consistent field procedure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar
from scipy.special import factorial

HARTREE_TO_EV = 27.211

def hydrogen_1s(r, zeta=1.0):
    """Normalized 1s Slater-type orbital."""
    return np.sqrt(zeta**3 / np.pi) * np.exp(-zeta * r)

def compute_one_electron_integral(zeta, Z=2):
    """
    Compute <phi|h|phi> = <phi|T|phi> + <phi|V_ne|phi>
    for 1s orbital with exponent zeta.
    """
    # Kinetic energy: <T> = zeta^2 / 2
    T = zeta**2 / 2

    # Nuclear attraction: <V_ne> = -Z * zeta
    V_ne = -Z * zeta

    return T + V_ne

def compute_coulomb_integral_1s(zeta):
    """
    Compute J_11 = <11|11> for 1s orbital.
    J = (5/8) * zeta for 1s orbital.
    """
    return (5/8) * zeta

def helium_hf_energy(zeta, Z=2):
    """
    Compute Hartree-Fock energy for helium with 1s orbital exponent zeta.
    E_HF = 2 * h_11 + J_11
    """
    h_11 = compute_one_electron_integral(zeta, Z)
    J_11 = compute_coulomb_integral_1s(zeta)

    return 2 * h_11 + J_11

def optimize_hf_helium():
    """Find optimal zeta for helium HF calculation."""

    # Minimize energy with respect to zeta
    result = minimize_scalar(
        lambda z: helium_hf_energy(z, Z=2),
        bounds=(0.5, 3.0),
        method='bounded'
    )

    zeta_opt = result.x
    E_opt = result.fun

    return zeta_opt, E_opt

def scf_iteration_demo():
    """Demonstrate SCF iteration procedure."""

    print("=" * 60)
    print("SELF-CONSISTENT FIELD (SCF) ITERATION FOR HELIUM")
    print("=" * 60)

    Z = 2
    zeta = 1.0  # Initial guess (hydrogen-like)

    print(f"\nInitial guess: zeta = {zeta:.4f}")
    print("-" * 60)
    print(f"{'Iteration':<12} {'Zeta':<12} {'Energy (Ha)':<15} {'Delta E':<15}")
    print("-" * 60)

    E_old = helium_hf_energy(zeta, Z)
    print(f"{0:<12} {zeta:<12.6f} {E_old:<15.8f} {'---':<15}")

    # Simple fixed-point iteration (in reality, more sophisticated)
    for iteration in range(1, 20):
        # Update zeta based on energy minimization
        result = minimize_scalar(
            lambda z: helium_hf_energy(z, Z),
            bounds=(zeta - 0.5, zeta + 0.5),
            method='bounded'
        )
        zeta_new = result.x
        E_new = result.fun

        delta_E = abs(E_new - E_old)
        print(f"{iteration:<12} {zeta_new:<12.6f} {E_new:<15.8f} {delta_E:<15.2e}")

        if delta_E < 1e-8:
            print("\nConverged!")
            break

        zeta = zeta_new
        E_old = E_new

    print("-" * 60)
    print(f"\nFinal Results:")
    print(f"  Optimal zeta = {zeta_new:.6f}")
    print(f"  HF Energy = {E_new:.6f} Hartree = {E_new * HARTREE_TO_EV:.4f} eV")
    print(f"  Exact Energy = -2.9037 Hartree")
    print(f"  Correlation Energy = {-2.9037 - E_new:.6f} Hartree")

    return zeta_new, E_new

def plot_orbital_evolution():
    """Show how orbital changes during SCF."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    r = np.linspace(0, 5, 200)

    # Different zeta values representing SCF convergence
    zetas = [1.0, 1.3, 1.5, 1.6875]
    labels = ['Initial (ζ=1.0)', 'Iteration 1', 'Iteration 2', 'Converged (ζ=1.69)']
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(zetas)))

    ax1 = axes[0]
    for zeta, label, color in zip(zetas, labels, colors):
        psi = hydrogen_1s(r, zeta)
        ax1.plot(r, psi, label=label, color=color, linewidth=2)

    ax1.set_xlabel('r (a₀)', fontsize=12)
    ax1.set_ylabel('ψ(r)', fontsize=12)
    ax1.set_title('Orbital Evolution During SCF', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy vs zeta
    ax2 = axes[1]
    zeta_range = np.linspace(0.5, 2.5, 100)
    energies = [helium_hf_energy(z, Z=2) for z in zeta_range]

    ax2.plot(zeta_range, energies, 'b-', linewidth=2)

    # Mark optimal point
    zeta_opt, E_opt = optimize_hf_helium()
    ax2.plot(zeta_opt, E_opt, 'ro', markersize=10, label=f'Optimal: ζ={zeta_opt:.4f}')
    ax2.axhline(y=-2.9037, color='green', linestyle='--', label='Exact')

    ax2.set_xlabel('Orbital Exponent ζ', fontsize=12)
    ax2.set_ylabel('Energy (Hartree)', fontsize=12)
    ax2.set_title('HF Energy vs Orbital Exponent', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scf_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

def coulomb_exchange_visualization():
    """Visualize Coulomb and exchange contributions."""

    print("\n" + "=" * 60)
    print("COULOMB AND EXCHANGE INTEGRALS")
    print("=" * 60)

    zeta_opt = 1.6875

    # One-electron integral
    h_11 = compute_one_electron_integral(zeta_opt, Z=2)
    print(f"\nOne-electron integral h_11 = {h_11:.6f} Ha")
    print(f"  Kinetic: {zeta_opt**2/2:.6f} Ha")
    print(f"  Nuclear attraction: {-2*zeta_opt:.6f} Ha")

    # Coulomb integral
    J_11 = compute_coulomb_integral_1s(zeta_opt)
    print(f"\nCoulomb integral J_11 = {J_11:.6f} Ha")

    # For closed shell, exchange within same orbital cancels Coulomb
    print(f"\nExchange integral K_11 = {J_11:.6f} Ha (equals J_11)")

    # Total HF energy
    E_HF = 2 * h_11 + J_11
    print(f"\nHartree-Fock energy:")
    print(f"  E_HF = 2*h_11 + J_11")
    print(f"       = 2*({h_11:.4f}) + {J_11:.4f}")
    print(f"       = {E_HF:.6f} Ha")

    # Breakdown
    print("\nEnergy Breakdown:")
    print(f"  Kinetic energy (2T): {2*zeta_opt**2/2:.4f} Ha")
    print(f"  Nuclear attraction: {-2*2*zeta_opt:.4f} Ha")
    print(f"  Electron repulsion: {J_11:.4f} Ha")
    print(f"  Total: {E_HF:.4f} Ha")

def vqe_connection():
    """Explain connection to VQE."""

    print("\n" + "=" * 60)
    print("HARTREE-FOCK TO VQE CONNECTION")
    print("=" * 60)

    print("""
    FROM CLASSICAL HF TO QUANTUM VQE
    =================================

    1. HARTREE-FOCK AS REFERENCE
       |Ψ_HF⟩ = |1s↑, 1s↓⟩ for helium

       In second quantization:
       |Ψ_HF⟩ = c†_{1s↑} c†_{1s↓} |0⟩

       On qubits (Jordan-Wigner):
       |Ψ_HF⟩ = |1100⟩ (4 spin-orbitals: 1s↑, 1s↓, 2s↑, 2s↓)

    2. VQE ANSATZ BUILT ON HF
       |Ψ_VQE⟩ = U(θ) |Ψ_HF⟩

       U(θ) includes excitations:
       - Singles: c†_a c_i (e.g., 1s→2s)
       - Doubles: c†_a c†_b c_j c_i

       UCCSD ansatz:
       U = exp[T - T†] where T = T₁ + T₂

    3. WHAT VQE CAPTURES BEYOND HF
       - Dynamic correlation (electron avoidance)
       - Left-right correlation (important for bonds)
       - Size-consistency (if ansatz is good)

    4. ENERGY COMPARISON FOR HELIUM
       Method          Energy (Ha)    Correlation captured
       ------------------------------------------------
       Hartree-Fock    -2.8617        0%
       VQE (minimal)   -2.875         ~30%
       VQE (extended)  -2.895         ~75%
       Full CI         -2.9037        100%

    5. ADVANTAGE OF QUANTUM
       - Polynomial scaling vs exponential classical
       - Natural representation of fermionic states
       - Direct access to multi-reference states

    6. CURRENT CHALLENGES
       - Noise in quantum hardware
       - Barren plateaus in optimization
       - Depth of required circuits
    """)

def compare_methods():
    """Compare different approximation methods."""

    print("\n" + "=" * 60)
    print("COMPARISON OF APPROXIMATION METHODS FOR HELIUM")
    print("=" * 60)

    methods = [
        ('Zeroth Order', -4.0, 'E = -Z²'),
        ('First-Order Pert.', -2.75, 'E = -Z² + 5Z/8'),
        ('Variational (Z_eff)', -2.8477, 'Optimize Z_eff'),
        ('Hartree-Fock', -2.8617, 'Self-consistent field'),
        ('Hylleraas (3 param)', -2.891, 'Explicit r₁₂ correlation'),
        ('Exact', -2.9037, 'Full solution'),
    ]

    E_exact = -2.9037

    print(f"\n{'Method':<25} {'E (Ha)':<12} {'Error (eV)':<12} {'% Corr':<10}")
    print("-" * 60)

    for method, E, description in methods:
        error_eV = (E - E_exact) * HARTREE_TO_EV
        if method == 'Exact':
            pct_corr = 100
        else:
            E_corr_total = E_exact - (-2.8617)  # HF correlation
            E_corr_captured = E_exact - E
            if E_corr_total != 0:
                pct_corr = 100 * (1 - E_corr_captured / E_corr_total)
            else:
                pct_corr = 0

        print(f"{method:<25} {E:<12.4f} {error_eV:<12.3f} {pct_corr:<10.1f}")

def main():
    """Run all demonstrations."""

    print("Day 496: Hartree-Fock Introduction")
    print("=" * 60)

    # SCF iteration
    zeta_opt, E_opt = scf_iteration_demo()

    # Coulomb and exchange
    coulomb_exchange_visualization()

    # Plot orbital evolution
    plot_orbital_evolution()

    # Compare methods
    compare_methods()

    # VQE connection
    vqe_connection()

if __name__ == "__main__":
    main()
```

---

## 11. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Slater determinant | Antisymmetric N-electron wave function from single orbitals |
| Hartree equations | Mean-field approximation ignoring exchange |
| Fock operator | Includes exchange via $\hat{K}$ operator |
| Self-consistency | Orbitals and potential determined simultaneously |
| Correlation energy | $E_{\text{corr}} = E_{\text{exact}} - E_{HF}$ |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$\hat{f} = \hat{h} + \sum_j(\hat{J}_j - \hat{K}_j)$$ | Fock operator |
| $$E_{HF} = \sum_i h_{ii} + \frac{1}{2}\sum_{ij}(J_{ij} - K_{ij})$$ | HF energy |
| $$-\varepsilon_i \approx IE_i$$ | Koopmans' theorem |
| $$E_{\text{corr}} = E_{\text{exact}} - E_{HF}$$ | Correlation energy definition |

---

## 12. Daily Checklist

### Conceptual Understanding
- [ ] I can explain why we use Slater determinants
- [ ] I understand the difference between Coulomb and exchange operators
- [ ] I know why SCF is required (self-consistency)
- [ ] I can describe what correlation energy represents

### Mathematical Skills
- [ ] I can write the Fock operator for a simple system
- [ ] I can calculate HF energy from integrals
- [ ] I can apply Koopmans' theorem
- [ ] I can describe the SCF algorithm

### Computational Skills
- [ ] I implemented a simple SCF calculation
- [ ] I visualized orbital convergence
- [ ] I compared HF with other methods

### Quantum Computing Connection
- [ ] I see how HF provides a reference for VQE
- [ ] I understand what VQE adds beyond HF
- [ ] I know the advantages and challenges of quantum approaches

---

## 13. Preview: Day 497

Tomorrow is **Week 71 Review**:

- Comprehensive summary of helium and many-body methods
- Problem set covering all topics
- Self-assessment
- Preview of advanced many-body techniques

---

## References

1. Szabo, A. & Ostlund, N.S. (1996). *Modern Quantum Chemistry*. Dover, Ch. 3.

2. Fock, V. (1930). "Näherungsmethode zur Lösung des quantenmechanischen Mehrkörperproblems." Z. Physik, 61, 126.

3. Levine, I.N. (2013). *Quantum Chemistry*, 7th ed., Ch. 11.

4. Helgaker, T., Jorgensen, P., & Olsen, J. (2014). *Molecular Electronic-Structure Theory*. Wiley.

---

*"The Hartree-Fock method is the foundation upon which almost all of computational quantum chemistry is built. While it misses correlation, understanding its successes and limitations is essential for any serious student of electronic structure theory."*
— Per-Olov Löwdin

---

**Day 496 Complete.** Tomorrow: Week 71 Review.
