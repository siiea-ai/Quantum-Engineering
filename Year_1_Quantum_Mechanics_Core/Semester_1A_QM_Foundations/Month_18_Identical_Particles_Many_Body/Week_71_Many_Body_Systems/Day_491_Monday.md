# Day 491: Helium Atom Setup

## Overview

**Day 491 of 2520 | Week 71, Day 1 | Month 18: Identical Particles & Many-Body Physics**

Today we begin our study of the helium atom—the simplest multi-electron system and the prototype for understanding electron-electron interactions in atoms. While hydrogen can be solved exactly, helium cannot: the electron-electron repulsion term destroys separability. This makes helium the ideal testing ground for approximation methods that we will need for all larger atoms and molecules.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Helium: Why It Matters | 45 min |
| 9:45 AM | The Two-Electron Hamiltonian | 75 min |
| 11:00 AM | Break | 15 min |
| 11:15 AM | Independent Particle Approximation | 90 min |
| 12:45 PM | Lunch | 60 min |
| 1:45 PM | Ground State Configuration | 75 min |
| 3:00 PM | Break | 15 min |
| 3:15 PM | Atomic Units and Scaling | 60 min |
| 4:15 PM | Computational Lab | 75 min |
| 5:30 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Write** the complete Hamiltonian for helium including all terms
2. **Identify** why the electron-electron repulsion prevents exact solution
3. **Apply** the independent particle approximation to estimate ground state energy
4. **Calculate** the zeroth-order energy and compare with experiment
5. **Explain** why screening effects make the approximation poor
6. **Set up** the framework for perturbation and variational treatments

---

## 1. The Helium Atom: Why It Matters

### The Simplest Non-Trivial Atom

Helium is:
- **Atomic number Z = 2**: Two protons, two electrons
- **Simplest closed-shell atom**: Both electrons in 1s orbital
- **Chemically inert**: Full first shell (noble gas)
- **Abundant**: Second most common element in universe

### The Quantum Three-Body Problem

The helium atom consists of:
- Nucleus (essentially fixed, Born-Oppenheimer approximation)
- Electron 1 at position $\mathbf{r}_1$
- Electron 2 at position $\mathbf{r}_2$

This is a **quantum three-body problem** with:
- 2 electron-nucleus attractions (tractable)
- 1 electron-electron repulsion (the challenge!)

### Why Helium Defies Exact Solution

For hydrogen: $\hat{H} = \hat{T} + V(r)$ - central potential, separable in spherical coordinates

For helium: The term $\frac{e^2}{|\mathbf{r}_1 - \mathbf{r}_2|}$ depends on both electron positions and the angle between them. **No coordinate system separates all variables!**

### Historical Significance

| Year | Development |
|------|-------------|
| 1926 | Heisenberg's perturbation treatment |
| 1928 | Hylleraas variational calculation |
| 1930s | Systematic improvements |
| 1957 | Pekeris: 1078-parameter calculation |
| 2006 | Drake: 0.000000001 Hartree precision |

The helium atom remains a benchmark for computational methods.

---

## 2. The Two-Electron Hamiltonian

### Complete Hamiltonian

In SI units, the helium Hamiltonian is:

$$\boxed{\hat{H} = -\frac{\hbar^2}{2m_e}\nabla_1^2 - \frac{\hbar^2}{2m_e}\nabla_2^2 - \frac{Ze^2}{4\pi\epsilon_0 r_1} - \frac{Ze^2}{4\pi\epsilon_0 r_2} + \frac{e^2}{4\pi\epsilon_0 r_{12}}}$$

where:
- $r_1 = |\mathbf{r}_1|$ = distance of electron 1 from nucleus
- $r_2 = |\mathbf{r}_2|$ = distance of electron 2 from nucleus
- $r_{12} = |\mathbf{r}_1 - \mathbf{r}_2|$ = electron-electron separation
- $Z = 2$ for helium

### Term-by-Term Analysis

**Kinetic energy:**
$$\hat{T} = -\frac{\hbar^2}{2m_e}(\nabla_1^2 + \nabla_2^2)$$

**Electron-nucleus attraction:**
$$\hat{V}_{en} = -\frac{Ze^2}{4\pi\epsilon_0}\left(\frac{1}{r_1} + \frac{1}{r_2}\right)$$

**Electron-electron repulsion:**
$$\hat{V}_{ee} = \frac{e^2}{4\pi\epsilon_0 r_{12}}$$

### Atomic Units

For atomic calculations, we use **atomic units** (a.u.):
- $\hbar = 1$
- $m_e = 1$
- $e = 1$
- $4\pi\epsilon_0 = 1$
- Length unit: $a_0 = 0.529$ Å (Bohr radius)
- Energy unit: $E_h = 27.2$ eV (Hartree)

In atomic units:

$$\boxed{\hat{H} = -\frac{1}{2}\nabla_1^2 - \frac{1}{2}\nabla_2^2 - \frac{Z}{r_1} - \frac{Z}{r_2} + \frac{1}{r_{12}}}$$

This is **much cleaner** for calculations!

### Decomposition for Perturbation Theory

We write:
$$\hat{H} = \hat{H}_0 + \hat{H}'$$

where:
$$\hat{H}_0 = \left(-\frac{1}{2}\nabla_1^2 - \frac{Z}{r_1}\right) + \left(-\frac{1}{2}\nabla_2^2 - \frac{Z}{r_2}\right) = \hat{h}_1 + \hat{h}_2$$

$$\hat{H}' = \frac{1}{r_{12}}$$

The unperturbed Hamiltonian $\hat{H}_0$ is **separable** - it's just two independent hydrogen-like atoms!

---

## 3. Independent Particle Approximation

### The Zeroth-Order Problem

If we ignore electron-electron repulsion ($\hat{H}' = 0$):

$$\hat{H}_0 \Psi_0 = E_0^{(0)} \Psi_0$$

Since $\hat{H}_0 = \hat{h}_1 + \hat{h}_2$, the solution is:

$$\Psi_0(\mathbf{r}_1, \mathbf{r}_2) = \psi_{n_1 l_1 m_1}(\mathbf{r}_1) \cdot \psi_{n_2 l_2 m_2}(\mathbf{r}_2)$$

$$E_0^{(0)} = E_{n_1} + E_{n_2}$$

### Hydrogen-like Energies

For a hydrogen-like atom with nuclear charge Z:

$$E_n = -\frac{Z^2}{2n^2} \text{ Hartree} = -\frac{Z^2 \times 13.6}{n^2} \text{ eV}$$

### Ground State of Helium (Zeroth Order)

Both electrons in the 1s orbital ($n_1 = n_2 = 1$):

$$E_0^{(0)} = E_1 + E_1 = -\frac{Z^2}{2} - \frac{Z^2}{2} = -Z^2 \text{ Hartree}$$

For helium ($Z = 2$):

$$\boxed{E_0^{(0)} = -4 \text{ Hartree} = -108.8 \text{ eV}}$$

### Comparison with Experiment

| Quantity | Value |
|----------|-------|
| Zeroth-order energy | -108.8 eV |
| Experimental energy | -79.0 eV |
| **Error** | **29.8 eV (38%)** |

This is a **terrible** approximation! The error is almost 30 eV.

### Why So Bad?

The independent particle approximation:
1. **Ignores repulsion**: Electrons don't avoid each other
2. **Over-binds**: Both electrons feel full nuclear charge Z = 2
3. **No screening**: Each electron should partially shield the nucleus from the other

The electron-electron repulsion energy is approximately:
$$\langle \hat{V}_{ee} \rangle \approx +34 \text{ eV}$$

This is a **huge** correction that must be included!

---

## 4. The Ground State Wave Function

### Spatial Part

For both electrons in the 1s state, the spatial wave function is:

$$\psi_{1s}(\mathbf{r}) = \frac{1}{\sqrt{\pi}}\left(\frac{Z}{a_0}\right)^{3/2} e^{-Zr/a_0}$$

In atomic units ($a_0 = 1$):

$$\psi_{1s}(\mathbf{r}) = \sqrt{\frac{Z^3}{\pi}} e^{-Zr}$$

The spatial part of the ground state:

$$\Psi_0(\mathbf{r}_1, \mathbf{r}_2) = \psi_{1s}(\mathbf{r}_1)\psi_{1s}(\mathbf{r}_2) = \frac{Z^3}{\pi} e^{-Z(r_1 + r_2)}$$

### Symmetry Requirements

For two **identical fermions** (electrons), the total wave function must be **antisymmetric**:

$$\Psi_{\text{total}}(\mathbf{r}_1, s_1; \mathbf{r}_2, s_2) = -\Psi_{\text{total}}(\mathbf{r}_2, s_2; \mathbf{r}_1, s_1)$$

Since the spatial part $\psi_{1s}(\mathbf{r}_1)\psi_{1s}(\mathbf{r}_2)$ is **symmetric**:

$$\Psi_{\text{spatial}}(\mathbf{r}_1, \mathbf{r}_2) = \Psi_{\text{spatial}}(\mathbf{r}_2, \mathbf{r}_1)$$

The spin part must be **antisymmetric** (singlet):

$$\chi_{\text{singlet}} = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$$

### Complete Ground State

$$\boxed{\Psi_{\text{ground}} = \frac{Z^3}{\pi} e^{-Z(r_1 + r_2)} \cdot \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)}$$

Properties:
- Total spin: $S = 0$ (singlet)
- Total orbital angular momentum: $L = 0$ (both in s orbitals)
- Term symbol: $^1S_0$ (para-helium ground state)

### The Pauli Exclusion at Work

Q: Why can both electrons be in the 1s orbital?

A: They have different spin quantum numbers ($m_s = +1/2$ and $m_s = -1/2$). The Pauli principle forbids two electrons in the **same quantum state**, but spin provides an additional quantum number.

---

## 5. Electron-Electron Repulsion Analysis

### Why $1/r_{12}$ is Problematic

The electron-electron repulsion term:

$$\hat{V}_{ee} = \frac{1}{r_{12}} = \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}$$

This term:
1. **Couples** the two electrons
2. **Destroys** separability of coordinates
3. **Prevents** exact analytical solution

### Geometric Interpretation

Using the law of cosines:

$$r_{12}^2 = r_1^2 + r_2^2 - 2r_1 r_2 \cos\theta_{12}$$

where $\theta_{12}$ is the angle between $\mathbf{r}_1$ and $\mathbf{r}_2$.

This angle dependence means no single-particle coordinate system works.

### Multipole Expansion

For $r_1 < r_2$:

$$\frac{1}{r_{12}} = \sum_{l=0}^{\infty} \frac{r_1^l}{r_2^{l+1}} P_l(\cos\theta_{12})$$

This is useful for:
- Perturbation theory calculations
- Understanding angular correlations
- Deriving selection rules

### Order of Magnitude

Average electron-electron separation: $\langle r_{12} \rangle \sim a_0$

Average electron-nucleus separation: $\langle r \rangle \sim a_0/Z$

Repulsion energy scale: $\langle 1/r_{12} \rangle \sim 1/a_0 \sim 27$ eV

This is **comparable** to the binding energy—not a small perturbation!

---

## 6. Worked Examples

### Example 1: Energy in Atomic Units

**Problem:** Express the zeroth-order helium ground state energy in Hartree, eV, and kJ/mol.

**Solution:**

In atomic units:
$$E_0^{(0)} = -Z^2 = -4 \text{ Hartree}$$

Converting to eV:
$$E_0^{(0)} = -4 \times 27.211 \text{ eV} = -108.8 \text{ eV}$$

Converting to kJ/mol:
$$E_0^{(0)} = -4 \times 2625.5 \text{ kJ/mol} = -10502 \text{ kJ/mol}$$

$$\boxed{E_0^{(0)} = -4 \text{ Ha} = -108.8 \text{ eV} = -10502 \text{ kJ/mol}}$$

### Example 2: Probability Density at Nucleus

**Problem:** Calculate the probability density $|\Psi|^2$ for finding electron 1 at the nucleus (r₁ = 0) in the independent particle ground state.

**Solution:**

The spatial wave function (atomic units):
$$\Psi(\mathbf{r}_1, \mathbf{r}_2) = \frac{Z^3}{\pi} e^{-Z(r_1 + r_2)}$$

At $r_1 = 0$:
$$|\Psi(0, \mathbf{r}_2)|^2 = \frac{Z^6}{\pi^2} e^{-2Zr_2}$$

The probability density at the nucleus is:
$$|\psi_{1s}(0)|^2 = \frac{Z^3}{\pi}$$

For helium (Z = 2):
$$|\psi_{1s}(0)|^2 = \frac{8}{\pi} \approx 2.55 \text{ a.u.}^{-3}$$

$$\boxed{|\psi_{1s}(0)|^2 = \frac{Z^3}{\pi} = 2.55 \text{ a}_0^{-3}}$$

### Example 3: Average Electron-Nucleus Distance

**Problem:** Calculate $\langle r \rangle$ for an electron in the helium ground state (zeroth order).

**Solution:**

For a hydrogen-like 1s orbital with nuclear charge Z:

$$\langle r \rangle = \int_0^\infty r |\psi_{1s}(r)|^2 4\pi r^2 dr$$

Using $\psi_{1s} = \sqrt{Z^3/\pi} e^{-Zr}$:

$$\langle r \rangle = 4\pi \cdot \frac{Z^3}{\pi} \int_0^\infty r^3 e^{-2Zr} dr = 4Z^3 \cdot \frac{3!}{(2Z)^4} = \frac{3}{2Z}$$

For helium (Z = 2):
$$\langle r \rangle = \frac{3}{4} a_0 = 0.75 a_0 \approx 0.40 \text{ Å}$$

$$\boxed{\langle r \rangle = \frac{3}{2Z} a_0 = 0.75 a_0}$$

---

## 7. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Write the Hamiltonian for the Li⁺ ion (lithium with one electron removed). What is its ground state energy?

**Problem 1.2:** For the zeroth-order helium wave function, calculate the normalization constant and verify that $\langle \Psi | \Psi \rangle = 1$.

**Problem 1.3:** What is the ionization energy of He⁺ (helium with one electron)? Compare with the experimental first ionization energy of helium (24.6 eV).

### Level 2: Intermediate

**Problem 2.1:** Show that the ground state spatial wave function $\psi_{1s}(r_1)\psi_{1s}(r_2)$ is symmetric under exchange $r_1 \leftrightarrow r_2$. Why does this require an antisymmetric spin state?

**Problem 2.2:** Calculate $\langle r^2 \rangle$ for the zeroth-order helium ground state and hence find the uncertainty $\Delta r = \sqrt{\langle r^2 \rangle - \langle r \rangle^2}$.

**Problem 2.3:** Estimate the kinetic energy $\langle \hat{T} \rangle$ in the zeroth-order ground state using the virial theorem for Coulomb potentials: $\langle T \rangle = -E_{\text{total}}$.

### Level 3: Challenging

**Problem 3.1:** Show that in the independent particle approximation, the probability of finding both electrons on the same side of the nucleus is exactly 1/2. (Hint: The angular distributions are independent.)

**Problem 3.2:** Using the multipole expansion of $1/r_{12}$, show that the leading term is:
$$\frac{1}{r_{12}} = \frac{1}{r_>} + \frac{r_<}{r_>^2}P_1(\cos\theta_{12}) + \ldots$$
where $r_> = \max(r_1, r_2)$ and $r_< = \min(r_1, r_2)$.

**Problem 3.3:** Explain physically why the experimental helium ground state energy (-79.0 eV) is much higher than the zeroth-order estimate (-108.8 eV), but lower than what you'd expect from simple addition of repulsion energy.

---

## 8. Computational Lab: Helium Atom Visualization

```python
"""
Day 491 Computational Lab: Helium Atom Setup
Visualization and analysis of the helium atom in the independent particle approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

# Physical constants in atomic units
# In atomic units: hbar = m_e = e = 4*pi*eps_0 = 1
# Energy unit: Hartree = 27.211 eV
# Length unit: Bohr radius a_0 = 0.529 Angstrom

HARTREE_TO_EV = 27.211

def hydrogen_like_1s(r, Z):
    """
    Hydrogen-like 1s orbital wave function.

    Parameters:
    -----------
    r : float or array
        Radial distance in atomic units (a_0)
    Z : float
        Nuclear charge

    Returns:
    --------
    psi : float or array
        Wave function value(s)
    """
    normalization = np.sqrt(Z**3 / np.pi)
    return normalization * np.exp(-Z * r)

def radial_probability_density(r, Z):
    """
    Radial probability density P(r) = 4*pi*r^2 |psi|^2.
    """
    psi = hydrogen_like_1s(r, Z)
    return 4 * np.pi * r**2 * np.abs(psi)**2

def helium_ground_state_spatial(r1, r2, Z=2):
    """
    Spatial part of helium ground state wave function (zeroth order).
    Psi(r1, r2) = psi_1s(r1) * psi_1s(r2)
    """
    psi1 = hydrogen_like_1s(r1, Z)
    psi2 = hydrogen_like_1s(r2, Z)
    return psi1 * psi2

def calculate_hydrogen_like_energy(Z, n=1):
    """
    Energy of hydrogen-like atom in atomic units.
    E_n = -Z^2 / (2n^2) Hartree
    """
    return -Z**2 / (2 * n**2)

def calculate_helium_zeroth_order_energy(Z=2):
    """
    Zeroth-order ground state energy of helium.
    Both electrons in 1s orbital, no e-e repulsion.
    """
    E_1s = calculate_hydrogen_like_energy(Z, n=1)
    return 2 * E_1s  # Two electrons

def plot_radial_wave_functions():
    """Compare radial wave functions for different Z."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    r = np.linspace(0.01, 5, 500)

    # Wave functions
    ax1 = axes[0]
    for Z in [1, 2, 3]:
        psi = hydrogen_like_1s(r, Z)
        label = f'Z = {Z}' + (' (H)' if Z == 1 else ' (He)' if Z == 2 else ' (Li⁺)')
        ax1.plot(r, psi, label=label, linewidth=2)

    ax1.set_xlabel('r (a₀)', fontsize=12)
    ax1.set_ylabel('ψ₁ₛ(r)', fontsize=12)
    ax1.set_title('1s Wave Functions for Different Nuclear Charges', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Radial probability densities
    ax2 = axes[1]
    for Z in [1, 2, 3]:
        P_r = radial_probability_density(r, Z)
        label = f'Z = {Z}'
        ax2.plot(r, P_r, label=label, linewidth=2)

        # Mark the maximum (most probable radius)
        r_max = r[np.argmax(P_r)]
        ax2.axvline(x=r_max, color=ax2.lines[-1].get_color(),
                   linestyle='--', alpha=0.5)

    ax2.set_xlabel('r (a₀)', fontsize=12)
    ax2.set_ylabel('P(r) = 4πr²|ψ|²', fontsize=12)
    ax2.set_title('Radial Probability Density', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('helium_wave_functions.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_helium_density_2d():
    """
    2D contour plot of helium electron density.
    Shows density in (r1, r2) space at fixed angles.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    r = np.linspace(0, 3, 100)
    R1, R2 = np.meshgrid(r, r)

    # Probability density |Psi|^2
    Z = 2
    Psi_sq = helium_ground_state_spatial(R1, R2, Z)**2

    # Plot 1: Probability density
    ax1 = axes[0]
    contour = ax1.contourf(R1, R2, Psi_sq, levels=30, cmap='viridis')
    ax1.set_xlabel('r₁ (a₀)', fontsize=12)
    ax1.set_ylabel('r₂ (a₀)', fontsize=12)
    ax1.set_title('|Ψ(r₁, r₂)|² for Helium Ground State\n(Zeroth-Order Approximation)', fontsize=12)
    ax1.set_aspect('equal')
    plt.colorbar(contour, ax=ax1, label='Probability Density')

    # Add diagonal line showing r1 = r2
    ax1.plot([0, 3], [0, 3], 'w--', linewidth=2, label='r₁ = r₂')
    ax1.legend()

    # Plot 2: Radial probability for each electron
    ax2 = axes[1]

    # Integrate over r2 to get marginal for r1
    P_r1 = np.zeros_like(r)
    for i, r1_val in enumerate(r):
        integrand = lambda r2: helium_ground_state_spatial(r1_val, r2, Z)**2 * 4*np.pi*r2**2
        P_r1[i], _ = integrate.quad(integrand, 0, 10)
    P_r1 *= 4 * np.pi * r**2

    # Also plot the independent particle result
    P_single = radial_probability_density(r, Z)

    ax2.plot(r, P_r1, 'b-', linewidth=2, label='Marginal P(r₁)')
    ax2.plot(r, P_single, 'r--', linewidth=2, label='Single-particle 4πr²|ψ₁ₛ|²')
    ax2.set_xlabel('r (a₀)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Radial Probability Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('helium_density_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

def calculate_expectation_values(Z=2):
    """
    Calculate key expectation values for helium ground state.
    """

    print("=" * 60)
    print("HELIUM ATOM: EXPECTATION VALUES (Zeroth-Order)")
    print("=" * 60)
    print(f"\nNuclear charge Z = {Z}")

    # Ground state energy (zeroth order)
    E_0 = calculate_helium_zeroth_order_energy(Z)
    print(f"\nZeroth-order energy:")
    print(f"  E₀⁽⁰⁾ = {E_0:.4f} Hartree = {E_0 * HARTREE_TO_EV:.2f} eV")

    # Experimental value
    E_exp = -2.9037  # Hartree
    print(f"\nExperimental energy:")
    print(f"  E_exp = {E_exp:.4f} Hartree = {E_exp * HARTREE_TO_EV:.2f} eV")
    print(f"\nError: {(E_0 - E_exp):.4f} Hartree = {(E_0 - E_exp) * HARTREE_TO_EV:.2f} eV")
    print(f"Relative error: {100 * (E_0 - E_exp) / abs(E_exp):.1f}%")

    # Average distance from nucleus
    r_avg = 3 / (2 * Z)  # <r> for 1s orbital
    print(f"\nAverage electron-nucleus distance:")
    print(f"  <r> = {r_avg:.4f} a₀ = {r_avg * 0.529:.4f} Å")

    # Most probable distance
    r_mp = 1 / Z  # Maximum of radial probability
    print(f"\nMost probable distance:")
    print(f"  r_mp = {r_mp:.4f} a₀ = {r_mp * 0.529:.4f} Å")

    # <r²> and uncertainty
    r_sq_avg = 3 / (Z**2)  # <r²> for 1s orbital
    delta_r = np.sqrt(r_sq_avg - r_avg**2)
    print(f"\nPosition uncertainty:")
    print(f"  <r²> = {r_sq_avg:.4f} a₀²")
    print(f"  Δr = {delta_r:.4f} a₀")

    # Kinetic and potential energies (virial theorem)
    T_avg = -E_0  # For Coulomb, <T> = -E
    V_avg = 2 * E_0  # For Coulomb, <V> = 2E
    print(f"\nVirial theorem (zeroth order):")
    print(f"  <T> = {T_avg:.4f} Hartree = {T_avg * HARTREE_TO_EV:.2f} eV")
    print(f"  <V_en> = {V_avg:.4f} Hartree = {V_avg * HARTREE_TO_EV:.2f} eV")

    # Missing repulsion energy (order of magnitude)
    V_ee_estimate = 5/4 * Z  # First-order perturbation result (preview)
    print(f"\nElectron-electron repulsion (to be calculated):")
    print(f"  <1/r₁₂> ~ {V_ee_estimate:.2f} Hartree (first-order)")

def compare_with_experiment():
    """
    Compare zeroth-order approximation with experimental data.
    """

    print("\n" + "=" * 60)
    print("COMPARISON WITH EXPERIMENT")
    print("=" * 60)

    # Data
    data = {
        'System': ['H', 'He (0th order)', 'He (experiment)', 'Li⁺ (experiment)'],
        'Z': [1, 2, 2, 3],
        'E_theory': [-0.5, -4.0, None, -4.5],
        'E_exp': [-0.5, -2.9037, -2.9037, -7.28]
    }

    print("\nGround State Energies (Hartree):")
    print("-" * 50)
    print(f"{'System':<20} {'Theory':<12} {'Experiment':<12}")
    print("-" * 50)

    for i in range(len(data['System'])):
        theory = data['E_theory'][i]
        exp = data['E_exp'][i]
        theory_str = f"{theory:.4f}" if theory is not None else "---"
        exp_str = f"{exp:.4f}" if exp is not None else "---"
        print(f"{data['System'][i]:<20} {theory_str:<12} {exp_str:<12}")

    print("\n" + "-" * 50)
    print("\nKey Insight: The ~30 eV error for helium shows that")
    print("electron-electron repulsion is NOT a small perturbation!")
    print("We need better approximation methods.")

def quantum_computing_preview():
    """
    Preview of helium in quantum computing context.
    """

    print("\n" + "=" * 60)
    print("QUANTUM COMPUTING CONNECTION")
    print("=" * 60)

    print("""
    THE HELIUM ATOM AS A VQE BENCHMARK
    ==================================

    Helium is a prime target for Variational Quantum Eigensolver (VQE):

    1. MINIMAL BASIS ENCODING
       - 2 spatial orbitals (1s, 2s) × 2 spins = 4 spin-orbitals
       - Maps to 4 qubits via Jordan-Wigner
       - Small enough for near-term devices

    2. HAMILTONIAN STRUCTURE
       H = h_pq c†_p c_q + h_pqrs c†_p c†_q c_r c_s

       One-body terms: kinetic + electron-nucleus
       Two-body terms: electron-electron repulsion

    3. WHY VQE?
       - Ground state energy is what we want
       - Variational principle guarantees E_trial ≥ E_exact
       - Quantum computers can prepare complex trial states
       - Classical optimization finds best parameters

    4. CURRENT STATUS (2024-2025)
       - Multiple VQE implementations of helium
       - Chemical accuracy (~1 mHa) achieved with error mitigation
       - Stepping stone to larger molecules

    Tomorrow: First-order perturbation theory for <1/r₁₂>
    Later this week: Variational method mirrors VQE philosophy!
    """)

def main():
    """Run all demonstrations."""

    print("Day 491: Helium Atom Setup")
    print("=" * 60)

    # Calculate and display key values
    calculate_expectation_values()
    compare_with_experiment()

    # Generate visualizations
    plot_radial_wave_functions()
    plot_helium_density_2d()

    # Quantum computing connection
    quantum_computing_preview()

if __name__ == "__main__":
    main()
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Helium Hamiltonian | Two-electron system with kinetic, electron-nucleus, and electron-electron terms |
| Independent particle approx. | Ignore $1/r_{12}$; treat as two non-interacting H-like atoms |
| Zeroth-order energy | $E_0^{(0)} = -Z^2 = -4$ Hartree = -108.8 eV |
| Experimental energy | -79.0 eV (38% error in zeroth order!) |
| Ground state | Both electrons in 1s, singlet spin state ($^1S_0$) |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$\hat{H} = -\frac{1}{2}\nabla_1^2 - \frac{1}{2}\nabla_2^2 - \frac{Z}{r_1} - \frac{Z}{r_2} + \frac{1}{r_{12}}$$ | Helium Hamiltonian (atomic units) |
| $$E_0^{(0)} = -Z^2$$ | Zeroth-order ground state energy |
| $$\Psi_0 = \frac{Z^3}{\pi}e^{-Z(r_1+r_2)}$$ | Spatial wave function |
| $$\langle r \rangle = \frac{3}{2Z}$$ | Average electron-nucleus distance |

---

## 10. Daily Checklist

### Conceptual Understanding
- [ ] I can explain why helium cannot be solved exactly
- [ ] I understand each term in the helium Hamiltonian
- [ ] I know why the independent particle approximation fails so badly
- [ ] I can describe the symmetry requirements for the ground state

### Mathematical Skills
- [ ] I can write the Hamiltonian in atomic units
- [ ] I can calculate hydrogen-like energies for any Z
- [ ] I can evaluate expectation values for 1s orbitals
- [ ] I can apply the virial theorem

### Computational Skills
- [ ] I implemented helium wave function visualization
- [ ] I compared theoretical and experimental energies
- [ ] I calculated radial probability distributions

### Quantum Computing Connection
- [ ] I understand why helium is a VQE benchmark
- [ ] I know the qubit requirements for minimal basis helium
- [ ] I see how variational methods connect to VQE

---

## 11. Preview: Day 492

Tomorrow we attack the electron-electron repulsion using **first-order perturbation theory**:

- Treating $\hat{V}_{ee} = 1/r_{12}$ as a perturbation
- Calculating $E^{(1)} = \langle \Psi_0 | 1/r_{12} | \Psi_0 \rangle$
- Evaluating the six-dimensional integral
- The result: $E^{(1)} = \frac{5}{4}Z$ Hartree = +34 eV
- Comparison with experiment: much better but still ~6% error

---

## References

1. Griffiths, D.J. & Schroeter, D.F. (2018). *Introduction to Quantum Mechanics*, 3rd ed., Ch. 7.2.

2. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 8.

3. Szabo, A. & Ostlund, N.S. (1996). *Modern Quantum Chemistry*. Dover, Ch. 2.

4. NIST Atomic Spectra Database: https://physics.nist.gov/PhysRefData/ASD/

---

*"The helium atom, with its two electrons, was one of the first serious tests of quantum mechanics. Its accurate solution required not just the right theory, but also the development of powerful approximation methods that remain essential tools today."*
— Attila Szabo

---

**Day 491 Complete.** Tomorrow: First-Order Perturbation Theory for Helium.
