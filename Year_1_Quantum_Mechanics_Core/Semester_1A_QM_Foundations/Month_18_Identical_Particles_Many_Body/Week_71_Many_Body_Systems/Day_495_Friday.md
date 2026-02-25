# Day 495: Multi-Electron Atoms

## Overview

**Day 495 of 2520 | Week 71, Day 5 | Month 18: Identical Particles & Many-Body Physics**

Today we extend our understanding from two-electron helium to the rich physics of multi-electron atoms. Using the central field approximation and the self-consistent field concept, we will understand how electron configurations arise, why Hund's rules work, and how the periodic table emerges from quantum mechanics. This day bridges atomic physics with chemistry and materials science.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | The Many-Electron Problem | 45 min |
| 9:45 AM | Central Field Approximation | 75 min |
| 11:00 AM | Break | 15 min |
| 11:15 AM | Aufbau Principle and Configurations | 90 min |
| 12:45 PM | Lunch | 60 min |
| 1:45 PM | Hund's Rules | 75 min |
| 3:00 PM | Break | 15 min |
| 3:15 PM | Periodic Table Structure | 60 min |
| 4:15 PM | Computational Lab | 75 min |
| 5:30 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** the central field approximation and its validity
2. **Apply** the aufbau principle to determine electron configurations
3. **Use** Hund's rules to predict ground state term symbols
4. **Understand** the structure of the periodic table from quantum mechanics
5. **Calculate** approximate ionization energies using screening constants
6. **Connect** atomic structure to chemical and physical properties

---

## 1. The Many-Electron Problem

### The Full Hamiltonian

For an atom with $N$ electrons and nuclear charge $Z$:

$$\hat{H} = \sum_{i=1}^{N} \left[-\frac{\hbar^2}{2m}\nabla_i^2 - \frac{Ze^2}{r_i}\right] + \sum_{i<j}^{N} \frac{e^2}{r_{ij}}$$

In atomic units:

$$\boxed{\hat{H} = \sum_{i=1}^{N} \left[-\frac{1}{2}\nabla_i^2 - \frac{Z}{r_i}\right] + \sum_{i<j}^{N} \frac{1}{r_{ij}}}$$

### Why Exact Solution is Impossible

- **Helium (N=2):** 6-dimensional problem, no closed-form solution
- **Lithium (N=3):** 9-dimensional, electron-electron terms couple all three
- **Carbon (N=6):** 18-dimensional, 15 electron-electron terms
- **Uranium (N=92):** 276-dimensional, 4186 repulsion terms!

The many-body problem scales **exponentially** with electron number.

### Approximation Strategy

We need a systematic approach:
1. Replace complex electron-electron interaction with average potential
2. Solve resulting single-particle equations
3. Build up many-electron state from single-particle orbitals
4. Correct for residual interactions

---

## 2. Central Field Approximation

### The Key Insight

Replace the electron-electron repulsion with a **spherically symmetric average potential**:

$$\sum_{j \neq i} \frac{1}{r_{ij}} \approx V_{\text{eff}}(r_i)$$

The total potential seen by electron $i$:

$$V_{\text{central}}(r_i) = -\frac{Z}{r_i} + V_{\text{screen}}(r_i)$$

### Physical Picture

- **Near nucleus ($r \to 0$):** Full nuclear charge $Z$ felt
- **Far from nucleus ($r \to \infty$):** Only $(Z - N + 1)$ effective charge (other electrons screen)
- **Intermediate:** Smooth interpolation

### Effective Nuclear Charge

$$\boxed{Z_{\text{eff}}(r) = Z - \sigma(r)}$$

where $\sigma(r)$ is the **screening function** that increases from 0 to approximately $N-1$.

### Separability Restored!

With a central potential, the Schrödinger equation separates:

$$\hat{H}_{\text{central}} = \sum_i \hat{h}_i$$

where:
$$\hat{h}_i = -\frac{1}{2}\nabla_i^2 + V_{\text{central}}(r_i)$$

**Solutions:** Single-particle orbitals $\psi_{nlm}(r, \theta, \phi)$

### Quantum Numbers Preserved

The spherical symmetry means:
- $l$ and $m_l$ are still good quantum numbers
- Orbitals characterized by $(n, l, m_l, m_s)$
- But **energy depends on both $n$ and $l$** (unlike hydrogen!)

---

## 3. Single-Particle Orbitals

### Notation

| $l$ | Letter | Shape |
|-----|--------|-------|
| 0 | s | Spherical |
| 1 | p | Dumbbell |
| 2 | d | Cloverleaf |
| 3 | f | Complex |

### Energy Ordering

For hydrogen: $E_n = -1/(2n^2)$, depends only on $n$

For multi-electron atoms: **$l$-degeneracy is broken!**

$$E_{ns} < E_{np} < E_{nd} < E_{nf}$$

Why? Higher $l$ orbitals have probability farther from nucleus → see more screening → higher energy.

### The $(n + l)$ Rule (Madelung Rule)

Orbitals fill approximately in order of increasing $(n + l)$:

$$1s < 2s < 2p < 3s < 3p < 4s < 3d < 4p < 5s < 4d < ...$$

For same $(n + l)$: lower $n$ fills first.

### Exceptions

The $(n+l)$ rule has exceptions:
- Chromium: $[Ar]3d^5 4s^1$ instead of $[Ar]3d^4 4s^2$
- Copper: $[Ar]3d^{10} 4s^1$ instead of $[Ar]3d^9 4s^2$

Half-filled and filled subshells have extra stability!

---

## 4. The Aufbau Principle

### Statement

**Aufbau** (German for "building up"): Electrons fill orbitals starting from the lowest available energy level, respecting the Pauli exclusion principle.

### Building Electron Configurations

**Hydrogen (Z=1):** $1s^1$
**Helium (Z=2):** $1s^2$
**Lithium (Z=3):** $1s^2 2s^1$ = $[He] 2s^1$
**Beryllium (Z=4):** $[He] 2s^2$
**Boron (Z=5):** $[He] 2s^2 2p^1$
**Carbon (Z=6):** $[He] 2s^2 2p^2$
...
**Neon (Z=10):** $[He] 2s^2 2p^6$ = $[Ne]$

### Capacity of Each Subshell

For a subshell with quantum number $l$:
- $m_l = -l, -l+1, ..., l-1, l$ → $(2l + 1)$ orbitals
- Each orbital holds 2 electrons (spin up and down)
- **Maximum capacity: $2(2l + 1)$**

| Subshell | $l$ | Orbitals | Max electrons |
|----------|-----|----------|---------------|
| s | 0 | 1 | 2 |
| p | 1 | 3 | 6 |
| d | 2 | 5 | 10 |
| f | 3 | 7 | 14 |

### The Periodic Table Emerges

- **Period 1:** Fill 1s (2 elements: H, He)
- **Period 2:** Fill 2s, 2p (8 elements: Li through Ne)
- **Period 3:** Fill 3s, 3p (8 elements: Na through Ar)
- **Period 4:** Fill 4s, 3d, 4p (18 elements: K through Kr)
- **Period 5:** Fill 5s, 4d, 5p (18 elements: Rb through Xe)
- **Period 6:** Fill 6s, 4f, 5d, 6p (32 elements: Cs through Rn)

---

## 5. Hund's Rules

### The Three Rules

For a partially filled subshell, the ground state term has:

**Rule 1 (Maximum Spin):**
$$\boxed{S = \text{maximum consistent with Pauli}}$$

**Rule 2 (Maximum Orbital):**
$$\boxed{L = \text{maximum consistent with Rule 1 and Pauli}}$$

**Rule 3 (Spin-Orbit):**
$$\boxed{J = |L - S| \text{ for less than half-filled}, \quad J = L + S \text{ for more than half-filled}}$$

### Physical Basis of Hund's Rules

**Rule 1:** Maximum spin means parallel spins → antisymmetric spatial → Fermi hole → reduced repulsion → lower energy.

**Rule 2:** Maximum $L$ means electrons preferentially occupy high-$|m_l|$ orbitals → reduces overlap → lower repulsion.

**Rule 3:** Spin-orbit coupling $\sim \mathbf{L} \cdot \mathbf{S}$ changes sign with shell filling.

### Example: Carbon ($2p^2$)

Two electrons in $2p$ subshell:

**Rule 1:** Maximum $S$: both spins parallel → $S = 1$

**Rule 2:** With parallel spins, maximum $L$: $m_{l,1} = 1$, $m_{l,2} = 0$ (can't both be 1 due to Pauli) → $M_L = 1 \Rightarrow L = 1$

**Rule 3:** Less than half-filled (2 of 6) → $J = |L - S| = |1 - 1| = 0$

Ground state term: $\boxed{^3P_0}$

### Term Symbol Notation

$$^{2S+1}L_J$$

- $2S + 1$: Spin multiplicity (1 = singlet, 2 = doublet, 3 = triplet, ...)
- $L$: Total orbital angular momentum (S, P, D, F, ...)
- $J$: Total angular momentum

---

## 6. Slater's Rules for Screening

### Empirical Screening Constants

Slater (1930) gave simple rules to estimate $Z_{\text{eff}}$:

$$Z_{\text{eff}} = Z - \sigma$$

where $\sigma$ is calculated by:

1. Group orbitals: (1s), (2s,2p), (3s,3p), (3d), (4s,4p), (4d), (4f), ...

2. For each electron, contributions to $\sigma$:
   - Same group: 0.35 (except 1s: 0.30)
   - One group lower (n-1): 0.85 for s,p; 1.00 for d,f
   - Two or more groups lower: 1.00

### Example: Oxygen (Z = 8, $1s^2 2s^2 2p^4$)

For a 2p electron:
- Other 2p and 2s electrons (5 total): $5 \times 0.35 = 1.75$
- 1s electrons (2): $2 \times 0.85 = 1.70$
- Total $\sigma = 3.45$

$$Z_{\text{eff}} = 8 - 3.45 = 4.55$$

### Ionization Energy Estimate

$$E_n \approx -\frac{Z_{\text{eff}}^2}{2n^2} \text{ Hartree}$$

First ionization energy of oxygen:
$$IE \approx \frac{(4.55)^2}{2 \times 2^2} \times 27.2 \text{ eV} \approx 14.0 \text{ eV}$$

Experimental: 13.6 eV (quite good!)

---

## 7. Periodic Trends

### Atomic Radius

**Across a period (→):** Decreases
- $Z$ increases, $Z_{\text{eff}}$ increases
- Stronger attraction pulls electrons closer

**Down a group (↓):** Increases
- New shell added
- Electrons farther from nucleus

### Ionization Energy

**Across a period (→):** Generally increases (with dips)
- $Z_{\text{eff}}$ increases → harder to remove electron
- Dips at half-filled and filled subshells (special stability)

**Down a group (↓):** Decreases
- Valence electrons farther from nucleus
- More shielding by inner shells

### Electronegativity

Similar trend to ionization energy:
- Increases across period
- Decreases down group
- Maximum: Fluorine (most electronegative)

### Magnetic Properties

Determined by unpaired electrons:
- Filled subshells: Diamagnetic (no net moment)
- Unfilled subshells: Paramagnetic (net moment)

---

## 8. Worked Examples

### Example 1: Electron Configuration of Iron

**Problem:** Write the electron configuration and ground state term for Fe (Z = 26).

**Solution:**

Following aufbau:
$$\text{Fe}: [Ar] 3d^6 4s^2$$

Or: $1s^2 2s^2 2p^6 3s^2 3p^6 3d^6 4s^2$

For the $3d^6$ subshell (4s is filled, doesn't contribute to term):

**Hund's Rule 1:** Maximum $S$
- 5 electrons: all spin up ($m_l = 2,1,0,-1,-2$)
- 6th electron: must pair → $m_l = 2$, spin down
- Total $S = 4 \times \frac{1}{2} = 2$

**Hund's Rule 2:** Maximum $L$
- $M_L = 2+1+0+(-1)+(-2)+2 = 2$
- So $L = 2$ (D term)

**Hund's Rule 3:** More than half-filled (6 > 5)
- $J = L + S = 2 + 2 = 4$

$$\boxed{\text{Fe}: [Ar]3d^6 4s^2, \quad ^5D_4}$$

### Example 2: Ionization Energy Comparison

**Problem:** Which has higher first ionization energy: N or O? Explain using Hund's rules.

**Solution:**

Nitrogen: $[He]2s^2 2p^3$ - exactly half-filled p subshell
Oxygen: $[He]2s^2 2p^4$ - one electron past half-filled

For N: All three 2p electrons have parallel spins (Hund's Rule 1)
→ Maximum exchange stabilization
→ Extra stability from half-filled subshell

For O: Fourth electron must pair in one orbital
→ Additional electron-electron repulsion
→ Easier to remove this paired electron

**Result:** $IE(\text{N}) = 14.5$ eV $>$ $IE(\text{O}) = 13.6$ eV

$$\boxed{\text{Nitrogen has higher ionization energy due to half-filled subshell stability}}$$

### Example 3: Magnetic Moment

**Problem:** Calculate the spin-only magnetic moment of Mn²⁺.

**Solution:**

Mn: $[Ar]3d^5 4s^2$
Mn²⁺: $[Ar]3d^5$ (lost 4s electrons first)

5 unpaired electrons (all parallel by Hund's Rule 1)

Spin-only magnetic moment:
$$\mu_s = \sqrt{n(n+2)} \, \mu_B = \sqrt{5 \times 7} \, \mu_B = \sqrt{35} \, \mu_B \approx 5.92 \, \mu_B$$

$$\boxed{\mu_s(\text{Mn}^{2+}) = 5.92 \, \mu_B}$$

---

## 9. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Write electron configurations for: (a) Si (Z=14), (b) Cl (Z=17), (c) Ca (Z=20).

**Problem 1.2:** Determine the ground state term symbol for nitrogen (N, Z=7).

**Problem 1.3:** Using Slater's rules, calculate $Z_{\text{eff}}$ for a 3p electron in argon (Z=18).

### Level 2: Intermediate

**Problem 2.1:** Explain the exception in chromium's configuration: $[Ar]3d^5 4s^1$ instead of $[Ar]3d^4 4s^2$.

**Problem 2.2:** Rank the following in order of increasing first ionization energy: Na, Mg, Al, Si. Explain any anomalies.

**Problem 2.3:** Determine the ground state term symbols for all elements from B (Z=5) to Ne (Z=10).

### Level 3: Challenging

**Problem 3.1:** Show that for a $d^n$ configuration, the number of microstates is $\binom{10}{n}$. Calculate this for Fe²⁺ ($d^6$).

**Problem 3.2:** Using the central field approximation, explain why the $3d$ orbitals are higher in energy than $4s$ for free atoms but lower for ions.

**Problem 3.3:** Derive Slater's rules qualitatively from the radial wave functions.

---

## 10. Computational Lab: Multi-Electron Atoms

```python
"""
Day 495 Computational Lab: Multi-Electron Atoms
Electron configurations, Slater's rules, and periodic trends.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Constants
HARTREE_TO_EV = 27.211

# Element data
ELEMENTS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
    26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
}

# Orbital order for filling (Madelung rule)
ORBITAL_ORDER = [
    '1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p',
    '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7p'
]

# Maximum electrons per subshell
def max_electrons(orbital):
    """Return maximum electrons for a subshell."""
    l = {'s': 0, 'p': 1, 'd': 2, 'f': 3}[orbital[1]]
    return 2 * (2 * l + 1)

def get_electron_configuration(Z):
    """
    Determine electron configuration using aufbau principle.
    Returns dict: {orbital: n_electrons}
    """
    config = {}
    electrons_remaining = Z

    for orbital in ORBITAL_ORDER:
        if electrons_remaining <= 0:
            break
        max_e = max_electrons(orbital)
        n_e = min(electrons_remaining, max_e)
        if n_e > 0:
            config[orbital] = n_e
        electrons_remaining -= n_e

    return config

def config_to_string(config):
    """Convert configuration dict to string."""
    return ' '.join(f"{orb}^{n}" if n > 1 else orb for orb, n in config.items())

def slater_screening(Z, orbital):
    """
    Calculate Slater screening constant for an electron in given orbital.

    Parameters:
    -----------
    Z : int
        Atomic number
    orbital : str
        Orbital like '2p', '3d', etc.

    Returns:
    --------
    sigma : float
        Screening constant
    """
    config = get_electron_configuration(Z)
    n = int(orbital[0])
    l_char = orbital[1]

    # Define groups
    if l_char in ['s', 'p']:
        groups = {
            1: ['1s'],
            2: ['2s', '2p'],
            3: ['3s', '3p'],
            4: ['4s', '4p'],
            5: ['5s', '5p'],
        }
    else:  # d or f
        groups = {
            1: ['1s'],
            2: ['2s', '2p'],
            3: ['3s', '3p', '3d'],
            4: ['4s', '4p', '4d'],
            5: ['5s', '5p', '5d'],
        }

    sigma = 0

    for orb, n_e in config.items():
        if orb == orbital:
            # Same orbital: count others
            sigma += (n_e - 1) * (0.30 if orb == '1s' else 0.35)
        else:
            orb_n = int(orb[0])
            orb_l = orb[1]

            # Determine contribution
            if orb_n == n and orb_l in ['s', 'p'] and l_char in ['s', 'p']:
                # Same n, same s/p group
                sigma += n_e * 0.35
            elif orb_n == n - 1:
                if l_char in ['s', 'p']:
                    sigma += n_e * 0.85
                else:
                    sigma += n_e * 1.00
            elif orb_n < n - 1:
                sigma += n_e * 1.00
            elif orb_n == n and l_char in ['d', 'f'] and orb_l in ['s', 'p']:
                # d/f electrons see s/p of same n differently
                sigma += n_e * 1.00

    return sigma

def Z_effective(Z, orbital):
    """Calculate effective nuclear charge."""
    sigma = slater_screening(Z, orbital)
    return Z - sigma

def estimate_ionization_energy(Z):
    """
    Estimate first ionization energy using Slater's rules.
    """
    config = get_electron_configuration(Z)
    # Find outermost orbital
    outermost = list(config.keys())[-1]
    n = int(outermost[0])
    Z_eff = Z_effective(Z, outermost)

    # Energy of electron in this orbital
    E = -Z_eff**2 / (2 * n**2)

    # Ionization energy (positive)
    return -E * HARTREE_TO_EV

def plot_periodic_trends():
    """Plot ionization energies across the periodic table."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Experimental ionization energies (eV)
    IE_exp = {
        1: 13.6, 2: 24.6, 3: 5.4, 4: 9.3, 5: 8.3, 6: 11.3, 7: 14.5, 8: 13.6, 9: 17.4, 10: 21.6,
        11: 5.1, 12: 7.6, 13: 6.0, 14: 8.2, 15: 10.5, 16: 10.4, 17: 13.0, 18: 15.8,
        19: 4.3, 20: 6.1, 21: 6.5, 22: 6.8, 23: 6.7, 24: 6.8, 25: 7.4,
        26: 7.9, 27: 7.9, 28: 7.6, 29: 7.7, 30: 9.4,
        31: 6.0, 32: 7.9, 33: 9.8, 34: 9.8, 35: 11.8, 36: 14.0,
    }

    Z_values = list(range(1, 37))
    IE_calc = [estimate_ionization_energy(Z) for Z in Z_values]
    IE_expt = [IE_exp.get(Z, np.nan) for Z in Z_values]

    # Plot 1: IE vs Z
    ax1 = axes[0, 0]
    ax1.plot(Z_values, IE_calc, 'b-o', markersize=4, label='Slater estimate')
    ax1.plot(Z_values, IE_expt, 'r-s', markersize=4, label='Experimental')

    # Mark noble gases
    noble_gases = [2, 10, 18, 36]
    for ng in noble_gases:
        ax1.axvline(x=ng, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Atomic Number Z', fontsize=12)
    ax1.set_ylabel('First Ionization Energy (eV)', fontsize=12)
    ax1.set_title('First Ionization Energy Across Periodic Table', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Z_eff for valence electrons
    ax2 = axes[0, 1]

    Z_eff_values = []
    for Z in Z_values:
        config = get_electron_configuration(Z)
        outermost = list(config.keys())[-1]
        Z_eff_values.append(Z_effective(Z, outermost))

    ax2.plot(Z_values, Z_eff_values, 'g-o', markersize=4)
    ax2.plot(Z_values, Z_values, 'k--', alpha=0.5, label='Z (no screening)')
    ax2.set_xlabel('Atomic Number Z', fontsize=12)
    ax2.set_ylabel('Z_eff (valence electron)', fontsize=12)
    ax2.set_title('Effective Nuclear Charge', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Electron configuration visualization (first 18 elements)
    ax3 = axes[1, 0]

    subshells = ['1s', '2s', '2p', '3s', '3p']
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for Z in range(1, 19):
        config = get_electron_configuration(Z)
        y_offset = 0
        for sub, color in zip(subshells, colors):
            n_e = config.get(sub, 0)
            if n_e > 0:
                ax3.bar(Z, n_e, bottom=y_offset, color=color, edgecolor='black', width=0.8)
                y_offset += n_e

    ax3.set_xlabel('Atomic Number Z', fontsize=12)
    ax3.set_ylabel('Number of Electrons', fontsize=12)
    ax3.set_title('Electron Configuration (Z=1-18)', fontsize=12)
    ax3.set_xticks(range(1, 19))
    ax3.set_xticklabels([ELEMENTS[Z] for Z in range(1, 19)], rotation=45)

    # Legend for subshells
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor='black', label=s)
                       for s, c in zip(subshells, colors)]
    ax3.legend(handles=legend_elements, loc='upper left')

    # Plot 4: Transition metals (3d series)
    ax4 = axes[1, 1]

    tm_Z = list(range(21, 31))
    tm_names = [ELEMENTS[Z] for Z in tm_Z]
    tm_3d = []
    tm_4s = []

    for Z in tm_Z:
        config = get_electron_configuration(Z)
        tm_3d.append(config.get('3d', 0))
        tm_4s.append(config.get('4s', 0))

    x = np.arange(len(tm_Z))
    width = 0.35

    ax4.bar(x - width/2, tm_3d, width, label='3d', color='blue')
    ax4.bar(x + width/2, tm_4s, width, label='4s', color='red')

    ax4.set_xlabel('Element', fontsize=12)
    ax4.set_ylabel('Number of Electrons', fontsize=12)
    ax4.set_title('3d Transition Metals: 3d vs 4s Occupation', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(tm_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('periodic_trends.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_configurations():
    """Print electron configurations for first 36 elements."""

    print("=" * 70)
    print("ELECTRON CONFIGURATIONS (First 36 Elements)")
    print("=" * 70)

    print(f"\n{'Z':<4} {'Element':<4} {'Configuration':<30} {'Z_eff(val)':<10}")
    print("-" * 70)

    for Z in range(1, 37):
        config = get_electron_configuration(Z)
        config_str = config_to_string(config)
        outermost = list(config.keys())[-1]
        Z_eff = Z_effective(Z, outermost)
        print(f"{Z:<4} {ELEMENTS[Z]:<4} {config_str:<30} {Z_eff:<10.2f}")

def hundsrules_demo():
    """Demonstrate Hund's rules for ground state terms."""

    print("\n" + "=" * 70)
    print("HUND'S RULES: GROUND STATE TERMS")
    print("=" * 70)

    # p-block examples
    print("\np-block elements (2p subshell):")
    print("-" * 50)

    p_elements = [
        ('B', 5, '2p¹', 1, 1, 1/2, 1/2, '²P₁/₂'),
        ('C', 6, '2p²', 2, 1, 1, 0, '³P₀'),
        ('N', 7, '2p³', 3, 0, 3/2, 3/2, '⁴S₃/₂'),
        ('O', 8, '2p⁴', 4, 1, 1, 2, '³P₂'),
        ('F', 9, '2p⁵', 5, 1, 1/2, 3/2, '²P₃/₂'),
        ('Ne', 10, '2p⁶', 6, 0, 0, 0, '¹S₀'),
    ]

    print(f"{'Elem':<5} {'Config':<8} {'n_e':<5} {'L':<5} {'S':<5} {'J':<5} {'Term':<10}")
    for elem, Z, config, n, L, S, J, term in p_elements:
        print(f"{elem:<5} {config:<8} {n:<5} {L:<5} {S:<5} {J:<5} {term:<10}")

    # d-block examples
    print("\nd-block elements (3d subshell):")
    print("-" * 50)

    d_elements = [
        ('Ti²⁺', '3d²', 2, 3, 1, 2, '³F₂'),
        ('V²⁺', '3d³', 3, 3, 3/2, 3/2, '⁴F₃/₂'),
        ('Cr³⁺', '3d³', 3, 3, 3/2, 3/2, '⁴F₃/₂'),
        ('Mn²⁺', '3d⁵', 5, 0, 5/2, 5/2, '⁶S₅/₂'),
        ('Fe²⁺', '3d⁶', 6, 2, 2, 4, '⁵D₄'),
        ('Co²⁺', '3d⁷', 7, 3, 3/2, 9/2, '⁴F₉/₂'),
        ('Ni²⁺', '3d⁸', 8, 3, 1, 4, '³F₄'),
        ('Cu²⁺', '3d⁹', 9, 2, 1/2, 5/2, '²D₅/₂'),
    ]

    print(f"{'Ion':<8} {'Config':<8} {'n_e':<5} {'L':<5} {'S':<5} {'J':<5} {'Term':<10}")
    for ion, config, n, L, S, J, term in d_elements:
        print(f"{ion:<8} {config:<8} {n:<5} {L:<5} {S:<5} {J:<5} {term:<10}")

def quantum_computing_connection():
    """Connection to quantum computing."""

    print("\n" + "=" * 70)
    print("QUANTUM COMPUTING CONNECTION")
    print("=" * 70)

    print("""
    QUANTUM SIMULATION OF MULTI-ELECTRON ATOMS
    ==========================================

    1. THE CHALLENGE
       - N electrons → 3N spatial + N spin degrees of freedom
       - Classical simulation: exponential scaling
       - Quantum simulation: polynomial scaling!

    2. MAPPING TO QUBITS
       Second quantization basis:
       - Each spin-orbital → one qubit
       - Carbon (6 electrons, minimal basis): ~10 qubits
       - Iron (26 electrons, minimal basis): ~50 qubits

       Active space:
       - Focus on valence electrons
       - Freeze core electrons
       - Fe with 6 active electrons: ~20 qubits

    3. VQE FOR ATOMS
       Variational Quantum Eigensolver:
       - Prepare parameterized state
       - Measure energy
       - Optimize parameters

       Recent results:
       - H₂: Chemical accuracy achieved
       - LiH, BeH₂: ~1 mHa accuracy
       - Water molecule: Active research

    4. BEYOND GROUND STATES
       Quantum phase estimation:
       - Full energy spectrum
       - Excited states
       - Required for spectroscopy

    5. CURRENT LIMITATIONS
       - Noise in current devices
       - Limited qubit count
       - Gate fidelity issues

       Near-term targets:
       - Transition metal atoms
       - Small molecules
       - Model systems (Hubbard, etc.)

    The periodic table from quantum computing:
    Teaching atoms to a quantum computer!
    """)

def main():
    """Run all demonstrations."""

    print("Day 495: Multi-Electron Atoms")
    print("=" * 70)

    # Print configurations
    print_configurations()

    # Hund's rules demonstration
    hundsrules_demo()

    # Plot periodic trends
    plot_periodic_trends()

    # Quantum computing connection
    quantum_computing_connection()

if __name__ == "__main__":
    main()
```

---

## 11. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Central field approx. | Replace e-e repulsion with spherical average potential |
| Aufbau principle | Fill orbitals in order of increasing energy |
| Hund's rules | Maximize S, then L; J from spin-orbit coupling |
| Screening | Inner electrons reduce effective nuclear charge |
| Slater's rules | Empirical method to estimate $Z_{\text{eff}}$ |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$Z_{\text{eff}} = Z - \sigma$$ | Effective nuclear charge |
| $$E_n \approx -\frac{Z_{\text{eff}}^2}{2n^2}$$ | Single-electron energy |
| $$\text{Capacity} = 2(2l+1)$$ | Maximum electrons in subshell |
| $$\mu_s = \sqrt{n(n+2)}\mu_B$$ | Spin-only magnetic moment |

---

## 12. Daily Checklist

### Conceptual Understanding
- [ ] I can explain the central field approximation
- [ ] I understand why energy depends on both n and l
- [ ] I can apply Hund's rules to find ground state terms
- [ ] I understand periodic trends from quantum mechanics

### Mathematical Skills
- [ ] I can write electron configurations for any element
- [ ] I can determine term symbols using Hund's rules
- [ ] I can estimate $Z_{\text{eff}}$ using Slater's rules
- [ ] I can calculate approximate ionization energies

### Computational Skills
- [ ] I implemented aufbau filling algorithm
- [ ] I calculated and plotted periodic trends
- [ ] I visualized electron configurations

### Quantum Computing Connection
- [ ] I understand qubit requirements for atomic simulation
- [ ] I see how active space reduces problem size
- [ ] I know current limitations and targets

---

## 13. Preview: Day 496

Tomorrow we introduce the **Hartree-Fock method**:

- Self-consistent field equations
- Hartree equations and mean-field approximation
- Fock operator and exchange potential
- Iterative solution procedure
- Connection to VQE ansatze

---

## References

1. Griffiths, D.J. & Schroeter, D.F. (2018). *Introduction to Quantum Mechanics*, 3rd ed., Section 5.2.

2. Slater, J.C. (1930). "Atomic Shielding Constants." Phys. Rev. 36, 57.

3. Levine, I.N. (2013). *Quantum Chemistry*, 7th ed., Ch. 11.

4. Szabo, A. & Ostlund, N.S. (1996). *Modern Quantum Chemistry*. Dover, Ch. 2.

---

*"The aufbau principle, combined with Hund's rules and the Pauli exclusion principle, allows us to understand the entire periodic table from basic quantum mechanics—one of the great triumphs of 20th century physics."*
— Linus Pauling

---

**Day 495 Complete.** Tomorrow: Hartree-Fock Method.
