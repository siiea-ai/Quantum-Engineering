# Day 494: Exchange and Spin States in Helium

## Overview

**Day 494 of 2520 | Week 71, Day 4 | Month 18: Identical Particles & Many-Body Physics**

Today we explore one of the most striking manifestations of quantum mechanics: the exchange interaction. We will study how the requirement of antisymmetry for fermions leads to a splitting between singlet and triplet states in helium, giving rise to two distinct spectroscopic series—parahelium and orthohelium. This purely quantum effect, with no classical analog, underpins Hund's rules and magnetic phenomena in atoms and solids.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Symmetry Requirements for Fermions | 45 min |
| 9:45 AM | Singlet and Triplet Spin States | 75 min |
| 11:00 AM | Break | 15 min |
| 11:15 AM | Exchange Energy and Splitting | 90 min |
| 12:45 PM | Lunch | 60 min |
| 1:45 PM | Parahelium vs Orthohelium | 75 min |
| 3:00 PM | Break | 15 min |
| 3:15 PM | Selection Rules and Spectra | 60 min |
| 4:15 PM | Computational Lab | 75 min |
| 5:30 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Construct** antisymmetric wave functions from spatial and spin components
2. **Distinguish** between singlet ($S=0$) and triplet ($S=1$) spin states
3. **Calculate** the exchange integral and explain its physical origin
4. **Explain** why triplet states have lower energy than singlets
5. **Describe** the spectroscopic differences between parahelium and orthohelium
6. **Apply** selection rules to predict allowed transitions

---

## 1. Symmetry Requirements Review

### The Pauli Principle

For **identical fermions** (like electrons), the total wave function must be **antisymmetric** under particle exchange:

$$\boxed{\Psi(\mathbf{r}_2, s_2; \mathbf{r}_1, s_1) = -\Psi(\mathbf{r}_1, s_1; \mathbf{r}_2, s_2)}$$

### Factorization of Wave Function

For states where spin and spatial coordinates are not entangled:

$$\Psi_{\text{total}} = \psi_{\text{spatial}}(\mathbf{r}_1, \mathbf{r}_2) \times \chi_{\text{spin}}(s_1, s_2)$$

For the product to be antisymmetric:
- **Symmetric spatial × Antisymmetric spin**, or
- **Antisymmetric spatial × Symmetric spin**

### Ground State Revisited

For the ground state (both electrons in 1s):

$$\psi_{\text{spatial}}^{\text{ground}} = \psi_{1s}(r_1)\psi_{1s}(r_2)$$

This is **symmetric** under exchange ($r_1 \leftrightarrow r_2$).

Therefore, the spin state **must be antisymmetric**: the **singlet**.

---

## 2. Spin States for Two Electrons

### Building Blocks

Single-electron spin states: $|\uparrow\rangle = |\alpha\rangle$ and $|\downarrow\rangle = |\beta\rangle$

For two electrons, we combine these using addition of angular momentum.

### The Four Two-Spin States

**Uncoupled basis:**
$$|\uparrow\uparrow\rangle, \quad |\uparrow\downarrow\rangle, \quad |\downarrow\uparrow\rangle, \quad |\downarrow\downarrow\rangle$$

**Coupled basis (definite total spin S):**

$$\boxed{\chi_{\text{triplet}} (S=1, M_S): \begin{cases} |1,1\rangle = |\uparrow\uparrow\rangle \\ |1,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle) \\ |1,-1\rangle = |\downarrow\downarrow\rangle \end{cases}}$$

$$\boxed{\chi_{\text{singlet}} (S=0): \quad |0,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)}$$

### Symmetry Under Exchange

**Triplet states** (all three): Exchange $1 \leftrightarrow 2$ gives the **same state** → **Symmetric**

**Singlet state**: Exchange $1 \leftrightarrow 2$ gives **minus** the original → **Antisymmetric**

### Physical Interpretation

- **Singlet ($S=0$):** Spins antiparallel, no net magnetic moment from spin
- **Triplet ($S=1$):** Spins parallel, net magnetic moment of $\sqrt{2}\hbar$

The triplet has three states ($M_S = 1, 0, -1$); the singlet has one.

---

## 3. Excited States of Helium

### Configuration: One Electron Excited

Consider the configuration $1s \, nl$: one electron in 1s, one in $nl$ orbital.

**Available spatial wave functions:**

$$\psi_{\text{symm}}(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\psi_{1s}(r_1)\psi_{nl}(r_2) + \psi_{nl}(r_1)\psi_{1s}(r_2)]$$

$$\psi_{\text{anti}}(\mathbf{r}_1, \mathbf{r}_2) = \frac{1}{\sqrt{2}}[\psi_{1s}(r_1)\psi_{nl}(r_2) - \psi_{nl}(r_1)\psi_{1s}(r_2)]$$

### Complete Wave Functions

**Singlet state ($^1L$):**
$$\Psi_{\text{singlet}} = \psi_{\text{symm}} \times \chi_{\text{singlet}}$$

**Triplet state ($^3L$):**
$$\Psi_{\text{triplet}} = \psi_{\text{anti}} \times \chi_{\text{triplet}}$$

### Why Different Energies?

Even though we're using the same orbitals (1s and $nl$), the symmetry of the spatial wave function affects the **electron-electron repulsion**!

---

## 4. The Exchange Integral

### Energy Expectation Value

For the $1s \, nl$ configuration:

$$E = \langle \psi | \hat{H} | \psi \rangle = E_{1s} + E_{nl} + \langle \psi | \frac{1}{r_{12}} | \psi \rangle$$

The key is evaluating the repulsion term for symmetric vs antisymmetric spatial states.

### Direct and Exchange Integrals

Define:

**Direct (Coulomb) integral:**
$$\boxed{J = \int d^3r_1 \int d^3r_2 \, |\psi_{1s}(\mathbf{r}_1)|^2 \frac{1}{r_{12}} |\psi_{nl}(\mathbf{r}_2)|^2}$$

**Exchange integral:**
$$\boxed{K = \int d^3r_1 \int d^3r_2 \, \psi_{1s}^*(\mathbf{r}_1)\psi_{nl}^*(\mathbf{r}_2) \frac{1}{r_{12}} \psi_{nl}(\mathbf{r}_1)\psi_{1s}(\mathbf{r}_2)}$$

### Physical Meaning

**Direct integral $J$:** Classical Coulomb repulsion between electron clouds $|\psi_{1s}|^2$ and $|\psi_{nl}|^2$.

**Exchange integral $K$:** No classical analog! Arises from quantum interference due to indistinguishability.

### Energy for Each Spin State

For symmetric spatial (singlet):
$$E_{\text{singlet}} = E_{1s} + E_{nl} + J + K$$

For antisymmetric spatial (triplet):
$$E_{\text{triplet}} = E_{1s} + E_{nl} + J - K$$

### The Exchange Splitting

$$\boxed{E_{\text{singlet}} - E_{\text{triplet}} = 2K}$$

Since $K > 0$ (the integrand is positive for real orbitals), the **triplet lies lower in energy**!

---

## 5. Why Does the Triplet Have Lower Energy?

### The Fermi Hole

For the triplet (antisymmetric spatial):
$$\psi_{\text{anti}} \propto [\psi_{1s}(r_1)\psi_{nl}(r_2) - \psi_{nl}(r_1)\psi_{1s}(r_2)]$$

When $\mathbf{r}_1 = \mathbf{r}_2$:
$$\psi_{\text{anti}}(\mathbf{r}, \mathbf{r}) = 0$$

**The two electrons cannot be at the same position!**

This creates a "**Fermi hole**" around each electron—a region where the other electron is unlikely to be found.

### Reduced Repulsion

With the Fermi hole:
- Electrons stay farther apart on average
- Less Coulomb repulsion
- Lower energy for triplet state

### No Fermi Hole for Singlet

For the singlet (symmetric spatial):
$$\psi_{\text{symm}}(\mathbf{r}, \mathbf{r}) \neq 0$$

Electrons **can** be at the same position—no exclusion effect.
Result: Higher repulsion, higher energy.

### Hund's First Rule

This explains **Hund's first rule**: For a given configuration, the state with **maximum spin** has **lowest energy**.

More parallel spins → antisymmetric spatial → larger Fermi hole → less repulsion.

---

## 6. Parahelium and Orthohelium

### Two Spectroscopic Series

Helium exhibits two distinct spectral series, as if it were two different elements!

**Parahelium (Singlet states, $S=0$):**
- Ground state: $1s^2 \, ^1S_0$
- Excited states: $1s \, ns \, ^1S_0$, $1s \, np \, ^1P_1$, etc.
- Spins antiparallel

**Orthohelium (Triplet states, $S=1$):**
- Lowest state: $1s \, 2s \, ^3S_1$ (metastable)
- Excited states: $1s \, ns \, ^3S_1$, $1s \, np \, ^3P_{0,1,2}$, etc.
- Spins parallel

### Energy Level Diagram

```
Energy (eV)
    |
    |                    Singlet (Para)    Triplet (Ortho)
 0  |   ----------------------------------------  Ionization
    |
-4  |                   1s3s ¹S            1s3s ³S
    |                   1s3p ¹P            1s3p ³P
    |
-5  |                   1s2s ¹S            1s2s ³S  (metastable)
    |                   1s2p ¹P            1s2p ³P
    |
-24 |   1s² ¹S₀  (Ground state)
    |
```

### The Metastable State

The $1s2s \, ^3S_1$ state is **metastable**:
- Cannot decay to ground state by electric dipole radiation
- $\Delta S = 0$ rule forbids singlet ↔ triplet transitions
- Lifetime: ~8000 seconds (compared to ~1 ns for typical excited states)

### Historical Discovery

Before quantum mechanics, the two series were thought to come from different elements:
- "Parhelium" (singlet)
- "Orthohelium" (triplet)

Understanding exchange explained this as a single element with two spin configurations!

---

## 7. Selection Rules

### Electric Dipole Selection Rules

For transitions driven by electromagnetic radiation:

$$\boxed{\begin{aligned}
\Delta L &= \pm 1 \\
\Delta S &= 0 \\
\Delta J &= 0, \pm 1 \quad (J=0 \not\to J=0)
\end{aligned}}$$

### Consequences for Helium

1. **No singlet ↔ triplet transitions:** $\Delta S = 0$ means para and ortho don't interconvert easily

2. **Metastable $1s2s \, ^3S_1$:** Cannot go to $1s^2 \, ^1S_0$ because:
   - $\Delta S = -1$ (forbidden)
   - $\Delta L = 0$ (would need $\Delta L = \pm 1$)

3. **Allowed transitions:**
   - $1s2p \, ^1P_1 \to 1s^2 \, ^1S_0$ (strong, UV at 58.4 nm)
   - $1s3p \, ^3P \to 1s2s \, ^3S_1$ (visible, yellow at 587.6 nm)

### Fine Structure

Triplet states have fine structure due to spin-orbit coupling:
- $^3P_0$, $^3P_1$, $^3P_2$ have slightly different energies
- Leads to multiplet lines in the spectrum

---

## 8. Worked Examples

### Example 1: Exchange Integral Sign

**Problem:** Show that the exchange integral $K$ is positive for real orbitals.

**Solution:**

$$K = \int d^3r_1 \int d^3r_2 \, \psi_a^*(\mathbf{r}_1)\psi_b^*(\mathbf{r}_2) \frac{1}{r_{12}} \psi_b(\mathbf{r}_1)\psi_a(\mathbf{r}_2)$$

For real orbitals:
$$K = \int d^3r_1 \int d^3r_2 \, [\psi_a(\mathbf{r}_1)\psi_b(\mathbf{r}_2)][\psi_b(\mathbf{r}_1)\psi_a(\mathbf{r}_2)] \frac{1}{r_{12}}$$

Define $f(\mathbf{r}_1, \mathbf{r}_2) = \psi_a(\mathbf{r}_1)\psi_b(\mathbf{r}_2)$.

Then:
$$K = \int d^3r_1 \int d^3r_2 \, f(\mathbf{r}_1, \mathbf{r}_2) f(\mathbf{r}_2, \mathbf{r}_1) \frac{1}{r_{12}}$$

Since $1/r_{12} > 0$ and the product $f(\mathbf{r}_1, \mathbf{r}_2)f(\mathbf{r}_2, \mathbf{r}_1)$ is positive when the overlap is significant:

$$\boxed{K > 0 \text{ for real orbitals with non-zero overlap}}$$

### Example 2: Exchange Splitting for 1s2s

**Problem:** The direct integral for the $1s \, 2s$ configuration of helium is $J = 0.42$ Hartree and the exchange integral is $K = 0.024$ Hartree. Calculate the singlet-triplet splitting.

**Solution:**

$$\Delta E = E_{\text{singlet}} - E_{\text{triplet}} = 2K$$

$$\Delta E = 2 \times 0.024 = 0.048 \text{ Hartree}$$

Converting to eV:
$$\Delta E = 0.048 \times 27.2 = 1.31 \text{ eV}$$

Experimental value: 0.80 eV

$$\boxed{\Delta E = 2K = 0.048 \text{ Hartree} = 1.31 \text{ eV}}$$

### Example 3: Why No Ground State Triplet?

**Problem:** Explain why there is no triplet ground state for helium.

**Solution:**

For the ground state, both electrons must be in the 1s orbital:
$$\psi_{\text{spatial}} = \psi_{1s}(r_1)\psi_{1s}(r_2)$$

This is automatically **symmetric** under particle exchange.

For a triplet state, we need **antisymmetric** spatial:
$$\psi_{\text{anti}} \propto \psi_{1s}(r_1)\psi_{1s}(r_2) - \psi_{1s}(r_1)\psi_{1s}(r_2) = 0$$

The antisymmetric combination vanishes identically when both orbitals are the same!

$$\boxed{\text{No triplet ground state because antisymmetric spatial part is zero for identical orbitals}}$$

---

## 9. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Write the complete wave function (spatial and spin) for the $1s2p \, ^3P$ state of helium.

**Problem 1.2:** How many substates (counting $M_S$ and $M_L$) does the $1s2p \, ^3P$ term have?

**Problem 1.3:** Which transition is allowed: $1s2p \, ^1P_1 \to 1s2s \, ^1S_0$ or $1s2p \, ^1P_1 \to 1s2s \, ^3S_1$?

### Level 2: Intermediate

**Problem 2.1:** Show that the triplet spin states are eigenstates of $\hat{S}^2$ with eigenvalue $2\hbar^2$.

**Problem 2.2:** For the $1s2p$ configuration, the exchange integral is approximately $K \approx 0.018$ Hartree. Estimate the wavelength of the photon emitted in the $1s2p \, ^3P \to 1s2s \, ^3S$ transition.

**Problem 2.3:** Explain why orthohelium has a larger mean electron separation than parahelium in the same configuration.

### Level 3: Challenging

**Problem 3.1:** Derive the matrix element $\langle S, M_S | \hat{S}_1 \cdot \hat{S}_2 | S, M_S \rangle$ for singlet and triplet states.

**Problem 3.2:** The $1s2s \, ^3S_1$ state decays to the ground state with a lifetime of about 8000 seconds. What type of transition (M1, E2, etc.) is responsible? Estimate the transition rate.

**Problem 3.3:** Using the Pauli exclusion principle, explain why the exchange interaction leads to ferromagnetism in some materials.

---

## 10. Computational Lab: Exchange Effects in Helium

```python
"""
Day 494 Computational Lab: Exchange and Spin States in Helium
Visualization of singlet-triplet splitting and exchange effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import factorial

HARTREE_TO_EV = 27.211

def hydrogen_1s(r, Z=2):
    """Normalized 1s orbital."""
    return np.sqrt(Z**3 / np.pi) * np.exp(-Z * r)

def hydrogen_2s(r, Z=2):
    """Normalized 2s orbital."""
    return np.sqrt(Z**3 / (32 * np.pi)) * (2 - Z * r) * np.exp(-Z * r / 2)

def hydrogen_2p(r, Z=2):
    """Radial part of 2p orbital (without angular factor)."""
    return np.sqrt(Z**5 / (32 * np.pi)) * r * np.exp(-Z * r / 2)

def spatial_symmetric(r1, r2, psi_a, psi_b):
    """Symmetric spatial wave function."""
    return (1/np.sqrt(2)) * (psi_a(r1) * psi_b(r2) + psi_b(r1) * psi_a(r2))

def spatial_antisymmetric(r1, r2, psi_a, psi_b):
    """Antisymmetric spatial wave function."""
    return (1/np.sqrt(2)) * (psi_a(r1) * psi_b(r2) - psi_b(r1) * psi_a(r2))

def plot_spin_states():
    """Visualize the four two-spin states."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Triplet states
    ax1 = axes[0, 0]
    ax1.annotate('', xy=(0.5, 0.8), xytext=(0.3, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax1.annotate('', xy=(0.5, 0.8), xytext=(0.7, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(r'Triplet $|1,1\rangle = |\uparrow\uparrow\rangle$', fontsize=14)
    ax1.axis('off')
    ax1.text(0.5, 0.2, r'$S=1, M_S=+1$', ha='center', fontsize=12)

    ax2 = axes[0, 1]
    ax2.annotate('', xy=(0.3, 0.8), xytext=(0.3, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax2.annotate('', xy=(0.7, 0.5), xytext=(0.7, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(r'Triplet $|1,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle)$', fontsize=12)
    ax2.axis('off')
    ax2.text(0.5, 0.2, r'$S=1, M_S=0$ (symmetric)', ha='center', fontsize=12)

    ax3 = axes[1, 0]
    ax3.annotate('', xy=(0.3, 0.5), xytext=(0.3, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax3.annotate('', xy=(0.7, 0.5), xytext=(0.7, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title(r'Triplet $|1,-1\rangle = |\downarrow\downarrow\rangle$', fontsize=14)
    ax3.axis('off')
    ax3.text(0.5, 0.2, r'$S=1, M_S=-1$', ha='center', fontsize=12)

    # Singlet state
    ax4 = axes[1, 1]
    ax4.annotate('', xy=(0.3, 0.8), xytext=(0.3, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax4.annotate('', xy=(0.7, 0.5), xytext=(0.7, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax4.text(0.5, 0.65, '-', fontsize=30, ha='center')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title(r'Singlet $|0,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$', fontsize=12)
    ax4.axis('off')
    ax4.text(0.5, 0.2, r'$S=0, M_S=0$ (antisymmetric)', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('spin_states.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_fermi_hole():
    """Visualize the Fermi hole in triplet state."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    r = np.linspace(0, 6, 200)
    r1_fixed = 1.0  # Fix electron 1 at r = 1

    # Symmetric spatial (singlet)
    psi_sym_sq = []
    psi_anti_sq = []

    for r2 in r:
        psi1_a = hydrogen_1s(r1_fixed)
        psi1_b = hydrogen_1s(r2)
        psi2_a = hydrogen_2s(r1_fixed)
        psi2_b = hydrogen_2s(r2)

        # Simplified 1D visualization
        sym = (psi1_a * psi2_b + psi2_a * psi1_b) / np.sqrt(2)
        anti = (psi1_a * psi2_b - psi2_a * psi1_b) / np.sqrt(2)

        psi_sym_sq.append(sym**2)
        psi_anti_sq.append(anti**2)

    psi_sym_sq = np.array(psi_sym_sq)
    psi_anti_sq = np.array(psi_anti_sq)

    # Normalize for visualization
    psi_sym_sq /= np.max(psi_sym_sq)
    psi_anti_sq /= np.max(psi_anti_sq)

    ax1 = axes[0]
    ax1.plot(r, psi_sym_sq, 'b-', linewidth=2, label='Singlet (symmetric spatial)')
    ax1.axvline(x=r1_fixed, color='gray', linestyle='--', label='Electron 1 position')
    ax1.set_xlabel('r₂ (a₀)', fontsize=12)
    ax1.set_ylabel('|ψ(r₁=1, r₂)|² (normalized)', fontsize=12)
    ax1.set_title('Singlet: No Fermi Hole', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(r, psi_anti_sq, 'r-', linewidth=2, label='Triplet (antisymmetric spatial)')
    ax2.axvline(x=r1_fixed, color='gray', linestyle='--', label='Electron 1 position')
    ax2.fill_between(r[r < r1_fixed + 0.5], 0, psi_anti_sq[r < r1_fixed + 0.5],
                     alpha=0.3, color='yellow', label='Fermi hole region')
    ax2.set_xlabel('r₂ (a₀)', fontsize=12)
    ax2.set_ylabel('|ψ(r₁=1, r₂)|² (normalized)', fontsize=12)
    ax2.set_title('Triplet: Fermi Hole (electrons avoid each other)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fermi_hole.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_helium_energy_levels():
    """Plot helium energy level diagram."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Experimental energy levels (eV, relative to He+ + e-)
    # Ionization energy of He is 24.59 eV
    levels = {
        # Singlet (parahelium)
        '1s² ¹S₀': -24.59,
        '1s2s ¹S₀': -4.77,
        '1s2p ¹P₁': -3.62,
        '1s3s ¹S₀': -1.87,
        '1s3p ¹P₁': -1.50,

        # Triplet (orthohelium)
        '1s2s ³S₁': -5.57,
        '1s2p ³P': -4.26,
        '1s3s ³S₁': -2.18,
        '1s3p ³P': -1.87,
    }

    # Plot singlet levels (left side)
    singlet_x = 0.3
    singlet_levels = [(k, v) for k, v in levels.items() if '¹' in k]

    for name, energy in singlet_levels:
        ax.hlines(energy, singlet_x - 0.15, singlet_x + 0.15, colors='blue', linewidth=2)
        ax.text(singlet_x - 0.18, energy, name, ha='right', va='center', fontsize=9, color='blue')

    # Plot triplet levels (right side)
    triplet_x = 0.7
    triplet_levels = [(k, v) for k, v in levels.items() if '³' in k]

    for name, energy in triplet_levels:
        ax.hlines(energy, triplet_x - 0.15, triplet_x + 0.15, colors='red', linewidth=2)
        ax.text(triplet_x + 0.18, energy, name, ha='left', va='center', fontsize=9, color='red')

    # Draw some transitions
    # Singlet allowed
    ax.annotate('', xy=(singlet_x, -24.59), xytext=(singlet_x, -3.62),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(singlet_x + 0.05, -14, '58.4 nm\n(UV)', fontsize=8, color='green')

    # Triplet allowed
    ax.annotate('', xy=(triplet_x, -5.57), xytext=(triplet_x, -4.26),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
    ax.text(triplet_x + 0.05, -4.9, '1083 nm\n(IR)', fontsize=8, color='orange')

    # Forbidden transition (singlet-triplet)
    ax.annotate('', xy=(0.5, -24.59), xytext=(0.5, -5.57),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
    ax.text(0.52, -15, 'Forbidden!\nΔS ≠ 0', fontsize=8, color='gray')

    # Labels
    ax.text(singlet_x, 0.5, 'PARAHELIUM\n(Singlet, S=0)', ha='center', fontsize=12, color='blue')
    ax.text(triplet_x, 0.5, 'ORTHOHELIUM\n(Triplet, S=1)', ha='center', fontsize=12, color='red')

    # Ionization limit
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.text(0.5, 0.3, 'Ionization (He⁺ + e⁻)', ha='center', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(-26, 2)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('Helium Energy Level Diagram', fontsize=14)
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig('helium_energy_levels.png', dpi=150, bbox_inches='tight')
    plt.show()

def calculate_exchange_splitting():
    """Demonstrate exchange splitting calculation."""

    print("=" * 60)
    print("EXCHANGE SPLITTING IN HELIUM")
    print("=" * 60)

    # Approximate values for 1s2s configuration
    J_1s2s = 0.419  # Hartree (direct integral)
    K_1s2s = 0.022  # Hartree (exchange integral)

    E_1s = -2.0  # Hartree (approximate)
    E_2s = -0.5  # Hartree (approximate)

    E_singlet = E_1s + E_2s + J_1s2s + K_1s2s
    E_triplet = E_1s + E_2s + J_1s2s - K_1s2s

    print(f"\n1s2s Configuration:")
    print(f"  Direct integral J = {J_1s2s:.4f} Hartree = {J_1s2s * HARTREE_TO_EV:.2f} eV")
    print(f"  Exchange integral K = {K_1s2s:.4f} Hartree = {K_1s2s * HARTREE_TO_EV:.2f} eV")
    print(f"\nEnergies (relative to He²⁺):")
    print(f"  E(¹S) = {E_singlet:.4f} Hartree = {E_singlet * HARTREE_TO_EV:.2f} eV")
    print(f"  E(³S) = {E_triplet:.4f} Hartree = {E_triplet * HARTREE_TO_EV:.2f} eV")
    print(f"\nSinglet-Triplet splitting:")
    print(f"  ΔE = 2K = {2*K_1s2s:.4f} Hartree = {2*K_1s2s * HARTREE_TO_EV:.2f} eV")
    print(f"\nExperimental splitting: ~0.80 eV")

    # More configurations
    print("\n" + "-" * 60)
    print("Exchange splittings for various configurations:")
    print("-" * 60)

    configs = [
        ('1s2s', 0.022, 0.80),
        ('1s2p', 0.018, 0.64),
        ('1s3s', 0.008, 0.35),
        ('1s3p', 0.006, 0.27),
    ]

    print(f"{'Config':<10} {'K (Ha)':<12} {'2K (eV)':<12} {'Expt (eV)':<12}")
    for config, K, expt in configs:
        print(f"{config:<10} {K:<12.4f} {2*K*HARTREE_TO_EV:<12.3f} {expt:<12.2f}")

def quantum_computing_connection():
    """Connection to quantum computing."""

    print("\n" + "=" * 60)
    print("QUANTUM COMPUTING CONNECTION")
    print("=" * 60)

    print("""
    EXCHANGE INTERACTION IN QUANTUM COMPUTING
    ==========================================

    1. ENCODING SPIN STATES ON QUBITS
       |↑⟩ → |0⟩,  |↓⟩ → |1⟩

       Singlet: (|01⟩ - |10⟩)/√2  (Bell state!)
       Triplet: |00⟩, (|01⟩ + |10⟩)/√2, |11⟩

    2. CREATING SINGLET/TRIPLET STATES
       Start with |00⟩
       Apply H gate to qubit 1: (|0⟩+|1⟩)|0⟩/√2
       Apply CNOT: (|00⟩ + |11⟩)/√2  (triplet M_S=0 is NOT this!)

       Actually:
       For singlet: H, then CNOT, then Z on qubit 2
       |00⟩ → (|00⟩ + |10⟩)/√2 → (|00⟩ + |11⟩)/√2 → (|01⟩ - |10⟩)/√2

    3. SIMULATING EXCHANGE INTERACTION
       Exchange Hamiltonian: H_ex = J S₁·S₂

       In qubit form:
       H_ex = (J/4)(XX + YY + ZZ)

       This naturally appears in:
       - Superconducting qubits (capacitive coupling)
       - Quantum dots (tunnel coupling)
       - Trapped ions (spin-spin interaction)

    4. VQE FOR HELIUM WITH EXCHANGE
       Helium Hamiltonian includes:
       H = ... + 1/r₁₂

       The electron-electron term maps to:
       - Direct Coulomb (diagonal terms)
       - Exchange (off-diagonal terms)

       VQE must capture both for accuracy!

    5. QUANTUM ADVANTAGE
       Exchange effects grow with system size
       Classical methods struggle with strong correlations
       Quantum computers handle exchange naturally
    """)

def selection_rules_demo():
    """Demonstrate selection rules."""

    print("\n" + "=" * 60)
    print("SELECTION RULES FOR HELIUM TRANSITIONS")
    print("=" * 60)

    transitions = [
        ('1s2p ¹P₁', '1s² ¹S₀', 1, 0, 'Allowed', '58.4 nm (UV)'),
        ('1s2s ¹S₀', '1s² ¹S₀', 0, 0, 'Forbidden (ΔL=0)', 'None'),
        ('1s2s ³S₁', '1s² ¹S₀', 0, -1, 'Forbidden (ΔS≠0)', 'Metastable 8000s'),
        ('1s2p ³P₁', '1s2s ³S₁', 1, 0, 'Allowed', '1083 nm (IR)'),
        ('1s3d ¹D₂', '1s2p ¹P₁', 1, 0, 'Allowed', '667.8 nm (red)'),
    ]

    print(f"\n{'Transition':<30} {'ΔL':<5} {'ΔS':<5} {'Status':<20} {'Wavelength':<15}")
    print("-" * 80)

    for init, final, dL, dS, status, wavelength in transitions:
        transition = f"{init} → {final}"
        print(f"{transition:<30} {dL:<5} {dS:<5} {status:<20} {wavelength:<15}")

def main():
    """Run all demonstrations."""

    print("Day 494: Exchange and Spin States in Helium")
    print("=" * 60)

    # Spin states visualization
    plot_spin_states()

    # Fermi hole
    plot_fermi_hole()

    # Energy level diagram
    plot_helium_energy_levels()

    # Exchange splitting calculation
    calculate_exchange_splitting()

    # Selection rules
    selection_rules_demo()

    # Quantum computing
    quantum_computing_connection()

if __name__ == "__main__":
    main()
```

---

## 11. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Singlet ($S=0$) | Antisymmetric spin, symmetric spatial |
| Triplet ($S=1$) | Symmetric spin, antisymmetric spatial |
| Exchange integral $K$ | Quantum interference term, no classical analog |
| Fermi hole | Region where another fermion is excluded |
| Parahelium | Singlet series (includes ground state) |
| Orthohelium | Triplet series (metastable lowest state) |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$E_{\text{singlet}} = E_0 + J + K$$ | Singlet energy (higher) |
| $$E_{\text{triplet}} = E_0 + J - K$$ | Triplet energy (lower) |
| $$\Delta E = 2K$$ | Singlet-triplet splitting |
| $$K = \int \psi_a^*\psi_b^* \frac{1}{r_{12}} \psi_b \psi_a$$ | Exchange integral |

---

## 12. Daily Checklist

### Conceptual Understanding
- [ ] I can distinguish singlet and triplet spin states
- [ ] I understand why triplet states have lower energy
- [ ] I can explain the Fermi hole concept
- [ ] I know why para and orthohelium don't easily interconvert

### Mathematical Skills
- [ ] I can construct antisymmetric wave functions
- [ ] I can set up direct and exchange integrals
- [ ] I can apply selection rules to transitions
- [ ] I can calculate singlet-triplet splitting

### Computational Skills
- [ ] I visualized singlet and triplet states
- [ ] I plotted the Fermi hole effect
- [ ] I created a helium energy level diagram

### Quantum Computing Connection
- [ ] I understand how spin states map to qubits
- [ ] I see how exchange appears in qubit Hamiltonians
- [ ] I know why quantum computers handle exchange naturally

---

## 13. Preview: Day 495

Tomorrow we extend to **multi-electron atoms**:

- Central field approximation
- Self-consistent field concept
- Aufbau principle and electron configurations
- Hund's rules from exchange considerations
- Understanding the periodic table structure
- Ionization energies and trends

---

## References

1. Griffiths, D.J. & Schroeter, D.F. (2018). *Introduction to Quantum Mechanics*, 3rd ed., Section 5.2.

2. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 8.

3. NIST Atomic Spectra Database: https://physics.nist.gov/PhysRefData/ASD/

4. Drake, G.W.F. (2006). "High-precision calculations for helium." Springer Handbook of Atomic, Molecular, and Optical Physics.

---

*"The exchange interaction, with its lowering of energy for parallel spins, is one of the most important effects in quantum mechanics, responsible for everything from the periodic table to ferromagnetism."*
— John C. Slater

---

**Day 494 Complete.** Tomorrow: Multi-Electron Atoms.
