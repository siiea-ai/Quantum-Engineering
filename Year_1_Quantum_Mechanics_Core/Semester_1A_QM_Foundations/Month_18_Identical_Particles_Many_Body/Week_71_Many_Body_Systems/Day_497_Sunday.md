# Day 497: Week 71 Review - Many-Body Systems

## Overview

**Day 497 of 2520 | Week 71, Day 7 | Month 18: Identical Particles & Many-Body Physics**

Today we consolidate our understanding of many-body systems through comprehensive review, problem solving, and self-assessment. This week covered the helium atom and multi-electron systems—from basic perturbation and variational methods to the sophisticated Hartree-Fock approach. These techniques form the foundation for computational quantum chemistry and connect directly to quantum computing applications.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | Concept Review: Helium Atom Methods | 60 min |
| 10:00 AM | Concept Review: Exchange and Spin States | 60 min |
| 11:00 AM | Break | 15 min |
| 11:15 AM | Concept Review: Multi-Electron Atoms & Hartree-Fock | 75 min |
| 12:30 PM | Lunch | 60 min |
| 1:30 PM | Comprehensive Problem Set | 120 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Problem Set Solutions & Discussion | 75 min |
| 5:00 PM | Self-Assessment & Reflection | 60 min |
| 6:00 PM | Preview of Upcoming Topics | 30 min |

**Total Study Time:** 7 hours

---

## 1. Concept Review: Week 71 Summary

### Day 491: Helium Atom Setup

**Key Concepts:**
- **Helium Hamiltonian:**
$$\hat{H} = -\frac{1}{2}\nabla_1^2 - \frac{1}{2}\nabla_2^2 - \frac{Z}{r_1} - \frac{Z}{r_2} + \frac{1}{r_{12}}$$

- **Independent Particle Approximation:**
$$E_0^{(0)} = -Z^2 = -4 \text{ Ha} = -108.8 \text{ eV}$$
Error: 38% (ignores electron-electron repulsion)

- **Ground State Configuration:** $1s^2$, singlet spin state

### Day 492: Perturbation Approach

**Key Result:**
$$E^{(1)} = \left\langle \frac{1}{r_{12}} \right\rangle = \frac{5Z}{8} = 1.25 \text{ Ha for He}$$

**First-Order Total Energy:**
$$E = -Z^2 + \frac{5Z}{8} = -2.75 \text{ Ha}$$

Error reduced to 5%—much better but still not quantitative.

### Day 493: Variational Method

**Trial Function:** $\psi = (Z_{\text{eff}}^3/\pi)e^{-Z_{\text{eff}}(r_1+r_2)}$

**Optimal Effective Charge:**
$$Z_{\text{eff}} = Z - \frac{5}{16} = 1.6875$$

**Variational Energy:** $E = -2.8477$ Ha (1.9% error)

**Physical Insight:** Screening reduces effective nuclear charge.

### Day 494: Exchange and Spin States

**Singlet-Triplet Splitting:**
$$E_{\text{singlet}} - E_{\text{triplet}} = 2K$$

where $K$ is the exchange integral.

**Key Results:**
- Triplet lies lower due to Fermi hole (reduced repulsion)
- Parahelium (singlet) and orthohelium (triplet) series
- Selection rule $\Delta S = 0$ prevents interconversion

### Day 495: Multi-Electron Atoms

**Central Field Approximation:** Spherically averaged electron-electron potential

**Aufbau Principle:** Fill orbitals in order: $1s < 2s < 2p < 3s < ...$

**Hund's Rules:**
1. Maximize $S$
2. Maximize $L$ (consistent with Rule 1)
3. $J = |L-S|$ (< half-filled) or $J = L+S$ (> half-filled)

**Slater Screening:** $Z_{\text{eff}} = Z - \sigma$

### Day 496: Hartree-Fock Introduction

**Fock Operator:**
$$\hat{f} = \hat{h} + \sum_j (\hat{J}_j - \hat{K}_j)$$

**Self-Consistent Field:** Iterate until orbitals and potential are consistent

**Hartree-Fock Energy for He:** $E_{HF} = -2.8617$ Ha

**Correlation Energy:** $E_{\text{corr}} = E_{\text{exact}} - E_{HF} = -0.042$ Ha

---

## 2. Master Formula Sheet

### Helium Atom

| Formula | Description |
|---------|-------------|
| $$E_0^{(0)} = -Z^2$$ | Zeroth-order (independent particle) |
| $$E^{(1)} = \frac{5Z}{8}$$ | First-order correction |
| $$Z_{\text{eff}} = Z - \frac{5}{16}$$ | Optimal variational parameter |
| $$E_{\text{var}} = -Z^2 + \frac{5Z}{8} - \frac{25}{256}$$ | Variational energy |

### Exchange and Spin

| Formula | Description |
|---------|-------------|
| $$\chi_{\text{singlet}} = \frac{1}{\sqrt{2}}(\|\uparrow\downarrow\rangle - \|\downarrow\uparrow\rangle)$$ | Singlet spin state |
| $$\chi_{\text{triplet}} = \|\uparrow\uparrow\rangle, \frac{1}{\sqrt{2}}(\|\uparrow\downarrow\rangle + \|\downarrow\uparrow\rangle), \|\downarrow\downarrow\rangle$$ | Triplet spin states |
| $$J = \int \|\psi_a\|^2 \frac{1}{r_{12}} \|\psi_b\|^2$$ | Coulomb (direct) integral |
| $$K = \int \psi_a^*\psi_b^* \frac{1}{r_{12}} \psi_b\psi_a$$ | Exchange integral |
| $$\Delta E = 2K$$ | Singlet-triplet splitting |

### Multi-Electron Atoms

| Formula | Description |
|---------|-------------|
| $$Z_{\text{eff}} = Z - \sigma$$ | Effective nuclear charge |
| $$E_n \approx -\frac{Z_{\text{eff}}^2}{2n^2}$$ | Orbital energy |
| $$\mu_s = \sqrt{n(n+2)}\mu_B$$ | Spin-only magnetic moment |

### Hartree-Fock

| Formula | Description |
|---------|-------------|
| $$\hat{f}\|\chi_i\rangle = \varepsilon_i\|\chi_i\rangle$$ | Hartree-Fock equation |
| $$E_{HF} = \sum_i h_{ii} + \frac{1}{2}\sum_{ij}(J_{ij} - K_{ij})$$ | HF total energy |
| $$E_{\text{corr}} = E_{\text{exact}} - E_{HF}$$ | Correlation energy |

---

## 3. Comprehensive Problem Set

### Part A: Helium Atom Fundamentals (25 points)

**Problem A1 (5 pts):** The helium Hamiltonian in atomic units is $\hat{H} = -\frac{1}{2}\nabla_1^2 - \frac{1}{2}\nabla_2^2 - \frac{2}{r_1} - \frac{2}{r_2} + \frac{1}{r_{12}}$.
(a) Identify each term physically.
(b) Why can't this be solved exactly analytically?

**Problem A2 (5 pts):** Calculate the zeroth-order ground state energy for Li⁺ (two electrons, Z=3). What is the percentage error compared to the experimental value of -7.28 Ha?

**Problem A3 (5 pts):** Using the result $\langle 1/r_{12} \rangle = 5Z_{\text{eff}}/8$, calculate the first-order energy correction for He with the variational wave function ($Z_{\text{eff}} = 1.6875$).

**Problem A4 (5 pts):** Show that the variational energy functional $E(Z_{\text{eff}}) = Z_{\text{eff}}^2 - 2ZZ_{\text{eff}} + \frac{5Z_{\text{eff}}}{8}$ has a minimum at $Z_{\text{eff}} = Z - 5/16$.

**Problem A5 (5 pts):** Compare the energies from zeroth-order, first-order perturbation, and variational methods for Be²⁺ (Z=4). Which is closest to the experimental value of -13.66 Ha?

### Part B: Exchange and Spin States (25 points)

**Problem B1 (5 pts):** Construct the complete wave function (spatial and spin) for the $1s2s \, ^3S_1$ state of helium.

**Problem B2 (5 pts):** Show that the triplet spin state $|1,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle)$ is symmetric under particle exchange.

**Problem B3 (5 pts):** For the $1s2p$ configuration of helium:
(a) What are the possible term symbols?
(b) Which has lower energy according to Hund's rules?

**Problem B4 (5 pts):** The exchange integral for the $1s2s$ configuration is $K = 0.022$ Ha. Calculate:
(a) The singlet-triplet splitting in eV
(b) The wavelength of a photon with this energy

**Problem B5 (5 pts):** Explain physically why two electrons in the same spatial orbital must have opposite spins (be in a singlet state).

### Part C: Multi-Electron Atoms (25 points)

**Problem C1 (5 pts):** Write the electron configuration and determine the ground state term symbol for:
(a) Vanadium (V, Z=23)
(b) Chlorine (Cl, Z=17)

**Problem C2 (5 pts):** Using Slater's rules, calculate $Z_{\text{eff}}$ for a 3d electron in iron (Z=26, configuration $[Ar]3d^64s^2$).

**Problem C3 (5 pts):** Explain the anomalous electron configurations:
(a) Chromium: $[Ar]3d^54s^1$ instead of $[Ar]3d^44s^2$
(b) Copper: $[Ar]3d^{10}4s^1$ instead of $[Ar]3d^94s^2$

**Problem C4 (5 pts):** Calculate the spin-only magnetic moment for:
(a) Ti³⁺ (d¹)
(b) Fe³⁺ (d⁵)

**Problem C5 (5 pts):** Why does ionization energy generally increase across a period but decrease down a group?

### Part D: Hartree-Fock and Quantum Computing (25 points)

**Problem D1 (5 pts):** For helium with both electrons in the 1s orbital, write the Hartree-Fock energy in terms of the one-electron integral $h_{11}$ and Coulomb integral $J_{11}$.

**Problem D2 (5 pts):** Explain why the Fock operator includes both Coulomb ($\hat{J}$) and exchange ($\hat{K}$) operators. What would happen if we omitted $\hat{K}$?

**Problem D3 (5 pts):** The Hartree-Fock energy of neon is -128.547 Ha and the exact energy is -128.937 Ha.
(a) What is the correlation energy?
(b) What percentage of the total energy is correlation?

**Problem D4 (5 pts):** Describe the self-consistent field (SCF) procedure in 4-5 steps.

**Problem D5 (5 pts):** How does the Variational Quantum Eigensolver (VQE) build upon the Hartree-Fock reference state? What types of effects can VQE capture that Hartree-Fock cannot?

---

## 4. Problem Set Solutions

### Part A Solutions

**A1:**
(a) Terms:
- $-\frac{1}{2}\nabla_1^2$: Kinetic energy of electron 1
- $-\frac{1}{2}\nabla_2^2$: Kinetic energy of electron 2
- $-\frac{2}{r_1}$: Attraction of electron 1 to nucleus
- $-\frac{2}{r_2}$: Attraction of electron 2 to nucleus
- $\frac{1}{r_{12}}$: Electron-electron repulsion

(b) The $1/r_{12}$ term depends on both electron positions and the angle between them. No coordinate system separates all variables simultaneously.

**A2:**
$E_0^{(0)} = -Z^2 = -9$ Ha

Error: $\frac{-9 - (-7.28)}{7.28} \times 100\% = -23.6\%$ (overbinds)

**A3:**
$E^{(1)} = \frac{5 \times 1.6875}{8} = 1.055$ Ha

**A4:**
$\frac{dE}{dZ_{\text{eff}}} = 2Z_{\text{eff}} - 2Z + \frac{5}{8} = 0$

$Z_{\text{eff}} = Z - \frac{5}{16}$ ✓

**A5:**
For Be²⁺ (Z=4):
- Zeroth: $E = -16$ Ha
- First-order: $E = -16 + \frac{20}{8} = -13.5$ Ha
- Variational: $E = (4-5/16)^2 - 8(4-5/16) + 5(4-5/16)/8 = -13.60$ Ha

Variational (-13.60 Ha) is closest to experiment (-13.66 Ha).

### Part B Solutions

**B1:**
$\Psi_{^3S_1} = \frac{1}{\sqrt{2}}[\psi_{1s}(r_1)\psi_{2s}(r_2) - \psi_{2s}(r_1)\psi_{1s}(r_2)] \times \chi_{\text{triplet}}$

For $M_S = 1$: $\times |\uparrow\uparrow\rangle$

**B2:**
Under exchange $1 \leftrightarrow 2$:
$\frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle) \to \frac{1}{\sqrt{2}}(|\downarrow\uparrow\rangle + |\uparrow\downarrow\rangle)$

Same state → symmetric ✓

**B3:**
(a) $1s2p$: $L = 1$ (P), $S = 0$ or $1$ → $^1P_1$ and $^3P_{0,1,2}$

(b) Triplet $^3P$ has lower energy (Hund's Rule 1)

**B4:**
(a) $\Delta E = 2K = 0.044$ Ha $= 1.2$ eV

(b) $\lambda = hc/E = \frac{1240 \text{ eV·nm}}{1.2 \text{ eV}} \approx 1000$ nm (infrared)

**B5:**
If both electrons are in the same spatial orbital, the spatial wave function is symmetric. For the total wave function to be antisymmetric (Pauli principle), the spin part must be antisymmetric, which is the singlet state.

### Part C Solutions

**C1:**
(a) V (Z=23): $[Ar]3d^34s^2$, ground term $^4F_{3/2}$
(b) Cl (Z=17): $[Ne]3s^23p^5$, ground term $^2P_{3/2}$

**C2:**
For 3d electron in Fe:
- Same group (other 3d): $5 \times 0.35 = 1.75$
- 4s electrons: $2 \times 0.00 = 0$ (higher n)
- 3s, 3p: $8 \times 1.00 = 8$
- 2s, 2p: $8 \times 1.00 = 8$
- 1s: $2 \times 1.00 = 2$

$\sigma = 19.75$, $Z_{\text{eff}} = 26 - 19.75 = 6.25$

**C3:**
(a) Chromium: Half-filled 3d subshell ($3d^5$) has extra stability from maximum exchange
(b) Copper: Filled 3d subshell ($3d^{10}$) has extra stability

Both cases sacrifice one 4s electron to achieve more stable d configuration.

**C4:**
(a) Ti³⁺ (d¹): $\mu_s = \sqrt{1(3)} \mu_B = 1.73 \mu_B$
(b) Fe³⁺ (d⁵): $\mu_s = \sqrt{5(7)} \mu_B = 5.92 \mu_B$

**C5:**
- Across period: $Z_{\text{eff}}$ increases while $n$ stays constant → electrons more tightly bound
- Down group: $n$ increases while $Z_{\text{eff}}$ increases less → electrons farther from nucleus and easier to remove

### Part D Solutions

**D1:**
$E_{HF} = 2h_{11} + J_{11}$

(Note: $K_{11} = J_{11}$ for same orbital, but for closed shell the net effect is just $J_{11}$)

**D2:**
$\hat{J}$ accounts for classical electrostatic repulsion between electron charge densities.
$\hat{K}$ accounts for exchange effects due to Pauli principle.

Without $\hat{K}$: Would have unphysical self-interaction (electron repelling itself), would miss exchange stabilization of parallel spins.

**D3:**
(a) $E_{\text{corr}} = -128.937 - (-128.547) = -0.390$ Ha

(b) Percentage: $\frac{0.390}{128.937} \times 100\% = 0.30\%$

**D4:**
1. Guess initial orbitals
2. Build Fock matrix from current orbitals
3. Diagonalize Fock matrix → new orbitals and energies
4. Check convergence (density change < threshold)
5. If not converged, return to step 2

**D5:**
VQE uses HF as reference: $|\Psi_{VQE}\rangle = U(\theta)|\Psi_{HF}\rangle$

VQE captures:
- Dynamic correlation (electron-electron avoidance)
- Static correlation (near-degeneracy)
- Multi-reference character

HF cannot capture these because it uses a single Slater determinant.

---

## 5. Self-Assessment Checklist

### Helium Atom
- [ ] I can write the helium Hamiltonian and identify all terms
- [ ] I can calculate zeroth-order energy for He-like ions
- [ ] I can evaluate first-order perturbation correction
- [ ] I can derive optimal $Z_{\text{eff}}$ using variational method
- [ ] I understand why each method gives different accuracy

### Exchange and Spin States
- [ ] I can construct singlet and triplet spin states
- [ ] I understand why triplet lies lower (Fermi hole)
- [ ] I can calculate singlet-triplet splitting from exchange integral
- [ ] I know the selection rules for para/orthohelium transitions
- [ ] I can write complete wave functions for excited helium

### Multi-Electron Atoms
- [ ] I can apply aufbau principle for electron configurations
- [ ] I can use Hund's rules to find ground state terms
- [ ] I can calculate $Z_{\text{eff}}$ using Slater's rules
- [ ] I understand periodic trends from quantum mechanics
- [ ] I can explain anomalous configurations (Cr, Cu)

### Hartree-Fock
- [ ] I understand the Fock operator components (J and K)
- [ ] I can describe the SCF procedure
- [ ] I know what correlation energy represents
- [ ] I can apply Koopmans' theorem
- [ ] I see connection between HF and VQE

---

## 6. Key Takeaways from Week 71

1. **Helium is the simplest non-trivial atom** - exact solution impossible, but approximation methods work well.

2. **Perturbation and variational methods complement each other** - both give physical insight into screening and correlation.

3. **Exchange is purely quantum** - it has no classical analog and leads to Hund's rules, magnetism, and chemical bonding.

4. **The periodic table emerges from quantum mechanics** - aufbau principle + Pauli exclusion + Hund's rules explain all of chemistry.

5. **Hartree-Fock captures ~99% of the energy** - but the remaining 1% (correlation) is crucial for chemistry.

6. **Quantum computing builds on classical foundations** - VQE uses HF as a starting point and captures correlation quantum mechanically.

---

## 7. Preview: Upcoming Topics

### Week 72: Molecules and Chemical Bonding

- Hydrogen molecule ion H₂⁺
- LCAO-MO theory
- Covalent bonding from quantum mechanics
- Molecular orbitals and electron configurations
- Hybridization and molecular geometry

### Later in Month 18

- Born-Oppenheimer approximation
- Potential energy surfaces
- Vibrational and rotational spectra
- Introduction to quantum chemistry software

### Connection to Quantum Computing

- Molecular simulation on quantum computers
- Ground and excited state algorithms
- Reaction pathway calculation
- Materials discovery applications

---

## 8. Reflection Questions

1. **Why is helium harder than hydrogen?** What specifically makes the electron-electron term problematic?

2. **How do perturbation and variational methods relate?** Both give $Z_{\text{eff}} = 27/16$—is this a coincidence?

3. **What's the physical meaning of exchange?** Why does it stabilize parallel spins?

4. **How does Hartree-Fock miss correlation?** Give a physical picture of dynamic vs static correlation.

5. **Why is VQE promising for chemistry?** What can quantum computers do that classical computers cannot (efficiently)?

---

## References for Further Study

### Textbooks
1. Griffiths & Schroeter, *Introduction to Quantum Mechanics*, Ch. 5, 7
2. Sakurai & Napolitano, *Modern Quantum Mechanics*, Ch. 8
3. Szabo & Ostlund, *Modern Quantum Chemistry* (comprehensive HF treatment)
4. Levine, *Quantum Chemistry* (applications to molecules)

### Review Articles
1. McArdle et al., "Quantum computational chemistry" Rev. Mod. Phys. (2020)
2. Cao et al., "Quantum chemistry in the age of quantum computing" Chem. Rev. (2019)

### Online Resources
1. NIST Atomic Spectra Database
2. PySCF documentation (quantum chemistry in Python)
3. Qiskit Chemistry tutorials

---

## Computational Lab: Week Summary

```python
"""
Day 497 Computational Lab: Week 71 Summary
Comprehensive comparison of methods for helium and extension to multi-electron atoms.
"""

import numpy as np
import matplotlib.pyplot as plt

HARTREE_TO_EV = 27.211

def comprehensive_helium_comparison():
    """Compare all methods for helium."""

    print("=" * 70)
    print("COMPREHENSIVE COMPARISON: HELIUM GROUND STATE")
    print("=" * 70)

    methods = [
        ('Independent Particle', -4.000, 'Ignore e-e repulsion'),
        ('First-Order Pert.', -2.750, 'Add <1/r12> to zeroth order'),
        ('Variational (Z_eff)', -2.8477, 'Optimize effective charge'),
        ('Hartree-Fock', -2.8617, 'Self-consistent field'),
        ('Hylleraas (3 param)', -2.891, 'Explicit r12 correlation'),
        ('Exact (limit)', -2.9037, 'Full quantum solution'),
    ]

    E_exact = -2.9037

    print(f"\n{'Method':<25} {'E (Ha)':<12} {'E (eV)':<12} {'Error (eV)':<12}")
    print("-" * 70)

    for name, E, description in methods:
        E_eV = E * HARTREE_TO_EV
        error = (E - E_exact) * HARTREE_TO_EV
        print(f"{name:<25} {E:<12.4f} {E_eV:<12.2f} {error:+<12.3f}")

    print("-" * 70)
    print(f"\nKey Observations:")
    print(f"  1. Independent particle error: {(-4.0 - E_exact)*HARTREE_TO_EV:.1f} eV (huge!)")
    print(f"  2. First-order correction recovers: {((-2.75) - (-4.0))*HARTREE_TO_EV:.1f} eV")
    print(f"  3. Variational adds: {((-2.8477) - (-2.75))*HARTREE_TO_EV:.2f} eV")
    print(f"  4. HF correlation energy: {(-2.9037 - (-2.8617))*HARTREE_TO_EV:.2f} eV")

def plot_method_comparison():
    """Visualize accuracy of different methods."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Methods and energies
    methods = ['Indep.\nParticle', 'First-\nOrder', 'Variational\n(Z_eff)', 'Hartree-\nFock', 'Hylleraas\n(3 param)', 'Exact']
    energies = [-4.0, -2.75, -2.8477, -2.8617, -2.891, -2.9037]
    E_exact = -2.9037

    # Bar plot of energies
    ax1 = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(methods)))
    bars = ax1.bar(methods, energies, color=colors, edgecolor='black')
    ax1.axhline(y=E_exact, color='red', linestyle='--', linewidth=2, label='Exact')
    ax1.set_ylabel('Energy (Hartree)', fontsize=12)
    ax1.set_title('Helium Ground State Energy by Method', fontsize=12)
    ax1.legend()
    ax1.set_ylim(-4.5, -2.5)

    # Error plot
    ax2 = axes[1]
    errors = [(E - E_exact) * HARTREE_TO_EV for E in energies[:-1]]
    ax2.bar(methods[:-1], errors, color=colors[:-1], edgecolor='black')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel('Error (eV)', fontsize=12)
    ax2.set_title('Error Relative to Exact Solution', fontsize=12)

    # Add chemical accuracy line
    ax2.axhline(y=0.043, color='green', linestyle=':', linewidth=2, label='Chemical accuracy')
    ax2.axhline(y=-0.043, color='green', linestyle=':', linewidth=2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('week71_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def exchange_summary():
    """Summarize exchange effects."""

    print("\n" + "=" * 70)
    print("EXCHANGE AND SPIN STATE SUMMARY")
    print("=" * 70)

    print("""
    SINGLET vs TRIPLET STATES
    -------------------------
    Configuration: 1s × nl (one electron excited)

    Singlet (S=0):
      - Spin: antisymmetric (|↑↓⟩ - |↓↑⟩)/√2
      - Spatial: symmetric ψ(r1)φ(r2) + φ(r1)ψ(r2)
      - Energy: E = E_0 + J + K (higher)

    Triplet (S=1):
      - Spin: symmetric |↑↑⟩, (|↑↓⟩ + |↓↑⟩)/√2, |↓↓⟩
      - Spatial: antisymmetric ψ(r1)φ(r2) - φ(r1)ψ(r2)
      - Energy: E = E_0 + J - K (lower)
      - Has FERMI HOLE: ψ(r,r) = 0

    WHY TRIPLET IS LOWER:
      Fermi hole → electrons avoid each other → less repulsion → lower E
    """)

    # Exchange splittings
    configs = [
        ('1s2s', 0.022, 0.80),
        ('1s2p', 0.018, 0.64),
        ('1s3s', 0.008, 0.35),
    ]

    print("Exchange splittings in helium:")
    print(f"{'Config':<10} {'K (Ha)':<10} {'2K (eV)':<10} {'Expt (eV)':<10}")
    print("-" * 40)
    for config, K, expt in configs:
        print(f"{config:<10} {K:<10.3f} {2*K*27.2:<10.2f} {expt:<10.2f}")

def periodic_table_summary():
    """Summarize multi-electron atom principles."""

    print("\n" + "=" * 70)
    print("MULTI-ELECTRON ATOMS: THE PERIODIC TABLE")
    print("=" * 70)

    print("""
    AUFBAU PRINCIPLE
    ----------------
    Fill orbitals in order of increasing (n + l):
    1s < 2s < 2p < 3s < 3p < 4s < 3d < 4p < 5s < ...

    For same (n+l): smaller n fills first

    HUND'S RULES (for ground state term)
    ------------------------------------
    1. Maximize S (parallel spins when possible)
    2. Maximize L (consistent with Rule 1)
    3. J = |L-S| if less than half-filled
       J = L+S if more than half-filled

    PHYSICAL BASIS:
    - Rule 1: Exchange stabilization (Fermi hole)
    - Rule 2: Reduced Coulomb repulsion
    - Rule 3: Spin-orbit coupling sign

    SCREENING (Slater's Rules)
    --------------------------
    Z_eff = Z - σ

    Contributions to σ:
    - Same group: 0.35 (0.30 for 1s)
    - (n-1) group: 0.85 (s,p) or 1.00 (d,f)
    - Lower groups: 1.00
    """)

def vqe_summary():
    """Summarize VQE and quantum computing connection."""

    print("\n" + "=" * 70)
    print("QUANTUM COMPUTING CONNECTION: VQE SUMMARY")
    print("=" * 70)

    print("""
    CLASSICAL vs QUANTUM APPROACH
    =============================

    Classical Electronic Structure:
      1. Hartree-Fock: ~99% of energy, misses correlation
      2. Post-HF (CI, CC): Accurate but expensive O(N^6-7)
      3. DFT: Approximate, often good accuracy

    Quantum VQE Approach:
      1. Encode Hamiltonian: H → Pauli strings
      2. Prepare HF reference on qubits
      3. Apply parameterized circuit U(θ)
      4. Measure ⟨H⟩ on quantum computer
      5. Optimize θ classically
      6. Repeat until converged

    ADVANTAGES OF QUANTUM:
      - Polynomial scaling (vs exponential classical)
      - Natural fermionic representation
      - No sign problem
      - Direct multi-reference capability

    CURRENT STATUS (2024-2025):
      - Small molecules demonstrated (H2, LiH, H2O)
      - Error mitigation crucial
      - Approaching chemical accuracy
      - Hardware still limiting factor

    HELIUM ON QUANTUM COMPUTER:
      - Minimal basis: 4 qubits
      - VQE achieves: E ≈ -2.90 Ha
      - Matches best classical: ✓
      - Proof of principle: ✓
    """)

def final_summary():
    """Print final week summary."""

    print("\n" + "=" * 70)
    print("WEEK 71 FINAL SUMMARY")
    print("=" * 70)

    print("""
    WHAT WE LEARNED THIS WEEK
    =========================

    1. HELIUM: The simplest non-trivial atom
       - Electron-electron repulsion prevents exact solution
       - Perturbation theory: 5% error
       - Variational method: 2% error
       - Hartree-Fock: 1.5% error
       - Key insight: Screening (Z_eff = 1.6875)

    2. EXCHANGE: A purely quantum effect
       - No classical analog
       - Leads to singlet-triplet splitting
       - Explains Hund's rules
       - Foundation of magnetism

    3. MULTI-ELECTRON ATOMS: The periodic table
       - Central field approximation
       - Aufbau + Pauli + Hund = chemistry!
       - Slater screening for Z_eff
       - Periodic trends explained

    4. HARTREE-FOCK: The workhorse method
       - Self-consistent field
       - Captures ~99% of energy
       - Misses correlation (1%)
       - Foundation for all advanced methods

    5. QUANTUM COMPUTING: The future
       - VQE builds on HF reference
       - Captures correlation quantum mechanically
       - Polynomial scaling advantage
       - Active area of research

    NEXT: Molecules and chemical bonding!
    """)

def main():
    """Run all summary functions."""

    print("Day 497: Week 71 Review - Many-Body Systems")
    print("=" * 70)

    comprehensive_helium_comparison()
    plot_method_comparison()
    exchange_summary()
    periodic_table_summary()
    vqe_summary()
    final_summary()

if __name__ == "__main__":
    main()
```

---

*"The helium atom, with only two electrons, taught us that the quantum many-body problem is fundamentally hard—but also that systematic approximations can yield remarkable accuracy. From perturbation theory to Hartree-Fock to modern quantum computing, the journey through many-body physics is one of the great intellectual adventures of our time."*
— Walter Kohn

---

**Week 71 Complete.** Next week: Molecular Structure and Chemical Bonding.
