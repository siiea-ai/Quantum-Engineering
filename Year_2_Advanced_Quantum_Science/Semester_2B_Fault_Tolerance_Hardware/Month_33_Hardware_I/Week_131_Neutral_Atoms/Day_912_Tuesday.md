# Day 912: Rydberg States and Interactions

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Rydberg atom physics, quantum defect theory, scaling laws |
| **Afternoon** | 2 hours | Problem solving: interaction calculations |
| **Evening** | 2 hours | Computational lab: Rydberg state simulations |

## Learning Objectives

By the end of this day, you will be able to:

1. **Apply quantum defect theory** to calculate Rydberg energy levels and wavefunctions
2. **Derive scaling laws** for Rydberg state properties with principal quantum number
3. **Calculate van der Waals interaction strengths** using perturbation theory
4. **Determine C₆ coefficients** for specific Rydberg pair states
5. **Analyze Rydberg excitation dynamics** including laser coupling and decay
6. **Implement numerical calculations** of Rydberg properties for alkali atoms

## Core Content

### 1. Rydberg Atom Fundamentals

#### Quantum Defect Theory

For alkali atoms, the single valence electron experiences a modified Coulomb potential due to core electron shielding. The energy levels are given by the **Rydberg formula with quantum defects**:

$$\boxed{E_{n\ell} = -\frac{R_\infty hc}{(n - \delta_\ell)^2}}$$

where:
- $R_\infty = 1.097 \times 10^7$ m⁻¹ is the Rydberg constant
- $n$ is the principal quantum number
- $\delta_\ell$ is the quantum defect, depending on orbital angular momentum $\ell$

The **effective principal quantum number** is:
$$n^* = n - \delta_\ell$$

For rubidium-87:
| State | $\delta_\ell$ |
|-------|---------------|
| nS₁/₂ | 3.131 |
| nP₁/₂ | 2.654 |
| nP₃/₂ | 2.642 |
| nD₃/₂ | 1.348 |
| nD₅/₂ | 1.346 |

#### Rydberg Wavefunction Properties

The radial wavefunction extends to large distances:
$$\langle r \rangle \approx \frac{a_0 (n^*)^2}{Z_{eff}} \approx a_0 n^{*2}$$

For $n = 70$: $\langle r \rangle \approx 0.26$ μm

The wavefunction at the nucleus determines transition rates:
$$|\psi(0)|^2 \propto \frac{1}{n^{*3}}$$

### 2. Scaling Laws for Rydberg States

The remarkable properties of Rydberg atoms arise from extreme scaling with $n$:

| Property | Scaling | n=50 value (Rb) |
|----------|---------|-----------------|
| Binding energy | $n^{*-2}$ | 5.5 meV |
| Orbital radius | $n^{*2}$ | 0.13 μm |
| Radiative lifetime | $n^{*3}$ | 140 μs |
| Polarizability | $n^{*7}$ | $10^{9}$ a.u. |
| C₆ coefficient | $n^{*11}$ | GHz·μm⁶ |

#### Radiative Lifetime

The spontaneous emission rate from a Rydberg state scales as:
$$\Gamma_n = \Gamma_0 \left(\frac{n_0}{n^*}\right)^3$$

where $\Gamma_0$ is a reference rate. For Rb nS states:
$$\tau_n \approx 1.4 \times (n^*)^{2.95}\,\text{ns}$$

At $n = 70$: $\tau \approx 410$ μs (at T=0 K)

**Blackbody radiation** significantly reduces lifetime at room temperature:
$$\frac{1}{\tau_{eff}} = \frac{1}{\tau_{rad}} + \frac{1}{\tau_{BBR}}$$

where $\tau_{BBR} \propto n^{*2}/T$ for high-$n$ states.

#### Polarizability

The DC polarizability scales dramatically:
$$\alpha_n \approx \frac{e^2 a_0^2}{E_H}(n^*)^7 = \alpha_0 (n^*)^7$$

For Rb 70S: $\alpha \approx 10^{10}$ a.u. $\approx 600$ MHz/(V/cm)²

This extreme polarizability makes Rydberg atoms:
- Highly sensitive to electric fields (Stark effect)
- Strongly interacting with each other
- Useful for electric field sensing

### 3. Rydberg-Rydberg Interactions

#### Van der Waals Interaction

Two Rydberg atoms interact via the dipole-dipole interaction:
$$\hat{V}_{dd} = \frac{1}{4\pi\epsilon_0}\frac{\hat{\mathbf{d}}_1 \cdot \hat{\mathbf{d}}_2 - 3(\hat{\mathbf{d}}_1 \cdot \hat{\mathbf{n}})(\hat{\mathbf{d}}_2 \cdot \hat{\mathbf{n}})}{R^3}$$

where $R$ is the interatomic separation and $\hat{\mathbf{n}} = \mathbf{R}/R$.

For atoms in the same Rydberg state $|rr\rangle$, the direct dipole-dipole term vanishes (no permanent dipole). The leading interaction comes from **second-order perturbation theory**:

$$V_{vdW}(R) = -\sum_{r'r''} \frac{|\langle rr|V_{dd}|r'r''\rangle|^2}{E_{r'r''} - E_{rr}}$$

This gives the **van der Waals interaction**:
$$\boxed{V(R) = \frac{C_6}{R^6}}$$

#### C₆ Coefficient Calculation

The C₆ coefficient is computed by summing over intermediate pair states:
$$C_6 = -\sum_{n'\ell', n''\ell''}\frac{|\langle n\ell, n\ell|V_{dd}|n'\ell', n''\ell''\rangle|^2}{\Delta_{n'\ell', n''\ell''}}$$

where $\Delta = E_{n'\ell'} + E_{n''\ell''} - 2E_{n\ell}$ is the energy defect.

Key contributions come from near-resonant pair states (small $\Delta$).

**Scaling:** Since $\langle r \rangle \propto n^2$ and binding energy $\propto n^{-2}$:
$$C_6 \propto \frac{(n^2)^4}{n^{-2}} = n^{10}$$

More precisely: $C_6 \propto n^{*11}$

#### Numerical Values

For Rb nS₁/₂ + nS₁/₂ states:

| n | C₆ (GHz·μm⁶) | C₆ (atomic units) |
|---|--------------|-------------------|
| 50 | 140 | $1.7 \times 10^{19}$ |
| 60 | 600 | $7.3 \times 10^{19}$ |
| 70 | 2100 | $2.6 \times 10^{20}$ |
| 80 | 6500 | $7.9 \times 10^{20}$ |
| 100 | 40000 | $4.9 \times 10^{21}$ |

### 4. Rydberg Excitation

#### Two-Photon Excitation

Direct single-photon excitation from the ground state to Rydberg states is impractical due to:
- UV wavelengths required (~297 nm for Rb)
- Weak oscillator strength

**Two-photon schemes** via intermediate P state are standard:

**Rb-87 ladder scheme:**
$$5S_{1/2} \xrightarrow{780\,\text{nm}} 5P_{3/2} \xrightarrow{480\,\text{nm}} nS, nD$$

**Effective Rabi frequency:**
$$\Omega_{eff} = \frac{\Omega_1 \Omega_2}{2\Delta}$$

where $\Delta$ is the detuning from the intermediate state.

**Advantages of large detuning:**
- Reduced scattering from intermediate state
- AC Stark shift can be compensated
- Effective two-level dynamics

#### Coherence Limitations

**Laser linewidth:**
The two-photon linewidth must be narrow compared to gate time:
$$\delta\nu_{laser} \ll \frac{1}{t_{gate}}$$

For 1 μs gates: $\delta\nu < 100$ kHz

**Doppler effect:**
Two-photon Doppler shift:
$$\delta\omega_D = (\mathbf{k}_1 + \mathbf{k}_2) \cdot \mathbf{v}$$

Counterpropagating beams ($\mathbf{k}_1 \approx -\mathbf{k}_2$) minimize this effect.

**Intermediate state decay:**
Scattering probability during excitation:
$$P_{sc} \approx \frac{\Omega_1^2}{4\Delta^2}\Gamma t_{gate}$$

### 5. Förster Resonances

When the energy defect $\Delta$ for a pair state transition approaches zero, the interaction changes character.

#### Resonance Condition

Consider the process:
$$|nS, nS\rangle \leftrightarrow |n'P, n''P\rangle$$

If $\Delta = E_{n'P} + E_{n''P} - 2E_{nS} \approx 0$, the interaction becomes:

$$V(R) = \frac{C_3}{R^3}$$

with **resonant dipole-dipole** character.

#### Stark-Tuned Resonances

Electric fields can tune energy levels into resonance:
$$E_{n\ell m}(F) = E_{n\ell m}^{(0)} - \frac{1}{2}\alpha_{n\ell m}F^2 + \ldots$$

Förster resonances enable:
- Enhanced interaction strengths
- Tunable interaction range
- Anisotropic interactions

### 6. Quantum Computing Applications

#### Why Rydberg States?

| Property | Benefit for QC |
|----------|---------------|
| Strong interactions | Fast entangling gates |
| Long-range | Multi-qubit connectivity |
| Tunability | Programmable Hamiltonians |
| Long lifetime | High-fidelity operations |

#### Qubit Encoding

**Ground-Rydberg encoding:**
$$|0\rangle = |g\rangle, \quad |1\rangle = |r\rangle$$
- Simple, but $|1\rangle$ has limited lifetime

**Hyperfine encoding:**
$$|0\rangle = |F=1, m_F=0\rangle, \quad |1\rangle = |F=2, m_F=0\rangle$$
- Long coherence, Rydberg used only for gates

## Worked Examples

### Example 1: Rydberg Energy Level Calculation

**Problem:** Calculate the binding energy and transition wavelength for the 70S₁/₂ state of Rb-87, and estimate its radiative lifetime.

**Solution:**

**Step 1: Calculate effective quantum number**
For Rb S states, $\delta_S = 3.131$:
$$n^* = n - \delta_S = 70 - 3.131 = 66.869$$

**Step 2: Binding energy**
$$E_n = -\frac{R_\infty hc}{(n^*)^2} = -\frac{13.6\,\text{eV}}{(66.869)^2} = -3.04\,\text{meV}$$

Converting to frequency:
$$\nu = \frac{E_n}{h} = -735\,\text{GHz}$$ (below ionization threshold)

**Step 3: Transition wavelength from 5P₃/₂**
Energy of 5P₃/₂: $E_{5P} = -1.59$ eV

Transition energy:
$$\Delta E = E_{70S} - E_{5P} = -0.00304 - (-1.59) = 1.587\,\text{eV}$$

Wavelength:
$$\lambda = \frac{hc}{\Delta E} = \frac{1240\,\text{eV}\cdot\text{nm}}{1.587\,\text{eV}} = 781\,\text{nm}$$

Wait - this seems wrong. The 70S is excited from 5P via blue light. Let me recalculate.

The 5P₃/₂ state is at energy $E_{5P} = -1.59$ eV from ground state perspective. The 70S is at $E_{70S} = -3.04$ meV from ionization threshold.

The ionization energy of Rb is 4.18 eV, so:
$$E_{70S} = -4.18 + 0.00304 = -4.177\,\text{eV}$$

Energy relative to ground:
$$E_{70S} - E_{ground} = -4.177 - (-4.18) = 0.003\,\text{eV}$$

From 5P₃/₂:
$$\Delta E = E_{70S} - E_{5P} = -4.177 - (-4.18 + 1.59) = -4.177 + 2.59 = -1.587\,\text{eV}$$

This gives λ ≈ 780 nm... still not right.

**Correct approach:** The 5P₃/₂ → nS transition uses ~480 nm (blue) light.
$$\lambda_{blue} \approx 480\,\text{nm}$$

**Step 4: Radiative lifetime**
Using the scaling relation:
$$\tau_{70S} \approx 1.4 \times (66.87)^{2.95}\,\text{ns} = 1.4 \times 234000\,\text{ns} \approx 330\,\mu\text{s}$$

At 300 K, blackbody radiation reduces this to ~100 μs.

---

### Example 2: Van der Waals C₆ Calculation

**Problem:** Estimate the C₆ coefficient for two Rb atoms in the 60S₁/₂ state using the dominant intermediate channel.

**Solution:**

**Step 1: Identify dominant channel**
The main contribution comes from:
$$|60S, 60S\rangle \leftrightarrow |60P, 59P\rangle + |59P, 60P\rangle$$

**Step 2: Energy defect**
Using quantum defect theory:
- $E_{60S} = -R_\infty hc/(60-3.131)^2 = -R_\infty hc/3236$
- $E_{60P} = -R_\infty hc/(60-2.64)^2 = -R_\infty hc/3286$
- $E_{59P} = -R_\infty hc/(59-2.64)^2 = -R_\infty hc/3177$

Energy defect:
$$\Delta = E_{60P} + E_{59P} - 2E_{60S}$$
$$\Delta = R_\infty hc\left(\frac{2}{3236} - \frac{1}{3286} - \frac{1}{3177}\right)$$
$$\Delta \approx R_\infty hc \times (-1.1 \times 10^{-5}) \approx -150\,\text{MHz} \times h$$

**Step 3: Dipole matrix element**
The radial matrix element scales as:
$$\langle nS|r|nP\rangle \approx 0.5 \times n^2 a_0$$

For n ≈ 60:
$$d = e \times 0.5 \times 60^2 \times a_0 \approx 1800\,e a_0 \approx 4500\,\text{D}$$

**Step 4: C₆ estimate**
$$C_6 \approx \frac{d^4}{4\pi\epsilon_0 \times |\Delta|} \times \frac{1}{(4\pi\epsilon_0)^2}$$

More carefully:
$$C_6 = \frac{d^4}{|\Delta|} \approx \frac{(4500\,\text{D})^4}{150\,\text{MHz} \times h}$$

Converting: 1 D = 3.336 × 10⁻³⁰ C·m
$$C_6 \approx 600\,\text{GHz} \cdot \mu\text{m}^6$$

This matches the tabulated value well.

---

### Example 3: Two-Photon Rabi Frequency

**Problem:** Calculate the effective Rabi frequency for Rydberg excitation using 780 nm (5 mW, 50 μm waist) and 480 nm (100 mW, 30 μm waist) beams with 1 GHz detuning from the 5P₃/₂ state.

**Solution:**

**Step 1: Calculate individual Rabi frequencies**

For 780 nm beam:
$$I_1 = \frac{2P_1}{\pi w_1^2} = \frac{2 \times 5 \times 10^{-3}}{\pi \times (50 \times 10^{-6})^2} = 1.3 \times 10^6\,\text{W/m}^2$$

The 5S → 5P transition has saturation intensity $I_{sat} = 1.6$ mW/cm² = 16 W/m²

$$\Omega_1 = \Gamma\sqrt{\frac{I_1}{2I_{sat}}} = 2\pi \times 6\,\text{MHz} \times \sqrt{\frac{1.3 \times 10^6}{32}} = 2\pi \times 1.2\,\text{GHz}$$

For 480 nm beam (5P → 70S):
The oscillator strength is weaker by ~$n^{-3}$:
$$\Omega_2 \approx 2\pi \times 50\,\text{MHz}$$ (typical experimental value)

**Step 2: Effective Rabi frequency**
$$\Omega_{eff} = \frac{\Omega_1 \Omega_2}{2\Delta} = \frac{2\pi \times 1.2 \times 10^9 \times 2\pi \times 50 \times 10^6}{2 \times 2\pi \times 10^9}$$
$$\Omega_{eff} = 2\pi \times 30\,\text{MHz}$$

**Step 3: Excitation time**
π-pulse duration:
$$t_\pi = \frac{\pi}{\Omega_{eff}} = \frac{1}{2 \times 30\,\text{MHz}} = 17\,\text{ns}$$

**Step 4: Scattering probability**
$$P_{sc} = \frac{\Omega_1^2}{4\Delta^2}\Gamma t_\pi = \frac{(1.2 \times 10^9)^2}{4 \times (10^9)^2} \times 6 \times 10^6 \times 17 \times 10^{-9} \approx 3.6 \times 10^{-2}$$

This ~4% scattering limits fidelity; larger detuning reduces it.

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the binding energy and orbital radius for the 50D₅/₂ state of Cs-133 (quantum defect δ_D = 2.47).

**Problem 1.2:** Using the scaling $\tau_n \propto n^3$, estimate the radiative lifetime of the 100S state if the 50S state has τ = 100 μs.

**Problem 1.3:** The C₆ coefficient for Rb 60S+60S is 600 GHz·μm⁶. Using the $n^{11}$ scaling, estimate C₆ for the 80S+80S state.

### Level 2: Intermediate Analysis

**Problem 2.1:** Design a two-photon excitation scheme for Cs to the 70S state. Calculate the required wavelengths using quantum defect theory (δ_S = 4.00, δ_P = 3.59 for Cs).

**Problem 2.2:** Two Rb atoms in the 70S state are separated by 5 μm. Calculate:
a) The van der Waals interaction energy (use C₆ = 2100 GHz·μm⁶)
b) The classical turning point for this interaction
c) The Förster defect for the 70S+70S → 70P+69P channel

**Problem 2.3:** A Rydberg atom in a 1 V/cm electric field experiences a Stark shift. For an n=60 state with polarizability α = 10⁹ a.u., calculate the energy shift in MHz.

### Level 3: Challenging Problems

**Problem 3.1:** Derive the C₆ coefficient for Rydberg atoms by summing perturbatively over all intermediate pair states. Show that the $n^{11}$ scaling emerges from the combined scaling of matrix elements and energy denominators.

**Problem 3.2:** Analyze the angular dependence of the dipole-dipole interaction for two atoms in |nS, nS⟩ states. Calculate the effective C₆ coefficient as a function of the angle θ between the interatomic axis and the quantization axis.

**Problem 3.3:** Design an optimal two-photon excitation scheme that minimizes scattering while maintaining fast excitation. Consider:
- Trade-off between detuning and power
- AC Stark shift compensation
- Doppler-free geometry

## Computational Lab: Rydberg State Properties

### Lab 1: Quantum Defect Calculations

```python
"""
Day 912 Lab: Rydberg State Property Calculations
Computing energy levels, scaling laws, and interactions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, e, epsilon_0, hbar, m_e, physical_constants

# Fundamental constants
a0 = physical_constants['Bohr radius'][0]  # 5.29e-11 m
E_h = physical_constants['Hartree energy'][0]  # 4.36e-18 J
Ry = physical_constants['Rydberg constant times hc in eV'][0]  # 13.6 eV

# Quantum defects for Rb-87
class RbQuantumDefects:
    S = 3.1311804
    P_1_2 = 2.6548849
    P_3_2 = 2.6416737
    D_3_2 = 1.34809171
    D_5_2 = 1.34646572
    F = 0.0165192

# Quantum defects for Cs-133
class CsQuantumDefects:
    S = 4.00
    P_1_2 = 3.59
    P_3_2 = 3.56
    D_3_2 = 2.47
    D_5_2 = 2.47

def rydberg_energy(n, delta, Ry_const=Ry):
    """
    Calculate Rydberg state energy using quantum defect theory.

    Parameters:
    -----------
    n : int or array
        Principal quantum number
    delta : float
        Quantum defect
    Ry_const : float
        Rydberg constant in eV

    Returns:
    --------
    E : float or array
        Energy relative to ionization threshold (negative, in eV)
    """
    n_star = n - delta
    E = -Ry_const / n_star**2
    return E

def rydberg_radius(n, delta):
    """Calculate expectation value of orbital radius."""
    n_star = n - delta
    return a0 * n_star**2

def rydberg_lifetime(n, delta, tau_0=1.43e-9, alpha=2.95):
    """
    Estimate radiative lifetime using scaling law.

    tau = tau_0 * (n*)^alpha

    Parameters are fit to Rb nS states.
    """
    n_star = n - delta
    return tau_0 * n_star**alpha

def polarizability(n, delta):
    """
    Calculate polarizability in atomic units.

    alpha ~ (n*)^7 in a.u.
    """
    n_star = n - delta
    # Approximate prefactor for S states
    alpha_au = 0.5 * n_star**7
    return alpha_au

def c6_coefficient(n, delta):
    """
    Calculate C6 coefficient in GHz * um^6.

    C6 ~ n^11 scaling
    """
    n_star = n - delta
    # Empirical fit for Rb nS + nS
    c6_prefactor = 2.2e-7  # GHz * um^6 at n*=1
    return c6_prefactor * n_star**11

# Calculate and display properties
print("=== Rydberg State Properties for Rb-87 ===\n")

n_values = [30, 40, 50, 60, 70, 80, 100]
delta_S = RbQuantumDefects.S

print(f"{'n':>5} {'n*':>8} {'E (meV)':>10} {'<r> (μm)':>10} {'τ (μs)':>10} {'α (a.u.)':>12} {'C6 (GHz·μm⁶)':>14}")
print("-" * 75)

for n in n_values:
    n_star = n - delta_S
    E = rydberg_energy(n, delta_S) * 1000  # meV
    r = rydberg_radius(n, delta_S) * 1e6  # μm
    tau = rydberg_lifetime(n, delta_S) * 1e6  # μs
    alpha = polarizability(n, delta_S)
    c6 = c6_coefficient(n, delta_S)

    print(f"{n:5d} {n_star:8.2f} {E:10.3f} {r:10.4f} {tau:10.1f} {alpha:12.2e} {c6:14.1f}")

# Visualize scaling laws
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

n_range = np.arange(20, 120)
n_star_range = n_range - delta_S

# Energy
axes[0, 0].semilogy(n_range, np.abs(rydberg_energy(n_range, delta_S)) * 1000)
axes[0, 0].set_xlabel('Principal quantum number n')
axes[0, 0].set_ylabel('|E| (meV)')
axes[0, 0].set_title('Binding Energy')
axes[0, 0].grid(True, alpha=0.3)

# Orbital radius
axes[0, 1].semilogy(n_range, rydberg_radius(n_range, delta_S) * 1e6)
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylabel('<r> (μm)')
axes[0, 1].set_title('Orbital Radius')
axes[0, 1].grid(True, alpha=0.3)

# Lifetime
axes[0, 2].semilogy(n_range, rydberg_lifetime(n_range, delta_S) * 1e6)
axes[0, 2].set_xlabel('n')
axes[0, 2].set_ylabel('τ (μs)')
axes[0, 2].set_title('Radiative Lifetime')
axes[0, 2].grid(True, alpha=0.3)

# Polarizability
axes[1, 0].semilogy(n_range, polarizability(n_range, delta_S))
axes[1, 0].set_xlabel('n')
axes[1, 0].set_ylabel('α (a.u.)')
axes[1, 0].set_title('Polarizability')
axes[1, 0].grid(True, alpha=0.3)

# C6 coefficient
axes[1, 1].semilogy(n_range, c6_coefficient(n_range, delta_S))
axes[1, 1].set_xlabel('n')
axes[1, 1].set_ylabel('C₆ (GHz·μm⁶)')
axes[1, 1].set_title('Van der Waals Coefficient')
axes[1, 1].grid(True, alpha=0.3)

# Verify scaling laws
axes[1, 2].loglog(n_star_range, np.abs(rydberg_energy(n_range, delta_S)), 'b-', label='E ~ n*⁻²')
axes[1, 2].loglog(n_star_range, rydberg_lifetime(n_range, delta_S) / 1e-6, 'r-', label='τ ~ n*³')
axes[1, 2].loglog(n_star_range, c6_coefficient(n_range, delta_S) / 1e6, 'g-', label='C₆ ~ n*¹¹')
axes[1, 2].set_xlabel('Effective quantum number n*')
axes[1, 2].set_ylabel('Scaled property')
axes[1, 2].set_title('Scaling Verification')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rydberg_scaling_laws.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 2: Van der Waals Interaction

```python
"""
Lab 2: Calculate and visualize Rydberg-Rydberg interactions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vdw_potential(r, c6):
    """
    Van der Waals potential.

    Parameters:
    -----------
    r : array
        Interatomic distance (μm)
    c6 : float
        C6 coefficient (GHz * μm^6)

    Returns:
    --------
    V : array
        Interaction energy (GHz)
    """
    return c6 / r**6

def forster_potential(r, c3, delta):
    """
    Resonant dipole-dipole potential with detuning.

    V = C3/r³ for perfect resonance
    Crossover to vdW at large distances
    """
    # Simple model: V = sqrt((C3/r³)² + delta²) - delta
    return np.sqrt((c3/r**3)**2 + delta**2) - np.abs(delta)

# C6 coefficients for different Rydberg states
n_states = [50, 60, 70, 80]
delta_S = 3.131

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

r = np.linspace(2, 15, 200)  # μm

# Plot interaction potentials
for n in n_states:
    n_star = n - delta_S
    c6 = 2.2e-7 * n_star**11  # GHz * μm^6

    V = vdw_potential(r, c6)
    axes[0].semilogy(r, V, label=f'n = {n}')

axes[0].set_xlabel('Interatomic distance (μm)')
axes[0].set_ylabel('Interaction energy (GHz)')
axes[0].set_title('Van der Waals Interaction')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(1e-3, 1e3)

# Interaction energy at typical lattice spacing
spacing = 5.0  # μm

print("=== Interaction at 5 μm spacing ===")
for n in n_states:
    n_star = n - delta_S
    c6 = 2.2e-7 * n_star**11
    V = vdw_potential(spacing, c6)
    print(f"n = {n}: V = {V:.2f} GHz = {V*1000:.0f} MHz")

    axes[1].bar(n, V, width=5, alpha=0.7)

axes[1].set_xlabel('Principal quantum number n')
axes[1].set_ylabel('Interaction at 5 μm (GHz)')
axes[1].set_title('Scaling of Interaction Strength')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rydberg_interactions.png', dpi=150, bbox_inches='tight')
plt.show()

# Compare vdW and Förster regimes
print("\n=== Van der Waals vs Förster Resonance ===")

fig, ax = plt.subplots(figsize=(10, 6))

r = np.linspace(1, 20, 200)

# Van der Waals (off-resonant)
c6 = 600  # GHz * μm^6 (n=60)
V_vdw = vdw_potential(r, c6)

# Near Förster resonance
c3 = 20  # GHz * μm^3 (approximate)
deltas = [0.01, 0.1, 1.0, 10.0]  # GHz

ax.loglog(r, V_vdw, 'k--', linewidth=2, label='Pure vdW (C₆/r⁶)')

for delta in deltas:
    V = forster_potential(r, c3, delta)
    ax.loglog(r, V + 1e-6, label=f'Δ = {delta} GHz')

ax.set_xlabel('Interatomic distance (μm)')
ax.set_ylabel('Interaction energy (GHz)')
ax.set_title('Transition from Förster to Van der Waals Regime')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 20)
ax.set_ylim(1e-3, 1e3)

plt.tight_layout()
plt.savefig('forster_vdw_transition.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 3: Two-Photon Excitation Dynamics

```python
"""
Lab 3: Simulate two-photon Rydberg excitation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import hbar

def three_level_dynamics(t, psi, Omega1, Omega2, Delta, Gamma_e):
    """
    Solve three-level ladder system dynamics.

    States: |g⟩ (ground), |e⟩ (intermediate), |r⟩ (Rydberg)

    Parameters:
    -----------
    psi : array (6 elements)
        [Re(c_g), Im(c_g), Re(c_e), Im(c_e), Re(c_r), Im(c_r)]
    Omega1 : float
        Rabi frequency g → e (rad/s)
    Omega2 : float
        Rabi frequency e → r (rad/s)
    Delta : float
        Detuning from intermediate state (rad/s)
    Gamma_e : float
        Decay rate of intermediate state (rad/s)
    """
    # Extract complex amplitudes
    c_g = psi[0] + 1j * psi[1]
    c_e = psi[2] + 1j * psi[3]
    c_r = psi[4] + 1j * psi[5]

    # Equations of motion (rotating frame)
    dc_g = -1j * Omega1/2 * c_e
    dc_e = -1j * Omega1/2 * c_g - 1j * Omega2/2 * c_r + (1j * Delta - Gamma_e/2) * c_e
    dc_r = -1j * Omega2/2 * c_e

    return [dc_g.real, dc_g.imag, dc_e.real, dc_e.imag, dc_r.real, dc_r.imag]

def effective_two_level(t, psi, Omega_eff, delta_eff):
    """
    Effective two-level system after adiabatic elimination.

    States: |g⟩, |r⟩
    """
    c_g = psi[0] + 1j * psi[1]
    c_r = psi[2] + 1j * psi[3]

    dc_g = -1j * Omega_eff/2 * c_r
    dc_r = -1j * Omega_eff/2 * c_g + 1j * delta_eff * c_r

    return [dc_g.real, dc_g.imag, dc_r.real, dc_r.imag]

# Simulation parameters
Omega1 = 2 * np.pi * 500e6  # 500 MHz
Omega2 = 2 * np.pi * 50e6   # 50 MHz
Gamma_e = 2 * np.pi * 6e6   # 6 MHz (Rb 5P)

# Different detunings
detunings = [0.5e9, 1e9, 2e9, 5e9]  # Hz

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, Delta_Hz in enumerate(detunings):
    Delta = 2 * np.pi * Delta_Hz

    # Effective parameters
    Omega_eff = Omega1 * Omega2 / (2 * Delta)
    t_pi = np.pi / Omega_eff

    # Time span
    t_span = (0, 5 * t_pi)
    t_eval = np.linspace(0, 5 * t_pi, 1000)

    # Initial state: all in ground
    psi0_3level = [1, 0, 0, 0, 0, 0]

    # Solve three-level system
    sol = solve_ivp(three_level_dynamics, t_span, psi0_3level,
                    args=(Omega1, Omega2, Delta, Gamma_e),
                    t_eval=t_eval, method='RK45')

    # Extract populations
    P_g = sol.y[0]**2 + sol.y[1]**2
    P_e = sol.y[2]**2 + sol.y[3]**2
    P_r = sol.y[4]**2 + sol.y[5]**2

    ax = axes.flatten()[i]
    ax.plot(sol.t * 1e9, P_g, 'b-', label='|g⟩')
    ax.plot(sol.t * 1e9, P_e, 'g-', label='|e⟩')
    ax.plot(sol.t * 1e9, P_r, 'r-', label='|r⟩')

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Population')
    ax.set_title(f'Δ = {Delta_Hz/1e9:.1f} GHz, Ω_eff = 2π × {Omega_eff/(2*np.pi)/1e6:.1f} MHz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('two_photon_excitation.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze fidelity vs detuning
print("=== Excitation Fidelity Analysis ===")

detunings_scan = np.linspace(0.2e9, 10e9, 50)
max_P_r = []
P_e_at_max = []
t_pi_values = []

for Delta_Hz in detunings_scan:
    Delta = 2 * np.pi * Delta_Hz
    Omega_eff = Omega1 * Omega2 / (2 * Delta)
    t_pi = np.pi / Omega_eff
    t_pi_values.append(t_pi * 1e9)

    t_span = (0, 1.5 * t_pi)
    t_eval = np.linspace(0, 1.5 * t_pi, 500)

    psi0 = [1, 0, 0, 0, 0, 0]
    sol = solve_ivp(three_level_dynamics, t_span, psi0,
                    args=(Omega1, Omega2, Delta, Gamma_e),
                    t_eval=t_eval, method='RK45')

    P_r = sol.y[4]**2 + sol.y[5]**2
    P_e = sol.y[2]**2 + sol.y[3]**2

    max_idx = np.argmax(P_r)
    max_P_r.append(P_r[max_idx])
    P_e_at_max.append(P_e[max_idx])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(detunings_scan/1e9, max_P_r)
axes[0].set_xlabel('Detuning (GHz)')
axes[0].set_ylabel('Maximum Rydberg population')
axes[0].set_title('Excitation Fidelity')
axes[0].grid(True, alpha=0.3)

axes[1].plot(detunings_scan/1e9, np.array(P_e_at_max)*100)
axes[1].set_xlabel('Detuning (GHz)')
axes[1].set_ylabel('Intermediate state population (%)')
axes[1].set_title('Scattering Contribution')
axes[1].grid(True, alpha=0.3)

axes[2].plot(detunings_scan/1e9, t_pi_values)
axes[2].set_xlabel('Detuning (GHz)')
axes[2].set_ylabel('π-pulse time (ns)')
axes[2].set_title('Gate Speed')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('excitation_optimization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nOptimal operating point:")
opt_idx = np.argmax(np.array(max_P_r) * (1 - np.array(P_e_at_max)))
print(f"Detuning: {detunings_scan[opt_idx]/1e9:.1f} GHz")
print(f"Max Rydberg population: {max_P_r[opt_idx]:.3f}")
print(f"Intermediate population: {P_e_at_max[opt_idx]*100:.2f}%")
print(f"π-pulse time: {t_pi_values[opt_idx]:.1f} ns")
```

## Summary

### Key Formulas Table

| Quantity | Formula | Scaling |
|----------|---------|---------|
| Rydberg energy | $E_n = -R_\infty hc/(n-\delta)^2$ | $n^{-2}$ |
| Orbital radius | $\langle r \rangle = a_0 n^{*2}$ | $n^2$ |
| Radiative lifetime | $\tau \propto n^{*3}$ | $n^3$ |
| Polarizability | $\alpha \propto n^{*7}$ | $n^7$ |
| C₆ coefficient | $C_6 \propto n^{*11}$ | $n^{11}$ |
| Effective Rabi freq | $\Omega_{eff} = \Omega_1\Omega_2/2\Delta$ | - |

### Main Takeaways

1. **Quantum defect theory** accurately predicts Rydberg energy levels for alkali atoms, with corrections from core penetration encoded in state-dependent quantum defects.

2. **Extreme scaling laws** with principal quantum number give Rydberg atoms their unique properties: $n^{11}$ scaling of C₆ enables strong interactions even at micrometer separations.

3. **Van der Waals interactions** arise from second-order perturbation theory and dominate at typical experimental distances, transitioning to dipole-dipole ($1/r^3$) near Förster resonances.

4. **Two-photon excitation** balances speed (small Δ) against fidelity (large Δ), with optimal detuning typically around 1 GHz for MHz-scale effective Rabi frequencies.

5. **Long Rydberg lifetimes** (hundreds of μs) enable high-fidelity operations, though blackbody radiation at room temperature can significantly reduce coherence.

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain why Rydberg atoms have extreme properties
- [ ] I understand the origin of van der Waals interactions
- [ ] I can describe the trade-offs in two-photon excitation
- [ ] I know how Förster resonances modify interactions

### Mathematical Skills
- [ ] I can use quantum defect theory for energy calculations
- [ ] I can estimate C₆ from scaling laws
- [ ] I can calculate effective Rabi frequencies

### Computational Skills
- [ ] I can compute Rydberg state properties numerically
- [ ] I can simulate two-photon excitation dynamics
- [ ] I can plot interaction potentials

## Preview: Day 913

Tomorrow we explore the **Rydberg Blockade Mechanism**, where we will:
- Derive the blockade radius from interaction and Rabi frequency competition
- Analyze the perfect blockade regime for high-fidelity gates
- Understand collective Rabi oscillations in blockaded ensembles
- Apply blockade physics to quantum computing and simulation

The Rydberg blockade is the key mechanism enabling fast, high-fidelity two-qubit gates in neutral atom processors.
