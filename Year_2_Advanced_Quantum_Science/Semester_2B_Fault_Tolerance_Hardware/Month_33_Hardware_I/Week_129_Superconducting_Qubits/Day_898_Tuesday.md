# Day 898: Transmon Qubit Design

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Transmon Hamiltonian, charge insensitivity, energy levels |
| Afternoon | 2 hours | Anharmonicity, design optimization, problem solving |
| Evening | 2 hours | Computational lab: Transmon spectrum simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive** the transmon Hamiltonian and its energy spectrum
2. **Explain** how the $E_J/E_C$ ratio determines charge sensitivity
3. **Calculate** qubit frequency and anharmonicity from circuit parameters
4. **Design** a transmon with specified frequency and anharmonicity
5. **Analyze** the tradeoff between charge noise immunity and anharmonicity
6. **Compare** transmon to charge qubit and Cooper pair box regimes

## Core Content

### 1. From Cooper Pair Box to Transmon

The Cooper pair box (CPB) Hamiltonian is:

$$\hat{H}_{CPB} = 4E_C(\hat{n} - n_g)^2 - E_J\cos\hat{\varphi}$$

where:
- $E_C = e^2/2C_\Sigma$ is the charging energy (total capacitance $C_\Sigma$)
- $E_J = \Phi_0 I_c/2\pi$ is the Josephson energy
- $\hat{n}$ is the Cooper pair number operator
- $n_g = C_g V_g/2e$ is the gate-induced charge offset

The **transmon** is a CPB operated at large $E_J/E_C \gg 1$, achieved by adding a large shunt capacitance $C_B \gg C_J$.

### 2. Charge Basis Representation

In the charge basis $|n\rangle$, where $\hat{n}|n\rangle = n|n\rangle$:

$$\hat{H} = \sum_n 4E_C(n - n_g)^2 |n\rangle\langle n| - \frac{E_J}{2}\sum_n (|n\rangle\langle n+1| + |n+1\rangle\langle n|)$$

The Josephson term couples adjacent charge states because $e^{\pm i\hat{\varphi}}|n\rangle = |n \mp 1\rangle$.

**Matrix form** (truncated to $n = -N, ..., N$):

$$H_{mn} = 4E_C(m - n_g)^2\delta_{mn} - \frac{E_J}{2}(\delta_{m,n+1} + \delta_{m,n-1})$$

### 3. Energy Spectrum

Diagonalizing the Hamiltonian gives energy levels $E_m$. The key results:

**Qubit frequency** (transition from ground to first excited state):

$$\boxed{\omega_{01} = (E_1 - E_0)/\hbar \approx \sqrt{8E_JE_C}/\hbar - E_C/\hbar}$$

**Anharmonicity** (deviation from harmonic spacing):

$$\boxed{\alpha = \omega_{12} - \omega_{01} \approx -E_C/\hbar}$$

The negative anharmonicity means $\omega_{12} < \omega_{01}$—higher transitions require less energy.

### 4. Charge Dispersion and Noise Sensitivity

The **charge dispersion** quantifies sensitivity to offset charge $n_g$:

$$\epsilon_m = E_m(n_g = 1/2) - E_m(n_g = 0)$$

For the transmon, the charge dispersion is exponentially suppressed:

$$\boxed{\epsilon_m \approx (-1)^m E_C \frac{2^{4m+5}}{m!}\sqrt{\frac{2}{\pi}}\left(\frac{E_J}{2E_C}\right)^{\frac{m}{2}+\frac{3}{4}} e^{-\sqrt{8E_J/E_C}}}$$

For $E_J/E_C = 50$: $\epsilon_0/E_C \approx 10^{-8}$!

This exponential suppression is the key advantage of the transmon—it makes the qubit frequency nearly independent of fluctuating offset charges.

### 5. Design Tradeoff

The fundamental tradeoff in transmon design:

| Larger $E_J/E_C$ | Effect |
|------------------|--------|
| Pro | Exponentially smaller charge noise sensitivity |
| Con | Linearly smaller anharmonicity |

Typical design point: $E_J/E_C \approx 30-80$

At $E_J/E_C = 50$:
- Charge dispersion: $\epsilon_0/h \sim 10$ Hz (negligible)
- Anharmonicity: $\alpha/2\pi \approx -200$ to $-300$ MHz

### 6. Frequency and Parameter Relations

The total capacitance sets the charging energy:
$$E_C = \frac{e^2}{2C_\Sigma} = \frac{e^2}{2(C_J + C_B + C_g)}$$

For typical transmons: $E_C/h \approx 200-300$ MHz, $E_J/h \approx 10-20$ GHz.

**Design equations**:

Given target frequency $\omega_{01}$ and anharmonicity $\alpha$:

$$E_C \approx -\hbar\alpha$$

$$E_J \approx \frac{(\hbar\omega_{01} + E_C)^2}{8E_C}$$

### 7. Junction and Capacitor Fabrication

**Josephson junction**: Al/AlOx/Al tunnel junction
- Critical current: $I_c \sim 20-50$ nA
- Junction capacitance: $C_J \sim 2-5$ fF
- Made by shadow evaporation (Dolan bridge or bridge-free)

**Shunt capacitor**: Interdigitated or parallel plate
- Capacitance: $C_B \sim 50-100$ fF
- Must be low-loss (use crystalline substrate)

### 8. Coupling to Resonators

The transmon couples to resonators via capacitive coupling:

$$\hat{H}_{int} = 2e\beta\hat{n}\hat{V}_r = 2e\beta\hat{n} \cdot V_{zp}(\hat{a} + \hat{a}^\dagger)$$

In the two-level approximation with matrix element $\langle 0|\hat{n}|1\rangle$:

$$g = 2e\beta V_{zp}|\langle 0|\hat{n}|1\rangle|/\hbar$$

For the transmon:
$$|\langle 0|\hat{n}|1\rangle| \approx \left(\frac{E_J}{8E_C}\right)^{1/4}/\sqrt{2}$$

Leading to coupling strength:
$$g/2\pi \approx 50-200 \text{ MHz}$$

### 9. Tunable Transmons

**Split transmon (SQUID-based)**:

Replace single junction with SQUID (two junctions in parallel):

$$E_J(\Phi_{ext}) = E_{J,\Sigma}\left|\cos\left(\pi\frac{\Phi_{ext}}{\Phi_0}\right)\right|\sqrt{1 + d^2\tan^2\left(\pi\frac{\Phi_{ext}}{\Phi_0}\right)}$$

where $E_{J,\Sigma} = E_{J1} + E_{J2}$ and $d = (E_{J1} - E_{J2})/E_{J,\Sigma}$ is the asymmetry.

For symmetric SQUID ($d = 0$):
$$E_J(\Phi_{ext}) = E_{J,\Sigma}\left|\cos\left(\pi\frac{\Phi_{ext}}{\Phi_0}\right)\right|$$

This allows **flux-tunable frequency** from 0 to maximum.

### 10. Fixed-Frequency vs Tunable Transmons

| Aspect | Fixed Frequency | Tunable |
|--------|-----------------|---------|
| Frequency control | None (set by fabrication) | Magnetic flux |
| Flux noise sensitivity | None | Significant at sweet spot edges |
| Two-qubit gates | Cross-resonance, microwave-activated | Fast flux pulses, parametric |
| Coherence | Generally better | Reduced by flux noise |
| Complexity | Simpler | Requires flux lines |

IBM primarily uses fixed-frequency transmons with cross-resonance gates.
Google uses tunable transmons with CZ gates.

## Quantum Computing Applications

### High-Fidelity Qubit Operations

The transmon's well-defined anharmonicity enables:

1. **Selective addressing**: Drive at $\omega_{01}$ without exciting $|1\rangle \to |2\rangle$
2. **Fast gates**: Large anharmonicity ($|\alpha| \gg$ pulse bandwidth) allows short pulses
3. **DRAG correction**: Compensate leakage to $|2\rangle$ state

### Scalability Considerations

- **Frequency crowding**: Many qubits require careful frequency allocation
- **Crosstalk**: Capacitive coupling creates always-on ZZ interaction
- **Reproducibility**: Junction fabrication has ~5% variation in $I_c$

### Current State of the Art (2025)

- Qubit frequencies: 4-6 GHz (avoiding 4.8 GHz TLS defects)
- Anharmonicity: 200-350 MHz
- $T_1$: 100-500 $\mu$s (material-limited)
- Gate fidelities: >99.9% (single-qubit), >99.5% (two-qubit)

## Worked Examples

### Example 1: Transmon Parameter Extraction

**Problem**: A transmon has measured transitions $\omega_{01}/2\pi = 5.2$ GHz and $\omega_{12}/2\pi = 4.9$ GHz. Determine $E_C$ and $E_J$.

**Solution**:

Calculate anharmonicity:
$$\alpha = \omega_{12} - \omega_{01} = 2\pi(4.9 - 5.2) \text{ GHz} = -2\pi \times 0.3 \text{ GHz}$$

From $\alpha \approx -E_C/\hbar$:
$$E_C/h = 0.3 \text{ GHz} = 300 \text{ MHz}$$

From the qubit frequency formula:
$$\hbar\omega_{01} = \sqrt{8E_JE_C} - E_C$$
$$h \times 5.2 \text{ GHz} = \sqrt{8E_J \times h \times 0.3 \text{ GHz}} - h \times 0.3 \text{ GHz}$$
$$5.5 \text{ GHz} = \sqrt{8E_J/h \times 0.3 \text{ GHz}}$$
$$E_J/h = \frac{(5.5)^2}{8 \times 0.3} = \frac{30.25}{2.4} = 12.6 \text{ GHz}$$

Check the ratio: $E_J/E_C = 12.6/0.3 = 42$ (good transmon regime).

### Example 2: Charge Dispersion Comparison

**Problem**: Compare the charge dispersion of a charge qubit ($E_J/E_C = 0.5$) with a transmon ($E_J/E_C = 50$).

**Solution**:

For charge qubit at $n_g = 0$ vs $n_g = 0.5$, we need numerical diagonalization. The approximate charge dispersion for small $E_J/E_C$ is $\epsilon_0 \approx E_J$.

For $E_J/E_C = 0.5$ with $E_C/h = 5$ GHz:
$$E_J/h = 2.5 \text{ GHz}$$
$$\epsilon_0/h \approx 2.5 \text{ GHz}$$

For transmon at $E_J/E_C = 50$:
$$\epsilon_0/E_C \approx 32\sqrt{\frac{2}{\pi}}\left(\frac{50}{2}\right)^{3/4} e^{-\sqrt{8 \times 50}}$$
$$= 32 \times 0.8 \times 9.5 \times e^{-20} = 243 \times 2 \times 10^{-9} \approx 5 \times 10^{-7}$$

With $E_C/h = 0.1$ GHz (to keep similar qubit frequency):
$$\epsilon_0/h \approx 50 \text{ Hz}$$

**Ratio**: $2.5 \text{ GHz} / 50 \text{ Hz} = 5 \times 10^7$

The transmon is **50 million times less sensitive** to charge noise!

### Example 3: Tunable Transmon Frequency Range

**Problem**: A symmetric split-junction transmon has $E_{J,max}/h = 20$ GHz and $E_C/h = 250$ MHz. Calculate the frequency range as flux varies from 0 to $\Phi_0/2$.

**Solution**:

At $\Phi_{ext} = 0$:
$$E_J = E_{J,max} = 20 \text{ GHz}$$
$$\omega_{01}/2\pi = \sqrt{8 \times 20 \times 0.25} - 0.25 = \sqrt{40} - 0.25 = 6.32 - 0.25 = 6.07 \text{ GHz}$$

At $\Phi_{ext} = \Phi_0/4$:
$$E_J = E_{J,max}\cos(\pi/4) = 20 \times 0.707 = 14.14 \text{ GHz}$$
$$\omega_{01}/2\pi = \sqrt{8 \times 14.14 \times 0.25} - 0.25 = \sqrt{28.3} - 0.25 = 5.32 - 0.25 = 5.07 \text{ GHz}$$

At $\Phi_{ext} = \Phi_0/2$:
$$E_J = E_{J,max}\cos(\pi/2) = 0$$

The qubit frequency drops to zero (avoided crossing with plasma frequency).

**Practical tuning range**: ~5.1 - 6.1 GHz (avoiding low-frequency regime).

## Practice Problems

### Level 1: Direct Application

1. A transmon has $E_C/h = 280$ MHz and $E_J/h = 15$ GHz. Calculate:
   (a) The $E_J/E_C$ ratio
   (b) The qubit frequency $\omega_{01}/2\pi$
   (c) The anharmonicity $\alpha/2\pi$

2. Design a transmon with frequency 5.0 GHz and anharmonicity -250 MHz. What are the required $E_C$ and $E_J$?

3. A shunt capacitance of 70 fF is added to a junction with capacitance 5 fF. What is the charging energy $E_C$?

### Level 2: Intermediate

4. For a transmon with $E_J/E_C = 60$, calculate the charge dispersion ratio $\epsilon_0/E_C$ and estimate the frequency shift caused by a random offset charge of $0.1e$.

5. A split-junction transmon has $E_{J1}/h = 12$ GHz and $E_{J2}/h = 8$ GHz with $E_C/h = 220$ MHz. Calculate:
   (a) The maximum and minimum qubit frequencies
   (b) The junction asymmetry $d$
   (c) The qubit frequency at $\Phi_{ext} = 0.3\Phi_0$

6. Derive the matrix element $\langle 0|\hat{n}|1\rangle$ for a transmon in the limit $E_J \gg E_C$ by treating the cosine potential as approximately harmonic near the bottom.

### Level 3: Challenging

7. **Thermal population**: At $T = 20$ mK, what fraction of a 5 GHz transmon population is in the excited state? How does this compare to the error threshold for fault-tolerant quantum computing?

8. **Higher levels**: Including the $|2\rangle$ state, calculate the frequencies $\omega_{01}$ and $\omega_{12}$ to second order in $E_C/E_J$ for a transmon. Compare with the approximate formulas.

9. **Capacitor design**: Design an interdigitated capacitor to achieve $C_B = 65$ fF using aluminum fingers on silicon ($\epsilon_r = 11.7$) with 2 $\mu$m finger width and 2 $\mu$m gap. How many finger pairs are needed?

## Computational Lab: Transmon Spectrum Simulation

```python
"""
Day 898 Computational Lab: Transmon Qubit Design
Simulating transmon energy levels, charge dispersion, and design optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

# Physical constants
h = 6.626e-34  # Planck constant
hbar = h / (2 * np.pi)
e = 1.602e-19  # electron charge

# =============================================================================
# Part 1: Transmon Hamiltonian in Charge Basis
# =============================================================================

def transmon_hamiltonian(EC, EJ, ng=0, nmax=30):
    """
    Construct transmon Hamiltonian in charge basis.

    Parameters:
    -----------
    EC : float
        Charging energy (in frequency units, e.g., GHz)
    EJ : float
        Josephson energy (in frequency units)
    ng : float
        Offset charge in units of 2e
    nmax : int
        Maximum charge number to include (basis: -nmax to +nmax)

    Returns:
    --------
    H : ndarray
        Hamiltonian matrix
    """
    dim = 2 * nmax + 1
    n_values = np.arange(-nmax, nmax + 1)

    # Charging energy (diagonal)
    H = np.diag(4 * EC * (n_values - ng)**2)

    # Josephson energy (off-diagonal, couples n to n±1)
    for i in range(dim - 1):
        H[i, i+1] = -EJ / 2
        H[i+1, i] = -EJ / 2

    return H

def transmon_spectrum(EC, EJ, ng=0, n_levels=5, nmax=30):
    """
    Calculate transmon energy spectrum.

    Returns:
    --------
    energies : ndarray
        Energy levels (in same units as EC, EJ)
    states : ndarray
        Eigenstates
    """
    H = transmon_hamiltonian(EC, EJ, ng, nmax)
    energies, states = eigh(H)

    # Shift so ground state is at 0
    energies = energies - energies[0]

    return energies[:n_levels], states[:, :n_levels]

# Example: Calculate spectrum for typical transmon
EC = 0.25  # GHz
EJ = 15.0  # GHz

print("=" * 60)
print("Transmon Energy Spectrum")
print("=" * 60)
print(f"EC = {EC} GHz")
print(f"EJ = {EJ} GHz")
print(f"EJ/EC = {EJ/EC:.1f}")

energies, states = transmon_spectrum(EC, EJ, n_levels=6)
print(f"\nEnergy levels (GHz):")
for i, E in enumerate(energies):
    print(f"  E_{i} = {E:.4f} GHz")

# Calculate transition frequencies
print(f"\nTransition frequencies:")
omega_01 = energies[1] - energies[0]
omega_12 = energies[2] - energies[1]
omega_23 = energies[3] - energies[2]
print(f"  ω_01/2π = {omega_01:.4f} GHz")
print(f"  ω_12/2π = {omega_12:.4f} GHz")
print(f"  ω_23/2π = {omega_23:.4f} GHz")

# Calculate anharmonicity
alpha = omega_12 - omega_01
print(f"\nAnharmonicity:")
print(f"  α/2π = {alpha*1000:.2f} MHz")
print(f"  Theory (-EC): {-EC*1000:.2f} MHz")

# Theoretical predictions
omega_01_theory = np.sqrt(8 * EJ * EC) - EC
print(f"\nTheoretical ω_01/2π = {omega_01_theory:.4f} GHz")

# =============================================================================
# Part 2: Charge Dispersion
# =============================================================================

def charge_dispersion(EC, EJ, level=0, n_ng=100, nmax=30):
    """
    Calculate energy vs offset charge for a given level.
    """
    ng_values = np.linspace(0, 1, n_ng)
    energies = []

    for ng in ng_values:
        E, _ = transmon_spectrum(EC, EJ, ng, n_levels=level+1, nmax=nmax)
        energies.append(E[level])

    return ng_values, np.array(energies)

# Calculate charge dispersion for different EJ/EC ratios
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Energy vs ng for different levels
ax1 = axes[0, 0]
ng_vals, E0 = charge_dispersion(EC, EJ, level=0)
_, E1 = charge_dispersion(EC, EJ, level=1)
_, E2 = charge_dispersion(EC, EJ, level=2)

ax1.plot(ng_vals, E0, 'b-', label=r'$E_0$', linewidth=2)
ax1.plot(ng_vals, E1, 'r-', label=r'$E_1$', linewidth=2)
ax1.plot(ng_vals, E2, 'g-', label=r'$E_2$', linewidth=2)
ax1.set_xlabel(r'Offset charge $n_g$ (2e)', fontsize=12)
ax1.set_ylabel('Energy (GHz)', fontsize=12)
ax1.set_title(f'Transmon Energy Levels (EJ/EC = {EJ/EC:.0f})', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Compare charge qubit vs transmon
ax2 = axes[0, 1]

# Charge qubit regime
EC_cq = 5.0  # GHz
EJ_cq = 1.0  # GHz
ng_vals, E0_cq = charge_dispersion(EC_cq, EJ_cq, level=0)
_, E1_cq = charge_dispersion(EC_cq, EJ_cq, level=1)

ax2.plot(ng_vals, E0_cq, 'b-', label=r'$E_0$ (CPB)', linewidth=2)
ax2.plot(ng_vals, E1_cq, 'r-', label=r'$E_1$ (CPB)', linewidth=2)
ax2.set_xlabel(r'Offset charge $n_g$ (2e)', fontsize=12)
ax2.set_ylabel('Energy (GHz)', fontsize=12)
ax2.set_title(f'Charge Qubit (EJ/EC = {EJ_cq/EC_cq:.1f})', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# =============================================================================
# Part 3: EJ/EC Dependence
# =============================================================================

# Calculate qubit frequency and anharmonicity vs EJ/EC
EJ_EC_ratios = np.linspace(5, 100, 50)
EC_fixed = 0.3  # GHz

frequencies = []
anharmonicities = []
dispersions = []

for ratio in EJ_EC_ratios:
    EJ_temp = ratio * EC_fixed
    E, _ = transmon_spectrum(EC_fixed, EJ_temp, n_levels=3)
    w01 = E[1] - E[0]
    w12 = E[2] - E[1]
    frequencies.append(w01)
    anharmonicities.append((w12 - w01))

    # Charge dispersion
    _, E0_ng0 = transmon_spectrum(EC_fixed, EJ_temp, ng=0, n_levels=1)
    _, E0_ng05 = transmon_spectrum(EC_fixed, EJ_temp, ng=0.5, n_levels=1)
    dispersions.append(abs(E0_ng05[0] - E0_ng0[0]))

frequencies = np.array(frequencies)
anharmonicities = np.array(anharmonicities)
dispersions = np.array(dispersions)

# Theoretical curves
w01_theory = np.sqrt(8 * EJ_EC_ratios * EC_fixed * EC_fixed) - EC_fixed
alpha_theory = -EC_fixed * np.ones_like(EJ_EC_ratios)

# Plot 3: Frequency and anharmonicity
ax3 = axes[1, 0]
ax3.plot(EJ_EC_ratios, frequencies, 'b-', linewidth=2, label='Numerical ω₀₁')
ax3.plot(EJ_EC_ratios, w01_theory, 'b--', linewidth=2, label='Theory ω₀₁')
ax3.set_xlabel(r'$E_J/E_C$', fontsize=12)
ax3.set_ylabel('Frequency (GHz)', fontsize=12, color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)

ax3b = ax3.twinx()
ax3b.plot(EJ_EC_ratios, anharmonicities * 1000, 'r-', linewidth=2, label='α')
ax3b.plot(EJ_EC_ratios, alpha_theory * 1000, 'r--', linewidth=2, label='Theory α')
ax3b.set_ylabel('Anharmonicity (MHz)', fontsize=12, color='r')
ax3b.tick_params(axis='y', labelcolor='r')
ax3b.legend(loc='upper right', fontsize=10)
ax3.set_title('Transmon Frequency and Anharmonicity', fontsize=14)

# Plot 4: Charge dispersion (log scale)
ax4 = axes[1, 1]
ax4.semilogy(EJ_EC_ratios, dispersions * 1e9, 'b-', linewidth=2, label='Numerical')

# Theoretical charge dispersion
disp_theory = EC_fixed * 32 * np.sqrt(2/np.pi) * (EJ_EC_ratios/2)**(3/4) * \
              np.exp(-np.sqrt(8 * EJ_EC_ratios))
ax4.semilogy(EJ_EC_ratios, disp_theory * 1e9, 'r--', linewidth=2, label='Theory')

ax4.set_xlabel(r'$E_J/E_C$', fontsize=12)
ax4.set_ylabel('Charge dispersion (Hz)', fontsize=12)
ax4.set_title('Exponential Suppression of Charge Noise', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([1e-3, 1e9])

plt.tight_layout()
plt.savefig('transmon_design.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 4: Tunable Transmon (SQUID)
# =============================================================================

def squid_EJ(EJ_sum, d, phi_ext):
    """
    Calculate effective Josephson energy for asymmetric SQUID.

    Parameters:
    -----------
    EJ_sum : float
        Sum of junction energies (EJ1 + EJ2)
    d : float
        Asymmetry (EJ1 - EJ2) / (EJ1 + EJ2)
    phi_ext : float
        External flux in units of Phi_0

    Returns:
    --------
    EJ_eff : float
        Effective Josephson energy
    """
    return EJ_sum * np.abs(np.cos(np.pi * phi_ext)) * \
           np.sqrt(1 + d**2 * np.tan(np.pi * phi_ext)**2)

print("\n" + "=" * 60)
print("Tunable Transmon Analysis")
print("=" * 60)

# Parameters
EJ_sum = 20.0  # GHz
EC_squid = 0.25  # GHz
d = 0.0  # Symmetric SQUID

phi_values = np.linspace(0, 0.5, 100)
EJ_values = [squid_EJ(EJ_sum, d, phi) for phi in phi_values]
freq_values = [np.sqrt(8 * EJ * EC_squid) - EC_squid for EJ in EJ_values]

# Plot tunable transmon
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes2[0]
ax1.plot(phi_values, EJ_values, 'b-', linewidth=2)
ax1.set_xlabel(r'External flux $\Phi_{ext}/\Phi_0$', fontsize=12)
ax1.set_ylabel(r'$E_J$ (GHz)', fontsize=12)
ax1.set_title('SQUID Josephson Energy vs Flux', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2 = axes2[1]
ax2.plot(phi_values, freq_values, 'r-', linewidth=2)
ax2.set_xlabel(r'External flux $\Phi_{ext}/\Phi_0$', fontsize=12)
ax2.set_ylabel('Qubit frequency (GHz)', fontsize=12)
ax2.set_title('Tunable Transmon Frequency', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tunable_transmon.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 5: Design Optimization
# =============================================================================

def design_transmon(target_freq, target_alpha, tolerance=0.01):
    """
    Design transmon parameters for target frequency and anharmonicity.

    Parameters:
    -----------
    target_freq : float
        Target qubit frequency ω₀₁/2π in GHz
    target_alpha : float
        Target anharmonicity α/2π in GHz (negative)
    tolerance : float
        Acceptable fractional error

    Returns:
    --------
    dict with EC, EJ, and achieved values
    """
    # From α ≈ -EC
    EC = -target_alpha

    # From ω₀₁ ≈ √(8EJ*EC) - EC
    # EJ = (ω₀₁ + EC)² / (8*EC)
    EJ = (target_freq + EC)**2 / (8 * EC)

    # Verify with numerical calculation
    E, _ = transmon_spectrum(EC, EJ, n_levels=3)
    actual_freq = E[1] - E[0]
    actual_alpha = (E[2] - E[1]) - (E[1] - E[0])

    return {
        'EC': EC,
        'EJ': EJ,
        'EJ_EC_ratio': EJ / EC,
        'target_freq': target_freq,
        'actual_freq': actual_freq,
        'freq_error': (actual_freq - target_freq) / target_freq,
        'target_alpha': target_alpha,
        'actual_alpha': actual_alpha,
        'alpha_error': (actual_alpha - target_alpha) / abs(target_alpha)
    }

print("\n" + "=" * 60)
print("Transmon Design Tool")
print("=" * 60)

# Design example
target_f = 5.0  # GHz
target_a = -0.25  # GHz

result = design_transmon(target_f, target_a)
print(f"\nDesign target: f = {target_f} GHz, α = {target_a*1000} MHz")
print(f"\nRequired parameters:")
print(f"  EC = {result['EC']*1000:.1f} MHz")
print(f"  EJ = {result['EJ']:.2f} GHz")
print(f"  EJ/EC = {result['EJ_EC_ratio']:.1f}")
print(f"\nAchieved values:")
print(f"  Frequency: {result['actual_freq']:.4f} GHz (error: {result['freq_error']*100:.2f}%)")
print(f"  Anharmonicity: {result['actual_alpha']*1000:.2f} MHz (error: {result['alpha_error']*100:.2f}%)")

# =============================================================================
# Part 6: Wavefunction Visualization
# =============================================================================

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

# Get wavefunctions
EC_vis = 0.25
EJ_vis = 15.0
nmax_vis = 20

H = transmon_hamiltonian(EC_vis, EJ_vis, ng=0, nmax=nmax_vis)
energies_vis, states_vis = eigh(H)
n_values = np.arange(-nmax_vis, nmax_vis + 1)

# Plot first few wavefunctions in charge basis
ax1 = axes3[0]
for i in range(4):
    psi = states_vis[:, i]
    prob = np.abs(psi)**2
    ax1.plot(n_values, prob + energies_vis[i], label=f'|{i}⟩', linewidth=2)
    ax1.fill_between(n_values, energies_vis[i], prob + energies_vis[i], alpha=0.3)

ax1.set_xlabel('Charge number n', fontsize=12)
ax1.set_ylabel('Energy / Probability + offset', fontsize=12)
ax1.set_title('Transmon Wavefunctions in Charge Basis', fontsize=14)
ax1.legend(fontsize=11)
ax1.set_xlim([-10, 10])
ax1.grid(True, alpha=0.3)

# Compare with harmonic oscillator
ax2 = axes3[1]

# Phase basis representation (approximate)
phi_vals = np.linspace(-np.pi, np.pi, 200)
E_potential = -EJ_vis * np.cos(phi_vals)
E_potential = E_potential - E_potential.min()  # Shift to start at 0

ax2.plot(phi_vals, E_potential, 'k-', linewidth=2, label='Potential')

# Add energy level lines
for i in range(5):
    E_level = energies_vis[i] - energies_vis[0]
    ax2.axhline(E_level, color=f'C{i}', linestyle='--', alpha=0.7,
                label=f'E_{i} = {E_level:.2f} GHz')

ax2.set_xlabel(r'Phase $\varphi$', fontsize=12)
ax2.set_ylabel('Energy (GHz)', fontsize=12)
ax2.set_title('Cosine Potential with Energy Levels', fontsize=14)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-np.pi, np.pi])
ax2.set_ylim([0, EJ_vis * 0.5])

plt.tight_layout()
plt.savefig('transmon_wavefunctions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Transmon Hamiltonian | $\hat{H} = 4E_C(\hat{n} - n_g)^2 - E_J\cos\hat{\varphi}$ |
| Qubit frequency | $\omega_{01} \approx \sqrt{8E_JE_C}/\hbar - E_C/\hbar$ |
| Anharmonicity | $\alpha \approx -E_C/\hbar$ |
| Charging energy | $E_C = e^2/2C_\Sigma$ |
| Charge dispersion | $\epsilon_m \propto e^{-\sqrt{8E_J/E_C}}$ |
| SQUID $E_J$ | $E_J(\Phi) = E_{J,\Sigma}|\cos(\pi\Phi/\Phi_0)|$ |

### Main Takeaways

1. **Transmon = CPB at large $E_J/E_C$**: Increasing shunt capacitance reduces charging energy, making qubit frequency insensitive to charge noise

2. **Exponential suppression**: Charge dispersion falls as $e^{-\sqrt{8E_J/E_C}}$, enabling ~$10^{-8}$ sensitivity reduction

3. **Anharmonicity-sensitivity tradeoff**: Larger $E_J/E_C$ means smaller $|\alpha| \approx E_C$, limiting pulse speed

4. **Typical design**: $E_J/E_C \approx 50$, giving $|\alpha|/2\pi \approx 250$ MHz with negligible charge noise

5. **Tunability via SQUID**: Split-junction transmons allow flux-tunable frequency at cost of flux noise sensitivity

## Daily Checklist

- [ ] I can derive energy levels from the transmon Hamiltonian
- [ ] I understand why large $E_J/E_C$ suppresses charge sensitivity
- [ ] I can calculate qubit frequency and anharmonicity from $E_C$ and $E_J$
- [ ] I understand the design tradeoff between charge immunity and anharmonicity
- [ ] I can design a transmon for specified frequency and anharmonicity
- [ ] I have run the computational lab and can interpret the results
- [ ] I understand tunable transmon operation via SQUID

## Preview: Day 899

Tomorrow we explore **flux qubits and fluxonium**—alternative superconducting qubit designs that use magnetic flux as the primary variable:

- Persistent current qubits with double-well potential
- Flux sweet spots for first-order noise immunity
- Fluxonium: superinductors and heavy fluxonium regime
- Comparative advantages for different applications

---

*"The transmon is not particularly clever—it's just a CPB with a bigger capacitor. But that simple change makes all the difference for practical quantum computing."*
