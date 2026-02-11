# Day 899: Flux Qubits and Fluxonium

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Flux qubit theory, persistent current states, double-well potential |
| Afternoon | 2 hours | Fluxonium design, superinductors, problem solving |
| Evening | 2 hours | Computational lab: Flux qubit and fluxonium simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive** the Hamiltonian of persistent current flux qubits
2. **Explain** the double-well potential and tunneling between flux states
3. **Identify** flux sweet spots and their protection against flux noise
4. **Describe** the fluxonium qubit and the role of superinductors
5. **Compare** flux-based qubits with transmon in terms of noise sensitivity
6. **Calculate** qubit frequencies and matrix elements for flux qubits

## Core Content

### 1. The Flux Qubit Concept

While transmons encode information in charge states (Cooper pair number), **flux qubits** encode information in magnetic flux states—specifically, the direction of persistent supercurrents circulating in a loop.

The key insight: A superconducting loop interrupted by Josephson junctions can sustain persistent currents. Near half-integer flux bias, two current directions become nearly degenerate.

### 2. RF-SQUID: Simplest Flux Qubit

Consider a superconducting loop with inductance $L$ interrupted by a single Josephson junction:

$$U(\varphi) = \frac{(\Phi - \Phi_{ext})^2}{2L} - E_J\cos\varphi$$

Using $\Phi = \Phi_0\varphi/2\pi$:

$$U(\varphi) = E_L\left(\varphi - 2\pi\frac{\Phi_{ext}}{\Phi_0}\right)^2 - E_J\cos\varphi$$

where the inductive energy is:

$$E_L = \frac{(\Phi_0/2\pi)^2}{2L} = \frac{\Phi_0^2}{8\pi^2 L}$$

**Potential landscape**:
- When $E_J > E_L$: Double-well potential near $\Phi_{ext} = \Phi_0/2$
- When $E_J < E_L$: Single well (not useful as qubit)

### 3. Persistent Current (PC) Flux Qubit

The most common flux qubit uses a loop with **three Josephson junctions**:

$$\hat{H} = \sum_{i=1}^{3}\left[\frac{\hat{Q}_i^2}{2C_i} - E_{Ji}\cos\hat{\varphi}_i\right] + E_L(\hat{\varphi}_1 + \hat{\varphi}_2 + \hat{\varphi}_3 - 2\pi f)^2$$

where $f = \Phi_{ext}/\Phi_0$ is the reduced external flux.

One junction is made smaller (by factor $\alpha \approx 0.7$):
$$E_{J3} = \alpha E_J, \quad E_{J1} = E_{J2} = E_J$$

**Effective two-level Hamiltonian** near $f = 0.5$:

$$\boxed{\hat{H}_{PC} = \frac{\epsilon}{2}\hat{\sigma}_z + \frac{\Delta}{2}\hat{\sigma}_x}$$

where:
- $\epsilon = 2I_p\Phi_0(f - 0.5)$ is the energy bias
- $I_p$ is the persistent current (~300 nA)
- $\Delta$ is the tunneling amplitude between wells

### 4. Flux Sweet Spot

At the **degeneracy point** ($f = 0.5$, $\epsilon = 0$):

$$\hat{H} = \frac{\Delta}{2}\hat{\sigma}_x$$

The qubit frequency is:

$$\omega_q = \sqrt{\epsilon^2 + \Delta^2}/\hbar$$

At $f = 0.5$: $\omega_q = \Delta/\hbar$

**First-order noise immunity**:

$$\frac{\partial\omega_q}{\partial\epsilon}\bigg|_{\epsilon=0} = 0$$

This is the **flux sweet spot**—first-order insensitivity to flux noise!

However, second-order sensitivity remains:
$$\frac{\partial^2\omega_q}{\partial\epsilon^2}\bigg|_{\epsilon=0} = \frac{1}{\Delta}$$

### 5. Flux Qubit Energy Levels

The persistent current qubit has states $|L\rangle$ (clockwise current) and $|R\rangle$ (counterclockwise).

At the sweet spot, the eigenstates are:

$$|0\rangle = \frac{1}{\sqrt{2}}(|L\rangle + |R\rangle)$$
$$|1\rangle = \frac{1}{\sqrt{2}}(|L\rangle - |R\rangle)$$

Away from sweet spot ($\epsilon \neq 0$):

$$|0\rangle \approx |L\rangle, \quad |1\rangle \approx |R\rangle \text{ (for } \epsilon > 0\text{)}$$

### 6. Fluxonium: Superinductor-Shunted Qubit

The **fluxonium** replaces the large loop inductance with a **superinductor**—a very large inductance ($L \sim 100-500$ nH) typically made from:

- Array of Josephson junctions
- High kinetic inductance materials (granular aluminum, NbN nanowires)

**Fluxonium Hamiltonian**:

$$\boxed{\hat{H}_{flx} = 4E_C\hat{n}^2 - E_J\cos\hat{\varphi} + \frac{1}{2}E_L(\hat{\varphi} - 2\pi f)^2}$$

Three energy scales compete:
- $E_C$: charging energy (~1 GHz)
- $E_J$: Josephson energy (~5-10 GHz)
- $E_L$: inductive energy (~0.5-2 GHz)

### 7. Fluxonium Regimes

**Heavy fluxonium** ($E_J \gg E_L, E_C$):
- Deep cosine wells with small $E_L$ tilt
- Very low qubit frequency at sweet spot (~100-500 MHz)
- Long coherence times due to reduced qubit frequency
- Transitions are "flux-like" between wells

**Light fluxonium** ($E_C \sim E_J$):
- More transmon-like behavior
- Higher frequency, larger anharmonicity
- Good for fast gates

### 8. Fluxonium Sweet Spot

At $f = 0.5$, the double-well becomes symmetric:

The ground state $|0\rangle$ and first excited $|1\rangle$ are symmetric/antisymmetric superpositions of left/right well states:

$$|0\rangle = \frac{1}{\sqrt{2}}(|\varphi_L\rangle + |\varphi_R\rangle)$$
$$|1\rangle = \frac{1}{\sqrt{2}}(|\varphi_L\rangle - |\varphi_R\rangle)$$

The transition frequency is set by inter-well tunneling:
$$\omega_{01} \approx \sqrt{\omega_p\cdot\omega_{01}^{heavy}} \cdot e^{-S/\hbar}$$

where $\omega_p = \sqrt{8E_JE_C}/\hbar$ is the plasma frequency and $S$ is the WKB action.

### 9. Matrix Elements and Transitions

**Flux qubit**: Large matrix element for flux operator

$$\langle 0|\hat{\Phi}|1\rangle = \Phi_0\langle 0|\hat{\varphi}|1\rangle/2\pi \neq 0$$

The transition is flux-dipole allowed, enabling:
- Coupling via mutual inductance to resonators
- Sensitivity to flux noise

**Fluxonium**: Unique selection rules

- At sweet spot: $\langle 0|\hat{\varphi}|1\rangle \approx 0$ (parity-protected)
- Away from sweet spot: $\langle 0|\hat{\varphi}|1\rangle$ finite
- Higher transitions: Large $\langle 1|\hat{\varphi}|2\rangle$ allows alternative readout

### 10. Noise Sensitivity Comparison

| Qubit Type | Charge Noise | Flux Noise | Sweet Spot |
|------------|--------------|------------|------------|
| Transmon | Exponentially suppressed | Via frequency tuning | None (fixed freq) |
| Flux qubit | Moderate | First-order protected | $\Phi = \Phi_0/2$ |
| Fluxonium | Moderate | First-order protected | $\Phi = \Phi_0/2$ |

**Fluxonium advantages**:
- Low frequency → reduced sensitivity to all noise
- Large anharmonicity → no leakage issues
- Sweet spot protection against flux noise

## Quantum Computing Applications

### Flux Qubits in Quantum Annealing

D-Wave quantum annealers use flux qubits:
- Operate away from sweet spot for programmable bias
- Tunneling $\Delta$ controlled by barrier height
- Coupler qubits mediate $\sigma_z\sigma_z$ interactions

### Fluxonium for Long Coherence

Heavy fluxonium has achieved:
- $T_1 > 1$ ms (record for superconducting qubits)
- High-fidelity single-qubit gates (>99.99%)
- Challenges: slow gates due to low frequency

### Hybrid Approaches

**Plasmonium**: Balance between transmon and fluxonium
- Moderate superinductor ($E_L/E_J \sim 0.1$)
- Higher frequency than fluxonium, lower than transmon
- Improved coherence while maintaining fast gates

### Protected Qubits

Flux-based designs enable intrinsic protection:
- **0-π qubit**: Disjoint wavefunctions reduce all transition rates
- **Bifluxon**: Exponentially protected sweet spot

## Worked Examples

### Example 1: Persistent Current Flux Qubit

**Problem**: A 3-junction flux qubit has $E_J/h = 200$ GHz, $\alpha = 0.7$, and loop inductance 50 pH. Calculate:
(a) The persistent current $I_p$
(b) The energy bias at $f = 0.502$
(c) The qubit frequency if $\Delta/h = 5$ GHz

**Solution**:

(a) The persistent current is approximately:
$$I_p = I_c\sqrt{1 - (2\alpha)^2} = I_c\sqrt{1 - 1.96}$$

Wait, this gives imaginary result for $\alpha = 0.7$. Let's use the correct formula:
$$I_p \approx \alpha I_c = 0.7 \times I_c$$

From $E_J = \Phi_0 I_c/2\pi$:
$$I_c = \frac{2\pi E_J}{\Phi_0} = \frac{2\pi \times 200 \times 10^9 \times h}{2.07 \times 10^{-15}}$$
$$I_c = \frac{2\pi \times 200 \times 6.63 \times 10^{-25}}{2.07 \times 10^{-15}} = 400 \text{ nA}$$

$$I_p \approx 0.7 \times 400 \text{ nA} = 280 \text{ nA}$$

(b) Energy bias:
$$\epsilon = 2I_p\Phi_0(f - 0.5) = 2 \times 280 \times 10^{-9} \times 2.07 \times 10^{-15} \times 0.002$$
$$\epsilon = 2.32 \times 10^{-24} \text{ J}$$
$$\epsilon/h = 2.32 \times 10^{-24} / 6.63 \times 10^{-34} = 3.5 \text{ GHz}$$

(c) Qubit frequency:
$$\omega_q = \sqrt{\epsilon^2 + \Delta^2}/\hbar$$
$$f_q = \sqrt{(3.5)^2 + (5)^2} = \sqrt{12.25 + 25} = \sqrt{37.25} = 6.1 \text{ GHz}$$

### Example 2: Fluxonium Design

**Problem**: A fluxonium has $E_C/h = 1$ GHz, $E_J/h = 8$ GHz, and $E_L/h = 0.5$ GHz.
(a) What regime is this (heavy or light)?
(b) Estimate the plasma frequency
(c) Why is this called "fluxonium"?

**Solution**:

(a) Compare energy scales:
- $E_J/E_C = 8$ (moderate)
- $E_J/E_L = 16$ (large)
- $E_L/E_C = 0.5$ (small)

This is **heavy fluxonium**: $E_J \gg E_L$ and $E_J \gg E_C$.

(b) Plasma frequency:
$$\omega_p/2\pi = \sqrt{8E_JE_C}/h = \sqrt{8 \times 8 \times 1} = \sqrt{64} = 8 \text{ GHz}$$

(c) The name "fluxonium" combines "flux" (the magnetic flux variable $\varphi$) with the "-onium" suffix (like positronium, muonium). At the sweet spot, the wavefunctions are superpositions of different flux states, making the flux variable the primary degree of freedom.

### Example 3: Sweet Spot Protection

**Problem**: A flux qubit has $\Delta/h = 4$ GHz and $\partial\epsilon/\partial\Phi = 2I_p = 500$ nA. Calculate:
(a) The sensitivity to flux noise at the sweet spot
(b) The second-order sensitivity
(c) If flux noise has spectral density $S_\Phi = (1\ \mu\Phi_0)^2/\text{Hz}$, estimate the dephasing rate

**Solution**:

(a) At sweet spot ($\epsilon = 0$):
$$\frac{\partial\omega_q}{\partial\Phi} = \frac{\partial\omega_q}{\partial\epsilon}\frac{\partial\epsilon}{\partial\Phi} = 0 \times \frac{2I_p}{\hbar} = 0$$

First-order sensitivity vanishes!

(b) Second-order:
$$\frac{\partial^2\omega_q}{\partial\epsilon^2} = \frac{1}{\Delta}$$
$$\frac{\partial^2\omega_q}{\partial\Phi^2} = \frac{1}{\Delta}\left(\frac{2I_p}{\hbar}\right)^2$$
$$= \frac{(2 \times 500 \times 10^{-9})^2}{h \times 4 \times 10^9 \times \hbar}$$
$$= \frac{10^{-12}}{6.63 \times 10^{-34} \times 4 \times 10^9 \times 1.055 \times 10^{-34}}$$

This gives a very large number; let's express more usefully:

$$\frac{\partial^2 f_q}{\partial\Phi^2} = \frac{(2I_p)^2}{h^2\Delta} = \frac{(10^{-6})^2}{(6.63 \times 10^{-34})^2 \times 4 \times 10^9}$$

The second-order term contributes dephasing proportional to $\langle(\delta\Phi)^2\rangle$.

(c) Second-order dephasing:
$$\Gamma_\phi^{(2)} \sim \frac{1}{2}\left(\frac{\partial^2\omega}{\partial\Phi^2}\right)S_\Phi \sim \frac{(2I_p)^2 S_\Phi}{2\Delta}$$

With $S_\Phi = (10^{-6}\Phi_0)^2/\text{Hz}$:
$$\Gamma_\phi \sim \frac{(10^{-6} \times 2.07 \times 10^{-15} \times 500 \times 10^{-9})^2}{4 \times 10^9 \times h}$$

This gives $T_\phi \sim$ tens of microseconds, significantly better than operating away from sweet spot.

## Practice Problems

### Level 1: Direct Application

1. A flux qubit has persistent current 250 nA and tunneling gap $\Delta/h = 6$ GHz. Calculate the qubit frequency at flux biases $f = 0.5$, $f = 0.51$, and $f = 0.52$.

2. Calculate the inductive energy $E_L$ for loop inductances of 10 pH, 100 pH, and 1 nH.

3. A fluxonium has $E_J/h = 10$ GHz and $E_L/h = 1$ GHz. What is the ratio $E_J/E_L$? What regime is this?

### Level 2: Intermediate

4. Derive the effective two-level Hamiltonian for a double-well potential near the degeneracy point using the tight-binding approximation (consider only the ground state of each well).

5. For a fluxonium with $E_C/h = 1.2$ GHz, $E_J/h = 9$ GHz, $E_L/h = 0.4$ GHz:
   (a) Calculate $E_J/E_C$ and compare to transmon
   (b) Estimate the plasma frequency
   (c) What makes fluxonium different from a tunable transmon despite similar $E_J/E_C$?

6. Explain why the matrix element $\langle 0|\hat{\varphi}|1\rangle$ vanishes at the fluxonium sweet spot. What are the consequences for qubit control and readout?

### Level 3: Challenging

7. **WKB analysis**: For a double-well potential with barrier height $U_b$ and well separation $\Delta\varphi$, estimate the tunneling splitting using the WKB approximation:
$$\Delta = \hbar\omega_0 \exp\left(-\frac{1}{\hbar}\int_{\varphi_1}^{\varphi_2}\sqrt{2m(U(\varphi) - E)}\,d\varphi\right)$$
Apply this to a flux qubit with $E_J/E_L = 50$.

8. **Noise engineering**: Design a flux qubit system where both charge and flux noise are suppressed to first order. What constraints does this place on the qubit parameters?

9. **Superinductor design**: Design a Josephson junction array to achieve $L = 200$ nH using junctions with $I_c = 50$ nA. How many junctions are needed? What is the plasma frequency of the array?

## Computational Lab: Flux Qubit and Fluxonium Simulation

```python
"""
Day 899 Computational Lab: Flux Qubits and Fluxonium
Simulating double-well potentials, sweet spots, and energy spectra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# Physical constants
h = 6.626e-34
hbar = h / (2 * np.pi)
Phi_0 = 2.068e-15
e = 1.602e-19

# =============================================================================
# Part 1: Fluxonium Hamiltonian
# =============================================================================

def fluxonium_hamiltonian(EC, EJ, EL, f_ext, n_phi=201, phi_max=4*np.pi):
    """
    Construct fluxonium Hamiltonian in phase basis.

    H = 4*EC*n^2 - EJ*cos(phi) + 0.5*EL*(phi - 2*pi*f)^2

    Parameters:
    -----------
    EC, EJ, EL : float
        Energies in GHz
    f_ext : float
        External flux in units of Phi_0
    n_phi : int
        Number of grid points in phase
    phi_max : float
        Maximum phase value

    Returns:
    --------
    H : ndarray
        Hamiltonian matrix
    phi : ndarray
        Phase grid
    """
    # Phase grid
    phi = np.linspace(-phi_max, phi_max, n_phi)
    dphi = phi[1] - phi[0]

    # Potential energy
    V = -EJ * np.cos(phi) + 0.5 * EL * (phi - 2 * np.pi * f_ext)**2

    # Kinetic energy (second derivative using finite differences)
    # -4*EC*d^2/dphi^2 in units where hbar = 1 (energies in GHz)
    # Need to convert: 4*EC*(hbar^2/2)*d^2/dphi^2, but in our units EC already
    # includes the correct prefactor
    kinetic_coeff = 4 * EC / dphi**2

    # Construct tridiagonal kinetic energy matrix
    diagonals = [
        kinetic_coeff * np.ones(n_phi),  # Main diagonal
        -0.5 * kinetic_coeff * np.ones(n_phi - 1),  # Upper diagonal
        -0.5 * kinetic_coeff * np.ones(n_phi - 1)   # Lower diagonal
    ]
    T = diags(diagonals, [0, 1, -1]).toarray()

    # Total Hamiltonian
    H = T + np.diag(V)

    return H, phi, V

def fluxonium_spectrum(EC, EJ, EL, f_ext, n_levels=6, n_phi=201):
    """
    Calculate fluxonium energy spectrum.
    """
    H, phi, V = fluxonium_hamiltonian(EC, EJ, EL, f_ext, n_phi)
    energies, states = eigh(H)

    # Shift ground state to zero
    energies = energies - energies[0]

    return energies[:n_levels], states[:, :n_levels], phi, V

# Example: Heavy fluxonium
EC = 1.0   # GHz
EJ = 8.0   # GHz
EL = 0.5   # GHz

print("=" * 60)
print("Fluxonium Energy Spectrum")
print("=" * 60)
print(f"EC = {EC} GHz")
print(f"EJ = {EJ} GHz")
print(f"EL = {EL} GHz")
print(f"EJ/EL = {EJ/EL:.1f}")
print(f"EJ/EC = {EJ/EC:.1f}")

# At sweet spot
f_sweet = 0.5
energies, states, phi, V = fluxonium_spectrum(EC, EJ, EL, f_sweet)

print(f"\nAt sweet spot (f = {f_sweet}):")
for i in range(5):
    print(f"  E_{i} = {energies[i]:.4f} GHz")

omega_01 = energies[1] - energies[0]
omega_12 = energies[2] - energies[1]
alpha = omega_12 - omega_01
print(f"\nω_01/2π = {omega_01:.4f} GHz")
print(f"ω_12/2π = {omega_12:.4f} GHz")
print(f"Anharmonicity α/2π = {alpha:.4f} GHz")

# =============================================================================
# Part 2: Flux Dependence and Sweet Spot
# =============================================================================

# Calculate spectrum vs external flux
f_values = np.linspace(0.3, 0.7, 100)
spectra = []

for f in f_values:
    E, _, _, _ = fluxonium_spectrum(EC, EJ, EL, f, n_levels=5)
    spectra.append(E)

spectra = np.array(spectra)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Energy levels vs flux
ax1 = axes[0, 0]
for i in range(5):
    ax1.plot(f_values, spectra[:, i], linewidth=2, label=f'|{i}⟩')
ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Sweet spot')
ax1.set_xlabel(r'External flux $\Phi_{ext}/\Phi_0$', fontsize=12)
ax1.set_ylabel('Energy (GHz)', fontsize=12)
ax1.set_title('Fluxonium Energy Spectrum vs Flux', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Transition frequency and sensitivity
ax2 = axes[0, 1]
omega_01_vs_f = spectra[:, 1] - spectra[:, 0]
ax2.plot(f_values, omega_01_vs_f, 'b-', linewidth=2)
ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel(r'External flux $\Phi_{ext}/\Phi_0$', fontsize=12)
ax2.set_ylabel(r'$\omega_{01}/2\pi$ (GHz)', fontsize=12)
ax2.set_title('Qubit Frequency vs Flux', fontsize=14)
ax2.grid(True, alpha=0.3)

# Mark minimum at sweet spot
min_idx = np.argmin(omega_01_vs_f)
ax2.scatter(f_values[min_idx], omega_01_vs_f[min_idx], color='red', s=100,
            zorder=5, label=f'Min: {omega_01_vs_f[min_idx]:.3f} GHz')
ax2.legend(fontsize=11)

# =============================================================================
# Part 3: Potential and Wavefunctions
# =============================================================================

ax3 = axes[1, 0]

# Plot potential and wavefunctions at sweet spot
energies, states, phi, V = fluxonium_spectrum(EC, EJ, EL, 0.5, n_levels=4, n_phi=401)
phi_plot = phi / np.pi  # Convert to units of pi

# Potential
ax3.plot(phi_plot, V, 'k-', linewidth=2, label='Potential')

# Wavefunctions (scaled for visibility)
scale = 3.0
for i in range(4):
    psi = states[:, i]
    psi_scaled = psi / np.max(np.abs(psi)) * scale
    ax3.fill_between(phi_plot, energies[i], energies[i] + psi_scaled**2 * 0.5,
                     alpha=0.5, label=f'|{i}⟩')
    ax3.axhline(energies[i], color=f'C{i}', linestyle='--', alpha=0.3)

ax3.set_xlabel(r'Phase $\varphi/\pi$', fontsize=12)
ax3.set_ylabel('Energy (GHz)', fontsize=12)
ax3.set_title('Fluxonium Potential and Wavefunctions (f = 0.5)', fontsize=14)
ax3.set_xlim([-3, 3])
ax3.set_ylim([-EJ-1, 10])
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# =============================================================================
# Part 4: Compare Different Regimes
# =============================================================================

ax4 = axes[1, 1]

# Heavy vs light fluxonium
params_list = [
    {'EC': 1.0, 'EJ': 8.0, 'EL': 0.5, 'label': 'Heavy (EJ/EL=16)'},
    {'EC': 1.0, 'EJ': 8.0, 'EL': 2.0, 'label': 'Medium (EJ/EL=4)'},
    {'EC': 1.0, 'EJ': 4.0, 'EL': 2.0, 'label': 'Light (EJ/EL=2)'},
]

for params in params_list:
    f_vals = np.linspace(0.3, 0.7, 50)
    omega_01 = []
    for f in f_vals:
        E, _, _, _ = fluxonium_spectrum(params['EC'], params['EJ'],
                                        params['EL'], f, n_levels=2)
        omega_01.append(E[1] - E[0])
    ax4.plot(f_vals, omega_01, linewidth=2, label=params['label'])

ax4.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel(r'External flux $\Phi_{ext}/\Phi_0$', fontsize=12)
ax4.set_ylabel(r'$\omega_{01}/2\pi$ (GHz)', fontsize=12)
ax4.set_title('Fluxonium Regimes Comparison', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fluxonium_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 5: Persistent Current Flux Qubit Model
# =============================================================================

def pc_flux_qubit_energy(Ip, Delta, f):
    """
    Two-level model for persistent current flux qubit.

    E_± = ± 0.5 * sqrt(epsilon^2 + Delta^2)
    epsilon = 2 * Ip * Phi_0 * (f - 0.5)
    """
    epsilon = 2 * Ip * Phi_0 * (f - 0.5)  # in Joules
    epsilon_ghz = epsilon / h / 1e9  # convert to GHz
    Delta_ghz = Delta  # already in GHz

    E = 0.5 * np.sqrt(epsilon_ghz**2 + Delta_ghz**2)
    return E, -E, epsilon_ghz

print("\n" + "=" * 60)
print("Persistent Current Flux Qubit")
print("=" * 60)

# Typical parameters
Ip = 300e-9  # 300 nA
Delta = 5.0  # 5 GHz

f_vals = np.linspace(0.48, 0.52, 100)
E_plus = []
E_minus = []
eps_vals = []

for f in f_vals:
    Ep, Em, eps = pc_flux_qubit_energy(Ip, Delta, f)
    E_plus.append(Ep)
    E_minus.append(Em)
    eps_vals.append(eps)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Energy levels
ax1 = axes2[0]
ax1.plot(f_vals, E_plus, 'b-', linewidth=2, label=r'$|1\rangle$')
ax1.plot(f_vals, E_minus, 'r-', linewidth=2, label=r'$|0\rangle$')
ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Sweet spot')
ax1.set_xlabel(r'External flux $\Phi_{ext}/\Phi_0$', fontsize=12)
ax1.set_ylabel('Energy (GHz)', fontsize=12)
ax1.set_title('PC Flux Qubit Energy Levels', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Frequency and sensitivity
ax2 = axes2[1]
omega_q = np.array(E_plus) - np.array(E_minus)
ax2.plot(f_vals, omega_q, 'g-', linewidth=2)
ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(Delta, color='red', linestyle=':', alpha=0.7, label=rf'$\Delta$ = {Delta} GHz')
ax2.set_xlabel(r'External flux $\Phi_{ext}/\Phi_0$', fontsize=12)
ax2.set_ylabel(r'$\omega_q/2\pi$ (GHz)', fontsize=12)
ax2.set_title('Qubit Frequency vs Flux', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pc_flux_qubit.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 6: Flux Sensitivity Analysis
# =============================================================================

print("\n" + "=" * 60)
print("Flux Sensitivity Analysis")
print("=" * 60)

# Calculate d(omega)/d(f) numerically
df = 1e-5
omega_center = np.sqrt(Delta**2) # at f=0.5, epsilon=0

_, _, eps_plus = pc_flux_qubit_energy(Ip, Delta, 0.5 + df)
_, _, eps_minus = pc_flux_qubit_energy(Ip, Delta, 0.5 - df)

omega_plus = np.sqrt(eps_plus**2 + Delta**2)
omega_minus = np.sqrt(eps_minus**2 + Delta**2)

d_omega_df = (omega_plus - omega_minus) / (2 * df)
d2_omega_df2 = (omega_plus - 2*omega_center + omega_minus) / (df**2)

print(f"At sweet spot (f = 0.5):")
print(f"  ω_q = {omega_center:.3f} GHz")
print(f"  dω/df ≈ {d_omega_df:.2e} GHz (should be ~0)")
print(f"  d²ω/df² = {d2_omega_df2:.2f} GHz")

# Away from sweet spot
f_off = 0.51
_, _, eps_off = pc_flux_qubit_energy(Ip, Delta, f_off)
omega_off = np.sqrt(eps_off**2 + Delta**2)

_, _, eps_off_plus = pc_flux_qubit_energy(Ip, Delta, f_off + df)
_, _, eps_off_minus = pc_flux_qubit_energy(Ip, Delta, f_off - df)
omega_off_plus = np.sqrt(eps_off_plus**2 + Delta**2)
omega_off_minus = np.sqrt(eps_off_minus**2 + Delta**2)

d_omega_df_off = (omega_off_plus - omega_off_minus) / (2 * df)

print(f"\nAway from sweet spot (f = {f_off}):")
print(f"  ω_q = {omega_off:.3f} GHz")
print(f"  dω/df = {d_omega_df_off:.2f} GHz per unit flux")
print(f"  Sensitivity ratio: {abs(d_omega_df_off/d2_omega_df2):.0f}x worse")

# =============================================================================
# Part 7: Matrix Element at Sweet Spot
# =============================================================================

print("\n" + "=" * 60)
print("Matrix Elements Analysis")
print("=" * 60)

# Calculate matrix elements for fluxonium
energies, states, phi, _ = fluxonium_spectrum(EC, EJ, EL, 0.5, n_levels=4, n_phi=401)
dphi = phi[1] - phi[0]

# Matrix elements of phi operator
phi_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        phi_matrix[i, j] = np.sum(states[:, i] * phi * states[:, j]) * dphi

print("Matrix elements <i|φ|j> (at sweet spot):")
print(np.array2string(phi_matrix, precision=4, suppress_small=True))

print(f"\n<0|φ|1> = {phi_matrix[0,1]:.4f} (should be ~0 at sweet spot)")
print(f"<1|φ|2> = {phi_matrix[1,2]:.4f}")
print(f"<0|φ|2> = {phi_matrix[0,2]:.4f}")

# Away from sweet spot
energies_off, states_off, phi_off, _ = fluxonium_spectrum(EC, EJ, EL, 0.45, n_levels=4, n_phi=401)

phi_matrix_off = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        phi_matrix_off[i, j] = np.sum(states_off[:, i] * phi_off * states_off[:, j]) * dphi

print(f"\nAway from sweet spot (f = 0.45):")
print(f"<0|φ|1> = {phi_matrix_off[0,1]:.4f}")
print(f"<1|φ|2> = {phi_matrix_off[1,2]:.4f}")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| PC Flux Qubit | $\hat{H} = \frac{\epsilon}{2}\hat{\sigma}_z + \frac{\Delta}{2}\hat{\sigma}_x$ |
| Energy bias | $\epsilon = 2I_p\Phi_0(f - 0.5)$ |
| Qubit frequency | $\omega_q = \sqrt{\epsilon^2 + \Delta^2}/\hbar$ |
| Fluxonium | $\hat{H} = 4E_C\hat{n}^2 - E_J\cos\hat{\varphi} + \frac{1}{2}E_L(\hat{\varphi} - 2\pi f)^2$ |
| Inductive energy | $E_L = \Phi_0^2/(8\pi^2 L)$ |
| Sweet spot | $f = \Phi_{ext}/\Phi_0 = 0.5$ |

### Main Takeaways

1. **Flux qubits** encode information in persistent current states (clockwise vs counterclockwise)

2. **Sweet spot at half flux quantum**: First-order insensitivity to flux noise when $\Phi_{ext} = \Phi_0/2$

3. **Double-well potential**: Tunneling between wells sets qubit frequency; barrier height determines coherence

4. **Fluxonium** uses superinductor to create flat double-well with very low qubit frequency

5. **Heavy fluxonium** ($E_J \gg E_L$): Extremely long coherence (>1 ms demonstrated), very large anharmonicity, but slow gates

6. **Matrix element vanishes** at sweet spot for parity-protected transitions

## Daily Checklist

- [ ] I can derive the persistent current flux qubit Hamiltonian
- [ ] I understand the double-well potential and its dependence on external flux
- [ ] I can identify and explain flux sweet spots
- [ ] I understand the difference between heavy and light fluxonium
- [ ] I can calculate qubit frequencies from circuit parameters
- [ ] I have run the computational lab and understand the results
- [ ] I can compare advantages/disadvantages of flux qubits vs transmons

## Preview: Day 900

Tomorrow we explore **single-qubit gates** in superconducting qubits:

- Microwave pulse design and Rabi oscillations
- DRAG pulse correction for leakage suppression
- Gate calibration protocols (Rabi, Ramsey, AllXY)
- Randomized benchmarking for gate fidelity
- Virtual-Z gates and frame tracking

---

*"The flux qubit teaches us that coherent quantum tunneling—a phenomenon that seems impossibly delicate—can be made robust enough for quantum computing."*
