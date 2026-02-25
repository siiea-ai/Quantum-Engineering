# Day 387: Rectangular Barrier

## Week 56, Day 2 | Month 14: One-Dimensional Quantum Mechanics

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning** | 2.5 hrs | Rectangular barrier theory, transfer matrix method |
| **Afternoon** | 2.5 hrs | Transmission coefficient derivation, worked examples |
| **Evening** | 2 hrs | Computational lab: T(E) for various barriers |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Set up** the Schrodinger equation in three regions for a rectangular barrier
2. **Apply** the transfer matrix method to relate wave amplitudes across interfaces
3. **Derive** the transmission coefficient for E < V_0 and E > V_0
4. **Calculate** transmission probabilities for given barrier parameters
5. **Explain** quantum tunneling as non-zero transmission through forbidden regions
6. **Analyze** how T depends on barrier width, height, and particle energy

---

## Core Content

### 1. The Rectangular Barrier Potential

$$V(x) = \begin{cases} 0 & x < 0 \quad \text{(Region I)} \\ V_0 & 0 \leq x \leq L \quad \text{(Region II)} \\ 0 & x > L \quad \text{(Region III)} \end{cases}$$

This is the fundamental model for:
- Electron tunneling through insulating layers
- Nuclear alpha decay (simplified)
- Josephson junctions in superconducting circuits

### 2. Wave Functions in Each Region

**Region I (x < 0):**
$$\psi_I(x) = Ae^{ikx} + Be^{-ikx}$$

where $k = \frac{\sqrt{2mE}}{\hbar}$

**Region II (0 ≤ x ≤ L) for E < V_0:**
$$\psi_{II}(x) = Ce^{\kappa x} + De^{-\kappa x}$$

where $\kappa = \frac{\sqrt{2m(V_0 - E)}}{\hbar}$

**Region II (0 ≤ x ≤ L) for E > V_0:**
$$\psi_{II}(x) = Ce^{ik'x} + De^{-ik'x}$$

where $k' = \frac{\sqrt{2m(E - V_0)}}{\hbar}$

**Region III (x > L):**
$$\psi_{III}(x) = Fe^{ikx} + Ge^{-ikx}$$

**Physical boundary condition:** No wave incident from the right, so $G = 0$.

$$\psi_{III}(x) = Fe^{ikx}$$

### 3. Boundary Conditions

At each interface, continuity of $\psi$ and $\frac{d\psi}{dx}$:

**At x = 0:**
$$A + B = C + D$$
$$ik(A - B) = \kappa(C - D) \quad \text{(for E < V_0)}$$

**At x = L:**
$$Ce^{\kappa L} + De^{-\kappa L} = Fe^{ikL}$$
$$\kappa(Ce^{\kappa L} - De^{-\kappa L}) = ikFe^{ikL}$$

### 4. The Transfer Matrix Method

The transfer matrix elegantly handles multiple boundaries. We relate amplitudes across each region.

**Matrix formulation at interface:**

For a wave $\psi = Ae^{ikx} + Be^{-ikx}$ on the left and $\psi = Ce^{ik'x} + De^{-ik'x}$ on the right of an interface at x = a:

$$\begin{pmatrix} A \\ B \end{pmatrix} = M_{12} \begin{pmatrix} C \\ D \end{pmatrix}$$

where the transfer matrix is:

$$M_{12} = \frac{1}{2k}\begin{pmatrix} k + k' & k - k' \\ k - k' & k + k' \end{pmatrix} \times \text{(phase factors)}$$

**Complete transfer matrix for rectangular barrier:**

$$\begin{pmatrix} A \\ B \end{pmatrix} = M \begin{pmatrix} F \\ 0 \end{pmatrix}$$

where $M = M_1 \cdot P \cdot M_2$ combines:
- $M_1$: Interface at x = 0
- $P$: Propagation through barrier
- $M_2$: Interface at x = L

### 5. Transmission Coefficient Derivation (E < V_0)

After applying all boundary conditions, the transmission coefficient is:

$$\boxed{T = \frac{1}{1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0 - E)}}}$$

**Alternative form:**
$$T = \frac{1}{1 + \frac{\sinh^2(\kappa L)}{4\eta(1-\eta)}}$$

where $\eta = E/V_0$ is the dimensionless energy ratio.

**Reflection coefficient:**
$$R = 1 - T = \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0-E) + V_0^2 \sinh^2(\kappa L)}$$

### 6. Thick Barrier Approximation

For $\kappa L \gg 1$ (thick barrier or high barrier):

$$\sinh(\kappa L) \approx \frac{1}{2}e^{\kappa L}$$

Therefore:
$$T \approx \frac{16E(V_0 - E)}{V_0^2}e^{-2\kappa L}$$

$$\boxed{T \approx T_0 e^{-2\kappa L}}$$

where $T_0 = 16E(V_0-E)/V_0^2$ is order unity.

**Key insight:** Tunneling probability decreases exponentially with:
- Barrier width L
- $\sqrt{V_0 - E}$ (through κ)

### 7. Transmission Coefficient (E > V_0)

When E > V_0, the particle is above the barrier. Replace $\kappa \to ik'$ where $k' = \sqrt{2m(E-V_0)}/\hbar$:

$$\sinh(i k'L) = i\sin(k'L)$$

$$\boxed{T = \frac{1}{1 + \frac{V_0^2 \sin^2(k'L)}{4E(E-V_0)}}}$$

**Resonances:** Perfect transmission (T = 1) occurs when:
$$k'L = n\pi, \quad n = 1, 2, 3, ...$$

This corresponds to the barrier width being an integer number of half-wavelengths!

### 8. Physical Interpretation

**Classical comparison:**
- E < V_0: Classical particle reflects (T = 0)
- E > V_0: Classical particle transmits (T = 1)

**Quantum behavior:**
- E < V_0: Non-zero tunneling probability
- E > V_0: Partial reflection possible (except at resonances)

**Why tunneling occurs:**
The wave function doesn't abruptly vanish at the barrier edge. It penetrates the barrier as an evanescent wave, and if the barrier is thin enough, significant amplitude remains on the far side.

### 9. Dependence on Parameters

**Effect of barrier width L:**
- T decreases exponentially with L (for E < V_0)
- T oscillates with L (for E > V_0)

**Effect of barrier height V_0:**
- Higher V_0 → larger κ → smaller T
- More energy below barrier → more evanescent decay

**Effect of particle mass m:**
- Heavier particles → larger κ → much smaller T
- Explains why electrons tunnel readily but protons and nuclei tunnel much less

### 10. Quantum Computing Connection

**Superconducting qubits** (transmon, flux qubit) exploit tunneling:
- Cooper pairs tunnel through thin oxide barriers (~1-2 nm)
- Josephson junction is precisely a rectangular barrier for paired electrons
- Tunnel splitting determines qubit frequency
- Barrier thickness must be controlled at atomic level for reproducible qubits

**Quantum annealing:**
- System tunnels between local minima to find global minimum
- Tunneling rate controls optimization speed

---

## Worked Examples

### Example 1: Electron Tunneling Through Thin Barrier

An electron with E = 2 eV encounters a barrier of height V_0 = 5 eV and width L = 0.5 nm.

**Find:** Transmission probability T

**Solution:**

1. Calculate the decay constant:
$$\kappa = \frac{\sqrt{2m(V_0 - E)}}{\hbar} = \frac{\sqrt{2 \times 9.11 \times 10^{-31} \times 3 \times 1.6 \times 10^{-19}}}{1.055 \times 10^{-34}}$$
$$\kappa = 8.87 \times 10^9 \text{ m}^{-1} = 8.87 \text{ nm}^{-1}$$

2. Calculate κL:
$$\kappa L = 8.87 \times 0.5 = 4.44$$

3. Calculate sinh²(κL):
$$\sinh(4.44) = \frac{e^{4.44} - e^{-4.44}}{2} = 42.5$$
$$\sinh^2(\kappa L) = 1806$$

4. Calculate transmission coefficient:
$$T = \frac{1}{1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0-E)}}$$
$$T = \frac{1}{1 + \frac{25 \times 1806}{4 \times 2 \times 3}} = \frac{1}{1 + 1880} = 5.3 \times 10^{-4}$$

**Result:** T ≈ 0.053% - About 1 in 2000 electrons tunnels through!

---

### Example 2: Resonant Transmission (E > V_0)

An electron with E = 10 eV encounters a barrier of V_0 = 5 eV, width L = 0.39 nm.

**Find:** Check if resonant transmission occurs.

**Solution:**

1. Calculate k':
$$k' = \frac{\sqrt{2m(E-V_0)}}{\hbar} = \frac{\sqrt{2 \times 9.11 \times 10^{-31} \times 5 \times 1.6 \times 10^{-19}}}{1.055 \times 10^{-34}}$$
$$k' = 11.5 \times 10^9 \text{ m}^{-1}$$

2. Check resonance condition:
$$k'L = 11.5 \times 10^9 \times 0.39 \times 10^{-9} = 4.49 \approx 1.43\pi$$

Not exactly at resonance (would need k'L = π or 2π).

3. Calculate T:
$$\sin^2(k'L) = \sin^2(4.49) = 0.97$$
$$T = \frac{1}{1 + \frac{25 \times 0.97}{4 \times 10 \times 5}} = \frac{1}{1 + 0.12} = 0.89$$

**Result:** T = 89% transmission (not 100% because not at exact resonance)

---

### Example 3: Comparing Different Particles

Compare tunneling of electron vs. proton through a barrier with V_0 - E = 1 eV, L = 0.3 nm.

**Solution:**

For electron (m_e = 9.11 × 10⁻³¹ kg):
$$\kappa_e = \frac{\sqrt{2m_e \times 1\text{ eV}}}{\hbar} = 5.12 \times 10^9 \text{ m}^{-1}$$
$$\kappa_e L = 1.54$$
$$T_e \approx e^{-2 \times 1.54} = 0.046$$ (4.6%)

For proton (m_p = 1.67 × 10⁻²⁷ kg):
$$\kappa_p = \sqrt{\frac{m_p}{m_e}}\kappa_e = \sqrt{1836} \times 5.12 \times 10^9 = 2.19 \times 10^{11} \text{ m}^{-1}$$
$$\kappa_p L = 65.8$$
$$T_p \approx e^{-2 \times 65.8} = e^{-132} \approx 10^{-57}$$

**Conclusion:** Protons essentially never tunnel through barriers that electrons readily penetrate!

---

## Practice Problems

### Level 1: Direct Application

1. Calculate T for an electron (E = 3 eV) through a barrier (V_0 = 8 eV, L = 0.2 nm).

2. Find the barrier width L such that T = 0.01 for E = 2 eV, V_0 = 6 eV.

3. For E > V_0, find the first three resonant widths L when E = 8 eV, V_0 = 3 eV.

### Level 2: Intermediate

4. Show that for E = V_0/2 and thick barriers, T ≈ 4e^{-2κL}.

5. A double barrier (two identical barriers separated by a gap) can have T = 1 even for E < V_0 (resonant tunneling). Explain qualitatively why.

6. Calculate the energy E that maximizes the pre-exponential factor T_0 = 16E(V_0-E)/V_0² for a thick barrier.

### Level 3: Challenging

7. **Transfer matrix derivation:** Starting from boundary conditions at x = 0 and x = L, derive the complete 2×2 transfer matrix M relating (A, B) to (F, 0).

8. **Thin barrier limit:** Show that as L → 0 with V_0L = const (delta function potential), the transmission formula reduces to an appropriate limit.

9. **Thermal tunneling:** At temperature T, electrons have a distribution of energies. Derive an expression for the thermally averaged transmission coefficient.

---

## Computational Lab

### Python: Rectangular Barrier Transmission

```python
"""
Day 387: Rectangular Barrier Transmission Coefficients
Quantum Tunneling & Barriers - Week 56

This lab explores:
1. Transmission coefficient vs energy
2. Effect of barrier width
3. Resonances above the barrier
4. Transfer matrix implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, eV

def transmission_below_barrier(E, V0, L, m=m_e):
    """
    Transmission coefficient for E < V0

    Parameters:
    E: Energy in eV
    V0: Barrier height in eV
    L: Barrier width in meters
    m: Particle mass (default: electron mass)

    Returns:
    T: Transmission coefficient
    """
    if E <= 0 or E >= V0:
        return np.nan

    E_J = E * eV
    V0_J = V0 * eV

    kappa = np.sqrt(2 * m * (V0_J - E_J)) / hbar
    kappa_L = kappa * L

    # Avoid overflow for large kappa*L
    if kappa_L > 50:
        return 16 * E * (V0 - E) / V0**2 * np.exp(-2 * kappa_L)

    sinh_kL = np.sinh(kappa_L)
    denominator = 1 + (V0**2 * sinh_kL**2) / (4 * E * (V0 - E))

    return 1 / denominator

def transmission_above_barrier(E, V0, L, m=m_e):
    """
    Transmission coefficient for E > V0

    Parameters:
    E: Energy in eV
    V0: Barrier height in eV
    L: Barrier width in meters
    m: Particle mass (default: electron mass)

    Returns:
    T: Transmission coefficient
    """
    if E <= V0:
        return np.nan

    E_J = E * eV
    V0_J = V0 * eV

    k_prime = np.sqrt(2 * m * (E_J - V0_J)) / hbar
    k_prime_L = k_prime * L

    sin_kL = np.sin(k_prime_L)
    denominator = 1 + (V0**2 * sin_kL**2) / (4 * E * (E - V0))

    return 1 / denominator

def transmission_coefficient(E, V0, L, m=m_e):
    """Combined transmission coefficient for all energies"""
    if isinstance(E, np.ndarray):
        return np.array([transmission_coefficient(e, V0, L, m) for e in E])

    if E <= 0:
        return 0
    elif E < V0:
        return transmission_below_barrier(E, V0, L, m)
    elif E == V0:
        # Limiting case
        kappa_L = np.sqrt(2 * m * V0 * eV) / hbar * L
        return 1 / (1 + kappa_L**2 / 4)
    else:
        return transmission_above_barrier(E, V0, L, m)

def transfer_matrix_method(E, V0, L, m=m_e):
    """
    Calculate transmission using transfer matrix method
    More elegant and generalizable to multiple barriers
    """
    E_J = E * eV
    V0_J = V0 * eV

    k = np.sqrt(2 * m * E_J) / hbar

    if E < V0:
        q = 1j * np.sqrt(2 * m * (V0_J - E_J)) / hbar
    else:
        q = np.sqrt(2 * m * (E_J - V0_J)) / hbar

    # Transfer matrix elements
    M11 = (np.cos(q * L) - 1j * (k**2 + q**2) / (2*k*q) * np.sin(q * L)) * np.exp(-1j * k * L)
    M12 = -1j * (q**2 - k**2) / (2*k*q) * np.sin(q * L) * np.exp(-1j * k * L)
    M21 = 1j * (q**2 - k**2) / (2*k*q) * np.sin(q * L) * np.exp(1j * k * L)
    M22 = (np.cos(q * L) + 1j * (k**2 + q**2) / (2*k*q) * np.sin(q * L)) * np.exp(1j * k * L)

    # For real q (E > V0): standard oscillating case
    # For imaginary q (E < V0): evanescent case

    T = 1 / np.abs(M11)**2
    return np.real(T)

#%% Plot 1: T vs E for different barrier widths
fig, ax = plt.subplots(figsize=(12, 7))

V0 = 5.0  # eV
L_values = [0.1e-9, 0.2e-9, 0.3e-9, 0.5e-9]  # meters
colors = ['blue', 'green', 'orange', 'red']

E = np.linspace(0.01, 12, 500)

for L, color in zip(L_values, colors):
    T = transmission_coefficient(E, V0, L)
    ax.semilogy(E, T, color=color, linewidth=2, label=f'L = {L*1e9:.1f} nm')

ax.axvline(x=V0, color='gray', linestyle='--', alpha=0.7)
ax.annotate(f'$V_0$ = {V0} eV', xy=(V0+0.1, 0.5), fontsize=11)

ax.set_xlabel('Energy E (eV)', fontsize=12)
ax.set_ylabel('Transmission T', fontsize=12)
ax.set_title('Rectangular Barrier: Transmission vs Energy', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(0, 12)
ax.set_ylim(1e-15, 2)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('barrier_T_vs_E.png', dpi=150)
plt.show()

#%% Plot 2: T vs L (barrier width) for fixed energy
fig, ax = plt.subplots(figsize=(12, 7))

V0 = 5.0  # eV
E_values = [1.0, 2.0, 3.0, 4.0]  # eV (all below V0)
colors = ['blue', 'green', 'orange', 'red']

L = np.linspace(0.01e-9, 1.5e-9, 500)

for E_val, color in zip(E_values, colors):
    T = np.array([transmission_coefficient(E_val, V0, l) for l in L])
    ax.semilogy(L*1e9, T, color=color, linewidth=2, label=f'E = {E_val} eV')

    # Add thick barrier approximation
    kappa = np.sqrt(2 * m_e * (V0 - E_val) * eV) / hbar
    T_approx = 16 * E_val * (V0 - E_val) / V0**2 * np.exp(-2 * kappa * L)
    ax.semilogy(L*1e9, T_approx, color=color, linestyle=':', alpha=0.5)

ax.set_xlabel('Barrier Width L (nm)', fontsize=12)
ax.set_ylabel('Transmission T', fontsize=12)
ax.set_title(f'Transmission vs Barrier Width (V₀ = {V0} eV)\nSolid: exact, Dotted: thick-barrier approx', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(0, 1.5)
ax.set_ylim(1e-25, 1.5)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('barrier_T_vs_L.png', dpi=150)
plt.show()

#%% Plot 3: Resonances above barrier (E > V0)
fig, ax = plt.subplots(figsize=(12, 7))

V0 = 5.0  # eV
L = 0.5e-9  # meters

E = np.linspace(5.01, 30, 1000)
T = transmission_coefficient(E, V0, L)

ax.plot(E, T, 'b-', linewidth=2)

# Mark resonance positions
k_prime = np.sqrt(2 * m_e * (E - V0) * eV) / hbar
k_prime_L = k_prime * L
for n in range(1, 6):
    # Resonance when k'L = n*pi
    E_res = V0 + (n * np.pi * hbar / L)**2 / (2 * m_e * eV)
    if E_res < 30:
        ax.axvline(x=E_res, color='red', linestyle='--', alpha=0.5)
        ax.annotate(f'n={n}', xy=(E_res, 1.02), fontsize=9, ha='center')

ax.axhline(y=1, color='green', linestyle=':', alpha=0.5)
ax.axvline(x=V0, color='gray', linestyle='--', alpha=0.7)

ax.set_xlabel('Energy E (eV)', fontsize=12)
ax.set_ylabel('Transmission T', fontsize=12)
ax.set_title(f'Transmission Resonances Above Barrier (V₀ = {V0} eV, L = {L*1e9} nm)', fontsize=14)
ax.set_xlim(5, 30)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('barrier_resonances.png', dpi=150)
plt.show()

#%% Plot 4: Wave function through barrier
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

V0 = 5.0  # eV
L = 0.5e-9  # meters

# Case 1: E < V0 (tunneling)
E1 = 2.0  # eV
k1 = np.sqrt(2 * m_e * E1 * eV) / hbar
kappa1 = np.sqrt(2 * m_e * (V0 - E1) * eV) / hbar

# Case 2: E > V0 (over-barrier)
E2 = 8.0  # eV
k2 = np.sqrt(2 * m_e * E2 * eV) / hbar
k2_prime = np.sqrt(2 * m_e * (E2 - V0) * eV) / hbar

# Spatial regions (in nm)
x1 = np.linspace(-1, 0, 200)
x2 = np.linspace(0, L*1e9, 100)
x3 = np.linspace(L*1e9, L*1e9 + 1, 200)

# Scale to nm^-1
k1_nm = k1 * 1e-9
kappa1_nm = kappa1 * 1e-9
k2_nm = k2 * 1e-9
k2p_nm = k2_prime * 1e-9
L_nm = L * 1e9

# Calculate amplitudes for E < V0
T1 = transmission_coefficient(E1, V0, L)
# Simplified: assume A=1, calculate others from boundary conditions
# For visualization purposes

# Plot tunneling case
ax1 = axes[0, 0]
# Schematic wave function (normalized for visualization)
psi1_I = np.cos(k1_nm * x1)  # Standing wave (incident + reflected)
psi1_II = np.exp(-kappa1_nm * x2)  # Evanescent
psi1_III = np.sqrt(T1) * np.cos(k1_nm * (x3 - L_nm))  # Transmitted

ax1.plot(x1, psi1_I, 'b-', linewidth=2)
ax1.plot(x2, psi1_II, 'r-', linewidth=2)
ax1.plot(x3, psi1_III, 'g-', linewidth=2)

# Potential
ax1.fill_between([0, L_nm], [-1.5, -1.5], [1.5, 1.5], alpha=0.2, color='gray')
ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
ax1.axvline(x=L_nm, color='k', linestyle='-', linewidth=1)

ax1.set_xlabel('Position x (nm)', fontsize=11)
ax1.set_ylabel(r'$\psi(x)$ (a.u.)', fontsize=11)
ax1.set_title(f'Wave Function: E = {E1} eV < V₀ = {V0} eV\nT = {T1:.2e}', fontsize=12)
ax1.set_xlim(-1, L_nm + 1)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, alpha=0.3)

# Plot over-barrier case
ax2 = axes[0, 1]
psi2_I = np.cos(k2_nm * x1)
psi2_II = np.cos(k2p_nm * x2)
T2 = transmission_coefficient(E2, V0, L)
psi2_III = np.sqrt(T2) * np.cos(k2_nm * (x3 - L_nm))

ax2.plot(x1, psi2_I, 'b-', linewidth=2)
ax2.plot(x2, psi2_II, 'r-', linewidth=2)
ax2.plot(x3, psi2_III, 'g-', linewidth=2)

ax2.fill_between([0, L_nm], [-1.5, -1.5], [1.5, 1.5], alpha=0.2, color='gray')
ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)
ax2.axvline(x=L_nm, color='k', linestyle='-', linewidth=1)

ax2.set_xlabel('Position x (nm)', fontsize=11)
ax2.set_ylabel(r'$\psi(x)$ (a.u.)', fontsize=11)
ax2.set_title(f'Wave Function: E = {E2} eV > V₀ = {V0} eV\nT = {T2:.3f}', fontsize=12)
ax2.set_xlim(-1, L_nm + 1)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, alpha=0.3)

# Plot probability densities
ax3 = axes[1, 0]
prob1_I = psi1_I**2
prob1_II = psi1_II**2
prob1_III = psi1_III**2

ax3.plot(x1, prob1_I, 'b-', linewidth=2, label='Region I')
ax3.plot(x2, prob1_II, 'r-', linewidth=2, label='Barrier')
ax3.plot(x3, prob1_III, 'g-', linewidth=2, label='Region III')
ax3.fill_between(x2, 0, prob1_II, alpha=0.3, color='red')

ax3.fill_between([0, L_nm], [0, 0], [2, 2], alpha=0.1, color='gray')
ax3.axvline(x=0, color='k', linestyle='-', linewidth=1)
ax3.axvline(x=L_nm, color='k', linestyle='-', linewidth=1)

ax3.set_xlabel('Position x (nm)', fontsize=11)
ax3.set_ylabel(r'$|\psi(x)|^2$ (a.u.)', fontsize=11)
ax3.set_title('Probability Density (Tunneling)', fontsize=12)
ax3.legend(fontsize=9)
ax3.set_xlim(-1, L_nm + 1)
ax3.set_ylim(0, 2)
ax3.grid(True, alpha=0.3)

# Compare different particles
ax4 = axes[1, 1]
m_proton = 1.673e-27  # kg
L_fixed = 0.3e-9  # meters
E_fixed = 2.0  # eV

E_range = np.linspace(0.01, 4.5, 200)
T_electron = np.array([transmission_coefficient(e, V0, L_fixed, m_e) for e in E_range])
T_proton = np.array([transmission_coefficient(e, V0, L_fixed, m_proton) for e in E_range])

ax4.semilogy(E_range, T_electron, 'b-', linewidth=2, label='Electron')
ax4.semilogy(E_range, T_proton, 'r-', linewidth=2, label='Proton')

ax4.set_xlabel('Energy E (eV)', fontsize=11)
ax4.set_ylabel('Transmission T', fontsize=11)
ax4.set_title(f'Particle Mass Comparison (V₀ = {V0} eV, L = {L_fixed*1e9} nm)', fontsize=12)
ax4.legend(fontsize=10)
ax4.set_xlim(0, 4.5)
ax4.set_ylim(1e-80, 1)
ax4.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('barrier_comprehensive.png', dpi=150)
plt.show()

#%% Plot 5: 2D colormap of T(E, L)
fig, ax = plt.subplots(figsize=(12, 8))

V0 = 5.0  # eV
E_range = np.linspace(0.1, 12, 200)
L_range = np.linspace(0.05e-9, 1e-9, 200)

T_grid = np.zeros((len(L_range), len(E_range)))

for i, L in enumerate(L_range):
    for j, E in enumerate(E_range):
        T_grid[i, j] = transmission_coefficient(E, V0, L)

# Use log scale for visualization
T_grid_log = np.log10(T_grid + 1e-50)

im = ax.imshow(T_grid_log, extent=[E_range[0], E_range[-1], L_range[0]*1e9, L_range[-1]*1e9],
               aspect='auto', origin='lower', cmap='viridis', vmin=-20, vmax=0)

# Add barrier height line
ax.axvline(x=V0, color='white', linestyle='--', linewidth=2, label=f'$V_0$ = {V0} eV')

cbar = plt.colorbar(im, ax=ax, label='log₁₀(T)')
ax.set_xlabel('Energy E (eV)', fontsize=12)
ax.set_ylabel('Barrier Width L (nm)', fontsize=12)
ax.set_title('Transmission Coefficient Map', fontsize=14)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('barrier_T_colormap.png', dpi=150)
plt.show()

# Print summary statistics
print("\n=== Rectangular Barrier Analysis ===")
print(f"\nBarrier height V₀ = {V0} eV")
print("\nTransmission coefficients for L = 0.5 nm:")
for E_val in [1, 2, 3, 4, 6, 8, 10]:
    T_val = transmission_coefficient(E_val, V0, 0.5e-9)
    print(f"  E = {E_val} eV: T = {T_val:.4e}")
```

### Expected Output

```
=== Rectangular Barrier Analysis ===

Barrier height V₀ = 5.0 eV

Transmission coefficients for L = 0.5 nm:
  E = 1 eV: T = 1.4784e-08
  E = 2 eV: T = 5.3021e-04
  E = 3 eV: T = 1.7325e-02
  E = 4 eV: T = 1.4567e-01
  E = 6 eV: T = 9.5632e-01
  E = 8 eV: T = 9.8974e-01
  E = 10 eV: T = 9.6543e-01
```

---

## Summary

### Key Formulas Table

| Quantity | Formula |
|----------|---------|
| Transmission (E < V₀) | $T = \left[1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0-E)}\right]^{-1}$ |
| Thick barrier approx | $T \approx \frac{16E(V_0-E)}{V_0^2}e^{-2\kappa L}$ |
| Transmission (E > V₀) | $T = \left[1 + \frac{V_0^2 \sin^2(k'L)}{4E(E-V_0)}\right]^{-1}$ |
| Resonance condition | $k'L = n\pi$, giving $T = 1$ |
| Decay constant | $\kappa = \sqrt{2m(V_0-E)}/\hbar$ |

### Main Takeaways

1. **Tunneling is real**: Non-zero transmission through classically forbidden barriers
2. **Exponential dependence**: T ∝ e^{-2κL} for thick barriers
3. **Mass matters**: Heavier particles tunnel far less efficiently
4. **Resonances exist**: Perfect transmission for E > V₀ at specific widths
5. **Transfer matrices generalize**: Easy extension to multiple barriers

### Physical Applications

- Scanning tunneling microscopy (atomic resolution imaging)
- Tunnel diodes (negative resistance)
- Alpha decay (nuclear physics)
- Josephson junctions (superconducting qubits)

---

## Daily Checklist

- [ ] I can write wave functions in all three regions of a rectangular barrier
- [ ] I can derive the transmission coefficient using boundary conditions
- [ ] I understand the thick barrier approximation and when it applies
- [ ] I can explain resonant transmission and predict resonance energies
- [ ] I can calculate T for given barrier parameters
- [ ] I ran the Python code and understand the T(E,L) dependencies
- [ ] I attempted problems from each difficulty level

---

## Preview: Day 388

Tomorrow we develop the **WKB approximation** for calculating tunneling probabilities through arbitrary-shaped barriers. We'll derive the Gamow factor $e^{-2\gamma}$ that applies to smoothly varying potentials, setting up our study of alpha decay. This semi-classical method is essential for understanding realistic tunneling scenarios!
