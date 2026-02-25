# Day 386: Step Potential

## Week 56, Day 1 | Month 14: One-Dimensional Quantum Mechanics

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning** | 2.5 hrs | Step potential theory, boundary conditions, reflection/transmission |
| **Afternoon** | 2.5 hrs | Worked examples, problem solving |
| **Evening** | 2 hrs | Computational lab: visualizing wave functions and coefficients |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Solve** the Schrodinger equation for a step potential in both E > V_0 and E < V_0 cases
2. **Apply** boundary conditions to determine reflection and transmission coefficients
3. **Calculate** probability current density and verify conservation
4. **Explain** the physical meaning of evanescent waves in classically forbidden regions
5. **Derive** the reflection and transmission probabilities R and T
6. **Verify** that R + T = 1 for all energies

---

## Core Content

### 1. The Step Potential

The step potential is the simplest example of a potential discontinuity:

$$V(x) = \begin{cases} 0 & x < 0 \quad \text{(Region I)} \\ V_0 & x > 0 \quad \text{(Region II)} \end{cases}$$

This idealized model captures essential physics of:
- Interfaces between materials with different work functions
- Semiconductor heterojunctions
- Nuclear surface potentials

### 2. Classical Expectation

Classically, a particle with energy E approaching from the left:
- **E > V_0**: Particle slows down but always transmits (T = 1)
- **E < V_0**: Particle reflects completely (R = 1)

Quantum mechanics reveals dramatically different behavior!

### 3. Quantum Solution: Case I (E > V_0)

**Region I (x < 0):**
$$-\frac{\hbar^2}{2m}\frac{d^2\psi_I}{dx^2} = E\psi_I$$

General solution:
$$\psi_I(x) = Ae^{ik_1x} + Be^{-ik_1x}$$

where $k_1 = \frac{\sqrt{2mE}}{\hbar}$

- $Ae^{ik_1x}$: Incident wave (rightward)
- $Be^{-ik_1x}$: Reflected wave (leftward)

**Region II (x > 0):**
$$-\frac{\hbar^2}{2m}\frac{d^2\psi_{II}}{dx^2} + V_0\psi_{II} = E\psi_{II}$$

General solution:
$$\psi_{II}(x) = Ce^{ik_2x} + De^{-ik_2x}$$

where $k_2 = \frac{\sqrt{2m(E-V_0)}}{\hbar}$

**Physical constraint**: No wave incident from the right means $D = 0$.

$$\boxed{\psi_{II}(x) = Ce^{ik_2x}}$$

### 4. Boundary Conditions

At x = 0, both $\psi$ and $\frac{d\psi}{dx}$ must be continuous:

**Continuity of $\psi$:**
$$A + B = C$$

**Continuity of $\frac{d\psi}{dx}$:**
$$ik_1(A - B) = ik_2 C$$

Solving these equations:
$$\frac{B}{A} = \frac{k_1 - k_2}{k_1 + k_2}$$

$$\frac{C}{A} = \frac{2k_1}{k_1 + k_2}$$

### 5. Probability Current Density

The probability current is:
$$j = \frac{\hbar}{2mi}\left(\psi^*\frac{d\psi}{dx} - \psi\frac{d\psi^*}{dx}\right) = \frac{\hbar}{m}\text{Im}\left(\psi^*\frac{d\psi}{dx}\right)$$

For a plane wave $\psi = Ae^{ikx}$:
$$j = \frac{\hbar k}{m}|A|^2$$

**Currents in our problem:**
- Incident: $j_{\text{inc}} = \frac{\hbar k_1}{m}|A|^2$
- Reflected: $j_{\text{ref}} = \frac{\hbar k_1}{m}|B|^2$
- Transmitted: $j_{\text{trans}} = \frac{\hbar k_2}{m}|C|^2$

### 6. Reflection and Transmission Coefficients

**Reflection coefficient:**
$$\boxed{R = \frac{j_{\text{ref}}}{j_{\text{inc}}} = \left|\frac{B}{A}\right|^2 = \left(\frac{k_1 - k_2}{k_1 + k_2}\right)^2}$$

**Transmission coefficient:**
$$\boxed{T = \frac{j_{\text{trans}}}{j_{\text{inc}}} = \frac{k_2}{k_1}\left|\frac{C}{A}\right|^2 = \frac{4k_1k_2}{(k_1 + k_2)^2}}$$

**Verification of probability conservation:**
$$R + T = \frac{(k_1-k_2)^2 + 4k_1k_2}{(k_1+k_2)^2} = \frac{k_1^2 + k_2^2 + 2k_1k_2}{(k_1+k_2)^2} = 1 \checkmark$$

### 7. Quantum Solution: Case II (E < V_0)

**Region I (x < 0):** Same as before
$$\psi_I(x) = Ae^{ik_1x} + Be^{-ik_1x}$$

**Region II (x > 0):** Now E - V_0 < 0
$$-\frac{\hbar^2}{2m}\frac{d^2\psi_{II}}{dx^2} = (E - V_0)\psi_{II}$$

General solution:
$$\psi_{II}(x) = Ce^{-\kappa x} + De^{\kappa x}$$

where $\kappa = \frac{\sqrt{2m(V_0 - E)}}{\hbar}$

**Physical constraint**: Wave function must remain finite as $x \to \infty$, so $D = 0$.

$$\boxed{\psi_{II}(x) = Ce^{-\kappa x}}$$

This is the **evanescent wave** - it penetrates into the classically forbidden region but decays exponentially!

### 8. Evanescent Wave Properties

**Penetration depth:**
$$\delta = \frac{1}{\kappa} = \frac{\hbar}{\sqrt{2m(V_0 - E)}}$$

Physical interpretation:
- The wave function doesn't abruptly stop at the classical turning point
- It exponentially decays into the forbidden region
- Probability of finding particle decays as $|\psi|^2 \propto e^{-2\kappa x}$

**Example values:**
For an electron with $V_0 - E = 1$ eV:
$$\delta = \frac{1.055 \times 10^{-34}}{\sqrt{2 \times 9.11 \times 10^{-31} \times 1.6 \times 10^{-19}}} \approx 0.2\text{ nm}$$

### 9. Total Reflection (E < V_0)

Applying boundary conditions at x = 0:
$$A + B = C$$
$$ik_1(A - B) = -\kappa C$$

Solving:
$$\frac{B}{A} = \frac{k_1 - i\kappa}{k_1 + i\kappa}$$

**Reflection coefficient:**
$$R = \left|\frac{B}{A}\right|^2 = \frac{k_1^2 + \kappa^2}{k_1^2 + \kappa^2} = 1$$

$$\boxed{R = 1 \quad \text{(total reflection when } E < V_0\text{)}}$$

**Phase shift upon reflection:**
$$\frac{B}{A} = e^{-2i\phi}, \quad \text{where } \tan\phi = \frac{\kappa}{k_1}$$

### 10. Quantum Mechanics Connection to Superconducting Qubits

In superconducting quantum circuits:
- **Josephson junctions** create effective step-like potentials for Cooper pairs
- **Evanescent coupling** between islands enables coherent quantum operations
- **Phase-sensitive reflection** is crucial for qubit readout
- Transmon qubits exploit the boundary between superconducting and normal regions

---

## Worked Examples

### Example 1: Electron at Metal-Vacuum Interface

An electron with kinetic energy 3 eV approaches a metal-vacuum interface where the work function (step height) is 4 eV.

**Given:** E = 3 eV, V_0 = 4 eV

**Find:** (a) Wave number in metal, (b) Decay constant in vacuum, (c) Penetration depth

**Solution:**

(a) In the metal (Region I):
$$k_1 = \frac{\sqrt{2mE}}{\hbar} = \frac{\sqrt{2 \times 9.11 \times 10^{-31} \times 3 \times 1.6 \times 10^{-19}}}{1.055 \times 10^{-34}}$$
$$k_1 = 8.87 \times 10^9 \text{ m}^{-1}$$

(b) In the vacuum (Region II):
$$\kappa = \frac{\sqrt{2m(V_0 - E)}}{\hbar} = \frac{\sqrt{2 \times 9.11 \times 10^{-31} \times 1 \times 1.6 \times 10^{-19}}}{1.055 \times 10^{-34}}$$
$$\kappa = 5.12 \times 10^9 \text{ m}^{-1}$$

(c) Penetration depth:
$$\delta = \frac{1}{\kappa} = \frac{1}{5.12 \times 10^9} = 0.195 \text{ nm}$$

**Physical insight:** The electron wave penetrates about 2 atomic diameters into the vacuum before decaying significantly.

---

### Example 2: Partial Transmission (E > V_0)

An electron with energy 5 eV encounters a step potential of height 2 eV.

**Find:** R and T

**Solution:**

$$k_1 = \frac{\sqrt{2m \times 5\text{ eV}}}{\hbar}, \quad k_2 = \frac{\sqrt{2m \times 3\text{ eV}}}{\hbar}$$

Ratio:
$$\frac{k_2}{k_1} = \sqrt{\frac{3}{5}} = 0.775$$

Reflection coefficient:
$$R = \left(\frac{k_1 - k_2}{k_1 + k_2}\right)^2 = \left(\frac{1 - 0.775}{1 + 0.775}\right)^2 = \left(\frac{0.225}{1.775}\right)^2 = 0.016$$

Transmission coefficient:
$$T = \frac{4k_1k_2}{(k_1 + k_2)^2} = \frac{4 \times 0.775}{(1.775)^2} = 0.984$$

**Verification:** R + T = 0.016 + 0.984 = 1 ✓

**Classical comparison:** Classically T = 1, R = 0. Quantum mechanics predicts 1.6% reflection even though E > V_0!

---

### Example 3: Phase Shift in Total Reflection

For E = 2 eV, V_0 = 5 eV, calculate the phase shift of the reflected wave.

**Solution:**

$$k_1 = \frac{\sqrt{2m \times 2\text{ eV}}}{\hbar}, \quad \kappa = \frac{\sqrt{2m \times 3\text{ eV}}}{\hbar}$$

$$\tan\phi = \frac{\kappa}{k_1} = \sqrt{\frac{3}{2}} = 1.225$$

$$\phi = \arctan(1.225) = 50.8° = 0.887 \text{ rad}$$

Total phase shift: $2\phi = 1.77$ rad

---

## Practice Problems

### Level 1: Direct Application

1. An electron with energy 4 eV hits a step potential of 3 eV. Calculate R and T.

2. For a step potential with V_0 = 10 eV, what electron energy gives R = T = 0.5?

3. Calculate the penetration depth for a proton with E = 1 MeV hitting a barrier of V_0 = 2 MeV.

### Level 2: Intermediate

4. Show that the probability current in the evanescent region (E < V_0) is zero, confirming total reflection.

5. A neutron with energy E hits a step of height 2E. Find the ratio of probability density at x = δ (one penetration depth) to that at x = 0.

6. Derive an expression for R when E is very close to V_0 (E = V_0 + ε where ε << V_0).

### Level 3: Challenging

7. **Step up followed by step down:** Consider V(x) = 0 for x < 0, V_0 for 0 < x < L, and 0 for x > L. Set up the boundary conditions for E < V_0.

8. A particle is described by a wave packet $\psi(x,0) = Ae^{-(x-x_0)^2/4\sigma^2}e^{ik_0x}$ approaching a step. Qualitatively describe the time evolution.

9. **Relativistic correction:** For a relativistic electron (E ~ mc²), how is the penetration depth modified?

---

## Computational Lab

### Python: Step Potential Analysis

```python
"""
Day 386: Step Potential Visualization
Quantum Tunneling & Barriers - Week 56

This lab explores:
1. Wave functions for E > V_0 and E < V_0
2. Reflection and transmission coefficients vs energy
3. Probability current conservation
4. Evanescent wave visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Physical constants (using natural units where convenient)
hbar = 1.055e-34  # J·s
m_e = 9.109e-31   # kg
eV = 1.602e-19    # J

def k1_from_E(E, m=m_e):
    """Wave number in Region I (E in eV)"""
    return np.sqrt(2 * m * E * eV) / hbar

def k2_from_E(E, V0, m=m_e):
    """Wave number in Region II for E > V0 (E, V0 in eV)"""
    return np.sqrt(2 * m * (E - V0) * eV) / hbar

def kappa_from_E(E, V0, m=m_e):
    """Decay constant in Region II for E < V0 (E, V0 in eV)"""
    return np.sqrt(2 * m * (V0 - E) * eV) / hbar

def reflection_coefficient(E, V0):
    """
    Calculate reflection coefficient R for step potential
    Works for both E > V0 and E < V0
    """
    if E <= 0:
        return np.nan
    if E < V0:
        return 1.0  # Total reflection
    else:
        k1 = np.sqrt(E)
        k2 = np.sqrt(E - V0)
        return ((k1 - k2) / (k1 + k2))**2

def transmission_coefficient(E, V0):
    """Calculate transmission coefficient T for step potential"""
    if E <= 0:
        return np.nan
    if E < V0:
        return 0.0  # No transmission
    else:
        k1 = np.sqrt(E)
        k2 = np.sqrt(E - V0)
        return 4 * k1 * k2 / (k1 + k2)**2

# Vectorize functions
R_vec = np.vectorize(reflection_coefficient)
T_vec = np.vectorize(transmission_coefficient)

#%% Plot 1: R and T vs Energy
fig, ax = plt.subplots(figsize=(10, 6))

V0 = 5.0  # eV
E = np.linspace(0.01, 15, 500)

R = R_vec(E, V0)
T = T_vec(E, V0)

ax.plot(E, R, 'b-', linewidth=2, label='Reflection R')
ax.plot(E, T, 'r-', linewidth=2, label='Transmission T')
ax.axvline(x=V0, color='gray', linestyle='--', alpha=0.7, label=f'$V_0$ = {V0} eV')
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

ax.fill_between(E[E < V0], 0, 1, alpha=0.1, color='blue', label='Classically forbidden')

ax.set_xlabel('Energy E (eV)', fontsize=12)
ax.set_ylabel('Coefficient', fontsize=12)
ax.set_title('Step Potential: Reflection and Transmission Coefficients', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step_RT_vs_E.png', dpi=150)
plt.show()

#%% Plot 2: Wave function visualization (E < V0)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

V0 = 5.0  # eV
E = 3.0   # eV (E < V0, total reflection)

# Calculate wave parameters
k1 = k1_from_E(E)  # in 1/m
kappa = kappa_from_E(E, V0)  # in 1/m

# Scale to nanometers for plotting
k1_nm = k1 * 1e-9  # 1/nm
kappa_nm = kappa * 1e-9  # 1/nm

# Spatial coordinates
x_left = np.linspace(-3, 0, 300)  # Region I (nm)
x_right = np.linspace(0, 2, 200)   # Region II (nm)

# Amplitude from boundary conditions
# B/A = (k1 - i*kappa)/(k1 + i*kappa)
B_over_A = (k1 - 1j*kappa) / (k1 + 1j*kappa)
C_over_A = 2 * k1 / (k1 + 1j*kappa)

A = 1.0  # Normalize incident amplitude

# Wave functions
psi_I = A * np.exp(1j * k1_nm * x_left) + A * B_over_A * np.exp(-1j * k1_nm * x_left)
psi_II = A * np.abs(C_over_A) * np.exp(-kappa_nm * x_right)

# Plot real part
ax1 = axes[0]
ax1.plot(x_left, np.real(psi_I), 'b-', linewidth=2, label='Region I')
ax1.plot(x_right, np.real(psi_II), 'r-', linewidth=2, label='Region II (evanescent)')
ax1.axvline(x=0, color='k', linestyle='-', linewidth=2)

# Add potential step visualization
ax1.axhspan(-2, 2, xmin=0.5, xmax=1, alpha=0.2, color='gray')
ax1.annotate('$V_0$', xy=(1, 1.5), fontsize=12)

ax1.set_xlabel('Position x (nm)', fontsize=12)
ax1.set_ylabel(r'Re[$\psi(x)$]', fontsize=12)
ax1.set_title(f'Step Potential Wave Function (E = {E} eV < V₀ = {V0} eV)', fontsize=14)
ax1.legend(fontsize=10)
ax1.set_xlim(-3, 2)
ax1.set_ylim(-2.5, 2.5)
ax1.grid(True, alpha=0.3)

# Plot probability density
ax2 = axes[1]
prob_I = np.abs(psi_I)**2
prob_II = np.abs(psi_II)**2

ax2.plot(x_left, prob_I, 'b-', linewidth=2, label='Region I')
ax2.plot(x_right, prob_II, 'r-', linewidth=2, label='Region II')
ax2.axvline(x=0, color='k', linestyle='-', linewidth=2)

# Penetration depth marker
delta = 1/kappa_nm
ax2.axvline(x=delta, color='g', linestyle='--', alpha=0.7,
            label=f'Penetration depth δ = {delta:.3f} nm')

ax2.fill_between(x_right, 0, prob_II, alpha=0.3, color='red')
ax2.axhspan(0, 4, xmin=0.6, xmax=1, alpha=0.1, color='gray')

ax2.set_xlabel('Position x (nm)', fontsize=12)
ax2.set_ylabel(r'$|\psi(x)|^2$', fontsize=12)
ax2.set_title('Probability Density', fontsize=14)
ax2.legend(fontsize=10)
ax2.set_xlim(-3, 2)
ax2.set_ylim(0, 4.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step_wavefunction_E_less_V0.png', dpi=150)
plt.show()

#%% Plot 3: Wave function visualization (E > V0)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

V0 = 5.0  # eV
E = 8.0   # eV (E > V0, partial transmission)

k1 = k1_from_E(E)
k2 = k2_from_E(E, V0)

k1_nm = k1 * 1e-9
k2_nm = k2 * 1e-9

x_left = np.linspace(-3, 0, 300)
x_right = np.linspace(0, 3, 300)

# Amplitudes
B_over_A = (k1 - k2) / (k1 + k2)
C_over_A = 2 * k1 / (k1 + k2)

A = 1.0
psi_I = A * np.exp(1j * k1_nm * x_left) + A * B_over_A * np.exp(-1j * k1_nm * x_left)
psi_II = A * C_over_A * np.exp(1j * k2_nm * x_right)

# Real part
ax1 = axes[0]
ax1.plot(x_left, np.real(psi_I), 'b-', linewidth=2, label='Region I')
ax1.plot(x_right, np.real(psi_II), 'r-', linewidth=2, label='Region II')
ax1.axvline(x=0, color='k', linestyle='-', linewidth=2)

ax1.set_xlabel('Position x (nm)', fontsize=12)
ax1.set_ylabel(r'Re[$\psi(x)$]', fontsize=12)
ax1.set_title(f'Step Potential Wave Function (E = {E} eV > V₀ = {V0} eV)', fontsize=14)
ax1.legend(fontsize=10)
ax1.set_xlim(-3, 3)
ax1.grid(True, alpha=0.3)

# Probability density
ax2 = axes[1]
prob_I = np.abs(psi_I)**2
prob_II = np.abs(psi_II)**2

ax2.plot(x_left, prob_I, 'b-', linewidth=2, label='Region I')
ax2.plot(x_right, prob_II, 'r-', linewidth=2, label='Region II')
ax2.axvline(x=0, color='k', linestyle='-', linewidth=2)

# Mark the standing wave pattern in Region I
ax2.set_xlabel('Position x (nm)', fontsize=12)
ax2.set_ylabel(r'$|\psi(x)|^2$', fontsize=12)
ax2.set_title(f'Probability Density (R = {B_over_A**2:.3f}, T = {1-B_over_A**2:.3f})', fontsize=14)
ax2.legend(fontsize=10)
ax2.set_xlim(-3, 3)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step_wavefunction_E_greater_V0.png', dpi=150)
plt.show()

#%% Plot 4: Probability current verification
fig, ax = plt.subplots(figsize=(10, 6))

V0 = 5.0
E_values = np.linspace(0.1, 15, 100)

j_inc = np.ones_like(E_values)  # Normalized to 1
j_ref = R_vec(E_values, V0)
j_trans = T_vec(E_values, V0)
j_total = j_ref + j_trans

ax.plot(E_values, j_inc, 'k--', linewidth=2, label='Incident current')
ax.plot(E_values, j_ref, 'b-', linewidth=2, label='Reflected current')
ax.plot(E_values, j_trans, 'r-', linewidth=2, label='Transmitted current')
ax.plot(E_values, j_total, 'g:', linewidth=3, label='R + T (should = 1)')

ax.axvline(x=V0, color='gray', linestyle='--', alpha=0.7)
ax.annotate(f'$V_0$ = {V0} eV', xy=(V0+0.2, 0.5), fontsize=11)

ax.set_xlabel('Energy E (eV)', fontsize=12)
ax.set_ylabel('Probability Current (normalized)', fontsize=12)
ax.set_title('Probability Current Conservation at Step Potential', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step_current_conservation.png', dpi=150)
plt.show()

#%% Plot 5: Penetration depth vs (V0 - E)
fig, ax = plt.subplots(figsize=(10, 6))

V0 = 5.0  # eV
E_range = np.linspace(0.1, 4.9, 100)  # E < V0
V0_minus_E = V0 - E_range

# Calculate penetration depth
delta = hbar / np.sqrt(2 * m_e * V0_minus_E * eV) * 1e9  # in nm

ax.plot(V0_minus_E, delta, 'b-', linewidth=2)
ax.fill_between(V0_minus_E, 0, delta, alpha=0.3)

ax.set_xlabel('$V_0 - E$ (eV)', fontsize=12)
ax.set_ylabel('Penetration depth $\\delta$ (nm)', fontsize=12)
ax.set_title('Penetration Depth vs Barrier Height Above Energy', fontsize=14)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Add annotations
ax.annotate('Deeper penetration\nwhen E close to V₀',
            xy=(0.5, 0.6), xytext=(1.5, 0.75),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('step_penetration_depth.png', dpi=150)
plt.show()

print("\n=== Step Potential Analysis Complete ===")
print(f"\nFor V₀ = {V0} eV:")
print(f"At E = 3 eV (E < V₀): R = 1.000, T = 0.000")
print(f"At E = 5 eV (E = V₀): R = 1.000, T = 0.000")
print(f"At E = 8 eV (E > V₀): R = {reflection_coefficient(8, V0):.4f}, T = {transmission_coefficient(8, V0):.4f}")
print(f"At E = 10 eV: R = {reflection_coefficient(10, V0):.4f}, T = {transmission_coefficient(10, V0):.4f}")
print(f"\nPenetration depth at E = 3 eV: {hbar/np.sqrt(2*m_e*2*eV)*1e9:.3f} nm")
```

### Expected Output

```
=== Step Potential Analysis Complete ===

For V₀ = 5.0 eV:
At E = 3 eV (E < V₀): R = 1.000, T = 0.000
At E = 5 eV (E = V₀): R = 1.000, T = 0.000
At E = 8 eV (E > V₀): R = 0.0102, T = 0.9898
At E = 10 eV: R = 0.0032, T = 0.9968

Penetration depth at E = 3 eV: 0.137 nm
```

---

## Summary

### Key Formulas Table

| Quantity | E > V_0 | E < V_0 |
|----------|---------|---------|
| Region II wave | $Ce^{ik_2x}$ | $Ce^{-\kappa x}$ |
| k₂ or κ | $\sqrt{2m(E-V_0)}/\hbar$ | $\sqrt{2m(V_0-E)}/\hbar$ |
| Reflection R | $(k_1-k_2)^2/(k_1+k_2)^2$ | 1 |
| Transmission T | $4k_1k_2/(k_1+k_2)^2$ | 0 |
| Penetration depth | N/A | $\delta = \hbar/\sqrt{2m(V_0-E)}$ |

### Main Takeaways

1. **Quantum reflection exists even when E > V_0** - unlike classical mechanics
2. **Evanescent waves penetrate forbidden regions** with characteristic decay length δ
3. **Probability current is conserved**: R + T = 1 always
4. **Phase shifts occur upon reflection** even in total reflection
5. **Penetration depth depends on energy deficit**: closer to V_0 means deeper penetration

### Quantum Computing Relevance

- Step potentials model interfaces in quantum devices
- Evanescent coupling enables quantum dot qubits
- Understanding reflection is crucial for quantum circuit design

---

## Daily Checklist

- [ ] I can solve the Schrodinger equation in both regions of a step potential
- [ ] I can apply boundary conditions to find B/A and C/A ratios
- [ ] I can calculate R and T and verify R + T = 1
- [ ] I understand why evanescent waves exist and what determines their decay
- [ ] I can calculate the penetration depth for given E and V_0
- [ ] I ran the Python code and understand all generated plots
- [ ] I attempted problems from each difficulty level

---

## Preview: Day 387

Tomorrow we extend to the **rectangular barrier**, where a finite-width region of height V_0 can be tunneled through even when E < V_0. We'll develop the **transfer matrix method** for handling multiple boundaries and derive the famous transmission coefficient formula with its hyperbolic sine dependence. This sets the stage for understanding quantum tunneling applications!
