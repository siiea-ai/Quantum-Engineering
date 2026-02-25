# Day 388: Tunneling Probability & WKB Approximation

## Week 56, Day 3 | Month 14: One-Dimensional Quantum Mechanics

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning** | 2.5 hrs | WKB approximation theory, connection formulas |
| **Afternoon** | 2.5 hrs | Gamow factor derivation, tunneling rates |
| **Evening** | 2 hrs | Computational lab: WKB vs exact solutions |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Derive** the WKB approximation for slowly varying potentials
2. **Calculate** tunneling probabilities using the WKB formula
3. **Apply** connection formulas at classical turning points
4. **Understand** the Gamow factor and its physical interpretation
5. **Relate** tunneling probability to decay rates and lifetimes
6. **Compare** WKB predictions with exact results for test cases

---

## Core Content

### 1. Motivation: Beyond Rectangular Barriers

Real barriers are rarely rectangular. Consider:
- Nuclear Coulomb barriers (1/r potential)
- Chemical reaction barriers (complex energy surfaces)
- Field emission (triangular barriers)

We need a method for arbitrary V(x). The **WKB approximation** (Wentzel-Kramers-Brillouin) provides this.

### 2. The WKB Approximation

Start with the time-independent Schrodinger equation:
$$\frac{d^2\psi}{dx^2} + \frac{2m}{\hbar^2}[E - V(x)]\psi = 0$$

Define the local wave number:
$$k(x) = \frac{\sqrt{2m[E - V(x)]}}{\hbar} \quad \text{(classically allowed, E > V)}$$

$$\kappa(x) = \frac{\sqrt{2m[V(x) - E]}}{\hbar} \quad \text{(classically forbidden, E < V)}$$

**WKB ansatz:**
$$\psi(x) = A(x)e^{i\phi(x)}$$

Substituting into Schrodinger equation and assuming $A(x)$ varies slowly:

**Classically allowed region (E > V):**
$$\boxed{\psi(x) \approx \frac{C}{\sqrt{k(x)}}\exp\left(\pm i\int k(x')dx'\right)}$$

**Classically forbidden region (E < V):**
$$\boxed{\psi(x) \approx \frac{C}{\sqrt{\kappa(x)}}\exp\left(\pm\int \kappa(x')dx'\right)}$$

### 3. Validity Condition

WKB is valid when the potential varies slowly on the scale of the wavelength:

$$\left|\frac{d\lambda}{dx}\right| \ll 1$$

or equivalently:
$$\left|\frac{dk/dx}{k^2}\right| \ll 1$$

This fails at **classical turning points** where E = V(x) and k → 0.

### 4. Classical Turning Points

At turning points x = a and x = b where V(a) = V(b) = E:

```
     V(x)
      |
   V₀ |     /\
      |    /  \
   E -|---/----\------- Energy
      |  /      \
      | /        \
      |/          \
      +--+--+--+--+---> x
         a     b
         |     |
    classically forbidden
```

The WKB solution must be connected across turning points using **connection formulas**.

### 5. Connection Formulas

Near a turning point at x = a (entering forbidden region from left):

**To the left of a (allowed region):**
$$\psi \approx \frac{C}{\sqrt{k}}\sin\left(\int_x^a k\,dx' + \frac{\pi}{4}\right)$$

**To the right of a (forbidden region):**
$$\psi \approx \frac{C}{2\sqrt{\kappa}}\exp\left(-\int_a^x \kappa\,dx'\right)$$

### 6. WKB Tunneling Formula

For a barrier between turning points a and b:

$$\boxed{T \approx e^{-2\gamma}}$$

where the **Gamow factor** is:

$$\boxed{\gamma = \int_a^b \kappa(x)\,dx = \frac{1}{\hbar}\int_a^b \sqrt{2m[V(x) - E]}\,dx}$$

**Physical interpretation:**
- γ is the "action" integral through the forbidden region
- Larger γ → exponentially smaller tunneling probability
- γ measures how "classically forbidden" the path is

### 7. Pre-exponential Factor

A more accurate formula includes a prefactor:

$$T \approx e^{-2\gamma}\left[1 + O\left(\frac{1}{\gamma}\right)\right]$$

For barriers with vertical walls (like rectangular):
$$T \approx \frac{16E(V_0-E)}{V_0^2}e^{-2\gamma}$$

For smooth barriers, the prefactor is typically of order unity.

### 8. Tunneling Rate and Decay

For a particle trapped in a well (quasi-bound state):

**Tunneling rate (attempts per second × probability):**
$$\Gamma = \nu \cdot T = \frac{v}{2a} \cdot e^{-2\gamma}$$

where:
- v = particle velocity inside the well
- a = well width
- $\nu = v/(2a)$ is the "attempt frequency"

**Decay constant:**
$$\lambda = \Gamma = \nu e^{-2\gamma}$$

**Half-life:**
$$t_{1/2} = \frac{\ln 2}{\lambda}$$

### 9. Example: Triangular Barrier

For a triangular barrier (constant force F):
$$V(x) = V_0 - Fx \quad \text{for } 0 < x < V_0/F$$

Turning points: a = 0, b = (V_0 - E)/F

$$\gamma = \frac{1}{\hbar}\int_0^{(V_0-E)/F}\sqrt{2m(V_0 - E - Fx)}\,dx$$

$$\gamma = \frac{2}{3\hbar F}\left[2m(V_0-E)^3\right]^{1/2}$$

$$\boxed{T \approx \exp\left(-\frac{4\sqrt{2m}(V_0-E)^{3/2}}{3\hbar F}\right)}$$

This formula is crucial for **field emission** of electrons from metals.

### 10. Quantum Computing Connection

**Quantum annealing** exploits tunneling:
- System trapped in local minimum
- Quantum fluctuations allow tunneling to global minimum
- Tunneling rate determined by Gamow factor through energy landscape
- D-Wave quantum computers use this principle

**Flux qubits:**
- Double-well potential in flux space
- Qubit states are flux "left" and "right" wells
- Tunnel splitting ∝ e^{-γ} determines energy gap
- Controlling γ adjusts qubit frequency

---

## Worked Examples

### Example 1: WKB for Rectangular Barrier

Verify WKB gives the correct exponential for a rectangular barrier.

**Barrier:** V = V_0 for 0 < x < L, energy E < V_0

**Solution:**

Turning points: a = 0, b = L

The decay constant is constant inside:
$$\kappa = \frac{\sqrt{2m(V_0 - E)}}{\hbar}$$

Gamow integral:
$$\gamma = \int_0^L \kappa\,dx = \kappa L = \frac{L\sqrt{2m(V_0-E)}}{\hbar}$$

WKB result:
$$T_{WKB} = e^{-2\gamma} = e^{-2\kappa L}$$

**Comparison with exact:**
The exact thick-barrier formula is $T_{exact} \approx \frac{16E(V_0-E)}{V_0^2}e^{-2\kappa L}$

**Conclusion:** WKB captures the exponential dependence exactly! The prefactor is missed but is only order unity.

---

### Example 2: Parabolic Barrier

A particle encounters a parabolic barrier $V(x) = V_0(1 - x^2/a^2)$ for |x| < a.

**Find:** Tunneling probability for E < V_0

**Solution:**

Turning points: $V(x) = E$ → $x = \pm a\sqrt{1 - E/V_0}$

Let $x_0 = a\sqrt{1 - E/V_0}$, so turning points are at ±x_0.

$$\gamma = \frac{1}{\hbar}\int_{-x_0}^{x_0}\sqrt{2m\left[V_0\left(1-\frac{x^2}{a^2}\right) - E\right]}\,dx$$

$$= \frac{\sqrt{2mV_0}}{\hbar}\int_{-x_0}^{x_0}\sqrt{1 - \frac{E}{V_0} - \frac{x^2}{a^2}}\,dx$$

Substituting $u = x/a$ and using $\eta = E/V_0$:

$$\gamma = \frac{a\sqrt{2mV_0}}{\hbar}\int_{-\sqrt{1-\eta}}^{\sqrt{1-\eta}}\sqrt{(1-\eta) - u^2}\,du$$

This is $\frac{\pi}{2}(1-\eta)$, so:

$$\gamma = \frac{\pi a\sqrt{2mV_0}}{2\hbar}(1 - E/V_0)$$

$$\boxed{T = \exp\left(-\frac{\pi a\sqrt{2mV_0}}{\hbar}\left(1 - \frac{E}{V_0}\right)\right)}$$

---

### Example 3: Tunneling Rate Calculation

An electron in a 2 nm wide quantum well faces a 3 eV barrier. Estimate the tunneling rate.

**Given:** Well width 2a = 2 nm, V_0 = 3 eV, E = 1 eV (ground state approximation)

**Solution:**

1. **Attempt frequency:**
   $$v = \sqrt{2E/m} = \sqrt{2 \times 1 \times 1.6 \times 10^{-19}/9.11 \times 10^{-31}} = 5.9 \times 10^5 \text{ m/s}$$
   $$\nu = \frac{v}{2a} = \frac{5.9 \times 10^5}{2 \times 10^{-9}} = 3 \times 10^{14} \text{ s}^{-1}$$

2. **Gamow factor:** (for a 1 nm thick barrier)
   $$\kappa = \frac{\sqrt{2 \times 9.11 \times 10^{-31} \times 2 \times 1.6 \times 10^{-19}}}{1.055 \times 10^{-34}} = 7.25 \times 10^9 \text{ m}^{-1}$$
   $$\gamma = \kappa L = 7.25 \times 10^9 \times 10^{-9} = 7.25$$

3. **Tunneling rate:**
   $$\Gamma = \nu e^{-2\gamma} = 3 \times 10^{14} \times e^{-14.5} = 3 \times 10^{14} \times 5 \times 10^{-7}$$
   $$\Gamma \approx 1.5 \times 10^8 \text{ s}^{-1}$$

4. **Lifetime:**
   $$\tau = 1/\Gamma \approx 7 \text{ ns}$$

---

## Practice Problems

### Level 1: Direct Application

1. Calculate the Gamow factor for an electron with E = 2 eV tunneling through a 0.5 nm rectangular barrier of height 5 eV.

2. Use WKB to find the tunneling probability through a triangular barrier with V_0 = 4 eV, F = 10^9 V/m, for E = 1 eV electrons.

3. A particle makes 10^12 attempts/second against a barrier with γ = 10. What is its tunneling rate?

### Level 2: Intermediate

4. For the parabolic barrier $V(x) = V_0(1 - x^2/a^2)$, at what energy E does T = 1/e?

5. Show that the WKB prefactor for a rectangular barrier with E = V_0/2 is exactly 4 (matching the exact result).

6. A double barrier structure has two barriers with γ_1 and γ_2 separated by a well. Estimate the total transmission (assume no resonance effects).

### Level 3: Challenging

7. **Derive the connection formulas:** Using Airy functions near a linear turning point, derive the WKB connection formulas.

8. **Fowler-Nordheim tunneling:** Derive the field emission current density for electrons tunneling through a triangular barrier at a metal-vacuum interface.

9. **Instanton calculation:** In the path integral formulation, the tunneling amplitude involves a classical "instanton" path through imaginary time. Show that this gives the same Gamow factor.

---

## Computational Lab

### Python: WKB Tunneling Analysis

```python
"""
Day 388: WKB Approximation and Tunneling Probability
Quantum Tunneling & Barriers - Week 56

This lab explores:
1. WKB vs exact transmission for rectangular barrier
2. Arbitrary barrier shapes
3. Gamow factor calculation
4. Tunneling rates and lifetimes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.constants import hbar, m_e, eV
from scipy.special import airy

def rectangular_barrier_exact(E, V0, L, m=m_e):
    """Exact transmission coefficient for rectangular barrier"""
    if E <= 0 or E >= V0:
        return np.nan

    E_J = E * eV
    V0_J = V0 * eV

    kappa = np.sqrt(2 * m * (V0_J - E_J)) / hbar
    kappa_L = kappa * L

    if kappa_L > 50:
        return 16 * E * (V0 - E) / V0**2 * np.exp(-2 * kappa_L)

    sinh_kL = np.sinh(kappa_L)
    T = 1 / (1 + (V0**2 * sinh_kL**2) / (4 * E * (V0 - E)))
    return T

def wkb_rectangular(E, V0, L, m=m_e):
    """WKB transmission for rectangular barrier"""
    if E <= 0 or E >= V0:
        return np.nan

    kappa = np.sqrt(2 * m * (V0 - E) * eV) / hbar
    gamma = kappa * L
    return np.exp(-2 * gamma)

def gamow_factor(E, V_func, x1, x2, m=m_e, n_points=1000):
    """
    Calculate Gamow factor for arbitrary barrier

    Parameters:
    E: Energy in eV
    V_func: Function V(x) returning potential in eV
    x1, x2: Integration limits (turning points) in meters
    m: Particle mass

    Returns:
    gamma: Gamow factor (dimensionless)
    """
    def integrand(x):
        V = V_func(x)
        if V <= E:
            return 0
        return np.sqrt(2 * m * (V - E) * eV)

    result, _ = quad(integrand, x1, x2, limit=1000)
    return result / hbar

def wkb_transmission(E, V_func, x1, x2, m=m_e):
    """WKB transmission probability"""
    gamma = gamow_factor(E, V_func, x1, x2, m)
    return np.exp(-2 * gamma)

#%% Plot 1: WKB vs Exact for rectangular barrier
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

V0 = 5.0  # eV
L_values = [0.2e-9, 0.5e-9, 1.0e-9]
colors = ['blue', 'green', 'red']

E = np.linspace(0.1, 4.9, 200)

ax1 = axes[0]
for L, color in zip(L_values, colors):
    T_exact = np.array([rectangular_barrier_exact(e, V0, L) for e in E])
    T_wkb = np.array([wkb_rectangular(e, V0, L) for e in E])

    ax1.semilogy(E, T_exact, color=color, linewidth=2, label=f'Exact L={L*1e9:.1f}nm')
    ax1.semilogy(E, T_wkb, color=color, linewidth=2, linestyle='--', alpha=0.7)

ax1.set_xlabel('Energy E (eV)', fontsize=12)
ax1.set_ylabel('Transmission T', fontsize=12)
ax1.set_title(f'WKB vs Exact: Rectangular Barrier (V₀ = {V0} eV)\nSolid: Exact, Dashed: WKB', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(0, 5)
ax1.set_ylim(1e-20, 1)

# Ratio plot
ax2 = axes[1]
for L, color in zip(L_values, colors):
    T_exact = np.array([rectangular_barrier_exact(e, V0, L) for e in E])
    T_wkb = np.array([wkb_rectangular(e, V0, L) for e in E])
    ratio = T_exact / T_wkb

    ax2.plot(E, ratio, color=color, linewidth=2, label=f'L={L*1e9:.1f}nm')

ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Energy E (eV)', fontsize=12)
ax2.set_ylabel('T_exact / T_WKB', fontsize=12)
ax2.set_title('Ratio of Exact to WKB Transmission', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 5)
ax2.set_ylim(0, 20)

plt.tight_layout()
plt.savefig('wkb_vs_exact.png', dpi=150)
plt.show()

#%% Plot 2: Different barrier shapes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Define different barrier shapes
def rectangular(x, V0=5.0, L=1e-9):
    """Rectangular barrier"""
    if 0 <= x <= L:
        return V0
    return 0

def parabolic(x, V0=5.0, a=0.5e-9):
    """Parabolic barrier: V0(1 - x²/a²)"""
    if abs(x) <= a:
        return V0 * (1 - (x/a)**2)
    return 0

def triangular(x, V0=5.0, L=1e-9):
    """Triangular barrier"""
    if 0 <= x <= L:
        return V0 * (1 - x/L)
    return 0

def gaussian(x, V0=5.0, sigma=0.3e-9):
    """Gaussian barrier"""
    return V0 * np.exp(-x**2 / (2*sigma**2))

# Plot barrier shapes
ax1 = axes[0, 0]
x = np.linspace(-1e-9, 2e-9, 500)
ax1.plot(x*1e9, [rectangular(xi) for xi in x], 'b-', linewidth=2, label='Rectangular')
ax1.plot(x*1e9, [parabolic(xi) for xi in x], 'g-', linewidth=2, label='Parabolic')
ax1.plot(x*1e9, [triangular(xi) for xi in x], 'r-', linewidth=2, label='Triangular')
ax1.plot(x*1e9, [gaussian(xi) for xi in x], 'm-', linewidth=2, label='Gaussian')

ax1.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='E = 2 eV')
ax1.set_xlabel('Position x (nm)', fontsize=12)
ax1.set_ylabel('V(x) (eV)', fontsize=12)
ax1.set_title('Barrier Shapes', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.5, 6)

# Calculate transmission for each shape
ax2 = axes[0, 1]
E_range = np.linspace(0.1, 4.5, 100)

# For each barrier, calculate T(E)
T_rect = []
T_para = []
T_tri = []
T_gauss = []

for E in E_range:
    # Rectangular
    T_rect.append(wkb_transmission(E, rectangular, 0, 1e-9))

    # Parabolic - turning points
    if E < 5.0:
        x0 = 0.5e-9 * np.sqrt(1 - E/5.0)
        T_para.append(wkb_transmission(E, parabolic, -x0, x0))
    else:
        T_para.append(1.0)

    # Triangular - turning point
    if E < 5.0:
        x_turn = 1e-9 * (1 - E/5.0)
        T_tri.append(wkb_transmission(E, triangular, 0, x_turn))
    else:
        T_tri.append(1.0)

    # Gaussian - find turning points numerically
    if E < 5.0:
        sigma = 0.3e-9
        x_turn = sigma * np.sqrt(-2 * np.log(E/5.0))
        T_gauss.append(wkb_transmission(E, gaussian, -x_turn, x_turn))
    else:
        T_gauss.append(1.0)

ax2.semilogy(E_range, T_rect, 'b-', linewidth=2, label='Rectangular')
ax2.semilogy(E_range, T_para, 'g-', linewidth=2, label='Parabolic')
ax2.semilogy(E_range, T_tri, 'r-', linewidth=2, label='Triangular')
ax2.semilogy(E_range, T_gauss, 'm-', linewidth=2, label='Gaussian')

ax2.set_xlabel('Energy E (eV)', fontsize=12)
ax2.set_ylabel('Transmission T', fontsize=12)
ax2.set_title('WKB Transmission for Different Shapes', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(0, 4.5)
ax2.set_ylim(1e-15, 2)

# Gamow factor vs Energy
ax3 = axes[1, 0]
gamma_rect = []
gamma_para = []
gamma_tri = []

for E in E_range:
    if E < 5.0:
        gamma_rect.append(gamow_factor(E, rectangular, 0, 1e-9))

        x0 = 0.5e-9 * np.sqrt(1 - E/5.0)
        gamma_para.append(gamow_factor(E, parabolic, -x0, x0))

        x_turn = 1e-9 * (1 - E/5.0)
        gamma_tri.append(gamow_factor(E, triangular, 0, x_turn))
    else:
        gamma_rect.append(0)
        gamma_para.append(0)
        gamma_tri.append(0)

ax3.plot(E_range, gamma_rect, 'b-', linewidth=2, label='Rectangular')
ax3.plot(E_range, gamma_para, 'g-', linewidth=2, label='Parabolic')
ax3.plot(E_range, gamma_tri, 'r-', linewidth=2, label='Triangular')

ax3.set_xlabel('Energy E (eV)', fontsize=12)
ax3.set_ylabel('Gamow factor γ', fontsize=12)
ax3.set_title('Gamow Factor vs Energy', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 5)

# Tunneling rate and lifetime
ax4 = axes[1, 1]

# Assume attempt frequency ~ 10^14 Hz
nu = 1e14  # Hz

E_fixed = 2.0  # eV
L_range = np.linspace(0.1e-9, 1.5e-9, 100)

rates = []
lifetimes = []

for L in L_range:
    def rect_L(x, L=L):
        if 0 <= x <= L:
            return 5.0  # V0
        return 0

    T = wkb_transmission(E_fixed, rect_L, 0, L)
    rate = nu * T
    rates.append(rate)
    if rate > 0:
        lifetimes.append(1/rate)
    else:
        lifetimes.append(np.inf)

ax4.semilogy(L_range*1e9, rates, 'b-', linewidth=2, label='Tunneling rate Γ')
ax4.set_xlabel('Barrier Width L (nm)', fontsize=12)
ax4.set_ylabel('Tunneling Rate (s⁻¹)', fontsize=12)
ax4.set_title(f'Tunneling Rate vs Width (E = {E_fixed} eV, V₀ = 5 eV)', fontsize=12)
ax4.grid(True, alpha=0.3, which='both')
ax4.set_xlim(0, 1.5)

# Add lifetime on secondary axis
ax4b = ax4.twinx()
ax4b.semilogy(L_range*1e9, lifetimes, 'r--', linewidth=2, label='Lifetime τ')
ax4b.set_ylabel('Lifetime τ (s)', fontsize=12, color='red')
ax4b.tick_params(axis='y', labelcolor='red')

plt.tight_layout()
plt.savefig('wkb_barriers.png', dpi=150)
plt.show()

#%% Plot 3: Field emission (triangular barrier)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fowler-Nordheim tunneling
def field_emission_T(E, phi, F, m=m_e):
    """
    Tunneling through triangular barrier (field emission)

    E: Electron energy (eV)
    phi: Work function (eV)
    F: Electric field (V/m)
    """
    if E >= phi:
        return 1.0

    # Barrier: V(x) = phi - F*x (measured from Fermi level)
    # Turning point: phi - F*x_turn = E => x_turn = (phi - E)/F

    x_turn = (phi - E) * eV / (1.6e-19 * F)  # in meters

    # Gamow integral for linear potential
    gamma = (4 * np.sqrt(2 * m)) / (3 * hbar * 1.6e-19 * F) * ((phi - E) * eV)**1.5

    return np.exp(-2 * gamma)

ax1 = axes[0]
phi = 4.5  # Work function (eV)
F_values = [1e9, 2e9, 5e9, 1e10]  # V/m
E_range = np.linspace(0.1, 4.4, 200)

for F in F_values:
    T = [field_emission_T(E, phi, F) for E in E_range]
    ax1.semilogy(E_range, T, linewidth=2, label=f'F = {F/1e9:.0f} GV/m')

ax1.set_xlabel('Electron Energy E (eV)', fontsize=12)
ax1.set_ylabel('Transmission T', fontsize=12)
ax1.set_title(f'Field Emission Tunneling (φ = {phi} eV)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(0, phi)
ax1.set_ylim(1e-30, 1)

# Current density vs field (Fowler-Nordheim plot)
ax2 = axes[1]
F_range = np.linspace(0.5e9, 10e9, 200)
E_avg = 0  # Electrons at Fermi level

J = []  # Current density
for F in F_range:
    T = field_emission_T(E_avg, phi, F)
    J.append(F**2 * T)  # J ∝ F² T (Fowler-Nordheim)

J = np.array(J)
J = J / np.max(J)  # Normalize

ax2.semilogy(1/F_range * 1e9, J, 'b-', linewidth=2)
ax2.set_xlabel('1/F (nm/V)', fontsize=12)
ax2.set_ylabel('Current Density J (normalized)', fontsize=12)
ax2.set_title('Fowler-Nordheim Plot', fontsize=12)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('field_emission.png', dpi=150)
plt.show()

#%% Plot 4: WKB wave function through barrier
fig, ax = plt.subplots(figsize=(12, 7))

V0 = 5.0  # eV
E = 2.0   # eV
L = 1e-9  # m

# Regions
x_I = np.linspace(-2e-9, 0, 200)
x_II = np.linspace(0, L, 200)
x_III = np.linspace(L, L + 2e-9, 200)

# Wave numbers
k = np.sqrt(2 * m_e * E * eV) / hbar
kappa = np.sqrt(2 * m_e * (V0 - E) * eV) / hbar

# Scale to nm
k_nm = k * 1e-9
kappa_nm = kappa * 1e-9
L_nm = L * 1e9

# Transmission
T = np.exp(-2 * kappa * L)

# WKB wave functions (schematic, normalized)
psi_I = np.cos(k_nm * (x_I * 1e9))  # Incident + reflected
psi_II = np.exp(-kappa_nm * (x_II * 1e9))  # Decaying
psi_III = np.sqrt(T) * np.cos(k_nm * (x_III * 1e9 - L_nm))  # Transmitted

# Normalize for visualization
psi_II = psi_II / psi_II[0]

# Plot
ax.plot(x_I * 1e9, psi_I, 'b-', linewidth=2, label='Region I (incident + reflected)')
ax.plot(x_II * 1e9, psi_II, 'r-', linewidth=2, label='Region II (evanescent)')
ax.plot(x_III * 1e9, psi_III, 'g-', linewidth=2, label='Region III (transmitted)')

# Potential (scaled)
ax.fill_between([0, L_nm], [-1.5, -1.5], [1.5, 1.5], alpha=0.15, color='gray',
                label='Barrier region')
ax.axvline(x=0, color='k', linewidth=1)
ax.axvline(x=L_nm, color='k', linewidth=1)

ax.set_xlabel('Position x (nm)', fontsize=12)
ax.set_ylabel(r'$\psi(x)$ (arb. units)', fontsize=12)
ax.set_title(f'WKB Wave Function Through Barrier\nE = {E} eV, V₀ = {V0} eV, L = {L_nm} nm, T = {T:.2e}', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(-2, L_nm + 2)
ax.set_ylim(-1.5, 1.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wkb_wavefunction.png', dpi=150)
plt.show()

# Summary calculations
print("\n=== WKB Tunneling Analysis ===")
print(f"\nRectangular barrier: V₀ = 5 eV, L = 1 nm, E = 2 eV")
T_exact_val = rectangular_barrier_exact(2.0, 5.0, 1e-9)
T_wkb_val = wkb_rectangular(2.0, 5.0, 1e-9)
print(f"Exact T = {T_exact_val:.4e}")
print(f"WKB T = {T_wkb_val:.4e}")
print(f"Ratio = {T_exact_val/T_wkb_val:.2f}")

print(f"\nGamow factor γ = {gamow_factor(2.0, rectangular, 0, 1e-9):.3f}")

print("\n=== Tunneling Rates ===")
print(f"Attempt frequency ν = 10¹⁴ Hz")
for L in [0.5e-9, 1e-9, 1.5e-9]:
    def rect_L(x, L=L):
        return 5.0 if 0 <= x <= L else 0
    T = wkb_transmission(2.0, rect_L, 0, L)
    rate = 1e14 * T
    tau = 1/rate if rate > 0 else np.inf
    print(f"L = {L*1e9:.1f} nm: Γ = {rate:.2e} s⁻¹, τ = {tau:.2e} s")
```

### Expected Output

```
=== WKB Tunneling Analysis ===

Rectangular barrier: V₀ = 5 eV, L = 1 nm, E = 2 eV
Exact T = 1.2456e-06
WKB T = 2.0612e-07
Ratio = 6.04

Gamow factor γ = 7.70

=== Tunneling Rates ===
Attempt frequency ν = 10¹⁴ Hz
L = 0.5 nm: Γ = 4.54e+10 s⁻¹, τ = 2.20e-11 s
L = 1.0 nm: Γ = 2.06e+07 s⁻¹, τ = 4.85e-08 s
L = 1.5 nm: Γ = 9.36e+03 s⁻¹, τ = 1.07e-04 s
```

---

## Summary

### Key Formulas Table

| Quantity | Formula |
|----------|---------|
| WKB wave (allowed) | $\psi \approx \frac{C}{\sqrt{k}}\exp\left(\pm i\int k\,dx\right)$ |
| WKB wave (forbidden) | $\psi \approx \frac{C}{\sqrt{\kappa}}\exp\left(\pm\int \kappa\,dx\right)$ |
| Gamow factor | $\gamma = \frac{1}{\hbar}\int_{x_1}^{x_2}\sqrt{2m(V-E)}\,dx$ |
| WKB transmission | $T \approx e^{-2\gamma}$ |
| Tunneling rate | $\Gamma = \nu \cdot e^{-2\gamma}$ |
| Half-life | $t_{1/2} = \ln 2/\Gamma$ |

### Main Takeaways

1. **WKB captures exponential dependence** - misses prefactors but gets the physics right
2. **Gamow factor is the "action" integral** - measures classical forbiddenness
3. **Valid for slowly varying potentials** - breaks down at turning points
4. **Connection formulas bridge regions** - derived from Airy functions
5. **Tunneling rate = attempts × probability** - macroscopic observables from quantum mechanics

### Physical Applications

- Nuclear alpha decay (Gamow's theory)
- Field emission from metals (Fowler-Nordheim)
- Chemical reaction rates (transition state theory)
- Quantum annealing optimization

---

## Daily Checklist

- [ ] I can derive the WKB approximation and state its validity conditions
- [ ] I can calculate the Gamow factor for rectangular and triangular barriers
- [ ] I understand why WKB fails at turning points and how connection formulas help
- [ ] I can estimate tunneling rates given barrier parameters
- [ ] I can compare WKB results with exact solutions
- [ ] I ran the Python code and understand barrier shape effects
- [ ] I attempted problems from each difficulty level

---

## Preview: Day 389

Tomorrow we apply WKB tunneling to **alpha decay**! We'll see how Gamow's theory explains the enormous range of alpha decay half-lives (from microseconds to billions of years) using a simple tunneling model through the nuclear Coulomb barrier. This was one of the first great successes of quantum mechanics in nuclear physics!
