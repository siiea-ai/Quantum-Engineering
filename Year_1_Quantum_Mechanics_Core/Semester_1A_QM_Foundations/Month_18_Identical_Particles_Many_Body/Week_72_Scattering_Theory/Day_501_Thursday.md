# Day 501: Low-Energy Scattering

## Overview

**Day 501 of 2520 | Week 72, Day 4 | Month 18: Identical Particles & Many-Body Physics**

Today we focus on low-energy scattering, where the de Broglie wavelength exceeds the range of the potential. In this regime, only s-wave scattering contributes, and the physics is characterized by a single parameter: the **scattering length**. This universal behavior underlies phenomena from neutron-proton interactions to ultracold atomic gases. We also explore the effective range expansion and the deep connection between scattering properties and bound states.

---

## Schedule

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM | s-Wave Dominance at Low Energy | 60 min |
| 10:00 AM | The Scattering Length | 90 min |
| 11:30 AM | Break | 15 min |
| 11:45 AM | Effective Range Expansion | 75 min |
| 1:00 PM | Lunch | 60 min |
| 2:00 PM | Bound States and Scattering | 90 min |
| 3:30 PM | Break | 15 min |
| 3:45 PM | Applications to Cold Atoms | 60 min |
| 4:45 PM | Computational Lab | 75 min |
| 6:00 PM | Summary & Reflection | 30 min |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** why s-wave dominates at low energies
2. **Define** and calculate the scattering length a
3. **Apply** the effective range expansion
4. **Connect** scattering length to bound state properties
5. **Distinguish** positive and negative scattering lengths
6. **Apply** these concepts to ultracold atomic physics

---

## 1. s-Wave Dominance

### Why Only s-Wave at Low Energy?

For scattering from a potential of range $R$, the centrifugal barrier for partial wave $\ell$ is:

$$V_{eff}(r) = V(r) + \frac{\hbar^2\ell(\ell+1)}{2mr^2}$$

At the edge of the potential ($r \sim R$):

$$E_{barrier} \sim \frac{\hbar^2\ell(\ell+1)}{2mR^2}$$

For a particle with kinetic energy $E = \hbar^2k^2/2m$ and $kR \ll 1$:

$$\frac{E}{E_{barrier}} \sim \frac{(kR)^2}{\ell(\ell+1)} \ll 1 \quad \text{for } \ell \geq 1$$

**Result:** Higher partial waves cannot penetrate the centrifugal barrier. Only $\ell = 0$ (s-wave) contributes.

### Quantitative Criterion

$$\boxed{\text{s-wave dominance: } kR \ll 1}$$

where $R$ is the potential range.

Equivalently: $\lambda = 2\pi/k \gg 2\pi R$ (wavelength much larger than potential).

### Phase Shift Behavior

At low energy, phase shifts scale as:

$$\boxed{\delta_\ell \propto k^{2\ell+1} \quad (k \to 0)}$$

Proof: The radial wave function for $\ell > 0$ must vanish at the origin, and the probability of penetrating to small $r$ scales as $k^{2\ell}$.

**Examples:**
- $\delta_0 \propto k$ (s-wave)
- $\delta_1 \propto k^3$ (p-wave)
- $\delta_2 \propto k^5$ (d-wave)

---

## 2. The Scattering Length

### Definition

For s-wave scattering at low energy:

$$k\cot\delta_0 \xrightarrow{k \to 0} -\frac{1}{a}$$

The **scattering length** $a$ is defined by:

$$\boxed{\delta_0 \approx -ka \quad (k \to 0)}$$

### Physical Interpretation

The scattering length is the intercept where the asymptotic wave function extrapolates to zero:

For $r$ outside the potential range:
$$u_0(r) = A\sin(kr + \delta_0) \approx A \cdot k(r - a) \quad (kr \ll 1)$$

The wave function appears to vanish at $r = a$.

### Sign of Scattering Length

**$a > 0$ (positive):**
- Wave function appears to vanish at positive $r$
- Typically repulsive interactions at low energy
- Example: Hard sphere gives $a = R$

**$a < 0$ (negative):**
- Wave function extrapolates to zero at negative $r$
- Attractive interactions
- A bound state may be present or nearby

### Low-Energy Cross Section

$$f_0 \approx \frac{\sin\delta_0}{k} \approx \frac{-ka}{k} = -a \quad (k \to 0)$$

$$\boxed{\sigma_{tot} \approx 4\pi a^2 \quad (k \to 0)}$$

The low-energy cross section is completely determined by the scattering length!

---

## 3. Effective Range Expansion

### Beyond the Scattering Length

The next term in the low-energy expansion:

$$\boxed{k\cot\delta_0 = -\frac{1}{a} + \frac{1}{2}r_0 k^2 + O(k^4)}$$

where $r_0$ is the **effective range**.

### Physical Meaning

- **Scattering length $a$:** Strength of interaction
- **Effective range $r_0$:** Size of the interaction region

For short-range potentials, $r_0 \sim R$ (potential range).

### Partial Wave Amplitude

$$f_0 = \frac{1}{k\cot\delta_0 - ik} = \frac{1}{-\frac{1}{a} + \frac{1}{2}r_0k^2 - ik}$$

$$\boxed{f_0 = \frac{-a}{1 + ika - \frac{1}{2}ar_0k^2}}$$

### Zero-Energy Limit

$$f_0(k = 0) = -a$$

### Low-Energy Cross Section with Range Correction

$$\sigma_0 = 4\pi|f_0|^2 = \frac{4\pi a^2}{(1 - \frac{1}{2}ar_0k^2)^2 + (ka)^2}$$

$$\boxed{\sigma_0 \approx 4\pi a^2 \left(1 - \frac{1}{2}r_0 a k^2\right)^{-2} + O(k^4)}$$

---

## 4. Scattering and Bound States

### Connection Between Scattering and Binding

**Remarkable fact:** The scattering length contains information about bound states.

For a potential that just barely supports a bound state:

$$\boxed{a = \frac{\hbar}{\sqrt{2m|E_B|}}}$$

where $E_B$ is the binding energy.

### Derivation

At a bound state energy $E = -|E_B|$, the wave function decays as:
$$u(r) \propto e^{-\kappa r}, \quad \kappa = \sqrt{2m|E_B|}/\hbar$$

The same potential's scattering solution at $E \to 0$ has:
$$u(r) \propto r - a$$

Continuity of the logarithmic derivative at the potential edge gives:
$$-\kappa = -\frac{1}{a} \implies a = \frac{1}{\kappa}$$

### Three Scenarios

**1. Deep bound state ($a > 0$, small):**
- Binding energy large
- Scattering length positive but small
- Example: Deuteron's excited state would have this character

**2. Weakly bound state ($a > 0$, large):**
- $E_B \approx \hbar^2/(2ma^2)$ small
- Large positive scattering length
- **Universality:** Physics independent of potential details

**3. Virtual state ($a < 0$):**
- No bound state, but "almost bound"
- Large negative scattering length
- Example: Neutron-neutron (a ≈ -18 fm)

### Levinson's Theorem

$$\delta_0(0) - \delta_0(\infty) = n_0\pi$$

where $n_0$ is the number of s-wave bound states.

If there's one bound state: $\delta_0(0) = \pi$ (or $n\pi$).

---

## 5. Examples and Applications

### Hard Sphere

For a hard sphere of radius $R$:
- Boundary condition: $u_0(R) = 0$
- $\sin(kR + \delta_0) = 0 \implies \delta_0 = -kR$

$$\boxed{a = R \quad (\text{hard sphere})}$$

The scattering length equals the sphere radius.

### Square Well

$$V(r) = \begin{cases} -V_0 & r < R \\ 0 & r \geq R \end{cases}$$

Inside ($r < R$): $u = A\sin(Kr)$ where $K = \sqrt{k^2 + 2mV_0/\hbar^2}$

Outside ($r > R$): $u = B\sin(kr + \delta_0)$

Matching at $r = R$:
$$K\cot(KR) = k\cot(kR + \delta_0)$$

At $k \to 0$: $K_0 = \sqrt{2mV_0}/\hbar$

$$a = R\left(1 - \frac{\tan(K_0R)}{K_0R}\right)$$

**Critical behavior:** When $K_0R = \pi/2$, a new bound state appears and $a \to \pm\infty$.

### Neutron-Proton Scattering

**Singlet (antiparallel spins):** $a_s \approx -23.7$ fm (no bound state)

**Triplet (parallel spins):** $a_t \approx +5.4$ fm (deuteron bound state)

The difference shows how spin affects nuclear forces.

### Ultracold Atoms: Feshbach Resonances

Near a Feshbach resonance, the scattering length varies with magnetic field:

$$a(B) = a_{bg}\left(1 - \frac{\Delta}{B - B_0}\right)$$

where:
- $a_{bg}$ = background scattering length
- $B_0$ = resonance position
- $\Delta$ = resonance width

**Applications:**
- Tunable interactions in BEC
- Creating ultracold molecules
- Quantum simulation of many-body physics

---

## 6. Worked Examples

### Example 1: Extracting Scattering Length

**Problem:** Low-energy neutron-nucleus scattering gives a total cross section of 4.2 barns. What is the scattering length?

**Solution:**

At low energy: $\sigma = 4\pi a^2$

$$a^2 = \frac{\sigma}{4\pi} = \frac{4.2 \times 10^{-24}\text{ cm}^2}{4\pi}$$

$$a = \sqrt{\frac{4.2 \times 10^{-24}}{4\pi}}\text{ cm} = 5.78 \times 10^{-13}\text{ cm} = \boxed{5.78\text{ fm}}$$

### Example 2: Bound State Energy

**Problem:** A system has a scattering length $a = 100$ fm. Estimate the binding energy of the weakly bound state.

**Solution:**

$$E_B = \frac{\hbar^2}{2ma^2}$$

For nucleons ($m \approx 939$ MeV/c²):

$$E_B = \frac{(\hbar c)^2}{2mc^2 a^2} = \frac{(197\text{ MeV·fm})^2}{2 \times 939\text{ MeV} \times (100\text{ fm})^2}$$

$$E_B = \frac{38809}{18780000}\text{ MeV} \approx \boxed{2.1\text{ keV}}$$

This is extremely weakly bound!

### Example 3: Effective Range Correction

**Problem:** For neutron-proton triplet scattering, $a_t = 5.4$ fm and $r_0 = 1.7$ fm. At what energy does the effective range correction become 10% of the leading term?

**Solution:**

The effective range expansion:
$$k\cot\delta_0 = -\frac{1}{a} + \frac{1}{2}r_0 k^2$$

10% correction means:
$$\frac{\frac{1}{2}r_0 k^2}{1/a} = 0.1$$

$$k^2 = \frac{0.2}{r_0 a} = \frac{0.2}{1.7 \times 5.4}\text{ fm}^{-2} = 0.0218\text{ fm}^{-2}$$

$$k = 0.148\text{ fm}^{-1}$$

Energy: $E = \frac{\hbar^2 k^2}{2m} = \frac{(197)^2 \times 0.0218}{2 \times 939}\text{ MeV} = \boxed{0.45\text{ MeV}}$

Below ~0.5 MeV, the scattering length alone is sufficient.

---

## 7. Practice Problems

### Level 1: Direct Application

**Problem 1.1:** A potential has scattering length $a = -10$ fm. Calculate:
(a) The low-energy total cross section
(b) The s-wave phase shift at $k = 0.1$ fm⁻¹

**Problem 1.2:** For a hard sphere of radius $R = 2$ fm, calculate $\sigma_{tot}$ at:
(a) $k \to 0$
(b) $k = 1$ fm⁻¹ (including higher partial waves)

**Problem 1.3:** A weakly bound state has $E_B = 1$ keV. What is the scattering length for nucleon-nucleon interaction?

### Level 2: Intermediate

**Problem 2.1:** Show that for the square well potential at threshold ($k = 0$), a bound state appears when $K_0R = (2n+1)\pi/2$ and the scattering length diverges at these points.

**Problem 2.2:** Derive the effective range $r_0$ for a hard sphere. (Hint: expand the exact phase shift $\delta_0 = -kR$ to order $k^2$.)

**Problem 2.3:** The scattering length near a Feshbach resonance varies as $a(B) = a_{bg}(1 - \Delta/(B-B_0))$. At what magnetic field is the interaction effectively turned off ($a = 0$)?

### Level 3: Challenging

**Problem 3.1:** Using the relation between scattering length and bound state energy, show that the cross section for scattering from a potential with a weakly bound state scales as:
$$\sigma \approx \frac{2\pi\hbar^2}{mE_B}$$

**Problem 3.2:** Prove Levinson's theorem for s-waves: if $V(r) < 0$ everywhere and supports exactly one bound state, then $\delta_0(0) = \pi$.

**Problem 3.3:** In a BEC, the mean-field interaction energy is $g = 4\pi\hbar^2 a/m$. Near a Feshbach resonance, what happens to the BEC as $B \to B_0$? Discuss the stability for $a > 0$ vs $a < 0$.

---

## 8. Computational Lab: Low-Energy Scattering

```python
"""
Day 501 Computational Lab: Low-Energy Scattering
Exploring scattering length, effective range, and bound states.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Physical constants
hbar = 1.0
m = 1.0


def solve_radial_equation(V_func, k, r_max=50, r_match=30):
    """
    Solve the s-wave radial equation and extract phase shift.
    """
    def equation(r, y):
        u, up = y
        U = 2 * m * V_func(r) / hbar**2
        return [up, (U - k**2) * u]

    # Initial conditions: u(r) ~ r for small r
    r_start = 0.01
    y0 = [r_start, 1.0]

    sol = solve_ivp(equation, [r_start, r_max], y0,
                    method='RK45', dense_output=True,
                    rtol=1e-10, atol=1e-12)

    # Extract at matching radius
    r_m = r_match
    u, up = sol.sol(r_m)

    # Match to sin(kr + delta)
    # u = A*sin(kr + delta)
    # up = Ak*cos(kr + delta)
    # tan(kr + delta) = k*u/up

    if abs(k) < 1e-10:
        # k -> 0 limit: u ~ r - a
        return -u / up  # This is the scattering length
    else:
        tan_phase = k * u / up
        phase = np.arctan(tan_phase) - k * r_m
        return phase


def compute_scattering_length(V_func, k_values=None, r_max=50):
    """
    Extract scattering length by fitting low-k phase shifts.
    """
    if k_values is None:
        k_values = np.linspace(0.01, 0.2, 20)

    phase_shifts = []
    for k in k_values:
        try:
            delta = solve_radial_equation(V_func, k, r_max)
            phase_shifts.append(delta)
        except Exception:
            phase_shifts.append(np.nan)

    phase_shifts = np.array(phase_shifts)

    # Linear fit: delta = -k*a
    valid = ~np.isnan(phase_shifts)
    k_fit = k_values[valid]
    delta_fit = phase_shifts[valid]

    # Simple linear fit
    a = -np.mean(delta_fit / k_fit)

    return a, k_values, phase_shifts


def effective_range_fit(k_values, phase_shifts):
    """
    Fit to effective range expansion: k*cot(delta) = -1/a + r_0*k^2/2
    """
    k_cot_delta = k_values / np.tan(phase_shifts)

    # Fit to -1/a + r_0*k^2/2
    # y = A + B*x where x = k^2, y = k*cot(delta)
    x = k_values**2
    y = k_cot_delta

    # Linear regression
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)

    A = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x**2)
    B = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

    a = -1 / A
    r0 = 2 * B

    return a, r0


def square_well(r, V0, R):
    """Square well potential."""
    return -V0 if r < R else 0


def gaussian_potential(r, V0, sigma):
    """Gaussian potential."""
    return -V0 * np.exp(-(r/sigma)**2)


def plot_scattering_length_vs_depth():
    """Show how scattering length varies with potential depth."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    R = 1.0
    V0_values = np.linspace(0.5, 15, 100)
    scattering_lengths = []

    for V0 in V0_values:
        V_func = lambda r, V0=V0: square_well(r, V0, R)
        try:
            a, _, _ = compute_scattering_length(V_func)
            scattering_lengths.append(a)
        except Exception:
            scattering_lengths.append(np.nan)

    ax1 = axes[0]
    ax1.plot(V0_values, scattering_lengths, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(y=R, color='r', linestyle=':', alpha=0.5,
                label=f'Hard sphere limit (a = R)')

    # Mark bound state thresholds
    K0_R_critical = [np.pi/2, 3*np.pi/2, 5*np.pi/2]
    for i, K0R in enumerate(K0_R_critical):
        V0_crit = (K0R / R)**2 * hbar**2 / (2 * m)
        ax1.axvline(x=V0_crit, color='g', linestyle='--', alpha=0.5)
        if i == 0:
            ax1.text(V0_crit + 0.2, 5, f'1st\nbound\nstate', fontsize=9)

    ax1.set_xlabel('Well depth V₀', fontsize=12)
    ax1.set_ylabel('Scattering length a', fontsize=12)
    ax1.set_title('Square Well: Scattering Length vs Depth', fontsize=12)
    ax1.set_ylim(-10, 10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Zoom near first resonance
    ax2 = axes[1]
    V0_crit1 = (np.pi/(2*R))**2 * hbar**2 / (2 * m)
    V0_zoom = np.linspace(V0_crit1 - 0.5, V0_crit1 + 0.5, 100)

    a_zoom = []
    for V0 in V0_zoom:
        V_func = lambda r, V0=V0: square_well(r, V0, R)
        try:
            a, _, _ = compute_scattering_length(V_func)
            a_zoom.append(a)
        except Exception:
            a_zoom.append(np.nan)

    ax2.plot(V0_zoom, a_zoom, 'b-', linewidth=2)
    ax2.axvline(x=V0_crit1, color='g', linestyle='--',
                label='Bound state threshold')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Well depth V₀', fontsize=12)
    ax2.set_ylabel('Scattering length a', fontsize=12)
    ax2.set_title('Resonance: a → ±∞ at Bound State Threshold', fontsize=12)
    ax2.set_ylim(-50, 50)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scattering_length_vs_depth.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_effective_range_expansion():
    """Demonstrate effective range expansion."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Square well with moderate depth
    V0 = 3.0
    R = 1.0
    V_func = lambda r: square_well(r, V0, R)

    k_values = np.linspace(0.05, 0.8, 50)
    phase_shifts = []

    for k in k_values:
        try:
            delta = solve_radial_equation(V_func, k)
            phase_shifts.append(delta)
        except Exception:
            phase_shifts.append(np.nan)

    phase_shifts = np.array(phase_shifts)
    valid = ~np.isnan(phase_shifts)

    # Plot k*cot(delta) vs k^2
    ax1 = axes[0]
    k_cot_delta = k_values[valid] / np.tan(phase_shifts[valid])
    k_sq = k_values[valid]**2

    ax1.plot(k_sq, k_cot_delta, 'bo', markersize=5, label='Numerical')

    # Fit effective range expansion
    a, r0 = effective_range_fit(k_values[valid][:20], phase_shifts[valid][:20])
    k_sq_fit = np.linspace(0, 0.7, 100)
    k_cot_fit = -1/a + r0/2 * k_sq_fit

    ax1.plot(k_sq_fit, k_cot_fit, 'r-', linewidth=2,
             label=f'Fit: a = {a:.3f}, r₀ = {r0:.3f}')

    ax1.set_xlabel('k²', fontsize=12)
    ax1.set_ylabel('k cot(δ₀)', fontsize=12)
    ax1.set_title('Effective Range Expansion', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot phase shift vs k
    ax2 = axes[1]
    ax2.plot(k_values[valid], np.degrees(phase_shifts[valid]),
             'bo-', markersize=4, linewidth=1, label='Numerical')

    # Low-k approximation: delta ≈ -ka
    k_low = np.linspace(0, 0.5, 50)
    delta_approx = -k_low * a
    ax2.plot(k_low, np.degrees(delta_approx), 'r--', linewidth=2,
             label=f'Linear approx: δ ≈ -ka')

    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('Phase shift δ₀ (degrees)', fontsize=12)
    ax2.set_title('s-Wave Phase Shift', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('effective_range_expansion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nEffective Range Parameters:")
    print(f"  Scattering length a = {a:.4f}")
    print(f"  Effective range r₀ = {r0:.4f}")


def plot_cross_section_vs_energy():
    """Compare full cross section with low-energy approximations."""

    fig, ax = plt.subplots(figsize=(10, 6))

    V0 = 3.0
    R = 1.0
    V_func = lambda r: square_well(r, V0, R)

    # Get scattering length and effective range
    k_low = np.linspace(0.05, 0.3, 20)
    deltas_low = []
    for k in k_low:
        try:
            delta = solve_radial_equation(V_func, k)
            deltas_low.append(delta)
        except Exception:
            deltas_low.append(np.nan)

    deltas_low = np.array(deltas_low)
    valid = ~np.isnan(deltas_low)
    a, r0 = effective_range_fit(k_low[valid], deltas_low[valid])

    # Full calculation
    k_values = np.linspace(0.05, 2.0, 100)
    sigma_full = []

    for k in k_values:
        try:
            delta = solve_radial_equation(V_func, k)
            sigma = 4 * np.pi / k**2 * np.sin(delta)**2
            sigma_full.append(sigma)
        except Exception:
            sigma_full.append(np.nan)

    sigma_full = np.array(sigma_full)

    # Approximations
    sigma_leading = 4 * np.pi * a**2 * np.ones_like(k_values)

    # With effective range
    f0_eff = -a / (1 + 1j*k_values*a - 0.5*a*r0*k_values**2)
    sigma_eff_range = 4 * np.pi * np.abs(f0_eff)**2

    # Plot
    ax.semilogy(k_values, sigma_full, 'b-', linewidth=2, label='Exact (s-wave)')
    ax.semilogy(k_values, sigma_leading, 'r--', linewidth=2,
                label=f'σ = 4πa² (a = {a:.2f})')
    ax.semilogy(k_values, sigma_eff_range, 'g:', linewidth=2,
                label=f'With effective range (r₀ = {r0:.2f})')

    ax.set_xlabel('Wave number k', fontsize=12)
    ax.set_ylabel('Cross section σ', fontsize=12)
    ax.set_title('Low-Energy Cross Section Approximations', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('low_energy_cross_section.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_bound_state_connection():
    """Show connection between scattering length and bound states."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    R = 1.0

    # Find bound state energies
    def find_bound_states(V0, R, n_max=5):
        """Find s-wave bound state energies."""
        bound_energies = []

        def equation(kappa):
            K = np.sqrt(2*m*V0/hbar**2 - kappa**2)
            if K*R < 0.01:
                return 1
            return K / np.tan(K*R) + kappa

        for n in range(n_max):
            kappa_min = 0.01
            kappa_max = np.sqrt(2*m*V0/hbar**2) - 0.01

            if kappa_max <= kappa_min:
                break

            try:
                kappa = brentq(equation, kappa_min, kappa_max)
                E_b = -hbar**2 * kappa**2 / (2*m)
                bound_energies.append(E_b)
            except Exception:
                pass

        return bound_energies

    # Scattering length and bound state energy vs V0
    V0_values = np.linspace(3, 15, 100)
    a_values = []
    E_b_values = []

    for V0 in V0_values:
        V_func = lambda r, V0=V0: square_well(r, V0, R)
        try:
            a, _, _ = compute_scattering_length(V_func)
            a_values.append(a)
        except Exception:
            a_values.append(np.nan)

        bound_states = find_bound_states(V0, R)
        if bound_states:
            E_b_values.append(bound_states[0])
        else:
            E_b_values.append(np.nan)

    a_values = np.array(a_values)
    E_b_values = np.array(E_b_values)

    # Plot
    ax1 = axes[0]
    ax1.plot(V0_values, a_values, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Well depth V₀', fontsize=12)
    ax1.set_ylabel('Scattering length a', fontsize=12)
    ax1.set_title('Scattering Length', fontsize=12)
    ax1.set_ylim(-10, 10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    valid_bound = ~np.isnan(E_b_values)
    ax2.plot(V0_values[valid_bound], np.abs(E_b_values[valid_bound]),
             'r-', linewidth=2, label='Numerical $|E_B|$')

    # Theoretical relation: E_B = hbar^2 / (2*m*a^2) for large a
    a_pos = a_values[a_values > 0.5]
    V0_pos = V0_values[a_values > 0.5]
    E_theory = hbar**2 / (2*m*a_pos**2)
    ax2.plot(V0_pos, E_theory, 'g--', linewidth=2,
             label='$|E_B| = \\hbar^2/(2ma^2)$')

    ax2.set_xlabel('Well depth V₀', fontsize=12)
    ax2.set_ylabel('Binding energy |E_B|', fontsize=12)
    ax2.set_title('Bound State Energy', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bound_state_connection.png', dpi=150, bbox_inches='tight')
    plt.show()


def feshbach_resonance_demo():
    """Demonstrate Feshbach resonance behavior."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Feshbach resonance parameters
    a_bg = 5.0  # background scattering length
    B_0 = 100   # resonance position (Gauss)
    Delta = 10  # width (Gauss)

    B = np.linspace(50, 150, 200)

    # Scattering length
    a = a_bg * (1 - Delta / (B - B_0))

    ax1 = axes[0]
    ax1.plot(B, a, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(y=a_bg, color='r', linestyle=':', alpha=0.5,
                label=f'$a_{{bg}}$ = {a_bg}')
    ax1.axvline(x=B_0, color='g', linestyle='--', alpha=0.5,
                label=f'$B_0$ = {B_0}')

    ax1.set_xlabel('Magnetic field B (Gauss)', fontsize=12)
    ax1.set_ylabel('Scattering length a', fontsize=12)
    ax1.set_title('Feshbach Resonance', fontsize=12)
    ax1.set_ylim(-50, 50)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cross section
    k = 0.1  # Low energy
    sigma = 4 * np.pi * a**2

    ax2 = axes[1]
    ax2.semilogy(B, np.abs(sigma), 'b-', linewidth=2)
    ax2.axvline(x=B_0, color='g', linestyle='--', alpha=0.5,
                label='Resonance')

    # Mark zero crossing
    B_zero = B_0 + Delta
    ax2.axvline(x=B_zero, color='r', linestyle=':', alpha=0.5,
                label=f'a = 0 at B = {B_zero}')

    ax2.set_xlabel('Magnetic field B (Gauss)', fontsize=12)
    ax2.set_ylabel('Cross section σ (arb. units)', fontsize=12)
    ax2.set_title('Cross Section Near Feshbach Resonance', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('feshbach_resonance.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFeshbach Resonance Parameters:")
    print(f"  Background scattering length: a_bg = {a_bg}")
    print(f"  Resonance position: B_0 = {B_0} G")
    print(f"  Resonance width: Δ = {Delta} G")
    print(f"  Zero crossing: B = {B_zero} G")


# Main execution
if __name__ == "__main__":
    print("Day 501: Low-Energy Scattering")
    print("=" * 60)

    plot_scattering_length_vs_depth()
    plot_effective_range_expansion()
    plot_cross_section_vs_energy()
    plot_bound_state_connection()
    feshbach_resonance_demo()
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| s-wave dominance | Only ℓ=0 contributes when kR ≪ 1 |
| Scattering length a | Low-energy parameter: δ₀ ≈ -ka |
| Effective range r₀ | Next-order correction in k² |
| Bound state connection | a = ℏ/√(2m\|E_B\|) for weak binding |
| Feshbach resonance | Magnetically tunable a near molecular state |

### Key Formulas

| Formula | Meaning |
|---------|---------|
| $$\delta_0 \approx -ka$$ | Low-energy phase shift |
| $$\sigma \approx 4\pi a^2$$ | Low-energy cross section |
| $$k\cot\delta_0 = -\frac{1}{a} + \frac{r_0 k^2}{2}$$ | Effective range expansion |
| $$E_B = \frac{\hbar^2}{2ma^2}$$ | Weakly bound state energy |

---

## 10. Daily Checklist

### Conceptual Understanding
- [ ] I understand why s-wave dominates at low energy
- [ ] I can interpret positive vs negative scattering length
- [ ] I see the connection to bound states
- [ ] I understand Feshbach resonances conceptually

### Mathematical Skills
- [ ] I can extract scattering length from phase shifts
- [ ] I can apply the effective range expansion
- [ ] I can relate binding energy to scattering length
- [ ] I can assess when approximations are valid

### Computational Skills
- [ ] I computed scattering lengths numerically
- [ ] I performed effective range fits
- [ ] I visualized resonance behavior
- [ ] I explored bound state connections

---

## 11. Preview: Day 502

Tomorrow we study **resonances**—dramatic enhancements in scattering when the energy matches a quasi-bound state:

- Breit-Wigner formula
- Width and lifetime
- Phase shift behavior near resonance
- Shape and Fano resonances
- Examples in nuclear and particle physics

Resonances are windows into the structure of composite systems and are the primary signals sought in particle physics experiments.

---

## References

1. Griffiths, D.J. (2018). *Introduction to Quantum Mechanics*, 3rd ed., Ch. 11.6.

2. Sakurai, J.J. & Napolitano, J. (2017). *Modern Quantum Mechanics*, 2nd ed., Ch. 7.7.

3. Landau, L.D. & Lifshitz, E.M. (1977). *Quantum Mechanics*, Ch. 130-132.

4. Chin, C. et al. (2010). *Feshbach Resonances in Ultracold Gases*, Rev. Mod. Phys.

---

*"The scattering length is the single number that characterizes all low-energy properties of quantum interactions."*
— Vitaly Efimov

---

**Day 501 Complete.** Tomorrow: Resonances.
