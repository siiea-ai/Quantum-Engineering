# Day 375: Finite Square Well - Setup and Transcendental Equations

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: FSW Schrodinger equation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving: Transcendental equations |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab: Graphical solutions |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you will be able to:

1. Define the finite square well potential and its physical motivation
2. Solve the Schrodinger equation in each region (inside and outside)
3. Identify the parameters $k$ and $\kappa$ and their physical meaning
4. Derive the transcendental eigenvalue equations for even and odd parity
5. Find bound state energies graphically
6. Relate well depth to the number of bound states

---

## Core Content

### 1. The Finite Square Well Potential

The **finite square well** (FSW) is a more realistic model than the infinite well, allowing finite barrier heights and wave function penetration.

#### Potential Definition

$$\boxed{V(x) = \begin{cases} -V_0 & \text{if } |x| < a \\ 0 & \text{if } |x| > a \end{cases}}$$

where $V_0 > 0$ is the **well depth** and $2a$ is the **well width**.

```
   V(x)
    ^
    |
    |  0 ───────────┬─────────┬─────────── 0
    |               │         │
    |               │         │
    |  -V₀ ─────────┴─────────┴─────────
    |         Region II
    |       (inside well)
    |
    +───────────────────────────────────> x
              -a    0    a
         Region I      Region III
```

#### Physical Realizations

1. **Semiconductor heterostructures**: GaAs sandwiched between AlGaAs
2. **Quantum wells**: Thin layers of narrow-gap semiconductor
3. **Nucleon in nucleus**: Approximate nuclear potential
4. **Atoms in optical dipole traps**: Laser-created potentials

### 2. Energy Regimes

For a particle with energy $E$:

- **Bound states**: $-V_0 < E < 0$ (particle localized near well)
- **Scattering states**: $E > 0$ (particle free to escape)

Today we focus on **bound states** with $-V_0 < E < 0$.

### 3. The Schrodinger Equation in Each Region

The time-independent Schrodinger equation:

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

#### Region II: Inside the Well ($|x| < a$)

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} - V_0\psi = E\psi$$

Rearranging:

$$\frac{d^2\psi}{dx^2} = -\frac{2m(E + V_0)}{\hbar^2}\psi$$

Since $E > -V_0$ for bound states, $E + V_0 > 0$. Define:

$$\boxed{k^2 = \frac{2m(E + V_0)}{\hbar^2}}$$

The equation becomes:

$$\frac{d^2\psi}{dx^2} = -k^2\psi$$

General solution: **oscillatory**

$$\psi_{\text{II}}(x) = A\cos(kx) + B\sin(kx)$$

#### Regions I and III: Outside the Well ($|x| > a$)

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi$$

Since $E < 0$:

$$\frac{d^2\psi}{dx^2} = -\frac{2mE}{\hbar^2}\psi = \frac{2m|E|}{\hbar^2}\psi$$

Define:

$$\boxed{\kappa^2 = \frac{2m|E|}{\hbar^2} = -\frac{2mE}{\hbar^2}}$$

The equation becomes:

$$\frac{d^2\psi}{dx^2} = \kappa^2\psi$$

General solution: **exponential**

$$\psi(x) = Ce^{\kappa x} + De^{-\kappa x}$$

### 4. Normalizability Constraints

For a bound state, $\psi(x) \to 0$ as $|x| \to \infty$.

**Region I** ($x < -a$): Must have $\psi_{\text{I}}(x) = Ce^{\kappa x}$ (decaying as $x \to -\infty$)

**Region III** ($x > a$): Must have $\psi_{\text{III}}(x) = Fe^{-\kappa x}$ (decaying as $x \to +\infty$)

### 5. Relationship Between $k$ and $\kappa$

From the definitions:

$$k^2 = \frac{2m(E + V_0)}{\hbar^2}, \quad \kappa^2 = \frac{2m|E|}{\hbar^2}$$

Adding:

$$k^2 + \kappa^2 = \frac{2m(E + V_0)}{\hbar^2} + \frac{2m|E|}{\hbar^2} = \frac{2m V_0}{\hbar^2}$$

$$\boxed{k^2 + \kappa^2 = \frac{2mV_0}{\hbar^2} \equiv \frac{1}{a^2}z_0^2}$$

where we define the **dimensionless parameter**:

$$\boxed{z_0 = \frac{a}{\hbar}\sqrt{2mV_0}}$$

This parameter characterizes the "strength" of the well.

### 6. Parity Symmetry

The potential $V(x)$ is symmetric: $V(-x) = V(x)$.

By the symmetry theorem, we can choose energy eigenfunctions to be:
- **Even parity**: $\psi(-x) = \psi(x)$
- **Odd parity**: $\psi(-x) = -\psi(x)$

This simplifies the problem significantly.

### 7. Even Parity Solutions

For even parity, $\psi(-x) = \psi(x)$:

**Inside ($|x| < a$):**
$$\psi_{\text{II}}(x) = A\cos(kx)$$
(no $\sin$ term since $\sin$ is odd)

**Outside ($|x| > a$):**
$$\psi_{\text{I}}(x) = Ce^{\kappa x} \quad (x < -a)$$
$$\psi_{\text{III}}(x) = Ce^{-\kappa x} \quad (x > a)$$

(same coefficient $C$ by parity)

#### Matching Conditions at $x = a$

The wave function must be **continuous**:

$$\psi_{\text{II}}(a) = \psi_{\text{III}}(a)$$
$$A\cos(ka) = Ce^{-\kappa a}$$

The derivative must be **continuous**:

$$\psi'_{\text{II}}(a) = \psi'_{\text{III}}(a)$$
$$-Ak\sin(ka) = -C\kappa e^{-\kappa a}$$

#### Transcendental Equation

Dividing the second equation by the first:

$$\frac{-Ak\sin(ka)}{A\cos(ka)} = \frac{-C\kappa e^{-\kappa a}}{Ce^{-\kappa a}}$$

$$\boxed{k\tan(ka) = \kappa} \quad \text{(even parity)}$$

### 8. Odd Parity Solutions

For odd parity, $\psi(-x) = -\psi(x)$:

**Inside ($|x| < a$):**
$$\psi_{\text{II}}(x) = B\sin(kx)$$

**Outside ($|x| > a$):**
$$\psi_{\text{I}}(x) = -Ce^{\kappa x} \quad (x < -a)$$
$$\psi_{\text{III}}(x) = Ce^{-\kappa x} \quad (x > a)$$

#### Matching at $x = a$

Continuity: $B\sin(ka) = Ce^{-\kappa a}$

Derivative: $Bk\cos(ka) = -C\kappa e^{-\kappa a}$

#### Transcendental Equation

$$\frac{Bk\cos(ka)}{B\sin(ka)} = \frac{-C\kappa e^{-\kappa a}}{Ce^{-\kappa a}}$$

$$\boxed{-k\cot(ka) = \kappa} \quad \text{(odd parity)}$$

Or equivalently: $k\cot(ka) = -\kappa$

### 9. Dimensionless Form

Define dimensionless variables:

$$\xi = ka, \quad \eta = \kappa a$$

From $k^2 + \kappa^2 = 2mV_0/\hbar^2$:

$$\xi^2 + \eta^2 = z_0^2$$

This is the equation of a **circle** of radius $z_0$ in the $(\xi, \eta)$ plane.

The transcendental equations become:

**Even parity:** $\xi\tan\xi = \eta$

**Odd parity:** $-\xi\cot\xi = \eta$

### 10. Graphical Solution

Plot in the $(\xi, \eta)$ plane (with $\eta > 0$):

1. The **constraint circle**: $\xi^2 + \eta^2 = z_0^2$

2. **Even parity curves**: $\eta = \xi\tan\xi$
   - These pass through origin with slope 1
   - Have vertical asymptotes at $\xi = \pi/2, 3\pi/2, 5\pi/2, \ldots$

3. **Odd parity curves**: $\eta = -\xi\cot\xi$
   - These have vertical asymptotes at $\xi = 0, \pi, 2\pi, \ldots$
   - Start from these asymptotes

**Intersections** of the circle with these curves give the allowed energies!

```
η
^
|     .  /       \  /       \
|    . \/   *     \/    *    \
|   .  /\    \    /\    \     \
|  .  /  \    \  /  \    \     \
| .  /    \    \/    \    \     \ (circle of radius z₀)
|.  /      \   /\     \    \
|──/────────\──/──\─────\────────> ξ
    π/2      π   3π/2    2π

* = intersection points = bound state energies
```

### 11. Number of Bound States

From the graphical solution:

- There is **always at least one bound state** (the ground state, even parity)
- New bound states appear as $z_0$ increases past $n\pi/2$

**Approximate formula for number of bound states:**

$$\boxed{N \approx \left\lfloor \frac{2z_0}{\pi} \right\rfloor + 1 = \left\lfloor \frac{2a}{\pi\hbar}\sqrt{2mV_0} \right\rfloor + 1}$$

| $z_0$ range | Number of bound states |
|-------------|------------------------|
| $0 < z_0 < \pi/2$ | 1 (even only) |
| $\pi/2 < z_0 < \pi$ | 2 (1 even, 1 odd) |
| $\pi < z_0 < 3\pi/2$ | 3 (2 even, 1 odd) |
| $(n-1)\pi/2 < z_0 < n\pi/2$ | $n$ states |

### 12. Limiting Cases

#### Deep Well Limit: $z_0 \to \infty$ (or $V_0 \to \infty$)

The circle becomes very large. Intersections approach:

$$\xi_n \to \frac{n\pi}{2}$$

For even states: $\xi = \pi/2, 3\pi/2, \ldots$ (odd $n$)
For odd states: $\xi = \pi, 2\pi, \ldots$ (even $n$)

So $ka \to n\pi/2$, giving:

$$E_n + V_0 \to \frac{n^2\pi^2\hbar^2}{8ma^2}$$

This approaches the **infinite square well** result with width $2a$!

#### Shallow Well Limit: $z_0 \to 0$

The circle shrinks. There is always exactly **one bound state** (even parity) with $\eta \to z_0$, $\xi \to 0$.

Energy: $E \to -V_0 z_0^2/(2ma^2/\hbar^2) \to 0^-$

The binding becomes very weak.

---

## Physical Interpretation

### Why Bound States Exist

A bound state represents a particle **trapped** in the potential well. The kinetic energy inside ($\frac{\hbar^2 k^2}{2m}$) plus the potential ($-V_0$) equals the total energy $E < 0$.

Outside, the particle is in a **classically forbidden region** where kinetic energy would be negative. Quantum mechanically, the wave function decays exponentially rather than oscillating.

### The Meaning of $\kappa$

The decay constant $\kappa$ determines how quickly the wave function falls off outside the well:

$$\psi(x) \sim e^{-\kappa|x|}$$

The **penetration depth** is:

$$\delta = \frac{1}{\kappa} = \frac{\hbar}{\sqrt{2m|E|}}$$

Deeper binding (larger $|E|$) means smaller penetration.

### Minimum Well Depth

For a finite well with given width $2a$, there is always at least one bound state regardless of how shallow the well is. This is a **1D peculiarity** - in 3D, sufficiently shallow wells have no bound states.

---

## Quantum Computing Connection

### Semiconductor Quantum Wells

Real quantum dot qubits are based on finite wells:
- **InAs/GaAs quantum dots**: $V_0 \sim 0.3-0.5$ eV, $2a \sim 5-20$ nm
- **GaAs/AlGaAs wells**: $V_0 \sim 0.2$ eV, tunable width

The finite barrier allows:
- **Tunneling** between adjacent dots for two-qubit gates
- **Electric field control** of wave function shape
- **Optical transitions** for initialization and readout

### Transmon Qubit

The Josephson junction potential is approximately a finite well (with anharmonic corrections). The penetration into classically forbidden regions affects:
- **Charge dispersion**: Sensitivity to charge noise
- **Junction coupling**: Inter-qubit coupling strengths
- **Relaxation rates**: Matrix elements for decay

### Quantum Well Lasers

Semiconductor lasers use quantum wells to create discrete energy levels. The finite well model predicts:
- Number of confined states (should be small, 2-3)
- Energy level spacing (determines emission wavelength)
- Wave function overlap (affects gain)

---

## Worked Examples

### Example 1: Single Bound State Condition

**Problem:** For an electron in a finite well of width $2a = 1$ nm, find the maximum well depth $V_0$ for which there is exactly one bound state.

**Solution:**

We have exactly one bound state when $0 < z_0 < \pi/2$.

The transition to two states occurs at $z_0 = \pi/2$:

$$\frac{a}{\hbar}\sqrt{2mV_0} = \frac{\pi}{2}$$

Solving for $V_0$:

$$V_0 = \frac{\pi^2\hbar^2}{8ma^2}$$

With $a = 0.5$ nm $= 0.5 \times 10^{-9}$ m:

$$V_0 = \frac{\pi^2 \times (1.055 \times 10^{-34})^2}{8 \times 9.109 \times 10^{-31} \times (0.5 \times 10^{-9})^2}$$

$$V_0 = \frac{1.096 \times 10^{-67}}{1.822 \times 10^{-48}} = 6.02 \times 10^{-20} \text{ J}$$

$$\boxed{V_0 \approx 0.376 \text{ eV}}$$

For $V_0 < 0.376$ eV, there is exactly one bound state.

---

### Example 2: Graphical Solution

**Problem:** A particle is in a finite well with $z_0 = 2$. Find the approximate bound state energies graphically.

**Solution:**

The constraint circle is $\xi^2 + \eta^2 = 4$.

**Even parity**: Solve $\eta = \xi\tan\xi$ with $\xi^2 + \eta^2 = 4$

Near $\xi \approx 1.0$: $\tan(1.0) \approx 1.56$, so $\eta \approx 1.56$
Check: $1^2 + 1.56^2 = 3.43 \neq 4$

More carefully, by iteration or graphically: $\xi_1 \approx 1.03$, $\eta_1 \approx 1.71$

**Odd parity**: Solve $\eta = -\xi\cot\xi$ with $\xi^2 + \eta^2 = 4$

Near $\xi \approx 2.0$ (first odd solution starts near $\xi = \pi/2$):
Since $z_0 = 2 > \pi/2 \approx 1.57$, there is one odd solution.

By graphical analysis: $\xi_2 \approx 1.90$, $\eta_2 \approx 0.65$

**Energies:**

$$E = -\frac{\hbar^2\kappa^2}{2m} = -\frac{\hbar^2\eta^2}{2ma^2}$$

Ground state (even): $\eta_1 = 1.71 \implies E_1/V_0 = -\eta_1^2/z_0^2 = -0.73$

First excited (odd): $\eta_2 = 0.65 \implies E_2/V_0 = -\eta_2^2/z_0^2 = -0.11$

---

### Example 3: Deep Well Approximation

**Problem:** A well has $z_0 = 10$. Estimate the energies of the first three bound states using the deep-well approximation.

**Solution:**

For large $z_0$, intersections occur near $\xi = n\pi/2$.

More precisely, for even states (odd $n$):
$$\tan(\xi) \approx \frac{\eta}{\xi} \approx \frac{\sqrt{z_0^2 - \xi^2}}{\xi}$$

For $\xi \approx n\pi/2$, the deviation is small.

**Zeroth-order approximation:**

$$\xi_n \approx \frac{n\pi}{2}, \quad n = 1, 2, 3, \ldots$$

$$E_n + V_0 = \frac{\hbar^2 k^2}{2m} = \frac{\hbar^2 \xi_n^2}{2ma^2} \approx \frac{n^2\pi^2\hbar^2}{8ma^2}$$

For $z_0 = 10$:
- $n = 1$ (even): $\xi_1 \approx 1.57$, $E_1/V_0 \approx 1.57^2/100 - 1 = -0.975$
- $n = 2$ (odd): $\xi_2 \approx 3.14$, $E_2/V_0 \approx 9.87/100 - 1 = -0.901$
- $n = 3$ (even): $\xi_3 \approx 4.71$, $E_3/V_0 \approx 22.2/100 - 1 = -0.778$

---

## Practice Problems

### Level 1: Direct Application

1. **Parameter calculation:** For an electron in a well with $V_0 = 1$ eV and $a = 2$ nm, calculate $z_0$.

2. **Energy constraint:** If $k = 2$ nm$^{-1}$ and $\kappa = 1$ nm$^{-1}$, what is $V_0$ for an electron?

3. **Bound state count:** How many bound states exist for $z_0 = 5$?

4. **Parity identification:** The third bound state (second excited state) has what parity?

### Level 2: Intermediate

5. **Graphical solution:** Sketch the graphical solution for $z_0 = 3$. How many intersections are there?

6. **Deep well limit:** For $z_0 = 20$, estimate the ground state energy as a fraction of $V_0$.

7. **Shallow well:** For $z_0 = 0.5$, estimate the ground state binding energy $|E_1|$.

8. **Penetration depth:** Calculate the penetration depth for a state with $\eta = 1.5$ and $a = 1$ nm.

### Level 3: Challenging

9. **Transcendental solution:** Solve $\xi\tan\xi = \sqrt{4 - \xi^2}$ numerically to find $\xi_1$ for $z_0 = 2$.

10. **Critical well depth:** Find the minimum $z_0$ (and hence $V_0$) for three bound states.

11. **Asymptotic expansion:** For large $z_0$, derive the first correction to $\xi_n \approx n\pi/2$.

12. **WKB comparison:** Use the WKB quantization condition to estimate bound state energies and compare to exact results.

---

## Computational Lab

### Exercise 1: Graphical Solution of Transcendental Equations

```python
"""
Day 375 Computational Lab: Finite Square Well
Graphical solution of transcendental equations for bound states
"""

import numpy as np
import matplotlib.pyplot as plt

def even_parity(xi):
    """Even parity: eta = xi * tan(xi)"""
    with np.errstate(invalid='ignore', divide='ignore'):
        result = xi * np.tan(xi)
        # Only positive eta makes sense
        result = np.where(result > 0, result, np.nan)
    return result

def odd_parity(xi):
    """Odd parity: eta = -xi * cot(xi)"""
    with np.errstate(invalid='ignore', divide='ignore'):
        result = -xi / np.tan(xi)
        result = np.where(result > 0, result, np.nan)
    return result

def constraint_circle(xi, z0):
    """Constraint: xi^2 + eta^2 = z0^2"""
    return np.sqrt(np.maximum(z0**2 - xi**2, 0))

# Parameters
z0 = 4.0  # Dimensionless well strength

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

xi = np.linspace(0.001, z0 + 0.5, 10000)

# Plot even parity curves
eta_even = even_parity(xi)
ax.plot(xi, eta_even, 'b-', linewidth=2, label='Even: $\\eta = \\xi\\tan\\xi$')

# Plot odd parity curves
eta_odd = odd_parity(xi)
ax.plot(xi, eta_odd, 'r-', linewidth=2, label='Odd: $\\eta = -\\xi\\cot\\xi$')

# Plot constraint circle
eta_circle = constraint_circle(xi, z0)
ax.plot(xi, eta_circle, 'k--', linewidth=2, label=f'Constraint: $\\xi^2 + \\eta^2 = z_0^2 = {z0**2:.0f}$')

# Find intersections numerically
from scipy.optimize import brentq

def even_eq(xi, z0):
    if xi <= 0 or xi >= z0:
        return np.inf
    eta = xi * np.tan(xi)
    return eta**2 - (z0**2 - xi**2)

def odd_eq(xi, z0):
    if xi <= 0 or xi >= z0:
        return np.inf
    eta = -xi / np.tan(xi)
    return eta**2 - (z0**2 - xi**2)

# Find even parity solutions
even_solutions = []
for i in range(int(z0/(np.pi/2)) + 1):
    xi_min = i * np.pi + 0.01
    xi_max = (i + 0.5) * np.pi - 0.01
    if xi_max > z0:
        xi_max = z0 - 0.01
    if xi_min >= xi_max:
        continue
    try:
        xi_sol = brentq(even_eq, xi_min, min(xi_max, z0), args=(z0,))
        eta_sol = xi_sol * np.tan(xi_sol)
        if eta_sol > 0:
            even_solutions.append((xi_sol, eta_sol))
    except:
        pass

# Find odd parity solutions
odd_solutions = []
for i in range(int(z0/np.pi) + 1):
    xi_min = (i + 0.5) * np.pi + 0.01
    xi_max = (i + 1) * np.pi - 0.01
    if xi_min >= z0:
        continue
    if xi_max > z0:
        xi_max = z0 - 0.01
    if xi_min >= xi_max:
        continue
    try:
        xi_sol = brentq(odd_eq, xi_min, min(xi_max, z0), args=(z0,))
        eta_sol = -xi_sol / np.tan(xi_sol)
        if eta_sol > 0:
            odd_solutions.append((xi_sol, eta_sol))
    except:
        pass

# Plot solutions
for i, (xi_s, eta_s) in enumerate(even_solutions):
    ax.plot(xi_s, eta_s, 'bo', markersize=12)
    ax.annotate(f'Even {i+1}\n$\\xi$={xi_s:.3f}\n$\\eta$={eta_s:.3f}',
                xy=(xi_s, eta_s), xytext=(xi_s+0.3, eta_s+0.3),
                fontsize=9, color='blue')

for i, (xi_s, eta_s) in enumerate(odd_solutions):
    ax.plot(xi_s, eta_s, 'rs', markersize=12)
    ax.annotate(f'Odd {i+1}\n$\\xi$={xi_s:.3f}\n$\\eta$={eta_s:.3f}',
                xy=(xi_s, eta_s), xytext=(xi_s+0.3, eta_s-0.5),
                fontsize=9, color='red')

ax.set_xlim(0, z0 + 0.5)
ax.set_ylim(0, z0 + 0.5)
ax.set_xlabel('$\\xi = ka$', fontsize=14)
ax.set_ylabel('$\\eta = \\kappa a$', fontsize=14)
ax.set_title(f'Finite Square Well: Graphical Solution for $z_0 = {z0}$', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('fsw_graphical_solution.png', dpi=150, bbox_inches='tight')
plt.show()

# Print solutions
print(f"\nBound State Solutions for z₀ = {z0}")
print("="*50)
all_solutions = [(xi, eta, 'even') for xi, eta in even_solutions] + \
                [(xi, eta, 'odd') for xi, eta in odd_solutions]
all_solutions.sort(key=lambda x: -x[1])  # Sort by eta (deepest binding first)

for i, (xi, eta, parity) in enumerate(all_solutions):
    E_ratio = -(eta/z0)**2  # E/V_0
    print(f"State {i+1} ({parity}): ξ = {xi:.4f}, η = {eta:.4f}, E/V₀ = {E_ratio:.4f}")
```

### Exercise 2: Number of Bound States vs Well Strength

```python
"""
Explore how the number of bound states depends on z_0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def count_bound_states(z0):
    """Count the number of bound states for given z_0"""
    def even_eq(xi):
        if xi <= 0 or xi >= z0:
            return np.inf
        eta = xi * np.tan(xi)
        if eta < 0:
            return np.inf
        return eta**2 - (z0**2 - xi**2)

    def odd_eq(xi):
        if xi <= 0 or xi >= z0:
            return np.inf
        eta = -xi / np.tan(xi)
        if eta < 0:
            return np.inf
        return eta**2 - (z0**2 - xi**2)

    count = 0

    # Even parity
    for i in range(int(z0/(np.pi/2)) + 1):
        xi_min = i * np.pi + 0.001
        xi_max = (i + 0.5) * np.pi - 0.001
        if xi_max > z0:
            xi_max = z0 - 0.001
        if xi_min >= xi_max:
            continue
        try:
            brentq(even_eq, xi_min, xi_max)
            count += 1
        except:
            pass

    # Odd parity
    for i in range(int(z0/np.pi) + 1):
        xi_min = (i + 0.5) * np.pi + 0.001
        xi_max = (i + 1) * np.pi - 0.001
        if xi_min >= z0:
            continue
        if xi_max > z0:
            xi_max = z0 - 0.001
        if xi_min >= xi_max:
            continue
        try:
            brentq(odd_eq, xi_min, xi_max)
            count += 1
        except:
            pass

    return count

# Calculate for range of z_0
z0_values = np.linspace(0.1, 10, 1000)
n_states = [count_bound_states(z0) for z0 in z0_values]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.step(z0_values, n_states, where='post', linewidth=2, color='blue')

# Mark transitions
for n in range(1, 7):
    z_crit = n * np.pi / 2
    if z_crit < 10:
        ax.axvline(x=z_crit, color='red', linestyle='--', alpha=0.5)
        ax.text(z_crit + 0.1, n + 0.5, f'$z_0 = {n}\\pi/2$', fontsize=9, color='red')

ax.set_xlabel('$z_0 = (a/\\hbar)\\sqrt{2mV_0}$', fontsize=12)
ax.set_ylabel('Number of Bound States', fontsize=12)
ax.set_title('Bound State Count vs Well Strength', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)

plt.tight_layout()
plt.savefig('fsw_state_count.png', dpi=150, bbox_inches='tight')
plt.show()

# Print critical z_0 values
print("\nCritical values of z₀ for new bound states:")
print("-"*40)
for n in range(1, 8):
    z_crit = n * np.pi / 2
    print(f"N = {n} state appears at z₀ = {n}π/2 = {z_crit:.4f}")
```

### Exercise 3: Energy Level Diagram

```python
"""
Create energy level diagram showing how energies evolve with z_0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def find_energies(z0, max_states=6):
    """Find all bound state energies for given z_0"""
    energies = []

    def even_eq(xi):
        eta = xi * np.tan(xi)
        return eta**2 - (z0**2 - xi**2)

    def odd_eq(xi):
        eta = -xi / np.tan(xi)
        return eta**2 - (z0**2 - xi**2)

    # Even parity
    for i in range(max_states):
        xi_min = i * np.pi + 0.001
        xi_max = (i + 0.5) * np.pi - 0.001
        if xi_max > z0:
            xi_max = z0 - 0.001
        if xi_min >= xi_max or xi_min >= z0:
            continue
        try:
            xi_sol = brentq(even_eq, xi_min, xi_max)
            eta_sol = np.sqrt(z0**2 - xi_sol**2)
            E_ratio = -(eta_sol/z0)**2
            energies.append(('even', E_ratio))
        except:
            pass

    # Odd parity
    for i in range(max_states):
        xi_min = (i + 0.5) * np.pi + 0.001
        xi_max = (i + 1) * np.pi - 0.001
        if xi_min >= z0:
            continue
        if xi_max > z0:
            xi_max = z0 - 0.001
        if xi_min >= xi_max:
            continue
        try:
            xi_sol = brentq(odd_eq, xi_min, xi_max)
            eta_sol = np.sqrt(z0**2 - xi_sol**2)
            E_ratio = -(eta_sol/z0)**2
            energies.append(('odd', E_ratio))
        except:
            pass

    # Sort by energy
    energies.sort(key=lambda x: x[1])
    return energies

# Calculate energies for range of z_0
z0_values = np.linspace(0.5, 8, 200)

fig, ax = plt.subplots(figsize=(12, 8))

# Track energies for plotting
for z0 in z0_values:
    energies = find_energies(z0)
    for parity, E in energies:
        color = 'blue' if parity == 'even' else 'red'
        ax.plot(z0, E, '.', color=color, markersize=3)

# Add labels
ax.plot([], [], 'b.', markersize=10, label='Even parity')
ax.plot([], [], 'r.', markersize=10, label='Odd parity')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.5, label='Well bottom $E=-V_0$')

ax.set_xlabel('$z_0 = (a/\\hbar)\\sqrt{2mV_0}$', fontsize=12)
ax.set_ylabel('$E/V_0$', fontsize=12)
ax.set_title('Finite Square Well: Energy Levels vs Well Strength', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 8)
ax.set_ylim(-1.1, 0.1)

# Shade bound state region
ax.fill_between([0, 8], -1, 0, alpha=0.1, color='green')
ax.text(7.5, -0.5, 'Bound\nstate\nregion', fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('fsw_energy_levels.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Potential | $V(x) = -V_0$ for $|x| < a$, $0$ otherwise |
| Inside wave number | $k^2 = 2m(E+V_0)/\hbar^2$ |
| Outside decay constant | $\kappa^2 = -2mE/\hbar^2$ |
| Constraint | $k^2 + \kappa^2 = 2mV_0/\hbar^2$ |
| Dimensionless parameter | $z_0 = (a/\hbar)\sqrt{2mV_0}$ |
| Even parity equation | $k\tan(ka) = \kappa$ |
| Odd parity equation | $-k\cot(ka) = \kappa$ |
| Approx. bound states | $N \approx \lfloor 2z_0/\pi \rfloor + 1$ |

### Main Takeaways

1. The finite well has **oscillatory solutions inside** and **exponentially decaying solutions outside**

2. **Parity symmetry** allows separation into even and odd states

3. Bound state energies satisfy **transcendental equations** solved graphically

4. There is always **at least one bound state** in 1D

5. The number of bound states increases with well strength $z_0$

6. The **deep well limit** recovers infinite square well results

---

## Daily Checklist

- [ ] I can set up the Schrodinger equation in all three regions
- [ ] I understand the physical meaning of $k$ and $\kappa$
- [ ] I can derive the transcendental equations for both parities
- [ ] I understand the graphical solution method
- [ ] I can estimate the number of bound states from $z_0$
- [ ] I understand the deep and shallow well limits
- [ ] I completed the computational lab exercises

---

## Preview: Day 376

Tomorrow we will study the **wave functions** of the finite square well bound states in detail:

- Explicit construction of even and odd solutions
- **Wave function penetration** into classically forbidden regions
- **Penetration depth** $\delta = \hbar/\sqrt{2m|E|}$
- Normalization and probability distributions
- Comparison with infinite well eigenfunctions

We'll see how the finite barriers allow the wave function to "leak" outside - a precursor to the tunneling phenomenon we'll study later.

---

*Day 375 of QSE Self-Study Curriculum*
*Week 54: Bound States - Infinite and Finite Wells*
*Month 14: One-Dimensional Systems*
