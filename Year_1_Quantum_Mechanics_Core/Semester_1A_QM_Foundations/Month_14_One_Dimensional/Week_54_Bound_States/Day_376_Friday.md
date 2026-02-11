# Day 376: Finite Square Well - Wave Functions and Penetration

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Wave function construction and penetration |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving: Probability calculations |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab: Wave function visualization |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you will be able to:

1. Construct explicit wave functions for even and odd parity bound states
2. Apply normalization conditions to determine wave function amplitudes
3. Calculate and interpret the penetration depth
4. Compute the probability of finding the particle in the classically forbidden region
5. Explain the relationship between binding energy and penetration
6. Compare finite well eigenfunctions to their infinite well counterparts

---

## Core Content

### 1. Complete Wave Function Construction

From Day 375, we established that bound states satisfy transcendental equations. Now we construct the explicit wave functions.

#### Even Parity States

For even parity, the wave function is symmetric: $\psi(-x) = \psi(x)$.

$$\boxed{\psi_{\text{even}}(x) = \begin{cases} Ce^{\kappa x} & x < -a \\ A\cos(kx) & |x| < a \\ Ce^{-\kappa x} & x > a \end{cases}}$$

The parameters $k$ and $\kappa$ are determined by the transcendental equation $k\tan(ka) = \kappa$.

#### Odd Parity States

For odd parity, the wave function is antisymmetric: $\psi(-x) = -\psi(x)$.

$$\boxed{\psi_{\text{odd}}(x) = \begin{cases} -Ce^{\kappa x} & x < -a \\ B\sin(kx) & |x| < a \\ Ce^{-\kappa x} & x > a \end{cases}}$$

The parameters satisfy $-k\cot(ka) = \kappa$.

### 2. Matching Conditions and Amplitude Ratios

#### Even Parity: Continuity at $x = a$

From $\psi$ continuity:
$$A\cos(ka) = Ce^{-\kappa a}$$

From $\psi'$ continuity:
$$-Ak\sin(ka) = -C\kappa e^{-\kappa a}$$

Dividing:
$$k\tan(ka) = \kappa$$

Solving for the amplitude ratio:
$$\boxed{\frac{C}{A} = \cos(ka)e^{\kappa a}}$$

#### Odd Parity: Continuity at $x = a$

From $\psi$ continuity:
$$B\sin(ka) = Ce^{-\kappa a}$$

From $\psi'$ continuity:
$$Bk\cos(ka) = -C\kappa e^{-\kappa a}$$

Dividing:
$$k\cot(ka) = -\kappa$$

Amplitude ratio:
$$\boxed{\frac{C}{B} = \sin(ka)e^{\kappa a}}$$

### 3. Normalization

The wave function must satisfy:
$$\int_{-\infty}^{\infty}|\psi(x)|^2 dx = 1$$

#### Even Parity Normalization

$$1 = 2\int_a^{\infty}C^2 e^{-2\kappa x}dx + \int_{-a}^{a}A^2\cos^2(kx)dx$$

Computing the integrals:

$$\int_a^{\infty}e^{-2\kappa x}dx = \frac{e^{-2\kappa a}}{2\kappa}$$

$$\int_{-a}^{a}\cos^2(kx)dx = a + \frac{\sin(2ka)}{2k}$$

Using $\sin(2ka) = 2\sin(ka)\cos(ka)$ and $\tan(ka) = \kappa/k$:

$$\int_{-a}^{a}\cos^2(kx)dx = a\left(1 + \frac{\sin(ka)\cos(ka)}{ka}\right) = a\left(1 + \frac{\kappa}{k^2a + \kappa^2 a}\cdot ka \cdot \frac{k}{ka}\right)$$

After simplification:

$$1 = A^2\left[a\left(1 + \frac{\sin(2ka)}{2ka}\right) + \frac{\cos^2(ka)}{\kappa}\right]$$

$$\boxed{A = \left[a\left(1 + \frac{\sin(2ka)}{2ka}\right) + \frac{\cos^2(ka)}{\kappa}\right]^{-1/2}}$$

### 4. Penetration into Classically Forbidden Regions

#### The Classically Forbidden Region

For a particle with energy $E < 0$, the classically allowed region is where $E > V(x)$, i.e., inside the well $|x| < a$.

Outside the well ($|x| > a$), $E < V = 0$, so the particle is **classically forbidden** - its kinetic energy would be negative!

Quantum mechanically, the wave function **doesn't vanish** but **decays exponentially**:

$$\psi(x) \sim e^{-\kappa|x|} \quad \text{for } |x| > a$$

#### Penetration Depth

The characteristic length scale of this decay is the **penetration depth**:

$$\boxed{\delta = \frac{1}{\kappa} = \frac{\hbar}{\sqrt{2m|E|}}}$$

Physical interpretation:
- Small $|E|$ (weakly bound): large penetration, wave function extends far
- Large $|E|$ (tightly bound): small penetration, wave function more confined

### 5. Probability in Forbidden Region

The probability of finding the particle outside the well is:

$$P_{\text{outside}} = 2\int_a^{\infty}|\psi(x)|^2 dx$$

#### For Even Parity

$$P_{\text{outside}} = 2C^2\int_a^{\infty}e^{-2\kappa x}dx = 2C^2 \cdot \frac{e^{-2\kappa a}}{2\kappa} = \frac{C^2 e^{-2\kappa a}}{\kappa}$$

Using $C = A\cos(ka)e^{\kappa a}$:

$$P_{\text{outside}} = \frac{A^2\cos^2(ka)}{\kappa}$$

For the ground state (where $\cos(ka) \approx 1$ for weak binding):

$$\boxed{P_{\text{outside}} \approx \frac{A^2}{\kappa} \approx \frac{1}{\kappa a + 1}}$$

### 6. Energy Dependence of Penetration

| Binding Strength | $\kappa a$ | $\delta/a$ | $P_{\text{outside}}$ |
|------------------|------------|------------|----------------------|
| Weak ($|E| \ll V_0$) | Small | Large | Up to 50% |
| Strong ($|E| \sim V_0$) | Large | Small | Few % |
| Infinite well limit | $\infty$ | 0 | 0% |

### 7. Wave Function Visualization

The ground state wave function has these features:

```
ψ(x)
  ^
  |      Exponential     Cosine      Exponential
  |        tail                        tail
  |        ╱╲                          ╱╲
  |       ╱  ╲     ╱──────────╲       ╱  ╲
  |      ╱    ╲   ╱            ╲     ╱    ╲
  |     ╱      ╲ ╱              ╲   ╱      ╲
  |    ╱        X                X         ╲
  |   ╱                                      ╲
  |  ╱                                        ╲
  +────────────────────────────────────────────> x
         -a     0      a

  |←──────────────→|      |←──────────────→|
    Classically           Classically
    forbidden             forbidden
```

### 8. Comparison: Finite vs Infinite Well

| Property | Infinite Well | Finite Well |
|----------|---------------|-------------|
| Wave function at boundary | $\psi = 0$ | $\psi \neq 0$ (continuous) |
| Derivative at boundary | Discontinuous | Continuous |
| Outside well | $\psi = 0$ | $\psi \sim e^{-\kappa|x|}$ |
| Effective width | $L$ | $> L$ (penetration) |
| Ground state energy | $E_1 = \frac{\pi^2\hbar^2}{2mL^2}$ | Lower (larger effective width) |
| Number of bound states | Infinite | Finite |

### 9. Effective Width and Lowered Energy

Because the wave function extends beyond the classical boundaries, the particle has a larger "effective box size."

Approximate effective width:

$$L_{\text{eff}} \approx 2a + 2\delta = 2a + \frac{2}{\kappa}$$

By the infinite well formula, this gives approximately:

$$E \approx -V_0 + \frac{\pi^2\hbar^2}{2m L_{\text{eff}}^2}$$

The finite well ground state energy is **lower** than the corresponding infinite well prediction.

### 10. Asymptotic Behavior

#### Near the Threshold ($E \to 0^-$)

As the binding becomes very weak:
- $\kappa \to 0$
- $\delta \to \infty$
- Wave function becomes very spread out
- Particle is barely bound

#### Deep Binding ($|E| \to V_0$)

As binding approaches the well depth:
- $\kappa \to \sqrt{2mV_0}/\hbar$
- $k \to 0$
- Wave function becomes flat inside, sharp decay outside
- Approaches ground state of very deep well

---

## Physical Interpretation

### Quantum Tunneling Precursor

The penetration of the wave function into classically forbidden regions is the **precursor to quantum tunneling**. In a finite barrier (rather than infinite), this exponential tail can:

1. Connect to another allowed region
2. Enable the particle to "tunnel" through the barrier
3. Create finite transmission probability

### The Uncertainty Principle Perspective

Why can the particle exist where $E < V$?

The uncertainty principle provides insight:
- Localizing to the forbidden region requires $\Delta x \sim \delta$
- This implies momentum uncertainty $\Delta p \sim \hbar/\delta$
- The kinetic energy uncertainty $\sim (\Delta p)^2/(2m) = \hbar^2/(2m\delta^2) = |E|$

The "negative kinetic energy" is hidden within the quantum uncertainty!

### Connection to Alpha Decay

In nuclear physics, alpha particles are bound inside nuclei by finite potential wells. The penetration depth determines:
- Probability of being at the nuclear surface
- Tunneling rate through the Coulomb barrier
- Alpha decay half-life

---

## Quantum Computing Connection

### Quantum Dot Wave Functions

Real semiconductor quantum dots have finite barriers (band offsets):
- GaAs/AlGaAs: $V_0 \approx 0.2-0.3$ eV
- InAs/GaAs: $V_0 \approx 0.5$ eV

The penetration affects:
- **Qubit coherence**: Interaction with surrounding material
- **Exchange coupling**: Overlap between adjacent dot wave functions
- **Gate fidelity**: Sensitivity to electric field fluctuations

### Tunnel Coupling in Double Dots

For two adjacent quantum dots, the overlap of exponential tails creates tunnel coupling:

$$t \propto e^{-\kappa d}$$

where $d$ is the barrier width. This coupling enables:
- Spin exchange (√SWAP gates)
- Charge transfer (readout)
- Two-qubit entangling operations

### Transmon Penetration

In superconducting transmon qubits, the Josephson potential creates a finite well. Wave function penetration:
- Reduces charge dispersion (good for coherence)
- Determines anharmonicity
- Affects inter-qubit coupling

---

## Worked Examples

### Example 1: Ground State Wave Function

**Problem:** For a finite well with $z_0 = 2$, construct the normalized ground state wave function.

**Solution:**

From Day 375, we found $\xi_1 \approx 1.03$ and $\eta_1 \approx 1.71$.

So:
- $k = \xi_1/a = 1.03/a$
- $\kappa = \eta_1/a = 1.71/a$

The wave function is (even parity):

$$\psi(x) = \begin{cases} Ce^{\kappa x} & x < -a \\ A\cos(kx) & |x| < a \\ Ce^{-\kappa x} & x > a \end{cases}$$

Matching at $x = a$:
$$C = A\cos(ka)e^{\kappa a} = A\cos(1.03)e^{1.71} = A \times 0.515 \times 5.53 = 2.85A$$

For normalization, we need:

$$1 = 2 \times (2.85A)^2 \times \frac{e^{-2 \times 1.71}}{2 \times 1.71/a} + A^2 \times a\left(1 + \frac{\sin(2.06)}{2.06}\right)$$

$$1 = \frac{16.2 A^2 a \times e^{-3.42}}{3.42} + A^2 a \times 1.43$$

$$1 = A^2 a \times (0.156 + 1.43) = 1.59 A^2 a$$

$$\boxed{A = \sqrt{\frac{1}{1.59a}} = \frac{0.79}{\sqrt{a}}}$$

The penetration depth is $\delta = a/1.71 = 0.58a$.

---

### Example 2: Probability Outside the Well

**Problem:** For the ground state in Example 1, calculate the probability of finding the particle outside the well.

**Solution:**

$$P_{\text{outside}} = 2\int_a^{\infty} C^2 e^{-2\kappa x}dx = 2C^2 \times \frac{e^{-2\kappa a}}{2\kappa}$$

$$P_{\text{outside}} = \frac{C^2 e^{-2\kappa a}}{\kappa}$$

With $C = 2.85A$, $\kappa = 1.71/a$, $\kappa a = 1.71$:

$$P_{\text{outside}} = \frac{(2.85)^2 A^2 e^{-3.42}}{1.71/a} = \frac{8.12 \times 0.63/(1.59)}{1.71/a} \times a$$

$$P_{\text{outside}} = \frac{8.12 \times 0.63}{1.59 \times 1.71} = \frac{5.12}{2.72} = 1.88 \times \frac{a}{a}$$

Wait, let me recalculate more carefully.

$$P_{\text{outside}} = \frac{C^2}{\kappa} e^{-2\kappa a} = \frac{(2.85A)^2 a}{1.71} e^{-3.42}$$

$$= \frac{8.12 A^2 a \times 0.0327}{1.71} = \frac{0.266 A^2 a}{1.71} = 0.156 A^2 a$$

Since $A^2 a = 1/1.59 = 0.63$:

$$\boxed{P_{\text{outside}} = 0.156 \times 0.63 / 0.63 = 0.156 / 1.59 \times 1.59 = 0.098 \approx 10\%}$$

About 10% of the probability is outside the well!

---

### Example 3: Penetration Depth for an Electron

**Problem:** An electron is bound in a quantum well with binding energy $|E| = 0.05$ eV. Calculate the penetration depth.

**Solution:**

$$\delta = \frac{\hbar}{\sqrt{2m_e|E|}}$$

Converting $|E| = 0.05$ eV $= 0.05 \times 1.602 \times 10^{-19}$ J $= 8.01 \times 10^{-21}$ J:

$$\delta = \frac{1.055 \times 10^{-34}}{\sqrt{2 \times 9.109 \times 10^{-31} \times 8.01 \times 10^{-21}}}$$

$$\delta = \frac{1.055 \times 10^{-34}}{\sqrt{1.46 \times 10^{-50}}} = \frac{1.055 \times 10^{-34}}{3.82 \times 10^{-25}}$$

$$\boxed{\delta = 2.76 \times 10^{-10} \text{ m} = 2.76 \text{ \AA} = 0.276 \text{ nm}}$$

This is on the atomic scale - significant penetration for shallow binding.

---

## Practice Problems

### Level 1: Direct Application

1. **Penetration depth:** Calculate $\delta$ for an electron with $|E| = 0.2$ eV.

2. **Amplitude ratio:** If $ka = 0.8$ and $\kappa a = 1.2$, find $C/A$ for an even state.

3. **Forbidden probability:** If $P_{\text{outside}} = 0.15$, what fraction is inside the well?

4. **Decay length:** How many penetration depths does it take for $|\psi|^2$ to decrease by a factor of 100?

### Level 2: Intermediate

5. **Ground vs first excited:** For $z_0 = 3$, compare the penetration depths of the ground and first excited states.

6. **Effective width:** For a well with $a = 2$ nm and $\delta = 0.5$ nm, estimate the effective width.

7. **Energy comparison:** A particle has ground state energy $E = -0.7V_0$ in a finite well. Estimate its energy if the well were infinite.

8. **Normalization check:** Verify numerically that the wave function in Example 1 is normalized.

### Level 3: Challenging

9. **Weak binding limit:** Show that as $|E| \to 0$, $P_{\text{outside}} \to 1/2$ for the ground state.

10. **Wave function matching:** Derive the ratio $B/C$ for the first excited (odd) state.

11. **Virial theorem:** For the finite well, verify the quantum virial theorem $2\langle T \rangle = \langle x \frac{dV}{dx} \rangle$.

12. **Delta function limit:** Show that as $V_0 \to \infty$ and $a \to 0$ with $2aV_0 = g$ fixed, the finite well approaches a delta function well.

---

## Computational Lab

### Exercise 1: Wave Function Visualization

```python
"""
Day 376 Computational Lab: Finite Square Well Wave Functions
Visualize bound state wave functions and penetration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Parameters
a = 1.0  # Half-width (natural units)
z0 = 3.0  # Dimensionless well strength

def find_bound_states(z0):
    """Find all bound state parameters (xi, eta, parity)"""
    states = []

    def even_eq(xi):
        if xi <= 0 or xi >= z0:
            return 1e10
        eta = xi * np.tan(xi)
        if eta < 0:
            return 1e10
        return eta**2 - (z0**2 - xi**2)

    def odd_eq(xi):
        if xi <= 0 or xi >= z0:
            return 1e10
        eta = -xi / np.tan(xi)
        if eta < 0:
            return 1e10
        return eta**2 - (z0**2 - xi**2)

    # Find even parity states
    for i in range(int(z0/(np.pi/2)) + 1):
        xi_min = i * np.pi + 0.001
        xi_max = (i + 0.5) * np.pi - 0.001
        if xi_max > z0:
            xi_max = z0 - 0.001
        if xi_min >= xi_max:
            continue
        try:
            xi = brentq(even_eq, xi_min, xi_max)
            eta = np.sqrt(z0**2 - xi**2)
            states.append((xi, eta, 'even'))
        except:
            pass

    # Find odd parity states
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
            xi = brentq(odd_eq, xi_min, xi_max)
            eta = np.sqrt(z0**2 - xi**2)
            states.append((xi, eta, 'odd'))
        except:
            pass

    # Sort by energy (eta descending = deeper binding first)
    states.sort(key=lambda x: -x[1])
    return states

def wave_function(x, xi, eta, parity, a=1.0):
    """
    Compute the (unnormalized) wave function

    Parameters:
    -----------
    x : array
        Position array
    xi : float
        Dimensionless parameter ka
    eta : float
        Dimensionless parameter kappa*a
    parity : str
        'even' or 'odd'
    a : float
        Half-width of well
    """
    k = xi / a
    kappa = eta / a

    psi = np.zeros_like(x)

    # Inside well
    inside = np.abs(x) < a
    if parity == 'even':
        psi[inside] = np.cos(k * x[inside])
    else:
        psi[inside] = np.sin(k * x[inside])

    # Outside well (x > a)
    right = x >= a
    if parity == 'even':
        psi[right] = np.cos(k * a) * np.exp(-kappa * (x[right] - a))
    else:
        psi[right] = np.sin(k * a) * np.exp(-kappa * (x[right] - a))

    # Outside well (x < -a)
    left = x <= -a
    if parity == 'even':
        psi[left] = np.cos(k * a) * np.exp(kappa * (x[left] + a))
    else:
        psi[left] = -np.sin(k * a) * np.exp(kappa * (x[left] + a))

    return psi

def normalize(x, psi):
    """Normalize wave function"""
    norm = np.sqrt(np.trapz(psi**2, x))
    return psi / norm

# Find bound states
states = find_bound_states(z0)
print(f"Found {len(states)} bound states for z₀ = {z0}")
for i, (xi, eta, parity) in enumerate(states):
    E_ratio = -(eta/z0)**2
    print(f"  State {i}: ξ={xi:.3f}, η={eta:.3f}, E/V₀={E_ratio:.3f}, {parity}")

# Create position array
x = np.linspace(-3*a, 3*a, 1000)

# Plot wave functions
fig, axes = plt.subplots(len(states), 1, figsize=(12, 3*len(states)))
if len(states) == 1:
    axes = [axes]

colors = plt.cm.viridis(np.linspace(0, 0.8, len(states)))

for i, ((xi, eta, parity), ax) in enumerate(zip(states, axes)):
    psi = wave_function(x, xi, eta, parity, a)
    psi = normalize(x, psi)
    E_ratio = -(eta/z0)**2

    ax.plot(x/a, psi, color=colors[i], linewidth=2, label=f'State {i+1} ({parity})')
    ax.fill_between(x/a, 0, psi, alpha=0.3, color=colors[i])

    # Mark well boundaries
    ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # Shade forbidden regions
    ax.axvspan(-3, -1, alpha=0.1, color='red')
    ax.axvspan(1, 3, alpha=0.1, color='red')

    # Penetration depth marker
    delta = a / eta
    ax.annotate(f'δ = {delta:.2f}a', xy=(1 + delta/a, 0.1), fontsize=10, color='red')

    ax.set_ylabel('ψ(x)', fontsize=11)
    ax.set_title(f'State {i+1}: E/V₀ = {E_ratio:.3f}, Penetration depth δ = {delta:.2f}a', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim(-3, 3)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('x/a', fontsize=12)

plt.tight_layout()
plt.savefig('fsw_wave_functions.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Exercise 2: Probability Density and Penetration

```python
"""
Analyze probability distribution and penetration for different states
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

a = 1.0
z0 = 4.0

# Find states (reusing functions from Exercise 1)
def find_bound_states(z0):
    states = []
    def even_eq(xi):
        if xi <= 0 or xi >= z0:
            return 1e10
        eta = xi * np.tan(xi)
        return eta**2 - (z0**2 - xi**2) if eta > 0 else 1e10
    def odd_eq(xi):
        if xi <= 0 or xi >= z0:
            return 1e10
        eta = -xi / np.tan(xi)
        return eta**2 - (z0**2 - xi**2) if eta > 0 else 1e10

    for i in range(int(z0/(np.pi/2)) + 1):
        xi_min, xi_max = i * np.pi + 0.001, (i + 0.5) * np.pi - 0.001
        if xi_max > z0: xi_max = z0 - 0.001
        if xi_min < xi_max:
            try:
                xi = brentq(even_eq, xi_min, xi_max)
                states.append((xi, np.sqrt(z0**2 - xi**2), 'even'))
            except: pass

    for i in range(int(z0/np.pi) + 1):
        xi_min, xi_max = (i + 0.5) * np.pi + 0.001, (i + 1) * np.pi - 0.001
        if xi_min >= z0: continue
        if xi_max > z0: xi_max = z0 - 0.001
        if xi_min < xi_max:
            try:
                xi = brentq(odd_eq, xi_min, xi_max)
                states.append((xi, np.sqrt(z0**2 - xi**2), 'odd'))
            except: pass

    states.sort(key=lambda x: -x[1])
    return states

def wave_function(x, xi, eta, parity, a=1.0):
    k, kappa = xi/a, eta/a
    psi = np.zeros_like(x)
    inside = np.abs(x) < a
    if parity == 'even':
        psi[inside] = np.cos(k * x[inside])
        psi[x >= a] = np.cos(k*a) * np.exp(-kappa*(x[x >= a] - a))
        psi[x <= -a] = np.cos(k*a) * np.exp(kappa*(x[x <= -a] + a))
    else:
        psi[inside] = np.sin(k * x[inside])
        psi[x >= a] = np.sin(k*a) * np.exp(-kappa*(x[x >= a] - a))
        psi[x <= -a] = -np.sin(k*a) * np.exp(kappa*(x[x <= -a] + a))
    return psi

states = find_bound_states(z0)
x = np.linspace(-4*a, 4*a, 2000)

# Calculate probabilities for each state
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Probability densities
for i, (xi, eta, parity) in enumerate(states):
    psi = wave_function(x, xi, eta, parity, a)
    norm = np.sqrt(np.trapz(psi**2, x))
    prob = (psi/norm)**2

    ax1.plot(x/a, prob, linewidth=2, label=f'State {i+1}')

ax1.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax1.axvspan(-4, -1, alpha=0.1, color='red', label='Classically forbidden')
ax1.axvspan(1, 4, alpha=0.1, color='red')
ax1.set_xlabel('x/a', fontsize=12)
ax1.set_ylabel('|ψ(x)|²', fontsize=12)
ax1.set_title('Probability Densities', fontsize=12)
ax1.legend()
ax1.set_xlim(-4, 4)
ax1.grid(True, alpha=0.3)

# Right plot: Probability outside vs state number
P_outside = []
for xi, eta, parity in states:
    psi = wave_function(x, xi, eta, parity, a)
    norm = np.sqrt(np.trapz(psi**2, x))
    prob = (psi/norm)**2

    # Integrate outside
    outside = np.abs(x) > a
    P_out = np.trapz(prob[outside], x[outside])
    P_outside.append(P_out)

state_nums = list(range(1, len(states) + 1))
colors = ['blue' if s[2]=='even' else 'red' for s in states]
ax2.bar(state_nums, P_outside, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('State Number', fontsize=12)
ax2.set_ylabel('Probability Outside Well', fontsize=12)
ax2.set_title('Penetration Probability', fontsize=12)
ax2.set_xticks(state_nums)
ax2.grid(True, alpha=0.3, axis='y')

# Add legend for parity
ax2.bar([], [], color='blue', alpha=0.7, label='Even parity')
ax2.bar([], [], color='red', alpha=0.7, label='Odd parity')
ax2.legend()

# Print values
print("\nProbability in classically forbidden region:")
print("-"*50)
for i, ((xi, eta, parity), P) in enumerate(zip(states, P_outside)):
    print(f"State {i+1} ({parity}): P_outside = {P:.3f} = {P*100:.1f}%")

plt.tight_layout()
plt.savefig('fsw_probability_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Exercise 3: Comparison with Infinite Well

```python
"""
Compare finite and infinite well wave functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Set up finite well
a = 1.0
z0 = 5.0  # Moderately deep well

def find_ground_state(z0):
    def eq(xi):
        eta = xi * np.tan(xi)
        return eta**2 - (z0**2 - xi**2) if eta > 0 else 1e10
    xi = brentq(eq, 0.01, np.pi/2 - 0.01)
    return xi, np.sqrt(z0**2 - xi**2)

xi, eta = find_ground_state(z0)
print(f"Finite well: ξ = {xi:.4f}, η = {eta:.4f}")
print(f"Penetration depth: δ/a = {1/eta:.4f}")

# Finite well wave function
def psi_finite(x, xi, eta, a):
    k, kappa = xi/a, eta/a
    psi = np.zeros_like(x)
    inside = np.abs(x) < a
    psi[inside] = np.cos(k * x[inside])
    psi[x >= a] = np.cos(k*a) * np.exp(-kappa*(x[x >= a] - a))
    psi[x <= -a] = np.cos(k*a) * np.exp(kappa*(x[x <= -a] + a))
    return psi

# Infinite well wave function (width 2a, centered at 0)
def psi_infinite(x, a):
    psi = np.zeros_like(x)
    inside = np.abs(x) < a
    psi[inside] = np.cos(np.pi * x[inside] / (2*a))
    return psi

x = np.linspace(-3*a, 3*a, 1000)

psi_f = psi_finite(x, xi, eta, a)
psi_f /= np.sqrt(np.trapz(psi_f**2, x))

psi_i = psi_infinite(x, a)
psi_i /= np.sqrt(np.trapz(psi_i**2, x))

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Wave functions
ax1.plot(x/a, psi_f, 'b-', linewidth=2, label='Finite well')
ax1.plot(x/a, psi_i, 'r--', linewidth=2, label='Infinite well')
ax1.axvline(x=-1, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.set_xlabel('x/a', fontsize=12)
ax1.set_ylabel('ψ(x)', fontsize=12)
ax1.set_title(f'Ground State Wave Functions (z₀ = {z0})', fontsize=12)
ax1.legend()
ax1.set_xlim(-2.5, 2.5)
ax1.grid(True, alpha=0.3)

# Probability densities
ax2.plot(x/a, psi_f**2, 'b-', linewidth=2, label='Finite well')
ax2.plot(x/a, psi_i**2, 'r--', linewidth=2, label='Infinite well')
ax2.axvline(x=-1, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
ax2.fill_between(x/a, 0, psi_f**2, alpha=0.2, color='blue')
ax2.axvspan(-2.5, -1, alpha=0.1, color='green', label='Penetration region')
ax2.axvspan(1, 2.5, alpha=0.1, color='green')
ax2.set_xlabel('x/a', fontsize=12)
ax2.set_ylabel('|ψ(x)|²', fontsize=12)
ax2.set_title('Probability Densities', fontsize=12)
ax2.legend()
ax2.set_xlim(-2.5, 2.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fsw_vs_isw.png', dpi=150, bbox_inches='tight')
plt.show()

# Quantitative comparison
print("\nQuantitative Comparison:")
print("-"*50)
print(f"Finite well ξ = ka = {xi:.4f}")
print(f"Infinite well would have ka = π/2 = {np.pi/2:.4f}")
print(f"Ratio: {xi/(np.pi/2):.4f}")
print(f"\nThe finite well wave function has:")
print(f"  - Lower effective momentum (k < π/2a)")
print(f"  - Larger effective wavelength")
print(f"  - Lower kinetic energy (lower total energy)")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Even parity wave function | $\psi = A\cos(kx)$ inside, $Ce^{-\kappa|x|}$ outside |
| Odd parity wave function | $\psi = B\sin(kx)$ inside, $\pm Ce^{-\kappa|x|}$ outside |
| Penetration depth | $\delta = 1/\kappa = \hbar/\sqrt{2m|E|}$ |
| Amplitude ratio (even) | $C/A = \cos(ka)e^{\kappa a}$ |
| Amplitude ratio (odd) | $C/B = \sin(ka)e^{\kappa a}$ |
| Outside probability | $P_{\text{out}} = C^2 e^{-2\kappa a}/\kappa$ |

### Main Takeaways

1. **Wave functions have exponential tails** that penetrate into classically forbidden regions

2. **Penetration depth** $\delta = \hbar/\sqrt{2m|E|}$ decreases with binding energy

3. **Finite barriers** result in lower energies than infinite wells due to larger effective widths

4. **Weakly bound states** have large penetration (up to 50% probability outside)

5. Wave function and its derivative must be **continuous everywhere** (unlike infinite well)

---

## Daily Checklist

- [ ] I can construct complete wave functions for both parities
- [ ] I understand the normalization procedure
- [ ] I can calculate and interpret penetration depth
- [ ] I can compute probability in the forbidden region
- [ ] I understand the energy-penetration relationship
- [ ] I can compare finite vs infinite well solutions
- [ ] I completed the visualization exercises

---

## Preview: Day 377

Tomorrow we focus on the **boundary matching conditions** in greater depth:

- The **logarithmic derivative** matching technique
- **Shooting method** for numerical eigenvalue determination
- Connection to general 1D scattering theory
- The **deep well limit** and approach to infinite well

We'll develop computational tools that apply to arbitrary 1D potentials.

---

*Day 376 of QSE Self-Study Curriculum*
*Week 54: Bound States - Infinite and Finite Wells*
*Month 14: One-Dimensional Systems*
