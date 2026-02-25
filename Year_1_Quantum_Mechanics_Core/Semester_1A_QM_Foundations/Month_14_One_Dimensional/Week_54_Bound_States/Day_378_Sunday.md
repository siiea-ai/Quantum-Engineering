# Day 378: Week 54 Review - Comprehensive Lab and Assessment

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Comprehensive review and synthesis |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Full computational lab |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Self-assessment and Week 55 preview |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you will be able to:

1. Synthesize knowledge from the entire week on bound states
2. Compare and contrast infinite vs finite square wells
3. Implement a complete eigenvalue solver using multiple methods
4. Solve comprehensive problems integrating all concepts
5. Demonstrate mastery through self-assessment
6. Prepare for the quantum harmonic oscillator (Week 55)

---

## Week 54 Summary

### Conceptual Overview

This week covered the foundational bound state problems in quantum mechanics:

```
Week 54: Bound States
├── Infinite Square Well (Days 372-374)
│   ├── Energy quantization: E_n = n²π²ℏ²/2mL²
│   ├── Eigenfunctions: ψ_n = √(2/L) sin(nπx/L)
│   ├── Orthonormality and completeness
│   └── Time evolution and quantum revivals
│
└── Finite Square Well (Days 375-377)
    ├── Transcendental eigenvalue equations
    ├── Wave function penetration
    ├── Penetration depth: δ = ℏ/√(2m|E|)
    └── Matching conditions and numerical methods
```

### Key Comparisons

| Property | Infinite Square Well | Finite Square Well |
|----------|---------------------|-------------------|
| **Potential** | $V = 0$ inside, $\infty$ outside | $V = -V_0$ inside, $0$ outside |
| **Boundary conditions** | $\psi = 0$ at walls | $\psi$ continuous, decays outside |
| **Number of states** | Infinite | Finite (depends on $V_0$) |
| **Energy spectrum** | $E_n = n^2 E_1$ | Requires transcendental equation |
| **Wave function at boundary** | Zero | Non-zero (exponential tail) |
| **Effective width** | $L$ | $> L$ (due to penetration) |
| **Analytical solution** | Complete | Only implicit |
| **Physical realization** | Idealization | Quantum wells, dots |

### Master Formula Sheet

#### Infinite Square Well

$$\boxed{E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}, \quad \psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)}$$

$$\langle\psi_m|\psi_n\rangle = \delta_{mn}, \quad T_{\text{rev}} = \frac{4mL^2}{\pi\hbar}$$

#### Finite Square Well

$$\boxed{k^2 + \kappa^2 = \frac{2mV_0}{\hbar^2}, \quad z_0 = \frac{a}{\hbar}\sqrt{2mV_0}}$$

**Even parity:** $k\tan(ka) = \kappa$

**Odd parity:** $-k\cot(ka) = \kappa$

$$\delta = \frac{1}{\kappa} = \frac{\hbar}{\sqrt{2m|E|}}, \quad N \approx \left\lfloor\frac{2z_0}{\pi}\right\rfloor + 1$$

---

## Comprehensive Review

### Topic 1: Energy Quantization

**Key Insight:** Quantization arises from boundary conditions, not the Schrodinger equation itself.

For the ISW, requiring $\psi(0) = \psi(L) = 0$ restricts:
$$k = \frac{n\pi}{L}$$

For the FSW, requiring decay at infinity and smoothness at boundaries restricts $k$ and $\kappa$ to satisfy transcendental equations.

### Topic 2: Wave Function Properties

**ISW eigenfunctions** are purely oscillatory with fixed nodes:
- $n - 1$ nodes inside
- Zero at boundaries
- No evanescent component

**FSW eigenfunctions** have both oscillatory and evanescent parts:
- Oscillatory inside ($\sim \cos(kx)$ or $\sin(kx)$)
- Exponential tails outside ($\sim e^{-\kappa|x|}$)
- Smooth matching at boundaries

### Topic 3: Time Evolution

For any initial state $\Psi(x, 0)$:

1. Expand in energy eigenstates: $\Psi(x, 0) = \sum_n c_n \psi_n(x)$
2. Evolve each component: $\Psi(x, t) = \sum_n c_n \psi_n(x) e^{-iE_n t/\hbar}$
3. Observe interference and dynamics

**ISW special feature:** Exact revivals at $T_{\text{rev}}$ due to $E_n \propto n^2$.

### Topic 4: Classical Correspondence

As $n \to \infty$ or as $\hbar \to 0$:
- Energy spacing becomes negligible compared to total energy
- Wave packet dynamics approach classical particle motion
- Probability density averages to classical uniform distribution

---

## Comprehensive Problem Set

### Part A: Infinite Square Well

**Problem A1: Electron in a Carbon Nanotube**

A single-walled carbon nanotube (SWCNT) has a segment of length $L = 5$ nm where electrons are confined.

(a) Calculate the ground state energy $E_1$ in eV.

(b) What is the wavelength of light needed for the $1 \to 2$ transition?

(c) At what temperature does thermal energy $k_B T$ equal the level spacing $E_2 - E_1$?

**Solution:**

(a) $E_1 = \frac{\pi^2 (1.055 \times 10^{-34})^2}{2(9.109 \times 10^{-31})(5 \times 10^{-9})^2}$

$E_1 = \frac{1.096 \times 10^{-67}}{4.55 \times 10^{-47}} = 2.41 \times 10^{-21}$ J $= \boxed{0.015 \text{ eV}}$

(b) $\Delta E = E_2 - E_1 = 3E_1 = 0.045$ eV

$\lambda = \frac{hc}{\Delta E} = \frac{1240 \text{ eV·nm}}{0.045 \text{ eV}} = \boxed{27.6 \text{ μm}}$ (mid-infrared)

(c) $k_B T = 3E_1 = 0.045$ eV

$T = \frac{0.045 \times 1.602 \times 10^{-19}}{1.38 \times 10^{-23}} = \boxed{522 \text{ K}}$

---

**Problem A2: Superposition Dynamics**

A particle in an ISW is prepared in the state:

$$\Psi(x, 0) = \sqrt{\frac{3}{2L}}\sin\left(\frac{\pi x}{L}\right) + \sqrt{\frac{1}{2L}}\sin\left(\frac{2\pi x}{L}\right)$$

(a) What are the expansion coefficients $c_1$ and $c_2$?

(b) What is the probability of measuring $E_1$?

(c) At what times is $|\Psi(x, t)|^2 = |\Psi(x, 0)|^2$?

**Solution:**

(a) Comparing with $\Psi = c_1\psi_1 + c_2\psi_2$:

$c_1 = \sqrt{3/2} \cdot \sqrt{2/L} / \sqrt{2/L} = \sqrt{3/2} \approx \boxed{0.866}$

$c_2 = \sqrt{1/2} \approx \boxed{0.707}$

Wait, let me recalculate. The given state is already normalized:

Check: $|c_1|^2 + |c_2|^2 = 3/(2) \times (L/L) + 1/(2) \times (L/L)$...

Actually, the normalization: $\int |\Psi|^2 dx = 3/(2L) \cdot L/2 + 1/(2L) \cdot L/2 = 3/4 + 1/4 = 1$ ✓

Hmm, but we need to extract $c_n$ from $\Psi = \sum c_n \psi_n$ where $\psi_n = \sqrt{2/L}\sin(n\pi x/L)$.

$\sqrt{3/(2L)} = c_1 \sqrt{2/L} \Rightarrow c_1 = \sqrt{3/4} = \sqrt{3}/2 \approx 0.866$

$\sqrt{1/(2L)} = c_2 \sqrt{2/L} \Rightarrow c_2 = \sqrt{1/4} = 1/2 = 0.5$

Check: $|c_1|^2 + |c_2|^2 = 3/4 + 1/4 = 1$ ✓

(b) $P(E_1) = |c_1|^2 = \boxed{3/4 = 75\%}$

(c) The probability density returns when all relative phases return to their initial values.

Phase difference: $\Delta\phi = (E_2 - E_1)t/\hbar = 3E_1 t/\hbar$

For $|\Psi(t)|^2 = |\Psi(0)|^2$, need $\Delta\phi = 2\pi n$.

$t = \frac{2\pi n \hbar}{3E_1} = \frac{n}{3} \cdot \frac{2\pi\hbar}{E_1}$

Since $T_{\text{rev}} = 2\pi\hbar \cdot 2/(E_1) = 4\pi\hbar/E_1$... wait, let me recalculate.

$T_{\text{rev}} = 4mL^2/(\pi\hbar)$

Also $E_1 = \pi^2\hbar^2/(2mL^2)$, so $T_{\text{rev}} = 4mL^2/(\pi\hbar) = 2\pi\hbar/(E_1/2) = 4\pi\hbar/E_1$... hmm.

Actually, more directly: $\omega_1 = E_1/\hbar$, so $T_1 = 2\pi\hbar/E_1$.

The beat period is $T_{21} = 2\pi\hbar/(3E_1) = T_1/3$.

$\boxed{t = n \cdot T_{21} = n \cdot T_{\text{rev}}/6 = 2nmL^2/(3\pi\hbar)}$

---

### Part B: Finite Square Well

**Problem B1: Quantum Well Design**

Design a GaAs/AlGaAs quantum well to have exactly 3 bound states.

Given: Electron effective mass $m^* = 0.067m_e$, band offset $V_0 = 0.25$ eV.

Find the required well width $2a$.

**Solution:**

For 3 bound states, need $\pi < z_0 < 3\pi/2$, so we choose $z_0 = 1.2\pi = 3.77$.

$$z_0 = \frac{a}{\hbar}\sqrt{2m^* V_0}$$

$$a = \frac{z_0 \hbar}{\sqrt{2m^* V_0}}$$

Computing:

$m^* = 0.067 \times 9.109 \times 10^{-31} = 6.10 \times 10^{-32}$ kg

$V_0 = 0.25 \times 1.602 \times 10^{-19} = 4.01 \times 10^{-20}$ J

$\sqrt{2m^* V_0} = \sqrt{2 \times 6.10 \times 10^{-32} \times 4.01 \times 10^{-20}}$

$= \sqrt{4.89 \times 10^{-51}} = 6.99 \times 10^{-26}$ kg·m/s

$a = \frac{3.77 \times 1.055 \times 10^{-34}}{6.99 \times 10^{-26}} = 5.69 \times 10^{-9}$ m

$$\boxed{2a \approx 11.4 \text{ nm}}$$

---

**Problem B2: Penetration Probability**

For the ground state of a finite well with $z_0 = 2$:

(a) Find $\xi_1$ and $\eta_1$ (the dimensionless parameters)

(b) Calculate the probability of finding the particle outside the well

**Solution:**

(a) Solve $\xi\tan\xi = \sqrt{4 - \xi^2}$ numerically.

By iteration or graphical method: $\xi_1 \approx 1.03$, $\eta_1 = \sqrt{4 - 1.06} \approx 1.71$

(b) Using the results from Day 376:

$P_{\text{outside}} = \frac{A^2\cos^2(ka)}{\kappa}$ normalized appropriately.

For $\xi = 1.03$: $\cos(\xi) = \cos(1.03) \approx 0.515$

The probability can be estimated as:

$$P_{\text{outside}} \approx \frac{\cos^2(\xi)}{\eta + \sin(\xi)\cos(\xi)}$$

Wait, using the proper normalization from Day 376:

$$P_{\text{outside}} = \frac{\cos^2(\xi)/\eta}{\left(1 + \frac{\sin(2\xi)}{2\xi}\right) + \frac{\cos^2(\xi)}{\eta}}$$

With $\xi = 1.03$, $\eta = 1.71$:

$\cos^2(1.03) = 0.265$

$\sin(2.06) = 0.891$

Denominator: $1 + 0.891/2.06 + 0.265/1.71 = 1 + 0.43 + 0.15 = 1.58$

$$P_{\text{outside}} \approx \frac{0.15}{1.58} \approx \boxed{10\%}$$

---

### Part C: Synthesis Problems

**Problem C1: Correspondence Limit**

For a particle in an infinite square well, show that as $n \to \infty$:

(a) The time-averaged probability density $\overline{|\psi_n|^2}$ approaches $1/L$

(b) The fractional energy spacing $(E_{n+1} - E_n)/E_n$ approaches zero

**Solution:**

(a) $|\psi_n(x)|^2 = \frac{2}{L}\sin^2\left(\frac{n\pi x}{L}\right)$

For large $n$, the oscillations are rapid. Averaging over many periods:

$\overline{\sin^2(n\pi x/L)} = 1/2$

Therefore $\overline{|\psi_n|^2} = \frac{2}{L} \cdot \frac{1}{2} = \boxed{\frac{1}{L}}$ ✓

(b) $\frac{E_{n+1} - E_n}{E_n} = \frac{(n+1)^2 - n^2}{n^2} = \frac{2n + 1}{n^2} \approx \frac{2}{n} \to \boxed{0}$ as $n \to \infty$ ✓

---

**Problem C2: Quantum Computing Application**

Two identical quantum dots (finite wells with $z_0 = 3$) are separated by a barrier of width $d = 5$ nm and height equal to the well depth.

(a) Estimate the tunnel coupling $t$ between the dots (assuming ground states).

(b) If an electron is initialized in the left dot, what is the period of oscillation between dots?

**Solution:**

(a) The tunnel coupling depends on the overlap of wave function tails:

$$t \sim E_1 \cdot e^{-\kappa d}$$

For $z_0 = 3$, the ground state has $\eta \approx 2.68$, so $\kappa = 2.68/a$.

With typical $a \sim 5$ nm: $\kappa \approx 0.54$ nm$^{-1}$

$\kappa d = 0.54 \times 5 = 2.7$

$e^{-\kappa d} = e^{-2.7} \approx 0.067$

If $E_1 \sim 50$ meV: $t \sim 50 \times 0.067 \approx \boxed{3.4 \text{ meV}}$

(b) The symmetric and antisymmetric states are split by $2t$.

The Rabi frequency: $\omega_R = 2t/\hbar$

Period: $T = \pi\hbar/t = \frac{\pi \times 6.58 \times 10^{-13}}{3.4 \times 10^{-3}} \text{ eV·s / eV}$

$T = 6.1 \times 10^{-10}$ s $= \boxed{0.61 \text{ ns}}$

---

## Comprehensive Computational Lab

### Complete Eigenvalue Solver

```python
"""
Day 378: Complete Bound State Solver
Comprehensive implementation comparing ISW and FSW
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import odeint
from scipy.linalg import eigh_tridiagonal

class InfiniteSquareWell:
    """Infinite Square Well Solver"""

    def __init__(self, L, m=1.0, hbar=1.0):
        self.L = L
        self.m = m
        self.hbar = hbar

    def energy(self, n):
        """Energy of nth state (n = 1, 2, 3, ...)"""
        return (n * np.pi * self.hbar)**2 / (2 * self.m * self.L**2)

    def psi(self, x, n):
        """Normalized wave function"""
        return np.sqrt(2/self.L) * np.sin(n * np.pi * x / self.L)

    def psi_t(self, x, t, coeffs):
        """Time-evolved wave function given expansion coefficients"""
        psi = np.zeros_like(x, dtype=complex)
        for n, cn in coeffs:
            omega = self.energy(n) / self.hbar
            psi += cn * self.psi(x, n) * np.exp(-1j * omega * t)
        return psi

    def revival_time(self):
        """Quantum revival time"""
        return 4 * self.m * self.L**2 / (np.pi * self.hbar)


class FiniteSquareWell:
    """Finite Square Well Solver"""

    def __init__(self, a, V0, m=1.0, hbar=1.0):
        self.a = a
        self.V0 = V0
        self.m = m
        self.hbar = hbar
        self.z0 = a * np.sqrt(2 * m * V0) / hbar

    def _even_equation(self, xi):
        """Transcendental equation for even parity"""
        if xi <= 0 or xi >= self.z0:
            return 1e10
        eta = xi * np.tan(xi)
        if eta < 0:
            return 1e10
        return eta**2 - (self.z0**2 - xi**2)

    def _odd_equation(self, xi):
        """Transcendental equation for odd parity"""
        if xi <= 0 or xi >= self.z0:
            return 1e10
        eta = -xi / np.tan(xi)
        if eta < 0:
            return 1e10
        return eta**2 - (self.z0**2 - xi**2)

    def find_bound_states(self):
        """Find all bound states"""
        states = []

        # Even parity
        for i in range(int(self.z0 / (np.pi/2)) + 1):
            xi_min = i * np.pi + 0.001
            xi_max = (i + 0.5) * np.pi - 0.001
            if xi_max > self.z0:
                xi_max = self.z0 - 0.001
            if xi_min >= xi_max:
                continue
            try:
                xi = brentq(self._even_equation, xi_min, xi_max)
                eta = np.sqrt(self.z0**2 - xi**2)
                E = -self.hbar**2 * eta**2 / (2 * self.m * self.a**2)
                states.append({'xi': xi, 'eta': eta, 'E': E, 'parity': 'even'})
            except:
                pass

        # Odd parity
        for i in range(int(self.z0 / np.pi) + 1):
            xi_min = (i + 0.5) * np.pi + 0.001
            xi_max = (i + 1) * np.pi - 0.001
            if xi_min >= self.z0:
                continue
            if xi_max > self.z0:
                xi_max = self.z0 - 0.001
            if xi_min >= xi_max:
                continue
            try:
                xi = brentq(self._odd_equation, xi_min, xi_max)
                eta = np.sqrt(self.z0**2 - xi**2)
                E = -self.hbar**2 * eta**2 / (2 * self.m * self.a**2)
                states.append({'xi': xi, 'eta': eta, 'E': E, 'parity': 'odd'})
            except:
                pass

        # Sort by energy (most negative first)
        states.sort(key=lambda s: s['E'])
        return states

    def psi(self, x, state):
        """Compute wave function for a given state"""
        xi = state['xi']
        eta = state['eta']
        k = xi / self.a
        kappa = eta / self.a

        psi = np.zeros_like(x)
        inside = np.abs(x) < self.a

        if state['parity'] == 'even':
            psi[inside] = np.cos(k * x[inside])
            psi[x >= self.a] = np.cos(k * self.a) * np.exp(-kappa * (x[x >= self.a] - self.a))
            psi[x <= -self.a] = np.cos(k * self.a) * np.exp(kappa * (x[x <= -self.a] + self.a))
        else:
            psi[inside] = np.sin(k * x[inside])
            psi[x >= self.a] = np.sin(k * self.a) * np.exp(-kappa * (x[x >= self.a] - self.a))
            psi[x <= -self.a] = -np.sin(k * self.a) * np.exp(kappa * (x[x <= -self.a] + self.a))

        # Normalize
        norm = np.sqrt(np.trapz(psi**2, x))
        return psi / norm

    def penetration_depth(self, state):
        """Return penetration depth for a state"""
        return self.a / state['eta']

    def prob_outside(self, x, state):
        """Probability outside the well"""
        psi = self.psi(x, state)
        outside = np.abs(x) > self.a
        return np.trapz(psi[outside]**2, x[outside])


def compare_wells():
    """Compare infinite and finite square wells"""

    # Parameters
    L = 2.0  # Width for ISW
    a = 1.0  # Half-width for FSW (so total width = 2a = L)
    V0 = 20.0  # Well depth for FSW

    isw = InfiniteSquareWell(L)
    fsw = FiniteSquareWell(a, V0)

    print("="*60)
    print("BOUND STATE COMPARISON: ISW vs FSW")
    print("="*60)
    print(f"ISW: L = {L}")
    print(f"FSW: 2a = {2*a}, V0 = {V0}, z0 = {fsw.z0:.2f}")
    print()

    # Find FSW states
    fsw_states = fsw.find_bound_states()

    print(f"Number of FSW bound states: {len(fsw_states)}")
    print()

    # Compare energies
    print("Energy Comparison:")
    print("-"*60)
    print(f"{'State':<10} {'ISW E':<15} {'FSW E':<15} {'FSW E + V0':<15}")
    print("-"*60)

    for i, state in enumerate(fsw_states):
        n = i + 1
        E_isw = isw.energy(n)
        E_fsw = state['E']
        print(f"{n:<10} {E_isw:<15.4f} {E_fsw:<15.4f} {E_fsw + V0:<15.4f}")

    # Create comparison plots
    x_isw = np.linspace(0, L, 500)
    x_fsw = np.linspace(-3*a, 3*a, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: ISW wave functions
    ax1 = axes[0, 0]
    for n in range(1, min(4, len(fsw_states)+1)):
        psi = isw.psi(x_isw, n)
        ax1.plot(x_isw/L, psi, linewidth=2, label=f'n={n}')
    ax1.set_xlabel('x/L', fontsize=11)
    ax1.set_ylabel('ψ(x)', fontsize=11)
    ax1.set_title('Infinite Square Well', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: FSW wave functions
    ax2 = axes[0, 1]
    for i, state in enumerate(fsw_states[:3]):
        psi = fsw.psi(x_fsw, state)
        ax2.plot(x_fsw/a, psi, linewidth=2, label=f'{state["parity"]} {i+1}')
    ax2.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('x/a', fontsize=11)
    ax2.set_ylabel('ψ(x)', fontsize=11)
    ax2.set_title('Finite Square Well', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Probability densities
    ax3 = axes[1, 0]
    psi_isw = isw.psi(x_isw, 1)
    ax3.plot(x_isw/L, psi_isw**2, 'b-', linewidth=2, label='ISW ground state')
    ax3.set_xlabel('x/L', fontsize=11)
    ax3.set_ylabel('|ψ(x)|²', fontsize=11)
    ax3.set_title('ISW Ground State Probability', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: FSW ground state with penetration
    ax4 = axes[1, 1]
    psi_fsw = fsw.psi(x_fsw, fsw_states[0])
    ax4.plot(x_fsw/a, psi_fsw**2, 'r-', linewidth=2, label='FSW ground state')
    ax4.fill_between(x_fsw/a, 0, psi_fsw**2, alpha=0.3, color='red')
    ax4.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax4.axvspan(-3, -1, alpha=0.1, color='blue', label='Forbidden region')
    ax4.axvspan(1, 3, alpha=0.1, color='blue')
    ax4.set_xlabel('x/a', fontsize=11)
    ax4.set_ylabel('|ψ(x)|²', fontsize=11)
    ax4.set_title('FSW Ground State with Penetration', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week54_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print penetration analysis
    print("\nPenetration Analysis:")
    print("-"*60)
    for i, state in enumerate(fsw_states):
        delta = fsw.penetration_depth(state)
        P_out = fsw.prob_outside(x_fsw, state)
        print(f"State {i+1}: δ/a = {delta:.3f}, P(outside) = {P_out:.3f} ({P_out*100:.1f}%)")

    return isw, fsw, fsw_states


# Run comparison
isw, fsw, states = compare_wells()
```

### Shooting Method Complete Implementation

```python
"""
Complete shooting method solver with visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def shooting_solver(a, V0, n_states=5, visualize=True):
    """
    Find bound states using shooting method

    Parameters:
    -----------
    a : float
        Half-width of well
    V0 : float
        Well depth (positive value)
    n_states : int
        Maximum number of states to find
    visualize : bool
        Whether to create plots

    Returns:
    --------
    states : list
        List of found states with energies and wave functions
    """

    # Natural units: hbar = m = 1
    z0 = a * np.sqrt(2 * V0)

    def schrodinger(x, y, E):
        """Schrodinger equation as first-order system"""
        psi, dpsi = y
        if abs(x) < a:
            k2 = 2 * (E + V0)
        else:
            k2 = 2 * E  # Note: E < 0, so k2 < 0 (evanescent)
        return [dpsi, -k2 * psi]

    def compute_mismatch(E, parity):
        """
        Compute mismatch in logarithmic derivative at x = a

        For bound states (E < 0), we expect exponential decay outside.
        """
        if E >= 0 or E <= -V0:
            return 1e10

        kappa = np.sqrt(-2 * E)

        # Initial conditions at x = 0
        if parity == 'even':
            y0 = [1.0, 0.0]
        else:
            y0 = [0.0, 1.0]

        # Integrate from 0 to a
        sol = solve_ivp(schrodinger, [0, a], y0, args=(E,),
                       t_eval=[a], dense_output=True)

        psi_a = sol.y[0, -1]
        dpsi_a = sol.y[1, -1]

        if abs(psi_a) < 1e-12:
            return 1e10

        # Required: rho = -kappa
        rho_actual = dpsi_a / psi_a
        rho_required = -kappa

        return rho_actual - rho_required

    # Search for bound states
    E_min = -V0 + 0.001
    E_max = -0.001
    E_search = np.linspace(E_min, E_max, 1000)

    states = []

    for parity in ['even', 'odd']:
        errors = [compute_mismatch(E, parity) for E in E_search]

        # Find zero crossings
        for i in range(len(errors) - 1):
            if errors[i] * errors[i+1] < 0 and abs(errors[i]) < 100 and abs(errors[i+1]) < 100:
                try:
                    E_bound = brentq(compute_mismatch, E_search[i], E_search[i+1],
                                    args=(parity,))
                    # Compute wave function
                    x_left = np.linspace(-4*a, 0, 200)
                    x_right = np.linspace(0, 4*a, 200)
                    x = np.concatenate([x_left[:-1], x_right])

                    if parity == 'even':
                        y0_left = [1.0, 0.0]
                        y0_right = [1.0, 0.0]
                    else:
                        y0_left = [0.0, -1.0]  # Odd about origin
                        y0_right = [0.0, 1.0]

                    # Integrate right side
                    sol_right = solve_ivp(schrodinger, [0, 4*a], y0_right,
                                         args=(E_bound,), t_eval=x_right,
                                         dense_output=True)

                    # For left side, integrate backwards from 0
                    sol_left = solve_ivp(schrodinger, [0, -4*a], y0_left,
                                        args=(E_bound,), t_eval=x_left[::-1],
                                        dense_output=True)

                    psi_right = sol_right.y[0]
                    psi_left = sol_left.y[0][::-1]

                    psi = np.concatenate([psi_left[:-1], psi_right])

                    # Normalize
                    norm = np.sqrt(np.trapz(psi**2, x))
                    psi = psi / norm

                    states.append({
                        'E': E_bound,
                        'parity': parity,
                        'x': x,
                        'psi': psi
                    })
                except Exception as e:
                    pass

    # Sort by energy
    states.sort(key=lambda s: s['E'])

    if visualize and states:
        fig, axes = plt.subplots(len(states), 1, figsize=(12, 3*len(states)))
        if len(states) == 1:
            axes = [axes]

        for i, (state, ax) in enumerate(zip(states, axes)):
            ax.plot(state['x']/a, state['psi'], 'b-', linewidth=2)
            ax.fill_between(state['x']/a, 0, state['psi'], alpha=0.3)
            ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='gray', linewidth=0.5)
            ax.set_ylabel('ψ(x)', fontsize=11)
            ax.set_title(f'State {i+1} ({state["parity"]}): E = {state["E"]:.4f}, E/V₀ = {state["E"]/(-V0):.4f}',
                        fontsize=11)
            ax.set_xlim(-4, 4)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('x/a', fontsize=11)

        plt.tight_layout()
        plt.savefig('shooting_results.png', dpi=150, bbox_inches='tight')
        plt.show()

    return states

# Test the shooting method
a = 1.0
V0 = 15.0  # Should give several bound states

print(f"Finite Square Well: a = {a}, V0 = {V0}")
print(f"z0 = {a * np.sqrt(2*V0):.2f}")
print(f"Expected ~{int(2*a*np.sqrt(2*V0)/np.pi) + 1} bound states")
print()

states = shooting_solver(a, V0, visualize=True)

print("\nBound states found by shooting method:")
print("-"*50)
for i, state in enumerate(states):
    print(f"State {i+1} ({state['parity']}): E = {state['E']:.6f}, E/V₀ = {state['E']/(-V0):.4f}")
```

---

## Self-Assessment Checklist

### Week 54 Mastery Checklist

Rate your confidence (1-5) on each topic:

**Infinite Square Well:**
- [ ] ___ Derive energy eigenvalues from boundary conditions
- [ ] ___ Write normalized eigenfunctions
- [ ] ___ Prove orthonormality
- [ ] ___ Explain completeness and expand arbitrary states
- [ ] ___ Calculate revival time and explain its origin
- [ ] ___ Time-evolve superposition states
- [ ] ___ Compute expectation values

**Finite Square Well:**
- [ ] ___ Set up and solve the Schrodinger equation in all regions
- [ ] ___ Derive transcendental eigenvalue equations
- [ ] ___ Find bound states graphically
- [ ] ___ Calculate penetration depth
- [ ] ___ Compute probability in forbidden region
- [ ] ___ Implement shooting method
- [ ] ___ Take the infinite well limit

**General Skills:**
- [ ] ___ Write Python code for eigenvalue problems
- [ ] ___ Visualize wave functions and probability densities
- [ ] ___ Connect to quantum computing applications
- [ ] ___ Explain classical correspondence

**Scoring:**
- 45-50: Mastery achieved
- 40-44: Strong understanding
- 35-39: Good progress, review weak areas
- Below 35: Significant review needed

---

## Preview: Week 55 - Quantum Harmonic Oscillator

Next week introduces the **quantum harmonic oscillator** (QHO), arguably the most important system in all of physics.

### Why the QHO Matters

1. **Exactly solvable** with beautiful mathematics
2. **Ubiquitous**: Approximates any potential near a minimum
3. **Foundation** for quantum field theory
4. **Applications**: Molecular vibrations, phonons, photons, trapped ions

### Key Topics

**Day 379 (Monday):** QHO Setup
- Potential: $V(x) = \frac{1}{2}m\omega^2 x^2$
- Schrodinger equation and dimensionless form

**Day 380 (Tuesday):** Analytic Solution
- Hermite polynomials $H_n(\xi)$
- Gaussian ground state: $\psi_0 \propto e^{-\xi^2/2}$
- Energy levels: $E_n = (n + 1/2)\hbar\omega$

**Day 381 (Wednesday):** Ladder Operators
- Creation: $\hat{a}^\dagger = \frac{1}{\sqrt{2}}(\hat{\xi} - i\hat{p}_\xi)$
- Annihilation: $\hat{a} = \frac{1}{\sqrt{2}}(\hat{\xi} + i\hat{p}_\xi)$
- Number operator: $\hat{N} = \hat{a}^\dagger\hat{a}$

**Day 382 (Thursday):** Coherent States
- $|\alpha\rangle = e^{-|\alpha|^2/2}\sum_n \frac{\alpha^n}{\sqrt{n!}}|n\rangle$
- Minimum uncertainty states
- Classical limit of quantum mechanics

**Days 383-385:** Applications and review

### Contrast with Square Wells

| Property | Square Well | Harmonic Oscillator |
|----------|-------------|---------------------|
| Energy spacing | $\propto n^2$ | Constant $\hbar\omega$ |
| Number of states | Finite (FSW) or infinite | Infinite |
| Ground state | $\psi \propto \sin$ | $\psi \propto e^{-x^2}$ |
| Creation/annihilation | Not natural | Central tool |
| Classical limit | Bouncing particle | Oscillating particle |

---

## Summary

### Week 54 Achievements

This week you:

1. **Mastered the infinite square well** - the paradigmatic bound state problem
2. **Learned the finite square well** - a more realistic model with penetration
3. **Developed numerical methods** - shooting, matching, matrix diagonalization
4. **Connected to applications** - quantum dots, qubits, nanostructures
5. **Prepared for the harmonic oscillator** - the next major milestone

### Looking Forward

The quantum harmonic oscillator will introduce:
- **Ladder operators** - the algebraic approach to quantum mechanics
- **Coherent states** - the closest quantum analog to classical oscillation
- **Foundation for field theory** - photons as harmonic oscillators

---

*Day 378 of QSE Self-Study Curriculum*
*Week 54: Bound States - Infinite and Finite Wells*
*Month 14: One-Dimensional Systems*

---

**Congratulations on completing Week 54!**

You now have a solid foundation in bound state quantum mechanics. The concepts of energy quantization, wave function penetration, and matching conditions will appear throughout your quantum journey.

---

*Week 54 Complete | Next: Week 55 - Quantum Harmonic Oscillator*
