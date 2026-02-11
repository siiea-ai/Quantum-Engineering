# Day 374: Infinite Square Well - Time Evolution and Quantum Revivals

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Time evolution and revival times |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem solving: Dynamics and expectation values |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational lab: Wave packet animation |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you will be able to:

1. Write the general time-dependent solution for the ISW
2. Calculate the quantum revival time and explain its physical origin
3. Analyze fractional revivals and their connection to number theory
4. Compute time-dependent expectation values $\langle x(t) \rangle$ and $\langle p(t) \rangle$
5. Visualize wave packet dynamics through probability density evolution
6. Connect quantum dynamics to classical bouncing particle behavior

---

## Core Content

### 1. Time Evolution of Energy Eigenstates

Each energy eigenstate $|\psi_n\rangle$ evolves according to the Schrodinger equation:

$$i\hbar\frac{\partial}{\partial t}|\psi_n(t)\rangle = \hat{H}|\psi_n(t)\rangle$$

Since $\hat{H}|\psi_n\rangle = E_n|\psi_n\rangle$, the solution is:

$$|\psi_n(t)\rangle = e^{-iE_n t/\hbar}|\psi_n\rangle$$

In position representation:

$$\psi_n(x, t) = \psi_n(x)e^{-i\omega_n t}$$

where the **angular frequency** associated with level $n$ is:

$$\omega_n = \frac{E_n}{\hbar} = \frac{n^2\pi^2\hbar}{2mL^2}$$

### 2. Time Evolution of Superposition States

A general initial state can be expanded in energy eigenstates:

$$\Psi(x, 0) = \sum_{n=1}^{\infty} c_n \psi_n(x)$$

The time evolution is obtained by attaching the time-dependent phases:

$$\boxed{\Psi(x, t) = \sum_{n=1}^{\infty} c_n \psi_n(x) e^{-iE_n t/\hbar}}$$

Each component oscillates at its own frequency $\omega_n$, leading to complex interference patterns.

### 3. Probability Density Dynamics

The probability density is:

$$|\Psi(x, t)|^2 = \left|\sum_n c_n \psi_n(x) e^{-iE_n t/\hbar}\right|^2$$

Expanding:

$$|\Psi(x, t)|^2 = \sum_{m,n} c_m^* c_n \psi_m^*(x)\psi_n(x) e^{-i(E_n - E_m)t/\hbar}$$

The **diagonal terms** ($m = n$) are time-independent:

$$\sum_n |c_n|^2 |\psi_n(x)|^2$$

The **off-diagonal terms** ($m \neq n$) oscillate at beat frequencies:

$$\omega_{mn} = \frac{E_n - E_m}{\hbar} = \frac{(n^2 - m^2)\pi^2\hbar}{2mL^2}$$

### 4. Quantum Revivals

#### The Revival Time

A remarkable property of the ISW is that the wave function **exactly reconstructs** itself after a specific time called the **revival time**:

$$\boxed{T_{\text{rev}} = \frac{4mL^2}{\pi\hbar}}$$

At $t = T_{\text{rev}}$, the wave function returns to its initial form:

$$\Psi(x, T_{\text{rev}}) = \Psi(x, 0)$$

#### Derivation of Revival Time

For a full revival, we need:

$$e^{-iE_n T_{\text{rev}}/\hbar} = 1 \quad \text{for all } n$$

This requires:

$$\frac{E_n T_{\text{rev}}}{\hbar} = 2\pi k_n \quad \text{for some integer } k_n$$

Substituting $E_n = n^2\pi^2\hbar^2/(2mL^2)$:

$$\frac{n^2\pi^2\hbar T_{\text{rev}}}{2mL^2} = 2\pi k_n$$

$$T_{\text{rev}} = \frac{4mL^2}{\pi\hbar} \cdot \frac{k_n}{n^2}$$

For this to work for **all** $n$, we need $k_n = n^2$, giving:

$$T_{\text{rev}} = \frac{4mL^2}{\pi\hbar}$$

#### Physical Interpretation

The revival time can be related to classical motion. The classical period for a particle bouncing in a box with energy $E_1$ is:

$$T_{\text{classical}} = \frac{2L}{v_1} = \frac{2L}{\sqrt{2E_1/m}} = \frac{2L\sqrt{m}}{\sqrt{2}\sqrt{\pi^2\hbar^2/(2mL^2)}} = \frac{2mL^2}{\pi\hbar}$$

So:

$$T_{\text{rev}} = 2 T_{\text{classical}}^{(n=1)}$$

### 5. Fractional Revivals

At rational fractions of the revival time, $t = (p/q)T_{\text{rev}}$ (with $p, q$ coprime integers), interesting structures emerge.

#### Half Revival: $t = T_{\text{rev}}/2$

At $t = T_{\text{rev}}/2$:

$$e^{-iE_n T_{\text{rev}}/(2\hbar)} = e^{-i n^2 \pi}$$

- For $n$ even: $e^{-i n^2 \pi} = 1$
- For $n$ odd: $e^{-i n^2 \pi} = -1$

The wave function becomes:

$$\Psi(x, T_{\text{rev}}/2) = \sum_{n \text{ even}} c_n \psi_n(x) - \sum_{n \text{ odd}} c_n \psi_n(x)$$

For many initial states, this is related to the **spatially translated** initial wave function!

#### Quarter Revival: $t = T_{\text{rev}}/4$

At $t = T_{\text{rev}}/4$, the wave function splits into multiple copies, creating a "cat state"-like superposition.

#### General Pattern

Fractional revivals at $t = (p/q)T_{\text{rev}}$ produce $q$ copies of the initial wave packet, each with phase shifts determined by number-theoretic properties (Gauss sums).

### 6. Expectation Values

#### Position Expectation Value

For a state $\Psi(x, t)$:

$$\langle x(t) \rangle = \int_0^L x|\Psi(x, t)|^2 dx$$

For a two-state superposition $\Psi = c_1\psi_1 + c_2\psi_2$:

$$\langle x(t) \rangle = |c_1|^2\langle x\rangle_1 + |c_2|^2\langle x\rangle_2 + 2\text{Re}[c_1^*c_2 e^{i(E_1-E_2)t/\hbar}\langle\psi_1|x|\psi_2\rangle]$$

The first two terms are stationary; the last term oscillates at frequency $\omega_{21} = (E_2 - E_1)/\hbar$.

#### Matrix Elements

For the ISW, useful matrix elements include:

$$\langle\psi_m|x|\psi_n\rangle = \frac{2}{L}\int_0^L x\sin\left(\frac{m\pi x}{L}\right)\sin\left(\frac{n\pi x}{L}\right)dx$$

For $m = n$:

$$\langle x \rangle_n = \frac{L}{2} \quad \text{(independent of } n\text{)}$$

For $m \neq n$:

$$\langle\psi_m|x|\psi_n\rangle = \frac{L}{\pi^2}\frac{4mn}{(m^2 - n^2)^2}[(-1)^{m+n} - 1]$$

This is non-zero only when $m + n$ is odd.

#### Momentum Expectation Value

Since each $\psi_n(x)$ is real and symmetric about $L/2$ (in a certain sense), the momentum expectation in an eigenstate is:

$$\langle p \rangle_n = 0$$

For superpositions, $\langle p(t) \rangle$ oscillates:

$$\langle p(t) \rangle = \frac{d}{dt}\left(m\langle x(t) \rangle\right)$$

### 7. Energy Uncertainty

For a general superposition, the energy expectation value is:

$$\langle E \rangle = \sum_n |c_n|^2 E_n$$

The energy uncertainty is:

$$(\Delta E)^2 = \sum_n |c_n|^2 E_n^2 - \left(\sum_n |c_n|^2 E_n\right)^2$$

For pure eigenstates, $\Delta E = 0$ (stationary states). For superpositions, $\Delta E > 0$, and the energy-time uncertainty relation gives:

$$\Delta E \cdot \Delta t \gtrsim \hbar$$

where $\Delta t$ characterizes the time scale of evolution.

### 8. Classical Limit and Correspondence

For high quantum numbers $n \gg 1$:

1. **Energy spacing becomes small**: $E_{n+1} - E_n \approx 2E_1 n \ll E_n$

2. **Wave packet follows classical trajectory**: A superposition of nearby levels centered at $n_0$ oscillates back and forth like a classical particle.

3. **Probability density approaches classical**: Time-averaged $|\Psi|^2$ approaches uniform distribution $1/L$.

---

## Physical Interpretation

### Why Do Revivals Occur?

Revivals are a consequence of the **commensurate** energy spectrum: $E_n = n^2 E_1$ means all frequencies $\omega_n = n^2 \omega_1$ share a common multiple.

In contrast:
- **Harmonic oscillator** ($E_n = n\omega$): Period $T = 2\pi/\omega$ for all states
- **Hydrogen atom** ($E_n \propto 1/n^2$): Incommensurate frequencies, no exact revivals

### Connection to Classical Motion

| Classical Particle | Quantum Wave Packet |
|-------------------|---------------------|
| Bounces between walls | Probability density oscillates |
| Fixed period $T = 2L/v$ | Multiple periods from different $\omega_n$ |
| Sharp position | Spread over region |
| Reversible motion | Revival restores initial state |

### Quantum Carpets

Plotting $|\Psi(x, t)|^2$ as a function of both $x$ and $t$ produces beautiful interference patterns called "quantum carpets." These patterns reflect the deep connection between quantum dynamics and number theory (Gauss sums, quadratic residues).

---

## Quantum Computing Connection

### Qubit Dynamics

For a qubit encoded in the two lowest ISW levels:

$$|\psi(t)\rangle = c_0|0\rangle + c_1 e^{-i(E_1-E_0)t/\hbar}|1\rangle$$

This precession around the Bloch sphere occurs at frequency:

$$\omega_{01} = \frac{E_1 - E_0}{\hbar} = \frac{3\pi^2\hbar}{2mL^2}$$

**Gate operations** manipulate this precession:
- $Z$-gate: Adds phase between $|0\rangle$ and $|1\rangle$
- Free evolution for time $t$: Implements $R_z(\omega_{01}t)$

### Coherence and Dephasing

In real quantum systems, coupling to the environment causes:
- **$T_1$ decay**: Energy relaxation ($|1\rangle \to |0\rangle$)
- **$T_2$ dephasing**: Loss of phase coherence (destroys revivals)

The revival time must be much shorter than $T_2$ to observe quantum revivals experimentally.

### Quantum Simulation

Cold atoms in optical lattice potentials approximate infinite square wells. Observing revivals and fractional revivals provides:
- Precision tests of quantum mechanics
- Probes of decoherence mechanisms
- Demonstrations of wave-particle duality

---

## Worked Examples

### Example 1: Two-State Superposition Dynamics

**Problem:** A particle starts in the state:
$$\Psi(x, 0) = \frac{1}{\sqrt{2}}[\psi_1(x) + \psi_2(x)]$$

Find $\Psi(x, t)$ and calculate $\langle x(t) \rangle$.

**Solution:**

The time-evolved state:

$$\Psi(x, t) = \frac{1}{\sqrt{2}}\left[\psi_1(x)e^{-iE_1 t/\hbar} + \psi_2(x)e^{-iE_2 t/\hbar}\right]$$

Factoring out a global phase:

$$\Psi(x, t) = \frac{e^{-iE_1 t/\hbar}}{\sqrt{2}}\left[\psi_1(x) + \psi_2(x)e^{-i(E_2-E_1)t/\hbar}\right]$$

The beat frequency is:

$$\omega_{21} = \frac{E_2 - E_1}{\hbar} = \frac{(4-1)\pi^2\hbar}{2mL^2} = \frac{3\pi^2\hbar}{2mL^2}$$

The period of oscillation:

$$T_{21} = \frac{2\pi}{\omega_{21}} = \frac{4mL^2}{3\pi\hbar} = \frac{T_{\text{rev}}}{3}$$

For the position expectation value, we need:

$$\langle\psi_1|x|\psi_1\rangle = \langle\psi_2|x|\psi_2\rangle = \frac{L}{2}$$

$$\langle\psi_1|x|\psi_2\rangle = \frac{2}{L}\int_0^L x\sin\left(\frac{\pi x}{L}\right)\sin\left(\frac{2\pi x}{L}\right)dx$$

Using the product formula and integration by parts:

$$\langle\psi_1|x|\psi_2\rangle = -\frac{16L}{9\pi^2}$$

Therefore:

$$\langle x(t) \rangle = \frac{1}{2}\left[\frac{L}{2} + \frac{L}{2}\right] + 2 \cdot \frac{1}{2} \cdot \left(-\frac{16L}{9\pi^2}\right)\cos(\omega_{21} t)$$

$$\boxed{\langle x(t) \rangle = \frac{L}{2} - \frac{16L}{9\pi^2}\cos\left(\frac{3\pi^2\hbar t}{2mL^2}\right)}$$

The position oscillates between approximately $0.32L$ and $0.68L$.

---

### Example 2: Revival Time Calculation

**Problem:** An electron is confined to a quantum dot of width $L = 10$ nm. Calculate:
(a) The revival time
(b) The number of classical bounces in one revival period

**Solution:**

(a) Revival time:

$$T_{\text{rev}} = \frac{4mL^2}{\pi\hbar} = \frac{4 \times 9.109 \times 10^{-31} \times (10^{-8})^2}{\pi \times 1.055 \times 10^{-34}}$$

$$T_{\text{rev}} = \frac{3.644 \times 10^{-47}}{3.314 \times 10^{-34}} = 1.10 \times 10^{-13} \text{ s}$$

$$\boxed{T_{\text{rev}} \approx 110 \text{ fs (femtoseconds)}}$$

(b) Classical period for $n = 1$:

$$T_{\text{classical}} = \frac{T_{\text{rev}}}{2} = 55 \text{ fs}$$

Number of bounces = 2 (one round trip).

For a particle in the $n = 10$ state:

$$T_{\text{classical}}^{(10)} = \frac{2mL^2}{10\pi\hbar} = \frac{T_{\text{rev}}}{20} = 5.5 \text{ fs}$$

Number of bounces = 20 round trips per revival.

---

### Example 3: Probability Density at Half Revival

**Problem:** For the superposition $\Psi(0) = (\psi_1 + \psi_3)/\sqrt{2}$, find $|\Psi(x, T_{\text{rev}}/2)|^2$.

**Solution:**

At $t = T_{\text{rev}}/2$:

$$e^{-iE_n T_{\text{rev}}/(2\hbar)} = e^{-in^2\pi} = (-1)^{n^2}$$

For odd $n$: $(-1)^{n^2} = -1$ (since $n^2$ is odd)

So:

$$\Psi(x, T_{\text{rev}}/2) = \frac{1}{\sqrt{2}}\left[(-1)\psi_1(x) + (-1)\psi_3(x)\right] = -\frac{1}{\sqrt{2}}[\psi_1(x) + \psi_3(x)]$$

The probability density:

$$|\Psi(x, T_{\text{rev}}/2)|^2 = \frac{1}{2}|\psi_1(x) + \psi_3(x)|^2 = |\Psi(x, 0)|^2$$

The probability density is **unchanged** at the half-revival for this particular superposition!

This happens because both components have the same phase factor $(-1)$.

---

## Practice Problems

### Level 1: Direct Application

1. **Phase accumulation:** After time $t$, what is the phase of the $n = 3$ component relative to $n = 1$?

2. **Beat frequency:** Calculate the beat frequency $\omega_{31}$ for the $n=3$ and $n=1$ superposition.

3. **Revival verification:** Show that $e^{-iE_n T_{\text{rev}}/\hbar} = 1$ for $n = 1, 2, 3$.

4. **Half revival phase:** What is $e^{-iE_2 T_{\text{rev}}/(2\hbar)}$?

### Level 2: Intermediate

5. **Three-state superposition:** For $\Psi(0) = (2\psi_1 + \psi_2)/\sqrt{5}$, find $|\Psi(x, t)|^2$ explicitly.

6. **Expectation value:** Calculate $\langle x \rangle$ for the ground state $\psi_1(x)$ and verify it equals $L/2$.

7. **Momentum oscillation:** For the superposition in Example 1, find $\langle p(t) \rangle$.

8. **Energy spread:** For $\Psi = (\psi_1 + \psi_3)/\sqrt{2}$, calculate $\langle E \rangle$ and $\Delta E$.

### Level 3: Challenging

9. **Fractional revival:** At $t = T_{\text{rev}}/4$, what phase does each component $n = 1, 2, 3, 4$ acquire? Interpret geometrically.

10. **Classical limit:** For a wave packet centered at $n_0 \gg 1$ with width $\Delta n \ll n_0$, show that $\langle x(t) \rangle$ follows classical motion.

11. **Ehrenfest's theorem:** Verify $\frac{d\langle x\rangle}{dt} = \frac{\langle p\rangle}{m}$ for a two-state superposition.

12. **Quantum carpet:** Derive the time scale for the finest structure in the $(x, t)$ probability plot.

---

## Computational Lab

### Exercise 1: Wave Packet Evolution Animation

```python
"""
Day 374 Computational Lab: Time Evolution in Infinite Square Well
Animate wave packet dynamics and observe quantum revivals
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Physical constants (atomic units for convenience)
# In a.u.: hbar = m = 1
L = 1.0  # Well width

def psi_n(x, n):
    """Normalized eigenfunction"""
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def E_n(n):
    """Energy eigenvalue (in units where hbar²/2m = 1)"""
    return (n * np.pi / L)**2

def time_evolve(x, t, coeffs):
    """
    Compute Psi(x, t) given expansion coefficients

    coeffs: list of (n, c_n) pairs
    """
    psi = np.zeros_like(x, dtype=complex)
    for n, cn in coeffs:
        phase = np.exp(-1j * E_n(n) * t)
        psi += cn * psi_n(x, n) * phase
    return psi

# Initial state: Gaussian-like wave packet
# Superposition of first 10 states
def gaussian_coefficients(x0, sigma, n_max=10):
    """
    Compute expansion coefficients for a Gaussian centered at x0
    """
    coeffs = []
    for n in range(1, n_max + 1):
        # Approximate c_n for Gaussian
        # c_n = integral of psi_n(x) * Gaussian
        cn = np.sqrt(2/L) * np.sqrt(2*np.pi*sigma**2) * \
             np.exp(-0.5*(n*np.pi*sigma/L)**2) * np.sin(n*np.pi*x0/L)
        coeffs.append((n, cn))

    # Normalize
    norm = np.sqrt(sum(abs(cn)**2 for _, cn in coeffs))
    coeffs = [(n, cn/norm) for n, cn in coeffs]
    return coeffs

# Parameters for wave packet
x0 = 0.25 * L  # Initial position
sigma = 0.1 * L  # Width
coeffs = gaussian_coefficients(x0, sigma, n_max=15)

print("Expansion coefficients:")
for n, cn in coeffs[:5]:
    print(f"  c_{n} = {cn:.4f}, |c_{n}|² = {abs(cn)**2:.4f}")

# Revival time (in natural units)
T_rev = 4 * L**2 / np.pi
print(f"\nRevival time: T_rev = {T_rev:.4f}")

# Create animation
x = np.linspace(0, L, 500)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Initial plots
line1, = ax1.plot([], [], 'b-', linewidth=2)
line2, = ax2.plot([], [], 'r-', linewidth=2)
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12)

ax1.set_xlim(0, L)
ax1.set_ylim(-3, 3)
ax1.set_xlabel('x/L')
ax1.set_ylabel('Re[Ψ(x,t)]')
ax1.set_title('Wave Function (Real Part)')
ax1.grid(True, alpha=0.3)

ax2.set_xlim(0, L)
ax2.set_ylim(0, 5)
ax2.set_xlabel('x/L')
ax2.set_ylabel('|Ψ(x,t)|²')
ax2.set_title('Probability Density')
ax2.grid(True, alpha=0.3)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line1, line2, time_text

def animate(frame):
    t = frame * T_rev / 100  # 100 frames per revival
    psi = time_evolve(x, t, coeffs)

    line1.set_data(x, np.real(psi))
    line2.set_data(x, np.abs(psi)**2)
    time_text.set_text(f't/T_rev = {t/T_rev:.3f}')
    return line1, line2, time_text

anim = FuncAnimation(fig, animate, init_func=init, frames=200,
                     interval=50, blit=True)

plt.tight_layout()
# anim.save('wave_evolution.gif', writer='pillow', fps=20)
plt.show()
```

### Exercise 2: Quantum Revival Visualization

```python
"""
Visualize quantum revivals and fractional revivals
"""

import numpy as np
import matplotlib.pyplot as plt

L = 1.0

def psi_n(x, n):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def E_n(n):
    return (n * np.pi / L)**2

# Superposition: equal mix of first 5 states
n_max = 5
coeffs = [(n, 1/np.sqrt(n_max)) for n in range(1, n_max + 1)]

# Revival time
T_rev = 4 * L**2 / np.pi

# Times to plot
times = [0, T_rev/4, T_rev/2, 3*T_rev/4, T_rev]
labels = ['t = 0', 't = T_rev/4', 't = T_rev/2', 't = 3T_rev/4', 't = T_rev']

x = np.linspace(0, L, 500)

fig, axes = plt.subplots(len(times), 1, figsize=(10, 12))

for ax, t, label in zip(axes, times, labels):
    psi = np.zeros_like(x, dtype=complex)
    for n, cn in coeffs:
        phase = np.exp(-1j * E_n(n) * t)
        psi += cn * psi_n(x, n) * phase

    prob = np.abs(psi)**2

    ax.fill_between(x, 0, prob, alpha=0.5, color='blue')
    ax.plot(x, prob, 'b-', linewidth=2)
    ax.set_ylabel('|Ψ|²')
    ax.set_title(label)
    ax.set_xlim(0, L)
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3)

    # Mark the initial peak position
    ax.axvline(x=L/2, color='gray', linestyle='--', alpha=0.5)

axes[-1].set_xlabel('x/L')

plt.tight_layout()
plt.savefig('quantum_revivals.png', dpi=150, bbox_inches='tight')
plt.show()

# Verify revival: compare t=0 and t=T_rev
psi_0 = np.zeros_like(x, dtype=complex)
psi_T = np.zeros_like(x, dtype=complex)

for n, cn in coeffs:
    psi_0 += cn * psi_n(x, n)
    psi_T += cn * psi_n(x, n) * np.exp(-1j * E_n(n) * T_rev)

overlap = np.abs(np.trapz(np.conj(psi_0) * psi_T, x))**2
print(f"\nOverlap |<Ψ(0)|Ψ(T_rev)>|² = {overlap:.6f}")
print("(Should be 1.0 for perfect revival)")
```

### Exercise 3: Expectation Value Dynamics

```python
"""
Track expectation values as functions of time
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

L = 1.0

def psi_n(x, n):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def E_n(n):
    return (n * np.pi / L)**2

# Two-state superposition
coeffs = [(1, 1/np.sqrt(2)), (2, 1/np.sqrt(2))]

T_rev = 4 * L**2 / np.pi
T_21 = T_rev / 3  # Beat period for n=1,2

x = np.linspace(0, L, 1000)
times = np.linspace(0, 2*T_21, 500)

# Compute expectation values
x_expect = []
p_expect = []

for t in times:
    psi = np.zeros_like(x, dtype=complex)
    for n, cn in coeffs:
        phase = np.exp(-1j * E_n(n) * t)
        psi += cn * psi_n(x, n) * phase

    prob = np.abs(psi)**2
    x_exp = trapezoid(x * prob, x)
    x_expect.append(x_exp)

# Compute momentum expectation from d<x>/dt
dt = times[1] - times[0]
p_expect = np.gradient(x_expect, dt)

# Theoretical prediction
omega_21 = E_n(2) - E_n(1)
x_theory = L/2 - (16*L/(9*np.pi**2)) * np.cos(omega_21 * times)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(times/T_21, x_expect, 'b-', linewidth=2, label='Numerical')
ax1.plot(times/T_21, x_theory, 'r--', linewidth=2, label='Analytical')
ax1.axhline(y=L/2, color='gray', linestyle=':', alpha=0.7)
ax1.set_xlabel('Time (in units of T₂₁)')
ax1.set_ylabel('⟨x⟩/L')
ax1.set_title('Position Expectation Value vs Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(times/T_21, p_expect, 'g-', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
ax2.set_xlabel('Time (in units of T₂₁)')
ax2.set_ylabel('⟨p⟩ (arb. units)')
ax2.set_title('Momentum Expectation Value vs Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('expectation_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Oscillation period T_21 = T_rev/3 = {T_21:.4f}")
print(f"⟨x⟩ oscillates between {min(x_expect)/L:.3f}L and {max(x_expect)/L:.3f}L")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Time evolution | $\Psi(x,t) = \sum_n c_n \psi_n(x) e^{-iE_n t/\hbar}$ |
| Angular frequency | $\omega_n = n^2\pi^2\hbar/(2mL^2)$ |
| Revival time | $T_{\text{rev}} = 4mL^2/(\pi\hbar)$ |
| Beat frequency | $\omega_{mn} = (n^2 - m^2)\pi^2\hbar/(2mL^2)$ |
| Half-revival phase | $e^{-iE_n T_{\text{rev}}/(2\hbar)} = (-1)^{n^2}$ |
| Position expectation | $\langle x \rangle_n = L/2$ for all $n$ |

### Main Takeaways

1. **Superposition states evolve** with each component acquiring a different phase $e^{-iE_n t/\hbar}$

2. **Quantum revivals** occur at $T_{\text{rev}} = 4mL^2/(\pi\hbar)$ due to the commensurate $n^2$ spectrum

3. **Fractional revivals** at rational fractions of $T_{\text{rev}}$ produce wave function splitting

4. **Expectation values oscillate** at beat frequencies $\omega_{mn}$

5. The **classical limit** emerges for wave packets with large $n$

---

## Daily Checklist

- [ ] I can write the time-evolved state for any initial superposition
- [ ] I understand why revivals occur and can calculate $T_{\text{rev}}$
- [ ] I can explain fractional revivals qualitatively
- [ ] I can compute time-dependent expectation values
- [ ] I understand the classical correspondence for large $n$
- [ ] I completed the wave packet animation lab
- [ ] I can relate ISW dynamics to qubit precession

---

## Preview: Day 375

Tomorrow we transition to the **finite square well**, where the potential is:

$$V(x) = \begin{cases} -V_0 & |x| < a \\ 0 & |x| > a \end{cases}$$

Unlike the infinite well:
- Wave functions **penetrate** into classically forbidden regions
- The number of bound states depends on well depth $V_0$
- Energy quantization involves **transcendental equations**

This more realistic model introduces the profound quantum phenomenon of **tunneling**.

---

*Day 374 of QSE Self-Study Curriculum*
*Week 54: Bound States - Infinite and Finite Wells*
*Month 14: One-Dimensional Systems*
