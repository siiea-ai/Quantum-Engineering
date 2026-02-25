# Day 900: Single-Qubit Gates

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Microwave control theory, Rabi oscillations, rotating frame |
| Afternoon | 2 hours | DRAG pulses, calibration protocols, problem solving |
| Evening | 2 hours | Computational lab: Pulse optimization and gate simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive** the rotating-frame Hamiltonian for driven transmon qubits
2. **Design** microwave pulses for X, Y, and arbitrary rotation gates
3. **Implement** DRAG corrections to suppress leakage to higher levels
4. **Describe** calibration protocols: Rabi, Ramsey, DRAG calibration
5. **Explain** virtual-Z gates and their advantages
6. **Calculate** gate fidelities and identify error sources

## Core Content

### 1. Qubit Control Hamiltonian

A transmon qubit driven by a microwave field has Hamiltonian:

$$\hat{H} = \frac{\hbar\omega_q}{2}\hat{\sigma}_z + \hbar\Omega(t)\cos(\omega_d t + \phi)\hat{\sigma}_x$$

where:
- $\omega_q$: qubit frequency
- $\omega_d$: drive frequency
- $\Omega(t)$: time-dependent Rabi frequency (pulse envelope)
- $\phi$: drive phase

In the **rotating frame** at drive frequency $\omega_d$, applying the transformation $\hat{U} = e^{i\omega_d t \hat{\sigma}_z/2}$:

$$\boxed{\hat{H}_{rot} = \frac{\hbar\Delta}{2}\hat{\sigma}_z + \frac{\hbar\Omega(t)}{2}(\cos\phi\,\hat{\sigma}_x + \sin\phi\,\hat{\sigma}_y)}$$

where $\Delta = \omega_q - \omega_d$ is the detuning.

### 2. Resonant Rabi Oscillations

On resonance ($\Delta = 0$) with drive phase $\phi = 0$:

$$\hat{H}_{rot} = \frac{\hbar\Omega}{2}\hat{\sigma}_x$$

The evolution operator is:

$$\hat{U}(t) = e^{-i\Omega t \hat{\sigma}_x/2} = \cos\frac{\Omega t}{2}\hat{I} - i\sin\frac{\Omega t}{2}\hat{\sigma}_x$$

Starting from $|0\rangle$:
$$|\psi(t)\rangle = \cos\frac{\Omega t}{2}|0\rangle - i\sin\frac{\Omega t}{2}|1\rangle$$

The excited state probability oscillates:

$$\boxed{P_1(t) = \sin^2\frac{\Omega t}{2} = \frac{1 - \cos(\Omega t)}{2}}$$

This is the **Rabi oscillation**—the foundation of single-qubit control.

### 3. X, Y, and Z Rotations

**X gate** ($R_x(\theta)$): Drive on resonance with $\phi = 0$

$$R_x(\theta) = e^{-i\theta\hat{\sigma}_x/2} = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

Pulse duration: $t_\pi = \theta/\Omega$ for rotation angle $\theta$.

**Y gate** ($R_y(\theta)$): Drive on resonance with $\phi = \pi/2$

$$R_y(\theta) = e^{-i\theta\hat{\sigma}_y/2} = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

**Z gate** ($R_z(\theta)$): Virtual—implemented by phase tracking (see Section 8)

$$R_z(\theta) = e^{-i\theta\hat{\sigma}_z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

### 4. Pulse Shapes

**Square pulse**: Simplest but has poor spectral properties (sinc sidelobes)

**Gaussian pulse**:
$$\Omega(t) = \Omega_0 \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right)$$

Advantages: Smooth, compact spectrum
Disadvantage: Infinite tails (must truncate)

**Gaussian derivative (DRAG)**:
$$\Omega(t) = \Omega_0 \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right)$$
$$\Omega^{DRAG}(t) = -\frac{\beta}{\alpha}\frac{d\Omega}{dt}$$

where $\beta$ is the DRAG parameter (see Section 5).

**Cosine pulse**:
$$\Omega(t) = \Omega_0 \frac{1 - \cos(2\pi t/T)}{2}$$

Compact support, smooth transitions.

### 5. DRAG Pulse Correction

The transmon is not a perfect two-level system. The third level $|2\rangle$ causes **leakage** during fast pulses.

Including the $|2\rangle$ state with frequency $\omega_{12} = \omega_{01} + \alpha$ (where $\alpha < 0$):

The DRAG (Derivative Removal by Adiabatic Gate) correction adds an out-of-phase component:

$$\boxed{\Omega_x(t) = \Omega(t), \quad \Omega_y(t) = -\frac{\dot{\Omega}(t)}{\alpha}}$$

The full drive becomes:
$$\Omega_{DRAG}(t) = \Omega(t) + i\frac{\dot{\Omega}(t)}{\alpha}$$

**Physical interpretation**: The derivative component creates destructive interference with the leakage pathway, keeping population in the qubit subspace.

**DRAG parameter**:
$$\beta = -\frac{1}{\alpha}$$

For transmon with $\alpha/2\pi = -250$ MHz: $\beta \approx 0.64$ ns.

### 6. Gate Calibration Protocols

**Rabi Calibration**: Determine pulse amplitude for $\pi$ rotation
1. Apply pulses with varying amplitude $A$
2. Measure excited state population
3. Fit to $P_1 = \sin^2(\pi A/A_\pi)$ to extract $A_\pi$

**Ramsey Experiment**: Calibrate qubit frequency
1. Apply $\pi/2$ pulse
2. Wait time $\tau$ (free evolution)
3. Apply second $\pi/2$ pulse
4. Measure
5. Fit oscillation frequency gives $\Delta = \omega_q - \omega_d$

**DRAG Calibration**: Optimize $\beta$ parameter
1. Apply $X_{\pi/2}$ then $Y_{\pi/2}$ (should give $|0\rangle$)
2. Vary $\beta$
3. Minimize leakage (maximize return to $|0\rangle$)

**AllXY Protocol**: Comprehensive pulse check
- Sequence of 21 pulse pairs testing gate errors
- Each pair should give known final state
- Detects amplitude, detuning, and phase errors

### 7. Pulse Duration and Bandwidth

**Fundamental tradeoff**: Faster gates require more power but increase leakage.

For a Gaussian pulse with width $\sigma$:
- Gate duration: $T_{gate} \approx 4\sigma$ (for 99.99% of pulse area)
- Bandwidth: $\Delta f \sim 1/\sigma$

**Leakage constraint**: Bandwidth must be less than anharmonicity
$$\frac{1}{\sigma} \ll |\alpha|$$

For $|\alpha|/2\pi = 250$ MHz: $\sigma \gtrsim 4$ ns, so $T_{gate} \gtrsim 16$ ns.

**State of the art**: Single-qubit gates in 20-50 ns with fidelity >99.99%.

### 8. Virtual-Z Gates

Z rotations don't require physical pulses! They can be implemented by **phase tracking**:

$$R_z(\theta) \cdot R_x(\phi) = R_x(\phi - \theta) \cdot R_z(\theta)$$

Instead of applying $R_z(\theta)$:
1. Update the frame of all subsequent pulses by $\theta$
2. This is mathematically equivalent to $R_z(\theta)$

**Advantages**:
- Zero duration (instantaneous)
- No decoherence
- Perfect fidelity
- Commutes to end of circuit for measurement

$$\boxed{R_z(\theta)|\psi\rangle \equiv \text{(update phase tracking by } \theta\text{)}}$$

### 9. Composite Pulses

**BB1 (Broadband 1)**: Corrects amplitude errors

$$BB1(\theta) = R_{\phi_1}(\pi) R_{\phi_2}(2\pi) R_{\phi_1}(\pi) R_0(\theta)$$

where $\phi_1 = \arccos(-\theta/4\pi)$ and $\phi_2 = 3\phi_1$.

**CORPSE**: Corrects off-resonance errors

**SK1 (Solovay-Kitaev 1)**: Corrects both simultaneously

### 10. Gate Fidelity Metrics

**Average gate fidelity**:
$$F_{avg} = \int d\psi\, |\langle\psi|U^\dagger_{ideal}U_{actual}|\psi\rangle|^2$$

For single qubit:
$$F_{avg} = \frac{1 + |\text{Tr}(U^\dagger_{ideal}U_{actual})|/2}{2}$$

**Infidelity** (error rate):
$$\epsilon = 1 - F_{avg}$$

**Randomized benchmarking**: Robust method to extract average Clifford gate fidelity
1. Apply random sequence of $m$ Clifford gates
2. Apply recovery Clifford
3. Measure survival probability
4. Fit $P(m) = A \cdot p^m + B$
5. Extract fidelity: $F = 1 - (1-p)(d-1)/d$ where $d=2$ for single qubit

## Quantum Computing Applications

### High-Fidelity Gates for Error Correction

Surface code threshold requires ~99% gate fidelity. Current superconducting systems achieve:
- Single-qubit: 99.95-99.99%
- Limited by: Decoherence during gate, control errors, leakage

### Pulse-Level Programming

Modern quantum computers expose pulse-level control:

```python
# Qiskit Pulse example
from qiskit import pulse
from qiskit.pulse import library as pulse_lib

with pulse.build() as x_gate:
    pulse.play(pulse_lib.Drag(duration=160, amp=0.1, sigma=40, beta=0.5),
               pulse.drive_channel(0))
```

### Optimal Control

GRAPE (Gradient Ascent Pulse Engineering):
- Numerically optimize pulse shape
- Maximize fidelity subject to constraints
- Can find pulses faster than DRAG with higher fidelity

## Worked Examples

### Example 1: Rabi Frequency Calculation

**Problem**: A transmon at 5 GHz is driven with microwave power such that the Rabi frequency is 50 MHz. Calculate:
(a) Duration of a $\pi$ pulse
(b) Duration of a $\pi/2$ pulse
(c) Number of Rabi oscillations in 100 ns

**Solution**:

(a) For a $\pi$ rotation: $\Omega t_\pi = \pi$
$$t_\pi = \frac{\pi}{\Omega} = \frac{\pi}{2\pi \times 50 \times 10^6} = 10 \text{ ns}$$

(b) For $\pi/2$ rotation:
$$t_{\pi/2} = \frac{\pi/2}{\Omega} = \frac{t_\pi}{2} = 5 \text{ ns}$$

(c) Rabi oscillation period: $T_{Rabi} = 2\pi/\Omega = 20$ ns

Number of oscillations in 100 ns:
$$N = \frac{100 \text{ ns}}{20 \text{ ns}} = 5 \text{ oscillations}$$

### Example 2: DRAG Pulse Design

**Problem**: Design a DRAG pulse for a transmon with $\alpha/2\pi = -280$ MHz. The base pulse is Gaussian with $\sigma = 10$ ns and peak amplitude $\Omega_0/2\pi = 40$ MHz.

**Solution**:

Base Gaussian pulse:
$$\Omega(t) = \Omega_0 \exp\left(-\frac{(t-t_0)^2}{2\sigma^2}\right)$$

Derivative:
$$\dot{\Omega}(t) = -\Omega_0 \frac{(t-t_0)}{\sigma^2}\exp\left(-\frac{(t-t_0)^2}{2\sigma^2}\right)$$

DRAG parameter:
$$\beta = -\frac{1}{\alpha} = -\frac{1}{-2\pi \times 280 \times 10^6} = 0.57 \text{ ns}$$

Y-quadrature amplitude:
$$\Omega_y(t) = \beta \dot{\Omega}(t) = -\frac{\Omega_0 \beta(t-t_0)}{\sigma^2}\exp\left(-\frac{(t-t_0)^2}{2\sigma^2}\right)$$

At $t = t_0 + \sigma$:
$$\Omega_y/\Omega_0 = -\frac{\beta}{\sigma}e^{-1/2} = -\frac{0.57}{10} \times 0.61 = -0.035$$

So the Y-component is about 3.5% of the X-component at one sigma.

### Example 3: Ramsey Detuning

**Problem**: A Ramsey experiment shows oscillations at 2.5 MHz. The drive is set to 5.000 GHz. What is the actual qubit frequency?

**Solution**:

The Ramsey oscillation frequency equals the detuning magnitude:
$$|\Delta| = 2\pi \times 2.5 \text{ MHz}$$

The qubit frequency is:
$$\omega_q = \omega_d \pm |\Delta|$$

So either:
$$f_q = 5.000 \pm 0.0025 \text{ GHz} = 5.0025 \text{ GHz or } 4.9975 \text{ GHz}$$

To determine the sign, one can:
1. Shift $\omega_d$ slightly and check if oscillation frequency increases or decreases
2. Use a second Ramsey with different wait time to track phase accumulation

## Practice Problems

### Level 1: Direct Application

1. Calculate the pulse duration for $X_\pi$, $X_{\pi/2}$, and $X_{\pi/4}$ gates if the Rabi frequency is 30 MHz.

2. A transmon has anharmonicity -220 MHz. What is the minimum recommended Gaussian pulse sigma to avoid significant leakage?

3. Design a $Y_{\pi/2}$ pulse: what drive phase $\phi$ is needed?

### Level 2: Intermediate

4. In a Rabi experiment, the population oscillates between 0.02 and 0.98 instead of 0 and 1. Identify the error(s) and explain how to correct them.

5. Derive the DRAG correction formula by considering the three-level Hamiltonian and applying adiabatic elimination of the $|2\rangle$ state.

6. Calculate the gate fidelity of an X gate that has 1% amplitude error (rotates by $1.01\pi$ instead of $\pi$).

### Level 3: Challenging

7. **Optimal control**: Set up the GRAPE optimization problem for a single-qubit NOT gate. Define the cost function and constraints. How would you handle the leakage to $|2\rangle$?

8. **Randomized benchmarking analysis**: RB data shows $P(m) = 0.5 + 0.48 \times (0.9985)^m$. Calculate:
   (a) The error per Clifford gate
   (b) The error per primitive gate (assuming each Clifford = 1.5 primitive gates on average)
   (c) $T_1$ limited fidelity if gates take 30 ns and $T_1 = 100$ $\mu$s

9. **Crosstalk**: Two qubits at 5.0 and 5.1 GHz are driven. When applying an X pulse to qubit 1, qubit 2 sees off-resonant driving. Calculate the Stark shift and unwanted rotation angle on qubit 2 for a 20 ns $\pi$ pulse.

## Computational Lab: Pulse Optimization and Gate Simulation

```python
"""
Day 900 Computational Lab: Single-Qubit Gates
Simulating Rabi oscillations, DRAG pulses, and gate calibration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# =============================================================================
# Part 1: Two-Level Rabi Dynamics
# =============================================================================

def rabi_evolution(Omega, delta, t_points, psi0=np.array([1, 0], dtype=complex)):
    """
    Simulate Rabi oscillations for a two-level system.

    Parameters:
    -----------
    Omega : float
        Rabi frequency (rad/s or GHz for convenience)
    delta : float
        Detuning (same units as Omega)
    t_points : array
        Time points
    psi0 : array
        Initial state

    Returns:
    --------
    P0, P1 : arrays
        Population in |0> and |1>
    """
    # Hamiltonian in rotating frame
    # H = delta/2 * sigma_z + Omega/2 * sigma_x
    H = np.array([[delta/2, Omega/2],
                  [Omega/2, -delta/2]])

    P0 = []
    P1 = []

    for t in t_points:
        U = expm(-1j * H * t)
        psi = U @ psi0
        P0.append(np.abs(psi[0])**2)
        P1.append(np.abs(psi[1])**2)

    return np.array(P0), np.array(P1)

# Rabi oscillation demonstration
print("=" * 60)
print("Rabi Oscillations")
print("=" * 60)

Omega = 2 * np.pi * 50  # 50 MHz Rabi frequency (in units of 2π MHz)
delta = 0  # On resonance

t_points = np.linspace(0, 100, 500)  # Time in ns
P0, P1 = rabi_evolution(Omega * 1e-3, delta, t_points)  # Convert to GHz

# Theoretical period
T_rabi = 2 * np.pi / (Omega * 1e-3)
print(f"Rabi frequency: {Omega/(2*np.pi)} MHz")
print(f"Rabi period: {T_rabi:.2f} ns")
print(f"π pulse time: {T_rabi/2:.2f} ns")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.plot(t_points, P0, 'b-', linewidth=2, label=r'$P_0$')
ax1.plot(t_points, P1, 'r-', linewidth=2, label=r'$P_1$')
ax1.set_xlabel('Time (ns)', fontsize=12)
ax1.set_ylabel('Population', fontsize=12)
ax1.set_title('Rabi Oscillations (On Resonance)', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Off-resonance Rabi
ax2 = axes[0, 1]
detunings = [0, 25, 50, 100]  # MHz
for det in detunings:
    delta_val = 2 * np.pi * det * 1e-3  # Convert to GHz
    _, P1_det = rabi_evolution(Omega * 1e-3, delta_val, t_points)
    effective_rabi = np.sqrt((Omega * 1e-3)**2 + delta_val**2)
    ax2.plot(t_points, P1_det, linewidth=2,
             label=rf'$\Delta/2\pi$ = {det} MHz')

ax2.set_xlabel('Time (ns)', fontsize=12)
ax2.set_ylabel(r'$P_1$', fontsize=12)
ax2.set_title('Detuned Rabi Oscillations', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# =============================================================================
# Part 2: DRAG Pulse Simulation
# =============================================================================

def gaussian_pulse(t, t0, sigma, amp):
    """Gaussian pulse envelope."""
    return amp * np.exp(-(t - t0)**2 / (2 * sigma**2))

def drag_pulse(t, t0, sigma, amp, beta):
    """DRAG pulse with in-phase and quadrature components."""
    gauss = gaussian_pulse(t, t0, sigma, amp)
    gauss_deriv = -amp * (t - t0) / sigma**2 * np.exp(-(t - t0)**2 / (2 * sigma**2))
    return gauss, beta * gauss_deriv

def three_level_evolution(pulse_func, alpha, t_span, dt=0.1):
    """
    Simulate three-level transmon dynamics under pulse.

    Parameters:
    -----------
    pulse_func : callable
        Function returning (Omega_x(t), Omega_y(t))
    alpha : float
        Anharmonicity in GHz (negative)
    t_span : tuple
        (t_start, t_end) in ns
    dt : float
        Time step in ns

    Returns:
    --------
    t, populations : arrays
    """
    # Three-level Hamiltonian in rotating frame at qubit frequency
    # |0>, |1>, |2> basis
    # Coupling: 01 transition driven, 12 transition also affected

    def hamiltonian(t, Omega_x, Omega_y):
        # Drive strength between levels (matrix elements)
        # sqrt(2) enhancement for 1-2 transition
        H = np.array([
            [0, Omega_x/2 - 1j*Omega_y/2, 0],
            [Omega_x/2 + 1j*Omega_y/2, 0, np.sqrt(2)*(Omega_x/2 - 1j*Omega_y/2)],
            [0, np.sqrt(2)*(Omega_x/2 + 1j*Omega_y/2), alpha]
        ], dtype=complex)
        return H

    def schrodinger(t, psi):
        Ox, Oy = pulse_func(t)
        H = hamiltonian(t, Ox, Oy)
        return -1j * H @ psi

    t_points = np.arange(t_span[0], t_span[1], dt)
    psi0 = np.array([1, 0, 0], dtype=complex)

    sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_points, method='RK45')

    populations = np.abs(sol.y)**2
    return sol.t, populations

# Parameters
alpha = -2 * np.pi * 0.25  # -250 MHz anharmonicity (in GHz)
sigma = 10  # ns
t0 = 40  # Center of pulse
gate_time = 80  # Total gate time in ns

# Find amplitude for π rotation
# Area of Gaussian = sqrt(2π) * sigma * amp
# For π pulse: Omega * t_eff = π, where t_eff ≈ sqrt(2π) * sigma
target_area = np.pi
amp_pi = target_area / (np.sqrt(2 * np.pi) * sigma)

print("\n" + "=" * 60)
print("DRAG Pulse Simulation")
print("=" * 60)
print(f"Anharmonicity: {alpha/(2*np.pi)*1000:.0f} MHz")
print(f"Pulse sigma: {sigma} ns")
print(f"π pulse amplitude: {amp_pi:.4f} GHz")

# Optimal DRAG beta
beta_optimal = -1 / alpha
print(f"Optimal DRAG beta: {beta_optimal:.3f} ns")

# Compare: no DRAG vs DRAG
def no_drag_pulse(t):
    Ox = gaussian_pulse(t, t0, sigma, amp_pi)
    return Ox, 0.0

def with_drag_pulse(t):
    Ox, Oy = drag_pulse(t, t0, sigma, amp_pi, beta_optimal)
    return Ox, Oy

t_nodrag, pop_nodrag = three_level_evolution(no_drag_pulse, alpha, (0, gate_time))
t_drag, pop_drag = three_level_evolution(with_drag_pulse, alpha, (0, gate_time))

# Plot comparison
ax3 = axes[1, 0]
ax3.plot(t_nodrag, pop_nodrag[0], 'b--', linewidth=2, label=r'$P_0$ (no DRAG)')
ax3.plot(t_nodrag, pop_nodrag[1], 'r--', linewidth=2, label=r'$P_1$ (no DRAG)')
ax3.plot(t_nodrag, pop_nodrag[2], 'g--', linewidth=2, label=r'$P_2$ (no DRAG)')
ax3.plot(t_drag, pop_drag[0], 'b-', linewidth=2, label=r'$P_0$ (DRAG)')
ax3.plot(t_drag, pop_drag[1], 'r-', linewidth=2, label=r'$P_1$ (DRAG)')
ax3.plot(t_drag, pop_drag[2], 'g-', linewidth=2, label=r'$P_2$ (DRAG)')
ax3.set_xlabel('Time (ns)', fontsize=12)
ax3.set_ylabel('Population', fontsize=12)
ax3.set_title('Three-Level Dynamics: No DRAG vs DRAG', fontsize=14)
ax3.legend(fontsize=9, ncol=2)
ax3.grid(True, alpha=0.3)

print(f"\nFinal populations (no DRAG):")
print(f"  P0 = {pop_nodrag[0, -1]:.6f}")
print(f"  P1 = {pop_nodrag[1, -1]:.6f}")
print(f"  P2 = {pop_nodrag[2, -1]:.6f} (leakage)")

print(f"\nFinal populations (with DRAG):")
print(f"  P0 = {pop_drag[0, -1]:.6f}")
print(f"  P1 = {pop_drag[1, -1]:.6f}")
print(f"  P2 = {pop_drag[2, -1]:.6f} (leakage)")

# =============================================================================
# Part 3: DRAG Beta Optimization
# =============================================================================

def evaluate_leakage(beta):
    """Evaluate leakage for given DRAG parameter."""
    def pulse(t):
        Ox, Oy = drag_pulse(t, t0, sigma, amp_pi, beta)
        return Ox, Oy
    _, pop = three_level_evolution(pulse, alpha, (0, gate_time))
    return pop[2, -1]  # Final P2

beta_range = np.linspace(0, 0.8, 30)
leakages = [evaluate_leakage(b) for b in beta_range]

ax4 = axes[1, 1]
ax4.semilogy(beta_range, leakages, 'b-', linewidth=2)
ax4.axvline(beta_optimal, color='r', linestyle='--', label=f'Theory: β = {beta_optimal:.3f}')
ax4.set_xlabel('DRAG parameter β (ns)', fontsize=12)
ax4.set_ylabel('Leakage to |2⟩', fontsize=12)
ax4.set_title('DRAG Parameter Optimization', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('single_qubit_gates.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 4: Ramsey Experiment Simulation
# =============================================================================

print("\n" + "=" * 60)
print("Ramsey Experiment Simulation")
print("=" * 60)

def ramsey_sequence(delta, T2star, tau_values):
    """
    Simulate Ramsey experiment.

    Parameters:
    -----------
    delta : float
        Detuning in MHz
    T2star : float
        Dephasing time in ns
    tau_values : array
        Wait times in ns

    Returns:
    --------
    P1 : array
        Excited state probability after sequence
    """
    P1 = []
    for tau in tau_values:
        # π/2 pulse, wait tau, π/2 pulse
        # Perfect pulses: phase accumulation = delta * tau
        phase = 2 * np.pi * delta * 1e-3 * tau  # Convert MHz to GHz
        # With dephasing
        visibility = np.exp(-tau / T2star)
        P1.append(0.5 * (1 + visibility * np.cos(phase)))
    return np.array(P1)

delta_actual = 3.5  # MHz detuning
T2star = 5000  # ns
tau_vals = np.linspace(0, 4000, 200)

P1_ramsey = ramsey_sequence(delta_actual, T2star, tau_vals)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes2[0]
ax1.plot(tau_vals / 1000, P1_ramsey, 'b-', linewidth=2)
ax1.set_xlabel('Wait time τ (μs)', fontsize=12)
ax1.set_ylabel(r'$P_1$', fontsize=12)
ax1.set_title(f'Ramsey Oscillations (Δ = {delta_actual} MHz)', fontsize=14)
ax1.grid(True, alpha=0.3)

print(f"Detuning: {delta_actual} MHz")
print(f"T2*: {T2star/1000:.1f} μs")
print(f"Ramsey oscillation period: {1/delta_actual * 1000:.1f} ns")

# =============================================================================
# Part 5: AllXY Calibration
# =============================================================================

def apply_gate(state, gate_type, error_amp=0, error_det=0, error_phase=0):
    """
    Apply a gate with optional errors.

    Parameters:
    -----------
    state : array
        2-element state vector
    gate_type : str
        'I', 'X', 'Y', 'Xp' (X_π/2), 'Yp' (Y_π/2), 'Xm' (X_-π/2), 'Ym' (Y_-π/2)
    error_amp : float
        Fractional amplitude error
    error_det : float
        Detuning error (relative to Rabi frequency)
    error_phase : float
        Phase error in radians
    """
    if gate_type == 'I':
        return state

    # Base rotation angles
    angles = {'X': np.pi, 'Y': np.pi,
              'Xp': np.pi/2, 'Yp': np.pi/2,
              'Xm': -np.pi/2, 'Ym': -np.pi/2}

    theta = angles[gate_type] * (1 + error_amp)

    # Rotation axis
    if 'X' in gate_type:
        axis_angle = 0 + error_phase
    else:  # Y gate
        axis_angle = np.pi/2 + error_phase

    # Include detuning error
    effective_theta = np.sqrt(theta**2 + error_det**2)
    if effective_theta > 0:
        tilt = np.arctan2(error_det, theta)
    else:
        tilt = 0

    # Rotation matrix around axis in x-y plane tilted toward z
    nx = np.cos(axis_angle) * np.cos(tilt)
    ny = np.sin(axis_angle) * np.cos(tilt)
    nz = np.sin(tilt)

    c = np.cos(effective_theta / 2)
    s = np.sin(effective_theta / 2)

    U = np.array([
        [c - 1j*nz*s, (-1j*nx - ny)*s],
        [(-1j*nx + ny)*s, c + 1j*nz*s]
    ], dtype=complex)

    return U @ state

def allxy_sequence(error_amp=0, error_det=0, error_phase=0):
    """
    Run AllXY protocol.

    Returns expected outcomes for 21 gate pairs.
    """
    sequences = [
        ('I', 'I'),    # 0: should give 0
        ('X', 'X'),    # 1: should give 0
        ('Y', 'Y'),    # 2: should give 0
        ('X', 'Y'),    # 3: should give 0
        ('Y', 'X'),    # 4: should give 0
        ('Xp', 'I'),   # 5: should give 0.5
        ('Yp', 'I'),   # 6: should give 0.5
        ('Xp', 'Yp'),  # 7: should give 0.5
        ('Yp', 'Xp'),  # 8: should give 0.5
        ('Xp', 'Y'),   # 9: should give 0.5
        ('Yp', 'X'),   # 10: should give 0.5
        ('X', 'Yp'),   # 11: should give 0.5
        ('Y', 'Xp'),   # 12: should give 0.5
        ('Xp', 'X'),   # 13: should give 0.5
        ('X', 'Xp'),   # 14: should give 0.5
        ('Yp', 'Y'),   # 15: should give 0.5
        ('Y', 'Yp'),   # 16: should give 0.5
        ('X', 'I'),    # 17: should give 1
        ('Y', 'I'),    # 18: should give 1
        ('Xp', 'Xp'),  # 19: should give 1
        ('Yp', 'Yp'),  # 20: should give 1
    ]

    expected = [0, 0, 0, 0, 0,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                1, 1, 1, 1]

    results = []
    psi0 = np.array([1, 0], dtype=complex)

    for g1, g2 in sequences:
        psi = apply_gate(psi0, g1, error_amp, error_det, error_phase)
        psi = apply_gate(psi, g2, error_amp, error_det, error_phase)
        P1 = np.abs(psi[1])**2
        results.append(P1)

    return np.array(results), np.array(expected)

# AllXY with different errors
ax2 = axes2[1]

results_ideal, expected = allxy_sequence(0, 0, 0)
results_amp_err, _ = allxy_sequence(0.05, 0, 0)
results_det_err, _ = allxy_sequence(0, 0.1, 0)

x = np.arange(21)
ax2.plot(x, expected, 'ko-', linewidth=2, markersize=8, label='Expected')
ax2.plot(x, results_amp_err, 'rs-', linewidth=1.5, markersize=6, label='5% amp error')
ax2.plot(x, results_det_err, 'b^-', linewidth=1.5, markersize=6, label='10% det error')
ax2.set_xlabel('AllXY sequence number', fontsize=12)
ax2.set_ylabel(r'$P_1$', fontsize=12)
ax2.set_title('AllXY Calibration Protocol', fontsize=14)
ax2.legend(fontsize=10)
ax2.set_xticks(x[::2])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gate_calibration.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 6: Gate Fidelity Calculation
# =============================================================================

print("\n" + "=" * 60)
print("Gate Fidelity Analysis")
print("=" * 60)

def gate_fidelity(U_ideal, U_actual):
    """Calculate average gate fidelity for single qubit."""
    trace = np.trace(U_ideal.conj().T @ U_actual)
    return (1 + np.abs(trace)**2 / 4) / 2

# Ideal X gate
X_ideal = np.array([[0, 1], [1, 0]], dtype=complex)

# X gate with amplitude error
def X_with_amp_error(eps):
    theta = np.pi * (1 + eps)
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

# Calculate fidelity vs amplitude error
eps_values = np.linspace(-0.1, 0.1, 100)
fidelities = [gate_fidelity(X_ideal, X_with_amp_error(eps)) for eps in eps_values]

fig3, ax = plt.subplots(figsize=(8, 5))
ax.plot(eps_values * 100, fidelities, 'b-', linewidth=2)
ax.axhline(0.999, color='g', linestyle='--', label='99.9% threshold')
ax.axhline(0.99, color='r', linestyle='--', label='99% threshold')
ax.set_xlabel('Amplitude error (%)', fontsize=12)
ax.set_ylabel('Gate fidelity', fontsize=12)
ax.set_title('X Gate Fidelity vs Amplitude Error', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.98, 1.001])

plt.tight_layout()
plt.savefig('gate_fidelity.png', dpi=150, bbox_inches='tight')
plt.show()

# Find threshold errors
for target_fid in [0.999, 0.99, 0.9]:
    for eps in eps_values:
        if gate_fidelity(X_ideal, X_with_amp_error(eps)) < target_fid:
            print(f"Fidelity drops below {target_fid*100:.1f}% at {abs(eps)*100:.2f}% amplitude error")
            break

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Rotating frame Hamiltonian | $\hat{H} = \frac{\hbar\Delta}{2}\hat{\sigma}_z + \frac{\hbar\Omega}{2}(\cos\phi\,\hat{\sigma}_x + \sin\phi\,\hat{\sigma}_y)$ |
| Rabi oscillation | $P_1(t) = \sin^2(\Omega t/2)$ |
| $\pi$ pulse time | $t_\pi = \pi/\Omega$ |
| DRAG correction | $\Omega_y(t) = -\dot{\Omega}(t)/\alpha$ |
| DRAG parameter | $\beta = -1/\alpha$ |
| Gate fidelity | $F = (1 + |\text{Tr}(U^\dagger_{ideal}U_{actual})|^2/4)/2$ |

### Main Takeaways

1. **Resonant driving** causes Rabi oscillations; pulse duration controls rotation angle

2. **DRAG pulses** suppress leakage to $|2\rangle$ by adding derivative quadrature component

3. **Virtual Z gates** are free—implemented by frame tracking with zero duration

4. **Calibration protocols** (Rabi, Ramsey, AllXY) systematically identify and correct errors

5. **Pulse bandwidth** must be less than anharmonicity: $1/\sigma \ll |\alpha|$

6. **State-of-the-art**: >99.99% single-qubit gate fidelity in 20-50 ns

## Daily Checklist

- [ ] I can derive the rotating-frame Hamiltonian
- [ ] I understand Rabi oscillations and their relationship to gate rotations
- [ ] I can design DRAG pulses and explain why they suppress leakage
- [ ] I understand the Ramsey experiment and AllXY protocol
- [ ] I can explain virtual-Z gates and their advantages
- [ ] I have run the computational lab and can interpret the results
- [ ] I can calculate and interpret gate fidelities

## Preview: Day 901

Tomorrow we explore **two-qubit gates** in superconducting systems:

- Cross-resonance gates (CR) for fixed-frequency qubits
- Controlled-Z (CZ) gates via flux tuning
- iSWAP and parametric gates
- ZZ coupling and crosstalk mitigation
- Fidelity limits and scaling challenges

---

*"The single-qubit gate is where physics meets engineering: we must control quantum systems with classical signals at nanosecond precision."*
