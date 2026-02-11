# Day 908: Two-Qubit Entangling Gates

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | MS gate theory, geometric phase gates, Cirac-Zoller |
| Afternoon | 2 hours | Problem solving: gate parameters and optimization |
| Evening | 2 hours | Computational lab: MS gate phase space simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive the Molmer-Sorensen gate** Hamiltonian and dynamics
2. **Explain geometric phase accumulation** in entangling gates
3. **Analyze the Cirac-Zoller gate** mechanism
4. **Calculate gate times and fidelities** for two-qubit operations
5. **Design optimal gate parameters** for maximum fidelity
6. **Compare different entangling gate schemes** and their trade-offs

## Core Content

### 1. Introduction to Two-Qubit Gates

Two-qubit entangling gates create correlations between qubits that cannot be produced by single-qubit operations alone. In trapped ions, the shared motional modes serve as a quantum bus to mediate these interactions.

**Universal gate set:** Single-qubit rotations + one entangling gate = universal quantum computing

**Key entangling gates:**
- Molmer-Sorensen (MS) gate: Most widely used
- Geometric phase gate: Conceptually elegant
- Cirac-Zoller gate: Historically first proposal
- Light-shift gate: Alternative approach

### 2. Ion-Laser Coupling with Motion

Recall the ion-laser interaction in the Lamb-Dicke regime:

$$\hat{H}_{int} = \frac{\hbar\Omega}{2}\hat{\sigma}_+\left(1 + i\eta(\hat{a}e^{-i\omega t} + \hat{a}^\dagger e^{i\omega t})\right)e^{-i\omega_L t} + h.c.$$

Tuning to different frequencies selects different transitions:
- **Carrier** ($\omega_L = \omega_0$): $|g,n\rangle \leftrightarrow |e,n\rangle$
- **Red sideband** ($\omega_L = \omega_0 - \omega$): $|g,n\rangle \leftrightarrow |e,n-1\rangle$
- **Blue sideband** ($\omega_L = \omega_0 + \omega$): $|g,n\rangle \leftrightarrow |e,n+1\rangle$

### 3. The Molmer-Sorensen Gate

The MS gate simultaneously drives both red and blue sidebands with a bichromatic field.

#### Bichromatic Drive

Two laser frequency components:
- $\omega_1 = \omega_0 + \omega + \delta$ (blue-detuned)
- $\omega_2 = \omega_0 - \omega - \delta$ (red-detuned)

where $\delta \ll \omega$ is a small detuning.

#### Hamiltonian

In the interaction picture, for two ions:

$$\boxed{\hat{H}_{MS} = \hbar\Omega\sum_{j=1,2}\hat{\sigma}_x^{(j)}\left(\hat{a}e^{-i\delta t} + \hat{a}^\dagger e^{i\delta t}\right)}$$

This can be written as:

$$\hat{H}_{MS} = \hbar\Omega\hat{S}_x(\hat{a}e^{-i\delta t} + \hat{a}^\dagger e^{i\delta t})$$

where $\hat{S}_x = \hat{\sigma}_x^{(1)} + \hat{\sigma}_x^{(2)}$ is the collective spin operator.

#### Eigenstate Structure

The eigenstates of $\hat{S}_x$ are:
- $|+x,+x\rangle$ with eigenvalue +2
- $|+x,-x\rangle, |-x,+x\rangle$ with eigenvalue 0
- $|-x,-x\rangle$ with eigenvalue -2

#### Phase Space Dynamics

The motion traces circles in phase space:

$$\alpha(t) = \frac{\Omega}{\delta}(e^{-i\delta t} - 1)$$

This is a displacement that depends on the spin state!

For $\hat{S}_x = \pm 2$: Displacement amplitude $\propto \pm 2$
For $\hat{S}_x = 0$: No displacement

#### Gate Unitary

After time $t = 2\pi/\delta$ (one complete loop):

$$\boxed{\hat{U}_{MS} = \exp\left(-i\frac{\Omega^2}{\delta}t\hat{S}_x^2\right) = \exp\left(-i\frac{2\pi\Omega^2}{\delta^2}\hat{S}_x^2\right)}$$

For a maximally entangling gate (equivalent to $\sqrt{XX}$):

$$\frac{\Omega^2}{\delta^2} \cdot 2\pi = \frac{\pi}{4}$$

$$\boxed{\delta = 2\Omega\sqrt{2\pi}}$$

#### Gate Time

$$\boxed{t_{gate} = \frac{2\pi}{\delta} = \frac{\sqrt{2\pi}}{\Omega} \approx \frac{2.5}{\Omega}}$$

For $\Omega/2\pi = 100$ kHz: $t_{gate} \approx 25$ μs

#### Resulting Entanglement

Starting from $|00\rangle$:

$$|00\rangle \xrightarrow{MS} \frac{1}{\sqrt{2}}(|00\rangle + i|11\rangle)$$

This is a maximally entangled Bell state!

### 4. Geometric Phase Gate

The MS gate is an example of a geometric phase gate. The key insight is that the acquired phase is proportional to the enclosed phase space area.

#### Phase Space Area

$$\phi_{geom} = \text{Area enclosed} = \pi|\alpha_{max}|^2 = \pi\left(\frac{2\Omega}{\delta}\right)^2$$

#### Spin-Dependent Displacement

| Spin state | $S_x$ | Displacement | Phase |
|------------|-------|--------------|-------|
| $|+x,+x\rangle$ | +2 | $2\alpha$ | $4\phi_0$ |
| $|+x,-x\rangle$ | 0 | 0 | 0 |
| $|-x,+x\rangle$ | 0 | 0 | 0 |
| $|-x,-x\rangle$ | -2 | $-2\alpha$ | $4\phi_0$ |

The relative phase between states with $S_x = \pm 2$ and $S_x = 0$ creates entanglement.

#### Disentanglement from Motion

**Critical condition:** At $t = 2\pi n/\delta$ (integer loops), motion returns to origin:

$$\alpha(t = 2\pi/\delta) = 0$$

The motional state is disentangled from the spin states, leaving only the geometric phase.

### 5. The Cirac-Zoller Gate

The original proposal (1995) uses sequential sideband pulses.

#### Protocol

1. **Initialize:** Prepare motional ground state $|00\rangle|n=0\rangle$
2. **Red sideband π-pulse on ion 1:** $|g_1\rangle|0\rangle \rightarrow |e_1\rangle|0\rangle$ (no change if $n=0$)
   - Actually: $|g_1,0\rangle \rightarrow |g_1,0\rangle$ (off-resonant)
3. **Apply conditional logic...**

A more practical version:

1. RSB π-pulse on ion 1: Creates entanglement with motion
2. Carrier 2π-pulse on ion 2: Conditional phase based on ion 1's state
3. RSB π-pulse on ion 1: Undoes motion entanglement

#### Cirac-Zoller Hamiltonian

$$\hat{H}_{CZ} = \frac{\hbar\Omega_{RSB}}{2}(|e\rangle\langle g|\hat{a} + |g\rangle\langle e|\hat{a}^\dagger)$$

This couples $|g,1\rangle \leftrightarrow |e,0\rangle$.

#### Advantages and Disadvantages

**Advantages:**
- Conceptually simple
- Works with single addressing beam

**Disadvantages:**
- Requires ground state preparation
- Slower than MS gate (sequential pulses)
- More sensitive to motional heating

### 6. Light-Shift Gate

An alternative using off-resonant laser-induced AC Stark shifts.

#### Mechanism

Apply laser tuned between carrier and blue sideband:

$$\hat{H}_{LS} = \hbar\chi\hat{\sigma}_z(\hat{a}^\dagger\hat{a} + 1/2)$$

This creates a qubit-state-dependent frequency shift of the motional mode.

#### Two-Ion Gate

With two ions addressed:

$$\hat{H} = \hbar\chi(\hat{\sigma}_z^{(1)} + \hat{\sigma}_z^{(2)})(\hat{a}^\dagger\hat{a})$$

Combined with a motional displacement, this generates $ZZ$ coupling.

### 7. Gate Optimization

#### Error Sources

1. **Motional decoherence:** Heating during gate
2. **Residual spin-motion entanglement:** Incomplete loops
3. **Off-resonant carrier:** Causes single-qubit rotations
4. **Spectator modes:** Other modes acquire unwanted phases
5. **Laser noise:** Intensity and phase fluctuations

#### Multi-Mode Considerations

For $N$ ions, there are $N$ axial modes. The MS gate must close loops in ALL modes:

$$\alpha_m(t_{gate}) = 0 \quad \forall m$$

**Solution:** Pulse shaping or multi-tone drives

#### Amplitude Modulation

Use time-dependent Rabi frequency $\Omega(t)$ to satisfy closure conditions:

$$\int_0^{t_{gate}} \Omega(t)e^{i\omega_m t}dt = 0$$

for all spectator modes $m$.

#### Optimal Gate Time

Balance between:
- Shorter gate → less heating
- Longer gate → smaller $\delta$ → better mode resolution

Typical: $t_{gate} = 50-200$ μs for two-ion gates

### 8. Fidelity Considerations

#### Theoretical Fidelity

The gate fidelity is limited by:

$$1 - F \approx \frac{\dot{\bar{n}}t_{gate}}{|\delta|/\Omega} + \left(\frac{\Omega}{\omega}\right)^2 + \left(\frac{\delta\Omega}{\Omega}\right)^2$$

**State-of-the-art:** $F > 99.9\%$ demonstrated in multiple labs

#### Scaling with Ion Number

For $N$ ions:
- Mode spectrum becomes denser
- Requires more careful pulse design
- Gate fidelity typically decreases with $N$

## Quantum Computing Applications

### Native Gate Set

Trapped ion quantum computers typically implement:
- Single-qubit rotations: $R_x(\theta), R_y(\theta), R_z(\theta)$
- MS gate: $XX(\theta)$ or $\sqrt{XX}$

All other gates compiled from these.

### CNOT Decomposition

$$CNOT = (I \otimes H) \cdot MS(\pi/2) \cdot (I \otimes H)$$

or with explicit phases:

$$CNOT = R_y^{(2)}(\pi/2) \cdot MS(\pi/2) \cdot R_x^{(1)}(-\pi/2) \cdot R_x^{(2)}(-\pi/2) \cdot R_y^{(2)}(-\pi/2)$$

### All-to-All Connectivity

Unlike superconducting qubits, trapped ions can perform MS gates between ANY pair of ions by addressing them with laser beams.

**Implications:**
- No SWAP overhead
- Efficient algorithm implementation
- But addressing requires optical control

## Worked Examples

### Example 1: MS Gate Parameters

**Problem:** Design an MS gate for two $^{171}$Yb$^+$ ions with trap frequency $\omega/2\pi = 1$ MHz and Rabi frequency $\Omega/2\pi = 50$ kHz. Find the detuning and gate time.

**Solution:**

For a maximally entangling gate, we need:

$$\frac{\Omega^2}{\delta^2} \cdot 2\pi = \frac{\pi}{4}$$

Solving for $\delta$:
$$\delta = \Omega\sqrt{8\pi} = 2\pi \times 50 \times 10^3 \times \sqrt{8\pi}$$

$$\delta = 2\pi \times 50 \times 10^3 \times 5.01 = 2\pi \times 250.6 \text{ kHz}$$

Wait, this seems too large compared to the mode frequency. Let me recalculate.

Actually, for the gate time $t = 2\pi/\delta$:

$$\frac{2\pi\Omega^2}{\delta^2} = \frac{\pi}{4}$$

$$\delta^2 = 8\Omega^2$$

$$\delta = 2\sqrt{2}\Omega = 2\sqrt{2} \times 2\pi \times 50 \times 10^3$$

$$\boxed{\delta/2\pi = 141 \text{ kHz}}$$

Gate time:
$$t_{gate} = \frac{2\pi}{\delta} = \frac{1}{141 \times 10^3}$$

$$\boxed{t_{gate} = 7.1 \text{ μs}}$$

We should verify $\delta \ll \omega$: 141 kHz << 1 MHz ✓

### Example 2: Phase Space Area

**Problem:** Calculate the phase space area enclosed during the MS gate from Example 1.

**Solution:**

Maximum displacement:
$$\alpha_{max} = \frac{\Omega}{\delta/2} = \frac{2\Omega}{\delta}$$

With $\Omega = 2\pi \times 50$ kHz and $\delta = 2\pi \times 141$ kHz:

$$\alpha_{max} = \frac{2 \times 50}{141} = 0.71$$

Phase space area (for the $S_x = +2$ state):
$$\text{Area} = \pi |2\alpha_{max}|^2 = \pi \times (2 \times 0.71)^2 = \pi \times 2.02$$

$$\boxed{\text{Area} = 6.33}$$

The geometric phase is:
$$\phi_{geom} = \text{Area} = 6.33 \text{ rad}$$

But for the $S_x = \pm 2$ states, the effective phase is $4\phi_0$ where $\phi_0 = \pi\alpha_{max}^2$.

This gives the entangling phase $\chi = 4\phi_0 = 4\pi(0.71)^2 = 6.33$ rad $\approx 2\pi$ + small correction.

### Example 3: Heating-Limited Fidelity

**Problem:** If the motional heating rate is $\dot{\bar{n}} = 100$ quanta/s, what is the heating-limited infidelity for the gate in Example 1?

**Solution:**

During the gate, the average phonon number increases by:
$$\Delta\bar{n} = \dot{\bar{n}} \times t_{gate} = 100 \times 7.1 \times 10^{-6} = 7.1 \times 10^{-4}$$

The infidelity from heating is approximately:
$$1 - F_{heating} \approx \Delta\bar{n} \times f(\delta/\omega)$$

For well-separated modes, $f \approx 1$:

$$\boxed{1 - F_{heating} \approx 7 \times 10^{-4} = 0.07\%}$$

This is compatible with 99.9% fidelity goals.

## Practice Problems

### Level 1: Direct Application

1. Calculate the gate time for an MS gate with $\Omega/2\pi = 100$ kHz.

2. What is the Lamb-Dicke parameter for $^{40}$Ca$^+$ at trap frequency 1.5 MHz using 729 nm light?

3. If the heating rate is 500 quanta/s, what is the maximum gate time to keep heating error below 0.1%?

### Level 2: Intermediate

4. Design a pulse sequence to implement CNOT using MS gates and single-qubit rotations. Verify your answer by matrix multiplication.

5. For a three-ion chain, calculate the mode frequencies (COM, tilt, zigzag) and design an MS gate that closes loops in all modes.

6. Compare the MS and Cirac-Zoller gates in terms of: (a) gate time, (b) heating sensitivity, (c) required ground state cooling.

### Level 3: Challenging

7. Derive the geometric phase formula $\phi = \pi|\alpha|^2$ from the evolution of a displaced coherent state.

8. Analyze the effect of off-resonant carrier transitions on MS gate fidelity. At what detuning-to-Rabi-frequency ratio does this error become limiting?

9. Design an amplitude-modulated pulse that implements a high-fidelity MS gate while suppressing errors from spectator modes.

## Computational Lab: MS Gate Phase Space Simulation

```python
"""
Day 908 Computational Lab: Two-Qubit Gate Simulation
Simulating Molmer-Sorensen gate phase space dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


class MSGate:
    """Molmer-Sorensen gate simulator"""

    def __init__(self, omega_trap, eta, omega_rabi, detuning):
        """
        Parameters:
        -----------
        omega_trap : float - Trap frequency (rad/s)
        eta : float - Lamb-Dicke parameter
        omega_rabi : float - Rabi frequency (rad/s)
        detuning : float - Detuning from sideband (rad/s)
        """
        self.omega = omega_trap
        self.eta = eta
        self.Omega = omega_rabi
        self.delta = detuning

        # Effective coupling
        self.g = eta * omega_rabi  # Sideband Rabi frequency

    def gate_time(self):
        """Calculate gate time for maximum entanglement"""
        return 2 * np.pi / self.delta

    def displacement_trajectory(self, Sx_eigenvalue, n_points=1000):
        """
        Calculate phase space trajectory for given Sx eigenvalue

        Parameters:
        -----------
        Sx_eigenvalue : float - Eigenvalue of collective Sx operator (+2, 0, or -2)

        Returns:
        --------
        t : array - Time points
        alpha_real : array - Real part of displacement
        alpha_imag : array - Imaginary part of displacement
        """
        t = np.linspace(0, self.gate_time(), n_points)

        # Displacement amplitude depends on spin state
        # α(t) = (g/δ) * Sx * (e^{-iδt} - 1)
        amplitude = (self.g / self.delta) * Sx_eigenvalue

        alpha = amplitude * (np.exp(-1j * self.delta * t) - 1)

        return t, np.real(alpha), np.imag(alpha)

    def geometric_phase(self, Sx_eigenvalue):
        """Calculate geometric phase for given Sx eigenvalue"""
        amplitude = np.abs((self.g / self.delta) * Sx_eigenvalue)
        return np.pi * amplitude**2

    def entanglement_phase(self):
        """Calculate the entanglement phase χ"""
        # Phase difference between Sx = ±2 and Sx = 0 states
        phi_2 = self.geometric_phase(2)
        phi_0 = self.geometric_phase(0)
        return phi_2 - phi_0


def create_ms_hamiltonian(Omega, delta, n_max=10):
    """
    Create full Hamiltonian for MS gate simulation

    Parameters:
    -----------
    Omega : float - Effective coupling (rad/s)
    delta : float - Detuning (rad/s)
    n_max : int - Maximum phonon number

    Returns:
    --------
    H_func : function - Returns Hamiltonian at time t
    dim : int - Hilbert space dimension
    """
    # Two-qubit basis: |00⟩, |01⟩, |10⟩, |11⟩
    # Combined with phonon states |n⟩

    dim_spin = 4
    dim_phonon = n_max + 1
    dim_total = dim_spin * dim_phonon

    # Spin operators (two qubits)
    Sx1 = np.kron(sigma_x, I)
    Sx2 = np.kron(I, sigma_x)
    Sx_total = Sx1 + Sx2

    # Phonon operators
    a = np.diag(np.sqrt(np.arange(1, dim_phonon)), 1)
    a_dag = a.T

    # Expand to full space
    Sx_full = np.kron(Sx_total, np.eye(dim_phonon))
    a_full = np.kron(np.eye(dim_spin), a)
    a_dag_full = np.kron(np.eye(dim_spin), a_dag)

    def H_func(t):
        # H = Omega * Sx * (a * e^{-iδt} + a† * e^{iδt})
        H = Omega * Sx_full @ (a_full * np.exp(-1j * delta * t) +
                               a_dag_full * np.exp(1j * delta * t))
        return H

    return H_func, dim_total


def simulate_ms_gate(Omega, delta, t_final, n_max=10, n_steps=500):
    """
    Full quantum simulation of MS gate

    Returns state evolution and observables
    """
    H_func, dim = create_ms_hamiltonian(Omega, delta, n_max)

    dim_spin = 4
    dim_phonon = n_max + 1

    # Initial state: |00⟩|n=0⟩
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0

    # Time evolution
    t = np.linspace(0, t_final, n_steps)
    dt = t[1] - t[0]

    states = np.zeros((n_steps, dim), dtype=complex)
    states[0] = psi0

    # Observables
    phonon_number = np.zeros(n_steps)
    concurrence = np.zeros(n_steps)

    # Number operator
    n_op = np.diag(np.arange(dim_phonon))
    N_full = np.kron(np.eye(dim_spin), n_op)

    psi = psi0.copy()

    for i in range(1, n_steps):
        # Simple Euler integration (for demonstration)
        H = H_func(t[i-1])
        U = expm(-1j * H * dt)
        psi = U @ psi
        psi = psi / np.linalg.norm(psi)  # Renormalize

        states[i] = psi
        phonon_number[i] = np.real(psi.conj() @ N_full @ psi)

        # Calculate reduced spin state and concurrence
        rho_full = np.outer(psi, psi.conj())
        rho_spin = np.zeros((4, 4), dtype=complex)
        for n in range(dim_phonon):
            for m in range(dim_phonon):
                if n == m:  # Only diagonal phonon terms for reduced state
                    block = rho_full[n::dim_phonon, m::dim_phonon]
                    for s1 in range(dim_spin):
                        for s2 in range(dim_spin):
                            if s1 * dim_phonon + n < dim and s2 * dim_phonon + m < dim:
                                rho_spin[s1, s2] += rho_full[s1*dim_phonon+n, s2*dim_phonon+n]

        # Simplified concurrence estimate (for |00⟩ + |11⟩ type states)
        concurrence[i] = 2 * np.abs(rho_spin[0, 3])

    return t, states, phonon_number, concurrence


def plot_phase_space():
    """Plot phase space trajectories during MS gate"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Gate parameters
    omega = 2 * np.pi * 1e6  # 1 MHz trap
    eta = 0.1
    Omega = 2 * np.pi * 50e3  # 50 kHz Rabi
    delta = 2 * np.pi * 20e3  # 20 kHz detuning

    ms = MSGate(omega, eta, Omega, delta)

    # Phase space trajectories
    ax1 = axes[0, 0]

    for Sx, color, label in [(2, 'red', '$S_x = +2$'),
                             (0, 'gray', '$S_x = 0$'),
                             (-2, 'blue', '$S_x = -2$')]:
        t, alpha_r, alpha_i = ms.displacement_trajectory(Sx)
        ax1.plot(alpha_r, alpha_i, color=color, linewidth=2, label=label)
        ax1.scatter([alpha_r[0]], [alpha_i[0]], color=color, s=100, marker='o')
        ax1.scatter([alpha_r[-1]], [alpha_i[-1]], color=color, s=100, marker='s')

    ax1.set_xlabel('Re(α)', fontsize=12)
    ax1.set_ylabel('Im(α)', fontsize=12)
    ax1.set_title('Phase Space Trajectories', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Displacement magnitude vs time
    ax2 = axes[0, 1]

    for Sx, color, label in [(2, 'red', '$S_x = +2$'),
                             (0, 'gray', '$S_x = 0$'),
                             (-2, 'blue', '$S_x = -2$')]:
        t, alpha_r, alpha_i = ms.displacement_trajectory(Sx)
        alpha_mag = np.sqrt(alpha_r**2 + alpha_i**2)
        ax2.plot(t * 1e6, alpha_mag, color=color, linewidth=2, label=label)

    ax2.set_xlabel('Time (μs)', fontsize=12)
    ax2.set_ylabel('|α|', fontsize=12)
    ax2.set_title('Displacement Magnitude vs Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Geometric phase accumulation
    ax3 = axes[1, 0]

    t, alpha_r, alpha_i = ms.displacement_trajectory(2)
    # Phase is area enclosed = integral of x dy
    phase = np.zeros_like(t)
    for i in range(1, len(t)):
        # Trapezoidal integration for enclosed area
        dphase = 0.5 * (alpha_r[i] + alpha_r[i-1]) * (alpha_i[i] - alpha_i[i-1])
        dphase -= 0.5 * (alpha_i[i] + alpha_i[i-1]) * (alpha_r[i] - alpha_r[i-1])
        phase[i] = phase[i-1] + dphase

    ax3.plot(t * 1e6, phase, 'b-', linewidth=2)
    ax3.axhline(y=np.pi * (2 * eta * Omega / delta)**2, color='red', linestyle='--',
                label='Expected $\pi|α_{max}|^2$')
    ax3.set_xlabel('Time (μs)', fontsize=12)
    ax3.set_ylabel('Accumulated phase (rad)', fontsize=12)
    ax3.set_title('Geometric Phase Accumulation', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Gate fidelity vs detuning
    ax4 = axes[1, 1]

    delta_range = np.linspace(0.1, 2, 50) * eta * Omega

    entanglement = []
    gate_times = []

    for d in delta_range:
        ms_temp = MSGate(omega, eta, Omega, d)
        chi = ms_temp.entanglement_phase()
        entanglement.append(chi)
        gate_times.append(ms_temp.gate_time() * 1e6)

    ax4.plot(delta_range / (eta * Omega), entanglement, 'b-', linewidth=2)
    ax4.axhline(y=np.pi/4, color='red', linestyle='--', label='Maximally entangling (π/4)')
    ax4.axhline(y=np.pi/2, color='orange', linestyle=':', label='Full entangling (π/2)')

    ax4_twin = ax4.twinx()
    ax4_twin.plot(delta_range / (eta * Omega), gate_times, 'g--', linewidth=2)
    ax4_twin.set_ylabel('Gate time (μs)', fontsize=12, color='green')

    ax4.set_xlabel('Detuning δ/(ηΩ)', fontsize=12)
    ax4.set_ylabel('Entanglement phase χ (rad)', fontsize=12, color='blue')
    ax4.set_title('Entanglement vs Detuning', fontsize=14)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ms_phase_space.png', dpi=150)
    plt.show()


def plot_gate_evolution():
    """Plot quantum state evolution during MS gate"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Parameters for faster simulation
    Omega = 2 * np.pi * 100e3  # Effective coupling
    delta = 2 * np.pi * 50e3   # Detuning
    t_gate = 2 * np.pi / delta

    print(f"Gate time: {t_gate * 1e6:.2f} μs")

    # Simulate
    t, states, n_phonon, concurrence = simulate_ms_gate(
        Omega, delta, t_gate, n_max=5, n_steps=200
    )

    # Population evolution
    ax1 = axes[0, 0]

    # Extract spin state populations (traced over phonon)
    dim_phonon = 6
    pop_00 = np.sum(np.abs(states[:, 0::4])**2, axis=1)
    pop_01 = np.sum(np.abs(states[:, 1::4])**2, axis=1)
    pop_10 = np.sum(np.abs(states[:, 2::4])**2, axis=1)
    pop_11 = np.sum(np.abs(states[:, 3::4])**2, axis=1)

    ax1.plot(t * 1e6, pop_00, 'b-', label='|00⟩', linewidth=2)
    ax1.plot(t * 1e6, pop_01, 'g-', label='|01⟩', linewidth=2)
    ax1.plot(t * 1e6, pop_10, 'orange', label='|10⟩', linewidth=2)
    ax1.plot(t * 1e6, pop_11, 'r-', label='|11⟩', linewidth=2)

    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Population', fontsize=12)
    ax1.set_title('Spin State Populations', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mean phonon number
    ax2 = axes[0, 1]
    ax2.plot(t * 1e6, n_phonon, 'b-', linewidth=2)
    ax2.set_xlabel('Time (μs)', fontsize=12)
    ax2.set_ylabel('Mean phonon number ⟨n⟩', fontsize=12)
    ax2.set_title('Motional Excitation', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Concurrence (entanglement measure)
    ax3 = axes[1, 0]
    ax3.plot(t * 1e6, concurrence, 'r-', linewidth=2)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Maximum')
    ax3.set_xlabel('Time (μs)', fontsize=12)
    ax3.set_ylabel('Concurrence', fontsize=12)
    ax3.set_title('Entanglement (Concurrence)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Final state visualization
    ax4 = axes[1, 1]

    # Extract final spin density matrix
    psi_final = states[-1]
    dim_phonon = 6
    rho_spin = np.zeros((4, 4), dtype=complex)
    for s1 in range(4):
        for s2 in range(4):
            for n in range(dim_phonon):
                idx1 = s1 * dim_phonon + n
                idx2 = s2 * dim_phonon + n
                if idx1 < len(psi_final) and idx2 < len(psi_final):
                    rho_spin[s1, s2] += psi_final[idx1] * psi_final[idx2].conj()

    # Plot density matrix
    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    im = ax4.imshow(np.abs(rho_spin), cmap='viridis', vmin=0, vmax=0.6)
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels(labels)
    ax4.set_yticklabels(labels)
    ax4.set_title('Final Spin State |ρ|', fontsize=14)
    plt.colorbar(im, ax=ax4)

    # Add text annotations
    for i in range(4):
        for j in range(4):
            val = np.abs(rho_spin[i, j])
            if val > 0.1:
                ax4.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig('gate_evolution.png', dpi=150)
    plt.show()


def plot_gate_comparison():
    """Compare different entangling gate schemes"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gate comparison data
    gates = ['MS Gate', 'Cirac-Zoller', 'Light-Shift', 'Geometric']

    # Typical parameters (approximate)
    gate_times = [50, 500, 100, 80]  # μs
    fidelities = [99.9, 99.5, 99.7, 99.8]  # %
    ground_state_req = [False, True, False, False]

    # Bar chart
    ax1 = axes[0]
    x = np.arange(len(gates))
    width = 0.35

    bars1 = ax1.bar(x - width/2, gate_times, width, label='Gate time (μs)', color='blue', alpha=0.7)
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, fidelities, width, label='Fidelity (%)', color='green', alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(gates)
    ax1.set_ylabel('Gate time (μs)', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Fidelity (%)', fontsize=12, color='green')
    ax1.set_title('Gate Comparison', fontsize=14)

    # Add ground state requirement markers
    for i, req in enumerate(ground_state_req):
        if req:
            ax1.scatter([i], [gate_times[i] + 50], marker='*', s=200, color='red', zorder=5)
    ax1.scatter([], [], marker='*', s=200, color='red', label='Requires ground state')
    ax1.legend(loc='upper left')

    # Fidelity vs heating rate
    ax2 = axes[1]

    heating_rates = np.logspace(0, 4, 50)  # quanta/s

    ms_fidelity = 1 - heating_rates * 50e-6  # 50 μs gate
    cz_fidelity = 1 - heating_rates * 500e-6  # 500 μs gate

    ms_fidelity = np.maximum(ms_fidelity, 0.9)
    cz_fidelity = np.maximum(cz_fidelity, 0.8)

    ax2.semilogx(heating_rates, ms_fidelity * 100, 'b-', linewidth=2, label='MS Gate (50 μs)')
    ax2.semilogx(heating_rates, cz_fidelity * 100, 'r--', linewidth=2, label='CZ Gate (500 μs)')

    ax2.axhline(y=99, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=99.9, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Heating rate (quanta/s)', fontsize=12)
    ax2.set_ylabel('Gate fidelity (%)', fontsize=12)
    ax2.set_title('Fidelity vs Heating Rate', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(95, 100)

    plt.tight_layout()
    plt.savefig('gate_comparison.png', dpi=150)
    plt.show()


def main():
    """Main simulation routine"""
    print("=" * 60)
    print("Day 908: Two-Qubit Entangling Gate Simulation")
    print("=" * 60)

    # MS gate parameters
    print("\n--- MS Gate Analysis ---")
    omega = 2 * np.pi * 1e6  # 1 MHz trap
    eta = 0.1
    Omega = 2 * np.pi * 50e3  # 50 kHz Rabi
    delta = 2 * np.pi * 20e3  # 20 kHz detuning

    ms = MSGate(omega, eta, Omega, delta)

    print(f"Gate time: {ms.gate_time() * 1e6:.2f} μs")
    print(f"Maximum displacement: |α_max| = {eta * Omega / delta * 2:.3f}")
    print(f"Entanglement phase: χ = {ms.entanglement_phase():.4f} rad")
    print(f"Expected for max entanglement: χ = π/4 = {np.pi/4:.4f} rad")

    print("\nGenerating phase space plots...")
    plot_phase_space()

    print("\nGenerating gate evolution plots...")
    plot_gate_evolution()

    print("\nGenerating gate comparison plots...")
    plot_gate_comparison()

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| MS Hamiltonian | $\hat{H}_{MS} = \hbar\Omega\hat{S}_x(\hat{a}e^{-i\delta t} + \hat{a}^\dagger e^{i\delta t})$ |
| Displacement | $\alpha(t) = \frac{\Omega}{\delta}(e^{-i\delta t} - 1)$ |
| Geometric phase | $\phi = \pi|\alpha_{max}|^2$ |
| Gate time | $t_{gate} = 2\pi/\delta$ |
| Max entanglement | $\Omega^2/\delta^2 \cdot 2\pi = \pi/4$ |
| RSB Rabi frequency | $\Omega_{RSB} = \eta\sqrt{n}\Omega_0$ |

### Main Takeaways

1. **MS gate** uses bichromatic drive to create spin-dependent displacements
2. **Geometric phase** from enclosed phase space area creates entanglement
3. **Critical condition:** Motion must return to origin (complete loops)
4. **Gate fidelity** limited by heating, off-resonant excitation, and laser noise
5. State-of-the-art: >99.9% two-qubit gate fidelity
6. All-to-all connectivity is a key advantage of trapped ions

## Daily Checklist

- [ ] I can derive the MS gate Hamiltonian and dynamics
- [ ] I understand geometric phase accumulation in phase space
- [ ] I can calculate gate parameters for maximum entanglement
- [ ] I understand the Cirac-Zoller gate protocol
- [ ] I can analyze error sources affecting gate fidelity
- [ ] I can compare different entangling gate schemes
- [ ] I have run the computational lab simulations

## Preview of Day 909

Tomorrow we explore **Ion Shuttling and QCCD Architecture**:
- Quantum charge-coupled device concept
- Ion transport in segmented traps
- Junction designs and shuttling protocols
- Scalability considerations

We will learn how trapped ion systems can scale to larger qubit numbers.

---

*Day 908 of the QSE PhD Curriculum - Year 2, Month 33: Hardware Implementations I*
