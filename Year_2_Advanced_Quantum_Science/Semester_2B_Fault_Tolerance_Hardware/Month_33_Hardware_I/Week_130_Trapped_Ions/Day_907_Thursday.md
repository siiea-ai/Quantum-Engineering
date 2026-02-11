# Day 907: Single-Qubit Gates in Trapped Ions

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Raman transitions, microwave gates, pulse design |
| Afternoon | 2 hours | Problem solving: gate parameters and fidelity |
| Evening | 2 hours | Computational lab: Rabi oscillation simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive Raman transition dynamics** for coherent qubit control
2. **Design microwave gates** for hyperfine qubits
3. **Calculate gate times and Rabi frequencies** for target rotations
4. **Analyze error sources** affecting single-qubit gate fidelity
5. **Implement composite pulse sequences** for robust gates
6. **Optimize pulse shapes** to minimize errors

## Core Content

### 1. Introduction to Single-Qubit Control

Single-qubit gates in trapped ions manipulate the internal qubit state while leaving the motional state unchanged. The goal is to implement arbitrary rotations:

$$R_{\hat{n}}(\theta) = \exp\left(-i\frac{\theta}{2}\hat{n}\cdot\vec{\sigma}\right)$$

**Key gates:**
- $X(\theta)$: Rotation about x-axis
- $Y(\theta)$: Rotation about y-axis
- $Z(\theta)$: Rotation about z-axis (phase gate)

The Hadamard gate and arbitrary single-qubit gates decompose into these rotations.

### 2. Direct Optical Transitions

For optical qubits (e.g., $^{40}$Ca$^+$ S-D transition), a single laser directly drives the qubit:

$$\hat{H}_{int} = \frac{\hbar\Omega}{2}(|e\rangle\langle g|e^{-i\phi} + |g\rangle\langle e|e^{i\phi})$$

where $\Omega$ is the Rabi frequency and $\phi$ is the laser phase.

**Rabi oscillations:**
$$P_e(t) = \sin^2\left(\frac{\Omega t}{2}\right)$$

**Gate implementation:**
- $\pi$-pulse: $\Omega t = \pi$ → $|g\rangle \leftrightarrow |e\rangle$ (bit flip)
- $\pi/2$-pulse: $\Omega t = \pi/2$ → creates superposition
- Phase $\phi$ controls rotation axis in x-y plane

### 3. Stimulated Raman Transitions

For hyperfine qubits, the qubit splitting ($\sim$ GHz) requires indirect driving via two laser beams.

#### Three-Level System

Consider states:
- $|g\rangle = |0\rangle$: qubit ground state (hyperfine $F=0$)
- $|e\rangle = |1\rangle$: qubit excited state (hyperfine $F=1$)
- $|r\rangle$: excited electronic state (e.g., $P_{1/2}$)

Two lasers with frequencies $\omega_1$ and $\omega_2$ couple these states:

$$\omega_1 - \omega_2 = \omega_{ge} + \delta$$

where $\omega_{ge}$ is the qubit splitting and $\delta$ is a small detuning.

#### Effective Two-Level Hamiltonian

When both beams are detuned from $|r\rangle$ by $\Delta \gg \Omega_1, \Omega_2, \Gamma$:

$$\boxed{\hat{H}_{eff} = \frac{\hbar\Omega_{eff}}{2}(|e\rangle\langle g|e^{-i\phi_{eff}} + h.c.) + \frac{\hbar\delta}{2}\hat{\sigma}_z}$$

**Effective Rabi frequency:**
$$\boxed{\Omega_{eff} = \frac{\Omega_1\Omega_2}{2\Delta}}$$

**AC Stark shifts (to be canceled):**
$$\delta_{AC} = \frac{|\Omega_1|^2 - |\Omega_2|^2}{4\Delta}$$

#### Advantages of Raman Transitions

1. **No spontaneous emission** (far detuned from excited state)
2. **Momentum transfer** $\Delta\vec{k} = \vec{k}_1 - \vec{k}_2$ enables motional coupling
3. **Phase control** via relative beam phases
4. **Fast gates** with high laser power

### 4. Microwave Gates

For hyperfine qubits, microwaves can directly drive transitions at the GHz splitting.

#### Direct Microwave Drive

$$\hat{H}_{MW} = \frac{\hbar\Omega_{MW}}{2}(\hat{\sigma}_+ e^{-i\omega_{MW}t} + \hat{\sigma}_- e^{i\omega_{MW}t})$$

**Rabi frequency:**
$$\Omega_{MW} = \frac{\vec{\mu}\cdot\vec{B}_{MW}}{\hbar}$$

where $\vec{\mu}$ is the magnetic dipole moment.

#### Comparison: Raman vs Microwave

| Property | Raman | Microwave |
|----------|-------|-----------|
| Frequency | Optical (THz) | Microwave (GHz) |
| Individual addressing | Yes (focused beams) | Challenging (wavelength ~ cm) |
| Rabi frequency | 10 kHz - 1 MHz | 1 - 100 kHz |
| Error sources | Laser noise, AC Stark | B-field noise |
| Complexity | High (two beams, locking) | Low |
| Motional coupling | Yes (sidebands) | No |

#### Global vs Individual Addressing

- **Global microwave:** Acts on all ions simultaneously
- **Individual Raman:** Focused beams select single ions ($\sim 1-2$ μm spot)
- **Addressed microwave:** Magnetic field gradients + frequency selection

### 5. Bloch Sphere Representation

The qubit state $|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$ maps to:

$$\vec{n} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$$

**Resonant pulses:**
- Drive at carrier frequency $\omega = \omega_{ge}$
- Rotation axis determined by phase $\phi_L$
- Rotation angle $\theta = \Omega t$

$$R_{\phi_L}(\theta) = \begin{pmatrix} \cos(\theta/2) & -ie^{-i\phi_L}\sin(\theta/2) \\ -ie^{i\phi_L}\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

**Standard gates:**
- $X = R_0(\pi)$: Phase $\phi_L = 0$
- $Y = R_{\pi/2}(\pi)$: Phase $\phi_L = \pi/2$
- $H = R_0(\pi/2) \cdot R_{\pi/2}(\pi)$: Hadamard

### 6. Gate Errors and Fidelity

#### Error Sources

1. **Rabi frequency fluctuations** ($\delta\Omega/\Omega$)
   - From laser intensity noise
   - Beam pointing fluctuations

2. **Detuning errors** ($\delta$)
   - Frequency drift
   - AC Stark shift variations

3. **Phase noise** ($\delta\phi$)
   - Laser phase noise (Raman)
   - Local oscillator noise (microwave)

4. **Timing errors** ($\delta t$)
   - Pulse generator resolution
   - Propagation delays

5. **Off-resonant excitation**
   - Leakage to other levels
   - Motional sideband excitation

#### Fidelity Calculation

Gate fidelity:
$$F = |\langle\psi_{ideal}|\psi_{actual}\rangle|^2$$

For a $\pi$-pulse with small errors:
$$\boxed{1 - F \approx \left(\frac{\delta\Omega}{\Omega}\right)^2 + \left(\frac{\delta}{\Omega}\right)^2 + (\delta\phi)^2}$$

**State-of-the-art:** $F > 99.99\%$ for single-qubit gates.

### 7. Composite Pulses

Composite pulse sequences cancel systematic errors through clever pulse combinations.

#### BB1 (Broadband 1) Sequence

Replaces a simple $\pi$-pulse with:
$$\pi_0 \rightarrow \pi_\phi \cdot 2\pi_{3\phi} \cdot \pi_\phi \cdot \pi_0$$

where $\phi = \arccos(-1/4) \approx 104.5°$

This cancels:
- First-order pulse area errors
- First-order detuning errors

#### CORPSE (Compensation for Off-Resonance with Pulse SEquence)

For a target rotation $\theta$:
$$\theta_0 \rightarrow \theta_1(\phi_1) \cdot \theta_2(\phi_2) \cdot \theta_3(\phi_3)$$

Parameters chosen to cancel off-resonance errors.

#### SK1 (Solovay-Kitaev) Sequence

$$\theta_0 \rightarrow \theta_0 \cdot \phi_1 \cdot \phi_2 \cdot \phi_1^\dagger \cdot \phi_2^\dagger$$

Provides arbitrary-order error suppression.

### 8. Pulse Shaping

Beyond square pulses, shaped pulses reduce errors:

#### Gaussian Pulses

$$\Omega(t) = \Omega_0 \exp\left(-\frac{(t-t_0)^2}{2\sigma^2}\right)$$

- Reduced spectral width → less off-resonant excitation
- Smooth turn-on/off → reduced diabatic errors

#### DRAG (Derivative Removal by Adiabatic Gate)

$$\Omega_x(t) = \Omega(t), \quad \Omega_y(t) = -\frac{\dot{\Omega}(t)}{\Delta}$$

Cancels leakage to nearby levels by adding quadrature component.

#### Optimal Control

Numerically optimize pulse shape to maximize fidelity:
$$\max_{\Omega(t)} F[\Omega(t)]$$

subject to constraints on power, bandwidth, etc.

GRAPE (Gradient Ascent Pulse Engineering) and Krotov methods are commonly used.

## Quantum Computing Applications

### Gate Compilation

Any single-qubit gate can be decomposed:
$$U = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)$$

For trapped ions:
- $R_z$ often implemented as virtual phase shift (zero time)
- $R_y$, $R_x$ require physical pulses

### Calibration Protocols

1. **Rabi oscillation scan:** Determine $\Omega$
2. **Ramsey sequence:** Measure detuning
3. **Spin echo:** Measure $T_2$
4. **Randomized benchmarking:** Extract gate fidelity

### Typical Parameters

| Parameter | Raman | Microwave |
|-----------|-------|-----------|
| $\Omega/2\pi$ | 100 kHz - 1 MHz | 10-100 kHz |
| Gate time ($\pi$) | 1-10 μs | 10-100 μs |
| Fidelity | 99.99% | 99.99% |
| Addressing | ~1 μm spot | Global or gradient |

## Worked Examples

### Example 1: Raman Rabi Frequency

**Problem:** Calculate the effective Rabi frequency for Raman beams with individual Rabi frequencies $\Omega_1/2\pi = 50$ MHz and $\Omega_2/2\pi = 40$ MHz, detuned by $\Delta/2\pi = 10$ GHz from the excited state.

**Solution:**

$$\Omega_{eff} = \frac{\Omega_1 \Omega_2}{2\Delta} = \frac{2\pi \times 50 \times 10^6 \times 2\pi \times 40 \times 10^6}{2 \times 2\pi \times 10 \times 10^9}$$

$$\Omega_{eff} = \frac{50 \times 40}{2 \times 10000} \times 2\pi \times 10^6 = \frac{2000}{20000} \times 2\pi \times 10^6$$

$$\boxed{\Omega_{eff}/2\pi = 100 \text{ kHz}}$$

Gate time for $\pi$-pulse:
$$t_\pi = \frac{\pi}{\Omega_{eff}} = \frac{1}{2 \times 100 \text{ kHz}} = 5 \text{ μs}$$

### Example 2: Spontaneous Emission Probability

**Problem:** For the Raman transition above, estimate the probability of spontaneous emission during a $\pi$-pulse. The excited state linewidth is $\Gamma/2\pi = 20$ MHz.

**Solution:**

Effective scattering rate during the pulse:
$$R_{sc} = \frac{\Gamma}{2}\left(\frac{\Omega_1^2 + \Omega_2^2}{4\Delta^2}\right)$$

$$R_{sc} = \frac{2\pi \times 20 \times 10^6}{2} \times \frac{(50)^2 + (40)^2}{4 \times (10000)^2} \times 10^{12}$$

$$R_{sc} = \pi \times 20 \times 10^6 \times \frac{4100}{4 \times 10^8} = \pi \times 20 \times 10^6 \times 1.025 \times 10^{-5}$$

$$R_{sc} \approx 640 \text{ s}^{-1}$$

Probability during $t_\pi = 5$ μs:
$$P_{sc} = R_{sc} \times t_\pi = 640 \times 5 \times 10^{-6}$$

$$\boxed{P_{sc} \approx 3.2 \times 10^{-3} = 0.32\%}$$

This limits gate fidelity. Increasing $\Delta$ reduces scattering but also reduces $\Omega_{eff}$.

### Example 3: Gate Error from Intensity Noise

**Problem:** If the laser intensity has 1% RMS fluctuations, what is the resulting gate infidelity for a $\pi$-pulse?

**Solution:**

For Raman transitions: $\Omega_{eff} \propto I_1 \cdot I_2 \propto I^2$ (if beams from same source)

So: $\delta\Omega/\Omega = 2 \cdot \delta I/I = 2 \times 0.01 = 0.02$

Gate infidelity:
$$1 - F \approx \left(\frac{\delta\Omega}{\Omega}\right)^2 \times \text{factor}$$

For a $\pi$-pulse with Gaussian noise:
$$1 - F \approx \frac{1}{2}\left(\frac{\pi \delta\Omega}{\Omega}\right)^2 = \frac{1}{2}(\pi \times 0.02)^2$$

$$\boxed{1 - F \approx 2 \times 10^{-3} = 0.2\%}$$

This motivates intensity stabilization to <0.1% for high-fidelity gates.

## Practice Problems

### Level 1: Direct Application

1. Calculate the $\pi$-pulse time for a microwave gate with $\Omega_{MW}/2\pi = 50$ kHz.

2. If a Raman beam has detuning $\Delta/2\pi = 5$ GHz and Rabi frequency $\Omega_1/2\pi = 30$ MHz, what is the AC Stark shift of the ground state?

3. For a gate with 99.9% fidelity, what is the error per gate in parts per million?

### Level 2: Intermediate

4. Design a Raman system to achieve $\Omega_{eff}/2\pi = 500$ kHz with spontaneous emission probability < 0.1% per gate. Specify $\Omega_1$, $\Omega_2$, and $\Delta$.

5. A Ramsey sequence shows oscillations at 1 kHz when the expected frequency is 0 Hz. What is the detuning? How does this affect gate fidelity?

6. Compare the susceptibility of Raman and microwave gates to (a) magnetic field noise and (b) laser phase noise.

### Level 3: Challenging

7. Derive the BB1 composite pulse sequence and show that it cancels first-order pulse area errors.

8. Design an optimal control pulse for a $\pi/2$-gate that minimizes motional excitation while maintaining 99.99% fidelity.

9. For a two-ion system, analyze crosstalk errors when addressing one ion with a focused Raman beam. How does beam waist affect crosstalk?

## Computational Lab: Rabi Oscillation Simulation

```python
"""
Day 907 Computational Lab: Single-Qubit Gate Simulation
Simulating Rabi oscillations, pulse sequences, and gate errors
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import expm
from scipy.optimize import minimize

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_p = (sigma_x + 1j * sigma_y) / 2
sigma_m = (sigma_x - 1j * sigma_y) / 2
identity = np.eye(2, dtype=complex)


class SingleQubitGate:
    """Simulate single-qubit gates in trapped ions"""

    def __init__(self, omega_rabi, detuning=0, phase=0):
        """
        Parameters:
        -----------
        omega_rabi : float - Rabi frequency (rad/s)
        detuning : float - Frequency detuning (rad/s)
        phase : float - Drive phase (rad)
        """
        self.Omega = omega_rabi
        self.delta = detuning
        self.phi = phase

    def hamiltonian(self, t, Omega_t=None):
        """Time-dependent Hamiltonian"""
        if Omega_t is None:
            Omega = self.Omega
        else:
            Omega = Omega_t

        # In rotating frame
        H = (self.delta / 2) * sigma_z
        H += (Omega / 2) * (np.cos(self.phi) * sigma_x + np.sin(self.phi) * sigma_y)
        return H

    def evolve(self, psi0, t_final, n_steps=1000, Omega_func=None):
        """Evolve state under Hamiltonian"""
        t = np.linspace(0, t_final, n_steps)
        dt = t[1] - t[0]

        psi = psi0.copy().astype(complex)
        populations = np.zeros((n_steps, 2))
        populations[0] = np.abs(psi)**2

        for i in range(1, n_steps):
            if Omega_func is not None:
                Omega_t = Omega_func(t[i])
            else:
                Omega_t = None

            H = self.hamiltonian(t[i], Omega_t)
            U = expm(-1j * H * dt)
            psi = U @ psi
            populations[i] = np.abs(psi)**2

        return t, populations, psi

    def rotation_matrix(self, theta, phi):
        """Rotation matrix R_phi(theta)"""
        return (np.cos(theta/2) * identity
                - 1j * np.sin(theta/2) * (np.cos(phi) * sigma_x + np.sin(phi) * sigma_y))

    def gate_fidelity(self, U_actual, U_target):
        """Calculate gate fidelity"""
        d = 2  # Qubit dimension
        return np.abs(np.trace(U_actual.conj().T @ U_target))**2 / d**2


def raman_transition(Omega1, Omega2, Delta, gamma):
    """
    Calculate Raman transition parameters

    Parameters:
    -----------
    Omega1, Omega2 : float - Individual Rabi frequencies (rad/s)
    Delta : float - Detuning from excited state (rad/s)
    gamma : float - Excited state decay rate (rad/s)

    Returns:
    --------
    Omega_eff : float - Effective Rabi frequency
    R_scatter : float - Scattering rate
    """
    Omega_eff = Omega1 * Omega2 / (2 * Delta)
    R_scatter = (gamma / 2) * (Omega1**2 + Omega2**2) / (4 * Delta**2)

    return Omega_eff, R_scatter


def bb1_pulse(theta):
    """
    Generate BB1 composite pulse sequence

    Parameters:
    -----------
    theta : float - Target rotation angle

    Returns:
    --------
    pulses : list of (theta, phi) pairs
    """
    phi = np.arccos(-1/4)  # ~104.5 degrees

    pulses = [
        (theta, 0),           # θ_0
        (np.pi, phi),         # π_φ
        (2*np.pi, 3*phi),     # 2π_{3φ}
        (np.pi, phi)          # π_φ
    ]

    return pulses


def plot_rabi_oscillations():
    """Plot basic Rabi oscillations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Basic Rabi oscillation
    ax1 = axes[0, 0]
    Omega = 2 * np.pi * 100e3  # 100 kHz
    gate = SingleQubitGate(omega_rabi=Omega)

    psi0 = np.array([1, 0], dtype=complex)  # Start in |0⟩
    t, pop, _ = gate.evolve(psi0, t_final=20e-6, n_steps=500)

    ax1.plot(t * 1e6, pop[:, 0], 'b-', label='P(|0⟩)', linewidth=2)
    ax1.plot(t * 1e6, pop[:, 1], 'r-', label='P(|1⟩)', linewidth=2)
    ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='π-pulse')
    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Population', fontsize=12)
    ax1.set_title('Rabi Oscillations (Ω/2π = 100 kHz)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Effect of detuning
    ax2 = axes[0, 1]
    detunings = [0, 0.2, 0.5, 1.0]  # In units of Omega

    for delta_ratio in detunings:
        gate = SingleQubitGate(omega_rabi=Omega, detuning=delta_ratio * Omega)
        t, pop, _ = gate.evolve(psi0, t_final=20e-6, n_steps=500)
        ax2.plot(t * 1e6, pop[:, 1], label=f'δ/Ω = {delta_ratio}', linewidth=2)

    ax2.set_xlabel('Time (μs)', fontsize=12)
    ax2.set_ylabel('P(|1⟩)', fontsize=12)
    ax2.set_title('Effect of Detuning on Rabi Oscillations', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Different pulse phases
    ax3 = axes[1, 0]
    phases = [0, np.pi/4, np.pi/2, np.pi]
    colors = ['blue', 'orange', 'green', 'red']

    # Start from superposition state
    psi0_super = np.array([1, 1], dtype=complex) / np.sqrt(2)

    for phase, color in zip(phases, colors):
        gate = SingleQubitGate(omega_rabi=Omega, phase=phase)
        t, pop, _ = gate.evolve(psi0_super, t_final=10e-6, n_steps=500)
        ax3.plot(t * 1e6, pop[:, 1], color=color,
                label=f'φ = {phase*180/np.pi:.0f}°', linewidth=2)

    ax3.set_xlabel('Time (μs)', fontsize=12)
    ax3.set_ylabel('P(|1⟩)', fontsize=12)
    ax3.set_title('Effect of Drive Phase (starting from |+⟩)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Generalized Rabi frequency
    ax4 = axes[1, 1]
    delta_range = np.linspace(-3, 3, 100)  # In units of Omega
    Omega_gen = np.sqrt(1 + delta_range**2)  # Generalized Rabi frequency

    ax4.plot(delta_range, Omega_gen, 'b-', linewidth=2)
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Detuning δ/Ω', fontsize=12)
    ax4.set_ylabel("Generalized Rabi Ω'/Ω", fontsize=12)
    ax4.set_title('Generalized Rabi Frequency', fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rabi_oscillations.png', dpi=150)
    plt.show()


def plot_gate_errors():
    """Analyze gate errors"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    Omega = 2 * np.pi * 100e3

    # Pulse area error
    ax1 = axes[0, 0]
    area_errors = np.linspace(-0.1, 0.1, 100)  # Fractional error

    infidelities = []
    for err in area_errors:
        # Apply pulse with error
        theta_actual = np.pi * (1 + err)
        gate = SingleQubitGate(omega_rabi=Omega)
        U_actual = gate.rotation_matrix(theta_actual, 0)
        U_target = gate.rotation_matrix(np.pi, 0)
        F = gate.gate_fidelity(U_actual, U_target)
        infidelities.append(1 - F)

    ax1.semilogy(area_errors * 100, infidelities, 'b-', linewidth=2)
    ax1.set_xlabel('Pulse area error (%)', fontsize=12)
    ax1.set_ylabel('Infidelity (1-F)', fontsize=12)
    ax1.set_title('Gate Error from Pulse Area', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e-4, color='red', linestyle='--', label='99.99% threshold')
    ax1.legend()

    # Detuning error
    ax2 = axes[0, 1]
    detuning_errors = np.linspace(-0.2, 0.2, 100)  # In units of Omega

    infidelities = []
    for delta in detuning_errors:
        gate = SingleQubitGate(omega_rabi=Omega, detuning=delta * Omega)
        # Pulse time for π rotation
        t_pi = np.pi / Omega
        psi0 = np.array([1, 0], dtype=complex)
        _, _, psi_final = gate.evolve(psi0, t_final=t_pi, n_steps=100)

        # Target is |1⟩
        F = np.abs(psi_final[1])**2
        infidelities.append(1 - F)

    ax2.semilogy(detuning_errors * 100, infidelities, 'r-', linewidth=2)
    ax2.set_xlabel('Detuning error δ/Ω (%)', fontsize=12)
    ax2.set_ylabel('Infidelity (1-F)', fontsize=12)
    ax2.set_title('Gate Error from Detuning', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Combined error map
    ax3 = axes[1, 0]
    area_err = np.linspace(-0.05, 0.05, 50)
    det_err = np.linspace(-0.1, 0.1, 50)
    AREA, DET = np.meshgrid(area_err, det_err)

    infidelity_map = np.zeros_like(AREA)
    gate = SingleQubitGate(omega_rabi=Omega)

    for i in range(len(det_err)):
        for j in range(len(area_err)):
            theta = np.pi * (1 + AREA[i, j])
            # Approximate detuning effect
            Omega_eff = np.sqrt(Omega**2 + (DET[i, j] * Omega)**2)
            theta_eff = theta * Omega_eff / Omega

            U_actual = gate.rotation_matrix(theta_eff, 0)
            U_target = gate.rotation_matrix(np.pi, 0)
            F = gate.gate_fidelity(U_actual, U_target)
            infidelity_map[i, j] = 1 - F

    contour = ax3.contourf(AREA * 100, DET * 100, np.log10(infidelity_map),
                          levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax3, label='log₁₀(1-F)')
    ax3.set_xlabel('Pulse area error (%)', fontsize=12)
    ax3.set_ylabel('Detuning error (%)', fontsize=12)
    ax3.set_title('Combined Error Map for π-pulse', fontsize=14)

    # Raman scattering vs gate speed trade-off
    ax4 = axes[1, 1]
    Delta_range = np.logspace(9, 11, 100)  # 1 GHz to 100 GHz detuning

    # Fixed individual Rabi frequencies
    Omega1 = 2 * np.pi * 50e6  # 50 MHz
    Omega2 = 2 * np.pi * 50e6
    gamma = 2 * np.pi * 20e6  # 20 MHz linewidth

    Omega_eff = []
    P_scatter = []

    for Delta in Delta_range:
        Omega_e, R_sc = raman_transition(Omega1, Omega2, Delta, gamma)
        Omega_eff.append(Omega_e / (2 * np.pi))  # Convert to Hz
        t_pi = np.pi / Omega_e
        P_scatter.append(R_sc * t_pi)

    ax4_twin = ax4.twinx()

    line1, = ax4.loglog(Delta_range / (2 * np.pi * 1e9), Omega_eff, 'b-',
                        linewidth=2, label='Ω_eff/2π')
    line2, = ax4_twin.loglog(Delta_range / (2 * np.pi * 1e9), P_scatter, 'r-',
                             linewidth=2, label='P_scatter')

    ax4.set_xlabel('Detuning Δ/2π (GHz)', fontsize=12)
    ax4.set_ylabel('Effective Rabi Ω_eff/2π (Hz)', fontsize=12, color='blue')
    ax4_twin.set_ylabel('Scattering probability', fontsize=12, color='red')
    ax4.set_title('Raman Gate: Speed vs Scattering Trade-off', fontsize=14)

    ax4.axhline(y=1e5, color='blue', linestyle=':', alpha=0.5)
    ax4_twin.axhline(y=1e-3, color='red', linestyle=':', alpha=0.5)

    lines = [line1, line2]
    ax4.legend(lines, [l.get_label() for l in lines], loc='upper right')

    plt.tight_layout()
    plt.savefig('gate_errors.png', dpi=150)
    plt.show()


def plot_composite_pulses():
    """Compare simple and composite pulse sequences"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    Omega = 2 * np.pi * 100e3

    # Compare pulse sequences with area error
    ax1 = axes[0, 0]
    area_errors = np.linspace(-0.2, 0.2, 100)

    simple_infidelity = []
    bb1_infidelity = []

    gate = SingleQubitGate(omega_rabi=Omega)

    for err in area_errors:
        # Simple π pulse
        theta = np.pi * (1 + err)
        U_simple = gate.rotation_matrix(theta, 0)

        # BB1 composite pulse
        phi_bb1 = np.arccos(-1/4)
        U_bb1 = (gate.rotation_matrix(np.pi * (1 + err), phi_bb1) @
                gate.rotation_matrix(2 * np.pi * (1 + err), 3 * phi_bb1) @
                gate.rotation_matrix(np.pi * (1 + err), phi_bb1) @
                gate.rotation_matrix(np.pi * (1 + err), 0))

        U_target = gate.rotation_matrix(np.pi, 0)

        simple_infidelity.append(1 - gate.gate_fidelity(U_simple, U_target))
        bb1_infidelity.append(1 - gate.gate_fidelity(U_bb1, U_target))

    ax1.semilogy(area_errors * 100, simple_infidelity, 'b-',
                label='Simple π', linewidth=2)
    ax1.semilogy(area_errors * 100, bb1_infidelity, 'r-',
                label='BB1', linewidth=2)
    ax1.set_xlabel('Pulse area error (%)', fontsize=12)
    ax1.set_ylabel('Infidelity (1-F)', fontsize=12)
    ax1.set_title('BB1 vs Simple Pulse: Area Error', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e-4, color='gray', linestyle='--', alpha=0.5)

    # Pulse shapes
    ax2 = axes[0, 1]
    t = np.linspace(0, 10, 1000)
    t_center = 5
    sigma = 1.5

    # Square pulse
    square = np.ones_like(t)
    square[t < 1] = 0
    square[t > 9] = 0

    # Gaussian pulse
    gaussian = np.exp(-(t - t_center)**2 / (2 * sigma**2))

    # DRAG-like (with quadrature)
    drag_x = gaussian
    drag_y = -(t - t_center) / sigma**2 * gaussian * 0.3

    ax2.plot(t, square, 'b-', label='Square', linewidth=2)
    ax2.plot(t, gaussian, 'r-', label='Gaussian', linewidth=2)
    ax2.plot(t, drag_x, 'g-', label='DRAG (X)', linewidth=2)
    ax2.plot(t, drag_y, 'g--', label='DRAG (Y)', linewidth=2)
    ax2.set_xlabel('Time (arb.)', fontsize=12)
    ax2.set_ylabel('Amplitude (arb.)', fontsize=12)
    ax2.set_title('Pulse Shapes', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bloch sphere trajectories
    ax3 = axes[1, 0]

    # Simulate evolution on Bloch sphere
    def bloch_coords(psi):
        rho = np.outer(psi, psi.conj())
        x = np.real(np.trace(sigma_x @ rho))
        y = np.real(np.trace(sigma_y @ rho))
        z = np.real(np.trace(sigma_z @ rho))
        return x, y, z

    # π/2 rotation from |0⟩
    psi0 = np.array([1, 0], dtype=complex)
    gate = SingleQubitGate(omega_rabi=Omega, phase=0)

    t, _, _ = gate.evolve(psi0, t_final=2.5e-6, n_steps=100)

    x_traj, y_traj, z_traj = [], [], []
    psi = psi0.copy()
    H = gate.hamiltonian(0)
    dt = t[1] - t[0]

    for i in range(len(t)):
        U = expm(-1j * H * dt)
        psi = U @ psi
        x, y, z = bloch_coords(psi)
        x_traj.append(x)
        y_traj.append(y)
        z_traj.append(z)

    ax3.plot(y_traj, z_traj, 'b-', linewidth=2, label='X rotation')

    # Y rotation
    gate_y = SingleQubitGate(omega_rabi=Omega, phase=np.pi/2)
    psi = psi0.copy()
    H = gate_y.hamiltonian(0)

    x_traj_y, y_traj_y, z_traj_y = [], [], []
    for i in range(len(t)):
        U = expm(-1j * H * dt)
        psi = U @ psi
        x, y, z = bloch_coords(psi)
        x_traj_y.append(x)
        y_traj_y.append(y)
        z_traj_y.append(z)

    ax3.plot(x_traj_y, z_traj_y, 'r-', linewidth=2, label='Y rotation')

    # Draw Bloch sphere outline
    theta_sphere = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta_sphere), np.sin(theta_sphere), 'k-', alpha=0.3)
    ax3.scatter([0], [1], color='green', s=100, marker='^', label='|0⟩')
    ax3.scatter([0], [-1], color='red', s=100, marker='v', label='|1⟩')

    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_aspect('equal')
    ax3.set_xlabel('Y (or X for Y-rot)', fontsize=12)
    ax3.set_ylabel('Z', fontsize=12)
    ax3.set_title('Bloch Sphere Trajectories', fontsize=14)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    # Fidelity vs number of composite pulse elements
    ax4 = axes[1, 1]

    # Simulate increasing composite pulse complexity
    err = 0.05  # 5% pulse area error

    fidelities = []
    pulse_counts = [1, 3, 5, 7, 9, 11]

    # Simplified model: each additional pair of pulses reduces error order
    for n in pulse_counts:
        if n == 1:
            F = 1 - (np.pi * err)**2 / 2
        elif n == 3:
            F = 1 - (np.pi * err)**3 / 4
        elif n == 5:
            F = 1 - (np.pi * err)**4 / 8
        else:
            F = 1 - (np.pi * err)**((n+1)/2) / (2**(n//2))
        fidelities.append(1 - F)

    ax4.semilogy(pulse_counts, fidelities, 'bo-', markersize=10, linewidth=2)
    ax4.set_xlabel('Number of pulses in sequence', fontsize=12)
    ax4.set_ylabel('Infidelity (1-F)', fontsize=12)
    ax4.set_title('Composite Pulse Error Suppression (5% area error)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1e-4, color='red', linestyle='--', label='99.99% target')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('composite_pulses.png', dpi=150)
    plt.show()


def main():
    """Main simulation routine"""
    print("=" * 60)
    print("Day 907: Single-Qubit Gate Simulation")
    print("=" * 60)

    # Raman transition parameters
    print("\n--- Raman Transition Analysis ---")
    Omega1 = 2 * np.pi * 50e6  # 50 MHz
    Omega2 = 2 * np.pi * 40e6  # 40 MHz
    Delta = 2 * np.pi * 10e9  # 10 GHz
    gamma = 2 * np.pi * 20e6  # 20 MHz

    Omega_eff, R_sc = raman_transition(Omega1, Omega2, Delta, gamma)
    t_pi = np.pi / Omega_eff
    P_sc = R_sc * t_pi

    print(f"Effective Rabi frequency: Ω_eff/2π = {Omega_eff/(2*np.pi)/1e3:.1f} kHz")
    print(f"π-pulse time: {t_pi*1e6:.2f} μs")
    print(f"Scattering probability per gate: {P_sc*100:.3f}%")

    print("\nGenerating Rabi oscillation plots...")
    plot_rabi_oscillations()

    print("\nGenerating gate error analysis...")
    plot_gate_errors()

    print("\nGenerating composite pulse comparison...")
    plot_composite_pulses()

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
| Rabi oscillation | $P_e(t) = \sin^2(\Omega t/2)$ |
| Raman Rabi frequency | $\Omega_{eff} = \frac{\Omega_1\Omega_2}{2\Delta}$ |
| π-pulse time | $t_\pi = \frac{\pi}{\Omega}$ |
| Gate infidelity | $1-F \approx (\delta\Omega/\Omega)^2 + (\delta/\Omega)^2$ |
| Scattering rate | $R_{sc} = \frac{\Gamma}{2}\frac{\Omega_1^2 + \Omega_2^2}{4\Delta^2}$ |
| Rotation matrix | $R_\phi(\theta) = \cos(\theta/2)I - i\sin(\theta/2)(\cos\phi\sigma_x + \sin\phi\sigma_y)$ |

### Main Takeaways

1. **Single-qubit gates** implement arbitrary Bloch sphere rotations
2. **Raman transitions** enable fast gates for hyperfine qubits with negligible spontaneous emission
3. **Microwave gates** are simpler but lack individual addressing capability
4. **Composite pulses** (BB1, CORPSE) cancel systematic errors to higher orders
5. **Pulse shaping** reduces off-resonant excitation and leakage errors
6. State-of-the-art achieves >99.99% single-qubit gate fidelity

## Daily Checklist

- [ ] I can derive Raman transition dynamics
- [ ] I can calculate effective Rabi frequencies and gate times
- [ ] I understand trade-offs between Raman and microwave gates
- [ ] I can analyze gate errors from various sources
- [ ] I understand composite pulse sequences and their benefits
- [ ] I can design pulse shapes for improved fidelity
- [ ] I have run the computational lab simulations

## Preview of Day 908

Tomorrow we explore **Two-Qubit Entangling Gates**:
- Molmer-Sorensen gate mechanism
- Geometric phase gates
- Cirac-Zoller gate
- Gate fidelity and error sources

We will learn how collective motion mediates entanglement between ion qubits.

---

*Day 907 of the QSE PhD Curriculum - Year 2, Month 33: Hardware Implementations I*
