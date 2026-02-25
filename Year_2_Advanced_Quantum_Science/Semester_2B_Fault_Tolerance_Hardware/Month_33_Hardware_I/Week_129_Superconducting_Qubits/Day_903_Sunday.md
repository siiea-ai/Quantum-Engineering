# Day 903: Coherence and Decoherence

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | $T_1$, $T_2$, $T_2^*$ theory, decoherence mechanisms |
| Afternoon | 2 hours | Noise spectroscopy, materials science, problem solving |
| Evening | 2 hours | Computational lab: Decoherence simulation and noise analysis |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** and **measure** $T_1$, $T_2$, and $T_2^*$ coherence times
2. **Identify** major decoherence mechanisms in superconducting qubits
3. **Analyze** noise spectral density and its impact on coherence
4. **Explain** the role of two-level systems (TLS) in decoherence
5. **Describe** materials and fabrication improvements for longer coherence
6. **Apply** noise spectroscopy techniques to characterize qubit environments

## Core Content

### 1. Coherence Time Definitions

**$T_1$ (Energy Relaxation Time)**:
Time constant for decay from $|1\rangle$ to $|0\rangle$ due to energy exchange with environment.

$$P_1(t) = P_1(0) \cdot e^{-t/T_1}$$

Rate: $\Gamma_1 = 1/T_1$

**$T_2$ (Transverse Relaxation / Dephasing Time)**:
Decay of coherence (off-diagonal elements of density matrix).

$$\rho_{01}(t) = \rho_{01}(0) \cdot e^{-t/T_2}$$

**$T_2^*$ (Inhomogeneous Dephasing Time)**:
Includes both intrinsic dephasing and low-frequency noise.

$$\boxed{\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}}$$

where $T_\phi$ is pure dephasing time.

**Fundamental limit**: $T_2 \leq 2T_1$ (equality when pure dephasing is zero)

### 2. Bloch-Redfield Theory

The qubit density matrix evolves according to the master equation:

$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[\hat{H}, \rho] + \mathcal{L}[\rho]$$

The Lindblad dissipator:

$$\mathcal{L}[\rho] = \sum_k \gamma_k \left(\hat{L}_k\rho\hat{L}_k^\dagger - \frac{1}{2}\{\hat{L}_k^\dagger\hat{L}_k, \rho\}\right)$$

For a qubit:
- **Relaxation**: $\hat{L}_1 = \hat{\sigma}^-$ with rate $\Gamma_1$
- **Pure dephasing**: $\hat{L}_\phi = \hat{\sigma}_z$ with rate $\Gamma_\phi$

### 3. Noise Spectral Density

Environmental fluctuations are characterized by spectral density:

$$S_X(\omega) = \int_{-\infty}^{\infty} \langle X(t)X(0) \rangle e^{i\omega t} dt$$

**Relaxation rate** (Fermi's golden rule):

$$\boxed{\Gamma_1 = \frac{1}{T_1} = \frac{|\langle 0|\hat{V}|1\rangle|^2}{\hbar^2} S_X(\omega_{01})}$$

where $\hat{V}$ is the coupling operator and $S_X(\omega_{01})$ is noise at the qubit frequency.

**Dephasing rate** (low-frequency noise):

$$\Gamma_\phi = \frac{|\langle 1|\hat{V}|1\rangle - \langle 0|\hat{V}|0\rangle|^2}{\hbar^2} S_X(\omega \to 0)$$

### 4. Common Noise Types

**1/f (Pink) Noise**:
$$S_X(\omega) = \frac{A}{\omega^\alpha}, \quad \alpha \approx 1$$

Common for charge and flux noise in superconducting circuits.

**White Noise**:
$$S_X(\omega) = S_0 = \text{constant}$$

Thermal noise at high temperature.

**Ohmic Noise**:
$$S_X(\omega) \propto \omega \cdot \coth\left(\frac{\hbar\omega}{2k_BT}\right)$$

Quantum noise from a resistive environment.

### 5. Major Decoherence Mechanisms

#### Dielectric Loss (Two-Level Systems)

**Two-level systems (TLS)** are defects in amorphous materials (substrate surface, junction oxide, metal interfaces) that couple to qubits.

TLS loss tangent:
$$\tan\delta = \frac{P \cdot d^2}{3\epsilon_0\epsilon_r} \cdot \tanh\left(\frac{\hbar\omega}{2k_BT}\right) \cdot \frac{1}{\hbar\omega}$$

where $P$ is TLS density and $d$ is dipole moment.

**Contribution to $T_1$**:
$$\frac{1}{T_1^{TLS}} = p_{surf} \cdot \omega \cdot \tan\delta_{surf}$$

where $p_{surf}$ is the participation ratio of surface fields.

#### Flux Noise

For flux-tunable qubits:
$$\frac{\partial\omega_q}{\partial\Phi} \neq 0$$

leads to dephasing from flux fluctuations:

$$\Gamma_\phi^{flux} = \left(\frac{\partial\omega_q}{\partial\Phi}\right)^2 S_\Phi(\omega \to 0)$$

Typical flux noise: $S_\Phi^{1/2} \sim 1-10$ $\mu\Phi_0/\sqrt{\text{Hz}}$ at 1 Hz

#### Photon Shot Noise

Residual photons in the readout resonator cause dephasing:

$$\Gamma_\phi^{photon} = 2\chi^2 \bar{n}_{thermal} / \kappa$$

where $\bar{n}_{thermal} = 1/(e^{\hbar\omega_r/k_BT} - 1)$.

At 20 mK for 7 GHz resonator: $\bar{n} \approx 0.01$

#### Quasiparticle Tunneling

Broken Cooper pairs (quasiparticles) can tunnel across the junction, causing:

$$\Gamma_1^{qp} = \frac{x_{qp}}{2\pi\tau_{qp}} \cdot \sqrt{\frac{2\Delta}{\hbar\omega_{01}}}$$

where $x_{qp}$ is quasiparticle density fraction and $\tau_{qp}$ is tunneling time.

### 6. Noise Spectroscopy

**Ramsey with variable delay**: Measures $T_2^*$
$$P_1(\tau) = \frac{1}{2}(1 + e^{-\tau/T_2^*}\cos(\Delta\omega\tau))$$

**Hahn Echo**: Measures $T_2$ (refocuses low-frequency noise)
$$P_1(\tau) = \frac{1}{2}(1 + e^{-\tau/T_2})$$

**CPMG (Carr-Purcell-Meiboom-Gill)**: Multiple $\pi$ pulses extend coherence
$$P_1(\tau, n) = \frac{1}{2}(1 + e^{-\chi(\tau,n)})$$

The decay function $\chi(\tau, n)$ depends on noise spectrum $S(\omega)$:

$$\chi(\tau, n) = \int_0^\infty \frac{S(\omega)}{\omega^2} |F_n(\omega\tau)|^2 d\omega$$

where $F_n$ is the filter function of the $n$-pulse sequence.

**Noise spectroscopy**: By varying $n$ and $\tau$, extract $S(\omega)$ at different frequencies.

### 7. Dynamical Decoupling

Applying $\pi$ pulses creates a "filter" that suppresses noise at frequencies below $1/\tau$:

**Filter function** for $n$ equally-spaced pulses:
$$|F_n(\omega)|^2 = 8\sin^4\left(\frac{\omega\tau}{4n}\right) \cdot \frac{\sin^2(n\omega\tau/2)}{\cos^2(\omega\tau/2n)}$$

**Effective $T_2$ with DD**:
- For 1/f noise: $T_2^{DD} \sim T_2 \cdot n$
- Practical limit: Gate imperfections accumulate with $n$

### 8. Materials and Fabrication

**Substrate improvements**:
- High-resistivity silicon (>10 kΩ·cm)
- Sapphire (lower TLS density)
- Epitaxial aluminum films

**Surface treatments**:
- Oxygen plasma cleaning
- Hydrofluoric acid dip
- Encapsulation of interfaces

**Junction quality**:
- Controlled oxidation (thermal, anodization)
- Ultra-clean deposition (MBE, high-vacuum evaporation)
- Junction-free designs (fluxonium with granular Al)

**Packaging**:
- 3D cavities for isolation
- On-chip filtering
- Magnetic shielding

### 9. State of the Art (2025)

| Qubit Type | Typical $T_1$ | Best $T_1$ | Typical $T_2$ |
|------------|---------------|------------|---------------|
| Transmon | 100-300 μs | ~500 μs | 50-200 μs |
| Fluxonium | 300-500 μs | >1 ms | 100-300 μs |
| 3D Transmon | 200-500 μs | ~1 ms | 100-300 μs |

**Limiting factors**:
1. Surface TLS (participation ~$10^{-3}$, $\tan\delta \sim 10^{-3}$)
2. Quasiparticle tunneling (~1 ppm QP density)
3. Flux noise (for tunable qubits)
4. Thermal photons (require better filtering)

### 10. Future Directions

**Materials science**:
- Understand TLS microscopic origin
- Engineer TLS-free surfaces
- New superconductors (tantalum, niobium nitride)

**Qubit design**:
- Protected qubits (0-π, bifluxon)
- Bosonic codes (cat qubits in resonators)
- Error-transparent operations

**Architecture**:
- 3D integration for isolation
- Modular approaches
- Better thermal anchoring

## Quantum Computing Applications

### Error Threshold Requirements

For fault-tolerant quantum computing:
$$\epsilon_{gate} < \epsilon_{threshold}$$

For surface code: $\epsilon_{threshold} \approx 1\%$

Gate error from decoherence:
$$\epsilon_{gate} \sim \frac{T_{gate}}{T_1} + \frac{T_{gate}}{T_2}$$

For 50 ns two-qubit gate and 99% fidelity: need $T_1, T_2 > 5$ μs

Current systems have 100x margin on $T_1$, enabling high fidelities.

### Coherence Budget

For a circuit of depth $D$ with $T_{cycle}$ cycle time:
$$\epsilon_{total} \sim D \cdot \frac{T_{cycle}}{T_{coherence}}$$

Example: $D = 100$, $T_{cycle} = 1$ μs, $T_2 = 100$ μs → $\epsilon_{total} \sim 1$

This limits algorithm depth without error correction.

## Worked Examples

### Example 1: Coherence Time Analysis

**Problem**: A transmon has $T_1 = 80$ μs and $T_2 = 50$ μs. Calculate:
(a) The pure dephasing time $T_\phi$
(b) Whether this qubit is $T_1$-limited or dephasing-limited

**Solution**:

(a) From $1/T_2 = 1/(2T_1) + 1/T_\phi$:
$$\frac{1}{T_\phi} = \frac{1}{T_2} - \frac{1}{2T_1} = \frac{1}{50} - \frac{1}{160} = \frac{160 - 50}{50 \times 160} = \frac{110}{8000}$$
$$T_\phi = \frac{8000}{110} = 73 \text{ μs}$$

(b) Compare contributions:
- $T_1$ contribution to $1/T_2$: $1/(2 \times 80) = 0.00625$ μs$^{-1}$
- Pure dephasing contribution: $1/73 = 0.0137$ μs$^{-1}$

Pure dephasing is ~2x larger, so this qubit is **dephasing-limited**.

If we could eliminate pure dephasing: $T_2 \to 2T_1 = 160$ μs.

### Example 2: TLS-Limited $T_1$

**Problem**: A transmon at 5 GHz has surface participation ratio $p_{surf} = 10^{-3}$ and surface loss tangent $\tan\delta_{surf} = 2 \times 10^{-3}$. Calculate the TLS-limited $T_1$.

**Solution**:

$$\frac{1}{T_1^{TLS}} = p_{surf} \cdot \omega_q \cdot \tan\delta_{surf}$$

$$= 10^{-3} \times 2\pi \times 5 \times 10^9 \times 2 \times 10^{-3}$$

$$= 2\pi \times 10^{4} \text{ s}^{-1}$$

$$T_1^{TLS} = \frac{1}{2\pi \times 10^4} = 16 \text{ μs}$$

This sets a limit. To achieve $T_1 > 100$ μs, we need:
$$p_{surf} \cdot \tan\delta < \frac{1}{\omega_q \times 100 \text{ μs}} = \frac{1}{2\pi \times 5 \times 10^9 \times 10^{-4}} = 3 \times 10^{-7}$$

### Example 3: Flux Noise Dephasing

**Problem**: A flux-tunable transmon has frequency sensitivity $\partial\omega_q/\partial\Phi = 2\pi \times 500$ MHz/$\Phi_0$ at its operating point. Flux noise is $S_\Phi^{1/2} = 2$ μ$\Phi_0/\sqrt{\text{Hz}}$ at 1 Hz. Estimate the dephasing rate assuming 1/f noise.

**Solution**:

For 1/f noise with $S_\Phi(\omega) = A/\omega$:

At $f = 1$ Hz: $S_\Phi(2\pi) = A/2\pi = (2 \times 10^{-6}\Phi_0)^2$
So $A = 2\pi \times 4 \times 10^{-12} \Phi_0^2$

The dephasing rate from 1/f noise (integrating with IR cutoff at experiment time $T$):

$$\Gamma_\phi \sim \left(\frac{\partial\omega_q}{\partial\Phi}\right)^2 \cdot A \cdot \ln(T\omega_{IR})$$

For $T = 100$ μs measurement: $\omega_{IR} \sim 1/T = 10^4$ rad/s

$$\Gamma_\phi \sim (2\pi \times 500 \times 10^6)^2 \times 2\pi \times 4 \times 10^{-12} \times \ln(10^4)$$
$$= \pi^3 \times 10^{18} \times 8 \times 10^{-12} \times 9.2$$
$$\approx 2 \times 10^9 \text{ rad/s} \times 10^{-12} \times 9 = 2 \times 10^{-2} \text{ s}^{-1}$$

Wait, let me redo this more carefully:

$$\Gamma_\phi = \left(\frac{\partial\omega_q}{\partial\Phi}\right)^2 S_\Phi$$

At the relevant frequency (~1 kHz for Ramsey):
$$S_\Phi(1 \text{ kHz}) = (2 \times 10^{-6})^2 \times \frac{1 \text{ Hz}}{1 \text{ kHz}} = 4 \times 10^{-15} \Phi_0^2/\text{Hz}$$

$$\Gamma_\phi = (2\pi \times 5 \times 10^8)^2 \times 4 \times 10^{-15}$$
$$= 4\pi^2 \times 2.5 \times 10^{17} \times 4 \times 10^{-15} = 4 \times 10^4 \text{ rad/s}$$

$$T_\phi = 1/\Gamma_\phi \approx 25 \text{ μs}$$

## Practice Problems

### Level 1: Direct Application

1. A qubit has $T_1 = 120$ μs and pure dephasing time $T_\phi = 200$ μs. Calculate $T_2$.

2. If TLS loss tangent is $10^{-3}$ and surface participation is $5 \times 10^{-4}$, what is the TLS-limited $T_1$ for a 6 GHz qubit?

3. A Ramsey experiment shows $T_2^* = 30$ μs, while echo gives $T_2 = 60$ μs. What does this tell you about the noise spectrum?

### Level 2: Intermediate

4. Derive the relationship $T_2 \leq 2T_1$ from the Bloch equations. Under what conditions is the bound saturated?

5. For a qubit with frequency sensitivity $\partial\omega/\partial\Phi = 1$ GHz/$\Phi_0$ and target $T_\phi > 100$ μs, what is the maximum allowable flux noise spectral density at 1 Hz (assuming 1/f)?

6. Design a CPMG sequence (number of pulses, timing) to filter out noise at frequencies below 10 kHz while passing noise above 100 kHz. What is the effective filter function?

### Level 3: Challenging

7. **Quasiparticle analysis**: A transmon shows $T_1 = 50$ μs at base temperature but $T_1 = 200$ μs after waiting 1 hour. The improvement is attributed to quasiparticle trapping.
   (a) Estimate the initial quasiparticle density
   (b) What is the quasiparticle recombination time?
   (c) Propose a method to actively reduce quasiparticles

8. **TLS spectroscopy**: You observe that $T_1$ varies by factor of 2 over a 50 MHz range of qubit frequencies. This suggests coupling to individual TLS. Estimate:
   (a) The number of strongly-coupled TLS
   (b) The TLS-qubit coupling strength
   (c) How would you map out the TLS landscape?

9. **Coherence scaling**: For a surface code with code distance $d$, the logical error rate scales as $(p/p_{th})^{(d+1)/2}$. If physical error rate $p = T_{cycle}/T_2$, derive the relationship between required $T_2$ and code distance for target logical error rate $10^{-15}$.

## Computational Lab: Decoherence Simulation and Noise Analysis

```python
"""
Day 903 Computational Lab: Coherence and Decoherence
Simulating qubit decoherence and noise spectroscopy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.optimize import curve_fit

# =============================================================================
# Part 1: Lindblad Master Equation Simulation
# =============================================================================

def lindblad_evolution(rho0, H, L_ops, gamma_ops, t_points):
    """
    Simulate Lindblad master equation.

    drho/dt = -i[H, rho] + sum_k gamma_k (L_k rho L_k† - 0.5{L_k†L_k, rho})

    Parameters:
    -----------
    rho0 : ndarray
        Initial density matrix
    H : ndarray
        Hamiltonian (in units where hbar = 1)
    L_ops : list of ndarray
        Lindblad operators
    gamma_ops : list of float
        Decay rates for each operator
    t_points : ndarray
        Time points

    Returns:
    --------
    rho_t : list of density matrices
    """
    from scipy.integrate import odeint

    dim = H.shape[0]

    def drho_dt_flat(rho_flat, t):
        rho = rho_flat.reshape(dim, dim)

        # Coherent evolution
        drho = -1j * (H @ rho - rho @ H)

        # Dissipator
        for L, gamma in zip(L_ops, gamma_ops):
            Ld = L.conj().T
            LdL = Ld @ L
            drho += gamma * (L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL))

        return drho.flatten()

    rho0_flat = rho0.flatten()
    solution = odeint(drho_dt_flat, rho0_flat.real, t_points) + \
               1j * odeint(drho_dt_flat, rho0_flat.imag, t_points)
    # Simplified: just use real part for this simulation
    solution = odeint(lambda r, t: drho_dt_flat(r, t).real, rho0_flat.real, t_points)

    rho_t = [sol.reshape(dim, dim) for sol in solution]
    return rho_t

# Define qubit operators
I = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)

print("=" * 60)
print("Lindblad Decoherence Simulation")
print("=" * 60)

# Parameters
T1 = 100e3  # 100 μs in ns
T_phi = 50e3  # 50 μs pure dephasing
gamma_1 = 1/T1
gamma_phi = 1/T_phi

# Calculate T2
T2 = 1 / (1/(2*T1) + 1/T_phi)
print(f"T1 = {T1/1000:.0f} μs")
print(f"T_phi = {T_phi/1000:.0f} μs")
print(f"T2 = {T2/1000:.1f} μs")

# Hamiltonian (in rotating frame, so H = 0 for on-resonance)
H = np.zeros((2, 2), dtype=complex)

# Lindblad operators
L_ops = [sigma_minus, sigma_z]
gamma_ops = [gamma_1, gamma_phi/2]  # Factor of 2 because L = sigma_z contributes 2*gamma to dephasing

# Simulate T1 decay
rho0_excited = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|
t_points = np.linspace(0, 300e3, 300)  # 0 to 300 μs

# Simple exponential decay for T1
P_excited = np.exp(-t_points / T1)

# Simulate Ramsey (T2* decay)
# Initial state: |+⟩ = (|0⟩ + |1⟩)/√2
rho0_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)

# Coherence decay
coherence = np.exp(-t_points / T2)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: T1 decay
ax1 = axes[0, 0]
ax1.plot(t_points/1000, P_excited, 'b-', linewidth=2)
ax1.axhline(1/np.e, color='r', linestyle='--', alpha=0.7, label=f'1/e level')
ax1.axvline(T1/1000, color='g', linestyle='--', alpha=0.7, label=f'T1 = {T1/1000:.0f} μs')
ax1.set_xlabel('Time (μs)', fontsize=12)
ax1.set_ylabel('Excited State Population', fontsize=12)
ax1.set_title('T1 Relaxation', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: T2 decay (Ramsey)
ax2 = axes[0, 1]
# Add oscillation for realistic Ramsey
delta = 0.01  # Small detuning in 1/ns
ramsey_signal = 0.5 * (1 + coherence * np.cos(2*np.pi * delta * t_points))
ax2.plot(t_points/1000, ramsey_signal, 'b-', linewidth=2)
ax2.plot(t_points/1000, 0.5 * (1 + coherence), 'r--', linewidth=1, alpha=0.7, label='Envelope')
ax2.plot(t_points/1000, 0.5 * (1 - coherence), 'r--', linewidth=1, alpha=0.7)
ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Time (μs)', fontsize=12)
ax2.set_ylabel('P(|1⟩)', fontsize=12)
ax2.set_title('Ramsey (T2*) Decay', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# =============================================================================
# Part 2: Noise Spectral Density
# =============================================================================

def one_over_f_noise(f, A):
    """1/f noise spectral density."""
    return A / np.abs(f)

def white_noise(f, S0):
    """White noise spectral density."""
    return S0 * np.ones_like(f)

def lorentzian_noise(f, A, f0, gamma):
    """Lorentzian (TLS) noise spectral density."""
    return A / ((f - f0)**2 + gamma**2)

# Frequency range for noise
f_range = np.logspace(-2, 8, 1000)  # 0.01 Hz to 100 MHz

# Example noise spectra
S_1f = one_over_f_noise(f_range, 1e-6)  # 1/f with A = 1e-6
S_white = white_noise(f_range, 1e-12)
S_tls = lorentzian_noise(f_range, 1e-8, 1e6, 1e5)  # TLS at 1 MHz

ax3 = axes[1, 0]
ax3.loglog(f_range, S_1f, 'b-', linewidth=2, label='1/f noise')
ax3.loglog(f_range, S_white, 'r-', linewidth=2, label='White noise')
ax3.loglog(f_range, S_tls, 'g-', linewidth=2, label='TLS (Lorentzian)')
ax3.set_xlabel('Frequency (Hz)', fontsize=12)
ax3.set_ylabel('Spectral Density (arb. units)', fontsize=12)
ax3.set_title('Noise Spectral Densities', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# =============================================================================
# Part 3: Echo and Dynamical Decoupling
# =============================================================================

def echo_decay(t, T2_echo):
    """Hahn echo decay."""
    return 0.5 * (1 + np.exp(-t/T2_echo))

def cpmg_decay(t, T2_cpmg, n_pulses):
    """CPMG decay with n pulses."""
    # Simplified model: T2 extends with number of pulses
    T2_eff = T2_cpmg * np.sqrt(n_pulses)  # For 1/f^2 noise
    return 0.5 * (1 + np.exp(-t/T2_eff))

T2_star = 30e3  # 30 μs
T2_echo = 60e3  # 60 μs (echo refocuses slow noise)

ramsey_decay = 0.5 * (1 + np.exp(-t_points/T2_star))
echo_signal = echo_decay(t_points, T2_echo)

ax4 = axes[1, 1]
ax4.plot(t_points/1000, ramsey_decay, 'b-', linewidth=2, label=f'Ramsey (T2* = {T2_star/1000:.0f} μs)')
ax4.plot(t_points/1000, echo_signal, 'r-', linewidth=2, label=f'Echo (T2 = {T2_echo/1000:.0f} μs)')

# CPMG with different number of pulses
for n in [4, 16, 64]:
    cpmg_signal = cpmg_decay(t_points, T2_echo, n)
    ax4.plot(t_points/1000, cpmg_signal, '--', linewidth=1.5, label=f'CPMG-{n}')

ax4.set_xlabel('Time (μs)', fontsize=12)
ax4.set_ylabel('Signal', fontsize=12)
ax4.set_title('Dynamical Decoupling', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decoherence_simulation.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 4: Filter Function Analysis
# =============================================================================

print("\n" + "=" * 60)
print("Dynamical Decoupling Filter Functions")
print("=" * 60)

def filter_function_ramsey(omega, tau):
    """Filter function for Ramsey (free induction decay)."""
    return 4 * np.sin(omega * tau / 2)**2 / omega**2

def filter_function_echo(omega, tau):
    """Filter function for Hahn echo."""
    return 16 * np.sin(omega * tau / 4)**4 / omega**2

def filter_function_cpmg(omega, tau, n):
    """Filter function for CPMG with n pulses."""
    if n == 0:
        return filter_function_ramsey(omega, tau)

    delta_t = tau / (2 * n)  # Time between pulses
    result = 8 * np.sin(omega * delta_t / 2)**4

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.sin(n * omega * delta_t / 2)**2 / np.cos(omega * delta_t / 2)**2
        term = np.where(np.isfinite(term), term, 0)

    return result * term / omega**2

tau = 100e-6  # 100 μs total time
omega_range = np.logspace(2, 8, 500)  # 100 Hz to 100 MHz

fig2, ax = plt.subplots(figsize=(10, 6))

F_ramsey = [filter_function_ramsey(w, tau) for w in omega_range]
F_echo = [filter_function_echo(w, tau) for w in omega_range]
F_cpmg_4 = [filter_function_cpmg(w, tau, 4) for w in omega_range]
F_cpmg_16 = [filter_function_cpmg(w, tau, 16) for w in omega_range]

ax.loglog(omega_range/(2*np.pi), F_ramsey, 'b-', linewidth=2, label='Ramsey')
ax.loglog(omega_range/(2*np.pi), F_echo, 'r-', linewidth=2, label='Echo')
ax.loglog(omega_range/(2*np.pi), F_cpmg_4, 'g-', linewidth=2, label='CPMG-4')
ax.loglog(omega_range/(2*np.pi), F_cpmg_16, 'm-', linewidth=2, label='CPMG-16')

ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Filter Function |F(ω)|²/ω²', fontsize=12)
ax.set_title('Dynamical Decoupling Filter Functions', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('filter_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 5: Noise Spectroscopy Simulation
# =============================================================================

print("\n" + "=" * 60)
print("Noise Spectroscopy")
print("=" * 60)

def simulate_noise_spectroscopy(S_noise, omega_range, tau_values, n_pulses_list):
    """
    Simulate decay curves for different pulse sequences.

    Returns decay rates that can be inverted to get noise spectrum.
    """
    decay_rates = {}

    for n in n_pulses_list:
        rates = []
        for tau in tau_values:
            # Integrate S(ω) * F_n(ω,τ)
            integrand = lambda w: S_noise(w) * filter_function_cpmg(w, tau, n)
            rate, _ = quad(integrand, omega_range[0], omega_range[-1])
            rates.append(rate)
        decay_rates[n] = np.array(rates)

    return decay_rates

# Define noise model: 1/f + white noise floor
def total_noise(omega):
    A_1f = 1e4  # 1/f amplitude
    S_white = 1e-2  # White noise floor
    return A_1f / omega + S_white

# Simulate for different pulse numbers
tau_values = np.linspace(10e-6, 200e-6, 20)
n_list = [1, 2, 4, 8, 16, 32]

# Simplified: show decay vs tau for different n
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes3[0]
for n in [0, 1, 4, 16]:
    if n == 0:
        label = 'Ramsey'
        decay = np.exp(-tau_values / (30e-6))  # T2* = 30 μs
    else:
        label = f'CPMG-{n}'
        T2_eff = 30e-6 * np.sqrt(n + 1)  # Extension with pulses
        decay = np.exp(-tau_values / T2_eff)
    ax1.plot(tau_values * 1e6, decay, linewidth=2, label=label)

ax1.set_xlabel('Total Time (μs)', fontsize=12)
ax1.set_ylabel('Coherence', fontsize=12)
ax1.set_title('Coherence vs Time for Different DD Sequences', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Extract noise spectrum from decay rates
ax2 = axes3[1]

# Simplified noise extraction: filter function peaks at ω = π*n/τ
# So different n samples different frequencies
frequencies_sampled = []
amplitudes_sampled = []

for n in [2, 4, 8, 16, 32, 64]:
    # Dominant frequency for CPMG-n at time τ
    tau_sample = 100e-6  # 100 μs
    f_sample = n / (2 * tau_sample)  # Approximate peak frequency
    frequencies_sampled.append(f_sample)
    # Amplitude proportional to inverse decay rate
    T2_n = 30e-6 * np.sqrt(n + 1)
    amplitudes_sampled.append(1 / T2_n)

ax2.loglog(frequencies_sampled, amplitudes_sampled, 'bo-', linewidth=2, markersize=8)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('1/T2 (arb.)', fontsize=12)
ax2.set_title('Extracted Noise Spectrum from CPMG', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_spectroscopy.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 6: Coherence Budget Analysis
# =============================================================================

print("\n" + "=" * 60)
print("Coherence Budget Analysis")
print("=" * 60)

# Typical contributions to 1/T1
contributions_T1 = {
    'TLS (surface)': 1/200e3,  # T1 = 200 μs
    'TLS (bulk)': 1/500e3,
    'Quasiparticles': 1/1000e3,
    'Purcell': 1/500e3,
    'Radiation': 1/2000e3,
}

total_gamma_1 = sum(contributions_T1.values())
T1_total = 1 / total_gamma_1

print(f"T1 contributions:")
for name, gamma in contributions_T1.items():
    T1_contrib = 1/gamma
    print(f"  {name}: T1 = {T1_contrib/1000:.0f} μs ({gamma/total_gamma_1*100:.1f}%)")
print(f"\nTotal T1 = {T1_total/1000:.1f} μs")

# Contributions to 1/T2
contributions_T2 = {
    '1/(2T1)': 1/(2*T1_total),
    'Flux noise': 1/100e3,
    'Charge noise': 1/500e3,
    'Photon shot noise': 1/300e3,
}

total_gamma_2 = sum(contributions_T2.values())
T2_total = 1 / total_gamma_2

print(f"\nT2 contributions:")
for name, gamma in contributions_T2.items():
    T2_contrib = 1/gamma
    print(f"  {name}: T2 = {T2_contrib/1000:.0f} μs ({gamma/total_gamma_2*100:.1f}%)")
print(f"\nTotal T2 = {T2_total/1000:.1f} μs")

# Pie chart of contributions
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes4[0]
labels_T1 = list(contributions_T1.keys())
sizes_T1 = [g/total_gamma_1 * 100 for g in contributions_T1.values()]
ax1.pie(sizes_T1, labels=labels_T1, autopct='%1.1f%%', startangle=90)
ax1.set_title(f'T1 Budget (Total: {T1_total/1000:.0f} μs)', fontsize=14)

ax2 = axes4[1]
labels_T2 = list(contributions_T2.keys())
sizes_T2 = [g/total_gamma_2 * 100 for g in contributions_T2.values()]
ax2.pie(sizes_T2, labels=labels_T2, autopct='%1.1f%%', startangle=90)
ax2.set_title(f'T2 Budget (Total: {T2_total/1000:.0f} μs)', fontsize=14)

plt.tight_layout()
plt.savefig('coherence_budget.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 7: Historical Progress
# =============================================================================

print("\n" + "=" * 60)
print("Historical Coherence Progress")
print("=" * 60)

years = [2000, 2005, 2010, 2015, 2020, 2025]
T1_history = [0.01, 0.5, 5, 50, 150, 500]  # μs
T2_history = [0.005, 0.2, 2, 30, 100, 300]  # μs

fig5, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(years, T1_history, 'bo-', linewidth=2, markersize=10, label='T1')
ax.semilogy(years, T2_history, 'rs-', linewidth=2, markersize=10, label='T2')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coherence Time (μs)', fontsize=12)
ax.set_title('Superconducting Qubit Coherence Progress', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coherence_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate improvement rate
improvement_factor = T1_history[-1] / T1_history[0]
years_span = years[-1] - years[0]
doubling_time = years_span * np.log(2) / np.log(improvement_factor)
print(f"T1 improved by {improvement_factor:.0f}x over {years_span} years")
print(f"Doubling time: ~{doubling_time:.1f} years")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| $T_2$ relation | $1/T_2 = 1/(2T_1) + 1/T_\phi$ |
| Relaxation rate | $\Gamma_1 = |\langle 0|\hat{V}|1\rangle|^2 S(\omega_{01})/\hbar^2$ |
| TLS-limited $T_1$ | $1/T_1^{TLS} = p_{surf} \cdot \omega \cdot \tan\delta$ |
| Flux dephasing | $\Gamma_\phi = (\partial\omega/\partial\Phi)^2 S_\Phi(0)$ |
| Purcell decay | $\Gamma_{Purcell} = (g^2/\Delta^2)\kappa$ |

### Main Takeaways

1. **$T_1$** (energy relaxation) sets fundamental limit on $T_2$: $T_2 \leq 2T_1$

2. **Major noise sources**: TLS defects, flux noise, quasiparticles, thermal photons

3. **Noise spectroscopy** using dynamical decoupling reveals noise spectrum vs frequency

4. **Dynamical decoupling** (Echo, CPMG) extends coherence by filtering low-frequency noise

5. **Materials science** is key: improving surfaces, interfaces, and substrates

6. **State of the art**: $T_1 \sim 500$ μs, $T_2 \sim 300$ μs (2025)

## Weekly Summary

This week covered the essential physics of superconducting qubits:

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 897 | Circuit QED | LC quantization, Jaynes-Cummings, dispersive regime |
| 898 | Transmon Design | $E_J/E_C$ ratio, charge insensitivity, anharmonicity |
| 899 | Flux Qubits | Double-well, sweet spots, fluxonium |
| 900 | Single-Qubit Gates | Rabi oscillations, DRAG pulses, virtual-Z |
| 901 | Two-Qubit Gates | CR, CZ, iSWAP, ZZ coupling |
| 902 | Readout | Dispersive shift, QND, multiplexing |
| 903 | Coherence | $T_1$, $T_2$, noise sources, materials |

## Daily Checklist

- [ ] I understand the difference between $T_1$, $T_2$, and $T_2^*$
- [ ] I can identify major decoherence mechanisms
- [ ] I understand noise spectral density and its effect on coherence
- [ ] I can explain dynamical decoupling and filter functions
- [ ] I understand the role of materials and fabrication
- [ ] I have run the computational lab and interpreted the results
- [ ] I can analyze a coherence budget for a superconducting qubit

## Looking Ahead

Next week explores **trapped ion qubits**:
- Laser cooling and ion trapping
- Motional modes and entangling gates
- Clock states and ultra-long coherence
- Optical and microwave control
- Shuttling and modular architectures

---

*"Coherence is the heartbeat of a quantum computer. Every microsecond we extend it brings us closer to quantum advantage."*
