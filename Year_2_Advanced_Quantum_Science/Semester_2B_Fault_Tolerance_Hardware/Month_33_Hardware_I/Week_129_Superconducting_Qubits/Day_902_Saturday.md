# Day 902: Readout Mechanisms

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Dispersive readout theory, measurement operators |
| Afternoon | 2 hours | QND measurement, fidelity, multiplexing, problem solving |
| Evening | 2 hours | Computational lab: Readout simulation and optimization |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive** the dispersive readout Hamiltonian and resonator response
2. **Explain** quantum non-demolition (QND) measurement requirements
3. **Calculate** readout fidelity from signal-to-noise ratio
4. **Analyze** sources of readout errors and mitigation strategies
5. **Describe** multiplexed readout for multi-qubit systems
6. **Understand** Purcell decay and filtering techniques

## Core Content

### 1. Dispersive Readout Principle

In the dispersive regime ($|\Delta| = |\omega_q - \omega_r| \gg g$), the Jaynes-Cummings Hamiltonian becomes:

$$\hat{H}_{disp} = \hbar\omega_r\hat{a}^\dagger\hat{a} + \frac{\hbar\tilde{\omega}_q}{2}\hat{\sigma}_z + \hbar\chi\hat{a}^\dagger\hat{a}\hat{\sigma}_z$$

The key term $\chi\hat{a}^\dagger\hat{a}\hat{\sigma}_z$ couples the resonator photon number to the qubit state.

**Rewriting**:
$$\hat{H}_{disp} = \hbar(\omega_r + \chi\hat{\sigma}_z)\hat{a}^\dagger\hat{a} + \frac{\hbar\tilde{\omega}_q}{2}\hat{\sigma}_z$$

The resonator frequency depends on qubit state:

$$\boxed{\omega_r^{|g\rangle} = \omega_r - \chi, \quad \omega_r^{|e\rangle} = \omega_r + \chi}$$

### 2. Dispersive Shift

For a transmon coupled to a resonator:

$$\boxed{\chi = \frac{g^2}{\Delta}\cdot\frac{\alpha}{\Delta + \alpha}}$$

where:
- $g$: qubit-resonator coupling
- $\Delta = \omega_q - \omega_r$: detuning
- $\alpha$: transmon anharmonicity (negative)

**Typical values**: $\chi/2\pi \sim 0.5-5$ MHz

The total frequency separation between $|g\rangle$ and $|e\rangle$:
$$\delta\omega_r = 2|\chi|$$

### 3. Resonator Response

When we drive the resonator at frequency $\omega_d$ near $\omega_r$, the steady-state field is:

$$\langle\hat{a}\rangle = \frac{\epsilon}{\kappa/2 + i(\omega_d - \omega_r^{|q\rangle})}$$

where $\epsilon$ is the drive amplitude and $\kappa$ is the resonator decay rate.

The transmitted signal (in terms of S-parameter):
$$S_{21} = 1 - \frac{\kappa_{ext}}{\kappa/2 + i(\omega_d - \omega_r^{|q\rangle})}$$

For qubit in $|g\rangle$ vs $|e\rangle$, the resonator response differs in both amplitude and phase.

### 4. IQ Representation

The resonator output is measured as in-phase (I) and quadrature (Q) components:

$$I = \text{Re}[\langle\hat{a}\rangle], \quad Q = \text{Im}[\langle\hat{a}\rangle]$$

In the IQ plane, qubit states $|g\rangle$ and $|e\rangle$ produce distinct "blobs" (Gaussian distributions due to quantum and thermal noise).

**Signal-to-noise ratio (SNR)**:
$$\text{SNR} = \frac{|\vec{r}_e - \vec{r}_g|}{\sigma}$$

where $\vec{r}_{g,e}$ are the centroids and $\sigma$ is the blob width.

### 5. Quantum Non-Demolition (QND) Measurement

A measurement is QND if:
1. **Repeatability**: Measuring twice gives same result
2. **No back-action on measured observable**: $[\hat{H}_{int}, \hat{\sigma}_z] = 0$

The dispersive interaction $\chi\hat{a}^\dagger\hat{a}\hat{\sigma}_z$ is QND for $\hat{\sigma}_z$:
- Photons in resonator don't flip qubit (to leading order)
- Repeated measurements yield same qubit state

**QND violation sources**:
- Higher-order terms in $g/\Delta$
- Measurement-induced transitions
- $T_1$ decay during readout

### 6. Readout Fidelity

**Assignment fidelity**:
$$F_{assign} = 1 - \frac{P(e|g) + P(g|e)}{2}$$

where $P(e|g)$ is probability of assigning state $|e\rangle$ given the qubit is in $|g\rangle$.

**Sources of infidelity**:
1. **Overlap**: Gaussian blobs overlap in IQ plane
2. **$T_1$ decay**: Qubit decays during measurement
3. **State preparation**: Thermal excitation before measurement
4. **Measurement-induced mixing**: Photon number fluctuations cause transitions

**Optimizing readout**:
- Increase $\chi$ (larger separation)
- Increase photon number $\bar{n}$ (larger signal)
- Decrease $\kappa$ (longer photon lifetime, but slower)
- Use quantum-limited amplifiers

### 7. Photon Number and Measurement Time

The steady-state photon number:
$$\bar{n} = \frac{\epsilon^2}{(\kappa/2)^2 + (\omega_d - \omega_r)^2}$$

On resonance:
$$\bar{n}_{res} = \frac{4\epsilon^2}{\kappa^2}$$

**Critical photon number**: The dispersive approximation breaks down when:
$$\bar{n} \gtrsim n_{crit} = \frac{\Delta^2}{4g^2}$$

Beyond this, higher-order effects cause state mixing.

**Measurement time** for desired SNR:
$$t_{meas} \sim \frac{1}{\kappa}\cdot\left(\frac{\sigma}{\chi\sqrt{\bar{n}}}\right)^2$$

### 8. Readout Errors and Mitigation

**Error types**:

| Error | Cause | Mitigation |
|-------|-------|------------|
| Overlap | Small $\chi$ or $\bar{n}$ | Increase power, optimize $\chi$ |
| $T_1$ decay | Long measurement | Shorter readout, mid-circuit heralding |
| Thermal population | $k_BT \sim \hbar\omega$ | Better cooling, reset protocols |
| Measurement-induced | High $\bar{n}$ | Stay below $n_{crit}$ |
| Purcell decay | Resonator coupling | Purcell filter |

**Readout error mitigation (REM)**:
Measure confusion matrix $M$ where $M_{ij} = P(measure\ i | prepared\ j)$.
Invert to correct measured distributions: $\vec{p}_{corr} = M^{-1}\vec{p}_{meas}$

### 9. Multiplexed Readout

For multi-qubit systems, give each qubit its own readout resonator at different frequency:

$$\omega_r^{(k)} = \omega_r^{(0)} + k\cdot\Delta\omega_{spacing}$$

**Frequency-division multiplexing**:
- Drive all resonators simultaneously with multi-tone pulse
- Digitize output and FFT to separate channels
- Typical spacing: 50-200 MHz

**Advantages**:
- Single output line for many qubits
- Reduced hardware complexity
- Scales well

**Challenges**:
- Crosstalk between tones
- Amplifier bandwidth limits
- Signal processing complexity

### 10. Purcell Effect and Filtering

The qubit can decay through the resonator—**Purcell decay**:

$$\boxed{\Gamma_{Purcell} = \frac{g^2}{\Delta^2}\kappa = \frac{\kappa}{\Delta^2/g^2}}$$

This limits qubit $T_1$ even with perfect materials.

**Purcell filter**: Add notch at qubit frequency in resonator output line
- Bandpass at $\omega_r$ for readout
- Strong attenuation at $\omega_q$
- Implemented as additional resonator or stub

**Enhanced Purcell limit**:
$$T_1^{Purcell} = \frac{\Delta^2}{g^2\kappa}$$

For $\Delta/2\pi = 1$ GHz, $g/2\pi = 100$ MHz, $\kappa/2\pi = 1$ MHz:
$$T_1^{Purcell} = \frac{(10^9)^2}{(10^8)^2 \times 10^6 \times 2\pi} = 160 \text{ }\mu\text{s}$$

## Quantum Computing Applications

### Fast Readout for Error Correction

Surface code requires fast, high-fidelity mid-circuit measurements:
- Measurement time: ~100-500 ns
- Fidelity: >99%
- QND: Minimize back-action for stabilizer measurements

### Heralded State Preparation

Use readout to post-select on ground state:
1. Measure qubit
2. If $|e\rangle$, discard run (or apply $X$ gate)
3. Guarantees starting in $|g\rangle$

### Real-Time Feedback

Modern systems implement real-time classical processing:
1. Measure qubit
2. Process result in ~200 ns
3. Apply conditional gate based on outcome

Enables: Active reset, error correction, adaptive circuits

## Worked Examples

### Example 1: Dispersive Shift Calculation

**Problem**: A transmon at 5 GHz with anharmonicity -250 MHz couples to a resonator at 7 GHz with $g/2\pi = 80$ MHz. Calculate:
(a) The dispersive shift $\chi$
(b) The frequency separation for readout
(c) The critical photon number

**Solution**:

(a) Dispersive shift:
$$\chi = \frac{g^2}{\Delta}\cdot\frac{\alpha}{\Delta + \alpha}$$

With $\Delta = \omega_q - \omega_r = 5 - 7 = -2$ GHz and $\alpha = -0.25$ GHz:

$$\chi = \frac{(0.08)^2}{-2}\cdot\frac{-0.25}{-2 + (-0.25)} = \frac{0.0064}{-2}\cdot\frac{-0.25}{-2.25}$$
$$= -0.0032 \times 0.111 = -0.000356 \text{ GHz} = -0.36 \text{ MHz}$$

So $\chi/2\pi = -0.36$ MHz.

(b) Frequency separation:
$$\delta\omega_r = 2|\chi| = 2 \times 0.36 = 0.72 \text{ MHz}$$

(c) Critical photon number:
$$n_{crit} = \frac{\Delta^2}{4g^2} = \frac{(2000)^2}{4 \times (80)^2} = \frac{4 \times 10^6}{25600} \approx 156 \text{ photons}$$

### Example 2: Readout SNR and Fidelity

**Problem**: A readout resonator has $\kappa/2\pi = 2$ MHz and dispersive shift $\chi/2\pi = 1$ MHz. The qubit $T_1 = 50$ $\mu$s. For readout with $\bar{n} = 10$ photons:
(a) Calculate the integration time needed for SNR = 5
(b) Estimate the readout fidelity including $T_1$ effects

**Solution**:

(a) The signal (separation in IQ plane) is proportional to $2\chi\sqrt{\bar{n}}/\kappa$.

Measurement time for SNR = 5:
$$t_{meas} \approx \frac{1}{2\kappa}\cdot\frac{1}{\eta}\cdot\text{SNR}^2\cdot\frac{1}{\bar{n}(2\chi/\kappa)^2}$$

Simplified estimate: $t_{meas} \sim \kappa^{-1} \times$ (SNR/separation)²

With $2\chi/\kappa = 2 \times 1/2 = 1$ and $\bar{n} = 10$:
$$t_{meas} \sim \frac{25}{10 \times 1 \times 2\pi \times 2 \times 10^6} \approx 200 \text{ ns}$$

(b) $T_1$ decay during readout:
$$P_{decay} = 1 - e^{-t_{meas}/T_1} = 1 - e^{-200 \times 10^{-9}/50 \times 10^{-6}} = 1 - e^{-0.004} \approx 0.4\%$$

Assuming IQ separation gives 99.5% assignment fidelity:
$$F_{total} \approx 0.995 \times (1 - 0.004) \approx 99.1\%$$

### Example 3: Multiplexed Readout

**Problem**: Design a multiplexed readout system for 5 qubits. Requirements:
- Resonator frequencies: 6.0-7.0 GHz
- Minimum separation: 100 MHz
- Dispersive shift: 1 MHz each

**Solution**:

With 5 resonators in 1 GHz band:
$$\Delta\omega_{spacing} = \frac{1000 \text{ MHz}}{5 - 1} = 250 \text{ MHz}$$

This exceeds minimum requirement (100 MHz). Frequencies:
- R1: 6.0 GHz (Q1 at 4.5 GHz, $\Delta$ = -1.5 GHz)
- R2: 6.25 GHz (Q2 at 4.75 GHz)
- R3: 6.5 GHz (Q3 at 5.0 GHz)
- R4: 6.75 GHz (Q4 at 5.25 GHz)
- R5: 7.0 GHz (Q5 at 5.5 GHz)

To achieve $\chi/2\pi = 1$ MHz with $\Delta = 1.5$ GHz and $\alpha = -250$ MHz:
$$1 \text{ MHz} = \frac{g^2}{1500}\cdot\frac{250}{1500 + 250} = \frac{g^2}{1500}\cdot\frac{250}{1750}$$
$$g^2 = 1 \times 1500 \times 7 = 10500 \text{ MHz}^2$$
$$g/2\pi = \sqrt{10500} \approx 102 \text{ MHz}$$

Multi-tone readout pulse: 5 frequencies with 250 MHz spacing, total bandwidth 1 GHz.

## Practice Problems

### Level 1: Direct Application

1. Calculate the dispersive shift for a qubit at 4.8 GHz, resonator at 6.3 GHz, coupling 60 MHz, and anharmonicity -220 MHz.

2. A resonator has $\kappa/2\pi = 5$ MHz. What is the Purcell-limited $T_1$ if $g/2\pi = 100$ MHz and $\Delta/2\pi = 1.5$ GHz?

3. For a readout with 99% assignment fidelity and 0.5% $T_1$ error, what is the total readout fidelity?

### Level 2: Intermediate

4. Derive the steady-state photon number in a driven resonator coupled to a two-level system in the dispersive regime. How does it depend on qubit state?

5. Design a Purcell filter for a system with qubit at 5 GHz and resonator at 7 GHz. Specify the filter center frequency, bandwidth, and attenuation requirements.

6. In multiplexed readout, crosstalk causes resonator $k$ to respond slightly when resonator $j$ is probed. If the crosstalk is -30 dB and the intended signal is -20 dBm, what is the crosstalk-induced error in the measured IQ point?

### Level 3: Challenging

7. **Measurement-induced dephasing**: Calculate the dephasing rate of the qubit due to photon number fluctuations in the resonator. Show that $\Gamma_\phi \propto \chi^2\bar{n}$.

8. **Optimal readout power**: Given $\chi$, $\kappa$, and $n_{crit}$, derive the optimal photon number and measurement time to maximize readout fidelity. Consider both SNR and measurement-induced transitions.

9. **Three-state discrimination**: For a transmon with measurable $|0\rangle$, $|1\rangle$, $|2\rangle$ states, design a readout protocol to distinguish all three. How does this change the IQ plane analysis?

## Computational Lab: Readout Simulation and Optimization

```python
"""
Day 902 Computational Lab: Readout Mechanisms
Simulating dispersive readout and optimizing readout fidelity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import minimize_scalar

# =============================================================================
# Part 1: Resonator Response
# =============================================================================

def resonator_response(omega_d, omega_r, kappa, epsilon):
    """
    Calculate resonator response to drive.

    Returns complex amplitude <a>
    """
    return epsilon / (kappa/2 + 1j * (omega_d - omega_r))

def transmission_S21(omega_d, omega_r, kappa, kappa_ext):
    """
    Transmission S21 parameter.
    """
    return 1 - kappa_ext / (kappa/2 + 1j * (omega_d - omega_r))

print("=" * 60)
print("Dispersive Readout: Resonator Response")
print("=" * 60)

# Parameters (in MHz for convenience)
omega_r_base = 7000  # 7 GHz resonator
chi = 1.0  # 1 MHz dispersive shift
kappa = 2.0  # 2 MHz linewidth
kappa_ext = 1.5  # External coupling
epsilon = 1.0  # Drive amplitude

# Resonator frequencies for |g> and |e>
omega_r_g = omega_r_base - chi
omega_r_e = omega_r_base + chi

# Frequency sweep
omega_d_range = np.linspace(omega_r_base - 10, omega_r_base + 10, 500)

S21_g = [transmission_S21(w, omega_r_g, kappa, kappa_ext) for w in omega_d_range]
S21_e = [transmission_S21(w, omega_r_e, kappa, kappa_ext) for w in omega_d_range]

S21_g = np.array(S21_g)
S21_e = np.array(S21_e)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Transmission magnitude
ax1 = axes[0, 0]
ax1.plot(omega_d_range - omega_r_base, np.abs(S21_g), 'b-', linewidth=2, label=r'|g⟩')
ax1.plot(omega_d_range - omega_r_base, np.abs(S21_e), 'r-', linewidth=2, label=r'|e⟩')
ax1.axvline(-chi, color='b', linestyle='--', alpha=0.5)
ax1.axvline(chi, color='r', linestyle='--', alpha=0.5)
ax1.set_xlabel('Detuning from bare resonator (MHz)', fontsize=12)
ax1.set_ylabel('|S21|', fontsize=12)
ax1.set_title('Resonator Transmission Magnitude', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Transmission phase
ax2 = axes[0, 1]
ax2.plot(omega_d_range - omega_r_base, np.angle(S21_g), 'b-', linewidth=2, label=r'|g⟩')
ax2.plot(omega_d_range - omega_r_base, np.angle(S21_e), 'r-', linewidth=2, label=r'|e⟩')
ax2.set_xlabel('Detuning from bare resonator (MHz)', fontsize=12)
ax2.set_ylabel('Phase (rad)', fontsize=12)
ax2.set_title('Resonator Transmission Phase', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# =============================================================================
# Part 2: IQ Plane Representation
# =============================================================================

# Drive at bare resonator frequency
omega_drive = omega_r_base

a_g = resonator_response(omega_drive, omega_r_g, kappa, epsilon)
a_e = resonator_response(omega_drive, omega_r_e, kappa, epsilon)

print(f"\nDrive at ω_r = {omega_drive} MHz:")
print(f"  |g⟩ response: I = {a_g.real:.4f}, Q = {a_g.imag:.4f}")
print(f"  |e⟩ response: I = {a_e.real:.4f}, Q = {a_e.imag:.4f}")
print(f"  Separation: {np.abs(a_e - a_g):.4f}")

# IQ plane with noise
n_samples = 1000
sigma = 0.1  # Noise standard deviation

I_g = a_g.real + np.random.normal(0, sigma, n_samples)
Q_g = a_g.imag + np.random.normal(0, sigma, n_samples)
I_e = a_e.real + np.random.normal(0, sigma, n_samples)
Q_e = a_e.imag + np.random.normal(0, sigma, n_samples)

ax3 = axes[1, 0]
ax3.scatter(I_g, Q_g, c='blue', alpha=0.3, s=10, label=r'|g⟩')
ax3.scatter(I_e, Q_e, c='red', alpha=0.3, s=10, label=r'|e⟩')
ax3.scatter([a_g.real], [a_g.imag], c='blue', s=200, marker='x', linewidths=3)
ax3.scatter([a_e.real], [a_e.imag], c='red', s=200, marker='x', linewidths=3)

# Draw decision boundary
midpoint = (a_g + a_e) / 2
direction = a_e - a_g
perpendicular = np.array([-direction.imag, direction.real])
perpendicular = perpendicular / np.linalg.norm(perpendicular)
line_points = np.array([midpoint.real - perpendicular[0], midpoint.real + perpendicular[0]])
line_points_q = np.array([midpoint.imag - perpendicular[1], midpoint.imag + perpendicular[1]])
ax3.plot(line_points, line_points_q, 'g-', linewidth=2, label='Threshold')

ax3.set_xlabel('I (arb. units)', fontsize=12)
ax3.set_ylabel('Q (arb. units)', fontsize=12)
ax3.set_title('IQ Plane: Qubit State Discrimination', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# =============================================================================
# Part 3: Readout Fidelity
# =============================================================================

def assignment_fidelity(separation, sigma):
    """
    Calculate assignment fidelity assuming Gaussian blobs.

    separation: distance between |g> and |e> centroids
    sigma: standard deviation of each blob
    """
    # Probability of misassignment = erfc(separation/(2*sqrt(2)*sigma))/2
    # Using erf: P_error = 0.5 * (1 - erf(separation / (2*sqrt(2)*sigma)))
    snr = separation / (np.sqrt(2) * sigma)
    p_error = 0.5 * (1 - erf(snr / 2))
    return 1 - p_error

separation = np.abs(a_e - a_g)
F_assign = assignment_fidelity(separation, sigma)
print(f"\nReadout SNR: {separation/sigma:.2f}")
print(f"Assignment fidelity: {F_assign*100:.2f}%")

# Fidelity vs sigma
sigma_range = np.linspace(0.01, 0.5, 100)
fidelities = [assignment_fidelity(separation, s) for s in sigma_range]

ax4 = axes[1, 1]
ax4.plot(separation / sigma_range, np.array(fidelities) * 100, 'b-', linewidth=2)
ax4.axhline(99.9, color='g', linestyle='--', label='99.9%')
ax4.axhline(99, color='r', linestyle='--', label='99%')
ax4.set_xlabel('SNR (separation/σ)', fontsize=12)
ax4.set_ylabel('Assignment Fidelity (%)', fontsize=12)
ax4.set_title('Fidelity vs Signal-to-Noise Ratio', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 10])
ax4.set_ylim([50, 100])

plt.tight_layout()
plt.savefig('dispersive_readout.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 4: T1 Effects on Readout
# =============================================================================

print("\n" + "=" * 60)
print("T1 Effects on Readout Fidelity")
print("=" * 60)

def total_readout_fidelity(F_assign, t_meas, T1):
    """
    Total fidelity including T1 decay during measurement.

    For |e> state: can decay to |g> during measurement
    """
    P_decay = 1 - np.exp(-t_meas / T1)
    # If we prepared |e> and it decays, we incorrectly measure |g>
    # F_total ≈ F_assign * (1 - P_decay/2) for simple estimate
    return F_assign * (1 - P_decay / 2)

T1 = 100e3  # 100 μs in ns
t_meas_values = np.linspace(100, 2000, 50)  # Measurement times in ns

# For each measurement time, calculate sigma from integration
# sigma ∝ 1/sqrt(t_meas) for averaging
sigma_base = 0.15
sigma_values = sigma_base / np.sqrt(t_meas_values / 500)  # Normalized to 500 ns

F_assign_values = [assignment_fidelity(separation, s) for s in sigma_values]
F_total_values = [total_readout_fidelity(F, t, T1) for F, t in zip(F_assign_values, t_meas_values)]

fig2, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_meas_values, np.array(F_assign_values) * 100, 'b-', linewidth=2, label='SNR-limited')
ax.plot(t_meas_values, np.array(F_total_values) * 100, 'r-', linewidth=2, label='Including T1')

# Find optimal measurement time
idx_opt = np.argmax(F_total_values)
ax.scatter([t_meas_values[idx_opt]], [F_total_values[idx_opt] * 100], c='green', s=100,
           zorder=5, label=f'Optimum: {t_meas_values[idx_opt]:.0f} ns')

ax.set_xlabel('Measurement time (ns)', fontsize=12)
ax.set_ylabel('Readout Fidelity (%)', fontsize=12)
ax.set_title(f'Optimal Readout Time (T1 = {T1/1000:.0f} μs)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('readout_optimization.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Optimal measurement time: {t_meas_values[idx_opt]:.0f} ns")
print(f"Maximum fidelity: {F_total_values[idx_opt]*100:.2f}%")

# =============================================================================
# Part 5: Purcell Effect
# =============================================================================

print("\n" + "=" * 60)
print("Purcell Decay Analysis")
print("=" * 60)

def purcell_T1(g, delta, kappa):
    """Calculate Purcell-limited T1 in ns."""
    gamma_purcell = (g**2 / delta**2) * kappa  # in MHz
    return 1 / gamma_purcell * 1000  # Convert to ns

# Parameters
g = 100  # MHz
delta_values = np.linspace(500, 3000, 50)  # MHz
kappa_values = [1, 2, 5]  # MHz

fig3, ax = plt.subplots(figsize=(8, 5))

for kappa in kappa_values:
    T1_purcell = [purcell_T1(g, d, kappa) / 1000 for d in delta_values]  # Convert to μs
    ax.plot(delta_values, T1_purcell, linewidth=2, label=f'κ = {kappa} MHz')

ax.axhline(100, color='gray', linestyle='--', alpha=0.7, label='100 μs target')
ax.set_xlabel('Qubit-Resonator Detuning (MHz)', fontsize=12)
ax.set_ylabel('Purcell-limited T1 (μs)', fontsize=12)
ax.set_title(f'Purcell Effect (g = {g} MHz)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 500])

plt.tight_layout()
plt.savefig('purcell_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# Example calculation
g_ex = 80
delta_ex = 2000
kappa_ex = 2
T1_purcell_ex = purcell_T1(g_ex, delta_ex, kappa_ex)
print(f"Example: g = {g_ex} MHz, Δ = {delta_ex} MHz, κ = {kappa_ex} MHz")
print(f"Purcell T1 = {T1_purcell_ex/1000:.1f} μs")

# =============================================================================
# Part 6: Multiplexed Readout
# =============================================================================

print("\n" + "=" * 60)
print("Multiplexed Readout Simulation")
print("=" * 60)

n_qubits = 5
freq_spacing = 200  # MHz
resonator_freqs = 6000 + np.arange(n_qubits) * freq_spacing

# Random qubit states
qubit_states = np.random.choice([0, 1], n_qubits)
print(f"Qubit states: {qubit_states}")

# Resonator frequencies shifted by chi based on qubit state
chi_mux = 1.0  # MHz
kappa_mux = 3.0  # MHz

shifted_freqs = resonator_freqs + chi_mux * (2 * qubit_states - 1)

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

# Multi-tone drive
omega_range = np.linspace(5800, 7200, 1000)
total_response = np.zeros(len(omega_range), dtype=complex)

for i, (omega_r, state) in enumerate(zip(shifted_freqs, qubit_states)):
    response = np.array([resonator_response(w, omega_r, kappa_mux, epsilon)
                         for w in omega_range])
    total_response += response

ax1 = axes4[0]
for i in range(n_qubits):
    color = 'red' if qubit_states[i] == 1 else 'blue'
    ax1.axvline(shifted_freqs[i], color=color, linestyle='--', alpha=0.5)
    ax1.axvline(resonator_freqs[i], color='gray', linestyle=':', alpha=0.3)

ax1.plot(omega_range, np.abs(total_response), 'k-', linewidth=2)
ax1.set_xlabel('Frequency (MHz)', fontsize=12)
ax1.set_ylabel('Response magnitude', fontsize=12)
ax1.set_title('Multiplexed Readout Spectrum', fontsize=14)
ax1.grid(True, alpha=0.3)

# IQ points for each qubit
ax2 = axes4[1]
colors = ['blue', 'red']
labels = [r'|g⟩', r'|e⟩']

for i in range(n_qubits):
    # IQ point for this qubit
    a = resonator_response(resonator_freqs[i], shifted_freqs[i], kappa_mux, epsilon)
    I_meas = a.real + np.random.normal(0, 0.05, 100)
    Q_meas = a.imag + np.random.normal(0, 0.05, 100)
    ax2.scatter(I_meas, Q_meas, c=colors[qubit_states[i]], alpha=0.5, s=10)
    ax2.annotate(f'Q{i}', (a.real, a.imag), fontsize=12)

ax2.set_xlabel('I', fontsize=12)
ax2.set_ylabel('Q', fontsize=12)
ax2.set_title('IQ Points for 5-Qubit Readout', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiplexed_readout.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 7: Readout Error Mitigation
# =============================================================================

print("\n" + "=" * 60)
print("Readout Error Mitigation")
print("=" * 60)

# Confusion matrix
P_g_given_g = 0.97  # Correctly measure |g> when prepared in |g>
P_e_given_e = 0.95  # Correctly measure |e> when prepared in |e>

M = np.array([
    [P_g_given_g, 1 - P_e_given_e],
    [1 - P_g_given_g, P_e_given_e]
])

print("Confusion matrix M:")
print(M)

M_inv = np.linalg.inv(M)
print("\nInverse confusion matrix M^-1:")
print(M_inv)

# Example: measured distribution
p_measured = np.array([0.4, 0.6])  # Measured 40% |g>, 60% |e>

# True distribution
p_corrected = M_inv @ p_measured

print(f"\nMeasured distribution: P(g) = {p_measured[0]:.3f}, P(e) = {p_measured[1]:.3f}")
print(f"Corrected distribution: P(g) = {p_corrected[0]:.3f}, P(e) = {p_corrected[1]:.3f}")

# Assignment fidelity
F_assignment = (P_g_given_g + P_e_given_e) / 2
print(f"\nAssignment fidelity: {F_assignment*100:.1f}%")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Dispersive Hamiltonian | $\hat{H} = \hbar(\omega_r + \chi\hat{\sigma}_z)\hat{a}^\dagger\hat{a}$ |
| Dispersive shift | $\chi = (g^2/\Delta) \cdot (\alpha/(\Delta + \alpha))$ |
| Resonator response | $\langle a \rangle = \epsilon/(\kappa/2 + i(\omega_d - \omega_r))$ |
| Critical photon number | $n_{crit} = \Delta^2/4g^2$ |
| Purcell decay | $\Gamma_{Purcell} = (g^2/\Delta^2)\kappa$ |

### Main Takeaways

1. **Dispersive readout**: Qubit state shifts resonator frequency by $2\chi$

2. **IQ discrimination**: Measure I and Q quadratures; qubit states appear as separated blobs

3. **QND measurement**: Dispersive interaction preserves $\hat{\sigma}_z$ to leading order

4. **Readout fidelity** limited by: overlap (SNR), $T_1$ decay, thermal population

5. **Multiplexed readout**: Different resonator frequencies enable single-line readout of many qubits

6. **Purcell filter**: Protects qubit from decay through resonator

## Daily Checklist

- [ ] I can derive the dispersive shift and resonator frequency shift
- [ ] I understand the IQ plane representation of readout
- [ ] I can explain QND measurement requirements
- [ ] I can calculate readout fidelity from SNR and other parameters
- [ ] I understand multiplexed readout implementation
- [ ] I can analyze Purcell effect and mitigation
- [ ] I have run the computational lab and can interpret the results

## Preview: Day 903

Tomorrow we conclude the week with **coherence and decoherence**:

- $T_1$, $T_2$, and $T_2^*$ definitions and measurement
- Decoherence mechanisms: TLS, flux noise, photon shot noise
- Noise spectroscopy techniques
- Materials and fabrication improvements
- Path to longer coherence times

---

*"Readout is where the quantum meets the classical—we must extract information without destroying it."*
