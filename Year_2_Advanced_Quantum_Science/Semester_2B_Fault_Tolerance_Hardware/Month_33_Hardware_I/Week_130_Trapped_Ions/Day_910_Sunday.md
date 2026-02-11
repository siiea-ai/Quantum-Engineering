# Day 910: Trapped Ion Error Sources and Benchmarking

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Error mechanisms, noise sources, decoherence |
| Afternoon | 2 hours | Problem solving: error budgets and mitigation |
| Evening | 2 hours | Computational lab: benchmarking simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Identify and quantify** major error sources in trapped ion systems
2. **Calculate motional heating rates** from electric field noise
3. **Analyze laser noise contributions** to gate infidelity
4. **Evaluate crosstalk effects** between addressed qubits
5. **Implement randomized benchmarking** protocols
6. **Interpret gate set tomography** results

## Core Content

### 1. Overview of Error Sources

Trapped ion quantum computers achieve world-leading gate fidelities, but multiple error mechanisms still limit performance:

| Error Source | Typical Contribution | Affected Operations |
|--------------|---------------------|---------------------|
| Motional heating | 0.01-0.1% | Two-qubit gates |
| Laser intensity noise | 0.01-0.1% | All gates |
| Laser phase noise | 0.001-0.01% | Raman gates |
| Qubit dephasing | 0.001-0.01% | Long circuits |
| Crosstalk | 0.01-0.1% | Individual addressing |
| Spontaneous emission | 0.01-0.1% | Raman gates |
| State prep/measurement | 0.1-1% | SPAM |

### 2. Motional Heating

Motional heating is often the dominant error source for two-qubit gates.

#### Physical Origin

Electric field noise at the ion position causes force fluctuations:

$$\hat{H}_{noise} = q E_{noise}(t) \hat{z}$$

This drives transitions between motional states.

#### Heating Rate

$$\boxed{\dot{\bar{n}} = \frac{q^2}{4m\hbar\omega} S_E(\omega)}$$

where $S_E(\omega)$ is the electric field noise spectral density at the trap frequency.

#### Anomalous Heating

Experimental observations:

$$S_E(\omega) \propto d^{-\alpha} \omega^{-\beta}$$

with:
- Distance scaling: $\alpha \approx 4$ (surface noise)
- Frequency scaling: $\beta \approx 0.5-2$ (varies by system)

**Typical values:**

| Trap type | Ion-electrode distance | $\dot{\bar{n}}$ at 1 MHz |
|-----------|----------------------|-------------------------|
| Macroscopic | 500 μm | 1-10 quanta/s |
| Microfabricated | 50 μm | 100-10,000 quanta/s |
| Cryogenic surface | 50 μm | 1-100 quanta/s |

#### Heating Rate Measurement

Protocol:
1. Cool to ground state ($\bar{n} \approx 0$)
2. Wait time $t_{wait}$
3. Measure $\bar{n}$ via sideband comparison

$$\bar{n}(t) = \bar{n}_0 + \dot{\bar{n}} \cdot t$$

Sideband ratio method:
$$\frac{P_{BSB}}{P_{RSB}} = \frac{\bar{n} + 1}{\bar{n}}$$

### 3. Laser Noise

#### Intensity Noise

For Raman transitions with $\Omega_{eff} = \Omega_1\Omega_2/(2\Delta)$:

$$\frac{\delta\Omega}{\Omega} = \frac{\delta I_1}{I_1} + \frac{\delta I_2}{I_2}$$

**Gate error from intensity noise:**
$$\epsilon_I \approx \left(\frac{\pi \delta\Omega/\Omega}{\sqrt{2}}\right)^2$$

For 1% intensity fluctuation:
$$\epsilon_I \approx (\pi \times 0.01)^2 / 2 \approx 5 \times 10^{-4}$$

#### Phase Noise

Laser phase noise causes dephasing, characterized by the linewidth $\Delta\nu$.

**Error contribution:**
$$\epsilon_\phi \approx (2\pi\Delta\nu \cdot t_{gate})^2$$

For 1 kHz linewidth and 10 μs gate:
$$\epsilon_\phi \approx (2\pi \times 10^3 \times 10^{-5})^2 \approx 4 \times 10^{-3}$$

#### Beam Pointing Fluctuations

Position noise causes both intensity and phase variations:

$$\frac{\delta\Omega}{\Omega} \approx \frac{\delta x}{w_0}$$

where $w_0$ is the beam waist.

### 4. Qubit Dephasing

#### $T_2$ Mechanisms

For hyperfine qubits on clock transitions:

$$\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$$

**Main dephasing sources:**
- Magnetic field fluctuations (second-order for clock)
- AC Stark shifts from stray light
- Motional dephasing

#### Error per Gate

For a gate of duration $t_g$:
$$\epsilon_{T_2} \approx \frac{t_g}{T_2}$$

With $T_2 = 1$ s and $t_g = 100$ μs:
$$\epsilon_{T_2} \approx 10^{-4}$$

### 5. Crosstalk

#### Optical Crosstalk

When addressing ion $j$, neighboring ions experience residual light:

$$\Omega_k = \Omega_j \cdot \exp\left(-\frac{(z_k - z_j)^2}{w_0^2}\right)$$

**Crosstalk ratio:**
$$\chi = \frac{\Omega_{neighbor}}{\Omega_{target}}$$

For typical parameters ($d = 5$ μm, $w_0 = 1$ μm):
$$\chi \approx \exp(-25) \approx 10^{-11}$$

But with realistic aberrations: $\chi \approx 10^{-2}$ to $10^{-3}$

#### Spectral Crosstalk

If ions have slightly different frequencies (from field gradients):

$$\delta\omega = \omega_j - \omega_k$$

Off-resonant driving causes:
$$\epsilon_{spec} \approx \left(\frac{\Omega}{\delta\omega}\right)^2$$

#### Motional Crosstalk

Gate on ion pair (1,2) affects ion 3 through shared modes:

$$\epsilon_{mode} \propto \eta_3^2 \cdot \left(\frac{\Omega}{\omega_3 - \omega_{target}}\right)^2$$

### 6. State Preparation and Measurement (SPAM)

#### State Preparation Errors

- Optical pumping infidelity: ~$10^{-4}$
- Residual motion after cooling: ~$10^{-2}$
- Magnetic field instability: ~$10^{-4}$

**Total preparation error:** $\epsilon_{prep} \approx 10^{-3}$

#### Measurement Errors

**Detection infidelity sources:**
- Finite photon collection efficiency
- Background counts
- Off-resonant scattering
- State decay during detection

For electron-shelving detection:

$$F_{readout} = 1 - P(0|1) - P(1|0)$$

Typical: $F_{readout} > 99.9\%$

### 7. Error Budgets

A complete error budget combines all sources:

$$\epsilon_{total} = \sum_i \epsilon_i$$

**Example: Two-Qubit Gate Error Budget**

| Source | Contribution | Notes |
|--------|--------------|-------|
| Motional heating | $2 \times 10^{-4}$ | $\dot{\bar{n}} = 50$/s, $t_g = 100$ μs |
| Laser intensity | $1 \times 10^{-4}$ | 0.5% RMS fluctuation |
| Laser phase | $0.5 \times 10^{-4}$ | 100 Hz linewidth |
| Off-resonant carrier | $1 \times 10^{-4}$ | $\Omega/\omega = 0.1$ |
| Spectator modes | $0.5 \times 10^{-4}$ | Pulse shaping |
| Spontaneous emission | $1 \times 10^{-4}$ | $\Delta = 1$ THz |
| **Total** | **$6 \times 10^{-4}$** | **99.94% fidelity** |

### 8. Randomized Benchmarking

Randomized benchmarking (RB) provides a SPAM-free measure of gate fidelity.

#### Standard RB Protocol

1. Prepare initial state $|0\rangle$
2. Apply sequence of $m$ random Clifford gates $C_1, C_2, ..., C_m$
3. Apply recovery Clifford $C_{m+1} = (C_m \cdot ... \cdot C_1)^{-1}$
4. Measure return probability to $|0\rangle$
5. Average over many random sequences
6. Repeat for different sequence lengths $m$

#### Decay Model

$$\boxed{P(m) = A \cdot p^m + B}$$

where:
- $p$ = depolarizing parameter
- $A, B$ = SPAM-related constants

**Average gate fidelity:**
$$\boxed{F_{avg} = 1 - \frac{(d-1)(1-p)}{d}}$$

For qubits ($d = 2$):
$$F_{avg} = \frac{1 + p}{2}$$

#### Interleaved RB

To characterize a specific gate $G$:

1. Run standard RB → get $p_{ref}$
2. Run RB with $G$ interleaved after each Clifford → get $p_G$

**Gate fidelity:**
$$F_G = 1 - \frac{(d-1)(1 - p_G/p_{ref})}{d}$$

### 9. Gate Set Tomography (GST)

GST provides detailed characterization beyond RB.

#### What GST Measures

- Complete gate process matrices
- SPAM errors separately
- Systematic vs. stochastic errors
- Coherent error rotations

#### GST Protocol

1. Define fiducial sequences for state preparation/measurement
2. Apply germs (short gate sequences) raised to various powers
3. Measure outcomes for all combinations
4. Maximum likelihood estimation of gate set

#### Interpreting GST Results

GST outputs:
- **Process matrices** for each gate
- **Diamond norm distance** from ideal
- **Error generators** (Hamiltonian vs. stochastic)
- **Gauge-invariant metrics**

### 10. Error Mitigation Strategies

#### Hardware Improvements

| Error | Mitigation |
|-------|------------|
| Heating | Cryogenics, electrode treatment |
| Laser noise | Intensity locking, optical filtering |
| Crosstalk | Tighter focusing, composite pulses |
| Dephasing | Better shielding, dynamical decoupling |

#### Software Techniques

- **Zero-noise extrapolation:** Run at different noise levels
- **Probabilistic error cancellation:** Invert noise channel
- **Dynamical decoupling:** Refocus slow noise
- **Composite pulses:** Cancel systematic errors

## Quantum Computing Applications

### Threshold Comparison

Fault tolerance requires error rate below threshold:

| Code | Threshold | Trapped Ion Status |
|------|-----------|-------------------|
| Surface code | ~1% | Met (99.5%+) |
| Color code | ~0.7% | Met |
| Steane code | ~$10^{-4}$ | Approaching |

### Current State-of-the-Art (2024-2025)

| Metric | Best Reported | Group |
|--------|---------------|-------|
| 1Q gate fidelity | 99.9999% | Oxford |
| 2Q gate fidelity | 99.914% | Quantinuum |
| SPAM fidelity | 99.99% | Multiple |
| T2 (hyperfine) | 10 minutes | NIST |

## Worked Examples

### Example 1: Heating-Limited Fidelity

**Problem:** Calculate the two-qubit gate error from motional heating with $\dot{\bar{n}} = 200$ quanta/s and gate time 80 μs.

**Solution:**

Phonons added during gate:
$$\Delta\bar{n} = \dot{\bar{n}} \times t_g = 200 \times 80 \times 10^{-6} = 0.016$$

For MS gate, error scales with added phonons:
$$\epsilon_{heat} \approx \Delta\bar{n} \times f$$

where $f$ depends on detuning; typically $f \approx 1$ for well-designed gates.

$$\boxed{\epsilon_{heat} \approx 1.6 \times 10^{-2} = 1.6\%}$$

This is too high! Need faster gates or lower heating rate.

**Mitigation:** Reduce gate time to 20 μs:
$$\epsilon_{heat} = 200 \times 20 \times 10^{-6} = 0.4\%$$

### Example 2: Crosstalk Analysis

**Problem:** Two ions separated by 5 μm are addressed with a beam of waist $w_0 = 1.2$ μm. Calculate the optical crosstalk.

**Solution:**

Gaussian intensity profile:
$$I(r) = I_0 \exp\left(-\frac{2r^2}{w_0^2}\right)$$

At distance $d = 5$ μm:
$$\chi = \exp\left(-\frac{2 \times 5^2}{1.2^2}\right) = \exp(-34.7)$$

$$\chi \approx 10^{-15}$$

This negligible crosstalk assumes perfect Gaussian beams.

**With aberrations** (typical $\chi \sim 10^{-3}$):

Gate error on neighbor:
$$\epsilon_{XT} \approx \left(\frac{\chi \times \Omega \times t}{\pi}\right)^2$$

For $\chi = 10^{-3}$, this gives:
$$\boxed{\epsilon_{XT} \approx 10^{-6}}$$

Acceptable for high-fidelity gates.

### Example 3: RB Decay Analysis

**Problem:** Randomized benchmarking data shows $p = 0.9985$ per Clifford. Calculate the average gate fidelity assuming each Clifford uses 1.5 physical gates on average.

**Solution:**

From RB decay:
$$F_{Clifford} = \frac{1 + p}{2} = \frac{1 + 0.9985}{2} = 0.99925$$

Error per Clifford:
$$\epsilon_{Clifford} = 1 - F_{Clifford} = 7.5 \times 10^{-4}$$

If each Clifford uses 1.5 gates:
$$\epsilon_{gate} = \frac{\epsilon_{Clifford}}{1.5} = 5 \times 10^{-4}$$

$$\boxed{F_{gate} = 99.95\%}$$

## Practice Problems

### Level 1: Direct Application

1. Convert a heating rate of 50 quanta/s to electric field noise spectral density for $^{171}$Yb$^+$ at 1 MHz trap frequency.

2. Calculate the laser intensity stabilization requirement to achieve gate error < $10^{-4}$ from intensity noise.

3. If RB gives $p = 0.998$ per Clifford and SPAM error is 0.5%, what is the SPAM-corrected gate fidelity?

### Level 2: Intermediate

4. Design a two-qubit gate with total error < 0.1%, given: heating rate 100/s, 1% intensity noise, 500 Hz laser linewidth. Specify gate time and detuning.

5. Analyze the error budget for a 10-qubit circuit with 50 two-qubit gates. Which error source dominates?

6. Compare standard RB and interleaved RB. If $p_{ref} = 0.999$ and $p_G = 0.998$, what is the gate fidelity?

### Level 3: Challenging

7. Derive the relationship between electric field noise spectral density and heating rate from first principles.

8. Design a crosstalk-mitigation pulse sequence that cancels leakage to neighboring ions to first order.

9. Analyze GST data to extract coherent vs. stochastic error contributions. What does each tell you about the physical error sources?

## Computational Lab: Benchmarking Simulation

```python
"""
Day 910 Computational Lab: Error Characterization and Benchmarking
Simulating randomized benchmarking and error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import expm

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


class NoisyGate:
    """Model for noisy quantum gates"""

    def __init__(self, depolarizing_rate=0.001, dephasing_rate=0.0005,
                 rotation_error=0.0, amplitude_error=0.0):
        """
        Parameters:
        -----------
        depolarizing_rate : float - Depolarizing error per gate
        dephasing_rate : float - Dephasing error per gate
        rotation_error : float - Systematic over/under rotation
        amplitude_error : float - RMS amplitude fluctuation
        """
        self.p_depol = depolarizing_rate
        self.p_dephase = dephasing_rate
        self.rot_err = rotation_error
        self.amp_err = amplitude_error

    def apply_gate(self, rho, U_ideal):
        """Apply noisy gate to density matrix"""
        # Amplitude noise
        if self.amp_err > 0:
            amp_factor = 1 + np.random.normal(0, self.amp_err)
            # Modify gate angle
            # This is simplified - real implementation would decompose U
            pass

        # Apply ideal gate
        rho_out = U_ideal @ rho @ U_ideal.conj().T

        # Depolarizing channel
        if self.p_depol > 0:
            rho_out = (1 - self.p_depol) * rho_out + self.p_depol * I / 2

        # Dephasing channel
        if self.p_dephase > 0:
            rho_out = (1 - self.p_dephase) * rho_out + self.p_dephase * Z @ rho_out @ Z

        return rho_out


def rotation_gate(axis, angle):
    """Create rotation gate"""
    if axis == 'x':
        return expm(-1j * angle/2 * X)
    elif axis == 'y':
        return expm(-1j * angle/2 * Y)
    elif axis == 'z':
        return expm(-1j * angle/2 * Z)
    else:
        raise ValueError("Invalid axis")


def generate_clifford():
    """Generate random single-qubit Clifford gate"""
    # Single-qubit Cliffords: 24 elements
    # Simplified: rotation by ±π/2, π about x, y, z
    clifford_generators = [
        ('x', np.pi/2), ('x', -np.pi/2), ('x', np.pi),
        ('y', np.pi/2), ('y', -np.pi/2), ('y', np.pi),
        ('z', np.pi/2), ('z', -np.pi/2), ('z', np.pi),
        ('identity', 0)
    ]

    idx = np.random.randint(len(clifford_generators))
    axis, angle = clifford_generators[idx]

    if axis == 'identity':
        return I, 'I'
    else:
        return rotation_gate(axis, angle), f'{axis.upper()}{int(angle*180/np.pi)}'


def compute_inverse_clifford(clifford_sequence):
    """Compute inverse of Clifford sequence"""
    total_U = I.copy()
    for U, _ in clifford_sequence:
        total_U = U @ total_U

    # Inverse
    return total_U.conj().T


def randomized_benchmarking(noisy_gate, sequence_lengths, n_sequences=50):
    """
    Perform randomized benchmarking

    Parameters:
    -----------
    noisy_gate : NoisyGate - Noise model
    sequence_lengths : array - Sequence lengths to test
    n_sequences : int - Number of random sequences per length

    Returns:
    --------
    lengths, survival_probs, errors
    """
    survival_probs = []
    errors = []

    for m in sequence_lengths:
        survivals = []

        for _ in range(n_sequences):
            # Generate random Clifford sequence
            sequence = [generate_clifford() for _ in range(m)]

            # Compute recovery Clifford
            U_inverse = compute_inverse_clifford(sequence)

            # Initialize in |0⟩
            rho = np.array([[1, 0], [0, 0]], dtype=complex)

            # Apply sequence
            for U, _ in sequence:
                rho = noisy_gate.apply_gate(rho, U)

            # Apply recovery
            rho = noisy_gate.apply_gate(rho, U_inverse)

            # Measure survival probability
            P_0 = np.real(rho[0, 0])
            survivals.append(P_0)

        survival_probs.append(np.mean(survivals))
        errors.append(np.std(survivals) / np.sqrt(n_sequences))

    return sequence_lengths, np.array(survival_probs), np.array(errors)


def fit_rb_decay(lengths, probs):
    """Fit RB decay curve"""
    def decay_model(m, A, p, B):
        return A * p**m + B

    try:
        popt, pcov = curve_fit(decay_model, lengths, probs,
                               p0=[0.5, 0.99, 0.5],
                               bounds=([0, 0, 0], [1, 1, 1]))
        return popt, np.sqrt(np.diag(pcov))
    except:
        return [0.5, 0.99, 0.5], [0, 0, 0]


def interleaved_rb(noisy_gate, target_gate, sequence_lengths, n_sequences=50):
    """Interleaved randomized benchmarking for specific gate"""
    survival_probs = []
    errors = []

    for m in sequence_lengths:
        survivals = []

        for _ in range(n_sequences):
            # Generate sequence with interleaved target gate
            sequence = []
            for _ in range(m):
                sequence.append(generate_clifford())
                sequence.append((target_gate, 'G'))

            # Compute recovery Clifford
            U_inverse = compute_inverse_clifford(sequence)

            # Initialize
            rho = np.array([[1, 0], [0, 0]], dtype=complex)

            # Apply sequence
            for U, _ in sequence:
                rho = noisy_gate.apply_gate(rho, U)

            # Apply recovery
            rho = noisy_gate.apply_gate(rho, U_inverse)

            # Measure
            P_0 = np.real(rho[0, 0])
            survivals.append(P_0)

        survival_probs.append(np.mean(survivals))
        errors.append(np.std(survivals) / np.sqrt(n_sequences))

    return sequence_lengths, np.array(survival_probs), np.array(errors)


def plot_rb_results():
    """Plot randomized benchmarking results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Different error rates
    ax1 = axes[0, 0]

    error_rates = [0.0005, 0.001, 0.002, 0.005]
    colors = ['blue', 'green', 'orange', 'red']

    seq_lengths = np.array([1, 2, 5, 10, 20, 50, 100, 200])

    for eps, color in zip(error_rates, colors):
        noisy = NoisyGate(depolarizing_rate=eps)
        lengths, probs, errs = randomized_benchmarking(noisy, seq_lengths, n_sequences=100)

        ax1.errorbar(lengths, probs, yerr=errs, fmt='o-', color=color,
                    label=f'ε = {eps*100:.2f}%', capsize=3)

        # Fit decay
        popt, _ = fit_rb_decay(lengths, probs)
        fit_lengths = np.linspace(1, 200, 100)
        fit_probs = popt[0] * popt[1]**fit_lengths + popt[2]
        ax1.plot(fit_lengths, fit_probs, '--', color=color, alpha=0.5)

    ax1.set_xlabel('Sequence length m', fontsize=12)
    ax1.set_ylabel('Survival probability', fontsize=12)
    ax1.set_title('Randomized Benchmarking', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Extract fidelity vs error rate
    ax2 = axes[0, 1]

    test_rates = np.logspace(-4, -2, 20)
    measured_fidelities = []

    for eps in test_rates:
        noisy = NoisyGate(depolarizing_rate=eps)
        lengths, probs, _ = randomized_benchmarking(noisy, seq_lengths, n_sequences=50)
        popt, _ = fit_rb_decay(lengths, probs)
        p = popt[1]
        F = (1 + p) / 2
        measured_fidelities.append(1 - F)

    ax2.loglog(test_rates, measured_fidelities, 'bo-', markersize=5)
    ax2.loglog(test_rates, test_rates, 'r--', label='Ideal (1:1)')
    ax2.set_xlabel('True error rate', fontsize=12)
    ax2.set_ylabel('Measured infidelity', fontsize=12)
    ax2.set_title('RB Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Interleaved RB comparison
    ax3 = axes[1, 0]

    noisy = NoisyGate(depolarizing_rate=0.001)
    target = rotation_gate('x', np.pi)  # X gate

    # Standard RB
    lengths, probs_ref, errs_ref = randomized_benchmarking(noisy, seq_lengths[:6], n_sequences=100)
    ax3.errorbar(lengths, probs_ref, yerr=errs_ref, fmt='bo-', label='Standard RB', capsize=3)

    # Interleaved RB (noisier X gate)
    noisy_x = NoisyGate(depolarizing_rate=0.002)  # Extra noise on X
    lengths, probs_int, errs_int = interleaved_rb(noisy_x, target, lengths, n_sequences=100)
    ax3.errorbar(lengths, probs_int, yerr=errs_int, fmt='rs-', label='Interleaved RB (X)', capsize=3)

    ax3.set_xlabel('Sequence length m', fontsize=12)
    ax3.set_ylabel('Survival probability', fontsize=12)
    ax3.set_title('Standard vs Interleaved RB', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Error budget breakdown
    ax4 = axes[1, 1]

    error_sources = ['Heating', 'Laser\nintensity', 'Laser\nphase', 'Crosstalk',
                    'Spont.\nemission', 'Dephasing']
    contributions = [2e-4, 1e-4, 0.5e-4, 1e-4, 1e-4, 0.5e-4]

    colors = plt.cm.Set3(np.linspace(0, 1, len(error_sources)))
    bars = ax4.bar(error_sources, [c * 100 for c in contributions], color=colors)

    # Add total line
    total = sum(contributions)
    ax4.axhline(y=total * 100, color='red', linestyle='--',
                label=f'Total: {total*100:.3f}%')

    ax4.set_ylabel('Error contribution (%)', fontsize=12)
    ax4.set_title('Error Budget (Two-Qubit Gate)', fontsize=14)
    ax4.legend()

    # Add value labels
    for bar, val in zip(bars, contributions):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val*1e4:.1f}×10⁻⁴', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('rb_results.png', dpi=150)
    plt.show()


def plot_error_analysis():
    """Analyze different error mechanisms"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Heating rate analysis
    ax1 = axes[0, 0]

    heating_rates = np.logspace(0, 4, 50)  # 1 to 10000 quanta/s
    gate_times = [10e-6, 50e-6, 100e-6, 200e-6]
    colors = ['blue', 'green', 'orange', 'red']

    for t_g, color in zip(gate_times, colors):
        error = heating_rates * t_g
        ax1.loglog(heating_rates, error * 100, color=color,
                  label=f't_gate = {t_g*1e6:.0f} μs', linewidth=2)

    ax1.axhline(y=0.1, color='gray', linestyle='--', label='0.1% threshold')
    ax1.set_xlabel('Heating rate (quanta/s)', fontsize=12)
    ax1.set_ylabel('Gate error (%)', fontsize=12)
    ax1.set_title('Heating-Limited Gate Error', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Laser noise analysis
    ax2 = axes[0, 1]

    intensity_noise = np.linspace(0.001, 0.05, 100)  # 0.1% to 5%
    gate_error_I = (np.pi * intensity_noise)**2 / 2

    ax2.semilogy(intensity_noise * 100, gate_error_I * 100, 'b-',
                label='Intensity noise', linewidth=2)

    phase_noise = np.linspace(10, 10000, 100)  # Hz linewidth
    t_gate = 100e-6
    gate_error_P = (2 * np.pi * phase_noise * t_gate)**2

    ax2_twin = ax2.twiny()
    ax2_twin.semilogy(phase_noise, gate_error_P * 100, 'r--',
                     label='Phase noise (100 μs gate)', linewidth=2)

    ax2.set_xlabel('Intensity noise (%)', fontsize=12, color='blue')
    ax2_twin.set_xlabel('Phase noise linewidth (Hz)', fontsize=12, color='red')
    ax2.set_ylabel('Gate error (%)', fontsize=12)
    ax2.set_title('Laser Noise Contributions', fontsize=14)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    ax2.grid(True, alpha=0.3)

    # Crosstalk analysis
    ax3 = axes[1, 0]

    beam_waist = np.linspace(0.5, 3, 50)  # μm
    ion_separation = 5  # μm

    crosstalk = np.exp(-2 * ion_separation**2 / beam_waist**2)
    gate_error = crosstalk**2  # Proportional to intensity^2 for Rabi

    ax3.semilogy(beam_waist, crosstalk, 'b-', label='Crosstalk ratio', linewidth=2)
    ax3.semilogy(beam_waist, gate_error, 'r--', label='Gate error', linewidth=2)

    ax3.axhline(y=1e-3, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Beam waist (μm)', fontsize=12)
    ax3.set_ylabel('Crosstalk / Error', fontsize=12)
    ax3.set_title(f'Optical Crosstalk (ion separation = {ion_separation} μm)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # T2 limited circuit depth
    ax4 = axes[1, 1]

    T2_values = [0.1, 1, 10, 100]  # seconds
    gate_time = 100e-6  # seconds
    colors = ['red', 'orange', 'green', 'blue']

    for T2, color in zip(T2_values, colors):
        circuit_depths = np.arange(1, 10001)
        error_per_gate = gate_time / T2
        total_error = 1 - (1 - error_per_gate)**circuit_depths

        ax4.loglog(circuit_depths, total_error * 100, color=color,
                  label=f'T₂ = {T2} s', linewidth=2)

    ax4.axhline(y=10, color='gray', linestyle='--', label='10% total error')
    ax4.set_xlabel('Circuit depth (gates)', fontsize=12)
    ax4.set_ylabel('Cumulative error (%)', fontsize=12)
    ax4.set_title('Dephasing-Limited Circuit Depth', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150)
    plt.show()


def plot_fidelity_trends():
    """Plot historical fidelity improvements"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Historical progress
    ax1 = axes[0]

    years = [2000, 2005, 2010, 2015, 2020, 2024]
    twoq_fidelity = [95, 97, 99, 99.5, 99.8, 99.9]  # Approximate values
    oneq_fidelity = [99, 99.5, 99.9, 99.99, 99.99, 99.9999]

    ax1.plot(years, [100 - f for f in twoq_fidelity], 'bo-',
            label='Two-qubit gate', linewidth=2, markersize=10)
    ax1.plot(years, [100 - f for f in oneq_fidelity], 'rs-',
            label='Single-qubit gate', linewidth=2, markersize=10)

    ax1.set_yscale('log')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Gate infidelity (%)', fontsize=12)
    ax1.set_title('Trapped Ion Gate Fidelity Progress', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # FT threshold comparison
    ax2 = axes[1]

    codes = ['Steane\n(7 qubit)', 'Surface\n(d=3)', 'Surface\n(d=5)', 'Surface\n(d=7)']
    thresholds = [1e-4, 1e-2, 5e-3, 3e-3]  # Approximate
    current_best = 6e-4  # Current best 2Q gate

    x = np.arange(len(codes))
    bars = ax2.bar(x, [t * 100 for t in thresholds], color='lightblue',
                   edgecolor='blue', linewidth=2)

    ax2.axhline(y=current_best * 100, color='red', linestyle='--', linewidth=2,
                label=f'Current best: {current_best*100:.2f}%')

    ax2.set_xticks(x)
    ax2.set_xticklabels(codes)
    ax2.set_ylabel('Threshold / Error rate (%)', fontsize=12)
    ax2.set_title('Fault Tolerance Thresholds', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fidelity_trends.png', dpi=150)
    plt.show()


def main():
    """Main simulation routine"""
    print("=" * 60)
    print("Day 910: Error Characterization and Benchmarking")
    print("=" * 60)

    print("\n--- Error Budget Analysis ---")

    # Example error budget
    errors = {
        'Heating': 2e-4,
        'Laser intensity': 1e-4,
        'Laser phase': 0.5e-4,
        'Crosstalk': 1e-4,
        'Spontaneous emission': 1e-4,
        'Dephasing': 0.5e-4
    }

    total = sum(errors.values())
    print("\nTwo-qubit gate error budget:")
    for source, error in errors.items():
        print(f"  {source}: {error*100:.3f}% ({error*1e4:.1f}×10⁻⁴)")
    print(f"  TOTAL: {total*100:.3f}% ({total*1e4:.1f}×10⁻⁴)")
    print(f"  Gate fidelity: {(1-total)*100:.3f}%")

    print("\nGenerating RB analysis plots...")
    plot_rb_results()

    print("\nGenerating error mechanism plots...")
    plot_error_analysis()

    print("\nGenerating fidelity trend plots...")
    plot_fidelity_trends()

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
| Heating rate | $\dot{\bar{n}} = \frac{q^2}{4m\hbar\omega}S_E(\omega)$ |
| Intensity error | $\epsilon_I \approx (\pi\delta\Omega/\Omega)^2/2$ |
| Phase error | $\epsilon_\phi \approx (2\pi\Delta\nu \cdot t)^2$ |
| RB decay | $P(m) = Ap^m + B$ |
| Gate fidelity | $F = (1+p)/2$ |
| Interleaved RB | $F_G = 1 - (d-1)(1-p_G/p_{ref})/d$ |

### Main Takeaways

1. **Motional heating** is often the dominant error for two-qubit gates
2. **Laser noise** (intensity and phase) contributes significantly to gate errors
3. **Crosstalk** requires careful beam design and pulse optimization
4. **Randomized benchmarking** provides SPAM-free gate fidelity measurement
5. **Error budgets** guide hardware and protocol improvements
6. State-of-the-art: >99.9% two-qubit gate fidelity achieved

## Daily Checklist

- [ ] I can identify and quantify major error sources
- [ ] I understand motional heating mechanisms
- [ ] I can calculate laser noise contributions
- [ ] I understand crosstalk and mitigation strategies
- [ ] I can implement and analyze RB protocols
- [ ] I can construct and interpret error budgets
- [ ] I have run the computational lab simulations

## Week 130 Summary

This week covered the foundations of trapped ion quantum computing:

| Day | Topic | Key Takeaway |
|-----|-------|--------------|
| 904 | Ion Trapping | Paul trap physics, pseudopotential, secular motion |
| 905 | Qubit Encoding | Hyperfine, Zeeman, optical qubits and trade-offs |
| 906 | Laser Cooling | Doppler and sideband cooling to ground state |
| 907 | Single-Qubit Gates | Raman/microwave control, composite pulses |
| 908 | Two-Qubit Gates | MS gate, geometric phases, entanglement |
| 909 | QCCD Architecture | Shuttling, junctions, scalability |
| 910 | Error Sources | Benchmarking, error budgets, mitigation |

## Preview of Week 131

Next week explores **Superconducting Qubit Systems**:
- Transmon and fluxonium qubits
- Circuit QED architecture
- Microwave gates and readout
- Comparison with trapped ions

We will see how solid-state systems achieve quantum computation.

---

*Day 910 of the QSE PhD Curriculum - Year 2, Month 33: Hardware Implementations I*
