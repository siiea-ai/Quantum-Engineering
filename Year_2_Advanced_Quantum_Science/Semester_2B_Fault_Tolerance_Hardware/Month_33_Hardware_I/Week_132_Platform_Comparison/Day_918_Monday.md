# Day 918: Coherence Time Comparison

## Schedule Overview

| Time Block | Duration | Topic |
|------------|----------|-------|
| Morning | 3 hours | T1, T2 theory and platform comparison |
| Afternoon | 2.5 hours | Decoherence mechanism analysis |
| Evening | 1.5 hours | Computational lab: Coherence data analysis |

## Learning Objectives

By the end of today, you will be able to:

1. Define and distinguish T1 (energy relaxation) and T2 (dephasing) times across platforms
2. Identify dominant decoherence mechanisms for each qubit technology
3. Calculate effective coherence limits from experimental data
4. Analyze scaling behavior of coherence with system size
5. Predict operational depth limits from coherence constraints
6. Evaluate coherence improvement trajectories for each platform

## Core Content

### 1. Coherence Time Fundamentals

Coherence times quantify how long quantum information survives in a qubit before being corrupted by environmental interactions.

#### Energy Relaxation Time (T1)

T1 characterizes the decay from the excited state |1⟩ to the ground state |0⟩:

$$\rho_{11}(t) = \rho_{11}(0) e^{-t/T_1}$$

The T1 process represents energy exchange with the environment, governed by:

$$\Gamma_1 = \frac{1}{T_1} = \frac{\pi}{\hbar^2} S_\perp(\omega_{01}) |\langle 0|\hat{O}_\perp|1\rangle|^2$$

where $S_\perp(\omega_{01})$ is the transverse noise spectral density at the qubit frequency.

#### Dephasing Time (T2)

T2 characterizes the decay of coherence (off-diagonal elements):

$$\rho_{01}(t) = \rho_{01}(0) e^{-t/T_2}$$

The total dephasing rate combines T1 contributions and pure dephasing:

$$\boxed{\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}}$$

where $T_\phi$ is the pure dephasing time from low-frequency noise.

#### Ramsey and Echo Experiments

Ramsey decay (free induction decay):
$$T_2^* = \left(\frac{1}{2T_1} + \frac{1}{T_\phi^*}\right)^{-1}$$

Hahn echo (refocused):
$$T_2^{echo} = \left(\frac{1}{2T_1} + \frac{1}{T_\phi^{echo}}\right)^{-1}$$

The echo sequence filters low-frequency noise, typically yielding $T_2^{echo} > T_2^*$.

### 2. Platform-Specific Coherence Analysis

#### Superconducting Qubits

**Typical Values (2024 state-of-art):**
- T1: 50-500 μs (transmons)
- T2,echo: 50-200 μs
- T2*: 20-100 μs

**Dominant Decoherence Mechanisms:**

1. **Dielectric Loss (TLS)**
   $$\Gamma_1^{TLS} \propto \tan\delta \cdot p_{surf}$$
   where $\tan\delta$ is the loss tangent and $p_{surf}$ is surface participation ratio.

2. **Quasiparticle Tunneling**
   $$\Gamma_1^{qp} = \frac{x_{qp}}{\tau_{qp}} \sqrt{\frac{2\Delta}{\hbar\omega_{01}}}$$
   where $x_{qp}$ is quasiparticle density.

3. **Flux Noise (for tunable qubits)**
   $$\Gamma_\phi^{flux} = \left|\frac{\partial\omega_{01}}{\partial\Phi}\right|^2 S_\Phi(0)$$

4. **Charge Noise**
   $$\Gamma_\phi^{charge} = \left|\frac{\partial\omega_{01}}{\partial n_g}\right|^2 S_{n_g}(0)$$

**Scaling Behavior:**
- T1 decreases with frequency: $T_1 \propto 1/\omega$ (Purcell limit)
- Crosstalk-induced dephasing scales as $\sim n_{neighbors}$

#### Trapped Ions

**Typical Values:**
- T1: Minutes to hours (radiative lifetime)
- T2: 1-10 seconds (with dynamical decoupling)
- T2*: 10-100 ms

**Dominant Decoherence Mechanisms:**

1. **Magnetic Field Fluctuations**
   $$\Gamma_\phi^{B} = \left(\frac{\partial\omega_{01}}{\partial B}\right)^2 S_B(0)$$

   Zeeman qubits (e.g., $^{40}$Ca$^+$): sensitive ~2.8 MHz/G
   Clock qubits (e.g., $^{171}$Yb$^+$): first-order insensitive

2. **Motional Heating**
   $$\dot{\bar{n}} = \frac{e^2}{4m\hbar\omega_m} S_E(\omega_m)$$

   Leads to gate infidelity: $\epsilon_{heat} \propto \dot{\bar{n}} \cdot t_{gate}$

3. **Laser Phase/Intensity Noise**
   For stimulated Raman transitions:
   $$\Gamma_\phi^{laser} \propto \frac{\Omega^2}{\Delta^2} S_\phi^{laser}(0)$$

4. **Off-Resonant Photon Scattering**
   $$\Gamma_{scatter} = \frac{\Gamma_{sp}}{2} \frac{\Omega^2}{\Delta^2}$$

**Scaling Behavior:**
- Individual ion coherence independent of chain length
- Collective mode coherence degrades: $T_2^{motional} \propto 1/N$
- Spectral crowding increases with N

#### Neutral Atoms

**Typical Values:**
- T1: 1-100 seconds (ground state)
- T2: 0.1-1 seconds (Rydberg-enhanced)
- T2*: 1-100 ms

**Dominant Decoherence Mechanisms:**

1. **Trap-Induced Light Shifts**
   $$\Delta\omega_{trap} = -\frac{|\Omega_{trap}|^2}{4\Delta_{trap}}(\alpha_1 - \alpha_0)$$

   Mitigated by magic wavelength trapping where $\alpha_1 = \alpha_0$.

2. **Rydberg State Decay**
   $$\Gamma_n \propto n^{-3}$$

   For n=50 Rydberg states: $\tau \sim 100$ μs at room temperature
   Reduced by blackbody radiation: $\Gamma_{BBR} \propto T^4$

3. **Doppler Dephasing**
   $$\Gamma_\phi^{Doppler} = k \sqrt{\frac{k_B T}{m}}$$

   Reduced by deep trapping and laser cooling.

4. **Atom Loss**
   Background gas collisions, Rydberg-Rydberg collisions:
   $$\tau_{loss} \sim 10-100 \text{ s (vacuum limited)}$$

**Scaling Behavior:**
- Ground-state coherence independent of array size
- Rydberg coherence limited by nearest-neighbor interactions
- Crosstalk scales as $1/r^6$ (van der Waals)

### 3. Comparative Coherence Analysis

#### Coherence Ratio: T2/T_gate

The number of operations possible within coherence time:

$$N_{ops} = \frac{T_2}{t_{gate}}$$

| Platform | T2 | t_gate (2Q) | N_ops |
|----------|-----|-------------|-------|
| SC | 100 μs | 50 ns | 2,000 |
| TI | 1 s | 200 μs | 5,000 |
| NA | 0.5 s | 1 μs | 500,000 |

#### Circuit Depth Limits

Maximum circuit depth before decoherence dominates:

$$d_{max} \approx \frac{T_2 \cdot F_{gate}^{d}}{t_{gate}}$$

For 99% gate fidelity and 50% circuit success probability:

| Platform | d_max (single qubit) | d_max (multi-qubit) |
|----------|---------------------|---------------------|
| SC | ~1,000 | ~200 |
| TI | ~3,000 | ~500 |
| NA | ~10,000 | ~1,000 |

### 4. Decoherence Scaling with System Size

#### Crosstalk-Induced Decoherence

For N qubits with nearest-neighbor coupling strength J:

$$T_2^{eff}(N) \approx \frac{T_2^{single}}{1 + \alpha N_{neighbors} (J/\Delta)^2}$$

**Superconducting:** $J \sim 1-10$ MHz, $\Delta \sim 100$ MHz → moderate scaling
**Trapped Ion:** All-to-all coupling → crosstalk scales as N
**Neutral Atom:** Blockade-limited, reconfigurable → controllable scaling

#### Collective Decoherence Modes

In multi-qubit systems, correlated errors emerge:

$$\hat{H}_{noise} = \sum_i \xi_i(t) \hat{\sigma}_z^{(i)} + \sum_{i<j} \chi_{ij}(t) \hat{\sigma}_z^{(i)}\hat{\sigma}_z^{(j)}$$

Correlated noise can be beneficial (decoherence-free subspaces) or detrimental (correlated errors).

### 5. Coherence Improvement Trajectories

#### Historical Progress

| Year | SC T1 (μs) | TI T2 (s) | NA T2 (s) |
|------|-----------|-----------|-----------|
| 2010 | 1 | 0.01 | 0.001 |
| 2015 | 50 | 0.1 | 0.01 |
| 2020 | 200 | 1 | 0.1 |
| 2024 | 500 | 10 | 1 |

Exponential improvement with approximate doubling every 2-3 years.

#### Fundamental Limits

**Superconducting:**
$$T_1^{max} \sim \frac{Q_{int}}{\omega_{01}} \sim 10 \text{ ms (materials limited)}$$

**Trapped Ion:**
$$T_2^{max} \sim T_1^{radiative} \cdot \eta_{decoupling} \sim \text{minutes (technical)}$$

**Neutral Atom:**
$$T_2^{max} \sim \tau_{trap} \cdot \eta_{magic} \sim 100 \text{ s (vacuum limited)}$$

## Quantum Computing Applications

### Circuit Depth Planning

Given coherence constraints, optimal circuit compilation minimizes:

$$\mathcal{L} = \alpha \cdot d + \beta \cdot N_{SWAP} + \gamma \cdot T_{idle}/T_2$$

where idle time during parallel execution contributes to decoherence.

### Error Budget Allocation

For a target circuit fidelity $F_{target}$:

$$1 - F_{target} = \epsilon_{gate} + \epsilon_{decoherence} + \epsilon_{readout}$$

Coherence-limited contribution:

$$\epsilon_{decoherence} \approx \frac{T_{circuit}}{T_2}$$

### Dynamical Decoupling Integration

Idle qubits benefit from DD sequences (CPMG, XY4):

$$T_2^{DD} = T_2^* \cdot \left(\frac{N_{pulses} \cdot \tau_c}{T}\right)^{\gamma}$$

where $\tau_c$ is the noise correlation time and $\gamma$ depends on noise spectrum.

## Worked Examples

### Example 1: Coherence-Limited Circuit Fidelity

**Problem:** A superconducting processor has T2 = 80 μs and executes a circuit with 100 two-qubit gates (50 ns each) and 200 single-qubit gates (20 ns each). Calculate the decoherence-induced error.

**Solution:**

1. Calculate total circuit time:
$$T_{circuit} = 100 \times 50\text{ ns} + 200 \times 20\text{ ns} = 5\text{ μs} + 4\text{ μs} = 9\text{ μs}$$

2. Estimate decoherence error per qubit:
$$\epsilon_{decoherence} = 1 - e^{-T_{circuit}/T_2} \approx \frac{T_{circuit}}{T_2} = \frac{9}{80} = 0.1125$$

3. For an n-qubit system with parallel execution (depth d ~ 50):
$$T_{effective} \approx 50 \times 50\text{ ns} = 2.5\text{ μs}$$

$$\epsilon_{per\_qubit} = \frac{2.5}{80} = 0.031$$

4. For n = 10 qubits (independent errors):
$$F_{total} \approx (1 - 0.031)^{10} \approx 0.73$$

**Answer:** The circuit loses approximately 27% fidelity to decoherence alone.

### Example 2: Platform Comparison for Long Algorithm

**Problem:** An algorithm requires 10,000 two-qubit gates on 50 qubits. Compare the coherence-limited fidelity for SC, TI, and NA platforms.

**Solution:**

1. Estimate circuit depth (assume ~50% parallelization):
$$d \approx \frac{10000}{50 \times 0.5} = 400$$

2. Calculate execution times:
- SC: $T = 400 \times 50\text{ ns} = 20$ μs
- TI: $T = 400 \times 200$ μs = 80 ms
- NA: $T = 400 \times 1$ μs = 400 μs

3. Coherence-limited fidelity per qubit:
- SC: $F = e^{-20/100} = 0.82$
- TI: $F = e^{-80/1000} = 0.92$
- NA: $F = e^{-0.4/500} = 0.999$

4. 50-qubit system fidelity:
- SC: $F_{50} = 0.82^{50} \approx 10^{-5}$
- TI: $F_{50} = 0.92^{50} \approx 0.015$
- NA: $F_{50} = 0.999^{50} \approx 0.95$

**Answer:** For long circuits, neutral atoms show the best coherence-limited performance, while superconducting qubits require error correction or circuit optimization.

### Example 3: Dynamical Decoupling Benefit

**Problem:** A trapped ion qubit has T2* = 10 ms. With XY4 dynamical decoupling (N = 100 pulses), estimate the improved T2 if the noise spectrum is 1/f.

**Solution:**

For 1/f noise with XY4 decoupling:

$$T_2^{DD} \approx T_2^* \cdot N^{2/3}$$

$$T_2^{DD} = 10\text{ ms} \times 100^{2/3} = 10\text{ ms} \times 21.5 = 215\text{ ms}$$

More sophisticated sequences (CPMG) achieve:
$$T_2^{CPMG} \approx T_2^* \cdot N$$
$$T_2^{CPMG} = 10\text{ ms} \times 100 = 1\text{ s}$$

**Answer:** Dynamical decoupling can extend coherence by 20-100x depending on the noise spectrum and sequence.

## Practice Problems

### Level 1: Direct Application

1. A neutral atom qubit has T1 = 10 s and pure dephasing time Tφ = 2 s. Calculate T2.

2. Calculate the maximum number of two-qubit gates (100 ns each) executable within 90% fidelity on a superconducting qubit with T2 = 50 μs.

3. For a trapped ion with magnetic field sensitivity 1 MHz/G and field noise $\sqrt{S_B} = 10$ nG/√Hz, estimate the dephasing rate.

### Level 2: Intermediate Analysis

4. Compare the circuit volume (qubits × depth × fidelity^depth) for 20-qubit systems on SC (T2=100μs, t_gate=50ns, F=99.5%) and TI (T2=1s, t_gate=200μs, F=99.9%) platforms.

5. A superconducting qubit couples to a resonator with Purcell rate Γ_P = 1/T_P. If T_P = 200 μs and intrinsic T1 = 500 μs, what is the effective T1?

6. Design a dynamical decoupling sequence to extend T2* = 5 μs to T2 > 100 μs given a 1/f noise spectrum.

### Level 3: Advanced Research-Level

7. Derive the scaling of effective T2 with chain length N for a trapped ion system with crosstalk strength J and frequency spread δ. At what N does crosstalk dominate over single-ion dephasing?

8. For a neutral atom array, calculate the coherence time of the N-qubit GHZ state $|\psi\rangle = (|0\rangle^{\otimes N} + |1\rangle^{\otimes N})/\sqrt{2}$ as a function of single-qubit T2 and N.

9. Develop a noise spectroscopy protocol using variable-duration Ramsey experiments to extract the noise power spectral density S(ω) for a superconducting qubit.

## Computational Lab: Coherence Data Analysis

```python
"""
Day 918 Computational Lab: Coherence Time Comparison Analysis
Analyzes and compares coherence properties across quantum computing platforms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import welch

# Set plotting style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# =============================================================================
# Part 1: Coherence Time Definitions and Fits
# =============================================================================

def t1_decay(t, T1, A, B):
    """T1 exponential decay model"""
    return A * np.exp(-t / T1) + B

def t2_ramsey(t, T2_star, A, B, omega, phi):
    """T2* Ramsey oscillation with decay"""
    return A * np.exp(-t / T2_star) * np.cos(omega * t + phi) + B

def t2_echo(t, T2, A, B, n=1):
    """T2 echo decay (can include stretch exponent)"""
    return A * np.exp(-(t / T2)**n) + B

# Generate synthetic experimental data for each platform
np.random.seed(42)

def generate_platform_data(T1, T2_star, T2_echo, t_max, noise_level=0.05):
    """Generate realistic coherence decay data"""
    t = np.linspace(0, t_max, 200)

    # T1 data
    t1_data = t1_decay(t, T1, 0.5, 0.5) + noise_level * np.random.randn(len(t))

    # T2* (Ramsey) data
    omega_det = 2 * np.pi * 0.1 / t_max  # Detuning frequency
    t2_star_data = t2_ramsey(t, T2_star, 0.5, 0.5, omega_det, 0) + noise_level * np.random.randn(len(t))

    # T2 echo data
    t2_echo_data = t2_echo(t, T2_echo, 0.5, 0.5) + noise_level * np.random.randn(len(t))

    return t, t1_data, t2_star_data, t2_echo_data

# Platform parameters (in their natural units)
platforms = {
    'Superconducting': {'T1': 100e-6, 'T2_star': 30e-6, 'T2_echo': 80e-6, 'unit': 'μs', 'scale': 1e6},
    'Trapped Ion': {'T1': 60, 'T2_star': 0.05, 'T2_echo': 1.0, 'unit': 's', 'scale': 1},
    'Neutral Atom': {'T1': 10, 'T2_star': 0.02, 'T2_echo': 0.5, 'unit': 's', 'scale': 1}
}

# Plot coherence comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Generate and plot data for each platform
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

for idx, (name, params) in enumerate(platforms.items()):
    t_max = params['T1'] * 3 if name == 'Superconducting' else max(params['T1'], params['T2_echo']) * 2
    t, t1_data, t2_star_data, t2_echo_data = generate_platform_data(
        params['T1'], params['T2_star'], params['T2_echo'], t_max
    )

    # Normalize time for plotting
    scale = params['scale']
    t_plot = t * scale

    # T1 plot
    axes[0, 0].plot(t_plot, t1_data, markers[idx], markersize=2, alpha=0.5, color=colors[idx])
    axes[0, 0].plot(t_plot, t1_decay(t, params['T1'], 0.5, 0.5), '-',
                   color=colors[idx], label=f"{name} (T1={params['T1']*scale:.0f} {params['unit']})", linewidth=2)

axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Population |1⟩')
axes[0, 0].set_title('T1 (Energy Relaxation) Comparison')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 300)
axes[0, 0].set_ylim(0.4, 1.05)

# =============================================================================
# Part 2: Coherence vs Operations Trade-off
# =============================================================================

# Define gate times for each platform
gate_times = {
    'Superconducting': {'1Q': 20e-9, '2Q': 50e-9},
    'Trapped Ion': {'1Q': 10e-6, '2Q': 200e-6},
    'Neutral Atom': {'1Q': 0.5e-6, '2Q': 1e-6}
}

coherence_times = {
    'Superconducting': 80e-6,
    'Trapped Ion': 1.0,
    'Neutral Atom': 0.5
}

# Calculate operations per coherence time
ops_per_t2 = {}
for platform in gate_times:
    t2 = coherence_times[platform]
    t_2q = gate_times[platform]['2Q']
    ops_per_t2[platform] = t2 / t_2q

# Bar chart comparison
ax = axes[0, 1]
x = np.arange(3)
bars = ax.bar(x, [ops_per_t2['Superconducting'], ops_per_t2['Trapped Ion'], ops_per_t2['Neutral Atom']],
              color=colors)
ax.set_xticks(x)
ax.set_xticklabels(['Superconducting', 'Trapped Ion', 'Neutral Atom'])
ax.set_ylabel('Two-Qubit Gates per T2')
ax.set_title('Operations Within Coherence Time')
ax.set_yscale('log')

# Add value labels
for bar, val in zip(bars, ops_per_t2.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}',
           ha='center', va='bottom', fontsize=10)

# =============================================================================
# Part 3: Decoherence Mechanism Breakdown
# =============================================================================

# Contribution breakdown for each platform
decoherence_mechanisms = {
    'Superconducting': {
        'TLS (dielectric)': 0.35,
        'Quasiparticles': 0.20,
        'Flux noise': 0.15,
        'Charge noise': 0.10,
        'Purcell': 0.15,
        'Other': 0.05
    },
    'Trapped Ion': {
        'Magnetic field': 0.40,
        'Motional heating': 0.25,
        'Laser noise': 0.20,
        'Photon scattering': 0.10,
        'Other': 0.05
    },
    'Neutral Atom': {
        'Trap light shift': 0.30,
        'Rydberg decay': 0.25,
        'Doppler/motion': 0.20,
        'Atom loss': 0.15,
        'Other': 0.10
    }
}

ax = axes[1, 0]
x = np.arange(3)
width = 0.7
bottom = np.zeros(3)

# Get all unique mechanisms
all_mechanisms = set()
for mech in decoherence_mechanisms.values():
    all_mechanisms.update(mech.keys())

# Color map for mechanisms
mechanism_colors = plt.cm.tab20(np.linspace(0, 1, 20))

# Stack bars
for i, mechanism in enumerate(['TLS (dielectric)', 'Quasiparticles', 'Flux noise', 'Magnetic field',
                                'Motional heating', 'Laser noise', 'Trap light shift', 'Rydberg decay',
                                'Doppler/motion', 'Purcell', 'Photon scattering', 'Charge noise',
                                'Atom loss', 'Other']):
    values = []
    for platform in ['Superconducting', 'Trapped Ion', 'Neutral Atom']:
        values.append(decoherence_mechanisms[platform].get(mechanism, 0))

    if sum(values) > 0:
        ax.bar(x, values, width, bottom=bottom, label=mechanism, color=mechanism_colors[i])
        bottom += values

ax.set_xticks(x)
ax.set_xticklabels(['Superconducting', 'Trapped Ion', 'Neutral Atom'])
ax.set_ylabel('Relative Contribution')
ax.set_title('Decoherence Mechanism Breakdown')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.set_ylim(0, 1.2)

# =============================================================================
# Part 4: Coherence Scaling with System Size
# =============================================================================

ax = axes[1, 1]

# Model coherence scaling
N_qubits = np.arange(1, 101)

# Superconducting: moderate degradation due to crosstalk
T2_SC = 80e-6 / (1 + 0.005 * N_qubits)  # Empirical model

# Trapped ion: all-to-all coupling causes spectral crowding
T2_TI = 1.0 / (1 + 0.02 * N_qubits)  # Collective mode dephasing

# Neutral atom: weak scaling due to blockade locality
T2_NA = 0.5 / (1 + 0.001 * N_qubits)  # Minimal degradation

# Normalize to single-qubit value
ax.semilogy(N_qubits, T2_SC / T2_SC[0], '-', color=colors[0], linewidth=2,
            label='Superconducting')
ax.semilogy(N_qubits, T2_TI / T2_TI[0], '-', color=colors[1], linewidth=2,
            label='Trapped Ion')
ax.semilogy(N_qubits, T2_NA / T2_NA[0], '-', color=colors[2], linewidth=2,
            label='Neutral Atom')

ax.set_xlabel('Number of Qubits')
ax.set_ylabel('T2(N) / T2(1)')
ax.set_title('Coherence Scaling with System Size')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 100)
ax.set_ylim(0.1, 1.1)

plt.tight_layout()
plt.savefig('coherence_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 5: Circuit Depth Limits Analysis
# =============================================================================

print("\n" + "="*70)
print("COHERENCE-LIMITED CIRCUIT ANALYSIS")
print("="*70)

def circuit_fidelity(depth, T2, t_gate, gate_fidelity):
    """
    Calculate circuit fidelity limited by coherence and gate errors
    """
    # Time for circuit execution
    T_circuit = depth * t_gate

    # Decoherence contribution
    F_decoherence = np.exp(-T_circuit / T2)

    # Gate error contribution
    F_gates = gate_fidelity ** depth

    return F_decoherence * F_gates

# Platform specifications
platforms_full = {
    'Superconducting': {
        'T2': 80e-6,
        't_2Q': 50e-9,
        'F_2Q': 0.995
    },
    'Trapped Ion': {
        'T2': 1.0,
        't_2Q': 200e-6,
        'F_2Q': 0.999
    },
    'Neutral Atom': {
        'T2': 0.5,
        't_2Q': 1e-6,
        'F_2Q': 0.995
    }
}

depths = np.arange(1, 1001)

plt.figure(figsize=(12, 5))

# Plot 1: Fidelity vs depth
plt.subplot(1, 2, 1)
for idx, (name, params) in enumerate(platforms_full.items()):
    F = circuit_fidelity(depths, params['T2'], params['t_2Q'], params['F_2Q'])
    plt.semilogy(depths, F, color=colors[idx], label=name, linewidth=2)

plt.axhline(y=0.5, color='gray', linestyle='--', label='50% threshold')
plt.xlabel('Circuit Depth (2Q gates)')
plt.ylabel('Circuit Fidelity')
plt.title('Fidelity vs Circuit Depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1000)
plt.ylim(1e-3, 1)

# Plot 2: Max depth at given fidelity threshold
plt.subplot(1, 2, 2)
thresholds = np.linspace(0.1, 0.99, 50)
max_depths = {name: [] for name in platforms_full}

for F_thresh in thresholds:
    for name, params in platforms_full.items():
        F = circuit_fidelity(depths, params['T2'], params['t_2Q'], params['F_2Q'])
        valid_depths = depths[F >= F_thresh]
        max_depth = valid_depths[-1] if len(valid_depths) > 0 else 0
        max_depths[name].append(max_depth)

for idx, (name, depths_list) in enumerate(max_depths.items()):
    plt.plot(thresholds, depths_list, color=colors[idx], label=name, linewidth=2)

plt.xlabel('Fidelity Threshold')
plt.ylabel('Maximum Circuit Depth')
plt.title('Maximum Depth for Given Fidelity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('circuit_depth_limits.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 6: Numerical Summary
# =============================================================================

print("\n" + "="*70)
print("PLATFORM COHERENCE SUMMARY")
print("="*70)

for name, params in platforms_full.items():
    # Calculate key metrics
    ops_per_t2 = params['T2'] / params['t_2Q']

    # Find depth at 50% fidelity
    F = circuit_fidelity(np.arange(1, 10000), params['T2'], params['t_2Q'], params['F_2Q'])
    d_50 = np.where(F < 0.5)[0][0] if np.any(F < 0.5) else 10000

    # Find depth at 90% fidelity
    d_90 = np.where(F < 0.9)[0][0] if np.any(F < 0.9) else 10000

    print(f"\n{name}:")
    print(f"  T2 = {params['T2']*1e6:.0f} μs" if params['T2'] < 1 else f"  T2 = {params['T2']:.2f} s")
    print(f"  2Q Gate Time = {params['t_2Q']*1e9:.0f} ns" if params['t_2Q'] < 1e-6 else f"  2Q Gate Time = {params['t_2Q']*1e6:.0f} μs")
    print(f"  2Q Gate Fidelity = {params['F_2Q']*100:.2f}%")
    print(f"  Operations per T2 = {ops_per_t2:.0f}")
    print(f"  Max depth (F>90%) = {d_90}")
    print(f"  Max depth (F>50%) = {d_50}")

print("\n" + "="*70)
print("Key Insight: Gate speed compensates for short coherence in SC qubits,")
print("while long coherence in TI/NA allows deeper circuits despite slower gates.")
print("="*70)
```

## Summary

### Key Coherence Metrics by Platform

| Platform | T1 | T2 (echo) | T2* | Dominant Mechanism |
|----------|-----|-----------|-----|-------------------|
| Superconducting | 50-500 μs | 50-200 μs | 20-100 μs | TLS, quasiparticles |
| Trapped Ion | minutes | 1-10 s | 10-100 ms | Magnetic field |
| Neutral Atom | 1-100 s | 0.1-1 s | 1-100 ms | Trap light shift |

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Dephasing rate | $$\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$$ |
| Operations per T2 | $$N_{ops} = T_2/t_{gate}$$ |
| Circuit decoherence | $$\epsilon_{dec} \approx T_{circuit}/T_2$$ |
| Echo enhancement | $$T_2^{echo}/T_2^* \sim N_{pulses}^\gamma$$ |

### Main Takeaways

1. **Coherence-operation trade-off**: Fast gates (SC) compensate for short T2; long T2 (TI, NA) allows slower gates
2. **Mechanism-specific mitigation**: Each platform requires different engineering approaches
3. **Scaling challenges**: Crosstalk and collective effects limit effective coherence at scale
4. **Dynamical decoupling**: Essential for extending T2* to approach T1 limits
5. **Application matching**: Long-coherence platforms (TI, NA) favor deep algorithms; fast-gate platforms (SC) favor shallow, high-throughput circuits

## Daily Checklist

- [ ] I can explain the difference between T1, T2, T2*, and T2,echo
- [ ] I can identify dominant decoherence mechanisms for each platform
- [ ] I can calculate coherence-limited circuit fidelity
- [ ] I can estimate the maximum circuit depth for each platform
- [ ] I understand how dynamical decoupling extends coherence
- [ ] I can analyze coherence scaling with system size

## Preview of Day 919

Tomorrow we examine **Gate Fidelity Benchmarks**, covering randomized benchmarking protocols, cross-entropy benchmarking (XEB), cycle benchmarking, and state-of-the-art fidelity results for each platform.
