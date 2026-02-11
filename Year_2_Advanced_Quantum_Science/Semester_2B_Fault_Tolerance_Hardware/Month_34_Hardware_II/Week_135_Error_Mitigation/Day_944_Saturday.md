# Day 944: Dynamical Decoupling - DD Sequences for Decoherence Suppression

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | DD theory, pulse sequences, error suppression mechanisms |
| Afternoon | 2 hours | Advanced DD sequences and optimization |
| Evening | 2 hours | Computational lab - DD implementation in Qiskit |

## Learning Objectives

By the end of this day, you will be able to:

1. **Explain DD mechanisms** - Understand how refocusing pulses average out noise
2. **Implement standard DD sequences** - Apply XY4, CPMG, and Uhrig DD
3. **Analyze error suppression** - Calculate fidelity improvement from DD
4. **Design concatenated DD** - Build multi-level DD for enhanced suppression
5. **Optimize DD for noise spectra** - Match DD sequences to specific noise environments
6. **Integrate DD with circuits** - Insert DD during idle periods in quantum algorithms

## Core Content

### 1. Dynamical Decoupling Fundamentals

#### 1.1 The Decoherence Problem

Qubits coupled to an environment experience decoherence:

$$H = H_S + H_E + H_{SE}$$

where $H_{SE}$ causes unwanted evolution:

$$H_{SE} = \sum_\alpha S_\alpha \otimes E_\alpha$$

For a qubit: $S_\alpha \in \{X, Y, Z\}$ couple to environment operators $E_\alpha$.

**Pure dephasing** (dominant in many systems):
$$H_{SE} = Z \otimes B_z$$

Evolution causes phase accumulation: $|\psi\rangle \to e^{i\phi(t)}|\psi\rangle$ where $\phi$ is random.

#### 1.2 The Refocusing Principle

A $\pi$-pulse ($X$ or $Y$ gate) flips the qubit, reversing the effect of $Z$ noise:

$$X \cdot Z \cdot X = -Z$$

If we apply $X$ at time $\tau/2$:

**Phase accumulated** in $[0, \tau/2]$: $+\phi_1$
**Phase accumulated** in $[\tau/2, \tau]$: $-\phi_2$ (sign flipped)

For slowly varying noise ($\phi_1 \approx \phi_2$): **net phase $\approx 0$**

#### 1.3 Average Hamiltonian Theory

Over a DD cycle of period $T$, the effective Hamiltonian is:

$$\boxed{\bar{H} = \frac{1}{T}\int_0^T U^\dagger(t) H_{SE} U(t) dt}$$

where $U(t)$ is the control unitary from DD pulses.

**Goal**: Design $U(t)$ such that $\bar{H} = 0$.

### 2. Standard DD Sequences

#### 2.1 Spin Echo (Hahn Echo)

Simplest DD sequence: single $\pi$-pulse at midpoint.

**Sequence**: $\tau/2 - X - \tau/2$

**Error suppression**:
$$\boxed{1 - F \propto \left(\frac{\tau}{T_2^*}\right)^2}$$

Converts linear decay ($T_2^*$) to quadratic decay.

#### 2.2 CPMG Sequence

Carr-Purcell-Meiboom-Gill: multiple refocusing pulses.

**Sequence**: $(\tau/2 - Y - \tau - Y - \tau/2)^n$

Using $Y$ pulses (instead of $X$) corrects for pulse errors.

**Error suppression** for $n$ pulses:
$$\boxed{1 - F \propto \left(\frac{\tau}{n T_2}\right)^2}$$

More pulses = better suppression (until pulse errors dominate).

#### 2.3 XY4 Sequence

Four-pulse sequence with alternating axes:

**Sequence**: $\tau - X - \tau - Y - \tau - X - \tau - Y$

**Advantages**:
- Cancels pulse imperfections to first order
- Symmetric under time reversal
- Robust against axis errors

**Toggling frame Hamiltonian**:
$$\bar{H}^{(0)} = 0 \quad \text{(zeroth order)}$$

#### 2.4 XY8 Sequence

Extended version of XY4:

**Sequence**: $X - Y - X - Y - Y - X - Y - X$

**Advantages**:
- Better cancellation of higher-order errors
- Robust against finite pulse width effects

### 3. Uhrig Dynamical Decoupling (UDD)

#### 3.1 Optimal Pulse Timing

Uhrig showed that for pure dephasing, optimal pulse positions are:

$$\boxed{t_j = T \sin^2\left(\frac{\pi j}{2n + 2}\right), \quad j = 1, 2, \ldots, n}$$

where $T$ is total evolution time and $n$ is number of pulses.

#### 3.2 Error Suppression Order

UDD achieves $(n+1)$-th order suppression:

$$\boxed{1 - F = O\left(\left(\frac{T}{\tau_c}\right)^{2(n+1)}\right)}$$

where $\tau_c$ is the correlation time of the noise.

**Comparison**:
- CPMG: 2nd order suppression
- UDD with $n$ pulses: $(n+1)$-th order

#### 3.3 Limitations

UDD is optimal for:
- Pure dephasing noise
- Noise with specific spectral properties

For general noise, XY4/CPMG may outperform UDD.

### 4. Filter Function Formalism

#### 4.1 Noise Spectral Density

Environment noise characterized by power spectral density $S(\omega)$:

$$\langle B(t) B(t') \rangle = \int_{-\infty}^{\infty} S(\omega) e^{-i\omega(t-t')} d\omega$$

#### 4.2 Filter Function

DD sequence creates a filter function $F(\omega)$:

$$\boxed{F(\omega) = \left| \sum_{j=0}^n (-1)^j e^{i\omega t_j} \right|^2}$$

where $t_j$ are pulse times.

#### 4.3 Fidelity Decay

The coherence decay is:

$$\boxed{1 - F = \frac{1}{\pi} \int_0^\infty \frac{S(\omega)}{\omega^2} F(\omega) d\omega}$$

**Design principle**: Choose pulse timing to minimize $F(\omega)$ where $S(\omega)$ is large.

### 5. Concatenated Dynamical Decoupling (CDD)

#### 5.1 Hierarchical Construction

Build complex sequences recursively:

**Level 0**: $C_0 = \tau$ (free evolution)
**Level 1**: $C_1 = C_0 - X - C_0 - Z - C_0 - X - C_0 - Z$
**Level $n$**: $C_n = C_{n-1} - X - C_{n-1} - Z - C_{n-1} - X - C_{n-1} - Z$

#### 5.2 Error Suppression

CDD achieves exponential suppression:

$$\boxed{1 - F \propto \epsilon^{2^n}}$$

where $\epsilon$ is the single-level error and $n$ is concatenation level.

**Trade-off**: Number of pulses grows as $4^n$, eventually limited by pulse errors.

### 6. Practical Implementation

#### 6.1 Pulse Errors

Real pulses have imperfections:
- **Rotation angle error**: $\pi + \delta\theta$
- **Axis error**: Rotation axis tilted
- **Timing jitter**: Pulses not at exact times

**Robust sequences** (XY4, XY8) partially cancel these errors.

#### 6.2 DD in Quantum Circuits

Insert DD during idle periods:

```
q0: ─[Gate]─────────[DD: XY4]─────────[Gate]─
q1: ─[Gate]─[Gate]────────────[Gate]─[Gate]─
```

**Qiskit Runtime** automatically inserts DD for transpiled circuits.

#### 6.3 Choosing DD Parameters

| Noise Type | Recommended DD | Notes |
|------------|----------------|-------|
| Low-frequency ($1/f$) | CPMG, UDD | Many pulses help |
| White noise | Spin echo | Simple is sufficient |
| Quasi-static | Single echo | One refocusing enough |
| Unknown | XY4 | Good general choice |

### 7. DD Combined with Other Mitigation

#### 7.1 DD + Measurement Mitigation

$$\text{DD (circuit)} \to \text{Measurement Mitigation (classical)}$$

Complementary: DD reduces coherent errors, measurement mitigation handles readout.

#### 7.2 DD + ZNE

DD improves baseline fidelity, making ZNE extrapolation more accurate:
- Smaller effective error rate
- More reliable at higher scale factors

## Quantum Computing Applications

### Quantum Memory

DD extends coherence time for storing quantum states:
- Quantum repeaters
- Error correction syndrome storage
- Variational algorithm parameter storage

### Variational Quantum Algorithms

Insert DD during classical optimization loop idle times:
- Maintains state fidelity while waiting for next iteration
- Critical for hybrid quantum-classical workflows

### Quantum Sensing

DD sequences can be tuned to specific frequencies:
- AC magnetometry using CPMG
- Noise spectroscopy
- Correlation spectroscopy

## Worked Examples

### Example 1: CPMG Error Suppression

**Problem**: A qubit has $T_2^* = 10 \mu s$ and $T_2 = 50 \mu s$. For a total evolution time of $20 \mu s$, calculate the fidelity without DD and with 10-pulse CPMG.

**Solution**:

**Without DD** (free decay):
$$F_{\text{free}} = e^{-t/T_2^*} = e^{-20/10} = e^{-2} \approx 0.135$$

**With CPMG** (10 pulses):
Inter-pulse spacing: $\tau = 20/10 = 2 \mu s$

Using CPMG, the decay is characterized by $T_2$ instead of $T_2^*$:
$$F_{\text{CPMG}} \approx e^{-t/T_2} = e^{-20/50} = e^{-0.4} \approx 0.670$$

More precisely, with refocusing:
$$1 - F \propto (t/nT_2)^2$$

$$F_{\text{CPMG}} \approx 1 - (20/(10 \times 50))^2 = 1 - 0.04^2 = 1 - 0.0016$$

$$\boxed{F_{\text{CPMG}} \approx 0.998}$$

Improvement factor: $\sim 7\times$ in fidelity.

### Example 2: UDD Pulse Positions

**Problem**: Calculate pulse positions for 4-pulse UDD over total time $T = 100$ ns.

**Solution**:

Using the Uhrig formula:
$$t_j = T \sin^2\left(\frac{\pi j}{2n + 2}\right) = 100 \sin^2\left(\frac{\pi j}{10}\right)$$

| $j$ | $\pi j / 10$ | $\sin^2$ | $t_j$ (ns) |
|-----|--------------|----------|------------|
| 1 | $\pi/10$ | 0.0955 | 9.55 |
| 2 | $2\pi/10$ | 0.345 | 34.5 |
| 3 | $3\pi/10$ | 0.655 | 65.5 |
| 4 | $4\pi/10$ | 0.905 | 90.5 |

$$\boxed{t = (9.55, 34.5, 65.5, 90.5) \text{ ns}}$$

Note: Not equally spaced (contrast with CPMG: 20, 40, 60, 80 ns).

### Example 3: Filter Function

**Problem**: For a spin echo (single pulse at $T/2$), compute the filter function at $\omega = 2\pi/T$.

**Solution**:

Pulse times: $t_0 = 0$, $t_1 = T/2$, $t_2 = T$ (start, pulse, end).

Signs: $(-1)^0 = 1$, $(-1)^1 = -1$, $(-1)^2 = 1$.

$$F(\omega) = \left| e^{i \cdot 0} - e^{i\omega T/2} + e^{i\omega T} \right|^2$$

At $\omega = 2\pi/T$:
$$= \left| 1 - e^{i\pi} + e^{i2\pi} \right|^2 = |1 - (-1) + 1|^2 = |3|^2$$

$$\boxed{F(2\pi/T) = 9}$$

At this frequency, spin echo does not provide suppression. This is why noise at the "resonant" frequency $2\pi/T$ leaks through.

## Practice Problems

### Level 1: Direct Application

1. For XY4 with inter-pulse delay $\tau = 50$ ns, what is the total sequence duration?

2. A qubit with $T_2 = 100 \mu s$ undergoes a 10 $\mu s$ evolution. Calculate fidelity with and without a single spin echo.

3. How many pulses does level-3 CDD contain?

### Level 2: Intermediate

4. Derive the filter function for CPMG with 2 pulses at $T/4$ and $3T/4$.

5. For $1/f$ noise spectrum $S(\omega) \propto 1/\omega$, which DD sequence (CPMG or UDD) provides better suppression and why?

6. Design a DD sequence that protects against both $X$ and $Z$ noise simultaneously.

### Level 3: Challenging

7. Prove that XY4 cancels first-order pulse rotation errors.

8. For noise spectrum $S(\omega) = S_0/(1 + (\omega\tau_c)^2)$ (Lorentzian), derive the optimal number of CPMG pulses given pulse error rate $\epsilon_p$.

9. Analyze the break-even point where pulse errors exceed the benefit of additional DD pulses.

## Computational Lab

```python
"""
Day 944: Dynamical Decoupling Implementation
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, YGate, ZGate
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable

# ============================================================
# Part 1: DD Sequence Generators
# ============================================================

def create_spin_echo(total_time: float, dt: float = 1e-9) -> QuantumCircuit:
    """Create spin echo (single X pulse at midpoint)."""
    qc = QuantumCircuit(1)

    # First half: free evolution (represented by identity gates)
    n_idles = int(total_time / (2 * dt))
    for _ in range(n_idles):
        qc.id(0)

    # Pi pulse
    qc.x(0)

    # Second half: free evolution
    for _ in range(n_idles):
        qc.id(0)

    return qc

def create_cpmg(total_time: float, n_pulses: int, dt: float = 1e-9) -> QuantumCircuit:
    """Create CPMG sequence with Y pulses."""
    qc = QuantumCircuit(1)

    tau = total_time / (2 * n_pulses)
    n_idles_per_tau = max(1, int(tau / dt))

    # tau/2 - (Y - tau - Y)^(n-1) - Y - tau/2
    # Simplified: equal spacing

    # First tau/2
    for _ in range(n_idles_per_tau // 2):
        qc.id(0)

    for i in range(n_pulses):
        qc.y(0)

        # tau (or tau/2 for last)
        n_idle = n_idles_per_tau if i < n_pulses - 1 else n_idles_per_tau // 2
        for _ in range(n_idle):
            qc.id(0)

    return qc

def create_xy4(total_time: float, n_cycles: int = 1, dt: float = 1e-9) -> QuantumCircuit:
    """Create XY4 sequence."""
    qc = QuantumCircuit(1)

    tau = total_time / (4 * n_cycles)
    n_idles = max(1, int(tau / dt))

    for _ in range(n_cycles):
        # tau - X - tau - Y - tau - X - tau - Y
        for _ in range(n_idles):
            qc.id(0)
        qc.x(0)

        for _ in range(n_idles):
            qc.id(0)
        qc.y(0)

        for _ in range(n_idles):
            qc.id(0)
        qc.x(0)

        for _ in range(n_idles):
            qc.id(0)
        qc.y(0)

    return qc

def create_xy8(total_time: float, n_cycles: int = 1, dt: float = 1e-9) -> QuantumCircuit:
    """Create XY8 sequence."""
    qc = QuantumCircuit(1)

    tau = total_time / (8 * n_cycles)
    n_idles = max(1, int(tau / dt))

    pulse_sequence = ['x', 'y', 'x', 'y', 'y', 'x', 'y', 'x']

    for _ in range(n_cycles):
        for pulse in pulse_sequence:
            for _ in range(n_idles):
                qc.id(0)
            if pulse == 'x':
                qc.x(0)
            else:
                qc.y(0)

    return qc

def create_udd(total_time: float, n_pulses: int, dt: float = 1e-9) -> QuantumCircuit:
    """Create Uhrig DD sequence."""
    qc = QuantumCircuit(1)

    # Compute pulse times
    pulse_times = []
    for j in range(1, n_pulses + 1):
        t_j = total_time * np.sin(np.pi * j / (2 * n_pulses + 2))**2
        pulse_times.append(t_j)

    current_time = 0
    for t_pulse in pulse_times:
        # Free evolution until pulse
        n_idles = max(1, int((t_pulse - current_time) / dt))
        for _ in range(n_idles):
            qc.id(0)

        # Pi pulse
        qc.x(0)
        current_time = t_pulse

    # Final free evolution
    n_idles = max(1, int((total_time - current_time) / dt))
    for _ in range(n_idles):
        qc.id(0)

    return qc

# ============================================================
# Part 2: Simulation with Decoherence
# ============================================================

def create_t2_noise_model(t1: float = 50e-6, t2: float = 30e-6,
                          gate_time: float = 50e-9) -> NoiseModel:
    """Create noise model with T1/T2 relaxation."""
    noise_model = NoiseModel()

    # Thermal relaxation for each gate
    error = thermal_relaxation_error(t1, t2, gate_time)
    noise_model.add_all_qubit_quantum_error(error, ['id', 'x', 'y', 'z', 'h'])

    return noise_model

def measure_fidelity_with_dd(dd_circuit: QuantumCircuit,
                            noise_model: NoiseModel,
                            initial_state: str = '+',
                            shots: int = 10000) -> float:
    """Measure fidelity after DD sequence."""

    # Prepare initial state
    qc = QuantumCircuit(1, 1)

    if initial_state == '+':
        qc.h(0)  # |+⟩ state
    elif initial_state == '0':
        pass  # Already |0⟩
    elif initial_state == '1':
        qc.x(0)

    # Apply DD
    qc = qc.compose(dd_circuit)

    # Measure in appropriate basis
    if initial_state == '+':
        qc.h(0)  # Measure in X basis

    qc.measure(0, 0)

    # Simulate
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Fidelity: probability of measuring original state
    if initial_state in ['+', '0']:
        fidelity = counts.get('0', 0) / shots
    else:
        fidelity = counts.get('1', 0) / shots

    return fidelity

def compare_dd_sequences():
    """Compare different DD sequences."""

    print("="*60)
    print("Dynamical Decoupling Comparison")
    print("="*60)

    # Parameters
    t1 = 100e-6  # 100 μs
    t2 = 50e-6   # 50 μs
    gate_time = 50e-9  # 50 ns

    noise_model = create_t2_noise_model(t1, t2, gate_time)

    # Test different evolution times
    evolution_times = np.linspace(1e-6, 20e-6, 10)  # 1-20 μs

    results = {
        'No DD': [],
        'Spin Echo': [],
        'CPMG (4)': [],
        'CPMG (8)': [],
        'XY4': [],
        'UDD (4)': []
    }

    for t_total in evolution_times:
        n_idles = max(1, int(t_total / gate_time))

        # No DD (just free evolution)
        qc_free = QuantumCircuit(1)
        for _ in range(n_idles):
            qc_free.id(0)
        results['No DD'].append(measure_fidelity_with_dd(qc_free, noise_model))

        # Spin Echo
        qc_echo = create_spin_echo(t_total, gate_time)
        results['Spin Echo'].append(measure_fidelity_with_dd(qc_echo, noise_model))

        # CPMG with 4 pulses
        qc_cpmg4 = create_cpmg(t_total, 4, gate_time)
        results['CPMG (4)'].append(measure_fidelity_with_dd(qc_cpmg4, noise_model))

        # CPMG with 8 pulses
        qc_cpmg8 = create_cpmg(t_total, 8, gate_time)
        results['CPMG (8)'].append(measure_fidelity_with_dd(qc_cpmg8, noise_model))

        # XY4
        qc_xy4 = create_xy4(t_total, 1, gate_time)
        results['XY4'].append(measure_fidelity_with_dd(qc_xy4, noise_model))

        # UDD with 4 pulses
        qc_udd = create_udd(t_total, 4, gate_time)
        results['UDD (4)'].append(measure_fidelity_with_dd(qc_udd, noise_model))

        print(f"t = {t_total*1e6:.1f} μs: "
              f"No DD = {results['No DD'][-1]:.3f}, "
              f"XY4 = {results['XY4'][-1]:.3f}")

    # Plot results
    plt.figure(figsize=(12, 6))

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    markers = ['o', 's', '^', 'v', 'D', 'p']

    for (name, fidelities), color, marker in zip(results.items(), colors, markers):
        plt.plot(evolution_times * 1e6, fidelities, marker=marker, color=color,
                label=name, markersize=8, linewidth=2)

    plt.xlabel('Evolution Time (μs)', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title(f'DD Sequence Comparison (T1={t1*1e6:.0f}μs, T2={t2*1e6:.0f}μs)', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.4, 1.05)

    plt.tight_layout()
    plt.savefig('dd_comparison.png', dpi=150)
    plt.show()

    return results

# ============================================================
# Part 3: Filter Function Analysis
# ============================================================

def compute_filter_function(pulse_times: List[float], total_time: float,
                           frequencies: np.ndarray) -> np.ndarray:
    """Compute the filter function F(ω) for a DD sequence."""
    # Pulse times include start (0) and end (T)
    all_times = [0] + list(pulse_times) + [total_time]

    F = np.zeros_like(frequencies)

    for omega_idx, omega in enumerate(frequencies):
        # Sum over intervals with alternating signs
        complex_sum = 0
        for j, t_j in enumerate(all_times):
            sign = (-1)**j
            complex_sum += sign * np.exp(1j * omega * t_j)

        F[omega_idx] = np.abs(complex_sum)**2

    return F

def analyze_filter_functions():
    """Analyze and compare filter functions of different DD sequences."""

    print("\n" + "="*60)
    print("Filter Function Analysis")
    print("="*60)

    total_time = 10e-6  # 10 μs
    frequencies = np.linspace(0, 5e6, 1000) * 2 * np.pi  # 0-5 MHz

    # Define pulse times for different sequences
    sequences = {
        'Free Evolution': [],
        'Spin Echo': [total_time/2],
        'CPMG (4)': [total_time * (2*i + 1) / 8 for i in range(4)],
        'XY4': [total_time * (i + 1) / 5 for i in range(4)],
        'UDD (4)': [total_time * np.sin(np.pi * j / 10)**2 for j in range(1, 5)]
    }

    plt.figure(figsize=(12, 6))

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for (name, pulse_times), color in zip(sequences.items(), colors):
        F = compute_filter_function(pulse_times, total_time, frequencies)
        # Normalize
        F = F / (len(pulse_times) + 1)**2

        plt.semilogy(frequencies / (2 * np.pi * 1e6), F, label=name,
                    color=color, linewidth=2)

    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('Filter Function F(ω) (normalized)', fontsize=12)
    plt.title('DD Sequence Filter Functions', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)

    plt.tight_layout()
    plt.savefig('filter_functions.png', dpi=150)
    plt.show()

    # Print pulse positions
    print("\nPulse positions (μs):")
    for name, times in sequences.items():
        if times:
            times_us = [t * 1e6 for t in times]
            print(f"  {name}: {times_us}")

# ============================================================
# Part 4: DD in Quantum Circuits
# ============================================================

def insert_dd_in_circuit(circuit: QuantumCircuit,
                        dd_type: str = 'xy4') -> QuantumCircuit:
    """Insert DD sequences in idle periods of a circuit."""

    # This is a simplified version - real implementation would
    # analyze circuit timing and insert DD in actual idle slots

    from qiskit.transpiler.passes import PadDynamicalDecoupling
    from qiskit.circuit.library import XGate, YGate

    # Define DD sequence based on type
    if dd_type == 'xy4':
        dd_sequence = [XGate(), YGate(), XGate(), YGate()]
    elif dd_type == 'xx':
        dd_sequence = [XGate(), XGate()]
    else:
        dd_sequence = [XGate()]

    # Note: In real Qiskit, you would use:
    # from qiskit.transpiler import PassManager
    # pm = PassManager([PadDynamicalDecoupling(durations, dd_sequence)])
    # return pm.run(circuit)

    # For demonstration, manually add DD
    new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

    for instruction in circuit.data:
        new_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)

    return new_circuit

def demonstrate_dd_in_vqe():
    """Demonstrate DD insertion in a VQE-like circuit."""

    print("\n" + "="*60)
    print("DD in Quantum Algorithm (VQE-like circuit)")
    print("="*60)

    # Create a simple variational circuit
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)

    # Ansatz layers with intentional idle periods
    for layer in range(2):
        # Rotation layer (all qubits active)
        for q in range(n_qubits):
            qc.ry(np.pi/4, q)

        # Entangling layer (only adjacent qubits)
        # Qubit 2 is idle during q0-q1 entanglement
        qc.cx(0, 1)
        qc.barrier()  # Visualize idle period
        qc.cx(1, 2)
        qc.barrier()

    qc.measure_all()

    print("\nOriginal circuit:")
    print(qc.draw())

    # Create version with DD during idle periods
    qc_dd = QuantumCircuit(n_qubits)

    for layer in range(2):
        for q in range(n_qubits):
            qc_dd.ry(np.pi/4, q)

        # During CX(0,1), qubit 2 gets DD
        qc_dd.cx(0, 1)
        qc_dd.x(2)  # DD pulse on idle qubit
        qc_dd.y(2)  # DD pulse on idle qubit

        qc_dd.cx(1, 2)
        qc_dd.x(0)  # DD pulse on idle qubit
        qc_dd.y(0)  # DD pulse on idle qubit

    qc_dd.measure_all()

    print("\nCircuit with DD on idle qubits:")
    print(qc_dd.draw())

    # Compare fidelities
    t1, t2 = 50e-6, 30e-6
    gate_time = 50e-9
    noise_model = create_t2_noise_model(t1, t2, gate_time)

    simulator = AerSimulator(noise_model=noise_model)

    # Run both circuits
    shots = 20000

    result_orig = simulator.run(qc, shots=shots).result()
    counts_orig = result_orig.get_counts()

    result_dd = simulator.run(qc_dd, shots=shots).result()
    counts_dd = result_dd.get_counts()

    # Compare top outcomes
    print("\nTop outcomes (original vs DD):")
    print("Original:")
    for bs, count in sorted(counts_orig.items(), key=lambda x: -x[1])[:5]:
        print(f"  {bs}: {count}")

    print("With DD:")
    for bs, count in sorted(counts_dd.items(), key=lambda x: -x[1])[:5]:
        print(f"  {bs}: {count}")

# ============================================================
# Part 5: Concatenated DD
# ============================================================

def create_concatenated_dd(level: int, base_time: float,
                          dt: float = 1e-9) -> QuantumCircuit:
    """Create concatenated DD sequence."""

    if level == 0:
        # Base level: just free evolution
        qc = QuantumCircuit(1)
        n_idles = max(1, int(base_time / dt))
        for _ in range(n_idles):
            qc.id(0)
        return qc

    # Recursive construction: C_n = C_{n-1} - X - C_{n-1} - Z - C_{n-1} - X - C_{n-1} - Z
    c_prev = create_concatenated_dd(level - 1, base_time / 4, dt)

    qc = QuantumCircuit(1)
    qc = qc.compose(c_prev)
    qc.x(0)
    qc = qc.compose(c_prev)
    qc.z(0)
    qc = qc.compose(c_prev)
    qc.x(0)
    qc = qc.compose(c_prev)
    qc.z(0)

    return qc

def analyze_concatenated_dd():
    """Analyze performance of concatenated DD."""

    print("\n" + "="*60)
    print("Concatenated DD Analysis")
    print("="*60)

    t1, t2 = 100e-6, 50e-6
    gate_time = 20e-9  # Faster gates for CDD
    total_time = 5e-6

    noise_model = create_t2_noise_model(t1, t2, gate_time)

    levels = [0, 1, 2, 3]
    fidelities = []
    pulse_counts = []

    for level in levels:
        qc = create_concatenated_dd(level, total_time, gate_time)

        # Count pulses
        n_pulses = sum(1 for inst in qc.data if inst.operation.name in ['x', 'y', 'z'])
        pulse_counts.append(n_pulses)

        # Measure fidelity
        fidelity = measure_fidelity_with_dd(qc, noise_model)
        fidelities.append(fidelity)

        print(f"Level {level}: {n_pulses} pulses, Fidelity = {fidelity:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(levels, fidelities, color='steelblue', edgecolor='black')
    ax1.set_xlabel('CDD Level')
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Fidelity vs CDD Level')
    ax1.set_ylim(0.5, 1.05)

    ax2.semilogy(levels, pulse_counts, 'ro-', markersize=10)
    ax2.set_xlabel('CDD Level')
    ax2.set_ylabel('Number of Pulses')
    ax2.set_title('Pulse Count vs CDD Level')

    plt.tight_layout()
    plt.savefig('concatenated_dd.png', dpi=150)
    plt.show()

# ============================================================
# Part 6: Optimal DD for Noise Spectrum
# ============================================================

def optimize_dd_for_noise():
    """Find optimal DD parameters for given noise spectrum."""

    print("\n" + "="*60)
    print("DD Optimization for Noise Spectrum")
    print("="*60)

    # Simulate different noise correlation times
    total_time = 10e-6
    frequencies = np.linspace(0, 10e6, 1000) * 2 * np.pi

    # Lorentzian noise spectrum: S(ω) = S0 / (1 + (ωτc)²)
    tau_c_values = [1e-6, 5e-6, 20e-6]  # Correlation times

    plt.figure(figsize=(12, 8))

    for i, tau_c in enumerate(tau_c_values):
        S = 1 / (1 + (frequencies * tau_c)**2)

        # Compute overlap with different DD filter functions
        n_pulses_range = range(1, 21)
        overlaps = []

        for n_pulses in n_pulses_range:
            # CPMG pulse times
            pulse_times = [total_time * (2*j + 1) / (2 * n_pulses)
                          for j in range(n_pulses)]

            F = compute_filter_function(pulse_times, total_time, frequencies)

            # Overlap integral: ∫ S(ω) F(ω) / ω² dω
            dw = frequencies[1] - frequencies[0]
            # Avoid division by zero
            omega_safe = np.where(frequencies > 1, frequencies, 1)
            overlap = np.sum(S * F / omega_safe**2) * dw

            overlaps.append(overlap)

        plt.subplot(2, 2, i + 1)
        plt.plot(list(n_pulses_range), overlaps, 'bo-')
        plt.xlabel('Number of CPMG Pulses')
        plt.ylabel('Error Integral')
        plt.title(f'τc = {tau_c*1e6:.0f} μs')
        plt.grid(True, alpha=0.3)

        # Find optimal
        optimal_n = n_pulses_range[np.argmin(overlaps)]
        print(f"τc = {tau_c*1e6:.0f} μs: Optimal CPMG pulses = {optimal_n}")

    plt.subplot(2, 2, 4)
    for tau_c, color in zip(tau_c_values, ['blue', 'green', 'red']):
        S = 1 / (1 + (frequencies * tau_c)**2)
        plt.plot(frequencies / (2*np.pi*1e6), S / S.max(),
                label=f'τc = {tau_c*1e6:.0f} μs', color=color)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Normalized S(ω)')
    plt.title('Noise Spectra')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dd_optimization.png', dpi=150)
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Day 944: Dynamical Decoupling Lab")
    print("="*60)

    # Part 1: Compare DD sequences
    print("\n--- Part 1: DD Sequence Comparison ---")
    results = compare_dd_sequences()

    # Part 2: Filter function analysis
    print("\n--- Part 2: Filter Function Analysis ---")
    analyze_filter_functions()

    # Part 3: DD in quantum algorithms
    print("\n--- Part 3: DD in Quantum Circuits ---")
    demonstrate_dd_in_vqe()

    # Part 4: Concatenated DD
    print("\n--- Part 4: Concatenated DD ---")
    analyze_concatenated_dd()

    # Part 5: Optimization
    print("\n--- Part 5: DD Optimization ---")
    optimize_dd_for_noise()

    print("\n" + "="*60)
    print("Lab Complete! Key Takeaways:")
    print("  1. DD extends coherence by refocusing slow noise")
    print("  2. XY4/XY8 are robust against pulse errors")
    print("  3. More pulses help until pulse errors dominate")
    print("  4. Match DD to noise spectrum for best results")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Average Hamiltonian | $\bar{H} = \frac{1}{T}\int_0^T U^\dagger(t) H_{SE} U(t) dt$ |
| CPMG Error | $1 - F \propto (\tau / nT_2)^2$ |
| UDD Pulse Times | $t_j = T \sin^2(\pi j / (2n+2))$ |
| Filter Function | $F(\omega) = \|\sum_j (-1)^j e^{i\omega t_j}\|^2$ |
| Fidelity Decay | $1 - F = \frac{1}{\pi}\int \frac{S(\omega)}{\omega^2} F(\omega) d\omega$ |
| CDD Error | $1 - F \propto \epsilon^{2^n}$ |

### Main Takeaways

1. **DD averages out slow noise**: Refocusing pulses flip the effective sign of the system-environment coupling

2. **Standard sequences**: CPMG for general use, XY4/XY8 for pulse-error robustness, UDD for optimal pure dephasing suppression

3. **Filter function design**: DD sequences act as bandpass filters; match to noise spectrum for best performance

4. **Concatenation amplifies suppression**: Multi-level CDD achieves exponential error reduction at the cost of more pulses

5. **Integration with algorithms**: Insert DD during idle periods in quantum circuits to maintain coherence

## Daily Checklist

- [ ] I understand how refocusing pulses cancel dephasing
- [ ] I can implement CPMG, XY4, and UDD sequences
- [ ] I can compute and interpret filter functions
- [ ] I understand the trade-off between more pulses and pulse errors
- [ ] I know how to insert DD into quantum circuits
- [ ] I can choose appropriate DD for different noise environments

## Preview of Day 945

Tomorrow we conclude the week with **Virtual Distillation**, an advanced technique that uses multiple copies of a noisy state to simulate measurement on a purer state:

- Exponential error suppression via $\text{Tr}(\rho^n O) / \text{Tr}(\rho^n)$
- Sample complexity analysis
- Practical implementation strategies
- Limitations and hybrid approaches

Virtual distillation offers the strongest error suppression among mitigation techniques but requires additional quantum resources.
