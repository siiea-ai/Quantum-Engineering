# Day 939: Error Mitigation Overview - NISQ Limitations and Mitigation Philosophy

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | NISQ era challenges, mitigation vs correction fundamentals |
| Afternoon | 2 hours | Overhead analysis and technique comparison |
| Evening | 2 hours | Computational lab - benchmarking noisy circuits |

## Learning Objectives

By the end of this day, you will be able to:

1. **Characterize NISQ limitations** - Quantify the impact of gate errors, decoherence, and measurement errors on circuit fidelity
2. **Distinguish mitigation from correction** - Explain fundamental differences in approach, overhead, and guarantees
3. **Compare mitigation techniques** - Evaluate overhead, accuracy, and applicability of major methods
4. **Calculate circuit fidelity bounds** - Estimate maximum useful circuit depth for given error rates
5. **Design mitigation strategies** - Select appropriate techniques for specific problem characteristics
6. **Assess computational overhead** - Quantify sampling and classical processing costs

## Core Content

### 1. The NISQ Era and Its Challenges

The Noisy Intermediate-Scale Quantum (NISQ) era, coined by John Preskill in 2018, describes quantum computers with 50-1000+ qubits but without full error correction. These devices face fundamental limitations:

#### 1.1 Error Sources in NISQ Devices

**Gate Errors**: Single-qubit gates typically achieve fidelities of 99.9%, while two-qubit gates reach 99-99.9%:

$$F_{\text{gate}} = \text{Tr}(\rho_{\text{ideal}} \rho_{\text{actual}})$$

For a circuit with $n_1$ single-qubit and $n_2$ two-qubit gates:

$$\boxed{F_{\text{circuit}} \approx F_1^{n_1} \cdot F_2^{n_2}}$$

**Decoherence**: Characterized by relaxation time $T_1$ and dephasing time $T_2$:

$$\rho(t) = \begin{pmatrix} \rho_{00} + (1-e^{-t/T_1})\rho_{11} & \rho_{01}e^{-t/T_2} \\ \rho_{10}e^{-t/T_2} & \rho_{11}e^{-t/T_1} \end{pmatrix}$$

**Measurement Errors**: Described by confusion matrix $M$:

$$M = \begin{pmatrix} P(0|0) & P(0|1) \\ P(1|0) & P(1|1) \end{pmatrix}$$

#### 1.2 Circuit Depth Limitations

For a circuit to produce useful results, we need $F_{\text{circuit}} > 1/2$ (better than random):

$$n_1 \log F_1 + n_2 \log F_2 > \log(1/2)$$

For $F_1 = 0.999$ and $F_2 = 0.99$:

$$\boxed{n_{\text{max}} \approx \frac{0.693}{0.001 \cdot r_1 + 0.01 \cdot r_2}}$$

where $r_1, r_2$ are the ratios of gate types.

### 2. Error Correction vs Error Mitigation

#### 2.1 Quantum Error Correction (QEC)

QEC uses redundancy to detect and correct errors:

- **Overhead**: Requires $O(d^2)$ physical qubits per logical qubit for distance-$d$ codes
- **Threshold**: Works when physical error rate $p < p_{\text{th}}$ (typically $\sim 1\%$)
- **Guarantee**: Provides exponential suppression of logical errors

$$\boxed{p_L \propto \left(\frac{p}{p_{\text{th}}}\right)^{\lfloor (d+1)/2 \rfloor}}$$

**Current Status**: 2024-2025 demonstrations show logical qubits with $\sim 10^{-3}$ error rates, but overhead remains prohibitive for practical algorithms.

#### 2.2 Error Mitigation

Error mitigation trades classical computation for improved accuracy:

| Property | Error Correction | Error Mitigation |
|----------|------------------|------------------|
| Qubit overhead | $O(d^2)$ per logical qubit | None (uses physical qubits) |
| Scalability | Arbitrary depth circuits | Limited depth (polynomial) |
| Classical overhead | Syndrome decoding | Varies by technique |
| Sampling overhead | Minimal | Can be exponential |
| Guarantee | Fault-tolerant | Approximate improvement |

### 3. Overview of Error Mitigation Techniques

#### 3.1 Zero-Noise Extrapolation (ZNE)

Run circuits at multiple noise levels and extrapolate to zero noise:

$$\boxed{\langle O \rangle_0 = \lim_{\lambda \to 0} f(\lambda)}$$

where $f(\lambda)$ is fitted to data at noise levels $\lambda_1, \lambda_2, \ldots$

**Overhead**: Multiplicative factor of $k$ (number of noise levels) in circuit executions.

#### 3.2 Probabilistic Error Cancellation (PEC)

Represent ideal operations as quasi-probability distributions over noisy operations:

$$\boxed{\mathcal{U}_{\text{ideal}} = \sum_i c_i \mathcal{O}_i, \quad \sum_i c_i = 1, \quad c_i \in \mathbb{R}}$$

**Overhead**: Sampling cost scales as $C = (\sum_i |c_i|)^2$, exponential in circuit depth.

#### 3.3 Symmetry Verification

Exploit known symmetries to detect errors:

$$\boxed{P(E|\text{symmetry violated}) > P(E|\text{symmetry preserved})}$$

Post-select on symmetric subspace to discard erroneous results.

#### 3.4 Measurement Error Mitigation

Characterize and invert the measurement confusion matrix:

$$\boxed{\mathbf{p}_{\text{ideal}} = M^{-1} \mathbf{p}_{\text{noisy}}}$$

#### 3.5 Dynamical Decoupling (DD)

Apply pulse sequences to average out noise during idle periods:

$$\boxed{H_{\text{eff}} = \frac{1}{T}\int_0^T U^\dagger(t) H_{\text{noise}} U(t) dt \approx 0}$$

#### 3.6 Virtual Distillation

Use multiple copies of a noisy state to simulate a purer state:

$$\boxed{\langle O \rangle_{\text{pure}} = \frac{\text{Tr}(\rho^n O)}{\text{Tr}(\rho^n)}}$$

### 4. Overhead Comparison and Trade-offs

#### 4.1 Sampling Overhead

| Technique | Sampling Overhead | Classical Overhead |
|-----------|-------------------|-------------------|
| ZNE | $O(k)$ | Polynomial fitting |
| PEC | $O(e^{\gamma d})$ | Noise tomography |
| Symmetry | $O(1/P_{\text{accept}})$ | Symmetry checking |
| Measurement | $O(1)$ | Matrix inversion |
| DD | $O(1)$ | Pulse optimization |
| Virtual Distillation | $O(1/\text{Tr}(\rho^n))$ | State preparation |

#### 4.2 Error Regime Applicability

For gate error rate $\epsilon$ and circuit depth $d$:

$$\text{Effective error} \approx \epsilon \cdot d$$

| Technique | Effective Error Range |
|-----------|----------------------|
| ZNE | $\epsilon d \lesssim 0.3$ |
| PEC | $\epsilon d \lesssim 0.5$ |
| Symmetry | Any (probabilistic) |
| Virtual Distillation | $\epsilon d \lesssim 1$ |

### 5. Combining Multiple Techniques

Error mitigation techniques are often combined for maximum benefit:

$$\boxed{\text{Pipeline: DD} \to \text{Circuit} \to \text{ZNE/PEC} \to \text{Measurement Mitigation}}$$

The composition order matters:
1. **DD**: Applied during circuit execution (reduces coherent errors)
2. **ZNE/PEC**: Applied to expectation values (reduces gate errors)
3. **Measurement Mitigation**: Applied to final outcomes (reduces readout errors)

## Quantum Computing Applications

### NISQ Algorithm Performance

For variational quantum eigensolvers (VQE) targeting chemical accuracy ($\Delta E < 1.6$ mHartree):

$$\Delta E_{\text{noisy}} = \Delta E_{\text{ideal}} + \sum_{\text{errors}} \epsilon_i \cdot \text{sensitivity}_i$$

Error mitigation enables chemical accuracy on NISQ devices for small molecules (H$_2$, LiH, BeH$_2$) that would otherwise be out of reach.

### Quantum Advantage Timeline

With error mitigation, useful quantum advantage may be achieved:
- **2024-2025**: Demonstrated improvements in VQE, QAOA
- **2025-2027**: Error mitigation + early logical qubits
- **2028+**: Full error correction for larger problems

## Worked Examples

### Example 1: Circuit Fidelity Estimation

**Problem**: A quantum circuit has 50 single-qubit gates ($F_1 = 0.999$) and 30 two-qubit gates ($F_2 = 0.99$). Estimate the circuit fidelity.

**Solution**:

$$F_{\text{circuit}} \approx F_1^{50} \cdot F_2^{30}$$

$$= 0.999^{50} \cdot 0.99^{30}$$

$$= 0.951 \cdot 0.740$$

$$\boxed{F_{\text{circuit}} \approx 0.704}$$

This circuit retains ~70% fidelity, marginal for useful computation without mitigation.

### Example 2: Maximum Circuit Depth

**Problem**: With $F_2 = 0.995$ for two-qubit gates and a target fidelity of 0.8, what is the maximum number of two-qubit gates?

**Solution**:

$$F_2^{n_2} \geq 0.8$$

$$n_2 \cdot \log(0.995) \geq \log(0.8)$$

$$n_2 \leq \frac{\log(0.8)}{\log(0.995)}$$

$$n_2 \leq \frac{-0.223}{-0.00501}$$

$$\boxed{n_2 \leq 44}$$

### Example 3: PEC Overhead Estimation

**Problem**: If each gate's quasi-probability norm is $\gamma = 1.1$, what is the sampling overhead for a 20-gate circuit?

**Solution**:

Total quasi-probability norm:
$$\gamma_{\text{total}} = \gamma^{20} = 1.1^{20} = 6.73$$

Sampling overhead:
$$C = \gamma_{\text{total}}^2 = 6.73^2$$

$$\boxed{C \approx 45}$$

Need 45 times more shots than ideal to achieve same statistical precision.

## Practice Problems

### Level 1: Direct Application

1. Calculate the circuit fidelity for 100 single-qubit gates ($F_1 = 0.9995$) and 80 two-qubit gates ($F_2 = 0.995$).

2. If the measurement confusion matrix is $M = \begin{pmatrix} 0.98 & 0.03 \\ 0.02 & 0.97 \end{pmatrix}$, what is $M^{-1}$?

3. For $T_2 = 100 \mu s$ and gate time $t_g = 100$ ns, how many gates can execute before coherence drops below 0.9?

### Level 2: Intermediate

4. A VQE circuit with 50 two-qubit gates achieves energy estimate $E_{\text{noisy}} = -1.12$ Ha. If ZNE extrapolation from noise levels $(1, 1.5, 2)$ gives readings $(-1.12, -1.08, -1.02)$ Ha, estimate the zero-noise energy.

5. Compare the total overhead (sampling + classical) for ZNE vs PEC on a 30-gate circuit with $\gamma = 1.05$ per gate.

6. Design a symmetry verification scheme for a 4-qubit state that should conserve total $Z$ magnetization.

### Level 3: Challenging

7. Derive the optimal number of noise levels for ZNE given a total shot budget $N$ and per-level measurement variance $\sigma^2$.

8. For a noise model $\mathcal{E}(\rho) = (1-p)\rho + p X\rho X$, find the quasi-probability decomposition for implementing identity.

9. Analyze the break-even point where error mitigation overhead exceeds the cost of error correction for a surface code with threshold $p_{\text{th}} = 0.01$.

## Computational Lab

```python
"""
Day 939: Error Mitigation Overview - Benchmarking NISQ Limitations
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# Part 1: Circuit Fidelity Degradation Analysis
# ============================================================

def create_random_circuit(n_qubits: int, depth: int) -> QuantumCircuit:
    """Create a random circuit for fidelity testing."""
    qc = QuantumCircuit(n_qubits)

    for d in range(depth):
        # Single-qubit layer
        for q in range(n_qubits):
            gate = np.random.choice(['h', 'x', 'y', 'z', 's', 't'])
            getattr(qc, gate)(q)

        # Two-qubit layer (linear connectivity)
        for q in range(0, n_qubits - 1, 2 if d % 2 == 0 else 1):
            if q + 1 < n_qubits:
                qc.cx(q, q + 1)

    return qc

def build_noise_model(p1: float = 0.001, p2: float = 0.01,
                      t1: float = 50e-6, t2: float = 70e-6,
                      gate_time_1q: float = 50e-9,
                      gate_time_2q: float = 300e-9) -> NoiseModel:
    """Build a realistic noise model."""
    noise_model = NoiseModel()

    # Depolarizing errors
    error_1q = depolarizing_error(p1, 1)
    error_2q = depolarizing_error(p2, 2)

    # Thermal relaxation
    thermal_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    thermal_2q = thermal_relaxation_error(t1, t2, gate_time_2q)

    # Compose errors
    combined_1q = error_1q.compose(thermal_1q)
    combined_2q = error_2q.compose(thermal_2q)

    # Apply to all gates
    noise_model.add_all_qubit_quantum_error(combined_1q, ['h', 'x', 'y', 'z', 's', 't'])
    noise_model.add_all_qubit_quantum_error(combined_2q, ['cx', 'cz'])

    return noise_model

def analyze_fidelity_vs_depth():
    """Analyze circuit fidelity degradation with depth."""
    n_qubits = 4
    depths = [1, 5, 10, 15, 20, 25, 30, 40, 50]
    shots = 8192

    # Create noise model
    noise_model = build_noise_model(p1=0.001, p2=0.01)

    # Simulators
    ideal_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=noise_model)

    fidelities = []

    for depth in depths:
        # Create circuit with measurement
        qc = create_random_circuit(n_qubits, depth)
        qc.measure_all()

        # Run ideal simulation
        ideal_result = ideal_sim.run(qc, shots=shots).result()
        ideal_counts = ideal_result.get_counts()

        # Run noisy simulation
        noisy_result = noisy_sim.run(qc, shots=shots).result()
        noisy_counts = noisy_result.get_counts()

        # Calculate classical fidelity (Bhattacharyya coefficient)
        all_bitstrings = set(ideal_counts.keys()) | set(noisy_counts.keys())
        fidelity = 0
        for bs in all_bitstrings:
            p_ideal = ideal_counts.get(bs, 0) / shots
            p_noisy = noisy_counts.get(bs, 0) / shots
            fidelity += np.sqrt(p_ideal * p_noisy)

        fidelities.append(fidelity)
        print(f"Depth {depth:3d}: Fidelity = {fidelity:.4f}")

    # Fit exponential decay
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    popt, _ = curve_fit(exp_decay, depths, fidelities, p0=[1, 0.05])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(depths, fidelities, s=100, c='blue', label='Measured Fidelity')

    x_fit = np.linspace(0, max(depths), 100)
    plt.plot(x_fit, exp_decay(x_fit, *popt), 'r--',
             label=f'Fit: F = {popt[0]:.3f} exp(-{popt[1]:.4f} d)')

    plt.axhline(y=0.5, color='green', linestyle=':', label='Quantum advantage threshold')
    plt.xlabel('Circuit Depth', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title('Circuit Fidelity vs Depth (NISQ Device Simulation)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fidelity_vs_depth.png', dpi=150)
    plt.show()

    # Estimate maximum useful depth
    max_depth = -np.log(0.5) / popt[1]
    print(f"\nMaximum useful depth (F > 0.5): {max_depth:.1f} layers")

    return depths, fidelities, popt

# ============================================================
# Part 2: Error Source Decomposition
# ============================================================

def decompose_error_sources():
    """Analyze contribution of different error sources."""
    n_qubits = 4
    depth = 20
    shots = 8192

    # Reference: no noise
    ideal_sim = AerSimulator()

    # Different noise models
    noise_configs = {
        'No Noise': NoiseModel(),
        'Gate Errors Only': build_noise_model(p1=0.001, p2=0.01, t1=1e6, t2=1e6),
        'Decoherence Only': build_noise_model(p1=0, p2=0, t1=50e-6, t2=70e-6),
        'Full Noise': build_noise_model(p1=0.001, p2=0.01, t1=50e-6, t2=70e-6)
    }

    qc = create_random_circuit(n_qubits, depth)
    qc.measure_all()

    results = {}

    # Get ideal distribution
    ideal_result = ideal_sim.run(qc, shots=shots).result()
    ideal_counts = ideal_result.get_counts()

    for name, noise_model in noise_configs.items():
        if name == 'No Noise':
            results[name] = 1.0
            continue

        sim = AerSimulator(noise_model=noise_model)
        result = sim.run(qc, shots=shots).result()
        noisy_counts = result.get_counts()

        # Calculate fidelity
        all_bitstrings = set(ideal_counts.keys()) | set(noisy_counts.keys())
        fidelity = sum(np.sqrt(ideal_counts.get(bs, 0) * noisy_counts.get(bs, 0))
                      for bs in all_bitstrings) / shots

        results[name] = fidelity

    # Visualize
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = list(results.values())
    colors = ['green', 'blue', 'orange', 'red']

    bars = plt.bar(names, values, color=colors, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12)

    plt.ylabel('Fidelity', fontsize=12)
    plt.title(f'Error Source Contribution (Depth={depth})', fontsize=14)
    plt.ylim(0, 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_sources.png', dpi=150)
    plt.show()

    return results

# ============================================================
# Part 3: Mitigation Technique Comparison (Preview)
# ============================================================

def preview_mitigation_impact():
    """Preview the impact of error mitigation (detailed in later days)."""
    print("\n" + "="*60)
    print("Error Mitigation Technique Comparison (Preview)")
    print("="*60)

    # Simulated improvement factors for different techniques
    techniques = {
        'No Mitigation': {'improvement': 1.0, 'overhead': 1},
        'Measurement Mitigation': {'improvement': 1.3, 'overhead': 2},
        'Dynamical Decoupling': {'improvement': 1.5, 'overhead': 1},
        'Zero-Noise Extrapolation': {'improvement': 2.0, 'overhead': 3},
        'Probabilistic Error Cancellation': {'improvement': 3.0, 'overhead': 50},
        'Virtual Distillation': {'improvement': 4.0, 'overhead': 100}
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Improvement factors
    names = list(techniques.keys())
    improvements = [t['improvement'] for t in techniques.values()]
    overheads = [t['overhead'] for t in techniques.values()]

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(techniques)))

    ax1.barh(names, improvements, color=colors)
    ax1.set_xlabel('Fidelity Improvement Factor', fontsize=12)
    ax1.set_title('Error Mitigation Improvement', fontsize=14)
    ax1.set_xlim(0, 5)

    for i, v in enumerate(improvements):
        ax1.text(v + 0.1, i, f'{v:.1f}x', va='center')

    # Overhead comparison
    ax2.barh(names, overheads, color=colors)
    ax2.set_xlabel('Sampling Overhead (multiplier)', fontsize=12)
    ax2.set_title('Computational Overhead', fontsize=14)
    ax2.set_xscale('log')

    for i, v in enumerate(overheads):
        ax2.text(v * 1.1, i, f'{v}x', va='center')

    plt.tight_layout()
    plt.savefig('mitigation_comparison.png', dpi=150)
    plt.show()

    # Print efficiency analysis
    print("\nEfficiency Analysis (Improvement / Overhead):")
    for name, data in techniques.items():
        efficiency = data['improvement'] / np.sqrt(data['overhead'])
        print(f"  {name:35s}: {efficiency:.3f}")

# ============================================================
# Part 4: NISQ Algorithm Simulation
# ============================================================

def simulate_vqe_with_noise():
    """Simulate VQE performance with and without noise."""
    from qiskit.circuit import Parameter

    # Simple 2-qubit VQE ansatz for H2
    theta = Parameter('θ')

    def create_ansatz(theta_val):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.ry(theta_val, 0)
        qc.ry(theta_val, 1)
        qc.cx(0, 1)
        return qc

    # Pauli terms for H2 Hamiltonian (simplified)
    # H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
    h2_coeffs = {
        'II': -0.81054,
        'ZI': 0.17218,
        'IZ': 0.17218,
        'ZZ': -0.22575,
        'XX': 0.16892,
        'YY': 0.16892
    }

    def measure_pauli(qc, pauli_string, simulator, shots=8192):
        """Measure a Pauli string expectation value."""
        meas_qc = qc.copy()

        for i, p in enumerate(pauli_string[::-1]):
            if p == 'X':
                meas_qc.h(i)
            elif p == 'Y':
                meas_qc.sdg(i)
                meas_qc.h(i)

        meas_qc.measure_all()

        result = simulator.run(meas_qc, shots=shots).result()
        counts = result.get_counts()

        expectation = 0
        for bitstring, count in counts.items():
            parity = sum(int(b) for b in bitstring) % 2
            expectation += (-1)**parity * count

        return expectation / shots

    # Sweep theta values
    theta_range = np.linspace(0, 2*np.pi, 30)

    ideal_sim = AerSimulator()
    noisy_sim = AerSimulator(noise_model=build_noise_model(p1=0.002, p2=0.02))

    ideal_energies = []
    noisy_energies = []

    for theta_val in theta_range:
        qc = create_ansatz(theta_val)

        ideal_energy = sum(coeff * measure_pauli(qc, pauli, ideal_sim)
                          for pauli, coeff in h2_coeffs.items())

        noisy_energy = sum(coeff * measure_pauli(qc, pauli, noisy_sim)
                          for pauli, coeff in h2_coeffs.items())

        ideal_energies.append(ideal_energy)
        noisy_energies.append(noisy_energy)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(theta_range, ideal_energies, 'b-', linewidth=2, label='Ideal')
    plt.plot(theta_range, noisy_energies, 'r--', linewidth=2, label='Noisy')

    plt.axhline(y=-1.137, color='green', linestyle=':', label='Exact ground state')

    plt.xlabel('θ (radians)', fontsize=12)
    plt.ylabel('Energy (Hartree)', fontsize=12)
    plt.title('VQE Energy Landscape: Ideal vs Noisy', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('vqe_noise_comparison.png', dpi=150)
    plt.show()

    # Calculate error at minimum
    ideal_min_idx = np.argmin(ideal_energies)
    noisy_min_idx = np.argmin(noisy_energies)

    print(f"\nVQE Results:")
    print(f"  Ideal minimum energy: {ideal_energies[ideal_min_idx]:.4f} Ha at θ = {theta_range[ideal_min_idx]:.3f}")
    print(f"  Noisy minimum energy: {noisy_energies[noisy_min_idx]:.4f} Ha at θ = {theta_range[noisy_min_idx]:.3f}")
    print(f"  Error due to noise: {noisy_energies[noisy_min_idx] - ideal_energies[ideal_min_idx]:.4f} Ha")
    print(f"  Chemical accuracy (1.6 mHa): {'Achieved' if abs(noisy_energies[noisy_min_idx] - (-1.137)) < 0.0016 else 'NOT achieved'}")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Day 939: Error Mitigation Overview Lab")
    print("="*60)

    # Part 1: Fidelity analysis
    print("\n--- Part 1: Fidelity vs Depth Analysis ---")
    depths, fidelities, fit_params = analyze_fidelity_vs_depth()

    # Part 2: Error source decomposition
    print("\n--- Part 2: Error Source Decomposition ---")
    error_contributions = decompose_error_sources()

    # Part 3: Mitigation preview
    print("\n--- Part 3: Mitigation Technique Comparison ---")
    preview_mitigation_impact()

    # Part 4: VQE simulation
    print("\n--- Part 4: VQE Noise Impact ---")
    simulate_vqe_with_noise()

    print("\n" + "="*60)
    print("Lab Complete! Key Takeaways:")
    print("  1. Circuit fidelity decays exponentially with depth")
    print("  2. Two-qubit gate errors dominate for most circuits")
    print("  3. Error mitigation can improve results without overhead")
    print("  4. Choice of technique depends on error regime and resources")
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Circuit Fidelity | $F_{\text{circuit}} \approx F_1^{n_1} \cdot F_2^{n_2}$ |
| Maximum Depth | $n_{\text{max}} \approx \frac{\log(F_{\text{target}})}{\log(F_{\text{gate}})}$ |
| QEC Logical Error | $p_L \propto (p/p_{\text{th}})^{(d+1)/2}$ |
| PEC Overhead | $C = (\sum_i |c_i|)^2$ |
| Measurement Error | $\mathbf{p}_{\text{noisy}} = M \cdot \mathbf{p}_{\text{ideal}}$ |

### Main Takeaways

1. **NISQ limitations are fundamental**: Gate errors, decoherence, and measurement errors limit useful circuit depth to ~50-100 layers with current technology

2. **Mitigation vs Correction trade-off**: Error correction provides guarantees but requires massive overhead; mitigation trades classical resources for approximate improvement

3. **Multiple techniques exist**: ZNE, PEC, symmetry verification, measurement mitigation, DD, and virtual distillation each have different overhead-accuracy profiles

4. **Combination is key**: Real applications often combine multiple mitigation techniques in a pipeline

5. **Error regime matters**: Technique choice depends on dominant error sources and circuit characteristics

## Daily Checklist

- [ ] I can calculate circuit fidelity given gate error rates
- [ ] I understand the difference between error correction and mitigation
- [ ] I can compare overhead and applicability of different mitigation techniques
- [ ] I can identify when each mitigation technique is most appropriate
- [ ] I can estimate the maximum useful circuit depth for a given noise model
- [ ] I understand how error mitigation enables near-term quantum applications

## Preview of Day 940

Tomorrow we dive deep into **Zero-Noise Extrapolation (ZNE)**, the most widely used error mitigation technique. We'll cover:

- Noise scaling methods (pulse stretching, gate folding)
- Richardson extrapolation and polynomial fitting
- Error analysis and optimal extrapolation strategies
- Qiskit/Mitiq implementation of ZNE for VQE circuits

ZNE provides a powerful way to estimate zero-noise expectation values with only linear overhead in circuit executions.
