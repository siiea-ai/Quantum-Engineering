# Day 919: Gate Fidelity Benchmarks

## Schedule Overview

| Time Block | Duration | Topic |
|------------|----------|-------|
| Morning | 3 hours | Benchmarking protocols and theory |
| Afternoon | 2.5 hours | Platform fidelity analysis |
| Evening | 1.5 hours | Computational lab: RB and XEB implementation |

## Learning Objectives

By the end of today, you will be able to:

1. Implement and analyze randomized benchmarking (RB) protocols
2. Understand cross-entropy benchmarking (XEB) and its applications
3. Apply cycle benchmarking for multi-qubit error analysis
4. Compare state-of-the-art gate fidelities across platforms
5. Distinguish between different error metrics and their relationships
6. Evaluate benchmarking protocols appropriate for different platforms

## Core Content

### 1. Fundamentals of Gate Fidelity

#### Average Gate Fidelity

The average gate fidelity between an ideal gate $U$ and noisy implementation $\mathcal{E}$ is:

$$F_{avg}(U, \mathcal{E}) = \int d\psi \langle\psi|U^\dagger \mathcal{E}(|\psi\rangle\langle\psi|) U|\psi\rangle$$

For a d-dimensional system:

$$\boxed{F_{avg} = \frac{d \cdot F_{ent} + 1}{d + 1}}$$

where $F_{ent}$ is the entanglement fidelity:

$$F_{ent} = \langle\Phi|(\mathcal{I} \otimes \mathcal{E})(|\Phi\rangle\langle\Phi|)|\Phi\rangle$$

with $|\Phi\rangle = \frac{1}{\sqrt{d}}\sum_i |i\rangle|i\rangle$ being the maximally entangled state.

#### Error Rate Definitions

Average error rate:
$$r = 1 - F_{avg}$$

Infidelity:
$$\epsilon = 1 - F_{avg}$$

Diamond norm distance (worst-case):
$$\|\mathcal{E} - \mathcal{U}\|_\diamond = \max_\rho \|\mathcal{E}(\rho) - \mathcal{U}(\rho)\|_1$$

Relation for depolarizing channel:
$$\boxed{\frac{d}{d+1}\epsilon_{avg} \leq \epsilon_\diamond \leq d \cdot \epsilon_{avg}}$$

### 2. Randomized Benchmarking (RB)

#### Standard Clifford RB

The protocol generates random sequences of Clifford gates with a recovery gate:

1. Prepare $|0\rangle$
2. Apply $m$ random Clifford gates $C_1, C_2, \ldots, C_m$
3. Apply recovery gate $C_{m+1} = (C_m \cdots C_1)^\dagger$
4. Measure survival probability

The survival probability decays as:

$$\boxed{F(m) = A \cdot p^m + B}$$

where:
- $p$ is the depolarizing parameter
- $A$ captures SPAM errors
- $B$ is the asymptotic value (1/d for complete depolarization)

The error per Clifford (EPC) is:

$$r = \frac{(d-1)(1-p)}{d}$$

For single qubit (d=2): $r = (1-p)/2$

#### Interleaved RB

To characterize a specific gate $G$:

1. Run standard RB → obtain $p_{ref}$
2. Run RB with $G$ interleaved between each Clifford → obtain $p_G$

The gate error is:
$$r_G = \frac{(d-1)(1 - p_G/p_{ref})}{d}$$

#### Simultaneous RB

For multi-qubit systems, characterize crosstalk:

$$F(m) = A \cdot p_{single}^m \cdot p_{crosstalk}^m + B$$

The crosstalk contribution is isolated by comparing single-qubit and simultaneous RB.

### 3. Cross-Entropy Benchmarking (XEB)

#### Linear XEB (Google's Protocol)

For random circuit $C$ producing output distribution $p_C(x)$:

$$\chi_{XEB} = 2^n \langle p_C(x) \rangle_{x \sim \mathcal{E}(C)} - 1$$

where the average is over bitstrings $x$ sampled from the noisy implementation $\mathcal{E}(C)$.

For ideal implementation: $\chi_{XEB} = 1$
For uniform random: $\chi_{XEB} = 0$

#### Fidelity Estimation

Under depolarizing noise:
$$\boxed{F_{XEB} \approx \chi_{XEB} = F_{circuit}}$$

For a circuit with $n$ qubits and depth $d$:
$$F_{XEB} \approx (1-\epsilon_{1Q})^{n_{1Q}} \cdot (1-\epsilon_{2Q})^{n_{2Q}} \cdot e^{-T/T_2}$$

#### XEB Protocol Steps

1. Generate $K$ random circuits of depth $d$
2. For each circuit, sample $N$ bitstrings from hardware
3. Calculate XEB fidelity:
$$\chi = \frac{1}{K} \sum_k \left[\frac{2^n}{N_k} \sum_{x_i \sim \text{hardware}} p_k(x_i) - 1\right]$$

4. Fit decay: $\chi(d) = A \cdot f^d$

### 4. Cycle Benchmarking

#### Motivation

Standard RB assumes gate-independent errors. Cycle benchmarking captures:
- Context-dependent errors
- Crosstalk during parallel operations
- Correlated errors across qubits

#### Protocol

1. Define a "cycle" = layer of parallel gates
2. Implement Pauli twirling: sandwich cycle between random Paulis
3. Measure decay of process fidelity

$$F_{cycle}(m) = A \cdot p_{cycle}^m + B$$

#### Dressed Cycle Error

The error per cycle captures all imperfections during parallel execution:

$$\epsilon_{cycle} = \frac{(4^n - 1)(1 - p_{cycle})}{4^n}$$

### 5. Platform-Specific Fidelity Analysis

#### Superconducting Qubits

**State-of-the-Art (2024):**
| Gate | Best Fidelity | Typical | Lab |
|------|---------------|---------|-----|
| 1Q | 99.99% | 99.9% | IBM, Google |
| CZ | 99.9% | 99.5% | Google |
| iSWAP | 99.8% | 99.4% | Google |
| ECR | 99.7% | 99.3% | IBM |

**Error Budget:**
$$\epsilon_{total} = \epsilon_{coherence} + \epsilon_{control} + \epsilon_{leakage}$$

Typical breakdown:
- Coherence: 30-50% of error
- Control imperfections: 30-40%
- Leakage to non-computational states: 10-20%

**Scaling Challenges:**
- Frequency crowding: $\epsilon_{crosstalk} \propto (J/\Delta)^2$
- Simultaneous operations: $\epsilon_{simult} > \epsilon_{single}$

#### Trapped Ions

**State-of-the-Art (2024):**
| Gate | Best Fidelity | Typical | Lab |
|------|---------------|---------|-----|
| 1Q | 99.9999% | 99.99% | Oxford, NIST |
| MS (2Q) | 99.92% | 99.5% | IonQ, Quantinuum |

**Error Budget:**
$$\epsilon_{MS} = \epsilon_{heating} + \epsilon_{laser} + \epsilon_{detuning} + \epsilon_{spontaneous}$$

Typical breakdown:
- Motional heating: 20-40%
- Laser intensity/phase: 30-40%
- Off-resonant coupling: 10-20%
- Spontaneous emission: 5-10%

**Scaling Challenges:**
- Mode spectral crowding: N modes for N ions
- Addressing crosstalk: $\epsilon_{crosstalk} \propto (\Omega_{neighbor}/\delta)^2$

#### Neutral Atoms

**State-of-the-Art (2024):**
| Gate | Best Fidelity | Typical | Lab |
|------|---------------|---------|-----|
| 1Q | 99.97% | 99.5% | Harvard, QuEra |
| CZ (Rydberg) | 99.5% | 99% | Lukin group |

**Error Budget:**
$$\epsilon_{CZ} = \epsilon_{Rydberg} + \epsilon_{laser} + \epsilon_{position} + \epsilon_{decay}$$

Typical breakdown:
- Rydberg decay: 20-30%
- Laser imperfections: 30-40%
- Atom position uncertainty: 20-30%
- Doppler shifts: 10-20%

**Scaling Challenges:**
- Global addressing: individual control limited
- Rydberg blockade leakage: $\epsilon \propto (\Omega/V_{dd})^2$

### 6. Comparative Analysis

#### Fidelity vs Gate Time Trade-off

$$\text{Operations per coherence time} = \frac{T_2}{t_{gate}}$$

| Platform | t_gate (2Q) | F_gate | Ops/T2 | Effective Quality |
|----------|-------------|--------|--------|-------------------|
| SC | 50 ns | 99.5% | 2000 | High throughput |
| TI | 200 μs | 99.9% | 5000 | High fidelity |
| NA | 1 μs | 99.5% | 500k | Balanced |

#### Error Model Comparison

**Superconducting:** Predominantly depolarizing + amplitude damping
$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z) + \text{AD terms}$$

**Trapped Ion:** Coherent + dephasing dominated
$$\mathcal{E}(\rho) = e^{-i\theta\sigma_z}\rho e^{i\theta\sigma_z} + \text{dephasing}$$

**Neutral Atom:** Stochastic + loss
$$\mathcal{E}(\rho) = (1-p_{loss})\rho_{ideal} + p_{loss}|loss\rangle\langle loss|$$

## Quantum Computing Applications

### Error Budget for Algorithms

For a quantum algorithm with $n_{1Q}$ single-qubit gates and $n_{2Q}$ two-qubit gates:

$$F_{algorithm} \approx (1-\epsilon_{1Q})^{n_{1Q}} \cdot (1-\epsilon_{2Q})^{n_{2Q}}$$

For high fidelity ($F > 0.5$):
$$n_{2Q,max} \approx \frac{\ln(2)}{\epsilon_{2Q}}$$

| Platform | ε_2Q | n_2Q (F>50%) | n_2Q (F>90%) |
|----------|------|--------------|--------------|
| SC | 0.5% | 138 | 21 |
| TI | 0.1% | 693 | 105 |
| NA | 0.5% | 138 | 21 |

### Quantum Volume Estimation

$$\log_2(QV) = \max_d \{d : F(d) > 2/3\}$$

where $d$ is the circuit depth achievable on $d$ qubits.

For heavy output generation:
$$F(d) = (F_{1Q})^{d} \cdot (F_{2Q})^{d \cdot (d-1)/2} \cdot e^{-d \cdot t_{layer}/T_2}$$

## Worked Examples

### Example 1: RB Data Analysis

**Problem:** RB data shows survival probabilities: {m=1: 0.99, m=10: 0.91, m=50: 0.62, m=100: 0.39, m=200: 0.15}. Extract the error per Clifford.

**Solution:**

1. Fit to exponential model $F(m) = A \cdot p^m + B$:
   - Using nonlinear least squares
   - Initial guess: A = 0.98, p = 0.99, B = 0.01

2. Linearize for initial estimate:
   $$\ln(F - B) = \ln(A) + m \cdot \ln(p)$$

   Taking B ≈ 0:
   - slope = ln(p) ≈ ln(0.39)/100 = -0.0094
   - p ≈ 0.9906

3. Refine fit with proper minimization:
   - Best fit: A = 0.99, p = 0.9905, B = 0.01

4. Calculate EPC:
   $$r = \frac{1-p}{2} = \frac{1-0.9905}{2} = 0.00475 = 0.475\%$$

**Answer:** Error per Clifford = 0.475%

### Example 2: XEB Fidelity Estimation

**Problem:** A 5-qubit random circuit of depth 20 has XEB fidelity χ = 0.15. The circuit contains 100 single-qubit gates and 80 two-qubit gates. Estimate individual gate errors assuming ε_1Q = ε_2Q/10.

**Solution:**

1. Circuit fidelity model:
   $$F = (1-\epsilon_{1Q})^{100} \cdot (1-\epsilon_{2Q})^{80}$$

2. For small errors:
   $$\ln(F) \approx -100\epsilon_{1Q} - 80\epsilon_{2Q}$$

3. With constraint ε_1Q = ε_2Q/10:
   $$\ln(0.15) = -10\epsilon_{2Q} - 80\epsilon_{2Q} = -90\epsilon_{2Q}$$

4. Solve:
   $$\epsilon_{2Q} = \frac{-\ln(0.15)}{90} = \frac{1.897}{90} = 0.0211 = 2.11\%$$
   $$\epsilon_{1Q} = 0.211\%$$

**Answer:** ε_1Q ≈ 0.21%, ε_2Q ≈ 2.1%

### Example 3: Interleaved RB Gate Error

**Problem:** Standard RB gives p_ref = 0.9985. Interleaved RB with CNOT gives p_CNOT = 0.9970. Calculate the CNOT error rate.

**Solution:**

1. Reference EPC:
   $$r_{ref} = \frac{3(1-p_{ref})}{4} = \frac{3(0.0015)}{4} = 0.001125$$

2. Interleaved ratio:
   $$\frac{p_{CNOT}}{p_{ref}} = \frac{0.9970}{0.9985} = 0.9985$$

3. CNOT error:
   $$r_{CNOT} = \frac{3(1 - p_{CNOT}/p_{ref})}{4} = \frac{3(0.0015)}{4} = 0.001125$$

4. For two-qubit system (d=4):
   $$r_{CNOT} = \frac{15(1 - p_{CNOT}/p_{ref})}{16} = \frac{15(0.0015)}{16} = 0.00141$$

**Answer:** CNOT error rate ≈ 0.14%

## Practice Problems

### Level 1: Direct Application

1. RB yields p = 0.995 for single-qubit Cliffords. Calculate the average gate error assuming each Clifford is composed of 1.5 physical gates on average.

2. XEB fidelity is 0.82 for a 3-qubit, depth-10 circuit with 30 single-qubit and 20 two-qubit gates. If ε_1Q = 0.1%, estimate ε_2Q.

3. Convert average error rate ε_avg = 0.3% to diamond norm bounds for a single-qubit gate.

### Level 2: Intermediate Analysis

4. Design an interleaved RB experiment to characterize a CZ gate with expected error ~0.5%. How many sequence lengths and samples per length are needed for 10% relative precision?

5. A superconducting processor shows r_single = 0.1% (single-qubit RB) but r_simult = 0.3% (simultaneous RB). Quantify the crosstalk contribution and its impact on circuit fidelity.

6. Compare the Quantum Volume expected for:
   - SC: 100 qubits, 99.5% 2Q fidelity, T2 = 100 μs, t_layer = 500 ns
   - TI: 30 qubits, 99.9% 2Q fidelity, T2 = 1 s, t_layer = 1 ms

### Level 3: Advanced Research-Level

7. Derive the relationship between RB decay parameter p and the Pauli transfer matrix representation of the average noise channel.

8. Design a cycle benchmarking protocol to isolate correlated two-qubit errors from independent single-qubit errors in a 4-qubit system.

9. For a leaky qubit (3-level system), derive the modified RB decay model including leakage rate γ and seepage rate λ.

## Computational Lab: Benchmarking Analysis

```python
"""
Day 919 Computational Lab: Gate Fidelity Benchmarking
Implements RB and XEB analysis for platform comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
from typing import Tuple, List

# Set plotting style
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# =============================================================================
# Part 1: Randomized Benchmarking Implementation
# =============================================================================

def rb_decay_model(m, A, p, B):
    """Standard RB exponential decay model"""
    return A * p**m + B

def generate_rb_data(error_per_clifford: float,
                     sequence_lengths: np.ndarray,
                     samples_per_length: int = 100,
                     spam_error: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic RB data

    Parameters:
    -----------
    error_per_clifford: Average error rate per Clifford gate
    sequence_lengths: Array of sequence lengths m
    samples_per_length: Number of random sequences per length
    spam_error: State preparation and measurement error

    Returns:
    --------
    mean_fidelity: Mean survival probability at each length
    std_fidelity: Standard deviation at each length
    """
    # Convert EPC to depolarizing parameter
    p = 1 - 2 * error_per_clifford  # For single qubit

    # SPAM parameters
    A = 1 - spam_error
    B = 0.5 * spam_error

    fidelities = []
    for m in sequence_lengths:
        # True fidelity
        F_true = A * p**m + B

        # Add shot noise (binomial statistics)
        samples = np.random.binomial(1000, F_true, samples_per_length) / 1000
        fidelities.append(samples)

    fidelities = np.array(fidelities)
    return np.mean(fidelities, axis=1), np.std(fidelities, axis=1)

def fit_rb_data(sequence_lengths: np.ndarray,
                mean_fidelity: np.ndarray,
                std_fidelity: np.ndarray) -> dict:
    """
    Fit RB data to extract error rate

    Returns dictionary with fit parameters and uncertainties
    """
    # Initial guesses
    p0 = [0.98, 0.99, 0.01]

    # Fit with error weights
    popt, pcov = curve_fit(rb_decay_model, sequence_lengths, mean_fidelity,
                           p0=p0, sigma=std_fidelity, absolute_sigma=True)

    perr = np.sqrt(np.diag(pcov))

    # Extract error per Clifford
    A, p, B = popt
    epc = (1 - p) / 2  # Single qubit
    epc_err = perr[1] / 2

    return {
        'A': popt[0], 'p': popt[1], 'B': popt[2],
        'A_err': perr[0], 'p_err': perr[1], 'B_err': perr[2],
        'EPC': epc, 'EPC_err': epc_err
    }

# Generate RB data for each platform
np.random.seed(42)

platform_epc = {
    'Superconducting': 0.003,  # 0.3% EPC
    'Trapped Ion': 0.0005,      # 0.05% EPC
    'Neutral Atom': 0.002       # 0.2% EPC
}

sequence_lengths = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot RB data for each platform
ax = axes[0, 0]
rb_results = {}

for idx, (platform, epc) in enumerate(platform_epc.items()):
    mean_f, std_f = generate_rb_data(epc, sequence_lengths)
    fit = fit_rb_data(sequence_lengths, mean_f, std_f)
    rb_results[platform] = fit

    # Plot data points
    ax.errorbar(sequence_lengths, mean_f, yerr=std_f, fmt='o',
                color=colors[idx], capsize=3, markersize=5,
                label=f'{platform}')

    # Plot fit
    m_fine = np.linspace(1, 500, 200)
    ax.plot(m_fine, rb_decay_model(m_fine, fit['A'], fit['p'], fit['B']),
           '-', color=colors[idx], linewidth=2, alpha=0.7)

ax.set_xlabel('Sequence Length (Cliffords)')
ax.set_ylabel('Survival Probability')
ax.set_title('Randomized Benchmarking Comparison')
ax.legend()
ax.set_xlim(0, 520)
ax.set_ylim(0.4, 1.02)
ax.grid(True, alpha=0.3)

# =============================================================================
# Part 2: XEB Implementation
# =============================================================================

def xeb_fidelity(bitstrings: np.ndarray,
                 ideal_probs: np.ndarray) -> float:
    """
    Calculate XEB fidelity from sampled bitstrings and ideal probabilities

    Parameters:
    -----------
    bitstrings: Array of sampled bitstring indices
    ideal_probs: Ideal output probability distribution

    Returns:
    --------
    chi: XEB fidelity estimator
    """
    n_qubits = int(np.log2(len(ideal_probs)))

    # Calculate average ideal probability of sampled bitstrings
    avg_prob = np.mean(ideal_probs[bitstrings])

    # XEB formula
    chi = 2**n_qubits * avg_prob - 1

    return chi

def simulate_xeb_circuit(n_qubits: int, depth: int,
                        error_1q: float, error_2q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate XEB circuit with depolarizing noise

    Returns ideal probabilities and noisy sample indices
    """
    dim = 2**n_qubits

    # Generate random ideal distribution (Porter-Thomas)
    ideal_probs = np.random.exponential(1/dim, dim)
    ideal_probs /= ideal_probs.sum()

    # Estimate circuit fidelity
    n_1q = n_qubits * depth  # Rough estimate
    n_2q = (n_qubits - 1) * depth // 2

    fidelity = (1 - error_1q)**n_1q * (1 - error_2q)**n_2q

    # Noisy distribution
    noisy_probs = fidelity * ideal_probs + (1 - fidelity) / dim
    noisy_probs /= noisy_probs.sum()

    # Sample bitstrings
    n_samples = 10000
    samples = np.random.choice(dim, size=n_samples, p=noisy_probs)

    return ideal_probs, samples

# XEB depth scaling comparison
ax = axes[0, 1]
depths = np.arange(2, 51, 2)

platform_errors = {
    'Superconducting': {'1q': 0.001, '2q': 0.005},
    'Trapped Ion': {'1q': 0.0001, '2q': 0.001},
    'Neutral Atom': {'1q': 0.003, '2q': 0.005}
}

n_qubits = 5

for idx, (platform, errors) in enumerate(platform_errors.items()):
    xeb_fidelities = []

    for d in depths:
        ideal_p, samples = simulate_xeb_circuit(n_qubits, d,
                                                errors['1q'], errors['2q'])
        chi = xeb_fidelity(samples, ideal_p)
        xeb_fidelities.append(max(chi, 0))  # Floor at 0

    ax.semilogy(depths, xeb_fidelities, 'o-', color=colors[idx],
                label=platform, markersize=4, linewidth=2)

ax.axhline(y=1/3, color='gray', linestyle='--', label='QV threshold (1/3)')
ax.set_xlabel('Circuit Depth')
ax.set_ylabel('XEB Fidelity')
ax.set_title(f'XEB Fidelity vs Depth ({n_qubits} qubits)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-3, 1.5)

# =============================================================================
# Part 3: Gate Fidelity Comparison Bar Chart
# =============================================================================

ax = axes[1, 0]

# State-of-the-art fidelities (2024)
fidelities = {
    'Superconducting': {'1Q': 99.95, '2Q (best)': 99.7, '2Q (typical)': 99.3},
    'Trapped Ion': {'1Q': 99.99, '2Q (best)': 99.9, '2Q (typical)': 99.5},
    'Neutral Atom': {'1Q': 99.7, '2Q (best)': 99.5, '2Q (typical)': 99.0}
}

x = np.arange(3)  # Three gate types
width = 0.25

for idx, (platform, fids) in enumerate(fidelities.items()):
    values = [fids['1Q'], fids['2Q (best)'], fids['2Q (typical)']]
    bars = ax.bar(x + idx*width, values, width, label=platform, color=colors[idx])

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val}%', ha='center', va='bottom', fontsize=8, rotation=45)

ax.set_xticks(x + width)
ax.set_xticklabels(['1Q Gate', '2Q Gate (Best)', '2Q Gate (Typical)'])
ax.set_ylabel('Gate Fidelity (%)')
ax.set_title('State-of-the-Art Gate Fidelities (2024)')
ax.legend()
ax.set_ylim(98.5, 100.2)
ax.grid(True, alpha=0.3, axis='y')

# =============================================================================
# Part 4: Error Budget Visualization
# =============================================================================

ax = axes[1, 1]

# Error contributions (normalized to 100%)
error_budgets = {
    'Superconducting': {
        'Coherence': 35,
        'Control': 30,
        'Leakage': 15,
        'Crosstalk': 12,
        'SPAM': 8
    },
    'Trapped Ion': {
        'Heating': 25,
        'Laser noise': 30,
        'Detuning': 20,
        'Scattering': 15,
        'SPAM': 10
    },
    'Neutral Atom': {
        'Rydberg decay': 30,
        'Laser noise': 25,
        'Position': 20,
        'Doppler': 15,
        'Loss': 10
    }
}

categories = ['Coherence/Heating', 'Control/Laser', 'Leakage/Detuning',
              'Crosstalk/Scattering', 'SPAM/Other']

x = np.arange(len(categories))
width = 0.25

sc_vals = [35, 30, 15, 12, 8]
ti_vals = [25, 30, 20, 15, 10]
na_vals = [30, 25, 20, 15, 10]

ax.bar(x - width, sc_vals, width, label='Superconducting', color=colors[0])
ax.bar(x, ti_vals, width, label='Trapped Ion', color=colors[1])
ax.bar(x + width, na_vals, width, label='Neutral Atom', color=colors[2])

ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15, ha='right')
ax.set_ylabel('Error Contribution (%)')
ax.set_title('2Q Gate Error Budget Breakdown')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gate_fidelity_benchmarks.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Part 5: Interleaved RB Simulation
# =============================================================================

print("\n" + "="*70)
print("INTERLEAVED RB SIMULATION")
print("="*70)

def interleaved_rb_simulation(epc_ref: float, gate_error: float,
                              sequence_lengths: np.ndarray) -> dict:
    """
    Simulate standard and interleaved RB to extract gate error
    """
    # Reference RB
    mean_ref, std_ref = generate_rb_data(epc_ref, sequence_lengths)
    fit_ref = fit_rb_data(sequence_lengths, mean_ref, std_ref)

    # Interleaved RB (gate adds to error)
    epc_interleaved = epc_ref + gate_error / 2  # Approximate
    mean_int, std_int = generate_rb_data(epc_interleaved, sequence_lengths)
    fit_int = fit_rb_data(sequence_lengths, mean_int, std_int)

    # Extract gate error
    p_ref = fit_ref['p']
    p_int = fit_int['p']

    extracted_error = (1 - p_int/p_ref) / 2  # Single qubit

    return {
        'p_ref': p_ref,
        'p_interleaved': p_int,
        'extracted_error': extracted_error,
        'true_error': gate_error
    }

# Simulate for a T gate
true_t_error = 0.002  # 0.2%
results = interleaved_rb_simulation(0.001, true_t_error, sequence_lengths)

print(f"\nInterleaved RB for T gate:")
print(f"  Reference p: {results['p_ref']:.5f}")
print(f"  Interleaved p: {results['p_interleaved']:.5f}")
print(f"  Extracted error: {results['extracted_error']*100:.3f}%")
print(f"  True error: {results['true_error']*100:.3f}%")

# =============================================================================
# Part 6: Platform Summary Table
# =============================================================================

print("\n" + "="*70)
print("PLATFORM GATE FIDELITY SUMMARY")
print("="*70)

print("\n{:<20} {:<15} {:<15} {:<15}".format(
    "Metric", "Superconducting", "Trapped Ion", "Neutral Atom"))
print("-" * 65)

metrics = [
    ("1Q Fidelity", "99.95%", "99.99%", "99.7%"),
    ("2Q Fidelity (best)", "99.7%", "99.9%", "99.5%"),
    ("2Q Fidelity (typ)", "99.3%", "99.5%", "99.0%"),
    ("Gate time (2Q)", "50 ns", "200 μs", "1 μs"),
    ("RB EPC", "0.3%", "0.05%", "0.2%"),
    ("Primary error", "Coherence", "Laser noise", "Rydberg decay"),
]

for metric, sc, ti, na in metrics:
    print("{:<20} {:<15} {:<15} {:<15}".format(metric, sc, ti, na))

print("\n" + "="*70)
print("Key Insight: Trapped ions achieve highest fidelity but slowest gates.")
print("Superconducting offers balanced speed-fidelity trade-off.")
print("Neutral atoms show promise for parallel, high-fidelity operations.")
print("="*70)
```

## Summary

### Benchmarking Protocol Comparison

| Protocol | Measures | Assumptions | Best For |
|----------|----------|-------------|----------|
| Standard RB | Average Clifford error | Gate-independent errors | Overall quality |
| Interleaved RB | Specific gate error | Same as RB | Individual gates |
| XEB | Circuit fidelity | Depolarizing model | Deep circuits |
| Cycle Benchmarking | Layer error | Includes crosstalk | Parallel operations |

### Key Formulas

| Quantity | Formula |
|----------|---------|
| RB decay | $$F(m) = A \cdot p^m + B$$ |
| EPC (1Q) | $$r = (1-p)/2$$ |
| XEB fidelity | $$\chi = 2^n \langle p(x)\rangle - 1$$ |
| Diamond norm bound | $$\epsilon_{avg} \leq \epsilon_\diamond \leq 2\epsilon_{avg}$$ |

### Platform Fidelity Summary

| Platform | 1Q Best | 2Q Best | 2Q Typical |
|----------|---------|---------|------------|
| Superconducting | 99.95% | 99.7% | 99.3% |
| Trapped Ion | 99.99% | 99.9% | 99.5% |
| Neutral Atom | 99.7% | 99.5% | 99.0% |

### Main Takeaways

1. **RB is the gold standard** for average gate quality but assumes gate-independent errors
2. **XEB captures circuit-level fidelity** including coherence and crosstalk effects
3. **Trapped ions lead in fidelity** but at the cost of gate speed
4. **Superconducting qubits** offer the best speed-fidelity product for many applications
5. **Error budgets differ significantly** across platforms, requiring platform-specific optimization

## Daily Checklist

- [ ] I can implement and analyze randomized benchmarking data
- [ ] I understand the difference between RB, XEB, and cycle benchmarking
- [ ] I can compare state-of-the-art gate fidelities across platforms
- [ ] I can estimate algorithm fidelity from gate error rates
- [ ] I understand the error budgets for each platform
- [ ] I can choose appropriate benchmarking protocols for different scenarios

## Preview of Day 920

Tomorrow we analyze **Connectivity and Topology**, comparing native qubit connectivity graphs, routing overhead for different architectures, and graph connectivity metrics relevant to quantum algorithm implementation.
