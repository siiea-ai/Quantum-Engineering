# Day 782: Near-Term QEC Experiments

## Year 2, Semester 2A: Error Correction | Month 28: Advanced Stabilizer Codes | Week 112

---

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Google and IBM experimental results |
| Afternoon | 2.5 hours | Trapped-ion and neutral atom QEC |
| Evening | 2 hours | Experimental data analysis and lessons |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Interpret Google's surface code experiments** and break-even demonstrations
2. **Analyze IBM's heavy-hex QEC architecture** and results
3. **Evaluate trapped-ion QEC experiments** from IonQ and Quantinuum
4. **Understand neutral atom QEC approaches** and recent breakthroughs
5. **Critically assess experimental claims** and metrics
6. **Identify key challenges** remaining for fault-tolerant quantum computing

---

## Core Content

### 1. The State of Experimental QEC (2023-2024)

Quantum error correction has transitioned from theoretical framework to experimental reality. Key milestones:

| Year | Milestone | Group |
|------|-----------|-------|
| 2021 | Repeated QEC cycles on surface code | Google |
| 2022 | Logical qubit lifetime > physical | Multiple |
| 2023 | Break-even: distance-5 > distance-3 | Google |
| 2023-24 | Real-time decoding | Multiple |
| 2024 | Logical operations between codes | QuEra, IBM |

The central question being addressed: **Does adding more physical qubits actually improve logical performance?**

### 2. Google Quantum AI: Surface Code Experiments

#### Sycamore Processor (2019-2023)

Google's superconducting processor evolved:
- **2019**: 53 qubits, quantum supremacy demonstration
- **2021**: 17 qubits for distance-3 surface code
- **2023**: 72 qubits for distance-3 and distance-5 comparison

#### The 2023 Break-Even Experiment

**Key result**: Logical error rate decreased when increasing from distance-3 to distance-5.

$$\boxed{p_L(d=5) < p_L(d=3) \quad \text{(first demonstration)}}$$

**Experimental parameters:**
- Physical error rates: ~0.5-1% for two-qubit gates
- Syndrome extraction: ~1 $\mu$s per round
- Number of rounds: 25 (main experiment)

**Measured logical error rates:**

| Configuration | Logical Error per Round |
|---------------|------------------------|
| Distance-3 surface code | $\sim 3\%$ |
| Distance-5 surface code | $\sim 2.9\%$ |

**Error suppression factor:**
$$\Lambda = \frac{p_L(d) / p_L(d+2)}{\text{expected from threshold}}$$

Google measured $\Lambda \approx 2$, indicating below-threshold operation.

#### Correlated Errors and Leakage

A major finding: **correlated errors** significantly impact performance.

$$\boxed{p_{\text{correlated}} \approx 10^{-3} \gg p_{\text{uncorrelated}}^2}$$

Sources:
- Leakage to non-computational states
- Crosstalk between qubits
- Cosmic ray events (!)

**Leakage mitigation**: Reset operations inserted every few rounds.

### 3. IBM Quantum: Heavy-Hex Architecture

#### Heavy-Hex Lattice

IBM's connectivity pattern differs from Google's:

```
Standard grid:         Heavy-hex:
● - ● - ●             ●       ●
|   |   |            / \     / \
● - ● - ●           ●   ● - ●   ●
|   |   |            \ /     \ /
● - ● - ●             ●       ●
```

**Advantages of heavy-hex:**
- Lower crosstalk between non-neighboring qubits
- Better isolation for flag-qubit protocols
- Matches fixed-frequency transmon constraints

**Disadvantages:**
- Cannot directly implement standard surface code
- Requires adapted code designs (heavy-hex codes)

#### IBM's QEC Results (2022-2024)

**Eagle and Heron processors:**
- 127-433 qubits
- Focus on error mitigation + QEC hybrid approaches

**Key experiments:**
1. Distance-3 heavy-hex codes
2. Error suppression with dynamic decoupling
3. Zero-noise extrapolation combined with QEC

**Reported metrics:**
- Two-qubit gate errors: ~0.5-1%
- Readout errors: ~1-2%
- Logical error rates: ~5% per round (distance-3)

#### Error Mitigation vs. Error Correction

IBM pioneered **error mitigation** as a near-term alternative:

| Approach | Overhead | Scalability | Threshold? |
|----------|----------|-------------|------------|
| Error correction | Exponential qubits | Unlimited (in principle) | Yes |
| Error mitigation | Exponential shots | Limited depth | No |

### 4. Trapped-Ion QEC: IonQ and Quantinuum

#### Advantages of Trapped Ions

1. **High gate fidelity**: 99.9%+ for two-qubit gates
2. **All-to-all connectivity**: Any pair can interact
3. **Long coherence**: Seconds to minutes
4. **Identical qubits**: No fabrication variation

**Challenges:**
- Slow gate speed (~100 $\mu$s for MS gates)
- Limited qubit count (currently ~30-50 in single trap)
- Ion heating and motional decoherence

#### IonQ QEC Experiments

**Approach**: Use high fidelity to demonstrate small codes perfectly.

**Key results (2022-2024):**
- [[4,2,2]] error detection code
- Repeated error detection cycles
- Real-time classical feedback

**Logical error suppression:**
$$p_L^{\text{detected}} \approx 10^{-4} \quad \text{vs} \quad p_{\text{phys}} \approx 10^{-3}$$

The [[4,2,2]] code detects (but doesn't correct) single errors.

#### Quantinuum H-Series

**Architecture**: Linear trap with shuttling for qubit rearrangement.

**Achievements:**
- Distance-5 color code demonstrations
- Logical CNOT between encoded qubits
- Below-threshold operation

**Reported metrics:**
- Physical two-qubit fidelity: 99.8%
- Logical fidelity (encoded operations): 99.5%

### 5. Neutral Atom QEC: QuEra and Others

#### Neutral Atom Advantages

1. **Scalability**: 1000+ atoms demonstrated
2. **Reconfigurable geometry**: Optical tweezers allow any layout
3. **Erasure detection**: Atom loss is detectable
4. **Long-range gates**: Rydberg interactions

**Challenges:**
- Atom loss (~0.1-1% per operation)
- Rydberg gate fidelity (~99%)
- Limited gate parallelism

#### QuEra Experiments (2023-2024)

**Key demonstration**: Large-scale surface codes with erasure conversion.

**Erasure conversion advantage:**
$$\boxed{p_{\text{erasure}} \text{ correctable} \gg p_{\text{Pauli}} \text{ correctable}}$$

An erasure error (detectable location) is much easier to correct than a Pauli error (unknown location).

**Results:**
- 48-qubit logical qubits
- Distance-7 surface codes
- Logical error rates approaching threshold

### 6. Repetition Code Experiments

The simplest QEC: repetition codes protect against one error type.

#### Why Repetition Codes Matter

1. **Proof of principle**: Demonstrate QEC scaling
2. **Lower overhead**: Requires fewer qubits than full codes
3. **Benchmark**: Compare across platforms

$$\boxed{|0\rangle_L = |0\rangle^{\otimes n}, \quad |1\rangle_L = |1\rangle^{\otimes n}}$$

Corrects bit-flip errors; Z errors are uncorrectable.

#### Experimental Results Summary

| Platform | Distance | Logical Error Rate | Physical Error Rate |
|----------|----------|-------------------|---------------------|
| Google (SC) | 11 | $2 \times 10^{-3}$ | $10^{-3}$ |
| IBM (SC) | 9 | $5 \times 10^{-3}$ | $10^{-3}$ |
| IonQ | 5 | $10^{-4}$ | $5 \times 10^{-4}$ |
| QuEra | 7 | $10^{-3}$ | $5 \times 10^{-3}$ |

### 7. Critical Assessment of Experimental Claims

#### Common Pitfalls in Interpreting Results

1. **Cherry-picking data**: Reporting best runs only
2. **Post-selection**: Discarding runs with detected errors
3. **Incomplete error models**: Ignoring leakage, crosstalk
4. **Optimistic extrapolation**: Assuming perfect scaling

#### Key Questions to Ask

- **What is the raw success rate?** (Not just post-selected)
- **How many QEC cycles?** (Single round vs. sustained)
- **What about non-Clifford operations?** (Often excluded)
- **State preparation and measurement errors?** (Often dominant)

#### The $\Lambda$ Factor

A useful metric: **error suppression factor per distance increase**:

$$\boxed{\Lambda = \frac{p_L(d)}{p_L(d+2)}}$$

- $\Lambda > 1$: Below threshold, QEC is working
- $\Lambda < 1$: Above threshold, adding qubits hurts
- $\Lambda \sim 2-3$: Typical for near-threshold operation
- $\Lambda \gg 1$: Deep below threshold (not yet achieved)

### 8. Path to Fault-Tolerant Computing

#### Near-Term Milestones (2024-2027)

1. **Sustained QEC**: Thousands of error correction cycles
2. **Logical operations**: Gates between logical qubits
3. **Real-time decoding**: Classical processing fast enough
4. **Magic state distillation**: Demonstrated at scale

#### Medium-Term Goals (2027-2030)

1. **Break-even for algorithms**: Logical advantage for real tasks
2. **Million physical qubits**: Required for practical FT
3. **Integration with classical HPC**: Hybrid workflows

---

## Quantum Mechanics Connection

### Decoherence in Real Systems

Experimental QEC confronts real decoherence mechanisms:

**Superconducting qubits:**
$$T_1 \text{ decay}: \quad \rho \to e^{-t/T_1}|\psi\rangle\langle\psi| + (1-e^{-t/T_1})|0\rangle\langle 0|$$

**Trapped ions:**
$$\text{Motional heating}: \quad \bar{n} \to \bar{n} + \dot{\bar{n}} \cdot t$$

**Neutral atoms:**
$$\text{Atom loss}: \quad |\psi\rangle \to |\text{vacuum}\rangle \quad \text{(detectable)}$$

### Measurement-Induced Decoherence

Syndrome measurement necessarily disturbs the system:
- Measurement backaction (quantum mechanics)
- Measurement errors (classical noise)
- Reset and re-initialization overhead

---

## Worked Examples

### Example 1: Analyzing Google's Break-Even Data

**Problem:** Google reported logical error rates of $p_L(d=3) = 3.0\%$ and $p_L(d=5) = 2.9\%$ per round. Calculate $\Lambda$ and estimate the physical error rate assuming threshold behavior.

**Solution:**

Error suppression factor:
$$\Lambda = \frac{p_L(d=3)}{p_L(d=5)} = \frac{0.030}{0.029} = 1.034$$

This is barely above 1, indicating near-threshold operation.

From threshold theory:
$$p_L(d) \approx A \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$

Taking the ratio:
$$\frac{p_L(d=3)}{p_L(d=5)} = \left(\frac{p}{p_{\text{th}}}\right)^{(4-3)/2} \cdot \left(\frac{p}{p_{\text{th}}}\right)^{-1} = \left(\frac{p}{p_{\text{th}}}\right)^{-1}$$

Wait, let me recalculate:
$$\frac{p_L(d=3)}{p_L(d=5)} = \frac{(p/p_{\text{th}})^2}{(p/p_{\text{th}})^3} = \frac{p_{\text{th}}}{p}$$

So:
$$\Lambda = \frac{p_{\text{th}}}{p} = 1.034$$

If $p_{\text{th}} \approx 1\%$:
$$p = \frac{p_{\text{th}}}{1.034} \approx 0.97\%$$

$$\boxed{p_{\text{phys}} \approx 1\%, \quad \Lambda = 1.03}$$

The system is operating at approximately the threshold.

### Example 2: Erasure Conversion Advantage

**Problem:** A neutral atom system has 1% atom loss (erasure) and 0.5% Pauli errors. Compare the logical error rate at distance $d = 5$ assuming: (a) erasures treated as Pauli errors, (b) erasures detected and handled optimally.

**Solution:**

**Case (a): Erasures as Paulis**

Effective Pauli rate: $p_{\text{eff}} = 0.01 + 0.005 = 0.015$

With $p_{\text{th}} = 0.01$, we're above threshold!

$$p_L^{(a)} = A \left(\frac{0.015}{0.01}\right)^{3} = A \times 3.4$$

The logical error rate increases with distance.

**Case (b): Erasure detection**

Erasure threshold is much higher: $p_{\text{th}}^{\text{erasure}} \approx 0.5$ (50%!)

Effective erasure contribution:
$$p_L^{\text{erasure}} \approx (0.01/0.5)^3 \times A = A \times 8 \times 10^{-6}$$

Pauli contribution:
$$p_L^{\text{Pauli}} \approx (0.005/0.01)^3 \times A = A \times 0.125$$

Total: $p_L^{(b)} \approx 0.125 A$

$$\boxed{\frac{p_L^{(a)}}{p_L^{(b)}} \approx \frac{3.4}{0.125} \approx 27\times \text{ improvement}}$$

Erasure conversion is highly advantageous.

### Example 3: Decoding Speed Requirement

**Problem:** A surface code operates at distance $d = 17$ with syndrome cycle time $t_{\text{cycle}} = 1 \mu$s. If the decoder must complete before the next round, what is the maximum decoding time? How many syndrome bits must be processed?

**Solution:**

Maximum decoding time:
$$t_{\text{decode}} < t_{\text{cycle}} = 1 \mu s$$

Number of syndrome bits per round:
- X stabilizers: $(d-1) \times d / 2 = 8 \times 17 = 136$ (roughly)
- Z stabilizers: Same count

Actually, for $d \times d$ surface code:
$$N_{\text{syndromes}} = d^2 - 1 = 288 \text{ (data)} + d^2 - 1 = 288 \text{ (measurement qubits)}$$

Simplified: $N \approx 2d^2 = 578$ syndrome bits per round.

Decoding throughput required:
$$\text{Throughput} = \frac{578 \text{ bits}}{1 \mu s} = 578 \text{ Mbits/s}$$

Modern FPGA-based decoders achieve:
- MWPM: ~100 $\mu$s latency (too slow for real-time)
- Union-Find: ~1 $\mu$s latency (feasible)
- Neural network: ~10 $\mu$s (improving)

$$\boxed{t_{\text{decode}} < 1 \mu s \text{ requires Union-Find or faster decoders}}$$

---

## Practice Problems

### Level A: Direct Application

**A1.** Google's experiment used 72 qubits for distance-5 surface code. Calculate how many data qubits and measurement qubits were used.

**A2.** If a trapped-ion system has physical two-qubit gate fidelity of 99.9%, what is the expected logical error rate for a distance-3 repetition code (assuming only gate errors)?

**A3.** An IBM heavy-hex processor has 127 qubits. Estimate the maximum distance surface code that could fit (accounting for ~30% overhead for connectivity).

### Level B: Intermediate Analysis

**B1.** Compare the space-time overhead for Google's superconducting approach (1 $\mu$s cycles, 0.5% error) versus IonQ's trapped-ion approach (100 $\mu$s cycles, 0.1% error) for achieving $p_L = 10^{-6}$.

**B2.** Derive the expected logical error rate scaling for QuEra's erasure-converted neutral atom system, assuming 1% erasure rate and 0.3% Pauli rate.

**B3.** IBM reports a "quantum volume" of 128 for their processor. Explain what this metric measures and how it relates to QEC capability.

### Level C: Research-Level Challenges

**C1.** Analyze the impact of correlated errors on surface code thresholds. If 10% of errors are correlated (affecting pairs of neighboring qubits), how does this modify the effective threshold?

**C2.** Design an experiment to distinguish between: (a) errors due to control noise, (b) errors due to decoherence, (c) errors due to crosstalk. What measurements would you perform?

**C3.** Propose a roadmap for a quantum computing company to go from current distance-5 demonstrations to practical fault-tolerant computing. Identify the key technical milestones and estimated resource requirements.

---

## Computational Lab

```python
"""
Day 782: Near-Term QEC Experiments
Analysis of experimental QEC data and projections
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, List, Dict

# =============================================================================
# EXPERIMENTAL DATA
# =============================================================================

# Google Sycamore 2023 data (approximate from paper)
GOOGLE_DATA = {
    'distance': [3, 5],
    'logical_error_per_round': [0.030, 0.029],
    'physical_error': 0.006,  # Approximate two-qubit gate error
    'cycle_time_us': 1.0,
}

# IBM Eagle/Heron data (approximate)
IBM_DATA = {
    'distance': [3, 5],
    'logical_error_per_round': [0.05, 0.06],  # Heavy-hex adapted
    'physical_error': 0.008,
    'cycle_time_us': 0.5,
}

# IonQ trapped-ion data (approximate)
IONQ_DATA = {
    'distance': [3, 5],
    'logical_error_per_round': [0.005, 0.003],
    'physical_error': 0.001,
    'cycle_time_us': 100.0,
}

# QuEra neutral atom data (approximate)
QUERA_DATA = {
    'distance': [3, 5, 7],
    'logical_error_per_round': [0.02, 0.008, 0.004],
    'physical_error': 0.005,
    'erasure_rate': 0.01,
    'cycle_time_us': 10.0,
}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def threshold_model(d: np.ndarray, A: float, p_ratio: float) -> np.ndarray:
    """
    Logical error rate model below threshold.

    p_L = A * (p/p_th)^((d+1)/2)

    Args:
        d: Code distance
        A: Prefactor
        p_ratio: p/p_th ratio

    Returns:
        Logical error rate
    """
    return A * (p_ratio ** ((d + 1) / 2))


def fit_threshold_model(distances: List[int],
                        errors: List[float]) -> Tuple[float, float]:
    """Fit threshold model to experimental data."""
    try:
        popt, _ = curve_fit(threshold_model,
                           np.array(distances),
                           np.array(errors),
                           p0=[0.1, 0.5],
                           bounds=([0, 0], [1, 1]))
        return popt[0], popt[1]
    except:
        return 0.1, 0.5


def calculate_lambda(data: Dict) -> float:
    """Calculate error suppression factor."""
    if len(data['distance']) < 2:
        return 1.0

    p_d1 = data['logical_error_per_round'][0]
    p_d2 = data['logical_error_per_round'][1]

    if p_d2 > 0:
        return p_d1 / p_d2
    return 1.0


def project_to_target_error(data: Dict,
                           target_error: float = 1e-10) -> Dict:
    """
    Project resources needed to reach target logical error rate.

    Returns:
        Dictionary with required distance, qubits, and time estimates
    """
    # Fit threshold model
    A, p_ratio = fit_threshold_model(
        data['distance'],
        data['logical_error_per_round']
    )

    # Find required distance
    # target = A * p_ratio^((d+1)/2)
    # d = 2 * log(target/A) / log(p_ratio) - 1
    if p_ratio < 1 and A > 0:
        d_required = 2 * np.log(target_error / A) / np.log(p_ratio) - 1
        d_required = max(3, int(np.ceil(d_required)))
        if d_required % 2 == 0:
            d_required += 1
    else:
        d_required = float('inf')

    # Physical qubits (surface code)
    n_qubits = 2 * d_required**2 if d_required != float('inf') else float('inf')

    return {
        'fitted_A': A,
        'fitted_p_ratio': p_ratio,
        'required_distance': d_required,
        'physical_qubits': n_qubits,
        'target_error': target_error
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_experimental_comparison():
    """Compare experimental results across platforms."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Logical error rate vs distance
    ax = axes[0, 0]

    platforms = [
        ('Google', GOOGLE_DATA, 'bo-'),
        ('IBM', IBM_DATA, 'rs-'),
        ('IonQ', IONQ_DATA, 'g^-'),
        ('QuEra', QUERA_DATA, 'mp-'),
    ]

    for name, data, style in platforms:
        ax.semilogy(data['distance'], data['logical_error_per_round'],
                   style, label=name, linewidth=2, markersize=10)

    ax.set_xlabel('Code Distance', fontsize=12)
    ax.set_ylabel('Logical Error per Round', fontsize=12)
    ax.set_title('Experimental Logical Error Rates', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 2. Lambda (error suppression factor)
    ax = axes[0, 1]

    lambdas = []
    names = []
    for name, data, _ in platforms:
        lam = calculate_lambda(data)
        lambdas.append(lam)
        names.append(name)

    colors = ['blue', 'red', 'green', 'purple']
    bars = ax.bar(names, lambdas, color=colors, alpha=0.7)
    ax.axhline(y=1, color='black', linestyle='--', label='Break-even')
    ax.set_ylabel('Error Suppression Factor Λ', fontsize=12)
    ax.set_title('Below-Threshold Indicator (Λ > 1 is good)', fontsize=14)

    for bar, lam in zip(bars, lambdas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{lam:.2f}', ha='center', fontsize=11)

    # 3. Space-time volume comparison
    ax = axes[1, 0]

    # Calculate space-time cost to achieve p_L = 10^-6
    target = 1e-6
    st_volumes = []

    for name, data, _ in platforms:
        proj = project_to_target_error(data, target)
        d = proj['required_distance']
        if d != float('inf') and d < 100:
            qubits = 2 * d**2
            # Assume need ~1000 rounds
            time = 1000 * data['cycle_time_us']
            st_vol = qubits * time
            st_volumes.append((name, st_vol / 1e6))  # in Mqubit-μs
        else:
            st_volumes.append((name, float('nan')))

    valid_data = [(n, v) for n, v in st_volumes if not np.isnan(v)]
    if valid_data:
        names_v = [x[0] for x in valid_data]
        vols = [x[1] for x in valid_data]
        ax.bar(names_v, vols, color=['blue', 'red', 'green', 'purple'][:len(names_v)],
               alpha=0.7)
        ax.set_ylabel('Space-Time Volume (M qubit·μs)', fontsize=12)
        ax.set_title(f'Projected Cost for p_L = {target:.0e}', fontsize=14)
        ax.set_yscale('log')

    # 4. Physical vs logical error rates
    ax = axes[1, 1]

    for name, data, style in platforms:
        p_phys = data['physical_error']
        p_log = data['logical_error_per_round']
        ax.semilogy([p_phys] * len(p_log), p_log, 'o',
                   markersize=12, label=name)

    # Add break-even line
    p_range = np.logspace(-4, -1, 50)
    ax.semilogy(p_range, p_range, 'k--', label='Break-even (p_L = p_phys)')

    ax.set_xlabel('Physical Error Rate', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('Physical vs Logical Error Rates', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('day_782_experimental_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_scaling_projections():
    """Project future scaling requirements."""

    fig, ax = plt.subplots(figsize=(10, 7))

    # Target logical error rates
    targets = np.logspace(-4, -15, 50)

    # Projection for each platform
    platforms = [
        ('Google', GOOGLE_DATA, 'b-'),
        ('IonQ', IONQ_DATA, 'g-'),
        ('QuEra', QUERA_DATA, 'm-'),
    ]

    for name, data, style in platforms:
        A, p_ratio = fit_threshold_model(
            data['distance'],
            data['logical_error_per_round']
        )

        if p_ratio < 1 and A > 0:
            distances = []
            for target in targets:
                d = 2 * np.log(target / A) / np.log(p_ratio) - 1
                d = max(3, d)
                distances.append(d)

            ax.loglog(targets, distances, style, label=name, linewidth=2)

    # Add reference lines
    ax.axvline(x=1e-10, color='gray', linestyle='--', alpha=0.5)
    ax.text(1.2e-10, 100, 'Useful QC\ntarget', fontsize=10)

    ax.axvline(x=1e-15, color='gray', linestyle='--', alpha=0.5)
    ax.text(1.2e-15, 100, 'Shor\'s\nalgorithm', fontsize=10)

    ax.set_xlabel('Target Logical Error Rate', fontsize=12)
    ax.set_ylabel('Required Code Distance', fontsize=12)
    ax.set_title('Scaling Projections from Current Data', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig('day_782_scaling_projections.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_decoding_requirements():
    """Analyze real-time decoding requirements."""

    print("=" * 60)
    print("REAL-TIME DECODING REQUIREMENTS")
    print("=" * 60)

    distances = [5, 11, 17, 23, 31]
    cycle_times = [1.0, 0.5, 0.1]  # μs

    print(f"\n{'Distance':>8} | {'Syndromes':>10} | ", end="")
    for ct in cycle_times:
        print(f"{'Rate @'+str(ct)+'μs':>12} | ", end="")
    print()
    print("-" * 60)

    for d in distances:
        n_syndromes = 2 * d**2
        print(f"{d:>8} | {n_syndromes:>10} | ", end="")
        for ct in cycle_times:
            rate = n_syndromes / ct  # bits per μs = Mbits/s
            print(f"{rate:>10.0f} Mb/s | ", end="")
        print()

    print("\nDecoder comparison:")
    print("-" * 40)
    print("MWPM: ~100 μs latency (offline only)")
    print("Union-Find: ~1 μs latency (real-time capable)")
    print("Neural network: ~10 μs (improving)")
    print("FPGA-based: <1 μs (custom hardware)")


def erasure_vs_pauli_comparison():
    """Compare erasure vs Pauli error correction."""

    print("\n" + "=" * 60)
    print("ERASURE vs PAULI ERROR COMPARISON")
    print("=" * 60)

    p_erasure = 0.01
    p_pauli = 0.005

    # Thresholds
    p_th_erasure = 0.5
    p_th_pauli = 0.01

    distances = [3, 5, 7, 9, 11]

    print(f"\np_erasure = {p_erasure}, p_Pauli = {p_pauli}")
    print(f"Thresholds: erasure = {p_th_erasure}, Pauli = {p_th_pauli}")

    print(f"\n{'Distance':>8} | {'Erasure p_L':>12} | {'Pauli p_L':>12} | {'Ratio':>10}")
    print("-" * 50)

    for d in distances:
        # Erasure contribution (detected errors are easy)
        p_L_erasure = (p_erasure / p_th_erasure) ** ((d + 1) / 2)

        # Pauli contribution
        p_L_pauli = (p_pauli / p_th_pauli) ** ((d + 1) / 2)

        ratio = p_L_pauli / p_L_erasure if p_L_erasure > 0 else float('inf')

        print(f"{d:>8} | {p_L_erasure:>12.2e} | {p_L_pauli:>12.2e} | {ratio:>10.1f}")


def generate_summary_table():
    """Generate comprehensive summary table."""

    print("\n" + "=" * 60)
    print("EXPERIMENTAL QEC SUMMARY (2023-2024)")
    print("=" * 60)

    platforms = {
        'Google Sycamore': {
            'qubits': 72,
            'architecture': 'Superconducting (grid)',
            'best_distance': 5,
            'p_phys': '0.6%',
            'p_L': '2.9%',
            'cycle_time': '1 μs',
            'key_result': 'First below-threshold scaling'
        },
        'IBM Heron': {
            'qubits': 133,
            'architecture': 'Superconducting (heavy-hex)',
            'best_distance': 3,
            'p_phys': '0.5%',
            'p_L': '5%',
            'cycle_time': '0.5 μs',
            'key_result': 'Error mitigation + QEC hybrid'
        },
        'IonQ Forte': {
            'qubits': 32,
            'architecture': 'Trapped ion',
            'best_distance': 5,
            'p_phys': '0.1%',
            'p_L': '0.3%',
            'cycle_time': '100 μs',
            'key_result': 'Highest fidelity operations'
        },
        'Quantinuum H2': {
            'qubits': 32,
            'architecture': 'Trapped ion (shuttling)',
            'best_distance': 5,
            'p_phys': '0.2%',
            'p_L': '0.5%',
            'cycle_time': '50 μs',
            'key_result': 'Logical CNOT demonstration'
        },
        'QuEra Aquila': {
            'qubits': 256,
            'architecture': 'Neutral atom',
            'best_distance': 7,
            'p_phys': '0.5%',
            'p_L': '0.4%',
            'cycle_time': '10 μs',
            'key_result': 'Erasure conversion advantage'
        }
    }

    for name, data in platforms.items():
        print(f"\n{name}")
        print("-" * 40)
        for key, value in data.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("Day 782: Near-Term QEC Experiments")
    print("=" * 60)

    # Generate summary
    generate_summary_table()

    # Decoding analysis
    analyze_decoding_requirements()

    # Erasure comparison
    erasure_vs_pauli_comparison()

    # Generate plots
    plot_experimental_comparison()
    plot_scaling_projections()

    # Calculate key metrics
    print("\n" + "=" * 60)
    print("KEY METRICS")
    print("=" * 60)

    for name, data in [('Google', GOOGLE_DATA), ('IonQ', IONQ_DATA), ('QuEra', QUERA_DATA)]:
        lam = calculate_lambda(data)
        proj = project_to_target_error(data, 1e-10)
        print(f"\n{name}:")
        print(f"  Λ = {lam:.3f}")
        print(f"  Distance needed for p_L = 10^-10: d = {proj['required_distance']}")
        print(f"  Physical qubits needed: {proj['physical_qubits']:,}")
```

---

## Summary

### Key Experimental Results

| Platform | Best Distance | Logical Error | Key Achievement |
|----------|--------------|---------------|-----------------|
| Google | 5 | 2.9%/round | First break-even |
| IBM | 3 | 5%/round | Heavy-hex codes |
| IonQ | 5 | 0.3%/round | Highest fidelity |
| Quantinuum | 5 | 0.5%/round | Logical gates |
| QuEra | 7 | 0.4%/round | Erasure advantage |

### Main Takeaways

1. **Break-even achieved**: Google demonstrated $p_L(d=5) < p_L(d=3)$ for first time
2. **Platform diversity**: Each technology has distinct advantages and challenges
3. **Correlated errors matter**: Real systems have more complex error models
4. **Erasure conversion is powerful**: Detectable errors are much easier to correct
5. **Real-time decoding is critical**: Classical processing must keep up with quantum
6. **Million-qubit scale needed**: Current demonstrations are far from practical FT

---

## Daily Checklist

- [ ] I can interpret experimental logical error rates
- [ ] I understand the break-even criterion for QEC
- [ ] I can compare different hardware platforms
- [ ] I know the challenges facing each technology
- [ ] I completed the computational lab
- [ ] I solved at least 2 practice problems from each level

---

## Preview: Day 783

Tomorrow we study **Quantum Computer Architecture**, examining the full stack from physical qubits to logical algorithms:
- Control electronics and FPGA systems
- Cryogenic engineering for superconducting qubits
- Classical-quantum interface design
- Compilation and scheduling

*"A quantum computer is not just qubits; it is an entire system working in concert."*

---

*Day 782 of 2184 | Year 2, Month 28, Week 112, Day 5*
*Quantum Engineering PhD Curriculum*
