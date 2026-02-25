# Day 827: Decoding Latency Constraints

## Week 119: Real-Time Decoding | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Schedule (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 2.5 hours | Real-time requirements, timing budgets |
| **Afternoon** | 2.5 hours | Backlog analysis, latency scaling |
| **Evening** | 2 hours | Latency benchmarking lab |

---

## Learning Objectives

By the end of Day 827, you will be able to:

1. **Derive** the fundamental real-time decoding constraint from QEC cycle timing
2. **Analyze** latency budgets for different qubit modalities (superconducting, trapped ion)
3. **Model** backlog accumulation and its effect on logical error rates
4. **Quantify** the decoder latency requirements as a function of code distance
5. **Evaluate** the trade-off between decoder accuracy and speed
6. **Benchmark** decoder implementations against real-time constraints

---

## Core Content

### 1. The Fundamental Timing Constraint

Quantum error correction operates in a continuous cycle:

1. **Syndrome Extraction**: Measure stabilizer generators (~200-800 ns)
2. **Classical Communication**: Transfer syndrome to decoder (~50-200 ns)
3. **Decoding**: Compute correction from syndrome (~??? ns)
4. **Feedback**: Apply correction or update Pauli frame (~50-100 ns)

The total cycle time $t_{\text{cycle}}$ must satisfy:

$$t_{\text{cycle}} = t_{\text{measure}} + t_{\text{comm}} + t_{\text{decode}} + t_{\text{feedback}}$$

For **continuous** error correction, each new syndrome round begins immediately after the previous measurement completes. The decoder must finish before the next syndrome arrives:

$$\boxed{t_{\text{decode}} < t_{\text{cycle}} - t_{\text{overhead}}}$$

where $t_{\text{overhead}} = t_{\text{comm}} + t_{\text{feedback}}$.

### 2. Platform-Specific Timing Budgets

#### Superconducting Qubits

| Component | Typical Time | Notes |
|-----------|--------------|-------|
| Single-qubit gate | 20-50 ns | Microwave pulses |
| Two-qubit gate | 30-100 ns | Cross-resonance, CZ |
| Measurement | 200-500 ns | Dispersive readout |
| Reset | 100-500 ns | Active reset techniques |
| **Syndrome cycle** | 400-1000 ns | Full stabilizer extraction |
| Communication | 100-300 ns | Room temp electronics |
| **Decode budget** | **100-500 ns** | Remaining time |

#### Trapped Ion Qubits

| Component | Typical Time | Notes |
|-----------|--------------|-------|
| Single-qubit gate | 1-10 μs | Laser-driven |
| Two-qubit gate | 10-200 μs | Molmer-Sorensen |
| Measurement | 100-1000 μs | Fluorescence detection |
| **Syndrome cycle** | 1-10 ms | Much slower |
| **Decode budget** | **~1-5 ms** | More relaxed |

The stringent superconducting timing drives most real-time decoder research.

### 3. Backlog Dynamics

When decoding takes longer than the cycle time, syndromes accumulate in a queue. Let:
- $t_c = t_{\text{cycle}}$: syndrome arrival period
- $t_d = t_{\text{decode}}$: decoding time per syndrome

If $t_d > t_c$, the backlog grows as:

$$\frac{dN_{\text{backlog}}}{dt} = \frac{1}{t_c} - \frac{1}{t_d} = \frac{t_d - t_c}{t_c \cdot t_d}$$

After time $T$, the backlog is:

$$N_{\text{backlog}}(T) = \frac{T(t_d - t_c)}{t_c \cdot t_d}$$

The **oldest syndrome** in the backlog has age:

$$\tau_{\text{oldest}} = N_{\text{backlog}} \cdot t_c$$

### 4. Backlog Impact on Logical Error Rate

Uncorrected syndromes allow errors to propagate. The logical error rate increases exponentially with backlog:

$$\boxed{p_L(\tau) \approx p_L^{(0)} \cdot \exp\left(\frac{\tau \cdot p}{t_c}\right)}$$

where:
- $p_L^{(0)}$ is the baseline logical error rate with real-time decoding
- $p$ is the physical error rate
- $\tau$ is the delay before correction is applied

For a backlog of $N$ syndromes:

$$p_L \sim p_L^{(0)} \cdot \left(1 + \frac{p \cdot N}{d/2}\right)^{d/2}$$

where $d$ is the code distance. Beyond a critical backlog $N_{\text{crit}} \sim d/(2p)$, error correction fails entirely.

### 5. Latency Scaling with Code Distance

For a distance-$d$ surface code:
- Number of data qubits: $n = d^2$
- Number of syndrome bits: $m \approx d^2 - 1$
- Matching graph size: $O(d^2)$ vertices, $O(d^2)$ edges per round

Decoder scaling:

| Algorithm | Time Complexity | Typical Scaling |
|-----------|-----------------|-----------------|
| MWPM (Blossom) | $O(n^3)$ worst | $O(d^6)$ |
| MWPM (optimized) | $O(n^2 \log n)$ | $O(d^4 \log d)$ |
| Union-Find | $O(n \cdot \alpha(n))$ | $O(d^2)$ |
| Neural Network | $O(n)$ | $O(d^2)$ |

For superconducting systems targeting $d = 20$:
- $n = 400$ data qubits
- MWPM: potentially $10^6$ operations
- Union-Find: $\sim 10^3$ operations

### 6. The Accuracy-Latency Trade-off

Faster decoders typically have lower thresholds:

$$p_{\text{th}}^{\text{fast}} < p_{\text{th}}^{\text{optimal}}$$

The **effective threshold** accounting for backlog:

$$\boxed{p_{\text{th}}^{\text{eff}} = p_{\text{th}}^{\text{alg}} \cdot \mathcal{D}(t_d / t_c)}$$

where $\mathcal{D}$ is a degradation function:

$$\mathcal{D}(r) = \begin{cases} 1 & r < 1 \\ e^{-\lambda(r-1)} & r \geq 1 \end{cases}$$

A decoder with 10% lower threshold but real-time operation often outperforms an optimal decoder with backlog.

---

## Worked Examples

### Example 1: Superconducting Timing Budget

**Problem**: A superconducting quantum computer has:
- Syndrome measurement: 500 ns
- Room temperature round-trip: 200 ns
- Feedback application: 50 ns

What is the maximum decoder latency for real-time operation?

**Solution**:

Total cycle time:
$$t_{\text{cycle}} = t_{\text{measure}} + t_{\text{round-trip}} + t_{\text{feedback}} = 500 + 200 + 50 = 750 \text{ ns}$$

Wait—this isn't quite right. The cycle time is determined by when the next measurement can begin. For continuous QEC:

$$t_{\text{cycle}} = t_{\text{measure}} \approx 500-700 \text{ ns}$$

The decoder must complete before the next syndrome arrives:
$$t_{\text{decode}}^{\text{max}} = t_{\text{cycle}} - t_{\text{comm}} = 500 - 200 = 300 \text{ ns}$$

With some margin for jitter:
$$\boxed{t_{\text{decode}} \lesssim 200-300 \text{ ns}}$$

### Example 2: Backlog Accumulation

**Problem**: A decoder takes $t_d = 1.5 \, \mu\text{s}$ while the cycle time is $t_c = 1.0 \, \mu\text{s}$. After 1 second of operation, what is the backlog?

**Solution**:

Backlog growth rate:
$$\frac{dN}{dt} = \frac{t_d - t_c}{t_c \cdot t_d} = \frac{1.5 - 1.0}{1.0 \times 1.5} = \frac{0.5}{1.5} = \frac{1}{3} \text{ syndromes/μs}$$

After $T = 1$ second $= 10^6 \, \mu\text{s}$:
$$N_{\text{backlog}} = \frac{10^6}{3} \approx 333,000 \text{ syndromes}$$

The oldest unprocessed syndrome is:
$$\tau_{\text{oldest}} = N \cdot t_c = 333,000 \times 1 \, \mu\text{s} = 333 \text{ ms}$$

This is catastrophic—errors have been accumulating for a third of a second!

$$\boxed{N_{\text{backlog}} = 333,000}$$

### Example 3: Critical Backlog Threshold

**Problem**: For a distance-7 surface code with physical error rate $p = 0.5\%$, what backlog causes logical failure?

**Solution**:

Critical backlog estimate:
$$N_{\text{crit}} \sim \frac{d}{2p} = \frac{7}{2 \times 0.005} = \frac{7}{0.01} = 700$$

More precisely, logical error rate scales as:
$$p_L(N) \sim p_L^{(0)} \cdot \left(1 + \frac{p \cdot N}{d/2}\right)^{d/2}$$

Setting $p_L(N_{\text{crit}}) = 1$:
$$\left(1 + \frac{0.005 \cdot N_{\text{crit}}}{3.5}\right)^{3.5} \approx \frac{1}{p_L^{(0)}}$$

For $p_L^{(0)} \sim 10^{-4}$ (typical for $d=7$ at $p=0.5\%$):
$$1 + \frac{0.005 \cdot N_{\text{crit}}}{3.5} \approx (10^4)^{1/3.5} \approx 30$$

$$N_{\text{crit}} \approx \frac{29 \times 3.5}{0.005} \approx 20,000$$

$$\boxed{N_{\text{crit}} \approx 700-20,000 \text{ (depending on model)}}$$

---

## Practice Problems

### Direct Application

**Problem 1**: A trapped ion system has syndrome cycle time of 5 ms. If the decoder uses MWPM with complexity $O(d^4)$ and takes 10 ms for $d=5$, can it operate in real-time for $d=7$?

**Problem 2**: Calculate the backlog after 10 seconds when $t_d = 1.2 \, \mu\text{s}$ and $t_c = 1.0 \, \mu\text{s}$.

### Intermediate

**Problem 3**: A distance-11 surface code requires $p_L < 10^{-10}$ for a quantum algorithm. If the baseline $p_L^{(0)} = 10^{-12}$ with real-time decoding, what is the maximum acceptable backlog?

**Problem 4**: Design a timing budget for a photonic quantum computer with:
- Photon generation: 10 ns
- Fusion measurement: 50 ns
- Classical processing at room temperature
- Target distance: 15

### Challenging

**Problem 5**: Derive the effective threshold formula including backlog effects:
$$p_{\text{th}}^{\text{eff}}(r) = p_{\text{th}} \cdot f(r, d)$$
where $r = t_d/t_c$ and $f$ captures the backlog penalty.

**Problem 6**: A quantum computer uses a hybrid decoder: fast Union-Find for immediate correction, slow MWPM for refinement. Analyze the error rate when Union-Find has threshold 9.5% and MWPM has 10.5%, with physical error rate 9.8%.

---

## Computational Lab: Latency Benchmarking

```python
"""
Day 827 Lab: Decoding Latency Analysis
Benchmarking decoder performance against real-time constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Part 1: Timing Budget Calculator
# =============================================================================

def calculate_timing_budget(platform='superconducting'):
    """
    Calculate timing budget for different quantum platforms.

    Returns dict with timing components in nanoseconds.
    """
    if platform == 'superconducting':
        budget = {
            'single_qubit_gate': 30,      # ns
            'two_qubit_gate': 60,          # ns
            'measurement': 400,            # ns
            'reset': 200,                  # ns
            'communication_oneway': 100,   # ns to room temp
            'communication_roundtrip': 200,
            'feedback': 50,
        }
        # Syndrome extraction: 1 round of CX gates + measurement
        budget['syndrome_extraction'] = (
            4 * budget['two_qubit_gate'] +  # 4 CX gates per stabilizer
            budget['measurement']
        )
        budget['cycle_time'] = budget['syndrome_extraction']
        budget['decode_budget'] = (
            budget['cycle_time'] -
            budget['communication_roundtrip']
        )

    elif platform == 'trapped_ion':
        budget = {
            'single_qubit_gate': 5000,     # 5 μs
            'two_qubit_gate': 100000,      # 100 μs
            'measurement': 300000,         # 300 μs
            'reset': 50000,                # 50 μs
            'communication_roundtrip': 1000,  # 1 μs (negligible)
            'feedback': 1000,
        }
        budget['syndrome_extraction'] = (
            4 * budget['two_qubit_gate'] +
            budget['measurement']
        )
        budget['cycle_time'] = budget['syndrome_extraction']
        budget['decode_budget'] = budget['cycle_time'] - budget['communication_roundtrip']

    elif platform == 'photonic':
        budget = {
            'photon_generation': 10,
            'fusion_gate': 50,
            'detection': 100,
            'communication_roundtrip': 50,
            'feedback': 20,
        }
        budget['syndrome_extraction'] = 200  # Approximate
        budget['cycle_time'] = budget['syndrome_extraction']
        budget['decode_budget'] = budget['cycle_time'] - budget['communication_roundtrip']

    return budget

# Display timing budgets
print("=" * 60)
print("TIMING BUDGETS BY PLATFORM")
print("=" * 60)

for platform in ['superconducting', 'trapped_ion', 'photonic']:
    budget = calculate_timing_budget(platform)
    print(f"\n{platform.upper()}")
    print(f"  Cycle time: {budget['cycle_time']:,} ns ({budget['cycle_time']/1000:.1f} μs)")
    print(f"  Decode budget: {budget['decode_budget']:,} ns ({budget['decode_budget']/1000:.1f} μs)")

# =============================================================================
# Part 2: Backlog Simulation
# =============================================================================

def simulate_backlog(t_cycle, t_decode, total_time, time_unit='us'):
    """
    Simulate backlog accumulation when decoder is slower than cycle time.

    Parameters:
    -----------
    t_cycle : float
        Syndrome cycle time
    t_decode : float
        Decoder processing time per syndrome
    total_time : float
        Total simulation time
    time_unit : str
        Time unit for display ('ns', 'us', 'ms')

    Returns:
    --------
    dict with simulation results
    """
    # Convert to common unit (ns)
    unit_factor = {'ns': 1, 'us': 1000, 'ms': 1e6}[time_unit]

    # Simulation state
    current_time = 0
    syndromes_arrived = 0
    syndromes_decoded = 0
    decoder_free_at = 0

    backlog_history = []
    time_history = []

    # Simulation loop
    while current_time < total_time:
        # Syndrome arrives every t_cycle
        if current_time % t_cycle == 0:
            syndromes_arrived += 1

        # Decoder finishes if it's been working long enough
        if current_time >= decoder_free_at and syndromes_decoded < syndromes_arrived:
            syndromes_decoded += 1
            decoder_free_at = current_time + t_decode

        backlog = syndromes_arrived - syndromes_decoded

        if current_time % (total_time // 100) == 0:  # Sample 100 points
            backlog_history.append(backlog)
            time_history.append(current_time)

        current_time += 1

    return {
        'time': np.array(time_history),
        'backlog': np.array(backlog_history),
        'final_backlog': syndromes_arrived - syndromes_decoded,
        'syndromes_total': syndromes_arrived
    }

# Simulate backlog for different decoder speeds
print("\n" + "=" * 60)
print("BACKLOG SIMULATION")
print("=" * 60)

t_cycle = 1000  # 1 μs in ns
total_time = 100000  # 100 μs

decode_times = [800, 1000, 1200, 1500, 2000]  # Various decoder speeds

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for t_decode in decode_times:
    result = simulate_backlog(t_cycle, t_decode, total_time)
    ratio = t_decode / t_cycle
    label = f't_decode/t_cycle = {ratio:.1f}'
    plt.plot(result['time']/1000, result['backlog'], label=label, linewidth=2)

plt.xlabel('Time (μs)', fontsize=12)
plt.ylabel('Backlog (syndromes)', fontsize=12)
plt.title('Backlog Accumulation vs Decoder Speed', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Analytical comparison
plt.subplot(1, 2, 2)
ratios = np.linspace(0.8, 2.0, 50)
final_backlogs = []

for r in ratios:
    t_d = r * t_cycle
    if r > 1:
        # Backlog grows
        backlog = (total_time * (t_d - t_cycle)) / (t_cycle * t_d)
    else:
        backlog = 0
    final_backlogs.append(backlog)

plt.plot(ratios, final_backlogs, 'b-', linewidth=2)
plt.axvline(x=1.0, color='r', linestyle='--', label='Real-time threshold')
plt.fill_between(ratios, 0, final_backlogs, where=np.array(ratios)>1,
                  alpha=0.3, color='red', label='Backlog region')
plt.xlabel('t_decode / t_cycle', fontsize=12)
plt.ylabel('Final Backlog after 100 μs', fontsize=12)
plt.title('Backlog vs Decoder Speed Ratio', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backlog_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nBacklog analysis figure saved to 'backlog_analysis.png'")

# =============================================================================
# Part 3: Decoder Latency Scaling
# =============================================================================

def mock_mwpm_decode(n_defects, complexity='cubic'):
    """
    Simulate MWPM decoding time (mock implementation).

    Parameters:
    -----------
    n_defects : int
        Number of syndrome defects
    complexity : str
        'cubic' for O(n^3), 'optimized' for O(n^2 log n)

    Returns:
    --------
    float : simulated decode time in arbitrary units
    """
    if complexity == 'cubic':
        # O(n^3) worst case
        return n_defects ** 3 * 1e-6
    else:
        # O(n^2 log n) optimized
        if n_defects > 0:
            return n_defects ** 2 * np.log(n_defects + 1) * 1e-6
        return 0

def mock_union_find_decode(n_defects):
    """
    Simulate Union-Find decoding time.
    Near-linear: O(n * α(n)) where α is inverse Ackermann.
    """
    if n_defects == 0:
        return 0
    # α(n) < 5 for all practical n, approximate as constant
    alpha = min(4, 1 + np.log(np.log(n_defects + 2) + 1))
    return n_defects * alpha * 1e-6

# Benchmark decoder scaling
print("\n" + "=" * 60)
print("DECODER LATENCY SCALING")
print("=" * 60)

distances = np.arange(3, 31, 2)  # Odd distances from 3 to 29

mwpm_cubic_times = []
mwpm_opt_times = []
union_find_times = []

for d in distances:
    # Expected number of defects ~ p * d^2 for error rate p
    p = 0.01  # 1% error rate
    n_expected = int(p * d * d)  # Rough estimate

    # Multiple samples for averaging
    n_samples = 100

    cubic_samples = [mock_mwpm_decode(np.random.poisson(n_expected), 'cubic')
                     for _ in range(n_samples)]
    opt_samples = [mock_mwpm_decode(np.random.poisson(n_expected), 'optimized')
                   for _ in range(n_samples)]
    uf_samples = [mock_union_find_decode(np.random.poisson(n_expected))
                  for _ in range(n_samples)]

    mwpm_cubic_times.append(np.mean(cubic_samples))
    mwpm_opt_times.append(np.mean(opt_samples))
    union_find_times.append(np.mean(uf_samples))

# Convert to microseconds (arbitrary scaling)
scale = 1e4
mwpm_cubic_times = np.array(mwpm_cubic_times) * scale
mwpm_opt_times = np.array(mwpm_opt_times) * scale
union_find_times = np.array(union_find_times) * scale

plt.figure(figsize=(10, 6))
plt.semilogy(distances, mwpm_cubic_times, 'ro-', label='MWPM O(n³)', linewidth=2)
plt.semilogy(distances, mwpm_opt_times, 'bs-', label='MWPM O(n² log n)', linewidth=2)
plt.semilogy(distances, union_find_times, 'g^-', label='Union-Find O(n α(n))', linewidth=2)

# Real-time threshold line
plt.axhline(y=0.5, color='k', linestyle='--', linewidth=2, label='Real-time budget (500 ns)')

plt.xlabel('Code Distance d', fontsize=12)
plt.ylabel('Decode Time (μs)', fontsize=12)
plt.title('Decoder Latency Scaling with Code Distance', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, which='both')
plt.ylim(1e-3, 1e4)

plt.tight_layout()
plt.savefig('decoder_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

print("Decoder scaling figure saved to 'decoder_scaling.png'")

# =============================================================================
# Part 4: Logical Error Rate with Backlog
# =============================================================================

def logical_error_rate_with_backlog(p, d, backlog, p_L0=None):
    """
    Estimate logical error rate accounting for decoding backlog.

    Parameters:
    -----------
    p : float
        Physical error rate
    d : int
        Code distance
    backlog : int
        Number of unprocessed syndromes
    p_L0 : float
        Baseline logical error rate (computed if None)

    Returns:
    --------
    float : estimated logical error rate
    """
    # Baseline logical error rate (approximate)
    if p_L0 is None:
        # Rough approximation: p_L ~ (p/p_th)^((d+1)/2)
        p_th = 0.103
        if p < p_th:
            p_L0 = (p / p_th) ** ((d + 1) / 2)
        else:
            p_L0 = 1.0

    # Backlog penalty
    # Each unprocessed syndrome allows ~p errors to accumulate
    # Effective distance reduction
    effective_errors = backlog * p
    penalty = (1 + effective_errors / (d / 2)) ** (d / 2)

    return min(1.0, p_L0 * penalty)

# Analyze logical error rate vs backlog
print("\n" + "=" * 60)
print("LOGICAL ERROR RATE VS BACKLOG")
print("=" * 60)

p = 0.005  # 0.5% physical error rate
distances_test = [5, 7, 11, 15]
backlogs = np.arange(0, 1001, 10)

plt.figure(figsize=(10, 6))

for d in distances_test:
    p_L_values = [logical_error_rate_with_backlog(p, d, N) for N in backlogs]
    plt.semilogy(backlogs, p_L_values, linewidth=2, label=f'd = {d}')

plt.xlabel('Backlog (unprocessed syndromes)', fontsize=12)
plt.ylabel('Logical Error Rate', fontsize=12)
plt.title(f'Logical Error Rate vs Backlog (p = {p*100}%)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, which='both')
plt.axhline(y=1e-10, color='k', linestyle='--', alpha=0.5, label='Target: 10⁻¹⁰')

plt.tight_layout()
plt.savefig('logical_error_backlog.png', dpi=150, bbox_inches='tight')
plt.show()

print("Logical error vs backlog figure saved to 'logical_error_backlog.png'")

# =============================================================================
# Part 5: Real-Time Decoder Performance Summary
# =============================================================================

print("\n" + "=" * 60)
print("REAL-TIME DECODER REQUIREMENTS SUMMARY")
print("=" * 60)

# Superconducting system requirements
print("\nSuperconducting System (t_cycle = 1 μs):")
print("-" * 40)

for d in [5, 7, 11, 15, 21]:
    n_qubits = d * d
    n_syndromes = d * d - 1

    # Estimate defects at 0.5% error rate
    n_defects_expected = int(0.005 * n_syndromes)

    # Required operations per decoder type
    mwpm_ops = n_defects_expected ** 3 if n_defects_expected > 0 else 1
    uf_ops = n_defects_expected * 4 if n_defects_expected > 0 else 1  # α(n) ~ 4

    # Assume 1 ns per "operation" on fast hardware
    mwpm_time = mwpm_ops * 1  # ns
    uf_time = uf_ops * 10  # ns (more overhead per op)

    mwpm_ok = "YES" if mwpm_time < 500 else "NO"
    uf_ok = "YES" if uf_time < 500 else "NO"

    print(f"  d={d:2d}: {n_qubits:3d} qubits, ~{n_defects_expected:2d} defects")
    print(f"        MWPM: ~{mwpm_time:,} ns ({mwpm_ok}), UF: ~{uf_time:,} ns ({uf_ok})")

# =============================================================================
# Part 6: Interactive Timing Calculator
# =============================================================================

def analyze_decoder_requirements(d, p_phys, t_cycle_ns, target_p_L):
    """
    Comprehensive analysis of decoder requirements.
    """
    print(f"\n{'='*50}")
    print(f"DECODER REQUIREMENTS ANALYSIS")
    print(f"{'='*50}")
    print(f"Code distance: d = {d}")
    print(f"Physical error rate: p = {p_phys*100}%")
    print(f"Cycle time: t_cycle = {t_cycle_ns} ns")
    print(f"Target logical error rate: {target_p_L:.0e}")
    print("-" * 50)

    # Calculate baseline logical error rate
    p_th = 0.103
    if p_phys < p_th:
        p_L0 = (p_phys / p_th) ** ((d + 1) / 2)
    else:
        p_L0 = 1.0

    print(f"Baseline p_L (real-time): {p_L0:.2e}")

    # Find maximum acceptable backlog
    N_max = 0
    for N in range(10000):
        p_L = logical_error_rate_with_backlog(p_phys, d, N, p_L0)
        if p_L > target_p_L:
            N_max = N - 1
            break

    print(f"Maximum acceptable backlog: {N_max} syndromes")

    # Time margin
    if N_max > 0:
        # Can afford to be slower by factor
        max_ratio = 1 + N_max / (t_cycle_ns / 1000)  # Rough estimate
        print(f"Decoder can be ~{max_ratio:.1f}x slower than real-time")
    else:
        print("Decoder MUST be real-time or faster")

    return {
        'p_L0': p_L0,
        'N_max': N_max,
        'd': d
    }

# Example analysis
analyze_decoder_requirements(d=11, p_phys=0.005, t_cycle_ns=1000, target_p_L=1e-10)
analyze_decoder_requirements(d=21, p_phys=0.003, t_cycle_ns=1000, target_p_L=1e-15)

print("\n" + "=" * 60)
print("LAB COMPLETE")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Real-time constraint | $t_{\text{decode}} < t_{\text{cycle}} - t_{\text{overhead}}$ |
| Backlog growth | $\frac{dN}{dt} = \frac{t_d - t_c}{t_c \cdot t_d}$ |
| Backlog after time $T$ | $N(T) = \frac{T(t_d - t_c)}{t_c \cdot t_d}$ |
| Logical error with backlog | $p_L(\tau) \approx p_L^{(0)} \exp(\tau p / t_c)$ |
| Critical backlog | $N_{\text{crit}} \sim d / (2p)$ |

### Key Insights

1. **Superconducting qubits** impose the strictest timing: ~500 ns decode budget
2. **Backlog accumulation** is exponentially catastrophic for logical error rates
3. **Decoder speed** often matters more than decoder optimality
4. **Code distance scaling** drives the need for near-linear algorithms
5. **Platform choice** significantly affects decoder requirements

---

## Daily Checklist

- [ ] I can calculate timing budgets for different qubit platforms
- [ ] I understand backlog dynamics and when they become critical
- [ ] I can estimate maximum acceptable backlog for a target logical error rate
- [ ] I know the complexity scaling of different decoder algorithms
- [ ] I can benchmark decoder implementations against real-time constraints
- [ ] I understand the accuracy-latency trade-off in decoder selection

---

## Preview: Day 828

Tomorrow we dive into **MWPM Optimization Techniques**, exploring how the Blossom algorithm can be accelerated through:
- Sparse matching graph construction
- Locality exploitation in surface codes
- The PyMatching library implementation
- Practical speedups for approaching real-time operation

---

*"In quantum error correction, a slow decoder is worse than no decoder at all—at least with no decoder, you know you're in trouble."*

---

[← Week 119 Overview](./README.md) | [Day 828: MWPM Optimization →](./Day_828_Tuesday.md)
