# Day 776: Real-Time Decoding Constraints

## Week 111: Decoding Algorithms | Month 28: Advanced Stabilizer Codes

---

## Daily Schedule

| Session | Time | Duration | Focus |
|---------|------|----------|-------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Latency Requirements & Backlog Analysis |
| Afternoon | 1:00 PM - 4:00 PM | 3 hours | Hardware Implementations & Parallel Architectures |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: Decoder Timing Simulation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Calculate** latency requirements for fault-tolerant quantum computation
2. **Analyze** decoder backlog conditions and stability criteria
3. **Design** pipelined decoder architectures for streaming syndromes
4. **Compare** FPGA, ASIC, and GPU implementations for real-time decoding
5. **Implement** windowed decoding strategies for manageable latency
6. **Evaluate** trade-offs between decoder accuracy, latency, and power consumption

---

## Core Content

### 1. The Real-Time Decoding Challenge

Fault-tolerant quantum computation creates a fundamental timing constraint: **corrections must be applied before errors accumulate faster than they can be corrected**.

**The Decoding Loop:**

```
Syndrome Extraction → Classical Decoding → Feedback Correction
     ~1 μs                ???                  ~0.1 μs
```

**Critical Requirement:**

$$\boxed{t_{\text{decode}} < t_{\text{syndrome cycle}} - t_{\text{measurement}} - t_{\text{feedback}}}$$

For typical superconducting systems:
- Syndrome cycle: 1-10 μs
- Measurement: 0.3-1 μs
- Feedback: 0.1-0.5 μs
- **Available decode time: 0.2-8 μs**

### 2. Latency Budget Analysis

A detailed timing breakdown for a distance-17 surface code:

| Component | Duration | Notes |
|-----------|----------|-------|
| Gate operations | 50 ns | Two-qubit gates |
| Syndrome extraction circuit | 500 ns | ~10 gate layers |
| Measurement | 500 ns | Dispersive readout |
| Classical communication | 100 ns | Chip to FPGA |
| **Decoding** | **??? ns** | **This is what we must minimize** |
| Feedback communication | 100 ns | FPGA to chip |
| Correction gate | 50 ns | Single-qubit gate |
| **Total cycle** | **1.3 μs + decode** | |

**Decode Latency Requirement:**

For 1 MHz syndrome rate:
$$t_{\text{decode}} < 1000 \text{ ns} - 1250 \text{ ns} = -250 \text{ ns}$$

This is impossible! We need **pipelining**.

### 3. Pipelining and Parallelism

Real-time decoding uses **pipelining** to hide latency:

**Pipelined Decoding:**

```
Time:     0     1     2     3     4     5     6
         ┌─────────────────────┐
Syndrome │ S1 │ S2 │ S3 │ S4 │ S5 │ S6 │ ...
         ├────┼────┼────┼────┼────┼────┤
Decode   │ -- │ D1 │ D2 │ D3 │ D4 │ D5 │ ...
         ├────┼────┼────┼────┼────┼────┤
Correct  │ -- │ -- │ C1 │ C2 │ C3 │ C4 │ ...
         └────┴────┴────┴────┴────┴────┘

Pipeline delay: 2 cycles
Throughput: 1 syndrome/cycle
```

**Key Insight:** Latency matters less than **throughput**. If we can decode one syndrome per cycle (on average), the pipeline stays stable.

**Parallel Decoding Architecture:**

For large codes, parallelize across:
1. Multiple syndrome regions (spatial)
2. Multiple time slices (temporal)
3. X and Z syndrome graphs (independent)

### 4. Backlog Analysis

If decoding takes longer than the syndrome interval, a **backlog** accumulates:

**Backlog Dynamics:**

Let:
- $T_s$ = syndrome interval
- $T_d$ = decode time (random variable)
- $Q(t)$ = backlog queue length at time $t$

$$Q(t+1) = \max(0, Q(t) + 1 - \mathbb{1}[T_d < T_s + Q(t) \cdot T_s])$$

**Stability Condition:**

The backlog is stable if:
$$\boxed{\mathbb{E}[T_d] < T_s}$$

Average decode time must be less than syndrome interval.

**Backlog Growth During High-Error Periods:**

When physical error rate spikes (e.g., cosmic ray event), decode time increases and backlog grows. The system must:
1. Buffer syndromes
2. Eventually catch up during quiescent periods
3. Or declare a failure if buffer overflows

### 5. FPGA Implementation

FPGAs (Field-Programmable Gate Arrays) are the workhorse for real-time decoding:

**FPGA Advantages:**
- Low latency (ns-scale)
- Deterministic timing
- Reconfigurable for different codes
- Moderate power (~10W)

**FPGA Decoder Architecture:**

```
┌─────────────────────────────────────────────────┐
│                    FPGA                         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│  │Syndrome │ → │ Graph   │ → │ MWPM/   │ → Out │
│  │ Buffer  │   │ Builder │   │ UF Core │       │
│  └─────────┘   └─────────┘   └─────────┘       │
│       ↑                                         │
│   Syndrome input from quantum chip              │
└─────────────────────────────────────────────────┘
```

**Timing for MWPM on FPGA:**

| Code Distance | Vertices (avg) | FPGA Cycles | Latency @ 200MHz |
|---------------|----------------|-------------|------------------|
| 5 | ~5 | ~1000 | 5 μs |
| 9 | ~10 | ~5000 | 25 μs |
| 17 | ~30 | ~50000 | 250 μs |

MWPM struggles for large codes! Union-Find is better:

**Union-Find on FPGA:**

| Code Distance | Latency @ 200MHz |
|---------------|------------------|
| 5 | 0.5 μs |
| 9 | 1 μs |
| 17 | 3 μs |
| 33 | 8 μs |

### 6. ASIC Implementation

For production quantum computers, ASICs (Application-Specific Integrated Circuits) offer ultimate performance:

**ASIC Advantages:**
- Lowest latency (<1 μs even for large codes)
- Lowest power (~1W)
- Highest throughput

**ASIC Disadvantages:**
- Fixed functionality (no reconfiguration)
- High development cost ($1M+)
- Long development time (1-2 years)

**Reported ASIC Results:**

| Design | Code | Latency | Power | Process |
|--------|------|---------|-------|---------|
| IBM (2023) | d=17 surface | 1.1 μs | 1.4 W | 14nm |
| Google (2024) | d=25 surface | 0.8 μs | 2.1 W | 12nm |

### 7. GPU Implementation

GPUs offer massive parallelism but higher latency:

**GPU Characteristics:**
- High throughput (millions of decodes/second)
- Higher latency (10-100 μs per decode)
- Good for batch processing and simulation

**When to Use GPUs:**
1. Simulation and characterization
2. Training neural decoders
3. Research and development
4. NOT for real-time feedback (usually)

**Hybrid Architectures:**

```
Real-time path (FPGA): Syndrome → Fast UF decode → Correction
                              ↓
Offline path (GPU):    Buffer → Batch MWPM → Calibration update
```

### 8. Windowed Decoding

For very large codes or long computations, use **windowed decoding**:

**Sliding Window Approach:**

Instead of decoding the entire space-time syndrome history:
1. Maintain a window of $W$ recent syndrome rounds
2. Decode only within the window
3. Slide window as new syndromes arrive
4. Apply corrections with delay

**Window Size Trade-off:**

| Window Size $W$ | Threshold | Latency | Memory |
|-----------------|-----------|---------|--------|
| 1 (instantaneous) | ~8% | Minimal | $O(d^2)$ |
| $d$ | ~9.5% | Moderate | $O(d^3)$ |
| $2d$ | ~10.0% | Higher | $O(d^3)$ |
| $\infty$ | ~10.3% | Unbounded | $O(d^2 T)$ |

Optimal window size balances threshold and latency.

---

## Worked Examples

### Example 1: Latency Budget Calculation

**Problem:** Design the latency budget for a distance-9 surface code with 1 μs syndrome cycle.

**Solution:**

**Fixed latencies:**
- Syndrome extraction: 400 ns
- Measurement: 300 ns
- Classical communication (both ways): 200 ns
- Correction gates: 50 ns

**Total fixed:** 950 ns

**Available for decoding:** 1000 - 950 = **50 ns**

This is extremely tight! Options:
1. Use Union-Find (can achieve ~50 ns on fast FPGA)
2. Pipeline over multiple cycles
3. Use faster hardware (ASIC)

**Pipelined solution:**

Allow 2-cycle decode latency:
- Available time: 1000 + 1000 - 950 = 1050 ns
- Comfortable for Union-Find (~100 ns)
- Acceptable for simple MWPM (~500 ns)

### Example 2: Backlog Probability

**Problem:** A decoder has mean decode time 0.8 μs with standard deviation 0.3 μs. Syndrome interval is 1 μs. Calculate the probability of backlog growth per cycle.

**Solution:**

Assuming decode time is normally distributed (approximation):
- Mean: $\mu = 0.8$ μs
- Std: $\sigma = 0.3$ μs
- Threshold: $T_s = 1.0$ μs

Probability of exceeding threshold:
$$P(T_d > T_s) = P\left(Z > \frac{1.0 - 0.8}{0.3}\right) = P(Z > 0.67) \approx 0.25$$

So 25% of cycles may add to backlog!

**Long-term behavior:**

Backlog grows when $T_d > T_s$ and shrinks otherwise.
- Growth rate: $+1$ with probability 0.25
- Shrink rate: $-1$ with probability 0.75 (when backlog > 0)

Average backlog change: $0.25 \times 1 - 0.75 \times 1 = -0.5$ (when backlog > 0)

System is stable but experiences frequent backlog fluctuations.

### Example 3: Window Size Selection

**Problem:** For a distance-17 code with 10% physical error rate, what window size achieves 99.9% of optimal threshold?

**Solution:**

Optimal threshold (infinite window): $p_{\text{th}}^{\infty} \approx 10.3\%$

Target: $0.999 \times 10.3\% = 10.29\%$

From empirical data, threshold vs window size:

| $W/d$ | Threshold |
|-------|-----------|
| 0.5 | 9.2% |
| 1.0 | 9.8% |
| 1.5 | 10.1% |
| 2.0 | 10.2% |
| 3.0 | 10.28% |

**Recommendation:** Window size $W = 3d = 51$ rounds achieves near-optimal threshold.

**Memory requirement:** $O(d^2 \times W) = O(17^2 \times 51) \approx 15,000$ syndrome bits = 2 KB

**Latency:** 51 syndrome cycles $\times$ 1 μs = 51 μs decode delay

This is acceptable for most algorithms (not for feed-forward Clifford gates requiring immediate correction).

---

## Practice Problems

### Level A: Direct Application

**A1.** Calculate the maximum decode time for a system with:
- Syndrome cycle: 2 μs
- Measurement time: 500 ns
- Feedback latency: 300 ns
- Pipeline depth: 3 cycles

**A2.** An FPGA decoder runs at 250 MHz and requires 2000 clock cycles. What is the decode latency?

**A3.** A backlog queue has 5 pending syndromes. If each decode takes 0.9 μs and syndrome interval is 1 μs, how long until the backlog clears (assuming no new errors)?

### Level B: Intermediate Analysis

**B1.** Design a two-stage pipelined decoder where stage 1 handles "easy" syndromes (few defects) and stage 2 handles "hard" syndromes. What fraction of syndromes should go to each stage?

**B2.** Compare the power consumption of decoding a distance-17 code using:
- FPGA (10W, 200 MHz, 5000 cycles)
- GPU (150W, 1 GHz, 100 cycles batched)
- ASIC (2W, 500 MHz, 500 cycles)

Calculate energy per decode.

**B3.** Analyze the impact of clock frequency variation (jitter) on decoder backlog stability. If the FPGA clock varies by ±5%, how does this affect stability margins?

### Level C: Advanced Problems

**C1.** Design a speculative decoding scheme where the decoder predicts the most likely syndrome before measurement completes. Analyze the accuracy vs speedup trade-off.

**C2.** Prove that for Poisson-distributed decode times with mean $\lambda$, the backlog queue length follows an M/M/1 queue with stability condition $\lambda < \mu$ (where $\mu$ is syndrome rate).

**C3.** Design an adaptive window decoder that adjusts window size based on current error rate and backlog. Specify the control algorithm and analyze its stability.

---

## Computational Lab: Decoder Timing Simulation

```python
"""
Day 776 Computational Lab: Real-Time Decoding Simulation
Simulating timing constraints, backlog, and pipelined decoding

This lab models the dynamics of real-time quantum error correction
systems and analyzes decoder performance under realistic conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class DecoderTiming:
    """Timing characteristics of a decoder."""
    mean_latency: float  # Mean decode time (microseconds)
    std_latency: float   # Standard deviation
    min_latency: float   # Minimum (best case)
    max_latency: float   # Maximum (worst case)

    def sample(self) -> float:
        """Sample a decode time from the distribution."""
        # Log-normal distribution (always positive, right-skewed)
        log_mean = np.log(self.mean_latency) - 0.5 * np.log(1 + (self.std_latency/self.mean_latency)**2)
        log_std = np.sqrt(np.log(1 + (self.std_latency/self.mean_latency)**2))
        sample = np.random.lognormal(log_mean, log_std)
        return np.clip(sample, self.min_latency, self.max_latency)


@dataclass
class SystemParameters:
    """Parameters for the QEC system."""
    syndrome_interval: float  # Microseconds between syndromes
    measurement_time: float   # Time for syndrome measurement
    feedback_latency: float   # Time to apply correction
    code_distance: int        # Code distance


class BacklogSimulator:
    """
    Simulates decoder backlog dynamics.
    """

    def __init__(self, system: SystemParameters, decoder: DecoderTiming):
        self.system = system
        self.decoder = decoder
        self.reset()

    def reset(self):
        """Reset simulation state."""
        self.backlog_queue = deque()
        self.time = 0
        self.backlog_history = []
        self.latency_history = []

    def simulate(self, n_syndromes: int) -> Dict:
        """
        Simulate n_syndromes arriving.

        Returns:
            Statistics dictionary
        """
        self.reset()

        decode_finish_time = 0  # When current decode completes

        for i in range(n_syndromes):
            arrival_time = i * self.system.syndrome_interval

            # Add syndrome to queue
            self.backlog_queue.append(arrival_time)

            # Process queue if decoder is free
            while self.backlog_queue and decode_finish_time <= arrival_time:
                syndrome_arrival = self.backlog_queue.popleft()
                decode_start = max(decode_finish_time, syndrome_arrival)
                decode_time = self.decoder.sample()
                decode_finish_time = decode_start + decode_time

                latency = decode_finish_time - syndrome_arrival
                self.latency_history.append(latency)

            # Record backlog
            self.backlog_history.append(len(self.backlog_queue))

        return {
            'mean_backlog': np.mean(self.backlog_history),
            'max_backlog': np.max(self.backlog_history),
            'mean_latency': np.mean(self.latency_history),
            'max_latency': np.max(self.latency_history),
            'backlog_history': self.backlog_history,
            'latency_history': self.latency_history
        }


class PipelinedDecoder:
    """
    Simulates a pipelined decoder with multiple stages.
    """

    def __init__(self, n_stages: int, stage_latency: float):
        self.n_stages = n_stages
        self.stage_latency = stage_latency
        self.pipeline = [None] * n_stages
        self.output_queue = deque()

    def tick(self, new_syndrome: Optional[int] = None) -> Optional[int]:
        """
        Advance pipeline by one cycle.

        Args:
            new_syndrome: New syndrome ID to insert

        Returns:
            Completed syndrome ID if any
        """
        # Output from last stage
        output = self.pipeline[-1]
        if output is not None:
            self.output_queue.append(output)

        # Shift pipeline
        for i in range(self.n_stages - 1, 0, -1):
            self.pipeline[i] = self.pipeline[i-1]

        # Insert new syndrome
        self.pipeline[0] = new_syndrome

        return output

    def get_latency(self) -> float:
        """Return total pipeline latency."""
        return self.n_stages * self.stage_latency


def analyze_backlog():
    """Analyze backlog behavior under various conditions."""
    print("=" * 60)
    print("Backlog Analysis")
    print("=" * 60)

    system = SystemParameters(
        syndrome_interval=1.0,  # 1 μs
        measurement_time=0.3,
        feedback_latency=0.2,
        code_distance=17
    )

    # Test different decoder speeds
    decoder_configs = [
        ("Fast (μ=0.5μs)", DecoderTiming(0.5, 0.1, 0.3, 1.0)),
        ("Medium (μ=0.8μs)", DecoderTiming(0.8, 0.2, 0.4, 2.0)),
        ("Slow (μ=1.1μs)", DecoderTiming(1.1, 0.3, 0.5, 3.0)),
        ("Variable (μ=0.7μs, σ=0.4)", DecoderTiming(0.7, 0.4, 0.2, 3.0)),
    ]

    n_syndromes = 10000

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for (name, decoder), ax in zip(decoder_configs, axes.flat):
        simulator = BacklogSimulator(system, decoder)
        results = simulator.simulate(n_syndromes)

        print(f"\n{name}:")
        print(f"  Mean backlog: {results['mean_backlog']:.2f}")
        print(f"  Max backlog: {results['max_backlog']}")
        print(f"  Mean latency: {results['mean_latency']:.2f} μs")
        print(f"  Max latency: {results['max_latency']:.2f} μs")

        # Plot backlog over time
        ax.plot(results['backlog_history'][:1000], 'b-', alpha=0.7)
        ax.set_xlabel('Syndrome Index')
        ax.set_ylabel('Backlog Queue Length')
        ax.set_title(f'{name}\nMean backlog: {results["mean_backlog"]:.2f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backlog_analysis.png', dpi=150)
    plt.show()


def analyze_pipeline_depth():
    """Analyze impact of pipeline depth on throughput."""
    print("\n" + "=" * 60)
    print("Pipeline Depth Analysis")
    print("=" * 60)

    syndrome_interval = 1.0  # μs
    decode_time = 3.0  # μs (without pipelining)

    pipeline_depths = range(1, 8)
    throughputs = []
    latencies = []

    for depth in pipeline_depths:
        # With pipelining, we can overlap decodes
        stage_time = decode_time / depth

        if stage_time <= syndrome_interval:
            throughput = 1.0 / syndrome_interval  # Full throughput
        else:
            throughput = 1.0 / stage_time  # Limited by slowest stage

        latency = depth * stage_time

        throughputs.append(throughput)
        latencies.append(latency)

        print(f"Depth {depth}: throughput={throughput:.2f} syn/μs, latency={latency:.1f} μs")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(pipeline_depths, throughputs, color='steelblue')
    ax1.axhline(y=1.0/syndrome_interval, color='r', linestyle='--',
                label=f'Target: {1.0/syndrome_interval:.1f} syn/μs')
    ax1.set_xlabel('Pipeline Depth')
    ax1.set_ylabel('Throughput (syndromes/μs)')
    ax1.set_title('Throughput vs Pipeline Depth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(pipeline_depths, latencies, color='coral')
    ax2.set_xlabel('Pipeline Depth')
    ax2.set_ylabel('Total Latency (μs)')
    ax2.set_title('Latency vs Pipeline Depth')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pipeline_analysis.png', dpi=150)
    plt.show()


def compare_hardware():
    """Compare different hardware implementations."""
    print("\n" + "=" * 60)
    print("Hardware Comparison")
    print("=" * 60)

    # Hardware specifications for distance-17 code
    hardware = {
        'FPGA (UF)': {
            'latency': 3.0,    # μs
            'power': 10.0,     # Watts
            'cost': 5000,      # $
            'flexibility': 'High',
        },
        'FPGA (MWPM)': {
            'latency': 50.0,
            'power': 15.0,
            'cost': 5000,
            'flexibility': 'High',
        },
        'GPU': {
            'latency': 100.0,
            'power': 150.0,
            'cost': 10000,
            'flexibility': 'Very High',
        },
        'ASIC (UF)': {
            'latency': 0.5,
            'power': 1.0,
            'cost': 1000000,
            'flexibility': 'None',
        },
        'ASIC (MWPM)': {
            'latency': 5.0,
            'power': 2.0,
            'cost': 1000000,
            'flexibility': 'None',
        },
    }

    print("\nHardware specifications for distance-17 surface code:")
    print("-" * 70)
    print(f"{'Hardware':<20} {'Latency (μs)':<15} {'Power (W)':<12} {'Cost ($)':<15}")
    print("-" * 70)

    for name, specs in hardware.items():
        print(f"{name:<20} {specs['latency']:<15.1f} {specs['power']:<12.1f} {specs['cost']:<15,}")

    # Calculate energy per decode
    print("\nEnergy per decode:")
    print("-" * 50)

    for name, specs in hardware.items():
        energy = specs['power'] * specs['latency'] * 1e-6  # Joules
        print(f"{name:<20} {energy*1e6:.3f} μJ")

    # Visualization
    names = list(hardware.keys())
    latencies = [hardware[n]['latency'] for n in names]
    powers = [hardware[n]['power'] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['blue', 'cyan', 'green', 'red', 'orange']

    ax1.barh(names, latencies, color=colors)
    ax1.set_xlabel('Latency (μs)')
    ax1.set_title('Decoder Latency by Hardware')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, axis='x')

    ax2.barh(names, powers, color=colors)
    ax2.set_xlabel('Power (W)')
    ax2.set_title('Power Consumption by Hardware')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('hardware_comparison.png', dpi=150)
    plt.show()


def windowed_decoding_analysis():
    """Analyze windowed decoding performance."""
    print("\n" + "=" * 60)
    print("Windowed Decoding Analysis")
    print("=" * 60)

    code_distance = 17

    # Simulated threshold vs window size (based on literature)
    window_sizes = [1, 2, 4, 8, 12, 17, 25, 34, 50]
    thresholds = [0.078, 0.085, 0.092, 0.097, 0.099, 0.100, 0.101, 0.102, 0.103]

    # Memory requirements (syndrome bits)
    memories = [w * code_distance * code_distance for w in window_sizes]

    # Latency (assuming 1μs per syndrome round)
    latencies = window_sizes  # μs

    print("\nWindow Size vs Performance:")
    print("-" * 60)
    print(f"{'Window (rounds)':<18} {'Threshold':<12} {'Memory (KB)':<15} {'Latency (μs)':<12}")
    print("-" * 60)

    for w, th, mem, lat in zip(window_sizes, thresholds, memories, latencies):
        print(f"{w:<18} {th*100:<12.1f}% {mem/8/1024:<15.2f} {lat:<12}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Threshold vs window
    axes[0].plot(window_sizes, [t*100 for t in thresholds], 'bo-', linewidth=2, markersize=8)
    axes[0].axhline(y=10.3, color='r', linestyle='--', label='Optimal (10.3%)')
    axes[0].set_xlabel('Window Size (syndrome rounds)')
    axes[0].set_ylabel('Threshold (%)')
    axes[0].set_title('Threshold vs Window Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Memory vs window
    axes[1].plot(window_sizes, [m/8/1024 for m in memories], 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Window Size (syndrome rounds)')
    axes[1].set_ylabel('Memory (KB)')
    axes[1].set_title('Memory Requirement vs Window Size')
    axes[1].grid(True, alpha=0.3)

    # Latency vs window
    axes[2].plot(window_sizes, latencies, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Window Size (syndrome rounds)')
    axes[2].set_ylabel('Decode Latency (μs)')
    axes[2].set_title('Latency vs Window Size')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('windowed_decoding.png', dpi=150)
    plt.show()


def timing_jitter_analysis():
    """Analyze impact of timing jitter on decoder performance."""
    print("\n" + "=" * 60)
    print("Timing Jitter Analysis")
    print("=" * 60)

    base_interval = 1.0  # μs
    jitter_levels = [0.0, 0.01, 0.02, 0.05, 0.10]  # Fraction of interval

    n_syndromes = 5000

    system = SystemParameters(
        syndrome_interval=base_interval,
        measurement_time=0.3,
        feedback_latency=0.2,
        code_distance=17
    )

    decoder = DecoderTiming(0.75, 0.15, 0.4, 1.5)

    results = []

    for jitter in jitter_levels:
        # Modify system to add jitter
        class JitteredSimulator(BacklogSimulator):
            def simulate(self, n_syndromes):
                self.reset()
                decode_finish_time = 0
                actual_time = 0

                for i in range(n_syndromes):
                    # Add jitter to interval
                    interval = self.system.syndrome_interval * (1 + np.random.uniform(-jitter, jitter))
                    actual_time += interval

                    self.backlog_queue.append(actual_time)

                    while self.backlog_queue and decode_finish_time <= actual_time:
                        syndrome_arrival = self.backlog_queue.popleft()
                        decode_start = max(decode_finish_time, syndrome_arrival)
                        decode_time = self.decoder.sample()
                        decode_finish_time = decode_start + decode_time
                        self.latency_history.append(decode_finish_time - syndrome_arrival)

                    self.backlog_history.append(len(self.backlog_queue))

                return {
                    'mean_backlog': np.mean(self.backlog_history),
                    'max_backlog': np.max(self.backlog_history),
                    'mean_latency': np.mean(self.latency_history),
                    'max_latency': np.max(self.latency_history),
                }

        sim = JitteredSimulator(system, decoder)
        sim_jitter = jitter  # Closure capture
        res = sim.simulate(n_syndromes)
        results.append(res)

        print(f"Jitter {jitter*100:.0f}%: mean_backlog={res['mean_backlog']:.2f}, "
              f"max_backlog={res['max_backlog']}, max_latency={res['max_latency']:.2f}μs")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar([f"{j*100:.0f}%" for j in jitter_levels],
            [r['mean_backlog'] for r in results], color='steelblue')
    ax1.set_xlabel('Timing Jitter')
    ax1.set_ylabel('Mean Backlog')
    ax1.set_title('Backlog vs Timing Jitter')
    ax1.grid(True, alpha=0.3)

    ax2.bar([f"{j*100:.0f}%" for j in jitter_levels],
            [r['max_latency'] for r in results], color='coral')
    ax2.set_xlabel('Timing Jitter')
    ax2.set_ylabel('Maximum Latency (μs)')
    ax2.set_title('Max Latency vs Timing Jitter')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('jitter_analysis.png', dpi=150)
    plt.show()


def real_time_budget():
    """Create a detailed timing budget visualization."""
    print("\n" + "=" * 60)
    print("Real-Time Timing Budget")
    print("=" * 60)

    # Timing breakdown (microseconds)
    budget = {
        'Gate Layer 1-3': 0.15,
        'Gate Layer 4-6': 0.15,
        'Gate Layer 7-10': 0.20,
        'Measurement': 0.50,
        'Classical TX': 0.05,
        'FPGA Decode': 0.30,
        'Classical RX': 0.05,
        'Correction': 0.05,
        'Margin': 0.05,
    }

    total = sum(budget.values())
    print(f"\nTotal syndrome cycle: {total:.2f} μs")
    print("\nBreakdown:")
    for name, time_us in budget.items():
        print(f"  {name:<20} {time_us:.2f} μs ({time_us/total*100:.1f}%)")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(budget)))

    left = 0
    for (name, time_us), color in zip(budget.items(), colors):
        ax.barh(0, time_us, left=left, color=color, edgecolor='black', height=0.5)
        if time_us > 0.1:
            ax.text(left + time_us/2, 0, f'{name}\n{time_us:.2f}μs',
                   ha='center', va='center', fontsize=9)
        left += time_us

    ax.set_xlim(0, total * 1.05)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Time (μs)')
    ax.set_title('Real-Time QEC Timing Budget (Distance-17 Surface Code)', fontsize=14)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('timing_budget.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    analyze_backlog()
    analyze_pipeline_depth()
    compare_hardware()
    windowed_decoding_analysis()
    timing_jitter_analysis()
    real_time_budget()

    print("\n" + "=" * 60)
    print("Lab Complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Decode Time Constraint | $$t_{\text{decode}} < t_{\text{syndrome}} - t_{\text{fixed}}$$ |
| Backlog Stability | $$\mathbb{E}[T_d] < T_s$$ |
| Pipeline Throughput | $$\text{throughput} = \min(1/T_s, 1/T_{\text{stage}})$$ |
| Pipeline Latency | $$T_{\text{total}} = N_{\text{stages}} \times T_{\text{stage}}$$ |
| Energy per Decode | $$E = P \times t_{\text{decode}}$$ |
| Window Threshold | $$p_{\text{th}}(W) \approx p_{\text{th}}^{\infty}(1 - c/W)$$ |

### Key Takeaways

1. **Latency budgets are tight**: Sub-microsecond decoding often required
2. **Pipelining enables throughput**: Can tolerate multi-cycle latency
3. **Backlog must be stable**: Average decode time < syndrome interval
4. **FPGAs are the workhorse**: Good balance of speed, power, flexibility
5. **ASICs for production**: Ultimate performance when design is fixed
6. **Windowed decoding trades accuracy for latency**: Choose window carefully

---

## Daily Checklist

- [ ] Calculated latency budgets for realistic systems
- [ ] Analyzed backlog dynamics and stability
- [ ] Understood pipelining benefits and trade-offs
- [ ] Compared FPGA, GPU, and ASIC implementations
- [ ] Implemented timing simulation
- [ ] Analyzed windowed decoding performance
- [ ] Completed practice problems (at least Level A and B)

---

## Preview: Day 777

Tomorrow in our **Week 111 Synthesis**, we bring together all decoder algorithms studied this week. We'll create a comprehensive comparison table, discuss when to use each decoder, and prepare for the advanced topics in Week 112.

Key synthesis questions:
- Which decoder is best for which use case?
- How do we choose the right trade-off between accuracy and speed?
- What are the open problems in decoder research?

---

*Day 776 of 2184 | Week 111 | Month 28 | Year 2: Advanced Quantum Science*
