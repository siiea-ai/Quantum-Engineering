# Day 833: Week 119 Synthesis

## Week 119: Real-Time Decoding | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Schedule (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 2.5 hours | Comprehensive review, integration |
| **Afternoon** | 2.5 hours | Research frontiers, open problems |
| **Evening** | 2 hours | Capstone implementation lab |

---

## Learning Objectives

By the end of Day 833, you will be able to:

1. **Synthesize** all real-time decoding concepts into a coherent framework
2. **Design** a complete decoder system from specifications
3. **Evaluate** decoder choices for specific quantum computing platforms
4. **Identify** current research frontiers and open problems
5. **Implement** a full decoder pipeline in software
6. **Plan** for future developments in real-time quantum error correction

---

## Core Content

### 1. Week 119 Synthesis: The Complete Picture

This week we explored the critical challenge of real-time decoding for fault-tolerant quantum computing:

```
┌─────────────────────────────────────────────────────────────────┐
│                  REAL-TIME DECODING LANDSCAPE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Day 827: LATENCY CONSTRAINTS                                   │
│  ├── t_decode < t_cycle - t_overhead                           │
│  ├── Backlog dynamics: exponential error accumulation          │
│  └── Platform-specific budgets: 500 ns (superconducting)       │
│                                                                  │
│  Day 828: MWPM OPTIMIZATION                                     │
│  ├── Sparse graph construction (locality exploitation)         │
│  ├── Blossom V algorithm: O(n² log n) achievable              │
│  └── PyMatching: production-ready implementation               │
│                                                                  │
│  Day 829: UNION-FIND                                            │
│  ├── O(n · α(n)) ≈ O(n) complexity                             │
│  ├── Cluster growth and fusion algorithm                       │
│  └── Threshold: 9.9% (vs 10.3% for MWPM)                       │
│                                                                  │
│  Day 830: NEURAL NETWORKS                                        │
│  ├── Pattern recognition approach to decoding                  │
│  ├── O(1) inference time (independent of syndrome)             │
│  └── Approaching optimal with sufficient training              │
│                                                                  │
│  Day 831: SLIDING WINDOW                                         │
│  ├── Bounded memory for infinite computation                   │
│  ├── W = O(d) window size sufficient                           │
│  └── Streaming enables continuous operation                    │
│                                                                  │
│  Day 832: HARDWARE CO-DESIGN                                     │
│  ├── FPGA: 50-100 ns achievable for d ≤ 21                     │
│  ├── ASIC: 20-50 ns with custom silicon                        │
│  └── Cryogenic: reduced latency, limited power                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Decoder Selection Framework

Choose a decoder based on your requirements:

```
                        ┌─────────────────┐
                        │ What's your     │
                        │ priority?       │
                        └────────┬────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
      ┌──────────┐        ┌──────────┐        ┌──────────┐
      │ Accuracy │        │ Speed    │        │ Adaptive │
      └────┬─────┘        └────┬─────┘        └────┬─────┘
           │                   │                   │
           ▼                   ▼                   ▼
      ┌─────────┐        ┌──────────┐        ┌──────────┐
      │  MWPM   │        │ Union-   │        │ Neural   │
      │ (10.3%) │        │ Find     │        │ Network  │
      └─────────┘        │ (9.9%)   │        │ (~10%)   │
                         └──────────┘        └──────────┘
```

**Decision table**:

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Superconducting, real-time | Union-Find | Fast enough, good threshold |
| Trapped ion | MWPM | More time available |
| Noisy hardware | Neural Network | Can learn actual noise |
| Research/simulation | MWPM | Maximum accuracy |
| Memory constrained | Sliding Window + UF | Bounded resources |
| Ultra-low latency | ASIC Union-Find | Custom hardware |

### 3. Complete System Architecture

A production decoder system includes:

```
┌──────────────────────────────────────────────────────────────────┐
│                    DECODER SYSTEM ARCHITECTURE                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Syndrome   │    │   Decoder   │    │ Correction  │          │
│  │  Interface  │───▶│   Engine    │───▶│  Output     │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│        │                  │                   │                  │
│        ▼                  ▼                   ▼                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Syndrome   │    │   Window    │    │   Pauli     │          │
│  │  Buffer     │    │   Manager   │    │   Frame     │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                           │                                      │
│                           ▼                                      │
│                    ┌─────────────┐                               │
│                    │   Metrics   │                               │
│                    │   Monitor   │                               │
│                    └─────────────┘                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Components**:
1. **Syndrome Interface**: Receives raw measurement data, converts to syndrome bits
2. **Decoder Engine**: Core matching/classification algorithm
3. **Correction Output**: Formats and routes correction signals
4. **Syndrome Buffer**: Manages sliding window storage
5. **Window Manager**: Controls commitment and stream processing
6. **Pauli Frame**: Tracks cumulative corrections (software layer)
7. **Metrics Monitor**: Logs performance, detects anomalies

### 4. Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Real-time constraint | $t_{\text{decode}} < t_{\text{cycle}} - t_{\text{overhead}}$ |
| Backlog accumulation | $N(T) = T(t_d - t_c)/(t_c \cdot t_d)$ |
| MWPM complexity | $O(n^3)$ worst, $O(n^2 \log n)$ optimized |
| Union-Find complexity | $O(n \cdot \alpha(n))$ where $\alpha(n) \leq 4$ |
| Neural inference | $O(1)$ after training |
| Optimal window size | $W_{\text{opt}} = O(d)$ |
| Logical error with backlog | $p_L(\tau) \approx p_L^{(0)} \exp(\tau p / t_c)$ |

### 5. Threshold Comparison

The effective threshold depends on both algorithm and latency:

$$\boxed{p_{\text{th}}^{\text{eff}} = p_{\text{th}}^{\text{alg}} \cdot \mathcal{D}(t_d / t_c)}$$

where $\mathcal{D}(r)$ is the degradation from backlog:

| Decoder | $p_{\text{th}}^{\text{alg}}$ | Real-time ($r < 1$)? | Effective threshold |
|---------|------------------------------|---------------------|---------------------|
| ML (optimal) | 10.9% | No | N/A |
| MWPM | 10.3% | Sometimes | ~10.3% or degraded |
| Union-Find | 9.9% | Yes | ~9.9% |
| Neural (large) | ~10.0% | Sometimes | ~10.0% or degraded |
| Neural (small) | ~8.5% | Yes | ~8.5% |

### 6. Research Frontiers

**Active research areas (2024-2026)**:

1. **Belief propagation decoders**
   - Near-optimal accuracy with iterative message passing
   - Parallelizable, potentially faster than MWPM
   - Challenge: convergence for loopy graphs

2. **Transformer-based decoders**
   - Attention mechanisms for long-range correlations
   - Self-supervised training on syndrome histories
   - Challenge: hardware implementation

3. **Correlated error decoding**
   - Beyond independent error models
   - Exploit spatial/temporal correlations
   - Challenge: characterization of correlations

4. **Fault-tolerant decoder design**
   - Classical errors in the decoder itself
   - Redundant decoder architectures
   - Challenge: overhead vs reliability

5. **Scalable cryogenic processing**
   - Beyond 4K: processing at mK stage
   - Superconducting classical logic
   - Challenge: power dissipation

### 7. Open Problems

**Fundamental questions**:

1. What is the minimum hardware complexity for achieving 10% threshold in real-time?

2. Can machine learning match ML-optimal decoding in all regimes?

3. How do we decode correlated errors arising from cosmic rays and other macroscopic events?

4. What is the optimal co-design of code and decoder for a given hardware budget?

5. Can we achieve fault-tolerant decoding (classical errors in decoder don't cause logical errors)?

---

## Worked Examples

### Example 1: Complete Decoder Specification

**Problem**: Design a decoder system for a superconducting quantum computer with:
- 100 logical qubits at distance 15
- Physical error rate: 0.3%
- Syndrome cycle: 800 ns
- Target logical error rate: $10^{-10}$ per logical operation

**Solution**:

**Step 1**: Verify target is achievable

At $p = 0.3\%$ and $d = 15$:
$$p_L \approx \left(\frac{0.003}{0.099}\right)^{8} \approx (0.030)^8 \approx 6.6 \times 10^{-13}$$

This exceeds the target. Good.

**Step 2**: Latency budget

- Cycle time: 800 ns
- Communication (assume room temp): 200 ns
- Feedback: 50 ns
- **Decode budget**: 800 - 200 - 50 = 550 ns

**Step 3**: Decoder selection

For d=15 at 550 ns:
- MWPM: ~1-10 μs → Too slow
- Union-Find: ~100-200 ns → Fits
- Neural: ~50-100 ns → Fits

Choose **Union-Find** for proven threshold.

**Step 4**: Hardware specification

- FPGA implementation at 400 MHz
- Cycles available: $550 \text{ ns} \times 0.4 \text{ GHz} = 220$ cycles
- Pipeline stages: 15-20 (comfortable margin)
- Resources: ~30K LUTs (fits in medium FPGA)

**Step 5**: Streaming configuration

- Window size: $W = 2d = 30$ rounds
- Commitment depth: $C = d = 15$ rounds
- Memory: $30 \times 224 \times 8 = 54$ KB per logical qubit
- Total memory: $100 \times 54$ KB = 5.4 MB (use BRAM)

**Step 6**: Parallelization

- 100 logical qubits, each needs decoder
- Options:
  - 100 parallel decoders: ~3M LUTs (large FPGA)
  - Time-multiplexed: 10 decoders × 10 qubits each (sequentially)
  - Hybrid: depends on syndrome complexity

Choose: **20 parallel decoders** handling 5 qubits each in rotation.

$$\boxed{\text{20 Union-Find decoders on single large FPGA}}$$

### Example 2: Threshold Degradation Analysis

**Problem**: A decoder achieves 9.5% threshold but runs 20% slower than real-time (backlog accumulates). At what physical error rate does this equal a slower 10.0% threshold decoder running in real-time?

**Solution**:

Fast decoder (with backlog): effective threshold
$$p_{\text{th,eff}}^{\text{fast}} = 9.5\% \times \mathcal{D}(1.2)$$

Degradation function (approximate):
$$\mathcal{D}(1.2) = e^{-\lambda(1.2 - 1)} = e^{-0.2\lambda}$$

With $\lambda \approx 5$ (typical):
$$\mathcal{D}(1.2) \approx e^{-1} \approx 0.37$$

Effective threshold:
$$p_{\text{th,eff}}^{\text{fast}} \approx 9.5\% \times 0.37 = 3.5\%$$

Slow decoder (real-time): 10.0%

The slow real-time decoder is far better!

**Crossover point**: When does fast decoder win?

$$9.5\% \times \mathcal{D}(r) > 10.0\%$$

This requires $\mathcal{D}(r) > 1.05$, which never happens for $r > 1$.

$$\boxed{\text{Real-time operation is always preferable to higher threshold with backlog}}$$

### Example 3: Neural vs Union-Find Break-Even

**Problem**: A neural decoder achieves 8% threshold with 50 ns latency. Union-Find achieves 10% threshold with 100 ns latency. At what syndrome complexity does neural win?

**Solution**:

Neural: Fixed 50 ns regardless of defects
Union-Find: $t = 100 + 10 \cdot k$ ns for $k$ defects (approximate)

Break-even:
$$50 < 100 + 10k$$
$$k > -5$$

Union-Find is always slower (for $k \geq 0$). But accuracy matters!

Compare logical error rates at $p = 5\%$, $d = 7$:

Neural:
$$p_L^{\text{NN}} = \left(\frac{0.05}{0.08}\right)^4 = 0.15$$

Union-Find:
$$p_L^{\text{UF}} = \left(\frac{0.05}{0.10}\right)^4 = 0.0625$$

Union-Find is 2.4× better despite being slower.

At $p = 7\%$:
$$p_L^{\text{NN}} = \left(\frac{0.07}{0.08}\right)^4 = 0.59$$
$$p_L^{\text{UF}} = \left(\frac{0.07}{0.10}\right)^4 = 0.24$$

Still Union-Find wins.

**Conclusion**: For this comparison, Union-Find is always better for accuracy. Neural wins only if:
- Latency is critical (backlog scenarios)
- Neural threshold can be improved with more training

$$\boxed{\text{Union-Find preferred for threshold; neural for constrained latency}}$$

---

## Practice Problems

### Direct Application

**Problem 1**: Design a decoder specification for a trapped ion system with 1 ms cycle time and 50 logical qubits at distance 9.

**Problem 2**: Calculate the memory bandwidth (bits per second) required for streaming decoding at 1 μs cycle time with distance 21.

### Intermediate

**Problem 3**: A quantum computer uses two decoder tiers: fast lookup for simple syndromes, slow MWPM for complex ones. Design the routing logic and analyze the latency distribution.

**Problem 4**: Compare the total system cost (FPGA units × power × latency) for Union-Find, neural, and MWPM implementations at distances 7, 11, 15.

### Challenging

**Problem 5**: Design a fault-tolerant decoder architecture where single classical bit flips in the decoder don't cause logical errors. What is the overhead?

**Problem 6**: Propose an experiment to measure the actual threshold of a hardware decoder implementation. What metrics would you track?

---

## Computational Lab: Complete Decoder Pipeline

```python
"""
Day 833 Lab: Complete Decoder Pipeline
Capstone implementation integrating week's concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Part 1: Unified Decoder Interface
# =============================================================================

class DecoderInterface:
    """Abstract base class for decoders."""

    def __init__(self, distance: int):
        self.d = distance
        self.n_syndrome = distance ** 2 - 1

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode syndrome to correction. Must be implemented."""
        raise NotImplementedError

    def reset(self):
        """Reset decoder state."""
        pass


class UnionFindDecoder(DecoderInterface):
    """Union-Find decoder implementation."""

    def __init__(self, distance: int):
        super().__init__(distance)
        self.parent = None
        self.rank = None

    def _find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self._find(self.parent[x])
        return self.parent[x]

    def _union(self, x: int, y: int):
        rx, ry = self._find(x), self._find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        # Find defects
        defects = np.where(syndrome == 1)[0]

        if len(defects) == 0:
            return np.zeros(self.n_syndrome, dtype=int)

        # Initialize union-find
        n = len(defects)
        self.parent = list(range(n))
        self.rank = [0] * n

        # Simple greedy matching (actual UF would grow clusters)
        remaining = list(range(n))
        matches = []

        while len(remaining) >= 2:
            # Find closest pair
            best_dist = float('inf')
            best_i, best_j = 0, 1

            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    d1, d2 = defects[remaining[i]], defects[remaining[j]]
                    dist = abs(d1 - d2)  # Simplified 1D distance
                    if dist < best_dist:
                        best_dist = dist
                        best_i, best_j = i, j

            # Match pair
            idx1, idx2 = remaining[best_i], remaining[best_j]
            self._union(idx1, idx2)
            matches.append((defects[idx1], defects[idx2]))

            # Remove matched
            remaining = [r for k, r in enumerate(remaining) if k not in [best_i, best_j]]

        # Generate correction
        correction = np.zeros(self.n_syndrome, dtype=int)
        for d1, d2 in matches:
            # Flip bits between matched defects
            for i in range(min(d1, d2), max(d1, d2)):
                correction[i] = 1 - correction[i]

        return correction


class NeuralDecoder(DecoderInterface):
    """Simple neural network decoder."""

    def __init__(self, distance: int, hidden_dim: int = 32):
        super().__init__(distance)
        self.hidden_dim = hidden_dim

        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(self.n_syndrome, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 2) * 0.1
        self.b2 = np.zeros(2)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        # Forward pass
        h = np.maximum(0, syndrome @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        prob = np.exp(logits) / np.sum(np.exp(logits))

        # Return simple correction based on classification
        if prob[1] > 0.5:
            return np.ones(self.n_syndrome, dtype=int)
        return np.zeros(self.n_syndrome, dtype=int)


class MWPMDecoder(DecoderInterface):
    """Simplified MWPM decoder (greedy approximation)."""

    def __init__(self, distance: int):
        super().__init__(distance)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        defects = np.where(syndrome == 1)[0]

        if len(defects) == 0:
            return np.zeros(self.n_syndrome, dtype=int)

        # Greedy matching (not true MWPM but similar behavior)
        remaining = list(defects)
        matches = []

        while len(remaining) >= 2:
            # Sort by position and match adjacent
            remaining.sort()
            d1, d2 = remaining[0], remaining[1]
            matches.append((d1, d2))
            remaining = remaining[2:]

        # Generate correction
        correction = np.zeros(self.n_syndrome, dtype=int)
        for d1, d2 in matches:
            for i in range(d1, d2):
                correction[i] = 1 - correction[i]

        return correction


# =============================================================================
# Part 2: Streaming Decoder Wrapper
# =============================================================================

class StreamingDecoder:
    """
    Streaming wrapper for any base decoder.

    Implements sliding window with configurable parameters.
    """

    def __init__(self, base_decoder: DecoderInterface,
                 window_size: int = 20, commit_depth: int = 5):
        self.decoder = base_decoder
        self.W = window_size
        self.C = commit_depth

        # Syndrome buffer
        self.buffer = deque(maxlen=window_size)
        self.round_count = 0

        # Statistics
        self.total_decode_time = 0
        self.decode_count = 0

    def process(self, syndrome: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Process one syndrome round.

        Returns:
        --------
        correction : array or None
            Committed correction (None if buffer not full)
        decode_time : float
            Time taken to decode (seconds)
        """
        self.buffer.append(syndrome)
        self.round_count += 1

        if len(self.buffer) < self.W:
            return None, 0

        # Aggregate syndromes for decoding
        aggregated = np.zeros_like(syndrome)
        for s in self.buffer:
            aggregated = (aggregated + s) % 2

        # Decode
        t0 = perf_counter()
        correction = self.decoder.decode(aggregated)
        decode_time = perf_counter() - t0

        self.total_decode_time += decode_time
        self.decode_count += 1

        return correction, decode_time

    def get_stats(self) -> Dict:
        return {
            'rounds': self.round_count,
            'decodes': self.decode_count,
            'avg_time_us': self.total_decode_time / max(1, self.decode_count) * 1e6,
            'total_time_ms': self.total_decode_time * 1000
        }


# =============================================================================
# Part 3: Syndrome Generator
# =============================================================================

class SyndromeGenerator:
    """Generate syndromes for testing."""

    def __init__(self, distance: int, error_rate: float):
        self.d = distance
        self.p = error_rate
        self.n_syndrome = distance ** 2 - 1

    def generate(self) -> np.ndarray:
        """Generate one syndrome round."""
        return (np.random.random(self.n_syndrome) < self.p).astype(int)

    def generate_batch(self, n: int) -> List[np.ndarray]:
        """Generate batch of syndromes."""
        return [self.generate() for _ in range(n)]


# =============================================================================
# Part 4: Benchmarking Framework
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from decoder benchmark."""
    decoder_name: str
    distance: int
    error_rate: float
    n_rounds: int
    avg_latency_us: float
    max_latency_us: float
    throughput_mhz: float
    accuracy: float


def benchmark_decoder(decoder: DecoderInterface,
                      syndromes: List[np.ndarray],
                      name: str) -> BenchmarkResult:
    """Benchmark a decoder on given syndromes."""

    latencies = []
    correct = 0

    for syndrome in syndromes:
        t0 = perf_counter()
        correction = decoder.decode(syndrome)
        latencies.append(perf_counter() - t0)

        # Simple accuracy check (actual would verify logical)
        if np.sum(correction) <= np.sum(syndrome):
            correct += 1

    latencies = np.array(latencies) * 1e6  # Convert to μs

    return BenchmarkResult(
        decoder_name=name,
        distance=decoder.d,
        error_rate=0,  # Would be set by caller
        n_rounds=len(syndromes),
        avg_latency_us=np.mean(latencies),
        max_latency_us=np.max(latencies),
        throughput_mhz=1 / np.mean(latencies),  # M syndromes/sec
        accuracy=correct / len(syndromes)
    )


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 70)
    print("DECODER BENCHMARKING")
    print("=" * 70)

    distances = [5, 7, 9, 11]
    error_rates = [0.005, 0.01, 0.02]
    n_syndromes = 1000

    results = []

    for d in distances:
        for p in error_rates:
            # Generate syndromes
            gen = SyndromeGenerator(d, p)
            syndromes = gen.generate_batch(n_syndromes)

            # Create decoders
            decoders = [
                (UnionFindDecoder(d), 'Union-Find'),
                (NeuralDecoder(d), 'Neural'),
                (MWPMDecoder(d), 'MWPM'),
            ]

            for decoder, name in decoders:
                result = benchmark_decoder(decoder, syndromes, name)
                result.error_rate = p
                results.append(result)

    # Print results
    print(f"\n{'Decoder':<12} {'d':<4} {'p (%)':<6} {'Avg (μs)':<10} {'Max (μs)':<10} {'MHz':<8}")
    print("-" * 60)

    for r in results:
        print(f"{r.decoder_name:<12} {r.distance:<4} {r.error_rate*100:<6.1f} "
              f"{r.avg_latency_us:<10.2f} {r.max_latency_us:<10.2f} {r.throughput_mhz:<8.3f}")

    return results

results = run_benchmarks()

# =============================================================================
# Part 5: Visualization
# =============================================================================

def visualize_results(results: List[BenchmarkResult]):
    """Create visualization of benchmark results."""
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    # Organize data
    decoders = ['Union-Find', 'Neural', 'MWPM']
    distances = sorted(set(r.distance for r in results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Latency vs Distance
    ax = axes[0, 0]
    for decoder in decoders:
        latencies = [r.avg_latency_us for r in results
                    if r.decoder_name == decoder and r.error_rate == 0.01]
        ax.plot(distances, latencies, 'o-', label=decoder, linewidth=2, markersize=8)

    ax.axhline(y=1, color='r', linestyle='--', label='1 μs target')
    ax.set_xlabel('Code Distance', fontsize=12)
    ax.set_ylabel('Average Latency (μs)', fontsize=12)
    ax.set_title('Latency vs Distance (p = 1%)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Latency vs Error Rate
    ax = axes[0, 1]
    error_rates = sorted(set(r.error_rate for r in results))

    for decoder in decoders:
        latencies = [r.avg_latency_us for r in results
                    if r.decoder_name == decoder and r.distance == 9]
        ax.plot([p*100 for p in error_rates], latencies, 'o-',
                label=decoder, linewidth=2, markersize=8)

    ax.set_xlabel('Error Rate (%)', fontsize=12)
    ax.set_ylabel('Average Latency (μs)', fontsize=12)
    ax.set_title('Latency vs Error Rate (d = 9)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Throughput comparison
    ax = axes[1, 0]
    x = np.arange(len(distances))
    width = 0.25

    for i, decoder in enumerate(decoders):
        throughputs = [r.throughput_mhz for r in results
                      if r.decoder_name == decoder and r.error_rate == 0.01]
        ax.bar(x + i*width, throughputs, width, label=decoder, alpha=0.7)

    ax.set_xlabel('Code Distance', fontsize=12)
    ax.set_ylabel('Throughput (M syndromes/s)', fontsize=12)
    ax.set_title('Throughput Comparison', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(distances)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary radar chart (simplified as bar)
    ax = axes[1, 1]

    # Normalize metrics for d=9, p=1%
    metrics = ['Speed', 'Accuracy', 'Scalability']
    values = {
        'Union-Find': [0.9, 0.85, 0.95],  # Normalized scores
        'Neural': [0.95, 0.75, 0.80],
        'MWPM': [0.6, 0.95, 0.70],
    }

    x = np.arange(len(metrics))
    for i, (decoder, vals) in enumerate(values.items()):
        ax.bar(x + i*0.25, vals, 0.25, label=decoder, alpha=0.7)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Decoder Characteristics (Qualitative)', fontsize=14)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('week_synthesis_benchmarks.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'week_synthesis_benchmarks.png'")

visualize_results(results)

# =============================================================================
# Part 6: Complete Pipeline Demonstration
# =============================================================================

def demonstrate_complete_pipeline():
    """Demonstrate complete decoding pipeline."""
    print("\n" + "=" * 70)
    print("COMPLETE DECODER PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Configuration
    d = 9
    p = 0.01
    n_rounds = 200
    window_size = 20
    commit_depth = 5

    print(f"\nConfiguration:")
    print(f"  Distance: {d}")
    print(f"  Error rate: {p*100}%")
    print(f"  Rounds: {n_rounds}")
    print(f"  Window: {window_size}")
    print(f"  Commit depth: {commit_depth}")

    # Create components
    gen = SyndromeGenerator(d, p)
    base_decoder = UnionFindDecoder(d)
    streaming = StreamingDecoder(base_decoder, window_size, commit_depth)

    # Run pipeline
    latencies = []
    corrections_made = 0

    print(f"\nProcessing syndromes...")

    for i in range(n_rounds):
        syndrome = gen.generate()
        correction, latency = streaming.process(syndrome)

        if correction is not None:
            latencies.append(latency)
            if np.sum(correction) > 0:
                corrections_made += 1

        if (i + 1) % 50 == 0:
            stats = streaming.get_stats()
            print(f"  Round {i+1}: Avg latency = {stats['avg_time_us']:.2f} μs")

    # Final statistics
    stats = streaming.get_stats()
    latencies = np.array(latencies) * 1e6

    print(f"\nFinal Statistics:")
    print(f"  Rounds processed: {stats['rounds']}")
    print(f"  Decodes performed: {stats['decodes']}")
    print(f"  Corrections made: {corrections_made}")
    print(f"  Average latency: {stats['avg_time_us']:.2f} μs")
    print(f"  Maximum latency: {np.max(latencies):.2f} μs" if len(latencies) > 0 else "")
    print(f"  Total time: {stats['total_time_ms']:.2f} ms")

    # Latency distribution
    if len(latencies) > 0:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.hist(latencies, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=1, color='r', linestyle='--', label='1 μs target')
        plt.xlabel('Decode Latency (μs)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Latency Distribution', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(latencies, 'b-', alpha=0.7)
        plt.axhline(y=1, color='r', linestyle='--', label='1 μs target')
        plt.xlabel('Decode Index', fontsize=12)
        plt.ylabel('Latency (μs)', fontsize=12)
        plt.title('Latency Over Time', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pipeline_latency.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("\nFigure saved to 'pipeline_latency.png'")

demonstrate_complete_pipeline()

# =============================================================================
# Part 7: Research Frontier Summary
# =============================================================================

def summarize_research_frontiers():
    """Summarize open research questions."""
    print("\n" + "=" * 70)
    print("RESEARCH FRONTIERS AND OPEN PROBLEMS")
    print("=" * 70)

    frontiers = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    OPEN RESEARCH QUESTIONS                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. ALGORITHMIC                                                      │
    │     • Can we achieve ML-optimal decoding in O(n) time?              │
    │     • What's the fundamental accuracy-latency tradeoff?             │
    │     • How do we decode highly correlated errors?                    │
    │                                                                      │
    │  2. MACHINE LEARNING                                                 │
    │     • Can transformers learn optimal decoding?                      │
    │     • How to train on real (vs simulated) noise?                    │
    │     • Self-supervised approaches for adaptation?                    │
    │                                                                      │
    │  3. HARDWARE                                                         │
    │     • Optimal ASIC architecture for scalable decoding?              │
    │     • Feasibility of mK-stage processing?                           │
    │     • Fault-tolerant classical hardware for the decoder?            │
    │                                                                      │
    │  4. SYSTEMS                                                          │
    │     • End-to-end co-design of code + decoder + hardware?            │
    │     • Modular/distributed decoding for large systems?               │
    │     • Verification and certification of decoders?                   │
    │                                                                      │
    │  5. SCALABILITY                                                      │
    │     • Decoding at d=100 in real-time?                               │
    │     • Memory-efficient streaming for week-long computations?        │
    │     • Graceful degradation under decoder failures?                  │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    print(frontiers)

summarize_research_frontiers()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("WEEK 119 SYNTHESIS: COMPLETE")
print("=" * 70)
print("""
This week we mastered real-time decoding for fault-tolerant quantum computing:

KEY TAKEAWAYS:

1. FUNDAMENTAL CONSTRAINT
   t_decode < t_cycle is non-negotiable for continuous QEC.
   Backlog = exponential error accumulation = failure.

2. ALGORITHM HIERARCHY
   MWPM (10.3%) > Union-Find (9.9%) > Neural (~10%) > Simple (~8%)
   But speed matters: UF wins for superconducting systems.

3. STREAMING IS ESSENTIAL
   Finite window enables infinite computation.
   W = 2d is typically sufficient.

4. HARDWARE DETERMINES FEASIBILITY
   FPGA: 50-100 ns achievable for d ≤ 21
   ASIC: 20-50 ns for production systems
   Cryogenic: Lower latency but power-limited

5. CO-DESIGN IS THE FUTURE
   Best systems optimize code + decoder + hardware jointly.

NEXT STEPS (Week 120+):
   • Logical gate implementation on surface codes
   • Lattice surgery for multi-qubit operations
   • Resource estimation for practical algorithms
""")
```

---

## Summary

### Week 119 Key Concepts

| Day | Topic | Key Result |
|-----|-------|------------|
| 827 | Latency Constraints | $t_{\text{decode}} < t_{\text{cycle}} - t_{\text{overhead}}$ |
| 828 | MWPM Optimization | Sparse graphs, $O(n^2 \log n)$ achievable |
| 829 | Union-Find | $O(n \cdot \alpha(n))$ near-linear time |
| 830 | Neural Networks | $O(1)$ inference, ~10% threshold |
| 831 | Sliding Window | $W = O(d)$ enables streaming |
| 832 | Hardware | FPGA: 50-100 ns, ASIC: 20-50 ns |

### Master Formula

$$\boxed{p_L^{\text{eff}} = \left(\frac{p}{p_{\text{th}}^{\text{alg}} \cdot \mathcal{D}(t_d/t_c)}\right)^{(d+1)/2}}$$

where:
- $p$ = physical error rate
- $p_{\text{th}}^{\text{alg}}$ = algorithmic threshold
- $\mathcal{D}$ = backlog degradation function
- $d$ = code distance

---

## Daily Checklist

- [ ] I can synthesize all real-time decoding concepts
- [ ] I can select appropriate decoders for different scenarios
- [ ] I can design a complete decoder system from specifications
- [ ] I can identify research frontiers and open problems
- [ ] I can implement a full streaming decoder pipeline
- [ ] I understand the path from today's prototypes to production systems

---

## Week 119 Complete

Congratulations on completing Week 119: Real-Time Decoding!

You have mastered:
- The fundamental timing constraints of quantum error correction
- Optimized algorithms for minimum-weight perfect matching
- Near-linear Union-Find decoders
- Neural network approaches to syndrome classification
- Streaming architectures for continuous operation
- Hardware implementation from FPGA to ASIC

**Next**: Week 120 explores **Logical Operations on Surface Codes**, where we apply corrections to implement fault-tolerant quantum gates.

---

*"The race between quantum errors and classical decoders is the pulse of fault-tolerant quantum computing."*

---

[← Day 832: Hardware Co-Design](./Day_832_Saturday.md) | [Week 120: Logical Operations →](../Week_120_Logical_Operations/)
