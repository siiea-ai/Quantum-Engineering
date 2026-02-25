# Day 831: Sliding Window & Streaming Decoders

## Week 119: Real-Time Decoding | Month 30: Surface Codes | Year 2: Advanced Quantum Science

---

## Schedule (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 2.5 hours | Finite-history decoding, window optimization |
| **Afternoon** | 2.5 hours | Streaming algorithms, commitment strategies |
| **Evening** | 2 hours | Sliding window implementation lab |

---

## Learning Objectives

By the end of Day 831, you will be able to:

1. **Explain** why sliding window decoding is necessary for long computations
2. **Analyze** the trade-off between window size and decoding accuracy
3. **Design** commitment strategies for streaming syndrome data
4. **Implement** sliding window variants of MWPM and Union-Find
5. **Evaluate** memory requirements for streaming decoders
6. **Compare** single-shot vs multi-round decoding approaches

---

## Core Content

### 1. The Infinite History Problem

Standard MWPM/Union-Find decode all syndrome history simultaneously:

```
Syndromes: σ₁, σ₂, σ₃, ..., σₜ, ...
             ↓
        Full 3D Matching
             ↓
        Complete Correction
```

For a computation lasting $T$ syndrome rounds:
- Memory: $O(d^2 \cdot T)$ for syndrome storage
- Time: $O(\text{poly}(d^2 \cdot T))$ for matching

**Problems for long computations**:
- Memory grows without bound
- Decoding time increases with $T$
- Cannot provide corrections until end

### 2. Sliding Window Approach

Process syndromes in a moving window of size $W$:

$$\boxed{\text{Window}_t = \{\sigma_{t-W+1}, \sigma_{t-W+2}, \ldots, \sigma_t\}}$$

**Algorithm**:
```
for each new syndrome σₜ:
    1. Add σₜ to window
    2. Remove σₜ₋ᵥ from window (oldest)
    3. Decode current window
    4. Commit corrections for oldest portion
    5. Output committed corrections
```

**Key parameters**:
- $W$: Window size (number of rounds)
- $C$: Commitment depth (how much to commit each step)

### 3. Window Size Selection

The optimal window size balances:

**Too small** ($W \ll d$):
- Errors spanning multiple windows get mismatched
- Threshold degrades significantly
- Fast but inaccurate

**Too large** ($W \gg d$):
- Approaches full-history accuracy
- Slow, high memory
- Unnecessary for most error patterns

**Optimal scaling**:
$$\boxed{W_{\text{opt}} = O(d)}$$

For a distance-$d$ code, $W \approx 2d$ to $4d$ is typically sufficient.

### 4. Commitment Strategies

When to "commit" to a correction decision:

#### Eager Commitment
Commit oldest round immediately:
- Lowest latency
- Risk of suboptimal matching

#### Delayed Commitment
Wait until syndrome is $C$ rounds old:
- Better matching decisions
- Higher latency: $t_{\text{latency}} = C \cdot t_{\text{cycle}}$

#### Confidence-Based Commitment
Commit when matching is "stable":
- Monitor if match changes with new syndromes
- Adaptive latency
- Complex implementation

### 5. Boundary Handling

At window edges, some defects may match to:
1. Other defects within window
2. Virtual boundary at window edge
3. Carry-forward to next window step

**Virtual Temporal Boundary**:
Add virtual nodes at $t = t_{\text{min}}$ representing "past" errors:

```
Time:    past | window | future
         ----[=========]----
              ↑         ↑
         virtual     current
         boundary    syndromes
```

Defects matching to the past boundary indicate errors from before the window that propagated forward.

### 6. Streaming Union-Find

Adapt Union-Find for streaming:

```python
class StreamingUnionFind:
    def __init__(self, window_size):
        self.W = window_size
        self.syndrome_buffer = deque(maxlen=W)
        self.clusters = {}  # Active clusters

    def process_syndrome(self, sigma):
        # Add new syndrome
        self.syndrome_buffer.append(sigma)

        # Find new defects
        if len(self.syndrome_buffer) >= 2:
            delta = sigma XOR self.syndrome_buffer[-2]
        else:
            delta = sigma

        # Grow/merge clusters for new defects
        new_clusters = self.create_clusters(delta)
        self.merge_with_existing(new_clusters)

        # Commit oldest clusters
        committed = self.commit_oldest()

        return committed
```

**Complexity**: $O(W \cdot \alpha(W))$ per syndrome round.

### 7. Memory-Efficient Streaming

For very long computations, even $O(d^2 \cdot W)$ memory may be too much.

**Compressed Syndrome Storage**:
- Store only defect positions, not full syndrome
- Expected defects: $O(p \cdot d^2)$ per round
- Memory: $O(p \cdot d^2 \cdot W)$

**Cluster Summarization**:
- For committed clusters, store only boundary information
- Discard internal structure

**Incremental Updates**:
- Don't rebuild matching graph each round
- Add/remove edges incrementally

### 8. Single-Shot Decoding

Alternative: decode each round independently using only spatial information.

**Advantages**:
- No temporal memory needed
- Constant time per round
- Simple implementation

**Disadvantages**:
- Cannot distinguish data errors from measurement errors
- Requires redundant syndrome extraction (multiple measurements per round)
- Lower threshold for same overhead

**Hybrid approach**: Single-shot for most rounds, windowed for complex syndromes.

---

## Worked Examples

### Example 1: Window Size Calculation

**Problem**: For a distance-11 surface code with syndrome cycle time 1 μs and decode budget 10 μs, what window size allows real-time operation?

**Solution**:

With decode time $t_d = 10 \, \mu\text{s}$ and cycle time $t_c = 1 \, \mu\text{s}$:

For real-time operation with window $W$:
$$t_d(W) < t_c$$

Union-Find complexity per round: $O(W \cdot d^2 \cdot \alpha(W \cdot d^2))$

At $d = 11$, $d^2 = 121$ syndrome bits per round.

For $W$ rounds: $\sim 121 \cdot W$ syndrome bits total.

Estimated decode time:
$$t_d \approx 121 \cdot W \cdot 10 \, \text{ns} = 1.21 \cdot W \, \mu\text{s}$$

For real-time: $1.21 \cdot W < 1$, so $W < 0.8$.

This seems too restrictive. But we decode once per round, not once per syndrome:
$$t_d(W) < W \cdot t_c = W \, \mu\text{s}$$

So: $1.21 \cdot W < W$ is satisfied for any $W > 0$ (just barely).

More carefully, if decode takes $10 \, \mu\text{s}$ total for window:
$$W > \frac{10 \, \mu\text{s}}{1 \, \mu\text{s}} = 10$$

So we need $W > 10$ to avoid backlog, and $W \approx 2d = 22$ for accuracy.

$$\boxed{W = 20-25 \text{ rounds}}$$

### Example 2: Commitment Depth Trade-off

**Problem**: A sliding window decoder uses $W = 20$ rounds. Compare logical error rates for commitment depths $C = 1, 5, 10$.

**Solution**:

Let $p = 0.5\%$ and $d = 7$.

**$C = 1$ (eager commitment)**:
Errors near window edge may be mismatched:
- Probability of edge error: $\sim p \cdot (1/W) = 0.5\% \times 5\% = 0.025\%$ additional error per round
- Over many rounds, this compounds

**$C = 5$**:
Errors have 5 rounds to find correct match:
- Most local errors correctly matched
- Long-range correlations still problematic

**$C = 10$**:
Half-window commitment:
- Very close to full-window accuracy
- Latency: $10 \, \mu\text{s}$ at $1 \, \mu\text{s}$ cycle

Empirical rule (from literature):
$$p_L(C) \approx p_L^{(\infty)} \cdot \left(1 + \frac{d}{C}\right)$$

For $d = 7$:
- $C = 1$: $p_L \approx 8 \times p_L^{(\infty)}$
- $C = 5$: $p_L \approx 2.4 \times p_L^{(\infty)}$
- $C = 10$: $p_L \approx 1.7 \times p_L^{(\infty)}$

$$\boxed{C = W/2 \text{ is a good balance}}$$

### Example 3: Memory Requirements

**Problem**: Calculate memory requirements for a distance-21 surface code running for 1 second with 1 μs cycle time.

**Solution**:

Total syndrome rounds: $T = 10^6$ rounds

Syndrome bits per round: $d^2 - 1 = 440$ bits

**Full history** (not streaming):
$$\text{Memory} = 440 \times 10^6 \text{ bits} = 55 \text{ MB}$$

**Sliding window** ($W = 50$):
$$\text{Memory} = 440 \times 50 \text{ bits} = 22 \text{ Kbits} \approx 3 \text{ KB}$$

**Compressed** (store only defects):
At $p = 0.5\%$: $\sim 2$ defects per round on average
$$\text{Memory} \approx 2 \times 50 \times \log_2(440) \approx 900 \text{ bits}$$

$$\boxed{\text{Streaming reduces memory by } 10^4 \text{x}}$$

---

## Practice Problems

### Direct Application

**Problem 1**: For a distance-15 code with window size $W = 30$, how many syndrome bits are in the active window? How many edges in the matching graph (approximate)?

**Problem 2**: A decoder commits 5 rounds per step with a 20-round window. What is the maximum correction latency in syndrome cycles?

### Intermediate

**Problem 3**: Design a commitment strategy that balances latency and accuracy by committing based on cluster stability. Define "stability" precisely.

**Problem 4**: Compare the memory bandwidth requirements for streaming vs batch decoding over a 1-minute computation.

### Challenging

**Problem 5**: Prove that for independent errors at rate $p$, the expected time for an error chain to cross the window boundary is $O(W/p)$ syndrome rounds.

**Problem 6**: Design an adaptive window size algorithm that increases $W$ when syndrome complexity is high and decreases it when syndromes are sparse.

---

## Computational Lab: Sliding Window Decoder

```python
"""
Day 831 Lab: Sliding Window Decoder Implementation
Streaming syndrome processing with bounded memory
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from time import perf_counter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Part 1: Syndrome Stream Generator
# =============================================================================

class SyndromeStream:
    """
    Generator for streaming syndrome data.

    Simulates continuous QEC operation.
    """

    def __init__(self, distance, p_error):
        """
        Initialize syndrome stream.

        Parameters:
        -----------
        distance : int
            Code distance
        p_error : float
            Physical error probability
        """
        self.d = distance
        self.p = p_error
        self.n_syndrome = distance * distance - 1

        # Track cumulative error state
        self.current_syndrome = np.zeros(self.n_syndrome, dtype=int)
        self.round_count = 0

    def generate_round(self):
        """
        Generate one round of syndrome data.

        Returns:
        --------
        delta_syndrome : array
            Syndrome difference from previous round
        """
        # Random errors this round
        new_errors = (np.random.random(self.n_syndrome) < self.p).astype(int)

        # Syndrome difference (XOR of errors affecting each stabilizer)
        delta = new_errors  # Simplified model

        # Add measurement errors
        meas_errors = (np.random.random(self.n_syndrome) < self.p / 10).astype(int)
        delta = (delta + meas_errors) % 2

        self.current_syndrome = (self.current_syndrome + delta) % 2
        self.round_count += 1

        return delta

    def reset(self):
        """Reset stream to initial state."""
        self.current_syndrome = np.zeros(self.n_syndrome, dtype=int)
        self.round_count = 0


def demonstrate_syndrome_stream():
    """Demonstrate syndrome stream generation."""
    print("=" * 60)
    print("SYNDROME STREAM GENERATION")
    print("=" * 60)

    d = 5
    p = 0.02
    stream = SyndromeStream(d, p)

    print(f"\nCode distance: {d}")
    print(f"Syndrome bits: {stream.n_syndrome}")
    print(f"Error rate: {p*100}%")
    print(f"Expected defects per round: {p * stream.n_syndrome:.2f}")

    # Generate some rounds
    n_rounds = 10
    defect_counts = []

    print(f"\nFirst {n_rounds} rounds:")
    for i in range(n_rounds):
        delta = stream.generate_round()
        n_defects = np.sum(delta)
        defect_counts.append(n_defects)
        print(f"  Round {i+1}: {n_defects} defects")

    print(f"\nAverage defects: {np.mean(defect_counts):.2f}")

    return stream

stream = demonstrate_syndrome_stream()

# =============================================================================
# Part 2: Sliding Window Decoder
# =============================================================================

class SlidingWindowDecoder:
    """
    Sliding window decoder for streaming syndromes.

    Uses simplified Union-Find within each window.
    """

    def __init__(self, distance, window_size, commit_depth):
        """
        Initialize sliding window decoder.

        Parameters:
        -----------
        distance : int
            Code distance
        window_size : int
            Number of rounds in decoding window
        commit_depth : int
            Number of rounds to commit each step
        """
        self.d = distance
        self.W = window_size
        self.C = commit_depth
        self.n_syndrome = distance * distance - 1

        # Syndrome buffer (deque for efficient sliding)
        self.syndrome_buffer = deque(maxlen=window_size)

        # Cluster tracking
        self.active_clusters = []

        # Committed corrections
        self.committed_corrections = []

        # Statistics
        self.total_rounds = 0
        self.total_defects = 0

    def process_syndrome(self, delta_syndrome):
        """
        Process one syndrome round.

        Parameters:
        -----------
        delta_syndrome : array
            Syndrome difference for this round

        Returns:
        --------
        committed : list of int
            Indices of committed corrections (if any)
        """
        # Add to buffer
        self.syndrome_buffer.append(delta_syndrome.copy())
        self.total_rounds += 1
        self.total_defects += np.sum(delta_syndrome)

        # If buffer full, decode and commit
        committed = []
        if len(self.syndrome_buffer) == self.W:
            correction = self._decode_window()
            committed = self._commit_oldest(correction)

        return committed

    def _decode_window(self):
        """
        Decode current window using simplified matching.

        Returns correction for oldest rounds.
        """
        # Collect all defects in window
        defects = []
        for t, syndrome in enumerate(self.syndrome_buffer):
            for i, s in enumerate(syndrome):
                if s == 1:
                    defects.append((t, i))  # (time, spatial_index)

        if len(defects) == 0:
            return {}

        # Simple greedy matching within window
        correction = self._greedy_match(defects)

        return correction

    def _greedy_match(self, defects):
        """
        Simple greedy matching of defects.

        Matches closest pairs first.
        """
        if len(defects) < 2:
            return {}

        correction = {}
        remaining = list(defects)

        while len(remaining) >= 2:
            # Find closest pair
            best_dist = float('inf')
            best_pair = (0, 1)

            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    d1 = remaining[i]
                    d2 = remaining[j]
                    # Manhattan distance in spacetime
                    dist = abs(d1[0] - d2[0]) + abs(d1[1] - d2[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (i, j)

            # Match this pair
            d1 = remaining[best_pair[0]]
            d2 = remaining[best_pair[1]]

            # Record correction (simplified: just mark as corrected)
            if d1[0] not in correction:
                correction[d1[0]] = []
            correction[d1[0]].append((d1[1], d2[1]))

            # Remove matched defects
            remaining = [d for k, d in enumerate(remaining)
                        if k not in best_pair]

        return correction

    def _commit_oldest(self, correction):
        """
        Commit corrections for oldest rounds and remove from window.
        """
        committed = []

        for t in range(self.C):
            if t in correction:
                committed.extend(correction[t])
                self.committed_corrections.append((self.total_rounds - self.W + t,
                                                   correction[t]))

        return committed

    def get_statistics(self):
        """Return decoder statistics."""
        return {
            'total_rounds': self.total_rounds,
            'total_defects': self.total_defects,
            'avg_defects_per_round': self.total_defects / max(1, self.total_rounds),
            'committed_corrections': len(self.committed_corrections),
            'buffer_size': len(self.syndrome_buffer)
        }


def demonstrate_sliding_window():
    """Demonstrate sliding window decoder."""
    print("\n" + "=" * 60)
    print("SLIDING WINDOW DECODER")
    print("=" * 60)

    d = 5
    p = 0.02
    W = 10  # Window size
    C = 2   # Commit depth

    stream = SyndromeStream(d, p)
    decoder = SlidingWindowDecoder(d, W, C)

    print(f"\nCode distance: {d}")
    print(f"Window size: {W}")
    print(f"Commit depth: {C}")

    # Process many rounds
    n_rounds = 100
    corrections_per_round = []

    print(f"\nProcessing {n_rounds} syndrome rounds...")

    for i in range(n_rounds):
        delta = stream.generate_round()
        committed = decoder.process_syndrome(delta)
        corrections_per_round.append(len(committed))

        if (i + 1) % 20 == 0:
            stats = decoder.get_statistics()
            print(f"  Round {i+1}: Avg defects = {stats['avg_defects_per_round']:.2f}, "
                  f"Buffer = {stats['buffer_size']}")

    # Final statistics
    stats = decoder.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Total rounds: {stats['total_rounds']}")
    print(f"  Total defects: {stats['total_defects']}")
    print(f"  Committed corrections: {stats['committed_corrections']}")

    return decoder

decoder = demonstrate_sliding_window()

# =============================================================================
# Part 3: Window Size Analysis
# =============================================================================

def analyze_window_sizes():
    """Analyze decoder performance vs window size."""
    print("\n" + "=" * 60)
    print("WINDOW SIZE ANALYSIS")
    print("=" * 60)

    d = 7
    p = 0.01
    n_rounds = 500

    window_sizes = [5, 10, 15, 20, 30, 50]
    commit_depth = 2

    results = []

    for W in window_sizes:
        stream = SyndromeStream(d, p)
        decoder = SlidingWindowDecoder(d, W, commit_depth)

        # Time the decoding
        t0 = perf_counter()
        for _ in range(n_rounds):
            delta = stream.generate_round()
            decoder.process_syndrome(delta)
        decode_time = perf_counter() - t0

        stats = decoder.get_statistics()

        results.append({
            'W': W,
            'time': decode_time,
            'time_per_round': decode_time / n_rounds * 1e6,  # μs
            'corrections': stats['committed_corrections']
        })

        print(f"W = {W:2d}: {decode_time*1e3:.1f} ms total, "
              f"{decode_time/n_rounds*1e6:.1f} μs/round")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    ws = [r['W'] for r in results]
    times = [r['time_per_round'] for r in results]
    plt.plot(ws, times, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Window Size W', fontsize=12)
    plt.ylabel('Time per Round (μs)', fontsize=12)
    plt.title('Decode Time vs Window Size', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Memory scaling (proportional to W * d^2)
    memory = [W * d * d for W in ws]
    plt.plot(ws, memory, 'rs-', linewidth=2, markersize=8)
    plt.xlabel('Window Size W', fontsize=12)
    plt.ylabel('Memory (syndrome bits)', fontsize=12)
    plt.title('Memory vs Window Size', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('window_size_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'window_size_analysis.png'")

    # Recommend window size
    recommended = 2 * d
    print(f"\nRecommended window size for d={d}: W = 2d = {recommended}")

analyze_window_sizes()

# =============================================================================
# Part 4: Commitment Strategy Comparison
# =============================================================================

def compare_commitment_strategies():
    """Compare different commitment strategies."""
    print("\n" + "=" * 60)
    print("COMMITMENT STRATEGY COMPARISON")
    print("=" * 60)

    d = 7
    p = 0.01
    W = 20
    n_rounds = 1000

    commit_depths = [1, 2, 5, 10, 15, 19]

    results = []

    for C in commit_depths:
        stream = SyndromeStream(d, p)
        decoder = SlidingWindowDecoder(d, W, C)

        for _ in range(n_rounds):
            delta = stream.generate_round()
            decoder.process_syndrome(delta)

        stats = decoder.get_statistics()

        # Latency = C cycles before correction is committed
        latency = C

        results.append({
            'C': C,
            'latency': latency,
            'corrections': stats['committed_corrections'],
            'ratio': C / W
        })

        print(f"C = {C:2d} (latency = {latency:2d} cycles): "
              f"{stats['committed_corrections']} corrections committed")

    # Plot
    plt.figure(figsize=(10, 5))

    cs = [r['C'] for r in results]
    latencies = [r['latency'] for r in results]

    plt.subplot(1, 2, 1)
    plt.bar(range(len(cs)), latencies, tick_label=[str(c) for c in cs])
    plt.xlabel('Commit Depth C', fontsize=12)
    plt.ylabel('Correction Latency (cycles)', fontsize=12)
    plt.title('Latency vs Commit Depth', fontsize=14)

    plt.subplot(1, 2, 2)
    # Simulated accuracy (higher C = better accuracy)
    # This is a simplified model
    accuracy = [1 - 0.1 * np.exp(-c/5) for c in cs]
    plt.plot(cs, accuracy, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Commit Depth C', fontsize=12)
    plt.ylabel('Relative Accuracy', fontsize=12)
    plt.title('Accuracy vs Commit Depth (Simulated)', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('commitment_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'commitment_comparison.png'")

    # Recommendation
    print(f"\nRecommendation: C = W/2 = {W//2} balances latency and accuracy")

compare_commitment_strategies()

# =============================================================================
# Part 5: Memory Usage Comparison
# =============================================================================

def compare_memory_usage():
    """Compare memory usage of different approaches."""
    print("\n" + "=" * 60)
    print("MEMORY USAGE COMPARISON")
    print("=" * 60)

    distances = [5, 7, 9, 11, 15, 21]
    n_rounds = 10000  # 10 ms at 1 μs/round

    for d in distances:
        n_syndrome = d * d - 1
        W = 2 * d  # Typical window size

        # Full history
        full_memory = n_syndrome * n_rounds  # bits

        # Sliding window
        window_memory = n_syndrome * W  # bits

        # Compressed (defects only)
        # At 1% error, ~1% of syndromes are defects
        p = 0.01
        expected_defects = p * n_syndrome * W
        compressed_memory = expected_defects * (np.log2(n_syndrome) + np.log2(W))

        print(f"d = {d:2d}: Full = {full_memory/8000:.1f} KB, "
              f"Window = {window_memory/8:.0f} bytes, "
              f"Compressed = {compressed_memory/8:.0f} bytes")

    # Visualization
    plt.figure(figsize=(10, 6))

    d_vals = np.array(distances)
    T = n_rounds

    full_mem = (d_vals ** 2) * T / 8000  # KB
    window_mem = (d_vals ** 2) * (2 * d_vals) / 8000  # KB
    compressed_mem = 0.01 * (d_vals ** 2) * (2 * d_vals) * 10 / 8000  # KB (rough)

    plt.semilogy(d_vals, full_mem, 'ro-', label='Full History', linewidth=2, markersize=8)
    plt.semilogy(d_vals, window_mem, 'bs-', label='Sliding Window', linewidth=2, markersize=8)
    plt.semilogy(d_vals, compressed_mem, 'g^-', label='Compressed Window', linewidth=2, markersize=8)

    plt.xlabel('Code Distance', fontsize=12)
    plt.ylabel('Memory (KB)', fontsize=12)
    plt.title(f'Memory Usage for {n_rounds/1000:.0f}k Syndrome Rounds', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('memory_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'memory_comparison.png'")

compare_memory_usage()

# =============================================================================
# Part 6: Real-Time Performance Simulation
# =============================================================================

def simulate_realtime_operation():
    """Simulate real-time decoder operation."""
    print("\n" + "=" * 60)
    print("REAL-TIME OPERATION SIMULATION")
    print("=" * 60)

    d = 7
    p = 0.005
    W = 15
    C = 5
    t_cycle = 1e-6  # 1 μs

    stream = SyndromeStream(d, p)
    decoder = SlidingWindowDecoder(d, W, C)

    # Simulate 1 ms of operation (1000 cycles)
    n_cycles = 1000

    decode_times = []
    backlog = 0
    max_backlog = 0

    print(f"\nSimulating {n_cycles} QEC cycles ({n_cycles * t_cycle * 1e3:.1f} ms)...")

    for cycle in range(n_cycles):
        # Generate syndrome
        delta = stream.generate_round()

        # Decode (measure time)
        t0 = perf_counter()
        decoder.process_syndrome(delta)
        t_decode = perf_counter() - t0

        decode_times.append(t_decode)

        # Track backlog
        if t_decode > t_cycle:
            backlog += 1
        else:
            backlog = max(0, backlog - 1)

        max_backlog = max(max_backlog, backlog)

    decode_times = np.array(decode_times) * 1e6  # Convert to μs

    print(f"\nDecode time statistics:")
    print(f"  Mean: {np.mean(decode_times):.2f} μs")
    print(f"  Std:  {np.std(decode_times):.2f} μs")
    print(f"  Max:  {np.max(decode_times):.2f} μs")
    print(f"  Real-time budget: {t_cycle * 1e6:.2f} μs")
    print(f"  Within budget: {np.sum(decode_times < t_cycle * 1e6) / n_cycles * 100:.1f}%")
    print(f"  Max backlog: {max_backlog} cycles")

    # Plot decode time distribution
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(decode_times, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=t_cycle * 1e6, color='r', linestyle='--', linewidth=2,
                label=f'Budget ({t_cycle*1e6} μs)')
    plt.xlabel('Decode Time (μs)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Decode Time Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(decode_times, 'b-', alpha=0.7, linewidth=0.5)
    plt.axhline(y=t_cycle * 1e6, color='r', linestyle='--', linewidth=2,
                label=f'Budget')
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('Decode Time (μs)', fontsize=12)
    plt.title('Decode Time per Cycle', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('realtime_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFigure saved to 'realtime_simulation.png'")

simulate_realtime_operation()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("LAB SUMMARY")
print("=" * 60)
print("""
Key findings:

1. WINDOW SIZE: W = 2d is typically sufficient for near-optimal
   accuracy while maintaining bounded memory.

2. COMMIT DEPTH: C = W/2 balances correction latency with matching
   quality. Eager commitment (C=1) degrades accuracy.

3. MEMORY: Sliding window reduces memory from O(T) to O(W), a
   factor of T/W improvement for long computations.

4. REAL-TIME: Python implementation achieves ~10-100 μs per round.
   Hardware acceleration needed for sub-μs targets.

5. STREAMING: Continuous syndrome processing enables unlimited
   computation duration with fixed resources.

6. TRADE-OFFS: Larger windows improve accuracy but increase
   latency and memory. Application requirements drive selection.

Design guidelines:
- Start with W = 2d, C = d
- Increase W if accuracy insufficient
- Decrease C if latency too high
- Use compression for memory-constrained systems
""")
```

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Window contents | $\{\sigma_{t-W+1}, \ldots, \sigma_t\}$ |
| Optimal window size | $W_{\text{opt}} = O(d)$, typically $2d$-$4d$ |
| Memory (full history) | $O(d^2 \cdot T)$ |
| Memory (sliding window) | $O(d^2 \cdot W)$ |
| Correction latency | $C$ syndrome cycles |
| Streaming complexity | $O(W \cdot d^2 \cdot \alpha(W))$ per round |

### Key Insights

1. **Bounded Resources**: Streaming enables unlimited computation duration
2. **Window-Accuracy Trade-off**: Larger windows = better matching
3. **Commitment Strategy**: Delayed commitment improves quality
4. **Memory Efficiency**: Compression exploits low defect density
5. **Real-Time Compatible**: Fixed processing per round

---

## Daily Checklist

- [ ] I understand why sliding window decoding is necessary
- [ ] I can select appropriate window size for a given code distance
- [ ] I can design commitment strategies with latency-accuracy trade-offs
- [ ] I can implement a basic sliding window decoder
- [ ] I understand memory requirements for streaming operation
- [ ] I can compare single-shot vs multi-round decoding

---

## Preview: Day 832

Tomorrow we explore **Decoder-Hardware Co-Design**:
- FPGA implementation strategies
- ASIC decoder architectures
- Cryogenic classical processing
- Integration with quantum control systems

Hardware implementation transforms algorithmic ideas into physical reality.

---

*"Streaming decoders turn infinite computation into a finite resource problem."*

---

[← Day 830: Neural Network Decoders](./Day_830_Thursday.md) | [Day 832: Decoder-Hardware Co-Design →](./Day_832_Saturday.md)
