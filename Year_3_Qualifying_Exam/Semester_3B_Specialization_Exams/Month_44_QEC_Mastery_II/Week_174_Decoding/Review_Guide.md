# Week 174: Decoding Algorithms - Comprehensive Review Guide

## Introduction

Decoding is the computational bridge between syndrome measurement and error correction. This review guide covers the theory, algorithms, and practical considerations for quantum error correction decoding, with emphasis on the surface code as the primary example.

---

## Part I: The Decoding Problem

### 1.1 Problem Formulation

Given a stabilizer code with check matrix $$H$$ and a measured syndrome $$s$$, the decoder must find an error $$E$$ such that:

1. $$HE^T = s$$ (syndrome consistency)
2. $$E$$ is in the most likely equivalence class (optimality)

**Equivalence classes:** Two errors $$E_1, E_2$$ are equivalent if $$E_1 E_2^{-1}$$ is in the stabilizer group. Equivalent errors have the same effect on the logical qubits.

### 1.2 Maximum Likelihood Decoding

The optimal decoder finds:
$$\hat{E} = \arg\max_{E: HE^T = s} P(E)$$

For independent depolarizing noise with rate $$p$$:
$$P(E) = p^{|E|}(1-p)^{n-|E|}$$

where $$|E|$$ is the weight of the error.

**Maximum likelihood simplification:**
$$\hat{E} = \arg\min_{E: HE^T = s} |E|$$

This is the minimum weight decoding problem.

### 1.3 Degeneracy

**Definition:** A code is degenerate if there exist distinct errors $$E_1 \neq E_2$$ with:
1. Same syndrome: $$HE_1^T = HE_2^T$$
2. Same logical effect: $$E_1 E_2^{-1} \in$$ stabilizer group

**Implication:** The decoder need not identify the exact error, only its equivalence class.

**Example:** For the surface code, errors differing by a stabilizer are equivalent. The decoder must only determine whether the error is closer to identity or to a logical operator.

### 1.4 Computational Complexity

**Optimal decoding:** NP-hard in general (reduction from SAT)

**For specific codes:**
- Surface code: Polynomial time via MWPM
- Toric code: Polynomial time via MWPM
- General stabilizer codes: #P-hard

The structure of the surface code (planar, local) enables efficient decoding.

---

## Part II: Minimum-Weight Perfect Matching (MWPM)

### 2.1 Algorithm Overview

MWPM decoding reduces syndrome decoding to a graph matching problem:

**Input:** Syndrome $$s$$ (locations of $$-1$$ outcomes)
**Output:** Pairing of defects that minimizes total weight

**Steps:**
1. Create graph $$G = (V, E)$$ with vertices at syndrome defects
2. Add edges between all pairs with weight = minimum physical errors connecting them
3. Find minimum-weight perfect matching
4. Interpret matching as error chains

### 2.2 Graph Construction for Surface Code

**X-type syndrome (detects Z errors):**
- Vertices: Plaquettes with $$-1$$ syndrome
- Edges: Connect pairs; weight = Manhattan distance
- Boundary: Add virtual vertices at boundaries

**Z-type syndrome (detects X errors):**
- Vertices: Stars with $$-1$$ syndrome
- Edges: Connect pairs; weight = Manhattan distance
- Boundary: Add virtual vertices at boundaries

**Critical insight:** X and Z errors can be decoded independently (for Pauli noise).

### 2.3 Edmonds' Blossom Algorithm

**Problem:** Find minimum-weight perfect matching in general graph

**Algorithm (high level):**
1. Start with empty matching
2. Find augmenting paths using BFS
3. Handle odd cycles ("blossoms") by contracting
4. Recursively solve contracted graph
5. Expand blossoms to recover matching

**Complexity:** $$O(|V|^3)$$ for dense graphs, $$O(|V||E|\log|V|)$$ with optimizations

### 2.4 Threshold Analysis

**Definition:** Threshold $$p_{\text{th}}$$ is the error rate below which logical error rate decreases with code distance.

**For surface code with MWPM:**

| Noise Model | Threshold |
|-------------|-----------|
| Perfect measurements | ~10.3% |
| Phenomenological | ~2.9% |
| Circuit-level | ~0.6-1% |

**Why high threshold?**
1. Local structure limits error propagation
2. MWPM approximates maximum likelihood well
3. Degeneracy helps: many errors are harmless

### 2.5 Handling Measurement Errors

With noisy measurements, the syndrome itself may be wrong.

**3D matching:**
- Extend graph to include time dimension
- Vertices: Syndrome changes between rounds
- Edges: Connect through space and time
- Match in 3D space-time

**Decoding window:**
- Can't wait for all measurements
- Use sliding window of $$w$$ rounds
- Trade-off: larger $$w$$ = better accuracy, more latency

### 2.6 Implementation: PyMatching

```python
import pymatching
import numpy as np

# Define parity check matrix (for repetition code example)
H = np.array([[1, 1, 0, 0],
              [0, 1, 1, 0],
              [0, 0, 1, 1]])

# Create matching object
matching = pymatching.Matching(H)

# Syndrome from error
syndrome = np.array([1, 0, 1])

# Decode
correction = matching.decode(syndrome)
print(correction)  # Output: estimated error
```

---

## Part III: Union-Find Decoder

### 3.1 Motivation

MWPM is optimal but expensive: $$O(n^3)$$ per syndrome.

For real-time decoding, we need:
- Sub-microsecond latency
- Scaling to large codes ($$d \sim 20-50$$)

Union-find achieves $$O(n \cdot \alpha(n)) \approx O(n)$$ with slight threshold reduction.

### 3.2 Union-Find Data Structure

**Operations:**
- `find(x)`: Return root of set containing $$x$$
- `union(x, y)`: Merge sets containing $$x$$ and $$y$$

**With path compression and union by rank:**
- Amortized time per operation: $$O(\alpha(n))$$
- $$\alpha(n) < 5$$ for any practical $$n$$ (inverse Ackermann function)

### 3.3 Decoder Algorithm

**Initialization:**
- Each syndrome defect starts as its own cluster
- Boundary is a special cluster

**Growth phase:**
- Grow all clusters simultaneously
- When clusters touch, merge them using union

**Termination:**
- Continue until all defects paired (even clusters) or connected to boundary

**Error chain:**
- Trace path from each defect to its pair
- Apply correction along this path

### 3.4 Complexity Analysis

**Space:** $$O(n)$$ for union-find structure

**Time:**
- Growth steps: $$O(d)$$ where $$d$$ is code distance
- Operations per step: $$O(n)$$
- Each operation: $$O(\alpha(n))$$
- Total: $$O(nd \cdot \alpha(n)) = O(n \cdot \alpha(n))$$ for fixed aspect ratio

### 3.5 Threshold Comparison

| Decoder | Threshold (code capacity) | Threshold (phenomenological) |
|---------|---------------------------|------------------------------|
| MWPM | 10.3% | 2.9% |
| Union-Find | 9.9% | 2.6% |
| Difference | -0.4% | -0.3% |

The threshold loss is acceptable given the complexity improvement.

### 3.6 Practical Advantages

1. **Parallelizable:** Cluster growth is local
2. **Streaming:** Can process measurements as they arrive
3. **FPGA-friendly:** Simple operations, predictable memory
4. **Scalable:** Sub-linear per-qubit cost

---

## Part IV: Belief Propagation

### 4.1 Overview

Belief propagation (BP) is the standard decoder for classical LDPC codes. Can it work for quantum codes?

**Factor graph representation:**
- Variable nodes: Error bits
- Factor nodes: Syndrome constraints
- Messages: Probability distributions

### 4.2 BP Algorithm

**Initialization:**
- Variable-to-factor message: Prior error probability

**Iteration:**
1. Factor-to-variable: Update based on syndrome constraint
2. Variable-to-factor: Update based on all other factors

**Termination:**
- Converge when messages stabilize
- Or after maximum iterations

**Decision:**
- Marginalize to get error probabilities
- Threshold to get error estimate

### 4.3 Problems with Quantum Codes

**Short cycles:**
- Classical LDPC: Large girth (no short cycles)
- Quantum codes: Unavoidable 4-cycles from CSS construction
- Effect: Messages correlated, BP assumptions violated

**Degeneracy:**
- BP estimates individual error probabilities
- Doesn't naturally handle equivalence classes
- May converge to wrong answer

**Stabilizer structure:**
- X and Z checks share qubits
- Creates correlations BP doesn't model

### 4.4 Improvements to BP

**Ordered statistics decoding (OSD):**
- Use BP as preprocessing
- Apply algebraic post-processing
- Achieves near-ML performance

**Neural BP:**
- Train neural network to correct BP output
- Learn to handle short cycles
- Combines BP speed with ML accuracy

**BP+OSD for QLDPC:**
- Promising for Panteleev-Kalachev codes
- Achieves reasonable thresholds
- Much faster than MWPM on sparse graphs

---

## Part V: Neural Network Decoders

### 5.1 Motivation

**Real hardware has correlated noise:**
- Crosstalk between qubits
- Leakage to non-computational states
- Drift in calibration

MWPM assumes independent noise; neural decoders can learn actual noise.

### 5.2 Architecture Approaches

**Feedforward networks:**
- Input: Syndrome vector
- Output: Error probability or correction
- Fast inference, limited by input size

**Recurrent networks (RNN/LSTM):**
- Process syndrome sequence
- Model temporal correlations
- Natural for streaming decoding

**Transformers:**
- Self-attention over syndrome history
- Capture long-range correlations
- State-of-the-art performance (AlphaQubit)

### 5.3 Training Methodology

**Supervised learning:**
1. Generate training data: (syndrome, error) pairs
2. Train network to predict error from syndrome
3. Use simulation or real hardware data

**Key considerations:**
- Need millions of training examples
- Must cover rare but important error patterns
- Transfer learning: train on simulator, fine-tune on hardware

### 5.4 AlphaQubit (Google, 2024)

**Architecture:**
- Transformer with recurrent elements
- Processes space-time syndrome volume
- Outputs logical error prediction

**Training:**
- Pre-trained on Stim simulations
- Fine-tuned on Google Sycamore data
- 100+ GPU hours training

**Results:**
- 6% lower logical error than MWPM on real hardware
- Captures non-Markovian noise
- Generalizes across code distances

### 5.5 Advantages and Challenges

**Advantages:**
- Adapts to actual noise
- Can improve with more data
- Fast inference (after training)

**Challenges:**
- Training cost
- Need retraining for hardware changes
- Harder to verify correctness
- GPU required for inference

---

## Part VI: Decoder Selection

### 6.1 Selection Criteria

1. **Threshold:** Higher is better (allows more physical noise)
2. **Complexity:** Must decode faster than syndrome measurement
3. **Latency:** End-to-end time from measurement to correction
4. **Hardware:** What compute resources are available?
5. **Noise model:** Is noise independent or correlated?

### 6.2 Comparison Matrix

| Criterion | MWPM | Union-Find | BP | Neural |
|-----------|------|------------|-----|--------|
| Threshold | Best | Good | Varies | Good+ |
| Complexity | $$O(n^3)$$ | $$O(n)$$ | $$O(n)$$ | $$O(n)$$ |
| Latency | High | Low | Low | Medium |
| Hardware | CPU/GPU | FPGA | FPGA | GPU |
| Correlated noise | Moderate | Poor | Good | Excellent |
| Maturity | High | High | Medium | Low |

### 6.3 Recommended Usage

**Research/benchmarking:** MWPM (gold standard)

**Near-term hardware:** Union-find or MWPM variants

**Large-scale FT:** Union-find with FPGA

**Correlated noise:** Neural network decoders

**QLDPC codes:** BP+OSD or specialized methods

### 6.4 Real-Time Decoding Challenge

**Requirement:** Decode within syndrome cycle time

**Surface code at 1 MHz:**
- Cycle time: 1 $$\mu$$s
- Syndrome bits: $$\sim d^2$$
- Must decode in $$< 1$$ $$\mu$$s

**Current solutions:**
- Parallel MWPM: Fusion Blossom achieves $$< 1$$ $$\mu$$s for $$d \leq 17$$
- Union-find: Sub-$$\mu$$s for $$d \leq 30$$
- FPGA implementation: Essential for real-time

---

## Part VII: Advanced Topics

### 7.1 Soft Information Decoding

**Idea:** Use analog syndrome information, not just $$\pm 1$$

**Benefits:**
- More information per measurement
- Can improve threshold
- Naturally handles weak measurements

**Implementation:** Weight edges in MWPM by measurement confidence

### 7.2 Correlated Decoding

**For non-independent noise:**
- Modify edge weights based on correlations
- Use tensor network methods for exact marginalization
- Neural networks learn correlations implicitly

### 7.3 Sliding Window Decoding

**For continuous operation:**
- Decode in windows of $$w$$ syndrome rounds
- Commit to corrections in oldest layer
- Balance: larger $$w$$ = better accuracy, more latency

### 7.4 Parallelization

**MWPM parallelization:**
- Divide syndrome into spatial regions
- Decode regions independently
- Handle boundaries with overlap or stitching

**Union-find parallelization:**
- Natural parallelism in cluster growth
- Merge step can be parallelized
- Near-linear speedup possible

---

## Summary of Key Results

### Threshold Values

$$\boxed{p_{\text{th}}^{\text{MWPM}} \approx 10.3\% \text{ (code capacity)}}$$

$$\boxed{p_{\text{th}}^{\text{UF}} \approx 9.9\% \text{ (code capacity)}}$$

$$\boxed{p_{\text{th}}^{\text{circuit}} \approx 0.5-1\% \text{ (realistic)}}$$

### Complexity

$$\boxed{T_{\text{MWPM}} = O(n^3), \quad T_{\text{UF}} = O(n \cdot \alpha(n)), \quad T_{\text{BP}} = O(n)}$$

### Real-Time Requirement

$$\boxed{T_{\text{decode}} < T_{\text{syndrome}} \approx 1 \mu s}$$

---

## References

1. Edmonds, J. (1965). "Paths, trees, and flowers." Canadian Journal of Mathematics, 17, 449-467.

2. Dennis, E., et al. (2002). "Topological quantum memory." Journal of Mathematical Physics, 43, 4452-4505.

3. Fowler, A. G., et al. (2012). "Surface codes: Towards practical large-scale quantum computation." Physical Review A, 86(3), 032324.

4. Delfosse, N., & Nickerson, N. H. (2021). "Almost-linear time decoding algorithm for topological codes." Quantum, 5, 595.

5. Higgott, O. (2022). "PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching." ACM Transactions on Quantum Computing.

6. Bausch, J., et al. (2024). "Learning to decode the surface code with a recurrent, transformer-based neural network." arXiv:2310.05900.
