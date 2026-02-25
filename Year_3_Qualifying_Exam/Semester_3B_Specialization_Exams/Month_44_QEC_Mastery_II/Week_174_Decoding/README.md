# Week 174: Decoding Algorithms for Quantum Error Correction

## Overview

**Days:** 1212-1218
**Theme:** Syndrome Processing and Error Correction Algorithms

This week focuses on the computational challenge at the heart of quantum error correction: given a syndrome measurement, determine the most likely error and apply the appropriate correction. We study classical algorithms for this task, from the well-established minimum-weight perfect matching (MWPM) decoder to cutting-edge neural network approaches.

## Learning Objectives

By the end of this week, students will be able to:

1. Formulate the decoding problem as a computational task
2. Implement and analyze the MWPM decoder for surface codes
3. Understand the union-find decoder and its complexity advantages
4. Evaluate belief propagation for quantum LDPC codes
5. Assess neural network decoders and their potential advantages
6. Compare decoder performance across threshold, complexity, and practicality

## Daily Schedule

### Day 1212 (Monday): The Decoding Problem

**Topics:**
- Syndrome measurement and interpretation
- Maximum likelihood decoding
- Degeneracy in quantum codes
- Computational complexity of optimal decoding

**Key Concepts:**
- **Syndrome:** Classical bit string from stabilizer measurements
- **Degeneracy:** Multiple errors with same syndrome and logical effect
- **Maximum likelihood:** Find most probable equivalence class

**Core Challenge:**
Optimal decoding is #P-hard in general, but efficient approximations exist for structured codes.

---

### Day 1213 (Tuesday): MWPM Decoder I - Foundations

**Topics:**
- Graph construction from syndrome
- Matching problem formulation
- Edmonds' blossom algorithm
- Complexity analysis

**Key Concepts:**
- **Syndrome graph:** Vertices = defects, edges = error chains
- **Perfect matching:** Pair all defects with minimum total weight
- **Blossom algorithm:** $$O(n^3)$$ optimal matching

**Construction for Surface Code:**
1. Place vertex at each $$-1$$ syndrome outcome
2. Edge weight = number of physical errors between defects
3. Add boundary vertices for edge defects
4. Find minimum-weight perfect matching

---

### Day 1214 (Wednesday): MWPM Decoder II - Analysis

**Topics:**
- Threshold calculation for surface codes
- PyMatching implementation
- Handling measurement errors
- Weighted edges and noise models

**Key Results:**
- **Threshold:** $$p_{\text{th}} \approx 10.3\%$$ for depolarizing noise
- **Complexity:** $$O(n^3)$$ per round, improvements to $$O(n^2)$$ possible
- **Optimality:** Near-optimal for independent noise

**Implementation Notes:**
```python
import pymatching
matching = pymatching.Matching(H)  # H = check matrix
correction = matching.decode(syndrome)
```

---

### Day 1215 (Thursday): Union-Find Decoder

**Topics:**
- Union-find data structure
- Peeling decoder concept
- Almost-linear complexity
- Threshold comparison

**Key Concepts:**
- **Union-find:** Disjoint set data structure with near-constant operations
- **Cluster growth:** Expand clusters from syndrome defects until they merge
- **Peeling:** Remove errors as clusters connect

**Performance:**
- **Complexity:** $$O(n \cdot \alpha(n))$$ where $$\alpha$$ is inverse Ackermann
- **Threshold:** $$p_{\text{th}} \approx 9.9\%$$ (slightly lower than MWPM)
- **Practical advantage:** Much faster for large codes

---

### Day 1216 (Friday): Belief Propagation

**Topics:**
- Message passing algorithms
- Factor graphs for quantum codes
- Convergence issues with short cycles
- Neural belief propagation

**Key Concepts:**
- **BP:** Iterative message passing on factor graph
- **Problem:** Quantum codes have many short cycles, causing convergence failure
- **Solutions:** Damping, neural augmentation, post-processing

**Application to QLDPC:**
Belief propagation is the natural decoder for LDPC codes, but requires modifications for quantum codes.

---

### Day 1217 (Saturday): Neural Network Decoders

**Topics:**
- Machine learning for decoding
- AlphaQubit and transformer architecture
- Training methodology
- Real hardware noise advantages

**Key Concepts:**
- **Supervised learning:** Train on syndrome-error pairs
- **Recurrent/transformer:** Handle temporal correlations
- **Transfer learning:** Train on simulator, deploy on hardware

**Recent Results (Google, 2024):**
- AlphaQubit: Transformer-based decoder
- 6% lower logical error rate than MWPM on real hardware
- Captures correlated noise patterns

---

### Day 1218 (Sunday): Decoder Comparison and Selection

**Topics:**
- Threshold vs complexity trade-offs
- Hardware implementation constraints
- Real-time decoding requirements
- Future decoder development

**Comparison Table:**

| Decoder | Threshold | Complexity | Real-time? | Correlated Noise |
|---------|-----------|------------|------------|------------------|
| MWPM | 10.3% | $$O(n^3)$$ | Challenging | Moderate |
| Union-Find | 9.9% | $$O(n\alpha(n))$$ | Yes | Limited |
| BP | Varies | $$O(n)$$ | Yes | Good for LDPC |
| Neural | ~10%+ | $$O(n)$$* | GPU needed | Excellent |

*After training; inference is efficient.

## Key Theorems and Results

### MWPM Threshold (Fowler et al.)

For the surface code with perfect syndrome measurements:
$$p_{\text{th}}^{\text{MWPM}} \approx 10.3\%$$

With noisy measurements (phenomenological model):
$$p_{\text{th}}^{\text{MWPM}} \approx 2.9\%$$

### Union-Find Complexity (Delfosse-Nickerson)

The union-find decoder achieves:
$$T(n) = O(n \cdot \alpha(n))$$

where $$\alpha(n) < 5$$ for any practical $$n$$.

### BP Convergence (Poulin-Chung)

Belief propagation fails to converge for quantum codes with:
$$\text{girth} < 2\log_q(n)$$

where girth is the shortest cycle length.

## Week 174 Files

| File | Description |
|------|-------------|
| [Review_Guide.md](Review_Guide.md) | Comprehensive review of decoding algorithms |
| [Problem_Set.md](Problem_Set.md) | 25-30 practice problems |
| [Problem_Solutions.md](Problem_Solutions.md) | Detailed solutions |
| [Oral_Practice.md](Oral_Practice.md) | Oral exam preparation questions |
| [Self_Assessment.md](Self_Assessment.md) | Self-evaluation checklist |

## Prerequisites Review

Before starting this week, ensure familiarity with:
- Graph theory: matchings, perfect matchings
- Data structures: union-find, priority queues
- Surface code structure and stabilizers
- Basic machine learning concepts

## Computational Resources

### Recommended Libraries

1. **PyMatching:** Fast MWPM decoder
   ```
   pip install pymatching
   ```

2. **Stim:** Fast stabilizer circuit simulator
   ```
   pip install stim
   ```

3. **Fusion Blossom:** Parallelized MWPM
   ```
   pip install fusion-blossom
   ```

### Sample Code Structure

```python
import stim
import pymatching

# Create surface code circuit
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5,
    rounds=10,
    after_clifford_depolarization=0.001
)

# Get detector error model
dem = circuit.detector_error_model()

# Create matcher
matching = pymatching.Matching.from_detector_error_model(dem)

# Sample and decode
sampler = circuit.compile_detector_sampler()
shots, observables = sampler.sample(1000, separate_observables=True)
predictions = matching.decode_batch(shots)
```

## Navigation

- **Previous:** [Week 173: Fault-Tolerant Operations](../Week_173_Fault_Tolerant_Operations/README.md)
- **Next:** [Week 175: QLDPC Codes](../Week_175_QLDPC_Codes/README.md)
- **Month:** [Month 44: QEC Mastery II](../README.md)
