# Week 174: Decoding Algorithms - Oral Examination Practice

## Introduction

This document provides oral examination practice for decoding algorithms in quantum error correction. Questions simulate a PhD qualifying exam format, testing both technical depth and communication skills.

---

## Part 1: Fundamental Concepts

### Question 1.1: The Decoding Problem

**Examiner:** "Explain the decoding problem in quantum error correction. What makes it different from classical decoding?"

**Follow-up probes:**
- What is degeneracy and why does it matter?
- How does the decoder interact with syndrome measurements?
- What is the computational complexity of optimal decoding?

**Key points for strong answer:**
1. Syndrome measurement gives partial information about error
2. Must find error (or equivalence class) consistent with syndrome
3. Quantum degeneracy: multiple errors with same syndrome and effect
4. Decoder need only identify equivalence class, not exact error
5. Optimal decoding is NP-hard in general, but structured codes allow efficient approximations

**Common pitfalls:**
- Forgetting degeneracy
- Not distinguishing syndrome from error
- Claiming all decoding is hard (surface code is tractable)

---

### Question 1.2: Maximum Likelihood vs Minimum Weight

**Examiner:** "When is minimum weight decoding equivalent to maximum likelihood decoding?"

**Follow-up probes:**
- Write down the probability distribution that justifies this
- What happens for biased noise?
- How should the decoder be modified for non-uniform noise?

**Key points for strong answer:**
1. For independent depolarizing noise: $$P(E) \propto p^{|E|}$$
2. If $$p < 0.5$$: $$\arg\max P(E) = \arg\min |E|$$
3. For biased noise: need weighted minimum weight
4. Edge weights should be $$-\log P(\text{error})$$
5. More generally: maximum likelihood over equivalence classes

---

## Part 2: MWPM Decoder

### Question 2.1: MWPM Overview

**Examiner:** "Describe the minimum-weight perfect matching decoder for the surface code."

**Follow-up probes:**
- How is the matching graph constructed?
- What algorithm computes the matching?
- What is the threshold for the surface code?

**Key points for strong answer:**
1. Construction: vertices at syndrome defects, edges weighted by distance
2. Include boundary vertices for open boundaries
3. Edmonds' blossom algorithm: $$O(n^3)$$ complexity
4. Threshold ~10.3% for code capacity noise
5. X and Z syndromes decoded independently (for Pauli noise)

**Common pitfalls:**
- Forgetting boundary vertices
- Confusing code distance with matching graph structure
- Not mentioning that X/Z decouple

---

### Question 2.2: Threshold Analysis

**Examiner:** "How is the threshold of a decoder determined, and what affects it?"

**Follow-up probes:**
- Define threshold precisely
- How do different noise models change the threshold?
- How does decoder choice affect threshold?

**Key points for strong answer:**
1. Threshold $$p_{\text{th}}$$: error rate below which logical error decreases with code distance
2. Numerical: Monte Carlo simulation, plot $$p_L$$ vs $$d$$
3. Noise models: code capacity > phenomenological > circuit-level
4. Better decoders have higher thresholds (closer to optimal)
5. MWPM is near-optimal for independent noise

---

### Question 2.3: Handling Measurement Errors

**Examiner:** "How does the MWPM decoder handle noisy syndrome measurements?"

**Follow-up probes:**
- Describe the 3D matching graph
- What is the sliding window approach?
- How does this affect threshold?

**Key points for strong answer:**
1. Extend to space-time: vertices at syndrome changes
2. Edges connect through space (same time) and time (same position)
3. Matching in 3D hypergraph
4. Sliding window: decode in chunks to bound latency
5. Threshold drops: ~10% → ~3% for phenomenological

---

## Part 3: Union-Find Decoder

### Question 3.1: Union-Find Basics

**Examiner:** "Explain the union-find decoder. Why is it faster than MWPM?"

**Follow-up probes:**
- Describe the data structure
- What is the complexity?
- What is the threshold trade-off?

**Key points for strong answer:**
1. Cluster growth from syndrome defects
2. Union-find tracks connected components
3. Merge clusters on contact, pair when even
4. Complexity: $$O(n \cdot \alpha(n)) \approx O(n)$$
5. Threshold: ~9.9% vs 10.3% for MWPM
6. Trade-off: speed vs accuracy

---

### Question 3.2: Real-Time Decoding

**Examiner:** "What are the requirements for real-time decoding and how does union-find meet them?"

**Follow-up probes:**
- What is the latency requirement?
- How can decoding be parallelized?
- What hardware is needed?

**Key points for strong answer:**
1. Must decode within syndrome measurement cycle (~1 μs)
2. Union-find: local operations, natural parallelism
3. FPGA implementation achieves sub-μs latency
4. Cluster growth is embarrassingly parallel
5. Memory access patterns are predictable

---

## Part 4: Belief Propagation

### Question 4.1: BP for Quantum Codes

**Examiner:** "Why does standard belief propagation fail for quantum codes?"

**Follow-up probes:**
- What is belief propagation?
- What property of quantum codes causes problems?
- How can BP be improved?

**Key points for strong answer:**
1. BP: iterative message passing on factor graph
2. Assumes local tree-like structure (no short cycles)
3. Quantum codes have unavoidable 4-cycles from CSS
4. Messages become correlated, violating independence assumption
5. Solutions: OSD post-processing, neural augmentation, modified BP

---

### Question 4.2: When to Use BP

**Examiner:** "For what codes is belief propagation a good choice?"

**Follow-up probes:**
- Compare BP to MWPM for different code families
- What is OSD?
- How does BP scale?

**Key points for strong answer:**
1. Good for: QLDPC codes with larger girth
2. BP+OSD competitive for Panteleev-Kalachev codes
3. BP complexity: $$O(n)$$ per iteration
4. Scales better than MWPM for large sparse codes
5. Not optimal for surface codes (MWPM better)

---

## Part 5: Neural Network Decoders

### Question 5.1: Neural Decoder Advantages

**Examiner:** "What advantages do neural network decoders offer over classical algorithms?"

**Follow-up probes:**
- What types of noise can neural decoders capture?
- How are they trained?
- What are the drawbacks?

**Key points for strong answer:**
1. Learn actual noise from data (correlated, non-Markovian)
2. Capture leakage, crosstalk, drift
3. Training: supervised on (syndrome, error) pairs
4. Can use real hardware data for fine-tuning
5. Drawbacks: training cost, GPU needed, less interpretable

---

### Question 5.2: AlphaQubit

**Examiner:** "Describe the AlphaQubit decoder and its results."

**Follow-up probes:**
- What architecture is used?
- How was it trained?
- What improvement did it achieve?

**Key points for strong answer:**
1. Transformer-based with recurrent elements
2. Pre-trained on Stim simulations
3. Fine-tuned on Google Sycamore data
4. 6% lower logical error than MWPM on real hardware
5. Captures non-Markovian noise patterns
6. Generalizes across code distances

---

## Part 6: Decoder Comparison

### Question 6.1: Choosing a Decoder

**Examiner:** "You're designing a fault-tolerant quantum computer. How do you choose a decoder?"

**Follow-up probes:**
- What criteria matter most?
- When would you use each decoder type?
- How might decoders evolve?

**Key points for strong answer:**
1. Criteria: threshold, complexity, latency, hardware requirements
2. Research/benchmarking: MWPM
3. Near-term hardware: union-find (real-time)
4. Correlated noise: neural networks
5. QLDPC codes: BP+OSD
6. Future: hybrid approaches, specialized hardware

---

### Question 6.2: Scalability

**Examiner:** "As we scale to thousands of logical qubits, what decoding challenges arise?"

**Follow-up probes:**
- How does decode time scale?
- What about memory requirements?
- How do you maintain real-time performance?

**Key points for strong answer:**
1. Syndrome data rate: $$O(d^2 \cdot n_{\text{logical}})$$ bits per μs
2. Need parallel decoding of independent logical qubits
3. Memory: $$O(d^3)$$ per logical qubit for windowed decoding
4. FPGA arrays or specialized ASICs
5. Hierarchical architecture: local fast decode, global error tracking

---

## Part 7: Advanced Topics

### Question 7.1: Soft Information

**Examiner:** "How can soft information from measurements improve decoding?"

**Key points for strong answer:**
1. Analog measurement outcomes vs binary syndrome
2. Soft info encodes measurement confidence
3. Modify MWPM edge weights based on confidence
4. Can improve threshold by 10-20%
5. Naturally handles weak measurements

---

### Question 7.2: Correlated Noise

**Examiner:** "How should decoders be modified for correlated noise?"

**Key points for strong answer:**
1. Modify edge weights to account for correlations
2. Tensor network methods for exact marginalization
3. Neural networks learn correlations implicitly
4. Need characterization of noise correlations
5. Threshold can be much lower for adversarial correlations

---

## Whiteboard Exercises

### Exercise 1: Matching Graph Construction

Draw the matching graph for a distance-5 surface code with syndrome:
```
. • . . .
. . . . •
. . . . .
. . • . .
. . . . .
```

Expected approach:
1. Place vertices at defects (3 total)
2. Add boundary vertices
3. Draw edges with Manhattan distance weights
4. Identify minimum weight matching

---

### Exercise 2: Union-Find Cluster Growth

Demonstrate 3 steps of union-find cluster growth for:
```
• . . . .
. . . . .
. . . . •
. . . . .
• . . . .
```

Expected approach:
1. Draw initial clusters (radius 0)
2. Grow each by 1
3. Identify when clusters merge
4. Determine final pairing

---

### Exercise 3: BP Message Update

For a simple repetition code factor graph, compute one round of message passing with:
- Prior: $$p = 0.1$$
- Syndrome: $$(1, 0)$$

Expected approach:
1. Draw factor graph
2. Initialize messages
3. Compute factor-to-variable update
4. Compute variable-to-factor update
5. Estimate errors

---

## Examination Tips

### Preparation

1. **Know the numbers:** Thresholds, complexities, decoder comparisons
2. **Understand trade-offs:** Speed vs accuracy, simplicity vs performance
3. **Practice whiteboard:** Graph construction, algorithm steps
4. **Read recent papers:** AlphaQubit, fusion blossom, QLDPC decoders

### During the Exam

1. **Start with context:** Why decoding matters for fault tolerance
2. **Use diagrams:** Matching graphs, factor graphs, circuits
3. **Be quantitative:** Give actual threshold values, complexities
4. **Acknowledge limitations:** No decoder is perfect for all situations
5. **Connect topics:** Link decoders to threshold theorem, fault tolerance

### Common Questions to Prepare

1. "Why is the surface code threshold so high?"
2. "Compare MWPM and union-find for a specific scenario"
3. "How would you decode in the presence of correlated noise?"
4. "What is the bottleneck for large-scale fault tolerance?"
5. "How might decoding change with QLDPC codes?"
