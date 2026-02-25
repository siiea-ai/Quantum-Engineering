# Week 174: Decoding Algorithms - Problem Set

## Instructions

This problem set contains 27 problems on decoding algorithms for quantum error correction. Problems cover MWPM, union-find, belief propagation, and neural network decoders.

**Difficulty Levels:**
- **(B)** Basic - Direct application of concepts
- **(I)** Intermediate - Requires synthesis and calculation
- **(A)** Advanced - Research-level or proof-based

**Time Estimate:** 8-10 hours total

---

## Section 1: The Decoding Problem (Problems 1-5)

### Problem 1 (B)

Consider a $$[[5,1,3]]$$ code with stabilizer generators:
$$S_1 = XZZXI, \quad S_2 = IXZZX, \quad S_3 = XIXZZ, \quad S_4 = ZXIXZ$$

(a) If an $$X_3$$ error occurs, calculate the syndrome (the eigenvalues of each stabilizer).

(b) If the syndrome is $$(+1, -1, +1, -1)$$, list all weight-1 errors consistent with this syndrome.

(c) What is the minimum weight error consistent with syndrome $$(−1, −1, −1, −1)$$?

---

### Problem 2 (I)

**Degeneracy in the Surface Code**

Consider a distance-3 surface code.

(a) How many distinct single-qubit errors are there?

(b) How many distinct syndromes can be produced by single-qubit errors?

(c) Explain why the number of syndromes is less than the number of errors.

(d) Give an example of two different errors that produce the same syndrome.

---

### Problem 3 (I)

**Maximum Likelihood Decoding**

For independent depolarizing noise with error rate $$p$$:

(a) Write the probability $$P(E)$$ for an error of weight $$w$$ on $$n$$ qubits.

(b) Show that maximum likelihood decoding reduces to minimum weight decoding when $$p < 0.5$$.

(c) What happens when $$p > 0.5$$? Does minimum weight decoding still work?

---

### Problem 4 (B)

**Decoding Complexity**

(a) Why is optimal decoding (finding the exact maximum likelihood error class) computationally hard in general?

(b) What special structure of the surface code makes efficient decoding possible?

(c) Name two complexity classes associated with optimal decoding and explain their significance.

---

### Problem 5 (A)

**Proof Problem: Equivalence Classes**

Prove that for a stabilizer code:

(a) Two errors $$E_1, E_2$$ with the same syndrome differ by a stabilizer element (i.e., $$E_1 E_2^\dagger \in S$$).

(b) If $$E_1 E_2^\dagger \in S$$ (stabilizer group), then $$E_1$$ and $$E_2$$ have the same effect on all logical states.

(c) Conclude that the decoder need only identify the equivalence class, not the exact error.

---

## Section 2: MWPM Decoder (Problems 6-12)

### Problem 6 (B)

**Graph Construction**

For a distance-5 surface code with the following X-error syndrome (marked with •):

```
. . . . .
. • . . .
. . . • .
. . . . .
. . . . .
```

(a) How many vertices are in the matching graph (excluding boundary)?

(b) Draw the matching graph including boundary vertices.

(c) What is the minimum weight matching?

---

### Problem 7 (I)

**MWPM Threshold Calculation**

The surface code has threshold $$p_{\text{th}} \approx 10.3\%$$ under MWPM decoding.

(a) At error rate $$p = 1\%$$, estimate the logical error rate for distance $$d = 5$$.

(b) What distance is needed to achieve logical error rate $$< 10^{-10}$$ at $$p = 1\%$$?

(c) How does the qubit count scale with target logical error rate?

---

### Problem 8 (I)

**Weighted Matching**

Consider a modified noise model where X errors have probability $$p_X = 0.01$$ and Z errors have probability $$p_Z = 0.001$$.

(a) How should edge weights in the matching graph be modified?

(b) If two defects are separated by 3 qubits, what is the weight if the path uses only X errors?

(c) What is the weight if the path uses only Z errors?

---

### Problem 9 (A)

**Blossom Algorithm Analysis**

Edmonds' blossom algorithm finds minimum-weight perfect matching in $$O(|V|^3)$$ time.

(a) Explain the role of "blossoms" (odd cycles) in the algorithm.

(b) Why can't a simpler greedy matching algorithm achieve optimality?

(c) Describe a graph where greedy matching fails but blossom algorithm succeeds.

---

### Problem 10 (I)

**Measurement Errors**

With measurement error rate $$p_m = 0.1\%$$:

(a) How does the matching graph change to handle measurement errors?

(b) Estimate the effective error rate seen by the decoder.

(c) What is the approximate threshold for combined data and measurement errors?

---

### Problem 11 (B)

**PyMatching Implementation**

Write pseudocode (or actual Python) to:

(a) Create a parity check matrix for the repetition code of length 5.

(b) Initialize a PyMatching decoder.

(c) Decode a syndrome $$s = (1, 0, 0, 1)$$.

(d) Verify the correction is correct.

---

### Problem 12 (A)

**MWPM Optimality**

(a) Prove that MWPM decoding is equivalent to maximum likelihood decoding for independent Pauli noise on the surface code (assuming no degeneracy tie-breaking).

(b) Under what conditions does MWPM fail to be optimal?

(c) How much can degeneracy-aware decoding improve upon MWPM?

---

## Section 3: Union-Find Decoder (Problems 13-17)

### Problem 13 (B)

**Union-Find Data Structure**

(a) Describe the `find` operation with path compression.

(b) Describe the `union` operation with union by rank.

(c) What is the amortized time complexity of $$m$$ operations on $$n$$ elements?

---

### Problem 14 (I)

**Cluster Growth Algorithm**

For the syndrome:
```
. • . . .
. . . . .
. . • . .
. . . . •
. . . . .
```

(a) Show the cluster state after 1 growth step (each cluster grows by 1 in each direction).

(b) After how many steps do the first two clusters merge?

(c) What is the final matching implied by cluster merging?

---

### Problem 15 (I)

**Complexity Analysis**

For a distance-$$d$$ surface code with $$n = O(d^2)$$ qubits:

(a) What is the maximum number of growth steps before termination?

(b) How many union-find operations occur per growth step?

(c) Derive the total time complexity.

---

### Problem 16 (I)

**Threshold Comparison**

The union-find decoder has threshold $$p_{\text{th}} \approx 9.9\%$$ vs MWPM's $$10.3\%$$.

(a) Calculate the logical error rate difference at $$p = 1\%$$, $$d = 11$$.

(b) At what physical error rate does this difference become significant?

(c) When might you prefer union-find despite the lower threshold?

---

### Problem 17 (A)

**Proof: Union-Find Correctness**

(a) Prove that when union-find terminates, every syndrome defect is paired (or connected to boundary).

(b) Show that the pairing found is "locally optimal" in a precise sense.

(c) Give an example where union-find produces a different (suboptimal) matching than MWPM.

---

## Section 4: Belief Propagation (Problems 18-21)

### Problem 18 (B)

**Factor Graph Construction**

For a CSS code with X-check matrix:
$$H_X = \begin{pmatrix} 1 & 1 & 0 & 1 \\ 0 & 1 & 1 & 1 \end{pmatrix}$$

(a) Draw the factor graph with variable nodes (qubits) and factor nodes (checks).

(b) How many edges are in the graph?

(c) What is the girth (shortest cycle length) of this graph?

---

### Problem 19 (I)

**BP Message Passing**

For the factor graph from Problem 18, with prior error probability $$p = 0.1$$:

(a) Initialize the variable-to-factor messages.

(b) Compute the first round of factor-to-variable messages for syndrome $$s = (1, 0)$$.

(c) Describe why convergence might fail for this code.

---

### Problem 20 (I)

**BP Failure Modes**

(a) Explain why short cycles cause BP to fail.

(b) What is the minimum girth needed for BP to be asymptotically optimal?

(c) How do quantum codes compare to classical LDPC codes in terms of girth?

---

### Problem 21 (A)

**Ordered Statistics Decoding (OSD)**

OSD post-processes BP output to handle failures.

(a) Describe the OSD algorithm at a high level.

(b) What is the complexity of order-$$k$$ OSD?

(c) How does OSD-$$k$$ performance compare to MWPM for surface codes?

---

## Section 5: Neural Network Decoders (Problems 22-25)

### Problem 22 (B)

**Supervised Learning for Decoding**

(a) What is the input and output of a neural network decoder?

(b) How are training examples generated?

(c) What loss function is appropriate for this task?

---

### Problem 23 (I)

**Architecture Design**

(a) Why might a recurrent architecture be preferable to feedforward for decoding?

(b) How do transformers handle the spatial structure of the syndrome?

(c) What is the inference complexity of a transformer decoder with $$n$$ syndrome bits?

---

### Problem 24 (I)

**Training Considerations**

(a) How many training examples are typically needed for good generalization?

(b) How do you ensure the training set covers rare but important error patterns?

(c) What is the challenge of training on real hardware data vs simulation?

---

### Problem 25 (A)

**Neural Decoder Analysis**

AlphaQubit achieves 6% lower logical error than MWPM on Google hardware.

(a) What types of noise does AlphaQubit capture that MWPM misses?

(b) How would you verify that a neural decoder is not "cheating" (overfitting to test data)?

(c) How might neural decoders be combined with MWPM for best results?

---

## Section 6: Decoder Selection and Comparison (Problems 26-27)

### Problem 26 (I)

**Trade-off Analysis**

You are designing a fault-tolerant quantum computer with:
- Physical error rate: $$p = 0.5\%$$
- Target logical error rate: $$p_L < 10^{-12}$$
- Syndrome measurement rate: 1 MHz
- Code: Surface code

(a) What code distance is needed?

(b) How many physical qubits per logical qubit?

(c) Which decoder would you choose? Justify your answer.

(d) What is the real-time decoding requirement?

---

### Problem 27 (A)

**Decoder Design Challenge**

Design a decoding strategy for a system with:
- 1000 logical qubits
- Distance-21 surface code
- Correlated noise (5% of errors are two-qubit)
- FPGA and GPU available

(a) Propose a decoder architecture.

(b) Estimate the computational resources needed.

(c) How would you validate the decoder before deployment?

(d) What is your estimated logical error rate?

---

## Bonus Problems

### Problem 28 (A)

**Soft Information Decoding**

Instead of binary syndrome $$\pm 1$$, you receive continuous measurement outcomes $$m_i \in \mathbb{R}$$.

(a) How should edge weights in MWPM be modified to use this soft information?

(b) What is the potential threshold improvement?

(c) Design an experiment to test soft information decoding.

---

### Problem 29 (A)

**Correlated Decoding**

Consider a noise model where errors on adjacent qubits are correlated:
$$P(E_i, E_j | \text{adjacent}) = (1-p)^2 + cp^2$$

where $$c$$ is the correlation strength.

(a) How does this correlation affect the matching graph?

(b) Derive modified edge weights that account for correlations.

(c) How does the threshold change with correlation strength?

---

### Problem 30 (A)

**Real-Time Decoding**

You must decode a distance-15 surface code at 1 MHz.

(a) Calculate the number of syndrome bits per round.

(b) Estimate the MWPM decode time on a modern CPU.

(c) Design a parallelization strategy to meet the real-time requirement.

(d) Compare with FPGA implementation of union-find.

---

## Submission Guidelines

1. Show all work, including intermediate steps
2. For coding problems, include runnable code or detailed pseudocode
3. For analysis problems, state assumptions clearly
4. Diagrams should be clearly labeled
5. Numerical answers should include units and significant figures

**Solutions available in:** [Problem_Solutions.md](Problem_Solutions.md)
