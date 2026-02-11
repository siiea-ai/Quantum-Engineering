# Week 171: Code Families - Oral Practice

## Introduction

This document provides oral examination practice for the code families material. The questions test both breadth (knowing multiple code families) and depth (understanding trade-offs and applications).

---

## Short-Answer Questions (2-3 minutes each)

### Question 1: Code Family Overview

**Examiner asks:** "Name five major quantum error-correcting code families and one distinguishing feature of each."

**Model answer:**
1. **CSS Codes:** Separate X and Z error correction; built from classical codes
2. **Reed-Muller Codes:** Support transversal gates at higher levels of Clifford hierarchy
3. **Color Codes:** Support transversal implementation of full Clifford group
4. **Reed-Solomon Codes:** Achieve quantum Singleton bound (MDS codes)
5. **Surface/Toric Codes:** Highest known error threshold (~1%); local stabilizers on 2D lattice

---

### Question 2: Transversal Gates

**Examiner asks:** "What is a transversal gate and why is it important for fault tolerance?"

**Key points:**
- Transversal: $$\overline{U} = U^{\otimes n}$$ (same gate on each physical qubit)
- Single fault can't propagate to multiple errors within a code block
- Enables fault-tolerant implementation of certain gates
- Different codes support different transversal gates

---

### Question 3: Threshold Theorem

**Examiner asks:** "State the threshold theorem and explain its significance."

**Model answer:** "The threshold theorem states that if the physical error rate $$p$$ is below a threshold $$p_{\text{th}}$$, then arbitrarily long quantum computations can be performed with arbitrarily small logical error rate using fault-tolerant techniques. The significance is that it proves scalable quantum computing is possible in principle—we don't need perfect qubits, just good enough ones. The threshold depends on the code family and error model; surface codes achieve ~1% while concatenated codes achieve ~0.01%."

---

### Question 4: Why Multiple Code Families?

**Examiner asks:** "Why do we need multiple code families? Why not just use the 'best' code?"

**Key points:**
- Different codes optimize different metrics (threshold vs. transversal gates vs. overhead)
- No code is best in all metrics
- Hardware constraints matter (2D connectivity vs. all-to-all)
- Application requirements vary (communication vs. computation)

---

### Question 5: Concatenated vs. Topological

**Examiner asks:** "Compare concatenated codes to topological codes."

**Comparison:**

| Aspect | Concatenated | Topological |
|--------|--------------|-------------|
| Threshold | ~$$10^{-4}$$ | ~$$10^{-2}$$ |
| Connectivity | All-to-all | 2D local |
| Transversal gates | Code-dependent | Limited |
| Overhead scaling | Polylogarithmic | Polynomial |

---

## Extended Explanation Questions (5-10 minutes)

### Question 6: Reed-Muller Code Properties

**Examiner asks:** "Explain quantum Reed-Muller codes and their transversal gate properties."

**Structure:**

1. **Classical RM background** (1-2 min):
   - $$RM(r, m)$$: polynomials of degree $$\leq r$$ over $$\mathbb{F}_2^m$$
   - Parameters: $$[2^m, \sum_{i=0}^r \binom{m}{i}, 2^{m-r}]$$
   - Duality: $$RM(r, m)^\perp = RM(m-r-1, m)$$

2. **Quantum construction** (2 min):
   - CSS construction from nested RM codes
   - Need $$RM(r_2, m)^\perp \subset RM(r_1, m)$$
   - Standard choice: $$r_1 + r_2 \geq m - 1$$

3. **Transversal gates** (2-3 min):
   - $$QRM(r, m)$$ supports transversal $$\mathcal{C}_{r+2}$$
   - Example: $$[[15, 1, 3]]$$ supports transversal T
   - Key insight: symmetry of RM codes under permutations

4. **Trade-offs** (1 min):
   - Higher $$r$$ → more gates but lower rate/distance
   - Code switching between RM codes for universal computation

---

### Question 7: Color Code Deep Dive

**Examiner asks:** "Describe color codes, their structure, and advantages."

**Cover:**

1. **Lattice structure:**
   - 2-colorable lattice with 3 colors (RGB)
   - Qubits on vertices, stabilizers on faces
   - Both X and Z stabilizers on same faces

2. **Stabilizers:**
   - $$X_f = \prod_{v \in f} X_v$$ for each face
   - $$Z_f = \prod_{v \in f} Z_v$$ for each face
   - Symmetric structure enables transversal H

3. **Transversal gates:**
   - Full Clifford group in 2D
   - 3D color codes: transversal T
   - Why: X and Z stabilizers have identical support

4. **Comparison to surface codes:**
   - Lower threshold than surface (~0.1% vs ~1%)
   - But better transversal gates
   - Different error models may favor different codes

---

### Question 8: Code Selection Problem

**Examiner asks:** "How would you select a code for a specific application?"

**Framework:**

1. **Assess constraints:**
   - Physical error rate
   - Connectivity of hardware
   - Required logical error rate
   - Gate set needed

2. **Filter by threshold:**
   - If $$p > 10^{-3}$$: Need surface codes
   - If $$p \sim 10^{-4}$$: Surface or color codes
   - If $$p < 10^{-5}$$: Concatenated codes viable

3. **Filter by connectivity:**
   - 2D nearest-neighbor: Surface or color codes
   - All-to-all: Any code family works

4. **Filter by gate requirements:**
   - Clifford only: Surface code with Hadamard trick
   - Native T gate: [[15,1,3]] RM or 3D color
   - Full universality: Magic state distillation with any code

5. **Optimize overhead:**
   - Calculate physical qubits needed
   - Consider decoding complexity
   - Account for auxiliary qubits

---

### Question 9: The Threshold Theorem Proof Sketch

**Examiner asks:** "Outline the proof of the threshold theorem using concatenated codes."

**Proof outline:**

1. **Setup:**
   - Base code: $$[[n, 1, d]]$$ with $$d \geq 3$$
   - Physical error rate: $$p$$
   - $$L$$ levels of concatenation

2. **Single level analysis:**
   - Code corrects $$t = 1$$ error
   - Failure needs $$\geq 2$$ errors
   - Failure probability: $$p_1 \leq C p^2$$

3. **Recursive argument:**
   - Level $$\ell$$ failure probability: $$p_\ell$$
   - $$p_{\ell+1} \leq C p_\ell^2$$
   - Solution: $$p_\ell \leq (Cp)^{2^\ell}/C$$

4. **Threshold condition:**
   - For convergence: $$Cp < 1$$
   - Threshold: $$p_{\text{th}} = 1/C$$

5. **Achieving target $$\epsilon$$:**
   - $$L = O(\log\log(1/\epsilon))$$ levels suffice
   - Physical qubits: $$n^L = \text{polylog}(1/\epsilon)$$

---

## Deep-Dive Questions (15-20 minutes)

### Question 10: Complete Code Comparison

**Examiner asks:** "Compare five code families for implementing Shor's algorithm."

**Analysis dimensions:**

1. **Physical qubits needed:**
   - Shor's algorithm needs ~2000 logical qubits
   - With error correction: $$2000 \times $$ overhead per logical qubit

2. **For each family:**

   **Surface codes:**
   - Threshold: 1%, overhead: ~1000 physical/logical for $$p = 10^{-3}$$
   - Total: ~2 million physical qubits
   - Limitation: Need magic state distillation for T gates

   **Color codes:**
   - Threshold: 0.1%, needs $$p < 10^{-3}$$
   - Overhead similar but with transversal Clifford
   - Better if T gates rare in algorithm

   **Concatenated (Steane):**
   - Threshold: $$10^{-4}$$, needs very low $$p$$
   - Overhead: polylog in $$1/\epsilon$$
   - Good for very low error rates

   **Reed-Muller [[15,1,3]]:**
   - Native T gate (valuable for Shor's)
   - Higher per-logical-qubit overhead
   - May reduce magic state distillation cost

3. **Recommendation:**
   Depends on $$p$$:
   - $$p \sim 10^{-3}$$: Surface code with magic states
   - $$p \sim 10^{-4}$$: Color codes or hybrid approach
   - $$p \sim 10^{-5}$$: Concatenated codes

---

### Question 11: Bounds and Limitations

**Examiner asks:** "Discuss the fundamental limits on quantum codes and how different families approach them."

**Discussion:**

1. **Quantum Singleton bound:** $$k \leq n - 2d + 2$$
   - RS codes achieve this (MDS)
   - Most qubit codes don't achieve it

2. **Quantum Hamming bound:**
   - Only [[5,1,3]] and related codes saturate
   - Degenerate codes can beat the bound

3. **Threshold limitations:**
   - 2D local codes: threshold bounded by percolation
   - Surface codes likely near optimal for 2D
   - Higher dimensions may help

4. **Transversal gate limitations:**
   - Eastin-Knill theorem: No code has transversal universal gate set
   - Must use magic states or code switching

5. **Practical limitations:**
   - Decoding complexity
   - Syndrome measurement errors
   - Connectivity constraints

---

### Question 12: Design Problem

**Examiner asks:** "Design a fault-tolerant architecture for a quantum computer with 2D connectivity and physical error rate $$10^{-3}$$."

**Design process:**

1. **Choose primary code:**
   - Surface code (highest threshold for 2D)
   - Target distance based on desired logical rate

2. **Determine parameters:**
   - For $$\epsilon = 10^{-10}$$ logical error rate
   - Need $$d \approx 15-20$$
   - Physical qubits: $$(2d-1)^2 \approx 900$$ per logical

3. **Handle non-Clifford gates:**
   - Magic state distillation
   - Estimate distillation overhead: ~10-100x for high-quality T states

4. **Layout:**
   - Data qubits in lattice
   - Magic state factories at edges
   - Routing for long-range connections

5. **Resource estimate:**
   - 1000 logical qubits → ~1M physical qubits
   - Plus magic state factories → ~10M total
   - Cycle time dominated by error correction

---

## Common Exam Mistakes

1. **Confusing parameters:** Keep track of which $$n, k, d$$ go with which code

2. **Forgetting constraints:** Not all CSS codes are color codes; not all stabilizer codes are CSS

3. **Overgeneralizing thresholds:** Thresholds depend on error model and decoder

4. **Missing trade-offs:** "Best code" depends on what you're optimizing

5. **Ignoring practicality:** Theoretical optimal may not be implementable

---

## Key Facts to Know Cold

| Code | Parameters | Transversal | Threshold |
|------|------------|-------------|-----------|
| [[5,1,3]] | Perfect | Paulis | - |
| Steane [[7,1,3]] | CSS | H, S, CNOT | ~$$10^{-4}$$ (concat) |
| Shor [[9,1,3]] | Non-CSS | CNOT | ~$$10^{-4}$$ (concat) |
| [[15,1,3]] RM | CSS | T gate | - |
| Surface | $$[[d^2, 1, d]]$$ | Paulis | ~1% |
| 2D Color | $$[[n, 1, d]]$$ | Clifford | ~0.1% |

---

**Oral Practice Document Created:** February 10, 2026
