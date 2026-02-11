# Week 175: QLDPC Codes - Comprehensive Review Guide

## Introduction

Quantum low-density parity-check (QLDPC) codes represent the frontier of quantum error correction theory. This review guide covers the journey from classical LDPC codes to the breakthrough asymptotically good quantum codes of Panteleev-Kalachev, and their implications for constant-overhead fault-tolerant quantum computation.

---

## Part I: Classical LDPC Foundations

### 1.1 Definition and Properties

**Definition:** A classical LDPC code is defined by a sparse parity-check matrix $$H$$ where:
- Each row has at most $$w_r$$ ones (row weight)
- Each column has at most $$w_c$$ ones (column weight)
- $$w_r, w_c = O(1)$$ as $$n \to \infty$$

**Parameters:**
- Block length: $$n$$
- Dimension: $$k = n - \text{rank}(H)$$
- Rate: $$R = k/n$$
- Distance: $$d$$ (minimum weight of non-zero codeword)

### 1.2 Tanner Graph Representation

A Tanner graph $$G = (V \cup C, E)$$ is a bipartite graph:
- Variable nodes $$V$$: one per bit (column of $$H$$)
- Check nodes $$C$$: one per parity check (row of $$H$$)
- Edge $$(v_i, c_j) \in E$$ iff $$H_{ji} = 1$$

**Decoding interpretation:**
- Messages flow along edges
- Belief propagation updates beliefs iteratively
- Converges to marginal probabilities (for tree-like graphs)

### 1.3 Asymptotically Good Classical LDPC

**Theorem (Gallager, Sipser-Spielman):** There exist families of classical LDPC codes with:
$$[n, k = \Theta(n), d = \Theta(n)]$$

**Construction methods:**
1. Random regular bipartite graphs (probabilistic)
2. Explicit algebraic constructions (expander-based)
3. Protograph lifting

**Key insight:** Expansion of the Tanner graph implies linear minimum distance.

### 1.4 Capacity-Achieving Property

**Shannon capacity:** Maximum rate for reliable communication over noisy channel.

**Result:** Properly designed LDPC codes with BP decoding achieve capacity on symmetric channels.

This is why LDPC codes are used in modern communications (5G, WiFi, etc.).

---

## Part II: From Classical to Quantum LDPC

### 2.1 The CSS Construction Challenge

For CSS codes, we need matrices $$H_X$$ and $$H_Z$$ satisfying:
$$H_X H_Z^T = 0 \mod 2$$

**Problem:** If $$H_X$$ and $$H_Z$$ are both sparse, the constraint $$H_X H_Z^T = 0$$ is highly restrictive.

**Naive approach:** Set $$H_Z = H_X^T$$ (self-orthogonal code)
- Works but typically gives $$k = O(1)$$ logical qubits
- Not efficient encoding

### 2.2 QLDPC Code Definition

**Definition:** A quantum LDPC code is a stabilizer code where:
1. Each stabilizer generator has weight $$w = O(1)$$
2. Each qubit participates in $$O(1)$$ stabilizers

Equivalently: Parity-check matrices $$H_X, H_Z$$ are sparse.

**Notation:** $$[[n, k, d]]_w$$ for QLDPC with stabilizer weight $$w$$.

### 2.3 Early Constructions

**Surface/Toric codes:**
- Weight-4 stabilizers
- $$[[n, O(1), \Theta(\sqrt{n})]]$$
- Not asymptotically good (sublinear rate and distance)

**Bicycle codes (MacKay et al.):**
- From circulant matrices
- Better rate but still $$d = O(\sqrt{n})$$

**Key limitation:** All early constructions had $$d = O(\sqrt{n})$$ or worse.

### 2.4 The QLDPC Conjecture

**Conjecture:** There exist QLDPC codes with parameters:
$$[[n, k = \Theta(n), d = \Theta(n)]]$$

This remained open for over 20 years until 2021.

---

## Part III: Hypergraph Product Codes

### 3.1 Construction

**Input:** Two classical codes $$C_1 = [n_1, k_1, d_1]$$ with check matrix $$H_1$$, and $$C_2 = [n_2, k_2, d_2]$$ with check matrix $$H_2$$.

Let $$m_i = n_i - k_i$$ (number of checks).

**Quantum code:** $$[[n, k, d]]$$ where:
- $$n = n_1 m_2 + m_1 n_2$$
- $$k = k_1 k_2$$
- $$d = \min(d_1, d_2)$$

**Parity-check matrices:**
$$H_X = \begin{pmatrix} H_1 \otimes I_{m_2} & I_{n_1} \otimes H_2^T \end{pmatrix}$$
$$H_Z = \begin{pmatrix} I_{m_1} \otimes H_2 & H_1^T \otimes I_{n_2} \end{pmatrix}$$

### 3.2 Commutativity Verification

$$H_X H_Z^T = (H_1 \otimes I)(I \otimes H_2^T) + (I \otimes H_2^T)(H_1^T \otimes I)$$
$$= H_1 \otimes H_2^T + H_1 \otimes H_2^T = 0 \mod 2$$ $$\checkmark$$

### 3.3 Parameter Analysis

Using good classical LDPC codes with $$[n, \Theta(n), \Theta(n)]$$:

- **Rate:** $$k/n = k_1 k_2 / (n_1 m_2 + m_1 n_2) = \Theta(1)$$
- **Distance:** $$d = \Theta(\sqrt{n})$$ (not linear!)

**Why not linear?**

The distance is limited by $$\min(d_1, d_2)$$, and since $$n \approx n_1 n_2$$, we have:
$$d = \Theta(n_1) = \Theta(\sqrt{n})$$

### 3.4 Significance

Hypergraph product codes were:
1. First constant-rate QLDPC codes
2. Proof that rate > 0 is achievable
3. Foundation for further improvements

But they did not resolve the QLDPC conjecture.

---

## Part IV: Lifted Product Codes

### 4.1 Lifting Operation

**Setting:** Group $$G$$ acting on set $$S$$, base graph $$\Gamma$$.

**Lifted graph:** $$\tilde{\Gamma}$$ with:
- Vertices: $$V(\Gamma) \times G$$
- Edges: lifted by group elements

**Effect:** Replaces each vertex with $$|G|$$ copies, each edge with a permuted connection.

### 4.2 Lifted Product Construction

**Input:**
- Base matrices $$A, B$$ over group ring $$\mathbb{F}_2[G]$$
- Classical Tanner codes from these matrices

**Output:** Quantum code via hypergraph product of lifted codes.

**Key advantage:** Group structure provides additional constraints that can improve distance.

### 4.3 Abelian vs Non-Abelian Groups

**Abelian lifting (e.g., $$G = \mathbb{Z}_\ell$$):**
- Explicit constructions possible
- Parameters: $$[[n, k = \Theta(n), d = \Theta(n/\log n)]]$$
- Almost linear distance, but not quite

**Non-Abelian lifting:**
- More complex structure
- Can achieve true linear distance
- Key to Panteleev-Kalachev breakthrough

### 4.4 Expander Graphs

**Definition:** A $$d$$-regular graph $$G$$ is a $$\lambda$$-expander if:
$$\lambda = \max_{v \perp \mathbf{1}} \frac{\|Av\|}{\|v\|} \leq \lambda < d$$

where $$A$$ is the adjacency matrix.

**Expansion property:** Every subset $$S \subset V$$ has boundary:
$$|\partial S| \geq h(G) |S|$$

where $$h(G) > 0$$ is the expansion constant.

**Relevance to codes:** Expansion implies errors must grow large before becoming degenerate $$\to$$ linear distance.

---

## Part V: Panteleev-Kalachev Codes

### 5.1 Main Result

**Theorem (Panteleev-Kalachev, 2021-2022):**
There exist explicit families of QLDPC codes with parameters:
$$[[n, k = \Theta(n), d = \Theta(n)]]$$
and constant stabilizer weight $$w = O(1)$$.

This resolved the 20+ year old QLDPC conjecture.

### 5.2 Construction Overview

1. **Cayley graph:** Take Cayley graph of a carefully chosen finite group $$G$$ with generators $$S$$.

2. **Double cover:** Modify to ensure CSS compatibility.

3. **Tanner code:** Define local codes at each vertex using small classical codes.

4. **Lifted product:** Apply the lifted product construction.

5. **Non-Abelian choice:** The group $$G$$ must be non-Abelian for linear distance.

### 5.3 Why Non-Abelian?

**Abelian limitation:** For Abelian groups, the lifted product has a quotient structure that limits distance to $$O(n/\log n)$$.

**Non-Abelian advantage:** The non-commutativity breaks this quotient structure, allowing errors to "spread" in ways that make low-weight logical operators impossible.

**Intuition:** Non-Abelian groups have more complex representation theory, providing richer structure for error correction.

### 5.4 Proof Sketch

1. **Expansion:** Show the Cayley graph has good expansion (spectral gap).

2. **Distance bound:** Use expansion to prove that any non-trivial logical operator must have weight $$\Omega(n)$$.

3. **Rate bound:** Count stabilizers vs qubits to show $$k = \Theta(n)$$.

4. **Locality:** Verify stabilizer weight is $$O(1)$$.

### 5.5 Explicit Parameters

For the Panteleev-Kalachev construction:
- Rate: $$R = k/n \geq c$$ for some constant $$c > 0$$
- Relative distance: $$d/n \geq c'$$ for some constant $$c' > 0$$
- Stabilizer weight: $$w \leq w_0$$ constant

Specific values of $$c, c', w_0$$ depend on the chosen parameters.

---

## Part VI: Constant-Overhead Fault Tolerance

### 6.1 Implications for Magic State Distillation

**Previous best:** $$N_{\text{magic}} = O(\log^\gamma(1/\epsilon))$$ with $$\gamma > 0$$

**With QLDPC:** $$N_{\text{magic}} = O(1)$$ per logical T gate!

### 6.2 How It Works

1. **Encode in QLDPC:** Encode many magic states in a QLDPC code
2. **Linear distance:** Provides polynomial error suppression per round
3. **Constant rate:** Only $$O(1)$$ overhead per output state
4. **Single round:** One round of distillation suffices for any target error

### 6.3 Single-Shot Error Correction

**Definition:** Error correction where a single round of noisy syndrome measurement suffices.

**QLDPC advantage:** Good QLDPC codes enable single-shot correction due to expansion properties.

**Implication:** No need for $$O(d)$$ syndrome measurement rounds.

### 6.4 Resource Estimates

**Space overhead:**
- QLDPC: $$O(1)$$ physical qubits per logical qubit
- Surface code: $$O(d^2) = O(\log^2(1/\epsilon))$$

**Time overhead:**
- QLDPC: $$O(1)$$ per logical gate (amortized)
- Surface code: $$O(\log^\gamma(1/\epsilon))$$ for magic states

**Total:** QLDPC achieves asymptotically optimal $$O(1)$$ overhead.

---

## Part VII: Practical Considerations

### 7.1 Connectivity Requirements

**Challenge:** QLDPC codes typically require non-local connectivity.

**Surface code advantage:** 2D nearest-neighbor layout matches hardware.

**QLDPC solutions:**
1. Use 3D or higher-dimensional embeddings
2. Accept long-range connections
3. Design codes with approximate geometric locality

### 7.2 Decoding QLDPC Codes

**MWPM:** Designed for surface code, doesn't directly apply.

**BP-based decoders:**
- Natural fit for LDPC structure
- Requires modifications for quantum (short cycles)
- BP+OSD achieves reasonable thresholds

**Current research:** Efficient decoders matching MWPM quality for QLDPC.

### 7.3 Threshold Estimates

**Challenge:** QLDPC codes may have lower thresholds than surface codes.

**Reasons:**
1. Higher-weight stabilizers (typically weight 6-10 vs 4)
2. More complex syndrome extraction
3. Less mature decoder algorithms

**Estimates:** Thresholds around 1-5% for phenomenological noise, 0.1-1% for circuit-level.

### 7.4 Recent Experimental Progress

**Small-scale demonstrations:**
- IBM/Google: Small QLDPC codes on superconducting qubits
- Trapped ions: Higher connectivity enables QLDPC
- Photonic systems: Natural non-locality

**Challenges:**
- Scaling to useful sizes
- Achieving below-threshold error rates
- Real-time decoding

---

## Part VIII: Open Problems and Future Directions

### 8.1 Geometrically Local QLDPC

**Question:** Can we achieve good QLDPC with geometric locality in 2D or 3D?

**Partial results:**
- 4D: Possible with good parameters
- 3D: Almost optimal codes exist (2025)
- 2D: Strong evidence against good codes (BPT bound)

### 8.2 Threshold Gap

**Question:** Can QLDPC codes achieve thresholds comparable to surface codes?

**Approaches:**
- Improved decoder design
- Tailored noise models
- Code optimization for specific hardware

### 8.3 Practical Code Selection

**Question:** At what scale do QLDPC codes become advantageous over surface codes?

**Factors:**
- Target logical error rate
- Hardware connectivity constraints
- Decoder latency requirements
- Physical error rate

---

## Summary of Key Results

### Code Parameters

$$\boxed{\text{Surface code: } [[d^2, 1, d]] \implies k/n = O(1/d^2), \quad d = O(\sqrt{n})}$$

$$\boxed{\text{Hypergraph product: } [[n, \Theta(n), \Theta(\sqrt{n})]] \implies R = \Theta(1), \quad d = O(\sqrt{n})}$$

$$\boxed{\text{Panteleev-Kalachev: } [[n, \Theta(n), \Theta(n)]] \implies R = \Theta(1), \quad d = \Theta(n)}$$

### Overhead Scaling

$$\boxed{\text{Surface code: } O(\log^2(1/\epsilon)) \text{ qubits per logical qubit}}$$

$$\boxed{\text{QLDPC: } O(1) \text{ qubits per logical qubit}}$$

### Magic State Distillation

$$\boxed{\text{Standard: } N = O(\log^\gamma(1/\epsilon)), \quad \gamma \approx 1-2.5}$$

$$\boxed{\text{QLDPC-based: } N = O(1) \text{ (constant overhead)}}$$

---

## References

1. Tillich, J.-P., & Zémor, G. (2014). "Quantum LDPC codes with positive rate and minimum distance proportional to the square root of the blocklength." IEEE Trans. Inf. Theory.

2. Panteleev, P., & Kalachev, G. (2022). "Asymptotically good quantum and locally testable classical LDPC codes." STOC 2022.

3. Leverrier, A., & Zémor, G. (2022). "Quantum Tanner codes." arXiv:2202.13641.

4. Dinur, I., et al. (2023). "Good quantum LDPC codes with linear time decoders." STOC 2023.

5. Bravyi, S., et al. (2024). "High-threshold and low-overhead fault-tolerant quantum memory." Nature.

6. Error Correction Zoo: https://errorcorrectionzoo.org/
