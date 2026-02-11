# Week 174: Decoding Algorithms - Problem Solutions

## Section 1: The Decoding Problem

### Solution 1

**(a)** Syndrome for $$X_3$$ error:

Check each stabilizer by computing commutation:
- $$S_1 = XZZXI$$: $$X_3$$ commutes with $$Z_3$$? No, anticommutes. Syndrome: $$-1$$
- $$S_2 = IXZZX$$: $$X_3$$ anticommutes with $$Z_3$$. Syndrome: $$-1$$
- $$S_3 = XIXZZ$$: $$X_3$$ commutes with $$I_3$$? Yes. Syndrome: $$+1$$
- $$S_4 = ZXIXZ$$: $$X_3$$ commutes with $$I_3$$? Yes. Syndrome: $$+1$$

**Syndrome: $$(−1, −1, +1, +1)$$**

**(b)** For syndrome $$(+1, -1, +1, -1)$$:

Check each single-qubit error:
- $$X_1$$: Check $$S_1(X), S_2(I), S_3(X), S_4(Z)$$ → $$(+1, +1, +1, -1)$$ ✗
- $$X_2$$: $$S_1(Z), S_2(X), S_3(I), S_4(X)$$ → $$(−1, +1, +1, +1)$$ ✗
- $$Z_2$$: $$S_1(Z), S_2(X), S_3(I), S_4(X)$$ → $$(+1, -1, +1, -1)$$ ✓
- $$Y_2 = iXZ$$: Same syndrome as $$Z_2$$ for checking.

**Weight-1 errors with this syndrome: $$Z_2$$ (and $$Y_2$$ if considering Y errors)**

**(c)** For syndrome $$(−1, −1, −1, −1)$$:

All stabilizers give $$-1$$, meaning the error anticommutes with all generators.

Checking: $$Y_3 = iX_3Z_3$$ anticommutes with $$S_1, S_2$$ (due to $$Z_3$$) and also affects others.

Actually, let's systematically check: No single-qubit Pauli gives all $$-1$$.

Weight-2: Try $$X_1 Z_4$$:
- $$S_1$$: $$X$$ at 1, $$I$$ at 4 → $$+1 \cdot +1 = +1$$ ✗

After systematic search: **Minimum weight error is weight 2 or 3**, depending on specific check.

---

### Solution 2

**(a)** Distinct single-qubit errors on distance-3 surface code:

A $$d=3$$ rotated surface code has $$d^2 = 9$$ data qubits.
Each qubit can have $$X, Y, Z$$ error: $$3 \times 9 = 27$$ single-qubit errors.
Including identity: 28 possibilities (or 27 non-trivial).

**(b)** Number of distinct syndromes:

For distance-3 surface code: 4 X-checks + 4 Z-checks = 8 stabilizers.
Maximum syndromes: $$2^8 = 256$$.

But not all syndromes achievable by weight-1 errors.
Single X error: triggers 1 or 2 X-checks → limited syndrome patterns.

For weight-1 errors specifically: roughly $$9 \times 2 + 9 \times 2 = 36$$ syndrome patterns (crude estimate), but with overlaps.

Actual: Many fewer due to locality constraints.

**(c)** Why fewer syndromes than errors:

1. **Degeneracy:** Different errors can produce the same syndrome
2. **Locality:** Each error only triggers nearby checks
3. **Stabilizer structure:** Syndrome space dimension $$< n - k$$

**(d)** Example of same syndrome:

An X error on any qubit on the "left boundary" might pair with the boundary (virtual defect), giving same syndrome pattern as an X error at a symmetric position.

More concretely: $$X_1$$ and $$X_1 S$$ for some stabilizer $$S$$ give the same syndrome.

---

### Solution 3

**(a)** Probability of weight-$$w$$ error:

$$P(E) = p^w (1-p)^{n-w} \cdot (\text{combinatorial factor})$$

For a specific error $$E$$ of weight $$w$$:
$$P(E) = \left(\frac{p}{3}\right)^w (1-p)^{n-w}$$

(Factor of 1/3 if distinguishing X, Y, Z)

**(b)** Maximum likelihood reduces to minimum weight:

$$\arg\max_E P(E) = \arg\max_E p^{|E|}(1-p)^{n-|E|}$$
$$= \arg\max_E \left(\frac{p}{1-p}\right)^{|E|}$$

If $$p < 0.5$$, then $$\frac{p}{1-p} < 1$$, so we want minimum $$|E|$$.

$$\therefore \arg\max P(E) = \arg\min |E|$$ ✓

**(c)** When $$p > 0.5$$:

$$\frac{p}{1-p} > 1$$, so maximum likelihood seeks maximum weight error.

This is unphysical/unstable regime. Minimum weight decoding would fail.

However, if $$p$$ is known, can invert: look for maximum weight error and flip interpretation.

---

### Solution 4

**(a)** Computational hardness:

Optimal decoding is NP-hard because:
1. Exponentially many error patterns
2. Finding minimum weight element in a coset is hard
3. Related to decoding random linear codes

**(b)** Surface code structure enabling efficiency:

1. **Locality:** Each qubit participates in $$\leq 4$$ checks
2. **Planarity:** Check graph embeds in 2D
3. **Independence:** X and Z errors can be decoded separately

These reduce the problem to minimum-weight perfect matching, which is polynomial.

**(c)** Complexity classes:

- **NP-hard:** Decision version of optimal decoding
- **#P-complete:** Counting degenerate errors (for calculating exact probabilities)

Significance: Proves no polynomial algorithm exists unless P = NP.

---

### Solution 5

**(a)** Proof that same syndrome implies stabilizer difference:

Let $$E_1, E_2$$ have syndrome $$s$$. Then:
$$HE_1^T = s = HE_2^T$$
$$H(E_1 - E_2)^T = 0$$

So $$E_1 - E_2$$ (or $$E_1 E_2^\dagger$$ in multiplicative notation) is in the kernel of $$H$$, which is the stabilizer group $$S$$.

**(b)** Same logical effect:

If $$E_1 E_2^\dagger = S \in \mathcal{S}$$ (stabilizer group), then for any code state $$|\psi\rangle$$:
$$E_1|\psi\rangle = E_1 E_2^\dagger E_2|\psi\rangle = S E_2|\psi\rangle = E_2|\psi\rangle$$

since $$S|\psi\rangle = |\psi\rangle$$ for stabilizer states.

**(c)** Conclusion:

Since equivalent errors (differing by stabilizer) produce identical corrupted states, the decoder need only determine which equivalence class the error belongs to, not the specific error.

This is the foundation of degeneracy-aware decoding.

---

## Section 2: MWPM Decoder

### Solution 6

**(a)** Vertices in matching graph:

Two syndrome defects (marked •), so **2 vertices** (excluding boundary).

**(b)** Matching graph:

```
    B_top
     |
     • (defect 1)
    / \
   /   \
  •-----• (defect 2)
   \   /
    \ /
   B_bottom
```

Include boundary vertices for each defect near boundary, plus edges between all defects with weights = distances.

**(c)** Minimum weight matching:

Distance between defects: approximately 2-3 (depending on exact positions).
Distance to boundaries: varies.

Optimal matching pairs the two defects together (weight ≈ 2-3) rather than each to boundary (would be higher total weight).

---

### Solution 7

**(a)** Logical error rate estimate at $$d=5$$, $$p=1\%$$:

Using $$p_L \approx A \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$ for surface codes:

$$p_L \approx 0.1 \times \left(\frac{0.01}{0.103}\right)^3 = 0.1 \times (0.097)^3 \approx 0.1 \times 9 \times 10^{-4} \approx 9 \times 10^{-5}$$

**(b)** Distance for $$p_L < 10^{-10}$$:

$$10^{-10} = A \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$

$$\left(\frac{0.01}{0.103}\right)^{(d+1)/2} \approx 10^{-9}$$ (assuming $$A \approx 0.1$$)

$$0.097^{(d+1)/2} = 10^{-9}$$

$$\frac{d+1}{2} \log(0.097) = -9 \log(10)$$

$$\frac{d+1}{2} \times (-1.01) = -9$$

$$d+1 \approx 18$$, so $$d \approx 17$$

**Need approximately distance 17.**

**(c)** Qubit scaling:

Physical qubits: $$n \approx 2d^2 \approx 2 \times 289 = 578$$ per logical qubit.

For target $$p_L$$: $$d \propto \log(1/p_L)$$, so $$n \propto \log^2(1/p_L)$$.

---

### Solution 8

**(a)** Modified edge weights:

Edge weight should be $$-\log P(\text{error path})$$.

For path of length $$\ell$$ using X errors: $$w = -\ell \log p_X$$
For Z errors: $$w = -\ell \log p_Z$$

**(b)** Weight for 3-qubit X-error path:

$$w_X = -3 \log(0.01) = -3 \times (-4.6) = 13.8$$

**(c)** Weight for 3-qubit Z-error path:

$$w_Z = -3 \log(0.001) = -3 \times (-6.9) = 20.7$$

Z errors are rarer, so Z paths have higher weight (less likely).

---

### Solution 9

**(a)** Role of blossoms:

Odd cycles cannot be perfectly matched internally (odd number of vertices). The blossom algorithm contracts odd cycles into single vertices, solves recursively, then expands.

**(b)** Why greedy fails:

Greedy matching picks locally optimal edges, which may block globally optimal matchings.

**(c)** Example:

```
A---B---C
|       |
D-------E
```

Greedy might match A-B and D-E first, leaving C unmatched (if we need perfect matching).

Optimal: A-D, B-C (or similar), E to boundary.

---

### Solution 10

**(a)** Graph modification for measurement errors:

- Extend to 3D: (x, y, time)
- Vertices: syndrome flips between time steps
- Edges: connect through space (same time) and time (same position)
- Weight: probability of error type

**(b)** Effective error rate:

Combined error: $$p_{\text{eff}} \approx p_d + p_m$$ for small error rates.

With $$p_d = 1\%$$, $$p_m = 0.1\%$$: $$p_{\text{eff}} \approx 1.1\%$$

**(c)** Combined threshold:

Phenomenological model threshold ≈ 2.9%.

With measurement errors, need $$p_d + p_m < p_{\text{th}}^{\text{phenom}}$$.

---

### Solution 11

```python
import numpy as np
import pymatching

# (a) Parity check matrix for length-5 repetition code
H = np.array([
    [1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1]
])

# (b) Initialize decoder
matching = pymatching.Matching(H)

# (c) Decode syndrome s = (1, 0, 0, 1)
syndrome = np.array([1, 0, 0, 1])
correction = matching.decode(syndrome)
print("Correction:", correction)

# (d) Verify: H @ correction should equal syndrome
recovered_syndrome = H @ correction % 2
print("Recovered syndrome:", recovered_syndrome)
print("Match:", np.array_equal(recovered_syndrome, syndrome))
```

Expected output: Correction indicates errors at positions 0 and 4 (or equivalent).

---

### Solution 12

**(a)** MWPM = Maximum Likelihood for independent noise:

For independent Pauli noise, $$P(E) \propto p^{|E|}$$.

Maximum likelihood: $$\arg\max P(E) = \arg\min |E|$$.

MWPM finds minimum weight error chain connecting syndrome defects.

For surface code, the minimum weight error is exactly what MWPM computes.

**(b)** When MWPM fails:

1. **Correlated noise:** Independence assumption violated
2. **Degeneracy ties:** MWPM may pick wrong equivalence class
3. **Non-Pauli noise:** Probability model incorrect

**(c)** Degeneracy-aware improvement:

Small improvement possible (< 0.1% threshold increase) by explicitly computing equivalence class probabilities.

---

## Section 3: Union-Find Decoder

### Solution 13

**(a)** `find` with path compression:

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # Path compression
    return parent[x]
```

**(b)** `union` with union by rank:

```python
def union(x, y):
    root_x, root_y = find(x), find(y)
    if root_x == root_y:
        return
    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1
```

**(c)** Amortized complexity:

$$O(m \cdot \alpha(n))$$ for $$m$$ operations on $$n$$ elements.

$$\alpha(n)$$ is the inverse Ackermann function, $$< 5$$ for any practical $$n$$.

---

### Solution 14

**(a)** After 1 growth step:

Each cluster expands by 1 in each direction (Manhattan distance).

```
. ● . . .    ● = original defect
. . . . .    ○ = cluster boundary after growth
. . ● . .
. . . . ●
. . . . .
```

Defect 1 (row 0, col 1): cluster covers (0,0)-(1,2)
Defect 2 (row 2, col 2): cluster covers (1,1)-(3,3)
Defect 3 (row 3, col 4): cluster covers (2,3)-(4,5) [with boundary]

**(b)** First merge:

Distance between defects 1 and 2: approximately 2-3 steps.
Clusters merge when combined growth equals distance.

Merge after approximately **2 growth steps**.

**(c)** Final matching:

Depends on growth order. Likely: (defect1, defect2) pair, defect3 to boundary.

---

### Solution 15

**(a)** Maximum growth steps:

Maximum distance between defects or to boundary: $$O(d)$$.
So at most $$O(d)$$ growth steps.

**(b)** Operations per step:

Each of $$O(n)$$ sites may be checked.
Each check involves $$O(1)$$ union-find operations.
Total: $$O(n)$$ operations per step.

**(c)** Total complexity:

$$T = O(d) \times O(n) \times O(\alpha(n)) = O(dn \cdot \alpha(n))$$

For fixed aspect ratio $$n = O(d^2)$$:
$$T = O(d^3 \cdot \alpha(d^2))$$

But often expressed as $$O(n \cdot \alpha(n))$$ for the full decode.

---

### Solution 16

**(a)** Error rate difference at $$p=1\%$$, $$d=11$$:

MWPM: $$p_L^{\text{MWPM}} \approx A(0.01/0.103)^6 \approx 0.1 \times (0.097)^6 \approx 8 \times 10^{-8}$$

UF: $$p_L^{\text{UF}} \approx A(0.01/0.099)^6 \approx 0.1 \times (0.101)^6 \approx 1.1 \times 10^{-7}$$

Difference: factor of ~1.4x higher error for UF.

**(b)** When difference is significant:

Near threshold ($$p \to p_{\text{th}}$$), the difference matters more.

For $$p > 5\%$$, the threshold difference becomes critical.

**(c)** When to prefer union-find:

1. Real-time decoding required (< 1 μs)
2. Large code distances ($$d > 15$$)
3. FPGA implementation needed
4. Physical error rate well below threshold

---

### Solution 17

**(a)** Termination proof:

Invariant: Each cluster either contains an even number of defects, or is connected to boundary.

Growth continues until no odd-defect clusters remain isolated.

Since defects come in pairs (from errors) or can match to boundary, termination is guaranteed.

**(b)** Local optimality:

Union-find produces a matching where each pair is connected by the shortest path found during growth.

This is locally optimal: no single swap improves the matching.

**(c)** Suboptimal example:

```
A . . . . B
. . . . . .
. . . . . .
C . . . . D
```

UF might pair (A,C) and (B,D) based on growth timing.
MWPM might find (A,B) and (C,D) is better if boundary connections differ.

---

## Section 4: Belief Propagation

### Solution 18

**(a)** Factor graph:

```
Variable nodes: q1, q2, q3, q4
Factor nodes: c1, c2

Edges:
c1 -- q1, q2, q4
c2 -- q2, q3, q4
```

**(b)** Number of edges: 6 (count from adjacency)

**(c)** Girth: Shortest cycle through q2 → c1 → q4 → c2 → q2

Length: 4 (very short)

---

### Solution 19

**(a)** Initialize variable-to-factor messages:

$$\mu_{q_i \to c_j}(0) = 1-p = 0.9$$
$$\mu_{q_i \to c_j}(1) = p = 0.1$$

(Prior probability of error)

**(b)** Factor-to-variable update for syndrome $$(1, 0)$$:

For $$c_1$$ (syndrome = 1): must have odd parity among $$q_1, q_2, q_4$$

$$\mu_{c_1 \to q_1}(x) \propto \sum_{x_2, x_4: x \oplus x_2 \oplus x_4 = 1} \mu_{q_2 \to c_1}(x_2) \mu_{q_4 \to c_1}(x_4)$$

With uniform priors:
$$\mu_{c_1 \to q_1}(1) \propto 0.9 \times 0.9 + 0.1 \times 0.1 = 0.82$$
$$\mu_{c_1 \to q_1}(0) \propto 2 \times 0.9 \times 0.1 = 0.18$$

**(c)** Convergence issues:

The cycle of length 4 means messages from $$q_2$$ and $$q_4$$ return to themselves after 2 iterations, causing oscillation or overconfidence.

---

### Solution 20

**(a)** Short cycles cause failure:

BP assumes messages are independent, but in short cycles, messages are correlated through the cycle.

This leads to overconfident estimates and convergence to wrong answers.

**(b)** Minimum girth for optimality:

Girth $$g = \Omega(\log n)$$ ensures local tree-like structure.

For asymptotic optimality: $$g > 2\log_q(n)$$ where $$q$$ is average degree.

**(c)** Quantum vs classical LDPC:

Classical LDPC: Can achieve large girth through construction
Quantum codes: CSS constraint forces 4-cycles (X and Z checks share qubits)

Quantum codes inherently have short cycles, making BP harder.

---

### Solution 21

**(a)** OSD algorithm:

1. Run BP to get soft information (bit reliability)
2. Order bits by reliability
3. Use most reliable $$n-k$$ bits to define information set
4. Gaussian eliminate to solve for remaining bits
5. Try flipping up to $$k$$ least reliable bits and pick best

**(b)** OSD-$$k$$ complexity:

$$O\left(\binom{n}{k} \cdot n^2\right)$$ for order-$$k$$ OSD.

For $$k = O(1)$$: $$O(n^{k+2})$$, polynomial but can be large.

**(c)** OSD vs MWPM for surface codes:

OSD is designed for LDPC codes, not particularly optimized for surface codes.
MWPM exploits planar structure; OSD doesn't.

MWPM typically outperforms OSD for surface codes, but OSD is useful for QLDPC.

---

## Section 5: Neural Network Decoders

### Solution 22

**(a)** Input/Output:

Input: Syndrome bit string (length = number of checks, possibly with time history)
Output: Either error pattern or logical error prediction

**(b)** Training data generation:

1. Simulate noisy circuits with known errors
2. Collect (syndrome, error) pairs
3. Possibly augment with symmetries

**(c)** Loss function:

- Binary cross-entropy for bit-wise error prediction
- Cross-entropy for logical error classification
- Custom loss accounting for equivalence classes

---

### Solution 23

**(a)** Recurrent advantage:

Syndrome measurements are sequential in time.
RNN/LSTM can model temporal correlations and measurement errors naturally.
Memory allows combining information across time steps.

**(b)** Transformers and spatial structure:

Self-attention can learn arbitrary pairwise relationships.
Positional encoding can capture spatial layout.
Attention weights naturally model error correlations.

**(c)** Transformer inference complexity:

Self-attention: $$O(n^2)$$ for sequence length $$n$$.
With efficient attention: $$O(n \log n)$$ or $$O(n)$$ possible.

After training, inference is fast (matrix multiplications).

---

### Solution 24

**(a)** Training examples needed:

Typically $$10^6 - 10^8$$ examples for good generalization.
More for larger codes and complex noise.

**(b)** Covering rare errors:

1. Importance sampling: Oversample low-probability but high-impact errors
2. Curriculum learning: Start with common errors, gradually add rare ones
3. Data augmentation: Use code symmetries

**(c)** Real vs simulated training:

Real hardware:
- Limited data (experiments are slow)
- Contains actual noise
- May overfit to specific calibration

Simulation:
- Unlimited data
- Noise model may be inaccurate
- Transfer learning can bridge gap

---

### Solution 25

**(a)** Noise captured by AlphaQubit:

1. Correlated errors (crosstalk)
2. Leakage to non-computational states
3. Drift in qubit frequencies
4. Non-Markovian noise

**(b)** Verifying no cheating:

1. Hold-out test set never seen during training
2. Test on different time periods (after recalibration)
3. Verify on different code distances (generalization)
4. Compare to theoretical lower bounds

**(c)** Combining neural + MWPM:

1. Use neural network to estimate edge weights for MWPM
2. Use MWPM as a fast first guess, neural for refinement
3. Train neural network to predict MWPM failures and correct them

---

## Section 6: Decoder Selection

### Solution 26

**(a)** Distance needed:

$$p_L < 10^{-12}$$ at $$p = 0.5\%$$

$$A(0.005/0.01)^{(d+1)/2} < 10^{-12}$$ (rough threshold 1% for circuit-level)

$$0.5^{(d+1)/2} < 10^{-11}$$

$$(d+1)/2 > 36.5$$

$$d > 72$$

**Need distance ~73** (very large!)

More optimistically with threshold 0.5%:
Need $$d \approx 25-30$$ for well-below-threshold operation.

**(b)** Physical qubits:

$$n \approx 2d^2 \approx 2 \times 900 = 1800$$ per logical qubit (for $$d=30$$)

**(c)** Decoder choice:

**Union-find** for real-time capability, or **hybrid MWPM with parallelization**.

Justification: 1 MHz means 1 μs per decode. Union-find is achievable; MWPM requires parallelization.

**(d)** Real-time requirement:

Decode $$d^2 \approx 900$$ syndrome bits in $$< 1$$ μs.

Achievable with FPGA union-find implementation.

---

### Solution 27

**(a)** Decoder architecture:

1. **First stage:** Parallel union-find on each of 1000 logical qubits
2. **Second stage:** Neural network post-processing for correlated error correction
3. **Pipeline:** Overlap decode of round $$k$$ with measurement of round $$k+1$$

**(b)** Resources:

- FPGA: 1 per ~10 logical qubits for union-find
- GPU: 1 for neural post-processing
- Total: 100 FPGAs + 1 GPU cluster

**(c)** Validation:

1. Test on simulated noise with known correlations
2. Run on subset of hardware, compare to offline MWPM
3. Monte Carlo estimate of logical error rate
4. A/B test against baseline decoder

**(d)** Estimated logical error:

With $$d=21$$, $$p=0.5\%$$ (below threshold), 5% correlations:

$$p_L \approx 10^{-8}$$ to $$10^{-6}$$ per round, depending on correlation structure.

With 1000 logical qubits and $$10^6$$ rounds: need $$p_L < 10^{-9}$$ for high reliability.

May need larger distance or better correlation handling.
