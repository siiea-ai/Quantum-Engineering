# Day 675: Syndrome Decoding

## Week 97: Classical Error Correction Review | Month 25: QEC Fundamentals I

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Syndrome Decoding |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 675, you will be able to:

1. Define and compute the syndrome of a received word
2. Explain why syndrome decoding is information-preserving (critical for quantum!)
3. Construct syndrome lookup tables for error correction
4. Understand cosets and the standard array
5. Implement maximum-likelihood decoding via syndromes
6. Analyze the computational complexity of decoding

---

## Core Content

### 1. The Syndrome: Error Signature

**Definition:** The **syndrome** of a received word $\mathbf{r}$ is:

$$\boxed{\mathbf{s} = H \cdot \mathbf{r}^T}$$

where $H$ is the parity-check matrix.

**Key Properties:**
- If $\mathbf{s} = \mathbf{0}$: $\mathbf{r}$ is a valid codeword (or an undetectable error occurred)
- If $\mathbf{s} \neq \mathbf{0}$: an error has been detected

**The Fundamental Insight:**

Suppose we transmit codeword $\mathbf{c}$ and receive $\mathbf{r} = \mathbf{c} + \mathbf{e}$ where $\mathbf{e}$ is the error pattern.

Then:
$$\mathbf{s} = H \cdot \mathbf{r}^T = H \cdot (\mathbf{c} + \mathbf{e})^T = H\mathbf{c}^T + H\mathbf{e}^T = \mathbf{0} + H\mathbf{e}^T$$

$$\boxed{\mathbf{s} = H \cdot \mathbf{e}^T}$$

**The syndrome depends only on the error, not on the transmitted codeword!**

This is exactly what quantum error correction needs — we can identify errors without learning the encoded information.

---

### 2. Syndrome Decoding Algorithm

**Algorithm:**
```
Input: Received word r, parity-check matrix H
Output: Corrected codeword c_hat

1. Compute syndrome: s = H · r^T
2. If s = 0: return r (no error detected)
3. Look up s in syndrome table to find error pattern e
4. Return c_hat = r + e (correction)
```

**Complexity:**
- Syndrome computation: $O((n-k) \cdot n)$
- Table lookup: $O(1)$ with hash table, or $O(2^{n-k})$ with linear search
- Correction: $O(n)$

---

### 3. The Syndrome Lookup Table

For a code with $n - k$ parity bits, there are $2^{n-k}$ possible syndromes.

**Table Construction:**
For each possible error pattern $\mathbf{e}$:
1. Compute $\mathbf{s} = H \cdot \mathbf{e}^T$
2. Store the mapping $\mathbf{s} \to \mathbf{e}$

**Example: [7, 4, 3] Hamming Code**

Using $H$ with columns as binary representations of 1-7:

$$H = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix}$$

| Syndrome | Binary | Error Position | Error Pattern |
|----------|--------|----------------|---------------|
| (0,0,0) | 0 | None | 0000000 |
| (0,0,1) | 1 | Position 1 | 1000000 |
| (0,1,0) | 2 | Position 2 | 0100000 |
| (0,1,1) | 3 | Position 3 | 0010000 |
| (1,0,0) | 4 | Position 4 | 0001000 |
| (1,0,1) | 5 | Position 5 | 0000100 |
| (1,1,0) | 6 | Position 6 | 0000010 |
| (1,1,1) | 7 | Position 7 | 0000001 |

**Beautiful Property:** The syndrome IS the error position (in binary)!

This is the signature of a **perfect code** — every syndrome corresponds to exactly one correctable error.

---

### 4. Cosets and the Standard Array

**Definition:** The **coset** of a vector $\mathbf{a}$ with respect to code $C$ is:

$$\mathbf{a} + C = \{\mathbf{a} + \mathbf{c} : \mathbf{c} \in C\}$$

**Properties:**
- All vectors in a coset have the same syndrome
- The cosets partition $\mathbb{F}_2^n$ into $2^{n-k}$ disjoint sets
- Each coset contains $2^k$ vectors

**Coset Leader:** The minimum-weight vector in a coset. This represents the most likely error pattern for that syndrome.

**Standard Array:**

| Coset Leader | Codewords in this coset... |
|--------------|---------------------------|
| $\mathbf{0}$ | $\mathbf{c}_0, \mathbf{c}_1, \ldots, \mathbf{c}_{2^k-1}$ |
| $\mathbf{e}_1$ | $\mathbf{e}_1 + \mathbf{c}_0, \mathbf{e}_1 + \mathbf{c}_1, \ldots$ |
| $\mathbf{e}_2$ | $\mathbf{e}_2 + \mathbf{c}_0, \mathbf{e}_2 + \mathbf{c}_1, \ldots$ |
| $\vdots$ | $\vdots$ |

The first row contains all codewords. Each subsequent row is a coset with its leader in the first column.

**Decoding via Standard Array:**
1. Find the column containing $\mathbf{r}$
2. The codeword at the top of that column is the decoded codeword
3. Equivalently: subtract the coset leader from $\mathbf{r}$

---

### 5. Maximum Likelihood Decoding

**Principle:** Choose the codeword $\hat{\mathbf{c}}$ that maximizes the probability $P(\mathbf{r} | \mathbf{c})$.

For a binary symmetric channel (BSC) with error probability $p < 1/2$:

$$P(\mathbf{r} | \mathbf{c}) = p^{d_H(\mathbf{r}, \mathbf{c})} (1-p)^{n - d_H(\mathbf{r}, \mathbf{c})}$$

Maximizing this is equivalent to minimizing $d_H(\mathbf{r}, \mathbf{c})$.

**Equivalence:** Maximum likelihood decoding = minimum distance decoding = coset leader decoding.

**Theorem:** Syndrome decoding with coset leaders implements maximum likelihood decoding.

---

### 6. Bounded Distance Decoding

**Definition:** A decoder that corrects all errors of weight $\leq t$ and fails (or miscorrects) for errors of weight $> t$.

**Guarantee:** A code with minimum distance $d$ can correct all error patterns of weight $t = \lfloor (d-1)/2 \rfloor$.

**Bounded Distance Decoder:**
1. Compute syndrome $\mathbf{s}$
2. If $\mathbf{s}$ corresponds to an error of weight $\leq t$: correct
3. Otherwise: report decoding failure

**Trade-off:** Bounded distance decoders are simpler but may miss some correctable errors.

---

### 7. Quantum Connection: Non-Destructive Syndrome Measurement

**The Quantum Challenge:**
- In quantum mechanics, measuring a state changes it
- We can't directly examine the encoded quantum information

**The Quantum Solution:**
- Measure the syndrome without measuring the data
- Syndrome measurement projects onto an error subspace
- The encoded logical information remains protected

**How It Works:**
| Classical | Quantum |
|-----------|---------|
| Compute $\mathbf{s} = H\mathbf{r}^T$ | Measure stabilizer generators |
| Syndrome reveals error | Syndrome eigenvalue reveals error |
| Data not accessed | Logical state not disturbed |

This is why studying classical syndrome decoding is essential — the same concepts transfer directly to quantum!

---

## Physical Interpretation

### Why Syndromes Work

Think of a syndrome as an "error fingerprint":
- Each error pattern leaves a unique mark
- The mark (syndrome) identifies the error
- Importantly, the mark doesn't reveal the original message

### Geometric Picture

In the vector space $\mathbb{F}_2^n$:
- The code $C$ is a $k$-dimensional subspace
- Cosets are parallel copies of $C$
- Each coset has a syndrome label
- Decoding = finding which coset you're in, then projecting to $C$

---

## Worked Examples

### Example 1: Complete Syndrome Decoding

**Problem:** Using the [7, 4, 3] Hamming code, decode the received word $\mathbf{r} = (1, 0, 1, 1, 0, 1, 0)$.

**Solution:**

$$H = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix}$$

**Step 1: Compute syndrome**

$$\mathbf{s} = H \cdot (1, 0, 1, 1, 0, 1, 0)^T$$

Row 1: $0·1 + 0·0 + 0·1 + 1·1 + 1·0 + 1·1 + 1·0 = 1 + 1 = 0$
Row 2: $0·1 + 1·0 + 1·1 + 0·1 + 0·0 + 1·1 + 1·0 = 1 + 1 = 0$
Row 3: $1·1 + 0·0 + 1·1 + 0·1 + 1·0 + 0·1 + 1·0 = 1 + 1 = 0$

$\mathbf{s} = (0, 0, 0)$

**Step 2: Interpret syndrome**
Syndrome $(0,0,0)$ means no error detected!

**Step 3: Decode**
$\hat{\mathbf{c}} = \mathbf{r} = (1, 0, 1, 1, 0, 1, 0)$

**Verification:** The first 4 bits give message $(1, 0, 1, 1)$. ∎

---

### Example 2: Error Correction

**Problem:** Decode $\mathbf{r} = (1, 0, 1, 1, 1, 1, 0)$ (note: position 5 flipped from Example 1).

**Solution:**

**Step 1: Compute syndrome**

Row 1: $0·1 + 0·0 + 0·1 + 1·1 + 1·1 + 1·1 + 1·0 = 1 + 1 + 1 = 1$
Row 2: $0·1 + 1·0 + 1·1 + 0·1 + 0·1 + 1·1 + 1·0 = 1 + 1 = 0$
Row 3: $1·1 + 0·0 + 1·1 + 0·1 + 1·1 + 0·1 + 1·0 = 1 + 1 + 1 = 1$

$\mathbf{s} = (1, 0, 1)$

**Step 2: Interpret syndrome**
Binary $(1, 0, 1) = 5$. Error is in position 5.

**Step 3: Correct**
$$\hat{\mathbf{c}} = \mathbf{r} \oplus (0,0,0,0,1,0,0) = (1, 0, 1, 1, 0, 1, 0)$$

Same as Example 1! The error was successfully corrected. ∎

---

### Example 3: Building a Syndrome Table

**Problem:** Construct the complete syndrome table for the [5, 2, 3] code with:

$$H = \begin{pmatrix} 1 & 0 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 & 1 \\ 1 & 1 & 0 & 1 & 1 \end{pmatrix}$$

**Solution:**

For a [5, 2, 3] code, we have $2^3 = 8$ syndromes.

Single-bit errors and their syndromes:

| Error $\mathbf{e}$ | $H\mathbf{e}^T$ = Syndrome |
|-------|----------|
| 10000 | $(1, 0, 1)$ |
| 01000 | $(0, 1, 1)$ |
| 00100 | $(1, 1, 0)$ |
| 00010 | $(1, 0, 1)$ — collision! |
| 00001 | $(0, 1, 1)$ — collision! |

Wait, we have collisions. Let me recalculate.

For error at position 1 (10000):
$H·(1,0,0,0,0)^T = $ column 1 of $H = (1, 0, 1)^T$

For position 4 (00010):
$H·(0,0,0,1,0)^T = $ column 4 of $H = (1, 0, 1)^T$

Columns 1 and 4 are the same! This means errors in positions 1 and 4 produce the same syndrome — the code cannot distinguish them.

**Observation:** This code has $d = 3$ but isn't a perfect code. Some syndromes correspond to two different single errors. We need a tie-breaking rule (usually: choose lexicographically first).

**Complete Table:**

| Syndrome | Coset Leader (Error) |
|----------|---------------------|
| (0,0,0) | 00000 |
| (1,0,1) | 10000 (or 00010) |
| (0,1,1) | 01000 (or 00001) |
| (1,1,0) | 00100 |
| (0,1,0) | — (no single error) |
| (1,0,0) | — (no single error) |
| (1,1,1) | — (no single error) |
| (0,0,1) | — (no single error) |

For syndromes not corresponding to single errors, we'd need to find weight-2 error patterns. ∎

---

## Practice Problems

### Level 1: Direct Application

1. For the [7, 4, 3] Hamming code, compute the syndrome of $\mathbf{r} = (0, 1, 1, 0, 1, 0, 1)$ and decode.

2. Construct the syndrome table for the [6, 3, 3] code with:
$$H = \begin{pmatrix} 1 & 1 & 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 & 0 & 1 \end{pmatrix}$$

3. Show that for the Hamming code, the syndrome $(s_1, s_2, s_3)$ interpreted as a binary number gives the error position directly.

### Level 2: Intermediate

4. Prove that two vectors have the same syndrome if and only if they are in the same coset.

5. For a [15, 11, 3] Hamming code, how many entries does the syndrome table have? How many correspond to correctable single errors?

6. Design an algorithm to find coset leaders by exhaustive search.

### Level 3: Challenging

7. Prove that for a perfect code, syndrome decoding achieves maximum likelihood decoding with equality (every syndrome has a unique minimum-weight error).

8. Analyze the probability of decoding failure for the [7, 4, 3] Hamming code on a BSC with error probability $p$.

9. **Research:** In quantum error correction, why must syndrome measurement be done without disturbing the encoded state? How do stabilizer measurements achieve this?

---

## Computational Lab

### Objective
Implement complete syndrome decoding and analyze error correction performance.

```python
"""
Day 675 Computational Lab: Syndrome Decoding
Year 2: Advanced Quantum Science
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Syndrome Computation and Decoding
# =============================================================================

print("=" * 60)
print("Part 1: Hamming [7,4,3] Syndrome Decoding")
print("=" * 60)

# Hamming code H with columns as binary 1-7
H = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1]
], dtype=int)

G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=int)

def compute_syndrome(r, H):
    """Compute syndrome s = H * r^T (mod 2)."""
    return np.mod(H @ r, 2)

def syndrome_to_position(s):
    """Convert syndrome to error position (Hamming code specific)."""
    # Syndrome as binary number
    pos = s[0] * 4 + s[1] * 2 + s[2]
    return pos  # 0 means no error, 1-7 means position 1-7

def hamming_decode(r, H):
    """Decode using Hamming code syndrome decoding."""
    r = np.array(r, dtype=int)
    s = compute_syndrome(r, H)
    pos = syndrome_to_position(s)

    if pos == 0:
        return r, s, None  # No error
    else:
        # Correct error at position pos (1-indexed)
        corrected = r.copy()
        corrected[pos - 1] = 1 - corrected[pos - 1]
        return corrected, s, pos

# Test decoding
print("\nTest 1: No error")
c = np.array([1, 0, 1, 1, 0, 1, 0])
decoded, syndrome, err_pos = hamming_decode(c, H)
print(f"  Received: {c}")
print(f"  Syndrome: {syndrome}")
print(f"  Error position: {err_pos}")
print(f"  Decoded: {decoded}")

print("\nTest 2: Error in position 5")
r = np.array([1, 0, 1, 1, 1, 1, 0])  # Bit 5 flipped
decoded, syndrome, err_pos = hamming_decode(r, H)
print(f"  Received: {r}")
print(f"  Syndrome: {syndrome} = binary {syndrome_to_position(syndrome)}")
print(f"  Error position: {err_pos}")
print(f"  Decoded: {decoded}")

# =============================================================================
# Part 2: Complete Syndrome Table
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Complete Syndrome Table")
print("=" * 60)

def build_syndrome_table(H, max_weight=1):
    """Build syndrome -> error mapping for errors up to max_weight."""
    n = H.shape[1]
    table = {}

    # Zero error
    zero_error = tuple([0] * n)
    zero_syndrome = tuple(compute_syndrome(np.zeros(n, dtype=int), H))
    table[zero_syndrome] = zero_error

    # Single errors
    for i in range(n):
        error = np.zeros(n, dtype=int)
        error[i] = 1
        syndrome = tuple(compute_syndrome(error, H))
        if syndrome not in table:
            table[syndrome] = tuple(error)

    return table

syndrome_table = build_syndrome_table(H)
print("\nSyndrome Table for Hamming [7,4,3]:")
print("-" * 50)
print(f"{'Syndrome':<15} {'Error Pattern':<25} {'Error Pos'}")
print("-" * 50)
for syndrome, error in sorted(syndrome_table.items()):
    pos = sum(i+1 for i, e in enumerate(error) if e == 1)
    pos_str = str(pos) if pos > 0 else "None"
    print(f"{str(syndrome):<15} {str(error):<25} {pos_str}")

# =============================================================================
# Part 3: Coset Structure
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Coset Structure Analysis")
print("=" * 60)

def generate_all_codewords(G):
    """Generate all codewords from generator matrix."""
    k = G.shape[0]
    codewords = []
    for m in product([0, 1], repeat=k):
        c = np.mod(np.array(m) @ G, 2)
        codewords.append(tuple(c))
    return codewords

def find_coset(v, codewords):
    """Find the coset containing vector v."""
    coset = set()
    for c in codewords:
        coset.add(tuple(np.mod(np.array(v) + np.array(c), 2)))
    return coset

codewords = generate_all_codewords(G)
print(f"\nNumber of codewords: {len(codewords)}")

# Find coset leaders for each syndrome
print("\nCoset leaders and their syndromes:")
all_vectors = list(product([0, 1], repeat=7))

# Group by syndrome
syndrome_groups = {}
for v in all_vectors:
    s = tuple(compute_syndrome(np.array(v), H))
    if s not in syndrome_groups:
        syndrome_groups[s] = []
    syndrome_groups[s].append(v)

print(f"\nNumber of cosets: {len(syndrome_groups)}")
print("\nCoset analysis:")
for syndrome, vectors in sorted(syndrome_groups.items()):
    # Find minimum weight vector (coset leader)
    min_weight = min(sum(v) for v in vectors)
    leaders = [v for v in vectors if sum(v) == min_weight]
    print(f"  Syndrome {syndrome}: {len(vectors)} vectors, "
          f"leader weight {min_weight}, {len(leaders)} leader(s)")

# =============================================================================
# Part 4: Error Correction Performance
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Error Correction Performance Simulation")
print("=" * 60)

def simulate_bsc(codeword, p):
    """Simulate binary symmetric channel with error probability p."""
    n = len(codeword)
    errors = np.random.random(n) < p
    return np.mod(codeword + errors.astype(int), 2)

def test_decoder(G, H, num_trials=10000, p=0.05):
    """Test decoder performance on BSC."""
    codewords = generate_all_codewords(G)
    codewords = [np.array(c) for c in codewords]

    successes = 0
    failures = 0
    undetected = 0

    for _ in range(num_trials):
        # Random codeword
        c = codewords[np.random.randint(len(codewords))]

        # Transmit through BSC
        r = simulate_bsc(c, p)

        # Decode
        decoded, syndrome, _ = hamming_decode(r, H)

        # Check result
        if np.array_equal(decoded, c):
            successes += 1
        elif np.array_equal(decoded, r) and not np.array_equal(r, c):
            undetected += 1  # No error detected but there was one
        else:
            failures += 1

    return {
        'success_rate': successes / num_trials,
        'failure_rate': failures / num_trials,
        'undetected_rate': undetected / num_trials
    }

print("\nSimulating BSC with various error probabilities...")
error_probs = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
results = []

for p in error_probs:
    result = test_decoder(G, H, num_trials=10000, p=p)
    results.append(result)
    print(f"  p = {p:.2f}: Success {result['success_rate']:.4f}, "
          f"Failure {result['failure_rate']:.4f}, "
          f"Undetected {result['undetected_rate']:.4f}")

# =============================================================================
# Part 5: Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Performance Visualization")
print("=" * 60)

# Theoretical analysis
def theoretical_error_rate(p, n=7, t=1):
    """Theoretical word error rate for t-error correcting code."""
    from math import comb

    # Probability of 0 or 1 errors (correctable)
    p_correct = sum(comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(t+1))
    return 1 - p_correct

p_values = np.linspace(0.001, 0.25, 100)
theoretical = [theoretical_error_rate(p) for p in p_values]

# Uncoded error rate (probability of any error in 4 bits)
uncoded = [1 - (1-p)**4 for p in p_values]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot theoretical curves
ax.semilogy(p_values, theoretical, 'b-', linewidth=2, label='Hamming [7,4,3] (theory)')
ax.semilogy(p_values, uncoded, 'r--', linewidth=2, label='Uncoded (4 bits)')

# Plot simulation points
sim_p = error_probs
sim_error = [1 - r['success_rate'] for r in results]
ax.semilogy(sim_p, sim_error, 'bo', markersize=8, label='Simulation')

ax.set_xlabel('Physical Error Probability p', fontsize=12)
ax.set_ylabel('Word Error Rate (log scale)', fontsize=12)
ax.set_title('Hamming Code Error Correction Performance', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.25])
ax.set_ylim([1e-4, 1])

plt.tight_layout()
plt.savefig('day_675_error_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_675_error_performance.png'")

# =============================================================================
# Part 6: Syndrome Pattern Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Syndrome Space Visualization")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Syndrome distribution for single errors
ax1 = axes[0]
syndromes_int = []
for i in range(7):
    e = np.zeros(7, dtype=int)
    e[i] = 1
    s = compute_syndrome(e, H)
    syndromes_int.append(s[0]*4 + s[1]*2 + s[2])

ax1.bar(range(7), syndromes_int, color='steelblue', alpha=0.7)
ax1.set_xlabel('Error Position', fontsize=12)
ax1.set_ylabel('Syndrome (as integer)', fontsize=12)
ax1.set_title('Syndrome for Single-Bit Errors', fontsize=12)
ax1.set_xticks(range(7))
ax1.set_xticklabels([f'Pos {i+1}' for i in range(7)], rotation=45)

# Right: Coset sizes
ax2 = axes[1]
coset_sizes = [len(v) for v in syndrome_groups.values()]
ax2.bar(range(len(coset_sizes)), coset_sizes, color='coral', alpha=0.7)
ax2.set_xlabel('Coset Index', fontsize=12)
ax2.set_ylabel('Coset Size', fontsize=12)
ax2.set_title('Coset Sizes (all equal for linear codes)', fontsize=12)
ax2.axhline(y=2**4, color='k', linestyle='--', label=f'Expected: 2^k = {2**4}')
ax2.legend()

plt.tight_layout()
plt.savefig('day_675_syndrome_structure.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as 'day_675_syndrome_structure.png'")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Syndrome | $\mathbf{s} = H \cdot \mathbf{r}^T$ |
| Error relationship | $\mathbf{s} = H \cdot \mathbf{e}^T$ |
| Syndrome count | $2^{n-k}$ possible syndromes |
| Correction | $\hat{\mathbf{c}} = \mathbf{r} + \mathbf{e}$ |
| Coset | $\mathbf{a} + C = \{\mathbf{a} + \mathbf{c} : \mathbf{c} \in C\}$ |

### Main Takeaways

1. **Syndrome depends only on error** — not on the transmitted codeword
2. **Syndrome decoding is non-destructive** — this is critical for quantum EC
3. **Perfect codes** have unique syndrome-error correspondence
4. **Cosets partition the space** — same syndrome = same coset
5. **ML decoding = coset leader decoding** for BSC

---

## Daily Checklist

- [ ] Compute syndromes for several received words
- [ ] Understand why syndrome reveals error but not data
- [ ] Build a syndrome table for a small code
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and analyze the error performance simulation
- [ ] Connect syndrome decoding to quantum stabilizer measurements

---

## Preview: Day 676

Tomorrow we dive deep into **Hamming codes** — the first family of perfect codes. We'll prove their optimality, analyze their structure, and see how they generalize to larger parameters. Hamming codes are the classical inspiration for the quantum Steane code.

---

*"The syndrome tells us what went wrong without revealing what was right."*
— The essence of error correction

---

**Next:** [Day_676_Thursday.md](Day_676_Thursday.md) — Hamming Codes Deep Dive
