# Day 676: Hamming Codes Deep Dive

## Week 97: Classical Error Correction Review | Month 25: QEC Fundamentals I

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Hamming Code Family |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 676, you will be able to:

1. Construct Hamming codes for any parameter $r \geq 2$
2. Prove that Hamming codes are perfect
3. Analyze the extended Hamming code and its properties
4. Implement efficient Hamming encoding and decoding
5. Connect Hamming codes to the quantum Steane code
6. Understand why Hamming codes are optimal single-error-correcting codes

---

## Core Content

### 1. The Hamming Code Family

**Definition:** For any integer $r \geq 2$, the binary **Hamming code** has parameters:

$$\boxed{[n, k, d] = [2^r - 1, \; 2^r - 1 - r, \; 3]}$$

| $r$ | $n$ | $k$ | Code |
|-----|-----|-----|------|
| 2 | 3 | 1 | [3, 1, 3] (repetition) |
| 3 | 7 | 4 | [7, 4, 3] |
| 4 | 15 | 11 | [15, 11, 3] |
| 5 | 31 | 26 | [31, 26, 3] |
| 6 | 63 | 57 | [63, 57, 3] |

**Construction:** The parity-check matrix $H$ has columns consisting of all $2^r - 1$ nonzero binary vectors of length $r$.

---

### 2. Perfect Codes

**Definition:** A code is **perfect** if it achieves the Hamming bound with equality.

**Hamming Bound:** For a code correcting $t$ errors:
$$|C| \cdot \sum_{i=0}^{t} \binom{n}{i} \leq 2^n$$

For $t = 1$ (single error correction):
$$|C| \cdot (1 + n) \leq 2^n$$

**Theorem:** Hamming codes are perfect.

**Proof:**
For the $[2^r - 1, 2^r - 1 - r, 3]$ Hamming code:
- $|C| = 2^k = 2^{2^r - 1 - r}$
- $n = 2^r - 1$
- $t = 1$

Check: $|C| \cdot (1 + n) = 2^{2^r - 1 - r} \cdot 2^r = 2^{2^r - 1} = 2^n$ ✓

The Hamming spheres of radius 1 exactly tile $\mathbb{F}_2^n$. ∎

**Implication:** Every syndrome corresponds to exactly one correctable error pattern.

---

### 3. The [7, 4, 3] Hamming Code in Detail

**Parity-Check Matrix** (systematic form):
$$H = \begin{pmatrix} 1 & 1 & 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 1 & 0 & 0 & 1 \end{pmatrix}$$

**Generator Matrix** (systematic form):
$$G = \begin{pmatrix} 1 & 0 & 0 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 1 & 1 & 1 & 1 \end{pmatrix}$$

**Alternative H** (columns = binary 1-7):
$$H_{alt} = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix}$$

With this form, the syndrome directly encodes the error position in binary!

**Weight Distribution:**

| Weight | Count |
|--------|-------|
| 0 | 1 |
| 3 | 7 |
| 4 | 7 |
| 7 | 1 |
| **Total** | **16** |

---

### 4. Extended Hamming Code

**Construction:** Add an overall parity bit to create $[n+1, k, d+1]$ code.

The **extended Hamming code** has parameters $[2^r, 2^r - 1 - r, 4]$.

**Example: [8, 4, 4] Extended Hamming Code**

$$H_{ext} = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{pmatrix}$$

The last row checks overall parity.

**Properties:**
- All codewords have even weight
- Can detect all patterns of 1, 2, or 3 errors
- Can correct all single-error patterns
- Can distinguish single errors from double errors

**Decoding Algorithm:**
1. Compute syndrome $s$ (first 3 bits) and parity check $p$ (last bit)
2. If $s = 0$ and $p = 0$: No error
3. If $s = 0$ and $p = 1$: Error in parity bit (position 8)
4. If $s \neq 0$ and $p = 1$: Single error at position $s$ (as binary)
5. If $s \neq 0$ and $p = 0$: Double error detected (uncorrectable)

---

### 5. The Dual of the Hamming Code

**Theorem:** The dual of the $[2^r - 1, 2^r - 1 - r, 3]$ Hamming code is the $[2^r - 1, r, 2^{r-1}]$ simplex code.

**Example:** Dual of [7, 4, 3] is [7, 3, 4]

The simplex code has:
- All nonzero codewords have the same weight $2^{r-1}$
- It's a constant-weight code (equidistant)

**Dual Relationship Table:**

| Hamming $[n, k, d]$ | Simplex $[n, n-k, d']$ |
|---------------------|------------------------|
| [3, 1, 3] | [3, 2, 2] |
| [7, 4, 3] | [7, 3, 4] |
| [15, 11, 3] | [15, 4, 8] |
| [31, 26, 3] | [31, 5, 16] |

---

### 6. Hamming Code as a Cyclic Code

**Theorem:** The $[2^r - 1, 2^r - 1 - r, 3]$ Hamming code is cyclic.

**Generator Polynomial:** The generator polynomial $g(x)$ is a primitive polynomial of degree $r$ over $\mathbb{F}_2$.

**Example: [7, 4, 3] Code**

Primitive polynomial: $g(x) = x^3 + x + 1$

This generates the Hamming code as a cyclic code!

**Cyclic Property:** If $(c_0, c_1, \ldots, c_6)$ is a codeword, so is $(c_6, c_0, c_1, \ldots, c_5)$.

---

### 7. Connection to Quantum: The Steane Code

**The Steane Code** $[[7, 1, 3]]$ is a quantum code constructed from the classical [7, 4, 3] Hamming code.

**Key Observation:** The Hamming code is **self-orthogonal** in the sense needed for CSS codes:
$$C^\perp \subset C$$

The dual [7, 3, 4] code is contained in the [7, 4, 3] Hamming code!

**Construction:**
- Use $C_X = C_Z = $ Hamming [7, 4, 3]
- CSS condition: $C^\perp \subseteq C$ ✓
- Quantum parameters: $[[7, k_X + k_Z - n, d]] = [[7, 4 + 4 - 7, 3]] = [[7, 1, 3]]$

**Stabilizer Generators:**
- X-type: rows of $H$ applied as X operators
- Z-type: rows of $H$ applied as Z operators

The Steane code inherits the beautiful syndrome decoding of the Hamming code!

---

## Physical Interpretation

### Why Hamming Codes Are Optimal

For single-error correction:
1. You need to distinguish $n+1$ cases (no error, or error in position 1 through $n$)
2. With $r$ parity bits, you can distinguish $2^r$ cases
3. Optimal: $2^r = n + 1$, giving $n = 2^r - 1$

This is exactly the Hamming code!

### The Sphere-Packing Picture

Imagine $\mathbb{F}_2^n$ as a space with $2^n$ points. Around each codeword, draw a "sphere" of radius 1 containing the codeword and all words at distance 1.

- Sphere size: $1 + n$ (center + $n$ neighbors)
- Number of spheres: $2^k$ (one per codeword)
- Total points covered: $2^k \cdot (1 + n)$

For Hamming codes: $2^k \cdot (1 + n) = 2^n$ — **perfect tiling!**

---

## Worked Examples

### Example 1: Constructing [15, 11, 3] Hamming Code

**Problem:** Construct the parity-check matrix for the [15, 11, 3] Hamming code.

**Solution:**

For $r = 4$: $n = 2^4 - 1 = 15$, $k = 15 - 4 = 11$.

$H$ has 4 rows and 15 columns. Columns are all nonzero 4-bit binary vectors:

$$H = \begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
0 & 0 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 1
\end{pmatrix}$$

**Verification:**
- Columns are binary representations of 1 through 15
- Any 2 columns are distinct, so minimum distance $d \geq 3$
- Columns 1, 2, 3 sum to zero: $(0,0,0,1)^T + (0,0,1,0)^T + (0,0,1,1)^T = (0,0,0,0)^T$

Wait, that's wrong. Let me recalculate.

Column 1: $(0,0,0,1)^T$
Column 2: $(0,0,1,0)^T$
Column 3: $(0,0,1,1)^T$

Sum: $(0, 0, 0+1+1, 1+0+1)^T = (0, 0, 0, 0)^T$ ✓

So $d = 3$ as expected. ∎

---

### Example 2: Extended Hamming Decoding

**Problem:** Decode the received word $\mathbf{r} = (1, 0, 1, 1, 0, 1, 1, 1)$ using the [8, 4, 4] extended Hamming code.

**Solution:**

**Step 1: Compute regular syndrome (first 3 parity checks)**

Using $H_{7,4}$ on first 7 bits: $\mathbf{r}_{1:7} = (1, 0, 1, 1, 0, 1, 1)$

Syndrome: $s = (s_1, s_2, s_3)$ where $s_i$ is $i$-th row of $H$ dotted with $\mathbf{r}_{1:7}$.

$s_1 = 1 + 0 + 0 + 1 + 0 + 1 + 1 = 0$ (mod 2)
$s_2 = 1 + 0 + 1 + 1 + 0 + 0 + 1 = 0$ (mod 2)
$s_3 = 0 + 0 + 1 + 1 + 0 + 1 + 1 = 0$ (mod 2)

$s = (0, 0, 0)$

**Step 2: Compute overall parity**
$p = 1 + 0 + 1 + 1 + 0 + 1 + 1 + 1 = 0$ (mod 2)

**Step 3: Interpret**
- $s = (0, 0, 0)$ and $p = 0$: No error!

The received word is a valid codeword. Message = first 4 bits = $(1, 0, 1, 1)$. ∎

---

### Example 3: Verifying Self-Orthogonality

**Problem:** Verify that the dual of the [7, 4, 3] Hamming code is contained in the code.

**Solution:**

The dual code has generator matrix $G^\perp = H$:
$$G^\perp = \begin{pmatrix} 1 & 1 & 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 1 & 0 & 0 & 1 \end{pmatrix}$$

We need to show each row of $G^\perp$ is a codeword of the Hamming code.

**Row 1:** $(1, 1, 0, 1, 1, 0, 0)$
Check: $H \cdot (1, 1, 0, 1, 1, 0, 0)^T = ?$

Using the systematic $H$:
$H_1 = 1 + 1 + 0 + 1 + 1 + 0 + 0 = 0$ ✓
$H_2 = 1 + 0 + 0 + 1 + 0 + 0 + 0 = 0$ ✓
$H_3 = 0 + 1 + 0 + 1 + 0 + 0 + 0 = 0$ ✓

Row 1 is a codeword. Similarly verify rows 2 and 3.

Since all rows of $G^\perp$ (generators of dual) are codewords, we have $C^\perp \subseteq C$. ∎

---

## Practice Problems

### Level 1: Direct Application

1. Construct the parity-check matrix for the [31, 26, 3] Hamming code (just describe the structure, don't write all 31 columns).

2. Encode the message $(1, 1, 0, 1)$ using the [7, 4, 3] Hamming code.

3. For the extended [8, 4, 4] code, decode: $\mathbf{r} = (0, 1, 1, 0, 1, 0, 0, 1)$.

### Level 2: Intermediate

4. Prove that all nonzero codewords of the $[2^r - 1, r, 2^{r-1}]$ simplex code have weight exactly $2^{r-1}$.

5. Show that the [7, 4, 3] Hamming code can be generated by the polynomial $g(x) = x^3 + x + 1$.

6. Calculate the weight distribution of the [15, 11, 3] Hamming code.

### Level 3: Challenging

7. Prove that the only binary perfect codes are: repetition codes, Hamming codes, and the [23, 12, 7] Golay code.

8. Analyze the probability that the extended Hamming code miscorrects a double error as a single error.

9. **Research:** How does the Steane code's error correction inherit properties from the Hamming code's syndrome decoding?

---

## Computational Lab

### Objective
Implement complete Hamming code family and analyze their properties.

```python
"""
Day 676 Computational Lab: Hamming Codes Deep Dive
Year 2: Advanced Quantum Science
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Hamming Code Construction
# =============================================================================

print("=" * 60)
print("Part 1: Hamming Code Family Construction")
print("=" * 60)

def construct_hamming_H(r):
    """
    Construct parity-check matrix for [2^r-1, 2^r-1-r, 3] Hamming code.
    Columns are binary representations of 1 to 2^r-1.
    """
    n = 2**r - 1
    H = np.zeros((r, n), dtype=int)

    for i in range(n):
        # Binary representation of i+1
        num = i + 1
        for j in range(r):
            H[r - 1 - j, i] = num % 2
            num //= 2

    return H

def construct_hamming_G(H):
    """Construct systematic generator matrix from H."""
    r, n = H.shape
    k = n - r

    # Rearrange H to systematic form [P^T | I_r]
    # Then G = [I_k | P]

    # Find columns that form identity
    identity_cols = []
    for i in range(r):
        for j in range(n):
            col = H[:, j]
            if col[i] == 1 and sum(col) == 1:
                identity_cols.append(j)
                break

    # Remaining columns are data columns
    data_cols = [j for j in range(n) if j not in identity_cols]

    # Reorder H
    col_order = data_cols + identity_cols
    H_sys = H[:, col_order]

    # Extract P^T (first k columns of reordered H)
    P_T = H_sys[:, :k]
    P = P_T.T

    # G = [I_k | P]
    G = np.hstack([np.eye(k, dtype=int), P])

    return G, col_order

# Construct Hamming codes for r = 2, 3, 4, 5
print("\nHamming Code Family:")
print("-" * 50)
print(f"{'r':<5} {'n':<8} {'k':<8} {'Rate':<10} {'Redundancy'}")
print("-" * 50)

for r in range(2, 6):
    n = 2**r - 1
    k = n - r
    rate = k / n
    print(f"{r:<5} {n:<8} {k:<8} {rate:<10.4f} {r} bits")

# =============================================================================
# Part 2: [7, 4, 3] Hamming Code Analysis
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: [7, 4, 3] Hamming Code Analysis")
print("=" * 60)

H_7 = construct_hamming_H(3)
G_7, col_order = construct_hamming_G(H_7)

print("\nParity-check matrix H (columns = binary 1-7):")
print(H_7)

print("\nColumn ordering for systematic form:", col_order)

print("\nGenerator matrix G (systematic form):")
print(G_7)

# Generate all codewords and analyze
print("\nAll 16 codewords:")
print("-" * 40)

codewords = []
weights = []
for m in range(16):
    message = np.array([(m >> i) & 1 for i in range(4)], dtype=int)
    codeword = np.mod(message @ G_7, 2)
    weight = np.sum(codeword)
    codewords.append(codeword)
    weights.append(weight)
    print(f"m={m:2d}: {message} -> {codeword}  (wt {weight})")

print("\nWeight distribution:")
for w in sorted(set(weights)):
    count = weights.count(w)
    print(f"  Weight {w}: {count} codewords")

# =============================================================================
# Part 3: Perfect Code Verification
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Perfect Code Verification")
print("=" * 60)

def verify_perfect(r):
    """Verify Hamming bound equality for Hamming code."""
    n = 2**r - 1
    k = n - r
    num_codewords = 2**k
    sphere_size = 1 + n  # |B(0,1)| for t=1

    total_coverage = num_codewords * sphere_size
    space_size = 2**n

    return {
        'n': n, 'k': k,
        'codewords': num_codewords,
        'sphere_size': sphere_size,
        'coverage': total_coverage,
        'space': space_size,
        'perfect': total_coverage == space_size
    }

print("\nHamming Bound Verification:")
print("-" * 60)
for r in range(2, 6):
    result = verify_perfect(r)
    status = "✓ Perfect" if result['perfect'] else "✗ Not perfect"
    print(f"r={r}: {result['codewords']} × {result['sphere_size']} = "
          f"{result['coverage']} vs 2^{result['n']} = {result['space']}  {status}")

# =============================================================================
# Part 4: Extended Hamming Code
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Extended [8, 4, 4] Hamming Code")
print("=" * 60)

def extend_hamming(G):
    """Add overall parity bit to create extended code."""
    k, n = G.shape
    # Add parity column
    parity_col = np.sum(G, axis=1, keepdims=True) % 2
    G_ext = np.hstack([G, parity_col])
    return G_ext

G_8 = extend_hamming(G_7)
print("\nExtended generator matrix G_ext:")
print(G_8)

# Generate extended codewords
print("\nExtended codewords:")
ext_weights = []
for m in range(16):
    message = np.array([(m >> i) & 1 for i in range(4)], dtype=int)
    codeword = np.mod(message @ G_8, 2)
    weight = np.sum(codeword)
    ext_weights.append(weight)
    print(f"m={m:2d}: {codeword}  (wt {weight})")

print("\nExtended weight distribution:")
for w in sorted(set(ext_weights)):
    count = ext_weights.count(w)
    print(f"  Weight {w}: {count} codewords")

print("\nAll weights even:", all(w % 2 == 0 for w in ext_weights))
print("Minimum nonzero weight (= distance):", min(w for w in ext_weights if w > 0))

# =============================================================================
# Part 5: Dual Code (Simplex Code)
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Dual Code - [7, 3, 4] Simplex Code")
print("=" * 60)

# Generator of dual = H of original
G_dual = H_7.copy()

print("\nSimplex code generator (= Hamming H):")
print(G_dual)

print("\nSimplex codewords:")
simplex_weights = []
for m in range(8):
    message = np.array([(m >> i) & 1 for i in range(3)], dtype=int)
    codeword = np.mod(message @ G_dual, 2)
    weight = np.sum(codeword)
    simplex_weights.append(weight)
    print(f"m={m}: {message} -> {codeword}  (wt {weight})")

print("\nSimplex code is equidistant:", len(set(w for w in simplex_weights if w > 0)) == 1)
print("Constant weight:", simplex_weights[1])

# =============================================================================
# Part 6: Self-Orthogonality Check
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Self-Orthogonality (C^⊥ ⊆ C)")
print("=" * 60)

def is_in_code(vector, G):
    """Check if vector is in the code generated by G."""
    k, n = G.shape
    # Try all messages
    for m in range(2**k):
        message = np.array([(m >> i) & 1 for i in range(k)], dtype=int)
        codeword = np.mod(message @ G, 2)
        if np.array_equal(codeword, vector):
            return True
    return False

print("\nChecking if dual codewords are in Hamming code:")
for m in range(8):
    message = np.array([(m >> i) & 1 for i in range(3)], dtype=int)
    dual_codeword = np.mod(message @ G_dual, 2)
    in_hamming = is_in_code(dual_codeword, G_7)
    status = "✓" if in_hamming else "✗"
    print(f"  {dual_codeword} in Hamming code: {status}")

print("\nC^⊥ ⊆ C: This enables CSS quantum code construction!")

# =============================================================================
# Part 7: Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Part 7: Visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Weight distribution comparison
ax1 = axes[0, 0]
hamming_dist = [weights.count(w) for w in range(8)]
extended_dist = [ext_weights.count(w) for w in range(9)]
x = np.arange(9)
width = 0.35
ax1.bar(x[:8] - width/2, hamming_dist, width, label='[7,4,3] Hamming', alpha=0.7)
ax1.bar(x - width/2 + width, extended_dist, width, label='[8,4,4] Extended', alpha=0.7)
ax1.set_xlabel('Weight')
ax1.set_ylabel('Count')
ax1.set_title('Weight Distributions')
ax1.legend()
ax1.set_xticks(range(9))

# Rate vs redundancy for Hamming family
ax2 = axes[0, 1]
r_values = range(2, 10)
rates = [(2**r - 1 - r) / (2**r - 1) for r in r_values]
ax2.plot(r_values, rates, 'bo-', linewidth=2, markersize=8)
ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('Parameter r')
ax2.set_ylabel('Code Rate k/n')
ax2.set_title('Hamming Code Rate vs Parameter')
ax2.grid(True, alpha=0.3)

# Syndrome structure
ax3 = axes[1, 0]
H_vis = construct_hamming_H(3)
im = ax3.imshow(H_vis, cmap='Blues', aspect='auto')
ax3.set_xlabel('Bit Position')
ax3.set_ylabel('Parity Check')
ax3.set_title('[7,4,3] Parity-Check Matrix H')
ax3.set_xticks(range(7))
ax3.set_xticklabels([f'{i+1}' for i in range(7)])
plt.colorbar(im, ax=ax3)

# Error correction capability
ax4 = axes[1, 1]
p_values = np.linspace(0.001, 0.2, 100)

# Word error rates
def wer_hamming(p, n=7, t=1):
    from math import comb
    p_correct = sum(comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(t+1))
    return 1 - p_correct

def wer_extended(p, n=8, t=1):
    from math import comb
    p_correct = sum(comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(t+1))
    return 1 - p_correct

def wer_uncoded(p, k=4):
    return 1 - (1-p)**k

wer_h = [wer_hamming(p) for p in p_values]
wer_e = [wer_extended(p) for p in p_values]
wer_u = [wer_uncoded(p) for p in p_values]

ax4.semilogy(p_values, wer_u, 'r--', label='Uncoded (4 bits)', linewidth=2)
ax4.semilogy(p_values, wer_h, 'b-', label='[7,4,3] Hamming', linewidth=2)
ax4.semilogy(p_values, wer_e, 'g-', label='[8,4,4] Extended', linewidth=2)
ax4.set_xlabel('Physical Error Probability')
ax4.set_ylabel('Word Error Rate')
ax4.set_title('Error Correction Performance')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 0.2])

plt.tight_layout()
plt.savefig('day_676_hamming_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_676_hamming_analysis.png'")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Hamming parameters | $[2^r - 1, 2^r - 1 - r, 3]$ |
| Extended Hamming | $[2^r, 2^r - 1 - r, 4]$ |
| Perfect condition | $2^k (1 + n) = 2^n$ |
| Simplex (dual) | $[2^r - 1, r, 2^{r-1}]$ |
| CSS condition | $C^\perp \subseteq C$ ✓ for Hamming |

### Main Takeaways

1. **Hamming codes are perfect** — they achieve the sphere-packing bound
2. **Syndrome = error position** (in binary) for Hamming codes
3. **Extended Hamming codes** increase distance from 3 to 4
4. **Dual of Hamming is simplex** — a constant-weight code
5. **Self-orthogonality enables** quantum CSS construction (Steane code)

---

## Daily Checklist

- [ ] Verify the perfect code property for [7, 4, 3]
- [ ] Understand syndrome decoding for Hamming codes
- [ ] Construct the extended [8, 4, 4] code
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Connect Hamming self-orthogonality to Steane code
- [ ] Run the computational lab and analyze weight distributions

---

## Preview: Day 677

Tomorrow we study **fundamental bounds in coding theory** — the Singleton bound, Hamming bound, and Gilbert-Varshamov bound. These bounds tell us what's possible and impossible in error correction, guiding both classical and quantum code design.

---

*"Hamming codes achieve the impossible: they perfectly tile the space of all possible received words."*

---

**Next:** [Day_677_Friday.md](Day_677_Friday.md) — Bounds in Coding Theory
