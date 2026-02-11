# Day 673: Welcome to Year 2 — Linear Codes Fundamentals

## Welcome to Year 2: Advanced Quantum Science

**Congratulations!** You have completed Years 0 and 1 — 672 days of foundational mathematics and quantum mechanics. Today begins your journey into the heart of fault-tolerant quantum computing: **quantum error correction**.

Everything you've learned — linear algebra, Hilbert spaces, quantum gates, stabilizers — converges here. Year 2 transforms you from understanding quantum algorithms to understanding how to make them *actually work* on real, noisy hardware.

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Linear Codes Fundamentals |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 673, you will be able to:

1. Define a linear code and its key parameters [n, k, d]
2. Construct and interpret generator matrices
3. Construct and interpret parity-check matrices
4. Convert between systematic and non-systematic forms
5. Calculate the rate and error-correction capability of a code
6. Explain why classical error correction is foundational for quantum codes

---

## Core Content

### 1. The Error Correction Problem

**The Fundamental Challenge:** Transmit or store information reliably through a noisy channel.

In classical computing, we face bit flips: a 0 might become a 1, or vice versa. In quantum computing, we face something worse: continuous errors, phase flips, and the no-cloning theorem preventing direct redundancy.

But here's the crucial insight: **quantum error correction builds on classical techniques**. Master classical codes first, then generalize.

**Shannon's Noisy Channel Coding Theorem (1948):**
> For any channel with capacity C > 0, there exist codes that achieve arbitrarily low error probability at any rate R < C.

This theorem guarantees that reliable communication is *possible* — our job is to construct codes that achieve it.

---

### 2. Linear Codes: Definition

A **linear code** $C$ over a finite field $\mathbb{F}_q$ is an $[n, k, d]$ code where:

| Parameter | Meaning |
|-----------|---------|
| $n$ | Block length (total symbols in codeword) |
| $k$ | Dimension (number of information symbols) |
| $d$ | Minimum distance (minimum Hamming weight of nonzero codeword) |

**Key Property:** $C$ is a $k$-dimensional linear subspace of $\mathbb{F}_q^n$.

For binary codes ($q = 2$), we work over $\mathbb{F}_2 = \{0, 1\}$ with addition and multiplication mod 2.

**Why Linear?**
- The sum of two codewords is also a codeword
- Enables efficient encoding via matrix multiplication
- Syndrome decoding becomes tractable
- Leads directly to quantum stabilizer codes

---

### 3. Code Parameters and Trade-offs

**Code Rate:**
$$\boxed{R = \frac{k}{n}}$$

The rate measures information efficiency. Higher rate = less redundancy = weaker error protection.

**Error Correction Capability:**
A code with minimum distance $d$ can:
- **Detect** up to $d - 1$ errors
- **Correct** up to $t = \lfloor (d-1)/2 \rfloor$ errors

$$\boxed{t = \left\lfloor \frac{d-1}{2} \right\rfloor}$$

**The Fundamental Trade-off:**
$$\text{Rate} \quad \longleftrightarrow \quad \text{Error Protection}$$

Higher rate codes have less redundancy and weaker error correction. The art of coding theory is optimizing this trade-off.

---

### 4. Generator Matrix G

The **generator matrix** $G$ is a $k \times n$ matrix whose rows form a basis for the code $C$.

**Encoding:**
$$\boxed{\mathbf{c} = \mathbf{m} \cdot G}$$

where:
- $\mathbf{m}$ is a $1 \times k$ message vector
- $\mathbf{c}$ is a $1 \times n$ codeword

**Example: [3, 1, 3] Repetition Code**

$$G = \begin{pmatrix} 1 & 1 & 1 \end{pmatrix}$$

Encoding:
- Message $\mathbf{m} = (0)$ → Codeword $\mathbf{c} = (0, 0, 0)$
- Message $\mathbf{m} = (1)$ → Codeword $\mathbf{c} = (1, 1, 1)$

**Systematic Form:**
A generator matrix is in **systematic form** if:

$$G = [I_k \mid P]$$

where $I_k$ is the $k \times k$ identity matrix and $P$ is a $k \times (n-k)$ parity matrix.

In systematic form, the first $k$ positions of the codeword contain the original message unchanged.

---

### 5. Parity-Check Matrix H

The **parity-check matrix** $H$ is an $(n-k) \times n$ matrix satisfying:

$$\boxed{H \cdot \mathbf{c}^T = \mathbf{0} \quad \text{for all } \mathbf{c} \in C}$$

**Key Relationship:**
$$\boxed{G \cdot H^T = \mathbf{0}}$$

**For Systematic G:**
If $G = [I_k \mid P]$, then:
$$H = [-P^T \mid I_{n-k}]$$

Over $\mathbb{F}_2$, $-P^T = P^T$, so:
$$H = [P^T \mid I_{n-k}]$$

**Example: [3, 1, 3] Repetition Code**

$$H = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix}$$

Verify: $H \cdot (1, 1, 1)^T = (0, 0)^T$ ✓

---

### 6. The Dual Code

The **dual code** $C^\perp$ consists of all vectors orthogonal to every codeword:

$$\boxed{C^\perp = \{\mathbf{z} \in \mathbb{F}_q^n : \langle \mathbf{x}, \mathbf{z} \rangle = 0, \; \forall \mathbf{x} \in C\}}$$

**Properties:**
- The parity-check matrix $H$ of $C$ is a generator matrix for $C^\perp$
- The generator matrix $G$ of $C$ is a parity-check matrix for $C^\perp$
- $\dim(C) + \dim(C^\perp) = n$
- $(C^\perp)^\perp = C$

**Self-Dual Codes:**
A code is **self-dual** if $C = C^\perp$. This requires $n = 2k$.

**Self-Orthogonal Codes:**
A code is **self-orthogonal** if $C \subseteq C^\perp$. This is crucial for CSS quantum codes!

---

### 7. Connection to Quantum Error Correction

**Why This Matters for Quantum:**

The CSS (Calderbank-Shor-Steane) construction creates quantum codes from pairs of classical codes $C_1$ and $C_2$ satisfying:

$$\boxed{C_2^\perp \subseteq C_1}$$

This condition ensures that X and Z stabilizer generators commute!

**The Bridge:**
| Classical Concept | Quantum Analog |
|-------------------|----------------|
| Linear code $C$ | Code space $\mathcal{C}$ |
| Generator matrix $G$ | Logical operators |
| Parity-check matrix $H$ | Stabilizer generators |
| Syndrome | Error syndrome |
| Dual code $C^\perp$ | Dual stabilizer structure |

---

## Physical Interpretation

### Why Redundancy Enables Error Correction

Consider sending a single bit through a noisy channel with 10% bit-flip probability:
- Without coding: 10% error rate
- With 3-bit repetition: Error requires ≥2 flips → ~2.8% error rate

The redundancy creates **distinguishability**: different errors produce different received words, allowing correction.

### The Geometry of Codes

Think of codewords as points in an n-dimensional space. The minimum distance $d$ is the smallest gap between any two codewords. To correct $t$ errors, we need non-overlapping "spheres" of radius $t$ around each codeword.

---

## Worked Examples

### Example 1: [6, 3, 3] Binary Code

**Problem:** Verify that the following is a valid [6, 3, 3] code.

Generator matrix:
$$G = \begin{pmatrix} 1 & 0 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 & 1 & 1 \end{pmatrix}$$

**Solution:**

1. **Dimensions:** $G$ is $3 \times 6$, so $k = 3$, $n = 6$ ✓

2. **Systematic form:** $G = [I_3 \mid P]$ where $P = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}$ ✓

3. **Parity-check matrix:**
$$H = [P^T \mid I_3] = \begin{pmatrix} 1 & 1 & 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 & 0 & 1 \end{pmatrix}$$

4. **Verify $GH^T = 0$:** (exercise: check this)

5. **Find minimum distance:** Generate all $2^3 = 8$ codewords and find minimum nonzero weight:

| Message | Codeword | Weight |
|---------|----------|--------|
| 000 | 000000 | 0 |
| 001 | 001011 | 3 |
| 010 | 010101 | 3 |
| 011 | 011110 | 4 |
| 100 | 100110 | 3 |
| 101 | 101101 | 4 |
| 110 | 110011 | 4 |
| 111 | 111000 | 3 |

Minimum nonzero weight = 3, so $d = 3$ ✓

**Conclusion:** This is a valid [6, 3, 3] code with rate $R = 1/2$ that can correct $t = 1$ error. ∎

---

### Example 2: Verifying Parity-Check Relationship

**Problem:** Show that $H \cdot \mathbf{c}^T = 0$ for the codeword $\mathbf{c} = (1, 0, 1, 1, 0, 0)$ from the [6, 3, 3] code above.

**Solution:**

$$H \cdot \mathbf{c}^T = \begin{pmatrix} 1 & 1 & 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \\ 1 \\ 1 \\ 0 \\ 0 \end{pmatrix}$$

Row 1: $1 \cdot 1 + 1 \cdot 0 + 0 \cdot 1 + 1 \cdot 1 + 0 \cdot 0 + 0 \cdot 0 = 1 + 1 = 0$ (mod 2) ✓
Row 2: $1 \cdot 1 + 0 \cdot 0 + 1 \cdot 1 + 0 \cdot 1 + 1 \cdot 0 + 0 \cdot 0 = 1 + 1 = 0$ (mod 2) ✓
Row 3: $0 \cdot 1 + 1 \cdot 0 + 1 \cdot 1 + 0 \cdot 1 + 0 \cdot 0 + 1 \cdot 0 = 1 \neq 0$...

Wait, let me recalculate. The codeword for message (1, 0, 1) should be:
$$\mathbf{c} = (1, 0, 1) \cdot G = (1, 0, 1, 1, 0, 0)$$

Hmm, let me verify: $1 \cdot (1,0,0,1,1,0) + 0 \cdot (0,1,0,1,0,1) + 1 \cdot (0,0,1,0,1,1)$
$= (1,0,0,1,1,0) + (0,0,1,0,1,1) = (1,0,1,1,0,1)$

The correct codeword is $(1, 0, 1, 1, 0, 1)$.

Now verify:
Row 3: $0 \cdot 1 + 1 \cdot 0 + 1 \cdot 1 + 0 \cdot 1 + 0 \cdot 0 + 1 \cdot 1 = 1 + 1 = 0$ ✓

**Result:** $H \cdot \mathbf{c}^T = (0, 0, 0)^T$ ∎

---

### Example 3: Finding the Dual Code

**Problem:** Find the dual of the [3, 1, 3] repetition code.

**Solution:**

The repetition code has:
$$G = \begin{pmatrix} 1 & 1 & 1 \end{pmatrix}, \quad H = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix}$$

The dual code $C^\perp$ has:
- Generator matrix: $H$ (the parity-check of $C$)
- Parameters: $[3, 2, 2]$ (3 bits, 2 info bits, distance 2)

Codewords of $C^\perp$:
| Message | Codeword |
|---------|----------|
| 00 | 000 |
| 01 | 101 |
| 10 | 110 |
| 11 | 011 |

This is the **[3, 2, 2] single parity-check code** — it can detect (but not correct) single errors.

**Observation:** Repetition and parity-check codes are duals! ∎

---

## Practice Problems

### Level 1: Direct Application

1. For the [7, 4, 3] Hamming code with generator matrix:
$$G = \begin{pmatrix} 1 & 0 & 0 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 1 & 1 & 1 & 1 \end{pmatrix}$$
   Find the parity-check matrix $H$.

2. Encode the message $\mathbf{m} = (1, 0, 1, 1)$ using the Hamming code above.

3. Compute the rate and error-correction capability of a [15, 11, 3] code.

### Level 2: Intermediate

4. Prove that for any linear code, the minimum distance equals the minimum number of linearly dependent columns of $H$.

5. Show that if $G$ is in systematic form $[I_k \mid P]$, then $G \cdot H^T = 0$ where $H = [P^T \mid I_{n-k}]$.

6. Construct a [5, 2, 3] binary linear code and verify its parameters.

### Level 3: Challenging

7. Prove that for a self-orthogonal code ($C \subseteq C^\perp$), all codewords have even weight.

8. Show that there is no [5, 3, 3] binary linear code. (Hint: Use the Singleton bound)

9. **Research:** Why does the CSS construction require $C_2^\perp \subseteq C_1$? Connect this to the commutativity of stabilizer generators.

---

## Computational Lab

### Objective
Implement linear code operations and verify the theory numerically.

```python
"""
Day 673 Computational Lab: Linear Codes Fundamentals
Year 2: Advanced Quantum Science
"""

import numpy as np
from itertools import product

# =============================================================================
# Part 1: Basic Linear Code Operations
# =============================================================================

print("=" * 60)
print("Part 1: Linear Code Operations over F_2")
print("=" * 60)

def gf2_matmul(A, B):
    """Matrix multiplication over GF(2)."""
    return np.mod(A @ B, 2)

def gf2_add(a, b):
    """Vector addition over GF(2)."""
    return np.mod(a + b, 2)

# Define the [6, 3, 3] code
G = np.array([
    [1, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 1]
], dtype=int)

P = G[:, 3:]  # Parity matrix
H = np.hstack([P.T, np.eye(3, dtype=int)])

print("\nGenerator matrix G:")
print(G)
print("\nParity-check matrix H:")
print(H)

# Verify G * H^T = 0
print("\nVerification: G * H^T (should be all zeros):")
print(gf2_matmul(G, H.T))

# =============================================================================
# Part 2: Encoding
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Encoding Messages")
print("=" * 60)

def encode(message, G):
    """Encode a message using generator matrix G."""
    return gf2_matmul(np.array([message]), G)[0]

# Encode all possible messages
print("\nAll codewords of the [6, 3, 3] code:")
print("-" * 40)
print(f"{'Message':<12} {'Codeword':<20} {'Weight'}")
print("-" * 40)

codewords = []
for m in product([0, 1], repeat=3):
    message = list(m)
    codeword = encode(message, G)
    weight = np.sum(codeword)
    codewords.append((message, codeword, weight))
    print(f"{str(message):<12} {str(codeword):<20} {weight}")

# Find minimum distance
nonzero_weights = [w for m, c, w in codewords if w > 0]
min_distance = min(nonzero_weights)
print(f"\nMinimum distance d = {min_distance}")
print(f"Error correction capability t = {(min_distance - 1) // 2}")

# =============================================================================
# Part 3: Parity Check Verification
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Parity Check Verification")
print("=" * 60)

print("\nVerifying H * c^T = 0 for all codewords:")
for message, codeword, weight in codewords:
    syndrome = gf2_matmul(H, codeword.reshape(-1, 1)).flatten()
    status = "✓" if np.all(syndrome == 0) else "✗"
    print(f"c = {codeword} -> syndrome = {syndrome} {status}")

# =============================================================================
# Part 4: The Dual Code
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: The Dual Code")
print("=" * 60)

# For our [6, 3, 3] code, the dual is [6, 3, ?]
# Generator of dual = H (parity-check of original)
G_dual = H.copy()

print("\nGenerator matrix of dual code (= H of original):")
print(G_dual)

print("\nCodewords of dual code:")
print("-" * 40)

dual_codewords = []
for m in product([0, 1], repeat=3):
    message = list(m)
    codeword = encode(message, G_dual)
    weight = np.sum(codeword)
    dual_codewords.append((message, codeword, weight))
    print(f"{str(message):<12} {str(codeword):<20} {weight}")

dual_nonzero_weights = [w for m, c, w in dual_codewords if w > 0]
dual_min_distance = min(dual_nonzero_weights)
print(f"\nDual code minimum distance d⊥ = {dual_min_distance}")
print(f"Dual code parameters: [6, 3, {dual_min_distance}]")

# =============================================================================
# Part 5: Hamming [7, 4, 3] Code
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: The [7, 4, 3] Hamming Code")
print("=" * 60)

G_hamming = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=int)

P_hamming = G_hamming[:, 4:]
H_hamming = np.hstack([P_hamming.T, np.eye(3, dtype=int)])

print("\nHamming code generator matrix G:")
print(G_hamming)
print("\nHamming code parity-check matrix H:")
print(H_hamming)

# Alternative H with columns as binary representations of 1-7
H_alt = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1]
], dtype=int)

print("\nAlternative H (columns = binary 1-7):")
print(H_alt)

# Count codewords and verify parameters
print("\nHamming code has 2^4 = 16 codewords")

# Verify it's a perfect code
n, k = 7, 4
hamming_sphere_size = 1 + n  # 1 + C(7,1) for t=1
total_space = 2**n
num_codewords = 2**k
coverage = num_codewords * hamming_sphere_size

print(f"\nPerfect code check:")
print(f"Total space: 2^{n} = {total_space}")
print(f"Sphere size (t=1): 1 + {n} = {hamming_sphere_size}")
print(f"Coverage: {num_codewords} × {hamming_sphere_size} = {coverage}")
print(f"Perfect code: {coverage == total_space} ✓")

# =============================================================================
# Part 6: Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Code Distance Visualization")
print("=" * 60)

import matplotlib.pyplot as plt

def hamming_distance(a, b):
    """Compute Hamming distance between two vectors."""
    return np.sum(a != b)

# Compute pairwise distances for [6, 3, 3] code
codeword_list = [c for m, c, w in codewords]
n_codewords = len(codeword_list)

distances = np.zeros((n_codewords, n_codewords), dtype=int)
for i in range(n_codewords):
    for j in range(n_codewords):
        distances[i, j] = hamming_distance(codeword_list[i], codeword_list[j])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(distances, cmap='Blues')
ax.set_title('[6, 3, 3] Code: Pairwise Hamming Distances', fontsize=12)
ax.set_xlabel('Codeword Index')
ax.set_ylabel('Codeword Index')
plt.colorbar(im, ax=ax, label='Hamming Distance')

# Add distance values
for i in range(n_codewords):
    for j in range(n_codewords):
        ax.text(j, i, distances[i, j], ha='center', va='center',
                color='white' if distances[i, j] > 2 else 'black')

plt.tight_layout()
plt.savefig('day_673_code_distances.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_673_code_distances.png'")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Code parameters | $[n, k, d]$ — length, dimension, distance |
| Code rate | $R = k/n$ |
| Error correction | $t = \lfloor(d-1)/2\rfloor$ |
| Encoding | $\mathbf{c} = \mathbf{m} \cdot G$ |
| Codeword test | $H \cdot \mathbf{c}^T = \mathbf{0}$ |
| Systematic form | $G = [I_k \mid P]$, $H = [P^T \mid I_{n-k}]$ |
| Fundamental relation | $G \cdot H^T = \mathbf{0}$ |

### Main Takeaways

1. **Linear codes** are subspaces — closure under addition enables efficient encoding
2. **Generator matrix** defines encoding; **parity-check matrix** defines the code
3. **Minimum distance** determines error-correction capability
4. **Dual codes** play a crucial role in quantum error correction (CSS construction)
5. **Rate vs. protection** is the fundamental trade-off in coding theory

---

## Daily Checklist

- [ ] Read Nielsen & Chuang Chapter 10.1 (introduction to QEC)
- [ ] Understand the [n, k, d] notation
- [ ] Verify generator/parity-check relationship for a code
- [ ] Complete Level 1 practice problems
- [ ] Attempt at least one Level 2 problem
- [ ] Run and understand the computational lab
- [ ] Write a one-paragraph summary connecting linear codes to quantum codes

---

## Preview: Day 674

Tomorrow we dive deeper into **generator and parity-check matrices**: systematic encoding algorithms, row reduction, and how to construct codes with specific properties. We'll also prove the fundamental theorem relating minimum distance to linear dependence of columns in $H$.

---

*"Information is physical."*
— Rolf Landauer

---

**Next:** [Day_674_Tuesday.md](Day_674_Tuesday.md) — Generator and Parity-Check Matrices Deep Dive
