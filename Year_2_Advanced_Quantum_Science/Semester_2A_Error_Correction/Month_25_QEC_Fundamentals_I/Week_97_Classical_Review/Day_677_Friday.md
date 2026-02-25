# Day 677: Bounds in Coding Theory

## Week 97: Classical Error Correction Review | Month 25: QEC Fundamentals I

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Fundamental Bounds |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 677, you will be able to:

1. State and prove the Singleton bound
2. Define MDS codes and give examples
3. Apply the Hamming (sphere-packing) bound
4. Understand the Gilbert-Varshamov existence bound
5. Analyze asymptotic behavior of these bounds
6. Connect classical bounds to quantum code constraints

---

## Core Content

### 1. The Singleton Bound

**Theorem (Singleton Bound):** For any $[n, k, d]$ code over $\mathbb{F}_q$:

$$\boxed{d \leq n - k + 1}$$

Equivalently: $k \leq n - d + 1$

**Proof:**

Consider the puncturing map $\pi: \mathbb{F}_q^n \to \mathbb{F}_q^{n-d+1}$ that keeps only the first $n - d + 1$ coordinates.

For distinct codewords $\mathbf{c}_1, \mathbf{c}_2$: they differ in at least $d$ positions.

If $\pi(\mathbf{c}_1) = \pi(\mathbf{c}_2)$, then $\mathbf{c}_1$ and $\mathbf{c}_2$ agree on the first $n - d + 1$ positions, so they differ only in the last $d - 1$ positions.

But $d(\mathbf{c}_1, \mathbf{c}_2) \geq d$, contradiction!

Therefore $\pi$ is injective on $C$, so:
$$|C| = q^k \leq q^{n-d+1}$$

Thus $k \leq n - d + 1$, i.e., $d \leq n - k + 1$. ∎

---

### 2. MDS Codes

**Definition:** A code achieving the Singleton bound ($d = n - k + 1$) is called **Maximum Distance Separable (MDS)**.

**Examples of MDS Codes:**
- **Trivial codes:** $[n, 1, n]$ repetition, $[n, n-1, 2]$ parity-check
- **Reed-Solomon codes:** $[n, k, n-k+1]$ over $\mathbb{F}_q$ with $n \leq q$

**Non-Example:** Hamming codes are NOT MDS.
- [7, 4, 3] Hamming: Singleton gives $d \leq 7 - 4 + 1 = 4$
- Actual $d = 3 < 4$

**MDS Conjecture:** Over $\mathbb{F}_q$, an MDS code has $n \leq q + 1$ (except for trivial exceptions).

---

### 3. The Hamming Bound (Sphere-Packing Bound)

**Theorem (Hamming Bound):** For a $q$-ary $[n, k, d]$ code with $d = 2t + 1$:

$$\boxed{q^k \cdot \sum_{i=0}^{t} \binom{n}{i}(q-1)^i \leq q^n}$$

**Interpretation:**
- Left side: Total volume of Hamming spheres of radius $t$ around all codewords
- Right side: Total space
- The spheres must not overlap (for $t$-error correction)

**Binary Case ($q = 2$):**
$$2^k \cdot \sum_{i=0}^{t} \binom{n}{i} \leq 2^n$$

**Volume of Hamming Ball:**
$$V_q(n, t) = \sum_{i=0}^{t} \binom{n}{i}(q-1)^i$$

For binary codes: $V_2(n, t) = \sum_{i=0}^{t} \binom{n}{i}$

**Example: Can a [10, 5, 5] binary code exist?**
- $t = (5-1)/2 = 2$
- $V_2(10, 2) = 1 + 10 + 45 = 56$
- Need: $2^5 \cdot 56 = 1792 \leq 2^{10} = 1024$? **No!**

A [10, 5, 5] binary code cannot exist.

---

### 4. Perfect Codes

**Definition:** A code achieving the Hamming bound with equality is **perfect**.

$$q^k \cdot V_q(n, t) = q^n$$

**Complete Classification (Tietäväinen-van Lint):**

The only nontrivial perfect codes are:
1. **Hamming codes:** $[2^r - 1, 2^r - 1 - r, 3]$ (any $q$, $t = 1$)
2. **Binary Golay code:** $[23, 12, 7]$ ($q = 2$, $t = 3$)
3. **Ternary Golay code:** $[11, 6, 5]$ ($q = 3$, $t = 2$)

**Implication:** Perfect codes are extremely rare!

---

### 5. The Gilbert-Varshamov Bound

**Theorem (Gilbert-Varshamov):** There exists a $q$-ary linear $[n, k, d]$ code if:

$$\boxed{q^{n-k} > \sum_{i=0}^{d-2} \binom{n-1}{i}(q-1)^i}$$

**Interpretation:** This is an **existence bound** — it guarantees that "good" codes exist but doesn't tell us how to construct them.

**Proof Idea (Greedy Construction):**
1. Start with $H$ having no columns
2. Repeatedly add columns that maintain minimum distance $\geq d$
3. A new column can be added if it's not in the span of any $d-2$ existing columns
4. This is possible as long as the number of "forbidden" vectors is less than $q^{n-k}$

**Binary Case:**
$$2^{n-k} > \sum_{i=0}^{d-2} \binom{n-1}{i}$$

---

### 6. Asymptotic Analysis

For large $n$, define:
- **Rate:** $R = k/n$
- **Relative distance:** $\delta = d/n$

**Asymptotic Singleton Bound:**
$$R \leq 1 - \delta$$

**Asymptotic Hamming Bound:**
$$R \leq 1 - H_q(\delta/2)$$

where $H_q(x) = x \log_q(q-1) - x \log_q(x) - (1-x) \log_q(1-x)$ is the $q$-ary entropy function.

**Asymptotic Gilbert-Varshamov Bound:**
$$R \geq 1 - H_q(\delta)$$

**The GV-Hamming Gap:**

For $0 < \delta < (q-1)/q$, there's a gap between what we can prove exists (GV) and what we can prove impossible (Hamming).

Whether codes exist in this gap for all parameters is a major open problem!

---

### 7. Quantum Bounds

**Quantum Singleton Bound:** For an $[[n, k, d]]$ quantum code:
$$\boxed{k \leq n - 2(d - 1)}$$

or equivalently: $d \leq \frac{n - k}{2} + 1$

**Difference from Classical:** The factor of 2 arises because quantum codes must correct both X and Z errors.

**Quantum Hamming Bound:** For $[[n, k, d]]$ with $d = 2t + 1$:
$$2^k \cdot \sum_{i=0}^{t} 3^i \binom{n}{i} \leq 2^n$$

The factor $3^i$ accounts for three types of single-qubit errors: X, Y, Z.

---

## Physical Interpretation

### Why Bounds Matter

Bounds tell us the **fundamental limits** of error correction:

1. **Singleton:** You can't have high rate AND high distance
2. **Hamming:** Perfect protection requires exponential redundancy
3. **GV:** Good codes exist (even if we can't always find them)

### The Rate-Distance Tradeoff

Think of it as a budget:
- Each bit of rate costs protection
- Each bit of protection costs rate
- Bounds quantify this tradeoff precisely

---

## Worked Examples

### Example 1: Applying Multiple Bounds

**Problem:** What constraints do the bounds place on a binary code with $n = 15$ and $k = 7$?

**Solution:**

**Singleton bound:**
$d \leq n - k + 1 = 15 - 7 + 1 = 9$

**Hamming bound:** For $d = 2t + 1$:

Try $t = 4$ ($d = 9$):
$V_2(15, 4) = 1 + 15 + 105 + 455 + 1365 = 1941$
$2^7 \cdot 1941 = 248,448 > 2^{15} = 32,768$ ✗

Try $t = 3$ ($d = 7$):
$V_2(15, 3) = 1 + 15 + 105 + 455 = 576$
$2^7 \cdot 576 = 73,728 > 32,768$ ✗

Try $t = 2$ ($d = 5$):
$V_2(15, 2) = 1 + 15 + 105 = 121$
$2^7 \cdot 121 = 15,488 \leq 32,768$ ✓

**Hamming bound allows:** $d \leq 5$

**Gilbert-Varshamov:** Check if $d = 5$ is achievable.
$2^{15-7} = 256 > \sum_{i=0}^{3} \binom{14}{i} = 1 + 14 + 91 + 364 = 470$?

No, $256 < 470$, so GV doesn't guarantee $d = 5$.

Try $d = 4$:
$2^8 = 256 > \sum_{i=0}^{2} \binom{14}{i} = 1 + 14 + 91 = 106$? ✓

**GV guarantees:** A [15, 7, 4] code exists.

**Summary:** For [15, 7, d]:
- Singleton: $d \leq 9$
- Hamming: $d \leq 5$
- GV: $d \geq 4$ guaranteed

Actual best known: The **BCH code** [15, 7, 5] achieves $d = 5$! ∎

---

### Example 2: Proving Non-Existence

**Problem:** Prove that no binary [12, 8, 4] code exists.

**Solution:**

**Method 1: Hamming Bound**

For $d = 4$, we have $t = 1$ (can correct 1 error, detect 3).

Wait, for $d = 4$, $t = \lfloor(4-1)/2\rfloor = 1$.

$V_2(12, 1) = 1 + 12 = 13$

Check: $2^8 \cdot 13 = 3328 \leq 2^{12} = 4096$? ✓

Hamming bound is satisfied. Need another approach.

**Method 2: Plotkin Bound**

For $d > n/2$, the Plotkin bound states:
$$|C| \leq \frac{d}{2d - n}$$

Here $d = 4$, $n = 12$, so $d = 4 \not> n/2 = 6$. Doesn't apply.

**Method 3: Linear Programming Bound**

This requires more sophisticated analysis.

Actually, let me check whether such a code might exist using the Griesmer bound.

**Griesmer Bound:** For a binary linear $[n, k, d]$ code:
$$n \geq \sum_{i=0}^{k-1} \lceil d/2^i \rceil$$

For [12, 8, 4]:
$\sum_{i=0}^{7} \lceil 4/2^i \rceil = 4 + 2 + 1 + 1 + 1 + 1 + 1 + 1 = 12$ ✓

The Griesmer bound is tight, so existence is possible in principle.

**Conclusion:** The basic bounds don't rule out [12, 8, 4]. Further analysis (or exhaustive search) shows such a code exists (shortened Golay). ∎

---

### Example 3: MDS Code Verification

**Problem:** Verify that the $[4, 2, 3]$ code over $\mathbb{F}_5$ with generator matrix
$$G = \begin{pmatrix} 1 & 0 & 1 & 2 \\ 0 & 1 & 3 & 4 \end{pmatrix}$$
is MDS.

**Solution:**

**Singleton bound:** $d \leq n - k + 1 = 4 - 2 + 1 = 3$

**Check minimum distance:** Find weight of all nonzero codewords.

Over $\mathbb{F}_5$, there are $5^2 - 1 = 24$ nonzero messages.

For message $(a, b)$: codeword = $(a, b, a + 3b, 2a + 4b)$

The minimum weight is 3 if no two coordinates are ever both zero for nonzero $(a, b)$.

Check: Can $(a + 3b, 2a + 4b) = (0, 0)$?
- $a + 3b = 0 \Rightarrow a = -3b = 2b$ (mod 5)
- $2a + 4b = 0 \Rightarrow 2(2b) + 4b = 8b = 3b = 0 \Rightarrow b = 0$

So only $(0, 0)$ gives zeros in positions 3, 4. ✓

Similar analysis for other position pairs shows minimum distance is exactly 3.

**Conclusion:** $d = 3 = n - k + 1$, so this is an MDS code. ∎

---

## Practice Problems

### Level 1: Direct Application

1. Use the Singleton bound to find the maximum possible distance of a [20, 12] code.

2. Verify the Hamming bound for the [23, 12, 7] Golay code (show it's perfect).

3. Apply the Gilbert-Varshamov bound to show a binary [10, 4, 4] code exists.

### Level 2: Intermediate

4. Prove that for any binary linear code, if $d$ is odd, then the extended code has distance $d + 1$.

5. Show that Reed-Solomon codes are MDS by analyzing their construction.

6. Calculate the asymptotic rate of Hamming codes as $r \to \infty$.

### Level 3: Challenging

7. Prove the quantum Singleton bound $k \leq n - 2(d-1)$ for CSS codes.

8. Show that random linear codes achieve the Gilbert-Varshamov bound with high probability.

9. **Research:** What is the current status of closing the gap between Hamming and Gilbert-Varshamov bounds?

---

## Computational Lab

```python
"""
Day 677 Computational Lab: Bounds in Coding Theory
Year 2: Advanced Quantum Science
"""

import numpy as np
from math import comb, log2, ceil
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Computing Bounds
# =============================================================================

print("=" * 60)
print("Part 1: Fundamental Bounds")
print("=" * 60)

def singleton_bound(n, k):
    """Maximum distance by Singleton bound."""
    return n - k + 1

def hamming_ball_volume(n, t, q=2):
    """Volume of Hamming ball of radius t in F_q^n."""
    return sum(comb(n, i) * (q-1)**i for i in range(t + 1))

def hamming_bound_max_k(n, d, q=2):
    """Maximum k allowed by Hamming bound for given n, d."""
    t = (d - 1) // 2
    V = hamming_ball_volume(n, t, q)
    # Need q^k * V <= q^n
    # So k <= n - log_q(V)
    max_k = n - log2(V) / log2(q)
    return int(max_k)

def gv_check(n, k, d, q=2):
    """Check if Gilbert-Varshamov allows [n, k, d] code."""
    # Need q^{n-k} > sum_{i=0}^{d-2} C(n-1, i) * (q-1)^i
    lhs = q**(n - k)
    rhs = sum(comb(n-1, i) * (q-1)**i for i in range(d - 1))
    return lhs > rhs

# Example: [15, k, d] codes
print("\nBounds for n = 15:")
print("-" * 50)
print(f"{'k':<5} {'Singleton d≤':<15} {'Hamming d≤':<15} {'GV d≥'}")
print("-" * 50)

for k in range(1, 15):
    s_bound = singleton_bound(15, k)

    # Hamming bound: find max d
    h_bound = 1
    for d in range(1, 16):
        t = (d - 1) // 2
        V = hamming_ball_volume(15, t)
        if 2**k * V <= 2**15:
            h_bound = d

    # GV bound: find min guaranteed d
    gv_bound = 1
    for d in range(15, 0, -1):
        if gv_check(15, k, d):
            gv_bound = d
            break

    print(f"{k:<5} {s_bound:<15} {h_bound:<15} {gv_bound}")

# =============================================================================
# Part 2: Asymptotic Bounds
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Asymptotic Bounds")
print("=" * 60)

def binary_entropy(x):
    """Binary entropy function H(x) = -x*log(x) - (1-x)*log(1-x)."""
    if x <= 0 or x >= 1:
        return 0
    return -x * log2(x) - (1 - x) * log2(1 - x)

# Plot asymptotic bounds
delta = np.linspace(0.001, 0.499, 200)

# Singleton: R <= 1 - delta
singleton_R = 1 - delta

# Hamming: R <= 1 - H(delta/2)
hamming_R = np.array([1 - binary_entropy(d/2) for d in delta])

# Gilbert-Varshamov: R >= 1 - H(delta)
gv_R = np.array([1 - binary_entropy(d) for d in delta])

fig, ax = plt.subplots(figsize=(10, 7))

ax.fill_between(delta, gv_R, hamming_R, alpha=0.3, color='green',
                label='Achievable (GV to Hamming gap)')
ax.plot(delta, singleton_R, 'r-', linewidth=2, label='Singleton Bound')
ax.plot(delta, hamming_R, 'b-', linewidth=2, label='Hamming Bound')
ax.plot(delta, gv_R, 'g--', linewidth=2, label='Gilbert-Varshamov Bound')

ax.set_xlabel('Relative Distance δ = d/n', fontsize=12)
ax.set_ylabel('Rate R = k/n', fontsize=12)
ax.set_title('Asymptotic Bounds for Binary Codes', fontsize=14)
ax.legend(loc='upper right')
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_677_asymptotic_bounds.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_677_asymptotic_bounds.png'")

# =============================================================================
# Part 3: Perfect Code Analysis
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Perfect Codes")
print("=" * 60)

def is_perfect(n, k, d, q=2):
    """Check if code parameters yield a perfect code."""
    t = (d - 1) // 2
    V = hamming_ball_volume(n, t, q)
    return q**k * V == q**n

print("\nChecking Hamming codes for perfectness:")
for r in range(2, 7):
    n = 2**r - 1
    k = n - r
    d = 3
    perfect = is_perfect(n, k, d)
    status = "✓ Perfect" if perfect else "✗ Not perfect"
    V = hamming_ball_volume(n, 1)
    print(f"  [{n}, {k}, {d}]: 2^{k} × {V} = {2**k * V}, 2^{n} = {2**n}  {status}")

print("\nBinary Golay code [23, 12, 7]:")
n, k, d = 23, 12, 7
t = 3
V = hamming_ball_volume(23, 3)
perfect = is_perfect(n, k, d)
print(f"  V(23, 3) = {V}")
print(f"  2^12 × {V} = {2**12 * V}, 2^23 = {2**23}")
print(f"  Perfect: {perfect}")

# =============================================================================
# Part 4: Quantum Bounds
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Quantum Bounds")
print("=" * 60)

def quantum_singleton(n, k):
    """Maximum d by quantum Singleton bound."""
    return (n - k) // 2 + 1

def quantum_hamming_ball_volume(n, t):
    """Volume of quantum Hamming ball (3 error types per position)."""
    return sum(comb(n, i) * 3**i for i in range(t + 1))

def quantum_hamming_bound_max_k(n, d):
    """Maximum k allowed by quantum Hamming bound."""
    t = (d - 1) // 2
    V = quantum_hamming_ball_volume(n, t)
    max_k = n - log2(V)
    return int(max_k)

print("\nQuantum bounds for small n:")
print("-" * 60)
print(f"{'n':<5} {'Classical S(k=1)':<18} {'Quantum S(k=1)':<18} {'Difference'}")
print("-" * 60)

for n in range(5, 16):
    c_sing = singleton_bound(n, 1)
    q_sing = quantum_singleton(n, 1)
    print(f"{n:<5} {c_sing:<18} {q_sing:<18} {c_sing - q_sing}")

# Steane code check
print("\n[[7, 1, 3]] Steane code bounds check:")
n, k, d = 7, 1, 3
q_s = quantum_singleton(n, k)
print(f"  Quantum Singleton: d ≤ {q_s}")
print(f"  Actual d = 3: {'✓ Satisfies' if d <= q_s else '✗ Violates'}")

t = 1
V_q = quantum_hamming_ball_volume(n, t)
print(f"  Quantum Hamming ball V_Q(7,1) = {V_q}")
print(f"  2^k × V = {2**k * V_q}, 2^n = {2**n}")
print(f"  Satisfies quantum Hamming: {2**k * V_q <= 2**n}")

# =============================================================================
# Part 5: Best Known Codes Table
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Comparing Bounds with Best Known Codes")
print("=" * 60)

# Table of best known binary codes (from coding theory tables)
best_known = {
    (7, 4): 3,   # Hamming
    (8, 4): 4,   # Extended Hamming
    (15, 11): 3, # Hamming
    (15, 7): 5,  # BCH
    (15, 5): 7,  # BCH
    (23, 12): 7, # Golay
    (24, 12): 8, # Extended Golay
    (31, 26): 3, # Hamming
    (31, 21): 5, # BCH
    (31, 16): 7, # BCH
}

print("\nBest known codes vs bounds:")
print("-" * 70)
print(f"{'[n,k]':<10} {'Best d':<10} {'Singleton':<12} {'Hamming':<12} {'GV':<10}")
print("-" * 70)

for (n, k), d_best in sorted(best_known.items()):
    s = singleton_bound(n, k)

    # Hamming
    h = 1
    for d in range(1, n+1):
        t = (d - 1) // 2
        V = hamming_ball_volume(n, t)
        if 2**k * V <= 2**n:
            h = d

    # GV
    gv = 1
    for d in range(n, 0, -1):
        if gv_check(n, k, d):
            gv = d
            break

    print(f"[{n},{k}]{'':<4} {d_best:<10} {s:<12} {h:<12} {gv:<10}")

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Bound | Statement | Type |
|-------|-----------|------|
| Singleton | $d \leq n - k + 1$ | Upper |
| Hamming | $q^k V_q(n,t) \leq q^n$ | Upper |
| Gilbert-Varshamov | $q^{n-k} > V_q(n-1, d-2)$ | Lower |
| Quantum Singleton | $k \leq n - 2(d-1)$ | Upper |

### Main Takeaways

1. **Singleton bound** is simple but often loose (MDS codes are rare)
2. **Hamming bound** limits sphere-packing; perfect codes are extremely rare
3. **Gilbert-Varshamov** guarantees good codes exist
4. **Asymptotic gap** between GV and Hamming remains open
5. **Quantum bounds** have extra factor of 2 for error types

---

## Daily Checklist

- [ ] Prove the Singleton bound
- [ ] Calculate Hamming ball volumes
- [ ] Apply GV bound to verify code existence
- [ ] Complete Level 1 practice problems
- [ ] Understand the asymptotic rate-distance tradeoff
- [ ] Connect to quantum Singleton bound

---

## Preview: Day 678

Tomorrow we explore **BCH and Reed-Solomon codes** — powerful algebraic constructions that achieve excellent parameters and form the basis for many practical error correction systems, including the connection to quantum CSS codes.

---

*"Bounds tell us the rules of the game; constructions show us how to play."*

---

**Next:** [Day_678_Saturday.md](Day_678_Saturday.md) — BCH and Reed-Solomon Codes
