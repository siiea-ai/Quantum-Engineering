# Day 611: Number Theory Background

## Overview

**Day 611** | Week 88, Day 2 | Month 22 | Quantum Algorithms I

Today we review the number theory foundations needed to fully understand Shor's algorithm, including Euler's theorem, the Chinese Remainder Theorem, and the structure of multiplicative groups.

---

## Learning Objectives

1. Review modular arithmetic fundamentals
2. Understand Euler's totient function
3. Apply Euler's theorem
4. Learn the Chinese Remainder Theorem
5. Analyze the multiplicative group structure

---

## Core Content

### Euler's Totient Function

**Definition:** $\phi(N)$ = count of integers $1 \leq k \leq N$ with $\gcd(k, N) = 1$.

**Key values:**
- $\phi(p) = p - 1$ for prime $p$
- $\phi(p^k) = p^{k-1}(p-1)$
- $\phi(pq) = (p-1)(q-1)$ for distinct primes

**Example:** $\phi(15) = \phi(3)\phi(5) = 2 \cdot 4 = 8$

### Euler's Theorem

**Theorem:** If $\gcd(a, N) = 1$, then:
$$a^{\phi(N)} \equiv 1 \pmod{N}$$

**Consequence:** The order $r$ of $a$ divides $\phi(N)$.

### The Multiplicative Group

$\mathbb{Z}_N^* = \{a : 1 \leq a < N, \gcd(a, N) = 1\}$

This is a group under multiplication mod $N$, with $|\mathbb{Z}_N^*| = \phi(N)$.

### Chinese Remainder Theorem (CRT)

**Theorem:** If $\gcd(m, n) = 1$, then:
$$\mathbb{Z}_{mn} \cong \mathbb{Z}_m \times \mathbb{Z}_n$$

**For $N = pq$:**
$$\mathbb{Z}_{pq}^* \cong \mathbb{Z}_p^* \times \mathbb{Z}_q^*$$

The order of $a$ mod $N$ equals $\text{lcm}(\text{ord}_p(a), \text{ord}_q(a))$.

### Why Order is Usually Even

For $N = pq$ with odd primes:
- $\phi(p) = p - 1$ is even
- $\phi(q) = q - 1$ is even
- Orders divide $\phi(N) = (p-1)(q-1)$, usually have even factors

**Probability analysis:** Random element has even order with probability $\geq 1 - 2^{1-k}$ where $k$ = number of distinct prime factors.

### The Structure of $\mathbb{Z}_p^*$

For prime $p$: $\mathbb{Z}_p^*$ is **cyclic** of order $p-1$.

There exist generators $g$ with $\text{ord}(g) = p-1$.

---

## Worked Examples

### Example 1: Compute $\phi(91)$

**Solution:**

$91 = 7 \times 13$

$\phi(91) = \phi(7)\phi(13) = 6 \times 12 = 72$

### Example 2: Find Order Using CRT

Find order of $a = 2$ mod $N = 15 = 3 \times 5$.

**Solution:**

Order of 2 mod 3: $2^1 = 2$, $2^2 = 4 \equiv 1 \pmod 3$. Order = 2.

Order of 2 mod 5: $2^1 = 2$, $2^2 = 4$, $2^3 = 8 \equiv 3$, $2^4 = 16 \equiv 1 \pmod 5$. Order = 4.

Order of 2 mod 15 = $\text{lcm}(2, 4) = 4$ ✓

### Example 3: Verify Euler's Theorem

Check $2^{\phi(15)} \equiv 1 \pmod{15}$.

**Solution:**

$\phi(15) = 8$

$2^8 = 256 = 17 \times 15 + 1 = 255 + 1$

$256 \equiv 1 \pmod{15}$ ✓

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Euler's totient | $\phi(pq) = (p-1)(q-1)$ |
| Euler's theorem | $a^{\phi(N)} \equiv 1 \pmod{N}$ |
| CRT | $\mathbb{Z}_{pq}^* \cong \mathbb{Z}_p^* \times \mathbb{Z}_q^*$ |
| Order via CRT | $\text{ord}_N(a) = \text{lcm}(\text{ord}_p(a), \text{ord}_q(a))$ |

### Key Takeaways

1. **Order divides $\phi(N)$** by Euler's theorem
2. **CRT decomposes** the group structure
3. **Even orders** are highly probable
4. **Group structure** explains why factoring-order reduction works

---

*Next: Day 612 - Quantum Period-Finding*
