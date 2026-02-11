# Day 610: Factoring to Order-Finding

## Overview

**Day 610** | Week 88, Day 1 | Month 22 | Quantum Algorithms I

Today we establish the crucial connection between factoring integers and finding the order of elements in modular arithmetic. This reduction is the key insight that makes Shor's algorithm possible.

---

## Learning Objectives

1. Understand the factoring problem
2. Define the order-finding problem
3. Prove the reduction from factoring to order-finding
4. Understand why this reduction enables quantum speedup
5. Analyze the probability of success

---

## Core Content

### The Factoring Problem

**Integer Factoring:** Given composite $N$, find non-trivial factors.

**Cryptographic Relevance:** RSA encryption security relies on factoring being hard.

**Classical Complexity:** Best known: General Number Field Sieve
$$O\left(\exp\left(c \cdot n^{1/3} (\log n)^{2/3}\right)\right) \text{ for } n = \log_2 N$$

Sub-exponential but not polynomial!

### The Order-Finding Problem

**Definition:** Given $a$ and $N$ with $\gcd(a, N) = 1$, find the smallest positive integer $r$ such that:
$$a^r \equiv 1 \pmod{N}$$

This $r$ is called the **order** of $a$ modulo $N$.

**Example:** For $a = 2$, $N = 15$:
$2^1 = 2$, $2^2 = 4$, $2^3 = 8$, $2^4 = 16 \equiv 1 \pmod{15}$

So $r = 4$.

### The Key Theorem

**Theorem:** If we can efficiently find orders, we can efficiently factor.

**Proof Sketch:**

Given $N$, suppose $N = pq$ for primes $p, q$.

1. **Pick random $a$:** Choose $1 < a < N$ randomly.

2. **Check GCD:** If $\gcd(a, N) > 1$, we found a factor! (Lucky case)

3. **Find order:** Compute $r$ = order of $a$ mod $N$.

4. **Check conditions:**
   - If $r$ is odd: try again with new $a$
   - If $a^{r/2} \equiv -1 \pmod{N}$: try again

5. **Compute factors:** If $r$ is even and $a^{r/2} \not\equiv \pm 1 \pmod{N}$:
   $$\gcd(a^{r/2} - 1, N) \text{ or } \gcd(a^{r/2} + 1, N)$$
   gives a non-trivial factor with high probability.

### Why This Works

Since $a^r \equiv 1 \pmod{N}$, we have:
$$a^r - 1 \equiv 0 \pmod{N}$$
$$(a^{r/2} - 1)(a^{r/2} + 1) \equiv 0 \pmod{N}$$

If $r$ is even:
- $N$ divides $(a^{r/2} - 1)(a^{r/2} + 1)$
- But $N$ doesn't divide either factor alone (unless trivial case)
- So $\gcd(a^{r/2} - 1, N)$ and $\gcd(a^{r/2} + 1, N)$ are non-trivial

### Success Probability

**Theorem:** For random $a$ with $\gcd(a, N) = 1$:
$$P(\text{r even and } a^{r/2} \not\equiv -1) \geq \frac{1}{2}$$

More precisely: $\geq 1 - 2^{1-k}$ where $k$ is the number of distinct prime factors.

### Algorithm Summary

```
Factor(N):
    1. If N is even: return 2
    2. If N = a^b for some a, b: return a
    3. Pick random a with 1 < a < N
    4. If gcd(a, N) > 1: return gcd(a, N)
    5. r = FindOrder(a, N)  # THE HARD PART
    6. If r odd: goto 3
    7. If a^{r/2} ≡ -1 (mod N): goto 3
    8. return gcd(a^{r/2} - 1, N)
```

The quantum speedup comes from Step 5!

---

## Worked Examples

### Example 1: Factor N = 15

**Solution:**

1. $N = 15$ is odd, not a perfect power
2. Pick $a = 7$, $\gcd(7, 15) = 1$
3. Find order of 7 mod 15:
   - $7^1 = 7$
   - $7^2 = 49 \equiv 4 \pmod{15}$
   - $7^3 = 343 \equiv 13 \pmod{15}$
   - $7^4 = 2401 \equiv 1 \pmod{15}$
   - So $r = 4$ (even!)
4. $a^{r/2} = 7^2 = 49 \equiv 4 \pmod{15}$
   - $4 \not\equiv -1 \equiv 14 \pmod{15}$ ✓
5. $\gcd(4 - 1, 15) = \gcd(3, 15) = 3$ ✓

**Factor found:** 3 (and $15/3 = 5$)

### Example 2: Factor N = 21

**Solution:**

1. Pick $a = 2$, $\gcd(2, 21) = 1$
2. Order of 2 mod 21:
   - $2^1 = 2$, $2^2 = 4$, $2^3 = 8$, $2^4 = 16$, $2^5 = 32 \equiv 11$, $2^6 = 64 \equiv 1$
   - $r = 6$ (even!)
3. $2^3 = 8$, $8 \not\equiv -1 \equiv 20 \pmod{21}$ ✓
4. $\gcd(8 - 1, 21) = \gcd(7, 21) = 7$ ✓

**Factor found:** 7 (and $21/7 = 3$)

### Example 3: Failure Case

**Solution:** Try $a = 8$ for $N = 15$:
- $8^2 = 64 \equiv 4 \pmod{15}$
- $8^4 \equiv 4^2 = 16 \equiv 1 \pmod{15}$
- $r = 4$, $a^{r/2} = 8^2 = 64 \equiv 4 \not\equiv -1$ ✓
- $\gcd(4-1, 15) = 3$ ✓

Actually works! Let's try $a = 4$:
- $4^2 = 16 \equiv 1 \pmod{15}$
- $r = 2$, $a^{r/2} = 4 \not\equiv -1$ ✓
- $\gcd(4-1, 15) = 3$ ✓

Also works! Finding a true failure case:

Try $a = 11$ for $N = 15$:
- $11 \equiv -4 \pmod{15}$
- $11^2 = 121 \equiv 1 \pmod{15}$
- $r = 2$, $a^{r/2} = 11 \equiv -4 \not\equiv -1$ ✓
- $\gcd(11-1, 15) = \gcd(10, 15) = 5$ ✓

Still works! The failure probability is low for small examples.

---

## Practice Problems

### Problem 1
Factor $N = 35$ using the order-finding reduction with $a = 2$.

### Problem 2
For $N = 91$, verify that $a = 3$ has order 6 and find a factor.

### Problem 3
Prove that if $r$ is odd, we cannot factor using this method.

### Problem 4
Calculate the exact success probability for $N = pq$ with distinct odd primes.

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Order definition | $a^r \equiv 1 \pmod{N}$, $r$ minimal |
| Factor extraction | $\gcd(a^{r/2} \pm 1, N)$ |
| Success probability | $\geq 1/2$ per attempt |

### Key Takeaways

1. **Factoring reduces to order-finding** efficiently
2. **Classical order-finding** is as hard as factoring
3. **Quantum order-finding** (via QPE) is polynomial
4. **High success probability** after few attempts
5. **This is why Shor's algorithm threatens RSA**

---

## Daily Checklist

- [ ] I understand the order-finding problem
- [ ] I can prove the reduction from factoring
- [ ] I know why r even and $a^{r/2} \neq -1$ is needed
- [ ] I can factor small numbers using this method
- [ ] I understand the success probability analysis

---

*Next: Day 611 - Number Theory Background*
