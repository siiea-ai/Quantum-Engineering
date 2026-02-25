# Day 613: Continued Fractions

## Overview

**Day 613** | Week 88, Day 4 | Month 22 | Quantum Algorithms I

Today we master continued fractions, the classical post-processing step that extracts the period $r$ from QPE measurements in Shor's algorithm. This elegant mathematical technique converts approximate rational numbers to their exact form.

---

## Learning Objectives

1. Understand continued fraction representation
2. Compute continued fraction expansions
3. Calculate convergents efficiently
4. Apply continued fractions to extract periods
5. Analyze when the method succeeds

---

## Core Content

### Motivation from QPE

QPE outputs measurement $m$ such that:
$$\frac{m}{2^n} \approx \frac{s}{r}$$

where $s$ is random in $\{0, 1, \ldots, r-1\}$ and $r$ is the period we seek.

**Problem:** Given $m/2^n$, find $r$ (the denominator of $s/r$).

**Solution:** Continued fractions!

### Continued Fraction Representation

Any real number $x$ can be written as:
$$x = a_0 + \cfrac{1}{a_1 + \cfrac{1}{a_2 + \cfrac{1}{a_3 + \cdots}}}$$

**Notation:** $x = [a_0; a_1, a_2, a_3, \ldots]$

**The $a_i$ values:**
- $a_0 = \lfloor x \rfloor$ (integer part)
- $a_i \geq 1$ for $i \geq 1$

### Computing Continued Fractions

**Algorithm:**
```
Given x:
1. a_0 = floor(x)
2. If x = a_0: stop (x is integer)
3. x_1 = 1/(x - a_0)
4. Repeat: a_i = floor(x_i), x_{i+1} = 1/(x_i - a_i)
```

**Example:** $x = 3.245$

$a_0 = 3$, remainder $= 0.245$
$x_1 = 1/0.245 \approx 4.082$, $a_1 = 4$
$x_2 = 1/0.082 \approx 12.2$, $a_2 = 12$
...

So $3.245 \approx [3; 4, 12, \ldots]$

### Convergents

The **$k$-th convergent** $p_k/q_k$ is obtained by truncating:
$$\frac{p_k}{q_k} = [a_0; a_1, \ldots, a_k]$$

**Recurrence relations:**
$$p_{-1} = 1, \quad p_0 = a_0, \quad p_k = a_k p_{k-1} + p_{k-2}$$
$$q_{-1} = 0, \quad q_0 = 1, \quad q_k = a_k q_{k-1} + q_{k-2}$$

### Key Theorem for Shor's Algorithm

**Theorem:** If $\left|\frac{s}{r} - \frac{m}{2^n}\right| \leq \frac{1}{2 \cdot 2^n}$ and $r < 2^n$, then $s/r$ is a convergent of $m/2^n$.

**Significance:** The period $r$ appears as a denominator of some convergent!

### Algorithm for Period Extraction

Given QPE output $m$ with $n$ bits:

1. Compute continued fraction expansion of $m/2^n$
2. For each convergent $p_k/q_k$ with $q_k < N$:
   - Test if $a^{q_k} \equiv 1 \pmod{N}$
   - If yes: $r$ divides $q_k$, so $q_k$ is a candidate
3. Return smallest valid $q_k$

### Properties of Convergents

**Best rational approximations:** Convergents are the best rational approximations with bounded denominator.

**Error bound:**
$$\left|\frac{p_k}{q_k} - x\right| < \frac{1}{q_k q_{k+1}} < \frac{1}{q_k^2}$$

**Determinant property:**
$$p_k q_{k-1} - p_{k-1} q_k = (-1)^{k-1}$$

### When Multiple Candidates Arise

If $\gcd(s, r) = d > 1$, the convergent gives $s/r = (s/d)/(r/d)$.

We get $r/d$ instead of $r$!

**Solution:** Repeat with new random $a$, take LCM of candidates.

---

## Worked Examples

### Example 1: Continued Fraction of 31/13

**Solution:**

$31/13 = 2 + 5/13$, so $a_0 = 2$

$13/5 = 2 + 3/5$, so $a_1 = 2$

$5/3 = 1 + 2/3$, so $a_2 = 1$

$3/2 = 1 + 1/2$, so $a_3 = 1$

$2/1 = 2$, so $a_4 = 2$

**Result:** $31/13 = [2; 2, 1, 1, 2]$

**Convergents:**

| $k$ | $a_k$ | $p_k$ | $q_k$ | $p_k/q_k$ |
|-----|-------|-------|-------|-----------|
| -1  | -     | 1     | 0     | -         |
| 0   | 2     | 2     | 1     | 2         |
| 1   | 2     | 5     | 2     | 5/2       |
| 2   | 1     | 7     | 3     | 7/3       |
| 3   | 1     | 12    | 5     | 12/5      |
| 4   | 2     | 31    | 13    | 31/13     |

### Example 2: Extract Period from QPE

QPE with $n = 4$ bits outputs $m = 11$. We have $N = 15$.

Approximate: $m/2^n = 11/16 = 0.6875$

**Continued fraction of 11/16:**

$11/16 = 0 + 11/16$, $a_0 = 0$

$16/11 = 1 + 5/11$, $a_1 = 1$

$11/5 = 2 + 1/5$, $a_2 = 2$

$5/1 = 5$, $a_3 = 5$

So $11/16 = [0; 1, 2, 5]$

**Convergents:**

| $k$ | $a_k$ | $p_k$ | $q_k$ |
|-----|-------|-------|-------|
| 0   | 0     | 0     | 1     |
| 1   | 1     | 1     | 1     |
| 2   | 2     | 2     | 3     |
| 3   | 5     | 11    | 16    |

**Testing (for $a = 7$, $N = 15$):**

- $q_1 = 1$: $7^1 = 7 \not\equiv 1 \pmod{15}$
- $q_2 = 3$: $7^3 = 343 = 22 \times 15 + 13 \equiv 13 \not\equiv 1$
- Try multiples: $2 \times q_2 = 6$? No...

Actually, $p_2/q_2 = 2/3$ suggests $s/r = 2/3$, so $r = 3$?

Check: $7^3 \equiv 13$, not 1. So $r \neq 3$.

The actual order of 7 mod 15 is 4. We need better measurement!

### Example 3: Successful Period Finding

QPE outputs $m = 4$ with $n = 4$ bits for $a = 2$, $N = 15$.

$m/2^n = 4/16 = 1/4$

**Continued fraction of 1/4:**

$1/4 = [0; 4]$

Convergents: $0/1$, $1/4$

**Test:** $q = 4$

$2^4 = 16 \equiv 1 \pmod{15}$ ✓

**Period found:** $r = 4$

---

## Practice Problems

### Problem 1: Basic Continued Fractions
Compute the continued fraction expansion and all convergents for:
(a) $7/5$
(b) $41/29$
(c) $\sqrt{2}$ (first 5 terms)

### Problem 2: Period Extraction
QPE outputs $m = 85$ with $n = 8$ bits (so $2^n = 256$). Find the convergents of $85/256$ and identify candidate periods $< 20$.

### Problem 3: Failure Analysis
If the true phase is $\phi = 3/8$ and QPE outputs $m = 96$ (with $2^n = 256$), show that 8 appears as a convergent denominator.

### Problem 4: Multiple Attempts
Explain why multiple QPE runs help when $\gcd(s, r) > 1$.

---

## Computational Lab

```python
"""
Day 613: Continued Fractions for Shor's Algorithm
"""

import numpy as np
from fractions import Fraction
from typing import List, Tuple

def continued_fraction_expansion(numerator: int, denominator: int,
                                  max_terms: int = 20) -> List[int]:
    """
    Compute continued fraction expansion [a_0; a_1, a_2, ...].

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction
        max_terms: Maximum number of terms to compute

    Returns:
        List of continued fraction coefficients
    """
    cf = []
    n, d = numerator, denominator

    for _ in range(max_terms):
        if d == 0:
            break
        q = n // d
        cf.append(q)
        n, d = d, n - q * d

    return cf

def compute_convergents(cf: List[int]) -> List[Tuple[int, int]]:
    """
    Compute convergents from continued fraction expansion.

    Args:
        cf: Continued fraction coefficients [a_0; a_1, ...]

    Returns:
        List of (p_k, q_k) convergent pairs
    """
    convergents = []

    # Initialize: p_{-1} = 1, q_{-1} = 0, p_0 = a_0, q_0 = 1
    p_prev, q_prev = 1, 0
    p_curr, q_curr = cf[0], 1
    convergents.append((p_curr, q_curr))

    for k in range(1, len(cf)):
        a_k = cf[k]
        p_next = a_k * p_curr + p_prev
        q_next = a_k * q_curr + q_prev
        convergents.append((p_next, q_next))
        p_prev, q_prev = p_curr, q_curr
        p_curr, q_curr = p_next, q_next

    return convergents

def extract_period_candidates(m: int, n_bits: int, N: int) -> List[int]:
    """
    Extract period candidates from QPE measurement.

    Args:
        m: QPE measurement outcome
        n_bits: Number of ancilla qubits used
        N: Number being factored

    Returns:
        List of candidate periods (convergent denominators < N)
    """
    two_n = 2 ** n_bits

    # Compute continued fraction of m/2^n
    cf = continued_fraction_expansion(m, two_n)
    print(f"Measurement: m = {m}, 2^n = {two_n}")
    print(f"Fraction: {m}/{two_n} = {m/two_n:.6f}")
    print(f"Continued fraction: {cf}")

    # Get convergents
    convergents = compute_convergents(cf)
    print(f"\nConvergents:")

    candidates = []
    for i, (p, q) in enumerate(convergents):
        print(f"  k={i}: {p}/{q} = {p/q if q > 0 else 'inf':.6f}")
        if 0 < q < N:
            candidates.append(q)

    return candidates

def test_period_candidate(a: int, r_candidate: int, N: int) -> bool:
    """
    Test if r_candidate is a valid period for a mod N.

    Args:
        a: Base for modular exponentiation
        r_candidate: Candidate period
        N: Modulus

    Returns:
        True if a^r_candidate ≡ 1 (mod N)
    """
    return pow(a, r_candidate, N) == 1

def find_period_from_qpe(m: int, n_bits: int, a: int, N: int) -> int:
    """
    Complete period-finding from QPE measurement.

    Args:
        m: QPE measurement
        n_bits: Number of ancilla qubits
        a: Base
        N: Modulus

    Returns:
        Found period r, or -1 if not found
    """
    print(f"\n{'='*50}")
    print(f"Finding period of {a} mod {N}")
    print(f"QPE output: m = {m} (n = {n_bits} bits)")
    print(f"{'='*50}")

    candidates = extract_period_candidates(m, n_bits, N)

    print(f"\nTesting candidates < {N}:")
    for r in candidates:
        result = test_period_candidate(a, r, N)
        status = "✓ VALID" if result else "✗"
        print(f"  r = {r}: {a}^{r} mod {N} = {pow(a, r, N)} {status}")

        if result:
            # Check if this is the smallest period
            for divisor in range(1, r):
                if r % divisor == 0:
                    if pow(a, divisor, N) == 1:
                        print(f"    Note: {divisor} is a smaller period")
                        return divisor
            return r

    # Also test multiples of candidates
    print(f"\nTesting small multiples of candidates:")
    for r in candidates:
        for mult in range(2, 5):
            r_mult = r * mult
            if r_mult < N:
                result = test_period_candidate(a, r_mult, N)
                if result:
                    print(f"  r = {r} × {mult} = {r_mult}: VALID")
                    return r_mult

    return -1

# Demonstration
print("CONTINUED FRACTIONS IN SHOR'S ALGORITHM")
print("="*50)

# Example 1: Basic continued fraction
print("\n1. Continued Fraction of 31/13:")
cf = continued_fraction_expansion(31, 13)
print(f"   31/13 = {cf}")
convs = compute_convergents(cf)
print("   Convergents:", [(p, q) for p, q in convs])

# Example 2: From QPE measurement for N=15
print("\n2. Period Finding for N=15, a=7:")
# If QPE gives m=12 with n=4 bits (phase ≈ 3/4)
find_period_from_qpe(m=12, n_bits=4, a=7, N=15)

# Example 3: Successful case for a=2
print("\n3. Period Finding for N=15, a=2:")
# QPE should give m ≈ 2^n × s/4 for some s
# If s=1, m ≈ 4 (for n=4)
find_period_from_qpe(m=4, n_bits=4, a=2, N=15)

# Example 4: Analysis of golden ratio
print("\n4. Golden Ratio φ = (1+√5)/2:")
# Golden ratio has the simplest continued fraction: [1; 1, 1, 1, ...]
phi = (1 + np.sqrt(5)) / 2
# Approximate as fraction
from fractions import Fraction
phi_frac = Fraction(phi).limit_denominator(1000)
cf_phi = continued_fraction_expansion(phi_frac.numerator, phi_frac.denominator)
print(f"   φ ≈ {phi_frac} = {cf_phi}")

# Example 5: Success probability analysis
print("\n5. Success Probability Analysis:")
print("   Simulating QPE outcomes for a=2, N=15, r=4")

n_bits = 8
two_n = 2 ** n_bits
true_r = 4

success_count = 0
trials = 100

for trial in range(trials):
    # Random s from 0 to r-1
    s = np.random.randint(0, true_r)

    # True phase
    true_phase = s / true_r

    # QPE measurement (with some random error)
    m = int(round(two_n * true_phase))

    # Add small random error to simulate finite precision
    m = (m + np.random.randint(-1, 2)) % two_n

    # Try to extract period
    cf = continued_fraction_expansion(m, two_n)
    convs = compute_convergents(cf)

    found = False
    for p, q in convs:
        if q > 0 and q < 15:
            if pow(2, q, 15) == 1:
                if q == true_r or true_r % q == 0:
                    found = True
                    break

    if found:
        success_count += 1

print(f"   Success rate: {success_count}/{trials} = {success_count/trials:.1%}")

# Example 6: Visualizing convergent approximations
print("\n6. Convergent Approximation Quality:")
target = Fraction(3, 7)  # Phase s/r = 3/7
n_bits = 10
m = int(round(2**n_bits * float(target)))
print(f"   Target: {target} = {float(target):.6f}")
print(f"   QPE output: m = {m}, m/2^{n_bits} = {m/2**n_bits:.6f}")

cf = continued_fraction_expansion(m, 2**n_bits)
convs = compute_convergents(cf)

print("   Convergent errors:")
for i, (p, q) in enumerate(convs):
    if q > 0:
        error = abs(float(target) - p/q)
        print(f"     {p}/{q}: error = {error:.2e}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| CF notation | $x = [a_0; a_1, a_2, \ldots]$ |
| Convergent recurrence | $p_k = a_k p_{k-1} + p_{k-2}$ |
| Error bound | $\|p_k/q_k - x\| < 1/q_k^2$ |
| Shor condition | $\|s/r - m/2^n\| \leq 1/(2 \cdot 2^n)$ |

### Key Takeaways

1. **Continued fractions** represent reals as nested reciprocals
2. **Convergents** are best rational approximations
3. **QPE outputs** approximate $s/r$; CF extracts $r$
4. **Period candidates** appear as convergent denominators
5. **Multiple runs** needed when $\gcd(s, r) > 1$

---

## Daily Checklist

- [ ] I can compute continued fraction expansions
- [ ] I can calculate convergents using recurrence
- [ ] I understand why convergents are best approximations
- [ ] I can extract period candidates from QPE output
- [ ] I know when continued fractions might give $r/d$ instead of $r$

---

*Next: Day 614 - Full Shor's Algorithm*
