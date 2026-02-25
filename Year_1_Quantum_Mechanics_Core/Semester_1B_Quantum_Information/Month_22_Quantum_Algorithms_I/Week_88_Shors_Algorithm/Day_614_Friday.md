# Day 614: Full Shor's Algorithm

## Overview

**Day 614** | Week 88, Day 5 | Month 22 | Quantum Algorithms I

Today we integrate all components into the complete Shor's factoring algorithm. We trace through the full algorithm from input $N$ to output factors, combining classical number theory, quantum period-finding, and continued fractions.

---

## Learning Objectives

1. Integrate all Shor's algorithm components
2. Trace through complete execution
3. Implement the full algorithm
4. Analyze success probability
5. Understand practical considerations

---

## Core Content

### The Complete Algorithm

**Input:** Composite odd integer $N$ (not a prime power)

**Output:** A non-trivial factor of $N$

```
Shor's Algorithm:
1. Classical Preprocessing:
   a. If N is even: return 2
   b. If N = a^b for integers a, b ≥ 2: return a
   c. Pick random 1 < a < N

2. Classical Check:
   d. Compute g = gcd(a, N)
   e. If g > 1: return g  (Lucky!)

3. Quantum Period-Finding:
   f. Use QPE to find period r of a mod N
      - Prepare |0⟩^⊗2n ⊗ |1⟩
      - Apply H^⊗2n to first register
      - Apply controlled-U_a^{2^j} operations
      - Apply QFT^{-1}
      - Measure to get m

4. Classical Post-Processing:
   g. Use continued fractions on m/2^{2n}
   h. Extract candidate r from convergent denominator
   i. Verify: a^r ≡ 1 (mod N)?

5. Factor Extraction:
   j. If r is odd: restart at step 1c
   k. If a^{r/2} ≡ -1 (mod N): restart at step 1c
   l. Compute:
      - g1 = gcd(a^{r/2} - 1, N)
      - g2 = gcd(a^{r/2} + 1, N)
   m. Return non-trivial factor g1 or g2
```

### The Quantum Circuit

```
|0⟩ ─[H]──●────────────────●─────...───●────[QFT^{-1}]── Measure → m₁
          │                │           │
|0⟩ ─[H]──┼────●───────────┼───────────┼────[QFT^{-1}]── Measure → m₂
          │    │           │           │
⋮         ⋮    ⋮           ⋮           ⋮
|0⟩ ─[H]──┼────┼───────────┼────●──────┼────[QFT^{-1}]── Measure → m_{2n}
          │    │           │    │      │
|1⟩ ─────[U_a^{2^{2n-1}}]─[U_a^{2^{2n-2}}]...─[U_a^1]────────────────
```

**Registers:**
- Ancilla: $2n$ qubits (where $n = \lceil \log_2 N \rceil$)
- Work: $n$ qubits for $|x \mod N\rangle$

### Controlled Modular Exponentiation

The controlled-$U_a^{2^k}$ operation is the most expensive part:

$$CU_a^{2^k}: |c\rangle|x\rangle \mapsto |c\rangle|a^{c \cdot 2^k} \cdot x \mod N\rangle$$

**Implementation strategy:**
1. Precompute $b_k = a^{2^k} \mod N$ classically
2. Implement controlled multiplication by $b_k$

**Circuit depth:** $O(n^2)$ elementary gates per controlled operation

### Success Probability Analysis

**Phase 1:** Random $a$ coprime to $N$
- Probability: $\phi(N)/N \approx 1$ for large $N$

**Phase 2:** Order $r$ is even
- Probability: $\geq 1 - 2^{1-k}$ where $k$ = number of prime factors
- For $N = pq$: probability $\geq 1/2$

**Phase 3:** $a^{r/2} \not\equiv -1 \pmod{N}$
- Probability: $\geq 1/2$

**Phase 4:** QPE gives good approximation
- Probability: $\geq 4/\pi^2 \approx 40\%$

**Combined:** $P(\text{success per attempt}) \geq \frac{1}{2} \cdot \frac{1}{2} \cdot 0.4 \approx 10\%$

After $O(\log N)$ attempts: probability $\to 1$

### The Eigenvalue Trick Revisited

Why does QPE on $|1\rangle$ work?

The eigenstates of $U_a$ are:
$$|u_s\rangle = \frac{1}{\sqrt{r}} \sum_{j=0}^{r-1} e^{-2\pi ijs/r} |a^j \mod N\rangle$$

And crucially:
$$|1\rangle = \frac{1}{\sqrt{r}} \sum_{s=0}^{r-1} |u_s\rangle$$

QPE on superposition of eigenstates:
- Projects to random $|u_s\rangle$
- Outputs phase $s/r$ for random $s$

### Complete State Evolution

**Initial:** $|0\rangle^{\otimes 2n} \otimes |1\rangle$

**After Hadamards:**
$$\frac{1}{\sqrt{2^{2n}}} \sum_{j=0}^{2^{2n}-1} |j\rangle \otimes |1\rangle$$

**After controlled-$U_a$:**
$$\frac{1}{\sqrt{2^{2n}}} \sum_{j=0}^{2^{2n}-1} |j\rangle \otimes |a^j \mod N\rangle$$

**Using eigenstate decomposition:**
$$= \frac{1}{\sqrt{r \cdot 2^{2n}}} \sum_{s=0}^{r-1} \sum_{j=0}^{2^{2n}-1} e^{2\pi ijs/r} |j\rangle \otimes |u_s\rangle$$

**After QFT$^{-1}$:** Peaks at $|j\rangle$ where $j/2^{2n} \approx s/r$

---

## Worked Examples

### Example 1: Factor N = 15 with a = 7

**Step 1:** $N = 15$ is odd, not a prime power ✓

**Step 2:** Choose $a = 7$, $\gcd(7, 15) = 1$ ✓

**Step 3:** Quantum period-finding
- True order of 7 mod 15 is $r = 4$
- QPE on $U_7$ with $2n = 8$ ancillas
- Suppose we measure $m = 64$
- Phase estimate: $64/256 = 1/4 = s/r$ with $s = 1$

**Step 4:** Continued fractions
- $64/256 = 1/4$ exactly
- Convergent: $1/4$, so $r = 4$

**Step 5:** Factor extraction
- $r = 4$ is even ✓
- $7^{4/2} = 7^2 = 49 \equiv 4 \pmod{15}$
- $4 \not\equiv -1 \equiv 14 \pmod{15}$ ✓
- $\gcd(4 - 1, 15) = \gcd(3, 15) = 3$ ✓
- $\gcd(4 + 1, 15) = \gcd(5, 15) = 5$ ✓

**Factors found:** $15 = 3 \times 5$

### Example 2: Factor N = 21 with a = 2

**Step 1:** $N = 21$ is odd, not a prime power ✓

**Step 2:** Choose $a = 2$, $\gcd(2, 21) = 1$ ✓

**Step 3:** Quantum period-finding
- Order of 2 mod 21 is $r = 6$
- QPE with $2n = 10$ ancillas ($2^{10} = 1024$)
- Suppose measurement gives $m = 171$ (near $1024 \cdot 1/6 \approx 170.67$)

**Step 4:** Continued fractions of $171/1024$
- $171/1024 \approx 0.167$
- CF: $[0; 5, 1, 23, ...]$
- Convergent $1/6$ gives $r = 6$ ✓

**Step 5:** Factor extraction
- $r = 6$ is even ✓
- $2^3 = 8 \not\equiv -1 \equiv 20 \pmod{21}$ ✓
- $\gcd(8 - 1, 21) = \gcd(7, 21) = 7$ ✓

**Factor found:** 7 (and $21/7 = 3$)

### Example 3: Failure Case

**N = 15, a = 4:**

- Order of 4 mod 15: $4^2 = 16 \equiv 1$, so $r = 2$
- $r = 2$ is even ✓
- $4^1 = 4 \not\equiv -1$ ✓
- $\gcd(4 - 1, 15) = 3$ ✓

Actually succeeds! Let's try a true failure:

**N = 15, a = 14:**

- Order of 14 mod 15: $14 \equiv -1$, so $14^2 = 1$, $r = 2$
- $14^{r/2} = 14 \equiv -1 \pmod{15}$
- **FAILURE:** $a^{r/2} \equiv -1$

Must restart with new random $a$.

---

## Practice Problems

### Problem 1: Complete Execution
Trace through Shor's algorithm for $N = 35$ with $a = 3$. Find the period and extract factors.

### Problem 2: QPE Requirements
For $N = 2047 = 23 \times 89$, determine:
(a) Minimum ancilla qubits needed
(b) Number of controlled-$U$ operations
(c) Estimated success probability per attempt

### Problem 3: Failure Analysis
For $N = pq$ with distinct odd primes, prove that the probability of $a^{r/2} \equiv -1$ is at most $1/2$.

### Problem 4: Gate Count
Estimate the number of elementary gates needed to factor a 10-bit number.

---

## Computational Lab

```python
"""
Day 614: Complete Shor's Algorithm Implementation
"""

import numpy as np
from fractions import Fraction
from typing import Tuple, Optional, List
from math import gcd, isqrt

def is_prime_power(N: int) -> Tuple[bool, int, int]:
    """
    Check if N = a^b for integers a, b ≥ 2.

    Returns:
        (is_prime_power, base, exponent)
    """
    if N < 2:
        return False, 0, 0

    for b in range(2, N.bit_length() + 1):
        a = int(round(N ** (1/b)))
        for candidate in [a-1, a, a+1]:
            if candidate > 1 and candidate ** b == N:
                return True, candidate, b

    return False, 0, 0

def classical_order(a: int, N: int) -> int:
    """
    Compute order of a mod N classically (for small examples).
    """
    if gcd(a, N) != 1:
        return -1

    order = 1
    current = a % N

    while current != 1 and order < N:
        current = (current * a) % N
        order += 1

    return order if current == 1 else -1

def continued_fraction_convergents(num: int, den: int) -> List[Tuple[int, int]]:
    """
    Compute convergents of num/den.
    """
    convergents = []

    # Continued fraction computation
    p_prev, p_curr = 1, 0
    q_prev, q_curr = 0, 1

    while den != 0:
        a = num // den
        num, den = den, num - a * den

        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev

        if q_curr > 0:
            convergents.append((p_curr, q_curr))

    return convergents

def simulate_qpe_measurement(s: int, r: int, n_bits: int) -> int:
    """
    Simulate QPE measurement for phase s/r.

    Returns measurement m where m/2^n ≈ s/r
    """
    two_n = 2 ** n_bits
    exact = two_n * s / r

    # Add small random noise to simulate finite precision
    noise = np.random.randn() * 0.5
    m = int(round(exact + noise)) % two_n

    return m

def extract_period_from_measurement(m: int, n_bits: int, N: int, a: int) -> Optional[int]:
    """
    Extract period from QPE measurement using continued fractions.
    """
    two_n = 2 ** n_bits

    convergents = continued_fraction_convergents(m, two_n)

    for p, q in convergents:
        if 0 < q < N:
            # Test if q is the period or a divisor
            if pow(a, q, N) == 1:
                return q

            # Try small multiples
            for mult in range(2, 10):
                r_candidate = q * mult
                if r_candidate < N and pow(a, r_candidate, N) == 1:
                    return r_candidate

    return None

def shors_algorithm(N: int, verbose: bool = True) -> Tuple[int, int]:
    """
    Complete Shor's algorithm (simulated).

    Args:
        N: Number to factor
        verbose: Print detailed output

    Returns:
        Tuple of factors (p, q) or (1, N) if failed
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"SHOR'S ALGORITHM: Factoring N = {N}")
        print(f"{'='*60}")

    # Step 1a: Check if even
    if N % 2 == 0:
        if verbose:
            print(f"N is even. Factor: 2")
        return (2, N // 2)

    # Step 1b: Check if prime power
    is_pp, base, exp = is_prime_power(N)
    if is_pp:
        if verbose:
            print(f"N = {base}^{exp} is a prime power. Factor: {base}")
        return (base, N // base)

    # Determine QPE parameters
    n = N.bit_length()
    n_ancilla = 2 * n  # Standard choice

    if verbose:
        print(f"N has {n} bits, using {n_ancilla} ancilla qubits")

    max_attempts = 20

    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"\n--- Attempt {attempt} ---")

        # Step 1c: Pick random a
        a = np.random.randint(2, N)
        if verbose:
            print(f"Random a = {a}")

        # Step 2: Check GCD
        g = gcd(a, N)
        if g > 1:
            if verbose:
                print(f"Lucky! gcd({a}, {N}) = {g}")
            return (g, N // g)

        # Step 3: Quantum period-finding (simulated)
        if verbose:
            print(f"gcd({a}, {N}) = 1, proceeding to period-finding")

        # Get true order for simulation
        true_r = classical_order(a, N)
        if verbose:
            print(f"True order r = {true_r}")

        # Simulate QPE measurement
        s = np.random.randint(0, true_r)
        m = simulate_qpe_measurement(s, true_r, n_ancilla)
        if verbose:
            print(f"QPE: s = {s}, measurement m = {m}")
            print(f"Estimated phase: {m}/{2**n_ancilla} ≈ {m/2**n_ancilla:.6f}")
            print(f"True phase: {s}/{true_r} = {s/true_r:.6f}")

        # Step 4: Continued fractions
        r = extract_period_from_measurement(m, n_ancilla, N, a)

        if r is None:
            if verbose:
                print("Failed to extract period, retrying...")
            continue

        if verbose:
            print(f"Extracted period r = {r}")

        # Step 5j: Check if r is odd
        if r % 2 == 1:
            if verbose:
                print(f"r = {r} is odd, retrying...")
            continue

        # Step 5k: Check a^{r/2} ≡ -1
        half_power = pow(a, r // 2, N)
        if verbose:
            print(f"a^(r/2) = {a}^{r//2} = {half_power} mod {N}")

        if half_power == N - 1:
            if verbose:
                print(f"a^(r/2) ≡ -1 (mod N), retrying...")
            continue

        # Step 5l,m: Extract factors
        g1 = gcd(half_power - 1, N)
        g2 = gcd(half_power + 1, N)

        if verbose:
            print(f"gcd({half_power} - 1, {N}) = {g1}")
            print(f"gcd({half_power} + 1, {N}) = {g2}")

        if 1 < g1 < N:
            if verbose:
                print(f"\n*** SUCCESS: {N} = {g1} × {N//g1} ***")
            return (g1, N // g1)

        if 1 < g2 < N:
            if verbose:
                print(f"\n*** SUCCESS: {N} = {g2} × {N//g2} ***")
            return (g2, N // g2)

        if verbose:
            print("No non-trivial factor found, retrying...")

    if verbose:
        print("\nFailed after maximum attempts")
    return (1, N)

# Demonstration
print("COMPLETE SHOR'S ALGORITHM DEMONSTRATION")
print("="*60)

# Test cases
test_numbers = [15, 21, 33, 35, 55, 77, 91]

for N in test_numbers:
    factors = shors_algorithm(N, verbose=False)
    print(f"\nN = {N}: factors = {factors[0]} × {factors[1]}")
    assert factors[0] * factors[1] == N
    assert 1 < factors[0] < N

# Detailed example
print("\n" + "="*60)
print("DETAILED TRACE: Factoring N = 35")
shors_algorithm(35, verbose=True)

# Success rate analysis
print("\n" + "="*60)
print("SUCCESS RATE ANALYSIS")
print("="*60)

N = 77  # = 7 × 11
successes = 0
total_attempts = 0
runs = 100

for _ in range(runs):
    np.random.seed(None)
    attempts = 0
    for attempt in range(50):
        attempts += 1
        a = np.random.randint(2, N)
        if gcd(a, N) > 1:
            successes += 1
            break
        r = classical_order(a, N)
        if r % 2 == 0 and pow(a, r//2, N) != N - 1:
            successes += 1
            break
    total_attempts += attempts

print(f"N = {N}")
print(f"Success rate: {successes}/{runs} = {successes/runs:.1%}")
print(f"Average attempts: {total_attempts/runs:.2f}")

# Resource estimation
print("\n" + "="*60)
print("RESOURCE ESTIMATION FOR RSA NUMBERS")
print("="*60)

rsa_bits = [256, 512, 1024, 2048]

for bits in rsa_bits:
    n_ancilla = 2 * bits
    n_work = bits
    total_qubits = n_ancilla + n_work

    # Rough gate count estimate
    # Controlled modular multiplication: O(n^2) gates per control
    gates_per_ctrl_mult = bits ** 2
    num_ctrl_mults = n_ancilla
    total_gates = num_ctrl_mults * gates_per_ctrl_mult

    print(f"\n{bits}-bit RSA:")
    print(f"  Ancilla qubits: {n_ancilla}")
    print(f"  Work qubits: {n_work}")
    print(f"  Total qubits: ~{total_qubits}")
    print(f"  Estimated gates: ~{total_gates:.2e}")
```

---

## Summary

### Key Formulas

| Component | Formula/Description |
|-----------|---------------------|
| Period condition | $a^r \equiv 1 \pmod{N}$ |
| Factor extraction | $\gcd(a^{r/2} \pm 1, N)$ |
| Success per attempt | $\geq 10\%$ |
| Total success | $1 - 2^{-k}$ after $k$ attempts |
| QPE ancillas | $2n$ where $n = \lceil\log_2 N\rceil$ |

### Key Takeaways

1. **Shor's algorithm** combines classical and quantum steps
2. **Quantum speedup** comes from period-finding via QPE
3. **Classical post-processing** uses continued fractions
4. **Multiple attempts** boost success probability
5. **Modular exponentiation** dominates circuit complexity

---

## Daily Checklist

- [ ] I can trace through the complete algorithm
- [ ] I understand each classical and quantum step
- [ ] I know when and why the algorithm might fail
- [ ] I can estimate resource requirements
- [ ] I see the cryptographic implications

---

*Next: Day 615 - Complexity Analysis*
