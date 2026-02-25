# Day 612: Quantum Period-Finding

## Overview

**Day 612** | Week 88, Day 3 | Month 22 | Quantum Algorithms I

Today we implement quantum period-finding using QPE, the quantum heart of Shor's algorithm. We'll see how to extract the period $r$ from the eigenvalues of the modular exponentiation operator.

---

## Learning Objectives

1. Define the modular exponentiation unitary
2. Identify eigenstates and eigenvalues
3. Apply QPE to extract period information
4. Understand the superposition of eigenstates
5. Analyze measurement outcomes

---

## Core Content

### The Modular Exponentiation Operator

For fixed $a$ with $\gcd(a, N) = 1$, define:
$$U_a|x\rangle = |ax \mod N\rangle$$

This is unitary on the computational basis $|0\rangle, |1\rangle, \ldots, |N-1\rangle$.

### Eigenstates of $U_a$

The eigenstates are:
$$|u_s\rangle = \frac{1}{\sqrt{r}}\sum_{j=0}^{r-1} e^{-2\pi ijs/r}|a^j \mod N\rangle$$

for $s = 0, 1, \ldots, r-1$.

**Verification:**
$$U_a|u_s\rangle = \frac{1}{\sqrt{r}}\sum_{j=0}^{r-1} e^{-2\pi ijs/r}|a^{j+1} \mod N\rangle = e^{2\pi is/r}|u_s\rangle$$

### Eigenvalues

$$U_a|u_s\rangle = e^{2\pi is/r}|u_s\rangle$$

The phase is $\phi_s = s/r$.

**Key observation:** QPE extracts $s/r$, and from this we can find $r$!

### The Superposition Trick

We don't know the eigenstates, but we know:
$$|1\rangle = \frac{1}{\sqrt{r}}\sum_{s=0}^{r-1}|u_s\rangle$$

Running QPE on $|1\rangle$ projects onto a random eigenstate, measuring $s/r$ for random $s$.

### The QPE Circuit for Period-Finding

```
|0⟩^⊗2n ─[H^⊗2n]──[CU_a^{2^{2n-1}}]──...──[CU_a^1]──[QFT^{-1}]── Measure
                       │                      │
|1⟩      ─────────────U_a────────────────────U_a─────────────────
```

Use $2n$ ancilla qubits for $n$-bit $N$ to ensure sufficient precision.

### Measurement Outcome

QPE outputs an approximation to $s/r$ where:
- $s$ is random from $\{0, 1, \ldots, r-1\}$
- Output is $m$ where $m/2^{2n} \approx s/r$

From $m/2^{2n}$, we recover $s/r$ using continued fractions (next day).

### Controlled Modular Exponentiation

The cost of $CU_a^{2^k}$ is crucial:
- Compute $a^{2^k} \mod N$ classically (fast via repeated squaring)
- Implement $U_{a^{2^k}}|x\rangle = |a^{2^k} \cdot x \mod N\rangle$

This is the most expensive part of Shor's algorithm!

---

## Worked Examples

### Example 1: Eigenstates for a=2, N=15

$r = 4$ (order of 2 mod 15).

Orbit: $\{1, 2, 4, 8\}$

**Eigenstates:**

$|u_0\rangle = \frac{1}{2}(|1\rangle + |2\rangle + |4\rangle + |8\rangle)$

$|u_1\rangle = \frac{1}{2}(|1\rangle - i|2\rangle - |4\rangle + i|8\rangle)$

$|u_2\rangle = \frac{1}{2}(|1\rangle - |2\rangle + |4\rangle - |8\rangle)$

$|u_3\rangle = \frac{1}{2}(|1\rangle + i|2\rangle - |4\rangle - i|8\rangle)$

**Eigenvalues:** $1, i, -1, -i$ (phases $0, 1/4, 1/2, 3/4$)

### Example 2: QPE Output

With $2n = 4$ ancillas for $N = 15$, measuring after QPE on $|1\rangle$:

If we project to $|u_1\rangle$ (phase $1/4$):
- QPE outputs $m$ where $m/16 \approx 1/4$
- So $m \approx 4$
- Measure $|0100\rangle$, estimate $s/r = 4/16 = 1/4$

From $1/4$, the denominator 4 is a candidate for $r$. ✓

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Eigenstate | $\|u_s\rangle = \frac{1}{\sqrt{r}}\sum_j e^{-2\pi ijs/r}\|a^j\rangle$ |
| Eigenvalue | $e^{2\pi is/r}$ |
| Superposition | $\|1\rangle = \frac{1}{\sqrt{r}}\sum_s \|u_s\rangle$ |
| QPE output | $m/2^{2n} \approx s/r$ |

### Key Takeaways

1. **Modular exponentiation** defines the relevant unitary
2. **Eigenstates** have phases $s/r$ encoding the period
3. **QPE on $|1\rangle$** samples from the eigenvalue spectrum
4. **Measurement** gives approximation to $s/r$
5. **Next step:** Extract $r$ from the measurement via continued fractions

---

*Next: Day 613 - Continued Fractions*
