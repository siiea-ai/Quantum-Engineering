# Day 605: QPE Analysis and Precision

## Overview

**Day 605** | Week 87, Day 3 | Month 22 | Quantum Algorithms I

Today we analyze the precision of Quantum Phase Estimation and understand how the number of ancilla qubits determines the accuracy of our phase estimate.

---

## Learning Objectives

1. Analyze QPE precision mathematically
2. Determine register size for desired accuracy
3. Understand the exact vs approximate cases
4. Derive the output probability distribution
5. Calculate confidence intervals for estimates

---

## Core Content

### Precision of Phase Estimation

With $n$ ancilla qubits, QPE estimates $\phi$ to $n$ bits of precision:

$$\boxed{|\phi - \tilde{\phi}| \leq \frac{1}{2^n}}$$

where $\tilde{\phi} = m/2^n$ for measured value $m$.

### Exact Case

If $2^n\phi = m$ for some integer $m$, then:
- The state before measurement is exactly $|m\rangle|\psi\rangle$
- Measurement yields $m$ with probability 1
- The estimate $\tilde{\phi} = m/2^n = \phi$ is exact

### Approximate Case

If $2^n\phi$ is not an integer, let:
- $\tilde{m} = \lfloor 2^n\phi \rfloor$ or $\lceil 2^n\phi \rceil$ (nearest integer)
- $\delta = 2^n\phi - \tilde{m}$ (fractional part)

The probability of measuring $m$ is:
$$P(m) = \frac{1}{2^{2n}}\left|\frac{1 - e^{2\pi i(2^n\phi - m)}}{1 - e^{2\pi i(2^n\phi - m)/2^n}}\right|^2$$

This simplifies to:
$$\boxed{P(m) = \frac{\sin^2(\pi(2^n\phi - m))}{2^{2n}\sin^2(\pi(2^n\phi - m)/2^n)}}$$

### Probability of Best Estimate

For $m = \tilde{m}$ (nearest integer to $2^n\phi$):
$$P(\tilde{m}) \geq \frac{4}{\pi^2} \approx 0.405$$

This is the minimum probability of getting the best n-bit approximation.

### Error Bounds

**Theorem:** The probability of measuring $m$ such that $|2^n\phi - m| \leq \epsilon$ is:
$$P(|m - 2^n\phi| \leq \epsilon) \geq 1 - \frac{1}{2(\epsilon - 1)}$$

For $\epsilon = 1$ (within 1 of true value): $P \geq 1 - 1/2(1-1) = $ undefined (need $\epsilon > 1$)

For getting within $\pm 1$ of the true value with high probability, we need extra ancilla bits.

### Adding Extra Precision

To estimate $\phi$ to $t$ bits with success probability $\geq 1 - \epsilon$:

Use $n = t + \lceil\log_2(2 + 1/(2\epsilon))\rceil$ ancilla qubits.

**Example:** For $t = 10$ bits with 99% success, need $n \approx 10 + 7 = 17$ qubits.

---

## Worked Examples

### Example 1: Precision Requirement

How many ancilla qubits for error $< 0.001$ with 95% probability?

**Solution:**

Error $< 0.001$ means $1/2^t < 0.001$, so $t \geq 10$ bits.

For 95% success: $\epsilon = 0.05$
Extra bits: $\lceil\log_2(2 + 1/(2 \cdot 0.05))\rceil = \lceil\log_2(12)\rceil = 4$

Total: $n = 10 + 4 = 14$ ancilla qubits.

### Example 2: Probability Distribution

For $\phi = 0.3$ with $n = 3$ ancillas, compute $P(m = 2)$ and $P(m = 3)$.

**Solution:**

$2^3 \cdot 0.3 = 2.4$

$P(2) = \frac{\sin^2(\pi \cdot 0.4)}{64\sin^2(\pi \cdot 0.4/8)} = \frac{\sin^2(0.4\pi)}{64\sin^2(0.05\pi)}$

$\sin(0.4\pi) \approx 0.951$, $\sin(0.05\pi) \approx 0.156$

$P(2) \approx \frac{0.904}{64 \cdot 0.024} \approx 0.59$

Similarly, $P(3) \approx 0.24$

---

## Practice Problems

### Problem 1
Calculate the minimum $n$ for estimating $\phi$ to 5 decimal places with 90% confidence.

### Problem 2
For $\phi = 1/3$ and $n = 4$, find the two most likely measurement outcomes and their probabilities.

### Problem 3
Derive the formula for $P(m)$ from the state before measurement.

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Precision | $\Delta\phi \leq 1/2^n$ |
| P(best estimate) | $\geq 4/\pi^2 \approx 0.405$ |
| Extra bits for confidence | $\lceil\log_2(2 + 1/(2\epsilon))\rceil$ |

### Key Takeaways

1. **n bits give n-bit precision** in phase estimate
2. **Exact phases** yield deterministic results
3. **Approximate phases** give probabilistic distribution
4. **Extra ancillas** increase success probability
5. **Trade-off** between precision and resources

---

*Next: Day 606 - Success Probability*
