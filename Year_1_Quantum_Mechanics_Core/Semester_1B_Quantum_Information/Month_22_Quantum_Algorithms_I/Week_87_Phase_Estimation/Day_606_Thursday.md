# Day 606: Success Probability

## Overview

**Day 606** | Week 87, Day 4 | Month 22 | Quantum Algorithms I

Today we analyze the success probability of QPE in detail, including strategies for boosting success rates and handling failure cases.

---

## Learning Objectives

1. Calculate exact success probabilities
2. Understand the sinc-squared distribution
3. Analyze failure modes
4. Learn repetition and boosting strategies
5. Apply error correction concepts to QPE

---

## Core Content

### The Success Probability Formula

For phase $\phi$ and measurement outcome $m$:

$$P(m) = \frac{\sin^2(\pi(2^n\phi - m))}{2^{2n}\sin^2(\pi(2^n\phi - m)/2^n)}$$

This is a **sinc-squared** function centered at $m = 2^n\phi$.

### Properties of the Distribution

1. **Peak:** Maximum at $m$ closest to $2^n\phi$
2. **Width:** Main lobe spans approximately $\pm 1$ around peak
3. **Tails:** Decay as $1/m^2$ far from peak
4. **Normalization:** $\sum_m P(m) = 1$

### Lower Bound on Success

**Theorem:** $P(\text{best } m) \geq \frac{4}{\pi^2} \approx 0.405$

**Proof sketch:** The minimum occurs when $2^n\phi$ is exactly halfway between integers.

### Failure Probability

Probability of being off by more than $k$ from best:
$$P(\text{error} > k) < \frac{1}{2(k-1)}$$

For $k = 1$: Need to include adjacent outcomes for high confidence.

### Repetition Strategy

Run QPE $R$ times, take majority vote.

If single-run success probability is $p > 1/2$:
$$P(\text{majority correct after } R \text{ runs}) \geq 1 - e^{-\Omega(R)}$$

### Amplitude Amplification

Apply Grover-like amplification to boost probability of correct outcome:
- Mark states within $\pm 1$ of correct answer
- Amplify their amplitude
- Reduces number of repetitions needed

---

## Worked Examples

### Example 1: Computing P(best)

For $\phi = 0.35$ with $n = 3$, compute $P(m = 3)$ (nearest to $2.8$).

**Solution:**

$2^3 \cdot 0.35 = 2.8$

$P(3) = \frac{\sin^2(\pi \cdot (-0.2))}{64\sin^2(\pi \cdot (-0.2)/8)}$

$= \frac{\sin^2(0.2\pi)}{64\sin^2(0.025\pi)} \approx \frac{0.345}{64 \cdot 0.00616} \approx 0.87$

### Example 2: Repetition Analysis

If $P(\text{success}) = 0.6$, how many runs for 99% confidence?

**Solution:**

Using Chernoff bound: $R \geq \frac{\ln(1/\delta)}{2(p - 0.5)^2} = \frac{\ln(100)}{2(0.1)^2} \approx 230$ runs.

More efficiently with amplitude amplification: $O(\sqrt{1/0.6}) \approx$ few runs.

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Success probability | $P(m) = \frac{\sin^2(\pi\Delta)}{2^{2n}\sin^2(\pi\Delta/2^n)}$ |
| Minimum at peak | $\geq 4/\pi^2 \approx 0.405$ |
| Repetitions for confidence | $O(\log(1/\delta))$ |

### Key Takeaways

1. **Sinc-squared distribution** governs measurement outcomes
2. **Minimum 40.5%** success probability for best estimate
3. **Repetition** boosts confidence exponentially
4. **Amplitude amplification** reduces overhead
5. **Trade-offs** between precision, success rate, and resources

---

*Next: Day 607 - Iterative Phase Estimation*
