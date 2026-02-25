# Day 2: The Epsilon-Delta Definition of Limits

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Epsilon-Delta Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Proof Practice |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Review & Problems |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State the formal epsilon-delta (ε-δ) definition of a limit
2. Interpret the definition geometrically
3. Write basic epsilon-delta proofs for polynomial limits
4. Understand why rigor is necessary in analysis
5. Identify the logical structure of limit proofs

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 2.4**: The Precise Definition of a Limit (pp. 110-121)
- **Appendix D**: Precise Definitions of Limits (if available)

### Supplementary Reading
- **Spivak's Calculus** Chapter 5 (for deeper rigor)
- Any Real Analysis textbook introduction to limits

### Reading Strategy
1. First read: Focus on understanding the notation (30 min)
2. Second read: Work through each proof line by line (90 min)
3. Attempt to reconstruct proofs without looking (60 min)

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC
**Recitation: Epsilon-Delta Definition**
- Focus on the visual/geometric interpretation
- Pause and work through examples before seeing solutions

### Khan Academy
**Epsilon-Delta Definition of Limits**
- URL: Search "Khan Academy epsilon delta"
- Multiple videos building from intuition to formal proofs

### Mathologer / 3Blue1Brown
Look for visual explanations that show the "game" interpretation of ε-δ

---

## 📖 Core Content: The Rigorous Definition

### 1. Why Do We Need Rigor?

Our intuitive definition "f(x) gets arbitrarily close to L as x gets arbitrarily close to c" has problems:
- What does "close" mean precisely?
- How do we prove limits mathematically?
- How do we know our intuition isn't deceiving us?

The ε-δ definition gives us a precise, logical foundation.

### 2. The Formal Definition

> **Definition (Limit):** Let f be a function defined on an open interval containing c (except possibly at c itself). We say that the **limit of f(x) as x approaches c is L**, written
> $$\lim_{x \to c} f(x) = L$$
> if for every number ε > 0 there exists a number δ > 0 such that
> $$0 < |x - c| < \delta \implies |f(x) - L| < \epsilon$$

### 3. Unpacking the Definition

Let's break down each component:

**ε (epsilon):** Represents how close we want f(x) to be to L
- |f(x) - L| < ε means f(x) is within ε of L
- Geometrically: f(x) is in the horizontal band (L - ε, L + ε)

**δ (delta):** Represents how close x needs to be to c
- |x - c| < δ means x is within δ of c
- Geometrically: x is in the vertical band (c - δ, c + δ)

**0 < |x - c|:** We exclude x = c itself
- We care about behavior *near* c, not *at* c
- This allows the limit to exist even if f(c) is undefined

**"For every ε > 0, there exists δ > 0":** 
- ε is chosen first (by an "adversary")
- We must respond with an appropriate δ
- This must work for ANY positive ε, no matter how small

### 4. The "Game" Interpretation

Think of ε-δ as a game between two players:

1. **Challenger** picks any ε > 0 (the tolerance)
2. **Prover** must respond with δ > 0
3. **Prover wins** if: whenever 0 < |x - c| < δ, then |f(x) - L| < ε

If Prover can always win regardless of ε, the limit is L.

### 5. Geometric Visualization

```
        f(x)
          │
    L + ε ┼─────────────────────────  Upper bound of ε-band
          │         ╭─────╮
        L ┼─────────│  ●  │──────────  Target value
          │         ╰─────╯
    L - ε ┼─────────────────────────  Lower bound of ε-band
          │
          └─────┼───┼─●─┼───┼─────── x
              c-δ    c    c+δ
              
              └─────────┘
               δ-neighborhood
               (excluding c)
```

The definition says: For any ε-band around L, we can find a δ-neighborhood around c such that whenever x is in the δ-neighborhood (but not at c), f(x) is in the ε-band.

---

## ✏️ Worked Examples

### Example 1: Linear Function

**Prove:** $\lim_{x \to 3} (2x + 1) = 7$

**Proof:**

Let ε > 0 be given. We need to find δ > 0 such that:
$$0 < |x - 3| < \delta \implies |(2x + 1) - 7| < \epsilon$$

**Step 1: Simplify the consequent**
$$|(2x + 1) - 7| = |2x - 6| = 2|x - 3|$$

**Step 2: Work backwards to find δ**
We want $2|x - 3| < ε$, which means $|x - 3| < \frac{\epsilon}{2}$

**Step 3: Choose δ**
Let $\delta = \frac{\epsilon}{2}$

**Step 4: Verify the proof**
Suppose $0 < |x - 3| < \delta = \frac{\epsilon}{2}$

Then:
$$|(2x + 1) - 7| = 2|x - 3| < 2 \cdot \frac{\epsilon}{2} = \epsilon$$

Therefore, $\lim_{x \to 3} (2x + 1) = 7$. ∎

---

### Example 2: Quadratic Function

**Prove:** $\lim_{x \to 2} x^2 = 4$

**Proof:**

Let ε > 0 be given. We need:
$$0 < |x - 2| < \delta \implies |x^2 - 4| < \epsilon$$

**Step 1: Factor and analyze**
$$|x^2 - 4| = |x - 2||x + 2|$$

We need to bound |x + 2|. If we restrict |x - 2| < 1, then:
- 1 < x < 3
- 3 < x + 2 < 5
- |x + 2| < 5

**Step 2: Derive the bound**
If |x - 2| < 1, then:
$$|x^2 - 4| = |x - 2||x + 2| < 5|x - 2|$$

We want this less than ε, so we need $|x - 2| < \frac{\epsilon}{5}$

**Step 3: Choose δ**
Let $\delta = \min\left(1, \frac{\epsilon}{5}\right)$

The minimum ensures both conditions are satisfied.

**Step 4: Verify**
Suppose $0 < |x - 2| < \delta$

Since δ ≤ 1, we have |x + 2| < 5.
Since δ ≤ ε/5, we have $|x - 2| < \frac{\epsilon}{5}$

Therefore:
$$|x^2 - 4| = |x - 2||x + 2| < \frac{\epsilon}{5} \cdot 5 = \epsilon$$

Thus, $\lim_{x \to 2} x^2 = 4$. ∎

---

### Example 3: Rational Function

**Prove:** $\lim_{x \to 1} \frac{1}{x} = 1$

**Proof:**

Let ε > 0 be given. We need:
$$0 < |x - 1| < \delta \implies \left|\frac{1}{x} - 1\right| < \epsilon$$

**Step 1: Simplify**
$$\left|\frac{1}{x} - 1\right| = \left|\frac{1 - x}{x}\right| = \frac{|x - 1|}{|x|}$$

**Step 2: Bound |x| away from 0**
If |x - 1| < 1/2, then 1/2 < x < 3/2, so |x| > 1/2, meaning $\frac{1}{|x|} < 2$

**Step 3: Derive the bound**
Under this restriction:
$$\frac{|x - 1|}{|x|} < 2|x - 1|$$

We want this less than ε, so need $|x - 1| < \frac{\epsilon}{2}$

**Step 4: Choose δ**
Let $\delta = \min\left(\frac{1}{2}, \frac{\epsilon}{2}\right)$

**Step 5: Verify**
If $0 < |x - 1| < \delta$, then:
$$\left|\frac{1}{x} - 1\right| = \frac{|x - 1|}{|x|} < 2 \cdot \frac{\epsilon}{2} = \epsilon$$

Therefore, $\lim_{x \to 1} \frac{1}{x} = 1$. ∎

---

## 📐 Proof-Writing Template

For proving $\lim_{x \to c} f(x) = L$:

```
PROOF:
Let ε > 0 be given.                          [Standard opening]

[Scratch work - find δ in terms of ε]
|f(x) - L| = ... = (expression in |x - c|)
We want (expression) < ε
So we need |x - c| < (something in terms of ε)

Choose δ = [your choice based on scratch work]    [State your δ]

Suppose 0 < |x - c| < δ.                          [Assume hypothesis]

Then:
|f(x) - L| = ...                                   [Work forward]
           = ...
           < ε                                     [Conclude]

Therefore, lim_{x→c} f(x) = L.  ∎                 [State conclusion]
```

---

## 📝 Practice Problems

### Level 1: Basic Proofs
1. Prove: $\lim_{x \to 4} (3x - 5) = 7$
2. Prove: $\lim_{x \to -2} (x + 5) = 3$
3. Prove: $\lim_{x \to 0} 5x = 0$

### Level 2: Quadratic Functions
4. Prove: $\lim_{x \to 1} x^2 = 1$
5. Prove: $\lim_{x \to 3} (x^2 - 2x) = 3$
6. Prove: $\lim_{x \to -1} (x^2 + x) = 0$

### Level 3: Rational Functions
7. Prove: $\lim_{x \to 2} \frac{x}{3} = \frac{2}{3}$
8. Prove: $\lim_{x \to 4} \frac{1}{x-1} = \frac{1}{3}$

### Level 4: Challenging
9. Prove: $\lim_{x \to 0} x^2 \sin\left(\frac{1}{x}\right) = 0$ (use squeeze theorem approach)
10. Prove that if $\lim_{x \to c} f(x) = L$ and $\lim_{x \to c} g(x) = M$, then $\lim_{x \to c} [f(x) + g(x)] = L + M$

---

## 📊 Key Insights and Common Mistakes

### Common Mistakes to Avoid

1. **Choosing δ that depends on x**
   - WRONG: δ = ε/x
   - δ must be a constant (depending only on ε and c)

2. **Forgetting the condition 0 < |x - c|**
   - We explicitly exclude x = c from consideration

3. **Not verifying the proof works**
   - Always check that your δ actually makes the inequality work

4. **Mixing up ε and δ**
   - ε controls the output (vertical), δ controls the input (horizontal)

### Useful Inequalities

- **Triangle Inequality:** $|a + b| \leq |a| + |b|$
- **Reverse Triangle:** $||a| - |b|| \leq |a - b|$
- **Product Rule:** $|ab| = |a||b|$
- **Quotient Rule:** $\left|\frac{a}{b}\right| = \frac{|a|}{|b|}$ (when b ≠ 0)

---

## 🔬 Why This Matters for Quantum Mechanics

The epsilon-delta approach embodies the rigor needed in quantum mechanics:

1. **Hilbert Space Theory**: Convergence of state vectors uses similar limit definitions
2. **Operator Theory**: Bounded operators are defined using analogous ε-δ conditions
3. **Spectral Theory**: The spectrum of an operator involves limits in function spaces
4. **Path Integrals**: Feynman path integrals require careful limiting processes
5. **Renormalization**: Removing infinities in QFT needs precise control of limits

---

## ✅ Daily Checklist

- [ ] Read Stewart 2.4 completely
- [ ] Understand the formal definition (can recite from memory)
- [ ] Work through Examples 1-3 independently
- [ ] Complete Level 1 practice problems (1-3)
- [ ] Attempt at least two Level 2 problems
- [ ] Write the proof template from memory
- [ ] Create flashcards for the definition and key inequalities
- [ ] Explain the "game" interpretation in your own words

---

## 📓 Reflection Questions

1. Why is the order of quantifiers (∀ε, ∃δ) important? What would change if we reversed them?
2. In your own words, why do we need the condition 0 < |x - c|?
3. How does the geometric picture help you understand the definition?
4. What's the hardest part of writing ε-δ proofs, and how might you overcome it?

---

## 🔜 Preview: Tomorrow's Topics

**Day 3: Limit Laws and Techniques**

Tomorrow we'll learn:
- Limit laws (sum, product, quotient)
- How to use limit laws to evaluate limits without ε-δ
- Squeeze theorem
- Limits involving infinity

**Preparation:** Review algebra with polynomials and rational functions.

---

*"In mathematics, you don't understand things. You just get used to them."*
— John von Neumann

*But with ε-δ, understanding comes through practice!*
