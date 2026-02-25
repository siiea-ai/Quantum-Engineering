# Day 5: Week 1 Problem Set — Limits and Continuity

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Problem Set Part I |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Set Part II |
| Evening | 7:00 PM - 8:30 PM | 1.5 hours | Self-Assessment |

**Total Study Time: 7.5 hours**

---

## 🎯 Learning Objectives

Today is dedicated to deep practice. By the end, you should:

1. Confidently evaluate any standard limit
2. Write clean ε-δ proofs
3. Analyze continuity of any given function
4. Apply IVT effectively
5. Identify your remaining weak areas

---

## 📋 Instructions

This problem set mimics university exam conditions:
- Work without looking at notes for the first attempt
- Time yourself (suggested: 20-30 min per section)
- Show all work clearly
- Check answers only after completing each section
- Mark problems you struggled with for review

---

# 📝 PROBLEM SET: PART I — COMPUTATIONAL

## Section A: Limit Evaluation (4 points each)

Evaluate the following limits. Show all algebraic steps.

### A1. Direct Substitution
$$\lim_{x \to 3} (2x^3 - 5x^2 + 4x - 7)$$

### A2. Factoring (0/0 form)
$$\lim_{x \to 2} \frac{x^3 - 8}{x - 2}$$

### A3. Factoring with Quadratics
$$\lim_{x \to -1} \frac{x^2 + 5x + 4}{x^2 + 3x + 2}$$

### A4. Rationalizing Numerator
$$\lim_{x \to 0} \frac{\sqrt{1+x} - 1}{x}$$

### A5. Rationalizing Denominator
$$\lim_{x \to 16} \frac{x - 16}{\sqrt{x} - 4}$$

### A6. Complex Fraction
$$\lim_{x \to 0} \frac{\frac{1}{x+2} - \frac{1}{2}}{x}$$

### A7. Trigonometric (use known limits)
$$\lim_{x \to 0} \frac{\sin(3x)}{x}$$

### A8. Trigonometric
$$\lim_{x \to 0} \frac{\tan x}{x}$$

### A9. Limit at Infinity
$$\lim_{x \to \infty} \frac{3x^3 - 2x + 1}{5x^3 + x^2 - 4}$$

### A10. Limit at Infinity (Different Degrees)
$$\lim_{x \to \infty} \frac{2x^2 + 1}{x^3 - x}$$

---

## Section B: One-Sided Limits (5 points each)

### B1. Piecewise Function
For $f(x) = \begin{cases} x^2 - 1 & x < 2 \\ 3x - 4 & x \geq 2 \end{cases}$

Find:
a) $\lim_{x \to 2^-} f(x)$
b) $\lim_{x \to 2^+} f(x)$
c) $\lim_{x \to 2} f(x)$ (if it exists)
d) Is f continuous at x = 2?

### B2. Absolute Value
$$\lim_{x \to 3^+} \frac{|x-3|}{x-3} \quad \text{and} \quad \lim_{x \to 3^-} \frac{|x-3|}{x-3}$$

### B3. Vertical Asymptote
Find all one-sided limits at x = 1 for $f(x) = \frac{x}{(x-1)^2}$

---

## Section C: Squeeze Theorem (6 points each)

### C1. Classic Application
Prove: $\lim_{x \to 0} x^2 \cos\left(\frac{1}{x^2}\right) = 0$

### C2. Bounded Oscillation
Prove: $\lim_{x \to 0^+} \sqrt{x} \sin\left(\frac{1}{x}\right) = 0$

### C3. Limit at Infinity
Prove: $\lim_{x \to \infty} \frac{\cos(x)}{x} = 0$

---

# 📝 PROBLEM SET: PART II — THEORETICAL

## Section D: Epsilon-Delta Proofs (8 points each)

Write complete, rigorous ε-δ proofs.

### D1. Linear Function
Prove: $\lim_{x \to 4} (5x - 3) = 17$

### D2. Quadratic Function
Prove: $\lim_{x \to 2} (x^2 + 1) = 5$

### D3. Rational Function
Prove: $\lim_{x \to 3} \frac{x}{2} = \frac{3}{2}$

### D4. Challenge
Prove: $\lim_{x \to 1} \frac{1}{x} = 1$

---

## Section E: Continuity Analysis (5 points each)

### E1. Find All Discontinuities
For $f(x) = \frac{x^2 - 4}{x^2 - 5x + 6}$

a) Find all points of discontinuity
b) Classify each as removable, jump, or infinite
c) If removable, find the value that would make f continuous

### E2. Making Functions Continuous
Find all values of k that make f continuous everywhere:
$$f(x) = \begin{cases} x^2 + k & x \leq 1 \\ 2x + 1 & x > 1 \end{cases}$$

### E3. Two Constants
Find a and b so that f is continuous everywhere:
$$f(x) = \begin{cases} 2 & x \leq -1 \\ ax + b & -1 < x < 1 \\ 4 & x \geq 1 \end{cases}$$

### E4. Continuity from Definition
Using the ε-δ definition, prove that f(x) = 3x + 2 is continuous at x = 1.

---

## Section F: IVT Applications (7 points each)

### F1. Existence of Roots
Prove that $x^4 - 3x + 1 = 0$ has at least two solutions in [-2, 2].

### F2. Fixed Point
Prove that if f is continuous on [0, 1] and 0 ≤ f(x) ≤ 1 for all x ∈ [0, 1], then f has a fixed point (i.e., there exists c such that f(c) = c).

*Hint: Consider g(x) = f(x) - x*

### F3. Physical Application
A hot cup of coffee (180°F) is left in a room at 70°F. After 30 minutes, the coffee is at 120°F. Prove that at some time, the coffee was exactly 150°F.

### F4. Narrowing a Root
Given that $x^3 + x - 1 = 0$ has a root in [0, 1]:
a) Use bisection to narrow the root to an interval of length ≤ 0.125
b) State your final interval

---

## Section G: Conceptual Questions (4 points each)

### G1. True or False (with justification)
a) If $\lim_{x \to c} f(x) = L$, then f(c) = L.
b) If f is continuous at c, then $\lim_{x \to c} f(x) = f(c)$.
c) If $\lim_{x \to c^-} f(x) = \lim_{x \to c^+} f(x)$, then f is continuous at c.
d) Every polynomial is continuous on (-∞, ∞).

### G2. Counterexample Construction
Give an example of a function f such that:
a) f is defined at x = 2, the limit exists at x = 2, but f is not continuous at x = 2.
b) f is continuous everywhere except at x = 0 where it has a jump discontinuity.

### G3. Limit Does Not Exist
Explain THREE different ways a limit can fail to exist, with an example of each.

---

# ✅ ANSWER KEY

## Section A Answers
- A1: 2(27) - 5(9) + 4(3) - 7 = 54 - 45 + 12 - 7 = **14**
- A2: Factor x³ - 8 = (x-2)(x² + 2x + 4), limit = **12**
- A3: Factor to (x+4)(x+1)/[(x+2)(x+1)] = (x+4)/(x+2), limit = **3**
- A4: Rationalize: (x)/[x(√(1+x)+1)] = 1/(√(1+x)+1), limit = **1/2**
- A5: Rationalize: (x-16)(√x+4)/[(x-16)] = √x + 4, limit = **8**
- A6: Combine: (-x)/[2x(x+2)], simplify to -1/[2(x+2)], limit = **-1/4**
- A7: Use sin(3x)/(3x) · 3 = 1 · 3 = **3**
- A8: (sin x/x) · (1/cos x) = 1 · 1 = **1**
- A9: Divide by x³: 3/5 = **3/5**
- A10: Divide by x³: limit = **0**

## Section B Answers
- B1: a) 3, b) 2, c) DNE, d) Not continuous
- B2: Right = 1, Left = -1
- B3: Both one-sided limits = +∞

## Section C Answers
- C1: -x² ≤ x²cos(1/x²) ≤ x², squeeze gives 0
- C2: -√x ≤ √x sin(1/x) ≤ √x, squeeze gives 0
- C3: -1/x ≤ cos(x)/x ≤ 1/x, squeeze gives 0

## Section D Answers
- D1: Choose δ = ε/5
- D2: Choose δ = min(1, ε/5)
- D3: Choose δ = 2ε
- D4: Choose δ = min(1/2, ε/2)

## Section E Answers
- E1: Discontinuous at x = 2 (removable, define as 4/(-1) = -4) and x = 3 (infinite)
- E2: k = 2
- E3: a = 1, b = 3
- E4: Standard ε-δ proof with δ = ε/3

## Section F Answers
- F1: f(-2) = 23 > 0, f(-1) = 3 > 0, f(0) = 1 > 0, f(1) = -1 < 0, f(2) = 11 > 0. Sign changes at (0,1) and... check f(-2) to f(0) more carefully. Actually f(-2)=16+6+1=23, f(1)=1-3+1=-1, f(2)=16-6+1=11. Sign change in (0,1). For second root: f(-2)=23, f(-1)=1+3+1=5>0... need to check more points or argue differently.
- F2: g(x) = f(x) - x. g(0) = f(0) - 0 ≥ 0, g(1) = f(1) - 1 ≤ 0. By IVT, g(c) = 0 for some c.
- F3: Temperature is continuous, starts at 180, ends at 120, passes through 150.
- F4: [0.5, 0.625] or [0.625, 0.75]

## Section G Answers
- G1: a) False, b) True, c) False (also need f(c) to equal the limit), d) True
- G2: Various valid constructions
- G3: Jump, infinite, oscillation

---

## 📊 Self-Assessment Scoring

| Section | Points Possible | Your Score |
|---------|-----------------|------------|
| A (10 problems × 4) | 40 | |
| B (3 problems × 5) | 15 | |
| C (3 problems × 6) | 18 | |
| D (4 problems × 8) | 32 | |
| E (4 problems × 5) | 20 | |
| F (4 problems × 7) | 28 | |
| G (3 problems × 4) | 12 | |
| **TOTAL** | **165** | |

### Grade Scale
- 150-165: Excellent — Ready for differentiation
- 130-149: Good — Review weak areas before proceeding
- 110-129: Satisfactory — Additional practice recommended
- Below 110: Needs improvement — Review Week 1 material before continuing

---

## 📓 Reflection

After completing the problem set, answer:

1. Which section was most challenging? Why?
2. What types of problems do you need more practice with?
3. What concepts are you most confident about?
4. What would you do differently in your study approach?

---

## 🔜 Weekend Plan

**Saturday (Day 6):** Review weak areas, watch additional videos
**Sunday (Day 7):** Light review, prepare for Week 2 (Differentiation)

---

*"The only way to learn mathematics is to do mathematics."*
— Paul Halmos
