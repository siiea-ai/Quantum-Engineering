# Day 4: Continuity and the Intermediate Value Theorem

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Continuity Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | IVT and Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Problem Solving |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State the three-part definition of continuity at a point
2. Identify and classify types of discontinuities
3. Determine intervals of continuity for a function
4. State and apply the Intermediate Value Theorem
5. Use IVT to prove existence of roots

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 2.5**: Continuity (pp. 122-134)

### Key Theorems to Master
- Definition of continuity (three conditions)
- Types of discontinuities
- Intermediate Value Theorem
- Continuity of compositions

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC
**Lecture on Continuity and IVT**

### 3Blue1Brown
**Essence of Calculus: Limits and Continuity**
- Beautiful visual intuition

### Professor Leonard
**Continuity: Finding Intervals of Continuity**
- Comprehensive worked examples

---

## 📖 Core Content: Continuity

### 1. The Definition of Continuity

> **Definition:** A function f is **continuous at a point c** if and only if all three conditions hold:
> 
> 1. **f(c) is defined** (c is in the domain of f)
> 2. **$\lim_{x \to c} f(x)$ exists**
> 3. **$\lim_{x \to c} f(x) = f(c)$** (the limit equals the function value)

**Intuitive meaning:** You can draw the graph without lifting your pencil.

**Formal (ε-δ) meaning:** For every ε > 0, there exists δ > 0 such that:
$$|x - c| < \delta \implies |f(x) - f(c)| < \epsilon$$

Note: Unlike the limit definition, we don't need 0 < |x - c| because we want f(c) itself to fit.

### 2. One-Sided Continuity

> **Definition:** f is **continuous from the right at c** if:
> $$\lim_{x \to c^+} f(x) = f(c)$$

> **Definition:** f is **continuous from the left at c** if:
> $$\lim_{x \to c^-} f(x) = f(c)$$

**Theorem:** f is continuous at c ⟺ f is continuous from both left and right at c.

### 3. Continuity on an Interval

> **Definition:** f is **continuous on an open interval (a, b)** if f is continuous at every point in (a, b).

> **Definition:** f is **continuous on a closed interval [a, b]** if:
> - f is continuous on (a, b)
> - f is continuous from the right at a
> - f is continuous from the left at b

---

## 📊 Types of Discontinuities

### Type 1: Removable Discontinuity

**Condition:** $\lim_{x \to c} f(x)$ exists, but either f(c) is undefined or f(c) ≠ limit.

**Example:** $f(x) = \frac{x^2 - 1}{x - 1}$ at x = 1

- f(1) is undefined
- $\lim_{x \to 1} f(x) = 2$
- Can be "fixed" by defining f(1) = 2

**Graphically:** A single "hole" in the graph.

### Type 2: Jump Discontinuity

**Condition:** Both one-sided limits exist but are unequal.
$$\lim_{x \to c^-} f(x) \neq \lim_{x \to c^+} f(x)$$

**Example:** $f(x) = \begin{cases} 1 & x < 0 \\ 2 & x \geq 0 \end{cases}$ at x = 0

**Graphically:** The graph "jumps" from one value to another.

### Type 3: Infinite Discontinuity

**Condition:** At least one of the one-sided limits is ±∞.

**Example:** $f(x) = \frac{1}{x}$ at x = 0

- $\lim_{x \to 0^+} f(x) = +\infty$
- $\lim_{x \to 0^-} f(x) = -\infty$

**Graphically:** Vertical asymptote.

### Type 4: Oscillating Discontinuity

**Condition:** The function oscillates infinitely as x → c.

**Example:** $f(x) = \sin\left(\frac{1}{x}\right)$ at x = 0

**Graphically:** Infinite oscillation with no limiting value.

---

## 📐 Continuity of Standard Functions

### Theorem: Continuous Function Types

The following functions are continuous on their domains:

1. **Polynomials:** Continuous on (-∞, ∞)
2. **Rational functions:** Continuous except where denominator = 0
3. **Root functions:** $\sqrt[n]{x}$ continuous on appropriate domain
4. **Trigonometric functions:** sin, cos continuous on (-∞, ∞)
5. **Exponential functions:** $e^x$, $a^x$ continuous on (-∞, ∞)
6. **Logarithmic functions:** ln(x), log(x) continuous on (0, ∞)

### Continuity Rules

If f and g are continuous at c, then:
- f + g is continuous at c
- f - g is continuous at c
- f · g is continuous at c
- f/g is continuous at c (if g(c) ≠ 0)
- f ∘ g is continuous at c (if g continuous at c and f continuous at g(c))

---

## 🏛️ The Intermediate Value Theorem

### Statement

> **Theorem (IVT):** If f is continuous on [a, b] and N is any number between f(a) and f(b), then there exists at least one c ∈ (a, b) such that f(c) = N.

### Visual Interpretation

```
    f(x)
      │
 f(a) ●─────────────────
      │        ╱╲
    N ┼───────●───────────  (there must be at least one crossing)
      │      ╱    ╲
 f(b) ●────╱───────╲────
      │
      └────┼────┼────┼──── x
           a    c    b
```

If f is continuous and goes from f(a) to f(b), it must hit every value in between at least once.

### Important Notes

1. **Continuity is essential:** Without continuity, the theorem fails.
2. **Existence only:** IVT tells us a value EXISTS but doesn't tell us how to FIND it.
3. **May have multiple solutions:** There could be many c values that work.

---

## ✏️ Worked Examples

### Example 1: Checking Continuity

**Question:** Is $f(x) = \frac{x^2 - 4}{x - 2}$ continuous at x = 2?

**Solution:**
Check the three conditions:
1. Is f(2) defined? NO - division by zero.

Since condition 1 fails, f is NOT continuous at x = 2.

However, $\lim_{x \to 2} f(x) = \lim_{x \to 2} (x + 2) = 4$ exists.

This is a **removable discontinuity**.

---

### Example 2: Making a Function Continuous

**Question:** Find k so that $f(x) = \begin{cases} x^2 + 1 & x < 2 \\ kx - 1 & x \geq 2 \end{cases}$ is continuous at x = 2.

**Solution:**
For continuity at x = 2, we need:
$$\lim_{x \to 2^-} f(x) = \lim_{x \to 2^+} f(x) = f(2)$$

Left limit: $\lim_{x \to 2^-} (x^2 + 1) = 5$

Right limit: $\lim_{x \to 2^+} (kx - 1) = 2k - 1$

f(2) = 2k - 1

For continuity: 5 = 2k - 1

Solving: k = 3

---

### Example 3: Using IVT to Prove Root Existence

**Question:** Prove that $x^3 - x - 1 = 0$ has a solution in [1, 2].

**Solution:**
Let f(x) = x³ - x - 1.

f is a polynomial, so it's continuous everywhere, including [1, 2].

Calculate:
- f(1) = 1 - 1 - 1 = -1 < 0
- f(2) = 8 - 2 - 1 = 5 > 0

Since f(1) < 0 < f(2) and f is continuous on [1, 2], by IVT:

There exists c ∈ (1, 2) such that f(c) = 0.

Therefore, x³ - x - 1 = 0 has at least one solution in (1, 2). ∎

---

### Example 4: Narrowing Down a Root (Bisection Preview)

**Question:** Narrow down the root of x³ - x - 1 = 0 to an interval of length 0.25.

**Solution:**
Start: f(1) < 0, f(2) > 0, root in (1, 2)

Test midpoint x = 1.5:
f(1.5) = 3.375 - 1.5 - 1 = 0.875 > 0

Root in (1, 1.5)

Test x = 1.25:
f(1.25) = 1.953 - 1.25 - 1 = -0.297 < 0

Root in (1.25, 1.5)

Interval length = 0.25 ✓

---

### Example 5: IVT Application in Physics

**Question:** A car travels from rest to 60 mph in 10 seconds. Prove that at some instant, the car's speed was exactly 30 mph.

**Solution:**
Let v(t) = velocity at time t.

- v(0) = 0 mph
- v(10) = 60 mph
- v is continuous (velocity doesn't jump instantaneously for a real car)

By IVT, since 0 < 30 < 60 and v is continuous on [0, 10]:

There exists some time c ∈ (0, 10) where v(c) = 30 mph. ∎

---

## 📝 Practice Problems

### Level 1: Continuity at a Point
1. Determine if $f(x) = \frac{x^2 - 9}{x - 3}$ is continuous at x = 3.
2. Determine if $f(x) = \begin{cases} x + 1 & x \leq 2 \\ 2x - 1 & x > 2 \end{cases}$ is continuous at x = 2.
3. For what value(s) of x is $f(x) = \frac{1}{x^2 - 4}$ discontinuous?

### Level 2: Finding Constants
4. Find k so $f(x) = \begin{cases} kx^2 & x \leq 1 \\ x + k & x > 1 \end{cases}$ is continuous everywhere.
5. Find a and b so $f(x) = \begin{cases} ax + b & x < 1 \\ x^2 & 1 \leq x \leq 2 \\ bx + a & x > 2 \end{cases}$ is continuous.

### Level 3: Classifying Discontinuities
6. Classify all discontinuities of $f(x) = \frac{x^2 - x - 2}{x^2 - 4}$
7. Classify discontinuities of $f(x) = \lfloor x \rfloor$ (floor function)
8. Classify discontinuities of $f(x) = \frac{|x-1|}{x-1}$

### Level 4: IVT Applications
9. Show that $x^5 - 3x^4 = 1$ has a solution in [3, 4].
10. Prove that $\cos x = x$ has a solution in (0, 1).
11. Show that every polynomial of odd degree has at least one real root.

### Level 5: Theoretical
12. Prove: If f is continuous on [a, b] and f(a) < 0 < f(b), then f has a root in (a, b).
13. Give an example of a function discontinuous at every point.
14. Prove: If f is continuous at c and f(c) > 0, then f(x) > 0 for all x in some interval around c.

---

## 📊 Answers

1. Not continuous (removable discontinuity)
2. Not continuous (jump discontinuity, left limit = 3, right limit = 3, but these are equal... wait, let me recalculate: f(2) = 3 from left, 2(2)-1=3 from right. Actually continuous!)
3. x = 2 and x = -2
4. k = 2
5. a = 2, b = 1
6. Removable at x = 2; infinite at x = -2
7. Jump discontinuity at every integer
8. Jump discontinuity at x = 1
9. f(3) = -80 < 0, f(4) = 255 > 0, apply IVT
10. f(x) = cos x - x: f(0) = 1 > 0, f(1) ≈ -0.46 < 0
11. Use limits at ±∞ and IVT
12. Direct application of IVT with N = 0
13. f(x) = 1 if x is rational, 0 if irrational (Dirichlet function)
14. Use ε-δ with ε = f(c)/2

---

## 🔬 Why Continuity Matters in Physics

1. **Wave functions in QM** must be continuous (and differentiable) for physical reasons
2. **Boundary conditions** require matching wave functions continuously
3. **Schrödinger equation** requires twice-differentiable (hence continuous) wave functions
4. **Physical observables** like position and momentum have continuous spectra in certain cases

---

## ✅ Daily Checklist

- [ ] Read Stewart 2.5 completely
- [ ] Memorize the three-part definition of continuity
- [ ] Know all four types of discontinuities with examples
- [ ] State IVT precisely
- [ ] Complete Level 1-3 problems
- [ ] Apply IVT to prove existence of roots
- [ ] Create continuity flowchart for problem-solving
- [ ] Understand connection to ε-δ definition

---

## 📓 Reflection Questions

1. Why does IVT require continuity? Give a counterexample showing IVT fails for discontinuous functions.
2. Can a function have infinitely many discontinuities on [0, 1]? Give an example.
3. How does continuity connect to the "no lifting pencil" intuition?
4. Why is continuity from the right at a and from the left at b needed for closed intervals?

---

## 🔜 Preview: Tomorrow

**Day 5: Review and Integration**

- Comprehensive review of Week 1 material
- Integration of limits and continuity concepts
- Practice exam-style problems
- Preparation for differentiation

---

*"Continuity is one of the most beautiful concepts in mathematics—it captures the idea that small changes in input produce small changes in output."*
