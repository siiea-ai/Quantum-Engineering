# Day 17: Maximum and Minimum Values

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Extreme Values Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Derivative Tests |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define and identify critical points
2. State and apply the Extreme Value Theorem
3. Find absolute maxima and minima on closed intervals
4. Apply the first derivative test
5. Apply the second derivative test
6. Classify critical points as local max, local min, or neither

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 4.1**: Maximum and Minimum Values (pp. 275-285)
- **Section 4.3**: How Derivatives Affect the Shape of a Graph (pp. 295-306)

---

## 📖 Core Content: Extrema

### 1. Definitions

> **Definition (Absolute/Global Extrema):**
> - f has an **absolute maximum** at c if f(c) ≥ f(x) for all x in the domain
> - f has an **absolute minimum** at c if f(c) ≤ f(x) for all x in the domain

> **Definition (Local/Relative Extrema):**
> - f has a **local maximum** at c if f(c) ≥ f(x) for all x near c
> - f has a **local minimum** at c if f(c) ≤ f(x) for all x near c

### 2. The Extreme Value Theorem

> **Theorem:** If f is continuous on a closed interval [a, b], then f attains an absolute maximum and an absolute minimum on [a, b].

**Key conditions:**
- f must be continuous
- Interval must be closed [a, b]

### 3. Critical Points

> **Definition:** A **critical number** (or critical point) of f is a number c in the domain of f where either:
> - f'(c) = 0, or
> - f'(c) does not exist

**Key insight:** Local extrema can ONLY occur at critical points!

> **Fermat's Theorem:** If f has a local maximum or minimum at c, and f'(c) exists, then f'(c) = 0.

### 4. Finding Absolute Extrema on [a, b]

**The Closed Interval Method:**
1. Find all critical numbers in (a, b)
2. Evaluate f at each critical number
3. Evaluate f at the endpoints a and b
4. The largest value is the absolute max; smallest is the absolute min

---

## ✏️ Worked Examples: Finding Extrema

### Example 1: Polynomial on Closed Interval

**Problem:** Find the absolute max and min of f(x) = x³ - 3x + 1 on [-2, 2].

**Solution:**

**Step 1:** Find critical numbers
f'(x) = 3x² - 3 = 3(x² - 1) = 3(x-1)(x+1)
f'(x) = 0 when x = ±1 (both in [-2, 2])

**Step 2:** Evaluate at critical numbers
f(-1) = (-1)³ - 3(-1) + 1 = -1 + 3 + 1 = 3
f(1) = (1)³ - 3(1) + 1 = 1 - 3 + 1 = -1

**Step 3:** Evaluate at endpoints
f(-2) = (-8) - (-6) + 1 = -8 + 6 + 1 = -1
f(2) = 8 - 6 + 1 = 3

**Step 4:** Compare
Values: {-1, 3, -1, 3}
- Absolute maximum: 3 at x = -1 and x = 2
- Absolute minimum: -1 at x = 1 and x = -2

---

### Example 2: With Undefined Derivative

**Problem:** Find extrema of f(x) = x^(2/3) on [-1, 8].

**Solution:**

f'(x) = (2/3)x^(-1/3) = 2/(3x^(1/3))

Critical points:
- f'(x) = 0: Never (numerator ≠ 0)
- f'(x) undefined: x = 0

Evaluate:
- f(-1) = (-1)^(2/3) = 1
- f(0) = 0
- f(8) = 8^(2/3) = 4

**Answer:** Absolute max = 4 at x = 8; Absolute min = 0 at x = 0

---

## 📖 The First Derivative Test

> **First Derivative Test:** Suppose c is a critical number of a continuous function f.
> 
> - If f' changes from **positive to negative** at c → f has a **local maximum** at c
> - If f' changes from **negative to positive** at c → f has a **local minimum** at c
> - If f' does **not change sign** at c → f has **no local extremum** at c

**Method:** Create a sign chart for f'(x).

### Example 3: First Derivative Test

**Problem:** Find local extrema of f(x) = x⁴ - 4x³.

**Solution:**

f'(x) = 4x³ - 12x² = 4x²(x - 3)

Critical points: x = 0, x = 3

**Sign chart for f'(x):**
```
        x < 0    0 < x < 3    x > 3
4x²       +         +          +
(x-3)     -         -          +
f'(x)     -         -          +
```

At x = 0: f' doesn't change sign → no extremum
At x = 3: f' changes from - to + → local minimum

f(3) = 81 - 108 = -27

**Answer:** Local minimum of -27 at x = 3; no local maximum.

---

## 📖 The Second Derivative Test

> **Second Derivative Test:** Suppose f'' is continuous near c and f'(c) = 0.
> 
> - If f''(c) > 0 → f has a **local minimum** at c
> - If f''(c) < 0 → f has a **local maximum** at c
> - If f''(c) = 0 → **Test is inconclusive** (use first derivative test)

**Intuition:** 
- f''(c) > 0 means concave up (cup shape) → minimum
- f''(c) < 0 means concave down (cap shape) → maximum

### Example 4: Second Derivative Test

**Problem:** Classify the critical points of f(x) = x³ - 6x² + 9x + 1.

**Solution:**

f'(x) = 3x² - 12x + 9 = 3(x² - 4x + 3) = 3(x - 1)(x - 3)

Critical points: x = 1, x = 3

f''(x) = 6x - 12

Test each critical point:
- f''(1) = 6 - 12 = -6 < 0 → local maximum at x = 1
- f''(3) = 18 - 12 = 6 > 0 → local minimum at x = 3

f(1) = 1 - 6 + 9 + 1 = 5 → local max
f(3) = 27 - 54 + 27 + 1 = 1 → local min

---

### Example 5: Inconclusive Second Derivative Test

**Problem:** Analyze f(x) = x⁴ at x = 0.

**Solution:**

f'(x) = 4x³, f'(0) = 0 (critical point)
f''(x) = 12x², f''(0) = 0 (inconclusive!)

Use first derivative test:
f'(x) = 4x³ is negative for x < 0, positive for x > 0
→ Sign changes from - to + → local minimum at x = 0

---

## 📖 Concavity

> **Definition:**
> - f is **concave up** on an interval if f' is increasing (f'' > 0)
> - f is **concave down** on an interval if f' is decreasing (f'' < 0)

> **Definition:** An **inflection point** is where f changes concavity.

### Example 6: Concavity Analysis

**Problem:** Determine concavity and inflection points for f(x) = x³ - 3x² + 2.

**Solution:**

f'(x) = 3x² - 6x
f''(x) = 6x - 6 = 6(x - 1)

f''(x) = 0 when x = 1
f''(x) < 0 when x < 1 (concave down)
f''(x) > 0 when x > 1 (concave up)

Inflection point at x = 1, where f(1) = 1 - 3 + 2 = 0.

**Answer:** Concave down on (-∞, 1), concave up on (1, ∞), inflection point at (1, 0).

---

## 📝 Practice Problems

### Level 1: Finding Critical Points
1. Find all critical numbers of f(x) = x³ - 6x² + 9x
2. Find all critical numbers of f(x) = x^(1/3)(x + 4)
3. Find all critical numbers of f(x) = |x² - 4|

### Level 2: Closed Interval Method
4. Find absolute max and min of f(x) = x² - 4x + 3 on [0, 4]
5. Find absolute max and min of f(x) = x³ - 3x + 1 on [0, 3]
6. Find absolute max and min of f(x) = sin(x) + cos(x) on [0, π]

### Level 3: Derivative Tests
7. Use first derivative test to classify critical points of f(x) = x⁴ - 8x²
8. Use second derivative test for f(x) = x·e^(-x)
9. Find all local extrema and inflection points of f(x) = x⁴ - 4x³ + 6x²

### Level 4: Comprehensive
10. Sketch the graph of f(x) = x/(x² + 1) showing all critical points, inflection points, and asymptotes
11. Find all extrema and inflection points of f(x) = x·ln(x) for x > 0
12. Analyze f(x) = x²·e^(-x) completely

---

## 📊 Answers

1. x = 1, x = 3
2. x = 0, x = -1
3. x = ±2, x = 0
4. Max = 3 at x = 0 or x = 4; Min = -1 at x = 2
5. Max = 19 at x = 3; Min = -1 at x = 1
6. Max = √2 at x = π/4; Min = -1 at x = π
7. Local max at x = 0; local min at x = ±2
8. Local max at x = 1, f(1) = 1/e
9. Local min at x = 0; inflection points at x = 1, x = 2
10. Local max at x = 1, local min at x = -1; inflection at x = 0, ±√3
11. Local min at x = 1/e; inflection at x = 1/e^(3/2)
12. Local max at x = 2; inflection at x = 2 ± √2

---

## 🔬 Quantum Mechanics Connection

Finding extrema is crucial in quantum mechanics:

- **Variational Principle:** The ground state energy is the minimum of ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩
- **Equilibrium Configurations:** Stable states minimize energy
- **Transition States:** Saddle points (max in one direction, min in another)

The condition ∂E/∂λ = 0 for energy extrema parallels f'(c) = 0.

---

## ✅ Daily Checklist

- [ ] Read Stewart 4.1 and 4.3
- [ ] Master the Closed Interval Method
- [ ] Practice first derivative test
- [ ] Practice second derivative test
- [ ] Complete Level 1-3 problems
- [ ] Understand concavity and inflection points
- [ ] Know when second derivative test fails

---

## 🔜 Preview: Tomorrow

**Day 18: Optimization Problems**
- Setting up optimization problems from word problems
- Finding dimensions that maximize/minimize quantities
- Applied max/min problems

---

*"The calculus was the first achievement of modern mathematics, and it is difficult to overestimate its importance."*
— John von Neumann
