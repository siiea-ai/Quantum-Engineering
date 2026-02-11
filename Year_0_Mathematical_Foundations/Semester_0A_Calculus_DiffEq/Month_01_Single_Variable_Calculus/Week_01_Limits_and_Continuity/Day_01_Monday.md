# Day 1: Introduction to Limits — Intuitive Understanding

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Reading & Note-taking |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Video Lectures & Examples |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice Problems |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Explain what a limit represents conceptually
2. Evaluate simple limits using tables and graphs
3. Identify when limits do not exist
4. Distinguish between limits from the left and right (one-sided limits)
5. Apply limit notation correctly

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Chapter 2.1**: The Tangent and Velocity Problems (pp. 78-83)
- **Chapter 2.2**: The Limit of a Function (pp. 84-96)

### Reading Strategy
1. First pass (45 min): Read through quickly, noting section headings
2. Second pass (90 min): Detailed reading with note-taking
3. Work through every example in the text with pencil and paper

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC Single Variable Calculus
**Lecture 1: Derivatives, Slope, Velocity, Rate of Change**
- URL: [MIT OCW 18.01](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)
- Watch: First 30 minutes for context on why limits matter
- Note: This lecture motivates limits through the derivative concept

### 3Blue1Brown - Essence of Calculus
**Chapter 1: The Essence of Calculus**
- URL: [YouTube - 3Blue1Brown](https://www.youtube.com/watch?v=WUvTyaaNkzM)
- Duration: 17 minutes
- **Key Insight**: Visual understanding of approaching a value

### Professor Leonard - Calculus 1
**Limits: An Intuitive Introduction**
- URL: Search "Professor Leonard Limits Introduction"
- Duration: ~1 hour
- **Note**: Excellent for building intuition with many worked examples

---

## 📖 Core Content: Theory and Concepts

### 1. What is a Limit?

The limit of a function f(x) as x approaches a value c is the value that f(x) gets arbitrarily close to as x gets arbitrarily close to c.

**Notation:**
$$\lim_{x \to c} f(x) = L$$

This is read as: "The limit of f(x) as x approaches c equals L."

**Critical Understanding:** 
- The limit is about what happens *near* c, not *at* c
- The function doesn't need to be defined at c for the limit to exist
- The function's value at c (if it exists) doesn't affect the limit

### 2. Intuitive Example: The Removed Point

Consider the function:
$$f(x) = \frac{x^2 - 1}{x - 1}$$

**Question:** What is $\lim_{x \to 1} f(x)$?

**Analysis:**
- f(x) is undefined at x = 1 (division by zero)
- But we can factor: $f(x) = \frac{(x-1)(x+1)}{x-1} = x + 1$ for x ≠ 1
- As x approaches 1, f(x) approaches 2

**Table of Values:**
| x | f(x) |
|---|------|
| 0.9 | 1.9 |
| 0.99 | 1.99 |
| 0.999 | 1.999 |
| 1.001 | 2.001 |
| 1.01 | 2.01 |
| 1.1 | 2.1 |

**Conclusion:** $\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = 2$

### 3. One-Sided Limits

**Left-Hand Limit:** The limit as x approaches c from values less than c
$$\lim_{x \to c^-} f(x)$$

**Right-Hand Limit:** The limit as x approaches c from values greater than c
$$\lim_{x \to c^+} f(x)$$

**Fundamental Theorem:** 
$$\lim_{x \to c} f(x) = L \iff \lim_{x \to c^-} f(x) = L \text{ and } \lim_{x \to c^+} f(x) = L$$

The two-sided limit exists if and only if both one-sided limits exist and are equal.

### 4. When Limits Don't Exist

A limit fails to exist in three common scenarios:

**Case 1: Jump Discontinuity**
$$f(x) = \begin{cases} 1 & \text{if } x < 0 \\ 2 & \text{if } x \geq 0 \end{cases}$$

Here: $\lim_{x \to 0^-} f(x) = 1$ but $\lim_{x \to 0^+} f(x) = 2$

Since the one-sided limits differ, $\lim_{x \to 0} f(x)$ does not exist (DNE).

**Case 2: Infinite Behavior**
$$f(x) = \frac{1}{x^2}$$

As x → 0, f(x) → ∞. We write $\lim_{x \to 0} \frac{1}{x^2} = \infty$

Note: This is technically "does not exist" since ∞ is not a real number, but we use this notation to describe the behavior.

**Case 3: Oscillation**
$$f(x) = \sin\left(\frac{1}{x}\right)$$

As x → 0, this function oscillates infinitely between -1 and 1. The limit does not exist.

### 5. Graphical Interpretation

When analyzing limits graphically:
1. Trace the curve from the left toward x = c
2. Note what y-value you're approaching
3. Trace the curve from the right toward x = c
4. Note what y-value you're approaching
5. If both approaches give the same y-value, that's the limit

---

## ✏️ Worked Examples

### Example 1: Polynomial Limit
**Find:** $\lim_{x \to 2} (3x^2 - 2x + 1)$

**Solution:**
For polynomials, we can simply substitute:
$$\lim_{x \to 2} (3x^2 - 2x + 1) = 3(2)^2 - 2(2) + 1 = 12 - 4 + 1 = 9$$

### Example 2: Rational Function with Removable Discontinuity
**Find:** $\lim_{x \to 3} \frac{x^2 - 9}{x - 3}$

**Solution:**
Direct substitution gives 0/0 (indeterminate form).

Factor the numerator:
$$\frac{x^2 - 9}{x - 3} = \frac{(x-3)(x+3)}{x-3} = x + 3 \text{ for } x \neq 3$$

Therefore:
$$\lim_{x \to 3} \frac{x^2 - 9}{x - 3} = \lim_{x \to 3} (x + 3) = 6$$

### Example 3: Piecewise Function
**Find:** $\lim_{x \to 1} f(x)$ where $f(x) = \begin{cases} x^2 & \text{if } x < 1 \\ 2x - 1 & \text{if } x > 1 \end{cases}$

**Solution:**
Left-hand limit: $\lim_{x \to 1^-} x^2 = 1$

Right-hand limit: $\lim_{x \to 1^+} (2x - 1) = 2(1) - 1 = 1$

Since both one-sided limits equal 1:
$$\lim_{x \to 1} f(x) = 1$$

### Example 4: Limit Does Not Exist
**Investigate:** $\lim_{x \to 0} \frac{|x|}{x}$

**Solution:**
For x > 0: $\frac{|x|}{x} = \frac{x}{x} = 1$

For x < 0: $\frac{|x|}{x} = \frac{-x}{x} = -1$

Therefore:
- $\lim_{x \to 0^+} \frac{|x|}{x} = 1$
- $\lim_{x \to 0^-} \frac{|x|}{x} = -1$

Since the one-sided limits differ, $\lim_{x \to 0} \frac{|x|}{x}$ **does not exist**.

---

## 📝 Practice Problems

### Level 1: Basic Evaluation
1. $\lim_{x \to 4} (x^2 + 3x - 2)$
2. $\lim_{x \to -1} \frac{x^2 - 1}{x + 1}$
3. $\lim_{x \to 0} \frac{\sin x}{x}$ (Use a calculator to make a table; don't solve analytically yet)

### Level 2: One-Sided Limits
4. For $f(x) = \begin{cases} x + 2 & x < 1 \\ x^2 & x \geq 1 \end{cases}$, find:
   - $\lim_{x \to 1^-} f(x)$
   - $\lim_{x \to 1^+} f(x)$
   - $\lim_{x \to 1} f(x)$ (if it exists)

5. Evaluate $\lim_{x \to 2^-} \frac{x}{x-2}$ and $\lim_{x \to 2^+} \frac{x}{x-2}$

### Level 3: Conceptual
6. Sketch a function f(x) where:
   - $\lim_{x \to 2} f(x) = 3$
   - f(2) = 5
   - f is continuous everywhere except at x = 2

7. Give an example of a function where $\lim_{x \to 0} f(x)$ does not exist due to oscillation.

### Level 4: Challenge
8. Prove that $\lim_{x \to 0} x \sin\left(\frac{1}{x}\right) = 0$ using the squeeze theorem (preview of tomorrow's topic).

---

## 📊 Answers to Practice Problems

1. 26
2. -2
3. Approaches 1 (exact proof coming in future lessons)
4. Left: 3, Right: 1, Two-sided: DNE
5. Left: -∞, Right: +∞
6. Various valid answers (removable discontinuity at x=2)
7. $\sin(1/x)$ as x → 0
8. Use -|x| ≤ x sin(1/x) ≤ |x| and squeeze theorem

---

## 🔬 Why This Matters for Quantum Mechanics

Understanding limits is foundational for quantum mechanics because:

1. **Wave Function Normalization**: Probability densities must integrate to 1, requiring understanding of improper integrals (limits at infinity)

2. **The Schrödinger Equation**: This differential equation involves derivatives, which are defined as limits

3. **Dirac Delta Function**: This "function" is defined as a limit of increasingly narrow, tall functions

4. **Perturbation Theory**: Taylor series expansions around equilibrium points rely on limits

5. **Quantum Field Theory**: Renormalization involves taking limits carefully to remove infinities

---

## ✅ Daily Checklist

- [ ] Read Stewart 2.1-2.2 (complete two passes)
- [ ] Watch MIT OCW Lecture 1 (first 30 min)
- [ ] Watch 3Blue1Brown Chapter 1
- [ ] Complete all worked examples independently
- [ ] Solve problems 1-5 from practice set
- [ ] Attempt challenge problem 8
- [ ] Create 5 flashcards for key concepts
- [ ] Write a one-paragraph summary of today's learning

---

## 📓 Reflection Questions

Before ending today's session, answer these in your study journal:

1. In your own words, what is the difference between f(c) and $\lim_{x \to c} f(x)$?
2. Why might a limit exist even when the function is undefined at that point?
3. What are the three ways a limit can fail to exist?
4. How would you explain limits to someone who has never studied calculus?

---

## 🔜 Preview: Tomorrow's Topics

**Day 2: Epsilon-Delta Definition of Limits**

Tomorrow we make limits rigorous. We'll learn:
- The formal ε-δ definition
- How to prove limits using ε-δ
- Why this rigor matters for advanced mathematics

**Preparation:** Review the intuitive limit concept and think about what "arbitrarily close" means mathematically.

---

*"The notion of limit is the central idea that distinguishes calculus from algebra and geometry."*
— Howard Eves
