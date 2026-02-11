# Day 22: Antiderivatives and Indefinite Integrals

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Antiderivative Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Integration Rules |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand the concept of antiderivative
2. Compute basic indefinite integrals
3. Apply the power rule for integration
4. Integrate trigonometric and exponential functions
5. Understand the role of the constant of integration

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 4.9**: Antiderivatives (pp. 352-360)

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC
**Lecture: Antiderivatives**

### 3Blue1Brown
**Chapter 8: Integration and the fundamental theorem of calculus**

### Professor Leonard
**Calculus 1: Introduction to Integrals**

---

## 📖 Core Content: Antiderivatives

### 1. The Central Question

We've studied: Given f(x), find f'(x).

Now we ask the **reverse question:**

> Given f'(x), find f(x).

This is called finding an **antiderivative**.

### 2. Definition

> **Definition:** A function F is called an **antiderivative** of f on an interval I if F'(x) = f(x) for all x in I.

**Example:** 
- F(x) = x² is an antiderivative of f(x) = 2x because d/dx[x²] = 2x
- But so is G(x) = x² + 5 because d/dx[x² + 5] = 2x
- And H(x) = x² - 17...

### 3. The General Antiderivative

> **Theorem:** If F is an antiderivative of f on an interval I, then the **most general antiderivative** of f on I is:
> $$F(x) + C$$
> where C is an arbitrary constant.

**Why?** If F'(x) = f(x) and G'(x) = f(x), then (F - G)' = 0, so F - G = constant.

### 4. Notation: The Indefinite Integral

We write:
$$\int f(x) \, dx = F(x) + C$$

Read as: "The integral of f(x) with respect to x equals F(x) plus C"

**Components:**
- ∫ : integral sign (elongated S for "sum")
- f(x) : integrand
- dx : differential (indicates variable of integration)
- C : constant of integration

---

## 📋 Basic Integration Rules

### The Power Rule for Integration

$$\int x^n \, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)$$

**Derivation:** Check by differentiating:
$$\frac{d}{dx}\left[\frac{x^{n+1}}{n+1}\right] = \frac{(n+1)x^n}{n+1} = x^n \checkmark$$

### Special Case: n = -1

$$\int \frac{1}{x} \, dx = \int x^{-1} \, dx = \ln|x| + C$$

(The absolute value is needed because ln is only defined for positive numbers, but 1/x exists for x < 0 too.)

### Table of Basic Integrals

| Function f(x) | Antiderivative ∫f(x)dx |
|---------------|------------------------|
| k (constant) | kx + C |
| xⁿ (n ≠ -1) | x^(n+1)/(n+1) + C |
| 1/x | ln\|x\| + C |
| eˣ | eˣ + C |
| aˣ | aˣ/ln(a) + C |
| sin(x) | -cos(x) + C |
| cos(x) | sin(x) + C |
| sec²(x) | tan(x) + C |
| csc²(x) | -cot(x) + C |
| sec(x)tan(x) | sec(x) + C |
| csc(x)cot(x) | -csc(x) + C |
| 1/√(1-x²) | arcsin(x) + C |
| 1/(1+x²) | arctan(x) + C |

### Linearity of Integration

$$\int [f(x) + g(x)] \, dx = \int f(x) \, dx + \int g(x) \, dx$$

$$\int cf(x) \, dx = c \int f(x) \, dx$$

---

## ✏️ Worked Examples

### Example 1: Power Rule
$$\int x^4 \, dx = \frac{x^5}{5} + C$$

**Check:** d/dx[x⁵/5] = 5x⁴/5 = x⁴ ✓

---

### Example 2: Polynomial
$$\int (3x^2 - 4x + 5) \, dx$$

$$= 3 \cdot \frac{x^3}{3} - 4 \cdot \frac{x^2}{2} + 5x + C$$

$$= x^3 - 2x^2 + 5x + C$$

---

### Example 3: Negative and Fractional Powers
$$\int \left(x^{-2} + \sqrt{x}\right) dx = \int \left(x^{-2} + x^{1/2}\right) dx$$

$$= \frac{x^{-1}}{-1} + \frac{x^{3/2}}{3/2} + C = -\frac{1}{x} + \frac{2x^{3/2}}{3} + C$$

---

### Example 4: Trigonometric
$$\int (2\cos x - 3\sin x) \, dx = 2\sin x + 3\cos x + C$$

---

### Example 5: Exponential
$$\int (e^x + 2^x) \, dx = e^x + \frac{2^x}{\ln 2} + C$$

---

### Example 6: Rewriting Before Integrating
$$\int \frac{x^2 + 3}{x} \, dx = \int \left(x + \frac{3}{x}\right) dx = \frac{x^2}{2} + 3\ln|x| + C$$

---

### Example 7: Initial Value Problem
**Problem:** Find f(x) if f'(x) = 3x² - 4 and f(1) = 5.

**Solution:**
$$f(x) = \int (3x^2 - 4) \, dx = x^3 - 4x + C$$

Use initial condition f(1) = 5:
$$5 = 1 - 4 + C = -3 + C$$
$$C = 8$$

**Answer:** f(x) = x³ - 4x + 8

---

## 📐 Geometric Interpretation

The antiderivative F(x) represents a family of curves. Each value of C gives a different curve, all vertical translations of each other.

The initial condition "pins down" which specific curve we want.

---

## 📝 Practice Problems

### Level 1: Basic Power Rule
1. ∫ x⁷ dx
2. ∫ x^(-3) dx
3. ∫ √x dx
4. ∫ ∛x² dx

### Level 2: Polynomials
5. ∫ (4x³ - 2x² + 7) dx
6. ∫ (x + 1)² dx (expand first)
7. ∫ (t² - t + 1)/t dt
8. ∫ x(x² + 3) dx

### Level 3: Trigonometric and Exponential
9. ∫ (5cos x + 3sin x) dx
10. ∫ sec²x dx
11. ∫ 4eˣ dx
12. ∫ (eˣ - e^(-x)) dx

### Level 4: Mixed
13. ∫ (3/x + 2x) dx
14. ∫ (sec x)(sec x + tan x) dx
15. ∫ (1 + sin²x)/sin²x dx

### Level 5: Initial Value Problems
16. f'(x) = 2x - 3, f(0) = 4. Find f(x).
17. f'(x) = cos x, f(0) = 1. Find f(x).
18. f''(x) = 12x, f'(0) = 2, f(0) = 1. Find f(x).

---

## 📊 Answers

1. x⁸/8 + C
2. -1/(2x²) + C
3. (2/3)x^(3/2) + C
4. (3/5)x^(5/3) + C
5. x⁴ - (2/3)x³ + 7x + C
6. (1/3)x³ + x² + x + C
7. (1/2)t² - t + ln|t| + C
8. (1/4)x⁴ + (3/2)x² + C
9. 5sin x - 3cos x + C
10. tan x + C
11. 4eˣ + C
12. eˣ + e^(-x) + C
13. 3ln|x| + x² + C
14. tan x + sec x + C
15. -cot x + x + C
16. f(x) = x² - 3x + 4
17. f(x) = sin x + 1
18. f(x) = 2x³ + 2x + 1

---

## 🔬 Physics Application: Motion

If acceleration a(t) is known:
$$v(t) = \int a(t) \, dt$$
$$s(t) = \int v(t) \, dt$$

**Example:** A ball dropped from rest has a(t) = -32 ft/s².

v(t) = ∫(-32)dt = -32t + C₁

Initial condition v(0) = 0: C₁ = 0, so v(t) = -32t

s(t) = ∫(-32t)dt = -16t² + C₂

If s(0) = 100 ft: C₂ = 100, so s(t) = -16t² + 100

---

## 🔬 Quantum Mechanics Connection

The wave function ψ must be normalized:
$$\int_{-\infty}^{\infty} |\psi(x)|^2 \, dx = 1$$

Finding this integral is essential for determining probability amplitudes!

---

## ✅ Daily Checklist

- [ ] Read Stewart 4.9
- [ ] Memorize basic integral formulas
- [ ] Complete Level 1-3 problems
- [ ] Solve initial value problems
- [ ] Always include +C for indefinite integrals
- [ ] Check answers by differentiating
- [ ] Understand geometric interpretation

---

## 📓 Reflection Questions

1. Why do we need the constant C?
2. How is integration the "reverse" of differentiation?
3. Why is ∫1/x dx = ln|x| + C (with absolute value)?
4. What does an initial condition tell us?

---

## 🔜 Preview: Tomorrow

**Day 23: The Definite Integral and Area**
- Riemann sums
- Area under curves
- The definite integral ∫ₐᵇ f(x)dx

---

*"Integration is the inverse of differentiation, but it's much harder—and more beautiful."*
