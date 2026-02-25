# Day 23: The Definite Integral and Area

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Riemann Sums |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Definite Integrals |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand Riemann sums as approximations of area
2. Define the definite integral as a limit of Riemann sums
3. Evaluate simple definite integrals from the definition
4. Interpret definite integrals as signed area
5. Apply basic properties of definite integrals

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 5.1**: Areas and Distances (pp. 363-374)
- **Section 5.2**: The Definite Integral (pp. 375-388)

---

## 🎬 Video Resources

### 3Blue1Brown
**Chapter 8: Integration and the fundamental theorem of calculus**
- Beautiful visual intuition

### MIT OpenCourseWare 18.01SC
**Lecture: Definite Integrals**

### Professor Leonard
**Riemann Sums and Definite Integrals**

---

## 📖 Core Content: The Area Problem

### 1. The Challenge

How do we find the area under a curve y = f(x) from x = a to x = b?

For rectangles: A = base × height
For triangles: A = ½ × base × height
For curves: We need calculus!

### 2. The Strategy: Approximation by Rectangles

Divide [a, b] into n subintervals of width Δx = (b-a)/n.

In each subinterval, approximate the curve with a rectangle.

Sum the areas of all rectangles → Riemann sum.

Take limit as n → ∞ → exact area.

---

## 📖 Riemann Sums

### 3. Setup

Partition [a, b] into n equal subintervals:
- Width: Δx = (b - a)/n
- Endpoints: x₀ = a, x₁ = a + Δx, x₂ = a + 2Δx, ..., xₙ = b
- General point: xᵢ = a + iΔx

### 4. Types of Riemann Sums

**Left Riemann Sum:**
$$L_n = \sum_{i=0}^{n-1} f(x_i) \Delta x = \sum_{i=0}^{n-1} f(a + i\Delta x) \Delta x$$

**Right Riemann Sum:**
$$R_n = \sum_{i=1}^{n} f(x_i) \Delta x = \sum_{i=1}^{n} f(a + i\Delta x) \Delta x$$

**Midpoint Riemann Sum:**
$$M_n = \sum_{i=1}^{n} f\left(\frac{x_{i-1} + x_i}{2}\right) \Delta x$$

### 5. Visual Representation

```
        Left Sum          Right Sum          Midpoint Sum
    ┌─┬─┬─┬─┐           ┌─┬─┬─┬─┐           ┌─┬─┬─┬─┐
    │█│█│█│█│/          │ │█│█│█│/          │▄│█│█│▄│/
    │█│█│█│/│           │█│█│█│/│           │█│█│█│█│
    │█│█│/│ │           │█│█│/│█│           │█│█│█│/│
    │█│/│ │ │           │█│/│█│█│           │█│/│█│█│
    └─┴─┴─┴─┘           └─┴─┴─┴─┘           └─┴─┴─┴─┘
```

---

## ✏️ Worked Example: Computing a Riemann Sum

**Problem:** Estimate ∫₀² x² dx using a right Riemann sum with n = 4.

**Solution:**

Δx = (2 - 0)/4 = 0.5

Right endpoints: x₁ = 0.5, x₂ = 1, x₃ = 1.5, x₄ = 2

$$R_4 = \sum_{i=1}^{4} f(x_i) \Delta x = [f(0.5) + f(1) + f(1.5) + f(2)] \cdot 0.5$$

$$= [0.25 + 1 + 2.25 + 4] \cdot 0.5 = 7.5 \cdot 0.5 = 3.75$$

**Actual value:** ∫₀² x² dx = [x³/3]₀² = 8/3 ≈ 2.67

The right sum overestimates (since x² is increasing on [0,2]).

---

## 📖 The Definite Integral

### 6. Definition

> **Definition:** The **definite integral** of f from a to b is:
> $$\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i^*) \Delta x$$
> provided this limit exists, where x_i* is any point in the i-th subinterval.

If the limit exists, f is called **integrable** on [a, b].

### 7. Notation

$$\int_a^b f(x) \, dx$$

- a is the **lower limit** of integration
- b is the **upper limit** of integration
- f(x) is the **integrand**
- dx indicates integration with respect to x

**Note:** The definite integral is a NUMBER, not a function!

### 8. Geometric Interpretation

$$\int_a^b f(x) \, dx = \text{(area above x-axis)} - \text{(area below x-axis)}$$

This is called **signed area** or **net area**.

---

## 📋 Properties of Definite Integrals

### Property 1: Reversing Limits
$$\int_a^b f(x) \, dx = -\int_b^a f(x) \, dx$$

### Property 2: Same Limits
$$\int_a^a f(x) \, dx = 0$$

### Property 3: Constant Multiple
$$\int_a^b cf(x) \, dx = c\int_a^b f(x) \, dx$$

### Property 4: Sum/Difference
$$\int_a^b [f(x) \pm g(x)] \, dx = \int_a^b f(x) \, dx \pm \int_a^b g(x) \, dx$$

### Property 5: Additivity
$$\int_a^b f(x) \, dx + \int_b^c f(x) \, dx = \int_a^c f(x) \, dx$$

### Property 6: Comparison
If f(x) ≤ g(x) for all x in [a, b], then:
$$\int_a^b f(x) \, dx \leq \int_a^b g(x) \, dx$$

### Property 7: Bounds
If m ≤ f(x) ≤ M for all x in [a, b], then:
$$m(b-a) \leq \int_a^b f(x) \, dx \leq M(b-a)$$

---

## ✏️ More Worked Examples

### Example 2: Evaluating from Definition

**Problem:** Find ∫₀¹ x dx using the limit definition.

**Solution:**

Δx = 1/n, xᵢ = i/n (using right endpoints)

$$R_n = \sum_{i=1}^{n} f(x_i)\Delta x = \sum_{i=1}^{n} \frac{i}{n} \cdot \frac{1}{n} = \frac{1}{n^2}\sum_{i=1}^{n} i$$

Using formula Σᵢ₌₁ⁿ i = n(n+1)/2:

$$R_n = \frac{1}{n^2} \cdot \frac{n(n+1)}{2} = \frac{n+1}{2n} = \frac{1}{2} + \frac{1}{2n}$$

$$\int_0^1 x \, dx = \lim_{n \to \infty} \left(\frac{1}{2} + \frac{1}{2n}\right) = \frac{1}{2}$$

**Check:** This is the area of triangle with base 1 and height 1: A = ½ ✓

---

### Example 3: Using Geometry

**Problem:** Evaluate ∫₋₂² √(4 - x²) dx.

**Solution:**

y = √(4 - x²) means x² + y² = 4, y ≥ 0

This is the upper half of a circle with radius 2!

Area = ½ π(2)² = 2π

$$\int_{-2}^{2} \sqrt{4 - x^2} \, dx = 2\pi$$

---

### Example 4: Signed Area

**Problem:** Evaluate ∫₋₁² x dx.

**Solution:**

From x = -1 to x = 0: area below axis = ½(1)(1) = ½ (negative contribution)
From x = 0 to x = 2: area above axis = ½(2)(2) = 2 (positive contribution)

∫₋₁² x dx = 2 - ½ = 3/2

---

## 📝 Practice Problems

### Level 1: Riemann Sums
1. Estimate ∫₁³ x dx using left Riemann sum with n = 4.
2. Estimate ∫₀⁴ √x dx using right Riemann sum with n = 4.
3. Estimate ∫₀^π sin(x) dx using midpoint sum with n = 4.

### Level 2: Using Geometry
4. ∫₀⁵ 3 dx (constant function)
5. ∫₀⁴ (2x + 1) dx (linear function → trapezoid)
6. ∫₋₃³ √(9 - x²) dx (semicircle)

### Level 3: Properties
7. If ∫₀⁵ f(x)dx = 10 and ∫₀² f(x)dx = 3, find ∫₂⁵ f(x)dx.
8. If ∫₁⁴ f(x)dx = 6, find ∫₁⁴ 5f(x)dx.
9. If ∫₀³ f(x)dx = 4 and ∫₀³ g(x)dx = -2, find ∫₀³ [2f(x) - 3g(x)]dx.

### Level 4: From Definition
10. Use the limit definition to find ∫₀² x² dx.
11. Use the limit definition to find ∫₁³ (2x + 1) dx.

### Level 5: Interpretation
12. ∫₀²^π sin(x) dx (sketch and think about areas)
13. What does ∫₀^∞ e^(-x) dx represent? (preview of improper integrals)

---

## 📊 Answers

1. L₄ = 3.5
2. R₄ = (0 + 1 + √2 + √3) ≈ 4.15
3. M₄ ≈ 2.05
4. 15
5. 20
6. (9π)/2
7. 7
8. 30
9. 14
10. 8/3
11. 10
12. 0 (equal areas above and below)
13. 1 (area under e^(-x) from 0 to ∞)

---

## 🔬 Quantum Mechanics Connection

**Probability as Area:**

For a particle with wave function ψ(x), the probability of finding it between a and b is:

$$P(a \leq x \leq b) = \int_a^b |\psi(x)|^2 \, dx$$

This is the area under |ψ|² from a to b!

**Normalization:**
$$\int_{-\infty}^{\infty} |\psi(x)|^2 \, dx = 1$$

Total probability = 1 (particle must be somewhere).

---

## ✅ Daily Checklist

- [ ] Read Stewart 5.1-5.2
- [ ] Understand Riemann sums geometrically
- [ ] Compute left, right, and midpoint sums
- [ ] Know the definition of definite integral
- [ ] Apply properties of definite integrals
- [ ] Interpret signed area
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 24: The Fundamental Theorem of Calculus**

The most important theorem in calculus! It connects differentiation and integration:
- Part 1: d/dx[∫ₐˣ f(t)dt] = f(x)
- Part 2: ∫ₐᵇ f(x)dx = F(b) - F(a)

---

*"The integral is the sum of an infinite number of infinitesimally small quantities—a concept that took centuries to make rigorous."*
