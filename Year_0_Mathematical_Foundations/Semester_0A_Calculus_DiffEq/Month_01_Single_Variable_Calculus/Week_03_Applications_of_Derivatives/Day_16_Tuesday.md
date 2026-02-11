# Day 16: Linear Approximation and Differentials

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Linear Approximation |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Differentials |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Error Analysis |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Use tangent lines to approximate function values
2. Understand and compute differentials
3. Estimate errors using differentials
4. Apply linear approximation to real problems
5. Understand the connection to Taylor series (preview)

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 3.10**: Linear Approximations and Differentials (pp. 254-261)

---

## 📖 Core Content: Linear Approximation

### 1. The Big Idea

Near a point where we know f(a) and f'(a), the tangent line provides a good approximation to f(x):

$$f(x) \approx f(a) + f'(a)(x - a)$$

This is called the **linearization** of f at a.

### 2. Why It Works

The tangent line has:
- Same value as f at x = a
- Same slope as f at x = a

For x close to a, the tangent line "hugs" the curve.

### 3. Linearization Formula

> **Definition:** The **linearization** of f at a is:
> $$L(x) = f(a) + f'(a)(x - a)$$

**Approximation:**
$$f(x) \approx L(x) \text{ for } x \text{ near } a$$

### 4. Standard Linear Approximations

For small x (a = 0):

| Function | Linearization |
|----------|---------------|
| (1 + x)^n | ≈ 1 + nx |
| sin(x) | ≈ x |
| cos(x) | ≈ 1 |
| tan(x) | ≈ x |
| e^x | ≈ 1 + x |
| ln(1 + x) | ≈ x |
| √(1 + x) | ≈ 1 + x/2 |

---

## ✏️ Worked Examples: Linear Approximation

### Example 1: Approximating √4.1

**Problem:** Estimate √4.1 using linearization.

**Solution:**
Let f(x) = √x, and use a = 4 (where we know the exact value).

f(a) = f(4) = 2
f'(x) = 1/(2√x), so f'(4) = 1/4

Linearization:
$$L(x) = 2 + \frac{1}{4}(x - 4)$$

At x = 4.1:
$$L(4.1) = 2 + \frac{1}{4}(0.1) = 2 + 0.025 = 2.025$$

**Check:** √4.1 ≈ 2.0248... (error ≈ 0.0002)

---

### Example 2: Approximating sin(0.1)

**Problem:** Estimate sin(0.1) using linearization at a = 0.

**Solution:**
f(x) = sin(x), a = 0

f(0) = 0
f'(x) = cos(x), f'(0) = 1

Linearization:
$$L(x) = 0 + 1 \cdot (x - 0) = x$$

So sin(0.1) ≈ 0.1

**Check:** sin(0.1) ≈ 0.0998... (error ≈ 0.0002)

---

### Example 3: Approximating (1.02)^10

**Problem:** Estimate (1.02)^10 without a calculator.

**Solution:**
Use f(x) = (1 + x)^10 with x = 0.02 near a = 0.

f(0) = 1
f'(x) = 10(1 + x)^9, f'(0) = 10

Linearization:
$$L(x) = 1 + 10x$$

At x = 0.02:
$$L(0.02) = 1 + 10(0.02) = 1.2$$

**Check:** (1.02)^10 ≈ 1.2189... (error ≈ 0.019 or ~1.5%)

---

## 📖 Core Content: Differentials

### 5. The Differential

> **Definition:** If y = f(x) is differentiable, then the **differential** dy is:
> $$dy = f'(x) \cdot dx$$
> where dx is an independent variable representing a small change in x.

### 6. Geometric Interpretation

- **Δy = f(x + Δx) - f(x):** actual change in f
- **dy = f'(x)·dx:** change along the tangent line

For small dx (= Δx):
$$\Delta y \approx dy$$

```
    y
    │         ╱ tangent
    │       ╱  
    │   ●─╱───────── Δy (actual change)
    │  ╱│ ╲
    │ ╱ │  ╲ dy (approximation)
    │╱  │   ╲
    ●───┼───────── x
        │
       dx
```

### 7. Differential Formulas

| y | dy |
|---|---|
| xⁿ | nx^(n-1) dx |
| sin(x) | cos(x) dx |
| cos(x) | -sin(x) dx |
| e^x | e^x dx |
| ln(x) | (1/x) dx |

---

## ✏️ Worked Examples: Differentials

### Example 4: Computing a Differential

**Problem:** Find dy if y = x³ + 2x² - 3x + 1.

**Solution:**
$$dy = (3x^2 + 4x - 3) dx$$

---

### Example 5: Using Differentials for Approximation

**Problem:** Use differentials to approximate √36.5.

**Solution:**
Let y = √x = x^(1/2), x = 36, dx = 0.5

$$dy = \frac{1}{2\sqrt{x}} dx = \frac{1}{2\sqrt{36}}(0.5) = \frac{0.5}{12} = \frac{1}{24} \approx 0.0417$$

√36 = 6, so:
$$\sqrt{36.5} \approx 6 + 0.0417 = 6.0417$$

**Check:** √36.5 ≈ 6.0415... ✓

---

### Example 6: Error Propagation

**Problem:** A sphere's radius is measured as r = 10 cm with possible error ±0.1 cm. Estimate the maximum error in the calculated volume.

**Solution:**
V = (4/3)πr³

dV = 4πr² dr

With r = 10 and dr = ±0.1:
$$dV = 4\pi(100)(±0.1) = ±40\pi \approx ±125.7 \text{ cm}^3$$

The actual volume is V = (4/3)π(1000) ≈ 4189 cm³.

**Relative error:** dV/V = (40π)/(4000π/3) = 3(0.1)/10 = 0.03 = 3%

(Note: Relative error in V is 3 times relative error in r for a cube/sphere!)

---

### Example 7: Percentage Error

**Problem:** If the side of a cube is measured with 2% error, what is the approximate percentage error in the volume?

**Solution:**
V = s³

dV = 3s² ds

Relative error: dV/V = (3s² ds)/(s³) = 3(ds/s) = 3(2%) = 6%

**Rule:** For V = sⁿ, relative error in V is n times relative error in s.

---

## 📐 Connection to Taylor Series

Linear approximation is the first-order Taylor polynomial:

$$f(x) \approx f(a) + f'(a)(x-a)$$

Higher-order approximations include more terms:

$$f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + ...$$

We'll study Taylor series in detail in Week 4.

---

## 📝 Practice Problems

### Level 1: Linear Approximation
1. Find the linearization of f(x) = x³ at a = 2.
2. Use linearization to approximate √9.1.
3. Approximate cos(0.05) using L(x) at a = 0.
4. Estimate ln(1.1) using linearization.

### Level 2: Differentials
5. Find dy if y = x⁴ - 3x² + 5.
6. Find dy if y = sin(2x).
7. Use differentials to approximate (2.01)⁵.
8. Approximate ∛8.1 using differentials.

### Level 3: Error Analysis
9. A circle's radius is measured as 5 cm ± 0.02 cm. Estimate the error in the area.
10. A cube's edge is measured with 1% error. What is the percentage error in surface area?
11. The period of a pendulum is T = 2π√(L/g). If L is measured with 3% error, what is the error in T?

### Level 4: Challenge
12. Show that for f(x) = (1+x)^n, the linear approximation gives the first two terms of the binomial expansion.
13. Estimate the error in using sin(x) ≈ x for x = 0.5 radians. Compare with actual error.
14. A hemispherical dome has radius r = 50m ± 0.5m. Estimate the error in the surface area (2πr²).

---

## 📊 Answers

1. L(x) = 8 + 12(x - 2) = 12x - 16
2. L(9.1) = 3 + (1/6)(0.1) ≈ 3.0167
3. L(0.05) ≈ 1
4. L(1.1) ≈ 0.1
5. dy = (4x³ - 6x)dx
6. dy = 2cos(2x)dx
7. ≈ 32.8
8. ≈ 2.0042
9. dA ≈ ±0.628 cm²
10. 2%
11. 1.5%
12. (1+x)^n ≈ 1 + nx matches binomial: 1 + nx + ...
13. Error ≈ -x³/6 ≈ -0.021 (actual sin(0.5) ≈ 0.479, approx gives 0.5)
14. dA = 4πr·dr = 4π(50)(0.5) = 100π ≈ 314 m²

---

## 🔬 Physics Application

### Quantum Mechanics: Small Perturbations

In quantum mechanics, when a system experiences a small perturbation, we use linear approximation:

$$E_n \approx E_n^{(0)} + \langle n | H' | n \rangle$$

This is the first-order energy correction in perturbation theory!

---

## ✅ Daily Checklist

- [ ] Read Stewart 3.10
- [ ] Understand linearization formula
- [ ] Know standard linear approximations
- [ ] Compute differentials
- [ ] Apply to error estimation
- [ ] Complete Level 1-3 problems
- [ ] Understand connection to Taylor series

---

## 🔜 Preview: Tomorrow

**Day 17: Maximum and Minimum Values**
- Critical points
- Extreme Value Theorem
- First and second derivative tests

---

*"The tangent line is the best linear approximation to a curve."*
