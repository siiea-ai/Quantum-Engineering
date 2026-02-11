# Day 25: Integration by Substitution

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | u-Substitution Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Recognize when to use u-substitution
2. Apply u-substitution to indefinite integrals
3. Apply u-substitution to definite integrals
4. Choose appropriate substitutions
5. Handle more complex substitutions

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 5.5**: The Substitution Rule (pp. 410-419)

---

## 📖 Core Content: The Substitution Rule

### 1. The Chain Rule in Reverse

Recall the chain rule:
$$\frac{d}{dx}[F(g(x))] = F'(g(x)) \cdot g'(x) = f(g(x)) \cdot g'(x)$$

Reading this backward gives the **Substitution Rule**:

> **Substitution Rule (Indefinite Integrals):**
> If u = g(x) is a differentiable function and f is continuous, then:
> $$\int f(g(x)) \cdot g'(x) \, dx = \int f(u) \, du$$
> where u = g(x) and du = g'(x)dx.

### 2. Why It Works

The integral ∫f(g(x))g'(x)dx has the form that results from differentiating F(g(x)).

So ∫f(g(x))g'(x)dx = F(g(x)) + C.

But F(g(x)) = F(u) = ∫f(u)du when u = g(x).

### 3. The Method

**Step 1:** Identify a composite function f(g(x))
**Step 2:** Let u = g(x) (the "inner" function)
**Step 3:** Compute du = g'(x)dx
**Step 4:** Substitute u and du into the integral
**Step 5:** Integrate with respect to u
**Step 6:** Substitute back u = g(x)

---

## ✏️ Worked Examples

### Example 1: Basic Substitution
$$\int 2x(x^2 + 1)^5 \, dx$$

Let u = x² + 1, then du = 2x dx

$$= \int u^5 \, du = \frac{u^6}{6} + C = \frac{(x^2+1)^6}{6} + C$$

**Check:** d/dx[(x²+1)⁶/6] = 6(x²+1)⁵ · 2x / 6 = 2x(x²+1)⁵ ✓

---

### Example 2: Adjusting for Constants
$$\int x(x^2 + 1)^5 \, dx$$

Let u = x² + 1, then du = 2x dx, so x dx = du/2

$$= \int u^5 \cdot \frac{du}{2} = \frac{1}{2} \int u^5 \, du = \frac{1}{2} \cdot \frac{u^6}{6} + C = \frac{(x^2+1)^6}{12} + C$$

---

### Example 3: Trigonometric
$$\int \sin^3(x) \cos(x) \, dx$$

Let u = sin(x), then du = cos(x) dx

$$= \int u^3 \, du = \frac{u^4}{4} + C = \frac{\sin^4(x)}{4} + C$$

---

### Example 4: Exponential
$$\int e^{3x} \, dx$$

Let u = 3x, then du = 3dx, so dx = du/3

$$= \int e^u \cdot \frac{du}{3} = \frac{1}{3}e^u + C = \frac{e^{3x}}{3} + C$$

---

### Example 5: Logarithmic
$$\int \frac{1}{x \ln(x)} \, dx$$

Let u = ln(x), then du = (1/x)dx

$$= \int \frac{1}{u} \, du = \ln|u| + C = \ln|\ln(x)| + C$$

---

### Example 6: Square Root
$$\int \frac{x}{\sqrt{x^2 + 4}} \, dx$$

Let u = x² + 4, then du = 2x dx, so x dx = du/2

$$= \int \frac{1}{\sqrt{u}} \cdot \frac{du}{2} = \frac{1}{2} \int u^{-1/2} \, du$$

$$= \frac{1}{2} \cdot \frac{u^{1/2}}{1/2} + C = \sqrt{u} + C = \sqrt{x^2 + 4} + C$$

---

## 📖 Substitution in Definite Integrals

### Two Methods

**Method 1: Substitute back before evaluating**
- Integrate to get F(u) + C
- Substitute u = g(x)
- Evaluate at original limits a and b

**Method 2: Change the limits (preferred)**
- When x = a, u = g(a)
- When x = b, u = g(b)
- Evaluate with new limits directly

### Example 7: Definite Integral (Changing Limits)
$$\int_0^2 x(x^2 + 1)^3 \, dx$$

Let u = x² + 1, du = 2x dx, so x dx = du/2

When x = 0: u = 0² + 1 = 1
When x = 2: u = 2² + 1 = 5

$$= \int_1^5 u^3 \cdot \frac{du}{2} = \frac{1}{2} \left[\frac{u^4}{4}\right]_1^5$$

$$= \frac{1}{8}[5^4 - 1^4] = \frac{1}{8}[625 - 1] = \frac{624}{8} = 78$$

---

### Example 8: Trigonometric Definite Integral
$$\int_0^{\pi/2} \sin(x)\cos^2(x) \, dx$$

Let u = cos(x), du = -sin(x) dx

When x = 0: u = cos(0) = 1
When x = π/2: u = cos(π/2) = 0

$$= \int_1^0 u^2 \cdot (-du) = \int_0^1 u^2 \, du = \left[\frac{u^3}{3}\right]_0^1 = \frac{1}{3}$$

---

## 📋 Common Substitution Patterns

| Integral Form | Substitution |
|---------------|--------------|
| ∫[f(x)]ⁿf'(x)dx | u = f(x) |
| ∫f(ax+b)dx | u = ax + b |
| ∫xf(x²)dx | u = x² |
| ∫sin(f(x))f'(x)dx | u = f(x) |
| ∫eᶠ⁽ˣ⁾f'(x)dx | u = f(x) |
| ∫f'(x)/f(x)dx | u = f(x), result: ln|f(x)| |

---

## 📝 Practice Problems

### Level 1: Basic
1. ∫ (2x + 3)⁵ dx
2. ∫ x(x² - 1)⁴ dx
3. ∫ cos(3x) dx
4. ∫ e^(5x) dx

### Level 2: Adjusting Constants
5. ∫ x²(x³ + 2)⁶ dx
6. ∫ sin(x)cos(x) dx
7. ∫ xe^(x²) dx
8. ∫ x/√(x² + 9) dx

### Level 3: More Complex
9. ∫ tan(x) dx [Hint: tan = sin/cos]
10. ∫ x/(1 + x²)² dx
11. ∫ (ln x)³/x dx
12. ∫ sec²(x)tan³(x) dx

### Level 4: Definite Integrals
13. ∫₀¹ x(1 - x²)³ dx
14. ∫₀^(π/4) tan(x)sec²(x) dx
15. ∫₁ᵉ (ln x)/x dx
16. ∫₀¹ x·e^(x²) dx

### Level 5: Challenge
17. ∫ √(1 + √x) / √x dx
18. ∫ 1/(x ln x ln(ln x)) dx
19. ∫ sin(x)/(1 + cos²(x)) dx

---

## 📊 Answers

1. (2x + 3)⁶/12 + C
2. (x² - 1)⁵/10 + C
3. sin(3x)/3 + C
4. e^(5x)/5 + C
5. (x³ + 2)⁷/21 + C
6. sin²(x)/2 + C [or -cos²(x)/2 + C]
7. e^(x²)/2 + C
8. √(x² + 9) + C
9. -ln|cos(x)| + C = ln|sec(x)| + C
10. -1/(2(1+x²)) + C
11. (ln x)⁴/4 + C
12. tan⁴(x)/4 + C
13. 1/8
14. 1/2
15. 1/2
16. (e - 1)/2
17. (4/3)(1 + √x)^(3/2) + C
18. ln|ln(ln x)| + C
19. -arctan(cos x) + C

---

## 🔬 Quantum Mechanics Connection

Many QM integrals require substitution:

**Gaussian integrals:**
$$\int_0^\infty e^{-ax^2} dx = \frac{1}{2}\sqrt{\frac{\pi}{a}}$$

**Probability calculations:**
$$P = \int |\psi(x)|^2 dx$$

often require u-substitution for complex wave functions.

---

## ✅ Daily Checklist

- [ ] Read Stewart 5.5
- [ ] Master the substitution process
- [ ] Know when to use substitution
- [ ] Handle constant adjustments
- [ ] Change limits for definite integrals
- [ ] Recognize common patterns
- [ ] Complete practice problems

---

## 📓 Reflection Questions

1. How do you recognize when substitution will work?
2. Why does du = g'(x)dx?
3. When is it better to change limits vs. substitute back?
4. What's the connection to the chain rule?

---

## 🔜 Preview: Tomorrow

**Day 26: Week 4 Problem Set**

Comprehensive assessment covering:
- Antiderivatives
- Riemann sums
- FTC Parts 1 and 2
- u-Substitution

---

*"Substitution is the art of seeing simplicity hidden within complexity."*
