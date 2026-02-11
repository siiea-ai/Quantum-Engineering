# Day 9: Differentiation Rules — Power, Sum, Product

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Basic Rules |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Product & Quotient |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Apply the power rule for any real exponent
2. Differentiate sums and differences of functions
3. Apply the product rule correctly
4. Apply the quotient rule correctly
5. Combine rules to differentiate complex expressions

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 3.1**: Derivatives of Polynomials and Exponential Functions (pp. 175-184)
- **Section 3.2**: The Product and Quotient Rules (pp. 185-193)

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC
**Lecture 3: Product and Quotient Rule**

### 3Blue1Brown
**Chapter 3: Derivative formulas through geometry**

### Professor Leonard
**Power Rule, Product Rule, and Quotient Rule**

---

## 📖 Core Content: Differentiation Rules

### 1. Constant Rule

> **Theorem:** If c is a constant, then $\frac{d}{dx}[c] = 0$

**Proof:** Let f(x) = c.
$$f'(x) = \lim_{h \to 0} \frac{c - c}{h} = \lim_{h \to 0} 0 = 0$$

**Interpretation:** Horizontal lines have zero slope.

---

### 2. Power Rule

> **Theorem (Power Rule):** If n is any real number, then
> $$\frac{d}{dx}[x^n] = nx^{n-1}$$

**Proof for positive integers (by limit definition):**
$$\frac{d}{dx}[x^n] = \lim_{h \to 0} \frac{(x+h)^n - x^n}{h}$$

Using binomial expansion: $(x+h)^n = x^n + nx^{n-1}h + \binom{n}{2}x^{n-2}h^2 + ...$

$$= \lim_{h \to 0} \frac{nx^{n-1}h + (\text{terms with } h^2, h^3, ...)}{h}$$
$$= \lim_{h \to 0} [nx^{n-1} + \text{terms with } h] = nx^{n-1}$$

**Examples:**
| Function | Derivative |
|----------|------------|
| x⁵ | 5x⁴ |
| x⁻² = 1/x² | -2x⁻³ = -2/x³ |
| x^(1/2) = √x | (1/2)x^(-1/2) = 1/(2√x) |
| x^(2/3) | (2/3)x^(-1/3) |

---

### 3. Constant Multiple Rule

> **Theorem:** If c is a constant and f is differentiable, then
> $$\frac{d}{dx}[cf(x)] = c \cdot f'(x)$$

**Proof:**
$$\frac{d}{dx}[cf(x)] = \lim_{h \to 0} \frac{cf(x+h) - cf(x)}{h} = c \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} = cf'(x)$$

**Example:** $\frac{d}{dx}[7x^4] = 7 \cdot 4x^3 = 28x^3$

---

### 4. Sum and Difference Rules

> **Theorem:** If f and g are differentiable, then
> $$\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)$$
> $$\frac{d}{dx}[f(x) - g(x)] = f'(x) - g'(x)$$

**Proof (Sum):**
$$\frac{d}{dx}[f+g] = \lim_{h \to 0} \frac{[f(x+h)+g(x+h)] - [f(x)+g(x)]}{h}$$
$$= \lim_{h \to 0} \left[\frac{f(x+h)-f(x)}{h} + \frac{g(x+h)-g(x)}{h}\right] = f'(x) + g'(x)$$

**Example:** $\frac{d}{dx}[x^3 + x^2 - 4x + 7] = 3x^2 + 2x - 4$

---

### 5. Product Rule

> **Theorem (Product Rule):** If f and g are differentiable, then
> $$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$$

**Memory aid:** "First times derivative of second, plus second times derivative of first"
Or: $(fg)' = f'g + fg'$

**Proof:**
$$\frac{d}{dx}[fg] = \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h}$$

Add and subtract f(x+h)g(x):
$$= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x) - f(x)g(x)}{h}$$
$$= \lim_{h \to 0} \left[f(x+h)\frac{g(x+h)-g(x)}{h} + g(x)\frac{f(x+h)-f(x)}{h}\right]$$
$$= f(x)g'(x) + g(x)f'(x)$$

**Example:** Find $\frac{d}{dx}[(x^2 + 1)(x^3 - 2x)]$

Let f(x) = x² + 1, g(x) = x³ - 2x
- f'(x) = 2x
- g'(x) = 3x² - 2

$(fg)' = f'g + fg' = (2x)(x^3 - 2x) + (x^2 + 1)(3x^2 - 2)$
$= 2x^4 - 4x^2 + 3x^4 - 2x^2 + 3x^2 - 2$
$= 5x^4 - 3x^2 - 2$

**Verification:** Expand first: $(x^2+1)(x^3-2x) = x^5 - 2x^3 + x^3 - 2x = x^5 - x^3 - 2x$
Derivative: $5x^4 - 3x^2 - 2$ ✓

---

### 6. Quotient Rule

> **Theorem (Quotient Rule):** If f and g are differentiable and g(x) ≠ 0, then
> $$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$$

**Memory aid:** "Low d-high minus high d-low, over low squared"
$$\left(\frac{f}{g}\right)' = \frac{gf' - fg'}{g^2}$$

**Proof:** Write f/g = f · (1/g) and use product rule with the derivative of 1/g = -g'/g².

**Example:** Find $\frac{d}{dx}\left[\frac{x^2 + 1}{x - 3}\right]$

Let f(x) = x² + 1, g(x) = x - 3
- f'(x) = 2x
- g'(x) = 1

$$\left(\frac{f}{g}\right)' = \frac{(x-3)(2x) - (x^2+1)(1)}{(x-3)^2}$$
$$= \frac{2x^2 - 6x - x^2 - 1}{(x-3)^2} = \frac{x^2 - 6x - 1}{(x-3)^2}$$

---

### 7. Exponential Function

> **Theorem:** $\frac{d}{dx}[e^x] = e^x$

This is the unique function equal to its own derivative!

**More generally:** $\frac{d}{dx}[a^x] = a^x \ln(a)$

---

## 📋 Summary Table of Basic Derivatives

| Function | Derivative |
|----------|------------|
| c (constant) | 0 |
| xⁿ | nxⁿ⁻¹ |
| eˣ | eˣ |
| aˣ | aˣ ln(a) |
| ln(x) | 1/x |
| sin(x) | cos(x) |
| cos(x) | -sin(x) |

---

## ✏️ Worked Examples

### Example 1: Polynomial
$$\frac{d}{dx}[4x^5 - 3x^3 + 2x - 7]$$
$$= 20x^4 - 9x^2 + 2$$

### Example 2: Fractional Exponents
$$\frac{d}{dx}[x^{3/2} + x^{-1/2}]$$
$$= \frac{3}{2}x^{1/2} - \frac{1}{2}x^{-3/2} = \frac{3\sqrt{x}}{2} - \frac{1}{2x^{3/2}}$$

### Example 3: Product Rule
$$\frac{d}{dx}[x^2 e^x]$$
$$= (2x)(e^x) + (x^2)(e^x) = e^x(2x + x^2) = e^x(x^2 + 2x)$$

### Example 4: Quotient Rule
$$\frac{d}{dx}\left[\frac{e^x}{x^2 + 1}\right]$$
$$= \frac{(x^2+1)(e^x) - (e^x)(2x)}{(x^2+1)^2} = \frac{e^x(x^2 + 1 - 2x)}{(x^2+1)^2} = \frac{e^x(x-1)^2}{(x^2+1)^2}$$

### Example 5: Complex Combination
$$\frac{d}{dx}\left[\frac{x^3}{x^2 - 4}\right]$$
$$= \frac{(x^2-4)(3x^2) - (x^3)(2x)}{(x^2-4)^2}$$
$$= \frac{3x^4 - 12x^2 - 2x^4}{(x^2-4)^2} = \frac{x^4 - 12x^2}{(x^2-4)^2} = \frac{x^2(x^2 - 12)}{(x^2-4)^2}$$

---

## 📝 Practice Problems

### Level 1: Power Rule
Differentiate:
1. f(x) = x⁷
2. f(x) = 5x⁴ - 2x² + 3
3. f(x) = √x + 1/x
4. f(x) = x^(2/3) - x^(-1/3)

### Level 2: Product Rule
5. f(x) = x²(x³ + 1)
6. f(x) = (2x + 1)(3x - 2)
7. f(x) = (x² + 1)eˣ
8. f(x) = x√x

### Level 3: Quotient Rule
9. f(x) = (x + 1)/(x - 1)
10. f(x) = x²/(x² + 1)
11. f(x) = (2x - 3)/(x² + 4)
12. f(x) = eˣ/(1 + eˣ)

### Level 4: Mixed
13. f(x) = (x² - 1)(x³ + x)/(x + 2)
14. Find f'(1) if f(x) = x³eˣ
15. Find the equation of the tangent line to y = x²/(x+1) at x = 1

### Level 5: Theoretical
16. Prove: If f(x) = xg(x), then f'(x) = g(x) + xg'(x)
17. Find a formula for d/dx[1/f(x)] in terms of f and f'
18. If (fg)' = f'g' for all x, what can you say about f and g?

---

## 📊 Answers

1. 7x⁶
2. 20x³ - 4x
3. 1/(2√x) - 1/x²
4. (2/3)x^(-1/3) + (1/3)x^(-4/3)
5. 5x⁴ + 2x
6. 12x - 1
7. eˣ(x² + 2x + 1) = eˣ(x+1)²
8. (3/2)√x
9. -2/(x-1)²
10. 2x/(x²+1)²
11. (-2x² + 6x + 8)/(x²+4)²
12. eˣ/(1+eˣ)²
13. Use product/quotient rules systematically
14. f'(1) = 3e + e = 4e
15. y - 1/2 = (1/4)(x - 1)
16. Apply product rule with f(x) = x · g(x)
17. d/dx[1/f] = -f'/f²
18. One of them is constant, or special exponential relationship

---

## 🔬 Physics Connection

### Newton's Second Law
$$F = ma = m\frac{d^2x}{dt^2}$$

Force equals mass times the **second derivative** of position (acceleration).

In quantum mechanics, the Hamiltonian operator involves the second derivative:
$$\hat{H} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x)$$

---

## ✅ Daily Checklist

- [ ] Read Stewart 3.1-3.2
- [ ] Memorize all basic derivative rules
- [ ] Master the product rule format
- [ ] Master the quotient rule format
- [ ] Complete Level 1-3 problems
- [ ] Attempt Level 4-5 problems
- [ ] Verify answers by expanding (when possible)
- [ ] Create a rules summary card

---

## 📓 Reflection Questions

1. When should you use the product rule vs. expanding first?
2. Can you derive the quotient rule from the product rule?
3. Why is d/dx[eˣ] = eˣ so special?
4. How does the power rule fail for f(x) = xˣ?

---

## 🔜 Preview: Tomorrow

**Day 10: The Chain Rule**

The chain rule handles compositions: d/dx[f(g(x))] = f'(g(x)) · g'(x)

This is perhaps the most important differentiation rule!

---

*"The chain rule is the most important rule for computing derivatives."*
— James Stewart
