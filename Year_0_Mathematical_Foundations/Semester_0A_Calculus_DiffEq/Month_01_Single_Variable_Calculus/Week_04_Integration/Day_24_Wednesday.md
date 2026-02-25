# Day 24: The Fundamental Theorem of Calculus

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | FTC Part 1 |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | FTC Part 2 |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State both parts of the Fundamental Theorem of Calculus
2. Understand why FTC is so important
3. Apply FTC Part 1 to differentiate integrals
4. Apply FTC Part 2 to evaluate definite integrals
5. Use FTC with chain rule (extended version)

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 5.3**: The Fundamental Theorem of Calculus (pp. 389-400)

---

## 🎬 Video Resources

### 3Blue1Brown
**Chapter 8: Integration and the fundamental theorem of calculus**
- The best visual explanation available

### MIT OpenCourseWare 18.01SC
**Lecture: FTC**

### Professor Leonard
**The Fundamental Theorem of Calculus**

---

## 📖 The Fundamental Theorem of Calculus

### Why It's Fundamental

The FTC reveals that **differentiation and integration are inverse operations**.

This is remarkable! Two seemingly different concepts:
- Derivatives: rates of change, tangent slopes
- Integrals: areas, accumulation

are intimately connected through this theorem.

---

## 📖 FTC Part 1: Differentiation of Integrals

### Statement

> **Fundamental Theorem of Calculus, Part 1:**
> If f is continuous on [a, b], then the function g defined by
> $$g(x) = \int_a^x f(t) \, dt, \quad a \leq x \leq b$$
> is continuous on [a, b], differentiable on (a, b), and
> $$g'(x) = f(x)$$

### In Leibniz Notation

$$\frac{d}{dx}\left[\int_a^x f(t) \, dt\right] = f(x)$$

### Interpretation

- g(x) = ∫ₐˣ f(t)dt is the "area so far" under f from a to x
- g'(x) = f(x) means: the rate of change of area equals the height

This makes geometric sense!

### Proof Sketch

$$g'(x) = \lim_{h \to 0} \frac{g(x+h) - g(x)}{h} = \lim_{h \to 0} \frac{1}{h}\int_x^{x+h} f(t) \, dt$$

For small h, ∫ₓˣ⁺ʰ f(t)dt ≈ f(x) · h (rectangle approximation)

So g'(x) = lim[h→0] f(x) · h / h = f(x).

---

## ✏️ FTC Part 1 Examples

### Example 1: Direct Application
$$\frac{d}{dx}\left[\int_1^x t^3 \, dt\right] = x^3$$

Just substitute the upper limit for the variable!

---

### Example 2: Lower Limit is Variable
$$\frac{d}{dx}\left[\int_x^5 \sin(t) \, dt\right]$$

Reverse limits (introduces minus sign):
$$= \frac{d}{dx}\left[-\int_5^x \sin(t) \, dt\right] = -\sin(x)$$

---

### Example 3: FTC with Chain Rule
$$\frac{d}{dx}\left[\int_0^{x^2} e^{-t^2} \, dt\right]$$

Let u = x², then by chain rule:
$$= e^{-(x^2)^2} \cdot \frac{d}{dx}[x^2] = e^{-x^4} \cdot 2x = 2xe^{-x^4}$$

**General Formula:**
$$\frac{d}{dx}\left[\int_a^{g(x)} f(t) \, dt\right] = f(g(x)) \cdot g'(x)$$

---

### Example 4: Both Limits are Functions
$$\frac{d}{dx}\left[\int_{x}^{x^3} \cos(t^2) \, dt\right]$$

Split using properties:
$$= \frac{d}{dx}\left[\int_0^{x^3} \cos(t^2) \, dt - \int_0^{x} \cos(t^2) \, dt\right]$$

$$= \cos((x^3)^2) \cdot 3x^2 - \cos(x^2) \cdot 1$$

$$= 3x^2\cos(x^6) - \cos(x^2)$$

---

## 📖 FTC Part 2: Evaluation of Integrals

### Statement

> **Fundamental Theorem of Calculus, Part 2:**
> If f is continuous on [a, b] and F is any antiderivative of f (so F' = f), then
> $$\int_a^b f(x) \, dx = F(b) - F(a)$$

### Notation

We write: $F(x)\Big|_a^b = F(b) - F(a)$

Or: $[F(x)]_a^b = F(b) - F(a)$

### Why It Works

By FTC Part 1, g(x) = ∫ₐˣ f(t)dt is an antiderivative of f.

So g(x) = F(x) + C for some constant C.

g(a) = ∫ₐᵃ f(t)dt = 0, so F(a) + C = 0, meaning C = -F(a).

Therefore: g(b) = F(b) - F(a)

And g(b) = ∫ₐᵇ f(t)dt. ∎

---

## ✏️ FTC Part 2 Examples

### Example 5: Polynomial
$$\int_1^4 x^2 \, dx = \left[\frac{x^3}{3}\right]_1^4 = \frac{64}{3} - \frac{1}{3} = \frac{63}{3} = 21$$

---

### Example 6: Trigonometric
$$\int_0^{\pi} \sin(x) \, dx = [-\cos(x)]_0^{\pi} = -\cos(\pi) - (-\cos(0))$$
$$= -(-1) - (-1) = 1 + 1 = 2$$

---

### Example 7: Exponential
$$\int_0^1 e^x \, dx = [e^x]_0^1 = e^1 - e^0 = e - 1$$

---

### Example 8: Involving ln
$$\int_1^e \frac{1}{x} \, dx = [\ln|x|]_1^e = \ln(e) - \ln(1) = 1 - 0 = 1$$

---

### Example 9: Sum of Functions
$$\int_0^{\pi/4} (2\cos x - \sec^2 x) \, dx$$

$$= [2\sin x - \tan x]_0^{\pi/4}$$

$$= \left(2 \cdot \frac{\sqrt{2}}{2} - 1\right) - (0 - 0)$$

$$= \sqrt{2} - 1$$

---

### Example 10: Absolute Value (Split Integral)
$$\int_{-2}^{3} |x| \, dx$$

|x| = -x for x < 0, |x| = x for x ≥ 0

$$= \int_{-2}^{0} (-x) \, dx + \int_{0}^{3} x \, dx$$

$$= \left[-\frac{x^2}{2}\right]_{-2}^{0} + \left[\frac{x^2}{2}\right]_{0}^{3}$$

$$= (0 - (-2)) + \left(\frac{9}{2} - 0\right) = 2 + \frac{9}{2} = \frac{13}{2}$$

---

## 📊 Summary: The Two Parts

| Part | What It Says | Used For |
|------|--------------|----------|
| **Part 1** | d/dx[∫ₐˣ f(t)dt] = f(x) | Differentiating integrals |
| **Part 2** | ∫ₐᵇ f(x)dx = F(b) - F(a) | Evaluating definite integrals |

**Part 2 is the computational workhorse!**

---

## 📝 Practice Problems

### FTC Part 1
Find the derivative:

1. $\frac{d}{dx}\int_2^x t^4 \, dt$

2. $\frac{d}{dx}\int_x^0 \cos(t^2) \, dt$

3. $\frac{d}{dx}\int_1^{x^2} \sqrt{t+1} \, dt$

4. $\frac{d}{dx}\int_{2x}^{3x} \ln(t) \, dt$

### FTC Part 2
Evaluate:

5. $\int_0^2 (3x^2 - 2x + 1) \, dx$

6. $\int_1^4 \sqrt{x} \, dx$

7. $\int_0^{\pi/2} \cos(x) \, dx$

8. $\int_1^e \frac{3}{x} \, dx$

9. $\int_0^1 (e^x + e^{-x}) \, dx$

10. $\int_{-1}^{1} |2x - 1| \, dx$

### Mixed Problems

11. If $F(x) = \int_0^x \frac{1}{1+t^2} \, dt$, find F'(x) and F(1).

12. Find the area under y = x² from x = 0 to x = 3.

13. Find the area between y = sin(x) and the x-axis from x = 0 to x = 2π.

---

## 📊 Answers

1. x⁴
2. cos(x²)
3. 2x√(x² + 1)
4. 3ln(3x) - 2ln(2x) = ln(27x/4x) + ... simplify
5. 6
6. 14/3
7. 1
8. 3
9. e + 1/e - 2
10. 5/2
11. F'(x) = 1/(1+x²), F(1) = π/4 (arctan!)
12. 9
13. 4 (two humps, each has area 2)

---

## 🔬 Quantum Mechanics Connection

**The momentum operator:**
$$\hat{p} = -i\hbar\frac{d}{dx}$$

When we compute ⟨p⟩, we integrate then differentiate. FTC shows these operations are inverses!

**Probability currents** involve:
$$j = \frac{\hbar}{2mi}\left(\psi^*\frac{d\psi}{dx} - \psi\frac{d\psi^*}{dx}\right)$$

Conservation of probability: ∂ρ/∂t + ∂j/∂x = 0, which is verified using FTC!

---

## ✅ Daily Checklist

- [ ] Read Stewart 5.3
- [ ] Understand both parts of FTC
- [ ] Know when to use each part
- [ ] Apply FTC Part 1 with chain rule
- [ ] Evaluate definite integrals using FTC Part 2
- [ ] Complete practice problems
- [ ] Appreciate the profound connection!

---

## 📓 Reflection Questions

1. Why is FTC called "fundamental"?
2. How does FTC Part 1 show that integration reverses differentiation?
3. Why don't we need +C in FTC Part 2?
4. What's the geometric meaning of FTC Part 1?

---

## 🔜 Preview: Tomorrow

**Day 25: Integration by Substitution**

The chain rule for integration! If ∫f(g(x))g'(x)dx = F(g(x)) + C, we can use u = g(x) to simplify.

---

*"The Fundamental Theorem of Calculus is one of the greatest achievements of the human mind."*
— Richard Feynman
