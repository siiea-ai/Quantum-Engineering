# Day 8: The Derivative — Definition and Geometric Meaning

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Derivative Concept |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Limit Definition |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Applications |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Explain the geometric meaning of the derivative
2. Compute derivatives using the limit definition
3. Understand the relationship between differentiability and continuity
4. Calculate instantaneous velocity from position functions
5. Interpret derivatives in physical contexts

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 2.7**: Derivatives and Rates of Change (pp. 143-154)
- **Section 2.8**: The Derivative as a Function (pp. 155-166)

### Focus Areas
- Definition of derivative as a limit
- Tangent line interpretation
- Differentiability vs. continuity

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC
**Lecture 1: Derivatives, Slope, Velocity, Rate of Change**
- Full lecture now that you have limit foundation
- URL: ocw.mit.edu/18-01SC

### 3Blue1Brown - Essence of Calculus
**Chapter 2: The paradox of the derivative**
- Beautiful visual explanation
- ~17 minutes

### Professor Leonard
**Calculus 1: The Definition of a Derivative**
- Comprehensive with many examples

---

## 📖 Core Content: The Derivative

### 1. Motivation: The Tangent Line Problem

**Question:** Given a curve y = f(x), what is the slope of the tangent line at point (a, f(a))?

**Approach:** 
- We can't use two points (tangent touches at one point)
- Use the **secant line** through (a, f(a)) and (a+h, f(a+h))
- Take the limit as h → 0

**Secant slope:** 
$$m_{sec} = \frac{f(a+h) - f(a)}{h}$$

**Tangent slope:**
$$m_{tan} = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

### 2. The Definition of the Derivative

> **Definition:** The **derivative of f at a**, denoted f'(a), is:
> $$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$
> provided this limit exists.

**Alternative notation:**
- f'(a) — Lagrange notation
- $\frac{df}{dx}\bigg|_{x=a}$ — Leibniz notation  
- Df(a) — Operator notation
- $\dot{f}$ — Newton notation (for time derivatives)

### 3. The Derivative as a Function

> **Definition:** The **derivative function** f'(x) is:
> $$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

The domain of f' is all x where this limit exists.

### 4. Geometric Interpretation

```
    y
    │           ╱ tangent line
    │         ╱   slope = f'(a)
    │    ●───╱
    │   ╱ ╲╱  curve y = f(x)
    │  ╱  
    │ ╱
    └──────────────── x
           a
```

**The derivative f'(a) IS the slope of the tangent line to y = f(x) at x = a.**

### 5. Physical Interpretation: Velocity

If s(t) represents position at time t:

**Average velocity** over [t, t+h]:
$$v_{avg} = \frac{s(t+h) - s(t)}{h}$$

**Instantaneous velocity** at time t:
$$v(t) = \lim_{h \to 0} \frac{s(t+h) - s(t)}{h} = s'(t)$$

**Key insight:** Velocity is the derivative of position!

### 6. Differentiability and Continuity

> **Theorem:** If f is differentiable at a, then f is continuous at a.

**Proof:**
We need to show $\lim_{x \to a} f(x) = f(a)$.

$$\lim_{x \to a} [f(x) - f(a)] = \lim_{x \to a} \frac{f(x) - f(a)}{x - a} \cdot (x - a)$$
$$= f'(a) \cdot 0 = 0$$

Therefore $\lim_{x \to a} f(x) = f(a)$. ∎

> **Warning:** The converse is FALSE!

Continuity does NOT imply differentiability.

**Counterexample:** f(x) = |x| at x = 0
- f is continuous at 0
- But f is NOT differentiable at 0 (corner/cusp)

### 7. When Derivatives Fail to Exist

A function fails to be differentiable at a point when:

1. **Corner/Cusp:** Sharp point (e.g., |x| at 0)
2. **Vertical Tangent:** Infinite slope (e.g., ∛x at 0)
3. **Discontinuity:** Function not continuous

---

## ✏️ Worked Examples

### Example 1: Derivative of f(x) = x² at a = 3

Using the definition:
$$f'(3) = \lim_{h \to 0} \frac{(3+h)^2 - 3^2}{h}$$
$$= \lim_{h \to 0} \frac{9 + 6h + h^2 - 9}{h}$$
$$= \lim_{h \to 0} \frac{6h + h^2}{h}$$
$$= \lim_{h \to 0} (6 + h) = 6$$

**Interpretation:** The tangent line to y = x² at (3, 9) has slope 6.

---

### Example 2: Derivative Function of f(x) = x²

$$f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h}$$
$$= \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h}$$
$$= \lim_{h \to 0} \frac{2xh + h^2}{h}$$
$$= \lim_{h \to 0} (2x + h) = 2x$$

**Result:** If f(x) = x², then f'(x) = 2x.

---

### Example 3: Derivative of f(x) = √x

$$f'(x) = \lim_{h \to 0} \frac{\sqrt{x+h} - \sqrt{x}}{h}$$

Rationalize:
$$= \lim_{h \to 0} \frac{(\sqrt{x+h} - \sqrt{x})(\sqrt{x+h} + \sqrt{x})}{h(\sqrt{x+h} + \sqrt{x})}$$
$$= \lim_{h \to 0} \frac{(x+h) - x}{h(\sqrt{x+h} + \sqrt{x})}$$
$$= \lim_{h \to 0} \frac{h}{h(\sqrt{x+h} + \sqrt{x})}$$
$$= \lim_{h \to 0} \frac{1}{\sqrt{x+h} + \sqrt{x}} = \frac{1}{2\sqrt{x}}$$

**Result:** If f(x) = √x, then f'(x) = 1/(2√x) for x > 0.

---

### Example 4: f(x) = 1/x

$$f'(x) = \lim_{h \to 0} \frac{\frac{1}{x+h} - \frac{1}{x}}{h}$$
$$= \lim_{h \to 0} \frac{\frac{x - (x+h)}{x(x+h)}}{h}$$
$$= \lim_{h \to 0} \frac{-h}{hx(x+h)}$$
$$= \lim_{h \to 0} \frac{-1}{x(x+h)} = \frac{-1}{x^2}$$

**Result:** If f(x) = 1/x, then f'(x) = -1/x² for x ≠ 0.

---

### Example 5: Non-Differentiability

Show f(x) = |x| is not differentiable at x = 0.

Left-hand derivative:
$$\lim_{h \to 0^-} \frac{|0+h| - |0|}{h} = \lim_{h \to 0^-} \frac{|h|}{h} = \lim_{h \to 0^-} \frac{-h}{h} = -1$$

Right-hand derivative:
$$\lim_{h \to 0^+} \frac{|h|}{h} = \lim_{h \to 0^+} \frac{h}{h} = 1$$

Since -1 ≠ 1, the derivative does not exist at x = 0.

---

## 📝 Practice Problems

### Level 1: Definition Practice
Use the limit definition to find f'(x):

1. f(x) = 3x + 5
2. f(x) = x² + 2x
3. f(x) = x³
4. f(x) = 1/(x+1)

### Level 2: Evaluating Derivatives
Find f'(a) for the given value of a:

5. f(x) = x² - 4x, a = 3
6. f(x) = √(x+1), a = 3
7. f(x) = 1/√x, a = 4

### Level 3: Tangent Lines
8. Find the equation of the tangent line to y = x² at (2, 4).
9. Find the equation of the tangent line to y = 1/x at (1, 1).
10. At what point on y = x² is the tangent line horizontal?

### Level 4: Velocity
A ball is thrown upward with position s(t) = 40t - 16t² feet.

11. Find the velocity function v(t).
12. What is the velocity at t = 1?
13. When does the ball reach its maximum height?
14. What is the velocity when the ball hits the ground?

### Level 5: Differentiability
15. Is f(x) = |x - 2| differentiable at x = 2? Prove your answer.
16. Is f(x) = x^(1/3) differentiable at x = 0? Prove your answer.
17. Find all points where f(x) = |x² - 4| is not differentiable.

---

## 📊 Answers

1. f'(x) = 3
2. f'(x) = 2x + 2
3. f'(x) = 3x²
4. f'(x) = -1/(x+1)²
5. f'(3) = 2
6. f'(3) = 1/4
7. f'(4) = -1/16
8. y - 4 = 4(x - 2), or y = 4x - 4
9. y - 1 = -1(x - 1), or y = -x + 2
10. x = 0 (where f'(x) = 2x = 0)
11. v(t) = 40 - 32t
12. v(1) = 8 ft/s
13. t = 1.25 s (when v = 0)
14. Ball hits at t = 2.5 s, v(2.5) = -40 ft/s
15. Not differentiable (corner at x = 2)
16. Not differentiable (vertical tangent, f'(0) = lim(h→0) h^(-2/3) = ∞)
17. x = -2 and x = 2

---

## 🔬 Connection to Physics

### The Schrödinger Equation

The time-dependent Schrödinger equation:
$$i\hbar \frac{\partial \Psi}{\partial t} = \hat{H}\Psi$$

This involves a **time derivative** of the wave function!

### Classical to Quantum

In classical mechanics:
- Position: x(t)
- Velocity: v(t) = dx/dt = x'(t)
- Acceleration: a(t) = dv/dt = x''(t)

In quantum mechanics, the wave function ψ replaces position, and its derivatives describe how probability amplitudes evolve.

---

## ✅ Daily Checklist

- [ ] Read Stewart 2.7-2.8
- [ ] Watch 3Blue1Brown Chapter 2
- [ ] Master the definition f'(x) = lim[f(x+h)-f(x)]/h
- [ ] Complete Examples 1-5 independently
- [ ] Solve Level 1-3 practice problems
- [ ] Understand why differentiability implies continuity
- [ ] Know examples where continuity ≠ differentiability
- [ ] Create flashcards for key definitions

---

## 📓 Reflection Questions

1. In your own words, why is the derivative a limit of secant slopes?
2. Why does |x| have a corner at x = 0?
3. How does the derivative relate to velocity?
4. Give a real-world example where you'd want to know an instantaneous rate of change.

---

## 🔜 Preview: Tomorrow

**Day 9: Differentiation Rules**
- Power rule: d/dx[xⁿ] = nxⁿ⁻¹
- Sum/difference rules
- Constant multiple rule
- Product rule

Tomorrow we'll learn shortcuts that make computing derivatives much faster!

---

*"The derivative is one of the most beautiful and powerful ideas in all of mathematics."*
— Gilbert Strang
