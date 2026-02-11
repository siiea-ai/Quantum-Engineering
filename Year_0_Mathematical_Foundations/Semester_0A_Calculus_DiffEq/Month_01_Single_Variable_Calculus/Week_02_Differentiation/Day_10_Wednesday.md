# Day 10: The Chain Rule — Compositions of Functions

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Chain Rule Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Complex Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State and understand the chain rule
2. Identify the "outer" and "inner" functions in a composition
3. Apply the chain rule to complex compositions
4. Combine the chain rule with other differentiation rules
5. Use Leibniz notation for chain rule problems

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 3.4**: The Chain Rule (pp. 199-209)

### Focus Areas
- Composition of functions review
- Chain rule in both notations
- Recognizing when to use chain rule

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC
**Lecture on Chain Rule**

### 3Blue1Brown
**Chapter 4: Visualizing the chain rule**
- Exceptional geometric intuition
- ~10 minutes

### Professor Leonard
**Calculus 1: The Chain Rule**
- Many worked examples

---

## 📖 Core Content: The Chain Rule

### 1. Motivation: Why We Need the Chain Rule

Consider f(x) = (x² + 1)¹⁰⁰

**Method 1:** Expand using binomial theorem (101 terms!) — Impractical!

**Method 2:** Recognize this as a composition and use the chain rule.

### 2. Composition of Functions Review

If y = f(u) and u = g(x), then y = f(g(x)) is the **composition**.

**Example:** 
- Outer function: f(u) = u¹⁰⁰
- Inner function: u = g(x) = x² + 1
- Composition: f(g(x)) = (x² + 1)¹⁰⁰

### 3. The Chain Rule

> **Theorem (Chain Rule):** If g is differentiable at x and f is differentiable at g(x), then the composition f ∘ g is differentiable at x and:
> $$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$$

**Leibniz Notation:**
If y = f(u) and u = g(x), then:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Memory Aid:** "The derivative of the outside times the derivative of the inside"

### 4. Proof Sketch

$$\frac{d}{dx}[f(g(x))] = \lim_{h \to 0} \frac{f(g(x+h)) - f(g(x))}{h}$$

Let Δu = g(x+h) - g(x). Then:
$$= \lim_{h \to 0} \frac{f(g(x) + \Delta u) - f(g(x))}{\Delta u} \cdot \frac{\Delta u}{h}$$

As h → 0, Δu → 0 (since g is continuous), giving us:
$$= f'(g(x)) \cdot g'(x)$$

### 5. The General Power Rule

> **Corollary:** If n is any real number and u = g(x) is differentiable, then:
> $$\frac{d}{dx}[u^n] = nu^{n-1} \cdot \frac{du}{dx}$$

Or equivalently:
$$\frac{d}{dx}[g(x)^n] = n[g(x)]^{n-1} \cdot g'(x)$$

---

## 📋 Chain Rule Strategy

### Step-by-Step Method:

1. **Identify** the composition: What's inside what?
2. **Label** u = inner function
3. **Find** du/dx (derivative of inner)
4. **Find** dy/du (derivative of outer, leaving inner alone)
5. **Multiply** dy/dx = (dy/du)(du/dx)
6. **Substitute** back u = g(x)

### Recognition Patterns:

| Expression | Outer f(u) | Inner u = g(x) |
|------------|------------|----------------|
| (3x + 1)⁵ | u⁵ | 3x + 1 |
| √(x² + 1) | √u | x² + 1 |
| sin(x²) | sin(u) | x² |
| e^(2x) | e^u | 2x |
| ln(x³ + x) | ln(u) | x³ + x |

---

## ✏️ Worked Examples

### Example 1: Simple Power
$$\frac{d}{dx}[(3x + 1)^5]$$

- Outer: u⁵, derivative = 5u⁴
- Inner: u = 3x + 1, derivative = 3

$$= 5(3x + 1)^4 \cdot 3 = 15(3x + 1)^4$$

---

### Example 2: Square Root
$$\frac{d}{dx}[\sqrt{x^2 + 1}]$$

Rewrite as (x² + 1)^(1/2)

- Outer: u^(1/2), derivative = (1/2)u^(-1/2)
- Inner: u = x² + 1, derivative = 2x

$$= \frac{1}{2}(x^2 + 1)^{-1/2} \cdot 2x = \frac{x}{\sqrt{x^2 + 1}}$$

---

### Example 3: Trigonometric
$$\frac{d}{dx}[\sin(x^2)]$$

- Outer: sin(u), derivative = cos(u)
- Inner: u = x², derivative = 2x

$$= \cos(x^2) \cdot 2x = 2x\cos(x^2)$$

---

### Example 4: Exponential
$$\frac{d}{dx}[e^{3x^2 + 2x}]$$

- Outer: e^u, derivative = e^u
- Inner: u = 3x² + 2x, derivative = 6x + 2

$$= e^{3x^2 + 2x} \cdot (6x + 2) = (6x + 2)e^{3x^2 + 2x}$$

---

### Example 5: Logarithmic
$$\frac{d}{dx}[\ln(x^3 + 5x)]$$

- Outer: ln(u), derivative = 1/u
- Inner: u = x³ + 5x, derivative = 3x² + 5

$$= \frac{1}{x^3 + 5x} \cdot (3x^2 + 5) = \frac{3x^2 + 5}{x^3 + 5x}$$

---

### Example 6: Double Chain (Nested Composition)
$$\frac{d}{dx}[\sin^2(3x)] = \frac{d}{dx}[(\sin(3x))^2]$$

Three layers: 
- Outermost: u², derivative = 2u
- Middle: sin(v), derivative = cos(v)  
- Innermost: v = 3x, derivative = 3

$$= 2\sin(3x) \cdot \cos(3x) \cdot 3 = 6\sin(3x)\cos(3x) = 3\sin(6x)$$

(Using identity: 2sin(θ)cos(θ) = sin(2θ))

---

### Example 7: Chain Rule with Product Rule
$$\frac{d}{dx}[x^2 \sin(3x)]$$

Use product rule, then chain rule on sin(3x):
$$= (2x)\sin(3x) + (x^2)[\cos(3x) \cdot 3]$$
$$= 2x\sin(3x) + 3x^2\cos(3x)$$

---

### Example 8: Chain Rule with Quotient Rule
$$\frac{d}{dx}\left[\frac{e^{2x}}{x + 1}\right]$$

Quotient rule with chain rule on e^(2x):
$$= \frac{(x+1)(2e^{2x}) - (e^{2x})(1)}{(x+1)^2}$$
$$= \frac{e^{2x}(2x + 2 - 1)}{(x+1)^2} = \frac{e^{2x}(2x + 1)}{(x+1)^2}$$

---

## 📝 Practice Problems

### Level 1: Basic Chain Rule
1. d/dx[(2x + 5)⁴]
2. d/dx[(x³ - 1)⁷]
3. d/dx[√(4x + 1)]
4. d/dx[(x² + 3x)^(-2)]

### Level 2: Trigonometric
5. d/dx[sin(5x)]
6. d/dx[cos(x²)]
7. d/dx[tan(3x + 1)]
8. d/dx[sin²(x)] = d/dx[(sin x)²]

### Level 3: Exponential and Logarithmic
9. d/dx[e^(5x)]
10. d/dx[e^(-x²)]
11. d/dx[ln(x² + 1)]
12. d/dx[ln(sin x)]

### Level 4: Multiple Chains
13. d/dx[sin(cos(x))]
14. d/dx[e^(sin x)]
15. d/dx[√(1 + e^x)]
16. d/dx[ln(ln x)]

### Level 5: Combined Rules
17. d/dx[x²e^(3x)]
18. d/dx[(x + 1)²(2x - 1)³]
19. d/dx[sin(x)/e^x]
20. d/dx[√x · sin(x²)]

### Level 6: Challenge
21. Find f'(0) if f(x) = sin(e^x)
22. Find the equation of the tangent line to y = e^(x²) at x = 1
23. If f(x) = [g(x)]³ and g(2) = 3, g'(2) = 4, find f'(2)
24. Prove: d/dx[ln|f(x)|] = f'(x)/f(x)

---

## 📊 Answers

1. 8(2x + 5)³
2. 21x²(x³ - 1)⁶
3. 2/√(4x + 1)
4. -2(2x + 3)(x² + 3x)^(-3)
5. 5cos(5x)
6. -2x sin(x²)
7. 3sec²(3x + 1)
8. 2sin(x)cos(x) = sin(2x)
9. 5e^(5x)
10. -2xe^(-x²)
11. 2x/(x² + 1)
12. cot(x)
13. -sin(x)cos(cos(x))
14. cos(x)e^(sin x)
15. e^x/(2√(1 + e^x))
16. 1/(x ln x)
17. e^(3x)(2x + 3x²)
18. (x+1)(2x-1)²(10x + 1)
19. (cos x - sin x)/e^x
20. sin(x²)/(2√x) + 2x√x cos(x²)
21. cos(1)
22. y - e = 2e(x - 1)
23. f'(2) = 3(9)(4) = 108
24. Use chain rule with u = |f(x)|

---

## 🔬 Physics Application: Time-Dependent Systems

In physics, we often have quantities that depend on other quantities that vary with time.

**Example:** Temperature T depends on position x, and position depends on time t.

$$\frac{dT}{dt} = \frac{dT}{dx} \cdot \frac{dx}{dt}$$

This is the chain rule! The rate of temperature change with time equals (temperature gradient) × (velocity).

### Quantum Mechanics Connection

The time evolution of quantum states involves:
$$\frac{d}{dt}\langle A \rangle = \frac{1}{i\hbar}\langle [A, H] \rangle + \left\langle \frac{\partial A}{\partial t} \right\rangle$$

Chain rule appears when observables depend on parameters that change with time.

---

## ✅ Daily Checklist

- [ ] Read Stewart 3.4
- [ ] Watch 3Blue1Brown chain rule video
- [ ] Master identification of inner/outer functions
- [ ] Complete Level 1-3 problems
- [ ] Attempt Level 4-5 problems
- [ ] Practice Leibniz notation
- [ ] Combine chain rule with product/quotient rules
- [ ] Create chain rule flowchart

---

## 📓 Reflection Questions

1. How do you identify when to use the chain rule?
2. Why is the chain rule sometimes called the "function of a function" rule?
3. Can you have a triple chain (three nested functions)? How would you handle it?
4. How does Leibniz notation make the chain rule intuitive?

---

## 🔜 Preview: Tomorrow

**Day 11: Implicit Differentiation**

What if y isn't explicitly written as a function of x?
- Equations like x² + y² = 1 (circle)
- Technique: Differentiate both sides, solve for dy/dx

---

*"The chain rule is perhaps the most important differentiation formula. Without it, we could differentiate relatively few functions."*
— James Stewart
