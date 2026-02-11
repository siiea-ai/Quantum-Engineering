# Day 11: Implicit Differentiation

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Implicit Differentiation Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications to Curves |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Recognize when implicit differentiation is needed
2. Apply implicit differentiation to find dy/dx
3. Find tangent lines to implicitly defined curves
4. Compute second derivatives implicitly
5. Apply to inverse functions

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 3.5**: Implicit Differentiation (pp. 210-218)

### Key Focus
- When explicit solutions for y are impossible or impractical
- Treating y as a function of x throughout
- Chain rule's crucial role

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.01SC
**Lecture: Implicit Differentiation**

### Professor Leonard
**Implicit Differentiation** - Very thorough explanation

### Organic Chemistry Tutor
**Implicit Differentiation Examples**

---

## 📖 Core Content: Implicit Differentiation

### 1. Explicit vs. Implicit Functions

**Explicit:** y is written directly in terms of x
- Example: y = x² + 3x - 1

**Implicit:** y is defined through an equation involving both x and y
- Example: x² + y² = 25 (circle)

Some implicit equations cannot be solved for y explicitly, or yield multiple solutions.

### 2. The Key Idea

In implicit differentiation, we:
1. Differentiate both sides of the equation with respect to x
2. Treat y as a function of x (so we use the chain rule when differentiating y)
3. Solve for dy/dx

### 3. The Chain Rule Connection

When we differentiate a term involving y:
$$\frac{d}{dx}[y^2] = 2y \cdot \frac{dy}{dx}$$

We need dy/dx because y is a function of x!

### 4. The Method

> **Implicit Differentiation Process:**
> 1. Differentiate both sides with respect to x
> 2. Every time you differentiate a y-term, multiply by dy/dx
> 3. Collect all dy/dx terms on one side
> 4. Factor out dy/dx
> 5. Solve for dy/dx

---

## ✏️ Worked Examples

### Example 1: Circle
**Find dy/dx for:** x² + y² = 25

**Solution:**
Differentiate both sides with respect to x:
$$\frac{d}{dx}[x^2] + \frac{d}{dx}[y^2] = \frac{d}{dx}[25]$$

$$2x + 2y\frac{dy}{dx} = 0$$

Solve for dy/dx:
$$2y\frac{dy}{dx} = -2x$$
$$\frac{dy}{dx} = -\frac{x}{y}$$

**Interpretation:** At point (3, 4) on the circle, slope = -3/4.

---

### Example 2: Product of x and y
**Find dy/dx for:** xy = 1

**Solution:**
Use product rule on the left side:
$$\frac{d}{dx}[xy] = \frac{d}{dx}[1]$$

$$(1)(y) + (x)\frac{dy}{dx} = 0$$

$$y + x\frac{dy}{dx} = 0$$

$$\frac{dy}{dx} = -\frac{y}{x}$$

**Verification:** This is a hyperbola y = 1/x, so dy/dx = -1/x² = -y/x ✓

---

### Example 3: Cubic Curve
**Find dy/dx for:** x³ + y³ = 6xy (Folium of Descartes)

**Solution:**
$$\frac{d}{dx}[x^3] + \frac{d}{dx}[y^3] = \frac{d}{dx}[6xy]$$

$$3x^2 + 3y^2\frac{dy}{dx} = 6y + 6x\frac{dy}{dx}$$

Collect dy/dx terms:
$$3y^2\frac{dy}{dx} - 6x\frac{dy}{dx} = 6y - 3x^2$$

Factor:
$$(3y^2 - 6x)\frac{dy}{dx} = 6y - 3x^2$$

$$\frac{dy}{dx} = \frac{6y - 3x^2}{3y^2 - 6x} = \frac{2y - x^2}{y^2 - 2x}$$

---

### Example 4: Trigonometric
**Find dy/dx for:** sin(xy) = x

**Solution:**
Using chain rule on sin(xy):
$$\cos(xy) \cdot \frac{d}{dx}[xy] = 1$$

$$\cos(xy) \cdot \left(y + x\frac{dy}{dx}\right) = 1$$

$$y\cos(xy) + x\cos(xy)\frac{dy}{dx} = 1$$

$$x\cos(xy)\frac{dy}{dx} = 1 - y\cos(xy)$$

$$\frac{dy}{dx} = \frac{1 - y\cos(xy)}{x\cos(xy)}$$

---

### Example 5: Exponential
**Find dy/dx for:** e^y = x + y

**Solution:**
$$e^y \cdot \frac{dy}{dx} = 1 + \frac{dy}{dx}$$

$$e^y\frac{dy}{dx} - \frac{dy}{dx} = 1$$

$$(e^y - 1)\frac{dy}{dx} = 1$$

$$\frac{dy}{dx} = \frac{1}{e^y - 1}$$

---

### Example 6: Finding Tangent Lines
**Find the equation of the tangent line to x² + y² = 25 at point (3, 4).**

**Solution:**
From Example 1: dy/dx = -x/y

At (3, 4): slope = -3/4

Tangent line: y - 4 = -3/4(x - 3)

$$y = -\frac{3}{4}x + \frac{9}{4} + 4 = -\frac{3}{4}x + \frac{25}{4}$$

Or: 3x + 4y = 25

---

### Example 7: Second Derivative
**Find d²y/dx² for:** x² + y² = 25

**Solution:**
We found: dy/dx = -x/y

Now differentiate again (using quotient rule):
$$\frac{d^2y}{dx^2} = \frac{d}{dx}\left[-\frac{x}{y}\right]$$

$$= -\frac{y(1) - x\frac{dy}{dx}}{y^2}$$

$$= -\frac{y - x(-\frac{x}{y})}{y^2}$$

$$= -\frac{y + \frac{x^2}{y}}{y^2}$$

$$= -\frac{y^2 + x^2}{y^3}$$

Since x² + y² = 25:
$$\frac{d^2y}{dx^2} = -\frac{25}{y^3}$$

---

## 📐 Derivatives of Inverse Functions

Implicit differentiation helps find derivatives of inverse functions.

### Inverse Trigonometric Functions

**Example: Derivative of arcsin(x)**

Let y = arcsin(x), so sin(y) = x.

Differentiate implicitly:
$$\cos(y) \cdot \frac{dy}{dx} = 1$$
$$\frac{dy}{dx} = \frac{1}{\cos(y)}$$

Since sin²(y) + cos²(y) = 1 and sin(y) = x:
$$\cos(y) = \sqrt{1 - x^2}$$ (taking positive root for principal value)

Therefore:
$$\frac{d}{dx}[\arcsin(x)] = \frac{1}{\sqrt{1 - x^2}}$$

### Table of Inverse Trig Derivatives

| Function | Derivative |
|----------|------------|
| arcsin(x) | 1/√(1 - x²) |
| arccos(x) | -1/√(1 - x²) |
| arctan(x) | 1/(1 + x²) |
| arccot(x) | -1/(1 + x²) |
| arcsec(x) | 1/(|x|√(x² - 1)) |
| arccsc(x) | -1/(|x|√(x² - 1)) |

---

## 📝 Practice Problems

### Level 1: Basic Implicit Differentiation
Find dy/dx:
1. x² + y² = 16
2. x³ + y³ = 1
3. xy = 10
4. x² - y² = 1

### Level 2: Product and Chain Rules
5. x²y + xy² = 6
6. √x + √y = 4
7. (x + y)² = x - y
8. xy² + x²y = 2

### Level 3: Trigonometric and Exponential
9. sin(y) = x²
10. cos(x + y) = y
11. e^(xy) = x - y
12. ln(xy) = x + y

### Level 4: Tangent Lines
13. Find the tangent line to x² + y² = 25 at (-3, 4)
14. Find the tangent line to xy = 6 at (2, 3)
15. Find all points on x² + y² = 9 where the tangent is horizontal

### Level 5: Second Derivatives
16. Find d²y/dx² for xy = 1
17. Find d²y/dx² for x³ + y³ = 1 at point (0, 1)

### Level 6: Inverse Functions
18. Use implicit differentiation to find d/dx[arctan(x)]
19. Use implicit differentiation to find d/dx[arccos(x)]
20. If f(x) = x³ + x, find (f⁻¹)'(2) [Hint: f(1) = 2]

---

## 📊 Answers

1. dy/dx = -x/y
2. dy/dx = -x²/y²
3. dy/dx = -y/x
4. dy/dx = x/y
5. dy/dx = -(2xy + y²)/(x² + 2xy)
6. dy/dx = -√y/√x
7. dy/dx = (1 - 2(x+y))/(2(x+y) + 1)
8. dy/dx = -(y² + 2xy)/(2xy + x²)
9. dy/dx = 2x/cos(y)
10. dy/dx = sin(x+y)/(1 + sin(x+y))
11. dy/dx = (1 - ye^(xy))/(xe^(xy) + 1)
12. dy/dx = (y - 1)/(1 - x) or equivalently
13. y - 4 = (3/4)(x + 3)
14. y - 3 = -(3/2)(x - 2)
15. (0, 3) and (0, -3)
16. d²y/dx² = 2y/x³
17. d²y/dx² = 0
18. d/dx[arctan(x)] = 1/(1 + x²)
19. d/dx[arccos(x)] = -1/√(1 - x²)
20. (f⁻¹)'(2) = 1/f'(1) = 1/4

---

## 🔬 Physics Application: Constraint Equations

In physics, many systems have constraints:
- A particle moving on a sphere: x² + y² + z² = R²
- A pendulum: x² + y² = L²

Implicit differentiation finds velocities consistent with constraints.

### Quantum Mechanics Connection

The normalization condition for wave functions:
$$\int_{-\infty}^{\infty} |\psi|^2 dx = 1$$

When parameters in ψ vary, implicit differentiation helps maintain normalization.

---

## ✅ Daily Checklist

- [ ] Read Stewart 3.5
- [ ] Master the implicit differentiation process
- [ ] Complete Level 1-3 problems
- [ ] Find tangent lines to curves
- [ ] Derive inverse trig derivatives
- [ ] Practice second derivatives
- [ ] Attempt Level 4-6 problems
- [ ] Create summary of inverse trig derivatives

---

## 📓 Reflection Questions

1. When is implicit differentiation necessary vs. convenient?
2. Why must we use the chain rule when differentiating y?
3. How does implicit differentiation help with inverse functions?
4. What's the geometric meaning of dy/dx for a curve defined implicitly?

---

## 🔜 Preview: Tomorrow

**Day 12: Week 2 Problem Set**

Comprehensive practice covering:
- Limit definition of derivative
- Power, product, quotient rules
- Chain rule
- Implicit differentiation

---

*"The implicit function theorem is one of the most important theorems in analysis."*
— Walter Rudin
