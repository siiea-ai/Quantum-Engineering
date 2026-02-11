# Day 37: Partial Derivatives

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Partial Derivative Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Computation Techniques |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define partial derivatives using limits
2. Compute partial derivatives by treating other variables as constants
3. Interpret partial derivatives geometrically
4. Calculate higher-order partial derivatives
5. Apply Clairaut's theorem on equality of mixed partials

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 14.3**: Partial Derivatives (pp. 911-924)

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.02SC
**Lecture: Partial Derivatives**

### Professor Leonard
**Calculus 3: Partial Derivatives**

### Khan Academy
**Partial derivatives introduction**

---

## 📖 Core Content: Partial Derivatives

### 1. Motivation

For f(x, y), we want to know:
- How does f change when only x changes? (y held constant)
- How does f change when only y changes? (x held constant)

These are **partial derivatives**.

### 2. Definition

> **Definition:** The **partial derivative of f with respect to x** at (a, b) is:
> $$f_x(a,b) = \lim_{h \to 0} \frac{f(a+h, b) - f(a, b)}{h}$$
> if this limit exists.

Similarly, the **partial derivative with respect to y**:
> $$f_y(a,b) = \lim_{h \to 0} \frac{f(a, b+h) - f(a, b)}{h}$$

### 3. Notation

All of these mean the same thing:

$$f_x = \frac{\partial f}{\partial x} = \frac{\partial}{\partial x}f(x,y) = \partial_x f = D_x f$$

$$f_y = \frac{\partial f}{\partial y} = \frac{\partial}{\partial y}f(x,y) = \partial_y f = D_y f$$

The symbol ∂ (partial) distinguishes from d (ordinary derivative).

### 4. Computation Rule

> **To find ∂f/∂x:** Treat y as a constant and differentiate with respect to x using standard rules.
>
> **To find ∂f/∂y:** Treat x as a constant and differentiate with respect to y.

This is the key insight that makes computation easy!

---

## ✏️ Worked Examples

### Example 1: Polynomial
Find both partial derivatives of f(x, y) = x³ + 3x²y - 2y².

$$\frac{\partial f}{\partial x} = 3x^2 + 6xy - 0 = 3x^2 + 6xy$$

(Treat y as constant: x³ → 3x², 3x²y → 6xy, -2y² → 0)

$$\frac{\partial f}{\partial y} = 0 + 3x^2 - 4y = 3x^2 - 4y$$

(Treat x as constant: x³ → 0, 3x²y → 3x², -2y² → -4y)

---

### Example 2: Product
Find the partial derivatives of f(x, y) = xy·e^(xy).

For ∂f/∂x (y is constant):
Using product rule with u = xy and v = e^(xy):
$$\frac{\partial f}{\partial x} = y \cdot e^{xy} + xy \cdot e^{xy} \cdot y = ye^{xy}(1 + xy)$$

For ∂f/∂y (x is constant):
$$\frac{\partial f}{\partial y} = x \cdot e^{xy} + xy \cdot e^{xy} \cdot x = xe^{xy}(1 + xy)$$

---

### Example 3: Trigonometric
Find the partial derivatives of f(x, y) = sin(x²y).

$$\frac{\partial f}{\partial x} = \cos(x^2y) \cdot 2xy = 2xy\cos(x^2y)$$

$$\frac{\partial f}{\partial y} = \cos(x^2y) \cdot x^2 = x^2\cos(x^2y)$$

---

### Example 4: Quotient
Find ∂f/∂x for f(x, y) = x/(x + y).

Using quotient rule:
$$\frac{\partial f}{\partial x} = \frac{(x+y) \cdot 1 - x \cdot 1}{(x+y)^2} = \frac{y}{(x+y)^2}$$

---

### Example 5: At a Specific Point
For f(x, y) = x²y - y³, find fₓ(2, 1) and f_y(2, 1).

First, find general partial derivatives:
$$f_x = 2xy, \quad f_y = x^2 - 3y^2$$

Then evaluate:
$$f_x(2, 1) = 2(2)(1) = 4$$
$$f_y(2, 1) = (2)^2 - 3(1)^2 = 4 - 3 = 1$$

---

## 📐 Geometric Interpretation

### 6. Partial Derivatives as Slopes

For z = f(x, y):

**∂f/∂x at (a, b)** = slope of the tangent line to the curve z = f(x, b) at x = a.

This is the slope in the x-direction (holding y = b constant).

**∂f/∂y at (a, b)** = slope of the tangent line to the curve z = f(a, y) at y = b.

This is the slope in the y-direction (holding x = a constant).

### 7. Visualization

Imagine slicing the surface with a plane:
- Plane y = b creates a curve; fₓ is its slope
- Plane x = a creates a curve; f_y is its slope

---

## 📖 Higher-Order Partial Derivatives

### 8. Second Partial Derivatives

We can take partial derivatives of partial derivatives:

$$f_{xx} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right) = \frac{\partial^2 f}{\partial x^2}$$

$$f_{yy} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial y}\right) = \frac{\partial^2 f}{\partial y^2}$$

**Mixed partial derivatives:**
$$f_{xy} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right) = \frac{\partial^2 f}{\partial y \partial x}$$

$$f_{yx} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial y}\right) = \frac{\partial^2 f}{\partial x \partial y}$$

### 9. Clairaut's Theorem

> **Theorem (Clairaut):** If f is defined on a disk D containing (a, b), and both f_{xy} and f_{yx} are continuous on D, then:
> $$f_{xy}(a, b) = f_{yx}(a, b)$$

**In practice:** For "nice" functions, the order of mixed partial derivatives doesn't matter!

### Example 6: Second Partials
Find all second partial derivatives of f(x, y) = x³y² + 2xy³.

First partials:
$$f_x = 3x^2y^2 + 2y^3, \quad f_y = 2x^3y + 6xy^2$$

Second partials:
$$f_{xx} = 6xy^2$$
$$f_{yy} = 2x^3 + 12xy$$
$$f_{xy} = 6x^2y + 6y^2$$
$$f_{yx} = 6x^2y + 6y^2$$

Note: f_{xy} = f_{yx} ✓ (Clairaut's theorem)

---

## 📖 Functions of More Variables

### 10. Three or More Variables

For f(x, y, z):
$$\frac{\partial f}{\partial x}: \text{treat } y, z \text{ as constants}$$
$$\frac{\partial f}{\partial y}: \text{treat } x, z \text{ as constants}$$
$$\frac{\partial f}{\partial z}: \text{treat } x, y \text{ as constants}$$

### Example 7: Three Variables
Find all first partial derivatives of f(x, y, z) = xy²z³ + 2xz.

$$f_x = y^2z^3 + 2z$$
$$f_y = 2xyz^3$$
$$f_z = 3xy^2z^2 + 2x$$

---

## 📝 Practice Problems

### Level 1: Basic Computation
1. f(x, y) = x⁴ - 3x²y + y³. Find fₓ and f_y.
2. f(x, y) = √(x² + y²). Find both partial derivatives.
3. f(x, y) = e^(x+y). Find fₓ and f_y.
4. f(x, y) = ln(x² + y). Find both partial derivatives.

### Level 2: Product and Quotient
5. f(x, y) = xe^y. Find fₓ and f_y.
6. f(x, y) = (x - y)/(x + y). Find both partial derivatives.
7. f(x, y) = x·sin(xy). Find fₓ and f_y.
8. f(x, y) = arctan(y/x). Find both partial derivatives.

### Level 3: Evaluation
9. For f(x, y) = x²e^y, find fₓ(1, 0) and f_y(1, 0).
10. For f(x, y) = sin(xy), find fₓ(π, 1/2) and f_y(π, 1/2).

### Level 4: Higher Order
11. Find f_{xx}, f_{yy}, f_{xy}, f_{yx} for f(x, y) = x⁴y³ - x²y⁵.
12. Verify Clairaut's theorem for f(x, y) = sin(xy).
13. Find f_{xyz} for f(x, y, z) = x²yz³.

### Level 5: Applications
14. The temperature at point (x, y) is T(x, y) = 100 - x² - 2y². Find the rate of change of T at (3, 2) in the x-direction.
15. The volume of a cylinder is V = πr²h. Find ∂V/∂r and ∂V/∂h and interpret.

---

## 📊 Answers

1. fₓ = 4x³ - 6xy, f_y = -3x² + 3y²
2. fₓ = x/√(x²+y²), f_y = y/√(x²+y²)
3. fₓ = e^(x+y), f_y = e^(x+y)
4. fₓ = 2x/(x²+y), f_y = 1/(x²+y)
5. fₓ = e^y, f_y = xe^y
6. fₓ = 2y/(x+y)², f_y = -2x/(x+y)²
7. fₓ = sin(xy) + xy·cos(xy), f_y = x²·cos(xy)
8. fₓ = -y/(x²+y²), f_y = x/(x²+y²)
9. fₓ(1,0) = 2, f_y(1,0) = 1
10. fₓ(π,1/2) = (1/2)cos(π/2) = 0, f_y(π,1/2) = π·cos(π/2) = 0
11. f_{xx} = 12x²y³ - 2y⁵, f_{yy} = 6x⁴y - 20x²y³, f_{xy} = f_{yx} = 12x³y² - 10xy⁴
12. f_{xy} = cos(xy) - xy·sin(xy) = f_{yx}
13. f_{xyz} = 6x²z²
14. ∂T/∂x|_{(3,2)} = -6
15. ∂V/∂r = 2πrh (rate of volume change per unit radius), ∂V/∂h = πr² (rate per unit height)

---

## 🔬 Quantum Mechanics Connection

### The Schrödinger Equation

The time-dependent Schrödinger equation involves partial derivatives:

$$i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2} + V\psi$$

- **∂ψ/∂t**: How the wave function changes in time
- **∂²ψ/∂x²**: The curvature of the wave function in space

### Physical Interpretation

The Laplacian operator:
$$\nabla^2\psi = \frac{\partial^2\psi}{\partial x^2} + \frac{\partial^2\psi}{\partial y^2} + \frac{\partial^2\psi}{\partial z^2}$$

measures how ψ differs from its average in nearby regions!

---

## ✅ Daily Checklist

- [ ] Read Stewart 14.3
- [ ] Understand partial derivative definition
- [ ] Master the "treat as constant" rule
- [ ] Compute partial derivatives fluently
- [ ] Find higher-order partial derivatives
- [ ] Verify Clairaut's theorem
- [ ] Complete practice problems

---

## 📓 Reflection Questions

1. How is a partial derivative different from an ordinary derivative?
2. What does ∂f/∂x tell you geometrically?
3. Why does Clairaut's theorem require continuity?
4. How do partial derivatives appear in physics?

---

## 🔜 Preview: Tomorrow

**Day 38: The Gradient and Directional Derivatives**
- The gradient vector ∇f
- Directional derivatives
- Rate of change in any direction
- Gradient points in direction of steepest ascent

---

*"Partial derivatives let us isolate the effect of each variable—a powerful analytical tool."*
