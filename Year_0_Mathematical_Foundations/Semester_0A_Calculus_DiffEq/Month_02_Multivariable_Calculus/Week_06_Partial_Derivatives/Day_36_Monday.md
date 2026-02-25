# Day 36: Functions of Several Variables

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Multivariable Functions |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Visualization |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand functions of two or more variables
2. Determine domains of multivariable functions
3. Sketch and interpret level curves
4. Visualize surfaces in 3D
5. Understand limits and continuity in multiple variables

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 14.1**: Functions of Several Variables (pp. 888-900)
- **Section 14.2**: Limits and Continuity (pp. 901-910)

---

## 🎬 Video Resources

### 3Blue1Brown
**Multivariable Calculus series (if available)**

### MIT OpenCourseWare 18.02SC
**Lecture: Functions of Two Variables**

### Professor Leonard
**Calculus 3: Functions of Several Variables**

---

## 📖 Core Content: Functions of Two Variables

### 1. Definition

> **Definition:** A **function of two variables** is a rule that assigns to each ordered pair (x, y) in a set D (the **domain**) a unique real number f(x, y).

We write: z = f(x, y)

**Examples:**
- f(x, y) = x² + y² (paraboloid)
- f(x, y) = √(9 - x² - y²) (hemisphere)
- f(x, y) = xy (saddle surface)
- f(x, y) = sin(x)cos(y) (wave pattern)

### 2. Domain

The **domain** is the set of all (x, y) for which f(x, y) is defined.

**Example:** Find the domain of f(x, y) = √(9 - x² - y²).

We need 9 - x² - y² ≥ 0, so x² + y² ≤ 9.

Domain: {(x, y) : x² + y² ≤ 9} — a disk of radius 3.

### 3. Range

The **range** is the set of all possible output values.

For f(x, y) = √(9 - x² - y²):
- Minimum: 0 (when x² + y² = 9)
- Maximum: 3 (when x = y = 0)

Range: [0, 3]

---

## 📐 Graphs and Level Curves

### 4. Graphs in 3D

The **graph** of z = f(x, y) is a **surface** in 3D space.

| Function | Surface Type |
|----------|--------------|
| z = x² + y² | Paraboloid |
| z = √(r² - x² - y²) | Hemisphere |
| z = xy | Hyperbolic paraboloid (saddle) |
| z = c | Horizontal plane |

### 5. Level Curves (Contours)

A **level curve** of f(x, y) is the set of points where f(x, y) = k for some constant k.

$$\{(x, y) : f(x, y) = k\}$$

**Interpretation:** Level curves are like topographic map contour lines—they show points of equal "height."

**Example:** For f(x, y) = x² + y²:
- Level curve k = 1: x² + y² = 1 (circle, radius 1)
- Level curve k = 4: x² + y² = 4 (circle, radius 2)
- Level curve k = 9: x² + y² = 9 (circle, radius 3)

### 6. Spacing of Level Curves

- **Close together:** Steep slope (rapid change)
- **Far apart:** Gentle slope (slow change)

---

## 📖 Functions of Three or More Variables

### 7. Functions of Three Variables

For w = f(x, y, z):
- Domain is a region in 3D space
- Graph would be in 4D (can't visualize directly)
- Use **level surfaces**: f(x, y, z) = k

**Example:** f(x, y, z) = x² + y² + z²

Level surfaces are spheres centered at origin.

### 8. Notation

For f(x, y):
- **Variables:** x, y (independent), z (dependent)
- **Notation:** f(x, y), z(x, y), or just z

Partial derivative notation (preview):
$$\frac{\partial f}{\partial x}, \quad \frac{\partial f}{\partial y}, \quad f_x, \quad f_y$$

---

## 📖 Limits and Continuity

### 9. Limits in Two Variables

> **Definition:** We write $\lim_{(x,y) \to (a,b)} f(x,y) = L$ if for every ε > 0 there exists δ > 0 such that:
> $$0 < \sqrt{(x-a)^2 + (y-b)^2} < \delta \implies |f(x,y) - L| < \epsilon$$

**Key difference from 1D:** In 2D, (x, y) can approach (a, b) along infinitely many paths!

### 10. Path Independence

For a limit to exist, f(x, y) must approach the same value along **every** path to (a, b).

**Strategy to show limit doesn't exist:** Find two paths giving different limits.

**Example:** Show $\lim_{(x,y) \to (0,0)} \frac{xy}{x^2 + y^2}$ doesn't exist.

Along y = 0: $\frac{x \cdot 0}{x^2 + 0} = 0$

Along y = x: $\frac{x \cdot x}{x^2 + x^2} = \frac{x^2}{2x^2} = \frac{1}{2}$

Different limits → limit doesn't exist!

### 11. Continuity

> **Definition:** f(x, y) is **continuous at (a, b)** if:
> 1. f(a, b) is defined
> 2. $\lim_{(x,y) \to (a,b)} f(x,y)$ exists
> 3. $\lim_{(x,y) \to (a,b)} f(x,y) = f(a,b)$

**Continuous functions:** Polynomials, rational functions (where defined), trigonometric, exponential, logarithmic compositions.

---

## ✏️ Worked Examples

### Example 1: Domain
Find the domain of f(x, y) = ln(x + y - 1).

Need x + y - 1 > 0, so x + y > 1.

Domain: {(x, y) : x + y > 1} — half-plane above the line x + y = 1.

---

### Example 2: Level Curves
Sketch level curves for f(x, y) = x² - y² at k = -4, -1, 0, 1, 4.

- k = 0: x² - y² = 0 → y = ±x (two lines)
- k = 1: x² - y² = 1 (hyperbola, opens left/right)
- k = -1: x² - y² = -1 (hyperbola, opens up/down)
- k = 4: x² - y² = 4 (hyperbola)
- k = -4: x² - y² = -4 (hyperbola)

---

### Example 3: Evaluating Limits
Find $\lim_{(x,y) \to (1,2)} (x^2y - 3xy + 5)$.

Since polynomials are continuous, substitute directly:
$$= (1)^2(2) - 3(1)(2) + 5 = 2 - 6 + 5 = 1$$

---

### Example 4: Showing Limit Doesn't Exist
Show $\lim_{(x,y) \to (0,0)} \frac{x^2 - y^2}{x^2 + y^2}$ doesn't exist.

Along x-axis (y = 0): $\frac{x^2 - 0}{x^2 + 0} = 1$

Along y-axis (x = 0): $\frac{0 - y^2}{0 + y^2} = -1$

Limits differ → limit doesn't exist.

---

### Example 5: Level Surfaces
Describe the level surfaces of f(x, y, z) = x² + y² - z.

Setting f = k: x² + y² = z + k

These are paraboloids shifted vertically by -k.

---

## 📝 Practice Problems

### Level 1: Domains
1. Find the domain of f(x, y) = √(x - y).
2. Find the domain of f(x, y) = 1/(x² - y).
3. Find the domain of f(x, y) = ln(xy).

### Level 2: Level Curves
4. Sketch level curves for f(x, y) = x + y at k = -2, 0, 2.
5. Sketch level curves for f(x, y) = x² + 4y² at k = 1, 4, 16.
6. Sketch level curves for f(x, y) = y/x at k = -1, 0, 1, 2.

### Level 3: Limits
7. Evaluate $\lim_{(x,y) \to (2,1)} (x^2 + y^2 - xy)$.
8. Evaluate $\lim_{(x,y) \to (0,0)} \frac{\sin(x^2 + y^2)}{x^2 + y^2}$.
9. Show $\lim_{(x,y) \to (0,0)} \frac{xy^2}{x^2 + y^4}$ doesn't exist.

### Level 4: Continuity
10. Where is f(x, y) = (x² - y²)/(x - y) discontinuous?
11. Can f(x, y) = xy/(x² + y²) be made continuous at (0, 0)?
12. Discuss continuity of f(x, y) = arctan(y/x).

### Level 5: Functions of Three Variables
13. Describe the level surfaces of f(x, y, z) = x² + y² + z².
14. Find the domain of f(x, y, z) = ln(z - x² - y²).
15. Describe level surfaces of f(x, y, z) = x² + y² - z².

---

## 📊 Answers

1. {(x, y) : x ≥ y}
2. {(x, y) : y ≠ x²}
3. {(x, y) : xy > 0} — first and third quadrants
4. Parallel lines with slope -1
5. Ellipses centered at origin
6. Lines through origin with slope k
7. 3
8. 1 (use polar coordinates)
9. Along y = 0: limit = 0; along x = y²: limit = 1/2
10. Along line y = x (but can be extended continuously)
11. No (limit doesn't exist)
12. Discontinuous along negative x-axis
13. Spheres centered at origin
14. {(x, y, z) : z > x² + y²} — above paraboloid
15. Hyperboloids (one sheet for k > 0, two sheets for k < 0, cone for k = 0)

---

## 🔬 Quantum Mechanics Connection

### Wave Functions as Multivariable Functions

The wave function ψ(x, y, z, t) is a function of **four variables**!

For a time-independent state:
$$\psi(x, y, z) = \text{amplitude at each point in space}$$

**Probability density:** |ψ(x, y, z)|²

**Level surfaces of |ψ|²** show regions of equal probability density—like electron "clouds" around atoms!

### Example: Hydrogen Atom

The 1s orbital: ψ₁ₛ(r) ∝ e^(-r/a₀)

Level surfaces of |ψ|² are spheres (constant probability density).

---

## ✅ Daily Checklist

- [ ] Read Stewart 14.1-14.2
- [ ] Understand multivariable function notation
- [ ] Find domains of various functions
- [ ] Sketch level curves
- [ ] Evaluate limits (path approach)
- [ ] Understand continuity in 2D
- [ ] Complete practice problems

---

## 📓 Reflection Questions

1. How is a level curve related to the graph of a function?
2. Why do we need to check multiple paths for limits?
3. What does a contour map tell you about terrain?
4. How do level curves help us visualize 3D surfaces?

---

## 🔜 Preview: Tomorrow

**Day 37: Partial Derivatives**
- Definition and computation
- Geometric interpretation
- Higher-order partial derivatives
- Clairaut's theorem

---

*"Functions of several variables are the gateway to understanding the physical world."*
