# Day 38: The Gradient and Directional Derivatives

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Gradient Vector |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Directional Derivatives |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define and compute the gradient vector
2. Understand the geometric meaning of the gradient
3. Compute directional derivatives
4. Find the direction of maximum rate of change
5. Apply gradients to optimization problems

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 14.6**: Directional Derivatives and the Gradient Vector (pp. 948-960)

---

## 📖 Core Content: The Gradient Vector

### 1. Definition

> **Definition:** The **gradient** of f(x, y) is the vector:
> $$\nabla f(x,y) = \left\langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right\rangle = f_x\mathbf{i} + f_y\mathbf{j}$$

For f(x, y, z):
$$\nabla f = \left\langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right\rangle$$

### 2. Notation

The symbol ∇ is called "nabla" or "del."

$$\nabla f = \text{grad } f = \langle f_x, f_y \rangle$$

### 3. The Gradient as an Operator

We can think of ∇ as a vector differential operator:
$$\nabla = \left\langle \frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z} \right\rangle$$

Applying it to a scalar function gives a vector!

---

## ✏️ Gradient Examples

### Example 1: Polynomial
Find ∇f for f(x, y) = x²y + y³.

$$f_x = 2xy, \quad f_y = x^2 + 3y^2$$

$$\nabla f = \langle 2xy, x^2 + 3y^2 \rangle$$

At (1, 2): ∇f(1, 2) = ⟨4, 13⟩

---

### Example 2: Exponential
Find ∇f for f(x, y) = e^(x²+y²).

$$f_x = 2xe^{x^2+y^2}, \quad f_y = 2ye^{x^2+y^2}$$

$$\nabla f = 2e^{x^2+y^2}\langle x, y \rangle$$

---

### Example 3: Three Variables
Find ∇f for f(x, y, z) = xyz.

$$\nabla f = \langle yz, xz, xy \rangle$$

---

## 📖 Directional Derivatives

### 4. Motivation

Partial derivatives give rates of change in the x and y directions.

What about the rate of change in **any** direction?

### 5. Definition

> **Definition:** The **directional derivative** of f at (a, b) in the direction of unit vector **u** = ⟨u₁, u₂⟩ is:
> $$D_\mathbf{u}f(a,b) = \lim_{h \to 0} \frac{f(a + hu_1, b + hu_2) - f(a, b)}{h}$$

### 6. Formula Using the Gradient

> **Theorem:** If f is differentiable, then:
> $$D_\mathbf{u}f = \nabla f \cdot \mathbf{u}$$

This is remarkable: the directional derivative is just the dot product of the gradient with the direction!

### 7. Important Notes

- **u must be a unit vector** (|**u**| = 1)
- If **u** = **i** = ⟨1, 0⟩, then D_**u**f = fₓ
- If **u** = **j** = ⟨0, 1⟩, then D_**u**f = f_y

---

## ✏️ Directional Derivative Examples

### Example 4: Computing D_**u**f
Find the directional derivative of f(x, y) = x²y³ at (2, 1) in the direction of **v** = ⟨3, 4⟩.

**Step 1:** Find unit vector.
$$|\mathbf{v}| = 5, \quad \mathbf{u} = \langle 3/5, 4/5 \rangle$$

**Step 2:** Find gradient.
$$\nabla f = \langle 2xy^3, 3x^2y^2 \rangle$$
$$\nabla f(2, 1) = \langle 4, 12 \rangle$$

**Step 3:** Compute dot product.
$$D_\mathbf{u}f = \langle 4, 12 \rangle \cdot \langle 3/5, 4/5 \rangle = \frac{12}{5} + \frac{48}{5} = 12$$

---

### Example 5: Direction from Angle
Find D_**u**f at (1, 2) for f(x, y) = e^x sin(y), where **u** makes angle θ = π/3 with positive x-axis.

**u** = ⟨cos(π/3), sin(π/3)⟩ = ⟨1/2, √3/2⟩

$$\nabla f = \langle e^x\sin y, e^x\cos y \rangle$$
$$\nabla f(1, 2) = \langle e\sin 2, e\cos 2 \rangle$$

$$D_\mathbf{u}f = e\sin 2 \cdot \frac{1}{2} + e\cos 2 \cdot \frac{\sqrt{3}}{2}$$
$$= \frac{e}{2}(\sin 2 + \sqrt{3}\cos 2)$$

---

## 📐 Properties of the Gradient

### 8. Maximum Rate of Change

> **Theorem:** The maximum value of D_**u**f is |∇f|, and it occurs when **u** is in the direction of ∇f.

**Proof:** 
$$D_\mathbf{u}f = \nabla f \cdot \mathbf{u} = |\nabla f||\mathbf{u}|\cos\theta = |\nabla f|\cos\theta$$

Maximum when cos θ = 1, i.e., when **u** points in same direction as ∇f.

### 9. Key Properties

1. **∇f points in the direction of steepest ascent**
2. **|∇f| gives the maximum rate of change**
3. **−∇f points in the direction of steepest descent**
4. **∇f is perpendicular to level curves**

### 10. Gradient Perpendicular to Level Curves

> **Theorem:** At any point, the gradient ∇f is perpendicular to the level curve f(x, y) = k passing through that point.

This is why contour lines are perpendicular to the direction of steepest ascent!

---

## ✏️ More Examples

### Example 6: Maximum Rate of Change
For f(x, y) = x² + 4y² at (1, 1):
(a) Find the direction of maximum increase.
(b) Find the maximum rate of change.
(c) Find a direction where the rate of change is 0.

$$\nabla f = \langle 2x, 8y \rangle, \quad \nabla f(1, 1) = \langle 2, 8 \rangle$$

(a) Direction of max increase: **u** = ⟨2, 8⟩/|⟨2, 8⟩| = ⟨2, 8⟩/√68 = ⟨1/√17, 4/√17⟩

(b) Maximum rate: |∇f(1, 1)| = √(4 + 64) = √68 = 2√17

(c) Zero rate: perpendicular to gradient, so **u** = ⟨4, -1⟩/√17 (or ⟨-4, 1⟩/√17)

---

### Example 7: Temperature Gradient
The temperature at (x, y) is T(x, y) = 100 - x² - 2y². An ant at (3, 2) wants to cool off as quickly as possible. In what direction should it move?

$$\nabla T = \langle -2x, -4y \rangle, \quad \nabla T(3, 2) = \langle -6, -8 \rangle$$

Direction of steepest descent (fastest cooling): −∇T/|∇T| = ⟨6, 8⟩/10 = ⟨0.6, 0.8⟩

---

## 📝 Practice Problems

### Level 1: Computing Gradients
1. Find ∇f for f(x, y) = 3x² - 2xy + y².
2. Find ∇f for f(x, y) = sin(x)cos(y).
3. Find ∇f for f(x, y, z) = x²y + yz².
4. Evaluate ∇f at (1, 2) for f(x, y) = xe^y.

### Level 2: Directional Derivatives
5. Find D_**u**f at (2, 1) for f(x, y) = x² + y², **u** = ⟨1/√2, 1/√2⟩.
6. Find D_**u**f at (1, 0) for f(x, y) = e^x cos(y) in direction of **v** = ⟨3, 4⟩.
7. Find D_**u**f at (0, 1, 2) for f(x, y, z) = xy + yz + xz, **u** = ⟨1, 2, 2⟩/3.

### Level 3: Maximum Rate
8. Find the direction of maximum increase of f(x, y) = xe^y at (2, 0).
9. Find the maximum rate of change of f(x, y) = √(x² + y²) at (3, 4).
10. At (1, 1), in what direction is the rate of change of f(x, y) = x² - y² equal to zero?

### Level 4: Applications
11. The altitude of a hill is h(x, y) = 1000 - x² - 2y². A hiker at (10, 5) wants to climb most steeply. What direction?
12. Temperature T(x, y) = 20 + x² - y². Find direction at (1, 1) where temperature doesn't change.
13. f(x, y) = x² + xy + y². At (1, 1), find the rate of change toward (4, 5).

### Level 5: Theory
14. Prove that ∇(fg) = f∇g + g∇f.
15. Show that if f(x, y) = g(x)h(y), then ∇f = ⟨g'(x)h(y), g(x)h'(y)⟩.

---

## 📊 Answers

1. ∇f = ⟨6x - 2y, -2x + 2y⟩
2. ∇f = ⟨cos x cos y, -sin x sin y⟩
3. ∇f = ⟨2xy, x² + z², 2yz⟩
4. ⟨e², 2e²⟩
5. 3√2
6. 3e/5
7. 7/3
8. ⟨1, 2⟩/√5
9. 1
10. ⟨1, 1⟩/√2 or ⟨-1, -1⟩/√2
11. ⟨20, 20⟩/√800 toward origin
12. ⟨1, 1⟩/√2
13. 7/√2
14. Use product rule on each component
15. Compute fₓ and f_y directly

---

## 🔬 Quantum Mechanics Connection

### The Momentum Operator

In quantum mechanics, the momentum operator is:
$$\hat{\mathbf{p}} = -i\hbar\nabla$$

The gradient connects to momentum!

### Probability Current

The probability current is:
$$\mathbf{j} = \frac{\hbar}{2mi}(\psi^*\nabla\psi - \psi\nabla\psi^*)$$

### Potential Energy

If a particle moves in a potential V(x, y, z), the force is:
$$\mathbf{F} = -\nabla V$$

The gradient of potential gives the force!

---

## ✅ Daily Checklist

- [ ] Read Stewart 14.6
- [ ] Master gradient computation
- [ ] Understand gradient as direction of steepest ascent
- [ ] Compute directional derivatives
- [ ] Find maximum/minimum rates of change
- [ ] Understand gradient perpendicular to level curves
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 39: Tangent Planes and Linear Approximation**
- Tangent planes to surfaces
- Linear approximation in 2D
- Total differential
- Error estimation

---

*"The gradient points the way uphill—nature's compass for optimization."*
