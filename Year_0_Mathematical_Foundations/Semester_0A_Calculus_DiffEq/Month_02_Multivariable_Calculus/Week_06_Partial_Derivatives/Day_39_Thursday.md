# Day 39: Tangent Planes and Linear Approximation

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Tangent Planes |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Linear Approximation |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Find equations of tangent planes to surfaces
2. Use tangent planes for linear approximation
3. Understand differentiability in multiple variables
4. Compute the total differential
5. Apply linear approximation to error estimation

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 14.4**: Tangent Planes and Linear Approximations (pp. 930-940)

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.02SC
**Lecture: Tangent Planes**

### Professor Leonard
**Calculus 3: Tangent Planes and Linear Approximation**

---

## 📖 Core Content: Tangent Planes

### 1. Motivation

In single-variable calculus, we approximated a curve with its tangent line:
$$y \approx f(a) + f'(a)(x - a)$$

In multivariable calculus, we approximate a surface with its **tangent plane**.

### 2. Surfaces as Graphs

A surface z = f(x, y) can be viewed as a level surface:
$$F(x, y, z) = f(x, y) - z = 0$$

### 3. Equation of the Tangent Plane

> **Theorem:** The tangent plane to z = f(x, y) at the point (a, b, f(a, b)) is:
> $$z - f(a,b) = f_x(a,b)(x - a) + f_y(a,b)(y - b)$$

Or equivalently:
$$z = f(a,b) + f_x(a,b)(x - a) + f_y(a,b)(y - b)$$

### 4. Why This Formula?

The tangent plane contains:
- The tangent line in the x-direction: slope = fₓ(a, b)
- The tangent line in the y-direction: slope = f_y(a, b)

The normal vector to the tangent plane is:
$$\mathbf{n} = \langle f_x(a,b), f_y(a,b), -1 \rangle$$

---

## ✏️ Tangent Plane Examples

### Example 1: Paraboloid
Find the equation of the tangent plane to z = x² + y² at (1, 1, 2).

**Step 1:** Compute partial derivatives.
$$f_x = 2x, \quad f_y = 2y$$

**Step 2:** Evaluate at (1, 1).
$$f_x(1, 1) = 2, \quad f_y(1, 1) = 2$$

**Step 3:** Write equation.
$$z - 2 = 2(x - 1) + 2(y - 1)$$
$$z = 2x + 2y - 2$$

---

### Example 2: Exponential Surface
Find the tangent plane to z = e^(xy) at (1, 0, 1).

$$f_x = ye^{xy}, \quad f_y = xe^{xy}$$
$$f_x(1, 0) = 0, \quad f_y(1, 0) = 1$$

$$z - 1 = 0(x - 1) + 1(y - 0)$$
$$z = 1 + y$$

---

### Example 3: Trigonometric Surface
Find the tangent plane to z = sin(x)cos(y) at (π/2, 0, 1).

$$f_x = \cos(x)\cos(y), \quad f_y = -\sin(x)\sin(y)$$
$$f_x(\pi/2, 0) = 0, \quad f_y(\pi/2, 0) = 0$$

$$z = 1$$

The tangent plane is horizontal at this point!

---

## 📖 Linear Approximation

### 5. The Linearization

> **Definition:** The **linearization** of f(x, y) at (a, b) is:
> $$L(x, y) = f(a, b) + f_x(a, b)(x - a) + f_y(a, b)(y - b)$$

This is the equation of the tangent plane!

### 6. Linear Approximation Formula

For (x, y) near (a, b):
$$f(x, y) \approx L(x, y) = f(a, b) + f_x(a, b)(x - a) + f_y(a, b)(y - b)$$

This extends the single-variable approximation f(x) ≈ f(a) + f'(a)(x - a).

### 7. When Is It Valid?

The approximation is good when:
- f is differentiable at (a, b)
- (x, y) is close to (a, b)

---

## ✏️ Linear Approximation Examples

### Example 4: Approximating √(x² + y²)
Use linear approximation to estimate √(3.02² + 3.99²).

Let f(x, y) = √(x² + y²). We want f(3.02, 3.99) near (3, 4).

$$f(3, 4) = 5$$

$$f_x = \frac{x}{\sqrt{x^2 + y^2}}, \quad f_y = \frac{y}{\sqrt{x^2 + y^2}}$$

$$f_x(3, 4) = 3/5 = 0.6, \quad f_y(3, 4) = 4/5 = 0.8$$

$$f(3.02, 3.99) \approx 5 + 0.6(0.02) + 0.8(-0.01)$$
$$= 5 + 0.012 - 0.008 = 5.004$$

**Actual value:** √(3.02² + 3.99²) = √25.0405 ≈ 5.00405 ✓

---

### Example 5: Compound Interest Approximation
The future value of an investment is A = P(1 + r)^t. Approximate the change in A if P = 1000, r = 0.05, t = 10, and both P and r increase slightly.

$$A_P = (1 + r)^t, \quad A_r = Pt(1 + r)^{t-1}$$

At (P, r) = (1000, 0.05):
$$A = 1000(1.05)^{10} \approx 1628.89$$
$$A_P = (1.05)^{10} \approx 1.629$$
$$A_r = 1000 \cdot 10 \cdot (1.05)^9 \approx 15513.28$$

If P increases by $10 and r increases by 0.001:
$$\Delta A \approx 1.629(10) + 15513.28(0.001) \approx 16.29 + 15.51 = 31.80$$

---

## 📖 The Total Differential

### 8. Definition

> **Definition:** The **total differential** of z = f(x, y) is:
> $$dz = f_x(x, y) \, dx + f_y(x, y) \, dy = \frac{\partial z}{\partial x}dx + \frac{\partial z}{\partial y}dy$$

### 9. Interpretation

- dx, dy are small changes in x and y
- dz is the corresponding change in z **along the tangent plane**
- Δz = f(x + Δx, y + Δy) - f(x, y) is the **actual change**
- For small changes: Δz ≈ dz

### 10. For Three Variables

$$dw = \frac{\partial w}{\partial x}dx + \frac{\partial w}{\partial y}dy + \frac{\partial w}{\partial z}dz$$

---

## ✏️ Differential Examples

### Example 6: Computing the Differential
Find the differential of z = x²y + xy³.

$$\frac{\partial z}{\partial x} = 2xy + y^3, \quad \frac{\partial z}{\partial y} = x^2 + 3xy^2$$

$$dz = (2xy + y^3)dx + (x^2 + 3xy^2)dy$$

---

### Example 7: Estimating Change
Use differentials to estimate the change in z = x³y² if (x, y) changes from (2, 3) to (2.01, 2.98).

At (2, 3): z = 8 · 9 = 72

$$z_x = 3x^2y^2 = 3(4)(9) = 108$$
$$z_y = 2x^3y = 2(8)(3) = 48$$

$$dz = 108(0.01) + 48(-0.02) = 1.08 - 0.96 = 0.12$$

Estimate: z ≈ 72.12

**Actual:** (2.01)³(2.98)² ≈ 72.1188 ✓

---

## 📐 Error Estimation

### 11. Propagation of Error

If z = f(x, y) and we measure x and y with errors Δx and Δy, then:
$$\Delta z \approx |f_x|\Delta x + |f_y|\Delta y$$

This is called **error propagation**.

### 12. Relative Error

The **relative error** in z is approximately:
$$\frac{\Delta z}{z} \approx \frac{|f_x|}{z}\Delta x + \frac{|f_y|}{z}\Delta y$$

---

### Example 8: Error in Cylinder Volume
A cylinder has radius r = 5 cm (±0.1 cm) and height h = 10 cm (±0.2 cm). Estimate the error in computing the volume.

$$V = \pi r^2 h$$
$$V_r = 2\pi rh = 2\pi(5)(10) = 100\pi$$
$$V_h = \pi r^2 = 25\pi$$

$$\Delta V \approx |V_r|\Delta r + |V_h|\Delta h = 100\pi(0.1) + 25\pi(0.2)$$
$$= 10\pi + 5\pi = 15\pi \approx 47.1 \text{ cm}^3$$

**Relative error:**
$$\frac{\Delta V}{V} \approx \frac{15\pi}{250\pi} = \frac{15}{250} = 6\%$$

---

## 📋 Differentiability

### 13. Definition

> **Definition:** f(x, y) is **differentiable** at (a, b) if:
> $$\Delta z = f_x(a,b)\Delta x + f_y(a,b)\Delta y + \epsilon_1\Delta x + \epsilon_2\Delta y$$
> where ε₁, ε₂ → 0 as (Δx, Δy) → (0, 0).

### 14. Sufficient Condition

> **Theorem:** If fₓ and f_y are continuous at (a, b), then f is differentiable at (a, b).

**Note:** Having partial derivatives is NOT sufficient for differentiability!

---

## 📝 Practice Problems

### Level 1: Tangent Planes
1. Find the tangent plane to z = x² + y² at (1, 2, 5).
2. Find the tangent plane to z = xy at (2, 3, 6).
3. Find the tangent plane to z = ln(x + y) at (1, 1, ln 2).

### Level 2: Linear Approximation
4. Use linear approximation to estimate (1.98)²(3.01)³.
5. Estimate √(8.9) + ∛(27.1) using linearization.
6. Approximate e^(0.1) sin(0.1) using L(x, y) at (0, 0).

### Level 3: Differentials
7. Find dz for z = x sin(y) + y cos(x).
8. Find dw for w = √(x² + y² + z²).
9. Use differentials to estimate the change in z = arctan(y/x) from (1, 1) to (1.1, 0.9).

### Level 4: Error Estimation
10. The period of a pendulum is T = 2π√(L/g). If L = 1.0 ± 0.01 m and g = 9.8 ± 0.1 m/s², estimate the error in T.
11. For a cone V = (1/3)πr²h, estimate dV if r = 3 ± 0.05 and h = 4 ± 0.1.
12. The resistance R = ρL/A. If ρ, L, A each have 2% error, estimate the maximum percentage error in R.

### Level 5: Theory
13. Show that if z = f(x, y) = ax + by + c (a plane), then dz = Δz exactly.
14. Find all points where the tangent plane to z = x² - y² is horizontal.
15. Find the equation of the tangent plane to x² + y² + z² = 14 at (1, 2, 3).

---

## 📊 Answers

1. z = 2x + 4y - 5
2. z = 3x + 2y - 6
3. z = (1/2)(x - 1) + (1/2)(y - 1) + ln 2
4. ≈ 107.46
5. ≈ 5.983
6. ≈ 0.1
7. dz = sin(y) dx - y sin(x) dx + cos(y) x dy + cos(x) dy
8. dw = (x dx + y dy + z dz)/√(x² + y² + z²)
9. ≈ -0.1
10. ΔT ≈ 0.016 s
11. dV ≈ 4.19
12. 6%
13. Both equal ax + by
14. Origin (0, 0)
15. x + 2y + 3z = 14

---

## 🔬 Quantum Mechanics Connection

### First-Order Perturbation Theory

When a quantum system has a small perturbation H' added to the Hamiltonian:
$$E_n \approx E_n^{(0)} + \langle n|H'|n \rangle$$

This is linear approximation in quantum mechanics!

### Uncertainty Propagation

When measuring observables, uncertainties propagate:
$$\Delta f = \sqrt{\left(\frac{\partial f}{\partial x}\right)^2(\Delta x)^2 + \left(\frac{\partial f}{\partial y}\right)^2(\Delta y)^2}$$

---

## ✅ Daily Checklist

- [ ] Read Stewart 14.4
- [ ] Derive tangent plane equations
- [ ] Master linear approximation
- [ ] Compute total differentials
- [ ] Apply to error estimation
- [ ] Understand differentiability condition
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 40: Week 6 Problem Set**
- Comprehensive review of partial derivatives
- Gradient and directional derivatives
- Tangent planes and linear approximation

---

*"Linearization is the art of replacing the complex with the simple—valid when changes are small."*
