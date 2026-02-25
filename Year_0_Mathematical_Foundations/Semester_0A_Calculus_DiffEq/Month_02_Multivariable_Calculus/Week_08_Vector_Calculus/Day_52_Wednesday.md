# Day 52: Green's Theorem

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Green's Theorem Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State and understand Green's Theorem
2. Apply Green's Theorem to evaluate line integrals
3. Use Green's Theorem to find areas
4. Understand circulation and flux forms
5. Apply to physical problems

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 16.4**: Green's Theorem (pp. 1112-1117)

---

## 🎬 Video Resources

### 3Blue1Brown
**Green's Theorem** - Visual intuition

### MIT OpenCourseWare 18.02SC
**Lecture: Green's Theorem**

### Professor Leonard
**Calculus 3: Green's Theorem**

---

## 📖 Core Content: Green's Theorem

### 1. Statement of Green's Theorem

> **Green's Theorem:** Let C be a positively oriented (counterclockwise), piecewise-smooth, simple closed curve in the plane, and let D be the region bounded by C. If P and Q have continuous partial derivatives on an open region containing D, then:
> $$\oint_C P \, dx + Q \, dy = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$$

### 2. What Green's Theorem Says

- **Left side:** Line integral around the boundary C
- **Right side:** Double integral over the interior D
- **Connection:** Circulation around boundary = integral of "curl" over interior

### 3. Notation

The symbol ∮ indicates integration around a **closed curve**.

Positive orientation: counterclockwise (region on left as you walk).

### 4. Vector Form

For **F** = ⟨P, Q⟩:
$$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_D (\text{curl } \mathbf{F}) \, dA$$

where curl **F** = ∂Q/∂x - ∂P/∂y (the scalar 2D curl).

---

## ✏️ Green's Theorem Examples

### Example 1: Basic Application
Evaluate ∮_C (xy dx + x² dy) where C is the rectangle with vertices (0,0), (2,0), (2,1), (0,1).

**Direct approach:** Would need 4 line integrals!

**Using Green's Theorem:**
P = xy, Q = x²
$$\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = 2x - x = x$$

$$\oint_C xy \, dx + x^2 \, dy = \iint_D x \, dA = \int_0^2 \int_0^1 x \, dy \, dx = \int_0^2 x \, dx = 2$$

Much easier!

---

### Example 2: Circle
Evaluate ∮_C (y² dx + 3xy dy) where C is the circle x² + y² = 9.

P = y², Q = 3xy
$$\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = 3y - 2y = y$$

$$\oint_C = \iint_D y \, dA$$

Using polar coordinates (D is a disk of radius 3):
$$= \int_0^{2\pi} \int_0^3 r\sin\theta \cdot r \, dr \, d\theta = \int_0^{2\pi} \sin\theta \, d\theta \cdot \int_0^3 r^2 \, dr = 0$$

(The θ integral is zero!)

---

### Example 3: Verifying a Line Integral
Verify Green's Theorem for **F** = ⟨-y, x⟩ and C the unit circle.

**Line integral:**
**r**(t) = ⟨cos t, sin t⟩, **r**'(t) = ⟨-sin t, cos t⟩

$$\oint_C \mathbf{F} \cdot d\mathbf{r} = \int_0^{2\pi} \langle -\sin t, \cos t \rangle \cdot \langle -\sin t, \cos t \rangle \, dt$$
$$= \int_0^{2\pi} (\sin^2 t + \cos^2 t) \, dt = 2\pi$$

**Double integral:**
$$\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = 1 - (-1) = 2$$

$$\iint_D 2 \, dA = 2 \cdot \text{Area}(D) = 2\pi$$ ✓

---

## 📖 Computing Area Using Green's Theorem

### 5. Area Formulas

From Green's Theorem, if ∂Q/∂x - ∂P/∂y = 1, then:
$$\text{Area}(D) = \oint_C P \, dx + Q \, dy$$

**Three useful choices:**
1. P = 0, Q = x: Area = ∮_C x dy
2. P = -y, Q = 0: Area = -∮_C y dx = ∮_C (-y) dx
3. P = -y/2, Q = x/2: Area = ½∮_C (x dy - y dx)

### Example 4: Area of Ellipse
Find the area of the ellipse x²/a² + y²/b² = 1.

**Parametrize:** x = a cos t, y = b sin t, 0 ≤ t ≤ 2π

Using Area = ½∮(x dy - y dx):
$$\text{Area} = \frac{1}{2}\oint_C (x \, dy - y \, dx)$$

dx = -a sin t dt, dy = b cos t dt

$$= \frac{1}{2}\int_0^{2\pi} [a\cos t \cdot b\cos t - b\sin t \cdot (-a\sin t)] \, dt$$
$$= \frac{ab}{2}\int_0^{2\pi} (\cos^2 t + \sin^2 t) \, dt = \frac{ab}{2} \cdot 2\pi = \pi ab$$

---

## 📖 Flux Form of Green's Theorem

### 6. Flux Form

> **Green's Theorem (Flux Form):** 
> $$\oint_C \mathbf{F} \cdot \mathbf{n} \, ds = \iint_D \text{div } \mathbf{F} \, dA$$

where **n** is the outward unit normal to C.

**Equivalently:**
$$\oint_C (P \, dy - Q \, dx) = \iint_D \left(\frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y}\right) dA$$

### 7. Interpretation

- **Circulation form:** Measures rotation (curl integrated)
- **Flux form:** Measures flow across boundary (divergence integrated)

---

## 📝 Practice Problems

### Level 1: Basic Green's Theorem
1. ∮_C (x² dx + xy dy) where C is the triangle (0,0), (1,0), (1,1)
2. ∮_C (y dx - x dy) where C is the circle x² + y² = 4
3. ∮_C (3y dx + 2x dy) where C is the boundary of [0,1] × [0,2]

### Level 2: Area Calculations
4. Find the area enclosed by x = cos³ t, y = sin³ t (astroid)
5. Find the area of the region between y = x² and y = x
6. Find the area enclosed by r = 1 + cos θ using Green's Theorem

### Level 3: Verification
7. Verify Green's Theorem for **F** = ⟨x², y²⟩ and D = [0,1]²
8. For **F** = ⟨y², x²⟩, compute ∮_C **F** · d**r** around the unit square both directly and via Green's

### Level 4: Applications
9. Find ∮_C (eˣ + y²) dx + (eʸ + x²) dy where C is any simple closed curve
10. If div **F** = 3 everywhere, and D has area 5, find ∮_C **F** · **n** ds

### Level 5: Theory
11. Use Green's Theorem to prove that ∮_C **F** · d**r** = 0 for any conservative field **F** and closed curve C
12. Derive the formula Area = ½∮(x dy - y dx) from Green's Theorem

---

## 📊 Answers

1. 1/6
2. -8π
3. -2
4. 3π/8
5. 1/6
6. 3π/2
7. Both give 0
8. Both give 0
9. 0 (∂Q/∂x - ∂P/∂y = 2x - 2y, but need to know C)
10. 15
11. curl(∇f) = 0, so double integral is 0
12. Set P = -y/2, Q = x/2

---

## 🔬 Quantum Mechanics Connection

### Stokes' Theorem in 2D

Green's Theorem is the 2D version of Stokes' Theorem. In physics:

$$\oint_C \mathbf{E} \cdot d\mathbf{l} = -\frac{d\Phi_B}{dt}$$

This is Faraday's law—a changing magnetic flux induces an electric circulation!

### Magnetic Flux

$$\Phi_B = \iint_S \mathbf{B} \cdot d\mathbf{A}$$

Green's Theorem connects this surface integral to a line integral.

---

## ✅ Daily Checklist

- [ ] Read Stewart 16.4
- [ ] State Green's Theorem both forms
- [ ] Apply to evaluate line integrals
- [ ] Compute areas using Green's Theorem
- [ ] Understand circulation vs flux interpretations
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 53: Surface Integrals and the Big Theorems**
- Surface integrals
- Stokes' Theorem (Green's in 3D)
- Divergence Theorem

---

*"Green's Theorem reveals the deep connection between boundaries and interiors—what happens on the edge determines what happens inside."*
