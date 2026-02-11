# Day 51: Line Integrals

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Scalar Line Integrals |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Vector Line Integrals |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Parametrize curves in 2D and 3D
2. Compute scalar line integrals ∫_C f ds
3. Compute vector line integrals ∫_C **F** · d**r**
4. Calculate work done by a force field
5. Understand path independence for conservative fields

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 16.2**: Line Integrals (pp. 1088-1101)
- **Section 16.3**: The Fundamental Theorem for Line Integrals (pp. 1102-1111)

---

## 📖 Core Content: Scalar Line Integrals

### 1. Parametrizing Curves

A curve C in space can be described by a vector function:
$$\mathbf{r}(t) = \langle x(t), y(t), z(t) \rangle, \quad a \leq t \leq b$$

**Examples:**
- Line from (0,0) to (1,2): **r**(t) = ⟨t, 2t⟩, 0 ≤ t ≤ 1
- Circle of radius R: **r**(t) = ⟨R cos t, R sin t⟩, 0 ≤ t ≤ 2π
- Helix: **r**(t) = ⟨cos t, sin t, t⟩

### 2. Arc Length Element

The **arc length element** is:
$$ds = |\mathbf{r}'(t)| \, dt = \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2 + \left(\frac{dz}{dt}\right)^2} \, dt$$

### 3. Scalar Line Integral

> **Definition:** The **line integral of f along C** is:
> $$\int_C f \, ds = \int_a^b f(\mathbf{r}(t)) \, |\mathbf{r}'(t)| \, dt$$

**Interpretation:** If f represents linear mass density, then ∫_C f ds is the total mass of a wire shaped like C.

---

## ✏️ Scalar Line Integral Examples

### Example 1: Line Segment
Evaluate ∫_C xy ds where C is the line from (0,0) to (1,2).

**Parametrize:** **r**(t) = ⟨t, 2t⟩, 0 ≤ t ≤ 1

**Compute |r'(t)|:** **r**'(t) = ⟨1, 2⟩, |**r**'(t)| = √5

**Evaluate:**
$$\int_C xy \, ds = \int_0^1 (t)(2t) \sqrt{5} \, dt = 2\sqrt{5} \int_0^1 t^2 \, dt = 2\sqrt{5} \cdot \frac{1}{3} = \frac{2\sqrt{5}}{3}$$

---

### Example 2: Circle
Evaluate ∫_C (x² + y²) ds where C is the circle x² + y² = 4.

**Parametrize:** **r**(t) = ⟨2cos t, 2sin t⟩, 0 ≤ t ≤ 2π

**Compute |r'(t)|:** **r**'(t) = ⟨-2sin t, 2cos t⟩, |**r**'(t)| = 2

**Evaluate:**
$$\int_C (x^2 + y^2) \, ds = \int_0^{2\pi} 4 \cdot 2 \, dt = 8 \cdot 2\pi = 16\pi$$

---

## 📖 Vector Line Integrals

### 4. Definition

> **Definition:** The **line integral of F along C** is:
> $$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t) \, dt$$

**Alternative notation:**
$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_C P \, dx + Q \, dy + R \, dz$$

where **F** = ⟨P, Q, R⟩.

### 5. Physical Interpretation: Work

The work done by force **F** in moving a particle along C is:
$$W = \int_C \mathbf{F} \cdot d\mathbf{r}$$

### 6. Orientation Matters!

$$\int_{-C} \mathbf{F} \cdot d\mathbf{r} = -\int_C \mathbf{F} \cdot d\mathbf{r}$$

The direction of traversal matters for vector line integrals!

---

## ✏️ Vector Line Integral Examples

### Example 3: Work Done
Find the work done by **F** = ⟨y, x⟩ along the parabola y = x² from (0,0) to (1,1).

**Parametrize:** **r**(t) = ⟨t, t²⟩, 0 ≤ t ≤ 1
**r**'(t) = ⟨1, 2t⟩

**F along curve:** **F**(**r**(t)) = ⟨t², t⟩

**Evaluate:**
$$W = \int_0^1 \langle t^2, t \rangle \cdot \langle 1, 2t \rangle \, dt = \int_0^1 (t^2 + 2t^2) \, dt = \int_0^1 3t^2 \, dt = 1$$

---

### Example 4: Component Form
Evaluate ∫_C y dx + x dy where C is the quarter circle from (1,0) to (0,1).

**Parametrize:** **r**(t) = ⟨cos t, sin t⟩, 0 ≤ t ≤ π/2
dx = -sin t dt, dy = cos t dt

$$\int_C y \, dx + x \, dy = \int_0^{\pi/2} [\sin t \cdot (-\sin t) + \cos t \cdot \cos t] \, dt$$

$$= \int_0^{\pi/2} (\cos^2 t - \sin^2 t) \, dt = \int_0^{\pi/2} \cos(2t) \, dt = \frac{1}{2}\sin(2t)\Big|_0^{\pi/2} = 0$$

---

## 📖 Fundamental Theorem for Line Integrals

### 7. The Theorem

> **Fundamental Theorem for Line Integrals:** If **F** = ∇f and C is a smooth curve from point A to point B, then:
> $$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_C \nabla f \cdot d\mathbf{r} = f(B) - f(A)$$

**Key insight:** For conservative fields, the line integral depends only on endpoints!

### 8. Path Independence

A vector field **F** is **path independent** if for any two points A and B:
$$\int_{C_1} \mathbf{F} \cdot d\mathbf{r} = \int_{C_2} \mathbf{F} \cdot d\mathbf{r}$$
for all paths C₁ and C₂ from A to B.

### 9. Equivalent Conditions

For a vector field **F** on a simply connected domain, the following are equivalent:
1. **F** is conservative (**F** = ∇f)
2. ∫_C **F** · d**r** is path independent
3. ∮_C **F** · d**r** = 0 for every closed curve C
4. curl **F** = **0**

---

## ✏️ Fundamental Theorem Examples

### Example 5: Using the Fundamental Theorem
Evaluate ∫_C ∇(x²y) · d**r** where C is any path from (1, 2) to (3, 4).

f(x, y) = x²y

$$\int_C \nabla f \cdot d\mathbf{r} = f(3, 4) - f(1, 2) = (9)(4) - (1)(2) = 36 - 2 = 34$$

No need to parametrize C!

---

### Example 6: Checking Path Independence
Is ∫_C (2xy dx + x² dy) path independent?

**F** = ⟨2xy, x²⟩

Check: ∂Q/∂x = 2x, ∂P/∂y = 2x → Equal! ✓

**F** is conservative with potential f(x, y) = x²y.

For any path from (0,0) to (2,3):
$$\int_C 2xy \, dx + x^2 \, dy = f(2,3) - f(0,0) = 12 - 0 = 12$$

---

## 📝 Practice Problems

### Level 1: Scalar Line Integrals
1. ∫_C x ds where C is the line from (0,0) to (3,4)
2. ∫_C (x + y) ds where C is the upper half of the unit circle
3. ∫_C xyz ds where C is the helix **r**(t) = ⟨cos t, sin t, t⟩, 0 ≤ t ≤ 2π

### Level 2: Vector Line Integrals
4. ∫_C **F** · d**r** where **F** = ⟨y, x⟩ and C is the line from (0,0) to (1,1)
5. ∫_C (x² dx + y² dy) where C is the quarter circle from (1,0) to (0,1)
6. ∫_C **F** · d**r** where **F** = ⟨y, -x⟩ around the unit circle (counterclockwise)

### Level 3: Fundamental Theorem
7. Use the FT to evaluate ∫_C ∇(eˣ sin y) · d**r** from (0,0) to (1, π/2)
8. Show **F** = ⟨2x + y, x + 2y⟩ is conservative and find the potential
9. Evaluate ∫_C (2x + y) dx + (x + 2y) dy from (0,0) to (1,1) by any method

### Level 4: Applications
10. Find the work done by **F** = ⟨-y, x⟩/(x² + y²) around the unit circle
11. A force **F** = ⟨x, y, z⟩ moves a particle from (1,0,0) to (0,1,1). Find the work.

### Level 5: Theory
12. Prove that if ∮_C **F** · d**r** = 0 for all closed curves, then **F** is conservative.
13. Why must the domain be simply connected for curl **F** = 0 to imply **F** is conservative?

---

## 📊 Answers

1. 15/2
2. π + 2
3. 0
4. 1
5. 1/3
6. -2π
7. e - 0 = e
8. f(x, y) = x² + xy + y²
9. 3
10. 2π
11. 1/2
12. Consider the potential f(P) = ∫ from fixed point to P
13. The winding number example (day 50, problem 12)

---

## 🔬 Quantum Mechanics Connection

### Berry Phase

When a quantum system evolves around a closed loop in parameter space:
$$\gamma = \oint_C \mathbf{A} \cdot d\mathbf{R}$$

This **Berry phase** is a line integral of the Berry connection!

### Aharonov-Bohm Effect

A charged particle acquires a phase:
$$\phi = \frac{e}{\hbar}\oint_C \mathbf{A} \cdot d\mathbf{r}$$

even when **B** = 0 along the path—a purely quantum effect!

---

## ✅ Daily Checklist

- [ ] Read Stewart 16.2-16.3
- [ ] Parametrize curves for integration
- [ ] Compute scalar line integrals (∫f ds)
- [ ] Compute vector line integrals (∫**F**·d**r**)
- [ ] Apply the Fundamental Theorem
- [ ] Test for path independence
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 52: Green's Theorem**
- Connecting line integrals to double integrals
- Circulation and flux forms
- Applications to area

---

*"Line integrals measure accumulation along paths—the mathematical foundation of work and circulation."*
