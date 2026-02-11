# Day 35: Rest, Review, and Week 6 Preparation

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 10:00 AM - 11:30 AM | 1.5 hours | Concept Review |
| Afternoon | 2:00 PM - 3:00 PM | 1 hour | Self-Assessment |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Week 6 Preview |

**Total Study Time: 3.5 hours (REST DAY)**

---

## 🧘 Rest Day Philosophy

You've completed the transition from single-variable to multivariable thinking:
- Vectors as fundamental objects
- Dot and cross products
- Lines and planes in 3D space

Your brain needs time to consolidate these spatial concepts.

---

## 📝 Week 5 Summary Sheet

### Vectors

**Component form:** $\mathbf{v} = \langle v_1, v_2, v_3 \rangle = v_1\mathbf{i} + v_2\mathbf{j} + v_3\mathbf{k}$

**Magnitude:** $|\mathbf{v}| = \sqrt{v_1^2 + v_2^2 + v_3^2}$

**Unit vector:** $\hat{\mathbf{v}} = \frac{\mathbf{v}}{|\mathbf{v}|}$

### Dot Product

**Algebraic:** $\mathbf{a} \cdot \mathbf{b} = a_1b_1 + a_2b_2 + a_3b_3$

**Geometric:** $\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos\theta$

**Angle:** $\theta = \arccos\left(\frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|}\right)$

**Projection:** $\text{proj}_\mathbf{a}\mathbf{b} = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}|^2}\mathbf{a}$

**Orthogonality:** $\mathbf{a} \perp \mathbf{b} \iff \mathbf{a} \cdot \mathbf{b} = 0$

### Cross Product

**Formula:** $\mathbf{a} \times \mathbf{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{vmatrix}$

**Magnitude:** $|\mathbf{a} \times \mathbf{b}| = |\mathbf{a}||\mathbf{b}|\sin\theta$

**Properties:**
- Result is perpendicular to both **a** and **b**
- Anti-commutative: **a** × **b** = −(**b** × **a**)
- Area of parallelogram = |**a** × **b**|

### Lines

**Parametric:** $x = x_0 + at, \quad y = y_0 + bt, \quad z = z_0 + ct$

**Symmetric:** $\frac{x - x_0}{a} = \frac{y - y_0}{b} = \frac{z - z_0}{c}$

### Planes

**Equation:** $ax + by + cz = d$

**Normal vector:** $\mathbf{n} = \langle a, b, c \rangle$

### Distance Formulas

**Point to plane:** $D = \frac{|ax_1 + by_1 + cz_1 - d|}{\sqrt{a^2 + b^2 + c^2}}$

**Point to line:** $D = \frac{|\overrightarrow{P_0P} \times \mathbf{v}|}{|\mathbf{v}|}$

---

## 🔄 Self-Assessment Quiz

**Q1:** Find the angle between ⟨1, 1, 0⟩ and ⟨1, 0, 1⟩.

<details>
<summary>Answer</summary>
cos θ = (1·1 + 1·0 + 0·1)/(√2 · √2) = 1/2
θ = 60°
</details>

**Q2:** Find a vector perpendicular to both ⟨1, 2, 0⟩ and ⟨0, 1, 1⟩.

<details>
<summary>Answer</summary>
⟨1, 2, 0⟩ × ⟨0, 1, 1⟩ = ⟨2-0, 0-1, 1-0⟩ = ⟨2, -1, 1⟩
</details>

**Q3:** Find the distance from (1, 2, 3) to the plane x + y + z = 0.

<details>
<summary>Answer</summary>
D = |1 + 2 + 3 - 0|/√3 = 6/√3 = 2√3
</details>

---

## 📊 Concept Map: Vectors and Space

```
                      VECTORS
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌──────────┐
    │   Dot   │    │  Cross   │    │ Position │
    │ Product │    │ Product  │    │ Vectors  │
    └────┬────┘    └────┬─────┘    └────┬─────┘
         │               │               │
    ┌────┴────┐    ┌────┴─────┐    ┌────┴─────┐
    │  Angle  │    │   Area   │    │  Lines   │
    │Projection│   │  Volume  │    │  Planes  │
    │  Work   │    │  Torque  │    │ Distance │
    └─────────┘    └──────────┘    └──────────┘
```

---

## 🔜 Week 6 Preview: Partial Derivatives

### The Big Picture

In single-variable calculus, we studied functions f(x) of one variable.

Now we study functions of **multiple variables**:
- f(x, y) — functions of two variables (surfaces)
- f(x, y, z) — functions of three variables (scalar fields)

### Key Questions

1. **How do we visualize** functions of two variables?
2. **How do we differentiate** when there are multiple variables?
3. **What is the "slope"** of a surface?

### Partial Derivatives

For f(x, y), we can ask:
- How does f change when x changes (holding y constant)?
- How does f change when y changes (holding x constant)?

These are **partial derivatives**:
$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$

$$\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y+h) - f(x, y)}{h}$$

### Preview Example

For f(x, y) = x²y + sin(y):

$$\frac{\partial f}{\partial x} = 2xy$$ (treat y as constant)

$$\frac{\partial f}{\partial y} = x² + \cos(y)$$ (treat x as constant)

### Topics Coming Up

**Day 36:** Functions of several variables, level curves
**Day 37:** Partial derivatives
**Day 38:** The gradient vector
**Day 39:** Directional derivatives
**Day 40:** Tangent planes
**Day 41:** Problem Set
**Day 42:** Rest and Review

### Quantum Mechanics Connection

The Schrödinger equation is a partial differential equation:
$$i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2} + V\psi$$

Partial derivatives are essential for quantum mechanics!

---

## 📈 Progress Tracker

| Week | Topic | Status |
|------|-------|--------|
| 1 | Limits & Continuity | ✅ |
| 2 | Differentiation | ✅ |
| 3 | Applications | ✅ |
| 4 | Integration | ✅ |
| 5 | Vectors & Space | ✅ |
| 6 | Partial Derivatives | 🔄 Starting |

**You are 42% through Month 2!**

---

## ✅ Checklist Before Week 6

- [ ] Can compute dot and cross products fluently
- [ ] Can find angles between vectors
- [ ] Can write equations of lines and planes
- [ ] Can calculate distances
- [ ] Can visualize 3D geometry
- [ ] Ready for functions of multiple variables!

---

## 📓 Reflection Questions

1. How is the dot product different from the cross product?
2. Why do we need two different products for vectors?
3. What's the physical significance of orthogonality?
4. How will partial derivatives extend what you learned in single-variable calculus?

---

## 💪 Motivation

You've now entered the world of multivariable mathematics. This is where:
- Physical intuition becomes essential
- Visualization skills pay dividends
- The mathematics of the real world lives

The partial derivatives you'll learn next week are the foundation of:
- Thermodynamics
- Fluid mechanics
- Electromagnetism
- Quantum mechanics

Every physical system with multiple variables requires these tools!

---

**Week 5 Complete! 🎉**

Tomorrow begins partial derivatives—where calculus meets functions of multiple variables.

*"Geometry is the archetype of the beauty of the world."* — Johannes Kepler
