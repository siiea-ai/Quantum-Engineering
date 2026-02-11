# Day 50: Vector Fields

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Vector Field Fundamentals |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Divergence and Curl |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand what a vector field is
2. Visualize vector fields in 2D and 3D
3. Recognize gradient, conservative, and source/sink fields
4. Compute divergence of a vector field
5. Compute curl of a vector field

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 16.1**: Vector Fields (pp. 1078-1087)
- **Section 16.5**: Curl and Divergence (pp. 1118-1128)

---

## 🎬 Video Resources

### 3Blue1Brown
**Divergence and Curl** - Outstanding visual intuition

### MIT OpenCourseWare 18.02SC
**Lecture: Vector Fields**

### Professor Leonard
**Calculus 3: Vector Fields, Curl, and Divergence**

---

## 📖 Core Content: Vector Fields

### 1. What is a Vector Field?

> **Definition:** A **vector field** on a region D is a function **F** that assigns to each point (x, y) [or (x, y, z)] a vector **F**(x, y) [or **F**(x, y, z)].

**2D Vector Field:**
$$\mathbf{F}(x, y) = P(x, y)\mathbf{i} + Q(x, y)\mathbf{j} = \langle P(x, y), Q(x, y) \rangle$$

**3D Vector Field:**
$$\mathbf{F}(x, y, z) = P\mathbf{i} + Q\mathbf{j} + R\mathbf{k} = \langle P, Q, R \rangle$$

### 2. Examples of Vector Fields

**Gravitational Field:**
$$\mathbf{F}(x, y, z) = -\frac{GMm}{(x^2+y^2+z^2)^{3/2}}\langle x, y, z \rangle$$

**Electric Field:**
$$\mathbf{E}(\mathbf{r}) = \frac{kq}{|\mathbf{r}|^2}\hat{\mathbf{r}}$$

**Velocity Field:**
$$\mathbf{v}(x, y) = \langle -y, x \rangle$$ (rotation)

### 3. Visualizing Vector Fields

At each point, draw an arrow representing **F** at that point.
- Direction: where **F** points
- Length: magnitude |**F**|

---

## 📖 Gradient Fields (Conservative Fields)

### 4. Definition

> **Definition:** A vector field **F** is a **gradient field** (or **conservative field**) if there exists a scalar function f such that:
> $$\mathbf{F} = \nabla f$$

The function f is called the **potential function**.

### 5. Example

If f(x, y) = x²y + y³, then:
$$\nabla f = \langle 2xy, x^2 + 3y^2 \rangle$$

So **F**(x, y) = ⟨2xy, x² + 3y²⟩ is a gradient field with potential f.

### 6. Why "Conservative"?

In physics, conservative forces (like gravity) can be derived from potential energy:
$$\mathbf{F} = -\nabla V$$

Work done by a conservative force depends only on endpoints, not the path!

---

## 📖 Divergence

### 7. Definition

> **Definition:** The **divergence** of **F** = ⟨P, Q, R⟩ is:
> $$\text{div } \mathbf{F} = \nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$$

In 2D: div **F** = ∂P/∂x + ∂Q/∂y

### 8. Interpretation

Divergence measures the **rate of expansion** (outward flux per unit volume):
- div **F** > 0: source (fluid flowing out)
- div **F** < 0: sink (fluid flowing in)
- div **F** = 0: incompressible flow

### 9. Example

**F**(x, y, z) = ⟨x, y, z⟩

$$\text{div } \mathbf{F} = \frac{\partial x}{\partial x} + \frac{\partial y}{\partial y} + \frac{\partial z}{\partial z} = 1 + 1 + 1 = 3$$

This field has constant positive divergence (uniform expansion).

---

## 📖 Curl

### 10. Definition

> **Definition:** The **curl** of **F** = ⟨P, Q, R⟩ is:
> $$\text{curl } \mathbf{F} = \nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix}$$

$$= \left(\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}\right)\mathbf{i} - \left(\frac{\partial R}{\partial x} - \frac{\partial P}{\partial z}\right)\mathbf{j} + \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right)\mathbf{k}$$

### 11. Interpretation

Curl measures the **rotation** or **circulation density**:
- |curl **F**| gives the rate of rotation
- Direction of curl **F** is the axis of rotation (right-hand rule)

### 12. 2D Curl (Scalar)

For **F**(x, y) = ⟨P, Q⟩:
$$\text{curl } \mathbf{F} = \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}$$

This is the z-component of the 3D curl.

### 13. Example

**F**(x, y, z) = ⟨y, -x, 0⟩

$$\text{curl } \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ y & -x & 0 \end{vmatrix}$$

$$= \mathbf{i}(0 - 0) - \mathbf{j}(0 - 0) + \mathbf{k}(-1 - 1) = -2\mathbf{k}$$

This field has constant negative curl (clockwise rotation when viewed from above).

---

## 📋 Important Identities

### Vector Calculus Identities

1. **curl(∇f) = 0** (gradient fields are irrotational)

2. **div(curl **F**) = 0** (curl fields are divergence-free)

3. **div(f**F**) = f(div **F**) + **F** · ∇f**

4. **curl(f**F**) = f(curl **F**) + (∇f) × **F****

---

## ✏️ Worked Examples

### Example 1: Compute Divergence and Curl
For **F** = ⟨x²y, yz², xz⟩, find div **F** and curl **F**.

**Divergence:**
$$\text{div } \mathbf{F} = \frac{\partial}{\partial x}(x^2y) + \frac{\partial}{\partial y}(yz^2) + \frac{\partial}{\partial z}(xz) = 2xy + z^2 + x$$

**Curl:**
$$\text{curl } \mathbf{F} = \langle \frac{\partial(xz)}{\partial y} - \frac{\partial(yz^2)}{\partial z}, \frac{\partial(x^2y)}{\partial z} - \frac{\partial(xz)}{\partial x}, \frac{\partial(yz^2)}{\partial x} - \frac{\partial(x^2y)}{\partial y} \rangle$$

$$= \langle 0 - 2yz, 0 - z, 0 - x^2 \rangle = \langle -2yz, -z, -x^2 \rangle$$

---

### Example 2: Is F Conservative?
Is **F** = ⟨2xy, x² + 1⟩ a gradient field?

Check if curl **F** = 0:
$$\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = \frac{\partial(x^2+1)}{\partial x} - \frac{\partial(2xy)}{\partial y} = 2x - 2x = 0$$

Yes! **F** is conservative. The potential function is f(x, y) = x²y + y.

---

### Example 3: Find Potential Function
Find f such that **F** = ⟨y cos(xy), x cos(xy)⟩ = ∇f.

From ∂f/∂x = y cos(xy):
$$f = \int y\cos(xy) \, dx = \sin(xy) + g(y)$$

From ∂f/∂y = x cos(xy):
$$\frac{\partial}{\partial y}[\sin(xy) + g(y)] = x\cos(xy) + g'(y) = x\cos(xy)$$

So g'(y) = 0, giving g(y) = C.

**Potential function:** f(x, y) = sin(xy) + C

---

## 📝 Practice Problems

### Level 1: Basic Computations
1. Find div **F** for **F** = ⟨x², xy, yz⟩
2. Find curl **F** for **F** = ⟨yz, xz, xy⟩
3. Find div **F** for **F** = ⟨eˣ, sin y, z²⟩

### Level 2: Conservative Fields
4. Is **F** = ⟨y, x⟩ conservative? If so, find f.
5. Is **F** = ⟨y, -x⟩ conservative?
6. Is **F** = ⟨2x + y, x + 2y⟩ conservative? If so, find f.

### Level 3: Vector Identities
7. Verify curl(∇f) = 0 for f(x, y, z) = x²y + z³
8. Verify div(curl **F**) = 0 for **F** = ⟨xy, yz, xz⟩

### Level 4: Physical Fields
9. For the velocity field **v** = ⟨-y/(x²+y²), x/(x²+y²)⟩, find curl **v**.
10. For **F** = ⟨x, y, z⟩/r³ where r = √(x²+y²+z²), find div **F** for r ≠ 0.

### Level 5: Theory
11. Prove that if **F** = ∇f, then curl **F** = 0.
12. Show that **F** = ⟨-y/(x²+y²), x/(x²+y²)⟩ has curl = 0 but is NOT conservative on ℝ² \ {0}.

---

## 📊 Answers

1. 2x + x + y = 3x + y
2. ⟨x - x, y - y, z - z⟩ = **0**
3. eˣ + cos y + 2z
4. Yes; f = xy
5. No (curl = -2)
6. Yes; f = x² + xy + y²
7. Direct computation
8. Direct computation
9. 0 (except at origin)
10. 0 (for r ≠ 0)
11. Use mixed partial equality
12. The region is not simply connected

---

## 🔬 Quantum Mechanics Connection

### Probability Current

The probability current in QM is:
$$\mathbf{j} = \frac{\hbar}{2mi}(\psi^*\nabla\psi - \psi\nabla\psi^*)$$

### Continuity Equation

$$\frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{j} = 0$$

This says probability is conserved—divergence of current equals rate of change of density!

### Angular Momentum

The angular momentum operator involves curl:
$$\hat{\mathbf{L}} = -i\hbar(\mathbf{r} \times \nabla)$$

---

## ✅ Daily Checklist

- [ ] Read Stewart 16.1 and 16.5
- [ ] Understand vector fields as arrow diagrams
- [ ] Compute divergence (measures expansion)
- [ ] Compute curl (measures rotation)
- [ ] Test if a field is conservative
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 51: Line Integrals**
- Scalar line integrals ∫_C f ds
- Vector line integrals ∫_C **F** · d**r**
- Work done by a force field
- Path independence

---

*"Vector fields are the language of physics—they describe forces, flows, and fields throughout space."*
