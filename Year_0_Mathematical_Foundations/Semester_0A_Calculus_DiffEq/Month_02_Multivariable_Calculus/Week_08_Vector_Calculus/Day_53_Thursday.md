# Day 53: Surface Integrals and the Fundamental Theorems

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Surface Integrals |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Stokes' & Divergence Theorems |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Parametrize surfaces and compute surface area
2. Evaluate scalar surface integrals
3. Evaluate flux integrals
4. State and apply Stokes' Theorem
5. State and apply the Divergence Theorem

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 16.6**: Parametric Surfaces and Their Areas (pp. 1129-1138)
- **Section 16.7**: Surface Integrals (pp. 1139-1150)
- **Section 16.8**: Stokes' Theorem (pp. 1151-1159)
- **Section 16.9**: The Divergence Theorem (pp. 1160-1167)

---

## 📖 Core Content: Surface Integrals

### 1. Parametric Surfaces

A surface S can be parametrized by:
$$\mathbf{r}(u, v) = \langle x(u,v), y(u,v), z(u,v) \rangle$$

where (u, v) ranges over some region D in the uv-plane.

**Example:** Sphere of radius a:
$$\mathbf{r}(\phi, \theta) = \langle a\sin\phi\cos\theta, a\sin\phi\sin\theta, a\cos\phi \rangle$$

### 2. Surface Area Element

$$dS = |\mathbf{r}_u \times \mathbf{r}_v| \, du \, dv$$

For a surface z = g(x, y):
$$dS = \sqrt{1 + g_x^2 + g_y^2} \, dA$$

### 3. Scalar Surface Integral

$$\iint_S f \, dS = \iint_D f(\mathbf{r}(u,v)) \, |\mathbf{r}_u \times \mathbf{r}_v| \, du \, dv$$

### 4. Flux Integral (Vector Surface Integral)

$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_S \mathbf{F} \cdot \mathbf{n} \, dS = \iint_D \mathbf{F} \cdot (\mathbf{r}_u \times \mathbf{r}_v) \, du \, dv$$

For z = g(x, y):
$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_D \mathbf{F} \cdot \langle -g_x, -g_y, 1 \rangle \, dA$$

---

## ✏️ Surface Integral Examples

### Example 1: Surface Area of Sphere
Find the surface area of a sphere of radius a.

$$\mathbf{r}(\phi, \theta) = \langle a\sin\phi\cos\theta, a\sin\phi\sin\theta, a\cos\phi \rangle$$

$$|\mathbf{r}_\phi \times \mathbf{r}_\theta| = a^2\sin\phi$$

$$\text{Area} = \int_0^{2\pi} \int_0^\pi a^2\sin\phi \, d\phi \, d\theta = 2\pi a^2 \cdot 2 = 4\pi a^2$$

---

### Example 2: Flux Through a Surface
Find ∬_S **F** · d**S** where **F** = ⟨x, y, z⟩ and S is the hemisphere z = √(1-x²-y²), z ≥ 0.

Using z = g(x, y) = √(1-x²-y²):
$$g_x = \frac{-x}{\sqrt{1-x^2-y^2}}, \quad g_y = \frac{-y}{\sqrt{1-x^2-y^2}}$$

$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_D \langle x, y, z \rangle \cdot \langle \frac{x}{z}, \frac{y}{z}, 1 \rangle \, dA$$

$$= \iint_D \left(\frac{x^2 + y^2}{z} + z\right) dA = \iint_D \frac{x^2+y^2+z^2}{z} \, dA = \iint_D \frac{1}{z} \, dA$$

In polar: $= \int_0^{2\pi} \int_0^1 \frac{r}{\sqrt{1-r^2}} \, dr \, d\theta = 2\pi$

---

## 📖 Stokes' Theorem

### 5. Statement

> **Stokes' Theorem:** Let S be an oriented piecewise-smooth surface bounded by a simple, closed, piecewise-smooth boundary curve C with positive orientation. If **F** has continuous partial derivatives, then:
> $$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$$

### 6. Interpretation

- **Left:** Circulation of **F** around boundary C
- **Right:** Flux of curl **F** through surface S
- **Meaning:** Circulation = curl flux

### 7. Special Case

When S is a flat region in the xy-plane, Stokes' Theorem reduces to **Green's Theorem**!

---

## ✏️ Stokes' Theorem Example

### Example 3: Verify Stokes' Theorem
Verify Stokes' Theorem for **F** = ⟨y, z, x⟩ where S is the hemisphere z = √(1-x²-y²) and C is the unit circle in the xy-plane.

**Line integral (around C):**
**r**(t) = ⟨cos t, sin t, 0⟩, **F** = ⟨sin t, 0, cos t⟩

$$\oint_C \mathbf{F} \cdot d\mathbf{r} = \int_0^{2\pi} \langle \sin t, 0, \cos t \rangle \cdot \langle -\sin t, \cos t, 0 \rangle \, dt = \int_0^{2\pi} -\sin^2 t \, dt = -\pi$$

**Surface integral:**
$$\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \partial_x & \partial_y & \partial_z \\ y & z & x \end{vmatrix} = \langle -1, -1, -1 \rangle$$

For the hemisphere with outward normal:
$$\iint_S \langle -1, -1, -1 \rangle \cdot d\mathbf{S} = -\iint_S 1 \cdot dS_{upward} = -\pi$$ ✓

---

## 📖 Divergence Theorem

### 8. Statement

> **Divergence Theorem (Gauss's Theorem):** Let E be a simple solid region bounded by the closed surface S with outward orientation. If **F** has continuous partial derivatives, then:
> $$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_E \nabla \cdot \mathbf{F} \, dV$$

### 9. Interpretation

- **Left:** Total flux out through boundary S
- **Right:** Total divergence inside E
- **Meaning:** What flows out = what's produced inside

---

## ✏️ Divergence Theorem Example

### Example 4: Apply Divergence Theorem
Find ∬_S **F** · d**S** where **F** = ⟨x³, y³, z³⟩ and S is the sphere x² + y² + z² = 1.

**Direct:** Would be very complicated!

**Using Divergence Theorem:**
$$\nabla \cdot \mathbf{F} = 3x^2 + 3y^2 + 3z^2 = 3(x^2 + y^2 + z^2)$$

In spherical coordinates:
$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_E 3\rho^2 \, dV = \int_0^{2\pi} \int_0^\pi \int_0^1 3\rho^2 \cdot \rho^2\sin\phi \, d\rho \, d\phi \, d\theta$$

$$= 3 \cdot 2\pi \cdot 2 \cdot \frac{1}{5} = \frac{12\pi}{5}$$

---

## 📋 Summary: The Three Major Theorems

| Theorem | Dimension | Statement |
|---------|-----------|-----------|
| **Green's** | 2D | $\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_D (\text{curl } \mathbf{F}) \, dA$ |
| **Stokes'** | 3D surface | $\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$ |
| **Divergence** | 3D solid | $\iint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_E \nabla \cdot \mathbf{F} \, dV$ |

All three are instances of the **Generalized Stokes' Theorem**!

---

## 📝 Practice Problems

### Level 1: Surface Integrals
1. Find the surface area of z = x² + y² over the disk x² + y² ≤ 1
2. ∬_S z dS where S is the hemisphere z = √(4-x²-y²)
3. ∬_S **F** · d**S** where **F** = ⟨0, 0, z⟩ and S is the disk z = 0, x² + y² ≤ 1

### Level 2: Stokes' Theorem
4. Use Stokes' to evaluate ∮_C **F** · d**r** where **F** = ⟨z, x, y⟩ and C is the triangle (1,0,0), (0,1,0), (0,0,1)
5. Verify Stokes' for **F** = ⟨-y, x, 0⟩ where S is the disk z = 0, x² + y² ≤ 1

### Level 3: Divergence Theorem
6. Use Divergence Theorem: ∬_S ⟨x, y, z⟩ · d**S** where S is the unit sphere
7. Use Divergence Theorem: ∬_S ⟨x², y², z²⟩ · d**S** where S bounds the cube [0,1]³

### Level 4: Applications
8. Find the flux of **F** = ⟨x, y, z⟩/r³ through a sphere of radius a centered at origin
9. Verify that ∬_S (∇ × **F**) · d**S** = 0 for any closed surface S

---

## 📊 Answers

1. π(5√5 - 1)/6
2. 8π
3. 0
4. -1/2
5. Both give 2π
6. 4π
7. 3
8. 4πa² · 1/a² = 4π
9. Use Divergence Theorem: div(curl **F**) = 0

---

## 🔬 Quantum Mechanics Connection

### Maxwell's Equations

The divergence and Stokes' theorems underpin Maxwell's equations:

$$\oint_S \mathbf{E} \cdot d\mathbf{A} = \frac{Q_{enc}}{\epsilon_0}$$ (Gauss's law)

$$\oint_C \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{enc} + \mu_0\epsilon_0\frac{d\Phi_E}{dt}$$ (Ampère's law)

### Continuity Equation

$$\frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{j} = 0$$

Integrating with the Divergence Theorem gives conservation of charge!

---

## ✅ Daily Checklist

- [ ] Read Stewart 16.6-16.9
- [ ] Parametrize surfaces
- [ ] Compute surface integrals
- [ ] Apply Stokes' Theorem
- [ ] Apply Divergence Theorem
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 54: Week 8 Problem Set**
- Comprehensive review of vector calculus

---

*"The fundamental theorems of vector calculus reveal the deep unity of differentiation and integration in higher dimensions."*
