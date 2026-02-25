# Day 46: Triple Integrals and Applications

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Triple Integrals |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Set up and evaluate triple integrals
2. Choose appropriate order of integration
3. Compute mass and center of mass
4. Calculate moments of inertia
5. Apply multiple integrals to physical problems

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 15.6**: Triple Integrals (pp. 1054-1064)
- **Section 15.4**: Applications (pp. 1034-1045)

---

## 📖 Core Content: Triple Integrals

### 1. Definition

> **Definition:** The **triple integral** of f over a box B = [a,b] × [c,d] × [r,s] is:
> $$\iiint_B f(x, y, z) \, dV = \lim \sum f(x_{ijk}^*, y_{ijk}^*, z_{ijk}^*) \Delta V$$

### 2. Iterated Triple Integral

By Fubini's Theorem:
$$\iiint_B f \, dV = \int_a^b \int_c^d \int_r^s f(x, y, z) \, dz \, dy \, dx$$

Any order works for continuous f on boxes!

### 3. General Regions

For more complex regions E:
$$\iiint_E f \, dV = \int_a^b \int_{g_1(x)}^{g_2(x)} \int_{u_1(x,y)}^{u_2(x,y)} f(x, y, z) \, dz \, dy \, dx$$

---

## ✏️ Triple Integral Examples

### Example 1: Box Region
Evaluate ∭_B xyz² dV where B = [0,1] × [0,2] × [0,3].

$$\iiint_B xyz^2 \, dV = \int_0^1 \int_0^2 \int_0^3 xyz^2 \, dz \, dy \, dx$$

$$= \int_0^1 \int_0^2 xy\left[\frac{z^3}{3}\right]_0^3 dy \, dx = \int_0^1 \int_0^2 9xy \, dy \, dx$$

$$= \int_0^1 9x\left[\frac{y^2}{2}\right]_0^2 dx = \int_0^1 18x \, dx = 9$$

---

### Example 2: Tetrahedron
Evaluate ∭_E z dV where E is bounded by x + y + z = 1, x = 0, y = 0, z = 0.

**Bounds:**
- x: 0 to 1
- y: 0 to 1-x
- z: 0 to 1-x-y

$$\iiint_E z \, dV = \int_0^1 \int_0^{1-x} \int_0^{1-x-y} z \, dz \, dy \, dx$$

$$= \int_0^1 \int_0^{1-x} \frac{(1-x-y)^2}{2} \, dy \, dx$$

Let u = 1 - x - y, du = -dy:
$$= \int_0^1 \left[-\frac{(1-x-y)^3}{6}\right]_0^{1-x} dx = \int_0^1 \frac{(1-x)^3}{6} \, dx = \frac{1}{24}$$

---

### Example 3: Volume
Find the volume of the region bounded by z = x² + y² and z = 4.

$$V = \iiint_E 1 \, dV = \int_{-2}^{2} \int_{-\sqrt{4-x^2}}^{\sqrt{4-x^2}} \int_{x^2+y^2}^{4} dz \, dy \, dx$$

Using polar is easier:
$$V = \int_0^{2\pi} \int_0^2 \int_{r^2}^4 r \, dz \, dr \, d\theta = \int_0^{2\pi} \int_0^2 (4-r^2)r \, dr \, d\theta$$

$$= \int_0^{2\pi} \left[2r^2 - \frac{r^4}{4}\right]_0^2 d\theta = \int_0^{2\pi} 4 \, d\theta = 8\pi$$

---

## 📖 Applications of Multiple Integrals

### 4. Mass

For a region with density function ρ(x, y, z):
$$m = \iiint_E \rho(x, y, z) \, dV$$

### 5. Center of Mass

$$\bar{x} = \frac{1}{m}\iiint_E x\rho \, dV, \quad \bar{y} = \frac{1}{m}\iiint_E y\rho \, dV, \quad \bar{z} = \frac{1}{m}\iiint_E z\rho \, dV$$

### 6. Moments of Inertia

About the z-axis:
$$I_z = \iiint_E (x^2 + y^2)\rho \, dV$$

About the x-axis:
$$I_x = \iiint_E (y^2 + z^2)\rho \, dV$$

---

## ✏️ Application Examples

### Example 4: Mass of a Solid
Find the mass of the solid bounded by z = 0, z = 1, x² + y² = 1 with density ρ = z.

$$m = \int_0^{2\pi} \int_0^1 \int_0^1 z \cdot r \, dz \, dr \, d\theta$$

$$= \int_0^{2\pi} \int_0^1 \frac{r}{2} \, dr \, d\theta = \int_0^{2\pi} \frac{1}{4} \, d\theta = \frac{\pi}{2}$$

---

### Example 5: Center of Mass
Find the center of mass of a hemisphere z = √(a² - x² - y²), z ≥ 0 with uniform density.

By symmetry: $\bar{x} = \bar{y} = 0$

$$m = \frac{2}{3}\pi a^3 \rho$$

$$M_{xy} = \int_0^{2\pi} \int_0^a \int_0^{\sqrt{a^2-r^2}} z \cdot r \, dz \, dr \, d\theta$$

Computing: $\bar{z} = \frac{3a}{8}$

---

## 📝 Practice Problems

### Level 1: Basic Triple Integrals
1. ∭_B xyz dV where B = [0,1]³
2. ∭_B (x + y + z) dV where B = [0,2] × [0,1] × [0,3]

### Level 2: Non-Rectangular
3. ∭_E z dV where E is below z = 4 - x² - y² and above z = 0
4. ∭_E xy dV where E is bounded by y = x, y = 0, z = 0, z = x, x = 1

### Level 3: Volume
5. Find volume bounded by z = x² + y², z = 8 - x² - y²
6. Find volume of the region inside both x² + y² + z² = 4 and x² + y² = 1

### Level 4: Applications
7. Find mass of unit cube with ρ(x,y,z) = x + y + z
8. Find center of mass of the tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1) with ρ = 1

### Level 5: Moments
9. Find I_z for a cylinder of radius R, height h, uniform density
10. Find I_z for a solid sphere of radius a, uniform density

---

## 📊 Answers

1. 1/8
2. 18
3. 8π
4. 1/12
5. 16π
6. 4π(2 - √3)/3
7. 3/2
8. (1/4, 1/4, 1/4)
9. ½MR²
10. ⅖Ma²

---

## 🔬 Quantum Mechanics Connection

### 3D Normalization

For a 3D wave function:
$$\iiint_{all space} |\psi(x, y, z)|^2 \, dV = 1$$

### Expectation Values

$$\langle x \rangle = \iiint x|\psi|^2 \, dV$$
$$\langle r \rangle = \iiint r|\psi|^2 \, dV$$

### Probability in a Region

$$P(E) = \iiint_E |\psi|^2 \, dV$$

---

## ✅ Daily Checklist

- [ ] Read Stewart 15.6 and 15.4
- [ ] Set up triple integrals for various regions
- [ ] Compute mass and center of mass
- [ ] Understand moments of inertia
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 47: Week 7 Problem Set**
- Comprehensive review of multiple integrals

---

*"Triple integrals let us measure properties of three-dimensional matter."*
