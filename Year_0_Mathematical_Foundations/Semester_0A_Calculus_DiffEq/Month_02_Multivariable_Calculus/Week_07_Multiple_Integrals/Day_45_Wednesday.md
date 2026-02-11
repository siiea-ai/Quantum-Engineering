# Day 45: Double Integrals in Polar Coordinates

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Polar Coordinate Review |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Integration in Polar |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Convert between Cartesian and polar coordinates
2. Express dA in polar coordinates
3. Recognize when polar coordinates simplify integration
4. Set up and evaluate double integrals in polar form
5. Apply to circular and angular regions

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 15.3**: Double Integrals in Polar Coordinates (pp. 1025-1033)

---

## 📖 Core Content: Polar Coordinates

### 1. Review of Polar Coordinates

**Conversion formulas:**
- x = r cos θ
- y = r sin θ
- r² = x² + y²
- tan θ = y/x

### 2. The Key Formula: dA in Polar

> **Theorem:** In polar coordinates:
> $$dA = r \, dr \, d\theta$$

**Not** dr dθ! The factor r is essential.

### 3. Why the Factor r?

A small polar "rectangle" has:
- Radial length: dr
- Arc length: r dθ

Area ≈ dr · (r dθ) = r dr dθ

### 4. Polar Double Integral

$$\iint_R f(x, y) \, dA = \iint_R f(r\cos\theta, r\sin\theta) \, r \, dr \, d\theta$$

---

## 📋 Common Polar Regions

### Polar Rectangle
$$R = \{(r, \theta) : a \leq r \leq b, \, \alpha \leq \theta \leq \beta\}$$

$$\iint_R f \, dA = \int_\alpha^\beta \int_a^b f(r\cos\theta, r\sin\theta) \cdot r \, dr \, d\theta$$

### Disk of Radius a
$$\iint_D f \, dA = \int_0^{2\pi} \int_0^a f \cdot r \, dr \, d\theta$$

### Sector
$$\iint_S f \, dA = \int_\alpha^\beta \int_0^{g(\theta)} f \cdot r \, dr \, d\theta$$

---

## ✏️ Worked Examples

### Example 1: Circle
Evaluate ∬_D (x² + y²) dA where D is the disk x² + y² ≤ 4.

**In polar:** x² + y² = r², disk is 0 ≤ r ≤ 2, 0 ≤ θ ≤ 2π

$$\iint_D (x^2 + y^2) \, dA = \int_0^{2\pi} \int_0^2 r^2 \cdot r \, dr \, d\theta$$

$$= \int_0^{2\pi} \int_0^2 r^3 \, dr \, d\theta = \int_0^{2\pi} \left[\frac{r^4}{4}\right]_0^2 d\theta$$

$$= \int_0^{2\pi} 4 \, d\theta = 8\pi$$

---

### Example 2: Semicircle
Evaluate ∬_D e^(-(x²+y²)) dA where D is the upper half of the disk x² + y² ≤ 1.

**Region:** 0 ≤ r ≤ 1, 0 ≤ θ ≤ π

$$\iint_D e^{-r^2} r \, dr \, d\theta = \int_0^\pi \int_0^1 re^{-r^2} \, dr \, d\theta$$

**Inner integral:** Let u = r², du = 2r dr
$$\int_0^1 re^{-r^2} \, dr = -\frac{1}{2}e^{-r^2}\Big|_0^1 = \frac{1}{2}(1 - e^{-1})$$

**Outer integral:**
$$\int_0^\pi \frac{1}{2}(1 - e^{-1}) \, d\theta = \frac{\pi}{2}(1 - e^{-1})$$

---

### Example 3: Annular Region
Find the area of the region between circles r = 1 and r = 3.

$$\text{Area} = \int_0^{2\pi} \int_1^3 r \, dr \, d\theta = \int_0^{2\pi} \left[\frac{r^2}{2}\right]_1^3 d\theta$$

$$= \int_0^{2\pi} \frac{9-1}{2} \, d\theta = 4 \cdot 2\pi = 8\pi$$

Check: π(3²) - π(1²) = 9π - π = 8π ✓

---

### Example 4: Cardioid
Find the area enclosed by r = 1 + cos θ.

$$\text{Area} = \int_0^{2\pi} \int_0^{1+\cos\theta} r \, dr \, d\theta = \int_0^{2\pi} \frac{(1+\cos\theta)^2}{2} \, d\theta$$

$$= \frac{1}{2}\int_0^{2\pi} (1 + 2\cos\theta + \cos^2\theta) \, d\theta$$

Using cos²θ = (1 + cos 2θ)/2:
$$= \frac{1}{2}\int_0^{2\pi} \left(\frac{3}{2} + 2\cos\theta + \frac{\cos 2\theta}{2}\right) d\theta = \frac{1}{2} \cdot \frac{3}{2} \cdot 2\pi = \frac{3\pi}{2}$$

---

### Example 5: The Gaussian Integral
Evaluate $\int_0^\infty e^{-x^2} \, dx$ using polar coordinates!

Let $I = \int_0^\infty e^{-x^2} dx$. Then:
$$I^2 = \left(\int_0^\infty e^{-x^2} dx\right)\left(\int_0^\infty e^{-y^2} dy\right) = \int_0^\infty \int_0^\infty e^{-(x^2+y^2)} dx \, dy$$

This is ∬ over the first quadrant. In polar:
$$I^2 = \int_0^{\pi/2} \int_0^\infty e^{-r^2} r \, dr \, d\theta$$

$$= \int_0^{\pi/2} \left[-\frac{1}{2}e^{-r^2}\right]_0^\infty d\theta = \int_0^{\pi/2} \frac{1}{2} \, d\theta = \frac{\pi}{4}$$

Therefore: $I = \frac{\sqrt{\pi}}{2}$

This is one of the most important results in mathematics!

---

## 📝 Practice Problems

### Level 1: Basic Polar
1. ∬_D 1 dA where D is the disk r ≤ 3
2. ∬_D xy dA where D is the quarter-disk x ≥ 0, y ≥ 0, x² + y² ≤ 1
3. ∬_D √(x² + y²) dA where D is 1 ≤ r ≤ 2

### Level 2: Exponential
4. ∬_D e^(x²+y²) dA where D is the unit disk
5. ∬_D (x² + y²)e^(-(x²+y²)) dA over all of ℝ²

### Level 3: Areas
6. Find the area inside r = 2cos θ
7. Find the area inside r = sin 2θ (one petal)
8. Find the area inside r = 2 + cos θ

### Level 4: Setting Up
9. Convert to polar and evaluate: ∬_D (x + y) dA where D: x² + y² ≤ 4, x ≥ 0
10. Volume of the solid under z = √(x² + y²) over the disk x² + y² ≤ 4

### Level 5: Challenge
11. Show: $\int_0^\infty e^{-x^2} \cos(2bx) \, dx = \frac{\sqrt{\pi}}{2}e^{-b^2}$
12. Find the volume inside both the cylinder x² + y² = 4 and the sphere x² + y² + z² = 16.

---

## 📊 Answers

1. 9π
2. 1/4
3. 14π/3
4. π(e - 1)
5. π
6. π
7. π/8
8. 9π/2
9. 16/3
10. 16π/3
11. (Use completing square in exponent)
12. 128π/3(2 - √3)

---

## 🔬 Quantum Mechanics Connection

### Hydrogen Atom in 2D

The 2D hydrogen wave functions naturally involve polar coordinates:
$$\psi_{n,m}(r, \theta) = R_{n,|m|}(r) e^{im\theta}$$

### Normalization
$$\int_0^{2\pi} \int_0^\infty |R(r)|^2 r \, dr \, d\theta = 1$$

The factor r in dA is crucial!

### Angular Momentum
The angular part e^(imθ) relates to angular momentum quantum number m.

---

## ✅ Daily Checklist

- [ ] Read Stewart 15.3
- [ ] Remember dA = r dr dθ (not dr dθ!)
- [ ] Practice converting regions to polar
- [ ] Evaluate Gaussian-type integrals
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 46: Applications of Double Integrals**
- Mass and center of mass
- Moments of inertia
- Surface area

---

*"Polar coordinates reveal the circular symmetry hidden in rectangular problems."*
