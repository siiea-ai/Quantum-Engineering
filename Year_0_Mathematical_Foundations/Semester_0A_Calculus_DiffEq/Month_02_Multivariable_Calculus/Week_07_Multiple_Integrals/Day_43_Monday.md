# Day 43: Double Integrals over Rectangles

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Double Integral Definition |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Iterated Integrals |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand double integrals as limits of Riemann sums
2. Interpret double integrals as volumes
3. Evaluate double integrals using iterated integrals
4. Apply Fubini's Theorem
5. Compute double integrals over rectangles

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 15.1**: Double Integrals over Rectangles (pp. 1000-1012)

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.02SC
**Lecture: Double Integrals**

### Professor Leonard
**Calculus 3: Introduction to Double Integrals**

### 3Blue1Brown
**Multivariable Calculus: Multiple Integrals**

---

## 📖 Core Content: Double Integrals

### 1. Motivation: Volume Under a Surface

Just as the single integral ∫ₐᵇ f(x)dx gives the area under a curve, the **double integral** ∬_R f(x,y)dA gives the **volume under a surface** z = f(x,y) over a region R.

### 2. Definition via Riemann Sums

Let f(x, y) be defined on a rectangle R = [a, b] × [c, d].

**Step 1:** Partition R into small subrectangles Rᵢⱼ with area ΔA = Δx·Δy

**Step 2:** Choose sample point (xᵢⱼ*, yᵢⱼ*) in each Rᵢⱼ

**Step 3:** Form the Riemann sum:
$$\sum_{i=1}^{m}\sum_{j=1}^{n} f(x_{ij}^*, y_{ij}^*) \Delta A$$

**Step 4:** Take the limit as m, n → ∞:

> **Definition:** The **double integral** of f over R is:
> $$\iint_R f(x, y) \, dA = \lim_{m,n \to \infty} \sum_{i=1}^{m}\sum_{j=1}^{n} f(x_{ij}^*, y_{ij}^*) \Delta A$$

### 3. Geometric Interpretation

If f(x, y) ≥ 0, then ∬_R f(x, y) dA represents the **volume** of the solid bounded by:
- Below: the region R in the xy-plane
- Above: the surface z = f(x, y)
- Sides: vertical walls over the boundary of R

---

## 📖 Iterated Integrals

### 4. Fubini's Theorem

Computing double integrals directly from the definition is impractical. Fubini's Theorem saves us!

> **Fubini's Theorem:** If f is continuous on R = [a, b] × [c, d], then:
> $$\iint_R f(x, y) \, dA = \int_a^b \int_c^d f(x, y) \, dy \, dx = \int_c^d \int_a^b f(x, y) \, dx \, dy$$

**Key point:** We can evaluate the double integral as two successive single integrals!

### 5. Iterated Integral Notation

$$\int_a^b \int_c^d f(x, y) \, dy \, dx$$

means: First integrate with respect to y (treating x as constant), then integrate the result with respect to x.

### 6. Order of Integration

The order doesn't matter for continuous functions on rectangles:
$$\int_a^b \int_c^d f(x, y) \, dy \, dx = \int_c^d \int_a^b f(x, y) \, dx \, dy$$

---

## ✏️ Worked Examples

### Example 1: Basic Double Integral
Evaluate ∬_R (x + 2y) dA where R = [0, 2] × [0, 1].

$$\iint_R (x + 2y) \, dA = \int_0^2 \int_0^1 (x + 2y) \, dy \, dx$$

**Inner integral** (integrate with respect to y):
$$\int_0^1 (x + 2y) \, dy = \left[xy + y^2\right]_0^1 = x + 1$$

**Outer integral:**
$$\int_0^2 (x + 1) \, dx = \left[\frac{x^2}{2} + x\right]_0^2 = 2 + 2 = 4$$

**Answer:** ∬_R (x + 2y) dA = 4

---

### Example 2: Verify Order Independence
Compute the same integral with reversed order:

$$\int_0^1 \int_0^2 (x + 2y) \, dx \, dy$$

**Inner integral:**
$$\int_0^2 (x + 2y) \, dx = \left[\frac{x^2}{2} + 2xy\right]_0^2 = 2 + 4y$$

**Outer integral:**
$$\int_0^1 (2 + 4y) \, dy = \left[2y + 2y^2\right]_0^1 = 2 + 2 = 4$$ ✓

---

### Example 3: Product of Functions
Evaluate ∬_R x²y³ dA where R = [0, 1] × [0, 2].

When f(x, y) = g(x)h(y), we have:
$$\iint_R g(x)h(y) \, dA = \left(\int_a^b g(x) \, dx\right)\left(\int_c^d h(y) \, dy\right)$$

$$\iint_R x^2 y^3 \, dA = \left(\int_0^1 x^2 \, dx\right)\left(\int_0^2 y^3 \, dy\right)$$

$$= \left[\frac{x^3}{3}\right]_0^1 \cdot \left[\frac{y^4}{4}\right]_0^2 = \frac{1}{3} \cdot 4 = \frac{4}{3}$$

---

### Example 4: Exponential Function
Evaluate ∬_R e^(x+y) dA where R = [0, 1] × [0, 1].

$$\iint_R e^{x+y} \, dA = \int_0^1 \int_0^1 e^x e^y \, dy \, dx$$

$$= \left(\int_0^1 e^x \, dx\right)\left(\int_0^1 e^y \, dy\right) = (e-1)(e-1) = (e-1)^2$$

---

### Example 5: Volume Under a Plane
Find the volume of the solid under z = 4 - x - y over R = [0, 1] × [0, 2].

$$V = \iint_R (4 - x - y) \, dA = \int_0^1 \int_0^2 (4 - x - y) \, dy \, dx$$

**Inner:**
$$\int_0^2 (4 - x - y) \, dy = \left[(4-x)y - \frac{y^2}{2}\right]_0^2 = 2(4-x) - 2 = 6 - 2x$$

**Outer:**
$$\int_0^1 (6 - 2x) \, dx = \left[6x - x^2\right]_0^1 = 6 - 1 = 5$$

**Volume = 5 cubic units**

---

## 📋 Properties of Double Integrals

### Linearity
$$\iint_R [f(x,y) + g(x,y)] \, dA = \iint_R f(x,y) \, dA + \iint_R g(x,y) \, dA$$

$$\iint_R cf(x,y) \, dA = c \iint_R f(x,y) \, dA$$

### Comparison
If f(x, y) ≤ g(x, y) on R, then:
$$\iint_R f(x,y) \, dA \leq \iint_R g(x,y) \, dA$$

### Area
$$\iint_R 1 \, dA = \text{Area of } R$$

---

## 📐 Average Value

The **average value** of f over R is:
$$f_{avg} = \frac{1}{\text{Area}(R)} \iint_R f(x, y) \, dA$$

---

## 📝 Practice Problems

### Level 1: Basic Iterated Integrals
1. $\int_0^2 \int_0^3 (x + y) \, dy \, dx$
2. $\int_1^2 \int_0^1 xy \, dx \, dy$
3. $\int_0^1 \int_0^1 (x^2 + y^2) \, dx \, dy$
4. $\int_0^{\pi} \int_0^1 y\sin(x) \, dy \, dx$

### Level 2: Volume Computations
5. Find the volume under z = xy over R = [0, 2] × [0, 3].
6. Find the volume under z = 1 + x + y over R = [0, 1] × [0, 1].
7. Find the volume under z = e^(-x-y) over R = [0, 1] × [0, 1].

### Level 3: Product Functions
8. $\iint_R x^3y^2 \, dA$ where R = [0, 1] × [0, 2]
9. $\iint_R \cos(x)\sin(y) \, dA$ where R = [0, π/2] × [0, π]
10. $\iint_R e^x e^{2y} \, dA$ where R = [0, 1] × [0, 1]

### Level 4: Verification
11. Verify Fubini's Theorem by computing $\iint_R xy^2 \, dA$ both ways for R = [0, 1] × [0, 2].

### Level 5: Average Value
12. Find the average value of f(x, y) = xy over R = [0, 2] × [0, 4].
13. Find the average value of f(x, y) = x² + y² over R = [0, 1] × [0, 1].

---

## 📊 Answers

1. 15
2. 3/4
3. 2/3
4. 1
5. 9
6. 2
7. (1 - e⁻¹)²
8. 2/3
9. 2
10. (e - 1)(e² - 1)
11. 4/3 both ways
12. 4
13. 2/3

---

## 🔬 Quantum Mechanics Connection

### Normalization in 2D

For a 2D wave function ψ(x, y), normalization requires:
$$\iint_R |\psi(x, y)|^2 \, dx \, dy = 1$$

### Probability

The probability of finding a particle in region R:
$$P = \iint_R |\psi(x, y)|^2 \, dx \, dy$$

### Expectation Values

$$\langle x \rangle = \iint x|\psi|^2 \, dA, \quad \langle y \rangle = \iint y|\psi|^2 \, dA$$

---

## ✅ Daily Checklist

- [ ] Read Stewart 15.1
- [ ] Understand double integral as volume
- [ ] Master iterated integral computation
- [ ] Know Fubini's Theorem
- [ ] Apply product function shortcut
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 44: Double Integrals over General Regions**
- Type I and Type II regions
- Variable limits of integration
- Setting up integrals from geometric descriptions

---

*"The double integral extends our ability to accumulate from lines to regions."*
