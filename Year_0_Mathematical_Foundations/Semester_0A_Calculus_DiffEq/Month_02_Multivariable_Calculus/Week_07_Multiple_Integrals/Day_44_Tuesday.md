# Day 44: Double Integrals over General Regions

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Type I and II Regions |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Setting Up Integrals |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Identify Type I and Type II regions
2. Set up double integrals with variable limits
3. Choose the best order of integration
4. Reverse the order of integration
5. Evaluate double integrals over non-rectangular regions

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 15.2**: Double Integrals over General Regions (pp. 1013-1024)

---

## 📖 Core Content: General Regions

### 1. Type I Regions (Vertically Simple)

> **Definition:** A **Type I region** is bounded between two curves y = g₁(x) and y = g₂(x) where g₁(x) ≤ g₂(x) for a ≤ x ≤ b:
> $$D = \{(x, y) : a \leq x \leq b, \, g_1(x) \leq y \leq g_2(x)\}$$

**Double integral over Type I:**
$$\iint_D f(x, y) \, dA = \int_a^b \int_{g_1(x)}^{g_2(x)} f(x, y) \, dy \, dx$$

The inner limits depend on x!

### 2. Type II Regions (Horizontally Simple)

> **Definition:** A **Type II region** is bounded between two curves x = h₁(y) and x = h₂(y) where h₁(y) ≤ h₂(y) for c ≤ y ≤ d:
> $$D = \{(x, y) : c \leq y \leq d, \, h_1(y) \leq x \leq h_2(y)\}$$

**Double integral over Type II:**
$$\iint_D f(x, y) \, dA = \int_c^d \int_{h_1(y)}^{h_2(y)} f(x, y) \, dx \, dy$$

The inner limits depend on y!

---

## ✏️ Worked Examples

### Example 1: Type I Region
Evaluate ∬_D xy dA where D is bounded by y = x² and y = 2x.

**Step 1:** Find intersection points.
x² = 2x → x(x - 2) = 0 → x = 0, 2

**Step 2:** Identify bounds.
- x ranges from 0 to 2
- For each x, y ranges from x² (lower) to 2x (upper)

**Step 3:** Set up and evaluate.
$$\iint_D xy \, dA = \int_0^2 \int_{x^2}^{2x} xy \, dy \, dx$$

**Inner integral:**
$$\int_{x^2}^{2x} xy \, dy = x\left[\frac{y^2}{2}\right]_{x^2}^{2x} = x\left(\frac{4x^2}{2} - \frac{x^4}{2}\right) = 2x^3 - \frac{x^5}{2}$$

**Outer integral:**
$$\int_0^2 \left(2x^3 - \frac{x^5}{2}\right) dx = \left[\frac{x^4}{2} - \frac{x^6}{12}\right]_0^2 = 8 - \frac{64}{12} = 8 - \frac{16}{3} = \frac{8}{3}$$

---

### Example 2: Type II Region
Evaluate ∬_D x dA where D is bounded by x = y² and x = 4.

**Step 1:** Find bounds.
y² = 4 → y = ±2

**Step 2:** Set up as Type II (easier!).
- y ranges from -2 to 2
- For each y, x ranges from y² to 4

$$\iint_D x \, dA = \int_{-2}^{2} \int_{y^2}^{4} x \, dx \, dy$$

**Inner integral:**
$$\int_{y^2}^{4} x \, dx = \left[\frac{x^2}{2}\right]_{y^2}^{4} = 8 - \frac{y^4}{2}$$

**Outer integral:**
$$\int_{-2}^{2} \left(8 - \frac{y^4}{2}\right) dy = \left[8y - \frac{y^5}{10}\right]_{-2}^{2} = 2\left(16 - \frac{32}{10}\right) = \frac{128}{5}$$

---

### Example 3: Choosing the Best Order
Evaluate ∬_D sin(y²) dA where D is the triangle with vertices (0, 0), (1, 1), (0, 1).

**As Type I:** x from 0 to 1, y from x to 1
$$\int_0^1 \int_x^1 \sin(y^2) \, dy \, dx$$

This is hard because ∫sin(y²)dy has no elementary antiderivative!

**As Type II:** y from 0 to 1, x from 0 to y
$$\int_0^1 \int_0^y \sin(y^2) \, dx \, dy$$

**Inner integral:**
$$\int_0^y \sin(y^2) \, dx = x\sin(y^2)\Big|_0^y = y\sin(y^2)$$

**Outer integral:**
$$\int_0^1 y\sin(y^2) \, dy$$

Let u = y², du = 2y dy:
$$= \frac{1}{2}\int_0^1 \sin(u) \, du = \frac{1}{2}[-\cos(u)]_0^1 = \frac{1 - \cos(1)}{2}$$

---

### Example 4: Reversing Order of Integration

Reverse the order: $\int_0^1 \int_0^{\sqrt{x}} f(x, y) \, dy \, dx$

**Step 1:** Sketch the region.
- x goes from 0 to 1
- y goes from 0 to √x
- So y = √x means x = y²

**Step 2:** Describe as Type II.
- y goes from 0 to 1
- x goes from y² to 1

**Answer:** $\int_0^1 \int_{y^2}^1 f(x, y) \, dx \, dy$

---

### Example 5: Area Using Double Integrals
Find the area enclosed by y = x² and y = x + 2.

Intersection: x² = x + 2 → x² - x - 2 = 0 → x = -1, 2

$$\text{Area} = \iint_D 1 \, dA = \int_{-1}^{2} \int_{x^2}^{x+2} 1 \, dy \, dx$$

$$= \int_{-1}^{2} (x + 2 - x^2) \, dx = \left[\frac{x^2}{2} + 2x - \frac{x^3}{3}\right]_{-1}^{2}$$

$$= \left(2 + 4 - \frac{8}{3}\right) - \left(\frac{1}{2} - 2 + \frac{1}{3}\right) = 6 - \frac{8}{3} + \frac{7}{6} = \frac{9}{2}$$

---

## 📋 Strategy for Setting Up Double Integrals

### Step-by-Step Method

1. **Sketch the region D**
2. **Identify if Type I, Type II, or both**
3. **Determine outer limits** (constants)
4. **Determine inner limits** (functions of outer variable)
5. **Write the iterated integral**
6. **Evaluate inside-out**

### Choosing Order

- Choose the order that makes the inner integral easier
- Sometimes one order is impossible while the other works
- Check if reversing order simplifies computation

---

## 📝 Practice Problems

### Level 1: Type I Regions
1. ∬_D y dA, where D is bounded by y = x and y = x²
2. ∬_D (x + y) dA, where D is bounded by y = 0, y = √x, x = 4
3. ∬_D xy dA, where D is bounded by y = x, y = 0, x = 1

### Level 2: Type II Regions
4. ∬_D x² dA, where D is bounded by x = y, x = y² - 2
5. ∬_D y dA, where D is bounded by x = 0, x = √(4 - y²)
6. ∬_D 1 dA, where D is bounded by x = y² and x = 2 - y²

### Level 3: Choosing Order
7. Evaluate ∬_D e^(y²) dA where D is bounded by y = x, y = 1, x = 0.
8. Evaluate ∬_D x³cos(y³) dA where D is the triangle (0,0), (1,0), (1,1).

### Level 4: Reversing Order
9. Reverse: $\int_0^4 \int_0^{\sqrt{y}} f \, dx \, dy$
10. Reverse: $\int_0^1 \int_x^1 f \, dy \, dx$
11. Reverse and evaluate: $\int_0^1 \int_{\sqrt{y}}^1 e^{x^3} \, dx \, dy$

### Level 5: Applications
12. Find the area between y = x³ and y = x.
13. Find the volume under z = x + y over the region bounded by y = x² and y = 4.

---

## 📊 Answers

1. 1/12
2. 64/15
3. 1/8
4. 81/20
5. 8/3
6. 8/3
7. (e - 1)/2
8. sin(1)/12
9. $\int_0^2 \int_0^{x^2} f \, dy \, dx$
10. $\int_0^1 \int_0^y f \, dx \, dy$
11. (e - 1)/3
12. 1/4
13. 256/15

---

## 🔬 Quantum Mechanics Connection

### 2D Particle in a Box

For a 2D rectangular box with 0 ≤ x ≤ a, 0 ≤ y ≤ b:
$$\psi_{n,m}(x, y) = \frac{2}{\sqrt{ab}}\sin\left(\frac{n\pi x}{a}\right)\sin\left(\frac{m\pi y}{b}\right)$$

Normalization check:
$$\int_0^a \int_0^b |\psi_{n,m}|^2 \, dy \, dx = 1$$

---

## ✅ Daily Checklist

- [ ] Read Stewart 15.2
- [ ] Distinguish Type I and Type II regions
- [ ] Set up integrals with variable limits
- [ ] Practice reversing order of integration
- [ ] Complete practice problems

---

## 🔜 Preview: Tomorrow

**Day 45: Double Integrals in Polar Coordinates**
- Converting to polar coordinates
- dA = r dr dθ
- Circular and angular regions

---

*"The art of integration is the art of choosing the right perspective."*
