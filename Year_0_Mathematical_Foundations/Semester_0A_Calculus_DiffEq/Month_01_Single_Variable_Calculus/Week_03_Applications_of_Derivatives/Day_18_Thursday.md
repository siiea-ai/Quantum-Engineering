# Day 18: Optimization Problems

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Problem Setup Techniques |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Solving Optimization |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Translate word problems into mathematical optimization problems
2. Identify the objective function and constraints
3. Reduce optimization to single-variable problems
4. Solve classic optimization problems (fencing, boxes, distance)
5. Verify solutions and interpret results

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 4.7**: Optimization Problems (pp. 328-340)

---

## 📖 Core Content: Optimization Strategy

### The Problem-Solving Framework

> **Step 1: Understand the Problem**
> - Read carefully, identify what's being maximized/minimized
> - Draw a diagram
> - Assign variables to all quantities

> **Step 2: Write the Objective Function**
> - Express the quantity to optimize in terms of variables
> - This is the function you'll differentiate

> **Step 3: Write the Constraints**
> - Identify relationships that limit the variables
> - Use constraints to reduce to ONE variable

> **Step 4: Find the Domain**
> - Determine valid range for your single variable
> - Consider physical constraints (lengths > 0, etc.)

> **Step 5: Optimize**
> - Find critical points using f'(x) = 0
> - Test endpoints and critical points
> - Use first/second derivative test if needed

> **Step 6: Verify and Interpret**
> - Check answer makes sense
> - Answer the original question with units

---

## ✏️ Worked Examples

### Example 1: The Classic Fence Problem

**Problem:** A farmer has 400 meters of fencing. What is the maximum area that can be enclosed in a rectangular field?

**Solution:**

**Step 1:** Draw diagram, assign variables
- Let x = width, y = length
- Maximize: Area

**Step 2:** Objective function
$$A = xy$$

**Step 3:** Constraint
$$2x + 2y = 400 \implies y = 200 - x$$

**Step 4:** Substitute to get single variable
$$A(x) = x(200 - x) = 200x - x^2$$

**Step 5:** Domain: 0 ≤ x ≤ 200

**Step 6:** Find critical points
$$A'(x) = 200 - 2x = 0 \implies x = 100$$

**Step 7:** Verify it's a maximum
$$A''(x) = -2 < 0$$ → concave down → maximum

**Step 8:** Find dimensions
x = 100m, y = 200 - 100 = 100m

**Answer:** Maximum area is **10,000 m²** with a **100m × 100m square**.

---

### Example 2: Three-Sided Fence (Against a Wall)

**Problem:** A farmer has 200m of fencing to create a rectangular pen against a barn wall (no fence needed on wall side). Find maximum area.

**Solution:**

Constraint: x + 2y = 200 (x is along wall)
So: x = 200 - 2y

Objective: A = xy = (200 - 2y)y = 200y - 2y²

A'(y) = 200 - 4y = 0 → y = 50

x = 200 - 100 = 100

**Answer:** Maximum area is **5,000 m²** (100m × 50m).

---

### Example 3: Box with Maximum Volume

**Problem:** A box with an open top is made from a 12" × 12" square piece of cardboard by cutting squares from corners and folding. What size squares maximize volume?

**Solution:**

**Step 1:** Let x = side of cut squares (height of box)
After folding: base is (12-2x) × (12-2x), height is x

**Step 2:** Objective
$$V = x(12-2x)^2$$

**Step 3:** Domain: 0 < x < 6

**Step 4:** Expand and differentiate
$$V = x(144 - 48x + 4x^2) = 144x - 48x^2 + 4x^3$$
$$V' = 144 - 96x + 12x^2 = 12(12 - 8x + x^2) = 12(x-2)(x-6)$$

Critical points: x = 2 or x = 6

**Step 5:** x = 6 is endpoint (gives V = 0), so x = 2 is our candidate

V(2) = 2(8)² = 128 in³

**Answer:** Cut 2" squares from corners for maximum volume of **128 cubic inches**.

---

### Example 4: Minimum Distance

**Problem:** Find the point on the line y = 2x + 1 closest to the point (3, 0).

**Solution:**

**Step 1:** Point on line: (x, 2x + 1)

**Step 2:** Distance to (3, 0):
$$D = \sqrt{(x-3)^2 + (2x+1)^2}$$

**Tip:** Minimize D² instead (same critical points, easier math)!

$$D^2 = (x-3)^2 + (2x+1)^2 = x^2 - 6x + 9 + 4x^2 + 4x + 1 = 5x^2 - 2x + 10$$

**Step 3:** Differentiate
$$\frac{d(D^2)}{dx} = 10x - 2 = 0 \implies x = \frac{1}{5}$$

**Step 4:** Find y
$$y = 2(1/5) + 1 = 7/5$$

**Answer:** Closest point is **(1/5, 7/5)**.

---

### Example 5: Minimum Material for a Can

**Problem:** A cylindrical can must hold 1000 cm³. Find dimensions that minimize surface area (material).

**Solution:**

**Step 1:** Variables: radius r, height h

**Step 2:** Constraint (volume)
$$V = \pi r^2 h = 1000 \implies h = \frac{1000}{\pi r^2}$$

**Step 3:** Objective (surface area: top + bottom + side)
$$S = 2\pi r^2 + 2\pi r h = 2\pi r^2 + 2\pi r \cdot \frac{1000}{\pi r^2}$$
$$S = 2\pi r^2 + \frac{2000}{r}$$

**Step 4:** Domain: r > 0

**Step 5:** Differentiate
$$S' = 4\pi r - \frac{2000}{r^2} = 0$$
$$4\pi r^3 = 2000$$
$$r^3 = \frac{500}{\pi}$$
$$r = \sqrt[3]{\frac{500}{\pi}} \approx 5.42 \text{ cm}$$

**Step 6:** Find height
$$h = \frac{1000}{\pi r^2} = \frac{1000}{\pi \cdot (500/\pi)^{2/3}} = \sqrt[3]{\frac{1000 \cdot \pi^2}{500^2 \cdot \pi^2}} \cdot 1000^{1/3}$$

After simplification: h = 2r

**Answer:** r ≈ 5.42 cm, h ≈ 10.84 cm (height = diameter).

---

### Example 6: Maximum Revenue

**Problem:** A company sells x units at price p = 100 - 0.5x dollars. What production level maximizes revenue?

**Solution:**

Revenue: R = xp = x(100 - 0.5x) = 100x - 0.5x²

R' = 100 - x = 0 → x = 100

R(100) = 100(100) - 0.5(10000) = 10000 - 5000 = 5000

**Answer:** Maximum revenue of **$5,000** at **100 units**.

---

## 📝 Practice Problems

### Level 1: Basic Geometry
1. Find two positive numbers whose sum is 100 and whose product is maximum.
2. Find two positive numbers whose product is 100 and whose sum is minimum.
3. A rectangle has perimeter 40. Find dimensions for maximum area.

### Level 2: Fencing Problems
4. 600m of fencing encloses a rectangular field divided into two pens by a fence parallel to one side. Find dimensions for maximum total area.
5. A field borders a river (no fence needed on river side). With 1000m of fencing, find maximum area.

### Level 3: Box/Container Problems
6. An open-top box is made from 24" × 24" cardboard. Find cut size for maximum volume.
7. A closed box with square base must have volume 1000 cm³. Find dimensions for minimum surface area.

### Level 4: Distance Problems
8. Find the point on y = x² closest to (0, 1).
9. Find the point on y = √x closest to (4, 0).

### Level 5: Applied
10. A poster must have 50 cm² of printed area with 4 cm margins at top/bottom and 2 cm on sides. Find dimensions for minimum total area.
11. A Norman window (rectangle topped by semicircle) has perimeter 10m. Find dimensions for maximum area.
12. A company's profit is P(x) = 100x - x² - 25 (x = thousands of units). Find maximum profit.

---

## 📊 Answers

1. 50 and 50
2. 10 and 10
3. 10 × 10 (square)
4. 150m × 100m (300m parallel to division)
5. 500m × 250m (500m along river)
6. Cut 4" squares, V = 1024 in³
7. 10 × 10 × 10 cm (cube)
8. (0, 1/2)
9. (7/2, √(7/2))
10. Width = 9 cm, Height = 14 cm
11. r = 10/(4+π) ≈ 1.4m for semicircle
12. Maximum profit of $2475 at x = 50 (50,000 units)

---

## 🔬 Physics Application: Minimum Energy Configurations

In physics, stable equilibria occur at energy minima:
- Soap films minimize surface area
- Orbits minimize action
- Molecules arrange to minimize potential energy

The condition dE/dx = 0 gives equilibrium configurations.

---

## ✅ Daily Checklist

- [ ] Read Stewart 4.7
- [ ] Master the 6-step optimization strategy
- [ ] Complete Examples 1-6 independently
- [ ] Solve Level 1-3 problems
- [ ] Attempt Level 4-5 problems
- [ ] Always verify answer is max/min (not the other)
- [ ] Include units in final answers

---

## 🔜 Preview: Tomorrow

**Day 19: Week 3 Problem Set**
- Comprehensive practice on applications
- Related rates, linear approximation, optimization combined

---

*"Optimization is the art of making the best decision under constraints."*
