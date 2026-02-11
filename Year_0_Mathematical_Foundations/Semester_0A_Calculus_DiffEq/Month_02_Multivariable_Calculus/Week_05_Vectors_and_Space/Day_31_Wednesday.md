# Day 31: The Cross Product

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Cross Product Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute cross products using the determinant formula
2. Understand the geometric meaning of cross product
3. Apply the right-hand rule for direction
4. Find areas of parallelograms and triangles
5. Apply cross products to torque and angular momentum

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 12.4**: The Cross Product (pp. 816-824)

---

## 🎬 Video Resources

### 3Blue1Brown
**Essence of Linear Algebra: Cross products**
- Exceptional visual explanation

### MIT OpenCourseWare 18.02SC
**Lecture: Cross Product**

### Professor Leonard
**Calculus 3: The Cross Product**

---

## 📖 Core Content: The Cross Product

### 1. Definition

> **Definition:** The **cross product** (or **vector product**) of vectors **a** = ⟨a₁, a₂, a₃⟩ and **b** = ⟨b₁, b₂, b₃⟩ is:
> $$\mathbf{a} \times \mathbf{b} = \langle a_2b_3 - a_3b_2, \, a_3b_1 - a_1b_3, \, a_1b_2 - a_2b_1 \rangle$$

**Key point:** The cross product of two vectors is a **vector**, not a scalar!

### 2. Determinant Formula

The cross product can be computed using a 3×3 determinant:

$$\mathbf{a} \times \mathbf{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{vmatrix}$$

Expanding along the first row:
$$= \mathbf{i}\begin{vmatrix} a_2 & a_3 \\ b_2 & b_3 \end{vmatrix} - \mathbf{j}\begin{vmatrix} a_1 & a_3 \\ b_1 & b_3 \end{vmatrix} + \mathbf{k}\begin{vmatrix} a_1 & a_2 \\ b_1 & b_2 \end{vmatrix}$$

$$= \mathbf{i}(a_2b_3 - a_3b_2) - \mathbf{j}(a_1b_3 - a_3b_1) + \mathbf{k}(a_1b_2 - a_2b_1)$$

### 3. Geometric Interpretation

The cross product **a** × **b** is:
- **Perpendicular** to both **a** and **b**
- Has **magnitude** |**a** × **b**| = |**a**||**b**|sin θ
- **Direction** given by the right-hand rule

### 4. The Right-Hand Rule

To find the direction of **a** × **b**:
1. Point fingers in direction of **a**
2. Curl fingers toward **b** (through the smaller angle)
3. Thumb points in direction of **a** × **b**

---

## 📋 Properties of the Cross Product

1. **Anti-commutative:** $\mathbf{a} \times \mathbf{b} = -(\mathbf{b} \times \mathbf{a})$

2. **Distributive:** $\mathbf{a} \times (\mathbf{b} + \mathbf{c}) = \mathbf{a} \times \mathbf{b} + \mathbf{a} \times \mathbf{c}$

3. **Scalar multiplication:** $(c\mathbf{a}) \times \mathbf{b} = c(\mathbf{a} \times \mathbf{b}) = \mathbf{a} \times (c\mathbf{b})$

4. **Self cross product:** $\mathbf{a} \times \mathbf{a} = \mathbf{0}$

5. **Zero vector:** $\mathbf{a} \times \mathbf{0} = \mathbf{0}$

6. **NOT associative:** $\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) \neq (\mathbf{a} \times \mathbf{b}) \times \mathbf{c}$ in general

### Standard Basis Cross Products

$$\mathbf{i} \times \mathbf{j} = \mathbf{k}, \quad \mathbf{j} \times \mathbf{k} = \mathbf{i}, \quad \mathbf{k} \times \mathbf{i} = \mathbf{j}$$
$$\mathbf{j} \times \mathbf{i} = -\mathbf{k}, \quad \mathbf{k} \times \mathbf{j} = -\mathbf{i}, \quad \mathbf{i} \times \mathbf{k} = -\mathbf{j}$$

**Memory aid:** Cyclic order (i → j → k → i) gives positive; reverse gives negative.

---

## 📐 Geometric Applications

### Area of a Parallelogram

The parallelogram with adjacent sides **a** and **b** has area:
$$\text{Area} = |\mathbf{a} \times \mathbf{b}|$$

### Area of a Triangle

The triangle with sides **a** and **b** from one vertex has area:
$$\text{Area} = \frac{1}{2}|\mathbf{a} \times \mathbf{b}|$$

### Parallel Vectors

**a** and **b** are parallel if and only if:
$$\mathbf{a} \times \mathbf{b} = \mathbf{0}$$

---

## ✏️ Worked Examples

### Example 1: Computing a Cross Product
Find **a** × **b** where **a** = ⟨2, 3, 4⟩ and **b** = ⟨5, 6, 7⟩.

$$\mathbf{a} \times \mathbf{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ 2 & 3 & 4 \\ 5 & 6 & 7 \end{vmatrix}$$

$$= \mathbf{i}(3 \cdot 7 - 4 \cdot 6) - \mathbf{j}(2 \cdot 7 - 4 \cdot 5) + \mathbf{k}(2 \cdot 6 - 3 \cdot 5)$$
$$= \mathbf{i}(21 - 24) - \mathbf{j}(14 - 20) + \mathbf{k}(12 - 15)$$
$$= -3\mathbf{i} + 6\mathbf{j} - 3\mathbf{k} = \langle -3, 6, -3 \rangle$$

**Verify perpendicularity:**
- **a** · (**a** × **b**) = (2)(-3) + (3)(6) + (4)(-3) = -6 + 18 - 12 = 0 ✓
- **b** · (**a** × **b**) = (5)(-3) + (6)(6) + (7)(-3) = -15 + 36 - 21 = 0 ✓

---

### Example 2: Area of Parallelogram
Find the area of the parallelogram with vertices P(1, 1, 0), Q(2, 3, 1), R(4, 2, 2), S(3, 0, 1).

First, find vectors for adjacent sides:
$$\overrightarrow{PQ} = \langle 1, 2, 1 \rangle, \quad \overrightarrow{PS} = \langle 2, -1, 1 \rangle$$

$$\overrightarrow{PQ} \times \overrightarrow{PS} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ 1 & 2 & 1 \\ 2 & -1 & 1 \end{vmatrix}$$

$$= \mathbf{i}(2 + 1) - \mathbf{j}(1 - 2) + \mathbf{k}(-1 - 4) = \langle 3, 1, -5 \rangle$$

$$\text{Area} = |\langle 3, 1, -5 \rangle| = \sqrt{9 + 1 + 25} = \sqrt{35}$$

---

### Example 3: Area of Triangle
Find the area of the triangle with vertices A(1, 0, 0), B(0, 2, 0), C(0, 0, 3).

$$\overrightarrow{AB} = \langle -1, 2, 0 \rangle, \quad \overrightarrow{AC} = \langle -1, 0, 3 \rangle$$

$$\overrightarrow{AB} \times \overrightarrow{AC} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ -1 & 2 & 0 \\ -1 & 0 & 3 \end{vmatrix}$$

$$= \mathbf{i}(6 - 0) - \mathbf{j}(-3 - 0) + \mathbf{k}(0 + 2) = \langle 6, 3, 2 \rangle$$

$$\text{Area} = \frac{1}{2}|\langle 6, 3, 2 \rangle| = \frac{1}{2}\sqrt{36 + 9 + 4} = \frac{7}{2}$$

---

### Example 4: Finding a Perpendicular Vector
Find a vector perpendicular to both **a** = ⟨1, 2, 3⟩ and **b** = ⟨4, 5, 6⟩.

$$\mathbf{a} \times \mathbf{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ 1 & 2 & 3 \\ 4 & 5 & 6 \end{vmatrix}$$

$$= \mathbf{i}(12 - 15) - \mathbf{j}(6 - 12) + \mathbf{k}(5 - 8) = \langle -3, 6, -3 \rangle$$

Any scalar multiple of ⟨-3, 6, -3⟩ = -3⟨1, -2, 1⟩ is also perpendicular.

---

### Example 5: Torque
A force **F** = ⟨3, 2, 1⟩ N is applied at point P = (1, 1, 1) m from the origin. Find the torque about the origin.

Torque **τ** = **r** × **F** where **r** is the position vector.

$$\boldsymbol{\tau} = \langle 1, 1, 1 \rangle \times \langle 3, 2, 1 \rangle = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ 1 & 1 & 1 \\ 3 & 2 & 1 \end{vmatrix}$$

$$= \mathbf{i}(1 - 2) - \mathbf{j}(1 - 3) + \mathbf{k}(2 - 3) = \langle -1, 2, -1 \rangle \text{ N·m}$$

---

## 📝 Practice Problems

### Level 1: Basic Cross Products
1. Find ⟨1, 0, 0⟩ × ⟨0, 1, 0⟩
2. Find ⟨2, 1, 0⟩ × ⟨0, 3, 4⟩
3. Find ⟨1, 2, 3⟩ × ⟨1, 2, 3⟩

### Level 2: Verification
4. For **a** = ⟨1, 2, -1⟩ and **b** = ⟨3, 1, 2⟩, find **a** × **b** and verify it's perpendicular to both.
5. Show that ⟨2, 4, 6⟩ and ⟨1, 2, 3⟩ are parallel using the cross product.
6. Find **a** × **b** and **b** × **a** for **a** = ⟨1, 2, 3⟩, **b** = ⟨4, 5, 6⟩. Verify anti-commutativity.

### Level 3: Geometric Applications
7. Find the area of the parallelogram with adjacent sides ⟨3, 1, 2⟩ and ⟨1, 2, 3⟩.
8. Find the area of the triangle with vertices (0, 0, 0), (1, 2, 3), (2, 1, 0).
9. Find a unit vector perpendicular to both ⟨1, 1, 0⟩ and ⟨0, 1, 1⟩.

### Level 4: Applications
10. Find the torque about the origin when force **F** = ⟨0, 10, 0⟩ N acts at point (3, 0, 0) m.
11. A wrench handle is along ⟨0.3, 0, 0⟩ m. A force ⟨0, 50, 0⟩ N is applied. Find the torque magnitude.
12. Find the volume of the parallelepiped with edges **a** = ⟨1, 0, 0⟩, **b** = ⟨1, 1, 0⟩, **c** = ⟨1, 1, 1⟩ using |**a** · (**b** × **c**)|.

### Level 5: Proofs and Theory
13. Prove that |**a** × **b**|² + (**a** · **b**)² = |**a**|²|**b**|²
14. Prove that **a** × (**b** × **c**) = (**a** · **c**)**b** - (**a** · **b**)**c** (BAC-CAB rule)
15. Show that the area of triangle with vertices P₁, P₂, P₃ is ½|(**P₂** - **P₁**) × (**P₃** - **P₁**)|

---

## 📊 Answers

1. ⟨0, 0, 1⟩
2. ⟨4, -8, 6⟩
3. ⟨0, 0, 0⟩
4. ⟨5, -5, -5⟩; verify by dot products
5. Cross product = ⟨0, 0, 0⟩
6. **a** × **b** = ⟨-3, 6, -3⟩, **b** × **a** = ⟨3, -6, 3⟩
7. √83
8. ½√83
9. ±⟨1, -1, 1⟩/√3
10. ⟨0, 0, -30⟩ N·m
11. 15 N·m
12. 1 cubic unit
13. Use sin²θ + cos²θ = 1
14. Expand using components
15. Direct application of triangle area formula

---

## 🔬 Quantum Mechanics Connection

### Angular Momentum

In quantum mechanics, angular momentum is defined as:
$$\mathbf{L} = \mathbf{r} \times \mathbf{p}$$

where **r** is position and **p** is momentum.

The quantum angular momentum operators satisfy:
$$[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z$$

This **commutation relation** is directly related to the cross product structure!

### Spin

Electron spin operators satisfy similar cross-product-like relations:
$$\mathbf{S} \times \mathbf{S} = i\hbar\mathbf{S}$$

---

## ✅ Daily Checklist

- [ ] Read Stewart 12.4
- [ ] Watch 3Blue1Brown cross product video
- [ ] Master the determinant computation
- [ ] Understand geometric meaning (perpendicular, area)
- [ ] Apply right-hand rule
- [ ] Calculate areas of parallelograms and triangles
- [ ] Understand torque applications
- [ ] Complete practice problems

---

## 📓 Reflection Questions

1. Why is the cross product anti-commutative?
2. What does it mean geometrically that **a** × **a** = **0**?
3. Why does the cross product only work in 3D?
4. How is torque related to the cross product?

---

## 🔜 Preview: Tomorrow

**Day 32: Lines and Planes in Space**
- Parametric equations of lines
- Vector and scalar equations of planes
- Distances between points, lines, and planes

---

*"The cross product creates a new dimension of understanding—literally perpendicular to what we knew before."*
