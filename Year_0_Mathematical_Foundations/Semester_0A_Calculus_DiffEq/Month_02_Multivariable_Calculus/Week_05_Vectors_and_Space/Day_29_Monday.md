# Day 29: Vectors in Two and Three Dimensions

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Vector Fundamentals |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | 3D Coordinate Systems |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand vectors as directed quantities
2. Perform vector operations (addition, scalar multiplication)
3. Compute vector magnitudes and unit vectors
4. Work in 3D coordinate systems
5. Express vectors in component form

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 12.1**: Three-Dimensional Coordinate Systems (pp. 790-796)
- **Section 12.2**: Vectors (pp. 797-806)

### Supplementary
- 3Blue1Brown "Essence of Linear Algebra" (Chapters 1-2)

---

## 🎬 Video Resources

### 3Blue1Brown
**Essence of Linear Algebra: Vectors**
- Outstanding visual intuition

### MIT OpenCourseWare 18.02SC
**Lecture 1: Vectors**

### Professor Leonard
**Calculus 3: Vectors in 2D and 3D**

---

## 📖 Core Content: Introduction to Vectors

### 1. What is a Vector?

A **vector** is a quantity that has both **magnitude** (size) and **direction**.

**Examples of vectors:**
- Velocity (speed + direction)
- Force (magnitude + direction)
- Displacement (distance + direction)
- Electric field

**Examples of scalars (not vectors):**
- Temperature
- Mass
- Speed (magnitude only)
- Energy

### 2. Geometric Representation

A vector is represented as an arrow:
- **Length** = magnitude
- **Direction** = where it points
- **Initial point** = tail
- **Terminal point** = head

Two vectors are **equal** if they have the same magnitude and direction, regardless of position.

### 3. Component Form

In 2D, a vector **v** can be written as:
$$\mathbf{v} = \langle v_1, v_2 \rangle = v_1\mathbf{i} + v_2\mathbf{j}$$

In 3D:
$$\mathbf{v} = \langle v_1, v_2, v_3 \rangle = v_1\mathbf{i} + v_2\mathbf{j} + v_3\mathbf{k}$$

where **i**, **j**, **k** are the **standard basis vectors**:
- **i** = ⟨1, 0, 0⟩ (points along positive x-axis)
- **j** = ⟨0, 1, 0⟩ (points along positive y-axis)
- **k** = ⟨0, 0, 1⟩ (points along positive z-axis)

### 4. Position Vectors

If P = (x, y, z) is a point, the **position vector** of P is:
$$\overrightarrow{OP} = \langle x, y, z \rangle$$

The vector from point A = (a₁, a₂, a₃) to point B = (b₁, b₂, b₃) is:
$$\overrightarrow{AB} = \langle b_1 - a_1, b_2 - a_2, b_3 - a_3 \rangle$$

---

## 📋 Vector Operations

### 5. Vector Addition

**Geometric:** Place vectors head-to-tail; sum is from first tail to last head.

**Algebraic:**
$$\mathbf{u} + \mathbf{v} = \langle u_1 + v_1, u_2 + v_2, u_3 + v_3 \rangle$$

**Example:** ⟨2, 3, 1⟩ + ⟨1, -2, 4⟩ = ⟨3, 1, 5⟩

### 6. Scalar Multiplication

**Geometric:** Scales length, may reverse direction if negative.

**Algebraic:**
$$c\mathbf{v} = \langle cv_1, cv_2, cv_3 \rangle$$

**Example:** 3⟨2, -1, 4⟩ = ⟨6, -3, 12⟩

### 7. Vector Subtraction

$$\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v}) = \langle u_1 - v_1, u_2 - v_2, u_3 - v_3 \rangle$$

---

## 📐 Magnitude and Direction

### 8. Magnitude (Length)

The **magnitude** (or **norm**) of **v** = ⟨v₁, v₂, v₃⟩ is:
$$|\mathbf{v}| = \|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + v_3^2}$$

This is the 3D distance formula!

**Example:** |⟨3, 4, 0⟩| = √(9 + 16 + 0) = 5

### 9. Unit Vectors

A **unit vector** has magnitude 1.

To find the unit vector in the direction of **v**:
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{|\mathbf{v}|}$$

**Example:** Find the unit vector in the direction of ⟨3, 4, 0⟩.
$$\hat{\mathbf{v}} = \frac{\langle 3, 4, 0 \rangle}{5} = \left\langle \frac{3}{5}, \frac{4}{5}, 0 \right\rangle$$

Check: |⟨3/5, 4/5, 0⟩| = √(9/25 + 16/25) = √(25/25) = 1 ✓

### 10. Direction Angles and Cosines

The **direction angles** α, β, γ are the angles **v** makes with the positive x, y, z axes.

**Direction cosines:**
$$\cos\alpha = \frac{v_1}{|\mathbf{v}|}, \quad \cos\beta = \frac{v_2}{|\mathbf{v}|}, \quad \cos\gamma = \frac{v_3}{|\mathbf{v}|}$$

**Important identity:**
$$\cos^2\alpha + \cos^2\beta + \cos^2\gamma = 1$$

---

## 📐 3D Coordinate System

### 11. The Right-Hand Rule

The standard 3D coordinate system is **right-handed**:
- Curl fingers from +x toward +y
- Thumb points in +z direction

### 12. Octants

The three coordinate planes divide space into 8 **octants**.

The first octant has x > 0, y > 0, z > 0.

### 13. Distance in 3D

Distance between P₁ = (x₁, y₁, z₁) and P₂ = (x₂, y₂, z₂):
$$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2 + (z_2-z_1)^2}$$

### 14. Spheres

A sphere with center (a, b, c) and radius r:
$$(x-a)^2 + (y-b)^2 + (z-c)^2 = r^2$$

---

## ✏️ Worked Examples

### Example 1: Vector Between Points
Find the vector from A = (1, 2, 3) to B = (4, -1, 5).

$$\overrightarrow{AB} = \langle 4-1, -1-2, 5-3 \rangle = \langle 3, -3, 2 \rangle$$

---

### Example 2: Vector Operations
Let **u** = ⟨2, -1, 3⟩ and **v** = ⟨1, 4, -2⟩. Find 2**u** - 3**v**.

$$2\mathbf{u} - 3\mathbf{v} = 2\langle 2, -1, 3 \rangle - 3\langle 1, 4, -2 \rangle$$
$$= \langle 4, -2, 6 \rangle - \langle 3, 12, -6 \rangle = \langle 1, -14, 12 \rangle$$

---

### Example 3: Magnitude and Unit Vector
Find |**v**| and the unit vector for **v** = ⟨1, 2, 2⟩.

$$|\mathbf{v}| = \sqrt{1 + 4 + 4} = \sqrt{9} = 3$$

$$\hat{\mathbf{v}} = \frac{\langle 1, 2, 2 \rangle}{3} = \left\langle \frac{1}{3}, \frac{2}{3}, \frac{2}{3} \right\rangle$$

---

### Example 4: Direction Angles
Find the direction angles of **v** = ⟨2, 2, 1⟩.

$$|\mathbf{v}| = \sqrt{4 + 4 + 1} = 3$$

$$\cos\alpha = \frac{2}{3}, \quad \cos\beta = \frac{2}{3}, \quad \cos\gamma = \frac{1}{3}$$

$$\alpha = \arccos(2/3) \approx 48.2°$$
$$\beta = \arccos(2/3) \approx 48.2°$$
$$\gamma = \arccos(1/3) \approx 70.5°$$

---

### Example 5: Sphere Equation
Find the equation of the sphere with center (2, -1, 3) and radius 4.

$$(x-2)^2 + (y+1)^2 + (z-3)^2 = 16$$

---

## 📝 Practice Problems

### Level 1: Basic Operations
1. Let **a** = ⟨3, -2, 5⟩ and **b** = ⟨-1, 4, 2⟩. Find **a** + **b**.
2. Find 4**a** - 2**b** for the vectors above.
3. Find the vector from P(1, 0, -2) to Q(3, 5, 1).

### Level 2: Magnitude and Unit Vectors
4. Find |⟨4, -3, 0⟩|.
5. Find the unit vector in the direction of ⟨6, 2, 3⟩.
6. Find a vector of length 5 in the direction of ⟨1, 1, 1⟩.

### Level 3: Direction Angles
7. Find the direction cosines of ⟨3, 4, 0⟩.
8. Find the direction angles of ⟨1, 1, √2⟩.
9. If a vector makes equal angles with all three axes, what are those angles?

### Level 4: 3D Geometry
10. Find the distance between (1, 2, 3) and (4, -2, 1).
11. Find the equation of the sphere with center (0, 1, -2) passing through (3, 1, 2).
12. Determine if the point (1, 2, 3) lies inside, on, or outside the sphere x² + y² + z² = 16.

### Level 5: Applications
13. A force **F** = ⟨10, 20, -15⟩ N acts on an object. Find the magnitude of the force.
14. Find the midpoint of the segment from A(2, 4, 6) to B(8, 2, 4).
15. If **u** + **v** = ⟨5, 1, 3⟩ and **u** - **v** = ⟨1, 3, -1⟩, find **u** and **v**.

---

## 📊 Answers

1. ⟨2, 2, 7⟩
2. ⟨14, -16, 16⟩
3. ⟨2, 5, 3⟩
4. 5
5. ⟨6/7, 2/7, 3/7⟩
6. (5/√3)⟨1, 1, 1⟩ = ⟨5√3/3, 5√3/3, 5√3/3⟩
7. cos α = 3/5, cos β = 4/5, cos γ = 0
8. α = β = 60°, γ = 45°
9. α = β = γ = arccos(1/√3) ≈ 54.7°
10. √29
11. x² + (y-1)² + (z+2)² = 25
12. Inside (1 + 4 + 9 = 14 < 16)
13. 5√29 ≈ 26.9 N
14. (5, 3, 5)
15. **u** = ⟨3, 2, 1⟩, **v** = ⟨2, -1, 2⟩

---

## 🔬 Quantum Mechanics Connection

### State Vectors

In quantum mechanics, the state of a system is represented by a **state vector** |ψ⟩ in a complex vector space called **Hilbert space**.

For a two-level system (qubit):
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where |α|² + |β|² = 1 (normalization).

### Dirac Notation

- **Ket**: |ψ⟩ (column vector)
- **Bra**: ⟨ψ| (row vector, conjugate transpose)
- **Inner product**: ⟨φ|ψ⟩

The vector concepts you're learning today extend directly to quantum mechanics!

---

## ✅ Daily Checklist

- [ ] Read Stewart 12.1-12.2
- [ ] Watch 3Blue1Brown vectors video
- [ ] Master component form notation
- [ ] Practice vector addition and scalar multiplication
- [ ] Compute magnitudes and unit vectors
- [ ] Understand 3D coordinate systems
- [ ] Complete practice problems

---

## 📓 Reflection Questions

1. How is a vector different from a point?
2. Why is the right-hand rule important?
3. What's the geometric meaning of vector addition?
4. How do vectors in physics relate to vectors in math?

---

## 🔜 Preview: Tomorrow

**Day 30: The Dot Product**
- Algebraic and geometric definitions
- Angle between vectors
- Projections
- Work as a dot product

---

*"Vectors are the language of physics—they let us describe the world's geometry."*
