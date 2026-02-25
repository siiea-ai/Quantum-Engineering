# Day 32: Lines and Planes in Space

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Lines in 3D |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Planes in 3D |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Write parametric and symmetric equations of lines
2. Find equations of planes from points and normal vectors
3. Determine if lines are parallel, intersecting, or skew
4. Calculate distances between points, lines, and planes
5. Find angles between planes

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 12.5**: Equations of Lines and Planes (pp. 825-835)

---

## 🎬 Video Resources

### MIT OpenCourseWare 18.02SC
**Lecture: Equations of Lines and Planes**

### Professor Leonard
**Calculus 3: Lines and Planes in 3D Space**

---

## 📖 Part I: Lines in Three Dimensions

### 1. Vector Equation of a Line

A line through point P₀ = (x₀, y₀, z₀) parallel to direction vector **v** = ⟨a, b, c⟩:

$$\mathbf{r}(t) = \mathbf{r}_0 + t\mathbf{v}$$

$$\langle x, y, z \rangle = \langle x_0, y_0, z_0 \rangle + t\langle a, b, c \rangle$$

### 2. Parametric Equations

$$x = x_0 + at, \quad y = y_0 + bt, \quad z = z_0 + ct$$

where t ∈ ℝ is the parameter.

### 3. Symmetric Equations

Solving each parametric equation for t and setting equal:

$$\frac{x - x_0}{a} = \frac{y - y_0}{b} = \frac{z - z_0}{c}$$

(assuming a, b, c ≠ 0)

### 4. Line Through Two Points

For a line through P₁ = (x₁, y₁, z₁) and P₂ = (x₂, y₂, z₂):

Direction vector: **v** = **P₂** - **P₁** = ⟨x₂ - x₁, y₂ - y₁, z₂ - z₁⟩

Parametric form:
$$x = x_1 + t(x_2 - x_1), \quad y = y_1 + t(y_2 - y_1), \quad z = z_1 + t(z_2 - z_1)$$

Note: t = 0 gives P₁, t = 1 gives P₂.

---

## ✏️ Line Examples

### Example 1: Line from Point and Direction
Find parametric equations for the line through (2, 4, -1) parallel to ⟨3, 5, -2⟩.

$$x = 2 + 3t, \quad y = 4 + 5t, \quad z = -1 - 2t$$

Symmetric form:
$$\frac{x - 2}{3} = \frac{y - 4}{5} = \frac{z + 1}{-2}$$

---

### Example 2: Line Through Two Points
Find equations for the line through (1, 2, 3) and (4, 6, 5).

Direction: ⟨4-1, 6-2, 5-3⟩ = ⟨3, 4, 2⟩

Parametric:
$$x = 1 + 3t, \quad y = 2 + 4t, \quad z = 3 + 2t$$

---

### Example 3: Parallel, Intersecting, or Skew?
Determine the relationship between:
- L₁: x = 1 + t, y = 2 + 2t, z = 3 + 3t
- L₂: x = 2 + s, y = 4 + 2s, z = 1 + 3s

Direction vectors: **v₁** = ⟨1, 2, 3⟩, **v₂** = ⟨1, 2, 3⟩

Since **v₁** = **v₂**, lines are parallel.

Check if same line: Point (1, 2, 3) on L₂? 2 + s = 1 → s = -1, then y = 4 - 2 = 2 ✓, z = 1 - 3 = -2 ≠ 3 ✗

Lines are **parallel but distinct**.

---

## 📖 Part II: Planes in Three Dimensions

### 5. Normal Vector

A plane is determined by:
- A point P₀ = (x₀, y₀, z₀) on the plane
- A **normal vector** **n** = ⟨a, b, c⟩ perpendicular to the plane

### 6. Vector Equation of a Plane

For any point P = (x, y, z) on the plane:
$$\mathbf{n} \cdot (\mathbf{r} - \mathbf{r}_0) = 0$$

or equivalently:
$$\mathbf{n} \cdot \mathbf{r} = \mathbf{n} \cdot \mathbf{r}_0$$

### 7. Scalar Equation of a Plane

$$a(x - x_0) + b(y - y_0) + c(z - z_0) = 0$$

Expanding:
$$ax + by + cz = d$$

where d = ax₀ + by₀ + cz₀.

### 8. Finding Normal Vectors

**From equation:** In ax + by + cz = d, the normal is **n** = ⟨a, b, c⟩.

**From three points:** If P, Q, R are on the plane:
$$\mathbf{n} = \overrightarrow{PQ} \times \overrightarrow{PR}$$

---

## ✏️ Plane Examples

### Example 4: Plane from Point and Normal
Find the equation of the plane through (2, 3, -1) with normal ⟨4, -2, 5⟩.

$$4(x - 2) - 2(y - 3) + 5(z + 1) = 0$$
$$4x - 8 - 2y + 6 + 5z + 5 = 0$$
$$4x - 2y + 5z = -3$$

---

### Example 5: Plane Through Three Points
Find the plane through A(1, 0, 0), B(0, 2, 0), C(0, 0, 3).

$$\overrightarrow{AB} = \langle -1, 2, 0 \rangle, \quad \overrightarrow{AC} = \langle -1, 0, 3 \rangle$$

$$\mathbf{n} = \overrightarrow{AB} \times \overrightarrow{AC} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ -1 & 2 & 0 \\ -1 & 0 & 3 \end{vmatrix} = \langle 6, 3, 2 \rangle$$

Using point A(1, 0, 0):
$$6(x - 1) + 3y + 2z = 0$$
$$6x + 3y + 2z = 6$$

---

### Example 6: Angle Between Planes
Find the angle between 2x + y - z = 1 and x - y + z = 2.

Normal vectors: **n₁** = ⟨2, 1, -1⟩, **n₂** = ⟨1, -1, 1⟩

$$\cos\theta = \frac{|\mathbf{n}_1 \cdot \mathbf{n}_2|}{|\mathbf{n}_1||\mathbf{n}_2|} = \frac{|2 - 1 - 1|}{\sqrt{6}\sqrt{3}} = \frac{0}{\sqrt{18}} = 0$$

θ = 90° — the planes are perpendicular!

---

## 📐 Distances

### Distance from Point to Plane

The distance from point P₁ = (x₁, y₁, z₁) to plane ax + by + cz + d = 0:

$$D = \frac{|ax_1 + by_1 + cz_1 + d|}{\sqrt{a^2 + b^2 + c^2}}$$

### Distance from Point to Line

The distance from point P to line through P₀ with direction **v**:

$$D = \frac{|\overrightarrow{P_0P} \times \mathbf{v}|}{|\mathbf{v}|}$$

---

### Example 7: Point-to-Plane Distance
Find the distance from (1, 2, 3) to the plane 2x - 2y + z = 5.

Rewrite as 2x - 2y + z - 5 = 0.

$$D = \frac{|2(1) - 2(2) + 1(3) - 5|}{\sqrt{4 + 4 + 1}} = \frac{|2 - 4 + 3 - 5|}{3} = \frac{|-4|}{3} = \frac{4}{3}$$

---

### Example 8: Point-to-Line Distance
Find the distance from P(1, 1, 1) to the line through (0, 0, 0) with direction ⟨1, 2, 2⟩.

$$\overrightarrow{P_0P} = \langle 1, 1, 1 \rangle$$

$$\overrightarrow{P_0P} \times \mathbf{v} = \langle 1, 1, 1 \rangle \times \langle 1, 2, 2 \rangle = \langle 0, -1, 1 \rangle$$

$$D = \frac{|\langle 0, -1, 1 \rangle|}{|\langle 1, 2, 2 \rangle|} = \frac{\sqrt{2}}{3}$$

---

## 📝 Practice Problems

### Level 1: Lines
1. Find parametric equations for the line through (1, -2, 3) with direction ⟨4, 5, -1⟩.
2. Find symmetric equations for the line through (2, 1, 0) and (5, -1, 2).
3. Find the point where the line x = 1 + t, y = 2t, z = 3 - t intersects the xy-plane.

### Level 2: Planes
4. Find the equation of the plane through (1, 2, 3) with normal ⟨2, -1, 4⟩.
5. Find the equation of the plane through (1, 0, 0), (0, 1, 0), (0, 0, 1).
6. Find where the line x = t, y = 1 + t, z = 2t intersects the plane x + y + z = 6.

### Level 3: Angles and Relationships
7. Find the angle between planes x + y = 1 and y + z = 1.
8. Are the planes 2x - y + z = 1 and 4x - 2y + 2z = 3 parallel?
9. Find the line of intersection of planes x + y + z = 1 and x - y + z = 0.

### Level 4: Distances
10. Find the distance from (2, -1, 3) to the plane 3x + 4y - 12z = 5.
11. Find the distance from (1, 0, 0) to the line x = t, y = t, z = t.
12. Find the distance between parallel planes 2x - y + z = 1 and 2x - y + z = 4.

### Level 5: Applications
13. A light ray travels along the line **r**(t) = ⟨1, 2, 3⟩ + t⟨1, 1, -1⟩. Where does it hit the plane z = 0?
14. Find the equation of the plane containing the line x = 1 + t, y = 2t, z = 3 - t and the point (0, 1, 2).
15. Find the shortest distance between skew lines L₁: **r** = ⟨1, 0, 0⟩ + t⟨1, 1, 0⟩ and L₂: **r** = ⟨0, 1, 1⟩ + s⟨0, 1, 1⟩.

---

## 📊 Answers

1. x = 1 + 4t, y = -2 + 5t, z = 3 - t
2. (x-2)/3 = (y-1)/(-2) = z/2
3. (4, 6, 0) when t = 3
4. 2x - y + 4z = 12
5. x + y + z = 1
6. (1, 2, 2) when t = 1
7. 60°
8. Yes (normal vectors are parallel)
9. x = ½, y = ½ - t, z = t (or equivalent)
10. 4/13
11. √(2/3)
12. 3/√6
13. (4, 5, 0)
14. 2x + y + z = 3
15. 1/√2

---

## 🔬 Quantum Mechanics Connection

### Configuration Space

In quantum mechanics of N particles, the state space has dimension 3N. Each particle's position is described by 3 coordinates, forming a point in 3N-dimensional configuration space.

### Planes in Hilbert Space

Constraints on quantum states (like symmetry or normalization) define hyperplanes in Hilbert space:

$$\langle\psi|\phi\rangle = 0 \quad \text{(orthogonality)}$$

is analogous to **n** · **r** = 0 defining a plane through the origin!

---

## ✅ Daily Checklist

- [ ] Read Stewart 12.5
- [ ] Master parametric and symmetric equations of lines
- [ ] Write plane equations from point and normal
- [ ] Find plane through three points
- [ ] Calculate distances (point-to-plane, point-to-line)
- [ ] Determine angle between planes
- [ ] Complete practice problems

---

## 📓 Reflection Questions

1. What information do you need to uniquely determine a line? A plane?
2. Why is the normal vector so important for describing planes?
3. How is the distance formula related to projection?
4. What does it mean geometrically for planes to be parallel?

---

## 🔜 Preview: Tomorrow

**Day 33: Vector Functions and Space Curves**
- Vector-valued functions
- Curves in space
- Derivatives and integrals of vector functions
- Arc length

---

*"Lines and planes are the building blocks of geometry—master them and you can construct any shape."*
