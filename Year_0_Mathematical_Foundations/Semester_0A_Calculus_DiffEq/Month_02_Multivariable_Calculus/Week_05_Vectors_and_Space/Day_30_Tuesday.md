# Day 30: The Dot Product

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Dot Product Theory |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Practice |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute dot products algebraically
2. Understand the geometric meaning of dot product
3. Find angles between vectors
4. Compute vector projections
5. Apply dot products to physics problems (work)

---

## 📚 Required Reading

### Primary Text: Stewart's Calculus (8th Edition)
- **Section 12.3**: The Dot Product (pp. 807-815)

---

## 🎬 Video Resources

### 3Blue1Brown
**Essence of Linear Algebra: Dot products and duality**
- Beautiful geometric intuition

### MIT OpenCourseWare 18.02SC
**Lecture: Dot Product**

### Professor Leonard
**Calculus 3: The Dot Product**

---

## 📖 Core Content: The Dot Product

### 1. Algebraic Definition

> **Definition:** The **dot product** (or **scalar product**) of two vectors **a** = ⟨a₁, a₂, a₃⟩ and **b** = ⟨b₁, b₂, b₃⟩ is:
> $$\mathbf{a} \cdot \mathbf{b} = a_1b_1 + a_2b_2 + a_3b_3$$

**Key point:** The dot product of two vectors is a **scalar** (number), not a vector!

**Example:**
$$\langle 2, 3, -1 \rangle \cdot \langle 4, -2, 5 \rangle = (2)(4) + (3)(-2) + (-1)(5) = 8 - 6 - 5 = -3$$

### 2. Geometric Definition

> **Theorem:** If θ is the angle between vectors **a** and **b** (0 ≤ θ ≤ π), then:
> $$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos\theta$$

This connects algebra to geometry!

### 3. Equivalence of Definitions

**Proof sketch using Law of Cosines:**

For vectors **a** and **b** with angle θ between them:
$$|\mathbf{a} - \mathbf{b}|^2 = |\mathbf{a}|^2 + |\mathbf{b}|^2 - 2|\mathbf{a}||\mathbf{b}|\cos\theta$$

Expanding the left side algebraically:
$$|\mathbf{a} - \mathbf{b}|^2 = (\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b}) = |\mathbf{a}|^2 - 2\mathbf{a} \cdot \mathbf{b} + |\mathbf{b}|^2$$

Comparing: $\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos\theta$ ∎

---

## 📋 Properties of the Dot Product

For vectors **a**, **b**, **c** and scalar k:

1. **Commutative:** $\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}$

2. **Distributive:** $\mathbf{a} \cdot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c}$

3. **Scalar multiplication:** $(k\mathbf{a}) \cdot \mathbf{b} = k(\mathbf{a} \cdot \mathbf{b}) = \mathbf{a} \cdot (k\mathbf{b})$

4. **Self dot product:** $\mathbf{a} \cdot \mathbf{a} = |\mathbf{a}|^2$

5. **Zero vector:** $\mathbf{0} \cdot \mathbf{a} = 0$

---

## 📐 Finding Angles Between Vectors

From the geometric definition:
$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|}$$

Therefore:
$$\theta = \arccos\left(\frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|}\right)$$

### Special Cases

| If... | Then... |
|-------|---------|
| **a** · **b** > 0 | θ is acute (0° < θ < 90°) |
| **a** · **b** = 0 | θ = 90° (perpendicular) |
| **a** · **b** < 0 | θ is obtuse (90° < θ < 180°) |

---

## 📐 Orthogonality (Perpendicularity)

> **Definition:** Two vectors are **orthogonal** (perpendicular) if and only if:
> $$\mathbf{a} \cdot \mathbf{b} = 0$$

This is the most important application of the dot product!

**Example:** Are ⟨2, -3, 1⟩ and ⟨1, 1, 1⟩ orthogonal?
$$\langle 2, -3, 1 \rangle \cdot \langle 1, 1, 1 \rangle = 2 - 3 + 1 = 0$$
Yes, they are orthogonal! ✓

---

## 📐 Vector Projections

### Scalar Projection

The **scalar projection** (or **component**) of **b** onto **a** is:
$$\text{comp}_\mathbf{a}\mathbf{b} = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}|} = |\mathbf{b}|\cos\theta$$

This is the signed length of the shadow of **b** onto **a**.

### Vector Projection

The **vector projection** of **b** onto **a** is:
$$\text{proj}_\mathbf{a}\mathbf{b} = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}|^2}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{a} \cdot \mathbf{a}}\mathbf{a}$$

This is the vector component of **b** in the direction of **a**.

### Orthogonal Decomposition

Any vector **b** can be decomposed as:
$$\mathbf{b} = \text{proj}_\mathbf{a}\mathbf{b} + (\mathbf{b} - \text{proj}_\mathbf{a}\mathbf{b})$$

where the second term is perpendicular to **a**.

---

## ✏️ Worked Examples

### Example 1: Computing Dot Products
Find **a** · **b** for **a** = ⟨3, -2, 5⟩ and **b** = ⟨4, 1, -2⟩.

$$\mathbf{a} \cdot \mathbf{b} = (3)(4) + (-2)(1) + (5)(-2) = 12 - 2 - 10 = 0$$

The vectors are orthogonal!

---

### Example 2: Angle Between Vectors
Find the angle between **u** = ⟨1, 2, 3⟩ and **v** = ⟨2, -1, 1⟩.

$$\mathbf{u} \cdot \mathbf{v} = 2 - 2 + 3 = 3$$
$$|\mathbf{u}| = \sqrt{1 + 4 + 9} = \sqrt{14}$$
$$|\mathbf{v}| = \sqrt{4 + 1 + 1} = \sqrt{6}$$

$$\cos\theta = \frac{3}{\sqrt{14}\sqrt{6}} = \frac{3}{\sqrt{84}} = \frac{3}{2\sqrt{21}}$$

$$\theta = \arccos\left(\frac{3}{2\sqrt{21}}\right) \approx 70.9°$$

---

### Example 3: Vector Projection
Find the projection of **b** = ⟨3, 4, 0⟩ onto **a** = ⟨1, 0, 0⟩.

$$\text{proj}_\mathbf{a}\mathbf{b} = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}|^2}\mathbf{a} = \frac{3}{1}\langle 1, 0, 0 \rangle = \langle 3, 0, 0 \rangle$$

This makes sense: projecting onto the x-axis gives the x-component!

---

### Example 4: General Projection
Find proj_**a****b** where **a** = ⟨2, 1, 2⟩ and **b** = ⟨1, 3, 2⟩.

$$\mathbf{a} \cdot \mathbf{b} = 2 + 3 + 4 = 9$$
$$|\mathbf{a}|^2 = 4 + 1 + 4 = 9$$

$$\text{proj}_\mathbf{a}\mathbf{b} = \frac{9}{9}\langle 2, 1, 2 \rangle = \langle 2, 1, 2 \rangle$$

---

### Example 5: Work Done by a Force
A force **F** = ⟨3, 4, 2⟩ N moves an object from P(1, 0, 2) to Q(4, 3, 4). Find the work done.

Displacement vector:
$$\mathbf{d} = \overrightarrow{PQ} = \langle 3, 3, 2 \rangle$$

Work = **F** · **d**:
$$W = \langle 3, 4, 2 \rangle \cdot \langle 3, 3, 2 \rangle = 9 + 12 + 4 = 25 \text{ J}$$

---

## 📝 Practice Problems

### Level 1: Basic Dot Products
1. Find ⟨2, 5⟩ · ⟨-3, 4⟩
2. Find ⟨1, -2, 3⟩ · ⟨4, 2, -1⟩
3. Find |⟨3, 4⟩|² using the dot product

### Level 2: Angles and Orthogonality
4. Find the angle between ⟨1, 0⟩ and ⟨1, 1⟩
5. Are ⟨2, -1, 3⟩ and ⟨3, 3, -1⟩ orthogonal?
6. Find a vector orthogonal to both ⟨1, 0, 0⟩ and ⟨0, 1, 0⟩
7. Find the angle between the diagonals of a cube

### Level 3: Projections
8. Find comp_**a****b** where **a** = ⟨3, 4⟩ and **b** = ⟨2, 1⟩
9. Find proj_**a****b** for the vectors above
10. Decompose **b** = ⟨5, 3⟩ into components parallel and perpendicular to **a** = ⟨1, 1⟩

### Level 4: Applications
11. A sled is pulled with force 50 N at 30° above horizontal for 20 m. Find the work done.
12. Find the angle at vertex A of triangle ABC where A = (1, 0, 0), B = (0, 1, 0), C = (0, 0, 1).
13. Three forces **F₁** = ⟨10, 0⟩, **F₂** = ⟨0, 8⟩, **F₃** = ⟨-4, -3⟩ act on an object moving along **d** = ⟨6, 2⟩. Find total work.

### Level 5: Proofs
14. Prove that |**a** + **b**|² = |**a**|² + 2(**a** · **b**) + |**b**|²
15. Use the dot product to prove the Pythagorean theorem for orthogonal vectors
16. Prove: |**a** + **b**|² + |**a** - **b**|² = 2(|**a**|² + |**b**|²) (Parallelogram law)

---

## 📊 Answers

1. 14
2. -3
3. 25
4. 45°
5. Yes (2·3 + (-1)·3 + 3·(-1) = 0)
6. ⟨0, 0, 1⟩ or any scalar multiple
7. arccos(1/3) ≈ 70.5°
8. 2
9. ⟨6/5, 8/5⟩
10. Parallel: ⟨4, 4⟩, Perpendicular: ⟨1, -1⟩
11. 500√3 ≈ 866 J
12. 60°
13. 70 J
14. Expand (**a** + **b**) · (**a** + **b**)
15. If **a** ⊥ **b**, then |**a** + **b**|² = |**a**|² + |**b**|²
16. Expand both sides using dot product properties

---

## 🔬 Quantum Mechanics Connection

### Inner Products in Hilbert Space

The dot product generalizes to the **inner product** in quantum mechanics:

$$\langle\psi|\phi\rangle = \int_{-\infty}^{\infty} \psi^*(x)\phi(x) \, dx$$

**Key properties:**
- Orthogonal states: ⟨ψ|φ⟩ = 0
- Normalization: ⟨ψ|ψ⟩ = 1
- Probability amplitude: |⟨ψ|φ⟩|²

The concepts you're learning today are exactly what you'll use in quantum mechanics!

---

## ✅ Daily Checklist

- [ ] Read Stewart 12.3
- [ ] Watch 3Blue1Brown dot product video
- [ ] Master both definitions of dot product
- [ ] Compute angles between vectors
- [ ] Understand orthogonality condition
- [ ] Calculate projections
- [ ] Apply to work problems
- [ ] Complete practice problems

---

## 📓 Reflection Questions

1. Why does **a** · **b** = 0 mean the vectors are perpendicular?
2. What's the physical meaning of the scalar projection?
3. How is work related to the dot product?
4. Why is the dot product a scalar, not a vector?

---

## 🔜 Preview: Tomorrow

**Day 31: The Cross Product**
- Definition and computation
- Geometric meaning (area, perpendicular vector)
- Right-hand rule
- Applications to torque and angular momentum

---

*"The dot product measures how much two vectors point in the same direction."*
