# Day 107: Norms and the Cauchy-Schwarz Inequality

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Norms and Inequalities |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define the norm induced by an inner product
2. Prove and apply the Cauchy-Schwarz inequality
3. Prove and apply the triangle inequality
4. Understand norms geometrically (length, distance)
5. Work with normalized vectors and unit vectors
6. Connect normalization to quantum state preparation

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 6.A**: Inner Products and Norms (pp. 178-186)
- Focus on: Definition 6.9 (norm), Theorem 6.13 (Cauchy-Schwarz), Theorem 6.18 (triangle inequality)

### Secondary Reading
- **Shankar, Chapter 1.3**: Review normalization discussion
- **Strang, Chapter 3.4**: Orthogonality (norm section)

---

## 📖 Core Content: Theory and Concepts

### 1. The Norm: Measuring Length

**Definition:** The **norm** of a vector v in an inner product space is:
$$\|v\| = \sqrt{\langle v | v \rangle}$$

**Properties of the norm** (for all vectors u, v and scalars α):

| Property | Statement | Name |
|----------|-----------|------|
| 1 | ‖v‖ ≥ 0, with equality iff v = 0 | Positive definiteness |
| 2 | ‖αv‖ = \|α\| · ‖v‖ | Absolute homogeneity |
| 3 | ‖u + v‖ ≤ ‖u‖ + ‖v‖ | Triangle inequality |

**Note:** Property 3 requires proof (via Cauchy-Schwarz).

### 2. Examples of Norms

#### Euclidean Norm on ℝⁿ
$$\|x\| = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$$

This is the familiar "length" from geometry.

#### Standard Norm on ℂⁿ
$$\|z\| = \sqrt{|z_1|^2 + |z_2|^2 + \cdots + |z_n|^2}$$

**Example:** ‖(1+i, 2)‖ = √(|1+i|² + |2|²) = √(2 + 4) = √6

#### L² Norm on Function Spaces
$$\|f\| = \sqrt{\int_a^b |f(x)|^2 \, dx}$$

**Example:** On [0, 2π]:
$$\|\sin\| = \sqrt{\int_0^{2\pi} \sin^2(x) \, dx} = \sqrt{\pi}$$

### 3. The Cauchy-Schwarz Inequality

**Theorem (Cauchy-Schwarz):** For all vectors u, v in an inner product space:
$$|\langle u | v \rangle| \leq \|u\| \cdot \|v\|$$

with equality if and only if one vector is a scalar multiple of the other.

**This is arguably the most important inequality in all of mathematics!**

#### Proof:

**Case 1:** If v = 0, then both sides equal 0. ✓

**Case 2:** Assume v ≠ 0. For any scalar t ∈ ℂ:
$$0 \leq \|u - tv\|^2 = \langle u - tv | u - tv \rangle$$

Expanding (using sesquilinearity):
$$= \langle u|u \rangle - t^*\langle v|u \rangle - t\langle u|v \rangle + |t|^2\langle v|v \rangle$$
$$= \|u\|^2 - t^*\langle v|u \rangle - t\langle u|v \rangle + |t|^2\|v\|^2$$

Choose $t = \frac{\langle v|u \rangle}{\|v\|^2}$ (projection coefficient):

Then $t^* = \frac{\langle u|v \rangle}{\|v\|^2}$ and $|t|^2 = \frac{|\langle u|v \rangle|^2}{\|v\|^4}$

Substituting:
$$0 \leq \|u\|^2 - \frac{|\langle u|v \rangle|^2}{\|v\|^2} - \frac{|\langle u|v \rangle|^2}{\|v\|^2} + \frac{|\langle u|v \rangle|^2}{\|v\|^2}$$
$$0 \leq \|u\|^2 - \frac{|\langle u|v \rangle|^2}{\|v\|^2}$$

Rearranging:
$$|\langle u|v \rangle|^2 \leq \|u\|^2 \|v\|^2$$

Taking square roots:
$$|\langle u|v \rangle| \leq \|u\| \cdot \|v\| \quad \blacksquare$$

**Equality condition:** Equality holds iff ‖u - tv‖ = 0, i.e., u = tv.

### 4. The Triangle Inequality

**Theorem:** For all vectors u, v:
$$\|u + v\| \leq \|u\| + \|v\|$$

#### Proof using Cauchy-Schwarz:

$$\|u + v\|^2 = \langle u+v | u+v \rangle = \|u\|^2 + \langle u|v \rangle + \langle v|u \rangle + \|v\|^2$$
$$= \|u\|^2 + 2\text{Re}(\langle u|v \rangle) + \|v\|^2$$

Since Re(z) ≤ |z|:
$$\leq \|u\|^2 + 2|\langle u|v \rangle| + \|v\|^2$$

By Cauchy-Schwarz:
$$\leq \|u\|^2 + 2\|u\|\|v\| + \|v\|^2 = (\|u\| + \|v\|)^2$$

Taking square roots:
$$\|u + v\| \leq \|u\| + \|v\| \quad \blacksquare$$

### 5. Normalized Vectors and Unit Vectors

**Definition:** A vector v is **normalized** (or a **unit vector**) if ‖v‖ = 1.

**Normalization procedure:** Given any v ≠ 0, define:
$$\hat{v} = \frac{v}{\|v\|}$$

Then ‖v̂‖ = 1.

**In Dirac notation:**
$$|\hat{\psi}\rangle = \frac{|\psi\rangle}{\sqrt{\langle\psi|\psi\rangle}}$$

### 6. Distance in Inner Product Spaces

**Definition:** The **distance** between vectors u and v is:
$$d(u, v) = \|u - v\|$$

**Properties (metric space axioms):**
1. d(u, v) ≥ 0, with equality iff u = v
2. d(u, v) = d(v, u) (symmetry)
3. d(u, w) ≤ d(u, v) + d(v, w) (triangle inequality)

### 7. Geometric Interpretation

In ℝ² or ℝ³, the inner product relates to the angle θ between vectors:
$$\langle u, v \rangle = \|u\| \|v\| \cos\theta$$

**Therefore:**
$$\cos\theta = \frac{\langle u, v \rangle}{\|u\| \|v\|}$$

Cauchy-Schwarz says |cos θ| ≤ 1, which we know geometrically!

**In complex spaces:** We don't have a simple angle interpretation, but:
$$|\langle u | v \rangle| = \|u\| \|v\| |\cos\theta_{\text{generalized}}|$$

where the "generalized angle" captures the complex relationship.

---

## 🔬 Quantum Mechanics Connection

### Normalization Condition

Physical quantum states must be normalized:
$$\langle \psi | \psi \rangle = 1 \quad \Leftrightarrow \quad \|\psi\| = 1$$

This ensures total probability = 1:
$$\sum_i |\langle i | \psi \rangle|^2 = \sum_i P(i) = 1$$

### Probability Bounds from Cauchy-Schwarz

The probability amplitude satisfies:
$$|\langle \phi | \psi \rangle| \leq \|\phi\| \|\psi\| = 1$$

(for normalized states)

So probabilities are automatically bounded: 0 ≤ P ≤ 1.

### Fidelity and State Overlap

The **fidelity** between quantum states:
$$F(|\phi\rangle, |\psi\rangle) = |\langle \phi | \psi \rangle|^2$$

- F = 1: identical states (up to global phase)
- F = 0: orthogonal (perfectly distinguishable)
- 0 < F < 1: partially overlapping

**Cauchy-Schwarz guarantees:** 0 ≤ F ≤ 1

### Distance Between Quantum States

**Trace distance:** Beyond this course, but related to:
$$D(\psi, \phi) = \sqrt{1 - |\langle\psi|\phi\rangle|^2}$$

This measures how distinguishable two states are.

### Uncertainty Principle Preview

The Cauchy-Schwarz inequality is key to proving:
$$\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle [A, B] \rangle|$$

(Heisenberg uncertainty principle)

---

## ✏️ Worked Examples

### Example 1: Computing Norms

Find ‖v‖ for v = (3, -4) in ℝ².

$$\|v\| = \sqrt{3^2 + (-4)^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

### Example 2: Complex Norm

Find ‖z‖ for z = (1+i, 2-i, 3) in ℂ³.

$$\|z\| = \sqrt{|1+i|^2 + |2-i|^2 + |3|^2}$$
$$= \sqrt{2 + 5 + 9} = \sqrt{16} = 4$$

### Example 3: Normalizing a Quantum State

Normalize |ψ⟩ = (1, 2i, -2).

$$\|\psi\| = \sqrt{|1|^2 + |2i|^2 + |-2|^2} = \sqrt{1 + 4 + 4} = 3$$

$$|\hat{\psi}\rangle = \frac{1}{3}(1, 2i, -2) = \left(\frac{1}{3}, \frac{2i}{3}, -\frac{2}{3}\right)$$

Verify: ‖ψ̂‖² = 1/9 + 4/9 + 4/9 = 9/9 = 1 ✓

### Example 4: Cauchy-Schwarz Verification

Verify Cauchy-Schwarz for u = (1, 2) and v = (3, 4) in ℝ².

**LHS:** |⟨u, v⟩| = |1·3 + 2·4| = |11| = 11

**RHS:** ‖u‖·‖v‖ = √5 · √25 = √5 · 5 = 5√5 ≈ 11.18

**Check:** 11 ≤ 11.18 ✓

### Example 5: When Cauchy-Schwarz is Equality

For u = (2, 4) and v = (1, 2):

**LHS:** |⟨u, v⟩| = |2 + 8| = 10

**RHS:** ‖u‖·‖v‖ = √20 · √5 = √100 = 10

**Equality!** Indeed, u = 2v (scalar multiple).

### Example 6: Triangle Inequality

For u = (3, 0) and v = (0, 4):

- ‖u‖ = 3
- ‖v‖ = 4
- ‖u + v‖ = ‖(3, 4)‖ = 5

Check: 5 ≤ 3 + 4 = 7 ✓

(This is the 3-4-5 right triangle!)

### Example 7: Function Norm

Compute ‖cos‖ on [0, 2π]:

$$\|\cos\|^2 = \int_0^{2\pi} \cos^2(x) \, dx = \int_0^{2\pi} \frac{1 + \cos(2x)}{2} \, dx = \pi$$

So ‖cos‖ = √π.

---

## 📝 Practice Problems

### Level 1: Basic Computations

1. Find ‖v‖ for v = (1, -2, 2, -1) in ℝ⁴.

2. Find ‖z‖ for z = (1-i, 1+i) in ℂ².

3. Normalize u = (1, 1, 1, 1) in ℝ⁴.

4. Normalize |ψ⟩ = (2, 1-i, i) in ℂ³.

### Level 2: Inequalities

5. Verify Cauchy-Schwarz for u = (1, 2, 3) and v = (1, 1, 1) in ℝ³.

6. For which c does Cauchy-Schwarz become equality for u = (1, c) and v = (2, 4)?

7. Show that ‖u + v‖² + ‖u - v‖² = 2‖u‖² + 2‖v‖² (parallelogram law).

8. Prove: ‖u - v‖ ≥ |‖u‖ - ‖v‖| (reverse triangle inequality).

### Level 3: Applications

9. Find the angle between u = (1, 2, 3) and v = (1, 0, -1) in ℝ³.

10. Given |ψ⟩ = (1, i)/√2 and |φ⟩ = (1, 1)/√2, find:
    - ⟨φ|ψ⟩
    - |⟨φ|ψ⟩|²  (fidelity)
    - The "angle" via |cos θ| = |⟨φ|ψ⟩|

11. Prove that for normalized states: ‖|ψ⟩ - |φ⟩‖² = 2(1 - Re⟨φ|ψ⟩).

12. Show that ‖u‖ = max{|⟨u, v⟩| : ‖v‖ = 1}.

### Level 4: Proofs

13. Prove: ⟨u, v⟩ = 0 for all v implies u = 0.

14. Prove: If ‖u + v‖ = ‖u‖ + ‖v‖, then u and v are parallel (v = cu for c ≥ 0).

15. Prove the Cauchy-Schwarz inequality for real inner products using calculus (minimize ‖u - tv‖² over t).

16. Show that the norm is continuous: if vₙ → v, then ‖vₙ‖ → ‖v‖.

---

## 📊 Answers and Hints

1. √(1+4+4+1) = √10
2. √(2+2) = 2
3. (1,1,1,1)/2
4. (2, 1-i, i)/√7
5. LHS = 6, RHS = √14·√3 ≈ 6.48 ✓
6. c = 2 (makes u parallel to v)
7. Expand both sides using inner products
8. Apply triangle inequality to u = (u-v) + v
9. cos θ = ⟨u,v⟩/(‖u‖‖v‖) = (1-3)/(√14·√2) = -2/√28 → θ ≈ 112°
10. ⟨φ|ψ⟩ = (1-i)/2, |...|² = 1/2, |cos θ| = 1/√2
11. Expand ‖ψ-φ‖² = ⟨ψ-φ|ψ-φ⟩
12. Maximum achieved when v = u/‖u‖
13. Take v = u
14. Trace back through triangle inequality proof; equality requires equality in Cauchy-Schwarz
15. Take derivative, set to zero, find minimum value
16. |‖vₙ‖ - ‖v‖| ≤ ‖vₙ - v‖ → 0

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# Lab 1: Norms in NumPy
# ============================================

# Real vectors
v = np.array([3, -4])
print(f"v = {v}")
print(f"||v|| = {np.linalg.norm(v)}")

# Complex vectors
z = np.array([1+1j, 2-1j, 3])
print(f"\nz = {z}")
print(f"||z|| = {np.linalg.norm(z)}")
print(f"Manual: {np.sqrt(np.sum(np.abs(z)**2))}")

# Different norms
x = np.array([1, 2, 3])
print(f"\nDifferent norms of {x}:")
print(f"L1 norm: {np.linalg.norm(x, ord=1)}")  # |x1| + |x2| + |x3|
print(f"L2 norm: {np.linalg.norm(x, ord=2)}")  # sqrt(x1² + x2² + x3²)
print(f"L∞ norm: {np.linalg.norm(x, ord=np.inf)}")  # max(|xi|)

# ============================================
# Lab 2: Normalization
# ============================================

def normalize(v):
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return v / norm

# Quantum state
psi = np.array([1, 2j, -2], dtype=complex)
psi_normalized = normalize(psi)

print(f"\n|ψ⟩ = {psi}")
print(f"||ψ|| = {np.linalg.norm(psi)}")
print(f"|ψ̂⟩ = {psi_normalized}")
print(f"||ψ̂|| = {np.linalg.norm(psi_normalized):.10f}")

# ============================================
# Lab 3: Cauchy-Schwarz Verification
# ============================================

def verify_cauchy_schwarz(u, v):
    """Verify Cauchy-Schwarz inequality"""
    inner = np.abs(np.vdot(u, v))
    product = np.linalg.norm(u) * np.linalg.norm(v)
    
    print(f"|⟨u|v⟩| = {inner:.6f}")
    print(f"||u|| · ||v|| = {product:.6f}")
    print(f"Satisfied: {inner <= product + 1e-10}")
    print(f"Ratio: {inner/product:.6f}")
    
    return inner, product

print("\n=== Cauchy-Schwarz Tests ===")
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
verify_cauchy_schwarz(u, v)

# Parallel vectors (equality case)
print("\nParallel case:")
u = np.array([2, 4, 6])
v = np.array([1, 2, 3])
verify_cauchy_schwarz(u, v)

# Random vectors
print("\nRandom complex vectors:")
u = np.random.randn(5) + 1j*np.random.randn(5)
v = np.random.randn(5) + 1j*np.random.randn(5)
verify_cauchy_schwarz(u, v)

# ============================================
# Lab 4: Triangle Inequality Visualization
# ============================================

fig, ax = plt.subplots(figsize=(8, 8))

# Vectors
u = np.array([3, 1])
v = np.array([1, 3])

# Plot vectors
origin = np.array([0, 0])
ax.quiver(*origin, *u, angles='xy', scale_units='xy', scale=1, 
          color='blue', width=0.02, label=f'u, ||u||={np.linalg.norm(u):.2f}')
ax.quiver(*u, *v, angles='xy', scale_units='xy', scale=1, 
          color='red', width=0.02, label=f'v, ||v||={np.linalg.norm(v):.2f}')
ax.quiver(*origin, *(u+v), angles='xy', scale_units='xy', scale=1, 
          color='green', width=0.02, label=f'u+v, ||u+v||={np.linalg.norm(u+v):.2f}')

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title(f'Triangle Inequality: {np.linalg.norm(u+v):.2f} ≤ {np.linalg.norm(u):.2f} + {np.linalg.norm(v):.2f} = {np.linalg.norm(u)+np.linalg.norm(v):.2f}')
plt.savefig('triangle_inequality.png', dpi=150)
plt.show()

# ============================================
# Lab 5: Quantum Fidelity
# ============================================

def fidelity(psi, phi):
    """Compute fidelity between normalized states"""
    return np.abs(np.vdot(psi, phi))**2

def distance(psi, phi):
    """Compute trace distance-like quantity"""
    return np.sqrt(1 - np.abs(np.vdot(psi, phi))**2)

# Computational basis
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Hadamard basis
ket_plus = normalize(ket_0 + ket_1)
ket_minus = normalize(ket_0 - ket_1)

print("\n=== Quantum State Fidelities ===")
print(f"F(|0⟩, |0⟩) = {fidelity(ket_0, ket_0):.4f}")
print(f"F(|0⟩, |1⟩) = {fidelity(ket_0, ket_1):.4f}")
print(f"F(|0⟩, |+⟩) = {fidelity(ket_0, ket_plus):.4f}")
print(f"F(|+⟩, |-⟩) = {fidelity(ket_plus, ket_minus):.4f}")

# Vary angle and plot fidelity
theta_vals = np.linspace(0, np.pi, 100)
fidelities = []
for theta in theta_vals:
    psi = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=complex)
    fidelities.append(fidelity(ket_0, psi))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(theta_vals, fidelities)
plt.xlabel('θ (radians)')
plt.ylabel('F(|0⟩, |ψ(θ)⟩)')
plt.title('Fidelity vs Bloch Sphere Angle')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(theta_vals, np.sqrt(1-np.array(fidelities)))
plt.xlabel('θ (radians)')
plt.ylabel('Distance')
plt.title('State Distance vs Bloch Sphere Angle')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fidelity_distance.png', dpi=150)
plt.show()

# ============================================
# Lab 6: Unit Ball Visualization
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Generate points on unit circle/sphere
theta = np.linspace(0, 2*np.pi, 100)

# L1 unit ball (diamond)
ax = axes[0]
x_l1 = np.concatenate([np.linspace(0, 1, 25), np.linspace(1, 0, 25), 
                       np.linspace(0, -1, 25), np.linspace(-1, 0, 25)])
y_l1 = np.concatenate([np.linspace(1, 0, 25), np.linspace(0, -1, 25),
                       np.linspace(-1, 0, 25), np.linspace(0, 1, 25)])
ax.plot(x_l1, y_l1, 'b-', linewidth=2)
ax.fill(x_l1, y_l1, alpha=0.3)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('L1 Unit Ball: |x| + |y| ≤ 1')
ax.grid(True, alpha=0.3)

# L2 unit ball (circle)
ax = axes[1]
x_l2 = np.cos(theta)
y_l2 = np.sin(theta)
ax.plot(x_l2, y_l2, 'r-', linewidth=2)
ax.fill(x_l2, y_l2, alpha=0.3, color='red')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('L2 Unit Ball: x² + y² ≤ 1')
ax.grid(True, alpha=0.3)

# L∞ unit ball (square)
ax = axes[2]
x_linf = [-1, 1, 1, -1, -1]
y_linf = [-1, -1, 1, 1, -1]
ax.plot(x_linf, y_linf, 'g-', linewidth=2)
ax.fill(x_linf, y_linf, alpha=0.3, color='green')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('L∞ Unit Ball: max(|x|,|y|) ≤ 1')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('unit_balls.png', dpi=150)
plt.show()

print("\n=== Lab Complete ===")
```

### Lab Exercises

1. Write a function to verify the triangle inequality for random vectors.

2. Plot the relationship between inner product magnitude and norm product for random vector pairs.

3. Implement a function to compute the "angle" between complex vectors.

4. Create an animation showing how fidelity changes as a qubit rotates on the Bloch sphere.

---

## ✅ Daily Checklist

- [ ] Read Axler 6.A (norm and Cauchy-Schwarz sections)
- [ ] Memorize the Cauchy-Schwarz inequality statement
- [ ] Understand the proof of Cauchy-Schwarz
- [ ] Prove triangle inequality from Cauchy-Schwarz
- [ ] Complete worked examples independently
- [ ] Solve problems 1-8 from practice set
- [ ] Complete computational lab
- [ ] Create flashcards for inequalities

---

## 📓 Reflection Questions

1. Why is Cauchy-Schwarz called "the most important inequality"?

2. How does the normalization requirement ‖ψ‖ = 1 relate to probability?

3. What does the fidelity F = |⟨φ|ψ⟩|² tell us physically?

4. Why do we use L² norms (and not L¹ or L∞) in quantum mechanics?

---

## 🔜 Preview: Tomorrow's Topics

**Day 108: Orthogonality and Orthogonal Complements**
- Orthogonal vectors: ⟨u|v⟩ = 0
- Orthogonal sets and bases
- Orthogonal complements: W⊥
- Direct sum decomposition

**QM preview:** Distinguishable states, measurement bases

---

*"The Cauchy-Schwarz inequality is the most useful, and most powerful of all inequalities."*
— J. Michael Steele
