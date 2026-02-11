# Day 108: Orthogonality and Orthogonal Complements

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Orthogonality |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define and identify orthogonal vectors
2. Work with orthogonal sets and understand their properties
3. Compute orthogonal complements of subspaces
4. Apply the direct sum decomposition theorem
5. Understand orthogonal projections geometrically
6. Connect orthogonality to quantum measurement and distinguishability

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 6.A**: Orthogonality (pp. 186-192)
- **Section 6.B**: Orthonormal Bases (first half, pp. 193-200)

### Secondary Reading
- **Strang, Chapter 4.1**: Orthogonality of the Four Subspaces
- **Shankar, Chapter 1.4**: Orthonormality and Completeness

---

## 📖 Core Content: Theory and Concepts

### 1. Orthogonal Vectors

**Definition:** Two vectors u and v are **orthogonal** (written u ⊥ v) if:
$$\langle u | v \rangle = 0$$

**Geometric interpretation (real case):** Perpendicular vectors.

**Properties:**
- 0 is orthogonal to every vector
- u ⊥ v implies v ⊥ u (by conjugate symmetry)
- u ⊥ u implies u = 0 (by positive definiteness)

### 2. The Pythagorean Theorem

**Theorem:** If u ⊥ v, then:
$$\|u + v\|^2 = \|u\|^2 + \|v\|^2$$

**Proof:**
$$\|u + v\|^2 = \langle u+v | u+v \rangle = \|u\|^2 + \langle u|v \rangle + \langle v|u \rangle + \|v\|^2$$
$$= \|u\|^2 + 0 + 0 + \|v\|^2 = \|u\|^2 + \|v\|^2 \quad \blacksquare$$

**Generalization:** If v₁, v₂, ..., vₙ are mutually orthogonal:
$$\|v_1 + v_2 + \cdots + v_n\|^2 = \|v_1\|^2 + \|v_2\|^2 + \cdots + \|v_n\|^2$$

### 3. Orthogonal Sets

**Definition:** A set of vectors {v₁, v₂, ..., vₖ} is **orthogonal** if:
$$\langle v_i | v_j \rangle = 0 \quad \text{for all } i \neq j$$

**Theorem:** An orthogonal set of nonzero vectors is linearly independent.

**Proof:** Suppose c₁v₁ + c₂v₂ + ... + cₖvₖ = 0.

Take inner product with vⱼ:
$$\langle v_j | c_1 v_1 + \cdots + c_k v_k \rangle = \langle v_j | 0 \rangle = 0$$
$$c_1 \langle v_j | v_1 \rangle + \cdots + c_j \langle v_j | v_j \rangle + \cdots + c_k \langle v_j | v_k \rangle = 0$$
$$c_j \|v_j\|^2 = 0$$

Since vⱼ ≠ 0, we have ‖vⱼ‖ ≠ 0, so cⱼ = 0.

This holds for all j, so all coefficients are zero. ∎

**Important consequence:** At most n orthogonal nonzero vectors in an n-dimensional space.

### 4. Orthonormal Sets

**Definition:** A set of vectors {e₁, e₂, ..., eₖ} is **orthonormal** if:
$$\langle e_i | e_j \rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

**In words:** Orthogonal AND each vector has norm 1.

**Orthonormal from orthogonal:** Given orthogonal set {v₁, ..., vₖ}, define:
$$e_i = \frac{v_i}{\|v_i\|}$$

### 5. Orthogonal Complement

**Definition:** The **orthogonal complement** of a subset S ⊆ V is:
$$S^\perp = \{v \in V : \langle v | s \rangle = 0 \text{ for all } s \in S\}$$

**Key properties:**

1. **S⊥ is always a subspace** (even if S isn't)

2. **{0}⊥ = V** and **V⊥ = {0}**

3. **S ⊆ T implies T⊥ ⊆ S⊥**

4. **S ⊆ (S⊥)⊥**

5. **For subspace W: W ∩ W⊥ = {0}**

### 6. Direct Sum Decomposition

**Theorem:** If W is a finite-dimensional subspace of inner product space V, then:
$$V = W \oplus W^\perp$$

This means every v ∈ V can be written **uniquely** as:
$$v = w + w^\perp$$
where w ∈ W and w⊥ ∈ W⊥.

**Moreover:** dim(W) + dim(W⊥) = dim(V)

### 7. Orthogonal Projection

**Definition:** The **orthogonal projection** of v onto subspace W is the unique vector w ∈ W such that (v - w) ⊥ W.

**Notation:** P_W(v) or proj_W(v)

**For one-dimensional W = span{u}:**
$$\text{proj}_u(v) = \frac{\langle u | v \rangle}{\langle u | u \rangle} u = \frac{\langle u | v \rangle}{\|u\|^2} u$$

**For orthonormal basis {e₁, ..., eₖ} of W:**
$$\text{proj}_W(v) = \sum_{i=1}^k \langle e_i | v \rangle e_i$$

### 8. Properties of Orthogonal Projections

Let P = P_W be the orthogonal projection onto W.

1. **P² = P** (idempotent)
2. **P† = P** (self-adjoint/Hermitian)
3. **‖Pv‖ ≤ ‖v‖** with equality iff v ∈ W
4. **P_W + P_{W⊥} = I** (identity)
5. **‖v - Pv‖ = min_{w∈W} ‖v - w‖** (closest point in W)

---

## 🔬 Quantum Mechanics Connection

### Orthogonality = Perfect Distinguishability

Two quantum states |ψ⟩ and |φ⟩ are **perfectly distinguishable** by some measurement if and only if they are orthogonal:
$$\langle \phi | \psi \rangle = 0$$

**Physical meaning:** There exists a measurement that gives outcome "ψ" with probability 1 for state |ψ⟩ and probability 0 for state |φ⟩.

### Measurement Bases

A quantum measurement in basis {|1⟩, |2⟩, ..., |n⟩} requires:
- **Orthonormality:** ⟨i|j⟩ = δᵢⱼ
- **Completeness:** Σᵢ |i⟩⟨i| = I

Then for state |ψ⟩ = Σᵢ cᵢ|i⟩:
- Probability of outcome i: P(i) = |cᵢ|² = |⟨i|ψ⟩|²
- Post-measurement state: |i⟩

### Projection Postulate

When we measure and get outcome i, the state **projects** onto |i⟩:
$$|\psi\rangle \xrightarrow{\text{measure } i} \frac{P_i |\psi\rangle}{\|P_i |\psi\rangle\|} = |i\rangle$$

where Pᵢ = |i⟩⟨i| is the projection onto the i-th eigenspace.

### Orthogonal Complements in QM

If W is the subspace of "spin-up" states, then W⊥ is the "spin-down" subspace.

The decomposition V = W ⊕ W⊥ corresponds to:
"Every state can be written as superposition of spin-up and spin-down components."

### Example: Qubit Measurements

**Z-basis:** {|0⟩, |1⟩} - orthonormal, measures spin along z-axis

**X-basis:** {|+⟩, |-⟩} where |±⟩ = (|0⟩ ± |1⟩)/√2 - also orthonormal!

Note: |+⟩ and |-⟩ are NOT orthogonal to |0⟩ and |1⟩ individually.

---

## ✏️ Worked Examples

### Example 1: Checking Orthogonality

Are u = (1, 2, 3) and v = (2, -1, 0) orthogonal in ℝ³?

$$\langle u, v \rangle = 1(2) + 2(-1) + 3(0) = 2 - 2 + 0 = 0$$ ✓

Yes, they are orthogonal!

### Example 2: Orthogonal Set Verification

Show that {(1,1,0), (1,-1,2), (1,-1,-1)} is orthogonal in ℝ³.

- ⟨(1,1,0), (1,-1,2)⟩ = 1 - 1 + 0 = 0 ✓
- ⟨(1,1,0), (1,-1,-1)⟩ = 1 - 1 + 0 = 0 ✓
- ⟨(1,-1,2), (1,-1,-1)⟩ = 1 + 1 - 2 = 0 ✓

All pairs orthogonal! ✓

### Example 3: Converting to Orthonormal

Make the set from Example 2 orthonormal.

- ‖(1,1,0)‖ = √2 → e₁ = (1,1,0)/√2
- ‖(1,-1,2)‖ = √6 → e₂ = (1,-1,2)/√6
- ‖(1,-1,-1)‖ = √3 → e₃ = (1,-1,-1)/√3

### Example 4: Orthogonal Complement

Find W⊥ where W = span{(1,1,1)} in ℝ³.

W⊥ = {(x,y,z) : ⟨(1,1,1), (x,y,z)⟩ = 0} = {(x,y,z) : x + y + z = 0}

This is a plane through the origin! dim(W⊥) = 2.

Basis for W⊥: {(1,-1,0), (1,0,-1)}

### Example 5: Orthogonal Projection

Project v = (3, 4) onto u = (1, 1) in ℝ².

$$\text{proj}_u(v) = \frac{\langle u, v \rangle}{\|u\|^2} u = \frac{3 + 4}{2}(1, 1) = \frac{7}{2}(1, 1) = (3.5, 3.5)$$

Check: v - proj_u(v) = (3,4) - (3.5, 3.5) = (-0.5, 0.5) ⊥ (1,1)?
⟨(-0.5, 0.5), (1,1)⟩ = -0.5 + 0.5 = 0 ✓

### Example 6: Projection onto Subspace

Let W = span{e₁, e₂} where e₁ = (1,0,0), e₂ = (0,1,0) (orthonormal).

Project v = (3, 4, 5) onto W:

$$\text{proj}_W(v) = \langle e_1 | v \rangle e_1 + \langle e_2 | v \rangle e_2 = 3e_1 + 4e_2 = (3, 4, 0)$$

The component in W⊥ is v - proj_W(v) = (0, 0, 5).

### Example 7: Quantum Measurement Projection

Let |ψ⟩ = (3|0⟩ + 4|1⟩)/5 (normalized).

Project onto the |+⟩ state:

$$P_{|+\rangle}|\psi\rangle = |+\rangle\langle +|\psi\rangle = |+\rangle \cdot \langle +|\psi\rangle$$

$$\langle +|\psi\rangle = \frac{1}{\sqrt{2}}(⟨0| + ⟨1|) \cdot \frac{1}{5}(3|0⟩ + 4|1⟩) = \frac{1}{5\sqrt{2}}(3 + 4) = \frac{7}{5\sqrt{2}}$$

$$P_{|+\rangle}|\psi\rangle = \frac{7}{5\sqrt{2}}|+\rangle$$

Probability = |7/(5√2)|² = 49/50 = 0.98

---

## 📝 Practice Problems

### Level 1: Basic Orthogonality

1. Determine if (1, 2, -1) and (3, 0, 3) are orthogonal in ℝ³.

2. Find all vectors in ℝ² orthogonal to (3, 4).

3. Verify that the standard basis {e₁, e₂, e₃} in ℝ³ is orthonormal.

4. Show that |0⟩ and |1⟩ are orthonormal in ℂ².

### Level 2: Orthogonal Complements

5. Find W⊥ if W = span{(1, 0, 1), (0, 1, 1)} in ℝ³.

6. Find dim(W⊥) if W is a 3-dimensional subspace of ℝ⁵.

7. Prove that (W⊥)⊥ = W for any subspace W.

8. Show that (U + W)⊥ = U⊥ ∩ W⊥.

### Level 3: Projections

9. Project (1, 2, 3) onto the line through (1, 1, 1).

10. Project (1, 2, 3, 4) onto the subspace W = span{(1,0,0,0), (0,1,0,0)}.

11. Find the distance from (1, 2, 3) to the plane x + y + z = 0.

12. Show that the projection matrix P_u = uu†/‖u‖² satisfies P² = P and P† = P.

### Level 4: Quantum Applications

13. Verify that {|+⟩, |-⟩} forms an orthonormal basis for ℂ².

14. Express |0⟩ in the {|+⟩, |-⟩} basis.

15. If |ψ⟩ = α|0⟩ + β|1⟩ is measured in the {|+⟩, |-⟩} basis, find P(+) and P(-).

16. Prove: For any two orthonormal bases {|eᵢ⟩} and {|fⱼ⟩}, we have Σᵢ |⟨eᵢ|ψ⟩|² = Σⱼ |⟨fⱼ|ψ⟩|² = 1.

---

## 📊 Answers and Hints

1. ⟨(1,2,-1),(3,0,3)⟩ = 3+0-3 = 0. Yes!
2. {(a,-3a/4) : a ∈ ℝ} = span{(4,-3)}
3. ⟨eᵢ,eⱼ⟩ = δᵢⱼ by definition
4. ⟨0|0⟩ = 1, ⟨1|1⟩ = 1, ⟨0|1⟩ = 0 ✓
5. Solve x + z = 0 and y + z = 0; W⊥ = span{(1,1,-1)}
6. dim(W⊥) = 5 - 3 = 2
7. Use definition and positive definiteness
8. v ∈ (U+W)⊥ ⟺ v ⊥ all u+w ⟺ v ⊥ U and v ⊥ W
9. proj = (6/3)(1,1,1) = (2,2,2)
10. proj = (1,2,0,0)
11. Distance = |proj onto normal| = |6/√3| = 2√3
12. Direct computation
13. Compute ⟨+|+⟩, ⟨-|-⟩, ⟨+|-⟩
14. |0⟩ = (|+⟩ + |-⟩)/√2
15. P(+) = |α+β|²/2, P(-) = |α-β|²/2
16. Both equal ⟨ψ|ψ⟩ = 1 (Parseval's identity)

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Lab 1: Orthogonality Checks
# ============================================

def is_orthogonal(u, v, tol=1e-10):
    """Check if two vectors are orthogonal"""
    return np.abs(np.vdot(u, v)) < tol

def is_orthogonal_set(vectors, tol=1e-10):
    """Check if a set of vectors is orthogonal"""
    n = len(vectors)
    for i in range(n):
        for j in range(i+1, n):
            if not is_orthogonal(vectors[i], vectors[j], tol):
                return False, (i, j)
    return True, None

def is_orthonormal_set(vectors, tol=1e-10):
    """Check if a set of vectors is orthonormal"""
    # Check orthogonality
    orth, pair = is_orthogonal_set(vectors, tol)
    if not orth:
        return False, f"Not orthogonal: vectors {pair}"
    
    # Check normalization
    for i, v in enumerate(vectors):
        if np.abs(np.linalg.norm(v) - 1) > tol:
            return False, f"Vector {i} not normalized"
    
    return True, "Orthonormal!"

# Test with standard basis
e1, e2, e3 = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
print("Standard basis orthonormal?", is_orthonormal_set([e1, e2, e3]))

# Test with custom set
v1 = np.array([1, 1, 0])
v2 = np.array([1, -1, 2])
v3 = np.array([1, -1, -1])
print("\nCustom set orthogonal?", is_orthogonal_set([v1, v2, v3]))

# Normalize
v1_norm = v1 / np.linalg.norm(v1)
v2_norm = v2 / np.linalg.norm(v2)
v3_norm = v3 / np.linalg.norm(v3)
print("After normalization:", is_orthonormal_set([v1_norm, v2_norm, v3_norm]))

# ============================================
# Lab 2: Orthogonal Complement
# ============================================

def orthogonal_complement_basis(W_basis, n):
    """
    Find basis for orthogonal complement of span(W_basis) in R^n
    Uses SVD to find null space
    """
    if len(W_basis) == 0:
        return np.eye(n)
    
    W = np.array(W_basis).T  # columns are basis vectors
    # SVD: W = U @ S @ Vh
    U, S, Vh = np.linalg.svd(W.T)
    
    # Null space is spanned by rows of Vh corresponding to zero singular values
    rank = np.sum(S > 1e-10)
    null_space = Vh[rank:].T
    
    return null_space

# W = span{(1,1,1)} in R^3
W_basis = [np.array([1, 1, 1])]
W_perp = orthogonal_complement_basis(W_basis, 3)
print("\n=== Orthogonal Complement ===")
print(f"W = span{{(1,1,1)}}")
print(f"W⊥ basis:\n{W_perp}")

# Verify orthogonality
w = np.array([1, 1, 1])
for i in range(W_perp.shape[1]):
    print(f"⟨w, W⊥[:,{i}]⟩ = {np.dot(w, W_perp[:,i]):.6f}")

# ============================================
# Lab 3: Orthogonal Projection
# ============================================

def project_onto_vector(v, u):
    """Project v onto the line spanned by u"""
    return (np.vdot(u, v) / np.vdot(u, u)) * u

def project_onto_subspace(v, orthonormal_basis):
    """Project v onto subspace spanned by orthonormal basis"""
    proj = np.zeros_like(v, dtype=complex)
    for e in orthonormal_basis:
        proj += np.vdot(e, v) * e
    return proj

# Project (3, 4) onto (1, 1)
v = np.array([3, 4])
u = np.array([1, 1])
proj = project_onto_vector(v, u)
print("\n=== Projection onto Vector ===")
print(f"v = {v}")
print(f"u = {u}")
print(f"proj_u(v) = {proj}")
print(f"v - proj = {v - proj}")
print(f"Verify orthogonal: ⟨u, v-proj⟩ = {np.dot(u, v-proj):.6f}")

# Project onto subspace
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
v = np.array([1, 2, 3])
proj = project_onto_subspace(v, [e1, e2])
print(f"\nProject {v} onto xy-plane:")
print(f"proj = {proj}")

# ============================================
# Lab 4: Quantum Bases
# ============================================

# Computational basis
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Hadamard basis
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

print("\n=== Quantum Bases ===")
print("Computational basis orthonormal?", is_orthonormal_set([ket_0, ket_1]))
print("Hadamard basis orthonormal?", is_orthonormal_set([ket_plus, ket_minus]))

# Express |0⟩ in Hadamard basis
c_plus = np.vdot(ket_plus, ket_0)
c_minus = np.vdot(ket_minus, ket_0)
print(f"\n|0⟩ = {c_plus:.4f}|+⟩ + {c_minus:.4f}|-⟩")

# Verify
reconstructed = c_plus * ket_plus + c_minus * ket_minus
print(f"Reconstructed: {reconstructed}")

# ============================================
# Lab 5: Measurement Probabilities
# ============================================

def measure_in_basis(psi, basis, labels=None):
    """Compute measurement probabilities in given basis"""
    if labels is None:
        labels = [f"|{i}⟩" for i in range(len(basis))]
    
    print("Measurement probabilities:")
    total_prob = 0
    for i, (e, label) in enumerate(zip(basis, labels)):
        amp = np.vdot(e, psi)
        prob = np.abs(amp)**2
        total_prob += prob
        print(f"  P({label}) = |{amp:.4f}|² = {prob:.4f}")
    print(f"  Total: {total_prob:.6f}")

# State
psi = np.array([3, 4], dtype=complex) / 5  # normalized

print("\n=== Measurement in Different Bases ===")
print(f"|ψ⟩ = {psi}")

print("\nZ-basis (computational):")
measure_in_basis(psi, [ket_0, ket_1], ["|0⟩", "|1⟩"])

print("\nX-basis (Hadamard):")
measure_in_basis(psi, [ket_plus, ket_minus], ["|+⟩", "|-⟩"])

# ============================================
# Lab 6: Visualization
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Projection in R²
ax = axes[0]
v = np.array([3, 4])
u = np.array([2, 1])
proj = project_onto_vector(v, u)

ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
          color='blue', width=0.02, label='v')
ax.quiver(0, 0, u[0]*2, u[1]*2, angles='xy', scale_units='xy', scale=1, 
          color='red', width=0.02, alpha=0.5, label='u (extended)')
ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, 
          color='green', width=0.02, label='proj_u(v)')
ax.plot([v[0], proj[0]], [v[1], proj[1]], 'k--', alpha=0.5, label='v - proj')

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Orthogonal Projection in ℝ²')

# Right: Bloch sphere with orthogonal states
ax = fig.add_subplot(122, projection='3d')

# Unit sphere
u_sphere = np.linspace(0, 2 * np.pi, 30)
v_sphere = np.linspace(0, np.pi, 20)
x = np.outer(np.cos(u_sphere), np.sin(v_sphere))
y = np.outer(np.sin(u_sphere), np.sin(v_sphere))
z = np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
ax.plot_surface(x, y, z, alpha=0.1, color='blue')

# Orthogonal state pairs
states = {
    '|0⟩': (0, 0, 1), '|1⟩': (0, 0, -1),  # Z-axis
    '|+⟩': (1, 0, 0), '|-⟩': (-1, 0, 0),  # X-axis
    '|+i⟩': (0, 1, 0), '|-i⟩': (0, -1, 0),  # Y-axis
}

colors = ['blue', 'blue', 'red', 'red', 'green', 'green']
for (name, coords), color in zip(states.items(), colors):
    ax.scatter(*coords, s=100, c=color)
    ax.text(coords[0]*1.2, coords[1]*1.2, coords[2]*1.2, name, fontsize=10)

# Draw axes
ax.plot([-1.5, 1.5], [0, 0], [0, 0], 'r--', alpha=0.3)
ax.plot([0, 0], [-1.5, 1.5], [0, 0], 'g--', alpha=0.3)
ax.plot([0, 0], [0, 0], [-1.5, 1.5], 'b--', alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Orthogonal States on Bloch Sphere\n(Opposite points are orthogonal)')

plt.tight_layout()
plt.savefig('orthogonality.png', dpi=150)
plt.show()

print("\n=== Lab Complete ===")
```

---

## ✅ Daily Checklist

- [ ] Read Axler 6.A-B on orthogonality
- [ ] Understand Pythagorean theorem for orthogonal vectors
- [ ] Know why orthogonal sets are independent
- [ ] Compute orthogonal complements
- [ ] Master orthogonal projection formula
- [ ] Complete all worked examples
- [ ] Solve problems 1-10 from practice set
- [ ] Complete computational lab

---

## 📓 Reflection Questions

1. Why does orthogonality imply linear independence?

2. What is the geometric meaning of V = W ⊕ W⊥?

3. How does orthogonal projection relate to "best approximation"?

4. Why are measurement bases required to be orthonormal in QM?

---

## 🔜 Preview: Tomorrow's Topics

**Day 109: Gram-Schmidt Orthogonalization**
- Converting any basis to orthonormal basis
- The Gram-Schmidt algorithm
- QR decomposition
- Applications to least squares

---

*"Orthogonality is the mathematician's version of independence."*
— Anonymous
