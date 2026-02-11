# Day 92: Definition of Linear Transformations

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Linear Maps |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Exploration |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define linear transformations precisely
2. Verify whether a function is linear
3. Understand the structure-preserving nature of linear maps
4. Work with standard examples (rotation, reflection, projection)
5. Prove basic properties of linear transformations
6. Connect linear maps to quantum operators

---

## 📚 Required Reading

**Before starting:**
- Axler, Chapter 3.A: "The Vector Space of Linear Maps" (pp. 65-76)
- Review: Definition of vector space (Week 13)

---

## 🌅 Morning Session: Theory (3.5 hours)

### Part 1: The Definition (60 min)

#### Motivation: Structure-Preserving Functions

In mathematics, we study not just objects but the relationships between them. For vector spaces, the natural maps are those that **preserve the vector space structure**.

**Question:** What does it mean to preserve the structure of a vector space?

A vector space has two operations:
1. Vector addition: v + w
2. Scalar multiplication: cv

A function that preserves these operations is called **linear**.

#### Definition: Linear Transformation (Linear Map)

Let V and W be vector spaces over the same field F.

A function T: V → W is a **linear transformation** (or **linear map**) if:

**Additivity:** T(u + v) = T(u) + T(v) for all u, v ∈ V

**Homogeneity:** T(cv) = cT(v) for all c ∈ F, v ∈ V

**Equivalent single condition:**
T(au + bv) = aT(u) + bT(v) for all a, b ∈ F and u, v ∈ V

#### Why "Linear"?

The name comes from the fact that the graph of a linear function f: ℝ → ℝ is a line through the origin.

More generally, linear transformations preserve:
- The zero vector: T(0) = 0
- Linear combinations: T(∑cᵢvᵢ) = ∑cᵢT(vᵢ)
- Lines through origin map to lines through origin

#### Notation and Terminology

- T: V → W means "T is a function from V to W"
- V is the **domain**, W is the **codomain**
- T(v) is the **image** of v under T
- Other names: linear map, linear operator (when V = W), linear function

### Part 2: Fundamental Properties (45 min)

#### Theorem: T(0) = 0

**Proof:**
T(0) = T(0 · v) = 0 · T(v) = 0

(Using homogeneity with c = 0)

**Note:** This is necessary but not sufficient for linearity!

#### Theorem: T(-v) = -T(v)

**Proof:**
T(-v) = T((-1) · v) = (-1) · T(v) = -T(v)

#### Theorem: Linear Maps Preserve Linear Combinations

If T is linear, then:
$$T\left(\sum_{i=1}^{n} c_i v_i\right) = \sum_{i=1}^{n} c_i T(v_i)$$

**Proof by induction:**
- Base case (n=1): T(c₁v₁) = c₁T(v₁) ✓
- Inductive step: Assume true for n-1, then:
  T(∑cᵢvᵢ) = T(cₙvₙ + ∑_{i<n}cᵢvᵢ) = T(cₙvₙ) + T(∑_{i<n}cᵢvᵢ)
           = cₙT(vₙ) + ∑_{i<n}cᵢT(vᵢ) = ∑cᵢT(vᵢ) ✓

#### Key Consequence: Determined by Action on Basis

**Theorem:** A linear transformation T: V → W is completely determined by its values on a basis of V.

**Proof:**
Let {v₁, ..., vₙ} be a basis of V.
Any v ∈ V can be written as v = c₁v₁ + ... + cₙvₙ.
Then: T(v) = T(c₁v₁ + ... + cₙvₙ) = c₁T(v₁) + ... + cₙT(vₙ)

This is **huge**: knowing T on n vectors determines T on infinitely many!

### Part 3: Standard Examples (60 min)

#### Example 1: The Zero Transformation

T: V → W defined by T(v) = 0 for all v ∈ V

**Verification:**
- T(u + v) = 0 = 0 + 0 = T(u) + T(v) ✓
- T(cv) = 0 = c · 0 = cT(v) ✓

#### Example 2: The Identity Transformation

I: V → V defined by I(v) = v for all v ∈ V

**Verification:**
- I(u + v) = u + v = I(u) + I(v) ✓
- I(cv) = cv = cI(v) ✓

#### Example 3: Scalar Multiplication Map

For fixed c ∈ F, define Tₖ: V → V by Tₖ(v) = cv

**Verification:**
- Tₖ(u + v) = c(u + v) = cu + cv = Tₖ(u) + Tₖ(v) ✓
- Tₖ(av) = c(av) = a(cv) = aTₖ(v) ✓

#### Example 4: Rotation in ℝ² by Angle θ

Rθ: ℝ² → ℝ² defined by:
$$R_\theta(x, y) = (x\cos\theta - y\sin\theta, x\sin\theta + y\cos\theta)$$

**Verification:**
Let u = (x₁, y₁), v = (x₂, y₂)

Rθ(u + v) = Rθ(x₁ + x₂, y₁ + y₂)
= ((x₁+x₂)cosθ - (y₁+y₂)sinθ, (x₁+x₂)sinθ + (y₁+y₂)cosθ)
= (x₁cosθ - y₁sinθ + x₂cosθ - y₂sinθ, x₁sinθ + y₁cosθ + x₂sinθ + y₂cosθ)
= Rθ(u) + Rθ(v) ✓

Similarly verify homogeneity.

#### Example 5: Projection onto a Line (ℝ²)

Let L be the line spanned by unit vector u.
P: ℝ² → ℝ² defined by P(v) = (v · u)u

**Verification:**
- P(w₁ + w₂) = ((w₁ + w₂) · u)u = (w₁ · u)u + (w₂ · u)u = P(w₁) + P(w₂) ✓
- P(cw) = ((cw) · u)u = c(w · u)u = cP(w) ✓

#### Example 6: Differentiation

D: P(ℝ) → P(ℝ) where P(ℝ) = polynomials
D(p) = p' (the derivative)

**Verification:**
- D(p + q) = (p + q)' = p' + q' = D(p) + D(q) ✓
- D(cp) = (cp)' = cp' = cD(p) ✓

**Note:** The derivative is linear! This is why calculus works.

#### Example 7: Integration

For continuous functions on [a, b]:
I: C[a,b] → ℝ defined by I(f) = ∫ₐᵇ f(x) dx

**Verification:**
- I(f + g) = ∫(f + g) = ∫f + ∫g = I(f) + I(g) ✓
- I(cf) = ∫cf = c∫f = cI(f) ✓

### Part 4: Non-Examples (30 min)

Understanding what is **not** linear is equally important.

#### Non-Example 1: f(x) = x + 1

f: ℝ → ℝ defined by f(x) = x + 1

**Test:** f(0) = 1 ≠ 0
Therefore f is NOT linear.

**Geometric view:** The graph is a line, but not through the origin.

#### Non-Example 2: f(x) = x²

f: ℝ → ℝ defined by f(x) = x²

**Test:** f(1 + 1) = f(2) = 4, but f(1) + f(1) = 1 + 1 = 2
Since 4 ≠ 2, f is NOT linear.

#### Non-Example 3: f(x, y) = (x², y)

f: ℝ² → ℝ²

**Test:** f((1,0) + (1,0)) = f(2,0) = (4, 0)
But f(1,0) + f(1,0) = (1,0) + (1,0) = (2, 0)
Since (4,0) ≠ (2,0), f is NOT linear.

#### Non-Example 4: f(x, y) = xy

f: ℝ² → ℝ

**Test:** f(2·(1,1)) = f(2,2) = 4
But 2·f(1,1) = 2·1 = 2
Since 4 ≠ 2, f is NOT linear.

---

## 🌆 Afternoon Session: Problem Solving (2.5 hours)

### Problem Set A: Verification (45 min)

**Problem 1.** Determine if each function is linear. If not, identify which condition fails.

a) T: ℝ³ → ℝ² defined by T(x, y, z) = (x + y, y - z)

b) T: ℝ² → ℝ² defined by T(x, y) = (x + 1, y)

c) T: ℝ² → ℝ defined by T(x, y) = √(x² + y²)

d) T: P₂ → P₁ defined by T(a + bx + cx²) = b + 2cx

e) T: M₂ₓ₂ → ℝ defined by T(A) = det(A)

**Solutions:**

a) **Linear.** 
T(u + v) = T((x₁+x₂, y₁+y₂, z₁+z₂)) = (x₁+x₂+y₁+y₂, y₁+y₂-z₁-z₂) = T(u) + T(v) ✓
T(cu) = T(cx, cy, cz) = (cx+cy, cy-cz) = c(x+y, y-z) = cT(u) ✓

b) **NOT linear.** T(0, 0) = (1, 0) ≠ (0, 0).

c) **NOT linear.** T(1,0) + T(0,1) = 1 + 1 = 2, but T((1,0)+(0,1)) = T(1,1) = √2.

d) **Linear.** This is differentiation: T(p) = p'. Differentiation is linear.

e) **NOT linear.** det(2I) = 4 ≠ 2·det(I) = 2.

### Problem Set B: Construction (45 min)

**Problem 2.** Find a linear transformation T: ℝ² → ℝ² such that:
T(1, 0) = (2, 3) and T(0, 1) = (-1, 4)

**Solution:**
Any v = (x, y) = x(1,0) + y(0,1)
T(v) = T(x(1,0) + y(0,1)) = xT(1,0) + yT(0,1) = x(2,3) + y(-1,4) = (2x - y, 3x + 4y)

**Problem 3.** Find ALL linear transformations T: ℝ² → ℝ such that T(1, 1) = 3.

**Solution:**
Let T(1, 0) = a and T(0, 1) = b.
Then T(1, 1) = T(1,0) + T(0,1) = a + b = 3.
So b = 3 - a.
Therefore T(x, y) = ax + (3-a)y = ax + 3y - ay for any a ∈ ℝ.
Infinitely many solutions, parameterized by a.

**Problem 4.** Let T: ℝ³ → ℝ² be linear with:
T(1, 0, 0) = (1, 0)
T(0, 1, 0) = (0, 1)
T(0, 0, 1) = (1, 1)

Find T(2, -3, 5).

**Solution:**
T(2, -3, 5) = 2T(1,0,0) - 3T(0,1,0) + 5T(0,0,1)
= 2(1,0) - 3(0,1) + 5(1,1) = (2,0) + (0,-3) + (5,5) = (7, 2)

### Problem Set C: Proofs (60 min)

**Problem 5.** Prove: If T: V → W is linear, then T(0_V) = 0_W.

**Proof:**
T(0_V) = T(0 · v) for any v ∈ V
       = 0 · T(v)  (by homogeneity)
       = 0_W ∎

**Problem 6.** Prove: The composition of two linear maps is linear.

**Proof:**
Let S: U → V and T: V → W be linear.
Define T∘S: U → W by (T∘S)(u) = T(S(u)).

Additivity:
(T∘S)(u₁ + u₂) = T(S(u₁ + u₂)) = T(S(u₁) + S(u₂))  (S linear)
                = T(S(u₁)) + T(S(u₂))  (T linear)
                = (T∘S)(u₁) + (T∘S)(u₂) ✓

Homogeneity:
(T∘S)(cu) = T(S(cu)) = T(cS(u))  (S linear)
          = cT(S(u))  (T linear)
          = c(T∘S)(u) ✓

Therefore T∘S is linear. ∎

**Problem 7.** Prove: The sum of two linear maps is linear.

**Proof:**
Let S, T: V → W be linear.
Define (S + T)(v) = S(v) + T(v).

Additivity:
(S + T)(u + v) = S(u + v) + T(u + v)
               = S(u) + S(v) + T(u) + T(v)
               = (S(u) + T(u)) + (S(v) + T(v))
               = (S + T)(u) + (S + T)(v) ✓

Homogeneity:
(S + T)(cv) = S(cv) + T(cv) = cS(v) + cT(v) = c(S(v) + T(v)) = c(S + T)(v) ✓ ∎

**Problem 8.** Prove: The set L(V, W) of all linear maps from V to W is itself a vector space.

**Proof sketch:**
- Zero: The zero transformation 0(v) = 0_W
- Addition: (S + T)(v) = S(v) + T(v)
- Scalar multiplication: (cT)(v) = cT(v)
- Verify all 8 axioms (similar to function space verification)

---

## 🌙 Evening Session: Computational Exploration (1 hour)

```python
"""
Day 92: Linear Transformations
Computational verification and visualization
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# Part 1: Testing Linearity
# =============================================================

def test_linearity(T, domain_dim, num_tests=100, tol=1e-10):
    """
    Test if a function T is linear.
    
    Parameters:
    -----------
    T : function that takes a numpy array and returns a numpy array
    domain_dim : dimension of the input vectors
    num_tests : number of random tests
    tol : numerical tolerance
    
    Returns:
    --------
    (is_linear: bool, failed_test: str or None)
    """
    # Test 1: T(0) = 0
    zero = np.zeros(domain_dim)
    if np.linalg.norm(T(zero)) > tol:
        return False, "T(0) ≠ 0"
    
    for _ in range(num_tests):
        # Random vectors and scalars
        u = np.random.randn(domain_dim)
        v = np.random.randn(domain_dim)
        c = np.random.randn()
        a, b = np.random.randn(2)
        
        # Test additivity: T(u + v) = T(u) + T(v)
        if np.linalg.norm(T(u + v) - (T(u) + T(v))) > tol:
            return False, f"Additivity failed: T(u+v) ≠ T(u)+T(v)"
        
        # Test homogeneity: T(cv) = cT(v)
        if np.linalg.norm(T(c * v) - c * T(v)) > tol:
            return False, f"Homogeneity failed: T(cv) ≠ cT(v)"
        
        # Combined test: T(au + bv) = aT(u) + bT(v)
        if np.linalg.norm(T(a*u + b*v) - (a*T(u) + b*T(v))) > tol:
            return False, f"Combined test failed"
    
    return True, None


# Test examples
print("="*60)
print("Testing Linearity of Various Functions")
print("="*60)

# Example 1: Linear transformation
T1 = lambda v: np.array([v[0] + v[1], 2*v[0] - v[1]])
is_linear, msg = test_linearity(T1, 2)
print(f"\n1. T(x,y) = (x+y, 2x-y): {'Linear' if is_linear else f'NOT linear ({msg})'}")

# Example 2: Translation (NOT linear)
T2 = lambda v: v + np.array([1, 0])
is_linear, msg = test_linearity(T2, 2)
print(f"2. T(v) = v + (1,0): {'Linear' if is_linear else f'NOT linear ({msg})'}")

# Example 3: Squaring (NOT linear)
T3 = lambda v: np.array([v[0]**2, v[1]**2])
is_linear, msg = test_linearity(T3, 2)
print(f"3. T(x,y) = (x², y²): {'Linear' if is_linear else f'NOT linear ({msg})'}")

# Example 4: Rotation by 45°
theta = np.pi/4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
T4 = lambda v: R @ v
is_linear, msg = test_linearity(T4, 2)
print(f"4. Rotation by 45°: {'Linear' if is_linear else f'NOT linear ({msg})'}")

# =============================================================
# Part 2: Visualizing Linear Transformations
# =============================================================

def visualize_transformation(T, title, ax, grid_range=2, grid_points=10):
    """Visualize how a transformation affects a grid."""
    # Create grid
    x = np.linspace(-grid_range, grid_range, grid_points)
    y = np.linspace(-grid_range, grid_range, grid_points)
    
    # Draw original grid (light)
    for xi in x:
        ax.axvline(x=xi, color='lightblue', linewidth=0.5, alpha=0.5)
    for yi in y:
        ax.axhline(y=yi, color='lightblue', linewidth=0.5, alpha=0.5)
    
    # Transform and draw grid
    # Vertical lines
    for xi in x:
        points = np.array([[xi, yi] for yi in np.linspace(-grid_range, grid_range, 50)])
        transformed = np.array([T(p) for p in points])
        ax.plot(transformed[:, 0], transformed[:, 1], 'b-', linewidth=0.8, alpha=0.7)
    
    # Horizontal lines
    for yi in y:
        points = np.array([[xi, yi] for xi in np.linspace(-grid_range, grid_range, 50)])
        transformed = np.array([T(p) for p in points])
        ax.plot(transformed[:, 0], transformed[:, 1], 'r-', linewidth=0.8, alpha=0.7)
    
    # Mark origin
    origin_transformed = T(np.array([0.0, 0.0]))
    ax.plot(0, 0, 'ko', markersize=8, label='Original origin')
    ax.plot(origin_transformed[0], origin_transformed[1], 'go', markersize=8, 
            label='Transformed origin')
    
    # Mark basis vectors
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    e1_trans = T(e1)
    e2_trans = T(e2)
    
    ax.quiver(0, 0, e1_trans[0], e1_trans[1], angles='xy', scale_units='xy', 
              scale=1, color='blue', width=0.03, label='T(e₁)')
    ax.quiver(0, 0, e2_trans[0], e2_trans[1], angles='xy', scale_units='xy', 
              scale=1, color='red', width=0.03, label='T(e₂)')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Identity
I = lambda v: v
visualize_transformation(I, 'Identity', axes[0, 0])

# 2. Scaling
S = lambda v: 2 * v
visualize_transformation(S, 'Scaling by 2', axes[0, 1])

# 3. Rotation by 45°
theta = np.pi/4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
Rot = lambda v: R @ v
visualize_transformation(Rot, 'Rotation by 45°', axes[0, 2])

# 4. Shear
Shear = lambda v: np.array([v[0] + 0.5*v[1], v[1]])
visualize_transformation(Shear, 'Horizontal Shear', axes[1, 0])

# 5. Projection onto x-axis
Proj = lambda v: np.array([v[0], 0.0])
visualize_transformation(Proj, 'Projection onto x-axis', axes[1, 1])

# 6. Reflection across y = x
Refl = lambda v: np.array([v[1], v[0]])
visualize_transformation(Refl, 'Reflection across y = x', axes[1, 2])

plt.tight_layout()
plt.savefig('day92_transformations.png', dpi=150)
plt.show()

# =============================================================
# Part 3: Determined by Basis Action
# =============================================================

print("\n" + "="*60)
print("Linear Map Determined by Action on Basis")
print("="*60)

def create_linear_map(basis_images, basis=None):
    """
    Create a linear map from its action on a basis.
    
    Parameters:
    -----------
    basis_images : list of images T(e_i)
    basis : list of basis vectors (default: standard basis)
    
    Returns:
    --------
    function T that applies the linear map
    """
    n = len(basis_images)
    
    if basis is None:
        basis = [np.zeros(n) for _ in range(n)]
        for i in range(n):
            basis[i][i] = 1
    
    # Create matrix: columns are T(e_i)
    A = np.column_stack(basis_images)
    B = np.column_stack(basis)
    
    # If basis is standard, A is the matrix
    # Otherwise, need change of basis: A = [T(e_i)] = matrix columns
    # For general basis: T(v) = A @ B^{-1} @ v
    
    if np.allclose(B, np.eye(n)):
        return lambda v: A @ v
    else:
        B_inv = np.linalg.inv(B)
        return lambda v: A @ (B_inv @ v)


# Example: Define T by T(e1) = (2, -1), T(e2) = (1, 3)
T_e1 = np.array([2, -1])
T_e2 = np.array([1, 3])

T = create_linear_map([T_e1, T_e2])

print(f"\nT defined by: T(e₁) = {T_e1}, T(e₂) = {T_e2}")
print(f"T(1, 0) = {T(np.array([1, 0]))}")
print(f"T(0, 1) = {T(np.array([0, 1]))}")
print(f"T(1, 1) = {T(np.array([1, 1]))}  (should be T(e₁)+T(e₂) = {T_e1 + T_e2})")
print(f"T(3, -2) = {T(np.array([3, -2]))}  (should be 3T(e₁)-2T(e₂) = {3*T_e1 - 2*T_e2})")

# =============================================================
# Part 4: Quantum Connection Preview
# =============================================================

print("\n" + "="*60)
print("Quantum Operators as Linear Maps")
print("="*60)

# Pauli matrices (quantum operators on C²)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard

# Basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

print("\nPauli operators (linear maps on ℂ²):")
print(f"σx|0⟩ = {sigma_x @ ket_0} = |1⟩")
print(f"σx|1⟩ = {sigma_x @ ket_1} = |0⟩")
print(f"σz|0⟩ = {sigma_z @ ket_0} = |0⟩")
print(f"σz|1⟩ = {sigma_z @ ket_1} = -|1⟩")

print("\nHadamard gate (creates superposition):")
print(f"H|0⟩ = {H @ ket_0} = (|0⟩+|1⟩)/√2")
print(f"H|1⟩ = {H @ ket_1} = (|0⟩-|1⟩)/√2")

# Verify linearity
psi = (ket_0 + 1j * ket_1) / np.sqrt(2)
print(f"\nFor |ψ⟩ = (|0⟩ + i|1⟩)/√2:")
print(f"H|ψ⟩ = {H @ psi}")
print(f"Should equal (H|0⟩ + i·H|1⟩)/√2 = {(H @ ket_0 + 1j * H @ ket_1)/np.sqrt(2)}")
print("Linearity verified!" if np.allclose(H @ psi, (H @ ket_0 + 1j * H @ ket_1)/np.sqrt(2)) else "ERROR!")
```

---

## 📝 Homework

### Written Problems (Complete before tomorrow)

1. Verify that T: P₃ → P₂ defined by T(p) = p' (derivative) is linear.

2. Let T: ℝ³ → ℝ³ be defined by T(x, y, z) = (x - y, y - z, z - x).
   - Verify T is linear.
   - Find T(1, 2, 3).
   - Find T(T(v)) for a general v.

3. Prove or disprove: If T: V → W is linear and T(v) = w for some nonzero v, then T(2v) = 2w.

4. Find all linear T: ℝ² → ℝ² such that T(T(v)) = v for all v (involutions).

5. Let T: ℝ² → ℝ² be rotation by 90°.
   - Find T(e₁) and T(e₂).
   - Find T(3, 4).

### Coding Assignment

1. Implement a function that tests whether a given function f: ℝⁿ → ℝᵐ is approximately linear.

2. Create visualizations for:
   - Rotation by various angles
   - Composition of two rotations
   - A projection followed by a rotation

---

## ✅ Daily Checklist

- [ ] Read Axler Chapter 3.A
- [ ] Understood definition of linear transformation
- [ ] Can verify linearity
- [ ] Worked through all examples
- [ ] Completed Problem Sets A, B, C
- [ ] Ran computational code
- [ ] Started homework problems
- [ ] Created flashcards for key definitions

---

## 🔮 Preview: Tomorrow

**Day 93: Matrix Representation of Linear Maps**
- How to represent T: V → W as a matrix
- The matrix depends on choice of bases
- Column j = coordinates of T(basis vector j)
- This is the bridge from abstract to computational!

---

## 💭 Key Insight of the Day

> **A linear transformation is completely determined by what it does to a basis.**

This is profound: instead of knowing T on infinitely many vectors, we only need to know it on n vectors (the basis). This reduction from infinity to a finite amount of data is why linear algebra is computationally tractable.

In quantum mechanics, this means:
- To know what a gate does, we only need to specify its action on |0⟩ and |1⟩
- The Hadamard gate is defined by H|0⟩ = |+⟩ and H|1⟩ = |-⟩
- Everything else follows from linearity

---

*"Mathematics is the art of giving the same name to different things."*
— Henri Poincaré
