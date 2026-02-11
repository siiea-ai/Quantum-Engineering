# Day 106: Inner Products — The Foundation of Quantum Geometry

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Inner Product Definition |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define inner products on real and complex vector spaces
2. Verify inner product axioms for given functions
3. Distinguish between real (symmetric) and complex (Hermitian) inner products
4. Compute inner products in ℝⁿ, ℂⁿ, and function spaces
5. Understand the sesquilinear nature of complex inner products
6. Connect inner products to quantum mechanical probability amplitudes

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 6.A**: Inner Products and Norms (pp. 164-178)
- Focus on: Definition 6.1, Examples 6.2-6.7

### Secondary Reading
- **Shankar, Chapter 1.3**: The Inner Product (pp. 10-18)
- **Griffiths QM, Section 3.1**: Hilbert Space (first 5 pages)

### Convention Alert!
⚠️ **Physics vs Mathematics Convention:**
- **Mathematics (Axler):** ⟨u, v⟩ linear in first argument
- **Physics (Shankar, Dirac):** ⟨u|v⟩ linear in second argument

We will use the **physics convention** to align with quantum mechanics:
$$\langle u | v \rangle \text{ is linear in } |v\rangle \text{ and antilinear in } \langle u|$$

---

## 🎬 Video Resources

### 3Blue1Brown
- **Essence of Linear Algebra, Chapter 9**: Dot products and duality
- URL: https://www.youtube.com/watch?v=LyGKycYT2v0

### MIT OCW 18.06
- **Lecture 15**: Projections onto Subspaces
- **Lecture 17**: Orthogonal Matrices and Gram-Schmidt

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: Why Inner Products?

Vector spaces give us **linear structure** (addition, scaling). But we're missing:
- **Length** of vectors
- **Angle** between vectors
- **Orthogonality** (perpendicularity)
- **Distance** between vectors

Inner products provide all of these!

**In Quantum Mechanics:**
- Inner product ⟨φ|ψ⟩ = probability amplitude
- |⟨φ|ψ⟩|² = probability
- ⟨ψ|ψ⟩ = 1 (normalization)
- ⟨φ|ψ⟩ = 0 (orthogonal/distinguishable states)

### 2. Definition: Real Inner Product

**Definition:** An **inner product** on a real vector space V is a function
$$\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$$
satisfying for all u, v, w ∈ V and c ∈ ℝ:

| Axiom | Name | Statement |
|-------|------|-----------|
| 1 | Symmetry | ⟨u, v⟩ = ⟨v, u⟩ |
| 2 | Linearity (1st arg) | ⟨u + v, w⟩ = ⟨u, w⟩ + ⟨v, w⟩ |
| 3 | Homogeneity (1st arg) | ⟨cu, v⟩ = c⟨u, v⟩ |
| 4 | Positive definiteness | ⟨v, v⟩ ≥ 0, with equality iff v = 0 |

**Note:** Axioms 2 & 3 together say: linear in first argument.
By symmetry, also linear in second argument → **bilinear**.

### 3. Definition: Complex Inner Product (Crucial for QM!)

**Definition:** An **inner product** on a complex vector space V is a function
$$\langle \cdot | \cdot \rangle : V \times V \to \mathbb{C}$$
satisfying for all |u⟩, |v⟩, |w⟩ ∈ V and α ∈ ℂ:

| Axiom | Name | Statement |
|-------|------|-----------|
| 1 | Conjugate Symmetry | ⟨u|v⟩ = ⟨v|u⟩* |
| 2 | Linearity in 2nd arg | ⟨u|v + w⟩ = ⟨u|v⟩ + ⟨u|w⟩ |
| 3 | Homogeneity in 2nd arg | ⟨u|αv⟩ = α⟨u|v⟩ |
| 4 | Positive definiteness | ⟨v|v⟩ ≥ 0 (real!), with equality iff v = 0 |

**Critical observation:** From conjugate symmetry:
$$\langle \alpha u | v \rangle = \langle v | \alpha u \rangle^* = (\alpha \langle v | u \rangle)^* = \alpha^* \langle v | u \rangle^* = \alpha^* \langle u | v \rangle$$

So: **antilinear (conjugate-linear) in first argument!**

This is called **sesquilinear** (one-and-a-half linear).

### 4. The Standard Inner Products

#### Example 1: Dot Product on ℝⁿ

For x = (x₁,...,xₙ) and y = (y₁,...,yₙ):
$$\langle x, y \rangle = x_1 y_1 + x_2 y_2 + \cdots + x_n y_n = \sum_{i=1}^n x_i y_i = x^T y$$

**Verification of axioms:**
- Symmetry: x·y = y·x ✓
- Linearity: (x+y)·z = x·z + y·z ✓
- Positive definite: x·x = Σxᵢ² ≥ 0, equals 0 iff all xᵢ = 0 ✓

#### Example 2: Standard Inner Product on ℂⁿ (The QM Inner Product!)

For z = (z₁,...,zₙ) and w = (w₁,...,wₙ):
$$\langle z | w \rangle = z_1^* w_1 + z_2^* w_2 + \cdots + z_n^* w_n = \sum_{i=1}^n z_i^* w_i = z^\dagger w$$

Where z† = (z*)ᵀ is the **conjugate transpose** (adjoint).

**Why the conjugate?** For positive definiteness!
$$\langle z | z \rangle = |z_1|^2 + |z_2|^2 + \cdots + |z_n|^2 \geq 0$$

Without conjugate: z·z = z₁² + z₂² + ... could be negative or complex!

#### Example 3: Function Space Inner Product

On V = C([a,b]) (continuous functions on [a,b]):
$$\langle f, g \rangle = \int_a^b f(x) g(x) \, dx$$

For complex functions:
$$\langle f | g \rangle = \int_a^b f(x)^* g(x) \, dx$$

**Verification:**
- Conjugate symmetry: ∫f*g = (∫g*f)* ✓
- Linearity: ∫f*(g+h) = ∫f*g + ∫f*h ✓
- Positive definite: ∫|f|² ≥ 0, equals 0 iff f = 0 (almost everywhere) ✓

#### Example 4: Weighted Inner Product

On ℝⁿ with positive weights w₁,...,wₙ > 0:
$$\langle x, y \rangle_w = \sum_{i=1}^n w_i x_i y_i$$

On function space with weight function w(x) > 0:
$$\langle f, g \rangle_w = \int_a^b w(x) f(x) g(x) \, dx$$

### 5. Dirac Bra-Ket Notation

**The genius of Dirac notation** separates vectors into:
- **Ket** |ψ⟩: column vector (element of V)
- **Bra** ⟨φ|: row vector (element of V*, the dual space)
- **Bracket** ⟨φ|ψ⟩: inner product (a number!)

**In matrix form for ℂⁿ:**
$$|ψ\rangle = \begin{pmatrix} ψ_1 \\ ψ_2 \\ \vdots \\ ψ_n \end{pmatrix}, \quad
\langle φ| = \begin{pmatrix} φ_1^* & φ_2^* & \cdots & φ_n^* \end{pmatrix}$$

$$\langle φ|ψ\rangle = \begin{pmatrix} φ_1^* & \cdots & φ_n^* \end{pmatrix} \begin{pmatrix} ψ_1 \\ \vdots \\ ψ_n \end{pmatrix} = \sum_i φ_i^* ψ_i$$

**Key relationships:**
- ⟨ψ| = |ψ⟩† (bra is adjoint of ket)
- ⟨φ|ψ⟩ = ⟨ψ|φ⟩* (conjugate symmetry)
- ⟨φ|(α|ψ⟩ + β|χ⟩) = α⟨φ|ψ⟩ + β⟨φ|χ⟩ (linearity in ket)
- (⟨φ| + ⟨χ|)|ψ⟩ = ⟨φ|ψ⟩ + ⟨χ|ψ⟩ (linearity in bra)

### 6. Properties Derived from Axioms

**Theorem:** In any inner product space:

1. **⟨v, 0⟩ = ⟨0, v⟩ = 0** for all v
   
   *Proof:* ⟨v, 0⟩ = ⟨v, 0·0⟩ = 0·⟨v, 0⟩ = 0 ∎

2. **Parallelogram Law:**
   $$\|u + v\|^2 + \|u - v\|^2 = 2\|u\|^2 + 2\|v\|^2$$

3. **Polarization Identity (Real):**
   $$\langle u, v \rangle = \frac{1}{4}(\|u+v\|^2 - \|u-v\|^2)$$

4. **Polarization Identity (Complex):**
   $$\langle u | v \rangle = \frac{1}{4}\sum_{k=0}^{3} i^k \|u + i^k v\|^2$$

The polarization identities show: **inner product is determined by the norm!**

### 7. Non-Examples

#### Non-Example 1: Missing Positive Definiteness
On ℝ²: ⟨(x₁,x₂), (y₁,y₂)⟩ = x₁y₁ - x₂y₂

This is the **Minkowski metric** from special relativity!
⟨(1,1), (1,1)⟩ = 1 - 1 = 0, but (1,1) ≠ 0.
Not an inner product (but still important in physics).

#### Non-Example 2: Not Sesquilinear
On ℂ²: ⟨z, w⟩ = z₁w₁ + z₂w₂ (no conjugate)

⟨(i,0), (i,0)⟩ = i² + 0 = -1 < 0
Fails positive definiteness!

---

## 🔬 Quantum Mechanics Connection

### The Born Rule

The inner product is the heart of quantum measurement:

$$P(\text{measure } |φ\rangle \text{ given state } |ψ\rangle) = |\langle φ | ψ \rangle|^2$$

**Physical interpretation:**
- ⟨φ|ψ⟩ is the **probability amplitude**
- |⟨φ|ψ⟩|² is the **probability**
- Amplitudes can interfere; probabilities cannot

### Normalization

A physical quantum state must satisfy:
$$\langle ψ | ψ \rangle = 1$$

This ensures total probability = 1.

### Orthogonality = Distinguishability

Two states are **perfectly distinguishable** iff they're orthogonal:
$$\langle φ | ψ \rangle = 0$$

If you prepare |ψ⟩ and measure "is it |φ⟩?", you get probability 0.

### Example: Qubit States

Standard basis: |0⟩ = (1,0), |1⟩ = (0,1)

Inner products:
- ⟨0|0⟩ = 1 (normalized)
- ⟨1|1⟩ = 1 (normalized)  
- ⟨0|1⟩ = 0 (orthogonal)
- ⟨1|0⟩ = 0 (orthogonal)

Superposition: |+⟩ = (|0⟩ + |1⟩)/√2

- ⟨+|+⟩ = ½⟨0|0⟩ + ½⟨0|1⟩ + ½⟨1|0⟩ + ½⟨1|1⟩ = ½ + 0 + 0 + ½ = 1 ✓
- ⟨0|+⟩ = 1/√2 → P(measure |0⟩) = 1/2 ✓

### Transition Amplitudes

For time evolution:
$$\langle φ | U(t) | ψ \rangle = \text{amplitude to go from } |ψ\rangle \text{ to } |φ\rangle$$

This is fundamental to Feynman's path integral formulation!

---

## ✏️ Worked Examples

### Example 1: Verify Standard Inner Product on ℂ²

Show ⟨z|w⟩ = z₁*w₁ + z₂*w₂ satisfies all axioms.

**Axiom 1 (Conjugate Symmetry):**
$$\langle w | z \rangle = w_1^* z_1 + w_2^* z_2 = (z_1^* w_1)^* + (z_2^* w_2)^* = (z_1^* w_1 + z_2^* w_2)^* = \langle z | w \rangle^*$$ ✓

**Axiom 2 (Additivity in 2nd):**
$$\langle z | w + u \rangle = z_1^*(w_1 + u_1) + z_2^*(w_2 + u_2) = (z_1^* w_1 + z_2^* w_2) + (z_1^* u_1 + z_2^* u_2) = \langle z|w\rangle + \langle z|u\rangle$$ ✓

**Axiom 3 (Homogeneity in 2nd):**
$$\langle z | \alpha w \rangle = z_1^*(\alpha w_1) + z_2^*(\alpha w_2) = \alpha(z_1^* w_1 + z_2^* w_2) = \alpha \langle z | w \rangle$$ ✓

**Axiom 4 (Positive Definiteness):**
$$\langle z | z \rangle = z_1^* z_1 + z_2^* z_2 = |z_1|^2 + |z_2|^2 \geq 0$$
Equals 0 iff |z₁| = |z₂| = 0 iff z₁ = z₂ = 0 iff z = 0 ✓

### Example 2: Orthogonality Check

Are |ψ⟩ = (1, i)/√2 and |φ⟩ = (1, -i)/√2 orthogonal?

$$\langle φ | ψ \rangle = \frac{1}{2}(1^* \cdot 1 + (-i)^* \cdot i) = \frac{1}{2}(1 + i \cdot i) = \frac{1}{2}(1 - 1) = 0$$ ✓

Yes! These are orthogonal (they're the |+⟩ and |-⟩ states in the Y-basis).

### Example 3: Function Space Inner Product

On V = C([0, 2π]), show that sin(x) and cos(x) are orthogonal:

$$\langle \sin, \cos \rangle = \int_0^{2\pi} \sin(x) \cos(x) \, dx = \frac{1}{2}\int_0^{2\pi} \sin(2x) \, dx = \frac{1}{2}\left[-\frac{\cos(2x)}{2}\right]_0^{2\pi} = 0$$ ✓

### Example 4: Antilinearity in First Argument

Verify: ⟨αu|v⟩ = α*⟨u|v⟩

Let |u⟩ = (1, 0), |v⟩ = (1, 1), α = i.

Direct: ⟨αu|v⟩ = ⟨(i, 0)|(1, 1)⟩ = (-i)·1 + 0·1 = -i

Via formula: α*⟨u|v⟩ = (-i)·⟨(1,0)|(1,1)⟩ = (-i)·(1·1 + 0·1) = -i ✓

---

## 📝 Practice Problems

### Level 1: Basic Computations

1. Compute ⟨u|v⟩ for u = (1, 2, 3) and v = (4, 5, 6) in ℝ³.

2. Compute ⟨z|w⟩ for z = (1+i, 2-i) and w = (3, 4i) in ℂ².

3. Verify that ⟨z|z⟩ is real for any z ∈ ℂⁿ.

4. Compute ∫₀^π sin²(x) dx using the function space inner product.

### Level 2: Verification and Properties

5. Show that ⟨u, v⟩ = 2u₁v₁ + 3u₂v₂ defines an inner product on ℝ².

6. Show that ⟨u, v⟩ = u₁v₁ - u₂v₂ does NOT define an inner product on ℝ².

7. Prove the polarization identity for real inner products.

8. For |ψ⟩ = α|0⟩ + β|1⟩, find α and β such that ⟨ψ|ψ⟩ = 1 and ⟨0|ψ⟩ = 1/√2.

### Level 3: Quantum Applications

9. Given |ψ⟩ = (1, 2i, -1)/√6:
   - Verify normalization
   - Find P(measure basis state |2⟩ = (0,0,1))
   - Find P(measure |φ⟩ = (1, 1, 1)/√3)

10. Show that the Hadamard states |+⟩ = (|0⟩+|1⟩)/√2 and |-⟩ = (|0⟩-|1⟩)/√2 are orthonormal.

11. Prove: |⟨u|v⟩|² ≤ ⟨u|u⟩⟨v|v⟩ (Cauchy-Schwarz, just state for now).

12. In the space of polynomials on [-1,1], show that 1 and x are orthogonal with inner product ⟨p,q⟩ = ∫₋₁¹ p(x)q(x)dx.

### Level 4: Proofs

13. Prove that if ⟨u|v⟩ = 0 for all v, then u = 0.

14. Prove the parallelogram law from the inner product axioms.

15. Show that any inner product on ℂⁿ can be written as ⟨u|v⟩ = u†Av for some positive definite Hermitian matrix A.

---

## 📊 Answers and Hints

1. 4 + 10 + 18 = 32
2. (1-i)·3 + (2+i)·4i = 3 - 3i + 8i - 4 = -1 + 5i
3. ⟨z|z⟩ = Σ|zᵢ|² which is a sum of real numbers
4. π/2 (use sin²x = (1-cos2x)/2)
5. Check all 4 axioms; weights 2,3 are positive
6. ⟨(1,1),(1,1)⟩ = 1-1 = 0 but (1,1) ≠ 0; fails positive definiteness
7. Expand ‖u+v‖² and ‖u-v‖² using inner product
8. α = 1/√2, β = e^(iθ)/√2 for any θ (phase freedom!)
9. Verify Σ|cᵢ|² = 1; P = |⟨2|ψ⟩|² = 1/6; P = |⟨φ|ψ⟩|² = |1-1+2i|²/18 = 4/18 = 2/9
10. Compute all four inner products
11. This is the Cauchy-Schwarz inequality (proved tomorrow)
12. ∫₋₁¹ x dx = 0
13. Take v = u
14. Expand both sides using ⟨·,·⟩
15. Define A by Aᵢⱼ = ⟨eᵢ|eⱼ⟩ for standard basis

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Lab 1: Inner Products in NumPy
# ============================================

# Real inner product (dot product)
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Method 1: np.dot
ip1 = np.dot(u, v)
print(f"Real inner product (dot): {ip1}")

# Method 2: @ operator
ip2 = u @ v
print(f"Real inner product (@): {ip2}")

# Complex inner product
z = np.array([1+1j, 2-1j])
w = np.array([3, 4j])

# CORRECT: conjugate first argument
ip_complex = np.vdot(z, w)  # vdot conjugates first arg
print(f"\nComplex inner product ⟨z|w⟩: {ip_complex}")

# WRONG way (no conjugate)
ip_wrong = np.dot(z, w)
print(f"Wrong (no conjugate): {ip_wrong}")

# Manual verification
ip_manual = np.conj(z[0])*w[0] + np.conj(z[1])*w[1]
print(f"Manual calculation: {ip_manual}")

# ============================================
# Lab 2: Qubit Inner Products
# ============================================

# Computational basis
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# Verify orthonormality
print("\n=== Computational Basis ===")
print(f"⟨0|0⟩ = {np.vdot(ket_0, ket_0)}")
print(f"⟨1|1⟩ = {np.vdot(ket_1, ket_1)}")
print(f"⟨0|1⟩ = {np.vdot(ket_0, ket_1)}")
print(f"⟨1|0⟩ = {np.vdot(ket_1, ket_0)}")

# Hadamard basis
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

print("\n=== Hadamard Basis ===")
print(f"|+⟩ = {ket_plus}")
print(f"|-⟩ = {ket_minus}")
print(f"⟨+|+⟩ = {np.vdot(ket_plus, ket_plus):.4f}")
print(f"⟨-|-⟩ = {np.vdot(ket_minus, ket_minus):.4f}")
print(f"⟨+|-⟩ = {np.vdot(ket_plus, ket_minus):.4f}")

# Y-basis
ket_plus_i = (ket_0 + 1j*ket_1) / np.sqrt(2)
ket_minus_i = (ket_0 - 1j*ket_1) / np.sqrt(2)

print("\n=== Y Basis ===")
print(f"|+i⟩ = {ket_plus_i}")
print(f"|-i⟩ = {ket_minus_i}")
print(f"⟨+i|-i⟩ = {np.vdot(ket_plus_i, ket_minus_i):.4f}")

# ============================================
# Lab 3: Probability Calculations
# ============================================

# Arbitrary state
psi = np.array([1, 2j, -1], dtype=complex)
psi = psi / np.linalg.norm(psi)  # Normalize

print("\n=== Measurement Probabilities ===")
print(f"|ψ⟩ = {psi}")
print(f"⟨ψ|ψ⟩ = {np.vdot(psi, psi):.6f}")

# Basis states
basis = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
for i, b in enumerate(basis):
    amp = np.vdot(b, psi)
    prob = np.abs(amp)**2
    print(f"P(|{i}⟩) = |⟨{i}|ψ⟩|² = |{amp:.4f}|² = {prob:.4f}")

print(f"Sum of probabilities: {sum(np.abs(np.vdot(b, psi))**2 for b in basis):.6f}")

# ============================================
# Lab 4: Function Space Inner Product
# ============================================

from scipy.integrate import quad

def inner_product_func(f, g, a, b):
    """Compute ⟨f|g⟩ = ∫_a^b f*(x)g(x)dx"""
    def integrand(x):
        return np.conj(f(x)) * g(x)
    result, _ = quad(lambda x: np.real(integrand(x)), a, b)
    result_imag, _ = quad(lambda x: np.imag(integrand(x)), a, b)
    return result + 1j*result_imag

# Verify sin and cos are orthogonal on [0, 2π]
ip_sin_cos = inner_product_func(np.sin, np.cos, 0, 2*np.pi)
print(f"\n⟨sin|cos⟩ on [0,2π] = {ip_sin_cos:.6f}")

# Norm of sin on [0, 2π]
norm_sin_sq = inner_product_func(np.sin, np.sin, 0, 2*np.pi)
print(f"⟨sin|sin⟩ = {np.real(norm_sin_sq):.6f}")
print(f"||sin|| = {np.sqrt(np.real(norm_sin_sq)):.6f}")

# ============================================
# Lab 5: Visualizing Inner Products
# ============================================

# Inner product as projection
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2D real case
ax = axes[0]
u = np.array([3, 1])
v = np.array([1, 2])
proj_coeff = np.dot(u, v) / np.dot(v, v)
proj = proj_coeff * v

ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, 
          color='blue', label='u')
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
          color='red', label='v')
ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, 
          color='green', label='proj_v(u)')
ax.plot([u[0], proj[0]], [u[1], proj[1]], 'k--', alpha=0.5)
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title(f'⟨u,v⟩ = {np.dot(u,v)}')

# Bloch sphere representation
ax = axes[1]
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)
X = np.sin(THETA) * np.cos(PHI)
Y = np.sin(THETA) * np.sin(PHI)
Z = np.cos(THETA)

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.1, color='blue')

# Plot some states
states = {
    '|0⟩': (0, 0, 1),
    '|1⟩': (0, 0, -1),
    '|+⟩': (1, 0, 0),
    '|-⟩': (-1, 0, 0),
    '|+i⟩': (0, 1, 0),
    '|-i⟩': (0, -1, 0),
}
for name, (x, y, z) in states.items():
    ax.scatter([x], [y], [z], s=50)
    ax.text(x*1.2, y*1.2, z*1.2, name)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Qubit States on Bloch Sphere')

plt.tight_layout()
plt.savefig('inner_product_visualization.png', dpi=150)
plt.show()

print("\n=== Lab Complete ===")
```

### Lab Exercises

1. Write a function to test if two complex vectors are orthogonal.

2. Create a function that normalizes a quantum state.

3. Compute the inner product of Legendre polynomials P₀(x)=1 and P₁(x)=x on [-1,1].

4. Verify that different bases for ℂ² give the same norm for a state.

---

## ✅ Daily Checklist

- [ ] Read Axler 6.A completely
- [ ] Read Shankar 1.3 for physics perspective
- [ ] Write out all inner product axioms (real and complex)
- [ ] Understand conjugate symmetry and sesquilinearity
- [ ] Master Dirac notation: bra, ket, bracket
- [ ] Complete worked examples independently
- [ ] Solve problems 1-8 from practice set
- [ ] Complete computational lab
- [ ] Create flashcards for axioms and QM connections

---

## 📓 Reflection Questions

1. Why must complex inner products be sesquilinear (not bilinear)?

2. What does the inner product represent physically in quantum mechanics?

3. How does the conjugate in ⟨z|w⟩ = Σzᵢ*wᵢ ensure positive definiteness?

4. Why is Dirac notation so powerful for quantum mechanics?

---

## 🔜 Preview: Tomorrow's Topics

**Day 107: Norms and Distance**
- Defining length via inner products
- The norm function ‖v‖ = √⟨v|v⟩
- Triangle inequality
- Cauchy-Schwarz inequality
- Distance and metric spaces

**QM preview:** Normalization condition ‖ψ‖ = 1

---

*"The inner product is to quantum mechanics what the arrow is to classical mechanics."*
— Anonymous physicist
