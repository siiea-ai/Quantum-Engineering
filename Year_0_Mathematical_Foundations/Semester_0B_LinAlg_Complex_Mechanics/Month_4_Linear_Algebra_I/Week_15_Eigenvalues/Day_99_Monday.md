# Day 99: Introduction to Eigenvalues and Eigenvectors

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Eigenvalue Fundamentals |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define eigenvalues and eigenvectors precisely
2. Understand the geometric meaning of eigenvectors (invariant directions)
3. Find eigenvalues and eigenvectors for 2×2 matrices
4. Recognize the eigenvalue equation Av = λv
5. Connect eigenvalues to quantum measurement outcomes
6. Begin understanding why eigenstates are special in QM

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 5.A**: Eigenvalues and Eigenvectors (pp. 132-140)
- Focus especially on: Definition 5.5, Examples 5.6-5.10

### Secondary Text: Strang, "Introduction to Linear Algebra"
- **Section 6.1**: Introduction to Eigenvalues (pp. 283-295)
- The computational approach complements Axler

### Supplementary
- **Shankar, Chapter 1.8**: The Eigenvalue Problem (QM perspective)

---

## 🎬 Video Resources

### 3Blue1Brown: Essence of Linear Algebra
- **Chapter 14**: Eigenvectors and Eigenvalues
- URL: https://www.youtube.com/watch?v=PFDu9oVAE-g
- Duration: 17 minutes
- **HIGHLY RECOMMENDED** — Watch before reading!

### MIT OCW 18.06 (Gilbert Strang)
- **Lecture 21**: Eigenvalues and Eigenvectors
- Focus on geometric interpretation

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: Why Eigenvalues Matter

Before the definition, consider this question:

**Given a linear transformation T, which vectors v have the property that T(v) is parallel to v?**

That is, which vectors only get scaled (not rotated) by T?

**Examples where this matters:**
| Domain | Application |
|--------|-------------|
| Physics | Stable configurations, principal axes |
| Vibrations | Normal modes, natural frequencies |
| Quantum Mechanics | Measurement outcomes, energy levels |
| Data Science | Principal Component Analysis (PCA) |
| Differential Equations | Decoupled solutions |

These "special directions" are eigenvectors, and the scaling factors are eigenvalues.

### 2. The Definition

**Definition:** Let T: V → V be a linear operator (same domain and codomain). A scalar λ ∈ F is called an **eigenvalue** of T if there exists a nonzero vector v ∈ V such that:

$$T(v) = \lambda v$$

The vector v is called an **eigenvector** of T corresponding to eigenvalue λ.

**For matrices:** If A is an n×n matrix, λ is an eigenvalue of A if:

$$Av = \lambda v$$

for some nonzero v ∈ Fⁿ.

**Critical Points:**
1. **v must be nonzero!** (Otherwise any λ works for v = 0)
2. **λ can be zero!** (Then Av = 0, so v ∈ ker(A))
3. **Same domain and codomain** — eigenvalues only for operators V → V
4. **The equation Av = λv can be rewritten as (A - λI)v = 0**

### 3. Geometric Interpretation

**Eigenvector:** A direction that T maps to itself (possibly reversed if λ < 0)

**Eigenvalue:** The factor by which T scales vectors in that direction

| λ | Geometric Effect |
|---|------------------|
| λ > 1 | Stretches the eigenvector direction |
| 0 < λ < 1 | Compresses the eigenvector direction |
| λ = 1 | Leaves eigenvector direction unchanged |
| λ = 0 | Collapses eigenvector direction to 0 |
| λ < 0 | Reverses and scales the eigenvector direction |
| λ ∈ ℂ | Rotation (in complex spaces) |

**Visual:** Imagine a 2D transformation. Most vectors get both stretched AND rotated. Eigenvectors are the special directions that only get stretched (or flipped).

### 4. Finding Eigenvalues: The Characteristic Equation

Starting from Av = λv:
$$Av = \lambda v$$
$$Av - \lambda v = 0$$
$$Av - \lambda I v = 0$$
$$(A - \lambda I)v = 0$$

This is a homogeneous system. It has a nonzero solution v ⟺ A - λI is not invertible ⟺ **det(A - λI) = 0**

**Definition:** The **characteristic polynomial** of A is:
$$p(\lambda) = \det(A - \lambda I)$$

**Eigenvalues are the roots of the characteristic polynomial.**

### 5. The 2×2 Case: Complete Analysis

For A = [[a, b], [c, d]], we have:

$$A - \lambda I = \begin{pmatrix} a - \lambda & b \\ c & d - \lambda \end{pmatrix}$$

$$\det(A - \lambda I) = (a-\lambda)(d-\lambda) - bc$$
$$= \lambda^2 - (a+d)\lambda + (ad - bc)$$
$$= \lambda^2 - \text{tr}(A)\lambda + \det(A)$$

**Key formula for 2×2:**
$$p(\lambda) = \lambda^2 - \text{tr}(A)\lambda + \det(A)$$

where tr(A) = a + d (trace = sum of diagonal entries).

**Using the quadratic formula:**
$$\lambda = \frac{\text{tr}(A) \pm \sqrt{\text{tr}(A)^2 - 4\det(A)}}{2}$$

### 6. Finding Eigenvectors

Once you have eigenvalue λ, find eigenvectors by solving:
$$(A - \lambda I)v = 0$$

The solution space is ker(A - λI), called the **eigenspace** for λ.

**Procedure:**
1. Form A - λI
2. Row reduce to echelon form
3. Find the general solution (parametric form)
4. The nonzero vectors in this solution are eigenvectors

### 7. Eigenspaces

**Definition:** The **eigenspace** E_λ corresponding to eigenvalue λ is:
$$E_\lambda = \ker(A - \lambda I) = \{v \in V : Av = \lambda v\}$$

**Fact:** E_λ is a subspace of V.

**Proof:** 
- 0 ∈ E_λ (but 0 is not an eigenvector by convention)
- If v, w ∈ E_λ: A(v+w) = Av + Aw = λv + λw = λ(v+w), so v+w ∈ E_λ
- If v ∈ E_λ, c ∈ F: A(cv) = cAv = c(λv) = λ(cv), so cv ∈ E_λ

**Geometric multiplicity** of λ = dim(E_λ) = number of linearly independent eigenvectors.

---

## 🔬 Quantum Mechanics Connection

### Eigenvalues = Measurement Outcomes

In quantum mechanics, **observables** (measurable quantities) are represented by **Hermitian operators** on the state space.

**The Measurement Postulate:** When you measure an observable Â, the only possible outcomes are the eigenvalues of Â.

| QM Concept | Linear Algebra |
|------------|----------------|
| Observable | Hermitian operator Â |
| Possible measurement values | Eigenvalues λᵢ of Â |
| Definite-value states | Eigenvectors \|λᵢ⟩ of Â |
| State after measurement | Collapses to eigenvector |

### Example: Spin-z Observable

The spin-z operator for a spin-1/2 particle (in units of ℏ/2):

$$S_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Finding eigenvalues:**
$$\det(S_z - \lambda I) = (1-\lambda)(-1-\lambda) = \lambda^2 - 1 = 0$$
$$\lambda = \pm 1$$

**Interpretation:** Measuring spin along z can give only +1 or -1 (in units of ℏ/2).

**Finding eigenvectors:**

For λ = +1:
$$(S_z - I)v = \begin{pmatrix} 0 & 0 \\ 0 & -2 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0$$
$$v_2 = 0 \implies v = \begin{pmatrix} 1 \\ 0 \end{pmatrix} = |0\rangle = |\uparrow\rangle$$

For λ = -1:
$$(S_z + I)v = \begin{pmatrix} 2 & 0 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0$$
$$v_1 = 0 \implies v = \begin{pmatrix} 0 \\ 1 \end{pmatrix} = |1\rangle = |\downarrow\rangle$$

**Physical meaning:**
- |↑⟩ = spin up = definite +1 eigenstate
- |↓⟩ = spin down = definite -1 eigenstate

### Energy Eigenstates

The **Hamiltonian** Ĥ is the energy operator. Its eigenvalue equation:
$$\hat{H}|\psi_n\rangle = E_n|\psi_n\rangle$$

- Eigenvalues Eₙ = possible energy levels
- Eigenstates |ψₙ⟩ = stationary states
- These are the "atomic orbitals" in chemistry!

### Why Eigenstates Are Special

If |ψ⟩ is an eigenstate of observable Â with eigenvalue λ:
1. Measuring  in state |ψ⟩ gives λ with **certainty** (probability 1)
2. The state doesn't change after measurement
3. Time evolution: |ψ(t)⟩ = e^{-iEt/ℏ}|ψ⟩ (just a phase!)

**Contrast:** Superposition states have uncertain measurement outcomes.

---

## ✏️ Worked Examples

### Example 1: Diagonal Matrix

Find eigenvalues and eigenvectors of:
$$A = \begin{pmatrix} 3 & 0 \\ 0 & -2 \end{pmatrix}$$

**Solution:**

Characteristic polynomial:
$$\det(A - \lambda I) = (3-\lambda)(-2-\lambda) = 0$$
$$\lambda_1 = 3, \quad \lambda_2 = -2$$

For λ₁ = 3:
$$A - 3I = \begin{pmatrix} 0 & 0 \\ 0 & -5 \end{pmatrix}$$
Row reduce: v₂ = 0, v₁ free.
$$E_3 = \text{span}\left\{\begin{pmatrix} 1 \\ 0 \end{pmatrix}\right\} = \text{span}\{e_1\}$$

For λ₂ = -2:
$$A + 2I = \begin{pmatrix} 5 & 0 \\ 0 & 0 \end{pmatrix}$$
Row reduce: v₁ = 0, v₂ free.
$$E_{-2} = \text{span}\left\{\begin{pmatrix} 0 \\ 1 \end{pmatrix}\right\} = \text{span}\{e_2\}$$

**Key insight:** For diagonal matrices, eigenvalues are diagonal entries, eigenvectors are standard basis vectors!

### Example 2: Rotation Matrix (No Real Eigenvalues)

Find eigenvalues of the 90° rotation:
$$R = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

**Solution:**

$$\det(R - \lambda I) = (0-\lambda)(0-\lambda) - (-1)(1) = \lambda^2 + 1 = 0$$
$$\lambda = \pm i$$

**No real eigenvalues!** In ℝ², rotation has no invariant lines.

In ℂ²:
- λ₁ = i: eigenvector (1, -i)
- λ₂ = -i: eigenvector (1, i)

**Lesson:** Over ℝ, not all matrices have eigenvalues. Over ℂ, they always do (Fundamental Theorem of Algebra).

### Example 3: Repeated Eigenvalues

$$A = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$$

**Eigenvalues:**
$$\det(A - \lambda I) = (2-\lambda)^2 = 0$$
$$\lambda = 2 \text{ (multiplicity 2)}$$

**Eigenvectors for λ = 2:**
$$A - 2I = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$$
Row reduces to v₂ = 0, v₁ free.
$$E_2 = \text{span}\left\{\begin{pmatrix} 1 \\ 0 \end{pmatrix}\right\}$$

**Only ONE independent eigenvector!** 

This matrix is called **defective** — it cannot be diagonalized.

### Example 4: Hadamard Gate

Find eigenvalues of the Hadamard gate:
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Solution:**

tr(H) = (1 + (-1))/√2 = 0
det(H) = (1·(-1) - 1·1)/2 = -1

$$\lambda^2 - 0 \cdot \lambda + (-1) = 0$$
$$\lambda^2 = 1$$
$$\lambda = \pm 1$$

For λ = 1:
$$H - I = \frac{1}{\sqrt{2}}\begin{pmatrix} 1-\sqrt{2} & 1 \\ 1 & -1-\sqrt{2} \end{pmatrix}$$
Eigenvector: ∝ (1 + √2, 1) or normalized: |+⟩ direction variant

For λ = -1:
Eigenvector: ∝ (1 - √2, 1) or normalized: |-⟩ direction variant

**Physical meaning:** H² = I, so H is its own inverse. Eigenvalues ±1 confirm this (λ² = 1).

---

## 📝 Practice Problems

### Level 1: Basic Computation
1. Find the eigenvalues of A = [[4, 2], [1, 3]].

2. For the matrix in (1), find the eigenvectors.

3. Find eigenvalues of the projection P = [[1, 0], [0, 0]].

4. What are the eigenvalues of the identity matrix I_n?

### Level 2: Working with Eigenspaces
5. Find all eigenvalues and a basis for each eigenspace:
   A = [[1, 2], [0, 1]]

6. Show that 0 is an eigenvalue of A ⟺ A is not invertible.

7. If λ is an eigenvalue of A, what are the eigenvalues of A²? Of A + 3I?

8. Find eigenvalues of the Pauli-X matrix [[0,1],[1,0]].

### Level 3: Proofs and Theory
9. Prove: Eigenvectors corresponding to distinct eigenvalues are linearly independent.

10. Prove: If A is upper triangular, its eigenvalues are its diagonal entries.

11. Let A be n×n. Prove: det(A) = product of all eigenvalues (counted with multiplicity).

12. Prove: tr(A) = sum of all eigenvalues (counted with multiplicity).

### Level 4: Quantum Applications
13. The Pauli-Y matrix is Y = [[0, -i], [i, 0]]. Find its eigenvalues and eigenvectors.

14. Show that the eigenvalues of a Hermitian matrix are real.

15. For the harmonic oscillator Hamiltonian, energy eigenvalues are E_n = ℏω(n + 1/2). What does this tell you about the spacing of energy levels?

---

## 📊 Answers and Hints

1. tr = 7, det = 10. λ² - 7λ + 10 = 0. λ = 2, 5.

2. For λ=2: solve (A-2I)v=0. For λ=5: solve (A-5I)v=0.

3. λ = 0 (for (0,1)) and λ = 1 (for (1,0)).

4. All eigenvalues are 1, with multiplicity n. Every nonzero vector is an eigenvector.

5. λ = 1 (multiplicity 2), but E₁ = span{(1,0)}, dimension 1. Defective!

6. 0 eigenvalue ⟹ (A-0I)v = Av = 0 has nonzero solution ⟹ ker(A) ≠ {0} ⟹ A not invertible.

7. A²: eigenvalues λ². A + 3I: eigenvalues λ + 3.

8. det = -1, tr = 0. λ² - 1 = 0, so λ = ±1.

9. Suppose v₁,...,vₖ are eigenvectors for λ₁,...,λₖ (distinct). Prove by induction.

10. det(A - λI) = product of (aᵢᵢ - λ). Roots are the diagonal entries.

11-15. See detailed solutions in review session.

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Lab 1: Finding Eigenvalues and Eigenvectors
# ============================================

def analyze_eigenstructure(A, name="A"):
    """Complete eigenvalue analysis of a matrix"""
    print(f"=== Eigenstructure of {name} ===\n")
    print(f"Matrix:\n{A}\n")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"Eigenvalues: {eigenvalues}")
    print(f"\nEigenvectors (as columns):\n{eigenvectors}\n")
    
    # Verify each eigenpair
    print("Verification (Av = λv):")
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = A @ v
        lam_v = lam * v
        error = np.linalg.norm(Av - lam_v)
        print(f"  λ_{i+1} = {lam:.4f}")
        print(f"  v_{i+1} = {v}")
        print(f"  ||Av - λv|| = {error:.2e}")
        print()
    
    # Characteristic polynomial coefficients
    n = A.shape[0]
    print(f"Trace: {np.trace(A):.4f} (= sum of eigenvalues: {np.sum(eigenvalues):.4f})")
    print(f"Determinant: {np.linalg.det(A):.4f} (= product of eigenvalues: {np.prod(eigenvalues):.4f})")
    
    return eigenvalues, eigenvectors

# Test matrices
A1 = np.array([[4, 2],
               [1, 3]])
analyze_eigenstructure(A1, "A1")

# Diagonal matrix
A2 = np.diag([3, -2, 5])
analyze_eigenstructure(A2, "Diagonal")

# Rotation matrix (no real eigenvalues over R)
theta = np.pi / 4  # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
analyze_eigenstructure(R, "Rotation 45°")

# ============================================
# Lab 2: Quantum Gates - Eigenanalysis
# ============================================

print("\n" + "="*60)
print("QUANTUM GATE EIGENSTRUCTURE")
print("="*60 + "\n")

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

gates = [("Pauli X", X), ("Pauli Y", Y), ("Pauli Z", Z), ("Hadamard", H)]

for name, gate in gates:
    print(f"--- {name} ---")
    evals, evecs = np.linalg.eig(gate)
    print(f"Eigenvalues: {evals}")
    for i, (lam, v) in enumerate(zip(evals, evecs.T)):
        print(f"  λ = {lam:.4f}, |v⟩ = {v}")
    print()

# ============================================
# Lab 3: Visualizing Eigenvectors
# ============================================

def visualize_eigen_2d(A, title="Transformation"):
    """Visualize how A transforms vectors, highlighting eigendirections"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(A)
    
    # Only proceed if eigenvalues are real
    if not np.allclose(evals.imag, 0):
        print(f"Complex eigenvalues for {title} - skipping 2D visualization")
        return
    
    evals = evals.real
    evecs = evecs.real
    
    # Create a unit circle of vectors
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Transform the circle
    transformed = A @ circle
    
    # Left plot: Original space
    ax1 = axes[0]
    ax1.plot(circle[0], circle[1], 'b-', linewidth=1.5, label='Unit circle')
    
    # Draw eigenvectors
    colors = ['red', 'green']
    for i, (lam, v) in enumerate(zip(evals, evecs.T)):
        v_norm = v / np.linalg.norm(v)
        ax1.arrow(0, 0, v_norm[0]*0.9, v_norm[1]*0.9, head_width=0.1, 
                  head_length=0.05, fc=colors[i], ec=colors[i], linewidth=2)
        ax1.text(v_norm[0]*1.1, v_norm[1]*1.1, f'v{i+1} (λ={lam:.2f})', fontsize=10)
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Original Space with Eigenvectors')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Right plot: Transformed space
    ax2 = axes[1]
    ax2.plot(transformed[0], transformed[1], 'b-', linewidth=1.5, label='Transformed')
    
    # Draw transformed eigenvectors
    for i, (lam, v) in enumerate(zip(evals, evecs.T)):
        v_norm = v / np.linalg.norm(v)
        Av = A @ v_norm
        ax2.arrow(0, 0, Av[0]*0.9, Av[1]*0.9, head_width=0.1, 
                  head_length=0.05, fc=colors[i], ec=colors[i], linewidth=2)
        ax2.text(Av[0]*1.1, Av[1]*1.1, f'Av{i+1} = {lam:.2f}v{i+1}', fontsize=10)
    
    max_val = max(np.max(np.abs(transformed)), 1.5)
    ax2.set_xlim(-max_val*1.2, max_val*1.2)
    ax2.set_ylim(-max_val*1.2, max_val*1.2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'After {title}')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    fig.suptitle(f'{title}: Eigenvectors stay on their lines!', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'eigen_{title.replace(" ", "_").lower()}.png', dpi=150)
    plt.show()

# Visualize various transformations
matrices = [
    (np.array([[2, 1], [0, 1]]), "Shear-like"),
    (np.array([[2, 0], [0, 0.5]]), "Scaling"),
    (np.array([[0.5, 0.5], [0.5, 0.5]]), "Projection"),
    (np.array([[1, 2], [2, 1]]), "Symmetric"),
]

for A, name in matrices:
    visualize_eigen_2d(A, name)

# ============================================
# Lab 4: Power Method for Dominant Eigenvalue
# ============================================

def power_method(A, num_iterations=50, tol=1e-10):
    """
    Find the dominant eigenvalue using the power method.
    
    The power method iteratively computes A^k v, which converges
    to the eigenvector with largest |eigenvalue|.
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    eigenvalue_history = []
    
    for i in range(num_iterations):
        # Apply A
        Av = A @ v
        
        # Estimate eigenvalue using Rayleigh quotient
        eigenvalue = np.dot(v, Av) / np.dot(v, v)
        eigenvalue_history.append(eigenvalue)
        
        # Normalize
        v_new = Av / np.linalg.norm(Av)
        
        # Check convergence
        if np.linalg.norm(v_new - v) < tol:
            print(f"Converged after {i+1} iterations")
            break
        
        v = v_new
    
    return eigenvalue, v, eigenvalue_history

A = np.array([[4, 1],
              [2, 3]])

print("\n=== Power Method Demo ===")
print(f"Matrix:\n{A}\n")

# True eigenvalues
true_evals = np.linalg.eigvals(A)
print(f"True eigenvalues: {true_evals}")

# Power method
dominant_eval, dominant_evec, history = power_method(A)
print(f"\nPower method result:")
print(f"  Dominant eigenvalue: {dominant_eval:.6f}")
print(f"  Dominant eigenvector: {dominant_evec}")

# Plot convergence
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history, 'b-o', markersize=3)
plt.axhline(y=max(true_evals.real), color='r', linestyle='--', label='True λ_max')
plt.xlabel('Iteration')
plt.ylabel('Eigenvalue estimate')
plt.title('Power Method Convergence')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
errors = np.abs(np.array(history) - max(true_evals.real))
plt.semilogy(errors, 'b-o', markersize=3)
plt.xlabel('Iteration')
plt.ylabel('Error |λ_est - λ_true|')
plt.title('Power Method Error (log scale)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('power_method_convergence.png', dpi=150)
plt.show()

print("\n=== Lab Complete ===")
```

### Lab Exercises

1. **Eigenvalue sensitivity:** Create a matrix with eigenvalues 1 and 2. Add small random perturbations and observe how eigenvalues change.

2. **Defective matrices:** Create the matrix [[1,1],[0,1]] and verify it has only one eigenvalue with one eigenvector. Try to diagonalize it (you can't!).

3. **Quantum state evolution:** Given H = σ_z (Pauli-Z), compute e^{-iHt} for various t values and observe how eigenstates evolve.

4. **Implement inverse power method:** Modify the power method to find the smallest eigenvalue.

---

## ✅ Daily Checklist

- [ ] Watch 3Blue1Brown eigenvector video
- [ ] Read Axler 5.A completely
- [ ] Practice 2×2 characteristic polynomial by hand
- [ ] Solve problems 1-8 from practice set
- [ ] Attempt at least one Level 3 proof
- [ ] Complete computational lab
- [ ] Create flashcards for:
  - Eigenvalue definition
  - Eigenvector definition  
  - Characteristic polynomial formula (2×2)
  - Connection to QM measurement
- [ ] Write reflection on eigenvalues in quantum mechanics

---

## 📓 Reflection Questions

1. Why can't non-square matrices have eigenvalues?

2. In what sense are eigenvectors "natural" for a transformation?

3. Why are real eigenvalues important for quantum observables?

4. If you rotate a vector by 90°, why doesn't it have real eigenvectors?

---

## 🔜 Preview: Tomorrow's Topics

**Day 100: The Characteristic Polynomial and Determinants**

Tomorrow we'll explore:
- Determinants in depth
- Characteristic polynomial for n×n matrices
- Algebraic vs geometric multiplicity
- The Cayley-Hamilton theorem

**Preparation:** Review determinant computation for 3×3 matrices.

---

*"The eigenvalues are the truly important quantities... they capture the essence of what a linear transformation does."*
— Gilbert Strang
