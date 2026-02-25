# Day 101: Diagonalization

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Diagonalization |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State the diagonalizability criterion
2. Find the diagonalizing matrix P and diagonal matrix D
3. Use diagonalization to compute matrix powers
4. Understand why some matrices aren't diagonalizable
5. Apply diagonalization to solve systems of ODEs
6. Connect to quantum mechanics: diagonal Hamiltonians

---

## 📚 Required Reading

### Primary Text: Axler, "Linear Algebra Done Right" (4th Edition)
- **Section 5.B**: Diagonalizable operators (pp. 147-155)

### Secondary Text: Strang, "Introduction to Linear Algebra"
- **Section 6.2**: Diagonalizing a Matrix (pp. 296-310)

---

## 📖 Core Content: Theory and Concepts

### 1. The Big Picture: Why Diagonalize?

Diagonal matrices are easy:
$$D = \begin{pmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{pmatrix}$$

**Easy computations with diagonal matrices:**
- Powers: $D^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$
- Exponential: $e^D = \text{diag}(e^{\lambda_1}, \ldots, e^{\lambda_n})$
- Inverse: $D^{-1} = \text{diag}(1/\lambda_1, \ldots, 1/\lambda_n)$
- Determinant: $\det(D) = \lambda_1 \cdots \lambda_n$

**Goal:** If A is "similar" to a diagonal matrix D, we can use D for computations.

### 2. Similarity and Diagonalization

**Definition:** Matrices A and B are **similar** if there exists invertible P such that:
$$B = P^{-1}AP$$

Equivalent: $A = PBP^{-1}$

**Definition:** Matrix A is **diagonalizable** if it's similar to a diagonal matrix:
$$D = P^{-1}AP \quad \text{or} \quad A = PDP^{-1}$$

### 3. The Diagonalization Theorem

**Theorem:** An n×n matrix A is diagonalizable if and only if A has n linearly independent eigenvectors.

**Construction when diagonalizable:**
- Let v₁, ..., vₙ be linearly independent eigenvectors
- Let λ₁, ..., λₙ be their corresponding eigenvalues
- Form P with eigenvectors as columns: $P = [v_1 | v_2 | \cdots | v_n]$
- Then $D = P^{-1}AP$ is diagonal with eigenvalues on diagonal

$$D = \begin{pmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{pmatrix}$$

**Why this works:** AP = PD because:
$$A[v_1 | \cdots | v_n] = [Av_1 | \cdots | Av_n] = [\lambda_1 v_1 | \cdots | \lambda_n v_n] = [v_1 | \cdots | v_n] D$$

### 4. Diagonalizability Criteria

**Criterion 1 (Eigenvalue Test):**
A is diagonalizable if and only if:
$$\sum_{\lambda} \text{geometric mult}(\lambda) = n$$

**Criterion 2 (Multiplicity Test):**
A is diagonalizable if and only if for every eigenvalue:
$$\text{geometric multiplicity} = \text{algebraic multiplicity}$$

**Criterion 3 (Distinct Eigenvalues):**
If A has n distinct eigenvalues, then A is diagonalizable.
(This is sufficient but not necessary.)

### 5. Computing Matrix Powers

If $A = PDP^{-1}$, then:
$$A^k = PD^kP^{-1}$$

**Proof:**
$$A^2 = (PDP^{-1})(PDP^{-1}) = PD(P^{-1}P)DP^{-1} = PD^2P^{-1}$$

By induction: $A^k = PD^kP^{-1}$

**Power of diagonal matrix:**
$$D^k = \begin{pmatrix} \lambda_1^k & 0 & \cdots & 0 \\ 0 & \lambda_2^k & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n^k \end{pmatrix}$$

### 6. Matrix Exponential via Diagonalization

**Definition:** The matrix exponential is:
$$e^A = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \cdots = \sum_{k=0}^{\infty} \frac{A^k}{k!}$$

If $A = PDP^{-1}$:
$$e^A = Pe^DP^{-1}$$

where $e^D = \text{diag}(e^{\lambda_1}, \ldots, e^{\lambda_n})$

### 7. Non-Diagonalizable (Defective) Matrices

**Example:** 
$$A = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$$

- Eigenvalue: λ = 2 (algebraic multiplicity 2)
- Eigenspace: E₂ = span{(1, 0)} (geometric multiplicity 1)
- 1 < 2, so A is NOT diagonalizable

**What to do?** Use Jordan normal form (beyond our scope, but:)
$$A = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$$
is already in Jordan form.

---

## 🔬 Quantum Mechanics Connection

### Diagonal Hamiltonians

When the Hamiltonian H is diagonal in some basis, that basis consists of **energy eigenstates**:

$$H = \begin{pmatrix} E_1 & 0 & \cdots & 0 \\ 0 & E_2 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & E_n \end{pmatrix}$$

The basis states |1⟩, |2⟩, ..., |n⟩ are eigenstates with energies E₁, E₂, ..., Eₙ.

### Time Evolution

The time evolution operator is:
$$U(t) = e^{-iHt/\hbar}$$

If H is diagonal:
$$U(t) = \begin{pmatrix} e^{-iE_1 t/\hbar} & 0 & \cdots \\ 0 & e^{-iE_2 t/\hbar} & \cdots \\ \vdots & & \ddots \end{pmatrix}$$

**Physical meaning:** Each energy eigenstate evolves by a phase factor!

### Stationary States

An energy eigenstate |n⟩ evolves as:
$$|n(t)\rangle = e^{-iE_n t/\hbar}|n\rangle$$

This is just a phase — the **probability distribution doesn't change**!
That's why they're called "stationary states."

### General State Evolution

For general state $|\psi(0)\rangle = \sum_n c_n |n\rangle$:

$$|\psi(t)\rangle = \sum_n c_n e^{-iE_n t/\hbar} |n\rangle$$

**The diagonalization lets us solve time evolution easily!**

### Measurement Basis

When you diagonalize an observable A:
$$A = PDP^{-1}$$

The columns of P are the eigenvectors — these form the **measurement basis** for observable A.

---

## ✏️ Worked Examples

### Example 1: Complete Diagonalization

Diagonalize:
$$A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$$

**Step 1: Find eigenvalues**
$$\det(A - \lambda I) = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = 0$$
$$\lambda = 5, 2$$

**Step 2: Find eigenvectors**

For λ = 5:
$$A - 5I = \begin{pmatrix} -1 & 1 \\ 2 & -2 \end{pmatrix}$$
Row reduce: $-x_1 + x_2 = 0$, so $x_1 = x_2$.
$v_1 = (1, 1)$

For λ = 2:
$$A - 2I = \begin{pmatrix} 2 & 1 \\ 2 & 1 \end{pmatrix}$$
Row reduce: $2x_1 + x_2 = 0$, so $x_2 = -2x_1$.
$v_2 = (1, -2)$

**Step 3: Form P and D**
$$P = \begin{pmatrix} 1 & 1 \\ 1 & -2 \end{pmatrix}, \quad D = \begin{pmatrix} 5 & 0 \\ 0 & 2 \end{pmatrix}$$

**Step 4: Verify**
$$P^{-1} = \frac{1}{-3}\begin{pmatrix} -2 & -1 \\ -1 & 1 \end{pmatrix} = \begin{pmatrix} 2/3 & 1/3 \\ 1/3 & -1/3 \end{pmatrix}$$

$$P^{-1}AP = \begin{pmatrix} 2/3 & 1/3 \\ 1/3 & -1/3 \end{pmatrix}\begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -2 \end{pmatrix} = D$$ ✓

### Example 2: Computing A¹⁰⁰

Using the diagonalization from Example 1:

$$A^{100} = PD^{100}P^{-1}$$

$$D^{100} = \begin{pmatrix} 5^{100} & 0 \\ 0 & 2^{100} \end{pmatrix}$$

$$A^{100} = \begin{pmatrix} 1 & 1 \\ 1 & -2 \end{pmatrix}\begin{pmatrix} 5^{100} & 0 \\ 0 & 2^{100} \end{pmatrix}\begin{pmatrix} 2/3 & 1/3 \\ 1/3 & -1/3 \end{pmatrix}$$

$$= \frac{1}{3}\begin{pmatrix} 2 \cdot 5^{100} + 2^{100} & 5^{100} - 2^{100} \\ 2 \cdot 5^{100} - 2 \cdot 2^{100} & 5^{100} + 2 \cdot 2^{100} \end{pmatrix}$$

### Example 3: Matrix Exponential

Compute $e^{At}$ for:
$$A = \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix}$$

**Step 1: Diagonalize A**

Eigenvalues: λ² + 3λ + 2 = 0 → λ = -1, -2

For λ = -1: v₁ = (1, -1)
For λ = -2: v₂ = (1, -2)

$$P = \begin{pmatrix} 1 & 1 \\ -1 & -2 \end{pmatrix}, \quad D = \begin{pmatrix} -1 & 0 \\ 0 & -2 \end{pmatrix}$$

**Step 2: Compute e^{Dt}**
$$e^{Dt} = \begin{pmatrix} e^{-t} & 0 \\ 0 & e^{-2t} \end{pmatrix}$$

**Step 3: Compute e^{At}**
$$P^{-1} = \begin{pmatrix} 2 & 1 \\ -1 & -1 \end{pmatrix}$$

$$e^{At} = Pe^{Dt}P^{-1} = \begin{pmatrix} 2e^{-t} - e^{-2t} & e^{-t} - e^{-2t} \\ -2e^{-t} + 2e^{-2t} & -e^{-t} + 2e^{-2t} \end{pmatrix}$$

**Application:** This solves the ODE system $\mathbf{x}' = A\mathbf{x}$!

### Example 4: Quantum Two-Level System

A qubit with Hamiltonian:
$$H = \frac{\hbar\omega}{2}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \frac{\hbar\omega}{2}\sigma_z$$

H is already diagonal! Energy eigenstates: |0⟩, |1⟩.

**Time evolution:**
$$U(t) = e^{-iHt/\hbar} = \begin{pmatrix} e^{-i\omega t/2} & 0 \\ 0 & e^{i\omega t/2} \end{pmatrix}$$

For initial state $|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:

$$|\psi(t)\rangle = \frac{1}{\sqrt{2}}(e^{-i\omega t/2}|0\rangle + e^{i\omega t/2}|1\rangle)$$

The relative phase between |0⟩ and |1⟩ oscillates! This is **Larmor precession**.

---

## 📝 Practice Problems

### Level 1: Basic Diagonalization
1. Diagonalize A = [[3, 1], [0, 2]].

2. Diagonalize A = [[1, 0], [0, 1]]. (Hint: already diagonal!)

3. Is A = [[1, 1], [0, 1]] diagonalizable? Why or why not?

4. Diagonalize A = [[0, -1], [1, 0]]. (Note: complex eigenvalues)

### Level 2: Applications
5. Use diagonalization to compute A¹⁰ for A = [[2, 1], [0, 3]].

6. Compute e^A for A = [[0, π], [-π, 0]].

7. Find a matrix A such that A² = [[4, 0], [0, 9]].

8. Solve the system x' = 3x + y, y' = 2y using diagonalization.

### Level 3: Theory
9. Prove: If A is diagonalizable and all eigenvalues are 0, then A = 0.

10. Prove: A and A^T have the same eigenvalues.

11. If A = PDP⁻¹, express A⁻¹ in terms of P, D, P⁻¹.

12. Prove: Similar matrices have the same characteristic polynomial.

### Level 4: Quantum Applications
13. The Pauli-X gate is X = [[0,1],[1,0]]. Diagonalize it and find e^{-iXt}.

14. For Hamiltonian H = ℏω(a†a + 1/2), the matrix representation in Fock states |0⟩, |1⟩, |2⟩ is:
    H = ℏω diag(1/2, 3/2, 5/2)
    Find U(t) = e^{-iHt/ℏ}.

15. A spin-1/2 in a magnetic field has H = -γB·σ_z. If the initial state is |+x⟩ = (|↑⟩ + |↓⟩)/√2, find |ψ(t)⟩.

---

## 💻 Evening Computational Lab (1 hour)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ============================================
# Lab 1: Diagonalization Procedure
# ============================================

def diagonalize(A, verbose=True):
    """
    Diagonalize matrix A.
    Returns P, D such that A = P @ D @ P^{-1}
    """
    if verbose:
        print("=== Diagonalization ===\n")
        print(f"Matrix A:\n{A}\n")
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    if verbose:
        print(f"Eigenvalues: {eigenvalues}")
        print(f"\nEigenvectors (as columns of P):\n{eigenvectors}\n")
    
    # P = matrix of eigenvectors (columns)
    P = eigenvectors
    
    # D = diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)
    
    # Verify diagonalization
    P_inv = np.linalg.inv(P)
    A_reconstructed = P @ D @ P_inv
    
    if verbose:
        print(f"D = P^(-1) A P:\n{P_inv @ A @ P}\n")
        print(f"Verification A = P D P^(-1):\n{A_reconstructed}")
        print(f"\nReconstruction error: {np.max(np.abs(A - A_reconstructed)):.2e}")
    
    return P, D, P_inv

# Test
A = np.array([[4, 1],
              [2, 3]])
P, D, P_inv = diagonalize(A)

# ============================================
# Lab 2: Matrix Powers via Diagonalization
# ============================================

def matrix_power_diag(A, k):
    """Compute A^k using diagonalization"""
    P, D, P_inv = diagonalize(A, verbose=False)
    D_k = np.diag(np.diag(D) ** k)
    return P @ D_k @ P_inv

print("\n=== Matrix Powers ===\n")
A = np.array([[2, 1],
              [1, 2]])

for k in [2, 10, 100]:
    A_k_diag = matrix_power_diag(A, k)
    A_k_direct = np.linalg.matrix_power(A, k)
    
    print(f"A^{k} via diagonalization:\n{A_k_diag}")
    print(f"Direct computation:\n{A_k_direct}")
    print(f"Difference: {np.max(np.abs(A_k_diag - A_k_direct)):.2e}\n")

# ============================================
# Lab 3: Matrix Exponential
# ============================================

def matrix_exp_diag(A, t=1):
    """Compute e^(At) using diagonalization"""
    P, D, P_inv = diagonalize(A, verbose=False)
    exp_D = np.diag(np.exp(np.diag(D) * t))
    return P @ exp_D @ P_inv

print("=== Matrix Exponential ===\n")

# Simple example
A = np.array([[0, -1],
              [1, 0]])  # Rotation generator

print(f"Matrix A (rotation generator):\n{A}\n")

t_values = [0, np.pi/4, np.pi/2, np.pi]
for t in t_values:
    exp_At = matrix_exp_diag(A, t)
    print(f"e^(A*{t:.4f}):\n{exp_At.real}\n")  # Real part (imaginary should be ~0)

# Compare with scipy
print("Verification with scipy.linalg.expm:")
print(f"scipy expm(A*π/2):\n{expm(A * np.pi/2)}")

# ============================================
# Lab 4: Quantum Time Evolution
# ============================================

print("\n=== Quantum Time Evolution ===\n")

# Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

# Hamiltonian H = (ℏω/2) σ_z (set ℏω = 1 for simplicity)
H = 0.5 * sigma_z
print(f"Hamiltonian H = (1/2)σ_z:\n{H}\n")

# Time evolution operator U(t) = e^{-iHt}
def time_evolution_operator(H, t):
    return expm(-1j * H * t)

# Initial state: |+x⟩ = (|0⟩ + |1⟩)/√2
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
psi_0 = (ket_0 + ket_1) / np.sqrt(2)

print(f"Initial state |ψ(0)⟩ = |+x⟩:\n{psi_0}\n")

# Evolve and track
t_vals = np.linspace(0, 4*np.pi, 100)
prob_0 = []
prob_1 = []

for t in t_vals:
    U = time_evolution_operator(H, t)
    psi_t = U @ psi_0
    prob_0.append(np.abs(psi_t[0])**2)
    prob_1.append(np.abs(psi_t[1])**2)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_vals, prob_0, 'b-', label='P(|0⟩)')
plt.plot(t_vals, prob_1, 'r-', label='P(|1⟩)')
plt.xlabel('Time t')
plt.ylabel('Probability')
plt.title('Probability Evolution under H = σ_z/2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Bloch sphere x-component: ⟨σ_x⟩
expect_x = []
expect_y = []
expect_z = []

sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

for t in t_vals:
    U = time_evolution_operator(H, t)
    psi_t = U @ psi_0
    expect_x.append(np.real(np.vdot(psi_t, sigma_x @ psi_t)))
    expect_y.append(np.real(np.vdot(psi_t, sigma_y @ psi_t)))
    expect_z.append(np.real(np.vdot(psi_t, sigma_z @ psi_t)))

plt.plot(t_vals, expect_x, 'r-', label='⟨σ_x⟩')
plt.plot(t_vals, expect_y, 'g-', label='⟨σ_y⟩')
plt.plot(t_vals, expect_z, 'b-', label='⟨σ_z⟩')
plt.xlabel('Time t')
plt.ylabel('Expectation value')
plt.title('Bloch Vector Components (Larmor Precession)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quantum_evolution.png', dpi=150)
plt.show()

# ============================================
# Lab 5: Solving ODEs via Diagonalization
# ============================================

print("\n=== ODE Solving ===\n")

# System: x' = Ax where A = [[0, 1], [-2, -3]]
A = np.array([[0, 1],
              [-2, -3]])

print(f"ODE system x' = Ax with A =\n{A}\n")

# Diagonalize
P, D, P_inv = diagonalize(A)

# Solution: x(t) = e^{At} x(0)
def solve_linear_ode(A, x0, t_max, num_points=100):
    """Solve x' = Ax with initial condition x(0) = x0"""
    t_vals = np.linspace(0, t_max, num_points)
    solution = []
    
    for t in t_vals:
        x_t = expm(A * t) @ x0
        solution.append(x_t)
    
    return t_vals, np.array(solution)

# Initial condition
x0 = np.array([1, 0])

t_vals, solution = solve_linear_ode(A, x0, t_max=5)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t_vals, solution[:, 0], 'b-', label='x₁(t)')
plt.plot(t_vals, solution[:, 1], 'r-', label='x₂(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution to x\' = Ax')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(solution[:, 0], solution[:, 1], 'b-')
plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
plt.plot(solution[-1, 0], solution[-1, 1], 'ro', markersize=10, label='End')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Phase Portrait')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig('ode_solution.png', dpi=150)
plt.show()

print("Eigenvalues of A:", np.linalg.eigvals(A))
print("(Negative real parts → solution decays to 0)")

print("\n=== Lab Complete ===")
```

---

## ✅ Daily Checklist

- [ ] Read Axler 5.B and Strang 6.2
- [ ] Understand diagonalizability criteria
- [ ] Manually diagonalize 2 matrices
- [ ] Compute matrix power via diagonalization
- [ ] Complete quantum time evolution example
- [ ] Run computational lab
- [ ] Understand connection to ODE solutions

---

## 🔜 Preview: Tomorrow's Topics

**Day 102: Spectral Theorem (Preview) and Applications**

Tomorrow we'll explore:
- Why Hermitian/symmetric matrices are always diagonalizable
- Orthogonal diagonalization
- Applications to quadratic forms
- QM: Spectral decomposition of observables

---

*"The spectral theorem is the single most important theorem in linear algebra."*
— Sheldon Axler
