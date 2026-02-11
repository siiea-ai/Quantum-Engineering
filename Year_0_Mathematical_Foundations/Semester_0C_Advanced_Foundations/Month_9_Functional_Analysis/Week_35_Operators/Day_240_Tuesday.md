# Day 240: Operator Norm and B(H)

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Operator norm properties, B(H) as Banach space |
| Afternoon | 3 hours | Problems: Norm computations, Banach algebra structure |
| Evening | 2 hours | Computational lab: Operator norm calculations |

## Learning Objectives

By the end of today, you will be able to:

1. **Compute** operator norms using various characterizations
2. **Prove** that the operator norm satisfies the norm axioms
3. **Demonstrate** that $\mathcal{B}(\mathcal{H})$ is a Banach space (completeness)
4. **Explain** the Banach algebra structure and the submultiplicativity property
5. **Work with** the Neumann series and operator inverses
6. **Connect** the algebraic structure to quantum mechanical observables

---

## 1. Core Content: The Operator Norm

### 1.1 Definition and Equivalent Formulas

For a bounded linear operator $A: \mathcal{H} \to \mathcal{K}$ between Hilbert spaces, the **operator norm** is:

$$\boxed{\|A\| = \sup_{\|x\| = 1} \|Ax\| = \sup_{x \neq 0} \frac{\|Ax\|}{\|x\|} = \inf\{M : \|Ax\| \leq M\|x\| \; \forall x\}}$$

**Theorem**: All three expressions are equal.

**Proof**: Let $\alpha = \sup_{\|x\|=1}\|Ax\|$, $\beta = \sup_{x\neq 0}\frac{\|Ax\|}{\|x\|}$, $\gamma = \inf\{M : \|Ax\| \leq M\|x\|\}$.

**$\alpha = \beta$**: For $x \neq 0$, let $y = x/\|x\|$, so $\|y\| = 1$ and $\frac{\|Ax\|}{\|x\|} = \|Ay\| \leq \alpha$. Taking supremum, $\beta \leq \alpha$. Conversely, if $\|x\| = 1$, then $\|Ax\| = \frac{\|Ax\|}{\|x\|} \leq \beta$, so $\alpha \leq \beta$.

**$\beta = \gamma$**: If $\|Ax\| \leq M\|x\|$ for all $x$, then $\frac{\|Ax\|}{\|x\|} \leq M$ for $x \neq 0$, so $\beta \leq M$. Taking infimum, $\beta \leq \gamma$. Conversely, $\|Ax\| = \|x\| \cdot \frac{\|Ax\|}{\|x\|} \leq \beta\|x\|$, so $\beta$ is an admissible $M$, hence $\gamma \leq \beta$. $\square$

### 1.2 The Operator Norm is a Norm

**Theorem**: The function $\|A\| = \sup_{\|x\|=1}\|Ax\|$ defines a norm on the vector space $\mathcal{B}(\mathcal{H}, \mathcal{K})$ of bounded linear operators.

**Proof**:

**(N1) Positivity**: $\|A\| = \sup_{\|x\|=1}\|Ax\| \geq 0$ since $\|Ax\| \geq 0$.

If $\|A\| = 0$, then $\|Ax\| = 0$ for all $\|x\| = 1$. By linearity, $Ax = 0$ for all $x$, so $A = 0$.

**(N2) Homogeneity**: For $\alpha \in \mathbb{C}$:
$$\|\alpha A\| = \sup_{\|x\|=1}\|(\alpha A)x\| = \sup_{\|x\|=1}|\alpha|\|Ax\| = |\alpha| \sup_{\|x\|=1}\|Ax\| = |\alpha|\|A\|$$

**(N3) Triangle Inequality**:
$$\|A + B\| = \sup_{\|x\|=1}\|(A+B)x\| = \sup_{\|x\|=1}\|Ax + Bx\| \leq \sup_{\|x\|=1}(\|Ax\| + \|Bx\|)$$
$$\leq \sup_{\|x\|=1}\|Ax\| + \sup_{\|x\|=1}\|Bx\| = \|A\| + \|B\|$$

$\square$

### 1.3 Fundamental Properties

**Theorem (Submultiplicativity)**: For composable bounded operators $A$ and $B$:

$$\boxed{\|AB\| \leq \|A\| \|B\|}$$

**Proof**: For any $x$ with $\|x\| = 1$:
$$\|ABx\| = \|A(Bx)\| \leq \|A\| \|Bx\| \leq \|A\| \|B\| \|x\| = \|A\| \|B\|$$

Taking supremum over $\|x\| = 1$ gives $\|AB\| \leq \|A\|\|B\|$. $\square$

**Corollary**: For $A \in \mathcal{B}(\mathcal{H})$:
$$\|A^n\| \leq \|A\|^n$$

**Important**: Equality need not hold! For example, if $N$ is nilpotent ($N^2 = 0$), then $\|N^2\| = 0 < \|N\|^2$.

---

## 2. B(H) as a Banach Space

### 2.1 Completeness Theorem

**Theorem**: The space $\mathcal{B}(\mathcal{H}, \mathcal{K})$ of bounded linear operators, equipped with the operator norm, is a Banach space.

**Proof**: Let $(A_n)$ be a Cauchy sequence in $\mathcal{B}(\mathcal{H}, \mathcal{K})$.

**Step 1: Pointwise convergence.**

For each $x \in \mathcal{H}$:
$$\|A_n x - A_m x\| = \|(A_n - A_m)x\| \leq \|A_n - A_m\| \|x\| \to 0$$

So $(A_n x)$ is Cauchy in $\mathcal{K}$. Since $\mathcal{K}$ is complete, define:
$$Ax = \lim_{n \to \infty} A_n x$$

**Step 2: $A$ is linear.**

$$A(\alpha x + \beta y) = \lim_n A_n(\alpha x + \beta y) = \lim_n(\alpha A_n x + \beta A_n y) = \alpha Ax + \beta Ay$$

**Step 3: $A$ is bounded.**

Since $(A_n)$ is Cauchy, it's bounded: $\|A_n\| \leq M$ for all $n$.

$$\|Ax\| = \lim_n \|A_n x\| \leq \limsup_n \|A_n\| \|x\| \leq M\|x\|$$

So $\|A\| \leq M < \infty$.

**Step 4: $A_n \to A$ in operator norm.**

Let $\varepsilon > 0$. Choose $N$ such that $\|A_n - A_m\| < \varepsilon$ for $n, m \geq N$.

For any $x$ with $\|x\| = 1$ and $n \geq N$:
$$\|A_n x - Ax\| = \lim_{m \to \infty} \|A_n x - A_m x\| \leq \limsup_m \|A_n - A_m\| \leq \varepsilon$$

Taking supremum: $\|A_n - A\| \leq \varepsilon$ for $n \geq N$. $\square$

### 2.2 Banach Algebra Structure

**Definition**: A **Banach algebra** is a Banach space $\mathcal{A}$ with an associative multiplication satisfying:
1. $(ab)c = a(bc)$ (associativity)
2. $a(b + c) = ab + ac$ and $(a + b)c = ac + bc$ (distributivity)
3. $\alpha(ab) = (\alpha a)b = a(\alpha b)$ (scalar compatibility)
4. $\|ab\| \leq \|a\| \|b\|$ (submultiplicativity)

If $\mathcal{A}$ has an identity element $e$ with $\|e\| = 1$, it's a **unital Banach algebra**.

**Theorem**: $\mathcal{B}(\mathcal{H})$ is a unital Banach algebra with identity $I$ (the identity operator).

**Proof**: We verified:
- $\mathcal{B}(\mathcal{H})$ is a Banach space (completeness)
- Composition is associative, distributive, scalar-compatible
- $\|AB\| \leq \|A\|\|B\|$ (submultiplicativity)
- $\|I\| = \sup_{\|x\|=1}\|Ix\| = \sup_{\|x\|=1}\|x\| = 1$ $\square$

---

## 3. The Neumann Series and Invertibility

### 3.1 The Neumann Series

**Theorem (Neumann Series)**: If $A \in \mathcal{B}(\mathcal{H})$ with $\|A\| < 1$, then $(I - A)$ is invertible and:

$$\boxed{(I - A)^{-1} = \sum_{n=0}^\infty A^n = I + A + A^2 + A^3 + \cdots}$$

Moreover, $\|(I-A)^{-1}\| \leq \frac{1}{1 - \|A\|}$.

**Proof**:

**Step 1: The series converges.**

$$\left\|\sum_{n=0}^N A^n\right\| \leq \sum_{n=0}^N \|A^n\| \leq \sum_{n=0}^N \|A\|^n$$

Since $\|A\| < 1$, this is a convergent geometric series. The partial sums form a Cauchy sequence in $\mathcal{B}(\mathcal{H})$, which is complete, so:
$$S = \sum_{n=0}^\infty A^n \quad \text{exists in } \mathcal{B}(\mathcal{H})$$

**Step 2: $S$ is the inverse of $(I - A)$.**

$$(I - A)S = (I - A)\sum_{n=0}^\infty A^n = \sum_{n=0}^\infty A^n - \sum_{n=0}^\infty A^{n+1} = \sum_{n=0}^\infty A^n - \sum_{n=1}^\infty A^n = I$$

Similarly, $S(I - A) = I$.

**Step 3: Norm bound.**

$$\|S\| \leq \sum_{n=0}^\infty \|A\|^n = \frac{1}{1 - \|A\|}$$

$\square$

### 3.2 Perturbation of the Identity

**Corollary**: If $\|B\| < 1$, then $I + B$ is invertible with:
$$(I + B)^{-1} = \sum_{n=0}^\infty (-1)^n B^n = I - B + B^2 - B^3 + \cdots$$

**Corollary**: The set of invertible operators in $\mathcal{B}(\mathcal{H})$ is **open**.

**Proof**: If $A$ is invertible and $\|B\| < 1/\|A^{-1}\|$, then:
$$A + B = A(I + A^{-1}B)$$
Since $\|A^{-1}B\| \leq \|A^{-1}\|\|B\| < 1$, the factor $I + A^{-1}B$ is invertible, hence $A + B$ is invertible. $\square$

---

## 4. Quantum Mechanics Connection

### 4.1 Observables and the Algebra of Operators

In quantum mechanics, physical observables form a (non-commutative) algebra:

| Algebraic Property | Physical Meaning |
|-------------------|------------------|
| Addition $A + B$ | Adding observables (e.g., kinetic + potential energy) |
| Scalar multiplication $\alpha A$ | Scaling units |
| Composition $AB$ | Sequential measurement / combined operation |
| $AB \neq BA$ | Non-commutativity → uncertainty principle |
| $\|A\|$ | Maximum expected value on unit states |

### 4.2 The Operator Norm as Physical Bound

For an observable $A$:
$$\|A\| = \sup_{\|\psi\|=1} \|A\psi\|$$

This relates to the spectral radius: $r(A) = \sup\{|\lambda| : \lambda \in \sigma(A)\}$.

For self-adjoint operators: $\|A\| = r(A)$ = largest magnitude eigenvalue.

**Physical interpretation**: $\|A\|$ bounds the possible measurement outcomes.

### 4.3 Time Evolution and Exponentials

The unitary time evolution operator is:
$$U(t) = e^{-iHt/\hbar} = \sum_{n=0}^\infty \frac{(-iHt/\hbar)^n}{n!}$$

For **bounded** Hamiltonians, this series converges in operator norm:
$$\left\|e^{-iHt/\hbar}\right\| \leq e^{\|H\| |t|/\hbar}$$

For unbounded $H$ (the physical case), the exponential requires spectral theory (Week 36).

### 4.4 Resolvents in Quantum Mechanics

The **resolvent** of an operator $A$ at $z \in \mathbb{C}$ is:
$$R_z(A) = (zI - A)^{-1}$$

By the Neumann series, if $|z| > \|A\|$:
$$R_z(A) = \frac{1}{z}(I - A/z)^{-1} = \frac{1}{z}\sum_{n=0}^\infty \frac{A^n}{z^n}$$

The resolvent is central to spectral theory and Green's functions.

---

## 5. Worked Examples

### Example 1: Computing Operator Norm via Singular Values

**Problem**: Find the operator norm of $A: \mathbb{C}^2 \to \mathbb{C}^2$ given by:
$$A = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$$

**Solution**:

The operator norm (spectral norm) equals the largest singular value, which is $\sqrt{\text{largest eigenvalue of } A^*A}$.

$$A^* = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}$$

$$A^* A = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 2 & 5 \end{pmatrix}$$

Eigenvalues of $A^*A$: $\det(A^*A - \lambda I) = (1-\lambda)(5-\lambda) - 4 = \lambda^2 - 6\lambda + 1 = 0$

$$\lambda = \frac{6 \pm \sqrt{36 - 4}}{2} = 3 \pm 2\sqrt{2}$$

Largest eigenvalue: $\lambda_{\max} = 3 + 2\sqrt{2}$

$$\boxed{\|A\| = \sqrt{3 + 2\sqrt{2}} = 1 + \sqrt{2} \approx 2.414}$$

(Note: $\sqrt{3 + 2\sqrt{2}} = \sqrt{(\sqrt{2}+1)^2} = \sqrt{2} + 1$.)

---

### Example 2: Neumann Series Application

**Problem**: Let $A: \ell^2 \to \ell^2$ be defined by $(Ax)_n = x_n / 2^n$. Find $(I - A)^{-1}$.

**Solution**:

**Step 1: Compute $\|A\|$.**

$$\|Ax\|^2 = \sum_{n=1}^\infty \left|\frac{x_n}{2^n}\right|^2 = \sum_{n=1}^\infty \frac{|x_n|^2}{4^n} \leq \frac{1}{4}\sum_{n=1}^\infty |x_n|^2 = \frac{\|x\|^2}{4}$$

Wait, this bound is loose. More carefully, the largest amplification is at $n=1$:
$$\|A\| = \sup_n \frac{1}{2^n} = \frac{1}{2}$$

Actually, $A$ is a diagonal operator with entries $1/2^n$, so $\|A\| = 1/2 < 1$.

**Step 2: Apply Neumann series.**

Since $\|A\| = 1/2 < 1$:
$$(I - A)^{-1} = \sum_{n=0}^\infty A^n$$

**Step 3: Find explicit formula.**

$(A^k x)_n = \frac{x_n}{2^{kn}}$, so:
$$((I-A)^{-1}x)_n = \sum_{k=0}^\infty \frac{x_n}{2^{kn}} = x_n \sum_{k=0}^\infty \frac{1}{2^{kn}} = x_n \cdot \frac{1}{1 - 2^{-n}} = \frac{2^n x_n}{2^n - 1}$$

$$\boxed{((I-A)^{-1}x)_n = \frac{2^n}{2^n - 1} x_n}$$

---

### Example 3: Verifying B(H) Completeness

**Problem**: Let $A_k: \ell^2 \to \ell^2$ be defined by $(A_k x)_n = x_n$ for $n \leq k$ and $(A_k x)_n = 0$ for $n > k$ (truncation operators). Show $(A_k)$ converges to $I$ and verify the convergence is in operator norm.

**Solution**:

**Step 1: $A_k$ is bounded with $\|A_k\| = 1$.**

$A_k$ is the projection onto $\text{span}\{e_1, \ldots, e_k\}$. For any $x$:
$$\|A_k x\|^2 = \sum_{n=1}^k |x_n|^2 \leq \sum_{n=1}^\infty |x_n|^2 = \|x\|^2$$

Equality holds when $x = e_1$, so $\|A_k\| = 1$.

**Step 2: Does $A_k \to I$?**

For any $x$:
$$\|(I - A_k)x\|^2 = \sum_{n=k+1}^\infty |x_n|^2 \to 0 \text{ as } k \to \infty$$

So $A_k \to I$ **pointwise** (strongly).

**Step 3: Check operator norm convergence.**

$$\|I - A_k\| = \sup_{\|x\|=1}\|(I - A_k)x\| = \sup_{\|x\|=1}\left(\sum_{n > k}|x_n|^2\right)^{1/2}$$

For $x = e_{k+1}$: $\|(I - A_k)e_{k+1}\| = \|e_{k+1}\| = 1$.

So $\|I - A_k\| = 1$ for all $k$.

**Conclusion**: $A_k \to I$ pointwise but **NOT in operator norm**.

This illustrates the difference between **strong convergence** and **norm convergence**. $\square$

---

## 6. Practice Problems

### Level 1: Direct Application

1. Compute the operator norm of $A: \mathbb{C}^2 \to \mathbb{C}^2$ given by $A = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$.

2. Let $S_R$ be the right shift on $\ell^2$. Compute $\|S_R^n\|$ for all $n \geq 1$.

3. If $\|A\| = 0.9$ and $\|B\| = 0.8$, what bound can you give for $\|AB\|$?

### Level 2: Intermediate

4. **Prove**: If $A$ is invertible and $\|B - A\| < 1/\|A^{-1}\|$, then $B$ is invertible with:
   $$\|B^{-1}\| \leq \frac{\|A^{-1}\|}{1 - \|A^{-1}\|\|B - A\|}$$

5. Let $V: L^2[0,1] \to L^2[0,1]$ be the Volterra operator $(Vf)(x) = \int_0^x f(t)\,dt$. Prove:
   - $\|V\| \leq 1$
   - $\|V^n\| \leq 1/n!$
   - Conclude that $r(V) = 0$ (spectral radius is zero).

6. **Quantum Connection**: For the Pauli matrices $\sigma_x, \sigma_y, \sigma_z$, verify that $\|\sigma_i\| = 1$ for $i = x, y, z$.

### Level 3: Challenging

7. **Prove**: For any $A \in \mathcal{B}(\mathcal{H})$:
   $$\|A\|^2 = \|A^*A\| = \|AA^*\|$$
   (This is the **C*-identity**, fundamental to operator algebras.)

8. Let $A_n \to A$ in operator norm. Prove that if each $A_n$ is invertible and $A$ is invertible, then $A_n^{-1} \to A^{-1}$ in operator norm.

9. **Research problem**: Show that if $H$ is infinite-dimensional, then the unit ball in $\mathcal{B}(\mathcal{H})$ is **not compact** in operator norm topology. (Hint: Consider shift operators.)

---

## 7. Computational Lab: Operator Norms

```python
"""
Day 240 Computational Lab: Operator Norm and B(H)
Computing norms, verifying Banach algebra properties, and Neumann series
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm, inv, expm
from numpy.linalg import matrix_power

# ============================================================
# Part 1: Computing Operator Norms
# ============================================================

def compute_operator_norms():
    """
    Compute operator norms for various matrices using different methods.
    """
    print("=" * 60)
    print("Part 1: Computing Operator Norms")
    print("=" * 60)

    # Test matrices
    matrices = {
        "Diagonal": np.diag([1, 2, 3, 4]),
        "Upper triangular": np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]]),
        "Rotation (45°)": np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                    [np.sin(np.pi/4), np.cos(np.pi/4)]]),
        "Nilpotent": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        "Pauli X": np.array([[0, 1], [1, 0]]),
        "Pauli Z": np.array([[1, 0], [0, -1]])
    }

    for name, A in matrices.items():
        # Operator norm = spectral norm = largest singular value
        op_norm = norm(A, ord=2)

        # Alternative: sqrt of largest eigenvalue of A^*A
        AstarA = A.conj().T @ A
        eigenvalues = np.linalg.eigvalsh(AstarA)
        alt_norm = np.sqrt(np.max(eigenvalues))

        print(f"\n{name}:")
        print(f"  ||A|| (spectral norm) = {op_norm:.6f}")
        print(f"  sqrt(max eig A*A) = {alt_norm:.6f}")

        # For comparison, other norms
        frob_norm = norm(A, ord='fro')  # Frobenius
        one_norm = norm(A, ord=1)       # max column sum
        inf_norm = norm(A, ord=np.inf)  # max row sum

        print(f"  ||A||_F (Frobenius) = {frob_norm:.6f}")
        print(f"  ||A||_1 (column sum) = {one_norm:.6f}")
        print(f"  ||A||_∞ (row sum) = {inf_norm:.6f}")

# ============================================================
# Part 2: Submultiplicativity Verification
# ============================================================

def verify_submultiplicativity():
    """
    Numerically verify ||AB|| ≤ ||A|| ||B||.
    """
    print("\n" + "=" * 60)
    print("Part 2: Submultiplicativity ||AB|| ≤ ||A|| ||B||")
    print("=" * 60)

    np.random.seed(42)
    n_trials = 10

    fig, ax = plt.subplots(figsize=(10, 6))

    ratios = []
    for i in range(n_trials):
        # Random matrices
        n = 5
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        B = np.random.randn(n, n) + 1j * np.random.randn(n, n)

        norm_A = norm(A, ord=2)
        norm_B = norm(B, ord=2)
        norm_AB = norm(A @ B, ord=2)

        ratio = norm_AB / (norm_A * norm_B)
        ratios.append(ratio)

        print(f"Trial {i+1}: ||AB|| = {norm_AB:.4f}, ||A||·||B|| = {norm_A*norm_B:.4f}, ratio = {ratio:.4f}")

    ax.bar(range(1, n_trials+1), ratios, color='steelblue', alpha=0.7)
    ax.axhline(y=1, color='r', linestyle='--', label='Upper bound (ratio = 1)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('||AB|| / (||A|| ||B||)')
    ax.set_title('Submultiplicativity: Ratio is Always ≤ 1')
    ax.legend()
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig('submultiplicativity.png', dpi=150)
    plt.show()

# ============================================================
# Part 3: Neumann Series Convergence
# ============================================================

def neumann_series_demo():
    """
    Demonstrate the Neumann series (I - A)^{-1} = sum A^n.
    """
    print("\n" + "=" * 60)
    print("Part 3: Neumann Series Convergence")
    print("=" * 60)

    # Create a matrix with ||A|| < 1
    n = 4
    A = np.array([
        [0.1, 0.2, 0, 0],
        [0.1, 0.1, 0.1, 0],
        [0, 0.2, 0.1, 0.1],
        [0, 0, 0.1, 0.2]
    ])

    norm_A = norm(A, ord=2)
    print(f"||A|| = {norm_A:.4f}")

    if norm_A >= 1:
        print("Warning: ||A|| >= 1, Neumann series may not converge!")
        return

    # Exact inverse
    I = np.eye(n)
    exact_inv = inv(I - A)

    # Neumann series partial sums
    max_terms = 20
    partial_sums = []
    errors = []

    S = np.zeros_like(A)
    A_power = I.copy()

    for k in range(max_terms):
        S = S + A_power
        A_power = A_power @ A

        error = norm(S - exact_inv, ord=2)
        partial_sums.append(S.copy())
        errors.append(error)

    print(f"\nNeumann series convergence:")
    print(f"  After 5 terms: error = {errors[4]:.6e}")
    print(f"  After 10 terms: error = {errors[9]:.6e}")
    print(f"  After 20 terms: error = {errors[19]:.6e}")

    # Theoretical bound
    theoretical_errors = [(norm_A**(k+1)) / (1 - norm_A) for k in range(max_terms)]

    # Plot convergence
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(range(max_terms), errors, 'b-o', label='Actual error', markersize=4)
    ax.semilogy(range(max_terms), theoretical_errors, 'r--', label='Theoretical bound')
    ax.set_xlabel('Number of terms')
    ax.set_ylabel('Error ||S_N - (I-A)^{-1}||')
    ax.set_title(f'Neumann Series Convergence (||A|| = {norm_A:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('neumann_series.png', dpi=150)
    plt.show()

# ============================================================
# Part 4: C*-Identity Verification
# ============================================================

def verify_cstar_identity():
    """
    Verify the C*-identity: ||A||^2 = ||A*A||.
    """
    print("\n" + "=" * 60)
    print("Part 4: C*-Identity ||A||² = ||A*A||")
    print("=" * 60)

    np.random.seed(123)

    fig, ax = plt.subplots(figsize=(10, 6))

    norm_A_squared = []
    norm_AstarA = []

    for i in range(20):
        n = np.random.randint(3, 10)
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

        val1 = norm(A, ord=2)**2
        val2 = norm(A.conj().T @ A, ord=2)

        norm_A_squared.append(val1)
        norm_AstarA.append(val2)

        print(f"Trial {i+1}: ||A||² = {val1:.6f}, ||A*A|| = {val2:.6f}, diff = {abs(val1-val2):.2e}")

    ax.scatter(norm_A_squared, norm_AstarA, s=50, alpha=0.7)
    ax.plot([0, max(norm_A_squared)], [0, max(norm_AstarA)], 'r--', label='y = x')
    ax.set_xlabel('||A||²')
    ax.set_ylabel('||A*A||')
    ax.set_title('C*-Identity: ||A||² = ||A*A|| (should lie on diagonal)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('cstar_identity.png', dpi=150)
    plt.show()

# ============================================================
# Part 5: Quantum Application - Exponentials
# ============================================================

def quantum_exponentials():
    """
    Compute operator exponentials for quantum dynamics.
    """
    print("\n" + "=" * 60)
    print("Part 5: Quantum Operator Exponentials")
    print("=" * 60)

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Hamiltonian: H = omega * sigma_z / 2 (two-level system)
    omega = 1.0  # frequency
    H = omega * sigma_z / 2

    # Time evolution U(t) = exp(-i H t)
    t_values = np.linspace(0, 4*np.pi, 100)

    # Track evolution of |+⟩ = (|0⟩ + |1⟩)/√2
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

    probs_0 = []
    probs_1 = []

    for t in t_values:
        U = expm(-1j * H * t)
        psi_t = U @ psi_plus

        probs_0.append(np.abs(psi_t[0])**2)
        probs_1.append(np.abs(psi_t[1])**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Probabilities
    axes[0].plot(t_values, probs_0, 'b-', label='$|\\langle 0|\\psi(t)\\rangle|^2$')
    axes[0].plot(t_values, probs_1, 'r-', label='$|\\langle 1|\\psi(t)\\rangle|^2$')
    axes[0].set_xlabel('Time $t$')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Two-Level System: $H = \\omega \\sigma_z / 2$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Norm of U(t) - should be 1 (unitary)
    norms = [norm(expm(-1j * H * t), ord=2) for t in t_values]
    axes[1].plot(t_values, norms, 'g-')
    axes[1].axhline(y=1, color='r', linestyle='--', label='||U|| = 1')
    axes[1].set_xlabel('Time $t$')
    axes[1].set_ylabel('||U(t)||')
    axes[1].set_title('Unitary Evolution: ||U(t)|| = 1 for all $t$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.99, 1.01)

    plt.tight_layout()
    plt.savefig('quantum_exponentials.png', dpi=150)
    plt.show()

    print("Unitary evolution preserves the operator norm: ||U(t)|| = 1")
    print(f"Initial state: |+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"Hamiltonian: H = ω σ_z / 2")
    print("The state acquires relative phases but probabilities oscillate.")

# ============================================================
# Part 6: Convergence Types Comparison
# ============================================================

def convergence_comparison():
    """
    Compare norm convergence vs strong (pointwise) convergence.
    """
    print("\n" + "=" * 60)
    print("Part 6: Norm vs Strong Convergence")
    print("=" * 60)

    # Truncation operators A_k on finite-dimensional approximation
    N = 50  # Dimension of truncated space

    k_values = range(1, N+1)

    # ||I - A_k|| for truncation operators
    norm_diff = [1] * len(k_values)  # Always 1 for truncation

    # Strong convergence: for a fixed vector x, ||(I - A_k)x|| → 0
    x = np.exp(-np.arange(N) / 10)  # Decaying sequence
    x = x / np.linalg.norm(x)  # Normalize

    strong_errors = []
    for k in k_values:
        # (I - A_k)x = (0, ..., 0, x_{k+1}, ...)
        error = np.linalg.norm(x[k:])
        strong_errors.append(error)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_values, norm_diff, 'b-', linewidth=2, label='Operator norm ||I - A_k|| = 1')
    ax.plot(k_values, strong_errors, 'r-', linewidth=2, label='Strong: ||(I - A_k)x|| → 0')
    ax.set_xlabel('k')
    ax.set_ylabel('Error')
    ax.set_title('Norm Convergence vs Strong Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=150)
    plt.show()

    print("Key insight:")
    print("  - Truncation operators A_k → I strongly (pointwise)")
    print("  - But ||I - A_k|| = 1 for all k (no norm convergence)")
    print("  - Strong convergence does NOT imply norm convergence!")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 240: Operator Norm and B(H) - Computational Lab")
    print("=" * 60)

    compute_operator_norms()
    verify_submultiplicativity()
    neumann_series_demo()
    verify_cstar_identity()
    quantum_exponentials()
    convergence_comparison()

    print("\n" + "=" * 60)
    print("Lab complete! Key takeaways:")
    print("  1. ||AB|| ≤ ||A|| ||B|| (submultiplicativity)")
    print("  2. Neumann series converges when ||A|| < 1")
    print("  3. C*-identity: ||A||² = ||A*A||")
    print("  4. Unitary operators preserve norm: ||U|| = 1")
    print("  5. Strong convergence ≠ norm convergence")
    print("=" * 60)
```

---

## 8. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| **Operator Norm** | $\|A\| = \sup_{\|x\|=1} \|Ax\|$ |
| **Banach Algebra** | Complete normed algebra with $\|ab\| \leq \|a\|\|b\|$ |
| **Neumann Series** | $(I-A)^{-1} = \sum_{n=0}^\infty A^n$ when $\|A\| < 1$ |

### Key Formulas

$$\boxed{\begin{aligned}
&\text{Operator Norm:} && \|A\| = \sup_{\|x\|=1}\|Ax\| = \sup_{x \neq 0}\frac{\|Ax\|}{\|x\|} \\
&\text{Submultiplicativity:} && \|AB\| \leq \|A\|\|B\| \\
&\text{Neumann Series:} && (I-A)^{-1} = \sum_{n=0}^\infty A^n, \quad \|A\| < 1 \\
&\text{Inverse Bound:} && \|(I-A)^{-1}\| \leq \frac{1}{1-\|A\|}
\end{aligned}}$$

### Key Theorems

| Theorem | Statement |
|---------|-----------|
| **$\mathcal{B}(\mathcal{H})$ Complete** | Space of bounded operators is a Banach space |
| **Neumann Series** | $\|A\| < 1 \Rightarrow (I-A)^{-1} = \sum A^n$ |
| **Invertibles Open** | Set of invertible operators is open |

### Key Insights

1. **The operator norm** measures maximum amplification
2. **$\mathcal{B}(\mathcal{H})$ is a Banach algebra** — complete + submultiplicative
3. **Neumann series** is the operator analogue of geometric series
4. **Invertible operators form an open set** — small perturbations preserve invertibility
5. **Strong convergence $\neq$ norm convergence** — crucial distinction

---

## 9. Daily Checklist

- [ ] I can compute operator norms using the supremum formula
- [ ] I can verify the operator norm satisfies norm axioms
- [ ] I can prove $\mathcal{B}(\mathcal{H})$ is complete
- [ ] I understand the Banach algebra structure
- [ ] I can apply the Neumann series when $\|A\| < 1$
- [ ] I understand why invertible operators form an open set
- [ ] I can distinguish between norm and strong convergence
- [ ] I completed the computational lab exercises

---

## 10. Preview: Day 241

Tomorrow we introduce the **adjoint operator** $A^\dagger$, defined by the relation:
$$\langle Ax, y \rangle = \langle x, A^\dagger y \rangle$$

The adjoint is the operator-theoretic generalization of the conjugate transpose of a matrix. It leads to crucial classifications:
- **Self-adjoint**: $A = A^\dagger$ (Hermitian matrices generalized)
- **Unitary**: $U^\dagger U = I$ (rotation/reflection generalized)
- **Normal**: $AA^\dagger = A^\dagger A$ (diagonalizable operators)

These classifications are fundamental to quantum mechanics, where observables must be self-adjoint and symmetries must be unitary.

---

*"The space of bounded operators on a Hilbert space is not merely a vector space but a Banach algebra—a structure that encodes both the linear and multiplicative properties essential for quantum mechanics."* — Irving Segal
