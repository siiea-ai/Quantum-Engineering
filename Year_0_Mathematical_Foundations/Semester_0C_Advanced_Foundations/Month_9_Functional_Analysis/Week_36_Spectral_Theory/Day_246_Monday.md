# Day 246: Spectrum of an Operator

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning I** | 2 hours | Definitions: Spectrum and Resolvent |
| **Morning II** | 2 hours | Classification of Spectral Types |
| **Afternoon** | 2 hours | Worked Examples and Problem Solving |
| **Evening** | 2 hours | Computational Lab: Spectral Analysis |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** the spectrum and resolvent of a bounded linear operator
2. **Classify** spectral points into point, continuous, and residual spectrum
3. **Compute** the spectral radius and relate it to the operator norm
4. **Prove** fundamental properties of the resolvent function
5. **Analyze** spectra of specific operators on Hilbert spaces
6. **Connect** spectral concepts to quantum mechanical observables

## Core Content

### 1. The Spectrum: Definition and Motivation

In finite dimensions, the eigenvalue problem $Ax = \lambda x$ completely characterizes a matrix. For infinite-dimensional operators, the situation is richer and more subtle.

**Definition (Resolvent Set and Spectrum)**

Let $A: \mathcal{H} \to \mathcal{H}$ be a bounded linear operator on a complex Hilbert space $\mathcal{H}$.

The **resolvent set** $\rho(A)$ consists of all $\lambda \in \mathbb{C}$ such that:
1. $(A - \lambda I)$ is injective (one-to-one)
2. $(A - \lambda I)$ is surjective (onto)
3. $(A - \lambda I)^{-1}$ is bounded

The **spectrum** is the complement:
$$\boxed{\sigma(A) = \mathbb{C} \setminus \rho(A)}$$

The **resolvent operator** at $\lambda \in \rho(A)$ is:
$$\boxed{R_\lambda(A) = R(\lambda, A) = (A - \lambda I)^{-1}}$$

**Remark**: Condition 3 is automatic by the Bounded Inverse Theorem when $A$ is bounded and conditions 1-2 hold. However, it becomes crucial for unbounded operators.

### 2. Classification of the Spectrum

The spectrum decomposes into three disjoint parts based on which condition fails:

**Definition (Spectral Classification)**

For $\lambda \in \sigma(A)$:

**Point Spectrum (Discrete Spectrum)**
$$\sigma_p(A) = \{\lambda \in \mathbb{C} : A - \lambda I \text{ is not injective}\}$$
These are the **eigenvalues** of $A$. There exists $x \neq 0$ with $Ax = \lambda x$.

**Continuous Spectrum**
$$\sigma_c(A) = \{\lambda \in \mathbb{C} : A - \lambda I \text{ is injective, } \overline{\text{Range}(A-\lambda I)} = \mathcal{H}, \text{ but Range} \neq \mathcal{H}\}$$
The inverse exists on a dense domain but is unbounded.

**Residual Spectrum**
$$\sigma_r(A) = \{\lambda \in \mathbb{C} : A - \lambda I \text{ is injective but } \overline{\text{Range}(A-\lambda I)} \neq \mathcal{H}\}$$
The range is not dense.

**Theorem (Spectral Decomposition)**
$$\sigma(A) = \sigma_p(A) \cup \sigma_c(A) \cup \sigma_r(A) \quad \text{(disjoint union)}$$

### 3. Properties of the Spectrum

**Theorem (Spectrum is Non-empty and Bounded)**

For any bounded operator $A \in \mathcal{B}(\mathcal{H})$:
1. $\sigma(A) \neq \emptyset$
2. $\sigma(A) \subseteq \overline{D}(0, \|A\|)$ (closed disk of radius $\|A\|$)
3. $\sigma(A)$ is compact

*Proof of (2):* If $|\lambda| > \|A\|$, then $\|A/\lambda\| < 1$, so:
$$(A - \lambda I) = -\lambda(I - A/\lambda)$$
has bounded inverse via the Neumann series:
$$(I - A/\lambda)^{-1} = \sum_{n=0}^\infty (A/\lambda)^n$$
Thus $\lambda \in \rho(A)$. $\square$

**Definition (Spectral Radius)**
$$\boxed{r(A) = \sup\{|\lambda| : \lambda \in \sigma(A)\}}$$

**Theorem (Spectral Radius Formula)**
$$\boxed{r(A) = \lim_{n \to \infty} \|A^n\|^{1/n} = \inf_{n \geq 1} \|A^n\|^{1/n}}$$

*Proof:* The limit exists by submultiplicativity. For $|\lambda| > r(A)$, the resolvent series:
$$R_\lambda(A) = -\frac{1}{\lambda}\sum_{n=0}^\infty \left(\frac{A}{\lambda}\right)^n$$
converges, giving $r(A) \leq \liminf \|A^n\|^{1/n}$. The reverse inequality uses the maximum modulus principle. $\square$

### 4. The Resolvent Function

**Theorem (Resolvent Identity)**

For $\lambda, \mu \in \rho(A)$:
$$\boxed{R_\lambda(A) - R_\mu(A) = (\lambda - \mu)R_\lambda(A)R_\mu(A)}$$

*Proof:*
$$R_\lambda - R_\mu = R_\lambda[(A - \mu I) - (A - \lambda I)]R_\mu = (\lambda - \mu)R_\lambda R_\mu$$
$\square$

**Corollary (Commutativity)**
$$R_\lambda(A) R_\mu(A) = R_\mu(A) R_\lambda(A)$$

**Theorem (Resolvent is Analytic)**

$R_\lambda(A)$ is an analytic $\mathcal{B}(\mathcal{H})$-valued function on $\rho(A)$ with:
$$\frac{d}{d\lambda}R_\lambda(A) = R_\lambda(A)^2$$

**Theorem (Neumann Series for Resolvent)**

For $|\lambda| > \|A\|$:
$$R_\lambda(A) = -\sum_{n=0}^\infty \frac{A^n}{\lambda^{n+1}}$$

For $\lambda_0 \in \rho(A)$ and $|\lambda - \lambda_0| < \|R_{\lambda_0}(A)\|^{-1}$:
$$R_\lambda(A) = \sum_{n=0}^\infty (\lambda_0 - \lambda)^n R_{\lambda_0}(A)^{n+1}$$

### 5. Spectrum of Self-Adjoint Operators

**Theorem (Self-Adjoint Spectrum)**

If $A = A^*$ (self-adjoint), then:
1. $\sigma(A) \subseteq \mathbb{R}$ (spectrum is real)
2. $\sigma_r(A) = \emptyset$ (no residual spectrum)
3. Eigenvectors for distinct eigenvalues are orthogonal
4. $r(A) = \|A\|$

*Proof of (1):* For $\lambda = \alpha + i\beta$ with $\beta \neq 0$:
$$\|(A - \lambda I)x\|^2 = \|(A - \alpha I)x\|^2 + \beta^2\|x\|^2 \geq \beta^2\|x\|^2$$
So $(A - \lambda I)$ is bounded below, hence injective with closed range. Self-adjointness gives dense range. $\square$

*Proof of (3):* If $Ax = \lambda x$ and $Ay = \mu y$ with $\lambda \neq \mu$:
$$\lambda\langle x, y\rangle = \langle Ax, y\rangle = \langle x, Ay\rangle = \mu\langle x, y\rangle$$
Since $\lambda, \mu \in \mathbb{R}$, we get $(\lambda - \mu)\langle x, y\rangle = 0$, so $\langle x, y\rangle = 0$. $\square$

### 6. Spectrum of Normal Operators

**Definition**: $A$ is **normal** if $AA^* = A^*A$.

**Theorem (Normal Operator Spectrum)**

If $A$ is normal:
1. $\sigma_r(A) = \emptyset$
2. $r(A) = \|A\|$
3. $\|R_\lambda(A)\| = 1/d(\lambda, \sigma(A))$ for $\lambda \in \rho(A)$

### 7. Quantum Mechanics Connection

In quantum mechanics, observables are represented by self-adjoint operators on Hilbert space. The spectrum has direct physical meaning:

| Spectral Concept | Quantum Mechanical Meaning |
|------------------|---------------------------|
| $\sigma(A)$ | Possible measurement outcomes |
| $\sigma_p(A)$ (eigenvalues) | Discrete allowed values (quantization) |
| $\sigma_c(A)$ | Continuous range of outcomes (scattering) |
| Eigenvector $|n\rangle$ | State with definite value $\lambda_n$ |
| $\|P_\lambda \psi\|^2$ | Probability of measuring $\lambda$ |

**Example: Hydrogen Atom Hamiltonian**

The Hamiltonian $H = -\frac{\hbar^2}{2m}\nabla^2 - \frac{e^2}{r}$ has:
- **Point spectrum**: $E_n = -\frac{13.6 \text{ eV}}{n^2}$ for $n = 1, 2, 3, \ldots$ (bound states)
- **Continuous spectrum**: $E \geq 0$ (scattering/ionization states)

The spectral theorem guarantees that any measurement of energy yields a value in $\sigma(H)$.

**The Measurement Postulate**

When measuring observable $A$ on state $|\psi\rangle$:
1. The result is some $\lambda \in \sigma(A)$
2. Probability of eigenvalue: $|\langle \lambda | \psi \rangle|^2$
3. Post-measurement state: projection onto eigenspace

---

## Worked Examples

### Example 1: Spectrum of the Right Shift Operator

**Problem**: Find the spectrum of the right shift $S: \ell^2 \to \ell^2$ defined by:
$$S(x_1, x_2, x_3, \ldots) = (0, x_1, x_2, x_3, \ldots)$$

**Solution**:

**Step 1**: Check if $S$ has eigenvalues.

If $Sx = \lambda x$ for $x \neq 0$:
$$(0, x_1, x_2, \ldots) = \lambda(x_1, x_2, x_3, \ldots)$$

This requires $\lambda x_1 = 0$. If $\lambda \neq 0$, then $x_1 = 0$, and by induction all $x_n = 0$.

For $\lambda = 0$: $Sx = 0$ requires $x = 0$.

**Conclusion**: $\sigma_p(S) = \emptyset$ (no eigenvalues).

**Step 2**: Analyze $(S - \lambda I)$ for $|\lambda| < 1$.

For $|\lambda| < 1$, consider the range. The equation $(S - \lambda I)x = e_1 = (1, 0, 0, \ldots)$ gives:
$$-\lambda x_1 = 1, \quad x_1 - \lambda x_2 = 0, \quad x_2 - \lambda x_3 = 0, \ldots$$

Solution: $x_n = -\lambda^{n-2}$ for $n \geq 1$ gives $x_1 = -1/\lambda$.

This is in $\ell^2$ since $\sum |x_n|^2 < \infty$ for $|\lambda| < 1$.

So $e_1$ is in the range, and similarly all $e_n$. The range is dense.

**Step 3**: Check boundedness of inverse.

For $|\lambda| < 1$, one can show $(S - \lambda I)^{-1}$ exists on a dense domain but is unbounded.

**Conclusion**: $\sigma_c(S) = \{\lambda : |\lambda| < 1\}$ (open unit disk).

**Step 4**: Boundary analysis.

For $|\lambda| = 1$: $(S - \lambda I)$ is injective but the range is not closed. Include in $\sigma(S)$.

**Step 5**: Exterior of disk.

For $|\lambda| > 1$: Use $\|S\| = 1$, so $|\lambda| > \|S\|$ implies $\lambda \in \rho(S)$.

$$\boxed{\sigma(S) = \overline{D}(0,1) = \{\lambda : |\lambda| \leq 1\}}$$
with $\sigma_p(S) = \emptyset$, $\sigma_c(S) = D(0,1)$, $\sigma_r(S) = \emptyset$ at $|\lambda| = 1$ requires more careful analysis showing it's continuous spectrum.

### Example 2: Spectrum of a Multiplication Operator

**Problem**: On $L^2[0,1]$, let $(Mf)(x) = xf(x)$. Find $\sigma(M)$.

**Solution**:

**Step 1**: Find eigenvalues.

$Mf = \lambda f$ means $xf(x) = \lambda f(x)$ for a.e. $x$, so $(x - \lambda)f(x) = 0$ a.e.

For $\lambda \in [0,1]$: $f(x) = 0$ for $x \neq \lambda$, which is a.e., so $f = 0$ in $L^2$.

**Conclusion**: $\sigma_p(M) = \emptyset$.

**Step 2**: Determine the full spectrum.

For $\lambda \notin [0,1]$: $(M - \lambda I)^{-1}f(x) = \frac{f(x)}{x - \lambda}$ is bounded.

So $\rho(M) = \mathbb{C} \setminus [0,1]$.

**Step 3**: Classify spectrum on $[0,1]$.

For $\lambda \in [0,1]$: $(M - \lambda I)$ is injective. The range consists of $g$ such that $g(x)/(x-\lambda) \in L^2$.

This requires $g(\lambda) = 0$ in an appropriate sense. The range is dense but not all of $L^2$.

$$\boxed{\sigma(M) = [0,1], \quad \sigma_c(M) = [0,1], \quad \sigma_p(M) = \sigma_r(M) = \emptyset}$$

### Example 3: Spectral Radius Calculation

**Problem**: For the Volterra operator $(Vf)(x) = \int_0^x f(t)\,dt$ on $L^2[0,1]$, find $r(V)$.

**Solution**:

**Step 1**: Compute powers of $V$.

$$V^2 f(x) = \int_0^x \int_0^s f(t)\,dt\,ds = \int_0^x (x-t)f(t)\,dt$$

By induction:
$$V^n f(x) = \int_0^x \frac{(x-t)^{n-1}}{(n-1)!}f(t)\,dt$$

**Step 2**: Estimate $\|V^n\|$.

$$|V^n f(x)| \leq \int_0^x \frac{(x-t)^{n-1}}{(n-1)!}|f(t)|\,dt \leq \frac{x^{n-1}}{(n-1)!}\int_0^x |f(t)|\,dt$$

After careful analysis:
$$\|V^n\| \leq \frac{1}{n!}$$

**Step 3**: Apply spectral radius formula.

$$r(V) = \lim_{n \to \infty}\|V^n\|^{1/n} \leq \lim_{n \to \infty}\left(\frac{1}{n!}\right)^{1/n} = 0$$

**Conclusion**:
$$\boxed{r(V) = 0, \quad \sigma(V) = \{0\}}$$

The Volterra operator is **quasinilpotent** (spectrum is just $\{0\}$), but $V \neq 0$.

Interestingly, $0 \in \sigma_r(V)$ since $V$ is injective but not surjective (constant functions not in range).

---

## Practice Problems

### Level 1: Direct Application

1. **Diagonal Operator**: On $\ell^2$, let $D(x_1, x_2, \ldots) = (d_1 x_1, d_2 x_2, \ldots)$ where $d_n = 1/n$. Find $\sigma(D)$, $\sigma_p(D)$, $\sigma_c(D)$.

2. **Resolvent Computation**: For the $2 \times 2$ matrix $A = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$, compute $R_\lambda(A)$ explicitly and verify the resolvent identity.

3. **Spectral Radius**: Compute $r(A)$ for $A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ both directly (find eigenvalues) and via the formula $\lim\|A^n\|^{1/n}$.

### Level 2: Intermediate

4. **Bilateral Shift**: On $\ell^2(\mathbb{Z})$, the bilateral shift is $(Sx)_n = x_{n-1}$. Show that $\sigma(S) = \{|\lambda| = 1\}$ (unit circle) and classify the spectrum type.

5. **Self-Adjoint Proof**: Prove that if $A = A^*$ and $\text{Range}(A) = \mathcal{H}$, then $0 \notin \sigma(A)$.

6. **Perturbation**: If $\sigma(A) \subset D(0, r)$ and $\|B\| < \epsilon$, show $\sigma(A + B) \subset D(0, r + \epsilon)$.

### Level 3: Challenging

7. **Spectral Mapping Theorem**: Prove that for a polynomial $p$, $\sigma(p(A)) = p(\sigma(A))$.

8. **Compact Perturbation**: Let $A$ be bounded and $K$ compact. Prove that $\sigma_{ess}(A) = \sigma_{ess}(A + K)$ where $\sigma_{ess}$ denotes the essential spectrum.

9. **Numerical Range**: The numerical range is $W(A) = \{\langle Ax, x\rangle : \|x\| = 1\}$. Prove $\sigma(A) \subseteq \overline{W(A)}$ and show this inclusion can be strict.

---

## Computational Lab: Spectral Analysis

```python
"""
Day 246 Computational Lab: Spectrum of Operators
Exploring point, continuous, and residual spectra computationally
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from typing import Tuple, List

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Part 1: Spectrum of Finite-Dimensional Operators (Matrices)
# =============================================================================

def analyze_matrix_spectrum(A: np.ndarray, name: str = "A") -> dict:
    """
    Compute and analyze the spectrum of a matrix.

    Parameters:
        A: Square matrix
        name: Name for display

    Returns:
        Dictionary with spectral information
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Spectral radius
    spectral_radius = np.max(np.abs(eigenvalues))

    # Operator norm (largest singular value)
    operator_norm = np.linalg.norm(A, ord=2)

    # Check if normal (AA* = A*A)
    is_normal = np.allclose(A @ A.conj().T, A.conj().T @ A)

    # Check if self-adjoint
    is_self_adjoint = np.allclose(A, A.conj().T)

    print(f"\n{'='*60}")
    print(f"Spectral Analysis of {name}")
    print(f"{'='*60}")
    print(f"Matrix shape: {A.shape}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Spectral radius r(A): {spectral_radius:.6f}")
    print(f"Operator norm ||A||: {operator_norm:.6f}")
    print(f"Is normal: {is_normal}")
    print(f"Is self-adjoint: {is_self_adjoint}")

    if is_self_adjoint:
        print("  -> Eigenvalues should be real: ",
              np.allclose(eigenvalues.imag, 0))
    if is_normal:
        print("  -> r(A) should equal ||A||: ",
              np.isclose(spectral_radius, operator_norm))

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'spectral_radius': spectral_radius,
        'operator_norm': operator_norm,
        'is_normal': is_normal,
        'is_self_adjoint': is_self_adjoint
    }

# Example matrices
print("PART 1: FINITE-DIMENSIONAL SPECTRA")
print("=" * 60)

# Self-adjoint matrix
A_sa = np.array([[2, 1, 0],
                 [1, 3, 1],
                 [0, 1, 2]], dtype=complex)
result_sa = analyze_matrix_spectrum(A_sa, "Self-Adjoint A")

# Non-normal matrix (has complex eigenvalues despite real entries)
A_nn = np.array([[0, -1],
                 [1, 0]], dtype=complex)
result_nn = analyze_matrix_spectrum(A_nn, "Rotation Matrix")

# Nilpotent matrix (all eigenvalues zero)
A_nil = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]], dtype=complex)
result_nil = analyze_matrix_spectrum(A_nil, "Nilpotent Matrix")

# =============================================================================
# Part 2: Resolvent Computation and Visualization
# =============================================================================

def compute_resolvent(A: np.ndarray, lambda_val: complex) -> Tuple[np.ndarray, bool]:
    """
    Compute the resolvent (A - lambda*I)^{-1}.

    Returns:
        (resolvent matrix, success flag)
    """
    n = A.shape[0]
    try:
        resolvent = np.linalg.inv(A - lambda_val * np.eye(n))
        return resolvent, True
    except np.linalg.LinAlgError:
        return np.zeros_like(A), False

def resolvent_norm_map(A: np.ndarray,
                       real_range: Tuple[float, float] = (-3, 3),
                       imag_range: Tuple[float, float] = (-3, 3),
                       resolution: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ||R_lambda(A)|| over a region of the complex plane.
    """
    real_vals = np.linspace(real_range[0], real_range[1], resolution)
    imag_vals = np.linspace(imag_range[0], imag_range[1], resolution)

    norm_map = np.zeros((resolution, resolution))

    for i, re in enumerate(real_vals):
        for j, im in enumerate(imag_vals):
            lambda_val = re + 1j * im
            resolvent, success = compute_resolvent(A, lambda_val)
            if success:
                norm_map[j, i] = min(np.linalg.norm(resolvent, ord=2), 100)
            else:
                norm_map[j, i] = 100  # Large value near spectrum

    return real_vals, imag_vals, norm_map

print("\n" + "="*60)
print("PART 2: RESOLVENT VISUALIZATION")
print("="*60)

# Create figure for resolvent norm visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

matrices = [A_sa, A_nn, A_nil]
names = ["Self-Adjoint", "Rotation", "Nilpotent"]

for ax, A, name in zip(axes, matrices, names):
    real_vals, imag_vals, norm_map = resolvent_norm_map(A, resolution=100)

    # Plot resolvent norm (log scale)
    im = ax.pcolormesh(real_vals, imag_vals, np.log10(norm_map + 1),
                       shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='log10(||R_lambda||)')

    # Mark eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    ax.scatter(eigenvalues.real, eigenvalues.imag,
               c='cyan', s=100, marker='*', zorder=5, label='Eigenvalues')

    ax.set_xlabel('Re(lambda)')
    ax.set_ylabel('Im(lambda)')
    ax.set_title(f'{name} Matrix\nResolvent Norm')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resolvent_norm_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved: resolvent_norm_visualization.png")

# =============================================================================
# Part 3: Spectral Radius Formula Verification
# =============================================================================

def verify_spectral_radius_formula(A: np.ndarray, max_n: int = 50) -> None:
    """
    Verify r(A) = lim ||A^n||^{1/n} numerically.
    """
    eigenvalues = np.linalg.eigvals(A)
    true_radius = np.max(np.abs(eigenvalues))

    # Compute ||A^n||^{1/n} for various n
    n_values = list(range(1, max_n + 1))
    estimates = []

    A_power = np.eye(A.shape[0], dtype=complex)
    for n in n_values:
        A_power = A_power @ A
        norm_An = np.linalg.norm(A_power, ord=2)
        estimate = norm_An ** (1/n)
        estimates.append(estimate)

    print(f"\nSpectral Radius Formula Verification:")
    print(f"True spectral radius r(A) = {true_radius:.6f}")
    print(f"||A^{max_n}||^{{1/{max_n}}} = {estimates[-1]:.6f}")
    print(f"Convergence achieved: {np.isclose(estimates[-1], true_radius, rtol=0.01)}")

    return n_values, estimates, true_radius

print("\n" + "="*60)
print("PART 3: SPECTRAL RADIUS FORMULA")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 6))

# Random matrix example
np.random.seed(123)
A_rand = np.random.randn(5, 5) + 1j * np.random.randn(5, 5)
A_rand = A_rand / np.linalg.norm(A_rand)  # Normalize

n_vals, estimates, true_r = verify_spectral_radius_formula(A_rand, max_n=100)

ax.plot(n_vals, estimates, 'b-', linewidth=2, label=r'$\|A^n\|^{1/n}$')
ax.axhline(y=true_r, color='r', linestyle='--', linewidth=2,
           label=f'True r(A) = {true_r:.4f}')
ax.axhline(y=np.linalg.norm(A_rand, ord=2), color='g', linestyle=':',
           linewidth=2, label=f'||A|| = {np.linalg.norm(A_rand, ord=2):.4f}')

ax.set_xlabel('n', fontsize=12)
ax.set_ylabel('Estimate', fontsize=12)
ax.set_title('Convergence of Spectral Radius Formula', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([1, 100])

plt.tight_layout()
plt.savefig('spectral_radius_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved: spectral_radius_convergence.png")

# =============================================================================
# Part 4: Approximating Spectrum of Infinite-Dimensional Operators
# =============================================================================

def right_shift_truncated(n: int) -> np.ndarray:
    """
    Create n x n truncation of right shift operator on l^2.
    """
    S = np.zeros((n, n), dtype=complex)
    for i in range(1, n):
        S[i, i-1] = 1
    return S

def multiplication_operator_discretized(n: int) -> np.ndarray:
    """
    Discretize multiplication operator (Mf)(x) = xf(x) on L^2[0,1].
    """
    x_vals = np.linspace(0, 1, n)
    return np.diag(x_vals)

print("\n" + "="*60)
print("PART 4: APPROXIMATING INFINITE-DIMENSIONAL SPECTRA")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Right shift approximation
ax1 = axes[0]
for n in [10, 50, 100, 200]:
    S_n = right_shift_truncated(n)
    eigenvalues = np.linalg.eigvals(S_n)
    ax1.scatter(eigenvalues.real, eigenvalues.imag,
                alpha=0.5, s=10, label=f'n={n}')

# True spectrum is unit disk
theta = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit circle')
ax1.fill(np.cos(theta), np.sin(theta), alpha=0.1, color='gray')

ax1.set_xlabel('Re(lambda)')
ax1.set_ylabel('Im(lambda)')
ax1.set_title('Right Shift: Finite Truncations\nTrue spectrum = closed unit disk')
ax1.legend()
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Multiplication operator approximation
ax2 = axes[1]
for n in [10, 50, 100, 200]:
    M_n = multiplication_operator_discretized(n)
    eigenvalues = np.linalg.eigvals(M_n)
    ax2.scatter(eigenvalues.real, eigenvalues.imag,
                alpha=0.7, s=20, label=f'n={n}')

# True spectrum is [0,1]
ax2.axhline(y=0, color='k', linewidth=2, label='True spectrum = [0,1]')
ax2.plot([0, 1], [0, 0], 'k-', linewidth=4)

ax2.set_xlabel('Re(lambda)')
ax2.set_ylabel('Im(lambda)')
ax2.set_title('Multiplication by x: Discretizations\nTrue spectrum = [0,1]')
ax2.legend()
ax2.set_ylim([-0.5, 0.5])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('infinite_dim_spectra_approximation.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved: infinite_dim_spectra_approximation.png")

# =============================================================================
# Part 5: Resolvent Identity Verification
# =============================================================================

print("\n" + "="*60)
print("PART 5: RESOLVENT IDENTITY VERIFICATION")
print("="*60)

def verify_resolvent_identity(A: np.ndarray, lambda1: complex, lambda2: complex) -> bool:
    """
    Verify R_lambda - R_mu = (lambda - mu) R_lambda R_mu
    """
    R1, ok1 = compute_resolvent(A, lambda1)
    R2, ok2 = compute_resolvent(A, lambda2)

    if not (ok1 and ok2):
        print("Error: One of the resolvents doesn't exist")
        return False

    LHS = R1 - R2
    RHS = (lambda1 - lambda2) * R1 @ R2

    is_equal = np.allclose(LHS, RHS)

    print(f"\nResolvent Identity for lambda={lambda1}, mu={lambda2}:")
    print(f"||R_lambda - R_mu - (lambda-mu)R_lambda R_mu|| = {np.linalg.norm(LHS - RHS):.2e}")
    print(f"Identity verified: {is_equal}")

    return is_equal

# Test on a matrix
A_test = np.array([[1, 2], [0, 3]], dtype=complex)
verify_resolvent_identity(A_test, 5+1j, -2+3j)
verify_resolvent_identity(A_test, 10, -5)

# =============================================================================
# Part 6: Quantum Mechanics Connection - Hydrogen-like Spectrum
# =============================================================================

print("\n" + "="*60)
print("PART 6: QUANTUM MECHANICS - HYDROGEN SPECTRUM VISUALIZATION")
print("="*60)

fig, ax = plt.subplots(figsize=(12, 8))

# Hydrogen energy levels (bound states - point spectrum)
n_levels = 10
E_n = -13.6 / np.arange(1, n_levels + 1)**2  # in eV

# Plot energy levels
for n, E in enumerate(E_n, 1):
    ax.hlines(E, 0, 0.8, colors='blue', linewidth=2)
    ax.text(0.85, E, f'n={n}\nE={E:.2f} eV', va='center', fontsize=9)

# Continuous spectrum (ionization)
ax.fill_between([0, 0.8], [0, 0], [5, 5], alpha=0.3, color='red',
                label='Continuous spectrum (E > 0)')
ax.axhline(y=0, color='red', linewidth=2, linestyle='--')

# Labels and formatting
ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-15, 6)
ax.set_ylabel('Energy (eV)', fontsize=12)
ax.set_title('Hydrogen Atom: Spectrum of the Hamiltonian\n'
             'Point spectrum (bound states) + Continuous spectrum (scattering)',
             fontsize=14)
ax.set_xticks([])

# Add annotations
ax.annotate('Point spectrum\n(discrete bound states)',
            xy=(0.4, -7), fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.annotate('Ionization threshold', xy=(0.4, 0.5), fontsize=10, ha='center')

ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hydrogen_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved: hydrogen_spectrum.png")

print("\n" + "="*60)
print("LAB COMPLETE")
print("="*60)
print("""
Key takeaways:
1. The spectrum contains all eigenvalues but may include more (continuous spectrum)
2. The resolvent norm blows up as lambda approaches the spectrum
3. The spectral radius formula r(A) = lim ||A^n||^{1/n} converges
4. Finite truncations approximate infinite-dimensional spectra
5. In quantum mechanics, the spectrum gives possible measurement outcomes
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Resolvent | $R_\lambda(A) = (A - \lambda I)^{-1}$ |
| Spectral radius | $r(A) = \sup\{|\lambda| : \lambda \in \sigma(A)\}$ |
| Spectral radius formula | $r(A) = \lim_{n \to \infty}\|A^n\|^{1/n}$ |
| Resolvent identity | $R_\lambda - R_\mu = (\lambda - \mu)R_\lambda R_\mu$ |
| Neumann series | $R_\lambda(A) = -\sum_{n=0}^\infty A^n/\lambda^{n+1}$ for $|\lambda| > \|A\|$ |
| Self-adjoint spectrum | $\sigma(A) \subseteq \mathbb{R}$, $r(A) = \|A\|$ |

### Main Takeaways

1. **Spectrum generalizes eigenvalues**: In infinite dimensions, the spectrum includes points where $(A - \lambda I)^{-1}$ fails to exist as a bounded operator, not just eigenvalues.

2. **Three types of spectrum**: Point (eigenvalues), continuous (dense range but unbounded inverse), residual (non-dense range).

3. **Resolvent is fundamental**: The resolvent function $R_\lambda(A)$ is analytic on the resolvent set and encodes all spectral information.

4. **Spectral radius bounds spectrum**: All spectral points lie within the closed disk of radius $r(A)$.

5. **Self-adjointness simplifies everything**: Real spectrum, no residual spectrum, orthogonal eigenvectors.

6. **Quantum connection**: The spectrum of an observable determines all possible measurement outcomes.

---

## Daily Checklist

- [ ] I can define the spectrum and resolvent of a bounded operator
- [ ] I can classify spectral points into point, continuous, and residual types
- [ ] I can compute the spectral radius using the limit formula
- [ ] I can apply the resolvent identity and Neumann series
- [ ] I can prove that self-adjoint operators have real spectrum
- [ ] I can explain why eigenvalues of self-adjoint operators are orthogonal
- [ ] I understand the physical meaning of spectrum in quantum mechanics
- [ ] I completed the computational lab and generated visualizations

---

## Preview: Day 247

Tomorrow we prove the **Spectral Theorem for Compact Self-Adjoint Operators**, which states that every compact self-adjoint operator can be written as:
$$A = \sum_{n=1}^\infty \lambda_n |e_n\rangle\langle e_n|$$

This is the infinite-dimensional generalization of matrix diagonalization and provides the foundation for quantum mechanics. We'll see how this connects to the expansion of quantum states in energy eigenbases.

---

*"The theory of the spectrum is the basis for much of our understanding of linear operators. The spectrum tells us not just about eigenvalues, but about the complete behavior of an operator."*
— Paul Halmos
