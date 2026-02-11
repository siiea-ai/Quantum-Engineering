# Day 249: Functional Calculus

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning I** | 2 hours | Continuous Functional Calculus |
| **Morning II** | 2 hours | Borel Functional Calculus |
| **Afternoon** | 2 hours | Applications and Operator Functions |
| **Evening** | 2 hours | Computational Lab: Functions of Operators |

## Learning Objectives

By the end of today, you will be able to:

1. **Construct** the continuous functional calculus for self-adjoint operators
2. **Extend** to the Borel functional calculus for measurable functions
3. **Compute** operator functions like $e^{iAt}$, $\sqrt{A}$, $|A|$, $A^+$, $A^-$
4. **Apply** the spectral mapping theorem to determine spectra of $f(A)$
5. **Connect** functional calculus to quantum mechanical time evolution
6. **Implement** numerical methods for computing functions of matrices

## Core Content

### 1. Motivation: Why Functions of Operators?

In quantum mechanics and many applications, we need to evaluate functions of operators:

| Application | Operator Function |
|-------------|-------------------|
| Time evolution | $e^{-iHt/\hbar}$ |
| Thermal states | $e^{-\beta H}$ |
| Propagators | $(H - E - i\epsilon)^{-1}$ |
| Square root | $\sqrt{A}$ for positive $A$ |
| Spectral projections | $\chi_{[a,b]}(A)$ |
| Operator sign | $\text{sgn}(A)$ |

The **functional calculus** provides a rigorous way to define $f(A)$ for a self-adjoint operator $A$ and a function $f$.

### 2. Continuous Functional Calculus

**Setting**: $A$ is a bounded self-adjoint operator with spectrum $\sigma(A) \subseteq [m, M]$.

**Theorem (Continuous Functional Calculus)**

There exists a unique *-homomorphism $\Phi: C(\sigma(A)) \to \mathcal{B}(\mathcal{H})$ such that:

1. $\Phi(1) = I$ (identity function maps to identity operator)
2. $\Phi(\text{id}) = A$ (function $f(\lambda) = \lambda$ maps to $A$)
3. $\Phi$ is isometric: $\|\Phi(f)\| = \|f\|_\infty$

We write $\Phi(f) = f(A)$.

**Properties**:
- **Linearity**: $(\alpha f + \beta g)(A) = \alpha f(A) + \beta g(A)$
- **Multiplicativity**: $(fg)(A) = f(A)g(A)$
- **Conjugation**: $\bar{f}(A) = f(A)^*$
- **Polynomial agreement**: For $p(\lambda) = \sum a_n \lambda^n$, $p(A) = \sum a_n A^n$

**Construction via Polynomials**:

*Step 1*: Define $p(A)$ for polynomials by substitution.

*Step 2*: For continuous $f$, use Weierstrass approximation: $p_n \to f$ uniformly.

*Step 3*: Define $f(A) = \lim_{n\to\infty} p_n(A)$ in operator norm.

*Step 4*: Show the limit is independent of the approximating sequence.

### 3. Borel Functional Calculus

The continuous functional calculus extends to bounded Borel-measurable functions.

**Definition (Borel Functional Calculus)**

For bounded Borel function $f: \sigma(A) \to \mathbb{C}$:
$$\boxed{f(A) = \int_{\sigma(A)} f(\lambda) \, dE_\lambda}$$

where $\{E_\lambda\}$ is the spectral family of $A$.

**Properties**:
- All properties of continuous calculus extend
- $\|f(A)\| = \|f\|_{\infty, \sigma(A)} = \sup_{\lambda \in \sigma(A)} |f(\lambda)|$
- Indicator functions give spectral projections: $\chi_\Delta(A) = E(\Delta)$

**Theorem (Spectral Mapping Theorem)**

For bounded Borel function $f$:
$$\boxed{\sigma(f(A)) = \overline{f(\sigma(A))}}$$

For continuous $f$: $\sigma(f(A)) = f(\sigma(A))$ (no closure needed).

### 4. Key Examples of Functional Calculus

#### Example 1: Powers and Polynomials

For $f(\lambda) = \lambda^n$:
$$A^n = \int_{\sigma(A)} \lambda^n \, dE_\lambda = \sum_k \lambda_k^n P_k$$

For polynomial $p(\lambda) = \sum_{n=0}^N a_n \lambda^n$:
$$p(A) = \sum_{n=0}^N a_n A^n$$

#### Example 2: Exponential Function

For $f(\lambda) = e^{i\alpha\lambda}$:
$$\boxed{e^{i\alpha A} = \int_{\sigma(A)} e^{i\alpha\lambda} \, dE_\lambda}$$

**Properties**:
- $e^{i\alpha A}$ is unitary when $A$ is self-adjoint
- $(e^{i\alpha A})^* = e^{-i\alpha A}$
- $e^{i\alpha A} e^{i\beta A} = e^{i(\alpha+\beta)A}$

#### Example 3: Square Root

For $A \geq 0$ (positive operator), define $f(\lambda) = \sqrt{\lambda}$:
$$\boxed{\sqrt{A} = \int_{\sigma(A)} \sqrt{\lambda} \, dE_\lambda}$$

**Properties**:
- $\sqrt{A} \geq 0$ (positive)
- $(\sqrt{A})^2 = A$
- $\sqrt{A}$ is unique positive square root

#### Example 4: Absolute Value and Polar Decomposition

For any self-adjoint $A$:
$$\boxed{|A| = \sqrt{A^2} = \int_{\sigma(A)} |\lambda| \, dE_\lambda}$$

**Positive and Negative Parts**:
$$A^+ = \frac{A + |A|}{2} = \int \lambda^+ \, dE_\lambda$$
$$A^- = \frac{|A| - A}{2} = \int \lambda^- \, dE_\lambda$$

where $\lambda^+ = \max(\lambda, 0)$ and $\lambda^- = \max(-\lambda, 0)$.

**Jordan Decomposition**: $A = A^+ - A^-$ and $|A| = A^+ + A^-$.

#### Example 5: Resolvent

For $\lambda \notin \sigma(A)$, define $f(\mu) = (\mu - \lambda)^{-1}$:
$$\boxed{(A - \lambda I)^{-1} = \int_{\sigma(A)} \frac{1}{\mu - \lambda} \, dE_\mu}$$

### 5. Functional Calculus for Normal Operators

The functional calculus extends to **normal operators** ($A^*A = AA^*$) on $\mathbb{C}$.

**Theorem (Functional Calculus for Normal Operators)**

For normal $N$ with spectrum $\sigma(N) \subseteq \mathbb{C}$ and continuous $f: \sigma(N) \to \mathbb{C}$:
$$f(N) = \int_{\sigma(N)} f(\lambda) \, dE_\lambda$$

**Special Case: Unitary Operators**

For unitary $U$, $\sigma(U) \subseteq \mathbb{T}$ (unit circle). We can write:
$$U = e^{iA}$$
for some self-adjoint $A$ with $\sigma(A) \subseteq [0, 2\pi)$.

### 6. Important Identities and Formulas

**Spectral Projections via Functional Calculus**

For Borel set $\Delta \subseteq \mathbb{R}$:
$$E(\Delta) = \chi_\Delta(A)$$

**Spectral Decomposition Identity**

$$\boxed{A = \int_{\sigma(A)} \lambda \, dE_\lambda = \text{id}(A)}$$

**Function of Function**

$$\boxed{f(g(A)) = (f \circ g)(A)}$$

**Commutator Identity**

If $f$ is differentiable:
$$[f(A), B] = \int_0^1 f'(A + t[A, B])[A, B] \, dt$$

(approximate formula, exact for special cases)

### 7. Quantum Mechanics Connection

The functional calculus is essential for quantum mechanics.

**Time Evolution Operator**

The Schrödinger equation $i\hbar \frac{\partial}{\partial t}|\psi\rangle = H|\psi\rangle$ has solution:
$$\boxed{|\psi(t)\rangle = e^{-iHt/\hbar}|\psi(0)\rangle}$$

Using functional calculus:
$$e^{-iHt/\hbar} = \int_{\sigma(H)} e^{-iEt/\hbar} \, dE_E$$

For discrete spectrum $H = \sum_n E_n |n\rangle\langle n|$:
$$|\psi(t)\rangle = \sum_n e^{-iE_n t/\hbar} \langle n|\psi(0)\rangle |n\rangle$$

**Functions of Observables**

If $A$ represents observable (energy, position, etc.), then $f(A)$ represents:
- $A^2$: mean square of the observable
- $e^{-\beta A}$: Boltzmann factor for thermal equilibrium
- $\theta(A)$: positive part (for conditional observables)

**Example: Harmonic Oscillator**

Hamiltonian: $H = \hbar\omega(a^\dagger a + \frac{1}{2})$

Number operator: $N = a^\dagger a$ with eigenvalues $n = 0, 1, 2, \ldots$

$$H = \hbar\omega(N + \frac{1}{2}) = f(N)$$

where $f(n) = \hbar\omega(n + \frac{1}{2})$.

The functional calculus gives:
$$e^{-iHt/\hbar} = e^{-i\omega t/2} e^{-i\omega N t} = e^{-i\omega t/2} \sum_{n=0}^\infty e^{-in\omega t}|n\rangle\langle n|$$

---

## Worked Examples

### Example 1: Computing $e^{iA}$ for a 2x2 Matrix

**Problem**: For $A = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$, compute $e^{iA}$.

**Solution**:

**Step 1**: Find eigenvalues and spectral decomposition.

$\det(A - \lambda I) = (1-\lambda)^2 - 1 = \lambda^2 - 2\lambda = \lambda(\lambda - 2)$

Eigenvalues: $\lambda_1 = 0$, $\lambda_2 = 2$.

Eigenvectors:
- $\lambda_1 = 0$: $v_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$
- $\lambda_2 = 2$: $v_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$

Spectral projections:
$$P_0 = \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}, \quad P_2 = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**Step 2**: Apply functional calculus.

$$e^{iA} = e^{i \cdot 0} P_0 + e^{i \cdot 2} P_2 = P_0 + e^{2i} P_2$$

$$= \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} + \frac{e^{2i}}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 1 + e^{2i} & -1 + e^{2i} \\ -1 + e^{2i} & 1 + e^{2i} \end{pmatrix}$$

**Step 3**: Simplify using Euler's formula.

$e^{2i} = \cos 2 + i\sin 2$

$$\boxed{e^{iA} = \frac{1}{2}\begin{pmatrix} 1 + \cos 2 + i\sin 2 & -1 + \cos 2 + i\sin 2 \\ -1 + \cos 2 + i\sin 2 & 1 + \cos 2 + i\sin 2 \end{pmatrix}}$$

**Verification**: $e^{iA}$ should be unitary. Check: $(e^{iA})^\dagger e^{iA} = e^{-iA}e^{iA} = I$. ✓

### Example 2: Square Root of a Positive Operator

**Problem**: Find $\sqrt{A}$ for $A = \begin{pmatrix} 5 & 2 \\ 2 & 2 \end{pmatrix}$.

**Solution**:

**Step 1**: Verify $A > 0$.

Eigenvalues: $\lambda = \frac{7 \pm \sqrt{9+16}}{2} = \frac{7 \pm 5}{2}$, so $\lambda_1 = 1$, $\lambda_2 = 6$.

Both positive, so $A > 0$. ✓

**Step 2**: Find eigenvectors and projections.

Eigenvector for $\lambda = 1$: $v_1 = \frac{1}{\sqrt{5}}\begin{pmatrix} -1 \\ 2 \end{pmatrix}$

Eigenvector for $\lambda = 6$: $v_2 = \frac{1}{\sqrt{5}}\begin{pmatrix} 2 \\ 1 \end{pmatrix}$

$$P_1 = \frac{1}{5}\begin{pmatrix} 1 & -2 \\ -2 & 4 \end{pmatrix}, \quad P_6 = \frac{1}{5}\begin{pmatrix} 4 & 2 \\ 2 & 1 \end{pmatrix}$$

**Step 3**: Apply $f(\lambda) = \sqrt{\lambda}$.

$$\sqrt{A} = \sqrt{1} \cdot P_1 + \sqrt{6} \cdot P_6 = P_1 + \sqrt{6} P_6$$

$$= \frac{1}{5}\begin{pmatrix} 1 & -2 \\ -2 & 4 \end{pmatrix} + \frac{\sqrt{6}}{5}\begin{pmatrix} 4 & 2 \\ 2 & 1 \end{pmatrix}$$

$$\boxed{\sqrt{A} = \frac{1}{5}\begin{pmatrix} 1 + 4\sqrt{6} & -2 + 2\sqrt{6} \\ -2 + 2\sqrt{6} & 4 + \sqrt{6} \end{pmatrix}}$$

**Verification**: $(\sqrt{A})^2 = A$.

$$\sqrt{A} \approx \begin{pmatrix} 2.159 & 0.580 \\ 0.580 & 1.290 \end{pmatrix}$$

$$(\sqrt{A})^2 \approx \begin{pmatrix} 5.0 & 2.0 \\ 2.0 & 2.0 \end{pmatrix} = A$$ ✓

### Example 3: Time Evolution in a Two-Level System

**Problem**: A spin-1/2 particle has Hamiltonian $H = \frac{\hbar\omega}{2}\sigma_z$ where $\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$. If the initial state is $|\psi(0)\rangle = \frac{1}{\sqrt{2}}(|{\uparrow}\rangle + |{\downarrow}\rangle)$, find $|\psi(t)\rangle$.

**Solution**:

**Step 1**: Compute the time evolution operator.

$$U(t) = e^{-iHt/\hbar} = e^{-i\omega t \sigma_z/2}$$

The spectral decomposition of $\sigma_z$:
- Eigenvalue $+1$: $|{\uparrow}\rangle\langle{\uparrow}| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$
- Eigenvalue $-1$: $|{\downarrow}\rangle\langle{\downarrow}| = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$

$$U(t) = e^{-i\omega t/2}|{\uparrow}\rangle\langle{\uparrow}| + e^{+i\omega t/2}|{\downarrow}\rangle\langle{\downarrow}|$$

$$= \begin{pmatrix} e^{-i\omega t/2} & 0 \\ 0 & e^{i\omega t/2} \end{pmatrix}$$

**Step 2**: Apply to initial state.

$$|\psi(t)\rangle = U(t)|\psi(0)\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} e^{-i\omega t/2} \\ e^{i\omega t/2} \end{pmatrix}$$

$$\boxed{|\psi(t)\rangle = \frac{1}{\sqrt{2}}(e^{-i\omega t/2}|{\uparrow}\rangle + e^{i\omega t/2}|{\downarrow}\rangle)}$$

**Step 3**: Physical interpretation.

The state precesses with frequency $\omega/2$. The spin component $\langle\sigma_z\rangle = 0$ for all $t$, but:

$$\langle\sigma_x\rangle = \cos(\omega t), \quad \langle\sigma_y\rangle = \sin(\omega t)$$

The spin vector rotates in the $xy$-plane—this is **Larmor precession**.

---

## Practice Problems

### Level 1: Direct Application

1. **Matrix Exponential**: For $A = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$, compute $e^{A}$ using spectral decomposition.

2. **Square Root**: Find $\sqrt{A}$ for $A = \begin{pmatrix} 4 & 0 \\ 0 & 9 \end{pmatrix}$.

3. **Absolute Value**: For $A = \begin{pmatrix} 1 & 2 \\ 2 & -2 \end{pmatrix}$, compute $|A| = \sqrt{A^2}$.

### Level 2: Intermediate

4. **Spectral Mapping**: If $\sigma(A) = [0, 2]$ and $f(\lambda) = \lambda^2 - 2\lambda + 1$, find $\sigma(f(A))$.

5. **Logarithm**: For positive $A$ with $\sigma(A) \subset (0, \infty)$, define $\log(A)$. Show $e^{\log A} = A$.

6. **Projection Formula**: Prove that $P = \chi_{\{1\}}(A)$ where $A$ is a self-adjoint operator with $A^2 = A$.

### Level 3: Challenging

7. **Duhamel's Formula**: Show that for time-dependent $H(t)$:
$$U(t) = I + \sum_{n=1}^\infty (-i)^n \int_0^t dt_1 \int_0^{t_1} dt_2 \cdots \int_0^{t_{n-1}} dt_n \, H(t_1)H(t_2)\cdots H(t_n)$$

8. **Fractional Powers**: For $A > 0$ and $\alpha \in (0,1)$, show:
$$A^\alpha = \frac{\sin(\pi\alpha)}{\pi} \int_0^\infty t^{\alpha-1}(t + A)^{-1}A \, dt$$

9. **Trotter-Kato Formula**: Prove $e^{A+B} = \lim_{n\to\infty}(e^{A/n}e^{B/n})^n$ for bounded $A, B$.

---

## Computational Lab: Functions of Operators

```python
"""
Day 249 Computational Lab: Functional Calculus
Computing and visualizing functions of operators
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from typing import Callable, List, Tuple

np.random.seed(42)

# =============================================================================
# Part 1: Functional Calculus via Spectral Decomposition
# =============================================================================

print("="*70)
print("PART 1: FUNCTIONAL CALCULUS VIA SPECTRAL DECOMPOSITION")
print("="*70)

def functional_calculus(A: np.ndarray, f: Callable) -> np.ndarray:
    """
    Compute f(A) using spectral decomposition.
    A must be Hermitian/self-adjoint.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    f_eigenvalues = np.array([f(ev) for ev in eigenvalues])

    # f(A) = V @ diag(f(lambda)) @ V^H
    return eigenvectors @ np.diag(f_eigenvalues) @ eigenvectors.conj().T

# Test matrix
A = np.array([[4, 2],
              [2, 1]], dtype=float)

print(f"Test matrix A:\n{A}")
print(f"Eigenvalues: {np.linalg.eigvalsh(A)}")

# Test various functions
print("\n--- Testing various functions f(A) ---")

# f(x) = x^2
A_squared_spectral = functional_calculus(A, lambda x: x**2)
A_squared_direct = A @ A
print(f"\nf(x) = x^2:")
print(f"Via spectral: \n{A_squared_spectral}")
print(f"Direct A^2:   \n{A_squared_direct}")
print(f"Match: {np.allclose(A_squared_spectral, A_squared_direct)}")

# f(x) = exp(x)
exp_A_spectral = functional_calculus(A, np.exp)
exp_A_scipy = linalg.expm(A)
print(f"\nf(x) = exp(x):")
print(f"Via spectral:\n{exp_A_spectral}")
print(f"Via scipy:   \n{exp_A_scipy}")
print(f"Match: {np.allclose(exp_A_spectral, exp_A_scipy)}")

# f(x) = sqrt(x)
sqrt_A_spectral = functional_calculus(A, np.sqrt)
sqrt_A_scipy = linalg.sqrtm(A)
print(f"\nf(x) = sqrt(x):")
print(f"Via spectral:\n{sqrt_A_spectral}")
print(f"Via scipy:   \n{np.real(sqrt_A_scipy)}")
print(f"Verification (sqrt(A))^2 = A: {np.allclose(sqrt_A_spectral @ sqrt_A_spectral, A)}")

# =============================================================================
# Part 2: Spectral Mapping Theorem Visualization
# =============================================================================

print("\n" + "="*70)
print("PART 2: SPECTRAL MAPPING THEOREM")
print("="*70)

def visualize_spectral_mapping(A: np.ndarray, f: Callable, f_name: str):
    """Visualize how f transforms the spectrum."""
    eigenvalues_A = np.linalg.eigvalsh(A)
    f_A = functional_calculus(A, f)
    eigenvalues_fA = np.linalg.eigvalsh(f_A)
    f_eigenvalues = np.array([f(ev) for ev in eigenvalues_A])

    print(f"\nFunction: {f_name}")
    print(f"sigma(A) = {eigenvalues_A}")
    print(f"f(sigma(A)) = {f_eigenvalues}")
    print(f"sigma(f(A)) = {eigenvalues_fA}")
    print(f"Match (spectral mapping theorem): {np.allclose(np.sort(f_eigenvalues), np.sort(eigenvalues_fA))}")

    return eigenvalues_A, f_eigenvalues, eigenvalues_fA

# Create test matrix with specific eigenvalues
eigenvals = np.array([1, 2, 3, 4])
V = linalg.orth(np.random.randn(4, 4))
A_test = V @ np.diag(eigenvals) @ V.T

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

functions = [
    (lambda x: x**2, '$f(\\lambda) = \\lambda^2$'),
    (np.exp, '$f(\\lambda) = e^\\lambda$'),
    (np.sin, '$f(\\lambda) = \\sin(\\lambda)$'),
    (lambda x: 1/(x+0.1), '$f(\\lambda) = 1/(\\lambda+0.1)$')
]

for ax, (f, f_name) in zip(axes.flat, functions):
    ev_A, f_ev, ev_fA = visualize_spectral_mapping(A_test, f, f_name)

    # Plot original spectrum
    ax.scatter(ev_A, np.zeros_like(ev_A), s=100, c='blue', marker='o',
               label='$\\sigma(A)$', zorder=5)

    # Plot transformed spectrum
    ax.scatter(f_ev, np.ones_like(f_ev), s=100, c='red', marker='^',
               label='$f(\\sigma(A))$', zorder=5)

    # Draw arrows showing mapping
    for orig, transformed in zip(ev_A, f_ev):
        ax.annotate('', xy=(transformed, 0.9), xytext=(orig, 0.1),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Value')
    ax.set_title(f_name)
    ax.legend()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['$\\sigma(A)$', '$\\sigma(f(A))$'])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectral_mapping_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: spectral_mapping_theorem.png")

# =============================================================================
# Part 3: Matrix Functions for Quantum Time Evolution
# =============================================================================

print("\n" + "="*70)
print("PART 3: QUANTUM TIME EVOLUTION")
print("="*70)

def time_evolution(H: np.ndarray, psi0: np.ndarray,
                   t_max: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time evolution |psi(t)> = exp(-iHt)|psi0>.
    Returns times and states.
    """
    times = np.linspace(0, t_max, n_steps)
    states = np.zeros((n_steps, len(psi0)), dtype=complex)

    # Get spectral decomposition of H
    energies, eigenstates = np.linalg.eigh(H)

    for i, t in enumerate(times):
        # U(t) = exp(-iHt) via spectral decomposition
        U_t = eigenstates @ np.diag(np.exp(-1j * energies * t)) @ eigenstates.conj().T
        states[i] = U_t @ psi0

    return times, states

# Two-level system (qubit)
omega = 2 * np.pi  # Frequency
H_qubit = 0.5 * omega * np.array([[1, 0], [0, -1]])  # H = omega/2 * sigma_z

# Initial state: |+> = (|0> + |1>)/sqrt(2)
psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)

times, states = time_evolution(H_qubit, psi0, t_max=2, n_steps=200)

# Compute expectation values of Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

exp_x = np.array([np.real(s.conj() @ sigma_x @ s) for s in states])
exp_y = np.array([np.real(s.conj() @ sigma_y @ s) for s in states])
exp_z = np.array([np.real(s.conj() @ sigma_z @ s) for s in states])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Expectation values vs time
ax1 = axes[0]
ax1.plot(times, exp_x, 'r-', linewidth=2, label='$\\langle\\sigma_x\\rangle$')
ax1.plot(times, exp_y, 'g-', linewidth=2, label='$\\langle\\sigma_y\\rangle$')
ax1.plot(times, exp_z, 'b-', linewidth=2, label='$\\langle\\sigma_z\\rangle$')
ax1.set_xlabel('Time $t$')
ax1.set_ylabel('Expectation value')
ax1.set_title('Qubit Time Evolution (Larmor Precession)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Bloch sphere trajectory (projected)
ax2 = axes[1]
ax2.plot(exp_x, exp_y, 'b-', linewidth=1.5, alpha=0.7)
ax2.scatter([exp_x[0]], [exp_y[0]], s=100, c='green', marker='o', label='Start')
ax2.scatter([exp_x[-1]], [exp_y[-1]], s=100, c='red', marker='s', label='End')

# Draw unit circle (equator of Bloch sphere)
theta = np.linspace(0, 2*np.pi, 100)
ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

ax2.set_xlabel('$\\langle\\sigma_x\\rangle$')
ax2.set_ylabel('$\\langle\\sigma_y\\rangle$')
ax2.set_title('Bloch Sphere (xy-plane projection)')
ax2.set_aspect('equal')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quantum_time_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: quantum_time_evolution.png")

# =============================================================================
# Part 4: Operator Functions - Square Root, Absolute Value, Sign
# =============================================================================

print("\n" + "="*70)
print("PART 4: SPECIAL OPERATOR FUNCTIONS")
print("="*70)

# Create a self-adjoint matrix with both positive and negative eigenvalues
A_mixed = np.array([[2, 1, 0],
                    [1, -1, 1],
                    [0, 1, 1]], dtype=float)

eigenvalues_mixed = np.linalg.eigvalsh(A_mixed)
print(f"Matrix A with mixed eigenvalues:")
print(f"A = \n{A_mixed}")
print(f"Eigenvalues: {eigenvalues_mixed}")

# Absolute value |A|
abs_A = functional_calculus(A_mixed, np.abs)
print(f"\n|A| = sqrt(A^2):")
print(f"{abs_A}")

# Positive and negative parts
A_plus = functional_calculus(A_mixed, lambda x: max(x, 0))
A_minus = functional_calculus(A_mixed, lambda x: max(-x, 0))
print(f"\nA+ (positive part):\n{A_plus}")
print(f"\nA- (negative part):\n{A_minus}")

# Verify Jordan decomposition: A = A+ - A-
print(f"\nVerification A = A+ - A-: {np.allclose(A_mixed, A_plus - A_minus)}")
print(f"Verification |A| = A+ + A-: {np.allclose(abs_A, A_plus + A_minus)}")

# Sign function
sign_A = functional_calculus(A_mixed, np.sign)
print(f"\nsign(A):\n{sign_A}")

# Verify A = |A| * sign(A)... approximately (sign is discontinuous at 0)
print(f"A ≈ |A| @ sign(A): {np.allclose(A_mixed, abs_A @ sign_A, atol=1e-10)}")

# =============================================================================
# Part 5: Comparison of Matrix Function Methods
# =============================================================================

print("\n" + "="*70)
print("PART 5: COMPARISON OF COMPUTATION METHODS")
print("="*70)

def matrix_exp_taylor(A: np.ndarray, n_terms: int = 20) -> np.ndarray:
    """Compute exp(A) via Taylor series."""
    result = np.eye(A.shape[0], dtype=complex)
    A_power = np.eye(A.shape[0], dtype=complex)
    for n in range(1, n_terms + 1):
        A_power = A_power @ A / n
        result = result + A_power
    return result

def matrix_exp_pade(A: np.ndarray) -> np.ndarray:
    """Compute exp(A) via scipy (uses Pade approximation)."""
    return linalg.expm(A)

def matrix_exp_spectral(A: np.ndarray) -> np.ndarray:
    """Compute exp(A) via spectral decomposition."""
    return functional_calculus(A, np.exp)

# Test on various matrices
test_matrices = [
    (np.array([[1, 0], [0, 2]]), "Diagonal"),
    (np.array([[0, 1], [-1, 0]]), "Skew-symmetric"),
    (np.array([[1, 2], [2, 1]]), "Symmetric"),
    (np.random.randn(5, 5), "Random 5x5")
]

print("\nComparing matrix exponential methods:")
print("-" * 60)

for A, name in test_matrices:
    # Make symmetric for spectral method
    A = (A + A.T) / 2

    exp_taylor = matrix_exp_taylor(A)
    exp_pade = matrix_exp_pade(A)
    exp_spectral = matrix_exp_spectral(A)

    err_taylor = np.linalg.norm(exp_taylor - exp_pade)
    err_spectral = np.linalg.norm(exp_spectral - exp_pade)

    print(f"\n{name} matrix:")
    print(f"  ||Taylor - Pade||   = {err_taylor:.2e}")
    print(f"  ||Spectral - Pade|| = {err_spectral:.2e}")

# =============================================================================
# Part 6: Visualizing f(A) for Various Functions
# =============================================================================

print("\n" + "="*70)
print("PART 6: VISUALIZATION OF f(A)")
print("="*70)

# Create a nice symmetric matrix
A_vis = np.array([[3, 1],
                  [1, 3]], dtype=float)
eigenvalues_vis = np.linalg.eigvalsh(A_vis)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

functions_vis = [
    (lambda x: x, 'Identity: $f(\\lambda) = \\lambda$'),
    (lambda x: x**2, 'Square: $f(\\lambda) = \\lambda^2$'),
    (np.sqrt, 'Square root: $f(\\lambda) = \\sqrt{\\lambda}$'),
    (np.exp, 'Exponential: $f(\\lambda) = e^\\lambda$'),
    (np.log, 'Logarithm: $f(\\lambda) = \\ln(\\lambda)$'),
    (lambda x: np.sin(x), 'Sine: $f(\\lambda) = \\sin(\\lambda)$')
]

for ax, (f, f_name) in zip(axes.flat, functions_vis):
    # Compute f(A)
    f_A = functional_calculus(A_vis, f)

    # Visualize as heatmap
    im = ax.imshow(np.real(f_A), cmap='coolwarm')
    ax.set_title(f_name)

    # Add values as text
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{np.real(f_A[i, j]):.3f}',
                          ha='center', va='center', fontsize=12)

    plt.colorbar(im, ax=ax)

plt.suptitle(f'Functions of A with eigenvalues {eigenvalues_vis}', fontsize=14)
plt.tight_layout()
plt.savefig('operator_functions_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: operator_functions_visualization.png")

# =============================================================================
# Part 7: Thermal States and Partition Functions
# =============================================================================

print("\n" + "="*70)
print("PART 7: THERMAL STATES IN QUANTUM MECHANICS")
print("="*70)

# Quantum harmonic oscillator (truncated)
n_levels = 20
omega = 1.0

# Hamiltonian: H = omega * (n + 1/2)
H_ho = np.diag(omega * (np.arange(n_levels) + 0.5))

# Thermal density matrix: rho = exp(-beta*H) / Z
def thermal_state(H: np.ndarray, beta: float) -> Tuple[np.ndarray, float]:
    """Compute thermal density matrix and partition function."""
    exp_minus_beta_H = functional_calculus(H, lambda x: np.exp(-beta * x))
    Z = np.trace(exp_minus_beta_H)
    rho = exp_minus_beta_H / Z
    return rho, Z

# Compute for various temperatures
betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Population distribution
ax1 = axes[0]
for beta in betas:
    rho, Z = thermal_state(H_ho, beta)
    populations = np.diag(np.real(rho))
    T = 1/beta
    ax1.plot(range(n_levels), populations, 'o-', label=f'T = {T:.2f}')

ax1.set_xlabel('Energy level n')
ax1.set_ylabel('Population $\\rho_{nn}$')
ax1.set_title('Thermal State Populations')
ax1.legend()
ax1.set_xlim(0, 15)
ax1.grid(True, alpha=0.3)

# Plot 2: Average energy vs temperature
ax2 = axes[1]
temperatures = np.linspace(0.1, 5, 50)
avg_energies = []
for T in temperatures:
    beta = 1/T
    rho, Z = thermal_state(H_ho, beta)
    E_avg = np.real(np.trace(rho @ H_ho))
    avg_energies.append(E_avg)

ax2.plot(temperatures, avg_energies, 'b-', linewidth=2)
ax2.axhline(y=omega/2, color='r', linestyle='--', label='Ground state energy')
ax2.set_xlabel('Temperature T')
ax2.set_ylabel('Average energy $\\langle H \\rangle$')
ax2.set_title('Average Energy vs Temperature')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thermal_states.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: thermal_states.png")

# Print summary
print("\nThermal state summary:")
for beta in [0.5, 1.0, 2.0]:
    rho, Z = thermal_state(H_ho, beta)
    E_avg = np.real(np.trace(rho @ H_ho))
    print(f"  beta = {beta}: <H> = {E_avg:.4f}, Z = {Z:.4f}")

print("\n" + "="*70)
print("LAB COMPLETE")
print("="*70)
print("""
Key takeaways:
1. Functional calculus: f(A) = sum f(lambda_i) P_i for spectral decomposition
2. Spectral mapping theorem: sigma(f(A)) = f(sigma(A)) for continuous f
3. Time evolution: U(t) = exp(-iHt) computed via spectral decomposition
4. Special functions: |A|, A+, A-, sign(A) defined via functional calculus
5. Thermal states: rho = exp(-beta*H)/Z uses matrix exponential
6. Multiple methods: Taylor series, Pade, spectral all compute exp(A)
""")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Functional calculus | $f(A) = \int_{\sigma(A)} f(\lambda) \, dE_\lambda$ |
| Spectral mapping | $\sigma(f(A)) = \overline{f(\sigma(A))}$ |
| Matrix exponential | $e^A = \sum_{n=0}^\infty \frac{A^n}{n!}$ |
| Time evolution | $U(t) = e^{-iHt/\hbar}$ |
| Square root | $\sqrt{A} = \int \sqrt{\lambda} \, dE_\lambda$ for $A \geq 0$ |
| Absolute value | $|A| = \sqrt{A^2}$ |
| Jordan decomposition | $A = A^+ - A^-$, $|A| = A^+ + A^-$ |

### Main Takeaways

1. **Functional calculus defines $f(A)$**: For self-adjoint $A$ and measurable $f$, we have $f(A) = \int f(\lambda) dE_\lambda$.

2. **Spectral mapping theorem**: The spectrum transforms under $f$ as $\sigma(f(A)) = f(\sigma(A))$ (closure for discontinuous $f$).

3. **Key operator functions**: Exponential, square root, absolute value, sign, and projections are all defined via functional calculus.

4. **Quantum time evolution**: $|\psi(t)\rangle = e^{-iHt/\hbar}|\psi(0)\rangle$ uses the matrix exponential.

5. **Thermal states**: $\rho = e^{-\beta H}/Z$ describes quantum systems at temperature $T = 1/\beta$.

6. **Computational methods**: Spectral decomposition, Taylor series, and Pade approximants all compute matrix functions.

---

## Daily Checklist

- [ ] I can define the continuous functional calculus
- [ ] I understand the Borel functional calculus extension
- [ ] I can apply the spectral mapping theorem
- [ ] I can compute $e^{iA}$, $\sqrt{A}$, $|A|$ via spectral decomposition
- [ ] I understand how functional calculus relates to quantum time evolution
- [ ] I can derive the Jordan decomposition $A = A^+ - A^-$
- [ ] I can compute thermal states using matrix exponentials
- [ ] I completed the computational lab

---

## Preview: Day 250

Tomorrow we enter the crucial territory of **unbounded operators**. In quantum mechanics, the most important observables—position $\hat{x}$ and momentum $\hat{p}$—are unbounded! We'll learn:
- How to define unbounded operators on dense domains
- The difference between symmetric, self-adjoint, and essentially self-adjoint
- The von Neumann criterion for self-adjointness
- How to extend the spectral theorem to unbounded operators

---

*"The functional calculus is the natural language for expressing physical laws in quantum mechanics. Every function of an observable—from time evolution to thermodynamics—flows from this single construction."*
— John von Neumann
