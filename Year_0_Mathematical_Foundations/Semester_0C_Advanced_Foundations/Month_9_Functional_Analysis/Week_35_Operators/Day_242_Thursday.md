# Day 242: Self-Adjoint and Unitary Operators

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Self-adjoint operators and spectral properties |
| Afternoon | 3 hours | Problems: Unitary operators and normal operators |
| Evening | 2 hours | Computational lab: Quantum mechanical applications |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** self-adjoint, unitary, and normal operators
2. **Prove** that self-adjoint operators have real spectrum
3. **Prove** that unitary operators preserve inner products and norms
4. **Verify** operator classification for specific examples
5. **Explain** why observables must be self-adjoint and symmetries must be unitary
6. **Connect** normal operators to the spectral theorem

---

## 1. Core Content: Self-Adjoint Operators

### 1.1 Definition and Characterization

**Definition**: A bounded operator $A: \mathcal{H} \to \mathcal{H}$ is **self-adjoint** (or **Hermitian**) if:

$$\boxed{A = A^\dagger}$$

Equivalently: $\langle Ax, y \rangle = \langle x, Ay \rangle$ for all $x, y \in \mathcal{H}$.

**Theorem (Characterization)**: $A$ is self-adjoint if and only if $\langle Ax, x \rangle \in \mathbb{R}$ for all $x \in \mathcal{H}$.

**Proof**:

($\Rightarrow$) If $A = A^\dagger$:
$$\overline{\langle Ax, x \rangle} = \langle x, Ax \rangle = \langle A^\dagger x, x \rangle = \langle Ax, x \rangle$$

So $\langle Ax, x \rangle = \overline{\langle Ax, x \rangle}$, meaning it's real.

($\Leftarrow$) Assume $\langle Ax, x \rangle \in \mathbb{R}$ for all $x$.

Using the polarization identity:
$$\langle Ax, y \rangle = \frac{1}{4}\sum_{k=0}^3 i^k \langle A(x + i^k y), x + i^k y \rangle$$

Each term $\langle A(x + i^k y), x + i^k y \rangle$ is real.

Similarly:
$$\langle x, Ay \rangle = \overline{\langle Ay, x \rangle} = \frac{1}{4}\sum_{k=0}^3 \overline{i^k} \langle A(y + i^k x), y + i^k x \rangle$$

A careful calculation shows these are equal, so $\langle Ax, y \rangle = \langle x, Ay \rangle$, meaning $A = A^\dagger$. $\square$

### 1.2 Spectral Properties of Self-Adjoint Operators

**Definition**: The **spectrum** of $A$ is:
$$\sigma(A) = \{\lambda \in \mathbb{C} : A - \lambda I \text{ is not invertible}\}$$

**Theorem**: If $A$ is self-adjoint, then $\sigma(A) \subseteq \mathbb{R}$.

**Proof**: We show that if $\lambda = a + bi$ with $b \neq 0$, then $A - \lambda I$ is invertible.

For any $x \in \mathcal{H}$:
$$\|(A - \lambda I)x\|^2 = \langle (A - \lambda I)x, (A - \lambda I)x \rangle$$
$$= \langle (A - aI)x - ibx, (A - aI)x - ibx \rangle$$
$$= \|(A - aI)x\|^2 + b^2\|x\|^2 + ib\langle x, (A-aI)x \rangle - ib\langle (A-aI)x, x \rangle$$

Since $A - aI$ is self-adjoint (if $A$ is), $\langle (A-aI)x, x \rangle \in \mathbb{R}$, so:
$$\|(A - \lambda I)x\|^2 = \|(A - aI)x\|^2 + b^2\|x\|^2 \geq b^2\|x\|^2$$

Thus $\|(A - \lambda I)x\| \geq |b| \|x\|$, so $A - \lambda I$ is bounded below.

By a similar argument, $(A - \lambda I)^\dagger = A - \bar{\lambda}I$ is also bounded below.

This implies $A - \lambda I$ has closed range equal to $\mathcal{H}$, hence is invertible. $\square$

**Theorem**: Eigenvectors of a self-adjoint operator corresponding to distinct eigenvalues are orthogonal.

**Proof**: Let $Ax = \lambda x$ and $Ay = \mu y$ with $\lambda \neq \mu$ (both real).

$$\lambda \langle x, y \rangle = \langle \lambda x, y \rangle = \langle Ax, y \rangle = \langle x, Ay \rangle = \langle x, \mu y \rangle = \mu \langle x, y \rangle$$

So $(\lambda - \mu)\langle x, y \rangle = 0$. Since $\lambda \neq \mu$, $\langle x, y \rangle = 0$. $\square$

### 1.3 The Numerical Range

**Definition**: The **numerical range** (or **field of values**) of $A$ is:
$$W(A) = \{\langle Ax, x \rangle : \|x\| = 1\}$$

**Theorem**: For self-adjoint $A$:
1. $W(A) \subseteq \mathbb{R}$
2. $W(A) = [\lambda_{\min}, \lambda_{\max}]$ where $\lambda_{\min} = \inf_{\|x\|=1}\langle Ax, x\rangle$ and $\lambda_{\max} = \sup_{\|x\|=1}\langle Ax, x\rangle$
3. $\sigma(A) \subseteq \overline{W(A)}$
4. $\|A\| = \max(|\lambda_{\min}|, |\lambda_{\max}|)$

---

## 2. Unitary Operators

### 2.1 Definition and Characterization

**Definition**: A bounded operator $U: \mathcal{H} \to \mathcal{H}$ is **unitary** if:

$$\boxed{U^\dagger U = UU^\dagger = I}$$

Equivalently, $U$ is invertible with $U^{-1} = U^\dagger$.

**Theorem (Characterizations)**: The following are equivalent:
1. $U$ is unitary
2. $U$ is surjective and $\langle Ux, Uy \rangle = \langle x, y \rangle$ for all $x, y$
3. $U$ is surjective and $\|Ux\| = \|x\|$ for all $x$

**Proof**:

(1) $\Rightarrow$ (2): $\langle Ux, Uy \rangle = \langle x, U^\dagger Uy \rangle = \langle x, y \rangle$. Surjectivity follows from $UU^\dagger = I$.

(2) $\Rightarrow$ (3): Set $y = x$ to get $\|Ux\|^2 = \|x\|^2$.

(3) $\Rightarrow$ (1): Since $\|Ux\| = \|x\|$, we have $\|U^\dagger Ux\|^2 = \langle U^\dagger Ux, U^\dagger Ux\rangle$...

Actually, let's use polarization. If $\|Ux\| = \|x\|$ for all $x$, then by the polarization identity:
$$\langle Ux, Uy \rangle = \langle x, y \rangle$$

So $\langle x, U^\dagger Uy \rangle = \langle x, y \rangle$ for all $x$, implying $U^\dagger U = I$.

Surjectivity of $U$ implies $U$ has a right inverse. Combined with $U^\dagger U = I$, we get $UU^\dagger = I$. $\square$

### 2.2 Properties of Unitary Operators

**Theorem**: If $U$ is unitary, then:
1. $\|U\| = 1$
2. $|\lambda| = 1$ for every eigenvalue $\lambda$ of $U$
3. Eigenvectors for distinct eigenvalues are orthogonal
4. $\sigma(U) \subseteq \{z \in \mathbb{C} : |z| = 1\}$ (the unit circle)

**Proof of (2)**: If $Ux = \lambda x$ with $x \neq 0$:
$$\|x\|^2 = \|Ux\|^2 = \|\lambda x\|^2 = |\lambda|^2 \|x\|^2$$

So $|\lambda| = 1$. $\square$

**Important**: Unitary operators are the "rotations" of Hilbert space. They preserve all geometric structure (lengths and angles).

### 2.3 Isometries vs. Unitaries

**Definition**: An operator $V: \mathcal{H} \to \mathcal{K}$ is an **isometry** if $\|Vx\| = \|x\|$ for all $x$.

**Note**: Every unitary is an isometry. But isometries need not be unitary!

**Example**: The right shift $S_R$ on $\ell^2$ satisfies $S_R^\dagger S_R = I$ (isometry) but $S_R S_R^\dagger \neq I$ (not unitary).

| Property | Isometry | Unitary |
|----------|----------|---------|
| $V^\dagger V = I$ | Yes | Yes |
| $VV^\dagger = I$ | Not necessarily | Yes |
| Preserves norm | Yes | Yes |
| Surjective | Not necessarily | Yes |

---

## 3. Normal Operators

### 3.1 Definition

**Definition**: An operator $A$ is **normal** if it commutes with its adjoint:

$$\boxed{AA^\dagger = A^\dagger A}$$

**Examples**:
- Self-adjoint operators ($A = A^\dagger$)
- Unitary operators ($U^\dagger U = UU^\dagger = I$)
- Skew-adjoint operators ($A^\dagger = -A$)

### 3.2 Characterization

**Theorem**: $A$ is normal if and only if $\|Ax\| = \|A^\dagger x\|$ for all $x$.

**Proof**:

$(\Rightarrow)$ If $A$ is normal:
$$\|Ax\|^2 = \langle Ax, Ax \rangle = \langle A^\dagger Ax, x \rangle = \langle AA^\dagger x, x \rangle = \langle A^\dagger x, A^\dagger x \rangle = \|A^\dagger x\|^2$$

$(\Leftarrow)$ If $\|Ax\| = \|A^\dagger x\|$ for all $x$:
$$\langle A^\dagger Ax, x \rangle = \|Ax\|^2 = \|A^\dagger x\|^2 = \langle AA^\dagger x, x \rangle$$

So $\langle (A^\dagger A - AA^\dagger)x, x \rangle = 0$ for all $x$.

Since $A^\dagger A - AA^\dagger$ is self-adjoint, this implies $A^\dagger A = AA^\dagger$. $\square$

### 3.3 Spectral Properties of Normal Operators

**Theorem**: For a normal operator $A$:
1. Eigenvectors for distinct eigenvalues are orthogonal
2. $\ker(A - \lambda I) = \ker(A^\dagger - \bar{\lambda}I)$
3. If $A$ is normal and compact, then $A$ has an orthonormal basis of eigenvectors

**Proof of (1)**: Let $Ax = \lambda x$ and $Ay = \mu y$ with $\lambda \neq \mu$.

Since $A$ is normal, $A^\dagger x = \bar{\lambda}x$ (prove this!).

$$\lambda\langle x, y \rangle = \langle Ax, y \rangle = \langle x, A^\dagger y \rangle = \langle x, \bar{\mu}y \rangle = \mu\langle x, y \rangle$$

So $(\lambda - \mu)\langle x, y \rangle = 0$, giving $\langle x, y \rangle = 0$. $\square$

---

## 4. Quantum Mechanics Connection

### 4.1 Why Observables Must Be Self-Adjoint

**Postulate**: In quantum mechanics, every observable corresponds to a self-adjoint operator.

**Reason 1 (Real Eigenvalues)**: Measurement outcomes must be real numbers. Self-adjoint operators have real spectrum.

**Reason 2 (Orthogonal Eigenstates)**: Distinct measurement outcomes should correspond to distinguishable (orthogonal) states.

**Reason 3 (Completeness)**: The spectral theorem guarantees a complete set of eigenstates.

**Reason 4 (Probability Conservation)**: The expected value $\langle\psi|A|\psi\rangle$ must be real for $A$ self-adjoint.

### 4.2 Why Symmetries Must Be Unitary

**Postulate**: Every symmetry transformation in quantum mechanics is represented by a unitary (or anti-unitary) operator.

**Reason 1 (Probability Preservation)**: If $|\psi\rangle \to U|\psi\rangle$, then:
$$|\langle\phi|U|\psi\rangle|^2 = |\langle U^\dagger\phi|\psi\rangle|^2 = |\langle\phi|\psi\rangle|^2$$

Transition probabilities are preserved.

**Reason 2 (Normalization)**: $\|U|\psi\rangle\| = \||\psi\rangle\| = 1$.

**Reason 3 (Reversibility)**: $U^{-1} = U^\dagger$ exists, so symmetries are reversible.

### 4.3 Time Evolution

The time evolution operator $U(t) = e^{-iHt/\hbar}$ where $H$ is the Hamiltonian.

**Verify Unitarity**:
$$U(t)^\dagger = (e^{-iHt/\hbar})^\dagger = e^{iH^\dagger t/\hbar} = e^{iHt/\hbar}$$ (since $H$ is self-adjoint)

$$U(t)^\dagger U(t) = e^{iHt/\hbar} e^{-iHt/\hbar} = e^0 = I$$

Similarly $U(t)U(t)^\dagger = I$. So $U(t)$ is unitary for all $t$.

### 4.4 Classification Table

| Operator Type | Defining Property | QM Role | Examples |
|---------------|-------------------|---------|----------|
| Self-adjoint | $A = A^\dagger$ | Observables | $\hat{x}, \hat{p}, \hat{H}$ |
| Unitary | $U^\dagger U = I$ | Symmetries, evolution | $e^{-iHt/\hbar}$, rotations |
| Normal | $[A, A^\dagger] = 0$ | Diagonalizable | Self-adjoint, unitary |
| Projection | $P^2 = P = P^\dagger$ | Measurement | $|n\rangle\langle n|$ |
| Isometry | $V^\dagger V = I$ | Partial symmetry | Right shift |

---

## 5. Worked Examples

### Example 1: Verifying Self-Adjointness

**Problem**: Show that the Pauli matrices $\sigma_x, \sigma_y, \sigma_z$ are self-adjoint.

**Solution**:

$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

For $\sigma_x$: $\sigma_x^\dagger = \overline{\sigma_x}^T = \sigma_x^T = \sigma_x$ ✓

For $\sigma_y$: $\sigma_y^\dagger = \overline{\begin{pmatrix} 0 & i \\ -i & 0 \end{pmatrix}} = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} = \sigma_y$ ✓

For $\sigma_z$: $\sigma_z^\dagger = \sigma_z^T = \sigma_z$ ✓

All three Pauli matrices are self-adjoint, confirming they represent spin observables.

**Eigenvalues**: Each has eigenvalues $\pm 1$ (real, as expected). $\square$

---

### Example 2: Unitary Rotation

**Problem**: Show that $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ is unitary (for real $\theta$).

**Solution**:

$$R(\theta)^\dagger = R(\theta)^T = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} = R(-\theta)$$

$$R(\theta)^\dagger R(\theta) = R(-\theta)R(\theta) = R(0) = I$$

So $R(\theta)$ is unitary.

**Eigenvalues**:
$$\det(R(\theta) - \lambda I) = (\cos\theta - \lambda)^2 + \sin^2\theta = \lambda^2 - 2\lambda\cos\theta + 1 = 0$$

$$\lambda = \cos\theta \pm i\sin\theta = e^{\pm i\theta}$$

Both eigenvalues have $|\lambda| = 1$, as required for unitary operators. $\square$

---

### Example 3: Normal but Not Self-Adjoint or Unitary

**Problem**: Show that $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ is NOT normal, but $B = \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$ IS normal.

**Solution**:

**For A**:
$$A^\dagger = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$$

$$AA^\dagger = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$$

$$A^\dagger A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}$$

$AA^\dagger \neq A^\dagger A$, so $A$ is **not normal**.

**For B**:
$$B^\dagger = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$

$$BB^\dagger = \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$

$$B^\dagger B = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$

$BB^\dagger = B^\dagger B$, so $B$ is **normal**.

Note: $B = \sqrt{2}R(\pi/4)$ (rotation scaled), which explains its normality.

Eigenvalues of $B$: $\lambda = 1 \pm i$, which are not real (not self-adjoint) and don't have $|\lambda|=1$ (not unitary). $\square$

---

## 6. Practice Problems

### Level 1: Direct Application

1. Verify that the matrix $A = \begin{pmatrix} 2 & 1+i \\ 1-i & 3 \end{pmatrix}$ is self-adjoint. Find its eigenvalues.

2. Show that $U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ (Hadamard gate) is unitary.

3. Classify each operator as self-adjoint, unitary, normal, or none:
   - $\begin{pmatrix} i & 0 \\ 0 & -i \end{pmatrix}$
   - $\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$
   - $\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$

### Level 2: Intermediate

4. **Prove**: If $A$ is self-adjoint and $f: \mathbb{R} \to \mathbb{R}$ is a polynomial, then $f(A)$ is self-adjoint.

5. **Prove**: If $U$ and $V$ are unitary, then so is $UV$.

6. **Quantum Connection**: The time evolution from $|\psi(0)\rangle$ to $|\psi(t)\rangle = U(t)|\psi(0)\rangle$ must preserve $\langle\psi|\psi\rangle = 1$. Show this requires $U(t)$ to be unitary.

### Level 3: Challenging

7. **Prove**: An operator $A$ is normal if and only if $A = S + iT$ where $S, T$ are self-adjoint and $[S, T] = 0$.

8. **Prove**: If $N$ is normal with $\sigma(N) \subseteq \mathbb{R}$, then $N$ is self-adjoint. (Hint: Use the spectral theorem for normal operators.)

9. **Research problem**: Wigner's theorem states that every symmetry of quantum mechanics is implemented by a unitary or anti-unitary operator. Research and summarize the key ideas of this theorem.

---

## 7. Computational Lab: Self-Adjoint and Unitary Operators

```python
"""
Day 242 Computational Lab: Self-Adjoint and Unitary Operators
Classification, verification, and quantum applications
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, norm, eigh, eig

# ============================================================
# Part 1: Operator Classification
# ============================================================

def classify_operator(A, name="A"):
    """
    Classify an operator as self-adjoint, unitary, normal, or none.
    """
    A_dag = A.conj().T
    n = A.shape[0]
    I = np.eye(n)

    is_self_adjoint = np.allclose(A, A_dag)
    is_unitary = np.allclose(A @ A_dag, I) and np.allclose(A_dag @ A, I)
    is_normal = np.allclose(A @ A_dag, A_dag @ A)
    is_isometry = np.allclose(A_dag @ A, I)

    print(f"\nOperator: {name}")
    print(f"  Self-adjoint (A = A†): {is_self_adjoint}")
    print(f"  Unitary (A†A = AA† = I): {is_unitary}")
    print(f"  Normal (AA† = A†A): {is_normal}")
    print(f"  Isometry (A†A = I): {is_isometry}")

    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvals(A)
    print(f"  Eigenvalues: {eigenvalues}")

    if is_self_adjoint:
        print(f"    → All eigenvalues are real: {np.allclose(eigenvalues.imag, 0)}")
    if is_unitary or is_isometry:
        print(f"    → All eigenvalues on unit circle: {np.allclose(np.abs(eigenvalues), 1)}")

    return {'self_adjoint': is_self_adjoint, 'unitary': is_unitary,
            'normal': is_normal, 'isometry': is_isometry}

def operator_classification_examples():
    """
    Classify various operators.
    """
    print("=" * 60)
    print("Part 1: Operator Classification")
    print("=" * 60)

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    classify_operator(sigma_x, "σ_x (Pauli X)")
    classify_operator(sigma_y, "σ_y (Pauli Y)")
    classify_operator(sigma_z, "σ_z (Pauli Z)")

    # Hadamard gate
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    classify_operator(H, "H (Hadamard)")

    # Phase gate
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    classify_operator(S, "S (Phase gate)")

    # Upper triangular (not normal)
    upper = np.array([[1, 1], [0, 1]], dtype=complex)
    classify_operator(upper, "Upper triangular")

    # Nilpotent
    nilp = np.array([[0, 1], [0, 0]], dtype=complex)
    classify_operator(nilp, "Nilpotent")

# ============================================================
# Part 2: Spectral Properties
# ============================================================

def spectral_properties():
    """
    Visualize spectral properties of different operator types.
    """
    print("\n" + "=" * 60)
    print("Part 2: Spectral Properties")
    print("=" * 60)

    np.random.seed(42)
    n = 20

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Self-adjoint: random Hermitian matrix
    A_hermitian = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A_hermitian = (A_hermitian + A_hermitian.conj().T) / 2
    eig_herm = np.linalg.eigvals(A_hermitian)

    axes[0, 0].scatter(eig_herm.real, eig_herm.imag, s=50, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Re(λ)')
    axes[0, 0].set_ylabel('Im(λ)')
    axes[0, 0].set_title('Self-Adjoint: Spectrum on Real Axis')
    axes[0, 0].grid(True, alpha=0.3)

    # Unitary: random unitary matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n) + 1j * np.random.randn(n, n))
    eig_unit = np.linalg.eigvals(Q)

    theta = np.linspace(0, 2*np.pi, 100)
    axes[0, 1].plot(np.cos(theta), np.sin(theta), 'r--', label='Unit circle')
    axes[0, 1].scatter(eig_unit.real, eig_unit.imag, s=50, alpha=0.7)
    axes[0, 1].set_xlabel('Re(λ)')
    axes[0, 1].set_ylabel('Im(λ)')
    axes[0, 1].set_title('Unitary: Spectrum on Unit Circle')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Normal (but not self-adjoint or unitary)
    # Construct diagonal with complex eigenvalues
    D = np.diag(np.random.randn(n) + 1j * np.random.randn(n))
    U, _ = np.linalg.qr(np.random.randn(n, n) + 1j * np.random.randn(n, n))
    A_normal = U @ D @ U.conj().T
    eig_norm = np.linalg.eigvals(A_normal)

    axes[1, 0].scatter(eig_norm.real, eig_norm.imag, s=50, alpha=0.7)
    axes[1, 0].set_xlabel('Re(λ)')
    axes[1, 0].set_ylabel('Im(λ)')
    axes[1, 0].set_title('Normal: Eigenvalues Can Be Anywhere')
    axes[1, 0].grid(True, alpha=0.3)

    # Not normal: random matrix
    A_general = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    eig_gen = np.linalg.eigvals(A_general)

    axes[1, 1].scatter(eig_gen.real, eig_gen.imag, s=50, alpha=0.7)
    axes[1, 1].set_xlabel('Re(λ)')
    axes[1, 1].set_ylabel('Im(λ)')
    axes[1, 1].set_title('General (Not Normal): Eigenvalues Anywhere')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Spectral Properties by Operator Type', fontsize=14)
    plt.tight_layout()
    plt.savefig('spectral_properties.png', dpi=150)
    plt.show()

# ============================================================
# Part 3: Unitary Time Evolution
# ============================================================

def unitary_time_evolution():
    """
    Demonstrate unitary time evolution in quantum mechanics.
    """
    print("\n" + "=" * 60)
    print("Part 3: Unitary Time Evolution")
    print("=" * 60)

    # Two-level system with Hamiltonian H = ω σ_x
    omega = 1.0
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    H = omega * sigma_x

    # Verify H is self-adjoint
    print(f"H is self-adjoint: {np.allclose(H, H.conj().T)}")

    # Time evolution U(t) = exp(-iHt)
    t_values = np.linspace(0, 4*np.pi, 200)

    # Initial state |0⟩
    psi_0 = np.array([1, 0], dtype=complex)

    probs_0 = []
    probs_1 = []
    unitarity_check = []

    for t in t_values:
        U = expm(-1j * H * t)

        # Check unitarity
        unitarity_check.append(np.allclose(U @ U.conj().T, np.eye(2)))

        # Evolve state
        psi_t = U @ psi_0

        probs_0.append(np.abs(psi_t[0])**2)
        probs_1.append(np.abs(psi_t[1])**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Probability evolution (Rabi oscillations)
    axes[0].plot(t_values, probs_0, 'b-', linewidth=2, label='$|\\langle 0|\\psi(t)\\rangle|^2$')
    axes[0].plot(t_values, probs_1, 'r-', linewidth=2, label='$|\\langle 1|\\psi(t)\\rangle|^2$')
    axes[0].plot(t_values, np.array(probs_0) + np.array(probs_1), 'g--', label='Total (should be 1)')
    axes[0].set_xlabel('Time $t$')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Rabi Oscillations: $H = \\omega \\sigma_x$, $|\\psi(0)\\rangle = |0\\rangle$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bloch sphere trajectory
    ax = axes[1]

    # Compute Bloch vector components
    bloch_x = []
    bloch_y = []
    bloch_z = []

    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    for t in t_values:
        U = expm(-1j * H * t)
        psi_t = U @ psi_0

        # Density matrix
        rho = np.outer(psi_t, psi_t.conj())

        # Bloch components: ⟨σ_i⟩ = Tr(ρ σ_i)
        bloch_x.append(np.real(np.trace(rho @ sigma_x)))
        bloch_y.append(np.real(np.trace(rho @ sigma_y)))
        bloch_z.append(np.real(np.trace(rho @ sigma_z)))

    ax.plot(t_values, bloch_x, 'r-', label='$\\langle\\sigma_x\\rangle$')
    ax.plot(t_values, bloch_y, 'g-', label='$\\langle\\sigma_y\\rangle$')
    ax.plot(t_values, bloch_z, 'b-', label='$\\langle\\sigma_z\\rangle$')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Bloch Component')
    ax.set_title('Bloch Vector Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('unitary_evolution.png', dpi=150)
    plt.show()

    print(f"Unitarity preserved at all times: {all(unitarity_check)}")

# ============================================================
# Part 4: Eigenvector Orthogonality
# ============================================================

def eigenvector_orthogonality():
    """
    Verify eigenvector orthogonality for self-adjoint and normal operators.
    """
    print("\n" + "=" * 60)
    print("Part 4: Eigenvector Orthogonality")
    print("=" * 60)

    np.random.seed(123)
    n = 5

    # Self-adjoint matrix
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = (A + A.conj().T) / 2

    eigenvalues, eigenvectors = eigh(A)  # eigh for Hermitian

    # Compute inner products
    overlaps = eigenvectors.conj().T @ eigenvectors

    print("Self-adjoint operator:")
    print(f"  Eigenvalues (should be real): {eigenvalues}")
    print(f"  Eigenvector overlaps (should be identity):")
    print(np.round(overlaps, 4))

    # Non-normal matrix
    B = np.array([[1, 1], [0, 2]], dtype=complex)
    eig_vals_B, eig_vecs_B = eig(B)

    overlaps_B = eig_vecs_B.conj().T @ eig_vecs_B

    print(f"\nNon-normal operator:")
    print(f"  Eigenvalues: {eig_vals_B}")
    print(f"  Eigenvector overlaps (NOT orthogonal):")
    print(np.round(overlaps_B, 4))

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(np.abs(overlaps), cmap='Blues', vmin=0, vmax=1.5)
    axes[0].set_title('Self-Adjoint: $|\\langle v_i | v_j \\rangle| = \\delta_{ij}$')
    axes[0].set_xlabel('Eigenvector $j$')
    axes[0].set_ylabel('Eigenvector $i$')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(np.abs(overlaps_B), cmap='Reds', vmin=0, vmax=1.5)
    axes[1].set_title('Non-Normal: Eigenvectors NOT Orthogonal')
    axes[1].set_xlabel('Eigenvector $j$')
    axes[1].set_ylabel('Eigenvector $i$')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig('eigenvector_orthogonality.png', dpi=150)
    plt.show()

# ============================================================
# Part 5: Numerical Range
# ============================================================

def numerical_range_visualization():
    """
    Visualize the numerical range W(A) = {⟨Ax,x⟩ : ||x||=1}.
    """
    print("\n" + "=" * 60)
    print("Part 5: Numerical Range")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    operators = [
        (np.array([[1, 0], [0, 2]], dtype=complex), "Self-adjoint: diag(1,2)"),
        (np.array([[0, -1j], [1j, 0]], dtype=complex), "Self-adjoint: σ_y"),
        (np.array([[0, 1], [0, 0]], dtype=complex), "Nilpotent"),
        (np.array([[1, 1], [-1, 1]], dtype=complex) / np.sqrt(2), "Normal (rotation)")
    ]

    for ax, (A, title) in zip(axes.flat, operators):
        # Sample many unit vectors
        n_samples = 1000
        numerical_range = []

        for _ in range(n_samples):
            # Random unit vector in C^2
            x = np.random.randn(2) + 1j * np.random.randn(2)
            x = x / np.linalg.norm(x)

            val = np.vdot(x, A @ x)  # ⟨Ax, x⟩
            numerical_range.append(val)

        numerical_range = np.array(numerical_range)

        ax.scatter(numerical_range.real, numerical_range.imag, s=1, alpha=0.3)

        # Mark eigenvalues
        eigenvalues = np.linalg.eigvals(A)
        ax.scatter(eigenvalues.real, eigenvalues.imag, s=100, c='red',
                  marker='*', zorder=5, label='Eigenvalues')

        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        ax.set_title(f'W(A): {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Numerical Range W(A) = {⟨Ax,x⟩ : ||x||=1}', fontsize=14)
    plt.tight_layout()
    plt.savefig('numerical_range.png', dpi=150)
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 242: Self-Adjoint and Unitary Operators - Computational Lab")
    print("=" * 60)

    operator_classification_examples()
    spectral_properties()
    unitary_time_evolution()
    eigenvector_orthogonality()
    numerical_range_visualization()

    print("\n" + "=" * 60)
    print("Lab complete! Key takeaways:")
    print("  1. Self-adjoint operators have real eigenvalues")
    print("  2. Unitary operators have eigenvalues on the unit circle")
    print("  3. Normal operators have orthogonal eigenvectors")
    print("  4. Time evolution U(t) = exp(-iHt) is unitary when H is self-adjoint")
    print("  5. The numerical range W(A) is real ⟺ A is self-adjoint")
    print("=" * 60)
```

---

## 8. Summary

### Key Definitions

| Concept | Definition | Property |
|---------|------------|----------|
| **Self-adjoint** | $A = A^\dagger$ | $\sigma(A) \subseteq \mathbb{R}$ |
| **Unitary** | $U^\dagger U = UU^\dagger = I$ | $\sigma(U) \subseteq S^1$ |
| **Normal** | $AA^\dagger = A^\dagger A$ | Orthogonal eigenvectors |
| **Isometry** | $V^\dagger V = I$ | $\|Vx\| = \|x\|$ |

### Key Formulas

$$\boxed{\begin{aligned}
&\text{Self-adjoint:} && A = A^\dagger \Leftrightarrow \langle Ax, x\rangle \in \mathbb{R} \; \forall x \\
&\text{Unitary:} && U^\dagger U = I \Leftrightarrow \langle Ux, Uy\rangle = \langle x, y\rangle \\
&\text{Normal:} && AA^\dagger = A^\dagger A \Leftrightarrow \|Ax\| = \|A^\dagger x\| \\
&\text{Time evolution:} && U(t) = e^{-iHt/\hbar}, \quad H = H^\dagger
\end{aligned}}$$

### Key Theorems

| Theorem | Statement |
|---------|-----------|
| **Real Spectrum** | Self-adjoint $\Rightarrow$ $\sigma(A) \subseteq \mathbb{R}$ |
| **Orthogonal Eigenvectors** | Self-adjoint/normal $\Rightarrow$ distinct eigenvectors orthogonal |
| **Unitary Spectrum** | Unitary $\Rightarrow$ $\sigma(U) \subseteq \{|z|=1\}$ |
| **Preservation** | Unitary $\Rightarrow$ preserves inner products |

### Key Insights

1. **Self-adjoint = Observable**: Real eigenvalues = real measurement outcomes
2. **Unitary = Symmetry**: Preserves probabilities and reversibility
3. **Normal = Diagonalizable**: Admits complete orthonormal eigenbasis
4. **Time evolution is unitary**: $H$ self-adjoint $\Rightarrow$ $e^{-iHt/\hbar}$ unitary

---

## 9. Daily Checklist

- [ ] I can define self-adjoint, unitary, and normal operators
- [ ] I can prove self-adjoint operators have real spectrum
- [ ] I can prove unitary operators preserve inner products
- [ ] I can classify operators and verify their properties
- [ ] I understand why observables must be self-adjoint
- [ ] I understand why symmetries must be unitary
- [ ] I can verify time evolution is unitary
- [ ] I completed the computational lab exercises

---

## 10. Preview: Day 243

Tomorrow we study **projection operators**, which satisfy $P^2 = P = P^\dagger$. Projections are fundamental to:
- Orthogonal decomposition of Hilbert spaces
- The measurement process in quantum mechanics
- Spectral decomposition of self-adjoint operators

We'll prove the **projection theorem**: every closed subspace has a unique orthogonal projection, and explore how this relates to quantum measurement and state collapse.

---

*"The self-adjoint operators represent the observables, the unitary operators represent the symmetries, and the interplay between them captures the essence of quantum mechanics."* — Rudolf Haag
