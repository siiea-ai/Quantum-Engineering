# Day 295: Introduction to Lie Groups — Continuous Symmetries

## Overview

**Month 11, Week 43, Day 1 — Monday**

Today we begin our study of **Lie groups**, named after the Norwegian mathematician Sophus Lie. Unlike the finite groups we've studied, Lie groups describe continuous symmetries—transformations that depend smoothly on continuous parameters. Rotations, translations, and Lorentz transformations are all Lie groups. Understanding Lie groups is essential for quantum mechanics, where continuous symmetries lead to conservation laws and classify particle types.

## Prerequisites

From Weeks 41-42:
- Abstract group theory
- Representation theory
- Matrix groups

From calculus/analysis:
- Derivatives and Taylor series
- Matrix exponentials

## Learning Objectives

By the end of today, you will be able to:

1. Define Lie groups and understand their manifold structure
2. Identify common matrix Lie groups
3. Understand the dimension of a Lie group
4. Connect Lie groups to physics symmetries
5. Work with one-parameter subgroups
6. Compute the dimension and basic properties of standard Lie groups

---

## 1. What is a Lie Group?

### Formal Definition

**Definition:** A **Lie group** is a group $G$ that is also a smooth manifold, such that:
1. Multiplication $m: G \times G \to G$, $(g, h) \mapsto gh$ is smooth
2. Inversion $i: G \to G$, $g \mapsto g^{-1}$ is smooth

### Matrix Lie Groups

For our purposes, we focus on **matrix Lie groups**: closed subgroups of $GL_n(\mathbb{C})$ or $GL_n(\mathbb{R})$.

**Theorem:** Every closed subgroup of $GL_n(\mathbb{C})$ is a Lie group.

This gives us many important examples without needing full manifold theory!

### Dimension

The **dimension** of a Lie group is its dimension as a manifold—the number of independent real parameters needed to specify an element.

---

## 2. The Classical Matrix Lie Groups

### The General Linear Group $GL_n$

$$GL_n(\mathbb{R}) = \{A \in M_n(\mathbb{R}) : \det(A) \neq 0\}$$

**Dimension:** $n^2$ (all matrix entries are free, minus the zero-measure set $\det = 0$)

### The Special Linear Group $SL_n$

$$SL_n(\mathbb{R}) = \{A \in GL_n(\mathbb{R}) : \det(A) = 1\}$$

**Dimension:** $n^2 - 1$ (one constraint: $\det = 1$)

### The Orthogonal Group $O(n)$

$$O(n) = \{A \in GL_n(\mathbb{R}) : A^T A = I\}$$

**Dimension:** $\frac{n(n-1)}{2}$

*Derivation:* $A^T A = I$ gives $\frac{n(n+1)}{2}$ equations (symmetric matrix), so $\dim = n^2 - \frac{n(n+1)}{2} = \frac{n(n-1)}{2}$.

### The Special Orthogonal Group $SO(n)$

$$SO(n) = \{A \in O(n) : \det(A) = 1\}$$

**Dimension:** $\frac{n(n-1)}{2}$ (same as $O(n)$, but connected component)

For $n = 3$: $\dim(SO(3)) = 3$ — three rotation angles!

### The Unitary Group $U(n)$

$$U(n) = \{A \in GL_n(\mathbb{C}) : A^\dagger A = I\}$$

**Dimension:** $n^2$ (as a real manifold)

### The Special Unitary Group $SU(n)$

$$SU(n) = \{A \in U(n) : \det(A) = 1\}$$

**Dimension:** $n^2 - 1$

For $n = 2$: $\dim(SU(2)) = 3$ — same as SO(3)!

### Summary Table

| Group | Defining Condition | Real Dimension |
|-------|-------------------|----------------|
| $GL_n(\mathbb{R})$ | $\det \neq 0$ | $n^2$ |
| $SL_n(\mathbb{R})$ | $\det = 1$ | $n^2 - 1$ |
| $O(n)$ | $A^T A = I$ | $n(n-1)/2$ |
| $SO(n)$ | $A^T A = I$, $\det = 1$ | $n(n-1)/2$ |
| $U(n)$ | $A^\dagger A = I$ | $n^2$ |
| $SU(n)$ | $A^\dagger A = I$, $\det = 1$ | $n^2 - 1$ |

---

## 3. One-Parameter Subgroups

### Definition

A **one-parameter subgroup** of Lie group $G$ is a smooth homomorphism:
$$\gamma: \mathbb{R} \to G$$

This satisfies: $\gamma(s + t) = \gamma(s) \gamma(t)$ and $\gamma(0) = e$.

### Matrix Exponential

For matrix Lie groups, every one-parameter subgroup has the form:
$$\gamma(t) = e^{tX} = \sum_{n=0}^\infty \frac{(tX)^n}{n!}$$

for some matrix $X$.

### Examples

**Rotations about the z-axis:**
$$R_z(t) = e^{t J_z} = \begin{pmatrix} \cos t & -\sin t & 0 \\ \sin t & \cos t & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

where $J_z = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$.

**Boosts in special relativity:**
$$\Lambda(v) = e^{\phi K}$$

where $K$ is the boost generator and $\phi = \text{arctanh}(v/c)$.

---

## 4. Connectedness and Components

### Connected Components

- $O(n)$ has two connected components: $\det = +1$ (SO(n)) and $\det = -1$
- $GL_n(\mathbb{R})$ has two components: $\det > 0$ and $\det < 0$
- $SO(n)$, $SU(n)$, $U(n)$ are connected

### Simply Connected

A group is **simply connected** if every loop can be contracted to a point.

- $SU(n)$ is simply connected
- $SO(3)$ is NOT simply connected (loops exist)
- $SU(2)$ is simply connected and is the **universal cover** of $SO(3)$

---

## 5. Quantum Mechanics Connection

### Symmetries in QM

In quantum mechanics, continuous symmetries are implemented by Lie groups acting on Hilbert space:

| Physical Symmetry | Lie Group | Dimension |
|------------------|-----------|-----------|
| Rotations | SO(3) or SU(2) | 3 |
| Translations | $\mathbb{R}^3$ | 3 |
| Lorentz transformations | SO(3,1) | 6 |
| Phase transformations | U(1) | 1 |
| Gauge (electroweak) | SU(2) × U(1) | 4 |
| Gauge (strong) | SU(3) | 8 |

### Conservation Laws (Noether's Theorem)

Each continuous symmetry corresponds to a conserved quantity:

| Symmetry | Group | Conserved Quantity |
|----------|-------|-------------------|
| Time translation | $\mathbb{R}$ | Energy |
| Space translation | $\mathbb{R}^3$ | Momentum |
| Rotation | SO(3) | Angular momentum |
| Phase | U(1) | Charge |

### Generators and Observables

The **generators** of a Lie group become **observables** in quantum mechanics:
- Rotation generators → Angular momentum $\hat{L}_x, \hat{L}_y, \hat{L}_z$
- Translation generator → Momentum $\hat{p}$
- Time translation generator → Hamiltonian $\hat{H}$

---

## 6. Worked Examples

### Example 1: Verify SO(2) is a Lie Group

$SO(2) = \{R(\theta) : \theta \in [0, 2\pi)\}$ where $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.

**Group axioms:**
- Closure: $R(\theta_1) R(\theta_2) = R(\theta_1 + \theta_2)$ ✓
- Identity: $R(0) = I$ ✓
- Inverse: $R(\theta)^{-1} = R(-\theta)$ ✓
- Associativity: inherited from matrix multiplication ✓

**Smoothness:** $R(\theta)$ depends smoothly on $\theta$, and multiplication $(θ_1, θ_2) \mapsto θ_1 + θ_2$ is smooth. ✓

**Dimension:** 1 (parameterized by single angle $\theta$).

### Example 2: Dimension of SU(2)

A general $2 \times 2$ complex matrix has 8 real parameters.

$U^\dagger U = I$ imposes 4 real constraints (Hermitian matrix has 4 independent entries).

$\det(U) = 1$ imposes 1 more constraint (complex with $|\det| = 1$ already, so just phase = 0).

Actually, $U^\dagger U = I$ already implies $|\det| = 1$. The constraint $\det = 1$ adds one more.

$\dim(SU(2)) = 8 - 4 - 1 = 3$.

**Explicit parameterization:** Any $U \in SU(2)$ can be written as:
$$U = \begin{pmatrix} \alpha & -\bar{\beta} \\ \beta & \bar{\alpha} \end{pmatrix}$$
where $|\alpha|^2 + |\beta|^2 = 1$ (3-sphere in $\mathbb{C}^2 \cong \mathbb{R}^4$).

### Example 3: The Lorentz Group

The Lorentz group $O(3,1)$ preserves the Minkowski metric:
$$\Lambda^T \eta \Lambda = \eta, \quad \eta = \text{diag}(1, -1, -1, -1)$$

**Dimension:** The constraint is a symmetric 4×4 matrix equation: 10 constraints on 16 parameters.
$\dim(O(3,1)) = 16 - 10 = 6$.

Components: 4 components based on $\det = \pm 1$ and $\Lambda^0{}_0 \gtrless 0$.

The proper orthochronous Lorentz group $SO^+(3,1)$ is connected and has dimension 6.

---

## 7. Computational Lab

```python
"""
Day 295: Introduction to Lie Groups
Exploring matrix Lie groups computationally
"""

import numpy as np
from scipy.linalg import expm, logm
from typing import Tuple

def is_orthogonal(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is orthogonal."""
    return np.allclose(A.T @ A, np.eye(A.shape[0]), atol=tol)

def is_unitary(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is unitary."""
    return np.allclose(A.conj().T @ A, np.eye(A.shape[0]), atol=tol)

def is_special(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if det(A) = 1."""
    return np.isclose(np.linalg.det(A), 1.0, atol=tol)

def random_SO3() -> np.ndarray:
    """Generate random element of SO(3)."""
    # Use QR decomposition of random matrix
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Ensure det = +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def random_SU2() -> np.ndarray:
    """Generate random element of SU(2)."""
    # Parameterize as (α, β) with |α|² + |β|² = 1
    z = np.random.randn(4)
    z = z / np.linalg.norm(z)
    alpha = z[0] + 1j * z[1]
    beta = z[2] + 1j * z[3]
    return np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])

def rotation_matrix_3d(axis: str, angle: float) -> np.ndarray:
    """Rotation matrix about given axis."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def exp_so3(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) to SO(3) using Rodrigues' formula.

    omega: 3-vector (angular velocity)
    Returns: rotation matrix
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)

    # Skew-symmetric matrix
    omega_hat = np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

    # Rodrigues' formula
    return np.eye(3) + (np.sin(theta)/theta) * omega_hat + \
           ((1 - np.cos(theta))/theta**2) * (omega_hat @ omega_hat)

def log_SO3(R: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SO(3) to so(3).
    Returns: angular velocity vector
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if theta < 1e-10:
        return np.zeros(3)

    omega_hat = (theta / (2 * np.sin(theta))) * (R - R.T)
    return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])

# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("INTRODUCTION TO LIE GROUPS")
    print("=" * 60)

    # Example 1: Verify group properties
    print("\n1. VERIFYING SO(3) GROUP PROPERTIES")
    print("-" * 40)

    R1 = random_SO3()
    R2 = random_SO3()

    print(f"R1 orthogonal: {is_orthogonal(R1)}")
    print(f"R2 orthogonal: {is_orthogonal(R2)}")
    print(f"det(R1) = {np.linalg.det(R1):.6f}")
    print(f"det(R2) = {np.linalg.det(R2):.6f}")

    # Closure
    R12 = R1 @ R2
    print(f"\nClosure: R1 R2 in SO(3): {is_orthogonal(R12) and is_special(R12)}")

    # Inverse
    R1_inv = R1.T
    print(f"Inverse: R1^T R1 = I: {np.allclose(R1_inv @ R1, np.eye(3))}")

    # Example 2: SU(2) elements
    print("\n2. SU(2) ELEMENTS")
    print("-" * 40)

    U1 = random_SU2()
    U2 = random_SU2()

    print(f"U1 unitary: {is_unitary(U1)}")
    print(f"det(U1) = {np.linalg.det(U1):.6f}")
    print(f"U1 @ U2 unitary: {is_unitary(U1 @ U2)}")

    # Example 3: One-parameter subgroups
    print("\n3. ONE-PARAMETER SUBGROUPS")
    print("-" * 40)

    # Rotation about z-axis
    print("Rotations about z-axis: R_z(t) = exp(t J_z)")

    J_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)

    for t in [0, np.pi/4, np.pi/2, np.pi]:
        R_t = expm(t * J_z)
        print(f"t = {t:.4f}: det = {np.linalg.det(R_t):.4f}, orthogonal = {is_orthogonal(R_t)}")

    # Verify group homomorphism property
    t1, t2 = np.pi/6, np.pi/4
    R_t1 = expm(t1 * J_z)
    R_t2 = expm(t2 * J_z)
    R_t1_t2 = expm((t1 + t2) * J_z)

    print(f"\nHomomorphism: exp(t₁J) exp(t₂J) = exp((t₁+t₂)J)")
    print(f"Match: {np.allclose(R_t1 @ R_t2, R_t1_t2)}")

    # Example 4: Exponential map
    print("\n4. EXPONENTIAL MAP so(3) → SO(3)")
    print("-" * 40)

    omega = np.array([0.5, 0.3, 0.8])
    R = exp_so3(omega)

    print(f"ω = {omega}")
    print(f"||ω|| = {np.linalg.norm(omega):.4f} rad = {np.degrees(np.linalg.norm(omega)):.2f}°")
    print(f"R = exp(ω^) is orthogonal: {is_orthogonal(R)}")
    print(f"det(R) = {np.linalg.det(R):.6f}")

    # Log map (inverse)
    omega_recovered = log_SO3(R)
    print(f"\nLog map: ω recovered = {omega_recovered}")
    print(f"Match: {np.allclose(omega, omega_recovered)}")

    # Example 5: Dimension count
    print("\n5. LIE GROUP DIMENSIONS")
    print("-" * 40)

    groups = [
        ("GL(n,R)", lambda n: n**2),
        ("SL(n,R)", lambda n: n**2 - 1),
        ("O(n)", lambda n: n*(n-1)//2),
        ("SO(n)", lambda n: n*(n-1)//2),
        ("U(n)", lambda n: n**2),
        ("SU(n)", lambda n: n**2 - 1),
    ]

    print(f"{'Group':<12} {'n=2':>6} {'n=3':>6} {'n=4':>6}")
    print("-" * 30)
    for name, dim_func in groups:
        dims = [dim_func(n) for n in [2, 3, 4]]
        print(f"{name:<12} {dims[0]:>6} {dims[1]:>6} {dims[2]:>6}")

    # Example 6: Composition of rotations
    print("\n6. COMPOSITION OF ROTATIONS")
    print("-" * 40)

    # Rotate by π/2 about x, then π/2 about y
    Rx = rotation_matrix_3d('x', np.pi/2)
    Ry = rotation_matrix_3d('y', np.pi/2)

    print("R_y(π/2) @ R_x(π/2):")
    R_composite = Ry @ Rx
    print(R_composite.round(4))

    # Find axis and angle
    omega = log_SO3(R_composite)
    angle = np.linalg.norm(omega)
    axis = omega / angle if angle > 1e-10 else np.array([0, 0, 1])

    print(f"\nEquivalent single rotation:")
    print(f"Angle: {np.degrees(angle):.2f}°")
    print(f"Axis: {axis.round(4)}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Lie groups = groups + smooth manifold structure
    2. Matrix Lie groups: closed subgroups of GL_n
    3. Dimension = number of independent parameters
    4. One-parameter subgroups: γ(t) = exp(tX)
    5. SO(3) and SU(2) both have dimension 3
    6. Continuous symmetries → conserved quantities (Noether)
    """)
```

---

## 8. Practice Problems

### Problem Set A: Basic Properties

**A1.** Verify that $SL_2(\mathbb{R})$ is a Lie group by showing it's closed under multiplication and inversion.

**A2.** Compute the dimension of the symplectic group $Sp(2n)$ defined by $A^T J A = J$ where $J = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}$.

**A3.** Show that $U(1) \cong SO(2)$ as Lie groups.

### Problem Set B: Exponentials

**B1.** Compute $e^{tA}$ where $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$.

**B2.** Show that for $X \in \mathfrak{su}(2)$ (traceless skew-Hermitian), $e^X \in SU(2)$.

**B3.** Find a matrix $X$ such that $e^X = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.

### Problem Set C: Physics Applications

**C1.** The time evolution operator is $U(t) = e^{-iHt/\hbar}$. Verify this is unitary when $H$ is Hermitian.

**C2.** Rotations in 3D can be parameterized by Euler angles $(\alpha, \beta, \gamma)$. Write the rotation matrix $R = R_z(\alpha) R_y(\beta) R_z(\gamma)$.

**C3.** **(Relativity)** A Lorentz boost in the x-direction is $\Lambda = e^{\phi K_x}$ where $K_x$ is the boost generator. Find $K_x$.

---

## 9. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| Lie group | Group + smooth manifold |
| Matrix Lie group | Closed subgroup of $GL_n$ |
| One-parameter subgroup | $\gamma(t) = e^{tX}$ |
| Dimension | Number of free parameters |

### Important Lie Groups

| Group | Dimension | Physical Role |
|-------|-----------|---------------|
| $SO(3)$ | 3 | Rotations in 3D |
| $SU(2)$ | 3 | Spin rotations |
| $U(1)$ | 1 | Phase / EM gauge |
| $SU(3)$ | 8 | Strong force (QCD) |
| $SO(3,1)$ | 6 | Lorentz transformations |

---

## 10. Preview: Day 296

Tomorrow we study the **rotation group SO(3)** in depth:
- Various parameterizations (axis-angle, Euler angles)
- Geometry of SO(3) as a manifold
- Covering by SU(2)
- Connection to angular momentum

---

*"The theory of Lie groups is the gateway to understanding continuous symmetries in physics." — Sophus Lie*
