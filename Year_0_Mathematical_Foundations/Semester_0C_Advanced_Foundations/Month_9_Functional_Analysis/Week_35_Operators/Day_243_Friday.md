# Day 243: Projections and Orthogonal Decomposition

## Schedule Overview (8 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Theory: Projections, orthogonal decomposition theorem |
| Afternoon | 3 hours | Problems: Projection computations, direct sum decompositions |
| Evening | 2 hours | Computational lab: Quantum measurement and state collapse |

## Learning Objectives

By the end of today, you will be able to:

1. **Define** projection operators and orthogonal projections
2. **Prove** the projection theorem: closed subspaces have unique orthogonal projections
3. **Characterize** orthogonal projections as $P^2 = P = P^\dagger$
4. **Decompose** Hilbert spaces as direct sums via projections
5. **Compute** projections onto various subspaces
6. **Connect** projections to quantum measurement and state collapse

---

## 1. Core Content: Projection Operators

### 1.1 Algebraic Definition

**Definition**: A bounded linear operator $P: \mathcal{H} \to \mathcal{H}$ is a **projection** (or **idempotent**) if:

$$\boxed{P^2 = P}$$

**Definition**: A projection $P$ is an **orthogonal projection** if additionally:

$$\boxed{P = P^\dagger}$$

So orthogonal projections satisfy $P^2 = P = P^\dagger$.

### 1.2 Geometric Interpretation

**Theorem**: Let $P$ be an orthogonal projection. Then:
1. $\mathcal{H} = \text{ran}(P) \oplus \ker(P)$ (orthogonal direct sum)
2. $\text{ran}(P) = \ker(I - P)$
3. $\ker(P) = \text{ran}(I - P) = (\text{ran}(P))^\perp$
4. $I - P$ is also an orthogonal projection (onto $\ker(P)$)

**Proof**:

**(1)** For any $x \in \mathcal{H}$: $x = Px + (I - P)x$.

We have $Px \in \text{ran}(P)$ and $(I-P)x \in \text{ran}(I-P)$.

Claim: $\text{ran}(P) \perp \text{ran}(I-P)$.

For $y = Pz$ and $w = (I-P)u$:
$$\langle y, w \rangle = \langle Pz, (I-P)u \rangle = \langle z, P^\dagger(I-P)u \rangle = \langle z, P(I-P)u \rangle = \langle z, (P - P^2)u \rangle = 0$$

**(2)** $x \in \text{ran}(P) \Leftrightarrow x = Py$ for some $y \Leftrightarrow Px = P^2y = Py = x \Leftrightarrow (I-P)x = 0$.

**(3)** From (1), $\text{ran}(I-P) = \ker(P)$. Since the decomposition is orthogonal, $\ker(P) = (\text{ran}(P))^\perp$.

**(4)** $(I-P)^2 = I - 2P + P^2 = I - 2P + P = I - P$ and $(I-P)^\dagger = I - P^\dagger = I - P$. $\square$

### 1.3 The Projection Theorem

**Theorem (Projection Theorem)**: Let $M$ be a **closed** subspace of a Hilbert space $\mathcal{H}$. Then:
1. Every $x \in \mathcal{H}$ has a unique decomposition $x = m + n$ where $m \in M$ and $n \in M^\perp$
2. The map $P_M: x \mapsto m$ is the unique orthogonal projection onto $M$
3. $\|x - P_M x\| = \text{dist}(x, M) = \inf_{y \in M}\|x - y\|$

**Proof**:

**Existence**: Let $d = \text{dist}(x, M) = \inf_{y \in M}\|x - y\|$.

Choose a sequence $(y_n)$ in $M$ with $\|x - y_n\| \to d$.

By the **parallelogram law**:
$$\|y_n - y_m\|^2 = 2\|y_n - x\|^2 + 2\|y_m - x\|^2 - \|y_n + y_m - 2x\|^2$$
$$= 2\|y_n - x\|^2 + 2\|y_m - x\|^2 - 4\left\|\frac{y_n + y_m}{2} - x\right\|^2$$

Since $\frac{y_n + y_m}{2} \in M$, $\left\|\frac{y_n + y_m}{2} - x\right\| \geq d$.

As $n, m \to \infty$: $\|y_n - y_m\|^2 \leq 2d^2 + 2d^2 - 4d^2 = 0$.

So $(y_n)$ is Cauchy, hence converges to some $m \in M$ (since $M$ is closed).

**Orthogonality**: We show $n = x - m \in M^\perp$.

For any $y \in M$ and $t \in \mathbb{R}$, $m + ty \in M$, so:
$$\|n\|^2 = d^2 \leq \|x - (m + ty)\|^2 = \|n - ty\|^2 = \|n\|^2 - 2t\text{Re}\langle n, y\rangle + t^2\|y\|^2$$

This gives $0 \leq -2t\text{Re}\langle n, y\rangle + t^2\|y\|^2$.

For small $t > 0$: $\text{Re}\langle n, y\rangle \leq 0$. For small $t < 0$: $\text{Re}\langle n, y\rangle \geq 0$.

So $\text{Re}\langle n, y\rangle = 0$. Similarly (using $ty$ replaced by $ity$), $\text{Im}\langle n, y\rangle = 0$.

Thus $\langle n, y\rangle = 0$ for all $y \in M$, i.e., $n \in M^\perp$.

**Uniqueness**: If $x = m_1 + n_1 = m_2 + n_2$ with $m_i \in M$, $n_i \in M^\perp$:

$m_1 - m_2 = n_2 - n_1 \in M \cap M^\perp = \{0\}$. $\square$

---

## 2. Properties of Projections

### 2.1 Characterization Theorem

**Theorem**: For a bounded operator $P$, the following are equivalent:
1. $P$ is an orthogonal projection
2. $P^2 = P$ and $P = P^\dagger$
3. $P^2 = P$ and $\|P\| \leq 1$
4. $P^2 = P$ and $\langle Px, x \rangle \geq 0$ for all $x$

**Proof sketch**: (1) $\Leftrightarrow$ (2) by definition.

(2) $\Rightarrow$ (3): For $\|x\| = 1$, $\|Px\|^2 = \langle Px, Px\rangle = \langle P^2x, x\rangle = \langle Px, x\rangle \leq \|Px\|\|x\|$, so $\|Px\| \leq 1$.

(3) $\Rightarrow$ (4): If $P^2 = P$ and $\|P\| \leq 1$, then $\text{ran}(P) \cap \ker(P) = \{0\}$ and the decomposition is direct. Using $\langle Px, x\rangle = \langle Px, Px + (I-P)x\rangle = \|Px\|^2 \geq 0$.

(4) $\Rightarrow$ (2): The non-negative condition on quadratic forms implies self-adjointness (via polarization). $\square$

### 2.2 Lattice of Projections

**Definition**: For projections $P$ and $Q$, define:
- $P \leq Q$ if $\text{ran}(P) \subseteq \text{ran}(Q)$

**Theorem**: $P \leq Q$ if and only if $PQ = QP = P$.

**Proof**: ($\Rightarrow$) If $\text{ran}(P) \subseteq \text{ran}(Q)$, then for any $x$, $Px \in \text{ran}(Q)$, so $Q(Px) = Px$, giving $QP = P$.

Since $P, Q$ are self-adjoint: $PQ = (QP)^\dagger = P^\dagger = P$.

($\Leftarrow$) If $PQ = P$, then $Px = PQx = P(Qx)$, so $Px \in \text{ran}(P) \subseteq \text{ran}(Q)$. $\square$

### 2.3 Operations on Projections

**Theorem**: Let $P$ and $Q$ be orthogonal projections.

1. $PQ$ is a projection if and only if $PQ = QP$ (they commute)
2. If $PQ = QP$, then $PQ$ projects onto $\text{ran}(P) \cap \text{ran}(Q)$
3. $P + Q$ is a projection if and only if $PQ = 0$ (orthogonal ranges)
4. If $PQ = 0$, then $P + Q$ projects onto $\text{ran}(P) \oplus \text{ran}(Q)$

---

## 3. Explicit Projection Formulas

### 3.1 Projection onto a Single Vector

For a unit vector $u \in \mathcal{H}$, the projection onto $\text{span}\{u\}$ is:

$$\boxed{P_u x = \langle u, x \rangle u}$$

In Dirac notation: $P_u = |u\rangle\langle u|$.

**Verification**:
- $P_u^2 x = \langle u, \langle u, x\rangle u\rangle u = \langle u, x\rangle \langle u, u\rangle u = \langle u, x\rangle u = P_u x$ ✓
- $(P_u)^\dagger = P_u$ since $\langle P_u x, y\rangle = \langle u, x\rangle\langle u, y\rangle = \langle x, \langle u, y\rangle u\rangle = \langle x, P_u y\rangle$ ✓

### 3.2 Projection onto a Finite-Dimensional Subspace

For an orthonormal set $\{e_1, \ldots, e_n\}$, the projection onto $M = \text{span}\{e_1, \ldots, e_n\}$ is:

$$\boxed{P_M x = \sum_{k=1}^n \langle e_k, x \rangle e_k}$$

In Dirac notation: $P_M = \sum_{k=1}^n |e_k\rangle\langle e_k|$.

### 3.3 Projection via Matrix Formula

For a matrix $A$ with full column rank, the projection onto $\text{col}(A)$ is:

$$\boxed{P = A(A^\dagger A)^{-1}A^\dagger}$$

This is the formula from linear algebra for the orthogonal projection.

---

## 4. Quantum Mechanics Connection

### 4.1 Measurement as Projection

**Postulate (Projection Postulate)**: When a quantum system in state $|\psi\rangle$ is measured for an observable $A$ with eigenvalue $\lambda$ and eigenspace $E_\lambda$, the state after measurement is:

$$|\psi'\rangle = \frac{P_\lambda |\psi\rangle}{\|P_\lambda |\psi\rangle\|}$$

where $P_\lambda$ is the projection onto $E_\lambda$.

**Probability**: The probability of obtaining $\lambda$ is:

$$P(\lambda) = \|P_\lambda |\psi\rangle\|^2 = \langle\psi|P_\lambda|\psi\rangle$$

### 4.2 The Resolution of Identity

For an observable $A$ with complete orthonormal eigenbasis $\{|n\rangle\}$:

$$\boxed{\sum_n |n\rangle\langle n| = I}$$

This is the **completeness relation** or **resolution of identity**.

It means: $|\psi\rangle = \sum_n |n\rangle\langle n|\psi\rangle = \sum_n c_n |n\rangle$ where $c_n = \langle n|\psi\rangle$.

### 4.3 Spectral Decomposition Preview

For a self-adjoint operator $A$ with eigenvalues $\{\lambda_n\}$ and eigenprojections $\{P_n\}$:

$$A = \sum_n \lambda_n P_n$$

This is the **spectral theorem** for operators with discrete spectrum (full treatment in Week 36).

### 4.4 Projection-Valued Measures

More generally, for continuous spectrum, we have a **projection-valued measure** $E$:

$$A = \int_{\sigma(A)} \lambda \, dE(\lambda)$$

Each $E(\Delta)$ for a Borel set $\Delta$ is an orthogonal projection.

### 4.5 Density Matrices and Projections

**Pure states** correspond to rank-1 projections:
$$\rho = |\psi\rangle\langle\psi|$$

**Mixed states** are convex combinations:
$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

Properties: $\rho^\dagger = \rho$, $\text{Tr}(\rho) = 1$, $\rho \geq 0$.

---

## 5. Worked Examples

### Example 1: Projection onto a Line in $\mathbb{C}^2$

**Problem**: Find the projection onto $M = \text{span}\{(1, i)^T\}$ in $\mathbb{C}^2$.

**Solution**:

Normalize: $u = \frac{1}{\sqrt{2}}(1, i)^T$.

$$P_M = |u\rangle\langle u| = \frac{1}{2}\begin{pmatrix} 1 \\ i \end{pmatrix}\begin{pmatrix} 1 & -i \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}$$

**Verification**:
$$P_M^2 = \frac{1}{4}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix} = \frac{1}{4}\begin{pmatrix} 1-i^2 & -i-i \\ i+i & -i^2+1 \end{pmatrix} = \frac{1}{4}\begin{pmatrix} 2 & -2i \\ 2i & 2 \end{pmatrix} = P_M$$ ✓

$$P_M^\dagger = \frac{1}{2}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}^\dagger = \frac{1}{2}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix} = P_M$$ ✓

---

### Example 2: Projection in $L^2[0,1]$

**Problem**: Find the projection of $f(x) = x$ onto the subspace $M = \text{span}\{1, \cos(\pi x)\}$ in $L^2[0,1]$.

**Solution**:

First, orthonormalize $\{1, \cos(\pi x)\}$.

$e_1 = 1$ with $\|e_1\|^2 = \int_0^1 1 \, dx = 1$, so $e_1 = 1$.

For $e_2$: $\langle e_1, \cos(\pi x)\rangle = \int_0^1 \cos(\pi x) \, dx = \frac{\sin(\pi x)}{\pi}\Big|_0^1 = 0$.

So $\{1, \cos(\pi x)\}$ is already orthogonal.

$\|\cos(\pi x)\|^2 = \int_0^1 \cos^2(\pi x) \, dx = \frac{1}{2}$, so $e_2 = \sqrt{2}\cos(\pi x)$.

Projection:
$$P_M f = \langle e_1, f\rangle e_1 + \langle e_2, f\rangle e_2$$

$\langle e_1, x\rangle = \int_0^1 x \, dx = \frac{1}{2}$

$\langle e_2, x\rangle = \sqrt{2}\int_0^1 x\cos(\pi x) \, dx$

Using integration by parts:
$$\int_0^1 x\cos(\pi x) \, dx = \frac{x\sin(\pi x)}{\pi}\Big|_0^1 - \frac{1}{\pi}\int_0^1 \sin(\pi x) \, dx = 0 + \frac{\cos(\pi x)}{\pi^2}\Big|_0^1 = \frac{-1-1}{\pi^2} = -\frac{2}{\pi^2}$$

So $\langle e_2, x\rangle = -\frac{2\sqrt{2}}{\pi^2}$.

$$\boxed{P_M x = \frac{1}{2} - \frac{4}{\pi^2}\cos(\pi x)}$$

---

### Example 3: Quantum Measurement

**Problem**: A spin-$\frac{1}{2}$ particle is in state $|\psi\rangle = \frac{1}{\sqrt{3}}|+\rangle + \sqrt{\frac{2}{3}}|-\rangle$. Find the state after measuring $S_z$ and obtaining $+\hbar/2$.

**Solution**:

The projection onto the $|+\rangle$ eigenspace is $P_+ = |+\rangle\langle +|$.

$$P_+ |\psi\rangle = |+\rangle\langle +|\psi\rangle = \frac{1}{\sqrt{3}}|+\rangle$$

Normalize:
$$|\psi'\rangle = \frac{P_+ |\psi\rangle}{\|P_+ |\psi\rangle\|} = \frac{(1/\sqrt{3})|+\rangle}{1/\sqrt{3}} = |+\rangle$$

**Probability**: $P(+\hbar/2) = \|P_+ |\psi\rangle\|^2 = \frac{1}{3}$.

After measurement, the state collapses to $|\psi'\rangle = |+\rangle$. $\square$

---

## 6. Practice Problems

### Level 1: Direct Application

1. Verify that $P = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & 1/2 \end{pmatrix}$ is an orthogonal projection. Find $\text{ran}(P)$ and $\ker(P)$.

2. Find the projection of $v = (1, 2, 3)$ onto the plane $x + y + z = 0$ in $\mathbb{R}^3$.

3. In $\ell^2$, let $P_n$ be the projection onto $\text{span}\{e_1, \ldots, e_n\}$. Write $P_n$ explicitly and verify $P_n \leq P_{n+1}$.

### Level 2: Intermediate

4. **Prove**: If $P$ and $Q$ are orthogonal projections with $PQ = QP$, then $PQ$ is an orthogonal projection onto $\text{ran}(P) \cap \text{ran}(Q)$.

5. **Prove**: For orthogonal projections $P$ and $Q$, $P + Q$ is a projection if and only if $PQ = 0$.

6. **Quantum Connection**: A qubit is in state $|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$. Compute the probability of measuring $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$ and the post-measurement state.

### Level 3: Challenging

7. **Prove**: If $P$ is an orthogonal projection, then $\sigma(P) \subseteq \{0, 1\}$.

8. Let $T: L^2[0,1] \to L^2[0,1]$ be defined by $(Tf)(x) = \int_0^1 f(t) \, dt$ (constant function). Prove $T$ is a projection and find $T^\dagger$. Is $T$ an orthogonal projection?

9. **Research problem**: Prove that the set of orthogonal projections is closed under strong limits but not under norm limits. Give an example.

---

## 7. Computational Lab: Projections in Quantum Mechanics

```python
"""
Day 243 Computational Lab: Projections and Orthogonal Decomposition
Quantum measurement, state collapse, and projection operators
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Part 1: Basic Projection Properties
# ============================================================

def projection_properties():
    """
    Demonstrate and verify properties of orthogonal projections.
    """
    print("=" * 60)
    print("Part 1: Projection Properties")
    print("=" * 60)

    # Create a projection onto span{(1,1,0), (0,1,1)} in C^3
    v1 = np.array([1, 1, 0], dtype=complex)
    v2 = np.array([0, 1, 1], dtype=complex)

    # Orthonormalize using Gram-Schmidt
    e1 = v1 / np.linalg.norm(v1)
    v2_orth = v2 - np.vdot(e1, v2) * e1
    e2 = v2_orth / np.linalg.norm(v2_orth)

    # Build projection matrix P = |e1><e1| + |e2><e2|
    P = np.outer(e1, e1.conj()) + np.outer(e2, e2.conj())

    print("Projection onto 2D subspace in C^3:")
    print(f"P =\n{np.round(P, 4)}")

    # Verify properties
    print(f"\nP² = P: {np.allclose(P @ P, P)}")
    print(f"P† = P: {np.allclose(P, P.conj().T)}")
    print(f"||P|| = {np.linalg.norm(P, ord=2):.4f} (should be ≤ 1)")

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(P)
    print(f"Eigenvalues: {np.round(eigenvalues, 4)} (should be 0 and/or 1)")

    # Complementary projection
    Q = np.eye(3) - P
    print(f"\nQ = I - P is also a projection: {np.allclose(Q @ Q, Q)}")
    print(f"PQ = 0 (orthogonal): {np.allclose(P @ Q, 0)}")

    return P

# ============================================================
# Part 2: Projection onto Subspaces
# ============================================================

def visualize_projection_2d():
    """
    Visualize projection onto a line in R^2.
    """
    print("\n" + "=" * 60)
    print("Part 2: Visualizing Projections")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Projection onto line y = x
    theta = np.pi / 4
    u = np.array([np.cos(theta), np.sin(theta)])
    P = np.outer(u, u)

    # Sample points
    np.random.seed(42)
    points = np.random.randn(20, 2)

    # Project points
    projected = (P @ points.T).T

    ax = axes[0]
    t = np.linspace(-3, 3, 100)
    ax.plot(t, t, 'b-', linewidth=2, label='Subspace $M$')

    for i in range(len(points)):
        ax.plot([points[i, 0], projected[i, 0]],
               [points[i, 1], projected[i, 1]], 'g--', alpha=0.5)

    ax.scatter(points[:, 0], points[:, 1], c='red', s=50, label='Original', zorder=5)
    ax.scatter(projected[:, 0], projected[:, 1], c='blue', s=50, label='Projected', zorder=5)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Projection onto $y = x$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Projection onto a plane in R^3
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    # Plane: z = 0
    xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    zz = np.zeros_like(xx)

    ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue')

    # Points and projections
    points_3d = np.random.randn(10, 3)
    projected_3d = points_3d.copy()
    projected_3d[:, 2] = 0  # Project onto z=0

    for i in range(len(points_3d)):
        ax.plot([points_3d[i, 0], projected_3d[i, 0]],
               [points_3d[i, 1], projected_3d[i, 1]],
               [points_3d[i, 2], projected_3d[i, 2]], 'g--', alpha=0.7)

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
              c='red', s=50, label='Original')
    ax.scatter(projected_3d[:, 0], projected_3d[:, 1], projected_3d[:, 2],
              c='blue', s=50, label='Projected')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Projection onto $z = 0$ plane')
    ax.legend()

    plt.tight_layout()
    plt.savefig('projection_visualization.png', dpi=150)
    plt.show()

# ============================================================
# Part 3: Quantum Measurement Simulation
# ============================================================

def quantum_measurement_simulation():
    """
    Simulate quantum measurement and state collapse.
    """
    print("\n" + "=" * 60)
    print("Part 3: Quantum Measurement Simulation")
    print("=" * 60)

    # Initial state: |ψ⟩ = (|0⟩ + |1⟩ + |2⟩)/√3 (3-level system)
    psi = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)

    # Observable with eigenstates |0⟩, |1⟩, |2⟩
    # Projections
    P0 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
    P1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)
    P2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=complex)

    # Probabilities
    prob_0 = np.real(np.vdot(psi, P0 @ psi))
    prob_1 = np.real(np.vdot(psi, P1 @ psi))
    prob_2 = np.real(np.vdot(psi, P2 @ psi))

    print(f"Initial state: |ψ⟩ = (|0⟩ + |1⟩ + |2⟩)/√3")
    print(f"\nMeasurement probabilities:")
    print(f"  P(0) = ⟨ψ|P₀|ψ⟩ = {prob_0:.4f}")
    print(f"  P(1) = ⟨ψ|P₁|ψ⟩ = {prob_1:.4f}")
    print(f"  P(2) = ⟨ψ|P₂|ψ⟩ = {prob_2:.4f}")
    print(f"  Total = {prob_0 + prob_1 + prob_2:.4f}")

    # Simulate many measurements
    n_measurements = 10000
    outcomes = np.random.choice([0, 1, 2], size=n_measurements,
                                p=[prob_0, prob_1, prob_2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of outcomes
    axes[0].hist(outcomes, bins=[-0.5, 0.5, 1.5, 2.5], density=True,
                alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].bar([0, 1, 2], [prob_0, prob_1, prob_2], width=0.3, alpha=0.5,
               color='red', label='Theoretical')
    axes[0].set_xlabel('Measurement outcome')
    axes[0].set_ylabel('Probability')
    axes[0].set_title(f'Measurement Statistics ({n_measurements} trials)')
    axes[0].legend()
    axes[0].set_xticks([0, 1, 2])

    # State collapse visualization
    ax = axes[1]

    # Represent states as points on simplex (for visualization)
    # |ψ⟩ = c₀|0⟩ + c₁|1⟩ + c₂|2⟩
    # Plot |c₀|², |c₁|², |c₂|²

    # Initial state probabilities
    initial_probs = np.abs(psi)**2

    # Collapsed states
    states = {
        'Initial': initial_probs,
        'After 0': np.array([1, 0, 0]),
        'After 1': np.array([0, 1, 0]),
        'After 2': np.array([0, 0, 1])
    }

    x = np.arange(3)
    width = 0.2
    for i, (name, probs) in enumerate(states.items()):
        ax.bar(x + i*width, probs, width, label=name, alpha=0.7)

    ax.set_xlabel('Basis state')
    ax.set_ylabel('$|c_n|^2$')
    ax.set_title('State Before and After Measurement')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(['|0⟩', '|1⟩', '|2⟩'])
    ax.legend()

    plt.tight_layout()
    plt.savefig('quantum_measurement.png', dpi=150)
    plt.show()

# ============================================================
# Part 4: Resolution of Identity
# ============================================================

def resolution_of_identity():
    """
    Demonstrate the resolution of identity Σ|n⟩⟨n| = I.
    """
    print("\n" + "=" * 60)
    print("Part 4: Resolution of Identity")
    print("=" * 60)

    # For a spin-1 system (3 states)
    n = 3

    # Standard basis projections
    projections = []
    for k in range(n):
        ek = np.zeros(n, dtype=complex)
        ek[k] = 1
        Pk = np.outer(ek, ek.conj())
        projections.append(Pk)
        print(f"\nP_{k} = |{k}⟩⟨{k}| =")
        print(Pk)

    # Verify resolution of identity
    total = sum(projections)
    print(f"\nΣ Pₖ = I:")
    print(np.round(total, 4))
    print(f"Equals identity: {np.allclose(total, np.eye(n))}")

    # Expand a random state
    psi = np.random.randn(n) + 1j * np.random.randn(n)
    psi = psi / np.linalg.norm(psi)

    print(f"\nRandom state |ψ⟩ = {np.round(psi, 4)}")

    # Components
    reconstructed = np.zeros(n, dtype=complex)
    for k in range(n):
        ck = np.vdot(projections[k] @ np.array([1 if i==k else 0 for i in range(n)]), psi)
        print(f"  c_{k} = ⟨{k}|ψ⟩ = {ck:.4f}")
        reconstructed += ck * np.array([1 if i==k else 0 for i in range(n)])

    print(f"\nReconstructed: Σ cₖ|k⟩ = {np.round(reconstructed, 4)}")
    print(f"Matches original: {np.allclose(reconstructed, psi)}")

# ============================================================
# Part 5: Projection Lattice
# ============================================================

def projection_lattice():
    """
    Demonstrate the lattice structure of projections.
    """
    print("\n" + "=" * 60)
    print("Part 5: Projection Lattice")
    print("=" * 60)

    # In C^4, create nested subspaces
    # M1 = span{e1}
    # M2 = span{e1, e2}
    # M3 = span{e1, e2, e3}

    n = 4
    e = [np.zeros(n, dtype=complex) for _ in range(n)]
    for i in range(n):
        e[i][i] = 1

    P1 = np.outer(e[0], e[0].conj())
    P2 = np.outer(e[0], e[0].conj()) + np.outer(e[1], e[1].conj())
    P3 = P2 + np.outer(e[2], e[2].conj())

    # Verify P1 ≤ P2 ≤ P3 (meaning P1P2 = P2P1 = P1, etc.)
    print("Verifying P1 ≤ P2 ≤ P3:")
    print(f"  P1 @ P2 = P1: {np.allclose(P1 @ P2, P1)}")
    print(f"  P2 @ P3 = P2: {np.allclose(P2 @ P3, P2)}")
    print(f"  P1 @ P3 = P1: {np.allclose(P1 @ P3, P1)}")

    # Products and sums
    print(f"\nP1 @ P2 is a projection: {np.allclose((P1 @ P2) @ (P1 @ P2), P1 @ P2)}")

    # Orthogonal projections: P onto span{e1,e2}, Q onto span{e3,e4}
    P = P2
    Q = np.outer(e[2], e[2].conj()) + np.outer(e[3], e[3].conj())

    print(f"\nP and Q are orthogonal (PQ = 0): {np.allclose(P @ Q, 0)}")
    print(f"P + Q is a projection: {np.allclose((P + Q) @ (P + Q), P + Q)}")
    print(f"P + Q = I: {np.allclose(P + Q, np.eye(n))}")

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (proj, name) in zip(axes, [(P1, '$P_1$'), (P2, '$P_2$'),
                                        (P3, '$P_3$'), (Q, '$Q$')]):
        im = ax.imshow(np.abs(proj), cmap='Blues', vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Projection Lattice: $P_1 \\leq P_2 \\leq P_3$, $Q \\perp P_2$')
    plt.tight_layout()
    plt.savefig('projection_lattice.png', dpi=150)
    plt.show()

# ============================================================
# Part 6: Density Matrix and Mixed States
# ============================================================

def density_matrices():
    """
    Demonstrate density matrices as projections and mixed states.
    """
    print("\n" + "=" * 60)
    print("Part 6: Density Matrices")
    print("=" * 60)

    # Pure state |ψ⟩ = (|0⟩ + |1⟩)/√2
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_pure = np.outer(psi, psi.conj())

    print("Pure state density matrix ρ = |ψ⟩⟨ψ|:")
    print(np.round(rho_pure, 4))
    print(f"  Tr(ρ) = {np.trace(rho_pure):.4f}")
    print(f"  Tr(ρ²) = {np.trace(rho_pure @ rho_pure):.4f} (= 1 for pure state)")
    print(f"  ρ² = ρ (projection): {np.allclose(rho_pure @ rho_pure, rho_pure)}")

    # Mixed state: 50% |0⟩, 50% |1⟩
    e0 = np.array([1, 0], dtype=complex)
    e1 = np.array([0, 1], dtype=complex)
    rho_mixed = 0.5 * np.outer(e0, e0.conj()) + 0.5 * np.outer(e1, e1.conj())

    print("\nMixed state ρ = 0.5|0⟩⟨0| + 0.5|1⟩⟨1|:")
    print(np.round(rho_mixed, 4))
    print(f"  Tr(ρ) = {np.trace(rho_mixed):.4f}")
    print(f"  Tr(ρ²) = {np.trace(rho_mixed @ rho_mixed):.4f} (< 1 for mixed)")
    print(f"  ρ² = ρ: {np.allclose(rho_mixed @ rho_mixed, rho_mixed)}")

    # Visualize Bloch sphere representation
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Draw Bloch sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    for ax, rho, title in [(ax1, rho_pure, 'Pure State'),
                           (ax2, rho_mixed, 'Mixed State')]:
        ax.plot_surface(x, y, z, alpha=0.1, color='blue')

        # Bloch vector: r = (Tr(ρσx), Tr(ρσy), Tr(ρσz))
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])

        rx = np.real(np.trace(rho @ sigma_x))
        ry = np.real(np.trace(rho @ sigma_y))
        rz = np.real(np.trace(rho @ sigma_z))

        ax.quiver(0, 0, 0, rx, ry, rz, color='red', arrow_length_ratio=0.1,
                 linewidth=3)
        ax.scatter([rx], [ry], [rz], color='red', s=100)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title}\n$|r| = {np.sqrt(rx**2+ry**2+rz**2):.2f}$')

    plt.tight_layout()
    plt.savefig('bloch_sphere.png', dpi=150)
    plt.show()

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 243: Projections - Computational Lab")
    print("=" * 60)

    projection_properties()
    visualize_projection_2d()
    quantum_measurement_simulation()
    resolution_of_identity()
    projection_lattice()
    density_matrices()

    print("\n" + "=" * 60)
    print("Lab complete! Key takeaways:")
    print("  1. Orthogonal projections satisfy P² = P = P†")
    print("  2. Projections decompose H = ran(P) ⊕ ker(P)")
    print("  3. Quantum measurement is projection onto eigenspaces")
    print("  4. Resolution of identity: Σ|n⟩⟨n| = I")
    print("  5. Pure states are rank-1 projections (density matrices)")
    print("=" * 60)
```

---

## 8. Summary

### Key Definitions

| Concept | Definition |
|---------|------------|
| **Projection** | $P^2 = P$ |
| **Orthogonal Projection** | $P^2 = P = P^\dagger$ |
| **Range** | $\text{ran}(P) = \{Px : x \in \mathcal{H}\}$ |
| **Kernel** | $\ker(P) = \{x : Px = 0\}$ |

### Key Formulas

$$\boxed{\begin{aligned}
&\text{Orthogonal projection:} && P^2 = P = P^\dagger \\
&\text{Direct sum:} && \mathcal{H} = \text{ran}(P) \oplus \ker(P) \\
&\text{Complement:} && \ker(P) = (\text{ran}(P))^\perp \\
&\text{Rank-1 projection:} && P_u = |u\rangle\langle u| \\
&\text{Resolution of identity:} && \sum_n |e_n\rangle\langle e_n| = I \\
&\text{Measurement probability:} && P(\lambda) = \langle\psi|P_\lambda|\psi\rangle
\end{aligned}}$$

### Key Theorems

| Theorem | Statement |
|---------|-----------|
| **Projection Theorem** | Closed $M \Rightarrow$ unique orthogonal projection $P_M$ |
| **Decomposition** | $\mathcal{H} = M \oplus M^\perp$ for closed $M$ |
| **Best Approximation** | $\|x - P_M x\| = \text{dist}(x, M)$ |

### Key Insights

1. **Projections decompose Hilbert space** into orthogonal subspaces
2. **Measurement = projection** onto eigenspaces
3. **State collapse** is normalization after projection
4. **Resolution of identity** ensures probability conservation
5. **Density matrices** generalize pure state projections

---

## 9. Daily Checklist

- [ ] I can define projection and orthogonal projection
- [ ] I can prove the projection theorem
- [ ] I can verify $P^2 = P = P^\dagger$ for specific projections
- [ ] I can compute projections onto subspaces
- [ ] I understand the resolution of identity
- [ ] I can explain quantum measurement as projection
- [ ] I can work with density matrices
- [ ] I completed the computational lab exercises

---

## 10. Preview: Day 244

Tomorrow we study **compact operators**, which are limits of finite-rank operators. Compact operators have remarkable spectral properties:
- Discrete spectrum (except possibly 0)
- Eigenspaces are finite-dimensional
- Eigenvalues accumulate only at 0

We'll study Hilbert-Schmidt operators, trace class operators, and their role in quantum mechanics (density matrices are trace class!).

---

*"The orthogonal projection is the mathematical embodiment of the physical process of measurement. When we observe a quantum system, we project it onto the eigenspace corresponding to the observed value."* — John von Neumann
