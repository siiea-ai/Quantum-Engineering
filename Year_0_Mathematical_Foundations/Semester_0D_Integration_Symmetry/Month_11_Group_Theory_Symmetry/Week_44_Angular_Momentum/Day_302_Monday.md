# Day 302: Angular Momentum Operators in Quantum Mechanics

## Overview

**Month 11, Week 44, Day 1 — Monday**

Today we connect our SU(2) representation theory directly to quantum mechanics. Angular momentum operators arise as generators of rotations, and their eigenvalue spectrum follows from the Lie algebra structure. This is where abstract mathematics becomes physical reality.

## Learning Objectives

1. Derive angular momentum operators from rotation generators
2. Establish the fundamental commutation relations
3. Solve the eigenvalue problem using algebraic methods
4. Understand the physical meaning of quantum numbers $(j, m)$
5. Connect to classical angular momentum via correspondence principle

---

## 1. From Rotations to Observables

### The Generator Connection

In quantum mechanics, a rotation by angle $\theta$ about axis $\hat{n}$ is implemented by:

$$\boxed{U(\hat{n}, \theta) = e^{-i\theta \hat{n} \cdot \mathbf{J}/\hbar}}$$

The angular momentum operators $J_x, J_y, J_z$ are the **generators** of rotations.

### Why the Factor of $\hbar$?

Dimensional analysis:
- $[J] = $ energy × time = $[\hbar]$
- Exponent must be dimensionless
- Thus $J/\hbar$ appears in the exponential

### Classical Correspondence

For a particle at position $\mathbf{r}$ with momentum $\mathbf{p}$:

$$\boxed{\mathbf{L} = \mathbf{r} \times \mathbf{p}}$$

In quantum mechanics:
$$\hat{L}_x = \hat{y}\hat{p}_z - \hat{z}\hat{p}_y = -i\hbar\left(y\frac{\partial}{\partial z} - z\frac{\partial}{\partial y}\right)$$

and cyclic permutations.

---

## 2. The Fundamental Commutation Relations

### Angular Momentum Algebra

From the rotation group structure (Week 43), we have:

$$\boxed{[J_i, J_j] = i\hbar \epsilon_{ijk} J_k}$$

Explicitly:
$$[J_x, J_y] = i\hbar J_z$$
$$[J_y, J_z] = i\hbar J_x$$
$$[J_z, J_x] = i\hbar J_y$$

### The Casimir Operator

$$\mathbf{J}^2 = J_x^2 + J_y^2 + J_z^2$$

**Key property:** $[\mathbf{J}^2, J_i] = 0$ for all $i$.

**Proof for $[\mathbf{J}^2, J_z]$:**
$$[\mathbf{J}^2, J_z] = [J_x^2, J_z] + [J_y^2, J_z]$$

Using $[AB, C] = A[B, C] + [A, C]B$:
$$[J_x^2, J_z] = J_x[J_x, J_z] + [J_x, J_z]J_x = J_x(-i\hbar J_y) + (-i\hbar J_y)J_x$$
$$= -i\hbar(J_x J_y + J_y J_x)$$

Similarly:
$$[J_y^2, J_z] = i\hbar(J_x J_y + J_y J_x)$$

These cancel: $[\mathbf{J}^2, J_z] = 0$ ✓

### Complete Set of Commuting Observables

Since $[\mathbf{J}^2, J_z] = 0$, we can find simultaneous eigenstates of both.

Convention: Label states by eigenvalues of $\mathbf{J}^2$ and $J_z$.

---

## 3. Ladder Operators

### Definition

$$J_+ = J_x + i J_y$$
$$J_- = J_x - i J_y$$

### Commutation Relations

$$[J_z, J_+] = \hbar J_+$$
$$[J_z, J_-] = -\hbar J_-$$
$$[J_+, J_-] = 2\hbar J_z$$

### Physical Interpretation

If $|m\rangle$ is eigenstate of $J_z$ with eigenvalue $m\hbar$:
$$J_z(J_+|m\rangle) = (J_+J_z + [J_z, J_+])|m\rangle = (m+1)\hbar (J_+|m\rangle)$$

Thus $J_+|m\rangle$ is eigenstate with eigenvalue $(m+1)\hbar$.

**$J_+$ raises $m$ by 1, $J_-$ lowers $m$ by 1.**

---

## 4. Solving the Eigenvalue Problem

### Step 1: Express $\mathbf{J}^2$ Using Ladder Operators

$$J_+ J_- = (J_x + iJ_y)(J_x - iJ_y) = J_x^2 + J_y^2 - i[J_x, J_y]$$
$$= J_x^2 + J_y^2 + \hbar J_z$$

Therefore:
$$\mathbf{J}^2 = J_+ J_- + J_z^2 - \hbar J_z = J_- J_+ + J_z^2 + \hbar J_z$$

### Step 2: Establish Bounds

Let $|j, m\rangle$ be eigenstate with $\mathbf{J}^2|j, m\rangle = \hbar^2 \lambda |j, m\rangle$ and $J_z|j, m\rangle = \hbar m |j, m\rangle$.

**Positivity of $J_x^2 + J_y^2$:**
$$\langle j, m | (J_x^2 + J_y^2) | j, m \rangle \geq 0$$
$$\hbar^2(\lambda - m^2) \geq 0$$
$$\implies |m| \leq \sqrt{\lambda}$$

### Step 3: Find Maximum and Minimum $m$

There must exist $m_{\max}$ and $m_{\min}$ such that:
$$J_+ |j, m_{\max}\rangle = 0$$
$$J_- |j, m_{\min}\rangle = 0$$

From $J_- J_+ |j, m_{\max}\rangle = 0$:
$$(\mathbf{J}^2 - J_z^2 - \hbar J_z)|j, m_{\max}\rangle = 0$$
$$\hbar^2(\lambda - m_{\max}^2 - m_{\max}) = 0$$
$$\lambda = m_{\max}(m_{\max} + 1)$$

Similarly, from $J_+ J_-|j, m_{\min}\rangle = 0$:
$$\lambda = m_{\min}(m_{\min} - 1)$$

### Step 4: Relate $m_{\max}$ and $m_{\min}$

From $m_{\max}(m_{\max}+1) = m_{\min}(m_{\min}-1)$:
$$(m_{\max} + m_{\min})(m_{\max} - m_{\min} + 1) = 0$$

Since $m_{\max} \geq m_{\min}$, we must have:
$$m_{\max} = -m_{\min} \equiv j$$

### Step 5: Quantization

The ladder connects $m_{\min} = -j$ to $m_{\max} = j$ in integer steps:
$$j - (-j) = 2j = \text{non-negative integer}$$

$$\boxed{j = 0, \frac{1}{2}, 1, \frac{3}{2}, 2, \ldots}$$

---

## 5. The Complete Spectrum

### Eigenvalue Equations

$$\boxed{\mathbf{J}^2 |j, m\rangle = \hbar^2 j(j+1) |j, m\rangle}$$
$$\boxed{J_z |j, m\rangle = \hbar m |j, m\rangle}$$

where:
- $j = 0, \frac{1}{2}, 1, \frac{3}{2}, \ldots$ (spin quantum number)
- $m = -j, -j+1, \ldots, j-1, j$ (magnetic quantum number)
- Each $j$ has $2j+1$ values of $m$

### Ladder Operator Matrix Elements

$$\boxed{J_\pm |j, m\rangle = \hbar\sqrt{j(j+1) - m(m \pm 1)} |j, m \pm 1\rangle}$$

**Derivation:**
$$\langle j, m \pm 1 | J_\pm | j, m \rangle = \sqrt{\langle j, m | J_\mp J_\pm | j, m \rangle}$$

Using $J_\mp J_\pm = \mathbf{J}^2 - J_z^2 \mp \hbar J_z$:
$$= \hbar\sqrt{j(j+1) - m^2 \mp m} = \hbar\sqrt{j(j+1) - m(m \pm 1)}$$

---

## 6. Physical Interpretation

### The Vector Model

Classically, angular momentum is a vector with definite direction. Quantum mechanically:

- **Magnitude:** $|\mathbf{J}| = \hbar\sqrt{j(j+1)}$
- **z-component:** $J_z = \hbar m$
- **x, y components:** Uncertain (don't commute with $J_z$)

The "cone of uncertainty":
$$J_x^2 + J_y^2 = \mathbf{J}^2 - J_z^2 = \hbar^2[j(j+1) - m^2]$$

Maximum z-projection: $m = j$ gives $J_z = \hbar j < \hbar\sqrt{j(j+1)} = |\mathbf{J}|$

**Angular momentum can never point exactly along any axis!**

### Uncertainty Relations

$$\Delta J_x \cdot \Delta J_y \geq \frac{\hbar}{2}|\langle J_z \rangle|$$

For $|j, m\rangle$:
$$\Delta J_x = \Delta J_y = \frac{\hbar}{2}\sqrt{j(j+1) - m^2 + \frac{1}{2}}$$

---

## 7. Examples: Constructing Representations

### Example 1: $j = 1/2$ (Spin-1/2)

Basis: $|{+}\rangle \equiv |{\frac{1}{2}, \frac{1}{2}}\rangle$, $|{-}\rangle \equiv |{\frac{1}{2}, -\frac{1}{2}}\rangle$

$$J_z = \frac{\hbar}{2}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

$$J_+ = \hbar\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad J_- = \hbar\begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$$

$$J_x = \frac{\hbar}{2}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad J_y = \frac{\hbar}{2}\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

These are the **Pauli matrices** (times $\hbar/2$):
$$\mathbf{J} = \frac{\hbar}{2}\boldsymbol{\sigma}$$

### Example 2: $j = 1$

Basis: $|1, 1\rangle$, $|1, 0\rangle$, $|1, -1\rangle$

$$J_z = \hbar\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

$$J_+ = \hbar\sqrt{2}\begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

Verify: $J_+|1, 0\rangle = \hbar\sqrt{1 \cdot 2 - 0 \cdot 1}|1, 1\rangle = \hbar\sqrt{2}|1, 1\rangle$ ✓

---

## 8. Computational Lab

```python
"""
Day 302: Angular Momentum Operators in Quantum Mechanics
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

class AngularMomentum:
    """
    Angular momentum operators for spin-j.
    Uses units where hbar = 1.
    """

    def __init__(self, j):
        """Initialize angular momentum with quantum number j."""
        if 2*j != int(2*j) or j < 0:
            raise ValueError("j must be non-negative half-integer")
        self.j = j
        self.dim = int(2*j + 1)
        self._build_operators()

    def _build_operators(self):
        """Construct Jx, Jy, Jz, J+, J-, J^2 matrices."""
        j = self.j
        dim = self.dim

        # m values from j down to -j
        self.m_values = np.arange(j, -j - 1, -1)

        # Jz is diagonal
        self.Jz = np.diag(self.m_values)

        # Ladder operators
        self.Jp = np.zeros((dim, dim), dtype=complex)
        self.Jm = np.zeros((dim, dim), dtype=complex)

        for i in range(dim - 1):
            m = self.m_values[i]
            # J+ |j, m-1> = sqrt(j(j+1) - (m-1)m) |j, m>
            self.Jp[i, i + 1] = np.sqrt(j*(j+1) - self.m_values[i+1]*(self.m_values[i+1] + 1))
            # J- |j, m> = sqrt(j(j+1) - m(m-1)) |j, m-1>
            self.Jm[i + 1, i] = np.sqrt(j*(j+1) - m*(m - 1))

        # Jx and Jy from ladder operators
        self.Jx = (self.Jp + self.Jm) / 2
        self.Jy = (self.Jp - self.Jm) / (2j)

        # J^2 = Jx^2 + Jy^2 + Jz^2
        self.J2 = self.Jx @ self.Jx + self.Jy @ self.Jy + self.Jz @ self.Jz

    def verify_commutation_relations(self):
        """Verify [Ji, Jj] = i*epsilon_ijk*Jk."""
        Jx, Jy, Jz = self.Jx, self.Jy, self.Jz

        results = {
            "[Jx, Jy] = iJz": np.allclose(Jx @ Jy - Jy @ Jx, 1j * Jz),
            "[Jy, Jz] = iJx": np.allclose(Jy @ Jz - Jz @ Jy, 1j * Jx),
            "[Jz, Jx] = iJy": np.allclose(Jz @ Jx - Jx @ Jz, 1j * Jy),
            "[J^2, Jz] = 0": np.allclose(self.J2 @ Jz - Jz @ self.J2, 0)
        }
        return results

    def verify_eigenvalues(self):
        """Verify J^2 and Jz eigenvalues."""
        j = self.j

        # J^2 should have all eigenvalues = j(j+1)
        J2_expected = j * (j + 1) * np.eye(self.dim)

        # Jz eigenvalues are m = j, j-1, ..., -j
        Jz_expected = np.diag(self.m_values)

        return {
            "J^2 = j(j+1)": np.allclose(self.J2, J2_expected),
            "Jz diagonal": np.allclose(self.Jz, Jz_expected)
        }

    def rotation_operator(self, axis, theta):
        """
        Generate rotation operator U = exp(-i*theta*n.J).

        Parameters:
            axis: rotation axis (will be normalized)
            theta: rotation angle
        """
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        J_n = axis[0] * self.Jx + axis[1] * self.Jy + axis[2] * self.Jz
        return expm(-1j * theta * J_n)

    def expectation_values(self, state):
        """Compute <Jx>, <Jy>, <Jz> for given state."""
        state = np.array(state, dtype=complex)
        state = state / np.linalg.norm(state)

        return {
            '<Jx>': np.real(state.conj() @ self.Jx @ state),
            '<Jy>': np.real(state.conj() @ self.Jy @ state),
            '<Jz>': np.real(state.conj() @ self.Jz @ state),
            '<J^2>': np.real(state.conj() @ self.J2 @ state)
        }

    def uncertainties(self, state):
        """Compute Delta Jx, Delta Jy, Delta Jz for given state."""
        state = np.array(state, dtype=complex)
        state = state / np.linalg.norm(state)

        def delta(J):
            J_avg = np.real(state.conj() @ J @ state)
            J2_avg = np.real(state.conj() @ J @ J @ state)
            return np.sqrt(J2_avg - J_avg**2)

        return {
            'Delta Jx': delta(self.Jx),
            'Delta Jy': delta(self.Jy),
            'Delta Jz': delta(self.Jz)
        }


def demonstrate_angular_momentum():
    """Demonstrate angular momentum operators."""
    print("=" * 60)
    print("ANGULAR MOMENTUM DEMONSTRATION")
    print("=" * 60)

    for j in [0.5, 1, 1.5, 2]:
        print(f"\n{'='*60}")
        print(f"j = {j}, dimension = {int(2*j + 1)}")
        print("=" * 60)

        am = AngularMomentum(j)

        # Verify commutation relations
        print("\nCommutation Relations:")
        for relation, satisfied in am.verify_commutation_relations().items():
            status = "✓" if satisfied else "✗"
            print(f"  {relation}: {status}")

        # Verify eigenvalues
        print("\nEigenvalue Verification:")
        for prop, satisfied in am.verify_eigenvalues().items():
            status = "✓" if satisfied else "✗"
            print(f"  {prop}: {status}")

        # Show matrices for small j
        if j <= 1:
            print(f"\nJz matrix:\n{np.real(am.Jz)}")
            print(f"\nJx matrix:\n{np.real(am.Jx).round(4)}")


def demonstrate_uncertainty_principle():
    """Show angular momentum uncertainty relations."""
    print("\n" + "=" * 60)
    print("UNCERTAINTY PRINCIPLE DEMONSTRATION")
    print("=" * 60)

    j = 1
    am = AngularMomentum(j)

    # Different states
    states = {
        '|1, 1>': [1, 0, 0],
        '|1, 0>': [0, 1, 0],
        '|1, -1>': [0, 0, 1],
        '(|1,1> + |1,-1>)/sqrt(2)': [1/np.sqrt(2), 0, 1/np.sqrt(2)]
    }

    for name, state in states.items():
        print(f"\nState: {name}")
        exp = am.expectation_values(state)
        unc = am.uncertainties(state)

        print(f"  <Jx> = {exp['<Jx>']:.4f}, Delta Jx = {unc['Delta Jx']:.4f}")
        print(f"  <Jy> = {exp['<Jy>']:.4f}, Delta Jy = {unc['Delta Jy']:.4f}")
        print(f"  <Jz> = {exp['<Jz>']:.4f}, Delta Jz = {unc['Delta Jz']:.4f}")

        # Check uncertainty relation: Delta Jx * Delta Jy >= |<Jz>|/2
        lhs = unc['Delta Jx'] * unc['Delta Jy']
        rhs = np.abs(exp['<Jz>']) / 2
        print(f"  Uncertainty: Delta Jx * Delta Jy = {lhs:.4f} >= {rhs:.4f} = |<Jz>|/2: {lhs >= rhs - 1e-10}")


def visualize_angular_momentum_cone():
    """Visualize the 'cone of uncertainty' for angular momentum."""
    fig = plt.figure(figsize=(12, 5))

    j = 2
    am = AngularMomentum(j)

    for idx, m in enumerate([2, 1, 0]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

        # The cone: |J| = sqrt(j(j+1)), Jz = m
        J_mag = np.sqrt(j * (j + 1))
        J_perp = np.sqrt(j * (j + 1) - m**2)

        # Draw cone
        theta = np.linspace(0, 2*np.pi, 100)
        x = J_perp * np.cos(theta)
        y = J_perp * np.sin(theta)
        z = np.ones_like(theta) * m

        ax.plot(x, y, z, 'b-', linewidth=2)

        # Draw z-axis
        ax.plot([0, 0], [0, 0], [-j-1, j+1], 'k--', alpha=0.5)

        # Draw J vector
        ax.quiver(0, 0, 0, 0, J_perp, m, color='r', arrow_length_ratio=0.1)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('Jx')
        ax.set_ylabel('Jy')
        ax.set_zlabel('Jz')
        ax.set_title(f'j = {j}, m = {m}')

    plt.tight_layout()
    plt.savefig('angular_momentum_cone.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: angular_momentum_cone.png")


def demonstrate_rotation():
    """Show rotation operator action."""
    print("\n" + "=" * 60)
    print("ROTATION OPERATOR DEMONSTRATION")
    print("=" * 60)

    j = 0.5
    am = AngularMomentum(j)

    # Start with |+> = |1/2, 1/2>
    state = np.array([1, 0], dtype=complex)

    print(f"\nInitial state: |+> (spin up along z)")
    exp = am.expectation_values(state)
    print(f"  <Jx> = {exp['<Jx>']:.4f}")
    print(f"  <Jy> = {exp['<Jy>']:.4f}")
    print(f"  <Jz> = {exp['<Jz>']:.4f}")

    # Rotate by pi/2 about y-axis
    U = am.rotation_operator([0, 1, 0], np.pi/2)
    rotated_state = U @ state

    print(f"\nAfter π/2 rotation about y-axis:")
    exp = am.expectation_values(rotated_state)
    print(f"  <Jx> = {exp['<Jx>']:.4f}")
    print(f"  <Jy> = {exp['<Jy>']:.4f}")
    print(f"  <Jz> = {exp['<Jz>']:.4f}")

    # Full 2π rotation
    U_2pi = am.rotation_operator([0, 0, 1], 2*np.pi)
    state_2pi = U_2pi @ state

    print(f"\nAfter 2π rotation about z-axis:")
    print(f"  |ψ> = {state.round(4)}")
    print(f"  U(2π)|ψ> = {state_2pi.round(4)}")
    print(f"  Phase factor: {state_2pi[0]/state[0]:.4f}")
    print("  (For spin-1/2, 2π rotation gives -1 phase!)")


# Main execution
if __name__ == "__main__":
    demonstrate_angular_momentum()
    demonstrate_uncertainty_principle()
    demonstrate_rotation()
    visualize_angular_momentum_cone()
```

---

## 9. Practice Problems

### Problem 1: Commutation Relations

Prove that $[J_+, J_-] = 2\hbar J_z$ directly from the definition of $J_\pm$.

### Problem 2: Casimir Eigenvalue

Show that $\mathbf{J}^2 = J_- J_+ + J_z^2 + \hbar J_z$ and use this to find the eigenvalue of $\mathbf{J}^2$ on $|j, j\rangle$.

### Problem 3: Matrix Elements

For $j = 3/2$, construct the explicit $4 \times 4$ matrices for $J_z$ and $J_+$.

### Problem 4: Expectation Values

For the state $|\psi\rangle = \frac{1}{\sqrt{2}}(|1, 1\rangle + |1, -1\rangle)$, calculate:
- (a) $\langle J_z \rangle$
- (b) $\langle J_x \rangle$
- (c) $\langle \mathbf{J}^2 \rangle$
- (d) $\Delta J_z$

### Problem 5: Rotation Operator

Show that for spin-1/2:
$$e^{-i\theta J_z/\hbar} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

---

## Summary

### Key Results

$$\boxed{[J_i, J_j] = i\hbar\epsilon_{ijk}J_k \implies j = 0, \frac{1}{2}, 1, \frac{3}{2}, \ldots}$$

### The Quantum Numbers

| Symbol | Name | Allowed Values | Physical Meaning |
|--------|------|----------------|------------------|
| $j$ | Total angular momentum | $0, \frac{1}{2}, 1, \frac{3}{2}, \ldots$ | $\|\mathbf{J}\| = \hbar\sqrt{j(j+1)}$ |
| $m$ | Magnetic quantum number | $-j, -j+1, \ldots, j$ | $J_z = \hbar m$ |

### The Ladder Structure

$$J_+ |j, m\rangle = \hbar\sqrt{j(j+1) - m(m+1)} |j, m+1\rangle$$
$$J_- |j, m\rangle = \hbar\sqrt{j(j+1) - m(m-1)} |j, m-1\rangle$$

---

## Preview: Day 303

Tomorrow we study **spherical harmonics** — the wavefunctions for orbital angular momentum. These are the eigenfunctions of $\hat{L}^2$ and $\hat{L}_z$ in position space: $Y_\ell^m(\theta, \phi)$.
