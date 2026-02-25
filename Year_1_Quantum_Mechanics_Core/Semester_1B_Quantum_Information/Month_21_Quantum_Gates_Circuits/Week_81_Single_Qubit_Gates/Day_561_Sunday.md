# Day 561: Pauli Gates (X, Y, Z)

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: Pauli matrices, eigenstructure, algebraic properties |
| Afternoon | 2.5 hours | Problem solving: Gate operations and commutation relations |
| Evening | 1.5 hours | Computational lab: Simulating Pauli gates |

## Learning Objectives

By the end of today, you will be able to:

1. **Write the matrix representations** of all three Pauli gates X, Y, Z
2. **Calculate eigenvalues and eigenvectors** of each Pauli matrix
3. **Apply Pauli gates** to arbitrary single-qubit states
4. **Derive anti-commutation relations** {σᵢ, σⱼ} = 2δᵢⱼI
5. **Visualize Pauli gate actions** on the Bloch sphere as π rotations
6. **Verify algebraic identities** including X² = Y² = Z² = I

---

## Core Content

### 1. Introduction to Quantum Gates

A **quantum gate** is a unitary operator that transforms quantum states:

$$|\psi'\rangle = U|\psi\rangle$$

For single-qubit gates, U is a 2×2 unitary matrix satisfying:

$$U^\dagger U = U U^\dagger = I$$

The **Pauli gates** are the most fundamental single-qubit gates, representing π rotations about the coordinate axes of the Bloch sphere.

### 2. The Pauli X Gate (Bit Flip)

The Pauli X gate is the quantum analog of the classical NOT gate:

$$\boxed{X = \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}}$$

**Action on computational basis:**
$$X|0\rangle = |1\rangle, \quad X|1\rangle = |0\rangle$$

**Action on general state** $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:
$$X|\psi\rangle = \alpha|1\rangle + \beta|0\rangle = \beta|0\rangle + \alpha|1\rangle$$

**Eigenstructure:**
- Eigenvalues: $\lambda = \pm 1$
- Eigenvectors: $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ for λ = +1
- Eigenvectors: $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$ for λ = -1

**Bloch sphere interpretation:** X is a π rotation about the x-axis:
$$X = e^{-i\pi \sigma_x/2} \cdot (-i) = -i R_x(\pi)$$

More precisely, $R_x(\pi) = -iX$, so X differs from a pure rotation by a global phase.

### 3. The Pauli Y Gate

$$\boxed{Y = \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}}$$

**Action on computational basis:**
$$Y|0\rangle = i|1\rangle, \quad Y|1\rangle = -i|0\rangle$$

Note the phase factors! Y is not simply a "bit flip with phase."

**Eigenstructure:**
- Eigenvalues: $\lambda = \pm 1$
- Eigenvectors: $|+_y\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$ for λ = +1
- Eigenvectors: $|-_y\rangle = \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle)$ for λ = -1

**Bloch sphere interpretation:** Y is a π rotation about the y-axis.

**Key identity:**
$$Y = iXZ = -iZX$$

### 4. The Pauli Z Gate (Phase Flip)

$$\boxed{Z = \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}}$$

**Action on computational basis:**
$$Z|0\rangle = |0\rangle, \quad Z|1\rangle = -|1\rangle$$

The Z gate leaves $|0\rangle$ unchanged but flips the sign of $|1\rangle$.

**Action on superposition:**
$$Z|+\rangle = |-\rangle, \quad Z|-\rangle = |+\rangle$$

**Eigenstructure:**
- Eigenvalues: $\lambda = \pm 1$
- Eigenvectors: $|0\rangle$ for λ = +1, $|1\rangle$ for λ = -1

The computational basis states are eigenstates of Z!

**Bloch sphere interpretation:** Z is a π rotation about the z-axis.

### 5. Algebraic Properties

**Fundamental identities:**

$$\boxed{X^2 = Y^2 = Z^2 = I}$$

This means each Pauli gate is its own inverse: applying it twice returns to the original state.

**Anti-commutation relations:**

$$\boxed{\{X, Y\} = XY + YX = 0}$$
$$\boxed{\{Y, Z\} = YZ + ZY = 0}$$
$$\boxed{\{Z, X\} = ZX + XZ = 0}$$

Any two different Pauli matrices anti-commute!

**Commutation relations:**

$$[X, Y] = XY - YX = 2iZ$$
$$[Y, Z] = YZ - ZY = 2iX$$
$$[Z, X] = ZX - ZX = 2iY$$

Or more compactly:
$$\boxed{[\sigma_j, \sigma_k] = 2i\epsilon_{jkl}\sigma_l}$$

**Product relations:**

$$XY = iZ, \quad YZ = iX, \quad ZX = iY$$

These are cyclic (like cross products)!

**General anti-commutation:**

$$\boxed{\{\sigma_j, \sigma_k\} = 2\delta_{jk}I}$$

Combined with the commutator:
$$\sigma_j \sigma_k = \delta_{jk}I + i\epsilon_{jkl}\sigma_l$$

### 6. Pauli Group

The **Pauli group** on one qubit consists of all products of Pauli matrices with phases:

$$\mathcal{P}_1 = \{\pm I, \pm iI, \pm X, \pm iX, \pm Y, \pm iY, \pm Z, \pm iZ\}$$

This group has 16 elements and is closed under multiplication.

**Group properties:**
- Identity: I
- Each element has finite order (divides 4)
- $X^4 = Y^4 = Z^4 = I$

### 7. Trace and Orthogonality

The Pauli matrices (including I) form an orthonormal basis for 2×2 Hermitian matrices under the Hilbert-Schmidt inner product:

$$\text{Tr}(\sigma_j^\dagger \sigma_k) = 2\delta_{jk}$$

Any 2×2 matrix M can be written as:

$$M = \frac{1}{2}\sum_{j=0}^{3} \text{Tr}(\sigma_j M)\sigma_j$$

where $\sigma_0 = I$.

### 8. Connection to Quantum Error Correction

The Pauli operators represent the fundamental types of single-qubit errors:

| Error Type | Pauli | Physical Meaning |
|------------|-------|------------------|
| Bit flip | X | Population exchange |
| Phase flip | Z | Relative phase error |
| Bit-phase flip | Y = iXZ | Combined error |

Any single-qubit error can be decomposed into combinations of these!

---

## Quantum Computing Connection

Pauli gates are foundational to quantum computing:

1. **Quantum Error Correction:** The stabilizer formalism is built on Pauli operators
2. **Variational Quantum Algorithms:** Pauli strings form the basis for Hamiltonian decomposition
3. **Measurement:** Pauli observables are the standard measurements in quantum computing
4. **Clifford Gates:** The Clifford group is defined by gates that preserve the Pauli group under conjugation

---

## Worked Examples

### Example 1: Verifying Anti-commutation

**Problem:** Verify that {X, Z} = XZ + ZX = 0.

**Solution:**

Calculate XZ:
$$XZ = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$

Calculate ZX:
$$ZX = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$$

Sum:
$$XZ + ZX = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} + \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix} = 0$$

**Verified!** ✓

### Example 2: Eigenvectors of X

**Problem:** Find the eigenvectors of X and verify they are orthonormal.

**Solution:**

The eigenvalue equation is $X|\lambda\rangle = \lambda|\lambda\rangle$.

For λ = +1:
$$\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} a \\ b \end{pmatrix}$$

This gives b = a, so $|+\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$.

For λ = -1:
$$\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} a \\ b \end{pmatrix} = -\begin{pmatrix} a \\ b \end{pmatrix}$$

This gives b = -a, so $|-\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$.

**Orthonormality check:**
$$\langle +|-\rangle = \frac{1}{2}(1 \cdot 1 + 1 \cdot (-1)) = 0 \checkmark$$
$$\langle +|+\rangle = \frac{1}{2}(1 + 1) = 1 \checkmark$$

### Example 3: Action on Superposition

**Problem:** Apply Y to $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$.

**Solution:**

$$Y|\psi\rangle = \frac{1}{\sqrt{3}}Y|0\rangle + \sqrt{\frac{2}{3}}Y|1\rangle$$

$$= \frac{1}{\sqrt{3}}(i|1\rangle) + \sqrt{\frac{2}{3}}(-i|0\rangle)$$

$$= -i\sqrt{\frac{2}{3}}|0\rangle + \frac{i}{\sqrt{3}}|1\rangle$$

We can factor out a global phase:
$$= i\left(-\sqrt{\frac{2}{3}}|0\rangle + \frac{1}{\sqrt{3}}|1\rangle\right)$$

The state's physical properties (probabilities) are changed: previously P(0) = 1/3, P(1) = 2/3, and now P(0) = 2/3, P(1) = 1/3.

---

## Practice Problems

### Direct Application

1. Calculate the matrix product ZY and verify ZY = -iX.

2. Show that the Y gate eigenvectors $|±_y\rangle$ satisfy $\langle +_y|-_y\rangle = 0$.

3. Apply X, Y, and Z gates to $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$. Express results in the computational basis.

### Intermediate

4. **Pauli decomposition:** Express the matrix $M = \begin{pmatrix} 2 & 1-i \\ 1+i & 0 \end{pmatrix}$ as a linear combination of I, X, Y, Z.

5. Prove that $e^{i\theta X} = \cos\theta \cdot I + i\sin\theta \cdot X$ using the Taylor series and the fact that $X^2 = I$.

6. Show that if $[A, B] = 0$ (A and B commute), then $e^{A+B} = e^A e^B$. Why doesn't this work for Pauli matrices?

### Challenging

7. **Simultaneous eigenstates:** Prove that two operators can share a complete set of eigenstates if and only if they commute. Use this to explain why Z and X cannot share eigenstates.

8. The **Pauli vector** is $\vec{\sigma} = (X, Y, Z)$. For a unit vector $\hat{n} = (n_x, n_y, n_z)$, show that:
   $$(\hat{n} \cdot \vec{\sigma})^2 = I$$
   and find the eigenvalues and eigenvectors of $\hat{n} \cdot \vec{\sigma}$.

9. **Density matrix evolution:** If $\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$ represents a qubit state with Bloch vector $\vec{r}$, find $X\rho X$ and express the result in terms of a new Bloch vector.

---

## Computational Lab: Simulating Pauli Gates

```python
"""
Day 561: Pauli Gates Simulation
Exploring X, Y, Z gates, their properties, and Bloch sphere visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Define computational basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Define Hadamard basis states
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

print("=" * 60)
print("PAULI GATES: FUNDAMENTAL PROPERTIES")
print("=" * 60)

# Verify X^2 = Y^2 = Z^2 = I
print("\n1. Verifying squared equals identity:")
print(f"   X² = I: {np.allclose(X @ X, I)}")
print(f"   Y² = I: {np.allclose(Y @ Y, I)}")
print(f"   Z² = I: {np.allclose(Z @ Z, I)}")

# Verify anti-commutation relations
print("\n2. Anti-commutation relations {σᵢ, σⱼ} = 0 for i ≠ j:")
print(f"   {{X, Y}} = 0: {np.allclose(X @ Y + Y @ X, np.zeros((2,2)))}")
print(f"   {{Y, Z}} = 0: {np.allclose(Y @ Z + Z @ Y, np.zeros((2,2)))}")
print(f"   {{Z, X}} = 0: {np.allclose(Z @ X + X @ Z, np.zeros((2,2)))}")

# Verify commutation relations
print("\n3. Commutation relations:")
print(f"   [X, Y] = 2iZ: {np.allclose(X @ Y - Y @ X, 2j * Z)}")
print(f"   [Y, Z] = 2iX: {np.allclose(Y @ Z - Z @ Y, 2j * X)}")
print(f"   [Z, X] = 2iY: {np.allclose(Z @ X - X @ Z, 2j * Y)}")

# Product relations
print("\n4. Product relations:")
print(f"   XY = iZ: {np.allclose(X @ Y, 1j * Z)}")
print(f"   YZ = iX: {np.allclose(Y @ Z, 1j * X)}")
print(f"   ZX = iY: {np.allclose(Z @ X, 1j * Y)}")

# Eigenvalue decomposition
print("\n5. Eigenvalues and Eigenvectors:")
for name, gate in [('X', X), ('Y', Y), ('Z', Z)]:
    eigenvalues, eigenvectors = np.linalg.eig(gate)
    print(f"\n   {name} gate:")
    print(f"   Eigenvalues: {eigenvalues}")
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        print(f"   Eigenvector for λ={val:.1f}: {vec}")

# Gate actions on basis states
print("\n6. Gate actions on computational basis:")
print(f"   X|0⟩ = |1⟩: {np.allclose(X @ ket_0, ket_1)}")
print(f"   X|1⟩ = |0⟩: {np.allclose(X @ ket_1, ket_0)}")
print(f"   Z|0⟩ = |0⟩: {np.allclose(Z @ ket_0, ket_0)}")
print(f"   Z|1⟩ = -|1⟩: {np.allclose(Z @ ket_1, -ket_1)}")

# Trace orthogonality
print("\n7. Trace orthogonality Tr(σᵢ†σⱼ) = 2δᵢⱼ:")
paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
print("   Trace inner products (should be 2 on diagonal, 0 off-diagonal):")
for name1, p1 in paulis.items():
    row = []
    for name2, p2 in paulis.items():
        trace = np.trace(p1.conj().T @ p2)
        row.append(f"{trace.real:.0f}")
    print(f"   {name1}: {row}")

# Function to compute Bloch vector from state
def state_to_bloch(psi):
    """Convert a pure state to Bloch sphere coordinates."""
    psi = psi.flatten()
    rho = np.outer(psi, psi.conj())
    x = np.real(np.trace(X @ rho))
    y = np.real(np.trace(Y @ rho))
    z = np.real(np.trace(Z @ rho))
    return np.array([x, y, z])

# Visualize Pauli gate actions on Bloch sphere
print("\n" + "=" * 60)
print("BLOCH SPHERE VISUALIZATION")
print("=" * 60)

fig = plt.figure(figsize=(15, 5))

# Create test states around the Bloch sphere
theta_vals = np.linspace(0, np.pi, 5)
phi_vals = np.linspace(0, 2*np.pi, 9)[:-1]

test_states = []
for theta in theta_vals:
    for phi in phi_vals:
        state = np.cos(theta/2) * ket_0 + np.exp(1j*phi) * np.sin(theta/2) * ket_1
        test_states.append(state)

# Add special states
special_states = [ket_0, ket_1, ket_plus, ket_minus]
test_states.extend(special_states)

# Plot original states and transformed states for each Pauli gate
gates = [('X gate', X), ('Y gate', Y), ('Z gate', Z)]

for idx, (gate_name, gate) in enumerate(gates):
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

    # Draw Bloch sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

    # Draw axes
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='r', alpha=0.5, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='g', alpha=0.5, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='b', alpha=0.5, arrow_length_ratio=0.1)
    ax.text(1.4, 0, 0, 'X', fontsize=10)
    ax.text(0, 1.4, 0, 'Y', fontsize=10)
    ax.text(0, 0, 1.4, 'Z', fontsize=10)

    # Plot state transformations
    for state in test_states:
        original_bloch = state_to_bloch(state)
        transformed_state = gate @ state
        transformed_bloch = state_to_bloch(transformed_state)

        # Draw arrow from original to transformed
        ax.scatter(*original_bloch, c='blue', s=20, alpha=0.5)
        ax.scatter(*transformed_bloch, c='red', s=20, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{gate_name}\nBlue=Original, Red=Transformed')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('pauli_gates_bloch.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: pauli_gates_bloch.png")

# Demonstrate Pauli decomposition of arbitrary matrix
print("\n" + "=" * 60)
print("PAULI DECOMPOSITION")
print("=" * 60)

def pauli_decomposition(M):
    """Decompose a 2x2 matrix into Pauli basis."""
    coeffs = {}
    for name, sigma in [('I', I), ('X', X), ('Y', Y), ('Z', Z)]:
        coeffs[name] = np.trace(sigma @ M) / 2
    return coeffs

# Example: decompose a Hermitian matrix
M = np.array([[2, 1-1j], [1+1j, 0]], dtype=complex)
print(f"\nMatrix M = \n{M}")

coeffs = pauli_decomposition(M)
print(f"\nPauli decomposition M = Σᵢ cᵢ σᵢ:")
for name, coeff in coeffs.items():
    if np.abs(coeff) > 1e-10:
        print(f"   c_{name} = {coeff}")

# Verify reconstruction
M_reconstructed = sum(coeff * sigma for (name, sigma), coeff
                      in zip([('I', I), ('X', X), ('Y', Y), ('Z', Z)], coeffs.values()))
print(f"\nReconstruction verified: {np.allclose(M, M_reconstructed)}")

# Pauli expectation values
print("\n" + "=" * 60)
print("PAULI EXPECTATION VALUES")
print("=" * 60)

def pauli_expectations(state):
    """Compute expectation values of Pauli operators."""
    state = state.flatten()
    rho = np.outer(state, state.conj())
    return {
        'X': np.real(np.trace(X @ rho)),
        'Y': np.real(np.trace(Y @ rho)),
        'Z': np.real(np.trace(Z @ rho))
    }

states_to_measure = [
    ('|0⟩', ket_0),
    ('|1⟩', ket_1),
    ('|+⟩', ket_plus),
    ('|-⟩', ket_minus),
    ('|+ᵢ⟩', (ket_0 + 1j*ket_1)/np.sqrt(2)),
    ('|-ᵢ⟩', (ket_0 - 1j*ket_1)/np.sqrt(2))
]

print("\nExpectation values ⟨σⱼ⟩ for various states:")
print("-" * 45)
print(f"{'State':<10} {'⟨X⟩':<10} {'⟨Y⟩':<10} {'⟨Z⟩':<10}")
print("-" * 45)
for name, state in states_to_measure:
    exp = pauli_expectations(state)
    print(f"{name:<10} {exp['X']:<10.4f} {exp['Y']:<10.4f} {exp['Z']:<10.4f}")

print("\nThese expectation values are the Bloch sphere coordinates!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Pauli X | $X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, bit flip |
| Pauli Y | $Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$, bit-phase flip |
| Pauli Z | $Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$, phase flip |
| Squared identity | $X^2 = Y^2 = Z^2 = I$ |
| Anti-commutation | $\{\sigma_j, \sigma_k\} = 2\delta_{jk}I$ |
| Commutation | $[\sigma_j, \sigma_k] = 2i\epsilon_{jkl}\sigma_l$ |
| Products | $XY = iZ$, $YZ = iX$, $ZX = iY$ |
| Eigenvalues | All Paulis have eigenvalues $\pm 1$ |

### Main Takeaways

1. **Pauli gates are fundamental:** They represent the basic errors and observables in quantum computing
2. **Geometric interpretation:** Each Pauli is a π rotation about its corresponding axis
3. **Algebraic structure:** Anti-commutation and squaring to identity are key properties
4. **Basis for decomposition:** Any 2×2 matrix can be written as a sum of Pauli matrices
5. **Eigenstructure:** All Paulis have eigenvalues ±1, but different eigenvectors

---

## Daily Checklist

- [ ] I can write all three Pauli matrices from memory
- [ ] I understand the action of each Pauli on computational basis states
- [ ] I can verify anti-commutation relations algebraically
- [ ] I understand the Bloch sphere interpretation as π rotations
- [ ] I can decompose a matrix in the Pauli basis
- [ ] I completed the computational lab and visualized gate actions
- [ ] I solved at least 3 practice problems

---

## Preview of Day 562

Tomorrow we study the **Hadamard gate**, the most important gate for creating superposition states. We'll see how H transforms between the computational (Z) basis and the X basis, and why H is essential for quantum parallelism. The key identity $H^2 = I$ will be explored along with the relationship $HXH = Z$.
