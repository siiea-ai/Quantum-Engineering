# Day 505: Density Operator Definition

## Week 73: Pure vs Mixed States | Month 19: Density Matrices

---

## Schedule Overview

| Session | Time | Topic |
|---------|------|-------|
| **Morning** | 3 hours | Pure states as projectors, ensemble interpretation |
| **Afternoon** | 2.5 hours | Problem solving with density operators |
| **Evening** | 1.5 hours | Computational lab: constructing density matrices |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define** the density operator for a pure quantum state as an outer product
2. **Construct** density matrices for statistical mixtures of quantum states
3. **Distinguish** between pure states and mixed states mathematically
4. **Interpret** the ensemble picture and understand its limitations
5. **Express** density matrices in different bases
6. **Connect** the density matrix formalism to quantum computing applications

---

## Core Content

### 1. From State Vectors to Density Operators

In standard quantum mechanics, we describe a system by a state vector $|\psi\rangle$ in a Hilbert space $\mathcal{H}$. However, this description has fundamental limitations:

**Limitation 1: Classical Uncertainty**
What if we don't know which quantum state the system is in? For example, a source might emit $|0\rangle$ with probability $p$ and $|1\rangle$ with probability $1-p$.

**Limitation 2: Subsystem Description**
What if our system is entangled with another system we cannot access? The subsystem alone cannot be described by a state vector.

**Solution: The Density Operator**

The density operator (or density matrix) provides a complete description of quantum systems that handles both situations elegantly.

### 2. Pure States as Projectors

For a pure state $|\psi\rangle$, the density operator is defined as:

$$\boxed{\rho = |\psi\rangle\langle\psi|}$$

This is the **outer product** or **projector** onto the state $|\psi\rangle$.

**Example: Single Qubit States**

For $|\psi\rangle = |0\rangle$:
$$\rho = |0\rangle\langle 0| = \begin{pmatrix} 1 \\ 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

For $|\psi\rangle = |1\rangle$:
$$\rho = |1\rangle\langle 1| = \begin{pmatrix} 0 \\ 1 \end{pmatrix}\begin{pmatrix} 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

For $|\psi\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:
$$\rho = |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix} 1 \\ 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**Key Observation**: The off-diagonal elements (coherences) contain information about superposition. For $|+\rangle$, these are non-zero, reflecting the quantum superposition.

### 3. General Pure State in Two Dimensions

Consider a general qubit state:
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1$$

The density matrix is:
$$\rho = |\psi\rangle\langle\psi| = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}\begin{pmatrix} \alpha^* & \beta^* \end{pmatrix} = \begin{pmatrix} |\alpha|^2 & \alpha\beta^* \\ \alpha^*\beta & |\beta|^2 \end{pmatrix}$$

**Matrix Elements**:
- **Diagonal elements** $\rho_{00} = |\alpha|^2$, $\rho_{11} = |\beta|^2$: Populations (probabilities)
- **Off-diagonal elements** $\rho_{01} = \alpha\beta^*$, $\rho_{10} = \alpha^*\beta$: Coherences

### 4. Mixed States: The Ensemble Interpretation

A **mixed state** arises when we have classical uncertainty about which quantum state the system is in. If the system is in state $|\psi_i\rangle$ with probability $p_i$, the density operator is:

$$\boxed{\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|}$$

where $p_i \geq 0$ and $\sum_i p_i = 1$.

**Physical Interpretation**: Imagine an ensemble of identically prepared systems. A fraction $p_i$ of systems are in state $|\psi_i\rangle$. The density matrix describes our knowledge of a randomly selected system.

**Example: Completely Random Qubit**

If $|0\rangle$ and $|1\rangle$ occur with equal probability:
$$\rho = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + \frac{1}{2}\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \frac{I}{2}$$

This is the **maximally mixed state**—we have no information about the qubit.

### 5. Pure vs Mixed: A Crucial Distinction

**Critical Example**: Consider these two situations:

**Situation A** (Pure State):
$$|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |+\rangle$$
$$\rho_A = |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**Situation B** (Mixed State):
50% probability of $|0\rangle$, 50% probability of $|1\rangle$
$$\rho_B = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

**Both have the same diagonal elements!** If we measure in the computational basis:
- Both give 50% $|0\rangle$, 50% $|1\rangle$

**But they differ in the off-diagonals (coherences)!**

If we measure in the $\{|+\rangle, |-\rangle\}$ basis:
- $\rho_A$: Always gives $|+\rangle$ (100%)
- $\rho_B$: Gives 50% $|+\rangle$, 50% $|-\rangle$

**The coherences encode the ability to interfere!**

### 6. Non-Uniqueness of Ensemble Decomposition

A crucial subtlety: the same density matrix can arise from different ensembles.

**Example**: Consider the maximally mixed state $\rho = I/2$.

**Decomposition 1**:
$$\rho = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|$$

**Decomposition 2**:
$$\rho = \frac{1}{2}|+\rangle\langle +| + \frac{1}{2}|-\rangle\langle -|$$

Both give the same $\rho = I/2$! This means:
- The density matrix captures all physically observable information
- We cannot determine which ensemble "actually" prepared the state
- This is a feature, not a bug: different preparations with identical predictions are physically indistinguishable

### 7. Spectral Decomposition

Every density matrix has a unique **spectral decomposition**:
$$\rho = \sum_k \lambda_k |k\rangle\langle k|$$

where $|k\rangle$ are orthonormal eigenvectors and $\lambda_k \geq 0$ are eigenvalues with $\sum_k \lambda_k = 1$.

**For pure states**: One eigenvalue is 1, all others are 0
$$\rho_{\text{pure}} = 1 \cdot |\psi\rangle\langle\psi| + 0 \cdot (\text{other terms})$$

**For mixed states**: Multiple non-zero eigenvalues

---

## Quantum Computing Connection

### Noise and Decoherence

In real quantum computers, pure states evolve into mixed states due to **decoherence**:

**Ideal**: Qubit in superposition $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$
$$\rho_{\text{ideal}} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**After dephasing** (T2 decay): Coherences decay
$$\rho_{\text{dephased}} = \frac{1}{2}\begin{pmatrix} 1 & e^{-t/T_2} \\ e^{-t/T_2} & 1 \end{pmatrix} \xrightarrow{t \to \infty} \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

### Quantum State Tomography

To experimentally determine $\rho$, we perform **state tomography**:
1. Prepare many copies of the state
2. Measure in different bases
3. Reconstruct the full density matrix

For a single qubit, we need measurements in 3 bases (X, Y, Z) to determine all 4 elements (with constraints).

### Quantum Error Characterization

The density matrix is essential for:
- **Benchmarking**: Comparing ideal vs actual states
- **Error mitigation**: Modeling and correcting noise
- **Certification**: Verifying quantum operations

---

## Worked Examples

### Example 1: Constructing a Density Matrix

**Problem**: A quantum source produces $|\psi_1\rangle = |0\rangle$ with probability 0.7 and $|\psi_2\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ with probability 0.3. Find the density matrix.

**Solution**:

Step 1: Write out the individual density matrices
$$\rho_1 = |0\rangle\langle 0| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

$$\rho_2 = |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

Step 2: Combine with probabilities
$$\rho = 0.7 \cdot \rho_1 + 0.3 \cdot \rho_2$$

$$\rho = 0.7 \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} + 0.3 \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

$$\rho = \begin{pmatrix} 0.7 & 0 \\ 0 & 0 \end{pmatrix} + \begin{pmatrix} 0.15 & 0.15 \\ 0.15 & 0.15 \end{pmatrix}$$

$$\boxed{\rho = \begin{pmatrix} 0.85 & 0.15 \\ 0.15 & 0.15 \end{pmatrix}}$$

Step 3: Verify trace = 1
$$\text{Tr}(\rho) = 0.85 + 0.15 = 1 \checkmark$$

---

### Example 2: Expressing a Density Matrix in a Different Basis

**Problem**: Given $\rho = |+\rangle\langle +|$ in the computational basis, express it in the $\{|+\rangle, |-\rangle\}$ basis.

**Solution**:

Step 1: In computational basis
$$\rho = |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

Step 2: Define basis transformation
$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle), \quad |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

The transformation matrix (Hadamard):
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Step 3: Transform the density matrix
$$\rho' = H^\dagger \rho H = H \rho H$$

$$\rho' = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

First multiplication:
$$H \cdot \rho = \frac{1}{2\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \frac{1}{2\sqrt{2}}\begin{pmatrix} 2 & 2 \\ 0 & 0 \end{pmatrix}$$

Second multiplication:
$$\rho' = \frac{1}{2\sqrt{2}}\begin{pmatrix} 2 & 2 \\ 0 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{4}\begin{pmatrix} 4 & 0 \\ 0 & 0 \end{pmatrix}$$

$$\boxed{\rho' = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}}$$

This confirms $|+\rangle$ is the pure state $|+\rangle$ in the $\{|+\rangle, |-\rangle\}$ basis!

---

### Example 3: Verifying a Matrix is a Valid Density Matrix

**Problem**: Is $\rho = \begin{pmatrix} 0.6 & 0.4 \\ 0.4 & 0.4 \end{pmatrix}$ a valid density matrix?

**Solution**:

Check the three required properties:

**Property 1: Hermiticity** ($\rho = \rho^\dagger$)
$$\rho^\dagger = \begin{pmatrix} 0.6 & 0.4 \\ 0.4 & 0.4 \end{pmatrix} = \rho \checkmark$$

**Property 2: Unit Trace**
$$\text{Tr}(\rho) = 0.6 + 0.4 = 1 \checkmark$$

**Property 3: Positive Semi-definiteness**
Find eigenvalues: $\det(\rho - \lambda I) = 0$
$$(0.6 - \lambda)(0.4 - \lambda) - 0.16 = 0$$
$$\lambda^2 - \lambda + 0.24 - 0.16 = 0$$
$$\lambda^2 - \lambda + 0.08 = 0$$
$$\lambda = \frac{1 \pm \sqrt{1 - 0.32}}{2} = \frac{1 \pm \sqrt{0.68}}{2}$$
$$\lambda_1 \approx 0.912, \quad \lambda_2 \approx 0.088$$

Both eigenvalues are positive! $\checkmark$

$$\boxed{\text{Yes, } \rho \text{ is a valid density matrix (mixed state)}}$$

---

## Practice Problems

### Direct Application

1. **Problem 1**: Compute the density matrix for $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$.

2. **Problem 2**: A source produces $|0\rangle$ with probability 0.25, $|1\rangle$ with probability 0.25, and $|+\rangle$ with probability 0.5. Find $\rho$.

3. **Problem 3**: Verify that $\rho = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$ satisfies all three density matrix properties.

### Intermediate

4. **Problem 4**: Show that any $2 \times 2$ density matrix can be written as $\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$ for some real vector $\vec{r}$ with $|\vec{r}| \leq 1$.

5. **Problem 5**: Given the mixed state $\rho = 0.8|0\rangle\langle 0| + 0.2|1\rangle\langle 1|$, find a different ensemble that produces the same density matrix.

6. **Problem 6**: If $|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$, express the density matrix elements in terms of $\theta$ and $\phi$.

### Challenging

7. **Problem 7**: Prove that for any density matrix $\rho$, we have $\text{Tr}(\rho^2) \leq 1$, with equality if and only if $\rho$ is pure.

8. **Problem 8**: Consider a two-qubit state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$. Show that the reduced density matrix for either qubit is the maximally mixed state $I/2$.

9. **Problem 9**: A qubit undergoes amplitude damping with decay probability $\gamma$. If the initial state is $|+\rangle$, find the final density matrix after the noise process.

---

## Computational Lab

```python
"""
Day 505 Computational Lab: Constructing Density Matrices
=======================================================
Topics: Pure states, mixed states, ensemble construction
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Define standard states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def ket_to_density(ket: np.ndarray) -> np.ndarray:
    """
    Convert a state vector to a density matrix.

    For pure state |ψ⟩, the density matrix is ρ = |ψ⟩⟨ψ|

    Parameters:
        ket: Column vector representing |ψ⟩

    Returns:
        Density matrix ρ = |ψ⟩⟨ψ|
    """
    return ket @ ket.conj().T


def ensemble_to_density(states: List[np.ndarray],
                        probabilities: List[float]) -> np.ndarray:
    """
    Construct density matrix from an ensemble of states.

    ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

    Parameters:
        states: List of state vectors
        probabilities: List of probabilities (must sum to 1)

    Returns:
        Mixed state density matrix
    """
    assert abs(sum(probabilities) - 1.0) < 1e-10, "Probabilities must sum to 1"
    assert all(p >= 0 for p in probabilities), "Probabilities must be non-negative"

    d = states[0].shape[0]
    rho = np.zeros((d, d), dtype=complex)

    for state, prob in zip(states, probabilities):
        rho += prob * ket_to_density(state)

    return rho


def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> dict:
    """
    Check if a matrix is a valid density matrix.

    Requirements:
    1. Hermiticity: ρ = ρ†
    2. Unit trace: Tr(ρ) = 1
    3. Positive semi-definite: all eigenvalues ≥ 0

    Parameters:
        rho: Matrix to check
        tol: Numerical tolerance

    Returns:
        Dictionary with verification results
    """
    results = {}

    # Check Hermiticity
    hermitian_diff = np.max(np.abs(rho - rho.conj().T))
    results['hermitian'] = hermitian_diff < tol
    results['hermitian_error'] = hermitian_diff

    # Check trace
    trace = np.trace(rho)
    results['unit_trace'] = abs(trace - 1) < tol
    results['trace_value'] = trace

    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(rho)
    results['positive'] = all(eigenvalues > -tol)
    results['eigenvalues'] = eigenvalues
    results['min_eigenvalue'] = min(eigenvalues)

    results['is_valid'] = (results['hermitian'] and
                          results['unit_trace'] and
                          results['positive'])

    return results


def display_density_matrix(rho: np.ndarray, name: str = "ρ"):
    """Pretty print a density matrix."""
    print(f"\n{name} =")
    print(np.array2string(rho, precision=4, suppress_small=True))
    print(f"Trace({name}) = {np.trace(rho).real:.6f}")

    # Check if pure or mixed
    purity = np.trace(rho @ rho).real
    print(f"Purity Tr({name}²) = {purity:.6f}")
    print(f"State type: {'Pure' if abs(purity - 1) < 1e-10 else 'Mixed'}")


# ===== DEMONSTRATIONS =====

print("=" * 60)
print("PART 1: Pure State Density Matrices")
print("=" * 60)

# Example 1: Computational basis states
rho_0 = ket_to_density(ket_0)
rho_1 = ket_to_density(ket_1)

display_density_matrix(rho_0, "ρ(|0⟩)")
display_density_matrix(rho_1, "ρ(|1⟩)")

# Example 2: Superposition state |+⟩
rho_plus = ket_to_density(ket_plus)
display_density_matrix(rho_plus, "ρ(|+⟩)")

# Example 3: General pure state with phase
theta, phi = np.pi/3, np.pi/4
ket_general = np.array([[np.cos(theta/2)],
                        [np.exp(1j*phi) * np.sin(theta/2)]])
rho_general = ket_to_density(ket_general)
display_density_matrix(rho_general, "ρ(θ=π/3, φ=π/4)")


print("\n" + "=" * 60)
print("PART 2: Mixed State Density Matrices")
print("=" * 60)

# Example 4: 50-50 mixture of |0⟩ and |1⟩
rho_mixed_01 = ensemble_to_density([ket_0, ket_1], [0.5, 0.5])
display_density_matrix(rho_mixed_01, "ρ(50% |0⟩ + 50% |1⟩)")

# Example 5: Mixture of |0⟩ and |+⟩
rho_mixed_0plus = ensemble_to_density([ket_0, ket_plus], [0.7, 0.3])
display_density_matrix(rho_mixed_0plus, "ρ(70% |0⟩ + 30% |+⟩)")

# Example 6: Three-state mixture
ket_y_plus = (ket_0 + 1j * ket_1) / np.sqrt(2)
rho_three = ensemble_to_density([ket_0, ket_plus, ket_y_plus], [0.5, 0.3, 0.2])
display_density_matrix(rho_three, "ρ(three-state mixture)")


print("\n" + "=" * 60)
print("PART 3: Pure vs Mixed - The Crucial Difference")
print("=" * 60)

print("\nCompare: |+⟩ pure state vs 50-50 mixture of |0⟩ and |1⟩")
print("\nPure state |+⟩:")
display_density_matrix(rho_plus, "ρ_pure")

print("\nMixed state (classical mixture):")
display_density_matrix(rho_mixed_01, "ρ_mixed")

print("\nKey difference: Off-diagonal elements (coherences)")
print(f"ρ_pure[0,1] = {rho_plus[0,1]:.4f}  (non-zero coherence)")
print(f"ρ_mixed[0,1] = {rho_mixed_01[0,1]:.4f}  (no coherence)")


print("\n" + "=" * 60)
print("PART 4: Verifying Density Matrix Properties")
print("=" * 60)

# Test a valid density matrix
print("\nTesting ρ_mixed_0plus:")
results = is_valid_density_matrix(rho_mixed_0plus)
for key, value in results.items():
    print(f"  {key}: {value}")

# Test an invalid matrix (not positive semi-definite)
print("\nTesting an invalid matrix:")
invalid_rho = np.array([[0.5, 0.6], [0.6, 0.5]])
results_invalid = is_valid_density_matrix(invalid_rho)
for key, value in results_invalid.items():
    print(f"  {key}: {value}")


print("\n" + "=" * 60)
print("PART 5: Spectral Decomposition")
print("=" * 60)

def spectral_decomposition(rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectral decomposition of a density matrix.

    ρ = Σₖ λₖ |k⟩⟨k|

    Returns eigenvalues and eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]

# Spectral decomposition of mixed state
eigenvals, eigenvecs = spectral_decomposition(rho_mixed_0plus)
print(f"\nSpectral decomposition of ρ(70% |0⟩ + 30% |+⟩):")
print(f"Eigenvalues: {eigenvals}")
print(f"\nEigenvector 1 (λ₁={eigenvals[0]:.4f}):")
print(eigenvecs[:, 0])
print(f"\nEigenvector 2 (λ₂={eigenvals[1]:.4f}):")
print(eigenvecs[:, 1])

# Verify reconstruction
rho_reconstructed = sum(eigenvals[i] * np.outer(eigenvecs[:, i], eigenvecs[:, i].conj())
                        for i in range(len(eigenvals)))
print(f"\nReconstruction error: {np.max(np.abs(rho_mixed_0plus - rho_reconstructed)):.2e}")


print("\n" + "=" * 60)
print("PART 6: Visualization of Density Matrix Elements")
print("=" * 60)

def visualize_density_matrix(rho: np.ndarray, title: str = "Density Matrix"):
    """
    Visualize the magnitude and phase of density matrix elements.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Magnitude plot
    im1 = axes[0].imshow(np.abs(rho), cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title(f'{title}\nMagnitude |ρᵢⱼ|')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['|0⟩', '|1⟩'])
    axes[0].set_yticklabels(['⟨0|', '⟨1|'])
    plt.colorbar(im1, ax=axes[0])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{np.abs(rho[i,j]):.3f}',
                        ha='center', va='center', fontsize=12)

    # Phase plot
    phases = np.angle(rho)
    im2 = axes[1].imshow(phases, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f'{title}\nPhase arg(ρᵢⱼ)')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['|0⟩', '|1⟩'])
    axes[1].set_yticklabels(['⟨0|', '⟨1|'])
    plt.colorbar(im2, ax=axes[1], label='Phase (rad)')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            if np.abs(rho[i,j]) > 1e-10:
                axes[1].text(j, i, f'{phases[i,j]:.2f}',
                            ha='center', va='center', fontsize=12, color='white')

    plt.tight_layout()
    plt.savefig('density_matrix_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: density_matrix_visualization.png")

# Visualize the general pure state
visualize_density_matrix(rho_general, "Pure State (θ=π/3, φ=π/4)")


print("\n" + "=" * 60)
print("PART 7: Ensemble Non-Uniqueness Demonstration")
print("=" * 60)

# The maximally mixed state I/2 from different ensembles
rho_mm_1 = ensemble_to_density([ket_0, ket_1], [0.5, 0.5])
rho_mm_2 = ensemble_to_density([ket_plus, ket_minus], [0.5, 0.5])

ket_y_minus = (ket_0 - 1j * ket_1) / np.sqrt(2)
rho_mm_3 = ensemble_to_density([ket_y_plus, ket_y_minus], [0.5, 0.5])

print("Three different ensembles producing I/2:")
print("\nEnsemble 1: 50% |0⟩ + 50% |1⟩")
display_density_matrix(rho_mm_1, "ρ₁")

print("\nEnsemble 2: 50% |+⟩ + 50% |-⟩")
display_density_matrix(rho_mm_2, "ρ₂")

print("\nEnsemble 3: 50% |+i⟩ + 50% |-i⟩")
display_density_matrix(rho_mm_3, "ρ₃")

print(f"\nAll three are identical: {np.allclose(rho_mm_1, rho_mm_2) and np.allclose(rho_mm_2, rho_mm_3)}")


print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Pure state density matrix | $\rho = \|\psi\rangle\langle\psi\|$ |
| Mixed state density matrix | $\rho = \sum_i p_i \|\psi_i\rangle\langle\psi_i\|$ |
| Matrix elements | $\rho_{ij} = \langle i\|\rho\|j\rangle$ |
| Diagonal elements | Populations (probabilities) |
| Off-diagonal elements | Coherences (interference) |

### Main Takeaways

1. **Density operators** generalize state vectors to handle classical uncertainty
2. **Pure states** are projectors $\rho = |\psi\rangle\langle\psi|$ with a single non-zero eigenvalue
3. **Mixed states** are convex combinations of pure state projectors
4. **Coherences** (off-diagonal elements) distinguish pure superpositions from classical mixtures
5. **Ensemble decompositions are not unique** - the same $\rho$ can arise from different preparations
6. The density matrix captures **all physically measurable information** about a quantum system

---

## Daily Checklist

- [ ] I can construct density matrices for pure states using outer products
- [ ] I understand the ensemble interpretation of mixed states
- [ ] I can distinguish pure from mixed states by examining the density matrix
- [ ] I understand why coherences are physically significant
- [ ] I recognize that ensemble decompositions are not unique
- [ ] I completed the computational lab exercises
- [ ] I attempted at least 3 practice problems

---

## Preview: Day 506

Tomorrow we will study the **three essential properties** of density matrices in detail:
- **Hermiticity**: $\rho = \rho^\dagger$
- **Positivity**: $\rho \geq 0$ (all eigenvalues non-negative)
- **Normalization**: $\text{Tr}(\rho) = 1$

We'll prove these properties, understand their physical meaning, and develop powerful techniques using the trace operation.

---

*"The density matrix is the most general way to describe a quantum system, encompassing both our quantum and classical ignorance."* — John Preskill
