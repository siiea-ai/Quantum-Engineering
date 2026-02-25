# Week 157: Density Matrices - Comprehensive Review Guide

## 1. Introduction and Motivation

The density matrix formalism, introduced by von Neumann in 1927, provides the most general description of quantum states. While pure states represented by state vectors suffice for isolated quantum systems, realistic scenarios involving statistical uncertainty, entanglement with environments, or partial knowledge of system preparation require the density matrix framework.

### Why Density Matrices?

Consider these scenarios where state vectors are insufficient:

1. **Statistical Mixtures**: A source emits $$|0\rangle$$ with probability $$p$$ and $$|1\rangle$$ with probability $$1-p$$. This ensemble cannot be described by any single state vector.

2. **Subsystem Description**: Given a bipartite entangled state $$|\Psi_{AB}\rangle$$, how do we describe the state of subsystem $$A$$ alone?

3. **Decoherence**: An initially pure state interacting with an environment becomes mixed. The density matrix captures this loss of quantum coherence.

4. **Incomplete Information**: In experimental settings, we may only know expectation values of certain observables, not the complete quantum state.

The density matrix elegantly handles all these cases within a unified mathematical framework.

---

## 2. Pure States as Density Matrices

### 2.1 Definition

For a pure state $$|\psi\rangle$$, the density matrix (or density operator) is defined as the outer product:

$$\rho = |\psi\rangle\langle\psi|$$

This is a projection operator onto the one-dimensional subspace spanned by $$|\psi\rangle$$.

### 2.2 Matrix Representation

In a computational basis $$\{|0\rangle, |1\rangle, \ldots\}$$, if $$|\psi\rangle = \sum_i c_i |i\rangle$$, then:

$$\rho_{ij} = c_i c_j^*$$

**Example**: For $$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$:

$$\rho_{|+\rangle} = |+\rangle\langle+| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

### 2.3 Properties of Pure State Density Matrices

1. **Hermiticity**: $$\rho = \rho^\dagger$$ (follows from $$(|a\rangle\langle b|)^\dagger = |b\rangle\langle a|$$)

2. **Unit Trace**: $$\text{Tr}(\rho) = \langle\psi|\psi\rangle = 1$$

3. **Positivity**: $$\langle\phi|\rho|\phi\rangle = |\langle\phi|\psi\rangle|^2 \geq 0$$ for all $$|\phi\rangle$$

4. **Idempotency**: $$\rho^2 = \rho$$ (projector property)

5. **Rank One**: $$\rho$$ has exactly one non-zero eigenvalue (equal to 1)

### 2.4 Expectation Values

For an observable $$A$$, the expectation value in state $$\rho = |\psi\rangle\langle\psi|$$ is:

$$\langle A \rangle = \langle\psi|A|\psi\rangle = \text{Tr}(\rho A) = \text{Tr}(A\rho)$$

The trace formula extends naturally to mixed states and is the fundamental link between density matrices and measurement.

---

## 3. Mixed States

### 3.1 Statistical Ensembles

A mixed state arises from a probabilistic ensemble of pure states. If state $$|\psi_i\rangle$$ occurs with probability $$p_i$$, the density matrix is:

$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

where $$\sum_i p_i = 1$$ and $$p_i \geq 0$$.

### 3.2 Non-Uniqueness of Ensemble Decomposition

A crucial insight: **the same density matrix can arise from different ensembles**. This is fundamentally different from classical probability distributions.

**Example**: The maximally mixed qubit state $$\rho = \frac{1}{2}I$$ can be written as:

- Equal mixture of $$|0\rangle$$ and $$|1\rangle$$: $$\frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|$$
- Equal mixture of $$|+\rangle$$ and $$|-\rangle$$: $$\frac{1}{2}|+\rangle\langle+| + \frac{1}{2}|-\rangle\langle-|$$
- Equal mixture of $$|+i\rangle$$ and $$|-i\rangle$$

All ensembles give identical measurement statistics for any observable. This indistinguishability is captured by the density matrix.

### 3.3 Properties of Mixed States

All density matrices (pure or mixed) satisfy:

1. **Hermiticity**: $$\rho = \rho^\dagger$$
2. **Unit Trace**: $$\text{Tr}(\rho) = 1$$
3. **Positive Semi-definiteness**: $$\rho \geq 0$$ (all eigenvalues non-negative)

A matrix satisfying these three properties is a valid density matrix.

**Proof of positivity for mixed states**: For any $$|\phi\rangle$$:
$$\langle\phi|\rho|\phi\rangle = \sum_i p_i |\langle\phi|\psi_i\rangle|^2 \geq 0$$

---

## 4. Purity and Mixedness

### 4.1 Purity Definition

The **purity** of a state quantifies how close it is to a pure state:

$$\gamma = \text{Tr}(\rho^2)$$

**Properties**:
- For pure states: $$\gamma = 1$$ (since $$\rho^2 = \rho$$)
- For the maximally mixed state in dimension $$d$$: $$\gamma = 1/d$$
- General bounds: $$\frac{1}{d} \leq \gamma \leq 1$$

### 4.2 Linear Entropy

The **linear entropy** provides a measure of mixedness:

$$S_L(\rho) = 1 - \text{Tr}(\rho^2) = 1 - \gamma$$

This ranges from 0 (pure) to $$(d-1)/d$$ (maximally mixed).

### 4.3 Von Neumann Entropy

The von Neumann entropy is the quantum analog of Shannon entropy:

$$S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

where $$\{\lambda_i\}$$ are the eigenvalues of $$\rho$$, with convention $$0 \log 0 = 0$$.

**Properties**:
- $$S(\rho) = 0$$ iff $$\rho$$ is pure
- $$S(\rho) \leq \log_2 d$$, with equality for the maximally mixed state
- Concave: $$S(\sum_i p_i \rho_i) \geq \sum_i p_i S(\rho_i)$$

---

## 5. Bloch Sphere Representation

### 5.1 Qubit Density Matrix Parameterization

Any qubit density matrix can be written as:

$$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma}) = \frac{1}{2}\begin{pmatrix} 1 + r_z & r_x - ir_y \\ r_x + ir_y & 1 - r_z \end{pmatrix}$$

where $$\vec{r} = (r_x, r_y, r_z)$$ is the **Bloch vector** and $$\vec{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$$ are Pauli matrices.

### 5.2 Bloch Vector Constraints

From positivity of $$\rho$$:
- Eigenvalues: $$\lambda_\pm = \frac{1}{2}(1 \pm |\vec{r}|)$$
- Non-negativity requires: $$|\vec{r}| \leq 1$$

The Bloch ball (interior + boundary of unit sphere) represents all valid qubit states:
- **Surface** ($$|\vec{r}| = 1$$): Pure states
- **Interior** ($$|\vec{r}| < 1$$): Mixed states
- **Origin** ($$\vec{r} = 0$$): Maximally mixed state $$\rho = I/2$$

### 5.3 Computing the Bloch Vector

Given a density matrix $$\rho$$, the Bloch components are:

$$r_i = \text{Tr}(\rho \sigma_i), \quad i \in \{x, y, z\}$$

**Example**: For $$\rho = |0\rangle\langle 0| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$:
- $$r_x = \text{Tr}\left(\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\right) = 0$$
- $$r_y = \text{Tr}\left(\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}\right) = 0$$
- $$r_z = \text{Tr}\left(\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\right) = 1$$

So $$|0\rangle$$ corresponds to $$\vec{r} = (0, 0, 1)$$, the north pole.

### 5.4 Purity from Bloch Vector

$$\gamma = \text{Tr}(\rho^2) = \frac{1}{2}(1 + |\vec{r}|^2)$$

### 5.5 Unitary Evolution on Bloch Sphere

A unitary $$U = e^{-i\theta \hat{n}\cdot\vec{\sigma}/2}$$ rotates the Bloch vector by angle $$\theta$$ about axis $$\hat{n}$$:

$$\vec{r}' = R_{\hat{n}}(\theta)\vec{r}$$

where $$R_{\hat{n}}(\theta)$$ is the corresponding $$3 \times 3$$ rotation matrix.

---

## 6. Trace Operations

### 6.1 Trace Properties

The trace of an operator $$A$$ is:

$$\text{Tr}(A) = \sum_i \langle i|A|i\rangle$$

independent of the choice of orthonormal basis $$\{|i\rangle\}$$.

**Key Properties**:
1. **Linearity**: $$\text{Tr}(\alpha A + \beta B) = \alpha\text{Tr}(A) + \beta\text{Tr}(B)$$
2. **Cyclic**: $$\text{Tr}(ABC) = \text{Tr}(BCA) = \text{Tr}(CAB)$$
3. **Basis Independence**: $$\text{Tr}(UAU^\dagger) = \text{Tr}(A)$$ for unitary $$U$$
4. **Inner Product**: $$\text{Tr}(A^\dagger B)$$ defines a Hilbert-Schmidt inner product

### 6.2 Trace Distance

The trace distance between states $$\rho$$ and $$\sigma$$ is:

$$D(\rho, \sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma| = \frac{1}{2}\sum_i |\lambda_i|$$

where $$\{\lambda_i\}$$ are eigenvalues of $$\rho - \sigma$$ and $$|A| = \sqrt{A^\dagger A}$$.

**Properties**:
- $$0 \leq D(\rho, \sigma) \leq 1$$
- $$D(\rho, \sigma) = 0$$ iff $$\rho = \sigma$$
- Operational meaning: Maximum distinguishability by any measurement

For qubits: $$D(\rho, \sigma) = \frac{1}{2}|\vec{r}_\rho - \vec{r}_\sigma|$$

### 6.3 Fidelity

The fidelity between $$\rho$$ and $$\sigma$$ is:

$$F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2$$

**Special cases**:
- Both pure: $$F(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|^2$$
- One pure: $$F(\rho, |\psi\rangle) = \langle\psi|\rho|\psi\rangle$$

**Properties**:
- $$0 \leq F \leq 1$$
- $$F = 1$$ iff $$\rho = \sigma$$
- Symmetric: $$F(\rho, \sigma) = F(\sigma, \rho)$$

**Relation to trace distance** (Fuchs-van de Graaf):
$$1 - \sqrt{F(\rho, \sigma)} \leq D(\rho, \sigma) \leq \sqrt{1 - F(\rho, \sigma)}$$

---

## 7. Partial Trace

### 7.1 Motivation

For a composite system $$AB$$ in state $$\rho_{AB}$$, the partial trace gives the reduced density matrix of subsystem $$A$$:

$$\rho_A = \text{Tr}_B(\rho_{AB})$$

This describes all local observations on $$A$$ regardless of what happens to $$B$$.

### 7.2 Definition

For an orthonormal basis $$\{|j\rangle_B\}$$ of $$B$$:

$$\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B) \rho_{AB} (I_A \otimes |j\rangle_B)$$

Equivalently, for operators of the form $$A \otimes B$$:

$$\text{Tr}_B(A \otimes B) = A \cdot \text{Tr}(B)$$

and extend by linearity.

### 7.3 Computational Procedure

For a two-qubit state $$\rho_{AB}$$, written as a $$4 \times 4$$ matrix in the basis $$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$$:

$$\rho_{AB} = \begin{pmatrix} \rho_{00,00} & \rho_{00,01} & \rho_{00,10} & \rho_{00,11} \\ \rho_{01,00} & \rho_{01,01} & \rho_{01,10} & \rho_{01,11} \\ \rho_{10,00} & \rho_{10,01} & \rho_{10,10} & \rho_{10,11} \\ \rho_{11,00} & \rho_{11,01} & \rho_{11,10} & \rho_{11,11} \end{pmatrix}$$

Then:
$$\rho_A = \text{Tr}_B(\rho_{AB}) = \begin{pmatrix} \rho_{00,00} + \rho_{01,01} & \rho_{00,10} + \rho_{01,11} \\ \rho_{10,00} + \rho_{11,01} & \rho_{10,10} + \rho_{11,11} \end{pmatrix}$$

### 7.4 Example: Bell State

For the Bell state $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:

$$\rho_{AB} = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

$$\rho_A = \text{Tr}_B(\rho_{AB}) = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \frac{I}{2}$$

The reduced state is maximally mixed, reflecting the entanglement: complete knowledge of the composite system but complete ignorance of the subsystem.

---

## 8. Time Evolution of Density Matrices

### 8.1 Unitary Evolution

For closed system evolution under Hamiltonian $$H$$:

$$\rho(t) = U(t)\rho(0)U^\dagger(t), \quad U(t) = e^{-iHt/\hbar}$$

Differential form (von Neumann equation):

$$i\hbar\frac{d\rho}{dt} = [H, \rho]$$

Note the sign difference from the Heisenberg equation for operators.

### 8.2 Measurement

After measuring observable $$A = \sum_a a |a\rangle\langle a|$$ and obtaining result $$a$$:

$$\rho \to \rho' = \frac{|a\rangle\langle a|\rho|a\rangle\langle a|}{\langle a|\rho|a\rangle} = |a\rangle\langle a|$$

For non-selective measurement (averaging over outcomes):

$$\rho \to \rho' = \sum_a |a\rangle\langle a|\rho|a\rangle\langle a|$$

This is a completely positive trace-preserving map (quantum channel), previewing Week 160.

---

## 9. Advanced Topics

### 9.1 Spectral Decomposition

Every density matrix has a spectral decomposition:

$$\rho = \sum_i \lambda_i |i\rangle\langle i|$$

where $$\lambda_i \geq 0$$, $$\sum_i \lambda_i = 1$$, and $$\{|i\rangle\}$$ are orthonormal eigenvectors.

This is the unique decomposition with orthogonal states and is related to the ensemble interpretation.

### 9.2 Purification

Any mixed state $$\rho_A$$ on system $$A$$ can be expressed as the reduced state of a pure state on a larger system $$AB$$:

$$\rho_A = \text{Tr}_B(|\Psi\rangle_{AB}\langle\Psi|)$$

If $$\rho_A = \sum_i \lambda_i |i\rangle\langle i|$$, then:

$$|\Psi\rangle_{AB} = \sum_i \sqrt{\lambda_i}|i\rangle_A|i\rangle_B$$

is a purification. This connects to Schmidt decomposition (Week 158).

### 9.3 Convex Structure

The set of density matrices forms a convex set:
- Any convex combination of density matrices is a density matrix
- Pure states are the extremal points (cannot be written as non-trivial mixtures)
- The maximally mixed state is the unique center

---

## 10. Common Qualifying Exam Problem Types

1. **State Identification**: Given a density matrix, determine if pure or mixed, calculate purity
2. **Bloch Vector**: Convert between matrix and Bloch representation
3. **Trace Distance/Fidelity**: Compute distinguishability measures
4. **Partial Trace**: Find reduced density matrices of composite states
5. **Entropy Calculation**: Compute von Neumann entropy from eigenvalues
6. **Evolution**: Apply unitary or measurement to density matrix
7. **Purification**: Construct purification of a mixed state

---

## 11. Summary Table

| Concept | Pure State | Mixed State |
|---------|-----------|-------------|
| Definition | $$\rho = \|\psi\rangle\langle\psi\|$$ | $$\rho = \sum_i p_i\|\psi_i\rangle\langle\psi_i\|$$ |
| Purity | $$\text{Tr}(\rho^2) = 1$$ | $$\text{Tr}(\rho^2) < 1$$ |
| Entropy | $$S(\rho) = 0$$ | $$S(\rho) > 0$$ |
| Bloch (qubit) | $$\|\vec{r}\| = 1$$ | $$\|\vec{r}\| < 1$$ |
| Rank | 1 | > 1 |
| Idempotent | Yes: $$\rho^2 = \rho$$ | No |

---

## 12. Python Implementation

```python
import numpy as np
from scipy import linalg

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def density_matrix(state_vector):
    """Construct density matrix from state vector."""
    psi = np.array(state_vector).reshape(-1, 1)
    return psi @ psi.conj().T

def is_valid_density_matrix(rho, tol=1e-10):
    """Check if matrix is a valid density matrix."""
    hermitian = np.allclose(rho, rho.conj().T, atol=tol)
    trace_one = np.isclose(np.trace(rho), 1, atol=tol)
    eigenvalues = np.linalg.eigvalsh(rho)
    positive = np.all(eigenvalues >= -tol)
    return hermitian and trace_one and positive

def purity(rho):
    """Compute purity Tr(rho^2)."""
    return np.real(np.trace(rho @ rho))

def von_neumann_entropy(rho):
    """Compute von Neumann entropy S(rho)."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def bloch_vector(rho):
    """Extract Bloch vector from qubit density matrix."""
    rx = np.real(np.trace(rho @ X))
    ry = np.real(np.trace(rho @ Y))
    rz = np.real(np.trace(rho @ Z))
    return np.array([rx, ry, rz])

def rho_from_bloch(r):
    """Construct density matrix from Bloch vector."""
    return 0.5 * (I + r[0]*X + r[1]*Y + r[2]*Z)

def trace_distance(rho, sigma):
    """Compute trace distance D(rho, sigma)."""
    diff = rho - sigma
    eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
    return 0.5 * np.sum(np.sqrt(np.abs(eigenvalues)))

def fidelity(rho, sigma):
    """Compute fidelity F(rho, sigma)."""
    sqrt_rho = linalg.sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    return np.real(np.trace(linalg.sqrtm(inner)))**2

def partial_trace_B(rho_AB, dim_A=2, dim_B=2):
    """Compute partial trace over B for system AB."""
    rho_AB = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho_AB, axis1=1, axis2=3)

# Example usage
if __name__ == "__main__":
    # Pure state |+>
    plus = np.array([1, 1]) / np.sqrt(2)
    rho_plus = density_matrix(plus)
    print(f"Purity of |+>: {purity(rho_plus)}")
    print(f"Bloch vector of |+>: {bloch_vector(rho_plus)}")

    # Maximally mixed state
    rho_mm = I / 2
    print(f"Purity of I/2: {purity(rho_mm)}")
    print(f"Entropy of I/2: {von_neumann_entropy(rho_mm)}")
```

---

*This review guide covers the essential theory for density matrices. Work through the problem set to develop computational fluency and prepare for oral examination questions.*
