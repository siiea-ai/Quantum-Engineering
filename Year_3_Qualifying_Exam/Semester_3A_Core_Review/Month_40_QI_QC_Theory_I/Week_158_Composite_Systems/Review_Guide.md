# Week 158: Composite Systems - Comprehensive Review Guide

## 1. Introduction

Quantum information science fundamentally deals with composite quantum systems. Whether we're studying entanglement between particles, the interaction of a system with its environment, or implementing multi-qubit quantum algorithms, we need a rigorous mathematical framework for describing systems composed of multiple subsystems.

This week covers the essential mathematical structures: tensor products for constructing composite state spaces, partial traces for describing subsystems, Schmidt decomposition for analyzing bipartite entanglement, and purification for relating mixed states to pure states of larger systems.

---

## 2. Tensor Products of Hilbert Spaces

### 2.1 Definition

Given two Hilbert spaces $$\mathcal{H}_A$$ of dimension $$d_A$$ and $$\mathcal{H}_B$$ of dimension $$d_B$$, their tensor product $$\mathcal{H}_A \otimes \mathcal{H}_B$$ is a Hilbert space of dimension $$d_A \cdot d_B$$.

For orthonormal bases $$\{|i\rangle_A\}_{i=1}^{d_A}$$ and $$\{|j\rangle_B\}_{j=1}^{d_B}$$, the set $$\{|i\rangle_A \otimes |j\rangle_B\}$$ forms an orthonormal basis for $$\mathcal{H}_A \otimes \mathcal{H}_B$$.

### 2.2 Notation

We use several equivalent notations:
$$|i\rangle_A \otimes |j\rangle_B = |i\rangle|j\rangle = |i,j\rangle = |ij\rangle$$

For qubits:
$$|0\rangle \otimes |0\rangle = |00\rangle$$
$$|0\rangle \otimes |1\rangle = |01\rangle$$
$$|1\rangle \otimes |0\rangle = |10\rangle$$
$$|1\rangle \otimes |1\rangle = |11\rangle$$

### 2.3 General States

A general state in $$\mathcal{H}_A \otimes \mathcal{H}_B$$:
$$|\psi\rangle = \sum_{i,j} c_{ij} |i\rangle_A \otimes |j\rangle_B = \sum_{i,j} c_{ij}|ij\rangle$$

with normalization $$\sum_{i,j}|c_{ij}|^2 = 1$$.

**Product states** have the special form:
$$|\psi\rangle = |a\rangle_A \otimes |b\rangle_B$$

States that cannot be written as product states are **entangled**.

### 2.4 Properties of Tensor Products

**Bilinearity:**
$$(c_1|a_1\rangle + c_2|a_2\rangle) \otimes |b\rangle = c_1|a_1\rangle \otimes |b\rangle + c_2|a_2\rangle \otimes |b\rangle$$

**Inner product:**
$$(\langle a| \otimes \langle b|)(|a'\rangle \otimes |b'\rangle) = \langle a|a'\rangle \cdot \langle b|b'\rangle$$

**Non-commutativity:**
$$|a\rangle \otimes |b\rangle \neq |b\rangle \otimes |a\rangle$$ in general (different spaces!)

---

## 3. Tensor Products of Operators

### 3.1 Definition

For operators $$A$$ on $$\mathcal{H}_A$$ and $$B$$ on $$\mathcal{H}_B$$, their tensor product $$A \otimes B$$ acts on $$\mathcal{H}_A \otimes \mathcal{H}_B$$:

$$(A \otimes B)(|a\rangle \otimes |b\rangle) = (A|a\rangle) \otimes (B|b\rangle)$$

### 3.2 Kronecker Product (Matrix Representation)

For $$A$$ an $$m \times n$$ matrix and $$B$$ a $$p \times q$$ matrix:

$$A \otimes B = \begin{pmatrix}
a_{11}B & a_{12}B & \cdots & a_{1n}B \\
a_{21}B & a_{22}B & \cdots & a_{2n}B \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1}B & a_{m2}B & \cdots & a_{mn}B
\end{pmatrix}$$

Result is an $$mp \times nq$$ matrix.

**Example: CNOT gate**
$$\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$$

$$= \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} \otimes \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

$$= \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

### 3.3 Properties

**Mixed-product property:**
$$(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$$

**Adjoint:**
$$(A \otimes B)^\dagger = A^\dagger \otimes B^\dagger$$

**Trace:**
$$\text{Tr}(A \otimes B) = \text{Tr}(A) \cdot \text{Tr}(B)$$

**Eigenvalues:**
If $$A|a\rangle = \alpha|a\rangle$$ and $$B|b\rangle = \beta|b\rangle$$, then:
$$(A \otimes B)(|a\rangle \otimes |b\rangle) = \alpha\beta|a\rangle \otimes |b\rangle$$

### 3.4 Local vs. Global Operators

**Local operators** act on only one subsystem:
$$A_{\text{local}} = A \otimes I_B$$ or $$I_A \otimes B$$

**Global operators** may not decompose as simple tensor products.

Any operator on $$\mathcal{H}_A \otimes \mathcal{H}_B$$ can be written as:
$$O = \sum_{k} A_k \otimes B_k$$

(operator Schmidt decomposition)

---

## 4. Partial Trace and Reduced Density Matrices

### 4.1 Motivation

Given a composite system in state $$\rho_{AB}$$, how do we describe subsystem $$A$$ alone?

The answer is the **reduced density matrix**:
$$\rho_A = \text{Tr}_B(\rho_{AB})$$

This gives the correct expectation values for any observable on $$A$$:
$$\langle O_A \rangle = \text{Tr}_{AB}(\rho_{AB}(O_A \otimes I_B)) = \text{Tr}_A(\rho_A O_A)$$

### 4.2 Definition of Partial Trace

For an orthonormal basis $$\{|j\rangle_B\}$$ of $$\mathcal{H}_B$$:

$$\text{Tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B)\rho_{AB}(I_A \otimes |j\rangle_B)$$

**Key property for tensor products:**
$$\text{Tr}_B(A \otimes B) = A \cdot \text{Tr}(B)$$

Extended by linearity to all operators.

### 4.3 Computational Procedure

**Method 1: Direct sum over basis**

For $$|\psi\rangle = \sum_{ij} c_{ij}|ij\rangle$$, we have $$\rho = |\psi\rangle\langle\psi|$$.

$$\rho_A = \text{Tr}_B(\rho) = \sum_k \langle k|_B \rho |k\rangle_B$$

**Method 2: Matrix block formula**

For a two-qubit density matrix written as $$4 \times 4$$ blocks:
$$\rho_{AB} = \begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix}$$

where each $$\rho_{ij}$$ is a $$2 \times 2$$ block. Then:
$$\rho_A = \text{Tr}_B(\rho_{AB})$$ has elements obtained by taking $$2 \times 2$$ sub-block traces.

### 4.4 Examples

**Product state:**
$$\rho_{AB} = \rho_A \otimes \rho_B$$
$$\text{Tr}_B(\rho_A \otimes \rho_B) = \rho_A \cdot \text{Tr}(\rho_B) = \rho_A$$

**Bell state** $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:

$$\rho_{AB} = \frac{1}{2}(|00\rangle + |11\rangle)(\langle 00| + \langle 11|)$$
$$= \frac{1}{2}(|00\rangle\langle 00| + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|)$$

$$\rho_A = \frac{1}{2}(\langle 0|0\rangle|0\rangle\langle 0| + \langle 0|1\rangle|0\rangle\langle 1| + \langle 1|0\rangle|1\rangle\langle 0| + \langle 1|1\rangle|1\rangle\langle 1|)$$
$$= \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{I}{2}$$

The reduced state is maximally mixed despite the global state being pure!

---

## 5. Schmidt Decomposition

### 5.1 Theorem Statement

**Schmidt Decomposition Theorem:**
For any pure state $$|\psi\rangle \in \mathcal{H}_A \otimes \mathcal{H}_B$$ with $$d_A \leq d_B$$, there exist orthonormal bases $$\{|a_i\rangle\}$$ for $$\mathcal{H}_A$$ and $$\{|b_i\rangle\}$$ for (a subspace of) $$\mathcal{H}_B$$ such that:

$$|\psi\rangle = \sum_{i=1}^{r} \lambda_i |a_i\rangle |b_i\rangle$$

where:
- $$\lambda_i > 0$$ are the **Schmidt coefficients**
- $$\sum_i \lambda_i^2 = 1$$ (normalization)
- $$r \leq \min(d_A, d_B)$$ is the **Schmidt rank**

### 5.2 Proof via SVD

Write $$|\psi\rangle = \sum_{i,j} c_{ij}|i\rangle_A|j\rangle_B$$ with coefficient matrix $$C = (c_{ij})$$.

Apply singular value decomposition:
$$C = U \Sigma V^\dagger$$

where $$U$$ is $$d_A \times d_A$$ unitary, $$V$$ is $$d_B \times d_B$$ unitary, and $$\Sigma$$ is diagonal with non-negative entries $$\lambda_i$$.

Define new bases:
$$|a_i\rangle = \sum_k U_{ki}^* |k\rangle_A$$
$$|b_i\rangle = \sum_l V_{li}^* |l\rangle_B$$

Then:
$$|\psi\rangle = \sum_i \lambda_i |a_i\rangle|b_i\rangle$$

### 5.3 Properties

**Uniqueness:**
- Schmidt coefficients are unique (up to ordering)
- Schmidt bases are unique up to phases (when all $$\lambda_i$$ are distinct)
- For degenerate $$\lambda_i$$, there is additional unitary freedom

**Connection to reduced density matrices:**
$$\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|) = \sum_i \lambda_i^2 |a_i\rangle\langle a_i|$$
$$\rho_B = \text{Tr}_A(|\psi\rangle\langle\psi|) = \sum_i \lambda_i^2 |b_i\rangle\langle b_i|$$

Both reduced states have the same non-zero eigenvalues $$\{\lambda_i^2\}$$!

### 5.4 Schmidt Rank and Entanglement

**Schmidt rank = 1:** Product state (not entangled)
$$|\psi\rangle = |a\rangle|b\rangle$$

**Schmidt rank > 1:** Entangled state

**Maximum Schmidt rank = $$\min(d_A, d_B)$$:** Maximally entangled (e.g., Bell states for qubits)

### 5.5 Examples

**Product state** $$|+\rangle|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)|0\rangle$$:
Already in Schmidt form with $$r = 1$$, $$\lambda_1 = 1$$.

**Bell state** $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$:
Already in Schmidt form with $$r = 2$$, $$\lambda_1 = \lambda_2 = \frac{1}{\sqrt{2}}$$.

**General state** $$|\psi\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$:
This is $$|+\rangle|+\rangle$$, a product state. Schmidt rank = 1.

---

## 6. Finding Schmidt Decomposition

### 6.1 Method 1: Eigenvalue Decomposition

1. Compute the reduced density matrix $$\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$$
2. Diagonalize: $$\rho_A = \sum_i p_i |a_i\rangle\langle a_i|$$
3. Schmidt coefficients: $$\lambda_i = \sqrt{p_i}$$
4. Find $$|b_i\rangle$$ from: $$|b_i\rangle = \frac{1}{\lambda_i}(\langle a_i| \otimes I_B)|\psi\rangle$$

### 6.2 Method 2: SVD of Coefficient Matrix

1. Write $$|\psi\rangle = \sum_{ij} c_{ij}|ij\rangle$$
2. Form matrix $$C = (c_{ij})$$ with $$C_{ij} = c_{ij}$$
3. Compute SVD: $$C = U\Sigma V^\dagger$$
4. Schmidt coefficients: diagonal entries of $$\Sigma$$
5. Schmidt bases: columns of $$U$$ and $$V$$ define $$|a_i\rangle$$, $$|b_i\rangle$$

### 6.3 Method 3: By Inspection

For simple states, Schmidt form may be apparent:
- $$|00\rangle + |11\rangle$$: manifestly Schmidt with $$\lambda_1 = \lambda_2 = 1/\sqrt{2}$$
- $$|00\rangle + |01\rangle = |0\rangle(|0\rangle + |1\rangle)$$: product, Schmidt rank 1

---

## 7. Purification

### 7.1 Definition

A **purification** of a mixed state $$\rho_A$$ is a pure state $$|\Psi\rangle_{AB}$$ on an enlarged Hilbert space such that:
$$\rho_A = \text{Tr}_B(|\Psi\rangle_{AB}\langle\Psi|)$$

### 7.2 Existence and Construction

**Theorem:** Every mixed state has a purification.

**Construction:**
Given $$\rho_A = \sum_i p_i |i\rangle\langle i|$$ (spectral decomposition), define:
$$|\Psi\rangle_{AB} = \sum_i \sqrt{p_i}|i\rangle_A|i\rangle_B$$

where $$\{|i\rangle_B\}$$ is any orthonormal set in $$\mathcal{H}_B$$ with $$\dim(\mathcal{H}_B) \geq \text{rank}(\rho_A)$$.

**Verification:**
$$\text{Tr}_B(|\Psi\rangle\langle\Psi|) = \sum_{i,j}\sqrt{p_i p_j}|i\rangle\langle j| \cdot \langle j|i\rangle = \sum_i p_i|i\rangle\langle i| = \rho_A$$

### 7.3 Non-Uniqueness

Purifications are not unique. If $$|\Psi\rangle_{AB}$$ is a purification of $$\rho_A$$, then so is:
$$(I_A \otimes U_B)|\Psi\rangle_{AB}$$

for any unitary $$U_B$$ on $$\mathcal{H}_B$$.

**Theorem:** All purifications of $$\rho_A$$ are related by unitaries on the ancilla:
$$|\Psi'\rangle = (I_A \otimes U_B)|\Psi\rangle$$

### 7.4 Minimum Ancilla Size

The minimum dimension of $$\mathcal{H}_B$$ needed for purification equals $$\text{rank}(\rho_A)$$.

For a qubit, the ancilla can also be a qubit (rank $$\leq 2$$).

### 7.5 Applications

1. **Entanglement characterization:** Mixedness arises from entanglement with an inaccessible system

2. **Quantum channel representation:** Channels can be viewed as unitary evolution on purified systems followed by partial trace

3. **Proof techniques:** Many quantum information inequalities are proved using purification

4. **State preparation:** Purifications provide physical models for how mixed states might arise

---

## 8. Advanced Topics

### 8.1 Partial Transpose

For a bipartite state $$\rho_{AB}$$, the partial transpose over $$B$$ is:
$$\rho_{AB}^{T_B} = \sum_{ijkl} \rho_{ij,kl} |i\rangle\langle k| \otimes (|j\rangle\langle l|)^T$$
$$= \sum_{ijkl} \rho_{ij,kl} |i\rangle\langle k| \otimes |l\rangle\langle j|$$

**PPT criterion:** If $$\rho_{AB}^{T_B}$$ has negative eigenvalues, the state is entangled.
(For $$2 \times 2$$ and $$2 \times 3$$ systems, PPT is also sufficient for separability.)

### 8.2 Operator Schmidt Decomposition

Any operator $$O$$ on $$\mathcal{H}_A \otimes \mathcal{H}_B$$ can be written:
$$O = \sum_k c_k A_k \otimes B_k$$

where $$\{A_k\}$$ and $$\{B_k\}$$ are orthonormal operator bases.

The minimum number of terms is the **operator Schmidt rank**.

### 8.3 Multipartite Systems

For three or more subsystems, tensor products extend naturally:
$$\mathcal{H}_{ABC} = \mathcal{H}_A \otimes \mathcal{H}_B \otimes \mathcal{H}_C$$

However, multipartite entanglement is richer:
- No simple Schmidt decomposition exists
- Multiple inequivalent entanglement classes (GHZ, W states)
- Partial traces give various reduced states

---

## 9. Common Qualifying Exam Problem Types

1. **Tensor product calculations:** Compute $$|\psi\rangle \otimes |\phi\rangle$$ or $$A \otimes B$$

2. **Partial trace:** Find reduced density matrices

3. **Schmidt decomposition:** Find coefficients, rank, and bases

4. **Entanglement detection:** Determine if a state is entangled via Schmidt rank

5. **Purification:** Construct a purification for a given mixed state

6. **PPT criterion:** Apply partial transpose to detect entanglement

---

## 10. Summary Table

| Concept | Key Formula | Notes |
|---------|-------------|-------|
| Tensor product | $$\|ij\rangle = \|i\rangle \otimes \|j\rangle$$ | Dimension multiplies |
| Kronecker product | Block matrix form | $$d_A d_B \times d_A d_B$$ |
| Partial trace | $$\text{Tr}_B(A \otimes B) = A \cdot \text{Tr}(B)$$ | Extend by linearity |
| Schmidt decomposition | $$\|\psi\rangle = \sum_i \lambda_i\|a_i\rangle\|b_i\rangle$$ | Via SVD |
| Schmidt rank | $$r = $$ # nonzero $$\lambda_i$$ | $$r=1$$: product state |
| Reduced state | $$\rho_A = \sum_i \lambda_i^2 \|a_i\rangle\langle a_i\|$$ | Same eigenvalues for A, B |
| Purification | $$\|\Psi\rangle = \sum_i \sqrt{p_i}\|i\rangle\|i\rangle$$ | Non-unique |

---

## 11. Python Implementation

```python
import numpy as np
from scipy import linalg

def tensor_product(A, B):
    """Compute tensor (Kronecker) product of two matrices."""
    return np.kron(A, B)

def partial_trace_B(rho_AB, dim_A, dim_B):
    """
    Compute partial trace over subsystem B.
    rho_AB: density matrix of composite system (dim_A*dim_B x dim_A*dim_B)
    """
    rho_AB = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho_AB, axis1=1, axis2=3)

def partial_trace_A(rho_AB, dim_A, dim_B):
    """Compute partial trace over subsystem A."""
    rho_AB = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho_AB, axis1=0, axis2=2)

def schmidt_decomposition(psi, dim_A, dim_B):
    """
    Compute Schmidt decomposition of pure state.
    psi: state vector (dim_A*dim_B,)
    Returns: coefficients, basis_A, basis_B
    """
    # Reshape to coefficient matrix
    C = psi.reshape(dim_A, dim_B)

    # SVD
    U, S, Vh = np.linalg.svd(C, full_matrices=False)

    # Filter out zero coefficients
    tol = 1e-10
    nonzero = S > tol

    coefficients = S[nonzero]
    basis_A = U[:, nonzero]  # columns are |a_i>
    basis_B = Vh[nonzero, :].conj().T  # columns are |b_i>

    return coefficients, basis_A, basis_B

def purification(rho, dim_ancilla=None):
    """
    Construct a purification of density matrix rho.
    Returns pure state vector on extended system.
    """
    dim = rho.shape[0]

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(rho)

    # Filter positive eigenvalues
    tol = 1e-10
    pos = eigenvalues > tol
    p = eigenvalues[pos]
    V = eigenvectors[:, pos]
    rank = len(p)

    if dim_ancilla is None:
        dim_ancilla = rank

    # Construct purification |Psi> = sum_i sqrt(p_i) |i>_A |i>_B
    psi = np.zeros(dim * dim_ancilla, dtype=complex)
    for i in range(rank):
        # |i>_A is column i of V
        # |i>_B is computational basis state
        for j in range(dim):
            psi[j * dim_ancilla + i] = np.sqrt(p[i]) * V[j, i]

    return psi

def is_product_state(psi, dim_A, dim_B, tol=1e-10):
    """Check if pure state is a product state (Schmidt rank 1)."""
    coeffs, _, _ = schmidt_decomposition(psi, dim_A, dim_B)
    return len(coeffs) == 1 or (len(coeffs) > 1 and coeffs[1] < tol)

# Example: Bell state analysis
if __name__ == "__main__":
    # |Phi+> = (|00> + |11>)/sqrt(2)
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

    # Schmidt decomposition
    coeffs, A, B = schmidt_decomposition(phi_plus, 2, 2)
    print("Schmidt coefficients:", coeffs)
    print("Schmidt rank:", len(coeffs))

    # Reduced density matrix
    rho_AB = np.outer(phi_plus, phi_plus.conj())
    rho_A = partial_trace_B(rho_AB, 2, 2)
    print("Reduced density matrix rho_A:\n", rho_A)

    # Verify eigenvalues match Schmidt coefficients squared
    eigenvalues = np.linalg.eigvalsh(rho_A)
    print("Eigenvalues of rho_A:", eigenvalues)
    print("Schmidt coefficients squared:", coeffs**2)
```

---

*This review guide covers the essential theory for composite quantum systems. Master tensor products, partial traces, and Schmidt decomposition before proceeding to entanglement measures.*
