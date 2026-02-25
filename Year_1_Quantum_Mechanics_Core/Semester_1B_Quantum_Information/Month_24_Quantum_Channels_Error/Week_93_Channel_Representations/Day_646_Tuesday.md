# Day 646: Choi-Jamiolkowski Isomorphism

## Week 93: Channel Representations | Month 24: Quantum Channels & Error Introduction

---

## Schedule Overview

| Session | Time | Topic |
|---------|------|-------|
| **Morning** | 3 hours | Channel-state duality, Choi matrix construction |
| **Afternoon** | 2.5 hours | Problem solving and applications |
| **Evening** | 1.5 hours | Computational lab: computing and analyzing Choi matrices |

---

## Learning Objectives

By the end of today, you will be able to:

1. **State** the Choi-Jamiolkowski isomorphism and its significance
2. **Construct** the Choi matrix for any quantum channel
3. **Determine** complete positivity from the Choi matrix
4. **Extract** Kraus operators from the Choi matrix via eigendecomposition
5. **Calculate** the Kraus rank as the rank of the Choi matrix
6. **Apply** the isomorphism to analyze channel properties

---

## Core Content

### 1. The Channel-State Duality

One of the most beautiful results in quantum information theory is the **Choi-Jamiolkowski isomorphism**: a one-to-one correspondence between quantum channels and quantum states (on a larger Hilbert space).

**The Key Idea:**
- A channel $\mathcal{E}: \mathcal{B}(\mathcal{H}_A) \to \mathcal{B}(\mathcal{H}_B)$ is completely characterized by how it acts on one half of a maximally entangled state.

### 2. The Maximally Entangled State

For a $d$-dimensional system, define the **maximally entangled state**:

$$|\Phi^+\rangle = \frac{1}{\sqrt{d}} \sum_{i=0}^{d-1} |i\rangle_A \otimes |i\rangle_{A'}$$

For qubits ($d=2$):
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

This is one of the Bell states, with maximum entanglement.

### 3. The Choi Matrix Definition

**Definition:** The **Choi matrix** (or Choi state) of a channel $\mathcal{E}$ is:

$$\boxed{J_\mathcal{E} = (\mathcal{I}_A \otimes \mathcal{E}_{A'})|\Phi^+\rangle\langle\Phi^+|}$$

where:
- $\mathcal{I}_A$ is the identity channel on system $A$
- $\mathcal{E}_{A'}$ acts on the second subsystem $A'$

**Alternative Expression:**

$$J_\mathcal{E} = \sum_{i,j=0}^{d-1} |i\rangle\langle j|_A \otimes \mathcal{E}(|i\rangle\langle j|_{A'})$$

This shows $J_\mathcal{E}$ encodes the action of $\mathcal{E}$ on all basis elements $|i\rangle\langle j|$.

### 4. The Choi-Jamiolkowski Theorem

**Theorem (Choi, 1975; Jamiolkowski, 1972):**

There is a one-to-one correspondence between:
- Linear maps $\mathcal{E}: \mathcal{B}(\mathcal{H}) \to \mathcal{B}(\mathcal{H})$
- Operators $J_\mathcal{E} \in \mathcal{B}(\mathcal{H} \otimes \mathcal{H})$

Moreover:

| Channel Property | Choi Matrix Property |
|-----------------|---------------------|
| Completely Positive | $J_\mathcal{E} \geq 0$ (positive semidefinite) |
| Trace Preserving | $\text{Tr}_B(J_\mathcal{E}) = I_A/d$ |
| Unital ($\mathcal{E}(I) = I$) | $\text{Tr}_A(J_\mathcal{E}) = I_B/d$ |

**Key Result:** $\mathcal{E}$ is CPTP $\Leftrightarrow$ $J_\mathcal{E} \geq 0$ and $\text{Tr}_B(J_\mathcal{E}) = I/d$.

### 5. Computing the Choi Matrix

**Method 1: Direct Definition**

Apply $\mathcal{I} \otimes \mathcal{E}$ to $|\Phi^+\rangle\langle\Phi^+|$.

**Method 2: From Kraus Operators**

If $\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger$, then:

$$J_\mathcal{E} = \sum_k |K_k\rangle\rangle\langle\langle K_k|$$

where $|K_k\rangle\rangle = (I \otimes K_k)|\Phi^+\rangle$ is the **vectorization** of $K_k$.

**Vectorization (vec or |·⟩⟩):**
For a matrix $A = \sum_{ij} A_{ij}|i\rangle\langle j|$:
$$|A\rangle\rangle = \sum_{ij} A_{ij} |i\rangle \otimes |j\rangle$$

**Key Property:** $|K\rangle\rangle = \text{vec}(K) = (I \otimes K)|\Phi^+\rangle \cdot \sqrt{d}$

### 6. Examples

#### Example 1: Identity Channel

$\mathcal{I}(\rho) = \rho$

Choi matrix:
$$J_\mathcal{I} = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

This is a pure state (rank 1), reflecting that the identity has Kraus rank 1.

#### Example 2: Completely Depolarizing Channel

$\mathcal{E}(\rho) = \frac{I}{2}$ for all $\rho$.

The channel maps everything to the maximally mixed state.

Choi matrix:
$$J_\mathcal{E} = \frac{I \otimes I}{4} = \frac{I_4}{4}$$

This is the maximally mixed state on two qubits!

#### Example 3: Bit-Flip Channel

$\mathcal{E}_X(\rho) = (1-p)\rho + pX\rho X$

Kraus operators: $K_0 = \sqrt{1-p}I$, $K_1 = \sqrt{p}X$

Choi matrix (for $d=2$):
$$J_{\mathcal{E}_X} = (1-p)|\Phi^+\rangle\langle\Phi^+| + p|\Psi^+\rangle\langle\Psi^+|$$

where $|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$.

In matrix form:
$$J_{\mathcal{E}_X} = \frac{1}{2}\begin{pmatrix} 1-p & 0 & 0 & 1-2p \\ 0 & p & p & 0 \\ 0 & p & p & 0 \\ 1-2p & 0 & 0 & 1-p \end{pmatrix}$$

#### Example 4: Amplitude Damping

Kraus operators: $K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$, $K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$

Choi matrix:
$$J_{\text{AD}} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & \sqrt{1-\gamma} \\ 0 & \gamma & 0 & 0 \\ 0 & 0 & 0 & 0 \\ \sqrt{1-\gamma} & 0 & 0 & 1-\gamma \end{pmatrix}$$

### 7. Extracting Kraus Operators from Choi Matrix

Given the Choi matrix $J_\mathcal{E}$, we can find Kraus operators:

**Step 1:** Eigendecompose $J_\mathcal{E} = \sum_k \lambda_k |\psi_k\rangle\langle\psi_k|$ (since $J_\mathcal{E} \geq 0$)

**Step 2:** For each eigenvector $|\psi_k\rangle$ with $\lambda_k > 0$:
- Reshape $|\psi_k\rangle$ (a $d^2$-dimensional vector) into a $d \times d$ matrix
- The Kraus operator is $K_k = \sqrt{d \cdot \lambda_k} \cdot \text{reshape}(|\psi_k\rangle)$

**The Kraus Rank:**
$$\text{Kraus rank} = \text{rank}(J_\mathcal{E})$$

### 8. Recovering the Channel from Choi Matrix

Given $J_\mathcal{E}$, we can compute $\mathcal{E}(\rho)$:

$$\mathcal{E}(\rho) = d \cdot \text{Tr}_A[(\rho^T \otimes I_B) J_\mathcal{E}]$$

where $\rho^T$ is the transpose.

**Alternative:** Use the extracted Kraus operators.

### 9. Physical Interpretation

The Choi matrix has a beautiful physical interpretation:

**Prepare-and-Measure Protocol:**
1. Prepare the maximally entangled state $|\Phi^+\rangle$ on systems $A$ and $A'$
2. Apply the channel $\mathcal{E}$ to system $A'$, getting output system $B$
3. The resulting state on $A$ and $B$ is exactly $J_\mathcal{E}$ (up to normalization)

**Entanglement as a Probe:**
The maximally entangled state "probes" the channel by sending one part through it. The resulting correlations completely characterize the channel.

---

## Quantum Computing Connection

### Process Tomography

The Choi-Jamiolkowski isomorphism underlies **quantum process tomography**:

1. To characterize an unknown channel $\mathcal{E}$, we don't need to test every possible input
2. Instead, apply $\mathcal{E}$ to half of an entangled state and perform state tomography on the result
3. This directly gives us the Choi matrix!

**Practical Implementation:**
- Prepare entangled pairs
- Send one qubit through the device under test
- Perform joint measurements
- Reconstruct $J_\mathcal{E}$

### Benchmarking Quantum Gates

Gate fidelity can be computed from the Choi matrix:

$$F_{\text{avg}}(\mathcal{E}, \mathcal{U}) = \frac{d \cdot F(J_\mathcal{E}, J_\mathcal{U}) + 1}{d + 1}$$

where $F(J_\mathcal{E}, J_\mathcal{U})$ is the fidelity between Choi matrices.

---

## Worked Examples

### Example 1: Computing Choi Matrix from Kraus Operators

**Problem:** Find the Choi matrix for the phase-flip channel $\mathcal{E}_Z(\rho) = (1-p)\rho + pZ\rho Z$.

**Solution:**

Step 1: Identify Kraus operators
$$K_0 = \sqrt{1-p}I = \sqrt{1-p}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$
$$K_1 = \sqrt{p}Z = \sqrt{p}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

Step 2: Vectorize each Kraus operator

$|K_0\rangle\rangle = \sqrt{1-p} \cdot \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ (since $I$ corresponds to $|\Phi^+\rangle$)

$|K_1\rangle\rangle = \sqrt{p} \cdot \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$ (since $Z$ gives $|\Phi^-\rangle$)

Wait, let's be more careful. For $d=2$:
$$|K\rangle\rangle = (I \otimes K)|\Phi^+\rangle \cdot \sqrt{2}$$

For $K_0 = \sqrt{1-p}I$:
$$(I \otimes \sqrt{1-p}I)|\Phi^+\rangle = \sqrt{1-p}|\Phi^+\rangle$$
So $|K_0\rangle\rangle/\sqrt{2} = \sqrt{1-p}|\Phi^+\rangle$

For $K_1 = \sqrt{p}Z$:
$$(I \otimes \sqrt{p}Z)|\Phi^+\rangle = \sqrt{p} \cdot \frac{1}{\sqrt{2}}(|0\rangle Z|0\rangle + |1\rangle Z|1\rangle)$$
$$= \sqrt{p} \cdot \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle) = \sqrt{p}|\Phi^-\rangle$$

Step 3: Compute Choi matrix
$$J_{\mathcal{E}_Z} = |K_0\rangle\rangle\langle\langle K_0| + |K_1\rangle\rangle\langle\langle K_1|$$
$$= (1-p)|\Phi^+\rangle\langle\Phi^+| + p|\Phi^-\rangle\langle\Phi^-|$$

In matrix form:
$$|\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

$$|\Phi^-\rangle\langle\Phi^-| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & -1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ -1 & 0 & 0 & 1 \end{pmatrix}$$

$$\boxed{J_{\mathcal{E}_Z} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1-2p \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1-2p & 0 & 0 & 1 \end{pmatrix}}$$

---

### Example 2: Verifying CPTP from Choi Matrix

**Problem:** Verify that the Choi matrix from Example 1 represents a CPTP map.

**Solution:**

**Check 1: Positive Semidefinite**

Find eigenvalues of $J_{\mathcal{E}_Z}$:

The matrix has block structure. Non-zero eigenvalues come from the 2×2 block in corners:
$$\begin{pmatrix} 1/2 & (1-2p)/2 \\ (1-2p)/2 & 1/2 \end{pmatrix}$$

Eigenvalues: $\lambda = \frac{1}{2} \pm \frac{1-2p}{2} = \{1-p, p\}$

For $0 \leq p \leq 1$: both eigenvalues $\geq 0$ ✓

**Check 2: Trace Preservation**

$\text{Tr}_B(J_{\mathcal{E}_Z}) = ?$

Partial trace over second system:
$$\text{Tr}_B(J_{\mathcal{E}_Z}) = \frac{1}{2}\begin{pmatrix} 1+0 & 0+0 \\ 0+0 & 0+1 \end{pmatrix} = \frac{I}{2}$$

This equals $I/d = I/2$ ✓

**Conclusion:** $\mathcal{E}_Z$ is CPTP.

---

### Example 3: Extracting Kraus Operators from Choi Matrix

**Problem:** Given the Choi matrix for the completely depolarizing channel $J = I_4/4$, find Kraus operators.

**Solution:**

Step 1: Eigendecompose $J = I_4/4$

All eigenvalues are $1/4$, with eigenvectors $|00\rangle, |01\rangle, |10\rangle, |11\rangle$.

Step 2: Convert to Kraus operators

For each eigenvector $|ij\rangle$ with eigenvalue $1/4$:
$$K_{ij} = \sqrt{d \cdot \lambda} \cdot \text{reshape}(|ij\rangle) = \sqrt{2 \cdot \frac{1}{4}} \cdot |i\rangle\langle j| = \frac{1}{\sqrt{2}}|i\rangle\langle j|$$

Kraus operators:
$$K_{00} = \frac{1}{\sqrt{2}}|0\rangle\langle 0|, \quad K_{01} = \frac{1}{\sqrt{2}}|0\rangle\langle 1|$$
$$K_{10} = \frac{1}{\sqrt{2}}|1\rangle\langle 0|, \quad K_{11} = \frac{1}{\sqrt{2}}|1\rangle\langle 1|$$

**Verification:**
$$\mathcal{E}(\rho) = \sum_{i,j} K_{ij} \rho K_{ij}^\dagger = \frac{1}{2}\sum_{i,j}|i\rangle\langle j|\rho|j\rangle\langle i| = \frac{I}{2}\text{Tr}(\rho) = \frac{I}{2}$$

This is indeed the completely depolarizing channel!

---

## Practice Problems

### Direct Application

1. **Problem 1:** Compute the Choi matrix for the unitary channel $\mathcal{U}(\rho) = H\rho H$ where $H$ is the Hadamard gate.

2. **Problem 2:** Given the Choi matrix $J = |\Phi^+\rangle\langle\Phi^+|$, verify it corresponds to the identity channel.

3. **Problem 3:** For the depolarizing channel with parameter $p$, show that the Choi matrix has eigenvalues $(1-3p/4)$ and $p/4$ (with multiplicity 3).

### Intermediate

4. **Problem 4:** Prove that if $\mathcal{E}$ is a unital channel ($\mathcal{E}(I) = I$), then $\text{Tr}_A(J_\mathcal{E}) = I/d$.

5. **Problem 5:** The transpose map $T(\rho) = \rho^T$ is positive but not completely positive. Compute its "Choi matrix" and show it has a negative eigenvalue.

6. **Problem 6:** For two channels $\mathcal{E}_1$ and $\mathcal{E}_2$, express the Choi matrix of their composition $\mathcal{E}_2 \circ \mathcal{E}_1$ in terms of $J_{\mathcal{E}_1}$ and $J_{\mathcal{E}_2}$.

### Challenging

7. **Problem 7:** Prove that the fidelity between two channels (defined via their Choi matrices) satisfies $F(\mathcal{E}_1, \mathcal{E}_2) = F(J_{\mathcal{E}_1}, J_{\mathcal{E}_2})$.

8. **Problem 8:** Show that a channel is entanglement-breaking if and only if its Choi matrix is separable.

9. **Problem 9:** Derive the formula for recovering a channel from its Choi matrix: $\mathcal{E}(\rho) = d \cdot \text{Tr}_A[(\rho^T \otimes I)J_\mathcal{E}]$.

---

## Computational Lab

```python
"""
Day 646 Computational Lab: Choi-Jamiolkowski Isomorphism
========================================================
Topics: Computing Choi matrices, verifying CPTP, extracting Kraus operators
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from typing import List, Tuple

# Standard operators
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def maximally_entangled_state(d: int) -> np.ndarray:
    """
    Create the maximally entangled state |Φ+⟩ = (1/√d) Σᵢ |ii⟩
    Returns the density matrix |Φ+⟩⟨Φ+|.
    """
    phi_plus = np.zeros((d * d, 1), dtype=complex)
    for i in range(d):
        phi_plus[i * d + i] = 1 / np.sqrt(d)
    return phi_plus @ phi_plus.conj().T


def kraus_to_choi(kraus_ops: List[np.ndarray]) -> np.ndarray:
    """
    Compute the Choi matrix from Kraus operators.

    J_E = Σₖ |Kₖ⟩⟩⟨⟨Kₖ|

    where |K⟩⟩ = vec(K) is the vectorization of K.
    """
    d = kraus_ops[0].shape[0]
    choi = np.zeros((d * d, d * d), dtype=complex)

    for K in kraus_ops:
        # Vectorize K: |K⟩⟩ = (I ⊗ K)|Φ+⟩ * √d
        # Equivalently, stack columns of K
        vec_K = K.flatten('F').reshape(-1, 1)  # Column-major vectorization
        choi += vec_K @ vec_K.conj().T

    return choi


def choi_to_kraus(choi: np.ndarray, tol: float = 1e-10) -> List[np.ndarray]:
    """
    Extract Kraus operators from Choi matrix via eigendecomposition.

    J = Σₖ λₖ |ψₖ⟩⟨ψₖ|
    Kₖ = √(d·λₖ) · reshape(|ψₖ⟩)
    """
    d_sq = choi.shape[0]
    d = int(np.sqrt(d_sq))

    # Eigendecompose
    eigenvalues, eigenvectors = np.linalg.eigh(choi)

    kraus_ops = []
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if lam > tol:
            # Reshape eigenvector to d×d matrix
            K = np.sqrt(lam) * vec.reshape((d, d), order='F')
            kraus_ops.append(K)

    return kraus_ops


def verify_cptp_from_choi(choi: np.ndarray, tol: float = 1e-10) -> dict:
    """
    Verify CPTP conditions from Choi matrix.

    CPTP ⟺ J ≥ 0 and Tr_B(J) = I/d
    """
    d_sq = choi.shape[0]
    d = int(np.sqrt(d_sq))

    results = {}

    # Check positive semidefinite
    eigenvalues = np.linalg.eigvalsh(choi)
    results['eigenvalues'] = eigenvalues
    results['min_eigenvalue'] = min(eigenvalues)
    results['is_positive'] = min(eigenvalues) > -tol

    # Check trace preservation: Tr_B(J) = I/d
    # Partial trace over second system
    partial_trace = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                partial_trace[i, j] += choi[i * d + k, j * d + k]

    expected = np.eye(d) / d
    tp_error = np.max(np.abs(partial_trace - expected))
    results['partial_trace'] = partial_trace
    results['tp_error'] = tp_error
    results['is_trace_preserving'] = tp_error < tol

    results['is_cptp'] = results['is_positive'] and results['is_trace_preserving']

    return results


def apply_channel_via_choi(rho: np.ndarray, choi: np.ndarray) -> np.ndarray:
    """
    Apply channel to state using Choi matrix.

    E(ρ) = d · Tr_A[(ρ^T ⊗ I) J]
    """
    d = rho.shape[0]

    # Compute ρ^T ⊗ I
    rho_T_tensor_I = np.kron(rho.T, np.eye(d))

    # Multiply with Choi matrix
    product = rho_T_tensor_I @ choi

    # Partial trace over first system
    output = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                output[i, j] += product[k * d + i, k * d + j]

    return d * output


# ===== DEMONSTRATIONS =====

print("=" * 70)
print("PART 1: Computing Choi Matrices for Standard Channels")
print("=" * 70)

# Define Kraus operators for various channels
def bit_flip_kraus(p):
    return [np.sqrt(1-p) * I, np.sqrt(p) * X]

def phase_flip_kraus(p):
    return [np.sqrt(1-p) * I, np.sqrt(p) * Z]

def depolarizing_kraus(p):
    return [np.sqrt(1-3*p/4) * I, np.sqrt(p/4) * X,
            np.sqrt(p/4) * Y, np.sqrt(p/4) * Z]

def amplitude_damping_kraus(gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]

# Compute and display Choi matrices
channels = {
    'Identity': [I],
    'Bit-flip (p=0.1)': bit_flip_kraus(0.1),
    'Phase-flip (p=0.2)': phase_flip_kraus(0.2),
    'Depolarizing (p=0.3)': depolarizing_kraus(0.3),
    'Amplitude damping (γ=0.5)': amplitude_damping_kraus(0.5),
    'Hadamard': [H]
}

for name, kraus_ops in channels.items():
    choi = kraus_to_choi(kraus_ops)
    result = verify_cptp_from_choi(choi)

    print(f"\n{name}:")
    print(f"  Choi matrix rank: {np.linalg.matrix_rank(choi)}")
    print(f"  Eigenvalues: {np.round(result['eigenvalues'], 4)}")
    print(f"  Is CPTP: {result['is_cptp']}")


print("\n" + "=" * 70)
print("PART 2: Round-Trip Verification (Kraus → Choi → Kraus)")
print("=" * 70)

# Test with depolarizing channel
p = 0.2
original_kraus = depolarizing_kraus(p)
choi = kraus_to_choi(original_kraus)
recovered_kraus = choi_to_kraus(choi)

print(f"\nDepolarizing channel (p={p}):")
print(f"  Original number of Kraus operators: {len(original_kraus)}")
print(f"  Recovered number of Kraus operators: {len(recovered_kraus)}")

# Verify by applying to a test state
ket_plus = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
rho_test = ket_plus @ ket_plus.conj().T

# Apply using original Kraus
rho_original = sum(K @ rho_test @ K.conj().T for K in original_kraus)

# Apply using recovered Kraus
rho_recovered = sum(K @ rho_test @ K.conj().T for K in recovered_kraus)

print(f"  Difference in outputs: {np.max(np.abs(rho_original - rho_recovered)):.2e}")


print("\n" + "=" * 70)
print("PART 3: Applying Channel via Choi Matrix")
print("=" * 70)

# Compare direct Kraus application vs Choi-based application
test_states = {
    '|0⟩': np.array([[1, 0], [0, 0]], dtype=complex),
    '|1⟩': np.array([[0, 0], [0, 1]], dtype=complex),
    '|+⟩': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
}

print("\nAmplitude Damping (γ=0.3) - Comparing methods:")
gamma = 0.3
ad_kraus = amplitude_damping_kraus(gamma)
ad_choi = kraus_to_choi(ad_kraus)

for name, rho in test_states.items():
    # Method 1: Direct Kraus
    rho_kraus = sum(K @ rho @ K.conj().T for K in ad_kraus)

    # Method 2: Via Choi
    rho_choi = apply_channel_via_choi(rho, ad_choi)

    diff = np.max(np.abs(rho_kraus - rho_choi))
    print(f"  {name}: difference = {diff:.2e}")


print("\n" + "=" * 70)
print("PART 4: Visualizing Choi Matrix Structure")
print("=" * 70)

def visualize_choi_matrix(choi: np.ndarray, title: str):
    """Visualize magnitude and phase of Choi matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Magnitude
    im1 = axes[0].imshow(np.abs(choi), cmap='Blues')
    axes[0].set_title(f'{title}\nMagnitude |J_ij|')
    axes[0].set_xlabel('Column index')
    axes[0].set_ylabel('Row index')
    plt.colorbar(im1, ax=axes[0])

    # Real part
    im2 = axes[1].imshow(np.real(choi), cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1].set_title(f'{title}\nReal part Re(J_ij)')
    axes[1].set_xlabel('Column index')
    axes[1].set_ylabel('Row index')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    filename = f'choi_{title.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# Visualize Choi matrices
print("\nVisualizing Choi matrices:")
visualize_choi_matrix(kraus_to_choi([I]), "Identity Channel")
visualize_choi_matrix(kraus_to_choi(depolarizing_kraus(0.5)), "Depolarizing (p=0.5)")
visualize_choi_matrix(kraus_to_choi(amplitude_damping_kraus(0.5)), "Amplitude Damping (gamma=0.5)")


print("\n" + "=" * 70)
print("PART 5: The Transpose Map (Not Completely Positive)")
print("=" * 70)

# The transpose map T(ρ) = ρ^T is positive but NOT completely positive
# Let's compute its "Choi matrix" and show it has negative eigenvalues

def transpose_action(rho: np.ndarray) -> np.ndarray:
    """Apply transpose map."""
    return rho.T

# Compute Choi-like matrix for transpose
# J_T = (I ⊗ T)(|Φ+⟩⟨Φ+|)
d = 2
phi_plus = maximally_entangled_state(d)

# Apply transpose to second system
# This is equivalent to partial transpose
J_transpose = np.zeros((d*d, d*d), dtype=complex)
for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                # (I ⊗ T)|ij⟩⟨kl| = |ij⟩⟨kl| with second index transposed
                # |i⟩|j⟩ → |i⟩|j⟩, but ⟨k|⟨l| → ⟨k|⟨l|
                # Actually T(|j⟩⟨l|) = |l⟩⟨j| (outer product transposed)
                row = i * d + l  # |i⟩|l⟩
                col = k * d + j  # ⟨k|⟨j|
                J_transpose[row, col] += phi_plus[i*d+j, k*d+l]

print("\n'Choi matrix' for transpose map:")
eigenvalues_T = np.linalg.eigvalsh(J_transpose)
print(f"Eigenvalues: {np.round(eigenvalues_T, 4)}")
print(f"Minimum eigenvalue: {min(eigenvalues_T):.4f}")
print(f"Is positive semidefinite: {min(eigenvalues_T) > -1e-10}")
print("\nThis demonstrates that transpose is positive but NOT completely positive!")


print("\n" + "=" * 70)
print("PART 6: Channel Fidelity via Choi Matrices")
print("=" * 70)

def channel_fidelity(choi1: np.ndarray, choi2: np.ndarray) -> float:
    """
    Compute fidelity between two channels via their Choi matrices.
    F(E1, E2) = F(J1, J2) for normalized Choi states.
    """
    # Normalize Choi matrices to be valid quantum states
    J1 = choi1 / np.trace(choi1)
    J2 = choi2 / np.trace(choi2)

    # State fidelity F(ρ,σ) = (Tr√(√ρ σ √ρ))²
    sqrt_J1 = linalg.sqrtm(J1)
    product = sqrt_J1 @ J2 @ sqrt_J1
    sqrt_product = linalg.sqrtm(product)
    return np.real(np.trace(sqrt_product))**2

# Compare depolarizing channels with different parameters
print("\nFidelity between depolarizing channels:")
p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Create fidelity matrix
fid_matrix = np.zeros((len(p_values), len(p_values)))
for i, p1 in enumerate(p_values):
    for j, p2 in enumerate(p_values):
        choi1 = kraus_to_choi(depolarizing_kraus(p1))
        choi2 = kraus_to_choi(depolarizing_kraus(p2))
        fid_matrix[i, j] = channel_fidelity(choi1, choi2)

print("\nChannel fidelity matrix (depolarizing channels):")
print("p values:", p_values)
print(np.array2string(fid_matrix, precision=3))

# Visualize
plt.figure(figsize=(8, 6))
plt.imshow(fid_matrix, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(label='Channel Fidelity')
plt.xticks(range(len(p_values)), [f'{p:.1f}' for p in p_values])
plt.yticks(range(len(p_values)), [f'{p:.1f}' for p in p_values])
plt.xlabel('p₂')
plt.ylabel('p₁')
plt.title('Fidelity Between Depolarizing Channels')
plt.savefig('channel_fidelity_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: channel_fidelity_matrix.png")


print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Maximally entangled state | $\|\Phi^+\rangle = \frac{1}{\sqrt{d}}\sum_i \|ii\rangle$ |
| Choi matrix definition | $J_\mathcal{E} = (\mathcal{I} \otimes \mathcal{E})(\|\Phi^+\rangle\langle\Phi^+\|)$ |
| CP condition | $J_\mathcal{E} \geq 0$ |
| TP condition | $\text{Tr}_B(J_\mathcal{E}) = I/d$ |
| Kraus rank | $\text{rank}(J_\mathcal{E})$ |

### Main Takeaways

1. **Choi-Jamiolkowski isomorphism** establishes a one-to-one correspondence between channels and operators
2. **Complete positivity** is equivalent to the Choi matrix being positive semidefinite
3. **Trace preservation** corresponds to a partial trace condition on the Choi matrix
4. **Kraus operators** can be extracted from the Choi matrix via eigendecomposition
5. **Kraus rank** equals the rank of the Choi matrix
6. The isomorphism underlies **quantum process tomography**

---

## Daily Checklist

- [ ] I can state the Choi-Jamiolkowski isomorphism
- [ ] I can compute the Choi matrix from Kraus operators
- [ ] I understand how to check CPTP from the Choi matrix
- [ ] I can extract Kraus operators from a Choi matrix
- [ ] I understand the physical interpretation of the isomorphism
- [ ] I completed the computational lab exercises
- [ ] I attempted at least 3 practice problems

---

## Preview: Day 647

Tomorrow we explore **Stinespring dilation**, which shows that every quantum channel can be realized as:
1. Coupling to an environment in a pure state
2. Applying a joint unitary evolution
3. Tracing out the environment

This provides the deepest physical insight into the nature of quantum channels.

---

*"The Choi-Jamiolkowski isomorphism reveals that quantum channels and entangled states are two sides of the same coin."* — John Preskill
