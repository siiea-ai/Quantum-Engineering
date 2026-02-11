# Week 159: Entanglement - Comprehensive Review Guide

## 1. Introduction

Entanglement is perhaps the most distinctively quantum feature of nature. As Schrödinger noted, it is "the characteristic trait of quantum mechanics, the one that enforces its entire departure from classical lines of thought." This week provides a comprehensive review of entanglement theory, covering detection, quantification, and fundamental tests like Bell inequalities.

---

## 2. Separability and Entanglement

### 2.1 Pure State Entanglement

A bipartite pure state $$|\psi\rangle_{AB}$$ is **separable** (or a product state) if:
$$|\psi\rangle_{AB} = |a\rangle_A \otimes |b\rangle_B$$

for some states $$|a\rangle$$ and $$|b\rangle$$. Otherwise, it is **entangled**.

**Criterion via Schmidt rank:**
- Schmidt rank = 1: Separable
- Schmidt rank > 1: Entangled

### 2.2 Mixed State Entanglement

A bipartite mixed state $$\rho_{AB}$$ is **separable** if it can be written as:
$$\rho_{AB} = \sum_i p_i \rho_A^{(i)} \otimes \rho_B^{(i)}$$

where $$p_i \geq 0$$, $$\sum_i p_i = 1$$, and $$\rho_A^{(i)}, \rho_B^{(i)}$$ are density matrices.

Equivalently, a separable state can be prepared by LOCC (Local Operations and Classical Communication) from a product state.

A state that is not separable is **entangled**.

### 2.3 The Separability Problem

Determining whether a given mixed state is separable is computationally hard (NP-hard in general). We rely on sufficient criteria for entanglement and necessary criteria for separability.

---

## 3. Bell States

### 3.1 The Four Bell States

The Bell states form a complete orthonormal basis for the two-qubit Hilbert space:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$
$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$
$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

### 3.2 Properties

**Maximally entangled:**
- Schmidt coefficients: $$\lambda_1 = \lambda_2 = 1/\sqrt{2}$$
- Entanglement entropy: $$E = 1$$ ebit (maximum for two qubits)
- Reduced states: $$\rho_A = \rho_B = I/2$$ (maximally mixed)

**Local unitary equivalence:**
All Bell states are related by local unitaries:
$$|\Phi^-\rangle = (Z \otimes I)|\Phi^+\rangle$$
$$|\Psi^+\rangle = (X \otimes I)|\Phi^+\rangle$$
$$|\Psi^-\rangle = (iY \otimes I)|\Phi^+\rangle$$

**Orthonormality:**
$$\langle\Phi^\pm|\Phi^\mp\rangle = 0, \quad \langle\Psi^\pm|\Psi^\mp\rangle = 0, \quad \langle\Phi^\pm|\Psi^\pm\rangle = 0$$

### 3.3 Bell State Measurement

Bell states can be distinguished by measuring in the Bell basis, achieved via:
1. CNOT gate
2. Hadamard on first qubit
3. Computational basis measurement

$$\text{CNOT} \cdot (H \otimes I)|\Phi^+\rangle = |00\rangle$$
$$\text{CNOT} \cdot (H \otimes I)|\Phi^-\rangle = |10\rangle$$
$$\text{CNOT} \cdot (H \otimes I)|\Psi^+\rangle = |01\rangle$$
$$\text{CNOT} \cdot (H \otimes I)|\Psi^-\rangle = |11\rangle$$

---

## 4. Bell Inequalities

### 4.1 EPR Argument

Einstein, Podolsky, and Rosen (1935) argued that quantum mechanics is incomplete because entangled particles seem to have instantaneous correlations that violate locality.

**Local hidden variable (LHV) theories** assume:
1. **Realism**: Measurement outcomes are predetermined by hidden variables
2. **Locality**: Measurements on one particle don't affect the other

### 4.2 CHSH Inequality

Clauser, Horne, Shimony, and Holt (1969) derived a testable inequality.

**Setup:**
- Alice measures observable $$A$$ (outcomes $$\pm 1$$) with settings $$a$$ or $$a'$$
- Bob measures observable $$B$$ (outcomes $$\pm 1$$) with settings $$b$$ or $$b'$$

**Correlation function:**
$$E(a,b) = \langle A_a \otimes B_b \rangle$$

**CHSH quantity:**
$$S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$

### 4.3 Classical Bound

**Theorem:** For any local hidden variable theory:
$$|S| \leq 2$$

**Proof sketch:**
Each particle has predetermined values $$A_a, A_{a'}, B_b, B_{b'} = \pm 1$$.

$$S = A_a(B_b - B_{b'}) + A_{a'}(B_b + B_{b'})$$

Since $$B_b, B_{b'} = \pm 1$$: either $$B_b - B_{b'} = 0$$ and $$B_b + B_{b'} = \pm 2$$, or vice versa.

Thus $$|S| = 2$$ for each run, and averaging: $$|\langle S \rangle| \leq 2$$.

### 4.4 Quantum Violation

**Quantum prediction:**
For the singlet state $$|\Psi^-\rangle$$ and measurements:
- $$A_a = \vec{a} \cdot \vec{\sigma}$$, $$A_{a'} = \vec{a}' \cdot \vec{\sigma}$$ on Alice's qubit
- $$B_b = \vec{b} \cdot \vec{\sigma}$$, $$B_{b'} = \vec{b}' \cdot \vec{\sigma}$$ on Bob's qubit

$$E(a,b) = -\vec{a} \cdot \vec{b}$$

**Optimal settings:**
Choose angles: $$a = 0°$$, $$a' = 90°$$, $$b = 45°$$, $$b' = 135°$$

$$S = -\cos 45° + \cos 135° - \cos 45° - \cos 135°$$
$$= -\frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{2}} + \frac{1}{\sqrt{2}} = -2\sqrt{2}$$

$$|S| = 2\sqrt{2} \approx 2.828 > 2$$

### 4.5 Tsirelson Bound

**Theorem (Tsirelson):** For any quantum state and measurements:
$$|S| \leq 2\sqrt{2}$$

The bound is achieved by maximally entangled states with optimal measurement settings.

---

## 5. Separability Criteria

### 5.1 PPT Criterion

The **Partial Transpose** over subsystem $$B$$ is:
$$(\rho_{AB})^{T_B}$$

defined by transposing only the $$B$$ indices.

**Peres-Horodecki criterion:**
- If $$\rho_{AB}$$ is separable, then $$\rho_{AB}^{T_B} \geq 0$$ (PPT)
- If $$\rho_{AB}^{T_B}$$ has negative eigenvalues, then $$\rho_{AB}$$ is entangled

**Completeness:**
- For $$2 \times 2$$ and $$2 \times 3$$ systems: PPT $$\Leftrightarrow$$ separable
- For larger systems: PPT is necessary but not sufficient (bound entangled states exist)

### 5.2 Computing Partial Transpose

For $$\rho = \sum_{ijkl} \rho_{ij,kl}|ij\rangle\langle kl|$$:

$$\rho^{T_B} = \sum_{ijkl} \rho_{ij,kl}|il\rangle\langle kj|$$

In matrix form: swap the second index with the fourth.

**Example: Bell state**
$$|\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

$$(\rho^{T_B})_{ij,kl} = \rho_{il,kj}$$

$$\rho^{T_B} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Eigenvalues: $$\{1/2, 1/2, 1/2, -1/2\}$$ - negative eigenvalue confirms entanglement.

### 5.3 Entanglement Witnesses

An **entanglement witness** is an operator $$W$$ such that:
- $$\text{Tr}(W\rho) \geq 0$$ for all separable $$\rho$$
- $$\text{Tr}(W\rho_E) < 0$$ for some entangled $$\rho_E$$

**Theorem:** $$\rho$$ is entangled iff there exists a witness $$W$$ with $$\text{Tr}(W\rho) < 0$$.

**Construction:** For PPT-violating states, witnesses can be constructed from the partial transpose.

---

## 6. Entanglement Measures

### 6.1 Requirements for Entanglement Measures

A good entanglement measure $$E(\rho)$$ should satisfy:

1. **Non-negativity:** $$E(\rho) \geq 0$$
2. **Zero for separable states:** $$E(\rho) = 0$$ if $$\rho$$ is separable
3. **LOCC monotonicity:** $$E$$ cannot increase under LOCC
4. **Convexity:** $$E(\sum_i p_i \rho_i) \leq \sum_i p_i E(\rho_i)$$ (optional)

### 6.2 Entanglement Entropy (Pure States)

For a pure bipartite state $$|\psi\rangle_{AB}$$:
$$E(|\psi\rangle) = S(\rho_A) = S(\rho_B)$$

where $$S$$ is von Neumann entropy.

**Properties:**
- $$E = 0$$ for product states
- $$E = \log_2 d$$ for maximally entangled states in dimension $$d$$
- $$E = 1$$ ebit for Bell states

**In terms of Schmidt coefficients:**
$$E = -\sum_i \lambda_i^2 \log_2 \lambda_i^2$$

### 6.3 Concurrence

For **pure two-qubit states:**
$$C(|\psi\rangle) = |\langle\psi|\tilde{\psi}\rangle|$$

where $$|\tilde{\psi}\rangle = (\sigma_y \otimes \sigma_y)|\psi^*\rangle$$ (spin-flip).

For **mixed two-qubit states** (Wootters formula):
$$C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)$$

where $$\lambda_i$$ are the decreasing square roots of eigenvalues of:
$$R = \rho \tilde{\rho}$$

with $$\tilde{\rho} = (\sigma_y \otimes \sigma_y)\rho^*(\sigma_y \otimes \sigma_y)$$.

**Properties:**
- $$0 \leq C \leq 1$$
- $$C = 0$$ for separable states
- $$C = 1$$ for Bell states

**Entanglement of formation:**
$$E_F(\rho) = h\left(\frac{1 + \sqrt{1-C^2}}{2}\right)$$

where $$h(x) = -x\log_2 x - (1-x)\log_2(1-x)$$ is binary entropy.

### 6.4 Negativity

Based on the PPT criterion:
$$\mathcal{N}(\rho) = \frac{\|\rho^{T_B}\|_1 - 1}{2}$$

where $$\|A\|_1 = \text{Tr}\sqrt{A^\dagger A}$$ is the trace norm.

Equivalently:
$$\mathcal{N}(\rho) = \sum_i \frac{|\mu_i| - \mu_i}{2}$$

where $$\mu_i$$ are eigenvalues of $$\rho^{T_B}$$ (sum of absolute values of negative eigenvalues).

**Logarithmic negativity:**
$$E_N(\rho) = \log_2 \|\rho^{T_B}\|_1 = \log_2(1 + 2\mathcal{N})$$

**Properties:**
- $$\mathcal{N} = 0$$ for PPT states
- For Bell states: $$\mathcal{N} = 1/2$$, $$E_N = 1$$
- Not convex, but an entanglement monotone
- Computable for any dimension

### 6.5 Comparison of Measures

| Measure | Pure States | Mixed States | Computable | Dimension |
|---------|-------------|--------------|------------|-----------|
| Entropy | $$S(\rho_A)$$ | N/A | Yes | Any |
| Concurrence | $$\|\langle\psi\|\tilde{\psi}\rangle\|$$ | Wootters formula | Two qubits only | 2×2 |
| Negativity | Easy | Easy | Yes | Any |
| Entanglement of Formation | = Entropy | Complex | Two qubits | 2×2 |

---

## 7. LOCC and Entanglement Monotones

### 7.1 LOCC Operations

**Local Operations and Classical Communication:**
- Alice and Bob can perform any local quantum operations
- They can send classical messages to coordinate
- They cannot send quantum information

### 7.2 Entanglement Cannot Increase Under LOCC

**Theorem:** For any entanglement measure $$E$$ satisfying the monotonicity axiom:
$$E(\Lambda_{LOCC}(\rho)) \leq E(\rho)$$

This is why entanglement is a resource - it cannot be created by separated parties using only LOCC.

### 7.3 Entanglement Distillation

Parties can extract maximally entangled Bell pairs from many copies of a less entangled state using LOCC:
$$\rho^{\otimes n} \xrightarrow{LOCC} |\Phi^+\rangle^{\otimes m}$$

The rate $$m/n$$ approaches the **distillable entanglement** $$E_D(\rho)$$.

---

## 8. Advanced Topics

### 8.1 Bound Entanglement

States that are:
- Entangled (not separable)
- PPT (positive partial transpose)
- Not distillable

These exist in dimensions $$\geq 3 \times 3$$.

### 8.2 Multipartite Entanglement

For three or more parties, entanglement is more complex:
- Multiple inequivalent classes (GHZ, W, etc.)
- No unique measure
- Genuine multipartite entanglement vs. partial entanglement

### 8.3 Monogamy of Entanglement

If A and B are maximally entangled, neither can be entangled with C:
$$E_{AB} + E_{AC} \leq E_{A:BC}$$

Formalized by Coffman-Kundu-Wootters inequality for concurrence.

---

## 9. Summary

| Concept | Key Result |
|---------|-----------|
| Separability | Product states or convex combinations |
| Bell states | Four maximally entangled two-qubit states |
| CHSH bound | Classical: 2, Quantum: $$2\sqrt{2}$$ |
| PPT criterion | Necessary for separability |
| Entropy | $$E = S(\rho_A)$$ for pure states |
| Concurrence | Wootters formula for two qubits |
| Negativity | Sum of negative eigenvalues of $$\rho^{T_B}$$ |

---

## 10. Python Implementation

```python
import numpy as np
from scipy import linalg

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def bell_state(name):
    """Return Bell state vector."""
    if name == 'Phi+':
        return np.array([1, 0, 0, 1]) / np.sqrt(2)
    elif name == 'Phi-':
        return np.array([1, 0, 0, -1]) / np.sqrt(2)
    elif name == 'Psi+':
        return np.array([0, 1, 1, 0]) / np.sqrt(2)
    elif name == 'Psi-':
        return np.array([0, 1, -1, 0]) / np.sqrt(2)

def partial_transpose_B(rho, dim_A=2, dim_B=2):
    """Compute partial transpose over subsystem B."""
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    rho_pt = rho_reshaped.transpose(0, 3, 2, 1)
    return rho_pt.reshape(dim_A * dim_B, dim_A * dim_B)

def negativity(rho, dim_A=2, dim_B=2):
    """Compute negativity."""
    rho_pt = partial_transpose_B(rho, dim_A, dim_B)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return sum(abs(e) - e for e in eigenvalues) / 2

def log_negativity(rho, dim_A=2, dim_B=2):
    """Compute logarithmic negativity."""
    neg = negativity(rho, dim_A, dim_B)
    return np.log2(1 + 2 * neg)

def concurrence_pure(psi):
    """Compute concurrence for pure two-qubit state."""
    # Spin-flip: sigma_y tensor sigma_y
    sigma_yy = np.kron(Y, Y)
    psi_tilde = sigma_yy @ psi.conj()
    return abs(np.vdot(psi, psi_tilde))

def concurrence_mixed(rho):
    """Compute concurrence for mixed two-qubit state (Wootters formula)."""
    sigma_yy = np.kron(Y, Y)
    rho_tilde = sigma_yy @ rho.conj() @ sigma_yy
    R = rho @ rho_tilde
    eigenvalues = np.linalg.eigvals(R)
    lambdas = np.sqrt(np.abs(eigenvalues))
    lambdas = np.sort(lambdas)[::-1]  # Descending order
    return max(0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])

def entanglement_entropy(psi, dim_A=2, dim_B=2):
    """Compute entanglement entropy for pure state."""
    # Compute reduced density matrix
    rho = np.outer(psi, psi.conj())
    rho_A = partial_trace_B(rho, dim_A, dim_B)

    # Von Neumann entropy
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def partial_trace_B(rho, dim_A=2, dim_B=2):
    """Compute partial trace over B."""
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho_reshaped, axis1=1, axis2=3)

# Example calculations
if __name__ == "__main__":
    # Bell state analysis
    phi_plus = bell_state('Phi+')
    rho_bell = np.outer(phi_plus, phi_plus.conj())

    print("Bell state |Phi+>:")
    print(f"  Concurrence: {concurrence_pure(phi_plus):.4f}")
    print(f"  Negativity: {negativity(rho_bell):.4f}")
    print(f"  Log-negativity: {log_negativity(rho_bell):.4f}")
    print(f"  Entanglement entropy: {entanglement_entropy(phi_plus):.4f}")
```

---

*This review guide covers the essential theory for entanglement. Master these concepts before proceeding to quantum channels.*
