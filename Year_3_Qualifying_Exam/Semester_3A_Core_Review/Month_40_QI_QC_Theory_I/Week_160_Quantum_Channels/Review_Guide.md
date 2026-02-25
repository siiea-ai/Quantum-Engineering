# Week 160: Quantum Channels - Comprehensive Review Guide

## 1. Introduction

Quantum channels describe the most general physically allowed evolution of quantum states. While isolated systems evolve unitarily, realistic quantum systems interact with environments, undergo measurements, or experience noise. The mathematical framework of CPTP (Completely Positive Trace-Preserving) maps elegantly captures all these scenarios.

This week covers the theory of quantum channels, their representations, and important examples that appear throughout quantum information and computing.

---

## 2. Definition: CPTP Maps

### 2.1 What is a Quantum Channel?

A **quantum channel** (or quantum operation) is a linear map $$\mathcal{E}: \mathcal{B}(\mathcal{H}_A) \to \mathcal{B}(\mathcal{H}_B)$$ that transforms density matrices to density matrices.

### 2.2 Complete Positivity

A map $$\mathcal{E}$$ is **positive** if $$\rho \geq 0 \Rightarrow \mathcal{E}(\rho) \geq 0$$.

A map is **completely positive (CP)** if $$\mathcal{E} \otimes \mathcal{I}_n$$ is positive for all $$n$$, where $$\mathcal{I}_n$$ is the identity on an $$n$$-dimensional ancilla.

**Why complete positivity?**
Positivity alone is not sufficient. Consider the transpose map $$T(\rho) = \rho^T$$. It's positive but:

$$(T \otimes \mathcal{I})(|\Phi^+\rangle\langle\Phi^+|) = (\rho^{T_A}) = \text{has negative eigenvalues!}$$

If we could apply a non-CP map to part of an entangled system, we'd get unphysical negative probabilities.

### 2.3 Trace Preservation

A map is **trace-preserving (TP)** if $$\text{Tr}(\mathcal{E}(\rho)) = \text{Tr}(\rho)$$ for all $$\rho$$.

This ensures probability conservation: a valid density matrix maps to a valid density matrix.

### 2.4 CPTP = Quantum Channel

A map that is both completely positive and trace-preserving is called a **CPTP map** or **quantum channel**.

---

## 3. Kraus Representation

### 3.1 Theorem Statement

**Theorem (Kraus):** A linear map $$\mathcal{E}$$ is CPTP if and only if it can be written as:

$$\mathcal{E}(\rho) = \sum_{k=1}^{r} K_k \rho K_k^\dagger$$

where the **Kraus operators** $$\{K_k\}$$ satisfy:

$$\sum_{k=1}^{r} K_k^\dagger K_k = I$$

### 3.2 Physical Interpretation

The Kraus representation has a physical interpretation via the Stinespring dilation:

1. System $$S$$ starts in state $$\rho_S$$
2. Attach environment $$E$$ in state $$|0\rangle_E$$
3. Apply joint unitary $$U_{SE}$$
4. Trace out environment

$$\mathcal{E}(\rho_S) = \text{Tr}_E(U_{SE}(\rho_S \otimes |0\rangle\langle 0|_E)U_{SE}^\dagger)$$

The Kraus operators are:
$$K_k = \langle k|_E U_{SE} |0\rangle_E$$

### 3.3 Non-Uniqueness

The Kraus representation is not unique. If $$\{K_k\}$$ is a valid representation, so is $$\{K'_j\}$$ where:

$$K'_j = \sum_k U_{jk} K_k$$

for any unitary matrix $$U_{jk}$$.

Different representations correspond to different environment measurements.

### 3.4 Minimum Number of Kraus Operators

For a channel on a $$d$$-dimensional system, the minimum number of Kraus operators needed is called the **Kraus rank**. It satisfies:

$$1 \leq r \leq d^2$$

---

## 4. Standard Quantum Channels

### 4.1 Depolarizing Channel

The depolarizing channel represents isotropic noise:

$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{d^2-1}\sum_{i \neq 0} P_i \rho P_i^\dagger$$

For a qubit ($$d=2$$):
$$\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

**Alternative form:**
$$\mathcal{E}_p(\rho) = (1-\frac{4p}{3})\rho + \frac{p}{3}I$$

**Kraus operators:**
$$K_0 = \sqrt{1-p}I, \quad K_1 = \sqrt{p/3}X, \quad K_2 = \sqrt{p/3}Y, \quad K_3 = \sqrt{p/3}Z$$

**Effect on Bloch sphere:**
The Bloch vector shrinks uniformly: $$\vec{r} \to (1-\frac{4p}{3})\vec{r}$$

For $$p = 3/4$$: completely depolarizing, output is $$I/2$$.

### 4.2 Amplitude Damping Channel

Models energy relaxation (T1 decay) - a qubit in $$|1\rangle$$ decays to $$|0\rangle$$:

**Kraus operators:**
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

**Action:**
- $$K_0|0\rangle = |0\rangle$$, $$K_0|1\rangle = \sqrt{1-\gamma}|1\rangle$$
- $$K_1|0\rangle = 0$$, $$K_1|1\rangle = \sqrt{\gamma}|0\rangle$$

**Physical interpretation:**
- With probability $$1-\gamma$$: qubit stays in current state (with some amplitude reduction)
- With probability $$\gamma$$: if in $$|1\rangle$$, decays to $$|0\rangle$$

**Effect on density matrix:**
$$\begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix} \to \begin{pmatrix} \rho_{00} + \gamma\rho_{11} & \sqrt{1-\gamma}\rho_{01} \\ \sqrt{1-\gamma}\rho_{10} & (1-\gamma)\rho_{11} \end{pmatrix}$$

**Effect on Bloch vector:**
$$(r_x, r_y, r_z) \to (\sqrt{1-\gamma}r_x, \sqrt{1-\gamma}r_y, (1-\gamma)r_z + \gamma)$$

The Bloch ball contracts toward the north pole ($$|0\rangle$$).

### 4.3 Phase Damping Channel

Models dephasing (T2 decay) without energy exchange:

**Kraus operators (one form):**
$$K_0 = \sqrt{1-\lambda}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad K_1 = \sqrt{\lambda}\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \quad K_2 = \sqrt{\lambda}\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

**Alternative (two operators):**
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$

**Effect on density matrix:**
$$\begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix} \to \begin{pmatrix} \rho_{00} & (1-\lambda)\rho_{01} \\ (1-\lambda)\rho_{10} & \rho_{11} \end{pmatrix}$$

Diagonal elements unchanged, off-diagonals decay.

**Effect on Bloch vector:**
$$(r_x, r_y, r_z) \to ((1-\lambda)r_x, (1-\lambda)r_y, r_z)$$

The Bloch ball contracts toward the z-axis.

### 4.4 Bit Flip and Phase Flip

**Bit flip channel:**
$$\mathcal{E}_{\text{bf}}(\rho) = (1-p)\rho + pX\rho X$$

Kraus: $$K_0 = \sqrt{1-p}I$$, $$K_1 = \sqrt{p}X$$

**Phase flip channel:**
$$\mathcal{E}_{\text{pf}}(\rho) = (1-p)\rho + pZ\rho Z$$

Kraus: $$K_0 = \sqrt{1-p}I$$, $$K_1 = \sqrt{p}Z$$

### 4.5 Summary of Bloch Sphere Effects

| Channel | Effect on Bloch vector |
|---------|----------------------|
| Depolarizing | Uniform shrinkage toward origin |
| Amplitude damping | Shrinkage + drift toward north pole |
| Phase damping | Shrinkage toward z-axis |
| Bit flip | Shrinkage toward z-axis |
| Phase flip | Shrinkage toward z-axis |

---

## 5. Choi-Jamiolkowski Isomorphism

### 5.1 Channel-State Duality

There is a one-to-one correspondence between:
- Quantum channels $$\mathcal{E}: \mathcal{B}(\mathcal{H}_A) \to \mathcal{B}(\mathcal{H}_B)$$
- Operators on $$\mathcal{H}_B \otimes \mathcal{H}_A$$

### 5.2 Choi Matrix Definition

For a channel $$\mathcal{E}$$, the **Choi matrix** (or Choi state) is:

$$J(\mathcal{E}) = (\mathcal{E} \otimes \mathcal{I})(|\Phi^+\rangle\langle\Phi^+|)$$

where $$|\Phi^+\rangle = \frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}|ii\rangle$$ is the maximally entangled state.

In terms of Kraus operators:
$$J(\mathcal{E}) = \sum_k (K_k \otimes I)(|\Phi^+\rangle\langle\Phi^+|)(K_k^\dagger \otimes I)$$
$$= \frac{1}{d}\sum_k K_k \otimes K_k^*$$

### 5.3 Properties

**Complete positivity criterion:**
$$\mathcal{E} \text{ is CP} \Leftrightarrow J(\mathcal{E}) \geq 0$$

**Trace preservation criterion:**
$$\mathcal{E} \text{ is TP} \Leftrightarrow \text{Tr}_B(J(\mathcal{E})) = \frac{I_A}{d}$$

**Recovering the channel:**
$$\mathcal{E}(\rho) = d \cdot \text{Tr}_A[(I_B \otimes \rho^T)J(\mathcal{E})]$$

### 5.4 Examples

**Identity channel:**
$$J(\mathcal{I}) = |\Phi^+\rangle\langle\Phi^+|$$

**Depolarizing channel ($$p = 1$$, completely depolarizing):**
$$J(\mathcal{E}_{p=1}) = \frac{I}{4}$$

**Amplitude damping:**
$$J(\mathcal{E}_\gamma) = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & \sqrt{1-\gamma} \\ 0 & \gamma & 0 & 0 \\ 0 & 0 & 0 & 0 \\ \sqrt{1-\gamma} & 0 & 0 & 1-\gamma \end{pmatrix}$$

---

## 6. Useful Identities

### 6.1 Composition of Channels

If $$\mathcal{E}_1$$ has Kraus operators $$\{A_i\}$$ and $$\mathcal{E}_2$$ has $$\{B_j\}$$, then $$\mathcal{E}_2 \circ \mathcal{E}_1$$ has:

$$K_{ij} = B_j A_i$$

### 6.2 Tensor Product of Channels

$$(\mathcal{E}_1 \otimes \mathcal{E}_2)$$ has Kraus operators:

$$K_{ij} = A_i \otimes B_j$$

### 6.3 Unital Channels

A channel is **unital** if $$\mathcal{E}(I) = I$$.

For unital channels: $$\sum_k K_k K_k^\dagger = I$$ (in addition to $$\sum_k K_k^\dagger K_k = I$$)

Examples: depolarizing, phase damping, bit flip, phase flip
Counter-example: amplitude damping

---

## 7. Applications

### 7.1 Quantum Error Correction

Kraus operators describe errors that need to be corrected:
- Depolarizing: any Pauli error
- Amplitude damping: relaxation errors
- Phase damping: dephasing errors

### 7.2 Decoherence

Interaction with environment causes pure states to become mixed:
$$|\psi\rangle \to \mathcal{E}(|\psi\rangle\langle\psi|) = \text{mixed state}$$

### 7.3 Channel Capacity

The quantum capacity of a channel determines how much quantum information can be reliably transmitted.

---

## 8. Summary Table

| Channel | Kraus Operators | Effect | Unital? |
|---------|-----------------|--------|---------|
| Depolarizing | $$\sqrt{1-p}I, \sqrt{p/3}X,Y,Z$$ | Shrink to origin | Yes |
| Amplitude Damping | $$\begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix}, \begin{pmatrix}0&\sqrt{\gamma}\\0&0\end{pmatrix}$$ | Shrink to $$\|0\rangle$$ | No |
| Phase Damping | $$\begin{pmatrix}1&0\\0&\sqrt{1-\lambda}\end{pmatrix}, \begin{pmatrix}0&0\\0&\sqrt{\lambda}\end{pmatrix}$$ | Shrink to z-axis | Yes |
| Bit Flip | $$\sqrt{1-p}I, \sqrt{p}X$$ | Shrink to z-axis | Yes |

---

## 9. Python Implementation

```python
import numpy as np
from scipy import linalg

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def apply_channel(rho, kraus_ops):
    """Apply quantum channel with given Kraus operators."""
    result = np.zeros_like(rho)
    for K in kraus_ops:
        result += K @ rho @ K.conj().T
    return result

def depolarizing_kraus(p):
    """Kraus operators for depolarizing channel."""
    return [np.sqrt(1-p)*I, np.sqrt(p/3)*X, np.sqrt(p/3)*Y, np.sqrt(p/3)*Z]

def amplitude_damping_kraus(gamma):
    """Kraus operators for amplitude damping channel."""
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]

def phase_damping_kraus(lam):
    """Kraus operators for phase damping channel."""
    K0 = np.array([[1, 0], [0, np.sqrt(1-lam)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    return [K0, K1]

def choi_matrix(kraus_ops, d=2):
    """Compute Choi matrix from Kraus operators."""
    # Maximally entangled state
    phi_plus = np.zeros((d*d, d*d), dtype=complex)
    for i in range(d):
        for j in range(d):
            phi_plus[i*d + i, j*d + j] = 1/d

    # Apply E tensor I
    choi = np.zeros((d*d, d*d), dtype=complex)
    for K in kraus_ops:
        K_ext = np.kron(K, np.eye(d))
        choi += K_ext @ phi_plus @ K_ext.conj().T

    return choi

def verify_cptp(kraus_ops, d=2):
    """Verify that Kraus operators define a CPTP map."""
    # Check trace preservation: sum K†K = I
    tp_check = sum(K.conj().T @ K for K in kraus_ops)
    tp_ok = np.allclose(tp_check, np.eye(d))

    # Check complete positivity: Choi matrix >= 0
    choi = choi_matrix(kraus_ops, d)
    eigenvalues = np.linalg.eigvalsh(choi)
    cp_ok = np.all(eigenvalues >= -1e-10)

    return tp_ok, cp_ok

# Example: Amplitude damping
if __name__ == "__main__":
    gamma = 0.3
    kraus = amplitude_damping_kraus(gamma)

    # Apply to |1><1|
    rho = np.array([[0, 0], [0, 1]], dtype=complex)
    rho_out = apply_channel(rho, kraus)
    print(f"Amplitude damping on |1>:\n{rho_out}")

    # Verify CPTP
    tp, cp = verify_cptp(kraus)
    print(f"Trace-preserving: {tp}, Completely positive: {cp}")
```

---

*This review guide covers the essential theory for quantum channels. Master these concepts to complete Month 40 and prepare for the qualifying examination.*
