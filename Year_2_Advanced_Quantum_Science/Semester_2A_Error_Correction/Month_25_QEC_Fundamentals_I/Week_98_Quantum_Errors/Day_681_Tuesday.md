# Day 681: CPTP Maps and Kraus Operators

## Week 98: Quantum Errors | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Quantum Channel Theory |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Problem Solving |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational Lab |

---

## Learning Objectives

By the end of Day 681, you will be able to:

1. **Define quantum channels** as CPTP (Completely Positive Trace-Preserving) maps
2. **Derive and apply the Kraus representation** $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$
3. **Understand complete positivity** and why it's essential
4. **Work with the Choi-Jamiołkowski isomorphism** (channel-state duality)
5. **Compose quantum channels** and analyze error accumulation
6. **Connect Kraus operators to physical error processes**

---

## Quantum Channels: The Mathematical Framework

### What is a Quantum Channel?

A **quantum channel** describes how a quantum state evolves, including:
- Unitary evolution (closed systems)
- Decoherence and noise (open systems)
- Measurement processes
- Any physically realizable quantum operation

**Definition:** A quantum channel $\mathcal{E}: \mathcal{B}(\mathcal{H}_A) \rightarrow \mathcal{B}(\mathcal{H}_B)$ is a linear map satisfying:

1. **Trace-Preserving (TP):** $\text{Tr}[\mathcal{E}(\rho)] = \text{Tr}[\rho]$ for all $\rho$
2. **Completely Positive (CP):** $(\mathcal{E} \otimes \mathcal{I}_n)(\sigma) \geq 0$ for all $n$ and all $\sigma \geq 0$

The combination CPTP ensures the output is always a valid density matrix.

### Why "Completely" Positive?

**Positive but not CP maps exist!** The transpose map $T(\rho) = \rho^T$ is positive but not CP.

Consider the entangled state on two qubits:
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$\rho_{\Phi^+} = |\Phi^+\rangle\langle\Phi^+| = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

Applying transpose to the second system only:
$$(I \otimes T)(\rho_{\Phi^+}) = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

This matrix has eigenvalue $-\frac{1}{2}$ — **not a valid density matrix!**

$$\boxed{\text{Physical operations must be CPTP, not just positive}}$$

---

## The Kraus Representation

### Operator-Sum Representation

**Theorem (Kraus):** A map $\mathcal{E}$ is CPTP if and only if it can be written as:

$$\boxed{\mathcal{E}(\rho) = \sum_{k=1}^{r} E_k \rho E_k^\dagger}$$

where the **Kraus operators** $\{E_k\}$ satisfy:

$$\boxed{\sum_{k=1}^{r} E_k^\dagger E_k = I} \quad \text{(completeness relation)}$$

The number $r$ is called the **Kraus rank**.

### Physical Interpretation

The Kraus representation has a natural physical interpretation:

1. **System-environment interaction:** The system couples to an environment
2. **Environment measurement:** The environment is measured (or traced over)
3. **Conditional evolution:** Each $E_k$ corresponds to a possible outcome

$$\mathcal{E}(\rho) = \text{Tr}_E[U(\rho \otimes |e_0\rangle\langle e_0|)U^\dagger] = \sum_k \langle e_k|U|e_0\rangle \rho \langle e_0|U^\dagger|e_k\rangle = \sum_k E_k \rho E_k^\dagger$$

where $E_k = \langle e_k|U|e_0\rangle$ and $\{|e_k\rangle\}$ is an orthonormal basis for the environment.

### Non-Uniqueness of Kraus Operators

Different sets of Kraus operators can represent the same channel!

**Theorem:** $\{E_k\}$ and $\{F_j\}$ represent the same channel if and only if:

$$F_j = \sum_k U_{jk} E_k$$

for some unitary matrix $U$.

**Example:** The depolarizing channel can be written as:
- $\{E_k\} = \{\sqrt{1-p}\,I, \sqrt{p/3}\,X, \sqrt{p/3}\,Y, \sqrt{p/3}\,Z\}$
- Or in another basis with different Kraus operators

---

## Standard Kraus Representations

### 1. Unitary Channel

The simplest channel: just unitary evolution.

$$\mathcal{E}_U(\rho) = U\rho U^\dagger$$

**Kraus operators:** Single operator $E_1 = U$

**Completeness:** $U^\dagger U = I$ ✓

### 2. Bit-Flip Channel

With probability $p$, apply X; otherwise identity.

$$\mathcal{E}_{bf}(\rho) = (1-p)\rho + pX\rho X$$

**Kraus operators:**
$$E_0 = \sqrt{1-p}\,I, \quad E_1 = \sqrt{p}\,X$$

**Verify completeness:**
$$E_0^\dagger E_0 + E_1^\dagger E_1 = (1-p)I + pI = I \quad \checkmark$$

### 3. Phase-Flip Channel

With probability $p$, apply Z; otherwise identity.

$$\mathcal{E}_{pf}(\rho) = (1-p)\rho + pZ\rho Z$$

**Kraus operators:**
$$E_0 = \sqrt{1-p}\,I, \quad E_1 = \sqrt{p}\,Z$$

### 4. Depolarizing Channel

Equal probability of X, Y, or Z error.

$$\mathcal{E}_{dep}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

**Kraus operators:**
$$E_0 = \sqrt{1-p}\,I, \quad E_1 = \sqrt{\frac{p}{3}}\,X, \quad E_2 = \sqrt{\frac{p}{3}}\,Y, \quad E_3 = \sqrt{\frac{p}{3}}\,Z$$

**Alternative form:** The depolarizing channel can also be written as:

$$\mathcal{E}_{dep}(\rho) = \left(1 - \frac{4p}{3}\right)\rho + \frac{p}{3}\text{Tr}(\rho)I$$

For $p = 3/4$, this becomes the completely depolarizing channel: $\mathcal{E}(\rho) = \frac{I}{2}$.

### 5. Amplitude Damping Channel

Models energy relaxation (T₁ decay):

$$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

**Physical meaning:**
- $\gamma = 1 - e^{-t/T_1}$ for time $t$ and relaxation time $T_1$
- $E_0$: no decay occurred
- $E_1$: system decayed from $|1\rangle$ to $|0\rangle$

**Verify completeness:**
$$E_0^\dagger E_0 + E_1^\dagger E_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1-\gamma \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & \gamma \end{pmatrix} = I \quad \checkmark$$

### 6. Phase Damping (Dephasing) Channel

Models T₂ decoherence:

$$E_0 = \sqrt{1-\lambda}\,I, \quad E_1 = \sqrt{\lambda}\,|0\rangle\langle 0|, \quad E_2 = \sqrt{\lambda}\,|1\rangle\langle 1|$$

Or equivalently:
$$E_0 = \sqrt{1-\frac{\lambda}{2}}\,I, \quad E_1 = \sqrt{\frac{\lambda}{2}}\,Z$$

**Effect:** Off-diagonal elements decay: $\rho_{01} \rightarrow (1-\lambda)\rho_{01}$

---

## The Choi-Jamiołkowski Isomorphism

### Channel-State Duality

Every quantum channel $\mathcal{E}$ corresponds to a unique quantum state (the **Choi matrix**):

$$\boxed{J(\mathcal{E}) = (\mathcal{E} \otimes \mathcal{I})(|\Phi^+\rangle\langle\Phi^+|)}$$

where $|\Phi^+\rangle = \frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}|ii\rangle$ is the maximally entangled state.

**Properties:**
- $\mathcal{E}$ is CP $\iff$ $J(\mathcal{E}) \geq 0$ (positive semidefinite)
- $\mathcal{E}$ is TP $\iff$ $\text{Tr}_1[J(\mathcal{E})] = \frac{I}{d}$

### Choi Matrix from Kraus Operators

If $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$, then:

$$J(\mathcal{E}) = \sum_k |E_k\rangle\!\rangle\langle\!\langle E_k|$$

where $|E_k\rangle\!\rangle = (E_k \otimes I)|\Phi^+\rangle$ is the **vectorization** of $E_k$.

### Why Choi Matrices Matter

1. **Checking physicality:** A map is CPTP iff its Choi matrix is positive and properly normalized
2. **Comparing channels:** Distance measures on states apply to channels via Choi
3. **Error correction theory:** The Knill-Laflamme conditions use the Choi framework
4. **Quantum process tomography:** Reconstruct $J(\mathcal{E})$ experimentally

---

## Channel Composition

### Sequential Application

If $\mathcal{E}_1$ and $\mathcal{E}_2$ are channels, their composition is:

$$(\mathcal{E}_2 \circ \mathcal{E}_1)(\rho) = \mathcal{E}_2(\mathcal{E}_1(\rho))$$

**Kraus operators:** If $\mathcal{E}_1$ has Kraus ops $\{E_k\}$ and $\mathcal{E}_2$ has $\{F_j\}$, then $\mathcal{E}_2 \circ \mathcal{E}_1$ has:

$$\{F_j E_k\}_{j,k}$$

### Error Accumulation

Consider $n$ applications of depolarizing channel with parameter $p$:

$$\mathcal{E}^n(\rho) = \mathcal{E}(\mathcal{E}(\cdots\mathcal{E}(\rho)\cdots))$$

The effective error probability grows (but not simply as $np$):

$$p_{\text{eff}}(n) = \frac{3}{4}\left[1 - \left(1 - \frac{4p}{3}\right)^n\right]$$

For small $p$ and not too large $n$: $p_{\text{eff}}(n) \approx np$.

---

## Worked Examples

### Example 1: Verify Kraus Representation

**Problem:** Verify that the bit-flip Kraus operators give the correct channel.

**Solution:**

Kraus operators: $E_0 = \sqrt{1-p}\,I$, $E_1 = \sqrt{p}\,X$

$$\mathcal{E}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger$$
$$= (1-p)I\rho I + p X\rho X$$
$$= (1-p)\rho + pX\rho X \quad \checkmark$$

### Example 2: Amplitude Damping Action

**Problem:** Apply amplitude damping ($\gamma = 0.5$) to $|1\rangle\langle 1|$.

**Solution:**

$$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{0.5} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{0.5} \\ 0 & 0 \end{pmatrix}$$

$$\rho = |1\rangle\langle 1| = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

$$E_0 \rho E_0^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{0.5} \end{pmatrix}\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{0.5} \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0.5 \end{pmatrix}$$

$$E_1 \rho E_1^\dagger = \begin{pmatrix} 0 & \sqrt{0.5} \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 0 & 0 \\ \sqrt{0.5} & 0 \end{pmatrix} = \begin{pmatrix} 0.5 & 0 \\ 0 & 0 \end{pmatrix}$$

$$\mathcal{E}(\rho) = \begin{pmatrix} 0.5 & 0 \\ 0 & 0.5 \end{pmatrix}$$

Half the population has decayed from $|1\rangle$ to $|0\rangle$.

### Example 3: Choi Matrix Calculation

**Problem:** Compute the Choi matrix for the bit-flip channel with $p = 0.3$.

**Solution:**

Maximally entangled state:
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$\rho_{\Phi^+} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$

Apply $\mathcal{E}_{bf} \otimes I$:

$$J = (\mathcal{E}_{bf} \otimes I)(\rho_{\Phi^+})$$

Since $\mathcal{E}_{bf}(\rho) = 0.7\rho + 0.3 X\rho X$:

$$J = 0.7 \rho_{\Phi^+} + 0.3 (X \otimes I)\rho_{\Phi^+}(X \otimes I)$$

After calculation:
$$J = \frac{1}{2}\begin{pmatrix} 0.7 & 0 & 0 & 0.7 \\ 0 & 0.3 & 0.3 & 0 \\ 0 & 0.3 & 0.3 & 0 \\ 0.7 & 0 & 0 & 0.7 \end{pmatrix}$$

All eigenvalues are non-negative, confirming CP.

---

## Practice Problems

### Problem Set A: Direct Application

**A.1** Write Kraus operators for the Y-error channel: $\mathcal{E}(\rho) = (1-p)\rho + pY\rho Y$.

**A.2** Verify the completeness relation $\sum_k E_k^\dagger E_k = I$ for amplitude damping.

**A.3** Apply the phase-flip channel ($p = 0.2$) to $\rho = |+\rangle\langle +|$. Express the result as a matrix.

### Problem Set B: Intermediate

**B.1** Show that composition of two depolarizing channels with parameters $p_1$ and $p_2$ is depolarizing with parameter:
$$p = p_1 + p_2 - \frac{4p_1 p_2}{3}$$

**B.2** Prove that amplitude damping is not unital: $\mathcal{E}_{AD}(I/2) \neq I/2$.

**B.3** Find the Kraus operators for the generalized amplitude damping channel (thermal relaxation at finite temperature).

### Problem Set C: Challenging

**C.1** Prove that if $J(\mathcal{E}) \geq 0$ (Choi matrix positive semidefinite), then $\mathcal{E}$ is completely positive.

**C.2** The **diamond norm** distance between channels is:
$$\|\mathcal{E}_1 - \mathcal{E}_2\|_\diamond = \max_\rho \|(\mathcal{E}_1 \otimes I)(\rho) - (\mathcal{E}_2 \otimes I)(\rho)\|_1$$

For the bit-flip channel vs identity, show $\|\mathcal{E}_{bf} - I\|_\diamond = 2p$ for $p \leq 1/2$.

**C.3** Derive the condition for a channel to be **correctable** by a recovery operation $\mathcal{R}$: when does $\mathcal{R} \circ \mathcal{E} = I$ on a subspace?

---

## Computational Lab: Kraus Operators and Choi Matrices

```python
"""
Day 681 Computational Lab: CPTP Maps and Kraus Operators
========================================================

Implementing and analyzing quantum channels mathematically.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# =============================================================================
# Part 1: Basic Definitions
# =============================================================================

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def apply_channel(rho: np.ndarray, kraus_ops: List[np.ndarray]) -> np.ndarray:
    """Apply quantum channel defined by Kraus operators."""
    result = np.zeros_like(rho)
    for E in kraus_ops:
        result += E @ rho @ E.conj().T
    return result

def verify_completeness(kraus_ops: List[np.ndarray]) -> Tuple[bool, np.ndarray]:
    """Verify that Kraus operators satisfy completeness relation."""
    d = kraus_ops[0].shape[0]
    sum_EE = np.zeros((d, d), dtype=complex)
    for E in kraus_ops:
        sum_EE += E.conj().T @ E
    is_complete = np.allclose(sum_EE, np.eye(d))
    return is_complete, sum_EE

print("=" * 60)
print("PART 1: Kraus Operator Definitions")
print("=" * 60)

# =============================================================================
# Part 2: Standard Channels
# =============================================================================

def bit_flip_channel(p: float) -> List[np.ndarray]:
    """Return Kraus operators for bit-flip channel."""
    return [np.sqrt(1-p) * I, np.sqrt(p) * X]

def phase_flip_channel(p: float) -> List[np.ndarray]:
    """Return Kraus operators for phase-flip channel."""
    return [np.sqrt(1-p) * I, np.sqrt(p) * Z]

def depolarizing_channel(p: float) -> List[np.ndarray]:
    """Return Kraus operators for depolarizing channel."""
    return [np.sqrt(1-p) * I,
            np.sqrt(p/3) * X,
            np.sqrt(p/3) * Y,
            np.sqrt(p/3) * Z]

def amplitude_damping_channel(gamma: float) -> List[np.ndarray]:
    """Return Kraus operators for amplitude damping."""
    E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [E0, E1]

def phase_damping_channel(lam: float) -> List[np.ndarray]:
    """Return Kraus operators for phase damping."""
    E0 = np.sqrt(1-lam) * I
    E1 = np.sqrt(lam) * np.array([[1, 0], [0, 0]], dtype=complex)
    E2 = np.sqrt(lam) * np.array([[0, 0], [0, 1]], dtype=complex)
    return [E0, E1, E2]

print("\nVerifying completeness for standard channels:")
channels = {
    'Bit-flip (p=0.1)': bit_flip_channel(0.1),
    'Phase-flip (p=0.2)': phase_flip_channel(0.2),
    'Depolarizing (p=0.15)': depolarizing_channel(0.15),
    'Amplitude damping (γ=0.3)': amplitude_damping_channel(0.3),
    'Phase damping (λ=0.25)': phase_damping_channel(0.25)
}

for name, kraus_ops in channels.items():
    is_complete, _ = verify_completeness(kraus_ops)
    print(f"  {name}: {'✓' if is_complete else '✗'}")

# =============================================================================
# Part 3: Channel Action on States
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Channel Action on Quantum States")
print("=" * 60)

# Define test states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

rho_0 = np.outer(ket_0, ket_0.conj())
rho_1 = np.outer(ket_1, ket_1.conj())
rho_plus = np.outer(ket_plus, ket_plus.conj())

print("\nAmplitude damping (γ=0.5) on |1⟩:")
AD = amplitude_damping_channel(0.5)
rho_out = apply_channel(rho_1, AD)
print(f"  Input:  {rho_1}")
print(f"  Output: {np.round(rho_out, 4)}")
print(f"  Population shift: P(0) = {rho_out[0,0].real:.3f}, P(1) = {rho_out[1,1].real:.3f}")

print("\nPhase damping (λ=0.5) on |+⟩:")
PD = phase_damping_channel(0.5)
rho_out = apply_channel(rho_plus, PD)
print(f"  Input:  {np.round(rho_plus, 4)}")
print(f"  Output: {np.round(rho_out, 4)}")
print(f"  Coherence decay: off-diagonal {rho_plus[0,1]:.3f} → {rho_out[0,1]:.3f}")

# =============================================================================
# Part 4: Choi-Jamiołkowski Isomorphism
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Choi Matrix Computation")
print("=" * 60)

def compute_choi_matrix(kraus_ops: List[np.ndarray]) -> np.ndarray:
    """
    Compute Choi matrix J(E) = (E ⊗ I)(|Φ+⟩⟨Φ+|)
    where |Φ+⟩ = (|00⟩ + |11⟩)/√2
    """
    d = kraus_ops[0].shape[0]

    # Maximally entangled state |Φ+⟩
    phi_plus = np.zeros((d*d,), dtype=complex)
    for i in range(d):
        phi_plus[i*d + i] = 1/np.sqrt(d)

    rho_phi = np.outer(phi_plus, phi_plus.conj())

    # Apply channel to first system: (E ⊗ I)
    choi = np.zeros((d*d, d*d), dtype=complex)
    for E in kraus_ops:
        E_tensor_I = np.kron(E, I)
        choi += E_tensor_I @ rho_phi @ E_tensor_I.conj().T

    return choi

def check_choi_properties(choi: np.ndarray, d: int = 2) -> dict:
    """Check CP and TP conditions from Choi matrix."""
    # CP: Choi matrix should be positive semidefinite
    eigenvalues = np.linalg.eigvalsh(choi)
    is_CP = all(eigenvalues >= -1e-10)

    # TP: Partial trace over first system should be I/d
    partial_trace = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                partial_trace[i, j] += choi[i*d + k, j*d + k]
    is_TP = np.allclose(partial_trace, np.eye(d)/d * d)  # Factor of d from normalization

    return {
        'eigenvalues': eigenvalues,
        'is_CP': is_CP,
        'partial_trace': partial_trace,
        'is_TP': is_TP
    }

# Compute Choi matrices
print("\nChoi matrix analysis:")
for name, kraus_ops in list(channels.items())[:3]:
    choi = compute_choi_matrix(kraus_ops)
    props = check_choi_properties(choi)
    print(f"\n{name}:")
    print(f"  Eigenvalues: {np.round(props['eigenvalues'], 4)}")
    print(f"  CP: {props['is_CP']}, TP: {props['is_TP']}")

# =============================================================================
# Part 5: Channel Composition
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Channel Composition")
print("=" * 60)

def compose_channels(kraus1: List[np.ndarray], kraus2: List[np.ndarray]) -> List[np.ndarray]:
    """Compose two channels: E2 ∘ E1 (E1 first, then E2)."""
    composed = []
    for E2 in kraus2:
        for E1 in kraus1:
            composed.append(E2 @ E1)
    return composed

# Compose two depolarizing channels
p1, p2 = 0.1, 0.15
dep1 = depolarizing_channel(p1)
dep2 = depolarizing_channel(p2)
composed = compose_channels(dep1, dep2)

print(f"\nComposing depolarizing channels (p1={p1}, p2={p2}):")
print(f"  Number of Kraus operators: {len(dep1)} × {len(dep2)} = {len(composed)}")

# Verify completeness
is_complete, _ = verify_completeness(composed)
print(f"  Composed channel completeness: {'✓' if is_complete else '✗'}")

# Theoretical effective parameter
p_eff_theory = p1 + p2 - 4*p1*p2/3
print(f"\n  Theoretical effective p: {p_eff_theory:.6f}")

# Test by applying to maximally mixed state
rho_mixed = I / 2
out1 = apply_channel(rho_mixed, dep1)
out12 = apply_channel(out1, dep2)
out_composed = apply_channel(rho_mixed, composed)
print(f"  Sequential application matches composition: {np.allclose(out12, out_composed)}")

# =============================================================================
# Part 6: Error Accumulation Analysis
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Error Accumulation")
print("=" * 60)

def effective_depolarizing_parameter(p: float, n: int) -> float:
    """Effective error parameter after n applications."""
    return 0.75 * (1 - (1 - 4*p/3)**n)

p = 0.01  # Low error rate
n_values = np.arange(1, 101)
p_linear = p * n_values  # Linear approximation
p_actual = [effective_depolarizing_parameter(p, n) for n in n_values]

plt.figure(figsize=(12, 5))

# Plot 1: Error accumulation
plt.subplot(1, 2, 1)
plt.plot(n_values, p_linear, 'b--', label='Linear approx (np)', alpha=0.7)
plt.plot(n_values, p_actual, 'r-', label='Actual', linewidth=2)
plt.axhline(0.75, color='gray', linestyle=':', label='Max (0.75)')
plt.xlabel('Number of Channel Applications (n)')
plt.ylabel('Effective Error Parameter')
plt.title(f'Error Accumulation (p = {p})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Bloch sphere shrinking
plt.subplot(1, 2, 2)
# The Bloch vector shrinks by factor (1-4p/3) each application
shrink_factors = [(1 - 4*p/3)**n for n in n_values]
plt.plot(n_values, shrink_factors, 'g-', linewidth=2)
plt.xlabel('Number of Channel Applications (n)')
plt.ylabel('Bloch Vector Length')
plt.title('Bloch Sphere Shrinking Under Depolarizing')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_681_error_accumulation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_681_error_accumulation.png")

# =============================================================================
# Part 7: Non-Unital Channel Example
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Unital vs Non-Unital Channels")
print("=" * 60)

def is_unital(kraus_ops: List[np.ndarray]) -> Tuple[bool, np.ndarray]:
    """Check if channel is unital: E(I) = I."""
    d = kraus_ops[0].shape[0]
    identity = np.eye(d, dtype=complex)
    output = apply_channel(identity, kraus_ops)
    is_unital = np.allclose(output, identity)
    return is_unital, output

print("\nUnital check (channel preserves identity):")
for name, kraus_ops in channels.items():
    unital, output = is_unital(kraus_ops)
    status = "Unital ✓" if unital else "Non-unital ✗"
    print(f"  {name}: {status}")
    if not unital:
        print(f"    E(I) = \n{np.round(output, 4)}")

# =============================================================================
# Part 8: Summary Visualization
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: CPTP Maps and Kraus Representation")
print("=" * 60)

summary = """
┌────────────────────────────────────────────────────────────────────┐
│                  CPTP Maps: Key Properties                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Kraus Representation: E(ρ) = Σₖ Eₖ ρ Eₖ†                           │
│                                                                     │
│ Completeness:         Σₖ Eₖ† Eₖ = I                                 │
│                                                                     │
│ Choi Matrix:          J(E) = (E ⊗ I)(|Φ⁺⟩⟨Φ⁺|)                     │
│                                                                     │
│ CP ⟺ J(E) ≥ 0        (positive semidefinite)                       │
│ TP ⟺ Tr₁[J(E)] = I/d (proper normalization)                        │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│ Channel Type         │ Unital? │ Physical Origin                    │
├──────────────────────┼─────────┼───────────────────────────────────│
│ Depolarizing         │   Yes   │ Random Pauli errors                │
│ Bit-flip             │   Yes   │ Transverse noise                   │
│ Phase-flip           │   Yes   │ Longitudinal noise                 │
│ Amplitude damping    │   No    │ T₁ relaxation (energy loss)        │
│ Phase damping        │   Yes   │ T₂ dephasing (pure decoherence)    │
└────────────────────────────────────────────────────────────────────┘
"""
print(summary)

print("\n✅ Day 681 Lab Complete!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Kraus representation | $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$ |
| Completeness relation | $\sum_k E_k^\dagger E_k = I$ |
| Choi matrix | $J(\mathcal{E}) = (\mathcal{E} \otimes \mathcal{I})(\|\Phi^+\rangle\langle\Phi^+\|)$ |
| CP condition | $J(\mathcal{E}) \geq 0$ |
| TP condition | $\text{Tr}_1[J(\mathcal{E})] = I/d$ |
| Depolarizing composition | $p_{\text{eff}} = p_1 + p_2 - \frac{4p_1 p_2}{3}$ |

### Standard Channels Summary

| Channel | Kraus Operators | Effect |
|---------|-----------------|--------|
| Bit-flip | $\sqrt{1-p}I, \sqrt{p}X$ | Flips $\|0\rangle \leftrightarrow \|1\rangle$ |
| Phase-flip | $\sqrt{1-p}I, \sqrt{p}Z$ | Introduces $-1$ phase on $\|1\rangle$ |
| Depolarizing | $\sqrt{1-p}I, \sqrt{p/3}X, \sqrt{p/3}Y, \sqrt{p/3}Z$ | Equal probability each Pauli |
| Amplitude damping | See above | $\|1\rangle \rightarrow \|0\rangle$ decay |
| Phase damping | See above | Off-diagonal decay |

### Main Takeaways

1. **CPTP = Physically realizable:** Complete positivity ensures valid outputs on entangled states
2. **Kraus representation:** Any CPTP map can be written with Kraus operators
3. **Choi-Jamiołkowski:** Channel-state duality connects channels to matrices
4. **Kraus non-uniqueness:** Different Kraus sets can represent the same channel
5. **Error accumulation:** Errors compound under channel composition

---

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain why complete positivity (not just positivity) is required
- [ ] I understand the physical interpretation of Kraus operators
- [ ] I can distinguish unital from non-unital channels
- [ ] I understand the Choi-Jamiołkowski isomorphism

### Mathematical Skills
- [ ] I can write Kraus operators for standard channels
- [ ] I can verify the completeness relation
- [ ] I can compute Choi matrices
- [ ] I can compose channels via Kraus operators

### Computational Skills
- [ ] I can implement arbitrary CPTP maps numerically
- [ ] I can verify CP and TP from the Choi matrix
- [ ] I can analyze error accumulation under composition

---

## Preview: Day 682

Tomorrow we dive deep into specific noise models:
- **Depolarizing channel:** Detailed analysis and parameter meaning
- **Amplitude damping:** T₁ relaxation physics
- **Generalized amplitude damping:** Finite temperature effects
- **Combined noise models:** Realistic error processes

---

*"The Kraus representation reveals that every quantum operation is an interaction with an environment that we trace away."*
— Michael Nielsen

---

**Day 681 Complete!** Week 98: 2/7 days (29%)
