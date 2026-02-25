# Day 648: Unitary Freedom in Kraus Representations

## Week 93: Channel Representations | Month 24: Quantum Channels & Error Introduction

---

## Schedule Overview

| Session | Time | Topic |
|---------|------|-------|
| **Morning** | 3 hours | Equivalent Kraus representations, unitary mixing theorem |
| **Afternoon** | 2.5 hours | Applications and canonical forms |
| **Evening** | 1.5 hours | Computational lab: exploring equivalent representations |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Prove** that Kraus representations are not unique for a given channel
2. **State** the unitary freedom theorem relating equivalent Kraus sets
3. **Construct** alternative Kraus representations using unitary transformations
4. **Identify** when two Kraus sets describe the same channel
5. **Apply** unitary freedom to simplify or modify Kraus representations
6. **Connect** unitary freedom to measurement interpretations and error correction

---

## Core Content

### 1. Non-Uniqueness of Kraus Representations

We've seen that every quantum channel has a Kraus representation:
$$\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger$$

**Key Question:** Is this representation unique?

**Answer:** No! Many different sets of Kraus operators can produce the same channel.

### 2. The Unitary Freedom Theorem

**Theorem:** Two sets of Kraus operators $\{K_k\}_{k=1}^r$ and $\{L_j\}_{j=1}^s$ represent the same quantum channel if and only if:

$$\boxed{L_j = \sum_{k=1}^r U_{jk} K_k}$$

where $U$ is a **unitary matrix** (or **isometry** if $s > r$) with $U^\dagger U = I_r$.

**In matrix notation:** If we stack Kraus operators as columns:
$$\vec{L} = U \cdot \vec{K}$$

### 3. Proof Sketch

**Direction 1: If Kraus sets are related by unitary, they give the same channel**

$$\mathcal{E}_L(\rho) = \sum_j L_j \rho L_j^\dagger = \sum_j \left(\sum_k U_{jk} K_k\right) \rho \left(\sum_l U_{jl}^* K_l^\dagger\right)$$

$$= \sum_{k,l} \left(\sum_j U_{jk} U_{jl}^*\right) K_k \rho K_l^\dagger = \sum_{k,l} \delta_{kl} K_k \rho K_l^\dagger = \sum_k K_k \rho K_k^\dagger = \mathcal{E}_K(\rho)$$

**Direction 2: If channels are equal, Kraus sets are related by unitary**

This follows from the Stinespring dilation uniqueness. Two dilations for the same channel differ by a unitary on the environment, which induces a unitary on the Kraus operators.

### 4. Examples of Equivalent Representations

#### Example 1: Depolarizing Channel

**Standard representation:**
$$K_0 = \sqrt{1-\frac{3p}{4}}I, \quad K_1 = \frac{\sqrt{p}}{2}X, \quad K_2 = \frac{\sqrt{p}}{2}Y, \quad K_3 = \frac{\sqrt{p}}{2}Z$$

**Alternative representation (Hadamard mixing):**

Apply the Hadamard-like unitary:
$$U = \frac{1}{2}\begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & -1 & 1 & -1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & -1 & 1 \end{pmatrix}$$

This gives new Kraus operators that are linear combinations of $I, X, Y, Z$.

#### Example 2: Bit-Flip Channel

**Standard representation:**
$$K_0 = \sqrt{1-p} \cdot I, \quad K_1 = \sqrt{p} \cdot X$$

**Alternative representation (2×2 unitary):**

For any $\theta$, define:
$$U = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}$$

New Kraus operators:
$$L_0 = \cos\theta \cdot K_0 + \sin\theta \cdot K_1 = \cos\theta\sqrt{1-p} \cdot I + \sin\theta\sqrt{p} \cdot X$$
$$L_1 = -\sin\theta \cdot K_0 + \cos\theta \cdot K_1 = -\sin\theta\sqrt{1-p} \cdot I + \cos\theta\sqrt{p} \cdot X$$

Both $\{K_0, K_1\}$ and $\{L_0, L_1\}$ describe the same bit-flip channel!

### 5. Padding with Zeros

**Important case:** We can always add Kraus operators that are zero:

If $\{K_k\}_{k=1}^r$ is a valid Kraus representation, so is $\{K_1, ..., K_r, 0, ..., 0\}$ with any number of zero operators added.

**Why this matters:**
- Allows comparing Kraus sets of different sizes
- The relevant dimension is the Kraus rank (non-zero operators)
- The unitary $U$ can be rectangular (isometry) when sizes differ

### 6. Canonical Forms

**Minimal Kraus representation:** Fewest possible Kraus operators (equals Kraus rank)

**Canonical choices:**
1. **Eigenoperator basis:** Kraus operators from eigendecomposition of Choi matrix
2. **Orthogonal Kraus operators:** $\text{Tr}(K_i^\dagger K_j) = \delta_{ij} \lambda_i$
3. **Pauli basis:** For qubit channels, express in terms of $I, X, Y, Z$

### 7. Connection to Measurements

The unitary freedom has a beautiful measurement interpretation:

**Measurement picture:** Kraus operators correspond to measurement outcomes
- $K_k$ is the operation for outcome $k$
- Probability: $p_k = \text{Tr}(K_k \rho K_k^\dagger)$

**Unitary freedom:** Different "ways of measuring the environment" give different Kraus sets!

If the Stinespring environment is measured in basis $\{|k\rangle\}$: get Kraus operators $K_k$
If measured in rotated basis $\{|j'\rangle = \sum_k U_{jk}^*|k\rangle\}$: get $L_j = \sum_k U_{jk} K_k$

**Physical insight:** The same channel can arise from different environment measurements!

### 8. Implications for Error Correction

**Error model ambiguity:** A noise channel doesn't specify unique error operators!

For bit-flip channel:
- Standard: errors are $\{I, X\}$ with probabilities $\{1-p, p\}$
- Alternative: errors could be $\{L_0, L_1\}$ with different probabilities

**Good news for QEC:** Error correction codes are designed to handle error spaces, not specific error operators. The stabilizer formalism naturally accommodates this freedom.

### 9. Trace-Preserving Condition Under Mixing

If $\{K_k\}$ satisfies $\sum_k K_k^\dagger K_k = I$, does $\{L_j\}$ also satisfy this?

$$\sum_j L_j^\dagger L_j = \sum_j \left(\sum_k U_{jk}^* K_k^\dagger\right)\left(\sum_l U_{jl} K_l\right)$$
$$= \sum_{k,l} \left(\sum_j U_{jk}^* U_{jl}\right) K_k^\dagger K_l = \sum_{k,l} \delta_{kl} K_k^\dagger K_l = \sum_k K_k^\dagger K_k = I$$

**Yes!** Unitary mixing preserves the trace-preserving condition.

### 10. Characterizing the Equivalence

**Two Kraus sets are equivalent if and only if:**
1. They give the same Choi matrix: $J_{\{K\}} = J_{\{L\}}$
2. They have the same action on all density matrices
3. They are related by a unitary/isometry transformation

**Testing equivalence:** Compute Choi matrices and compare!

---

## Quantum Computing Connection

### Gate Decomposition Flexibility

When implementing a channel (intentionally or as noise), the unitary freedom suggests multiple circuit implementations:

**Same channel, different circuits:**
- Different ancilla measurements
- Different intermediate states
- Different noise distributions

### Error Mitigation

Understanding unitary freedom helps in error mitigation:
- Model noise with convenient Kraus operators
- Choose representation suited to correction strategy
- Exploit symmetries in the Kraus representation

### Randomized Compiling

**Pauli twirling** exploits unitary freedom to convert arbitrary noise to Pauli noise:
$$\mathcal{E}_{\text{twirled}}(\rho) = \frac{1}{4}\sum_P P^\dagger \mathcal{E}(P\rho P^\dagger) P$$

This produces a Pauli channel, which is easier to analyze and correct.

---

## Worked Examples

### Example 1: Verifying Equivalent Representations

**Problem:** Show that the following two Kraus sets represent the same channel:

Set A: $K_0 = \frac{1}{\sqrt{2}}I$, $K_1 = \frac{1}{\sqrt{2}}Z$

Set B: $L_0 = \frac{1}{\sqrt{2}}|0\rangle\langle 0|$, $L_1 = \frac{1}{\sqrt{2}}|1\rangle\langle 1|$, $L_2 = \frac{1}{\sqrt{2}}|0\rangle\langle 0|$, $L_3 = -\frac{1}{\sqrt{2}}|1\rangle\langle 1|$

**Solution:**

Method 1: Compute both channels on a general state

Set A channel:
$$\mathcal{E}_A(\rho) = \frac{1}{2}I\rho I + \frac{1}{2}Z\rho Z = \frac{1}{2}(\rho + Z\rho Z)$$

For $\rho = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:
$$Z\rho Z = \begin{pmatrix} a & -b \\ -c & d \end{pmatrix}$$
$$\mathcal{E}_A(\rho) = \begin{pmatrix} a & 0 \\ 0 & d \end{pmatrix}$$

Set B channel:
Note that $L_0 + L_2 = \sqrt{2}|0\rangle\langle 0|$, $L_1 + L_3 = 0$, $L_0 - L_2 = 0$, $L_1 - L_3 = \sqrt{2}|1\rangle\langle 1|$

Actually, let me compute directly:
$$\mathcal{E}_B(\rho) = L_0\rho L_0^\dagger + L_1\rho L_1^\dagger + L_2\rho L_2^\dagger + L_3\rho L_3^\dagger$$
$$= \frac{1}{2}|0\rangle\langle 0|\rho|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|\rho|1\rangle\langle 1| + \frac{1}{2}|0\rangle\langle 0|\rho|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1|\rho|1\rangle\langle 1|$$
$$= |0\rangle\langle 0|\rho|0\rangle\langle 0| + |1\rangle\langle 1|\rho|1\rangle\langle 1| = \rho_{00}|0\rangle\langle 0| + \rho_{11}|1\rangle\langle 1| = \begin{pmatrix} a & 0 \\ 0 & d \end{pmatrix}$$

Both channels give the same result: complete dephasing! ✓

Method 2: Find the unitary relating them

Set B has 4 operators but rank 2 (since $L_0 = L_2$ and $L_1 = -L_3$). After reducing:
$$\tilde{L}_0 = \frac{1}{\sqrt{2}}(L_0 + L_2) = |0\rangle\langle 0|/\sqrt{2}$$
Wait, this doesn't match. Let me reconsider.

The unitary relating $\{K_0, K_1\}$ to $\{L_0', L_1'\}$ where $L_0' = |0\rangle\langle 0|$ and $L_1' = |1\rangle\langle 1|$ is:
$$\begin{pmatrix} |0\rangle\langle 0| \\ |1\rangle\langle 1| \end{pmatrix} = U \begin{pmatrix} I/\sqrt{2} \\ Z/\sqrt{2} \end{pmatrix}$$

Since $I = |0\rangle\langle 0| + |1\rangle\langle 1|$ and $Z = |0\rangle\langle 0| - |1\rangle\langle 1|$:
$$|0\rangle\langle 0| = \frac{I+Z}{2}, \quad |1\rangle\langle 1| = \frac{I-Z}{2}$$

So: $L_0' = \frac{1}{\sqrt{2}}K_0 + \frac{1}{\sqrt{2}}K_1$ and $L_1' = \frac{1}{\sqrt{2}}K_0 - \frac{1}{\sqrt{2}}K_1$

The unitary is $U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ (Hadamard)! ✓

---

### Example 2: Finding All Equivalent Representations

**Problem:** For the amplitude damping channel with $\gamma = 0.5$, find a different Kraus representation using a rotation by angle $\theta = \pi/4$.

**Solution:**

Original Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \frac{1}{\sqrt{2}} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \frac{1}{\sqrt{2}} \\ 0 & 0 \end{pmatrix}$$

Rotation unitary:
$$U = \begin{pmatrix} \cos(\pi/4) & \sin(\pi/4) \\ -\sin(\pi/4) & \cos(\pi/4) \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$$

New Kraus operators:
$$L_0 = \frac{1}{\sqrt{2}}(K_0 + K_1) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & \frac{1}{\sqrt{2}} \\ 0 & \frac{1}{\sqrt{2}} \end{pmatrix}$$

$$L_1 = \frac{1}{\sqrt{2}}(-K_0 + K_1) = \frac{1}{\sqrt{2}}\begin{pmatrix} -1 & \frac{1}{\sqrt{2}} \\ 0 & -\frac{1}{\sqrt{2}} \end{pmatrix}$$

**Verification:** Check $L_0^\dagger L_0 + L_1^\dagger L_1 = I$

$$L_0^\dagger L_0 = \frac{1}{2}\begin{pmatrix} 1 & 0 \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}\begin{pmatrix} 1 & \frac{1}{\sqrt{2}} \\ 0 & \frac{1}{\sqrt{2}} \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & 1 \end{pmatrix}$$

$$L_1^\dagger L_1 = \frac{1}{2}\begin{pmatrix} 1 & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & 1 \end{pmatrix}$$

$$L_0^\dagger L_0 + L_1^\dagger L_1 = \begin{pmatrix} 1 & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & 1 \end{pmatrix} \neq I$$

Hmm, this doesn't equal $I$. Let me check the original:
$$K_0^\dagger K_0 + K_1^\dagger K_1 = \begin{pmatrix} 1 & 0 \\ 0 & \frac{1}{2} \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & \frac{1}{2} \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I ✓$$

The sum should be preserved. Let me recalculate $L_0^\dagger L_0$:

$$L_0 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1/\sqrt{2} \\ 0 & 1/\sqrt{2} \end{pmatrix}$$

$$L_0^\dagger = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 \\ 1/\sqrt{2} & 1/\sqrt{2} \end{pmatrix}$$

$$L_0^\dagger L_0 = \frac{1}{2}\begin{pmatrix} 1 & 1/\sqrt{2} \\ 1/\sqrt{2} & 1 \end{pmatrix}$$

Similarly $L_1^\dagger L_1 = \frac{1}{2}\begin{pmatrix} 1 & -1/\sqrt{2} \\ -1/\sqrt{2} & 1 \end{pmatrix}$

Sum: $L_0^\dagger L_0 + L_1^\dagger L_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I$ ✓

Great, it works! The new representation $\boxed{\{L_0, L_1\}}$ describes the same amplitude damping channel.

---

### Example 3: From Environment Measurement Basis

**Problem:** The amplitude damping Stinespring unitary is measured in the $\{|+\rangle, |-\rangle\}$ basis instead of $\{|0\rangle, |1\rangle\}$ on the environment. What are the resulting Kraus operators?

**Solution:**

Original Kraus operators (environment measured in $\{|0\rangle, |1\rangle\}$):
$$K_0 = \langle 0|_E U |0\rangle_E, \quad K_1 = \langle 1|_E U |0\rangle_E$$

New basis: $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$

New Kraus operators:
$$L_+ = \langle +|_E U |0\rangle_E = \frac{1}{\sqrt{2}}(\langle 0| + \langle 1|)_E U |0\rangle_E = \frac{1}{\sqrt{2}}(K_0 + K_1)$$
$$L_- = \langle -|_E U |0\rangle_E = \frac{1}{\sqrt{2}}(K_0 - K_1)$$

This is exactly a Hadamard transformation on the Kraus operators!

For amplitude damping with $\gamma$:
$$L_+ = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & \sqrt{\gamma} \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$$
$$L_- = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -\sqrt{\gamma} \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$$

---

## Practice Problems

### Direct Application

1. **Problem 1:** Show that $\{I/\sqrt{2}, X/\sqrt{2}\}$ and $\{(I+X)/2, (I-X)/2\}$ represent the same channel.

2. **Problem 2:** For the phase-flip channel with $p=0.25$, find an equivalent Kraus representation using a $45°$ rotation.

3. **Problem 3:** Verify that adding zero operators doesn't change a channel: show $\{K_0, K_1\}$ and $\{K_0, K_1, 0\}$ give identical Choi matrices.

### Intermediate

4. **Problem 4:** Find all 2×2 unitary matrices $U$ such that $U \cdot (K_0, K_1)^T$ gives Kraus operators that are proportional to Pauli matrices.

5. **Problem 5:** For the completely dephasing channel $\mathcal{E}(\rho) = |0\rangle\langle 0|\rho|0\rangle\langle 0| + |1\rangle\langle 1|\rho|1\rangle\langle 1|$, find two different Kraus representations with 2 operators each.

6. **Problem 6:** Prove that if $\{K_k\}$ are mutually orthogonal ($\text{Tr}(K_i^\dagger K_j) = 0$ for $i \neq j$), then any equivalent Kraus representation $\{L_j = \sum_k U_{jk} K_k\}$ has $\text{Tr}(L_i^\dagger L_j) = 0$ for $i \neq j$.

### Challenging

7. **Problem 7:** Prove that the Kraus rank is invariant under unitary freedom transformations.

8. **Problem 8:** Show that a channel is unitary if and only if it has a unique Kraus representation (up to global phase).

9. **Problem 9:** For the depolarizing channel, find a Kraus representation where all four operators have equal trace norm.

---

## Computational Lab

```python
"""
Day 648 Computational Lab: Unitary Freedom in Kraus Representations
==================================================================
Topics: Equivalent representations, unitary mixing, canonical forms
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


def apply_channel(rho: np.ndarray, kraus_ops: List[np.ndarray]) -> np.ndarray:
    """Apply quantum channel to density matrix."""
    return sum(K @ rho @ K.conj().T for K in kraus_ops)


def kraus_to_choi(kraus_ops: List[np.ndarray]) -> np.ndarray:
    """Compute Choi matrix from Kraus operators."""
    d = kraus_ops[0].shape[0]
    choi = np.zeros((d * d, d * d), dtype=complex)
    for K in kraus_ops:
        vec_K = K.flatten('F').reshape(-1, 1)
        choi += vec_K @ vec_K.conj().T
    return choi


def mix_kraus_operators(kraus_ops: List[np.ndarray],
                        U: np.ndarray) -> List[np.ndarray]:
    """
    Apply unitary mixing to Kraus operators.

    L_j = Σ_k U_{jk} K_k
    """
    n_orig = len(kraus_ops)
    n_new = U.shape[0]

    # Pad with zeros if needed
    d = kraus_ops[0].shape[0]
    padded_kraus = kraus_ops + [np.zeros((d, d), dtype=complex)] * (n_new - n_orig)

    new_kraus = []
    for j in range(n_new):
        L_j = np.zeros_like(kraus_ops[0])
        for k in range(len(padded_kraus)):
            if k < U.shape[1]:
                L_j += U[j, k] * padded_kraus[k]
        new_kraus.append(L_j)

    return new_kraus


def verify_completeness(kraus_ops: List[np.ndarray], tol: float = 1e-10) -> bool:
    """Verify Σ K†K = I."""
    d = kraus_ops[0].shape[0]
    sum_kdk = np.zeros((d, d), dtype=complex)
    for K in kraus_ops:
        sum_kdk += K.conj().T @ K
    return np.allclose(sum_kdk, np.eye(d), atol=tol)


def channels_equal(kraus1: List[np.ndarray], kraus2: List[np.ndarray],
                   n_test: int = 100, tol: float = 1e-10) -> bool:
    """Test if two Kraus sets give the same channel by sampling."""
    d = kraus1[0].shape[0]

    for _ in range(n_test):
        # Random density matrix
        psi = np.random.randn(d) + 1j * np.random.randn(d)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())

        rho1 = apply_channel(rho, kraus1)
        rho2 = apply_channel(rho, kraus2)

        if np.max(np.abs(rho1 - rho2)) > tol:
            return False

    return True


# ===== DEMONSTRATIONS =====

print("=" * 70)
print("PART 1: Verifying Unitary Freedom")
print("=" * 70)

# Bit-flip channel
p = 0.2
K0 = np.sqrt(1 - p) * I
K1 = np.sqrt(p) * X
kraus_bf = [K0, K1]

print(f"\nBit-flip channel (p={p}):")
print("Original Kraus operators:")
print(f"K0 = √{1-p:.1f} I =\n{K0}")
print(f"K1 = √{p:.1f} X =\n{K1}")
print(f"Completeness: {verify_completeness(kraus_bf)}")

# Apply rotation mixing
theta = np.pi / 6  # 30 degrees
U_rot = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]
], dtype=complex)

kraus_bf_rotated = mix_kraus_operators(kraus_bf, U_rot)

print(f"\nRotated by θ={np.degrees(theta):.0f}°:")
print(f"L0 =\n{kraus_bf_rotated[0]}")
print(f"L1 =\n{kraus_bf_rotated[1]}")
print(f"Completeness: {verify_completeness(kraus_bf_rotated)}")
print(f"Same channel: {channels_equal(kraus_bf, kraus_bf_rotated)}")


print("\n" + "=" * 70)
print("PART 2: Choi Matrix Invariance")
print("=" * 70)

# Compute Choi matrices for both representations
choi_original = kraus_to_choi(kraus_bf)
choi_rotated = kraus_to_choi(kraus_bf_rotated)

print("\nChoi matrix (original):")
print(np.array2string(choi_original, precision=4))

print("\nChoi matrix (rotated):")
print(np.array2string(choi_rotated, precision=4))

print(f"\nChoi matrices equal: {np.allclose(choi_original, choi_rotated)}")


print("\n" + "=" * 70)
print("PART 3: Exploring Parameter Space of Equivalent Representations")
print("=" * 70)

def visualize_kraus_norms(kraus_original: List[np.ndarray], n_angles: int = 50):
    """
    For a 2-Kraus channel, visualize how operator norms vary with rotation angle.
    """
    angles = np.linspace(0, 2*np.pi, n_angles)
    norms_0 = []
    norms_1 = []

    for theta in angles:
        U = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ], dtype=complex)

        rotated = mix_kraus_operators(kraus_original, U)
        norms_0.append(np.linalg.norm(rotated[0], 'fro'))
        norms_1.append(np.linalg.norm(rotated[1], 'fro'))

    plt.figure(figsize=(10, 6))
    plt.plot(np.degrees(angles), norms_0, 'b-', linewidth=2, label='||L₀||')
    plt.plot(np.degrees(angles), norms_1, 'r-', linewidth=2, label='||L₁||')
    plt.xlabel('Rotation angle θ (degrees)')
    plt.ylabel('Frobenius norm')
    plt.title('Kraus Operator Norms Under Unitary Mixing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('kraus_norms_rotation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: kraus_norms_rotation.png")

print("\nVisualizing Kraus norms for bit-flip channel under rotation:")
visualize_kraus_norms(kraus_bf)


print("\n" + "=" * 70)
print("PART 4: Hadamard Mixing for Dephasing Channel")
print("=" * 70)

# Dephasing channel: E(ρ) = (1-p)ρ + p Z ρ Z
p_deph = 0.3
K0_deph = np.sqrt(1 - p_deph) * I
K1_deph = np.sqrt(p_deph) * Z
kraus_deph = [K0_deph, K1_deph]

print(f"Dephasing channel (p={p_deph}):")
print("Original representation: {√(1-p) I, √p Z}")

# Apply Hadamard mixing
H_mix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
kraus_deph_hadamard = mix_kraus_operators(kraus_deph, H_mix)

print("\nAfter Hadamard mixing:")
print(f"L0 = (K0 + K1)/√2 =\n{kraus_deph_hadamard[0]}")
print(f"L1 = (K0 - K1)/√2 =\n{kraus_deph_hadamard[1]}")

# Simplify
print("\nSimplified:")
print("L0 ∝ |0⟩⟨0| + √(1-2p+2p)|0⟩⟨0| = ... (projection-like)")
print(f"Same channel: {channels_equal(kraus_deph, kraus_deph_hadamard)}")


print("\n" + "=" * 70)
print("PART 5: Depolarizing Channel - Multiple Representations")
print("=" * 70)

def depolarizing_kraus(p: float) -> List[np.ndarray]:
    """Standard depolarizing channel Kraus operators."""
    return [
        np.sqrt(1 - 3*p/4) * I,
        np.sqrt(p/4) * X,
        np.sqrt(p/4) * Y,
        np.sqrt(p/4) * Z
    ]

p_dep = 0.4
kraus_dep = depolarizing_kraus(p_dep)

print(f"Depolarizing channel (p={p_dep}):")
print("Standard representation: {√(1-3p/4) I, √(p/4) X, √(p/4) Y, √(p/4) Z}")

# Apply a 4x4 unitary mixing (QFT-like)
omega = np.exp(2j * np.pi / 4)
U_qft = np.array([
    [1, 1, 1, 1],
    [1, omega, omega**2, omega**3],
    [1, omega**2, 1, omega**2],
    [1, omega**3, omega**2, omega]
], dtype=complex) / 2

kraus_dep_mixed = mix_kraus_operators(kraus_dep, U_qft)

print("\nAfter QFT-like mixing:")
for i, L in enumerate(kraus_dep_mixed):
    norm = np.linalg.norm(L, 'fro')
    print(f"  ||L{i}|| = {norm:.4f}")

print(f"\nSame channel: {channels_equal(kraus_dep, kraus_dep_mixed)}")
print(f"Completeness preserved: {verify_completeness(kraus_dep_mixed)}")


print("\n" + "=" * 70)
print("PART 6: Finding the Unitary Between Two Representations")
print("=" * 70)

def find_mixing_unitary(kraus1: List[np.ndarray],
                        kraus2: List[np.ndarray]) -> np.ndarray:
    """
    Find the unitary U such that L_j = Σ_k U_{jk} K_k.

    Uses least squares on vectorized operators.
    """
    n1, n2 = len(kraus1), len(kraus2)
    n = max(n1, n2)

    # Pad with zeros
    d = kraus1[0].shape[0]
    K1_padded = kraus1 + [np.zeros((d, d), dtype=complex)] * (n - n1)
    K2_padded = kraus2 + [np.zeros((d, d), dtype=complex)] * (n - n2)

    # Stack vectorized operators
    K_matrix = np.column_stack([K.flatten() for K in K1_padded])
    L_matrix = np.column_stack([L.flatten() for L in K2_padded])

    # Solve L = K @ U^T  =>  U^T = K^+ @ L
    U_T, residuals, rank, s = np.linalg.lstsq(K_matrix, L_matrix, rcond=None)
    U = U_T.T

    return U

# Test: find unitary between original and rotated bit-flip representations
U_found = find_mixing_unitary(kraus_bf, kraus_bf_rotated)

print("Finding unitary between bit-flip representations:")
print(f"Applied rotation angle: {np.degrees(theta):.0f}°")
print(f"\nRecovered unitary U:")
print(np.array2string(U_found, precision=4))
print(f"\nOriginal unitary:")
print(np.array2string(U_rot, precision=4))
print(f"\nMatch: {np.allclose(U_found, U_rot)}")


print("\n" + "=" * 70)
print("PART 7: Measurement Interpretation Visualization")
print("=" * 70)

def visualize_measurement_bases(kraus_ops: List[np.ndarray],
                                 n_angles: int = 8):
    """
    Show how different environment measurement bases lead to different
    Kraus operators (but same channel).
    """
    print("\nEnvironment measurement bases and resulting Kraus operators:")
    print("-" * 60)

    for i, theta in enumerate(np.linspace(0, np.pi, n_angles)):
        U = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ], dtype=complex)

        mixed = mix_kraus_operators(kraus_ops, U)

        # Compute probabilities for |0⟩ input
        rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        p0 = np.real(np.trace(mixed[0] @ rho_0 @ mixed[0].conj().T))
        p1 = np.real(np.trace(mixed[1] @ rho_0 @ mixed[1].conj().T))

        print(f"θ = {np.degrees(theta):5.1f}°: P(outcome 0|ρ=|0⟩⟨0|) = {p0:.4f}, "
              f"P(outcome 1) = {p1:.4f}")

print("\nAmplitude damping (γ=0.3) with different env measurement bases:")
gamma = 0.3
K0_ad = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
K1_ad = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
visualize_measurement_bases([K0_ad, K1_ad])


print("\n" + "=" * 70)
print("PART 8: Canonical Orthogonal Representation")
print("=" * 70)

def orthogonalize_kraus(kraus_ops: List[np.ndarray]) -> List[np.ndarray]:
    """
    Find orthogonal Kraus representation via Choi eigendecomposition.
    """
    choi = kraus_to_choi(kraus_ops)
    d = kraus_ops[0].shape[0]

    # Eigendecompose Choi matrix
    eigenvalues, eigenvectors = np.linalg.eigh(choi)

    # Convert to Kraus operators
    ortho_kraus = []
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if lam > 1e-10:
            K = np.sqrt(lam) * vec.reshape((d, d), order='F')
            ortho_kraus.append(K)

    return ortho_kraus

# Get orthogonal representation for depolarizing channel
kraus_dep_ortho = orthogonalize_kraus(kraus_dep)

print(f"Depolarizing channel orthogonal representation:")
print(f"Number of operators: {len(kraus_dep_ortho)}")

# Verify orthogonality
print("\nOrthogonality check Tr(Ki† Kj):")
for i in range(len(kraus_dep_ortho)):
    for j in range(len(kraus_dep_ortho)):
        inner = np.trace(kraus_dep_ortho[i].conj().T @ kraus_dep_ortho[j])
        if abs(inner) > 1e-10 or i == j:
            print(f"  Tr(K{i}† K{j}) = {inner:.4f}")


print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Unitary freedom | $L_j = \sum_k U_{jk} K_k$ where $U^\dagger U = I$ |
| Equivalence test | $J_{\{K\}} = J_{\{L\}}$ (same Choi matrix) |
| Kraus from env measurement | $K_k = \langle k\|_E U \|0\rangle_E$ |
| Rotated env basis | $L_j = \sum_k \langle j'\|k\rangle K_k$ |

### Main Takeaways

1. **Kraus representations are not unique** - many sets describe the same channel
2. **Equivalent sets are related by unitary** (or isometry) transformations
3. **Physical interpretation**: different environment measurements give different Kraus sets
4. The **Kraus rank is invariant** under unitary mixing
5. **Choi matrix is the invariant** - same for all equivalent representations
6. Unitary freedom has implications for **error correction** and **noise modeling**

---

## Daily Checklist

- [ ] I can prove that Kraus representations are not unique
- [ ] I understand and can apply the unitary freedom theorem
- [ ] I can construct equivalent Kraus representations
- [ ] I understand the measurement interpretation of unitary freedom
- [ ] I can verify channel equality using Choi matrices
- [ ] I completed the computational lab exercises
- [ ] I attempted at least 3 practice problems

---

## Preview: Day 649

Tomorrow we study **channel composition**:
- Sequential application of channels
- Parallel (tensor product) channels
- How errors accumulate
- Concatenated operations

---

*"The unitary freedom in Kraus representations reflects the fundamental indistinguishability of different environmental measurement schemes—quantum mechanics doesn't tell us 'which error occurred,' only the overall effect."* — Daniel Gottesman
