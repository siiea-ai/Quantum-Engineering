# Day 647: Stinespring Dilation

## Week 93: Channel Representations | Month 24: Quantum Channels & Error Introduction

---

## Schedule Overview

| Session | Time | Topic |
|---------|------|-------|
| **Morning** | 3 hours | Stinespring theorem, unitary extensions |
| **Afternoon** | 2.5 hours | Constructing dilations, physical interpretation |
| **Evening** | 1.5 hours | Computational lab: simulating open system dynamics |

---

## Learning Objectives

By the end of today, you will be able to:

1. **State** the Stinespring dilation theorem and its significance
2. **Construct** explicit unitary dilations for quantum channels
3. **Relate** Kraus operators to the Stinespring unitary
4. **Interpret** quantum channels as open system dynamics
5. **Calculate** environment states after channel application
6. **Apply** the dilation perspective to understand decoherence

---

## Core Content

### 1. The Physical Picture of Quantum Channels

Yesterday we saw that channels can be characterized by their Choi matrix. Today we develop an even more physical perspective: every quantum channel arises from **unitary evolution on a larger system**.

**The Key Insight:**
- Quantum mechanics is fundamentally unitary
- Non-unitary evolution (noise, decoherence) arises from tracing out degrees of freedom
- Every channel can be "purified" to a unitary on system + environment

### 2. The Stinespring Dilation Theorem

**Theorem (Stinespring, 1955):** For every CPTP map $\mathcal{E}: \mathcal{B}(\mathcal{H}_S) \to \mathcal{B}(\mathcal{H}_S)$, there exists:
- An environment Hilbert space $\mathcal{H}_E$
- A pure state $|0\rangle_E \in \mathcal{H}_E$
- A unitary $U: \mathcal{H}_S \otimes \mathcal{H}_E \to \mathcal{H}_S \otimes \mathcal{H}_E$

such that:

$$\boxed{\mathcal{E}(\rho_S) = \text{Tr}_E[U(\rho_S \otimes |0\rangle\langle 0|_E)U^\dagger]}$$

**Interpretation:**
1. System starts in state $\rho_S$
2. Environment starts in pure state $|0\rangle_E$
3. Joint system undergoes unitary $U$
4. We trace out (forget) the environment

### 3. Minimal Dilations

**Definition:** A dilation is **minimal** if $\dim(\mathcal{H}_E)$ equals the Kraus rank.

**Key Facts:**
- The minimum environment dimension equals the Kraus rank
- Minimal dilations are unique up to unitary equivalence on $\mathcal{H}_E$
- For a $d$-dimensional system, we need at most $\dim(\mathcal{H}_E) = d^2$

### 4. Connecting Kraus and Stinespring

Given Kraus operators $\{K_k\}_{k=1}^r$, the Stinespring unitary acts as:

$$U|{\psi}\rangle_S |0\rangle_E = \sum_{k=1}^r (K_k|\psi\rangle_S) \otimes |k\rangle_E$$

**Derivation:**
$$\text{Tr}_E[U(\rho_S \otimes |0\rangle\langle 0|_E)U^\dagger] = \sum_{k,l} \text{Tr}_E[(K_k \rho_S K_l^\dagger) \otimes |k\rangle\langle l|_E]$$
$$= \sum_k K_k \rho_S K_k^\dagger = \mathcal{E}(\rho_S)$$

**Key Relationship:**
$$K_k = \langle k|_E U |0\rangle_E$$

The Kraus operator $K_k$ is the "matrix element" of $U$ between environment states $|0\rangle$ and $|k\rangle$.

### 5. Constructing the Stinespring Unitary

**Method 1: From Kraus Operators**

Given $\{K_k\}_{k=1}^r$ with $\sum_k K_k^\dagger K_k = I$:

1. Define $V: \mathcal{H}_S \to \mathcal{H}_S \otimes \mathcal{H}_E$ by:
   $$V|\psi\rangle = \sum_k (K_k|\psi\rangle) \otimes |k\rangle$$

2. This $V$ is an **isometry**: $V^\dagger V = I_S$

3. Extend $V$ to a unitary $U$ on $\mathcal{H}_S \otimes \mathcal{H}_E$:
   $$U = V \oplus W$$
   where $W$ acts on the orthogonal complement

**Method 2: From Choi Matrix**

1. Eigendecompose $J_\mathcal{E} = \sum_k \lambda_k |\psi_k\rangle\langle\psi_k|$
2. Extract Kraus operators $K_k = \sqrt{\lambda_k} \cdot \text{reshape}(|\psi_k\rangle)$
3. Use Method 1

### 6. Example: Bit-Flip Channel

**Channel:** $\mathcal{E}_X(\rho) = (1-p)\rho + pX\rho X$

**Kraus operators:** $K_0 = \sqrt{1-p}I$, $K_1 = \sqrt{p}X$

**Stinespring dilation:**

Environment: 2-dimensional with basis $\{|0\rangle_E, |1\rangle_E\}$

The isometry $V$ acts as:
$$V|0\rangle_S = \sqrt{1-p}|0\rangle_S|0\rangle_E + \sqrt{p}|1\rangle_S|1\rangle_E$$
$$V|1\rangle_S = \sqrt{1-p}|1\rangle_S|0\rangle_E + \sqrt{p}|0\rangle_S|1\rangle_E$$

Extend to unitary $U$ on 4-dimensional space (details depend on choice of extension).

**Physical interpretation:** The environment "measures" whether a bit flip occurred!

### 7. Example: Amplitude Damping

**Channel:** Spontaneous emission with decay probability $\gamma$

**Kraus operators:**
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

**Stinespring dilation:**

The unitary $U$ on system + environment (both qubits):
$$U = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \sqrt{1-\gamma} & \sqrt{\gamma} & 0 \\ 0 & 0 & 0 & 1 \\ 0 & -\sqrt{\gamma} & \sqrt{1-\gamma} & 0 \end{pmatrix}$$

in the basis $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$ (system first, environment second).

**Physical interpretation:**
- $|0\rangle_S|0\rangle_E \to |0\rangle_S|0\rangle_E$ (ground state stable)
- $|1\rangle_S|0\rangle_E \to \sqrt{1-\gamma}|1\rangle_S|0\rangle_E + \sqrt{\gamma}|0\rangle_S|1\rangle_E$ (excited state can decay, emitting a "photon" into environment)

### 8. Environment as Quantum Memory

The Stinespring picture reveals something profound:

**Information Leakage:**
After the channel acts, information about the input state has leaked into the environment!

For amplitude damping starting from $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:

$$U(|\psi\rangle_S|0\rangle_E) = \alpha|0\rangle_S|0\rangle_E + \beta\sqrt{1-\gamma}|1\rangle_S|0\rangle_E + \beta\sqrt{\gamma}|0\rangle_S|1\rangle_E$$

The environment state is:
$$\rho_E = \text{Tr}_S[\cdot] = (|\alpha|^2 + |\beta|^2(1-\gamma))|0\rangle\langle 0|_E + |\beta|^2\gamma|1\rangle\langle 1|_E$$

The environment "knows" something about $|\beta|^2$!

### 9. Complementary Channels

**Definition:** The **complementary channel** $\mathcal{E}^c$ is obtained by tracing out the system instead of the environment:

$$\mathcal{E}^c(\rho_S) = \text{Tr}_S[U(\rho_S \otimes |0\rangle\langle 0|_E)U^\dagger]$$

**Properties:**
- $\mathcal{E}^c$ maps system states to environment states
- Information not preserved in $\mathcal{E}$ goes to $\mathcal{E}^c$
- For degradable channels: $\mathcal{E}^c = \mathcal{D} \circ \mathcal{E}$ for some $\mathcal{D}$

### 10. Uniqueness of Dilations

**Theorem:** If $U_1$ and $U_2$ are two Stinespring dilations for the same channel (with same environment dimension), then:
$$U_2 = (I_S \otimes V_E) U_1$$

for some unitary $V_E$ on the environment.

This connects to unitary freedom in Kraus representations!

---

## Quantum Computing Connection

### Decoherence as System-Environment Entanglement

Decoherence occurs when the system becomes **entangled with its environment**:

$$|\psi\rangle_S|0\rangle_E \xrightarrow{U} \sum_k (K_k|\psi\rangle_S)|k\rangle_E$$

If the environment states $|k\rangle_E$ are orthogonal, this is a perfect "measurement" by the environment, leading to complete decoherence.

### Error Correction Perspective

The Stinespring picture suggests error correction strategies:
1. **Prevent** system-environment interaction (isolation)
2. **Reverse** the unitary if we have access to the environment
3. **Encode** information in a protected subspace (error correction)

### Quantum Simulation

To simulate a noisy channel, we can:
1. Add ancilla qubits as "environment"
2. Apply the Stinespring unitary
3. Trace out (ignore) the ancilla qubits

This is how noise is simulated in quantum circuits!

---

## Worked Examples

### Example 1: Constructing Dilation for Phase Damping

**Problem:** Find the Stinespring unitary for the phase damping channel with Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$

**Solution:**

Step 1: Define the isometry $V$
$$V|\psi\rangle = K_0|\psi\rangle \otimes |0\rangle_E + K_1|\psi\rangle \otimes |1\rangle_E$$

Step 2: Compute on basis states
$$V|0\rangle_S = |0\rangle_S|0\rangle_E + 0 = |0\rangle_S|0\rangle_E$$
$$V|1\rangle_S = \sqrt{1-\lambda}|1\rangle_S|0\rangle_E + \sqrt{\lambda}|1\rangle_S|1\rangle_E$$

Step 3: Write in matrix form (system-environment basis: $|00\rangle, |01\rangle, |10\rangle, |11\rangle$)

$$V = \begin{pmatrix} 1 & 0 \\ 0 & 0 \\ 0 & \sqrt{1-\lambda} \\ 0 & \sqrt{\lambda} \end{pmatrix}$$

Step 4: Extend to unitary $U$

We need to specify $U$ on $|01\rangle_S, |11\rangle_E$. One choice:
$$\boxed{U = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \sqrt{1-\lambda} & -\sqrt{\lambda} \\ 0 & 0 & \sqrt{\lambda} & \sqrt{1-\lambda} \end{pmatrix}}$$

Verification: $U^\dagger U = I$ ✓

---

### Example 2: Environment State After Channel

**Problem:** The amplitude damping channel with $\gamma = 0.3$ acts on initial state $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$. What is the final environment state?

**Solution:**

Step 1: Write the Stinespring unitary action
$$U(|+\rangle_S|0\rangle_E) = \frac{1}{\sqrt{2}}[|0\rangle_S|0\rangle_E + \sqrt{1-\gamma}|1\rangle_S|0\rangle_E + \sqrt{\gamma}|0\rangle_S|1\rangle_E]$$

$$= \frac{1}{\sqrt{2}}[(|0\rangle + \sqrt{0.7}|1\rangle)_S|0\rangle_E + \sqrt{0.3}|0\rangle_S|1\rangle_E]$$

Step 2: Compute the joint state density matrix
$$\rho_{SE} = U(|+\rangle\langle+|_S \otimes |0\rangle\langle 0|_E)U^\dagger$$

Step 3: Trace out the system
$$\rho_E = \text{Tr}_S(\rho_{SE})$$

The joint state (unnormalized) is:
$$|\Psi\rangle_{SE} = \frac{1}{\sqrt{2}}[|0\rangle|0\rangle + \sqrt{0.7}|1\rangle|0\rangle + \sqrt{0.3}|0\rangle|1\rangle]$$

Environment reduced state:
$$\rho_E = \text{Tr}_S(|\Psi\rangle\langle\Psi|) = \frac{1}{2}[(1 + 0.7)|0\rangle\langle 0| + 0.3|1\rangle\langle 1|]$$
$$= \frac{1}{2}\begin{pmatrix} 1.7 & 0 \\ 0 & 0.3 \end{pmatrix} = \begin{pmatrix} 0.85 & 0 \\ 0 & 0.15 \end{pmatrix}$$

**Result:** The environment is in state $\boxed{\rho_E = 0.85|0\rangle\langle 0| + 0.15|1\rangle\langle 1|}$

The environment has 15% probability of being in state $|1\rangle$, indicating a photon was emitted.

---

### Example 3: Verifying Kraus from Stinespring

**Problem:** Given the Stinespring unitary
$$U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \end{pmatrix}$$
extract the Kraus operators.

**Solution:**

The Kraus operators are $K_k = \langle k|_E U |0\rangle_E$.

Environment computational basis: $|0\rangle_E, |1\rangle_E$

$K_0 = \langle 0|_E U |0\rangle_E$: Take columns 1,2 (corresponding to $|0\rangle_E$ input) and rows 1,3 (corresponding to $|0\rangle_E$ output).

Wait, let me be more careful. In the basis $|00\rangle, |01\rangle, |10\rangle, |11\rangle$:
- First index is system, second is environment
- $|0\rangle_E$ corresponds to indices 1,3 (for system 0,1)
- $|1\rangle_E$ corresponds to indices 2,4

$K_0 = \langle 0|_E U |0\rangle_E$:
$$K_0 = \frac{1}{\sqrt{2}}\begin{pmatrix} U_{11} & U_{13} \\ U_{31} & U_{33} \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}}(I + Z) \cdot \frac{1}{\sqrt{2}} = \frac{H}{\sqrt{2}} \cdot \sqrt{2} = H$$

Hmm, let me recalculate:
$$K_0 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = H$$

$K_1 = \langle 1|_E U |0\rangle_E$:
$$K_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} U_{21} & U_{23} \\ U_{41} & U_{43} \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix} = 0$$

This gives Kraus rank 1, so it's a unitary channel!

$$\boxed{K_0 = H, \quad K_1 = 0}$$

The channel is $\mathcal{E}(\rho) = H\rho H$, the Hadamard gate!

---

## Practice Problems

### Direct Application

1. **Problem 1:** Write the Stinespring isometry for the phase-flip channel $\mathcal{E}_Z(\rho) = (1-p)\rho + pZ\rho Z$.

2. **Problem 2:** For the bit-flip channel, compute the environment state after the channel acts on $|0\rangle$.

3. **Problem 3:** Verify that the identity channel has Stinespring unitary $U = I \otimes I$ with any environment initialization.

### Intermediate

4. **Problem 4:** Find the complementary channel for the amplitude damping channel.

5. **Problem 5:** Show that if a channel has Kraus rank 1, its Stinespring dilation can use a 1-dimensional environment (i.e., the channel is unitary).

6. **Problem 6:** For the depolarizing channel with $p = 0.5$, construct a Stinespring dilation using a 4-dimensional environment.

### Challenging

7. **Problem 7:** Prove that if two Stinespring dilations give the same channel, they are related by a unitary on the environment.

8. **Problem 8:** A channel is called **degradable** if its complementary channel can be obtained from it: $\mathcal{E}^c = \mathcal{D} \circ \mathcal{E}$ for some channel $\mathcal{D}$. Show that the amplitude damping channel is degradable.

9. **Problem 9:** Design a quantum circuit using the Stinespring picture to simulate the depolarizing channel with error probability $p$ on a single qubit.

---

## Computational Lab

```python
"""
Day 647 Computational Lab: Stinespring Dilation
===============================================
Topics: Constructing dilations, simulating channels, environment states
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


def kraus_to_isometry(kraus_ops: List[np.ndarray]) -> np.ndarray:
    """
    Construct the Stinespring isometry V from Kraus operators.

    V|ψ⟩ = Σₖ (Kₖ|ψ⟩) ⊗ |k⟩

    Returns V as a matrix from system to system⊗environment.
    """
    d_sys = kraus_ops[0].shape[0]
    n_kraus = len(kraus_ops)

    # V: d_sys → d_sys * n_kraus
    V = np.zeros((d_sys * n_kraus, d_sys), dtype=complex)

    for k, K in enumerate(kraus_ops):
        # Block k corresponds to environment state |k⟩
        V[k*d_sys:(k+1)*d_sys, :] = K

    return V


def isometry_to_unitary(V: np.ndarray) -> np.ndarray:
    """
    Extend an isometry V to a full unitary U.

    Uses QR decomposition on the orthogonal complement.
    """
    d_total, d_in = V.shape

    # V†V = I already (isometry)
    # We need to find W such that U = [V | W] is unitary

    # Find orthogonal complement of range(V)
    Q, R = np.linalg.qr(V, mode='complete')

    # Q is unitary, first d_in columns span range(V)
    # We need to rearrange to get V as part of U

    # Simple approach: pad V and use SVD to complete
    if d_total == d_in:
        # Already square, just use V
        return V

    # Extend V to square matrix by adding arbitrary orthonormal columns
    U = np.zeros((d_total, d_total), dtype=complex)
    U[:, :d_in] = V

    # Find null space of V† to get orthogonal complement
    # Use QR on V to get orthonormal basis for range and complement
    U[:, d_in:] = Q[:, d_in:]

    return U


def apply_channel_via_stinespring(rho: np.ndarray, U: np.ndarray,
                                   d_sys: int, d_env: int) -> np.ndarray:
    """
    Apply channel via Stinespring dilation.

    E(ρ) = Tr_E[U(ρ ⊗ |0⟩⟨0|)U†]
    """
    # Initial environment state |0⟩⟨0|
    env_state = np.zeros((d_env, d_env), dtype=complex)
    env_state[0, 0] = 1

    # Joint initial state ρ ⊗ |0⟩⟨0|
    rho_init = np.kron(rho, env_state)

    # Apply unitary
    rho_final_joint = U @ rho_init @ U.conj().T

    # Partial trace over environment
    rho_sys = partial_trace_env(rho_final_joint, d_sys, d_env)

    return rho_sys


def partial_trace_env(rho_joint: np.ndarray, d_sys: int, d_env: int) -> np.ndarray:
    """Partial trace over environment (second system)."""
    rho_sys = np.zeros((d_sys, d_sys), dtype=complex)

    for i in range(d_sys):
        for j in range(d_sys):
            for k in range(d_env):
                rho_sys[i, j] += rho_joint[i*d_env + k, j*d_env + k]

    return rho_sys


def partial_trace_sys(rho_joint: np.ndarray, d_sys: int, d_env: int) -> np.ndarray:
    """Partial trace over system (first system)."""
    rho_env = np.zeros((d_env, d_env), dtype=complex)

    for i in range(d_env):
        for j in range(d_env):
            for k in range(d_sys):
                rho_env[i, j] += rho_joint[k*d_env + i, k*d_env + j]

    return rho_env


def get_environment_state(rho_sys: np.ndarray, U: np.ndarray,
                          d_sys: int, d_env: int) -> np.ndarray:
    """
    Get the environment state after channel application.

    This is the complementary channel output.
    """
    # Initial environment state |0⟩⟨0|
    env_state = np.zeros((d_env, d_env), dtype=complex)
    env_state[0, 0] = 1

    # Joint initial state
    rho_init = np.kron(rho_sys, env_state)

    # Apply unitary
    rho_final_joint = U @ rho_init @ U.conj().T

    # Partial trace over system
    rho_env = partial_trace_sys(rho_final_joint, d_sys, d_env)

    return rho_env


# ===== DEMONSTRATIONS =====

print("=" * 70)
print("PART 1: Constructing Stinespring Dilations")
print("=" * 70)

# Amplitude damping channel
def amplitude_damping_kraus(gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]

gamma = 0.3
ad_kraus = amplitude_damping_kraus(gamma)

# Construct isometry
V_ad = kraus_to_isometry(ad_kraus)
print("\nAmplitude Damping (γ=0.3):")
print(f"Isometry V shape: {V_ad.shape}")
print("V =")
print(np.array2string(V_ad, precision=4))

# Verify V†V = I
print(f"\nV†V = I? {np.allclose(V_ad.conj().T @ V_ad, np.eye(2))}")

# Extend to unitary
U_ad = isometry_to_unitary(V_ad)
print(f"\nUnitary U shape: {U_ad.shape}")
print("U =")
print(np.array2string(U_ad, precision=4))

# Verify unitarity
print(f"U†U = I? {np.allclose(U_ad.conj().T @ U_ad, np.eye(4))}")
print(f"UU† = I? {np.allclose(U_ad @ U_ad.conj().T, np.eye(4))}")


print("\n" + "=" * 70)
print("PART 2: Verifying Stinespring Reproduces Kraus Result")
print("=" * 70)

# Test states
test_states = {
    '|0⟩': np.array([[1, 0], [0, 0]], dtype=complex),
    '|1⟩': np.array([[0, 0], [0, 1]], dtype=complex),
    '|+⟩': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
}

print("\nComparing Kraus vs Stinespring for amplitude damping:")
for name, rho in test_states.items():
    # Kraus method
    rho_kraus = sum(K @ rho @ K.conj().T for K in ad_kraus)

    # Stinespring method
    rho_stine = apply_channel_via_stinespring(rho, U_ad, d_sys=2, d_env=2)

    diff = np.max(np.abs(rho_kraus - rho_stine))
    print(f"  {name}: difference = {diff:.2e}")


print("\n" + "=" * 70)
print("PART 3: Environment States (Complementary Channel)")
print("=" * 70)

print("\nEnvironment states after amplitude damping:")
for name, rho in test_states.items():
    rho_env = get_environment_state(rho, U_ad, d_sys=2, d_env=2)
    print(f"\n{name}:")
    print(f"  Environment state:")
    print(f"    {np.array2string(rho_env, precision=4)}")
    print(f"  P(photon emitted) = {np.real(rho_env[1,1]):.4f}")


print("\n" + "=" * 70)
print("PART 4: Information Flow to Environment")
print("=" * 70)

def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ)."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros
    return -np.sum(eigenvalues * np.log2(eigenvalues))

# Track entropy flow during amplitude damping
gamma_values = np.linspace(0, 1, 21)
rho_init = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # |+⟩

S_sys = []
S_env = []
S_joint = []

for gamma in gamma_values:
    kraus = amplitude_damping_kraus(gamma)
    V = kraus_to_isometry(kraus)
    U = isometry_to_unitary(V)

    # Get final states
    rho_sys_final = apply_channel_via_stinespring(rho_init, U, 2, 2)
    rho_env_final = get_environment_state(rho_init, U, 2, 2)

    # Compute entropies
    S_sys.append(von_neumann_entropy(rho_sys_final))
    S_env.append(von_neumann_entropy(rho_env_final))

    # Joint state entropy (should be zero for pure initial state)
    env_init = np.array([[1, 0], [0, 0]], dtype=complex)
    rho_joint_init = np.kron(rho_init, env_init)
    rho_joint_final = U @ rho_joint_init @ U.conj().T
    S_joint.append(von_neumann_entropy(rho_joint_final))

# Plot entropy flow
plt.figure(figsize=(10, 6))
plt.plot(gamma_values, S_sys, 'b-', linewidth=2, label='S(system)')
plt.plot(gamma_values, S_env, 'r-', linewidth=2, label='S(environment)')
plt.plot(gamma_values, S_joint, 'g--', linewidth=2, label='S(joint)')
plt.xlabel('Damping parameter γ')
plt.ylabel('von Neumann Entropy (bits)')
plt.title('Entropy Flow in Amplitude Damping\n(Initial state: |+⟩)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('entropy_flow_amplitude_damping.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: entropy_flow_amplitude_damping.png")


print("\n" + "=" * 70)
print("PART 5: Depolarizing Channel Dilation")
print("=" * 70)

def depolarizing_kraus(p):
    """Kraus operators for depolarizing channel."""
    return [
        np.sqrt(1 - 3*p/4) * I,
        np.sqrt(p/4) * X,
        np.sqrt(p/4) * Y,
        np.sqrt(p/4) * Z
    ]

p = 0.2
dep_kraus = depolarizing_kraus(p)
V_dep = kraus_to_isometry(dep_kraus)
U_dep = isometry_to_unitary(V_dep)

print(f"\nDepolarizing channel (p={p}):")
print(f"Number of Kraus operators: {len(dep_kraus)}")
print(f"Environment dimension: {len(dep_kraus)}")
print(f"Stinespring unitary shape: {U_dep.shape}")

# Verify
rho_test = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
rho_kraus = sum(K @ rho_test @ K.conj().T for K in dep_kraus)
rho_stine = apply_channel_via_stinespring(rho_test, U_dep, d_sys=2, d_env=4)
print(f"Kraus vs Stinespring difference: {np.max(np.abs(rho_kraus - rho_stine)):.2e}")


print("\n" + "=" * 70)
print("PART 6: Extracting Kraus from Stinespring")
print("=" * 70)

def stinespring_to_kraus(U: np.ndarray, d_sys: int, d_env: int) -> List[np.ndarray]:
    """
    Extract Kraus operators from Stinespring unitary.

    Kₖ = ⟨k|_E U |0⟩_E
    """
    kraus_ops = []

    for k in range(d_env):
        K = np.zeros((d_sys, d_sys), dtype=complex)
        for i in range(d_sys):
            for j in range(d_sys):
                # Matrix element ⟨i,k|U|j,0⟩
                # In combined index: (i*d_env + k, j*d_env + 0)
                K[i, j] = U[i*d_env + k, j*d_env + 0]
        kraus_ops.append(K)

    return kraus_ops

# Extract Kraus from amplitude damping Stinespring
extracted_kraus = stinespring_to_kraus(U_ad, d_sys=2, d_env=2)

print("\nOriginal amplitude damping Kraus operators:")
for i, K in enumerate(ad_kraus):
    print(f"K{i}:\n{np.array2string(K, precision=4)}")

print("\nExtracted Kraus operators from Stinespring:")
for i, K in enumerate(extracted_kraus):
    print(f"K{i}:\n{np.array2string(K, precision=4)}")

# Verify completeness
sum_kdk = sum(K.conj().T @ K for K in extracted_kraus)
print(f"\nΣ K†K = I? {np.allclose(sum_kdk, np.eye(2))}")


print("\n" + "=" * 70)
print("PART 7: Quantum Circuit Representation")
print("=" * 70)

def visualize_stinespring_circuit(channel_name: str, d_env: int):
    """Create ASCII circuit diagram for Stinespring dilation."""
    print(f"\nStinespring circuit for {channel_name}:")
    print("=" * 50)
    print(f"System:      ─────┤     ├─────○─────")
    print(f"             ρ_in │  U  │     |  ρ_out")
    for i in range(d_env - 1):
        print(f"             ─────┤     ├─────○─────")
    print(f"Env |0⟩:     ─────┤     ├─────⊗  Tr")
    print("=" * 50)

visualize_stinespring_circuit("Amplitude Damping", 2)
visualize_stinespring_circuit("Depolarizing", 4)


print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Stinespring dilation | $\mathcal{E}(\rho) = \text{Tr}_E[U(\rho \otimes \|0\rangle\langle 0\|_E)U^\dagger]$ |
| Kraus from Stinespring | $K_k = \langle k\|_E U \|0\rangle_E$ |
| Isometry definition | $V\|\psi\rangle = \sum_k (K_k\|\psi\rangle) \otimes \|k\rangle$ |
| Complementary channel | $\mathcal{E}^c(\rho) = \text{Tr}_S[U(\rho \otimes \|0\rangle\langle 0\|)U^\dagger]$ |

### Main Takeaways

1. **Stinespring dilation** shows every channel arises from unitary evolution + partial trace
2. The **environment dimension** equals the Kraus rank
3. **Information leakage** to the environment causes decoherence
4. The **complementary channel** captures information lost to the environment
5. **Quantum simulation** of noise uses the Stinespring circuit: ancilla + unitary + trace
6. **Minimal dilations** are unique up to unitary freedom on the environment

---

## Daily Checklist

- [ ] I can state the Stinespring dilation theorem
- [ ] I can construct the Stinespring isometry from Kraus operators
- [ ] I understand the physical interpretation of environment coupling
- [ ] I can compute environment states after channel application
- [ ] I understand the connection between Kraus and Stinespring
- [ ] I completed the computational lab exercises
- [ ] I attempted at least 3 practice problems

---

## Preview: Day 648

Tomorrow we explore **unitary freedom in Kraus representations**:
- Multiple Kraus sets can describe the same channel
- These sets are related by unitary transformations
- This has implications for error correction and measurement

---

*"The Stinespring dilation theorem reveals the fundamental truth that all quantum noise arises from entanglement with an environment we cannot access."* — John Preskill
