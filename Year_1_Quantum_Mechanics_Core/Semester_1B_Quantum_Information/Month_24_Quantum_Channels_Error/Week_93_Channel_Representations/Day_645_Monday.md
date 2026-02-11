# Day 645: Kraus Representation Deep Dive

## Week 93: Channel Representations | Month 24: Quantum Channels & Error Introduction

---

## Schedule Overview

| Session | Time | Topic |
|---------|------|-------|
| **Morning** | 3 hours | Operator-sum representation, mathematical foundations |
| **Afternoon** | 2.5 hours | Problem solving with Kraus operators |
| **Evening** | 1.5 hours | Computational lab: implementing quantum channels |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Define** the Kraus operator-sum representation of a quantum channel
2. **Verify** trace preservation and complete positivity conditions
3. **Derive** Kraus operators for fundamental quantum channels
4. **Calculate** the output of a channel given input state and Kraus operators
5. **Understand** the physical interpretation of Kraus operators as measurement outcomes
6. **Connect** the Kraus representation to open quantum systems

---

## Core Content

### 1. From Unitary Evolution to General Quantum Operations

In isolated quantum systems, evolution is unitary:
$$\rho \mapsto U\rho U^\dagger$$

This preserves purity: pure states remain pure. But real quantum systems interact with their environment, leading to **decoherence** and **noise**. We need a more general framework.

**Key Question:** What is the most general physically allowed transformation of a density matrix?

**Answer:** Completely Positive Trace-Preserving (CPTP) maps, also called **quantum channels**.

### 2. The Kraus Representation Theorem

**Theorem (Kraus, 1983):** Every quantum channel $\mathcal{E}: \mathcal{B}(\mathcal{H}) \to \mathcal{B}(\mathcal{H})$ can be written as:

$$\boxed{\mathcal{E}(\rho) = \sum_{k=1}^{r} K_k \rho K_k^\dagger}$$

where the **Kraus operators** $\{K_k\}$ satisfy the **completeness relation**:

$$\boxed{\sum_{k=1}^{r} K_k^\dagger K_k = I}$$

**Terminology:**
- This is also called the **operator-sum representation**
- The Kraus operators are sometimes called **operation elements**
- The number $r$ of Kraus operators is called the **Kraus rank**

### 3. Why This Form?

The Kraus representation naturally arises from considering system-environment interactions:

**Physical Picture:**
1. System $S$ starts in state $\rho_S$
2. Environment $E$ starts in pure state $|0\rangle_E$
3. Joint system undergoes unitary evolution $U_{SE}$
4. We trace out the environment

$$\mathcal{E}(\rho_S) = \text{Tr}_E[U_{SE}(\rho_S \otimes |0\rangle\langle 0|_E)U_{SE}^\dagger]$$

Expanding the partial trace:
$$\mathcal{E}(\rho_S) = \sum_k \langle k|_E U_{SE} |0\rangle_E \cdot \rho_S \cdot \langle 0|_E U_{SE}^\dagger |k\rangle_E$$

Defining $K_k = \langle k|_E U_{SE} |0\rangle_E$, we get:
$$\mathcal{E}(\rho_S) = \sum_k K_k \rho_S K_k^\dagger$$

### 4. Properties of Quantum Channels

**Trace Preservation (TP):**
$$\text{Tr}[\mathcal{E}(\rho)] = \text{Tr}(\rho) = 1$$

This requires $\sum_k K_k^\dagger K_k = I$.

**Proof:**
$$\text{Tr}[\mathcal{E}(\rho)] = \text{Tr}\left[\sum_k K_k \rho K_k^\dagger\right] = \sum_k \text{Tr}[K_k^\dagger K_k \rho] = \text{Tr}\left[\left(\sum_k K_k^\dagger K_k\right) \rho\right]$$

For this to equal $\text{Tr}(\rho)$ for all $\rho$, we need $\sum_k K_k^\dagger K_k = I$.

**Complete Positivity (CP):**

A map $\mathcal{E}$ is **positive** if $\rho \geq 0 \Rightarrow \mathcal{E}(\rho) \geq 0$.

A map is **completely positive** if $(\mathcal{I}_n \otimes \mathcal{E})$ is positive for all $n$, where $\mathcal{I}_n$ is the identity on an $n$-dimensional ancilla.

**Why Complete Positivity?**
Positivity alone is not enough! The transpose map $T(\rho) = \rho^T$ is positive but not completely positive. If we applied transpose to one half of an entangled state, we could get negative eigenvalues (non-physical).

**Key Theorem:** The Kraus form automatically guarantees complete positivity.

### 5. Fundamental Examples

#### Example 1: Identity Channel
$$\mathcal{I}(\rho) = \rho$$

Kraus representation: Single operator $K_0 = I$.

#### Example 2: Unitary Channel
$$\mathcal{U}(\rho) = U\rho U^\dagger$$

Kraus representation: Single operator $K_0 = U$.

Check: $K_0^\dagger K_0 = U^\dagger U = I$ ✓

#### Example 3: Bit-Flip Channel

With probability $p$, apply $X$ (bit flip):
$$\mathcal{E}_X(\rho) = (1-p)\rho + p X\rho X$$

Kraus operators:
$$K_0 = \sqrt{1-p} \cdot I, \quad K_1 = \sqrt{p} \cdot X$$

**Verification:**
$$K_0^\dagger K_0 + K_1^\dagger K_1 = (1-p)I + p X^\dagger X = (1-p)I + pI = I \checkmark$$

#### Example 4: Phase-Flip Channel

With probability $p$, apply $Z$ (phase flip):
$$\mathcal{E}_Z(\rho) = (1-p)\rho + p Z\rho Z$$

Kraus operators:
$$K_0 = \sqrt{1-p} \cdot I, \quad K_1 = \sqrt{p} \cdot Z$$

#### Example 5: Depolarizing Channel

With probability $p$, completely randomize the state:
$$\mathcal{E}_{\text{dep}}(\rho) = (1-p)\rho + p\frac{I}{2}$$

This can be rewritten using Pauli operators:
$$\mathcal{E}_{\text{dep}}(\rho) = \left(1-\frac{3p}{4}\right)\rho + \frac{p}{4}(X\rho X + Y\rho Y + Z\rho Z)$$

Kraus operators:
$$K_0 = \sqrt{1-\frac{3p}{4}} \cdot I, \quad K_1 = \frac{\sqrt{p}}{2} X, \quad K_2 = \frac{\sqrt{p}}{2} Y, \quad K_3 = \frac{\sqrt{p}}{2} Z$$

#### Example 6: Amplitude Damping Channel

Models spontaneous emission (decay from $|1\rangle$ to $|0\rangle$):

Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

where $\gamma \in [0,1]$ is the decay probability.

**Verification:**
$$K_0^\dagger K_0 + K_1^\dagger K_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1-\gamma \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & \gamma \end{pmatrix} = I \checkmark$$

### 6. Interpretation: Kraus Operators as Generalized Measurements

The Kraus representation has a beautiful measurement interpretation:

**Measurement Interpretation:**
- Each $K_k$ corresponds to a possible measurement outcome
- Probability of outcome $k$: $p_k = \text{Tr}(K_k \rho K_k^\dagger) = \text{Tr}(K_k^\dagger K_k \rho)$
- Post-measurement state (if outcome $k$ observed): $\rho_k = \frac{K_k \rho K_k^\dagger}{p_k}$

If we don't record which outcome occurred:
$$\rho_{\text{final}} = \sum_k p_k \rho_k = \sum_k K_k \rho K_k^\dagger = \mathcal{E}(\rho)$$

This connects to POVMs from Month 19!

### 7. Kraus Rank and Minimal Representations

**Definition:** The **Kraus rank** of a channel is the minimum number of Kraus operators needed.

**Bounds:**
- For a $d$-dimensional system: Kraus rank $\leq d^2$
- Unitary channels have Kraus rank 1
- Most noisy channels have Kraus rank > 1

**Finding Minimal Kraus Representation:**
The Kraus rank equals the rank of the Choi matrix (we'll see this tomorrow).

---

## Quantum Computing Connection

### NISQ Device Noise

In near-term quantum computers, every gate is a noisy channel:

**Ideal gate:** $U_{\text{ideal}}$

**Actual gate:** $\mathcal{E}_{\text{actual}}(\rho) = (1-\epsilon)U\rho U^\dagger + \epsilon \cdot \text{noise}(\rho)$

Understanding Kraus representations helps us:
- Model gate errors precisely
- Design error mitigation strategies
- Benchmark quantum hardware

### Gate Fidelity

The **average gate fidelity** measures how close a noisy channel is to the ideal:
$$F_{\text{avg}}(\mathcal{E}, U) = \int d\psi \langle\psi|U^\dagger \mathcal{E}(|\psi\rangle\langle\psi|)U|\psi\rangle$$

This can be computed from the Kraus operators!

---

## Worked Examples

### Example 1: Computing Channel Output

**Problem:** Apply the bit-flip channel with $p = 0.1$ to the state $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$.

**Solution:**

Step 1: Write the initial density matrix
$$\rho = |+\rangle\langle+| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

Step 2: Identify Kraus operators
$$K_0 = \sqrt{0.9} \cdot I = \sqrt{0.9}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$
$$K_1 = \sqrt{0.1} \cdot X = \sqrt{0.1}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

Step 3: Compute each term
$$K_0 \rho K_0^\dagger = 0.9 \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 0.9 & 0.9 \\ 0.9 & 0.9 \end{pmatrix}$$

$$K_1 \rho K_1^\dagger = 0.1 \cdot X \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \cdot X = 0.1 \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 0.1 & 0.1 \\ 0.1 & 0.1 \end{pmatrix}$$

Note: $X|+\rangle\langle+|X = |+\rangle\langle+|$ since $|+\rangle$ is an eigenstate of $X$.

Step 4: Sum to get output
$$\mathcal{E}(\rho) = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = |+\rangle\langle+|$$

**Result:** $|+\rangle$ is unchanged by the bit-flip channel! This is because $|+\rangle$ is an eigenstate of $X$.

---

### Example 2: Amplitude Damping Effect

**Problem:** Apply amplitude damping with $\gamma = 0.5$ to the excited state $|1\rangle$.

**Solution:**

Step 1: Initial state
$$\rho = |1\rangle\langle 1| = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$

Step 2: Kraus operators
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{0.5} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{0.5} \\ 0 & 0 \end{pmatrix}$$

Step 3: Compute terms
$$K_0 \rho K_0^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{0.5} \end{pmatrix}\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{0.5} \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0.5 \end{pmatrix}$$

$$K_1 \rho K_1^\dagger = \begin{pmatrix} 0 & \sqrt{0.5} \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 0 & 0 \\ \sqrt{0.5} & 0 \end{pmatrix} = \begin{pmatrix} 0.5 & 0 \\ 0 & 0 \end{pmatrix}$$

Step 4: Output state
$$\boxed{\mathcal{E}(\rho) = \begin{pmatrix} 0.5 & 0 \\ 0 & 0.5 \end{pmatrix} = \frac{I}{2}}$$

The excited state has decayed to a 50-50 mixture of ground and excited states!

---

### Example 3: Verifying Completeness

**Problem:** Show that the following operators form a valid Kraus representation:
$$K_0 = \frac{1}{\sqrt{2}}|0\rangle\langle 0|, \quad K_1 = \frac{1}{\sqrt{2}}|0\rangle\langle 1|, \quad K_2 = \frac{1}{\sqrt{2}}|1\rangle\langle 0|, \quad K_3 = \frac{1}{\sqrt{2}}|1\rangle\langle 1|$$

**Solution:**

Compute $\sum_k K_k^\dagger K_k$:

$$K_0^\dagger K_0 = \frac{1}{2}|0\rangle\langle 0|0\rangle\langle 0| = \frac{1}{2}|0\rangle\langle 0|$$
$$K_1^\dagger K_1 = \frac{1}{2}|1\rangle\langle 0|0\rangle\langle 1| = \frac{1}{2}|1\rangle\langle 1|$$
$$K_2^\dagger K_2 = \frac{1}{2}|0\rangle\langle 1|1\rangle\langle 0| = \frac{1}{2}|0\rangle\langle 0|$$
$$K_3^\dagger K_3 = \frac{1}{2}|1\rangle\langle 1|1\rangle\langle 1| = \frac{1}{2}|1\rangle\langle 1|$$

Sum:
$$\sum_k K_k^\dagger K_k = |0\rangle\langle 0| + |1\rangle\langle 1| = I \checkmark$$

This is a valid CPTP map (it's actually the completely depolarizing channel that maps everything to $I/2$).

---

## Practice Problems

### Direct Application

1. **Problem 1:** Write the Kraus operators for the phase-flip channel $\mathcal{E}_Z(\rho) = (1-p)\rho + pZ\rho Z$ and verify the completeness relation.

2. **Problem 2:** Apply the depolarizing channel with $p = 0.2$ to the state $|0\rangle$. What is the purity of the output state?

3. **Problem 3:** Show that any unitary channel $\mathcal{U}(\rho) = U\rho U^\dagger$ has Kraus rank 1.

### Intermediate

4. **Problem 4:** Consider the "partial measurement" channel that measures in the computational basis but doesn't record the outcome:
   $$\mathcal{E}(\rho) = |0\rangle\langle 0|\rho|0\rangle\langle 0| + |1\rangle\langle 1|\rho|1\rangle\langle 1|$$
   Find the Kraus operators and verify the completeness relation.

5. **Problem 5:** The generalized amplitude damping channel models decay in the presence of thermal noise. Its Kraus operators are:
   $$K_0 = \sqrt{p}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \sqrt{p}\begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
   $$K_2 = \sqrt{1-p}\begin{pmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{pmatrix}, \quad K_3 = \sqrt{1-p}\begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}$$
   Verify this satisfies the completeness relation.

6. **Problem 6:** Find the fixed points of the amplitude damping channel—states $\rho$ such that $\mathcal{E}(\rho) = \rho$.

### Challenging

7. **Problem 7:** Prove that if $\mathcal{E}$ is a CPTP map, then $\text{Tr}(\mathcal{E}(\rho)^2) \leq \text{Tr}(\rho^2)$ for all $\rho$. (Hint: Use convexity.)

8. **Problem 8:** Show that the composition of two CPTP maps is CPTP. If $\mathcal{E}_1$ has Kraus operators $\{K_k\}$ and $\mathcal{E}_2$ has Kraus operators $\{L_j\}$, find the Kraus operators for $\mathcal{E}_2 \circ \mathcal{E}_1$.

9. **Problem 9:** A qubit undergoes amplitude damping with parameter $\gamma$ followed by phase damping with parameter $\lambda$. Find the combined Kraus operators and determine the effect on an arbitrary initial state $\rho = \begin{pmatrix} \rho_{00} & \rho_{01} \\ \rho_{10} & \rho_{11} \end{pmatrix}$.

---

## Computational Lab

```python
"""
Day 645 Computational Lab: Kraus Representation of Quantum Channels
===================================================================
Topics: Implementing channels, verifying CPTP, visualizing effects
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D

# Define standard states and operators
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def apply_channel(rho: np.ndarray, kraus_ops: List[np.ndarray]) -> np.ndarray:
    """
    Apply a quantum channel defined by Kraus operators to a density matrix.

    E(rho) = sum_k K_k @ rho @ K_k^dag

    Parameters:
        rho: Input density matrix
        kraus_ops: List of Kraus operators

    Returns:
        Output density matrix
    """
    output = np.zeros_like(rho)
    for K in kraus_ops:
        output += K @ rho @ K.conj().T
    return output


def verify_cptp(kraus_ops: List[np.ndarray], tol: float = 1e-10) -> dict:
    """
    Verify that Kraus operators define a valid CPTP map.

    Checks:
    1. Trace preservation: sum_k K_k^dag K_k = I
    2. (CP is automatic from Kraus form)
    """
    d = kraus_ops[0].shape[0]

    # Check trace preservation
    sum_kdk = np.zeros((d, d), dtype=complex)
    for K in kraus_ops:
        sum_kdk += K.conj().T @ K

    tp_error = np.max(np.abs(sum_kdk - np.eye(d)))
    is_tp = tp_error < tol

    return {
        'is_trace_preserving': is_tp,
        'tp_error': tp_error,
        'sum_K_dag_K': sum_kdk,
        'num_kraus_ops': len(kraus_ops)
    }


def bit_flip_kraus(p: float) -> List[np.ndarray]:
    """Kraus operators for bit-flip channel with probability p."""
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p) * X
    return [K0, K1]


def phase_flip_kraus(p: float) -> List[np.ndarray]:
    """Kraus operators for phase-flip channel with probability p."""
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p) * Z
    return [K0, K1]


def depolarizing_kraus(p: float) -> List[np.ndarray]:
    """Kraus operators for depolarizing channel with probability p."""
    K0 = np.sqrt(1 - 3*p/4) * I
    K1 = np.sqrt(p/4) * X
    K2 = np.sqrt(p/4) * Y
    K3 = np.sqrt(p/4) * Z
    return [K0, K1, K2, K3]


def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    """Kraus operators for amplitude damping with decay probability gamma."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]


def phase_damping_kraus(lam: float) -> List[np.ndarray]:
    """Kraus operators for phase damping (dephasing) with parameter lambda."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    return [K0, K1]


def density_to_bloch(rho: np.ndarray) -> Tuple[float, float, float]:
    """Convert density matrix to Bloch vector (r_x, r_y, r_z)."""
    r_x = 2 * np.real(rho[0, 1])
    r_y = 2 * np.imag(rho[1, 0])
    r_z = np.real(rho[0, 0] - rho[1, 1])
    return r_x, r_y, r_z


def bloch_to_density(r_x: float, r_y: float, r_z: float) -> np.ndarray:
    """Convert Bloch vector to density matrix."""
    return 0.5 * (I + r_x * X + r_y * Y + r_z * Z)


# ===== DEMONSTRATIONS =====

print("=" * 70)
print("PART 1: Verifying CPTP for Common Channels")
print("=" * 70)

channels = {
    'Bit-flip (p=0.1)': bit_flip_kraus(0.1),
    'Phase-flip (p=0.2)': phase_flip_kraus(0.2),
    'Depolarizing (p=0.3)': depolarizing_kraus(0.3),
    'Amplitude damping (γ=0.5)': amplitude_damping_kraus(0.5),
    'Phase damping (λ=0.4)': phase_damping_kraus(0.4)
}

for name, kraus_ops in channels.items():
    result = verify_cptp(kraus_ops)
    print(f"\n{name}:")
    print(f"  Number of Kraus operators: {result['num_kraus_ops']}")
    print(f"  Trace preserving: {result['is_trace_preserving']}")
    print(f"  TP error: {result['tp_error']:.2e}")


print("\n" + "=" * 70)
print("PART 2: Channel Effects on Specific States")
print("=" * 70)

# Test states
rho_0 = ket_0 @ ket_0.conj().T  # |0⟩
rho_1 = ket_1 @ ket_1.conj().T  # |1⟩
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
rho_plus = ket_plus @ ket_plus.conj().T  # |+⟩

states = {'|0⟩': rho_0, '|1⟩': rho_1, '|+⟩': rho_plus}

print("\nAmplitude Damping (γ=0.5) Effects:")
print("-" * 40)
ad_kraus = amplitude_damping_kraus(0.5)

for name, rho in states.items():
    rho_out = apply_channel(rho, ad_kraus)
    purity_in = np.real(np.trace(rho @ rho))
    purity_out = np.real(np.trace(rho_out @ rho_out))
    print(f"\n{name}:")
    print(f"  Input purity:  {purity_in:.4f}")
    print(f"  Output purity: {purity_out:.4f}")
    print(f"  Output state:\n{np.array2string(rho_out, precision=4)}")


print("\n" + "=" * 70)
print("PART 3: Repeated Application of Channels")
print("=" * 70)

def apply_n_times(rho: np.ndarray, kraus_ops: List[np.ndarray], n: int) -> np.ndarray:
    """Apply channel n times."""
    for _ in range(n):
        rho = apply_channel(rho, kraus_ops)
    return rho

# Track purity under repeated amplitude damping
gamma = 0.1
ad_kraus = amplitude_damping_kraus(gamma)
n_steps = 50

purities = []
excited_probs = []
rho = rho_1.copy()  # Start in |1⟩

for n in range(n_steps + 1):
    if n > 0:
        rho = apply_channel(rho, ad_kraus)
    purities.append(np.real(np.trace(rho @ rho)))
    excited_probs.append(np.real(rho[1, 1]))

# Plot decay
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(n_steps + 1), excited_probs, 'b-', linewidth=2)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax1.set_xlabel('Number of channel applications')
ax1.set_ylabel('P(|1⟩)')
ax1.set_title(f'Amplitude Damping: Excited State Population (γ={gamma})')
ax1.grid(True, alpha=0.3)

ax2.plot(range(n_steps + 1), purities, 'g-', linewidth=2)
ax2.set_xlabel('Number of channel applications')
ax2.set_ylabel('Purity Tr(ρ²)')
ax2.set_title('State Purity Evolution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('amplitude_damping_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: amplitude_damping_evolution.png")


print("\n" + "=" * 70)
print("PART 4: Bloch Sphere Visualization of Channel Effects")
print("=" * 70)

def visualize_channel_effect(kraus_ops: List[np.ndarray], channel_name: str,
                              n_points: int = 100):
    """
    Visualize how a channel transforms the Bloch sphere.
    Sample points on Bloch sphere, apply channel, plot transformation.
    """
    fig = plt.figure(figsize=(14, 6))

    # Generate points on Bloch sphere surface
    theta = np.linspace(0, np.pi, n_points)
    phi = np.linspace(0, 2*np.pi, n_points)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Input points (on sphere surface)
    x_in = np.sin(theta_grid) * np.cos(phi_grid)
    y_in = np.sin(theta_grid) * np.sin(phi_grid)
    z_in = np.cos(theta_grid)

    # Apply channel to each point
    x_out = np.zeros_like(x_in)
    y_out = np.zeros_like(y_in)
    z_out = np.zeros_like(z_in)

    for i in range(n_points):
        for j in range(n_points):
            rho = bloch_to_density(x_in[i,j], y_in[i,j], z_in[i,j])
            rho_out = apply_channel(rho, kraus_ops)
            x_out[i,j], y_out[i,j], z_out[i,j] = density_to_bloch(rho_out)

    # Plot input (Bloch sphere)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x_in, y_in, z_in, alpha=0.3, color='blue')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Input: Bloch Sphere')
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([-1.1, 1.1])

    # Plot output
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x_out, y_out, z_out, alpha=0.3, color='red')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Output: After {channel_name}')
    ax2.set_xlim([-1.1, 1.1])
    ax2.set_ylim([-1.1, 1.1])
    ax2.set_zlim([-1.1, 1.1])

    plt.tight_layout()
    filename = f'bloch_{channel_name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# Visualize depolarizing channel
print("\nVisualizing Depolarizing Channel (p=0.5):")
visualize_channel_effect(depolarizing_kraus(0.5), "Depolarizing p=0.5", n_points=30)

print("\nVisualizing Amplitude Damping (γ=0.5):")
visualize_channel_effect(amplitude_damping_kraus(0.5), "Amplitude Damping γ=0.5", n_points=30)


print("\n" + "=" * 70)
print("PART 5: Composition of Channels")
print("=" * 70)

def compose_kraus(kraus1: List[np.ndarray], kraus2: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compose two channels: E2 ∘ E1 (E1 applied first, then E2).
    Kraus operators: {L_j K_k} for all j, k.
    """
    composed = []
    for K in kraus1:
        for L in kraus2:
            composed.append(L @ K)
    return composed

# Compose bit-flip and phase-flip
bf_kraus = bit_flip_kraus(0.1)
pf_kraus = phase_flip_kraus(0.1)
composed = compose_kraus(bf_kraus, pf_kraus)

print(f"\nBit-flip (p=0.1): {len(bf_kraus)} Kraus operators")
print(f"Phase-flip (p=0.1): {len(pf_kraus)} Kraus operators")
print(f"Composed channel: {len(composed)} Kraus operators")

# Verify the composition is still CPTP
result = verify_cptp(composed)
print(f"Composed channel is CPTP: {result['is_trace_preserving']}")

# Compare direct vs composed application
rho_test = rho_plus.copy()
rho_direct = apply_channel(apply_channel(rho_test, bf_kraus), pf_kraus)
rho_composed = apply_channel(rho_test, composed)

print(f"\nDifference between direct and composed: {np.max(np.abs(rho_direct - rho_composed)):.2e}")


print("\n" + "=" * 70)
print("PART 6: Fidelity Calculation")
print("=" * 70)

def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Calculate fidelity between two density matrices.
    F(ρ, σ) = (Tr√(√ρ σ √ρ))²
    For pure state σ = |ψ⟩⟨ψ|: F = ⟨ψ|ρ|ψ⟩
    """
    sqrt_rho = np.linalg.matrix_power(rho, 1)  # For general case, use sqrtm
    # Simplified for when one state is pure
    return np.real(np.trace(rho @ sigma))

def average_gate_fidelity(kraus_ops: List[np.ndarray], U_ideal: np.ndarray,
                          n_samples: int = 1000) -> float:
    """
    Estimate average gate fidelity by Monte Carlo sampling.
    F_avg = ∫ ⟨ψ|U† E(|ψ⟩⟨ψ|) U|ψ⟩ dψ
    """
    fid_sum = 0
    for _ in range(n_samples):
        # Random pure state
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        psi = np.array([[np.cos(theta/2)], [np.exp(1j*phi)*np.sin(theta/2)]])
        rho_in = psi @ psi.conj().T

        # Apply channel and ideal gate
        rho_out = apply_channel(rho_in, kraus_ops)
        ideal_out = U_ideal @ rho_in @ U_ideal.conj().T

        fid_sum += fidelity(rho_out, ideal_out)

    return fid_sum / n_samples

# Calculate fidelity of noisy identity vs ideal identity
print("\nAverage Gate Fidelity (noisy vs ideal identity):")
for p in [0.01, 0.05, 0.1, 0.2]:
    dep_kraus = depolarizing_kraus(p)
    fid = average_gate_fidelity(dep_kraus, I)
    print(f"  Depolarizing p={p:.2f}: F_avg = {fid:.4f}")


print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Kraus representation | $\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger$ |
| Trace preservation | $\sum_k K_k^\dagger K_k = I$ |
| Bit-flip channel | $K_0 = \sqrt{1-p}I, \; K_1 = \sqrt{p}X$ |
| Amplitude damping | $K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \; K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$ |

### Main Takeaways

1. **Kraus representation** expresses any quantum channel as $\mathcal{E}(\rho) = \sum_k K_k \rho K_k^\dagger$
2. **Completeness relation** $\sum_k K_k^\dagger K_k = I$ ensures trace preservation
3. **Complete positivity** is automatic from the Kraus form
4. Kraus operators have a **measurement interpretation**: each $K_k$ is a possible operation outcome
5. **Common channels** (bit-flip, phase-flip, depolarizing, amplitude damping) all have simple Kraus forms
6. Channels **reduce purity** in general—pure states become mixed states

---

## Daily Checklist

- [ ] I can write Kraus operators for standard quantum channels
- [ ] I understand the trace preservation condition and its physical meaning
- [ ] I can verify whether given operators form a valid Kraus representation
- [ ] I can compute the output of a channel given input state and Kraus operators
- [ ] I understand the measurement interpretation of Kraus operators
- [ ] I completed the computational lab exercises
- [ ] I attempted at least 3 practice problems

---

## Preview: Day 646

Tomorrow we explore the **Choi-Jamiolkowski isomorphism**, a powerful equivalence between:
- Quantum channels (CPTP maps)
- Positive operators on a larger Hilbert space

This duality provides:
- An elegant test for complete positivity
- A direct way to count Kraus rank
- Deep connections between channels and entanglement

---

*"The Kraus representation reveals that every quantum channel can be understood as a generalized measurement where we forget the outcome."* — Michael Nielsen
