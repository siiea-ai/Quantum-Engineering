# Day 604: QPE Circuit Design

## Overview

**Day 604** | Week 87, Day 2 | Month 22 | Quantum Algorithms I

Today we design the complete Quantum Phase Estimation (QPE) circuit. We'll understand how controlled unitary operations combined with the inverse QFT extract eigenvalue phases with high precision.

---

## Learning Objectives

1. Design the complete QPE circuit
2. Understand the role of each component
3. Trace the quantum state through the circuit
4. Analyze the controlled-$U^{2^k}$ operations
5. Connect to the QFT framework
6. Implement QPE computationally

---

## Core Content

### QPE Circuit Overview

The QPE circuit consists of three stages:
1. **Initialization**: Hadamard on ancilla register
2. **Phase Kickback**: Controlled-$U$ operations
3. **Phase Extraction**: Inverse QFT on ancilla

```
|0⟩ ─[H]──●────────────────────────────┐
          │                            │
|0⟩ ─[H]──┼───●────────────────────────┤
          │   │                        │ QFT⁻¹
|0⟩ ─[H]──┼───┼───●────────────────────┤
          │   │   │                    │
|0⟩ ─[H]──┼───┼───┼───●────────────────┘
          │   │   │   │
|ψ⟩ ─────U⁸──U⁴──U²──U¹─────────────────
```

### Stage 1: Hadamard Initialization

Apply Hadamard to all $n$ ancilla qubits:

$$|0\rangle^{\otimes n}|\psi\rangle \xrightarrow{H^{\otimes n} \otimes I} \frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1}|k\rangle|\psi\rangle$$

Each ancilla is now in $|+\rangle$.

### Stage 2: Controlled-U Operations

Apply controlled-$U^{2^j}$ with ancilla qubit $j$ as control:

**Key insight:** $U^{2^j}|\psi\rangle = (e^{2\pi i\phi})^{2^j}|\psi\rangle = e^{2\pi i \cdot 2^j\phi}|\psi\rangle$

After all controlled operations:
$$\frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1}e^{2\pi ik\phi}|k\rangle|\psi\rangle$$

Each $|k\rangle$ accumulates phase based on which controlled-$U^{2^j}$ operations fire (determined by the binary representation of $k$).

### Detailed State Evolution

Let $k = k_1 2^{n-1} + k_2 2^{n-2} + \cdots + k_n 2^0$ in binary.

When $|k\rangle = |k_1 k_2 \cdots k_n\rangle$ is in the control register:
- $CU^{2^{n-1}}$ applies phase $e^{2\pi i \cdot 2^{n-1}\phi}$ if $k_1 = 1$
- $CU^{2^{n-2}}$ applies phase $e^{2\pi i \cdot 2^{n-2}\phi}$ if $k_2 = 1$
- ...and so on

Total phase accumulated:
$$e^{2\pi i\phi(k_1 2^{n-1} + k_2 2^{n-2} + \cdots + k_n)} = e^{2\pi ik\phi}$$

### Stage 3: Inverse QFT

The state before inverse QFT:
$$\frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1}e^{2\pi ik\phi}|k\rangle|\psi\rangle$$

If $\phi = m/2^n$ for integer $m$, this is exactly $QFT|m\rangle|\psi\rangle$!

Applying $QFT^{-1}$:
$$QFT^{-1}\left[\frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1}e^{2\pi ikm/2^n}|k\rangle\right] = |m\rangle$$

**Result:** Measure $|m\rangle$ with certainty when $2^n\phi$ is an integer.

### Circuit Components in Detail

**Controlled-$U^{2^j}$ Implementation:**

If $U$ requires $g$ gates, then $U^{2^j}$ might be implemented by:
1. Repeated squaring: compute $U^2$, $U^4$, $U^8$, etc.
2. Or direct implementation if structure is known

**Gate Count:**
- Hadamards: $n$
- Controlled-U operations: $n$ (one per ancilla)
- Inverse QFT: $O(n^2)$
- Total: $O(n^2) + n \cdot (\text{cost of controlled-}U)$

### The Role of Each Ancilla

**Ancilla $j$ (counting from 0):**
- Controls $U^{2^{n-1-j}}$
- Contributes to bit $j$ of the phase estimate
- Higher-indexed ancillas resolve finer phase distinctions

### Complete QPE Algorithm

```
Algorithm QPE(U, |ψ⟩, n):
    1. Initialize: |0⟩^⊗n |ψ⟩

    2. Apply H^⊗n to ancilla register

    3. For j = 0 to n-1:
         Apply controlled-U^{2^{n-1-j}} with ancilla j as control

    4. Apply inverse QFT to ancilla register

    5. Measure ancilla register → m

    6. Return φ_estimate = m/2^n
```

---

## Worked Examples

### Example 1: 2-Ancilla QPE for Z Gate

Run QPE on $U = Z$ with eigenstate $|1\rangle$ (eigenvalue $-1 = e^{i\pi}$, $\phi = 0.5$).

**Solution:**

**Initial:** $|00\rangle|1\rangle$

**After Hadamards:**
$$\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)|1\rangle$$

**After CZ² = CZ·CZ = I (controlled by ancilla 0):**
$Z^2 = I$, so no change on the computational part. But wait, $Z^2|1\rangle = |1\rangle$, so phase is $e^{2\pi i \cdot 2 \cdot 0.5} = e^{2\pi i} = 1$.

No phase accumulated from first controlled gate!

**After CZ (controlled by ancilla 1):**
$Z|1\rangle = -|1\rangle = e^{i\pi}|1\rangle$

Phase $e^{i\pi}$ applied when ancilla 1 is $|1\rangle$:
$$\frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)|1\rangle$$

**Rearranging:**
$$= \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle - |1\rangle)|1\rangle = |+\rangle|-\rangle|1\rangle$$

**After inverse QFT (2-qubit):**
$QFT^{-1}|+\rangle|-\rangle = |01\rangle$

Wait, let me recalculate. We have:
$$\frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$$

This is $\frac{1}{2}\sum_k e^{2\pi ik \cdot 0.5}|k\rangle = \frac{1}{2}(|0\rangle - |1\rangle + |2\rangle - |3\rangle)$

After $QFT^{-1}$: This should give $|10\rangle$ (binary for 2), since $\phi = 0.5 = 2/4$.

**Measurement:** $|10\rangle$ → $m = 2$, $\phi = 2/4 = 0.5$ ✓

### Example 2: 3-Ancilla QPE for T Gate

$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$, eigenstate $|1\rangle$, $\phi = 1/8$.

**Solution:**

$2^3 \cdot \phi = 8 \cdot 1/8 = 1$ (exactly representable in 3 bits).

**Expected output:** $|001\rangle = |1\rangle$, giving $\phi = 1/8$.

**Controlled operations:**
- $CT^4 = CZ$ (phase $\pi$) on $|1\rangle$
- $CT^2 = CS$ (phase $\pi/2$) on $|1\rangle$
- $CT$ (phase $\pi/4$) on $|1\rangle$

Total phase when all controls active: $e^{i(\pi + \pi/2 + \pi/4)} = e^{i7\pi/4}$

This matches $e^{2\pi i \cdot 7 \cdot 1/8} = e^{7i\pi/4}$ for $k = 7 = 111$. ✓

### Example 3: Non-Exact Phase

$\phi = 0.3$ (not exactly representable), 3 ancillas.

**Solution:**

$2^3 \cdot 0.3 = 2.4$ (not integer).

The output is a superposition centered around $|2\rangle$ and $|3\rangle$.

Expected probabilities:
- $P(|010\rangle) \approx P(|2\rangle)$: high
- $P(|011\rangle) \approx P(|3\rangle)$: moderate
- Others: low

Measurement gives $m = 2$ or $m = 3$, estimating $\phi \approx 0.25$ or $\phi \approx 0.375$.

---

## Practice Problems

### Problem 1: Gate Count

For $n = 5$ ancilla QPE, how many controlled-$U$ operations are needed?

### Problem 2: Circuit Trace

Trace 2-ancilla QPE for $U = X$ with eigenstate $|+\rangle$ (eigenvalue $+1$).

### Problem 3: Power of U

If $U^{16}$ can be implemented more efficiently than applying $U$ sixteen times, how does this help 5-ancilla QPE?

### Problem 4: Two-Qubit U

Design QPE circuit (sketch) for a 2-qubit unitary $U$ with 3 ancillas.

---

## Computational Lab

```python
"""Day 604: QPE Circuit Design"""
import numpy as np

def qft_matrix(n):
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega**(j*k) for k in range(N)]
                     for j in range(N)]) / np.sqrt(N)

def inverse_qft_matrix(n):
    return qft_matrix(n).conj().T

def qpe_circuit(U, psi, n_ancilla, verbose=False):
    """
    Quantum Phase Estimation
    U: unitary matrix (d x d)
    psi: eigenstate of U (d-dimensional vector)
    n_ancilla: number of ancilla qubits
    Returns: measurement probability distribution
    """
    d = len(psi)
    N = 2**n_ancilla

    # Full state dimension
    full_dim = N * d

    # Initialize |0⟩^⊗n |ψ⟩
    state = np.zeros(full_dim, dtype=complex)
    for i in range(d):
        state[i] = psi[i]  # |0...0⟩|ψ⟩

    if verbose:
        print("Initial state: |0...0⟩|ψ⟩")

    # Apply H^⊗n to ancilla
    H_n = qft_matrix(n_ancilla) * np.sqrt(N)  # Unnormalized for simplicity
    # Actually use Hadamard: H^⊗n|0⟩ = (1/√N) Σ|k⟩
    # Reshape and apply
    state_reshaped = state.reshape(N, d)
    hadamard_state = np.ones(N, dtype=complex) / np.sqrt(N)
    state = np.outer(hadamard_state, psi).flatten()

    if verbose:
        print("After Hadamards: (1/√N) Σ_k |k⟩|ψ⟩")

    # Apply controlled-U^{2^j} operations
    state_reshaped = state.reshape(N, d)

    for j in range(n_ancilla):
        power = 2**(n_ancilla - 1 - j)
        U_power = np.linalg.matrix_power(U, power)

        if verbose:
            print(f"Applying controlled-U^{power} (control: ancilla {j})")

        # For each ancilla state k, if bit j is 1, apply U^{2^{n-1-j}}
        for k in range(N):
            if (k >> (n_ancilla - 1 - j)) & 1:
                state_reshaped[k] = U_power @ state_reshaped[k]

    state = state_reshaped.flatten()

    if verbose:
        print("After controlled-U operations: (1/√N) Σ_k e^{2πikφ}|k⟩|ψ⟩")

    # Apply inverse QFT to ancilla
    QFT_inv = inverse_qft_matrix(n_ancilla)

    # Apply to ancilla part only
    state_reshaped = state.reshape(N, d)
    for i in range(d):
        state_reshaped[:, i] = QFT_inv @ state_reshaped[:, i]
    state = state_reshaped.flatten()

    if verbose:
        print("After inverse QFT")

    # Measurement probabilities for ancilla
    state_reshaped = state.reshape(N, d)
    probs = np.zeros(N)
    for k in range(N):
        probs[k] = np.sum(np.abs(state_reshaped[k])**2)

    return probs

def verify_qpe(U, psi, true_phase, n_ancilla):
    """Run QPE and verify result"""
    probs = qpe_circuit(U, psi, n_ancilla)
    N = 2**n_ancilla

    # Find most likely outcome
    k_max = np.argmax(probs)
    estimated_phase = k_max / N
    error = min(abs(true_phase - estimated_phase),
                abs(true_phase - estimated_phase + 1),
                abs(true_phase - estimated_phase - 1))

    return estimated_phase, error, probs[k_max]

# Test on various gates
print("=" * 60)
print("QPE CIRCUIT TESTS")
print("=" * 60)

# Z gate, eigenstate |1⟩, phase = 0.5
print("\n--- Z gate, eigenstate |1⟩ ---")
Z = np.array([[1, 0], [0, -1]])
psi_1 = np.array([0, 1])
true_phase = 0.5

for n in [2, 3, 4]:
    est_phase, error, prob = verify_qpe(Z, psi_1, true_phase, n)
    print(f"n={n}: estimated φ = {est_phase:.4f}, error = {error:.4f}, "
          f"P(max) = {prob:.4f}")

# T gate, eigenstate |1⟩, phase = 1/8
print("\n--- T gate, eigenstate |1⟩ ---")
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
psi_1 = np.array([0, 1])
true_phase = 1/8

for n in [3, 4, 5]:
    est_phase, error, prob = verify_qpe(T, psi_1, true_phase, n)
    print(f"n={n}: estimated φ = {est_phase:.4f}, error = {error:.4f}, "
          f"P(max) = {prob:.4f}")

# Non-exact phase
print("\n--- R_z(0.3π), eigenstate |1⟩ ---")
phi_true = 0.15  # phase = 0.3π/(2π) = 0.15
Rz = np.array([[1, 0], [0, np.exp(2j * np.pi * phi_true)]])
psi_1 = np.array([0, 1])

for n in [3, 4, 5, 6]:
    est_phase, error, prob = verify_qpe(Rz, psi_1, phi_true, n)
    print(f"n={n}: estimated φ = {est_phase:.4f} (true: {phi_true:.4f}), "
          f"error = {error:.4f}, P(max) = {prob:.4f}")

# Detailed trace
print("\n" + "=" * 60)
print("DETAILED QPE TRACE")
print("=" * 60)

print("\n2-ancilla QPE on Z gate, eigenstate |1⟩:")
probs = qpe_circuit(Z, psi_1, 2, verbose=True)

print("\nMeasurement probabilities:")
for k in range(4):
    phase_est = k / 4
    print(f"  |{k:02b}⟩ → φ = {phase_est:.4f}, P = {probs[k]:.4f}")

# Visualize distribution for non-exact case
print("\n" + "=" * 60)
print("GENERATING VISUALIZATION")
print("=" * 60)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

phi_values = [0.25, 0.3, 0.123]
n_ancilla = 4
N = 2**n_ancilla

for ax, phi in zip(axes, phi_values):
    U_phi = np.array([[1, 0], [0, np.exp(2j * np.pi * phi)]])
    probs = qpe_circuit(U_phi, np.array([0, 1]), n_ancilla)

    colors = ['blue' if probs[k] == max(probs) else 'lightblue'
              for k in range(N)]
    ax.bar(range(N), probs, color=colors, edgecolor='black', alpha=0.7)
    ax.axvline(x=phi * N, color='red', linestyle='--', linewidth=2,
               label=f'True: φ={phi}')
    ax.set_xlabel('Measurement outcome m')
    ax.set_ylabel('Probability')
    ax.set_title(f'φ = {phi} ({n_ancilla} ancillas)')
    ax.legend()

plt.suptitle('QPE Measurement Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('qpe_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Distribution plots saved to 'qpe_distributions.png'")

# Circuit diagram
print("\n" + "=" * 60)
print("QPE CIRCUIT STRUCTURE")
print("=" * 60)
print("""
Standard QPE Circuit (n ancillas):

|0⟩ ──[H]──●───────────────────────────┐
           │                           │
|0⟩ ──[H]──┼────●──────────────────────┤
           │    │                      │
|0⟩ ──[H]──┼────┼────●─────────────────┤ QFT⁻¹
           │    │    │                 │
|0⟩ ──[H]──┼────┼────┼────●────────────┘
           │    │    │    │
|ψ⟩ ──────U^8──U^4──U^2──U^1───────────────

Gate count:
- Hadamard: n
- Controlled-U^{2^k}: n operations (k = 0, 1, ..., n-1)
- Inverse QFT: O(n²)
Total: O(n² + n·cost(CU))
""")
```

---

## Summary

### Key Formulas

| Stage | State |
|-------|-------|
| Initial | $\|0\rangle^{\otimes n}\|\psi\rangle$ |
| After H | $\frac{1}{\sqrt{2^n}}\sum_k \|k\rangle\|\psi\rangle$ |
| After CU's | $\frac{1}{\sqrt{2^n}}\sum_k e^{2\pi ik\phi}\|k\rangle\|\psi\rangle$ |
| After QFT⁻¹ | $\|2^n\phi\rangle\|\psi\rangle$ (if exact) |

### Key Takeaways

1. **Three stages**: Hadamard, controlled-U, inverse QFT
2. **Phase kickback** accumulates phase based on control register value
3. **Inverse QFT** converts phases to computational basis
4. **Gate count** is $O(n^2)$ plus cost of controlled-U operations
5. **Exact phases** give deterministic output

---

## Daily Checklist

- [ ] I can draw the QPE circuit
- [ ] I understand each stage's purpose
- [ ] I can trace the state evolution
- [ ] I know the gate count analysis
- [ ] I understand when output is exact vs probabilistic
- [ ] I ran the lab and verified QPE behavior

---

*Next: Day 605 - QPE Analysis and Precision*
