# Day 680: Introduction to Quantum Errors

## Week 98: Quantum Errors | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Quantum Error Theory |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Problem Solving |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational Lab |

---

## Learning Objectives

By the end of Day 680, you will be able to:

1. **Explain why quantum errors differ fundamentally from classical errors**
2. **Describe the three basic Pauli errors:** X (bit-flip), Z (phase-flip), Y (combined)
3. **Understand the no-cloning theorem's implications** for error correction
4. **Analyze how continuous errors discretize** under measurement
5. **Connect classical syndrome decoding** to quantum error detection
6. **Recognize why phase errors have no classical analog**

---

## The Quantum Error Problem

### Why Quantum Error Correction is Hard

Classical error correction seems straightforward: measure the bits, compare to expected values, fix any flipped bits. But quantum mechanics introduces three fundamental obstacles:

**Obstacle 1: No-Cloning Theorem**

We cannot copy an unknown quantum state to create redundant backups:

$$\text{No unitary } U \text{ exists such that } U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle \text{ for all } |\psi\rangle$$

*Proof sketch:* If cloning worked for $|\psi\rangle$ and $|\phi\rangle$:
$$\langle\psi|\phi\rangle = \langle\psi|\phi\rangle^2$$

This only holds for $\langle\psi|\phi\rangle = 0$ or $1$—not for general states.

**Obstacle 2: Measurement Destroys Information**

Measuring a qubit in state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ collapses it:
- Outcome 0: state becomes $|0\rangle$ (lost $\alpha, \beta$)
- Outcome 1: state becomes $|1\rangle$ (lost $\alpha, \beta$)

We cannot simply "look" at the quantum state to check for errors!

**Obstacle 3: Continuous Errors**

Unlike classical bits (0 or 1), qubits live on the Bloch sphere. Errors can be continuous rotations:

$$R_x(\theta) = e^{-i\theta X/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}X$$

How do we correct an arbitrary rotation by angle $\theta$?

---

## The Pauli Error Basis

### Single-Qubit Pauli Operators

The Pauli matrices form a basis for all single-qubit operations:

$$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Key Properties:**

1. **Hermitian:** $P^\dagger = P$ for all Pauli $P$
2. **Unitary:** $P^\dagger P = I$
3. **Involutory:** $P^2 = I$
4. **Traceless:** $\text{Tr}(P) = 0$ for $P \neq I$

**Commutation Relations:**

$$[X, Y] = 2iZ, \quad [Y, Z] = 2iX, \quad [Z, X] = 2iY$$

**Anticommutation:**

$$\{X, Y\} = \{Y, Z\} = \{Z, X\} = 0$$

$$\boxed{Y = iXZ}$$

---

## The Three Fundamental Quantum Errors

### Error Type 1: Bit-Flip Error (X)

The X operator acts like a classical bit-flip:

$$X|0\rangle = |1\rangle, \quad X|1\rangle = |0\rangle$$

For a general state:
$$X(\alpha|0\rangle + \beta|1\rangle) = \alpha|1\rangle + \beta|0\rangle$$

**Connection to Classical:** This is the direct quantum analog of the classical bit-flip error we studied in Week 97!

### Error Type 2: Phase-Flip Error (Z)

The Z operator introduces a relative phase:

$$Z|0\rangle = |0\rangle, \quad Z|1\rangle = -|1\rangle$$

For a general state:
$$Z(\alpha|0\rangle + \beta|1\rangle) = \alpha|0\rangle - \beta|1\rangle$$

**No Classical Analog!** Classical bits have no concept of "phase." This is a purely quantum error.

**In the Hadamard Basis:**

Let $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ and $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$.

Then:
$$Z|+\rangle = |-\rangle, \quad Z|-\rangle = |+\rangle$$

*A phase-flip in the Z-basis is a bit-flip in the X-basis!*

### Error Type 3: Combined Error (Y)

The Y operator combines both bit and phase flips:

$$Y = iXZ$$

$$Y|0\rangle = i|1\rangle, \quad Y|1\rangle = -i|0\rangle$$

**Physical Interpretation:** The Y error can arise from a combination of X and Z errors, or from specific physical processes like certain spin-orbit couplings.

---

## Why Pauli Errors Suffice

### The Digitization of Quantum Errors

The remarkable insight enabling quantum error correction: **continuous errors discretize under syndrome measurement**.

Consider an arbitrary single-qubit error:
$$E = e_0 I + e_1 X + e_2 Y + e_3 Z$$

where $e_i \in \mathbb{C}$.

When we perform syndrome measurement (without measuring the encoded data), the error "collapses" to one of the discrete Pauli errors:

$$\boxed{\text{Any single-qubit error } \rightarrow \text{ probabilistic mixture of } I, X, Y, Z}$$

**Why This Works:**

1. Pauli matrices form a basis for $2\times 2$ matrices
2. Syndrome measurement projects onto error subspaces
3. Each error subspace corresponds to a Pauli error
4. Correction only needs to handle $I, X, Y, Z$

This is analogous to how measuring a qubit in state $\alpha|0\rangle + \beta|1\rangle$ gives discrete outcomes 0 or 1, not continuous superpositions.

---

## Error Models

### Independent Depolarizing Channel (Preview)

The most common quantum error model assigns equal probability to each Pauli error:

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

where $p$ is the error probability.

**Interpretation:**
- With probability $1-p$: no error ($I$)
- With probability $p/3$ each: X, Y, or Z error

### Bit-Flip Channel

A simpler model where only X errors occur:

$$\mathcal{E}_{bf}(\rho) = (1-p)\rho + pX\rho X$$

### Phase-Flip Channel

Only Z errors occur:

$$\mathcal{E}_{pf}(\rho) = (1-p)\rho + pZ\rho Z$$

---

## Connecting to Classical Error Correction

### The Classical-Quantum Dictionary

| Classical Concept | Quantum Analog |
|-------------------|----------------|
| Bit flip | X error (Pauli-X) |
| — (no analog) | Phase flip (Z error) |
| Bit string $\mathbf{c}$ | Quantum state $\|\psi\rangle$ |
| Redundancy (copy bits) | Entanglement (encode logically) |
| Check bits directly | Syndrome measurement (indirect) |
| Parity check matrix $H$ | Stabilizer generators |
| Syndrome $\mathbf{s} = H\mathbf{e}^T$ | Stabilizer measurement outcomes |

### Key Insight from Week 97

Recall syndrome decoding: the syndrome $\mathbf{s} = H\mathbf{e}^T$ depends only on the error pattern, not the encoded data. This principle carries over to quantum:

$$\boxed{\text{Quantum syndromes reveal error information without revealing encoded data}}$$

This is the key to overcoming the measurement problem!

---

## Multi-Qubit Errors

### The n-Qubit Pauli Group

For $n$ qubits, errors are tensor products of single-qubit Paulis:

$$\mathcal{P}_n = \{i^k P_1 \otimes P_2 \otimes \cdots \otimes P_n : P_j \in \{I, X, Y, Z\}, k \in \{0,1,2,3\}\}$$

**Example (2 qubits):**
- $X \otimes I$: bit-flip on qubit 1 only
- $I \otimes Z$: phase-flip on qubit 2 only
- $X \otimes Z$: bit-flip on qubit 1, phase-flip on qubit 2
- $Y \otimes Y$: Y error on both qubits

**Size of Pauli Group:**
$$|\mathcal{P}_n| = 4 \cdot 4^n = 4^{n+1}$$

(Factor of 4 for the phase $i^k$.)

### Error Weight

The **weight** of a Pauli error is the number of qubits on which it acts non-trivially:

$$\text{wt}(X \otimes I \otimes Z) = 2$$

A code with distance $d$ can detect all errors of weight $< d$ and correct errors of weight $\leq \lfloor(d-1)/2\rfloor$.

---

## Physical Origins of Quantum Errors

### Decoherence

Quantum systems interact with their environment, causing:

1. **T₁ Relaxation (Amplitude Damping):**
   - Excited state $|1\rangle$ decays to $|0\rangle$
   - Energy loss to environment
   - Timescale: T₁ (typically 10-100 μs for superconducting qubits)

2. **T₂ Dephasing (Phase Damping):**
   - Superposition loses coherence
   - $\alpha|0\rangle + \beta|1\rangle \rightarrow$ mixed state
   - Timescale: T₂ ≤ 2T₁

### Gate Errors

Imperfect control leads to:
- Over/under rotation
- Cross-talk between qubits
- Leakage to non-computational states

### Current Technology Error Rates

| Platform | 1-Qubit Gate Error | 2-Qubit Gate Error | T₁ | T₂ |
|----------|-------------------|-------------------|-----|-----|
| Superconducting | ~0.1% | ~0.5-1% | 50-100 μs | 50-100 μs |
| Trapped Ion | ~0.01% | ~0.1-0.5% | seconds-minutes | seconds |
| Neutral Atom | ~0.1% | ~0.5-2% | seconds | 100 ms |

---

## Worked Examples

### Example 1: Pauli Error Effects

**Problem:** Apply each Pauli error to the state $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$.

**Solution:**

Original state: $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$

**X error:**
$$X|\psi\rangle = \frac{1}{\sqrt{3}}|1\rangle + \sqrt{\frac{2}{3}}|0\rangle = \sqrt{\frac{2}{3}}|0\rangle + \frac{1}{\sqrt{3}}|1\rangle$$

**Z error:**
$$Z|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle - \sqrt{\frac{2}{3}}|1\rangle$$

**Y error:**
$$Y|\psi\rangle = iXZ|\psi\rangle = iX\left(\frac{1}{\sqrt{3}}|0\rangle - \sqrt{\frac{2}{3}}|1\rangle\right) = i\left(\frac{1}{\sqrt{3}}|1\rangle - \sqrt{\frac{2}{3}}|0\rangle\right)$$
$$= -i\sqrt{\frac{2}{3}}|0\rangle + \frac{i}{\sqrt{3}}|1\rangle$$

### Example 2: Error in Different Bases

**Problem:** A qubit is in state $|+\rangle$. What state results from each Pauli error?

**Solution:**

$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

**X error:**
$$X|+\rangle = \frac{1}{\sqrt{2}}(|1\rangle + |0\rangle) = |+\rangle$$

*The $|+\rangle$ state is an eigenstate of X!*

**Z error:**
$$Z|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = |-\rangle$$

*Z acts like a bit-flip in the Hadamard basis!*

**Y error:**
$$Y|+\rangle = iXZ|+\rangle = iX|-\rangle = i|−\rangle$$

### Example 3: Two-Qubit Error Analysis

**Problem:** Identify all weight-1 errors on a 2-qubit system.

**Solution:**

Weight-1 errors act non-trivially on exactly one qubit:

**Qubit 1 errors:**
- $X \otimes I$
- $Y \otimes I$
- $Z \otimes I$

**Qubit 2 errors:**
- $I \otimes X$
- $I \otimes Y$
- $I \otimes Z$

Total: **6 weight-1 errors** (excluding identity).

---

## Practice Problems

### Problem Set A: Direct Application

**A.1** Compute $XZX$ and express the result as a Pauli operator (possibly with a phase).

**A.2** Show that $\{X, Z\} = XZ + ZX = 0$ by explicit matrix multiplication.

**A.3** For $|\psi\rangle = \frac{1}{2}|0\rangle + \frac{\sqrt{3}}{2}|1\rangle$, compute $X|\psi\rangle$, $Z|\psi\rangle$, and verify $Y|\psi\rangle = iXZ|\psi\rangle$.

### Problem Set B: Intermediate

**B.1** Prove that the Pauli matrices (including I) form an orthonormal basis for $2\times 2$ matrices under the inner product $\langle A, B\rangle = \frac{1}{2}\text{Tr}(A^\dagger B)$.

**B.2** An arbitrary single-qubit error can be written as $E = e_0 I + e_1 X + e_2 Y + e_3 Z$. If $E$ is unitary, what constraint do the coefficients satisfy?

**B.3** How many distinct Pauli errors (up to global phase) exist for 3 qubits? How many have weight at most 1?

### Problem Set C: Challenging

**C.1** Prove that any two distinct Pauli operators either commute or anticommute. Specifically, for $P, Q \in \{I, X, Y, Z\}^{\otimes n}$, show $PQ = \pm QP$.

**C.2** The **depolarizing channel** with parameter $p$ acts as:
$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

Show this can be rewritten as:
$$\mathcal{E}(\rho) = \left(1 - \frac{4p}{3}\right)\rho + \frac{p}{3}I$$

**C.3** A qubit undergoes X error with probability $p_x$ and Z error with probability $p_z$ (independently). What is the probability of a Y error? Express the effective error channel.

---

## Computational Lab: Simulating Quantum Errors

```python
"""
Day 680 Computational Lab: Quantum Error Simulation
===================================================

Simulating Pauli errors on single qubits and analyzing their effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Part 1: Pauli Matrices
# =============================================================================

# Define Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

print("=" * 60)
print("PART 1: Pauli Matrix Properties")
print("=" * 60)

# Verify key properties
print("\n1. Verify Y = iXZ:")
print(f"iXZ = \n{1j * X @ Z}")
print(f"Y = \n{Y}")
print(f"Equal: {np.allclose(Y, 1j * X @ Z)}")

print("\n2. Verify anticommutation {X, Z} = 0:")
anticomm = X @ Z + Z @ X
print(f"XZ + ZX = \n{anticomm}")
print(f"Is zero: {np.allclose(anticomm, 0)}")

print("\n3. Verify P² = I for each Pauli:")
for name, P in paulis.items():
    is_involutory = np.allclose(P @ P, I)
    print(f"  {name}² = I: {is_involutory}")

# =============================================================================
# Part 2: Error Effects on Quantum States
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Pauli Errors on Quantum States")
print("=" * 60)

def normalize(state):
    """Normalize a quantum state vector."""
    return state / np.linalg.norm(state)

def apply_error(state, error):
    """Apply a Pauli error to a state."""
    return error @ state

def state_to_bloch(state):
    """Convert state vector to Bloch sphere coordinates."""
    # Density matrix
    rho = np.outer(state, np.conj(state))
    # Bloch coordinates
    x = np.real(np.trace(X @ rho))
    y = np.real(np.trace(Y @ rho))
    z = np.real(np.trace(Z @ rho))
    return x, y, z

# Test states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = normalize(np.array([1, 1], dtype=complex))
ket_minus = normalize(np.array([1, -1], dtype=complex))
ket_i_plus = normalize(np.array([1, 1j], dtype=complex))  # |+i⟩

test_state = normalize(np.array([1, np.sqrt(2)], dtype=complex))

print(f"\nOriginal state |ψ⟩ = (1/√3)|0⟩ + (√2/√3)|1⟩")
print(f"State vector: {test_state}")

for name, error in paulis.items():
    new_state = apply_error(test_state, error)
    print(f"\n{name}|ψ⟩ = {new_state}")

# =============================================================================
# Part 3: Bloch Sphere Visualization
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Bloch Sphere Error Visualization")
print("=" * 60)

# Create random state for visualization
theta, phi = np.pi/3, np.pi/4
initial_state = np.array([np.cos(theta/2),
                          np.exp(1j*phi)*np.sin(theta/2)], dtype=complex)

# Calculate Bloch coordinates for original and error states
states_bloch = {'Original': state_to_bloch(initial_state)}
for name, error in [('X', X), ('Y', Y), ('Z', Z)]:
    error_state = apply_error(initial_state, error)
    states_bloch[name + ' error'] = state_to_bloch(error_state)

print("\nBloch sphere coordinates:")
for name, (x, y, z) in states_bloch.items():
    print(f"  {name}: ({x:.3f}, {y:.3f}, {z:.3f})")

# Create Bloch sphere plot
fig = plt.figure(figsize=(12, 5))

# Plot 1: Bloch sphere with error states
ax1 = fig.add_subplot(121, projection='3d')

# Draw sphere wireframe
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 20)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

# Draw axes
ax1.quiver(0, 0, 0, 1.3, 0, 0, color='red', alpha=0.5, arrow_length_ratio=0.1)
ax1.quiver(0, 0, 0, 0, 1.3, 0, color='green', alpha=0.5, arrow_length_ratio=0.1)
ax1.quiver(0, 0, 0, 0, 0, 1.3, color='blue', alpha=0.5, arrow_length_ratio=0.1)
ax1.text(1.4, 0, 0, 'X', fontsize=10)
ax1.text(0, 1.4, 0, 'Y', fontsize=10)
ax1.text(0, 0, 1.4, 'Z', fontsize=10)

# Plot states
colors = {'Original': 'black', 'X error': 'red', 'Y error': 'green', 'Z error': 'blue'}
for name, (x, y, z) in states_bloch.items():
    color = colors[name]
    ax1.scatter([x], [y], [z], color=color, s=100, label=name)
    ax1.quiver(0, 0, 0, x, y, z, color=color, alpha=0.7, arrow_length_ratio=0.1)

ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.5, 1.5])
ax1.set_zlim([-1.5, 1.5])
ax1.set_title('Pauli Error Effects on Bloch Sphere')
ax1.legend(loc='upper left')

# Plot 2: Error transformations summary
ax2 = fig.add_subplot(122)
ax2.axis('off')

error_table = """
Pauli Error Transformations
══════════════════════════════

X (bit-flip):
  • |0⟩ ↔ |1⟩
  • |+⟩ → |+⟩ (eigenstate!)
  • |−⟩ → |−⟩ (eigenstate!)
  • Rotation by π around X-axis

Z (phase-flip):
  • |0⟩ → |0⟩
  • |1⟩ → −|1⟩
  • |+⟩ ↔ |−⟩ (bit-flip in X-basis!)
  • Rotation by π around Z-axis

Y (combined):
  • Y = iXZ
  • |0⟩ → i|1⟩
  • |1⟩ → −i|0⟩
  • Rotation by π around Y-axis
"""
ax2.text(0.1, 0.9, error_table, transform=ax2.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('day_680_pauli_errors.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_680_pauli_errors.png")

# =============================================================================
# Part 4: Error Probability Analysis
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Error Channel Statistics")
print("=" * 60)

def depolarizing_channel(rho, p):
    """Apply depolarizing channel with error probability p."""
    return (1 - p) * rho + (p/3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)

def bit_flip_channel(rho, p):
    """Apply bit-flip channel with error probability p."""
    return (1 - p) * rho + p * (X @ rho @ X)

def phase_flip_channel(rho, p):
    """Apply phase-flip channel with error probability p."""
    return (1 - p) * rho + p * (Z @ rho @ Z)

# Test on |+⟩ state
rho_plus = np.outer(ket_plus, np.conj(ket_plus))
print(f"\nInitial state: |+⟩")
print(f"ρ = \n{rho_plus}")

p = 0.1
print(f"\nAfter depolarizing channel (p={p}):")
rho_depol = depolarizing_channel(rho_plus, p)
print(f"ρ' = \n{np.round(rho_depol, 4)}")

print(f"\nAfter bit-flip channel (p={p}):")
rho_bf = bit_flip_channel(rho_plus, p)
print(f"ρ' = \n{np.round(rho_bf, 4)}")

print(f"\nAfter phase-flip channel (p={p}):")
rho_pf = phase_flip_channel(rho_plus, p)
print(f"ρ' = \n{np.round(rho_pf, 4)}")

# =============================================================================
# Part 5: Multi-Qubit Errors
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Multi-Qubit Pauli Errors")
print("=" * 60)

def tensor(A, B):
    """Compute tensor product of two matrices."""
    return np.kron(A, B)

# Two-qubit Paulis
print("\nTwo-qubit Pauli errors (weight 1):")
for name1, P1 in [('X', X), ('Y', Y), ('Z', Z)]:
    # Error on qubit 1
    P_1 = tensor(P1, I)
    print(f"  {name1}⊗I (error on qubit 1)")

for name2, P2 in [('X', X), ('Y', Y), ('Z', Z)]:
    # Error on qubit 2
    P_2 = tensor(I, P2)
    print(f"  I⊗{name2} (error on qubit 2)")

# Count all n-qubit Paulis
def count_paulis(n):
    """Count Pauli operators for n qubits."""
    return 4**n  # Excluding phases

def count_weight_at_most_t(n, t):
    """Count Paulis with weight at most t."""
    from math import comb
    total = 0
    for w in range(t + 1):
        total += comb(n, w) * 3**w  # Choose positions, choose X/Y/Z
    return total

print("\n\nPauli counting:")
for n in [1, 2, 3, 5, 10]:
    total = count_paulis(n)
    w1 = count_weight_at_most_t(n, 1)
    print(f"  n={n}: Total = {total:,}, Weight ≤ 1 = {w1}")

# =============================================================================
# Part 6: Summary Statistics
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Quantum Error Types")
print("=" * 60)

summary = """
┌─────────────────────────────────────────────────────────────┐
│                    Quantum Error Summary                     │
├─────────────────────────────────────────────────────────────┤
│ Error │ Matrix    │ Action        │ Classical Analog        │
├───────┼───────────┼───────────────┼─────────────────────────┤
│   X   │ σₓ        │ |0⟩↔|1⟩       │ Bit flip                │
│   Y   │ σᵧ = iXZ  │ i|0⟩↔−i|1⟩   │ Combined (no analog)    │
│   Z   │ σᵤ        │ |1⟩→−|1⟩     │ Phase flip (no analog)  │
├─────────────────────────────────────────────────────────────┤
│ Key Insight: Continuous errors discretize under syndrome    │
│ measurement → Only need to correct discrete Pauli errors!   │
└─────────────────────────────────────────────────────────────┘
"""
print(summary)

print("\n✅ Day 680 Lab Complete!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Pauli anticommutation | $\{X, Z\} = XZ + ZX = 0$ |
| Y in terms of X, Z | $Y = iXZ$ |
| Pauli involution | $P^2 = I$ for all Paulis |
| n-qubit Pauli count | $4^n$ (excluding phase) |
| Error weight | Number of non-identity tensor factors |
| Depolarizing channel | $\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$ |

### Main Takeaways

1. **Three fundamental quantum errors:** X (bit-flip), Z (phase-flip), Y (combined)
2. **Phase errors have no classical analog** — this is the key quantum complexity
3. **No-cloning and measurement collapse** prevent naive error correction
4. **Continuous errors discretize** under syndrome measurement
5. **Pauli matrices form a basis** for all single-qubit errors
6. **Multi-qubit errors** are tensor products; error weight counts affected qubits

### Classical to Quantum Correspondence

| Week 97 (Classical) | Week 98 (Quantum) |
|--------------------|-------------------|
| Bit flip error | Pauli X |
| — | Pauli Z (phase flip) |
| — | Pauli Y = iXZ |
| $[n, k, d]$ linear code | $[[n, k, d]]$ quantum code |
| Syndrome $\mathbf{s} = H\mathbf{e}^T$ | Stabilizer measurements |

---

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain why quantum error correction is harder than classical
- [ ] I understand the three obstacles: no-cloning, measurement, continuous errors
- [ ] I can describe what each Pauli error does to the computational basis
- [ ] I understand why Pauli errors suffice (discretization)
- [ ] I recognize phase errors as the uniquely quantum challenge

### Mathematical Skills
- [ ] I can compute Pauli error effects on any single-qubit state
- [ ] I can verify Pauli commutation and anticommutation relations
- [ ] I can construct multi-qubit Pauli operators using tensor products
- [ ] I understand error weight and its significance

### Computational Skills
- [ ] I can simulate Pauli errors numerically
- [ ] I can visualize error effects on the Bloch sphere
- [ ] I can implement basic error channels (depolarizing, bit-flip, phase-flip)

---

## Preview: Day 681

Tomorrow we formalize quantum error channels mathematically:
- **CPTP maps:** Completely Positive, Trace-Preserving operations
- **Kraus operator representation:** $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$
- **Choi-Jamiołkowski isomorphism:** Channel-state duality
- **Channel composition and error accumulation**

The mathematical framework will make rigorous what we've introduced intuitively today!

---

*"The key insight that makes quantum error correction possible is that continuous errors discretize when we measure the syndrome."*
— John Preskill

---

**Day 680 Complete!** Week 98: 1/7 days (14%)
