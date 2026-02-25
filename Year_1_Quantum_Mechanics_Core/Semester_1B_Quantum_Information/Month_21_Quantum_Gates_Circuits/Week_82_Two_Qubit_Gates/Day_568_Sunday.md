# Day 568: The CNOT Gate

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: CNOT definition, matrix form, circuit notation |
| Afternoon | 2.5 hours | Problem solving: CNOT applications and Bell states |
| Evening | 1.5 hours | Computational lab: Entanglement generation |

## Learning Objectives

By the end of today, you will be able to:

1. **Write the CNOT matrix** in the computational basis
2. **Interpret CNOT circuit notation** with control and target qubits
3. **Create Bell states** using CNOT and Hadamard
4. **Understand CNOT as a parity gate** for the target qubit
5. **Verify CNOT properties** including CNOT² = I
6. **Apply CNOT** in basic quantum circuits

---

## Core Content

### 1. Introduction to Two-Qubit Gates

A **two-qubit gate** is a unitary operation on a 4-dimensional Hilbert space:
$$U: \mathcal{H}_2 \otimes \mathcal{H}_2 \to \mathcal{H}_2 \otimes \mathcal{H}_2$$

Represented as a 4×4 unitary matrix acting on the two-qubit computational basis:
$$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$$

### 2. The CNOT Gate Definition

The **Controlled-NOT** (CNOT) gate is the most important two-qubit gate:

$$\boxed{\text{CNOT}|a, b\rangle = |a, a \oplus b\rangle}$$

where $\oplus$ denotes XOR (addition mod 2).

**Truth table:**
| Input | Output |
|-------|--------|
| $\|00\rangle$ | $\|00\rangle$ |
| $\|01\rangle$ | $\|01\rangle$ |
| $\|10\rangle$ | $\|11\rangle$ |
| $\|11\rangle$ | $\|10\rangle$ |

**Interpretation:**
- First qubit is the **control**
- Second qubit is the **target**
- If control = 1, target is flipped (X applied)
- If control = 0, target unchanged

### 3. Matrix Representation

$$\boxed{\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}}$$

The matrix acts on basis states ordered as $|00\rangle, |01\rangle, |10\rangle, |11\rangle$.

**Block structure:**
$$\text{CNOT} = \begin{pmatrix} I & 0 \\ 0 & X \end{pmatrix}$$

This shows: apply I when control=0, apply X when control=1.

### 4. Operator Form

Using projectors and tensor products:

$$\boxed{\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X}$$

**Derivation:**
- $|0\rangle\langle 0|$ projects onto control=0, then applies I to target
- $|1\rangle\langle 1|$ projects onto control=1, then applies X to target

### 5. Circuit Notation

In circuit diagrams:
```
Control: ─────●─────
              │
Target:  ─────⊕─────
```

The control qubit has a solid dot (●), the target has a circled plus (⊕ or XOR symbol).

**Alternative notation:**
```
q0: ───●───
       │
q1: ───X───
```

### 6. Key Properties

**Self-inverse:**
$$\boxed{\text{CNOT}^2 = I}$$

Applying CNOT twice returns to the original state.

**Unitary verification:**
$$\text{CNOT}^\dagger \cdot \text{CNOT} = I$$

**Determinant:**
$$\det(\text{CNOT}) = -1$$

Note: CNOT is unitary but not in SU(4)!

### 7. Creating Entanglement: Bell States

The CNOT gate is the key to creating entanglement:

**Bell state $|\Phi^+\rangle$:**
$$|\Phi^+\rangle = \text{CNOT}(H \otimes I)|00\rangle$$

**Step-by-step:**
1. Start: $|00\rangle$
2. Apply H to first qubit: $(H \otimes I)|00\rangle = |+\rangle|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)|0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$
3. Apply CNOT: $\text{CNOT}\frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$

### 8. The Four Bell States

All Bell states can be created from $|00\rangle$:

$$\boxed{|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = \text{CNOT}(H \otimes I)|00\rangle}$$

$$\boxed{|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle) = \text{CNOT}(H \otimes I)|10\rangle}$$

$$\boxed{|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle) = \text{CNOT}(H \otimes I)|01\rangle}$$

$$\boxed{|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle) = \text{CNOT}(H \otimes I)|11\rangle}$$

### 9. CNOT and Parity

The CNOT gate encodes the **parity** of the control into the target:

After CNOT, the target qubit contains $a \oplus b$ (XOR of original values).

**Parity measurement:** Applying CNOT followed by measuring the target gives the parity of the input.

### 10. CNOT in Different Bases

**In the X-basis** (Hadamard basis):

$$(H \otimes H) \cdot \text{CNOT} \cdot (H \otimes H) = \text{CNOT}^{\text{reversed}}$$

The conjugation by H⊗H **swaps control and target**!

This leads to the identity:
$$\text{CNOT}_{1\to 2} = (H \otimes H) \cdot \text{CNOT}_{2\to 1} \cdot (H \otimes H)$$

### 11. Action on Product States

For product states $|\psi\rangle|\phi\rangle$:

$$\text{CNOT}|\psi\rangle|\phi\rangle$$

where $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ and $|\phi\rangle = \gamma|0\rangle + \delta|1\rangle$:

$$= \alpha|0\rangle(\gamma|0\rangle + \delta|1\rangle) + \beta|1\rangle(\gamma|1\rangle + \delta|0\rangle)$$
$$= \alpha\gamma|00\rangle + \alpha\delta|01\rangle + \beta\delta|10\rangle + \beta\gamma|11\rangle$$

The result is generally **entangled** (not a product state)!

---

## Quantum Computing Connection

The CNOT gate is foundational to quantum computing:

1. **Universal computation:** {Single-qubit gates, CNOT} is universal
2. **Entanglement resource:** Creates the non-classical correlations needed for quantum advantage
3. **Error correction:** CNOT is used for syndrome extraction in quantum error correcting codes
4. **Quantum teleportation:** Bell measurement uses CNOT
5. **Native gate:** CNOT (or CZ) is a native gate on most quantum hardware

---

## Worked Examples

### Example 1: Verify CNOT² = I

**Problem:** Show that applying CNOT twice gives the identity.

**Solution:**
$$\text{CNOT}^2 = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}^2 = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} = I_4$$

For basis states: $\text{CNOT}|10\rangle = |11\rangle$, then $\text{CNOT}|11\rangle = |10\rangle$ ✓

### Example 2: Create |Ψ⁺⟩

**Problem:** Show that $|\Psi^+\rangle = \text{CNOT}(H \otimes I)|01\rangle$.

**Solution:**

**Step 1:** Apply H ⊗ I to |01⟩:
$$(H \otimes I)|01\rangle = H|0\rangle \otimes |1\rangle = |+\rangle|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)|1\rangle$$
$$= \frac{1}{\sqrt{2}}(|01\rangle + |11\rangle)$$

**Step 2:** Apply CNOT:
$$\text{CNOT}\frac{1}{\sqrt{2}}(|01\rangle + |11\rangle) = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle) = |\Psi^+\rangle$$

### Example 3: CNOT on Superposition

**Problem:** Compute $\text{CNOT}|+\rangle|+\rangle$.

**Solution:**

$$|+\rangle|+\rangle = \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle + |1\rangle) = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$

Applying CNOT:
$$\text{CNOT}|+\rangle|+\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |11\rangle + |10\rangle)$$
$$= \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle) = |+\rangle|+\rangle$$

**Surprise!** The state is unchanged! This is because $|+\rangle|+\rangle$ is an eigenstate of CNOT with eigenvalue +1.

---

## Practice Problems

### Direct Application

1. Compute CNOT$|10\rangle$ and CNOT$|11\rangle$ directly.

2. Verify that the CNOT matrix is unitary by computing CNOT†·CNOT.

3. Write CNOT in the form $\sum_{i,j} c_{ij}|i\rangle\langle j|$ using outer products.

### Intermediate

4. **Reversed CNOT:** Find the matrix for CNOT with target as first qubit and control as second: $\text{CNOT}_{2\to 1}|a,b\rangle = |a \oplus b, b\rangle$.

5. Prove that $(H \otimes H)\text{CNOT}(H \otimes H) = \text{CNOT}^{\text{reversed}}$.

6. **Bell measurement:** Show that applying CNOT followed by H ⊗ I transforms the Bell states to the computational basis states.

### Challenging

7. **Eigenvalues:** Find all eigenvalues and eigenstates of CNOT. Verify that the Bell states are eigenstates.

8. **Three-qubit extension:** The Toffoli gate (CCNOT) flips the third qubit if both first two are 1. Write its 8×8 matrix.

9. **CNOT decomposition:** Express CNOT using only CZ and Hadamard gates: $\text{CNOT} = (I \otimes H) \cdot \text{CZ} \cdot (I \otimes H)$. Verify this identity.

---

## Computational Lab: CNOT and Entanglement

```python
"""
Day 568: CNOT Gate and Entanglement
Exploring the fundamental two-qubit gate
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Single-qubit gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Two-qubit identity
I4 = np.eye(4, dtype=complex)

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

def ket(bitstring):
    """Create computational basis state from bitstring."""
    state = np.array([[1]], dtype=complex)
    for bit in bitstring:
        state = np.kron(state, ket_0 if bit == '0' else ket_1)
    return state

# Define CNOT gate
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

print("=" * 60)
print("CNOT GATE FUNDAMENTALS")
print("=" * 60)

# 1. Display CNOT matrix
print("\n1. CNOT Matrix:")
print(CNOT)

# 2. Verify CNOT properties
print("\n2. CNOT Properties:")
print(f"   CNOT² = I: {np.allclose(CNOT @ CNOT, I4)}")
print(f"   CNOT† = CNOT: {np.allclose(CNOT.conj().T, CNOT)}")
print(f"   Unitary (CNOT†·CNOT = I): {np.allclose(CNOT.conj().T @ CNOT, I4)}")
print(f"   det(CNOT) = {np.linalg.det(CNOT):.1f}")

# 3. Action on computational basis
print("\n3. Action on Computational Basis:")
basis_states = ['00', '01', '10', '11']
for bs in basis_states:
    input_state = ket(bs)
    output_state = CNOT @ input_state
    # Find which basis state the output is
    for out_bs in basis_states:
        if np.allclose(output_state, ket(out_bs)):
            print(f"   CNOT|{bs}⟩ = |{out_bs}⟩")
            break

# 4. Operator form verification
print("\n4. Operator Form Verification:")
proj_0 = np.outer(ket_0, ket_0.conj())
proj_1 = np.outer(ket_1, ket_1.conj())
CNOT_operator = np.kron(proj_0, I) + np.kron(proj_1, X)
print(f"   |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X = CNOT: {np.allclose(CNOT_operator, CNOT)}")

# 5. Bell state creation
print("\n" + "=" * 60)
print("BELL STATE CREATION")
print("=" * 60)

# Circuit: |ab⟩ → (H⊗I)|ab⟩ → CNOT(H⊗I)|ab⟩ = Bell state
H_I = np.kron(H, I)  # H on first qubit

bell_states = {}
bell_names = ['Φ⁺', 'Ψ⁺', 'Φ⁻', 'Ψ⁻']
input_states = ['00', '01', '10', '11']

print("\n5. Creating Bell States:")
for inp, name in zip(input_states, ['Φ⁺', 'Φ⁻', 'Ψ⁺', 'Ψ⁻']):
    state = CNOT @ H_I @ ket(inp)
    bell_states[name] = state
    print(f"   |{name}⟩ = CNOT(H⊗I)|{inp}⟩:")
    for i, bs in enumerate(basis_states):
        amp = state[i, 0]
        if np.abs(amp) > 1e-10:
            print(f"      {amp.real:+.4f}|{bs}⟩", end="")
    print()

# 6. Verify Bell states are entangled (check if separable)
print("\n6. Entanglement Verification:")

def is_separable(state):
    """Check if a 2-qubit state is separable using Schmidt decomposition."""
    # Reshape to 2x2 matrix
    psi = state.reshape(2, 2)
    # Compute singular values
    _, s, _ = np.linalg.svd(psi)
    # Separable if rank 1 (one non-zero singular value)
    return np.sum(s > 1e-10) == 1

def schmidt_number(state):
    """Return Schmidt number (rank of coefficient matrix)."""
    psi = state.reshape(2, 2)
    _, s, _ = np.linalg.svd(psi)
    return np.sum(s > 1e-10)

for name, state in bell_states.items():
    sep = is_separable(state)
    schmidt = schmidt_number(state)
    print(f"   |{name}⟩: Separable = {sep}, Schmidt number = {schmidt}")

# 7. Eigenvalue analysis
print("\n" + "=" * 60)
print("CNOT EIGENVALUE ANALYSIS")
print("=" * 60)

eigenvalues, eigenvectors = np.linalg.eig(CNOT)
print("\n7. Eigenvalues of CNOT:")
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    print(f"   λ_{i+1} = {val.real:+.4f}")

print("\n8. Bell States as Eigenstates:")
for name, state in bell_states.items():
    result = CNOT @ state
    # Find eigenvalue
    for i in range(4):
        if np.abs(state[i, 0]) > 1e-10:
            ratio = result[i, 0] / state[i, 0]
            break
    print(f"   CNOT|{name}⟩ = {ratio.real:+.4f}|{name}⟩")

# 9. Reversed CNOT
print("\n" + "=" * 60)
print("REVERSED CNOT (TARGET ↔ CONTROL)")
print("=" * 60)

# CNOT with control on second qubit
CNOT_reversed = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
], dtype=complex)

print("\n9. CNOT_reversed matrix (control=qubit 2, target=qubit 1):")
print(CNOT_reversed)

# Verify H⊗H conjugation
H_H = np.kron(H, H)
CNOT_via_conjugation = H_H @ CNOT @ H_H
print(f"\n   (H⊗H)·CNOT·(H⊗H) = CNOT_reversed: {np.allclose(CNOT_via_conjugation, CNOT_reversed)}")

# 10. CNOT on superposition states
print("\n" + "=" * 60)
print("CNOT ON SUPERPOSITION STATES")
print("=" * 60)

ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

test_states = [
    ('|+⟩|0⟩', np.kron(ket_plus, ket_0)),
    ('|+⟩|1⟩', np.kron(ket_plus, ket_1)),
    ('|+⟩|+⟩', np.kron(ket_plus, ket_plus)),
    ('|0⟩|+⟩', np.kron(ket_0, ket_plus)),
]

print("\n10. CNOT on Various States:")
for name, state in test_states:
    output = CNOT @ state
    sep = is_separable(output)
    print(f"   CNOT{name}: Separable = {sep}")

# Visualization
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: CNOT truth table as circuit
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)

# Draw circuit
ax1.plot([1, 4], [4, 4], 'k-', linewidth=2)  # Control wire
ax1.plot([1, 4], [2, 2], 'k-', linewidth=2)  # Target wire
ax1.plot([2.5, 2.5], [2, 4], 'k-', linewidth=2)  # Vertical line
ax1.scatter([2.5], [4], s=100, c='black', zorder=5)  # Control dot
circle = plt.Circle((2.5, 2), 0.3, fill=False, linewidth=2)
ax1.add_patch(circle)
ax1.plot([2.2, 2.8], [2, 2], 'k-', linewidth=2)  # Plus horizontal
ax1.plot([2.5, 2.5], [1.7, 2.3], 'k-', linewidth=2)  # Plus vertical

ax1.text(0.5, 4, 'Control', fontsize=12, va='center')
ax1.text(0.5, 2, 'Target', fontsize=12, va='center')
ax1.text(5, 4, '|a⟩', fontsize=12, va='center')
ax1.text(5, 2, '|a⊕b⟩', fontsize=12, va='center')

ax1.set_title('CNOT Circuit Diagram', fontsize=14)
ax1.axis('off')

# Plot 2: CNOT matrix visualization
ax2 = axes[0, 1]
im = ax2.imshow(np.abs(CNOT), cmap='Blues', vmin=0, vmax=1)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
ax2.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
ax2.set_title('CNOT Matrix (absolute values)', fontsize=14)
for i in range(4):
    for j in range(4):
        val = CNOT[i, j].real
        if np.abs(val) > 0.5:
            ax2.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=14)
plt.colorbar(im, ax=ax2)

# Plot 3: Bell state creation circuit
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 6)

# Input
ax3.text(0.5, 4, '|0⟩', fontsize=12, va='center')
ax3.text(0.5, 2, '|0⟩', fontsize=12, va='center')

# Wires
ax3.plot([1, 8], [4, 4], 'k-', linewidth=2)
ax3.plot([1, 8], [2, 2], 'k-', linewidth=2)

# H gate
rect = plt.Rectangle((2, 3.5), 1, 1, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
ax3.add_patch(rect)
ax3.text(2.5, 4, 'H', ha='center', va='center', fontsize=14)

# CNOT
ax3.plot([5, 5], [2, 4], 'k-', linewidth=2)
ax3.scatter([5], [4], s=100, c='black', zorder=5)
circle = plt.Circle((5, 2), 0.3, fill=False, linewidth=2)
ax3.add_patch(circle)
ax3.plot([4.7, 5.3], [2, 2], 'k-', linewidth=2)
ax3.plot([5, 5], [1.7, 2.3], 'k-', linewidth=2)

# Output
ax3.text(8.5, 3, '|Φ⁺⟩', fontsize=14, va='center')

ax3.set_title('Bell State |Φ⁺⟩ Creation Circuit', fontsize=14)
ax3.axis('off')

# Plot 4: Bell state amplitudes
ax4 = axes[1, 1]
x = np.arange(4)
width = 0.2
colors = ['blue', 'green', 'red', 'purple']

for i, (name, state) in enumerate(bell_states.items()):
    amps = np.abs(state.flatten())**2
    ax4.bar(x + i*width - 0.3, amps, width, label=f'|{name}⟩', color=colors[i], alpha=0.7)

ax4.set_xticks(x)
ax4.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
ax4.set_ylabel('Probability', fontsize=12)
ax4.set_title('Bell State Probability Distributions', fontsize=14)
ax4.legend()
ax4.set_ylim(0, 0.7)

plt.tight_layout()
plt.savefig('cnot_fundamentals.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: cnot_fundamentals.png")

# CNOT entanglement demo
print("\n" + "=" * 60)
print("ENTANGLEMENT DEMO")
print("=" * 60)

def concurrence(state):
    """Compute concurrence of a pure 2-qubit state."""
    # For pure states: C = 2|ad - bc| where state = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩
    state = state.flatten()
    a, b, c, d = state
    return 2 * np.abs(a*d - b*c)

print("\n11. Entanglement (Concurrence) for Various States:")

states_to_check = [
    ('|00⟩', ket('00')),
    ('|+⟩|0⟩', np.kron(ket_plus, ket_0)),
    ('|Φ⁺⟩', bell_states['Φ⁺']),
    ('|Ψ⁺⟩', bell_states['Ψ⁺']),
    ('CNOT|+⟩|0⟩', CNOT @ np.kron(ket_plus, ket_0)),
    ('CNOT|0⟩|+⟩', CNOT @ np.kron(ket_0, ket_plus)),
]

for name, state in states_to_check:
    C = concurrence(state)
    print(f"   {name:20}: Concurrence = {C:.4f} ({'maximally entangled' if np.isclose(C, 1) else 'separable' if np.isclose(C, 0) else 'partially entangled'})")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| CNOT action | $\text{CNOT}\|a,b\rangle = \|a, a \oplus b\rangle$ |
| CNOT matrix | $\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$ |
| Operator form | $\|0\rangle\langle 0\| \otimes I + \|1\rangle\langle 1\| \otimes X$ |
| Self-inverse | $\text{CNOT}^2 = I$ |
| Bell state | $\|\Phi^+\rangle = \text{CNOT}(H \otimes I)\|00\rangle$ |
| Control/target swap | $(H \otimes H)\text{CNOT}(H \otimes H) = \text{CNOT}^{\text{rev}}$ |

### Main Takeaways

1. **CNOT is the canonical entangling gate:** Flips target if control is |1⟩
2. **Creates Bell states:** Combined with H, produces maximally entangled states
3. **Self-inverse:** Applying twice returns to original state
4. **Parity encoding:** Target qubit becomes XOR of both inputs
5. **Universal:** {Single-qubit gates, CNOT} gives universal quantum computation

---

## Daily Checklist

- [ ] I can write the CNOT matrix from memory
- [ ] I understand control and target qubit roles
- [ ] I can create all four Bell states using CNOT and H
- [ ] I verified CNOT² = I
- [ ] I understand the control/target swap identity
- [ ] I completed the computational lab
- [ ] I solved at least 3 practice problems

---

## Preview of Day 569

Tomorrow we study **controlled gates** in general: CZ (controlled-Z), controlled-U for arbitrary U, and the mathematical formalism of control qubit operations. We'll see how CNOT is just one example of a larger family of controlled operations.
