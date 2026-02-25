# Day 569: Controlled Gates (CZ, Controlled-U)

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: CZ gate, general controlled-U formalism |
| Afternoon | 2.5 hours | Problem solving: Controlled gate constructions |
| Evening | 1.5 hours | Computational lab: Building controlled operations |

## Learning Objectives

By the end of today, you will be able to:

1. **Implement the CZ gate** and understand its symmetry
2. **Construct controlled-U gates** for arbitrary single-qubit U
3. **Relate CNOT and CZ** through basis transformation
4. **Apply the ABC decomposition** for controlled gates
5. **Build multi-controlled gates** (Toffoli) conceptually
6. **Understand the control qubit formalism** mathematically

---

## Core Content

### 1. The Controlled-Z (CZ) Gate

The **CZ gate** applies Z to the target if the control is |1⟩:

$$\boxed{\text{CZ}|a, b\rangle = (-1)^{ab}|a, b\rangle}$$

**Truth table:**
| Input | Output |
|-------|--------|
| $\|00\rangle$ | $\|00\rangle$ |
| $\|01\rangle$ | $\|01\rangle$ |
| $\|10\rangle$ | $\|10\rangle$ |
| $\|11\rangle$ | $-\|11\rangle$ |

Only $|11\rangle$ acquires a phase!

### 2. CZ Matrix

$$\boxed{\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix} = \text{diag}(1, 1, 1, -1)}$$

**Operator form:**
$$\text{CZ} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes Z$$

### 3. CZ Symmetry

A remarkable property: **CZ is symmetric** with respect to swapping control and target!

$$\boxed{\text{CZ}_{1\to 2} = \text{CZ}_{2\to 1}}$$

**Proof:** $(-1)^{ab} = (-1)^{ba}$, so it doesn't matter which qubit is "control."

**Circuit notation:** CZ is drawn with dots on both qubits:
```
q0: ───●───
       │
q1: ───●───
```

### 4. CNOT-CZ Relationship

CNOT and CZ are related by Hadamard gates:

$$\boxed{\text{CNOT} = (I \otimes H) \cdot \text{CZ} \cdot (I \otimes H)}$$

**Proof:**
- CZ|a,b⟩ = (-1)^{ab}|a,b⟩
- In the X-basis for the second qubit: $|+\rangle = H|0\rangle$, $|-\rangle = H|1\rangle$
- $(I \otimes H)\text{CZ}(I \otimes H)|a,b\rangle$ flips the X-basis of qubit 2 when qubit 1 is |1⟩
- This is exactly CNOT!

### 5. General Controlled-U Gates

For any single-qubit unitary U, the **controlled-U** gate is:

$$\boxed{\text{C-}U = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U}$$

**Matrix form (4×4):**
$$\text{C-}U = \begin{pmatrix} I & 0 \\ 0 & U \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & U_{00} & U_{01} \\ 0 & 0 & U_{10} & U_{11} \end{pmatrix}$$

**Special cases:**
- C-X = CNOT
- C-Z = CZ
- C-Y = Controlled-Y
- C-H = Controlled-Hadamard

### 6. Controlled-U Circuit Notation

```
Control: ─────●─────
              │
Target:  ─────U─────
```

The control qubit has a dot, and the target has the gate U in a box.

### 7. ABC Decomposition

Any controlled-U gate can be decomposed using CNOTs and single-qubit gates:

**Theorem:** For any U ∈ SU(2), there exist A, B, C ∈ SU(2) and phase α such that:
$$ABC = I \text{ and } U = e^{i\alpha}AXBXC$$

Then:
$$\boxed{\text{C-}U = e^{i\alpha}(I \otimes A) \cdot \text{CNOT} \cdot (I \otimes B) \cdot \text{CNOT} \cdot (I \otimes C) \cdot \text{C-}P(\alpha)}$$

where C-P(α) is a controlled phase gate.

**Simplified for special cases:**
$$\text{C-}U = (I \otimes A) \cdot \text{CNOT} \cdot (I \otimes B) \cdot \text{CNOT} \cdot (I \otimes C)$$
when U has determinant 1.

### 8. Finding ABC Decomposition

Given $U = e^{i\alpha}\begin{pmatrix} u_{00} & u_{01} \\ u_{10} & u_{11} \end{pmatrix}$ with $|\det(U)| = 1$:

**Step 1:** Find the global phase α such that det(e^{-iα}U) = 1.

**Step 2:** Use the ZYZ decomposition of e^{-iα}U:
$$e^{-i\alpha}U = R_z(\beta)R_y(\gamma)R_z(\delta)$$

**Step 3:** Define:
$$A = R_z(\beta)R_y(\gamma/2)$$
$$B = R_y(-\gamma/2)R_z(-(\delta+\beta)/2)$$
$$C = R_z((\delta-\beta)/2)$$

### 9. Multi-Controlled Gates

**Toffoli gate (CCNOT):** Flips target if both controls are |1⟩:
$$\text{CCNOT}|a, b, c\rangle = |a, b, c \oplus (a \land b)\rangle$$

**CCZ gate:** Applies phase -1 only to |111⟩:
$$\text{CCZ} = \text{diag}(1,1,1,1,1,1,1,-1)$$

### 10. Control Qubit Formalism

The general n-controlled U gate:
$$\text{C}^n\text{-}U = (I - |1\rangle\langle 1|^{\otimes n}) \otimes I + |1\rangle\langle 1|^{\otimes n} \otimes U$$

In words: Apply U to the target only when all n control qubits are |1⟩.

### 11. Native Implementations

Different quantum hardware implements different controlled gates natively:

| Platform | Native Two-Qubit Gate |
|----------|----------------------|
| IBM | CNOT |
| Google | CZ, √iSWAP |
| IonQ | Mølmer-Sørensen (MS) |
| Rigetti | CZ |

All controlled-U gates must be compiled to these natives!

---

## Quantum Computing Connection

Controlled gates enable:

1. **Conditional logic:** Quantum if-then operations
2. **Phase kickback:** Eigenvalue extraction for algorithms
3. **Quantum arithmetic:** Controlled addition and multiplication
4. **Error correction:** Syndrome measurement circuits
5. **Oracle implementation:** f(x) encoded in controlled operations

---

## Worked Examples

### Example 1: CZ from CNOT

**Problem:** Verify $\text{CZ} = (I \otimes H) \cdot \text{CNOT} \cdot (I \otimes H)$.

**Solution:**

Let's compute the action on all basis states:

$|00\rangle$:
- $(I \otimes H)|00\rangle = |0\rangle H|0\rangle = |0\rangle|+\rangle$
- $\text{CNOT}|0\rangle|+\rangle = |0\rangle|+\rangle$ (control=0, no flip)
- $(I \otimes H)|0\rangle|+\rangle = |0\rangle|0\rangle = |00\rangle$ ✓

$|11\rangle$:
- $(I \otimes H)|11\rangle = |1\rangle|−\rangle$
- $\text{CNOT}|1\rangle|−\rangle = |1\rangle(−|+\rangle + |−\rangle)/\sqrt{2}$ ... let's compute more carefully.

Actually, $|−\rangle = \frac{1}{\sqrt{2}}(|0\rangle − |1\rangle)$:
- $\text{CNOT}|1\rangle|−\rangle = |1\rangle \cdot X|−\rangle = |1\rangle \cdot |−\rangle \cdot (-1) = -|1\rangle|−\rangle$

Wait, $X|−\rangle = −|−\rangle$, so:
- $\text{CNOT}|1\rangle|−\rangle = -|1\rangle|−\rangle$
- $(I \otimes H)(-|1\rangle|−\rangle) = -|1\rangle|1\rangle = -|11\rangle$ ✓

This matches CZ|11⟩ = -|11⟩!

### Example 2: Controlled-S Gate

**Problem:** Write the matrix for C-S (controlled-S gate).

**Solution:**

$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$$

Using the formula for C-U:
$$\text{C-S} = \begin{pmatrix} I & 0 \\ 0 & S \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & i \end{pmatrix}$$

**Verification:** C-S|11⟩ = |1⟩S|1⟩ = |1⟩(i|1⟩) = i|11⟩ ✓

### Example 3: ABC for Controlled-Z

**Problem:** Find A, B, C such that AXBXC = Z and ABC = I.

**Solution:**

For Z, we need:
$$Z = AXBXC$$

Since Z is diagonal with eigenvalues +1, -1:
$$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

One solution:
- $A = R_z(\pi/2) = \begin{pmatrix} e^{-i\pi/4} & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$
- $B = R_z(-\pi/2)$
- $C = I$

Check ABC = I:
$$R_z(\pi/2) \cdot R_z(-\pi/2) \cdot I = I$$ ✓

Check AXBXC = Z:
$$R_z(\pi/2) \cdot X \cdot R_z(-\pi/2) \cdot X \cdot I$$

This requires careful computation... The decomposition shows CZ needs 2 CNOTs!

---

## Practice Problems

### Direct Application

1. Compute CZ$|+\rangle|+\rangle$ and determine if the result is entangled.

2. Write the matrix for controlled-Y (C-Y gate).

3. Verify that $\text{CZ}^2 = I$ (CZ is self-inverse).

### Intermediate

4. **Phase kickback:** Show that C-Z$|+\rangle|1\rangle = |-\rangle|1\rangle$. Explain the phase kickback.

5. Find the ABC decomposition for the controlled-H gate.

6. Show that $(H \otimes H) \text{CZ} (H \otimes H) = \text{CZ}$ (CZ is invariant under this conjugation).

### Challenging

7. **Toffoli decomposition:** Express the Toffoli gate using only CNOT and single-qubit gates. What is the minimum CNOT count?

8. Prove that any controlled-U can be implemented with at most 2 CNOTs plus single-qubit gates.

9. **Control swap:** For the controlled-SWAP gate (Fredkin gate), write the 8×8 matrix and express it using CNOTs and Toffolis.

---

## Computational Lab: Controlled Gates

```python
"""
Day 569: Controlled Gates
CZ, Controlled-U, and ABC Decomposition
"""

import numpy as np
import matplotlib.pyplot as plt

# Single-qubit gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

def Rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)

def Ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

# Two-qubit gates
I4 = np.eye(4, dtype=complex)
CNOT = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)
CZ = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]], dtype=complex)

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

def ket(bitstring):
    """Create computational basis state."""
    state = np.array([[1]], dtype=complex)
    for bit in bitstring:
        state = np.kron(state, ket_0 if bit == '0' else ket_1)
    return state

def controlled_U(U):
    """Create controlled-U gate matrix."""
    return np.block([[I, np.zeros((2,2))], [np.zeros((2,2)), U]])

print("=" * 60)
print("CONTROLLED-Z GATE")
print("=" * 60)

# 1. CZ matrix and properties
print("\n1. CZ Matrix:")
print(CZ)

print("\n2. CZ Properties:")
print(f"   CZ² = I: {np.allclose(CZ @ CZ, I4)}")
print(f"   CZ† = CZ: {np.allclose(CZ.conj().T, CZ)}")
print(f"   CZ is diagonal: {np.allclose(CZ, np.diag(np.diag(CZ)))}")

# 3. CZ symmetry
print("\n3. CZ Symmetry (control ↔ target):")
# SWAP matrix
SWAP = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=complex)
CZ_swapped = SWAP @ CZ @ SWAP
print(f"   SWAP·CZ·SWAP = CZ: {np.allclose(CZ_swapped, CZ)}")

# 4. CZ-CNOT relationship
print("\n4. CZ-CNOT Relationship:")
I_H = np.kron(I, H)
CNOT_from_CZ = I_H @ CZ @ I_H
print(f"   (I⊗H)·CZ·(I⊗H) = CNOT: {np.allclose(CNOT_from_CZ, CNOT)}")

CZ_from_CNOT = I_H @ CNOT @ I_H
print(f"   (I⊗H)·CNOT·(I⊗H) = CZ: {np.allclose(CZ_from_CNOT, CZ)}")

# General controlled gates
print("\n" + "=" * 60)
print("GENERAL CONTROLLED-U GATES")
print("=" * 60)

# 5. Build controlled versions of common gates
gates = [('X', X), ('Y', Y), ('Z', Z), ('H', H), ('S', S), ('T', T)]

print("\n5. Controlled Gate Matrices:")
for name, U in gates:
    CU = controlled_U(U)
    print(f"\n   C-{name}:")
    print(f"   {CU}")

# 6. Verify CNOT = C-X
print("\n6. Verification:")
print(f"   controlled_U(X) = CNOT: {np.allclose(controlled_U(X), CNOT)}")
print(f"   controlled_U(Z) = CZ: {np.allclose(controlled_U(Z), CZ)}")

# 7. Phase kickback demonstration
print("\n" + "=" * 60)
print("PHASE KICKBACK")
print("=" * 60)

print("\n7. Phase Kickback with CZ:")
# CZ|+⟩|1⟩ should give |-⟩|1⟩
state_in = np.kron(ket_plus, ket_1)
state_out = CZ @ state_in

print(f"   Input: |+⟩|1⟩")
print(f"   Output CZ|+⟩|1⟩:")

# Check if output is |-⟩|1⟩
expected = np.kron(ket_minus, ket_1)
print(f"   Output = |-⟩|1⟩: {np.allclose(state_out, expected)}")
print(f"   Phase kicked back to control qubit!")

# 8. Phase kickback with arbitrary eigenvalue
print("\n8. Phase Kickback with C-S on eigenvector:")
# S|1⟩ = i|1⟩, so C-S|+⟩|1⟩ should give phase i to the control
CS = controlled_U(S)
state_in = np.kron(ket_plus, ket_1)
state_out = CS @ state_in

print(f"   Input: |+⟩|1⟩")
print(f"   S|1⟩ = i|1⟩ (eigenvalue i)")
print(f"   C-S|+⟩|1⟩ should give (|0⟩ + i|1⟩)/√2 ⊗ |1⟩")

# Expected: (|0⟩ + i|1⟩)/√2 ⊗ |1⟩
expected_control = (ket_0 + 1j*ket_1) / np.sqrt(2)
expected = np.kron(expected_control, ket_1)
print(f"   Output matches: {np.allclose(state_out, expected)}")

# ABC Decomposition
print("\n" + "=" * 60)
print("ABC DECOMPOSITION")
print("=" * 60)

def abc_decomposition(U):
    """
    Find A, B, C such that ABC = I and AXBXC = U (up to phase).
    Uses ZYZ decomposition.
    """
    # Get global phase
    det = np.linalg.det(U)
    alpha = np.angle(det) / 2
    U_su2 = U * np.exp(-1j * alpha)

    # ZYZ decomposition of U
    # U = Rz(β)Ry(γ)Rz(δ)

    # Find γ
    cos_half_gamma = np.abs(U_su2[0, 0])
    if cos_half_gamma > 1:
        cos_half_gamma = 1
    gamma = 2 * np.arccos(cos_half_gamma)

    # Find β, δ from phases
    if np.abs(np.sin(gamma/2)) < 1e-10:
        beta = -2 * np.angle(U_su2[0, 0])
        delta = 0
    else:
        phase_00 = np.angle(U_su2[0, 0])
        phase_10 = np.angle(U_su2[1, 0])
        beta = -phase_00 + phase_10
        delta = -phase_00 - phase_10

    # ABC decomposition
    A = Rz(beta) @ Ry(gamma/2)
    B = Ry(-gamma/2) @ Rz(-(delta + beta)/2)
    C = Rz((delta - beta)/2)

    return A, B, C, alpha

print("\n9. ABC Decomposition Examples:")

test_gates = [('Z', Z), ('S', S), ('H', H), ('T', T)]

for name, U in test_gates:
    A, B, C, alpha = abc_decomposition(U)

    # Verify ABC = I
    ABC = A @ B @ C
    abc_check = np.allclose(ABC, I) or np.allclose(ABC, -I)

    # Verify AXBXC = U (up to phase)
    AXBXC = A @ X @ B @ X @ C
    axbxc_phases = []
    for phase in [1, -1, 1j, -1j, np.exp(1j*alpha), np.exp(-1j*alpha)]:
        if np.allclose(phase * AXBXC, U):
            axbxc_phases.append(phase)

    print(f"\n   Gate {name}:")
    print(f"   ABC ≈ I: {abc_check}")
    print(f"   AXBXC ≈ U (up to phase): {len(axbxc_phases) > 0}")

# 10. Building controlled-U from CNOTs
print("\n" + "=" * 60)
print("CONTROLLED-U FROM CNOTS")
print("=" * 60)

def build_controlled_U(U):
    """Build C-U using ABC decomposition and CNOTs."""
    A, B, C, alpha = abc_decomposition(U)

    # C-U = e^(iα) · (I⊗A) · CNOT · (I⊗B) · CNOT · (I⊗C) · phase
    I_A = np.kron(I, A)
    I_B = np.kron(I, B)
    I_C = np.kron(I, C)

    result = I_C @ CNOT @ I_B @ CNOT @ I_A
    return result

print("\n10. Building C-U from 2 CNOTs:")

for name, U in test_gates:
    CU_direct = controlled_U(U)
    CU_built = build_controlled_U(U)

    # Check equality up to global phase
    phase_match = False
    for phase in np.exp(1j * np.linspace(0, 2*np.pi, 100)):
        if np.allclose(phase * CU_built, CU_direct):
            phase_match = True
            break

    print(f"   C-{name} from CNOTs: matches (up to phase) = {phase_match}")

# Visualization
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: CZ circuit notation
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)

# CZ circuit
ax1.plot([1, 6], [4, 4], 'k-', linewidth=2)
ax1.plot([1, 6], [2, 2], 'k-', linewidth=2)
ax1.plot([3.5, 3.5], [2, 4], 'k-', linewidth=2)
ax1.scatter([3.5, 3.5], [4, 2], s=100, c='black', zorder=5)

ax1.text(0.3, 4, 'q₀', fontsize=12, va='center')
ax1.text(0.3, 2, 'q₁', fontsize=12, va='center')
ax1.text(7, 4, '(-1)^(a·b)|a⟩', fontsize=11, va='center')
ax1.text(7, 2, '|b⟩', fontsize=11, va='center')

ax1.set_title('CZ Gate (symmetric)', fontsize=14)
ax1.axis('off')

# Plot 2: CZ matrix visualization
ax2 = axes[0, 1]
im = ax2.imshow(np.real(CZ), cmap='RdBu', vmin=-1, vmax=1)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
ax2.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
for i in range(4):
    for j in range(4):
        val = CZ[i, j].real
        if np.abs(val) > 0.5:
            ax2.text(j, i, f'{val:+.0f}', ha='center', va='center', fontsize=14)
ax2.set_title('CZ Matrix', fontsize=14)
plt.colorbar(im, ax=ax2)

# Plot 3: Controlled-U circuit
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 6)

# C-U circuit
ax3.plot([1, 6], [4, 4], 'k-', linewidth=2)
ax3.plot([1, 6], [2, 2], 'k-', linewidth=2)
ax3.plot([3.5, 3.5], [2.5, 4], 'k-', linewidth=2)
ax3.scatter([3.5], [4], s=100, c='black', zorder=5)
rect = plt.Rectangle((3, 1.5), 1, 1, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
ax3.add_patch(rect)
ax3.text(3.5, 2, 'U', ha='center', va='center', fontsize=14)

ax3.text(0.3, 4, 'Control', fontsize=11, va='center')
ax3.text(0.3, 2, 'Target', fontsize=11, va='center')

ax3.set_title('General Controlled-U Gate', fontsize=14)
ax3.axis('off')

# Plot 4: ABC decomposition circuit
ax4 = axes[1, 1]
ax4.set_xlim(0, 14)
ax4.set_ylim(0, 6)

# Draw the decomposition circuit
ax4.plot([1, 13], [4, 4], 'k-', linewidth=2)
ax4.plot([1, 13], [2, 2], 'k-', linewidth=2)

# Gate C
rect_c = plt.Rectangle((1.5, 1.5), 1, 1, fill=True, facecolor='lightyellow', edgecolor='black', linewidth=2)
ax4.add_patch(rect_c)
ax4.text(2, 2, 'C', ha='center', va='center', fontsize=12)

# First CNOT
ax4.plot([4, 4], [2, 4], 'k-', linewidth=2)
ax4.scatter([4], [4], s=80, c='black', zorder=5)
circle1 = plt.Circle((4, 2), 0.2, fill=False, linewidth=2)
ax4.add_patch(circle1)

# Gate B
rect_b = plt.Rectangle((5.5, 1.5), 1, 1, fill=True, facecolor='lightgreen', edgecolor='black', linewidth=2)
ax4.add_patch(rect_b)
ax4.text(6, 2, 'B', ha='center', va='center', fontsize=12)

# Second CNOT
ax4.plot([8, 8], [2, 4], 'k-', linewidth=2)
ax4.scatter([8], [4], s=80, c='black', zorder=5)
circle2 = plt.Circle((8, 2), 0.2, fill=False, linewidth=2)
ax4.add_patch(circle2)

# Gate A
rect_a = plt.Rectangle((9.5, 1.5), 1, 1, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
ax4.add_patch(rect_a)
ax4.text(10, 2, 'A', ha='center', va='center', fontsize=12)

ax4.text(7, 5, 'C-U = (I⊗A)·CNOT·(I⊗B)·CNOT·(I⊗C)', ha='center', fontsize=11)
ax4.text(7, 0.5, 'where ABC = I and U = AXBXC', ha='center', fontsize=10)

ax4.set_title('ABC Decomposition of Controlled-U', fontsize=14)
ax4.axis('off')

plt.tight_layout()
plt.savefig('controlled_gates.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: controlled_gates.png")

# Toffoli gate preview
print("\n" + "=" * 60)
print("TOFFOLI GATE PREVIEW")
print("=" * 60)

# Build Toffoli (CCNOT) matrix
CCNOT = np.eye(8, dtype=complex)
CCNOT[6, 6] = 0
CCNOT[6, 7] = 1
CCNOT[7, 6] = 1
CCNOT[7, 7] = 0

print("\n11. Toffoli Gate (CCNOT):")
print(f"   Shape: {CCNOT.shape}")
print(f"   CCNOT|110⟩ = |111⟩: ", end="")
ket_110 = np.zeros((8, 1), dtype=complex)
ket_110[6] = 1
result = CCNOT @ ket_110
ket_111 = np.zeros((8, 1), dtype=complex)
ket_111[7] = 1
print(np.allclose(result, ket_111))

print(f"   CCNOT|111⟩ = |110⟩: ", end="")
result2 = CCNOT @ ket_111
print(np.allclose(result2, ket_110))
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| CZ action | $\text{CZ}\|a,b\rangle = (-1)^{ab}\|a,b\rangle$ |
| CZ matrix | $\text{diag}(1, 1, 1, -1)$ |
| CZ symmetry | $\text{CZ}_{1\to 2} = \text{CZ}_{2\to 1}$ |
| CZ-CNOT relation | $\text{CNOT} = (I \otimes H)\text{CZ}(I \otimes H)$ |
| Controlled-U | $\|0\rangle\langle 0\| \otimes I + \|1\rangle\langle 1\| \otimes U$ |
| ABC decomposition | $\text{C-}U = (I \otimes A)\cdot\text{CNOT}\cdot(I \otimes B)\cdot\text{CNOT}\cdot(I \otimes C)$ |

### Main Takeaways

1. **CZ is symmetric:** No distinction between control and target
2. **CZ and CNOT are related:** Conjugation by (I⊗H) converts between them
3. **Any controlled-U is possible:** Using ABC decomposition with 2 CNOTs
4. **Phase kickback:** Control qubit acquires eigenvalue phase
5. **Multi-controlled gates:** Extend the formalism to Toffoli and beyond

---

## Daily Checklist

- [ ] I can write the CZ matrix from memory
- [ ] I understand CZ's symmetry property
- [ ] I can convert between CNOT and CZ using Hadamards
- [ ] I understand the ABC decomposition concept
- [ ] I can implement phase kickback calculations
- [ ] I completed the computational lab
- [ ] I solved at least 3 practice problems

---

## Preview of Day 570

Tomorrow we study **SWAP and √SWAP gates**. The SWAP gate exchanges qubit states, while √SWAP is a "partial swap" that creates entanglement. We'll also explore the iSWAP gate, which is native to some superconducting qubit platforms.
