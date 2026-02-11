# Day 570: SWAP and √SWAP Gates

## Schedule Overview

| Session | Time | Focus |
|---------|------|-------|
| Morning | 3 hours | Theory: SWAP gate, √SWAP, iSWAP definitions |
| Afternoon | 2.5 hours | Problem solving: SWAP decompositions and applications |
| Evening | 1.5 hours | Computational lab: Partial swap entanglement |

## Learning Objectives

By the end of today, you will be able to:

1. **Implement the SWAP gate** and understand qubit exchange
2. **Decompose SWAP** into three CNOTs
3. **Define √SWAP** as a "partial swap" creating entanglement
4. **Understand iSWAP** and its role in superconducting circuits
5. **Relate SWAP family gates** mathematically
6. **Apply SWAP gates** in quantum algorithms

---

## Core Content

### 1. The SWAP Gate

The **SWAP gate** exchanges the states of two qubits:

$$\boxed{\text{SWAP}|a, b\rangle = |b, a\rangle}$$

**Truth table:**
| Input | Output |
|-------|--------|
| $\|00\rangle$ | $\|00\rangle$ |
| $\|01\rangle$ | $\|10\rangle$ |
| $\|10\rangle$ | $\|01\rangle$ |
| $\|11\rangle$ | $\|11\rangle$ |

### 2. SWAP Matrix

$$\boxed{\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}}$$

**Operator form:**
$$\text{SWAP} = |00\rangle\langle 00| + |01\rangle\langle 10| + |10\rangle\langle 01| + |11\rangle\langle 11|$$

Or using Pauli matrices:
$$\text{SWAP} = \frac{1}{2}(I \otimes I + X \otimes X + Y \otimes Y + Z \otimes Z)$$

### 3. Key SWAP Properties

**Self-inverse:**
$$\boxed{\text{SWAP}^2 = I}$$

**Symmetry:**
$$\text{SWAP}^T = \text{SWAP}, \quad \text{SWAP}^\dagger = \text{SWAP}$$

SWAP is both symmetric and Hermitian!

**Determinant:**
$$\det(\text{SWAP}) = -1$$

**Eigenvalues:**
- λ = +1 (multiplicity 3): symmetric states
- λ = -1 (multiplicity 1): antisymmetric state

### 4. SWAP Eigenstates

**Eigenvalue +1:** Symmetric states
- $|00\rangle$
- $|11\rangle$
- $|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$

**Eigenvalue -1:** Antisymmetric state
- $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$

The antisymmetric Bell state is special: SWAP$|\Psi^-\rangle = -|\Psi^-\rangle$.

### 5. SWAP from CNOTs

SWAP can be decomposed into three CNOTs:

$$\boxed{\text{SWAP} = \text{CNOT}_{1\to 2} \cdot \text{CNOT}_{2\to 1} \cdot \text{CNOT}_{1\to 2}}$$

**Circuit:**
```
q0: ───●───────X───●───
       │       │   │
q1: ───X───●───────X───
           │
```

**Proof (by tracking basis states):**
- $|01\rangle \xrightarrow{\text{CNOT}_{1\to 2}} |01\rangle \xrightarrow{\text{CNOT}_{2\to 1}} |11\rangle \xrightarrow{\text{CNOT}_{1\to 2}} |10\rangle$ ✓

### 6. The √SWAP Gate

The **square root of SWAP** is a gate V such that $V^2 = \text{SWAP}$:

$$\boxed{\sqrt{\text{SWAP}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \frac{1+i}{2} & \frac{1-i}{2} & 0 \\ 0 & \frac{1-i}{2} & \frac{1+i}{2} & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}}$$

**Action:**
$$\sqrt{\text{SWAP}}|01\rangle = \frac{1+i}{2}|01\rangle + \frac{1-i}{2}|10\rangle$$

This is a **partial swap** - it creates a superposition of swapped and unswapped states!

### 7. √SWAP Creates Entanglement

Starting from a product state:
$$\sqrt{\text{SWAP}}|01\rangle = \frac{1+i}{2}|01\rangle + \frac{1-i}{2}|10\rangle$$

This is an **entangled state**! The √SWAP gate is an entangling gate.

**Entangling power:** √SWAP is maximally entangling for specific input states.

### 8. √SWAP Properties

**Square:**
$$(\sqrt{\text{SWAP}})^2 = \text{SWAP}$$

**Fourth power:**
$$(\sqrt{\text{SWAP}})^4 = I$$

**Inverse:**
$$(\sqrt{\text{SWAP}})^{-1} = (\sqrt{\text{SWAP}})^3 = (\sqrt{\text{SWAP}})^\dagger$$

### 9. The iSWAP Gate

The **iSWAP** gate swaps states with an additional phase:

$$\boxed{\text{iSWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}}$$

**Action:**
$$\text{iSWAP}|01\rangle = i|10\rangle, \quad \text{iSWAP}|10\rangle = i|01\rangle$$

**Key property:**
$$(\text{iSWAP})^2 = \text{SWAP} \cdot (Z \otimes Z) = -\text{SWAP} \text{ on subspace}$$

Actually: $\text{iSWAP}^2|01\rangle = -|01\rangle$, $\text{iSWAP}^2|10\rangle = -|10\rangle$.

### 10. iSWAP in Hardware

iSWAP is a **native gate** on many superconducting qubit platforms:

**Physical origin:** When two qubits are coupled, the natural evolution under the XY (exchange) interaction gives:
$$e^{-it(XX + YY)/2} = \cos(t)I + i\sin(t)\text{SWAP}$$

At $t = \pi/2$: This gives the iSWAP gate!

### 11. √iSWAP Gate

The square root of iSWAP:

$$\sqrt{\text{iSWAP}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \frac{1}{\sqrt{2}} & \frac{i}{\sqrt{2}} & 0 \\ 0 & \frac{i}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

This is another native gate on Google's Sycamore processor!

### 12. Relationships Between SWAP Gates

$$\text{SWAP} = (\sqrt{\text{SWAP}})^2$$
$$\text{iSWAP} = \text{SWAP} \cdot \text{CZ} = \text{SWAP} \cdot (S \otimes S^\dagger)^{\text{controlled}}$$
$$\sqrt{\text{iSWAP}}^2 = \text{iSWAP}$$

**Interconversion:** All these gates can be converted to each other using single-qubit gates.

---

## Quantum Computing Connection

SWAP gates are essential for:

1. **Qubit routing:** Moving quantum information across a chip with limited connectivity
2. **State preparation:** Reordering qubits for algorithm execution
3. **Measurement:** Bringing qubits to measurable positions
4. **Fermionic simulation:** SWAP captures exchange statistics
5. **Native gates:** iSWAP and √iSWAP are natural in superconducting systems

---

## Worked Examples

### Example 1: Verify SWAP from CNOTs

**Problem:** Show that three CNOTs compose to SWAP.

**Solution:**

Label the CNOTs: $C_{1\to 2}$ (control=1, target=2) and $C_{2\to 1}$ (control=2, target=1).

Track $|01\rangle$:
1. $C_{1\to 2}|01\rangle = |01\rangle$ (control=0, no flip)
2. $C_{2\to 1}|01\rangle = |11\rangle$ (control=1, flip first qubit)
3. $C_{1\to 2}|11\rangle = |10\rangle$ (control=1, flip second)

Result: $|01\rangle \to |10\rangle$ ✓

Track $|10\rangle$:
1. $C_{1\to 2}|10\rangle = |11\rangle$
2. $C_{2\to 1}|11\rangle = |01\rangle$
3. $C_{1\to 2}|01\rangle = |01\rangle$

Result: $|10\rangle \to |01\rangle$ ✓

### Example 2: √SWAP Entanglement

**Problem:** Compute √SWAP$|01\rangle$ and verify it's entangled.

**Solution:**

$$\sqrt{\text{SWAP}}|01\rangle = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \frac{1+i}{2} & \frac{1-i}{2} & 0 \\ 0 & \frac{1-i}{2} & \frac{1+i}{2} & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}\begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ \frac{1+i}{2} \\ \frac{1-i}{2} \\ 0 \end{pmatrix}$$

The state is:
$$|\psi\rangle = \frac{1+i}{2}|01\rangle + \frac{1-i}{2}|10\rangle$$

**Check entanglement (Schmidt decomposition):**
Reshape to 2×2 matrix:
$$\begin{pmatrix} 0 & \frac{1+i}{2} \\ \frac{1-i}{2} & 0 \end{pmatrix}$$

Singular values: $\sigma_1 = \sigma_2 = \frac{1}{\sqrt{2}}$ (two non-zero)

Schmidt rank = 2 → **Entangled!**

### Example 3: iSWAP Action

**Problem:** Compute iSWAP on the Bell state $|\Phi^+\rangle$.

**Solution:**

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$\text{iSWAP}|\Phi^+\rangle = \frac{1}{\sqrt{2}}(\text{iSWAP}|00\rangle + \text{iSWAP}|11\rangle)$$

Since iSWAP$|00\rangle = |00\rangle$ and iSWAP$|11\rangle = |11\rangle$:

$$= \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$$

$|\Phi^+\rangle$ is an eigenstate of iSWAP with eigenvalue 1!

---

## Practice Problems

### Direct Application

1. Verify that SWAP$|\Psi^-\rangle = -|\Psi^-\rangle$ where $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$.

2. Compute $(\sqrt{\text{SWAP}})^2$ by matrix multiplication and verify it equals SWAP.

3. Find iSWAP$|01\rangle$ and iSWAP$|10\rangle$.

### Intermediate

4. **Pauli decomposition:** Verify that $\text{SWAP} = \frac{1}{2}(I \otimes I + X \otimes X + Y \otimes Y + Z \otimes Z)$.

5. Show that SWAP commutes with any gate of the form $U \otimes U$: $[\text{SWAP}, U \otimes U] = 0$.

6. **Entanglement measure:** Compute the concurrence of $\sqrt{\text{SWAP}}|01\rangle$.

### Challenging

7. **iSWAP decomposition:** Express iSWAP using CNOT and single-qubit gates. What is the minimum CNOT count?

8. Find the eigenvalues and eigenvectors of √SWAP.

9. **Fermionic exchange:** Show that for fermionic states (antisymmetric under exchange), SWAP introduces a -1 phase, consistent with the Pauli exclusion principle.

---

## Computational Lab: SWAP Gate Family

```python
"""
Day 570: SWAP and √SWAP Gates
Exploring qubit exchange and partial swap operations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# Define basic gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

I4 = np.eye(4, dtype=complex)

# Define CNOT gates
CNOT_12 = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)  # Control=1, Target=2
CNOT_21 = np.array([[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]], dtype=complex)  # Control=2, Target=1

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

def ket(bitstring):
    state = np.array([[1]], dtype=complex)
    for bit in bitstring:
        state = np.kron(state, ket_0 if bit == '0' else ket_1)
    return state

print("=" * 60)
print("SWAP GATE")
print("=" * 60)

# Define SWAP
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

print("\n1. SWAP Matrix:")
print(SWAP)

# Verify properties
print("\n2. SWAP Properties:")
print(f"   SWAP² = I: {np.allclose(SWAP @ SWAP, I4)}")
print(f"   SWAP† = SWAP: {np.allclose(SWAP.conj().T, SWAP)}")
print(f"   SWAP^T = SWAP: {np.allclose(SWAP.T, SWAP)}")
print(f"   det(SWAP) = {np.linalg.det(SWAP).real:.0f}")

# Action on basis states
print("\n3. SWAP Action on Basis States:")
for bits in ['00', '01', '10', '11']:
    input_state = ket(bits)
    output_state = SWAP @ input_state
    # Find output
    for out_bits in ['00', '01', '10', '11']:
        if np.allclose(output_state, ket(out_bits)):
            print(f"   SWAP|{bits}⟩ = |{out_bits}⟩")
            break

# Eigenvalues and eigenvectors
print("\n4. SWAP Eigenvalues:")
eigenvalues, eigenvectors = np.linalg.eig(SWAP)
for val in np.unique(np.round(eigenvalues.real, 4)):
    count = np.sum(np.isclose(eigenvalues.real, val))
    print(f"   λ = {val:+.0f} (multiplicity {count})")

# Verify SWAP from CNOTs
print("\n5. SWAP from CNOTs:")
SWAP_from_CNOT = CNOT_12 @ CNOT_21 @ CNOT_12
print(f"   CNOT₁₂·CNOT₂₁·CNOT₁₂ = SWAP: {np.allclose(SWAP_from_CNOT, SWAP)}")

# Pauli decomposition
print("\n6. Pauli Decomposition:")
SWAP_pauli = 0.5 * (np.kron(I, I) + np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z))
print(f"   ½(I⊗I + X⊗X + Y⊗Y + Z⊗Z) = SWAP: {np.allclose(SWAP_pauli, SWAP)}")

# √SWAP
print("\n" + "=" * 60)
print("√SWAP GATE")
print("=" * 60)

# Define √SWAP
sqrt_SWAP = np.array([
    [1, 0, 0, 0],
    [0, (1+1j)/2, (1-1j)/2, 0],
    [0, (1-1j)/2, (1+1j)/2, 0],
    [0, 0, 0, 1]
], dtype=complex)

print("\n7. √SWAP Matrix:")
print(sqrt_SWAP)

# Verify (√SWAP)² = SWAP
print("\n8. √SWAP Properties:")
print(f"   (√SWAP)² = SWAP: {np.allclose(sqrt_SWAP @ sqrt_SWAP, SWAP)}")
print(f"   (√SWAP)⁴ = I: {np.allclose(np.linalg.matrix_power(sqrt_SWAP, 4), I4)}")

# Alternative: compute using matrix square root
sqrt_SWAP_computed = sqrtm(SWAP)
print(f"   Matrix sqrtm(SWAP) ≈ √SWAP: {np.allclose(sqrt_SWAP_computed, sqrt_SWAP) or np.allclose(-sqrt_SWAP_computed, sqrt_SWAP)}")

# Entanglement from √SWAP
print("\n9. Entanglement from √SWAP:")

def concurrence(state):
    """Compute concurrence for 2-qubit pure state."""
    state = state.flatten()
    a, b, c, d = state
    return 2 * np.abs(a*d - b*c)

def schmidt_rank(state):
    """Compute Schmidt rank."""
    psi = state.reshape(2, 2)
    _, s, _ = np.linalg.svd(psi)
    return np.sum(s > 1e-10)

test_inputs = [
    ('|00⟩', ket('00')),
    ('|01⟩', ket('01')),
    ('|10⟩', ket('10')),
    ('|11⟩', ket('11')),
]

for name, state in test_inputs:
    output = sqrt_SWAP @ state
    C = concurrence(output)
    print(f"   √SWAP{name}: Concurrence = {C:.4f} ({'entangled' if C > 0.01 else 'separable'})")

# iSWAP
print("\n" + "=" * 60)
print("iSWAP GATE")
print("=" * 60)

# Define iSWAP
iSWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

print("\n10. iSWAP Matrix:")
print(iSWAP)

# Properties
print("\n11. iSWAP Properties:")
print(f"   iSWAP|01⟩ = i|10⟩: {np.allclose(iSWAP @ ket('01'), 1j * ket('10'))}")
print(f"   iSWAP|10⟩ = i|01⟩: {np.allclose(iSWAP @ ket('10'), 1j * ket('01'))}")

# iSWAP²
iSWAP_sq = iSWAP @ iSWAP
print(f"\n12. iSWAP² action:")
print(f"   iSWAP²|01⟩ = -|01⟩: {np.allclose(iSWAP_sq @ ket('01'), -ket('01'))}")
print(f"   iSWAP²|10⟩ = -|10⟩: {np.allclose(iSWAP_sq @ ket('10'), -ket('10'))}")
print(f"   iSWAP²|00⟩ = |00⟩: {np.allclose(iSWAP_sq @ ket('00'), ket('00'))}")

# √iSWAP
print("\n13. √iSWAP Gate:")
sqrt_iSWAP = np.array([
    [1, 0, 0, 0],
    [0, 1/np.sqrt(2), 1j/np.sqrt(2), 0],
    [0, 1j/np.sqrt(2), 1/np.sqrt(2), 0],
    [0, 0, 0, 1]
], dtype=complex)

print(f"   (√iSWAP)² = iSWAP: {np.allclose(sqrt_iSWAP @ sqrt_iSWAP, iSWAP)}")

# Relationship between gates
print("\n" + "=" * 60)
print("GATE RELATIONSHIPS")
print("=" * 60)

# CZ gate
CZ = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]], dtype=complex)

# iSWAP = SWAP · (some phase gate)
print("\n14. iSWAP vs SWAP relationship:")
ratio = iSWAP / SWAP
# The non-trivial part is in the 01, 10 subspace
print(f"   iSWAP[1,2] / SWAP[1,2] = {iSWAP[1,2] / SWAP[1,2]}")
print(f"   iSWAP = SWAP with phase i on swapped elements")

# Visualization
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: SWAP circuit from CNOTs
ax1 = axes[0, 0]
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 6)

ax1.plot([1, 11], [4, 4], 'k-', linewidth=2)
ax1.plot([1, 11], [2, 2], 'k-', linewidth=2)

# Three CNOTs
for x, (ctrl, tgt) in [(2.5, (4, 2)), (5.5, (2, 4)), (8.5, (4, 2))]:
    ax1.plot([x, x], [tgt, ctrl], 'k-', linewidth=2)
    ax1.scatter([x], [ctrl], s=100, c='black', zorder=5)
    circle = plt.Circle((x, tgt), 0.2, fill=False, linewidth=2)
    ax1.add_patch(circle)

ax1.text(0.3, 4, 'q₀', fontsize=12, va='center')
ax1.text(0.3, 2, 'q₁', fontsize=12, va='center')
ax1.set_title('SWAP = CNOT · CNOT · CNOT', fontsize=12)
ax1.axis('off')

# Plot 2: SWAP matrix
ax2 = axes[0, 1]
im = ax2.imshow(np.abs(SWAP), cmap='Blues', vmin=0, vmax=1)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
ax2.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
for i in range(4):
    for j in range(4):
        if np.abs(SWAP[i,j]) > 0.5:
            ax2.text(j, i, '1', ha='center', va='center', fontsize=14)
ax2.set_title('SWAP Matrix', fontsize=12)

# Plot 3: √SWAP matrix (real and imag)
ax3 = axes[0, 2]
sqrt_SWAP_display = np.real(sqrt_SWAP) + np.imag(sqrt_SWAP)  # Simplified visualization
im3 = ax3.imshow(np.abs(sqrt_SWAP), cmap='Purples', vmin=0, vmax=1)
ax3.set_xticks(range(4))
ax3.set_yticks(range(4))
ax3.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
ax3.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
for i in range(4):
    for j in range(4):
        val = sqrt_SWAP[i,j]
        if np.abs(val) > 0.3:
            text = f'{val.real:.2f}' if np.abs(val.imag) < 0.01 else f'{val:.1f}'
            ax3.text(j, i, text, ha='center', va='center', fontsize=9)
ax3.set_title('√SWAP Matrix', fontsize=12)

# Plot 4: iSWAP matrix
ax4 = axes[1, 0]
im4 = ax4.imshow(np.abs(iSWAP), cmap='Greens', vmin=0, vmax=1)
ax4.set_xticks(range(4))
ax4.set_yticks(range(4))
ax4.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
ax4.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
for i in range(4):
    for j in range(4):
        val = iSWAP[i,j]
        if np.abs(val) > 0.5:
            text = '1' if np.isclose(val, 1) else 'i'
            ax4.text(j, i, text, ha='center', va='center', fontsize=14)
ax4.set_title('iSWAP Matrix', fontsize=12)

# Plot 5: Entanglement from √SWAP
ax5 = axes[1, 1]
thetas = np.linspace(0, np.pi, 50)

# Parametrize input as cos(θ)|01⟩ + sin(θ)|10⟩
concurrences = []
for theta in thetas:
    input_state = np.cos(theta) * ket('01') + np.sin(theta) * ket('10')
    output_state = sqrt_SWAP @ input_state
    C = concurrence(output_state)
    concurrences.append(C)

ax5.plot(thetas / np.pi, concurrences, 'b-', linewidth=2)
ax5.set_xlabel('θ/π (input = cos(θ)|01⟩ + sin(θ)|10⟩)', fontsize=10)
ax5.set_ylabel('Concurrence', fontsize=10)
ax5.set_title('Entanglement from √SWAP', fontsize=12)
ax5.grid(True, alpha=0.3)
ax5.set_xlim([0, 1])
ax5.set_ylim([0, 1.1])

# Plot 6: SWAP eigenspace
ax6 = axes[1, 2]
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 6)

# Eigenvalue +1 space
ax6.text(5, 5.5, 'SWAP Eigenspaces', fontsize=14, ha='center', fontweight='bold')

ax6.text(5, 4.5, 'λ = +1 (Symmetric, dim=3):', fontsize=11, ha='center')
ax6.text(5, 3.8, '|00⟩, |11⟩, |Ψ⁺⟩ = (|01⟩+|10⟩)/√2', fontsize=10, ha='center', family='monospace')

ax6.text(5, 2.5, 'λ = -1 (Antisymmetric, dim=1):', fontsize=11, ha='center')
ax6.text(5, 1.8, '|Ψ⁻⟩ = (|01⟩-|10⟩)/√2', fontsize=10, ha='center', family='monospace')

ax6.text(5, 0.8, 'Note: |Ψ⁻⟩ is the singlet state', fontsize=9, ha='center', style='italic')

ax6.axis('off')

plt.tight_layout()
plt.savefig('swap_gates.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: swap_gates.png")

# Additional analysis
print("\n" + "=" * 60)
print("SWAP GATE APPLICATIONS")
print("=" * 60)

print("\n15. SWAP for Qubit Routing:")
print("   On hardware with limited connectivity, SWAP moves quantum")
print("   information between non-adjacent qubits.")
print("   Cost: 3 CNOTs per SWAP")

print("\n16. Bell State Behavior under SWAP:")
bell_states = {
    'Φ⁺': (ket('00') + ket('11')) / np.sqrt(2),
    'Φ⁻': (ket('00') - ket('11')) / np.sqrt(2),
    'Ψ⁺': (ket('01') + ket('10')) / np.sqrt(2),
    'Ψ⁻': (ket('01') - ket('10')) / np.sqrt(2),
}

for name, state in bell_states.items():
    result = SWAP @ state
    # Find eigenvalue
    ratio = result / state
    eigenval = ratio[np.argmax(np.abs(state))]
    print(f"   SWAP|{name}⟩ = {eigenval.real:+.0f}|{name}⟩")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| SWAP action | $\text{SWAP}\|a,b\rangle = \|b,a\rangle$ |
| SWAP matrix | Exchange rows 2 and 3 of $I_4$ |
| SWAP from CNOTs | $\text{CNOT}_{1\to 2} \cdot \text{CNOT}_{2\to 1} \cdot \text{CNOT}_{1\to 2}$ |
| Pauli form | $\frac{1}{2}(I \otimes I + X \otimes X + Y \otimes Y + Z \otimes Z)$ |
| √SWAP | $(\sqrt{\text{SWAP}})^2 = \text{SWAP}$ |
| iSWAP action | $\text{iSWAP}\|01\rangle = i\|10\rangle$ |

### Main Takeaways

1. **SWAP exchanges qubits:** $|a,b\rangle \to |b,a\rangle$
2. **Three CNOTs required:** SWAP decomposes to 3 CNOTs
3. **√SWAP creates entanglement:** Partial swap is an entangling gate
4. **iSWAP is native:** Natural gate in superconducting hardware
5. **Eigenspace structure:** Symmetric states have eigenvalue +1, antisymmetric has -1

---

## Daily Checklist

- [ ] I can write the SWAP matrix from memory
- [ ] I understand SWAP decomposition into CNOTs
- [ ] I can compute √SWAP action and verify entanglement
- [ ] I understand iSWAP and its hardware relevance
- [ ] I know the eigenvalue structure of SWAP
- [ ] I completed the computational lab
- [ ] I solved at least 3 practice problems

---

## Preview of Day 571

Tomorrow we study **entangling power** - how to quantify the ability of two-qubit gates to create entanglement. We'll define measures like concurrence and entangling capacity, and understand which gates are maximally entangling. CNOT, CZ, and √iSWAP all play important roles!
