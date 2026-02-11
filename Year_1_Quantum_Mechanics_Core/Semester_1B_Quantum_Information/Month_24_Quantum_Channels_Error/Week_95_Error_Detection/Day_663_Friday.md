# Day 663: Nine-Qubit Shor Code

## Week 95: Error Detection/Correction Intro | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **Understand** concatenation of error correcting codes
2. **Construct** the 9-qubit Shor code from bit-flip and phase-flip codes
3. **Verify** that Shor code corrects arbitrary single-qubit errors
4. **Analyze** syndrome measurement and correction procedures
5. **Appreciate** the historical significance of this discovery

---

## Core Content

### 1. The Problem with 3-Qubit Codes

**Bit-flip code:** Corrects X errors, fails on Z errors

**Phase-flip code:** Corrects Z errors, fails on X errors

**General single-qubit error:**
$$E = \alpha I + \beta X + \gamma Y + \delta Z$$

Since $Y = iXZ$, we need to correct both X and Z simultaneously!

### 2. The Concatenation Idea

**Shor's brilliant insight (1995):** Nest one code inside another!

**Strategy:**
1. First encode against phase errors (outer code)
2. Then encode each physical qubit against bit errors (inner code)

### 3. Shor Code Construction

**Step 1: Phase-flip encoding**
$$|0\rangle \to |+\rangle|+\rangle|+\rangle = |+++\rangle$$
$$|1\rangle \to |-\rangle|-\rangle|-\rangle = |---\rangle$$

**Step 2: Bit-flip encoding of each qubit**
$$|+\rangle \to |+++\rangle \quad \text{(block of 3)}$$
$$|-\rangle \to |---\rangle \quad \text{(block of 3)}$$

Wait, that's not right. Let me be more careful:

$$|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} \to \frac{|000\rangle + |111\rangle}{\sqrt{2}}$$
$$|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}} \to \frac{|000\rangle - |111\rangle}{\sqrt{2}}$$

### 4. Full Shor Code States

$$\boxed{|0_L\rangle = \frac{1}{2\sqrt{2}}\left(|000\rangle + |111\rangle\right)\left(|000\rangle + |111\rangle\right)\left(|000\rangle + |111\rangle\right)}$$

$$\boxed{|1_L\rangle = \frac{1}{2\sqrt{2}}\left(|000\rangle - |111\rangle\right)\left(|000\rangle - |111\rangle\right)\left(|000\rangle - |111\rangle\right)}$$

**Structure:** 3 blocks of 3 qubits = 9 qubits total

**Notation:** Qubits labeled 1-9, blocks are (1,2,3), (4,5,6), (7,8,9)

### 5. Encoding Circuit

```
|ψ⟩ ───●───●───H───●───────●─────────────────────── (qubit 1)
       │   │       │       │
|0⟩ ───⊕───┼───────┼───────┼───●───●─────────────── (qubit 2)
           │       │       │   │   │
|0⟩ ───────⊕───────┼───────┼───⊕───┼───●───●─────── (qubit 3)
                   │       │       │   │   │
|0⟩ ───────────────⊕───H───┼───────┼───⊕───┼─────── (qubit 4)
                           │       │       │
|0⟩ ─────────────────────────────────────⊕─────────... (qubits 5-9)
```

(Full circuit continues similarly for all 9 qubits)

**Simplified encoding:**
1. Apply CNOT from qubit 1 to qubits 4 and 7
2. Apply Hadamard to qubits 1, 4, 7
3. Apply CNOT within each block: 1→2,3 and 4→5,6 and 7→8,9

### 6. Bit-Flip Syndrome Operators

**Within each block, measure parity:**

Block 1: $Z_1Z_2$, $Z_2Z_3$
Block 2: $Z_4Z_5$, $Z_5Z_6$
Block 3: $Z_7Z_8$, $Z_8Z_9$

**6 syndrome bits** for bit-flip detection (2 per block)

### 7. Phase-Flip Syndrome Operators

**Between blocks, measure X-parity:**

$$X_1X_2X_3 \cdot X_4X_5X_6 = X^{\otimes 6} \otimes I^{\otimes 3}$$
$$X_4X_5X_6 \cdot X_7X_8X_9 = I^{\otimes 3} \otimes X^{\otimes 6}$$

Actually, the correct operators are:
$$S_7 = X_1X_2X_3X_4X_5X_6$$
$$S_8 = X_4X_5X_6X_7X_8X_9$$

**2 syndrome bits** for phase-flip detection between blocks

### 8. Complete Syndrome Table

**Bit-flip syndromes (6 bits):** Identify which qubit in which block

| Block | $Z_iZ_{i+1}$ | $Z_{i+1}Z_{i+2}$ | Error |
|-------|-------------|-----------------|-------|
| 1 | +1, +1 | - | None in block 1 |
| 1 | -1, +1 | - | $X_1$ |
| 1 | -1, -1 | - | $X_2$ |
| 1 | +1, -1 | - | $X_3$ |

(Similar for blocks 2 and 3)

**Phase-flip syndromes (2 bits):** Identify which block

| $S_7$ | $S_8$ | Error location |
|-------|-------|----------------|
| +1 | +1 | None |
| -1 | +1 | Block 1 |
| -1 | -1 | Block 2 |
| +1 | -1 | Block 3 |

### 9. Error Correction Procedure

**For X (bit-flip) error on qubit $j$ in block $k$:**
1. Bit-flip syndromes identify qubit $j$
2. Apply $X_j$

**For Z (phase-flip) error on qubit $j$ in block $k$:**
1. Phase-flip syndromes identify block $k$
2. Any Z error in block affects the whole block identically
3. Apply $Z$ to any qubit in block $k$ (usually qubit $3k-2$)

**For Y error:** Since $Y = iXZ$, correct both X and Z components!

### 10. Why It Works: Knill-Laflamme Verification

**Claim:** For any single-qubit error $E \in \{I, X_j, Y_j, Z_j\}_{j=1}^{9}$:
$$\langle 0_L|E_a^\dagger E_b|0_L\rangle = \langle 1_L|E_a^\dagger E_b|1_L\rangle$$
$$\langle 0_L|E_a^\dagger E_b|1_L\rangle = 0$$

**Proof sketch:**
- Different single-qubit errors produce orthogonal states (distinguishable syndromes)
- Same error on $|0_L\rangle$ and $|1_L\rangle$ has same inner product structure
- Cross-terms vanish due to orthogonality of error subspaces

### 11. Discretization of Errors

The Shor code corrects **any** single-qubit error, not just Pauli errors!

**Why?** Any error can be written as:
$$E = \alpha I + \beta X + \gamma Y + \delta Z$$

Syndrome measurement **projects** the error onto one of {I, X, Y, Z}.

**After measurement:**
- With probability $|\alpha|^2$: no error occurred
- With probability $|\beta|^2$: X error occurred (correct with X)
- With probability $|\gamma|^2$: Y error occurred (correct with Y)
- With probability $|\delta|^2$: Z error occurred (correct with Z)

### 12. Code Parameters

**Shor code [[9, 1, 3]]:**
- **n = 9:** Physical qubits
- **k = 1:** Logical qubits
- **d = 3:** Code distance (minimum weight of logical operator)

**Can correct:** $t = \lfloor(d-1)/2\rfloor = 1$ error

**Rate:** $R = k/n = 1/9 \approx 0.11$

---

## Worked Example

**Problem:** A Shor-encoded state experiences error $Y_5$. Describe the correction.

**Solution:**

1. **Identify error type:** $Y_5 = iX_5Z_5$ is on qubit 5 (block 2, middle qubit)

2. **Bit-flip syndrome (X component):**
   - $X_5$ flips qubit 5 in block 2
   - $Z_4Z_5$: eigenvalue changes sign → syndrome bit = 1
   - $Z_5Z_6$: eigenvalue changes sign → syndrome bit = 1
   - Block 2 syndrome: (1, 1) → error on qubit 5

3. **Phase-flip syndrome (Z component):**
   - $Z_5$ applies phase to block 2
   - $S_7 = X_1X_2X_3X_4X_5X_6$: eigenvalue flips (block 2 affected)
   - $S_8 = X_4X_5X_6X_7X_8X_9$: eigenvalue flips (block 2 affected)
   - Phase syndrome: (1, 1) → error in block 2

4. **Combined diagnosis:** Both syndromes point to qubit 5, error consistent with Y

5. **Correction:** Apply $Y_5$ (or equivalently $X_5$ then $Z_5$)

6. **Result:** State recovered (up to global phase)

---

## Practice Problems

1. Show that $|0_L\rangle$ and $|1_L\rangle$ are orthonormal.

2. Verify that $Z_1$ and $Z_2$ have the same effect on the encoded state (both are phase errors in block 1).

3. Calculate the syndrome for error $X_1Z_7$ (two-qubit error). Can it be corrected?

4. Prove that the code distance is 3 by finding the minimum weight logical operator.

5. How many syndrome bits are needed total? Explain the counting.

---

## Computational Lab

```python
"""Day 663: Nine-Qubit Shor Code"""

import numpy as np
from itertools import product

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def tensor(*matrices):
    """Compute tensor product of matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# GHZ-like states for each block
ghz_plus = (tensor(ket_0, ket_0, ket_0) + tensor(ket_1, ket_1, ket_1)) / np.sqrt(2)
ghz_minus = (tensor(ket_0, ket_0, ket_0) - tensor(ket_1, ket_1, ket_1)) / np.sqrt(2)

# Shor code logical states (9 qubits = 512-dimensional)
ket_0L = tensor(ghz_plus, ghz_plus, ghz_plus)
ket_1L = tensor(ghz_minus, ghz_minus, ghz_minus)

print("Nine-Qubit Shor Code")
print("=" * 60)
print(f"Hilbert space dimension: {len(ket_0L)}")
print(f"|0_L⟩ norm: {np.linalg.norm(ket_0L):.6f}")
print(f"|1_L⟩ norm: {np.linalg.norm(ket_1L):.6f}")
print(f"⟨0_L|1_L⟩: {np.abs(ket_0L.conj().T @ ket_1L)[0,0]:.6f}")

# Build error operators for 9 qubits
def make_error(pauli, qubit, n_qubits=9):
    """Create single-qubit Pauli error on specified qubit."""
    ops = [I] * n_qubits
    ops[qubit] = pauli
    return tensor(*ops)

# Identity on 9 qubits
I9 = tensor(*[I]*9)

# Bit-flip syndrome operators (Z_i Z_{i+1} for each block)
Z1Z2 = make_error(Z, 0) @ make_error(Z, 1)
Z2Z3 = make_error(Z, 1) @ make_error(Z, 2)
Z4Z5 = make_error(Z, 3) @ make_error(Z, 4)
Z5Z6 = make_error(Z, 4) @ make_error(Z, 5)
Z7Z8 = make_error(Z, 6) @ make_error(Z, 7)
Z8Z9 = make_error(Z, 7) @ make_error(Z, 8)

# Phase-flip syndrome operators (X products between blocks)
X123 = make_error(X, 0) @ make_error(X, 1) @ make_error(X, 2)
X456 = make_error(X, 3) @ make_error(X, 4) @ make_error(X, 5)
X789 = make_error(X, 6) @ make_error(X, 7) @ make_error(X, 8)

S7 = X123 @ X456  # Parity between blocks 1 and 2
S8 = X456 @ X789  # Parity between blocks 2 and 3

bit_flip_syndromes = [Z1Z2, Z2Z3, Z4Z5, Z5Z6, Z7Z8, Z8Z9]
phase_flip_syndromes = [S7, S8]

def get_full_syndrome(state):
    """Get all 8 syndrome bits."""
    syndrome = []
    for S in bit_flip_syndromes:
        val = np.real(state.conj().T @ S @ state)[0, 0]
        syndrome.append(0 if val > 0 else 1)
    for S in phase_flip_syndromes:
        val = np.real(state.conj().T @ S @ state)[0, 0]
        syndrome.append(0 if val > 0 else 1)
    return tuple(syndrome)

def decode_syndrome(syndrome):
    """Decode syndrome to identify error."""
    bf = syndrome[:6]  # Bit-flip part
    pf = syndrome[6:]  # Phase-flip part

    # Identify bit-flip error qubit
    bf_qubit = None
    for block in range(3):
        s1, s2 = bf[2*block], bf[2*block + 1]
        if (s1, s2) == (1, 0):
            bf_qubit = 3*block  # First qubit in block
        elif (s1, s2) == (1, 1):
            bf_qubit = 3*block + 1  # Middle qubit
        elif (s1, s2) == (0, 1):
            bf_qubit = 3*block + 2  # Last qubit

    # Identify phase-flip error block
    pf_block = None
    if pf == (1, 0):
        pf_block = 0
    elif pf == (1, 1):
        pf_block = 1
    elif pf == (0, 1):
        pf_block = 2

    return bf_qubit, pf_block

# Test single-qubit errors
print("\n" + "=" * 60)
print("Single-Qubit Error Detection")
print("=" * 60)

psi_L = (ket_0L + ket_1L) / np.sqrt(2)

errors_to_test = [
    ("I (no error)", I9),
    ("X_1", make_error(X, 0)),
    ("X_5", make_error(X, 4)),
    ("X_9", make_error(X, 8)),
    ("Z_1", make_error(Z, 0)),
    ("Z_5", make_error(Z, 4)),
    ("Y_3", make_error(Y, 2)),
]

for name, E in errors_to_test:
    psi_error = E @ psi_L
    syndrome = get_full_syndrome(psi_error)
    bf_q, pf_b = decode_syndrome(syndrome)
    print(f"{name:12s}: Syndrome {syndrome}, BF qubit: {bf_q}, PF block: {pf_b}")

# Full error correction demonstration
print("\n" + "=" * 60)
print("Full Error Correction")
print("=" * 60)

def correct_shor(state, syndrome):
    """Apply correction based on syndrome."""
    bf_qubit, pf_block = decode_syndrome(syndrome)

    corrected = state.copy()

    # Correct bit-flip
    if bf_qubit is not None:
        corrected = make_error(X, bf_qubit) @ corrected

    # Correct phase-flip (apply Z to first qubit of affected block)
    if pf_block is not None:
        corrected = make_error(Z, 3*pf_block) @ corrected

    return corrected

# Test correction for all single-qubit Pauli errors
paulis = [I, X, Y, Z]
pauli_names = ['I', 'X', 'Y', 'Z']
success_count = 0
total_count = 0

for q in range(9):
    for p_idx, P in enumerate(paulis):
        if p_idx == 0 and q > 0:  # Only test I once
            continue

        E = make_error(P, q) if p_idx > 0 else I9
        psi_error = E @ psi_L

        syndrome = get_full_syndrome(psi_error)
        psi_corrected = correct_shor(psi_error, syndrome)

        fidelity = np.abs(psi_L.conj().T @ psi_corrected)[0, 0]**2

        if fidelity > 0.99:
            success_count += 1
        total_count += 1

print(f"Correction success: {success_count}/{total_count} single-qubit errors")

# Test a two-qubit error (should fail)
print("\n--- Two-Qubit Error Test ---")
E_two = make_error(X, 0) @ make_error(X, 3)  # X1 X4
psi_two_error = E_two @ psi_L
syndrome_two = get_full_syndrome(psi_two_error)
psi_two_corrected = correct_shor(psi_two_error, syndrome_two)
fidelity_two = np.abs(psi_L.conj().T @ psi_two_corrected)[0, 0]**2
print(f"X1 X4 error: Fidelity after correction: {fidelity_two:.4f} (should be < 1)")
```

---

## Summary

| Property | Shor [[9, 1, 3]] Code |
|----------|----------------------|
| Physical qubits | 9 |
| Logical qubits | 1 |
| Code distance | 3 |
| Corrects | Any single-qubit error |
| Syndrome bits | 8 (6 bit-flip + 2 phase-flip) |
| Rate | 1/9 ≈ 11% |

**Key Insights:**
- **Concatenation**: Nest phase-flip code inside bit-flip code
- **Discretization**: Syndrome measurement projects continuous errors to Pauli basis
- **Universality**: Corrects X, Y, Z, and any linear combination
- **Historical**: First code to correct arbitrary single-qubit errors (Shor, 1995)

---

## Preview: Day 664

Tomorrow: **Stabilizer Formalism Preview** - a systematic framework for constructing and analyzing quantum error correcting codes!
