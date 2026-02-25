# Day 690: Shor Code Deep Analysis

## Week 99: Three-Qubit Codes | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Shor Code Structure |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Error Correction Protocol |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational Lab |

---

## Learning Objectives

By the end of Day 690, you will be able to:

1. **Write all 8 Shor code stabilizers** and understand their structure
2. **Construct the complete syndrome table** for single-qubit errors
3. **Implement the full error correction protocol**
4. **Understand degeneracy** in the Shor code
5. **Analyze code efficiency** compared to other codes
6. **Appreciate the historical significance** of the Shor code

---

## Shor Code Structure Review

### The Construction

**Step 1: Phase-flip protection**
$$|0\rangle \to |0_L^{(1)}\rangle = |{+}{+}{+}\rangle$$
$$|1\rangle \to |1_L^{(1)}\rangle = |{-}{-}{-}\rangle$$

**Step 2: Bit-flip protection** (for each $|+\rangle$ and $|-\rangle$)
$$|+\rangle \to \frac{|000\rangle + |111\rangle}{\sqrt{2}}$$
$$|-\rangle \to \frac{|000\rangle - |111\rangle}{\sqrt{2}}$$

### Final Logical States

$$\boxed{|0_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle + |111\rangle)(|000\rangle + |111\rangle)(|000\rangle + |111\rangle)}$$

$$\boxed{|1_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle - |111\rangle)(|000\rangle - |111\rangle)(|000\rangle - |111\rangle)}$$

### Qubit Organization

```
Block 1: q₁ q₂ q₃     Block 2: q₄ q₅ q₆     Block 3: q₇ q₈ q₉
   ├─────────┤           ├─────────┤           ├─────────┤
   Inner code            Inner code            Inner code
   (bit-flip)            (bit-flip)            (bit-flip)
         └───────────────────┼───────────────────┘
                        Outer code
                       (phase-flip)
```

---

## The Eight Stabilizer Generators

### Complete List

| Generator | Formula | Type | Detects |
|-----------|---------|------|---------|
| $g_1$ | $Z_1Z_2$ | Z-type | X errors in block 1 |
| $g_2$ | $Z_2Z_3$ | Z-type | X errors in block 1 |
| $g_3$ | $Z_4Z_5$ | Z-type | X errors in block 2 |
| $g_4$ | $Z_5Z_6$ | Z-type | X errors in block 2 |
| $g_5$ | $Z_7Z_8$ | Z-type | X errors in block 3 |
| $g_6$ | $Z_8Z_9$ | Z-type | X errors in block 3 |
| $g_7$ | $X_1X_2X_3X_4X_5X_6$ | X-type | Z errors blocks 1-2 |
| $g_8$ | $X_4X_5X_6X_7X_8X_9$ | X-type | Z errors blocks 2-3 |

### Stabilizer Verification

**Check $g_i|0_L\rangle = |0_L\rangle$:**

For $g_1 = Z_1Z_2$:
$$Z_1Z_2 \cdot \frac{|000\rangle + |111\rangle}{\sqrt{2}} = \frac{(+1)(+1)|000\rangle + (-1)(-1)|111\rangle}{\sqrt{2}} = \frac{|000\rangle + |111\rangle}{\sqrt{2}} \checkmark$$

For $g_7 = X_1X_2X_3X_4X_5X_6$:

Acting on block 1: $X^{\otimes 3}(|000\rangle + |111\rangle) = |111\rangle + |000\rangle$ ✓ (same state)
Acting on block 2: similarly preserves
Block 3 unchanged

All eigenvalues are +1.

---

## Complete Syndrome Table

### Single X Errors

| Error | $g_1$ | $g_2$ | $g_3$ | $g_4$ | $g_5$ | $g_6$ | $g_7$ | $g_8$ | Syndrome |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|----------|
| $I$ | + | + | + | + | + | + | + | + | 00000000 |
| $X_1$ | − | + | + | + | + | + | + | + | 10000000 |
| $X_2$ | − | − | + | + | + | + | + | + | 11000000 |
| $X_3$ | + | − | + | + | + | + | + | + | 01000000 |
| $X_4$ | + | + | − | + | + | + | + | + | 00100000 |
| $X_5$ | + | + | − | − | + | + | + | + | 00110000 |
| $X_6$ | + | + | + | − | + | + | + | + | 00010000 |
| $X_7$ | + | + | + | + | − | + | + | + | 00001000 |
| $X_8$ | + | + | + | + | − | − | + | + | 00001100 |
| $X_9$ | + | + | + | + | + | − | + | + | 00000100 |

### Single Z Errors

| Error | $g_1$ | $g_2$ | $g_3$ | $g_4$ | $g_5$ | $g_6$ | $g_7$ | $g_8$ | Syndrome |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|----------|
| $Z_1$ | + | + | + | + | + | + | − | + | 00000010 |
| $Z_2$ | + | + | + | + | + | + | − | + | 00000010 |
| $Z_3$ | + | + | + | + | + | + | − | + | 00000010 |
| $Z_4$ | + | + | + | + | + | + | − | − | 00000011 |
| $Z_5$ | + | + | + | + | + | + | − | − | 00000011 |
| $Z_6$ | + | + | + | + | + | + | − | − | 00000011 |
| $Z_7$ | + | + | + | + | + | + | + | − | 00000001 |
| $Z_8$ | + | + | + | + | + | + | + | − | 00000001 |
| $Z_9$ | + | + | + | + | + | + | + | − | 00000001 |

**Key Observation:** All Z errors within a block have the **same syndrome**!

This is **degeneracy** — $Z_1, Z_2, Z_3$ are indistinguishable but equivalent.

### Single Y Errors

Y = iXZ, so Y errors trigger both X-type and Z-type syndromes:

| Error | Syndrome (combines X and Z) |
|-------|----------------------------|
| $Y_1$ | 10000010 |
| $Y_2$ | 11000010 |
| $Y_3$ | 01000010 |
| $Y_4$ | 00100011 |
| ... | ... |

---

## Error Correction Protocol

### Step 1: Syndrome Measurement

Measure all 8 stabilizer generators using ancilla qubits.

**Circuit for $Z_1Z_2$ measurement:**
```
|0⟩ ──H──●──●──H──M── syndrome bit
         │  │
 q₁ ─────Z──┼─────────
            │
 q₂ ────────Z─────────
```

### Step 2: Syndrome Lookup

```
syndrome[0:6] → X error location (if any)
syndrome[6:8] → Z error block (if any)
```

### Step 3: Apply Correction

**For X errors:** Direct lookup from syndrome bits 0-5
- 10,00,00 → $X_1$
- 11,00,00 → $X_2$
- 01,00,00 → $X_3$
- 00,10,00 → $X_4$
- etc.

**For Z errors:** Any Z in the identified block works!
- 00,00,00,10 → Apply $Z_1$ (or $Z_2$ or $Z_3$)
- 00,00,00,11 → Apply $Z_4$ (or $Z_5$ or $Z_6$)
- 00,00,00,01 → Apply $Z_7$ (or $Z_8$ or $Z_9$)

### Step 4: State Recovery

After correction, the state is back in the code space.

---

## Degeneracy Explained

### Why Z Errors Are Degenerate

Within block 1, consider $Z_1$ and $Z_2$:

$$Z_1^\dagger Z_2 = Z_1 Z_2$$

Is $Z_1Z_2$ a stabilizer? **Yes!** It's $g_1$.

Therefore: $Z_1|\psi_L\rangle = Z_2|\psi_L\rangle \cdot (\text{trivial stabilizer action})$

**Physical meaning:** Applying $Z_1$ or $Z_2$ has the same effect on any codeword.

### Degeneracy Degree

For Z errors:
- 3 errors per block, 3 blocks
- But only 3 distinct syndromes (one per block)
- Degeneracy factor: 3 (within each block)

For X errors:
- Each X error has a unique syndrome
- Non-degenerate for X errors

---

## Code Efficiency Analysis

### Comparison Table

| Code | n | k | d | Rate k/n | Qubits per logical |
|------|---|---|---|----------|-------------------|
| Shor [[9,1,3]] | 9 | 1 | 3 | 11.1% | 9 |
| Steane [[7,1,3]] | 7 | 1 | 3 | 14.3% | 7 |
| [[5,1,3]] | 5 | 1 | 3 | 20% | 5 |
| Surface code (d=3) | 17 | 1 | 3 | 5.9% | 17 |

### Shor Code Pros and Cons

**Advantages:**
- Simple conceptual structure (concatenation)
- Easy to understand pedagogically
- Clear separation of X and Z protection

**Disadvantages:**
- Low rate (11.1%)
- No natural transversal T gate
- Not optimal for physical implementation

---

## Logical Operators

### Logical X

$$\bar{X} = X_1X_2X_3X_4X_5X_6X_7X_8X_9 = X^{\otimes 9}$$

**Minimum weight representative:** $\bar{X} = X_1X_4X_7$ (weight 3)

Any X on one qubit per block acts as logical X.

### Logical Z

$$\bar{Z} = Z_1Z_2Z_3Z_4Z_5Z_6Z_7Z_8Z_9 = Z^{\otimes 9}$$

**Minimum weight representative:** $\bar{Z} = Z_1Z_2Z_3$ (weight 3)

Any Z on an entire block acts as logical Z.

### Verification

$[\bar{X}, \bar{Z}]$: X and Z on same qubits anticommute, but there are 9 positions.

$\bar{X}\bar{Z} = (-1)^9 \bar{Z}\bar{X} = -\bar{Z}\bar{X}$

They anticommute ✓

---

## Computational Lab

```python
"""
Day 690 Computational Lab: Shor Code Deep Analysis
=================================================
"""

import numpy as np
from itertools import product

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

def tensor(*args):
    result = args[0]
    for a in args[1:]:
        result = np.kron(result, a)
    return result

print("=" * 70)
print("SHOR [[9,1,3]] CODE COMPLETE ANALYSIS")
print("=" * 70)

# =============================================================================
# Part 1: Build Logical States
# =============================================================================

print("\n--- Part 1: Logical States ---")

# |+⟩ encoded: (|000⟩ + |111⟩)/√2
ket_000 = tensor(np.array([1,0]), np.array([1,0]), np.array([1,0]))
ket_111 = tensor(np.array([0,1]), np.array([0,1]), np.array([0,1]))
plus_enc = (ket_000 + ket_111) / np.sqrt(2)
minus_enc = (ket_000 - ket_111) / np.sqrt(2)

# Full 9-qubit logical states
ket_0L = tensor(plus_enc, plus_enc, plus_enc)
ket_1L = tensor(minus_enc, minus_enc, minus_enc)

print(f"|0_L⟩ has {np.sum(np.abs(ket_0L) > 1e-10)} non-zero amplitudes")
print(f"|1_L⟩ has {np.sum(np.abs(ket_1L) > 1e-10)} non-zero amplitudes")
print(f"⟨0_L|1_L⟩ = {np.abs(np.vdot(ket_0L, ket_1L)):.10f}")

# =============================================================================
# Part 2: Build All 8 Stabilizers
# =============================================================================

print("\n--- Part 2: Stabilizer Generators ---")

def pauli_on_qubit(P, qubit, n=9):
    """Apply Pauli P to specified qubit in n-qubit system."""
    ops = [I] * n
    ops[qubit] = P
    return tensor(*ops)

def multi_pauli(paulis_qubits, n=9):
    """Apply multiple Paulis. paulis_qubits = [(P, q), ...]"""
    ops = [I] * n
    for P, q in paulis_qubits:
        ops[q] = P
    return tensor(*ops)

# ZZ stabilizers (within blocks)
g1 = multi_pauli([(Z,0), (Z,1)])  # Z1Z2
g2 = multi_pauli([(Z,1), (Z,2)])  # Z2Z3
g3 = multi_pauli([(Z,3), (Z,4)])  # Z4Z5
g4 = multi_pauli([(Z,4), (Z,5)])  # Z5Z6
g5 = multi_pauli([(Z,6), (Z,7)])  # Z7Z8
g6 = multi_pauli([(Z,7), (Z,8)])  # Z8Z9

# XXXXXX stabilizers (across blocks)
g7 = multi_pauli([(X,i) for i in range(6)])    # X1..X6
g8 = multi_pauli([(X,i) for i in range(3,9)])  # X4..X9

stabilizers = [g1, g2, g3, g4, g5, g6, g7, g8]
stab_names = ['Z₁Z₂', 'Z₂Z₃', 'Z₄Z₅', 'Z₅Z₆', 'Z₇Z₈', 'Z₈Z₉', 'X₁₂₃₄₅₆', 'X₄₅₆₇₈₉']

print("\nStabilizer eigenvalues on |0_L⟩:")
for name, g in zip(stab_names, stabilizers):
    eigenval = np.real(ket_0L.conj() @ g @ ket_0L)
    print(f"  {name}: {eigenval:+.0f}")

# =============================================================================
# Part 3: Complete Syndrome Table
# =============================================================================

print("\n--- Part 3: Syndrome Table ---")

def get_syndrome(state, stabilizers):
    """Get syndrome as tuple of 0/1."""
    syndrome = []
    for g in stabilizers:
        eigenval = np.real(state.conj() @ g @ state)
        syndrome.append(0 if eigenval > 0 else 1)
    return tuple(syndrome)

def syndrome_to_str(syn):
    return ''.join(str(s) for s in syn)

# Single X errors
print("\nSingle X Errors:")
print("Error   Syndrome    Correction")
print("-" * 35)
for q in range(9):
    E = pauli_on_qubit(X, q)
    state = E @ ket_0L
    syn = get_syndrome(state, stabilizers)
    print(f"X_{q+1}     {syndrome_to_str(syn)}    X_{q+1}")

# Single Z errors
print("\nSingle Z Errors (note degeneracy!):")
print("Error   Syndrome    Block")
print("-" * 35)
for q in range(9):
    E = pauli_on_qubit(Z, q)
    state = E @ ket_0L
    syn = get_syndrome(state, stabilizers)
    block = q // 3 + 1
    print(f"Z_{q+1}     {syndrome_to_str(syn)}    Block {block}")

# =============================================================================
# Part 4: Degeneracy Verification
# =============================================================================

print("\n--- Part 4: Degeneracy Verification ---")

# Check Z1, Z2, Z3 equivalence
Z1 = pauli_on_qubit(Z, 0)
Z2 = pauli_on_qubit(Z, 1)
Z3 = pauli_on_qubit(Z, 2)

# Z1|0_L⟩ should equal Z2|0_L⟩ up to stabilizer
state_Z1 = Z1 @ ket_0L
state_Z2 = Z2 @ ket_0L
state_Z3 = Z3 @ ket_0L

# Check if Z1Z2 is stabilizer
Z1Z2 = Z1 @ Z2
eigenval = np.real(ket_0L.conj() @ Z1Z2 @ ket_0L)
print(f"Z₁Z₂ eigenvalue on |0_L⟩: {eigenval:+.0f} → {'Stabilizer!' if eigenval == 1 else 'Not stabilizer'}")

# They should be equal
print(f"||Z₁|0_L⟩ - Z₂|0_L⟩|| = {np.linalg.norm(state_Z1 - state_Z2):.10f}")
print("Z₁, Z₂, Z₃ all have identical effect on code space!")

# =============================================================================
# Part 5: Error Correction Simulation
# =============================================================================

print("\n--- Part 5: Error Correction Simulation ---")

def correct_error(state, stabilizers):
    """Apply error correction based on syndrome."""
    syn = get_syndrome(state, stabilizers)

    # X error lookup (first 6 bits)
    x_syn = syn[:6]
    x_corrections = {
        (1,0,0,0,0,0): 0, (1,1,0,0,0,0): 1, (0,1,0,0,0,0): 2,
        (0,0,1,0,0,0): 3, (0,0,1,1,0,0): 4, (0,0,0,1,0,0): 5,
        (0,0,0,0,1,0): 6, (0,0,0,0,1,1): 7, (0,0,0,0,0,1): 8,
    }

    # Z error lookup (last 2 bits)
    z_syn = syn[6:]
    z_corrections = {
        (1,0): 0,  # Block 1: any of qubits 0,1,2
        (1,1): 3,  # Block 2: any of qubits 3,4,5
        (0,1): 6,  # Block 3: any of qubits 6,7,8
    }

    corrected = state.copy()

    # Apply X correction
    if x_syn in x_corrections:
        q = x_corrections[x_syn]
        corrected = pauli_on_qubit(X, q) @ corrected

    # Apply Z correction
    if z_syn in z_corrections:
        q = z_corrections[z_syn]
        corrected = pauli_on_qubit(Z, q) @ corrected

    return corrected

# Test with random encoded state
alpha = 1/np.sqrt(3)
beta = np.sqrt(2/3) * np.exp(1j * np.pi/4)
psi_L = alpha * ket_0L + beta * ket_1L
psi_L /= np.linalg.norm(psi_L)

print(f"\nTest state: |ψ_L⟩ = {alpha:.3f}|0_L⟩ + ({beta:.3f})|1_L⟩")

# Apply various errors and correct
test_errors = ['I', 'X1', 'X5', 'X9', 'Z1', 'Z5', 'Z9', 'Y3', 'Y7']
print(f"\n{'Error':<8} {'Syndrome':<12} {'Fidelity':>10}")
print("-" * 32)

for err_name in test_errors:
    if err_name == 'I':
        E = np.eye(512)
    elif err_name[0] == 'X':
        E = pauli_on_qubit(X, int(err_name[1])-1)
    elif err_name[0] == 'Z':
        E = pauli_on_qubit(Z, int(err_name[1])-1)
    elif err_name[0] == 'Y':
        E = pauli_on_qubit(Y, int(err_name[1])-1)

    error_state = E @ psi_L
    syn = syndrome_to_str(get_syndrome(error_state, stabilizers))
    corrected = correct_error(error_state, stabilizers)
    fidelity = np.abs(np.vdot(psi_L, corrected))**2
    print(f"{err_name:<8} {syn:<12} {fidelity:>10.6f}")

# =============================================================================
# Part 6: Two-Error Test
# =============================================================================

print("\n--- Part 6: Two-Qubit Error (Beyond Correction) ---")

X1X2 = pauli_on_qubit(X, 0) @ pauli_on_qubit(X, 1)
error_state = X1X2 @ psi_L
syn = syndrome_to_str(get_syndrome(error_state, stabilizers))
corrected = correct_error(error_state, stabilizers)
fidelity = np.abs(np.vdot(psi_L, corrected))**2

print(f"X₁X₂ error: syndrome {syn}, fidelity after 'correction' = {fidelity:.6f}")
print("Note: Two-qubit errors cannot be correctly identified!")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SHOR CODE SUMMARY")
print("=" * 70)

summary = """
┌───────────────────────────────────────────────────────────────────┐
│                    SHOR [[9,1,3]] CODE                             │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ STRUCTURE:                                                         │
│   • 9 physical qubits in 3 blocks of 3                            │
│   • Inner code (bit-flip) protects each block                     │
│   • Outer code (phase-flip) protects across blocks                │
│                                                                    │
│ STABILIZERS (8 generators):                                        │
│   • ZZ type: Z₁Z₂, Z₂Z₃, Z₄Z₅, Z₅Z₆, Z₇Z₈, Z₈Z₉ (detect X)      │
│   • XX type: X₁₂₃₄₅₆, X₄₅₆₇₈₉ (detect Z across blocks)          │
│                                                                    │
│ DEGENERACY:                                                        │
│   • Z errors within same block are equivalent                      │
│   • Z₁ ≡ Z₂ ≡ Z₃ (all correct to same logical state)             │
│   • X errors are non-degenerate                                    │
│                                                                    │
│ LOGICAL OPERATORS:                                                 │
│   • X̄ = X₁X₄X₇ (min weight 3)                                     │
│   • Z̄ = Z₁Z₂Z₃ (min weight 3)                                     │
│                                                                    │
│ DISTANCE: d = 3                                                    │
│   • Corrects any single-qubit error                               │
│   • First code to correct all Pauli errors!                       │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
"""
print(summary)

print("✅ Day 690 Lab Complete!")
```

---

## Summary

### Shor Code Parameters

| Parameter | Value |
|-----------|-------|
| Physical qubits (n) | 9 |
| Logical qubits (k) | 1 |
| Distance (d) | 3 |
| Stabilizer generators | 8 |
| Rate | 11.1% |

### Stabilizers

| Type | Generators | Function |
|------|------------|----------|
| ZZ (6 total) | $Z_iZ_{i+1}$ within blocks | Detect X errors |
| XX (2 total) | $X_{1-6}$, $X_{4-9}$ | Detect Z errors across blocks |

### Main Takeaways

1. **Concatenation** of bit-flip and phase-flip codes
2. **8 stabilizers** give unique syndromes for single-qubit errors
3. **Degenerate for Z errors** — block location matters, not qubit
4. **First complete quantum code** (protects X, Y, Z)
5. **Logical operators** have weight 3 (determines distance)

---

## Preview: Day 691

Tomorrow: **Steane [[7,1,3]] Code**
- More efficient than Shor (7 qubits vs 9)
- Built from Hamming code
- Transversal Clifford gates

---

**Day 690 Complete!** Week 99: 4/7 days (57%)
