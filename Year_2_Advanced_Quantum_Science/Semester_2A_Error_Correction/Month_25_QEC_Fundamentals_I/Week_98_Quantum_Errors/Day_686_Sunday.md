# Day 686: Week 98 Synthesis and Shor Code Preview

## Week 98: Quantum Errors | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Week Synthesis & Shor Code |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Comprehensive Problems |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Implementation Lab |

---

## Learning Objectives

By the end of Day 686, you will be able to:

1. **Synthesize all Week 98 concepts** into a unified understanding
2. **Construct Shor's 9-qubit code** by concatenating bit-flip and phase-flip
3. **Understand why Shor code achieves distance 3**
4. **Preview the stabilizer formalism** that generalizes these codes
5. **Connect quantum and classical error correction** comprehensively
6. **Prepare for Week 99:** Three-qubit codes deep dive

---

## Week 98 Comprehensive Review

### Day 680: Introduction to Quantum Errors

**Key Concepts:**
- Three obstacles: no-cloning, measurement collapse, continuous errors
- Pauli errors: X (bit-flip), Z (phase-flip), Y = iXZ
- Pauli basis: any error decomposes into $\{I, X, Y, Z\}$
- Discretization under syndrome measurement

**Critical Formula:**
$$E = e_0 I + e_1 X + e_2 Y + e_3 Z \xrightarrow{\text{syndrome}} \text{discrete Pauli}$$

### Day 681: CPTP Maps and Kraus Operators

**Key Concepts:**
- CPTP = Completely Positive Trace-Preserving
- Kraus representation: $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$
- Completeness: $\sum_k E_k^\dagger E_k = I$
- Choi-Jamiołkowski isomorphism

**Critical Formula:**
$$J(\mathcal{E}) = (\mathcal{E} \otimes I)(|\Phi^+\rangle\langle\Phi^+|)$$

### Day 682: Depolarizing and Amplitude Damping

**Key Concepts:**
- Depolarizing: symmetric Pauli noise, $\mathcal{E}(\rho) = \lambda\rho + (1-\lambda)\frac{I}{2}$
- Amplitude damping: T₁ relaxation, $|1\rangle \to |0\rangle$
- Unital vs non-unital channels
- Physical parameters: T₁, gate error rates

**Critical Formula:**
$$\gamma(t) = 1 - e^{-t/T_1}$$

### Day 683: Phase Damping and Combined Noise

**Key Concepts:**
- Phase damping: T₂ dephasing, coherence decay
- Combined noise: T₁ + T_φ → T₂
- Pauli twirling: any noise → Pauli noise
- Model selection for QEC analysis

**Critical Formula:**
$$\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}$$

### Day 684: Three-Qubit Bit-Flip Code

**Key Concepts:**
- Logical states: $|0_L\rangle = |000\rangle$, $|1_L\rangle = |111\rangle$
- Stabilizers: $Z_1Z_2$, $Z_2Z_3$
- Syndrome measurement without data collapse
- Limitation: cannot detect Z errors

**Critical Formula:**
$$|\psi_L\rangle = \alpha|000\rangle + \beta|111\rangle$$

### Day 685: Three-Qubit Phase-Flip Code

**Key Concepts:**
- Logical states: $|0_L\rangle = |{+}{+}{+}\rangle$, $|1_L\rangle = |{-}{-}{-}\rangle$
- Stabilizers: $X_1X_2$, $X_2X_3$
- Hadamard duality with bit-flip code
- Limitation: cannot detect X errors

**Critical Formula:**
$$\text{Phase-flip} = H^{\otimes 3} (\text{Bit-flip}) H^{\otimes 3}$$

---

## The Shor Code: Concatenation Magic

### The Problem Restated

| Code | Corrects | Cannot Detect |
|------|----------|---------------|
| Bit-flip [[3,1,1]] | X errors | Z errors |
| Phase-flip [[3,1,1]] | Z errors | X errors |

**Neither code can handle general Pauli errors!**

### Shor's Insight: Concatenation

**Idea:** Encode against one type of error, then encode again against the other.

**Step 1:** Encode against Z errors using phase-flip code
$$|0\rangle \to |{+}{+}{+}\rangle, \quad |1\rangle \to |{-}{-}{-}\rangle$$

**Step 2:** Encode each $|+\rangle$ and $|-\rangle$ against X errors using bit-flip code
$$|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} \to \frac{|000\rangle + |111\rangle}{\sqrt{2}}$$
$$|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}} \to \frac{|000\rangle - |111\rangle}{\sqrt{2}}$$

### Shor Code Logical States

$$\boxed{|0_L\rangle = \frac{1}{2\sqrt{2}}\left(|000\rangle + |111\rangle\right)^{\otimes 3}}$$

$$\boxed{|1_L\rangle = \frac{1}{2\sqrt{2}}\left(|000\rangle - |111\rangle\right)^{\otimes 3}}$$

**Expanded form:**

$$|0_L\rangle = \frac{1}{2\sqrt{2}}\left(|000\rangle + |111\rangle\right)\left(|000\rangle + |111\rangle\right)\left(|000\rangle + |111\rangle\right)$$

This is 9 physical qubits encoding 1 logical qubit: **[[9, 1, 3]]**

### Code Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| n | 9 | Physical qubits |
| k | 1 | Logical qubits |
| d | 3 | Code distance |
| Rate | 1/9 ≈ 11% | Encoding efficiency |
| Error correction | $\lfloor(3-1)/2\rfloor = 1$ | Corrects 1 error |

---

## How Shor Code Detects All Errors

### Structure

The 9 qubits are organized into 3 blocks of 3:

```
Block 1: qubits 1, 2, 3
Block 2: qubits 4, 5, 6
Block 3: qubits 7, 8, 9
```

### X Error Detection (Within Blocks)

Within each block, we have bit-flip code structure:
- Stabilizers: $Z_1Z_2$, $Z_2Z_3$ for block 1
- Similar for blocks 2 and 3

**6 stabilizers** for X error detection:
$$Z_1Z_2, Z_2Z_3, Z_4Z_5, Z_5Z_6, Z_7Z_8, Z_8Z_9$$

### Z Error Detection (Across Blocks)

Across blocks, we have phase-flip code structure:
- Compare block 1 with block 2
- Compare block 2 with block 3

**2 stabilizers** for Z error detection:
$$X_1X_2X_3X_4X_5X_6, \quad X_4X_5X_6X_7X_8X_9$$

### Complete Stabilizer Set

The 8 stabilizers of the Shor code:

| Stabilizer | Type | What it detects |
|------------|------|-----------------|
| $Z_1Z_2$ | X-syndrome | X on qubit 1 or 2 |
| $Z_2Z_3$ | X-syndrome | X on qubit 2 or 3 |
| $Z_4Z_5$ | X-syndrome | X on qubit 4 or 5 |
| $Z_5Z_6$ | X-syndrome | X on qubit 5 or 6 |
| $Z_7Z_8$ | X-syndrome | X on qubit 7 or 8 |
| $Z_8Z_9$ | X-syndrome | X on qubit 8 or 9 |
| $X_1X_2X_3X_4X_5X_6$ | Z-syndrome | Z in blocks 1 or 2 |
| $X_4X_5X_6X_7X_8X_9$ | Z-syndrome | Z in blocks 2 or 3 |

Together: **8 stabilizers**, $2^8 = 256$ syndrome outcomes.

But we only have $9 \times 4 = 36$ single-qubit Pauli errors (including identity).

Plenty of syndromes to uniquely identify each single-qubit error!

---

## Shor Code Syndrome Decoding

### X Error Syndromes

Within each block, X errors are detected by ZZ stabilizers:

**Block 1:**
| Error | $Z_1Z_2$ | $Z_2Z_3$ |
|-------|----------|----------|
| I | +1 | +1 |
| $X_1$ | -1 | +1 |
| $X_2$ | -1 | -1 |
| $X_3$ | +1 | -1 |

Similarly for blocks 2 and 3.

### Z Error Syndromes

Z errors anywhere in a block produce the same effect on the inter-block stabilizers:

| Z error location | $X_{123456}$ | $X_{456789}$ |
|------------------|--------------|--------------|
| Block 1 only | -1 | +1 |
| Block 2 only | -1 | -1 |
| Block 3 only | +1 | -1 |

**Wait!** How do we distinguish $Z_1$ from $Z_2$ from $Z_3$?

**Answer:** We don't need to! All Z errors within a block have the same effect on the logical qubit. Applying $Z_1$, $Z_2$, or $Z_3$ to correct is equivalent.

This is the **degeneracy** of the Shor code — multiple physical corrections give the same logical result.

---

## Y Errors: The Combined Case

Y = iXZ, so a Y error triggers both X and Z syndromes:

$$Y_j = iX_jZ_j$$

The syndrome from $Y_j$ is the combination of:
1. X syndrome from $X_j$
2. Z syndrome from $Z_j$

Since both are detectable, Y errors are correctable!

---

## Preview: Stabilizer Formalism

### What We've Discovered

Looking at the pattern:
1. **Stabilizers** are commuting operators that fix the code space
2. **Syndrome** = eigenvalues of stabilizer measurements
3. **Correction** = determined by syndrome lookup

### The General Framework

The **stabilizer formalism** (Week 99-100) generalizes this:

**Definition:** A stabilizer code is defined by an abelian subgroup $\mathcal{S} \subset \mathcal{P}_n$ that:
1. Does not contain $-I$
2. Fixes the code space: $S|\psi_L\rangle = |\psi_L\rangle$ for all $S \in \mathcal{S}$

**Key properties:**
- If $\mathcal{S}$ has $n-k$ independent generators, code encodes $k$ logical qubits
- Errors either commute or anticommute with stabilizers
- Anticommuting errors are detectable

### CSS Codes

Both bit-flip and phase-flip codes are **CSS codes** (Calderbank-Shor-Steane):
- Built from two classical codes $C_1, C_2$ with $C_2^\perp \subseteq C_1$
- X stabilizers from $C_2^\perp$
- Z stabilizers from $C_1$

The Shor code is a CSS code!

---

## The Road Ahead

### Week 99: Three-Qubit Codes

Deep dive into:
- Complete analysis of 3-qubit codes
- Shor code detailed implementation
- Steane [[7,1,3]] code introduction
- Knill-Laflamme conditions

### Week 100: QEC Conditions

Fundamental theory:
- Error correction conditions
- Approximate QEC
- Code distance and error correction capability
- Degenerate vs non-degenerate codes

---

## Worked Examples

### Example 1: Shor Code Encoding

**Problem:** Write out the Shor code encoding of $|1\rangle$.

**Solution:**

Step 1: Phase-flip encoding of $|1\rangle$:
$$|1\rangle \to |{-}{-}{-}\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

Step 2: Bit-flip encoding of each $|-\rangle$:
$$|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) \to \frac{1}{\sqrt{2}}(|000\rangle - |111\rangle)$$

Final result:
$$|1_L\rangle = \frac{1}{2\sqrt{2}}\left(|000\rangle - |111\rangle\right)\left(|000\rangle - |111\rangle\right)\left(|000\rangle - |111\rangle\right)$$

### Example 2: Syndrome for $Z_5$ Error

**Problem:** What syndrome does a $Z_5$ error produce on the Shor code?

**Solution:**

$Z_5$ acts on qubit 5, which is in block 2.

**X syndromes (ZZ stabilizers):**
- $Z_1Z_2$: commutes with $Z_5$ → +1
- $Z_2Z_3$: commutes with $Z_5$ → +1
- $Z_4Z_5$: anticommutes with $Z_5$ → -1
- $Z_5Z_6$: anticommutes with $Z_5$ → -1
- $Z_7Z_8$: commutes → +1
- $Z_8Z_9$: commutes → +1

Wait, the ZZ stabilizers anticommute with Z on their support!

Actually, $[Z_4Z_5, Z_5] = 0$ (Z commutes with Z). Let me reconsider.

The ZZ stabilizers are for detecting X errors, not Z errors. They should all give +1 for a Z error.

**Z syndromes (XX...X stabilizers):**
- $X_1X_2X_3X_4X_5X_6$: anticommutes with $Z_5$ → -1
- $X_4X_5X_6X_7X_8X_9$: anticommutes with $Z_5$ → -1

**Syndrome:** (0,0,0,0,0,0,-1,-1) or in short: X-syndrome = (0,0,0,0,0,0), Z-syndrome = (1,1)

This identifies a Z error in block 2. Correction: apply $Z_4$ (or $Z_5$ or $Z_6$).

### Example 3: Verify $Y_1$ Syndrome

**Problem:** Verify that $Y_1$ error is detectable.

**Solution:**

$Y_1 = iX_1Z_1$.

**X part ($X_1$):**
- $Z_1Z_2$ anticommutes with $X_1$ → -1
- $Z_2Z_3$ commutes → +1
- Other ZZ stabilizers commute → +1

X-syndrome: (1,0,0,0,0,0)

**Z part ($Z_1$):**
- $X_1X_2X_3X_4X_5X_6$ anticommutes with $Z_1$ → -1
- $X_4X_5X_6X_7X_8X_9$ commutes → +1

Z-syndrome: (1,0)

**Combined:** Unique syndrome identifies $Y_1$. Correction: apply $Y_1$.

---

## Practice Problems

### Problem Set A: Synthesis Review

**A.1** List all three fundamental obstacles to quantum error correction and how we overcome each.

**A.2** Write the Kraus operators for the depolarizing channel and verify the completeness relation.

**A.3** Draw the encoding circuit for the phase-flip code.

### Problem Set B: Shor Code

**B.1** How many distinct syndromes can the Shor code produce? How many single-qubit Pauli errors exist?

**B.2** Write out $|0_L\rangle$ for the Shor code as a sum of computational basis states.

**B.3** What is the syndrome for the error $X_2X_5$ (two-qubit error)?

### Problem Set C: Looking Forward

**C.1** The Steane code is [[7,1,3]]. How does its efficiency (rate k/n) compare to Shor?

**C.2** Explain why the Shor code is called "degenerate."

**C.3** If we want a code that corrects 2 errors, what minimum distance do we need?

---

## Computational Lab: Shor Code Implementation

```python
"""
Day 686 Computational Lab: Shor Code Implementation
===================================================

Building and simulating Shor's 9-qubit code.
"""

import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from itertools import product

# =============================================================================
# Part 1: Basic Setup
# =============================================================================

# Single qubit states and operators
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def tensor(*args):
    """Compute tensor product."""
    result = args[0]
    for arg in args[1:]:
        result = np.kron(result, arg)
    return result

def apply_single_qubit(op, qubit, n_qubits):
    """Apply single-qubit operator to specified qubit in n-qubit system."""
    ops = [I] * n_qubits
    ops[qubit] = op
    return tensor(*ops)

print("=" * 70)
print("SHOR CODE [[9,1,3]] IMPLEMENTATION")
print("=" * 70)

# =============================================================================
# Part 2: Shor Code Logical States
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: Constructing Logical States")
print("=" * 70)

# Build |0_L⟩ and |1_L⟩
# |+⟩ encoded in bit-flip: (|000⟩ + |111⟩)/√2
# |-⟩ encoded in bit-flip: (|000⟩ - |111⟩)/√2

ket_plus_encoded = (tensor(ket_0, ket_0, ket_0) + tensor(ket_1, ket_1, ket_1)) / np.sqrt(2)
ket_minus_encoded = (tensor(ket_0, ket_0, ket_0) - tensor(ket_1, ket_1, ket_1)) / np.sqrt(2)

# |0_L⟩ = |+_enc⟩|+_enc⟩|+_enc⟩
ket_0L_shor = tensor(ket_plus_encoded, ket_plus_encoded, ket_plus_encoded)

# |1_L⟩ = |-_enc⟩|-_enc⟩|-_enc⟩
ket_1L_shor = tensor(ket_minus_encoded, ket_minus_encoded, ket_minus_encoded)

print(f"\n|0_L⟩ norm: {np.linalg.norm(ket_0L_shor):.6f}")
print(f"|1_L⟩ norm: {np.linalg.norm(ket_1L_shor):.6f}")
print(f"⟨0_L|1_L⟩: {np.abs(np.vdot(ket_0L_shor, ket_1L_shor)):.10f} (should be 0)")

# Count non-zero amplitudes
nonzero_0L = np.sum(np.abs(ket_0L_shor) > 1e-10)
nonzero_1L = np.sum(np.abs(ket_1L_shor) > 1e-10)
print(f"\nNumber of non-zero amplitudes in |0_L⟩: {nonzero_0L}")
print(f"Number of non-zero amplitudes in |1_L⟩: {nonzero_1L}")

# =============================================================================
# Part 3: Stabilizer Generators
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: Stabilizer Generators")
print("=" * 70)

n = 9  # 9 qubits

# ZZ stabilizers (for X error detection within blocks)
def make_ZZ(i, j):
    """Create Z_i Z_j stabilizer on 9 qubits."""
    ops = [I] * n
    ops[i] = Z
    ops[j] = Z
    return tensor(*ops)

ZZ_12 = make_ZZ(0, 1)  # Z_1 Z_2
ZZ_23 = make_ZZ(1, 2)  # Z_2 Z_3
ZZ_45 = make_ZZ(3, 4)  # Z_4 Z_5
ZZ_56 = make_ZZ(4, 5)  # Z_5 Z_6
ZZ_78 = make_ZZ(6, 7)  # Z_7 Z_8
ZZ_89 = make_ZZ(7, 8)  # Z_8 Z_9

# XX...X stabilizers (for Z error detection across blocks)
def make_XXXXXX(qubits):
    """Create X on specified qubits."""
    ops = [I] * n
    for q in qubits:
        ops[q] = X
    return tensor(*ops)

XXXXXX_123456 = make_XXXXXX([0, 1, 2, 3, 4, 5])  # X_1...X_6
XXXXXX_456789 = make_XXXXXX([3, 4, 5, 6, 7, 8])  # X_4...X_9

stabilizers = [ZZ_12, ZZ_23, ZZ_45, ZZ_56, ZZ_78, ZZ_89, XXXXXX_123456, XXXXXX_456789]
stab_names = ['Z₁Z₂', 'Z₂Z₃', 'Z₄Z₅', 'Z₅Z₆', 'Z₇Z₈', 'Z₈Z₉', 'X₁₂₃₄₅₆', 'X₄₅₆₇₈₉']

# Verify stabilizers fix the logical states
print("\nVerifying stabilizers fix |0_L⟩:")
for name, S in zip(stab_names, stabilizers):
    eigenval = np.real(ket_0L_shor.conj() @ S @ ket_0L_shor)
    print(f"  {name}: eigenvalue = {eigenval:+.4f}")

# =============================================================================
# Part 4: Syndrome Calculation
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: Syndrome Calculation")
print("=" * 70)

def get_syndrome(state: np.ndarray) -> Tuple[int, ...]:
    """Calculate syndrome (tuple of 8 bits)."""
    syndrome = []
    for S in stabilizers:
        eigenval = np.real(state.conj() @ S @ state)
        syndrome.append(0 if eigenval > 0 else 1)
    return tuple(syndrome)

def format_syndrome(s: Tuple[int, ...]) -> str:
    """Format syndrome for display."""
    return f"({','.join(map(str, s[:6])}) | {','.join(map(str, s[6:]))})"

# Build syndrome table for single-qubit errors
print("\nSyndrome table for single-qubit Pauli errors:")
print("-" * 80)
print(f"{'Error':<10} {'ZZ Syndromes (6 bits)':<30} {'XX Syndromes (2 bits)':<20}")
print("-" * 80)

# Create all single-qubit Pauli errors
pauli_errors = {}
for q in range(9):
    for P, P_name in [(X, 'X'), (Y, 'Y'), (Z, 'Z')]:
        name = f"{P_name}{q+1}"
        error_op = apply_single_qubit(P, q, n)
        pauli_errors[name] = error_op

# Identity
pauli_errors['I'] = np.eye(2**n, dtype=complex)

# Calculate syndromes
syndrome_to_error = {}
for name, E in pauli_errors.items():
    error_state = E @ ket_0L_shor
    s = get_syndrome(error_state)
    syndrome_to_error[s] = name
    if name == 'I' or name.startswith('X1') or name.startswith('Y5') or name.startswith('Z9'):
        zz_part = ''.join(map(str, s[:6]))
        xx_part = ''.join(map(str, s[6:]))
        print(f"{name:<10} {zz_part:<30} {xx_part:<20}")

print(f"\nTotal unique syndromes: {len(syndrome_to_error)}")
print(f"Total single-qubit errors (including I): {len(pauli_errors)}")

# =============================================================================
# Part 5: Error Correction
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: Error Correction Demonstration")
print("=" * 70)

def correct_error(state: np.ndarray) -> np.ndarray:
    """Correct single-qubit error based on syndrome."""
    s = get_syndrome(state)

    if s == (0,0,0,0,0,0,0,0):
        return state  # No error

    # Look up correction
    if s in syndrome_to_error:
        error_name = syndrome_to_error[s]
        correction = pauli_errors[error_name]
        return correction @ state  # Apply same error to cancel
    else:
        print(f"Unknown syndrome: {s}")
        return state

def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute fidelity."""
    return np.abs(np.vdot(state1, state2))**2

# Test error correction on random logical state
print("\nError correction test on |ψ_L⟩ = (|0_L⟩ + i|1_L⟩)/√2:")

psi_L = (ket_0L_shor + 1j * ket_1L_shor) / np.sqrt(2)

# Test several errors
test_errors = ['I', 'X1', 'X5', 'X9', 'Y3', 'Y7', 'Z2', 'Z6']
print("-" * 60)
print(f"{'Error':<10} {'Syndrome':<25} {'Fidelity After':<15}")
print("-" * 60)

for err_name in test_errors:
    E = pauli_errors[err_name]
    error_state = E @ psi_L
    s = get_syndrome(error_state)
    corrected = correct_error(error_state)
    fid = fidelity(corrected, psi_L)
    print(f"{err_name:<10} {format_syndrome(s):<25} {fid:.8f}")

# =============================================================================
# Part 6: Two-Qubit Error Analysis
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: Two-Qubit Error Analysis (Code Limit)")
print("=" * 70)

# Create some two-qubit errors
X1X2 = apply_single_qubit(X, 0, n) @ apply_single_qubit(X, 1, n)
Z1Z4 = apply_single_qubit(Z, 0, n) @ apply_single_qubit(Z, 3, n)
X3Y7 = apply_single_qubit(X, 2, n) @ apply_single_qubit(Y, 6, n)

two_errors = {'X1X2': X1X2, 'Z1Z4': Z1Z4, 'X3Y7': X3Y7}

print("\nTwo-qubit errors (beyond correction capability):")
print("-" * 60)
print(f"{'Error':<10} {'Detected?':<15} {'Correctly Fixed?':<20}")
print("-" * 60)

for name, E in two_errors.items():
    error_state = E @ psi_L
    s = get_syndrome(error_state)
    detected = s != (0,0,0,0,0,0,0,0)
    corrected = correct_error(error_state)
    fid = fidelity(corrected, psi_L)
    correctly_fixed = fid > 0.99
    print(f"{name:<10} {'Yes' if detected else 'No':<15} {'Yes' if correctly_fixed else 'No (fid=' + f'{fid:.4f})' :<20}")

# =============================================================================
# Part 7: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Shor code structure
ax1 = axes[0]
ax1.axis('off')

structure = """
┌────────────────────────────────────────────────────┐
│               SHOR CODE [[9,1,3]]                   │
├────────────────────────────────────────────────────┤
│                                                     │
│  Structure: 3 blocks × 3 qubits = 9 qubits         │
│                                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐          │
│  │ Block 1 │   │ Block 2 │   │ Block 3 │          │
│  │ q1 q2 q3│   │ q4 q5 q6│   │ q7 q8 q9│          │
│  │  |+++⟩  │   │  |+++⟩  │   │  |+++⟩  │ → |0_L⟩  │
│  │  |---⟩  │   │  |---⟩  │   │  |---⟩  │ → |1_L⟩  │
│  └─────────┘   └─────────┘   └─────────┘          │
│       ↑              ↑              ↑              │
│       └──── Z_iZ_j stabilizers ─────┘  (6 total)  │
│                                                     │
│  ←───── X_1...X_6 ─────→ ←── X_4...X_9 ──→        │
│         (2 X-type stabilizers)                     │
│                                                     │
│  Corrects: Any single X, Y, or Z error             │
│  Distance: d = 3                                    │
│                                                     │
└────────────────────────────────────────────────────┘
"""
ax1.text(0.02, 0.98, structure, transform=ax1.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax1.set_title('Shor Code Structure')

# Plot 2: Error correction performance
ax2 = axes[1]

# Simulate error correction under depolarizing noise
n_trials = 500
error_probs = np.linspace(0, 0.2, 15)
success_rates = []
uncoded_rates = []

for p in error_probs:
    successes = 0
    uncoded_successes = 0

    for _ in range(n_trials):
        # Random logical state
        alpha = np.random.uniform(0, 1)
        beta = np.sqrt(1 - alpha**2) * np.exp(1j * np.random.uniform(0, 2*np.pi))
        psi_rand_L = alpha * ket_0L_shor + beta * ket_1L_shor

        # Apply depolarizing-like noise (single-qubit errors)
        state = psi_rand_L.copy()
        for q in range(9):
            if np.random.random() < p:
                error_type = np.random.choice([X, Y, Z])
                E = apply_single_qubit(error_type, q, n)
                state = E @ state

        # Error correction
        corrected = correct_error(state)
        if fidelity(corrected, psi_rand_L) > 0.99:
            successes += 1

        # Uncoded comparison (single qubit)
        psi_1q = alpha * ket_0 + beta * ket_1
        if np.random.random() < p:
            error_type = np.random.choice([X, Y, Z])
            psi_1q = error_type @ psi_1q
        if fidelity(psi_1q, alpha * ket_0 + beta * ket_1) > 0.99:
            uncoded_successes += 1

    success_rates.append(successes / n_trials)
    uncoded_rates.append(uncoded_successes / n_trials)

ax2.plot(error_probs * 100, np.array(success_rates) * 100, 'b-', linewidth=2, label='Shor [[9,1,3]]')
ax2.plot(error_probs * 100, np.array(uncoded_rates) * 100, 'r--', linewidth=2, label='Uncoded')
ax2.set_xlabel('Error Rate per Qubit (%)')
ax2.set_ylabel('Logical Success Rate (%)')
ax2.set_title('Error Correction Performance')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

# Plot 3: Week 98 summary
ax3 = axes[2]
ax3.axis('off')

week_summary = """
┌──────────────────────────────────────────────────┐
│          WEEK 98 SYNTHESIS COMPLETE              │
├──────────────────────────────────────────────────┤
│                                                  │
│ Day 680: Quantum Errors                          │
│   • Pauli errors X, Y, Z                         │
│   • Discretization principle                     │
│                                                  │
│ Day 681: CPTP Maps                               │
│   • Kraus representation                         │
│   • Choi-Jamiołkowski                           │
│                                                  │
│ Day 682: Depolarizing & Amplitude Damping        │
│   • Physical noise models                        │
│   • T₁ relaxation                                │
│                                                  │
│ Day 683: Phase Damping & Combined                │
│   • T₂ dephasing                                 │
│   • Pauli twirling                               │
│                                                  │
│ Day 684: Bit-Flip Code [[3,1,1]]                 │
│   • Protects X errors                            │
│                                                  │
│ Day 685: Phase-Flip Code [[3,1,1]]               │
│   • Protects Z errors                            │
│   • Hadamard duality                             │
│                                                  │
│ Day 686: Shor Code [[9,1,3]]                     │
│   • Concatenation magic                          │
│   • First complete quantum code!                 │
│                                                  │
│ NEXT: Week 99 - Stabilizer Formalism             │
└──────────────────────────────────────────────────┘
"""
ax3.text(0.02, 0.98, week_summary, transform=ax3.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax3.set_title('Week 98 Summary')

plt.tight_layout()
plt.savefig('day_686_shor_code.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_686_shor_code.png")

# =============================================================================
# Final Summary
# =============================================================================

print("\n" + "=" * 70)
print("WEEK 98 COMPLETE - SUMMARY")
print("=" * 70)

final_summary = """
┌────────────────────────────────────────────────────────────────────────┐
│                    WEEK 98: QUANTUM ERRORS - COMPLETE                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ KEY ACHIEVEMENTS:                                                       │
│ ✓ Understood quantum error types (X, Y, Z) and discretization          │
│ ✓ Mastered CPTP maps and Kraus operator formalism                      │
│ ✓ Analyzed physical noise: depolarizing, amplitude/phase damping       │
│ ✓ Built bit-flip code (protects X)                                     │
│ ✓ Built phase-flip code (protects Z)                                   │
│ ✓ Constructed Shor [[9,1,3]] code (protects all single-qubit errors)   │
│                                                                         │
│ MATHEMATICAL TOOLS:                                                     │
│ • Pauli group: {I, X, Y, Z}^⊗n                                         │
│ • Kraus representation: E(ρ) = Σₖ Eₖ ρ Eₖ†                             │
│ • Syndrome = stabilizer eigenvalues                                     │
│ • Code concatenation                                                    │
│                                                                         │
│ PHYSICAL INSIGHTS:                                                      │
│ • T₁: energy relaxation (amplitude damping)                            │
│ • T₂: total decoherence (T₂ ≤ 2T₁)                                     │
│ • Gate errors ~ 0.1-1% in modern hardware                              │
│                                                                         │
│ LOOKING AHEAD (Week 99):                                                │
│ • Stabilizer formalism                                                  │
│ • General theory of quantum codes                                       │
│ • Knill-Laflamme conditions                                            │
│ • Steane [[7,1,3]] code                                                 │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
"""
print(final_summary)

print("\n✅ Day 686 and Week 98 Complete!")
print("=" * 70)
```

---

## Summary

### Week 98 Key Formulas

| Day | Topic | Key Formula |
|-----|-------|-------------|
| 680 | Pauli errors | $Y = iXZ$, Paulis form error basis |
| 681 | Kraus operators | $\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$ |
| 682 | Depolarizing | $\lambda = 1 - 4p/3$ (shrinking factor) |
| 683 | T₂ | $1/T_2 = 1/(2T_1) + 1/T_\phi$ |
| 684 | Bit-flip | $\|0_L\rangle = \|000\rangle$, stabilizers $Z_iZ_j$ |
| 685 | Phase-flip | $\|0_L\rangle = \|{+}{+}{+}\rangle$, stabilizers $X_iX_j$ |
| 686 | Shor code | $[[9,1,3]]$ = concatenation of both |

### The Shor Code

$$|0_L\rangle = \frac{1}{2\sqrt{2}}\left(|000\rangle + |111\rangle\right)^{\otimes 3}$$

- **9 physical qubits** encoding **1 logical qubit**
- **Distance 3:** corrects any single-qubit error
- **8 stabilizers:** 6 ZZ-type, 2 XXXXXX-type
- **Degenerate:** multiple physical corrections → same logical correction

### Main Takeaways

1. **Quantum errors discretize** under syndrome measurement
2. **CPTP maps** are the mathematical framework for noise
3. **Physical noise** characterized by T₁, T₂ timescales
4. **Bit-flip and phase-flip codes** are dual via Hadamard
5. **Concatenation** combines codes to protect against all errors
6. **Shor code** was the first complete quantum error correcting code

---

## Daily Checklist

- [ ] I can synthesize Week 98 concepts coherently
- [ ] I understand how Shor code is constructed via concatenation
- [ ] I can list all 8 stabilizers of the Shor code
- [ ] I understand syndrome calculation for the Shor code
- [ ] I know why the code is called "degenerate"
- [ ] I'm ready for the stabilizer formalism in Week 99

---

## Preview: Week 99

**Week 99: Three-Qubit Codes (Deep Dive)**
- Complete analysis of small codes
- Stabilizer formalism introduction
- Knill-Laflamme conditions
- Steane [[7,1,3]] code
- CSS code construction

---

*"The Shor code was the first quantum error correcting code to protect against all single-qubit errors. It showed that fault-tolerant quantum computation was possible in principle."*
— Peter Shor (1995)

---

**Day 686 Complete! Week 98: 7/7 days (100%)**

---

## Week 98 Achievement

🎉 **Week 98 Complete!**

You have mastered:
- Quantum error theory
- CPTP maps and Kraus operators
- Physical noise models (T₁, T₂)
- Three-qubit repetition codes
- Shor's 9-qubit code

**Next:** Week 99 — Three-Qubit Codes & Stabilizer Formalism
