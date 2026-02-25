# Day 685: Three-Qubit Phase-Flip Code

## Week 98: Quantum Errors | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Phase-Flip Code Theory |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Problem Solving |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational Lab |

---

## Learning Objectives

By the end of Day 685, you will be able to:

1. **Construct the three-qubit phase-flip code** using the Hadamard basis
2. **Understand the duality** between bit-flip and phase-flip codes
3. **Implement syndrome measurement** for Z errors using X-basis measurements
4. **Recognize that Z-errors in Z-basis = X-errors in X-basis**
5. **Analyze limitations** of the phase-flip code against X errors
6. **Motivate the need for concatenation** to protect against both error types

---

## The Duality Principle

### Bit-Flip vs Phase-Flip: A Symmetry

Recall from Day 680:
- X errors in the **computational basis** $\{|0\rangle, |1\rangle\}$ flip the bit
- Z errors in the **Hadamard basis** $\{|+\rangle, |-\rangle\}$ flip the bit!

$$Z|+\rangle = |-\rangle, \quad Z|-\rangle = |+\rangle$$

**Key insight:** A Z error is just an X error in a different basis!

### The Hadamard Transformation

The Hadamard gate transforms between bases:

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$$H|0\rangle = |+\rangle, \quad H|1\rangle = |-\rangle$$
$$H|+\rangle = |0\rangle, \quad H|-\rangle = |1\rangle$$

And importantly:
$$HXH = Z, \quad HZH = X$$

**The Hadamard gate swaps X and Z!**

---

## The Three-Qubit Phase-Flip Code

### Logical States

Instead of encoding in $\{|0\rangle, |1\rangle\}$, we encode in $\{|+\rangle, |-\rangle\}$:

$$\boxed{|0_L\rangle = |{+}{+}{+}\rangle, \quad |1_L\rangle = |{-}{-}{-}\rangle}$$

Expanding in computational basis:

$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle), \quad |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

$$|{+}{+}{+}\rangle = \frac{1}{2\sqrt{2}}(|0\rangle + |1\rangle)^{\otimes 3} = \frac{1}{2\sqrt{2}}\sum_{x \in \{0,1\}^3} |x\rangle$$

$$|{-}{-}{-}\rangle = \frac{1}{2\sqrt{2}}(|0\rangle - |1\rangle)^{\otimes 3} = \frac{1}{2\sqrt{2}}\sum_{x \in \{0,1\}^3} (-1)^{|x|} |x\rangle$$

where $|x|$ is the Hamming weight (number of 1s).

### General Encoded State

$$|\psi_L\rangle = \alpha|0_L\rangle + \beta|1_L\rangle = \alpha|{+}{+}{+}\rangle + \beta|{-}{-}{-}\rangle$$

---

## Code Parameters: [[3, 1, 1]] (Different Error Set)

The phase-flip code also has parameters $[[3, 1, 1]]$:

- **n = 3:** Three physical qubits
- **k = 1:** One logical qubit
- **d = 1:** Distance 1 against general errors (X errors undetected!)

But against Z errors only: $d_Z = 3$.

| Code | Protects Against | Vulnerable To | Distance |
|------|------------------|---------------|----------|
| Bit-flip | X errors | Z errors | $d_X = 3$ |
| Phase-flip | Z errors | X errors | $d_Z = 3$ |

They are **dual codes** — swapping X ↔ Z.

---

## Encoding Circuit

### Hadamard-Transformed Bit-Flip Encoding

The phase-flip encoding is the bit-flip encoding **wrapped in Hadamards**:

```
|ψ⟩ ──H──●──●──H── |ψ_L⟩ qubit 1
         │  │
|0⟩ ──H──⊕──┼──H── |ψ_L⟩ qubit 2
            │
|0⟩ ──H─────⊕──H── |ψ_L⟩ qubit 3
```

**Simplified circuit:**

```
|ψ⟩ ──H──●──●── |ψ_L⟩ qubit 1
         │  │
|+⟩ ─────⊕──┼── |ψ_L⟩ qubit 2
            │
|+⟩ ────────⊕── |ψ_L⟩ qubit 3
```

Starting with $|+\rangle$ on ancillas (achieved by $H|0\rangle$).

### Verification

Start with $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:

1. Apply H: $\alpha|+\rangle + \beta|-\rangle$
2. Tensor with $|+\rangle|+\rangle$: $(\alpha|+\rangle + \beta|-\rangle)|+\rangle|+\rangle$
3. Apply CNOT$_{12}$: $\alpha|{+}{+}\rangle|+\rangle + \beta|{-}{-}\rangle|+\rangle$
4. Apply CNOT$_{13}$: $\alpha|{+}{+}{+}\rangle + \beta|{-}{-}{-}\rangle$

**Note:** CNOT in the Hadamard basis acts as: controlled-Z in computational basis!

$$\text{CNOT}(|+\rangle \otimes |+\rangle) = |{+}{+}\rangle$$
$$\text{CNOT}(|-\rangle \otimes |+\rangle) = |{-}{-}\rangle$$

---

## Syndrome Measurement

### Syndrome Operators

For the phase-flip code, we measure **X-type stabilizers**:

$$\boxed{X_1X_2 \text{ and } X_2X_3}$$

These detect Z errors because:
$$X_1X_2 \cdot Z_1 = -Z_1 \cdot X_1X_2$$

(Z anticommutes with the X that acts on the same qubit.)

### Syndrome Table

| Error | $X_1X_2$ | $X_2X_3$ | Syndrome |
|-------|----------|----------|----------|
| I (no error) | +1 | +1 | (0, 0) |
| $Z_1$ | −1 | +1 | (1, 0) |
| $Z_2$ | −1 | −1 | (1, 1) |
| $Z_3$ | +1 | −1 | (0, 1) |

Each single Z error has a unique syndrome — exactly like X errors in the bit-flip code!

### Measurement Circuit

To measure $X_1X_2$:

```
|0⟩ ──H──●──●──H──M──  ancilla
         │  │
 q1 ─────X──┼─────────
            │
 q2 ────────X─────────
```

Or equivalently using controlled-X (CNOT) with the ancilla as target:

```
|+⟩ ─────●──●──────M_X──  ancilla
         │  │
 q1 ─────Z──┼──────────
            │
 q2 ────────Z──────────
```

---

## Duality: A Unified View

### The Bit-Flip/Phase-Flip Correspondence

| Bit-Flip Code | Phase-Flip Code | Transformation |
|---------------|-----------------|----------------|
| $\|0_L\rangle = \|000\rangle$ | $\|0_L\rangle = \|{+}{+}{+}\rangle$ | $H^{\otimes 3}$ |
| $\|1_L\rangle = \|111\rangle$ | $\|1_L\rangle = \|{-}{-}{-}\rangle$ | $H^{\otimes 3}$ |
| Stabilizers: $Z_1Z_2, Z_2Z_3$ | Stabilizers: $X_1X_2, X_2X_3$ | $HZH = X$ |
| Corrects: X errors | Corrects: Z errors | $HXH = Z$ |
| $X_L = X_1X_2X_3$ | $X_L = X_1X_2X_3$ | Same! |
| $Z_L = Z_1Z_2Z_3$ | $Z_L = Z_1Z_2Z_3$ | Same! |

**The codes are Hadamard-conjugates of each other!**

### CSS Code Structure (Preview)

Both bit-flip and phase-flip codes are **CSS codes** (Calderbank-Shor-Steane):
- Classical code $C$ for X errors
- Classical code $C^\perp$ (or related) for Z errors

This structure becomes central in Weeks 99-100.

---

## Limitations and the Need for Shor Code

### X Errors Are Undetected

Just as Z errors pass through the bit-flip code undetected, X errors pass through the phase-flip code:

$$X_1|{+}{+}{+}\rangle = |{+}{+}{+}\rangle$$
$$X_1|{-}{-}{-}\rangle = |{-}{-}{-}\rangle$$

Wait, $X|+\rangle = |+\rangle$ (eigenstates!), so the syndrome is (0,0) — no error detected!

But the logical state IS affected:
$$X_L = X_1X_2X_3$$

A single X error is undetectable but doesn't corrupt the logical qubit... unless combined with other errors.

### Neither Code Is Complete

| Error Type | Bit-Flip Code | Phase-Flip Code |
|------------|---------------|-----------------|
| X only | ✓ Corrects | ✗ Undetected |
| Z only | ✗ Undetected | ✓ Corrects |
| Y = iXZ | ✗ Miscorrects | ✗ Miscorrects |
| General | ✗ Fails | ✗ Fails |

### The Solution: Concatenation

**Shor's 9-qubit code** concatenates both codes:
1. Encode against Z using phase-flip code (3 qubits)
2. Encode each of those 3 qubits against X using bit-flip code

Result: 9 physical qubits protecting against all single-qubit Pauli errors!

---

## Worked Examples

### Example 1: Encoding |1⟩ into Phase-Flip Code

**Problem:** Encode $|1\rangle$ into the phase-flip code.

**Solution:**

$$|1\rangle \xrightarrow{H} |-\rangle$$

Then encode $|-\rangle$ using the phase-flip encoding:

$$|1_L\rangle = |{-}{-}{-}\rangle = \frac{1}{2\sqrt{2}}(|000\rangle - |001\rangle - |010\rangle + |011\rangle - |100\rangle + |101\rangle + |110\rangle - |111\rangle)$$

### Example 2: Z Error and Syndrome

**Problem:** Apply $Z_2$ to $|0_L\rangle = |{+}{+}{+}\rangle$ and compute the syndrome.

**Solution:**

$$Z_2|{+}{+}{+}\rangle = |+\rangle Z|+\rangle |+\rangle = |+\rangle|-\rangle|+\rangle = |{+}{-}{+}\rangle$$

Syndrome calculation:

$X_1X_2|{+}{-}{+}\rangle$:
- $X_1|+\rangle = |+\rangle$ (eigenvalue +1)
- $X_2|-\rangle = -|-\rangle$ (eigenvalue -1)

Product eigenvalue: $(+1)(-1) = -1 \Rightarrow s_1 = 1$

$X_2X_3|{+}{-}{+}\rangle$:
- $X_2|-\rangle = -|-\rangle$ (eigenvalue -1)
- $X_3|+\rangle = |+\rangle$ (eigenvalue +1)

Product eigenvalue: $(-1)(+1) = -1 \Rightarrow s_2 = 1$

**Syndrome: (1, 1)** → identifies $Z_2$ error. ✓

### Example 3: X Error Goes Undetected

**Problem:** Apply $X_1$ to $|\psi_L\rangle = \frac{1}{\sqrt{2}}(|{+}{+}{+}\rangle + |{-}{-}{-}\rangle)$ and show the syndrome is (0,0).

**Solution:**

$$X_1|\psi_L\rangle = \frac{1}{\sqrt{2}}(X_1|{+}{+}{+}\rangle + X_1|{-}{-}{-}\rangle)$$

Since $X|+\rangle = |+\rangle$ and $X|-\rangle = |-\rangle$:

$$= \frac{1}{\sqrt{2}}(|{+}{+}{+}\rangle + |{-}{-}{-}\rangle) = |\psi_L\rangle$$

The state is **unchanged**! The syndrome must be (0,0).

This happens because $|+\rangle$ and $|-\rangle$ are X eigenstates.

---

## Practice Problems

### Problem Set A: Direct Application

**A.1** Write out $|{+}{+}{+}\rangle$ as a superposition of computational basis states.

**A.2** Apply $Z_3$ to $|1_L\rangle = |{-}{-}{-}\rangle$ and compute the resulting state.

**A.3** What is the syndrome for the state $|{+}{-}{-}\rangle$?

### Problem Set B: Intermediate

**B.1** Prove that $X_1X_2$ commutes with $X_2X_3$ but anticommutes with $Z_1$.

**B.2** Show that applying $H^{\otimes 3}$ to the bit-flip codewords $|000\rangle$, $|111\rangle$ gives the phase-flip codewords.

**B.3** What syndrome does the error $Z_1Z_2$ produce? What correction will be applied?

### Problem Set C: Challenging

**C.1** Prove that the logical operators $X_L = X_1X_2X_3$ and $Z_L = Z_1Z_2Z_3$ are the same for both bit-flip and phase-flip codes.

**C.2** Design a circuit that performs syndrome measurement for $X_1X_2$ using only CNOT gates and measurements in the Z basis.

**C.3** Consider a "Y-flip" repetition code defined by $|0_L\rangle = |R\rangle^{\otimes 3}$, $|1_L\rangle = |L\rangle^{\otimes 3}$ where $|R\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$. What errors does it correct?

---

## Computational Lab: Phase-Flip Code Simulation

```python
"""
Day 685 Computational Lab: Three-Qubit Phase-Flip Code
=====================================================

Implementing the phase-flip code and demonstrating duality with bit-flip.
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Basic Definitions
# =============================================================================

# Single qubit states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# Pauli and Hadamard matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def tensor(*args):
    """Compute tensor product of multiple matrices/vectors."""
    result = args[0]
    for arg in args[1:]:
        result = np.kron(result, arg)
    return result

def CNOT(control: int, target: int, n_qubits: int) -> np.ndarray:
    """Create CNOT gate on n qubits."""
    dim = 2**n_qubits
    result = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        new_bits = bits.copy()
        if bits[control] == 1:
            new_bits[target] = 1 - bits[target]
        j = sum(b << (n_qubits - 1 - q) for q, b in enumerate(new_bits))
        result[j, i] = 1
    return result

print("=" * 60)
print("PART 1: Phase-Flip Code Logical States")
print("=" * 60)

# =============================================================================
# Part 2: Phase-Flip Code States
# =============================================================================

# Logical states in phase-flip code
ket_0L_pf = tensor(ket_plus, ket_plus, ket_plus)  # |+++⟩
ket_1L_pf = tensor(ket_minus, ket_minus, ket_minus)  # |---⟩

print("\n|0_L⟩ = |+++⟩ in computational basis:")
labels = ['000', '001', '010', '011', '100', '101', '110', '111']
for i, label in enumerate(labels):
    amp = ket_0L_pf[i]
    if np.abs(amp) > 1e-10:
        print(f"  |{label}⟩: {amp.real:.4f}")

print("\n|1_L⟩ = |---⟩ in computational basis:")
for i, label in enumerate(labels):
    amp = ket_1L_pf[i]
    if np.abs(amp) > 1e-10:
        print(f"  |{label}⟩: {amp.real:+.4f}")

# =============================================================================
# Part 3: Encoding Circuit
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Encoding Circuit")
print("=" * 60)

def encode_phase_flip(state_1q: np.ndarray) -> np.ndarray:
    """
    Encode a single-qubit state into the phase-flip code.
    |ψ⟩ → |ψ_L⟩ = α|+++⟩ + β|---⟩
    """
    # Apply Hadamard to input
    state_H = H @ state_1q

    # Start with |ψ_H⟩|+⟩|+⟩ = |ψ_H⟩ ⊗ H|0⟩ ⊗ H|0⟩
    state_3q = tensor(state_H, ket_plus, ket_plus)

    # Apply CNOTs (in Hadamard basis, acts as controlled-Z pattern)
    # We use standard CNOTs here
    cnot_01 = CNOT(0, 1, 3)
    cnot_02 = CNOT(0, 2, 3)

    state_3q = cnot_01 @ state_3q
    state_3q = cnot_02 @ state_3q

    return state_3q

# Test encoding
alpha, beta = 1/np.sqrt(2), 1/np.sqrt(2)  # |+⟩ state
psi = alpha * ket_0 + beta * ket_1

psi_L_pf = encode_phase_flip(psi)

print(f"\nOriginal state: |ψ⟩ = {alpha:.4f}|0⟩ + {beta:.4f}|1⟩")
print(f"\nEncoded state should be |ψ_L⟩ = α|+++⟩ + β|---⟩")
print(f"where α' = ⟨+|ψ⟩, β' = ⟨-|ψ⟩")

# Verify encoding
expected = alpha * np.vdot(ket_plus, psi) * ket_0L_pf + beta * np.vdot(ket_minus, psi) * ket_1L_pf
# Actually for |+⟩ input: H|+⟩ = |0⟩, so we get |0_L⟩ = |+++⟩

# Show the encoded state
print("\nEncoded state amplitudes:")
for i, label in enumerate(labels):
    amp = psi_L_pf[i]
    if np.abs(amp) > 1e-10:
        print(f"  |{label}⟩: {amp.real:.4f}")

# =============================================================================
# Part 4: Syndrome Measurement for Z Errors
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Syndrome Measurement for Z Errors")
print("=" * 60)

# Syndrome operators: X1X2 and X2X3
X1 = tensor(X, I, I)
X2 = tensor(I, X, I)
X3 = tensor(I, I, X)
X1X2 = tensor(X, X, I)
X2X3 = tensor(I, X, X)

# Z error operators
Z1 = tensor(Z, I, I)
Z2 = tensor(I, Z, I)
Z3 = tensor(I, I, Z)

def get_eigenvalue(state: np.ndarray, operator: np.ndarray) -> float:
    """Get eigenvalue of state under operator."""
    return np.real(state.conj() @ operator @ state)

def get_syndrome_pf(state: np.ndarray) -> Tuple[int, int]:
    """Get syndrome for phase-flip code (X1X2 and X2X3)."""
    e1 = get_eigenvalue(state, X1X2)
    e2 = get_eigenvalue(state, X2X3)
    s1 = 0 if e1 > 0 else 1
    s2 = 0 if e2 > 0 else 1
    return s1, s2

# Build syndrome table
print("\nSyndrome table for Z errors:")
print("-" * 45)
print(f"{'Error':<10} {'X1X2':<10} {'X2X3':<10} {'Syndrome':<12}")
print("-" * 45)

z_errors = {'I': np.eye(8, dtype=complex), 'Z1': Z1, 'Z2': Z2, 'Z3': Z3}

for name, E in z_errors.items():
    # Apply error to |0_L⟩
    error_state = E @ ket_0L_pf
    s1, s2 = get_syndrome_pf(error_state)
    e1 = get_eigenvalue(error_state, X1X2)
    e2 = get_eigenvalue(error_state, X2X3)
    print(f"{name:<10} {'+1' if e1>0 else '-1':<10} {'+1' if e2>0 else '-1':<10} ({s1}, {s2})")

# =============================================================================
# Part 5: Error Correction
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Error Correction")
print("=" * 60)

def correct_z_error(state: np.ndarray) -> np.ndarray:
    """Correct Z errors based on X-syndrome."""
    s1, s2 = get_syndrome_pf(state)

    correction_table = {
        (0, 0): np.eye(8, dtype=complex),
        (1, 0): Z1,
        (1, 1): Z2,
        (0, 1): Z3
    }

    return correction_table[(s1, s2)] @ state

def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute fidelity."""
    return np.abs(np.vdot(state1, state2))**2

# Test error correction
print("\nZ error correction:")
print("-" * 50)

# Encode a random state
theta, phi = np.pi/3, np.pi/4
psi_rand = np.cos(theta/2) * ket_0 + np.exp(1j*phi) * np.sin(theta/2) * ket_1
psi_rand_L = encode_phase_flip(psi_rand)

for name, E in z_errors.items():
    error_state = E @ psi_rand_L
    corrected = correct_z_error(error_state)
    fid = fidelity(corrected, psi_rand_L)
    syndrome = get_syndrome_pf(error_state)
    print(f"{name}: syndrome {syndrome}, fidelity after correction = {fid:.6f}")

# =============================================================================
# Part 6: X Errors Are Undetected
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: X Errors Go Undetected!")
print("=" * 60)

x_errors = {'X1': X1, 'X2': X2, 'X3': X3}

print("\nX error syndromes on phase-flip code:")
print("-" * 50)

# Use |0_L⟩ + |1_L⟩ superposition
psi_super_L = (ket_0L_pf + ket_1L_pf) / np.sqrt(2)

for name, E in x_errors.items():
    error_state = E @ psi_super_L
    syndrome = get_syndrome_pf(error_state)
    fid_original = fidelity(error_state, psi_super_L)
    print(f"{name}: syndrome {syndrome}, fidelity with original = {fid_original:.6f}")

print("\nX errors produce (0,0) syndrome but may not corrupt the state!")
print("(Because |+⟩, |-⟩ are X eigenstates)")

# Demonstrate X eigenstate property
print("\nX eigenstate verification:")
print(f"  X|+⟩ = |+⟩: {np.allclose(X @ ket_plus, ket_plus)}")
print(f"  X|-⟩ = |-⟩: {np.allclose(X @ ket_minus, ket_minus)}")

# =============================================================================
# Part 7: Duality with Bit-Flip Code
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Duality - Hadamard Transforms Codes")
print("=" * 60)

# Bit-flip codewords
ket_0L_bf = tensor(ket_0, ket_0, ket_0)  # |000⟩
ket_1L_bf = tensor(ket_1, ket_1, ket_1)  # |111⟩

# Apply H⊗3 to bit-flip codewords
H3 = tensor(H, H, H)

transformed_0L = H3 @ ket_0L_bf
transformed_1L = H3 @ ket_1L_bf

print("\nH⊗³|000⟩ = |+++⟩ ?")
print(f"  Match: {np.allclose(transformed_0L, ket_0L_pf)}")

print("\nH⊗³|111⟩ = |---⟩ ?")
print(f"  Match: {np.allclose(transformed_1L, ket_1L_pf)}")

# Stabilizer transformation
print("\nStabilizer transformation under H⊗³:")
Z1Z2 = tensor(Z, Z, I)
Z2Z3 = tensor(I, Z, Z)

print(f"  H⊗³(Z₁Z₂)H⊗³ = X₁X₂: {np.allclose(H3 @ Z1Z2 @ H3, X1X2)}")
print(f"  H⊗³(Z₂Z₃)H⊗³ = X₂X₃: {np.allclose(H3 @ Z2Z3 @ H3, X2X3)}")

# =============================================================================
# Part 8: Visualization
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Visualization")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Code comparison
ax1 = axes[0]
ax1.axis('off')

comparison = """
┌──────────────────────────────────────────────────┐
│          BIT-FLIP vs PHASE-FLIP DUALITY          │
├──────────────────────────────────────────────────┤
│                                                  │
│  BIT-FLIP CODE         PHASE-FLIP CODE           │
│  ─────────────         ───────────────           │
│  |0_L⟩ = |000⟩         |0_L⟩ = |+++⟩            │
│  |1_L⟩ = |111⟩         |1_L⟩ = |---⟩            │
│                                                  │
│  Stabilizers:          Stabilizers:              │
│    Z₁Z₂, Z₂Z₃           X₁X₂, X₂X₃              │
│                                                  │
│  Corrects: X errors    Corrects: Z errors        │
│                                                  │
│  Connected by: H⊗³                               │
│  HZH = X,  HXH = Z                               │
│                                                  │
└──────────────────────────────────────────────────┘
"""
ax1.text(0.05, 0.95, comparison, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
ax1.set_title('Code Duality')

# Plot 2: Syndrome tables
ax2 = axes[1]

# Create syndrome visualization for both codes
syndrome_bf = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # For X errors
syndrome_pf = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # For Z errors (same pattern!)

im = ax2.imshow(syndrome_bf, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['First\nStabilizer', 'Second\nStabilizer'])
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(['No error', 'Error on q1', 'Error on q2', 'Error on q3'])
ax2.set_title('Syndrome Pattern\n(Same for both codes!)')

for i in range(4):
    for j in range(2):
        ax2.text(j, i, f'{syndrome_bf[i,j]}', ha='center', va='center',
                color='white' if syndrome_bf[i,j]==1 else 'black', fontsize=14)

# Plot 3: Need for Shor code
ax3 = axes[2]
ax3.axis('off')

shor_motivation = """
┌──────────────────────────────────────────────────┐
│              WHY WE NEED THE SHOR CODE           │
├──────────────────────────────────────────────────┤
│                                                  │
│  PROBLEM:                                        │
│    Bit-flip code: ✗ Can't detect Z errors        │
│    Phase-flip code: ✗ Can't detect X errors      │
│                                                  │
│  Real errors include both X AND Z!               │
│  (Y = iXZ is a combination)                      │
│                                                  │
│  SOLUTION: CONCATENATION                         │
│    1. First encode against Z (phase-flip)        │
│    2. Then encode each qubit against X (bit-flip)│
│                                                  │
│  RESULT: Shor [[9,1,3]] code                     │
│    • 9 physical qubits                           │
│    • Corrects ANY single-qubit error             │
│    • Distance d = 3                              │
│                                                  │
│  Tomorrow: Building the Shor code!               │
│                                                  │
└──────────────────────────────────────────────────┘
"""
ax3.text(0.02, 0.95, shor_motivation, transform=ax3.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax3.set_title('Motivation for Shor Code')

plt.tight_layout()
plt.savefig('day_685_phase_flip_code.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: day_685_phase_flip_code.png")

# =============================================================================
# Part 9: Performance Comparison
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: Code Performance Comparison")
print("=" * 60)

# Simulate error correction under different noise models
n_trials = 1000
error_prob = 0.1

# Bit-flip code performance
bf_success_xnoise = 0
bf_success_znoise = 0
bf_success_depol = 0

# Phase-flip code performance
pf_success_xnoise = 0
pf_success_znoise = 0
pf_success_depol = 0

for _ in range(n_trials):
    # Random state
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    psi = np.cos(theta/2) * ket_0 + np.exp(1j*phi) * np.sin(theta/2) * ket_1

    # Encode in both codes
    # Bit-flip encoding
    psi_bf = tensor(psi, ket_0, ket_0)
    psi_bf = CNOT(0, 1, 3) @ psi_bf
    psi_bf = CNOT(0, 2, 3) @ psi_bf

    # Phase-flip encoding
    psi_pf = encode_phase_flip(psi)

    # X-only noise
    if np.random.random() < error_prob:
        qubit = np.random.choice([0, 1, 2])
        x_ops = [X1, X2, X3]
        psi_bf_x = x_ops[qubit] @ psi_bf
        psi_pf_x = x_ops[qubit] @ psi_pf
    else:
        psi_bf_x = psi_bf
        psi_pf_x = psi_pf

    # Z-only noise
    if np.random.random() < error_prob:
        qubit = np.random.choice([0, 1, 2])
        z_ops = [Z1, Z2, Z3]
        psi_bf_z = z_ops[qubit] @ psi_bf
        psi_pf_z = z_ops[qubit] @ psi_pf
    else:
        psi_bf_z = psi_bf
        psi_pf_z = psi_pf

    # Bit-flip correction (for X errors)
    def correct_bf(state):
        e1 = get_eigenvalue(state, Z1Z2)
        e2 = get_eigenvalue(state, Z2Z3)
        s1 = 0 if e1 > 0.5 else 1
        s2 = 0 if e2 > 0.5 else 1
        corr = {(0,0): np.eye(8), (1,0): X1, (1,1): X2, (0,1): X3}
        return corr[(s1, s2)] @ state

    # Check bit-flip code vs X noise
    if fidelity(correct_bf(psi_bf_x), psi_bf) > 0.99:
        bf_success_xnoise += 1

    # Check bit-flip code vs Z noise
    if fidelity(correct_bf(psi_bf_z), psi_bf) > 0.99:
        bf_success_znoise += 1

    # Check phase-flip code vs X noise
    if fidelity(correct_z_error(psi_pf_x), psi_pf) > 0.99:
        pf_success_xnoise += 1

    # Check phase-flip code vs Z noise
    if fidelity(correct_z_error(psi_pf_z), psi_pf) > 0.99:
        pf_success_znoise += 1

print(f"\nPerformance (p={error_prob}, {n_trials} trials):")
print("-" * 50)
print(f"{'Code':<15} {'X Noise':<15} {'Z Noise':<15}")
print("-" * 50)
print(f"{'Bit-flip':<15} {bf_success_xnoise/n_trials*100:.1f}%{' ✓':<10} {bf_success_znoise/n_trials*100:.1f}%")
print(f"{'Phase-flip':<15} {pf_success_xnoise/n_trials*100:.1f}%{'':<10} {pf_success_znoise/n_trials*100:.1f}% ✓")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

summary = """
┌────────────────────────────────────────────────────────────────────┐
│                Three-Qubit Phase-Flip Code [[3,1,1]]                │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ENCODING:                                                           │
│   |0_L⟩ = |+++⟩ = H⊗³|000⟩                                         │
│   |1_L⟩ = |---⟩ = H⊗³|111⟩                                         │
│                                                                     │
│ STABILIZERS: X₁X₂ and X₂X₃                                         │
│                                                                     │
│ SYNDROME TABLE:                                                     │
│   (0,0) → I,  (1,0) → Z₁,  (1,1) → Z₂,  (0,1) → Z₃                │
│                                                                     │
│ DUALITY:                                                            │
│   Phase-flip = H⊗³ (Bit-flip) H⊗³                                  │
│   Stabilizers transform: HZH = X                                    │
│                                                                     │
│ LIMITATIONS:                                                        │
│   ✗ Cannot detect X errors (|±⟩ are X eigenstates)                 │
│   ✗ Not a complete quantum code                                     │
│                                                                     │
│ NEXT: Shor code combines both → corrects all single-qubit errors   │
└────────────────────────────────────────────────────────────────────┘
"""
print(summary)

print("\n✅ Day 685 Lab Complete!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Logical states | $\|0_L\rangle = \|{+}{+}{+}\rangle$, $\|1_L\rangle = \|{-}{-}{-}\rangle$ |
| Stabilizers | $X_1X_2$, $X_2X_3$ |
| Logical Z | $Z_L = Z_1Z_2Z_3$ |
| Duality transformation | Phase-flip = $H^{\otimes 3}$(Bit-flip)$H^{\otimes 3}$ |

### Syndrome Table (Phase-Flip Code)

| Error | $X_1X_2$ | $X_2X_3$ | Correction |
|-------|----------|----------|------------|
| None | +1 | +1 | I |
| $Z_1$ | −1 | +1 | $Z_1$ |
| $Z_2$ | −1 | −1 | $Z_2$ |
| $Z_3$ | +1 | −1 | $Z_3$ |

### Main Takeaways

1. **Phase-flip code** protects against Z errors, not X errors
2. **Hadamard duality:** Phase-flip = Hadamard-transformed bit-flip
3. **X errors undetected** because $|+\rangle$, $|-\rangle$ are X eigenstates
4. **Neither code alone suffices** for general quantum errors
5. **Concatenation needed** → Shor's 9-qubit code

---

## Daily Checklist

- [ ] I can construct the phase-flip logical states
- [ ] I understand the Hadamard duality between codes
- [ ] I can compute syndromes for Z errors
- [ ] I know why X errors go undetected
- [ ] I understand the need for concatenation

---

## Preview: Day 686

Tomorrow we synthesize Week 98 and preview the Shor code:
- Week 98 comprehensive review
- Shor 9-qubit code construction
- Path to stabilizer formalism
- Preparing for Week 99: QEC theory

---

**Day 685 Complete!** Week 98: 6/7 days (86%)
