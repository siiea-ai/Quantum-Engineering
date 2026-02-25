# Day 662: Three-Qubit Phase-Flip Code

## Week 95: Error Detection/Correction Intro | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **Implement** the 3-qubit phase-flip code encoding
2. **Understand** the Hadamard basis transformation
3. **Construct** syndrome measurements for phase errors
4. **Compare** with the bit-flip code structure
5. **Apply** duality between X and Z errors

---

## Core Content

### 1. The Phase Error Problem

Recall from Day 661: the bit-flip code **cannot** correct phase errors.

**The issue:**
$$Z|0_L\rangle = Z|000\rangle = |000\rangle = |0_L\rangle$$
$$Z|1_L\rangle = Z|111\rangle = -|111\rangle = -|1_L\rangle$$

A phase error looks like a **logical Z error**—undetectable!

**Solution:** Encode in the **Hadamard basis**.

### 2. Hadamard Basis Review

$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle), \quad |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

**Key property:** $Z|+\rangle = |-\rangle$ and $Z|-\rangle = |+\rangle$

Phase flips act like **bit flips in the Hadamard basis**!

### 3. Phase-Flip Code Definition

**Logical states:**
$$|0_L\rangle = |+++\rangle = \frac{1}{2\sqrt{2}}(|000\rangle + |001\rangle + |010\rangle + |011\rangle + |100\rangle + |101\rangle + |110\rangle + |111\rangle)$$

$$|1_L\rangle = |---\rangle = \frac{1}{2\sqrt{2}}(|000\rangle - |001\rangle - |010\rangle + |011\rangle - |100\rangle + |101\rangle + |110\rangle - |111\rangle)$$

**General encoded state:**
$$|\psi_L\rangle = \alpha|0_L\rangle + \beta|1_L\rangle = \alpha|+++\rangle + \beta|---\rangle$$

### 4. Encoding Circuit

```
|ψ⟩ ─────H─────●─────●───── |ψ_L⟩ (qubit 1)
               │     │
|0⟩ ─────H─────⊕─────┼───── (qubit 2)
                     │
|0⟩ ─────H───────────⊕───── (qubit 3)
```

**Steps:**
1. Apply Hadamard to all three qubits
2. Apply CNOTs (same as bit-flip code)

**Equivalently:** Apply bit-flip encoding, then Hadamard to each qubit.

### 5. Error Effects

| Error | $\|0_L\rangle = \|+++\rangle$ becomes | $\|1_L\rangle = \|---\rangle$ becomes |
|-------|--------------------------------------|--------------------------------------|
| None | $\|+++\rangle$ | $\|---\rangle$ |
| $Z_1$ | $\|-++\rangle$ | $\|+--\rangle$ |
| $Z_2$ | $\|+-+\rangle$ | $\|-+-\rangle$ |
| $Z_3$ | $\|++-\rangle$ | $\|--+\rangle$ |

### 6. Syndrome Measurement

**Syndrome operators:**
$$X_1X_2 = X \otimes X \otimes I$$
$$X_2X_3 = I \otimes X \otimes X$$

These are the **duals** of $Z_1Z_2$ and $Z_2Z_3$ from the bit-flip code!

**Syndrome table:**

| Error | $X_1X_2$ | $X_2X_3$ | Syndrome |
|-------|---------|---------|----------|
| None | +1 | +1 | (0, 0) |
| $Z_1$ | -1 | +1 | (1, 0) |
| $Z_2$ | -1 | -1 | (1, 1) |
| $Z_3$ | +1 | -1 | (0, 1) |

### 7. Why This Works

In the Hadamard basis:
- Z errors flip the $\pm$ sign of $|+\rangle$ and $|-\rangle$
- $X_iX_j$ measures parity of "phase flips" on qubits $i$ and $j$

**Mathematical reason:**
$$X_1X_2|+-+\rangle = X_1|+\rangle \otimes X_2|-\rangle \otimes |+\rangle$$
$$= |-\rangle \otimes |+\rangle \otimes |+\rangle = -|+-+\rangle$$

The eigenvalue is $-1$ because there's an odd number of $|-\rangle$ states in positions 1-2.

### 8. Correction Procedure

Based on syndrome, apply correction:
- (0, 0): No error, do nothing
- (1, 0): $Z_1$ error, apply $Z$ to qubit 1
- (1, 1): $Z_2$ error, apply $Z$ to qubit 2
- (0, 1): $Z_3$ error, apply $Z$ to qubit 3

### 9. Syndrome Measurement Circuit

```
|ψ_L⟩ ───●───────────── (qubit 1)
         │
─────────●───●───────── (qubit 2)
         │   │
─────────────●───●───── (qubit 3)
         │   │   │
|0⟩ ─H───⊕───┼───┼───H─M─ (ancilla 1: X₁X₂)
             │   │
|0⟩ ─H───────⊕───⊕───H─M─ (ancilla 2: X₂X₃)
```

**Note:** Hadamard gates on ancillas convert CNOT to measure X instead of Z.

### 10. Duality: Bit-Flip vs Phase-Flip

| Property | Bit-Flip Code | Phase-Flip Code |
|----------|--------------|-----------------|
| Logical $\|0_L\rangle$ | $\|000\rangle$ | $\|+++\rangle$ |
| Logical $\|1_L\rangle$ | $\|111\rangle$ | $\|---\rangle$ |
| Corrects | X errors | Z errors |
| Syndromes | $Z_iZ_j$ | $X_iX_j$ |
| Encoding | CNOTs | H + CNOTs + H |
| Fails on | Z errors | X errors |

**Key insight:** The codes are related by Hadamard transformation!

### 11. Why Can't We Correct Both?

Each 3-qubit code only corrects **one type** of error:
- Bit-flip code: 2D code space in 8D Hilbert space
- Phase-flip code: Different 2D code space

**Need more qubits** to correct both X and Z errors → Shor code (Day 663)!

### 12. Logical Operators

**Logical X:** $\bar{X} = Z_1Z_2Z_3$ (apply phase to all qubits—flips between $|+++\rangle$ and $|---\rangle$)

Wait, that's not right. Let's be careful:

**Logical X:** $\bar{X} = X_1 = X_2 = X_3$ (any single X flips the sign pattern)
$$\bar{X}|+++\rangle = |-++\rangle \neq |---\rangle$$

Actually, for this code:
$$\bar{X} = X_1X_2X_3$$
$$\bar{Z} = Z_1 = Z_2 = Z_3$$

This is dual to the bit-flip code's logical operators!

---

## Worked Example

**Problem:** An encoded state $|\psi_L\rangle = \frac{1}{\sqrt{2}}(|+++\rangle + |---\rangle)$ experiences error $Z_2$. Walk through the correction procedure.

**Solution:**

1. **Initial state:**
   $$|\psi_L\rangle = \frac{1}{\sqrt{2}}(|+++\rangle + |---\rangle)$$

2. **After $Z_2$ error:**
   $$Z_2|\psi_L\rangle = \frac{1}{\sqrt{2}}(Z_2|+++\rangle + Z_2|---\rangle)$$
   $$= \frac{1}{\sqrt{2}}(|+-+\rangle + |-+-\rangle)$$

3. **Syndrome measurement:**
   For the $|+-+\rangle$ component:
   - $X_1X_2|+-+\rangle = (-1)|+-+\rangle$ (qubit 1 is +, qubit 2 is -)
   - $X_2X_3|+-+\rangle = (-1)|+-+\rangle$ (qubit 2 is -, qubit 3 is +)

   Syndrome: (1, 1)

4. **Lookup:** (1, 1) indicates $Z_2$ error

5. **Correction:** Apply $Z_2$
   $$Z_2 \cdot \frac{1}{\sqrt{2}}(|+-+\rangle + |-+-\rangle) = \frac{1}{\sqrt{2}}(|+++\rangle + |---\rangle) = |\psi_L\rangle$$

6. **Result:** Original state recovered!

---

## Practice Problems

1. Show that the encoding circuit produces $|0\rangle \to |+++\rangle$ and $|1\rangle \to |---\rangle$.

2. Verify that $X_1X_2$ and $X_2X_3$ both commute with the logical states (stabilize the code space).

3. What happens if an X error occurs on qubit 1? Can it be detected?

4. Prove that $H^{\otimes 3}$ transforms the bit-flip code into the phase-flip code.

5. Design a circuit that decodes the phase-flip code (extracts the logical qubit).

---

## Computational Lab

```python
"""Day 662: Three-Qubit Phase-Flip Code"""

import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def tensor(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)
ket_plus = H @ ket_0
ket_minus = H @ ket_1

# Logical states for phase-flip code
ket_0L = tensor(ket_plus, ket_plus, ket_plus)   # |+++⟩
ket_1L = tensor(ket_minus, ket_minus, ket_minus) # |---⟩

print("Phase-Flip Code Demo")
print("=" * 50)

# Verify logical states
print("\n|0_L⟩ = |+++⟩:")
print(f"  Norm: {np.linalg.norm(ket_0L):.4f}")
print("\n|1_L⟩ = |---⟩:")
print(f"  Norm: {np.linalg.norm(ket_1L):.4f}")

# Check orthogonality
inner = ket_0L.conj().T @ ket_1L
print(f"\n⟨0_L|1_L⟩ = {inner[0,0]:.6f}")

# Error operators
Z1 = tensor(Z, I, I)
Z2 = tensor(I, Z, I)
Z3 = tensor(I, I, Z)
I8 = tensor(I, I, I)

# Syndrome operators
X1X2 = tensor(X, X, I)
X2X3 = tensor(I, X, X)

def get_syndrome(state):
    """Compute syndrome for phase-flip code."""
    s1 = np.real(state.conj().T @ X1X2 @ state)[0, 0]
    s2 = np.real(state.conj().T @ X2X3 @ state)[0, 0]
    return (0 if s1 > 0 else 1, 0 if s2 > 0 else 1)

def correct_error(state, syndrome):
    """Apply correction based on syndrome."""
    corrections = {
        (0, 0): I8,
        (1, 0): Z1,
        (1, 1): Z2,
        (0, 1): Z3
    }
    return corrections[syndrome] @ state

# Test error correction
print("\n" + "=" * 50)
print("Error Correction Test")
print("=" * 50)

# Encoded |+_L⟩ state (superposition of logical 0 and 1)
psi_L = (ket_0L + ket_1L) / np.sqrt(2)

errors = {'None': I8, 'Z1': Z1, 'Z2': Z2, 'Z3': Z3}

for name, E in errors.items():
    # Apply error
    psi_error = E @ psi_L

    # Get syndrome
    syndrome = get_syndrome(psi_error)

    # Correct
    psi_corrected = correct_error(psi_error, syndrome)

    # Check fidelity with original
    fidelity = np.abs(psi_L.conj().T @ psi_corrected)[0, 0]**2

    print(f"Error {name}: Syndrome {syndrome}, Fidelity after correction: {fidelity:.4f}")

# Test with bit-flip error (should fail)
print("\n--- Bit-Flip Error Test ---")
X1 = tensor(X, I, I)
psi_X_error = X1 @ psi_L
syndrome_X = get_syndrome(psi_X_error)
print(f"X1 error: Syndrome {syndrome_X} (looks like no error!)")

# Demonstrate duality with Hadamard
print("\n" + "=" * 50)
print("Duality Demonstration")
print("=" * 50)

# Bit-flip code states
bf_0L = tensor(ket_0, ket_0, ket_0)  # |000⟩
bf_1L = tensor(ket_1, ket_1, ket_1)  # |111⟩

# Apply H⊗3
H3 = tensor(H, H, H)
pf_0L_from_bf = H3 @ bf_0L
pf_1L_from_bf = H3 @ bf_1L

# Compare
print(f"H⊗3|000⟩ = |+++⟩? {np.allclose(pf_0L_from_bf, ket_0L)}")
print(f"H⊗3|111⟩ = |---⟩? {np.allclose(pf_1L_from_bf, ket_1L)}")

# Verify X and Z roles swap under Hadamard
print("\nHadamard conjugation:")
print(f"H Z H† = X? {np.allclose(H @ Z @ H.conj().T, X)}")
print(f"H X H† = Z? {np.allclose(H @ X @ H.conj().T, Z)}")
```

---

## Summary

| Concept | Phase-Flip Code |
|---------|-----------------|
| Logical states | $\|0_L\rangle = \|+++\rangle$, $\|1_L\rangle = \|---\rangle$ |
| Corrects | Single Z (phase) errors |
| Syndrome operators | $X_1X_2$, $X_2X_3$ |
| Cannot correct | X (bit-flip) errors |
| Duality | Hadamard-conjugate of bit-flip code |

**Key Insights:**
- Phase-flip code is the **Hadamard transform** of the bit-flip code
- X and Z errors swap roles under Hadamard
- Each 3-qubit code only corrects one error type
- Need concatenation (Shor code) to correct both

---

## Preview: Day 663

Tomorrow: **Nine-Qubit Shor Code** - concatenating bit-flip and phase-flip codes to correct ALL single-qubit errors!
