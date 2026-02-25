# Day 661: Three-Qubit Bit-Flip Code

## Week 95: Error Detection/Correction Intro | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **Implement** the 3-qubit bit-flip code encoding
2. **Construct** syndrome measurement circuits
3. **Apply** correction based on syndrome
4. **Understand** why this code only corrects X errors

---

## Core Content

### 1. Code Definition

**Logical states:**
$$|0_L\rangle = |000\rangle, \quad |1_L\rangle = |111\rangle$$

**General encoded state:**
$$|\psi_L\rangle = \alpha|0_L\rangle + \beta|1_L\rangle = \alpha|000\rangle + \beta|111\rangle$$

**Note:** This is a GHZ-like entangled state!

### 2. Encoding Circuit

```
|ψ⟩ ─────●─────●───── |ψ_L⟩ (qubit 1)
         │     │
|0⟩ ─────⊕─────┼───── (qubit 2)
               │
|0⟩ ───────────⊕───── (qubit 3)
```

**Circuit:** Two CNOT gates with the data qubit as control.

### 3. Error Effects

| Error | $\|0_L\rangle$ becomes | $\|1_L\rangle$ becomes |
|-------|----------------------|----------------------|
| None | $\|000\rangle$ | $\|111\rangle$ |
| $X_1$ | $\|100\rangle$ | $\|011\rangle$ |
| $X_2$ | $\|010\rangle$ | $\|101\rangle$ |
| $X_3$ | $\|001\rangle$ | $\|110\rangle$ |

### 4. Syndrome Measurement

**Syndrome operators:**
$$Z_1Z_2 = Z \otimes Z \otimes I$$
$$Z_2Z_3 = I \otimes Z \otimes Z$$

These measure *parity* between adjacent qubits without measuring individual qubits!

**Syndrome table:**

| Error | $Z_1Z_2$ | $Z_2Z_3$ | Syndrome |
|-------|---------|---------|----------|
| None | +1 | +1 | (0, 0) |
| $X_1$ | -1 | +1 | (1, 0) |
| $X_2$ | -1 | -1 | (1, 1) |
| $X_3$ | +1 | -1 | (0, 1) |

### 5. Syndrome Measurement Circuit

```
|ψ_L⟩ ───●───────────── (qubit 1)
         │
─────────●───●───────── (qubit 2)
         │   │
─────────────●───●───── (qubit 3)
         │   │   │
|0⟩ ─────⊕───┼───┼───M─ (ancilla 1: Z₁Z₂)
             │   │
|0⟩ ─────────⊕───⊕───M─ (ancilla 2: Z₂Z₃)
```

### 6. Correction Procedure

Based on syndrome, apply correction:
- (0, 0): No error, do nothing
- (1, 0): $X_1$ error, apply $X$ to qubit 1
- (1, 1): $X_2$ error, apply $X$ to qubit 2
- (0, 1): $X_3$ error, apply $X$ to qubit 3

### 7. Why Only Bit-Flip?

The code **cannot correct phase errors**:
$$Z_1|0_L\rangle = |000\rangle = |0_L\rangle$$
$$Z_1|1_L\rangle = -|111\rangle = -|1_L\rangle$$

A $Z$ error on any qubit gives the same syndrome (0, 0)!

**Phase errors look like logical Z errors.**

### 8. Logical Operators

**Logical X:** $\bar{X} = X_1X_2X_3$ (flip all qubits)
$$\bar{X}|0_L\rangle = |1_L\rangle, \quad \bar{X}|1_L\rangle = |0_L\rangle$$

**Logical Z:** $\bar{Z} = Z_1 = Z_2 = Z_3$ (any single Z)
$$\bar{Z}|0_L\rangle = |0_L\rangle, \quad \bar{Z}|1_L\rangle = -|1_L\rangle$$

---

## Worked Example

**Problem:** An encoded state $|\psi_L\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$ experiences error $X_2$. Walk through the correction procedure.

**Solution:**

1. **After error:** $X_2|\psi_L\rangle = \frac{1}{\sqrt{2}}(|010\rangle + |101\rangle)$

2. **Syndrome measurement:**
   - $Z_1Z_2|010\rangle = (-1)(+1)|010\rangle = -|010\rangle$ → eigenvalue -1
   - $Z_2Z_3|010\rangle = (+1)(-1)|010\rangle = -|010\rangle$ → eigenvalue -1
   - Syndrome: (1, 1)

3. **Lookup:** (1, 1) indicates $X_2$ error

4. **Correction:** Apply $X_2$
   $$X_2 \cdot \frac{1}{\sqrt{2}}(|010\rangle + |101\rangle) = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle) = |\psi_L\rangle$$

5. **Result:** Original state recovered!

---

## Practice Problems

1. Show that the encoding circuit produces the correct logical states.
2. Verify that syndrome measurement doesn't reveal $\alpha$ or $\beta$.
3. What happens if errors $X_1$ and $X_2$ both occur?
4. Design a circuit that measures $Z_1Z_2$ using a single ancilla.

---

## Computational Lab

```python
"""Day 661: Three-Qubit Bit-Flip Code"""

import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

# Logical states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

ket_0L = tensor(ket_0, ket_0, ket_0)  # |000⟩
ket_1L = tensor(ket_1, ket_1, ket_1)  # |111⟩

# Error operators
X1 = tensor(X, I, I)
X2 = tensor(I, X, I)
X3 = tensor(I, I, X)
I8 = tensor(I, I, I)

# Syndrome operators
Z1Z2 = tensor(Z, Z, I)
Z2Z3 = tensor(I, Z, Z)

def get_syndrome(state):
    """Compute syndrome for a state."""
    s1 = np.real(state.conj().T @ Z1Z2 @ state)[0, 0]
    s2 = np.real(state.conj().T @ Z2Z3 @ state)[0, 0]
    return (0 if s1 > 0 else 1, 0 if s2 > 0 else 1)

def correct_error(state, syndrome):
    """Apply correction based on syndrome."""
    corrections = {
        (0, 0): I8,
        (1, 0): X1,
        (1, 1): X2,
        (0, 1): X3
    }
    return corrections[syndrome] @ state

# Test error correction
print("3-Qubit Bit-Flip Code Demo")
print("=" * 50)

# Encoded |+⟩ state
psi_L = (ket_0L + ket_1L) / np.sqrt(2)

errors = {'None': I8, 'X1': X1, 'X2': X2, 'X3': X3}

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

# Test with phase error (should fail)
print("\n--- Phase Error Test ---")
Z1 = tensor(Z, I, I)
psi_Z_error = Z1 @ psi_L
syndrome_Z = get_syndrome(psi_Z_error)
print(f"Z1 error: Syndrome {syndrome_Z} (looks like no error!)")
```

---

## Summary

- **3-qubit bit-flip code**: $|0_L\rangle = |000\rangle$, $|1_L\rangle = |111\rangle$
- **Syndrome operators**: $Z_1Z_2$ and $Z_2Z_3$ detect bit flips
- **Correction**: Apply $X$ to identified qubit
- **Limitation**: Cannot correct phase (Z) errors
- **Logical operators**: $\bar{X} = X_1X_2X_3$, $\bar{Z} = Z_i$

---

## Preview: Day 662

Tomorrow: **Three-Qubit Phase-Flip Code** - the dual code that corrects Z errors!
