# Day 660: Quantum Error Correction Conditions

## Week 95: Error Detection/Correction Intro | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **State** the Knill-Laflamme quantum error correction conditions
2. **Understand** why QEC is fundamentally different from classical EC
3. **Apply** the conditions to verify correctability
4. **Connect** to physical error models

---

## Core Content

### 1. The Quantum Error Correction Problem

**Setup:**
- Code space $\mathcal{C} \subset \mathcal{H}$
- Code projector $P$ onto $\mathcal{C}$
- Error operators $\{E_a\}$ (Kraus operators of noise channel)

**Goal:** Recover the original encoded state after errors.

### 2. Three Key Quantum Challenges

**Challenge 1: No Cloning**
Can't simply copy the qubit multiple times.
**Solution:** Encode in entanglement across multiple qubits.

**Challenge 2: Measurement Destroys**
Can't measure to check for errors without disturbing the state.
**Solution:** Measure only the *syndrome* (error information, not data).

**Challenge 3: Continuous Errors**
Errors can be arbitrary rotations, not just bit flips.
**Solution:** Discretization—projective syndrome measurement collapses continuous errors to discrete set.

### 3. The Knill-Laflamme Conditions

**Theorem (Knill-Laflamme, 1997):** A code with projector $P$ can correct errors $\{E_a\}$ if and only if:

$$\boxed{PE_a^\dagger E_b P = \alpha_{ab} P}$$

for some Hermitian matrix $\alpha$.

**Equivalent form:** For orthonormal code basis $\{|i_L\rangle\}$:
$$\langle i_L|E_a^\dagger E_b|j_L\rangle = \alpha_{ab}\delta_{ij}$$

### 4. Interpretation

The condition says:
1. Errors take code states to orthogonal subspaces (different syndromes)
2. Within each error subspace, the code structure is preserved
3. Recovery operation: measure syndrome, apply correction

**Geometric picture:** Different errors "rotate" the code space to orthogonal positions, allowing identification.

### 5. Special Cases

**Detectable errors:** $PE_a^\dagger E_b P = 0$ for $a \neq b$ (errors distinguishable)

**Non-degenerate codes:** $\alpha_{ab} = 0$ for $a \neq b$ (errors perfectly distinguishable)

**Degenerate codes:** Some $\alpha_{ab} \neq 0$ for $a \neq b$ (different errors have same effect on code)

### 6. Discretization of Errors

**Key insight:** We don't need to correct ALL errors, just a discrete set!

If we can correct $\{E_a\}$, we can correct any linear combination:
$$E = \sum_a c_a E_a$$

**Why?** Syndrome measurement projects the error onto one of the $E_a$.

**Consequence:** Correcting Pauli errors X, Y, Z suffices for arbitrary single-qubit errors!

### 7. Error Correction Procedure

1. **Encode:** Map logical qubit to code space
2. **Error occurs:** $|\psi_L\rangle \to E_a|\psi_L\rangle$
3. **Syndrome measurement:** Project onto error subspaces
4. **Correction:** Apply $E_a^\dagger$ based on syndrome
5. **Result:** Recover original $|\psi_L\rangle$

### 8. The Recovery Map

The recovery operation is a CPTP map $\mathcal{R}$:
$$\mathcal{R}(\rho) = \sum_a R_a \rho R_a^\dagger$$

where $R_a = PE_a^\dagger M_a$ and $M_a$ is syndrome measurement.

**Condition for perfect recovery:**
$$\mathcal{R} \circ \mathcal{E}(\rho) = \rho \quad \forall \rho \in \mathcal{C}$$

---

## Worked Example

**Problem:** Verify the 3-bit repetition code corrects single X errors.

**Solution:**
Code: $|0_L\rangle = |000\rangle$, $|1_L\rangle = |111\rangle$
Errors: $\{I, X_1, X_2, X_3\}$

Check Knill-Laflamme condition $\langle i_L|E_a^\dagger E_b|j_L\rangle = \alpha_{ab}\delta_{ij}$:

For $E_a = E_b = I$:
- $\langle 0_L|I|0_L\rangle = 1 = \alpha_{II}$
- $\langle 0_L|I|1_L\rangle = 0$ ✓

For $E_a = X_1$, $E_b = X_1$:
- $\langle 0_L|X_1X_1|0_L\rangle = \langle 000|000\rangle = 1 = \alpha_{11}$
- $\langle 0_L|X_1X_1|1_L\rangle = \langle 000|111\rangle = 0$ ✓

For $E_a = I$, $E_b = X_1$:
- $\langle 0_L|X_1|0_L\rangle = \langle 000|100\rangle = 0 = \alpha_{I,1}$ ✓

All cross-terms vanish because errors produce orthogonal states!

---

## Practice Problems

1. Prove that a code cannot correct more than $\lfloor(n-1)/2\rfloor$ errors on $n$ qubits.
2. Verify Knill-Laflamme for the 3-qubit phase-flip code with single Z errors.
3. Show that no 2-qubit code can correct arbitrary single-qubit errors.
4. Explain why the condition $PE_a^\dagger E_b P \propto P$ allows degenerate codes.

---

## Computational Lab

```python
"""Day 660: Quantum Error Correction Conditions"""

import numpy as np
from itertools import product

def check_knill_laflamme(code_basis, errors, tol=1e-10):
    """
    Check Knill-Laflamme conditions for a code.

    code_basis: list of code state vectors [|0_L⟩, |1_L⟩, ...]
    errors: list of error operators [E_0, E_1, ...]

    Returns: (satisfies_conditions, alpha_matrix)
    """
    k = len(code_basis)  # Number of logical states
    n_errors = len(errors)

    # Build alpha matrix
    alpha = np.zeros((n_errors, n_errors), dtype=complex)

    for a, E_a in enumerate(errors):
        for b, E_b in enumerate(errors):
            # Check ⟨i_L|E_a† E_b|j_L⟩ = α_ab δ_ij
            values = np.zeros((k, k), dtype=complex)

            for i in range(k):
                for j in range(k):
                    values[i, j] = code_basis[i].conj().T @ E_a.conj().T @ E_b @ code_basis[j]

            # Check diagonal condition
            if not np.allclose(values - np.diag(np.diag(values)), 0, atol=tol):
                return False, None

            # Check all diagonal elements equal
            if not np.allclose(np.diag(values), values[0, 0], atol=tol):
                return False, None

            alpha[a, b] = values[0, 0]

    return True, alpha

# 3-qubit bit-flip code
ket_000 = np.array([[1], [0], [0], [0], [0], [0], [0], [0]], dtype=complex)
ket_111 = np.array([[0], [0], [0], [0], [0], [0], [0], [1]], dtype=complex)

# Error operators (8x8 for 3 qubits)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)

def tensor3(A, B, C):
    return np.kron(np.kron(A, B), C)

I_total = tensor3(I, I, I)
X1 = tensor3(X, I, I)
X2 = tensor3(I, X, I)
X3 = tensor3(I, I, X)

errors = [I_total, X1, X2, X3]
code_basis = [ket_000, ket_111]

satisfies, alpha = check_knill_laflamme(code_basis, errors)
print("3-Qubit Bit-Flip Code")
print(f"Satisfies Knill-Laflamme: {satisfies}")
if satisfies:
    print(f"Alpha matrix:\n{np.real(alpha)}")

# Test with two X errors (should fail)
X12 = tensor3(X, X, I)
errors_2 = [I_total, X1, X2, X3, X12]
satisfies_2, _ = check_knill_laflamme(code_basis, errors_2)
print(f"\nWith 2-qubit errors: {satisfies_2}")
```

---

## Summary

- **Knill-Laflamme theorem**: $PE_a^\dagger E_b P = \alpha_{ab}P$ is necessary and sufficient
- **Syndrome measurement**: Identifies error without revealing encoded information
- **Discretization**: Correcting discrete errors suffices for continuous errors
- **Key insight**: Different errors must take code to distinguishable states
- **Degenerate codes**: Multiple errors can have same effect (still correctable)

---

## Preview: Day 661

Tomorrow: **Three-Qubit Bit-Flip Code** - our first complete quantum error correcting code!
