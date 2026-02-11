# Day 665: Week 95 Review - Error Detection and Correction

## Week 95: Error Detection/Correction Intro | Month 24: Quantum Channels & Error Introduction

---

## Week Summary

This week introduced the foundations of quantum error correction:

| Day | Topic | Key Insight |
|-----|-------|-------------|
| 659 | Classical EC Review | Parity checks, syndromes, redundancy |
| 660 | QEC Conditions | Knill-Laflamme theorem for correctability |
| 661 | Bit-Flip Code | Corrects X errors, fails on Z |
| 662 | Phase-Flip Code | Corrects Z errors, Hadamard dual |
| 663 | Shor Code | Concatenation corrects all single-qubit errors |
| 664 | Stabilizer Formalism | Systematic framework for QEC |

---

## Concept Map

```
Classical EC ─────────────────────────────────────────────────┐
     │                                                        │
     ├── Redundancy (repetition)                              │
     ├── Parity checks (syndrome)                             │
     └── Linear codes [n,k,d]                                 │
                                                              │
     ┌────────────────────────────────────────────────────────┘
     ▼
Quantum Challenges ───────────────────────────────────────────┐
     │                                                        │
     ├── No cloning → Encode in entanglement                  │
     ├── Measurement disturbs → Syndrome measurement          │
     └── Continuous errors → Discretization via projection    │
                                                              │
     ┌────────────────────────────────────────────────────────┘
     ▼
Knill-Laflamme Conditions
     │
     └── PE†a Eb P = αab P

3-Qubit Codes ────────────────────────────────────────────────┐
     │                                                        │
     ├── Bit-flip: |0L⟩=|000⟩, |1L⟩=|111⟩                     │
     │   └── Syndromes: Z₁Z₂, Z₂Z₃                            │
     │                                                        │
     └── Phase-flip: |0L⟩=|+++⟩, |1L⟩=|---⟩                   │
         └── Syndromes: X₁X₂, X₂X₃                            │
                                                              │
     ┌────────────────────────────────────────────────────────┘
     ▼
Shor [[9,1,3]] Code ──────────────────────────────────────────┐
     │                                                        │
     ├── Concatenation: phase-flip inside bit-flip            │
     ├── 8 syndrome bits                                      │
     └── Corrects any single-qubit error                      │
                                                              │
     ┌────────────────────────────────────────────────────────┘
     ▼
Stabilizer Formalism
     │
     ├── Pauli group Pn
     ├── Stabilizer group S ⊂ Pn (abelian, no -I)
     ├── Code space: +1 eigenspace of all stabilizers
     ├── Syndrome: commutation with stabilizers
     └── Logical operators: centralizer mod stabilizers
```

---

## Key Formulas

### Classical Error Correction

**Parity check:** $Hc = 0 \pmod{2}$

**Syndrome:** $s = He$

**Code parameters:** $[n, k, d]$ with $t = \lfloor(d-1)/2\rfloor$ correctable errors

### Knill-Laflamme Conditions

$$\boxed{PE_a^\dagger E_b P = \alpha_{ab} P}$$

Equivalently: $\langle i_L|E_a^\dagger E_b|j_L\rangle = \alpha_{ab}\delta_{ij}$

### 3-Qubit Bit-Flip Code

| State | Definition |
|-------|------------|
| $\|0_L\rangle$ | $\|000\rangle$ |
| $\|1_L\rangle$ | $\|111\rangle$ |
| Syndromes | $Z_1Z_2$, $Z_2Z_3$ |
| $\bar{X}$ | $X_1X_2X_3$ |
| $\bar{Z}$ | $Z_1 = Z_2 = Z_3$ |

### 3-Qubit Phase-Flip Code

| State | Definition |
|-------|------------|
| $\|0_L\rangle$ | $\|+++\rangle$ |
| $\|1_L\rangle$ | $\|---\rangle$ |
| Syndromes | $X_1X_2$, $X_2X_3$ |
| $\bar{X}$ | $X_1X_2X_3$ |
| $\bar{Z}$ | $Z_1 = Z_2 = Z_3$ |

### Shor [[9,1,3]] Code

$$|0_L\rangle = \frac{1}{2\sqrt{2}}\left(|000\rangle + |111\rangle\right)^{\otimes 3}$$

$$|1_L\rangle = \frac{1}{2\sqrt{2}}\left(|000\rangle - |111\rangle\right)^{\otimes 3}$$

### Stabilizer Code Parameters

$$[[n, k, d]] \text{ with } k = n - r$$

where $r$ = number of independent stabilizer generators

---

## Syndrome Tables Reference

### Bit-Flip Code

| Error | $Z_1Z_2$ | $Z_2Z_3$ | Syndrome |
|-------|---------|---------|----------|
| None | +1 | +1 | (0, 0) |
| $X_1$ | -1 | +1 | (1, 0) |
| $X_2$ | -1 | -1 | (1, 1) |
| $X_3$ | +1 | -1 | (0, 1) |

### Phase-Flip Code

| Error | $X_1X_2$ | $X_2X_3$ | Syndrome |
|-------|---------|---------|----------|
| None | +1 | +1 | (0, 0) |
| $Z_1$ | -1 | +1 | (1, 0) |
| $Z_2$ | -1 | -1 | (1, 1) |
| $Z_3$ | +1 | -1 | (0, 1) |

---

## Comprehensive Problems

### Problem 1: Knill-Laflamme Verification

Verify that the 3-qubit bit-flip code satisfies the Knill-Laflamme conditions for errors $\{I, X_1, X_2, X_3\}$.

**Solution:**
Need to check $\langle i_L|E_a^\dagger E_b|j_L\rangle = \alpha_{ab}\delta_{ij}$

For $E_a = E_b = X_1$:
- $\langle 0_L|X_1X_1|0_L\rangle = \langle 000|000\rangle = 1$
- $\langle 1_L|X_1X_1|1_L\rangle = \langle 111|111\rangle = 1$
- $\langle 0_L|X_1X_1|1_L\rangle = \langle 000|111\rangle = 0$ ✓

For $E_a = I, E_b = X_1$:
- $\langle 0_L|X_1|0_L\rangle = \langle 000|100\rangle = 0$
- $\langle 0_L|X_1|1_L\rangle = \langle 000|011\rangle = 0$

All conditions satisfied with $\alpha = I$ (identity matrix).

### Problem 2: Code Comparison

Compare the bit-flip, phase-flip, and Shor codes:

| Property | Bit-Flip | Phase-Flip | Shor |
|----------|---------|-----------|------|
| Qubits | 3 | 3 | 9 |
| Parameters | [[3,1,1]] | [[3,1,1]] | [[9,1,3]] |
| Corrects X | ✓ | ✗ | ✓ |
| Corrects Z | ✗ | ✓ | ✓ |
| Corrects Y | ✗ | ✗ | ✓ |
| Stabilizers | 2 | 2 | 8 |
| Rate | 1/3 | 1/3 | 1/9 |

Note: The 3-qubit codes have distance 1 for the error type they can't correct!

### Problem 3: Stabilizer Calculation

Given stabilizers $S_1 = XZZXI$ and $S_2 = IXZZX$ for a 5-qubit code:

a) What is $S_1 \cdot S_2$?
b) Do these stabilizers commute?
c) How many logical qubits does this code have?

**Solution:**
a) $S_1 \cdot S_2 = (XZZXI)(IXZZX) = ?$

Position by position:
- Position 1: $X \cdot I = X$
- Position 2: $Z \cdot X = iY$...

Wait, we need to track the phase. For stabilizer codes we typically work with $\pm 1$ phases:
$ZX = iY$ but in stabilizer formalism $ZX = -XZ$, so:
$S_1 S_2 = XYZYX$ (up to sign)

b) To check commutation, count positions where both have non-identity and anticommute:
- Position 2: $Z$ and $X$ → anticommute
- Position 4: $X$ and $Z$ → anticommute

Two anticommuting pairs → overall commute ✓

c) With $n=5$ qubits and $r=2$ generators: $k = 5-2 = 3$ logical qubits.

### Problem 4: Error Propagation

An X error occurs on qubit 2 of a bit-flip encoded state, but before correction, a CNOT is applied with qubit 2 as control and qubit 3 as target. What is the resulting error?

**Solution:**
- Initial error: $X_2$
- CNOT propagates X on control to target: $CNOT_{2\to 3} \cdot X_2 = X_2 X_3 \cdot CNOT_{2\to 3}$
- Final error: $X_2 X_3$ (two-qubit error, now uncorrectable!)

This illustrates why **fault tolerance** is critical.

### Problem 5: Decoding Circuit

Design a circuit to decode the 3-qubit bit-flip code (extract the logical qubit).

**Solution:**
Reverse the encoding:
```
(qubit 1) ───────●─────●───── |ψ⟩
                 │     │
(qubit 2) ───────⊕─────┼─────  (should be |0⟩)
                       │
(qubit 3) ─────────────⊕─────  (should be |0⟩)
```

After this, qubit 1 holds $|\psi\rangle$ and qubits 2,3 are in $|0\rangle$.

---

## Self-Assessment Checklist

### Conceptual Understanding
- [ ] Can I explain why quantum error correction is possible despite no-cloning?
- [ ] Do I understand the role of syndrome measurement?
- [ ] Can I state the Knill-Laflamme conditions?
- [ ] Do I understand why discretization works?

### Code Knowledge
- [ ] Can I construct the bit-flip and phase-flip code states?
- [ ] Can I write the syndrome operators for each code?
- [ ] Do I understand the Shor code structure?
- [ ] Can I identify stabilizer generators?

### Computational Skills
- [ ] Can I calculate syndromes for given errors?
- [ ] Can I apply corrections based on syndromes?
- [ ] Can I verify the Knill-Laflamme conditions?
- [ ] Can I work with stabilizer operators?

---

## Computational Lab

```python
"""Day 665: Week 95 Comprehensive Review"""

import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

print("Week 95 Review: Quantum Error Correction")
print("=" * 60)

# ============================================
# Part 1: Compare all three codes
# ============================================
print("\n" + "=" * 60)
print("PART 1: Code Comparison")
print("=" * 60)

# Bit-flip code
bf_0L = tensor(ket_0, ket_0, ket_0)
bf_1L = tensor(ket_1, ket_1, ket_1)

# Phase-flip code
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
ket_plus = H @ ket_0
ket_minus = H @ ket_1
pf_0L = tensor(ket_plus, ket_plus, ket_plus)
pf_1L = tensor(ket_minus, ket_minus, ket_minus)

# Verify orthonormality
print("\nBit-flip code:")
print(f"  ⟨0L|0L⟩ = {np.real((bf_0L.conj().T @ bf_0L)[0,0]):.4f}")
print(f"  ⟨0L|1L⟩ = {np.abs((bf_0L.conj().T @ bf_1L)[0,0]):.4f}")

print("\nPhase-flip code:")
print(f"  ⟨0L|0L⟩ = {np.real((pf_0L.conj().T @ pf_0L)[0,0]):.4f}")
print(f"  ⟨0L|1L⟩ = {np.abs((pf_0L.conj().T @ pf_1L)[0,0]):.4f}")

# ============================================
# Part 2: Syndrome comparison
# ============================================
print("\n" + "=" * 60)
print("PART 2: Syndrome Comparison")
print("=" * 60)

# Bit-flip syndromes
Z1Z2_bf = tensor(Z, Z, I)
Z2Z3_bf = tensor(I, Z, Z)

# Phase-flip syndromes
X1X2_pf = tensor(X, X, I)
X2X3_pf = tensor(I, X, X)

# Errors
X1 = tensor(X, I, I)
X2 = tensor(I, X, I)
Z1 = tensor(Z, I, I)
Z2 = tensor(I, Z, I)

def get_eigenvalue(state, operator):
    """Get eigenvalue of state under operator."""
    return np.real((state.conj().T @ operator @ state)[0, 0])

# Test superposition states
bf_plus_L = (bf_0L + bf_1L) / np.sqrt(2)
pf_plus_L = (pf_0L + pf_1L) / np.sqrt(2)

print("\nBit-flip code response to errors:")
for name, E in [("None", tensor(I,I,I)), ("X1", X1), ("X2", X2), ("Z1", Z1)]:
    state = E @ bf_plus_L
    s1 = 0 if get_eigenvalue(state, Z1Z2_bf) > 0 else 1
    s2 = 0 if get_eigenvalue(state, Z2Z3_bf) > 0 else 1
    print(f"  {name}: syndrome ({s1}, {s2})")

print("\nPhase-flip code response to errors:")
for name, E in [("None", tensor(I,I,I)), ("Z1", Z1), ("Z2", Z2), ("X1", X1)]:
    state = E @ pf_plus_L
    s1 = 0 if get_eigenvalue(state, X1X2_pf) > 0 else 1
    s2 = 0 if get_eigenvalue(state, X2X3_pf) > 0 else 1
    print(f"  {name}: syndrome ({s1}, {s2})")

# ============================================
# Part 3: Knill-Laflamme verification
# ============================================
print("\n" + "=" * 60)
print("PART 3: Knill-Laflamme Conditions")
print("=" * 60)

def check_knill_laflamme(code_states, errors, error_names):
    """Check Knill-Laflamme conditions."""
    ket_0L, ket_1L = code_states
    n_errors = len(errors)

    print("\nMatrix elements ⟨iL|Ea†Eb|jL⟩:")

    all_good = True
    for a, (Ea, name_a) in enumerate(zip(errors, error_names)):
        for b, (Eb, name_b) in enumerate(zip(errors, error_names)):
            # Compute all four matrix elements
            m00 = (ket_0L.conj().T @ Ea.conj().T @ Eb @ ket_0L)[0, 0]
            m11 = (ket_1L.conj().T @ Ea.conj().T @ Eb @ ket_1L)[0, 0]
            m01 = (ket_0L.conj().T @ Ea.conj().T @ Eb @ ket_1L)[0, 0]
            m10 = (ket_1L.conj().T @ Ea.conj().T @ Eb @ ket_0L)[0, 0]

            # Check conditions
            diag_equal = np.isclose(m00, m11)
            off_diag_zero = np.isclose(m01, 0) and np.isclose(m10, 0)

            if not (diag_equal and off_diag_zero):
                all_good = False
                print(f"  {name_a}†{name_b}: FAILS")
            elif a <= b:  # Only print upper triangle
                print(f"  {name_a}†{name_b}: α = {m00:.2f}")

    return all_good

I3 = tensor(I, I, I)
X1 = tensor(X, I, I)
X2 = tensor(I, X, I)
X3 = tensor(I, I, X)

errors_bf = [I3, X1, X2, X3]
names_bf = ['I', 'X1', 'X2', 'X3']

print("\nBit-flip code with X errors:")
result = check_knill_laflamme([bf_0L, bf_1L], errors_bf, names_bf)
print(f"Satisfies Knill-Laflamme: {result}")

# ============================================
# Part 4: Error correction simulation
# ============================================
print("\n" + "=" * 60)
print("PART 4: Error Correction Simulation")
print("=" * 60)

def simulate_error_correction(code_name, code_states, syndromes,
                               corrections, test_errors, error_names):
    """Simulate full error correction cycle."""
    ket_0L, ket_1L = code_states
    psi_L = (ket_0L + ket_1L) / np.sqrt(2)  # |+_L⟩

    print(f"\n{code_name}:")
    print("Error | Syndrome | Fidelity after correction")
    print("-" * 45)

    for E, name in zip(test_errors, error_names):
        # Apply error
        psi_err = E @ psi_L

        # Measure syndrome
        syn = []
        for S in syndromes:
            ev = np.real((psi_err.conj().T @ S @ psi_err)[0, 0])
            syn.append(0 if ev > 0 else 1)
        syn = tuple(syn)

        # Apply correction
        C = corrections.get(syn, tensor(I, I, I))
        psi_corr = C @ psi_err

        # Calculate fidelity
        fidelity = np.abs((psi_L.conj().T @ psi_corr)[0, 0])**2

        print(f"{name:6s}| {syn}    | {fidelity:.4f}")

# Bit-flip code
bf_corrections = {
    (0, 0): tensor(I, I, I),
    (1, 0): tensor(X, I, I),
    (1, 1): tensor(I, X, I),
    (0, 1): tensor(I, I, X)
}

simulate_error_correction(
    "Bit-flip code",
    [bf_0L, bf_1L],
    [Z1Z2_bf, Z2Z3_bf],
    bf_corrections,
    [tensor(I,I,I), X1, X2, X3, Z1],
    ['None', 'X1', 'X2', 'X3', 'Z1']
)

# Phase-flip code
pf_corrections = {
    (0, 0): tensor(I, I, I),
    (1, 0): tensor(Z, I, I),
    (1, 1): tensor(I, Z, I),
    (0, 1): tensor(I, I, Z)
}

simulate_error_correction(
    "Phase-flip code",
    [pf_0L, pf_1L],
    [X1X2_pf, X2X3_pf],
    pf_corrections,
    [tensor(I,I,I), Z1, Z2, tensor(I,I,Z), X1],
    ['None', 'Z1', 'Z2', 'Z3', 'X1']
)

print("\n" + "=" * 60)
print("Week 95 Review Complete!")
print("=" * 60)
```

---

## Looking Ahead: Week 96

Next week is the **Semester 1B Review**:

| Day | Topic |
|-----|-------|
| 666 | Month 19-20 Review (Density Matrices & Entanglement) |
| 667 | Month 21 Review (Open Systems) |
| 668 | Month 22 Review (Quantum Algorithms I) |
| 669 | Month 23 Review (Quantum Channels) |
| 670 | Month 24 Review (Error Channels & Correction) |
| 671 | Comprehensive Problems |
| 672 | Year 1 Semester 1B Complete |

---

## Key Takeaways

1. **Quantum error correction is possible** despite no-cloning and measurement disturbance
2. **Syndrome measurement** extracts error information without revealing encoded data
3. **Discretization** means correcting Pauli errors suffices for all errors
4. **Concatenation** combines codes to correct more error types
5. **Stabilizer formalism** provides a systematic framework for code construction
6. **Code parameters** $[[n, k, d]]$ characterize the code's capabilities

---

**Week 95 Complete!**

You now understand the foundations of quantum error correction. The stabilizer formalism previewed this week will be developed fully in Year 2, leading to advanced topics like CSS codes, the surface code, and fault-tolerant quantum computing.
