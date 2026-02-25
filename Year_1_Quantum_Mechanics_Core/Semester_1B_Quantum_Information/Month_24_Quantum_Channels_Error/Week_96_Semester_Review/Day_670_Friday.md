# Day 670: Month 24 Review - Error Channels and Quantum Error Correction

## Week 96: Semester 1B Review | Month 24: Quantum Channels & Error Introduction

---

## Review Scope

**Month 24: Quantum Channels & Error Introduction (Days 645-672)**
- Week 93: Channel Representations (Kraus, Choi, Stinespring)
- Week 94: Quantum Error Types (Bit-flip, Phase-flip, Depolarizing, Amplitude Damping)
- Week 95: Error Detection/Correction Introduction

---

## Core Concepts: Error Channels

### 1. Error Channel Classification

| Channel | Kraus Operators | Effect |
|---------|-----------------|--------|
| Bit-flip | $K_0 = \sqrt{1-p}I$, $K_1 = \sqrt{p}X$ | Flips $\|0\rangle \leftrightarrow \|1\rangle$ |
| Phase-flip | $K_0 = \sqrt{1-p}I$, $K_1 = \sqrt{p}Z$ | Applies relative phase |
| Bit-phase flip | $K_0 = \sqrt{1-p}I$, $K_1 = \sqrt{p}Y$ | Both X and Z |
| Depolarizing | Symmetric Pauli | Contracts Bloch sphere |
| Amplitude damping | Energy decay | Non-unital, T1 process |

### 2. Pauli Channel General Form

$$\mathcal{E}(\rho) = (1-p_x-p_y-p_z)\rho + p_x X\rho X + p_y Y\rho Y + p_z Z\rho Z$$

**Bloch sphere effect:**
$$\vec{r} \to \begin{pmatrix}1-2(p_y+p_z) & 0 & 0\\0 & 1-2(p_x+p_z) & 0\\0 & 0 & 1-2(p_x+p_y)\end{pmatrix}\vec{r}$$

### 3. Depolarizing Channel

$$\mathcal{E}_{dep}(\rho) = (1-p)\rho + p\frac{I}{2} = \left(1-\frac{3p}{4}\right)\rho + \frac{p}{4}(X\rho X + Y\rho Y + Z\rho Z)$$

**Properties:**
- Isotropic contraction: $\vec{r} \to (1-p)\vec{r}$
- Unital: $\mathcal{E}(I) = I$
- "Worst-case" noise model

### 4. Amplitude Damping

$$K_0 = \begin{pmatrix}1 & 0\\0 & \sqrt{1-\gamma}\end{pmatrix}, \quad K_1 = \begin{pmatrix}0 & \sqrt{\gamma}\\0 & 0\end{pmatrix}$$

**Properties:**
- Non-unital: $\mathcal{E}(I) \neq I$
- Fixed point: $|0\rangle$
- Models T1 relaxation: $\gamma = 1 - e^{-t/T_1}$

### 5. T1/T2 Parameters

$$\boxed{T_2 \leq 2T_1}$$

| Process | Rate | Channel |
|---------|------|---------|
| T1 (relaxation) | $\gamma = 1/T_1$ | Amplitude damping |
| T2 (dephasing) | $\gamma_\phi$ | Phase damping |
| Pure dephasing | $1/T_\phi$ | $1/T_2 = 1/(2T_1) + 1/T_\phi$ |

---

## Core Concepts: Error Correction

### 6. Classical vs Quantum EC

| Challenge | Classical | Quantum Solution |
|-----------|-----------|------------------|
| Copy data | Repeat bits | Entanglement encoding |
| Check errors | Measure directly | Syndrome measurement |
| Error types | Bit flip only | X, Y, Z, continuous |
| Discretization | N/A | Pauli projection |

### 7. Knill-Laflamme Conditions

A code with projector $P$ corrects errors $\{E_a\}$ iff:
$$\boxed{PE_a^\dagger E_b P = \alpha_{ab} P}$$

**Equivalent:** $\langle i_L|E_a^\dagger E_b|j_L\rangle = \alpha_{ab}\delta_{ij}$

**Interpretation:**
- Different errors → distinguishable syndromes
- Same error effect on all logical states

### 8. Three-Qubit Bit-Flip Code

**Encoding:**
$$|0_L\rangle = |000\rangle, \quad |1_L\rangle = |111\rangle$$

**Syndromes:** $Z_1Z_2$, $Z_2Z_3$

| Error | Syndrome | Correction |
|-------|----------|------------|
| None | (0,0) | None |
| $X_1$ | (1,0) | $X_1$ |
| $X_2$ | (1,1) | $X_2$ |
| $X_3$ | (0,1) | $X_3$ |

**Limitation:** Cannot correct Z errors

### 9. Three-Qubit Phase-Flip Code

**Encoding:**
$$|0_L\rangle = |+++\rangle, \quad |1_L\rangle = |---\rangle$$

**Syndromes:** $X_1X_2$, $X_2X_3$

**Duality:** Hadamard transform of bit-flip code

**Limitation:** Cannot correct X errors

### 10. Shor [[9,1,3]] Code

**Concatenation:** Phase-flip code over bit-flip code

$$|0_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle + |111\rangle)^{\otimes 3}$$
$$|1_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle - |111\rangle)^{\otimes 3}$$

**Properties:**
- Corrects ANY single-qubit error
- 8 syndrome bits (6 bit-flip + 2 phase-flip)
- First complete QEC code (Shor, 1995)

### 11. Stabilizer Formalism Preview

**Stabilizer group:** Abelian subgroup $\mathcal{S} \subset \mathcal{P}_n$

**Code space:** Simultaneous +1 eigenspace of all stabilizers

**Parameters:** $[[n, k, d]]$ where $k = n - r$ (r = number of generators)

**Syndrome:** Commutation pattern with stabilizers

---

## Integration: The QEC Pipeline

```
Physical Noise (T1, T2, gate errors)
            │
            ▼
Error Channels (Kraus representation)
            │
            ▼
Error Discretization (Pauli basis)
            │
            ▼
Syndrome Measurement
            │
            ▼
Correction Operation
            │
            ▼
Recovered Quantum State
```

---

## Practice Problems

### Problem 1: Error Identification

A qubit prepared in $|+\rangle$ is measured after some noise and found to have density matrix:
$$\rho = \begin{pmatrix}0.5 & 0.4\\0.4 & 0.5\end{pmatrix}$$

What type of noise occurred? Estimate the parameter.

**Solution:**
For $|+\rangle$: $\rho_0 = \begin{pmatrix}0.5 & 0.5\\0.5 & 0.5\end{pmatrix}$

Coherence decayed: $0.5 \to 0.4$ (factor of 0.8)

This is consistent with:
- Dephasing with $(1-2p) = 0.8 \Rightarrow p = 0.1$
- Or amplitude damping with $\sqrt{1-\gamma} = 0.8 \Rightarrow \gamma = 0.36$

Population unchanged → pure dephasing (p = 0.1)

### Problem 2: Syndrome Calculation

For the bit-flip code, calculate the syndrome when error $X_1X_2$ occurs.

**Solution:**
Syndromes: $Z_1Z_2$, $Z_2Z_3$

For $X_1X_2$:
- $Z_1Z_2$ anti-commutes with $X_1$ and $X_2$ → commutes overall → syndrome 0
- $Z_2Z_3$ anti-commutes with $X_2$ only → anti-commutes → syndrome 1

Syndrome: (0, 1) → Interpreted as $X_3$ error!

**Conclusion:** Two-qubit error misdiagnosed as single-qubit → correction fails.

### Problem 3: Channel Composition

A qubit experiences amplitude damping ($\gamma = 0.1$) followed by dephasing ($p = 0.05$). Find the effective coherence decay.

**Solution:**
After amplitude damping: coherence $\to \sqrt{1-\gamma} = \sqrt{0.9} \approx 0.949$
After dephasing: coherence $\to (1-2p) = 0.9$

Total: $0.949 \times 0.9 \approx 0.854$

Compare to pure T2 process with same final coherence.

### Problem 4: Code Distance

The Shor code has $d = 3$. How many errors can it:
a) Detect?
b) Correct?

**Solution:**
a) Can detect up to $d-1 = 2$ errors
b) Can correct up to $\lfloor(d-1)/2\rfloor = 1$ error

---

## Computational Lab

```python
"""Day 670: Month 24 Review - Error Channels and QEC"""

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

print("Month 24 Review: Error Channels and QEC")
print("=" * 60)

# ============================================
# Part 1: Error Channel Comparison
# ============================================
print("\nPART 1: Error Channel Comparison")
print("-" * 40)

def apply_channel(rho, kraus_ops):
    result = np.zeros_like(rho)
    for K in kraus_ops:
        result += K @ rho @ K.conj().T
    return result

# Initial state |+⟩
rho_plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)

# Error channels
p = 0.1
gamma = 0.1

channels = {
    "Bit-flip": [np.sqrt(1-p) * I, np.sqrt(p) * X],
    "Phase-flip": [np.sqrt(1-p) * I, np.sqrt(p) * Z],
    "Depolarizing": [np.sqrt(1-3*p/4) * I, np.sqrt(p/4) * X,
                     np.sqrt(p/4) * Y, np.sqrt(p/4) * Z],
    "Amp. damping": [np.array([[1, 0], [0, np.sqrt(1-gamma)]]),
                     np.array([[0, np.sqrt(gamma)], [0, 0]])]
}

print(f"Initial state |+⟩, error parameter = {p}")
print(f"\n{'Channel':<15} {'Purity':>10} {'Coherence':>12} {'P(|1⟩)':>10}")
print("-" * 50)

for name, K in channels.items():
    rho_out = apply_channel(rho_plus, K)
    purity = np.real(np.trace(rho_out @ rho_out))
    coherence = np.abs(rho_out[0, 1])
    p1 = np.real(rho_out[1, 1])
    print(f"{name:<15} {purity:>10.4f} {coherence:>12.4f} {p1:>10.4f}")

# ============================================
# Part 2: Error Correction Simulation
# ============================================
print("\n" + "=" * 60)
print("PART 2: Error Correction Simulation")
print("-" * 40)

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Bit-flip code logical states
ket_0L = tensor(ket_0, ket_0, ket_0)
ket_1L = tensor(ket_1, ket_1, ket_1)

# Syndrome operators
Z1Z2 = tensor(Z, Z, I)
Z2Z3 = tensor(I, Z, Z)

def get_syndrome(state, syndromes):
    """Get syndrome bits."""
    result = []
    for S in syndromes:
        ev = np.real((state.conj().T @ S @ state)[0, 0])
        result.append(0 if ev > 0 else 1)
    return tuple(result)

def correct_bitflip(state, syndrome):
    """Apply correction for bit-flip code."""
    corrections = {
        (0, 0): tensor(I, I, I),
        (1, 0): tensor(X, I, I),
        (1, 1): tensor(I, X, I),
        (0, 1): tensor(I, I, X)
    }
    return corrections[syndrome] @ state

# Encoded |+_L⟩
psi_L = (ket_0L + ket_1L) / np.sqrt(2)

# Test single-qubit errors
errors = {
    "None": tensor(I, I, I),
    "X1": tensor(X, I, I),
    "X2": tensor(I, X, I),
    "X3": tensor(I, I, X),
    "Z1": tensor(Z, I, I),
    "X1X2": tensor(X, X, I)
}

print("Bit-flip code error correction:")
print(f"{'Error':<8} {'Syndrome':<12} {'Corrected':>12} {'Fidelity':>10}")
print("-" * 45)

for name, E in errors.items():
    psi_err = E @ psi_L
    syn = get_syndrome(psi_err, [Z1Z2, Z2Z3])
    psi_corr = correct_bitflip(psi_err, syn)
    fidelity = np.abs((psi_L.conj().T @ psi_corr)[0, 0])**2
    success = "Yes" if fidelity > 0.99 else "No"
    print(f"{name:<8} {str(syn):<12} {success:>12} {fidelity:>10.4f}")

# ============================================
# Part 3: Knill-Laflamme Verification
# ============================================
print("\n" + "=" * 60)
print("PART 3: Knill-Laflamme Verification")
print("-" * 40)

errors_to_check = [tensor(I, I, I), tensor(X, I, I), tensor(I, X, I), tensor(I, I, X)]
error_names = ["I", "X1", "X2", "X3"]

print("Matrix ⟨0L|Ea†Eb|0L⟩:")
print("    ", end="")
for name in error_names:
    print(f"{name:>8}", end="")
print()

for i, (Ea, name_a) in enumerate(zip(errors_to_check, error_names)):
    print(f"{name_a:>4}", end="")
    for j, (Eb, name_b) in enumerate(zip(errors_to_check, error_names)):
        val = (ket_0L.conj().T @ Ea.conj().T @ Eb @ ket_0L)[0, 0]
        print(f"{np.real(val):>8.2f}", end="")
    print()

print("\nMatrix ⟨0L|Ea†Eb|1L⟩ (should be all zeros):")
print("    ", end="")
for name in error_names:
    print(f"{name:>8}", end="")
print()

for i, (Ea, name_a) in enumerate(zip(errors_to_check, error_names)):
    print(f"{name_a:>4}", end="")
    for j, (Eb, name_b) in enumerate(zip(errors_to_check, error_names)):
        val = (ket_0L.conj().T @ Ea.conj().T @ Eb @ ket_1L)[0, 0]
        print(f"{np.abs(val):>8.2f}", end="")
    print()

print("\nKnill-Laflamme satisfied: errors produce orthogonal subspaces ✓")

# ============================================
# Part 4: Circuit Error Accumulation
# ============================================
print("\n" + "=" * 60)
print("PART 4: Circuit Error Accumulation")
print("-" * 40)

def circuit_fidelity(n_gates, gate_error):
    """Estimate circuit fidelity with independent gate errors."""
    return (1 - gate_error) ** n_gates

gate_errors = [0.001, 0.01, 0.1]
depths = [10, 100, 1000]

print(f"{'Depth':<10}", end="")
for err in gate_errors:
    print(f"ε={err:<8}", end="")
print()
print("-" * 40)

for d in depths:
    print(f"{d:<10}", end="")
    for err in gate_errors:
        fid = circuit_fidelity(d, err)
        print(f"{fid:<10.4f}", end="")
    print()

print("\nConclusion: Need error correction for deep circuits!")

# ============================================
# Part 5: Threshold Estimate
# ============================================
print("\n" + "=" * 60)
print("PART 5: Error Threshold Concept")
print("-" * 40)

print("""
Error Correction Threshold:
- If physical error rate ε < ε_th, arbitrary accuracy achievable
- Typical threshold: ε_th ~ 1% for surface code
- Overhead: O(log(1/ε_target)) code distance

Current technology:
- Superconducting qubits: ε ~ 0.1-1%
- Trapped ions: ε ~ 0.01-0.1%
- At or near threshold → Error correction essential!
""")

print("=" * 60)
print("Review Complete!")
```

---

## Summary

### Error Channels

| Channel | Corrects | Effect |
|---------|----------|--------|
| Bit-flip | With bit-flip code | X errors |
| Phase-flip | With phase-flip code | Z errors |
| Depolarizing | With Shor/CSS codes | All Pauli |
| Amplitude damping | Requires special codes | T1 decay |

### Error Correction Codes

| Code | Parameters | Corrects |
|------|------------|----------|
| Bit-flip | [[3,1,1]] for X | Single X |
| Phase-flip | [[3,1,1]] for Z | Single Z |
| Shor | [[9,1,3]] | Any single-qubit |

### Key Principles

1. **Discretization:** Continuous errors → Pauli errors via measurement
2. **Syndrome:** Extract error info without disturbing data
3. **Concatenation:** Combine codes for more error types
4. **Threshold:** Below critical error rate → scalable QC possible

---

## Preview: Day 671

Tomorrow: **Comprehensive Problems** - Integration across all Semester 1B topics!
