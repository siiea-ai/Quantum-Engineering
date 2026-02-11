# Day 689: Knill-Laflamme Conditions

## Week 99: Three-Qubit Codes | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | QEC Conditions Theory |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Problem Solving |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational Lab |

---

## Learning Objectives

By the end of Day 689, you will be able to:

1. **State and prove the Knill-Laflamme theorem**
2. **Apply the QEC conditions** to verify code correctability
3. **Distinguish degenerate from non-degenerate codes**
4. **Connect Knill-Laflamme to stabilizer formalism**
5. **Understand approximate error correction** concepts
6. **Derive the quantum Singleton bound**

---

## The Knill-Laflamme Theorem

### Statement

**Theorem (Knill-Laflamme, 1997):**

Let $\mathcal{C}$ be a quantum code with orthonormal basis $\{|\psi_i\rangle\}_{i=1}^{2^k}$, and let $\{E_a\}$ be a set of error operators. The code can correct errors $\{E_a\}$ if and only if:

$$\boxed{\langle \psi_i | E_a^\dagger E_b | \psi_j \rangle = C_{ab} \delta_{ij}}$$

where $C_{ab}$ is a Hermitian matrix depending only on the errors (not on $i,j$).

### Equivalent Formulations

**Projector Form:** Let $\Pi = \sum_i |\psi_i\rangle\langle\psi_i|$ be the projector onto $\mathcal{C}$. Then:

$$\boxed{\Pi E_a^\dagger E_b \Pi = C_{ab} \Pi}$$

**Stabilizer Form:** For stabilizer codes with stabilizer group $\mathcal{S}$:

$$E_a^\dagger E_b \in \mathcal{S} \cup (\mathcal{P}_n \setminus N(\mathcal{S}))$$

Either $E_a^\dagger E_b$ is a stabilizer (acts trivially) or it anticommutes with some stabilizer (gives orthogonal subspaces).

---

## Understanding the Conditions

### The Two Parts

The Knill-Laflamme conditions split into:

**1. Diagonal Conditions ($i = j$):**

$$\langle \psi_i | E_a^\dagger E_b | \psi_i \rangle = C_{ab}$$

This says: errors affect all codewords **uniformly**. The "overlap" of error effects doesn't depend on which codeword.

**2. Off-Diagonal Conditions ($i \neq j$):**

$$\langle \psi_i | E_a^\dagger E_b | \psi_j \rangle = 0$$

This says: different codewords remain **orthogonal** after errors. Information isn't leaked between codewords.

### Physical Interpretation

Think of $E_a|\psi_i\rangle$ as "error $a$ applied to codeword $i$."

**Correction is possible if:**
- We can determine which error (or error class) occurred (from syndrome)
- Once we know the error, we can undo it (via recovery operation)

The KL conditions ensure that error spaces $E_a\mathcal{C}$ are either:
1. **Orthogonal** (distinguishable syndromes), or
2. **Identical** (same effect on code space — degenerate)

---

## Proof of Knill-Laflamme

### Necessity ($\Rightarrow$)

**Assume** a recovery operation $\mathcal{R}$ exists such that:
$$(\mathcal{R} \circ \mathcal{E})(|\psi_i\rangle\langle\psi_j|) = |\psi_i\rangle\langle\psi_j|$$

for all $i, j$ where $\mathcal{E}(\rho) = \sum_a E_a \rho E_a^\dagger$.

Then for any encoded state:
$$\mathcal{R}\left(\sum_a E_a |\psi_i\rangle\langle\psi_j| E_a^\dagger\right) = |\psi_i\rangle\langle\psi_j|$$

Taking the trace with $|\psi_k\rangle\langle\psi_l|$ and using properties of CPTP maps leads to the KL conditions.

### Sufficiency ($\Leftarrow$)

**Given** the KL conditions, we construct an explicit recovery.

**Step 1:** Diagonalize $C = UDU^\dagger$ where $D = \text{diag}(d_1, \ldots, d_r)$.

**Step 2:** Define new error operators:
$$F_a = \sum_b U_{ab} E_b$$

These satisfy $\langle\psi_i|F_a^\dagger F_b|\psi_j\rangle = d_a \delta_{ab} \delta_{ij}$.

**Step 3:** For $d_a > 0$, define normalized errors:
$$\tilde{F}_a = \frac{F_a}{\sqrt{d_a}}$$

Now: $\langle\psi_i|\tilde{F}_a^\dagger \tilde{F}_b|\psi_j\rangle = \delta_{ab}\delta_{ij}$

**Step 4:** The subspaces $\tilde{F}_a\mathcal{C}$ are mutually orthogonal!

**Step 5:** Construct recovery:
$$R_a = \sum_i |\psi_i\rangle\langle\psi_i|\tilde{F}_a^\dagger$$

Then $\mathcal{R}(\rho) = \sum_a R_a \rho R_a^\dagger$ is the recovery operation. ∎

---

## The QEC Matrix $C_{ab}$

### Structure

The matrix $C$ with entries $C_{ab} = \langle\psi_i|E_a^\dagger E_b|\psi_i\rangle$ is:
- **Hermitian:** $C_{ba}^* = C_{ab}$
- **Positive semidefinite:** $C \geq 0$ (it's a Gram matrix)

### Non-Degenerate Codes

**Definition:** A code is **non-degenerate** with respect to $\{E_a\}$ if $C$ has full rank.

In this case, all error subspaces $E_a\mathcal{C}$ are mutually orthogonal, and each error maps to a unique syndrome.

### Degenerate Codes

**Definition:** A code is **degenerate** if $\text{rank}(C) < |\{E_a\}|$.

This means some errors have the **same effect** on the code space:
$$E_1|\psi\rangle = E_2|\psi\rangle \quad \forall |\psi\rangle \in \mathcal{C}$$

Equivalently: $E_1^\dagger E_2$ acts as the identity on $\mathcal{C}$ (i.e., $E_1^\dagger E_2 \in \mathcal{S}$ for stabilizer codes).

**Example: Shor Code**

$Z_1$, $Z_2$, $Z_3$ all have the same effect within block 1 because $Z_1Z_2, Z_2Z_3 \in \mathcal{S}$.

---

## Connecting to Stabilizer Codes

### Stabilizer Version of KL Conditions

For stabilizer codes with stabilizer group $\mathcal{S}$:

**Theorem:** The code can correct error set $\{E_a\}$ if and only if for all pairs $E_a, E_b$:

$$\boxed{E_a^\dagger E_b \in \mathcal{S} \text{ or } E_a^\dagger E_b \notin N(\mathcal{S})}$$

**Interpretation:**
- $E_a^\dagger E_b \in \mathcal{S}$: Errors are equivalent (degenerate) — same syndrome, same correction
- $E_a^\dagger E_b \notin N(\mathcal{S})$: Errors are distinguishable — different syndromes

**What's forbidden:** $E_a^\dagger E_b \in N(\mathcal{S}) \setminus \mathcal{S}$ (logical operator)

This would mean two errors lead to the same syndrome but different logical effects — uncorrectable!

### Distance Condition

For a code with distance $d$:
- All Paulis of weight $< d$ are either in $\mathcal{S}$ or have non-zero syndrome
- All single-qubit errors ($\text{wt} \leq 1$) satisfy KL if $d \geq 3$

---

## Quantum Singleton Bound

### Statement

**Theorem (Quantum Singleton Bound):**

For an $[[n, k, d]]$ quantum code:

$$\boxed{n - k \geq 2(d - 1)}$$

or equivalently:

$$k \leq n - 2d + 2$$

### Proof Sketch

Consider encoding $k$ qubits into $n$ qubits. To correct $t = \lfloor(d-1)/2\rfloor$ errors:

1. The error syndrome must identify which $t$ qubits (of $n$) have errors
2. The syndrome must also identify which of $3^t$ error types occurred
3. Total information: roughly $t \log n + t \log 3$ bits

The syndrome has $n - k$ bits (from $n - k$ stabilizer generators).

For large $n$ and $d \sim n$, this gives $n - k \geq 2(d-1)$.

### MDS Codes

Codes achieving equality are called **quantum MDS (Maximum Distance Separable)**:

$$n - k = 2(d - 1)$$

The [[5,1,3]] code is quantum MDS: $5 - 1 = 4 = 2(3-1)$.

---

## Worked Examples

### Example 1: Verify KL for Bit-Flip Code

**Problem:** Verify the Knill-Laflamme conditions for the bit-flip code against single X errors.

**Solution:**

Code basis: $|0_L\rangle = |000\rangle$, $|1_L\rangle = |111\rangle$

Error set: $\{I, X_1, X_2, X_3\}$

**Compute $E_a^\dagger E_b$ products:**
- $I^\dagger X_1 = X_1$: anticommutes with $Z_1Z_2$ → not in $N(\mathcal{S})$ ✓
- $X_1^\dagger X_2 = X_1X_2$: anticommutes with $Z_2Z_3$ → not in $N(\mathcal{S})$ ✓
- $X_1^\dagger X_1 = I \in \mathcal{S}$ ✓

All pairs satisfy KL conditions!

**Alternatively, check matrix elements:**

$\langle 000|X_1^\dagger X_2|000\rangle = \langle 100|010\rangle = 0$ ✓
$\langle 111|X_1^\dagger X_2|111\rangle = \langle 011|101\rangle = 0$ ✓

### Example 2: Show Shor Code is Degenerate

**Problem:** Demonstrate degeneracy of the Shor code.

**Solution:**

Consider errors $Z_1$ and $Z_2$ on the Shor code.

$$Z_1^\dagger Z_2 = Z_1 Z_2$$

Is $Z_1Z_2$ a stabilizer? Yes! $Z_1Z_2 \in \mathcal{S}$.

Therefore $Z_1$ and $Z_2$ have the **same effect** on all codewords:
$$Z_1|\psi_L\rangle = Z_2|\psi_L\rangle \cdot (\text{stabilizer action})$$

The code is degenerate.

### Example 3: Why [[3,1,1]] Codes Fail

**Problem:** Show the bit-flip code cannot correct Z errors.

**Solution:**

Consider $E_1 = I$ and $E_2 = Z_1$.

$$E_1^\dagger E_2 = Z_1$$

Is $Z_1 \in \mathcal{S}$? No.
Is $Z_1 \in N(\mathcal{S})$? Yes! $Z_1$ commutes with $Z_1Z_2$ and $Z_2Z_3$.
Is $Z_1 \in N(\mathcal{S}) \setminus \mathcal{S}$? Yes — it's a logical operator!

KL conditions are violated: $Z_1$ produces the same syndrome as $I$ but has different logical effect.

---

## Practice Problems

### Problem Set A: Direct Application

**A.1** For the phase-flip code, verify KL conditions for single Z errors.

**A.2** Show that the bit-flip code is non-degenerate with respect to $\{I, X_1, X_2, X_3\}$.

**A.3** Compute the QEC matrix $C_{ab}$ for the bit-flip code with error set $\{I, X_1\}$.

### Problem Set B: Intermediate

**B.1** Prove: If $d \geq 3$, the code can correct all single-qubit errors.

**B.2** For the [[5,1,3]] code, verify it saturates the quantum Singleton bound.

**B.3** Show that for stabilizer codes, the KL conditions reduce to checking $E_a^\dagger E_b$.

### Problem Set C: Challenging

**C.1** Prove the sufficiency direction of the Knill-Laflamme theorem in detail.

**C.2** A code can correct errors $\{E_a\}$ and $\{F_b\}$ separately. Can it correct $\{E_a F_b\}$? When?

**C.3** Derive the quantum Singleton bound rigorously using entanglement arguments.

---

## Computational Lab

```python
"""
Day 689 Computational Lab: Knill-Laflamme Conditions
====================================================
"""

import numpy as np
from typing import List

# =============================================================================
# Part 1: Setup
# =============================================================================

I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

def tensor(*args):
    result = args[0]
    for a in args[1:]:
        result = np.kron(result, a)
    return result

print("=" * 65)
print("KNILL-LAFLAMME CONDITIONS VERIFICATION")
print("=" * 65)

# =============================================================================
# Part 2: Bit-Flip Code KL Verification
# =============================================================================

print("\n--- Bit-Flip Code vs Single X Errors ---")

# Codewords
psi_0L = tensor(np.array([1,0]), np.array([1,0]), np.array([1,0]))  # |000⟩
psi_1L = tensor(np.array([0,1]), np.array([0,1]), np.array([0,1]))  # |111⟩
code_basis = [psi_0L, psi_1L]

# Errors
errors = {
    'I': tensor(I, I, I),
    'X1': tensor(X, I, I),
    'X2': tensor(I, X, I),
    'X3': tensor(I, I, X)
}
error_names = list(errors.keys())

# Compute QEC matrix C_ab
def compute_qec_matrix(code_basis, errors):
    """Compute the QEC matrix C_ab = ⟨ψ_i|E_a†E_b|ψ_i⟩."""
    n_errors = len(errors)
    error_list = list(errors.values())

    # Check all basis states give same result
    C = np.zeros((n_errors, n_errors), dtype=complex)
    for a in range(n_errors):
        for b in range(n_errors):
            E_dag_E = error_list[a].conj().T @ error_list[b]
            # Should be same for all |ψ_i⟩
            vals = [psi.conj() @ E_dag_E @ psi for psi in code_basis]
            C[a, b] = vals[0]
            # Check uniformity
            if not np.allclose(vals, vals[0]):
                print(f"  Warning: Non-uniform diagonal at ({a},{b})")

    return C

def check_off_diagonal(code_basis, errors):
    """Check that ⟨ψ_i|E_a†E_b|ψ_j⟩ = 0 for i ≠ j."""
    error_list = list(errors.values())
    violations = []

    for a, E_a in enumerate(error_list):
        for b, E_b in enumerate(error_list):
            E_dag_E = E_a.conj().T @ E_b
            for i, psi_i in enumerate(code_basis):
                for j, psi_j in enumerate(code_basis):
                    if i != j:
                        val = psi_i.conj() @ E_dag_E @ psi_j
                        if np.abs(val) > 1e-10:
                            violations.append((a, b, i, j, val))
    return violations

C_bf = compute_qec_matrix(code_basis, errors)
print("\nQEC Matrix C_ab:")
print("     ", "  ".join(f"{n:>4}" for n in error_names))
for i, row in enumerate(C_bf):
    print(f" {error_names[i]:>3}", " ".join(f"{v.real:5.2f}" for v in row))

violations = check_off_diagonal(code_basis, errors)
print(f"\nOff-diagonal violations: {len(violations)}")

print(f"\nKL Conditions satisfied: {len(violations) == 0}")

# =============================================================================
# Part 3: Testing Z Errors (Should Fail)
# =============================================================================

print("\n--- Bit-Flip Code vs Z Errors ---")

errors_z = {
    'I': tensor(I, I, I),
    'Z1': tensor(Z, I, I),
}

C_z = compute_qec_matrix(code_basis, errors_z)
print("\nQEC Matrix C_ab for {I, Z1}:")
print(C_z)

# Check if Z1 acts the same on different codewords
print("\nZ1 action on codewords:")
print(f"  Z1|0_L⟩ = |0_L⟩ eigenvalue: {psi_0L.conj() @ tensor(Z,I,I) @ psi_0L}")
print(f"  Z1|1_L⟩ = -|1_L⟩ eigenvalue: {psi_1L.conj() @ tensor(Z,I,I) @ psi_1L}")

# The diagonal elements differ!
print("\nDiagonal KL condition check:")
print(f"  ⟨0_L|Z1†Z1|0_L⟩ = {psi_0L.conj() @ psi_0L}")  # = 1
print(f"  ⟨1_L|Z1†Z1|1_L⟩ = {psi_1L.conj() @ psi_1L}")  # = 1 (same)
print(f"  ⟨0_L|I†Z1|0_L⟩ = {psi_0L.conj() @ tensor(Z,I,I) @ psi_0L}")  # = 1
print(f"  ⟨1_L|I†Z1|1_L⟩ = {psi_1L.conj() @ tensor(Z,I,I) @ psi_1L}")  # = -1 (different!)

print("\n⚠️ KL CONDITIONS VIOLATED: I†Z1 gives different values on |0_L⟩ vs |1_L⟩")

# =============================================================================
# Part 4: Shor Code Degeneracy
# =============================================================================

print("\n" + "=" * 65)
print("SHOR CODE DEGENERACY")
print("=" * 65)

# Shor code logical states (simplified representation)
# |0_L⟩ = (|000⟩+|111⟩)⊗3 / 2√2
plus_enc = (tensor(np.array([1,0]),np.array([1,0]),np.array([1,0])) +
            tensor(np.array([0,1]),np.array([0,1]),np.array([0,1]))) / np.sqrt(2)
minus_enc = (tensor(np.array([1,0]),np.array([1,0]),np.array([1,0])) -
             tensor(np.array([0,1]),np.array([0,1]),np.array([0,1]))) / np.sqrt(2)

shor_0L = tensor(plus_enc, plus_enc, plus_enc)
shor_1L = tensor(minus_enc, minus_enc, minus_enc)

print(f"\nShor |0_L⟩ norm: {np.linalg.norm(shor_0L):.4f}")
print(f"Shor |1_L⟩ norm: {np.linalg.norm(shor_1L):.4f}")

# Check Z1 and Z2 equivalence
Z1 = np.kron(np.kron(Z, I), I)  # Z on qubit 1 of 3-qubit block
Z2 = np.kron(np.kron(I, Z), I)  # Z on qubit 2

# In full 9-qubit space
I3 = np.eye(8)
Z1_full = np.kron(Z1, I3)  # Z on qubit 1 (of 9)
Z2_full = np.kron(Z2, I3)  # Z on qubit 2 (of 9)

# Check if Z1|ψ⟩ = Z2|ψ⟩ (up to phase in stabilizer)
diff = Z1_full @ shor_0L - Z2_full @ shor_0L
print(f"\n||Z1|0_L⟩ - Z2|0_L⟩|| = {np.linalg.norm(diff):.6f}")

if np.linalg.norm(diff) < 1e-10:
    print("✓ Z1 and Z2 have identical effect → Shor code is DEGENERATE")
else:
    print("Different effects")

# =============================================================================
# Part 5: Quantum Singleton Bound
# =============================================================================

print("\n" + "=" * 65)
print("QUANTUM SINGLETON BOUND")
print("=" * 65)

codes = [
    ("Bit-flip [[3,1,1]]", 3, 1, 1),
    ("Phase-flip [[3,1,1]]", 3, 1, 1),
    ("[[5,1,3]]", 5, 1, 3),
    ("Steane [[7,1,3]]", 7, 1, 3),
    ("Shor [[9,1,3]]", 9, 1, 3),
]

print("\nQuantum Singleton Bound: n - k ≥ 2(d-1)")
print("-" * 55)
print(f"{'Code':<20} {'n-k':<8} {'2(d-1)':<8} {'Saturates?':<12}")
print("-" * 55)

for name, n, k, d in codes:
    lhs = n - k
    rhs = 2 * (d - 1)
    saturates = "MDS ✓" if lhs == rhs else ""
    print(f"{name:<20} {lhs:<8} {rhs:<8} {saturates:<12}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 65)
print("SUMMARY: KNILL-LAFLAMME CONDITIONS")
print("=" * 65)

summary = """
┌───────────────────────────────────────────────────────────────┐
│           Knill-Laflamme Theorem (QEC Conditions)              │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│ THEOREM: Code C corrects errors {E_a} iff:                     │
│                                                                │
│   ⟨ψ_i|E_a†E_b|ψ_j⟩ = C_ab δ_ij                               │
│                                                                │
│ MEANING:                                                       │
│   • Diagonal (i=j): Errors affect all codewords uniformly      │
│   • Off-diagonal (i≠j): Codewords stay orthogonal after error  │
│                                                                │
│ FOR STABILIZER CODES:                                          │
│   E_a†E_b must be either:                                      │
│   • In S (stabilizer) → degenerate errors                      │
│   • Not in N(S) → distinguishable errors                       │
│                                                                │
│ DEGENERACY:                                                    │
│   • Non-degenerate: rank(C) = |{E_a}|                          │
│   • Degenerate: rank(C) < |{E_a}| (e.g., Shor code)           │
│                                                                │
│ QUANTUM SINGLETON: n - k ≥ 2(d-1)                              │
│   Codes achieving equality are MDS (e.g., [[5,1,3]])          │
│                                                                │
└───────────────────────────────────────────────────────────────┘
"""
print(summary)

print("✅ Day 689 Lab Complete!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Knill-Laflamme | $\langle \psi_i \| E_a^\dagger E_b \| \psi_j \rangle = C_{ab} \delta_{ij}$ |
| Stabilizer version | $E_a^\dagger E_b \in \mathcal{S} \cup (\mathcal{P}_n \setminus N(\mathcal{S}))$ |
| Quantum Singleton | $n - k \geq 2(d-1)$ |
| Non-degenerate | $\text{rank}(C) = \|\{E_a\}\|$ |

### Main Takeaways

1. **Knill-Laflamme conditions** are necessary and sufficient for QEC
2. **Two requirements:** uniform diagonal, orthogonal off-diagonal
3. **Stabilizer codes:** check $E_a^\dagger E_b$ membership
4. **Degeneracy:** multiple errors with same effect (Shor code)
5. **Singleton bound** limits how much we can encode

---

## Preview: Day 690

Tomorrow: **Shor Code Deep Analysis**
- Complete syndrome table
- All 8 stabilizers detailed
- Error correction protocol
- Degeneracy in action

---

**Day 689 Complete!** Week 99: 3/7 days (43%)
