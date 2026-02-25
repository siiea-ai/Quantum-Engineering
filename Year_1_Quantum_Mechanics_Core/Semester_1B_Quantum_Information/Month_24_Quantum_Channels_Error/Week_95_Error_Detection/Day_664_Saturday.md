# Day 664: Stabilizer Formalism Preview

## Week 95: Error Detection/Correction Intro | Month 24: Quantum Channels & Error Introduction

---

## Learning Objectives

1. **Introduce** the Pauli group and stabilizer formalism
2. **Define** stabilizer codes using group generators
3. **Reformulate** the 3-qubit and Shor codes as stabilizer codes
4. **Understand** the connection between stabilizers and syndromes
5. **Preview** the CSS code construction

---

## Core Content

### 1. Motivation: A Systematic Framework

So far we've constructed codes by:
- Intuition (repetition)
- Duality (phase-flip from bit-flip)
- Concatenation (Shor code)

**Question:** Is there a systematic way to construct and analyze quantum codes?

**Answer:** The **stabilizer formalism** (Gottesman, 1996)!

### 2. The Pauli Group

**Single-qubit Pauli group:**
$$\mathcal{P}_1 = \{\pm I, \pm iI, \pm X, \pm iX, \pm Y, \pm iY, \pm Z, \pm iZ\}$$

16 elements, forms a group under matrix multiplication.

**n-qubit Pauli group:**
$$\mathcal{P}_n = \{c \cdot P_1 \otimes P_2 \otimes \cdots \otimes P_n : c \in \{\pm 1, \pm i\}, P_j \in \{I, X, Y, Z\}\}$$

$4 \times 4^n$ elements.

### 3. Key Properties of Pauli Operators

**Hermitian (up to phase):** $P^\dagger = \pm P$

**Involutory:** $P^2 = \pm I$

**Commutation:** Two Paulis either commute or anti-commute:
$$[P, Q] = 0 \text{ or } \{P, Q\} = 0$$

**Eigenvalues:** $\pm 1$ (for Hermitian Paulis)

### 4. Stabilizer Definition

**Definition:** A **stabilizer group** $\mathcal{S}$ is an abelian subgroup of $\mathcal{P}_n$ that:
1. Does not contain $-I$
2. All elements are Hermitian

**Code space:** The simultaneous +1 eigenspace of all stabilizers:
$$\mathcal{C} = \{|\psi\rangle : S|\psi\rangle = |\psi\rangle \text{ for all } S \in \mathcal{S}\}$$

### 5. Generators

A stabilizer group is typically specified by its **generators**:
$$\mathcal{S} = \langle S_1, S_2, \ldots, S_r \rangle$$

where $S_i$ generate $\mathcal{S}$ via multiplication.

**Code parameters:**
- $n$ physical qubits
- $r$ independent generators
- $k = n - r$ logical qubits

Code dimension = $2^k$

### 6. Three-Qubit Bit-Flip as Stabilizer Code

**Generators:**
$$S_1 = Z_1Z_2 = ZZI$$
$$S_2 = Z_2Z_3 = IZZ$$

**Stabilizer group:**
$$\mathcal{S} = \{III, ZZI, IZZ, ZIZ\}$$

**Code space:** States $|\psi\rangle$ with $S_1|\psi\rangle = S_2|\psi\rangle = |\psi\rangle$

This gives $\mathcal{C} = \text{span}\{|000\rangle, |111\rangle\}$ ✓

**Parameters:** $n=3$, $r=2$, $k=1$

### 7. Three-Qubit Phase-Flip as Stabilizer Code

**Generators:**
$$S_1 = X_1X_2 = XXI$$
$$S_2 = X_2X_3 = IXX$$

**Code space:** $\mathcal{C} = \text{span}\{|+++\rangle, |---\rangle\}$ ✓

### 8. Shor Code Stabilizers

**8 generators for [[9,1,3]] Shor code:**

Bit-flip detection (within blocks):
- $S_1 = Z_1Z_2$
- $S_2 = Z_2Z_3$
- $S_3 = Z_4Z_5$
- $S_4 = Z_5Z_6$
- $S_5 = Z_7Z_8$
- $S_6 = Z_8Z_9$

Phase-flip detection (between blocks):
- $S_7 = X_1X_2X_3X_4X_5X_6$
- $S_8 = X_4X_5X_6X_7X_8X_9$

**Parameters:** $n=9$, $r=8$, $k=1$ ✓

### 9. Syndrome Measurement as Stabilizer Eigenvalues

**Key insight:** Measuring stabilizer generators gives the syndrome!

For error $E$ acting on code state $|\psi\rangle$:
$$S_j(E|\psi\rangle) = S_jE|\psi\rangle = \pm ES_j|\psi\rangle = \pm E|\psi\rangle$$

The sign is:
- $+1$ if $[S_j, E] = 0$ (commute)
- $-1$ if $\{S_j, E\} = 0$ (anti-commute)

**Syndrome bit $s_j$:** 0 if commute, 1 if anti-commute

### 10. Error Correction Condition (Stabilizer Version)

**Correctable errors:** Error set $\{E_a\}$ is correctable if for all $a \neq b$:

$$E_a^\dagger E_b \notin \mathcal{S} \quad \text{or} \quad E_a^\dagger E_b \in \mathcal{S}$$

**Interpretation:**
- Different errors have different syndromes, OR
- Different errors have the same effect on code (degenerate)

### 11. Logical Operators

**Logical operators** commute with all stabilizers but are not in $\mathcal{S}$.

**Centralizer:** $C(\mathcal{S}) = \{P \in \mathcal{P}_n : [P, S] = 0 \text{ for all } S \in \mathcal{S}\}$

**Logical operators:** $\bar{\mathcal{P}} = C(\mathcal{S}) / \mathcal{S}$

For k logical qubits, need $2k$ logical operators: $\bar{X}_1, \bar{Z}_1, \ldots, \bar{X}_k, \bar{Z}_k$

### 12. Code Distance

**Definition:** The code distance $d$ is the minimum weight of a logical operator.

$$d = \min\{|P| : P \in C(\mathcal{S}) \setminus \mathcal{S}\}$$

where $|P|$ is the number of non-identity components.

**Relation to error correction:** Corrects up to $t = \lfloor(d-1)/2\rfloor$ errors.

### 13. CSS Codes Preview

**Calderbank-Shor-Steane (CSS) codes:** A special construction from classical codes.

Given classical codes $C_1$ and $C_2$ with $C_2 \subset C_1$:

**Stabilizers:**
- X-type: From $C_2^\perp$ (parity checks of $C_2$)
- Z-type: From $C_1^\perp$ (parity checks of $C_1$)

**Example:** The 7-qubit Steane code comes from the Hamming [7,4,3] code!

### 14. Advantages of Stabilizer Formalism

1. **Systematic construction:** Build codes from generator matrices
2. **Efficient simulation:** Gottesman-Knill theorem for Clifford circuits
3. **Fault-tolerance:** Natural framework for fault-tolerant gates
4. **Code design:** Search for codes with desired parameters

---

## Worked Example

**Problem:** Show that the bit-flip code stabilizers correctly identify an X error on qubit 2.

**Solution:**

Stabilizers: $S_1 = ZZI$, $S_2 = IZZ$

Error: $E = X_2 = IXI$

Check commutation:
1. $S_1 E = (ZZI)(IXI) = ZXI$ vs $ES_1 = (IXI)(ZZI) = ZXI$ → Same?

   Actually, need to be more careful:
   $$S_1 E = Z \otimes Z \otimes I \cdot I \otimes X \otimes I = Z \otimes ZX \otimes I$$
   $$E S_1 = I \otimes X \otimes I \cdot Z \otimes Z \otimes I = Z \otimes XZ \otimes I$$

   Since $ZX = -XZ$, we have $S_1 E = -E S_1$ → **anti-commute**, syndrome bit = 1

2. $S_2 E = (IZZ)(IXI) = IXZ$ vs $ES_2 = (IXI)(IZZ) = IXZ$

   $$S_2 E = I \otimes ZX \otimes Z = I \otimes (-XZ) \otimes Z$$
   $$E S_2 = I \otimes XZ \otimes Z$$

   Again anti-commute → syndrome bit = 1

**Syndrome:** (1, 1) → indicates error on qubit 2 ✓

---

## Practice Problems

1. Find all elements of the stabilizer group for the 3-qubit bit-flip code.

2. Verify that $\bar{X} = XXX$ and $\bar{Z} = ZII$ are logical operators for the bit-flip code.

3. Prove that logical operators commute with all stabilizers.

4. Calculate the code distance of the Shor code by finding the minimum weight logical operator.

5. Show that $X_1X_2$ and $X_2X_3$ generate the same group as $X_1X_2$ and $X_1X_3$.

---

## Computational Lab

```python
"""Day 664: Stabilizer Formalism Preview"""

import numpy as np
from itertools import product

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

def tensor(*matrices):
    """Compute tensor product."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

def pauli_string_to_matrix(s):
    """Convert 'XYZ' to X⊗Y⊗Z matrix."""
    mats = [paulis[c] for c in s]
    return tensor(*mats)

def commutes(A, B, tol=1e-10):
    """Check if A and B commute."""
    comm = A @ B - B @ A
    return np.allclose(comm, 0, atol=tol)

def anticommutes(A, B, tol=1e-10):
    """Check if A and B anticommute."""
    anticomm = A @ B + B @ A
    return np.allclose(anticomm, 0, atol=tol)

# 3-qubit bit-flip code stabilizers
print("3-Qubit Bit-Flip Code Stabilizer Analysis")
print("=" * 50)

S1 = pauli_string_to_matrix('ZZI')
S2 = pauli_string_to_matrix('IZZ')

print("\nGenerators: S1 = ZZI, S2 = IZZ")

# Generate full stabilizer group
S3 = S1 @ S2  # ZIZ
I3 = pauli_string_to_matrix('III')

stabilizers = [I3, S1, S2, S3]
stab_names = ['III', 'ZZI', 'IZZ', 'ZIZ']

print("\nFull stabilizer group:")
for name in stab_names:
    print(f"  {name}")

# Find code space
print("\nCode space (eigenvalue +1 for all stabilizers):")

# Build projector onto code space
P = I3.copy()
for S in stabilizers[1:]:  # Skip identity
    P = P @ (I3 + S) / 2

# Find code basis by looking at eigenspace
eigenvalues, eigenvectors = np.linalg.eigh(P)
code_basis = eigenvectors[:, np.abs(eigenvalues - 1) < 1e-10]
print(f"Code dimension: {code_basis.shape[1]}")

# Check that |000⟩ and |111⟩ are in code space
ket_000 = np.zeros(8, dtype=complex)
ket_000[0] = 1
ket_111 = np.zeros(8, dtype=complex)
ket_111[7] = 1

in_code_000 = np.allclose(P @ ket_000, ket_000)
in_code_111 = np.allclose(P @ ket_111, ket_111)
print(f"|000⟩ in code space: {in_code_000}")
print(f"|111⟩ in code space: {in_code_111}")

# Syndrome calculation
print("\n" + "=" * 50)
print("Syndrome Analysis for Errors")
print("=" * 50)

errors = ['III', 'XII', 'IXI', 'IIX']

print("\nError | S1=ZZI | S2=IZZ | Syndrome")
print("-" * 40)

for e_name in errors:
    E = pauli_string_to_matrix(e_name)

    # Syndrome = 0 if commute, 1 if anticommute
    s1 = 1 if anticommutes(S1, E) else 0
    s2 = 1 if anticommutes(S2, E) else 0

    print(f"  {e_name}  |   {'+' if s1==0 else '-'}    |   {'+' if s2==0 else '-'}    | ({s1}, {s2})")

# Logical operators
print("\n" + "=" * 50)
print("Logical Operators")
print("=" * 50)

X_L = pauli_string_to_matrix('XXX')
Z_L = pauli_string_to_matrix('ZII')  # Or ZII, IZI, IIZ - all equivalent mod stabilizers

print("\nX_L = XXX, Z_L = ZII")

# Check they commute with stabilizers
print("\nCommutation with stabilizers:")
for name, S in zip(['ZZI', 'IZZ'], [S1, S2]):
    print(f"  [X_L, {name}] = 0: {commutes(X_L, S)}")
    print(f"  [Z_L, {name}] = 0: {commutes(Z_L, S)}")

# Check they anticommute with each other
print(f"\n{{X_L, Z_L}} = 0: {anticommutes(X_L, Z_L)}")

# Verify logical action
print("\nLogical action on code states:")
# X_L|000⟩ should give |111⟩
X_L_on_000 = X_L @ ket_000
print(f"X_L|000⟩ = |111⟩: {np.allclose(X_L_on_000, ket_111)}")

# Z_L|000⟩ should give |000⟩, Z_L|111⟩ should give -|111⟩
Z_L_on_000 = Z_L @ ket_000
Z_L_on_111 = Z_L @ ket_111
print(f"Z_L|000⟩ = |000⟩: {np.allclose(Z_L_on_000, ket_000)}")
print(f"Z_L|111⟩ = -|111⟩: {np.allclose(Z_L_on_111, -ket_111)}")

# Code distance
print("\n" + "=" * 50)
print("Code Distance")
print("=" * 50)

print("\nLogical operators and their weights:")
logical_ops = [
    ('XXX', 3),
    ('ZII', 1),
    ('IZI', 1),
    ('IIZ', 1),
]

min_weight = float('inf')
for op, w in logical_ops:
    print(f"  {op}: weight {w}")
    min_weight = min(min_weight, w)

print(f"\nCode distance d = {min_weight}")
print(f"Corrects t = floor((d-1)/2) = {(min_weight-1)//2} errors")

# Note: This shows the limitation - bit-flip code has d=1 for Z errors!
print("\nNote: The logical Z operators have weight 1, meaning")
print("phase errors cannot be detected/corrected.")
```

---

## Summary

| Concept | Description |
|---------|-------------|
| Pauli group | $\mathcal{P}_n = \{\pm 1, \pm i\} \times \{I,X,Y,Z\}^{\otimes n}$ |
| Stabilizer group | Abelian subgroup of $\mathcal{P}_n$, no $-I$ |
| Code space | Simultaneous +1 eigenspace of all stabilizers |
| Generators | $r$ independent stabilizers give $k = n-r$ logical qubits |
| Syndrome | Eigenvalue pattern under stabilizer measurement |
| Logical operators | Centralizer modulo stabilizer group |
| Code distance | Minimum weight logical operator |

**Key Formulas:**
- Code parameters: $[[n, k, d]]$ where $k = n - r$
- Error correction capability: $t = \lfloor(d-1)/2\rfloor$
- Syndrome bit: 0 if commute, 1 if anti-commute

---

## Preview: Day 665

Tomorrow: **Week 95 Review** - comprehensive integration of error detection and correction concepts!
