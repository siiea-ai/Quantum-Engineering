# Day 576: Circuit Composition

## Overview
**Day 576** | Week 83, Day 2 | Year 1, Month 21 | Sequential and Parallel Operations

Today we formalize how quantum circuits compose—sequential gates multiply matrices right-to-left, while parallel (simultaneous) gates combine via tensor products.

---

## Learning Objectives

1. Apply the matrix multiplication rule for sequential gates
2. Use tensor products for parallel gate operations
3. Distinguish between sequential and parallel composition
4. Compute circuit unitaries for arbitrary gate arrangements
5. Understand the mathematical structure of circuit composition
6. Handle mixed sequential-parallel circuits systematically

---

## Core Content

### Sequential Composition

When gates are applied **in sequence** (one after another), their matrices multiply in **reverse order**:

```
         ┌───┐   ┌───┐   ┌───┐
|ψ⟩ ─────┤ A ├───┤ B ├───┤ C ├───── |ψ'⟩
         └───┘   └───┘   └───┘
```

$$\boxed{|ψ'\rangle = C \cdot B \cdot A \cdot |ψ\rangle}$$

**Why reverse order?** In linear algebra, the rightmost operator acts first:
$$(CBA)|ψ\rangle = C(B(A|ψ\rangle))$$

### Parallel Composition

When gates act **simultaneously** on different qubits, use the **tensor product**:

```
         ┌───┐
q₀: ─────┤ A ├─────
         └───┘
         ┌───┐
q₁: ─────┤ B ├─────
         └───┘
```

$$\boxed{U_{parallel} = A \otimes B}$$

For $n$ qubits with simultaneous gates:
$$U = U_0 \otimes U_1 \otimes \cdots \otimes U_{n-1}$$

### Mixed Composition

Real circuits combine both sequential and parallel operations. Strategy:

1. **Identify time slices** (vertical cuts through the circuit)
2. **For each slice**: tensor product all gates in that slice
3. **Combine slices**: multiply slice matrices right-to-left

**Example:**
```
         ┌───┐   ┌───┐
q₀: ─────┤ H ├───┤ X ├───●────
         └───┘   └───┘   │
         ┌───┐           │
q₁: ─────┤ Z ├───────────⊕────
         └───┘
```

**Time slices:**
- Slice 1: H on q₀, Z on q₁ → $H \otimes Z$
- Slice 2: X on q₀, I on q₁ → $X \otimes I$
- Slice 3: CNOT → $CNOT$

**Total unitary:**
$$U = CNOT \cdot (X \otimes I) \cdot (H \otimes Z)$$

### The Identity Gate

When nothing happens to a qubit in a time slice, use the **identity** $I$:

```
         ┌───┐
q₀: ─────┤ H ├─────
         └───┘
q₁: ───────────────  (nothing happens)
```

$$U = H \otimes I$$

### Tensor Product Properties

For matrices $A$, $B$, $C$, $D$:

**Mixed-product property:**
$$(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$$

This is crucial for circuit simplification!

**Associativity:**
$$(A \otimes B) \otimes C = A \otimes (B \otimes C)$$

### Controlled Gate Decomposition

The controlled-U gate cannot be written as a simple tensor product:

$$CU \neq A \otimes B$$

for any $A$, $B$. This is what makes controlled gates **entangling**.

**Explicit form:**
$$C_U = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U$$

### Circuit Depth and Parallelism

**Circuit depth** = number of time slices (sequential steps)
**Circuit width** = number of qubits

Parallel composition reduces depth:

```
Sequential (depth 2):        Parallel (depth 1):
    ┌───┐   ┌───┐               ┌───┐
q₀: ┤ H ├───┤   ├───        q₀: ┤ H ├───
    └───┘   │ X │               └───┘
q₁: ────────┤   ├───        q₁: ─[X]────
            └───┘
```

### Matrix Dimensions

For $n$ qubits:
- State vector: $2^n \times 1$
- Gate matrix: $2^n \times 2^n$

Single-qubit gate on qubit $k$ in $n$-qubit system:
$$U_k = I^{\otimes k} \otimes U \otimes I^{\otimes(n-k-1)}$$

---

## Worked Examples

### Example 1: Two-Qubit Sequential Circuit

Compute the unitary for:
```
         ┌───┐   ┌───┐
q₀: ─────┤ H ├───┤ S ├─────
         └───┘   └───┘
         ┌───┐   ┌───┐
q₁: ─────┤ X ├───┤ H ├─────
         └───┘   └───┘
```

**Solution:**

**Slice 1:** $H \otimes X$
$$H \otimes X = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \otimes \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

$$= \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & -1 \\ 1 & 0 & -1 & 0 \end{pmatrix}$$

**Slice 2:** $S \otimes H$
$$S \otimes H = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} \otimes \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$$= \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 & 0 & 0 \\ 1 & -1 & 0 & 0 \\ 0 & 0 & i & i \\ 0 & 0 & i & -i \end{pmatrix}$$

**Total:** $U = (S \otimes H)(H \otimes X)$

### Example 2: Using the Mixed-Product Property

Simplify $(H \otimes I)(H \otimes Z)$.

**Solution:**

Using $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$:

$$(H \otimes I)(H \otimes Z) = (HH) \otimes (IZ) = I \otimes Z = I \otimes Z$$

Since $H^2 = I$, the Hadamards cancel, leaving just $Z$ on qubit 1.

### Example 3: CNOT in Different Positions

Find the matrix for CNOT where q₁ is control and q₀ is target:
```
q₀: ───⊕───
       │
q₁: ───●───
```

**Solution:**

Standard CNOT has q₀ as control:
$$CNOT_{01} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$$

For q₁ as control:
$$CNOT_{10} = I \otimes |0\rangle\langle 0| + X \otimes |1\rangle\langle 1|$$

**Matrix form:**
$$CNOT_{10} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}$$

Action: $|00\rangle \to |00\rangle$, $|01\rangle \to |11\rangle$, $|10\rangle \to |10\rangle$, $|11\rangle \to |01\rangle$

---

## Practice Problems

### Problem 1: Sequential Composition
Compute the total unitary for:
```
         ┌───┐   ┌───┐   ┌───┐
|ψ⟩ ─────┤ X ├───┤ Y ├───┤ Z ├───
         └───┘   └───┘   └───┘
```
Simplify your answer.

### Problem 2: Parallel Composition
Compute $(H \otimes H \otimes H)|000\rangle$.

### Problem 3: Mixed Circuit
Find the unitary for:
```
         ┌───┐
q₀: ─────┤ H ├───●───────────
         └───┘   │
q₁: ─────────────⊕───[H]─────
```

### Problem 4: Simplification
Use circuit identities to simplify:
```
         ┌───┐   ┌───┐
q₀: ─────┤ H ├───┤ H ├─────
         └───┘   └───┘
```

---

## Computational Lab

```python
"""Day 576: Circuit Composition"""
import numpy as np
from functools import reduce

# Standard gates
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]])
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

def tensor(*gates):
    """Tensor product of multiple gates"""
    return reduce(np.kron, gates)

def sequential(*gates):
    """Sequential composition (reversed multiplication)"""
    return reduce(lambda a, b: b @ a, gates)

def state_string(psi, n_qubits):
    """Pretty print quantum state"""
    terms = []
    for i, amp in enumerate(psi):
        if np.abs(amp) > 1e-10:
            bits = format(i, f'0{n_qubits}b')
            if np.abs(amp.imag) < 1e-10:
                terms.append(f"{amp.real:.4f}|{bits}>")
            else:
                terms.append(f"({amp:.4f})|{bits}>")
    return " + ".join(terms)

# ===== Example 1: Sequential Composition =====
print("=" * 60)
print("Example 1: Sequential Single-Qubit Gates")
print("=" * 60)
print("\nCircuit: |psi> --[X]--[Y]--[Z]--")
print("\nMatrix multiplication: U = Z * Y * X")

U_sequential = Z @ Y @ X
print("\nZ * Y * X =")
print(np.round(U_sequential, 4))

# Verify: XYZ = iI (up to global phase)
print(f"\nNote: XYZ = iI (the Pauli matrices anticommute)")
print(f"Trace of XYZ: {np.trace(U_sequential):.4f} (should be 2i)")

# ===== Example 2: Parallel Composition =====
print("\n" + "=" * 60)
print("Example 2: Parallel Composition with Tensor Products")
print("=" * 60)
print("\nCircuit:")
print("q0: --[H]--")
print("q1: --[H]--")
print("q2: --[H]--")
print("\nU = H tensor H tensor H")

HHH = tensor(H, H, H)
print(f"\nH^tensor3 shape: {HHH.shape}")

# Apply to |000>
state_000 = np.zeros(8, dtype=complex)
state_000[0] = 1

result = HHH @ state_000
print(f"\n(H tensor H tensor H)|000> =")
print(state_string(result, 3))
print("\nAll amplitudes equal: uniform superposition!")

# ===== Example 3: Mixed-Product Property =====
print("\n" + "=" * 60)
print("Example 3: Mixed-Product Property")
print("=" * 60)
print("\n(A tensor B)(C tensor D) = (AC) tensor (BD)")
print("\nVerify: (H tensor X)(Z tensor Y) = (HZ) tensor (XY)")

# Left side
left = tensor(H, X) @ tensor(Z, Y)

# Right side
right = tensor(H @ Z, X @ Y)

print(f"\nMatrices equal: {np.allclose(left, right)}")

# ===== Example 4: Building a Full Circuit =====
print("\n" + "=" * 60)
print("Example 4: Full Circuit Analysis")
print("=" * 60)
print("\nCircuit:")
print("         +---+   ")
print("q0: -----| H |---*---")
print("         +---+   |   ")
print("         +---+   |   ")
print("q1: -----| X |---X---")
print("         +---+")
print("\nSlice 1: H tensor X")
print("Slice 2: CNOT")

slice1 = tensor(H, X)
slice2 = CNOT

U_total = slice2 @ slice1
print("\nTotal U = CNOT * (H tensor X)")
print(np.round(U_total, 4))

# Apply to |00>
state_00 = np.array([1, 0, 0, 0], dtype=complex)
final = U_total @ state_00
print(f"\nU|00> = {state_string(final, 2)}")

# ===== Example 5: CNOT with Different Control =====
print("\n" + "=" * 60)
print("Example 5: CNOT with Different Control/Target")
print("=" * 60)

# Standard CNOT: q0 control, q1 target
print("Standard CNOT (q0 control, q1 target):")
print(CNOT)

# Reversed CNOT: q1 control, q0 target
# |00> -> |00>, |01> -> |11>, |10> -> |10>, |11> -> |01>
CNOT_10 = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
])

print("\nReversed CNOT (q1 control, q0 target):")
print(CNOT_10)

# Verify using SWAP
print("\nRelation: CNOT_10 = SWAP * CNOT_01 * SWAP")
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

CNOT_10_via_swap = SWAP @ CNOT @ SWAP
print(f"Verified: {np.allclose(CNOT_10, CNOT_10_via_swap)}")

# ===== Example 6: Circuit Identities =====
print("\n" + "=" * 60)
print("Example 6: Circuit Identities and Simplifications")
print("=" * 60)

# H^2 = I
print("\n1. H^2 = I")
print(f"   H @ H = I: {np.allclose(H @ H, I)}")

# X^2 = Y^2 = Z^2 = I
print("\n2. Pauli squares = I")
print(f"   X^2 = I: {np.allclose(X @ X, I)}")
print(f"   Y^2 = I: {np.allclose(Y @ Y, I)}")
print(f"   Z^2 = I: {np.allclose(Z @ Z, I)}")

# HXH = Z, HZH = X
print("\n3. Hadamard conjugation")
print(f"   HXH = Z: {np.allclose(H @ X @ H, Z)}")
print(f"   HZH = X: {np.allclose(H @ Z @ H, X)}")
print(f"   HYH = -Y: {np.allclose(H @ Y @ H, -Y)}")

# CNOT identities
print("\n4. CNOT on |++> creates Bell state")
plus_plus = tensor(np.array([1, 1])/np.sqrt(2), np.array([1, 1])/np.sqrt(2))
print(f"   CNOT|++> = {state_string(CNOT @ plus_plus, 2)}")

# ===== Example 7: Circuit Depth Analysis =====
print("\n" + "=" * 60)
print("Example 7: Sequential vs Parallel Depth")
print("=" * 60)

# Sequential circuit (depth 4)
print("\nSequential circuit (depth 4):")
print("q0: --[H]--[X]--[H]--[Z]--")

U_seq = sequential(H, X, H, Z)
print(f"U = ZHXH = {np.round(U_seq, 4)}")

# Using HXH = Z identity
print("\nSimplified: U = ZZ = I")
print(f"Actually: ZHXH = Z(HXH) = ZZ = I: {np.allclose(U_seq, I)}")

print("\n" + "=" * 60)
print("Circuit Composition Summary")
print("=" * 60)
print("""
Key Rules:
1. Sequential gates: U = Un * ... * U2 * U1 (right to left)
2. Parallel gates: U = U1 tensor U2 tensor ... tensor Un
3. Mixed-product: (A tensor B)(C tensor D) = (AC) tensor (BD)
4. Controlled gates: CU = |0><0| tensor I + |1><1| tensor U
""")
```

**Expected Output:**
```
============================================================
Example 1: Sequential Single-Qubit Gates
============================================================

Circuit: |psi> --[X]--[Y]--[Z]--

Matrix multiplication: U = Z * Y * X

Z * Y * X =
[[0.+1.j 0.+0.j]
 [0.+0.j 0.+1.j]]

Note: XYZ = iI (the Pauli matrices anticommute)
Trace of XYZ: 0.0000+2.0000j (should be 2i)
```

---

## Summary

### Composition Rules

| Type | Operation | Circuit | Matrix |
|------|-----------|---------|--------|
| Sequential | One after another | `--[A]--[B]--` | $BA$ |
| Parallel | Simultaneous on different qubits | Lines stacked | $A \otimes B$ |
| Controlled | Conditional on control qubit | $\bullet$-connected | $\|0\rangle\langle 0\| \otimes I + \|1\rangle\langle 1\| \otimes U$ |

### Key Formulas

| Concept | Formula |
|---------|---------|
| Sequential | $U = U_n \cdots U_2 U_1$ |
| Parallel | $U = U_0 \otimes U_1 \otimes \cdots \otimes U_{n-1}$ |
| Mixed-product | $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$ |
| Gate on qubit k | $I^{\otimes k} \otimes U \otimes I^{\otimes(n-k-1)}$ |

### Key Takeaways

1. **Sequential gates multiply right-to-left** (opposite to reading direction)
2. **Parallel gates tensor together** (order matches qubit ordering)
3. **Mixed-product property** enables circuit simplification
4. **Controlled gates** are inherently non-tensor-product
5. **Circuit depth** counts sequential time steps
6. **Identity gates** fill empty slots in parallel composition

---

## Daily Checklist

- [ ] I can compose sequential gates using matrix multiplication
- [ ] I can compose parallel gates using tensor products
- [ ] I understand why matrix order is reversed from circuit order
- [ ] I can apply the mixed-product property for simplification
- [ ] I can construct CNOT with different control/target assignments
- [ ] I ran the computational lab and verified circuit identities

---

*Next: Day 577 — Measurement in Circuits*
