# Day 688: Pauli Group and Logical Operators

## Week 99: Three-Qubit Codes | Month 25: QEC Fundamentals I | Year 2

---

## Schedule Overview

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Pauli Group Structure |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Problem Solving |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Computational Lab |

---

## Learning Objectives

By the end of Day 688, you will be able to:

1. **Understand the centralizer and normalizer** of the stabilizer group
2. **Define and construct logical operators** $\bar{X}$ and $\bar{Z}$
3. **Calculate code distance** from logical operator weights
4. **Identify correctable vs detectable errors** using stabilizer structure
5. **Apply the Pauli weight** concept to error analysis
6. **Understand the Gottesman-Knill theorem** implications

---

## Centralizer and Normalizer

### The Centralizer of $\mathcal{S}$

**Definition:** The **centralizer** of $\mathcal{S}$ in $\mathcal{P}_n$ is:

$$C(\mathcal{S}) = \{P \in \mathcal{P}_n : [P, S] = 0 \text{ for all } S \in \mathcal{S}\}$$

The centralizer contains all Paulis that commute with every stabilizer.

**Theorem:** $C(\mathcal{S}) = \mathcal{S}$ for stabilizer codes.

**Why?** In the binary symplectic picture, $C(\mathcal{S})$ is the symplectic complement of $\mathcal{S}$, which equals $\mathcal{S}$ for self-orthogonal codes.

### The Normalizer of $\mathcal{S}$

**Definition:** The **normalizer** of $\mathcal{S}$ in $\mathcal{P}_n$ is:

$$N(\mathcal{S}) = \{P \in \mathcal{P}_n : P\mathcal{S}P^\dagger = \mathcal{S}\}$$

For abelian $\mathcal{S}$, this simplifies to:

$$N(\mathcal{S}) = \{P \in \mathcal{P}_n : [P, S] = 0 \text{ for all } S \in \mathcal{S}\}$$

**Key Insight:** $N(\mathcal{S})$ contains:
1. All stabilizers: $\mathcal{S} \subseteq N(\mathcal{S})$
2. Logical operators that preserve the code space
3. Detectable errors

---

## Logical Operators

### Definition

**Logical operators** are Paulis that:
1. **Commute with all stabilizers** (preserve the code space)
2. **Are not in $\mathcal{S}$** (act non-trivially on logical qubits)

$$\boxed{\text{Logical operators} = N(\mathcal{S}) \setminus \mathcal{S}}$$

### Logical X and Z

For a code encoding $k$ logical qubits, we need $2k$ logical operators:
- $\bar{X}_1, \bar{X}_2, \ldots, \bar{X}_k$ (logical bit-flips)
- $\bar{Z}_1, \bar{Z}_2, \ldots, \bar{Z}_k$ (logical phase-flips)

**Requirements:**
1. $[\bar{X}_i, S] = 0$ and $[\bar{Z}_i, S] = 0$ for all $S \in \mathcal{S}$
2. $\{\bar{X}_i, \bar{Z}_i\} = 0$ (anticommute for same logical qubit)
3. $[\bar{X}_i, \bar{Z}_j] = 0$ for $i \neq j$ (commute for different qubits)
4. $[\bar{X}_i, \bar{X}_j] = [\bar{Z}_i, \bar{Z}_j] = 0$ (all X's commute, all Z's commute)

### Example: Bit-Flip Code Logical Operators

**Stabilizers:** $Z_1Z_2$, $Z_2Z_3$

**Find $\bar{X}$:** Must commute with $Z_1Z_2$ and $Z_2Z_3$, but not be in $\mathcal{S}$.

Try $X_1X_2X_3$:
- $[X_1X_2X_3, Z_1Z_2] = ?$

  $X_1X_2X_3$ has X on positions 1,2,3. $Z_1Z_2$ has Z on positions 1,2.
  Anticommutations at positions 1 and 2: $(-1)^2 = 1$ → commutes ✓

- $[X_1X_2X_3, Z_2Z_3] = ?$

  Anticommutations at positions 2 and 3: $(-1)^2 = 1$ → commutes ✓

So $\bar{X} = X_1X_2X_3$ is a valid logical X.

**Find $\bar{Z}$:** Must commute with stabilizers, anticommute with $\bar{X}$.

Try $Z_1$:
- $[Z_1, Z_1Z_2] = 0$ ✓
- $[Z_1, Z_2Z_3] = 0$ ✓
- $\{Z_1, X_1X_2X_3\}$: anticommutes at position 1 only → anticommutes ✓

But $Z_1 \sim Z_1 \cdot (Z_1Z_2) = Z_2$ and $Z_1 \sim Z_1 \cdot (Z_2Z_3) \cdot (Z_1Z_2) = Z_3$.

All single-qubit Z's are equivalent: $\bar{Z} = Z_1 \sim Z_2 \sim Z_3$

**Canonical choice:** $\bar{X} = X_1X_2X_3$, $\bar{Z} = Z_1$ (or $Z_1Z_2Z_3$)

---

## Code Distance

### Definition

**Definition:** The **distance** $d$ of a stabilizer code is the minimum weight of any logical operator:

$$\boxed{d = \min_{L \in N(\mathcal{S}) \setminus \mathcal{S}} \text{wt}(L)}$$

Equivalently, $d$ is the minimum weight of any Pauli that:
- Commutes with all stabilizers
- Is not a stabilizer

### Distance and Error Correction

**Theorem:** A code with distance $d$ can:
- **Detect** any error of weight $\leq d - 1$
- **Correct** any error of weight $\leq \lfloor(d-1)/2\rfloor$

**Why?**
- An error $E$ is undetectable if $E \in N(\mathcal{S})$
- Minimum weight undetectable error has weight $d$
- Errors of weight $< d$ are detectable
- Distinguishing errors up to weight $t$ requires $d > 2t$

### Calculating Distance

**For bit-flip code:**
- $\bar{X} = X_1X_2X_3$ has weight 3
- $\bar{Z} = Z_1$ has weight 1

$$d = \min(3, 1) = 1$$

The code has distance 1 against general errors (because single Z errors are undetectable).

**For Shor [[9,1,3]] code:**
- Minimum weight X-type logical: weight 3 (e.g., $X_1X_2X_3$)
- Minimum weight Z-type logical: weight 3 (e.g., $Z_1Z_4Z_7$)

$$d = \min(3, 3) = 3$$

---

## Error Classification

### Types of Errors

Given stabilizer group $\mathcal{S}$, errors partition into:

| Error Type | Condition | Syndrome | Example (bit-flip code) |
|------------|-----------|----------|-------------------------|
| **No error** | $E \in \mathcal{S}$ | $(0,0,\ldots,0)$ | $I$, $Z_1Z_2$ |
| **Detectable** | $E \notin N(\mathcal{S})$ | Non-zero | $X_1$, $X_2$, $X_3$ |
| **Logical** | $E \in N(\mathcal{S}) \setminus \mathcal{S}$ | $(0,0,\ldots,0)$ | $Z_1$, $X_1X_2X_3$ |

### Correctable Error Sets

**Definition:** A set of errors $\{E_a\}$ is **correctable** if:
1. Each $E_a$ is detectable (has unique syndrome), OR
2. Different errors with same syndrome differ by a stabilizer: $E_a^\dagger E_b \in \mathcal{S}$

The second condition is **degeneracy** — multiple errors may have the same effect on codewords.

### Example: Shor Code Degeneracy

In the Shor code:
- $Z_1$, $Z_2$, $Z_3$ all produce syndrome indicating "Z error in block 1"
- $Z_1^\dagger Z_2 = Z_1 Z_2 \in \mathcal{S}$ ✓

All three Z errors are equivalent — any correction works!

---

## The Weight Enumerator

### Pauli Weight Distribution

For analysis, we track how many Paulis of each weight are in different sets:

**Stabilizer weight enumerator:** $A_j = |\{S \in \mathcal{S} : \text{wt}(S) = j\}|$

**Normalizer weight enumerator:** Counts Paulis in $N(\mathcal{S})$ by weight.

### MacWilliams Identity (Quantum)

There's a quantum analog of the classical MacWilliams identity relating weight enumerators of $\mathcal{S}$ and $N(\mathcal{S})/\mathcal{S}$.

---

## Gottesman-Knill Theorem Preview

### Statement

**Theorem (Gottesman-Knill):** A quantum circuit can be efficiently simulated on a classical computer if:
1. Initial state is a computational basis state
2. Gates are from the **Clifford group** (H, S, CNOT)
3. Measurements are in the computational basis

### Relevance to Stabilizers

Stabilizer states (states that are +1 eigenstates of some stabilizer group) can be efficiently represented by their stabilizer generators — only $O(n^2)$ bits for $n$ qubits!

**Implications:**
- Stabilizer codes can be efficiently simulated
- Error correction with Clifford gates is classically tractable
- Non-Clifford gates (like T) are needed for quantum advantage

---

## Worked Examples

### Example 1: Phase-Flip Code Logical Operators

**Problem:** Find $\bar{X}$ and $\bar{Z}$ for the phase-flip code.

**Solution:**

Stabilizers: $X_1X_2$, $X_2X_3$

**$\bar{Z}$:** Must commute with X-stabilizers, not be a stabilizer.

Try $Z_1Z_2Z_3$:
- $[Z_1Z_2Z_3, X_1X_2]$: anticommutes at positions 1,2 → $(-1)^2 = 1$ ✓
- $[Z_1Z_2Z_3, X_2X_3]$: anticommutes at positions 2,3 → $(-1)^2 = 1$ ✓

$\bar{Z} = Z_1Z_2Z_3$

**$\bar{X}$:** Must anticommute with $\bar{Z}$, commute with stabilizers.

Try $X_1$:
- $[X_1, X_1X_2] = 0$ ✓
- $[X_1, X_2X_3] = 0$ ✓
- $\{X_1, Z_1Z_2Z_3\}$: anticommutes at position 1 → anticommutes ✓

$\bar{X} = X_1$ (or equivalently $X_2$ or $X_3$)

**Distance:** $d = \min(\text{wt}(\bar{X}), \text{wt}(\bar{Z})) = \min(1, 3) = 1$

### Example 2: Calculate Shor Code Distance

**Problem:** Verify that the Shor code has distance 3.

**Solution:**

Need to find minimum weight element of $N(\mathcal{S}) \setminus \mathcal{S}$.

**X-type logical operators:**
- $\bar{X} = X_1X_2X_3X_4X_5X_6X_7X_8X_9$ (weight 9)
- Minimum weight: $X_1X_2X_3$ (acts on one block) has weight 3

**Z-type logical operators:**
- $\bar{Z} = Z_1Z_2Z_3Z_4Z_5Z_6Z_7Z_8Z_9$ (weight 9)
- Minimum weight: $Z_1Z_4Z_7$ (one qubit per block) has weight 3

**Distance:** $d = \min(3, 3) = 3$ ✓

### Example 3: Error Correction Capability

**Problem:** How many errors can the [[5,1,3]] code correct?

**Solution:**

For distance $d = 3$:
$$t = \left\lfloor \frac{d-1}{2} \right\rfloor = \left\lfloor \frac{2}{2} \right\rfloor = 1$$

The [[5,1,3]] code can correct any **single-qubit error**.

---

## Practice Problems

### Problem Set A: Direct Application

**A.1** For the bit-flip code, verify that $X_1$ produces syndrome $(1,0)$ while $Z_1$ produces $(0,0)$.

**A.2** List all elements of the normalizer $N(\mathcal{S})$ for the 3-qubit bit-flip code.

**A.3** Show that $X_1X_2$ is in the stabilizer group of the Shor code.

### Problem Set B: Intermediate

**B.1** Prove that for any stabilizer code, $\mathcal{S} \subseteq N(\mathcal{S})$.

**B.2** For the phase-flip code, show that all single-qubit X operators are equivalent logical operators.

**B.3** If a code has stabilizers $g_1 = XZZX$ and $g_2 = ZXXZ$, find a valid logical $\bar{Z}$.

### Problem Set C: Challenging

**C.1** Prove that the distance of a stabilizer code equals the minimum weight of $N(\mathcal{S}) \setminus \mathcal{S}$.

**C.2** Show that for CSS codes, the X-distance and Z-distance can be computed independently.

**C.3** Construct a [[4,2,2]] code (4 qubits, 2 logical qubits, distance 2).

---

## Computational Lab

```python
"""
Day 688 Computational Lab: Logical Operators and Code Distance
=============================================================
"""

import numpy as np
from typing import List, Set, Tuple
from itertools import product

# Binary Pauli class from Day 687
class PauliBinary:
    def __init__(self, x_part, z_part):
        self.x = np.array(x_part, dtype=int) % 2
        self.z = np.array(z_part, dtype=int) % 2
        self.n = len(self.x)

    @classmethod
    def from_string(cls, s: str):
        n = len(s)
        x, z = np.zeros(n, int), np.zeros(n, int)
        for i, c in enumerate(s):
            if c in 'XY': x[i] = 1
            if c in 'ZY': z[i] = 1
        return cls(x, z)

    def to_string(self) -> str:
        chars = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
        return ''.join(chars[(self.x[i], self.z[i])] for i in range(self.n))

    def symplectic_product(self, other) -> int:
        return (np.dot(self.x, other.z) + np.dot(other.x, self.z)) % 2

    def commutes_with(self, other) -> bool:
        return self.symplectic_product(other) == 0

    def weight(self) -> int:
        return sum((self.x[i] | self.z[i]) for i in range(self.n))

    def multiply(self, other):
        return PauliBinary((self.x + other.x) % 2, (self.z + other.z) % 2)

    def __eq__(self, other):
        return np.array_equal(self.x, other.x) and np.array_equal(self.z, other.z)

    def __hash__(self):
        return hash((tuple(self.x), tuple(self.z)))

print("=" * 65)
print("PART 1: Finding Logical Operators")
print("=" * 65)

def find_normalizer(stabilizers: List[PauliBinary], n: int) -> Set[PauliBinary]:
    """Find all elements of N(S) by brute force (for small n)."""
    normalizer = set()
    for bits in product([0, 1], repeat=2*n):
        x = np.array(bits[:n])
        z = np.array(bits[n:])
        P = PauliBinary(x, z)
        if all(P.commutes_with(g) for g in stabilizers):
            normalizer.add(P)
    return normalizer

def find_stabilizer_group(generators: List[PauliBinary]) -> Set[PauliBinary]:
    """Generate full stabilizer group from generators."""
    # Start with identity
    n = generators[0].n
    identity = PauliBinary(np.zeros(n, int), np.zeros(n, int))
    group = {identity}

    # Add products
    frontier = set(generators)
    while frontier:
        new_frontier = set()
        for g in frontier:
            for s in list(group):
                prod = g.multiply(s)
                if prod not in group:
                    group.add(prod)
                    new_frontier.add(prod)
        frontier = new_frontier

    return group

# Bit-flip code
bf_gens = [PauliBinary.from_string("ZZI"), PauliBinary.from_string("IZZ")]
bf_stab = find_stabilizer_group(bf_gens)
bf_norm = find_normalizer(bf_gens, 3)

print(f"\nBit-flip code:")
print(f"  Stabilizer group S (size {len(bf_stab)}):")
for s in sorted(bf_stab, key=lambda p: p.to_string()):
    print(f"    {s.to_string()}")

print(f"\n  Normalizer N(S) (size {len(bf_norm)}):")
for p in sorted(bf_norm, key=lambda p: (p.weight(), p.to_string())):
    in_stab = "∈ S" if p in bf_stab else "logical"
    print(f"    {p.to_string()} (wt={p.weight()}) {in_stab}")

# Find logical operators
logicals = bf_norm - bf_stab
print(f"\n  Logical operators (N(S) \\ S):")
for p in sorted(logicals, key=lambda p: (p.weight(), p.to_string())):
    print(f"    {p.to_string()} (weight {p.weight()})")

# Code distance
if logicals:
    distance = min(p.weight() for p in logicals)
    print(f"\n  Code distance d = {distance}")

print("\n" + "=" * 65)
print("PART 2: Phase-Flip Code Analysis")
print("=" * 65)

pf_gens = [PauliBinary.from_string("XXI"), PauliBinary.from_string("IXX")]
pf_stab = find_stabilizer_group(pf_gens)
pf_norm = find_normalizer(pf_gens, 3)
pf_logicals = pf_norm - pf_stab

print(f"\nPhase-flip code:")
print(f"  |S| = {len(pf_stab)}, |N(S)| = {len(pf_norm)}")
print(f"  Logical operators:")
for p in sorted(pf_logicals, key=lambda p: (p.weight(), p.to_string())):
    print(f"    {p.to_string()} (weight {p.weight()})")

distance_pf = min(p.weight() for p in pf_logicals) if pf_logicals else 0
print(f"  Distance d = {distance_pf}")

print("\n" + "=" * 65)
print("PART 3: Error Classification")
print("=" * 65)

def classify_error(error: PauliBinary, generators: List[PauliBinary],
                   stabilizer_group: Set[PauliBinary], normalizer: Set[PauliBinary]) -> str:
    """Classify an error."""
    if error in stabilizer_group:
        return "Stabilizer (no error)"
    elif error in normalizer:
        return "Logical (undetectable)"
    else:
        return "Detectable"

print("\nError classification for bit-flip code:")
test_errors = ['III', 'XII', 'IXI', 'IIX', 'ZII', 'IZI', 'IIZ', 'XXX', 'ZZI', 'ZZZ']
for e_str in test_errors:
    e = PauliBinary.from_string(e_str)
    classification = classify_error(e, bf_gens, bf_stab, bf_norm)
    # Get syndrome
    syndrome = tuple(e.symplectic_product(g) for g in bf_gens)
    print(f"  {e_str}: {classification}, syndrome={syndrome}")

print("\n" + "=" * 65)
print("PART 4: Shor Code Distance Verification")
print("=" * 65)

# Define Shor code generators
shor_gens = []
# ZZ within blocks
for block in range(3):
    base = block * 3
    shor_gens.append(PauliBinary(np.zeros(9, int),
        np.array([1 if i in [base, base+1] else 0 for i in range(9)])))
    shor_gens.append(PauliBinary(np.zeros(9, int),
        np.array([1 if i in [base+1, base+2] else 0 for i in range(9)])))
# XXXXXX across blocks
shor_gens.append(PauliBinary(np.array([1,1,1,1,1,1,0,0,0]), np.zeros(9, int)))
shor_gens.append(PauliBinary(np.array([0,0,0,1,1,1,1,1,1]), np.zeros(9, int)))

print(f"Shor code: {len(shor_gens)} generators")
for i, g in enumerate(shor_gens):
    print(f"  g{i+1} = {g.to_string()}")

# Check some logical operators
logical_X = PauliBinary.from_string("XXXIIIIII")  # X on block 1
logical_Z = PauliBinary(np.zeros(9, int), np.array([1,0,0,1,0,0,1,0,0]))  # Z on first of each

print(f"\nCandidate logical operators:")
print(f"  X_L = {logical_X.to_string()} (weight {logical_X.weight()})")
commutes_X = all(logical_X.commutes_with(g) for g in shor_gens)
print(f"    Commutes with all stabilizers: {commutes_X}")

print(f"  Z_L = {logical_Z.to_string()} (weight {logical_Z.weight()})")
commutes_Z = all(logical_Z.commutes_with(g) for g in shor_gens)
print(f"    Commutes with all stabilizers: {commutes_Z}")

print(f"\n  X_L anticommutes with Z_L: {not logical_X.commutes_with(logical_Z)}")

# The Shor code has distance 3
print(f"\n  Shor code distance: d = 3 (minimum logical weight)")

print("\n" + "=" * 65)
print("SUMMARY: Logical Operators and Distance")
print("=" * 65)

summary = """
┌───────────────────────────────────────────────────────────────┐
│           Logical Operators and Code Distance                  │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│ NORMALIZER N(S):                                               │
│   • All Paulis commuting with stabilizers                      │
│   • Contains S as subgroup                                     │
│   • Logical operators = N(S) \\ S                              │
│                                                                │
│ LOGICAL OPERATORS:                                             │
│   • Preserve code space (commute with S)                       │
│   • Act non-trivially (not in S)                               │
│   • X̄ and Z̄ must anticommute                                  │
│                                                                │
│ CODE DISTANCE:                                                 │
│   d = min weight of N(S) \\ S                                  │
│   • Detect errors of weight ≤ d-1                              │
│   • Correct errors of weight ≤ ⌊(d-1)/2⌋                       │
│                                                                │
│ ERROR CLASSIFICATION:                                          │
│   • Stabilizer: E ∈ S → no effect                              │
│   • Detectable: E ∉ N(S) → non-zero syndrome                  │
│   • Logical: E ∈ N(S)\\S → zero syndrome, changes state       │
│                                                                │
└───────────────────────────────────────────────────────────────┘
"""
print(summary)

print("✅ Day 688 Lab Complete!")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Normalizer | $N(\mathcal{S}) = \{P \in \mathcal{P}_n : [P,S]=0 \; \forall S \in \mathcal{S}\}$ |
| Logical operators | $N(\mathcal{S}) \setminus \mathcal{S}$ |
| Code distance | $d = \min_{L \in N(\mathcal{S}) \setminus \mathcal{S}} \text{wt}(L)$ |
| Error correction | $t = \lfloor(d-1)/2\rfloor$ |

### Code Distance Examples

| Code | $\bar{X}$ weight | $\bar{Z}$ weight | Distance $d$ |
|------|-----------------|-----------------|--------------|
| Bit-flip [[3,1,1]] | 3 | 1 | 1 |
| Phase-flip [[3,1,1]] | 1 | 3 | 1 |
| Shor [[9,1,3]] | 3 | 3 | 3 |
| Steane [[7,1,3]] | 3 | 3 | 3 |

---

## Daily Checklist

- [ ] I understand centralizer and normalizer
- [ ] I can find logical operators from stabilizers
- [ ] I can calculate code distance
- [ ] I understand error classification (stabilizer/detectable/logical)
- [ ] I know the Gottesman-Knill theorem basics

---

## Preview: Day 689

Tomorrow: **Knill-Laflamme Conditions**
- The fundamental theorem of quantum error correction
- Necessary and sufficient conditions
- Degenerate codes

---

**Day 688 Complete!** Week 99: 2/7 days (29%)
