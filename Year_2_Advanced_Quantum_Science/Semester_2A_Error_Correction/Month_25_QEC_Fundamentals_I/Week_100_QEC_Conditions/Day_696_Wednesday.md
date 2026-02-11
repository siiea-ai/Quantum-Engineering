# Day 696: Degeneracy in Quantum Codes

## Overview

**Week:** 100 (QEC Conditions)
**Day:** Wednesday
**Date:** Year 2, Month 25, Day 696
**Topic:** Degenerate Quantum Codes — Beyond Sphere-Packing
**Hours:** 7 (3.5 theory + 2.5 problems + 1 computational lab)

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Degeneracy definition, theory |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Degenerate code analysis |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Degeneracy simulation |

---

## Prerequisites

From Days 694-695:
- Quantum Singleton and Hamming bounds
- Perfect codes and sphere-packing
- Stabilizer formalism

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Define** degenerate vs non-degenerate quantum codes
2. **Explain** why degeneracy allows "violating" the Hamming bound
3. **Identify** degeneracy in Steane and Shor codes
4. **Analyze** how degeneracy improves error correction
5. **Understand** the role of degeneracy in surface codes
6. **Design** decoders that exploit degeneracy

---

## Core Content

### 1. What Is Degeneracy?

#### Non-Degenerate Codes

A quantum code is **non-degenerate** if every correctable error produces a unique syndrome AND acts distinctly on the code space.

**Mathematically:** For errors $E_a, E_b$ within the correctable set:
$$E_a^\dagger E_b \notin S \Rightarrow \text{syn}(E_a) \neq \text{syn}(E_b)$$

**Example:** The [[5,1,3]] code is non-degenerate — all 16 single-qubit errors have distinct syndromes.

#### Degenerate Codes

A quantum code is **degenerate** if there exist distinct errors $E_a \neq E_b$ that:
1. Produce the **same syndrome**: $\text{syn}(E_a) = \text{syn}(E_b)$
2. But act **identically** on the code space: $E_a|c\rangle \propto E_b|c\rangle$ for all codewords

**Mathematically:** $E_a^\dagger E_b \in S$ (their product is a stabilizer)

#### Key Insight

Degenerate codes can correct "more errors than they can distinguish" because multiple errors with the same syndrome all map to the same logical state — any correction works!

---

### 2. Formal Characterization

#### Stabilizer Code Perspective

For a stabilizer code with stabilizer group $S$:

**Non-degenerate error pair:** $E_a^\dagger E_b \notin S$ and $E_a^\dagger E_b \notin N(S) \setminus S$

**Degenerate error pair:** $E_a^\dagger E_b \in S$

#### Knill-Laflamme Conditions Revisited

The Knill-Laflamme conditions state:
$$\langle c_i | E_a^\dagger E_b | c_j \rangle = C_{ab} \delta_{ij}$$

For degenerate codes:
- Multiple $(a, b)$ pairs can have $C_{ab} \neq 0$
- Errors $E_a$ and $E_b$ with $E_a^\dagger E_b \in S$ satisfy $C_{ab} = 1$ (stabilizer acts as identity on code space)

---

### 3. The Shor Code: A Degenerate Example

#### Syndrome Degeneracy

The Shor [[9,1,3]] code has:
- 8 stabilizer generators → $2^8 = 256$ syndromes
- Only 28 single-qubit errors (1 identity + 27 Pauli errors)
- Many syndromes are "degenerate"

#### Explicit Degeneracy

Consider two errors on the first block (qubits 1,2,3):

$$E_1 = X_1, \quad E_2 = X_2$$

These produce different X-syndromes:
- $X_1$: anticommutes with $g_1 = X_1 X_2$
- $X_2$: anticommutes with $g_1 = X_1 X_2$

But their product:
$$E_1^\dagger E_2 = X_1 X_2 = g_1 \in S$$

**Interpretation:** $X_1$ and $X_2$ errors differ by a stabilizer — they're "equivalent" on the code space!

#### Syndrome Equivalence

On logical states:
$$X_1 |0_L\rangle = X_2 |0_L\rangle$$

Both errors flip the first block from $|+++\rangle$ to something with a single X error, but the *logical* effect is identical after error correction.

---

### 4. The Steane Code: Another Degenerate Code

#### Structure Analysis

The Steane [[7,1,3]] code:
- 6 stabilizer generators → $2^6 = 64$ syndromes
- 22 single-qubit errors
- $64 - 22 = 42$ "unused" syndromes

But this isn't waste — it's degeneracy!

#### Finding Degenerate Pairs

Consider $E_1 = X_1 X_2 X_3$ and $E_2 = X_4 X_5 X_6 X_7$:

Both are weight-3 or weight-4 Pauli operators. Their product:
$$E_1^\dagger E_2 = X_1 X_2 X_3 X_4 X_5 X_6 X_7 = \bar{X}$$

Wait — that's a logical operator, not a stabilizer! So these are NOT degenerate with each other.

Let's try: $E_1 = X_1$ and $E_2 = X_1 \cdot g_3$ where $g_3 = X_1 X_3 X_5 X_7$:
$$E_2 = X_1 \cdot X_1 X_3 X_5 X_7 = X_3 X_5 X_7$$

Then:
$$E_1^\dagger E_2 = X_1 \cdot X_3 X_5 X_7 = g_3 \notin S$$

Hmm, that's not a single stabilizer. The Steane code's degeneracy is more subtle — it appears in higher-weight errors and in the structure of multi-qubit error correction.

---

### 5. Why Degeneracy Helps

#### Beyond Sphere-Packing

The Hamming bound assumes each error needs a unique syndrome. Degenerate codes "reuse" syndromes for equivalent errors.

**Non-degenerate:** Each syndrome → one error
**Degenerate:** Each syndrome → equivalence class of errors

#### Improved Error Rates

For realistic noise models, degenerate codes achieve:
- Lower logical error rates for the same physical error rate
- Better performance under correlated noise
- Closer approach to information-theoretic limits

#### Example: Correlated Noise

If qubits 1 and 2 often experience errors together (correlated noise), a degenerate code where $X_1 X_2 \in S$ treats this correlated error as "no error" — automatic protection!

---

### 6. Degeneracy in Surface Codes

Surface codes are highly degenerate — this is key to their success!

#### Local Degeneracy

In a surface code:
- X-stabilizers are products of X on plaquette boundaries
- Z-stabilizers are products of Z at vertices
- Small error loops create stabilizers: $E \in S$

#### Error Chains

Consider an error chain $E = X_1 X_2 X_3$ forming a path:
- The chain creates anyon pairs at endpoints
- A closed chain (loop) is a stabilizer
- Different chains with same endpoints are degenerate!

#### Decoding Implication

Surface code decoders don't need to identify the *exact* error — only the equivalence class. This is why **minimum-weight perfect matching** works:
- Find any error chain connecting syndrome defects
- Doesn't matter which chain — all are equivalent

---

### 7. Decoding with Degeneracy

#### Standard Decoder (Ignoring Degeneracy)

1. Measure syndrome $s$
2. Find minimum-weight error $E$ with syndrome $s$
3. Apply $E^\dagger$ as correction

**Problem:** May not be optimal if degenerate errors have different weights.

#### Degeneracy-Aware Decoder

1. Measure syndrome $s$
2. Find the **equivalence class** $[E]$ of errors with syndrome $s$
3. Apply any representative from $[E]$

**Advantage:** Can choose correction based on likelihood, not just weight.

#### Maximum Likelihood Decoding

For noise model with error probabilities $p(E)$:

$$\text{correction} = \arg\max_{[E]} \sum_{E' \in [E]} p(E')$$

Sum over the entire equivalence class — degeneracy-aware!

---

## Quantum Mechanics Connection

### Superposition and Equivalence

Degeneracy reflects a deep quantum principle:

**Classical:** Different errors are always distinguishable
**Quantum:** Errors differing by a stabilizer are *physically indistinguishable* on the code space

This is because stabilizers act as identity on encoded states — a manifestation of gauge symmetry in quantum error correction.

### Topological Protection

In topological codes (surface, toric), degeneracy is related to:
- Topological invariants (genus, boundary conditions)
- Anyon statistics and braiding
- Non-local encoding of quantum information

The high degeneracy of topological codes provides robust protection against local errors.

---

## Worked Examples

### Example 1: Degeneracy in Three-Qubit Code

**Problem:** Show that the three-qubit bit-flip code [[3,1,1]] is degenerate.

**Solution:**

Stabilizers: $S = \langle Z_1 Z_2, Z_2 Z_3 \rangle$

Consider errors $E_1 = I$ (no error) and "errors" within the stabilizer:
- $E_2 = Z_1 Z_2$
- $E_3 = Z_2 Z_3$
- $E_4 = Z_1 Z_3$

All of these produce syndrome $(0, 0)$!

Their products:
- $E_1^\dagger E_2 = Z_1 Z_2 \in S$
- $E_1^\dagger E_3 = Z_2 Z_3 \in S$
- etc.

**Conclusion:** The three-qubit code is degenerate — stabilizer elements are "trivial errors."

Wait, but we typically don't count stabilizers as errors. Let's look at actual correctable errors:

For X errors:
- $X_1$: syndrome (1, 0)
- $X_2$: syndrome (1, 1)
- $X_3$: syndrome (0, 1)

These are all distinct! So for single X errors, the code is non-degenerate.

The degeneracy appears when considering Z errors — but the bit-flip code doesn't correct Z errors at all.

**Proper analysis:** The [[3,1,1]] bit-flip code is non-degenerate for the errors it corrects (single X).

---

### Example 2: Shor Code Degeneracy

**Problem:** Find two distinct single-qubit errors on the Shor code with the same syndrome.

**Solution:**

The Shor code encodes: $|0_L\rangle = \frac{1}{2\sqrt{2}}(|000\rangle + |111\rangle)^{\otimes 3}$

Consider the first block (qubits 1, 2, 3).

The X-stabilizers for this block are $g_1 = X_1 X_2$ and $g_2 = X_2 X_3$.

For Z errors:
- $Z_1$: anticommutes with $g_1$ only → syndrome includes bit for $g_1$
- $Z_2$: anticommutes with $g_1$ AND $g_2$
- $Z_3$: anticommutes with $g_2$ only

All three have different X-syndromes within the block.

For the Z-stabilizer $g_7 = Z_1 Z_2 Z_3 Z_4 Z_5 Z_6$:
- Any single Z in first block anticommutes with $g_7$

So $Z_1, Z_2, Z_3$ all have the same syndrome component from $g_7$!

**Answer:** $Z_1, Z_2, Z_3$ have the same Z-stabilizer syndrome component (the inter-block part), though they differ in X-stabilizer syndromes. True degeneracy in Shor code appears in how correction works — any correction that flips the right "block parity" succeeds.

---

### Example 3: Counting Degeneracy

**Problem:** For the [[9,1,3]] Shor code, count the number of distinct syndrome values used by single-qubit errors.

**Solution:**

Single-qubit errors: 27 (9 qubits × 3 Pauli types) + 1 identity = 28 total

Shor code has 8 stabilizers → $2^8 = 256$ possible syndromes

For X errors:
- Block 1: $X_1, X_2, X_3$ → detected by $g_1, g_2$ (intra-block)
- Block 2: $X_4, X_5, X_6$ → detected by $g_3, g_4$
- Block 3: $X_7, X_8, X_9$ → detected by $g_5, g_6$
- No inter-block detection for X errors

For Z errors:
- All 9 single Z errors detected by $g_7, g_8$
- Plus intra-block detection differences

Counting distinct syndromes requires careful analysis...

**Rough estimate:** ~22-25 distinct syndromes for 28 single-qubit errors.

**Syndrome efficiency:** $\frac{28}{256} \approx 11\%$ — very "wasteful" by Hamming standards, but the extra syndromes allow degeneracy-aware decoding!

---

## Practice Problems

### Level 1: Direct Application

1. **Definition Check:**
   State the precise definition of a degenerate quantum code in terms of the stabilizer group.

2. **Simple Example:**
   For the 3-qubit phase-flip code, is it degenerate for Z errors? For X errors?

3. **Syndrome Counting:**
   The [[7,1,3]] Steane code has 6 stabilizers. How many syndromes? How many single-qubit errors?

### Level 2: Intermediate

4. **Product Analysis:**
   Given errors $E_1 = X_1 Z_2$ and $E_2 = X_1 Z_2 Z_3 Z_5$ on the Steane code, is $E_1^\dagger E_2$ a stabilizer?

5. **Decoder Design:**
   Describe how a degeneracy-aware decoder differs from a minimum-weight decoder. When does it matter?

6. **Surface Code Preview:**
   In a surface code, explain why error chains with the same endpoints are degenerate.

### Level 3: Challenging

7. **Hamming Bound Violation:**
   Show that degenerate codes can "violate" the Hamming bound's sphere-packing interpretation. Give an example.

8. **Optimal Decoding:**
   Derive the maximum-likelihood decoder formula that sums over equivalence classes. Why is this better than minimum-weight?

9. **Degeneracy and Distance:**
   Prove that if $E_1^\dagger E_2 \in S$, then $E_1$ and $E_2$ act identically on all codewords.

---

## Computational Lab

### Exploring Degeneracy

```python
"""
Day 696 Computational Lab: Degeneracy in Quantum Codes
Analysis of degenerate error classes
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from itertools import product
from collections import defaultdict

class PauliOperator:
    """Simple Pauli operator representation."""

    def __init__(self, paulis: List[int]):
        """
        Args:
            paulis: List of integers 0=I, 1=X, 2=Y, 3=Z
        """
        self.paulis = tuple(paulis)
        self.n = len(paulis)

    def __mul__(self, other: 'PauliOperator') -> 'PauliOperator':
        """Multiply Pauli operators (ignoring phase)."""
        # XZ multiplication table (result only, ignoring phase)
        # I*I=I, I*X=X, I*Y=Y, I*Z=Z
        # X*I=X, X*X=I, X*Y=Z, X*Z=Y
        # Y*I=Y, Y*X=Z, Y*Y=I, Y*Z=X
        # Z*I=Z, Z*X=Y, Z*Y=X, Z*Z=I
        mult = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        new_paulis = [mult[p1][p2] for p1, p2 in zip(self.paulis, other.paulis)]
        return PauliOperator(new_paulis)

    def __eq__(self, other: 'PauliOperator') -> bool:
        return self.paulis == other.paulis

    def __hash__(self) -> int:
        return hash(self.paulis)

    def __str__(self) -> str:
        symbols = ['I', 'X', 'Y', 'Z']
        return ''.join(symbols[p] for p in self.paulis)

    @property
    def weight(self) -> int:
        return sum(1 for p in self.paulis if p != 0)

    def commutes_with(self, other: 'PauliOperator') -> bool:
        """Check if commutes (ignoring phase)."""
        # Count anticommuting positions
        anticomm = 0
        for p1, p2 in zip(self.paulis, other.paulis):
            # X and Z anticommute, Y anticommutes with X and Z
            if (p1, p2) in [(1, 3), (3, 1), (1, 2), (2, 1), (2, 3), (3, 2)]:
                anticomm += 1
        return anticomm % 2 == 0


def create_steane_stabilizers() -> List[PauliOperator]:
    """Create Steane code stabilizer generators."""
    return [
        PauliOperator([0, 0, 0, 1, 1, 1, 1]),  # IIIXXXX
        PauliOperator([0, 1, 1, 0, 0, 1, 1]),  # IXXIIXX
        PauliOperator([1, 0, 1, 0, 1, 0, 1]),  # XIXIXIX
        PauliOperator([0, 0, 0, 3, 3, 3, 3]),  # IIIZZZZ
        PauliOperator([0, 3, 3, 0, 0, 3, 3]),  # IZZIIZZ
        PauliOperator([3, 0, 3, 0, 3, 0, 3]),  # ZIZIZIZ
    ]


def create_shor_stabilizers() -> List[PauliOperator]:
    """Create Shor code stabilizer generators."""
    return [
        PauliOperator([1, 1, 0, 0, 0, 0, 0, 0, 0]),  # XX...
        PauliOperator([0, 1, 1, 0, 0, 0, 0, 0, 0]),  # .XX..
        PauliOperator([0, 0, 0, 1, 1, 0, 0, 0, 0]),  # ...XX
        PauliOperator([0, 0, 0, 0, 1, 1, 0, 0, 0]),
        PauliOperator([0, 0, 0, 0, 0, 0, 1, 1, 0]),
        PauliOperator([0, 0, 0, 0, 0, 0, 0, 1, 1]),
        PauliOperator([3, 3, 3, 3, 3, 3, 0, 0, 0]),  # ZZZZZZ...
        PauliOperator([0, 0, 0, 3, 3, 3, 3, 3, 3]),  # ...ZZZZZZ
    ]


def generate_stabilizer_group(generators: List[PauliOperator]) -> Set[PauliOperator]:
    """Generate full stabilizer group from generators."""
    n = generators[0].n
    identity = PauliOperator([0] * n)

    group = {identity}
    for gen in generators:
        group.add(gen)

    # Generate all products
    changed = True
    while changed:
        changed = False
        new_elements = set()
        for g1 in group:
            for g2 in group:
                product = g1 * g2
                if product not in group:
                    new_elements.add(product)
                    changed = True
        group.update(new_elements)

    return group


def get_syndrome(error: PauliOperator, stabilizers: List[PauliOperator]) -> Tuple[int, ...]:
    """Calculate syndrome for an error."""
    syndrome = []
    for stab in stabilizers:
        # 0 if commutes, 1 if anticommutes
        commutes = error.commutes_with(stab)
        syndrome.append(0 if commutes else 1)
    return tuple(syndrome)


def analyze_degeneracy(code_name: str, stabilizers: List[PauliOperator]):
    """Analyze degeneracy of a stabilizer code."""

    n = stabilizers[0].n
    print(f"\n{'=' * 60}")
    print(f"DEGENERACY ANALYSIS: {code_name}")
    print(f"{'=' * 60}")

    # Generate stabilizer group
    print("\nGenerating stabilizer group...")
    stab_group = generate_stabilizer_group(stabilizers)
    print(f"Stabilizer group size: {len(stab_group)}")

    # Generate single-qubit errors
    single_errors = []
    for qubit in range(n):
        for pauli_type in [1, 2, 3]:  # X, Y, Z
            paulis = [0] * n
            paulis[qubit] = pauli_type
            single_errors.append(PauliOperator(paulis))

    print(f"Number of single-qubit errors: {len(single_errors)}")

    # Calculate syndromes
    syndrome_to_errors = defaultdict(list)
    for error in single_errors:
        syn = get_syndrome(error, stabilizers)
        syndrome_to_errors[syn].append(error)

    print(f"Number of distinct syndromes: {len(syndrome_to_errors)}")

    # Find degenerate pairs
    print("\n" + "-" * 40)
    print("DEGENERATE ERROR PAIRS:")
    print("-" * 40)

    degenerate_pairs = []
    for syn, errors in syndrome_to_errors.items():
        if len(errors) > 1:
            for i in range(len(errors)):
                for j in range(i + 1, len(errors)):
                    e1, e2 = errors[i], errors[j]
                    product = e1 * e2
                    is_stabilizer = product in stab_group

                    if is_stabilizer:
                        degenerate_pairs.append((e1, e2, product))
                        print(f"  {e1} and {e2}")
                        print(f"    Product: {product} ∈ S")

    if not degenerate_pairs:
        print("  No degenerate single-qubit error pairs found.")

    # Syndrome usage statistics
    print("\n" + "-" * 40)
    print("SYNDROME STATISTICS:")
    print("-" * 40)

    total_syndromes = 2 ** len(stabilizers)
    used_syndromes = len(syndrome_to_errors)
    unused_syndromes = total_syndromes - used_syndromes

    print(f"Total possible syndromes: {total_syndromes}")
    print(f"Used by single-qubit errors: {used_syndromes}")
    print(f"Unused syndromes: {unused_syndromes}")
    print(f"Efficiency: {100 * used_syndromes / total_syndromes:.1f}%")

    # Syndrome distribution
    print("\nSyndrome → Error count:")
    for syn, errors in sorted(syndrome_to_errors.items()):
        error_strs = [str(e) for e in errors]
        if len(errors) > 1:
            print(f"  {syn}: {len(errors)} errors (potential degeneracy)")
        else:
            print(f"  {syn}: {error_strs[0]}")


def compare_codes():
    """Compare degeneracy properties of different codes."""

    print("\n" + "=" * 60)
    print("CODE COMPARISON: DEGENERACY PROPERTIES")
    print("=" * 60)

    codes = [
        ("Steane [[7,1,3]]", create_steane_stabilizers()),
        ("Shor [[9,1,3]]", create_shor_stabilizers()),
    ]

    comparison_data = []

    for name, stabs in codes:
        n = stabs[0].n
        num_stabs = len(stabs)
        total_syn = 2 ** num_stabs
        single_errors = 3 * n

        # Count used syndromes
        syndrome_set = set()
        for qubit in range(n):
            for pauli_type in [1, 2, 3]:
                paulis = [0] * n
                paulis[qubit] = pauli_type
                error = PauliOperator(paulis)
                syn = get_syndrome(error, stabs)
                syndrome_set.add(syn)

        used_syn = len(syndrome_set)
        efficiency = 100 * single_errors / total_syn

        comparison_data.append({
            'name': name,
            'n': n,
            'stabilizers': num_stabs,
            'total_syndromes': total_syn,
            'single_errors': single_errors,
            'used_syndromes': used_syn,
            'efficiency': efficiency
        })

    print("\n| Code | n | Stabs | Syndromes | Errors | Used | Efficiency |")
    print("|------|---|-------|-----------|--------|------|------------|")
    for d in comparison_data:
        print(f"| {d['name']:<15} | {d['n']:>1} | {d['stabilizers']:>5} | "
              f"{d['total_syndromes']:>9} | {d['single_errors']:>6} | "
              f"{d['used_syndromes']:>4} | {d['efficiency']:>7.1f}% |")


def demonstrate_degeneracy_benefit():
    """Show how degeneracy helps error correction."""

    print("\n" + "=" * 60)
    print("DEGENERACY BENEFIT: CORRELATED NOISE")
    print("=" * 60)

    print("""
    Scenario: Qubits 1 and 2 often experience correlated X errors.

    NON-DEGENERATE CODE:
    ────────────────────
    - X₁ has syndrome s₁
    - X₂ has syndrome s₂
    - X₁X₂ (correlated) has syndrome s₃
    - Must distinguish all three for correct decoding

    DEGENERATE CODE (where X₁X₂ ∈ S):
    ──────────────────────────────────
    - X₁ has syndrome s₁
    - X₂ has syndrome s₁ (same!)
    - X₁X₂ (correlated) has syndrome 0 (no error!)

    The correlated error X₁X₂ is automatically corrected because
    it's a stabilizer — this is degeneracy protecting against
    correlated noise!

    This is why surface codes excel: local error chains that
    form loops are stabilizers, providing natural protection
    against spatially correlated noise.
    """)


if __name__ == "__main__":
    # Analyze both codes
    analyze_degeneracy("Steane [[7,1,3]]", create_steane_stabilizers())
    analyze_degeneracy("Shor [[9,1,3]]", create_shor_stabilizers())

    # Compare codes
    compare_codes()

    # Show benefit
    demonstrate_degeneracy_benefit()
```

---

## Summary

### Key Formulas

| Concept | Definition |
|---------|------------|
| Non-degenerate | All correctable errors have distinct syndromes AND actions |
| Degenerate | $\exists E_a \neq E_b$: $\text{syn}(E_a) = \text{syn}(E_b)$ and $E_a^\dagger E_b \in S$ |
| Stabilizer equivalence | $E_a \sim E_b \iff E_a^\dagger E_b \in S$ |
| Degeneracy-aware decoding | $\arg\max_{[E]} \sum_{E' \in [E]} p(E')$ |

### Main Takeaways

1. **Degeneracy Definition:** Multiple errors with same syndrome AND same code space action
2. **Stabilizer Criterion:** $E_a^\dagger E_b \in S$ means $E_a$ and $E_b$ are degenerate
3. **Beyond Hamming:** Degenerate codes can correct more errors than sphere-packing suggests
4. **Practical Codes:** Shor, Steane, and surface codes are all degenerate
5. **Decoding:** Degeneracy-aware decoders can improve logical error rates

---

## Daily Checklist

- [ ] Can define degenerate vs non-degenerate codes
- [ ] Understand the stabilizer criterion for degeneracy
- [ ] Can identify degeneracy in Shor and Steane codes
- [ ] Know why degeneracy helps error correction
- [ ] Understand degeneracy in surface codes
- [ ] Can describe degeneracy-aware decoding

---

## Preview: Day 697

Tomorrow we explore **Approximate Quantum Error Correction**:

- Relaxing Knill-Laflamme conditions
- Approximate codes and their advantages
- Bosonic codes (GKP)
- Near-optimal error correction

Approximate QEC bridges the gap between theoretical perfection and practical implementation!

---

*"Degeneracy is not a bug, it's a feature — the secret weapon of topological quantum codes."*
