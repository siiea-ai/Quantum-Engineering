# Day 715: Introduction to Subsystem Codes

## Overview

**Date:** Day 715 of 1008
**Week:** 103 (Subsystem Codes)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Generalizing Stabilizer Codes with Gauge Degrees of Freedom

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Subsystem code definitions and structure |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Gauge operators and gauge qubits |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Computational examples |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define subsystem codes** and distinguish them from subspace codes
2. **Identify gauge qubits** and their role in error correction
3. **Explain the gauge group** and its relationship to the stabilizer
4. **Construct simple subsystem codes** from examples
5. **Recognize the advantages** of gauge degrees of freedom
6. **Connect** subsystem codes to the stabilizer formalism

---

## Core Content

### 1. From Subspace Codes to Subsystem Codes

#### Review: Subspace (Stabilizer) Codes

In a standard $[[n, k, d]]$ stabilizer code:
- Code space $\mathcal{C}$ is a $2^k$-dimensional **subspace** of $\mathbb{C}^{2^n}$
- All $2^k$ dimensions encode logical information
- Stabilizer group $\mathcal{S}$ has $n - k$ generators

#### The New Idea: Subsystem Codes

In a subsystem code:
$$\mathcal{C} = \mathcal{A} \otimes \mathcal{B}$$

where:
- $\mathcal{A}$: **Logical subsystem** (encodes information)
- $\mathcal{B}$: **Gauge subsystem** (doesn't encode information)

**Key insight:** We can ignore errors that only affect $\mathcal{B}$!

---

### 2. Formal Definition

#### Subsystem Code Parameters

An $[[n, k, r, d]]$ subsystem code:
- $n$: physical qubits
- $k$: logical qubits (in $\mathcal{A}$)
- $r$: gauge qubits (in $\mathcal{B}$)
- $d$: distance

**Dimension relationship:**
$$\dim(\mathcal{C}) = 2^k \cdot 2^r = 2^{k+r}$$

#### Comparison with Stabilizer Codes

| Property | Stabilizer $[[n,k,d]]$ | Subsystem $[[n,k,r,d]]$ |
|----------|------------------------|-------------------------|
| Code dimension | $2^k$ | $2^{k+r}$ |
| Stabilizer generators | $n-k$ | $n-k-r$ |
| Gauge generators | 0 | $2r$ |
| Logical operators | $2k$ | $2k$ |

---

### 3. The Gauge Group

#### Definition

The **gauge group** $\mathcal{G}$ is a subgroup of the Pauli group with:
$$\mathcal{S} = \mathcal{G} \cap Z(\mathcal{G})$$

where $Z(\mathcal{G})$ is the center of $\mathcal{G}$ (elements that commute with everything in $\mathcal{G}$).

**Properties:**
- $\mathcal{G}$ may be non-abelian!
- Stabilizer $\mathcal{S}$ is always abelian (it's the center)
- Gauge operators $\mathcal{G} \setminus \mathcal{S}$ act on gauge qubits

#### Structure of the Gauge Group

$$\mathcal{G} = \langle \mathcal{S}, g_1, g_1', g_2, g_2', \ldots, g_r, g_r' \rangle$$

where:
- $\{g_i, g_i'\}$ are pairs of anticommuting gauge operators
- Each pair acts like $X, Z$ on one gauge qubit
- $[g_i, g_j] = [g_i', g_j'] = 0$ for $i \neq j$

---

### 4. Gauge Qubits

#### Intuition

Gauge qubits are "internal" degrees of freedom that:
- Are part of the code space
- Do NOT encode logical information
- Can be in any state without affecting the logical state

**Analogy:** Like choosing a gauge in electromagnetism — different choices give the same physics.

#### Formal Definition

The gauge subsystem $\mathcal{B}$ has dimension $2^r$ and is spanned by:
$$|b_1, b_2, \ldots, b_r\rangle_{\mathcal{B}}$$

A logical state $|\psi_L\rangle$ in the full code space:
$$|\psi_L\rangle \otimes |\text{any}\rangle_{\mathcal{B}}$$

All these states represent the **same** logical information!

---

### 5. Error Correction in Subsystem Codes

#### The Key Advantage

We only need to correct errors that affect the logical subsystem $\mathcal{A}$!

**Correctable errors:** Any error $E$ such that $E$ doesn't map between distinguishable logical states.

**Formally:** Error $E$ is correctable if for all logical states $|\psi\rangle, |\phi\rangle$:
$$\langle\psi| \otimes \langle b| E^\dagger E |\phi\rangle \otimes |b'\rangle = c_E \delta_{\psi\phi}$$

(The gauge states $|b\rangle, |b'\rangle$ can differ!)

#### Syndrome Measurement

Can measure **gauge operators** instead of stabilizer generators!

**Advantage:** Gauge operators may be lower weight than stabilizers.

---

### 6. First Example: The [[4,1,1,2]] Code

#### Construction

Take the 4-qubit code with:
- **Gauge group:** $\mathcal{G} = \langle X_1X_2, X_3X_4, Z_1Z_2, Z_3Z_4 \rangle$
- **Stabilizer:** $\mathcal{S} = \langle X_1X_2X_3X_4, Z_1Z_2Z_3Z_4 \rangle$

**Note:** $X_1X_2$ and $Z_1Z_2$ anticommute, so $\mathcal{G}$ is non-abelian!

#### Analysis

- $n = 4$ physical qubits
- Stabilizer has 2 generators → $2^{4-2} = 4$ dimensional code space
- But only $k = 1$ logical qubit
- Therefore $r = 1$ gauge qubit

**Logical operators:** $\bar{X} = X_1X_3$, $\bar{Z} = Z_1Z_3$

#### Why It's Useful

Gauge generators $X_1X_2, Z_1Z_2$ are weight-2 (local).
Stabilizer generators $X_1X_2X_3X_4$ are weight-4 (non-local).

**Measuring weight-2 operators is easier!**

---

### 7. Relationship to Stabilizer Codes

#### Every Stabilizer Code is a Subsystem Code

A stabilizer code $[[n, k, d]]$ is a subsystem code $[[n, k, 0, d]]$ with no gauge qubits.

#### Gauge Fixing

Given a subsystem code, we can "fix the gauge" to get a stabilizer code:
1. Choose a state for each gauge qubit
2. Add corresponding gauge operators to stabilizer
3. Result: stabilizer code with more logical qubits

**Trade-off:** Fixing gauge → more logical qubits, but lose subsystem advantages.

---

### 8. Why Use Subsystem Codes?

#### Advantages

1. **Simpler syndrome measurement:** Gauge operators can be lower weight
2. **Fault tolerance:** Easier to implement fault-tolerant measurements
3. **Single-shot error correction:** Some subsystem codes allow this
4. **Flexibility:** Can choose gauge state based on convenience

#### Disadvantages

1. **Fewer logical qubits:** Pay with gauge qubits
2. **More complex structure:** Need to track gauge group
3. **Potentially lower rate:** $k/(n) < (k+r)/n$ for subspace codes

---

## Worked Examples

### Example 1: Verify the [[4,1,1,2]] Code Structure

**Problem:** Confirm that $\mathcal{S}$ is the center of $\mathcal{G}$ for the [[4,1,1,2]] code.

**Solution:**

Gauge generators: $X_1X_2, X_3X_4, Z_1Z_2, Z_3Z_4$

**Check commutation:**

| | $X_1X_2$ | $X_3X_4$ | $Z_1Z_2$ | $Z_3Z_4$ |
|---|:---:|:---:|:---:|:---:|
| $X_1X_2$ | ✓ | ✓ | ✗ | ✓ |
| $X_3X_4$ | ✓ | ✓ | ✓ | ✗ |
| $Z_1Z_2$ | ✗ | ✓ | ✓ | ✓ |
| $Z_3Z_4$ | ✓ | ✗ | ✓ | ✓ |

**Center elements:** Must commute with all generators.

$X_1X_2 \cdot X_3X_4 = X_1X_2X_3X_4$: commutes with all ✓
$Z_1Z_2 \cdot Z_3Z_4 = Z_1Z_2Z_3Z_4$: commutes with all ✓

Therefore: $\mathcal{S} = \langle X_1X_2X_3X_4, Z_1Z_2Z_3Z_4 \rangle = Z(\mathcal{G})$ ✓

---

### Example 2: Count Logical and Gauge Qubits

**Problem:** A subsystem code has $n = 9$ physical qubits, gauge group with 8 generators (4 pairs of anticommuting operators), and stabilizer with 4 generators. Find $k$ and $r$.

**Solution:**

**From gauge structure:**
- 4 pairs of anticommuting generators → $r = 4$ gauge qubits
- Each pair contributes 2 generators, but only 1 independent after removing products

**From stabilizer:**
- 4 stabilizer generators
- Code space dimension: $2^{9-4} = 2^5 = 32$
- This equals $2^k \cdot 2^r = 2^{k+4}$
- Therefore $k + 4 = 5$, so $k = 1$

**Answer:** $k = 1$ logical qubit, $r = 4$ gauge qubits. This is a $[[9, 1, 4, d]]$ code.

---

### Example 3: Gauge Fixing

**Problem:** Fix the gauge of the [[4,1,1,2]] code to obtain a [[4,2,2]] stabilizer code.

**Solution:**

Original:
- Stabilizer: $\mathcal{S} = \langle X_1X_2X_3X_4, Z_1Z_2Z_3Z_4 \rangle$
- Gauge: $X_1X_2, Z_1Z_2$ (one pair acting on gauge qubit)

**Fix gauge by adding $Z_1Z_2$ to stabilizer:**

New stabilizer: $\mathcal{S}' = \langle X_1X_2X_3X_4, Z_1Z_2Z_3Z_4, Z_1Z_2 \rangle$

But wait: $Z_1Z_2 \cdot Z_1Z_2Z_3Z_4 = Z_3Z_4$, so we can simplify:

$\mathcal{S}' = \langle X_1X_2X_3X_4, Z_1Z_2, Z_3Z_4 \rangle$

**Check:** 3 generators → $2^{4-3} = 2$ dimensional code space → $k = 1$?

Actually, we need to be more careful. The original has $X_1X_2$ anticommuting with $Z_1Z_2$, so adding $Z_1Z_2$ kills the gauge qubit.

**Result:** [[4,2,2]] stabilizer code with stabilizer $\langle X_1X_2X_3X_4, Z_1Z_2Z_3Z_4 \rangle$.

---

## Practice Problems

### Direct Application

1. **Problem 1:** For an $[[n, k, r, d]]$ subsystem code, how many generators does the stabilizer have?

2. **Problem 2:** If gauge group $\mathcal{G}$ has 10 generators and stabilizer $\mathcal{S}$ has 6 generators, how many gauge qubits are there?

3. **Problem 3:** Verify that $X_1X_2$ and $Z_1Z_2$ anticommute and thus cannot both be in the stabilizer.

### Intermediate

4. **Problem 4:** Construct the gauge group for a code with $n = 6$, $k = 1$, $r = 2$ gauge qubits.

5. **Problem 5:** Prove that the center of a group is always a subgroup.

6. **Problem 6:** Show that if $\mathcal{G}$ is abelian, then $\mathcal{S} = \mathcal{G}$ and we have a standard stabilizer code.

### Challenging

7. **Problem 7:** Design a subsystem code with weight-2 gauge operators but weight-4 stabilizer generators.

8. **Problem 8:** Prove that gauge fixing always increases (or maintains) the number of logical qubits.

9. **Problem 9:** Analyze the error correction properties of the [[4,1,1,2]] code. What is its distance?

---

## Computational Lab

```python
"""
Day 715: Introduction to Subsystem Codes
Week 103: Subsystem Codes

Implements basic subsystem code structures.
"""

import numpy as np
from typing import List, Tuple, Set
from itertools import combinations

class PauliOperator:
    """Represents an n-qubit Pauli operator."""

    def __init__(self, n: int, x_bits: List[int], z_bits: List[int], phase: int = 0):
        """
        Initialize Pauli operator.

        x_bits[i] = 1 means X on qubit i
        z_bits[i] = 1 means Z on qubit i
        phase: 0,1,2,3 for 1,i,-1,-i
        """
        self.n = n
        self.x = np.array(x_bits, dtype=int)
        self.z = np.array(z_bits, dtype=int)
        self.phase = phase % 4

    def __str__(self):
        phase_str = ['', 'i', '-', '-i'][self.phase]
        paulis = []
        for i in range(self.n):
            if self.x[i] == 0 and self.z[i] == 0:
                paulis.append('I')
            elif self.x[i] == 1 and self.z[i] == 0:
                paulis.append('X')
            elif self.x[i] == 0 and self.z[i] == 1:
                paulis.append('Z')
            else:
                paulis.append('Y')
        return phase_str + ''.join(paulis)

    def commutes_with(self, other: 'PauliOperator') -> bool:
        """Check if this operator commutes with other."""
        # Symplectic inner product
        inner = (np.dot(self.x, other.z) + np.dot(self.z, other.x)) % 2
        return inner == 0

    def multiply(self, other: 'PauliOperator') -> 'PauliOperator':
        """Multiply two Pauli operators."""
        new_x = (self.x + other.x) % 2
        new_z = (self.z + other.z) % 2

        # Phase calculation (simplified)
        phase_contrib = np.dot(self.z, other.x) % 2
        new_phase = (self.phase + other.phase + 2 * phase_contrib) % 4

        return PauliOperator(self.n, list(new_x), list(new_z), new_phase)

    def weight(self) -> int:
        """Return the weight (number of non-identity positions)."""
        return sum((self.x + self.z) > 0)


class SubsystemCode:
    """Represents a subsystem quantum error correcting code."""

    def __init__(self, n: int, gauge_generators: List[PauliOperator]):
        """
        Initialize subsystem code from gauge generators.

        n: number of physical qubits
        gauge_generators: generators of the gauge group
        """
        self.n = n
        self.gauge_generators = gauge_generators
        self.gauge_group = self._generate_group(gauge_generators)
        self.stabilizer = self._find_center()

    def _generate_group(self, generators: List[PauliOperator],
                       max_size: int = 1000) -> List[PauliOperator]:
        """Generate group from generators (up to max_size)."""
        # Simplified: just return generators for now
        # Full implementation would generate all products
        return generators

    def _find_center(self) -> List[PauliOperator]:
        """Find the center of the gauge group (stabilizer)."""
        center = []

        # For each generator, check if it commutes with all others
        for i, g in enumerate(self.gauge_generators):
            commutes_all = True
            for j, h in enumerate(self.gauge_generators):
                if not g.commutes_with(h):
                    commutes_all = False
                    break

            if commutes_all:
                center.append(g)

        # Also check products of generators
        for i in range(len(self.gauge_generators)):
            for j in range(i+1, len(self.gauge_generators)):
                product = self.gauge_generators[i].multiply(self.gauge_generators[j])
                commutes_all = True
                for h in self.gauge_generators:
                    if not product.commutes_with(h):
                        commutes_all = False
                        break
                if commutes_all:
                    # Check if already in center
                    is_new = True
                    for c in center:
                        if np.array_equal(product.x, c.x) and np.array_equal(product.z, c.z):
                            is_new = False
                            break
                    if is_new:
                        center.append(product)

        return center

    def count_gauge_qubits(self) -> int:
        """Count the number of gauge qubits."""
        # Count pairs of anticommuting gauge generators
        pairs = 0
        used = set()

        for i, g in enumerate(self.gauge_generators):
            if i in used:
                continue
            for j, h in enumerate(self.gauge_generators):
                if j <= i or j in used:
                    continue
                if not g.commutes_with(h):
                    pairs += 1
                    used.add(i)
                    used.add(j)
                    break

        return pairs

    def print_structure(self):
        """Print the code structure."""
        print(f"Subsystem Code on {self.n} qubits")
        print("-" * 40)

        print("\nGauge generators:")
        for i, g in enumerate(self.gauge_generators):
            print(f"  G_{i+1} = {g} (weight {g.weight()})")

        print("\nStabilizer (center of gauge group):")
        for i, s in enumerate(self.stabilizer):
            print(f"  S_{i+1} = {s} (weight {s.weight()})")

        print("\nCommutation table for gauge generators:")
        print("     ", end="")
        for i in range(len(self.gauge_generators)):
            print(f" G_{i+1} ", end="")
        print()

        for i, g in enumerate(self.gauge_generators):
            print(f" G_{i+1} ", end="")
            for h in self.gauge_generators:
                comm = "✓" if g.commutes_with(h) else "✗"
                print(f"  {comm}  ", end="")
            print()


def demonstrate_subsystem_codes():
    """Demonstrate subsystem code concepts."""

    print("=" * 70)
    print("INTRODUCTION TO SUBSYSTEM CODES")
    print("=" * 70)

    # Example 1: [[4,1,1,2]] code
    print("\n1. THE [[4,1,1,2]] SUBSYSTEM CODE")
    print("-" * 50)

    # Define gauge generators
    n = 4
    # X1X2
    g1 = PauliOperator(n, [1,1,0,0], [0,0,0,0])
    # X3X4
    g2 = PauliOperator(n, [0,0,1,1], [0,0,0,0])
    # Z1Z2
    g3 = PauliOperator(n, [0,0,0,0], [1,1,0,0])
    # Z3Z4
    g4 = PauliOperator(n, [0,0,0,0], [0,0,1,1])

    code_411 = SubsystemCode(n, [g1, g2, g3, g4])
    code_411.print_structure()

    r = code_411.count_gauge_qubits()
    print(f"\nGauge qubits: r = {r}")

    # Calculate k
    n_stab = len(code_411.stabilizer)
    dim_code = 2**(n - n_stab)
    k = int(np.log2(dim_code)) - r
    print(f"Logical qubits: k = {k}")
    print(f"This is an [[{n},{k},{r},d]] subsystem code")

    # Show advantage
    print("\n2. ADVANTAGE OF SUBSYSTEM CODES")
    print("-" * 50)

    print("\nGauge operators are weight-2:")
    for g in [g1, g2, g3, g4]:
        print(f"  {g}: weight {g.weight()}")

    print("\nStabilizer operators are weight-4:")
    for s in code_411.stabilizer:
        print(f"  {s}: weight {s.weight()}")

    print("\n  → Can measure weight-2 gauge operators instead of weight-4 stabilizers!")

    # Compare to stabilizer codes
    print("\n3. COMPARISON: SUBSYSTEM vs SUBSPACE CODES")
    print("-" * 50)

    print("""
    Subspace (Stabilizer) Code [[n,k,d]]:
    - Code space = 2^k dimensions
    - All dimensions encode information
    - Stabilizer has n-k generators

    Subsystem Code [[n,k,r,d]]:
    - Code space = 2^(k+r) dimensions
    - Only 2^k dimensions encode information
    - 2^r dimensions are "gauge freedom"
    - Stabilizer has n-k-r generators

    Trade-off: Fewer logical qubits, but simpler operations!
    """)

    # Gauge fixing example
    print("\n4. GAUGE FIXING")
    print("-" * 50)

    print("""
    Starting with [[4,1,1,2]] subsystem code:
    - 1 logical qubit, 1 gauge qubit

    Fix gauge by adding Z1Z2 to stabilizer:
    - Gauge qubit fixed to |0⟩ eigenstate of Z1Z2
    - Result: [[4,2,2]] stabilizer code
    - Now have 2 logical qubits!

    Trade-off: Lost the weight-2 measurement advantage.
    """)


if __name__ == "__main__":
    demonstrate_subsystem_codes()
```

---

## Summary

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Subsystem code** | $\mathcal{C} = \mathcal{A} \otimes \mathcal{B}$ (logical ⊗ gauge) |
| **Gauge group** | Non-abelian group $\mathcal{G}$ with $\mathcal{S} = Z(\mathcal{G})$ |
| **Gauge qubits** | Degrees of freedom that don't encode information |
| **Parameters** | $[[n, k, r, d]]$ — physical, logical, gauge, distance |
| **Gauge fixing** | Convert to stabilizer code by adding gauge operators to $\mathcal{S}$ |

### Main Takeaways

1. **Subsystem codes** generalize stabilizer codes with gauge degrees of freedom
2. **Gauge group** may be non-abelian; stabilizer is its center
3. **Gauge qubits** provide flexibility without affecting logical information
4. **Simpler measurements** — gauge operators can have lower weight
5. **Trade-off** — fewer logical qubits for operational advantages

---

## Daily Checklist

- [ ] Define subsystem codes and parameters $[[n,k,r,d]]$
- [ ] Explain the gauge group and its center
- [ ] Identify gauge qubits in examples
- [ ] Understand gauge fixing
- [ ] Recognize advantages of subsystem codes
- [ ] Connect to stabilizer formalism

---

## Preview: Day 716

Tomorrow we dive deeper into **Gauge Operators and Gauge Qubits**, covering:
- Formal properties of gauge operators
- Gauge transformations
- The bare logical operators
- Dressed vs bare operators
