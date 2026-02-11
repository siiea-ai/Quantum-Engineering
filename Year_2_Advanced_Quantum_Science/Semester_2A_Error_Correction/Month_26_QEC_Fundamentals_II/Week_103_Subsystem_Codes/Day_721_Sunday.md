# Day 721: Week 103 Synthesis — Subsystem Codes

## Overview

**Date:** Day 721 of 1008
**Week:** 103 (Subsystem Codes)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Comprehensive Review and Integration

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Concept review and integration |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Comprehensive problem set |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Connections to advanced topics |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Synthesize** all subsystem code concepts from the week
2. **Solve** problems spanning the full range of difficulty
3. **Connect** subsystem codes to advanced QEC topics
4. **Evaluate** when to use subsystem vs stabilizer codes
5. **Design** complete subsystem code systems
6. **Prepare** for the transition to code capacity (Week 104)

---

## Week 103 Concept Map

```
                    SUBSYSTEM CODES
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    CODE STRUCTURE    GAUGE GROUP    ADVANTAGES
          │               │               │
    ┌─────┴─────┐    ┌────┴────┐    ┌────┴────┐
    ▼           ▼    ▼         ▼    ▼         ▼
 C = A⊗B    [[n,k,r,d]]  G ⊂ Pn   S=Z(G)  Weight   Fault
 Logical    Parameters   Gauge    Stab    Reduction  Tolerance
 ⊗ Gauge                 Ops
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              BACON-SHOR  PROPERTIES  APPLICATIONS
                    │         │         │
              m×n lattice  Distance   Single-shot
              Weight-2    Singleton   Universal QC
              3×3=Shor    Bounds     Threshold
```

---

## Core Concepts Review

### 1. Subsystem Code Structure

**The Fundamental Decomposition:**
$$\mathcal{H} = \mathcal{C} \oplus \mathcal{C}^\perp = (A \otimes B) \oplus \mathcal{C}^\perp$$

| Component | Role | Dimension |
|-----------|------|-----------|
| $A$ | Logical subsystem (protected) | $2^k$ |
| $B$ | Gauge subsystem (unprotected) | $2^r$ |
| $\mathcal{C}^\perp$ | Non-code space | $2^n - 2^{k+r}$ |

**Key equation:**
$$k + r + \text{stabilizer generators} = n$$

### 2. Gauge and Stabilizer Groups

**Hierarchy:**
$$\mathcal{S} = Z(\mathcal{G}) \cap \mathcal{G} \subseteq \mathcal{G} \subseteq N(\mathcal{G}) \subseteq \mathcal{P}_n$$

| Group | Definition | Generators |
|-------|------------|------------|
| $\mathcal{G}$ | Gauge group | $n - k$ independent |
| $\mathcal{S}$ | Stabilizer = center of $\mathcal{G}$ | $n - k - r$ independent |
| $N(\mathcal{G})$ | Normalizer | Contains logical ops |

### 3. Bacon-Shor Code Family

**Parameters:** $[[mn, 1, (m-1)(n-1), \min(m,n)]]$

| Property | Formula |
|----------|---------|
| Physical qubits | $n \cdot m$ |
| Logical qubits | $1$ |
| Gauge qubits | $(m-1)(n-1)$ |
| Distance | $\min(m, n)$ |
| X-gauges | $m(n-1)$ |
| Z-gauges | $(m-1)n$ |
| X-stabilizers | $n-1$ |
| Z-stabilizers | $m-1$ |

### 4. Error Correction

**Subsystem Knill-Laflamme:**
$$P_{\mathcal{C}} E_a^\dagger E_b P_{\mathcal{C}} = I_A \otimes B_{ab}$$

**Key difference from stabilizer codes:** Allows operators on gauge subsystem, not just scalars.

### 5. Advantages Summary

| Advantage | Mechanism | Benefit |
|-----------|-----------|---------|
| Weight reduction | Gauge ops are low-weight | Simpler circuits |
| Fault tolerance | Weight-2 measurements | Natural FT |
| Single-shot | Gauge redundancy | Fewer rounds |
| Threshold | Fewer gates | Higher threshold |

---

## Master Formula Sheet

### Code Parameters
$$\boxed{[[n, k, r, d]]: \quad k + r = n - |\mathcal{S}|}$$

### Singleton Bound
$$\boxed{k + r \leq n - 2(d-1)}$$

### Bacon-Shor Specifics
$$\boxed{d = \min(m, n), \quad r = (m-1)(n-1)}$$

### Gauge-Stabilizer Relation
$$\boxed{\mathcal{S} = Z(\mathcal{G}) \cap \mathcal{G}}$$

### Error Correction Condition
$$\boxed{P E_a^\dagger E_b P = I_A \otimes B_{ab} \text{ for correctable } E_a, E_b}$$

### Logical Operators
$$\boxed{\bar{X}, \bar{Z} \in N(\mathcal{G}) \setminus \mathcal{G}}$$

### Dressed vs Bare
$$\boxed{d = d_{\text{dressed}} \leq d_{\text{bare}}}$$

---

## Comprehensive Problem Set

### Part A: Foundations (Days 715-716)

**A1.** For a general subsystem code with $n = 10$ physical qubits, $k = 2$ logical qubits, and $r = 3$ gauge qubits:
a) How many stabilizer generators are there?
b) How many gauge generators are there?
c) What is the maximum distance (Singleton bound)?

**A2.** Prove that if $g_1, g_2 \in \mathcal{G}$ (gauge group), then $[g_1, g_2] = 0$ or $[g_1, g_2] = 2g_1g_2$.

**A3.** For the $[[4, 1, 1, 2]]$ subsystem code with gauge group $\mathcal{G} = \langle X_1X_2, X_3X_4, Z_1Z_3, Z_2Z_4 \rangle$:
a) Find the stabilizer group $\mathcal{S}$.
b) Verify that $\bar{X} = X_1X_3$ and $\bar{Z} = Z_1Z_2$ are valid logical operators.
c) Show that $X_1$ is a dressed logical operator (identify the gauge dressing).

---

### Part B: Bacon-Shor Codes (Day 717)

**B1.** For a $5 \times 3$ Bacon-Shor code:
a) List all parameters $[[n, k, r, d]]$.
b) How many gauge operators of each type?
c) What is the weight of the stabilizers?
d) Verify the Singleton bound.

**B2.** Prove that the $3 \times 3$ Bacon-Shor code can be gauge-fixed to produce the Shor [[9,1,3]] code by:
a) Identifying which gauge operators to add to the stabilizer.
b) Counting that you get 8 stabilizer generators (correct for [[9,1,3]]).

**B3.** For the $2 \times 4$ Bacon-Shor code:
a) Draw the lattice and label all qubits.
b) Write out all X-gauge and Z-gauge operators explicitly.
c) Write out all X-stabilizer and Z-stabilizer generators.
d) Find the bare logical operators $\bar{X}$ and $\bar{Z}$.

---

### Part C: Code Properties (Day 718)

**C1.** Prove that for any subsystem code, the distance satisfies:
$$d \leq n - k - r + 2$$

**C2.** Show that the $[[9, 1, 4, 3]]$ Bacon-Shor code saturates the Singleton bound.

**C3.** For a code with stabilizer syndrome $(s_1, s_2, s_3)$ where:
- $s_1 = $ product of Z-gauges in column 1
- $s_2 = $ product of Z-gauges in column 2
- $s_3 = $ product of Z-gauges in column 3

If an X error occurs on qubit $(2, 2)$:
a) Which individual Z-gauge measurements are affected?
b) Which stabilizer syndromes are affected?
c) How can you uniquely identify the error location?

---

### Part D: Advantages (Day 719)

**D1.** Compare the fault-tolerance properties:

For measuring a weight-6 stabilizer directly vs. via three weight-2 gauge operators:
a) Calculate the maximum error spread from a single fault in each case.
b) How many CNOTs are needed in each case?
c) Which approach has better fault tolerance and why?

**D2.** A quantum computer has the following constraints:
- Physical error rate: 0.5%
- Only nearest-neighbor connectivity on a 2D grid
- Flag qubits are expensive

Argue whether a stabilizer code or subsystem code would be preferable.

**D3.** Explain why the Bacon-Shor code has partial single-shot capability for one error type but not the other. What would be needed for full single-shot?

---

### Part E: Fault Tolerance (Day 720)

**E1.** Design the complete fault-tolerant syndrome extraction circuit for the $2 \times 2$ Bacon-Shor code:
a) Draw the circuits for all 4 gauge measurements.
b) Show how to compute the 2 stabilizer syndromes from gauge outcomes.
c) Analyze what happens if there's a single X fault on the ancilla during the first Z-gauge measurement.

**E2.** Prove that transversal $H^{\otimes 9}$ implements logical $\bar{H}$ on the $3 \times 3$ Bacon-Shor code by showing:
a) How X-gauges transform under $H$.
b) How Z-gauges transform under $H$.
c) How $\bar{X}$ and $\bar{Z}$ transform.

**E3.** For magic state distillation on Bacon-Shor:
a) Why can't we implement $T$ transversally?
b) Sketch the state injection protocol using an ancilla $|T\rangle$ state.
c) What Clifford gate might be needed for correction?

---

### Part F: Integration and Design (Comprehensive)

**F1. Design Challenge:** Create a subsystem code with the following properties:
- At least distance 4
- Weight-2 gauge operators only
- Minimum physical qubits possible

Justify your design and compute all parameters.

**F2. Analysis Challenge:** A researcher claims to have a $[[16, 2, 6, 4]]$ subsystem code.
a) Check if this satisfies the Singleton bound.
b) If it violates the bound, explain why such a code cannot exist.
c) What is the maximum $r$ for a $[[16, 2, ?, 4]]$ code?

**F3. Application Challenge:** You need to implement a fault-tolerant Toffoli gate on a Bacon-Shor encoded system.
a) What resource states would you need?
b) Sketch the protocol (high-level).
c) What is the dominant resource cost?

---

## Solutions to Selected Problems

### Solution A1

a) Stabilizer generators: $n - k - r = 10 - 2 - 3 = 5$

b) Gauge generators: $n - k = 10 - 2 = 8$
(Since stabilizers are derived from gauge, we have 8 gauge generators generating $\mathcal{G}$, with 5 generating $\mathcal{S}$.)

c) Singleton bound: $k + r \leq n - 2(d-1)$
$2 + 3 \leq 10 - 2(d-1)$
$5 \leq 10 - 2d + 2$
$2d \leq 7$
$d \leq 3.5$
So $d_{\max} = 3$.

### Solution B1

$5 \times 3$ Bacon-Shor:
a) $n = 15$, $k = 1$, $r = (5-1)(3-1) = 8$, $d = \min(5,3) = 3$
   Parameters: $[[15, 1, 8, 3]]$

b) X-gauges: $5 \times (3-1) = 10$
   Z-gauges: $(5-1) \times 3 = 12$

c) X-stabilizers: weight $2 \times 5 = 10$ (span 2 columns, all 5 rows)
   Z-stabilizers: weight $2 \times 3 = 6$ (span 2 rows, all 3 columns)

d) Singleton: $1 + 8 \leq 15 - 2(3-1) = 15 - 4 = 11$ ✓

### Solution C2

$[[9, 1, 4, 3]]$:
Singleton: $k + r \leq n - 2(d-1)$
$1 + 4 \leq 9 - 2(3-1)$
$5 \leq 9 - 4 = 5$ ✓

Equality holds — the code saturates the bound.

### Solution E2

$H^{\otimes 9}$ on $3 \times 3$ Bacon-Shor:

a) X-gauges: $X_{i,j}X_{i,j+1} \xrightarrow{H} Z_{i,j}Z_{i,j+1}$
   These become "horizontal ZZ" operators.

b) Z-gauges: $Z_{i,j}Z_{i+1,j} \xrightarrow{H} X_{i,j}X_{i+1,j}$
   These become "vertical XX" operators.

c) For square lattice, horizontal ↔ vertical swap is a symmetry.
   Under this transformation:
   - X-gauge ↔ Z-gauge (different direction)
   - Gauge group maps to itself

   Logical operators:
   - $\bar{X} = X^{\otimes 3}_{\text{row}} \xrightarrow{H} Z^{\otimes 3}_{\text{row}} = \bar{Z}$ (in row basis)
   - $\bar{Z} = Z^{\otimes 3}_{\text{col}} \xrightarrow{H} X^{\otimes 3}_{\text{col}} = \bar{X}$ (in column basis)

   This is exactly $\bar{H}: \bar{X} \leftrightarrow \bar{Z}$. ✓

---

## Connections to Advanced Topics

### 1. LDPC Subsystem Codes

**Low-Density Parity-Check (LDPC)** codes have sparse check matrices.

**Subsystem LDPC codes:**
- Gauge operators are low-weight (like Bacon-Shor)
- Stabilizers may be higher-weight but sparse
- Key examples: hypergraph product codes, fiber bundle codes

**Advantage:** Combine LDPC efficiency with subsystem fault tolerance.

### 2. Topological Subsystem Codes

**3D Gauge Color Code:**
- Gauge operators on faces
- Stabilizers on volumes
- Achieves single-shot error correction
- Important for fault-tolerant architectures

**Subsystem Surface Codes:**
- Variants of surface code with gauge structure
- Different trade-offs in measurement vs encoding

### 3. Floquet Codes

**Dynamic subsystem codes:**
- Gauge operators measured in sequence
- Code structure emerges from measurement pattern
- Example: Hastings-Haah Floquet code

**Key insight:** Time evolution of gauge measurements can create effective error correction.

### 4. Quantum LDPC Revolution

Recent breakthroughs (2020s):
- **Good qLDPC codes:** $[[n, k, d]]$ with $k, d = \Theta(n)$
- Many are naturally subsystem codes
- Potential for constant-overhead fault tolerance

**Connection:** Subsystem structure enables efficient syndrome extraction even for complex codes.

---

## Week 103 Summary Table

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 715 | Introduction | $\mathcal{C} = A \otimes B$, $[[n,k,r,d]]$, gauge group |
| 716 | Gauge Operators | Bare vs dressed, gauge transformations, logical operators |
| 717 | Bacon-Shor | $m \times n$ lattice, weight-2 gauges, Shor connection |
| 718 | Properties | Distance, Singleton bound, error correction conditions |
| 719 | Advantages | Weight reduction, fault tolerance, single-shot |
| 720 | Fault Tolerance | FT gadgets, transversal gates, threshold |
| 721 | Synthesis | Integration, comprehensive problems, advanced connections |

---

## Computational Lab: Complete Subsystem Code Toolkit

```python
"""
Day 721: Week 103 Synthesis
Complete subsystem code analysis toolkit
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import combinations, product

@dataclass
class SubsystemCodeParams:
    """Complete parameters for a subsystem code."""
    n: int  # Physical qubits
    k: int  # Logical qubits
    r: int  # Gauge qubits
    d: int  # Distance
    name: str = ""

    def __post_init__(self):
        self.num_stabilizers = self.n - self.k - self.r
        self.num_gauge_generators = self.n - self.k

    def check_singleton(self) -> Tuple[bool, int]:
        """Check Singleton bound and return max distance."""
        d_max = (self.n - self.k - self.r) // 2 + 1
        satisfies = self.d <= d_max
        return satisfies, d_max

    def __str__(self):
        name_str = f" ({self.name})" if self.name else ""
        return f"[[{self.n}, {self.k}, {self.r}, {self.d}]]{name_str}"

class BaconShorCode:
    """Complete Bacon-Shor code implementation."""

    def __init__(self, m: int, n: int):
        self.m = m
        self.n_col = n
        self.n_qubits = m * n

        # Compute parameters
        self.k = 1
        self.r = (m - 1) * (n - 1)
        self.d = min(m, n)

        self.params = SubsystemCodeParams(
            self.n_qubits, self.k, self.r, self.d,
            f"{m}×{n} Bacon-Shor"
        )

        # Generate operators
        self._generate_gauge_operators()
        self._generate_stabilizers()
        self._generate_logical_operators()

    def _qubit_index(self, i: int, j: int) -> int:
        """Convert (row, col) to linear index."""
        return i * self.n_col + j

    def _qubit_coords(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to (row, col)."""
        return idx // self.n_col, idx % self.n_col

    def _generate_gauge_operators(self):
        """Generate all gauge operators."""
        self.x_gauges = []
        self.z_gauges = []

        # X-gauges (horizontal XX)
        for i in range(self.m):
            for j in range(self.n_col - 1):
                q1 = self._qubit_index(i, j)
                q2 = self._qubit_index(i, j + 1)
                self.x_gauges.append({
                    'qubits': [q1, q2],
                    'type': 'X',
                    'coords': [(i, j), (i, j + 1)]
                })

        # Z-gauges (vertical ZZ)
        for i in range(self.m - 1):
            for j in range(self.n_col):
                q1 = self._qubit_index(i, j)
                q2 = self._qubit_index(i + 1, j)
                self.z_gauges.append({
                    'qubits': [q1, q2],
                    'type': 'Z',
                    'coords': [(i, j), (i + 1, j)]
                })

    def _generate_stabilizers(self):
        """Generate stabilizer generators from gauge products."""
        self.x_stabilizers = []
        self.z_stabilizers = []

        # X-stabilizers (column pairs)
        for j in range(self.n_col - 1):
            qubits = []
            for i in range(self.m):
                qubits.extend([self._qubit_index(i, j), self._qubit_index(i, j + 1)])
            # Remove duplicates and sort
            qubits = sorted(set(qubits))
            self.x_stabilizers.append({
                'qubits': qubits,
                'type': 'X',
                'columns': [j, j + 1]
            })

        # Z-stabilizers (row pairs)
        for i in range(self.m - 1):
            qubits = []
            for j in range(self.n_col):
                qubits.extend([self._qubit_index(i, j), self._qubit_index(i + 1, j)])
            qubits = sorted(set(qubits))
            self.z_stabilizers.append({
                'qubits': qubits,
                'type': 'Z',
                'rows': [i, i + 1]
            })

    def _generate_logical_operators(self):
        """Generate logical operators."""
        # X_bar: X on any row
        self.x_logical = {
            'qubits': [self._qubit_index(0, j) for j in range(self.n_col)],
            'type': 'X',
            'row': 0
        }

        # Z_bar: Z on any column
        self.z_logical = {
            'qubits': [self._qubit_index(i, 0) for i in range(self.m)],
            'type': 'Z',
            'column': 0
        }

    def get_syndrome(self, x_error: Optional[int] = None,
                     z_error: Optional[int] = None) -> Dict:
        """
        Compute syndrome for given errors.

        Parameters:
        -----------
        x_error : int or None
            Qubit index with X error
        z_error : int or None
            Qubit index with Z error
        """
        x_gauge_syn = [1] * len(self.x_gauges)
        z_gauge_syn = [1] * len(self.z_gauges)

        # X error affects Z gauges
        if x_error is not None:
            for idx, gauge in enumerate(self.z_gauges):
                if x_error in gauge['qubits']:
                    z_gauge_syn[idx] *= -1

        # Z error affects X gauges
        if z_error is not None:
            for idx, gauge in enumerate(self.x_gauges):
                if z_error in gauge['qubits']:
                    x_gauge_syn[idx] *= -1

        # Compute stabilizer syndromes from gauge products
        x_stab_syn = []
        for j in range(self.n_col - 1):
            # Product of X-gauges in column pair j, j+1
            prod = 1
            for idx, gauge in enumerate(self.x_gauges):
                if gauge['coords'][0][1] == j:  # Column j
                    prod *= x_gauge_syn[idx]
            x_stab_syn.append(prod)

        z_stab_syn = []
        for i in range(self.m - 1):
            # Product of Z-gauges in row pair i, i+1
            prod = 1
            for idx, gauge in enumerate(self.z_gauges):
                if gauge['coords'][0][0] == i:  # Row i
                    prod *= z_gauge_syn[idx]
            z_stab_syn.append(prod)

        return {
            'x_gauge': x_gauge_syn,
            'z_gauge': z_gauge_syn,
            'x_stabilizer': x_stab_syn,
            'z_stabilizer': z_stab_syn
        }

    def decode_syndrome(self, syndrome: Dict) -> Dict:
        """Simple decoder for single errors."""
        x_correction = None
        z_correction = None

        # X errors detected by Z-stabilizer syndrome
        z_stab = syndrome['z_stabilizer']
        if -1 in z_stab:
            # Find row from Z-stabilizer failures
            failed_rows = [i for i, s in enumerate(z_stab) if s == -1]
            if len(failed_rows) == 1:
                row = failed_rows[0]  # Error between row and row+1
            elif len(failed_rows) == 2:
                row = failed_rows[0] + 1  # Error at middle row
            else:
                row = 0  # Default

            # Find column from Z-gauge failures
            z_gauge = syndrome['z_gauge']
            # Count failures per column
            col_failures = {}
            for idx, val in enumerate(z_gauge):
                if val == -1:
                    col = self.z_gauges[idx]['coords'][0][1]
                    col_failures[col] = col_failures.get(col, 0) + 1

            if col_failures:
                col = max(col_failures, key=col_failures.get)
                x_correction = self._qubit_index(row, col)

        # Z errors detected by X-stabilizer syndrome
        x_stab = syndrome['x_stabilizer']
        if -1 in x_stab:
            failed_cols = [j for j, s in enumerate(x_stab) if s == -1]
            if len(failed_cols) == 1:
                col = failed_cols[0]
            elif len(failed_cols) == 2:
                col = failed_cols[0] + 1
            else:
                col = 0

            x_gauge = syndrome['x_gauge']
            row_failures = {}
            for idx, val in enumerate(x_gauge):
                if val == -1:
                    row = self.x_gauges[idx]['coords'][0][0]
                    row_failures[row] = row_failures.get(row, 0) + 1

            if row_failures:
                row = max(row_failures, key=row_failures.get)
                z_correction = self._qubit_index(row, col)

        return {
            'x_correction': x_correction,
            'z_correction': z_correction
        }

    def print_lattice(self):
        """Print the lattice structure."""
        print(f"\n{self.m}×{self.n_col} Bacon-Shor Lattice:")
        print("-" * (self.n_col * 8))
        for i in range(self.m):
            row_str = "  ".join(f"q({i},{j})" for j in range(self.n_col))
            print(f"Row {i}: {row_str}")
        print()

    def resource_count(self) -> Dict:
        """Count resources for syndrome extraction."""
        return {
            'x_gauges': len(self.x_gauges),
            'z_gauges': len(self.z_gauges),
            'total_gauges': len(self.x_gauges) + len(self.z_gauges),
            'cnots_per_round': 2 * (len(self.x_gauges) + len(self.z_gauges)),
            'ancilla_parallel': len(self.x_gauges) + len(self.z_gauges),
            'ancilla_sequential': 1
        }

def compare_codes(codes: List[SubsystemCodeParams]):
    """Compare multiple subsystem codes."""
    print(f"\n{'Code':<25} {'n':<5} {'k':<5} {'r':<5} {'d':<5} {'Singleton':<12} {'k/n':<8}")
    print("-" * 75)

    for code in codes:
        satisfies, d_max = code.check_singleton()
        status = f"✓ (≤{d_max})" if satisfies else f"✗ (>{d_max})"
        ratio = f"{code.k/code.n:.3f}"
        print(f"{code.name:<25} {code.n:<5} {code.k:<5} {code.r:<5} {code.d:<5} {status:<12} {ratio:<8}")

# Main demonstration
print("=" * 70)
print("WEEK 103 SYNTHESIS: Complete Subsystem Code Toolkit")
print("=" * 70)

# 1. Bacon-Shor family comparison
print("\n1. Bacon-Shor Code Family")
print("-" * 50)

codes = []
for m in [2, 3, 4, 5]:
    for n in [m, m+1]:
        if m * n <= 30:
            code = BaconShorCode(m, n)
            codes.append(code.params)

compare_codes(codes)

# 2. Detailed analysis of 3×3 Bacon-Shor
print("\n2. Detailed 3×3 Bacon-Shor Analysis")
print("-" * 50)

bs33 = BaconShorCode(3, 3)
bs33.print_lattice()

print("Gauge Operators:")
print(f"  X-gauges ({len(bs33.x_gauges)}):")
for g in bs33.x_gauges:
    print(f"    X{g['coords'][0]}X{g['coords'][1]}")

print(f"\n  Z-gauges ({len(bs33.z_gauges)}):")
for g in bs33.z_gauges:
    print(f"    Z{g['coords'][0]}Z{g['coords'][1]}")

print(f"\nStabilizers:")
print(f"  X-stabilizers ({len(bs33.x_stabilizers)}):")
for s in bs33.x_stabilizers:
    print(f"    Columns {s['columns']}: weight {len(s['qubits'])}")

print(f"  Z-stabilizers ({len(bs33.z_stabilizers)}):")
for s in bs33.z_stabilizers:
    print(f"    Rows {s['rows']}: weight {len(s['qubits'])}")

print(f"\nLogical Operators:")
print(f"  X_bar: X on qubits {bs33.x_logical['qubits']} (row {bs33.x_logical['row']})")
print(f"  Z_bar: Z on qubits {bs33.z_logical['qubits']} (column {bs33.z_logical['column']})")

# 3. Syndrome and decoding test
print("\n3. Error Detection and Correction Test")
print("-" * 50)

print("\nTesting single X error on qubit (1,1):")
error_qubit = bs33._qubit_index(1, 1)
syndrome = bs33.get_syndrome(x_error=error_qubit)

print(f"  Error location: qubit {error_qubit} = position (1,1)")
print(f"  Z-gauge syndrome: {syndrome['z_gauge']}")
print(f"  Z-stabilizer syndrome: {syndrome['z_stabilizer']}")

correction = bs33.decode_syndrome(syndrome)
print(f"  Decoded X correction: qubit {correction['x_correction']}")
print(f"  Correct: {correction['x_correction'] == error_qubit}")

print("\nTesting single Z error on qubit (2,0):")
error_qubit = bs33._qubit_index(2, 0)
syndrome = bs33.get_syndrome(z_error=error_qubit)

print(f"  Error location: qubit {error_qubit} = position (2,0)")
print(f"  X-gauge syndrome: {syndrome['x_gauge']}")
print(f"  X-stabilizer syndrome: {syndrome['x_stabilizer']}")

correction = bs33.decode_syndrome(syndrome)
print(f"  Decoded Z correction: qubit {correction['z_correction']}")
print(f"  Correct: {correction['z_correction'] == error_qubit}")

# 4. Resource comparison
print("\n4. Resource Comparison Across Code Sizes")
print("-" * 50)

print(f"\n{'Size':<8} {'Qubits':<8} {'Gauges':<8} {'CNOTs':<10} {'Ancilla(||)':<12}")
print("-" * 50)

for m in [2, 3, 4, 5, 6]:
    code = BaconShorCode(m, m)
    resources = code.resource_count()
    print(f"{m}×{m:<6} {code.n_qubits:<8} {resources['total_gauges']:<8} "
          f"{resources['cnots_per_round']:<10} {resources['ancilla_parallel']:<12}")

# 5. Singleton bound exploration
print("\n5. Singleton Bound Exploration")
print("-" * 50)

print("\nFor n=25, k=1, exploring r vs d_max:")
n, k = 25, 1
print(f"{'r':<5} {'d_max':<8} {'Example':<30}")
print("-" * 45)
for r in [0, 4, 8, 12, 16]:
    d_max = (n - k - r) // 2 + 1
    # Check if Bacon-Shor can achieve this
    # For m×n Bacon-Shor: r = (m-1)(n-1), d = min(m,n), n_qubits = m×n = 25
    # 5×5: r=16, d=5
    # Other factorizations don't give r exactly
    if r == 16:
        example = "5×5 Bacon-Shor (d=5)"
    elif r == 0:
        example = "Stabilizer code"
    else:
        example = "—"
    print(f"{r:<5} {d_max:<8} {example:<30}")

# 6. Week 103 concept summary
print("\n6. Week 103 Concept Summary")
print("-" * 50)

print("""
┌─────────────────────────────────────────────────────────────┐
│                    SUBSYSTEM CODES                          │
├─────────────────────────────────────────────────────────────┤
│  Structure: C = A ⊗ B (logical ⊗ gauge)                     │
│  Parameters: [[n, k, r, d]]                                 │
│                                                             │
│  Key Groups:                                                │
│    • Gauge G: acts trivially on A                           │
│    • Stabilizer S = Z(G): center of gauge group             │
│    • Logical: N(G) \\ G                                      │
│                                                             │
│  Bacon-Shor [[mn, 1, (m-1)(n-1), min(m,n)]]:               │
│    • X-gauge: horizontal XX (weight 2)                      │
│    • Z-gauge: vertical ZZ (weight 2)                        │
│    • Natural fault tolerance                                │
│                                                             │
│  Advantages:                                                │
│    • Low-weight measurements                                │
│    • Higher threshold (~0.5-1%)                             │
│    • Transversal H (square codes)                           │
│    • Simple 2D connectivity                                 │
│                                                             │
│  Trade-offs:                                                │
│    • Fewer logical qubits (gauge overhead)                  │
│    • More syndrome bits                                     │
│    • Gauge-aware decoding needed                            │
└─────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("Week 103 Synthesis Complete!")
print("Next: Week 104 — Code Capacity")
```

---

## Preparation for Week 104: Code Capacity

### What's Coming

**Week 104: Code Capacity** will cover:
- Channel capacity and quantum capacity
- Hashing bound for QEC
- Coherent information
- Threshold behavior from capacity perspective
- LDPC code capacity

### Key Questions to Consider

1. What is the maximum rate $k/n$ for a given distance?
2. How does noise strength affect code requirements?
3. What are fundamental limits on quantum error correction?

### Preview Concepts

**Quantum Channel Capacity:**
$$Q(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} \max_\rho I_c(\rho, \mathcal{N}^{\otimes n})$$

**Hashing Bound:**
$$k/n \leq 1 - H(p) - p \log_2 3$$

for depolarizing channel with error rate $p$.

---

## Daily Checklist

- [ ] I can explain all subsystem code concepts from the week
- [ ] I can solve problems across all difficulty levels
- [ ] I understand the connections to advanced topics
- [ ] I can design complete subsystem code systems
- [ ] I know when to choose subsystem vs stabilizer codes
- [ ] I completed the comprehensive problem set
- [ ] I am prepared for Week 104 (Code Capacity)

---

## Week 103 Complete! 🎉

**Summary of Achievements:**
- Mastered subsystem code structure and parameters
- Understood gauge groups and their role
- Learned Bacon-Shor code family in detail
- Analyzed error correction properties and bounds
- Explored practical advantages (weight reduction, fault tolerance)
- Studied fault-tolerant operations and thresholds
- Connected to advanced topics (LDPC, topological, Floquet)

**Next:** Week 104 — Code Capacity (Days 722-728)
