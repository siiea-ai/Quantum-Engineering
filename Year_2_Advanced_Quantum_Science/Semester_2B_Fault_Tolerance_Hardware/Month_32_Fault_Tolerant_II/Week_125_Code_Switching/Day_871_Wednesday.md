# Day 871: Subsystem Codes Review

## Overview

**Day:** 871 of 1008
**Week:** 125 (Code Switching & Gauge Fixing)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** Subsystem Codes, Bacon-Shor Code, Gauge Qubits, and Gauge Operators

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Subsystem code theory and structure |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Bacon-Shor code analysis |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational exploration |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Distinguish** between subspace codes and subsystem codes
2. **Define** gauge qubits and explain their role in error correction
3. **Construct** the Bacon-Shor [[9,1,3]] code with its gauge structure
4. **Identify** gauge operators vs. stabilizer generators
5. **Analyze** the trade-offs between subsystem and stabilizer codes
6. **Explain** how gauge freedom enables simplified error correction

---

## From Subspace Codes to Subsystem Codes

### Review: Subspace Codes (Stabilizer Codes)

In a standard **stabilizer code** $[[n, k, d]]$:

- Physical Hilbert space: $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$
- Code space: $\mathcal{H}_C = $ simultaneous +1 eigenspace of all stabilizers
- Dimension: $\dim(\mathcal{H}_C) = 2^k$
- Logical information encoded in a **subspace**

$$\boxed{\mathcal{H} = \mathcal{H}_C \oplus \mathcal{H}_C^\perp}$$

### The Subsystem Code Idea

In a **subsystem code** $[[n, k, r, d]]$:

- Physical Hilbert space: $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$
- Code space decomposes as: $\mathcal{H}_C = \mathcal{H}_L \otimes \mathcal{H}_G$
- $\mathcal{H}_L$: **Logical subsystem** (dimension $2^k$)
- $\mathcal{H}_G$: **Gauge subsystem** (dimension $2^r$)
- Information encoded in a **subsystem**, not the full subspace

$$\boxed{\mathcal{H}_C = \mathcal{H}_L \otimes \mathcal{H}_G \quad (\text{tensor product structure})}$$

### Key Difference

| Property | Subspace Code | Subsystem Code |
|----------|---------------|----------------|
| Code space | $2^k$-dimensional subspace | $2^{k+r}$-dimensional subspace |
| Logical info | Uses full code space | Uses $2^k$-dim subsystem |
| Gauge qubits | None | $r$ gauge qubits |
| Stabilizer group | Fully determines code space | Subset of gauge group |

### Why Subsystem Codes?

**Advantages:**

1. **Simpler syndrome measurements:** Fewer qubits need to be measured together
2. **Reduced weight:** Gauge operators can have lower weight than stabilizers
3. **Flexible error correction:** Gauge freedom provides options
4. **Natural fault tolerance:** Some operations become easier

**Trade-offs:**

1. **More physical qubits:** Need $n$ qubits for $k$ logical + $r$ gauge
2. **Gauge uncertainty:** Must track or fix gauge state
3. **Potentially lower threshold:** Depends on implementation

---

## Mathematical Framework

### The Gauge Group

**Definition:** The **gauge group** $\mathcal{G}$ of a subsystem code is a subgroup of the Pauli group such that:

1. $-I \in \mathcal{G}$ (contains phases)
2. The center of $\mathcal{G}$ (modulo phases) is the stabilizer group $\mathcal{S}$

$$\boxed{\mathcal{S} = Z(\mathcal{G}) / \langle iI \rangle}$$

where $Z(\mathcal{G})$ is the center (elements commuting with everything in $\mathcal{G}$).

### Stabilizers vs. Gauge Operators

**Stabilizer generators:** Commute with ALL gauge operators
$$[S, G] = 0 \quad \forall S \in \mathcal{S}, G \in \mathcal{G}$$

**Gauge generators:** May not commute with each other
$$\exists G_1, G_2 \in \mathcal{G}: [G_1, G_2] \neq 0$$

### Logical Operators

**Logical operators** must:
1. Commute with all gauge operators: $[L, G] = 0 \; \forall G \in \mathcal{G}$
2. NOT be in the gauge group: $L \notin \mathcal{G}$

The logical operators act on the logical subsystem $\mathcal{H}_L$.

### Code Parameters

For a subsystem code $[[n, k, r, d]]$:
- $n$ = physical qubits
- $k$ = logical qubits
- $r$ = gauge qubits
- $d$ = code distance

**Constraint:**
$$n - k - r = \text{number of independent stabilizer generators}$$

---

## The Bacon-Shor Code [[9,1,3]]

### Introduction

The **Bacon-Shor code** is the canonical example of a subsystem code. It encodes:
- 1 logical qubit in 9 physical qubits
- Has 4 gauge qubits
- Distance 3 (corrects 1 error)

### Physical Layout

Arrange 9 qubits in a 3×3 grid:

```
1 - 2 - 3
|   |   |
4 - 5 - 6
|   |   |
7 - 8 - 9
```

### Gauge Operators

**X-type gauge operators** (horizontal pairs):
$$\begin{aligned}
G_X^{(1)} &= X_1 X_2, \quad G_X^{(2)} = X_2 X_3 \\
G_X^{(3)} &= X_4 X_5, \quad G_X^{(4)} = X_5 X_6 \\
G_X^{(5)} &= X_7 X_8, \quad G_X^{(6)} = X_8 X_9
\end{aligned}$$

**Z-type gauge operators** (vertical pairs):
$$\begin{aligned}
G_Z^{(1)} &= Z_1 Z_4, \quad G_Z^{(2)} = Z_4 Z_7 \\
G_Z^{(3)} &= Z_2 Z_5, \quad G_Z^{(4)} = Z_5 Z_8 \\
G_Z^{(5)} &= Z_3 Z_6, \quad G_Z^{(6)} = Z_6 Z_9
\end{aligned}$$

### Stabilizer Generators

The **stabilizers** are products of gauge operators:

**X-type stabilizers** (full rows):
$$\begin{aligned}
S_X^{(1)} &= X_1 X_2 X_3 = G_X^{(1)} G_X^{(2)} \\
S_X^{(2)} &= X_4 X_5 X_6 = G_X^{(3)} G_X^{(4)} \\
S_X^{(3)} &= X_7 X_8 X_9 = G_X^{(5)} G_X^{(6)}
\end{aligned}$$

But only 2 are independent (product of all three = $\bar{X}^2 = I$).

**Z-type stabilizers** (full columns):
$$\begin{aligned}
S_Z^{(1)} &= Z_1 Z_4 Z_7 = G_Z^{(1)} G_Z^{(2)} \\
S_Z^{(2)} &= Z_2 Z_5 Z_8 = G_Z^{(3)} G_Z^{(4)} \\
S_Z^{(3)} &= Z_3 Z_6 Z_9 = G_Z^{(5)} G_Z^{(6)}
\end{aligned}$$

Again, 2 independent.

### Logical Operators

$$\boxed{\bar{X} = X_1 X_2 X_3 \quad \text{(any row)}}$$
$$\boxed{\bar{Z} = Z_1 Z_4 Z_7 \quad \text{(any column)}}$$

Note: Different rows/columns give equivalent logical operators (differ by stabilizers).

### Why This is a Subsystem Code

**Key observation:** X gauge operators don't commute with Z gauge operators on shared qubits!

$$[X_1 X_2, Z_1 Z_4] = X_1 X_2 Z_1 Z_4 - Z_1 Z_4 X_1 X_2$$

At qubit 1: $XZ = -ZX$, so:
$$X_1 X_2 Z_1 Z_4 = -Z_1 X_1 X_2 Z_4 = -Z_1 Z_4 X_1 X_2$$

Therefore:
$$\boxed{[G_X^{(1)}, G_Z^{(1)}] \neq 0}$$

This non-commutativity is the hallmark of subsystem codes!

---

## Gauge Structure Analysis

### Counting Degrees of Freedom

**Physical qubits:** 9

**Stabilizer generators:** 4 (2 X-type + 2 Z-type)

**Gauge pairs:** Each X gauge operator pairs with a Z gauge operator:
$$\{G_X^{(i)}, G_Z^{(j)}\} \text{ with } [G_X^{(i)}, G_Z^{(j)}] \neq 0$$

**Gauge qubits:** 4

**Logical qubits:** 1

**Check:** $9 = 1 + 4 + 4$ ✓ (logical + gauge + stabilizer constraints)

### The Gauge Qubit Space

The 4 gauge qubits span a $2^4 = 16$-dimensional space.

Each gauge qubit corresponds to a pair of anti-commuting gauge operators:
$$(G_X, G_Z) \text{ with } \{G_X, G_Z\} = 0$$

The gauge qubit state is determined by the eigenvalue of (say) $G_Z$:
- $G_Z = +1$: gauge qubit in $|0_G\rangle$
- $G_Z = -1$: gauge qubit in $|1_G\rangle$

### Total Code Space

$$\mathcal{H}_C = \mathcal{H}_L \otimes \mathcal{H}_G = (\mathbb{C}^2) \otimes (\mathbb{C}^2)^{\otimes 4} = 32\text{-dimensional}$$

This is the simultaneous +1 eigenspace of all 4 stabilizers.

---

## Error Correction in Bacon-Shor

### Advantage: Weight-2 Measurements

**Key benefit:** We only need to measure weight-2 gauge operators, not weight-3 stabilizers!

**Standard stabilizer approach:**
- Measure $S_X^{(1)} = X_1 X_2 X_3$ (weight 3)
- Requires 3-qubit entangling operations

**Subsystem approach:**
- Measure $G_X^{(1)} = X_1 X_2$ (weight 2)
- Measure $G_X^{(2)} = X_2 X_3$ (weight 2)
- Infer stabilizer: $S_X^{(1)} = G_X^{(1)} \cdot G_X^{(2)}$

### The Measurement Trade-off

**More measurements, simpler each:**

| Approach | Measurements | Max Weight | Ancilla Coupling |
|----------|--------------|------------|------------------|
| Direct stabilizer | 4 | 3 | 3-body |
| Gauge operators | 12 | 2 | 2-body |

### Syndrome Extraction

**X-error syndrome** (from Z gauge measurements):

Measure $G_Z^{(1)} = Z_1 Z_4$, $G_Z^{(2)} = Z_4 Z_7$, etc.

An X error on qubit 5 anticommutes with:
- $G_Z^{(3)} = Z_2 Z_5$
- $G_Z^{(4)} = Z_5 Z_8$

This identifies column 2 as having an error.

**Z-error syndrome** (from X gauge measurements):

Measure $G_X^{(3)} = X_4 X_5$, $G_X^{(4)} = X_5 X_6$, etc.

A Z error on qubit 5 anticommutes with:
- $G_X^{(3)} = X_4 X_5$
- $G_X^{(4)} = X_5 X_6$

This identifies row 2 as having an error.

**Combined:** Row 2, Column 2 → Qubit 5!

### Gauge Randomization

**Important:** Measuring gauge operators randomizes the gauge qubit states!

When we measure $G_Z^{(1)} = Z_1 Z_4$:
- We get outcome $\pm 1$
- The gauge qubit is projected
- This doesn't affect the logical information (in $\mathcal{H}_L$)

This is why subsystem codes work: logical information is protected from gauge randomization.

---

## Comparison: Bacon-Shor vs. Shor Code

### The Shor Code [[9,1,3]]

The Shor code is a **stabilizer code** (subspace code) with:

**Stabilizers:**
$$\begin{aligned}
S_1 &= Z_1 Z_2, \quad S_2 = Z_2 Z_3 \\
S_3 &= Z_4 Z_5, \quad S_4 = Z_5 Z_6 \\
S_5 &= Z_7 Z_8, \quad S_6 = Z_8 Z_9 \\
S_7 &= X_1 X_2 X_3 X_4 X_5 X_6 \\
S_8 &= X_4 X_5 X_6 X_7 X_8 X_9
\end{aligned}$$

**Logical operators:**
$$\bar{X} = X^{\otimes 9}, \quad \bar{Z} = Z_1 Z_4 Z_7$$

### Relationship

**Theorem:** The Shor code is a **gauge-fixed** version of the Bacon-Shor code.

By measuring and fixing all Z gauge operators to $+1$:
$$G_Z^{(i)} = +1 \quad \forall i$$

the Bacon-Shor code reduces to the Shor code.

### Comparison Table

| Property | Bacon-Shor | Shor |
|----------|------------|------|
| Type | Subsystem | Stabilizer |
| Physical qubits | 9 | 9 |
| Logical qubits | 1 | 1 |
| Gauge qubits | 4 | 0 |
| Stabilizer generators | 4 | 8 |
| Max stabilizer weight | 3 | 6 |
| Gauge measurement weight | 2 | N/A |

---

## Worked Examples

### Example 1: Verifying Gauge Algebra

**Problem:** Show that $G_X^{(3)} = X_4 X_5$ and $G_Z^{(3)} = Z_2 Z_5$ anticommute.

**Solution:**

Compute the commutator by checking shared qubits:
- $G_X^{(3)}$ acts on qubits 4, 5
- $G_Z^{(3)}$ acts on qubits 2, 5
- Shared qubit: 5

At qubit 5:
$$X_5 Z_5 = iY_5 = -Z_5 X_5$$

Therefore:
$$G_X^{(3)} G_Z^{(3)} = (X_4 X_5)(Z_2 Z_5) = X_4 Z_2 (X_5 Z_5) = X_4 Z_2 (-Z_5 X_5)$$
$$= -X_4 Z_2 Z_5 X_5 = -(Z_2 Z_5)(X_4 X_5) = -G_Z^{(3)} G_X^{(3)}$$

$$\boxed{\{G_X^{(3)}, G_Z^{(3)}\} = 0 \text{ (anticommute)}}$$

### Example 2: Syndrome for Z Error

**Problem:** A Z error occurs on qubit 6. Determine which gauge operators are affected.

**Solution:**

**Z gauge operators** (Z_6 commutes with all Z operators):
$$[Z_6, G_Z^{(i)}] = 0 \quad \forall i$$

No Z gauge syndromes.

**X gauge operators:**
- $G_X^{(4)} = X_5 X_6$: Contains $X_6$, so $\{Z_6, X_6\} = 0$
  - $G_X^{(4)}$ anticommutes with $Z_6$ ✓
- $G_X^{(1)} = X_1 X_2$: No overlap, commutes
- $G_X^{(2)} = X_2 X_3$: No overlap, commutes
- $G_X^{(3)} = X_4 X_5$: No overlap, commutes
- $G_X^{(5)} = X_7 X_8$: No overlap, commutes
- $G_X^{(6)} = X_8 X_9$: No overlap, commutes

**Syndrome:** Only $G_X^{(4)}$ flags.

**Interpretation:** Error in row 2 (from $G_X^{(4)}$).

To locate the column, check Z stabilizers or use additional gauge info.

$$\boxed{\text{Z error on qubit 6 triggers } G_X^{(4)} \text{ syndrome}}$$

### Example 3: Constructing Logical States

**Problem:** Write the logical $|0_L\rangle$ state for Bacon-Shor, assuming all gauge qubits are in $|0_G\rangle$ (i.e., $G_Z^{(i)} = +1$).

**Solution:**

With gauge fixing $G_Z^{(i)} = +1$ for all vertical pairs, we have the Shor code encoding.

**Shor's $|0_L\rangle$:**

Each row must satisfy $Z_i Z_{i+1} = +1$ within the row:
- Row 1: $|000\rangle$ or $|111\rangle$ (but need Z stabilizers +1)
- Actually, rows encode $|+\rangle$ states

For Shor code:
$$|0_L\rangle = |+++\rangle_{\text{rows}} = \frac{1}{2\sqrt{2}}(|000\rangle + |111\rangle)^{\otimes 3}$$

Expanded:
$$|0_L\rangle = \frac{1}{2\sqrt{2}}[(|000\rangle + |111\rangle)(|000\rangle + |111\rangle)(|000\rangle + |111\rangle)]$$

$$= \frac{1}{2\sqrt{2}}[|000000000\rangle + |000000111\rangle + |000111000\rangle + \cdots + |111111111\rangle]$$

(8 terms total, each with coefficient $\frac{1}{2\sqrt{2}}$)

$$\boxed{|0_L\rangle = \frac{1}{2\sqrt{2}}\sum_{a,b,c \in \{0,1\}} |a^3 b^3 c^3\rangle}$$

where $a^3$ means $aaa$ (three copies).

---

## Practice Problems

### Level 1: Direct Application

**P1.1** List all 6 X-type gauge operators and all 6 Z-type gauge operators for the Bacon-Shor code.

**P1.2** Verify that the product $G_X^{(1)} G_X^{(2)} = X_1 X_2 X_3$ (a row) commutes with all Z gauge operators.

**P1.3** For an X error on qubit 8, which gauge operators anticommute with it?

### Level 2: Intermediate

**P2.1** Prove that logical $\bar{X} = X_1 X_2 X_3$ commutes with all gauge operators (both X and Z type).

**P2.2** Show that measuring all Z gauge operators and getting +1 outcomes projects the Bacon-Shor code to the Shor code.

**P2.3** Design a syndrome extraction circuit for the Bacon-Shor code that uses only weight-2 gauge measurements. How many ancilla qubits are needed?

### Level 3: Challenging

**P3.1** Generalize the Bacon-Shor construction to an $m \times m$ grid. What are the parameters $[[n, k, r, d]]$ as functions of $m$?

**P3.2** The Bacon-Shor code can be viewed as a quantum version of a product code. Explain this connection and derive the code parameters from the classical component codes.

**P3.3** Analyze the threshold of the Bacon-Shor code under depolarizing noise. How does it compare to the Shor code threshold?

---

## Computational Lab

```python
"""
Day 871: Subsystem Codes and the Bacon-Shor Code
=================================================

Implementation and analysis of the [[9,1,3]] Bacon-Shor subsystem code.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from itertools import product

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def tensor_list(ops: List[np.ndarray]) -> np.ndarray:
    """Compute tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def pauli_on_qubits(pauli: str, qubits: List[int], n: int) -> np.ndarray:
    """
    Create n-qubit Pauli operator with specified Pauli on given qubits.

    Parameters:
    -----------
    pauli : str
        'X', 'Y', or 'Z'
    qubits : List[int]
        Qubit indices (0-indexed)
    n : int
        Total number of qubits
    """
    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    P = paulis[pauli]

    ops = [I] * n
    for q in qubits:
        ops[q] = P
    return tensor_list(ops)


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute [A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute {A, B} = AB + BA."""
    return A @ B + B @ A


class BaconShorCode:
    """
    The [[9,1,3]] Bacon-Shor subsystem code.

    Qubit layout (0-indexed):
    0 - 1 - 2
    |   |   |
    3 - 4 - 5
    |   |   |
    6 - 7 - 8
    """

    def __init__(self):
        self.n = 9
        self.k = 1  # logical qubits
        self.r = 4  # gauge qubits
        self.d = 3  # distance

        # Qubit grid positions
        self.grid = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ])

        # Define gauge operators
        self._define_gauge_operators()

        # Define stabilizers
        self._define_stabilizers()

        # Define logical operators
        self._define_logical_operators()

    def _define_gauge_operators(self):
        """Define X and Z gauge operators."""
        # X gauge operators (horizontal pairs)
        self.gauge_x = [
            [0, 1], [1, 2],  # Row 0
            [3, 4], [4, 5],  # Row 1
            [6, 7], [7, 8],  # Row 2
        ]

        # Z gauge operators (vertical pairs)
        self.gauge_z = [
            [0, 3], [3, 6],  # Column 0
            [1, 4], [4, 7],  # Column 1
            [2, 5], [5, 8],  # Column 2
        ]

    def _define_stabilizers(self):
        """Define stabilizer generators from gauge products."""
        # X stabilizers (full rows) - only 2 independent
        self.stabilizers_x = [
            [0, 1, 2],  # Row 0
            [3, 4, 5],  # Row 1
            # Row 2 = Row 0 * Row 1 * X_L^2 = Row 0 * Row 1
        ]

        # Z stabilizers (full columns) - only 2 independent
        self.stabilizers_z = [
            [0, 3, 6],  # Column 0
            [1, 4, 7],  # Column 1
            # Column 2 = Column 0 * Column 1 * Z_L^2 = Column 0 * Column 1
        ]

    def _define_logical_operators(self):
        """Define logical X and Z operators."""
        # Logical X: any row
        self.logical_x = [0, 1, 2]  # Row 0

        # Logical Z: any column
        self.logical_z = [0, 3, 6]  # Column 0

    def get_gauge_x_operator(self, idx: int) -> np.ndarray:
        """Get the idx-th X gauge operator as a matrix."""
        return pauli_on_qubits('X', self.gauge_x[idx], self.n)

    def get_gauge_z_operator(self, idx: int) -> np.ndarray:
        """Get the idx-th Z gauge operator as a matrix."""
        return pauli_on_qubits('Z', self.gauge_z[idx], self.n)

    def get_stabilizer_x(self, idx: int) -> np.ndarray:
        """Get X stabilizer as matrix."""
        return pauli_on_qubits('X', self.stabilizers_x[idx], self.n)

    def get_stabilizer_z(self, idx: int) -> np.ndarray:
        """Get Z stabilizer as matrix."""
        return pauli_on_qubits('Z', self.stabilizers_z[idx], self.n)

    def get_logical_x(self) -> np.ndarray:
        """Get logical X operator."""
        return pauli_on_qubits('X', self.logical_x, self.n)

    def get_logical_z(self) -> np.ndarray:
        """Get logical Z operator."""
        return pauli_on_qubits('Z', self.logical_z, self.n)

    def check_commutation_relations(self) -> Dict[str, bool]:
        """Verify the commutation relations of gauge operators."""
        results = {}

        # X gauge ops should commute with each other
        all_x_commute = True
        for i in range(len(self.gauge_x)):
            for j in range(i+1, len(self.gauge_x)):
                Gxi = self.get_gauge_x_operator(i)
                Gxj = self.get_gauge_x_operator(j)
                if not np.allclose(commutator(Gxi, Gxj), 0):
                    all_x_commute = False
        results['X_gauge_commute'] = all_x_commute

        # Z gauge ops should commute with each other
        all_z_commute = True
        for i in range(len(self.gauge_z)):
            for j in range(i+1, len(self.gauge_z)):
                Gzi = self.get_gauge_z_operator(i)
                Gzj = self.get_gauge_z_operator(j)
                if not np.allclose(commutator(Gzi, Gzj), 0):
                    all_z_commute = False
        results['Z_gauge_commute'] = all_z_commute

        # Some X and Z gauge ops anticommute
        found_anticommuting = False
        for i in range(len(self.gauge_x)):
            for j in range(len(self.gauge_z)):
                Gx = self.get_gauge_x_operator(i)
                Gz = self.get_gauge_z_operator(j)
                # Check if they anticommute
                if np.allclose(anticommutator(Gx, Gz), 0):
                    found_anticommuting = True
                    break
            if found_anticommuting:
                break
        results['XZ_some_anticommute'] = found_anticommuting

        # Logical ops commute with all gauge ops
        Lx = self.get_logical_x()
        Lz = self.get_logical_z()

        lx_commutes = True
        for i in range(len(self.gauge_x)):
            if not np.allclose(commutator(Lx, self.get_gauge_x_operator(i)), 0):
                lx_commutes = False
        for i in range(len(self.gauge_z)):
            if not np.allclose(commutator(Lx, self.get_gauge_z_operator(i)), 0):
                lx_commutes = False
        results['Logical_X_commutes'] = lx_commutes

        lz_commutes = True
        for i in range(len(self.gauge_x)):
            if not np.allclose(commutator(Lz, self.get_gauge_x_operator(i)), 0):
                lz_commutes = False
        for i in range(len(self.gauge_z)):
            if not np.allclose(commutator(Lz, self.get_gauge_z_operator(i)), 0):
                lz_commutes = False
        results['Logical_Z_commutes'] = lz_commutes

        return results

    def error_syndrome(self, error_type: str, qubit: int) -> Dict[str, List[int]]:
        """
        Compute the syndrome for a single-qubit error.

        Parameters:
        -----------
        error_type : str
            'X', 'Y', or 'Z'
        qubit : int
            Qubit index (0-8)

        Returns:
        --------
        Dict with X and Z gauge operator syndromes
        """
        E = pauli_on_qubits(error_type, [qubit], self.n)

        syndrome_x = []
        syndrome_z = []

        # Check which X gauge operators anticommute with the error
        for i, qubits in enumerate(self.gauge_x):
            Gx = self.get_gauge_x_operator(i)
            # Anticommute if product gives -1 eigenvalue change
            if not np.allclose(commutator(E, Gx), 0):
                syndrome_x.append(i)

        # Check which Z gauge operators anticommute with the error
        for i, qubits in enumerate(self.gauge_z):
            Gz = self.get_gauge_z_operator(i)
            if not np.allclose(commutator(E, Gz), 0):
                syndrome_z.append(i)

        return {'X_gauge': syndrome_x, 'Z_gauge': syndrome_z}

    def print_code_structure(self):
        """Print the code structure."""
        print("Bacon-Shor [[9,1,3]] Code Structure")
        print("=" * 50)

        print("\nQubit Layout:")
        print("  0 - 1 - 2")
        print("  |   |   |")
        print("  3 - 4 - 5")
        print("  |   |   |")
        print("  6 - 7 - 8")

        print("\nX Gauge Operators (horizontal):")
        for i, qubits in enumerate(self.gauge_x):
            print(f"  G_X^({i}): X_{qubits[0]} X_{qubits[1]}")

        print("\nZ Gauge Operators (vertical):")
        for i, qubits in enumerate(self.gauge_z):
            print(f"  G_Z^({i}): Z_{qubits[0]} Z_{qubits[1]}")

        print("\nX Stabilizers (rows):")
        for i, qubits in enumerate(self.stabilizers_x):
            print(f"  S_X^({i}): X_{qubits[0]} X_{qubits[1]} X_{qubits[2]}")

        print("\nZ Stabilizers (columns):")
        for i, qubits in enumerate(self.stabilizers_z):
            print(f"  S_Z^({i}): Z_{qubits[0]} Z_{qubits[1]} Z_{qubits[2]}")

        print(f"\nLogical X: X_{self.logical_x}")
        print(f"Logical Z: Z_{self.logical_z}")


def verify_gauge_algebra():
    """Verify the gauge algebra of Bacon-Shor code."""
    print("\n" + "=" * 60)
    print("Verifying Gauge Algebra")
    print("=" * 60)

    code = BaconShorCode()
    results = code.check_commutation_relations()

    for property_name, is_satisfied in results.items():
        status = "PASS" if is_satisfied else "FAIL"
        print(f"  {property_name}: {status}")


def syndrome_demo():
    """Demonstrate syndrome extraction."""
    print("\n" + "=" * 60)
    print("Syndrome Extraction Demo")
    print("=" * 60)

    code = BaconShorCode()

    # Test errors on different qubits
    test_cases = [
        ('X', 4, "X error on qubit 4 (center)"),
        ('Z', 4, "Z error on qubit 4 (center)"),
        ('X', 0, "X error on qubit 0 (corner)"),
        ('Z', 8, "Z error on qubit 8 (corner)"),
    ]

    for error_type, qubit, description in test_cases:
        print(f"\n{description}")
        print("-" * 40)

        syndrome = code.error_syndrome(error_type, qubit)

        print(f"  X gauge syndromes: {syndrome['X_gauge']}")
        print(f"  Z gauge syndromes: {syndrome['Z_gauge']}")

        # Interpret syndrome
        if error_type == 'X':
            # X errors trigger Z gauge syndromes
            if syndrome['Z_gauge']:
                cols = set()
                for gz_idx in syndrome['Z_gauge']:
                    gz_qubits = code.gauge_z[gz_idx]
                    for q in gz_qubits:
                        cols.add(q % 3)
                print(f"  Interpretation: Error in column(s) {cols}")
        else:  # Z error
            # Z errors trigger X gauge syndromes
            if syndrome['X_gauge']:
                rows = set()
                for gx_idx in syndrome['X_gauge']:
                    gx_qubits = code.gauge_x[gx_idx]
                    for q in gx_qubits:
                        rows.add(q // 3)
                print(f"  Interpretation: Error in row(s) {rows}")


def compare_with_shor():
    """Compare Bacon-Shor with Shor code."""
    print("\n" + "=" * 60)
    print("Comparison: Bacon-Shor vs Shor Code")
    print("=" * 60)

    print("\n" + "-" * 40)
    print("Bacon-Shor [[9,1,3]] (Subsystem Code)")
    print("-" * 40)
    print("  Physical qubits: 9")
    print("  Logical qubits: 1")
    print("  Gauge qubits: 4")
    print("  Stabilizer generators: 4")
    print("  Gauge operators: 12 (weight 2)")
    print("  Syndrome measurement: weight-2 operators")

    print("\n" + "-" * 40)
    print("Shor [[9,1,3]] (Stabilizer Code)")
    print("-" * 40)
    print("  Physical qubits: 9")
    print("  Logical qubits: 1")
    print("  Gauge qubits: 0")
    print("  Stabilizer generators: 8")
    print("  Syndrome measurement: weight-2 and weight-6 operators")

    print("\n" + "-" * 40)
    print("Key Differences")
    print("-" * 40)
    print("  1. Bacon-Shor uses only weight-2 measurements")
    print("  2. Shor code has weight-6 X stabilizers")
    print("  3. Bacon-Shor has gauge freedom (4 gauge qubits)")
    print("  4. Shor code is gauge-fixed Bacon-Shor")


def gauge_fixing_demo():
    """Demonstrate gauge fixing from Bacon-Shor to Shor."""
    print("\n" + "=" * 60)
    print("Gauge Fixing: Bacon-Shor -> Shor")
    print("=" * 60)

    code = BaconShorCode()

    print("\nGauge fixing procedure:")
    print("  1. Measure all Z gauge operators")
    print("  2. Record outcomes (+1 or -1)")
    print("  3. If outcome is -1, apply X correction")
    print("  4. Result: All Z gauge operators = +1")

    print("\nZ Gauge operators to fix:")
    for i, qubits in enumerate(code.gauge_z):
        print(f"  G_Z^({i}) = Z_{qubits[0]} Z_{qubits[1]}")

    print("\nAfter fixing all G_Z = +1:")
    print("  - Gauge qubits are in definite state |0_G>")
    print("  - Code reduces to Shor code")
    print("  - Can now measure weight-3 row stabilizers")

    print("\nImportant: Gauge fixing is a projection!")
    print("  - Logical information is preserved")
    print("  - Gauge information is lost (projected)")


def main():
    """Run all demonstrations."""
    code = BaconShorCode()
    code.print_code_structure()

    verify_gauge_algebra()
    syndrome_demo()
    compare_with_shor()
    gauge_fixing_demo()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Subsystem codes encode in subsystem, not full subspace")
    print("2. Gauge operators don't all commute (unlike stabilizers)")
    print("3. Bacon-Shor enables weight-2 syndrome measurements")
    print("4. Gauge fixing converts subsystem code to stabilizer code")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Subsystem decomposition | $\mathcal{H}_C = \mathcal{H}_L \otimes \mathcal{H}_G$ |
| Gauge group center | $\mathcal{S} = Z(\mathcal{G})/\langle iI \rangle$ |
| Bacon-Shor X gauge | $G_X^{(ij)} = X_i X_j$ (horizontal pairs) |
| Bacon-Shor Z gauge | $G_Z^{(ij)} = Z_i Z_j$ (vertical pairs) |
| Stabilizers from gauge | $S_X = \prod_{\text{row}} G_X$, $S_Z = \prod_{\text{col}} G_Z$ |
| Logical operators | $\bar{X} = X_{\text{row}}$, $\bar{Z} = Z_{\text{col}}$ |

### Main Takeaways

1. **Subsystem codes** encode information in a subsystem, not the full code space
2. **Gauge qubits** provide extra degrees of freedom that don't carry logical information
3. **Gauge operators** may not commute, unlike stabilizers which always commute
4. **Bacon-Shor code** is the prototypical subsystem code with 9 qubits
5. **Weight reduction:** Gauge measurements can be lower weight than stabilizers
6. **Gauge fixing** converts a subsystem code to a stabilizer code

---

## Daily Checklist

- [ ] I can distinguish subspace codes from subsystem codes
- [ ] I can define gauge qubits and the gauge group
- [ ] I can construct the Bacon-Shor code gauge operators
- [ ] I understand which gauge operators commute/anticommute
- [ ] I can explain the advantage of weight-2 measurements
- [ ] I understand the relationship between Bacon-Shor and Shor codes

---

## Preview: Day 872

Tomorrow we explore **Gauge Fixing Protocols**:

- How to systematically fix gauge freedom
- Measurement-based gauge fixing procedures
- How gauge fixing enables different transversal gates
- The Paetznick-Reichardt universality result
- Connection to code switching via gauge manipulation

Gauge fixing provides yet another route to universal fault-tolerant computation!
