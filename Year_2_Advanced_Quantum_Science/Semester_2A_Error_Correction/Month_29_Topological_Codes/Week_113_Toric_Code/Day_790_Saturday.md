# Day 790: Error Model and Distance

## Overview

**Day:** 790 of 1008
**Week:** 113 (Toric Code Fundamentals)
**Month:** 29 (Topological Codes)
**Topic:** Error Models, Code Distance, and Error Thresholds

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Error models and syndrome patterns |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Code distance and thresholds |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Describe** X, Z, and Y errors in terms of stabilizer violations
2. **Visualize** errors as creating pairs of anyonic excitations
3. **Define** the code distance as minimum non-trivial loop length
4. **Explain** the error correction process and minimum-weight decoding
5. **State** the error threshold (~10.9%) for independent noise
6. **Compare** toric code performance with other quantum codes

---

## Core Theory

### 1. Single-Qubit Errors

**Pauli X error** on edge $e$:
- Anti-commutes with plaquettes containing $e$
- Creates violations at exactly 2 plaquettes (the faces adjacent to $e$)
- Interpretation: Creates a pair of **magnetic fluxes (m particles)**

**Pauli Z error** on edge $e$:
- Anti-commutes with stars containing $e$
- Creates violations at exactly 2 stars (the vertices at ends of $e$)
- Interpretation: Creates a pair of **electric charges (e particles)**

**Pauli Y error** on edge $e$:
- $Y = iXZ$, so creates both X and Z violations
- Creates violations at 2 plaquettes AND 2 stars
- Interpretation: Creates both e and m particles

### 2. Error Syndromes

The **syndrome** is the pattern of stabilizer violations:

$$\sigma_v = \langle A_v \rangle = \begin{cases} +1 & \text{no charge at } v \\ -1 & \text{charge at } v \end{cases}$$

$$\sigma_p = \langle B_p \rangle = \begin{cases} +1 & \text{no flux at } p \\ -1 & \text{flux at } p \end{cases}$$

**Key property:** Syndromes always come in pairs!
- X-errors create pairs of plaquette violations
- Z-errors create pairs of vertex violations

This is because the boundary of an edge has two endpoints.

### 3. Error Chains

A sequence of single-qubit errors forms an **error chain**:

**Z-error chain:** Product of Z on edges $e_1, e_2, \ldots, e_k$
$$E_Z = Z_{e_1} Z_{e_2} \cdots Z_{e_k}$$

**Syndrome:** Star violations at the endpoints of the chain.

If the chain is **closed** (no endpoints), the syndrome is trivial!

```
Open chain (creates defects):    Closed chain (no defects):
    ●─Z─Z─Z─●                       ●─Z─Z─●
    ↑       ↑                       │     │
  defect  defect                    Z     Z
                                    │     │
                                    ●─Z─Z─●
```

### 4. Homology and Error Equivalence

Two errors are **equivalent** if they differ by a stabilizer:
$$E_1 \sim E_2 \iff E_1 E_2^\dagger \in \text{Stabilizer group}$$

**For Z-errors:**
- Open chains with same endpoints are equivalent (differ by plaquette products)
- Closed contractible chains are equivalent to identity (products of plaquettes)
- Closed non-contractible chains are logical operators!

**Error correction strategy:**
1. Measure syndrome (identify defect locations)
2. Find a chain connecting the defects
3. Apply correction (any chain with those endpoints works)

### 5. Code Distance

**Definition:** The code distance $d$ is the minimum weight of a non-trivial logical operator.

**For the toric code:**

$$\boxed{d = L}$$

**Proof:**
- Logical operators are non-contractible loops
- Minimum non-contractible loop on $L \times L$ torus has length $L$
- Therefore $d = L$

**Error correction capability:**
$$t = \left\lfloor \frac{d-1}{2} \right\rfloor = \left\lfloor \frac{L-1}{2} \right\rfloor$$

The toric code can correct any error of weight $\leq t$.

### 6. Minimum Weight Perfect Matching (MWPM)

The optimal decoder for the toric code uses **MWPM**:

1. **Measure syndrome:** Identify all defect locations
2. **Construct graph:** Vertices = defects, edges = all pairs, weights = distances
3. **Find MWPM:** Pair up defects with minimum total distance
4. **Apply correction:** For each pair, apply errors along shortest path

**Why it works:**
- Defects come in pairs (from original error)
- If we pair correctly, correction recovers original state
- Minimum weight matching maximizes success probability

### 7. Error Threshold

**Theorem (Dennis-Kitaev-Landahl-Preskill, 2002):**

For independent bit-flip and phase-flip noise with error probability $p$ per qubit:

$$\boxed{p < p_{th} \approx 10.9\%}$$

implies arbitrarily reliable quantum memory as $L \to \infty$.

**Interpretation:**
- Below threshold: Logical error rate → 0 exponentially in L
- Above threshold: Logical error rate → 1 (no protection)

**Comparison with other codes:**

| Code | Threshold |
|------|-----------|
| Toric/Surface Code | ~10.9% (bit-flip only) |
| Surface Code (phenomenological) | ~2.9% |
| Surface Code (circuit-level) | ~0.5-1% |
| Concatenated [[7,1,3]] | ~10⁻⁴ to 10⁻³ |

### 8. Relationship to Random Bond Ising Model

The error threshold of the toric code maps to the **phase transition** in the random-bond Ising model:

- **Low error rate:** Ordered phase → error correction succeeds
- **High error rate:** Disordered phase → error correction fails

The threshold corresponds to the **Nishimori line** critical point.

---

## Quantum Mechanics Connection

### Anyonic Picture

Errors create anyons:
- **Z-error:** Creates e-m pair (charge-anticharge)
- **X-error:** Creates m-m pair (flux-antiflux)
- **Y-error:** Creates fermion pair (e × m)

**Error correction = anyon annihilation:**
Bring anyons together to annihilate them.

**Logical error = anyon winding:**
If an anyon winds around the torus before annihilation, a logical error occurs.

### Topological vs. Local Protection

**Traditional codes:** Protect via redundancy, local errors affect encoded information
**Topological codes:** Protect via global topology, only non-local errors are dangerous

The toric code is the simplest example of topological protection.

### Connection to Fault Tolerance

For fault-tolerant operation:
1. Syndrome measurement must not spread errors
2. Need repeated syndrome rounds to correct measurement errors
3. Circuit-level noise reduces threshold from ~10% to ~1%

---

## Worked Examples

### Example 1: Syndrome from Z-Error Chain

**Problem:** A Z-error chain occurs on edges 0, 1, 2 (horizontal edges at row 0, columns 0, 1, 2) in a $4 \times 4$ toric code. What is the syndrome?

**Solution:**

Error: $E = Z_0 Z_1 Z_2$

This is a chain of 3 horizontal edges starting at vertex (0, 0) and ending at vertex (0, 3).

**Star syndrome:** Defects at the endpoints of the chain.
- Vertex (0, 0): edge 0 is the right edge, so $A_{(0,0)}$ is violated
- Vertex (0, 3): edge 2 is the left edge (since (0,2,0) connects (0,2) to (0,3)), so $A_{(0,3)}$ is violated

Wait, let me be more careful with indexing.

Edge $(i, j, 0)$ connects vertices $(i, j)$ and $(i, j+1 \mod L)$.

So edges 0, 1, 2 are:
- Edge 0 = $(0, 0, 0)$: connects (0, 0) to (0, 1)
- Edge 1 = $(0, 1, 0)$: connects (0, 1) to (0, 2)
- Edge 2 = $(0, 2, 0)$: connects (0, 2) to (0, 3)

The chain starts at (0, 0) and ends at (0, 3).

**Syndrome:** $A_{(0,0)} = -1$ and $A_{(0,3)} = -1$, all other stars = +1.

### Example 2: Error Correction

**Problem:** Given syndrome with defects at (0, 0) and (0, 3), find the minimum weight correction.

**Solution:**

Defects are at vertices (0, 0) and (0, 3).

**Option 1:** Connect via horizontal path (length 3)
Correction: $Z_0 Z_1 Z_2$ (edges 0, 1, 2)

**Option 2:** Connect via wrap-around path (length 1)
Correction: $Z_3$ (edge 3, which is $(0, 3, 0)$ connecting (0, 3) to (0, 0) via periodic boundary)

Minimum weight = 1.

Apply $Z_3$ to correct.

**Note:** If original error was $Z_0 Z_1 Z_2$, then total operation is:
$$Z_3 \cdot Z_0 Z_1 Z_2 = Z_0 Z_1 Z_2 Z_3 = \bar{Z}_1$$

This is a logical error! The correction succeeded in removing the syndrome but introduced a logical error because it used the wrong path.

**Lesson:** MWPM works well when errors are sparse, but can fail when error chains wrap around.

### Example 3: Logical Error Probability

**Problem:** Estimate the logical error rate for an $L = 5$ toric code with independent Z-errors at rate $p = 0.01$.

**Solution:**

A logical Z error requires a chain of Z-errors spanning the torus (length ≥ L = 5).

**Simple estimate:** Probability of L consecutive errors:
$$P_{logical} \approx \binom{5}{5} p^5 = (0.01)^5 = 10^{-10}$$

But this is too optimistic. A more careful analysis considers all paths.

**Better estimate:** Number of length-L paths ~ $L \cdot 3^{L-1}$ (L starting points, ~3 choices per step).

$$P_{logical} \approx L \cdot 3^{L-1} \cdot p^L = 5 \cdot 81 \cdot 10^{-10} \approx 4 \times 10^{-8}$$

Still very small because $p < p_{th}$.

---

## Practice Problems

### Level 1: Direct Application

**P1.1** A single Z-error occurs on a vertical edge in the interior of a $5 \times 5$ toric code. How many stars are violated?

**P1.2** What is the code distance of a $7 \times 7$ toric code? How many errors can it correct?

**P1.3** If the syndrome shows defects at (1, 2) and (3, 2), what is the minimum distance between them on the lattice?

### Level 2: Intermediate

**P2.1** Show that a closed contractible loop of Z-errors produces no syndrome.

**P2.2** Two defects are at (0, 0) and (0, L/2) on an $L \times L$ torus. There are two minimum-weight paths connecting them. When might MWPM fail?

**P2.3** Calculate the probability that MWPM succeeds for independent Z-errors at rate $p$ when the minimum matching is unique.

### Level 3: Challenging

**P3.1** Prove that the error threshold for the toric code is equivalent to the order-disorder transition in the 2D random-bond Ising model on the Nishimori line.

**P3.2** For a toric code with Y-errors at rate $p$, derive an expression for the logical error rate to leading order in $p$.

**P3.3** Compare the qubit overhead of the toric code with the concatenated [[7,1,3]] code for achieving logical error rate $10^{-10}$.

---

## Computational Lab

```python
"""
Day 790: Error Model and Distance
==================================

Simulating errors and analyzing code distance for the toric code.
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
import random
from collections import defaultdict


class ToricCodeErrors:
    """
    Error simulation and analysis for the toric code.
    """

    def __init__(self, L: int):
        """Initialize L x L toric code."""
        self.L = L
        self.n_qubits = 2 * L * L

    def edge_index(self, i: int, j: int, d: int) -> int:
        """Convert (i, j, d) to linear edge index."""
        i, j = i % self.L, j % self.L
        return d * self.L**2 + i * self.L + j

    def edge_from_index(self, idx: int) -> Tuple[int, int, int]:
        """Convert linear index to (i, j, d)."""
        d = idx // (self.L ** 2)
        rem = idx % (self.L ** 2)
        i = rem // self.L
        j = rem % self.L
        return (i, j, d)

    def star_support(self, vi: int, vj: int) -> Set[int]:
        """Return edges in star operator at vertex (vi, vj)."""
        vi, vj = vi % self.L, vj % self.L
        return {
            self.edge_index(vi, vj, 0),
            self.edge_index(vi, vj - 1, 0),
            self.edge_index(vi, vj, 1),
            self.edge_index(vi - 1, vj, 1),
        }

    def plaquette_support(self, pi: int, pj: int) -> Set[int]:
        """Return edges in plaquette operator at face (pi, pj)."""
        pi, pj = pi % self.L, pj % self.L
        return {
            self.edge_index(pi, pj, 0),
            self.edge_index(pi + 1, pj, 0),
            self.edge_index(pi, pj, 1),
            self.edge_index(pi, pj + 1, 1),
        }

    def compute_syndrome(self, z_errors: Set[int], x_errors: Set[int]) -> Tuple[Set[Tuple], Set[Tuple]]:
        """
        Compute syndrome from errors.

        Returns:
            (star_defects, plaquette_defects) - sets of (i, j) coordinates
        """
        star_defects = set()
        plaquette_defects = set()

        # Z-errors create star defects
        for i in range(self.L):
            for j in range(self.L):
                star = self.star_support(i, j)
                parity = len(star & z_errors) % 2
                if parity == 1:
                    star_defects.add((i, j))

        # X-errors create plaquette defects
        for i in range(self.L):
            for j in range(self.L):
                plaq = self.plaquette_support(i, j)
                parity = len(plaq & x_errors) % 2
                if parity == 1:
                    plaquette_defects.add((i, j))

        return star_defects, plaquette_defects

    def manhattan_distance_torus(self, v1: Tuple[int, int], v2: Tuple[int, int]) -> int:
        """Compute Manhattan distance on torus between two vertices."""
        i1, j1 = v1
        i2, j2 = v2

        di = min(abs(i2 - i1), self.L - abs(i2 - i1))
        dj = min(abs(j2 - j1), self.L - abs(j2 - j1))

        return di + dj

    def generate_random_errors(self, p_z: float, p_x: float) -> Tuple[Set[int], Set[int]]:
        """Generate random errors with given probabilities."""
        z_errors = set()
        x_errors = set()

        for e in range(self.n_qubits):
            if random.random() < p_z:
                z_errors.add(e)
            if random.random() < p_x:
                x_errors.add(e)

        return z_errors, x_errors

    def simple_mwpm_decode(self, defects: Set[Tuple[int, int]]) -> List[Tuple[Tuple, Tuple]]:
        """
        Simple MWPM decoder (greedy, not optimal).

        Returns list of matched pairs.
        """
        defects = list(defects)
        pairs = []

        while len(defects) >= 2:
            # Find closest pair (greedy)
            best_dist = float('inf')
            best_pair = None

            for i in range(len(defects)):
                for j in range(i + 1, len(defects)):
                    dist = self.manhattan_distance_torus(defects[i], defects[j])
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (i, j)

            if best_pair:
                i, j = best_pair
                pairs.append((defects[i], defects[j]))
                # Remove in reverse order to preserve indices
                defects.pop(j)
                defects.pop(i)

        return pairs

    def correction_weight(self, v1: Tuple[int, int], v2: Tuple[int, int]) -> int:
        """Minimum correction weight to connect two vertices."""
        return self.manhattan_distance_torus(v1, v2)

    def check_logical_error(self, original_errors: Set[int], correction: Set[int],
                           error_type: str) -> bool:
        """
        Check if correction introduces a logical error.

        error_type: 'Z' or 'X'
        """
        total = original_errors.symmetric_difference(correction)

        if error_type == 'Z':
            # Check if total is equivalent to logical Z
            # (forms non-contractible loop)
            pass  # Simplified: would need full homology check

        return False  # Placeholder


def analyze_code_distance(L: int) -> None:
    """Analyze code distance for L x L toric code."""
    print(f"\n{'='*60}")
    print(f"Code Distance Analysis for L = {L}")
    print(f"{'='*60}")

    code = ToricCodeErrors(L)

    d = L
    t = (d - 1) // 2

    print(f"\nCode parameters: [[{code.n_qubits}, 2, {d}]]")
    print(f"Distance: d = {d}")
    print(f"Correctable errors: t = floor((d-1)/2) = {t}")

    # Minimum logical operator
    print(f"\nMinimum logical Z operator: horizontal loop of {L} edges")
    print(f"Minimum logical X operator: vertical loop of {L} edges")


def simulate_error_correction(L: int, p: float, n_trials: int = 1000) -> Dict:
    """Simulate error correction for toric code."""
    print(f"\n{'='*60}")
    print(f"Error Correction Simulation (L = {L}, p = {p})")
    print(f"{'='*60}")

    code = ToricCodeErrors(L)

    successes = 0
    total_weight = 0

    for _ in range(n_trials):
        # Generate Z-errors only for simplicity
        z_errors, _ = code.generate_random_errors(p, 0)

        # Compute syndrome
        star_defects, _ = code.compute_syndrome(z_errors, set())

        if len(star_defects) == 0:
            successes += 1
            continue

        # Decode
        pairs = code.simple_mwpm_decode(star_defects)

        # Calculate correction weight
        weight = sum(code.correction_weight(v1, v2) for v1, v2 in pairs)
        total_weight += weight

        # Simplified success criterion: if correction weight < L, likely success
        # (This is a heuristic, not exact)
        if weight < L // 2:
            successes += 1

    results = {
        'success_rate': successes / n_trials,
        'avg_correction_weight': total_weight / n_trials if n_trials > 0 else 0,
    }

    print(f"\nResults over {n_trials} trials:")
    print(f"  Success rate (heuristic): {results['success_rate']:.2%}")
    print(f"  Avg correction weight: {results['avg_correction_weight']:.2f}")

    return results


def demonstrate_error_syndromes(L: int) -> None:
    """Demonstrate syndrome patterns for various errors."""
    print(f"\n{'='*60}")
    print(f"Error Syndrome Demonstration (L = {L})")
    print(f"{'='*60}")

    code = ToricCodeErrors(L)

    # Single Z-error on interior edge
    print("\n1. Single Z-error on edge (1, 1, 0) [horizontal, interior]:")
    z_errors = {code.edge_index(1, 1, 0)}
    stars, plaqs = code.compute_syndrome(z_errors, set())
    print(f"   Z-errors: edge {list(z_errors)}")
    print(f"   Star defects: {stars}")
    print(f"   Plaquette defects: {plaqs}")

    # Single X-error
    print("\n2. Single X-error on edge (1, 1, 1) [vertical, interior]:")
    x_errors = {code.edge_index(1, 1, 1)}
    stars, plaqs = code.compute_syndrome(set(), x_errors)
    print(f"   X-errors: edge {list(x_errors)}")
    print(f"   Star defects: {stars}")
    print(f"   Plaquette defects: {plaqs}")

    # Chain of Z-errors (open)
    print("\n3. Chain of Z-errors (open, length 3):")
    z_chain = {code.edge_index(0, j, 0) for j in range(3)}
    stars, plaqs = code.compute_syndrome(z_chain, set())
    print(f"   Z-errors: edges {sorted(z_chain)}")
    print(f"   Star defects: {stars}")
    print(f"   (Two defects at chain endpoints)")

    # Closed loop of Z-errors (contractible)
    print("\n4. Closed loop of Z-errors (single plaquette boundary):")
    z_loop = code.plaquette_support(1, 1)
    stars, plaqs = code.compute_syndrome(z_loop, set())
    print(f"   Z-errors: edges {sorted(z_loop)}")
    print(f"   Star defects: {stars}")
    print(f"   (No defects - closed contractible loop!)")


def threshold_estimate(L_values: List[int], p_values: List[float],
                       n_trials: int = 100) -> None:
    """Estimate threshold by varying L and p."""
    print(f"\n{'='*60}")
    print(f"Threshold Estimation")
    print(f"{'='*60}")

    results = defaultdict(list)

    for L in L_values:
        for p in p_values:
            code = ToricCodeErrors(L)
            successes = 0

            for _ in range(n_trials):
                z_errors, _ = code.generate_random_errors(p, 0)
                stars, _ = code.compute_syndrome(z_errors, set())

                # Very simplified: success if number of defects is small
                if len(stars) <= L // 2:
                    successes += 1

            results[L].append((p, successes / n_trials))

    print("\nSuccess rate by L and p:")
    print(f"{'p':>8}", end='')
    for L in L_values:
        print(f"{L:>10}", end='')
    print()

    for i, p in enumerate(p_values):
        print(f"{p:>8.3f}", end='')
        for L in L_values:
            rate = results[L][i][1]
            print(f"{rate:>10.2%}", end='')
        print()


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 790: ERROR MODEL AND CODE DISTANCE")
    print("=" * 70)

    # Demo 1: Code distance analysis
    for L in [3, 5, 7, 10]:
        analyze_code_distance(L)

    # Demo 2: Error syndromes
    demonstrate_error_syndromes(5)

    # Demo 3: Error correction simulation
    for L in [5, 7]:
        for p in [0.01, 0.05, 0.10]:
            simulate_error_correction(L, p, n_trials=500)

    # Demo 4: Threshold estimation
    threshold_estimate([3, 5, 7], [0.02, 0.05, 0.08, 0.10, 0.12], n_trials=200)

    # Demo 5: Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Error Model and Code Distance")
    print("=" * 70)

    print("""
    ERROR MODEL:
    ------------
    Z-error on edge e: Creates star defects at e's two endpoints
    X-error on edge e: Creates plaquette defects at e's two adjacent faces
    Y-error = i*XZ: Creates both types of defects

    ERROR CHAINS:
    -------------
    - Open chain: Defects at endpoints
    - Closed contractible chain: No defects (equivalent to stabilizer)
    - Closed non-contractible chain: No defects, but logical error!

    CODE DISTANCE:
    --------------
    d = L (minimum non-contractible loop length)
    Can correct t = floor((L-1)/2) errors

    ERROR THRESHOLD:
    ----------------
    p_th ~ 10.9% for independent Z or X errors
    p_th ~ 2-3% for phenomenological noise
    p_th ~ 0.5-1% for circuit-level noise

    Below threshold: Logical error rate -> 0 as L -> infinity
    Above threshold: No protection, logical error rate -> 1

    DECODER:
    --------
    MWPM (Minimum Weight Perfect Matching):
    1. Identify defect pairs from syndrome
    2. Match defects with minimum total path length
    3. Apply correction along matched paths

    SCALING:
    --------
    P_logical ~ exp(-c * L) for p < p_th
    Polynomial overhead: n ~ d^2 qubits for distance d
    """)

    print("=" * 70)
    print("Day 790 Complete: Error Model and Distance Mastered")
    print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Code parameters | $[[2L^2, 2, L]]$ |
| Code distance | $d = L$ |
| Correctable errors | $t = \lfloor (L-1)/2 \rfloor$ |
| Error threshold | $p_{th} \approx 10.9\%$ (independent noise) |
| Logical error rate | $P_{logical} \sim e^{-cL}$ for $p < p_{th}$ |

### Main Takeaways

1. **Single errors create defect pairs:** Z-errors → star defects, X-errors → plaquette defects
2. **Error chains:** Open chains create defects at endpoints, closed chains may create logical errors
3. **Code distance = L:** Minimum non-contractible loop length
4. **MWPM decoder:** Matches defect pairs with minimum total path length
5. **Threshold ~10.9%:** Below this, exponential suppression of logical errors
6. **Topological protection:** Only errors spanning the torus cause logical faults

---

## Daily Checklist

- [ ] I understand how X, Z, Y errors create syndromes
- [ ] I can trace error chains and identify their defects
- [ ] I understand why code distance = L
- [ ] I can describe the MWPM decoding algorithm
- [ ] I know the error threshold and its significance
- [ ] I ran the computational lab and observed threshold behavior

---

## Preview: Day 791

Tomorrow we complete **Week 113 Synthesis**:

- Comprehensive concept map of the toric code
- Master formula reference sheet
- Integration problems combining all concepts
- Preview of anyons and topological order (Week 114)

The synthesis day consolidates all toric code fundamentals before moving to anyonic excitations.
