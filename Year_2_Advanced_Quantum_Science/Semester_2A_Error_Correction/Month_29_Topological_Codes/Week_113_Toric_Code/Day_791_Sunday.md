# Day 791: Week 113 Synthesis - Toric Code Fundamentals

## Overview

**Day:** 791 of 1008
**Week:** 113 (Toric Code Fundamentals)
**Month:** 29 (Topological Codes)
**Topic:** Week Synthesis and Integration

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Concept map and formula review |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Comprehensive problems |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Week 114 preview and preparation |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Synthesize** all toric code concepts into a unified picture
2. **Apply** toric code theory to multi-step problems
3. **Derive** key results from first principles
4. **Connect** the toric code to broader QEC and physics contexts
5. **Prepare** for anyonic excitations and topological order
6. **Assess** mastery of Week 113 material

---

## Concept Map

```
                          TORIC CODE
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
       LATTICE            OPERATORS           TOPOLOGY
           │                  │                  │
    ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐
    │             │    │             │    │             │
  Torus      Qubits   Stars    Plaquettes  Homology  Cycles
  L×L        on       A_v        B_p       H_1       γ₁,γ₂
             edges                                    │
    │             │    │             │                │
    └─────┬───────┘    └──────┬──────┘         ┌─────┴─────┐
          │                   │                │           │
       n=2L²           Stabilizer Group    Logical    Code
          │                   │            Operators  Distance
          │            ┌──────┴──────┐         │         │
          │            │             │         │         │
          └────────────┤  CSS Code   ├─────────┘       d=L
                       │             │
                       │   H_X,H_Z   │
                       │  ∂₁∂₂ = 0   │
                       └──────┬──────┘
                              │
                    Ground State Degeneracy = 4
                              │
                       Topological Protection
                              │
                    Error Threshold ~ 10.9%
```

---

## Master Formula Reference

### Lattice and Code Parameters

| Quantity | Formula | Description |
|----------|---------|-------------|
| Physical qubits | $n = 2L^2$ | Edges on $L \times L$ torus |
| Logical qubits | $k = 2$ | Non-contractible cycles |
| Code distance | $d = L$ | Minimum non-trivial loop |
| Code notation | $[[2L^2, 2, L]]$ | Standard form |
| Encoding rate | $R = 1/L^2$ | k/n |
| Vertices | $V = L^2$ | Lattice vertices |
| Faces | $F = L^2$ | Lattice faces |
| Euler characteristic | $\chi = V - E + F = 0$ | Torus topology |

### Stabilizer Operators

| Operator | Formula | Type |
|----------|---------|------|
| Star operator | $A_v = \prod_{e \ni v} X_e$ | X-type (vertex) |
| Plaquette operator | $B_p = \prod_{e \in \partial p} Z_e$ | Z-type (face) |
| Number of stars | $L^2$ | Vertices |
| Number of plaquettes | $L^2$ | Faces |
| Independent generators | $2L^2 - 2$ | Two constraints |
| Star constraint | $\prod_v A_v = I$ | Product identity |
| Plaquette constraint | $\prod_p B_p = I$ | Product identity |

### Commutation Relations

| Relation | Formula | Reason |
|----------|---------|--------|
| Star-Star | $[A_v, A_{v'}] = 0$ | X operators commute |
| Plaquette-Plaquette | $[B_p, B_{p'}] = 0$ | Z operators commute |
| Star-Plaquette | $[A_v, B_p] = 0$ | Even overlap (0 or 2) |

### Ground State and Hamiltonian

| Quantity | Formula | Description |
|----------|---------|-------------|
| Ground state condition | $A_v\|\psi\rangle = B_p\|\psi\rangle = +1\|\psi\rangle$ | All +1 eigenvalues |
| Hamiltonian | $H = -\sum_v A_v - \sum_p B_p$ | Sum of stabilizers |
| Ground energy | $E_0 = -2L^2$ | All terms -1 |
| Energy gap | $\Delta = 2$ | Single violation |
| Ground degeneracy | 4 | Topological |

### CSS Structure

| Component | Formula | Description |
|-----------|---------|-------------|
| X-stabilizer matrix | $[H_X]_{v,e} = 1$ if $e \ni v$ | Incidence matrix |
| Z-stabilizer matrix | $[H_Z]_{p,e} = 1$ if $e \in \partial p$ | Boundary matrix |
| CSS condition | $H_X H_Z^T = 0 \pmod{2}$ | Commutation |
| Chain complex | $C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0$ | Faces → Edges → Vertices |
| Boundary relation | $\partial_1 \partial_2 = 0$ | Closed boundary |

### Logical Operators

| Operator | Formula | Support |
|----------|---------|---------|
| Logical $\bar{Z}_1$ | $\prod_{j=0}^{L-1} Z_{(0,j,0)}$ | Horizontal loop |
| Logical $\bar{Z}_2$ | $\prod_{i=0}^{L-1} Z_{(i,0,1)}$ | Vertical loop |
| Logical $\bar{X}_1$ | $\prod_{i=0}^{L-1} X_{(i,0,1)}$ | Vertical string |
| Logical $\bar{X}_2$ | $\prod_{j=0}^{L-1} X_{(0,j,0)}$ | Horizontal string |
| Weight | $\text{wt}(\bar{O}) = L$ | Code distance |

### Error Correction

| Quantity | Formula | Description |
|----------|---------|-------------|
| Error threshold | $p_{th} \approx 10.9\%$ | Independent noise |
| Correctable errors | $t = \lfloor(L-1)/2\rfloor$ | Maximum correctable |
| Logical error rate | $P_L \sim e^{-cL}$ | Below threshold |
| Z-error syndrome | Star violations at endpoints | Charge defects |
| X-error syndrome | Plaquette violations | Flux defects |

---

## Comprehensive Problems

### Problem Set A: Foundational (30 minutes)

**A1. Lattice Counting**
For a $6 \times 6$ toric code:
a) Calculate the number of physical qubits
b) How many stabilizer generators are there?
c) How many are independent?
d) What are the code parameters?

**A2. Operator Construction**
Write the star operator $A_{(2,3)}$ for a $5 \times 5$ toric code:
a) List all edge indices involved
b) Write the operator as a product of Paulis
c) Verify it has 4 terms

**A3. Commutation Proof**
Prove that $A_{(0,0)}$ and $B_{(0,0)}$ commute in the $4 \times 4$ toric code by:
a) Listing edges in each operator
b) Finding the intersection
c) Computing the commutation factor

### Problem Set B: Intermediate (45 minutes)

**B1. Ground State Analysis**
For the $3 \times 3$ toric code:
a) Write the Hamiltonian explicitly
b) Calculate the ground state energy
c) What is the energy of a state with exactly one star violation?
d) How many distinct excited energy levels exist?

**B2. CSS Structure**
Construct the parity check matrices $H_X$ and $H_Z$ for the $2 \times 2$ toric code:
a) Write both matrices explicitly
b) Verify $H_X H_Z^T = 0 \pmod{2}$
c) Compute the ranks and confirm $k = 2$

**B3. Logical Operators**
For a $4 \times 4$ toric code:
a) Write explicit expressions for all 4 logical operators
b) Verify that $\bar{X}_1$ commutes with all stabilizers
c) Show that $\bar{X}_1$ and $\bar{Z}_2$ have specific commutation

### Problem Set C: Advanced (60 minutes)

**C1. Error Correction Scenario**
A $5 \times 5$ toric code experiences Z-errors on edges at indices 0, 1, 5, 6.
a) Determine the syndrome (which stars are violated)
b) Propose a minimum-weight correction
c) Check if your correction plus the error forms a logical operator
d) What is the probability of logical error if errors were random at rate $p$?

**C2. Threshold Analysis**
Derive an approximate expression for the logical error rate in terms of $L$ and $p$:
a) Count the number of minimum-weight logical operators
b) Estimate the probability of each occurring
c) Show that below threshold, $P_L \to 0$ as $L \to \infty$
d) Above threshold, argue that $P_L \to 1$

**C3. Generalization**
Consider a rectangular torus with dimensions $L_1 \times L_2$ (not necessarily equal):
a) What are the code parameters $[[n, k, d]]$?
b) What are the weights of the four logical operators?
c) For $L_1 = 3$, $L_2 = 7$, what is the code distance?
d) Design an asymmetric code optimized for protecting against Z-errors more than X-errors

---

## Solutions to Comprehensive Problems

### Solution A1: Lattice Counting (L = 6)

a) Physical qubits: $n = 2 \times 6^2 = 72$

b) Stabilizer generators: $L^2 + L^2 = 36 + 36 = 72$

c) Independent generators: $72 - 2 = 70$ (two constraints)

d) Code parameters: $[[72, 2, 6]]$

### Solution A2: Star Operator at (2,3) for L = 5

Edge index formula: $\text{idx}(i, j, d) = d \cdot 25 + i \cdot 5 + j$

Edges at vertex (2, 3):
- Right horizontal: $(2, 3, 0)$ → $0 \cdot 25 + 2 \cdot 5 + 3 = 13$
- Left horizontal: $(2, 2, 0)$ → $0 \cdot 25 + 2 \cdot 5 + 2 = 12$
- Down vertical: $(2, 3, 1)$ → $1 \cdot 25 + 2 \cdot 5 + 3 = 38$
- Up vertical: $(1, 3, 1)$ → $1 \cdot 25 + 1 \cdot 5 + 3 = 33$

$$A_{(2,3)} = X_{12} X_{13} X_{33} X_{38}$$

### Solution B1: Ground State Analysis (L = 3)

a) $H = -\sum_{v=0}^{8} A_v - \sum_{p=0}^{8} B_p$ (9 star + 9 plaquette terms)

b) Ground energy: $E_0 = -9 - 9 = -18$

c) One star violation: $E_1 = -18 + 2 = -16$ (flips one eigenvalue +1 → -1)

d) Energy levels: $E = -18 + 2m$ where $m$ is number of violations.
   - $m = 0$: E = -18 (ground)
   - $m = 1$: E = -16 (impossible! defects come in pairs)
   - $m = 2$: E = -14 (minimum excitation)
   - And so on...

   Actually, star and plaquette violations are independent, so:
   - $(n_s, n_p)$ violations give $E = -18 + 2n_s + 2n_p$
   - Constraints: $n_s$ and $n_p$ must be even

### Solution C1: Error Correction Scenario

Edges 0, 1, 5, 6 for L = 5:
- Edge 0 = $(0, 0, 0)$: connects (0,0) to (0,1)
- Edge 1 = $(0, 1, 0)$: connects (0,1) to (0,2)
- Edge 5 = $(1, 0, 0)$: connects (1,0) to (1,1)
- Edge 6 = $(1, 1, 0)$: connects (1,1) to (1,2)

The syndrome involves stars at vertices touched by odd number of error edges.

Chain 1: 0-1 forms path from (0,0) to (0,2)
Chain 2: 5-6 forms path from (1,0) to (1,2)

Defects: (0,0), (0,2), (1,0), (1,2)

Minimum correction: Match (0,0)-(1,0) and (0,2)-(1,2) with vertical edges.

---

## Week 113 Self-Assessment

Rate your understanding (1-5) for each topic:

| Topic | Day | Self-Rating | Review Needed? |
|-------|-----|-------------|----------------|
| Lattice structure and qubit placement | 785 | ___ | ___ |
| Star operators | 786 | ___ | ___ |
| Plaquette operators | 786 | ___ | ___ |
| Commutation proofs | 786 | ___ | ___ |
| Ground state condition | 787 | ___ | ___ |
| Ground state degeneracy | 787 | ___ | ___ |
| Hamiltonian and energy gap | 787 | ___ | ___ |
| CSS code structure | 788 | ___ | ___ |
| Chain complex and homology | 788 | ___ | ___ |
| Logical Z operators | 789 | ___ | ___ |
| Logical X operators | 789 | ___ | ___ |
| Commutation/anti-commutation | 789 | ___ | ___ |
| Error syndromes | 790 | ___ | ___ |
| Code distance | 790 | ___ | ___ |
| Error threshold | 790 | ___ | ___ |
| MWPM decoding | 790 | ___ | ___ |

**Scoring:**
- 4-5: Strong understanding, ready for Week 114
- 3: Adequate, minor review recommended
- 1-2: Significant review needed before proceeding

---

## Preview: Week 114 - Anyons and Topological Order

Next week explores the **physical interpretation** of toric code excitations:

### Day 792: Anyonic Excitations
- Star violations as "electric charges" (e particles)
- Plaquette violations as "magnetic fluxes" (m particles)
- Particle creation by string operators

### Day 793: Fusion Rules
- $e \times e = 1$ (charge neutralization)
- $m \times m = 1$ (flux cancellation)
- $e \times m = \epsilon$ (composite fermion)

### Day 794: Braiding Statistics
- Moving e around m gives phase $-1$
- Mutual semionic statistics
- Connection to Aharonov-Bohm effect

### Day 795: Topological Order
- Long-range entanglement
- Ground state degeneracy from topology
- Local indistinguishability

### Day 796: Topological Entanglement Entropy
- $S = \alpha |\partial A| - \gamma$
- $\gamma = \log D$ where $D = 2$
- Signature of topological order

### Day 797: Topological Quantum Memory
- Thermal stability
- Memory time scaling
- Self-correcting codes

### Day 798: Week 114 Synthesis
- Integration of anyon concepts
- Connection to topological quantum computing

---

## Computational Lab: Complete Toric Code Simulator

```python
"""
Day 791: Week 113 Synthesis - Complete Toric Code Simulator
============================================================

A comprehensive implementation combining all Week 113 concepts.
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
import random


@dataclass
class ToricCodeComplete:
    """
    Complete toric code implementation for L x L torus.

    Combines all functionality from Week 113.
    """
    L: int

    def __post_init__(self):
        """Initialize derived quantities."""
        self.n_qubits = 2 * self.L**2
        self.n_vertices = self.L**2
        self.n_faces = self.L**2
        self._build_stabilizers()
        self._build_logical_ops()

    # === Indexing ===

    def edge_idx(self, i: int, j: int, d: int) -> int:
        """Edge (i, j, d) to linear index."""
        i, j = i % self.L, j % self.L
        return d * self.L**2 + i * self.L + j

    def vertex_idx(self, i: int, j: int) -> int:
        """Vertex (i, j) to linear index."""
        return (i % self.L) * self.L + (j % self.L)

    def face_idx(self, i: int, j: int) -> int:
        """Face (i, j) to linear index."""
        return (i % self.L) * self.L + (j % self.L)

    # === Stabilizers ===

    def _build_stabilizers(self):
        """Build star and plaquette supports."""
        self.stars = []
        self.plaquettes = []

        for i in range(self.L):
            for j in range(self.L):
                self.stars.append(frozenset([
                    self.edge_idx(i, j, 0),
                    self.edge_idx(i, j-1, 0),
                    self.edge_idx(i, j, 1),
                    self.edge_idx(i-1, j, 1),
                ]))

                self.plaquettes.append(frozenset([
                    self.edge_idx(i, j, 0),
                    self.edge_idx(i+1, j, 0),
                    self.edge_idx(i, j, 1),
                    self.edge_idx(i, j+1, 1),
                ]))

    # === Logical Operators ===

    def _build_logical_ops(self):
        """Build logical operator supports."""
        self.Z1 = frozenset([self.edge_idx(0, j, 0) for j in range(self.L)])
        self.Z2 = frozenset([self.edge_idx(i, 0, 1) for i in range(self.L)])
        self.X1 = frozenset([self.edge_idx(i, 0, 1) for i in range(self.L)])
        self.X2 = frozenset([self.edge_idx(0, j, 0) for j in range(self.L)])

    # === Parity Check Matrices ===

    def H_X(self) -> np.ndarray:
        """X-stabilizer (star) parity check matrix."""
        H = np.zeros((self.n_vertices, self.n_qubits), dtype=int)
        for v, star in enumerate(self.stars):
            for e in star:
                H[v, e] = 1
        return H

    def H_Z(self) -> np.ndarray:
        """Z-stabilizer (plaquette) parity check matrix."""
        H = np.zeros((self.n_faces, self.n_qubits), dtype=int)
        for p, plaq in enumerate(self.plaquettes):
            for e in plaq:
                H[p, e] = 1
        return H

    # === Code Properties ===

    def code_parameters(self) -> Tuple[int, int, int]:
        """Return [[n, k, d]]."""
        return (self.n_qubits, 2, self.L)

    def verify_css_condition(self) -> bool:
        """Verify H_X @ H_Z.T = 0 mod 2."""
        product = (self.H_X() @ self.H_Z().T) % 2
        return np.all(product == 0)

    def ground_state_energy(self) -> float:
        """Return ground state energy."""
        return -float(self.n_vertices + self.n_faces)

    # === Error Simulation ===

    def random_errors(self, p_z: float = 0, p_x: float = 0) -> Tuple[Set[int], Set[int]]:
        """Generate random Z and X errors."""
        z_err = {e for e in range(self.n_qubits) if random.random() < p_z}
        x_err = {e for e in range(self.n_qubits) if random.random() < p_x}
        return z_err, x_err

    def syndrome(self, z_errors: Set[int], x_errors: Set[int]) -> Tuple[Set[int], Set[int]]:
        """Compute syndrome from errors."""
        star_defects = {v for v, star in enumerate(self.stars)
                       if len(star & z_errors) % 2 == 1}
        plaq_defects = {p for p, plaq in enumerate(self.plaquettes)
                       if len(plaq & x_errors) % 2 == 1}
        return star_defects, plaq_defects

    # === Decoding ===

    def torus_distance(self, v1: int, v2: int) -> int:
        """Manhattan distance between vertices on torus."""
        i1, j1 = v1 // self.L, v1 % self.L
        i2, j2 = v2 // self.L, v2 % self.L
        di = min(abs(i2-i1), self.L - abs(i2-i1))
        dj = min(abs(j2-j1), self.L - abs(j2-j1))
        return di + dj

    def greedy_match(self, defects: Set[int]) -> List[Tuple[int, int]]:
        """Greedy MWPM (not optimal, but simple)."""
        defects = list(defects)
        pairs = []
        while len(defects) >= 2:
            best = min(((i, j, self.torus_distance(defects[i], defects[j]))
                       for i in range(len(defects))
                       for j in range(i+1, len(defects))),
                      key=lambda x: x[2])
            i, j, _ = best
            pairs.append((defects[i], defects[j]))
            defects = [d for k, d in enumerate(defects) if k not in (i, j)]
        return pairs

    # === Verification ===

    def verify_all_commutations(self) -> bool:
        """Verify all stabilizer commutations."""
        for star in self.stars:
            for plaq in self.plaquettes:
                if len(star & plaq) % 2 != 0:
                    return False
        return True

    def verify_logical_commutations(self) -> Dict[str, str]:
        """Verify logical operator commutations."""
        results = {}
        for name, (X_op, Z_op) in [
            ('X1_Z1', (self.X1, self.Z1)),
            ('X1_Z2', (self.X1, self.Z2)),
            ('X2_Z1', (self.X2, self.Z1)),
            ('X2_Z2', (self.X2, self.Z2)),
        ]:
            overlap = len(X_op & Z_op)
            comm = 'commute' if overlap % 2 == 0 else 'anti-commute'
            results[name] = f'{overlap} overlap -> {comm}'
        return results

    # === Summary ===

    def summary(self) -> str:
        """Return complete summary of toric code."""
        n, k, d = self.code_parameters()
        return f"""
        Toric Code Summary (L = {self.L})
        ================================
        Code parameters: [[{n}, {k}, {d}]]

        Lattice:
          Vertices: {self.n_vertices}
          Edges: {self.n_qubits}
          Faces: {self.n_faces}
          Euler characteristic: {self.n_vertices - self.n_qubits + self.n_faces}

        Stabilizers:
          Star operators: {len(self.stars)}
          Plaquette operators: {len(self.plaquettes)}
          Independent generators: {2*self.L**2 - 2}
          CSS condition satisfied: {self.verify_css_condition()}
          All commutations valid: {self.verify_all_commutations()}

        Hamiltonian:
          Ground state energy: {self.ground_state_energy()}
          Energy gap: 2

        Logical operators:
          Z1 weight: {len(self.Z1)}
          Z2 weight: {len(self.Z2)}
          X1 weight: {len(self.X1)}
          X2 weight: {len(self.X2)}
          Commutations: {self.verify_logical_commutations()}

        Error correction:
          Code distance: {d}
          Correctable errors: {(d-1)//2}
          Threshold (independent): ~10.9%
        """


# ============================================================
# Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 791: WEEK 113 SYNTHESIS - TORIC CODE FUNDAMENTALS")
    print("=" * 70)

    # Complete demonstration for various L
    for L in [3, 5, 7]:
        code = ToricCodeComplete(L)
        print(code.summary())

    # Error simulation demo
    print("\n" + "=" * 70)
    print("Error Simulation Demo (L = 5, p = 0.05)")
    print("=" * 70)

    code = ToricCodeComplete(5)
    z_err, x_err = code.random_errors(p_z=0.05, p_x=0.05)
    star_def, plaq_def = code.syndrome(z_err, x_err)

    print(f"Z-errors on edges: {sorted(z_err)}")
    print(f"X-errors on edges: {sorted(x_err)}")
    print(f"Star defects: {star_def}")
    print(f"Plaquette defects: {plaq_def}")

    if star_def:
        pairs = code.greedy_match(star_def)
        print(f"MWPM pairing for Z-errors: {pairs}")

    print("\n" + "=" * 70)
    print("Week 113 Complete: Toric Code Fundamentals Mastered!")
    print("=" * 70)
    print("\nNext: Week 114 - Anyons and Topological Order")
```

---

## Summary

### Week 113 Key Takeaways

1. **Toric code** is a topological quantum error-correcting code on a torus with $[[2L^2, 2, L]]$ parameters

2. **Star and plaquette operators** form a CSS stabilizer structure with all operators commuting

3. **Ground state** is the simultaneous +1 eigenspace of all stabilizers, with 4-fold degeneracy encoding 2 logical qubits

4. **Chain complex** $C_2 \to C_1 \to C_0$ with $\partial_1\partial_2 = 0$ underlies the CSS structure

5. **Logical operators** are non-contractible loops with minimum weight L = code distance

6. **Errors** create defect pairs (anyons) that can be corrected by MWPM decoding

7. **Error threshold** ~10.9% makes the toric code practically attractive

### Connections Made

- Stabilizer formalism → Toric code stabilizers
- CSS codes → X/Z separation in toric code
- Algebraic topology → Homology and logical operators
- Condensed matter physics → Topological order
- Fault tolerance → Error thresholds

### Looking Forward

Week 114 will reveal the **physical meaning** of toric code excitations as anyonic quasiparticles with exotic braiding statistics, connecting quantum error correction to topological quantum computing.

---

## Daily Checklist

- [ ] Completed concept map review
- [ ] Mastered all formulas in reference sheet
- [ ] Solved comprehensive problems A, B, C
- [ ] Achieved score 4+ on all self-assessment items
- [ ] Ran complete simulator code
- [ ] Prepared for Week 114 topics

---

## Congratulations!

You have completed **Week 113: Toric Code Fundamentals**. The toric code represents a paradigm shift in quantum error correction - from algebraic redundancy to topological protection. This foundation will support your understanding of:

- Anyonic excitations and topological order
- Surface codes and practical implementations
- Topological quantum computing
- Advanced QEC architectures

Proceed to Week 114 to explore the fascinating world of anyons!
