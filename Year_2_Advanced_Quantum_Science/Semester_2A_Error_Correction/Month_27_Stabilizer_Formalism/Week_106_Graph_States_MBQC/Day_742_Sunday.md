# Day 742: Week 106 Synthesis

## Overview

**Day:** 742 of 1008
**Week:** 106 (Graph States & MBQC) — Final Day
**Month:** 27 (Stabilizer Formalism)
**Topic:** Comprehensive Review and Integration

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Week review |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Integration problems |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Preparation for Week 107 |

---

## Week 106 Concept Map

```
                      GRAPH STATES
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    CONSTRUCTION      STABILIZERS       EQUIVALENCE
          │                │                │
     ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
     ▼         ▼      ▼         ▼      ▼         ▼
  CZ Gates  Adjacency  K_a=X_aZ^N(a)  Binary    LC
   |+⟩^n    Matrix    Formula        Form    Operations
     │         │           │           │         │
     └─────────┴───────────┴───────────┴─────────┘
                           │
                    LC EQUIVALENCE
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     INVARIANTS       ORBITS           MBQC
          │              │                │
     ┌────┴────┐    ┌────┴────┐     ┌────┴────┐
     ▼         ▼    ▼         ▼     ▼         ▼
   Rank   Interlace  Orbit    Canon  1D/2D   Universal
   F_2    Graph     Size     Form  Clusters  Computation
```

---

## Week 106 Daily Summary

| Day | Topic | Key Results |
|-----|-------|-------------|
| 736 | Graph State Introduction | $\|G\rangle = \prod CZ_{ij}\|+\rangle^n$, amplitude formula |
| 737 | Stabilizer Structure | $K_a = X_a Z^{N(a)}$, H = (I\|Γ) |
| 738 | Local Complementation | Toggle N(a) edges, $U_a = e^{-iπX/4}\prod\sqrt{Z}$ |
| 739 | LC Equivalence | Invariants, orbits, classification |
| 740 | MBQC Foundations | Wire, rotations, byproducts |
| 741 | Cluster States | 2D universality, depth, errors |

---

## Master Formula Sheet

### Graph States

$$\boxed{|G\rangle = \prod_{(i,j) \in E} CZ_{ij} |+\rangle^{\otimes n}}$$

$$\boxed{\langle x | G \rangle = \frac{1}{\sqrt{2^n}}(-1)^{q(x)}, \quad q(x) = \sum_{i<j} \Gamma_{ij}x_i x_j}$$

### Stabilizers

$$\boxed{K_a = X_a \prod_{b \in N(a)} Z_b}$$

$$\boxed{H = (I_n | \Gamma)}$$

### Local Complementation

$$\boxed{G*a: \text{toggle edges within } N(a)}$$

$$\boxed{|G*a\rangle = U_a^\dagger |G\rangle, \quad U_a = e^{-i\frac{\pi}{4}X_a}\prod_{b \in N(a)}\sqrt{Z_b}}$$

### LC Invariants

$$\boxed{\text{rank}_{\mathbb{F}_2}(\Gamma_G) = \text{rank}_{\mathbb{F}_2}(\Gamma_H) \text{ if } G \sim_{LC} H}$$

### MBQC

$$\boxed{\text{Wire: } |\psi\rangle \to Z^s|\psi\rangle}$$

$$\boxed{R_Z(\theta): |\psi\rangle \to X^s R_Z(\theta)|\psi\rangle}$$

$$\boxed{\text{Adaptive angle: } \theta' = (-1)^s \theta}$$

---

## Comprehensive Problem Set

### Part A: Graph States (Days 736-737)

**A1.** For the "bowtie" graph (two triangles sharing a vertex):
```
1—2
 \|/
  3
 /|\
4—5
```
a) Write the adjacency matrix
b) Compute the amplitude ⟨11111|G⟩
c) Write all stabilizer generators

**A2.** Prove that the graph state amplitude satisfies:
$$|\langle x | G \rangle|^2 = \frac{1}{2^n}$$
for all computational basis states x.

**A3.** For the cycle graph C₅ (pentagon):
a) Write the parity check matrix H
b) Verify H·Ω·H^T = 0

### Part B: Local Complementation (Days 738-739)

**B1.** Starting with the square graph C₄:
a) Apply LC at vertex 0
b) Apply LC at vertex 1 to the result
c) Is the final graph in the same LC orbit as C₄?

**B2.** Compute the interlacement graph for K₄ (complete graph on 4 vertices).

**B3.** Find the minimum-edge representative of the LC orbit containing:
a) The 5-vertex path P₅
b) The 5-vertex cycle C₅

### Part C: MBQC (Days 740-741)

**C1.** Design an MBQC pattern to implement the gate sequence:
$$U = R_Z(\pi/4) \cdot R_X(\pi/2) \cdot R_Z(\pi/4)$$

Include:
a) The required graph state
b) Measurement angles
c) Byproduct operators

**C2.** For a 3×3 2D cluster state:
a) How many qubits can be used as input?
b) How many as output?
c) What is the maximum circuit depth implementable?

**C3.** Analyze error propagation:
If an X error occurs on the center qubit of a 3×3 cluster before any measurements, which output qubits are affected after the computation?

### Part D: Integration

**D1.** Prove that measuring all qubits of a graph state |G⟩ in the Z-basis produces a uniformly random classical bit string.

**D2.** Show that the GHZ state $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$ is LC-equivalent to a graph state. What graph?

**D3.** Design a complete MBQC protocol for the quantum circuit:
```
|0⟩ —H—●—H— |output⟩
       |
|0⟩ —H—⊕—H— |output⟩
```

---

## Solutions to Selected Problems

### Solution A1

a) **Adjacency matrix:**
$$\Gamma = \begin{pmatrix}
0 & 1 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 & 0 \\
1 & 1 & 0 & 1 & 1 \\
0 & 0 & 1 & 0 & 1 \\
0 & 0 & 1 & 1 & 0
\end{pmatrix}$$

b) **Amplitude ⟨11111|G⟩:**
$$q(11111) = \Gamma_{12} + \Gamma_{13} + \Gamma_{23} + \Gamma_{34} + \Gamma_{35} + \Gamma_{45}$$
$$= 1 + 1 + 1 + 1 + 1 + 1 = 6 \equiv 0 \pmod{2}$$
$$\langle 11111 | G \rangle = \frac{1}{\sqrt{32}}(-1)^0 = \frac{1}{4\sqrt{2}}$$

c) **Stabilizers:**
- $K_1 = X_1 Z_2 Z_3$
- $K_2 = Z_1 X_2 Z_3$
- $K_3 = Z_1 Z_2 X_3 Z_4 Z_5$
- $K_4 = Z_3 X_4 Z_5$
- $K_5 = Z_3 Z_4 X_5$

### Solution C1

**Gate:** $U = R_Z(\pi/4) R_X(\pi/2) R_Z(\pi/4)$

**Graph:** 4-qubit linear chain: 1—2—3—4

**Pattern:**
1. Input |ψ⟩ on qubit 1
2. Measure qubit 1 at angle θ₁ = π/4 → R_Z(π/4)
3. Measure qubit 2 at angle θ₂ = (-1)^{s₁}(π/2) → R_X(π/2)
4. Measure qubit 3 at angle θ₃ = (-1)^{s₂}(π/4) → R_Z(π/4)
5. Output on qubit 4

**Byproduct:** $X^{s_1 \oplus s_3} Z^{s_2}$

### Solution D2

**GHZ and graph states:**

The GHZ state $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$ is the graph state for the star graph!

**Star S₃:** Center connected to all others
$$|S_3\rangle = CZ_{01}CZ_{02}|+++\rangle$$

After local Hadamards on leaves:
$$H_1 H_2 |S_3\rangle = \frac{1}{\sqrt{2}}(|0++\rangle + |1--\rangle)$$

This equals GHZ up to local unitaries.

---

## Self-Assessment

### Mastery Checklist

| Skill | Day | Confidence (1-5) |
|-------|-----|------------------|
| Graph state construction | 736 | ___ |
| Amplitude calculation | 736 | ___ |
| Stabilizer derivation | 737 | ___ |
| Parity check from graph | 737 | ___ |
| Local complementation | 738 | ___ |
| LC orbit computation | 738-739 | ___ |
| LC invariants | 739 | ___ |
| MBQC wire | 740 | ___ |
| Rotation via MBQC | 740 | ___ |
| Byproduct tracking | 740-741 | ___ |
| 2D cluster universality | 741 | ___ |
| Error propagation | 741 | ___ |

---

## Connections to Future Topics

### Week 107: CSS Codes

**From Graph States:**
- Graph states on bipartite graphs have CSS-like structure
- Surface codes are graph state codes

**Key Connection:**
$$\text{Bipartite graph} \to X \text{ on one part}, Z \text{ on other} \to \text{CSS structure}$$

### Week 108: Code Families

**Graph State Codes:**
- Many code families are based on graph states
- Color codes, surface codes, etc.

### Beyond Month 27

**Topological Codes:**
- Surface codes as graph states on toric lattices
- Topological MBQC uses 3D structures

---

## Preparation for Week 107

### Coming Next: CSS Codes

**Topics:**
- CSS code construction from classical codes
- Dual containment: C₂⊥ ⊆ C₁
- Surface codes as CSS codes
- Product constructions

### Prerequisites Check

Ensure mastery of:
- [ ] Binary symplectic representation (Week 105)
- [ ] Graph states and their stabilizers (Week 106)
- [ ] Parity check matrix formalism
- [ ] Classical linear codes (basic understanding)

### Preview Questions

1. What is a CSS code?
2. How does the dual code C⊥ relate to quantum codes?
3. Why is the condition C₂⊥ ⊆ C₁ necessary?
4. What are the advantages of CSS codes?

---

## Week 106 Complete!

### Summary of Achievements

This week you learned:

1. **Graph States:** Construction from graphs via CZ gates
2. **Stabilizer Structure:** The elegant formula $K_a = X_a Z^{N(a)}$
3. **Local Complementation:** Graph operation corresponding to local Clifford
4. **LC Equivalence:** Classification of graph states
5. **MBQC Foundations:** Computation via measurement
6. **Cluster State Universality:** 2D clusters enable universal QC

### Key Insight

Graph states bridge combinatorics and quantum information:
- Graph structure → Entanglement pattern
- LC equivalence → Local Clifford equivalence
- 2D lattice → Universal computation

### Progress

- **Week 106:** 100% complete (7/7 days)
- **Month 27:** 50% complete (14/28 days)
- **Next:** Week 107 — CSS Codes

---

## Computational Synthesis

```python
"""
Week 106 Synthesis: Complete Graph State Toolkit
================================================
"""

import numpy as np
from typing import List, Tuple, Dict, Set

class GraphState:
    """Complete graph state analysis toolkit."""

    def __init__(self, n: int, edges: List[Tuple[int, int]]):
        self.n = n
        self.edges = edges
        self.Gamma = self._build_adjacency()

    def _build_adjacency(self) -> np.ndarray:
        Gamma = np.zeros((self.n, self.n), dtype=int)
        for i, j in self.edges:
            Gamma[i, j] = 1
            Gamma[j, i] = 1
        return Gamma

    def neighborhood(self, a: int) -> Set[int]:
        return {b for b in range(self.n) if self.Gamma[a, b] == 1}

    def stabilizer(self, a: int) -> str:
        pauli = ['I'] * self.n
        pauli[a] = 'X'
        for b in self.neighborhood(a):
            pauli[b] = 'Z'
        return ''.join(pauli)

    def all_stabilizers(self) -> List[str]:
        return [self.stabilizer(a) for a in range(self.n)]

    def parity_check(self) -> np.ndarray:
        return np.hstack([np.eye(self.n, dtype=int), self.Gamma])

    def amplitude(self, x: int) -> complex:
        bits = [(x >> (self.n-1-i)) & 1 for i in range(self.n)]
        q = sum(self.Gamma[i,j] * bits[i] * bits[j]
                for i in range(self.n) for j in range(i+1, self.n))
        return ((-1)**q) / np.sqrt(2**self.n)

    def local_complement(self, a: int) -> 'GraphState':
        new_Gamma = self.Gamma.copy()
        N_a = list(self.neighborhood(a))
        for i, b in enumerate(N_a):
            for c in N_a[i+1:]:
                new_Gamma[b, c] = 1 - new_Gamma[b, c]
                new_Gamma[c, b] = 1 - new_Gamma[c, b]
        new_edges = [(i, j) for i in range(self.n)
                     for j in range(i+1, self.n) if new_Gamma[i,j] == 1]
        return GraphState(self.n, new_edges)

    def lc_orbit(self) -> List['GraphState']:
        def graph_hash(G):
            return tuple(map(tuple, G.Gamma))

        orbit = {graph_hash(self): self}
        queue = [self]

        while queue:
            G = queue.pop(0)
            for a in range(self.n):
                G_new = G.local_complement(a)
                h = graph_hash(G_new)
                if h not in orbit:
                    orbit[h] = G_new
                    queue.append(G_new)

        return list(orbit.values())

    def rank_f2(self) -> int:
        M = self.Gamma.copy()
        rank = 0
        for col in range(self.n):
            pivot = None
            for row in range(rank, self.n):
                if M[row, col] == 1:
                    pivot = row
                    break
            if pivot is None:
                continue
            M[[rank, pivot]] = M[[pivot, rank]]
            for row in range(self.n):
                if row != rank and M[row, col] == 1:
                    M[row] = (M[row] + M[rank]) % 2
            rank += 1
        return rank

# Quick demonstration
if __name__ == "__main__":
    print("Week 106 Synthesis: Graph State Toolkit")
    print("=" * 50)

    # Example graph
    G = GraphState(4, [(0,1), (1,2), (2,3), (3,0)])  # Square

    print(f"\nSquare graph C₄:")
    print(f"Stabilizers: {G.all_stabilizers()}")
    print(f"F₂ rank: {G.rank_f2()}")
    print(f"LC orbit size: {len(G.lc_orbit())}")

    print(f"\nAmplitudes |⟨x|G⟩|²:")
    for x in range(16):
        amp = G.amplitude(x)
        print(f"  |{format(x,'04b')}⟩: {np.abs(amp)**2:.4f}")
```

---

**Week 106 Complete! Next: Week 107 — CSS Codes & Related Constructions**
