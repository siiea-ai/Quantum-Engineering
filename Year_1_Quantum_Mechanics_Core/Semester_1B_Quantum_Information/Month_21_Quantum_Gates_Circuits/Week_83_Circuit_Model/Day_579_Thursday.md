# Day 579: Circuit Complexity

## Overview
**Day 579** | Week 83, Day 5 | Year 1, Month 21 | Complexity Metrics for Quantum Circuits

Today we study how to measure the "cost" of quantum circuits through various complexity metrics: depth, width, gate count, and the critically important T-count for fault-tolerant quantum computing.

---

## Learning Objectives

1. Define and compute circuit depth
2. Calculate circuit width and its relation to qubit count
3. Count gates by type and understand gate cost hierarchies
4. Understand T-count as the dominant metric for fault tolerance
5. Analyze trade-offs between depth and width
6. Apply complexity analysis to real quantum algorithms

---

## Core Content

### Circuit Depth

**Definition:** The **depth** of a circuit is the length of the longest path from input to output, counting layers of gates.

```
Depth 3 circuit:
         ┌───┐   ┌───┐   ┌───┐
q₀: ─────┤ H ├───┤ T ├───┤ H ├────
         └───┘   └───┘   └───┘
         ┌───┐
q₁: ─────┤ X ├─────────────────────
         └───┘

Layer 1: H, X  (parallel)
Layer 2: T
Layer 3: H
```

**Why depth matters:**
- Determines circuit **execution time**
- Limits **decoherence** effects (shallower = less noise)
- Related to **parallelism** capability

### Circuit Width

**Definition:** The **width** of a circuit is the number of qubits.

$$\boxed{\text{Width} = n \text{ (number of qubits)}}$$

**Memory cost:** $2^n$ amplitudes to store classically.

### Gate Count

**Definition:** Total number of gates in the circuit.

```
Example:
         ┌───┐   ┌───┐
q₀: ─────┤ H ├───┤ T ├───●────
         └───┘   └───┘   │
         ┌───┐   ┌───┐   │
q₁: ─────┤ X ├───┤ S ├───⊕────
         └───┘   └───┘

Gate count: 5 (H, T, X, S, CNOT)
```

### Gate Count by Type

Not all gates have equal cost! Typical hierarchy:

| Gate | Relative Cost | Notes |
|------|---------------|-------|
| Pauli (X, Y, Z) | 1 | Trivial in most frameworks |
| H, S, S† | 1-2 | Clifford gates |
| CNOT, CZ | 2-5 | Two-qubit, hardware-dependent |
| T, T† | 10-100 | Expensive for fault tolerance |
| Arbitrary rotation | Variable | May need decomposition |

### T-Count: The Critical Metric

**Definition:** The **T-count** is the number of T gates (and T†) in a circuit.

$$\boxed{\text{T-count} = \#T + \#T^\dagger}$$

**Why T-count dominates fault tolerance:**
- Clifford gates can be implemented "transversally" (easily)
- T gates require **magic state distillation**
- Magic state distillation is **extremely expensive**
- T-count determines fault-tolerant resource cost

**Example:** Toffoli gate
- Naive: many gates
- T-count optimized: 7 T gates (minimal known)

### T-Depth

**Definition:** The **T-depth** is the number of layers containing at least one T gate.

```
T-depth 2 circuit:
         ┌───┐   ┌───┐   ┌───┐   ┌───┐
q₀: ─────┤ T ├───┤ H ├───┤ T ├───┤ H ├────
         └───┘   └───┘   └───┘   └───┘
         ┌───┐   ┌───┐
q₁: ─────┤ T ├───┤ X ├─────────────────────
         └───┘   └───┘

Layer 1: T, T  (T-layer 1)
Layer 2: H, X  (no T)
Layer 3: T     (T-layer 2)
Layer 4: H     (no T)

T-count = 3, T-depth = 2
```

### Depth-Width Trade-offs

**Principle:** Many algorithms allow trading depth for width (and vice versa).

**Example: Parallel CNOT ladder**
```
Serial (depth n-1, width n):        Parallel (depth log n, width 2n):
q₀: ──●──────────────────           q₀: ──●────────────────
      │                                   │
q₁: ──⊕──●───────────────           q₁: ──⊕────●───────────
         │                                     │
q₂: ─────⊕──●────────────    vs     q₂: ───────⊕────●──────
            │                                       │
q₃: ────────⊕────────────           q₃: ────────────⊕──────
                                    a₀: ──●────────────────
Depth: 3                                  │
                                    a₁: ──⊕────●───────────
                                               │    ...
                                    Depth: 2 (with ancillas)
```

### Space-Time Volume

**Definition:** $\text{Volume} = \text{Depth} \times \text{Width}$

This captures total "quantum resources" used.

### CNOT Count

For many hardware platforms, CNOT is the most expensive "native" gate:

**CNOT count** = number of two-qubit entangling gates

**Example costs (IBM superconducting):**
- Single-qubit gate: ~20-50 ns, error ~0.1%
- CNOT: ~200-500 ns, error ~1%

### Circuit Size

**Definition:** $\text{Size} = $ total gate count, sometimes weighted.

$$\text{Size} = \sum_i w_i \cdot n_i$$

where $n_i$ = count of gate type $i$, $w_i$ = weight/cost.

### Asymptotic Complexity

For algorithm analysis, express complexity in terms of input size $n$:

| Algorithm | Gate Count | Depth | T-count |
|-----------|------------|-------|---------|
| QFT | $O(n^2)$ | $O(n)$ | $O(n \log n)$ |
| Grover | $O(\sqrt{N})$ | $O(\sqrt{N})$ | $O(\sqrt{N})$ |
| Shor (factor n-bit) | $O(n^3)$ | $O(n^3)$ | $O(n^3)$ |

### Lower Bounds

**No-cloning complexity:** To transform $|0\rangle^{\otimes n}$ to a general $n$-qubit state requires $\Omega(2^n/n)$ gates in the worst case.

**CNOT lower bound:** Creating certain entanglement patterns requires $\Omega(n)$ CNOTs.

---

## Worked Examples

### Example 1: QFT Complexity Analysis

Analyze the complexity of the $n$-qubit Quantum Fourier Transform.

**Circuit structure:**
```
         ┌───┐   ┌────┐   ┌────┐
q₀: ─────┤ H ├───┤R(2)├───┤R(3)├──── ... ───
         └───┘   └──┬─┘   └──┬─┘
                    │        │
q₁: ────────────────●────────│───[H]─[R(2)]── ...
                             │
q₂: ─────────────────────────●──────────[H]── ...
```

**Solution:**

**Gate count:**
- H gates: $n$ (one per qubit)
- Controlled rotations: $\frac{n(n-1)}{2}$
- Total: $n + \frac{n(n-1)}{2} = \frac{n(n+1)}{2} = O(n^2)$

**Depth:**
- Each qubit's gates can partially parallelize
- Depth: $O(n)$ with parallelization

**T-count:**
- Each $R_k$ rotation requires $O(1)$ T gates (after decomposition)
- Total T-count: $O(n^2)$

### Example 2: Comparing Toffoli Implementations

Compare T-count for different Toffoli decompositions.

**Solution:**

**Nielsen-Chuang decomposition:**
```
a: ──────●────────●──────●──────●──────●──────●──────●──────
         │        │      │      │      │      │      │
b: ──●───│────●───│──●───│──●───│──●───│──●───│──●───│──●───
     │   │    │   │  │   │  │   │  │   │  │   │  │   │  │
c: ──⊕─[T†]──⊕─[T]─⊕─[T†]─⊕─[T]─⊕─[T†]─⊕─[T]─⊕─[T†]─⊕─[T]─
```
T-count: 7 (optimal for Toffoli without ancillas)

**With one ancilla (T-depth optimized):**
T-count: 7 (same)
T-depth: 4 (reduced from 7)

### Example 3: Depth Analysis of CNOT Cascade

Calculate the depth of a linear CNOT cascade.

```
q₀: ──●────────────────────
      │
q₁: ──⊕──●─────────────────
         │
q₂: ─────⊕──●──────────────
            │
q₃: ────────⊕──●───────────
               │
q₄: ───────────⊕───────────
```

**Solution:**

Each CNOT depends on the previous one (target becomes next control).

**Depth:** $n - 1$ (for $n$ qubits)
**CNOT count:** $n - 1$
**This is serial** - no parallelization possible with this specific pattern.

---

## Practice Problems

### Problem 1: Gate Count
Count all gates in this circuit:
```
         ┌───┐   ┌───┐
q₀: ─────┤ H ├───┤ T ├───●───●───
         └───┘   └───┘   │   │
         ┌───┐   ┌───┐   │   │
q₁: ─────┤ X ├───┤ S ├───⊕───│───
         └───┘   └───┘       │
         ┌───┐               │
q₂: ─────┤ H ├───────────────⊕───
         └───┘
```

### Problem 2: T-Count
What is the T-count of a circuit with 5 T gates, 3 T† gates, and 10 Clifford gates?

### Problem 3: Depth Optimization
Can this circuit be parallelized to reduce depth?
```
         ┌───┐   ┌───┐   ┌───┐
q₀: ─────┤ H ├───┤ X ├───┤ H ├───
         └───┘   └───┘   └───┘
         ┌───┐   ┌───┐   ┌───┐
q₁: ─────┤ H ├───┤ Y ├───┤ H ├───
         └───┘   └───┘   └───┘
```

### Problem 4: CNOT Count
How many CNOTs are needed to prepare $|GHZ_n\rangle = \frac{1}{\sqrt{2}}(|0\rangle^{\otimes n} + |1\rangle^{\otimes n})$?

---

## Computational Lab

```python
"""Day 579: Circuit Complexity Analysis"""
import numpy as np
from collections import Counter

class QuantumCircuit:
    """Simple circuit representation for complexity analysis"""

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []  # List of (gate_name, qubits, layer)
        self.current_layer = [0] * n_qubits  # Track layer for each qubit

    def add_gate(self, name, qubits):
        """Add a gate acting on specified qubits"""
        if isinstance(qubits, int):
            qubits = [qubits]

        # Determine layer (max of involved qubits + 1)
        layer = max(self.current_layer[q] for q in qubits)

        self.gates.append((name, tuple(qubits), layer))

        # Update layer tracking
        for q in qubits:
            self.current_layer[q] = layer + 1

    def depth(self):
        """Calculate circuit depth"""
        if not self.gates:
            return 0
        return max(g[2] for g in self.gates) + 1

    def width(self):
        """Return number of qubits"""
        return self.n_qubits

    def gate_count(self):
        """Total number of gates"""
        return len(self.gates)

    def gate_counts_by_type(self):
        """Count gates by type"""
        return Counter(g[0] for g in self.gates)

    def t_count(self):
        """Count T and T† gates"""
        counts = self.gate_counts_by_type()
        return counts.get('T', 0) + counts.get('Tdg', 0)

    def t_depth(self):
        """Count layers containing T gates"""
        t_layers = set()
        for name, qubits, layer in self.gates:
            if name in ['T', 'Tdg']:
                t_layers.add(layer)
        return len(t_layers)

    def cnot_count(self):
        """Count CNOT gates"""
        counts = self.gate_counts_by_type()
        return counts.get('CNOT', 0) + counts.get('CX', 0)

    def volume(self):
        """Space-time volume"""
        return self.depth() * self.width()

    def weighted_size(self, weights=None):
        """Weighted gate count"""
        if weights is None:
            weights = {
                'I': 0, 'X': 1, 'Y': 1, 'Z': 1,
                'H': 1, 'S': 1, 'Sdg': 1,
                'T': 10, 'Tdg': 10,
                'CNOT': 5, 'CX': 5, 'CZ': 5
            }
        counts = self.gate_counts_by_type()
        return sum(weights.get(g, 1) * c for g, c in counts.items())

    def summary(self):
        """Print complexity summary"""
        print(f"Circuit Complexity Analysis")
        print(f"=" * 40)
        print(f"Width (qubits):    {self.width()}")
        print(f"Depth:             {self.depth()}")
        print(f"Gate count:        {self.gate_count()}")
        print(f"T-count:           {self.t_count()}")
        print(f"T-depth:           {self.t_depth()}")
        print(f"CNOT count:        {self.cnot_count()}")
        print(f"Volume:            {self.volume()}")
        print(f"Weighted size:     {self.weighted_size()}")
        print(f"\nGates by type: {dict(self.gate_counts_by_type())}")

# ===== Example 1: Bell State Circuit =====
print("Example 1: Bell State Preparation")
print("-" * 40)
bell = QuantumCircuit(2)
bell.add_gate('H', 0)
bell.add_gate('CNOT', [0, 1])
bell.summary()

# ===== Example 2: GHZ State Circuit =====
print("\n\nExample 2: GHZ State (n=5)")
print("-" * 40)
n = 5
ghz = QuantumCircuit(n)
ghz.add_gate('H', 0)
for i in range(n-1):
    ghz.add_gate('CNOT', [0, i+1])
ghz.summary()

# ===== Example 3: QFT Circuit =====
print("\n\nExample 3: Quantum Fourier Transform (n=4)")
print("-" * 40)
n = 4
qft = QuantumCircuit(n)

for i in range(n):
    qft.add_gate('H', i)
    for j in range(i+1, n):
        qft.add_gate(f'CR{j-i+1}', [j, i])  # Controlled rotation

# Add SWAPs at end (for proper QFT)
for i in range(n//2):
    qft.add_gate('SWAP', [i, n-1-i])

qft.summary()

# ===== Example 4: Toffoli Decomposition =====
print("\n\nExample 4: Toffoli Gate Decomposition")
print("-" * 40)
toffoli = QuantumCircuit(3)

# Standard Toffoli decomposition with 7 T gates
toffoli.add_gate('H', 2)
toffoli.add_gate('CNOT', [1, 2])
toffoli.add_gate('Tdg', 2)
toffoli.add_gate('CNOT', [0, 2])
toffoli.add_gate('T', 2)
toffoli.add_gate('CNOT', [1, 2])
toffoli.add_gate('Tdg', 2)
toffoli.add_gate('CNOT', [0, 2])
toffoli.add_gate('T', 1)
toffoli.add_gate('T', 2)
toffoli.add_gate('H', 2)
toffoli.add_gate('CNOT', [0, 1])
toffoli.add_gate('T', 0)
toffoli.add_gate('Tdg', 1)
toffoli.add_gate('CNOT', [0, 1])

toffoli.summary()

# ===== Example 5: Complexity Scaling =====
print("\n\nExample 5: GHZ State Complexity Scaling")
print("-" * 40)
print(f"{'n':>4} {'Depth':>8} {'Gates':>8} {'CNOTs':>8} {'Volume':>10}")
print("-" * 40)

for n in [2, 4, 8, 16, 32, 64]:
    ghz = QuantumCircuit(n)
    ghz.add_gate('H', 0)
    for i in range(n-1):
        ghz.add_gate('CNOT', [0, i+1])

    print(f"{n:>4} {ghz.depth():>8} {ghz.gate_count():>8} {ghz.cnot_count():>8} {ghz.volume():>10}")

# ===== Example 6: Depth vs Width Trade-off =====
print("\n\nExample 6: Depth-Width Trade-off")
print("-" * 40)

# Serial CNOT cascade
print("\nSerial CNOT cascade (n=8):")
n = 8
serial = QuantumCircuit(n)
for i in range(n-1):
    serial.add_gate('CNOT', [i, i+1])
print(f"  Depth: {serial.depth()}, Width: {serial.width()}, CNOTs: {serial.cnot_count()}")

# Parallel version (conceptual - would need ancillas)
print("\nParallel version (conceptual):")
print(f"  With O(n) ancillas, depth can be O(log n)")
print(f"  For n=8: Depth ~3 vs 7 for serial")

# ===== Example 7: T-Count Comparison =====
print("\n\nExample 7: T-Count for Different Operations")
print("-" * 40)

operations = [
    ("Single T gate", [('T', 0)]),
    ("S gate (no T)", [('S', 0)]),
    ("Rz(pi/8) ~ 1 T", [('T', 0)]),
    ("Toffoli (optimal)", [('T', i % 3) for i in range(7)]),  # 7 T gates
    ("Controlled-S", [('T', 0), ('Tdg', 1), ('CNOT', [0, 1]), ('T', 1)]),  # 3 T
]

for name, gates in operations:
    circ = QuantumCircuit(3)
    for g in gates:
        if len(g) == 2:
            circ.add_gate(g[0], g[1])
        else:
            circ.add_gate(g[0], [g[1], g[2]])
    print(f"  {name}: T-count = {circ.t_count()}")

# ===== Summary Table =====
print("\n\n" + "=" * 60)
print("Complexity Metrics Summary")
print("=" * 60)
print("""
Metric          | Definition                    | Importance
----------------|-------------------------------|---------------------------
Depth           | Longest gate path             | Execution time, decoherence
Width           | Number of qubits              | Memory requirement
Gate Count      | Total gates                   | Overall resources
T-Count         | Number of T/T† gates          | Fault-tolerant cost
T-Depth         | Layers with T gates           | FT parallel time
CNOT Count      | Two-qubit gates               | Hardware error budget
Volume          | Depth × Width                 | Space-time resources
""")
```

---

## Summary

### Complexity Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| Depth | Longest path | Execution time, decoherence |
| Width | Number of qubits | Memory, hardware size |
| Gate count | Total gates | Overall resources |
| T-count | #T + #T† | Fault-tolerant cost |
| CNOT count | Two-qubit gates | Hardware error rate |
| Volume | Depth × Width | Total quantum resources |

### Key Formulas

| Concept | Formula |
|---------|---------|
| Volume | $V = D \times W$ |
| T-count | $\#T + \#T^\dagger$ |
| Weighted size | $\sum_i w_i n_i$ |
| QFT gates | $O(n^2)$ |

### Key Takeaways

1. **Depth** determines decoherence and execution time
2. **T-count** dominates fault-tolerant resource estimates
3. **Depth-width trade-offs** enable optimization for different constraints
4. **CNOT count** matters most for NISQ hardware
5. **Asymptotic analysis** reveals algorithm scalability
6. **Different metrics** matter for different contexts (NISQ vs FT)

---

## Daily Checklist

- [ ] I can calculate circuit depth by counting layers
- [ ] I understand why T-count is critical for fault tolerance
- [ ] I can perform gate counting by type
- [ ] I understand depth-width trade-offs
- [ ] I can analyze asymptotic complexity of quantum algorithms
- [ ] I ran the computational lab and analyzed circuit complexity

---

*Next: Day 580 — Circuit Optimization*
