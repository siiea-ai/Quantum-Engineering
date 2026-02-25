# Day 587: Compiling to Hardware

## Overview
**Day 587** | Week 84, Day 6 | Year 1, Month 21 | Transpilation and Optimization

Today we study how abstract quantum circuits are compiled to hardware-executable form—transpilation, gate decomposition, qubit routing, and optimization passes.

---

## Learning Objectives

1. Understand the transpilation pipeline
2. Implement gate decomposition to native gates
3. Solve the qubit routing problem with SWAPs
4. Apply optimization passes to reduce circuit cost
5. Use compilation tools (Qiskit, Cirq)
6. Evaluate trade-offs in compilation strategies

---

## Core Content

### The Compilation Pipeline

```
Abstract Circuit
      ↓
[Unrolling] - Decompose to basic gates
      ↓
[Basis Translation] - Convert to native gate set
      ↓
[Layout] - Map virtual to physical qubits
      ↓
[Routing] - Insert SWAPs for connectivity
      ↓
[Optimization] - Simplify circuit
      ↓
[Scheduling] - Order gates for hardware
      ↓
Hardware-Executable Circuit
```

### Stage 1: Unrolling

**Goal:** Decompose high-level gates into basic gates.

**Example:**
```
Toffoli → 15 basic gates (CNOT + 1Q)
CCZ → CNOTs + T gates
Custom gates → Standard decompositions
```

### Stage 2: Basis Translation

**Goal:** Convert gates to hardware native set.

**IBM example:** Convert to {$R_z$, $\sqrt{X}$, $CNOT$}

**Hadamard:**
$$H = R_z(\pi) \cdot \sqrt{X} \cdot R_z(\pi/2)$$

**S gate:**
$$S = R_z(\pi/2)$$

**T gate:**
$$T = R_z(\pi/4)$$

### Stage 3: Qubit Layout

**Problem:** Map logical qubits to physical qubits.

**Considerations:**
- Connectivity graph of hardware
- Two-qubit gate error rates (vary by qubit pair)
- Coherence times (vary by qubit)

**Strategies:**
- **Trivial:** Logical qubit $i$ → Physical qubit $i$
- **Heuristic:** Minimize expected SWAP count
- **Noise-aware:** Prefer high-fidelity qubits

### Stage 4: Routing

**Problem:** Given layout and connectivity, ensure all two-qubit gates act on adjacent qubits.

**Solution:** Insert SWAP gates to move qubits.

**Example:**
```
Hardware: 0 - 1 - 2 - 3 (linear chain)
Circuit requires: CNOT(0, 3)

Solution: SWAP(1,2), SWAP(0,1), CNOT(1,3), SWAP(0,1), SWAP(1,2)
```

**Algorithms:**
- **SABRE:** Heuristic search with lookahead
- **BIP:** Binary integer programming (optimal but slow)
- **Noise-aware:** Consider gate errors in routing

### Stage 5: Optimization

**Goal:** Reduce gate count, depth, or error.

**Techniques:**

**1. Gate Cancellation:**
```
──[H]──[H]── → ────
──[CNOT]──[CNOT]── → ────
```

**2. Gate Merging:**
```
──[Rz(θ₁)]──[Rz(θ₂)]── → ──[Rz(θ₁+θ₂)]──
```

**3. Commutation:**
Move gates past each other to enable cancellation.

**4. Template Matching:**
Replace known patterns with shorter equivalents.

**5. Peephole Optimization:**
Sliding window of 2-3 gates, optimize locally.

### Stage 6: Scheduling

**Goal:** Determine execution order respecting dependencies.

**Constraints:**
- Gate dependencies (data flow)
- Hardware parallelism limits
- Measurement timing

**Strategies:**
- **ASAP:** As Soon As Possible
- **ALAP:** As Late As Possible
- **Balanced:** Minimize depth while respecting idle times

### Optimization Levels

**Qiskit optimization levels:**

| Level | Description |
|-------|-------------|
| 0 | No optimization |
| 1 | Light optimization (cancellation) |
| 2 | Medium (+ commutation, templates) |
| 3 | Heavy (+ resynthesis, noise-aware) |

### SWAP Insertion Algorithms

**SABRE (SWAP-based Bidirectional heuristic):**
1. Process circuit front-to-back
2. For each gate requiring SWAP, choose SWAP minimizing future cost
3. Use lookahead to consider upcoming gates
4. Repeat back-to-front for improvement

**Complexity:** $O(n \cdot d)$ where $n$ = gates, $d$ = depth

### Noise-Aware Compilation

**Idea:** Use calibration data to minimize error.

**Inputs:**
- Gate error rates (per qubit, per pair)
- T1, T2 coherence times
- Readout errors

**Optimization:**
- Route through low-error qubits
- Avoid long idle times on decoherent qubits
- Schedule measurements accounting for crosstalk

### Compilation Metrics

| Metric | Description |
|--------|-------------|
| Gate count | Total gates after compilation |
| Two-qubit count | CNOTs/CZs in final circuit |
| Depth | Critical path length |
| SWAP count | Routing overhead |
| Estimated error | Product of gate fidelities |

### Advanced Techniques

**Resynthesis:** Re-decompose blocks of gates for better result.

**Unitary synthesis:** Given unitary matrix, find optimal circuit.

**Approximate synthesis:** Allow small error for fewer gates.

**ZX-calculus:** Graphical language for circuit optimization.

---

## Worked Examples

### Example 1: Complete Transpilation

Transpile this circuit for IBM linear topology 0-1-2:

```
q0: ──[H]──●──────
           │
q1: ───────│──[X]─
           │
q2: ───────⊕──────
```

**Solution:**

**Step 1: Layout**
Map q0→0, q1→1, q2→2 (trivial)

**Step 2: Routing**
CNOT(0,2) requires SWAPs on linear chain.

Insert SWAP(1,2):
```
0: ──[H]──●────────────────
          │
1: ───────×──×──────[X]────
          │  │
2: ───────×──⊕──×──────────
```

Wait, this isn't quite right. Let me redo:

After SWAP(1,2), qubit originally at position 2 is now at position 1.
So CNOT(0, "original q2") is now CNOT(0, 1).

```
Original: CNOT(0,2), X(1)
After SWAP(1,2): qubits at positions are:
  - Position 0: q0
  - Position 1: q2
  - Position 2: q1

So: CNOT(0,1) [now between original q0 and q2]
Then: X(2) [now on original q1]
Then: SWAP(1,2) to restore
```

**Final transpiled circuit:**
```
0: ──[H]───────●─────────────
               │
1: ──────×─────⊕───────×─────
         │             │
2: ──────×─────────[X]─×─────
```

**Step 3: Basis translation**
Convert H to native gates:
```
0: ──[Rz(π)]──[√X]──[Rz(π/2)]───●────...
```

### Example 2: Optimization Pass

Optimize this circuit:
```
──[H]──[T]──[T]──[T]──[T]──[H]──[H]──
```

**Solution:**

**Pass 1: Merge rotations**
$T \cdot T \cdot T \cdot T = Z$
```
──[H]──[Z]──[H]──[H]──
```

**Pass 2: Cancel H-H**
```
──[H]──[Z]──
```

**Pass 3: Use identity $HZH = X$**
Actually, we have $H$ then $Z$ then nothing, so:
```
──[H]──[Z]──
```

Can't simplify further without context.

**Original: 7 gates → Final: 2 gates**

### Example 3: SABRE Routing

Route this circuit on a 4-qubit ring (0-1-2-3-0):
```
CNOT(0,2), CNOT(1,3)
```

**Solution:**

Neither gate is between adjacent qubits on the ring.

**Option 1:** SWAP(0,1), then CNOT(1,2), CNOT(0,3)
- 1 SWAP + 2 CNOTs = 5 CNOTs total (SWAP = 3 CNOTs)

**Option 2:** SWAP(1,2), then CNOT(0,1), CNOT(2,3)
- 1 SWAP + 2 CNOTs = 5 CNOTs total

Both options use 1 SWAP. SABRE would consider future gates (if any) to break ties.

---

## Practice Problems

### Problem 1: Layout Selection
For a circuit using qubits 0,1,2 with gates CNOT(0,1), CNOT(1,2), CNOT(0,2), which layout minimizes SWAPs on a linear chain?

### Problem 2: SWAP Count
How many SWAPs are needed to route CNOT(0,4) on a 5-qubit linear chain?

### Problem 3: Optimization
Simplify: $R_z(\pi/4) \cdot R_z(\pi/4) \cdot R_z(\pi/2) \cdot R_z(-\pi)$

### Problem 4: Depth vs Gates
A circuit can be compiled to:
- Option A: 10 gates, depth 8
- Option B: 15 gates, depth 5
Which is better for a noisy device with T2 = 100μs and gate time = 1μs?

---

## Computational Lab

```python
"""Day 587: Compiling to Hardware"""
import numpy as np
from collections import defaultdict

# ===== Gate Definitions =====
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def Rz(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)

def sqrt_X():
    return (1/2) * np.array([
        [1 + 1j, 1 - 1j],
        [1 - 1j, 1 + 1j]
    ], dtype=complex)

# ===== Simple Circuit Representation =====
class Circuit:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []  # List of (gate_name, qubits, params)

    def add(self, name, qubits, params=None):
        if isinstance(qubits, int):
            qubits = (qubits,)
        self.gates.append((name, tuple(qubits), params))
        return self

    def copy(self):
        c = Circuit(self.n_qubits)
        c.gates = self.gates.copy()
        return c

    def __repr__(self):
        lines = [f"Circuit({self.n_qubits} qubits):"]
        for name, qubits, params in self.gates:
            p_str = f"({params})" if params else ""
            lines.append(f"  {name}{p_str} on {qubits}")
        return '\n'.join(lines)

    def gate_count(self):
        return len(self.gates)

    def two_qubit_count(self):
        return sum(1 for _, q, _ in self.gates if len(q) == 2)

# ===== Transpilation Functions =====

def decompose_to_basis(circuit, basis={'Rz', 'SX', 'CNOT'}):
    """Decompose gates to basis set"""
    result = Circuit(circuit.n_qubits)

    decompositions = {
        'H': [('Rz', np.pi), ('SX', None), ('Rz', np.pi/2)],
        'X': [('SX', None), ('SX', None)],
        'Y': [('Rz', np.pi), ('SX', None), ('SX', None)],
        'Z': [('Rz', np.pi)],
        'S': [('Rz', np.pi/2)],
        'T': [('Rz', np.pi/4)],
        'Sdg': [('Rz', -np.pi/2)],
        'Tdg': [('Rz', -np.pi/4)],
    }

    for name, qubits, params in circuit.gates:
        if name in basis:
            result.add(name, qubits, params)
        elif name in decompositions:
            for dec_name, dec_param in decompositions[name]:
                result.add(dec_name, qubits, dec_param)
        elif name == 'CNOT' and 'CNOT' in basis:
            result.add('CNOT', qubits)
        elif name == 'CZ':
            # CZ = (I ⊗ H) CNOT (I ⊗ H)
            result.add('H', qubits[1])
            result.add('CNOT', qubits)
            result.add('H', qubits[1])
        else:
            result.add(name, qubits, params)  # Pass through unknown

    return result

def simple_routing(circuit, connectivity):
    """Simple routing with SWAP insertion (greedy)"""
    # connectivity: dict mapping qubit -> set of neighbors
    result = Circuit(circuit.n_qubits)

    # Track current mapping: logical -> physical
    mapping = {i: i for i in range(circuit.n_qubits)}
    inverse = {i: i for i in range(circuit.n_qubits)}

    def add_swap(p1, p2):
        """Add SWAP between physical qubits p1, p2"""
        result.add('SWAP', (p1, p2))
        # Update inverse mapping
        l1, l2 = inverse[p1], inverse[p2]
        mapping[l1], mapping[l2] = p2, p1
        inverse[p1], inverse[p2] = l2, l1

    def distance(p1, p2):
        """BFS distance between physical qubits"""
        if p1 == p2:
            return 0
        visited = {p1}
        queue = [(p1, 0)]
        while queue:
            curr, d = queue.pop(0)
            for neighbor in connectivity.get(curr, set()):
                if neighbor == p2:
                    return d + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
        return float('inf')

    for name, qubits, params in circuit.gates:
        if len(qubits) == 1:
            # Single-qubit gate: just map
            physical = mapping[qubits[0]]
            result.add(name, physical, params)
        else:
            # Two-qubit gate: may need SWAPs
            l1, l2 = qubits
            p1, p2 = mapping[l1], mapping[l2]

            # Route until adjacent
            while p2 not in connectivity.get(p1, set()):
                # Find neighbor of p1 closer to p2
                best_neighbor = None
                best_dist = distance(p1, p2)
                for neighbor in connectivity.get(p1, set()):
                    d = distance(neighbor, p2)
                    if d < best_dist:
                        best_dist = d
                        best_neighbor = neighbor
                if best_neighbor is None:
                    break  # Can't route
                add_swap(p1, best_neighbor)
                p1 = best_neighbor

            # Now apply the gate
            result.add(name, (p1, p2), params)

    return result

def optimize_circuit(circuit):
    """Simple optimization pass"""
    result = Circuit(circuit.n_qubits)

    # Merge consecutive Rz gates
    pending_rz = {}  # qubit -> accumulated angle

    for name, qubits, params in circuit.gates:
        if name == 'Rz' and len(qubits) == 1:
            q = qubits[0]
            if q in pending_rz:
                pending_rz[q] += params
            else:
                pending_rz[q] = params
        else:
            # Flush pending Rz for involved qubits
            for q in qubits:
                if q in pending_rz:
                    angle = pending_rz.pop(q) % (2 * np.pi)
                    if abs(angle) > 1e-10 and abs(angle - 2*np.pi) > 1e-10:
                        result.add('Rz', q, angle)
            result.add(name, qubits, params)

    # Flush remaining Rz
    for q, angle in pending_rz.items():
        angle = angle % (2 * np.pi)
        if abs(angle) > 1e-10 and abs(angle - 2*np.pi) > 1e-10:
            result.add('Rz', q, angle)

    return result

# ===== Example 1: Full Transpilation =====
print("=" * 60)
print("Example 1: Complete Transpilation Pipeline")
print("=" * 60)

# Original circuit
original = Circuit(3)
original.add('H', 0)
original.add('CNOT', (0, 2))
original.add('X', 1)

print("\nOriginal circuit:")
print(original)
print(f"Gates: {original.gate_count()}, 2Q: {original.two_qubit_count()}")

# Linear chain connectivity
linear_3 = {0: {1}, 1: {0, 2}, 2: {1}}

# Step 1: Route
routed = simple_routing(original, linear_3)
print("\nAfter routing (linear chain 0-1-2):")
print(routed)
print(f"Gates: {routed.gate_count()}, 2Q: {routed.two_qubit_count()}")

# Step 2: Decompose to basis
decomposed = decompose_to_basis(routed)
print("\nAfter basis decomposition:")
print(decomposed)
print(f"Gates: {decomposed.gate_count()}, 2Q: {decomposed.two_qubit_count()}")

# Step 3: Optimize
optimized = optimize_circuit(decomposed)
print("\nAfter optimization:")
print(optimized)
print(f"Gates: {optimized.gate_count()}, 2Q: {optimized.two_qubit_count()}")

# ===== Example 2: Routing on Grid =====
print("\n" + "=" * 60)
print("Example 2: Routing on 2x2 Grid")
print("=" * 60)

# 2x2 grid: 0-1
#           | |
#           2-3
grid_2x2 = {
    0: {1, 2},
    1: {0, 3},
    2: {0, 3},
    3: {1, 2}
}

circuit2 = Circuit(4)
circuit2.add('CNOT', (0, 3))  # Diagonal - not adjacent
circuit2.add('CNOT', (1, 2))  # Also diagonal

print("\nOriginal circuit (needs routing):")
print(circuit2)

routed2 = simple_routing(circuit2, grid_2x2)
print("\nAfter routing on 2x2 grid:")
print(routed2)
print(f"SWAPs inserted: {sum(1 for n, _, _ in routed2.gates if n == 'SWAP')}")

# ===== Example 3: Optimization Comparison =====
print("\n" + "=" * 60)
print("Example 3: Optimization Levels")
print("=" * 60)

circuit3 = Circuit(1)
for _ in range(4):
    circuit3.add('T', 0)
circuit3.add('H', 0)
circuit3.add('H', 0)
circuit3.add('S', 0)
circuit3.add('Sdg', 0)

print("\nOriginal circuit:")
print(circuit3)
print(f"Gate count: {circuit3.gate_count()}")

# Decompose first
decomposed3 = decompose_to_basis(circuit3)
print("\nAfter decomposition:")
print(decomposed3)
print(f"Gate count: {decomposed3.gate_count()}")

# Optimize
optimized3 = optimize_circuit(decomposed3)
print("\nAfter optimization:")
print(optimized3)
print(f"Gate count: {optimized3.gate_count()}")

# ===== Example 4: Compilation Metrics =====
print("\n" + "=" * 60)
print("Example 4: Compilation Metrics")
print("=" * 60)

def compilation_metrics(circuit, gate_errors=None):
    """Compute compilation metrics"""
    if gate_errors is None:
        gate_errors = {'CNOT': 0.01, 'SWAP': 0.03, 'default': 0.001}

    metrics = {
        'gate_count': circuit.gate_count(),
        '2q_count': circuit.two_qubit_count(),
        'depth': estimate_depth(circuit),
    }

    # Estimate total error
    error = 0
    for name, qubits, _ in circuit.gates:
        err = gate_errors.get(name, gate_errors['default'])
        error += err

    metrics['estimated_error'] = 1 - np.exp(-error)
    return metrics

def estimate_depth(circuit):
    """Estimate circuit depth"""
    qubit_time = [0] * circuit.n_qubits
    for _, qubits, _ in circuit.gates:
        max_t = max(qubit_time[q] for q in qubits)
        for q in qubits:
            qubit_time[q] = max_t + 1
    return max(qubit_time)

print("\nMetrics for example circuits:")
print(f"{'Circuit':<20} {'Gates':<8} {'2Q':<6} {'Depth':<6} {'Error':<10}")
print("-" * 55)

for name, circ in [("Original", original), ("Routed", routed),
                   ("Decomposed", decomposed), ("Optimized", optimized)]:
    m = compilation_metrics(circ)
    print(f"{name:<20} {m['gate_count']:<8} {m['2q_count']:<6} "
          f"{m['depth']:<6} {m['estimated_error']:.4f}")

# ===== Summary =====
print("\n" + "=" * 60)
print("Compilation Pipeline Summary")
print("=" * 60)
print("""
TRANSPILATION STAGES:
1. Unrolling     - Decompose complex gates to basic gates
2. Basis Trans.  - Convert to hardware native gates
3. Layout        - Map logical qubits to physical qubits
4. Routing       - Insert SWAPs for connectivity constraints
5. Optimization  - Reduce gate count, depth, error
6. Scheduling    - Order gates for execution

KEY OPTIMIZATIONS:
- Gate cancellation: HH → I, CNOT·CNOT → I
- Rotation merging: Rz(θ₁)·Rz(θ₂) → Rz(θ₁+θ₂)
- Commutation: Reorder gates to enable cancellation
- Template matching: Replace patterns with shorter equivalents

ROUTING ALGORITHMS:
- SABRE: Bidirectional heuristic with lookahead
- Greedy: Minimize immediate SWAP cost
- Noise-aware: Consider gate error rates

TRADE-OFFS:
- Gates vs Depth: More parallelism = more gates
- Optimization time vs Quality: Harder optimization = better results
- Generic vs Specialized: Custom decompositions can be better
""")
```

---

## Summary

### Compilation Pipeline

| Stage | Goal | Output |
|-------|------|--------|
| Unrolling | Basic gates | Standard gate set |
| Basis Translation | Native gates | Hardware gates |
| Layout | Map qubits | Physical assignment |
| Routing | Adjacency | SWAP-inserted circuit |
| Optimization | Reduce cost | Simplified circuit |
| Scheduling | Timing | Executable sequence |

### Key Formulas

| Metric | Importance |
|--------|------------|
| Gate count | Total operations |
| 2Q count | Error-prone operations |
| Depth | Execution time |
| SWAP count | Routing overhead |

### Key Takeaways

1. **Transpilation** converts abstract circuits to hardware-executable form
2. **Routing** is often the biggest overhead (SWAPs are expensive)
3. **Optimization** can significantly reduce circuit cost
4. **Trade-offs** exist between depth, gate count, and compilation time
5. **Noise-aware** compilation uses calibration data
6. **Different backends** require different compilation strategies

---

## Daily Checklist

- [ ] I understand the transpilation pipeline stages
- [ ] I can decompose gates to native gate sets
- [ ] I understand the qubit routing problem
- [ ] I can apply basic optimization passes
- [ ] I understand compilation trade-offs
- [ ] I ran the computational lab and traced through transpilation

---

*Next: Day 588 — Month 21 Review*
