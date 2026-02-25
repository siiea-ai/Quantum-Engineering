# Day 580: Circuit Optimization

## Overview
**Day 580** | Week 83, Day 6 | Year 1, Month 21 | Reducing Circuit Cost

Today we explore techniques for optimizing quantum circuits—reducing depth, gate count, and T-count through algebraic identities, gate cancellation, commutation rules, and template matching.

---

## Learning Objectives

1. Apply gate cancellation rules to simplify circuits
2. Use commutation relations to reorder gates
3. Implement template matching for pattern-based optimization
4. Optimize T-count using T-gate synthesis techniques
5. Understand the role of ancilla qubits in optimization
6. Apply peephole optimization strategies

---

## Core Content

### Gate Cancellation

**Principle:** Adjacent inverse gates cancel.

$$U U^\dagger = I$$

```
Before:                     After:
         ┌───┐   ┌────┐
q: ──────┤ H ├───┤ H  ├──    q: ─────────────────
         └───┘   └────┘

(H² = I, so HH cancels)
```

**Common cancellations:**
- $H \cdot H = I$
- $X \cdot X = Y \cdot Y = Z \cdot Z = I$
- $S \cdot S^\dagger = I$
- $T \cdot T^\dagger = I$
- $CNOT \cdot CNOT = I$

### Self-Inverse Gates

Gates that are their own inverse (involutions):

$$\boxed{H^2 = X^2 = Y^2 = Z^2 = CNOT^2 = SWAP^2 = I}$$

### Rotation Merging

Adjacent rotations about the same axis combine:

$$R_z(\theta_1) R_z(\theta_2) = R_z(\theta_1 + \theta_2)$$

```
Before:                         After:
         ┌──────┐   ┌──────┐           ┌────────────┐
q: ──────┤Rz(θ₁)├───┤Rz(θ₂)├──    q: ──┤Rz(θ₁ + θ₂)├──
         └──────┘   └──────┘           └────────────┘
```

### Commutation Rules

Gates that **commute** can be reordered to enable further optimization.

**Same-qubit commutation:**
- All diagonal gates commute: $R_z(\alpha) R_z(\beta) = R_z(\beta) R_z(\alpha)$
- Rotations about same axis commute

**Different-qubit commutation:**
Gates on different qubits always commute (tensor product is commutative up to reordering):

```
These are equivalent:
         ┌───┐                       ┌───┐
q₀: ─────┤ H ├─────────      q₀: ────┤ H ├─────
         └───┘                       └───┘
         ┌───┐                 ┌───┐
q₁: ─────┤ X ├─────────      q₁: ┤ X ├─────────
         └───┘                 └───┘
(Can parallelize or reorder)
```

### CNOT Commutation Rules

**Control qubit commutation:**
```
         ┌───┐                       ┌───┐
q₀: ─────┤ Z ├───●───   =   q₀: ──●──┤ Z ├───
         └───┘   │                │  └───┘
q₁: ─────────────⊕───       q₁: ──⊕──────────
```
Z commutes through CNOT control.

**Target qubit commutation:**
```
         ┌───┐                       ┌───┐
q₀: ─────────────●───   =   q₀: ──●──────────
                 │                │
         ┌───┐   │                │  ┌───┐
q₁: ─────┤ X ├───⊕───       q₁: ──⊕──┤ X ├───
         └───┘                       └───┘
```
X commutes through CNOT target.

### Hadamard Conjugation Rules

Conjugation by Hadamard swaps X and Z:

$$HXH = Z, \quad HZH = X, \quad HYH = -Y$$

This enables converting between different gate types:

```
Before:                     After:
         ┌───┐   ┌───┐   ┌───┐           ┌───┐
q: ──────┤ H ├───┤ X ├───┤ H ├──    q: ──┤ Z ├──
         └───┘   └───┘   └───┘           └───┘
```

### Template Matching

**Concept:** Replace gate patterns with equivalent shorter sequences.

**Example templates:**

```
Template 1: CNOT chain simplification
  ●───●───   →   ●───
  │   │          │
  ⊕───│───   →   │
      │          │
  ────⊕───   →   ⊕───

Template 2: CZ symmetry
  ●───   =   ───●
  │          │
  ●───   =   ●───
```

### T-Gate Optimization

T gates are expensive—minimize them!

**T-gate identities:**
$$T^8 = I, \quad T^4 = Z, \quad T^2 = S$$

```
Before:                     After:
──[T]──[T]──[T]──[T]──   =   ──[Z]──
```

**Phase polynomial optimization:**
Represent circuit as polynomial in Z-rotations, then synthesize optimally.

### Peephole Optimization

Scan circuit for local patterns that can be simplified:

```
Window of 2-3 gates:
   ┌───┐   ┌───┐
───┤ S ├───┤ S ├───   →   ───[Z]───
   └───┘   └───┘

   ┌───┐   ┌────┐
───┤ T ├───┤ T† ├───  →   ─────────
   └───┘   └────┘
```

### Ancilla-Assisted Optimization

Adding ancilla qubits can reduce depth/T-count:

```
Without ancilla (T-depth 7):    With ancilla (T-depth 4):
...many T gates serial...        ...T gates in parallel on ancilla...
```

Trade-off: more qubits for less depth.

### CNOT Optimization Strategies

**CNOT synthesis:** Given a linear reversible function, find minimal CNOT circuit.

**Greedy approach:**
1. Process each output bit
2. Choose CNOTs that make progress toward target
3. Minimize total CNOT count

**LU decomposition:** For certain patterns, CNOT count is $O(n^2/\log n)$.

### Circuit Identities for Optimization

**Identity 1:** CNOT direction reversal
```
  ───●───   =   ──[H]──●──[H]──
     │                 │
  ───⊕───   =   ──[H]──⊕──[H]──
```

**Identity 2:** Controlled-Z decomposition
```
  ───●───   =   ──────────●──────────
     │                    │
  ───●───   =   ──[H]─────⊕─────[H]──
```

**Identity 3:** SWAP decomposition
```
  ───×───   =   ───●───⊕───●───
     │             │   │   │
  ───×───   =   ───⊕───●───⊕───
```

---

## Worked Examples

### Example 1: Gate Cancellation

Simplify this circuit:
```
         ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
q: ──────┤ H ├───┤ T ├───┤ T ├───┤ T ├───┤ H ├──
         └───┘   └───┘   └───┘   └───┘   └───┘
```

**Solution:**

Step 1: Merge T gates
$$T \cdot T \cdot T = T^3$$

```
         ┌───┐   ┌────┐   ┌───┐
q: ──────┤ H ├───┤ T³ ├───┤ H ├──
         └───┘   └────┘   └───┘
```

Step 2: Note that $T^2 = S$, so $T^3 = T \cdot S$

```
         ┌───┐   ┌───┐   ┌───┐   ┌───┐
q: ──────┤ H ├───┤ S ├───┤ T ├───┤ H ├──
         └───┘   └───┘   └───┘   └───┘
```

**Result:** 4 gates instead of 5 (reduced T-count from 3 to 1).

### Example 2: Commutation for Cancellation

Simplify using commutation:
```
         ┌───┐         ┌───┐
q₀: ─────┤ Z ├────●────┤ Z ├────
         └───┘    │    └───┘
q₁: ──────────────⊕─────────────
```

**Solution:**

Z commutes through CNOT control, so:
```
                       ┌───┐   ┌───┐
q₀: ─────────────●─────┤ Z ├───┤ Z ├────
                 │     └───┘   └───┘
q₁: ─────────────⊕──────────────────────
```

Now Z·Z = I:
```
q₀: ─────────────●──────────────────────
                 │
q₁: ─────────────⊕──────────────────────
```

**Result:** 3 gates reduced to 1!

### Example 3: Template Matching

Apply template matching to:
```
q₀: ───●───●───
       │   │
q₁: ───⊕───⊕───
```

**Solution:**

Recognize: CNOT · CNOT = I

```
q₀: ───────────
q₁: ───────────
```

**Result:** Both CNOTs cancel completely.

---

## Practice Problems

### Problem 1: Rotation Merging
Simplify: $R_z(\pi/4) \cdot R_z(\pi/4) \cdot R_z(\pi/2)$

### Problem 2: Commutation
Can these gates be reordered to enable cancellation?
```
         ┌───┐         ┌───┐   ┌───┐
q₀: ─────┤ X ├────●────┤ X ├───┤ H ├──
         └───┘    │    └───┘   └───┘
q₁: ──────────────⊕────────────────────
```

### Problem 3: T-Count Reduction
What is the minimum T-count equivalent of $T \cdot T \cdot T \cdot T \cdot T$ (5 T gates)?

### Problem 4: CNOT Pattern
Simplify this CNOT pattern:
```
q₀: ───●───────●───
       │       │
q₁: ───⊕───●───⊕───
           │
q₂: ───────⊕───────
```

---

## Computational Lab

```python
"""Day 580: Circuit Optimization"""
import numpy as np
from collections import defaultdict

# Gate definitions
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Tdg = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

GATES = {'I': I, 'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'Sdg': Sdg, 'T': T, 'Tdg': Tdg}

def matrices_equal(A, B, tol=1e-10):
    """Check if matrices are equal up to global phase"""
    if A.shape != B.shape:
        return False
    # Find first non-zero element
    for i in range(A.size):
        if np.abs(A.flat[i]) > tol:
            phase = B.flat[i] / A.flat[i]
            return np.allclose(A * phase, B, atol=tol)
    return np.allclose(A, B, atol=tol)

class SimpleCircuit:
    """Circuit for optimization demonstration"""

    def __init__(self):
        self.gates = []  # List of (name, qubit)

    def add(self, name, qubit=0):
        self.gates.append((name, qubit))
        return self

    def to_matrix(self):
        """Compute circuit matrix (single qubit only)"""
        result = I.copy()
        for name, _ in self.gates:
            result = GATES[name] @ result
        return result

    def copy(self):
        c = SimpleCircuit()
        c.gates = self.gates.copy()
        return c

    def __repr__(self):
        return ' - '.join(f'{name}' for name, _ in self.gates) or 'I'

    def gate_count(self):
        return len(self.gates)

    def t_count(self):
        return sum(1 for name, _ in self.gates if name in ['T', 'Tdg'])

def optimize_adjacent_inverses(circuit):
    """Remove adjacent inverse gate pairs"""
    inverses = {
        'H': 'H', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
        'S': 'Sdg', 'Sdg': 'S', 'T': 'Tdg', 'Tdg': 'T'
    }

    optimized = SimpleCircuit()
    for name, qubit in circuit.gates:
        if optimized.gates and optimized.gates[-1][1] == qubit:
            last_name = optimized.gates[-1][0]
            if inverses.get(name) == last_name or inverses.get(last_name) == name:
                optimized.gates.pop()
                continue
        optimized.add(name, qubit)

    return optimized

def merge_rotations(circuit):
    """Merge consecutive T and S gates"""
    # Track rotation count (in units of pi/4)
    optimized = SimpleCircuit()
    pending_rotation = 0  # Units of pi/4

    rotation_values = {'T': 1, 'Tdg': -1, 'S': 2, 'Sdg': -2, 'Z': 4}

    for name, qubit in circuit.gates:
        if name in rotation_values:
            pending_rotation += rotation_values[name]
            pending_rotation %= 8  # T^8 = I
        else:
            # Flush pending rotation
            flush_rotation(optimized, pending_rotation, qubit)
            pending_rotation = 0
            optimized.add(name, qubit)

    # Final flush
    if pending_rotation != 0:
        flush_rotation(optimized, pending_rotation, 0)

    return optimized

def flush_rotation(circuit, rotation, qubit):
    """Convert rotation count to minimal gates"""
    rotation = rotation % 8
    if rotation == 0:
        return
    elif rotation == 1:
        circuit.add('T', qubit)
    elif rotation == 2:
        circuit.add('S', qubit)
    elif rotation == 3:
        circuit.add('S', qubit)
        circuit.add('T', qubit)
    elif rotation == 4:
        circuit.add('Z', qubit)
    elif rotation == 5:
        circuit.add('Z', qubit)
        circuit.add('T', qubit)
    elif rotation == 6:
        circuit.add('Sdg', qubit)
    elif rotation == 7:
        circuit.add('Tdg', qubit)

def optimize_circuit(circuit):
    """Apply all optimization passes"""
    prev_count = -1
    current = circuit.copy()

    while current.gate_count() != prev_count:
        prev_count = current.gate_count()
        current = optimize_adjacent_inverses(current)
        current = merge_rotations(current)

    return current

# ===== Example 1: Adjacent Inverse Cancellation =====
print("=" * 60)
print("Example 1: Adjacent Inverse Cancellation")
print("=" * 60)

c1 = SimpleCircuit()
c1.add('H').add('T').add('Tdg').add('H')
print(f"Original: {c1}")
print(f"Gates: {c1.gate_count()}, T-count: {c1.t_count()}")

c1_opt = optimize_circuit(c1)
print(f"Optimized: {c1_opt}")
print(f"Gates: {c1_opt.gate_count()}, T-count: {c1_opt.t_count()}")

# Verify equivalence
print(f"Matrices equal: {matrices_equal(c1.to_matrix(), c1_opt.to_matrix())}")

# ===== Example 2: Rotation Merging =====
print("\n" + "=" * 60)
print("Example 2: Rotation Merging")
print("=" * 60)

c2 = SimpleCircuit()
c2.add('T').add('T').add('T').add('T').add('T')  # 5 T gates
print(f"Original: {c2}")
print(f"Gates: {c2.gate_count()}, T-count: {c2.t_count()}")

c2_opt = optimize_circuit(c2)
print(f"Optimized: {c2_opt}")
print(f"Gates: {c2_opt.gate_count()}, T-count: {c2_opt.t_count()}")

print(f"Matrices equal: {matrices_equal(c2.to_matrix(), c2_opt.to_matrix())}")

# ===== Example 3: T^8 = I =====
print("\n" + "=" * 60)
print("Example 3: T^8 = I")
print("=" * 60)

c3 = SimpleCircuit()
for _ in range(8):
    c3.add('T')
print(f"Original: {c3}")
print(f"Gates: {c3.gate_count()}, T-count: {c3.t_count()}")

c3_opt = optimize_circuit(c3)
print(f"Optimized: {c3_opt}")
print(f"Gates: {c3_opt.gate_count()}, T-count: {c3_opt.t_count()}")

print(f"T^8 = I verified: {matrices_equal(c3.to_matrix(), I)}")

# ===== Example 4: Complex Optimization =====
print("\n" + "=" * 60)
print("Example 4: Complex Sequence")
print("=" * 60)

c4 = SimpleCircuit()
c4.add('H').add('T').add('T').add('T').add('T').add('H').add('H').add('S').add('Sdg')
print(f"Original: {c4}")
print(f"Gates: {c4.gate_count()}, T-count: {c4.t_count()}")

c4_opt = optimize_circuit(c4)
print(f"Optimized: {c4_opt}")
print(f"Gates: {c4_opt.gate_count()}, T-count: {c4_opt.t_count()}")

print(f"Matrices equal: {matrices_equal(c4.to_matrix(), c4_opt.to_matrix())}")

# ===== Example 5: Hadamard Conjugation =====
print("\n" + "=" * 60)
print("Example 5: Hadamard Conjugation Rules")
print("=" * 60)

print("Verifying: HXH = Z, HZH = X, HYH = -Y")
print(f"HXH = Z: {matrices_equal(H @ X @ H, Z)}")
print(f"HZH = X: {matrices_equal(H @ Z @ H, X)}")
print(f"HYH = -Y: {matrices_equal(H @ Y @ H, -Y)}")

# ===== Example 6: Gate Count Statistics =====
print("\n" + "=" * 60)
print("Example 6: Optimization Statistics")
print("=" * 60)

test_cases = [
    ("H-H", ['H', 'H']),
    ("T-T-T-T", ['T', 'T', 'T', 'T']),
    ("S-S-S-S", ['S', 'S', 'S', 'S']),
    ("H-X-H", ['H', 'X', 'H']),
    ("T-Tdg-T-Tdg", ['T', 'Tdg', 'T', 'Tdg']),
    ("Complex", ['H', 'T', 'T', 'S', 'H', 'H', 'Z', 'T', 'T', 'T']),
]

print(f"{'Circuit':<20} {'Before':<10} {'After':<10} {'Saved':<10}")
print("-" * 50)

for name, gates in test_cases:
    c = SimpleCircuit()
    for g in gates:
        c.add(g)
    c_opt = optimize_circuit(c)

    before = c.gate_count()
    after = c_opt.gate_count()
    saved = before - after

    print(f"{name:<20} {before:<10} {after:<10} {saved:<10}")

# ===== Example 7: CNOT Identities (conceptual) =====
print("\n" + "=" * 60)
print("Example 7: CNOT Optimization Identities")
print("=" * 60)

print("""
Key CNOT identities for optimization:

1. CNOT · CNOT = I
   ──●──●──  =  ───────
     │  │
   ──⊕──⊕──  =  ───────

2. Z commutes through control:
   ──Z──●──  =  ──●──Z──
        │         │
   ─────⊕──  =  ──⊕─────

3. X commutes through target:
   ─────●──  =  ──●─────
        │         │
   ──X──⊕──  =  ──⊕──X──

4. CNOT direction change with H:
   ──●──  =  ──H──⊕──H──
     │            │
   ──⊕──  =  ──H──●──H──

5. CZ is symmetric:
   ──●──  =  ──●──
     │         │
   ──●──  =  ──●──
""")

# ===== Summary =====
print("\n" + "=" * 60)
print("Circuit Optimization Summary")
print("=" * 60)
print("""
Optimization Techniques:
1. Gate Cancellation: UU† = I
2. Rotation Merging: Rz(α)Rz(β) = Rz(α+β)
3. T-gate Relations: T^8=I, T^4=Z, T^2=S
4. Commutation: Reorder to enable cancellation
5. Template Matching: Replace patterns with shorter equivalents
6. Peephole: Local sliding window optimization

Priority Order:
1. Cancel inverses (cheapest)
2. Merge rotations
3. Use commutation rules
4. Apply templates
5. Consider ancilla-based optimization
""")
```

---

## Summary

### Optimization Techniques

| Technique | Description | Example |
|-----------|-------------|---------|
| Gate cancellation | $UU^\dagger = I$ | $HH = I$ |
| Rotation merging | Combine same-axis rotations | $T \cdot T = S$ |
| Commutation | Reorder to enable cancellation | Move Z past CNOT control |
| Template matching | Replace patterns | CNOT-CNOT = I |
| Peephole | Local window optimization | Scan for patterns |

### Key Formulas

| Identity | Formula |
|----------|---------|
| Involutions | $H^2 = X^2 = Y^2 = Z^2 = I$ |
| T powers | $T^2 = S$, $T^4 = Z$, $T^8 = I$ |
| H conjugation | $HXH = Z$, $HZH = X$ |
| CNOT squared | $CNOT^2 = I$ |

### Key Takeaways

1. **Gate cancellation** is the simplest and most effective optimization
2. **Rotation merging** reduces gate count without changing functionality
3. **Commutation rules** enable moving gates past each other
4. **T-count optimization** is crucial for fault tolerance
5. **Template matching** finds and replaces inefficient patterns
6. **Multiple passes** may be needed for full optimization

---

## Daily Checklist

- [ ] I can identify and cancel inverse gate pairs
- [ ] I can merge consecutive rotations
- [ ] I understand commutation rules for optimization
- [ ] I can apply template matching
- [ ] I understand T-gate power identities
- [ ] I ran the computational lab and optimized sample circuits

---

*Next: Day 581 — Week 83 Review*
