# Day 878: Flag Circuit Design — Construction and Optimization

## Month 32: Fault-Tolerant Quantum Computing II | Week 126: Flag Qubits & Syndrome Extraction

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Flag Circuit Construction |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving: Circuit Optimization |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 878, you will be able to:

1. Construct flag circuits for arbitrary weight stabilizers
2. Optimize CNOT ordering for minimal flag count
3. Design circuits for both X-type and Z-type stabilizers
4. Apply weight-2 flag patterns systematically
5. Verify fault tolerance of constructed circuits
6. Handle non-CSS stabilizers with flag circuits

---

## Core Content

### 1. Systematic Flag Circuit Construction

The goal is to build a circuit that measures a stabilizer $S$ while using minimal ancillas and detecting all dangerous fault patterns.

**Construction Algorithm for Weight-$w$ Z-Stabilizer:**

Given $S = Z_{i_1} Z_{i_2} \cdots Z_{i_w}$:

**Step 1:** Initialize ancillas
- Syndrome qubit: $|+\rangle = H|0\rangle$
- Flag qubit(s): $|0\rangle$

**Step 2:** Apply controlled-Z gates from syndrome to data
- CNOT from syndrome to each data qubit in $S$
- Order determines error propagation

**Step 3:** Insert flag connections
- Place CNOT from flag to syndrome at strategic positions
- Flag starts in $|0\rangle$, controlled by syndrome

**Step 4:** Measure
- Flag in Z basis
- Syndrome in X basis (apply H, then measure Z)

**General Circuit Template:**

```
Flag     |0⟩ ────────────●────●────────────M_Z → f
                         │    │
Syndrome |+⟩ ──●──●── ··─X────X─·· ──●──●──M_X → s
               │  │                  │  │
Data i₁  ─────Z──┼── ·· ─────────·· ─┼──┼─────
Data i₂  ────────Z── ·· ─────────·· ─┼──┼─────
   ⋮                                 │  │
Data i_{w-1} ───────── ·· ───────·· ─Z──┼─────
Data i_w ───────────── ·· ───────·· ────Z─────
```

---

### 2. The Weight-2 Flag Pattern

The simplest and most common flag pattern uses a single flag qubit connected at two points:

**Weight-2 Flag Connection:**

```
Flag     |0⟩ ───●───●───M_Z
                │   │
Syndrome |+⟩ ───X───X───
```

This creates a "window" in the circuit. Errors inside the window are detected.

**How It Works:**

1. Flag starts in $|0\rangle$
2. First CNOT: If syndrome is in $|+\rangle$, flag becomes entangled
3. Second CNOT: If no error, flag returns to $|0\rangle$
4. X error on syndrome between CNOTs: Flag ends in $|1\rangle$

**Mathematical Analysis:**

Initial state: $|0\rangle_f |+\rangle_s = |0\rangle_f \frac{|0\rangle + |1\rangle}{\sqrt{2}}_s$

After first CNOT$_{f \to s}$ (controlled by flag... wait, that's backward):

Actually, the CNOTs are controlled by the **syndrome** qubit, targeting the flag:

```
Flag     |0⟩ ───X───X───M_Z
                │   │
Syndrome |+⟩ ───●───●───
```

After first CNOT$_{s \to f}$:
$$\frac{1}{\sqrt{2}}(|0\rangle_s|0\rangle_f + |1\rangle_s|1\rangle_f)$$

After second CNOT$_{s \to f}$:
$$\frac{1}{\sqrt{2}}(|0\rangle_s|0\rangle_f + |1\rangle_s|0\rangle_f) = |+\rangle_s|0\rangle_f$$

**Flag returns to $|0\rangle$!** (No flag triggered)

**With X error on syndrome between CNOTs:**

After first CNOT, then X on syndrome:
$$X_s \cdot \frac{1}{\sqrt{2}}(|0\rangle_s|0\rangle_f + |1\rangle_s|1\rangle_f) = \frac{1}{\sqrt{2}}(|1\rangle_s|0\rangle_f + |0\rangle_s|1\rangle_f)$$

After second CNOT:
$$\frac{1}{\sqrt{2}}(|1\rangle_s|1\rangle_f + |0\rangle_s|1\rangle_f) = |+\rangle_s|1\rangle_f$$

**Flag is $|1\rangle$!** (Flag triggered)

---

### 3. CNOT Ordering Strategies

The order of CNOTs to data qubits determines which faults are dangerous and where flags must be placed.

**Ordering Principle:**

For a single flag to catch all weight $> t$ errors, the CNOTs should be ordered so that:
- First $k$ CNOTs are before the flag window
- Last $w - k$ CNOTs are after the flag window
- $k \leq t$ and $w - k \leq t$

**For t = 1 (distance-3 codes):**

Need $k \leq 1$ and $w - k \leq 1$, so $k = 1$ and $w = 2$.

For $w > 2$, we need additional strategies.

**Solution 1: Multiple Flags**

Use multiple flag windows:

```
Flag 1   |0⟩ ───●───●───────────────M_Z
                │   │
Flag 2   |0⟩ ───┼───┼───●───●───────M_Z
                │   │   │   │
Syndrome |+⟩ ───X───X───X───X───────M_X
```

**Solution 2: Interleaved CNOTs**

Interleave data CNOTs with flag connections:

```
Flag     |0⟩ ─────────●─────────●───────────M_Z
                      │         │
Syndrome |+⟩ ───●─────X───●─────X───●───●───M_X
                │         │         │   │
Data q₁  ──────Z─────────┼─────────┼───┼────
Data q₂  ────────────────Z─────────┼───┼────
Data q₃  ──────────────────────────Z───┼────
Data q₄  ──────────────────────────────Z────
```

This creates overlapping detection windows.

---

### 4. X-Type vs Z-Type Stabilizers

**Z-Type Stabilizer:** $S = Z_{i_1} \cdots Z_{i_w}$

Uses controlled-Z (or CNOT with H conjugation):
```
Syndrome |+⟩ ───●───●───M_X
                │   │
Data         ──Z───Z──
```

**X-Type Stabilizer:** $S = X_{i_1} \cdots X_{i_w}$

Uses controlled-X (CNOT with data as control):
```
Syndrome |0⟩ ───X───X───M_Z
                │   │
Data         ──●───●──
```

Or equivalently with different basis:
```
Syndrome |+⟩ ──H──●───●──H──M_X
                  │   │
Data           ──X───X──
```

**Key Difference:**

For Z-type: X errors on syndrome propagate Z to data
For X-type: Z errors on syndrome propagate X to data

**Flag placement is symmetric** - the same principles apply, just with X↔Z exchange.

---

### 5. Optimal Circuit for Weight-4 Stabilizer

**The Chamberland-Beverland Circuit:**

For weight-4 stabilizers (common in Steane and surface codes):

```
Flag     |0⟩ ───────────●───●───────────M_Z → f
                        │   │
Syndrome |+⟩ ───●───●───X───X───●───●───M_X → s
                │   │           │   │
Data q₁  ──────Z───┼───────────┼───┼────────
Data q₂  ──────────Z───────────┼───┼────────
Data q₃  ──────────────────────Z───┼────────
Data q₄  ──────────────────────────Z────────
```

**Fault Analysis:**

| Fault Location | Data Error | Flag |
|----------------|------------|------|
| Before CNOT 1 | $Z_1 Z_2 Z_3 Z_4$ | 1 |
| After CNOT 1 | $Z_2 Z_3 Z_4$ | 1 |
| After CNOT 2 | $Z_3 Z_4$ | 1 |
| After flag window | $Z_3 Z_4$ | 0 (but weight = 2) |
| After CNOT 3 | $Z_4$ | 0 |
| After CNOT 4 | None | 0 |

**Problem:** Error after CNOT 2 or after flag window both give weight-2 errors, but different flag values!

**Solution:** Use the flag outcome to distinguish:
- Flag = 1: Error might be $Z_1Z_2$, $Z_2Z_3Z_4$, etc.
- Flag = 0: Error is $Z_3$, $Z_4$, or $Z_3Z_4$

Combined with syndrome, we can decode correctly.

---

### 6. Circuit Depth and Optimization

**Circuit Depth Metrics:**

| Metric | Definition |
|--------|------------|
| Total depth | Number of time steps |
| CNOT depth | Number of CNOT layers |
| Measurement depth | When measurements occur |

**Optimization Strategies:**

**1. Parallelize Independent CNOTs:**

If data qubits don't share stabilizers, their CNOTs can be parallel:

```
Before:                  After:
──●────────────          ──●──
  │                        │
──┼──●─────────          ──┼──●──
  │  │                     │  │
──Z──┼─────────          ──Z──┼──
     │                        │
─────Z─────────          ─────Z──
```

**2. Minimize Flag Overhead:**

Share flags between stabilizers when possible (advanced technique).

**3. Reduce Idle Time:**

Schedule operations to minimize qubit idle time (decoherence exposure).

---

### 7. Verification of Fault Tolerance

**Systematic Verification Procedure:**

1. **List all fault locations:** Every gate, preparation, and measurement
2. **For each single fault:**
   - Compute resulting data error
   - Determine flag outcome
3. **Check FT conditions:**
   - Weight ≤ t OR flag triggered
   - Distinguishable (syndrome, flag) for different errors

**Formal Verification (for complex circuits):**

Use stabilizer simulation to track error propagation:

```python
def verify_ft(circuit, t=1):
    for fault in all_single_faults(circuit):
        error = propagate_error(fault, circuit)
        weight = error.weight()
        flag = flag_outcome(fault, circuit)

        if weight > t and not flag:
            return False, fault
    return True, None
```

---

### 8. Non-CSS Stabilizers

For stabilizers with mixed X and Z (e.g., $Y = iXZ$):

**Example:** $S = X_1 Z_2 X_3 Z_4$

**Challenge:** Different Pauli types require different control directions.

**Solution:** Use the general controlled-Pauli gate:

$$\text{C-}P = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes P$$

For $P = X$: Standard CNOT
For $P = Z$: Controlled-Z
For $P = Y$: Controlled-Y

**Flag Circuit for Mixed Stabilizer:**

```
Flag     |0⟩ ─────────────●───●───────────────M_Z
                          │   │
Syndrome |+⟩ ───●────●────X───X────●────●────M_X
                │    │             │    │
Data q₁  ──────X────┼─────────────┼────┼─────  (X type)
Data q₂  ───────────Z─────────────┼────┼─────  (Z type)
Data q₃  ─────────────────────────X────┼─────  (X type)
Data q₄  ────────────────────────────Z─────  (Z type)
```

**The same flag principle applies:** Flag detects when faults in the middle of the circuit would cause high-weight errors.

---

## Practical Applications

### Hardware-Aware Circuit Design

**Connectivity Constraints:**

Real quantum hardware has limited qubit connectivity. Flag circuits must respect this:

```
Linear connectivity:     All-to-all:
q1 ─ q2 ─ q3 ─ q4       q1 ─ q2
                            ╲ ╱
                         q3 ─ q4
```

**Adaptation Strategy:**

Insert SWAP gates to enable required CNOTs, but account for additional fault locations.

### Compilation to Native Gates

Most hardware uses specific gate sets:

| Hardware | Native 2-qubit gate |
|----------|---------------------|
| IBM | CNOT or ECR |
| Google | $\sqrt{iSWAP}$ or Sycamore |
| IonQ | Molmer-Sorensen (XX) |

**Flag circuits must be compiled** to native gates while preserving fault tolerance.

---

## Worked Examples

### Example 1: Construct Flag Circuit for [[5,1,3]] Code

**Problem:** The [[5,1,3]] perfect code has weight-4 stabilizers. Design a flag circuit for $S_1 = XZZXI$.

**Solution:**

**Step 1: Identify stabilizer structure**
- Qubits involved: 1, 2, 3, 4 (qubit 5 has I)
- Types: $X_1, Z_2, Z_3, X_4$

**Step 2: Choose CNOT ordering**

Group by type for efficiency:
- X-type CNOTs: qubits 1, 4 (data controls syndrome)
- Z-type CNOTs: qubits 2, 3 (syndrome controls data)

**Step 3: Insert flag**

```
Flag     |0⟩ ─────────────────●───●─────────────────M_Z
                              │   │
Syndrome |0⟩ ───X───X────H────X───X────H────●───●───M_Z
                │   │                       │   │
Data q₁  ──────●───┼───────────────────────┼───┼────  (X)
Data q₄  ──────────●───────────────────────┼───┼────  (X)
Data q₂  ──────────────────────────────────Z───┼────  (Z)
Data q₃  ──────────────────────────────────────Z────  (Z)
```

Wait, this is getting complicated. Let me use a cleaner approach.

**Alternative: Uniform treatment**

Convert to all controlled-from-syndrome:
- For X-type: Use CNOT with H gates
- For Z-type: Use CZ

```
Flag     |0⟩ ─────────────●───●───────────────M_Z
                          │   │
Syndrome |+⟩ ───●────●────X───X────●────●────M_X
                │    │             │    │
Data q₁  ─────[X]───┼─────────────┼────┼─────
Data q₂  ───────────Z─────────────┼────┼─────
Data q₃  ─────────────────────────Z────┼─────
Data q₄  ────────────────────────────[X]─────
```

Where $[X]$ represents the X-type interaction (implemented as H-CZ-H or CNOT with reversed control).

**Step 4: Verify FT**

X faults on syndrome:
- Before any gate: Propagates all → flag = 1 ✓
- After q₁: Weight 3 → flag = 1 ✓
- After q₂: Weight 2 → flag = 1 ✓
- After flag: Weight 2 → flag = 0 (need to check correction)
- After q₃: Weight 1 → OK
- After q₄: None → OK

**Result:** Circuit is 1-flag fault-tolerant. $\square$

---

### Example 2: Optimize CNOT Ordering

**Problem:** For $S = Z_1 Z_2 Z_3 Z_4 Z_5 Z_6$ (weight-6), find CNOT ordering with minimal flags.

**Solution:**

**Single flag analysis:**

With one flag dividing the circuit at position $k$:
- Before flag: $k$ CNOTs → max error weight $6 - k + 1$ when fault before first CNOT
- After flag: $6 - k$ CNOTs → max error weight when fault right after flag

For 1-flag FT (t = 1), we need both regions to have max error ≤ 1:
- $k \leq 1$ and $6 - k \leq 1$

This is impossible! ($k$ can't be both ≤ 1 and ≥ 5)

**Two-flag solution:**

Divide into three regions with two flags:

```
Region 1: 2 CNOTs | Flag 1 | Region 2: 2 CNOTs | Flag 2 | Region 3: 2 CNOTs
```

Now each region has at most 2 CNOTs:
- Fault in region 1: Propagates to region 1 data only if after those CNOTs...

Actually, let me reconsider. An X fault on the syndrome propagates Z to *all subsequent* data qubits, not just those in the same region.

**Correct analysis:**

Place flags at positions 2 and 4:

```
Flag1    |0⟩ ─────────●───●───────────────────────M_Z
                      │   │
Flag2    |0⟩ ─────────┼───┼───────────●───●───────M_Z
                      │   │           │   │
Syndrome |+⟩ ──●──●───X───X───●──●────X───X──●──●──M_X
               │  │           │  │           │  │
Data q₁  ─────Z──┼───────────┼──┼───────────┼──┼───
Data q₂  ────────Z───────────┼──┼───────────┼──┼───
Data q₃  ────────────────────Z──┼───────────┼──┼───
Data q₄  ───────────────────────Z───────────┼──┼───
Data q₅  ───────────────────────────────────Z──┼───
Data q₆  ──────────────────────────────────────Z───
```

**Fault analysis:**

| Fault after CNOT # | Data error weight | Flag 1 | Flag 2 |
|--------------------|-------------------|--------|--------|
| 0 (before any) | 6 | 1 | 1 |
| 1 | 5 | 1 | 1 |
| 2 | 4 | 1 | 1 |
| 3 | 3 | 0 | 1 |
| 4 | 2 | 0 | 1 |
| 5 | 1 | 0 | 0 |
| 6 | 0 | 0 | 0 |

High-weight errors (> 1) all trigger at least one flag! ✓

**Result:** Two flags suffice for weight-6 stabilizer with t = 1. $\square$

---

### Example 3: Hardware-Constrained Design

**Problem:** Design a flag circuit for $Z_1 Z_2 Z_3 Z_4$ on hardware with linear connectivity: ancilla-q₁-q₂-q₃-q₄.

**Solution:**

With linear connectivity, we can't directly CNOT from ancilla to all data qubits.

**Strategy 1: Use SWAPs**

SWAP syndrome qubit along the line:

```
Syndrome ──●──SWAP──●──SWAP──●──SWAP──●──SWAP──
           │   ╳    │   ╳    │   ╳    │   ╳
Data q₁  ──Z───╳────┼───┼────┼───┼────┼───┼────
Data q₂  ──────────Z───╳────┼───┼────┼───┼────
Data q₃  ──────────────────Z───╳────┼───┼────
Data q₄  ──────────────────────────Z───╳────
```

But SWAPs introduce more fault locations!

**Strategy 2: Ancilla in middle**

Place syndrome ancilla between q₂ and q₃:

```
q₁ ─ q₂ ─ Syndrome ─ q₃ ─ q₄
          Flag
```

Now syndrome can reach q₂, q₃ directly, and q₁, q₄ with one SWAP each.

**Circuit:**

```
Flag      |0⟩ ────────────────●───●──────────────M_Z
                              │   │
Syndrome  |+⟩ ──SWAP──●──SWAP─X───X──●──SWAP──●──M_X
               ╳     │  ╳          │  ╳     │
Data q₁   ─────╳─────┼──╳──────────┼──┼─────┼────
Data q₂   ───────────Z─────────────┼──╳─────┼────
Data q₃   ─────────────────────────Z────────┼────
Data q₄   ──────────────────────────────────Z────
```

**Fault locations increase** but flag still detects dangerous patterns.

**Result:** Connectivity constraints require careful adaptation but don't break the flag principle. $\square$

---

## Practice Problems

### Level 1: Direct Application

1. **Basic construction:** Draw the flag circuit for measuring $S = Z_1 Z_2 Z_3$ with one flag.

2. **X-type stabilizer:** Convert your answer from problem 1 to measure $S = X_1 X_2 X_3$.

3. **Fault counting:** How many distinct single-fault locations are there in a flag circuit with 4 data CNOTs?

### Level 2: Intermediate

4. **Multi-flag design:** Design a two-flag circuit for weight-8 stabilizer. Verify 1-flag FT.

5. **Mixed stabilizer:** Construct a flag circuit for $S = X_1 Y_2 Z_3$ (note: $Y = iXZ$).

6. **Depth optimization:** Given a weight-4 flag circuit with depth 8, propose a parallelization that reduces depth to 6 while maintaining FT.

### Level 3: Challenging

7. **Connectivity adaptation:** Design a flag circuit for $Z_1 Z_2 Z_3 Z_4$ on a 2D grid where syndrome can only access q₁ and q₂ directly.

8. **Prove optimality:** Show that for weight-$w$ stabilizers with $w > 2t$, at least $\lceil w/(t+1) \rceil - 1$ flags are needed for t-flag FT.

9. **Research extension:** How would you modify flag circuits for non-Pauli stabilizers (e.g., $S = CZ_{12} \cdot X_3$)?

---

## Computational Lab

### Objective
Implement flag circuit construction and verify fault tolerance.

```python
"""
Day 878 Computational Lab: Flag Circuit Design
Week 126: Flag Qubits & Syndrome Extraction
"""

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Flag Circuit Data Structure
# =============================================================================

print("=" * 70)
print("Part 1: Flag Circuit Representation")
print("=" * 70)

class FlagCircuitDesigner:
    """Design and analyze flag circuits for stabilizer measurement."""

    def __init__(self, stabilizer):
        """
        Args:
            stabilizer: Dict mapping qubit index to Pauli type ('X', 'Y', 'Z')
                       Example: {0: 'Z', 1: 'Z', 2: 'Z', 3: 'Z'}
        """
        self.stabilizer = stabilizer
        self.qubits = sorted(stabilizer.keys())
        self.weight = len(self.qubits)
        self.cnot_order = list(self.qubits)  # Default order
        self.flag_positions = []

    def set_cnot_order(self, order):
        """Set the order of CNOT operations."""
        assert set(order) == set(self.qubits), "Order must include all qubits"
        self.cnot_order = list(order)

    def add_flag(self, position):
        """Add flag after CNOT at given position (0 to weight-1)."""
        assert 0 <= position <= self.weight
        self.flag_positions.append(position)
        self.flag_positions.sort()

    def analyze_fault(self, fault_position, fault_type='X'):
        """
        Analyze effect of fault on syndrome qubit.

        Args:
            fault_position: After which CNOT (0 = before any)
            fault_type: 'X' or 'Z'

        Returns:
            (affected_qubits, flags_triggered)
        """
        if fault_type == 'X':
            # X on syndrome propagates Z to subsequent data qubits
            affected = self.cnot_order[fault_position:]
        else:
            # Z on syndrome doesn't propagate to data
            affected = []

        # Determine which flags are triggered
        triggered = [i for i, fpos in enumerate(self.flag_positions)
                     if fault_position < fpos]

        return affected, triggered

    def check_fault_tolerance(self, t=1):
        """
        Check if circuit is t-flag fault-tolerant.

        Returns:
            (is_ft, problematic_faults)
        """
        problems = []

        for pos in range(self.weight + 1):
            affected, flags = self.analyze_fault(pos, 'X')
            error_weight = len(affected)

            if error_weight > t and len(flags) == 0:
                problems.append({
                    'position': pos,
                    'error_weight': error_weight,
                    'affected_qubits': affected
                })

        return len(problems) == 0, problems

    def print_circuit(self):
        """Print ASCII representation of the circuit."""
        w = self.weight
        nf = len(self.flag_positions)

        print(f"\nFlag Circuit for Stabilizer: ", end="")
        print("".join(f"{self.stabilizer[q]}_{q}" for q in self.qubits))
        print("-" * 60)

        # Flag lines
        for i, fpos in enumerate(self.flag_positions):
            line = f"Flag {i+1}  |0⟩ ─"
            for j in range(w):
                if j == fpos - 1:
                    line += "──●──●"
                else:
                    line += "──┼──┼" if j >= fpos else "─────"
            line += "── M_Z"
            print(line)

        # Syndrome line
        line = "Syndrome |+⟩ ─"
        for j in range(w):
            has_flag = j in [p-1 for p in self.flag_positions]
            if has_flag:
                line += "──X──X"
            line += "──●──" if j < w - 1 else "──●──"
        line += " M_X"
        print(line)

        # Data lines
        for i, q in enumerate(self.cnot_order):
            ptype = self.stabilizer[q]
            line = f"Data q{q}  ─────"
            for j in range(w):
                if j == i:
                    line += f"──{ptype}──"
                else:
                    line += "──┼──"
            print(line)

# Example usage
stab = {0: 'Z', 1: 'Z', 2: 'Z', 3: 'Z'}
designer = FlagCircuitDesigner(stab)
designer.add_flag(2)  # Flag after 2nd CNOT
designer.print_circuit()

# Check FT
is_ft, problems = designer.check_fault_tolerance(t=1)
print(f"\n1-Flag Fault Tolerant: {is_ft}")
if problems:
    for p in problems:
        print(f"  Problem: Fault at position {p['position']} causes weight-{p['error_weight']} error")

# =============================================================================
# Part 2: Optimal Flag Placement
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Optimal Flag Placement")
print("=" * 70)

def find_minimum_flags(weight, t=1):
    """
    Find minimum number of flags needed for t-flag FT.

    Returns:
        (n_flags, positions)
    """
    stab = {i: 'Z' for i in range(weight)}

    # Try increasing number of flags
    for n_flags in range(weight):
        # Try all possible positions for n_flags flags
        for positions in combinations(range(1, weight + 1), n_flags):
            designer = FlagCircuitDesigner(stab)
            for pos in positions:
                designer.add_flag(pos)

            is_ft, _ = designer.check_fault_tolerance(t)
            if is_ft:
                return n_flags, list(positions)

    return weight, list(range(1, weight + 1))

print("\nMinimum flags needed for 1-flag FT:")
print("-" * 40)
print(f"{'Weight':<10} {'# Flags':<10} {'Positions'}")
print("-" * 40)

for w in range(2, 10):
    n_flags, positions = find_minimum_flags(w, t=1)
    print(f"{w:<10} {n_flags:<10} {positions}")

# =============================================================================
# Part 3: CNOT Ordering Optimization
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: CNOT Ordering Optimization")
print("=" * 70)

def optimize_cnot_order(stabilizer, n_flags, t=1):
    """
    Find CNOT ordering that achieves FT with given number of flags.

    Returns:
        (success, best_order, flag_positions)
    """
    from itertools import permutations

    qubits = list(stabilizer.keys())
    weight = len(qubits)

    best = None

    for order in permutations(qubits):
        for positions in combinations(range(1, weight + 1), n_flags):
            designer = FlagCircuitDesigner(stabilizer)
            designer.set_cnot_order(list(order))
            for pos in positions:
                designer.add_flag(pos)

            is_ft, problems = designer.check_fault_tolerance(t)

            if is_ft:
                # Calculate circuit depth (simplified metric)
                depth = weight + 2 * n_flags
                if best is None or depth < best['depth']:
                    best = {
                        'order': list(order),
                        'flags': list(positions),
                        'depth': depth
                    }

    if best:
        return True, best['order'], best['flags']
    return False, None, None

# Test on weight-4 stabilizer
stab = {0: 'Z', 1: 'Z', 2: 'Z', 3: 'Z'}
success, order, flags = optimize_cnot_order(stab, n_flags=1, t=1)
print(f"\nWeight-4 stabilizer with 1 flag:")
print(f"  Success: {success}")
print(f"  CNOT order: {order}")
print(f"  Flag positions: {flags}")

# =============================================================================
# Part 4: Circuit Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Visualization")
print("=" * 70)

def visualize_flag_circuit(stabilizer, flag_positions, title="Flag Circuit"):
    """Create visual representation of flag circuit."""
    weight = len(stabilizer)
    n_flags = len(flag_positions)

    fig, ax = plt.subplots(figsize=(12, 4 + n_flags))

    # Drawing parameters
    y_spacing = 0.8
    x_spacing = 1.2
    qubit_y = {
        'flags': [4 + n_flags - i - 1 for i in range(n_flags)],
        'syndrome': 3,
        'data': [2 - i * y_spacing for i in range(weight)]
    }

    # Draw wire lines
    max_x = (weight + 2) * x_spacing

    # Flag wires
    for i, y in enumerate(qubit_y['flags']):
        ax.plot([0, max_x], [y, y], 'b-', linewidth=1.5)
        ax.text(-0.5, y, f'Flag {i+1}', ha='right', va='center', fontsize=10)
        ax.scatter([0], [y], s=100, c='blue', zorder=5)  # |0⟩ prep

    # Syndrome wire
    ax.plot([0, max_x], [qubit_y['syndrome'], qubit_y['syndrome']], 'g-', linewidth=1.5)
    ax.text(-0.5, qubit_y['syndrome'], 'Syndrome', ha='right', va='center', fontsize=10)
    ax.scatter([0], [qubit_y['syndrome']], s=100, c='green', zorder=5)  # |+⟩ prep

    # Data wires
    for i, y in enumerate(qubit_y['data']):
        ax.plot([0, max_x], [y, y], 'k-', linewidth=1.5)
        ax.text(-0.5, y, f'Data q{i}', ha='right', va='center', fontsize=10)

    # Draw CNOTs and flag connections
    for j in range(weight):
        x = (j + 1) * x_spacing

        # CNOT from syndrome to data
        y_syn = qubit_y['syndrome']
        y_data = qubit_y['data'][j]

        ax.plot([x, x], [y_syn, y_data], 'k-', linewidth=1)
        ax.scatter([x], [y_syn], s=80, c='black', marker='o', zorder=5)  # Control
        ax.scatter([x], [y_data], s=150, c='black', marker='$\\oplus$', zorder=5)  # Target

        # Flag connections
        for i, fpos in enumerate(flag_positions):
            if j == fpos - 1:
                y_flag = qubit_y['flags'][i]
                x_f1 = x + 0.2
                x_f2 = x + 0.5
                ax.plot([x_f1, x_f1], [y_flag, y_syn], 'b-', linewidth=1)
                ax.plot([x_f2, x_f2], [y_flag, y_syn], 'b-', linewidth=1)
                ax.scatter([x_f1, x_f2], [y_flag, y_flag], s=80, c='blue', zorder=5)
                ax.scatter([x_f1, x_f2], [y_syn, y_syn], s=150, c='blue', marker='$\\oplus$', zorder=5)

    # Measurements
    x_meas = max_x - 0.3
    for y in qubit_y['flags']:
        ax.annotate('M', (x_meas, y), fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.annotate('M', (x_meas, qubit_y['syndrome']), fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.set_xlim(-1.5, max_x + 0.5)
    ax.set_ylim(min(qubit_y['data']) - 0.5, max(qubit_y['flags']) + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14)

    return fig

# Visualize different configurations
fig1 = visualize_flag_circuit({i: 'Z' for i in range(4)}, [2],
                               "Weight-4 Z Stabilizer with 1 Flag")
plt.tight_layout()
plt.savefig('day_878_flag_circuit_w4.png', dpi=150, bbox_inches='tight')
plt.show()

fig2 = visualize_flag_circuit({i: 'Z' for i in range(6)}, [2, 4],
                               "Weight-6 Z Stabilizer with 2 Flags")
plt.tight_layout()
plt.savefig('day_878_flag_circuit_w6.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figures saved!")

# =============================================================================
# Part 5: Fault Analysis Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Fault Analysis")
print("=" * 70)

def analyze_all_faults(weight, flag_positions):
    """Analyze all single faults in flag circuit."""
    stab = {i: 'Z' for i in range(weight)}
    designer = FlagCircuitDesigner(stab)
    for pos in flag_positions:
        designer.add_flag(pos)

    results = []
    for pos in range(weight + 1):
        affected, flags = designer.analyze_fault(pos, 'X')
        results.append({
            'position': pos,
            'error_weight': len(affected),
            'affected': affected,
            'flags_triggered': flags,
            'is_safe': len(affected) <= 1 or len(flags) > 0
        })

    return results

# Analyze weight-4 with single flag
print("\nFault analysis for weight-4 stabilizer with flag at position 2:")
print("-" * 70)
print(f"{'Fault Pos':<12} {'Error Wt':<12} {'Affected Qubits':<20} {'Flags':<10} {'Safe?'}")
print("-" * 70)

results = analyze_all_faults(4, [2])
for r in results:
    safe = "Yes" if r['is_safe'] else "NO!"
    print(f"{r['position']:<12} {r['error_weight']:<12} {str(r['affected']):<20} {str(r['flags_triggered']):<10} {safe}")

# =============================================================================
# Part 6: Resource Comparison
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Resource Comparison")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Flags needed vs weight
ax1 = axes[0]
weights = range(2, 12)
flags_needed_t1 = [find_minimum_flags(w, t=1)[0] for w in weights]
flags_needed_t2 = [find_minimum_flags(w, t=2)[0] for w in weights]

ax1.plot(weights, flags_needed_t1, 'bo-', label='t=1 (distance 3)', linewidth=2)
ax1.plot(weights, flags_needed_t2, 'rs-', label='t=2 (distance 5)', linewidth=2)
ax1.set_xlabel('Stabilizer Weight')
ax1.set_ylabel('Minimum Flags Needed')
ax1.set_title('Flag Count vs Stabilizer Weight')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Total ancillas comparison
ax2 = axes[1]
weights = range(2, 10)
shor_ancillas = [w + 2 for w in weights]  # w for cat + 2 for verification
flag_ancillas = [1 + find_minimum_flags(w, t=1)[0] for w in weights]  # 1 syndrome + flags

x = np.arange(len(weights))
width = 0.35

bars1 = ax2.bar(x - width/2, shor_ancillas, width, label='Shor-style', color='coral')
bars2 = ax2.bar(x + width/2, flag_ancillas, width, label='Flag-based', color='steelblue')

ax2.set_xlabel('Stabilizer Weight')
ax2.set_ylabel('Ancilla Qubits')
ax2.set_title('Ancilla Overhead: Shor vs Flag')
ax2.set_xticks(x)
ax2.set_xticklabels(weights)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('day_878_resource_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_878_resource_comparison.png'")

# =============================================================================
# Part 7: Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary: Flag Circuit Design")
print("=" * 70)

print("""
Key Design Principles:

1. CNOT ORDERING: Order determines error propagation
   - Faults on syndrome propagate to subsequent data qubits
   - Optimal ordering minimizes max error in any region

2. FLAG PLACEMENT: Divide circuit into safe regions
   - Weight-2 flag window detects errors in the middle
   - Multiple flags for high-weight stabilizers

3. MINIMUM FLAGS:
   - Weight 2: 0 flags (already safe)
   - Weight 3-4: 1 flag
   - Weight 5-6: 2 flags
   - General: ⌈w/2⌉ - 1 flags for t=1

4. VERIFICATION: Check all single faults
   - Each fault: weight ≤ t OR triggers flag
   - Use systematic enumeration

5. OPTIMIZATION:
   - Minimize flag count
   - Reduce circuit depth
   - Respect hardware connectivity

Tomorrow: Complete Flag-FT Error Correction Protocols
""")

print("=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Weight-2 flag pattern | Two CNOTs from syndrome to flag create detection window |
| Minimum flags (t=1) | $\lceil w/2 \rceil - 1$ for weight-$w$ stabilizer |
| Flag positions | Divide CNOTs into regions of size ≤ $t + 1$ |
| Circuit depth | Base depth + $2 \times n_{\text{flags}}$ |

### Main Takeaways

1. **Systematic construction:** Follow standard template for any stabilizer
2. **CNOT ordering matters:** Determines which faults are dangerous
3. **Weight-2 flag pattern:** Two CNOTs create a detection window
4. **X and Z types:** Same principle, different control directions
5. **Verification is essential:** Check all single faults before deployment

---

## Daily Checklist

- [ ] Construct a flag circuit from scratch for a given stabilizer
- [ ] Determine optimal flag positions for weight-5 stabilizer
- [ ] Verify fault tolerance by enumerating all faults
- [ ] Design circuit for mixed X/Z stabilizer
- [ ] Complete Level 1 practice problems
- [ ] Run computational lab and generate visualizations
- [ ] Explain the weight-2 flag detection mechanism

---

## Preview: Day 879

Tomorrow we complete the picture with **full flag-FT error correction protocols**: how to use syndrome and flag outcomes together, build lookup tables, handle multiple rounds of extraction, and achieve complete fault-tolerant operation.

---

*"Good circuit design is the foundation of practical fault tolerance."*

---

**Next:** [Day_879_Thursday.md](Day_879_Thursday.md) — Flag FT Error Correction: Complete Protocols
