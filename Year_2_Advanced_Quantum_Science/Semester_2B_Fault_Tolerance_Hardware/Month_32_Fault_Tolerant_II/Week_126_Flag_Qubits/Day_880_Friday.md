# Day 880: Flags on Various Codes — Steane, Surface, and Color Codes

## Month 32: Fault-Tolerant Quantum Computing II | Week 126: Flag Qubits & Syndrome Extraction

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Code-Specific Flag Implementations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving: Comparative Analysis |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 880, you will be able to:

1. Implement flag circuits for the Steane [[7,1,3]] code
2. Analyze syndrome extraction in surface codes
3. Design flag circuits for color codes
4. Compare flag overhead across different code families
5. Identify code-specific optimizations for flag circuits
6. Evaluate which codes benefit most from flag techniques

---

## Core Content

### 1. The Steane [[7,1,3]] Code with Flags

The Steane code is the canonical example for flag qubit techniques.

**Code Properties:**
- 7 physical qubits
- 1 logical qubit
- Distance 3 (corrects 1 error)
- CSS code: X and Z stabilizers separate
- All stabilizers weight-4

**Stabilizers:**

| Generator | Qubits | Type |
|-----------|--------|------|
| $S_1^X$ | $X_1 X_3 X_5 X_7$ | X-type |
| $S_2^X$ | $X_2 X_3 X_6 X_7$ | X-type |
| $S_3^X$ | $X_4 X_5 X_6 X_7$ | X-type |
| $S_1^Z$ | $Z_1 Z_3 Z_5 Z_7$ | Z-type |
| $S_2^Z$ | $Z_2 Z_3 Z_6 Z_7$ | Z-type |
| $S_3^Z$ | $Z_4 Z_5 Z_6 Z_7$ | Z-type |

**Flag Circuit for Weight-4 Stabilizer:**

```
Flag     |0⟩ ─────────────●───●───────────M_Z → f
                          │   │
Syndrome |+⟩ ───●───●─────X───X─────●───●───M_X → s
                │   │               │   │
Data q₁  ──────Z───┼───────────────┼───┼─────
Data q₃  ──────────Z───────────────┼───┼─────
Data q₅  ──────────────────────────Z───┼─────
Data q₇  ──────────────────────────────Z─────
```

**Resource Count:**

- Per stabilizer: 2 ancillas (1 syndrome + 1 flag)
- Total for 6 stabilizers: 12 ancillas
- Compare to Shor-style: 36 ancillas

$$\boxed{\text{Steane code savings: } \frac{36 - 12}{36} = 67\%}$$

**Reichardt's Minimal Implementation:**

Remarkably, the Steane code can be corrected with just **2 ancilla qubits total** using sequential extraction and clever circuit design.

---

### 2. Surface Code Syndrome Extraction

The surface code uses a different paradigm: local stabilizers on a 2D lattice.

**Surface Code Structure:**

```
    ○───●───○───●───○
    │   │   │   │   │
    ●───○───●───○───●
    │   │   │   │   │
    ○───●───○───●───○
    │   │   │   │   │
    ●───○───●───○───●

    ○ = X-type stabilizer (plaquette)
    ● = Z-type stabilizer (vertex)
```

**Stabilizer Properties:**
- Weight-4 in bulk (weight-2 at boundaries)
- Local interactions only (nearest-neighbor)
- Each data qubit in exactly 2 X-plaquettes and 2 Z-vertices

**Standard Surface Code Syndrome Extraction:**

The surface code typically uses **bare ancilla** syndrome extraction:

```
Ancilla |0⟩ ──H──●───●───●───●──H──M
                 │   │   │   │
Data NW  ───────Z───┼───┼───┼─────
Data NE  ───────────Z───┼───┼─────
Data SW  ───────────────Z───┼─────
Data SE  ───────────────────Z─────
```

**Why Flags Are Less Common for Surface Codes:**

1. **High threshold** without flags (~1% with MWPM decoder)
2. **Spatial redundancy:** Errors manifest as syndrome chains
3. **Efficient classical decoder:** MWPM handles correlated errors
4. **No transversal non-Clifford gates** anyway

**When Flags Help Surface Codes:**

- Very low distance (d=3, d=5)
- Memory-limited implementations
- Hybrid architectures

---

### 3. Color Codes with Flags

Color codes are topological codes defined on 3-colorable lattices.

**Color Code Properties:**
- Qubits on vertices of triangular lattice
- Stabilizers on faces (plaquettes)
- Each face has one color; edges connect different colors
- Transversal implementation of all Clifford gates

**Example: [[7,1,3]] Color Code (Steane Code as Color Code):**

The Steane code is actually the smallest triangular color code!

```
        1
       /|\
      / | \
     5--+--3
    /|\ | /|\
   / | \|/ | \
  6--+--7--+--2
     |     |
     4─────
```

**Stabilizers as Face Operators:**
- Red face: qubits 1, 2, 3
- Green face: qubits 1, 5, 6
- Blue face: qubits 1, 4, 7
- Central face: qubits 3, 5, 7

**Flag Circuits for Color Codes:**

Color code stabilizers can have various weights:
- Triangular faces: weight-3
- Hexagonal faces: weight-6

**Weight-6 Hexagonal Stabilizer:**

```
Flag 1   |0⟩ ─────────●───●─────────────────M_Z
                      │   │
Flag 2   |0⟩ ─────────┼───┼───────●───●─────M_Z
                      │   │       │   │
Syndrome |+⟩ ───●─●───X───X───●───X───X─●─●─M_X
                │ │           │         │ │
Data q₁  ──────Z─┼───────────┼─────────┼─┼──
Data q₂  ────────Z───────────┼─────────┼─┼──
Data q₃  ────────────────────Z─────────┼─┼──
Data q₄  ──────────────────────────────Z─┼──
Data q₅  ────────────────────────────────Z──
Data q₆  ────────────────────────────────────
```

Wait, that's only 5 qubits. Let me correct for weight-6:

```
Flag 1   |0⟩ ───────────●───●─────────────────────M_Z
                        │   │
Flag 2   |0⟩ ───────────┼───┼───────────●───●─────M_Z
                        │   │           │   │
Syndrome |+⟩ ───●───●───X───X───●───●───X───X───M_X
                │   │           │   │
Data q₁  ──────Z───┼───────────┼───┼───────────────
Data q₂  ──────────Z───────────┼───┼───────────────
Data q₃  ──────────────────────Z───┼───────────────
Data q₄  ──────────────────────────Z───────────────
Data q₅  ──────────────────────────────────────Z───
Data q₆  ────────────────────────────────────────Z─
```

**Resource Count for Color Codes:**

| Stabilizer Weight | Flags Needed | Total Ancillas |
|-------------------|--------------|----------------|
| 3 | 0 | 1 |
| 4 | 1 | 2 |
| 6 | 2 | 3 |
| 8 | 3 | 4 |

---

### 4. Comparative Analysis

**Resource Comparison Across Code Families:**

| Code | Distance | Data Qubits | Shor Ancillas | Flag Ancillas | Savings |
|------|----------|-------------|---------------|---------------|---------|
| [[5,1,3]] Perfect | 3 | 5 | 24 | 8 | 67% |
| [[7,1,3]] Steane | 3 | 7 | 36 | 12 | 67% |
| [[9,1,3]] Shor | 3 | 9 | 12 | 10 | 17% |
| Surface d=3 | 3 | 17 | 32 | 16 | 50% |
| Color d=3 | 3 | 7 | 36 | 12 | 67% |

**Threshold Comparison:**

| Code + Method | Approximate Threshold |
|---------------|----------------------|
| Steane + Shor-style | 0.3% |
| Steane + Flag | 0.2% |
| Surface + bare ancilla | 1.0% |
| Surface + Flag | 0.8% |
| Color + Flag | 0.2% |

**Key Observations:**

1. **CSS codes benefit most** from flag techniques (structured stabilizers)
2. **Surface codes** already have high threshold without flags
3. **Small codes** see largest relative resource savings
4. **Transversal gate support** makes Steane/color codes attractive despite lower threshold

---

### 5. Code-Specific Optimizations

**Steane Code Optimizations:**

1. **Parallel extraction:** X and Z stabilizers can be measured simultaneously
2. **Shared ancillas:** Sequential extraction allows ancilla reuse
3. **Reichardt's 2-ancilla scheme:** Extreme minimization

**Surface Code Optimizations:**

1. **Hook errors:** Special care for weight-2 errors from ancilla faults
2. **Alternating schedules:** Reduce crosstalk between neighboring stabilizers
3. **Flag bridges:** Connect ancillas to detect multi-qubit errors

**Color Code Optimizations:**

1. **Gauge fixing:** Reduce effective stabilizer weight
2. **Flag sharing:** Neighboring stabilizers can share flag qubits
3. **Restriction decoders:** Leverage color code structure

---

### 6. Hardware Considerations

**Superconducting Qubits (IBM, Google):**

- Heavy-hex connectivity limits direct flag implementations
- SWAP overhead can negate flag benefits
- Surface code more natural for 2D lattice

**Trapped Ions (IonQ, Quantinuum):**

- All-to-all connectivity ideal for flags
- Steane/color codes particularly suitable
- Mid-circuit measurement well-supported

**Neutral Atoms (QuEra, Pasqal):**

- Reconfigurable connectivity
- Can adapt to flag circuit requirements
- Color codes on triangular lattice natural

---

### 7. Flag Bridges and Advanced Techniques

**Flag Bridge Concept (Chamberland et al.):**

Share flag qubits between multiple stabilizer measurements:

```
                    Stabilizer 1        Stabilizer 2
                         │                   │
Flag Bridge |0⟩ ────●────┼────●────────●────┼────●────M
                    │    │    │        │    │    │
Syndrome 1  |+⟩ ────X────●────X────────┼────┼────┼────M
                         │             │    │    │
Syndrome 2  |+⟩ ────────────────────X────●────X────M
```

**Advantages:**
- Reduced ancilla count
- Detect correlated errors across stabilizers
- Better suited for some code geometries

**Challenges:**
- More complex circuits
- Increased depth
- Harder classical decoding

---

## Practical Applications

### IBM Quantum Implementation

IBM's heavy-hex topology requires adaptation:

```
Standard Heavy-Hex Unit:

    ○───○
   /│   │\
  ○─●───●─○
   \│   │/
    ○───○

  ● = Data qubit
  ○ = Ancilla qubit
```

**Flag-Compatible Layout:**
- Ancillas at degree-2 vertices for syndrome
- Additional ancillas at degree-3 vertices for flags
- Limited by physical connectivity

### Google Sycamore

Google's grid topology:
```
○─○─○─○─○
│ │ │ │ │
○─○─○─○─○
│ │ │ │ │
○─○─○─○─○
```

Surface code is native; flags add overhead without proportional benefit.

### Quantinuum H-Series

All-to-all connectivity:
- Any qubit can interact with any other
- Flag circuits implementable without SWAP overhead
- Steane code demonstrations already published

---

## Worked Examples

### Example 1: Complete Steane Code Flag Implementation

**Problem:** Design the full set of flag circuits for the [[7,1,3]] Steane code.

**Solution:**

**X-type stabilizers (measured with Z ancilla):**

$S_1^X = X_1 X_3 X_5 X_7$:
```
Flag₁    |0⟩ ─────────────●───●───────────M_Z → f₁
                          │   │
Synd₁    |0⟩ ───●───●─────X───X─────●───●───M_Z → s₁
                │   │               │   │
q₁       ──────X───┼───────────────┼───┼─────
q₃       ──────────X───────────────┼───┼─────
q₅       ──────────────────────────X───┼─────
q₇       ──────────────────────────────X─────
```

$S_2^X = X_2 X_3 X_6 X_7$:
```
Flag₂    |0⟩ ─────────────●───●───────────M_Z → f₂
                          │   │
Synd₂    |0⟩ ───●───●─────X───X─────●───●───M_Z → s₂
                │   │               │   │
q₂       ──────X───┼───────────────┼───┼─────
q₃       ──────────X───────────────┼───┼─────
q₆       ──────────────────────────X───┼─────
q₇       ──────────────────────────────X─────
```

$S_3^X = X_4 X_5 X_6 X_7$:
```
Flag₃    |0⟩ ─────────────●───●───────────M_Z → f₃
                          │   │
Synd₃    |0⟩ ───●───●─────X───X─────●───●───M_Z → s₃
                │   │               │   │
q₄       ──────X───┼───────────────┼───┼─────
q₅       ──────────X───────────────┼───┼─────
q₆       ──────────────────────────X───┼─────
q₇       ──────────────────────────────X─────
```

**Z-type stabilizers (measured with X ancilla):**

Similar structure with CNOT direction reversed (data controls ancilla).

$S_1^Z = Z_1 Z_3 Z_5 Z_7$:
```
Flag₄    |0⟩ ─────────────●───●───────────M_Z → f₄
                          │   │
Synd₄    |+⟩ ───●───●─────X───X─────●───●───M_X → s₄
                │   │               │   │
q₁       ──────Z───┼───────────────┼───┼─────
q₃       ──────────Z───────────────┼───┼─────
q₅       ──────────────────────────Z───┼─────
q₇       ──────────────────────────────Z─────
```

(Similar for $S_2^Z$ and $S_3^Z$)

**Total Resources:**
- 12 ancilla qubits (6 syndrome + 6 flag)
- Can be reduced to 2 with sequential extraction

**Result:** Complete flag implementation for Steane code. $\square$

---

### Example 2: Surface Code Hook Error Analysis

**Problem:** Analyze how a single ancilla fault creates a "hook error" in surface code syndrome extraction and whether flags help.

**Solution:**

**Standard Surface Code Z-Stabilizer Circuit:**

```
Ancilla |+⟩ ───●───●───●───●───M_X
               │   │   │   │
Data NW  ─────Z───┼───┼───┼─────
Data NE  ─────────Z───┼───┼─────
Data SW  ─────────────Z───┼─────
Data SE  ─────────────────Z─────
```

**Hook Error Scenario:**

X fault on ancilla between NE and SW CNOTs:

```
Ancilla |+⟩ ───●───●───X───●───●───M_X
               │   │   ↑   │   │
Data NW  ─────Z───┼───────┼───┼─────
Data NE  ─────────Z───────┼───┼─────
Data SW  ─────────────────Z───┼─────
Data SE  ─────────────────────Z─────
```

The X error propagates to $Z_{SW} Z_{SE}$ - a weight-2 error!

**With Flag Circuit:**

```
Flag     |0⟩ ─────────────●───●───────────M_Z
                          │   │
Ancilla  |+⟩ ───●───●─────X───X─────●───●───M_X
                │   │               │   │
Data NW  ──────Z───┼───────────────┼───┼─────
Data NE  ──────────Z───────────────┼───┼─────
Data SW  ──────────────────────────Z───┼─────
Data SE  ──────────────────────────────Z─────
```

**Analysis:**

X fault on ancilla after NE CNOT, before flag:
- Triggers flag (flag = 1)
- Propagates to $Z_{SW} Z_{SE}$
- Flagged, so decoder knows to look for weight-2

**Benefit:**

The flag allows distinguishing hook errors from single data errors, improving correction accuracy for small codes.

**Caveat:**

For large surface codes, MWPM decoder handles hook errors statistically without flags.

**Result:** Flags help small surface codes but are less critical for large ones. $\square$

---

### Example 3: Color Code Flag Sharing

**Problem:** Design a flag-sharing scheme for adjacent color code plaquettes.

**Solution:**

Consider two adjacent hexagonal faces sharing an edge:

```
    1───2
   /     \
  6       3
   \     /
    5───4
   /     \
  10      7
   \     /
    9───8
```

**Face 1:** Qubits 1, 2, 3, 4, 5, 6 (weight-6)
**Face 2:** Qubits 4, 5, 7, 8, 9, 10 (weight-6)
**Shared edge:** Qubits 4, 5

**Standard approach:** 2 flags per face = 4 total flags

**Flag-sharing approach:**

Use 3 shared flags that detect errors on both faces:

```
Flag A: Detects faults affecting {1,2,3} in Face 1
Flag B: Detects faults affecting {4,5} (shared edge)
Flag C: Detects faults affecting {7,8,9} in Face 2
```

**Circuit (conceptual):**

```
Flag A   |0⟩ ───●───●─────────────────────────M_Z
                │   │
Flag B   |0⟩ ───┼───┼───●───●─────────────────M_Z  (shared)
                │   │   │   │
Flag C   |0⟩ ───┼───┼───┼───┼───●───●─────────M_Z
                │   │   │   │   │   │
Synd 1   |+⟩ ───X───X───X───X───┼───┼─────────M_X
                                │   │
Synd 2   |+⟩ ─────────────X───X───X───X───────M_X
```

**Savings:** 3 flags instead of 4 (25% reduction in flag qubits)

**Result:** Flag sharing reduces overhead for adjacent stabilizers. $\square$

---

## Practice Problems

### Level 1: Direct Application

1. **Steane syndrome:** For the Steane code, compute the syndrome for a single $Z_4$ error using X-stabilizers.

2. **Flag count:** How many flags are needed for a weight-8 stabilizer with t=1 fault tolerance?

3. **Surface vs Steane:** For distance-5, compare the data qubit count of surface code vs Steane code.

### Level 2: Intermediate

4. **Color code circuits:** Design flag circuits for all face operators of the [[7,1,3]] color code (treating it as a color code, not Steane code).

5. **Hook error rate:** If physical error rate is $p = 0.001$, estimate the probability of an undetected hook error in surface code syndrome extraction.

6. **Sequential extraction:** Design a scheme to extract all 6 Steane code syndromes using only 2 ancilla qubits.

### Level 3: Challenging

7. **Threshold comparison:** Using numerical simulation or published data, compare the threshold of [[7,1,3]] Steane code with flags vs surface code distance-3.

8. **Flag bridges:** Design a flag bridge circuit for three adjacent surface code stabilizers. How many flag qubits are needed?

9. **Optimal codes:** For a 20-qubit quantum computer, what code/flag combination maximizes effective distance while maintaining $p_{th} > 0.1\%$?

---

## Computational Lab

### Objective
Implement and compare flag circuits across different code families.

```python
"""
Day 880 Computational Lab: Flags on Various Codes
Week 126: Flag Qubits & Syndrome Extraction
"""

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Code Definitions
# =============================================================================

print("=" * 70)
print("Part 1: Quantum Error Correction Code Definitions")
print("=" * 70)

class QECCode:
    """Base class for quantum error correction codes."""

    def __init__(self, name, n, k, d):
        self.name = name
        self.n = n  # Physical qubits
        self.k = k  # Logical qubits
        self.d = d  # Distance
        self.x_stabilizers = []  # List of qubit lists
        self.z_stabilizers = []

    def max_stabilizer_weight(self):
        all_stabs = self.x_stabilizers + self.z_stabilizers
        return max(len(s) for s in all_stabs) if all_stabs else 0

    def total_stabilizers(self):
        return len(self.x_stabilizers) + len(self.z_stabilizers)


class SteaneCode(QECCode):
    """[[7,1,3]] Steane code."""

    def __init__(self):
        super().__init__("Steane [[7,1,3]]", 7, 1, 3)

        # X-type stabilizers (indices are 0-based)
        self.x_stabilizers = [
            [0, 2, 4, 6],  # X₁X₃X₅X₇
            [1, 2, 5, 6],  # X₂X₃X₆X₇
            [3, 4, 5, 6],  # X₄X₅X₆X₇
        ]

        # Z-type stabilizers
        self.z_stabilizers = [
            [0, 2, 4, 6],  # Z₁Z₃Z₅Z₇
            [1, 2, 5, 6],  # Z₂Z₃Z₆Z₇
            [3, 4, 5, 6],  # Z₄Z₅Z₆Z₇
        ]


class PerfectCode(QECCode):
    """[[5,1,3]] Perfect code."""

    def __init__(self):
        super().__init__("Perfect [[5,1,3]]", 5, 1, 3)

        # Mixed XZZXI type stabilizers
        self.x_stabilizers = [
            [0, 2],  # X positions in XZZXI
            [1, 3],  # etc.
        ]
        self.z_stabilizers = [
            [1, 2],  # Z positions
            [2, 3],
        ]
        # Note: [[5,1,3]] has non-CSS stabilizers, simplified here


class SurfaceCode(QECCode):
    """Distance-3 surface code (rotated)."""

    def __init__(self, d=3):
        n_data = d ** 2
        super().__init__(f"Surface d={d}", n_data, 1, d)

        # For d=3: 9 data qubits in 3x3 grid
        # Stabilizers are weight-4 (bulk) or weight-2 (boundary)

        if d == 3:
            # X-type (plaquette) stabilizers
            self.x_stabilizers = [
                [0, 1, 3, 4],  # Center plaquette
                [1, 2],       # Right edge
                [3, 6],       # Bottom left
                [5, 8],       # Bottom right
            ]

            # Z-type (vertex) stabilizers
            self.z_stabilizers = [
                [0, 3],       # Top left
                [2, 5],       # Top right
                [3, 4, 6, 7], # Center
                [5, 8],       # Bottom
            ]


class ColorCode(QECCode):
    """[[7,1,3]] Color code (triangular lattice)."""

    def __init__(self):
        super().__init__("Color [[7,1,3]]", 7, 1, 3)

        # For triangular color code, X and Z stabilizers are same qubits
        # Each face is both X-type and Z-type
        self.x_stabilizers = [
            [0, 1, 2],     # Red face
            [0, 3, 4],     # Green face
            [0, 5, 6],     # Blue face
            [2, 4, 6],     # Central triangle
        ]

        self.z_stabilizers = [
            [0, 1, 2],
            [0, 3, 4],
            [0, 5, 6],
            [2, 4, 6],
        ]


# Create code instances
codes = {
    'steane': SteaneCode(),
    'perfect': PerfectCode(),
    'surface_d3': SurfaceCode(3),
    'color': ColorCode(),
}

print("\nCode Summary:")
print("-" * 60)
print(f"{'Code':<25} {'n':<5} {'k':<5} {'d':<5} {'Max Weight':<12} {'# Stabs'}")
print("-" * 60)
for name, code in codes.items():
    print(f"{code.name:<25} {code.n:<5} {code.k:<5} {code.d:<5} "
          f"{code.max_stabilizer_weight():<12} {code.total_stabilizers()}")

# =============================================================================
# Part 2: Flag Resource Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Flag Resource Analysis")
print("=" * 70)

def flags_needed(weight, t=1):
    """Calculate minimum flags needed for weight-w stabilizer."""
    if weight <= t + 1:
        return 0
    return (weight - 1) // (t + 1)

def shor_ancillas(weight):
    """Ancillas for Shor-style (cat state + verification)."""
    return weight + 2

def analyze_code_resources(code, t=1):
    """Analyze resource requirements for a code."""
    all_stabs = code.x_stabilizers + code.z_stabilizers

    shor_total = sum(shor_ancillas(len(s)) for s in all_stabs)
    flag_total = sum(1 + flags_needed(len(s), t) for s in all_stabs)

    return {
        'shor_ancillas': shor_total,
        'flag_ancillas': flag_total,
        'savings': (shor_total - flag_total) / shor_total * 100 if shor_total > 0 else 0,
        'stabilizers': len(all_stabs),
    }

print("\nResource Comparison (Shor-style vs Flag-based):")
print("-" * 70)
print(f"{'Code':<25} {'Shor Anc':<12} {'Flag Anc':<12} {'Savings':<12} {'# Stabs'}")
print("-" * 70)

for name, code in codes.items():
    resources = analyze_code_resources(code)
    print(f"{code.name:<25} {resources['shor_ancillas']:<12} "
          f"{resources['flag_ancillas']:<12} {resources['savings']:.1f}%"
          f"{'':<8} {resources['stabilizers']}")

# =============================================================================
# Part 3: Steane Code Flag Circuits
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Steane Code Flag Circuit Details")
print("=" * 70)

steane = codes['steane']

def print_flag_circuit(stabilizer_type, stab_idx, qubits):
    """Print ASCII representation of flag circuit."""
    print(f"\n{stabilizer_type}-stabilizer {stab_idx + 1}: ", end="")
    print("".join(f"{stabilizer_type}{q+1}" for q in qubits))
    print("-" * 50)

    # Flag line
    print(f"Flag     |0⟩ ─────────────●───●───────────M_Z")
    print(f"                          │   │")

    # Syndrome line
    basis = "+⟩" if stabilizer_type == "Z" else "0⟩"
    meas = "M_X" if stabilizer_type == "Z" else "M_Z"
    print(f"Syndrome |{basis} ───●───●─────X───X─────●───●───{meas}")
    print(f"                │   │               │   │")

    # Data lines
    for i, q in enumerate(qubits):
        gate = "Z" if stabilizer_type == "Z" else "X"
        spacing = "───" * i + gate + "───" + "───┼" * (3 - i)
        print(f"q{q+1:<7} ──────{spacing}────")

print("Steane Code Flag Circuits:")

for i, stab in enumerate(steane.z_stabilizers):
    print_flag_circuit("Z", i, stab)

# =============================================================================
# Part 4: Error Detection Simulation
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Error Detection Simulation")
print("=" * 70)

def simulate_syndrome(code, error_qubits, error_type='X'):
    """Compute syndrome for given error pattern."""
    if error_type == 'X':
        # X errors detected by Z stabilizers
        stabilizers = code.z_stabilizers
    else:
        # Z errors detected by X stabilizers
        stabilizers = code.x_stabilizers

    syndrome = []
    for stab in stabilizers:
        overlap = len(set(error_qubits) & set(stab))
        syndrome.append(overlap % 2)

    return tuple(syndrome)

# Test on Steane code
print("\nSteane code syndrome table for single X errors:")
print("-" * 40)
print(f"{'Error':<15} {'Syndrome':<20}")
print("-" * 40)

for q in range(steane.n):
    syndrome = simulate_syndrome(steane, [q], 'X')
    print(f"X_{q+1:<13} {str(syndrome):<20}")

# Weight-2 errors
print("\nSelected weight-2 X errors:")
print("-" * 40)
for q1, q2 in [(0, 1), (0, 2), (2, 4), (0, 6)]:
    syndrome = simulate_syndrome(steane, [q1, q2], 'X')
    print(f"X_{q1+1}X_{q2+1:<10} {str(syndrome):<20}")

# =============================================================================
# Part 5: Code Comparison Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Ancilla comparison
ax1 = axes[0, 0]
code_names = [codes[k].name for k in codes]
shor_anc = [analyze_code_resources(codes[k])['shor_ancillas'] for k in codes]
flag_anc = [analyze_code_resources(codes[k])['flag_ancillas'] for k in codes]

x = np.arange(len(code_names))
width = 0.35

bars1 = ax1.bar(x - width/2, shor_anc, width, label='Shor-style', color='coral')
bars2 = ax1.bar(x + width/2, flag_anc, width, label='Flag-based', color='steelblue')

ax1.set_ylabel('Total Ancilla Qubits')
ax1.set_title('Ancilla Overhead: Shor vs Flag')
ax1.set_xticks(x)
ax1.set_xticklabels([n.replace(' ', '\n') for n in code_names], fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add savings percentage
for i, (s, f) in enumerate(zip(shor_anc, flag_anc)):
    savings = (s - f) / s * 100 if s > 0 else 0
    ax1.annotate(f'{savings:.0f}%', (i, max(s, f) + 1), ha='center', fontsize=9)

# Plot 2: Stabilizer weight distribution
ax2 = axes[0, 1]
for name, code in codes.items():
    all_stabs = code.x_stabilizers + code.z_stabilizers
    weights = [len(s) for s in all_stabs]
    weight_counts = {}
    for w in weights:
        weight_counts[w] = weight_counts.get(w, 0) + 1

    ax2.bar([w + 0.15 * list(codes.keys()).index(name) for w in weight_counts.keys()],
            weight_counts.values(), width=0.15, label=code.name, alpha=0.7)

ax2.set_xlabel('Stabilizer Weight')
ax2.set_ylabel('Count')
ax2.set_title('Stabilizer Weight Distribution')
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Flags needed vs stabilizer weight
ax3 = axes[1, 0]
weights = range(2, 12)
flags_t1 = [flags_needed(w, t=1) for w in weights]
flags_t2 = [flags_needed(w, t=2) for w in weights]

ax3.plot(weights, flags_t1, 'bo-', label='t=1 (distance 3)', linewidth=2)
ax3.plot(weights, flags_t2, 'rs-', label='t=2 (distance 5)', linewidth=2)
ax3.set_xlabel('Stabilizer Weight')
ax3.set_ylabel('Flags Needed')
ax3.set_title('Flag Count vs Stabilizer Weight')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Total qubit overhead
ax4 = axes[1, 1]
distances = [3, 5, 7, 9]

steane_qubits = [7]  # Only distance 3 for Steane
surface_qubits = [d**2 for d in distances]

ax4.plot(distances[:len(steane_qubits)], steane_qubits, 'go', markersize=12,
         label='Steane family (d=3 only)')
ax4.plot(distances, surface_qubits, 'b^-', markersize=10, label='Surface code')

# Add concatenated Steane for higher distance
concat_steane = [7, 49, 343]  # 7, 7², 7³
ax4.plot([3, 5, 7], concat_steane[:3], 'g--', alpha=0.5, label='Steane concatenated')

ax4.set_xlabel('Code Distance')
ax4.set_ylabel('Data Qubits')
ax4.set_title('Qubit Count Scaling')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('day_880_code_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_880_code_comparison.png'")

# =============================================================================
# Part 6: Summary Table
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Summary")
print("=" * 70)

print("""
Code Family Comparison for Flag Techniques:

╔═══════════════════╦═══════════╦═══════════╦═════════════╦═════════════════╗
║ Code Family       ║ Best For  ║ Threshold ║ Flag Benefit║ Transversal     ║
╠═══════════════════╬═══════════╬═══════════╬═════════════╬═════════════════╣
║ Steane [[7,1,3]]  ║ Near-term ║ ~0.2%     ║ High (67%)  ║ H, S, CNOT      ║
║ Perfect [[5,1,3]] ║ Research  ║ ~0.1%     ║ High (67%)  ║ H, limited      ║
║ Surface Code      ║ Scaling   ║ ~1.0%     ║ Low (~50%)  ║ None            ║
║ Color Code        ║ Gates     ║ ~0.2%     ║ High (67%)  ║ Full Clifford   ║
╚═══════════════════╩═══════════╩═══════════╩═════════════╩═════════════════╝

Key Insights:

1. STEANE CODE: Ideal for flag techniques
   - Uniform weight-4 stabilizers
   - CSS structure simplifies circuits
   - 2-ancilla minimum implementation possible

2. SURFACE CODE: Flags less critical
   - Already has high threshold
   - MWPM decoder handles hook errors
   - Flags help only for small distances

3. COLOR CODE: Good flag compatibility
   - Variable stabilizer weights
   - Flag sharing across faces possible
   - Preserves transversal Clifford gates

4. PERFECT CODE: Theoretical importance
   - Non-CSS makes flags trickier
   - Good for understanding principles
   - Less practical for hardware
""")

print("=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Code | Data Qubits | Max Stab Weight | Flag Ancillas | Savings |
|------|-------------|-----------------|---------------|---------|
| Steane [[7,1,3]] | 7 | 4 | 12 | 67% |
| Surface d=3 | 9 | 4 | ~8 | 50% |
| Color [[7,1,3]] | 7 | 4 | 12 | 67% |

### Main Takeaways

1. **Steane/color codes** benefit most from flag techniques (uniform weights, CSS structure)
2. **Surface codes** have high threshold without flags due to MWPM decoding
3. **Flag sharing** between adjacent stabilizers reduces overhead further
4. **Hardware connectivity** determines practical flag implementation
5. **Trade-off:** Flags reduce qubits at cost of slightly lower threshold

---

## Daily Checklist

- [ ] Design complete flag circuits for Steane code
- [ ] Explain why surface codes need fewer flags
- [ ] Compare resource requirements across codes
- [ ] Identify code-specific optimizations
- [ ] Complete Level 1 practice problems
- [ ] Run computational lab
- [ ] Determine best code for a given hardware

---

## Preview: Day 881

Tomorrow's computational lab integrates everything: full simulation of flag-FT error correction, error injection, syndrome analysis, and performance comparison.

---

*"Different codes call for different strategies - there is no one-size-fits-all in quantum error correction."*

---

**Next:** [Day_881_Saturday.md](Day_881_Saturday.md) — Computational Lab: Full Flag Circuit Simulation
