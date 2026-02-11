# Day 805: Week 115 Synthesis — Surface Code Implementation

## Month 29: Topological Codes | Week 115: Surface Code Implementation
### Year 2: Advanced Quantum Science

---

## Schedule Overview

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Comprehensive review, concept integration |
| **Afternoon** | 2.5 hours | Design principles, architecture comparison |
| **Evening** | 1.5 hours | Preview of Week 116: Error chains and decoding |

**Total Study Time**: 7 hours

---

## Learning Objectives

By the end of Day 805, you will be able to:

1. **Synthesize** all Week 115 concepts into a unified understanding
2. **Apply** surface code design principles to new scenarios
3. **Compare** different surface code architectures systematically
4. **Evaluate** tradeoffs for specific hardware constraints
5. **Prepare** for error chain analysis and decoding (Week 116)
6. **Demonstrate** mastery through comprehensive problem solving

---

## Morning Session: Comprehensive Review (3 hours)

### 1. Week 115 Concept Map

```
                    SURFACE CODE IMPLEMENTATION
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    STRUCTURE              OPERATIONS          VARIANTS
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐         ┌────┴────┐
    │         │          │         │         │         │
Boundaries  Qubits   Syndrome   Logical   Rotated  Defects
    │         │      Extraction Operators    │         │
    │         │          │         │         │         │
  Smooth   Data     CNOT order  X-chains  45° rot  Holes
  Rough    Ancilla  Hook errors Z-chains  Qubit    Twists
                    Weight-2/3  Anticomm  savings  Braiding
```

### 2. Key Concepts Summary

#### Day 799: From Torus to Plane — Boundaries

**Core Insight**: Periodic boundaries are impossible in planar hardware; we must use open boundaries.

| Boundary Type | Terminates | Condensed Anyon | Logical Operator |
|---------------|------------|------------------|------------------|
| Smooth (X-type) | Plaquettes | m-anyons | Logical X ends here |
| Rough (Z-type) | Vertices | e-anyons | Logical Z ends here |

**Key Formula**: With 2 smooth + 2 rough boundaries → $k = 1$ logical qubit

#### Day 800: Planar Surface Code Structure

**Core Insight**: The surface code is a CSS code with qubits on a lattice.

| Code Type | Data Qubits | Parameters | Total Qubits |
|-----------|-------------|------------|--------------|
| Unrotated | $d^2$ | $[[d^2, 1, d]]$ | $2d^2 - 1$ |
| Rotated | $(d^2+1)/2$ | $[[(d^2+1)/2, 1, d]]$ | $d^2$ |

**Key Property**: CSS structure ensures $H_X H_Z^T = 0$ (mod 2)

#### Day 801: Syndrome Extraction Circuits

**Core Insight**: Ancilla-based measurement extracts syndrome without collapsing logical state.

| Circuit Element | X-Stabilizer | Z-Stabilizer |
|-----------------|--------------|--------------|
| Ancilla prep | $\|0\rangle \to \|+\rangle$ | $\|0\rangle$ |
| CNOT direction | Ancilla → Data | Data → Ancilla |
| Measurement | X-basis (via H+Z) | Z-basis |

**Key Warning**: Hook errors from mid-circuit faults; mitigate with diagonal CNOT ordering

#### Day 802: Logical Operators on Planar Code

**Core Insight**: Logical operators are strings connecting same-type boundaries.

| Operator | Path | Weight | Boundary Connection |
|----------|------|--------|---------------------|
| $\bar{Z}$ | Vertical | $d$ | Rough → Rough |
| $\bar{X}$ | Horizontal | $d$ | Smooth → Smooth |

**Key Property**: $\bar{X}$ and $\bar{Z}$ anticommute (overlap = 1 qubit)

#### Day 803: Defects and Holes

**Core Insight**: Internal defects provide alternative qubit encodings and gate mechanisms.

| Defect Type | Description | Encoding | Gate Method |
|-------------|-------------|----------|-------------|
| Smooth hole | X-type internal boundary | Pair → 1 qubit | Braiding |
| Rough hole | Z-type internal boundary | Pair → 1 qubit | Braiding |
| Twist (genon) | Boundary type transition | 4 → 2 qubits | Non-Abelian braiding |

**Key Comparison**: Defect braiding vs. lattice surgery for logical operations

#### Day 804: Rotated Surface Code

**Core Insight**: 45° rotation saves ~50% of physical qubits.

| Property | Unrotated | Rotated |
|----------|-----------|---------|
| Data qubits | $d^2$ | $(d^2+1)/2$ |
| Bulk stabilizers | Weight-4 | Weight-4 |
| Boundary stabilizers | Weight-2,3,4 | Weight-2 only |
| Hardware mapping | Square grid | Checkerboard |

### 3. Master Formula Reference

$$\boxed{\text{Surface Code Key Equations}}$$

**Encoding parameters**:
$$[[n, k, d]] = \begin{cases}
[[d^2, 1, d]] & \text{unrotated} \\
[[\frac{d^2+1}{2}, 1, d]] & \text{rotated}
\end{cases}$$

**Stabilizer count**:
$$n_X + n_Z = n - k = \begin{cases}
d^2 - 1 & \text{unrotated} \\
\frac{d^2-1}{2} & \text{rotated}
\end{cases}$$

**Logical error rate** (below threshold):
$$p_L \approx C \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}$$

**Error suppression factor**:
$$\Lambda = \frac{p_{\text{th}}}{p} \approx \frac{0.01}{p}$$

---

## Afternoon Session: Design Principles and Comparison (2.5 hours)

### 1. Surface Code Design Principles

When designing a surface code implementation, follow these principles:

#### Principle 1: Match Distance to Target Error Rate

Given physical error rate $p$ and target logical error rate $p_L$:

$$d \geq 2 \cdot \frac{\log(C/p_L)}{\log(p_{\text{th}}/p)} - 1$$

**Example**: For $p = 0.1\%$, $p_{\text{th}} = 1\%$, $p_L = 10^{-10}$:
$$d \geq 2 \cdot \frac{\log(0.1/10^{-10})}{\log(10)} - 1 = 2 \cdot 9 - 1 = 17$$

#### Principle 2: Choose Layout for Hardware Constraints

| Hardware Type | Recommended Layout | Rationale |
|---------------|-------------------|-----------|
| Dense square grid | Rotated surface code | Minimum qubit count |
| Sparse connectivity | Heavy-hex adapted | Reduced crosstalk |
| Reconfigurable | Standard boundaries | Flexibility |
| Fixed 2D array | Lattice surgery compatible | Easy patching |

#### Principle 3: Optimize Syndrome Extraction

1. **Use diagonal CNOT ordering** to minimize hook error impact
2. **Parallelize X and Z syndrome** extraction (they commute)
3. **Match syndrome rounds to distance** for time-like protection
4. **Consider flag qubits** for weight-4 stabilizers if threshold is marginal

#### Principle 4: Plan for Logical Operations

| Operation Type | Method | Overhead |
|----------------|--------|----------|
| Pauli gates | Frame tracking | None |
| Clifford gates | Lattice surgery / Transversal | Space |
| T gate | Magic state injection | Magic states |
| Measurement | Transversal | None |

### 2. Architecture Comparison Framework

#### Evaluation Criteria

Score each architecture (1-5) on:

1. **Qubit Efficiency**: Qubits per logical qubit at target distance
2. **Gate Speed**: Time to execute logical operations
3. **Threshold**: Error threshold for fault tolerance
4. **Connectivity**: Required physical qubit connectivity
5. **Scalability**: Ease of increasing code distance

#### Comparison Table

| Architecture | Qubit Eff. | Gate Speed | Threshold | Connectivity | Scalability |
|--------------|------------|------------|-----------|--------------|-------------|
| Unrotated + boundaries | 3 | 4 | 4 | 4 | 4 |
| Rotated + boundaries | 5 | 4 | 4 | 4 | 4 |
| Hole-encoded | 3 | 3 | 3 | 4 | 3 |
| Defect/genon | 2 | 4 | 3 | 3 | 2 |
| Heavy-hex adapted | 4 | 4 | 3 | 5 | 4 |

**Recommendation**: Rotated surface code with lattice surgery is the current best choice for most near-term platforms.

### 3. Decision Tree for Implementation

```
START: Choose surface code variant
  │
  ├─> Is connectivity limited?
  │     │
  │     ├─> YES: Use heavy-hex or sparse layout
  │     │         Consider reduced threshold
  │     │
  │     └─> NO: Use rotated surface code
  │
  ├─> What is physical error rate p?
  │     │
  │     ├─> p > 0.5%: Need aggressive optimization
  │     │              Consider concatenation
  │     │
  │     └─> p < 0.5%: Standard implementation
  │                   Focus on scaling distance
  │
  ├─> What logical operations needed?
  │     │
  │     ├─> Clifford only: Lattice surgery
  │     │                   No magic states
  │     │
  │     └─> Universal: Magic state factories
  │                     Space overhead ~10x
  │
  └─> Target logical error rate?
        │
        ├─> p_L > 10^-6: Distance 7-11
        │
        ├─> 10^-10 < p_L < 10^-6: Distance 11-21
        │
        └─> p_L < 10^-10: Distance 21+
                          Consider code concatenation
```

### 4. Worked Example: Complete Design Exercise

**Scenario**: Design a fault-tolerant quantum computer with:
- 100 logical qubits
- Logical error rate $< 10^{-12}$ per gate
- Physical error rate $p = 0.1\%$
- Square lattice connectivity

**Solution**:

Step 1: Determine required distance
$$p_L = C (p/p_{\text{th}})^{(d+1)/2} < 10^{-12}$$

With $C = 0.1$, $p = 0.001$, $p_{\text{th}} = 0.01$:
$$0.1 \times 0.1^{(d+1)/2} < 10^{-12}$$
$$0.1^{(d+1)/2} < 10^{-11}$$
$$(d+1)/2 > 11$$
$$d > 21$$

Choose $d = 23$ (next odd number).

Step 2: Calculate qubits per logical qubit (rotated)
$$n_{\text{data}} = \frac{23^2 + 1}{2} = 265$$
$$n_{\text{total}} = 23^2 = 529$$

Step 3: Total physical qubits for computation
- Logical qubits: 100 × 529 = 52,900
- Magic state factory: ~10 × 529 = 5,290 per factory
- Need ~10 factories: 52,900
- **Total: ~106,000 physical qubits**

Step 4: Syndrome extraction overhead
- Per round: 6 circuit layers
- Rounds per logical gate: $d = 23$
- **Total: 138 layers per logical gate**

$$\boxed{\text{Design: } d=23 \text{ rotated code, } \sim 10^5 \text{ physical qubits}}$$

---

## Practice Problems: Comprehensive Assessment

### Problem Set 805

#### Conceptual Understanding

1. **Boundary analysis**: Explain why a surface code with all four boundaries of the same type encodes zero logical qubits. What is the dimension of the code space?

2. **Error propagation**: Trace a Z-error on a central data qubit through one complete syndrome extraction round. Which ancillas report non-trivial syndrome?

3. **Logical equivalence**: Prove that two Z-strings connecting rough boundaries, differing by a horizontal "jog", represent the same logical Z operator.

#### Quantitative Analysis

4. **Qubit budget**: A quantum computer has 1000 physical qubits with square connectivity. What is the maximum logical qubit count and code distance achievable using rotated surface codes?

5. **Threshold calculation**: If a rotated surface code has 80% of stabilizers at weight-4 (with error probability $4p$) and 20% at weight-2 (with error probability $2p$), estimate the effective threshold compared to uniform weight-4.

6. **Time overhead**: For a distance-11 surface code with syndrome extraction taking 6 layers at 100ns per layer, how long does one logical CNOT via lattice surgery take (assuming $d$ syndrome rounds)?

#### Design Challenges

7. **Asymmetric code**: Design a surface code where X-distance is 5 but Z-distance is 9. Sketch the layout and identify the logical operators.

8. **Defect encoding**: You have a surface code with two smooth holes and one rough hole. How many logical qubits are encoded? What are their logical operators?

9. **Error correction simulation**: Given syndrome measurements from 5 consecutive rounds, where round 3 shows a pair of adjacent X-syndromes that wasn't present in rounds 1-2 and disappears in rounds 4-5, diagnose the most likely error.

#### Advanced Synthesis

10. **Architecture optimization**: Compare the total physical qubit count and logical gate time for implementing a 50-logical-qubit computer at $p_L = 10^{-10}$ using:
    (a) Rotated surface codes with lattice surgery
    (b) Hole-encoded qubits with defect braiding
    Assume $p = 0.2\%$ and $p_{\text{th}} = 1\%$.

---

## Evening Session: Preview of Week 116 (1.5 hours)

### 1. Error Chains and the Decoding Problem

Week 116 addresses the central question: **Given a syndrome, which error occurred?**

#### The Ambiguity Problem

Multiple errors can produce the same syndrome:

```
Error 1: Z on qubit A       Error 2: Z on qubit B
    ↓                            ↓
Syndrome: (1,1,0,0)         Syndrome: (1,1,0,0)

Same syndrome, different errors!
```

The decoder must choose the **most likely** error consistent with the syndrome.

#### Error Chains

Errors form **chains** on the lattice:
- Single Z-error: Creates two e-anyons (endpoints of chain)
- Error chain: Connected path of errors with anyons at endpoints
- Syndrome: Locations of chain endpoints

The decoder reconstructs chains from endpoints.

### 2. Minimum Weight Perfect Matching (MWPM)

The dominant decoding algorithm:

1. **View syndrome as graph problem**: Syndrome bits are nodes
2. **Find minimum weight matching**: Pair syndrome nodes with shortest paths
3. **Correction**: Apply Pauli string along matched paths

**Complexity**: $O(n^3)$ for $n$ syndrome nodes (or better with optimizations)

### 3. Thresholds and Phase Transitions

The surface code exhibits a **phase transition**:
- Below threshold: Logical errors exponentially suppressed
- Above threshold: Logical errors grow with code size
- At threshold: Critical behavior (universality class)

The threshold corresponds to a **random-bond Ising model** phase transition.

### 4. Topics for Week 116

| Day | Topic |
|-----|-------|
| 806 | Error chains and syndrome graphs |
| 807 | Minimum weight perfect matching |
| 808 | Union-find decoder |
| 809 | Neural network decoders |
| 810 | Threshold analysis |
| 811 | Space-time decoding |
| 812 | Week 116 synthesis |

---

## Computational Lab: Integration Test

```python
"""
Day 805 Computational Lab: Comprehensive Surface Code Integration
Testing understanding of all Week 115 concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass

class ComprehensiveSurfaceCode:
    """
    Unified surface code implementation combining all Week 115 concepts.
    """

    def __init__(self, d: int, rotated: bool = True):
        """Initialize surface code with specified parameters."""
        self.d = d
        self.rotated = rotated

        if rotated:
            self.n_data = (d * d + 1) // 2
        else:
            self.n_data = d * d

        self.k = 1
        self._build_code()

    def _build_code(self):
        """Build complete code structure."""
        # Placeholder for full implementation
        d = self.d
        self.x_stabilizers = []
        self.z_stabilizers = []

        # Build stabilizers based on code type
        if self.rotated:
            self._build_rotated_stabilizers()
        else:
            self._build_unrotated_stabilizers()

    def _build_rotated_stabilizers(self):
        """Build stabilizers for rotated code."""
        d = self.d
        # Simplified: weight-4 bulk, weight-2 boundary
        n_bulk = (d - 1) * (d - 1) // 2
        n_boundary = 2 * (d - 1)

        for i in range(n_bulk):
            self.x_stabilizers.append({'weight': 4, 'type': 'bulk'})
            self.z_stabilizers.append({'weight': 4, 'type': 'bulk'})

        for i in range(n_boundary // 2):
            self.x_stabilizers.append({'weight': 2, 'type': 'boundary'})
            self.z_stabilizers.append({'weight': 2, 'type': 'boundary'})

    def _build_unrotated_stabilizers(self):
        """Build stabilizers for unrotated code."""
        d = self.d
        # Mix of weight-2, 3, 4
        for i in range((d-1) * (d-1)):
            self.x_stabilizers.append({'weight': 4, 'type': 'bulk'})
        for i in range(4):
            self.x_stabilizers.append({'weight': 2, 'type': 'corner'})
        for i in range(4 * (d - 2)):
            self.x_stabilizers.append({'weight': 3, 'type': 'edge'})

        # Similar for Z
        self.z_stabilizers = self.x_stabilizers.copy()

    def get_summary(self) -> Dict:
        """Return comprehensive code summary."""
        x_weights = {}
        for stab in self.x_stabilizers:
            w = stab['weight']
            x_weights[w] = x_weights.get(w, 0) + 1

        return {
            'code_type': 'rotated' if self.rotated else 'unrotated',
            'distance': self.d,
            'data_qubits': self.n_data,
            'logical_qubits': self.k,
            'total_qubits': self.d * self.d if self.rotated else 2 * self.d * self.d - 1,
            'x_stabilizers': len(self.x_stabilizers),
            'z_stabilizers': len(self.z_stabilizers),
            'stabilizer_weights': x_weights
        }

    def calculate_logical_error_rate(self, p: float, p_th: float = 0.01, C: float = 0.1) -> float:
        """Calculate logical error rate."""
        exponent = (self.d + 1) / 2
        return C * (p / p_th) ** exponent


def comprehensive_analysis():
    """Perform comprehensive analysis of surface codes."""
    print("="*70)
    print("COMPREHENSIVE SURFACE CODE ANALYSIS - WEEK 115 SYNTHESIS")
    print("="*70)

    # 1. Code comparison
    print("\n" + "-"*70)
    print("1. CODE TYPE COMPARISON")
    print("-"*70)

    for d in [5, 7, 9]:
        rotated = ComprehensiveSurfaceCode(d, rotated=True)
        unrotated = ComprehensiveSurfaceCode(d, rotated=False)

        rot_summary = rotated.get_summary()
        unrot_summary = unrotated.get_summary()

        print(f"\nDistance {d}:")
        print(f"  Rotated:   {rot_summary['data_qubits']} data, {rot_summary['total_qubits']} total")
        print(f"  Unrotated: {unrot_summary['data_qubits']} data, {unrot_summary['total_qubits']} total")
        savings = (unrot_summary['data_qubits'] - rot_summary['data_qubits']) / unrot_summary['data_qubits'] * 100
        print(f"  Savings:   {savings:.1f}%")

    # 2. Logical error rate scaling
    print("\n" + "-"*70)
    print("2. LOGICAL ERROR RATE SCALING")
    print("-"*70)

    p_values = [0.001, 0.002, 0.005]
    distances = [5, 7, 9, 11, 13]

    print(f"\n{'p':<10}", end="")
    for d in distances:
        print(f"{'d=' + str(d):<12}", end="")
    print()

    for p in p_values:
        print(f"{p*100:.1f}%{'':<6}", end="")
        for d in distances:
            code = ComprehensiveSurfaceCode(d)
            p_L = code.calculate_logical_error_rate(p)
            print(f"{p_L:.2e}{'':>2}", end="")
        print()

    # 3. Resource estimation
    print("\n" + "-"*70)
    print("3. RESOURCE ESTIMATION FOR FAULT-TOLERANT COMPUTING")
    print("-"*70)

    target_logical_qubits = [10, 50, 100, 500]
    target_p_L = 1e-10
    p_phys = 0.001

    print(f"\nTarget logical error rate: {target_p_L:.0e}")
    print(f"Physical error rate: {p_phys*100:.1f}%")
    print()

    # Find required distance
    for d in range(3, 51, 2):
        code = ComprehensiveSurfaceCode(d)
        p_L = code.calculate_logical_error_rate(p_phys)
        if p_L < target_p_L:
            required_d = d
            break

    print(f"Required distance: {required_d}")
    print()

    print(f"{'Logical Qubits':<18} {'Physical Qubits':<18} {'Overhead':<12}")
    print("-" * 48)

    for n_logical in target_logical_qubits:
        code = ComprehensiveSurfaceCode(required_d)
        n_physical = n_logical * code.get_summary()['total_qubits']
        overhead = n_physical / n_logical

        print(f"{n_logical:<18} {n_physical:<18} {overhead:.0f}x")

    # 4. Architecture decision matrix
    print("\n" + "-"*70)
    print("4. ARCHITECTURE DECISION MATRIX")
    print("-"*70)

    print("""
    Scenario                      Recommended Architecture
    ─────────────────────────────────────────────────────────────────────
    Dense square grid             Rotated surface code + lattice surgery
    Sparse connectivity           Heavy-hex adapted surface code
    Maximum flexibility           Hole-based encoding
    Research/exploration          Defect braiding demonstration
    Near-term NISQ               Small distance, focus on threshold demo
    Long-term fault-tolerant      d=15+ rotated with magic state factory
    """)

    # 5. Key formulas summary
    print("\n" + "-"*70)
    print("5. KEY FORMULAS SUMMARY")
    print("-"*70)

    print("""
    QUBIT COUNTS:
      Rotated:   n_data = (d² + 1)/2,  n_total = d²
      Unrotated: n_data = d²,          n_total = 2d² - 1

    STABILIZER STRUCTURE:
      Rotated:   Weight-4 (bulk), Weight-2 (boundary)
      Unrotated: Weight-4,3,2 (depending on position)

    LOGICAL OPERATORS:
      Z̄: Vertical string (rough → rough), weight = d
      X̄: Horizontal string (smooth → smooth), weight = d

    ERROR SCALING:
      p_L ≈ C · (p/p_th)^((d+1)/2)
      Threshold: p_th ≈ 1% for standard surface codes

    SYNDROME EXTRACTION:
      Circuit depth: 6 layers (prep + 4 CNOT + measure)
      Rounds needed: d (for full space-time protection)
    """)


def visualization_gallery():
    """Generate visualization gallery for week summary."""
    print("\n" + "="*70)
    print("VISUALIZATION GALLERY")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Qubit scaling
    ax = axes[0, 0]
    distances = range(3, 25, 2)
    rotated = [(d**2 + 1)//2 for d in distances]
    unrotated = [d**2 for d in distances]

    ax.plot(distances, unrotated, 'b-o', label='Unrotated')
    ax.plot(distances, rotated, 'r-s', label='Rotated')
    ax.set_xlabel('Distance d')
    ax.set_ylabel('Data Qubits')
    ax.set_title('Qubit Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Logical error rate
    ax = axes[0, 1]
    p_ratios = np.linspace(0.1, 0.9, 50)

    for d in [5, 9, 13, 17]:
        p_L = 0.1 * p_ratios ** ((d+1)/2)
        ax.semilogy(p_ratios, p_L, label=f'd={d}')

    ax.set_xlabel('p / p_threshold')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title('Error Suppression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Total overhead
    ax = axes[0, 2]
    p_values = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    target_p_L = 1e-10
    C = 0.1
    p_th = 0.01

    for target in [1e-8, 1e-10, 1e-12]:
        overheads = []
        for p in p_values:
            # Find required d
            for d in range(3, 101, 2):
                p_L = C * (p / p_th) ** ((d+1)/2)
                if p_L < target:
                    overheads.append(d**2)
                    break

        ax.plot(p_values * 100, overheads, '-o', label=f'p_L={target:.0e}')

    ax.set_xlabel('Physical Error Rate (%)')
    ax.set_ylabel('Total Qubits per Logical Qubit')
    ax.set_title('Resource Overhead')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Boundary types
    ax = axes[1, 0]
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)

    # Draw simplified surface code
    for i in range(5):
        for j in range(5):
            ax.scatter(j, 4-i, s=100, c='black')

    ax.axvline(x=-0.3, color='blue', linewidth=4, label='Smooth')
    ax.axvline(x=4.3, color='blue', linewidth=4)
    ax.axhline(y=-0.3, color='red', linewidth=4, label='Rough')
    ax.axhline(y=4.3, color='red', linewidth=4)

    ax.set_title('Boundary Types')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.axis('off')

    # 5. Stabilizer weights
    ax = axes[1, 1]
    weights_rotated = {'Weight-4': 8, 'Weight-2': 4}
    weights_unrotated = {'Weight-4': 8, 'Weight-3': 4, 'Weight-2': 4}

    x = np.arange(3)
    width = 0.35

    ax.bar(x[:2] - width/2, list(weights_rotated.values()),
           width, label='Rotated', color='steelblue')
    ax.bar(x - width/2 + width, list(weights_unrotated.values())[:3] if len(weights_unrotated) > 2 else list(weights_unrotated.values()) + [0],
           width, label='Unrotated', color='coral')

    ax.set_xlabel('Stabilizer Weight')
    ax.set_ylabel('Count')
    ax.set_title('Stabilizer Distribution (d=5)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Weight-2', 'Weight-3', 'Weight-4'])
    ax.legend()

    # 6. Threshold region
    ax = axes[1, 2]
    p = np.linspace(0, 0.02, 100)
    p_th = 0.01

    for d in [3, 5, 7, 11]:
        p_L = 0.1 * (p / p_th) ** ((d+1)/2)
        p_L = np.clip(p_L, 0, 1)
        ax.plot(p * 100, p_L, label=f'd={d}')

    ax.axvline(x=1, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Physical Error Rate (%)')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title('Threshold Behavior')
    ax.legend()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('week_115_synthesis.png', dpi=150, bbox_inches='tight')
    plt.show()


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DAY 805: WEEK 115 SYNTHESIS")
    print("Surface Code Implementation - Comprehensive Review")
    print("="*70)

    # Run comprehensive analysis
    comprehensive_analysis()

    # Generate visualizations
    visualization_gallery()

    print("\n" + "="*70)
    print("WEEK 115 COMPLETE")
    print("="*70)
    print("""
    Key Accomplishments:
    ✓ Understood why boundaries replace periodic conditions
    ✓ Mastered planar surface code structure
    ✓ Learned syndrome extraction with hook error mitigation
    ✓ Identified logical operators as boundary-to-boundary strings
    ✓ Explored defects and holes for alternative encodings
    ✓ Compared rotated vs unrotated code efficiency

    Ready for Week 116: Decoding and Thresholds
    """)
```

---

## Summary

### Week 115 Master Reference

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 799 | Boundaries | Smooth (X-type) and rough (Z-type) replace periodic |
| 800 | Structure | $[[d^2, 1, d]]$ unrotated, $[[(d^2+1)/2, 1, d]]$ rotated |
| 801 | Syndrome | Ancilla circuits, hook errors, CNOT ordering |
| 802 | Logical Ops | Z: rough→rough, X: smooth→smooth, anticommute |
| 803 | Defects | Holes and twists for encoding and gates |
| 804 | Rotated | 45° rotation saves ~50% qubits |
| 805 | Synthesis | Design principles, architecture comparison |

### Key Formulas Card

$$\boxed{\begin{aligned}
&\textbf{Rotated Surface Code:}\\
&n_{\text{data}} = \frac{d^2 + 1}{2}, \quad n_{\text{total}} = d^2\\[2mm]
&\textbf{Logical Error Rate:}\\
&p_L \approx C \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}\\[2mm]
&\textbf{Threshold:}\\
&p_{\text{th}} \approx 1\% \text{ (depolarizing)}
\end{aligned}}$$

---

## Daily Checklist: Week 115 Mastery

Verify complete understanding:

- [ ] Explain boundary types and their effect on logical qubits
- [ ] Construct stabilizer layouts for rotated and unrotated codes
- [ ] Design syndrome extraction circuits avoiding hook errors
- [ ] Identify logical operators from boundary structure
- [ ] Compare defect-based and boundary-based encodings
- [ ] Calculate resource requirements for target error rates
- [ ] Evaluate architecture tradeoffs for specific hardware

---

## Preview: Week 116

**Week 116: Decoding and Thresholds**

Next week addresses the fundamental algorithmic challenge: given syndrome measurements, how do we infer and correct errors?

Topics include:
- Error chains and syndrome graphs
- Minimum weight perfect matching (MWPM)
- Union-find decoder
- Neural network approaches
- Threshold analysis and phase transitions
- Space-time decoding for repeated measurements

The decoder is the "brain" of the surface code—without it, the beautiful structure we've built remains inert.

---

## References

1. Fowler, A. G., et al. "Surface codes: Towards practical large-scale quantum computation." *Physical Review A* 86, 032324 (2012)
2. Dennis, E., et al. "Topological quantum memory." *Journal of Mathematical Physics* 43, 4452 (2002)
3. Google Quantum AI. "Suppressing quantum errors by scaling a surface code logical qubit." *Nature* 614, 676 (2023)
4. Bombin, H. "An introduction to topological quantum codes." arXiv:1311.0277 (2013)
5. Terhal, B. M. "Quantum error correction for quantum memories." *Reviews of Modern Physics* 87, 307 (2015)

---

*Week 115 has established the complete foundation for surface code quantum error correction—from abstract topology to concrete hardware implementation. The surface code stands ready to protect quantum information.*
