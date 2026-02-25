# Day 719: Advantages of Subsystem Codes

## Overview

**Date:** Day 719 of 1008
**Week:** 103 (Subsystem Codes)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Practical Benefits and Applications of Subsystem Codes

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Measurement weight reduction |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Fault tolerance and single-shot |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Implementation considerations |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Explain** why gauge operators enable lower-weight measurements
2. **Describe** the fault-tolerance advantages of subsystem codes
3. **Define** single-shot error correction and its requirements
4. **Compare** measurement circuits for stabilizer vs subsystem codes
5. **Analyze** the resource trade-offs in practical implementations
6. **Evaluate** when subsystem codes are preferable to stabilizer codes

---

## Core Content

### 1. The Measurement Weight Problem

#### Why Measurement Weight Matters

In stabilizer codes, syndrome extraction requires measuring stabilizer generators.

**High-weight stabilizers create problems:**

1. **More CNOT gates:** Weight-$w$ stabilizer needs $w$ CNOTs
2. **Error propagation:** A fault can spread to $w$ data qubits
3. **Measurement errors:** More gates = more noise
4. **Connectivity requirements:** May need non-local connections

#### Example: Shor Code vs Bacon-Shor

**Shor [[9,1,3]] stabilizer code:**
- X-stabilizers: weight 6 (e.g., $X_1X_2X_3X_4X_5X_6$)
- Z-stabilizers: weight 6
- Syndrome extraction: 6-CNOT circuits

**Bacon-Shor [[9,1,4,3]] subsystem code:**
- Gauge operators: weight 2 (e.g., $X_1X_2$)
- Syndrome from products: still get same information
- Measurement circuits: 2-CNOT circuits!

---

### 2. Weight Reduction via Gauge Operators

#### The Key Insight

**Theorem:** For a subsystem code, stabilizer syndromes can be inferred from gauge measurements.

Since $\mathcal{S} \subseteq \mathcal{G}$ (center of gauge group):
$$S = \prod_{i \in I} G_i \quad \text{for some gauge generators } G_i$$

**Implication:** Measure each $G_i$ (low weight), compute syndrome of $S$ by taking products.

#### Measurement Protocol

**Step 1:** Measure all gauge generators $\{G_i\}$
- Get outcomes $\{g_i = \pm 1\}$

**Step 2:** Compute stabilizer syndromes
- For stabilizer $S = \prod_i G_i^{a_i}$, syndrome is $s = \prod_i g_i^{a_i}$

**Step 3:** Decode using stabilizer syndrome
- Same decoding as stabilizer code

#### Weight Comparison Table

| Code | Stabilizer Weight | Gauge Weight | Reduction |
|------|-------------------|--------------|-----------|
| Shor [[9,1,3]] | 6 | N/A | — |
| Bacon-Shor [[9,1,4,3]] | 6 | 2 | 3× |
| [[25,1,16,5]] Bacon-Shor | 10 | 2 | 5× |
| 15-qubit Reed-Muller | 8 | N/A | — |

---

### 3. Fault Tolerance Benefits

#### What is Fault Tolerance?

A fault-tolerant operation ensures that:
- A single fault creates at most one error in each code block
- Errors don't propagate uncontrollably

**Problem with high-weight measurements:**

```
High-weight stabilizer measurement:
Data: ─●─●─●─●─●─●─
       │ │ │ │ │ │
Ancilla:├─┼─┼─┼─┼─┤─M

One ancilla error → can propagate to multiple data qubits!
```

#### Subsystem Code Advantage

**Low-weight gauge measurement:**

```
Weight-2 gauge measurement:
Data: ─●─●───────────
       │ │
Ancilla:├─┤─M

One error → propagates to at most 2 qubits
```

**Key benefit:** Fault path is shorter, error propagation limited.

#### Fault-Tolerant Syndrome Extraction

**For stabilizer codes:** Need **flag qubits** or **Shor-style ancilla** to catch hook errors

**For subsystem codes:** Natural fault tolerance from low-weight measurements

**Comparison:**

| Approach | Ancilla Qubits | Gate Count | Fault Tolerant |
|----------|----------------|------------|----------------|
| Direct stabilizer | 1 per stabilizer | $w$ CNOTs | No |
| Shor-style ancilla | $w$ per stabilizer | $2w$ CNOTs | Yes |
| Subsystem (gauge) | 1 per gauge | 2 CNOTs | Yes |

---

### 4. Single-Shot Error Correction

#### The Concept

**Single-shot error correction:** Correct errors using only ONE round of syndrome measurement (no repetition needed).

**Why is this remarkable?**

Standard QEC requires $O(d)$ syndrome measurement rounds to handle measurement errors.

Single-shot codes correct both:
1. Data errors (from noise on qubits)
2. Measurement errors (from noisy syndrome extraction)

...in a single round!

#### Requirements for Single-Shot

**Theorem (Bombin):** A subsystem code supports single-shot error correction if:

1. The gauge group has sufficient redundancy
2. Measurement errors cause localized syndrome errors
3. There exist "gauge fixing" recovery operations

#### Example: 3D Gauge Color Code

The 3D gauge color code is single-shot:
- Gauge operators are face operators
- Stabilizers are volume operators
- Measurement errors create syndrome chains
- Chains can be decoded in single round

**Single-shot codes:**
- 3D gauge color codes
- Certain LDPC subsystem codes
- Bacon-Shor variants (partial single-shot)

---

### 5. Gauge Freedom in Error Correction

#### Flexible Recovery

**Key insight:** Different gauge configurations encode the same logical state.

**Implication:** Recovery doesn't need to perfectly restore the pre-error state—only the logical information.

#### Gauge-Aware Decoding

**Standard stabilizer decoding:**
1. Get syndrome $s$
2. Find error $E$ with syndrome $s$
3. Apply correction $E^\dagger$

**Subsystem code decoding:**
1. Get gauge syndrome $\{g_i\}$
2. Compute stabilizer syndrome
3. Find recovery $R$ such that $R \cdot E \in \mathcal{G}$ (up to gauge)
4. Apply $R$

**More flexible!** Any recovery that brings us back to code space (mod gauge) works.

#### Syndrome Degeneracy

Many error patterns give same stabilizer syndrome but different gauge syndromes.

**Advantage:** Additional gauge information can improve decoding.

**Challenge:** Must track gauge or be gauge-agnostic.

---

### 6. Implementation Considerations

#### Hardware Requirements

**Connectivity:**
- Subsystem codes often need only local (2D) connectivity
- Bacon-Shor needs 2D grid with NN connections
- Compare: some stabilizer codes need long-range connections

**Gate counts:**

| Operation | Stabilizer Code | Subsystem Code |
|-----------|-----------------|----------------|
| Syndrome round | $O(nw)$ gates | $O(n \cdot w_g)$ gates |
| Total ancilla | $O(n/k)$ | $O(n/k \cdot \frac{w}{w_g})$ |

Where $w$ = stabilizer weight, $w_g$ = gauge weight.

#### Time Overhead

**Trade-off:** More measurements, but each is simpler.

| Aspect | Stabilizer | Subsystem |
|--------|------------|-----------|
| Measurements per round | $n-k$ | $n-k+r$ |
| Gates per measurement | $w$ | $w_g$ |
| Total gates | $(n-k) \cdot w$ | $(n-k+r) \cdot w_g$ |

Often $(n-k+r) \cdot w_g < (n-k) \cdot w$ due to $w_g \ll w$.

#### Decoding Complexity

**Subsystem decoding challenges:**
1. More syndrome bits to process
2. Must handle gauge degeneracy
3. Can exploit redundancy for reliability

**Practical decoders:**
- MWPM on derived stabilizer syndrome
- Belief propagation with gauge structure
- Union-find adapted for subsystem codes

---

### 7. When to Use Subsystem Codes

#### Prefer Subsystem Codes When:

1. **Hardware has limited connectivity**
   - 2D grids with nearest-neighbor
   - Superconducting qubit chips

2. **Fault tolerance is critical**
   - Medical/safety applications
   - Space-based quantum computers

3. **Measurement fidelity is low**
   - Single-shot capabilities help
   - Redundant gauge information

4. **Gate errors dominate**
   - Fewer gates = fewer errors

#### Prefer Stabilizer Codes When:

1. **Qubit resources are scarce**
   - Need maximum $k/n$ ratio
   - Gauge qubits are "wasted"

2. **High-fidelity measurements available**
   - Direct stabilizer measurement is fine
   - No need for weight reduction

3. **Simple decoding is needed**
   - Fewer syndromes to process
   - Standard algorithms apply directly

---

## Worked Examples

### Example 1: Measurement Circuit Comparison

**Problem:** Compare the syndrome extraction circuits for Shor [[9,1,3]] and Bacon-Shor [[9,1,4,3]] for the $X$-type stabilizer $X_1X_2X_3X_4X_5X_6$.

**Solution:**

**Shor code (direct):**
```
q1: ─●───────────────────
     │
q2: ─┼─●─────────────────
     │ │
q3: ─┼─┼─●───────────────
     │ │ │
q4: ─┼─┼─┼─●─────────────
     │ │ │ │
q5: ─┼─┼─┼─┼─●───────────
     │ │ │ │ │
q6: ─┼─┼─┼─┼─┼─●─────────
     │ │ │ │ │ │
anc: H─┴─┴─┴─┴─┴─┴─H─M
```
- 6 CNOT gates
- Single ancilla, 6 interactions

**Bacon-Shor (gauge-based):**

Measure $X_1X_2$:
```
q1: ─●─────
     │
q2: ─┼─●───
     │ │
a1: H─┴─┴─H─M
```

Measure $X_2X_3$:
```
q2: ─●─────
     │
q3: ─┼─●───
     │ │
a2: H─┴─┴─H─M
```

(Similarly for $X_4X_5$, $X_5X_6$)

- Each measurement: 2 CNOT gates
- Stabilizer syndrome: product of gauge syndromes
- Total: 4 measurements × 2 CNOTs = 8 CNOTs

**But:** Each circuit is fault-tolerant without flags!

---

### Example 2: Error Propagation Analysis

**Problem:** Analyze how a single CNOT fault propagates in (a) weight-6 measurement vs (b) weight-2 measurement.

**Solution:**

**(a) Weight-6 measurement:**

Fault: X error on ancilla after 3rd CNOT
```
q1: ─●─────────── → no error
q2: ─●─────────── → no error
q3: ─●─────────── → no error
q4: ─●─────────── → X error propagates!
q5: ──●────────── → X error propagates!
q6: ───●───────── → X error propagates!
anc: ──×─●─●─●─── (fault here)
```

**Result:** 3-qubit X error on data!

If distance is 3, this causes a **logical error**.

**(b) Weight-2 measurement:**

Fault: X error on ancilla after 1st CNOT
```
q1: ─●─── → no error
q2: ─●─── → X error propagates
anc: ─×─●─ (fault)
```

**Result:** 1-qubit X error on data.

**Conclusion:** Subsystem codes limit error propagation naturally.

---

### Example 3: Single-Shot Property Check

**Problem:** Explain why the Bacon-Shor code has partial single-shot capability for Z errors.

**Solution:**

**Z-error syndrome measurement:**
- Measure X-gauge operators (horizontal XX pairs)
- Each row has $(n-1)$ such measurements

**Key observation:** Z error on qubit $(i,j)$ affects:
- $X_{i,j-1}X_{i,j}$ (if $j > 1$)
- $X_{i,j}X_{i,j+1}$ (if $j < n$)

**Redundancy:** Multiple gauge measurements per data qubit.

**Measurement error resilience:**
- A measurement error on one gauge creates syndrome error
- But adjacent gauge measurements provide redundancy
- Can detect measurement errors without repetition

**Partial single-shot:** Works for certain error patterns but not all.

True single-shot requires 3D topology (like 3D gauge color code).

---

## Practice Problems

### Direct Application

1. **Problem 1:** For a weight-$w$ stabilizer measured via weight-2 gauge operators, how many gauge measurements are needed?

2. **Problem 2:** A subsystem code has 20 gauge generators and 8 stabilizers. How much syndrome redundancy exists?

3. **Problem 3:** Calculate the CNOT count for syndrome extraction in a $4 \times 4$ Bacon-Shor code (all stabilizers).

### Intermediate

4. **Problem 4:** Design a fault-tolerant syndrome extraction circuit for the $[[4,1,1,2]]$ code.

5. **Problem 5:** Prove that if all gauge operators have weight 2, any single fault in measurement creates at most a weight-2 data error.

6. **Problem 6:** Compare the threshold error rates for direct stabilizer measurement vs gauge-based measurement (qualitative analysis).

### Challenging

7. **Problem 7:** Prove that the 2D Bacon-Shor code cannot achieve true single-shot error correction.

8. **Problem 8:** Design a gauge measurement schedule that minimizes circuit depth for the $3 \times 3$ Bacon-Shor code.

9. **Problem 9:** Analyze the trade-off between gauge measurement redundancy and decoding complexity.

---

## Computational Lab

```python
"""
Day 719: Advantages of Subsystem Codes
Comparative analysis of measurement circuits and fault tolerance
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class MeasurementCircuit:
    """Represents a syndrome measurement circuit."""
    name: str
    num_data_qubits: int
    num_ancilla: int
    cnot_count: int
    max_error_spread: int  # Max data qubits affected by single fault
    is_fault_tolerant: bool

def analyze_stabilizer_measurement(weight: int) -> MeasurementCircuit:
    """Analyze direct stabilizer measurement."""
    return MeasurementCircuit(
        name=f"Direct weight-{weight}",
        num_data_qubits=weight,
        num_ancilla=1,
        cnot_count=weight,
        max_error_spread=weight - 1,  # Fault after k-th CNOT affects w-k qubits
        is_fault_tolerant=False
    )

def analyze_shor_ancilla(weight: int) -> MeasurementCircuit:
    """Analyze Shor-style cat state ancilla measurement."""
    return MeasurementCircuit(
        name=f"Shor ancilla weight-{weight}",
        num_data_qubits=weight,
        num_ancilla=weight,  # Cat state needs w ancilla
        cnot_count=2 * weight - 1,  # Prepare cat + measure
        max_error_spread=1,
        is_fault_tolerant=True
    )

def analyze_gauge_measurement(stabilizer_weight: int, gauge_weight: int = 2) -> MeasurementCircuit:
    """Analyze gauge-based subsystem code measurement."""
    num_gauges = stabilizer_weight // gauge_weight  # Approximate
    return MeasurementCircuit(
        name=f"Gauge (w={gauge_weight})",
        num_data_qubits=stabilizer_weight,
        num_ancilla=num_gauges,
        cnot_count=num_gauges * gauge_weight,
        max_error_spread=gauge_weight - 1,
        is_fault_tolerant=True
    )

def compare_approaches(weights: List[int]) -> Dict:
    """Compare measurement approaches for various weights."""
    results = {'direct': [], 'shor': [], 'gauge': []}

    for w in weights:
        results['direct'].append(analyze_stabilizer_measurement(w))
        results['shor'].append(analyze_shor_ancilla(w))
        results['gauge'].append(analyze_gauge_measurement(w))

    return results

def bacon_shor_syndrome_cost(m: int, n: int) -> Dict:
    """Calculate syndrome extraction cost for m×n Bacon-Shor."""

    # Gauge operators
    x_gauges = m * (n - 1)  # Horizontal XX
    z_gauges = (m - 1) * n  # Vertical ZZ
    total_gauges = x_gauges + z_gauges

    # Stabilizers
    x_stabilizers = n - 1
    z_stabilizers = m - 1
    total_stabilizers = x_stabilizers + z_stabilizers

    # As stabilizer code (direct measurement)
    x_stab_weight = 2 * m  # Each X-stabilizer spans 2 columns, all m rows
    z_stab_weight = 2 * n  # Each Z-stabilizer spans 2 rows, all n columns

    direct_cnots = x_stabilizers * x_stab_weight + z_stabilizers * z_stab_weight

    # As subsystem code (gauge measurement)
    gauge_cnots = total_gauges * 2  # Each gauge is weight-2

    return {
        'm': m, 'n': n,
        'num_qubits': m * n,
        'x_gauges': x_gauges,
        'z_gauges': z_gauges,
        'total_gauges': total_gauges,
        'stabilizers': total_stabilizers,
        'direct_cnots': direct_cnots,
        'gauge_cnots': gauge_cnots,
        'cnot_reduction': direct_cnots / gauge_cnots if gauge_cnots > 0 else 0,
        'max_error_spread_direct': max(x_stab_weight, z_stab_weight) - 1,
        'max_error_spread_gauge': 1
    }

def plot_comparison(weights: List[int], results: Dict):
    """Plot comparison of measurement approaches."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # CNOT count comparison
    ax1 = axes[0]
    ax1.plot(weights, [r.cnot_count for r in results['direct']], 'b-o', label='Direct')
    ax1.plot(weights, [r.cnot_count for r in results['shor']], 'r-s', label='Shor ancilla')
    ax1.plot(weights, [r.cnot_count for r in results['gauge']], 'g-^', label='Gauge-based')
    ax1.set_xlabel('Stabilizer Weight')
    ax1.set_ylabel('CNOT Count')
    ax1.set_title('Gate Count Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ancilla count comparison
    ax2 = axes[1]
    ax2.plot(weights, [r.num_ancilla for r in results['direct']], 'b-o', label='Direct')
    ax2.plot(weights, [r.num_ancilla for r in results['shor']], 'r-s', label='Shor ancilla')
    ax2.plot(weights, [r.num_ancilla for r in results['gauge']], 'g-^', label='Gauge-based')
    ax2.set_xlabel('Stabilizer Weight')
    ax2.set_ylabel('Ancilla Qubits')
    ax2.set_title('Ancilla Requirements')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Error spread comparison
    ax3 = axes[2]
    ax3.plot(weights, [r.max_error_spread for r in results['direct']], 'b-o', label='Direct')
    ax3.plot(weights, [r.max_error_spread for r in results['shor']], 'r-s', label='Shor ancilla')
    ax3.plot(weights, [r.max_error_spread for r in results['gauge']], 'g-^', label='Gauge-based')
    ax3.set_xlabel('Stabilizer Weight')
    ax3.set_ylabel('Max Error Spread')
    ax3.set_title('Fault Propagation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='FT threshold')

    plt.tight_layout()
    plt.savefig('measurement_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: measurement_comparison.png")

def plot_bacon_shor_scaling():
    """Plot Bacon-Shor scaling analysis."""
    sizes = range(2, 8)
    data = [bacon_shor_syndrome_cost(m, m) for m in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # CNOT scaling
    ax1 = axes[0]
    direct = [d['direct_cnots'] for d in data]
    gauge = [d['gauge_cnots'] for d in data]
    qubits = [d['num_qubits'] for d in data]

    ax1.plot(qubits, direct, 'b-o', label='Direct stabilizer', linewidth=2)
    ax1.plot(qubits, gauge, 'g-^', label='Gauge-based', linewidth=2)
    ax1.set_xlabel('Number of Qubits (m×m)')
    ax1.set_ylabel('Total CNOTs per Syndrome Round')
    ax1.set_title('Bacon-Shor: CNOT Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reduction factor
    ax2 = axes[1]
    reduction = [d['cnot_reduction'] for d in data]
    ax2.bar(qubits, reduction, color='green', alpha=0.7)
    ax2.set_xlabel('Number of Qubits (m×m)')
    ax2.set_ylabel('CNOT Reduction Factor')
    ax2.set_title('Subsystem Code Advantage')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bacon_shor_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: bacon_shor_scaling.png")

# Main analysis
print("=" * 60)
print("Subsystem Code Advantages Analysis")
print("=" * 60)

# Example 1: Measurement approach comparison
print("\n1. Measurement Approach Comparison")
print("-" * 40)

weights = [4, 6, 8, 10, 12]
results = compare_approaches(weights)

print(f"\n{'Weight':<8} {'Approach':<20} {'CNOTs':<8} {'Ancilla':<10} {'Max Spread':<12} {'FT?'}")
print("-" * 70)

for i, w in enumerate(weights):
    for approach, name in [('direct', 'Direct'), ('shor', 'Shor'), ('gauge', 'Gauge')]:
        r = results[approach][i]
        print(f"{w:<8} {name:<20} {r.cnot_count:<8} {r.num_ancilla:<10} {r.max_error_spread:<12} {r.is_fault_tolerant}")
    print()

# Example 2: Bacon-Shor specific analysis
print("\n2. Bacon-Shor Code Analysis")
print("-" * 40)

print(f"\n{'Size':<8} {'Qubits':<8} {'Gauges':<8} {'Direct CNOTs':<14} {'Gauge CNOTs':<12} {'Reduction'}")
print("-" * 65)

for m in [2, 3, 4, 5, 6]:
    data = bacon_shor_syndrome_cost(m, m)
    print(f"{m}×{m:<6} {data['num_qubits']:<8} {data['total_gauges']:<8} "
          f"{data['direct_cnots']:<14} {data['gauge_cnots']:<12} {data['cnot_reduction']:.1f}×")

# Example 3: Error propagation simulation
print("\n3. Error Propagation Simulation")
print("-" * 40)

def simulate_fault_propagation(measurement_type: str, weight: int, fault_position: int):
    """Simulate error propagation from a single CNOT fault."""
    if measurement_type == 'direct':
        # Fault after position k affects qubits k+1 to w
        affected = list(range(fault_position + 1, weight + 1))
    elif measurement_type == 'gauge':
        # Fault in weight-2 measurement affects at most 1 qubit
        affected = [fault_position + 1] if fault_position < 1 else []
    else:
        affected = []
    return affected

print("\nDirect weight-6 measurement fault analysis:")
for pos in range(6):
    affected = simulate_fault_propagation('direct', 6, pos)
    print(f"  Fault after CNOT {pos+1}: affects {len(affected)} data qubit(s) {affected}")

print("\nGauge weight-2 measurement fault analysis:")
for pos in range(2):
    affected = simulate_fault_propagation('gauge', 2, pos)
    print(f"  Fault after CNOT {pos+1}: affects {len(affected)} data qubit(s) {affected}")

# Example 4: Threshold comparison (qualitative)
print("\n4. Threshold Considerations")
print("-" * 40)

print("""
Error threshold comparison (approximate):

| Approach           | Typical Threshold | Notes                    |
|--------------------|-------------------|--------------------------|
| Direct (weight-6)  | ~0.1%             | Hook errors dominate     |
| Shor ancilla       | ~0.5%             | More ancilla overhead    |
| Gauge-based        | ~1%               | Natural fault tolerance  |

Note: Actual thresholds depend on specific code, decoder, and noise model.
""")

# Example 5: Resource comparison summary
print("\n5. Resource Comparison Summary")
print("-" * 40)

print("""
Trade-off Summary for n-qubit codes:

| Resource          | Stabilizer Code    | Subsystem Code        |
|-------------------|--------------------|-----------------------|
| Logical qubits    | k                  | k' < k                |
| Gauge qubits      | 0                  | r                     |
| Measurement weight| w (can be large)   | w_g (typically 2)     |
| Measurements/round| n - k              | n - k' + r (more)     |
| Fault tolerance   | Requires flags     | Natural               |
| Decoding          | Standard           | Gauge-aware           |
""")

# Generate plots
print("\n6. Generating Comparison Plots...")
plot_comparison(weights, results)
plot_bacon_shor_scaling()

# Example 6: Single-shot analysis
print("\n7. Single-Shot Error Correction")
print("-" * 40)

print("""
Single-shot capability comparison:

| Code Family          | Single-Shot? | Reason                         |
|----------------------|--------------|--------------------------------|
| 2D Surface Code      | No           | Requires O(d) rounds           |
| 2D Bacon-Shor        | Partial      | Z-errors only, limited         |
| 3D Gauge Color Code  | Yes          | Sufficient gauge redundancy    |
| 3D Toric Code        | No           | Still requires repetition      |
| Hyperbolic codes     | Yes (some)   | High connectivity helps        |

Single-shot requires:
1. Sufficient syndrome redundancy
2. Confined measurement error effects
3. Polynomial-time gauge fixing decoder
""")

print("\n" + "=" * 60)
print("Analysis complete!")
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Weight reduction** | Gauge ops enable lower-weight measurements |
| **Fault tolerance** | Limited error propagation with small gauges |
| **Single-shot** | Error correction in one measurement round |
| **Resource trade-off** | More measurements vs simpler circuits |
| **Gauge decoding** | Flexible recovery via gauge freedom |

### Comparison Table

| Property | Stabilizer Code | Subsystem Code |
|----------|-----------------|----------------|
| Measurement weight | $w$ (arbitrary) | $w_g$ (small) |
| Fault tolerance | Needs flags | Natural |
| Measurements/round | $n-k$ | $n-k+r$ |
| Logical qubits | $k$ | $k' < k$ |
| Decoder complexity | Standard | Gauge-aware |

### Main Takeaways

1. **Weight reduction** is the primary practical advantage of subsystem codes
2. **Fault tolerance** comes naturally from low-weight gauge measurements
3. **Single-shot** capability possible with sufficient gauge redundancy (3D codes)
4. **Trade-off:** More measurements but each is simpler and safer
5. **Best for:** Limited connectivity, high fault-tolerance requirements

---

## Daily Checklist

- [ ] I understand why low-weight measurements matter
- [ ] I can explain fault-tolerance benefits of gauge operators
- [ ] I understand single-shot error correction requirements
- [ ] I can compare measurement circuits quantitatively
- [ ] I know when to prefer subsystem vs stabilizer codes
- [ ] I completed the computational lab

---

## Preview: Day 720

Tomorrow we study **Subsystem Codes and Fault Tolerance**, including:
- Fault-tolerant syndrome extraction protocols
- Transversal gates on subsystem codes
- Gauge fixing for universal computation
- The role of subsystem codes in fault-tolerant architectures
