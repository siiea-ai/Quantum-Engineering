# Day 877: The Flag Qubit Concept — Detecting Dangerous Errors

## Month 32: Fault-Tolerant Quantum Computing II | Week 126: Flag Qubits & Syndrome Extraction

---

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Flag Qubit Fundamentals |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving: Flag Patterns |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of Day 877, you will be able to:

1. Define flag qubits and explain their role in fault-tolerant syndrome extraction
2. Identify dangerous error patterns in syndrome extraction circuits
3. Explain how flag qubits detect high-weight error propagation
4. Distinguish between flagged and unflagged error correction
5. Analyze flag patterns for different fault locations
6. State the fault-tolerance conditions for flag-based protocols

---

## Core Content

### 1. The Flag Qubit Paradigm Shift

Yesterday we saw that traditional syndrome extraction has high ancilla overhead: $O(w)$ qubits per weight-$w$ stabilizer. The flag qubit approach fundamentally changes our strategy:

**Traditional Approach (Shor-style):**
> "Prevent high-weight errors from ever occurring."

**Flag Qubit Approach (Chao-Reichardt):**
> "Allow high-weight errors, but detect when they occur."

$$\boxed{\text{Flag qubit: Detects when single fault} \to \text{high-weight error}}$$

**The Key Insight:**

A single fault in a syndrome extraction circuit can cause a high-weight data error. But:
1. Not all faults cause high-weight errors
2. Faults that do cause high-weight errors have a specific *pattern*
3. A carefully placed "flag" qubit can detect this pattern

If the flag triggers, we know a potentially dangerous error occurred and can handle it appropriately.

---

### 2. Anatomy of a Flag Circuit

Consider measuring the stabilizer $S = Z_1 Z_2 Z_3 Z_4$ (weight-4).

**Naive Circuit (Not Fault-Tolerant):**
```
Syndrome |+⟩ ───●───●───●───●───M_X
                │   │   │   │
Data q₁  ──────Z───┼───┼───┼─────
Data q₂  ──────────Z───┼───┼─────
Data q₃  ────────────Z───┼─────
Data q₄  ──────────────Z─────
```

**Flag Circuit:**
```
Flag     |0⟩ ─────────────●───●───────────M_Z ──→ f
                          │   │
Syndrome |+⟩ ───●───●─────X───X───●───●───M_X ──→ s
                │   │             │   │
Data q₁  ──────Z───┼─────────────┼───┼─────
Data q₂  ──────────Z─────────────┼───┼─────
Data q₃  ────────────────────────Z───┼─────
Data q₄  ──────────────────────────Z─────
```

**The flag qubit is placed to detect dangerous fault locations!**

---

### 3. How Flags Detect Dangerous Errors

**Dangerous Fault Definition:**

A fault is *dangerous* for a distance-$d$ code if it causes a data error of weight $\geq t + 1$ where $t = \lfloor(d-1)/2\rfloor$.

For distance-3 codes ($t = 1$): Weight-2 or higher errors are dangerous.

**Error Propagation Analysis:**

In the naive circuit for $Z_1 Z_2 Z_3 Z_4$:

| Fault Location | Resulting Data Error | Weight | Dangerous? |
|----------------|---------------------|--------|------------|
| X on syndrome before CNOT 1 | $Z_1 Z_2 Z_3 Z_4$ | 4 | Yes |
| X on syndrome after CNOT 1 | $Z_2 Z_3 Z_4$ | 3 | Yes |
| X on syndrome after CNOT 2 | $Z_3 Z_4$ | 2 | Yes |
| X on syndrome after CNOT 3 | $Z_4$ | 1 | No |
| X on syndrome after CNOT 4 | None | 0 | No |

**Flag Placement Strategy:**

The flag qubit is connected to the syndrome qubit at a position that divides the CNOT chain. If a fault occurs *before* the flag connection, it will flip the flag.

**In the flag circuit above:**

- Flag connects after CNOTs to $q_1, q_2$ and before CNOTs to $q_3, q_4$
- X error before flag connection: Propagates through flag CNOTs → flag flips
- X error after flag connection: Doesn't affect flag → flag stays 0

---

### 4. Flag Pattern Analysis

**Formal Definition:**

A **flag qubit** $f$ is an ancilla that:
1. Starts in $|0\rangle$
2. Is entangled with the syndrome qubit at specific locations
3. Is measured in the Z basis
4. Outcome $f = 1$ indicates a potentially dangerous fault occurred

**Flag Patterns for Weight-4 Stabilizer:**

| Fault | Syndrome Outcome | Flag Outcome | Data Error | Action |
|-------|------------------|--------------|------------|--------|
| No fault | $s$ (correct) | 0 | None | Normal correction |
| X before flag | $s \oplus 1$ | 1 | Weight ≥ 2 | Special handling |
| X after flag | $s \oplus 1$ | 0 | Weight ≤ 1 | Normal correction |
| Z on flag | $s$ | 1 | None | Ignore (false flag) |

**The Critical Insight:**

When flag = 1, we know either:
1. A dangerous fault occurred (needs special handling), OR
2. A benign Z error on the flag occurred (no data damage)

Both cases are handled correctly by the flag-FT protocol!

---

### 5. Mathematical Framework

**Fault Path Analysis:**

Let $\mathcal{F}$ be the set of fault locations and $E(\mathcal{F})$ be the resulting data error.

**Definition (t-Flag Circuit):**

A syndrome extraction circuit with flag is **t-flag fault-tolerant** if:
- Any single fault either:
  - Causes data error of weight ≤ $t$, OR
  - Triggers the flag (flag = 1)

Mathematically:

$$\boxed{\forall f \in \mathcal{F}_{\text{single}}: \quad \text{wt}(E(f)) \leq t \;\lor\; \text{Flag}(f) = 1}$$

**Theorem (Chao-Reichardt 2018):**

For any stabilizer code of distance $d \geq 3$, there exists a syndrome extraction circuit using only **2 ancilla qubits** (1 syndrome + 1 flag) that is 1-flag fault-tolerant.

---

### 6. Single-Flag vs. Multi-Flag Circuits

**Single-Flag Circuit:**

For weight-$w$ stabilizers with $w \leq 4$, a single flag qubit suffices:

```
Flag     |0⟩ ───────●───●───────M_Z
                    │   │
Syndrome |+⟩ ───●───X───X───●───M_X
                │           │
Data q₁  ──────Z───────────┼─────
Data q₂  ──────────────────Z─────
```

**Multi-Flag Circuits:**

For higher-weight stabilizers or stricter fault tolerance, multiple flags may be needed:

```
Flag 1   |0⟩ ─────●───●───────────────M_Z
                  │   │
Flag 2   |0⟩ ─────┼───┼───●───●───────M_Z
                  │   │   │   │
Syndrome |+⟩ ──●──X───X───X───X───●───M_X
               │                   │
Data q₁  ─────Z───────────────────┼───
    ⋮                              ⋮
Data qw  ─────────────────────────Z───
```

**Trade-off:**

| Configuration | Flags | Overhead | Fault Detection |
|---------------|-------|----------|-----------------|
| Single flag | 1 | Minimal | Weight-2 errors |
| Two flags | 2 | Low | Weight-3 errors |
| Full cat (Shor) | $w-1$ | High | All single faults |

---

### 7. Fault-Tolerance Conditions

**Condition 1: Error Detection**

For any single fault $f$, either:
- $\text{wt}(E(f)) \leq t$, or
- $\text{Flag}(f) = 1$

**Condition 2: Distinguishability**

Different error patterns must have distinguishable (syndrome, flag) pairs:

$$E_1 \neq E_2 \Rightarrow (s_1, f_1) \neq (s_2, f_2)$$

**Condition 3: Correctability**

Given (syndrome, flag), there exists a unique correction that maps back to the code space:

$$\text{Lookup}(s, f) \to \text{Correction operator } R$$

**The Flag-FT Protocol:**

1. Extract syndrome with flag circuit
2. If flag = 0: Apply standard decoder
3. If flag = 1: Apply flag-aware decoder (handles high-weight possibilities)

---

### 8. Why Flags Work: Information-Theoretic View

**Without Flags:**

The syndrome alone doesn't distinguish between:
- Low-weight error (correctable)
- High-weight error from fault (uncorrectable)

**With Flags:**

The (syndrome, flag) pair provides additional information:

$$\boxed{I(\text{error} ; \text{syndrome}, \text{flag}) > I(\text{error} ; \text{syndrome})}$$

The flag carries information about *where* in the circuit a fault occurred, which correlates with the *weight* of the resulting error.

---

## Practical Applications

### Hardware Implementation Considerations

**Mid-Circuit Measurement:**

Flag circuits require measuring the flag qubit *before* the final syndrome measurement. This needs:
- Fast qubit measurement
- Classical feedforward capability
- Low measurement error rates

**Current Hardware Support:**

| Platform | Mid-Circuit Measurement | Flag Circuit Feasibility |
|----------|------------------------|--------------------------|
| Superconducting (IBM, Google) | Yes | Demonstrated |
| Trapped Ions (IonQ, Quantinuum) | Yes | Optimal |
| Neutral Atoms | Limited | Developing |
| Photonic | Inherent | Natural fit |

### Resource Savings

For the [[7,1,3]] Steane code:

| Method | Ancilla Qubits | Total Qubits |
|--------|----------------|--------------|
| Shor-style | 36 | 43 |
| Steane-style | 7 | 14 |
| Flag-based | 2 | 9 |

$$\boxed{\text{Flag savings: } \frac{36 - 2}{36} = 94\% \text{ reduction in ancillas}}$$

---

## Worked Examples

### Example 1: Flag Circuit for Weight-4 Stabilizer

**Problem:** Design a flag circuit for measuring $S = X_1 X_2 X_3 X_4$ and analyze all single faults.

**Solution:**

**Circuit Design:**
```
Flag     |0⟩ ─────────────●───●───────────M_Z ──→ f
                          │   │
Syndrome |0⟩ ───H───●───●─X───X─●───●───H─M_Z ──→ s
                    │   │       │   │
Data q₁  ──────────X───┼───────┼───┼─────────
Data q₂  ──────────────X───────┼───┼─────────
Data q₃  ──────────────────────X───┼─────────
Data q₄  ──────────────────────────X─────────
```

**Fault Analysis:**

| Fault Location | Data Error | Weight | Flag | Dangerous? |
|----------------|------------|--------|------|------------|
| Z on syndrome before CNOT 1 | None | 0 | 0 | No |
| Z on syndrome after CNOT 1 | $X_1$ | 1 | 0 | No |
| Z on syndrome after CNOT 2, before flag | $X_1 X_2$ | 2 | 1 | Flagged! |
| Z on syndrome after flag, before CNOT 3 | $X_1 X_2$ | 2 | 1 | Flagged! |
| Z on syndrome after CNOT 3 | $X_1 X_2 X_3$ | 3 | 0 | Wait... |

**Issue:** The last case has weight-3 error but flag = 0!

**Solution:** Reorder CNOTs so flag divides evenly:

```
Flag     |0⟩ ───────────●───●─────────────M_Z
                        │   │
Syndrome |0⟩ ──H──●──●──X───X──●──●──H────M_Z
                  │  │         │  │
Data q₁  ────────X──┼─────────┼──┼────────
Data q₂  ───────────X─────────┼──┼────────
Data q₃  ─────────────────────X──┼────────
Data q₄  ────────────────────────X────────
```

Now Z errors on syndrome:
- Before flag: Affects $q_1, q_2$ → flag = 1
- After flag: Affects only $q_3, q_4$ → weight ≤ 2, handled normally

**Result:** Single flag suffices when CNOT order is chosen carefully. $\square$

---

### Example 2: Analyzing Flag Outcomes

**Problem:** For the circuit in Example 1, enumerate all (syndrome, flag) outcomes and their interpretations.

**Solution:**

Assuming the data is in a code state (syndrome should be +1 without errors):

| Syndrome $s$ | Flag $f$ | Interpretation | Action |
|--------------|----------|----------------|--------|
| +1 | 0 | No error | None |
| +1 | 1 | Z error on flag (benign) | Ignore |
| -1 | 0 | Low-weight X error ($q_3$ or $q_4$) | Standard decode |
| -1 | 1 | Possibly high-weight X error | Flag decode |

**Flag Decode (when $s = -1, f = 1$):**

The error could be:
1. Single X on $q_1$ or $q_2$ (weight-1)
2. Circuit fault causing $X_1 X_2$ (weight-2)

**Distinguish via additional syndrome measurements!**

The flag protocol typically requires multiple rounds:
1. First round with flag
2. If flagged, second round with different circuit
3. Combined information uniquely identifies error

**Result:** Flag = 1 triggers a more careful decoding procedure. $\square$

---

### Example 3: Proving Fault Tolerance

**Problem:** Prove that the flag circuit for weight-4 Z stabilizer is 1-flag fault-tolerant.

**Solution:**

**1-Flag Fault Tolerance Condition:**
Every single fault either:
- Causes weight ≤ 1 data error, OR
- Triggers the flag

**Enumerate all single fault locations:**

**Faults on Data Qubits:**
- Any single Pauli error on data: Weight 1, no flag needed ✓

**Faults on Syndrome Qubit:**
- $X$ before CNOT 1: Causes $Z_1 Z_2 Z_3 Z_4$ → Weight 4, but flag = 1 ✓
- $X$ after CNOT 1, before CNOT 2: Causes $Z_2 Z_3 Z_4$ → Weight 3, flag = 1 ✓
- $X$ after CNOT 2, before flag: Causes $Z_3 Z_4$ → Weight 2, flag = 1 ✓
- $X$ after flag, before CNOT 3: Causes $Z_3 Z_4$ → Weight 2, flag = 1 ✓
- $X$ after CNOT 3, before CNOT 4: Causes $Z_4$ → Weight 1 ✓
- $X$ after CNOT 4: No effect → Weight 0 ✓
- $Z$ anywhere: Only affects syndrome, not data ✓

**Faults on Flag Qubit:**
- $X$ on flag: Propagates to syndrome as $Z$, no data error ✓
- $Z$ on flag: Flips flag outcome but no data error ✓

**Faults on CNOT Gates:**
- Each CNOT fault can be decomposed into before/after single-qubit errors
- Analysis follows from above cases ✓

**Conclusion:** All single faults satisfy the 1-flag FT condition. $\square$

---

## Practice Problems

### Level 1: Direct Application

1. **Flag placement:** For a weight-6 stabilizer, where should a single flag be placed to balance the CNOT chain?

2. **Fault enumeration:** List all single X fault locations in a flag circuit for $Z_1 Z_2 Z_3$ and determine which trigger the flag.

3. **Resource counting:** Compare ancilla counts for Shor-style vs. flag-based extraction for a weight-5 stabilizer.

### Level 2: Intermediate

4. **Multi-flag design:** Design a two-flag circuit for a weight-6 stabilizer. Verify that all weight-3 errors from single faults trigger at least one flag.

5. **Syndrome-flag table:** Construct the complete (syndrome, flag) lookup table for a flag circuit measuring $X_1 X_2 X_3 X_4$ on an initial $|++++\rangle$ state with possible single-qubit errors.

6. **False positives:** Calculate the probability that the flag triggers even though no dangerous error occurred (false positive rate) as a function of physical error rate $p$.

### Level 3: Challenging

7. **Optimal ordering:** Prove that for a single flag on a weight-$w$ stabilizer, the optimal flag placement minimizes $\max(k, w-k)$ where $k$ is the number of CNOTs before the flag.

8. **Distance-5 extension:** Extend the flag concept to distance-5 codes. What is the maximum weight error that must be detected? How many flags are needed?

9. **Research problem:** Read Chamberland & Beverland (2018). How do they generalize flag circuits to arbitrary distance codes?

---

## Computational Lab

### Objective
Simulate flag circuits and verify fault-tolerance properties.

```python
"""
Day 877 Computational Lab: Flag Qubit Concept
Week 126: Flag Qubits & Syndrome Extraction
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Flag Circuit Representation
# =============================================================================

print("=" * 70)
print("Part 1: Flag Circuit Structure")
print("=" * 70)

class FlagCircuit:
    """Represents a flag circuit for stabilizer measurement."""

    def __init__(self, stabilizer_weight, flag_position):
        """
        Args:
            stabilizer_weight: Number of qubits in stabilizer
            flag_position: After how many CNOTs to place flag (0 to w)
        """
        self.w = stabilizer_weight
        self.flag_pos = flag_position

    def analyze_x_fault(self, fault_position):
        """
        Analyze effect of X fault on syndrome qubit.

        Args:
            fault_position: After which CNOT (0 = before any, w = after all)

        Returns:
            (data_error_weight, flag_triggered)
        """
        # X error propagates Z to remaining data qubits
        data_error_weight = self.w - fault_position

        # Flag triggers if fault is before flag position
        flag_triggered = fault_position < self.flag_pos

        return data_error_weight, flag_triggered

    def is_fault_tolerant(self, t=1):
        """Check if circuit is t-flag fault tolerant."""
        for pos in range(self.w + 1):
            weight, flag = self.analyze_x_fault(pos)
            if weight > t and not flag:
                return False
        return True

# Test flag circuits
print("\nAnalyzing flag placement for weight-4 stabilizer:")
print("-" * 60)

for flag_pos in range(5):
    circuit = FlagCircuit(4, flag_pos)
    ft = circuit.is_fault_tolerant(t=1)
    print(f"\nFlag after CNOT {flag_pos}: {'FT' if ft else 'NOT FT'}")
    print("  Fault pos | Error wt | Flag | Status")
    print("  " + "-" * 40)
    for pos in range(5):
        wt, flag = circuit.analyze_x_fault(pos)
        status = "OK" if wt <= 1 or flag else "FAIL"
        print(f"  {pos:^9} | {wt:^8} | {'Y' if flag else 'N':^4} | {status}")

# =============================================================================
# Part 2: Optimal Flag Placement
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Optimal Flag Placement")
print("=" * 70)

def find_optimal_flag_positions(weight, t=1):
    """Find all flag positions that achieve t-flag FT."""
    optimal = []
    for pos in range(weight + 1):
        circuit = FlagCircuit(weight, pos)
        if circuit.is_fault_tolerant(t):
            optimal.append(pos)
    return optimal

print("\nOptimal single-flag positions by stabilizer weight:")
print("-" * 50)

for w in range(2, 9):
    optimal = find_optimal_flag_positions(w, t=1)
    if optimal:
        print(f"Weight {w}: Flag after CNOTs {optimal}")
        # The condition: flag_pos >= w - t
        # For t=1: flag_pos >= w - 1
    else:
        print(f"Weight {w}: No single flag achieves t=1 FT")

# =============================================================================
# Part 3: Multi-Flag Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Multi-Flag Circuits")
print("=" * 70)

class MultiFlagCircuit:
    """Represents a circuit with multiple flags."""

    def __init__(self, stabilizer_weight, flag_positions):
        """
        Args:
            stabilizer_weight: Number of qubits in stabilizer
            flag_positions: List of positions where flags are placed
        """
        self.w = stabilizer_weight
        self.flag_positions = sorted(flag_positions)

    def analyze_x_fault(self, fault_position):
        """
        Analyze effect of X fault on syndrome qubit.

        Returns:
            (data_error_weight, list of triggered flags)
        """
        data_error_weight = self.w - fault_position

        triggered = []
        for i, fpos in enumerate(self.flag_positions):
            if fault_position < fpos:
                triggered.append(i)

        return data_error_weight, triggered

    def is_fault_tolerant(self, t):
        """Check if circuit is t-flag fault tolerant."""
        for pos in range(self.w + 1):
            weight, flags = self.analyze_x_fault(pos)
            if weight > t and len(flags) == 0:
                return False
        return True

# Analyze multi-flag circuits
print("\nTwo-flag circuit for weight-6 stabilizer:")
print("-" * 60)

# Try different two-flag placements
best_config = None
for f1 in range(1, 6):
    for f2 in range(f1 + 1, 7):
        circuit = MultiFlagCircuit(6, [f1, f2])
        if circuit.is_fault_tolerant(t=1):
            print(f"Flags at {f1}, {f2}: FT for t=1")
            if circuit.is_fault_tolerant(t=2):
                print(f"  Also FT for t=2!")
                best_config = (f1, f2)

# =============================================================================
# Part 4: Syndrome-Flag Lookup Table
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: Syndrome-Flag Lookup Table")
print("=" * 70)

def build_lookup_table(stabilizer_weight, flag_position, t=1):
    """
    Build lookup table for (syndrome, flag) -> correction.

    Assumes Z-type stabilizer and X errors on data.
    """
    circuit = FlagCircuit(stabilizer_weight, flag_position)
    w = stabilizer_weight

    table = {}

    # No error case
    table[('+1', 0)] = "No correction"

    # Single data qubit errors (X_i causes syndrome flip)
    for i in range(w):
        # Single X error on qubit i
        table[('-1', 0)] = f"Single X error (decode normally)"

    # Circuit faults causing high-weight errors
    table[('-1', 1)] = "Possible high-weight error (flag decode)"
    table[('+1', 1)] = "Z error on flag (benign)"

    return table

print("\nLookup table for weight-4 Z stabilizer with flag at position 2:")
print("-" * 60)

lookup = build_lookup_table(4, 2)
print(f"{'(Syndrome, Flag)':<25} {'Action'}")
print("-" * 60)
for key, action in lookup.items():
    print(f"{str(key):<25} {action}")

# =============================================================================
# Part 5: Error Rate Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Error Rate Analysis")
print("=" * 70)

def simulate_flag_circuit(weight, flag_pos, p_error, n_trials=100000):
    """
    Simulate flag circuit with random errors.

    Args:
        weight: Stabilizer weight
        flag_pos: Flag position
        p_error: Probability of X error at each location
        n_trials: Number of simulation trials

    Returns:
        Dictionary of outcome statistics
    """
    results = {
        'no_error': 0,
        'low_weight_unflagged': 0,
        'high_weight_flagged': 0,
        'low_weight_flagged': 0,  # False positive
        'high_weight_unflagged': 0  # Failure!
    }

    circuit = FlagCircuit(weight, flag_pos)

    for _ in range(n_trials):
        # Random X errors on syndrome qubit at each position
        errors = np.random.random(weight + 1) < p_error
        n_errors = np.sum(errors)

        if n_errors == 0:
            results['no_error'] += 1
        elif n_errors == 1:
            pos = np.where(errors)[0][0]
            wt, flag = circuit.analyze_x_fault(pos)

            if wt <= 1:
                if flag:
                    results['low_weight_flagged'] += 1
                else:
                    results['low_weight_unflagged'] += 1
            else:
                if flag:
                    results['high_weight_flagged'] += 1
                else:
                    results['high_weight_unflagged'] += 1
        else:
            # Multiple errors - complex analysis, skip for now
            pass

    return {k: v/n_trials for k, v in results.items()}

# Run simulation
print("\nSimulating flag circuit (weight-4, flag at position 2):")
print("-" * 70)

p_values = [0.001, 0.005, 0.01, 0.02, 0.05]
print(f"{'p_error':<10} {'No Error':<12} {'Low-Unflag':<12} {'High-Flag':<12} {'Failure'}")
print("-" * 70)

for p in p_values:
    stats = simulate_flag_circuit(4, 2, p)
    print(f"{p:<10.3f} {stats['no_error']:<12.4f} {stats['low_weight_unflagged']:<12.4f} "
          f"{stats['high_weight_flagged']:<12.4f} {stats['high_weight_unflagged']:.6f}")

# =============================================================================
# Part 6: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Flag FT regions for different weights
ax1 = axes[0, 0]
weights = range(2, 9)
for w in weights:
    circuit = FlagCircuit(w, w - 1)  # Optimal single flag position
    positions = range(w + 1)
    is_flagged = [circuit.analyze_x_fault(p)[1] for p in positions]
    error_weights = [circuit.analyze_x_fault(p)[0] for p in positions]

    ax1.scatter([w] * len(positions), positions,
                c=['green' if f else 'red' for f in is_flagged],
                s=[50 + 20*ew for ew in error_weights], alpha=0.7)

ax1.set_xlabel('Stabilizer Weight')
ax1.set_ylabel('Fault Position')
ax1.set_title('Flag Triggering by Fault Position\n(Green=Flagged, Size=Error Weight)')
ax1.grid(alpha=0.3)

# Plot 2: Comparison of flag positions
ax2 = axes[0, 1]
w = 6
flag_positions = range(1, w)

for fpos in flag_positions:
    circuit = FlagCircuit(w, fpos)
    fault_positions = range(w + 1)
    error_weights = [circuit.analyze_x_fault(p)[0] for p in fault_positions]
    flagged = [circuit.analyze_x_fault(p)[1] for p in fault_positions]

    # Plot unflagged high-weight as failures
    failures = [ew if (ew > 1 and not f) else 0 for ew, f in zip(error_weights, flagged)]

    ax2.bar([p + 0.15*(fpos-1) for p in fault_positions], failures,
            width=0.15, label=f'Flag at {fpos}', alpha=0.7)

ax2.set_xlabel('Fault Position')
ax2.set_ylabel('Unflagged Error Weight (Failures)')
ax2.set_title(f'Weight-{w} Stabilizer: Failures by Flag Position')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Error rate vs failure probability
ax3 = axes[1, 0]
p_range = np.linspace(0.001, 0.1, 50)

for fpos in [1, 2, 3]:
    failure_rates = []
    for p in p_range:
        stats = simulate_flag_circuit(4, fpos, p, n_trials=10000)
        failure_rates.append(stats['high_weight_unflagged'])
    ax3.semilogy(p_range, [max(f, 1e-6) for f in failure_rates],
                 label=f'Flag at position {fpos}')

ax3.set_xlabel('Physical Error Rate p')
ax3.set_ylabel('Failure Rate (Unflagged High-Weight)')
ax3.set_title('Weight-4 Stabilizer: Failure Rate vs Flag Position')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Resource comparison
ax4 = axes[1, 1]
methods = ['Shor\n(Cat State)', 'Steane\n(Encoded)', 'Flag\n(1 flag)', 'Flag\n(2 flags)']
ancillas = [4, 7, 2, 3]  # For weight-4 stabilizer
colors = ['coral', 'steelblue', 'green', 'lightgreen']

bars = ax4.bar(methods, ancillas, color=colors, edgecolor='black')
ax4.set_ylabel('Ancilla Qubits')
ax4.set_title('Ancilla Count for Weight-4 Stabilizer\n([[7,1,3]] Steane Code)')
ax4.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, ancillas):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             str(count), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('day_877_flag_concept.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'day_877_flag_concept.png'")

# =============================================================================
# Part 7: Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary: The Flag Qubit Concept")
print("=" * 70)

print("""
Key Insights:

1. PARADIGM SHIFT: Don't prevent high-weight errors, DETECT them
   - Flag qubit signals when dangerous fault patterns occur
   - Reduces ancilla count from O(w) to O(1)

2. FLAG PLACEMENT: Position divides CNOT chain
   - Faults before flag → flag triggers
   - Faults after flag → low-weight error (correctable)
   - Optimal: flag_pos >= w - t for t-flag FT

3. FAULT TOLERANCE CONDITION:
   - Every single fault either causes weight ≤ t error OR triggers flag
   - (syndrome, flag) pair provides more information than syndrome alone

4. LOOKUP TABLE: Extended decoder handles flagged cases
   - Flag = 0: Standard decoding
   - Flag = 1: Consider high-weight error possibilities

5. RESOURCE SAVINGS: ~94% reduction for typical codes
   - Shor-style: 36 ancillas for [[7,1,3]]
   - Flag-based: 2 ancillas for [[7,1,3]]

Tomorrow: Flag Circuit Design - How to construct optimal flag circuits
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
| Flag FT condition | $\forall f: \text{wt}(E(f)) \leq t \lor \text{Flag}(f) = 1$ |
| Optimal flag position | $\text{flag\_pos} \geq w - t$ for t-flag FT |
| Ancilla savings | $(w - 2)/w \times 100\%$ reduction |
| Flag circuit qubits | 1 syndrome + $n_{\text{flags}}$ flags |

### Main Takeaways

1. **Paradigm shift:** Detect dangerous errors rather than prevent them
2. **Flag qubits** signal when single faults cause high-weight data errors
3. **Optimal placement** divides the CNOT chain to catch all dangerous faults
4. **(Syndrome, flag) pairs** enable distinguishing error patterns
5. **Dramatic resource reduction** from $O(w)$ to $O(1)$ ancillas per stabilizer

---

## Daily Checklist

- [ ] Explain what a flag qubit does in one sentence
- [ ] Determine optimal flag position for a weight-5 stabilizer
- [ ] Trace through fault analysis for a flag circuit
- [ ] Build a syndrome-flag lookup table
- [ ] Complete Level 1 practice problems
- [ ] Run computational lab and analyze output
- [ ] Explain why flag = 1 doesn't always mean an error occurred

---

## Preview: Day 878

Tomorrow we focus on **flag circuit design**: systematic methods for constructing flag circuits, optimizing CNOT ordering, handling different stabilizer types, and verifying fault tolerance. We'll develop practical skills for building flag circuits for any stabilizer code.

---

*"The flag qubit doesn't prevent disaster - it sounds the alarm."*

---

**Next:** [Day_878_Wednesday.md](Day_878_Wednesday.md) — Flag Circuit Design: Construction and Optimization
