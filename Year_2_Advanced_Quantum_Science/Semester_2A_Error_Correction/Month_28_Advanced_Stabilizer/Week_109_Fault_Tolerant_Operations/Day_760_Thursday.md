# Day 760: Fault-Tolerant Measurement

## Overview

**Day:** 760 of 1008
**Week:** 109 (Fault-Tolerant Quantum Operations)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Syndrome Extraction Without Error Propagation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Shor-style measurement |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Steane-style measurement |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Repeated measurement protocols |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Design** Shor-style syndrome extraction using cat state ancillas
2. **Construct** Steane-style measurement using encoded ancillas
3. **Analyze** the fault tolerance of both approaches
4. **Compare** resource requirements for different methods
5. **Implement** repeated measurement protocols
6. **Distinguish** between data errors and measurement errors

---

## Core Content

### 1. The Measurement Challenge

Syndrome measurement is the most error-prone part of QEC:
- Must extract information about errors
- Cannot collapse the encoded quantum information
- Must not propagate errors from ancilla to data

#### Why Standard Measurement Fails

Standard syndrome extraction:
```
data: ──●──●──●──●──
        │  │  │  │
ancilla: ⊕──⊕──⊕──⊕── Measure
```

**Problem:** X error on ancilla after first CNOT spreads to all subsequent data qubits!

### 2. Shor-Style Syndrome Extraction

**Key Idea:** Use a cat state ancilla to distribute measurement errors.

#### The Protocol

For measuring stabilizer $S = Z_1 Z_2 Z_3 Z_4$:

**Step 1:** Prepare and verify cat state
```
|0⟩ ──H──●──●──●──
         │  │  │
|0⟩ ─────⊕──│──│──
            │  │
|0⟩ ────────⊕──│──
               │
|0⟩ ───────────⊕── |CAT₄⟩
```

**Step 2:** Couple cat to data
```
data₁: ──●─────────────
         │
CAT₁: ───⊕─────────────
data₂: ─────●──────────
            │
CAT₂: ──────⊕──────────
data₃: ────────●───────
               │
CAT₃: ─────────⊕───────
data₄: ───────────●────
                  │
CAT₄: ────────────⊕────
```

**Step 3:** Measure all cat qubits in X basis

**Step 4:** XOR the results → syndrome bit

#### Why It's Fault-Tolerant

- Each cat qubit couples to exactly one data qubit
- X error on any cat qubit → wrong syndrome bit (detected by repeat)
- Z error on cat → no effect (Z commutes with measurement)
- Single fault → at most weight-1 error on data

### 3. Steane-Style Syndrome Extraction

**Key Idea:** Use an encoded ancilla state for transversal coupling.

#### The Protocol

For CSS code with n physical qubits:

**Step 1:** Prepare encoded ancilla
$$|\bar{0}\rangle \text{ for measuring X stabilizers}$$
$$|\bar{+}\rangle \text{ for measuring Z stabilizers}$$

**Step 2:** Transversal CNOT
```
Data block:   |ψ⟩_L ──┬──┬──┬──┬──┬──┬── |ψ'⟩_L
                      │  │  │  │  │  │
Ancilla:      |0⟩_L ──⊕──⊕──⊕──⊕──⊕──⊕── Measure each
```

**Step 3:** Measure ancilla qubits → extract syndrome

#### Advantage: Transversal = Fault-Tolerant

Since the coupling is transversal:
- Single fault affects at most one qubit per block
- Measurement errors on ancilla are correctable
- Can distinguish data errors from ancilla errors

### 4. Comparing Shor vs Steane

| Aspect | Shor Style | Steane Style |
|--------|------------|--------------|
| Ancilla size | = stabilizer weight | = code block |
| Ancilla prep | Cat state | Encoded state |
| Coupling depth | 1 per data qubit | 1 (transversal) |
| Measurement | All cat qubits | All ancilla qubits |
| Error detection | Repeat needed | Built-in |

**Trade-off:**
- Shor: Smaller ancilla, more rounds needed
- Steane: Larger ancilla, fewer rounds

### 5. Repeated Measurements

Even with FT circuits, measurement errors occur. Solution: **repeat!**

#### Basic Repeat Protocol

1. Measure syndrome
2. Repeat measurement (independent ancilla)
3. If results agree → accept
4. If results differ → measure again, use majority

#### Analysis

For error probability p per measurement:
- Single measurement: Wrong with probability p
- Two measurements agree: Error probability ≈ p² (both wrong)
- Three measurements, majority: Error probability ≈ 3p² (two wrong)

$$\boxed{P(\text{wrong syndrome}) \sim O(p^2) \text{ with repeated measurement}}$$

### 6. Distinguishing Error Types

Critical question: **Is the error on the data or the measurement?**

#### The Insight

- Data error: Present in BOTH measurements
- Measurement error: Present in ONE measurement only

**Protocol:**
1. Measure syndrome, get s₁
2. Measure again, get s₂
3. If s₁ = s₂: Likely data error, correct
4. If s₁ ≠ s₂: Possible measurement error, measure again

#### Advanced: Sliding Window

Keep track of last k syndrome measurements:
- Pattern of changes reveals error type
- Steady syndrome → stable error
- Alternating syndrome → measurement noise

---

## Worked Examples

### Example 1: Shor Syndrome for [[7,1,3]] Code

**Problem:** Design Shor-style syndrome extraction for one Z stabilizer of the Steane code.

**Solution:**

The Steane code has stabilizer $Z_1 Z_2 Z_3 Z_4$ (weight 4).

**Step 1: Cat state (verified)**
```
|0⟩ ──H──●──●──●── }
         │  │  │   }  Verify
|0⟩ ─────⊕──│──│── }  with
         │  │  │   }  parity
|0⟩ ────────⊕──│── }  checks
               │   }
|0⟩ ───────────⊕── }
```

**Step 2: Controlled-Z coupling**
```
d₁: ──●─────────────
      │
c₁: ──⊕─────────────
d₂: ─────●──────────
         │
c₂: ─────⊕──────────
d₃: ────────●───────
            │
c₃: ────────⊕───────
d₄: ───────────●────
               │
c₄: ───────────⊕────
```

Actually, for Z stabilizer, use CZ gates:
```
d₁: ──●─────────
      │
c₁: ──Z──H── M
...
```

**Step 3: Measure all cat in X basis**

XOR of measurements gives syndrome bit.

### Example 2: Steane Measurement Analysis

**Problem:** In Steane-style X syndrome extraction, what happens if ancilla qubit 3 has a Z error?

**Solution:**

**Setup:**
```
Data: |ψ⟩_L ──┬──┬──┬──┬──┬──┬──
              │  │  │  │  │  │
Ancilla: |0⟩_L (with Z₃ error)
```

**After transversal CNOT:**
- Z₃ on ancilla → Z₃ on data (backward propagation)
- Weight-1 error on data block

**Measurement outcome:**
- Ancilla measurement gives syndrome of ANCILLA
- The Z₃ error on data must be caught in NEXT round

**Conclusion:** Single ancilla error creates correctable data error. ✓

### Example 3: Repeated Measurement Protocol

**Problem:** Design a repeat protocol that achieves O(p³) syndrome error rate.

**Solution:**

**Protocol: Best-of-5 (majority vote)**

1. Measure syndrome: s₁
2. Measure syndrome: s₂
3. Measure syndrome: s₃
4. Measure syndrome: s₄
5. Measure syndrome: s₅
6. Output: Majority of {s₁, s₂, s₃, s₄, s₅}

**Error analysis:**
- Need 3 or more wrong for incorrect majority
- Probability: $\binom{5}{3}p^3 + \binom{5}{4}p^4 + \binom{5}{5}p^5$
- Leading term: $10p^3$

**Improvement:** From O(p) to O(p³)!

**Note:** In practice, need to balance:
- More repeats → lower syndrome error
- More repeats → more time for data errors to accumulate

---

## Practice Problems

### Problem Set A: Shor-Style Measurement

**A1.** For a weight-6 stabilizer, design the cat state ancilla and coupling circuit.

**A2.** If cat state verification fails, what should the protocol do?

**A3.** Calculate the number of two-qubit gates needed for Shor-style extraction of all 6 stabilizers of the [[7,1,3]] code.

### Problem Set B: Steane-Style Measurement

**B1.** For the [[5,1,3]] code, how many ancilla qubits are needed for Steane-style measurement of one stabilizer?

**B2.** In Steane measurement, what's the advantage of preparing |+⟩_L vs |0⟩_L for Z syndrome extraction?

**B3.** If the encoded ancilla has a weight-2 error before coupling, what happens to the data after transversal CNOT?

### Problem Set C: Repeated Measurements

**C1.** A syndrome is measured 4 times with results {+1, -1, +1, +1}. What should the protocol conclude?

**C2.** Design a repeat protocol that distinguishes between:
a) No error
b) Data error
c) Single measurement error
Using minimum measurements.

**C3.** In a 3-repeat protocol, what's the probability of:
a) Correct syndrome (no errors)
b) Correct syndrome (one error corrected by majority)
c) Wrong syndrome (two or more errors)
Given error probability p.

---

## Computational Lab

```python
"""
Day 760 Computational Lab: Fault-Tolerant Measurement
=====================================================

Simulate syndrome extraction protocols.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SyndromeResult(Enum):
    """Syndrome measurement outcomes."""
    PLUS = +1
    MINUS = -1


@dataclass
class MeasurementOutcome:
    """Result of syndrome measurement with metadata."""
    syndrome: List[int]  # +1 or -1 for each stabilizer
    confidence: float    # Confidence level
    rounds: int          # Number of measurement rounds


class ShorSyndromeMeasurement:
    """
    Shor-style syndrome extraction using cat state ancillas.
    """

    def __init__(self, stabilizer_weight: int, error_prob: float = 0.01):
        self.weight = stabilizer_weight
        self.p = error_prob

    def prepare_cat_state(self) -> Tuple[bool, int]:
        """
        Simulate cat state preparation with verification.

        Returns: (is_valid, preparation_errors)
        """
        # Simulate errors during preparation
        prep_error = np.random.random() < self.p * self.weight

        if prep_error:
            # Verification should catch most errors
            caught = np.random.random() < 0.9
            return not caught, 1

        return True, 0

    def measure_stabilizer(self) -> Tuple[int, Dict]:
        """
        Perform Shor-style syndrome measurement.

        Returns: (syndrome_bit, metadata)
        """
        metadata = {
            'cat_prep_attempts': 0,
            'measurement_errors': 0
        }

        # Prepare verified cat state
        while True:
            metadata['cat_prep_attempts'] += 1
            valid, _ = self.prepare_cat_state()
            if valid or metadata['cat_prep_attempts'] > 10:
                break

        # Simulate individual X-basis measurements
        measurements = []
        for i in range(self.weight):
            # Each measurement can flip with probability p
            if np.random.random() < self.p:
                measurements.append(-1)
                metadata['measurement_errors'] += 1
            else:
                measurements.append(1)

        # XOR all measurements (product of ±1)
        syndrome = np.prod(measurements)

        return int(syndrome), metadata


class SteaneSyndromeMeasurement:
    """
    Steane-style syndrome extraction using encoded ancillas.
    """

    def __init__(self, code_distance: int, n_qubits: int,
                 error_prob: float = 0.01):
        self.d = code_distance
        self.n = n_qubits
        self.p = error_prob

    def prepare_encoded_ancilla(self) -> Tuple[bool, List[int]]:
        """
        Prepare encoded |0⟩_L or |+⟩_L state.

        Returns: (is_valid, error_locations)
        """
        errors = []
        for i in range(self.n):
            if np.random.random() < self.p:
                errors.append(i)

        # Valid if fewer than d/2 errors
        is_valid = len(errors) < self.d // 2
        return is_valid, errors

    def measure_syndrome(self) -> Tuple[List[int], Dict]:
        """
        Perform Steane-style syndrome measurement.

        Returns: (syndrome_bits, metadata)
        """
        metadata = {
            'ancilla_prep_errors': 0,
            'transversal_errors': 0,
            'measurement_errors': 0
        }

        # Prepare encoded ancilla
        valid, ancilla_errors = self.prepare_encoded_ancilla()
        metadata['ancilla_prep_errors'] = len(ancilla_errors)

        # Transversal CNOT errors
        for i in range(self.n):
            if np.random.random() < self.p:
                metadata['transversal_errors'] += 1

        # Measure each ancilla qubit
        syndrome_bits = []
        for i in range(self.n):
            if np.random.random() < self.p:
                syndrome_bits.append(-1)
                metadata['measurement_errors'] += 1
            else:
                syndrome_bits.append(1)

        return syndrome_bits, metadata


class RepeatedMeasurement:
    """
    Repeated syndrome measurement with majority voting.
    """

    def __init__(self, base_measurer, num_repeats: int = 3):
        self.measurer = base_measurer
        self.repeats = num_repeats

    def measure_with_voting(self) -> Tuple[int, Dict]:
        """
        Measure syndrome multiple times and take majority.

        Returns: (final_syndrome, detailed_results)
        """
        results = []
        all_metadata = []

        for i in range(self.repeats):
            if hasattr(self.measurer, 'measure_stabilizer'):
                syndrome, meta = self.measurer.measure_stabilizer()
            else:
                syndrome, meta = self.measurer.measure_syndrome()
                syndrome = syndrome[0] if isinstance(syndrome, list) else syndrome
            results.append(syndrome)
            all_metadata.append(meta)

        # Majority vote
        final = 1 if sum(results) > 0 else -1

        return final, {
            'individual_results': results,
            'agreement': all(r == final for r in results),
            'metadata': all_metadata
        }


def simulate_measurement_protocol(protocol_type: str,
                                 n_trials: int = 1000,
                                 error_prob: float = 0.01) -> Dict:
    """
    Simulate measurement protocol and collect statistics.
    """
    results = {
        'protocol': protocol_type,
        'n_trials': n_trials,
        'error_prob': error_prob,
        'success_rate': 0,
        'avg_rounds': 0,
        'disagreement_rate': 0
    }

    successes = 0
    total_rounds = 0
    disagreements = 0

    for _ in range(n_trials):
        if protocol_type == 'shor':
            measurer = ShorSyndromeMeasurement(4, error_prob)
            repeated = RepeatedMeasurement(measurer, 3)
            syndrome, meta = repeated.measure_with_voting()
        else:
            measurer = SteaneSyndromeMeasurement(3, 7, error_prob)
            repeated = RepeatedMeasurement(measurer, 3)
            syndrome, meta = repeated.measure_with_voting()

        # Assume correct syndrome is +1 for this simulation
        if syndrome == 1:
            successes += 1

        if not meta['agreement']:
            disagreements += 1

        total_rounds += 3  # Always 3 rounds in this protocol

    results['success_rate'] = successes / n_trials
    results['avg_rounds'] = total_rounds / n_trials
    results['disagreement_rate'] = disagreements / n_trials

    return results


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 760: FAULT-TOLERANT MEASUREMENT")
    print("=" * 70)

    # Demo 1: Shor-style measurement
    print("\n" + "=" * 70)
    print("Demo 1: Shor-Style Syndrome Extraction")
    print("=" * 70)

    shor = ShorSyndromeMeasurement(stabilizer_weight=4, error_prob=0.05)

    print("\nSingle Shor measurements (weight-4 stabilizer):")
    for i in range(5):
        syndrome, meta = shor.measure_stabilizer()
        print(f"  Trial {i+1}: syndrome={syndrome:+d}, "
              f"cat_attempts={meta['cat_prep_attempts']}, "
              f"meas_errors={meta['measurement_errors']}")

    # Demo 2: Steane-style measurement
    print("\n" + "=" * 70)
    print("Demo 2: Steane-Style Syndrome Extraction")
    print("=" * 70)

    steane = SteaneSyndromeMeasurement(code_distance=3, n_qubits=7,
                                       error_prob=0.05)

    print("\nSingle Steane measurements ([[7,1,3]] code):")
    for i in range(5):
        syndrome, meta = steane.measure_syndrome()
        print(f"  Trial {i+1}: ancilla_errs={meta['ancilla_prep_errors']}, "
              f"transversal_errs={meta['transversal_errors']}, "
              f"meas_errs={meta['measurement_errors']}")

    # Demo 3: Repeated measurement
    print("\n" + "=" * 70)
    print("Demo 3: Repeated Measurement Protocol")
    print("=" * 70)

    repeated_shor = RepeatedMeasurement(shor, num_repeats=3)

    print("\nRepeated Shor measurements (3 rounds, majority vote):")
    for i in range(5):
        syndrome, meta = repeated_shor.measure_with_voting()
        agree = "agree" if meta['agreement'] else "DISAGREE"
        print(f"  Trial {i+1}: final={syndrome:+d}, "
              f"votes={meta['individual_results']} ({agree})")

    # Demo 4: Protocol comparison
    print("\n" + "=" * 70)
    print("Demo 4: Protocol Comparison")
    print("=" * 70)

    for p in [0.01, 0.02, 0.05]:
        print(f"\nError probability p = {p}:")

        shor_stats = simulate_measurement_protocol('shor', 500, p)
        print(f"  Shor: success={shor_stats['success_rate']:.3f}, "
              f"disagree={shor_stats['disagreement_rate']:.3f}")

        steane_stats = simulate_measurement_protocol('steane', 500, p)
        print(f"  Steane: success={steane_stats['success_rate']:.3f}, "
              f"disagree={steane_stats['disagreement_rate']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("FT MEASUREMENT SUMMARY")
    print("=" * 70)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  FAULT-TOLERANT SYNDROME MEASUREMENT                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  SHOR STYLE:                                                │
    │    • Use verified cat state ancilla                        │
    │    • One cat qubit per stabilizer weight                   │
    │    • Measure in X basis, XOR results                       │
    │    • Single error → one wrong measurement                  │
    │                                                             │
    │  STEANE STYLE:                                              │
    │    • Use encoded ancilla |0⟩_L or |+⟩_L                    │
    │    • Transversal CNOT to data                              │
    │    • Measure each ancilla qubit                            │
    │    • Error correction on ancilla syndrome                  │
    │                                                             │
    │  REPEATED MEASUREMENT:                                      │
    │    • Multiple rounds with majority voting                  │
    │    • Distinguishes data vs measurement errors              │
    │    • k rounds: syndrome error ~ O(p^{⌈k/2⌉})               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)

    print("=" * 70)
    print("Day 760 Complete: FT Measurement Protocols Mastered")
    print("=" * 70)
```

---

## Summary

### Measurement Methods Comparison

| Method | Ancilla | Coupling | FT Property |
|--------|---------|----------|-------------|
| Shor | Cat state | Sequential | Error → wrong bit |
| Steane | Encoded | Transversal | Error → weight-1 |
| Flag | Single + flag | With flags | Flag catches spread |

### Critical Equations

$$\boxed{\text{Shor: } n_{\text{ancilla}} = w \text{ (stabilizer weight)}}$$
$$\boxed{\text{Steane: } n_{\text{ancilla}} = n \text{ (code length)}}$$
$$\boxed{\text{Repeated: } P(\text{wrong}) \sim O(p^{\lceil k/2 \rceil}) \text{ for } k \text{ rounds}}$$

---

## Daily Checklist

- [ ] Designed Shor-style syndrome extraction
- [ ] Understood Steane-style measurement
- [ ] Compared resource requirements
- [ ] Analyzed repeated measurement protocols
- [ ] Ran computational simulations
- [ ] Completed practice problems

---

## Preview: Day 761

Tomorrow we study **Transversal Gates & Universality**:
- Full analysis of transversal gate sets
- Proof of Eastin-Knill theorem
- The Clifford hierarchy
- Approaches to universal fault-tolerant computation
