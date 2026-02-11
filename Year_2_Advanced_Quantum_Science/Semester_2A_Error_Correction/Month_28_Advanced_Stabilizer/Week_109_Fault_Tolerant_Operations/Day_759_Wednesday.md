# Day 759: Fault-Tolerant State Preparation

## Overview

**Day:** 759 of 1008
**Week:** 109 (Fault-Tolerant Quantum Operations)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Preparing Encoded States Without Creating Uncorrectable Errors

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Encoded state preparation |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Verification protocols |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Cat states and flags |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Explain** why naive state preparation fails fault-tolerance requirements
2. **Construct** fault-tolerant preparation circuits for |0⟩_L and |+⟩_L
3. **Design** verification circuits to detect preparation errors
4. **Build** cat state ancillas for syndrome extraction
5. **Apply** flag qubit methods for efficient FT preparation
6. **Analyze** the resource overhead of FT state preparation

---

## Core Content

### 1. The State Preparation Challenge

To begin fault-tolerant computation, we need encoded logical states like |0⟩_L and |+⟩_L. But naive preparation can create correlated errors!

#### Naive Preparation of |0⟩_L

For the [[7,1,3]] Steane code:
$$|0\rangle_L = \frac{1}{\sqrt{8}}\sum_{c \in C_2} |c\rangle$$

Direct preparation uses entangling gates that can spread errors.

**Example non-FT circuit:**
```
|0⟩ ──●──●──●────────
      │  │  │
|0⟩ ──⊕──│──│──●──●──
         │  │  │  │
|0⟩ ─────⊕──│──⊕──│──
            │     │
|0⟩ ────────⊕─────⊕──
...
```

Single CNOT failure → multiple data qubits affected!

### 2. Fault-Tolerant Preparation Strategy

**Key Insight:** We don't need to prepare perfect states. We need to ensure:

$$\boxed{\text{Single fault during prep} \Rightarrow \text{Weight-1 error on output}}$$

#### Method 1: Postselection

1. Prepare state using any circuit
2. Verify the state is correct
3. If verification fails → reject and retry
4. Accept only verified states

**Trade-off:** Lower success probability but guaranteed quality.

#### Method 2: Verification + Correction

1. Prepare state (possibly imperfect)
2. Measure stabilizers without propagating errors
3. If errors detected → correct or retry
4. Continue only with valid states

### 3. Verification Circuits

A **verification circuit** checks if the prepared state satisfies all stabilizer conditions.

#### Simple Verification

For Steane code |0⟩_L preparation:

```
Prepared |ψ⟩_L ──┬──┬──┬──┬──┬──┬──
                 │  │  │  │  │  │
|0⟩_L (ancilla) ─⊕──⊕──⊕──⊕──⊕──⊕── Measure (transversal)
```

**Check:** If ancilla measures all |0⟩, data is likely correct.

**Problem:** Verification itself can introduce errors!

#### Robust Verification

Use multiple rounds of verification:

1. Prepare state
2. First verification round
3. Second verification round (independent ancilla)
4. Accept only if both pass

**Probability analysis:**
- Single fault → caught by one round with high probability
- Two faults needed to slip through → probability O(p²)

### 4. Cat State Ancillas

**Cat states** distribute single errors across multiple qubits:

$$|CAT_n\rangle = \frac{1}{\sqrt{2}}(|0\rangle^{\otimes n} + |1\rangle^{\otimes n})$$

#### Cat State Preparation

```
|0⟩ ──H──●──●──●── |CAT₄⟩
         │  │  │
|0⟩ ─────⊕──│──│──
            │  │
|0⟩ ────────⊕──│──
               │
|0⟩ ───────────⊕──
```

#### Why Cat States Help

For syndrome extraction, use cat state as ancilla:

```
data₁: ──●────────────
         │
data₂: ──│──●─────────
         │  │
data₃: ──│──│──●──────
         │  │  │
CAT:  ───⊕──⊕──⊕── (measure all)
```

**Property:** X error on one cat qubit → affects one data qubit
- Cat state verified before use
- Error on cat → wrong syndrome OR weight-1 data error
- Both are handled by FT protocol

### 5. Cat State Verification

Before using cat state, verify it was prepared correctly:

$$\boxed{|CAT_n\rangle: Z_1Z_2 = Z_2Z_3 = \cdots = Z_{n-1}Z_n = +1}$$

**Verification circuit:**
```
|CAT⟩ qubit 1: ──●─────────────
                 │
|CAT⟩ qubit 2: ──⊕──●──────────
                    │
|0⟩ (ancilla): ─────⊕── Measure
```

Repeat for each pair → verify all parities are even.

### 6. Flag Qubit Methods

**Flag qubits** detect when errors might have spread dangerously.

#### Basic Flag Circuit

```
data₁: ─────●─────────────────────
            │
flag:  ──⊕──┼──●─────────────⊕── M_flag
            │  │             │
data₂: ─────│──⊕──●──────────│──
            │     │          │
data₃: ─────│─────⊕──●───────│──
            │        │       │
ancilla: ───⊕────────⊕───────⊕── M_syndrome
```

**How it works:**
- Normal operation: flag qubit unchanged
- Dangerous fault (e.g., X on ancilla between CNOTs): flag flips
- If flag = 1: Extra syndrome measurements needed

#### Flag-FT Condition

$$\boxed{\text{Flag triggered} \Leftrightarrow \text{Potentially dangerous error}}$$

When flag is triggered:
- Don't trust this syndrome
- Take additional measurements
- Use extra info to identify error

---

## Worked Examples

### Example 1: Steane Code |0⟩_L Preparation

**Problem:** Design a fault-tolerant preparation of |0⟩_L for the Steane code.

**Solution:**

**Step 1: Understand the target**
$$|0\rangle_L = \frac{1}{\sqrt{8}}\sum_{c \in C_2} |c\rangle$$

where C₂ is the [7,4,3] Hamming code.

**Step 2: Non-FT encoding circuit**

Standard encoding:
1. Start with |0000000⟩
2. Apply H to first 4 qubits
3. Apply controlled operations to create codewords

**Step 3: Add verification**

After encoding, verify X stabilizers:
- Measure $X_1X_2X_3X_4$, $X_2X_3X_5X_6$, $X_1X_3X_5X_7$
- If all +1 → accept
- If any -1 → reject and retry

**Step 4: Use cat state for Z stabilizers**

Prepare |CAT₄⟩, verify, then measure Z stabilizers.

**Final protocol:**
1. Encode (non-FT but simple)
2. Verify X stabilizers with cat states
3. Verify Z stabilizers with cat states
4. Accept or retry based on verification

### Example 2: Cat State Error Analysis

**Problem:** A 4-qubit cat state is prepared and an X error occurs on the second CNOT. Analyze the resulting state.

**Solution:**

**Normal preparation:**
```
|0⟩ ──H──●──●──●──
         │  │  │
|0⟩ ─────⊕──│──│──
            │  │
|0⟩ ────────⊕──│──
               │
|0⟩ ───────────⊕──
```

**With X error after CNOT₂:**

Before error:
$$\frac{1}{\sqrt{2}}(|0000\rangle + |1100\rangle)$$

After X on qubit 3 (from faulty CNOT):
$$\frac{1}{\sqrt{2}}(|0010\rangle + |1110\rangle)$$

After remaining CNOT:
$$\frac{1}{\sqrt{2}}(|0011\rangle + |1111\rangle)$$

**Verification check:** Measure Z₂Z₃:
- Normal cat: Z₂Z₃|CAT⟩ = +|CAT⟩
- Error state: Z₂Z₃ gives -1

**Conclusion:** Verification catches the error! ✓

### Example 3: Flag Qubit Syndrome

**Problem:** For the 4-qubit syndrome extraction with flag, determine which faults trigger the flag.

**Solution:**

Circuit (repeated from above):
```
d₁: ─────●─────────────────────
         │
f:  ──⊕──┼──●─────────────⊕── M_f
         │  │             │
d₂: ─────│──⊕──●──────────│──
         │     │          │
d₃: ─────│─────⊕──●───────│──
         │        │       │
a:  ─────⊕────────⊕───────⊕── M_s
```

**Fault analysis:**

| Fault Location | Effect on Flag | Dangerous? |
|----------------|---------------|------------|
| X on d₁ | No flag | No (weight-1) |
| X on f (first gate) | No flag | No (X stays on f) |
| X on a after first CNOT | Flag = 1 | Yes! Would spread |
| X on a after middle CNOTs | Flag = 1 | Yes! |
| X on a after last CNOT | No flag | No (no spreading) |

**Key insight:** Flag triggers exactly when X on ancilla could spread to multiple data qubits.

---

## Practice Problems

### Problem Set A: State Preparation

**A1.** For the [[5,1,3]] code, list the stabilizer conditions that must be verified for |0⟩_L.

**A2.** If preparation succeeds with probability 0.9 per attempt, and we need verified success, how many attempts (on average) are needed?

**A3.** Compare the qubit overhead of:
a) Verification by measuring each stabilizer with fresh ancilla
b) Verification using Steane-style transversal measurement

### Problem Set B: Cat States

**B1.** Write the stabilizer generators for |CAT₆⟩.

**B2.** An 8-qubit cat state is used for syndrome extraction. If one cat qubit has an X error:
a) What syndrome is reported?
b) What error appears on the data?

**B3.** Design a verification circuit for |CAT₅⟩ that uses only 2 ancilla qubits total.

### Problem Set C: Flag Qubits

**C1.** For a weight-4 stabilizer XXXX measured with flag qubit:
a) How many flag qubits are needed?
b) Which fault locations are caught?

**C2.** The flag protocol requires extra syndrome measurements when flag triggers. For the [[7,1,3]] code, how many extra measurements are needed per triggered flag?

**C3.** Compare the circuit depth of:
a) Shor-style syndrome extraction (using cat states)
b) Flag-based syndrome extraction
For a weight-4 stabilizer.

---

## Computational Lab

```python
"""
Day 759 Computational Lab: Fault-Tolerant State Preparation
===========================================================

Simulate cat state preparation and verification.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class QuantumState:
    """Simple quantum state vector representation."""
    amplitudes: np.ndarray
    n_qubits: int

    @classmethod
    def computational_basis(cls, n: int, state: int = 0) -> 'QuantumState':
        """Create |state⟩ in n-qubit system."""
        amps = np.zeros(2**n, dtype=complex)
        amps[state] = 1.0
        return cls(amps, n)

    def apply_H(self, qubit: int) -> 'QuantumState':
        """Apply Hadamard to qubit."""
        new_amps = np.zeros_like(self.amplitudes)
        for i in range(2**self.n_qubits):
            bit = (i >> qubit) & 1
            i_flip = i ^ (1 << qubit)
            sign = (-1)**bit
            new_amps[i] = (self.amplitudes[i] + sign * self.amplitudes[i_flip]) / np.sqrt(2)
        return QuantumState(new_amps, self.n_qubits)

    def apply_CNOT(self, control: int, target: int) -> 'QuantumState':
        """Apply CNOT from control to target."""
        new_amps = np.zeros_like(self.amplitudes)
        for i in range(2**self.n_qubits):
            if (i >> control) & 1:
                new_amps[i ^ (1 << target)] = self.amplitudes[i]
            else:
                new_amps[i] = self.amplitudes[i]
        return QuantumState(new_amps, self.n_qubits)

    def apply_X(self, qubit: int) -> 'QuantumState':
        """Apply X to qubit."""
        new_amps = np.zeros_like(self.amplitudes)
        for i in range(2**self.n_qubits):
            new_amps[i ^ (1 << qubit)] = self.amplitudes[i]
        return QuantumState(new_amps, self.n_qubits)

    def measure_ZZ(self, q1: int, q2: int) -> int:
        """Measure Z⊗Z on two qubits, return ±1."""
        expect = 0
        for i in range(2**self.n_qubits):
            bit1 = (i >> q1) & 1
            bit2 = (i >> q2) & 1
            parity = (-1)**(bit1 ^ bit2)
            expect += parity * np.abs(self.amplitudes[i])**2
        return 1 if expect > 0 else -1

    def is_cat_state(self) -> bool:
        """Check if state is a valid cat state."""
        # Valid cat has amplitudes only at |00...0⟩ and |11...1⟩
        all_zeros = 0
        all_ones = 2**self.n_qubits - 1

        for i in range(2**self.n_qubits):
            if i not in [all_zeros, all_ones]:
                if np.abs(self.amplitudes[i]) > 1e-10:
                    return False

        return True


def prepare_cat_state(n: int, error_location: Optional[int] = None,
                     error_type: str = 'X') -> QuantumState:
    """
    Prepare n-qubit cat state with optional error injection.

    Returns: Final state (possibly with error)
    """
    state = QuantumState.computational_basis(n, 0)

    # Apply H to first qubit
    state = state.apply_H(0)

    # Apply CNOTs to spread entanglement
    for i in range(1, n):
        state = state.apply_CNOT(0, i)

        # Inject error if specified
        if error_location == i and error_type == 'X':
            state = state.apply_X(i)

    return state


def verify_cat_state(state: QuantumState) -> Tuple[bool, List[int]]:
    """
    Verify cat state by checking ZZ parities.

    Returns: (is_valid, list of parity measurements)
    """
    n = state.n_qubits
    parities = []

    for i in range(n - 1):
        parity = state.measure_ZZ(i, i + 1)
        parities.append(parity)

    is_valid = all(p == 1 for p in parities)
    return is_valid, parities


class FTStatePreparation:
    """
    Fault-tolerant state preparation protocols.
    """

    def __init__(self, n_qubits: int, max_attempts: int = 10):
        self.n = n_qubits
        self.max_attempts = max_attempts

    def prepare_verified_cat(self, error_prob: float = 0.0
                            ) -> Tuple[Optional[QuantumState], int]:
        """
        Prepare cat state with verification.

        Returns: (state or None, number of attempts)
        """
        for attempt in range(self.max_attempts):
            # Simulate random error
            error_loc = None
            if np.random.random() < error_prob:
                error_loc = np.random.randint(1, self.n)

            state = prepare_cat_state(self.n, error_loc)
            is_valid, parities = verify_cat_state(state)

            if is_valid:
                return state, attempt + 1

        return None, self.max_attempts

    def compute_success_probability(self, error_prob: float,
                                   num_trials: int = 1000) -> float:
        """Estimate probability of successful verified preparation."""
        successes = 0
        for _ in range(num_trials):
            state, _ = self.prepare_verified_cat(error_prob)
            if state is not None:
                successes += 1
        return successes / num_trials


def analyze_flag_circuit(n_data: int = 4) -> dict:
    """
    Analyze flag qubit circuit for syndrome extraction.
    """
    results = {
        'n_data': n_data,
        'fault_locations': [],
        'flags_triggered': 0,
        'undetected_weight2': 0
    }

    # Fault locations in order: d1-a, f-a, a-d2, a-d3, f-a (final)
    # For simplicity, analyze key locations

    fault_scenarios = [
        ('X on ancilla after CNOT 1', True, 1),   # flag, weight-1 with flag
        ('X on ancilla after CNOT 2', True, 1),   # flag triggered
        ('X on ancilla after CNOT 3', True, 1),   # flag triggered
        ('X on data 1', False, 1),                 # no flag, weight-1
        ('X on flag (early)', False, 0),           # no flag, no data error
    ]

    for desc, flag_triggered, data_weight in fault_scenarios:
        results['fault_locations'].append({
            'description': desc,
            'flag_triggered': flag_triggered,
            'data_error_weight': data_weight
        })
        if flag_triggered:
            results['flags_triggered'] += 1
        if not flag_triggered and data_weight > 1:
            results['undetected_weight2'] += 1

    return results


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 759: FAULT-TOLERANT STATE PREPARATION")
    print("=" * 70)

    # Demo 1: Cat state preparation
    print("\n" + "=" * 70)
    print("Demo 1: Cat State Preparation")
    print("=" * 70)

    n = 4
    cat = prepare_cat_state(n)
    print(f"\nPrepared {n}-qubit cat state:")
    print(f"  Is valid cat state: {cat.is_cat_state()}")

    is_valid, parities = verify_cat_state(cat)
    print(f"  Verification result: {'PASS' if is_valid else 'FAIL'}")
    print(f"  ZZ parities: {parities}")

    # Demo 2: Cat state with error
    print("\n" + "=" * 70)
    print("Demo 2: Cat State with X Error During Preparation")
    print("=" * 70)

    cat_error = prepare_cat_state(n, error_location=2, error_type='X')
    print(f"\nCat state with X error after CNOT 2:")
    print(f"  Is valid cat state: {cat_error.is_cat_state()}")

    is_valid, parities = verify_cat_state(cat_error)
    print(f"  Verification result: {'PASS' if is_valid else 'FAIL'}")
    print(f"  ZZ parities: {parities} (look for -1)")

    # Demo 3: Verified preparation with errors
    print("\n" + "=" * 70)
    print("Demo 3: Verified Preparation Protocol")
    print("=" * 70)

    ft_prep = FTStatePreparation(4, max_attempts=20)

    print("\nWith error probability 0.3:")
    for trial in range(3):
        state, attempts = ft_prep.prepare_verified_cat(error_prob=0.3)
        status = "SUCCESS" if state else "FAILED"
        print(f"  Trial {trial+1}: {status} after {attempts} attempts")

    # Demo 4: Flag circuit analysis
    print("\n" + "=" * 70)
    print("Demo 4: Flag Qubit Circuit Analysis")
    print("=" * 70)

    flag_results = analyze_flag_circuit(4)
    print(f"\nFlag circuit for weight-{flag_results['n_data']} stabilizer:")
    print(f"  Flags triggered: {flag_results['flags_triggered']} scenarios")
    print(f"  Undetected weight-2: {flag_results['undetected_weight2']} scenarios")

    print("\nFault scenarios:")
    for scenario in flag_results['fault_locations']:
        flag_str = "FLAG" if scenario['flag_triggered'] else "no flag"
        print(f"  {scenario['description']}: {flag_str}, "
              f"weight={scenario['data_error_weight']}")

    # Summary
    print("\n" + "=" * 70)
    print("FT STATE PREPARATION SUMMARY")
    print("=" * 70)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  FAULT-TOLERANT STATE PREPARATION                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  VERIFICATION METHOD:                                       │
    │    1. Prepare state (possibly imperfect)                   │
    │    2. Measure stabilizers to verify                        │
    │    3. Reject and retry if verification fails               │
    │                                                             │
    │  CAT STATE ANCILLAS:                                        │
    │    |CAT_n⟩ = (|00...0⟩ + |11...1⟩)/√2                      │
    │    - Verified by ZZ parity checks                          │
    │    - Distribute single errors across qubits                │
    │                                                             │
    │  FLAG QUBITS:                                               │
    │    - Detect dangerous fault patterns                        │
    │    - Trigger additional syndrome measurements               │
    │    - Reduce ancilla overhead vs cat states                  │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)

    print("=" * 70)
    print("Day 759 Complete: FT State Preparation Protocols Mastered")
    print("=" * 70)
```

---

## Summary

### State Preparation Strategies

| Method | Ancilla Overhead | Success Rate | Complexity |
|--------|-----------------|--------------|------------|
| Verification | +n per stabilizer | ~1-p | Moderate |
| Cat states | +n per cat | ~1-p | Higher |
| Flag qubits | +1-2 per stabilizer | ~1-p | Lower |

### Critical Equations

$$\boxed{|CAT_n\rangle = \frac{1}{\sqrt{2}}(|0\rangle^{\otimes n} + |1\rangle^{\otimes n})}$$

$$\boxed{\text{Cat verification: } Z_i Z_{i+1} = +1 \text{ for all } i}$$

### Key Insight

> **Verified preparation** ensures that even if preparation has errors, we only accept states with correctable errors. The verification process must itself be fault-tolerant!

---

## Daily Checklist

- [ ] Understood why naive preparation fails
- [ ] Designed verification circuits
- [ ] Built and analyzed cat states
- [ ] Applied flag qubit methods
- [ ] Ran computational simulations
- [ ] Completed practice problems

---

## Preview: Day 760

Tomorrow we tackle **Fault-Tolerant Measurement**, learning:
- Syndrome extraction without error propagation
- Shor-style ancilla measurement
- Steane-style measurement
- Repeated measurements for reliability
