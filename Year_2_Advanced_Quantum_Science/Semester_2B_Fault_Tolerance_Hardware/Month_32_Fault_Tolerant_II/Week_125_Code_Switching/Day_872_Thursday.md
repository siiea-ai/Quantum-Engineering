# Day 872: Gauge Fixing Protocols

## Overview

**Day:** 872 of 1008
**Week:** 125 (Code Switching & Gauge Fixing)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** Gauge Fixing for Code Switching and Universality via Gauge Manipulation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Gauge fixing theory and protocols |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Universality via gauge fixing |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational implementation |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Define** gauge fixing as a projection operation on subsystem codes
2. **Design** measurement-based gauge fixing protocols
3. **Implement** fault-tolerant gauge fixing with error detection
4. **Explain** how gauge fixing enables transversal gates not available in the original code
5. **Describe** the Paetznick-Reichardt universality construction
6. **Connect** gauge fixing to code switching mechanisms

---

## Gauge Fixing: Fundamental Concepts

### What is Gauge Fixing?

**Definition:** **Gauge fixing** is the process of measuring gauge operators to project a subsystem code state into a specific gauge sector.

For a subsystem code with gauge group $\mathcal{G}$:

$$\boxed{|\psi\rangle_{\text{subsystem}} \xrightarrow{\text{gauge fixing}} |\psi\rangle_{\text{stabilizer}}}$$

The logical information is preserved; only the gauge degrees of freedom are fixed.

### Mathematical Description

Let $\{G_1, G_2, \ldots, G_r\}$ be gauge operators that form $r$ gauge qubits (with anti-commuting pairs).

**Before gauge fixing:**
$$|\Psi\rangle = |\psi_L\rangle \otimes |\phi_G\rangle$$

where $|\phi_G\rangle$ is an arbitrary state of the gauge qubits.

**After gauge fixing** (measuring all gauge operators):
$$|\Psi'\rangle = |\psi_L\rangle \otimes |g_1, g_2, \ldots, g_r\rangle_G$$

where $g_i \in \{0, 1\}$ are the measurement outcomes.

### The Projection

Gauge fixing with outcome $\mathbf{g} = (g_1, \ldots, g_r)$ projects:

$$\boxed{P_{\mathbf{g}} = \prod_{i=1}^{r} \frac{I + (-1)^{g_i} G_i^Z}{2}}$$

where $G_i^Z$ is the Z-type operator of the $i$-th gauge qubit pair.

---

## Measurement-Based Gauge Fixing

### Protocol for Bacon-Shor Code

**Goal:** Fix the Bacon-Shor [[9,1,3]] code to the Shor [[9,1,3]] code.

**Gauge operators to fix:** All Z-type gauge operators (vertical pairs)

$$G_Z^{(1)} = Z_0 Z_3, \quad G_Z^{(2)} = Z_3 Z_6$$
$$G_Z^{(3)} = Z_1 Z_4, \quad G_Z^{(4)} = Z_4 Z_7$$
$$G_Z^{(5)} = Z_2 Z_5, \quad G_Z^{(6)} = Z_5 Z_8$$

**Protocol:**

```
Step 1: Measure G_Z^(1) = Z_0 Z_3
        Outcome: m_1 ∈ {+1, -1}
        If m_1 = -1: Apply X_0 or X_3 (correction)

Step 2: Measure G_Z^(2) = Z_3 Z_6
        Outcome: m_2 ∈ {+1, -1}
        If m_2 = -1: Apply X_6

Step 3-6: Repeat for remaining gauge operators
```

**Result:** All Z gauge operators now have eigenvalue $+1$.

### Circuit for Gauge Measurement

```
Data qubit 0:  ─────●─────────
                    │
Data qubit 3:  ─────●─────────
                    │
Ancilla:       |0⟩──⊕──[H]──M── outcome m
```

**Measurement extracts** $\langle Z_0 Z_3 \rangle$ into the ancilla.

### Fault-Tolerant Gauge Fixing

**Challenge:** Measurement errors can cause incorrect gauge fixing.

**Solution:** Repeat measurements and take majority vote.

**Verified Gauge Fixing Protocol:**

```
For each gauge operator G:
    outcomes = []
    Repeat 3 times:
        m = measure(G)
        outcomes.append(m)

    majority = majority_vote(outcomes)

    If majority == -1:
        Apply correction
```

This tolerates single measurement errors.

---

## Gauge Fixing Enables New Transversal Gates

### The Key Insight

Different gauge fixings of the same subsystem code can yield **different stabilizer codes** with **different transversal gates**!

### Example: Gauge Color Code

Bombin's **gauge color code** (2015) demonstrates this principle:

**Subsystem code:** 3D color code with gauge structure

**Gauge fixing A:** Yields a 2D color code with transversal $\{H, S, \text{CNOT}\}$

**Gauge fixing B:** Yields a different code with transversal $\{T, \text{CNOT}\}$

By switching between gauge fixings, we access complementary gate sets!

### The Switching Procedure

$$\boxed{\text{Gauge Fix A} \xrightarrow{\text{unfix}} \text{Subsystem Code} \xrightarrow{\text{Gauge Fix B}} \text{Gauge Fix B}}$$

**Steps:**
1. Start in gauge-fixed code A
2. "Unfix" by applying random gauge operators (or measuring X-type)
3. Re-fix in configuration B by measuring new gauge operators
4. Now have different transversal gates available

---

## The Paetznick-Reichardt Construction

### Universality via Gauge Fixing (2013)

**Theorem (Paetznick-Reichardt):** There exists a subsystem code and gauge fixing protocol that achieves **universal fault-tolerant quantum computation using only transversal gates and gauge fixing**.

No magic state distillation required!

### The Construction

**Base Code:** A 3D subsystem color code

**Gauge Fixings:**
1. **Fixing A:** Transversal Clifford gates {H, S, CNOT}
2. **Fixing B:** Transversal T gate

**Protocol for T gate:**
1. Start in Fixing A (Clifford-transversal)
2. Gauge-unfix (measure X-type gauge operators)
3. Re-fix to Fixing B (T-transversal)
4. Apply $T^{\otimes n}$ transversally
5. Gauge-unfix again
6. Re-fix to Fixing A

### Why This Works

The gauge fixing/unfixing is **fault-tolerant**:
- Each gauge measurement is weight-2
- Error propagation is bounded
- Logical information is preserved throughout

### Circuit Complexity

| Operation | Circuit Depth | Ancilla Qubits |
|-----------|---------------|----------------|
| Gauge unfix | $O(1)$ | $O(n)$ |
| Gauge refix | $O(1)$ | $O(n)$ |
| Transversal gate | $O(1)$ | 0 |

Total for one T gate: $O(1)$ depth (constant!)

Compare to magic state distillation: $O(\log(1/\epsilon))$ depth.

---

## Formal Analysis of Gauge Transformations

### Gauge Equivalence Classes

**Definition:** Two states $|\psi\rangle$ and $|\phi\rangle$ are **gauge equivalent** if:
$$|\phi\rangle = G|\psi\rangle$$

for some gauge operator $G \in \mathcal{G}$.

Gauge equivalent states encode the same logical information.

### The Gauge Orbit

The **gauge orbit** of a state $|\psi\rangle$:
$$\mathcal{O}(|\psi\rangle) = \{G|\psi\rangle : G \in \mathcal{G}\}$$

All states in an orbit are logically equivalent.

### Gauge Fixing as Orbit Selection

Gauge fixing selects a **canonical representative** from each orbit:

$$\boxed{|\psi\rangle_{\text{fixed}} = \text{canonical}(\mathcal{O}(|\psi\rangle))}$$

Different gauge fixing protocols select different canonical representatives.

---

## Subsystem Lattice Surgery

### Connecting to Code Switching

**Subsystem Lattice Surgery (SLS)** unifies:
- Code switching
- Gauge fixing
- Lattice surgery

into a single framework.

### The SLS Framework

**Key Idea:** View lattice surgery operations as gauge fixing on a larger subsystem code.

**Merge operation:**
1. Create enlarged subsystem code from two surface code patches
2. The "seam" contains gauge qubits
3. Gauge fix by measuring seam operators
4. Result: Merged patch with logical operation applied

**Split operation:**
1. Start with single patch
2. Introduce gauge qubits along cut line
3. Gauge fix to create two separate patches
4. Logical information is distributed

### Formal Description

Let $\mathcal{C}_1$ and $\mathcal{C}_2$ be two codes to merge.

**Combined system:**
$$\mathcal{H}_{12} = \mathcal{H}_{C_1} \otimes \mathcal{H}_{C_2} \otimes \mathcal{H}_{\text{boundary}}$$

**Subsystem structure:**
$$\mathcal{H}_{12} = \mathcal{H}_{L_{12}} \otimes \mathcal{H}_{G_{\text{boundary}}}$$

**Gauge fixing** the boundary yields the merged code.

---

## Worked Examples

### Example 1: Gauge Fixing Bacon-Shor to Shor

**Problem:** Starting from an arbitrary Bacon-Shor state, perform gauge fixing to obtain the Shor code. Track the state transformation.

**Solution:**

**Initial state:**
$$|\Psi\rangle = |\psi_L\rangle \otimes |\phi_G\rangle$$

where $|\phi_G\rangle = \alpha|0000\rangle_G + \beta|0001\rangle_G + \cdots$ is an arbitrary 4-gauge-qubit state.

**Step 1:** Measure $G_Z^{(1)} = Z_0 Z_3$

This projects onto eigenspace of $G_Z^{(1)}$:
- Outcome $+1$: $|\phi_G\rangle \to$ projection onto $G_Z^{(1)} = +1$ sector
- Outcome $-1$: Apply $X_0$ correction, then $G_Z^{(1)} = +1$

**Step 2:** Measure $G_Z^{(2)} = Z_3 Z_6$

- If $-1$: Apply $X_6$

**Continue** for all 6 Z-gauge operators (4 independent).

**Final state:**
$$|\Psi_{\text{fixed}}\rangle = |\psi_L\rangle \otimes |++++\rangle_G$$

where $|++++\rangle_G$ means all Z-gauge eigenvalues are $+1$.

This is exactly the Shor code:
$$\boxed{\text{Bacon-Shor} + \text{gauge fix} = \text{Shor code}}$$

### Example 2: Computing Gauge Fixing Probability

**Problem:** For a Bacon-Shor state with gauge qubits in the maximally mixed state, what is the probability of measuring all Z-gauge operators to be $+1$?

**Solution:**

**Maximally mixed gauge state:**
$$\rho_G = \frac{I_{2^4}}{2^4} = \frac{I_{16}}{16}$$

**Probability of $G_Z^{(i)} = +1$ for each:**
$$P(G_Z^{(i)} = +1) = \frac{1}{2}$$

For 4 independent gauge measurements:
$$P(\text{all } +1) = \left(\frac{1}{2}\right)^4 = \frac{1}{16}$$

**With correction:**
If we correct after each $-1$ outcome, the probability of needing $k$ corrections:
$$P(k \text{ corrections}) = \binom{4}{k} \left(\frac{1}{2}\right)^4 = \binom{4}{k} \frac{1}{16}$$

$$\boxed{P(\text{all } +1 \text{ after correction}) = 1 \text{ (deterministic with correction)}}$$

### Example 3: Gauge Fixing for T Gate

**Problem:** Describe how gauge fixing enables a transversal T gate on a code that doesn't normally have one.

**Solution:**

**Setup:** Consider a subsystem code with two gauge fixings:
- **Fixing A:** Stabilizers $\mathcal{S}_A$, transversal gates $\{H, S, \text{CNOT}\}$
- **Fixing B:** Stabilizers $\mathcal{S}_B$, transversal gates $\{T, \text{CNOT}\}$

**Protocol for T gate (starting in Fixing A):**

**Step 1:** Prepare state $|\psi_L\rangle_A$ in gauge-fixed code A.

**Step 2:** Unfix gauge (randomize gauge sector):
- Measure X-type gauge operators
- Or apply Hadamard to gauge qubits: $H_G: |g_i\rangle \to |+_i\rangle$

**Step 3:** Re-fix to gauge B:
- Measure B's gauge operators
- Apply corrections as needed

**Step 4:** Apply transversal T:
$$T^{\otimes n}: |\psi_L\rangle_B \to T|\psi_L\rangle_B$$

**Step 5:** Unfix and re-fix to A (reverse of steps 2-3).

**Result:**
$$\boxed{|\psi_L\rangle_A \xrightarrow{\text{gauge protocol}} T|\psi_L\rangle_A}$$

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For the Bacon-Shor code, how many independent gauge operators must be measured to fully gauge-fix the code?

**P1.2** Write the projection operator for gauge-fixing $G_Z = Z_0 Z_3$ to eigenvalue $+1$.

**P1.3** If a single Z-gauge measurement gives outcome $-1$, what Pauli correction restores it to $+1$?

### Level 2: Intermediate

**P2.1** Design a circuit to measure $G_Z = Z_i Z_j$ using a single ancilla qubit and CNOT gates. Show it extracts the correct eigenvalue.

**P2.2** Prove that gauge fixing preserves logical information: if $|\psi\rangle = \alpha|0_L\rangle + \beta|1_L\rangle$ before gauge fixing, then $|\psi'\rangle = \alpha|0_L\rangle + \beta|1_L\rangle$ after.

**P2.3** For a subsystem code with $r$ gauge qubits, show that the gauge-fixed code has $2^r$ distinct stabilizer codes (one for each gauge sector).

### Level 3: Challenging

**P3.1** Prove that gauge fixing is fault-tolerant: a single error during gauge fixing causes at most a single error in the final gauge-fixed state.

**P3.2** Design a gauge fixing protocol that simultaneously fixes multiple commuting gauge operators in parallel. What is the minimum circuit depth?

**P3.3** For the Paetznick-Reichardt construction, estimate the total resource overhead for implementing a logical T gate via gauge fixing, compared to magic state distillation.

---

## Computational Lab

```python
"""
Day 872: Gauge Fixing Protocols
===============================

Implementation of gauge fixing procedures for subsystem codes.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def tensor_list(ops: List[np.ndarray]) -> np.ndarray:
    """Tensor product of list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def pauli_on_qubits(pauli: np.ndarray, qubits: List[int], n: int) -> np.ndarray:
    """Apply Pauli to specified qubits in n-qubit system."""
    ops = [I] * n
    for q in qubits:
        ops[q] = pauli
    return tensor_list(ops)


class GaugeFixingOutcome(Enum):
    PLUS = +1
    MINUS = -1


@dataclass
class GaugeMeasurement:
    """Result of a gauge operator measurement."""
    operator_qubits: Tuple[int, int]
    operator_type: str  # 'X' or 'Z'
    outcome: GaugeFixingOutcome
    correction_applied: bool


class GaugeFixingProtocol:
    """
    Implements gauge fixing for the Bacon-Shor code.
    """

    def __init__(self, n_qubits: int = 9):
        self.n = n_qubits
        self.dim = 2 ** n_qubits

        # Define Z-gauge operators for Bacon-Shor
        self.z_gauge_operators = [
            (0, 3), (3, 6),  # Column 0
            (1, 4), (4, 7),  # Column 1
            (2, 5), (5, 8),  # Column 2
        ]

        # Only 4 are independent (6 operators, 2 redundancies)
        self.independent_z_gauge = [
            (0, 3), (3, 6),
            (1, 4),
            (2, 5),
        ]

    def gauge_operator(self, qubits: Tuple[int, int],
                       pauli_type: str = 'Z') -> np.ndarray:
        """Create gauge operator Z_i Z_j or X_i X_j."""
        P = Z if pauli_type == 'Z' else X
        return pauli_on_qubits(P, list(qubits), self.n)

    def projection_operator(self, qubits: Tuple[int, int],
                           eigenvalue: int = +1) -> np.ndarray:
        """
        Create projection onto eigenspace of Z_i Z_j.

        P_± = (I ± Z_i Z_j) / 2
        """
        G = self.gauge_operator(qubits, 'Z')
        I_full = np.eye(self.dim, dtype=complex)
        return (I_full + eigenvalue * G) / 2

    def measure_gauge_operator(self, state: np.ndarray,
                               qubits: Tuple[int, int]) -> Tuple[np.ndarray, int]:
        """
        Simulate measurement of Z_i Z_j gauge operator.

        Returns:
        --------
        (new_state, outcome)
        """
        G = self.gauge_operator(qubits, 'Z')

        # Calculate probabilities
        P_plus = self.projection_operator(qubits, +1)
        P_minus = self.projection_operator(qubits, -1)

        prob_plus = np.real(np.vdot(state, P_plus @ state))
        prob_minus = np.real(np.vdot(state, P_minus @ state))

        # Normalize probabilities
        total = prob_plus + prob_minus
        prob_plus /= total
        prob_minus /= total

        # Random measurement outcome
        outcome = np.random.choice([+1, -1], p=[prob_plus, prob_minus])

        # Project state
        if outcome == +1:
            new_state = P_plus @ state
        else:
            new_state = P_minus @ state

        # Normalize
        new_state = new_state / np.linalg.norm(new_state)

        return new_state, outcome

    def apply_correction(self, state: np.ndarray,
                        qubit: int) -> np.ndarray:
        """Apply X correction to specified qubit."""
        X_q = pauli_on_qubits(X, [qubit], self.n)
        return X_q @ state

    def gauge_fix_all(self, state: np.ndarray,
                      verbose: bool = True) -> Tuple[np.ndarray, List[GaugeMeasurement]]:
        """
        Perform complete gauge fixing to convert Bacon-Shor to Shor code.

        Parameters:
        -----------
        state : np.ndarray
            Initial state (can be any Bacon-Shor encoded state)
        verbose : bool
            Print intermediate steps

        Returns:
        --------
        (fixed_state, measurements): Final state and measurement record
        """
        measurements = []
        current_state = state.copy()

        if verbose:
            print("Gauge Fixing Protocol: Bacon-Shor -> Shor")
            print("=" * 50)

        for i, qubits in enumerate(self.independent_z_gauge):
            if verbose:
                print(f"\nStep {i+1}: Measure G_Z = Z_{qubits[0]} Z_{qubits[1]}")

            # Measure
            current_state, outcome = self.measure_gauge_operator(
                current_state, qubits
            )

            if verbose:
                print(f"  Outcome: {'+1' if outcome == +1 else '-1'}")

            correction_applied = False
            if outcome == -1:
                # Apply correction (X on second qubit)
                current_state = self.apply_correction(current_state, qubits[1])
                correction_applied = True
                if verbose:
                    print(f"  Applied X_{qubits[1]} correction")

            measurements.append(GaugeMeasurement(
                operator_qubits=qubits,
                operator_type='Z',
                outcome=GaugeFixingOutcome(outcome),
                correction_applied=correction_applied
            ))

        if verbose:
            print("\n" + "=" * 50)
            print("Gauge fixing complete!")

        return current_state, measurements

    def verify_gauge_fixed(self, state: np.ndarray) -> Dict[str, bool]:
        """
        Verify that all Z-gauge operators have eigenvalue +1.
        """
        results = {}

        for qubits in self.z_gauge_operators:
            G = self.gauge_operator(qubits, 'Z')
            expectation = np.real(np.vdot(state, G @ state))
            is_fixed = np.isclose(expectation, 1.0)
            results[f"Z_{qubits[0]}Z_{qubits[1]}"] = is_fixed

        return results


def create_random_bacon_shor_state() -> np.ndarray:
    """
    Create a random logical state in the Bacon-Shor code.

    For simplicity, we create a state that's in the code space
    but with random gauge components.
    """
    # For demonstration, create a simple state
    # In practice, would properly encode into Bacon-Shor

    n = 9
    dim = 2**n

    # Start with |0...0⟩
    state = np.zeros(dim, dtype=complex)
    state[0] = 1

    # Apply some gauge operations to randomize
    # This is a simplified version

    return state / np.linalg.norm(state)


def fault_tolerant_gauge_fixing_demo():
    """Demonstrate fault-tolerant gauge fixing with repeated measurements."""
    print("\n" + "=" * 60)
    print("Fault-Tolerant Gauge Fixing")
    print("=" * 60)

    print("\nProtocol with repeated measurements:")
    print("-" * 40)

    n_repetitions = 3

    print(f"\nFor each gauge operator G_Z:")
    print(f"  1. Repeat measurement {n_repetitions} times")
    print(f"  2. Take majority vote")
    print(f"  3. Apply correction if majority is -1")

    print("\nExample sequence:")
    print("  Measure G_Z^(1): outcomes = [+1, +1, -1] -> majority = +1 (no correction)")
    print("  Measure G_Z^(2): outcomes = [-1, -1, +1] -> majority = -1 (apply X)")
    print("  ...")

    print("\nFault tolerance property:")
    print("  - Single measurement error is outvoted")
    print("  - Single gate error during measurement is detected")
    print("  - Overall error rate: O(p^2) instead of O(p)")


def gauge_switching_demo():
    """Demonstrate gauge switching between different fixings."""
    print("\n" + "=" * 60)
    print("Gauge Switching for Transversal Gates")
    print("=" * 60)

    print("\nScenario: Code with two gauge fixings")
    print("-" * 40)

    print("\nGauge Fixing A:")
    print("  Stabilizers: S_A = {row stabilizers}")
    print("  Transversal: {H, S, CNOT} (Clifford)")

    print("\nGauge Fixing B:")
    print("  Stabilizers: S_B = {column stabilizers}")
    print("  Transversal: {T, CNOT}")

    print("\nTo apply T gate (starting in Fixing A):")
    print("-" * 40)

    steps = [
        "1. Start: |ψ⟩ in gauge-fixed code A",
        "2. Unfix: Measure X-gauge operators (randomizes Z-gauge)",
        "3. Refix B: Measure B's Z-gauge operators",
        "4. Apply T: T^⊗n transversally",
        "5. Unfix: Measure X-gauge operators again",
        "6. Refix A: Measure A's Z-gauge operators",
        "7. Result: T|ψ⟩ in gauge-fixed code A",
    ]

    for step in steps:
        print(f"  {step}")

    print("\nCircuit depth: O(1) per T gate!")
    print("Compare: Magic state distillation is O(log(1/ε))")


def paetznick_reichardt_overview():
    """Overview of the Paetznick-Reichardt universality result."""
    print("\n" + "=" * 60)
    print("Paetznick-Reichardt Universality (2013)")
    print("=" * 60)

    print("\nTheorem Statement:")
    print("-" * 40)
    print("There exists a subsystem code and gauge-fixing protocol")
    print("achieving universal fault-tolerant quantum computation")
    print("using ONLY transversal gates and error correction.")

    print("\nKey Elements:")
    print("-" * 40)
    print("  1. 3D subsystem color code")
    print("  2. Two gauge fixings: Clifford-transversal and T-transversal")
    print("  3. Gauge switching between fixings")

    print("\nImplications:")
    print("-" * 40)
    print("  - No magic state distillation needed!")
    print("  - Constant depth per non-Clifford gate")
    print("  - Alternative path to fault-tolerant universality")

    print("\nCaveats:")
    print("-" * 40)
    print("  - Requires 3D architecture (challenging for 2D chips)")
    print("  - Gauge switching has its own overhead")
    print("  - Practical threshold may differ from theory")


def resource_comparison():
    """Compare resources for gauge fixing vs magic states."""
    print("\n" + "=" * 60)
    print("Resource Comparison: Gauge Fixing vs Magic States")
    print("=" * 60)

    print("\nFor implementing one logical T gate:")
    print("-" * 40)

    print("\nMagic State Distillation:")
    print(f"  Physical qubits: ~100-1000 (for distillation)")
    print(f"  Circuit depth: O(log(1/ε)) ≈ 50-200")
    print(f"  Success probability: ~85-99%")
    print(f"  Ancilla overhead: High")

    print("\nGauge Fixing (Paetznick-Reichardt):")
    print(f"  Physical qubits: ~n (code block size)")
    print(f"  Circuit depth: O(1) ≈ 10-20")
    print(f"  Success probability: Deterministic")
    print(f"  Ancilla overhead: Moderate (for syndrome)")

    print("\nTrade-offs:")
    print("-" * 40)
    print("  Gauge fixing: Better depth, requires 3D connectivity")
    print("  Magic states: Works in 2D, higher qubit/time cost")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Day 872: Gauge Fixing Protocols")
    print("=" * 60)

    # Create protocol
    protocol = GaugeFixingProtocol()

    # Create test state
    state = create_random_bacon_shor_state()

    print("\nDemonstrating gauge fixing on Bacon-Shor code...")
    fixed_state, measurements = protocol.gauge_fix_all(state, verbose=True)

    # Verify
    print("\nVerifying gauge-fixed state:")
    verification = protocol.verify_gauge_fixed(fixed_state)
    for op, is_fixed in verification.items():
        status = "FIXED (+1)" if is_fixed else "NOT FIXED"
        print(f"  {op}: {status}")

    # Additional demonstrations
    fault_tolerant_gauge_fixing_demo()
    gauge_switching_demo()
    paetznick_reichardt_overview()
    resource_comparison()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Gauge fixing projects subsystem code to stabilizer code")
    print("2. Different gauge fixings yield different transversal gates")
    print("3. Gauge switching enables universality without magic states")
    print("4. Paetznick-Reichardt: O(1) depth per T gate via gauge fixing")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Gauge fixing projection | $P_g = \prod_i \frac{I + (-1)^{g_i} G_i}{2}$ |
| Gauge orbit | $\mathcal{O}(|\psi\rangle) = \{G|\psi\rangle : G \in \mathcal{G}\}$ |
| Correction for $-1$ outcome | Apply $X$ on one qubit of $G_Z = Z_i Z_j$ |
| Fault-tolerant fixing | Repeat 3x, majority vote |
| Gauge switching | Fixing A $\to$ Unfix $\to$ Fixing B |

### Main Takeaways

1. **Gauge fixing** measures gauge operators to project into a specific sector
2. **Logical information is preserved** during gauge fixing
3. **Different gauge fixings** of the same subsystem code yield different stabilizer codes
4. **Complementary transversal gates** can be accessed by switching gauge fixings
5. **Paetznick-Reichardt** showed universality via gauge fixing alone
6. **O(1) depth** per non-Clifford gate (vs. O(log(1/ε)) for magic states)

---

## Daily Checklist

- [ ] I can define gauge fixing as a projection operation
- [ ] I can design a measurement circuit for gauge operators
- [ ] I understand how gauge fixing converts Bacon-Shor to Shor code
- [ ] I can explain how different gauge fixings yield different transversal gates
- [ ] I understand the Paetznick-Reichardt universality construction
- [ ] I can compare gauge fixing with magic state distillation

---

## Preview: Day 873

Tomorrow we explore **Lattice Surgery as Code Switching**:

- How merge/split operations can be viewed as code transitions
- The topological interpretation of lattice surgery
- Surface code to color code conversion
- Generalized lattice surgery framework
- Connection to subsystem lattice surgery

Lattice surgery provides a practical implementation of code switching ideas!
