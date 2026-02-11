# Day 859: Circumventing Eastin-Knill

## Overview

**Day:** 859 of 1008
**Week:** 123 (Transversal Gates & Eastin-Knill)
**Month:** 31 (Fault-Tolerant Quantum Computing I)
**Topic:** Methods to Achieve Universal Fault-Tolerant Computation Despite Eastin-Knill

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Magic states and gate teleportation |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Code switching and gauge fixing |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Implementation analysis |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Explain** how magic states enable non-transversal gates
2. **Describe** the gate teleportation protocol for T-gate injection
3. **Analyze** code switching between complementary codes
4. **Understand** gauge fixing in subsystem codes for transversal gates
5. **Compare** different approaches to circumventing Eastin-Knill
6. **Evaluate** resource costs for each universality method

---

## The Universality Challenge

### Eastin-Knill Recap

**The Constraint:** No error-correcting code has a transversal universal gate set.

**The Goal:** Achieve universal fault-tolerant quantum computation anyway.

**The Solution:** Use non-transversal methods for the "missing" gates.

### Three Main Approaches

| Approach | Key Idea | Overhead | Complexity |
|----------|----------|----------|------------|
| Magic States | Inject pre-prepared states | $O(\log^c(1/\epsilon))$ | High |
| Code Switching | Change codes for different gates | Constant per switch | Medium |
| Gauge Fixing | Change gauge within subsystem code | Low | Specialized |

---

## Approach 1: Magic States

### The Magic State Paradigm

**Key Insight:** We can implement non-Clifford gates using:
1. A specially prepared "magic" state
2. Clifford operations (which ARE transversal)
3. Measurement and classical feedback

### The T-State

The **T-magic state** (or $|T\rangle$ state):

$$\boxed{|T\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + e^{i\pi/4}|1\rangle\right) = T|+\rangle}$$

**Property:** Given $|T\rangle$ and Clifford operations, we can implement the T gate.

### Gate Teleportation Protocol

**Protocol for T-gate via magic state:**

```
Input: |ψ⟩ = α|0⟩ + β|1⟩, Magic state: |T⟩

Circuit:
         ┌───┐
|ψ⟩  ────┤   ├──────●─────M────────
         │ T │      │           │
|T⟩  ────┤   ├──────X─────┬─────┴── → S^m |ψ'⟩
         └───┘            │
                          v
                    Classical
                    Correction

Output: T|ψ⟩ (after S^m correction if m=1)
```

**Step-by-step:**

1. **CNOT from $|\psi\rangle$ to $|T\rangle$:**
   $$\text{CNOT}|\psi\rangle|T\rangle = \frac{1}{\sqrt{2}}[\alpha|0\rangle(|0\rangle + e^{i\pi/4}|1\rangle) + \beta|1\rangle(|1\rangle + e^{i\pi/4}|0\rangle)]$$

2. **Measure first qubit in X basis:**
   - Outcome $+$: Get $T|\psi\rangle$ on second qubit
   - Outcome $-$: Get $ST|\psi\rangle$ on second qubit

3. **Correction:** Apply $S^\dagger$ if needed.

### Mathematical Derivation

Starting with $|\psi\rangle|T\rangle$:

$$|\Psi\rangle = (\alpha|0\rangle + \beta|1\rangle) \otimes \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$

After CNOT:
$$= \frac{\alpha}{\sqrt{2}}|0\rangle(|0\rangle + e^{i\pi/4}|1\rangle) + \frac{\beta}{\sqrt{2}}|1\rangle(|1\rangle + e^{i\pi/4}|0\rangle)$$

Rewrite in X basis for first qubit ($|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$):

$$= \frac{1}{2}|+\rangle \otimes (\alpha + \beta e^{i\pi/4})|0\rangle + (\alpha e^{i\pi/4} + \beta)|1\rangle) + ...$$

After measurement and simplification:
- If $|+\rangle$: Second qubit is $\alpha|0\rangle + \beta e^{i\pi/4}|1\rangle = T|\psi\rangle$ $\checkmark$

### Why This Works

The magic state "contains" the T-gate information. Gate teleportation transfers this to the computational qubit using only Clifford operations.

**Crucially:** All operations on the encoded qubits are Clifford (transversal), only the magic state preparation requires non-Clifford resources.

---

## Magic State Distillation

### The Problem

Preparing perfect $|T\rangle$ states is as hard as doing T gates directly!

**Solution:** Start with noisy magic states and **distill** them to high fidelity.

### The 15-to-1 Protocol

**Bravyi-Kitaev (2005):**

$$\boxed{15 \text{ noisy } |T\rangle \text{ states} \rightarrow 1 \text{ clean } |T\rangle \text{ state}}$$

**Error suppression:**
$$\epsilon_{out} \approx 35 \epsilon_{in}^3$$

If input error $\epsilon_{in} = 1\%$, output error $\epsilon_{out} \approx 0.0035\%$.

### Distillation Circuit

The 15-to-1 protocol uses:
1. 15 input $|T\rangle$ states with error rate $\epsilon$
2. Clifford gates to entangle them
3. Syndrome measurement
4. Post-selection on correct syndrome
5. Output: 1 state with error $O(\epsilon^3)$

**Key Insight:** The protocol uses the [[15,1,3]] Reed-Muller code, which has transversal T!

### Iterated Distillation

For target error $\epsilon_{target}$:

$$\boxed{\text{Rounds needed} = O\left(\log \log \frac{1}{\epsilon_{target}}\right)}$$

**Resource scaling:**
$$\text{Total magic states} = O\left(\log^c \frac{1}{\epsilon_{target}}\right)$$

where $c \approx 1.6$ for 15-to-1.

---

## Approach 2: Code Switching

### The Complementarity Principle

Different codes have different transversal gates:

| Code | Transversal Non-Clifford | Missing Clifford |
|------|--------------------------|------------------|
| Steane [[7,1,3]] | None | None (has all Clifford) |
| Reed-Muller [[15,1,3]] | T | H, S |

**Idea:** Switch between codes to access all gates!

### Code Switching Protocol

**Goal:** Implement T on data encoded in Steane code.

**Protocol:**
1. Start with data in Steane code (has transversal Clifford)
2. **Switch** to Reed-Muller code (has transversal T)
3. Apply transversal T
4. **Switch** back to Steane code
5. Continue with Clifford gates

### Implementing the Switch

**Steane to Reed-Muller:**

This is NOT a simple operation! Options include:

**Option A: Decode and Re-encode**
1. Decode Steane code to physical qubit
2. Re-encode in Reed-Muller code
3. Problem: Exposes data to errors during transition!

**Option B: Lattice Surgery / Code Deformation**
1. Gradually deform code structure
2. Maintain protection throughout
3. More complex but fault-tolerant

**Option C: Concatenation**
1. Concatenate codes
2. Apply T at appropriate level
3. High overhead but well-understood

### Fault-Tolerant Code Switching

**Key Challenge:** The switching process itself must not spread errors.

**Approach:** Use intermediate codes or gauge fixing.

**Paetznick-Reichardt (2013):** Developed fault-tolerant code switching protocols achieving:
- Constant overhead per switch
- No loss of error correction capability

---

## Approach 3: Gauge Fixing

### Subsystem Codes Review

A **subsystem code** encodes logical qubits into:
$$\mathcal{H} = \mathcal{H}_L \otimes \mathcal{H}_G \otimes \mathcal{H}_E$$

where:
- $\mathcal{H}_L$: Logical (protected) space
- $\mathcal{H}_G$: Gauge (unprotected but controlled)
- $\mathcal{H}_E$: Error (syndrome) space

### Gauge Operators

**Gauge group:** Operators that act on $\mathcal{H}_G$ but leave $\mathcal{H}_L$ invariant.

**Key property:** Different "gauge fixings" correspond to different stabilizer codes!

### Gauge Fixing for Universality

**Bombin (2015):** In 3D color codes:

1. **Gauge A:** Transversal Clifford gates
2. **Gauge B:** Transversal T gate

By **gauge fixing** (measuring gauge operators), we can switch between these!

### Protocol

1. **Fix to Gauge A:** Measure gauge operators to project to Clifford-transversal code
2. **Apply Clifford gates:** Transversally
3. **Fix to Gauge B:** Measure different gauge operators
4. **Apply T gate:** Now transversal!
5. **Fix back to Gauge A:** Continue computation

**Advantage:** The gauge fixing is a local operation, not a full code switch.

---

## Approach 4: Concatenation

### Hierarchical Codes

**Idea:** Use different codes at different levels of concatenation.

$$\text{Level 0: Physical qubits}$$
$$\text{Level 1: Inner code (e.g., Steane)}$$
$$\text{Level 2: Outer code (e.g., Reed-Muller)}$$

### Concatenated Transversal Gates

At each level, use the appropriate transversal gates:
- Level 1 Steane: Transversal Clifford
- Level 2 RM: Transversal T (at the logical level of Level 1)

**Result:** By combining levels, achieve universality!

### Jochym-O'Connor and Laflamme (2014)

**Theorem:** For any desired gate $G$, there exists a concatenation scheme where $G$ is transversal at some level.

**Cost:** Overhead grows with concatenation depth, but:
- Each gate type has bounded depth
- Total overhead is polynomial in $\log(1/\epsilon)$

---

## Comparison of Approaches

### Resource Comparison

| Approach | Space Overhead | Time Overhead | Complexity |
|----------|----------------|---------------|------------|
| Magic States | $O(\log^c(1/\epsilon))$ | $O(\log(1/\epsilon))$ | Distillation factories |
| Code Switching | $O(1)$ per switch | $O(1)$ per switch | Gauge measurement |
| Gauge Fixing | $O(1)$ | $O(1)$ | 3D codes required |
| Concatenation | $O(n^k)$ | $O(n^k)$ | Deep circuits |

### When to Use Each

**Magic States:**
- Standard choice for most architectures
- Well-understood overhead
- Works with 2D codes

**Code Switching:**
- When distillation is too expensive
- With compatible code families
- Moderate T-count algorithms

**Gauge Fixing:**
- With subsystem codes (especially 3D)
- Low-overhead T gates
- Experimental architectures supporting 3D

**Concatenation:**
- For theoretical constructions
- Very high precision requirements
- When simpler methods fail

---

## Modern Developments (2020s)

### Improved Distillation

**Litinski (2019):** "Magic State Distillation: Not as Costly as You Think"
- Amortized cost analysis
- Pipelining distillation
- Practical overhead: ~1000 physical qubits per logical T

### qLDPC and Beyond

**Quantum Low-Density Parity-Check (qLDPC) codes:**
- Constant rate: $k/n = \Theta(1)$
- Potentially better transversal properties
- Active research area

### Floquet Codes

**Hastings-Haah (2021):**
- Dynamically changing codes
- Different transversal gates at different times
- Potentially simpler universality

---

## Worked Examples

### Example 1: Magic State Injection Resource Count

**Problem:** How many physical qubits are needed for a fault-tolerant T gate using distillation on a surface code?

**Solution:**

**Step 1: Target error rate**
- Logical circuit needs $\epsilon_T < 10^{-10}$ per T gate (typical for algorithms)

**Step 2: Distillation rounds**
- 15-to-1 protocol: $\epsilon_{out} = 35\epsilon_{in}^3$
- Start with $\epsilon_{in} = 0.1\%$ (from error correction)
- Round 1: $\epsilon = 35 \times (10^{-3})^3 = 3.5 \times 10^{-8}$
- Round 2: $\epsilon = 35 \times (3.5 \times 10^{-8})^3 \approx 10^{-21}$
- Need 2 rounds.

**Step 3: Magic state count**
- Round 1: 15 input states
- Round 2: 15 outputs from Round 1 needed
- Total: $15 \times 15 = 225$ raw magic states per T gate

**Step 4: Surface code overhead**
- Each magic state: ~10 surface code patches
- Each patch: ~$d^2 = 25$ physical qubits (for $d=5$)
- Total: $225 \times 10 \times 25 \approx 56,000$ qubit-cycles per T gate

This is the overhead Eastin-Knill forces upon us!

### Example 2: Code Switching for Single T Gate

**Problem:** Design a code switching protocol from [[7,1,3]] Steane to [[15,1,3]] RM for one T gate.

**Solution:**

**Protocol:**

1. **Prepare ancilla:** 15-qubit RM-encoded $|\bar{0}\rangle$
2. **Lattice surgery:** Merge Steane logical with RM ancilla
   - Measurement-based state transfer
   - Takes $O(d)$ time steps
3. **Apply transversal T:** $T^{\otimes 15}$ on RM code
4. **Reverse surgery:** Transfer back to Steane encoding
5. **Discard RM ancilla**

**Error analysis:**
- Surgery operations are Clifford: fault-tolerant
- T application is transversal: fault-tolerant
- Overall: constant overhead, full protection

### Example 3: Gauge Fixing Sequence

**Problem:** In a 3D color code with gauge operators, outline the gauge fixing sequence for implementing $H$, $T$, $H$.

**Solution:**

**3D Color Code Properties:**
- Gauge A (primal): Transversal $H$, $S$, $CNOT$
- Gauge B (dual): Transversal $T$, $X$, $Z$

**Sequence:**

1. **Fix to Gauge A** (measure gauge operators projecting to primal)
2. **Apply $\bar{H} = H^{\otimes n}$** (transversal)
3. **Fix to Gauge B** (measure different gauge operators)
4. **Apply $\bar{T} = T^{\otimes n}$** (transversal)
5. **Fix to Gauge A** (return to primal gauge)
6. **Apply $\bar{H} = H^{\otimes n}$** (transversal)

Each gauge fixing requires $O(1)$ syndrome measurements.

---

## Practice Problems

### Level 1: Direct Application

**P1.1** The $|H\rangle$ magic state is $\cos(\pi/8)|0\rangle + \sin(\pi/8)|1\rangle$. Show that $H|H\rangle = |H\rangle$ up to a phase.

**P1.2** How many rounds of 15-to-1 distillation are needed to achieve $\epsilon < 10^{-15}$ starting from $\epsilon = 1\%$?

**P1.3** List the gates that are transversal on the [[7,1,3]] Steane code and the [[15,1,3]] Reed-Muller code. What gates require code switching?

### Level 2: Intermediate

**P2.1** Design a gate teleportation circuit for the $S$ gate using an appropriate magic state. What state is needed?

**P2.2** In the 15-to-1 distillation protocol, why does the error go as $\epsilon^3$ rather than $\epsilon^2$ or $\epsilon^{15}$?

**P2.3** Explain why gauge fixing in subsystem codes is "cheaper" than full code switching. What operations are avoided?

### Level 3: Challenging

**P3.1** Prove that the gate teleportation protocol for T is correct by explicit calculation of the post-measurement state.

**P3.2** Design a concatenated code scheme where Toffoli (CCZ) is transversal at some level.

**P3.3** Analyze the threshold for distillation: below what input error rate does distillation always succeed?

---

## Computational Lab

```python
"""
Day 859: Circumventing Eastin-Knill
=====================================

Implementing magic state protocols and code switching analysis.
"""

import numpy as np
from typing import Tuple, List

# Quantum gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Computational basis
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ketplus = (ket0 + ket1) / np.sqrt(2)
ketminus = (ket0 - ket1) / np.sqrt(2)


def tensor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tensor product."""
    return np.kron(a, b)


def magic_t_state() -> np.ndarray:
    """Create the |T⟩ magic state."""
    return T @ ketplus


def gate_teleportation_t(psi: np.ndarray, magic: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Implement T gate via gate teleportation.

    Parameters:
    -----------
    psi : np.ndarray
        Input state |psi⟩
    magic : np.ndarray
        Magic state |T⟩

    Returns:
    --------
    output : np.ndarray
        Resulting state (should be T|psi⟩ after correction)
    measurement : int
        Measurement outcome (0 or 1)
    """
    # CNOT gate (control: psi, target: magic)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    # Initial state
    state = tensor(psi, magic)

    # Apply CNOT
    state = CNOT @ state

    # Measure first qubit in X basis
    # Project onto |+⟩ or |-⟩

    # Reshape to 2x2 for partial measurement
    state_reshaped = state.reshape(2, 2)

    # Probability of measuring |+⟩
    proj_plus = np.array([1, 1]) / np.sqrt(2)
    prob_plus = np.abs(proj_plus @ state_reshaped @ np.array([1, 1]) / np.sqrt(2)) ** 2

    # Simulate measurement (for demonstration, compute both outcomes)
    # Project onto |+⟩
    result_plus = proj_plus @ state_reshaped
    result_plus = result_plus / np.linalg.norm(result_plus)

    # Project onto |-⟩
    proj_minus = np.array([1, -1]) / np.sqrt(2)
    result_minus = proj_minus @ state_reshaped
    if np.linalg.norm(result_minus) > 1e-10:
        result_minus = result_minus / np.linalg.norm(result_minus)
    else:
        result_minus = np.array([0, 0], dtype=complex)

    return result_plus, result_minus


def verify_gate_teleportation():
    """Verify that gate teleportation implements T correctly."""
    print("=" * 60)
    print("Verifying Gate Teleportation for T Gate")
    print("=" * 60)

    # Test on various input states
    test_states = [
        (ket0, "|0⟩"),
        (ket1, "|1⟩"),
        (ketplus, "|+⟩"),
        (ketminus, "|-⟩"),
        ((ket0 + 1j * ket1) / np.sqrt(2), "|i⟩")
    ]

    magic = magic_t_state()
    print(f"\nMagic state |T⟩ = {magic.round(4)}")

    for psi, name in test_states:
        result_plus, result_minus = gate_teleportation_t(psi, magic)

        # Expected result: T|psi⟩
        expected = T @ psi

        print(f"\nInput: {name}")
        print(f"  T|{name}⟩ expected: {expected.round(4)}")
        print(f"  Result (+): {result_plus.round(4)}")

        # Check if result matches (up to phase)
        overlap = np.abs(np.vdot(result_plus, expected))
        print(f"  Overlap with T|{name}⟩: {overlap:.6f}")

        if overlap > 0.999:
            print(f"  MATCH!")
        else:
            # Check if S correction needed
            expected_s = S @ T @ psi
            overlap_s = np.abs(np.vdot(result_minus, expected_s))
            print(f"  After S correction overlap: {overlap_s:.6f}")


def distillation_analysis():
    """Analyze magic state distillation protocols."""
    print("\n" + "=" * 60)
    print("Magic State Distillation Analysis")
    print("=" * 60)

    def fifteen_to_one(eps_in: float) -> float:
        """15-to-1 distillation output error."""
        return 35 * eps_in ** 3

    def bravyi_haah(eps_in: float) -> Tuple[float, int]:
        """Bravyi-Haah protocol: 10 -> 2 with quadratic suppression."""
        eps_out = 10 * eps_in ** 2
        return eps_out, 2

    # Compare protocols
    eps_values = [0.1, 0.01, 0.001]

    print("\n15-to-1 Protocol:")
    print("-" * 40)
    for eps in eps_values:
        rounds = 0
        current = eps
        while current > 1e-15:
            current = fifteen_to_one(current)
            rounds += 1
            if rounds > 10:
                break
        print(f"  eps_in = {eps:.1e}: {rounds} rounds -> eps_out = {current:.2e}")

    # Resource estimation
    print("\nResource Estimation (15-to-1):")
    print("-" * 40)

    target_errors = [1e-6, 1e-10, 1e-15]
    eps_in = 0.01

    for target in target_errors:
        rounds = 0
        current = eps_in
        total_input = 1

        while current > target:
            current = fifteen_to_one(current)
            total_input *= 15
            rounds += 1
            if rounds > 5:
                break

        print(f"  Target {target:.0e}: {rounds} rounds, {total_input} input states")


def code_switching_overhead():
    """Analyze code switching overhead."""
    print("\n" + "=" * 60)
    print("Code Switching Overhead Analysis")
    print("=" * 60)

    # Steane vs Reed-Muller comparison
    codes = {
        'Steane [[7,1,3]]': {
            'n': 7,
            'transversal': ['X', 'Z', 'H', 'S', 'CNOT'],
            'missing': ['T']
        },
        'Reed-Muller [[15,1,3]]': {
            'n': 15,
            'transversal': ['X', 'Z', 'T'],
            'missing': ['H', 'S', 'CNOT']
        }
    }

    print("\nCode Comparison:")
    print("-" * 50)
    for name, info in codes.items():
        print(f"\n{name}:")
        print(f"  Physical qubits: {info['n']}")
        print(f"  Transversal: {', '.join(info['transversal'])}")
        print(f"  Missing: {', '.join(info['missing'])}")

    # Switching cost
    print("\nCode Switching Costs:")
    print("-" * 50)
    print("Steane -> RM -> Steane (for one T gate):")
    print("  - Prepare RM ancilla: O(n) operations")
    print("  - State transfer: O(d) time steps")
    print("  - Apply T: O(1) transversal")
    print("  - Transfer back: O(d) time steps")
    print("  - Total: O(d) time, O(n) space overhead")

    print("\nComparison with Distillation:")
    print("  - Distillation: O(log(1/eps)^c) overhead")
    print("  - Code switching: O(1) overhead per T")
    print("  - For many T gates: switching can be cheaper!")


def gauge_fixing_example():
    """Illustrate gauge fixing concept."""
    print("\n" + "=" * 60)
    print("Gauge Fixing in Subsystem Codes")
    print("=" * 60)

    print("\n3D Color Code Example:")
    print("-" * 50)

    print("""
    Subsystem Structure:
    - Full gauge group: G = <stabilizers> x <gauge generators>
    - Stabilizers: Commuting subset of G

    Gauge Fixing Options:
    """)

    gauges = {
        'Primal (A)': {
            'stabilizers': 'Face operators (X-type)',
            'transversal': ['H', 'S', 'CNOT'],
            'missing': ['T']
        },
        'Dual (B)': {
            'stabilizers': 'Vertex operators (Z-type)',
            'transversal': ['T', 'X', 'Z'],
            'missing': ['H', 'S']
        }
    }

    for name, info in gauges.items():
        print(f"\n  Gauge {name}:")
        print(f"    Stabilizers: {info['stabilizers']}")
        print(f"    Transversal: {', '.join(info['transversal'])}")
        print(f"    Missing: {', '.join(info['missing'])}")

    print("""
    Switching Procedure:
    1. Measure gauge operators to project to desired gauge
    2. Apply transversal gate
    3. Measure different gauge operators to switch gauge
    4. Continue computation

    Advantage: No ancilla codes needed!
    """)


def universal_protocol_summary():
    """Summarize complete universal FT protocol."""
    print("\n" + "=" * 60)
    print("Universal Fault-Tolerant Computation Protocol")
    print("=" * 60)

    print("""
    Standard Approach (Surface Code + Magic States):
    ================================================

    1. ENCODING
       - Encode logical qubits in surface code
       - Surface code has transversal {X, Z, CNOT}

    2. CLIFFORD GATES
       - X, Z: Transversal (apply to all physical qubits)
       - CNOT: Transversal between code patches
       - H, S: Via lattice surgery (non-transversal but FT)

    3. T GATES (non-Clifford)
       - Prepare noisy |T⟩ states
       - Distill using 15-to-1 protocol
       - Inject via gate teleportation
       - Apply S correction if needed

    4. MEASUREMENT
       - Transversal Z measurement
       - Pauli frame tracking for corrections

    Resource Budget:
    ----------------
    - Per Clifford gate: O(1) physical operations
    - Per T gate: ~1000-10000 physical qubits (distillation)
    - Threshold: ~0.1-1% physical error rate
    """)


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Day 859: Circumventing Eastin-Knill")
    print("=" * 60)

    # Part 1: Gate teleportation
    verify_gate_teleportation()

    # Part 2: Distillation analysis
    distillation_analysis()

    # Part 3: Code switching
    code_switching_overhead()

    # Part 4: Gauge fixing
    gauge_fixing_example()

    # Part 5: Summary
    universal_protocol_summary()

    print("\n" + "=" * 60)
    print("Key Takeaway:")
    print("Despite Eastin-Knill, we CAN achieve universality!")
    print("The cost is in resources, not possibility.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula/Protocol |
|---------|------------------|
| T-magic state | $\|T\rangle = T\|+\rangle = \frac{1}{\sqrt{2}}(\|0\rangle + e^{i\pi/4}\|1\rangle)$ |
| 15-to-1 distillation | $\epsilon_{out} = 35\epsilon_{in}^3$ |
| Gate teleportation | CNOT + measurement + Clifford correction |
| Code switching | Transfer between codes with complementary transversal gates |
| Gauge fixing | Measure gauge operators to switch stabilizer structure |

### Approach Comparison

| Method | Pros | Cons |
|--------|------|------|
| Magic States | Universal, well-understood | High overhead |
| Code Switching | Low per-gate overhead | Complex transitions |
| Gauge Fixing | Very low overhead | Requires 3D codes |
| Concatenation | Systematic | Deep circuits |

### Main Takeaways

1. **Magic state injection** is the standard method for non-Clifford gates
2. **Distillation** purifies noisy magic states with polynomial overhead
3. **Code switching** offers lower overhead for T-heavy algorithms
4. **Gauge fixing** in subsystem codes provides elegant universality
5. **All methods** require overhead beyond pure transversal gates
6. The **Eastin-Knill theorem** shapes the entire fault-tolerant architecture

---

## Daily Checklist

- [ ] I can explain gate teleportation for the T gate
- [ ] I understand the 15-to-1 distillation protocol
- [ ] I can describe code switching between Steane and Reed-Muller
- [ ] I understand gauge fixing in subsystem codes
- [ ] I can compare resource costs of different approaches
- [ ] I know which method is appropriate for different scenarios

---

## Preview: Day 860

Tomorrow is our **Computational Lab Day**:

- Implement complete transversal gate analysis
- Verify Eastin-Knill constraints numerically
- Simulate distillation protocols
- Compare overhead of different universality approaches
- Build tools for fault-tolerant circuit design

Hands-on experience with the week's concepts!
