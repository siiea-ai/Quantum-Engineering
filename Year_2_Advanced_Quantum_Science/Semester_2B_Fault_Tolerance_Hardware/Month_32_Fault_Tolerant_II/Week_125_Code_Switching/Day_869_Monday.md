# Day 869: Code Switching Motivation

## Overview

**Day:** 869 of 1008
**Week:** 125 (Code Switching & Gauge Fixing)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Topic:** Why Switch Codes? Complementary Transversal Gate Sets and Universality via Switching

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Universality problem and code switching theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Complementary gate sets and protocol design |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational exploration |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Explain** why the Eastin-Knill theorem necessitates code switching or magic states
2. **Identify** complementary transversal gate sets across different quantum codes
3. **Describe** the basic strategy for achieving universality via code switching
4. **Compare** code switching with magic state distillation as paths to universality
5. **Analyze** fault tolerance requirements for code conversion protocols
6. **Outline** the historical development of code switching techniques

---

## Motivation: The Universality Challenge

### Recap: The Eastin-Knill Barrier

From Week 123, we established the **Eastin-Knill theorem**:

> **Theorem (Eastin-Knill, 2009):** No quantum error-correcting code can have a transversal implementation of a universal gate set.

This poses a fundamental challenge: if we want **fault-tolerant universal quantum computation**, we cannot rely solely on transversal gates within a single code.

### Two Paths to Universality

Given Eastin-Knill, there are two primary approaches:

$$\boxed{\text{Universality} = \begin{cases} \text{Transversal Cliffords} + \text{Magic State Injection} \\ \text{Code Switching between complementary codes} \end{cases}}$$

**Path 1: Magic States (Weeks 121-122)**
- Use transversal Clifford gates (easy, fault-tolerant)
- Implement T-gate via magic state injection
- Requires costly magic state distillation

**Path 2: Code Switching (This Week)**
- Use different codes with different transversal gates
- Switch between codes to access complementary gate sets
- No magic state distillation required

---

## Core Theory

### The Key Insight: Complementary Transversal Gates

Different quantum error-correcting codes have **different transversal gate sets**:

| Code | Parameters | Transversal Gates | Missing for Universality |
|------|------------|-------------------|-------------------------|
| Steane | [[7,1,3]] | H, S, CNOT (Clifford) | T |
| Reed-Muller | [[15,1,3]] | T, CNOT | H, S |
| Color Code (2D) | Various | H, S, CNOT | T |
| Color Code (3D) | Various | H, S, CNOT, CCZ | Just T technically |
| Triorthogonal | [[15,1,3]] | T | H |

**Critical Observation:**
$$\text{Steane transversal} = \{H, S, \text{CNOT}\}$$
$$\text{Reed-Muller transversal} = \{T, \text{CNOT}\}$$
$$\text{Steane} \cup \text{Reed-Muller} = \{H, S, T, \text{CNOT}\} = \text{Universal!}$$

### Definition: Code Switching

**Definition:** A **code switching protocol** is a procedure that converts a quantum state encoded in code $\mathcal{C}_1$ to the same logical state encoded in code $\mathcal{C}_2$:

$$\boxed{|\psi_L\rangle_{\mathcal{C}_1} \xrightarrow{\text{switch}} |\psi_L\rangle_{\mathcal{C}_2}}$$

For fault tolerance, this conversion must:
1. Preserve the logical information
2. Not spread errors uncontrollably
3. Have bounded failure probability

### Formal Requirements for Code Switching

**Theorem (Fault-Tolerant Switching):** A code switching protocol from $\mathcal{C}_1 = [[n_1, k, d_1]]$ to $\mathcal{C}_2 = [[n_2, k, d_2]]$ is **fault-tolerant** if:

1. **Logical preservation:** The logical Hilbert space is preserved
$$\mathcal{H}_L^{(1)} \cong \mathcal{H}_L^{(2)} \cong (\mathbb{C}^2)^{\otimes k}$$

2. **Error bounded growth:** A single fault creates at most one error in each code block
$$\text{wt}(E_{\text{out}}) \leq \text{wt}(E_{\text{in}}) + 1$$

3. **Correctability:** Output errors are within the correction capability of $\mathcal{C}_2$

### The Switching Strategy

**Protocol Outline:**

```
1. Start with |ψ_L⟩ encoded in C₁
2. Apply transversal gates available in C₁
3. Perform fault-tolerant code switch: C₁ → C₂
4. Apply transversal gates available in C₂
5. Switch back if needed: C₂ → C₁
6. Repeat as computation requires
```

**Example Sequence for Universal Computation:**

$$|0_L\rangle_{\text{Steane}} \xrightarrow{H^{\otimes 7}} |+_L\rangle_{\text{Steane}} \xrightarrow{\text{switch}} |+_L\rangle_{\text{RM}} \xrightarrow{T^{\otimes 15}} T|+_L\rangle_{\text{RM}} \xrightarrow{\text{switch}} T|+_L\rangle_{\text{Steane}}$$

---

## Why Not Just Use Magic States?

### Magic State Distillation Costs

From Week 122, magic state distillation has significant overhead:

**15-to-1 Protocol:**
- Input: 15 noisy $|T\rangle$ states
- Output: 1 cleaner $|T\rangle$ state
- Error reduction: $\epsilon \to 35\epsilon^3$
- Resource cost: $O(\log^{\gamma}(1/\epsilon))$ for target error $\epsilon$

For practical error rates, this can require:
- **Thousands of physical qubits** per logical T-gate
- **Deep circuits** for multi-level distillation
- **High latency** due to sequential distillation

### Code Switching Advantages

**Potential Benefits:**
1. **Lower qubit overhead:** Direct encoding rather than distillation
2. **Simpler circuits:** Encoding circuits can be shallower
3. **Parallelization:** Multiple T-gates via parallel RM blocks
4. **Deterministic:** No probabilistic distillation

**Trade-offs:**
1. **Larger code blocks:** RM needs 15 qubits vs. Steane's 7
2. **Switching circuit complexity:** Must be fault-tolerant
3. **Error accumulation:** Each switch can introduce errors

### Quantitative Comparison

**Resource Comparison for One Logical T-gate:**

| Method | Physical Qubits | Circuit Depth | Success Probability |
|--------|-----------------|---------------|---------------------|
| Magic State (1 level) | ~100 | ~50 | >99% |
| Magic State (2 levels) | ~1000 | ~200 | >99.9% |
| Code Switching | ~22 (7+15) | ~30 | Deterministic |

*Note: Actual numbers depend on error rates and specific protocols.*

---

## Historical Development

### Early Ideas (2013-2014)

**Paetznick & Reichardt (2013):**
- Proposed using gauge fixing for universality
- Showed subsystem codes can achieve universal transversal gates

**Anderson, Duclos-Cianci & Svore (2014):**
- First explicit Steane ↔ Reed-Muller switching protocol
- Proved fault tolerance of the conversion
- Established theoretical foundation

### Theoretical Advances (2015-2020)

**Bombin (2015):**
- Gauge color codes with universal transversal gates
- Connected gauge fixing to code switching

**Kubica & Beverland (2015):**
- 3D color codes with transversal CCZ
- Alternative path to universality

### Experimental Era (2024-2025)

**Quantinuum Demonstration (2024):**
- First experimental fault-tolerant code switch
- Steane [[7,1,3]] ↔ Reed-Muller [[15,1,3]]
- Achieved magic state fidelity: $1 - 5.1 \times 10^{-4}$
- **Below pseudo-threshold** for T-gate!

This breakthrough showed code switching is not just theoretically interesting but **practically competitive** with magic state distillation.

---

## The Complementarity Principle

### Which Codes Complement Each Other?

For effective code switching, we need codes with:
1. **Complementary transversal gates** (together = universal)
2. **Compatible logical structure** (same $k$)
3. **Efficient conversion** (low-depth circuits)

**Prime Example: Steane + Reed-Muller**

**Steane Code [[7,1,3]]:**
- CSS code from [7,4,3] Hamming code
- Self-dual: $H^{\otimes 7}$ is transversal Hadamard
- Transversal: $\{X, Z, H, S, \text{CNOT}\}$ (full Clifford)

**Reed-Muller Code [[15,1,3]]:**
- CSS code from punctured Reed-Muller codes
- NOT self-dual: H is not transversal
- Transversal: $\{X, Z, T, \text{CNOT}, \text{CCZ}\}$

### The Universal Combination

$$\boxed{\text{Clifford}_{\text{Steane}} + T_{\text{RM}} = \text{Universal}}$$

**Decomposition of any unitary:**
Any single-qubit unitary can be approximated by:
$$U \approx H^{a_1} T^{b_1} H^{a_2} T^{b_2} \cdots$$

Using Solovay-Kitaev, $O(\log^c(1/\epsilon))$ gates suffice for precision $\epsilon$.

---

## Fault Tolerance Analysis

### Error Model During Switching

Consider switching from $\mathcal{C}_1$ to $\mathcal{C}_2$:

**Sources of Error:**
1. **Pre-existing errors** in $\mathcal{C}_1$ (weight $\leq t_1$)
2. **Gate errors** during switching circuit (probability $p$ per gate)
3. **Measurement errors** if measurements are used

**Fault-Tolerance Condition:**
$$P(\text{uncorrectable error in } \mathcal{C}_2) \leq c \cdot p^{t_2+1}$$

for some constant $c$, where $t_2 = \lfloor(d_2-1)/2\rfloor$ is the correction capability.

### Designing Fault-Tolerant Switching

**Key Principles:**

1. **Transversal operations when possible:** Don't spread errors within blocks

2. **Flag qubits:** Detect error spread during non-transversal operations

3. **Verification:** Check that switching succeeded via stabilizer measurements

4. **Recursive structure:** Use fault-tolerant gadgets for each step

### Error Propagation Example

**Dangerous Scenario:**
```
Single error in Steane block
    ↓ (during switch)
Multiple errors in RM block
    ↓ (exceeds correction)
Logical error!
```

**Safe Scenario:**
```
Single error in Steane block
    ↓ (fault-tolerant switch)
Single error in RM block
    ↓ (within correction)
Error corrected!
```

---

## Worked Examples

### Example 1: Gate Count for Universal Circuit

**Problem:** A quantum algorithm requires 100 Hadamard gates and 50 T-gates. Compare the resources needed for:
(a) Magic state injection
(b) Code switching

**Solution:**

**(a) Magic State Approach (Steane code):**

- Hadamards: $100 \times H^{\otimes 7}$ (transversal) = 700 physical H gates
- T-gates: 50 magic states needed
  - Each magic state: ~15 raw states for one level of distillation
  - Total magic state prep: ~750 noisy states, ~50 clean states
  - Injection circuits: 50 teleportation circuits

**Resource tally:**
- 7 data qubits (Steane block)
- ~15 ancilla qubits for distillation
- 700 + 50(injection depth) ~ 700 + 500 = 1200 circuit layers
- Total physical operations: ~2000

**(b) Code Switching Approach:**

- Start in Steane: 7 qubits
- Apply 100 H gates (transversal): 700 physical gates
- Switch to RM: ~50 operations (encoding)
- Apply 50 T gates (transversal on RM): 750 physical gates
- Switch back to Steane: ~50 operations

**Resource tally:**
- 7 + 15 = 22 data qubits maximum
- ~8 ancilla for switching verification
- 700 + 50 + 750 + 50 = 1550 circuit layers
- Total physical operations: ~1800

**Comparison:**
- Code switching uses more data qubits (22 vs 7)
- Code switching has lower ancilla requirements
- Similar total operation counts
- Code switching is deterministic (no distillation failure)

### Example 2: Transversal Gate Verification

**Problem:** Verify that $T^{\otimes 15}$ implements logical $T$ on the [[15,1,3]] Reed-Muller code.

**Solution:**

The [[15,1,3]] RM code has:
- Logical $|0_L\rangle = $ uniform superposition over codewords of weight 0, 4, 8, 12
- Logical $|1_L\rangle = $ uniform superposition over codewords of weight 3, 7, 11, 15

For the logical T gate, we need:
$$T|0_L\rangle = |0_L\rangle$$
$$T|1_L\rangle = e^{i\pi/4}|1_L\rangle$$

**Verification of $T^{\otimes 15}|0_L\rangle = |0_L\rangle$:**

$T|0\rangle = |0\rangle$ (T is diagonal with $T_{00} = 1$)

For a codeword $|c\rangle$ with weight $w$ (where $w \equiv 0 \pmod 4$):
$$T^{\otimes 15}|c\rangle = \prod_{i: c_i=1} e^{i\pi/4} \cdot |c\rangle = e^{i\pi w/4}|c\rangle$$

Since $w \equiv 0 \pmod 4$ for all codewords in $|0_L\rangle$:
$$e^{i\pi w/4} = e^{i\pi \cdot 4k/4} = e^{i\pi k} = (\pm 1)$$

More careful analysis using the triorthogonal property shows this is exactly $+1$.

**Verification of $T^{\otimes 15}|1_L\rangle = e^{i\pi/4}|1_L\rangle$:**

For codewords in $|1_L\rangle$ with weight $w \equiv 3 \pmod 4$:
$$e^{i\pi w/4} = e^{i\pi(4k+3)/4} = e^{i\pi k} \cdot e^{i3\pi/4}$$

The triorthogonal structure ensures the phases align to give $e^{i\pi/4}$ overall.

$$\boxed{T^{\otimes 15} = \bar{T} \text{ on [[15,1,3]] RM code}}$$

### Example 3: Counting Switches for Quantum Circuit

**Problem:** Given a circuit with gate sequence $H-T-H-T-T-S-H-T$, how many code switches are needed?

**Solution:**

Organize gates by which code supports them:

| Gate | Supported by |
|------|--------------|
| H | Steane |
| T | Reed-Muller |
| S | Steane |

**Sequence with code assignments:**
```
H (Steane) → T (RM) → H (Steane) → T (RM) → T (RM) → S (Steane) → H (Steane) → T (RM)
```

**Naive switching (every boundary):**
Steane → RM → Steane → RM (T) → RM (T) → Steane → Steane → RM

Count: 5 switches

**Optimized (minimize switches):**
- Group: H | T | H | TT | SH | T
- Start Steane: H
- Switch to RM: T (switch 1)
- Switch to Steane: H (switch 2)
- Switch to RM: TT (switch 3)
- Switch to Steane: SH (switch 4)
- Switch to RM: T (switch 5)

**Further optimization using commutation:**
The circuit $H-T-H-T-T-S-H-T$ might be recompiled.

Using $HTH = R_X(\pi/4)$ type identities:
$$HTH = \frac{1}{\sqrt{2}}\begin{pmatrix}1+e^{i\pi/4} & 1-e^{i\pi/4}\\ 1-e^{i\pi/4} & 1+e^{i\pi/4}\end{pmatrix}$$

This doesn't directly simplify, but **gate rearrangement** might help:
- If H and T don't need to alternate, we could do: HH (Steane) → TTTT (RM) → S (Steane)

The minimum switches depend on circuit structure.

$$\boxed{\text{Minimum switches} = \text{(number of Steane→RM boundaries)}}$$

---

## Practice Problems

### Level 1: Direct Application

**P1.1** List the transversal gates for each code:
(a) [[5,1,3]] perfect code
(b) [[9,1,3]] Shor code
(c) [[7,1,3]] Steane code
(d) [[15,1,3]] Reed-Muller code

**P1.2** A quantum circuit has 20 Hadamard gates and 10 T-gates. If each code switch has overhead of 30 physical gates, what is the minimum total physical gate count using code switching?

**P1.3** Explain why code switching between Steane and Shor codes does NOT help with universality.

### Level 2: Intermediate

**P2.1** Prove that any CSS code has transversal CNOT between two code blocks.

**P2.2** The [[23,1,7]] Golay code has transversal $\{X, Z, \text{CNOT}\}$. What additional transversal gate would make it universal? Design a hypothetical complementary code.

**P2.3** Analyze the error propagation when a weight-1 X error is present during a Steane→RM code switch. What conditions on the switching circuit ensure the output has at most weight-1 error?

### Level 3: Challenging

**P3.1** Prove that if code $\mathcal{C}_1$ has transversal H and code $\mathcal{C}_2$ has transversal T, then fault-tolerant switching between them enables universal computation (assuming both have transversal CNOT).

**P3.2** Design a code switching protocol that minimizes the number of two-qubit gates. What is the lower bound on CNOT count for Steane→RM switching?

**P3.3** Consider a family of codes $\mathcal{C}_n$ with increasing distance. How does the code switching complexity scale with $n$? Is there a fundamental trade-off between distance and switching efficiency?

---

## Computational Lab

```python
"""
Day 869: Code Switching Motivation
==================================

Exploring the fundamentals of code switching for universal quantum computation.
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from itertools import combinations

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Single-qubit gates
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


class QuantumCode:
    """Represents a quantum error-correcting code with its properties."""

    def __init__(self, name: str, n: int, k: int, d: int,
                 transversal_gates: Set[str]):
        """
        Initialize quantum code.

        Parameters:
        -----------
        name : str
            Code name
        n : int
            Number of physical qubits
        k : int
            Number of logical qubits
        d : int
            Code distance
        transversal_gates : Set[str]
            Set of transversal gate names
        """
        self.name = name
        self.n = n
        self.k = k
        self.d = d
        self.transversal_gates = transversal_gates
        self.t = (d - 1) // 2  # Correction capability

    def __repr__(self):
        return f"{self.name} [[{self.n},{self.k},{self.d}]]"

    def supports(self, gate: str) -> bool:
        """Check if gate is transversal on this code."""
        return gate in self.transversal_gates


# Define standard codes
STEANE_CODE = QuantumCode(
    name="Steane",
    n=7, k=1, d=3,
    transversal_gates={'X', 'Z', 'H', 'S', 'CNOT', 'CZ'}
)

REED_MULLER_CODE = QuantumCode(
    name="Reed-Muller",
    n=15, k=1, d=3,
    transversal_gates={'X', 'Z', 'T', 'CNOT', 'CZ', 'CCZ'}
)

SHOR_CODE = QuantumCode(
    name="Shor",
    n=9, k=1, d=3,
    transversal_gates={'X', 'Z', 'CNOT'}
)

SURFACE_CODE = QuantumCode(
    name="Surface",
    n=17, k=1, d=5,  # Example size
    transversal_gates={'X', 'Z', 'CNOT'}
)

COLOR_CODE_2D = QuantumCode(
    name="Color-2D",
    n=7, k=1, d=3,
    transversal_gates={'X', 'Z', 'H', 'S', 'CNOT'}
)


def is_universal(gates: Set[str]) -> bool:
    """
    Check if a gate set is universal for quantum computation.

    Universal set must include:
    - Clifford group generators: H, S, CNOT
    - At least one non-Clifford gate: T (or equivalent)
    """
    clifford_generators = {'H', 'S', 'CNOT'}
    has_clifford = clifford_generators.issubset(gates)
    has_non_clifford = 'T' in gates or 'CCZ' in gates

    return has_clifford and has_non_clifford


def combined_gate_set(codes: List[QuantumCode]) -> Set[str]:
    """Get the union of transversal gates from multiple codes."""
    combined = set()
    for code in codes:
        combined = combined.union(code.transversal_gates)
    return combined


def find_complementary_codes(codes: List[QuantumCode]) -> List[Tuple[QuantumCode, QuantumCode]]:
    """Find pairs of codes whose combined transversal gates are universal."""
    universal_pairs = []

    for c1, c2 in combinations(codes, 2):
        combined = combined_gate_set([c1, c2])
        if is_universal(combined):
            universal_pairs.append((c1, c2))

    return universal_pairs


def analyze_circuit(gate_sequence: List[str],
                   codes: List[QuantumCode]) -> Dict:
    """
    Analyze a circuit to find optimal code assignment.

    Parameters:
    -----------
    gate_sequence : List[str]
        Sequence of gates in the circuit
    codes : List[QuantumCode]
        Available codes for encoding

    Returns:
    --------
    Dict with analysis results
    """
    results = {
        'total_gates': len(gate_sequence),
        'gate_counts': {},
        'code_support': {},
        'min_switches_naive': 0
    }

    # Count gates
    for gate in gate_sequence:
        results['gate_counts'][gate] = results['gate_counts'].get(gate, 0) + 1

    # Find which codes support each gate
    for gate in set(gate_sequence):
        supporting_codes = [c.name for c in codes if c.supports(gate)]
        results['code_support'][gate] = supporting_codes

    # Count minimum switches (naive: switch whenever needed)
    current_code = None
    for gate in gate_sequence:
        supporting = [c for c in codes if c.supports(gate)]
        if supporting:
            best_code = supporting[0]  # Simplistic choice
            if current_code is not None and current_code != best_code:
                results['min_switches_naive'] += 1
            current_code = best_code

    return results


def optimal_code_assignment(gate_sequence: List[str],
                           steane: QuantumCode = STEANE_CODE,
                           rm: QuantumCode = REED_MULLER_CODE) -> List[Tuple[str, str]]:
    """
    Find optimal code assignment for each gate to minimize switches.
    Uses dynamic programming approach.
    """
    n = len(gate_sequence)
    if n == 0:
        return []

    # dp[i][code] = minimum switches to execute gates 0..i ending in code
    codes = [steane, rm]
    code_names = [c.name for c in codes]

    # Initialize
    INF = float('inf')
    dp = [[INF, INF] for _ in range(n)]

    # First gate
    for j, code in enumerate(codes):
        if code.supports(gate_sequence[0]):
            dp[0][j] = 0

    # Fill DP table
    for i in range(1, n):
        gate = gate_sequence[i]
        for j, code in enumerate(codes):
            if code.supports(gate):
                # Stay in same code
                dp[i][j] = min(dp[i][j], dp[i-1][j])
                # Switch from other code
                other = 1 - j
                dp[i][j] = min(dp[i][j], dp[i-1][other] + 1)

    # Backtrack to find assignment
    min_switches = min(dp[n-1])
    assignment = []

    # Find ending code
    current = 0 if dp[n-1][0] <= dp[n-1][1] else 1
    assignment.append((gate_sequence[n-1], code_names[current]))

    for i in range(n-2, -1, -1):
        gate = gate_sequence[i]
        # Check if we need to switch or stay
        if codes[current].supports(gate) and dp[i][current] == dp[i+1][current]:
            pass  # Stay
        else:
            current = 1 - current  # Switch
        assignment.append((gate, code_names[current]))

    assignment.reverse()
    return assignment


def count_switches(assignment: List[Tuple[str, str]]) -> int:
    """Count number of code switches in an assignment."""
    switches = 0
    for i in range(1, len(assignment)):
        if assignment[i][1] != assignment[i-1][1]:
            switches += 1
    return switches


def visualize_code_comparison():
    """Visualize transversal gate comparison between codes."""
    print("=" * 70)
    print("TRANSVERSAL GATE COMPARISON")
    print("=" * 70)

    all_gates = {'X', 'Z', 'H', 'S', 'T', 'CNOT', 'CZ', 'CCZ'}
    codes = [STEANE_CODE, REED_MULLER_CODE, SHOR_CODE, COLOR_CODE_2D]

    # Header
    header = f"{'Gate':<8}"
    for code in codes:
        header += f"{code.name:<12}"
    print(header)
    print("-" * 70)

    # Each gate
    for gate in sorted(all_gates):
        row = f"{gate:<8}"
        for code in codes:
            status = "Yes" if code.supports(gate) else "No"
            row += f"{status:<12}"
        print(row)

    print("-" * 70)

    # Universality check
    print("\nUniversality Analysis:")
    for code in codes:
        is_univ = is_universal(code.transversal_gates)
        print(f"  {code}: {'UNIVERSAL' if is_univ else 'NOT universal'}")

    # Find complementary pairs
    print("\nComplementary Pairs (Combined = Universal):")
    pairs = find_complementary_codes(codes)
    for c1, c2 in pairs:
        combined = combined_gate_set([c1, c2])
        print(f"  {c1.name} + {c2.name}: {sorted(combined)}")


def demo_circuit_analysis():
    """Demonstrate circuit analysis with code switching."""
    print("\n" + "=" * 70)
    print("CIRCUIT ANALYSIS DEMO")
    print("=" * 70)

    # Example circuit: quantum Fourier transform-like
    circuit = ['H', 'T', 'H', 'T', 'T', 'S', 'H', 'T', 'H', 'S']

    print(f"\nCircuit: {' - '.join(circuit)}")
    print(f"Total gates: {len(circuit)}")

    # Analyze
    analysis = analyze_circuit(circuit, [STEANE_CODE, REED_MULLER_CODE])

    print(f"\nGate counts: {analysis['gate_counts']}")
    print(f"Code support:")
    for gate, codes in analysis['code_support'].items():
        print(f"  {gate}: {codes}")

    # Optimal assignment
    assignment = optimal_code_assignment(circuit)
    switches = count_switches(assignment)

    print(f"\nOptimal code assignment:")
    current_code = None
    for gate, code in assignment:
        marker = " <-- SWITCH" if code != current_code and current_code is not None else ""
        print(f"  {gate}: {code}{marker}")
        current_code = code

    print(f"\nMinimum switches required: {switches}")


def resource_comparison():
    """Compare resource costs of magic states vs code switching."""
    print("\n" + "=" * 70)
    print("RESOURCE COMPARISON: Magic States vs Code Switching")
    print("=" * 70)

    # Parameters
    n_t_gates = 100
    n_clifford_gates = 500

    print(f"\nCircuit requirements:")
    print(f"  Clifford gates: {n_clifford_gates}")
    print(f"  T gates: {n_t_gates}")

    # Magic state approach (Steane code)
    print("\n--- Magic State Approach (Steane [[7,1,3]]) ---")
    ms_data_qubits = 7
    ms_ancilla_per_t = 15  # For 15-to-1 distillation
    ms_total_ancilla = ms_ancilla_per_t  # Reused
    ms_distillation_depth = 50  # Per T-gate
    ms_injection_depth = 10

    ms_clifford_ops = n_clifford_gates * 7  # Transversal
    ms_t_ops = n_t_gates * (ms_distillation_depth + ms_injection_depth)

    print(f"  Data qubits: {ms_data_qubits}")
    print(f"  Ancilla qubits: {ms_total_ancilla}")
    print(f"  Clifford operations: {ms_clifford_ops}")
    print(f"  T-gate operations: {ms_t_ops}")
    print(f"  Total operations: {ms_clifford_ops + ms_t_ops}")

    # Code switching approach
    print("\n--- Code Switching Approach (Steane + RM) ---")
    cs_steane_qubits = 7
    cs_rm_qubits = 15
    cs_max_qubits = cs_steane_qubits + cs_rm_qubits  # During switch
    cs_switch_depth = 30  # Per switch

    # Estimate number of switches (group consecutive same-type gates)
    # Pessimistic: assume T gates are interleaved
    n_switches = min(n_t_gates, n_clifford_gates // 5)  # Rough estimate

    cs_clifford_ops = n_clifford_gates * 7
    cs_t_ops = n_t_gates * 15
    cs_switch_ops = n_switches * cs_switch_depth

    print(f"  Data qubits (Steane): {cs_steane_qubits}")
    print(f"  Data qubits (RM): {cs_rm_qubits}")
    print(f"  Max concurrent qubits: {cs_max_qubits}")
    print(f"  Clifford operations: {cs_clifford_ops}")
    print(f"  T operations: {cs_t_ops}")
    print(f"  Switch operations: {cs_switch_ops}")
    print(f"  Total operations: {cs_clifford_ops + cs_t_ops + cs_switch_ops}")
    print(f"  Estimated switches: {n_switches}")

    # Comparison
    print("\n--- Comparison ---")
    ms_total = ms_clifford_ops + ms_t_ops
    cs_total = cs_clifford_ops + cs_t_ops + cs_switch_ops

    print(f"  Magic state total ops: {ms_total}")
    print(f"  Code switching total ops: {cs_total}")
    print(f"  Ratio (CS/MS): {cs_total/ms_total:.2f}")

    if cs_total < ms_total:
        print("  Winner: Code Switching")
    else:
        print("  Winner: Magic States")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("Day 869: Code Switching Motivation")
    print("=" * 70)

    # Part 1: Compare transversal gates
    visualize_code_comparison()

    # Part 2: Circuit analysis
    demo_circuit_analysis()

    # Part 3: Resource comparison
    resource_comparison()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. Different codes have different transversal gate sets")
    print("2. Steane (Clifford) + Reed-Muller (T) = Universal")
    print("3. Code switching minimization is an optimization problem")
    print("4. Resource costs depend on circuit structure")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Eastin-Knill constraint | No code has transversal universal gates |
| Steane transversal | $\{H, S, \text{CNOT}\}$ (Clifford group) |
| Reed-Muller transversal | $\{T, \text{CNOT}, \text{CCZ}\}$ |
| Universal combination | Steane $\cup$ RM $= \{H, S, T, \text{CNOT}\}$ |
| Fault-tolerant switch | $\text{wt}(E_{\text{out}}) \leq \text{wt}(E_{\text{in}}) + 1$ |
| Switch count | Minimize transitions between code types |

### Main Takeaways

1. **Eastin-Knill forces alternatives:** No single code has universal transversal gates
2. **Complementary codes:** Different codes have different transversal gates that combine to universality
3. **Code switching strategy:** Switch between codes to access their respective transversal gates
4. **Fault tolerance matters:** Switching must not uncontrollably spread errors
5. **Practical competitiveness:** Recent experiments show code switching can outperform magic states
6. **Circuit optimization:** Minimizing switches is a key optimization goal

---

## Daily Checklist

- [ ] I can explain why Eastin-Knill necessitates code switching or magic states
- [ ] I can list transversal gates for Steane and Reed-Muller codes
- [ ] I understand how their combination provides universality
- [ ] I can compare resource requirements of code switching vs magic states
- [ ] I can analyze a circuit to count required code switches
- [ ] I understand fault tolerance requirements for code switching

---

## Preview: Day 870

Tomorrow we dive into the specific **Steane ↔ Reed-Muller code switching protocol**:

- Detailed structure of both codes
- The Anderson-Duclos-Cianci-Svore encoding circuits
- Step-by-step switching procedure
- Error analysis during conversion
- The recent Quantinuum experimental demonstration
- Practical implementation considerations

We will construct explicit circuits for converting between these complementary codes.
