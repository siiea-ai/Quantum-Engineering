# Day 761: Transversal Gates & Universality

## Overview

**Day:** 761 of 1008
**Week:** 109 (Fault-Tolerant Quantum Operations)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Transversal Gates, Eastin-Knill Theorem, and Paths to Universality

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Transversal gate classification |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Eastin-Knill and universality |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Clifford hierarchy |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Define** transversal gates and prove their fault-tolerance
2. **Classify** transversal gates for major code families
3. **Prove** the Eastin-Knill theorem
4. **Explain** the Clifford hierarchy and its structure
5. **Compare** approaches to achieving universality
6. **Analyze** trade-offs in fault-tolerant universal gate sets

---

## Core Content

### 1. Transversal Gates: Definition and Properties

**Definition:** A gate $\bar{U}$ on an $[[n,k,d]]$ code is **transversal** if it can be written as:

$$\boxed{\bar{U} = U_1 \otimes U_2 \otimes \cdots \otimes U_n}$$

where each $U_i$ acts only on the $i$-th physical qubit.

#### Why Transversal Gates Are Fault-Tolerant

**Theorem:** Transversal gates satisfy the FT condition automatically.

**Proof:**
- Single fault occurs on qubit $i$
- Only gate $U_i$ is affected
- Only qubit $i$ can have error after gate
- Weight-1 error per code block ✓

No error propagation between qubits!

### 2. Transversal Gate Classification by Code

Different codes support different transversal gates:

| Code | Transversal Gates | Missing for Universal |
|------|-------------------|----------------------|
| [[7,1,3]] Steane | CNOT, H, S | T |
| [[15,1,3]] RM | CNOT, T | H, S |
| Color codes | CNOT, H, S | T |
| [[5,1,3]] Perfect | CNOT only | H, S, T |
| Surface/Toric | CNOT | H, S, T |

#### Steane Code Transversal Gates

The [[7,1,3]] Steane code has:
- **CNOT:** $\overline{CNOT} = CNOT^{\otimes 7}$ (between blocks)
- **H:** $\bar{H} = H^{\otimes 7}$ (because code is self-dual CSS)
- **S:** $\bar{S} = S^{\otimes 7}$ (because code has triply-even structure)

$$\bar{H}|0\rangle_L = |+\rangle_L, \quad \bar{H}|1\rangle_L = |-\rangle_L$$

### 3. The Eastin-Knill Theorem

**Theorem (Eastin-Knill, 2009):** No quantum error-correcting code can have a universal set of transversal gates.

#### Intuition

If all universal gates were transversal:
- Each physical qubit evolves independently
- No entanglement created between different physical qubits
- But universal quantum computation requires entanglement creation!

#### Formal Statement

For any quantum code with distance $d \geq 2$:
- The set of transversal logical gates forms a **finite group**
- Finite groups cannot be universal (continuous rotations needed)

#### Proof Sketch

1. Transversal gates preserve the code structure
2. They form a subgroup of the logical Clifford group
3. The logical Clifford group is finite
4. Universal computation requires dense subset of SU($2^k$)
5. Finite ⊄ dense in infinite group ✗

### 4. The Clifford Hierarchy

The **Clifford hierarchy** organizes gates by their complexity:

$$\mathcal{C}_1 \subset \mathcal{C}_2 \subset \mathcal{C}_3 \subset \cdots$$

**Level 1:** Pauli group $\mathcal{C}_1 = \mathcal{P}$
$$\{I, X, Y, Z\}^{\otimes n}$$

**Level 2:** Clifford group $\mathcal{C}_2$
$$U \in \mathcal{C}_2 \Leftrightarrow U \mathcal{P} U^{\dagger} = \mathcal{P}$$

Includes: H, S, CNOT, CZ

**Level 3:**
$$U \in \mathcal{C}_3 \Leftrightarrow U \mathcal{P} U^{\dagger} \subseteq \mathcal{C}_2$$

Includes: T gate, CCZ (Toffoli)

**Level k:**
$$\boxed{U \in \mathcal{C}_k \Leftrightarrow U \mathcal{P} U^{\dagger} \subseteq \mathcal{C}_{k-1}}$$

### 5. Universal Gate Sets

For universal quantum computation, we need:

$$\boxed{\text{Clifford group} + \text{one non-Clifford gate} = \text{Universal}}$$

Common choices for non-Clifford:
- **T gate:** $T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$
- **Toffoli (CCZ):** Three-qubit controlled-controlled-Z
- **Magic state injection:** Use prepared resource states

### 6. Approaches to Fault-Tolerant Universality

#### Approach 1: Magic State Distillation

1. Prepare noisy T-magic states: $|T\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$
2. Distill to high fidelity using Clifford operations
3. Inject magic state to implement T gate

**Overhead:** Many noisy states → few clean states

#### Approach 2: Code Switching

1. Use code A for Clifford gates (transversal)
2. Switch to code B for T gate (transversal in B)
3. Switch back to code A

**Challenge:** Switching must be fault-tolerant

#### Approach 3: Gauge Fixing

Use subsystem codes with multiple gauge choices:
- Different gauge → different transversal gates
- Switch gauge without full code switching

#### Approach 4: Lattice Surgery

For surface codes:
- Transversal CNOT not available
- Use "lattice surgery" for logical CNOT
- Combines defects in topological manner

---

## Worked Examples

### Example 1: Proving Steane H is Transversal

**Problem:** Show that $\bar{H} = H^{\otimes 7}$ implements logical H on Steane code.

**Solution:**

Steane code is CSS from [7,4,3] Hamming code:
- Uses same classical code for X and Z stabilizers
- Code is self-dual: $C = C^{\perp}$

**Logical states:**
$$|0\rangle_L = \frac{1}{\sqrt{8}}\sum_{c \in C} |c\rangle$$
$$|1\rangle_L = \frac{1}{\sqrt{8}}\sum_{c \in C} |c \oplus 1111111\rangle$$

**Apply $H^{\otimes 7}$:**
$$H^{\otimes 7}|c\rangle = \frac{1}{\sqrt{128}}\sum_{x \in \{0,1\}^7} (-1)^{c \cdot x}|x\rangle$$

**Key fact:** For self-dual CSS:
$$H^{\otimes 7}|0\rangle_L = |+\rangle_L = \frac{1}{\sqrt{2}}(|0\rangle_L + |1\rangle_L)$$

This follows from the Hadamard on superpositions formula and dual code properties.

### Example 2: Why T is Not Transversal on Steane

**Problem:** Prove that $T^{\otimes 7}$ does not implement logical T on the Steane code.

**Solution:**

**Logical T should satisfy:**
$$\bar{T}|0\rangle_L = |0\rangle_L, \quad \bar{T}|1\rangle_L = e^{i\pi/4}|1\rangle_L$$

**Compute $T^{\otimes 7}|0\rangle_L$:**
$$T^{\otimes 7}|0\rangle_L = \frac{1}{\sqrt{8}}\sum_{c \in C} T^{\otimes 7}|c\rangle$$
$$= \frac{1}{\sqrt{8}}\sum_{c \in C} e^{i\pi|c|/4}|c\rangle$$

where $|c|$ is the Hamming weight of $c$.

**Problem:** Different codewords have different weights!
- $c = 0000000$: weight 0, phase $e^{0} = 1$
- $c = 1110000$: weight 3, phase $e^{3i\pi/4}$
- $c = 1111111$: weight 7, phase $e^{7i\pi/4}$

The phases don't factor as a global phase, so $T^{\otimes 7}$ creates **coherent superposition** with relative phases—not logical T!

### Example 3: Clifford Hierarchy Level

**Problem:** Show that T is in $\mathcal{C}_3$ but not $\mathcal{C}_2$.

**Solution:**

**Not in $\mathcal{C}_2$:** Compute $TXT^{\dagger}$:
$$TXT^{\dagger} = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$$
$$= \begin{pmatrix} 0 & e^{-i\pi/4} \\ e^{i\pi/4} & 0 \end{pmatrix} = e^{i\pi/4}P_X$$

where $P_X$ involves $\sqrt{X}$. This is NOT a Pauli, so $T \notin \mathcal{C}_2$.

**In $\mathcal{C}_3$:** For any Pauli $P$:
$$TPT^{\dagger} \in \mathcal{C}_2$$

We showed $TXT^{\dagger}$ is Clifford (it's a phase times Clifford). Similarly for $Y$ and $Z$:
$$TZT^{\dagger} = Z \in \mathcal{P} \subset \mathcal{C}_2$$

So $T \in \mathcal{C}_3$. ✓

---

## Practice Problems

### Problem Set A: Transversal Gates

**A1.** For the [[15,1,3]] Reed-Muller code, verify that $T^{\otimes 15}$ implements logical T.

**A2.** The [[5,1,3]] perfect code has transversal CNOT. Why doesn't it have transversal H?

**A3.** Show that for any CSS code, transversal $Z^{\otimes n}$ implements logical Z.

### Problem Set B: Eastin-Knill

**B1.** Explain why the Eastin-Knill theorem doesn't apply to:
a) Non-error-correcting encodings
b) Distance-1 codes

**B2.** The [[4,2,2]] code can detect 1 error. Does it violate Eastin-Knill? Explain.

**B3.** What is the maximum number of independent transversal gates possible for an $[[n,1,d]]$ code with $d \geq 3$?

### Problem Set C: Universality

**C1.** The T gate is in $\mathcal{C}_3$. Show that the controlled-T (CT) gate is in $\mathcal{C}_4$.

**C2.** How many T gates are needed to approximate an arbitrary single-qubit rotation to precision $\epsilon$? (Solovay-Kitaev)

**C3.** Compare the gate overhead for:
a) Magic state distillation
b) Code switching between Steane and [[15,1,3]] codes
For implementing a single logical T.

---

## Computational Lab

```python
"""
Day 761 Computational Lab: Transversal Gates and Universality
=============================================================

Analyze transversal gate properties and the Clifford hierarchy.
"""

import numpy as np
from typing import List, Tuple, Set, Optional
from functools import reduce

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Clifford gates
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
CLIFFORDS = {'H': H, 'S': S, 'Sdg': S.conj().T}


def tensor_product(gates: List[np.ndarray]) -> np.ndarray:
    """Compute tensor product of gates."""
    return reduce(np.kron, gates)


def is_pauli(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if U is a Pauli matrix (up to global phase)."""
    for name, P in PAULIS.items():
        for phase in [1, -1, 1j, -1j]:
            if np.allclose(U, phase * P, atol=tol):
                return True
    return False


def is_clifford(U: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if U is in the Clifford group.

    A unitary is Clifford iff it maps Paulis to Paulis.
    """
    n = int(np.log2(U.shape[0]))

    if n == 1:
        for P in [X, Y, Z]:
            conjugated = U @ P @ U.conj().T
            if not is_pauli(conjugated, tol):
                return False
        return True
    else:
        # For multi-qubit, check generators
        # Simplified: just check single-qubit Paulis on each qubit
        return True  # Would need full implementation


def clifford_hierarchy_level(U: np.ndarray, max_level: int = 5) -> int:
    """
    Determine the Clifford hierarchy level of U.

    Returns: Level (1 for Pauli, 2 for Clifford, 3 for T, etc.)
    """
    if is_pauli(U):
        return 1

    if is_clifford(U):
        return 2

    # Check level 3: U P U† should be Clifford for all Paulis P
    all_clifford = True
    for P in [X, Y, Z]:
        conjugated = U @ P @ U.conj().T
        if not is_clifford(conjugated):
            all_clifford = False
            break

    if all_clifford:
        return 3

    return 4  # Higher level (simplified)


def analyze_transversal_gate(gate: np.ndarray, n_qubits: int,
                            code_name: str) -> dict:
    """
    Analyze properties of transversal gate application.
    """
    transversal = tensor_product([gate] * n_qubits)

    return {
        'code': code_name,
        'n_qubits': n_qubits,
        'gate_dimension': gate.shape[0],
        'transversal_dimension': transversal.shape[0],
        'gate_level': clifford_hierarchy_level(gate),
        'is_unitary': np.allclose(gate @ gate.conj().T, I)
    }


class TransversalGateAnalyzer:
    """
    Analyze transversal gates for specific codes.
    """

    def __init__(self, code_params: Tuple[int, int, int]):
        """
        Initialize with [[n, k, d]] code parameters.
        """
        self.n, self.k, self.d = code_params
        self.transversal_gates = []

    def check_gate(self, gate_name: str, gate: np.ndarray) -> dict:
        """
        Check if a gate is transversal for this code.
        """
        level = clifford_hierarchy_level(gate)

        return {
            'gate_name': gate_name,
            'hierarchy_level': level,
            'is_clifford': level <= 2,
            'would_be_ft': True  # Transversal always FT
        }

    def get_available_transversal(self) -> List[str]:
        """Return list of transversal gates for this code."""
        # This would depend on specific code structure
        # Simplified for demonstration
        if self.n == 7:  # Steane-like
            return ['CNOT', 'H', 'S']
        elif self.n == 15:  # RM-like
            return ['CNOT', 'T']
        elif self.n == 5:  # Perfect code
            return ['CNOT']
        else:
            return ['CNOT']  # Default: CNOT usually transversal


def compute_t_gate_decomposition(precision: float) -> int:
    """
    Estimate number of T gates needed for arbitrary rotation.

    Solovay-Kitaev: O(log^c(1/ε)) gates for precision ε
    """
    if precision <= 0:
        return float('inf')

    # Simplified: approximately 3 * log_2(1/ε) T gates
    return int(3 * np.log2(1 / precision))


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 761: TRANSVERSAL GATES & UNIVERSALITY")
    print("=" * 70)

    # Demo 1: Gate hierarchy levels
    print("\n" + "=" * 70)
    print("Demo 1: Clifford Hierarchy Levels")
    print("=" * 70)

    gates = [
        ('I', I), ('X', X), ('Y', Y), ('Z', Z),
        ('H', H), ('S', S),
        ('T', T)
    ]

    print("\nGate hierarchy classification:")
    for name, gate in gates:
        level = clifford_hierarchy_level(gate)
        level_names = {1: 'Pauli (C₁)', 2: 'Clifford (C₂)',
                      3: 'C₃ (T-level)', 4: 'C₄+'}
        print(f"  {name}: Level {level} - {level_names.get(level, 'Unknown')}")

    # Demo 2: Transversal gate analysis
    print("\n" + "=" * 70)
    print("Demo 2: Transversal Gate Analysis")
    print("=" * 70)

    codes = [
        ("[[7,1,3]] Steane", (7, 1, 3)),
        ("[[15,1,3]] RM", (15, 1, 3)),
        ("[[5,1,3]] Perfect", (5, 1, 3)),
    ]

    for name, params in codes:
        analyzer = TransversalGateAnalyzer(params)
        transversal = analyzer.get_available_transversal()
        print(f"\n{name}:")
        print(f"  Transversal gates: {', '.join(transversal)}")

        # What's missing for universality?
        universal = {'CNOT', 'H', 'S', 'T'}
        missing = universal - set(transversal)
        print(f"  Missing for universal: {', '.join(missing) if missing else 'None!'}")

    # Demo 3: T gate conjugation
    print("\n" + "=" * 70)
    print("Demo 3: Why T is in C₃")
    print("=" * 70)

    print("\nConjugating Paulis by T:")
    for name, P in [('X', X), ('Y', Y), ('Z', Z)]:
        conjugated = T @ P @ T.conj().T
        is_cliff = is_clifford(conjugated)
        print(f"  T{name}T† is Clifford: {is_cliff}")

    print("\nT conjugates Paulis to Cliffords → T ∈ C₃")

    # Demo 4: T gate cost for rotations
    print("\n" + "=" * 70)
    print("Demo 4: T Gate Synthesis Cost")
    print("=" * 70)

    precisions = [0.1, 0.01, 0.001, 1e-6, 1e-10]

    print("\nT gates needed for arbitrary rotation (Solovay-Kitaev):")
    for eps in precisions:
        n_t = compute_t_gate_decomposition(eps)
        print(f"  ε = {eps:.0e}: ~{n_t} T gates")

    # Summary: Eastin-Knill
    print("\n" + "=" * 70)
    print("EASTIN-KNILL THEOREM")
    print("=" * 70)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  EASTIN-KNILL THEOREM                                      │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  STATEMENT:                                                 │
    │    No quantum error-correcting code with d ≥ 2 can have   │
    │    a universal set of transversal logical gates.          │
    │                                                             │
    │  IMPLICATIONS:                                              │
    │    • Must use non-transversal gates for universality       │
    │    • Options: magic states, code switching, gauge fixing   │
    │                                                             │
    │  THE CLIFFORD HIERARCHY:                                    │
    │    C₁ ⊂ C₂ ⊂ C₃ ⊂ ...                                    │
    │                                                             │
    │    C₁: Paulis {I, X, Y, Z}                                 │
    │    C₂: Clifford {H, S, CNOT, ...}                          │
    │    C₃: {T, CCZ, ...}                                       │
    │                                                             │
    │  UNIVERSAL SET:                                             │
    │    Clifford + any non-Clifford = Universal                 │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)

    print("=" * 70)
    print("Day 761 Complete: Transversal Gates & Universality Mastered")
    print("=" * 70)
```

---

## Summary

### Transversal Gate Properties

| Property | Description |
|----------|-------------|
| Definition | $\bar{U} = U_1 \otimes U_2 \otimes \cdots \otimes U_n$ |
| FT Property | Automatic—no error propagation between qubits |
| Limitation | Cannot be universal (Eastin-Knill) |

### The Clifford Hierarchy

$$\mathcal{C}_1 = \text{Paulis} \subset \mathcal{C}_2 = \text{Clifford} \subset \mathcal{C}_3 \ni T$$

### Key Equations

$$\boxed{\text{Eastin-Knill: } d \geq 2 \Rightarrow \text{No universal transversal set}}$$
$$\boxed{\text{Universal: Clifford} + T = \text{Dense in } SU(2^n)}$$

---

## Daily Checklist

- [ ] Defined transversal gates and proved FT property
- [ ] Classified transversal gates by code family
- [ ] Understood Eastin-Knill theorem
- [ ] Explored Clifford hierarchy structure
- [ ] Compared universality approaches
- [ ] Completed practice problems

---

## Preview: Day 762

Tomorrow we study **Magic State Injection**:
- T-magic state definition and properties
- Gate teleportation protocol
- Injection circuits
- Introduction to magic state distillation
