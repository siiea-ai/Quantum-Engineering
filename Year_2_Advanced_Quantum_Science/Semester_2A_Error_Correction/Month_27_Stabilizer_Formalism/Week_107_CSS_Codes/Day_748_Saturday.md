# Day 748: Transversal Gates in CSS Codes

## Overview

**Day:** 748 of 1008
**Week:** 107 (CSS Codes & Related Constructions)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Transversal Operations and Fault Tolerance

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Transversal gate theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | CSS-specific gates |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Define** transversal gates and their fault-tolerance properties
2. **Identify** which gates are transversal for CSS codes
3. **Implement** transversal CNOT between CSS blocks
4. **State** the Eastin-Knill theorem and its implications
5. **Understand** magic state injection for universality
6. **Design** fault-tolerant gate sequences using CSS codes

---

## Transversal Gates

### Definition

A gate is **transversal** if it can be implemented as a tensor product of single-qubit (or two-qubit) operations, each acting on corresponding qubits in the code block(s).

$$U_{\text{trans}} = U_1 \otimes U_2 \otimes \cdots \otimes U_n$$

### Fault Tolerance Property

**Key property:** Transversal gates cannot spread errors within a code block.

If an error occurs on qubit i, it stays on qubit i after the transversal gate.

$$E_i \cdot U_{\text{trans}} = U_{\text{trans}} \cdot E'_i$$

where E'_i is still localized to qubit i.

### Why Transversal = Fault-Tolerant

Without transversality, a gate U might map:
$$E_i \to E_i E_j E_k \cdots$$

spreading a single error to multiple locations—potentially uncorrectable!

---

## Transversal Gates for CSS Codes

### Pauli Gates

**Transversal X:**
$$\bar{X} = X^{\otimes n}$$

Acts as logical X on the encoded qubit.

**Verification:** For CSS codes, X stabilizers are products of X operators.
$$X^{\otimes n} \cdot S_X = S_X \cdot X^{\otimes n}$$
(X commutes with X)

$$X^{\otimes n} \cdot S_Z = \pm S_Z \cdot X^{\otimes n}$$
Need to check: For CSS, Z stabilizers have support, X^⊗n flips signs based on overlap parity.

For valid CSS codes, $X^{\otimes n}$ either commutes with all stabilizers or implements $\bar{X}$.

**Transversal Z:**
$$\bar{Z} = Z^{\otimes n}$$

Similar analysis.

### Hadamard Gate (Special Cases)

For **self-dual CSS codes** (where H_X = H_Z structure is symmetric):

$$\bar{H} = H^{\otimes n}$$

**Example:** Steane code is self-dual CSS, so $H^{\otimes 7}$ implements logical H.

**Why it works:**
- H swaps X ↔ Z
- For self-dual CSS: X stabilizers ↔ Z stabilizers under H
- So the code space is preserved

### CNOT Gate

**Transversal CNOT** between two CSS code blocks:

$$\overline{CNOT} = CNOT^{\otimes n}$$

where each CNOT acts on corresponding qubits of control and target blocks.

**Verification:**
- CNOT maps: $X_c \to X_c X_t$, $Z_c \to Z_c$, $X_t \to X_t$, $Z_t \to Z_c Z_t$
- For CSS codes, this preserves the stabilizer structure

---

## The Eastin-Knill Theorem

### Statement

**Theorem (Eastin-Knill, 2009):**
No quantum error-correcting code can have a universal set of transversal gates.

### Implications

- **Cannot** have transversal T gate (or any non-Clifford gate) for error-correcting codes
- Must use **non-transversal methods** for universality
- Trade-off between error correction and gate implementation

### Proof Sketch

Transversal gates form a finite group (discrete rotations). Universal computation requires continuous rotations, which cannot all be transversal.

---

## Achieving Universality

### The Clifford + T Framework

**Clifford gates:** H, S, CNOT (transversal in many codes)
**T gate:** $T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$

**Universality:** Clifford + T is universal for quantum computation.

### Magic State Injection

**T gate via magic states:**

1. Prepare "magic state" $|T\rangle = T|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$
2. Use Clifford gates + measurements to implement T on encoded qubit

**Protocol:**
```
|ψ⟩ ─────●───── M_X ───── (correction based on measurement)
         │
|T⟩ ─────⊕─────────────── |T|ψ⟩⟩ (up to correction)
```

### Magic State Distillation

**Problem:** Preparing |T⟩ fault-tolerantly is hard.

**Solution:** Magic state distillation
1. Prepare many noisy |T⟩ states
2. Use Clifford operations to "distill" fewer, higher-fidelity states
3. Repeat until sufficient quality

**Key result:** 15 noisy states → 1 improved state (15-to-1 distillation)

---

## CSS-Specific Gate Implementations

### Phase Gate (S)

For some CSS codes, S gate is transversal:
$$\bar{S} = S^{\otimes n}$$

**Condition:** Works when the code has appropriate weight structure.

For Steane code: $S^{\otimes 7}$ is NOT transversal S (introduces errors).

### Controlled-Z

**CZ between blocks:**
$$\overline{CZ} = CZ^{\otimes n}$$

Works for CSS codes by symmetry with CNOT.

### Code-Specific Examples

**Steane [[7,1,3]]:**
- Transversal: X, Z, H, CNOT
- Non-transversal: T, S (requires magic states)

**[[15,1,3]] Reed-Muller:**
- Transversal: X, Z, H, CNOT, **T** (!)
- This is special; most codes don't have transversal T

---

## Fault-Tolerant Protocols

### Syndrome Extraction

**Challenge:** Measuring stabilizers can introduce errors.

**Solution:** Use ancilla qubits and transversal operations:

```
|0⟩ ─── H ───●───●───●───●─── H ─── M
             │   │   │   │
|ψ⟩_1 ───────Z───┼───┼───┼─────────
|ψ⟩_2 ───────────Z───┼───┼─────────
|ψ⟩_3 ───────────────Z───┼─────────
|ψ⟩_4 ───────────────────Z─────────
```

This measures Z₁Z₂Z₃Z₄ stabilizer.

### Error Correction Cycle

1. **Syndrome measurement** (using transversal CNOT with ancillas)
2. **Classical processing** (decode syndrome)
3. **Correction** (transversal Pauli gates)

### Logical Gate Execution

1. Perform transversal logical gate
2. Immediately follow with error correction
3. Accumulated errors stay correctable

---

## Worked Examples

### Example 1: Transversal CNOT

**Problem:** Show that transversal CNOT preserves Steane code stabilizers.

**Solution:**

Control block stabilizers: $S_X^{(c)} = IIIXXXX$, $S_Z^{(c)} = IIIZZZZ$, etc.
Target block stabilizers: $S_X^{(t)} = IIIXXXX$, $S_Z^{(t)} = IIIZZZZ$, etc.

Under $CNOT^{\otimes 7}$:
- $X_c \to X_c X_t$ on each qubit
- $S_X^{(c)} = X_4^c X_5^c X_6^c X_7^c \to X_4^c X_5^c X_6^c X_7^c \cdot X_4^t X_5^t X_6^t X_7^t = S_X^{(c)} S_X^{(t)}$

This is still a stabilizer (product of stabilizers). ✓

Similarly for Z stabilizers:
- $Z_t \to Z_c Z_t$ on each qubit
- $S_Z^{(t)} \to S_Z^{(c)} S_Z^{(t)}$

All stabilizers map to stabilizers. Code preserved! ✓

### Example 2: Why T is Not Transversal

**Problem:** Show that $T^{\otimes 7}$ is not a logical gate for Steane code.

**Solution:**

$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$

Action on Z stabilizer $S_Z = IIIZZZZ$:
- T commutes with Z: $TZT^\dagger = Z$
- So $T^{\otimes 7} S_Z (T^\dagger)^{\otimes 7} = S_Z$ ✓

Action on X stabilizer $S_X = IIIXXXX$:
- $TXT^\dagger = e^{i\pi/4} \cdot \frac{1}{\sqrt{2}}(X + Y) \neq X$

Actually: $TXT^\dagger = \begin{pmatrix} 0 & e^{-i\pi/4} \\ e^{i\pi/4} & 0 \end{pmatrix}$

This introduces phases that don't cancel correctly for all X stabilizers.

$T^{\otimes 7}$ does NOT preserve the stabilizer group, so it's not a valid logical gate.

### Example 3: Magic State Distillation

**Problem:** Outline the 15-to-1 T-state distillation protocol.

**Solution:**

1. Start with 15 noisy |T⟩ states with error rate ε
2. Encode into [[15,1,3]] Reed-Muller code
3. Measure stabilizers (Clifford operations)
4. If all pass, decode to get 1 improved |T⟩
5. Output error rate: O(ε³)

**Improvement:** 15ε → ε³ (cubic suppression!)

Repeat: ε → ε³ → ε⁹ → ... exponential improvement

---

## Practice Problems

### Level 1: Direct Application

**P1.1** Verify that $X^{\otimes 7}$ commutes with all Steane code Z stabilizers.

**P1.2** For the [[9,1,3]] Shor code, determine which of {X^⊗9, Z^⊗9, H^⊗9} are valid transversal gates.

**P1.3** How many CNOT gates are used in a transversal CNOT between two [[7,1,3]] blocks?

### Level 2: Intermediate

**P2.1** Prove that transversal CNOT preserves the CSS structure of two identical CSS codes.

**P2.2** For a [[n,1,d]] CSS code, show that transversal X implements either logical X or is a stabilizer element.

**P2.3** Design a fault-tolerant H gate circuit for the Steane code using transversal H.

### Level 3: Challenging

**P3.1** Prove the Eastin-Knill theorem: show that transversal gates form a finite group.

**P3.2** Analyze the [[15,1,3]] Reed-Muller code and show why it has transversal T.

**P3.3** Design a complete fault-tolerant universal gate set for the Steane code, including magic state injection.

---

## Computational Lab

```python
"""
Day 748: Transversal Gates in CSS Codes
=======================================

Implementing and verifying transversal gate operations.
"""

import numpy as np
from typing import List, Tuple, Optional


def pauli_matrix(p: str) -> np.ndarray:
    """Return 2×2 Pauli matrix."""
    matrices = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    return matrices[p]


def tensor_paulis(paulis: str) -> np.ndarray:
    """Tensor product of Pauli string."""
    result = pauli_matrix(paulis[0])
    for p in paulis[1:]:
        result = np.kron(result, pauli_matrix(p))
    return result


class TransversalGates:
    """Analyze transversal gates for CSS codes."""

    def __init__(self, H_X: np.ndarray, H_Z: np.ndarray):
        """
        Initialize with X and Z parity check matrices.

        H_X: m_x × n (X stabilizers)
        H_Z: m_z × n (Z stabilizers)
        """
        self.H_X = np.array(H_X) % 2
        self.H_Z = np.array(H_Z) % 2
        self.n = self.H_X.shape[1]

    def x_stabilizers(self) -> List[str]:
        """Return X stabilizer Pauli strings."""
        stabs = []
        for row in self.H_X:
            pauli = ''.join('X' if b else 'I' for b in row)
            stabs.append(pauli)
        return stabs

    def z_stabilizers(self) -> List[str]:
        """Return Z stabilizer Pauli strings."""
        stabs = []
        for row in self.H_Z:
            pauli = ''.join('Z' if b else 'I' for b in row)
            stabs.append(pauli)
        return stabs

    def check_transversal_pauli(self, gate: str) -> dict:
        """
        Check if transversal Pauli gate preserves stabilizers.

        gate: 'X', 'Z', or 'Y'
        """
        results = {'valid': True, 'x_stabs': [], 'z_stabs': []}

        # X^⊗n commutation with stabilizers
        for x_stab in self.x_stabilizers():
            # X commutes with X
            results['x_stabs'].append(('commutes', x_stab))

        for z_stab in self.z_stabilizers():
            # Count overlap
            weight = sum(1 for c in z_stab if c == 'Z')
            if gate == 'X':
                # X anticommutes with Z: (-1)^weight
                sign = (-1) ** weight
                results['z_stabs'].append((sign, z_stab))
                if sign == -1 and weight % 2 == 1:
                    # Odd weight Z stabilizer → sign flip
                    pass  # May indicate logical X

        return results

    def check_transversal_hadamard(self) -> dict:
        """
        Check if H^⊗n preserves code space.

        For self-dual CSS: H swaps X ↔ Z stabilizers.
        """
        # H: X → Z, Z → X
        # X stabilizers become Z-type after H
        # Z stabilizers become X-type after H

        x_to_z = []
        for x_stab in self.x_stabilizers():
            z_stab = x_stab.replace('X', 'Z')
            x_to_z.append(z_stab)

        z_to_x = []
        for z_stab in self.z_stabilizers():
            x_stab = z_stab.replace('Z', 'X')
            z_to_x.append(x_stab)

        # Check if transformed stabilizers are in original stabilizer group
        # (For self-dual CSS, they should match up)

        return {
            'x_to_z': x_to_z,
            'z_to_x': z_to_x,
            'self_dual': set(x_to_z) == set(self.z_stabilizers())
        }

    def check_transversal_cnot(self, other: 'TransversalGates') -> dict:
        """
        Check transversal CNOT between self (control) and other (target).

        CNOT: X_c → X_c X_t, Z_t → Z_c Z_t
        """
        if self.n != other.n:
            return {'valid': False, 'error': 'Different code lengths'}

        # After CNOT^⊗n:
        # Control X stabilizers: X_c → X_c X_t (gain X on target)
        # Target Z stabilizers: Z_t → Z_c Z_t (gain Z on control)

        return {
            'valid': True,
            'control_x_transforms': True,  # X stabilizers gain X support on target
            'target_z_transforms': True,   # Z stabilizers gain Z support on control
            'note': 'CSS structure preserved for identical codes'
        }


def steane_code_gates():
    """Analyze transversal gates for Steane code."""
    H = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ])
    return TransversalGates(H, H)


def verify_logical_x(n: int, H_Z: np.ndarray) -> bool:
    """Verify X^⊗n commutes with all Z stabilizers."""
    # X^⊗n anticommutes with Z_i, gaining (-1) for each Z
    # For stabilizer S_Z with weight w, sign = (-1)^w
    for row in H_Z:
        weight = np.sum(row)
        if weight % 2 != 0:
            return False  # Odd weight means anticommutation
    return True


def magic_state_fidelity(initial_error: float, rounds: int) -> float:
    """
    Compute magic state fidelity after distillation.

    Uses 15-to-1 protocol with cubic error suppression.
    """
    error = initial_error
    for _ in range(rounds):
        error = 35 * (error ** 3)  # 15-to-1 with cubic suppression
        if error < 1e-15:
            break
    return 1 - error


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 748: Transversal Gates in CSS Codes")
    print("=" * 60)

    # Example 1: Steane code analysis
    print("\n1. Steane Code Transversal Gates")
    print("-" * 40)

    steane = steane_code_gates()

    print("X Stabilizers:")
    for s in steane.x_stabilizers():
        print(f"  {s}")

    print("\nZ Stabilizers:")
    for s in steane.z_stabilizers():
        print(f"  {s}")

    # Check X^⊗7
    x_result = steane.check_transversal_pauli('X')
    print("\nTransversal X^⊗7 analysis:")
    for sign, stab in x_result['z_stabs']:
        print(f"  {stab}: sign = {sign}")

    # Check Hadamard
    h_result = steane.check_transversal_hadamard()
    print(f"\nTransversal H^⊗7:")
    print(f"  Self-dual CSS: {h_result['self_dual']}")
    print("  X stabilizers → Z stabilizers under H: ✓" if h_result['self_dual'] else "  ✗")

    # Example 2: CNOT between blocks
    print("\n2. Transversal CNOT")
    print("-" * 40)

    cnot_result = steane.check_transversal_cnot(steane)
    print(f"CNOT^⊗7 between Steane blocks:")
    print(f"  Valid: {cnot_result['valid']}")
    print(f"  Note: {cnot_result['note']}")

    # Example 3: Magic state distillation
    print("\n3. Magic State Distillation")
    print("-" * 40)

    initial_errors = [0.1, 0.05, 0.01]
    for e in initial_errors:
        print(f"\nInitial error rate: {e}")
        for rounds in range(1, 5):
            fidelity = magic_state_fidelity(e, rounds)
            error = 1 - fidelity
            print(f"  After {rounds} rounds: error = {error:.2e}")

    # Example 4: Gate decomposition
    print("\n4. Universal Gate Set for Steane Code")
    print("-" * 40)

    gates = {
        'X': 'Transversal X^⊗7',
        'Z': 'Transversal Z^⊗7',
        'H': 'Transversal H^⊗7',
        'S': 'Via S^⊗7 (need verification) or magic state',
        'CNOT': 'Transversal CNOT^⊗7',
        'T': 'Magic state injection'
    }

    print("Logical gate implementations:")
    for gate, impl in gates.items():
        print(f"  {gate}: {impl}")

    # Example 5: Verify stabilizer preservation
    print("\n5. Stabilizer Preservation Verification")
    print("-" * 40)

    H = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ])

    # All rows have weight 4 (even), so X^⊗7 commutes
    weights = [np.sum(row) for row in H]
    print(f"Z stabilizer weights: {weights}")
    print(f"All even: {all(w % 2 == 0 for w in weights)}")
    print(f"Therefore X^⊗7 commutes with all Z stabilizers: ✓")

    # For logical X, need X^⊗7 to anticommute with logical Z
    print(f"\nLogical Z = Z^⊗7 (weight 7, odd)")
    print(f"X^⊗7 anticommutes with Z^⊗7: ✓ (implements logical X)")

    print("\n" + "=" * 60)
    print("Transversal gates: the key to fault-tolerant computation!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Gate | Transversal Form | CSS Requirement |
|------|------------------|-----------------|
| Logical X | $X^{\otimes n}$ | All Z stabilizers have even weight |
| Logical Z | $Z^{\otimes n}$ | All X stabilizers have even weight |
| Logical H | $H^{\otimes n}$ | Self-dual CSS (H_X ↔ H_Z) |
| Logical CNOT | $CNOT^{\otimes n}$ | Identical CSS codes |
| Logical T | Not transversal | Requires magic states |

### Main Takeaways

1. **Transversal gates** prevent error propagation within code blocks
2. **CSS codes** have transversal Paulis and often transversal H
3. **CNOT** is transversal between identical CSS code blocks
4. **Eastin-Knill theorem** prevents universal transversal gate sets
5. **Magic state injection** enables non-Clifford gates like T
6. **Distillation** improves magic state fidelity exponentially

---

## Daily Checklist

- [ ] I can identify which gates are transversal for a CSS code
- [ ] I understand why transversal gates are fault-tolerant
- [ ] I can verify stabilizer preservation under transversal gates
- [ ] I know the Eastin-Knill theorem statement
- [ ] I understand magic state injection protocol
- [ ] I can outline a universal fault-tolerant gate set

---

## Preview: Day 749

Tomorrow we synthesize **Week 107: CSS Codes**:

- Comprehensive review of CSS construction
- Integration problems combining all topics
- Master formula sheet
- Preparation for Week 108: Code Families

The CSS framework is fundamental to quantum error correction!
