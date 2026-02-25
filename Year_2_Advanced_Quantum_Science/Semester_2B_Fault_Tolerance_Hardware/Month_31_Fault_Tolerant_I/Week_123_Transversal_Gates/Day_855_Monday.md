# Day 855: Transversal Gate Definition

## Overview

**Day:** 855 of 1008
**Week:** 123 (Transversal Gates & Eastin-Knill)
**Month:** 31 (Fault-Tolerant Quantum Computing I)
**Topic:** Formal Definition of Transversal Gates and Error Propagation

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Transversal gate theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Error propagation analysis |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational exploration |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Define** transversal gates formally using tensor product structure
2. **Explain** why transversal gates prevent error propagation within code blocks
3. **Distinguish** between strictly transversal and weakly transversal gates
4. **Analyze** error propagation for transversal vs. non-transversal gates
5. **Identify** the connection between transversality and fault tolerance
6. **Construct** examples of transversal and non-transversal logical gates

---

## Motivation: The Error Propagation Problem

### Why Fault Tolerance Needs Special Gates

In fault-tolerant quantum computation, our primary enemy is **error propagation**. Consider a two-qubit gate like CNOT:

$$\text{CNOT}|+\rangle|0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

If an X error occurs on the control qubit BEFORE the CNOT:
$$\text{CNOT}(X \otimes I)|+\rangle|0\rangle = \text{CNOT}|{-}\rangle|0\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$

But if we track through:
$$(X \otimes I)|+\rangle|0\rangle = |-\rangle|0\rangle$$
$$\text{CNOT}|-\rangle|0\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle) = (X \otimes X)\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

The single-qubit error has **spread** to become a two-qubit error!

### The Transversal Solution

**Key Insight:** If gates act independently on each qubit, errors cannot spread.

---

## Core Theory

### Definition 1: Transversal Gate (Strict Form)

**Definition:** Let $\mathcal{C}$ be an $[[n, k, d]]$ quantum error-correcting code. A logical gate $\bar{U}$ is **transversal** if it can be implemented as:

$$\boxed{\bar{U} = U_1 \otimes U_2 \otimes \cdots \otimes U_n}$$

where each $U_i$ is a unitary acting only on qubit $i$.

**Special Case (Homogeneous Transversal):**

$$\bar{U} = U^{\otimes n}$$

where the same unitary $U$ is applied to every physical qubit.

### Definition 2: Transversal Gate (Between Code Blocks)

For two-block operations (like logical CNOT between two encoded qubits):

**Definition:** A logical gate $\bar{U}$ acting on code blocks $A$ and $B$ is **transversal** if:

$$\bar{U} = \bigotimes_{i=1}^{n} V_i$$

where each $V_i$ acts only on qubits $(A_i, B_i)$ from corresponding positions in the two blocks.

$$\boxed{\overline{\text{CNOT}}_{AB} = \prod_{i=1}^{n} \text{CNOT}_{A_i \to B_i}}$$

### Definition 3: Weakly Transversal Gate

**Definition:** A gate is **weakly transversal** if it can be decomposed into a constant-depth circuit of transversal operations.

This allows some limited coupling but prevents unbounded error propagation.

---

## Error Propagation Analysis

### Theorem: Transversal Gates Don't Spread Errors

**Theorem:** If $\bar{U} = \bigotimes_{i=1}^n U_i$ is transversal and $E = E_j$ is an error on qubit $j$ only, then:

$$\bar{U} E \bar{U}^\dagger = E'_j$$

where $E'_j$ is still localized to qubit $j$.

**Proof:**
$$\bar{U} E_j \bar{U}^\dagger = \left(\bigotimes_{i=1}^n U_i\right) E_j \left(\bigotimes_{i=1}^n U_i^\dagger\right)$$

Since $E_j$ acts only on qubit $j$:
$$= \left(\bigotimes_{i \neq j} U_i U_i^\dagger\right) \otimes (U_j E_j U_j^\dagger)$$
$$= \left(\bigotimes_{i \neq j} I_i\right) \otimes (U_j E_j U_j^\dagger)$$
$$= I^{\otimes (n-1)} \otimes E'_j$$

where $E'_j = U_j E_j U_j^\dagger$ acts only on qubit $j$. $\square$

### Corollary: Error Weight Preservation

**Corollary:** Transversal gates preserve error weight:
$$\text{wt}(\bar{U} E \bar{U}^\dagger) = \text{wt}(E)$$

where weight is the number of qubits with non-identity errors.

### Counterexample: Non-Transversal Gates

Consider a logical CNOT within a single code block:
$$\overline{\text{CNOT}}_{\text{internal}} = \text{CNOT}_{1 \to 2} \cdot \text{CNOT}_{3 \to 4} \cdots$$

This is NOT transversal because CNOT couples pairs of qubits.

An X error on qubit 1 propagates:
$$\text{CNOT}_{1 \to 2} (X_1) = X_1 X_2$$

The weight-1 error becomes weight-2!

---

## Formal Framework

### The Logical Gate Group

For an $[[n, k, d]]$ code $\mathcal{C}$ with code space $\mathcal{H}_L \cong (\mathbb{C}^2)^{\otimes k}$:

**Definition:** The **logical gate group** is:
$$\mathcal{G}_L = \{U \in U(2^k) : \exists \tilde{U} \in U(2^n) \text{ s.t. } \tilde{U}|_{\mathcal{H}_L} \cong U\}$$

**Definition:** The **transversal gate group** is:
$$\mathcal{T} = \{U \in \mathcal{G}_L : U \text{ has transversal implementation}\}$$

**Key Observation:** $\mathcal{T} \subseteq \mathcal{G}_L$, and typically $\mathcal{T} \subsetneq \mathcal{G}_L$.

### Structure Theorem

**Theorem:** For any quantum code, $\mathcal{T}$ forms a group under composition.

**Proof:**
1. **Closure:** If $\bar{U} = \bigotimes U_i$ and $\bar{V} = \bigotimes V_i$ are transversal, then $\bar{U}\bar{V} = \bigotimes (U_i V_i)$ is transversal.
2. **Identity:** $\bar{I} = I^{\otimes n}$ is transversal.
3. **Inverse:** $\bar{U}^{-1} = \bigotimes U_i^{-1}$ is transversal. $\square$

---

## Examples of Transversal Gates

### Example 1: Transversal Pauli X

For any CSS code with logical X operator $\bar{X} = X^{\otimes n}$:

$$\bar{X} = X \otimes X \otimes \cdots \otimes X = X^{\otimes n}$$

**Verification:** This is manifestly transversal with $U_i = X$ for all $i$.

### Example 2: Transversal Pauli Z

For any CSS code with logical Z operator $\bar{Z} = Z^{\otimes n}$:

$$\bar{Z} = Z \otimes Z \otimes \cdots \otimes Z = Z^{\otimes n}$$

### Example 3: Transversal CNOT (Between Blocks)

For two code blocks A and B of the same CSS code:

$$\overline{\text{CNOT}}_{A \to B} = \prod_{i=1}^n \text{CNOT}_{A_i \to B_i}$$

**Why it works:** Each CNOT acts on a pair $(A_i, B_i)$, coupling different code blocks but not spreading errors within a block.

### Example 4: Transversal Hadamard (Self-Dual Codes)

For **self-dual** CSS codes (where $C_1 = C_2$ and $H_X = H_Z$):

$$\bar{H} = H^{\otimes n}$$

**Requirement:** The code must be self-dual for this to implement a logical Hadamard.

### Example 5: Non-Transversal T-Gate

For any CSS code, the T-gate:
$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

is **never transversal**.

If $\bar{T} = T^{\otimes n}$ were the logical T-gate, it would not preserve the code space properly. We will prove this rigorously when studying Eastin-Knill.

---

## The Steane Code Example

### Steane Code [[7,1,3]]

The Steane code is a self-dual CSS code with particularly nice transversal properties.

**Transversal Gates:**

| Gate | Implementation | Transversal? |
|------|----------------|--------------|
| $\bar{X}$ | $X^{\otimes 7}$ | Yes |
| $\bar{Z}$ | $Z^{\otimes 7}$ | Yes |
| $\bar{H}$ | $H^{\otimes 7}$ | Yes |
| $\bar{S}$ | $S^{\otimes 7}$ | Yes (up to phase) |
| $\overline{\text{CNOT}}$ | $\text{CNOT}^{\otimes 7}$ | Yes (between blocks) |
| $\bar{T}$ | ??? | **No** |

**Key Result:** The Steane code has transversal Clifford group, but NOT transversal T.

### Verification: Transversal H on Steane Code

The Steane code is CSS(C, C) where C is the [7,4,3] Hamming code.

For a self-dual CSS code, the logical states are:
$$|\bar{0}\rangle = \frac{1}{\sqrt{|C^\perp|}} \sum_{c \in C^\perp} |c\rangle$$
$$|\bar{1}\rangle = \frac{1}{\sqrt{|C^\perp|}} \sum_{c \in C^\perp} |c + \bar{x}\rangle$$

where $\bar{x}$ is a representative of the non-trivial coset.

Applying $H^{\otimes 7}$:
$$H^{\otimes 7} |\bar{0}\rangle = \frac{1}{\sqrt{|C^\perp|}} \sum_{c \in C^\perp} H^{\otimes 7}|c\rangle$$
$$= \frac{1}{\sqrt{|C^\perp|}} \sum_{c \in C^\perp} \frac{1}{\sqrt{2^7}} \sum_{x \in \{0,1\}^7} (-1)^{c \cdot x} |x\rangle$$

Using the dual code structure, this equals $|\bar{+}\rangle$. Similarly $H^{\otimes 7}|\bar{1}\rangle = |\bar{-}\rangle$.

---

## Fault Tolerance Connection

### Why Transversality Implies Fault Tolerance

**Definition (Fault-Tolerant Gate):** A gate implementation is **fault-tolerant** if:
1. A single fault in the implementation causes at most one error in each output code block
2. If the input has errors, they don't multiply beyond what's correctable

**Theorem:** Transversal gates are inherently fault-tolerant.

**Proof:**
- Single fault on qubit $i$ → single error on qubit $i$ (by transversality)
- Code can correct $t = \lfloor(d-1)/2\rfloor$ errors
- Since error doesn't spread, single fault is always correctable $\square$

### Comparison: Transversal vs. Non-Transversal

| Property | Transversal | Non-Transversal |
|----------|-------------|-----------------|
| Error spread | None | Possible |
| Fault tolerance | Automatic | Requires gadgets |
| Implementation | Simple | Complex |
| Universality | Limited | Possible |

---

## Worked Examples

### Example 1: Error Propagation Through Transversal Gate

**Problem:** Consider the [[7,1,3]] Steane code with a Z error on qubit 3. Track the error through transversal $\bar{H} = H^{\otimes 7}$.

**Solution:**

Initial error: $E = Z_3 = I \otimes I \otimes Z \otimes I \otimes I \otimes I \otimes I$

After transversal Hadamard:
$$H^{\otimes 7} Z_3 (H^{\otimes 7})^\dagger = H^{\otimes 7} Z_3 H^{\otimes 7}$$

Using $HZH = X$:
$$= I \otimes I \otimes (HZH) \otimes I \otimes I \otimes I \otimes I$$
$$= I \otimes I \otimes X \otimes I \otimes I \otimes I \otimes I$$
$$= X_3$$

**Result:** The Z error on qubit 3 becomes an X error on qubit 3. Weight is preserved!

**Syndrome Analysis:**
- Before: Z error detected by X stabilizers
- After: X error detected by Z stabilizers
- Both are single-qubit errors: correctable by the [[7,1,3]] code

### Example 2: Non-Transversal Gate Error Spread

**Problem:** Show how a logical CNOT within a single code block would spread errors.

**Solution:**

Consider a hypothetical internal CNOT that couples qubits 1 and 4:
$$\text{CNOT}_{1 \to 4}$$

Start with X error on qubit 1: $E = X_1$

After the gate:
$$\text{CNOT}_{1 \to 4} X_1 \text{CNOT}_{1 \to 4}^\dagger = X_1 X_4$$

The weight-1 error has become weight-2!

**Impact:** If we had $t+1$ such gates in sequence, a single error could spread to $t+1$ qubits, potentially exceeding the correction capability.

### Example 3: Transversal CNOT Between Blocks

**Problem:** Two logical qubits are encoded in separate [[7,1,3]] code blocks A and B. Show that $\overline{\text{CNOT}}_{A \to B} = \text{CNOT}^{\otimes 7}$ is transversal and analyze error behavior.

**Solution:**

The transversal CNOT applies:
$$\text{CNOT}_{A_1 \to B_1} \cdot \text{CNOT}_{A_2 \to B_2} \cdots \text{CNOT}_{A_7 \to B_7}$$

**Error on Block A, qubit 3:** $E = X_{A_3}$

$$\text{CNOT}^{\otimes 7} (X_{A_3}) (\text{CNOT}^{\otimes 7})^\dagger = X_{A_3} X_{B_3}$$

**Analysis:**
- Block A: 1 error (qubit 3)
- Block B: 1 error (qubit 3)
- Each block has only 1 error: both correctable!

This is the key property: errors spread *between* blocks (which is acceptable) but not *within* blocks.

---

## Practice Problems

### Level 1: Direct Application

**P1.1** For a [[5,1,3]] code, write out the transversal $\bar{X} = X^{\otimes 5}$ explicitly as a $32 \times 32$ matrix (or describe its action on the computational basis).

**P1.2** Verify that the composition of two transversal gates $\bar{U} = U^{\otimes n}$ and $\bar{V} = V^{\otimes n}$ is transversal.

**P1.3** If $\bar{U} = U^{\otimes n}$ is transversal and an error $E = Y_j$ occurs on qubit $j$, what is the error after the gate? Express in terms of $U$.

### Level 2: Intermediate

**P2.1** Prove that for any stabilizer code with stabilizer group $\mathcal{S}$, a transversal gate $\bar{U} = U^{\otimes n}$ is a valid logical operation if and only if $U^{\otimes n} \mathcal{S} (U^{\otimes n})^\dagger = \mathcal{S}$.

**P2.2** Consider a CSS code CSS(C1, C2). Show that $\bar{X} = X^{\otimes n}$ is a valid logical X if the all-ones vector $\mathbf{1} \in C_1 \setminus C_2^\perp$.

**P2.3** For the [[7,1,3]] Steane code, verify that $S^{\otimes 7}$ implements logical S (up to a global phase) by checking its action on logical $|\bar{0}\rangle$ and $|\bar{1}\rangle$.

### Level 3: Challenging

**P3.1** Prove that if a code has transversal $T = \text{diag}(1, e^{i\pi/4})$, then it cannot be a CSS code. (Hint: Consider how T transforms X and Z stabilizers.)

**P3.2** For a general [[n, k, d]] code with $k > 1$ logical qubits, characterize the structure of transversal gates. How do they act on the $k$-qubit logical space?

**P3.3** Design a family of codes where the transversal gate set approaches (but never reaches) universality. What is the maximum achievable gate set?

---

## Computational Lab

```python
"""
Day 855: Transversal Gate Analysis
===================================

Exploring transversal gates and their error propagation properties.
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import product

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Common single-qubit gates
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Two-qubit gates
def cnot():
    """CNOT gate matrix."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)


def tensor_power(U: np.ndarray, n: int) -> np.ndarray:
    """Compute U^{otimes n}."""
    result = U.copy()
    for _ in range(n - 1):
        result = np.kron(result, U)
    return result


def apply_single_qubit(U: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply single-qubit gate U to specified qubit in n-qubit system."""
    ops = [I] * n_qubits
    ops[qubit] = U
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def transversal_gate(U: np.ndarray, n: int) -> np.ndarray:
    """Create transversal gate U^{otimes n}."""
    return tensor_power(U, n)


def error_weight(error: np.ndarray, n_qubits: int) -> int:
    """
    Compute the weight of a Pauli error.
    Weight = number of qubits with non-identity action.
    """
    dim = 2 ** n_qubits
    # Decompose into Pauli basis (simplified check)
    # Check if error equals identity on each qubit
    weight = 0
    for q in range(n_qubits):
        # Project onto qubit q
        I_check = apply_single_qubit(I, q, n_qubits)
        if not np.allclose(error @ I_check, I_check @ error):
            weight += 1
    return weight


def conjugate_error(gate: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Compute gate @ error @ gate^dagger."""
    return gate @ error @ gate.conj().T


def pauli_to_string(P: np.ndarray) -> str:
    """Convert 2x2 Pauli matrix to string."""
    if np.allclose(P, I):
        return 'I'
    elif np.allclose(P, X):
        return 'X'
    elif np.allclose(P, Y):
        return 'Y'
    elif np.allclose(P, Z):
        return 'Z'
    else:
        return '?'


class StabilizerCode:
    """Simple representation of a stabilizer code."""

    def __init__(self, n: int, stabilizers: List[np.ndarray],
                 logical_x: np.ndarray, logical_z: np.ndarray):
        """
        Initialize stabilizer code.

        Parameters:
        -----------
        n : int
            Number of physical qubits
        stabilizers : List[np.ndarray]
            List of stabilizer generators (as matrices)
        logical_x : np.ndarray
            Logical X operator
        logical_z : np.ndarray
            Logical Z operator
        """
        self.n = n
        self.stabilizers = stabilizers
        self.logical_x = logical_x
        self.logical_z = logical_z

    def is_valid_logical(self, gate: np.ndarray) -> bool:
        """
        Check if gate is a valid logical operation.
        Must commute with all stabilizers.
        """
        for S in self.stabilizers:
            # Check if gate @ S @ gate^dag is still a stabilizer
            conjugated = conjugate_error(gate, S)
            # For now, just check commutation
            if not np.allclose(gate @ S, S @ gate):
                return False
        return True

    def logical_action(self, gate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the logical action of a gate.
        Returns transformed (X_L, Z_L).
        """
        new_x = conjugate_error(gate, self.logical_x)
        new_z = conjugate_error(gate, self.logical_z)
        return new_x, new_z


def steane_code() -> StabilizerCode:
    """Create the [[7,1,3]] Steane code (simplified)."""
    n = 7

    # Stabilizer generators as Pauli strings
    # X stabilizers: IIIXXXX, IXXIIXX, XIXIXIX
    # Z stabilizers: IIIZZZZ, IZZIIZZ, ZIZIZIZ

    def pauli_string_to_matrix(s: str) -> np.ndarray:
        """Convert Pauli string to matrix."""
        paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        result = paulis[s[0]]
        for c in s[1:]:
            result = np.kron(result, paulis[c])
        return result

    stabilizers = [
        pauli_string_to_matrix('IIIXXXX'),
        pauli_string_to_matrix('IXXIIXX'),
        pauli_string_to_matrix('XIXIXIX'),
        pauli_string_to_matrix('IIIZZZZ'),
        pauli_string_to_matrix('IZZIIZZ'),
        pauli_string_to_matrix('ZIZIZIZ'),
    ]

    logical_x = pauli_string_to_matrix('XXXXXXX')
    logical_z = pauli_string_to_matrix('ZZZZZZZ')

    return StabilizerCode(n, stabilizers, logical_x, logical_z)


def analyze_transversal_gate(code: StabilizerCode, U: np.ndarray,
                             gate_name: str = "U"):
    """
    Analyze a transversal gate U^{otimes n} on the code.
    """
    print(f"\nAnalyzing transversal {gate_name}^{{otimes {code.n}}}")
    print("-" * 50)

    trans_gate = transversal_gate(U, code.n)

    # Check if it's a valid logical operation
    is_valid = code.is_valid_logical(trans_gate)
    print(f"Valid logical gate: {is_valid}")

    if is_valid:
        # Determine logical action
        new_x, new_z = code.logical_action(trans_gate)

        # Compare to original
        x_preserved = np.allclose(new_x, code.logical_x) or \
                     np.allclose(new_x, -code.logical_x)
        z_preserved = np.allclose(new_z, code.logical_z) or \
                     np.allclose(new_z, -code.logical_z)

        print(f"X_L preservation: {x_preserved}")
        print(f"Z_L preservation: {z_preserved}")


def error_propagation_demo():
    """Demonstrate error propagation through transversal gates."""
    print("\n" + "=" * 60)
    print("Error Propagation Through Transversal Gates")
    print("=" * 60)

    n = 3  # Use 3 qubits for visualization

    # Create Z error on qubit 1
    Z1 = apply_single_qubit(Z, 0, n)
    print(f"\nInitial error: Z on qubit 1")

    # Apply transversal Hadamard
    H_trans = transversal_gate(H, n)
    Z1_after_H = conjugate_error(H_trans, Z1)

    # Z becomes X under Hadamard
    X1 = apply_single_qubit(X, 0, n)
    print(f"After H^{{otimes {n}}}: Error becomes X on qubit 1")
    print(f"Verify: {np.allclose(Z1_after_H, X1)}")

    # Create X error on qubit 2
    X2 = apply_single_qubit(X, 1, n)
    print(f"\nInitial error: X on qubit 2")

    # Apply transversal S
    S_trans = transversal_gate(S, n)
    X2_after_S = conjugate_error(S_trans, X2)

    # S X S^dag = Y (up to phase)
    Y2 = apply_single_qubit(Y, 1, n)
    print(f"After S^{{otimes {n}}}: Error becomes Y on qubit 2")
    print(f"Verify: {np.allclose(X2_after_S, Y2)}")


def non_transversal_demo():
    """Show error spreading with non-transversal gate."""
    print("\n" + "=" * 60)
    print("Error Spreading with Non-Transversal Gate")
    print("=" * 60)

    # 2-qubit example with CNOT
    print("\nCNOT from qubit 1 to qubit 2:")

    # X error on control (qubit 1)
    X1 = np.kron(X, I)
    print("Initial: X error on qubit 1 (control)")

    CNOT = cnot()
    X1_after_CNOT = conjugate_error(CNOT, X1)

    X1X2 = np.kron(X, X)
    print(f"After CNOT: Error spreads to X on both qubits")
    print(f"X1 -> X1 X2: {np.allclose(X1_after_CNOT, X1X2)}")

    # Z error on target (qubit 2)
    Z2 = np.kron(I, Z)
    print("\nInitial: Z error on qubit 2 (target)")

    Z2_after_CNOT = conjugate_error(CNOT, Z2)

    Z1Z2 = np.kron(Z, Z)
    print(f"After CNOT: Error spreads to Z on both qubits")
    print(f"Z2 -> Z1 Z2: {np.allclose(Z2_after_CNOT, Z1Z2)}")

    print("\nKey insight: CNOT is transversal BETWEEN code blocks,")
    print("not within a single block!")


def steane_transversal_analysis():
    """Analyze transversal gates on the Steane code."""
    print("\n" + "=" * 60)
    print("Steane Code [[7,1,3]] Transversal Gate Analysis")
    print("=" * 60)

    code = steane_code()

    # Test various transversal gates
    gates = [
        (X, "X"),
        (Z, "Z"),
        (H, "H"),
        (S, "S"),
        (T, "T")
    ]

    for U, name in gates:
        analyze_transversal_gate(code, U, name)


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Day 855: Transversal Gate Definition")
    print("=" * 60)

    # Part 1: Error propagation with transversal gates
    error_propagation_demo()

    # Part 2: Non-transversal gate error spread
    non_transversal_demo()

    # Part 3: Steane code analysis
    steane_transversal_analysis()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Transversal gates = U^{otimes n} structure")
    print("2. Errors don't spread within code blocks")
    print("3. Steane has transversal {X, Z, H, S, CNOT}")
    print("4. T is NOT transversal on Steane (or any CSS code)")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Transversal gate | $\bar{U} = U_1 \otimes U_2 \otimes \cdots \otimes U_n$ |
| Homogeneous transversal | $\bar{U} = U^{\otimes n}$ |
| Error conjugation | $\bar{U} E_j \bar{U}^\dagger = E'_j$ (localized) |
| Weight preservation | $\text{wt}(\bar{U} E \bar{U}^\dagger) = \text{wt}(E)$ |
| CNOT between blocks | $\overline{\text{CNOT}} = \text{CNOT}^{\otimes n}$ |

### Main Takeaways

1. **Transversal gates** act independently on each qubit: $\bar{U} = \bigotimes_i U_i$
2. **Error non-propagation** is the key property: single-qubit errors stay single-qubit
3. **Automatic fault tolerance**: transversal gates cannot spread errors within a code block
4. **Structural constraint**: transversal gates form a group, but this group is limited
5. **The T-gate problem**: T is not transversal on CSS codes, requiring alternative methods

---

## Daily Checklist

- [ ] I can define a transversal gate formally
- [ ] I understand why transversal gates don't spread errors
- [ ] I can compute error evolution through transversal gates
- [ ] I can identify transversal gates for the Steane code
- [ ] I understand the difference between transversal and non-transversal gates
- [ ] I can explain why transversality implies fault tolerance

---

## Preview: Day 856

Tomorrow we focus on **Transversal Gates on CSS Codes**:

- Why X and Z are always transversal on CSS codes
- The transversal CNOT between CSS code blocks
- When Hadamard is transversal (self-dual codes)
- Complete characterization of CSS transversal gates
- The S gate and phase considerations

Understanding which gates are transversal on CSS codes sets the stage for the Eastin-Knill theorem.
