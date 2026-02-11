# Day 618: The Grover Oracle

## Overview
**Day 618** | Week 89, Day 2 | Year 1, Month 23 | Grover's Search Algorithm

Today we construct the Grover oracle—the quantum gate that marks the solution states by flipping their phase. We explore both the mathematical formulation and circuit implementations.

---

## Learning Objectives

1. Define the phase oracle mathematically
2. Construct oracle circuits for specific functions
3. Understand oracle-as-black-box vs explicit construction
4. Implement multi-controlled phase gates
5. Analyze oracle query complexity
6. Build oracles for practical search problems

---

## Core Content

### The Phase Oracle

The Grover oracle (also called "phase oracle" or "marking oracle") acts as:

$$\boxed{O_f|x\rangle = (-1)^{f(x)}|x\rangle}$$

where $f(x) = 1$ for marked states and $f(x) = 0$ otherwise.

**Matrix Representation (single marked state $w$):**

$$O_f = I - 2|w\rangle\langle w|$$

This is a **reflection** about the hyperplane orthogonal to $|w\rangle$.

### From Query Oracle to Phase Oracle

**Query Oracle:** $O_q|x\rangle|b\rangle = |x\rangle|b \oplus f(x)\rangle$

**Phase Oracle Construction:**
1. Prepare ancilla in $|{-}\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$
2. Apply query oracle
3. The phase kickback creates $(-1)^{f(x)}|x\rangle|{-}\rangle$

**Derivation:**
$$O_q|x\rangle|{-}\rangle = |x\rangle\frac{1}{\sqrt{2}}(|f(x)\rangle - |1 \oplus f(x)\rangle)$$

If $f(x) = 0$: state remains $|x\rangle|{-}\rangle$
If $f(x) = 1$: state becomes $|x\rangle\frac{1}{\sqrt{2}}(|1\rangle - |0\rangle) = -|x\rangle|{-}\rangle$

Therefore: $O_q|x\rangle|{-}\rangle = (-1)^{f(x)}|x\rangle|{-}\rangle$

### Oracle Circuit Construction

For a marked state $|w\rangle = |w_{n-1}...w_1w_0\rangle$:

```
     ┌───────┐
|x_0⟩─┤       ├──
     │       │
|x_1⟩─┤  MCZ  ├── (with X gates for 0-bits)
     │       │
  ⋮  │       │
     │       │
|x_n⟩─┤       ├──
     └───────┘
```

**Algorithm:**
1. Apply X gate to each qubit where $w_i = 0$
2. Apply multi-controlled Z (MCZ) gate
3. Apply X gate again to restore (uncompute)

### Multi-Controlled Z Gate

The MCZ gate applies a Z (phase flip) only when all control qubits are $|1\rangle$:

$$\text{MCZ} = I - 2|11...1\rangle\langle 11...1|$$

**Circuit decomposition using Toffoli gates:**

For 3 qubits:
```
     ────●────●────
         │    │
     ────●────┼────
         │    │
     ────Z────●────
```

Or using ancilla qubits for larger systems.

### Oracle for Multiple Marked States

For $M$ marked states $\{w_1, w_2, ..., w_M\}$:

$$O_f = I - 2\sum_{i=1}^{M}|w_i\rangle\langle w_i|$$

**Linearity property:** The oracle for multiple solutions is the product of single-solution oracles (up to global phase considerations).

### Example: Oracle for SAT Problems

For Boolean satisfiability, we mark states satisfying a formula $\phi(x_1, ..., x_n)$:

$$f(x) = \begin{cases} 1 & \text{if } \phi(x) = \text{TRUE} \\ 0 & \text{otherwise} \end{cases}$$

**Circuit construction:**
1. Encode the formula in reversible gates
2. Compute result into ancilla
3. Apply controlled-Z
4. Uncompute the formula

### Oracle Complexity Considerations

The oracle is treated as a "black box" for complexity analysis, but in practice:

- Simple functions: Oracle depth $O(\text{poly}(n))$
- NP-complete problems: Oracle construction may dominate
- Database search: Oracle represents database lookup

**Query complexity** counts oracle calls, not gate operations within the oracle.

---

## Worked Examples

### Example 1: Two-Qubit Oracle
Construct the oracle for marking $|w\rangle = |10\rangle$.

**Solution:**
The oracle should satisfy:
- $O_f|00\rangle = |00\rangle$
- $O_f|01\rangle = |01\rangle$
- $O_f|10\rangle = -|10\rangle$
- $O_f|11\rangle = |11\rangle$

Matrix form:
$$O_f = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Circuit:**
```
|x_1⟩ ───●─────
         │
|x_0⟩ ─X─●─X───  (CZ with X gates on x_0)
```

Or equivalently: $O_f = (X \otimes I) \cdot CZ \cdot (X \otimes I)$

### Example 2: Three-Qubit Oracle for |101⟩
Construct the oracle marking $|w\rangle = |101\rangle$.

**Solution:**
Need phase flip when $x_2=1, x_1=0, x_0=1$.

**Circuit:**
```
|x_2⟩ ─────●─────
           │
|x_1⟩ ──X──●──X──
           │
|x_0⟩ ─────●─────
           │
          [Z]
```

The middle qubit gets X gates to flip the control condition.

### Example 3: Oracle for Two Marked States
Construct oracle for marking both $|01\rangle$ and $|10\rangle$.

**Solution:**

Method 1: Apply individual oracles sequentially
$$O_f = O_{|10\rangle} \cdot O_{|01\rangle}$$

Method 2: Direct construction
$$O_f = I - 2(|01\rangle\langle 01| + |10\rangle\langle 10|)$$

Matrix:
$$O_f = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

This is the SWAP gate with a phase: $O_f = Z \otimes Z$

---

## Practice Problems

### Problem 1: Oracle Verification
Verify that $O_f = I - 2|w\rangle\langle w|$ satisfies:
a) $O_f|w\rangle = -|w\rangle$
b) $O_f|x\rangle = |x\rangle$ for $x \neq w$
c) $O_f^2 = I$ (involution)

### Problem 2: Phase Kickback
Starting with $|x\rangle|{-}\rangle$, trace through the phase kickback mechanism for $f(x) = 1$.

### Problem 3: Oracle for Parity
Construct an oracle that marks states with even parity (even number of 1s) for 3 qubits.

---

## Computational Lab

```python
"""Day 618: Grover Oracle Construction"""
import numpy as np
from scipy.linalg import block_diag

def phase_oracle_single(n, marked_state):
    """
    Construct phase oracle for a single marked state.

    Args:
        n: Number of qubits
        marked_state: Integer representing the marked state

    Returns:
        2^n x 2^n unitary matrix
    """
    N = 2**n
    oracle = np.eye(N)
    oracle[marked_state, marked_state] = -1
    return oracle

def phase_oracle_multiple(n, marked_states):
    """
    Construct phase oracle for multiple marked states.

    Args:
        n: Number of qubits
        marked_states: List of integers representing marked states

    Returns:
        2^n x 2^n unitary matrix
    """
    N = 2**n
    oracle = np.eye(N)
    for m in marked_states:
        oracle[m, m] = -1
    return oracle

def verify_oracle_properties(oracle, marked_states):
    """Verify oracle properties."""
    N = oracle.shape[0]

    print("Oracle Verification:")
    print("-" * 40)

    # Check marked states get phase flip
    for m in marked_states:
        state = np.zeros(N)
        state[m] = 1
        result = oracle @ state
        expected_phase = -1
        actual_phase = result[m]
        print(f"  |{m:b}⟩ → phase = {actual_phase:.1f} (expected {expected_phase})")

    # Check unmarked states unchanged
    unmarked = [i for i in range(N) if i not in marked_states]
    for u in unmarked[:2]:  # Just check first two
        state = np.zeros(N)
        state[u] = 1
        result = oracle @ state
        print(f"  |{u:b}⟩ → phase = {result[u]:.1f} (expected 1)")

    # Check O^2 = I
    O_squared = oracle @ oracle
    is_identity = np.allclose(O_squared, np.eye(N))
    print(f"  O² = I: {is_identity}")

    # Check unitarity
    is_unitary = np.allclose(oracle @ oracle.T.conj(), np.eye(N))
    print(f"  Unitary: {is_unitary}")

def oracle_circuit_simulation(n, marked_state):
    """
    Simulate oracle circuit construction using basic gates.
    """
    # Basic gates
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    def tensor_product(*matrices):
        """Compute tensor product of matrices."""
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    def controlled_z(n, controls, target):
        """Multi-controlled Z gate."""
        N = 2**n
        gate = np.eye(N)
        # Flip phase of state where all controls and target are 1
        control_mask = sum(2**c for c in controls)
        target_mask = 2**target
        flip_state = control_mask | target_mask
        gate[flip_state, flip_state] = -1
        return gate

    # Get binary representation of marked state
    binary = format(marked_state, f'0{n}b')
    print(f"\nCircuit for marking |{binary}⟩:")

    # Determine which qubits need X gates (those with 0 in marked state)
    x_positions = [i for i, b in enumerate(reversed(binary)) if b == '0']
    print(f"  X gates on qubits: {x_positions}")

    # Build the circuit
    N = 2**n

    # Step 1: Apply X gates
    x_layer = np.eye(N)
    for pos in x_positions:
        single_x = [I] * n
        single_x[pos] = X
        x_gate = tensor_product(*reversed(single_x))
        x_layer = x_gate @ x_layer

    # Step 2: Multi-controlled Z
    mcz = np.eye(N)
    mcz[N-1, N-1] = -1  # Flip |11...1⟩

    # Complete circuit: X @ MCZ @ X
    circuit = x_layer @ mcz @ x_layer

    return circuit

# Test oracle construction
print("=" * 50)
print("Phase Oracle Construction Tests")
print("=" * 50)

# Two-qubit oracle
n = 2
marked = 2  # |10⟩
oracle = phase_oracle_single(n, marked)
print(f"\nOracle for n={n}, marked=|{marked:02b}⟩:")
print(oracle)
verify_oracle_properties(oracle, [marked])

# Three-qubit oracle
n = 3
marked = 5  # |101⟩
oracle = phase_oracle_single(n, marked)
print(f"\nOracle for n={n}, marked=|{marked:03b}⟩:")
verify_oracle_properties(oracle, [marked])

# Multiple marked states
n = 2
marked_list = [1, 2]  # |01⟩ and |10⟩
oracle = phase_oracle_multiple(n, marked_list)
print(f"\nOracle for multiple marked states |01⟩, |10⟩:")
print(oracle)
verify_oracle_properties(oracle, marked_list)

# Circuit construction
circuit = oracle_circuit_simulation(3, 5)
direct = phase_oracle_single(3, 5)
print(f"\nCircuit matches direct construction: {np.allclose(circuit, direct)}")

# Visualize oracle action
import matplotlib.pyplot as plt

n = 3
marked = 5
N = 2**n

# Initial uniform superposition
psi_0 = np.ones(N) / np.sqrt(N)

# Apply oracle
oracle = phase_oracle_single(n, marked)
psi_1 = oracle @ psi_0

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Before oracle
axes[0].bar(range(N), psi_0, color='blue', alpha=0.7)
axes[0].set_xlabel('State')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Before Oracle')
axes[0].set_xticks(range(N))
axes[0].set_xticklabels([f'|{i:03b}⟩' for i in range(N)], rotation=45)

# After oracle
colors = ['red' if i == marked else 'blue' for i in range(N)]
axes[1].bar(range(N), psi_1, color=colors, alpha=0.7)
axes[1].set_xlabel('State')
axes[1].set_ylabel('Amplitude')
axes[1].set_title(f'After Oracle (marked = |{marked:03b}⟩)')
axes[1].set_xticks(range(N))
axes[1].set_xticklabels([f'|{i:03b}⟩' for i in range(N)], rotation=45)

plt.tight_layout()
plt.savefig('grover_oracle.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Expected Output:**
```
==================================================
Phase Oracle Construction Tests
==================================================

Oracle for n=2, marked=|10⟩:
[[ 1.  0.  0.  0.]
 [ 0.  1.  0.  0.]
 [ 0.  0. -1.  0.]
 [ 0.  0.  0.  1.]]
Oracle Verification:
----------------------------------------
  |10⟩ → phase = -1.0 (expected -1)
  |0⟩ → phase = 1.0 (expected 1)
  |1⟩ → phase = 1.0 (expected 1)
  O² = I: True
  Unitary: True
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Phase oracle | $O_f\|x\rangle = (-1)^{f(x)}\|x\rangle$ |
| Reflection form | $O_f = I - 2\|w\rangle\langle w\|$ |
| Multiple solutions | $O_f = I - 2\sum_i\|w_i\rangle\langle w_i\|$ |
| Involution | $O_f^2 = I$ |

### Key Takeaways

1. **Phase oracle** flips the phase of marked states
2. **Phase kickback** converts query oracle to phase oracle
3. **Circuit construction** uses X gates and multi-controlled Z
4. **Multiple marked states** can be handled by sequential oracles
5. **Oracle is a reflection** about the orthogonal complement of $|w\rangle$
6. **Query complexity** counts oracle calls as unit operations

---

## Daily Checklist

- [ ] I can define the phase oracle mathematically
- [ ] I understand the phase kickback mechanism
- [ ] I can construct oracle circuits for specific marked states
- [ ] I can handle multiple marked states
- [ ] I understand the reflection interpretation
- [ ] I ran the computational lab and verified oracle properties

---

*Next: Day 619 — The Diffusion Operator*
