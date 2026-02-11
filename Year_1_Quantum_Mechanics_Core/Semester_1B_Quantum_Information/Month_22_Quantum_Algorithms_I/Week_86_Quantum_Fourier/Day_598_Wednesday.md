# Day 598: QFT Circuit Construction

## Overview

**Day 598** | Week 86, Day 3 | Month 22 | Quantum Algorithms I

Today we construct the quantum circuit that implements the QFT. The circuit uses only Hadamard gates and controlled phase rotation gates, achieving $O(n^2)$ gate complexity. This efficient construction is what makes the QFT practically useful for quantum algorithms.

---

## Learning Objectives

1. Define the controlled rotation gates $R_k$
2. Derive the QFT circuit from the product representation
3. Implement the complete QFT circuit with SWAP gates
4. Analyze gate count and circuit depth
5. Understand the bit-reversal in QFT output
6. Compare approximate vs exact QFT

---

## Core Content

### Controlled Rotation Gates

The **controlled-$R_k$** gate applies a phase rotation controlled by another qubit:

$$R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i/2^k} \end{pmatrix}$$

**Phase angles:**
- $R_1 = Z$ (phase $\pi$)
- $R_2 = S$ (phase $\pi/2$)
- $R_3 = T$ (phase $\pi/4$)
- $R_k$: phase $2\pi/2^k = \pi/2^{k-1}$

The controlled version:
$$CR_k = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & e^{2\pi i/2^k}
\end{pmatrix}$$

### Deriving the Circuit from Product Representation

Recall the product representation:
$$QFT|j_1 j_2 \cdots j_n\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{l=1}^{n} \left(|0\rangle + e^{2\pi i \cdot 0.j_{n-l+1} \cdots j_n}|1\rangle\right)$$

For the **first output qubit** (l=1):
$$\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 0.j_n}|1\rangle)$$

Since $0.j_n = j_n/2$, the phase is $e^{i\pi j_n}$:
- If $j_n = 0$: $|+\rangle$
- If $j_n = 1$: $|-\rangle$

This is just $H|j_n\rangle$!

For the **second output qubit** (l=2):
$$\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i \cdot 0.j_{n-1}j_n}|1\rangle)$$

Phase = $e^{i\pi(j_{n-1} + j_n/2)}$

This requires:
1. Hadamard on $j_{n-1}$
2. Controlled-$R_2$ from $j_n$ to $j_{n-1}$

### The General Pattern

To create output qubit $l$ with phase $0.j_{n-l+1} \cdots j_n$:

1. Apply Hadamard to qubit $n-l+1$
2. Apply controlled-$R_2$ from qubit $n-l+2$
3. Apply controlled-$R_3$ from qubit $n-l+3$
4. ... and so on through controlled-$R_l$ from qubit $n$

### Complete QFT Circuit (3 qubits)

```
j₁ ─[H]─[R₂]─[R₃]─────────────────×─── output qubit 3
         │    │                   │
j₂ ──────●────┼───[H]─[R₂]────────┼─── output qubit 2
              │        │          │
j₃ ───────────●────────●────[H]───×─── output qubit 1
```

**Note:** Output qubits are in reverse order! SWAP gates (×) fix this.

### QFT Circuit Algorithm

```
QFT(n qubits):
    for i = 1 to n:
        Apply H to qubit i
        for j = i+1 to n:
            Apply controlled-R_{j-i+1} with control j, target i

    Reverse qubit order (SWAP gates)
```

### With Explicit Gates (n=3)

**Step 1:** Process qubit 1
- H on qubit 1
- CR₂ (control=2, target=1)
- CR₃ (control=3, target=1)

**Step 2:** Process qubit 2
- H on qubit 2
- CR₂ (control=3, target=2)

**Step 3:** Process qubit 3
- H on qubit 3

**Step 4:** SWAP qubits 1 and 3

### Gate Count Analysis

For n qubits:
- **Hadamard gates:** $n$
- **Controlled rotations:** $\frac{n(n-1)}{2}$
- **SWAP gates:** $\lfloor n/2 \rfloor$

Total: $\frac{n(n+1)}{2} + \lfloor n/2 \rfloor = O(n^2)$ gates

**Circuit depth:** $O(n)$ with parallelization, $O(n^2)$ sequential

### SWAP Gate Implementation

The SWAP gate exchanges two qubits:
$$SWAP|a\rangle|b\rangle = |b\rangle|a\rangle$$

Circuit implementation using 3 CNOTs:
```
a ───●───⊕───●─── b
     │   │   │
b ───⊕───●───⊕─── a
```

Or use the identity: $SWAP = (CNOT_{12})(CNOT_{21})(CNOT_{12})$

### Approximate QFT

For large $n$, small rotations ($R_k$ for large $k$) contribute little.

**Approximate QFT:** Omit rotations with $k > m$ for some cutoff $m$.

**Error:** $O(n/2^m)$

**Gate count:** $O(nm)$ instead of $O(n^2)$

For $m = O(\log n)$: $O(n \log n)$ gates with polynomial precision.

---

## Worked Examples

### Example 1: 2-Qubit QFT Circuit

Draw and trace the 2-qubit QFT circuit.

**Solution:**

**Circuit:**
```
q₁ ─[H]─[CR₂]──×─── out₂
         │     │
q₂ ──────●──[H]×─── out₁
```

**Trace for input |01⟩** ($j_1=0$, $j_2=1$):

Initial: $|01\rangle$

After H on q₁: $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)|1\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |11\rangle)$

After CR₂ (q₂ controls q₁):
- When q₂=1, apply R₂ to q₁, giving phase $e^{i\pi/2}=i$ to $|1\rangle$
- $\frac{1}{\sqrt{2}}(|01\rangle + i|11\rangle)$

After H on q₂:
- $|1\rangle \to \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$
- $\frac{1}{2}(|00\rangle - |01\rangle + i|10\rangle - i|11\rangle)$

After SWAP:
- $\frac{1}{2}(|00\rangle - |10\rangle + i|01\rangle - i|11\rangle)$
- $= \frac{1}{2}(|00\rangle + i|01\rangle - |10\rangle - i|11\rangle)$

This matches $QFT|01\rangle$ from Day 597!

### Example 2: Gate Sequence for 4-Qubit QFT

List the gate sequence (before SWAPs) for 4-qubit QFT.

**Solution:**

| Step | Gate | Control | Target |
|------|------|---------|--------|
| 1 | H | - | 1 |
| 2 | CR₂ | 2 | 1 |
| 3 | CR₃ | 3 | 1 |
| 4 | CR₄ | 4 | 1 |
| 5 | H | - | 2 |
| 6 | CR₂ | 3 | 2 |
| 7 | CR₃ | 4 | 2 |
| 8 | H | - | 3 |
| 9 | CR₂ | 4 | 3 |
| 10 | H | - | 4 |

Then: SWAP(1,4), SWAP(2,3)

Total: 10 rotation/Hadamard gates + 2 SWAPs = 16 primitive gates (using CNOT decomposition).

### Example 3: Controlled-R_k Decomposition

Express CR₃ in terms of basic gates.

**Solution:**

$R_3 = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = T$ gate

Controlled-T can be decomposed as:
$$CT = (I \otimes T^{1/2}) \cdot CNOT \cdot (I \otimes T^{-1/2})$$

where $T^{1/2}$ applies phase $e^{i\pi/8}$.

More generally, using the standard decomposition for controlled-U.

---

## Practice Problems

### Problem 1: 3-Qubit Trace

Trace the 3-qubit QFT circuit for input $|110\rangle$.

### Problem 2: Gate Count Verification

Verify that the 5-qubit QFT requires exactly 15 controlled rotations (not counting Hadamards or SWAPs).

### Problem 3: Approximate QFT

For 8-qubit QFT with cutoff $m=4$, how many controlled rotations are needed?

### Problem 4: Inverse QFT

Write the inverse QFT circuit for 2 qubits (reverse the order of operations and conjugate phases).

---

## Computational Lab

```python
"""Day 598: QFT Circuit Construction"""
import numpy as np
from typing import List, Tuple

# Gate definitions
def H():
    """Hadamard gate"""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def Rk(k):
    """R_k rotation gate: phase 2π/2^k"""
    phase = np.exp(2j * np.pi / (2**k))
    return np.array([[1, 0], [0, phase]])

def controlled_gate(U):
    """Create controlled version of single-qubit gate U"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, U[0,0], U[0,1]],
        [0, 0, U[1,0], U[1,1]]
    ])

def SWAP():
    """SWAP gate"""
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

def apply_gate(state, gate, qubits, n_total):
    """
    Apply gate to specified qubits in an n-qubit state
    qubits: list of qubit indices (0-indexed from top)
    """
    # Build full gate matrix using tensor products
    if len(qubits) == 1:
        q = qubits[0]
        full_gate = np.eye(1)
        for i in range(n_total):
            if i == q:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
    elif len(qubits) == 2:
        # For 2-qubit gates, need more careful construction
        c, t = qubits  # control, target
        dim = 2**n_total
        full_gate = np.eye(dim, dtype=complex)

        # Apply controlled gate
        for i in range(dim):
            bits = [(i >> (n_total-1-j)) & 1 for j in range(n_total)]
            if bits[c] == 1:  # Control is 1
                # Apply single-qubit gate to target
                old_t = bits[t]
                for new_t in range(2):
                    bits_new = bits.copy()
                    bits_new[t] = new_t
                    j = sum(bits_new[k] << (n_total-1-k) for k in range(n_total))
                    # gate element from old_t to new_t
                    full_gate[j, i] = gate[new_t, old_t]
                    if new_t != old_t:
                        full_gate[i, i] = 0
    return full_gate @ state

def qft_circuit(n: int, state: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Apply QFT circuit to state
    Returns the transformed state
    """
    current_state = state.copy()
    gate_count = {'H': 0, 'CR': 0, 'SWAP': 0}

    if verbose:
        print(f"\n--- QFT Circuit for {n} qubits ---")

    # Main QFT operations
    for i in range(n):
        # Hadamard on qubit i
        current_state = apply_gate(current_state, H(), [i], n)
        gate_count['H'] += 1

        if verbose:
            print(f"H on qubit {i}")

        # Controlled rotations
        for j in range(i+1, n):
            k = j - i + 1  # R_k parameter
            current_state = apply_gate(current_state, Rk(k), [j, i], n)
            gate_count['CR'] += 1

            if verbose:
                print(f"CR_{k} (control={j}, target={i})")

    # SWAP for bit reversal
    for i in range(n // 2):
        j = n - 1 - i
        # Apply SWAP between qubits i and j
        current_state = apply_swap(current_state, i, j, n)
        gate_count['SWAP'] += 1

        if verbose:
            print(f"SWAP({i}, {j})")

    if verbose:
        print(f"\nGate count: {gate_count}")
        print(f"Total gates: {sum(gate_count.values())}")

    return current_state

def apply_swap(state, q1, q2, n):
    """Apply SWAP between qubits q1 and q2"""
    dim = 2**n
    new_state = np.zeros(dim, dtype=complex)

    for i in range(dim):
        bits = [(i >> (n-1-j)) & 1 for j in range(n)]
        # Swap bits at positions q1 and q2
        bits[q1], bits[q2] = bits[q2], bits[q1]
        j = sum(bits[k] << (n-1-k) for k in range(n))
        new_state[j] = state[i]

    return new_state

def qft_matrix(n: int) -> np.ndarray:
    """QFT matrix for verification"""
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega**(j*k) for k in range(N)]
                     for j in range(N)]) / np.sqrt(N)

# Test QFT circuit
print("=" * 60)
print("QFT CIRCUIT CONSTRUCTION TEST")
print("=" * 60)

# Test for different n
for n in [2, 3, 4]:
    print(f"\n{'='*40}")
    print(f"Testing {n}-qubit QFT circuit")
    print(f"{'='*40}")

    # Test all basis states
    all_passed = True
    for j in range(2**n):
        # Create basis state |j⟩
        state = np.zeros(2**n, dtype=complex)
        state[j] = 1

        # Apply QFT circuit
        circuit_result = qft_circuit(n, state, verbose=(n==2 and j==1))

        # Compare with QFT matrix
        matrix_result = qft_matrix(n) @ state

        if not np.allclose(circuit_result, matrix_result):
            print(f"  FAILED for |{j:0{n}b}⟩")
            all_passed = False

    if all_passed:
        print(f"  All {2**n} basis states: PASSED")

# Gate count analysis
print("\n" + "=" * 60)
print("GATE COUNT ANALYSIS")
print("=" * 60)

print("\n| n | Hadamard | Controlled-R | SWAP | Total |")
print("|---|----------|--------------|------|-------|")

for n in range(2, 9):
    h_count = n
    cr_count = n * (n - 1) // 2
    swap_count = n // 2
    total = h_count + cr_count + swap_count
    print(f"| {n} | {h_count:8d} | {cr_count:12d} | {swap_count:4d} | {total:5d} |")

# Visualize the circuit structure
print("\n" + "=" * 60)
print("CIRCUIT VISUALIZATION (3 qubits)")
print("=" * 60)

def draw_qft_circuit(n: int):
    """Create ASCII visualization of QFT circuit"""
    lines = [f"q{i}: " for i in range(n)]

    # Hadamard and controlled rotations
    for i in range(n):
        # Add Hadamard
        for j in range(n):
            if j == i:
                lines[j] += "[H]"
            else:
                lines[j] += "───"

        # Add controlled rotations
        for k_idx in range(i+1, n):
            k = k_idx - i + 1
            for j in range(n):
                if j == i:
                    lines[j] += f"[R{k}]"
                elif j == k_idx:
                    lines[j] += f"─●──"
                elif j > i and j < k_idx:
                    lines[j] += "─│──"
                else:
                    lines[j] += "────"

        # Separator
        for j in range(n):
            lines[j] += "─"

    # SWAPs
    for i in range(n // 2):
        j = n - 1 - i
        for q in range(n):
            if q == i or q == j:
                lines[q] += "×"
            elif q > i and q < j:
                lines[q] += "│"
            else:
                lines[q] += "─"
        for q in range(n):
            lines[q] += "─"

    return "\n".join(lines)

print(draw_qft_circuit(3))

# Approximate QFT analysis
print("\n" + "=" * 60)
print("APPROXIMATE QFT ANALYSIS")
print("=" * 60)

def approximate_qft_error(n: int, m: int) -> float:
    """
    Estimate error when omitting rotations R_k for k > m
    """
    # Each omitted rotation contributes phase error up to 2π/2^k
    # Total error bounded by sum of omitted angles
    error = 0
    for i in range(n):
        for j in range(i+1, n):
            k = j - i + 1
            if k > m:
                error += 2 * np.pi / (2**k)
    return error

print("\n| n  | m  | Omitted CR | Error bound |")
print("|----|----|-----------:|-------------|")

for n in [8, 16, 32]:
    for m in [2, 4, 8]:
        full_cr = n * (n-1) // 2
        approx_cr = sum(min(k, m-1) for k in range(1, n))
        omitted = full_cr - approx_cr
        error = approximate_qft_error(n, m)
        print(f"| {n:2d} | {m:2d} | {omitted:10d} | {error:.6f}    |")

# Phase accumulation demonstration
print("\n" + "=" * 60)
print("PHASE ACCUMULATION IN QFT")
print("=" * 60)

n = 3
j = 0b101  # |101⟩
j_bits = [(j >> (n-1-i)) & 1 for i in range(n)]

print(f"\nInput: |{j:03b}⟩ = |j₁j₂j₃⟩ where j₁={j_bits[0]}, j₂={j_bits[1]}, j₃={j_bits[2]}")
print("\nPhase accumulation for each output qubit:")

for l in range(n):
    # Output qubit l has phase 0.j_{n-l}...j_n
    phase_bits = j_bits[n-l-1:]
    phase_value = sum(b / (2**(i+1)) for i, b in enumerate(phase_bits))
    phase_angle = 2 * np.pi * phase_value

    print(f"\n  Output qubit {l+1}:")
    print(f"    Phase bits: 0.{''.join(str(b) for b in phase_bits)}")
    print(f"    Phase value: {phase_value:.4f}")
    print(f"    Phase angle: {phase_angle:.4f} rad = {np.degrees(phase_angle):.1f}°")
    print(f"    Contributed by:", end=" ")

    contributions = []
    for i, b in enumerate(phase_bits):
        if b == 1:
            if i == 0:
                contributions.append("H")
            else:
                contributions.append(f"CR_{i+1}")

    print(", ".join(contributions) if contributions else "none (phase 0)")

# Save circuit diagram
import matplotlib.pyplot as plt

def visualize_qft_circuit_matplotlib(n: int, filename: str):
    """Create publication-quality circuit diagram"""
    fig, ax = plt.subplots(figsize=(14, 2*n))

    # Calculate positions
    gate_width = 0.8
    gate_spacing = 1.2

    # Count total steps
    total_steps = 0
    for i in range(n):
        total_steps += 1  # Hadamard
        total_steps += (n - i - 1)  # Controlled rotations
    total_steps += n // 2  # SWAPs

    ax.set_xlim(-1, total_steps * gate_spacing + 2)
    ax.set_ylim(-0.5, n + 0.5)
    ax.axis('off')

    # Draw qubit lines
    for i in range(n):
        ax.plot([-0.5, total_steps * gate_spacing + 1], [n-1-i, n-1-i],
               'k-', linewidth=1)
        ax.text(-0.8, n-1-i, f'$q_{i+1}$', fontsize=12, va='center', ha='right')

    # Draw gates
    step = 0
    for i in range(n):
        # Hadamard
        y = n - 1 - i
        x = step * gate_spacing
        rect = plt.Rectangle((x-0.3, y-0.3), 0.6, 0.6,
                            facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, 'H', fontsize=10, ha='center', va='center')
        step += 1

        # Controlled rotations
        for j in range(i+1, n):
            k = j - i + 1
            x = step * gate_spacing
            y_target = n - 1 - i
            y_control = n - 1 - j

            # Control dot
            ax.plot(x, y_control, 'ko', markersize=8)

            # Target gate
            rect = plt.Rectangle((x-0.3, y_target-0.3), 0.6, 0.6,
                                facecolor='lightyellow', edgecolor='black')
            ax.add_patch(rect)
            ax.text(x, y_target, f'$R_{k}$', fontsize=9, ha='center', va='center')

            # Vertical line
            ax.plot([x, x], [y_control, y_target], 'k-', linewidth=1)
            step += 1

    # SWAPs
    for i in range(n // 2):
        j = n - 1 - i
        x = step * gate_spacing
        y1 = n - 1 - i
        y2 = n - 1 - j

        ax.plot(x, y1, 'x', markersize=15, mew=2, color='red')
        ax.plot(x, y2, 'x', markersize=15, mew=2, color='red')
        ax.plot([x, x], [y1, y2], 'r-', linewidth=1)
        step += 1

    ax.set_title(f'{n}-Qubit QFT Circuit', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Circuit saved to '{filename}'")

visualize_qft_circuit_matplotlib(4, 'qft_circuit_4qubit.png')
```

---

## Summary

### Key Formulas

| Gate | Matrix | Phase |
|------|--------|-------|
| $R_k$ | $\begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i/2^k} \end{pmatrix}$ | $2\pi/2^k$ |
| Gate count | $\frac{n(n+1)}{2}$ | $O(n^2)$ |

### QFT Circuit Structure

1. For qubit $i$ from 1 to $n$:
   - Apply Hadamard
   - Apply CR$_k$ from qubits $i+1, \ldots, n$
2. Apply SWAPs to reverse bit order

### Key Takeaways

1. **Product representation** leads directly to circuit construction
2. **Controlled rotations** accumulate phases from input bits
3. **SWAP gates** correct the bit-reversal in output
4. **Approximate QFT** reduces gate count with bounded error
5. **$O(n^2)$ complexity** enables efficient quantum algorithms

---

## Daily Checklist

- [ ] I can write out the QFT circuit for small n
- [ ] I understand the role of each controlled rotation
- [ ] I can trace the circuit for a specific input
- [ ] I know why SWAPs are needed
- [ ] I understand approximate QFT tradeoffs
- [ ] I ran the lab and verified circuit correctness

---

*Next: Day 599 - Phase Kickback and QFT*
