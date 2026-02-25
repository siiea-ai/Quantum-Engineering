# Day 607: Iterative Phase Estimation

## Overview

**Day 607** | Week 87, Day 5 | Month 22 | Quantum Algorithms I

Today we study iterative phase estimation, which uses only a single ancilla qubit to extract phase bits one at a time. This approach requires fewer qubits but more measurements.

---

## Learning Objectives

1. Understand the single-ancilla approach
2. Learn the iterative bit extraction process
3. Compare standard vs iterative QPE
4. Analyze trade-offs in resources
5. Implement iterative QPE

---

## Core Content

### Motivation

Standard QPE needs $n$ ancilla qubits for $n$-bit precision. In near-term devices, qubits are precious!

**Iterative QPE:** Use 1 ancilla qubit, extract bits sequentially.

### The Key Insight

The phase $\phi = 0.b_1 b_2 b_3 \cdots b_n$ can be extracted bit by bit:

1. **Round 1:** Extract $b_n$ (least significant bit)
2. **Round 2:** Use knowledge of $b_n$ to extract $b_{n-1}$
3. **Continue:** Each round extracts one more bit

### Single-Ancilla Circuit

```
|0⟩ ─[H]─●─[R_z(-θ)]─[H]─ Measure → b_k
         │
|ψ⟩ ────U^{2^{n-k}}──────────────────
```

The rotation $R_z(-\theta)$ corrects for previously measured bits.

### Algorithm

```
Iterative-QPE(U, |ψ⟩, n):
    θ = 0  # Accumulated correction
    for k = n down to 1:
        1. Prepare |0⟩|ψ⟩
        2. Apply H to ancilla
        3. Apply controlled-U^{2^{n-k}}
        4. Apply R_z(-2πθ) to ancilla  # Correction
        5. Apply H to ancilla
        6. Measure ancilla → b_k
        7. Update θ = θ/2 + b_k/2
    return 0.b_1 b_2 ... b_n
```

### Why Correction is Needed

After measuring $b_n$, the phase of interest becomes:
$$\phi - 0.0\cdots 0 b_n = 0.b_1 b_2 \cdots b_{n-1} 0$$

The correction rotation removes the already-known bits.

### Resource Comparison

| Resource | Standard QPE | Iterative QPE |
|----------|--------------|---------------|
| Ancilla qubits | $n$ | $1$ |
| Controlled-U operations | $n$ (in parallel) | $n$ (sequential) |
| Total measurements | $1$ | $n$ |
| Circuit depth | $O(n)$ | $O(1)$ per round |

### Advantages and Disadvantages

**Advantages:**
- Fewer qubits
- Shallower circuits per round
- Feedback between rounds

**Disadvantages:**
- Sequential (slower)
- Error propagation between rounds
- Requires mid-circuit measurement

---

## Worked Examples

### Example 1: Extracting $\phi = 0.75$

Use iterative QPE with $n = 2$ bits.

**Solution:**

$\phi = 0.75 = 0.11$ in binary.

**Round 1 (k=2):** Extract $b_2$
- Apply $H|0\rangle = |+\rangle$
- $CU^1$ gives $(|0\rangle + e^{2\pi i \cdot 0.75}|1\rangle)/\sqrt{2} = (|0\rangle - i|1\rangle)/\sqrt{2}$
- $\theta = 0$, no correction
- Apply $H$: projects to $|1\rangle$ (phase is $\pi/2$)
- Measure: $b_2 = 1$
- Update: $\theta = 0 + 0.5 = 0.5$

**Round 2 (k=1):** Extract $b_1$
- Apply $H|0\rangle$
- $CU^2$ gives phase $e^{2\pi i \cdot 2 \cdot 0.75} = e^{3\pi i} = -1$
- State: $(|0\rangle - |1\rangle)/\sqrt{2}$
- Apply $R_z(-2\pi \cdot 0.5) = R_z(-\pi)$ correction
- This gives phase $e^{-i\pi/2}$ to $|1\rangle$: $(|0\rangle + i|1\rangle)/\sqrt{2}$
- Apply $H$: measure $|1\rangle$
- $b_1 = 1$

**Result:** $\phi = 0.b_1 b_2 = 0.11 = 0.75$ ✓

---

## Practice Problems

### Problem 1
Trace iterative QPE for $\phi = 0.5$ with $n = 3$.

### Problem 2
What happens if a bit is measured incorrectly in round $k$? How does it affect subsequent rounds?

### Problem 3
Design a fault-tolerant version with voting per bit.

---

## Summary

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Correction angle | $\theta_{k+1} = \theta_k/2 + b_k/2$ |
| Power of U | $U^{2^{n-k}}$ in round $k$ |
| Total rounds | $n$ for $n$ bits |

### Key Takeaways

1. **Single ancilla** suffices with iteration
2. **Feedback** from previous measurements guides corrections
3. **Trade-off:** fewer qubits for more rounds
4. **Error propagation** is a concern
5. **NISQ-friendly** due to shallow circuits

---

*Next: Day 608 - Kitaev's Algorithm*
