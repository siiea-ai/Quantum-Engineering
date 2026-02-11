# Day 590: Deutsch's Algorithm

## Overview

**Day 590** | Week 85, Day 2 | Month 22 | Quantum Algorithms I

Today we study Deutsch's algorithm - the first quantum algorithm to demonstrate a computational advantage over classical computation. Though the problem is simple (distinguishing two types of single-bit functions), it introduces the key techniques used in all subsequent quantum algorithms.

---

## Learning Objectives

1. State the Deutsch problem precisely
2. Derive the complete quantum circuit for Deutsch's algorithm
3. Trace the quantum state through each step of the algorithm
4. Understand why one query suffices quantumly
5. Identify the role of interference in the algorithm
6. Connect to the broader theme of quantum parallelism

---

## Core Content

### The Deutsch Problem

**Problem Statement:** Given a function $f: \{0,1\} \to \{0,1\}$ promised to be either:
- **Constant**: $f(0) = f(1)$ (both outputs equal)
- **Balanced**: $f(0) \neq f(1)$ (outputs different)

Determine which type $f$ is using as few oracle queries as possible.

**The Four Possible Functions:**

| Function | $f(0)$ | $f(1)$ | Type |
|----------|--------|--------|------|
| $f_0$ | 0 | 0 | Constant |
| $f_1$ | 1 | 1 | Constant |
| $f_2$ | 0 | 1 | Balanced |
| $f_3$ | 1 | 0 | Balanced |

**Classical Requirement:** Must query $f(0)$ AND $f(1)$ to compare them. Minimum 2 queries.

**Quantum Result:** 1 query suffices!

### The Key Insight

We want to compute:
$$f(0) \oplus f(1) = \begin{cases} 0 & \text{if constant} \\ 1 & \text{if balanced} \end{cases}$$

This is a **global property** - it depends on both $f(0)$ and $f(1)$ but in a specific combination.

Quantum mechanics allows us to evaluate this combination directly through interference.

### Deutsch's Algorithm - The Circuit

```
|0⟩ ─────[H]─────●─────[H]───── Measure
                 │
|1⟩ ─────[H]─────⊕─────────────
                U_f
```

**Steps:**
1. Prepare $|0\rangle|1\rangle$
2. Apply Hadamard to both qubits
3. Apply the oracle $U_f$
4. Apply Hadamard to the first qubit
5. Measure the first qubit

### Detailed State Evolution

**Step 1: Initial State**
$$|\psi_0\rangle = |0\rangle|1\rangle$$

**Step 2: After Hadamards**
$$|\psi_1\rangle = H|0\rangle \otimes H|1\rangle = |+\rangle|-\rangle$$

$$= \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

$$= \frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle - |1\rangle)$$

**Step 3: After Oracle (Using Phase Kickback)**

Recall: $U_f|x\rangle|-\rangle = (-1)^{f(x)}|x\rangle|-\rangle$

$$|\psi_2\rangle = U_f|\psi_1\rangle = \frac{1}{\sqrt{2}}((-1)^{f(0)}|0\rangle + (-1)^{f(1)}|1\rangle)|-\rangle$$

$$= \frac{(-1)^{f(0)}}{\sqrt{2}}(|0\rangle + (-1)^{f(0) \oplus f(1)}|1\rangle)|-\rangle$$

Now consider the two cases:

**Case 1: Constant ($f(0) = f(1)$, so $f(0) \oplus f(1) = 0$):**
$$|\psi_2\rangle = \frac{(-1)^{f(0)}}{\sqrt{2}}(|0\rangle + |1\rangle)|-\rangle = \pm|+\rangle|-\rangle$$

**Case 2: Balanced ($f(0) \neq f(1)$, so $f(0) \oplus f(1) = 1$):**
$$|\psi_2\rangle = \frac{(-1)^{f(0)}}{\sqrt{2}}(|0\rangle - |1\rangle)|-\rangle = \pm|-\rangle|-\rangle$$

**Step 4: After Final Hadamard on First Qubit**

Using $H|+\rangle = |0\rangle$ and $H|-\rangle = |1\rangle$:

**Constant:** $|\psi_3\rangle = \pm|0\rangle|-\rangle$

**Balanced:** $|\psi_3\rangle = \pm|1\rangle|-\rangle$

**Step 5: Measurement**

$$\boxed{\text{Measure } |0\rangle \Rightarrow f \text{ is constant}}$$
$$\boxed{\text{Measure } |1\rangle \Rightarrow f \text{ is balanced}}$$

### Why It Works: Interference

The algorithm exploits quantum interference:

1. **Superposition**: After the first Hadamard, the query qubit is in $|+\rangle$, allowing simultaneous evaluation of $f(0)$ and $f(1)$

2. **Phase Encoding**: The oracle encodes $f(0)$ and $f(1)$ as phases: $(-1)^{f(0)}$ and $(-1)^{f(1)}$

3. **Interference**: The final Hadamard causes:
   - **Constructive interference** at $|0\rangle$ when phases are equal (constant)
   - **Destructive interference** at $|0\rangle$ when phases differ (balanced)

### The Role of the Ancilla

The second qubit (ancilla) prepared in $|-\rangle$ is crucial:
- It enables the phase kickback trick
- It remains unchanged throughout (can be reused)
- Without it, we couldn't convert function values to phases

### Mathematical Summary

$$\boxed{|0\rangle|1\rangle \xrightarrow{H^{\otimes 2}} |+\rangle|-\rangle \xrightarrow{U_f} \pm|f(0) \oplus f(1)\rangle|-\rangle \xrightarrow{H \otimes I} \pm|f(0) \oplus f(1)\rangle|-\rangle}$$

Wait, let me be more precise:

$$|+\rangle \xrightarrow{U_f \text{ (phase)}} \frac{1}{\sqrt{2}}((-1)^{f(0)}|0\rangle + (-1)^{f(1)}|1\rangle)$$

$$= (-1)^{f(0)} \cdot \frac{1}{\sqrt{2}}(|0\rangle + (-1)^{f(0)\oplus f(1)}|1\rangle)$$

$$\xrightarrow{H} (-1)^{f(0)}|f(0) \oplus f(1)\rangle$$

---

## Worked Examples

### Example 1: Constant Function $f(x) = 0$

Trace through Deutsch's algorithm for $f(x) = 0$ for all $x$.

**Solution:**

**Initial:** $|01\rangle$

**After Hadamards:**
$$\frac{1}{2}(|0\rangle + |1\rangle)(|0\rangle - |1\rangle) = \frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$$

**After Oracle:** Since $f(0) = f(1) = 0$:
$$U_f|xy\rangle = |x\rangle|y \oplus 0\rangle = |xy\rangle$$
State unchanged.

**After Final Hadamard on first qubit:**
$$H \otimes I: \frac{1}{2}((|0\rangle + |1\rangle)(|0\rangle - |1\rangle)) = |0\rangle|-\rangle$$

**Measurement:** First qubit gives $|0\rangle$ with probability 1.

**Conclusion:** $f$ is constant. Correct!

### Example 2: Balanced Function $f(x) = x$

Trace through for the identity function.

**Solution:**

**Initial:** $|01\rangle$

**After Hadamards:** Same as above.

**After Oracle:**
- $U_f|0\rangle|y\rangle = |0\rangle|y \oplus 0\rangle = |0\rangle|y\rangle$
- $U_f|1\rangle|y\rangle = |1\rangle|y \oplus 1\rangle$

$$|\psi_2\rangle = \frac{1}{2}(|00\rangle - |01\rangle + |11\rangle - |10\rangle)$$

$$= \frac{1}{2}(|0\rangle(|0\rangle - |1\rangle) + |1\rangle(|1\rangle - |0\rangle))$$

$$= \frac{1}{2}(|0\rangle - |1\rangle)(|0\rangle - |1\rangle) = |-\rangle|-\rangle$$

**After Final Hadamard:**
$$H|-\rangle = |1\rangle$$

State becomes $|1\rangle|-\rangle$.

**Measurement:** First qubit gives $|1\rangle$ with probability 1.

**Conclusion:** $f$ is balanced. Correct!

### Example 3: Balanced Function $f(x) = 1 - x$

For $f(0) = 1$, $f(1) = 0$.

**Solution:**

Using phase kickback directly:
$$|+\rangle \xrightarrow{U_f} \frac{1}{\sqrt{2}}((-1)^1|0\rangle + (-1)^0|1\rangle) = \frac{1}{\sqrt{2}}(-|0\rangle + |1\rangle) = -|-\rangle$$

$$\xrightarrow{H} -|1\rangle$$

**Measurement:** $|1\rangle$ - balanced. Correct!

---

## Practice Problems

### Problem 1: State Verification

Verify that for $f(x) = 1$ (constant), Deutsch's algorithm outputs $|0\rangle$.

### Problem 2: Global Phase

In Example 3, we got $-|1\rangle$ instead of $|1\rangle$. Explain why this global phase doesn't affect the measurement outcome.

### Problem 3: Error Analysis

Suppose the initial state has a small error: $|0\rangle + \epsilon|1\rangle$ (unnormalized). How does this propagate through Deutsch's algorithm?

### Problem 4: Without Phase Kickback

Design an alternative version of Deutsch's algorithm that uses the standard oracle $U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$ without the phase kickback trick. What measurement strategy would you use?

*Hint: Consider measuring in the Bell basis.*

---

## Computational Lab

```python
"""Day 590: Deutsch's Algorithm Implementation"""
import numpy as np

def hadamard():
    """Single-qubit Hadamard gate"""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def identity():
    """2x2 identity"""
    return np.eye(2)

def tensor(A, B):
    """Kronecker/tensor product"""
    return np.kron(A, B)

def oracle_deutsch(f):
    """
    Build 4x4 oracle matrix for f: {0,1} -> {0,1}
    U_f|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩
    """
    U = np.zeros((4, 4))
    for x in [0, 1]:
        for y in [0, 1]:
            input_idx = 2*x + y
            output_idx = 2*x + (y ^ f(x))
            U[output_idx, input_idx] = 1
    return U

def deutsch_algorithm(f, verbose=True):
    """
    Execute Deutsch's algorithm for function f: {0,1} -> {0,1}
    Returns: 'constant' or 'balanced'
    """
    # Initial state |0⟩|1⟩
    state = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
    if verbose:
        print(f"Function: f(0)={f(0)}, f(1)={f(1)}")
        print(f"Initial state |01⟩: {state}")

    # Apply H ⊗ H
    HH = tensor(hadamard(), hadamard())
    state = HH @ state
    if verbose:
        print(f"After H⊗H: {np.round(state, 4)}")

    # Apply oracle
    U_f = oracle_deutsch(f)
    state = U_f @ state
    if verbose:
        print(f"After U_f: {np.round(state, 4)}")

    # Apply H ⊗ I
    HI = tensor(hadamard(), identity())
    state = HI @ state
    if verbose:
        print(f"After H⊗I: {np.round(state, 4)}")

    # Measure first qubit
    # Probability of |0⟩ in first qubit = |state[0]|² + |state[1]|²
    prob_0 = abs(state[0])**2 + abs(state[1])**2
    prob_1 = abs(state[2])**2 + abs(state[3])**2

    if verbose:
        print(f"P(first qubit = 0) = {prob_0:.4f}")
        print(f"P(first qubit = 1) = {prob_1:.4f}")

    # Determine result
    if prob_0 > 0.5:
        return 'constant'
    else:
        return 'balanced'

def run_all_functions():
    """Test Deutsch's algorithm on all four single-bit functions"""
    functions = [
        (lambda x: 0, "f(x) = 0", "constant"),
        (lambda x: 1, "f(x) = 1", "constant"),
        (lambda x: x, "f(x) = x", "balanced"),
        (lambda x: 1-x, "f(x) = 1-x", "balanced"),
    ]

    print("="*60)
    print("DEUTSCH'S ALGORITHM - ALL CASES")
    print("="*60)

    for f, name, expected in functions:
        print(f"\n--- {name} (Expected: {expected}) ---")
        result = deutsch_algorithm(f, verbose=True)
        print(f"Result: {result}")
        assert result == expected, f"FAILED for {name}!"
        print("✓ Correct!")

def analyze_interference():
    """Visualize the interference pattern in Deutsch's algorithm"""
    print("\n" + "="*60)
    print("INTERFERENCE ANALYSIS")
    print("="*60)

    # After oracle, before final Hadamard
    # State of first qubit: ((-1)^f(0)|0⟩ + (-1)^f(1)|1⟩)/√2

    cases = [
        (0, 0, "constant (both 0)"),
        (1, 1, "constant (both 1)"),
        (0, 1, "balanced (0,1)"),
        (1, 0, "balanced (1,0)"),
    ]

    for f0, f1, description in cases:
        # First qubit state after oracle
        coeff_0 = (-1)**f0 / np.sqrt(2)
        coeff_1 = (-1)**f1 / np.sqrt(2)

        state_before_H = np.array([coeff_0, coeff_1])

        # Apply Hadamard
        H = hadamard()
        state_after_H = H @ state_before_H

        print(f"\n{description}:")
        print(f"  Before final H: {coeff_0:.3f}|0⟩ + {coeff_1:.3f}|1⟩")
        print(f"  After final H:  {state_after_H[0]:.3f}|0⟩ + {state_after_H[1]:.3f}|1⟩")

        # Explain interference
        if f0 == f1:
            print(f"  → Phases equal: constructive interference at |0⟩")
        else:
            print(f"  → Phases opposite: destructive interference at |0⟩")

def quantum_vs_classical():
    """Compare quantum and classical query counts"""
    print("\n" + "="*60)
    print("QUANTUM VS CLASSICAL COMPARISON")
    print("="*60)

    # Simulate classical algorithm
    def classical_deutsch(f):
        """Classical algorithm requires 2 queries"""
        query1 = f(0)  # Query 1
        query2 = f(1)  # Query 2
        return 'constant' if query1 == query2 else 'balanced'

    functions = [
        lambda x: 0,
        lambda x: 1,
        lambda x: x,
        lambda x: 1-x,
    ]

    print("\n| Function | Classical Queries | Quantum Queries |")
    print("|----------|-------------------|-----------------|")
    for i, f in enumerate(functions):
        print(f"| f_{i}       | 2                 | 1               |")

    print("\nQuantum advantage: 2x reduction in queries")
    print("(This is not exponential, but it's the first demonstration!)")

# Run all demonstrations
run_all_functions()
analyze_interference()
quantum_vs_classical()

# Bonus: Visualize the state evolution
print("\n" + "="*60)
print("STATE EVOLUTION VISUALIZATION")
print("="*60)

def visualize_bloch_projection(f):
    """Show first qubit state on Bloch sphere projection"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    states = []
    labels = ['Initial |0⟩', 'After H: |+⟩', f'After U_f', 'After H']

    # Initial
    states.append(np.array([1, 0]))

    # After H
    states.append(np.array([1, 1]) / np.sqrt(2))

    # After oracle (phase kickback)
    states.append(np.array([(-1)**f(0), (-1)**f(1)]) / np.sqrt(2))

    # After final H
    H = hadamard()
    states.append(H @ states[2])

    for ax, state, label in zip(axes, states, labels):
        # Plot on unit circle (X-Z plane of Bloch sphere)
        theta = 2 * np.arccos(np.abs(state[0]))  # Angle from |0⟩
        if np.abs(state[1]) > 1e-10:
            phase = np.angle(state[1]) - np.angle(state[0]) if np.abs(state[0]) > 1e-10 else np.angle(state[1])
        else:
            phase = 0

        # X-Z projection
        x = np.sin(theta) * np.cos(phase)
        z = np.cos(theta)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        circle = plt.Circle((0, 0), 1, fill=False, color='gray')
        ax.add_patch(circle)
        ax.arrow(0, 0, x*0.9, z*0.9, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='lightgray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(label)
        ax.set_aspect('equal')
        ax.text(0.05, -1.3, f'{state[0]:.2f}|0⟩ + {state[1]:.2f}|1⟩', fontsize=8)

    plt.suptitle(f'Deutsch Algorithm State Evolution: f(0)={f(0)}, f(1)={f(1)}')
    plt.tight_layout()
    plt.savefig('deutsch_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization for f(0)={f(0)}, f(1)={f(1)}")

# Generate visualizations
for f in [lambda x: 0, lambda x: x]:
    visualize_bloch_projection(f)

print("\nVisualizations saved to deutsch_evolution.png")
```

**Expected Output:**
```
============================================================
DEUTSCH'S ALGORITHM - ALL CASES
============================================================

--- f(x) = 0 (Expected: constant) ---
Function: f(0)=0, f(1)=0
Initial state |01⟩: [0 1 0 0]
After H⊗H: [ 0.5 -0.5  0.5 -0.5]
After U_f: [ 0.5 -0.5  0.5 -0.5]
After H⊗I: [ 0.7071 -0.7071  0.      0.    ]
P(first qubit = 0) = 1.0000
P(first qubit = 1) = 0.0000
Result: constant
✓ Correct!
...
```

---

## Summary

### Key Formulas

| Step | State |
|------|-------|
| Initial | $\|0\rangle\|1\rangle$ |
| After $H^{\otimes 2}$ | $\|+\rangle\|-\rangle$ |
| After $U_f$ | $\frac{(-1)^{f(0)}}{\sqrt{2}}(\|0\rangle + (-1)^{f(0)\oplus f(1)}\|1\rangle)\|-\rangle$ |
| After $H \otimes I$ | $(-1)^{f(0)}\|f(0) \oplus f(1)\rangle\|-\rangle$ |

### Key Takeaways

1. **Deutsch's algorithm** solves the constant vs. balanced problem with 1 query (vs. 2 classical)
2. **Phase kickback** encodes function values as phases on the input register
3. **Interference** at the final Hadamard distinguishes constant from balanced
4. **Global phase** $(-1)^{f(0)}$ doesn't affect measurement outcomes
5. **This is the template** for more powerful algorithms (Deutsch-Jozsa, Simon, Shor)

---

## Daily Checklist

- [ ] I can state the Deutsch problem precisely
- [ ] I can trace the quantum state through each step
- [ ] I understand why one query suffices
- [ ] I can explain the role of interference
- [ ] I understand phase kickback
- [ ] I ran the simulation and verified all four function cases

---

*Next: Day 591 - Deutsch-Jozsa n-Qubit Generalization*
