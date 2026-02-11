# Day 592: Bernstein-Vazirani Algorithm

## Overview

**Day 592** | Week 85, Day 4 | Month 22 | Quantum Algorithms I

Today we study the Bernstein-Vazirani algorithm, which solves the "hidden string" problem. This algorithm demonstrates a different type of quantum speedup: reducing $n$ classical queries to exactly 1 quantum query, while also highlighting the power of the Hadamard transform for extracting global function information.

---

## Learning Objectives

1. Define the hidden string problem precisely
2. Prove the classical $n$ query lower bound
3. Derive the Bernstein-Vazirani algorithm
4. Understand the connection to Deutsch-Jozsa
5. Recognize the practical significance in learning parity functions
6. Implement and verify the algorithm computationally

---

## Core Content

### The Hidden String Problem

**Problem Statement:** Given oracle access to a function $f: \{0,1\}^n \to \{0,1\}$ of the form:

$$f(x) = s \cdot x \pmod{2} = \bigoplus_{i=1}^{n} s_i x_i$$

where $s \in \{0,1\}^n$ is a hidden string, determine $s$ using as few queries as possible.

The function computes the **bitwise inner product** (mod 2) of $x$ with the secret string $s$.

**Example:** For $n = 4$ and $s = 1011$:
- $f(0000) = 1 \cdot 0 \oplus 0 \cdot 0 \oplus 1 \cdot 0 \oplus 1 \cdot 0 = 0$
- $f(1000) = 1 \cdot 1 \oplus 0 \cdot 0 \oplus 1 \cdot 0 \oplus 1 \cdot 0 = 1$
- $f(0100) = 1 \cdot 0 \oplus 0 \cdot 1 \oplus 1 \cdot 0 \oplus 1 \cdot 0 = 0$
- $f(1100) = 1 \cdot 1 \oplus 0 \cdot 1 \oplus 1 \cdot 0 \oplus 1 \cdot 0 = 1$

### Classical Lower Bound

**Theorem:** Any classical algorithm requires exactly $n$ queries to determine $s$.

**Proof:**
Query $e_i = 0\ldots010\ldots0$ (1 in position $i$ only):
$$f(e_i) = s \cdot e_i = s_i$$

This directly reveals bit $s_i$. So $n$ queries suffice.

For the lower bound: each query reveals at most 1 bit of information about $s$ (the output is a single bit). Since $s$ has $n$ bits of entropy, at least $n$ queries are necessary. $\square$

### Quantum Algorithm: One Query!

The Bernstein-Vazirani algorithm uses the **exact same circuit** as Deutsch-Jozsa:

```
|0⟩ ─[H]─────●─────[H]───── Measure → s₁
|0⟩ ─[H]─────●─────[H]───── Measure → s₂
 ⋮           ⋮       ⋮
|0⟩ ─[H]─────●─────[H]───── Measure → sₙ
            U_f
|1⟩ ─[H]───────────────────
```

### State Evolution

**Step 1: Initial State**
$$|\psi_0\rangle = |0\rangle^{\otimes n}|1\rangle$$

**Step 2: After Hadamards**
$$|\psi_1\rangle = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle|-\rangle$$

**Step 3: After Oracle (Phase Kickback)**
$$|\psi_2\rangle = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}(-1)^{f(x)}|x\rangle|-\rangle$$

$$= \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}(-1)^{s \cdot x}|x\rangle|-\rangle$$

**Step 4: After Final Hadamards**

The key insight is recognizing the structure of this state.

Recall the Hadamard transform:
$$H^{\otimes n}|y\rangle = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}(-1)^{x \cdot y}|x\rangle$$

Therefore, the inverse:
$$H^{\otimes n}\left[\frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}(-1)^{s \cdot x}|x\rangle\right] = |s\rangle$$

This is because $H^{\otimes n}$ is its own inverse!

**Final State:**
$$|\psi_3\rangle = |s\rangle|-\rangle$$

### Proof Using Amplitude Formula

Using the Deutsch-Jozsa amplitude formula for $f(x) = s \cdot x$:

$$a_y = \frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{s \cdot x + x \cdot y} = \frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{x \cdot (s \oplus y)}$$

For $y = s$: $s \oplus y = 0$, so:
$$a_s = \frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{0} = \frac{2^n}{2^n} = 1$$

For $y \neq s$: $s \oplus y \neq 0$, so:
$$a_y = \frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{x \cdot (s \oplus y)} = 0$$

(Equal numbers of +1 and -1 terms when the inner product is with a non-zero string.)

$$\boxed{\text{Measurement yields } |s\rangle \text{ with probability } 1}$$

### Why This Works: Fourier Transform Interpretation

The quantum state $\frac{1}{\sqrt{2^n}}\sum_x (-1)^{s \cdot x}|x\rangle$ is the **Fourier transform** of the delta function $\delta_{s}$.

Applying the Hadamard transform (which equals the inverse Fourier transform over $\mathbb{Z}_2^n$) recovers $|s\rangle$.

This is the first instance of the **Quantum Fourier Transform** principle that underlies Shor's algorithm!

### Comparison with Deutsch-Jozsa

| Aspect | Deutsch-Jozsa | Bernstein-Vazirani |
|--------|---------------|-------------------|
| Problem | Constant vs balanced? | Find hidden string |
| Output | 1 bit (yes/no) | n bits (the string s) |
| Classical | $2^{n-1}+1$ queries | n queries |
| Quantum | 1 query | 1 query |
| Speedup type | Exponential | Linear to constant |

Both use the same circuit but extract different information!

---

## Worked Examples

### Example 1: n = 3, s = 101

Find the hidden string for $f(x) = x_1 \oplus x_3$ (s = 101).

**Solution:**

**Truth table verification:**
- $f(000) = 0 \oplus 0 = 0$
- $f(001) = 0 \oplus 1 = 1$
- $f(010) = 0 \oplus 0 = 0$
- $f(011) = 0 \oplus 1 = 1$
- $f(100) = 1 \oplus 0 = 1$
- $f(101) = 1 \oplus 1 = 0$
- $f(110) = 1 \oplus 0 = 1$
- $f(111) = 1 \oplus 1 = 0$

**After oracle:**
$$|\psi_2\rangle = \frac{1}{\sqrt{8}}(|000\rangle - |001\rangle + |010\rangle - |011\rangle - |100\rangle + |101\rangle - |110\rangle + |111\rangle)|-\rangle$$

**Factorization trick:**
Notice the pattern of signs follows $(-1)^{x_1 \oplus x_3}$:
$$= \frac{1}{\sqrt{8}}(|0\rangle - |1\rangle)_1 \otimes (|0\rangle + |1\rangle)_2 \otimes (|0\rangle - |1\rangle)_3$$
$$= |-\rangle|+\rangle|-\rangle$$

**After Hadamards:**
$$H|-\rangle = |1\rangle, \quad H|+\rangle = |0\rangle$$
$$H^{\otimes 3}|-\rangle|+\rangle|-\rangle = |1\rangle|0\rangle|1\rangle = |101\rangle$$

**Result:** Measure $|101\rangle = s$. Success!

### Example 2: n = 4, s = 0110

For $f(x) = x_2 \oplus x_3$.

**Solution:**

The state after the oracle:
$$\frac{1}{4}\sum_x (-1)^{x_2 \oplus x_3}|x\rangle$$

Factor by qubit:
- Qubit 1: $(-1)^0 = 1$ regardless of $x_1$ → $|+\rangle$
- Qubit 2: $(-1)^{x_2}$ → $|-\rangle$
- Qubit 3: $(-1)^{x_3}$ → $|-\rangle$
- Qubit 4: $(-1)^0 = 1$ regardless of $x_4$ → $|+\rangle$

State: $|+\rangle|-\rangle|-\rangle|+\rangle$

After Hadamards: $|0\rangle|1\rangle|1\rangle|0\rangle = |0110\rangle = s$

### Example 3: n = 2, s = 00 (Zero String)

For $f(x) = 0$ (constant zero, since $s = 00$).

**Solution:**

$f(x) = 0 \cdot x_1 \oplus 0 \cdot x_2 = 0$ for all $x$.

After oracle: all phases are $(-1)^0 = +1$:
$$\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle) = |+\rangle|+\rangle$$

After Hadamards: $|0\rangle|0\rangle = |00\rangle = s$

**Note:** This is also a valid Deutsch-Jozsa constant function case!

---

## Practice Problems

### Problem 1: Manual Calculation

For $n = 3$ and $s = 011$, manually trace through the Bernstein-Vazirani algorithm, writing out:
(a) The truth table for $f$
(b) The state after each step
(c) Verify the measurement gives $|011\rangle$

### Problem 2: Classical Strategy

Prove that querying the $n$ standard basis vectors $e_1, e_2, \ldots, e_n$ is the optimal classical strategy and always works.

### Problem 3: Noise Robustness

If the oracle has a 1% error rate (returns wrong bit with probability 0.01), how does this affect:
(a) The classical algorithm?
(b) The quantum algorithm?

### Problem 4: Generalization

Can you modify Bernstein-Vazirani to work for functions $f(x) = s \cdot x \oplus b$ where $b$ is an unknown constant bit? How would you determine both $s$ and $b$?

---

## Computational Lab

```python
"""Day 592: Bernstein-Vazirani Algorithm Implementation"""
import numpy as np
from typing import List, Callable

def hadamard_n(n: int) -> np.ndarray:
    """n-qubit Hadamard transform matrix"""
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    result = H1
    for _ in range(n - 1):
        result = np.kron(result, H1)
    return result

def inner_product_mod2(x: int, y: int) -> int:
    """Compute x · y mod 2 (bitwise AND then XOR)"""
    return bin(x & y).count('1') % 2

def create_bv_oracle(s: int, n: int) -> np.ndarray:
    """
    Create oracle for f(x) = s · x mod 2
    Returns (n+1)-qubit unitary
    """
    dim = 2 ** (n + 1)
    U = np.zeros((dim, dim))

    for x in range(2**n):
        fx = inner_product_mod2(s, x)
        for y in range(2):
            input_idx = (x << 1) | y
            output_y = y ^ fx
            output_idx = (x << 1) | output_y
            U[output_idx, input_idx] = 1

    return U

def bernstein_vazirani(s_hidden: int, n: int, verbose: bool = True) -> int:
    """
    Execute Bernstein-Vazirani algorithm to find hidden string s
    Returns: recovered string s as integer
    """
    dim = 2 ** (n + 1)

    # Initial state |0...01⟩
    state = np.zeros(dim, dtype=complex)
    state[1] = 1

    if verbose:
        print(f"Hidden string s = {s_hidden:0{n}b} (decimal: {s_hidden})")
        print(f"n = {n} qubits")

    # Apply H^⊗(n+1)
    H_full = hadamard_n(n + 1)
    state = H_full @ state

    # Apply oracle
    U_f = create_bv_oracle(s_hidden, n)
    state = U_f @ state

    # Apply H^⊗n ⊗ I
    H_n = hadamard_n(n)
    H_input = np.kron(H_n, np.eye(2))
    state = H_input @ state

    # Measure input register
    # For each possible outcome y, probability = |state[2y]|² + |state[2y+1]|²
    probabilities = np.zeros(2**n)
    for y in range(2**n):
        probabilities[y] = abs(state[2*y])**2 + abs(state[2*y+1])**2

    # Find the outcome with probability ~1
    measured = np.argmax(probabilities)

    if verbose:
        print(f"\nMeasurement probabilities:")
        for y in range(min(2**n, 8)):
            if probabilities[y] > 1e-10:
                print(f"  |{y:0{n}b}⟩: {probabilities[y]:.6f}")
        if 2**n > 8:
            print("  ...")
        print(f"\nMeasured: |{measured:0{n}b}⟩ = s")

    return measured

def classical_bv(oracle: Callable[[int], int], n: int) -> int:
    """
    Classical algorithm for Bernstein-Vazirani
    Requires n queries
    """
    s = 0
    queries = 0
    for i in range(n):
        # Query e_i = 2^(n-1-i) (1 in position i from left)
        e_i = 1 << (n - 1 - i)
        bit = oracle(e_i)
        queries += 1
        if bit:
            s |= e_i
    return s, queries

def test_all_strings(n: int):
    """Test Bernstein-Vazirani on all possible hidden strings"""
    print("=" * 60)
    print(f"TESTING ALL {2**n} HIDDEN STRINGS FOR n = {n}")
    print("=" * 60)

    all_correct = True
    for s in range(2**n):
        result = bernstein_vazirani(s, n, verbose=False)
        if result != s:
            print(f"FAILED: s = {s:0{n}b}, got {result:0{n}b}")
            all_correct = False

    if all_correct:
        print(f"All {2**n} test cases passed!")
    return all_correct

def compare_classical_quantum():
    """Compare classical and quantum approaches"""
    print("\n" + "=" * 60)
    print("CLASSICAL VS QUANTUM COMPARISON")
    print("=" * 60)

    for n in [4, 8, 16]:
        # Random hidden string
        s = np.random.randint(0, 2**n)

        # Classical
        oracle = lambda x, s=s: inner_product_mod2(s, x)
        s_classical, queries = classical_bv(oracle, n)

        # Quantum (always 1 query)
        s_quantum = bernstein_vazirani(s, n, verbose=False)

        print(f"\nn = {n}, s = {s:0{min(n,16)}b}{'...' if n > 16 else ''}")
        print(f"  Classical: {queries} queries, found s = {s_classical:0{min(n,16)}b}")
        print(f"  Quantum:   1 query,   found s = {s_quantum:0{min(n,16)}b}")
        print(f"  Speedup:   {queries}x fewer queries")

def analyze_phase_structure():
    """Analyze the phase structure that enables the algorithm"""
    print("\n" + "=" * 60)
    print("PHASE STRUCTURE ANALYSIS")
    print("=" * 60)

    n = 3
    s = 0b101  # s = 101

    print(f"\nHidden string s = {s:0{n}b}")
    print(f"\nPhases (-1)^(s·x) for each x:")

    phases = []
    for x in range(2**n):
        phase = (-1) ** inner_product_mod2(s, x)
        phases.append(phase)
        print(f"  x = {x:0{n}b}: s·x = {inner_product_mod2(s, x)}, phase = {phase:+d}")

    # Show factorization
    print("\nFactorization into single-qubit states:")
    print("  State = ", end="")
    for i in range(n):
        bit = (s >> (n - 1 - i)) & 1
        if bit:
            print("|−⟩", end="")
        else:
            print("|+⟩", end="")
        if i < n - 1:
            print(" ⊗ ", end="")
    print()
    print(f"  (|−⟩ for s_i=1, |+⟩ for s_i=0)")

def demonstrate_single_query_power():
    """Show how one superposition query learns all bits"""
    print("\n" + "=" * 60)
    print("SINGLE QUERY POWER DEMONSTRATION")
    print("=" * 60)

    n = 4
    s = 0b1011

    print(f"\nHidden string s = {s:0{n}b}")
    print("\nClassical approach (4 queries needed):")
    for i in range(n):
        e_i = 1 << (n - 1 - i)
        result = inner_product_mod2(s, e_i)
        print(f"  Query {i+1}: f({e_i:0{n}b}) = s·e_{i+1} = s_{i+1} = {result}")

    print("\nQuantum approach (1 query):")
    print("  Prepare: |ψ⟩ = (1/√16) Σ_x |x⟩|−⟩")
    print("  Query:   U_f|ψ⟩ = (1/√16) Σ_x (-1)^(s·x) |x⟩|−⟩")
    print("  This single query encodes ALL correlations with s!")
    print("  Apply H^⊗n: interference reveals |s⟩ = |1011⟩")

# Run all demonstrations
print("="*60)
print("BERNSTEIN-VAZIRANI ALGORITHM")
print("="*60)

# Test specific case
print("\n--- Single Test Case ---")
bernstein_vazirani(0b101, 3, verbose=True)

# Test all strings for small n
print()
test_all_strings(3)

# Compare approaches
compare_classical_quantum()

# Analyze structure
analyze_phase_structure()

# Demonstrate power
demonstrate_single_query_power()

# Visualization
print("\n" + "="*60)
print("GENERATING VISUALIZATION")
print("="*60)

import matplotlib.pyplot as plt

def visualize_bv_circuit():
    """Create circuit diagram visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Draw qubit lines
    n = 3
    for i in range(n + 1):
        y = 5 - i
        ax.plot([0.5, 9.5], [y, y], 'k-', linewidth=1)
        if i < n:
            ax.text(0.2, y, f'|0⟩', fontsize=12, va='center')
        else:
            ax.text(0.2, y, f'|1⟩', fontsize=12, va='center')

    # Hadamard gates (initial)
    for i in range(n + 1):
        y = 5 - i
        rect = plt.Rectangle((1.5, y-0.3), 0.6, 0.6, fill=True,
                            facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(1.8, y, 'H', fontsize=10, va='center', ha='center')

    # Oracle box
    rect = plt.Rectangle((3.5, 1.2), 1.5, 4.1, fill=True,
                         facecolor='lightyellow', edgecolor='black')
    ax.add_patch(rect)
    ax.text(4.25, 3.2, '$U_f$', fontsize=14, va='center', ha='center')

    # Hadamard gates (final, only on input qubits)
    for i in range(n):
        y = 5 - i
        rect = plt.Rectangle((6.5, y-0.3), 0.6, 0.6, fill=True,
                            facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(6.8, y, 'H', fontsize=10, va='center', ha='center')

    # Measurement symbols
    for i in range(n):
        y = 5 - i
        ax.plot([8.5, 8.5], [y-0.2, y+0.2], 'k-', linewidth=2)
        ax.plot([8.3, 8.5, 8.7], [y+0.2, y-0.1, y+0.2], 'k-', linewidth=2)
        ax.text(9.2, y, f'$s_{i+1}$', fontsize=12, va='center')

    # Labels
    ax.text(5, 0.5, 'Bernstein-Vazirani Circuit', fontsize=14,
            ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('bv_circuit.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Circuit diagram saved to 'bv_circuit.png'")

visualize_bv_circuit()
```

**Expected Output:**
```
============================================================
BERNSTEIN-VAZIRANI ALGORITHM
============================================================

--- Single Test Case ---
Hidden string s = 101 (decimal: 5)
n = 3 qubits

Measurement probabilities:
  |101⟩: 1.000000

Measured: |101⟩ = s

============================================================
TESTING ALL 8 HIDDEN STRINGS FOR n = 3
============================================================
All 8 test cases passed!

============================================================
CLASSICAL VS QUANTUM COMPARISON
============================================================

n = 4, s = 1011
  Classical: 4 queries, found s = 1011
  Quantum:   1 query,   found s = 1011
  Speedup:   4x fewer queries
...
```

---

## Summary

### Key Formulas

| Expression | Formula |
|------------|---------|
| Hidden string function | $f(x) = s \cdot x = \bigoplus_i s_i x_i$ |
| Oracle action | $U_f\|x\rangle\|-\rangle = (-1)^{s \cdot x}\|x\rangle\|-\rangle$ |
| State after oracle | $\frac{1}{\sqrt{2^n}}\sum_x (-1)^{s \cdot x}\|x\rangle$ |
| Hadamard inversion | $H^{\otimes n}\left[\frac{1}{\sqrt{2^n}}\sum_x (-1)^{s \cdot x}\|x\rangle\right] = \|s\rangle$ |

### Key Takeaways

1. **Linear speedup**: n classical queries reduced to 1 quantum query
2. **Same circuit as Deutsch-Jozsa** but extracts different information
3. **Fourier transform structure**: Hadamard is QFT over $\mathbb{Z}_2^n$
4. **Perfect success**: Algorithm succeeds with probability 1
5. **Practical relevance**: Learning parity functions, error syndrome measurement

---

## Daily Checklist

- [ ] I can define the hidden string problem
- [ ] I understand why n classical queries are necessary
- [ ] I can trace the algorithm and see why it outputs s
- [ ] I see the connection to Deutsch-Jozsa
- [ ] I understand the Fourier transform interpretation
- [ ] I ran the lab and verified the algorithm on multiple strings

---

*Next: Day 593 - Simon's Algorithm Introduction*
