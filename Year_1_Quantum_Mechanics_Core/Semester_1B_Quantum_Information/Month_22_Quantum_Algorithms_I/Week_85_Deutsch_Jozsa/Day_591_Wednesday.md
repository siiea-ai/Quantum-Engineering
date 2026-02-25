# Day 591: Deutsch-Jozsa n-Qubit Generalization

## Overview

**Day 591** | Week 85, Day 3 | Month 22 | Quantum Algorithms I

Today we extend Deutsch's algorithm to n qubits, creating the Deutsch-Jozsa algorithm. This demonstrates an exponential quantum speedup: while classical algorithms require $2^{n-1}+1$ queries in the worst case, the quantum algorithm needs only ONE query.

---

## Learning Objectives

1. Generalize the constant vs. balanced problem to n bits
2. Prove the classical lower bound of $2^{n-1}+1$ queries
3. Derive the complete Deutsch-Jozsa algorithm
4. Prove correctness via Hadamard transform properties
5. Understand the exponential quantum speedup
6. Recognize limitations of the algorithm in practice

---

## Core Content

### The Generalized Problem

**Deutsch-Jozsa Problem:** Given $f: \{0,1\}^n \to \{0,1\}$ promised to be either:
- **Constant**: $f(x) = c$ for all $x \in \{0,1\}^n$ (same output for all $2^n$ inputs)
- **Balanced**: $f(x) = 0$ for exactly $2^{n-1}$ inputs and $f(x) = 1$ for the other $2^{n-1}$

Determine which type $f$ is.

### Classical Query Complexity

**Theorem:** Any deterministic classical algorithm requires at least $2^{n-1}+1$ queries.

**Proof:**
Consider querying $f$ on inputs $x_1, x_2, \ldots, x_k$ where $k \leq 2^{n-1}$.

If all queries return the same value (say 0), we cannot distinguish:
- $f$ is constant (all outputs are 0)
- $f$ is balanced with $f(x_i) = 0$ for our queried inputs and $f(x) = 1$ for the remaining $2^n - k \geq 2^{n-1}$ inputs

Only after seeing more than half the inputs with the same value can we conclude $f$ is constant.

Therefore, $k \geq 2^{n-1} + 1$ queries are necessary. $\square$

### Quantum Algorithm: Circuit

```
|0⟩ ─[H]─────●─────[H]───── Measure
|0⟩ ─[H]─────●─────[H]───── Measure
 ⋮           ⋮       ⋮
|0⟩ ─[H]─────●─────[H]───── Measure
            U_f
|1⟩ ─[H]───────────────────
```

**Circuit Structure:**
1. n input qubits initialized to $|0\rangle$, one ancilla to $|1\rangle$
2. Apply $H^{\otimes(n+1)}$ (Hadamard on all qubits)
3. Apply oracle $U_f$
4. Apply $H^{\otimes n}$ to input register only
5. Measure input register in computational basis

### State Evolution Analysis

**Step 1: Initial State**
$$|\psi_0\rangle = |0\rangle^{\otimes n}|1\rangle$$

**Step 2: After Hadamards**
$$|\psi_1\rangle = H^{\otimes n}|0\rangle^{\otimes n} \otimes H|1\rangle$$

Using $H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle$:

$$|\psi_1\rangle = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle \otimes |-\rangle$$

**Step 3: After Oracle**

Using phase kickback:
$$|\psi_2\rangle = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}(-1)^{f(x)}|x\rangle \otimes |-\rangle$$

**Step 4: After Final Hadamards**

Apply $H^{\otimes n}$ to the input register. Using the Hadamard transform:

$$H^{\otimes n}|x\rangle = \frac{1}{\sqrt{2^n}}\sum_{y=0}^{2^n-1}(-1)^{x \cdot y}|y\rangle$$

where $x \cdot y = \bigoplus_{i=1}^n x_i y_i$ is the bitwise inner product modulo 2.

Therefore:
$$|\psi_3\rangle = \frac{1}{2^n}\sum_{x=0}^{2^n-1}\sum_{y=0}^{2^n-1}(-1)^{f(x)+x \cdot y}|y\rangle \otimes |-\rangle$$

$$= \sum_{y=0}^{2^n-1}\left[\frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{f(x)+x \cdot y}\right]|y\rangle \otimes |-\rangle$$

The amplitude of $|y\rangle$ is:
$$\boxed{a_y = \frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{f(x)+x \cdot y}}$$

### Proof of Correctness

**Claim:** Measuring $|0\rangle^{\otimes n}$ with probability 1 iff $f$ is constant.

**For $y = 0$ (all zeros):**
$$a_0 = \frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{f(x)}$$

**Case 1: $f$ is constant**

If $f(x) = c$ for all $x$:
$$a_0 = \frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^c = \frac{(-1)^c \cdot 2^n}{2^n} = \pm 1$$

So $|a_0|^2 = 1$, meaning we measure $|0\rangle^{\otimes n}$ with certainty.

**Case 2: $f$ is balanced**

Half the inputs give $f(x) = 0$, half give $f(x) = 1$:
$$a_0 = \frac{1}{2^n}\left[\sum_{f(x)=0}(+1) + \sum_{f(x)=1}(-1)\right] = \frac{1}{2^n}[2^{n-1} - 2^{n-1}] = 0$$

So $|a_0|^2 = 0$, meaning we NEVER measure $|0\rangle^{\otimes n}$.

**Conclusion:**
$$\boxed{\text{Measure } |0\rangle^{\otimes n} \Leftrightarrow f \text{ is constant}}$$

### Exponential Speedup

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| Queries (worst case) | $2^{n-1}+1$ | 1 |
| Queries (best case) | 2 | 1 |
| Speedup | - | Exponential |

For $n = 100$: Classical needs up to $2^{99} + 1 \approx 10^{29}$ queries; quantum needs 1.

### Limitations

1. **Promise Problem:** The algorithm only works with the promise that $f$ is constant OR balanced. For arbitrary functions, behavior is undefined.

2. **No Practical Application:** There's no known real-world problem that maps to constant vs. balanced with an oracle.

3. **Deterministic Classical Sampling:** With high probability, just 2 random queries to a balanced function give different values.

4. **Historical Importance:** Despite limitations, Deutsch-Jozsa was the first exponential quantum speedup, inspiring the development of more practical algorithms.

---

## Worked Examples

### Example 1: n = 2, Constant Function

For $f(x) = 0$ for all $x \in \{00, 01, 10, 11\}$.

**Solution:**

**After Hadamards:**
$$|\psi_1\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)|-\rangle$$

**After Oracle:**
All phases are $(-1)^0 = 1$:
$$|\psi_2\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)|-\rangle$$

**After Final Hadamards:**
$$H^{\otimes 2}\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle) = H^{\otimes 2}H^{\otimes 2}|00\rangle = |00\rangle$$

**Measurement:** $|00\rangle$ with probability 1. Constant confirmed!

### Example 2: n = 2, Balanced Function

For $f(00) = f(11) = 0$ and $f(01) = f(10) = 1$.

**Solution:**

**After Oracle:**
$$|\psi_2\rangle = \frac{1}{2}(|00\rangle - |01\rangle - |10\rangle + |11\rangle)|-\rangle$$

**After Final Hadamards:**

Compute $H^{\otimes 2}|\psi_{input}\rangle$ where $|\psi_{input}\rangle = \frac{1}{2}(|00\rangle - |01\rangle - |10\rangle + |11\rangle)$.

Note: $|\psi_{input}\rangle = \frac{1}{2}(|0\rangle - |1\rangle)(|0\rangle - |1\rangle) = |-\rangle|-\rangle$

$$H^{\otimes 2}|-\rangle|-\rangle = H|-\rangle \otimes H|-\rangle = |1\rangle|1\rangle = |11\rangle$$

**Measurement:** $|11\rangle$ with probability 1. NOT $|00\rangle$, so balanced!

### Example 3: n = 3, Parity Function

Let $f(x) = x_1 \oplus x_2 \oplus x_3$ (parity of input bits). Is this balanced?

**Solution:**

First, verify it's balanced: parity is 0 for half the inputs (even number of 1s) and 1 for the other half (odd number of 1s). Yes, balanced!

Using the algorithm:
- $a_0 = \frac{1}{8}\sum_{x}(-1)^{x_1 \oplus x_2 \oplus x_3} = \frac{1}{8}(4 \cdot 1 + 4 \cdot (-1)) = 0$

So measuring $|000\rangle$ has probability 0. What DO we measure?

For parity function, the state after the oracle is:
$$\frac{1}{\sqrt{8}}\sum_x (-1)^{x_1 \oplus x_2 \oplus x_3}|x\rangle$$

This equals:
$$\frac{1}{\sqrt{8}}(|0\rangle - |1\rangle)(|0\rangle - |1\rangle)(|0\rangle - |1\rangle) = |-\rangle^{\otimes 3}$$

After Hadamards: $H^{\otimes 3}|-\rangle^{\otimes 3} = |111\rangle$

**Measurement:** $|111\rangle$ with certainty!

---

## Practice Problems

### Problem 1: Amplitude Calculation

For $n = 2$ and $f(00) = f(01) = 0$, $f(10) = f(11) = 1$, calculate all amplitudes $a_y$ and verify $\sum_y |a_y|^2 = 1$.

### Problem 2: Non-Promise Function

Consider $f: \{0,1\}^2 \to \{0,1\}$ with $f(00) = f(01) = f(10) = 0$ and $f(11) = 1$. This is neither constant nor balanced. What does Deutsch-Jozsa output?

### Problem 3: Query Complexity Proof

Prove that randomized classical algorithms still need $\Omega(1)$ queries but can solve the problem with $O(1)$ queries with high probability. What does this say about the "real" advantage?

### Problem 4: Circuit Depth

Calculate the circuit depth (number of time steps) of the Deutsch-Jozsa algorithm assuming:
- Hadamard gates can be applied in parallel
- The oracle has depth $d$

---

## Computational Lab

```python
"""Day 591: Deutsch-Jozsa Algorithm Implementation"""
import numpy as np
from itertools import product

def hadamard_n(n):
    """Construct n-qubit Hadamard transform"""
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H_n = H1
    for _ in range(n - 1):
        H_n = np.kron(H_n, H1)
    return H_n

def build_oracle(f, n):
    """
    Build oracle U_f for function f: {0,1}^n -> {0,1}
    Total space: n input qubits + 1 ancilla = n+1 qubits
    """
    dim = 2 ** (n + 1)
    U = np.zeros((dim, dim))

    for x in range(2**n):
        for y in range(2):
            input_idx = (x << 1) | y  # x in high bits, y in low bit
            output_y = y ^ f(x)
            output_idx = (x << 1) | output_y
            U[output_idx, input_idx] = 1

    return U

def deutsch_jozsa(f, n, verbose=True):
    """
    Execute Deutsch-Jozsa algorithm
    f: function from n-bit strings to {0,1}
    Returns: 'constant' or 'balanced'
    """
    dim_input = 2 ** n
    dim_total = 2 ** (n + 1)

    # Initial state: |0...0⟩|1⟩
    state = np.zeros(dim_total, dtype=complex)
    state[1] = 1  # |0...01⟩ (ancilla = 1)

    if verbose:
        print(f"n = {n} qubits")
        print(f"Initial state: |{'0'*n}1⟩")

    # Apply H^⊗(n+1)
    H_full = hadamard_n(n + 1)
    state = H_full @ state

    if verbose:
        print(f"After H^⊗{n+1}: uniform superposition")

    # Apply oracle
    U_f = build_oracle(f, n)
    state = U_f @ state

    if verbose:
        print("After oracle: phases encoded")

    # Apply H^⊗n ⊗ I to input register
    H_n = hadamard_n(n)
    I_1 = np.eye(2)
    H_input = np.kron(H_n, I_1)
    state = H_input @ state

    # Measure: probability of |0...0⟩ in input register
    # This corresponds to indices 0 (|0...00⟩) and 1 (|0...01⟩)
    prob_all_zeros = abs(state[0])**2 + abs(state[1])**2

    if verbose:
        print(f"P(measure |{'0'*n}⟩) = {prob_all_zeros:.6f}")

    # Threshold at 0.5 (should be exactly 0 or 1)
    result = 'constant' if prob_all_zeros > 0.5 else 'balanced'

    if verbose:
        print(f"Result: {result}")

    return result

def create_constant_function(n, value):
    """Create constant function returning 'value'"""
    return lambda x: value

def create_balanced_function(n, method='parity'):
    """Create balanced function using different methods"""
    if method == 'parity':
        return lambda x: bin(x).count('1') % 2
    elif method == 'msb':
        return lambda x: (x >> (n-1)) & 1
    elif method == 'random':
        # Create a random balanced function
        half = 2 ** (n - 1)
        ones = set(np.random.choice(2**n, half, replace=False))
        return lambda x: 1 if x in ones else 0
    else:
        raise ValueError(f"Unknown method: {method}")

def test_deutsch_jozsa():
    """Test Deutsch-Jozsa on various functions"""
    print("=" * 60)
    print("DEUTSCH-JOZSA ALGORITHM TESTS")
    print("=" * 60)

    # Test for different n
    for n in [2, 3, 4]:
        print(f"\n{'='*40}")
        print(f"Testing n = {n}")
        print(f"{'='*40}")

        # Constant functions
        for val in [0, 1]:
            f = create_constant_function(n, val)
            print(f"\n--- Constant f(x) = {val} ---")
            result = deutsch_jozsa(f, n)
            assert result == 'constant', "FAILED!"
            print("✓ Correct")

        # Balanced functions
        for method in ['parity', 'msb']:
            f = create_balanced_function(n, method)
            print(f"\n--- Balanced ({method}) ---")
            result = deutsch_jozsa(f, n)
            assert result == 'balanced', "FAILED!"
            print("✓ Correct")

def analyze_amplitudes():
    """Detailed analysis of final state amplitudes"""
    print("\n" + "=" * 60)
    print("AMPLITUDE ANALYSIS")
    print("=" * 60)

    n = 3

    # Parity function
    f_parity = lambda x: bin(x).count('1') % 2

    print(f"\nFunction: parity (XOR of all bits)")
    print("Truth table:")
    for x in range(2**n):
        print(f"  f({x:03b}) = {f_parity(x)}")

    # Manual amplitude calculation
    print("\nAmplitude calculation:")
    for y in range(2**n):
        amplitude = 0
        for x in range(2**n):
            # x · y = bitwise AND then XOR all bits
            dot_product = bin(x & y).count('1') % 2
            phase = (-1) ** (f_parity(x) + dot_product)
            amplitude += phase
        amplitude /= 2**n
        print(f"  a_{y:03b} = {amplitude:+.4f}")

def compare_complexity():
    """Compare classical vs quantum query complexity"""
    print("\n" + "=" * 60)
    print("QUERY COMPLEXITY COMPARISON")
    print("=" * 60)

    print("\n| n | Classical (worst) | Quantum |")
    print("|---|-------------------|---------|")
    for n in range(1, 11):
        classical = 2**(n-1) + 1
        quantum = 1
        print(f"| {n:2d} | {classical:17d} | {quantum:7d} |")

    print("\nFor n=50:")
    print(f"  Classical: 2^49 + 1 ≈ 5.6 × 10^14 queries")
    print(f"  Quantum: 1 query")
    print(f"  Speedup: ~10^14 times faster!")

def visualize_interference():
    """Visualize the interference pattern for Deutsch-Jozsa"""
    import matplotlib.pyplot as plt

    n = 3
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Constant function
    f_const = lambda x: 0
    amplitudes_const = []
    for y in range(2**n):
        amp = sum((-1)**(f_const(x) + bin(x & y).count('1') % 2)
                  for x in range(2**n)) / 2**n
        amplitudes_const.append(amp)

    # Balanced function (parity)
    f_bal = lambda x: bin(x).count('1') % 2
    amplitudes_bal = []
    for y in range(2**n):
        amp = sum((-1)**(f_bal(x) + bin(x & y).count('1') % 2)
                  for x in range(2**n)) / 2**n
        amplitudes_bal.append(amp)

    x_labels = [f'{i:0{n}b}' for i in range(2**n)]

    axes[0].bar(range(2**n), amplitudes_const, color='blue', alpha=0.7)
    axes[0].set_xticks(range(2**n))
    axes[0].set_xticklabels(x_labels, rotation=45)
    axes[0].set_xlabel('Measurement outcome |y⟩')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Constant Function: f(x) = 0')
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    axes[1].bar(range(2**n), amplitudes_bal, color='red', alpha=0.7)
    axes[1].set_xticks(range(2**n))
    axes[1].set_xticklabels(x_labels, rotation=45)
    axes[1].set_xlabel('Measurement outcome |y⟩')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Balanced Function: f(x) = parity(x)')
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.suptitle('Deutsch-Jozsa Interference Pattern (n=3)')
    plt.tight_layout()
    plt.savefig('deutsch_jozsa_interference.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nInterference visualization saved to 'deutsch_jozsa_interference.png'")
    print("Constant: amplitude concentrated at |000⟩")
    print("Balanced: amplitude zero at |000⟩, spread elsewhere")

# Run all analyses
test_deutsch_jozsa()
analyze_amplitudes()
compare_complexity()
visualize_interference()
```

**Expected Output:**
```
============================================================
DEUTSCH-JOZSA ALGORITHM TESTS
============================================================

========================================
Testing n = 2
========================================

--- Constant f(x) = 0 ---
n = 2 qubits
Initial state: |001⟩
After H^⊗3: uniform superposition
After oracle: phases encoded
P(measure |00⟩) = 1.000000
Result: constant
✓ Correct

--- Balanced (parity) ---
n = 2 qubits
Initial state: |001⟩
After H^⊗3: uniform superposition
After oracle: phases encoded
P(measure |00⟩) = 0.000000
Result: balanced
✓ Correct
...
```

---

## Summary

### Key Formulas

| Expression | Formula |
|------------|---------|
| Final amplitude | $a_y = \frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{f(x)+x \cdot y}$ |
| Constant: $a_0$ | $\pm 1$ |
| Balanced: $a_0$ | $0$ |
| Classical queries | $2^{n-1} + 1$ (worst case) |
| Quantum queries | $1$ |

### Key Takeaways

1. **Exponential speedup**: From $O(2^n)$ to $O(1)$ queries
2. **Hadamard transform**: Converts phase differences to amplitude differences
3. **Interference**: Constructive at $\|0\rangle^{\otimes n}$ for constant, destructive for balanced
4. **Promise is essential**: Without the promise, no speedup is possible
5. **Historical significance**: First exponential quantum speedup, template for future algorithms

---

## Daily Checklist

- [ ] I can state the Deutsch-Jozsa problem and promise
- [ ] I understand why classical algorithms need $2^{n-1}+1$ queries
- [ ] I can derive the amplitude formula for any output state
- [ ] I understand how interference distinguishes constant from balanced
- [ ] I recognize the limitations of the algorithm
- [ ] I ran the lab and verified the exponential speedup

---

*Next: Day 592 - Bernstein-Vazirani Algorithm*
