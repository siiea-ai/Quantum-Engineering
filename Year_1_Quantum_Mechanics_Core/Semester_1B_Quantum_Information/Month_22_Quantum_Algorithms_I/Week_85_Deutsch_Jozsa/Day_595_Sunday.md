# Day 595: Week 85 Review - Deutsch-Jozsa Family

## Overview

**Day 595** | Week 85, Day 7 | Month 22 | Quantum Algorithms I

Today we consolidate our understanding of the Deutsch-Jozsa family of algorithms. We'll compare the algorithms, work through comprehensive problems, and prepare for the Quantum Fourier Transform next week.

---

## Learning Objectives

1. Compare and contrast all algorithms from this week
2. Identify the common structure underlying these algorithms
3. Solve challenging problems combining multiple concepts
4. Understand the historical development and significance
5. Preview connections to QFT and Shor's algorithm

---

## Week Summary

### The Algorithms at a Glance

| Algorithm | Problem | Classical | Quantum | Speedup Type |
|-----------|---------|-----------|---------|--------------|
| Deutsch | 1-bit constant vs balanced | 2 | 1 | 2x |
| Deutsch-Jozsa | n-bit constant vs balanced | $2^{n-1}+1$ | 1 | Exponential |
| Bernstein-Vazirani | Find hidden string $s$ | $n$ | 1 | Linear → Constant |
| Simon | Find XOR period $s$ | $\Omega(2^{n/2})$ | $O(n)$ | Exponential |

### Common Circuit Structure

All four algorithms share the same basic circuit:

```
|0⟩^⊗n ──[H^⊗n]──●──[H^⊗n]── Measure
                 │
|0/1⟩^⊗m ──[H^⊗m]──⊕─────────
                 U_f
```

The differences lie in:
1. **Initial ancilla state** (determines phase vs standard oracle use)
2. **Oracle structure** (encodes the specific problem)
3. **Measurement interpretation** (what the output means)
4. **Post-processing** (classical computation after measurement)

### The Power of Hadamard

The n-qubit Hadamard transform creates and decodes superpositions:

$$H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle$$

$$H^{\otimes n}|x\rangle = \frac{1}{\sqrt{2^n}}\sum_{y=0}^{2^n-1}(-1)^{x \cdot y}|y\rangle$$

This is the **Fourier transform over $\mathbb{Z}_2^n$**, the foundation of quantum speedups.

### Phase Kickback Principle

When the oracle acts on $|x\rangle|-\rangle$:

$$U_f|x\rangle|-\rangle = (-1)^{f(x)}|x\rangle|-\rangle$$

The function value becomes a **phase** on the input register, enabling interference.

---

## Core Concepts Review

### Why Quantum Wins: Interference

**Classical limitation:** Each query reveals only local information about $f$.

**Quantum advantage:** One query in superposition evaluates $f$ globally, and interference extracts global properties.

```
Classical:    f(x₁), f(x₂), ... (local bits)
                ↓
Quantum:   Σₓ (-1)^f(x)|x⟩ (global phase pattern)
                ↓
         Interference reveals global structure
```

### Promise Problems vs General Functions

All these algorithms require **promises** about $f$:
- Deutsch-Jozsa: constant OR balanced
- Bernstein-Vazirani: $f(x) = s \cdot x$ for some $s$
- Simon: $f(x) = f(y) \Leftrightarrow y \in \{x, x \oplus s\}$

Without promises, no exponential speedup is possible for black-box problems!

### Query Complexity vs Total Complexity

| Algorithm | Query Complexity | Gate Complexity | Post-Processing |
|-----------|------------------|-----------------|-----------------|
| Deutsch-Jozsa | 1 | $O(n)$ | None |
| Bernstein-Vazirani | 1 | $O(n)$ | None |
| Simon | $O(n)$ | $O(n)$ per query | $O(n^3)$ linear algebra |

Query complexity measures oracle calls; total complexity includes everything.

---

## Comprehensive Problems

### Problem Set 1: Algorithm Identification

For each function, identify which algorithm applies and solve:

**1a.** $f: \{0,1\}^4 \to \{0,1\}$ where $f(x) = 1$ iff $x$ has an even number of 1s.

**Solution:**
This is constant vs balanced? Let's check: $f(x) = 1$ for inputs with 0, 2, or 4 ones.
- 0 ones: 1 input (0000)
- 2 ones: 6 inputs
- 4 ones: 1 input (1111)
Total: 8 inputs give $f = 1$, 8 inputs give $f = 0$. **Balanced!**

Also, $f(x) = \lnot(x_1 \oplus x_2 \oplus x_3 \oplus x_4) = 1 \oplus (x_1 \oplus x_2 \oplus x_3 \oplus x_4)$.

This is $f(x) = 1 \oplus s \cdot x$ with $s = 1111$. It's a **Bernstein-Vazirani** problem with a constant offset!

**1b.** $f: \{0,1\}^3 \to \{0,1\}^3$ where $f(000) = f(101) = 001$, $f(001) = f(100) = 010$, etc.

**Solution:**
Check if $f(x) = f(x \oplus s)$ for some $s$:
- $f(000) = f(101)$ → $s$ could be $101$
- $f(001) = f(100)$ → $001 \oplus 100 = 101$ ✓
- This is a **Simon's problem** with $s = 101$.

**1c.** $f(x) = x_1 \land x_2 \land x_3$ (3-bit AND)

**Solution:**
Truth table: $f = 1$ only for $x = 111$, so $f = 0$ for 7 inputs, $f = 1$ for 1 input.
This is **neither constant nor balanced** (7 vs 1). Deutsch-Jozsa promise violated!

### Problem Set 2: State Evolution

**2a.** Trace through Deutsch-Jozsa for $n = 2$ with $f(00) = f(11) = 0$, $f(01) = f(10) = 1$.

**Solution:**

**Initial:** $|00\rangle|1\rangle$

**After $H^{\otimes 3}$:**
$$\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle) \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

**After oracle (phases):**
$$\frac{1}{2}((+1)|00\rangle + (-1)|01\rangle + (-1)|10\rangle + (+1)|11\rangle)|-\rangle$$
$$= \frac{1}{2}(|0\rangle - |1\rangle)(|0\rangle - |1\rangle)|-\rangle = |-\rangle|-\rangle|-\rangle$$

**After $H^{\otimes 2}$ on first register:**
$$H|-\rangle \otimes H|-\rangle = |1\rangle|1\rangle = |11\rangle$$

**Measurement:** $|11\rangle$ → NOT $|00\rangle$, so **balanced**. ✓

**2b.** For Bernstein-Vazirani with $s = 110$ (3 bits), compute all amplitudes after the algorithm.

**Solution:**

After oracle: $\frac{1}{\sqrt{8}}\sum_x (-1)^{s \cdot x}|x\rangle$

Factor: $(-1)^{x_1}(-1)^{x_2}(-1)^0 = (-1)^{x_1 + x_2}$

$$= \frac{1}{\sqrt{8}}(|0\rangle - |1\rangle)(|0\rangle - |1\rangle)(|0\rangle + |1\rangle)$$
$$= |-\rangle|-\rangle|+\rangle$$

After Hadamards: $|1\rangle|1\rangle|0\rangle = |110\rangle$

Amplitudes: $a_{110} = 1$, all others = 0. ✓

### Problem Set 3: Oracle Analysis

**3a.** Design an oracle that marks all strings of Hamming weight $\geq 2$ for 3-bit inputs.

**Solution:**

Marked strings: $\{011, 101, 110, 111\}$ (4 out of 8)

This is a balanced function! Can use in Deutsch-Jozsa.

Boolean expression: $(x_1 \land x_2) \lor (x_1 \land x_3) \lor (x_2 \land x_3)$ = Majority

Oracle circuit uses Toffoli gates for each AND term, then OR the results.

**3b.** Show that the Bernstein-Vazirani oracle for $s$ uses exactly $\text{wt}(s)$ CNOT gates, where $\text{wt}(s)$ is the Hamming weight.

**Solution:**

The oracle computes $f(x) = s \cdot x = \bigoplus_{i: s_i = 1} x_i$.

Each term $x_i$ (when $s_i = 1$) requires one CNOT from $x_i$ to the output.

Total CNOTs = number of 1s in $s$ = $\text{wt}(s)$. ✓

### Problem Set 4: Simon's Algorithm Deep Dive

**4a.** For Simon's algorithm with $n = 4$ and $s = 0110$, list all valid measurement outcomes.

**Solution:**

Valid $y$: those with $y \cdot s = 0$, i.e., $y_2 \oplus y_3 = 0$ ($y_2 = y_3$).

$$\{0000, 0011, 0100, 0111, 1000, 1011, 1100, 1111\}$$

8 valid outcomes (half of 16), forming a subspace of dimension 3.

**4b.** If the first three measurements give $y_1 = 0011$, $y_2 = 1100$, $y_3 = 0111$, find $s$.

**Solution:**

Constraints:
- $y_1 \cdot s = 0$: $s_3 + s_4 = 0$ → $s_3 = s_4$
- $y_2 \cdot s = 0$: $s_1 + s_2 = 0$ → $s_1 = s_2$
- $y_3 \cdot s = 0$: $s_2 + s_3 + s_4 = 0$ → $s_2 + s_3 + s_3 = 0$ → $s_2 = 0$

From $s_1 = s_2 = 0$ and $s_3 = s_4$:

Non-trivial solution: $s = 0011$ or $s = 0000$.

Wait, let me recheck. If $s = 0110$:
- $y_1 \cdot 0110 = 0 \cdot 0 + 0 \cdot 1 + 1 \cdot 1 + 1 \cdot 0 = 1 \neq 0$

That's wrong! Let me recalculate with the correct $s = 0110$:

$y \cdot s = y_2 \oplus y_3$ (since $s = 0110$ means positions 2 and 3 are 1).

Valid $y$: $y_2 = y_3$.

Given measurements must satisfy this. If $y_1 = 0011$: $y_2 = 0, y_3 = 1$. Not valid for $s = 0110$!

The measurements given are inconsistent with $s = 0110$. Either the problem setup is wrong, or I should derive $s$ from the measurements:

Given $y_1, y_2, y_3$, the solution is $s$ such that $s \cdot y_i = 0$ for all $i$, and $s \neq 0$.

Matrix:
$$\begin{pmatrix} 0 & 0 & 1 & 1 \\ 1 & 1 & 0 & 0 \\ 0 & 1 & 1 & 1 \end{pmatrix}$$

Row reduce:
$$\begin{pmatrix} 1 & 1 & 0 & 0 \\ 0 & 1 & 1 & 1 \\ 0 & 0 & 1 & 1 \end{pmatrix}$$

From row 3: $s_3 = s_4$
From row 2: $s_2 = s_3 + s_4 = 0$
From row 1: $s_1 = s_2 = 0$

So $s_1 = s_2 = 0$ and $s_3 = s_4$. Non-trivial: $s = 0011$. ✓

---

## Historical Context

### Timeline

1. **1985:** Deutsch proposes quantum computing model
2. **1992:** Deutsch-Jozsa algorithm (first quantum speedup)
3. **1993:** Bernstein-Vazirani (learning parity functions)
4. **1994:** Simon's algorithm (exponential speedup, inspires Shor)
5. **1994:** Shor's factoring algorithm announced
6. **1996:** Grover's search algorithm

### Impact

- Deutsch-Jozsa proved quantum computers could outperform classical
- Simon showed **exponential** separation was possible
- Simon's structure directly led to Shor's algorithm
- These algorithms established the field of quantum algorithms

---

## Preview: Quantum Fourier Transform

Next week, we'll study the **Quantum Fourier Transform (QFT)**, which generalizes the Hadamard transform:

**Hadamard** = Fourier transform over $\mathbb{Z}_2^n$ (binary group)

**QFT** = Fourier transform over $\mathbb{Z}_N$ (integers mod N)

The QFT enables:
- Quantum phase estimation
- Period finding over $\mathbb{Z}_N$ (not just XOR)
- Shor's algorithm for factoring

Connection: Simon finds period under XOR; Shor finds period under modular addition using QFT.

---

## Computational Lab

```python
"""Day 595: Week 85 Review - Comprehensive Testing"""
import numpy as np
from typing import Callable, Tuple, List

# Import building blocks from previous days
def hadamard_n(n: int) -> np.ndarray:
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    result = H1
    for _ in range(n - 1):
        result = np.kron(result, H1)
    return result

def build_oracle(f: Callable[[int], int], n_in: int, n_out: int = 1) -> np.ndarray:
    """Build standard oracle U_f|x⟩|y⟩ = |x⟩|y⊕f(x)⟩"""
    dim = 2 ** (n_in + n_out)
    U = np.zeros((dim, dim))
    for x in range(2**n_in):
        fx = f(x) & ((1 << n_out) - 1)
        for y in range(2**n_out):
            in_idx = (x << n_out) | y
            out_y = y ^ fx
            out_idx = (x << n_out) | out_y
            U[out_idx, in_idx] = 1
    return U

def run_deutsch_jozsa(f: Callable[[int], int], n: int) -> str:
    """Run Deutsch-Jozsa and return 'constant' or 'balanced'"""
    # State: |0...0⟩|1⟩
    state = np.zeros(2**(n+1), dtype=complex)
    state[1] = 1

    # H on all
    state = np.kron(hadamard_n(n), hadamard_n(1)) @ state

    # Oracle
    U_f = build_oracle(f, n, 1)
    state = U_f @ state

    # H on first n
    state = np.kron(hadamard_n(n), np.eye(2)) @ state

    # P(|0...0⟩ in first n qubits)
    prob_zero = sum(abs(state[i])**2 for i in [0, 1])

    return 'constant' if prob_zero > 0.5 else 'balanced'

def run_bernstein_vazirani(f: Callable[[int], int], n: int) -> int:
    """Run Bernstein-Vazirani and return hidden string s"""
    state = np.zeros(2**(n+1), dtype=complex)
    state[1] = 1

    state = np.kron(hadamard_n(n), hadamard_n(1)) @ state
    U_f = build_oracle(f, n, 1)
    state = U_f @ state
    state = np.kron(hadamard_n(n), np.eye(2)) @ state

    # Measure: find s with highest probability
    probs = np.array([abs(state[2*s])**2 + abs(state[2*s+1])**2
                      for s in range(2**n)])
    return np.argmax(probs)

def inner_product_z2(x: int, y: int) -> int:
    return bin(x & y).count('1') % 2

# Comprehensive tests
print("=" * 70)
print("WEEK 85 COMPREHENSIVE REVIEW")
print("=" * 70)

# Test 1: Deutsch-Jozsa on various functions
print("\n" + "=" * 50)
print("TEST 1: Deutsch-Jozsa Algorithm")
print("=" * 50)

test_cases_dj = [
    (lambda x: 0, 3, "constant", "f(x) = 0"),
    (lambda x: 1, 3, "constant", "f(x) = 1"),
    (lambda x: x & 1, 3, "balanced", "f(x) = LSB"),
    (lambda x: bin(x).count('1') % 2, 3, "balanced", "f(x) = parity"),
    (lambda x: (x >> 2) & 1, 3, "balanced", "f(x) = MSB"),
]

for f, n, expected, desc in test_cases_dj:
    result = run_deutsch_jozsa(f, n)
    status = "PASS" if result == expected else "FAIL"
    print(f"  {desc:20s}: {result:10s} (expected {expected:10s}) [{status}]")

# Test 2: Bernstein-Vazirani on various hidden strings
print("\n" + "=" * 50)
print("TEST 2: Bernstein-Vazirani Algorithm")
print("=" * 50)

for n in [3, 4, 5]:
    for s in [0, 1, (1 << n) - 1, (1 << (n//2))]:
        f = lambda x, s=s: inner_product_z2(x, s)
        result = run_bernstein_vazirani(f, n)
        status = "PASS" if result == s else "FAIL"
        print(f"  n={n}, s={s:0{n}b}: found {result:0{n}b} [{status}]")

# Test 3: Algorithm comparison
print("\n" + "=" * 50)
print("TEST 3: Query Complexity Comparison")
print("=" * 50)

print("\n| Algorithm          | Classical  | Quantum | Speedup       |")
print("|-------------------|------------|---------|---------------|")

for n in [4, 8, 16, 32]:
    # Deutsch-Jozsa
    classical_dj = 2**(n-1) + 1
    quantum_dj = 1
    print(f"| DJ (n={n:2d})          | {classical_dj:10.2e} | {quantum_dj:7d} | {classical_dj/quantum_dj:.2e}x |")

print()
for n in [4, 8, 16, 32]:
    # Bernstein-Vazirani
    classical_bv = n
    quantum_bv = 1
    print(f"| BV (n={n:2d})          | {classical_bv:10d} | {quantum_bv:7d} | {classical_bv/quantum_bv:.0f}x           |")

print()
for n in [4, 8, 16, 32]:
    # Simon
    classical_simon = 2**(n/2)
    quantum_simon = 2*n  # O(n) queries
    print(f"| Simon (n={n:2d})       | {classical_simon:10.2e} | {quantum_simon:7d} | {classical_simon/quantum_simon:.2e}x |")

# Test 4: Oracle efficiency
print("\n" + "=" * 50)
print("TEST 4: Oracle Gate Counts")
print("=" * 50)

print("\n| Function Type      | Input | CNOT | Toffoli | Total |")
print("|-------------------|-------|------|---------|-------|")

# Parity (XOR of all bits)
for n in [2, 4, 8]:
    cnot_count = n  # One CNOT per bit
    print(f"| Parity (n={n:2d})      | {n:5d} | {cnot_count:4d} | {0:7d} | {cnot_count:5d} |")

# BV with various weight strings
for n, wt in [(4, 1), (4, 2), (4, 4), (8, 4)]:
    print(f"| BV (n={n}, wt={wt})     | {n:5d} | {wt:4d} | {0:7d} | {wt:5d} |")

# Test 5: Problem identification
print("\n" + "=" * 50)
print("TEST 5: Problem Classification")
print("=" * 50)

problems = [
    ("f(x) = s·x for unknown s", "Bernstein-Vazirani", "1 query"),
    ("f constant or balanced?", "Deutsch-Jozsa", "1 query"),
    ("f(x)=f(x⊕s), find s", "Simon", "O(n) queries"),
    ("f(x)=f(x+r mod N)", "Shor (next week)", "O(log N) queries"),
    ("Find x where f(x)=1", "Grover (Month 23)", "O(√N) queries"),
]

for problem, algorithm, complexity in problems:
    print(f"  {problem:30s} → {algorithm:20s} ({complexity})")

# Test 6: Interference visualization
print("\n" + "=" * 50)
print("TEST 6: Interference Pattern Analysis")
print("=" * 50)

import matplotlib.pyplot as plt

def visualize_interference_patterns():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    n = 3

    # Constant function
    ax = axes[0, 0]
    f_const = lambda x: 0
    amps = []
    for y in range(2**n):
        amp = sum((-1)**(f_const(x) + inner_product_z2(x, y))
                  for x in range(2**n)) / 2**n
        amps.append(amp)
    ax.bar(range(2**n), amps, color='blue', alpha=0.7)
    ax.set_title('Constant: f(x) = 0')
    ax.set_xlabel('Output |y⟩')
    ax.set_ylabel('Amplitude')
    ax.set_xticks(range(2**n))
    ax.set_xticklabels([f'{i:03b}' for i in range(2**n)], rotation=45)

    # Balanced function (parity)
    ax = axes[0, 1]
    f_parity = lambda x: bin(x).count('1') % 2
    amps = []
    for y in range(2**n):
        amp = sum((-1)**(f_parity(x) + inner_product_z2(x, y))
                  for x in range(2**n)) / 2**n
        amps.append(amp)
    ax.bar(range(2**n), amps, color='red', alpha=0.7)
    ax.set_title('Balanced: f(x) = parity(x)')
    ax.set_xlabel('Output |y⟩')
    ax.set_ylabel('Amplitude')
    ax.set_xticks(range(2**n))
    ax.set_xticklabels([f'{i:03b}' for i in range(2**n)], rotation=45)

    # BV with s = 101
    ax = axes[1, 0]
    s = 0b101
    f_bv = lambda x: inner_product_z2(x, s)
    amps = []
    for y in range(2**n):
        amp = sum((-1)**(f_bv(x) + inner_product_z2(x, y))
                  for x in range(2**n)) / 2**n
        amps.append(amp)
    ax.bar(range(2**n), amps, color='green', alpha=0.7)
    ax.set_title('BV: f(x) = (101)·x')
    ax.set_xlabel('Output |y⟩')
    ax.set_ylabel('Amplitude')
    ax.set_xticks(range(2**n))
    ax.set_xticklabels([f'{i:03b}' for i in range(2**n)], rotation=45)
    ax.axvline(x=5, color='black', linestyle='--', label='s=101')
    ax.legend()

    # BV with s = 011
    ax = axes[1, 1]
    s = 0b011
    f_bv = lambda x: inner_product_z2(x, s)
    amps = []
    for y in range(2**n):
        amp = sum((-1)**(f_bv(x) + inner_product_z2(x, y))
                  for x in range(2**n)) / 2**n
        amps.append(amp)
    ax.bar(range(2**n), amps, color='purple', alpha=0.7)
    ax.set_title('BV: f(x) = (011)·x')
    ax.set_xlabel('Output |y⟩')
    ax.set_ylabel('Amplitude')
    ax.set_xticks(range(2**n))
    ax.set_xticklabels([f'{i:03b}' for i in range(2**n)], rotation=45)
    ax.axvline(x=3, color='black', linestyle='--', label='s=011')
    ax.legend()

    plt.suptitle('Interference Patterns in DJ/BV Algorithms (n=3)', fontsize=14)
    plt.tight_layout()
    plt.savefig('week85_interference.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Interference patterns saved to 'week85_interference.png'")

visualize_interference_patterns()

# Summary statistics
print("\n" + "=" * 50)
print("WEEK 85 SUMMARY")
print("=" * 50)
print("""
Key Achievements This Week:
✓ Understood oracle model and query complexity
✓ Implemented Deutsch and Deutsch-Jozsa algorithms
✓ Mastered Bernstein-Vazirani for hidden string problems
✓ Explored Simon's exponential speedup
✓ Built quantum oracles from Boolean functions

Next Week Preview (QFT):
→ Fourier transform over Z_N (not just Z_2^n)
→ Efficient circuit construction
→ Foundation for phase estimation and Shor
""")
```

---

## Summary

### Week 85 Key Formulas

| Algorithm | Key Formula |
|-----------|-------------|
| Phase Kickback | $U_f\|x\rangle\|-\rangle = (-1)^{f(x)}\|x\rangle\|-\rangle$ |
| Hadamard Transform | $H^{\otimes n}\|x\rangle = \frac{1}{\sqrt{2^n}}\sum_y (-1)^{x \cdot y}\|y\rangle$ |
| DJ Amplitude | $a_y = \frac{1}{2^n}\sum_x (-1)^{f(x) + x \cdot y}$ |
| BV Result | Measure $\|s\rangle$ with certainty |
| Simon Constraint | Measure $y$ with $s \cdot y = 0$ |

### Week 85 Key Takeaways

1. **Common structure**: All algorithms use H-Oracle-H pattern
2. **Phase kickback**: Converts function values to phases
3. **Interference**: Final Hadamard extracts global properties
4. **Query advantage**: Exponential reduction in oracle calls
5. **Promise essential**: Speedups require structured problems
6. **Oracle construction**: Practical implementation uses standard gates

---

## Daily Checklist

- [ ] I can explain the common structure of all four algorithms
- [ ] I understand why interference enables quantum speedups
- [ ] I can solve problems involving all algorithms
- [ ] I can construct oracles for given functions
- [ ] I understand the historical development
- [ ] I'm ready for the Quantum Fourier Transform

---

## Looking Ahead

**Week 86: Quantum Fourier Transform**
- Generalize from $\mathbb{Z}_2^n$ to $\mathbb{Z}_N$
- Efficient $O(n^2)$ circuit construction
- Phase estimation preparation
- Foundation for Shor's algorithm

---

*End of Week 85 | Next: Week 86 - Quantum Fourier Transform*
