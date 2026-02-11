# Day 589: Oracle Model and Query Complexity

## Overview

**Day 589** | Week 85, Day 1 | Month 22 | Quantum Algorithms I

Today we establish the theoretical framework for analyzing quantum algorithms: the oracle (black-box) model. This model allows us to prove rigorous separations between classical and quantum computational power by counting the number of queries needed to solve a problem.

---

## Learning Objectives

1. Define the oracle model of computation
2. Distinguish between standard and phase oracles
3. Understand query complexity as a measure of algorithmic efficiency
4. Prove classical lower bounds for specific problems
5. Introduce quantum parallelism through superposition queries
6. Set the stage for Deutsch's algorithm

---

## Core Content

### The Oracle Model

In the **oracle model** (also called the black-box model), we have access to a function $f: \{0,1\}^n \to \{0,1\}^m$ only through queries to an oracle $O_f$. We cannot examine the internal structure of $f$; we can only evaluate it on specific inputs.

**Classical Oracle:**

$$O_f: x \mapsto f(x)$$

We input $x$, receive $f(x)$.

**Quantum Oracle (Standard Form):**

The quantum oracle must be unitary and reversible:

$$\boxed{U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle}$$

where:
- $|x\rangle$ is the n-qubit input register
- $|y\rangle$ is the m-qubit output register
- $\oplus$ denotes bitwise XOR

This is reversible because applying $U_f$ twice returns to the original state:
$$U_f(U_f|x\rangle|y\rangle) = U_f|x\rangle|y \oplus f(x)\rangle = |x\rangle|y \oplus f(x) \oplus f(x)\rangle = |x\rangle|y\rangle$$

### Phase Oracle

A crucial technique is converting the standard oracle to a **phase oracle** using the $|-\rangle$ state:

$$|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

When we apply $U_f$ with $|y\rangle = |-\rangle$:

$$U_f|x\rangle|-\rangle = |x\rangle\frac{1}{\sqrt{2}}(|0 \oplus f(x)\rangle - |1 \oplus f(x)\rangle)$$

For $f(x) = 0$:
$$|x\rangle\frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = |x\rangle|-\rangle$$

For $f(x) = 1$:
$$|x\rangle\frac{1}{\sqrt{2}}(|1\rangle - |0\rangle) = -|x\rangle|-\rangle$$

Therefore:
$$\boxed{U_f|x\rangle|-\rangle = (-1)^{f(x)}|x\rangle|-\rangle}$$

This is called **phase kickback** - the function value appears as a phase on the input register.

### Query Complexity

**Definition:** The query complexity $Q(P)$ of a problem $P$ is the minimum number of oracle queries needed to solve $P$ with bounded error.

We distinguish:
- **Deterministic query complexity** $D(f)$: worst-case queries for deterministic algorithms
- **Randomized query complexity** $R(f)$: expected queries with randomization
- **Quantum query complexity** $Q(f)$: queries for quantum algorithms

**Key relations:**
$$Q(f) \leq R(f) \leq D(f)$$

### Classical Query Lower Bounds

**Theorem (OR Function):** For the OR function on n bits, any deterministic classical algorithm requires n queries in the worst case.

*Proof:* Consider inputs where $f(x) = 0$ for all $x$ versus inputs where exactly one $f(x) = 1$. An adversary can always place the 1 at the last queried position. $\square$

**Theorem (Parity):** Computing $\bigoplus_{i=1}^n x_i$ requires n classical queries.

*Proof:* Each query reveals only one bit of information about the parity. $\square$

### Quantum Parallelism

The power of quantum computing in the oracle model comes from **superposition queries**:

$$U_f\left(\frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle|0\rangle\right) = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle|f(x)\rangle$$

A single quantum query evaluates $f$ on ALL $2^n$ inputs simultaneously!

However, extracting useful information is subtle - measurement collapses the superposition to a single outcome.

### The Deutsch Problem

**Problem:** Given $f: \{0,1\} \to \{0,1\}$, determine if $f$ is constant ($f(0) = f(1)$) or balanced ($f(0) \neq f(1)$).

**Classical:** Requires 2 queries (evaluate $f(0)$ and $f(1)$)

**Quantum:** We will show tomorrow that 1 query suffices!

The key insight is that we need to learn a **global property** of $f$ (is it constant or balanced?), not the specific values $f(0)$ and $f(1)$.

### Promise Problems

The Deutsch problem is a **promise problem**: we are guaranteed that $f$ belongs to a restricted set (constant or balanced functions).

Without the promise, distinguishing would require full knowledge of $f$.

Promise problems are natural in cryptography and optimization.

---

## Worked Examples

### Example 1: Oracle for AND Function

Construct the quantum oracle for $f(x_1, x_2) = x_1 \land x_2$ (AND).

**Solution:**

The truth table:

| $x_1$ | $x_2$ | $f(x_1, x_2)$ |
|-------|-------|---------------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

The oracle acts as:
$$U_f|x_1 x_2\rangle|y\rangle = |x_1 x_2\rangle|y \oplus (x_1 \land x_2)\rangle$$

This is implemented by a **Toffoli gate** (CCNOT):
- Control qubits: $x_1$ and $x_2$
- Target qubit: $y$
- Flips $y$ only when $x_1 = x_2 = 1$

Circuit:
```
x₁ ───●─── x₁
      │
x₂ ───●─── x₂
      │
y  ───⊕─── y ⊕ (x₁ ∧ x₂)
```

### Example 2: Phase Oracle Action

Show the action of the phase oracle for $f(x) = x$ (identity) on the superposition $|+\rangle$.

**Solution:**

Start with $|+\rangle|-\rangle$:
$$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

Apply the phase oracle:
$$U_f|+\rangle|-\rangle = \frac{1}{\sqrt{2}}(U_f|0\rangle|-\rangle + U_f|1\rangle|-\rangle)$$

$$= \frac{1}{\sqrt{2}}((-1)^{f(0)}|0\rangle + (-1)^{f(1)}|1\rangle)|-\rangle$$

For $f(x) = x$: $f(0) = 0$, $f(1) = 1$:
$$= \frac{1}{\sqrt{2}}((-1)^0|0\rangle + (-1)^1|1\rangle)|-\rangle$$

$$= \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)|-\rangle = |-\rangle|-\rangle$$

The $|+\rangle$ state became $|-\rangle$ through the phase oracle!

### Example 3: Classical Lower Bound for XOR

Prove that computing $f(x_1, x_2) = x_1 \oplus x_2$ requires 2 classical queries.

**Solution:**

Suppose we query only $x_1$. There are two possibilities:
- $x_1 = 0$: Then $f$ could be 0 (if $x_2 = 0$) or 1 (if $x_2 = 1$)
- $x_1 = 1$: Then $f$ could be 1 (if $x_2 = 0$) or 0 (if $x_2 = 1$)

In either case, one query leaves two possible values for $f$. The same argument applies if we query $x_2$ first.

Therefore, 2 queries are necessary. $\square$

---

## Practice Problems

### Problem 1: Oracle Construction

Construct the quantum oracle circuit for $f(x_1, x_2) = x_1 \oplus x_2$ (XOR).

*Hint: Which two-qubit gate implements XOR?*

### Problem 2: Phase Oracle Verification

Show that for any function $f: \{0,1\}^n \to \{0,1\}$:
$$U_f\left(\sum_x \alpha_x |x\rangle\right)|-\rangle = \left(\sum_x (-1)^{f(x)}\alpha_x |x\rangle\right)|-\rangle$$

### Problem 3: Query Complexity Analysis

Consider the function $f: \{0,1\}^3 \to \{0,1\}$ that outputs 1 if and only if the input has Hamming weight exactly 2.

(a) What is the deterministic query complexity?
(b) What is a lower bound for randomized query complexity?

### Problem 4: Reversibility

Show that any classical function $f: \{0,1\}^n \to \{0,1\}^m$ can be made reversible by using the $U_f$ construction, regardless of whether $f$ is injective.

---

## Computational Lab

```python
"""Day 589: Oracle Model and Query Complexity"""
import numpy as np
from itertools import product

def classical_oracle(f, x):
    """Classical oracle: simply evaluates f(x)"""
    return f(x)

def build_quantum_oracle(f, n_input, n_output=1):
    """
    Build the unitary matrix for quantum oracle U_f
    U_f|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩

    Total dimension: 2^(n_input + n_output)
    """
    n_total = n_input + n_output
    dim = 2 ** n_total
    U = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        # Extract x (high bits) and y (low bits)
        x = i >> n_output
        y = i & ((1 << n_output) - 1)

        # Compute output: |x⟩|y ⊕ f(x)⟩
        fx = f(x) & ((1 << n_output) - 1)
        y_new = y ^ fx
        j = (x << n_output) | y_new

        U[j, i] = 1

    return U

def verify_oracle_unitary(U):
    """Verify that U is unitary: U†U = I"""
    identity = np.eye(U.shape[0])
    product = U.conj().T @ U
    return np.allclose(product, identity)

def phase_oracle_action(U_f, input_state, n_input):
    """
    Compute action of phase oracle on input_state ⊗ |−⟩
    Returns the modified input register state
    """
    # Create |−⟩ state
    minus = np.array([1, -1]) / np.sqrt(2)

    # Tensor product: input_state ⊗ |−⟩
    full_state = np.kron(input_state, minus)

    # Apply oracle
    output = U_f @ full_state

    # The output is (phase-modified input) ⊗ |−⟩
    # Extract the input register amplitudes
    dim_input = 2 ** n_input
    result = np.zeros(dim_input, dtype=complex)

    for i in range(dim_input):
        # Coefficient comes from |i⟩|−⟩ component
        result[i] = output[2*i] * np.sqrt(2)  # |i⟩|0⟩ coefficient × √2

    return result

def count_queries_classical(f, n, problem_type="exact"):
    """
    Simulate classical query counting
    Returns minimum queries needed in worst case
    """
    if problem_type == "exact":
        # Need to know all values
        return 2**n
    elif problem_type == "or":
        # Worst case: all zeros except last
        return 2**n
    elif problem_type == "constant_vs_balanced":
        # Worst case: need to see more than half
        return 2**(n-1) + 1

# Example functions
def f_constant_0(x): return 0
def f_constant_1(x): return 1
def f_balanced_identity(x): return x & 1  # LSB
def f_and(x): return (x >> 1) & (x & 1)   # For 2-bit: x1 AND x0
def f_xor(x): return ((x >> 1) ^ (x & 1)) & 1  # For 2-bit: x1 XOR x0

# Build and verify oracles
print("=== Oracle Construction and Verification ===\n")

functions = [
    ("Constant 0", f_constant_0),
    ("Constant 1", f_constant_1),
    ("Identity (LSB)", f_balanced_identity),
    ("AND", f_and),
    ("XOR", f_xor)
]

n_input = 2
for name, f in functions:
    U = build_quantum_oracle(f, n_input)
    is_unitary = verify_oracle_unitary(U)
    print(f"{name}: Unitary = {is_unitary}")

    # Show truth table
    print(f"  Truth table: ", end="")
    for x in range(2**n_input):
        print(f"f({x})={f(x)}", end=" ")
    print()

# Phase oracle demonstration
print("\n=== Phase Oracle Action ===\n")

# Create |+⟩ state for 1 qubit
plus = np.array([1, 1]) / np.sqrt(2)

# For identity function (balanced)
U_identity = build_quantum_oracle(lambda x: x, 1)
result = phase_oracle_action(U_identity, plus, 1)
print(f"Identity function on |+⟩:")
print(f"  Input:  |+⟩ = {plus}")
print(f"  Output: {result}")
print(f"  This is |−⟩: {np.allclose(result, np.array([1,-1])/np.sqrt(2))}")

# For constant-0 function
U_const0 = build_quantum_oracle(lambda x: 0, 1)
result = phase_oracle_action(U_const0, plus, 1)
print(f"\nConstant-0 function on |+⟩:")
print(f"  Input:  |+⟩ = {plus}")
print(f"  Output: {result}")
print(f"  This is |+⟩: {np.allclose(result, plus)}")

# Query complexity analysis
print("\n=== Classical Query Complexity ===\n")
for n in range(1, 5):
    queries_exact = count_queries_classical(None, n, "exact")
    queries_cb = count_queries_classical(None, n, "constant_vs_balanced")
    print(f"n={n}: Exact evaluation: {queries_exact}, Constant vs Balanced: {queries_cb}")

# Demonstrate quantum parallelism
print("\n=== Quantum Parallelism ===\n")
n = 3
dim = 2**n

# Create uniform superposition
uniform = np.ones(dim) / np.sqrt(dim)
print(f"Uniform superposition over {dim} states:")
print(f"  Amplitudes: {uniform[:4]}... (showing first 4)")

# After one oracle query, we have f(x) encoded for all x simultaneously
def f_majority(x):
    """Returns 1 if majority of bits are 1"""
    return 1 if bin(x).count('1') > n//2 else 0

# This creates entanglement between input and output
U_maj = build_quantum_oracle(f_majority, n)
ancilla = np.array([1, 0])  # |0⟩
full_input = np.kron(uniform, ancilla)
full_output = U_maj @ full_input

print(f"\nAfter querying majority function oracle:")
print(f"  Input register in superposition of all {dim} basis states")
print(f"  Output register entangled with input, encoding f(x) for each x")
print(f"  Single query computed {dim} function values!")
```

**Expected Output:**
```
=== Oracle Construction and Verification ===

Constant 0: Unitary = True
  Truth table: f(0)=0 f(1)=0 f(2)=0 f(3)=0
Constant 1: Unitary = True
  Truth table: f(0)=1 f(1)=1 f(2)=1 f(3)=1
Identity (LSB): Unitary = True
  Truth table: f(0)=0 f(1)=1 f(2)=0 f(3)=1
AND: Unitary = True
  Truth table: f(0)=0 f(1)=0 f(2)=0 f(3)=1
XOR: Unitary = True
  Truth table: f(0)=0 f(1)=1 f(2)=1 f(3)=0

=== Phase Oracle Action ===

Identity function on |+⟩:
  Input:  |+⟩ = [0.70710678 0.70710678]
  Output: [ 0.70710678-0.j -0.70710678+0.j]
  This is |−⟩: True

Constant-0 function on |+⟩:
  Input:  |+⟩ = [0.70710678 0.70710678]
  Output: [0.70710678-0.j 0.70710678-0.j]
  This is |+⟩: True

=== Classical Query Complexity ===

n=1: Exact evaluation: 2, Constant vs Balanced: 2
n=2: Exact evaluation: 4, Constant vs Balanced: 3
n=3: Exact evaluation: 8, Constant vs Balanced: 5
n=4: Exact evaluation: 16, Constant vs Balanced: 9

=== Quantum Parallelism ===

Uniform superposition over 8 states:
  Amplitudes: [0.35355339 0.35355339 0.35355339 0.35355339]... (showing first 4)

After querying majority function oracle:
  Input register in superposition of all 8 basis states
  Output register entangled with input, encoding f(x) for each x
  Single query computed 8 function values!
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Standard Oracle | $U_f\|x\rangle\|y\rangle = \|x\rangle\|y \oplus f(x)\rangle$ |
| Phase Oracle | $U_f\|x\rangle\|-\rangle = (-1)^{f(x)}\|x\rangle\|-\rangle$ |
| Minus State | $\|-\rangle = \frac{1}{\sqrt{2}}(\|0\rangle - \|1\rangle)$ |
| Query Complexity Order | $Q(f) \leq R(f) \leq D(f)$ |

### Key Takeaways

1. **Oracle model** abstracts away function implementation, focusing on queries
2. **Phase kickback** converts function values to phases on input register
3. **Quantum parallelism** evaluates all inputs in superposition with one query
4. **Extracting information** from superposition is the algorithmic challenge
5. **Promise problems** allow quantum speedups even for global properties

---

## Daily Checklist

- [ ] I understand the difference between standard and phase oracles
- [ ] I can construct a quantum oracle matrix from a truth table
- [ ] I can explain phase kickback and why it's useful
- [ ] I understand classical query lower bounds
- [ ] I see how quantum parallelism differs from parallel classical queries
- [ ] I ran the computational lab and verified oracle unitarity

---

*Next: Day 590 - Deutsch's Algorithm*
