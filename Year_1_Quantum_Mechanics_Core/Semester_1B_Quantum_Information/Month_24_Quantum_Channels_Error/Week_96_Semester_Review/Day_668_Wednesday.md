# Day 668: Month 22 Review - Quantum Algorithms I

## Week 96: Semester 1B Review | Month 24: Quantum Channels & Error Introduction

---

## Review Scope

**Month 22: Quantum Algorithms I (Days 589-616)**
- Week 85: Quantum Parallelism and Interference
- Week 86: Deutsch-Jozsa Algorithm
- Week 87: Grover's Search Algorithm
- Week 88: Month Review and Applications

---

## Core Concepts: Quantum Computation Model

### 1. The Circuit Model

**Components:**
- Qubits in states $|0\rangle$, $|1\rangle$, or superpositions
- Quantum gates (unitary operations)
- Measurements (projective, in computational basis)

**Universal gate sets:**
- CNOT + all single-qubit unitaries
- H, T, CNOT (discrete universal set)

### 2. Quantum Parallelism

**Classical:** Evaluate $f(x)$ for one $x$ at a time

**Quantum:** Apply $U_f$ to superposition
$$U_f: |x\rangle|0\rangle \to |x\rangle|f(x)\rangle$$

$$U_f\left(\frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle|0\rangle\right) = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle|f(x)\rangle$$

**"All $N$ values computed simultaneously"** - but accessing them requires interference!

### 3. Quantum Interference

The key to extracting useful information:
- Amplitudes can interfere constructively or destructively
- Algorithm design = engineering interference patterns
- Correct answer amplified, wrong answers suppressed

---

## Core Concepts: Oracle-Based Algorithms

### 4. The Oracle Model

**Oracle:** Black box computing $f: \{0,1\}^n \to \{0,1\}^m$

**Quantum oracle:**
$$O_f: |x\rangle|y\rangle \to |x\rangle|y \oplus f(x)\rangle$$

**Phase oracle variant:**
$$O_f: |x\rangle \to (-1)^{f(x)}|x\rangle$$

(Obtained by applying $O_f$ to $|x\rangle|{-}\rangle$)

### 5. Deutsch's Algorithm

**Problem:** Is $f: \{0,1\} \to \{0,1\}$ constant or balanced?

**Classical:** 2 queries needed
**Quantum:** 1 query suffices!

**Circuit:**
```
|0⟩ ─H─────────●─────────H─ M → f(0)⊕f(1)
               │
|1⟩ ─H─────────⊕───────────
```

**Result:** Measures 0 if constant, 1 if balanced.

### 6. Deutsch-Jozsa Algorithm

**Problem:** Is $f: \{0,1\}^n \to \{0,1\}$ constant or balanced?
(Promise: function is one or the other)

**Classical:** $2^{n-1} + 1$ queries worst case
**Quantum:** 1 query!

**Algorithm:**
1. Prepare $|0\rangle^{\otimes n}|1\rangle$
2. Apply $H^{\otimes (n+1)}$
3. Apply oracle $O_f$
4. Apply $H^{\otimes n}$ to first $n$ qubits
5. Measure first $n$ qubits

**Result:**
- All zeros → constant
- Any non-zero → balanced

**Key formula:**
$$H^{\otimes n}|x\rangle = \frac{1}{\sqrt{2^n}}\sum_{z=0}^{2^n-1}(-1)^{x \cdot z}|z\rangle$$

---

## Core Concepts: Grover's Search

### 7. The Search Problem

**Problem:** Find marked item in unstructured database of $N$ items.

**Classical:** $O(N)$ queries (expected $N/2$)
**Quantum:** $O(\sqrt{N})$ queries!

**Quadratic speedup** - significant for large $N$

### 8. Grover's Algorithm Structure

**Oracle:** $O|x\rangle = (-1)^{f(x)}|x\rangle$ where $f(x) = 1$ for marked items

**Diffusion operator:**
$$D = 2|s\rangle\langle s| - I$$
where $|s\rangle = H^{\otimes n}|0\rangle^{\otimes n}$ is uniform superposition.

**Grover iteration:**
$$G = D \cdot O$$

**Algorithm:**
1. Prepare $|s\rangle = H^{\otimes n}|0\rangle^{\otimes n}$
2. Apply $G$ approximately $\frac{\pi}{4}\sqrt{N}$ times
3. Measure

### 9. Geometric Interpretation

The state lives in 2D subspace spanned by:
- $|w\rangle$: marked states
- $|s'\rangle$: unmarked states

Each Grover iteration rotates by angle $2\theta$ where $\sin\theta = 1/\sqrt{N}$.

After $k$ iterations: angle from $|s'\rangle$ is $(2k+1)\theta$.

Optimal: $k \approx \frac{\pi}{4\theta} \approx \frac{\pi}{4}\sqrt{N}$

### 10. Amplitude Amplification

**General principle:**
- Start with any algorithm $\mathcal{A}$ that produces correct answer with probability $p$
- Amplify to near-certainty with $O(1/\sqrt{p})$ iterations

$$\boxed{\text{Amplitude amplification: } O\left(\frac{1}{\sqrt{p}}\right) \text{ vs classical } O\left(\frac{1}{p}\right)}$$

---

## Quantum Speedups Summary

| Algorithm | Problem | Classical | Quantum | Speedup |
|-----------|---------|-----------|---------|---------|
| Deutsch-Jozsa | Constant vs balanced | $O(2^n)$ | $O(1)$ | Exponential |
| Bernstein-Vazirani | Find hidden string | $O(n)$ | $O(1)$ | Polynomial |
| Simon's | Find period (promise) | $O(2^{n/2})$ | $O(n)$ | Exponential |
| Grover | Unstructured search | $O(N)$ | $O(\sqrt{N})$ | Quadratic |

---

## Integration with Error Correction

### 11. Why Algorithms Need Error Correction

**The problem:**
- Algorithms require many coherent operations
- Each gate has error probability $\epsilon$
- Circuit depth $d$ → error accumulates

**Without correction:** Fidelity $\approx (1-\epsilon)^d \to 0$ for large $d$

**Solution:** Fault-tolerant quantum computing
- Logical qubits encoded in error-correcting codes
- Logical gates applied transversally
- Error threshold theorem: if $\epsilon < \epsilon_{th}$, arbitrary long computation possible

### 12. Resource Overhead

**Rough estimates for fault-tolerant Grover:**
- Physical qubits: $O(n \cdot \text{code overhead})$
- Gate overhead: $O(\text{log}(1/\epsilon_{target}))$
- Time: $O(\sqrt{N} \cdot \text{code distance})$

---

## Practice Problems

### Problem 1: Deutsch-Jozsa

For $n=2$, work through the Deutsch-Jozsa algorithm for:
a) $f(x) = 0$ (constant)
b) $f(x) = x_1 \oplus x_2$ (balanced)

**Solution for (b):**
1. Initial: $|00\rangle|1\rangle$
2. After $H^{\otimes 3}$: $\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle) \otimes |-\rangle$
3. Oracle applies phase $(-1)^{x_1 \oplus x_2}$:
   $\frac{1}{2}(|00\rangle - |01\rangle - |10\rangle + |11\rangle) \otimes |-\rangle$
4. After $H^{\otimes 2}$ on first register: $|11\rangle \otimes |-\rangle$
5. Measure: get 11 (non-zero) → balanced ✓

### Problem 2: Grover Iterations

For $N = 16$ items with 1 marked, how many Grover iterations are optimal?

**Solution:**
$$k_{opt} = \text{round}\left(\frac{\pi}{4}\sqrt{16}\right) = \text{round}(3.14) = 3$$

After 3 iterations, probability of finding marked item ≈ 0.96.

### Problem 3: Speedup Comparison

Compare classical and quantum query complexity for:
a) Searching 1 million items
b) Searching 1 billion items

**Solution:**
| N | Classical | Quantum | Ratio |
|---|-----------|---------|-------|
| $10^6$ | $5 \times 10^5$ | 1000 | 500x |
| $10^9$ | $5 \times 10^8$ | 31623 | 15811x |

---

## Computational Lab

```python
"""Day 668: Month 22 Review - Quantum Algorithms I"""

import numpy as np

# Basis states and gates
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)

def tensor(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

print("Month 22 Review: Quantum Algorithms")
print("=" * 60)

# ============================================
# Part 1: Deutsch's Algorithm
# ============================================
print("\nPART 1: Deutsch's Algorithm")
print("-" * 40)

def deutsch_algorithm(f):
    """
    Run Deutsch's algorithm for f: {0,1} -> {0,1}.
    Returns 0 if constant, 1 if balanced.
    """
    # Oracle: |x⟩|y⟩ -> |x⟩|y⊕f(x)⟩
    def oracle(f):
        O = np.zeros((4, 4), dtype=complex)
        for x in [0, 1]:
            for y in [0, 1]:
                y_out = y ^ f(x)
                O[2*x + y_out, 2*x + y] = 1
        return O

    O = oracle(f)

    # Initial state |0⟩|1⟩
    psi = tensor(ket_0, ket_1)

    # Apply H⊗H
    H2 = tensor(H, H)
    psi = H2 @ psi

    # Apply oracle
    psi = O @ psi

    # Apply H to first qubit
    H_I = tensor(H, I)
    psi = H_I @ psi

    # Measure first qubit
    prob_0 = np.abs(psi[0])**2 + np.abs(psi[1])**2
    prob_1 = np.abs(psi[2])**2 + np.abs(psi[3])**2

    return 0 if prob_0 > 0.5 else 1

# Test all possible functions
functions = {
    "f(x) = 0 (constant)": lambda x: 0,
    "f(x) = 1 (constant)": lambda x: 1,
    "f(x) = x (balanced)": lambda x: x,
    "f(x) = NOT x (balanced)": lambda x: 1 - x
}

for name, f in functions.items():
    result = deutsch_algorithm(f)
    expected = "constant" if f(0) == f(1) else "balanced"
    got = "constant" if result == 0 else "balanced"
    print(f"{name}: {got} (expected {expected}) {'✓' if expected == got else '✗'}")

# ============================================
# Part 2: Deutsch-Jozsa (n=2)
# ============================================
print("\n" + "=" * 60)
print("PART 2: Deutsch-Jozsa (n=2)")
print("-" * 40)

def deutsch_jozsa_2bit(f):
    """
    Deutsch-Jozsa for n=2.
    f: {0,1,2,3} -> {0,1}
    """
    n = 2
    N = 4

    # Build phase oracle
    def phase_oracle(f):
        O = np.zeros((N, N), dtype=complex)
        for x in range(N):
            O[x, x] = (-1)**f(x)
        return O

    O = phase_oracle(f)

    # Initial state |00⟩
    psi = np.zeros((N, 1), dtype=complex)
    psi[0] = 1

    # Apply H⊗H
    H2 = tensor(H, H)
    psi = H2 @ psi

    # Apply oracle
    psi = O @ psi

    # Apply H⊗H again
    psi = H2 @ psi

    # Measure
    probs = np.abs(psi.flatten())**2

    # Constant if prob(|00⟩) = 1
    return "constant" if probs[0] > 0.99 else "balanced"

# Test functions
dj_functions = {
    "f(x) = 0": lambda x: 0,
    "f(x) = 1": lambda x: 1,
    "f(x) = x mod 2": lambda x: x % 2,
    "f(x) = (x+1) mod 2": lambda x: (x + 1) % 2
}

for name, f in dj_functions.items():
    result = deutsch_jozsa_2bit(f)
    values = [f(x) for x in range(4)]
    expected = "constant" if values[0] == values[1] == values[2] == values[3] else "balanced"
    print(f"{name}: {result} (expected {expected}) {'✓' if expected == result else '✗'}")

# ============================================
# Part 3: Grover's Algorithm
# ============================================
print("\n" + "=" * 60)
print("PART 3: Grover's Algorithm")
print("-" * 40)

def grover_search(n, marked_items):
    """
    Grover's algorithm for n qubits.
    marked_items: list of marked indices
    """
    N = 2**n

    # Oracle (phase flip marked items)
    O = np.eye(N, dtype=complex)
    for m in marked_items:
        O[m, m] = -1

    # Diffusion operator D = 2|s⟩⟨s| - I
    s = np.ones((N, 1), dtype=complex) / np.sqrt(N)
    D = 2 * (s @ s.conj().T) - np.eye(N, dtype=complex)

    # Grover operator
    G = D @ O

    # Initial state |s⟩
    psi = s.copy()

    # Number of iterations
    num_marked = len(marked_items)
    theta = np.arcsin(np.sqrt(num_marked / N))
    k_opt = int(np.round(np.pi / (4 * theta) - 0.5))

    probabilities = []
    for k in range(k_opt + 1):
        prob_marked = sum(np.abs(psi[m])**2 for m in marked_items)
        probabilities.append((k, prob_marked))
        psi = G @ psi

    return probabilities, k_opt

# Test with N=16, 1 marked item
n = 4
marked = [7]  # Arbitrary marked item
probs, k_opt = grover_search(n, marked)

print(f"N = {2**n}, marked items = {marked}")
print(f"Optimal iterations: {k_opt}")
print("\nProbability of finding marked item:")
for k, p in probs:
    bar = '#' * int(p * 40)
    print(f"  k={k}: {p:.4f} {bar}")

# Multiple marked items
print("\nWith 4 marked items:")
marked_multi = [1, 5, 9, 13]
probs_multi, k_opt_multi = grover_search(n, marked_multi)
print(f"Optimal iterations: {k_opt_multi}")
for k, p in probs_multi:
    bar = '#' * int(p * 40)
    print(f"  k={k}: {p:.4f} {bar}")

# ============================================
# Part 4: Speedup Analysis
# ============================================
print("\n" + "=" * 60)
print("PART 4: Speedup Analysis")
print("-" * 40)

print("\nQuery complexity comparison:")
print(f"{'N':>12} | {'Classical':>12} | {'Quantum':>12} | {'Speedup':>10}")
print("-" * 52)

for N in [16, 256, 1024, 1_000_000, 1_000_000_000]:
    classical = N // 2
    quantum = int(np.ceil(np.pi * np.sqrt(N) / 4))
    speedup = classical / quantum
    print(f"{N:>12,} | {classical:>12,} | {quantum:>12,} | {speedup:>10.1f}x")

print("\n" + "=" * 60)
print("Review Complete!")
```

---

## Summary

### Key Algorithms

| Algorithm | Key Insight | Speedup |
|-----------|-------------|---------|
| Deutsch | Interference reveals global property | 2x → 1 query |
| Deutsch-Jozsa | Exponential parallelism | $2^n$ → 1 query |
| Grover | Amplitude amplification | $N$ → $\sqrt{N}$ |

### Core Techniques

1. **Quantum parallelism:** Superposition encodes all inputs
2. **Oracle calls:** Black-box function evaluation
3. **Interference:** Amplify correct answers
4. **Amplitude amplification:** General technique for boosting probability

### Connection to Error Correction

- Algorithms require many coherent operations
- Noise accumulates with circuit depth
- Fault-tolerant QC enables practical algorithms
- Error correction overhead significant but manageable

---

## Preview: Day 669

Tomorrow: **Month 23 Review** - Quantum channels, CPTP maps, and channel representations!
