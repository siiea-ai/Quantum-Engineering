# Day 602: Week 86 Review - Quantum Fourier Transform

## Overview

**Day 602** | Week 86, Day 7 | Month 22 | Quantum Algorithms I

Today we consolidate our understanding of the Quantum Fourier Transform. We'll review the key concepts, work through comprehensive problems, and prepare for quantum phase estimation next week.

---

## Learning Objectives

1. Synthesize all QFT concepts from this week
2. Compare QFT with classical Fourier transforms
3. Solve challenging problems combining QFT concepts
4. Understand QFT's role in the quantum algorithm ecosystem
5. Preview phase estimation applications

---

## Week Summary

### Key Concepts Covered

| Day | Topic | Main Ideas |
|-----|-------|------------|
| 596 | Classical DFT | Roots of unity, FFT, $O(N \log N)$ |
| 597 | QFT Definition | Unitary matrix, product representation |
| 598 | QFT Circuit | Hadamard + controlled-$R_k$, $O(n^2)$ |
| 599 | Phase Kickback | Eigenvalue encoding, QFT structure |
| 600 | Inverse QFT | Phase extraction, $R_k^\dagger$ gates |
| 601 | Applications | Arithmetic, simulation, counting |

### The Central Formula

$$\boxed{QFT|j\rangle = \frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi ijk/2^n}|k\rangle}$$

### The Product Representation

$$QFT|j_1\cdots j_n\rangle = \frac{1}{\sqrt{2^n}}\bigotimes_{l=1}^{n}(|0\rangle + e^{2\pi i \cdot 0.j_{n-l+1}\cdots j_n}|1\rangle)$$

### The Circuit Structure

```
|j₁⟩ ─[H]─[R₂]─[R₃]─...─[Rₙ]───────────────────────────×───
          │    │       │                              │
|j₂⟩ ─────●────┼───────┼───[H]─[R₂]─...─[Rₙ₋₁]───────┼───
               │       │        │       │             │
 ⋮             ●       │        ●       │             ⋮
                       │                │             │
|jₙ⟩ ──────────────────●────────────────●─────[H]─────×───
```

### Complexity Comparison

| Transform | Input Size | Operations | Speedup |
|-----------|------------|------------|---------|
| Classical DFT | $N$ | $O(N^2)$ | - |
| FFT | $N$ | $O(N \log N)$ | $O(N/\log N)$ |
| **QFT** | $n = \log N$ | $O(n^2)$ | **Exponential** |

---

## Comprehensive Review Problems

### Problem Set 1: QFT Computation

**1a.** Compute $QFT|5\rangle$ for 3 qubits.

**Solution:**
$n = 3$, $N = 8$, $j = 5 = 101$ in binary.

$$QFT|5\rangle = \frac{1}{\sqrt{8}}\sum_{k=0}^{7} e^{2\pi i \cdot 5k/8}|k\rangle$$

Using product form with $j_1 = 1$, $j_2 = 0$, $j_3 = 1$:

$$= \frac{1}{\sqrt{8}}(|0\rangle + e^{2\pi i \cdot 0.1}|1\rangle)(|0\rangle + e^{2\pi i \cdot 0.01}|1\rangle)(|0\rangle + e^{2\pi i \cdot 0.101}|1\rangle)$$

$$= \frac{1}{\sqrt{8}}(|0\rangle - |1\rangle)(|0\rangle + i|1\rangle)(|0\rangle + e^{5\pi i/4}|1\rangle)$$

**1b.** Verify that $||QFT|5\rangle||^2 = 1$.

**Solution:**
Each factor $(|0\rangle + e^{i\theta}|1\rangle)/\sqrt{2}$ has norm 1.
Product of three such states: $1 \cdot 1 \cdot 1 = 1$. ✓

**1c.** What is $\langle 3|QFT|5\rangle$?

**Solution:**
$$\langle 3|QFT|5\rangle = \langle 3|\frac{1}{\sqrt{8}}\sum_k e^{2\pi i \cdot 5k/8}|k\rangle = \frac{1}{\sqrt{8}}e^{2\pi i \cdot 5 \cdot 3/8} = \frac{1}{\sqrt{8}}e^{15\pi i/4}$$
$$= \frac{1}{\sqrt{8}}e^{-\pi i/4} = \frac{1}{\sqrt{8}}\cdot\frac{1-i}{\sqrt{2}} = \frac{1-i}{4}$$

### Problem Set 2: Circuit Analysis

**2a.** How many gates are in an 8-qubit QFT circuit (including SWAPs)?

**Solution:**
- Hadamards: $n = 8$
- Controlled rotations: $\frac{n(n-1)}{2} = \frac{8 \times 7}{2} = 28$
- SWAPs: $\lfloor n/2 \rfloor = 4$
- **Total: 40 gates**

**2b.** What is the circuit depth of the 8-qubit QFT?

**Solution:**
With parallelization:
- Each "layer" of rotations for qubit $i$ takes $O(1)$ depth (can parallelize)
- Total: $O(n) = O(8)$ depth

Without parallelization: $O(n^2)$ depth.

**2c.** List all controlled-R gates applied to qubit 2 in a 4-qubit QFT.

**Solution:**
Qubit 2 receives:
- $CR_2$ from qubit 3 (when processing qubit 2)
- $CR_3$ from qubit 4 (when processing qubit 2)

Qubit 2 controls:
- $CR_2$ to qubit 1 (when processing qubit 1)

### Problem Set 3: Phase Estimation

**3a.** A unitary $U$ has eigenvalue $e^{2\pi i \cdot 3/16}$. With 4 ancilla qubits, what does phase estimation output?

**Solution:**
$\phi = 3/16$, $2^n \phi = 2^4 \cdot 3/16 = 3$ (exactly).

Output: $|0011\rangle = |3\rangle$ with probability 1.

**3b.** Same as (a) but with eigenvalue $e^{2\pi i \cdot 0.1}$.

**Solution:**
$\phi = 0.1$, $2^4 \cdot 0.1 = 1.6$ (not integer).

Closest integers: 1 and 2.

Measurement probabilities:
- $P(|1\rangle)$: probability of measuring 1
- $P(|2\rangle)$: probability of measuring 2

Using the formula $P(m) = \frac{\sin^2(\pi(2^n\phi - m)N)}{\sin^2(\pi(2^n\phi - m)/N)}$:

$P(|2\rangle) > P(|1\rangle)$ since 1.6 is closer to 2.

**3c.** How many ancilla qubits are needed to estimate $\phi$ with error $< 0.001$?

**Solution:**
Error $\leq \frac{1}{2^n}$

Need $\frac{1}{2^n} < 0.001$
$2^n > 1000$
$n > \log_2(1000) \approx 10$

**Answer: 11 ancilla qubits**

### Problem Set 4: Applications

**4a.** Using QFT-based addition, add $|5\rangle$ and $|3\rangle$ modulo 8.

**Solution:**
1. Start with $|011\rangle$ (b = 3)
2. QFT: $|\tilde{3}\rangle = \frac{1}{\sqrt{8}}\sum_k e^{2\pi i \cdot 3k/8}|k\rangle$
3. Add phases for a = 5: multiply each $|k\rangle$ by $e^{2\pi i \cdot 5k/8}$
4. Result: $\frac{1}{\sqrt{8}}\sum_k e^{2\pi i \cdot 8k/8}|k\rangle = \frac{1}{\sqrt{8}}\sum_k |k\rangle$

Wait, that's the uniform superposition. Let me recalculate.

$\frac{1}{\sqrt{8}}\sum_k e^{2\pi i(3+5)k/8}|k\rangle = \frac{1}{\sqrt{8}}\sum_k e^{2\pi ik}|k\rangle = \frac{1}{\sqrt{8}}\sum_k |k\rangle$

Hmm, $e^{2\pi ik} = 1$ for all integer $k$. So this is $|\tilde{0}\rangle$!

5. Inverse QFT: $|000\rangle = |0\rangle$

**Result:** $5 + 3 = 8 \equiv 0 \pmod 8$. ✓

**4b.** A state has period 4 in a space of size 16. After QFT, where are the peaks?

**Solution:**
Period $r = 4$, total size $N = 16$.

Peaks at multiples of $N/r = 16/4 = 4$:
- $|0\rangle$, $|4\rangle$, $|8\rangle$, $|12\rangle$

---

## Synthesis: QFT in the Algorithm Ecosystem

### Connection Map

```
                    QFT
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   Phase Est.   Period Find.  Arithmetic
        │            │            │
        │      ┌─────┴─────┐      │
        │      │           │      │
        ▼      ▼           ▼      │
      QPE   Simon's    Shor's ◀───┘
        │      │           │
        │      │     ┌─────┴─────┐
        ▼      │     │           │
      HHL   ───┘   Factor    Discrete Log
```

### Why QFT is Fundamental

1. **Converts between conjugate bases**: position ↔ momentum (or computation ↔ Fourier)

2. **Extracts global properties**: periodicity, eigenvalues

3. **Enables interference**: constructive at "correct" answers, destructive elsewhere

4. **Efficient implementation**: exponential speedup over classical

### Key Insight

Classical FFT processes $N$ complex numbers in $O(N \log N)$ time.

QFT processes $n = \log N$ qubits in $O(n^2) = O((\log N)^2)$ time.

**But:** We can't directly read all $N$ Fourier coefficients from the quantum output!

The speedup is useful when we need **specific information** (like periodicity) that can be extracted through measurement.

---

## Computational Lab

```python
"""Day 602: Week 86 Review - Comprehensive QFT Testing"""
import numpy as np
import matplotlib.pyplot as plt

def qft_matrix(n):
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega**(j*k) for k in range(N)]
                     for j in range(N)]) / np.sqrt(N)

def inverse_qft_matrix(n):
    return qft_matrix(n).conj().T

# ===========================================
# Comprehensive Tests
# ===========================================

print("=" * 70)
print("WEEK 86 COMPREHENSIVE REVIEW")
print("=" * 70)

# Test 1: QFT Properties
print("\n" + "=" * 50)
print("TEST 1: QFT Mathematical Properties")
print("=" * 50)

for n in [2, 3, 4]:
    QFT = qft_matrix(n)
    N = 2**n

    # Unitarity
    is_unitary = np.allclose(QFT @ QFT.conj().T, np.eye(N))

    # QFT^4 = I (for properly normalized QFT)
    QFT4 = QFT @ QFT @ QFT @ QFT
    is_fourth_power_identity = np.allclose(QFT4, np.eye(N))

    # Symmetric (up to transpose structure)
    is_symmetric = np.allclose(QFT, QFT.T)

    print(f"\nn = {n}:")
    print(f"  Unitary: {is_unitary}")
    print(f"  QFT^4 = I: {is_fourth_power_identity}")
    print(f"  Symmetric: {is_symmetric}")

# Test 2: Product Representation
print("\n" + "=" * 50)
print("TEST 2: Product Representation Verification")
print("=" * 50)

def binary_fraction(bits):
    """Convert list of bits to binary fraction"""
    return sum(b / (2**(i+1)) for i, b in enumerate(bits))

def qft_product_form(j, n):
    """Compute QFT|j⟩ using product representation"""
    bits = [(j >> (n-1-i)) & 1 for i in range(n)]

    result = np.array([1.0], dtype=complex)
    for l in range(n):
        phase_bits = bits[n-l-1:]
        phase = binary_fraction(phase_bits)
        qubit = np.array([1, np.exp(2j * np.pi * phase)]) / np.sqrt(2)
        result = np.kron(result, qubit)

    return result

for n in [2, 3, 4]:
    print(f"\nn = {n}:")
    all_match = True
    for j in range(2**n):
        matrix_result = qft_matrix(n) @ np.eye(2**n)[:, j]
        product_result = qft_product_form(j, n)
        if not np.allclose(matrix_result, product_result):
            print(f"  Mismatch at j = {j}")
            all_match = False
    if all_match:
        print(f"  All {2**n} states match product form ✓")

# Test 3: Phase Estimation Simulation
print("\n" + "=" * 50)
print("TEST 3: Phase Estimation Accuracy")
print("=" * 50)

def phase_estimation_simulate(phi, n):
    """Simulate phase estimation result"""
    state = np.array([np.exp(2j * np.pi * k * phi)
                      for k in range(2**n)]) / np.sqrt(2**n)
    output = inverse_qft_matrix(n) @ state
    probs = np.abs(output)**2
    k_max = np.argmax(probs)
    return k_max / 2**n, np.max(probs)

print("\n| True φ    | n=4        | n=6        | n=8        |")
print("|-----------|------------|------------|------------|")

for phi in [0.125, 0.25, 0.3, 0.7, 0.9]:
    results = []
    for n in [4, 6, 8]:
        est, prob = phase_estimation_simulate(phi, n)
        error = abs(phi - est)
        results.append(f"{est:.4f}±{error:.4f}")
    print(f"| {phi:.4f}    | {results[0]:10s} | {results[1]:10s} | {results[2]:10s} |")

# Test 4: Circuit Gate Count
print("\n" + "=" * 50)
print("TEST 4: Gate Count Analysis")
print("=" * 50)

def count_gates(n):
    h = n
    cr = n * (n-1) // 2
    swap = n // 2
    return h, cr, swap, h + cr + swap

print("\n| n  | Hadamard | C-Rotation | SWAP | Total | Depth* |")
print("|----|----------|------------|------|-------|--------|")

for n in range(2, 13):
    h, cr, swap, total = count_gates(n)
    depth = 2*n - 1  # Approximate parallelized depth
    print(f"| {n:2d} | {h:8d} | {cr:10d} | {swap:4d} | {total:5d} | {depth:6d} |")

print("\n* Depth assumes maximum parallelization of controlled rotations")

# Test 5: QFT Applications
print("\n" + "=" * 50)
print("TEST 5: QFT Applications Summary")
print("=" * 50)

# 5a: Draper addition
print("\nDraper Addition Test:")
def qft_add(a, b, n):
    """Add a to b using QFT"""
    N = 2**n
    state = np.zeros(N, dtype=complex)
    state[b % N] = 1
    QFT = qft_matrix(n)
    fourier = QFT @ state
    for k in range(N):
        fourier[k] *= np.exp(2j * np.pi * a * k / N)
    result = inverse_qft_matrix(n) @ fourier
    return np.argmax(np.abs(result)**2)

for a, b in [(3, 5), (7, 2), (1, 7), (6, 6)]:
    result = qft_add(a, b, 4)
    expected = (a + b) % 16
    status = "✓" if result == expected else "✗"
    print(f"  {a} + {b} mod 16 = {result} (expected {expected}) {status}")

# 5b: Period detection
print("\nPeriod Detection Test:")
def detect_period(positions, N):
    """Detect period from peak positions after QFT"""
    n = int(np.log2(N))
    state = np.zeros(N, dtype=complex)
    for p in positions:
        state[p] = 1
    state /= np.linalg.norm(state)
    fourier = qft_matrix(n) @ state
    probs = np.abs(fourier)**2
    peaks = [k for k in range(N) if probs[k] > 0.05]
    if len(peaks) > 1:
        period = N // np.gcd.reduce(np.diff(peaks + [N + peaks[0]]))
        return period, peaks
    return None, peaks

for period in [2, 4, 8]:
    N = 16
    positions = list(range(0, N, N//period))
    detected, peaks = detect_period(positions, N)
    status = "✓" if detected == period else "✗"
    print(f"  Period {period}: positions={positions}, peaks={peaks}, "
          f"detected={detected} {status}")

# Visualization: QFT action on different states
print("\n" + "=" * 50)
print("GENERATING VISUALIZATIONS")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

n = 4
N = 2**n
QFT = qft_matrix(n)

# Row 1: Input states
# Row 2: After QFT

# State 1: Basis state |5⟩
state1 = np.zeros(N, dtype=complex)
state1[5] = 1

# State 2: Uniform superposition
state2 = np.ones(N, dtype=complex) / np.sqrt(N)

# State 3: Periodic state (period 4)
state3 = np.zeros(N, dtype=complex)
state3[::4] = 1
state3 /= np.linalg.norm(state3)

states = [state1, state2, state3]
titles = ['Basis |5⟩', 'Uniform', 'Period 4']

for i, (state, title) in enumerate(zip(states, titles)):
    # Input
    axes[0, i].bar(range(N), np.abs(state)**2, color='blue', alpha=0.7)
    axes[0, i].set_title(f'Input: {title}')
    axes[0, i].set_xlabel('Computational basis')
    axes[0, i].set_ylabel('Probability')

    # After QFT
    qft_state = QFT @ state
    axes[1, i].bar(range(N), np.abs(qft_state)**2, color='red', alpha=0.7)
    axes[1, i].set_title(f'After QFT')
    axes[1, i].set_xlabel('Fourier basis')
    axes[1, i].set_ylabel('Probability')

plt.suptitle('QFT Action on Different Input States', fontsize=14)
plt.tight_layout()
plt.savefig('week86_qft_action.png', dpi=150, bbox_inches='tight')
plt.close()
print("QFT action visualization saved to 'week86_qft_action.png'")

# Complexity comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

n_values = np.arange(2, 21)
classical_dft = 2**(2*n_values)  # O(N²) = O(2^2n)
classical_fft = n_values * 2**n_values  # O(N log N) = O(n 2^n)
quantum_qft = n_values**2  # O(n²)

ax.semilogy(n_values, classical_dft, 'r--', linewidth=2, label='Classical DFT: O(N²)')
ax.semilogy(n_values, classical_fft, 'b-', linewidth=2, label='Classical FFT: O(N log N)')
ax.semilogy(n_values, quantum_qft, 'g-', linewidth=3, label='Quantum QFT: O(n²)')

ax.set_xlabel('Number of qubits n (N = 2ⁿ)', fontsize=12)
ax.set_ylabel('Operations (log scale)', fontsize=12)
ax.set_title('Fourier Transform Complexity Comparison', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('week86_complexity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Complexity comparison saved to 'week86_complexity.png'")

# Summary
print("\n" + "=" * 70)
print("WEEK 86 SUMMARY")
print("=" * 70)
print("""
KEY ACCOMPLISHMENTS THIS WEEK:

1. CLASSICAL FOUNDATION (Day 596)
   - DFT definition and FFT algorithm
   - O(N log N) complexity of FFT
   - Roots of unity properties

2. QFT DEFINITION (Day 597)
   - QFT as unitary transformation
   - Product representation
   - O(n²) gate complexity

3. QFT CIRCUIT (Day 598)
   - Hadamard + controlled-R_k gates
   - SWAP for bit reversal
   - Approximate QFT for large n

4. PHASE KICKBACK (Day 599)
   - Eigenvalue encoding mechanism
   - Connection to phase estimation
   - Binary phase representation

5. INVERSE QFT (Day 600)
   - Conjugate transpose structure
   - R_k^† gates
   - Phase extraction

6. APPLICATIONS (Day 601)
   - Quantum arithmetic (Draper adder)
   - Period detection
   - Quantum simulation

NEXT WEEK PREVIEW:
- Formal Quantum Phase Estimation
- QPE circuit design
- Precision and success probability
- Iterative phase estimation
- Kitaev's algorithm
""")
```

---

## Summary

### Week 86 Key Formulas

| Concept | Formula |
|---------|---------|
| QFT definition | $QFT\|j\rangle = \frac{1}{\sqrt{N}}\sum_k e^{2\pi ijk/N}\|k\rangle$ |
| Product form | $\bigotimes_l (\|0\rangle + e^{2\pi i \cdot 0.j_{n-l+1}\cdots j_n}\|1\rangle)/\sqrt{2}$ |
| Gate count | $n + \frac{n(n-1)}{2} + \lfloor n/2 \rfloor = O(n^2)$ |
| Inverse QFT | $QFT^{-1} = QFT^\dagger$, replace $R_k \to R_k^\dagger$ |

### Week 86 Key Takeaways

1. **QFT is exponentially faster** than classical: $O(n^2)$ vs $O(N \log N)$
2. **Product representation** enables efficient circuit construction
3. **Phase kickback** connects QFT to eigenvalue problems
4. **Inverse QFT** extracts phases from quantum states
5. **Applications** span arithmetic, simulation, and period finding

---

## Daily Checklist

- [ ] I can compute QFT for small inputs
- [ ] I understand the circuit construction
- [ ] I know how phase kickback creates QFT structure
- [ ] I can apply inverse QFT for phase extraction
- [ ] I see applications beyond phase estimation
- [ ] I'm ready for quantum phase estimation

---

## Looking Ahead

**Week 87: Quantum Phase Estimation**
- Formal QPE algorithm
- Circuit design with ancilla qubits
- Precision analysis
- Success probability bounds
- Iterative and robust variants

---

*End of Week 86 | Next: Week 87 - Phase Estimation*
