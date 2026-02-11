# Day 597: QFT Definition and Properties

## Overview

**Day 597** | Week 86, Day 2 | Month 22 | Quantum Algorithms I

Today we define the Quantum Fourier Transform (QFT) and explore its mathematical properties. The QFT is the quantum analog of the classical DFT, but implemented as a unitary operation on quantum states. Remarkably, it can be computed exponentially faster than the classical FFT.

---

## Learning Objectives

1. Define the Quantum Fourier Transform on computational basis states
2. Understand the QFT as a unitary matrix
3. Derive the product representation of QFT output states
4. Compare QFT with classical DFT
5. Recognize why QFT provides speedup (but with caveats)
6. Understand the Fourier basis and its properties

---

## Core Content

### Definition of the Quantum Fourier Transform

The **Quantum Fourier Transform** on $n$ qubits transforms computational basis states as:

$$\boxed{QFT|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle}$$

where $N = 2^n$ and $j, k \in \{0, 1, \ldots, N-1\}$.

**Note:** Unlike the classical DFT which uses $e^{-2\pi i}$ (minus sign), the QFT uses $e^{+2\pi i}$. This is a convention; either choice works.

### QFT as a Unitary Matrix

The QFT is represented by the $N \times N$ unitary matrix:

$$QFT_N = \frac{1}{\sqrt{N}}\begin{pmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega & \omega^2 & \cdots & \omega^{N-1} \\
1 & \omega^2 & \omega^4 & \cdots & \omega^{2(N-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega^{N-1} & \omega^{2(N-1)} & \cdots & \omega^{(N-1)^2}
\end{pmatrix}$$

where $\omega = e^{2\pi i/N}$.

The $(j,k)$ entry is $\frac{1}{\sqrt{N}}\omega^{jk}$.

### Unitarity Verification

**Claim:** $QFT_N^\dagger QFT_N = I$

**Proof:**
$$(QFT_N^\dagger QFT_N)_{jk} = \sum_{m=0}^{N-1} (QFT_N^\dagger)_{jm} (QFT_N)_{mk}$$

$$= \sum_{m=0}^{N-1} \frac{1}{\sqrt{N}}\omega^{-mj} \cdot \frac{1}{\sqrt{N}}\omega^{mk}$$

$$= \frac{1}{N}\sum_{m=0}^{N-1} \omega^{m(k-j)}$$

By the orthogonality of roots of unity:
$$= \begin{cases} 1 & \text{if } j = k \\ 0 & \text{if } j \neq k \end{cases} = \delta_{jk}$$

Therefore $QFT_N^\dagger QFT_N = I$. $\square$

### Binary Representation and Fractions

For quantum computing, it's useful to write $j$ in binary:
$$j = j_1 j_2 \cdots j_n = j_1 \cdot 2^{n-1} + j_2 \cdot 2^{n-2} + \cdots + j_n \cdot 2^0$$

**Binary fraction notation:**
$$0.j_l j_{l+1} \cdots j_m = \frac{j_l}{2} + \frac{j_{l+1}}{4} + \cdots + \frac{j_m}{2^{m-l+1}}$$

**Example:** $0.110 = \frac{1}{2} + \frac{1}{4} + 0 = 0.75$

### Product Representation of QFT

**Theorem:** The QFT of a computational basis state can be written as a tensor product:

$$\boxed{QFT|j_1 j_2 \cdots j_n\rangle = \frac{1}{\sqrt{2^n}} \bigotimes_{l=1}^{n} \left(|0\rangle + e^{2\pi i \cdot 0.j_{n-l+1} j_{n-l+2} \cdots j_n}|1\rangle\right)}$$

**Expanded form:**
$$QFT|j\rangle = \frac{1}{\sqrt{2^n}} (|0\rangle + e^{2\pi i \cdot 0.j_n}|1\rangle) \otimes (|0\rangle + e^{2\pi i \cdot 0.j_{n-1}j_n}|1\rangle) \otimes \cdots \otimes (|0\rangle + e^{2\pi i \cdot 0.j_1 j_2 \cdots j_n}|1\rangle)$$

**Proof:**
Start with:
$$QFT|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}|k\rangle$$

Write $k = k_1 k_2 \cdots k_n$ in binary:
$$e^{2\pi ijk/N} = e^{2\pi ij(k_1 2^{n-1} + k_2 2^{n-2} + \cdots + k_n)/2^n}$$
$$= e^{2\pi ij \cdot k_1/2} \cdot e^{2\pi ij \cdot k_2/4} \cdots e^{2\pi ij \cdot k_n/2^n}$$

Since $k_l \in \{0,1\}$ and we sum over all combinations:
$$= \prod_{l=1}^{n} \sum_{k_l=0}^{1} e^{2\pi ij \cdot k_l/2^l} |k_l\rangle$$

$$= \prod_{l=1}^{n} (|0\rangle + e^{2\pi ij/2^l}|1\rangle)$$

Using $j = j_1 2^{n-1} + \cdots + j_n$:
$$e^{2\pi ij/2^l} = e^{2\pi i(j_1 2^{n-l-1} + j_2 2^{n-l-2} + \cdots)}$$

Terms with $2^{k}$ for $k \geq 0$ contribute integer multiples of $2\pi$ (no phase). Only fractional parts matter:
$$e^{2\pi ij/2^l} = e^{2\pi i \cdot 0.j_{n-l+1} j_{n-l+2} \cdots j_n}$$

$\square$

### Fourier Basis

The QFT defines the **Fourier basis** (also called **phase basis**):

$$|\tilde{k}\rangle = QFT|k\rangle = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} e^{2\pi ijk/N}|j\rangle$$

These states are orthonormal: $\langle\tilde{k}|\tilde{l}\rangle = \delta_{kl}$

**Interpretation:** $|\tilde{k}\rangle$ is an equal superposition of all computational basis states with phase that "winds" $k$ times around the unit circle.

### QFT vs Classical DFT

| Aspect | Classical DFT | Quantum QFT |
|--------|--------------|-------------|
| Input | $N$ complex numbers | $\log_2 N$ qubits |
| Output | $N$ complex numbers | $\log_2 N$ qubits |
| Complexity | $O(N \log N)$ | $O((\log N)^2)$ |
| Output Access | All amplitudes | One measurement |

**The Catch:** While QFT is exponentially faster, we cannot directly read out all Fourier coefficients! Measurement gives one sample. The speedup is useful when we need specific information (like periodicity) that can be extracted probabilistically.

### Linearity of QFT

The QFT is linear (it's a matrix!):

$$QFT\left(\sum_j \alpha_j |j\rangle\right) = \sum_j \alpha_j \cdot QFT|j\rangle$$

For a general input state:
$$QFT\sum_j \alpha_j |j\rangle = \sum_k \left(\frac{1}{\sqrt{N}}\sum_j \alpha_j e^{2\pi ijk/N}\right)|k\rangle$$

The amplitude of $|k\rangle$ in the output is the DFT of the input amplitudes!

---

## Worked Examples

### Example 1: 2-Qubit QFT

Write out the QFT matrix for $n = 2$ ($N = 4$).

**Solution:**

$\omega = e^{2\pi i/4} = i$

$$QFT_4 = \frac{1}{2}\begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & i & i^2 & i^3 \\
1 & i^2 & i^4 & i^6 \\
1 & i^3 & i^6 & i^9
\end{pmatrix} = \frac{1}{2}\begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & i & -1 & -i \\
1 & -1 & 1 & -1 \\
1 & -i & -1 & i
\end{pmatrix}$$

### Example 2: QFT Action on Basis States

Compute $QFT|01\rangle$ for 2 qubits.

**Solution:**

Method 1 (Matrix): $j = 1$ in decimal.
$$QFT|01\rangle = QFT_4 \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 \\ i \\ -1 \\ -i \end{pmatrix}$$

$$= \frac{1}{2}(|00\rangle + i|01\rangle - |10\rangle - i|11\rangle)$$

Method 2 (Product representation):
$j = j_1 j_2 = 01$, so $j_1 = 0$, $j_2 = 1$.

$$QFT|01\rangle = \frac{1}{2}(|0\rangle + e^{2\pi i \cdot 0.1}|1\rangle) \otimes (|0\rangle + e^{2\pi i \cdot 0.01}|1\rangle)$$

$$0.1_{\text{binary}} = 1/2$$, $$0.01_{\text{binary}} = 1/4$$

$$= \frac{1}{2}(|0\rangle + e^{i\pi}|1\rangle) \otimes (|0\rangle + e^{i\pi/2}|1\rangle)$$

$$= \frac{1}{2}(|0\rangle - |1\rangle) \otimes (|0\rangle + i|1\rangle)$$

$$= \frac{1}{2}(|00\rangle + i|01\rangle - |10\rangle - i|11\rangle)$$ ✓

### Example 3: QFT of Superposition

Compute $QFT|+\rangle$ for 1 qubit.

**Solution:**

$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$

$$QFT|+\rangle = \frac{1}{\sqrt{2}}(QFT|0\rangle + QFT|1\rangle)$$

For 1 qubit, $QFT = H$:
$$QFT|0\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$
$$QFT|1\rangle = |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

$$QFT|+\rangle = \frac{1}{\sqrt{2}}\left(\frac{|0\rangle + |1\rangle}{\sqrt{2}} + \frac{|0\rangle - |1\rangle}{\sqrt{2}}\right) = |0\rangle$$

The QFT undoes the Hadamard! (For 1 qubit, $QFT = H$ and $H^2 = I$.)

---

## Practice Problems

### Problem 1: 3-Qubit QFT

Write out the product representation for $QFT|101\rangle$ (3 qubits).

### Problem 2: QFT Unitarity

Verify that $(QFT_4)^\dagger QFT_4 = I$ by explicit matrix multiplication.

### Problem 3: Fourier Basis

Express $|0\rangle$ and $|1\rangle$ (1 qubit) in the Fourier basis $\{|\tilde{0}\rangle, |\tilde{1}\rangle\}$.

### Problem 4: Phase Structure

Show that $QFT|j\rangle$ and $QFT|j+N/2\rangle$ differ only by relative phases when the second qubit is measured.

---

## Computational Lab

```python
"""Day 597: QFT Definition and Properties"""
import numpy as np
from typing import List

def qft_matrix(n: int) -> np.ndarray:
    """Construct the QFT matrix for n qubits"""
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    j, k = np.meshgrid(np.arange(N), np.arange(N))
    return (omega ** (j * k)) / np.sqrt(N)

def verify_unitary(U: np.ndarray, name: str = "U") -> bool:
    """Verify that a matrix is unitary"""
    n = U.shape[0]
    product = U.conj().T @ U
    is_unitary = np.allclose(product, np.eye(n))
    print(f"{name} is unitary: {is_unitary}")
    return is_unitary

def binary_fraction(bits: List[int]) -> float:
    """Convert binary fraction 0.b1b2...bn to decimal"""
    result = 0
    for i, bit in enumerate(bits):
        result += bit / (2 ** (i + 1))
    return result

def qft_product_representation(j: int, n: int) -> np.ndarray:
    """Compute QFT|j⟩ using the product representation"""
    # Extract bits (j_1 is MSB)
    bits = [(j >> (n - 1 - i)) & 1 for i in range(n)]

    # Build tensor product
    result = np.array([1.0], dtype=complex)

    for l in range(n):
        # Phase: 0.j_{n-l}...j_n
        phase_bits = bits[n-l-1:]
        phase = binary_fraction(phase_bits)

        # Single qubit state: (|0⟩ + e^{2πi·phase}|1⟩)/√2
        qubit_state = np.array([1, np.exp(2j * np.pi * phase)]) / np.sqrt(2)

        result = np.kron(result, qubit_state)

    return result

def apply_qft_matrix(state: np.ndarray, n: int) -> np.ndarray:
    """Apply QFT using matrix multiplication"""
    QFT = qft_matrix(n)
    return QFT @ state

# Test QFT matrices
print("=" * 60)
print("QFT MATRIX CONSTRUCTION")
print("=" * 60)

for n in [1, 2, 3]:
    print(f"\n--- n = {n} qubits (N = {2**n}) ---")
    QFT = qft_matrix(n)
    print(f"QFT_{2**n} =")
    print(np.round(QFT, 4))
    verify_unitary(QFT, f"QFT_{2**n}")

# Test product representation
print("\n" + "=" * 60)
print("PRODUCT REPRESENTATION TEST")
print("=" * 60)

n = 2
for j in range(2**n):
    # Method 1: Matrix
    basis_state = np.zeros(2**n)
    basis_state[j] = 1
    qft_matrix_result = apply_qft_matrix(basis_state, n)

    # Method 2: Product representation
    qft_product_result = qft_product_representation(j, n)

    # Compare
    match = np.allclose(qft_matrix_result, qft_product_result)
    j_binary = f"{j:0{n}b}"
    print(f"QFT|{j_binary}⟩: Matrix vs Product match: {match}")

    # Show state
    print(f"  State: ", end="")
    for k in range(2**n):
        amp = qft_matrix_result[k]
        if abs(amp) > 1e-10:
            k_binary = f"{k:0{n}b}"
            print(f"{amp:.3f}|{k_binary}⟩ ", end="")
    print()

# Detailed 2-qubit example
print("\n" + "=" * 60)
print("DETAILED 2-QUBIT EXAMPLE")
print("=" * 60)

n = 2
j = 1  # |01⟩
j_bits = [(j >> (n-1-i)) & 1 for i in range(n)]  # [0, 1]
print(f"\nj = {j} = {j:02b}")
print(f"j_1 = {j_bits[0]}, j_2 = {j_bits[1]}")

print("\nProduct representation breakdown:")
for l in range(n):
    phase_bits = j_bits[n-l-1:]
    phase = binary_fraction(phase_bits)
    phase_str = '0.' + ''.join(str(b) for b in phase_bits)
    print(f"  Qubit {l+1}: phase = {phase_str} (binary) = {phase:.4f} (decimal)")
    print(f"           e^(2πi·{phase:.4f}) = {np.exp(2j*np.pi*phase):.4f}")

# Verify with matrix
print("\nVerification:")
state = np.zeros(4)
state[j] = 1
result = apply_qft_matrix(state, 2)
print(f"QFT|01⟩ = {result}")

# Fourier basis
print("\n" + "=" * 60)
print("FOURIER BASIS")
print("=" * 60)

n = 2
print(f"\nFourier basis states |~k⟩ for n={n} qubits:")

for k in range(2**n):
    state = np.zeros(2**n)
    state[k] = 1
    fourier_state = apply_qft_matrix(state, n)

    print(f"|~{k}⟩ = ", end="")
    terms = []
    for j in range(2**n):
        amp = fourier_state[j]
        if abs(amp) > 1e-10:
            terms.append(f"({amp:.3f})|{j:02b}⟩")
    print(" + ".join(terms))

# Comparison with classical DFT
print("\n" + "=" * 60)
print("QFT vs CLASSICAL DFT")
print("=" * 60)

# Classical DFT
def classical_dft(x):
    N = len(x)
    return np.array([sum(x[j] * np.exp(-2j * np.pi * j * k / N)
                         for j in range(N)) for k in range(N)])

# QFT on amplitudes
n = 3
N = 2**n
amplitudes = np.random.randn(N) + 1j * np.random.randn(N)
amplitudes /= np.linalg.norm(amplitudes)

# Classical DFT of amplitudes
classical_result = classical_dft(amplitudes) / np.sqrt(N)

# QFT matrix on state with these amplitudes
qft_result = qft_matrix(n) @ amplitudes

print(f"Input state amplitudes: {np.round(amplitudes[:4], 3)}...")
print(f"Classical DFT (normalized): {np.round(classical_result[:4], 3)}...")
print(f"QFT result:                 {np.round(qft_result[:4], 3)}...")

# Note: Classical DFT uses e^{-2πi}, QFT uses e^{+2πi}, so they're conjugates
print(f"\nNote: QFT uses e^(+2πi), classical uses e^(-2πi)")
print(f"QFT result conjugate: {np.round(np.conj(qft_result[:4]), 3)}...")
print(f"Match with classical: {np.allclose(classical_result, np.conj(qft_result))}")

# Complexity comparison visualization
print("\n" + "=" * 60)
print("COMPLEXITY COMPARISON")
print("=" * 60)

import matplotlib.pyplot as plt

n_values = np.arange(2, 21)
N_values = 2**n_values

# Classical FFT: O(N log N) = O(n * 2^n)
fft_complexity = n_values * N_values

# QFT: O(n²)
qft_complexity = n_values**2

fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(n_values, fft_complexity, 'r-', linewidth=2, label='Classical FFT: O(N log N)')
ax.semilogy(n_values, qft_complexity, 'b-', linewidth=2, label='QFT: O(n²)')

ax.set_xlabel('Number of qubits n', fontsize=12)
ax.set_ylabel('Operations (log scale)', fontsize=12)
ax.set_title('QFT vs Classical FFT Complexity', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate
ax.annotate(f'n=20: FFT ≈ {int(fft_complexity[-1]):.2e}',
           xy=(20, fft_complexity[-1]), xytext=(15, fft_complexity[-1]*10),
           arrowprops=dict(arrowstyle='->'), fontsize=10)
ax.annotate(f'n=20: QFT = {int(qft_complexity[-1])}',
           xy=(20, qft_complexity[-1]), xytext=(15, qft_complexity[-1]*10),
           arrowprops=dict(arrowstyle='->'), fontsize=10)

plt.tight_layout()
plt.savefig('qft_complexity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Complexity comparison saved to 'qft_complexity.png'")

print(f"\nFor n=20 qubits (N = 2²⁰ ≈ 1 million):")
print(f"  Classical FFT: ~{int(20 * 2**20):,} operations")
print(f"  Quantum QFT:   ~{20**2} operations")
print(f"  Speedup:       ~{int(20 * 2**20 / 400):,}x")
```

---

## Summary

### Key Formulas

| Expression | Formula |
|------------|---------|
| QFT definition | $QFT\|j\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi ijk/N}\|k\rangle$ |
| QFT matrix element | $(QFT_N)_{jk} = \frac{1}{\sqrt{N}}\omega^{jk}$ |
| Product representation | $QFT\|j\rangle = \frac{1}{\sqrt{2^n}}\bigotimes_{l=1}^n (\|0\rangle + e^{2\pi i \cdot 0.j_{n-l+1}\cdots j_n}\|1\rangle)$ |
| Binary fraction | $0.b_1 b_2 \cdots b_m = \sum_{i=1}^{m} b_i / 2^i$ |

### Key Takeaways

1. **QFT** is the quantum analog of classical DFT
2. **Product representation** enables efficient circuit construction
3. **Unitarity** ensures reversibility (inverse QFT exists)
4. **Exponential speedup** in gate count: $O(n^2)$ vs $O(N \log N)$
5. **Measurement limitation**: Can't extract all coefficients directly

---

## Daily Checklist

- [ ] I can write the QFT matrix for small n
- [ ] I understand the product representation
- [ ] I can compute QFT of basis states by hand
- [ ] I understand the binary fraction notation
- [ ] I know the complexity advantage of QFT
- [ ] I ran the lab and verified QFT computations

---

*Next: Day 598 - QFT Circuit Construction*
