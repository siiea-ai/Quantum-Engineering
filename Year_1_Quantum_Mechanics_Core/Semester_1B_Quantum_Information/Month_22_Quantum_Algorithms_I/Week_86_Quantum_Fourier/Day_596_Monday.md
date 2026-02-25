# Day 596: Classical DFT Review

## Overview

**Day 596** | Week 86, Day 1 | Month 22 | Quantum Algorithms I

Today we review the classical Discrete Fourier Transform (DFT), which provides the mathematical foundation for understanding the Quantum Fourier Transform. We'll cover the definition, properties, and the Fast Fourier Transform (FFT) algorithm that achieves $O(N \log N)$ complexity.

---

## Learning Objectives

1. Define the Discrete Fourier Transform mathematically
2. Understand the role of roots of unity
3. Derive key properties: linearity, convolution theorem
4. Analyze the computational complexity of naive DFT
5. Understand the FFT algorithm and its divide-and-conquer structure
6. Connect DFT to signal processing and polynomial multiplication

---

## Core Content

### The Discrete Fourier Transform

The **Discrete Fourier Transform** transforms a sequence of $N$ complex numbers $\{x_0, x_1, \ldots, x_{N-1}\}$ into another sequence $\{X_0, X_1, \ldots, X_{N-1}\}$:

$$\boxed{X_k = \sum_{j=0}^{N-1} x_j \cdot e^{-2\pi ijk/N}}$$

or equivalently:

$$\boxed{X_k = \sum_{j=0}^{N-1} x_j \cdot \omega_N^{-jk}}$$

where $\omega_N = e^{2\pi i/N}$ is the **primitive $N$th root of unity**.

### Roots of Unity

The $N$th roots of unity are solutions to $z^N = 1$:

$$\omega_N^k = e^{2\pi ik/N} \text{ for } k = 0, 1, \ldots, N-1$$

**Key Properties:**

1. **Periodicity:** $\omega_N^{k+N} = \omega_N^k$

2. **Symmetry:** $\omega_N^{k + N/2} = -\omega_N^k$ (for even $N$)

3. **Orthogonality:**
$$\sum_{j=0}^{N-1} \omega_N^{jk} = \begin{cases} N & \text{if } k \equiv 0 \pmod{N} \\ 0 & \text{otherwise} \end{cases}$$

4. **Roots sum to zero:** $\sum_{k=0}^{N-1} \omega_N^k = 0$

### Matrix Form

The DFT can be written as matrix multiplication $\mathbf{X} = F_N \mathbf{x}$:

$$F_N = \begin{pmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega_N^{-1} & \omega_N^{-2} & \cdots & \omega_N^{-(N-1)} \\
1 & \omega_N^{-2} & \omega_N^{-4} & \cdots & \omega_N^{-2(N-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega_N^{-(N-1)} & \omega_N^{-2(N-1)} & \cdots & \omega_N^{-(N-1)^2}
\end{pmatrix}$$

The $(j,k)$ entry is $\omega_N^{-jk}$.

### Inverse DFT

The inverse DFT recovers the original sequence:

$$\boxed{x_j = \frac{1}{N}\sum_{k=0}^{N-1} X_k \cdot \omega_N^{jk}}$$

The inverse DFT matrix is:
$$F_N^{-1} = \frac{1}{N} F_N^*$$

where $F_N^*$ is the conjugate transpose.

### Properties of the DFT

**1. Linearity:**
$$\text{DFT}(\alpha x + \beta y) = \alpha \cdot \text{DFT}(x) + \beta \cdot \text{DFT}(y)$$

**2. Time Shift:**
If $y_j = x_{j-m}$ (cyclic shift), then $Y_k = X_k \cdot \omega_N^{-mk}$

**3. Frequency Shift:**
If $y_j = x_j \cdot \omega_N^{mj}$, then $Y_k = X_{k-m}$

**4. Convolution Theorem:**
$$\text{DFT}(x * y) = \text{DFT}(x) \cdot \text{DFT}(y)$$

where $*$ is circular convolution and $\cdot$ is pointwise multiplication.

**5. Parseval's Theorem:**
$$\sum_{j=0}^{N-1} |x_j|^2 = \frac{1}{N}\sum_{k=0}^{N-1} |X_k|^2$$

### Computational Complexity: Naive DFT

Computing one $X_k$ requires:
- $N$ multiplications (each $x_j \cdot \omega_N^{-jk}$)
- $N-1$ additions

Total for all $N$ outputs: $O(N^2)$ operations.

For large $N$, this is prohibitively expensive!

### The Fast Fourier Transform (FFT)

The **Cooley-Tukey FFT** (1965) reduces complexity to $O(N \log N)$ using divide-and-conquer.

**Key Insight:** Split into even and odd indices:

$$X_k = \sum_{j=0}^{N-1} x_j \omega_N^{-jk} = \sum_{m=0}^{N/2-1} x_{2m} \omega_N^{-2mk} + \sum_{m=0}^{N/2-1} x_{2m+1} \omega_N^{-(2m+1)k}$$

$$= \sum_{m=0}^{N/2-1} x_{2m} \omega_{N/2}^{-mk} + \omega_N^{-k} \sum_{m=0}^{N/2-1} x_{2m+1} \omega_{N/2}^{-mk}$$

$$= E_k + \omega_N^{-k} O_k$$

where $E_k$ is the DFT of even-indexed elements and $O_k$ is the DFT of odd-indexed elements.

**Butterfly Operation:**
$$X_k = E_k + \omega_N^{-k} O_k$$
$$X_{k+N/2} = E_k - \omega_N^{-k} O_k$$

### FFT Complexity Analysis

**Recurrence:** $T(N) = 2T(N/2) + O(N)$

**Solution:** $T(N) = O(N \log N)$

For $N = 2^n$:
- Naive DFT: $O(4^n)$ operations
- FFT: $O(n \cdot 2^n)$ operations

| $N$ | Naive DFT | FFT | Speedup |
|-----|-----------|-----|---------|
| $2^{10}$ | $\sim 10^6$ | $\sim 10^4$ | 100x |
| $2^{20}$ | $\sim 10^{12}$ | $\sim 2 \times 10^7$ | 50,000x |

### Applications of DFT

1. **Signal Processing:** Frequency analysis, filtering
2. **Image Processing:** 2D DFT for image compression (JPEG)
3. **Polynomial Multiplication:** $O(N \log N)$ instead of $O(N^2)$
4. **Cryptography:** Number-theoretic transforms
5. **Quantum Computing:** QFT enables phase estimation, Shor's algorithm

---

## Worked Examples

### Example 1: 4-Point DFT

Compute the DFT of $x = (1, 0, -1, 0)$.

**Solution:**

$N = 4$, $\omega_4 = e^{2\pi i/4} = i$

Powers: $\omega_4^0 = 1$, $\omega_4^1 = i$, $\omega_4^2 = -1$, $\omega_4^3 = -i$

$$X_k = \sum_{j=0}^{3} x_j \omega_4^{-jk}$$

$X_0 = x_0 + x_1 + x_2 + x_3 = 1 + 0 + (-1) + 0 = 0$

$X_1 = x_0 + x_1 \omega_4^{-1} + x_2 \omega_4^{-2} + x_3 \omega_4^{-3}$
$= 1 + 0 \cdot (-i) + (-1) \cdot (-1) + 0 \cdot i = 1 + 1 = 2$

$X_2 = x_0 + x_1 \omega_4^{-2} + x_2 \omega_4^{-4} + x_3 \omega_4^{-6}$
$= 1 + 0 \cdot (-1) + (-1) \cdot 1 + 0 \cdot (-1) = 0$

$X_3 = x_0 + x_1 \omega_4^{-3} + x_2 \omega_4^{-6} + x_3 \omega_4^{-9}$
$= 1 + 0 \cdot i + (-1) \cdot (-1) + 0 \cdot (-i) = 2$

**Result:** $X = (0, 2, 0, 2)$

### Example 2: Inverse DFT

Recover $x$ from $X = (0, 2, 0, 2)$.

**Solution:**

$$x_j = \frac{1}{4}\sum_{k=0}^{3} X_k \omega_4^{jk}$$

$x_0 = \frac{1}{4}(X_0 + X_1 + X_2 + X_3) = \frac{1}{4}(0 + 2 + 0 + 2) = 1$

$x_1 = \frac{1}{4}(X_0 + X_1 \omega_4 + X_2 \omega_4^2 + X_3 \omega_4^3)$
$= \frac{1}{4}(0 + 2i + 0 + 2(-i)) = 0$

$x_2 = \frac{1}{4}(X_0 + X_1 \omega_4^2 + X_2 \omega_4^4 + X_3 \omega_4^6)$
$= \frac{1}{4}(0 + 2(-1) + 0 + 2(-1)) = -1$

$x_3 = \frac{1}{4}(X_0 + X_1 \omega_4^3 + X_2 \omega_4^6 + X_3 \omega_4^9)$
$= \frac{1}{4}(0 + 2(-i) + 0 + 2i) = 0$

**Recovered:** $x = (1, 0, -1, 0)$ ✓

### Example 3: FFT Butterfly

Apply one level of FFT decomposition to $x = (a, b, c, d)$.

**Solution:**

Split: Even = $(a, c)$, Odd = $(b, d)$

Compute 2-point DFTs:
- $E_0 = a + c$, $E_1 = a - c$
- $O_0 = b + d$, $O_1 = b - d$

Combine with twiddle factors ($\omega_4^0 = 1$, $\omega_4^{-1} = -i$):
- $X_0 = E_0 + \omega_4^0 O_0 = (a+c) + (b+d)$
- $X_1 = E_1 + \omega_4^{-1} O_1 = (a-c) - i(b-d)$
- $X_2 = E_0 - \omega_4^0 O_0 = (a+c) - (b+d)$
- $X_3 = E_1 - \omega_4^{-1} O_1 = (a-c) + i(b-d)$

---

## Practice Problems

### Problem 1: DFT Computation

Compute the 8-point DFT of $x = (1, 1, 1, 1, 0, 0, 0, 0)$.

### Problem 2: Orthogonality

Verify the orthogonality relation:
$$\sum_{j=0}^{7} \omega_8^{j \cdot 3} = 0$$

### Problem 3: Convolution

Using the convolution theorem, compute the circular convolution of $(1, 2)$ and $(3, 4)$ with $N = 2$.

### Problem 4: FFT Steps

How many levels of recursion and how many butterfly operations are needed for an $N = 16$ point FFT?

---

## Computational Lab

```python
"""Day 596: Classical DFT Review"""
import numpy as np
import matplotlib.pyplot as plt

def dft_naive(x):
    """Compute DFT using naive O(N²) algorithm"""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for j in range(N):
            X[k] += x[j] * np.exp(-2j * np.pi * j * k / N)
    return X

def idft_naive(X):
    """Compute inverse DFT using naive algorithm"""
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for j in range(N):
        for k in range(N):
            x[j] += X[k] * np.exp(2j * np.pi * j * k / N)
    return x / N

def fft_recursive(x):
    """Compute FFT using Cooley-Tukey recursive algorithm"""
    N = len(x)
    if N == 1:
        return x.copy()

    # Split into even and odd
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])

    # Twiddle factors
    twiddle = np.exp(-2j * np.pi * np.arange(N//2) / N)

    # Combine
    X = np.zeros(N, dtype=complex)
    X[:N//2] = even + twiddle * odd
    X[N//2:] = even - twiddle * odd

    return X

def dft_matrix(N):
    """Construct the N×N DFT matrix"""
    j, k = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2j * np.pi / N)
    return omega ** (j * k)

# Test DFT implementations
print("=" * 60)
print("DFT IMPLEMENTATION TESTS")
print("=" * 60)

# Test case from worked example
x = np.array([1, 0, -1, 0], dtype=complex)
print(f"\nInput: x = {x}")

X_naive = dft_naive(x)
X_fft = fft_recursive(x)
X_numpy = np.fft.fft(x)

print(f"Naive DFT:  {np.round(X_naive, 4)}")
print(f"FFT:        {np.round(X_fft, 4)}")
print(f"NumPy FFT:  {np.round(X_numpy, 4)}")

# Verify inverse
x_recovered = idft_naive(X_naive)
print(f"Recovered:  {np.round(x_recovered, 4)}")

# Verify DFT matrix
print("\n" + "=" * 60)
print("DFT MATRIX (N=4)")
print("=" * 60)

F4 = dft_matrix(4)
print("\nDFT Matrix F_4:")
print(np.round(F4, 4))

print("\nF_4 @ x:")
print(np.round(F4 @ x, 4))

# Roots of unity visualization
print("\n" + "=" * 60)
print("ROOTS OF UNITY")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, N in zip(axes, [4, 8, 16]):
    roots = [np.exp(2j * np.pi * k / N) for k in range(N)]

    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'gray', linestyle='--', alpha=0.5)

    # Plot roots
    for k, r in enumerate(roots):
        ax.plot(r.real, r.imag, 'bo', markersize=10)
        ax.annotate(f'$\\omega_{N}^{k}$', (r.real*1.15, r.imag*1.15),
                   fontsize=8, ha='center', va='center')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(f'{N}th Roots of Unity')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')

plt.tight_layout()
plt.savefig('roots_of_unity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Roots of unity saved to 'roots_of_unity.png'")

# Complexity comparison
print("\n" + "=" * 60)
print("COMPLEXITY COMPARISON")
print("=" * 60)

import time

sizes = [2**k for k in range(4, 12)]
naive_times = []
fft_times = []
numpy_times = []

for N in sizes:
    x = np.random.randn(N) + 1j * np.random.randn(N)

    if N <= 512:
        start = time.time()
        _ = dft_naive(x)
        naive_times.append(time.time() - start)
    else:
        naive_times.append(None)

    start = time.time()
    _ = fft_recursive(x)
    fft_times.append(time.time() - start)

    start = time.time()
    _ = np.fft.fft(x)
    numpy_times.append(time.time() - start)

print("\n| N    | Naive DFT    | Recursive FFT | NumPy FFT   |")
print("|------|--------------|---------------|-------------|")
for i, N in enumerate(sizes):
    naive_str = f"{naive_times[i]*1000:.3f} ms" if naive_times[i] else "too slow"
    print(f"| {N:4d} | {naive_str:12s} | {fft_times[i]*1000:.3f} ms     | {numpy_times[i]*1000:.6f} ms |")

# Signal processing example
print("\n" + "=" * 60)
print("SIGNAL PROCESSING EXAMPLE")
print("=" * 60)

# Create a signal with two frequencies
N = 256
t = np.arange(N)
freq1, freq2 = 10, 25  # Frequencies
signal = np.sin(2 * np.pi * freq1 * t / N) + 0.5 * np.sin(2 * np.pi * freq2 * t / N)
signal += 0.3 * np.random.randn(N)  # Add noise

# Compute DFT
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, 1/N)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(t, signal, 'b-', linewidth=0.5)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Time Domain Signal')

axes[1].plot(freqs[:N//2], np.abs(spectrum[:N//2]) / N, 'r-')
axes[1].set_xlabel('Frequency')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Frequency Domain (DFT)')
axes[1].axvline(x=freq1, color='green', linestyle='--', label=f'f={freq1}')
axes[1].axvline(x=freq2, color='green', linestyle='--', label=f'f={freq2}')
axes[1].legend()

plt.tight_layout()
plt.savefig('dft_signal_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Signal analysis saved to 'dft_signal_analysis.png'")

# Convolution theorem demonstration
print("\n" + "=" * 60)
print("CONVOLUTION THEOREM")
print("=" * 60)

# Two signals
a = np.array([1, 2, 3, 0, 0, 0])
b = np.array([1, 1, 1, 0, 0, 0])

# Direct convolution
conv_direct = np.convolve(a, b, mode='full')[:len(a)]

# Via FFT
A = np.fft.fft(a)
B = np.fft.fft(b)
conv_fft = np.real(np.fft.ifft(A * B))

print(f"a = {a}")
print(f"b = {b}")
print(f"Direct convolution: {conv_direct}")
print(f"FFT convolution:    {np.round(conv_fft, 4)}")

# Properties verification
print("\n" + "=" * 60)
print("DFT PROPERTIES VERIFICATION")
print("=" * 60)

N = 8
x = np.random.randn(N) + 1j * np.random.randn(N)
X = np.fft.fft(x)

# Parseval's theorem
energy_time = np.sum(np.abs(x)**2)
energy_freq = np.sum(np.abs(X)**2) / N
print(f"\nParseval's theorem:")
print(f"  Σ|x_j|² = {energy_time:.6f}")
print(f"  (1/N)Σ|X_k|² = {energy_freq:.6f}")
print(f"  Equal: {np.isclose(energy_time, energy_freq)}")

# Orthogonality
print(f"\nOrthogonality (k=3):")
omega_8 = np.exp(2j * np.pi / 8)
sum_roots = sum(omega_8**(j*3) for j in range(8))
print(f"  Σ ω₈^(3j) = {sum_roots:.6f} ≈ 0")
```

---

## Summary

### Key Formulas

| Transform | Formula |
|-----------|---------|
| DFT | $X_k = \sum_{j=0}^{N-1} x_j \omega_N^{-jk}$ |
| IDFT | $x_j = \frac{1}{N}\sum_{k=0}^{N-1} X_k \omega_N^{jk}$ |
| Root of Unity | $\omega_N = e^{2\pi i/N}$ |
| FFT Butterfly | $X_k = E_k + \omega_N^{-k} O_k$ |

### Complexity Comparison

| Algorithm | Complexity |
|-----------|------------|
| Naive DFT | $O(N^2)$ |
| FFT | $O(N \log N)$ |
| **QFT (preview)** | $O((\log N)^2)$ |

### Key Takeaways

1. **DFT** transforms between time and frequency domains
2. **Roots of unity** are the building blocks of the DFT
3. **FFT** achieves $O(N \log N)$ via divide-and-conquer
4. **Convolution theorem** enables fast multiplication
5. **QFT** will achieve exponential speedup over FFT!

---

## Daily Checklist

- [ ] I can compute the DFT by hand for small inputs
- [ ] I understand roots of unity and their properties
- [ ] I can explain the FFT divide-and-conquer structure
- [ ] I know the complexity of naive DFT vs FFT
- [ ] I understand the convolution theorem
- [ ] I ran the lab and verified DFT implementations

---

*Next: Day 597 - QFT Definition and Properties*
