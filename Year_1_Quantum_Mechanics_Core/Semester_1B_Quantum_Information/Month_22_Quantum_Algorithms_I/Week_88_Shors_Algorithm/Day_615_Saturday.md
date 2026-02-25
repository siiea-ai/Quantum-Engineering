# Day 615: Complexity Analysis

## Overview

**Day 615** | Week 88, Day 6 | Month 22 | Quantum Algorithms I

Today we analyze the computational complexity of Shor's algorithm in detail. We count gates, analyze circuit depth, examine space requirements, and compare quantum versus classical factoring complexity.

---

## Learning Objectives

1. Count gates in Shor's algorithm
2. Analyze circuit depth and parallelism
3. Understand space-time tradeoffs
4. Compare quantum vs classical complexity
5. Evaluate cryptographic implications

---

## Core Content

### Overview of Complexity

**Classical factoring (GNFS):**
$$T_{\text{classical}} = O\left(\exp\left(c \cdot n^{1/3} (\log n)^{2/3}\right)\right)$$

where $n = \log_2 N$ is the bit-length.

**Quantum factoring (Shor):**
$$T_{\text{quantum}} = O(n^3)$$

**Exponential speedup:** From sub-exponential to polynomial!

### Gate Count Breakdown

#### 1. Quantum Fourier Transform

**Standard QFT on $m$ qubits:**
- Hadamard gates: $m$
- Controlled-phase gates: $m(m-1)/2$
- Swap gates: $\lfloor m/2 \rfloor$

**Total:** $O(m^2)$ gates

For $m = 2n$ ancillas: $O(n^2)$ gates

#### 2. Controlled Modular Exponentiation

This is the dominant cost!

**Single modular multiplication $|x\rangle \to |ax \mod N\rangle$:**
- Uses modular arithmetic circuits
- Cost: $O(n^2)$ elementary gates (schoolbook method)
- Or: $O(n \log n \log \log n)$ with FFT-based multiplication

**Controlled version $CU_a^{2^k}$:**
- Same as uncontrolled with additional control lines
- Cost: $O(n^2)$ per controlled multiplication

**Total controlled exponentiations:**
- $2n$ controlled operations (one per ancilla)
- Each controls multiplication by $a^{2^k} \mod N$
- Total: $O(n \cdot n^2) = O(n^3)$ gates

### Total Gate Count

| Component | Gate Count |
|-----------|------------|
| Hadamards (initial) | $O(n)$ |
| Controlled mod-exp | $O(n^3)$ |
| Inverse QFT | $O(n^2)$ |
| **Total** | $O(n^3)$ |

With optimizations: Can approach $O(n^2 \log n \log \log n)$

### Circuit Depth Analysis

**Sequential depth:**
- Controlled operations: $2n$ layers
- Each layer: depth $O(n^2)$
- Total depth: $O(n^3)$

**Parallelized depth:**
- QFT: $O(n)$ with approximate QFT
- Mod-exp: $O(n^2)$ per operation (inherently sequential)
- Best known: $O(n^3)$ depth

### Space Complexity

**Qubit count:**

| Register | Qubits | Purpose |
|----------|--------|---------|
| Ancilla | $2n$ | Phase estimation |
| Work | $n$ | Store $\|x \mod N\rangle$ |
| Scratch | $O(n)$ | Intermediate computations |
| **Total** | $O(n)$ | Linear in input size |

**Memory-time tradeoffs:**
- Standard: $O(n)$ qubits, $O(n^3)$ gates
- Beauregard's circuit: $2n + 3$ qubits with more gates
- Zalka's improvements: Various tradeoffs available

### Classical Post-Processing

**Continued fractions:** $O(n)$ operations

**GCD computation:** $O(n^2)$ bit operations (Euclidean algorithm)

**Primality testing:** $O(n^3)$ (Miller-Rabin)

All polynomial - negligible compared to quantum cost.

### Success Probability

**Per QPE run:**
$$P(\text{correct } s/r) \geq \frac{4}{\pi^2} \approx 0.405$$

**Per complete attempt:**
$$P(\text{factor found}) \geq \frac{1}{2} \cdot \frac{1}{2} \cdot 0.405 \approx 0.1$$

**Expected attempts:** $O(1)$ to $O(\log N)$

**Overall complexity:** Still $O(n^3)$ with constant factor increase.

### Comparison Table

| Algorithm | Time Complexity | Space | Type |
|-----------|-----------------|-------|------|
| Trial division | $O(2^{n/2})$ | $O(n)$ | Classical |
| Pollard's rho | $O(2^{n/4})$ | $O(n)$ | Classical |
| Quadratic sieve | $O(\exp(cn^{1/2}))$ | $O(\exp(cn^{1/2}))$ | Classical |
| GNFS | $O(\exp(cn^{1/3}(\log n)^{2/3}))$ | Large | Classical |
| **Shor's** | $O(n^3)$ | $O(n)$ | Quantum |

### Cryptographic Implications

**RSA key sizes and estimated breaking time:**

| Key Size (bits) | Classical (years) | Quantum (operations) |
|-----------------|-------------------|----------------------|
| 1024 | ~10^6 | ~10^9 |
| 2048 | ~10^{20} | ~10^{10} |
| 4096 | ~10^{40} | ~10^{11} |

**Current estimates (2020s):**
- Fault-tolerant Shor requires ~20 million physical qubits for RSA-2048
- With error correction overhead
- Projected: 2030-2040 for practical threat

### Optimizations

#### 1. Approximate QFT
- Drop controlled-phases below threshold angle
- Reduces QFT to $O(n \log n)$ gates
- Negligible error impact

#### 2. Windowed Arithmetic
- Compute several bits simultaneously
- Reduces depth at cost of more qubits

#### 3. Quantum Carry-Lookahead
- Parallelize addition operations
- Reduces depth of arithmetic circuits

#### 4. Semi-Classical Shor
- Measure ancilla qubits sequentially
- Trade space for time
- Only need $O(1)$ ancilla qubits + classical feedback

---

## Worked Examples

### Example 1: Gate Count for N = 15

$N = 15$, so $n = 4$ bits.

**Ancilla register:** $2n = 8$ qubits

**QFT gates:**
- Hadamards: 8
- Controlled-phase: $8 \times 7 / 2 = 28$
- Total QFT: ~36 gates

**Modular exponentiation (simplified):**
- 8 controlled multiplications
- Each: ~$4^2 = 16$ gates (optimistic)
- Total: ~128 gates

**Approximate total:** ~200 elementary gates

### Example 2: Resource Estimate for RSA-2048

$n = 2048$ bits

**Qubits:**
- Ancilla: $2 \times 2048 = 4096$
- Work: 2048
- Scratch: ~4096
- **Total:** ~10,000 logical qubits

**Gate count:**
- $O(n^3) = O(2048^3) \approx 8.6 \times 10^9$ gates

**With error correction:**
- Code distance ~27 for reasonable error rates
- Physical qubits: ~$10^4 \times 27^2 \approx 7 \times 10^6$
- More pessimistic: 20 million physical qubits

### Example 3: Time Estimate

Assuming:
- Gate time: 100 ns (superconducting)
- Parallelization factor: 100

**Sequential time:**
$$T = \frac{10^{10} \text{ gates}}{10^7 \text{ gates/s}} = 10^3 \text{ s} \approx 17 \text{ min}$$

**With error correction overhead:** Hours to days

---

## Practice Problems

### Problem 1: Gate Counting
Calculate the exact gate count for Shor's algorithm on $N = 21$ using:
(a) Standard modular multiplication
(b) Approximate QFT (drop phases $< \pi/64$)

### Problem 2: Depth Analysis
For an $n$-bit number, determine the circuit depth if:
(a) All operations are sequential
(b) Independent operations are parallelized

### Problem 3: Space-Time Tradeoff
Compare the resources needed for:
(a) Standard Shor with $2n$ ancillas
(b) Semi-classical Shor with 1 ancilla and classical feedback

### Problem 4: Error Analysis
If each gate has error probability $p = 10^{-4}$, estimate the total error probability for factoring a 256-bit number. How much error correction is needed?

---

## Computational Lab

```python
"""
Day 615: Complexity Analysis of Shor's Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def gate_count_analysis(n: int) -> dict:
    """
    Analyze gate count for factoring n-bit number.

    Args:
        n: Bit length of number to factor

    Returns:
        Dictionary of gate counts by component
    """
    # Hadamards on ancilla
    h_gates = 2 * n

    # Standard QFT gates
    qft_h = 2 * n
    qft_cphase = (2*n) * (2*n - 1) // 2
    qft_swap = n
    qft_total = qft_h + qft_cphase + qft_swap

    # Approximate QFT (drop phases < pi/n)
    # Only keep phases between qubits at most log(n) apart
    approx_qft_cphase = (2*n) * min(2*n - 1, int(np.log2(n)) + 1) // 2
    approx_qft_total = qft_h + approx_qft_cphase + qft_swap

    # Modular exponentiation (schoolbook)
    # Each controlled multiplication: O(n^2)
    gates_per_mult = n * n
    num_ctrl_mults = 2 * n
    mod_exp_total = num_ctrl_mults * gates_per_mult

    # FFT-based multiplication
    fft_mult = n * np.log2(n) * np.log2(np.log2(max(n, 2)))
    mod_exp_fft = num_ctrl_mults * fft_mult

    return {
        'n': n,
        'hadamards': h_gates,
        'qft_standard': qft_total,
        'qft_approximate': approx_qft_total,
        'mod_exp_schoolbook': mod_exp_total,
        'mod_exp_fft': mod_exp_fft,
        'total_standard': h_gates + qft_total + mod_exp_total,
        'total_optimized': h_gates + approx_qft_total + mod_exp_fft
    }

def qubit_count_analysis(n: int, method: str = 'standard') -> dict:
    """
    Analyze qubit count for different implementations.
    """
    if method == 'standard':
        ancilla = 2 * n
        work = n
        scratch = 2 * n  # For addition/multiplication
        total = ancilla + work + scratch
    elif method == 'beauregard':
        ancilla = 2 * n
        work = 1
        scratch = 2
        total = ancilla + work + scratch
    elif method == 'semiclassical':
        ancilla = 1
        work = n
        scratch = 2 * n
        total = ancilla + work + scratch

    return {
        'method': method,
        'ancilla': ancilla,
        'work': work,
        'scratch': scratch,
        'total': total
    }

def error_correction_overhead(logical_qubits: int,
                              gate_count: int,
                              physical_error_rate: float = 1e-3,
                              target_logical_error: float = 1e-10) -> dict:
    """
    Estimate error correction overhead.
    """
    # Using surface code estimates

    # Required code distance (rough estimate)
    # Logical error rate ~ (p/p_th)^d for p < p_th
    p_th = 0.01  # Threshold
    if physical_error_rate >= p_th:
        return {'error': 'Physical error rate above threshold'}

    # Solve for d: target = gate_count * (p/p_th)^d
    log_ratio = np.log(physical_error_rate / p_th)
    required_suppression = target_logical_error / gate_count
    d = np.ceil(np.log(required_suppression) / log_ratio)
    d = max(d, 3)  # Minimum distance

    # Physical qubits per logical: 2d^2 for surface code
    physical_per_logical = 2 * d * d

    total_physical = logical_qubits * physical_per_logical

    return {
        'logical_qubits': logical_qubits,
        'code_distance': int(d),
        'physical_per_logical': int(physical_per_logical),
        'total_physical_qubits': int(total_physical),
        'physical_error_rate': physical_error_rate,
        'target_logical_error': target_logical_error
    }

def classical_complexity(n: int) -> dict:
    """
    Estimate classical factoring complexity for n-bit number.
    """
    # General Number Field Sieve
    # L(n) = exp(c * n^{1/3} * (log n)^{2/3})
    c = 1.9  # Approximate constant

    log_complexity = c * (n ** (1/3)) * (np.log(n) ** (2/3))
    gnfs_ops = np.exp(log_complexity)

    # Trial division
    trial_ops = 2 ** (n / 2)

    # Pollard rho
    pollard_ops = 2 ** (n / 4)

    return {
        'n': n,
        'gnfs_log_ops': log_complexity,
        'gnfs_ops': gnfs_ops,
        'trial_division_ops': trial_ops,
        'pollard_rho_ops': pollard_ops
    }

# Generate analysis
print("COMPLEXITY ANALYSIS OF SHOR'S ALGORITHM")
print("="*60)

# 1. Gate count scaling
print("\n1. GATE COUNT SCALING")
print("-"*40)

bit_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

print(f"{'Bits':<8} {'Standard':<15} {'Optimized':<15} {'Ratio':<10}")
print("-"*48)

for n in bit_sizes:
    analysis = gate_count_analysis(n)
    ratio = analysis['total_standard'] / max(analysis['total_optimized'], 1)
    print(f"{n:<8} {analysis['total_standard']:<15.2e} {analysis['total_optimized']:<15.2e} {ratio:<10.1f}")

# 2. Qubit requirements
print("\n2. QUBIT REQUIREMENTS")
print("-"*40)

print(f"{'Bits':<8} {'Standard':<12} {'Beauregard':<12} {'Semi-class':<12}")
print("-"*44)

for n in [64, 256, 1024, 2048]:
    std = qubit_count_analysis(n, 'standard')
    bea = qubit_count_analysis(n, 'beauregard')
    semi = qubit_count_analysis(n, 'semiclassical')
    print(f"{n:<8} {std['total']:<12} {bea['total']:<12} {semi['total']:<12}")

# 3. Error correction overhead
print("\n3. ERROR CORRECTION OVERHEAD (RSA-2048)")
print("-"*40)

n = 2048
analysis = gate_count_analysis(n)
logical_qubits = 3 * n + 3  # Rough estimate

ec_overhead = error_correction_overhead(
    logical_qubits=logical_qubits,
    gate_count=int(analysis['total_standard']),
    physical_error_rate=1e-3,
    target_logical_error=1e-10
)

for key, value in ec_overhead.items():
    print(f"  {key}: {value}")

# 4. Quantum vs Classical comparison
print("\n4. QUANTUM VS CLASSICAL COMPARISON")
print("-"*40)

print(f"{'Bits':<8} {'Shor (ops)':<15} {'GNFS (log ops)':<15} {'Speedup':<15}")
print("-"*53)

for n in [256, 512, 1024, 2048]:
    shor_ops = n ** 3
    classical = classical_complexity(n)

    if classical['gnfs_log_ops'] < 100:  # Computable
        speedup = classical['gnfs_ops'] / shor_ops
        print(f"{n:<8} {shor_ops:<15.2e} {classical['gnfs_log_ops']:<15.1f} {speedup:<15.2e}")
    else:
        print(f"{n:<8} {shor_ops:<15.2e} {classical['gnfs_log_ops']:<15.1f} {'Enormous':<15}")

# 5. Visualization
print("\n5. GENERATING COMPLEXITY PLOTS...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Gate count scaling
ax1 = axes[0, 0]
n_vals = np.array([2**k for k in range(2, 12)])
gates_standard = [gate_count_analysis(n)['total_standard'] for n in n_vals]
gates_optimized = [gate_count_analysis(n)['total_optimized'] for n in n_vals]

ax1.loglog(n_vals, gates_standard, 'b-o', label='Standard')
ax1.loglog(n_vals, gates_optimized, 'r-s', label='Optimized')
ax1.loglog(n_vals, n_vals**3, 'k--', alpha=0.5, label='$O(n^3)$')
ax1.set_xlabel('Bit size n')
ax1.set_ylabel('Gate count')
ax1.set_title('Gate Count Scaling')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Qubit scaling
ax2 = axes[0, 1]
qubits_std = [qubit_count_analysis(n, 'standard')['total'] for n in n_vals]
qubits_bea = [qubit_count_analysis(n, 'beauregard')['total'] for n in n_vals]
qubits_semi = [qubit_count_analysis(n, 'semiclassical')['total'] for n in n_vals]

ax2.loglog(n_vals, qubits_std, 'b-o', label='Standard')
ax2.loglog(n_vals, qubits_bea, 'r-s', label='Beauregard')
ax2.loglog(n_vals, qubits_semi, 'g-^', label='Semi-classical')
ax2.set_xlabel('Bit size n')
ax2.set_ylabel('Qubit count')
ax2.set_title('Qubit Requirements')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Quantum vs Classical
ax3 = axes[1, 0]
n_crypto = np.array([128, 256, 512, 768, 1024, 2048])
shor_ops = n_crypto ** 3
gnfs_log = [classical_complexity(n)['gnfs_log_ops'] for n in n_crypto]

ax3.semilogy(n_crypto, shor_ops, 'b-o', label="Shor's Algorithm")
ax3_twin = ax3.twinx()
ax3_twin.plot(n_crypto, gnfs_log, 'r-s', label='GNFS (log scale)')
ax3.set_xlabel('Bit size n')
ax3.set_ylabel('Quantum operations', color='blue')
ax3_twin.set_ylabel('log(Classical operations)', color='red')
ax3.set_title('Quantum vs Classical Factoring')
ax3.grid(True, alpha=0.3)

# Plot 4: Time estimates
ax4 = axes[1, 1]
gate_time_ns = 100  # 100 ns per gate
parallel_factor = 100

n_time = np.array([256, 512, 1024, 2048, 4096])
sequential_time = np.array([gate_count_analysis(n)['total_standard'] for n in n_time]) * gate_time_ns * 1e-9
parallel_time = sequential_time / parallel_factor

ax4.semilogy(n_time, sequential_time, 'b-o', label='Sequential')
ax4.semilogy(n_time, parallel_time, 'r-s', label=f'Parallel (×{parallel_factor})')
ax4.axhline(y=3600, color='k', linestyle='--', alpha=0.5, label='1 hour')
ax4.axhline(y=86400, color='k', linestyle=':', alpha=0.5, label='1 day')
ax4.set_xlabel('Bit size n')
ax4.set_ylabel('Time (seconds)')
ax4.set_title('Estimated Runtime (ignoring error correction)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shor_complexity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to 'shor_complexity_analysis.png'")

# Summary table
print("\n" + "="*60)
print("SUMMARY: RESOURCES FOR BREAKING RSA")
print("="*60)

print("""
| RSA Size | Logical Qubits | Physical Qubits* | Gates      | Est. Time** |
|----------|----------------|------------------|------------|-------------|
| 1024-bit | ~6,000         | ~10 million      | ~10^9      | Hours       |
| 2048-bit | ~12,000        | ~20 million      | ~10^10     | Days        |
| 4096-bit | ~24,000        | ~50 million      | ~10^11     | Weeks       |

* Assuming surface code with ~1000 physical qubits per logical qubit
** Very rough estimates; actual time depends on hardware and error rates
""")
```

---

## Summary

### Key Formulas

| Metric | Complexity |
|--------|------------|
| Gate count | $O(n^3)$ standard, $O(n^2 \log n \log \log n)$ optimized |
| Qubit count | $O(n)$ |
| Circuit depth | $O(n^3)$ |
| Success probability | $\geq 10\%$ per attempt |
| Classical GNFS | $O(\exp(cn^{1/3}(\log n)^{2/3}))$ |

### Key Takeaways

1. **Shor's algorithm** achieves exponential speedup over classical
2. **Modular exponentiation** dominates the gate count
3. **Linear qubit scaling** is remarkably efficient
4. **Error correction** multiplies resource requirements significantly
5. **RSA-2048** requires ~20 million physical qubits with current technology

---

## Daily Checklist

- [ ] I can count gates in Shor's algorithm
- [ ] I understand the dominant cost components
- [ ] I can compare quantum and classical complexity
- [ ] I know the resource requirements for practical sizes
- [ ] I understand the cryptographic timeline implications

---

*Next: Day 616 - Month 22 Review*
