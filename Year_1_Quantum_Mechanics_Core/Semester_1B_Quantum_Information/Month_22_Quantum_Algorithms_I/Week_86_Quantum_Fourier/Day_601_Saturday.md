# Day 601: QFT Applications

## Overview

**Day 601** | Week 86, Day 6 | Month 22 | Quantum Algorithms I

Today we explore practical applications of the Quantum Fourier Transform beyond phase estimation. The QFT enables quantum arithmetic, quantum signal processing, and serves as a key component in many quantum algorithms including quantum simulation and quantum machine learning.

---

## Learning Objectives

1. Apply QFT to quantum addition circuits
2. Understand QFT-based multiplication
3. Explore quantum phase detection applications
4. Connect QFT to quantum simulation
5. Recognize QFT patterns in various algorithms
6. Implement QFT-based arithmetic operations

---

## Core Content

### QFT in Quantum Arithmetic

The QFT transforms between computational basis (position) and Fourier basis (momentum/phase). This duality enables efficient arithmetic operations.

**Key insight:** Addition in the computational basis becomes phase multiplication in the Fourier basis!

### Draper Adder

The **Draper adder** adds two numbers using QFT:

$$|a\rangle|b\rangle \xrightarrow{QFT\text{-add}} |a\rangle|a+b\rangle$$

**Algorithm:**
1. Apply QFT to $|b\rangle$: creates $|\tilde{b}\rangle$
2. Add phases corresponding to $a$
3. Apply QFT^{-1}: gives $|a+b\rangle$

**Why it works:**

In the Fourier basis:
$$|\tilde{b}\rangle = \frac{1}{\sqrt{N}}\sum_k e^{2\pi ibk/N}|k\rangle$$

Adding phase $e^{2\pi iak/N}$:
$$\frac{1}{\sqrt{N}}\sum_k e^{2\pi i(a+b)k/N}|k\rangle = |\widetilde{a+b}\rangle$$

Inverse QFT gives $|a+b\rangle$!

### Phase Addition Circuit

To add the value $a$ (classically known) to a quantum register:

```
|b⟩ ─[QFT]─[R_a]─[QFT^-1]─ |a+b mod N⟩
```

where $R_a$ applies controlled rotations encoding $a$.

For each qubit $j$ of $b$, apply:
$$R_j = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi ia/2^{n-j+1}} \end{pmatrix}$$

### Controlled Addition

For quantum addition (both $a$ and $b$ quantum):

```
|a⟩ ─────────────●──────────────────────
                 │
|b⟩ ─[QFT]──[CR_a]──[QFT^-1]─ |a+b mod N⟩
```

The controlled rotations depend on the bits of $|a\rangle$.

### QFT-based Multiplication

Multiplication can be decomposed into repeated additions:
$$a \times b = \sum_{j=0}^{n-1} a \cdot b_j \cdot 2^j$$

Using QFT adders, this achieves multiplication in $O(n^2)$ operations.

### Modular Arithmetic

For Shor's algorithm, we need modular arithmetic: $a+b \mod N$.

The QFT naturally computes modulo $2^n$. For arbitrary $N$:
1. Compute $a+b$
2. If result $\geq N$, subtract $N$

This requires comparison and conditional subtraction.

### Quantum Phase Detection

The QFT detects periodic structures in quantum states.

**Example:** If a state has periodicity $r$:
$$|\psi\rangle = \frac{1}{\sqrt{r}}\sum_{j=0}^{r-1}|jN/r\rangle$$

QFT transforms this to peaks at multiples of $N/r$.

### QFT in Quantum Simulation

Hamiltonian simulation often uses QFT:

**Kinetic energy:** In position basis, kinetic energy $\hat{T} = \frac{\hat{p}^2}{2m}$ is non-local.

In momentum basis (Fourier transform): $\hat{T}$ becomes diagonal!

**Simulation strategy:**
1. QFT to momentum basis
2. Apply diagonal kinetic energy operator
3. QFT^{-1} to position basis
4. Apply potential energy (diagonal)

This is the **split-operator method**.

### QFT for Solving Linear Systems

The HHL algorithm (quantum linear systems) uses QFT-based phase estimation to:
1. Find eigenvalues of matrix $A$
2. Invert eigenvalues
3. Reconstruct solution

### Approximate Counting

QFT enables **quantum counting** to estimate the number of solutions to a search problem:
1. Run Grover iterations
2. Apply QFT to amplitude
3. Measure to estimate solution count

---

## Worked Examples

### Example 1: Draper Adder for 2-bit Numbers

Add $a = 1$ and $b = 2$ using the Draper adder (3-bit output to avoid overflow).

**Solution:**

$a = 01$ (binary), $b = 010$ (3-bit)

**Step 1: QFT on b-register**
$$|010\rangle \xrightarrow{QFT_8} |\tilde{2}\rangle = \frac{1}{\sqrt{8}}\sum_{k=0}^{7} e^{2\pi i \cdot 2k/8}|k\rangle$$
$$= \frac{1}{\sqrt{8}}\sum_{k=0}^{7} e^{i\pi k/2}|k\rangle$$

**Step 2: Add phase for a = 1**
For each qubit $j$ of the Fourier state, apply phase $e^{2\pi i \cdot 1/2^{3-j+1}}$:
- Qubit 0: phase $e^{2\pi i/8} = e^{i\pi/4}$
- Qubit 1: phase $e^{2\pi i/4} = e^{i\pi/2}$
- Qubit 2: phase $e^{2\pi i/2} = e^{i\pi} = -1$

After phase addition:
$$\frac{1}{\sqrt{8}}\sum_{k=0}^{7} e^{i\pi k/2} \cdot e^{2\pi ik/8}|k\rangle = \frac{1}{\sqrt{8}}\sum_{k=0}^{7} e^{2\pi i \cdot 3k/8}|k\rangle = |\tilde{3}\rangle$$

**Step 3: Inverse QFT**
$$|\tilde{3}\rangle \xrightarrow{QFT^{-1}} |011\rangle = |3\rangle$$

**Result:** $1 + 2 = 3$ ✓

### Example 2: Period Detection

A 3-qubit state has values at positions 0, 2, 4, 6 (period 2):
$$|\psi\rangle = \frac{1}{2}(|000\rangle + |010\rangle + |100\rangle + |110\rangle)$$

Find the period using QFT.

**Solution:**

This state is periodic with period $r = 2$ in an 8-element space.

The positions are $\{0, 2, 4, 6\} = \{0 \cdot 2, 1 \cdot 2, 2 \cdot 2, 3 \cdot 2\}$.

Apply QFT:
$$QFT|\psi\rangle = \frac{1}{2}\sum_{j \in \{0,2,4,6\}} QFT|j\rangle$$

The QFT of a periodic function has peaks at frequencies that are multiples of $N/r = 8/2 = 4$.

Expected peaks at $|0\rangle$ and $|4\rangle$.

**Verification:**
$$\frac{1}{2}(|\tilde{0}\rangle + |\tilde{2}\rangle + |\tilde{4}\rangle + |\tilde{6}\rangle)$$

The sum of these Fourier states constructively interferes only at $k = 0$ and $k = 4$:
$$= \sqrt{\frac{2}{8}}(|0\rangle + |4\rangle) = \frac{1}{2}(|000\rangle + |100\rangle)$$

Measuring gives 0 or 4 with equal probability. The period is $N/\gcd(0,4,8) = 8/4 = 2$. ✓

### Example 3: QFT in Split-Operator Simulation

Simulate free particle evolution for time $\Delta t$.

**Solution:**

**Hamiltonian:** $\hat{H} = \frac{\hat{p}^2}{2m}$

**Evolution operator:** $e^{-i\hat{H}\Delta t/\hbar}$

**Split-operator steps:**
1. Start with $|\psi(x)\rangle$ in position basis
2. Apply QFT: $|\psi(x)\rangle \to |\psi(p)\rangle$
3. Apply diagonal phase: $e^{-ip^2\Delta t/(2m\hbar)}|\psi(p)\rangle$
4. Apply QFT^{-1}: back to position basis

Each momentum state $|k\rangle$ gets phase $e^{-i(\hbar k)^2\Delta t/(2m\hbar)} = e^{-i\hbar k^2\Delta t/(2m)}$.

---

## Practice Problems

### Problem 1: 3-bit Draper Addition

Use the Draper adder to compute $3 + 5 \mod 8$.

### Problem 2: Controlled Phase

Design the controlled rotation sequence to add quantum register $|a\rangle$ to $|b\rangle$ where both are 2 qubits.

### Problem 3: Period Detection

A state is uniform over $\{0, 3, 6, 9, 12, 15\}$ in a 16-element space. What peaks appear after QFT?

### Problem 4: Modular Reduction

How would you modify the Draper adder to compute $a + b \mod 5$ instead of $\mod 8$?

---

## Computational Lab

```python
"""Day 601: QFT Applications"""
import numpy as np
import matplotlib.pyplot as plt

def qft_matrix(n):
    """QFT matrix"""
    N = 2**n
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega**(j*k) for k in range(N)]
                     for j in range(N)]) / np.sqrt(N)

def inverse_qft_matrix(n):
    """Inverse QFT"""
    return qft_matrix(n).conj().T

# ===========================================
# Draper Adder Implementation
# ===========================================

def phase_add(fourier_state, a, n):
    """
    Add classical value 'a' to a state already in Fourier basis
    """
    N = 2**n
    result = fourier_state.copy()

    for k in range(N):
        # In Fourier basis, state |k⟩ gets phase e^{2πi·a·k/N}
        result[k] *= np.exp(2j * np.pi * a * k / N)

    return result

def draper_add_classical(b, a, n):
    """
    Add classical value 'a' to quantum state |b⟩ using Draper adder
    Returns state |a+b mod N⟩
    """
    N = 2**n

    # Create basis state |b⟩
    state = np.zeros(N, dtype=complex)
    state[b % N] = 1

    # Step 1: QFT
    QFT = qft_matrix(n)
    fourier_state = QFT @ state

    # Step 2: Add phases for 'a'
    fourier_state = phase_add(fourier_state, a, n)

    # Step 3: Inverse QFT
    QFT_inv = inverse_qft_matrix(n)
    result = QFT_inv @ fourier_state

    return result

def controlled_phase_add(state_ab, n_a, n_b):
    """
    Controlled addition: |a⟩|b⟩ → |a⟩|a+b mod N⟩
    state_ab has n_a + n_b qubits
    """
    N_a = 2**n_a
    N_b = 2**n_b
    N_total = N_a * N_b

    result = np.zeros(N_total, dtype=complex)

    for a in range(N_a):
        for b in range(N_b):
            idx_in = a * N_b + b
            if abs(state_ab[idx_in]) > 1e-10:
                # Compute a + b mod N_b
                sum_val = (a + b) % N_b
                idx_out = a * N_b + sum_val
                result[idx_out] += state_ab[idx_in]

    return result

# Test Draper adder
print("=" * 60)
print("DRAPER ADDER TEST")
print("=" * 60)

n = 4  # 4-bit arithmetic (mod 16)
test_cases = [
    (3, 5),
    (7, 8),
    (1, 15),
    (10, 10),
]

print(f"\n{n}-bit arithmetic (mod {2**n}):")
print("| a  | b  | a+b mod N | Draper result |")
print("|----|----|-----------:|---------------|")

for a, b in test_cases:
    result_state = draper_add_classical(b, a, n)
    # Find the measured value (highest probability)
    probs = np.abs(result_state)**2
    measured = np.argmax(probs)
    expected = (a + b) % (2**n)
    status = "✓" if measured == expected else "✗"
    print(f"| {a:2d} | {b:2d} | {expected:10d} | {measured:13d} {status}")

# ===========================================
# Period Detection
# ===========================================

print("\n" + "=" * 60)
print("PERIOD DETECTION")
print("=" * 60)

def create_periodic_state(period, n):
    """Create state with given period"""
    N = 2**n
    positions = [i * (N // period) for i in range(period) if i * (N // period) < N]
    state = np.zeros(N, dtype=complex)
    for pos in positions:
        state[pos] = 1
    return state / np.linalg.norm(state)

n = 4
N = 2**n

for period in [2, 4, 8]:
    state = create_periodic_state(period, n)
    positions = [i for i in range(N) if abs(state[i]) > 1e-10]

    # Apply QFT
    QFT = qft_matrix(n)
    fourier_state = QFT @ state
    probs = np.abs(fourier_state)**2

    peaks = [k for k in range(N) if probs[k] > 0.1]

    print(f"\nPeriod r = {period}:")
    print(f"  Non-zero positions: {positions}")
    print(f"  QFT peaks at: {peaks}")
    print(f"  Expected: multiples of N/r = {N}/{period} = {N//period}")

# ===========================================
# QFT for Quantum Simulation
# ===========================================

print("\n" + "=" * 60)
print("QFT IN QUANTUM SIMULATION")
print("=" * 60)

def simulate_free_particle(psi_x, dt, hbar=1, m=1):
    """
    Simulate free particle evolution using QFT
    """
    n = int(np.log2(len(psi_x)))
    N = len(psi_x)

    # QFT to momentum space
    QFT = qft_matrix(n)
    psi_p = QFT @ psi_x

    # Apply kinetic energy phase
    # p = hbar * k, so T = (hbar*k)^2 / (2m)
    # Phase = e^{-i T dt / hbar} = e^{-i hbar k^2 dt / (2m)}
    for k in range(N):
        # Handle negative momenta for k > N/2
        k_shifted = k if k < N//2 else k - N
        phase = np.exp(-1j * hbar * k_shifted**2 * dt / (2*m))
        psi_p[k] *= phase

    # Inverse QFT back to position space
    QFT_inv = inverse_qft_matrix(n)
    psi_x_evolved = QFT_inv @ psi_p

    return psi_x_evolved

# Create Gaussian wave packet
n = 6
N = 2**n
x = np.arange(N)
x0 = N // 4  # Initial position
sigma = 3   # Width
k0 = 2      # Initial momentum

# Gaussian wave packet
psi_0 = np.exp(-(x - x0)**2 / (2*sigma**2)) * np.exp(1j * k0 * x)
psi_0 = psi_0 / np.linalg.norm(psi_0)

# Evolve
dt = 0.5
psi_1 = simulate_free_particle(psi_0, dt)
psi_2 = simulate_free_particle(psi_1, dt)
psi_3 = simulate_free_particle(psi_2, dt)

print(f"\nFree particle simulation (n={n} qubits, N={N} grid points)")
print(f"Initial position x0 = {x0}, momentum k0 = {k0}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

states = [psi_0, psi_1, psi_2, psi_3]
times = [0, dt, 2*dt, 3*dt]

for ax, psi, t in zip(axes.flat, states, times):
    probs = np.abs(psi)**2
    ax.bar(x, probs, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Position')
    ax.set_ylabel('Probability')
    ax.set_title(f't = {t:.1f}')
    ax.set_xlim(0, N)

plt.suptitle('Free Particle Wave Packet Evolution (QFT-based)', fontsize=14)
plt.tight_layout()
plt.savefig('qft_simulation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Simulation saved to 'qft_simulation.png'")

# ===========================================
# Quantum Counting (simplified)
# ===========================================

print("\n" + "=" * 60)
print("QFT FOR QUANTUM COUNTING (CONCEPT)")
print("=" * 60)

print("""
Quantum Counting uses QFT to estimate the number of solutions M
to a search problem with N total items.

1. Grover's algorithm rotates the state by angle θ where sin²(θ/2) = M/N
2. After t iterations, state has rotated by ~2tθ
3. QFT extracts this rotation angle (as a phase)
4. From θ, we can estimate M

For M solutions out of N items:
  θ ≈ 2√(M/N)  (for small M)

QFT precision of n bits gives estimate within ±N/2^n of true M.
""")

# Demonstrate amplitude estimation principle
def amplitude_estimation_demo(M, N, n_ancilla):
    """Demonstrate amplitude estimation principle"""
    # True angle
    theta = 2 * np.arcsin(np.sqrt(M/N))

    # Grover iteration acts as rotation by 2θ
    # QPE extracts eigenvalue e^{±iθ}

    # Simulated measurement: phase kickback gives θ/(2π) in n-bit approximation
    phase = theta / (2 * np.pi)
    k_measured = round(phase * 2**n_ancilla) % 2**n_ancilla
    phase_estimated = k_measured / 2**n_ancilla
    theta_estimated = phase_estimated * 2 * np.pi
    M_estimated = N * np.sin(theta_estimated/2)**2

    return M_estimated, theta, theta_estimated

print("\nAmplitude estimation demo:")
print("| M (true) | N  | n_ancilla | M (estimated) | Error |")
print("|----------|-------|-----------|---------------|-------|")

for M in [10, 50, 100]:
    N = 1000
    for n_a in [6, 8, 10]:
        M_est, theta, theta_est = amplitude_estimation_demo(M, N, n_a)
        error = abs(M - M_est)
        print(f"| {M:8d} | {N:5d} | {n_a:9d} | {M_est:13.1f} | {error:5.1f} |")

print("\n" + "=" * 60)
print("SUMMARY OF QFT APPLICATIONS")
print("=" * 60)

print("""
1. QUANTUM ARITHMETIC (Draper Adder)
   - Addition becomes phase multiplication in Fourier basis
   - Enables efficient quantum addition without carry propagation
   - Used in Shor's algorithm for modular exponentiation

2. PERIOD/FREQUENCY DETECTION
   - QFT maps periodic functions to peaks at harmonic frequencies
   - Central to Shor's algorithm (period finding)
   - Used in quantum signal processing

3. QUANTUM SIMULATION
   - Split-operator method: position ↔ momentum via QFT
   - Kinetic energy diagonal in momentum basis
   - Enables efficient Hamiltonian simulation

4. QUANTUM COUNTING/ESTIMATION
   - Combines Grover with QFT-based phase estimation
   - Estimates solution count in search problems
   - Quadratic speedup over classical counting

5. OTHER APPLICATIONS
   - Quantum linear algebra (HHL algorithm)
   - Quantum machine learning
   - Quantum error correction (stabilizer measurements)
""")
```

---

## Summary

### Key Applications

| Application | QFT Role |
|-------------|----------|
| Draper Adder | Position → Phase addition → Position |
| Period Finding | Detects periodicity in amplitudes |
| Quantum Simulation | Position ↔ Momentum transform |
| Quantum Counting | Extracts rotation angle from Grover |

### Key Takeaways

1. **Addition becomes phase multiplication** in Fourier basis
2. **Period detection** uses QFT's frequency analysis property
3. **Quantum simulation** exploits diagonal representation in Fourier basis
4. **QFT is ubiquitous** in quantum algorithms beyond phase estimation
5. **Modular arithmetic** requires careful handling of overflow

---

## Daily Checklist

- [ ] I understand QFT-based quantum addition
- [ ] I can explain how period detection works
- [ ] I see the connection to quantum simulation
- [ ] I understand the role of QFT in quantum counting
- [ ] I can identify QFT patterns in various algorithms
- [ ] I ran the lab and experimented with QFT applications

---

*Next: Day 602 - Week Review*
