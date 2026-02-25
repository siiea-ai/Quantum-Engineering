# Day 649: Channel Composition

## Week 93: Channel Representations | Month 24: Quantum Channels & Error Introduction

---

## Schedule Overview

| Session | Time | Topic |
|---------|------|-------|
| **Morning** | 3 hours | Sequential and parallel channel composition |
| **Afternoon** | 2.5 hours | Error accumulation and channel algebra |
| **Evening** | 1.5 hours | Computational lab: composing channels |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Compose** quantum channels sequentially (one after another)
2. **Compute** Kraus operators for composed channels
3. **Construct** parallel (tensor product) channels
4. **Analyze** how noise accumulates under repeated operations
5. **Understand** the semigroup structure of quantum channels
6. **Apply** composition to model realistic quantum circuits

---

## Core Content

### 1. Sequential Channel Composition

When two channels are applied in sequence, the result is another channel.

**Definition:** For channels $\mathcal{E}_1$ and $\mathcal{E}_2$, the **sequential composition** is:
$$(\mathcal{E}_2 \circ \mathcal{E}_1)(\rho) = \mathcal{E}_2(\mathcal{E}_1(\rho))$$

**Order matters!** Generally $\mathcal{E}_2 \circ \mathcal{E}_1 \neq \mathcal{E}_1 \circ \mathcal{E}_2$.

### 2. Kraus Operators for Sequential Composition

**Theorem:** If $\mathcal{E}_1$ has Kraus operators $\{K_k\}_{k=1}^{r_1}$ and $\mathcal{E}_2$ has Kraus operators $\{L_j\}_{j=1}^{r_2}$, then $\mathcal{E}_2 \circ \mathcal{E}_1$ has Kraus operators:

$$\boxed{M_{jk} = L_j K_k}$$

for all pairs $(j, k)$.

**Proof:**
$$(\mathcal{E}_2 \circ \mathcal{E}_1)(\rho) = \mathcal{E}_2\left(\sum_k K_k \rho K_k^\dagger\right) = \sum_j L_j \left(\sum_k K_k \rho K_k^\dagger\right) L_j^\dagger$$
$$= \sum_{j,k} (L_j K_k) \rho (L_j K_k)^\dagger = \sum_{j,k} M_{jk} \rho M_{jk}^\dagger$$

**Kraus rank:** The composed channel has at most $r_1 \cdot r_2$ Kraus operators (but may have fewer if some $M_{jk}$ are linearly dependent).

### 3. Choi Matrix of Composition

For sequential composition:
$$J_{\mathcal{E}_2 \circ \mathcal{E}_1} = d \cdot \text{Tr}_A[(I_B \otimes J_{\mathcal{E}_1}^{T_A})(J_{\mathcal{E}_2} \otimes I_A)]$$

This is called the **link product** of Choi matrices.

**Simpler formula** (via vectorization):
$$\text{vec}(J_{\mathcal{E}_2 \circ \mathcal{E}_1}) = (J_{\mathcal{E}_2} \otimes I) \cdot S \cdot (I \otimes J_{\mathcal{E}_1}) \cdot S^{-1} \cdot \text{vec}(|\Phi^+\rangle\langle\Phi^+|)$$

where $S$ is an appropriate swap/reshape operator.

### 4. Examples of Sequential Composition

#### Example 1: Two Bit-Flip Channels

$\mathcal{E}_1$: bit-flip with probability $p_1$
$\mathcal{E}_2$: bit-flip with probability $p_2$

Composed Kraus operators (4 total):
- $M_{00} = \sqrt{(1-p_1)(1-p_2)} \cdot I$
- $M_{01} = \sqrt{(1-p_1)p_2} \cdot X$
- $M_{10} = \sqrt{p_1(1-p_2)} \cdot X$
- $M_{11} = \sqrt{p_1 p_2} \cdot I$ (since $X \cdot X = I$)

Simplified:
$$(\mathcal{E}_2 \circ \mathcal{E}_1)(\rho) = [(1-p_1)(1-p_2) + p_1 p_2]\rho + [(1-p_1)p_2 + p_1(1-p_2)]X\rho X$$

Effective bit-flip probability:
$$p_{\text{eff}} = p_1 + p_2 - 2p_1 p_2 = p_1(1-p_2) + p_2(1-p_1)$$

#### Example 2: Bit-Flip followed by Phase-Flip

$\mathcal{E}_X$: bit-flip with probability $p$
$\mathcal{E}_Z$: phase-flip with probability $q$

Composed Kraus operators:
- $M_{00} = \sqrt{(1-p)(1-q)} \cdot I$
- $M_{01} = \sqrt{(1-p)q} \cdot Z$
- $M_{10} = \sqrt{p(1-q)} \cdot X$
- $M_{11} = \sqrt{pq} \cdot ZX = i\sqrt{pq} \cdot Y$

This creates a general Pauli channel!

#### Example 3: Repeated Amplitude Damping

Amplitude damping with parameter $\gamma$ applied twice:

First application: $K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$, $K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$

Composed Kraus operators:
$$M_{00} = K_0 K_0 = \begin{pmatrix} 1 & 0 \\ 0 & 1-\gamma \end{pmatrix}$$
$$M_{01} = K_0 K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
$$M_{10} = K_1 K_0 = \begin{pmatrix} 0 & \sqrt{\gamma(1-\gamma)} \\ 0 & 0 \end{pmatrix}$$
$$M_{11} = K_1 K_1 = 0$$

Effective parameter: $\gamma_{\text{eff}} = 2\gamma - \gamma^2 = 1 - (1-\gamma)^2$

After $n$ applications: $\gamma_n = 1 - (1-\gamma)^n$

### 5. Parallel (Tensor Product) Composition

When channels act on different subsystems:

**Definition:** For $\mathcal{E}_A$ acting on system $A$ and $\mathcal{E}_B$ acting on system $B$:
$$(\mathcal{E}_A \otimes \mathcal{E}_B)(\rho_{AB}) = \text{local action on each subsystem}$$

**Kraus operators:** If $\mathcal{E}_A$ has $\{K_k^A\}$ and $\mathcal{E}_B$ has $\{L_j^B\}$:
$$(\mathcal{E}_A \otimes \mathcal{E}_B)(\rho_{AB}) = \sum_{j,k} (K_k^A \otimes L_j^B) \rho_{AB} (K_k^A \otimes L_j^B)^\dagger$$

**Choi matrix:** $J_{\mathcal{E}_A \otimes \mathcal{E}_B} = J_{\mathcal{E}_A} \otimes J_{\mathcal{E}_B}$

### 6. Error Accumulation

**Key insight:** Noise compounds under composition!

For depolarizing channels with small error rate $p$:
$$\mathcal{E}_{\text{dep}}^{(n)}(\rho) \approx (1-np)\rho + \text{noise terms}$$

For large $n$, the state approaches the maximally mixed state.

**Threshold behavior:** If $p < p_{\text{threshold}}$, error correction can keep errors bounded.

### 7. Channel Semigroup Structure

**Quantum channels form a semigroup:**
- **Closure:** Composition of CPTP maps is CPTP
- **Associativity:** $(\mathcal{E}_3 \circ \mathcal{E}_2) \circ \mathcal{E}_1 = \mathcal{E}_3 \circ (\mathcal{E}_2 \circ \mathcal{E}_1)$
- **Identity:** The identity channel $\mathcal{I}(\rho) = \rho$

**Not a group:** Most channels don't have inverses (information is lost).

**Invertible channels:** Only unitary channels are invertible.

### 8. Convex Combinations

The set of quantum channels is **convex**:

If $\mathcal{E}_1$ and $\mathcal{E}_2$ are channels, so is:
$$\mathcal{E} = \lambda \mathcal{E}_1 + (1-\lambda) \mathcal{E}_2, \quad 0 \leq \lambda \leq 1$$

**Interpretation:** Randomly apply $\mathcal{E}_1$ with probability $\lambda$, $\mathcal{E}_2$ with probability $1-\lambda$.

**Kraus operators:** $\{\sqrt{\lambda} K_k^{(1)}\} \cup \{\sqrt{1-\lambda} K_j^{(2)}\}$

### 9. Fixed Points and Invariant States

**Definition:** A state $\rho_*$ is a **fixed point** of channel $\mathcal{E}$ if:
$$\mathcal{E}(\rho_*) = \rho_*$$

**Examples:**
- Identity channel: all states are fixed
- Depolarizing channel: only $I/d$ is fixed
- Amplitude damping: only $|0\rangle\langle 0|$ is fixed

**Repeated application:** Under iteration, states often converge to fixed points.

### 10. Divisibility and Markovianity

**Definition:** A channel $\mathcal{E}$ is **divisible** if it can be written as:
$$\mathcal{E} = \mathcal{E}_2 \circ \mathcal{E}_1$$

for non-trivial CPTP maps $\mathcal{E}_1, \mathcal{E}_2$.

**CP-divisibility:** Related to Markovian dynamics in open quantum systems.

**Indivisible channels:** Cannot be broken into smaller CPTP steps—indicate non-Markovian dynamics.

---

## Quantum Computing Connection

### Modeling Noisy Circuits

A quantum circuit with noisy gates:

$$\rho_{\text{final}} = \mathcal{E}_n \circ \mathcal{U}_n \circ \cdots \circ \mathcal{E}_1 \circ \mathcal{U}_1(\rho_{\text{init}})$$

where $\mathcal{U}_i$ are ideal gates and $\mathcal{E}_i$ are error channels.

### Error Rates in Practice

**Gate error rates** (typical current values):
- Single-qubit gates: $10^{-4}$ to $10^{-3}$
- Two-qubit gates: $10^{-3}$ to $10^{-2}$
- Measurement: $10^{-2}$

**Circuit fidelity:** For depth-$d$ circuit with per-gate error $p$:
$$F \approx (1-p)^{n_{\text{gates}}} \approx e^{-p \cdot n_{\text{gates}}}$$

### Error Threshold

**Fault-tolerant threshold theorem:** If physical error rate $p < p_{\text{threshold}}$, arbitrarily long computations are possible with error correction.

Threshold depends on:
- Error model (depolarizing, correlated, etc.)
- Code used (surface code, etc.)
- Typical values: $p_{\text{threshold}} \approx 0.1\%$ to $1\%$

---

## Worked Examples

### Example 1: Composing Depolarizing Channels

**Problem:** Find the effective error rate when two depolarizing channels with parameters $p_1 = 0.1$ and $p_2 = 0.2$ are applied sequentially.

**Solution:**

Depolarizing channel: $\mathcal{E}_p(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$

For small $p$, depolarizing channels compose approximately as:
$$\mathcal{E}_{p_2} \circ \mathcal{E}_{p_1} \approx \mathcal{E}_{p_{\text{eff}}}$$

where $p_{\text{eff}} = p_1 + p_2 - \frac{4}{3}p_1 p_2$ (to first order in products).

For $p_1 = 0.1$, $p_2 = 0.2$:
$$p_{\text{eff}} \approx 0.1 + 0.2 - \frac{4}{3}(0.02) = 0.3 - 0.027 = 0.273$$

More precisely, the composition is another depolarizing channel with:
$$\boxed{p_{\text{eff}} = p_1 + p_2 - \frac{4}{3}p_1 p_2 = 0.273}$$

---

### Example 2: Bit-Flip Followed by Amplitude Damping

**Problem:** A qubit undergoes bit-flip with $p = 0.1$ followed by amplitude damping with $\gamma = 0.2$. Find the Kraus operators for the composed channel.

**Solution:**

Bit-flip Kraus: $K_0^{BF} = \sqrt{0.9}I$, $K_1^{BF} = \sqrt{0.1}X$

Amplitude damping Kraus:
$K_0^{AD} = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{0.8} \end{pmatrix}$, $K_1^{AD} = \begin{pmatrix} 0 & \sqrt{0.2} \\ 0 & 0 \end{pmatrix}$

Composed Kraus operators $M_{jk} = K_j^{AD} K_k^{BF}$:

$$M_{00} = K_0^{AD} K_0^{BF} = \sqrt{0.9}\begin{pmatrix} 1 & 0 \\ 0 & \sqrt{0.8} \end{pmatrix}$$

$$M_{01} = K_0^{AD} K_1^{BF} = \sqrt{0.1}\begin{pmatrix} 0 & 1 \\ \sqrt{0.8} & 0 \end{pmatrix}$$

$$M_{10} = K_1^{AD} K_0^{BF} = \sqrt{0.9}\begin{pmatrix} 0 & \sqrt{0.2} \\ 0 & 0 \end{pmatrix}$$

$$M_{11} = K_1^{AD} K_1^{BF} = \sqrt{0.1}\begin{pmatrix} \sqrt{0.2} & 0 \\ 0 & 0 \end{pmatrix}$$

These 4 operators define the composed channel.

---

### Example 3: Fixed Point Analysis

**Problem:** Find all fixed points of the bit-flip channel with $p \neq 0, 1$.

**Solution:**

Channel: $\mathcal{E}(\rho) = (1-p)\rho + pX\rho X$

For fixed point: $\mathcal{E}(\rho_*) = \rho_*$

Write $\rho_* = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ with $a + d = 1$, $c = b^*$.

$$X\rho_* X = \begin{pmatrix} d & c \\ b & a \end{pmatrix}$$

Fixed point condition:
$$(1-p)\begin{pmatrix} a & b \\ c & d \end{pmatrix} + p\begin{pmatrix} d & c \\ b & a \end{pmatrix} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$

Diagonal: $(1-p)a + pd = a \Rightarrow p(d-a) = 0 \Rightarrow a = d = 1/2$

Off-diagonal: $(1-p)b + pc = b \Rightarrow p(c-b) = 0$

For $b = c$ (real): any value works
For $b \neq c$: need $p = 0$

**Fixed points:** $\rho_* = \frac{1}{2}\begin{pmatrix} 1 & r \\ r & 1 \end{pmatrix}$ for real $r \in [-1, 1]$

These are all states on the line from $|+\rangle\langle+|$ to $|-\rangle\langle-|$ through $I/2$.

---

## Practice Problems

### Direct Application

1. **Problem 1:** Compute the Kraus operators for two phase-flip channels with $p_1 = 0.1$ and $p_2 = 0.15$ applied sequentially.

2. **Problem 2:** Find the effective bit-flip probability when three identical bit-flip channels (each with $p = 0.05$) are composed.

3. **Problem 3:** Show that the identity channel is a fixed point of any channel composition: $\mathcal{E} \circ \mathcal{I} = \mathcal{I} \circ \mathcal{E} = \mathcal{E}$.

### Intermediate

4. **Problem 4:** Prove that the composition of two unital channels is unital. (A channel is unital if $\mathcal{E}(I) = I$.)

5. **Problem 5:** For the amplitude damping channel, show that $(1-\gamma_1)(1-\gamma_2) = 1 - \gamma_{\text{eff}}$ when two channels are composed.

6. **Problem 6:** Find the Choi matrix of the composition of bit-flip ($p=0.1$) and phase-flip ($q=0.1$) channels.

### Challenging

7. **Problem 7:** Prove that the convex combination of CPTP maps is CPTP.

8. **Problem 8:** Show that if $\mathcal{E}$ has a unique fixed point, then repeated application converges to it: $\lim_{n\to\infty} \mathcal{E}^n(\rho) = \rho_*$.

9. **Problem 9:** Design a channel $\mathcal{E}$ such that $\mathcal{E} \circ \mathcal{E} = \mathcal{I}$ (involutive channel). What constraints does this impose?

---

## Computational Lab

```python
"""
Day 649 Computational Lab: Channel Composition
==============================================
Topics: Sequential composition, parallel channels, error accumulation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Standard operators
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def apply_channel(rho: np.ndarray, kraus_ops: List[np.ndarray]) -> np.ndarray:
    """Apply quantum channel to density matrix."""
    return sum(K @ rho @ K.conj().T for K in kraus_ops)


def compose_kraus(kraus1: List[np.ndarray],
                  kraus2: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compose two channels: E2 ∘ E1 (E1 first, then E2).
    Kraus operators: {L_j K_k} for all j, k.
    """
    composed = []
    for K in kraus1:
        for L in kraus2:
            M = L @ K
            # Only add if non-zero
            if np.linalg.norm(M) > 1e-12:
                composed.append(M)
    return composed


def parallel_kraus(kraus_A: List[np.ndarray],
                   kraus_B: List[np.ndarray]) -> List[np.ndarray]:
    """
    Parallel composition: E_A ⊗ E_B.
    Kraus operators: {K_k^A ⊗ L_j^B}
    """
    parallel = []
    for K_A in kraus_A:
        for K_B in kraus_B:
            parallel.append(np.kron(K_A, K_B))
    return parallel


def verify_cptp(kraus_ops: List[np.ndarray], tol: float = 1e-10) -> bool:
    """Verify trace preservation."""
    d = kraus_ops[0].shape[0]
    sum_kdk = np.zeros((d, d), dtype=complex)
    for K in kraus_ops:
        sum_kdk += K.conj().T @ K
    return np.allclose(sum_kdk, np.eye(d), atol=tol)


def bit_flip_kraus(p: float) -> List[np.ndarray]:
    return [np.sqrt(1-p) * I, np.sqrt(p) * X]


def phase_flip_kraus(p: float) -> List[np.ndarray]:
    return [np.sqrt(1-p) * I, np.sqrt(p) * Z]


def depolarizing_kraus(p: float) -> List[np.ndarray]:
    return [np.sqrt(1 - 3*p/4) * I, np.sqrt(p/4) * X,
            np.sqrt(p/4) * Y, np.sqrt(p/4) * Z]


def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]


# ===== DEMONSTRATIONS =====

print("=" * 70)
print("PART 1: Sequential Composition of Channels")
print("=" * 70)

# Compose two bit-flip channels
p1, p2 = 0.1, 0.2
bf1 = bit_flip_kraus(p1)
bf2 = bit_flip_kraus(p2)

composed_bf = compose_kraus(bf1, bf2)

print(f"\nBit-flip ({p1}) ∘ Bit-flip ({p2}):")
print(f"Number of composed Kraus operators: {len(composed_bf)}")
print(f"CPTP verification: {verify_cptp(composed_bf)}")

# Theoretical effective probability
p_eff_theory = p1 + p2 - 2*p1*p2
print(f"Theoretical effective p: {p_eff_theory:.4f}")

# Verify by comparing channel action
rho_test = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)

# Direct composition
rho_composed = apply_channel(rho_test, composed_bf)

# Effective bit-flip
bf_eff = bit_flip_kraus(p_eff_theory)
rho_eff = apply_channel(rho_test, bf_eff)

print(f"Composed vs effective difference: {np.max(np.abs(rho_composed - rho_eff)):.2e}")


print("\n" + "=" * 70)
print("PART 2: Bit-Flip + Phase-Flip = Pauli Channel")
print("=" * 70)

p_bf, p_pf = 0.1, 0.15
bf = bit_flip_kraus(p_bf)
pf = phase_flip_kraus(p_pf)

# Compose: phase-flip after bit-flip
composed_pauli = compose_kraus(bf, pf)

print(f"\nPhase-flip ({p_pf}) ∘ Bit-flip ({p_bf}):")
print(f"Number of Kraus operators: {len(composed_pauli)}")

# Analyze the Pauli structure
print("\nComposed Kraus operators:")
pauli_names = ['I', 'X', 'Y', 'Z']
paulis = [I, X, Y, Z]

for i, M in enumerate(composed_pauli):
    # Find which Pauli it's proportional to
    for j, P in enumerate(paulis):
        coef = np.trace(M @ P.conj().T) / 2
        if abs(coef) > 1e-10:
            print(f"  M{i} ≈ {abs(coef):.4f} × {pauli_names[j]}")


print("\n" + "=" * 70)
print("PART 3: Error Accumulation Under Repeated Application")
print("=" * 70)

def repeated_channel_application(kraus_ops: List[np.ndarray],
                                  rho_init: np.ndarray,
                                  n_steps: int) -> List[np.ndarray]:
    """Apply channel n times, return list of states."""
    states = [rho_init.copy()]
    rho = rho_init.copy()

    for _ in range(n_steps):
        rho = apply_channel(rho, kraus_ops)
        states.append(rho.copy())

    return states

# Track purity under repeated depolarizing channel
p_dep = 0.05
dep_kraus = depolarizing_kraus(p_dep)
rho_init = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩

n_steps = 100
states = repeated_channel_application(dep_kraus, rho_init, n_steps)

purities = [np.real(np.trace(rho @ rho)) for rho in states]
dist_to_mixed = [np.max(np.abs(rho - I/2)) for rho in states]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(n_steps + 1), purities, 'b-', linewidth=2)
ax1.axhline(y=0.5, color='r', linestyle='--', label='Maximally mixed')
ax1.set_xlabel('Number of applications')
ax1.set_ylabel('Purity Tr(ρ²)')
ax1.set_title(f'Purity Decay (Depolarizing p={p_dep})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.semilogy(range(n_steps + 1), dist_to_mixed, 'g-', linewidth=2)
ax2.set_xlabel('Number of applications')
ax2.set_ylabel('||ρ - I/2||')
ax2.set_title('Distance to Maximally Mixed State')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_accumulation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: error_accumulation.png")


print("\n" + "=" * 70)
print("PART 4: Parallel Channel Composition")
print("=" * 70)

# Two-qubit system: bit-flip on first, phase-flip on second
bf_kraus = bit_flip_kraus(0.1)
pf_kraus = phase_flip_kraus(0.2)

parallel_kraus_ops = parallel_kraus(bf_kraus, pf_kraus)

print(f"\nBit-flip ⊗ Phase-flip:")
print(f"Number of Kraus operators: {len(parallel_kraus_ops)}")
print(f"Operator dimension: {parallel_kraus_ops[0].shape}")
print(f"CPTP verification: {verify_cptp(parallel_kraus_ops)}")

# Test on Bell state
bell_plus = np.array([[0.5, 0, 0, 0.5],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0.5, 0, 0, 0.5]], dtype=complex)

bell_after_noise = apply_channel(bell_plus, parallel_kraus_ops)

print("\nBell state |Φ+⟩ after local noise:")
print(np.array2string(bell_after_noise, precision=4))

# Check entanglement (via partial trace and purity)
def partial_trace_B(rho_AB):
    """Partial trace over second qubit."""
    d = 2
    rho_A = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                rho_A[i, j] += rho_AB[i*d + k, j*d + k]
    return rho_A

rho_A_after = partial_trace_B(bell_after_noise)
purity_A = np.real(np.trace(rho_A_after @ rho_A_after))
print(f"\nReduced state purity (A): {purity_A:.4f}")
print(f"(1 for product state, 0.5 for maximally entangled)")


print("\n" + "=" * 70)
print("PART 5: Finding Fixed Points")
print("=" * 70)

def find_fixed_point(kraus_ops: List[np.ndarray],
                     tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    """Find fixed point by iteration."""
    d = kraus_ops[0].shape[0]
    rho = np.eye(d, dtype=complex) / d  # Start from maximally mixed

    for i in range(max_iter):
        rho_new = apply_channel(rho, kraus_ops)
        if np.max(np.abs(rho_new - rho)) < tol:
            print(f"  Converged in {i+1} iterations")
            return rho_new
        rho = rho_new

    print(f"  Warning: Did not converge in {max_iter} iterations")
    return rho

print("\nFinding fixed points:")

# Depolarizing channel
print("\nDepolarizing (p=0.3):")
fp_dep = find_fixed_point(depolarizing_kraus(0.3))
print("Fixed point:")
print(np.array2string(fp_dep, precision=4))

# Amplitude damping
print("\nAmplitude damping (γ=0.5):")
fp_ad = find_fixed_point(amplitude_damping_kraus(0.5))
print("Fixed point:")
print(np.array2string(fp_ad, precision=4))


print("\n" + "=" * 70)
print("PART 6: Composition Order Matters")
print("=" * 70)

# Show that E2 ∘ E1 ≠ E1 ∘ E2 in general
ad = amplitude_damping_kraus(0.3)
bf = bit_flip_kraus(0.2)

# E_AD ∘ E_BF
ad_then_bf = compose_kraus(bf, ad)

# E_BF ∘ E_AD
bf_then_ad = compose_kraus(ad, bf)

print("\nComparing orders: Amplitude Damping ↔ Bit-Flip")

# Test on |+⟩
rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

rho_ad_bf = apply_channel(rho_plus, ad_then_bf)
rho_bf_ad = apply_channel(rho_plus, bf_then_ad)

print(f"\nInput: |+⟩")
print(f"AD ∘ BF output:\n{np.array2string(rho_ad_bf, precision=4)}")
print(f"BF ∘ AD output:\n{np.array2string(rho_bf_ad, precision=4)}")
print(f"Difference: {np.max(np.abs(rho_ad_bf - rho_bf_ad)):.4f}")


print("\n" + "=" * 70)
print("PART 7: Circuit Fidelity Under Noise")
print("=" * 70)

def simulate_noisy_circuit(n_gates: int, gate_error: float,
                           gate_type: str = 'depolarizing') -> float:
    """
    Simulate a circuit of n gates with noise after each.
    Return fidelity with ideal |0⟩ → |0⟩.
    """
    if gate_type == 'depolarizing':
        noise_kraus = depolarizing_kraus(gate_error)
    else:
        noise_kraus = bit_flip_kraus(gate_error)

    # Identity gates (just noise)
    rho = np.array([[1, 0], [0, 0]], dtype=complex)

    for _ in range(n_gates):
        # Apply noise after each "gate"
        rho = apply_channel(rho, noise_kraus)

    # Fidelity with |0⟩
    return np.real(rho[0, 0])

# Simulate circuits of various depths
depths = range(1, 101)
error_rates = [0.001, 0.005, 0.01, 0.02]

plt.figure(figsize=(10, 6))
for p in error_rates:
    fidelities = [simulate_noisy_circuit(d, p) for d in depths]
    plt.plot(depths, fidelities, linewidth=2, label=f'p={p}')

plt.xlabel('Circuit Depth (number of gates)')
plt.ylabel('Fidelity with |0⟩')
plt.title('Fidelity Decay with Circuit Depth (Depolarizing Noise)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.savefig('circuit_fidelity_decay.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: circuit_fidelity_decay.png")


print("\n" + "=" * 70)
print("PART 8: Convex Combinations of Channels")
print("=" * 70)

def convex_combine_kraus(kraus1: List[np.ndarray],
                         kraus2: List[np.ndarray],
                         lam: float) -> List[np.ndarray]:
    """
    Convex combination: λ E1 + (1-λ) E2
    Kraus: {√λ K_k^(1)} ∪ {√(1-λ) K_j^(2)}
    """
    combined = []
    for K in kraus1:
        combined.append(np.sqrt(lam) * K)
    for K in kraus2:
        combined.append(np.sqrt(1 - lam) * K)
    return combined

# Mix bit-flip and amplitude damping
bf = bit_flip_kraus(0.2)
ad = amplitude_damping_kraus(0.3)

mixed_channel = convex_combine_kraus(bf, ad, 0.5)

print("Convex combination: 0.5 × BitFlip + 0.5 × AmpDamp")
print(f"Number of Kraus operators: {len(mixed_channel)}")
print(f"CPTP verification: {verify_cptp(mixed_channel)}")

# Test on |1⟩
rho_1 = np.array([[0, 0], [0, 1]], dtype=complex)
rho_mixed = apply_channel(rho_1, mixed_channel)

# Compare with applying each separately
rho_bf = apply_channel(rho_1, bf)
rho_ad = apply_channel(rho_1, ad)
rho_expected = 0.5 * rho_bf + 0.5 * rho_ad

print(f"\nConvex combination output on |1⟩:")
print(np.array2string(rho_mixed, precision=4))
print(f"Expected (0.5×BF + 0.5×AD on |1⟩):")
print(np.array2string(rho_expected, precision=4))
print(f"Match: {np.allclose(rho_mixed, rho_expected)}")


print("\n" + "=" * 70)
print("Lab Complete!")
print("=" * 70)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Sequential composition | $M_{jk} = L_j K_k$ |
| Parallel composition | $M_{jk} = K_j^A \otimes L_k^B$ |
| Convex combination | $\{\sqrt{\lambda}K_k\} \cup \{\sqrt{1-\lambda}L_j\}$ |
| Repeated bit-flip | $p_n = \frac{1}{2}[1 - (1-2p)^n]$ |
| Repeated amplitude damping | $\gamma_n = 1 - (1-\gamma)^n$ |

### Main Takeaways

1. **Sequential composition** multiplies Kraus operators: $M_{jk} = L_j K_k$
2. **Order matters**: $\mathcal{E}_2 \circ \mathcal{E}_1 \neq \mathcal{E}_1 \circ \mathcal{E}_2$ in general
3. **Parallel composition** uses tensor products of Kraus operators
4. **Errors accumulate** under repeated channel application
5. **Fixed points** are states invariant under the channel
6. Channels form a **semigroup** (composition is associative, has identity)

---

## Daily Checklist

- [ ] I can compute Kraus operators for composed channels
- [ ] I understand the difference between sequential and parallel composition
- [ ] I can analyze how errors accumulate
- [ ] I can find fixed points of channels
- [ ] I understand the semigroup structure of channels
- [ ] I completed the computational lab exercises
- [ ] I attempted at least 3 practice problems

---

## Preview: Day 650

Tomorrow we introduce **quantum process tomography**:
- How to experimentally characterize an unknown quantum channel
- Informationally complete state preparations and measurements
- Reconstructing Kraus operators or Choi matrices from data

---

*"Error rates compound exponentially with circuit depth—this is why quantum error correction is not optional, but essential for scalable quantum computing."* — John Preskill
