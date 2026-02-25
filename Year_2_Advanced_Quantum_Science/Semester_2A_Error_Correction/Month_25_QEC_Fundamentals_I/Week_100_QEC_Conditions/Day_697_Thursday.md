# Day 697: Approximate Quantum Error Correction

## Overview

**Week:** 100 (QEC Conditions)
**Day:** Thursday
**Date:** Year 2, Month 25, Day 697
**Topic:** Approximate QEC — Relaxing Perfect Correction
**Hours:** 7 (3.5 theory + 2.5 problems + 1 computational lab)

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| **Morning** | 9:00 AM - 12:30 PM | 3.5 hrs | Approximate Knill-Laflamme, theory |
| **Afternoon** | 2:00 PM - 4:30 PM | 2.5 hrs | Applications and examples |
| **Evening** | 7:00 PM - 8:00 PM | 1 hr | Fidelity analysis implementation |

---

## Prerequisites

From Days 694-696:
- Quantum bounds (Singleton, Hamming)
- Degeneracy in quantum codes
- Exact Knill-Laflamme conditions

---

## Learning Objectives

By the end of this day, you will be able to:

1. **State** the approximate Knill-Laflamme conditions
2. **Quantify** error correction quality using fidelity measures
3. **Explain** advantages of approximate over exact correction
4. **Describe** bosonic codes as approximate QEC examples
5. **Analyze** the trade-offs in approximate error correction
6. **Apply** approximate QEC concepts to practical scenarios

---

## Core Content

### 1. Motivation: Why Approximate?

#### Limitations of Exact QEC

Exact quantum error correction requires:
- Perfect syndrome measurement
- Precise correction operations
- Exact satisfaction of Knill-Laflamme conditions

**Reality:** Physical systems can only approximate these requirements.

#### Benefits of Relaxation

Approximate QEC offers:
1. **Reduced overhead:** Fewer physical resources
2. **Simpler implementation:** Less demanding hardware
3. **Noise tolerance:** Built-in tolerance for imperfections
4. **Near-optimal performance:** Often achieves practical goals

---

### 2. Approximate Knill-Laflamme Conditions

#### Exact Conditions (Review)

For code projection $P$ and error set $\{E_a\}$:

$$P E_a^\dagger E_b P = C_{ab} P$$

where $C_{ab}$ is independent of the encoded state.

#### Approximate Relaxation

**Definition (Approximate QEC):**

A code approximately corrects error set $\{E_a\}$ with precision $\epsilon$ if:

$$\boxed{\|P E_a^\dagger E_b P - C_{ab} P\| \leq \epsilon}$$

for some constants $C_{ab}$ and all $a, b$.

#### Interpretation

- **$\epsilon = 0$:** Exact correction
- **$\epsilon > 0$:** Small deviations allowed
- **Smaller $\epsilon$:** Better approximation

---

### 3. Fidelity Measures

#### Entanglement Fidelity

The quality of error correction is measured by **entanglement fidelity**:

$$\boxed{F_e(\mathcal{R} \circ \mathcal{E}) = \langle \Phi | (\mathcal{R} \circ \mathcal{E} \otimes \mathcal{I}) | \Phi \rangle \langle \Phi |}$$

where:
- $\mathcal{E}$ is the error channel
- $\mathcal{R}$ is the recovery operation
- $|\Phi\rangle$ is a maximally entangled reference state

#### Worst-Case Fidelity

$$F_{worst} = \min_{|\psi\rangle \in \mathcal{C}} \langle \psi | \mathcal{R}(\mathcal{E}(|\psi\rangle\langle\psi|)) | \psi \rangle$$

#### Approximate Correction Criterion

A code $\epsilon$-approximately corrects errors if:

$$\boxed{F_e(\mathcal{R} \circ \mathcal{E}) \geq 1 - \epsilon}$$

for an optimal recovery operation $\mathcal{R}$.

---

### 4. Advantages of Approximate QEC

#### 1. Reduced Physical Requirements

Exact codes require:
- High gate fidelities ($>99.9\%$)
- Many ancilla qubits
- Deep circuits for syndrome extraction

Approximate codes tolerate:
- Moderate gate fidelities ($\sim 99\%$)
- Fewer ancillas
- Shallower circuits

#### 2. Continuous Variable Systems

For bosonic (continuous variable) systems:
- Exact QEC is often impossible
- Approximate codes (GKP, cat, binomial) achieve practical protection
- Trade precision for implementability

#### 3. Near-Threshold Operation

When operating near the error threshold:
- Exact correction may be infeasible
- Approximate correction extends useful operation regime
- Graceful degradation instead of catastrophic failure

---

### 5. Examples of Approximate QEC

#### Bosonic Codes

**Gottesman-Kitaev-Preskill (GKP) Code:**

Encodes a qubit in an oscillator mode using:
$$|0_L\rangle \propto \sum_{n=-\infty}^{\infty} |2n\sqrt{\pi}\rangle_q$$
$$|1_L\rangle \propto \sum_{n=-\infty}^{\infty} |(2n+1)\sqrt{\pi}\rangle_q$$

where $|x\rangle_q$ are position eigenstates.

**Approximate aspects:**
- Ideal states have infinite energy
- Physical states are approximate (finite squeezing)
- Error correction is inherently approximate

**Advantage:** Corrects small displacement errors with single-mode encoding!

#### Cat Codes

**Cat states:**
$$|C_\alpha^\pm\rangle \propto |\alpha\rangle \pm |-\alpha\rangle$$

Protect against photon loss (dominant error in optical systems):
- Photon loss: $a|C_\alpha^+\rangle \propto |C_\alpha^-\rangle$
- Parity measurement detects error
- Approximate because of finite $\alpha$

#### Binomial Codes

Encode in specific photon number superpositions:
$$|0_L\rangle \propto \sum_{k=0}^{N} \binom{N}{k}^{1/2} |k(S+1)\rangle$$

Approximate correction of up to $L$ photon losses.

---

### 6. Mathematical Framework

#### Petz Recovery Map

For approximate QEC, the **Petz recovery map** provides near-optimal recovery:

$$\mathcal{R}_{Petz}(\rho) = \sqrt{\sigma} \mathcal{E}^\dagger\left(\mathcal{E}(\sigma)^{-1/2} \rho \mathcal{E}(\sigma)^{-1/2}\right) \sqrt{\sigma}$$

where $\sigma$ is the average encoded state.

#### Bounds on Recovery Fidelity

**Theorem:** For any error channel $\mathcal{E}$ and code $\mathcal{C}$:

$$1 - F_{optimal} \leq \epsilon_{KL}^2$$

where $\epsilon_{KL}$ measures violation of Knill-Laflamme conditions.

#### Concatenation with Approximate Codes

For concatenated codes with level-$\ell$ error rate $p_\ell$:

**Exact:** $p_{\ell+1} \approx c \cdot p_\ell^2$ (quadratic suppression)

**Approximate:** $p_{\ell+1} \approx c \cdot p_\ell^2 + \epsilon$ (additive penalty)

The additive $\epsilon$ limits concatenation depth but often still allows practical improvements.

---

### 7. Approximate vs Exact: Trade-offs

| Aspect | Exact QEC | Approximate QEC |
|--------|-----------|-----------------|
| Knill-Laflamme | Exactly satisfied | $\epsilon$-approximate |
| Fidelity | $F = 1$ (ideal) | $F \geq 1 - \epsilon$ |
| Resources | Higher | Lower |
| Implementation | Demanding | More practical |
| Concatenation | Unlimited depth | Limited by $\epsilon$ |
| Threshold | Well-defined | Modified threshold |

#### When to Use Approximate QEC

1. **Limited resources:** Can't meet exact requirements
2. **Continuous variables:** GKP, cat codes
3. **Near-threshold:** Extend operation range
4. **Specific noise:** Codes optimized for dominant error

---

## Quantum Mechanics Connection

### The No-Go and Approximate Go

**No-cloning theorem:** Cannot perfectly copy unknown quantum states

**Approximate cloning:** Can create imperfect copies with bounded error

Similarly:

**Exact QEC:** Requires specific algebraic conditions

**Approximate QEC:** Allows small violations, often sufficient for practical use

### Quantum Channels and Fidelity

Approximate QEC connects to:
- **Channel capacity:** How much information survives noise
- **Entanglement distillation:** Purifying noisy entanglement
- **Quantum state discrimination:** Distinguishing error subspaces

---

## Worked Examples

### Example 1: Quantifying Approximation Error

**Problem:** A code has recovery fidelity $F_e = 0.995$. What is the approximation parameter $\epsilon$?

**Solution:**

From the definition: $F_e \geq 1 - \epsilon$

Therefore: $\epsilon \leq 1 - F_e = 1 - 0.995 = 0.005$

The code $\epsilon$-approximately corrects with $\epsilon = 0.005$.

**Interpretation:** Each syndrome cycle introduces at most 0.5% logical error.

---

### Example 2: Concatenation Limit

**Problem:** An approximate code has $\epsilon = 10^{-4}$ additive error per level. After how many concatenation levels does the additive error dominate over error suppression?

**Solution:**

Assume base physical error rate $p = 10^{-2}$.

At level $\ell$:
$$p_\ell \approx c \cdot p_{\ell-1}^2 + \epsilon$$

With $c \approx 1$ for simplicity:

- Level 1: $p_1 \approx 10^{-4} + 10^{-4} = 2 \times 10^{-4}$
- Level 2: $p_2 \approx 4 \times 10^{-8} + 10^{-4} \approx 10^{-4}$
- Level 3: $p_3 \approx 10^{-8} + 10^{-4} \approx 10^{-4}$

**Conclusion:** After level 1, the additive $\epsilon$ dominates. The code saturates at $p \approx \epsilon = 10^{-4}$.

For lower logical error rates, need smaller $\epsilon$ (better approximate code).

---

### Example 3: GKP Code Approximation

**Problem:** An approximate GKP code with finite squeezing $\Delta$ has effective $\epsilon \approx e^{-\pi/\Delta^2}$. What squeezing is needed for $\epsilon = 10^{-3}$?

**Solution:**

Solve: $e^{-\pi/\Delta^2} = 10^{-3}$

$$-\frac{\pi}{\Delta^2} = \ln(10^{-3}) = -3\ln(10) \approx -6.9$$

$$\Delta^2 = \frac{\pi}{6.9} \approx 0.46$$

$$\Delta \approx 0.68$$

**In dB:** Squeezing $= -10\log_{10}(\Delta^2) \approx 3.4$ dB

This is achievable with current optical technology!

---

## Practice Problems

### Level 1: Direct Application

1. **Fidelity Calculation:**
   If $F_e = 0.99$, what is the maximum $\epsilon$ for approximate correction?

2. **Exact vs Approximate:**
   List three physical scenarios where approximate QEC is preferable to exact.

3. **Error Accumulation:**
   After 100 syndrome cycles with $\epsilon = 10^{-4}$ per cycle, what is the total accumulated error (assuming independence)?

### Level 2: Intermediate

4. **Concatenation Analysis:**
   For $p = 0.01$ and $\epsilon = 10^{-5}$, calculate the logical error rate after 3 levels of concatenation.

5. **GKP Analysis:**
   Derive the approximate Knill-Laflamme violation for a GKP code with finite squeezing.

6. **Trade-off Curve:**
   Sketch the trade-off between resources (number of qubits) and approximation error $\epsilon$ for a family of codes.

### Level 3: Challenging

7. **Petz Recovery:**
   Show that the Petz recovery map reduces to standard syndrome-based recovery for exact QEC codes.

8. **Capacity Bound:**
   How does the approximate QEC capacity relate to the quantum channel capacity?

9. **Optimal Recovery:**
   For a given error channel and code, describe an algorithm to find the optimal recovery map numerically.

---

## Computational Lab

### Approximate QEC Analysis

```python
"""
Day 697 Computational Lab: Approximate Quantum Error Correction
Fidelity analysis and approximation trade-offs
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.linalg import expm, sqrtm

def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply depolarizing channel with error rate p."""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    return (1 - p) * rho + (p/3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)


def bit_flip_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply bit-flip channel with error rate p."""
    X = np.array([[0, 1], [1, 0]])
    return (1 - p) * rho + p * (X @ rho @ X)


def calculate_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Calculate fidelity between two density matrices."""
    sqrt_rho1 = sqrtm(rho1)
    product = sqrt_rho1 @ rho2 @ sqrt_rho1
    return np.real(np.trace(sqrtm(product)))**2


def entanglement_fidelity(channel_func, p: float, n_samples: int = 100) -> float:
    """
    Estimate entanglement fidelity of a channel.

    Uses random pure states to estimate worst-case fidelity.
    """
    fidelities = []

    for _ in range(n_samples):
        # Random pure state
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        psi = np.array([np.cos(theta/2), np.exp(1j*phi) * np.sin(theta/2)])
        rho = np.outer(psi, psi.conj())

        # Apply channel
        rho_out = channel_func(rho, p)

        # Calculate fidelity
        f = np.real(psi.conj() @ rho_out @ psi)
        fidelities.append(f)

    return np.min(fidelities)  # Worst-case


def approximate_qec_simulation():
    """Simulate approximate QEC performance."""

    print("=" * 60)
    print("APPROXIMATE QEC SIMULATION")
    print("=" * 60)

    # 1. Fidelity vs error rate
    print("\n1. CHANNEL FIDELITY VS ERROR RATE")
    print("-" * 40)

    error_rates = np.linspace(0, 0.3, 31)

    dep_fidelities = [entanglement_fidelity(depolarizing_channel, p, 50)
                      for p in error_rates]
    bf_fidelities = [entanglement_fidelity(bit_flip_channel, p, 50)
                     for p in error_rates]

    print(f"{'p':>6} | {'Depolarizing F':>15} | {'Bit-flip F':>15}")
    print("-" * 45)
    for i in range(0, len(error_rates), 5):
        print(f"{error_rates[i]:>6.3f} | {dep_fidelities[i]:>15.4f} | {bf_fidelities[i]:>15.4f}")

    return error_rates, dep_fidelities, bf_fidelities


def concatenation_analysis():
    """Analyze concatenation with approximate codes."""

    print("\n" + "=" * 60)
    print("CONCATENATION WITH APPROXIMATE CODES")
    print("=" * 60)

    p_phys = 0.01  # Physical error rate
    epsilon_values = [0, 1e-5, 1e-4, 1e-3]  # Approximation errors
    levels = range(1, 6)

    print("\nLogical error rate vs concatenation level:")
    print(f"Physical error rate: p = {p_phys}")
    print("-" * 60)

    results = {}

    for epsilon in epsilon_values:
        p_logical = [p_phys]  # Level 0

        for L in levels:
            # p_L = c * p_{L-1}^2 + epsilon
            p_new = p_logical[-1]**2 + epsilon
            p_logical.append(min(p_new, 1.0))

        results[epsilon] = p_logical

        print(f"\nε = {epsilon}:")
        for L, p in enumerate(p_logical):
            print(f"  Level {L}: p_L = {p:.2e}")

    return results


def plot_approximate_qec():
    """Visualize approximate QEC trade-offs."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Fidelity vs error rate
    ax1 = axes[0, 0]
    error_rates = np.linspace(0, 0.3, 50)
    dep_fidelities = [1 - p for p in error_rates]  # Simplified model
    approx_fidelities = [1 - p - 0.01 for p in error_rates]  # With epsilon = 0.01

    ax1.plot(error_rates, dep_fidelities, 'b-', label='Ideal recovery', linewidth=2)
    ax1.plot(error_rates, approx_fidelities, 'r--', label='Approximate (ε=0.01)', linewidth=2)
    ax1.axhline(y=0.99, color='green', linestyle=':', label='F = 0.99 threshold')
    ax1.set_xlabel('Physical error rate p')
    ax1.set_ylabel('Recovery fidelity F')
    ax1.set_title('Exact vs Approximate Recovery Fidelity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Concatenation performance
    ax2 = axes[0, 1]
    levels = np.arange(0, 6)
    p_phys = 0.01

    for epsilon, color, style in [(0, 'blue', '-'), (1e-4, 'green', '--'),
                                   (1e-3, 'orange', ':'), (5e-3, 'red', '-.')]:
        p_logical = [p_phys]
        for L in range(5):
            p_new = p_logical[-1]**2 + epsilon
            p_logical.append(min(p_new, 1.0))

        label = f'ε = {epsilon}' if epsilon > 0 else 'Exact (ε = 0)'
        ax2.semilogy(levels, p_logical, color + style, label=label, linewidth=2, marker='o')

    ax2.set_xlabel('Concatenation level')
    ax2.set_ylabel('Logical error rate')
    ax2.set_title('Concatenation with Approximate Codes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Resource trade-off
    ax3 = axes[1, 0]
    epsilon_range = np.logspace(-5, -2, 50)
    # Simplified model: resources ~ 1/epsilon^0.5
    resources = 10 / np.sqrt(epsilon_range)

    ax3.loglog(epsilon_range, resources, 'b-', linewidth=2)
    ax3.set_xlabel('Approximation error ε')
    ax3.set_ylabel('Required resources (relative)')
    ax3.set_title('Resource vs Approximation Trade-off')
    ax3.grid(True, alpha=0.3)

    # Plot 4: GKP squeezing requirement
    ax4 = axes[1, 1]
    squeezing_db = np.linspace(3, 15, 50)
    Delta_sq = 10**(-squeezing_db/10)
    epsilon_gkp = np.exp(-np.pi / Delta_sq)

    ax4.semilogy(squeezing_db, epsilon_gkp, 'g-', linewidth=2)
    ax4.axhline(y=1e-3, color='red', linestyle='--', label='ε = 10⁻³')
    ax4.axhline(y=1e-4, color='orange', linestyle=':', label='ε = 10⁻⁴')
    ax4.set_xlabel('Squeezing (dB)')
    ax4.set_ylabel('GKP approximation error ε')
    ax4.set_title('GKP Code: Squeezing vs Approximation Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('approximate_qec_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved: approximate_qec_analysis.png")


def demonstrate_approximate_benefit():
    """Show when approximate QEC is beneficial."""

    print("\n" + "=" * 60)
    print("WHEN APPROXIMATE QEC WINS")
    print("=" * 60)

    print("""
    SCENARIO 1: Limited squeezing in bosonic systems
    ────────────────────────────────────────────────
    • Exact GKP requires infinite squeezing (impossible)
    • 10 dB squeezing gives ε ≈ 10⁻³ (achievable!)
    • Practical quantum memory with moderate hardware

    SCENARIO 2: High physical error rate
    ────────────────────────────────────
    • Physical p = 0.02, exact threshold is p_th = 0.01
    • Approximate code with ε = 0.005 extends threshold
    • Operating regime: 0.01 < p < 0.015 becomes useful

    SCENARIO 3: Resource constraints
    ────────────────────────────────
    • Budget: 100 physical qubits
    • Exact [[7,1,3]]: 14 logical qubits (7 qubits each)
    • Approximate [[5,1,3]]: 20 logical qubits (5 qubits each)
    • Trade-off: 43% more logical qubits for slight fidelity loss

    SCENARIO 4: Noise-adapted codes
    ───────────────────────────────
    • Dominant error: amplitude damping (photon loss)
    • Cat code: approximate correction for this specific error
    • Much simpler than full [[n,k,d]] stabilizer codes
    """)


if __name__ == "__main__":
    # Run simulations
    approximate_qec_simulation()
    concatenation_analysis()
    demonstrate_approximate_benefit()

    print("\n" + "=" * 60)
    print("Generating visualization...")
    print("=" * 60)
    plot_approximate_qec()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Approximate K-L | $\|PE_a^\dagger E_b P - C_{ab}P\| \leq \epsilon$ |
| Fidelity criterion | $F_e \geq 1 - \epsilon$ |
| Concatenation | $p_{\ell+1} \approx cp_\ell^2 + \epsilon$ |
| GKP approximation | $\epsilon \approx e^{-\pi/\Delta^2}$ |

### Main Takeaways

1. **Approximate QEC** relaxes exact Knill-Laflamme conditions
2. **Fidelity-based** criteria quantify correction quality
3. **Practical advantages:** Reduced resources, simpler implementation
4. **Bosonic codes** naturally require approximate correction
5. **Concatenation limited** by additive $\epsilon$ error

---

## Daily Checklist

- [ ] Can state approximate Knill-Laflamme conditions
- [ ] Understand fidelity measures for QEC
- [ ] Know advantages of approximate correction
- [ ] Can analyze GKP and cat codes as examples
- [ ] Understand concatenation with approximate codes

---

## Preview: Day 698

Tomorrow we study the **Threshold Theorem** — the fundamental result enabling fault-tolerant quantum computation:

- Statement and significance of the theorem
- Concatenated code thresholds
- Surface code thresholds
- Experimental progress toward threshold

The threshold theorem is why quantum error correction makes large-scale quantum computing possible!

---

*"Perfect is the enemy of good — and in quantum error correction, good is often good enough."*
