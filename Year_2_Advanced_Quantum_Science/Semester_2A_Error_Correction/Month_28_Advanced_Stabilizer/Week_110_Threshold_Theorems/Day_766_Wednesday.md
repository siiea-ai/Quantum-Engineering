# Day 766: Noise Models & Assumptions

## Overview

**Day:** 766 of 1008
**Week:** 110 (Threshold Theorems & Analysis)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Understanding Noise Channels and Their Impact on Fault-Tolerant Thresholds

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Noise channel theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Threshold dependence |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational simulations |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Derive** the depolarizing channel and its properties
2. **Analyze** the erasure channel and its high threshold
3. **Model** biased noise with Z-dominant errors
4. **Compare** thresholds across different noise models
5. **Identify** how noise assumptions affect threshold theorems
6. **Evaluate** realistic noise in experimental systems

---

## Core Content

### 1. The Depolarizing Channel

The **depolarizing channel** is the most common noise model in quantum error correction. It represents symmetric noise that affects all Pauli directions equally.

#### Mathematical Definition

$$\boxed{\mathcal{E}_{dep}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)}$$

Equivalently, in terms of the identity:
$$\mathcal{E}_{dep}(\rho) = \left(1 - \frac{4p}{3}\right)\rho + \frac{p}{3}(I + X\rho X + Y\rho Y + Z\rho Z)$$

This can be rewritten as:
$$\mathcal{E}_{dep}(\rho) = \left(1 - \frac{4p}{3}\right)\rho + \frac{4p}{3} \cdot \frac{I}{2}$$

#### Kraus Representation

$$\mathcal{E}_{dep}(\rho) = \sum_{i=0}^{3} K_i \rho K_i^\dagger$$

with Kraus operators:
- $K_0 = \sqrt{1-p} \cdot I$
- $K_1 = \sqrt{p/3} \cdot X$
- $K_2 = \sqrt{p/3} \cdot Y$
- $K_3 = \sqrt{p/3} \cdot Z$

#### Properties

| Property | Value |
|----------|-------|
| Error probability | p |
| X error probability | p/3 |
| Y error probability | p/3 |
| Z error probability | p/3 |
| Complete mixing (p=3/4) | $\rho \to I/2$ |

### 2. Two-Qubit Depolarizing Channel

For two-qubit systems:

$$\boxed{\mathcal{E}_{dep}^{(2)}(\rho) = (1-p)\rho + \frac{p}{15}\sum_{P \in \mathcal{P}_2 \setminus I} P\rho P}$$

where $\mathcal{P}_2 = \{I, X, Y, Z\}^{\otimes 2} \setminus \{I \otimes I\}$ contains 15 non-trivial Pauli operators.

#### Correlated vs Independent Errors

**Independent model:**
$$\mathcal{E}^{(2)}_{indep}(\rho) = \mathcal{E}_{dep} \otimes \mathcal{E}_{dep}(\rho)$$

**Two-qubit correlated:**
$$P(IX) = P(XI) = P(XX) = \frac{p}{15}$$

### 3. The Erasure Channel

The **erasure channel** models qubit loss where we know which qubit was lost.

#### Definition

$$\boxed{\mathcal{E}_{erase}(\rho) = (1-p)\rho \otimes |0\rangle\langle 0|_f + p |e\rangle\langle e| \otimes |1\rangle\langle 1|_f}$$

where:
- $|e\rangle$: Erasure state (known lost qubit)
- Flag qubit indicates erasure occurrence

#### Why Erasure Has High Threshold

**Key insight:** We know WHERE the error is!

For an [[n,k,d]] code:
- **Depolarizing:** Can correct $\lfloor(d-1)/2\rfloor$ errors
- **Erasure:** Can correct $d-1$ erasures

**Surface code thresholds:**

| Noise Type | Threshold |
|------------|-----------|
| Depolarizing | ~1% |
| Erasure | ~50% |

The factor of ~50x improvement comes from known error locations!

### 4. Biased Noise Models

Real physical qubits often have **asymmetric** noise, with one Pauli error dominating.

#### Z-Biased Noise

$$\boxed{\mathcal{E}_{biased}(\rho) = (1-p)\rho + p_Z Z\rho Z + p_X X\rho X + p_Y Y\rho Y}$$

with bias ratio:
$$\eta = \frac{p_Z}{p_X + p_Y}$$

Common biases:
- Superconducting qubits: $\eta \sim 1-10$
- Cat qubits: $\eta \sim 10^2 - 10^4$
- Kerr cats: Exponential bias $\eta \sim e^{2|\alpha|^2}$

#### Threshold Enhancement

Biased noise can significantly improve thresholds:

$$\boxed{p_{th}^{biased} \approx p_{th}^{unbiased} \cdot f(\eta)}$$

where $f(\eta)$ is an enhancement factor that grows with bias.

For surface codes:
$$p_{th}(\eta \to \infty) \approx 50\% \text{ (for pure Z errors)}$$

### 5. Pauli vs Non-Pauli Noise

#### Pauli Channels

$$\mathcal{E}_{Pauli}(\rho) = \sum_{P \in \mathcal{P}_n} p_P \cdot P\rho P$$

**Properties:**
- Diagonal in Pauli basis
- Commute with Pauli measurements
- Threshold analysis straightforward

#### Coherent Errors

$$U_{coh} = e^{-i\theta Z} = \cos\theta \cdot I - i\sin\theta \cdot Z$$

**Coherent overrotation:**
$$\rho \to U_{coh} \rho U_{coh}^\dagger$$

**Key difference from Pauli:**
- Errors can accumulate coherently
- May not be detectable by stabilizers
- Can be worse than depolarizing at same rate

#### Conversion to Pauli

Under repeated measurements (syndrome extraction):
$$\boxed{\text{Coherent errors} \xrightarrow{\text{twirling}} \text{Pauli errors}}$$

Effective Pauli rate: $p_{eff} \approx \sin^2\theta$

### 6. Threshold Dependence on Noise Model

The threshold theorem holds for different noise models with different threshold values:

$$\boxed{p_{th} = p_{th}(\mathcal{E}, \mathcal{C}, \mathcal{D})}$$

depends on:
- $\mathcal{E}$: Error model
- $\mathcal{C}$: Code family
- $\mathcal{D}$: Decoder

#### Threshold Comparison

| Noise Model | Surface Code | Concatenated [[7,1,3]] |
|-------------|--------------|------------------------|
| Depolarizing | ~1% | ~0.01% |
| Erasure | ~50% | ~50% |
| Z-biased ($\eta=100$) | ~10% | ~0.1% |
| Amplitude damping | ~0.5% | ~0.005% |

---

## Worked Examples

### Example 1: Depolarizing Channel Fidelity

**Problem:** Compute the fidelity of the depolarizing channel with parameter p applied to state $|\psi\rangle = |+\rangle$.

**Solution:**

Initial state:
$$\rho_0 = |+\rangle\langle +| = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

Apply depolarizing channel:
$$\rho_1 = (1-p)\rho_0 + \frac{p}{3}(X\rho_0 X + Y\rho_0 Y + Z\rho_0 Z)$$

Compute each term:
- $X\rho_0 X = |+\rangle\langle +|$
- $Y\rho_0 Y = |+\rangle\langle +|$
- $Z\rho_0 Z = |-\rangle\langle -|$

Therefore:
$$\rho_1 = (1-p)|+\rangle\langle +| + \frac{p}{3}(2|+\rangle\langle +| + |-\rangle\langle -|)$$
$$= \left(1 - \frac{p}{3}\right)|+\rangle\langle +| + \frac{p}{3}|-\rangle\langle -|$$

Fidelity:
$$F = \langle +|\rho_1|+\rangle = 1 - \frac{p}{3}$$

**Answer:** $F = 1 - p/3$

### Example 2: Biased Noise Syndrome

**Problem:** For a 3-qubit bit-flip code with Z-biased noise ($p_X = p_Y = 0.001$, $p_Z = 0.05$), what fraction of errors are detectable?

**Solution:**

The 3-qubit bit-flip code has stabilizers $Z_1Z_2$ and $Z_2Z_3$.

Detectable errors (anti-commute with at least one stabilizer):
- X errors: $X_1, X_2, X_3$ (all detectable)
- Y errors: $Y_1, Y_2, Y_3$ (all detectable, since $Y = iXZ$)

Undetectable errors (commute with all stabilizers):
- Z errors: All $Z_i, Z_iZ_j, Z_1Z_2Z_3$ commute with stabilizers

Error probabilities:
- Detectable: $3p_X + 3p_Y = 3(0.001) + 3(0.001) = 0.006$
- Undetectable (single Z): $3p_Z = 0.15$

Fraction detectable:
$$f_{det} = \frac{0.006}{0.006 + 0.15} \approx 3.8\%$$

**Answer:** Only ~3.8% of errors are detectable by this code under Z-biased noise.

**Insight:** This code is poorly matched to Z-biased noise. A phase-flip code would be better.

### Example 3: Erasure Threshold Calculation

**Problem:** For a repetition code of length n with erasure probability p, what is the threshold?

**Solution:**

For length-n repetition code:
- Can correct up to $n-1$ erasures (majority voting among surviving qubits)
- Fails when $\geq \lceil (n+1)/2 \rceil$ qubits are erased

For large n, failure probability:
$$P_{fail} = \sum_{k=\lceil(n+1)/2\rceil}^{n} \binom{n}{k} p^k (1-p)^{n-k}$$

At threshold: $P_{fail} = 1/2$

By symmetry, this occurs at:
$$p_{th} = 50\%$$

For any erasure fraction below 50%, majority voting succeeds with probability approaching 1 as $n \to \infty$.

**Answer:** $p_{th} = 50\%$ for erasure noise.

---

## Practice Problems

### Problem Set A: Depolarizing Channel

**A1.** Show that the depolarizing channel with p = 3/4 produces the maximally mixed state regardless of input.

**A2.** Compute the quantum capacity of the depolarizing channel as a function of p.

**A3.** For two independent depolarizing channels with parameters $p_1$ and $p_2$ applied sequentially, what is the effective single parameter $p_{eff}$?

### Problem Set B: Biased Noise

**B1.** Design an optimal code for purely Z-biased noise (only Z errors). What is its threshold?

**B2.** For cat qubit bias $\eta = 100$, compute the effective threshold improvement factor for the surface code.

**B3.** Show that the XZZX surface code has better performance under biased noise than the standard CSS surface code.

### Problem Set C: Threshold Analysis

**C1.** Derive the threshold for the [[5,1,3]] code under pure depolarizing noise using the recursion relation.

**C2.** How does correlated noise (spatial correlation length $\xi$) affect the threshold? Derive a scaling relation.

**C3.** For amplitude damping noise $\mathcal{E}_{AD}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger$ where $E_0 = |0\rangle\langle 0| + \sqrt{1-\gamma}|1\rangle\langle 1|$ and $E_1 = \sqrt{\gamma}|0\rangle\langle 1|$, compute the effective Pauli error rate.

---

## Computational Lab

```python
"""
Day 766 Computational Lab: Noise Models & Thresholds
====================================================

Explore different noise models and their impact on thresholds.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Apply single-qubit depolarizing channel.

    E(rho) = (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)
    """
    return ((1 - p) * rho +
            (p / 3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z))


def biased_noise_channel(rho: np.ndarray, p_x: float,
                         p_y: float, p_z: float) -> np.ndarray:
    """
    Apply biased Pauli noise channel.

    E(rho) = (1-p_x-p_y-p_z)*rho + p_x*X*rho*X + p_y*Y*rho*Y + p_z*Z*rho*Z
    """
    p_total = p_x + p_y + p_z
    return ((1 - p_total) * rho +
            p_x * X @ rho @ X +
            p_y * Y @ rho @ Y +
            p_z * Z @ rho @ Z)


def erasure_channel(rho: np.ndarray, p: float) -> Tuple[np.ndarray, bool]:
    """
    Apply erasure channel with flag indicating if erasure occurred.

    Returns (output_state, erased_flag)
    """
    if np.random.random() < p:
        # Erasure occurred - return maximally mixed state
        return np.eye(2) / 2, True
    else:
        return rho, False


def coherent_error(rho: np.ndarray, theta: float) -> np.ndarray:
    """
    Apply coherent Z-rotation error.

    U = exp(-i*theta*Z)
    """
    U = np.array([[np.exp(-1j * theta), 0],
                  [0, np.exp(1j * theta)]], dtype=complex)
    return U @ rho @ U.conj().T


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute fidelity between two density matrices."""
    sqrt_rho = np.linalg.cholesky(rho + 1e-10 * np.eye(len(rho)))
    M = sqrt_rho @ sigma @ sqrt_rho.conj().T
    eigenvalues = np.linalg.eigvalsh(M)
    return np.real(np.sum(np.sqrt(np.maximum(eigenvalues, 0)))) ** 2


def compare_noise_models(p_range: np.ndarray) -> Dict[str, List[float]]:
    """Compare fidelity degradation under different noise models."""

    # Initial state |+>
    psi_plus = np.array([[1], [1]]) / np.sqrt(2)
    rho_0 = psi_plus @ psi_plus.conj().T

    results = {
        'depolarizing': [],
        'z_biased_10': [],
        'z_biased_100': [],
        'coherent': []
    }

    for p in p_range:
        # Depolarizing
        rho_dep = depolarizing_channel(rho_0, p)
        results['depolarizing'].append(fidelity(rho_0, rho_dep))

        # Z-biased with eta=10
        eta = 10
        p_z = p * eta / (eta + 2)
        p_x = p_y = p / (eta + 2)
        rho_biased = biased_noise_channel(rho_0, p_x, p_y, p_z)
        results['z_biased_10'].append(fidelity(rho_0, rho_biased))

        # Z-biased with eta=100
        eta = 100
        p_z = p * eta / (eta + 2)
        p_x = p_y = p / (eta + 2)
        rho_biased = biased_noise_channel(rho_0, p_x, p_y, p_z)
        results['z_biased_100'].append(fidelity(rho_0, rho_biased))

        # Coherent error (theta such that sin^2(theta) = p)
        theta = np.arcsin(np.sqrt(min(p, 1)))
        rho_coh = coherent_error(rho_0, theta)
        results['coherent'].append(fidelity(rho_0, rho_coh))

    return results


def simulate_repetition_code(n: int, p: float,
                             noise_type: str = 'bit_flip',
                             trials: int = 10000) -> float:
    """
    Simulate n-qubit repetition code under different noise.

    Returns logical error rate.
    """
    errors = 0

    for _ in range(trials):
        # Initialize in |0...0>
        if noise_type == 'bit_flip':
            # Apply independent bit flips
            flipped = np.random.random(n) < p
            # Majority vote
            if np.sum(flipped) > n // 2:
                errors += 1

        elif noise_type == 'erasure':
            # Erasure channel - know which qubits are lost
            erased = np.random.random(n) < p
            # If more than n/2 erased, fail
            if np.sum(erased) >= (n + 1) // 2:
                errors += 1

        elif noise_type == 'phase_flip':
            # Phase flip on |+...+> encoded state
            flipped = np.random.random(n) < p
            # Majority vote
            if np.sum(flipped) > n // 2:
                errors += 1

    return errors / trials


def threshold_estimation(code_distance: int,
                        noise_type: str = 'depolarizing') -> float:
    """
    Estimate threshold by finding crossover point.

    Uses simple model: p_L = c * p^((d+1)/2)
    Threshold where p_L = p
    """
    if noise_type == 'depolarizing':
        # Surface code-like threshold
        c = 0.1  # Typical constant
        d = code_distance
        # p_th where c * p^((d+1)/2) = p
        # c * p^((d-1)/2) = 1
        # p = (1/c)^(2/(d-1))
        return (1/c) ** (2/(d-1))

    elif noise_type == 'erasure':
        # Much higher threshold for erasure
        return 0.5 * (1 - 1/code_distance)

    elif noise_type == 'biased':
        # Biased noise gives higher threshold
        return 0.03 * np.sqrt(code_distance)

    return 0.01


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 766: NOISE MODELS & ASSUMPTIONS")
    print("=" * 70)

    # Demo 1: Depolarizing channel properties
    print("\n" + "=" * 70)
    print("Demo 1: Depolarizing Channel Properties")
    print("=" * 70)

    rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|

    print("\nInput state: |0><0|")
    print("\nDepolarizing channel output for various p:")
    print(f"{'p':<10} {'rho_00':<15} {'rho_11':<15} {'Purity':<15}")
    print("-" * 55)

    for p in [0, 0.25, 0.5, 0.75, 1.0]:
        rho_out = depolarizing_channel(rho_0, p)
        purity = np.real(np.trace(rho_out @ rho_out))
        print(f"{p:<10.2f} {np.real(rho_out[0,0]):<15.4f} "
              f"{np.real(rho_out[1,1]):<15.4f} {purity:<15.4f}")

    print("\nNote: At p = 3/4, state becomes maximally mixed (purity = 0.5)")

    # Demo 2: Noise model comparison
    print("\n" + "=" * 70)
    print("Demo 2: Fidelity Under Different Noise Models")
    print("=" * 70)

    p_range = np.linspace(0, 0.3, 7)
    results = compare_noise_models(p_range)

    print(f"\nFidelity of |+> state after noise:")
    print(f"{'p':<8} {'Depol':<12} {'Z-bias(10)':<12} "
          f"{'Z-bias(100)':<12} {'Coherent':<12}")
    print("-" * 60)

    for i, p in enumerate(p_range):
        print(f"{p:<8.3f} {results['depolarizing'][i]:<12.4f} "
              f"{results['z_biased_10'][i]:<12.4f} "
              f"{results['z_biased_100'][i]:<12.4f} "
              f"{results['coherent'][i]:<12.4f}")

    # Demo 3: Repetition code under different noise
    print("\n" + "=" * 70)
    print("Demo 3: Repetition Code Performance")
    print("=" * 70)

    print("\n5-qubit repetition code logical error rate:")
    print(f"{'p_phys':<10} {'Bit-flip':<15} {'Erasure':<15}")
    print("-" * 40)

    for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
        p_L_bf = simulate_repetition_code(5, p, 'bit_flip', 5000)
        p_L_er = simulate_repetition_code(5, p, 'erasure', 5000)
        print(f"{p:<10.2f} {p_L_bf:<15.4f} {p_L_er:<15.4f}")

    print("\nErasure has MUCH higher threshold (~50%) vs bit-flip (~50%)")
    print("But erasure requires knowing WHICH qubit was lost!")

    # Demo 4: Threshold comparison
    print("\n" + "=" * 70)
    print("Demo 4: Threshold Estimates by Noise Type")
    print("=" * 70)

    print("\nEstimated thresholds for distance-d surface code:")
    print(f"{'d':<6} {'Depolarizing':<15} {'Erasure':<15} {'Biased':<15}")
    print("-" * 55)

    for d in [3, 5, 7, 9, 11]:
        th_dep = threshold_estimation(d, 'depolarizing')
        th_era = threshold_estimation(d, 'erasure')
        th_bias = threshold_estimation(d, 'biased')
        print(f"{d:<6} {th_dep:<15.3%} {th_era:<15.3%} {th_bias:<15.3%}")

    # Demo 5: Biased noise analysis
    print("\n" + "=" * 70)
    print("Demo 5: Z-Biased Noise Analysis")
    print("=" * 70)

    print("\nFor total error rate p = 0.01 with bias ratio eta:")
    print(f"{'eta':<10} {'p_X':<12} {'p_Y':<12} {'p_Z':<12}")
    print("-" * 50)

    p_total = 0.01
    for eta in [1, 10, 100, 1000]:
        p_z = p_total * eta / (eta + 2)
        p_x = p_y = p_total / (eta + 2)
        print(f"{eta:<10} {p_x:<12.6f} {p_y:<12.6f} {p_z:<12.6f}")

    print("\nAt high bias, almost all errors are Z errors!")
    print("This allows codes optimized for Z-error correction.")

    # Summary
    print("\n" + "=" * 70)
    print("NOISE MODELS SUMMARY")
    print("=" * 70)

    print("""
    +---------------------------------------------------------------+
    |  NOISE MODEL COMPARISON                                       |
    +---------------------------------------------------------------+
    |                                                               |
    |  DEPOLARIZING CHANNEL:                                        |
    |    E(rho) = (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ)          |
    |    - Symmetric noise                                          |
    |    - Surface code threshold: ~1%                              |
    |                                                               |
    |  ERASURE CHANNEL:                                             |
    |    - Qubit lost but LOCATION KNOWN                           |
    |    - Threshold: ~50% (!!)                                     |
    |    - Key insight: known errors are easy to correct           |
    |                                                               |
    |  BIASED NOISE (Z-dominant):                                   |
    |    eta = p_Z / (p_X + p_Y)                                   |
    |    - Cat qubits: eta ~ 10^2 - 10^4                           |
    |    - Enables tailored codes (XZZX surface code)              |
    |    - Threshold increases with bias                            |
    |                                                               |
    |  COHERENT ERRORS:                                             |
    |    U = exp(-i*theta*Z)                                       |
    |    - Can accumulate constructively                           |
    |    - Syndrome measurement "twirls" to Pauli                  |
    |                                                               |
    +---------------------------------------------------------------+

    KEY INSIGHT: Threshold depends critically on noise model!
                 Same code can have 1% or 50% threshold.
    """)

    print("=" * 70)
    print("Day 766 Complete: Noise Models & Assumptions Mastered")
    print("=" * 70)
```

---

## Summary

### Noise Model Hierarchy

| Model | Description | Typical Threshold |
|-------|-------------|-------------------|
| Depolarizing | Symmetric Pauli noise | ~1% |
| Erasure | Known loss location | ~50% |
| Z-Biased | Dominant dephasing | 1-10% (depends on eta) |
| Coherent | Systematic rotations | <1% (code-dependent) |

### Critical Equations

$$\boxed{\mathcal{E}_{dep}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)}$$

$$\boxed{\eta = \frac{p_Z}{p_X + p_Y} \quad \text{(bias ratio)}}$$

$$\boxed{p_{th}^{erasure} \approx 50\% \quad \text{(for surface code)}}$$

---

## Daily Checklist

- [ ] Derived depolarizing channel properties
- [ ] Understood erasure channel advantage
- [ ] Analyzed biased noise models
- [ ] Computed threshold dependence on noise
- [ ] Compared coherent vs incoherent errors
- [ ] Ran computational simulations

---

## Preview: Day 767

Tomorrow we explore **Topological Code Thresholds**:
- Surface code ~1% threshold derivation
- Random bond Ising model mapping
- Phase transition interpretation
- Numerical threshold estimation
