# Day 767: Topological Code Thresholds

## Overview

**Day:** 767 of 1008
**Week:** 110 (Threshold Theorems & Analysis)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Surface Code Thresholds and Statistical Mechanics Mappings

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Surface code threshold theory |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Random bond Ising model |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Numerical threshold estimation |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Derive** the surface code threshold from statistical mechanics
2. **Explain** the random bond Ising model (RBIM) mapping
3. **Interpret** the threshold as a phase transition
4. **Compute** numerical threshold estimates
5. **Compare** thresholds across topological code variants
6. **Analyze** how decoder choice affects threshold

---

## Core Content

### 1. Surface Code Error Correction Review

The surface code on an $L \times L$ lattice:
- **Data qubits:** $2L^2 - 2L + 1 \approx 2L^2$ on edges
- **X-stabilizers:** Face operators $A_f = \prod_{e \in f} X_e$
- **Z-stabilizers:** Vertex operators $B_v = \prod_{e \in v} Z_e$
- **Distance:** $d = L$

#### Error Model

For independent depolarizing noise:
$$P(E) = p^{|E|}(1-p)^{n-|E|}$$

where $|E|$ is the weight of error $E$.

#### Decoding

Given syndrome $\sigma$, find most likely error class:
$$\hat{E} = \arg\max_{E: \sigma(E) = \sigma} P(E)$$

### 2. Mapping to Random Bond Ising Model

The key insight connecting surface codes to statistical mechanics.

#### The Ising Model

Classical 2D Ising model:
$$\boxed{H_{Ising} = -J\sum_{\langle ij \rangle} s_i s_j, \quad s_i = \pm 1}$$

**Random Bond Ising Model (RBIM):**
$$H_{RBIM} = -\sum_{\langle ij \rangle} J_{ij} s_i s_j$$

where $J_{ij}$ are random couplings:
$$J_{ij} = \begin{cases} +J & \text{prob } 1-p \\ -J & \text{prob } p \end{cases}$$

#### The Mapping

Consider Z errors on surface code:

**Step 1: Error configuration**
- Z errors form a configuration $E \subset \text{edges}$
- Syndrome: vertices with odd number of adjacent errors

**Step 2: Ising spins**
- Place Ising spin $s_v = \pm 1$ at each vertex
- $s_v = +1$ if no error string terminates at $v$
- $s_v = -1$ if error string terminates at $v$

**Step 3: Bond assignment**
$$J_e = \begin{cases} +J & \text{if no error on edge } e \\ -J & \text{if error on edge } e \end{cases}$$

**Key result:**
$$\boxed{\text{Surface code partition function} = \text{RBIM partition function}}$$

### 3. The Nishimori Line

The mapping is exact along a special line in parameter space.

#### Nishimori Condition

$$\boxed{e^{-2\beta J} = \frac{p}{1-p} \quad \Leftrightarrow \quad \tanh(\beta J) = 1 - 2p}$$

This relates:
- Physical error probability $p$
- Ising temperature $T = 1/(\beta k_B)$
- Coupling strength $J$

#### Phase Diagram

The RBIM has three phases:
1. **Ferromagnetic (ordered):** Spins aligned, errors correctable
2. **Paramagnetic (disordered):** Random spins, errors uncorrectable
3. **Spin glass:** Complex behavior

Along Nishimori line:
- **Below threshold:** Ferromagnetic phase
- **Above threshold:** Paramagnetic phase
- **At threshold:** Phase transition

### 4. Threshold as Phase Transition

The error correction threshold corresponds to a **phase transition** in the RBIM.

#### Order Parameter

Define the string tension:
$$\boxed{\tau = -\lim_{L \to \infty} \frac{1}{L} \ln P(\text{no logical error})}$$

- $\tau > 0$: Ordered phase (correctable)
- $\tau = 0$: Disordered phase (uncorrectable)

#### Critical Point

At the Nishimori point:
$$\boxed{p_c \approx 10.9\% \quad \text{(2D RBIM on square lattice)}}$$

This gives the **optimal decoder threshold** for the surface code!

#### Finite-Size Scaling

Near threshold:
$$P_L(\text{logical error}) \sim f\left((p - p_c) L^{1/\nu}\right)$$

where $\nu \approx 1.5$ is the correlation length exponent.

### 5. Decoder-Dependent Thresholds

The mapping gives the **optimal** threshold. Practical decoders achieve lower thresholds.

#### Comparison

| Decoder | Threshold | Notes |
|---------|-----------|-------|
| Optimal (RBIM bound) | ~10.9% | Computationally intractable |
| Maximum likelihood | ~10.9% | Still intractable for large L |
| Minimum Weight Perfect Matching | ~10.3% | Practical, polynomial time |
| Union-Find | ~9.9% | Near-linear time |
| Neural Network | ~10.5% | Requires training |

#### Why MWPM Works Well

MWPM finds minimum weight error consistent with syndrome.
- Approximates maximum likelihood for low p
- Fails to account for error degeneracy
- Gap to optimal shrinks for low error rates

### 6. Threshold for Different Noise Models

The surface code threshold varies significantly with noise type.

#### Depolarizing Noise

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

Threshold: $p_{th} \approx 15.2\%$ (per-qubit) or $\approx 1\%$ (per-gate, circuit level)

#### Independent X and Z

If X and Z errors independent with rates $p_X$, $p_Z$:
$$\boxed{p_X + p_Z \leq p_{th} \approx 10.9\%}$$

#### Biased Noise

For Z-biased noise with bias $\eta = p_Z/p_X$:
$$p_{th}(Z) \approx 50\% \text{ as } \eta \to \infty$$

Using XZZX variant surface code optimizes for biased noise.

#### Erasure

$$\boxed{p_{th}^{erasure} \approx 50\%}$$

Erasure threshold is the percolation threshold on the lattice!

---

## Worked Examples

### Example 1: Nishimori Temperature

**Problem:** At physical error rate $p = 0.05$, what is the Nishimori temperature?

**Solution:**

Nishimori condition:
$$\tanh(\beta J) = 1 - 2p = 1 - 2(0.05) = 0.9$$

Therefore:
$$\beta J = \text{arctanh}(0.9) = \frac{1}{2}\ln\frac{1.9}{0.1} = \frac{1}{2}\ln(19) \approx 1.47$$

Temperature:
$$T = \frac{1}{k_B \beta} = \frac{J}{1.47 k_B}$$

Setting $J = k_B = 1$:
$$\boxed{T_N = 0.68}$$

**Interpretation:** This is above the critical temperature $T_c \approx 0.95$ of the pure Ising model, but the Nishimori line crosses the phase boundary at higher disorder.

### Example 2: Threshold from Finite-Size Scaling

**Problem:** From simulations, logical error rates at different system sizes are:

| p | L=5 | L=9 | L=13 |
|---|-----|-----|------|
| 0.08 | 0.12 | 0.06 | 0.03 |
| 0.10 | 0.18 | 0.15 | 0.14 |
| 0.12 | 0.25 | 0.28 | 0.32 |

Estimate the threshold.

**Solution:**

At threshold, logical error rate is independent of system size.

Looking at the data:
- $p = 0.08$: Error decreases with L (below threshold)
- $p = 0.12$: Error increases with L (above threshold)
- $p = 0.10$: Error nearly constant (near threshold)

Linear interpolation for crossing point:
$$p_{th} \approx 0.10 + \frac{0.14 - 0.15}{(0.15 - 0.14) + (0.14 - 0.15)} \times 0.02$$

More carefully, look for where curves cross. From the pattern:
$$\boxed{p_{th} \approx 10\%}$$

### Example 3: MWPM vs Optimal Threshold

**Problem:** If optimal threshold is 10.9% and MWPM achieves 10.3%, what is the relative efficiency?

**Solution:**

Define efficiency:
$$\eta = \frac{p_{th}^{MWPM}}{p_{th}^{optimal}} = \frac{10.3\%}{10.9\%} = 94.5\%$$

At $p = 10\%$:
- Optimal decoder: Correctable (below threshold)
- MWPM: Correctable (below threshold)

At $p = 10.5\%$:
- Optimal decoder: Correctable (below 10.9%)
- MWPM: Uncorrectable (above 10.3%)

**Gap interpretation:**
The 0.6% gap comes from MWPM not accounting for:
1. Error degeneracy (multiple error patterns with same syndrome)
2. Higher-order correlations

$$\boxed{\text{Efficiency} \approx 94.5\%}$$

---

## Practice Problems

### Problem Set A: Statistical Mechanics Mapping

**A1.** Derive the partition function equivalence between surface code and RBIM explicitly for a $2 \times 2$ surface code.

**A2.** Show that along the Nishimori line, the RBIM has special gauge symmetry properties.

**A3.** For the 3D surface code (on cubic lattice), what statistical mechanics model does error correction map to?

### Problem Set B: Threshold Analysis

**B1.** Using finite-size scaling with $\nu = 1.5$, collapse the data from Example 2 onto a universal curve.

**B2.** Derive the relationship between the RBIM critical disorder strength and the surface code threshold.

**B3.** For a surface code with phenomenological noise (measurement errors too), how does the threshold change?

### Problem Set C: Decoder Comparison

**C1.** Explain why the Union-Find decoder has lower threshold than MWPM despite being faster.

**C2.** Design a decoding strategy that interpolates between MWPM and optimal decoding.

**C3.** For biased noise with $\eta = 100$, estimate the threshold advantage of the XZZX code over CSS surface code.

---

## Computational Lab

```python
"""
Day 767 Computational Lab: Topological Code Thresholds
======================================================

Simulate surface code error correction and threshold estimation.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import heapq


class SurfaceCode:
    """
    Simple surface code simulator for threshold estimation.
    """

    def __init__(self, L: int):
        """Initialize L x L surface code."""
        self.L = L
        self.n_data = 2 * L * L - 2 * L + 1  # Approximate
        self.n_x_stab = (L - 1) * L  # Face stabilizers
        self.n_z_stab = L * (L - 1)  # Vertex stabilizers

    def generate_errors(self, p: float, error_type: str = 'depolarizing'
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random errors on data qubits.

        Returns (x_errors, z_errors) as boolean arrays.
        """
        n = self.L * self.L  # Simplified qubit count

        if error_type == 'depolarizing':
            # Each qubit has p/3 chance of X, Y, or Z
            rand = np.random.random((n, 3))
            x_errors = (rand[:, 0] < p/3) | (rand[:, 1] < p/3)  # X or Y
            z_errors = (rand[:, 1] < p/3) | (rand[:, 2] < p/3)  # Y or Z
        elif error_type == 'independent':
            x_errors = np.random.random(n) < p
            z_errors = np.random.random(n) < p
        elif error_type == 'z_only':
            x_errors = np.zeros(n, dtype=bool)
            z_errors = np.random.random(n) < p
        else:
            raise ValueError(f"Unknown error type: {error_type}")

        return x_errors, z_errors

    def get_syndrome(self, x_errors: np.ndarray,
                    z_errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute syndrome from errors.

        Returns (x_syndrome, z_syndrome).
        X syndrome detects Z errors, Z syndrome detects X errors.
        """
        L = self.L

        # Z syndrome from X errors (simplified model)
        z_syndrome = np.zeros((L-1, L), dtype=bool)
        for i in range(L-1):
            for j in range(L):
                # Parity of X errors on adjacent edges
                idx1 = i * L + j
                idx2 = (i + 1) * L + j
                if idx1 < len(x_errors) and idx2 < len(x_errors):
                    z_syndrome[i, j] = x_errors[idx1] ^ x_errors[idx2]

        # X syndrome from Z errors
        x_syndrome = np.zeros((L, L-1), dtype=bool)
        for i in range(L):
            for j in range(L-1):
                idx1 = i * L + j
                idx2 = i * L + j + 1
                if idx1 < len(z_errors) and idx2 < len(z_errors):
                    x_syndrome[i, j] = z_errors[idx1] ^ z_errors[idx2]

        return x_syndrome, z_syndrome

    def decode_mwpm_simple(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Simple minimum weight decoder (not full MWPM).

        Pairs syndrome defects greedily by distance.
        """
        # Find syndrome locations
        defects = list(zip(*np.where(syndrome)))

        if len(defects) == 0:
            return np.zeros(self.L * self.L, dtype=bool)

        # Simple greedy pairing
        correction = np.zeros(self.L * self.L, dtype=bool)
        used = set()

        for i, d1 in enumerate(defects):
            if i in used:
                continue

            # Find nearest unmatched defect
            min_dist = float('inf')
            best_j = -1

            for j, d2 in enumerate(defects):
                if j <= i or j in used:
                    continue
                dist = abs(d1[0] - d2[0]) + abs(d1[1] - d2[1])
                if dist < min_dist:
                    min_dist = dist
                    best_j = j

            if best_j >= 0:
                used.add(i)
                used.add(best_j)
                # Add correction along path (simplified)

        return correction

    def check_logical_error(self, x_errors: np.ndarray,
                           z_errors: np.ndarray,
                           x_correction: np.ndarray,
                           z_correction: np.ndarray) -> bool:
        """
        Check if residual error is a logical error.

        Logical X error: Z chain from top to bottom
        Logical Z error: X chain from left to right
        """
        L = self.L

        # Residual errors
        x_residual = x_errors ^ x_correction
        z_residual = z_errors ^ z_correction

        # Check for logical X (horizontal chain)
        # Simplified: count parity along middle row
        logical_x = np.sum(z_residual[:L]) % 2 == 1

        # Check for logical Z (vertical chain)
        logical_z = np.sum(x_residual[::L]) % 2 == 1

        return logical_x or logical_z


def simulate_threshold(L: int, p: float, trials: int = 1000,
                      error_type: str = 'z_only') -> float:
    """
    Estimate logical error rate for given parameters.
    """
    code = SurfaceCode(L)
    logical_errors = 0

    for _ in range(trials):
        # Generate errors
        x_errors, z_errors = code.generate_errors(p, error_type)

        # Get syndrome
        x_synd, z_synd = code.get_syndrome(x_errors, z_errors)

        # Decode (simplified - just check if errors cross)
        # For Z-only errors, logical error if odd parity across code
        if error_type == 'z_only':
            # Logical error if chain spans the code
            # Simplified: check if number of Z errors has odd parity
            # in any row (indicating spanning chain)
            for row in range(L):
                row_errors = z_errors[row*L:(row+1)*L] if (row+1)*L <= len(z_errors) else []
                if len(row_errors) > 0 and np.sum(row_errors) > L // 2:
                    logical_errors += 1
                    break

    return logical_errors / trials


def find_threshold_crossing(L_values: List[int], p_range: np.ndarray,
                           trials: int = 500) -> Tuple[float, Dict]:
    """
    Find threshold by looking for crossing of P_L(error) curves.
    """
    results = {L: [] for L in L_values}

    for L in L_values:
        print(f"  Simulating L = {L}...")
        for p in p_range:
            p_logical = simulate_threshold(L, p, trials, 'z_only')
            results[L].append(p_logical)

    # Find crossing point
    # Look for where smaller L has higher error rate than larger L
    crossings = []
    for i in range(len(p_range) - 1):
        for j in range(len(L_values) - 1):
            L1, L2 = L_values[j], L_values[j+1]
            p1_low = results[L1][i]
            p1_high = results[L1][i+1]
            p2_low = results[L2][i]
            p2_high = results[L2][i+1]

            # Check for crossing
            if (p1_low - p2_low) * (p1_high - p2_high) < 0:
                # Linear interpolation
                dp = p_range[i+1] - p_range[i]
                alpha = (p1_low - p2_low) / ((p1_low - p2_low) - (p1_high - p2_high))
                p_cross = p_range[i] + alpha * dp
                crossings.append(p_cross)

    threshold = np.mean(crossings) if crossings else 0.1

    return threshold, results


def nishimori_temperature(p: float, J: float = 1.0) -> float:
    """Compute Nishimori temperature for given error rate."""
    if p <= 0 or p >= 0.5:
        return float('inf')
    beta_J = np.arctanh(1 - 2*p)
    return J / beta_J


def rbim_partition_function_small(L: int, p: float, beta: float,
                                  samples: int = 100) -> float:
    """
    Estimate RBIM partition function for small system via sampling.
    """
    J = 1.0
    Z_total = 0

    for _ in range(samples):
        # Random bonds
        bonds = np.random.choice([1, -1], size=(L, L, 2),
                                p=[1-p, p])

        # Sum over all spin configurations
        Z_sample = 0
        for config in range(2**L**2):
            spins = np.array([(config >> i) & 1 for i in range(L**2)]) * 2 - 1
            spins = spins.reshape((L, L))

            # Energy
            E = 0
            for i in range(L):
                for j in range(L):
                    if i < L-1:
                        E -= J * bonds[i, j, 0] * spins[i, j] * spins[i+1, j]
                    if j < L-1:
                        E -= J * bonds[i, j, 1] * spins[i, j] * spins[i, j+1]

            Z_sample += np.exp(-beta * E)

        Z_total += Z_sample

    return Z_total / samples


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 767: TOPOLOGICAL CODE THRESHOLDS")
    print("=" * 70)

    # Demo 1: Nishimori line
    print("\n" + "=" * 70)
    print("Demo 1: The Nishimori Line")
    print("=" * 70)

    print("\nNishimori temperature as function of error rate p:")
    print(f"{'p':<10} {'T_N':<15} {'beta*J':<15}")
    print("-" * 40)

    for p in [0.01, 0.05, 0.10, 0.109, 0.15, 0.20]:
        T_N = nishimori_temperature(p)
        beta_J = np.arctanh(1 - 2*p) if p < 0.5 else float('inf')
        print(f"{p:<10.3f} {T_N:<15.4f} {beta_J:<15.4f}")

    print("\nAt p = 10.9% (threshold), Nishimori line crosses phase boundary")

    # Demo 2: Simple threshold estimation
    print("\n" + "=" * 70)
    print("Demo 2: Threshold Estimation via Simulation")
    print("=" * 70)

    print("\nLogical error rate for different system sizes:")
    print(f"{'p':<10} {'L=3':<12} {'L=5':<12} {'L=7':<12}")
    print("-" * 50)

    p_values = np.linspace(0.05, 0.20, 6)
    L_values = [3, 5, 7]

    for p in p_values:
        row = f"{p:<10.3f} "
        for L in L_values:
            p_L = simulate_threshold(L, p, trials=500, error_type='z_only')
            row += f"{p_L:<12.4f} "
        print(row)

    print("\nThreshold: Where curves cross (P_L independent of L)")

    # Demo 3: Decoder comparison
    print("\n" + "=" * 70)
    print("Demo 3: Decoder Threshold Comparison")
    print("=" * 70)

    decoders = [
        ("Optimal (RBIM bound)", 0.109),
        ("Maximum Likelihood", 0.109),
        ("MWPM", 0.103),
        ("Union-Find", 0.099),
        ("Neural Network", 0.105),
        ("Lookup Table (small L)", 0.108),
    ]

    print(f"\n{'Decoder':<30} {'Threshold':<12} {'% of Optimal':<15}")
    print("-" * 60)

    for name, threshold in decoders:
        pct = 100 * threshold / 0.109
        print(f"{name:<30} {threshold:<12.3f} {pct:<15.1f}%")

    # Demo 4: Noise model thresholds
    print("\n" + "=" * 70)
    print("Demo 4: Surface Code Thresholds by Noise Model")
    print("=" * 70)

    noise_thresholds = [
        ("Independent X, Z", "10.9%", "RBIM phase transition"),
        ("Depolarizing (code capacity)", "15.2%", "Higher due to Y errors"),
        ("Circuit-level depolarizing", "~1%", "Includes gate errors"),
        ("Pure Z noise", "10.9%", "Single RBIM"),
        ("Erasure", "50%", "Percolation threshold"),
        ("Z-biased (eta=100)", "~20%", "XZZX code better"),
    ]

    print(f"\n{'Noise Model':<30} {'Threshold':<12} {'Notes':<30}")
    print("-" * 75)

    for model, threshold, notes in noise_thresholds:
        print(f"{model:<30} {threshold:<12} {notes:<30}")

    # Demo 5: Finite-size scaling
    print("\n" + "=" * 70)
    print("Demo 5: Finite-Size Scaling")
    print("=" * 70)

    print("""
    Near threshold, logical error rate follows scaling form:

       P_L(error) = f((p - p_c) * L^(1/nu))

    where nu ~ 1.5 is the correlation length exponent.

    Scaling collapse:
    - Plot P_L vs (p - p_c) * L^(1/nu)
    - At correct p_c and nu, all curves collapse to one
    """)

    # Illustrative scaling data
    print("\nScaling collapse illustration:")
    print(f"{'(p-p_c)*L^0.67':<20} {'P_L':<10}")
    print("-" * 30)

    p_c = 0.109
    nu = 1.5
    for L in [5, 9, 13]:
        for p in [0.08, 0.10, 0.12]:
            x = (p - p_c) * (L ** (1/nu))
            P_L = 0.5 * (1 + np.tanh(5*x))  # Approximate scaling function
            print(f"{x:<20.3f} {P_L:<10.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("TOPOLOGICAL CODE THRESHOLDS SUMMARY")
    print("=" * 70)

    print("""
    +---------------------------------------------------------------+
    |  SURFACE CODE THRESHOLD                                       |
    +---------------------------------------------------------------+
    |                                                               |
    |  STATISTICAL MECHANICS MAPPING:                               |
    |    Surface code  <-->  Random Bond Ising Model (RBIM)        |
    |    Error correction <--> Finding ground state                |
    |    Threshold <--> Phase transition                           |
    |                                                               |
    |  NISHIMORI LINE:                                              |
    |    tanh(beta*J) = 1 - 2p                                     |
    |    Special symmetry along this line                          |
    |    Threshold at Nishimori point: p_c ~ 10.9%                 |
    |                                                               |
    |  KEY THRESHOLDS:                                              |
    |    - Optimal decoder: 10.9% (RBIM critical point)            |
    |    - MWPM decoder: 10.3%                                     |
    |    - Circuit-level: ~1%                                      |
    |    - Erasure: 50%                                            |
    |                                                               |
    |  FINITE-SIZE SCALING:                                         |
    |    P_L ~ f((p - p_c) * L^(1/nu))                             |
    |    nu ~ 1.5 (correlation length exponent)                    |
    |                                                               |
    +---------------------------------------------------------------+
    """)

    print("=" * 70)
    print("Day 767 Complete: Topological Code Thresholds Analyzed")
    print("=" * 70)
```

---

## Summary

### Statistical Mechanics Mapping

| Quantum EC Concept | Statistical Mechanics |
|-------------------|----------------------|
| Error configuration | Bond disorder |
| Syndrome | Domain wall |
| Logical error | Spanning cluster |
| Threshold | Phase transition |

### Critical Equations

$$\boxed{H_{RBIM} = -\sum_{\langle ij \rangle} J_{ij} s_i s_j}$$

$$\boxed{\tanh(\beta J) = 1 - 2p \quad \text{(Nishimori condition)}}$$

$$\boxed{p_c^{RBIM} \approx 10.9\% \quad \text{(square lattice)}}$$

$$\boxed{P_L \sim f\left((p - p_c) L^{1/\nu}\right) \quad \text{(finite-size scaling)}}$$

---

## Daily Checklist

- [ ] Understood the RBIM mapping
- [ ] Derived Nishimori condition
- [ ] Interpreted threshold as phase transition
- [ ] Compared decoder thresholds
- [ ] Analyzed finite-size scaling
- [ ] Ran threshold simulations

---

## Preview: Day 768

Tomorrow we study **Threshold Computation Methods**:
- Monte Carlo simulation techniques
- Tensor network decoding
- Analytical bounds on thresholds
- Optimal decoder approximations
