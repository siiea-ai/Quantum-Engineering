# Day 768: Threshold Computation Methods

## Overview

**Day:** 768 of 1008
**Week:** 110 (Threshold Theorems & Analysis)
**Month:** 28 (Advanced Stabilizer Applications)
**Topic:** Numerical and Analytical Methods for Computing Error Thresholds

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Monte Carlo methods |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Tensor network approaches |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Implementation and analysis |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Implement** Monte Carlo threshold simulations
2. **Apply** tensor network methods for exact threshold bounds
3. **Use** statistical mechanics mappings for analytical estimates
4. **Analyze** finite-size effects in threshold estimation
5. **Compare** different numerical approaches
6. **Derive** analytical upper and lower bounds on thresholds

---

## Core Content

### 1. Monte Carlo Threshold Simulation

Monte Carlo simulation is the workhorse for threshold estimation.

#### Basic Algorithm

**Algorithm: Monte Carlo Threshold Estimation**
```
1. For each physical error rate p:
   2. For trial = 1 to N_trials:
      3. Generate random errors E with probability p per qubit
      4. Compute syndrome sigma = syndrome(E)
      5. Apply decoder: E_corrected = decode(sigma)
      6. Check if E + E_corrected is logical error
      7. Record success/failure
   8. Compute P_logical(p) = failures / N_trials
9. Find p_th where P_logical changes behavior
```

#### Error Bars and Statistics

For $N$ trials with $k$ failures:
$$\hat{p} = \frac{k}{N}, \quad \sigma = \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}$$

**Wilson confidence interval:**
$$\boxed{p \in \frac{\hat{p} + \frac{z^2}{2N} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}}$$

where $z = 1.96$ for 95% confidence.

#### Variance Reduction Techniques

**Importance sampling:**
Sample errors from distribution $Q(E)$ instead of $P(E)$:
$$\langle f \rangle_P = \sum_E f(E) P(E) = \sum_E f(E) \frac{P(E)}{Q(E)} Q(E) = \langle f \cdot w \rangle_Q$$

**Optimal proposal:**
$$Q(E) \propto P(E) |f(E)|$$

### 2. Finite-Size Scaling Analysis

Extracting thresholds from finite-size simulations.

#### Crossing Method

At threshold, logical error rate is independent of system size:
$$\boxed{P_L(p_{th}, L_1) = P_L(p_{th}, L_2) \quad \forall L_1, L_2}$$

**Algorithm:**
1. Simulate $P_L(p)$ for multiple sizes $L$
2. Find crossing point of curves
3. Extrapolate $L \to \infty$ using scaling

#### Scaling Collapse

Near threshold:
$$\boxed{P_L(p) = \tilde{f}\left((p - p_c) L^{1/\nu}\right)}$$

**Fitting procedure:**
1. Parameterize: $(p_c, \nu)$
2. Plot $P_L$ vs $(p - p_c) L^{1/\nu}$
3. Minimize scatter in collapsed data
4. Best fit gives $p_c$ and $\nu$

#### Corrections to Scaling

More accurate form:
$$P_L(p) = \tilde{f}\left((p - p_c) L^{1/\nu}\right) + L^{-\omega} g\left((p - p_c) L^{1/\nu}\right)$$

where $\omega$ is the leading correction exponent.

### 3. Tensor Network Methods

Tensor networks provide exact or near-exact threshold calculations.

#### Matrix Product State Decoding

Represent the syndrome probability as a tensor network:
$$P(\sigma | E) = \text{tTr}\left[\prod_i T_i^{[E_i, \sigma_i]}\right]$$

**Boundary MPS:**
$$|\psi_\sigma\rangle = \sum_E P(E) |E\rangle_{\sigma}$$

Decode by finding maximum weight state.

#### Approximate Contracted Networks

For 2D codes, contract tensor network approximately:
$$\boxed{Z = \text{tTr}[T_1 \cdot T_2 \cdots T_n] \approx \text{MPS contraction}}$$

**Bond dimension scaling:**
- Exact: $\chi \sim 2^L$ (exponential)
- Approximate: $\chi$ fixed, polynomial time
- Error controlled by bond dimension

#### MERA for Hierarchical Codes

Multi-scale Entanglement Renormalization Ansatz for concatenated codes:
- Natural structure matches concatenation
- Efficient threshold computation
- Captures multi-scale correlations

### 4. Statistical Mechanics Approaches

Leveraging the stat-mech mapping for analytical results.

#### Series Expansion

**High-temperature expansion:**
$$Z = \sum_n a_n \beta^n$$

Near threshold (low $\beta$):
$$P_{logical} \sim \sum_n c_n p^n$$

Threshold from radius of convergence.

#### Replica Method

For RBIM:
$$\overline{\ln Z} = \lim_{n \to 0} \frac{\overline{Z^n} - 1}{n}$$

Self-averaging at large system size gives threshold.

#### Cavity Method

Belief propagation on the factor graph:
$$\boxed{m_{i \to a}(\sigma_i) \propto \prod_{b \in \partial i \setminus a} \sum_{\sigma_j: j \in \partial b \setminus i} \psi_b(\sigma_{\partial b})}$$

Fixed point gives marginal probabilities.

### 5. Analytical Bounds

Rigorous bounds without simulation.

#### Upper Bounds (Achievability)

**Union bound:**
$$P_{logical} \leq \sum_{E: \text{logical}} P(E) \leq n \cdot p^{d/2}$$

For distance $d$ code:
$$\boxed{p_{th} \geq \frac{1}{n^{2/d}} \quad \text{(weak bound)}}$$

**Peierls argument:**
For surface code with perimeter $\ell$:
$$P(\text{logical error}) \leq \sum_{\ell \geq L} N_\ell \cdot p^\ell$$

where $N_\ell$ is the number of paths of length $\ell$.

#### Lower Bounds (Converse)

**Hashing bound:**
$$R < 1 - H(p) \quad \Rightarrow \quad p_{th} \leq H^{-1}(1-R)$$

where $R = k/n$ is the code rate.

For surface codes ($R \to 0$):
$$p_{th} \leq H^{-1}(1) \approx 11\%$$

### 6. Decoder-Specific Methods

Threshold depends on decoder; compute for specific decoder.

#### MWPM Threshold

Minimum Weight Perfect Matching:
1. Build complete graph on syndrome defects
2. Weights = distances
3. Find minimum weight perfect matching
4. Correction = union of shortest paths

**Threshold:** Analyze when MWPM fails.

$$P_{fail}^{MWPM} \leq \sum_{\text{ambiguous}} P(\text{matching wrong})$$

#### Union-Find Threshold

Faster decoder, slightly lower threshold:
1. Union-Find data structure
2. Grow clusters from defects
3. Merge when clusters touch

Threshold lower due to greedy nature.

---

## Worked Examples

### Example 1: Monte Carlo Error Analysis

**Problem:** After 10,000 trials at $p = 0.10$ with 1,234 logical errors, what is the 95% confidence interval for $P_{logical}$?

**Solution:**

Point estimate:
$$\hat{p} = \frac{1234}{10000} = 0.1234$$

Standard error:
$$\sigma = \sqrt{\frac{0.1234 \times 0.8766}{10000}} = 0.00329$$

Wilson interval (z = 1.96):
$$\text{Numerator} = \hat{p} + \frac{1.96^2}{2 \times 10000} \pm 1.96 \times 0.00329$$

Simplified 95% CI:
$$P_{logical} \in [0.1234 - 1.96 \times 0.00329, 0.1234 + 1.96 \times 0.00329]$$
$$\boxed{P_{logical} \in [0.117, 0.130]}$$

### Example 2: Finite-Size Scaling

**Problem:** Given data $P_L$ at sizes $L = 5, 7, 9$ and error rates $p = 0.08, 0.10, 0.12$, estimate threshold using crossing method.

| p | L=5 | L=7 | L=9 |
|---|-----|-----|-----|
| 0.08 | 0.15 | 0.10 | 0.06 |
| 0.10 | 0.22 | 0.20 | 0.19 |
| 0.12 | 0.32 | 0.35 | 0.38 |

**Solution:**

At threshold, curves cross. Looking at differences:

For $L=5$ vs $L=9$:
- $p=0.08$: $0.15 - 0.06 = +0.09$ (L=5 higher)
- $p=0.10$: $0.22 - 0.19 = +0.03$ (L=5 higher, but close)
- $p=0.12$: $0.32 - 0.38 = -0.06$ (L=9 higher)

Crossing between $p=0.10$ and $p=0.12$.

Linear interpolation:
$$p_{th} \approx 0.10 + 0.02 \times \frac{0.03}{0.03 + 0.06} = 0.10 + 0.02 \times 0.33 = 0.107$$

$$\boxed{p_{th} \approx 10.7\%}$$

### Example 3: Peierls Bound

**Problem:** Derive an upper bound on surface code threshold using Peierls argument.

**Solution:**

A logical error requires an error chain spanning the code (length $\geq L$).

Number of paths of length $\ell$ starting from boundary:
$$N_\ell \leq L \cdot 3^{\ell-1}$$

(Starting point on boundary, then 3 choices per step)

Probability of such path:
$$P(\text{path of length } \ell) \leq L \cdot 3^{\ell-1} \cdot p^\ell$$

For logical error:
$$P_{logical} \leq \sum_{\ell=L}^{\infty} L \cdot 3^{\ell-1} p^\ell = \frac{L}{3} \cdot \frac{(3p)^L}{1-3p}$$

For this to vanish as $L \to \infty$:
$$3p < 1 \quad \Rightarrow \quad p < \frac{1}{3}$$

**Refined bound:** Including path counting more carefully:
$$\boxed{p_{th} \leq 18.9\%}$$

Actual threshold ~10.9% is lower due to decoder limitations.

---

## Practice Problems

### Problem Set A: Monte Carlo Methods

**A1.** Design an importance sampling scheme for threshold estimation that samples errors from high-syndrome-weight configurations.

**A2.** How many trials are needed to distinguish $p_{th} = 10\%$ from $p_{th} = 11\%$ with 99% confidence?

**A3.** Implement parallel tempering for RBIM to accelerate threshold estimation.

### Problem Set B: Tensor Networks

**B1.** Write the tensor network contraction for a 3x3 surface code with specific error configuration.

**B2.** How does MPS bond dimension scale with accuracy for approximate decoding?

**B3.** Design a MERA-based decoder for a 3-level concatenated [[7,1,3]] code.

### Problem Set C: Analytical Methods

**C1.** Use the cavity method to derive self-consistent equations for surface code threshold.

**C2.** Compute the first few terms in the high-temperature expansion for 2D RBIM.

**C3.** Derive the quantum Gilbert-Varshamov bound and apply it to surface codes.

---

## Computational Lab

```python
"""
Day 768 Computational Lab: Threshold Computation Methods
========================================================

Implement and compare threshold estimation techniques.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, curve_fit
from scipy.special import comb
import warnings


@dataclass
class SimulationResult:
    """Container for Monte Carlo results."""
    p: float
    L: int
    trials: int
    failures: int
    p_logical: float
    confidence_interval: Tuple[float, float]


def monte_carlo_threshold(L: int, p: float, trials: int = 1000,
                         decoder: str = 'mwpm') -> SimulationResult:
    """
    Monte Carlo threshold simulation for simple model.

    Simulates Z-only errors on L x L repetition-like code.
    """
    failures = 0

    for _ in range(trials):
        # Generate random Z errors
        errors = np.random.random((L, L)) < p

        # Simple syndrome: parity along each row
        syndrome = np.zeros((L, L-1), dtype=bool)
        for i in range(L):
            for j in range(L-1):
                syndrome[i, j] = errors[i, j] ^ errors[i, j+1]

        # Decode: try to find minimum weight correction
        if decoder == 'mwpm':
            correction = decode_mwpm_simple(syndrome, L)
        else:
            correction = np.zeros((L, L), dtype=bool)

        # Residual error
        residual = errors ^ correction

        # Logical error: odd parity across any row
        logical_error = False
        for i in range(L):
            if np.sum(residual[i, :]) % 2 == 1:
                # Check if it spans (simplified)
                if np.sum(residual[i, :]) > 0:
                    logical_error = True
                    break

        if logical_error:
            failures += 1

    p_logical = failures / trials
    ci = wilson_interval(failures, trials)

    return SimulationResult(p, L, trials, failures, p_logical, ci)


def decode_mwpm_simple(syndrome: np.ndarray, L: int) -> np.ndarray:
    """Simplified MWPM-like decoder."""
    correction = np.zeros((L, L), dtype=bool)

    # Find syndrome defects
    defects = list(zip(*np.where(syndrome)))

    # Greedy pairing by proximity
    used = set()
    for i, (r1, c1) in enumerate(defects):
        if i in used:
            continue

        # Find nearest unmatched defect
        min_dist = float('inf')
        best_j = -1
        for j, (r2, c2) in enumerate(defects):
            if j <= i or j in used:
                continue
            dist = abs(r1 - r2) + abs(c1 - c2)
            if dist < min_dist:
                min_dist = dist
                best_j = j

        if best_j >= 0:
            used.add(i)
            used.add(best_j)
            # Apply correction along path
            r2, c2 = defects[best_j]
            # Horizontal path
            for c in range(min(c1, c2), max(c1, c2) + 1):
                correction[r1, c] ^= True

    return correction


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return (0.0, 1.0)

    p_hat = k / n
    denominator = 1 + z**2 / n

    center = (p_hat + z**2 / (2*n)) / denominator
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator

    return (max(0, center - half_width), min(1, center + half_width))


def importance_sampling(L: int, p: float, p_sample: float,
                       trials: int = 1000) -> float:
    """
    Importance sampling with different error probability.

    Weight = P(E | p) / P(E | p_sample)
    """
    total_weight = 0
    weighted_failures = 0

    for _ in range(trials):
        # Sample errors at rate p_sample
        errors = np.random.random((L, L)) < p_sample
        n_errors = np.sum(errors)
        n_qubits = L * L

        # Importance weight
        # P(E | p) / P(E | p_sample) = (p/p_sample)^n_err * ((1-p)/(1-p_sample))^(n-n_err)
        log_weight = (n_errors * np.log(p / p_sample) +
                     (n_qubits - n_errors) * np.log((1-p) / (1-p_sample)))
        weight = np.exp(log_weight)

        # Check for logical error (simplified)
        logical_error = any(np.sum(errors[i, :]) % 2 == 1 for i in range(L))

        total_weight += weight
        if logical_error:
            weighted_failures += weight

    return weighted_failures / total_weight


def finite_size_scaling_fit(p_values: np.ndarray,
                           L_values: List[int],
                           P_L_data: Dict[int, np.ndarray]
                           ) -> Tuple[float, float]:
    """
    Fit threshold and critical exponent from finite-size data.

    Returns (p_c, nu).
    """

    def scaling_function(x):
        """Universal scaling function approximation."""
        return 0.5 * (1 + np.tanh(x))

    def compute_residual(params):
        p_c, nu = params
        residuals = []

        for L in L_values:
            for i, p in enumerate(p_values):
                x = (p - p_c) * (L ** (1/nu))
                predicted = scaling_function(x)
                actual = P_L_data[L][i]
                residuals.append((predicted - actual)**2)

        return np.sum(residuals)

    # Grid search for initial guess
    best_residual = float('inf')
    best_params = (0.1, 1.5)

    for p_c in np.linspace(0.08, 0.12, 20):
        for nu in np.linspace(1.0, 2.0, 10):
            res = compute_residual((p_c, nu))
            if res < best_residual:
                best_residual = res
                best_params = (p_c, nu)

    return best_params


def crossing_threshold(p_values: np.ndarray,
                      L1_data: np.ndarray,
                      L2_data: np.ndarray) -> float:
    """Find threshold as crossing point of two curves."""

    # Find sign change in difference
    diff = L1_data - L2_data

    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            # Linear interpolation
            alpha = abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
            return p_values[i] + alpha * (p_values[i+1] - p_values[i])

    return np.mean(p_values)  # Fallback


def peierls_bound(L: int, p: float) -> float:
    """
    Peierls-style upper bound on logical error probability.

    Sum over all error chains that span the code.
    """
    # Number of paths of length l starting from boundary
    # N_l <= L * 3^(l-1)  (rough bound)

    P_logical = 0
    for l in range(L, 3*L):  # Chain length
        N_l = L * (3 ** (l - 1))
        P_l = N_l * (p ** l)
        P_logical += P_l
        if P_l < 1e-15:
            break

    return min(1.0, P_logical)


def tensor_network_contraction_1d(L: int, p: float) -> float:
    """
    Simplified 1D tensor network contraction.

    Models repetition code as MPS.
    """
    # Transfer matrix for repetition code
    # T[s1, s2] = P(error) if s1 != s2, else P(no error)

    T = np.array([[1-p, p],
                  [p, 1-p]])

    # Contract L times
    result = np.linalg.matrix_power(T, L)

    # Logical error = odd parity
    P_even = (result[0, 0] + result[1, 1]) / 2
    P_odd = (result[0, 1] + result[1, 0]) / 2

    return P_odd  # Logical error rate


# ============================================================
# Main Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 768: THRESHOLD COMPUTATION METHODS")
    print("=" * 70)

    # Demo 1: Basic Monte Carlo
    print("\n" + "=" * 70)
    print("Demo 1: Monte Carlo Threshold Estimation")
    print("=" * 70)

    print("\nRunning Monte Carlo simulations...")
    print(f"{'p':<10} {'L=5':<20} {'L=7':<20} {'L=9':<20}")
    print("-" * 70)

    p_values = np.array([0.06, 0.08, 0.10, 0.12, 0.14])
    L_values = [5, 7, 9]
    results = {L: [] for L in L_values}

    for p in p_values:
        row = f"{p:<10.2f} "
        for L in L_values:
            res = monte_carlo_threshold(L, p, trials=500)
            results[L].append(res.p_logical)
            row += f"{res.p_logical:.3f} ({res.confidence_interval[0]:.3f}-{res.confidence_interval[1]:.3f}) "
        print(row)

    # Demo 2: Crossing method
    print("\n" + "=" * 70)
    print("Demo 2: Threshold via Crossing Method")
    print("=" * 70)

    p_th_5_7 = crossing_threshold(p_values, np.array(results[5]), np.array(results[7]))
    p_th_7_9 = crossing_threshold(p_values, np.array(results[7]), np.array(results[9]))

    print(f"\nCrossing L=5 vs L=7: p_th = {p_th_5_7:.3f}")
    print(f"Crossing L=7 vs L=9: p_th = {p_th_7_9:.3f}")
    print(f"Average estimate: p_th = {(p_th_5_7 + p_th_7_9)/2:.3f}")

    # Demo 3: Finite-size scaling
    print("\n" + "=" * 70)
    print("Demo 3: Finite-Size Scaling Analysis")
    print("=" * 70)

    P_L_data = {L: np.array(results[L]) for L in L_values}
    p_c_fit, nu_fit = finite_size_scaling_fit(p_values, L_values, P_L_data)

    print(f"\nScaling fit results:")
    print(f"  Critical point: p_c = {p_c_fit:.4f}")
    print(f"  Correlation exponent: nu = {nu_fit:.3f}")

    print("\nScaling collapse:")
    print(f"{'x = (p-p_c)*L^(1/nu)':<25} {'P_L':<10} {'L':<5}")
    print("-" * 45)

    for L in L_values:
        for i, p in enumerate(p_values):
            x = (p - p_c_fit) * (L ** (1/nu_fit))
            print(f"{x:<25.3f} {results[L][i]:<10.3f} {L:<5}")

    # Demo 4: Importance sampling
    print("\n" + "=" * 70)
    print("Demo 4: Importance Sampling")
    print("=" * 70)

    print("\nComparing standard vs importance sampling:")
    print(f"{'Method':<25} {'p=0.05':<12} {'p=0.08':<12}")
    print("-" * 50)

    for p_target in [0.05, 0.08]:
        # Standard sampling
        res_std = monte_carlo_threshold(5, p_target, trials=1000)

        # Importance sampling (sample at p=0.10)
        p_is = importance_sampling(5, p_target, p_sample=0.10, trials=1000)

        print(f"{'Standard MC':<25} {res_std.p_logical:<12.4f}", end="")
        print(f"{'Importance (p_s=0.10)':<25} {p_is:<12.4f}")

    # Demo 5: Analytical bounds
    print("\n" + "=" * 70)
    print("Demo 5: Analytical Bounds (Peierls)")
    print("=" * 70)

    print("\nPeierls upper bound on P_logical:")
    print(f"{'L':<6} {'p=0.05':<15} {'p=0.10':<15} {'p=0.15':<15}")
    print("-" * 55)

    for L in [3, 5, 7, 9]:
        row = f"{L:<6} "
        for p in [0.05, 0.10, 0.15]:
            bound = peierls_bound(L, p)
            row += f"{bound:<15.4e} "
        print(row)

    print("\nPeierls bound shows p_th < 1/3 = 33%")
    print("Actual surface code threshold ~11%")

    # Demo 6: 1D tensor network
    print("\n" + "=" * 70)
    print("Demo 6: 1D Tensor Network (Repetition Code)")
    print("=" * 70)

    print("\nExact logical error rate via transfer matrix:")
    print(f"{'L':<6} {'p=0.05':<12} {'p=0.10':<12} {'p=0.20':<12}")
    print("-" * 45)

    for L in [3, 5, 7, 11, 21]:
        row = f"{L:<6} "
        for p in [0.05, 0.10, 0.20]:
            P_L = tensor_network_contraction_1d(L, p)
            row += f"{P_L:<12.4e} "
        print(row)

    print("\nNote: P_L -> 0 for p < 0.5 as L -> infinity")
    print("Repetition code threshold = 50%")

    # Summary
    print("\n" + "=" * 70)
    print("THRESHOLD COMPUTATION METHODS SUMMARY")
    print("=" * 70)

    print("""
    +---------------------------------------------------------------+
    |  THRESHOLD COMPUTATION METHODS                                |
    +---------------------------------------------------------------+
    |                                                               |
    |  MONTE CARLO:                                                 |
    |    - Direct simulation of error/decode cycle                  |
    |    - Statistical uncertainty: sigma ~ 1/sqrt(N)              |
    |    - Importance sampling for rare events                      |
    |                                                               |
    |  FINITE-SIZE SCALING:                                         |
    |    - P_L(p) = f((p - p_c) * L^(1/nu))                        |
    |    - Crossing method: P_L1(p_th) = P_L2(p_th)                |
    |    - Fit p_c and nu from data collapse                       |
    |                                                               |
    |  TENSOR NETWORKS:                                             |
    |    - Exact for 1D (transfer matrix)                          |
    |    - Approximate for 2D (MPS, PEPS)                          |
    |    - Bond dimension controls accuracy                        |
    |                                                               |
    |  ANALYTICAL BOUNDS:                                           |
    |    - Peierls argument: upper bound from path counting        |
    |    - Hashing bound: information-theoretic limit              |
    |    - RBIM critical point: optimal decoder threshold          |
    |                                                               |
    +---------------------------------------------------------------+
    """)

    # Method comparison table
    print("\nMethod Comparison:")
    print(f"{'Method':<25} {'Accuracy':<15} {'Speed':<15} {'Scalability':<15}")
    print("-" * 70)

    methods = [
        ("Monte Carlo", "Statistical", "Moderate", "Any size"),
        ("Finite-size scaling", "Extrapolation", "Moderate", "Limited L"),
        ("Tensor network (exact)", "Exact", "Slow", "Small systems"),
        ("Tensor network (approx)", "Controllable", "Fast", "Large systems"),
        ("Peierls bound", "Upper bound", "Very fast", "Any size"),
    ]

    for method, acc, speed, scale in methods:
        print(f"{method:<25} {acc:<15} {speed:<15} {scale:<15}")

    print("=" * 70)
    print("Day 768 Complete: Threshold Computation Methods Mastered")
    print("=" * 70)
```

---

## Summary

### Method Comparison

| Method | Type | Complexity | Best Use Case |
|--------|------|------------|---------------|
| Monte Carlo | Numerical | O(N trials) | General threshold estimation |
| Finite-size scaling | Statistical | O(N * L sizes) | Precision threshold |
| Tensor Network | Exact/Approx | O(chi^3 L) | Small/medium systems |
| Peierls Bound | Analytical | O(1) | Quick upper bound |

### Critical Equations

$$\boxed{P_L(p) = f\left((p - p_c) L^{1/\nu}\right) \quad \text{(finite-size scaling)}}$$

$$\boxed{\sigma = \sqrt{\frac{\hat{p}(1-\hat{p})}{N}} \quad \text{(MC standard error)}}$$

$$\boxed{P_{logical} \leq \sum_{\ell \geq L} N_\ell \cdot p^\ell \quad \text{(Peierls bound)}}$$

$$\boxed{Z = \text{tTr}[T_1 \cdot T_2 \cdots T_n] \quad \text{(tensor network)}}$$

---

## Daily Checklist

- [ ] Implemented Monte Carlo threshold simulation
- [ ] Applied finite-size scaling analysis
- [ ] Understood tensor network contraction
- [ ] Derived analytical bounds
- [ ] Compared computational methods
- [ ] Ran numerical experiments

---

## Preview: Day 769

Tomorrow we analyze **Resource Scaling**:
- Qubit overhead formulas
- Gate count scaling
- Time overhead and clock cycles
- Space-time tradeoffs
- Architecture comparison
