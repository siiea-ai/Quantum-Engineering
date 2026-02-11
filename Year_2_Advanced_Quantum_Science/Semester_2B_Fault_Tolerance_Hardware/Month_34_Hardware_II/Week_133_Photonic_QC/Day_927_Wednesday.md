# Day 927: Boson Sampling

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Computational complexity, permanent calculation, Aaronson-Arkhipov theorem |
| Afternoon | 2.5 hours | Problem solving: Boson sampling analysis and verification |
| Evening | 1.5 hours | Computational lab: Boson sampling simulation |

## Learning Objectives

By the end of today, you will be able to:

1. Explain the computational complexity of calculating matrix permanents
2. Derive the connection between photon sampling and permanent calculation
3. Understand the Aaronson-Arkhipov theorem and its implications
4. Describe Gaussian boson sampling and its advantages
5. Analyze experimental implementations including Jiuzhang and Borealis
6. Implement small-scale boson sampling simulations

## Core Content

### 1. The Permanent Problem

**Definition of the Permanent:**
For an $n \times n$ matrix $A$, the permanent is:
$$\text{perm}(A) = \sum_{\sigma \in S_n} \prod_{i=1}^{n} A_{i,\sigma(i)}$$

Compare to the determinant:
$$\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} A_{i,\sigma(i)}$$

**Computational Complexity:**
- Determinant: $O(n^3)$ using Gaussian elimination
- Permanent: **#P-hard** (Valiant, 1979)

The permanent counts the number of perfect matchings in a bipartite graph - no known efficient algorithm exists!

**Why is Permanent Hard?**
The sign factors in the determinant allow cancellation that makes Gaussian elimination work. The permanent lacks this structure.

Best known algorithm: Ryser's formula, $O(n \cdot 2^n)$

$$\text{perm}(A) = (-1)^n \sum_{S \subseteq \{1,...,n\}} (-1)^{|S|} \prod_{i=1}^{n} \sum_{j \in S} A_{ij}$$

### 2. Photons and Permanents

**The Key Insight:**
The probability amplitude for $n$ photons to go from input modes to output modes through a linear optical network is proportional to a matrix permanent!

**Setup:**
- $m$-mode linear optical network described by unitary $U$
- $n$ photons in input modes $(s_1, s_2, ..., s_n)$
- Detect photons in output modes $(t_1, t_2, ..., t_n)$

**Transition Amplitude:**
$$\langle t_1, ..., t_n | \hat{U} | s_1, ..., s_n \rangle = \frac{\text{perm}(U_{S,T})}{\sqrt{\prod_i n_{s_i}! \prod_j n_{t_j}!}}$$

where $U_{S,T}$ is the $n \times n$ submatrix of $U$ with rows from $S$ and columns from $T$.

**Example: HOM Effect Revisited**
For two photons entering modes 1 and 2 of a 50:50 beam splitter:
$$U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Probability of output $(1,1)$ (one photon each):
$$P(1,1) = \left|\text{perm}\begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{pmatrix}\right|^2 = |1/2 - 1/2|^2 = 0$$

This is the HOM effect from the permanent perspective!

### 3. Boson Sampling Problem

**Definition (Aaronson-Arkhipov, 2011):**

**Input:**
- An $m \times m$ unitary matrix $U$ (with $m \geq n^2$)
- $n$ single photons in distinct input modes

**Output:**
- Sample from the output distribution $\{P(\mathbf{t})\}$

**Probability of output configuration $\mathbf{t}$:**
$$P(\mathbf{t}) = \frac{|\text{perm}(U_{S,T})|^2}{\prod_j t_j!}$$

**The Theorem:**
*Assuming the Permanent-of-Gaussians Conjecture and the Polynomial Hierarchy does not collapse, there is no efficient classical algorithm to sample from this distribution.*

**Implications:**
1. Quantum computers can efficiently sample (just run the experiment!)
2. Classical computers cannot efficiently sample
3. This is a **quantum computational advantage** demonstration

### 4. Complexity-Theoretic Arguments

**Why Not Just Compute the Probabilities?**
Each probability involves a #P-hard permanent. But we only need to *sample*, not compute exact probabilities.

**The Hiding Argument:**
The output distribution hides the permanent in a way that:
1. Any efficient classical sampler could be used to approximate permanents
2. This would collapse the polynomial hierarchy
3. Strong evidence this doesn't happen

**The No-Collision Regime:**
With $m \gg n^2$, output collisions (multiple photons in same mode) are exponentially unlikely. This simplifies analysis.

**Probability of no collisions:**
$$P(\text{no collision}) \approx e^{-n^2/m}$$

For $m = n^2$: $P \approx 1/e \approx 0.37$
For $m = 2n^2$: $P \approx 0.61$

### 5. Experimental Boson Sampling

**Requirements:**
1. $n$ indistinguishable single photons
2. High-fidelity $m$-mode linear optical network
3. Photon-number-resolving detection
4. Low loss throughout

**Scaling Challenges:**
- Single-photon sources: Must be highly indistinguishable
- Photon loss: Scales as $\eta^n$ where $\eta$ is transmission
- Mode count: Need $m \sim n^2$ modes

**Key Experiments:**

| Year | Group | Photons | Modes | Notes |
|------|-------|---------|-------|-------|
| 2013 | Multiple | 3-4 | 5-6 | First demonstrations |
| 2017 | USTC | 5 | 9 | Improved sources |
| 2019 | USTC | 14 | 60 | Near quantum advantage |
| 2020 | USTC (Jiuzhang) | 50+ | 100 | Claimed quantum advantage |
| 2022 | Xanadu (Borealis) | 216 | - | GBS, programmable |

### 6. Gaussian Boson Sampling (GBS)

**Motivation:**
Single-photon sources are hard. Squeezed states are easier to generate!

**GBS Setup:**
- Input: Squeezed vacuum states in some modes
- Network: Linear optical unitary
- Detection: Photon-number-resolving

**Squeezed Vacuum:**
$$|sq(r)\rangle = \hat{S}(r)|0\rangle = \sum_{n=0}^{\infty} \frac{\sqrt{(2n)!}}{2^n n!} \tanh^n(r) \cdot \text{sech}(r) |2n\rangle$$

Only even photon numbers (photons created in pairs).

**GBS Probability:**
For detection pattern $\mathbf{n}$:
$$P(\mathbf{n}) \propto \frac{|\text{Haf}(A_{\mathbf{n}})|^2}{\prod_i n_i!}$$

The **Hafnian** replaces the permanent:
$$\text{Haf}(A) = \sum_{\text{matchings}} \prod_{(i,j) \in M} A_{ij}$$

**Hafnian Complexity:**
Also #P-hard! GBS maintains computational hardness.

### 7. Applications of GBS

**Graph Problems:**
The Hafnian relates to perfect matching counting:
$$\text{Haf}(A_G) = \#\text{(perfect matchings in graph } G\text{)}$$

GBS can help solve:
- Dense subgraph problems
- Graph similarity
- Molecular vibronic spectra

**Molecular Simulation:**
Vibronic spectra of molecules can be computed via GBS:
$$I(\omega) \propto \sum_{\mathbf{n}} |\langle \mathbf{n} | \hat{U}_{Duschinsky} | 0 \rangle|^2 \delta(\omega - \omega_{\mathbf{n}})$$

The Duschinsky transformation is a linear optical transformation!

**Machine Learning:**
- Feature extraction via GBS samples
- Quantum kernels using photon statistics
- Generative models

### 8. Verification Challenges

**The Problem:**
How do we verify that a quantum device is sampling correctly from an exponentially large distribution?

**Verification Methods:**

1. **Coarse-grained tests:**
   - Check marginal distributions
   - Verify photon number conservation
   - Statistical distance tests

2. **Bayesian validation:**
   - Compare to uniform distribution
   - Compare to distinguishable photon simulation
   - Calculate likelihood ratios

3. **Symmetry tests:**
   - Bosonic clouding (bunching statistics)
   - Higher-order correlations

**Spoofing Attacks:**
Classical algorithms that try to fake quantum sampling:
- Mean-field approximation
- Metropolis sampling
- Thermal state approximation

These must be ruled out by verification tests.

## Quantum Computing Applications

### Quantum Advantage Demonstrations

Boson sampling provided the first clear demonstrations of quantum advantage:
1. **Jiuzhang (2020):** 76 photons, $10^{14}$ times faster than classical
2. **Jiuzhang 2.0 (2021):** 113 photons
3. **Borealis (2022):** Programmable GBS with 216 squeezed modes

### Path to Useful Quantum Computing

While boson sampling is not universal, it provides:
1. Validation of quantum hardware at scale
2. Near-term applications (molecular simulation, optimization)
3. Components for universal photonic QC

## Worked Examples

### Example 1: Three-Photon Sampling

**Problem:** Calculate the probability that three indistinguishable photons in modes 1, 2, 3 exit in modes 4, 5, 6 through a 6-mode DFT unitary.

**Solution:**
The 6-mode DFT matrix:
$$U_{jk} = \frac{1}{\sqrt{6}} \omega^{(j-1)(k-1)}, \quad \omega = e^{2\pi i/6}$$

The relevant $3 \times 3$ submatrix (rows 4,5,6 and columns 1,2,3):
$$U_{sub} = \frac{1}{\sqrt{6}}\begin{pmatrix} \omega^3 & \omega^6 & \omega^9 \\ \omega^4 & \omega^8 & \omega^{12} \\ \omega^5 & \omega^{10} & \omega^{15} \end{pmatrix}$$

Since $\omega^6 = 1$:
$$U_{sub} = \frac{1}{\sqrt{6}}\begin{pmatrix} -1 & 1 & -1 \\ \omega^4 & \omega^2 & 1 \\ \omega^5 & \omega^4 & \omega^3 \end{pmatrix}$$

The permanent:
$$\text{perm}(U_{sub}) = \frac{1}{6\sqrt{6}}\sum_{\sigma \in S_3} U_{1,\sigma(1)}U_{2,\sigma(2)}U_{3,\sigma(3)}$$

After calculation:
$$\text{perm}(U_{sub}) = \frac{1}{6\sqrt{6}}(1 - \omega^4 + \omega^5 - 1 + \omega^3 - \omega^2)$$

The probability:
$$P(4,5,6|1,2,3) = |\text{perm}(U_{sub})|^2 = \frac{1}{216}|...|^2$$

### Example 2: GBS with Two Squeezed Modes

**Problem:** For GBS with two squeezed modes (squeezing $r$) through a 50:50 beam splitter, calculate the probability of detecting 2 photons total.

**Solution:**
Input: $|sq(r)\rangle_1 \otimes |sq(r)\rangle_2$

The two-mode squeezed state components:
$$\propto |0,0\rangle + \tanh(r)|2,0\rangle + \tanh(r)|0,2\rangle + \tanh^2(r)|2,2\rangle + ...$$

After beam splitter, the $|1,1\rangle$ output comes from interference of $|2,0\rangle$ and $|0,2\rangle$ components.

For detecting exactly 2 photons:
$$P(n_{total} = 2) \approx 2\tanh^2(r)\text{sech}^2(r)$$

The distribution over patterns $(2,0)$, $(1,1)$, $(0,2)$ follows the Hafnian formula.

### Example 3: Classical Simulation Scaling

**Problem:** Estimate the classical computation time to simulate 50-photon boson sampling.

**Solution:**
Using Ryser's algorithm for permanent: $O(n \cdot 2^n)$

For $n = 50$:
$$T_{perm} \approx 50 \times 2^{50} \approx 5.6 \times 10^{16} \text{ operations}$$

At 1 GHz: $\approx 5.6 \times 10^7$ seconds $\approx 1.8$ years per sample

For the full distribution (exponentially many patterns), multiply by output dimension.

Classical supercomputer ($10^{18}$ FLOPS): still $\sim 56$ seconds per sample

Jiuzhang produces $\sim 10^{14}$ samples in 200 seconds, equivalent to $10^{10}$ years classically!

$$\boxed{\text{Quantum advantage ratio} \approx 10^{14}}$$

## Practice Problems

### Level 1: Direct Application

1. **Permanent Calculation**

   Calculate the permanent of:
   $$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$
   Compare to the determinant.

2. **Two-Photon Amplitudes**

   For the Hadamard-like unitary $U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$, calculate the amplitudes for all output patterns when two photons enter mode 1.

3. **Collision Probability**

   With $n = 10$ photons and $m = 200$ modes, estimate the probability of no output collisions.

### Level 2: Intermediate

4. **GBS Photon Statistics**

   A squeezed state with $r = 0.5$ enters one port of a 50:50 beam splitter, vacuum in the other. Calculate:
   a) Mean photon number at each output
   b) Probability of detecting exactly 2 photons at one output

5. **Verification Test**

   Design a statistical test to distinguish true boson sampling from uniform sampling. What is the expected value of this test statistic for each case?

6. **Submatrix Counting**

   For $n = 3$ photons and $m = 6$ modes, how many distinct $3 \times 3$ submatrices could contribute to the output distribution? How many permanents need to be calculated?

### Level 3: Challenging

7. **Complexity Lower Bound**

   Prove that if there existed a polynomial-time classical algorithm for exact boson sampling, then $P^{\#P} = BPP^{NP}$, implying $PH$ collapses to the third level.

8. **Lossy Boson Sampling**

   With photon loss rate $\gamma$ per mode, the effective Hilbert space dimension reduces. At what loss rate does boson sampling become efficiently classically simulable? Express in terms of $n$ and $m$.

9. **GBS for Graph Isomorphism**

   Explain how GBS could potentially help distinguish non-isomorphic graphs. What is the relationship between the adjacency matrix and the Hafnian? Why doesn't this solve graph isomorphism efficiently?

## Computational Lab: Boson Sampling Simulation

```python
"""
Day 927 Computational Lab: Boson Sampling Simulation
Permanent calculations and small-scale boson sampling
"""

import numpy as np
from scipy.special import factorial
from itertools import permutations, combinations
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

def permanent_naive(A: np.ndarray) -> complex:
    """
    Calculate permanent using naive definition (sum over permutations).
    Complexity: O(n! * n)
    Only practical for n <= 10
    """
    n = A.shape[0]
    perm_sum = 0
    for sigma in permutations(range(n)):
        product = 1
        for i, j in enumerate(sigma):
            product *= A[i, j]
        perm_sum += product
    return perm_sum


def permanent_ryser(A: np.ndarray) -> complex:
    """
    Calculate permanent using Ryser's formula.
    Complexity: O(n * 2^n)
    Practical for n <= 25-30
    """
    n = A.shape[0]
    result = 0

    # Iterate over all subsets of {0, 1, ..., n-1}
    for k in range(2**n):
        subset = [j for j in range(n) if k & (1 << j)]
        if len(subset) == 0:
            continue

        # Calculate row sums for this subset
        row_sums = np.sum(A[:, subset], axis=1)
        product = np.prod(row_sums)

        # Add or subtract based on subset size parity
        if (n - len(subset)) % 2 == 0:
            result += product
        else:
            result -= product

    return result * ((-1) ** n)


def permanent_glynn(A: np.ndarray) -> complex:
    """
    Calculate permanent using Glynn's formula (more numerically stable).
    Complexity: O(n * 2^n)
    """
    n = A.shape[0]
    row_sums = np.zeros(n, dtype=complex)
    result = 0

    # All 2^(n-1) sign combinations (first always +1)
    for k in range(2**(n-1)):
        signs = np.ones(n)
        for j in range(n-1):
            if k & (1 << j):
                signs[j+1] = -1

        # Row sums with signs
        weighted = A @ signs
        product = np.prod(weighted)
        result += product

    return result / (2**(n-1))


def random_unitary(n: int) -> np.ndarray:
    """Generate random unitary matrix using QR decomposition."""
    # Random complex matrix
    Z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    # Make unique by fixing phases
    D = np.diag(R)
    D = D / np.abs(D)
    Q = Q @ np.diag(D)
    return Q


def boson_sampling_probability(U: np.ndarray,
                                input_modes: List[int],
                                output_modes: List[int]) -> float:
    """
    Calculate probability for specific input/output configuration.

    Args:
        U: m x m unitary matrix
        input_modes: list of input mode indices (with repetition for multiple photons)
        output_modes: list of output mode indices

    Returns:
        Probability of this output configuration
    """
    n = len(input_modes)
    assert len(output_modes) == n, "Must have same number of input and output photons"

    # Extract submatrix
    submatrix = U[np.ix_(output_modes, input_modes)]

    # Calculate permanent
    perm = permanent_glynn(submatrix)

    # Count multiplicities for normalization
    from collections import Counter
    input_counts = Counter(input_modes)
    output_counts = Counter(output_modes)

    input_factor = np.prod([factorial(c) for c in input_counts.values()])
    output_factor = np.prod([factorial(c) for c in output_counts.values()])

    probability = np.abs(perm)**2 / (input_factor * output_factor)
    return probability


def enumerate_outputs(n_photons: int, n_modes: int, max_per_mode: int = None) -> List[Tuple[int, ...]]:
    """
    Enumerate all possible output configurations for n photons in m modes.
    """
    if max_per_mode is None:
        max_per_mode = n_photons

    def generate(remaining, start_mode, current):
        if remaining == 0:
            return [tuple(current)]
        if start_mode >= n_modes:
            return []

        results = []
        for k in range(min(remaining, max_per_mode) + 1):
            results.extend(generate(remaining - k, start_mode + 1, current + [k]))
        return results

    # Returns list of tuples (n_0, n_1, ..., n_{m-1})
    return generate(n_photons, 0, [])


def full_boson_sampling(U: np.ndarray, input_config: Tuple[int, ...]) -> dict:
    """
    Calculate full output distribution for boson sampling.

    Args:
        U: m x m unitary
        input_config: tuple (n_0, n_1, ..., n_{m-1}) of photon numbers per mode

    Returns:
        Dictionary mapping output configurations to probabilities
    """
    m = U.shape[0]
    n = sum(input_config)

    # Convert input config to mode list
    input_modes = []
    for mode, count in enumerate(input_config):
        input_modes.extend([mode] * count)

    # Enumerate all outputs
    outputs = enumerate_outputs(n, m)

    distribution = {}
    for output_config in outputs:
        output_modes = []
        for mode, count in enumerate(output_config):
            output_modes.extend([mode] * count)

        prob = boson_sampling_probability(U, input_modes, output_modes)
        if prob > 1e-15:
            distribution[output_config] = prob

    return distribution


def sample_boson_sampling(U: np.ndarray, input_config: Tuple[int, ...],
                          n_samples: int) -> List[Tuple[int, ...]]:
    """
    Sample from boson sampling distribution.
    """
    dist = full_boson_sampling(U, input_config)

    configs = list(dist.keys())
    probs = list(dist.values())
    probs = np.array(probs) / sum(probs)  # Normalize

    indices = np.random.choice(len(configs), size=n_samples, p=probs)
    return [configs[i] for i in indices]


def hong_ou_mandel_demo():
    """
    Demonstrate HOM effect through permanent calculation.
    """
    print("=" * 60)
    print("Hong-Ou-Mandel Effect via Permanent")
    print("=" * 60)

    # 50:50 beam splitter
    U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    # Input: |1,1⟩ (one photon each mode)
    input_config = (1, 1)

    dist = full_boson_sampling(U, input_config)

    print("\nInput: |1,1⟩ through 50:50 beam splitter")
    print("\nOutput distribution:")
    for config, prob in sorted(dist.items()):
        print(f"  |{config[0]},{config[1]}⟩: P = {prob:.6f}")

    print("\nNote: P(|1,1⟩) = 0 is the HOM effect!")


def three_photon_sampling():
    """
    Demonstrate 3-photon boson sampling.
    """
    print("\n" + "=" * 60)
    print("Three-Photon Boson Sampling")
    print("=" * 60)

    # 6-mode random unitary
    m = 6
    U = random_unitary(m)

    # Input: one photon in each of first 3 modes
    input_config = (1, 1, 1, 0, 0, 0)

    print(f"\nInput configuration: {input_config}")
    print(f"Unitary size: {m} x {m}")

    start_time = time.time()
    dist = full_boson_sampling(U, input_config)
    elapsed = time.time() - start_time

    print(f"\nComputed {len(dist)} non-zero outputs in {elapsed:.4f} seconds")

    # Show top 10 probabilities
    sorted_dist = sorted(dist.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 output configurations:")
    for config, prob in sorted_dist:
        print(f"  {config}: P = {prob:.6f}")

    # Check normalization
    total_prob = sum(dist.values())
    print(f"\nTotal probability: {total_prob:.6f} (should be 1)")

    # Visualize
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    all_probs = sorted(dist.values(), reverse=True)
    plt.bar(range(len(all_probs)), all_probs)
    plt.xlabel('Output configuration (sorted)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Boson Sampling Output Distribution', fontsize=14)

    plt.subplot(1, 2, 2)
    # Sample and show histogram
    n_samples = 10000
    samples = sample_boson_sampling(U, input_config, n_samples)

    sample_counts = {}
    for s in samples:
        sample_counts[s] = sample_counts.get(s, 0) + 1

    # Compare sampled to theoretical
    configs = sorted(dist.keys())[:20]
    theoretical = [dist.get(c, 0) for c in configs]
    sampled = [sample_counts.get(c, 0) / n_samples for c in configs]

    x = np.arange(len(configs))
    width = 0.35
    plt.bar(x - width/2, theoretical, width, label='Theoretical', alpha=0.7)
    plt.bar(x + width/2, sampled, width, label=f'Sampled (n={n_samples})', alpha=0.7)
    plt.xlabel('Configuration index', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Sampling vs Theoretical Distribution', fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.savefig('boson_sampling_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: boson_sampling_demo.png")


def permanent_scaling_analysis():
    """
    Analyze how permanent calculation scales with matrix size.
    """
    print("\n" + "=" * 60)
    print("Permanent Calculation Scaling")
    print("=" * 60)

    sizes = list(range(2, 16))
    times_ryser = []
    times_glynn = []

    for n in sizes:
        A = random_unitary(n)

        # Time Ryser
        start = time.time()
        for _ in range(3):
            permanent_ryser(A)
        times_ryser.append((time.time() - start) / 3)

        # Time Glynn
        start = time.time()
        for _ in range(3):
            permanent_glynn(A)
        times_glynn.append((time.time() - start) / 3)

        print(f"n={n}: Ryser {times_ryser[-1]:.4f}s, Glynn {times_glynn[-1]:.4f}s")

    plt.figure(figsize=(10, 6))
    plt.semilogy(sizes, times_ryser, 'bo-', label='Ryser', markersize=8)
    plt.semilogy(sizes, times_glynn, 'rs-', label='Glynn', markersize=8)

    # Theoretical scaling
    theoretical = [times_ryser[0] * (n * 2**n) / (sizes[0] * 2**sizes[0]) for n in sizes]
    plt.semilogy(sizes, theoretical, 'k--', label='O(n·2ⁿ) scaling', alpha=0.5)

    plt.xlabel('Matrix size n', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Permanent Calculation Scaling', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('permanent_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: permanent_scaling.png")


def bunching_analysis():
    """
    Analyze bunching statistics in boson sampling.
    """
    print("\n" + "=" * 60)
    print("Bunching Statistics Analysis")
    print("=" * 60)

    m = 10  # modes
    n_photons_list = [2, 3, 4, 5]

    plt.figure(figsize=(12, 5))

    for n in n_photons_list:
        U = random_unitary(m)
        input_config = tuple([1] * n + [0] * (m - n))

        dist = full_boson_sampling(U, input_config)

        # Analyze bunching: count outputs with k collisions
        bunching_stats = {}
        for config, prob in dist.items():
            # Number of modes with >1 photon
            n_bunched = sum(1 for c in config if c > 1)
            bunching_stats[n_bunched] = bunching_stats.get(n_bunched, 0) + prob

        ks = sorted(bunching_stats.keys())
        probs = [bunching_stats[k] for k in ks]

        plt.bar([k + 0.2 * (n - 3) for k in ks], probs,
                width=0.15, label=f'n={n}', alpha=0.7)

    plt.xlabel('Number of bunched modes (>1 photon)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Photon Bunching Statistics in Boson Sampling', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('bunching_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: bunching_statistics.png")


def validation_test():
    """
    Implement a simple validation test for boson sampling.
    """
    print("\n" + "=" * 60)
    print("Boson Sampling Validation Test")
    print("=" * 60)

    m = 6
    n = 3
    U = random_unitary(m)
    input_config = (1, 1, 1, 0, 0, 0)

    # Get theoretical distribution
    dist = full_boson_sampling(U, input_config)

    # Sample from true distribution
    n_samples = 5000
    true_samples = sample_boson_sampling(U, input_config, n_samples)

    # Generate "fake" uniform samples
    outputs = list(dist.keys())
    fake_samples = [outputs[np.random.randint(len(outputs))] for _ in range(n_samples)]

    # Calculate log-likelihood ratio
    def log_likelihood(samples, distribution):
        ll = 0
        for s in samples:
            if s in distribution and distribution[s] > 0:
                ll += np.log(distribution[s])
            else:
                ll += np.log(1e-10)  # Small probability for unseen
        return ll / len(samples)

    # Uniform distribution
    uniform_dist = {k: 1/len(dist) for k in dist.keys()}

    ll_true_vs_true = log_likelihood(true_samples, dist)
    ll_true_vs_uniform = log_likelihood(true_samples, uniform_dist)
    ll_fake_vs_true = log_likelihood(fake_samples, dist)
    ll_fake_vs_uniform = log_likelihood(fake_samples, uniform_dist)

    print(f"\nLog-likelihood per sample:")
    print(f"  True samples vs true dist: {ll_true_vs_true:.4f}")
    print(f"  True samples vs uniform:   {ll_true_vs_uniform:.4f}")
    print(f"  Fake samples vs true dist: {ll_fake_vs_true:.4f}")
    print(f"  Fake samples vs uniform:   {ll_fake_vs_uniform:.4f}")

    ratio_true = ll_true_vs_true - ll_true_vs_uniform
    ratio_fake = ll_fake_vs_true - ll_fake_vs_uniform

    print(f"\nLog-likelihood ratio (should be positive for true samples):")
    print(f"  True samples: {ratio_true:.4f}")
    print(f"  Fake samples: {ratio_fake:.4f}")


def main():
    """Run all boson sampling simulations."""
    print("\n" + "=" * 60)
    print("DAY 927: BOSON SAMPLING SIMULATIONS")
    print("=" * 60)

    # Test permanent calculations
    print("\n--- Permanent Calculation Test ---")
    A = np.array([[1, 2], [3, 4]], dtype=complex)
    print(f"Matrix:\n{A}")
    print(f"Permanent (naive): {permanent_naive(A)}")
    print(f"Permanent (Ryser): {permanent_ryser(A)}")
    print(f"Permanent (Glynn): {permanent_glynn(A)}")
    print(f"Determinant: {np.linalg.det(A)}")

    # Run demos
    hong_ou_mandel_demo()
    three_photon_sampling()
    permanent_scaling_analysis()
    bunching_analysis()
    validation_test()

    print("\n" + "=" * 60)
    print("Simulations Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Permanent definition | $\text{perm}(A) = \sum_{\sigma \in S_n} \prod_i A_{i,\sigma(i)}$ |
| Boson sampling probability | $P(\mathbf{t}) = \|\text{perm}(U_{S,T})\|^2 / \prod_j t_j!$ |
| Collision-free probability | $P \approx e^{-n^2/m}$ |
| Hafnian (for GBS) | $\text{Haf}(A) = \sum_{M} \prod_{(i,j) \in M} A_{ij}$ |
| Ryser complexity | $O(n \cdot 2^n)$ |
| Quantum advantage ratio | $\sim 10^{14}$ (Jiuzhang) |

### Key Takeaways

1. **Boson sampling** connects quantum optics to computational complexity via the permanent
2. The permanent is **#P-hard** - believed to be classically intractable
3. Photon indistinguishability enables quantum interference that's hard to simulate
4. **Gaussian boson sampling** uses squeezed states and Hafnians instead of permanents
5. **Verification** of large-scale boson sampling remains challenging
6. Experimental demonstrations have achieved clear quantum advantage for sampling tasks

## Daily Checklist

- [ ] I can explain why permanent calculation is computationally hard
- [ ] I understand the connection between photon amplitudes and permanents
- [ ] I can state the Aaronson-Arkhipov theorem and its implications
- [ ] I understand the differences between standard and Gaussian boson sampling
- [ ] I completed the computational lab simulations
- [ ] I solved at least 3 practice problems

## Preview of Day 928

Tomorrow we transition to **Continuous Variable Quantum Computing**, where information is encoded in the continuous quadrature variables of light fields rather than discrete photon numbers. Key topics:
- Quadrature operators and phase space
- Gaussian states: coherent, squeezed, thermal
- CV gates: displacement, squeezing, rotation
- Measurement-based CV quantum computing
