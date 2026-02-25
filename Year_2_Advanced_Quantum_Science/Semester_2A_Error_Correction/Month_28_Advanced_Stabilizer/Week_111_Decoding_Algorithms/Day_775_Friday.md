# Day 775: Belief Propagation & LDPC Decoding

## Week 111: Decoding Algorithms | Month 28: Advanced Stabilizer Codes

---

## Daily Schedule

| Session | Time | Duration | Focus |
|---------|------|----------|-------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Factor Graphs & Message Passing Theory |
| Afternoon | 1:00 PM - 4:00 PM | 3 hours | Sum-Product, Min-Sum & Quantum LDPC |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: BP Decoder Implementation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Construct** factor graphs representing stabilizer code constraints
2. **Derive** the sum-product (belief propagation) message update rules
3. **Implement** the min-sum algorithm as a practical approximation
4. **Analyze** the impact of short cycles on BP performance
5. **Apply** BP decoding to quantum LDPC codes
6. **Compare** BP thresholds with MWPM and other decoders

---

## Core Content

### 1. Factor Graphs for Stabilizer Codes

A **factor graph** is a bipartite graph representing the factorization of a probability distribution:

$$P(x_1, \ldots, x_n) = \frac{1}{Z} \prod_{a} f_a(x_{\partial a})$$

where $f_a$ are factor functions and $x_{\partial a}$ are variables connected to factor $a$.

**For Stabilizer Codes:**

- **Variable nodes**: Error bits $e_i \in \{0, 1\}$ for each qubit
- **Check nodes**: Stabilizer generators $g_a$
- **Edges**: Connect variable $i$ to check $a$ if $g_a$ acts non-trivially on qubit $i$

**Factor Functions:**

For syndrome measurement $s_a \in \{0, 1\}$:

$$f_a(e_{\partial a}) = \begin{cases} 1 & \text{if } \bigoplus_{i \in \partial a} e_i = s_a \\ 0 & \text{otherwise} \end{cases}$$

This enforces that the parity of errors equals the syndrome.

**Prior Factors:**

For independent errors with probability $p$:
$$f_i(e_i) = \begin{cases} 1-p & \text{if } e_i = 0 \\ p & \text{if } e_i = 1 \end{cases}$$

### 2. The Sum-Product Algorithm

Belief Propagation computes marginal probabilities by passing **messages** between nodes:

**Variable-to-Check Messages:**

$$\boxed{\mu_{i \to a}(e_i) = f_i(e_i) \prod_{b \in \partial i \setminus a} \mu_{b \to i}(e_i)}$$

**Check-to-Variable Messages:**

$$\boxed{\mu_{a \to i}(e_i) = \sum_{e_{\partial a \setminus i}} f_a(e_{\partial a}) \prod_{j \in \partial a \setminus i} \mu_{j \to a}(e_j)}$$

**Belief (Marginal) Computation:**

$$b_i(e_i) \propto f_i(e_i) \prod_{a \in \partial i} \mu_{a \to i}(e_i)$$

**Decision:**

$$\hat{e}_i = \text{argmax}_{e_i} \, b_i(e_i)$$

### 3. Log-Likelihood Ratio Formulation

For binary variables, use **log-likelihood ratios** (LLRs) for numerical stability:

$$L_i = \log \frac{P(e_i = 0)}{P(e_i = 1)}$$

**Prior LLR:**
$$L_i^{(0)} = \log \frac{1-p}{p}$$

**Variable-to-Check LLR:**
$$\lambda_{i \to a} = L_i^{(0)} + \sum_{b \in \partial i \setminus a} \Lambda_{b \to i}$$

**Check-to-Variable LLR:**

$$\boxed{\Lambda_{a \to i} = 2 \tanh^{-1}\left( (1 - 2s_a) \prod_{j \in \partial a \setminus i} \tanh\left(\frac{\lambda_{j \to a}}{2}\right) \right)}$$

This elegant formula comes from the Fourier transform of parity constraints.

### 4. The Min-Sum Algorithm

The sum-product algorithm involves expensive $\tanh$ computations. The **min-sum** approximation replaces products with minima:

$$\Lambda_{a \to i} \approx (1 - 2s_a) \cdot \left( \prod_{j \in \partial a \setminus i} \text{sign}(\lambda_{j \to a}) \right) \cdot \min_{j \in \partial a \setminus i} |\lambda_{j \to a}|$$

**Advantages:**
- Integer arithmetic possible
- Faster computation
- Hardware-friendly

**Disadvantages:**
- Suboptimal threshold
- May require normalization/offset corrections

**Normalized Min-Sum:**
$$\Lambda_{a \to i} = \alpha \cdot \text{min-sum}(\lambda_{j \to a})$$

with $\alpha \approx 0.75$ to improve accuracy.

### 5. Convergence and Cycle Issues

BP is exact on **tree-structured** factor graphs. For graphs with cycles:

**Short Cycles Problem:**

Short cycles cause messages to reinforce themselves incorrectly:
- 4-cycles: Message returns after 2 iterations
- 6-cycles: Message returns after 3 iterations

**Impact on Quantum Codes:**

Classical LDPC codes are designed with large girth (minimum cycle length). Quantum LDPC codes often have smaller girth due to CSS construction constraints.

**Typical Girths:**
| Code | Minimum Girth |
|------|---------------|
| Classical LDPC | 8-12 |
| Surface Code | 4 |
| Quantum LDPC (CSS) | 6 |
| Quantum LDPC (recent) | 8+ |

**Mitigation Strategies:**
1. **Damping**: $\lambda^{(t+1)} = \alpha \lambda^{(t)} + (1-\alpha) \lambda^{\text{new}}$
2. **Max iterations**: Stop before oscillation
3. **Post-processing**: Use BP output as prior for another decoder

### 6. Quantum LDPC Codes and BP

Recent breakthroughs in quantum LDPC codes make BP increasingly important:

**Classical LDPC Review:**

For an $[n, k, d]$ classical LDPC code:
- Sparse parity-check matrix $H$
- Constant row/column weights
- Near-Shannon-limit performance with BP

**Quantum LDPC Codes:**

CSS construction from classical codes $C_X, C_Z$ where $C_X^{\perp} \subseteq C_Z$:
- Stabilizer generators from $H_X, H_Z$
- Must satisfy $H_X H_Z^T = 0$

**Recent Quantum LDPC Constructions:**

| Code Family | Rate | Distance | BP Threshold |
|-------------|------|----------|--------------|
| Hypergraph Product | $\Theta(1/\sqrt{n})$ | $\Theta(\sqrt{n})$ | ~4-8% |
| Lifted Product | $\Theta(1)$ | $\Theta(\sqrt{n})$ | ~7-10% |
| Balanced Product | $\Theta(1)$ | $\Theta(\sqrt{n})$ | ~8-11% |

**Why BP for Quantum LDPC:**

1. Sparse structure makes BP efficient: $O(n)$ per iteration
2. Near-optimal for tree-like regions
3. Scalable to very large codes
4. Parallel implementation straightforward

### 7. BP-OSD: Combining BP with Ordered Statistics

When BP fails to converge, **Ordered Statistics Decoding (OSD)** can rescue:

**BP-OSD Algorithm:**

1. Run BP for $T$ iterations
2. Extract soft information (LLRs) from BP
3. Order bits by reliability: $|L_1| \geq |L_2| \geq \cdots$
4. Apply OSD on least reliable bits

**Complexity:**

For order-$w$ OSD: $O(n^3 + n \cdot \binom{k}{w})$

This hybrid achieves excellent thresholds while remaining practical.

---

## Worked Examples

### Example 1: Factor Graph for 3-Qubit Code

**Problem:** Construct the factor graph for the 3-qubit bit-flip code.

**Solution:**

Stabilizers: $g_1 = ZZI$, $g_2 = IZZ$

**Variable Nodes:** $e_1, e_2, e_3$ (X-error on each qubit)

**Check Nodes:** $c_1$ (for $g_1$), $c_2$ (for $g_2$)

**Edges:**
- $c_1$ connects to $e_1, e_2$ (ZZI acts on qubits 1, 2)
- $c_2$ connects to $e_2, e_3$ (IZZ acts on qubits 2, 3)

**Factor Graph:**
```
     e1 ---- c1 ---- e2 ---- c2 ---- e3
```

**Factor Functions:**

For syndrome $s = (s_1, s_2)$:
- $f_{c_1}(e_1, e_2) = \mathbb{1}[e_1 \oplus e_2 = s_1]$
- $f_{c_2}(e_2, e_3) = \mathbb{1}[e_2 \oplus e_3 = s_2]$

### Example 2: BP Message Update

**Problem:** Compute one round of BP messages for the 3-qubit code with syndrome $s = (1, 0)$ and prior $p = 0.1$.

**Solution:**

**Initial LLRs:**
$$L^{(0)} = \log \frac{0.9}{0.1} = \log 9 \approx 2.20$$

**Variable-to-Check (round 1):**

No incoming check messages yet:
$$\lambda_{1 \to c_1} = L^{(0)} = 2.20$$
$$\lambda_{2 \to c_1} = L^{(0)} = 2.20$$
$$\lambda_{2 \to c_2} = L^{(0)} = 2.20$$
$$\lambda_{3 \to c_2} = L^{(0)} = 2.20$$

**Check-to-Variable (round 1):**

Using the tanh formula with $s_1 = 1, s_2 = 0$:

$$\Lambda_{c_1 \to 1} = 2 \tanh^{-1}\left((1-2 \cdot 1) \tanh(2.20/2)\right)$$
$$= 2 \tanh^{-1}(-\tanh(1.10)) = 2 \tanh^{-1}(-0.800) \approx -2.20$$

$$\Lambda_{c_1 \to 2} = 2 \tanh^{-1}(-\tanh(1.10)) \approx -2.20$$

$$\Lambda_{c_2 \to 2} = 2 \tanh^{-1}((1-0) \tanh(1.10)) \approx +2.20$$

$$\Lambda_{c_2 \to 3} = 2 \tanh^{-1}(\tanh(1.10)) \approx +2.20$$

**Beliefs after round 1:**

$$b_1 = L^{(0)} + \Lambda_{c_1 \to 1} = 2.20 - 2.20 = 0$$
$$b_2 = L^{(0)} + \Lambda_{c_1 \to 2} + \Lambda_{c_2 \to 2} = 2.20 - 2.20 + 2.20 = 2.20$$
$$b_3 = L^{(0)} + \Lambda_{c_2 \to 3} = 2.20 + 2.20 = 4.40$$

**Interpretation:**
- $b_1 = 0$: Uncertain about $e_1$
- $b_2 > 0$: Likely $e_2 = 0$
- $b_3 > 0$: Very likely $e_3 = 0$

This suggests error on qubit 1, which is correct for syndrome $(1, 0)$!

### Example 3: Min-Sum vs Sum-Product

**Problem:** Compare min-sum and sum-product for a degree-4 check with incoming LLRs $(3.0, 1.5, 2.0, 0.8)$ and syndrome 0.

**Solution:**

**Sum-Product:**
$$\Lambda = 2 \tanh^{-1}\left(\prod_{j} \tanh(\lambda_j/2)\right)$$

$$= 2 \tanh^{-1}(\tanh(1.5) \cdot \tanh(0.75) \cdot \tanh(1.0) \cdot \tanh(0.4))$$
$$= 2 \tanh^{-1}(0.905 \cdot 0.635 \cdot 0.762 \cdot 0.380)$$
$$= 2 \tanh^{-1}(0.166) = 0.335$$

Wait, we're computing the outgoing message to one variable. Let's compute $\Lambda_{c \to 1}$ (excluding $\lambda_1 = 3.0$):

$$\Lambda_{c \to 1} = 2 \tanh^{-1}(\tanh(0.75) \cdot \tanh(1.0) \cdot \tanh(0.4))$$
$$= 2 \tanh^{-1}(0.635 \cdot 0.762 \cdot 0.380) = 2 \tanh^{-1}(0.184) \approx 0.372$$

**Min-Sum:**
$$\Lambda_{c \to 1} = \text{sign}(1.5) \cdot \text{sign}(2.0) \cdot \text{sign}(0.8) \cdot \min(1.5, 2.0, 0.8) = +0.8$$

**Comparison:**
- Sum-product: 0.372
- Min-sum: 0.800

Min-sum overestimates confidence. Normalized min-sum with $\alpha = 0.5$:
$$\Lambda_{c \to 1}^{\text{norm}} = 0.5 \times 0.8 = 0.4$$

Much closer to sum-product!

---

## Practice Problems

### Level A: Direct Application

**A1.** Draw the factor graph for the $[[5, 1, 3]]$ code given its stabilizer generators.

**A2.** Compute the prior LLR for error probability $p = 0.05$.

**A3.** For a degree-3 check with incoming LLRs $(2.0, 1.5, 1.0)$ and syndrome 1, compute the min-sum outgoing message.

### Level B: Intermediate Analysis

**B1.** Prove that on a tree-structured factor graph, BP computes exact marginals after diameter iterations.

**B2.** Analyze the girth of the factor graph for the surface code. Identify all 4-cycles.

**B3.** Derive the tanh update rule from the sum-product formula for binary parity checks.

### Level C: Advanced Problems

**C1.** Design a BP schedule (order of message updates) that maximizes information flow on the surface code factor graph.

**C2.** Prove that min-sum is equivalent to sum-product in the limit of high SNR ($L \to \infty$).

**C3.** Implement and analyze BP-OSD for a quantum LDPC code. What order $w$ is needed to match MWPM threshold?

---

## Computational Lab: BP Decoder Implementation

```python
"""
Day 775 Computational Lab: Belief Propagation Decoding
Implementing sum-product and min-sum BP for stabilizer codes

This lab builds BP decoders from scratch and analyzes their
performance on various quantum codes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class FactorGraph:
    """
    Factor graph representation for BP decoding.
    """
    n_vars: int  # Number of variable nodes
    n_checks: int  # Number of check nodes
    var_to_checks: Dict[int, List[int]]  # Variable -> connected checks
    check_to_vars: Dict[int, List[int]]  # Check -> connected variables


def build_factor_graph(H: np.ndarray) -> FactorGraph:
    """
    Build factor graph from parity check matrix.

    Args:
        H: Binary parity check matrix (checks x variables)

    Returns:
        FactorGraph object
    """
    n_checks, n_vars = H.shape

    var_to_checks = defaultdict(list)
    check_to_vars = defaultdict(list)

    for c in range(n_checks):
        for v in range(n_vars):
            if H[c, v] == 1:
                var_to_checks[v].append(c)
                check_to_vars[c].append(v)

    return FactorGraph(
        n_vars=n_vars,
        n_checks=n_checks,
        var_to_checks=dict(var_to_checks),
        check_to_vars=dict(check_to_vars)
    )


class SumProductDecoder:
    """
    Sum-Product (Belief Propagation) decoder for binary codes.
    """

    def __init__(self, H: np.ndarray, max_iter: int = 50):
        """
        Initialize decoder.

        Args:
            H: Parity check matrix
            max_iter: Maximum BP iterations
        """
        self.H = H
        self.max_iter = max_iter
        self.graph = build_factor_graph(H)

    def decode(self, syndrome: np.ndarray, p_error: float,
               verbose: bool = False) -> Tuple[np.ndarray, bool]:
        """
        Decode syndrome using sum-product BP.

        Args:
            syndrome: Binary syndrome vector
            p_error: Prior error probability
            verbose: Print iteration info

        Returns:
            (decoded_error, converged)
        """
        n_vars = self.graph.n_vars
        n_checks = self.graph.n_checks

        # Initialize LLRs
        L_prior = np.log((1 - p_error) / p_error)

        # Messages: var -> check
        lambda_vc = {}
        for v in range(n_vars):
            for c in self.graph.var_to_checks.get(v, []):
                lambda_vc[(v, c)] = L_prior

        # Messages: check -> var
        Lambda_cv = {}
        for c in range(n_checks):
            for v in self.graph.check_to_vars.get(c, []):
                Lambda_cv[(c, v)] = 0.0

        # Iterative message passing
        for iteration in range(self.max_iter):
            # Check-to-variable messages
            for c in range(n_checks):
                vars_in_check = self.graph.check_to_vars.get(c, [])
                s_c = syndrome[c]

                for v in vars_in_check:
                    # Product of tanh of incoming messages
                    other_vars = [u for u in vars_in_check if u != v]
                    if not other_vars:
                        Lambda_cv[(c, v)] = 0.0
                        continue

                    tanh_prod = 1.0
                    for u in other_vars:
                        tanh_prod *= np.tanh(lambda_vc[(u, c)] / 2)

                    # Apply syndrome sign
                    tanh_prod *= (1 - 2 * s_c)

                    # Clip for numerical stability
                    tanh_prod = np.clip(tanh_prod, -0.9999, 0.9999)
                    Lambda_cv[(c, v)] = 2 * np.arctanh(tanh_prod)

            # Variable-to-check messages
            for v in range(n_vars):
                checks_on_var = self.graph.var_to_checks.get(v, [])

                for c in checks_on_var:
                    other_checks = [b for b in checks_on_var if b != c]
                    incoming_sum = sum(Lambda_cv[(b, v)] for b in other_checks)
                    lambda_vc[(v, c)] = L_prior + incoming_sum

            # Compute beliefs and check convergence
            beliefs = np.zeros(n_vars)
            for v in range(n_vars):
                checks_on_var = self.graph.var_to_checks.get(v, [])
                beliefs[v] = L_prior + sum(Lambda_cv[(c, v)] for c in checks_on_var)

            # Hard decision
            decoded = (beliefs < 0).astype(int)

            # Check if syndrome satisfied
            computed_syndrome = (self.H @ decoded) % 2
            if np.array_equal(computed_syndrome, syndrome):
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                return decoded, True

        if verbose:
            print(f"Did not converge after {self.max_iter} iterations")

        return decoded, False


class MinSumDecoder:
    """
    Min-Sum decoder - efficient approximation to sum-product.
    """

    def __init__(self, H: np.ndarray, max_iter: int = 50,
                 normalization: float = 0.75):
        """
        Initialize decoder.

        Args:
            H: Parity check matrix
            max_iter: Maximum iterations
            normalization: Scaling factor for min-sum (typically 0.7-0.8)
        """
        self.H = H
        self.max_iter = max_iter
        self.alpha = normalization
        self.graph = build_factor_graph(H)

    def decode(self, syndrome: np.ndarray, p_error: float,
               verbose: bool = False) -> Tuple[np.ndarray, bool]:
        """
        Decode syndrome using min-sum BP.

        Args:
            syndrome: Binary syndrome vector
            p_error: Prior error probability
            verbose: Print iteration info

        Returns:
            (decoded_error, converged)
        """
        n_vars = self.graph.n_vars
        n_checks = self.graph.n_checks

        # Initialize LLRs
        L_prior = np.log((1 - p_error) / p_error)

        # Messages
        lambda_vc = {}
        Lambda_cv = {}

        for v in range(n_vars):
            for c in self.graph.var_to_checks.get(v, []):
                lambda_vc[(v, c)] = L_prior

        for c in range(n_checks):
            for v in self.graph.check_to_vars.get(c, []):
                Lambda_cv[(c, v)] = 0.0

        # Iterative message passing
        for iteration in range(self.max_iter):
            # Check-to-variable messages (min-sum)
            for c in range(n_checks):
                vars_in_check = self.graph.check_to_vars.get(c, [])
                s_c = syndrome[c]

                for v in vars_in_check:
                    other_vars = [u for u in vars_in_check if u != v]
                    if not other_vars:
                        Lambda_cv[(c, v)] = 0.0
                        continue

                    # Product of signs
                    sign_prod = (1 - 2 * s_c)
                    for u in other_vars:
                        sign_prod *= np.sign(lambda_vc[(u, c)]) if lambda_vc[(u, c)] != 0 else 1

                    # Minimum of magnitudes
                    min_mag = min(abs(lambda_vc[(u, c)]) for u in other_vars)

                    Lambda_cv[(c, v)] = self.alpha * sign_prod * min_mag

            # Variable-to-check messages
            for v in range(n_vars):
                checks_on_var = self.graph.var_to_checks.get(v, [])

                for c in checks_on_var:
                    other_checks = [b for b in checks_on_var if b != c]
                    incoming_sum = sum(Lambda_cv[(b, v)] for b in other_checks)
                    lambda_vc[(v, c)] = L_prior + incoming_sum

            # Compute beliefs
            beliefs = np.zeros(n_vars)
            for v in range(n_vars):
                checks_on_var = self.graph.var_to_checks.get(v, [])
                beliefs[v] = L_prior + sum(Lambda_cv[(c, v)] for c in checks_on_var)

            # Hard decision
            decoded = (beliefs < 0).astype(int)

            # Check convergence
            computed_syndrome = (self.H @ decoded) % 2
            if np.array_equal(computed_syndrome, syndrome):
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                return decoded, True

        if verbose:
            print(f"Did not converge after {self.max_iter} iterations")

        return decoded, False


def create_repetition_code(n: int) -> np.ndarray:
    """Create parity check matrix for n-bit repetition code."""
    H = np.zeros((n - 1, n), dtype=int)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def create_surface_code_H(d: int) -> np.ndarray:
    """
    Create simplified parity check matrix for surface code.
    (X-stabilizers only for simplicity)
    """
    n_data = d * d
    n_checks = (d - 1) * d

    H = np.zeros((n_checks, n_data), dtype=int)

    check_idx = 0
    for i in range(d - 1):
        for j in range(d):
            # Plaquette at (i, j)
            H[check_idx, i * d + j] = 1
            H[check_idx, (i + 1) * d + j] = 1
            if j > 0:
                H[check_idx, i * d + (j - 1)] = 1
            if j < d - 1:
                H[check_idx, (i + 1) * d + (j + 1)] = 1
            check_idx += 1

    # Simplify: just adjacent pairs
    H = np.zeros(((d-1)*d, d*d), dtype=int)
    check_idx = 0
    for i in range(d):
        for j in range(d - 1):
            H[check_idx, i * d + j] = 1
            H[check_idx, i * d + j + 1] = 1
            check_idx += 1

    return H


def test_repetition_code():
    """Test BP on repetition code."""
    print("=" * 60)
    print("BP on 7-bit Repetition Code")
    print("=" * 60)

    n = 7
    H = create_repetition_code(n)

    print(f"Parity check matrix H ({H.shape[0]} x {H.shape[1]}):")
    print(H)

    # Test cases
    test_cases = [
        ("No error", np.zeros(n, dtype=int), 0.1),
        ("Single error (bit 3)", np.array([0,0,0,1,0,0,0]), 0.1),
        ("Double error", np.array([0,1,0,0,0,1,0]), 0.1),
    ]

    sp_decoder = SumProductDecoder(H, max_iter=20)
    ms_decoder = MinSumDecoder(H, max_iter=20)

    for name, error, p in test_cases:
        syndrome = (H @ error) % 2
        print(f"\n{name}:")
        print(f"  True error: {error}")
        print(f"  Syndrome: {syndrome}")

        decoded_sp, conv_sp = sp_decoder.decode(syndrome, p, verbose=False)
        decoded_ms, conv_ms = ms_decoder.decode(syndrome, p, verbose=False)

        print(f"  Sum-Product: {decoded_sp} (converged: {conv_sp})")
        print(f"  Min-Sum: {decoded_ms} (converged: {conv_ms})")


def analyze_convergence():
    """Analyze BP convergence vs iterations."""
    print("\n" + "=" * 60)
    print("BP Convergence Analysis")
    print("=" * 60)

    n = 15
    H = create_repetition_code(n)
    p_error = 0.1
    n_trials = 200

    max_iters = [5, 10, 20, 50, 100]
    convergence_rates = []
    error_rates = []

    for max_iter in max_iters:
        decoder = SumProductDecoder(H, max_iter=max_iter)
        converged_count = 0
        correct_count = 0

        for _ in range(n_trials):
            # Generate random error
            error = (np.random.random(n) < p_error).astype(int)
            syndrome = (H @ error) % 2

            decoded, converged = decoder.decode(syndrome, p_error)

            if converged:
                converged_count += 1
            if np.array_equal(decoded, error):
                correct_count += 1

        convergence_rates.append(converged_count / n_trials)
        error_rates.append(1 - correct_count / n_trials)

        print(f"max_iter={max_iter:3d}: convergence={converged_count/n_trials:.2%}, "
              f"error_rate={1-correct_count/n_trials:.2%}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(max_iters, convergence_rates, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Maximum Iterations', fontsize=12)
    ax1.set_ylabel('Convergence Rate', fontsize=12)
    ax1.set_title('BP Convergence vs Iterations', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    ax2.semilogy(max_iters, error_rates, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Maximum Iterations', fontsize=12)
    ax2.set_ylabel('Decoding Error Rate', fontsize=12)
    ax2.set_title('Error Rate vs Iterations', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bp_convergence.png', dpi=150)
    plt.show()


def compare_sum_product_min_sum():
    """Compare sum-product and min-sum performance."""
    print("\n" + "=" * 60)
    print("Sum-Product vs Min-Sum Comparison")
    print("=" * 60)

    n = 21
    H = create_repetition_code(n)
    error_rates = np.linspace(0.01, 0.2, 10)
    n_trials = 300

    sp_errors = []
    ms_errors = []
    ms_norm_errors = []

    sp_decoder = SumProductDecoder(H, max_iter=50)
    ms_decoder = MinSumDecoder(H, max_iter=50, normalization=1.0)
    ms_norm_decoder = MinSumDecoder(H, max_iter=50, normalization=0.75)

    for p in error_rates:
        sp_wrong = 0
        ms_wrong = 0
        ms_norm_wrong = 0

        for _ in range(n_trials):
            error = (np.random.random(n) < p).astype(int)
            syndrome = (H @ error) % 2

            decoded_sp, _ = sp_decoder.decode(syndrome, p)
            decoded_ms, _ = ms_decoder.decode(syndrome, p)
            decoded_ms_norm, _ = ms_norm_decoder.decode(syndrome, p)

            # Check for logical error (for repetition code: wrong majority)
            if np.sum(decoded_sp) % 2 != np.sum(error) % 2:
                sp_wrong += 1
            if np.sum(decoded_ms) % 2 != np.sum(error) % 2:
                ms_wrong += 1
            if np.sum(decoded_ms_norm) % 2 != np.sum(error) % 2:
                ms_norm_wrong += 1

        sp_errors.append(sp_wrong / n_trials)
        ms_errors.append(ms_wrong / n_trials)
        ms_norm_errors.append(ms_norm_wrong / n_trials)

        print(f"p={p:.2f}: SP={sp_wrong/n_trials:.3f}, "
              f"MS={ms_wrong/n_trials:.3f}, MS-norm={ms_norm_wrong/n_trials:.3f}")

    # Plot
    plt.figure(figsize=(10, 7))
    plt.semilogy(error_rates * 100, sp_errors, 'b-o', label='Sum-Product', linewidth=2)
    plt.semilogy(error_rates * 100, ms_errors, 'r-s', label='Min-Sum', linewidth=2)
    plt.semilogy(error_rates * 100, ms_norm_errors, 'g-^',
                label='Min-Sum (normalized)', linewidth=2)

    plt.xlabel('Physical Error Rate (%)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('Sum-Product vs Min-Sum Decoder Performance', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sp_vs_ms.png', dpi=150)
    plt.show()


def analyze_cycles():
    """Analyze the impact of cycles on BP performance."""
    print("\n" + "=" * 60)
    print("Cycle Analysis in Factor Graphs")
    print("=" * 60)

    # Create matrices with different cycle structures
    # Repetition code: long cycles (girth = 2n)
    H_rep = create_repetition_code(8)

    # Dense code: many short cycles
    H_dense = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
    ], dtype=int)

    print("Testing codes with different cycle structures...")

    codes = [
        ("Repetition (tree-like)", H_rep),
        ("Dense (short cycles)", H_dense),
    ]

    p_error = 0.1
    n_trials = 200

    for name, H in codes:
        graph = build_factor_graph(H)
        decoder = SumProductDecoder(H, max_iter=100)

        converged_count = 0
        correct_count = 0

        for _ in range(n_trials):
            error = (np.random.random(H.shape[1]) < p_error).astype(int)
            syndrome = (H @ error) % 2

            decoded, converged = decoder.decode(syndrome, p_error)

            if converged:
                converged_count += 1
            # Check syndrome match
            if np.array_equal((H @ decoded) % 2, syndrome):
                correct_count += 1

        print(f"{name}:")
        print(f"  Convergence rate: {converged_count/n_trials:.2%}")
        print(f"  Valid codeword rate: {correct_count/n_trials:.2%}")
        print(f"  Check degrees: {[len(graph.check_to_vars.get(c, [])) for c in range(graph.n_checks)]}")


def timing_benchmark():
    """Benchmark BP decoder timing."""
    print("\n" + "=" * 60)
    print("BP Decoder Timing Benchmark")
    print("=" * 60)

    sizes = [10, 20, 50, 100, 200]
    sp_times = []
    ms_times = []

    for n in sizes:
        H = create_repetition_code(n)

        sp_decoder = SumProductDecoder(H, max_iter=50)
        ms_decoder = MinSumDecoder(H, max_iter=50)

        # Generate test syndrome
        error = (np.random.random(n) < 0.1).astype(int)
        syndrome = (H @ error) % 2

        # Time sum-product
        start = time.time()
        for _ in range(100):
            sp_decoder.decode(syndrome, 0.1)
        sp_time = (time.time() - start) / 100

        # Time min-sum
        start = time.time()
        for _ in range(100):
            ms_decoder.decode(syndrome, 0.1)
        ms_time = (time.time() - start) / 100

        sp_times.append(sp_time)
        ms_times.append(ms_time)

        print(f"n={n:3d}: SP={sp_time*1000:.3f}ms, MS={ms_time*1000:.3f}ms, "
              f"ratio={sp_time/ms_time:.2f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(sizes, [t*1000 for t in sp_times], 'b-o',
               label='Sum-Product', linewidth=2, markersize=8)
    plt.loglog(sizes, [t*1000 for t in ms_times], 'r-s',
               label='Min-Sum', linewidth=2, markersize=8)

    plt.xlabel('Code Length (n)', fontsize=12)
    plt.ylabel('Decode Time (ms)', fontsize=12)
    plt.title('BP Decoder Timing vs Code Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bp_timing.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    test_repetition_code()
    analyze_convergence()
    compare_sum_product_min_sum()
    analyze_cycles()
    timing_benchmark()

    print("\n" + "=" * 60)
    print("Lab Complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Variable-to-Check | $$\mu_{i \to a}(e_i) = f_i(e_i) \prod_{b \neq a} \mu_{b \to i}(e_i)$$ |
| Check-to-Variable | $$\mu_{a \to i}(e_i) = \sum_{e_{\partial a \setminus i}} f_a(e_{\partial a}) \prod_{j \neq i} \mu_{j \to a}(e_j)$$ |
| LLR Check Update | $$\Lambda_{a \to i} = 2 \tanh^{-1}\left((1-2s_a)\prod_{j \neq i}\tanh(\lambda_{j \to a}/2)\right)$$ |
| Min-Sum | $$\Lambda_{a \to i} = \alpha \cdot \text{sign} \cdot \min_j \vert\lambda_{j \to a}\vert$$ |
| BP Complexity | $$O(n \cdot d_{\max} \cdot T)$$ per decode |

### Key Takeaways

1. **Factor graphs represent constraints**: Variables (errors) and checks (stabilizers) as bipartite graph
2. **BP passes messages iteratively**: Updates beliefs based on local information
3. **LLR formulation is numerically stable**: Log-domain prevents underflow
4. **Min-sum is practical**: Hardware-friendly approximation with normalization
5. **Short cycles hurt performance**: Quantum codes often have problematic girth
6. **BP-OSD combines strengths**: BP soft information + OSD for hard cases

---

## Daily Checklist

- [ ] Constructed factor graphs for stabilizer codes
- [ ] Derived sum-product message updates
- [ ] Implemented min-sum approximation
- [ ] Analyzed cycle impact on convergence
- [ ] Tested BP on simple codes
- [ ] Compared sum-product vs min-sum performance
- [ ] Completed practice problems (at least Level A and B)

---

## Preview: Day 776

Tomorrow we tackle **Real-Time Decoding Constraints**, the practical challenges of implementing decoders in actual quantum computers. We'll analyze latency requirements, FPGA implementations, and strategies for avoiding decoder backlog.

Key questions for tomorrow:
- What latency is required for fault-tolerant computation?
- How do we pipeline decoding across syndrome rounds?
- What hardware architectures enable real-time decoding?

---

*Day 775 of 2184 | Week 111 | Month 28 | Year 2: Advanced Quantum Science*
