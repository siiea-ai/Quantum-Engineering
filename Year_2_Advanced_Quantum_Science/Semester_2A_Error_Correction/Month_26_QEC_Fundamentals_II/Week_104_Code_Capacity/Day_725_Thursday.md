# Day 725: LDPC Code Capacity

## Overview

**Date:** Day 725 of 1008
**Week:** 104 (Code Capacity)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Low-Density Parity-Check Codes and Capacity-Approaching Constructions

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | Classical LDPC and belief propagation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hrs | Quantum LDPC constructions |
| Evening | 7:00 PM - 8:00 PM | 1 hr | Good qLDPC codes |

---

## Learning Objectives

By the end of this day, you should be able to:

1. **Define** LDPC codes and their parity-check matrices
2. **Explain** belief propagation decoding
3. **Describe** the CSS-LDPC construction
4. **State** the parameters of good qLDPC codes
5. **Understand** why qLDPC codes can approach capacity
6. **Compare** surface codes to qLDPC codes

---

## Core Content

### 1. Classical LDPC Codes

#### Definition

A **Low-Density Parity-Check (LDPC)** code is a linear code defined by a sparse parity-check matrix $H$.

**Sparse means:**
- Each row has $O(1)$ ones (bounded check weight)
- Each column has $O(1)$ ones (bounded variable degree)

**Notation:** $(d_v, d_c)$-regular LDPC:
- Each variable node has degree $d_v$
- Each check node has degree $d_c$

#### Tanner Graph Representation

```
Variable nodes (bits):    ○ ○ ○ ○ ○ ○
                          │╲│╱│╲│╱│╲│
Check nodes (parities):    □   □   □
```

Each edge connects a variable to a check it participates in.

#### Why LDPC?

1. **Efficient encoding:** $O(n)$ or $O(n \log n)$ operations
2. **Efficient decoding:** Belief propagation in $O(n)$ per iteration
3. **Near-capacity performance:** Approach Shannon limit

#### Classical Capacity Achievement

**Theorem (Richardson-Urbanke):**
Properly designed LDPC codes can achieve capacity of the binary symmetric channel under belief propagation decoding.

**Key:** Irregular degree distributions optimized for channel.

---

### 2. Belief Propagation Decoding

#### The Algorithm

**Setup:** Tanner graph with variable nodes $v$ and check nodes $c$.

**Messages:**
- $\mu_{v \to c}$: Message from variable to check
- $\mu_{c \to v}$: Message from check to variable

**Initialization:**
$$\mu_{v \to c}^{(0)} = \log \frac{P(v=0|y)}{P(v=1|y)}$$

based on channel observation $y$.

**Variable to check update:**
$$\mu_{v \to c}^{(t+1)} = \mu_v^{\text{channel}} + \sum_{c' \neq c} \mu_{c' \to v}^{(t)}$$

**Check to variable update:**
$$\mu_{c \to v}^{(t+1)} = 2 \tanh^{-1}\left(\prod_{v' \neq v} \tanh\frac{\mu_{v' \to c}^{(t)}}{2}\right)$$

**Decision:**
After convergence, decide $\hat{v} = 0$ if total message > 0.

#### Complexity

- **Per iteration:** $O(|E|)$ where $|E|$ is number of edges
- **For LDPC:** $|E| = O(n)$, so $O(n)$ per iteration
- **Total:** $O(n \cdot \text{iterations})$, typically $O(n \log n)$

---

### 3. Quantum LDPC Codes

#### The Challenge

**Classical LDPC:** Sparse $H$ matrix

**Quantum:** Need $H_X$ and $H_Z$ such that $H_X H_Z^T = 0$

**Problem:** Commutativity constraint makes construction harder!

#### CSS-LDPC Construction

**From classical codes $C_1, C_2$ with $C_2^\perp \subseteq C_1$:**

$$H_X = H_2 \quad (\text{parity check of } C_2)$$
$$H_Z = H_1 \quad (\text{parity check of } C_1)$$

**Commutativity:** $H_X H_Z^T = H_2 H_1^T = 0$ iff $C_2^\perp \subseteq C_1$

**Challenge:** Both $H_X$ and $H_Z$ sparse AND commuting is restrictive.

#### Hypergraph Product Construction

**Tillich-Zémor (2009):**

Given classical codes $C_1 = [n_1, k_1, d_1]$ and $C_2 = [n_2, k_2, d_2]$:

$$H_X = \begin{pmatrix} H_1 \otimes I_{n_2} & I_{r_1} \otimes H_2^T \end{pmatrix}$$
$$H_Z = \begin{pmatrix} I_{n_1} \otimes H_2 & H_1^T \otimes I_{r_2} \end{pmatrix}$$

**Parameters:**
- $n = n_1 n_2 + r_1 r_2$ (physical qubits)
- $k = k_1 k_2$ (logical qubits)
- $d = \min(d_1, d_2)$ (distance)

**Trade-off:** Distance limited by classical code distances.

---

### 4. Good qLDPC Codes: The Breakthrough

#### What is a "Good" Code?

**Definition:** A code family $\{[[n_i, k_i, d_i]]\}$ is **good** if:
$$k_i = \Theta(n_i) \quad \text{and} \quad d_i = \Theta(n_i)$$

Both rate and relative distance are constant!

#### Historical Context

**Before 2020:**
- Best known: $k = \Theta(n)$ OR $d = \Theta(n)$, not both
- Surface codes: $k = O(1)$, $d = \Theta(\sqrt{n})$
- Hypergraph product: $d = O(\sqrt{n})$

**The Question:** Do good qLDPC codes exist?

#### Breakthroughs (2020-2022)

**1. Panteleev-Kalachev (2020):**
First construction of asymptotically good qLDPC codes!
- $k = \Theta(n)$
- $d = \Theta(n)$
- Sparse parity checks

**2. Leverrier-Zémor (2022):**
Good codes from high-dimensional expanders

**3. Dinur et al. (2022):**
c³ construction (constant rate, constant relative distance)

#### Parameters of Good qLDPC

| Property | Value |
|----------|-------|
| Rate $k/n$ | $\Theta(1)$ (e.g., 0.1) |
| Relative distance $d/n$ | $\Theta(1)$ (e.g., 0.01) |
| Check weight | $O(1)$ (e.g., 10) |
| Decoding | Polynomial time |

---

### 5. Why Good qLDPC Matters for Capacity

#### Approaching the Hashing Bound

**For rate $R$ and distance $d$:**
- Need $R < Q(\mathcal{N})$ (capacity constraint)
- Need $d$ large enough to correct errors

**Good qLDPC:**
- $R = \Theta(1)$ can be tuned close to capacity
- $d = \Theta(n)$ enables correction of $\Theta(n)$ errors

#### Implications for Fault Tolerance

**Surface codes:**
- Overhead: $O(d^2)$ physical per logical qubit
- For distance $d = 100$: need 10,000 physical qubits

**Good qLDPC:**
- Overhead: $O(1)$ physical per logical qubit
- Much more efficient!

**Caveat:** Decoding complexity and connectivity requirements differ.

#### Capacity Achievement

**Theorem (informal):**
Good qLDPC codes can achieve rates arbitrarily close to the hashing bound with polynomial decoding complexity.

This is the quantum analog of classical LDPC capacity achievement!

---

### 6. Comparison: Surface Codes vs qLDPC

#### Parameter Comparison

| Property | Surface Code | Good qLDPC |
|----------|--------------|------------|
| Rate $k/n$ | $O(1/d^2)$ | $\Theta(1)$ |
| Distance | $d$ | $\Theta(n)$ |
| Check weight | 4 | $O(1)$ (larger, ~10-20) |
| Connectivity | 2D local | Non-local |
| Decoding | MWPM, $O(n^3)$ | BP, $O(n \log n)$ |
| Threshold | ~1% | ~10%+ (code capacity) |

#### Trade-offs

**Surface codes excel at:**
- 2D local connectivity (hardware-friendly)
- Well-understood decoders
- Lower check weight

**qLDPC codes excel at:**
- Encoding efficiency
- Asymptotic scaling
- Capacity approach

#### The Future

Likely: **Hybrid approaches**
- Surface code for near-term devices
- qLDPC for large-scale, fault-tolerant QC
- Possible: surface-like qLDPC with improved locality

---

### 7. Decoding qLDPC Codes

#### Belief Propagation for Quantum

**Challenge:** Degeneracy — multiple errors give same syndrome

**Adaptation:**
- BP on Tanner graph of $H_X$ for Z-errors
- BP on Tanner graph of $H_Z$ for X-errors
- Handle degeneracy via coset structure

#### Performance

**For good qLDPC:**
- BP achieves threshold close to hashing bound
- May need post-processing for degeneracy

**Comparison:**
| Decoder | Complexity | Threshold |
|---------|------------|-----------|
| ML (optimal) | Exponential | Capacity |
| BP | $O(n \log n)$ | Near-capacity |
| MWPM (surface) | $O(n^3)$ | ~1% |

---

## Worked Examples

### Example 1: Classical LDPC Parameters

**Problem:** A (3,6)-regular LDPC code has $n = 1000$ bits and $m = 500$ checks. Calculate the rate.

**Solution:**

**Counting edges:**
- From variable side: $3n = 3000$ edges
- From check side: $6m = 3000$ edges ✓

**Rate:**
For LDPC, rate $R \geq 1 - m/n = 1 - 500/1000 = 0.5$.

(Equality if all rows independent.)

---

### Example 2: Hypergraph Product Parameters

**Problem:** Two classical $[15, 5, 5]$ codes are used in hypergraph product construction. Find the quantum code parameters.

**Solution:**

**Classical code parameters:**
- $n_1 = n_2 = 15$
- $k_1 = k_2 = 5$
- $d_1 = d_2 = 5$
- $r_1 = r_2 = n - k = 10$ (check bits)

**Quantum code:**
- $n = n_1 n_2 + r_1 r_2 = 15 \times 15 + 10 \times 10 = 225 + 100 = 325$
- $k = k_1 k_2 = 5 \times 5 = 25$
- $d = \min(d_1, d_2) = 5$

**Result:** $[[325, 25, 5]]$ code with rate $k/n = 25/325 \approx 0.077$.

**Note:** Distance is limited to classical distance (not linear in $n$).

---

### Example 3: Good qLDPC Efficiency

**Problem:** Compare the overhead of surface code vs good qLDPC for 1000 logical qubits at distance 50.

**Solution:**

**Surface code:**
- Physical per logical: $\approx 2d^2 = 2 \times 50^2 = 5000$
- Total: $1000 \times 5000 = 5,000,000$ physical qubits

**Good qLDPC (rate 0.1):**
- Physical per logical: $1/0.1 = 10$
- For distance 50 with $d = \Theta(n)$: need $n \approx 500$ per block
- With $k/n = 0.1$: $k = 50$ logical per block
- Blocks needed: $1000/50 = 20$
- Total: $20 \times 500 = 10,000$ physical qubits

**Ratio:** Surface needs 500× more physical qubits!

**Caveat:** qLDPC requires non-local connectivity, which may be harder to implement.

---

## Practice Problems

### Direct Application

1. **Problem 1:** A (4,8)-regular LDPC code has 2000 variable nodes. How many check nodes?

2. **Problem 2:** Using hypergraph product with two $[7, 4, 3]$ Hamming codes, what are the quantum code parameters?

3. **Problem 3:** A good qLDPC code has rate 0.05 and relative distance 0.02. For 10000 physical qubits, how many logical qubits and what distance?

### Intermediate

4. **Problem 4:** Prove that the hypergraph product of two LDPC codes is also LDPC.

5. **Problem 5:** Why can't the hypergraph product achieve linear distance from classical codes with constant distance?

6. **Problem 6:** Design a belief propagation schedule for a quantum code that handles X and Z errors separately.

### Challenging

7. **Problem 7:** Prove that any CSS code with $H_X$ and $H_Z$ both sparse must have $d = O(\sqrt{n})$ unless using special structure (Bravyi-Terhal bound).

8. **Problem 8:** Explain how expander graphs enable good qLDPC constructions (high-level).

9. **Problem 9:** Analyze the trade-off between check weight and error correction capability in qLDPC codes.

---

## Computational Lab

```python
"""
Day 725: LDPC Code Capacity
LDPC constructions and belief propagation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from typing import Tuple, List, Optional

def create_regular_ldpc(n: int, dv: int, dc: int) -> np.ndarray:
    """
    Create a (dv, dc)-regular LDPC parity check matrix.

    Parameters:
    -----------
    n : int
        Number of variable nodes (bits)
    dv : int
        Variable node degree
    dc : int
        Check node degree
    """
    # Number of check nodes
    m = n * dv // dc

    # Create edge list
    edges_per_var = dv
    edges_per_check = dc

    H = np.zeros((m, n), dtype=int)

    # Simple regular construction (may not be optimal)
    for i in range(n):
        # Connect variable i to dv checks
        checks = np.random.choice(m, size=dv, replace=False)
        for c in checks:
            H[c, i] = 1

    # This may not be exactly regular; for demo purposes
    return H

def belief_propagation_bsc(H: np.ndarray, y: np.ndarray, p: float,
                           max_iter: int = 50) -> Tuple[np.ndarray, bool]:
    """
    Belief propagation decoder for BSC.

    Parameters:
    -----------
    H : np.ndarray
        Parity check matrix (m x n)
    y : np.ndarray
        Received bits (length n)
    p : float
        Channel crossover probability
    max_iter : int
        Maximum iterations

    Returns:
    --------
    decoded : np.ndarray
        Decoded codeword
    success : bool
        Whether decoding succeeded
    """
    m, n = H.shape

    # Log-likelihood ratio from channel
    llr_channel = np.log((1-p)/p) * (1 - 2*y)

    # Initialize messages
    # mu_vc[c, v] = message from v to c
    # mu_cv[c, v] = message from c to v
    mu_vc = np.zeros((m, n))
    mu_cv = np.zeros((m, n))

    # Initialize with channel LLRs
    for c in range(m):
        for v in range(n):
            if H[c, v]:
                mu_vc[c, v] = llr_channel[v]

    for iteration in range(max_iter):
        # Check to variable messages
        for c in range(m):
            vars_in_check = np.where(H[c, :] > 0)[0]
            for v in vars_in_check:
                # Product of tanh of other messages
                prod = 1.0
                for v2 in vars_in_check:
                    if v2 != v:
                        prod *= np.tanh(mu_vc[c, v2] / 2)
                # Clip to avoid numerical issues
                prod = np.clip(prod, -0.9999, 0.9999)
                mu_cv[c, v] = 2 * np.arctanh(prod)

        # Variable to check messages
        for v in range(n):
            checks_of_var = np.where(H[:, v] > 0)[0]
            for c in checks_of_var:
                # Sum of other messages plus channel
                total = llr_channel[v]
                for c2 in checks_of_var:
                    if c2 != c:
                        total += mu_cv[c2, v]
                mu_vc[c, v] = total

        # Make decisions
        llr_total = llr_channel.copy()
        for v in range(n):
            checks_of_var = np.where(H[:, v] > 0)[0]
            for c in checks_of_var:
                llr_total[v] += mu_cv[c, v]

        decoded = (llr_total < 0).astype(int)

        # Check if valid codeword
        syndrome = (H @ decoded) % 2
        if np.all(syndrome == 0):
            return decoded, True

    return decoded, False

def hypergraph_product(H1: np.ndarray, H2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct quantum code via hypergraph product.

    Parameters:
    -----------
    H1, H2 : np.ndarray
        Classical parity check matrices

    Returns:
    --------
    HX, HZ : np.ndarray
        Quantum parity check matrices
    """
    r1, n1 = H1.shape
    r2, n2 = H2.shape

    # HX = [H1 ⊗ I_n2 | I_r1 ⊗ H2^T]
    # HZ = [I_n1 ⊗ H2 | H1^T ⊗ I_r2]

    I_n1 = np.eye(n1, dtype=int)
    I_n2 = np.eye(n2, dtype=int)
    I_r1 = np.eye(r1, dtype=int)
    I_r2 = np.eye(r2, dtype=int)

    HX_left = np.kron(H1, I_n2)
    HX_right = np.kron(I_r1, H2.T)
    HX = np.hstack([HX_left, HX_right])

    HZ_left = np.kron(I_n1, H2)
    HZ_right = np.kron(H1.T, I_r2)
    HZ = np.hstack([HZ_left, HZ_right])

    return HX.astype(int), HZ.astype(int)

def analyze_hypergraph_product(H1: np.ndarray, H2: np.ndarray):
    """Analyze parameters of hypergraph product code."""
    r1, n1 = H1.shape
    r2, n2 = H2.shape

    k1 = n1 - np.linalg.matrix_rank(H1)
    k2 = n2 - np.linalg.matrix_rank(H2)

    HX, HZ = hypergraph_product(H1, H2)

    n_quantum = n1 * n2 + r1 * r2
    # k = k1 * k2 for hypergraph product
    k_quantum = k1 * k2

    # Check commutativity
    comm = (HX @ HZ.T) % 2
    commutes = np.all(comm == 0)

    return {
        'n': n_quantum,
        'k': k_quantum,
        'n1': n1, 'k1': k1, 'r1': r1,
        'n2': n2, 'k2': k2, 'r2': r2,
        'HX_shape': HX.shape,
        'HZ_shape': HZ.shape,
        'commutes': commutes,
        'rate': k_quantum / n_quantum
    }

def plot_ldpc_performance():
    """Simulate and plot LDPC performance."""
    # Simple simulation with small code
    n = 100
    dv, dc = 3, 6
    H = create_regular_ldpc(n, dv, dc)

    # Ensure H has full row rank (approximately)
    m = H.shape[0]
    k = n - np.linalg.matrix_rank(H)

    print(f"Created ({dv},{dc})-regular LDPC code: [{n}, ~{k}]")

    # Simulate at different error rates
    p_values = np.linspace(0.01, 0.15, 10)
    block_error_rates = []

    num_trials = 100

    for p in p_values:
        errors = 0
        for _ in range(num_trials):
            # Generate random codeword (all-zeros for simplicity)
            codeword = np.zeros(n, dtype=int)

            # Add BSC noise
            noise = (np.random.random(n) < p).astype(int)
            received = (codeword + noise) % 2

            # Decode
            decoded, success = belief_propagation_bsc(H, received, p)

            if not success or not np.array_equal(decoded, codeword):
                errors += 1

        block_error_rates.append(errors / num_trials)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(p_values * 100, block_error_rates, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=11, color='red', linestyle='--', label='BSC capacity threshold')

    ax.set_xlabel('Channel Error Rate (%)')
    ax.set_ylabel('Block Error Rate')
    ax.set_title(f'LDPC ({dv},{dc})-regular Code Performance (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ldpc_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ldpc_performance.png")

def compare_code_families():
    """Compare different code families."""
    print("=" * 70)
    print("Code Family Comparison")
    print("=" * 70)

    # Surface code scaling
    print("\nSurface Code Scaling:")
    print(f"{'d':<10} {'n':<15} {'k':<10} {'Rate':<15}")
    print("-" * 50)
    for d in [3, 5, 7, 11, 15, 21]:
        n = 2 * d**2 - 1  # Approximate
        k = 1
        rate = k / n
        print(f"{d:<10} {n:<15} {k:<10} {rate:<15.4f}")

    # Hypergraph product scaling
    print("\nHypergraph Product Scaling (from Hamming codes):")
    print(f"{'Classical':<15} {'n_q':<10} {'k_q':<10} {'Rate':<10}")
    print("-" * 50)

    # Hamming codes [2^r-1, 2^r-1-r, 3]
    for r in [3, 4, 5, 6]:
        n_c = 2**r - 1
        k_c = n_c - r
        d_c = 3

        # Hypergraph product with itself
        n_q = n_c**2 + r**2
        k_q = k_c**2
        rate = k_q / n_q
        print(f"[{n_c},{k_c},{d_c}]     {n_q:<10} {k_q:<10} {rate:<10.4f}")

    # Good qLDPC (theoretical)
    print("\nGood qLDPC (Theoretical):")
    print(f"{'n':<15} {'k (rate=0.1)':<15} {'d (rel=0.02)':<15}")
    print("-" * 50)
    for n in [1000, 10000, 100000, 1000000]:
        k = int(0.1 * n)
        d = int(0.02 * n)
        print(f"{n:<15} {k:<15} {d:<15}")

def main():
    print("=" * 70)
    print("LDPC Code Capacity Analysis")
    print("=" * 70)

    # Classical LDPC example
    print("\n1. Classical LDPC Code Example")
    print("-" * 50)

    H = create_regular_ldpc(50, 3, 6)
    m, n = H.shape
    row_weights = np.sum(H, axis=1)
    col_weights = np.sum(H, axis=0)

    print(f"Matrix size: {m} x {n}")
    print(f"Average row weight: {np.mean(row_weights):.2f}")
    print(f"Average column weight: {np.mean(col_weights):.2f}")
    print(f"Sparsity: {100 * np.sum(H) / (m * n):.2f}%")

    # Hypergraph product example
    print("\n2. Hypergraph Product Example")
    print("-" * 50)

    # Small Hamming code H = [7, 4, 3]
    H_hamming = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])

    params = analyze_hypergraph_product(H_hamming, H_hamming)
    print(f"Classical code: [{params['n1']}, {params['k1']}]")
    print(f"Quantum code: [[{params['n']}, {params['k']}]]")
    print(f"Rate: {params['rate']:.4f}")
    print(f"Commutativity check: {'PASS' if params['commutes'] else 'FAIL'}")

    # Code family comparison
    compare_code_families()

    # Simulate BP
    print("\n3. Belief Propagation Simulation")
    print("-" * 50)
    print("Running simulations... (this may take a moment)")
    plot_ldpc_performance()

    # Summary table
    print("\n4. Summary: Capacity Approach")
    print("-" * 50)
    print("""
    Code Family         | Rate      | Distance  | Decoding  | Capacity
    --------------------|-----------|-----------|-----------|----------
    Random stabilizer   | Optimal   | Optimal   | Exp       | 100%
    Surface code        | O(1/d²)   | d         | Poly      | ~0%
    Hypergraph product  | Θ(1)      | O(√n)     | Poly      | Moderate
    Good qLDPC          | Θ(1)      | Θ(n)      | Poly      | ~100%

    Key insight: Good qLDPC codes achieve BOTH optimal rate AND
    optimal distance with polynomial decoding — the holy grail!
    """)

if __name__ == "__main__":
    main()
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **LDPC** | Codes with sparse parity-check matrix |
| **Belief propagation** | Iterative message-passing decoder |
| **Hypergraph product** | qLDPC from classical LDPC |
| **Good qLDPC** | $k, d = \Theta(n)$ simultaneously |
| **Capacity approach** | Good qLDPC can approach hashing bound |

### Key Equations

$$\boxed{\text{Good qLDPC: } k = \Theta(n), \quad d = \Theta(n)}$$

$$\boxed{\text{Hypergraph product: } k = k_1 k_2, \quad d = \min(d_1, d_2)}$$

$$\boxed{\text{Surface code: } k = O(1), \quad d = O(\sqrt{n})}$$

### Main Takeaways

1. **Classical LDPC** codes approach capacity with efficient decoding
2. **Quantum LDPC** requires commutativity constraint
3. **Hypergraph product** gives qLDPC but limited distance
4. **Good qLDPC** codes (2020+) achieve linear rate AND distance
5. **Trade-off:** qLDPC needs non-local connectivity vs surface code locality

---

## Daily Checklist

- [ ] I understand classical LDPC codes and BP decoding
- [ ] I can explain the hypergraph product construction
- [ ] I know what makes a qLDPC code "good"
- [ ] I understand why good qLDPC codes matter for capacity
- [ ] I can compare surface codes to qLDPC codes
- [ ] I completed the computational lab

---

## Preview: Day 726

Tomorrow we study **Capacity Bounds and Calculations**, including:
- Numerical methods for capacity computation
- Semi-definite programming bounds
- Channel simulation for capacity estimation
- Practical capacity calculations
