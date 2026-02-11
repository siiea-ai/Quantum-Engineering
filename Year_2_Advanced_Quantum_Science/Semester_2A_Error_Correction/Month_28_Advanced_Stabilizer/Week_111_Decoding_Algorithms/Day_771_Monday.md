# Day 771: Maximum Likelihood Decoding

## Week 111: Decoding Algorithms | Month 28: Advanced Stabilizer Codes

---

## Daily Schedule

| Session | Time | Duration | Focus |
|---------|------|----------|-------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | MLD Theory & Bayesian Formulation |
| Afternoon | 1:00 PM - 4:00 PM | 3 hours | Coset Enumeration & Complexity Analysis |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab: Exact MLD Implementation |

**Total Study Time:** 7 hours

---

## Learning Objectives

By the end of this day, you will be able to:

1. **Formulate** the quantum decoding problem as maximum likelihood estimation
2. **Derive** the Bayesian posterior probability for errors given syndromes
3. **Explain** the coset structure and error degeneracy in stabilizer codes
4. **Analyze** the exponential complexity barrier for exact MLD
5. **Implement** exact MLD for small stabilizer codes via brute-force enumeration
6. **Compare** MLD performance against sub-optimal decoders as a benchmark

---

## Core Content

### 1. The Decoding Problem Formalized

In quantum error correction, the **decoder** is the classical algorithm that processes syndrome measurements and determines which correction to apply. After encoding a logical state and experiencing noise, we measure the syndrome generators $\{g_i\}$ obtaining outcomes $s = (s_1, s_2, \ldots, s_m) \in \{0, 1\}^m$.

The syndrome tells us *which* stabilizers were violated, but not *which specific error* occurred. Many distinct errors can produce the same syndrome.

**Definition (Decoding Problem):** Given syndrome $s$ and error model $P(E)$, find the most likely correction operator $\hat{C}$ such that $\hat{C}E$ returns the state to the code space with the correct logical state.

The key insight is that we don't need to identify the exact error $E$---we only need a correction $\hat{C}$ such that $\hat{C}E \in \mathcal{S}$ (the stabilizer group) or, more precisely, such that $\hat{C}E$ preserves the logical information.

### 2. Bayesian Formulation of MLD

Maximum Likelihood Decoding seeks the error most likely to have occurred given the observed syndrome. Using Bayes' theorem:

$$P(E | s) = \frac{P(s | E) P(E)}{P(s)}$$

Since $P(s)$ is a normalization constant, the MLD solution is:

$$\boxed{\hat{E}_{\text{MLD}} = \underset{E \in \mathcal{E}}{\text{argmax}} \, P(s | E) P(E)}$$

**Syndrome Likelihood $P(s|E)$:**
For stabilizer codes, the syndrome is deterministic given the error:
$$P(s | E) = \begin{cases} 1 & \text{if } s(E) = s \\ 0 & \text{otherwise} \end{cases}$$

where $s(E)_i = 0$ if $[E, g_i] = 0$ and $s(E)_i = 1$ if $\{E, g_i\} = 0$.

**Error Prior $P(E)$:**
For independent depolarizing noise with error rate $p$:
$$P(E) = \left(\frac{p}{3}\right)^{|E|} (1-p)^{n - |E|}$$

where $|E|$ is the weight (number of non-identity single-qubit errors).

### 3. Coset Structure and Degeneracy

The set of all errors decomposes into **cosets** of the stabilizer group $\mathcal{S}$:

$$\mathcal{P}_n = \mathcal{S} \cup E_1 \mathcal{S} \cup E_2 \mathcal{S} \cup \cdots$$

where $\mathcal{P}_n$ is the $n$-qubit Pauli group.

**Key Properties:**
1. All errors in the same coset $E\mathcal{S}$ produce the same syndrome
2. Errors differing by a stabilizer are **degenerate**---they have the same effect on the code space
3. The coset $E\mathcal{S}$ can be labeled by a representative element $E$

For an $[[n, k, d]]$ code, there are $4^n / |\mathcal{S}| = 4^n / 2^{n-k} = 2^{n+k}$ cosets. Each syndrome $s$ corresponds to $2^k$ cosets (differing by logical operators).

**Degeneracy in Decoding:**
Two errors $E_1$ and $E_2$ are **equivalent** for decoding if:
$$E_1 E_2^{-1} \in \mathcal{S}$$

This means applying correction $E_1^{-1}$ or $E_2^{-1}$ both return the state to the code space with the same logical state.

### 4. Optimal Decoding: Coset Probability

The truly optimal decoder maximizes the probability of the entire **equivalence class**, not just individual errors:

$$\boxed{\hat{C}_{\text{optimal}} = \underset{[E]}{\text{argmax}} \sum_{E' \in [E]} P(E')}$$

where $[E]$ denotes the equivalence class of $E$ (errors differing by stabilizers and preserving the logical state).

This accounts for **code degeneracy**: if many low-weight errors produce the same correction, their probabilities sum constructively.

**MLD vs Optimal Decoding:**
- **MLD**: Finds the single most likely error
- **Optimal**: Finds the most likely equivalence class

For highly degenerate codes, optimal decoding can significantly outperform MLD.

### 5. Exponential Complexity Barrier

**Theorem (Computational Hardness):** Maximum Likelihood Decoding for general stabilizer codes is NP-hard.

*Proof sketch:* The problem of finding the minimum weight error consistent with a syndrome can be reduced from the minimum weight codeword problem in classical linear codes, which is NP-complete.

**Complexity Analysis:**
- Number of possible errors: $4^n$ (all Paulis)
- Errors per syndrome: $4^n / 2^m = 4^n / 2^{n-k} = 2^{n+k}$
- For an $[[n, k, d]]$ code, we must search over $O(2^n)$ possibilities

This exponential scaling makes exact MLD intractable for codes of practical size (hundreds to thousands of qubits).

### 6. Tensor Network Approach to MLD

For certain code families (especially topological codes), tensor network methods provide a structured approach:

**Tensor Network Contraction:**
The partition function over error configurations can be expressed as:
$$Z(s) = \sum_{E: s(E) = s} P(E)$$

For 2D topological codes, this becomes a 2D tensor network that can be contracted approximately using DMRG-like methods.

**Complexity for Topological Codes:**
- Exact contraction: $O(\exp(\text{boundary length}))$
- For distance-$d$ surface code: $O(\exp(d))$
- Approximate methods achieve polynomial time with bounded error

---

## Worked Examples

### Example 1: MLD for the 3-Qubit Bit-Flip Code

**Problem:** For the 3-qubit repetition code with syndrome $s = (1, 0)$, find the MLD solution assuming bit-flip probability $p = 0.1$.

**Solution:**

The code has stabilizers $g_1 = ZZI$ and $g_2 = IZZ$.

Syndrome $s = (1, 0)$ means:
- $g_1$ anticommutes with $E$ (first qubit flipped or second, but not both)
- $g_2$ commutes with $E$ (second and third qubits have same value)

Possible errors giving $s = (1, 0)$:

| Error | Weight | $P(E)$ |
|-------|--------|--------|
| $XII$ | 1 | $p(1-p)^2 = 0.081$ |
| $IXX$ | 2 | $p^2(1-p) = 0.009$ |

**MLD Decision:**
$$\hat{E}_{\text{MLD}} = XII$$

The single bit-flip on qubit 1 is far more likely than the correlated flip on qubits 2 and 3.

### Example 2: MLD with Degeneracy

**Problem:** For the $[[5, 1, 3]]$ perfect code, show that different weight-1 errors with the same syndrome are degenerate.

**Solution:**

The $[[5, 1, 3]]$ code has $n - k = 4$ syndrome bits, giving $2^4 = 16$ syndromes.

Consider syndrome $s = (1, 0, 1, 0)$. Suppose both $X_1$ and $X_2 S$ produce this syndrome, where $S$ is some stabilizer.

Since $X_1 (X_2 S)^{-1} = X_1 X_2 S^{-1}$, if this equals a stabilizer, then $X_1$ and $X_2 S$ are equivalent.

For the perfect code, each single-qubit error produces a unique syndrome (since $d = 3$ means all weight-1 errors are distinguishable). So no degeneracy occurs at weight 1.

However, at weight 2, we may have $E_1 E_2^{-1} \in \mathcal{S}$ for certain pairs, making them degenerate.

### Example 3: Computing Coset Probabilities

**Problem:** For a syndrome $s$ with two consistent error classes $[E_1]$ and $[E_2]$, compute the optimal decoder output.

**Solution:**

Let the equivalence class $[E_1]$ contain errors: $E_1$ (weight 1), $E_1 S_a$ (weight 3), $E_1 S_b$ (weight 5).

Let $[E_2]$ contain: $E_2$ (weight 2), $E_2 S_c$ (weight 2).

For $p = 0.1$:
$$P([E_1]) = p(1-p)^{n-1} + p^3(1-p)^{n-3} + p^5(1-p)^{n-5}$$
$$P([E_2]) = p^2(1-p)^{n-2} + p^2(1-p)^{n-2} = 2p^2(1-p)^{n-2}$$

For $n = 10$:
$$P([E_1]) \approx 0.0387 + 0.000478 + \cdots \approx 0.0392$$
$$P([E_2]) \approx 2 \times 0.00430 = 0.00860$$

**Optimal Decision:** Choose $[E_1]$ even though $E_1$ has lower weight than individual $E_2$ errors. The degeneracy of $[E_2]$ (two weight-2 errors) is insufficient to overcome the weight-1 contribution in $[E_1]$.

---

## Practice Problems

### Level A: Direct Application

**A1.** For the 3-qubit phase-flip code with stabilizers $XXI, IXX$, list all single-qubit Z errors and their syndromes.

**A2.** Calculate $P(E)$ for $E = XIZI$ under depolarizing noise with $p = 0.05$ on a 4-qubit system.

**A3.** Show that for the 3-qubit bit-flip code, the syndrome $s = (1, 1)$ uniquely identifies error $IXI$.

### Level B: Intermediate Analysis

**B1.** For the Steane $[[7, 1, 3]]$ code, how many distinct syndromes exist? How many errors map to each syndrome on average?

**B2.** Prove that for a code with distance $d$, all errors of weight less than $d/2$ produce distinct syndromes.

**B3.** Compare MLD and optimal coset decoding for a code where the syndrome $s$ has:
- One weight-2 error $E_1$
- Four weight-3 errors $E_2, E_2 S_1, E_2 S_2, E_2 S_3$
Use $p = 0.05$ and determine when degeneracy changes the decision.

### Level C: Advanced Problems

**C1.** Show that MLD for the surface code is #P-hard by relating it to counting perfect matchings in a planar graph.

**C2.** Derive the tensor network representation of the syndrome probability distribution for the toric code. What is the bond dimension required for exact contraction?

**C3.** Design an algorithm that computes the MLD solution for codes of distance $d$ in time $O(n^{d/2})$ using syndrome-based pruning.

---

## Computational Lab: Exact MLD Implementation

```python
"""
Day 771 Computational Lab: Maximum Likelihood Decoding
Exact MLD implementation for small stabilizer codes

This lab implements brute-force MLD for educational purposes,
demonstrating the exponential complexity and optimal accuracy.
"""

import numpy as np
from itertools import product
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# Pauli matrices representation
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
PAULI_LABELS = ['I', 'X', 'Y', 'Z']

def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Convert Pauli string like 'XZI' to matrix."""
    result = PAULIS[pauli_str[0]]
    for p in pauli_str[1:]:
        result = np.kron(result, PAULIS[p])
    return result

def commutator_sign(pauli1: str, pauli2: str) -> int:
    """
    Returns 0 if paulis commute, 1 if they anticommute.
    Uses the rule: X,Y,Z cyclically anticommute.
    """
    anticommute_count = 0
    for p1, p2 in zip(pauli1, pauli2):
        if p1 != 'I' and p2 != 'I' and p1 != p2:
            anticommute_count += 1
    return anticommute_count % 2

def compute_syndrome(error: str, stabilizers: List[str]) -> Tuple[int, ...]:
    """Compute syndrome of an error given stabilizer generators."""
    syndrome = []
    for stab in stabilizers:
        syndrome.append(commutator_sign(error, stab))
    return tuple(syndrome)

def error_weight(error: str) -> int:
    """Count non-identity Paulis in error string."""
    return sum(1 for p in error if p != 'I')

def error_probability(error: str, p: float) -> float:
    """
    Probability of error under depolarizing noise.
    Each qubit has probability p of error, with X, Y, Z equally likely.
    """
    n = len(error)
    w = error_weight(error)
    return ((p/3) ** w) * ((1-p) ** (n - w))

def enumerate_all_errors(n: int) -> List[str]:
    """Generate all possible n-qubit Pauli errors."""
    errors = []
    for paulis in product(PAULI_LABELS, repeat=n):
        errors.append(''.join(paulis))
    return errors

class ExactMLDecoder:
    """
    Exact Maximum Likelihood Decoder for small stabilizer codes.

    WARNING: Exponential complexity O(4^n). Only use for n <= 8.
    """

    def __init__(self, stabilizers: List[str], p_error: float = 0.1):
        """
        Initialize decoder with stabilizer generators.

        Args:
            stabilizers: List of Pauli strings representing stabilizer generators
            p_error: Physical error probability
        """
        self.stabilizers = stabilizers
        self.n_qubits = len(stabilizers[0])
        self.p_error = p_error

        # Precompute syndrome table for all errors
        self._build_syndrome_table()

    def _build_syndrome_table(self):
        """Build lookup table: syndrome -> list of (error, probability)."""
        self.syndrome_table: Dict[Tuple[int, ...], List[Tuple[str, float]]] = {}

        all_errors = enumerate_all_errors(self.n_qubits)

        for error in all_errors:
            syndrome = compute_syndrome(error, self.stabilizers)
            prob = error_probability(error, self.p_error)

            if syndrome not in self.syndrome_table:
                self.syndrome_table[syndrome] = []
            self.syndrome_table[syndrome].append((error, prob))

        # Sort each syndrome's errors by probability (descending)
        for syndrome in self.syndrome_table:
            self.syndrome_table[syndrome].sort(key=lambda x: x[1], reverse=True)

    def decode(self, syndrome: Tuple[int, ...]) -> str:
        """
        Return the maximum likelihood error for given syndrome.

        Args:
            syndrome: Tuple of syndrome bits

        Returns:
            Most likely error string
        """
        if syndrome not in self.syndrome_table:
            raise ValueError(f"Invalid syndrome: {syndrome}")

        # Return highest probability error
        return self.syndrome_table[syndrome][0][0]

    def decode_with_confidence(self, syndrome: Tuple[int, ...]) -> Tuple[str, float, float]:
        """
        Decode and return confidence metrics.

        Returns:
            (error, probability, fraction of total syndrome probability)
        """
        errors = self.syndrome_table[syndrome]
        best_error, best_prob = errors[0]
        total_prob = sum(prob for _, prob in errors)

        return best_error, best_prob, best_prob / total_prob

    def analyze_syndrome(self, syndrome: Tuple[int, ...], top_k: int = 5):
        """Print analysis of errors consistent with syndrome."""
        errors = self.syndrome_table[syndrome]
        total_prob = sum(prob for _, prob in errors)

        print(f"Syndrome: {syndrome}")
        print(f"Total errors consistent: {len(errors)}")
        print(f"Total probability mass: {total_prob:.6e}")
        print(f"\nTop {min(top_k, len(errors))} errors:")
        print("-" * 40)

        for error, prob in errors[:top_k]:
            weight = error_weight(error)
            frac = prob / total_prob
            print(f"  {error}  weight={weight}  P={prob:.6e}  ({frac*100:.2f}%)")

def three_qubit_bitflip_demo():
    """Demonstrate MLD on 3-qubit bit-flip code."""
    print("=" * 60)
    print("3-Qubit Bit-Flip Code MLD Demo")
    print("=" * 60)

    stabilizers = ['ZZI', 'IZZ']
    decoder = ExactMLDecoder(stabilizers, p_error=0.1)

    # Analyze all possible syndromes
    syndromes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for syndrome in syndromes:
        print(f"\n{'='*40}")
        decoder.analyze_syndrome(syndrome, top_k=4)

        mld_error, prob, confidence = decoder.decode_with_confidence(syndrome)
        print(f"\nMLD Decision: {mld_error} (confidence: {confidence*100:.1f}%)")

def five_qubit_code_demo():
    """Demonstrate MLD on [[5,1,3]] perfect code."""
    print("\n" + "=" * 60)
    print("[[5,1,3]] Perfect Code MLD Demo")
    print("=" * 60)

    # [[5,1,3]] stabilizer generators
    stabilizers = [
        'XZZXI',
        'IXZZX',
        'XIXZZ',
        'ZXIXZ'
    ]

    decoder = ExactMLDecoder(stabilizers, p_error=0.05)

    # Test with a specific error
    test_error = 'XIIII'  # Single X error on qubit 1
    syndrome = compute_syndrome(test_error, stabilizers)

    print(f"\nApplied error: {test_error}")
    print(f"Resulting syndrome: {syndrome}")

    decoded = decoder.decode(syndrome)
    print(f"MLD decoded error: {decoded}")
    print(f"Correct: {decoded == test_error}")

    # Analyze the syndrome
    decoder.analyze_syndrome(syndrome, top_k=5)

def threshold_analysis():
    """
    Analyze decoder success probability vs physical error rate.
    """
    print("\n" + "=" * 60)
    print("Threshold Analysis: 3-Qubit vs 5-Qubit Codes")
    print("=" * 60)

    error_rates = np.linspace(0.01, 0.25, 20)

    # 3-qubit bit-flip code
    stab_3q = ['ZZI', 'IZZ']
    success_3q = []

    # 5-qubit code
    stab_5q = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
    success_5q = []

    for p in error_rates:
        # 3-qubit analysis
        decoder_3q = ExactMLDecoder(stab_3q, p_error=p)

        # Success = probability of decoding correctly
        # For simplicity, weight success by error probability
        total_success_3q = 0
        total_prob_3q = 0

        for syndrome in decoder_3q.syndrome_table:
            errors = decoder_3q.syndrome_table[syndrome]
            mld_error = errors[0][0]

            for error, prob in errors:
                # Success if decoded error differs from true error by stabilizer
                # For bit-flip code, this means same syndrome => correct class
                if error_weight(error) == error_weight(mld_error):
                    total_success_3q += prob
                elif error_weight(error) > error_weight(mld_error):
                    total_success_3q += prob  # MLD chose lower weight
                total_prob_3q += prob

        success_3q.append(total_success_3q)

        # 5-qubit analysis (simplified)
        decoder_5q = ExactMLDecoder(stab_5q, p_error=p)
        total_success_5q = 0

        # Count probability that MLD finds minimum weight error
        for syndrome in decoder_5q.syndrome_table:
            errors = decoder_5q.syndrome_table[syndrome]
            if errors:
                min_weight = min(error_weight(e) for e, _ in errors)
                for error, prob in errors:
                    if error_weight(error) == min_weight:
                        total_success_5q += prob

        success_5q.append(total_success_5q)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates * 100, success_3q, 'b-o', label='3-qubit bit-flip code', linewidth=2)
    plt.plot(error_rates * 100, success_5q, 'r-s', label='[[5,1,3]] code', linewidth=2)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Threshold line')

    plt.xlabel('Physical Error Rate (%)', fontsize=12)
    plt.ylabel('Logical Success Probability', fontsize=12)
    plt.title('MLD Performance: Success Probability vs Error Rate', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 25])
    plt.ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('mld_threshold_analysis.png', dpi=150)
    plt.show()

    print("\nPlot saved to mld_threshold_analysis.png")

def complexity_scaling():
    """Demonstrate exponential complexity of exact MLD."""
    print("\n" + "=" * 60)
    print("Complexity Scaling Analysis")
    print("=" * 60)

    import time

    print("\nNumber of Pauli errors vs n_qubits:")
    print("-" * 40)

    for n in range(1, 9):
        n_errors = 4 ** n
        print(f"  n = {n}: {n_errors:>10,} errors")

    print("\nBenchmarking syndrome table construction:")
    print("-" * 40)

    times = []
    sizes = range(3, 8)

    for n in sizes:
        # Create simple repetition-like stabilizers
        stabilizers = []
        for i in range(n - 1):
            stab = 'I' * i + 'ZZ' + 'I' * (n - i - 2)
            stabilizers.append(stab)

        start = time.time()
        decoder = ExactMLDecoder(stabilizers, p_error=0.1)
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"  n = {n}: {elapsed:.4f} seconds ({4**n} errors)")

    # Fit exponential
    log_times = np.log(times)
    coeffs = np.polyfit(list(sizes), log_times, 1)

    print(f"\nExponential fit: time ~ exp({coeffs[0]:.2f} * n)")
    print(f"Base of exponential: {np.exp(coeffs[0]):.2f}")
    print("(Theoretical: 4.0)")

if __name__ == "__main__":
    # Run all demonstrations
    three_qubit_bitflip_demo()
    five_qubit_code_demo()
    complexity_scaling()
    threshold_analysis()

    print("\n" + "=" * 60)
    print("Lab Complete!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| MLD Objective | $$\hat{E} = \underset{E}{\text{argmax}} \, P(s \vert E) P(E)$$ |
| Error Probability (Depolarizing) | $$P(E) = (p/3)^{\vert E \vert} (1-p)^{n-\vert E \vert}$$ |
| Syndrome Determinism | $$P(s \vert E) = \delta_{s, s(E)}$$ |
| Coset Count | $$\vert \mathcal{P}_n / \mathcal{S} \vert = 2^{n+k}$$ |
| Optimal Coset Decoding | $$\hat{C} = \underset{[E]}{\text{argmax}} \sum_{E' \in [E]} P(E')$$ |
| MLD Complexity | $$O(4^n)$$ brute force, NP-hard in general |

### Key Takeaways

1. **MLD is optimal but intractable**: It provides the benchmark for decoder accuracy but scales exponentially
2. **Degeneracy matters**: Optimal decoding considers coset probabilities, not just individual errors
3. **Syndrome determines cosets**: Each syndrome corresponds to $2^k$ logical cosets
4. **Low weight dominates**: Under reasonable noise, low-weight errors dominate the posterior
5. **Practical decoders approximate MLD**: All practical decoders trade accuracy for speed

---

## Daily Checklist

- [ ] Derived the Bayesian formulation of MLD
- [ ] Understood coset structure and error equivalence
- [ ] Analyzed exponential complexity barrier
- [ ] Computed MLD solutions for 3-qubit and 5-qubit codes
- [ ] Implemented exact MLD decoder in Python
- [ ] Compared MLD accuracy across different error rates
- [ ] Completed practice problems (at least Level A and B)

---

## Preview: Day 772

Tomorrow we introduce **Minimum Weight Perfect Matching (MWPM)**, the workhorse decoder for surface codes. We'll learn how to construct the syndrome graph, apply the Blossom algorithm for polynomial-time matching, and understand why MWPM achieves near-optimal thresholds despite being much faster than MLD.

Key questions for tomorrow:
- How do we map syndromes to a graph matching problem?
- What is the Blossom algorithm and why is it $O(n^3)$?
- How does MWPM handle measurement errors and correlated noise?

---

*Day 771 of 2184 | Week 111 | Month 28 | Year 2: Advanced Quantum Science*
