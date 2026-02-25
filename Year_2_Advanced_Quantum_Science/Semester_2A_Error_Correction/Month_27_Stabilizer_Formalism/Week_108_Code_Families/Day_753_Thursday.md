# Day 753: Good qLDPC Constructions

## Overview

**Day:** 753 of 1008
**Week:** 108 (Code Families & Construction Techniques)
**Month:** 27 (Stabilizer Formalism)
**Topic:** Quantum Low-Density Parity-Check Codes with Optimal Scaling

---

## Schedule

| Block | Time | Duration | Focus |
|-------|------|----------|-------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hrs | qLDPC fundamentals |
| Afternoon | 2:00 PM - 5:30 PM | 3.5 hrs | Good code constructions |
| Evening | 7:00 PM - 9:00 PM | 2 hrs | Computational lab |

---

## Learning Objectives

By the end of today, you should be able to:

1. **Define** "good" quantum codes and LDPC properties
2. **Explain** the significance of constant rate and linear distance
3. **Understand** expander-based constructions
4. **Describe** the 2021 breakthrough results
5. **Compare** qLDPC codes with surface codes
6. **Analyze** implications for fault-tolerant computing

---

## What Makes a Code "Good"?

### Classical Definition

A family of codes $\{C_n\}$ is **good** if:
- **Constant rate:** $k/n \to R > 0$ as $n \to \infty$
- **Linear distance:** $d/n \to \delta > 0$ as $n \to \infty$

Good codes encode $\Theta(n)$ bits with distance $\Theta(n)$.

### Quantum Definition

A family of quantum codes $\{[[n, k, d]]\}$ is **good** if:
$$\boxed{k = \Theta(n) \text{ and } d = \Theta(n)}$$

### LDPC Property

A code is **LDPC** (Low-Density Parity-Check) if:
- Each parity check involves O(1) bits (constant row weight)
- Each bit is in O(1) parity checks (constant column weight)

**Quantum LDPC:** Both X and Z stabilizers have constant weight.

---

## Why Good qLDPC Matters

### Surface Code Limitations

Surface codes: [[n, O(1), $\sqrt{n}$]]
- Distance only $\sqrt{n}$
- Need $n = O(d^2)$ qubits for distance d
- Rate approaches 0

### The Dream

Good qLDPC: [[n, Θ(n), Θ(n)]]
- Linear distance: $d = \Theta(n)$
- Constant rate: encode many qubits
- Potential for efficient fault tolerance

### Practical Impact

| Property | Surface Code | Good qLDPC |
|----------|-------------|------------|
| Rate | 1/n → 0 | Θ(1) |
| Distance | √n | Θ(n) |
| Qubits for d=100 | ~10,000 | ~O(100) |
| Overhead | High | Much lower |

---

## Historical Progress

### Before 2021

**Known results:**
- Surface codes: [[n, O(1), √n]] (not good)
- Hypergraph products: [[n, Θ(n), Θ(√n)]] (good rate, sublinear d)
- Fiber bundle codes: [[n, Θ(n), Θ(n^{1/2+ε})]] (almost linear d)

**Open question:** Do good quantum LDPC codes exist?

### The 2021 Breakthrough

**Theorem (Panteleev-Kalachev, 2021):**
Good qLDPC codes exist! There exist [[n, Θ(n), Θ(n)]] codes with O(1) weight stabilizers.

**Construction:** Based on Cayley graphs and lifted products.

### Further Developments

**2022:** Quantum Tanner codes (Leverrier-Zémor)
- Simpler construction
- Linear distance proven
- Better constants

---

## Construction Approaches

### Expander Graphs

**Definition:** An $(n, d, λ)$-expander is a d-regular graph on n vertices with spectral gap at least λ.

**Key property:** Good expansion → good error correction.

Classical LDPC codes on expanders achieve capacity.

### Classical Product Codes

**Hypergraph product:** HP(C₁, C₂)
- Preserves LDPC property
- Rate: $R_1 R_2 / (R_1 + R_2 - R_1 R_2)$
- Distance: min(d₁, d₂)

**Limitation:** Distance doesn't grow with n.

### Lifted Products

**Key innovation:** "Lift" the product construction.

For group G acting on codes:
$$LP(C_1, C_2, G) = \text{lifted hypergraph product}$$

This can achieve linear distance!

### Quantum Tanner Codes

**Construction:**
1. Start with classical Tanner code on expander
2. Apply quantum version with local codes
3. Distance grows linearly if expansion is strong enough

---

## Technical Details

### The Panteleev-Kalachev Construction

**Ingredients:**
1. Cayley graph of group G
2. Classical code C over G
3. Balanced product construction

**Result:** CSS code with:
- n = O(|G|²)
- k = Ω(|G|)
- d = Ω(|G|)

### Distance Proof Idea

**Classical:** Expansion prevents small-weight errors from being undetectable.

**Quantum:** Need to handle both X and Z errors.

**Key insight:** The product structure ensures that any low-weight error must violate either X or Z checks.

### Rate Analysis

For well-chosen parameters:
$$R = k/n \geq c > 0$$

where c is a constant independent of n.

---

## Comparison with Other Codes

### Code Family Comparison

| Family | Rate | Distance | LDPC? | Practical? |
|--------|------|----------|-------|------------|
| Surface | 1/n | √n | Yes | Yes |
| Steane | 1/7 | 3 | No | Yes |
| HP | Θ(1) | Θ(√n) | Yes | Maybe |
| Tanner | Θ(1) | Θ(n) | Yes | Future |

### Decoding Complexity

**Surface codes:** Efficient decoders (MWPM, union-find)
**Good qLDPC:** Decoding is more challenging
- Belief propagation may fail
- Need specialized algorithms

### Threshold Behavior

**Surface codes:** ~1% threshold well-understood
**Good qLDPC:** Thresholds being actively studied

---

## Implications for Quantum Computing

### Reduced Overhead

With d = Θ(n) instead of d = Θ(√n):
- For same logical error rate, need fewer physical qubits
- Could reduce overhead by orders of magnitude

### Challenges

1. **Decoding:** Need efficient algorithms
2. **Implementation:** Non-local connectivity
3. **Threshold:** May be lower than surface codes
4. **Fault tolerance:** Gate implementations more complex

### Future Outlook

Good qLDPC codes represent a paradigm shift:
- From "surface codes everywhere" to diverse code choices
- Trade-offs between overhead, threshold, and complexity

---

## Worked Examples

### Example 1: Rate Comparison

**Problem:** Compare qubits needed for logical error rate 10⁻¹⁵.

**Solution:**

**Surface code:**
- Need d such that $p_{logical} \approx (p/p_{th})^{d/2} < 10^{-15}$
- With p = 0.001, p_th = 0.01: need d ≈ 30
- n ≈ d² = 900 physical qubits per logical

**Good qLDPC (hypothetical):**
- With d = cn for constant c
- Need n such that similar error suppression
- Could be n ≈ 100 with better constants

### Example 2: HP vs Good qLDPC

**Problem:** For HP of [100, 50, 10] codes, what are parameters?

**Solution:**

HP(C, C) where C = [100, 50, 10]:
- n = 100² + 50² = 12,500
- k = 50² = 2,500
- d = min(10, 10) = 10

Rate: 2500/12500 = 0.2 (good!)
Distance: 10 (not scaling with n)

Good qLDPC would have d = Θ(n) ≈ Θ(12500).

### Example 3: Scaling Analysis

**Problem:** If a good qLDPC family has k = n/10 and d = n/20, how many physical qubits are needed to encode 1000 logical qubits with distance 100?

**Solution:**

Need k ≥ 1000 and d ≥ 100.

From k = n/10: n ≥ 10,000
From d = n/20: n ≥ 2,000

Binding constraint: n = 10,000

**Answer:** 10,000 physical qubits for 1000 logical qubits with distance 100.

(Compare to surface codes: would need ~10,000 per logical qubit!)

---

## Practice Problems

### Level 1: Direct Application

**P1.1** A code family has n = 1000, k = 100, d = 50. Is it good?

**P1.2** If d = √n, how many physical qubits are needed for d = 100?

**P1.3** Calculate the rate of HP(C₁, C₂) where C₁ = [20, 10, 5] and C₂ = [30, 15, 6].

### Level 2: Intermediate

**P2.1** Prove that surface codes are not "good" codes.

**P2.2** For a code family with k = n/5 and d = n/10:
a) What is the rate?
b) How many physical qubits for 100 logical qubits with d ≥ 50?

**P2.3** Compare total qubit overhead for achieving 10⁻¹² logical error using surface codes vs good qLDPC (assuming equal thresholds).

### Level 3: Challenging

**P3.1** Explain why the hypergraph product doesn't achieve linear distance.

**P3.2** Describe the key innovation in lifted products that enables linear distance.

**P3.3** Analyze the decoding challenge for good qLDPC: why is belief propagation insufficient?

---

## Computational Lab

```python
"""
Day 753: Good qLDPC Constructions
=================================

Analyzing code scaling and comparing code families.
"""

import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt


def code_parameters(family: str, n: int) -> Tuple[int, int]:
    """
    Return (k, d) for code family at size n.

    Families: 'surface', 'hp', 'good_qldpc'
    """
    if family == 'surface':
        # [[n, 1, sqrt(n)]]
        d = int(np.sqrt(n))
        k = 1
    elif family == 'hp':
        # [[n, n/5, sqrt(n)]] approximately
        k = n // 5
        d = int(np.sqrt(n))
    elif family == 'good_qldpc':
        # [[n, n/10, n/20]]
        k = n // 10
        d = n // 20
    else:
        raise ValueError(f"Unknown family: {family}")

    return k, d


def qubits_for_distance(family: str, target_d: int) -> int:
    """Compute n needed to achieve target distance."""
    if family == 'surface':
        return target_d ** 2
    elif family == 'hp':
        return target_d ** 2
    elif family == 'good_qldpc':
        return 20 * target_d  # d = n/20
    return -1


def qubits_for_logical(family: str, target_k: int, min_d: int) -> int:
    """Compute n needed for target_k logical qubits with min distance."""
    if family == 'surface':
        # k = 1 always, so need target_k code blocks
        n_per_block = min_d ** 2
        return target_k * n_per_block
    elif family == 'hp':
        # k = n/5, d = sqrt(n)
        n_from_k = 5 * target_k
        n_from_d = min_d ** 2
        return max(n_from_k, n_from_d)
    elif family == 'good_qldpc':
        # k = n/10, d = n/20
        n_from_k = 10 * target_k
        n_from_d = 20 * min_d
        return max(n_from_k, n_from_d)
    return -1


def logical_error_rate(physical_error: float, distance: int,
                       threshold: float = 0.01) -> float:
    """
    Estimate logical error rate using simplified model.

    p_L ≈ (p/p_th)^(d/2) for surface-code-like behavior.
    """
    if physical_error >= threshold:
        return 1.0
    ratio = physical_error / threshold
    return ratio ** (distance / 2)


def compare_families(target_logical_error: float,
                     physical_error: float) -> Dict:
    """
    Compare code families for achieving target logical error.
    """
    families = ['surface', 'hp', 'good_qldpc']
    results = {}

    for family in families:
        # Find minimum distance needed
        d = 2
        while d < 10000:
            p_L = logical_error_rate(physical_error, d)
            if p_L < target_logical_error:
                break
            d += 2

        n = qubits_for_distance(family, d)
        k, actual_d = code_parameters(family, n)

        results[family] = {
            'distance_needed': d,
            'n_qubits': n,
            'k_logical': k,
            'rate': k / n if n > 0 else 0
        }

    return results


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Day 753: Good qLDPC Code Analysis")
    print("=" * 60)

    # Example 1: Scaling comparison
    print("\n1. Code Family Scaling")
    print("-" * 40)

    print(f"{'n':<10} {'Surface':<20} {'HP':<20} {'Good qLDPC':<20}")
    print(f"{'':10} {'k,d':<20} {'k,d':<20} {'k,d':<20}")
    print("-" * 70)

    for n in [100, 1000, 10000]:
        s_k, s_d = code_parameters('surface', n)
        h_k, h_d = code_parameters('hp', n)
        g_k, g_d = code_parameters('good_qldpc', n)
        print(f"{n:<10} {f'{s_k},{s_d}':<20} {f'{h_k},{h_d}':<20} {f'{g_k},{g_d}':<20}")

    # Example 2: Qubits for target distance
    print("\n2. Qubits for Target Distance")
    print("-" * 40)

    print(f"{'Distance':<12} {'Surface':<15} {'HP':<15} {'Good qLDPC':<15}")
    print("-" * 57)

    for d in [10, 50, 100, 500]:
        s_n = qubits_for_distance('surface', d)
        h_n = qubits_for_distance('hp', d)
        g_n = qubits_for_distance('good_qldpc', d)
        print(f"{d:<12} {s_n:<15} {h_n:<15} {g_n:<15}")

    # Example 3: Overhead for logical qubits
    print("\n3. Overhead for 100 Logical Qubits with d ≥ 50")
    print("-" * 40)

    target_k = 100
    min_d = 50

    for family in ['surface', 'hp', 'good_qldpc']:
        n = qubits_for_logical(family, target_k, min_d)
        k, d = code_parameters(family, n)
        print(f"{family:<15}: n = {n:>8}, k = {k:>5}, d = {d:>5}")

    # Example 4: Error rate comparison
    print("\n4. Achieving Target Logical Error Rate")
    print("-" * 40)

    physical_error = 0.001
    target = 1e-12

    print(f"Physical error: {physical_error}")
    print(f"Target logical error: {target}")

    results = compare_families(target, physical_error)
    print(f"\n{'Family':<15} {'d needed':<12} {'n qubits':<12} {'k logical':<12} {'Rate':<10}")
    print("-" * 61)

    for family, data in results.items():
        print(f"{family:<15} {data['distance_needed']:<12} "
              f"{data['n_qubits']:<12} {data['k_logical']:<12} "
              f"{data['rate']:<10.4f}")

    # Example 5: Historical progress
    print("\n5. Historical Progress in Quantum Codes")
    print("-" * 40)

    timeline = [
        ("1995", "Shor code", "[[9,1,3]]", "First QEC"),
        ("1996", "CSS codes", "Various", "Systematic construction"),
        ("1997", "Surface codes", "[[n,1,√n]]", "Topological protection"),
        ("2013", "HP codes", "[[n,Θ(n),Θ(√n)]]", "Constant rate"),
        ("2021", "P-K codes", "[[n,Θ(n),Θ(n)]]", "GOOD qLDPC!"),
        ("2022", "Tanner codes", "[[n,Θ(n),Θ(n)]]", "Simpler construction")
    ]

    print(f"{'Year':<8} {'Code':<15} {'Parameters':<20} {'Significance':<25}")
    print("-" * 70)
    for year, code, params, sig in timeline:
        print(f"{year:<8} {code:<15} {params:<20} {sig:<25}")

    print("\n" + "=" * 60)
    print("Good qLDPC codes: the future of quantum error correction!")
    print("=" * 60)
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Good codes | $k = \Theta(n)$, $d = \Theta(n)$ |
| Surface scaling | $[[n, O(1), \sqrt{n}]]$ |
| HP scaling | $[[n, \Theta(n), \Theta(\sqrt{n})]]$ |
| Good qLDPC | $[[n, \Theta(n), \Theta(n)]]$ |

### Main Takeaways

1. **Good qLDPC codes** have constant rate AND linear distance
2. **2021 breakthrough** proved existence
3. **Potential for huge overhead reduction** vs surface codes
4. **Challenges remain:** decoding, implementation, thresholds
5. Active area of research with rapid progress

---

## Daily Checklist

- [ ] I can define "good" quantum codes
- [ ] I understand the significance of linear distance
- [ ] I know the historical progression of quantum codes
- [ ] I can compare overhead between code families
- [ ] I understand why good qLDPC matters for fault tolerance
- [ ] I know the main challenges for practical implementation

---

## Preview: Day 754

Tomorrow we explore the **Gottesman-Knill theorem**:

- Efficient classical simulation of stabilizer circuits
- Tableau representation
- Why Clifford alone isn't universal
- Implications for quantum advantage

The Gottesman-Knill theorem reveals the boundary between classical and quantum!
