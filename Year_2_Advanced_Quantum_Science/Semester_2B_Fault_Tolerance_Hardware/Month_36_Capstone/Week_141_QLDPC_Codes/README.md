# Week 141: QLDPC Codes & Constant-Overhead QEC

## Overview

**Days:** 981-987 (7 days)
**Month:** 36 (Year 2 Capstone)
**Semester:** 2B (Fault Tolerance & Hardware)
**Focus:** Quantum Low-Density Parity-Check (qLDPC) codes and the breakthrough of constant-overhead fault tolerance

---

## Status: In Progress

| Day | Topic | Status |
|-----|-------|--------|
| 981 | Classical LDPC Codes & Belief Propagation | Not Started |
| 982 | Quantum LDPC Code Construction | Not Started |
| 983 | Good qLDPC Codes: Constant Rate & Distance | Not Started |
| 984 | Panteleev-Kalachev & Quantum Tanner Codes | Not Started |
| 985 | Constant-Overhead Fault Tolerance | Not Started |
| 986 | QLDPC vs Surface Code Comparison | Not Started |
| 987 | Implementation Challenges & Week Synthesis | Not Started |

**Progress:** 0/7 days (0%)

---

## Learning Objectives

By the end of Week 141, you will be able to:

1. **Explain** classical LDPC codes, Tanner graphs, and belief propagation decoding
2. **Construct** quantum LDPC codes using CSS and hypergraph product methods
3. **Prove** why good qLDPC codes with $[[n, \Theta(n), \Theta(n)]]$ parameters exist
4. **Describe** the Panteleev-Kalachev construction and lifted product codes
5. **Analyze** constant-overhead fault tolerance and its asymptotic advantages
6. **Compare** qLDPC codes with surface codes across multiple metrics
7. **Evaluate** practical implementation challenges and near-term prospects
8. **Implement** LDPC graph analysis and decoding algorithms

---

## Daily Breakdown

### Day 981: Classical LDPC Codes & Belief Propagation

The foundation: capacity-approaching classical codes that inspired quantum breakthroughs.

**Core Topics:**
- Sparse parity-check matrices
- Tanner graph representation
- Regular vs irregular LDPC codes
- Sum-product (belief propagation) algorithm
- Performance near Shannon limit
- Connections to quantum error correction

**Key Equations:**
$$H \cdot c^T = 0 \pmod{2}$$
$$m_{\text{v}\to\text{c}}(x) = \prod_{c' \in \mathcal{N}(v) \setminus c} m_{c'\to v}(x)$$

---

### Day 982: Quantum LDPC Code Construction

Translating classical LDPC success to the quantum realm.

**Core Topics:**
- CSS codes from classical LDPC
- Hypergraph product construction
- Tillich-Zémor codes
- Challenges: degeneracy and syndrome decoding
- Quantum belief propagation
- Code families and parameters

**Key Equations:**
$$[[n, k, d]] = [[n_1 n_2 + n_1 n_2, k_1 k_2, \min(d_1, d_2)]]$$
$$H_X = [H_1 \otimes I, I \otimes H_2^T]$$
$$H_Z = [I \otimes H_1, H_2^T \otimes I]$$

---

### Day 983: Good qLDPC Codes: Constant Rate & Distance

The breakthrough: codes that scale optimally in all parameters.

**Core Topics:**
- Definition of "good" codes: $k = \Theta(n)$, $d = \Theta(n)$
- Historical barriers to good qLDPC
- Freedman-Meyer-Luo homological construction attempts
- Quantum expander codes
- The breakthrough of 2021-2022
- Implications for fault tolerance

**Key Equations:**
$$\boxed{[[n, k, d]] \text{ with } k = \Theta(n), \, d = \Theta(n)}$$
$$\text{Rate: } R = k/n = \Theta(1)$$
$$\text{Relative distance: } \delta = d/n = \Theta(1)$$

---

### Day 984: Panteleev-Kalachev & Quantum Tanner Codes

The constructions that achieved the impossible.

**Core Topics:**
- Lifted product codes (Panteleev-Kalachev 2022)
- Group algebra and lifting
- Cayley graphs and expanders
- Quantum Tanner codes (Leverrier-Zémor 2022)
- Left-right Cayley complexes
- Explicit constructions and parameters

**Key Equations:**
$$\tilde{H} = H \otimes_{\pi} A$$
$$C(G, S_L, S_R) = \{(V, E_L, E_R)\}$$
$$d \geq \Omega(\sqrt{n})$$ to $d = \Theta(n)$

---

### Day 985: Constant-Overhead Fault Tolerance

The holy grail: finite resources per logical qubit.

**Core Topics:**
- Overhead definition: physical/logical ratio
- Surface code overhead: $O(d^2)$ per logical qubit
- Good qLDPC overhead: $O(1)$ asymptotically
- Threshold behavior with qLDPC
- Gate implementation challenges
- Transversal and non-transversal operations
- Implications for scalable quantum computing

**Key Equations:**
$$\text{Overhead}_{\text{surface}} = O(d^2) = O(1/p^2)$$
$$\text{Overhead}_{\text{qLDPC}} = O(1) \text{ (asymptotically)}$$
$$n_{\text{physical}} \sim k_{\text{logical}} \cdot \text{const}$$

---

### Day 986: QLDPC vs Surface Code Comparison

Detailed analysis of competing paradigms.

**Core Topics:**
- Rate comparison: $O(1)$ vs $O(1/d^2)$
- Locality: 2D planar vs non-local
- Decoding complexity
- Syndrome measurement circuits
- Gate implementation
- Near-term practicality
- Long-term scalability

**Key Comparison:**

| Property | Surface Code | Good QLDPC |
|----------|--------------|------------|
| Rate $k/n$ | $O(1/d^2)$ | $\Theta(1)$ |
| Distance | $O(\sqrt{n})$ | $\Theta(n)$ |
| Locality | 2D local | Non-local |
| Decoding | Efficient | More complex |
| Overhead | $O(d^2)$ | $O(1)$ |

---

### Day 987: Implementation Challenges & Week Synthesis

Bridging theory and practice, synthesizing the week.

**Core Topics:**
- Non-locality challenges in hardware
- Long-range connectivity requirements
- Syndrome extraction circuit depth
- Decoding latency issues
- Proposals: 3D architectures, modular systems
- Recent experimental progress
- Week synthesis and outlook

---

## Key Concepts

### LDPC Code Structure

```
Tanner Graph:
Variable nodes -------- Check nodes
    (n bits)           (m checks)

Edges: sparse connections (low density)
Degree: typically constant (3-6)
```

### Hypergraph Product Construction

Given classical codes $C_1$ and $C_2$:

$$\text{qLDPC} = C_1 \otimes C_2 \oplus C_1^T \otimes C_2^T$$

| Component | Size | Role |
|-----------|------|------|
| $H_1$ | $m_1 \times n_1$ | X-checks (part 1) |
| $H_2$ | $m_2 \times n_2$ | Z-checks (part 2) |
| Product | $(n_1 n_2 + m_1 m_2)$ qubits | Combined code |

### Good Code Parameters

| Code Type | Rate $k/n$ | Distance $d$ | Overhead |
|-----------|------------|--------------|----------|
| Repetition | $O(1/n)$ | $n$ | Very high |
| Steane | $1/7$ | $3$ | Constant but low $d$ |
| Surface | $O(1/d^2)$ | $O(\sqrt{n})$ | $O(d^2)$ |
| **Good qLDPC** | $\Theta(1)$ | $\Theta(n)$ | $O(1)$ |

---

## Prerequisites

### From Semester 2A
- Stabilizer formalism mastery
- CSS code construction
- Surface code architecture
- Decoding algorithms

### From Month 35
- Fault-tolerant principles
- Hardware platform knowledge
- Error correction fundamentals

### Mathematical Background
- Graph theory and expanders
- Group theory and Cayley graphs
- Homological algebra basics
- Probability and belief propagation

---

## Resources

### Primary References
- Panteleev & Kalachev, "Asymptotically Good QLDPC Codes" (2022)
- Leverrier & Zémor, "Quantum Tanner Codes" (2022)
- Breuckmann & Eberhardt, "Quantum Low-Density Parity-Check Codes" (2021)
- Tillich & Zémor, "Quantum LDPC Codes with Positive Rate" (2014)

### Key Papers
- Gallager, "Low-Density Parity-Check Codes" (1962) - The original LDPC paper
- Gottesman, "Fault-Tolerant QC with Constant Overhead" (2014) - Vision paper
- Hastings, Haah, O'Donnell, "Fiber Bundle Codes" (2021)
- Dinur et al., "Good QLDPC Codes with Linear Time Decoders" (2022)

### Online Resources
- [Breuckmann's QLDPC Tutorial](https://arxiv.org/abs/2103.06309)
- [QIP 2023 QLDPC Sessions](https://qipconference.org/)
- [Quantum Error Correction Zoo](https://errorcorrectionzoo.org/)

---

## Computational Tools

```python
# Week 141 computational stack
import numpy as np
from scipy import sparse, linalg
import networkx as nx
import matplotlib.pyplot as plt

# Graph and code analysis
from ldpc import bp_decoder  # Classical LDPC tools
from pymatching import Matching  # For comparison with surface codes

# Custom implementations
# - Tanner graph construction
# - Belief propagation for classical LDPC
# - Hypergraph product code construction
# - Check matrix sparsity analysis
```

---

## Connections

### From Week 140
- Advanced variational methods -> Understanding QEC limitations
- Error mitigation -> Why we need fault tolerance
- NISQ algorithms -> Contrast with FT approach

### To Weeks 142-144
- QLDPC foundations -> Research frontiers
- Constant overhead -> Scalability discussions
- Synthesis -> Year 2 capstone completion

---

## Why QLDPC Matters

### The Scalability Crisis

Surface codes have been the leading QEC paradigm:
- 2D local interactions (excellent for hardware)
- Efficient decoding (polynomial time)
- High threshold (~1%)

**BUT**: Overhead grows as $O(d^2)$ per logical qubit

For $10^6$ logical qubits at $d=100$: need $10^{10}$ physical qubits!

### The QLDPC Promise

Good qLDPC codes offer $O(1)$ overhead:
- Same $10^6$ logical qubits might need only $10^7-10^8$ physical qubits
- **100-1000x improvement** in asymptotic scaling

### The Trade-off

Non-locality is the price:
- Qubits must interact over long distances
- Syndrome extraction becomes complex
- Hardware requirements fundamentally different

---

## Summary

Week 141 explores the revolutionary development of quantum LDPC codes that achieve constant rate and linear distance simultaneously. Starting from classical LDPC codes (Day 981), we build up to quantum constructions (Day 982), then examine the breakthrough "good" codes (Day 983). The Panteleev-Kalachev and Leverrier-Zémor constructions (Day 984) show how expansion properties enable these optimal parameters. Constant-overhead fault tolerance (Day 985) reveals the profound implications for scalable quantum computing. The comparison with surface codes (Day 986) highlights trade-offs, while practical challenges (Day 987) ground the theory in reality.

These codes represent perhaps the most significant theoretical breakthrough in quantum error correction since the field's inception, fundamentally changing our understanding of what is asymptotically possible.

---

*"The construction of good quantum LDPC codes settles a 25-year-old open problem and revolutionizes our understanding of quantum fault tolerance."*
--- Contemporary perspective on the 2021-2022 breakthroughs

---

**Last Updated:** February 7, 2026
**Status:** In Progress - 0/7 days complete (0%)
