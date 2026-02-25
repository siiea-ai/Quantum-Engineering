# Week 103: Subsystem Codes

## Overview

**Days:** 715-721 (7 days)
**Month:** 26 (QEC Fundamentals II)
**Topic:** Subsystem Codes — Structure, Properties, and Fault Tolerance

---

## Status: ✅ COMPLETE

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 715 | Monday | Introduction to Subsystem Codes | ✅ Complete |
| 716 | Tuesday | Gauge Operators and Logical Operators | ✅ Complete |
| 717 | Wednesday | The Bacon-Shor Code | ✅ Complete |
| 718 | Thursday | Subsystem Code Properties | ✅ Complete |
| 719 | Friday | Advantages of Subsystem Codes | ✅ Complete |
| 720 | Saturday | Subsystem Codes and Fault Tolerance | ✅ Complete |
| 721 | Sunday | Week Synthesis | ✅ Complete |

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Define** subsystem codes and the decomposition $\mathcal{C} = A \otimes B$
2. **Construct** gauge and stabilizer groups from generators
3. **Compute** the parameters $[[n, k, r, d]]$ for subsystem codes
4. **Analyze** the Bacon-Shor code family on $m \times n$ lattices
5. **Prove** error correction conditions for subsystem codes
6. **Apply** the Singleton bound to subsystem codes
7. **Explain** advantages: weight reduction, fault tolerance, single-shot
8. **Design** fault-tolerant syndrome extraction circuits
9. **Identify** transversal gates on Bacon-Shor codes
10. **Compare** subsystem codes to stabilizer codes

---

## Core Concepts

### The Subsystem Code Framework

**Code Structure:**
$$\mathcal{H} = (A \otimes B) \oplus \mathcal{C}^\perp$$

| Component | Role | Dimension |
|-----------|------|-----------|
| $A$ | Logical subsystem (protected information) | $2^k$ |
| $B$ | Gauge subsystem (unprotected, flexible) | $2^r$ |
| $\mathcal{C}^\perp$ | Non-code space | $2^n - 2^{k+r}$ |

### Group Hierarchy

$$\mathcal{S} = Z(\mathcal{G}) \cap \mathcal{G} \subseteq \mathcal{G} \subseteq N(\mathcal{G}) \subseteq \mathcal{P}_n$$

- **Gauge group** $\mathcal{G}$: Acts trivially on logical subsystem $A$
- **Stabilizer** $\mathcal{S}$: Center of gauge group (fixes entire code space)
- **Normalizer** $N(\mathcal{G})$: Contains logical operators

### The Bacon-Shor Code Family

**Parameters:** $[[mn, 1, (m-1)(n-1), \min(m,n)]]$

| Aspect | Formula |
|--------|---------|
| Physical qubits | $m \cdot n$ |
| Logical qubits | $1$ |
| Gauge qubits | $(m-1)(n-1)$ |
| Distance | $\min(m, n)$ |
| X-gauge operators | $m(n-1)$ (horizontal XX) |
| Z-gauge operators | $(m-1)n$ (vertical ZZ) |

### Key Advantages

| Advantage | Mechanism |
|-----------|-----------|
| **Weight reduction** | Gauge operators have weight 2 (vs weight-$m$ or -$n$ stabilizers) |
| **Natural fault tolerance** | Single fault in weight-2 measurement → at most 1 data error |
| **Higher threshold** | Simpler circuits → fewer error opportunities |
| **Single-shot potential** | Gauge redundancy enables one-round correction (in 3D) |

---

## Key Equations

### Code Parameters
$$k + r + |\mathcal{S}| = n$$

### Singleton Bound
$$k + r \leq n - 2(d-1)$$

### Error Correction Condition
$$P_{\mathcal{C}} E_a^\dagger E_b P_{\mathcal{C}} = I_A \otimes B_{ab}$$

### Bacon-Shor Distance
$$d = \min(m, n)$$

---

## Daily Breakdown

### Day 715: Introduction to Subsystem Codes
- Motivation and historical context
- Subsystem vs subspace decomposition
- The $[[n, k, r, d]]$ parameter notation
- First example: $[[4, 1, 1, 2]]$ code

### Day 716: Gauge Operators and Logical Operators
- Gauge group structure and properties
- Bare vs dressed logical operators
- Gauge transformations
- Relationship between gauge and stabilizer

### Day 717: The Bacon-Shor Code
- $m \times n$ lattice construction
- Horizontal XX and vertical ZZ gauge operators
- Stabilizer generators as gauge products
- Connection to Shor code ($3 \times 3$ case)

### Day 718: Subsystem Code Properties
- Distance definitions (bare vs dressed)
- Subsystem Knill-Laflamme conditions
- Singleton bound for subsystem codes
- Gauge-distance trade-off

### Day 719: Advantages of Subsystem Codes
- Measurement weight reduction
- Fault-tolerance benefits
- Single-shot error correction concepts
- Resource trade-off analysis

### Day 720: Subsystem Codes and Fault Tolerance
- Fault-tolerant syndrome extraction
- Transversal gates on Bacon-Shor
- Gauge fixing for computation
- Threshold analysis

### Day 721: Week Synthesis
- Comprehensive review
- Problem set across all difficulties
- Connections to advanced topics
- Preparation for Week 104

---

## Computational Skills

```python
# Key implementations from this week

# 1. Bacon-Shor code construction
class BaconShorCode:
    def __init__(self, m, n):
        self.m, self.n = m, n
        self.k = 1
        self.r = (m-1) * (n-1)
        self.d = min(m, n)

# 2. Gauge operators
def generate_x_gauges(m, n):
    """Horizontal XX pairs."""
    return [(i, j, i, j+1) for i in range(m) for j in range(n-1)]

def generate_z_gauges(m, n):
    """Vertical ZZ pairs."""
    return [(i, j, i+1, j) for i in range(m-1) for j in range(n)]

# 3. Syndrome extraction
def compute_stabilizer_syndrome(gauge_outcomes, gauge_to_stab_map):
    """Compute stabilizer syndrome from gauge measurements."""
    return [np.prod([gauge_outcomes[g] for g in stab]) for stab in gauge_to_stab_map]

# 4. Fault-tolerant circuit analysis
def max_error_spread(measurement_weight):
    """Maximum data errors from single measurement fault."""
    return measurement_weight - 1
```

---

## References

### Primary Sources
- Nielsen & Chuang, Chapter 10.5 (Subsystem codes)
- Preskill Lecture Notes, Chapter 7
- Poulin, "Stabilizer Formalism for Operator Quantum Error Correction" (2005)
- Bacon, "Operator quantum error-correcting subsystems" (2006)

### Key Papers
- Bacon-Shor original: Bacon, "Operator quantum error-correcting subsystems for self-correcting quantum memories" (2006)
- Subsystem codes: Kribs, Laflamme, Poulin, "Unified and Generalized Approach to Quantum Error Correction" (2005)
- Single-shot: Bombin, "Single-Shot Fault-Tolerant Quantum Error Correction" (2015)

### Online Resources
- [Error Correction Zoo: Subsystem codes](https://errorcorrectionzoo.org/c/subsystem_stabilizer)
- [IBM Qiskit: Error Correction](https://learning.quantum.ibm.com/)

---

## Connections

### Prerequisites (from earlier weeks)
- Week 97-100: QEC fundamentals
- Week 101: Advanced stabilizer theory
- Week 102: Gottesman-Knill theorem

### Leads to (future weeks)
- Week 104: Code capacity
- Month 27: Stabilizer formalism deep dive
- Month 29: Topological codes (subsystem variants)
- Month 31: Fault-tolerant QC

---

## Summary

Subsystem codes provide a powerful framework that generalizes stabilizer codes by introducing gauge freedom. The Bacon-Shor code family demonstrates the key advantages: weight-2 measurements that are naturally fault-tolerant, leading to higher error thresholds. While subsystem codes "waste" some qubits on gauge degrees of freedom, this trade-off enables simpler, more robust implementations—a crucial consideration for near-term quantum hardware with limited connectivity and high error rates.

---

**Week 103 Complete!**

**Next:** Week 104 — Code Capacity (Days 722-728)
