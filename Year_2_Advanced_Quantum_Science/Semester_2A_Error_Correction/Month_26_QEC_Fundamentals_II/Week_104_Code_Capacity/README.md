# Week 104: Code Capacity

## Overview

**Days:** 722-728 (7 days)
**Month:** 26 (QEC Fundamentals II) — Final Week
**Topic:** Fundamental Limits on Quantum Error Correction

---

## Status: ✅ COMPLETE

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 722 | Monday | Introduction to Code Capacity | ✅ Complete |
| 723 | Tuesday | Quantum Channel Capacity | ✅ Complete |
| 724 | Wednesday | Hashing Bound and Threshold Theorem | ✅ Complete |
| 725 | Thursday | LDPC Code Capacity | ✅ Complete |
| 726 | Friday | Capacity Bounds and Calculations | ✅ Complete |
| 727 | Saturday | Practical Capacity Applications | ✅ Complete |
| 728 | Sunday | Week & Month Synthesis | ✅ Complete |

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Define** and calculate classical and quantum channel capacities
2. **State** the Lloyd-Shor-Devetak theorem for quantum capacity
3. **Compute** the hashing bound for Pauli channels
4. **Explain** threshold behavior from capacity perspective
5. **Describe** LDPC codes and their capacity-approaching properties
6. **Apply** numerical methods for capacity bounds
7. **Analyze** practical implications of capacity theory
8. **Compare** code efficiency to fundamental limits

---

## Core Concepts

### Channel Capacity Hierarchy

$$C_E(\mathcal{N}) \geq C(\mathcal{N}) \geq P(\mathcal{N}) \geq Q(\mathcal{N})$$

| Capacity | Description | Use |
|----------|-------------|-----|
| $C_E$ | Entanglement-assisted classical | Classical bits with shared entanglement |
| $C$ | Classical | Classical bits over quantum channel |
| $P$ | Private | Secure classical communication |
| $Q$ | Quantum | Quantum state transmission |

### Key Equations

**Classical BSC Capacity:**
$$C = 1 - H(p)$$

**Quantum Capacity (LSD):**
$$Q(\mathcal{N}) = \lim_{n \to \infty} \frac{1}{n} \max_\rho I_c(\rho^{(n)}, \mathcal{N}^{\otimes n})$$

**Coherent Information:**
$$I_c(\rho, \mathcal{N}) = S(\mathcal{N}(\rho)) - S(\mathcal{N}^c(\rho))$$

**Hashing Bound (Depolarizing):**
$$Q \geq 1 - H(p) - p\log_2 3$$

### Channel Classification

| Type | Definition | $Q$ Computation |
|------|------------|-----------------|
| **Degradable** | $\mathcal{N}^c = \mathcal{D} \circ \mathcal{N}$ | $Q = Q^{(1)} = \max I_c$ |
| **Anti-degradable** | $\mathcal{N} = \mathcal{D} \circ \mathcal{N}^c$ | $Q = 0$ |
| **General** | Neither | Requires regularization |

### Important Thresholds

| Channel | Hashing Threshold | Notes |
|---------|-------------------|-------|
| Depolarizing | 18.93% | Believed tight |
| Erasure | 50% | Exact |
| Amplitude damping | 50% | Exact (γ threshold) |
| Pure dephasing | 50% | $Q = 1 - H(p)$ |
| Biased Z (10:1) | ~29% | Bias helps |

---

## Daily Breakdown

### Day 722: Introduction to Code Capacity
- Shannon's classical capacity theorem
- Multiple quantum capacities (Q, C, C_E, P)
- Connection to QEC rate constraints
- Basic capacity calculations

### Day 723: Quantum Channel Capacity
- Coherent information derivation
- Lloyd-Shor-Devetak theorem
- Degradable and anti-degradable channels
- Superadditivity and regularization

### Day 724: Hashing Bound and Threshold Theorem
- Hashing bound derivation
- Random stabilizer codes
- Threshold theorem statement
- Code families approaching capacity

### Day 725: LDPC Code Capacity
- Classical LDPC codes
- Belief propagation decoding
- Quantum LDPC constructions
- Good qLDPC codes breakthrough

### Day 726: Capacity Bounds and Calculations
- Upper bounds (Rains, PPT)
- Lower bounds (coherent information)
- Numerical optimization methods
- SDP approaches

### Day 727: Practical Capacity Applications
- Realistic noise models
- Resource overhead analysis
- Threshold gap analysis
- Application to quantum communication

### Day 728: Week & Month Synthesis
- Comprehensive review
- Integration problems
- Month 26 summary
- Preparation for Month 27

---

## Key Results Summary

### Good qLDPC Codes

| Property | Surface Code | Good qLDPC |
|----------|--------------|------------|
| Rate $k/n$ | $O(1/d^2)$ | $\Theta(1)$ |
| Distance | $d$ | $\Theta(n)$ |
| Check weight | 4 | $O(1)$ (~10-20) |
| Capacity efficiency | ~0% | ~100% |

### Resource Overhead

$$\text{Overhead}_{\min} = \frac{1}{Q(\mathcal{N})}$$

$$\text{Efficiency} = \frac{R_{\text{code}}}{Q(\mathcal{N})}$$

---

## Computational Skills

```python
# Key calculations from this week

def hashing_bound(p):
    """Hashing bound for depolarizing channel."""
    H_p = -p*np.log2(p) - (1-p)*np.log2(1-p)
    return max(0, 1 - H_p - p*np.log2(3))

def coherent_information(rho, channel):
    """Compute I_c(ρ, N)."""
    rho_out = channel.apply(rho)
    rho_env = channel.complementary(rho)
    return von_neumann_entropy(rho_out) - von_neumann_entropy(rho_env)

def code_efficiency(code_rate, channel_capacity):
    """Compute efficiency η = R/Q."""
    return code_rate / channel_capacity
```

---

## References

### Primary Sources
- Nielsen & Chuang, Chapter 12 (Quantum Shannon Theory)
- Preskill Lecture Notes, Chapter 7
- Wilde, "Quantum Information Theory" (comprehensive)

### Key Papers
- Lloyd, Shor, Devetak: Quantum capacity theorem (1997-2005)
- Panteleev-Kalachev: Good qLDPC codes (2020)
- Richardson-Urbanke: LDPC capacity achievement

### Online Resources
- [Error Correction Zoo](https://errorcorrectionzoo.org/)
- arXiv quantum information theory papers

---

## Connections

### Prerequisites (Month 26)
- Week 101: Stabilizer theory (random codes)
- Week 102: Gottesman-Knill (simulation)
- Week 103: Subsystem codes (alternative framework)

### Leads to (Future)
- Month 27: Stabilizer formalism deep dive
- Month 28: Advanced stabilizer codes
- Month 29: Topological codes (different capacity regime)

---

## Summary

Code capacity provides fundamental limits on quantum error correction. The quantum capacity $Q(\mathcal{N})$ determines the maximum rate at which quantum information can be reliably transmitted or stored in the presence of noise. The hashing bound gives an achievable lower bound using random stabilizer codes, while good qLDPC codes (discovered in 2020) can approach this limit with polynomial complexity. Understanding capacity is essential for evaluating code efficiency and designing optimal QEC systems.

---

## Month 26 Complete! 🎉

**QEC Fundamentals II** covered:
- Advanced stabilizer theory (Week 101)
- Gottesman-Knill theorem (Week 102)
- Subsystem codes (Week 103)
- Code capacity (Week 104)

**Next:** Month 27 — Stabilizer Formalism (Days 729-756)
