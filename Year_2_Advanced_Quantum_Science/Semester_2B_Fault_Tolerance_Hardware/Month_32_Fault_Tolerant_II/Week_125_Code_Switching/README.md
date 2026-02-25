# Week 125: Code Switching & Gauge Fixing

## Overview

**Days:** 869-875 (7 days)
**Month:** 32 (Fault-Tolerant Quantum Computing II)
**Semester:** 2B (Fault Tolerance & Hardware)
**Topic:** Code Switching, Gauge Fixing, and Alternative Paths to Universality

---

## Status: In Progress

| Day | Date | Topic | Status |
|-----|------|-------|--------|
| 869 | Monday | Code Switching Motivation | Not Started |
| 870 | Tuesday | Steane to Reed-Muller Switching | Not Started |
| 871 | Wednesday | Subsystem Codes Review | Not Started |
| 872 | Thursday | Gauge Fixing Protocols | Not Started |
| 873 | Friday | Lattice Surgery as Code Switching | Not Started |
| 874 | Saturday | Computational Lab | Not Started |
| 875 | Sunday | Week Synthesis | Not Started |

---

## Learning Objectives

By the end of this week, you should be able to:

1. **Explain** why code switching is necessary for fault-tolerant universality
2. **Design** fault-tolerant switching circuits between Steane [[7,1,3]] and Reed-Muller [[15,1,3]] codes
3. **Distinguish** between stabilizer codes and subsystem codes (gauge qubits)
4. **Implement** gauge fixing protocols to switch between equivalent code representations
5. **Analyze** lattice surgery operations as instances of code switching
6. **Compare** code switching approaches with magic state distillation for universality
7. **Calculate** resource overheads for different code switching protocols
8. **Evaluate** trade-offs between code switching and magic state approaches

---

## Core Concepts

### Why Code Switching?

The **Eastin-Knill theorem** (Week 123) states that no single quantum code has a transversal universal gate set. However, **different codes have different transversal gates**:

| Code | Transversal Gates |
|------|-------------------|
| Steane [[7,1,3]] | {H, S, CNOT} (Clifford) |
| Reed-Muller [[15,1,3]] | {T, CCZ, CNOT} |
| Color Code | {H, S, CNOT, CCZ} |

**Key Insight:** By switching between codes, we can access **complementary transversal gate sets** to achieve universality!

### The Universality Equation

$$\boxed{\text{Steane transversal} \cup \text{RM transversal} = \{H, S, T, \text{CNOT}\} = \text{Universal}}$$

### What is Gauge Fixing?

In **subsystem codes**, information is encoded in a subsystem rather than a subspace. The code has additional degrees of freedom called **gauge qubits**.

**Gauge fixing** = measuring gauge operators to convert a subsystem code to a stabilizer code.

$$\boxed{\text{Subsystem Code} \xrightarrow{\text{gauge fixing}} \text{Stabilizer Code}}$$

This provides another route to code switching without explicit encoding/decoding.

---

## Weekly Breakdown

### Day 869: Code Switching Motivation

- The universality problem and Eastin-Knill
- Complementary transversal gate sets
- Code switching as an alternative to magic states
- Fault tolerance requirements for code conversion
- Historical development and recent breakthroughs

### Day 870: Steane to Reed-Muller Switching

- [[7,1,3]] Steane code structure review
- [[15,1,3]] Reed-Muller code construction
- Fault-tolerant encoding/decoding circuits
- The Anderson-Duclos-Cianci-Svore protocol
- Error analysis during code switching
- Recent Quantinuum experimental demonstration

### Day 871: Subsystem Codes Review

- Subsystem vs. subspace encoding
- The Bacon-Shor code [[9,1,3]]
- Gauge operators and gauge qubits
- Stabilizers vs. gauge generators
- Simplified error correction in subsystem codes
- Trade-offs: fewer measurements, weaker protection

### Day 872: Gauge Fixing Protocols

- Gauge fixing as partial measurement
- Converting Bacon-Shor to Shor code
- Measurement-based gauge fixing circuits
- Achieving transversal gates via gauge manipulation
- The Paetznick-Reichardt universality result
- Connection to subsystem lattice surgery

### Day 873: Lattice Surgery as Code Switching

- Review of lattice surgery (merge/split operations)
- Lattice surgery as code concatenation/deconcat
- Topological interpretation of code switching
- Surface code to color code conversion
- Generalized lattice surgery framework
- Fault-tolerant interfaces

### Day 874: Computational Lab

- Simulate Steane-RM code switching
- Implement Bacon-Shor gauge fixing
- Verify transversal gates after code switch
- Compare error rates with magic state approach
- Resource counting and optimization

### Day 875: Week Synthesis

- Code switching vs. magic states: comprehensive comparison
- Resource trade-offs analysis
- Hybrid approaches
- State of the art (2025-2026)
- Preparation for hardware implementation topics

---

## Key Equations

**Steane Code Transversal Gates:**
$$\boxed{\bar{H} = H^{\otimes 7}, \quad \bar{S} = S^{\otimes 7}, \quad \overline{\text{CNOT}} = \text{CNOT}^{\otimes 7}}$$

**Reed-Muller Transversal T:**
$$\boxed{\bar{T} = T^{\otimes 15} \quad \text{(on [[15,1,3]] RM code)}}$$

**Bacon-Shor Gauge Operators:**
$$\boxed{G_X^{(i,j)} = X_i X_j, \quad G_Z^{(k,l)} = Z_k Z_l \quad \text{(adjacent pairs)}}$$

**Code Switching Condition:**
$$\boxed{\mathcal{C}_1 \text{ and } \mathcal{C}_2 \text{ share logical basis} \Rightarrow \text{fault-tolerant switch possible}}$$

**Resource Comparison:**
$$\boxed{\text{Code switch: } O(n) \text{ ancillas} \quad vs. \quad \text{Distillation: } O(n \log(1/\epsilon)) \text{ ancillas}}$$

---

## Transversal Gate Complementarity

| Operation | Steane [[7,1,3]] | RM [[15,1,3]] | Combined |
|-----------|------------------|---------------|----------|
| X | Yes | Yes | Yes |
| Z | Yes | Yes | Yes |
| H | Yes | No | Yes |
| S | Yes | No | Yes |
| T | **No** | **Yes** | Yes |
| CNOT | Yes | Yes | Yes |
| CCZ | No | Yes | Yes |

The combination provides a **universal gate set** through code switching!

---

## Prerequisites

### From Week 123 (Transversal Gates)
- Eastin-Knill theorem
- Transversal gate definition
- CSS code transversal gates

### From Week 124 (Universal FT Computation)
- Magic state injection
- Gate teleportation
- Resource state preparation

### From Month 27 (Stabilizer Formalism)
- Stabilizer groups
- Logical operators
- Code space structure

---

## Resources

### Primary References

- Anderson, Duclos-Cianci, Svore, "Fault-Tolerant Conversion between the Steane and Reed-Muller Quantum Codes" (2014)
- Paetznick & Reichardt, "Universal Fault-Tolerant Quantum Computation with Only Transversal Gates and Error Correction" (2013)
- Bombin, "Gauge Color Codes" (2015)

### Key Papers

- Kubica & Beverland, "Universal Transversal Gates with Color Codes" (2015)
- Webster et al., "Universal fault-tolerant quantum computing with stabilizer codes" (2022)
- Quantinuum, "Experimental fault-tolerant code switching" Nature Physics (2024)

### Recent Developments

- Efficient fault-tolerant code switching via one-way transversal CNOT gates (Quantum, 2025)
- Subsystem codes with high thresholds by gauge fixing (PRX, 2021)
- Lattice surgery compilation beyond surface codes (2025)

### Online Resources

- [Error Correction Zoo - Code Switching](https://errorcorrectionzoo.org/)
- [Quantinuum Technical Reports](https://www.quantinuum.com/)
- [Preskill Notes - Fault Tolerance](http://theory.caltech.edu/~preskill/ph219/)

---

## Connections

### From Previous Weeks

- Week 121: Magic states as alternative to code switching
- Week 122: Distillation overhead motivates code switching
- Week 123: Eastin-Knill forces us to combine techniques
- Week 124: Universal FT computation framework

### To Future Weeks

- Week 126: Threshold theorem and fault tolerance levels
- Month 33: Hardware-specific code implementations
- Month 34: Near-term quantum error correction

---

## Historical Note

Code switching was proposed in the early 2010s but was considered impractical due to circuit complexity. The 2024-2025 experimental demonstrations by Quantinuum showed that **code switching can actually outperform magic state distillation** for certain tasks, producing magic states with infidelity below $5.1 \times 10^{-4}$.

---

## Summary

Week 125 explores **code switching** and **gauge fixing** as alternatives to magic state distillation for achieving universal fault-tolerant quantum computation. By exploiting the complementary transversal gate sets of different codes (e.g., Steane for Clifford, Reed-Muller for T), we can achieve universality without the overhead of state distillation. Gauge fixing in subsystem codes provides yet another mechanism for switching between code representations. These techniques represent the cutting edge of fault-tolerant quantum computing research.

---

*"Code switching turns the Eastin-Knill theorem from a no-go theorem into a recipe for universality."*
--- Modern Fault-Tolerant Computing Perspective

---

**Last Updated:** February 2026
**Status:** In Progress
