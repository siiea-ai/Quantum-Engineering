# Month 29: Topological Codes

## Overview

**Days:** 785-812 (28 days)
**Weeks:** 113-116
**Semester:** 2A (Error Correction)
**Focus:** Toric codes, anyonic excitations, surface codes, and topological error correction

---

## Status: ✅ COMPLETE

| Week | Days | Topic | Status |
|------|------|-------|--------|
| 113 | 785-791 | Toric Code Fundamentals | ✅ Complete |
| 114 | 792-798 | Anyons & Topological Order | ✅ Complete |
| 115 | 799-805 | Surface Code Implementation | ✅ Complete |
| 116 | 806-812 | Error Chains & Logical Operations | ✅ Complete |

**Progress:** 28/28 days (100%)

---

## Learning Objectives

By the end of this month, you should be able to:

1. **Construct** toric codes on 2D lattices with periodic boundaries
2. **Identify** star and plaquette stabilizers and their commutation
3. **Classify** anyonic excitations (e, m particles) and their statistics
4. **Analyze** homology and logical operators on the torus
5. **Implement** surface codes with boundaries (planar codes)
6. **Trace** error chains and their equivalence classes
7. **Design** logical gate implementations via code deformation
8. **Connect** topological order to fault-tolerant quantum memory

---

## Weekly Breakdown

### Week 113: Toric Code Fundamentals (Days 785-791)

The toric code, introduced by Kitaev (1997), is the foundational topological code that encodes logical qubits in the ground space of a local Hamiltonian.

**Core Topics:**
- Toric lattice geometry and qubit placement
- Star operators $A_v = \prod_{e \ni v} X_e$
- Plaquette operators $B_p = \prod_{e \in p} Z_e$
- Ground state degeneracy and logical qubits
- Toric code as CSS code
- Dual lattice interpretation

**Key Equations:**
$$A_v = \prod_{e \ni v} X_e, \quad B_p = \prod_{e \in \partial p} Z_e$$
$$[[2L^2, 2, L]] \text{ (toric code parameters)}$$

### Week 114: Anyons & Topological Order (Days 792-798)

Violations of stabilizers create quasiparticle excitations with exotic statistics, providing the foundation for topological quantum computing.

**Core Topics:**
- Electric charges (e) from star violations
- Magnetic fluxes (m) from plaquette violations
- Mutual semionic statistics
- Fusion rules: e × e = 1, m × m = 1, e × m = ε
- Braiding and topological phases
- Topological order and long-range entanglement

**Key Equations:**
$$\text{Braiding: } e \circlearrowleft m \Rightarrow \text{phase } e^{i\pi} = -1$$

### Week 115: Surface Code Implementation (Days 799-805)

The planar surface code adapts toric code ideas to practical architectures with boundaries, becoming the leading candidate for near-term QEC.

**Core Topics:**
- Smooth and rough boundaries
- Surface code on a square lattice
- $[[d^2, 1, d]]$ or $[[(d^2+1)/2, 1, d]]$ parameters
- Defect-based logical qubits
- Comparison with toric code
- Measurement-based preparation

**Key Equations:**
$$\text{Planar: } [[n, 1, d]] \text{ with } n \approx 2d^2 - 1$$

### Week 116: Error Chains & Logical Operations (Days 806-812)

Understanding how errors form chains and implementing logical gates through topological operations.

**Core Topics:**
- Error chains and homology classes
- Minimum weight decoding on chains
- Logical X and Z as non-contractible loops
- Twist defects and their braiding
- Lattice surgery for logical gates
- Month synthesis and integration

**Key Equations:**
$$\bar{X}_1 = \prod_{e \in \gamma_1} X_e, \quad \bar{Z}_1 = \prod_{e \in \gamma_1^*} Z_e$$

---

## Key Concepts

### Toric Code Stabilizers

| Operator | Type | Location | Formula |
|----------|------|----------|---------|
| Star $A_v$ | X-type | Vertex v | $\prod_{e \ni v} X_e$ |
| Plaquette $B_p$ | Z-type | Face p | $\prod_{e \in \partial p} Z_e$ |

### Anyon Types in Toric Code

| Particle | Created by | Stabilizer violated | Self-statistics |
|----------|------------|---------------------|-----------------|
| e (charge) | Z error | Star $A_v$ | Boson |
| m (flux) | X error | Plaquette $B_p$ | Boson |
| ε = e×m | Y error | Both | Fermion |

### Code Parameters Comparison

| Code | Qubits | Logical | Distance | Encoding Rate |
|------|--------|---------|----------|---------------|
| Toric [[2L²,2,L]] | 2L² | 2 | L | 1/L² |
| Planar [[d²,1,d]] | ~d² | 1 | d | 1/d² |

---

## Prerequisites

### From Month 28 (Advanced Stabilizer Applications)
- Fault-tolerant operations
- Threshold theorems
- Decoding algorithms (MWPM)
- Practical QEC constraints

### Mathematical Background
- Basic algebraic topology (homology groups)
- Lattice geometry
- Group theory (Abelian anyons)

---

## Resources

### Primary References
- Kitaev, "Fault-tolerant quantum computation by anyons" (2003)
- Dennis et al., "Topological quantum memory" (2002)
- Fowler et al., "Surface codes: Towards practical large-scale quantum computation" (2012)

### Key Papers
- Bombin & Martin-Delgado, "Topological Quantum Distillation" (2006)
- Raussendorf & Harrington, "Fault-Tolerant Quantum Computation with High Threshold" (2007)
- Litinski, "A Game of Surface Codes" (2019)

### Online Resources
- [Toric Code - Error Correction Zoo](https://errorcorrectionzoo.org/c/toric)
- [Surface Code - Error Correction Zoo](https://errorcorrectionzoo.org/c/surface)
- [Toric Code Introduction](https://leftasexercise.com/2019/03/25/qec-an-introduction-to-toric-codes/)

---

## Connections

### From Previous Months
- Month 27: Stabilizer Formalism → CSS structure of toric code
- Month 28: Decoding Algorithms → MWPM for topological codes

### To Future Months
- Month 30: Advanced Topological Codes → Twist defects, color codes
- Semester 2B: Quantum Algorithms → Topological quantum computing

---

## Summary

Month 29 explores the beautiful connection between topology and quantum error correction. The toric code demonstrates how global topological properties can protect quantum information from local noise. Understanding anyonic excitations provides physical intuition for error correction, while the surface code brings these ideas to practical implementation. This month establishes the foundation for the most promising approach to scalable fault-tolerant quantum computing.
