# Week 172: Topological Codes

## Overview

**Days:** 1198-1204
**Theme:** Topological Quantum Error Correction

This week explores topological quantum error-correcting codes, which represent the leading approach to fault-tolerant quantum computing in near-term devices. We study Kitaev's toric code, its planar variant (surface code), anyonic excitations, and the practical aspects of surface code quantum computing.

---

## Daily Schedule

### Day 1198 (Monday): The Toric Code

**Topics:**
- Kitaev's toric code on the torus
- Vertex and plaquette operators
- Ground state degeneracy and topology
- Code parameters [[2n^2, 2, n]]

**Key Concepts:**
- Stabilizer generators: $$A_v = \prod_{e \ni v} X_e$$, $$B_p = \prod_{e \in p} Z_e$$
- Four-fold ground state degeneracy from topology

### Day 1199 (Tuesday): Anyonic Excitations

**Topics:**
- Electric charges (e) and magnetic vortices (m)
- Anyon fusion rules
- Braiding statistics
- Connection to topological order

**Key Result:**
$$e \times m = \epsilon$$ (fermion), braiding gives $$-1$$ phase

### Day 1200 (Wednesday): Surface Code Basics

**Topics:**
- Planar variant with boundaries
- Rough and smooth boundaries
- Code parameters [[n, 1, d]]
- Logical operators as string operators

**Key Insight:**
Surface code encodes 1 logical qubit with distance = lattice linear dimension

### Day 1201 (Thursday): Surface Code Architecture

**Topics:**
- Physical layout and connectivity
- Syndrome measurement circuits
- Minimum-weight perfect matching decoder
- Error threshold (~1%)

**Practical Focus:**
- Realistic noise models
- Measurement errors
- Decoder implementation

### Day 1202 (Friday): Logical Operations

**Topics:**
- Transversal gates (limited to Paulis)
- Lattice surgery for CNOT
- Magic state injection and distillation
- Universal computation schemes

**Key Challenge:**
No transversal non-Clifford gates

### Day 1203 (Saturday): Computational Workshop

**Activities:**
- Implement toric code simulation
- Surface code syndrome extraction
- MWPM decoder
- Error threshold estimation

### Day 1204 (Sunday): Month Review and Integration

**Activities:**
- Comprehensive QEC review
- Cross-week integration
- Oral exam practice
- Month 43 completion

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Construct** the toric code and identify its stabilizers
2. **Explain** anyonic excitations and their braiding properties
3. **Analyze** the surface code including boundaries and logical operators
4. **Design** syndrome measurement circuits for the surface code
5. **Apply** minimum-weight perfect matching for decoding
6. **Describe** fault-tolerant operations using lattice surgery
7. **Evaluate** the surface code for practical quantum computing

---

## Key Formulas

| Concept | Formula |
|---------|---------|
| Toric code parameters | $$[[2L^2, 2, L]]$$ on $$L \times L$$ torus |
| Vertex operator | $$A_v = \prod_{e \ni v} X_e$$ |
| Plaquette operator | $$B_p = \prod_{e \in p} Z_e$$ |
| Surface code parameters | $$[[d^2, 1, d]]$$ (rotated) |
| Error threshold | $$p_{\text{th}} \approx 1\%$$ (phenomenological) |

---

## Files in This Week

| File | Description |
|------|-------------|
| `README.md` | This overview document |
| `Review_Guide.md` | Comprehensive theory review (2000+ words) |
| `Problem_Set.md` | 25-30 PhD qualifying exam level problems |
| `Problem_Solutions.md` | Complete solutions with detailed derivations |
| `Oral_Practice.md` | Oral exam questions and discussion points |
| `Self_Assessment.md` | Checklist and self-evaluation rubric |

---

## References

1. Kitaev, A. "Fault-tolerant quantum computation by anyons" Ann. Phys. 303, 2-30 (2003)
2. Dennis, E. et al. "Topological quantum memory" J. Math. Phys. 43, 4452 (2002)
3. Fowler, A.G. et al. "Surface codes: Towards practical large-scale quantum computation" PRA 86, 032324 (2012)
4. Terhal, B.M. "Quantum error correction for quantum memories" RMP 87, 307 (2015)

---

**Week 172 Created:** February 10, 2026
