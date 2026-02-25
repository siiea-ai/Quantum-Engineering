# Week 165: Quantum Complexity Theory

## Overview

**Days:** 1149-1155
**Theme:** Quantum Computational Complexity and Query Complexity
**Hours:** 45 hours (7.5 hours/day × 6 days)

---

## Learning Objectives

By the end of this week, you should be able to:

1. Define BQP precisely and explain its relationship to classical complexity classes
2. Define QMA and explain the significance of QMA-complete problems
3. Prove the Local Hamiltonian problem is QMA-complete (proof sketch)
4. Apply the polynomial method to derive quantum query lower bounds
5. Prove Grover's algorithm is optimal using the polynomial method
6. Analyze oracle problems and quantum-classical query separations
7. Explain the oracle evidence for P ≠ BQP and BQP ≠ NP

---

## Daily Schedule

### Day 1149 (Monday): BQP Fundamentals

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: BQP definition, containment proofs |
| 2:00-5:00 | 3 hrs | Review: Quantum Turing machines, quantum circuits |
| 7:00-8:30 | 1.5 hrs | Written practice: BQP characterization |

**Topics:**
- Formal definition of BQP via quantum Turing machines
- Circuit-based definition of BQP
- Proof: P ⊆ BQP (classical computation is a special case)
- Proof: BQP ⊆ PSPACE (simulation argument)

### Day 1150 (Tuesday): BQP and Classical Classes

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: BQP vs. BPP, NP relationships |
| 2:00-5:00 | 3 hrs | Review: Oracle separations, relativization |
| 7:00-8:30 | 1.5 hrs | Oral practice: Explain BQP to a committee |

**Topics:**
- BQP ⊆ PP (postselection argument)
- Oracle separations: O such that BQP^O ≠ BPP^O
- Simon's problem as evidence for quantum advantage
- Recursive Fourier sampling

### Day 1151 (Wednesday): QMA Definition and Properties

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: QMA verifier constructions |
| 2:00-5:00 | 3 hrs | Review: Quantum witnesses, completeness/soundness |
| 7:00-8:30 | 1.5 hrs | Written practice: QMA problem analysis |

**Topics:**
- QMA formal definition (quantum verifier, quantum witness)
- Completeness and soundness parameters
- QMA = QMA(1/3, 2/3) (error reduction)
- QMA vs. NP: quantum witnesses can encode more information

### Day 1152 (Thursday): QMA-Complete Problems

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Local Hamiltonian reductions |
| 2:00-5:00 | 3 hrs | Review: Proof of k-local Hamiltonian QMA-completeness |
| 7:00-8:30 | 1.5 hrs | Oral practice: Explain QMA-completeness |

**Topics:**
- Local Hamiltonian problem definition
- k-local Hamiltonian is QMA-complete for k ≥ 2
- Perturbation theory in complexity reductions
- Connection to ground state physics

### Day 1153 (Friday): Query Complexity and Polynomial Method

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Query complexity lower bounds |
| 2:00-5:00 | 3 hrs | Review: Polynomial method, degree bounds |
| 7:00-8:30 | 1.5 hrs | Computational lab: Query algorithm simulation |

**Topics:**
- Query model (black-box/oracle model)
- Polynomial method: algorithm → multilinear polynomial
- Approximate degree and quantum query complexity
- Q(f) ≥ deg̃(f)/2

### Day 1154 (Saturday): Grover Lower Bound and Oracle Problems

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Problem solving: Optimality proofs |
| 2:00-5:00 | 3 hrs | Review: Adversary method, oracle problems |
| 7:00-8:30 | 1.5 hrs | Written practice: Comprehensive problems |

**Topics:**
- Proof: Q(OR) = Ω(√n) using polynomial method
- Adversary method for lower bounds
- Deutsch-Jozsa, Bernstein-Vazirani query analysis
- Quantum counting complexity

### Day 1155 (Sunday): Integration and Assessment

| Time | Duration | Activity |
|------|----------|----------|
| 9:00-12:00 | 3 hrs | Comprehensive problem set completion |
| 2:00-5:00 | 3 hrs | Oral exam practice session |
| 7:00-8:30 | 1.5 hrs | Self-assessment and gap analysis |

---

## Key Concepts Summary

### BQP (Bounded-Error Quantum Polynomial Time)

$$\text{BQP} = \{L : \exists \text{ uniform quantum circuit family } \{C_n\} \text{ s.t.}$$
$$x \in L \Rightarrow \Pr[C_{|x|}(x) = 1] \geq 2/3$$
$$x \notin L \Rightarrow \Pr[C_{|x|}(x) = 1] \leq 1/3\}$$

**Key containments:**
$$\text{P} \subseteq \text{BPP} \subseteq \text{BQP} \subseteq \text{PP} \subseteq \text{PSPACE}$$

### QMA (Quantum Merlin-Arthur)

$$\text{QMA} = \{L : \exists \text{ polynomial-time quantum verifier } V \text{ s.t.}$$
$$x \in L \Rightarrow \exists |w\rangle : \Pr[V(x, |w\rangle) = 1] \geq 2/3$$
$$x \notin L \Rightarrow \forall |w\rangle : \Pr[V(x, |w\rangle) = 1] \leq 1/3\}$$

### Local Hamiltonian Problem

**Input:** k-local Hamiltonian H = ∑ᵢ Hᵢ on n qubits, thresholds a < b

**Promise:** Either λ_min(H) ≤ a or λ_min(H) ≥ b

**Output:** Determine which case holds

**Theorem (Kempe-Kitaev-Regev):** k-local Hamiltonian is QMA-complete for k ≥ 2.

### Query Complexity

**Polynomial Method:**
If a quantum algorithm computes f with T queries:
- Final state amplitudes are multilinear polynomials in x₁,...,xₙ of degree ≤ T
- Acceptance probability is a polynomial of degree ≤ 2T

$$Q(f) \geq \frac{\widetilde{\text{deg}}(f)}{2}$$

**Grover Lower Bound:**
$$Q(\text{OR}_n) = \Omega(\sqrt{n})$$

---

## Resources

### Primary Reading
- Watrous, "Quantum Computational Complexity" (comprehensive survey)
- Kempe, Kitaev, Regev, "The Complexity of the Local Hamiltonian Problem"
- Beals et al., "Quantum Lower Bounds by Polynomials"

### Supplementary Materials
- [Quantum Complexity Theory - Cambridge Course](https://www.cl.cam.ac.uk/teaching/2526/QCT/)
- [CMU Lecture 23: Quantum Complexity Theory](https://www.cs.cmu.edu/~odonnell/quantum15/lecture23.pdf)
- Aaronson, "Quantum Computing Since Democritus" (Chapter 10)

### Video Lectures
- Sevag Gharibian's Quantum Complexity Theory lectures
- Henry Yuen's COMS 4281 (Fall 2024) lectures

---

## Connections

### From Previous Weeks
| Previous Topic | Connection to This Week |
|----------------|------------------------|
| Quantum algorithms (Week 162-163) | Algorithms define BQP |
| Quantum gates (Week 161) | Circuit model for BQP |
| Grover's algorithm | Optimality proof via polynomials |

### To Future Topics
| This Week's Topic | Future Application |
|-------------------|-------------------|
| QMA-completeness | Hardness of quantum simulation |
| Query complexity | Protocol security proofs |
| Oracle separations | Cryptographic assumptions |

---

**Created:** February 9, 2026
**Status:** Not Started
