# Week 172: Topological Codes - Self Assessment

## Overview

This self-assessment evaluates your mastery of topological quantum error correction, the leading approach to fault-tolerant quantum computing. Complete this before your qualifying examination.

**Rating Scale:**
- **4 - Mastery:** Can explain to others, derive from first principles
- **3 - Proficiency:** Understand well, can solve standard problems
- **2 - Developing:** Basic understanding, struggle with applications
- **1 - Beginning:** Significant gaps, need review

---

## Core Concept Checklist

### Toric Code

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Lattice structure and qubit placement | | |
| Vertex operators (A_v) | | |
| Plaquette operators (B_p) | | |
| Commutation relations | | |
| Code parameters [[2L^2, 2, L]] | | |
| Ground state degeneracy | | |
| Logical operators | | |

**Competency check:**
- [ ] Can draw the toric code lattice for L=2?
- [ ] Can prove vertex and plaquette operators commute?
- [ ] Can identify logical operators as non-contractible loops?

---

### Anyonic Excitations

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Electric charges (e) | | |
| Magnetic vortices (m) | | |
| Fusion rules | | |
| Braiding statistics | | |
| Topological order | | |

**Competency check:**
- [ ] Can explain how errors create anyons?
- [ ] Can prove braiding gives -1 phase?
- [ ] Can explain why e×m is a fermion?

---

### Surface Code

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Boundaries (rough vs smooth) | | |
| Code parameters | | |
| Logical operators | | |
| Syndrome measurement | | |
| Standard vs rotated layout | | |

**Competency check:**
- [ ] Can draw a distance-3 surface code?
- [ ] Can identify logical X and Z operators?
- [ ] Can design syndrome measurement circuits?

---

### Decoding

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Error model (X, Z, Y errors) | | |
| Syndrome interpretation | | |
| MWPM algorithm | | |
| Error threshold | | |
| Measurement errors | | |

**Competency check:**
- [ ] Can trace through MWPM for a simple example?
- [ ] Can explain why threshold exists?
- [ ] Can state approximate threshold values?

---

### Logical Operations

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Transversal gate limitations | | |
| Lattice surgery basics | | |
| Merge and split operations | | |
| CNOT via surgery | | |
| Magic state distillation | | |

**Competency check:**
- [ ] Can explain why no transversal T gate?
- [ ] Can describe lattice surgery for ZZ measurement?
- [ ] Can outline magic state distillation protocol?

---

## Quick Reference Test

Fill in from memory:

### Toric Code
- Number of qubits on L×L torus: _______
- Number of logical qubits: _______
- Code distance: _______

### Surface Code (distance d)
- Number of data qubits (rotated): _______
- Number of logical qubits: _______
- Circuit-level threshold: _______

### Anyons
- e × e = _______
- m × m = _______
- e × m = _______
- Braiding phase: _______

---

## Diagram Exercises

### Exercise 1: Toric Code
Draw the L=2 toric code showing:
- [ ] All qubits (edges)
- [ ] All vertices
- [ ] All plaquettes
- [ ] One vertex operator
- [ ] One plaquette operator

### Exercise 2: Surface Code
Draw a distance-3 surface code showing:
- [ ] Data qubits
- [ ] Boundary types
- [ ] Logical Z operator
- [ ] Logical X operator

### Exercise 3: Anyon Creation
Draw the result of:
- [ ] Single X error (m particles)
- [ ] Single Z error (e particles)
- [ ] String of X errors

---

## Calculation Practice

### Problem 1: Code Parameters
For an L×L toric code:
(a) Total qubits: _______
(b) Independent stabilizers: _______
(c) Verify k = n - # stabilizers: _______

### Problem 2: Threshold
If physical error rate is p = 10^-3 and threshold is p_th = 10^-2:
(a) Is error correction beneficial? _______
(b) For d = 11, estimate logical error rate: _______

### Problem 3: Resource Estimate
For 1000 logical qubits at distance 17:
(a) Data qubits per logical: _______
(b) Total data qubits: _______

---

## Oral Exam Readiness

Rate your ability to explain (3 minutes each):

| Topic | Rating (1-4) |
|-------|--------------|
| Toric code construction | |
| Anyonic excitations | |
| Surface code basics | |
| MWPM decoding | |
| Lattice surgery | |
| Magic state distillation | |
| Error threshold concept | |
| Resource estimates | |

---

## Month 43 Integration

### Connections to Previous Weeks

**From Week 169 (Classical to Quantum):**
- [ ] How does Knill-Laflamme apply to toric code?
- [ ] How is syndrome measurement similar?

**From Week 170 (Stabilizer Formalism):**
- [ ] Are topological codes stabilizer codes? How?
- [ ] What is the stabilizer group of the toric code?

**From Week 171 (Code Families):**
- [ ] How does surface code threshold compare to others?
- [ ] What transversal gates does surface code have?

---

## Final Checklist for Month 43

### Essential Knowledge
- [ ] Knill-Laflamme conditions
- [ ] Stabilizer formalism and CSS codes
- [ ] Major code families and trade-offs
- [ ] Toric and surface code structure
- [ ] Error threshold concept
- [ ] Path to universal fault-tolerant QC

### Key Calculations
- [ ] Verify Knill-Laflamme for specific codes
- [ ] Construct stabilizer codes from generators
- [ ] Calculate code parameters
- [ ] Estimate resources for fault-tolerant computing

### Proofs to Know
- [ ] Quantum Singleton bound
- [ ] CSS codes are valid
- [ ] Gottesman-Knill theorem (outline)
- [ ] Threshold existence (outline)

---

## Gap Analysis

### Strong Areas
1. _______
2. _______
3. _______

### Weak Areas
1. _______
2. _______
3. _______

### Study Plan
For each weak area:
- Resource to study: _______
- Practice problems: _______
- Target date: _______

---

## Reflection

1. What is the most important concept from this month?

2. What was most challenging?

3. How does QEC enable fault-tolerant quantum computing?

4. What are the main obstacles to building a fault-tolerant quantum computer?

5. How confident are you for the qualifying exam?

---

## Resources for Further Study

### If struggling with toric code:
- Kitaev's original paper (conceptual)
- Dennis et al. (detailed calculations)

### If struggling with surface code:
- Fowler et al. review article
- Interactive tutorials (Google's Quantum AI)

### If struggling with anyons:
- Nayak et al. review on topological QC
- Pachos textbook

### If struggling with decoding:
- Implement MWPM on small examples
- PyMatching library documentation

---

**Self-Assessment Created:** February 10, 2026
**Month 43 Complete**
