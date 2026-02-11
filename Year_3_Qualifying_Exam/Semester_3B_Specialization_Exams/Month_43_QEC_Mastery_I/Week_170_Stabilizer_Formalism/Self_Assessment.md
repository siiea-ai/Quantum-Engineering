# Week 170: Stabilizer Formalism - Self Assessment

## Overview

Use this self-assessment to evaluate your mastery of the stabilizer formalism before proceeding to Week 171. The stabilizer framework is fundamental to all modern quantum error correction and appears heavily on qualifying examinations.

**Rating Scale:**
- **4 - Mastery:** Can explain to others, derive from first principles, apply to novel problems
- **3 - Proficiency:** Understand well, can solve standard problems, minor gaps
- **2 - Developing:** Basic understanding, struggle with applications, need review
- **1 - Beginning:** Significant gaps, need substantial study

---

## Core Concept Checklist

### Pauli Group

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Single-qubit Pauli operators (I, X, Y, Z) | | |
| Pauli multiplication rules | | |
| n-qubit Pauli group structure | | |
| Commutation vs anticommutation | | |
| Symplectic inner product | | |
| Pauli weight and support | | |

**Minimum competency questions:**
- [ ] Can you compute $$(X_1Z_2Y_3)(Z_1X_2Z_3)$$ quickly?
- [ ] Can you determine if two Paulis commute by inspection?
- [ ] Can you state the size of $$\mathcal{G}_n$$?

---

### Stabilizer States

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Definition of stabilizer state | | |
| Stabilizer group properties | | |
| Generators and independence | | |
| Finding stabilizer from state | | |
| Finding state from stabilizer | | |
| Graph states | | |

**Minimum competency questions:**
- [ ] Can you find the stabilizer of $$|GHZ\rangle = (|000\rangle + |111\rangle)/\sqrt{2}$$?
- [ ] Given generators $$\{X_1X_2, Z_1Z_2\}$$, can you identify the state?
- [ ] Can you prove the stabilizer group must be abelian?

---

### Stabilizer Codes

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Code as +1 eigenspace | | |
| Code parameters [[n, k, d]] | | |
| Logical operators definition | | |
| Finding logical operators | | |
| Syndrome measurement | | |
| Distance calculation | | |

**Minimum competency questions:**
- [ ] For the bit-flip code, can you list stabilizers, logical operators, and syndromes?
- [ ] Can you prove the distance formula $$d = \min\text{wt}(C(\mathcal{S}) \setminus \mathcal{S})$$?
- [ ] Can you design a simple code given constraints?

---

### CSS Codes

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| CSS construction from classical codes | | |
| X-stabilizers and Z-stabilizers | | |
| Commutation proof | | |
| Parameter formula | | |
| CSS codeword structure | | |
| Advantages of CSS | | |

**Minimum competency questions:**
- [ ] Can you construct the Steane code from the Hamming code?
- [ ] Can you prove CSS X and Z stabilizers commute?
- [ ] Can you explain why CSS codes have transversal CNOT?

---

### The Steane Code

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Stabilizer generators | | |
| Logical operators | | |
| Transversal gates | | |
| Syndrome table | | |
| Comparison with Shor code | | |

**Minimum competency questions:**
- [ ] Can you write all 6 Steane code stabilizers from memory?
- [ ] Can you explain which gates are transversal and why?
- [ ] Can you compute the syndrome for a given error?

---

### Gottesman-Knill Theorem

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Clifford group definition | | |
| Clifford gate actions on Paulis | | |
| Stabilizer tableau representation | | |
| Tableau update rules | | |
| Measurement simulation | | |
| Why T gates break simulation | | |

**Minimum competency questions:**
- [ ] Can you state the Gottesman-Knill theorem precisely?
- [ ] Can you track a state through H-CNOT using tableaux?
- [ ] Can you explain why Clifford circuits are classically simulable?

---

## Problem-Solving Skills Assessment

### Calculation Skills

| Skill | Test Problem | Correct? | Time |
|-------|--------------|----------|------|
| Pauli multiplication | Problem 1 | | |
| Find stabilizer of state | Problem 6 | | |
| Find state from stabilizer | Problem 7 | | |
| Syndrome calculation | Problem 12 | | |
| CSS code construction | Problem 18 | | |
| Tableau tracking | Problem 24-25 | | |

**Target:** 80%+ accuracy, 10-15 minutes per problem

---

### Proof Skills

Attempt without notes:

| Proof | Completed? | Gaps Identified |
|-------|------------|-----------------|
| Stabilizer group is abelian | | |
| Stabilizer uniquely determines state | | |
| CSS X,Z stabilizers commute | | |
| Gottesman-Knill (outline) | | |
| Distance = min weight logical | | |

---

## Computational Exercises

### Exercise 1: Pauli Products
Compute these without writing matrices:

1. $$XYZ = $$ ______
2. $$(X \otimes Z)(Z \otimes X) = $$ ______
3. $$(XYZ)^2 = $$ ______

### Exercise 2: Commutation
Do these commute or anticommute?

1. $$X_1X_2X_3$$ and $$Z_1Z_2Z_3$$: ______
2. $$X_1Z_2$$ and $$Y_1Y_2$$: ______
3. $$XZZXI$$ and $$IXZZX$$: ______

### Exercise 3: Stabilizer Identification
What state is stabilized by:

1. $$\langle Z \rangle$$: ______
2. $$\langle X_1X_2, Z_1Z_2 \rangle$$: ______
3. $$\langle X_1X_2X_3, Z_1Z_2, Z_2Z_3 \rangle$$: ______

### Exercise 4: Tableau Update
Start with $$|00\rangle$$ (stabilizer $$Z_1, Z_2$$). Track through:

1. After $$H_1$$: ______
2. After $$CNOT_{12}$$: ______
3. What state? ______

---

## Oral Examination Readiness

### Can You Explain... (3 minutes each)

| Topic | Rating (1-4) | Practice Needed? |
|-------|--------------|------------------|
| What is a stabilizer state? | | |
| How do stabilizer codes work? | | |
| What makes CSS codes special? | | |
| The Gottesman-Knill theorem | | |
| Steane code construction | | |
| Tableau representation | | |

---

### Whiteboard Exercises

Practice these on a whiteboard or paper without notes:

1. **Draw** the syndrome measurement circuit for the bit-flip code
2. **Derive** the Steane code stabilizers from the Hamming code
3. **Prove** that CSS stabilizers commute
4. **Track** a 2-qubit state through a Clifford circuit using tableaux
5. **Find** logical operators for the [[5,1,3]] code

---

## Study Plan Based on Assessment

### If mostly 4s and 3s:
- Focus on speed and elegance
- Practice explaining concepts aloud
- Move to advanced topics (subsystem codes, magic states)

### If mostly 3s and 2s:
- Review Gottesman's thesis (clear exposition)
- Work through more tableau examples
- Practice CSS construction repeatedly

### If mostly 2s and 1s:
- Start with single-qubit Pauli arithmetic
- Build up to multi-qubit systematically
- Use visualization tools (Quirk, IBM Quantum)

---

## Remediation Resources

### For Pauli Group:
- Nielsen & Chuang Appendix on Pauli matrices
- Practice multiplication tables

### For Stabilizer States:
- Gottesman thesis Chapter 2-3
- Interactive tutorials (IBM Quantum Learning)

### For CSS Codes:
- Original Calderbank-Shor-Steane papers
- Preskill notes Chapter 7

### For Gottesman-Knill:
- Aaronson-Gottesman "Improved Simulation" paper
- Implement tableau simulator in Python

---

## Weekly Progress Tracker

| Day | Material Reviewed | Problems Completed | Questions/Gaps |
|-----|-------------------|-------------------|----------------|
| Monday | | | |
| Tuesday | | | |
| Wednesday | | | |
| Thursday | | | |
| Friday | | | |
| Saturday | | | |
| Sunday | | | |

---

## Final Self-Assessment

Before moving to Week 171, verify:

### Essential Knowledge (Must have all)
- [ ] Can multiply Paulis quickly and correctly
- [ ] Can find stabilizer from state and vice versa
- [ ] Understand code parameters [[n, k, d]]
- [ ] Can construct CSS codes from classical codes
- [ ] Can state and explain Gottesman-Knill theorem

### Important Skills (Should have most)
- [ ] Can track states through Clifford circuits
- [ ] Can compute syndromes efficiently
- [ ] Can find logical operators
- [ ] Can prove basic stabilizer theorems

### Advanced Understanding (Aim for some)
- [ ] Understand Clifford hierarchy
- [ ] Can design new stabilizer codes
- [ ] Understand limitations of stabilizer formalism

---

## Reflection Questions

1. What is the most elegant result from this week?

2. Which concept required the most effort to understand?

3. How does the stabilizer formalism simplify error correction?

4. What are the limitations of stabilizer codes?

5. How confident are you explaining this material on an oral exam?

---

## Connection to Other Weeks

### Prerequisites Used (Week 169):
- Knill-Laflamme conditions
- Basic code structure
- Syndrome measurement concept

### Foundation for Future Weeks:
- Week 171: Code families (built using stabilizer formalism)
- Week 172: Topological codes (stabilizer codes on lattices)

---

**Self-Assessment Created:** February 10, 2026
**Next Review:** After completing Week 171
