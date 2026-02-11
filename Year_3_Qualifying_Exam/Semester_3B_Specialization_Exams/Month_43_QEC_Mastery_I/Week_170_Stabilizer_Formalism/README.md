# Week 170: Stabilizer Formalism

## Overview

**Days:** 1184-1190
**Theme:** The Stabilizer Framework for Quantum Error Correction

The stabilizer formalism, developed by Daniel Gottesman, provides a powerful algebraic framework for describing and analyzing quantum error-correcting codes. This week develops the theory systematically, culminating in the Gottesman-Knill theorem on classical simulability and the construction of CSS codes.

---

## Daily Schedule

### Day 1184 (Monday): The Pauli Group

**Topics:**
- Single-qubit Pauli group $$\mathcal{G}_1 = \{\pm 1, \pm i\} \times \{I, X, Y, Z\}$$
- n-qubit Pauli group $$\mathcal{G}_n$$
- Group structure: multiplication, commutation, center
- Pauli weight and support

**Key Concepts:**
- $$|\mathcal{G}_n| = 4 \cdot 4^n = 4^{n+1}$$
- Commutation: $$[P, Q] = 0$$ or $$\{P, Q\} = 0$$ for Paulis
- Center: $$Z(\mathcal{G}_n) = \{\pm I, \pm iI\}$$

### Day 1185 (Tuesday): Stabilizer States

**Topics:**
- Definition: States $$|\psi\rangle$$ with $$S|\psi\rangle = |\psi\rangle$$ for $$S \in \mathcal{S}$$
- Stabilizer group structure
- Unique state from maximal stabilizer
- Examples: computational basis, Bell states, GHZ states

**Key Result:**
$$|\psi\rangle = \frac{1}{|\mathcal{S}|}\sum_{S \in \mathcal{S}} S |\phi\rangle$$
for any $$|\phi\rangle$$ with $$\langle\phi|\psi\rangle \neq 0$$

### Day 1186 (Wednesday): Stabilizer Codes

**Topics:**
- Code space as simultaneous +1 eigenspace
- Stabilizer generators (independent, commuting)
- $$[[n, k, d]]$$ codes: $$n$$ qubits, $$k = n - r$$ logical qubits, $$r$$ generators
- Logical operators: centralizer modulo stabilizer

**Key Insight:**
$$\text{Code space} = \{|\psi\rangle : S|\psi\rangle = |\psi\rangle \text{ for all } S \in \mathcal{S}\}$$

### Day 1187 (Thursday): CSS Codes

**Topics:**
- Calderbank-Shor-Steane construction
- Classical codes $$C_1 \supset C_2$$ with $$C_2^\perp \subset C_1$$
- X-stabilizers from $$C_2^\perp$$, Z-stabilizers from $$C_1^\perp$$
- CSS code parameters from classical code parameters

**Construction:**
$$|x + C_2\rangle = \frac{1}{\sqrt{|C_2|}}\sum_{y \in C_2} |x + y\rangle$$

### Day 1188 (Friday): The Steane Code

**Topics:**
- [[7, 1, 3]] Steane code from Hamming code
- Explicit stabilizer generators
- Transversal gates
- Comparison with Shor code

**Stabilizers:**
$$\begin{aligned}
&IIIXXXX, \quad IXXIIXX, \quad XIXIXIX \\
&IIIZZZZ, \quad IZZIIZZ, \quad ZIZIZIZ
\end{aligned}$$

### Day 1189 (Saturday): Gottesman-Knill Theorem

**Topics:**
- Clifford group definition
- Clifford gates: H, S, CNOT
- Tableau representation of stabilizer states
- Classical simulation of Clifford circuits

**Theorem:**
Clifford circuits with stabilizer input states and Pauli measurements can be efficiently simulated classically.

### Day 1190 (Sunday): Review and Integration

**Activities:**
- Stabilizer tableau calculations
- Code construction practice
- Oral exam preparation
- Preview of Week 171

---

## Learning Objectives

By the end of this week, students will be able to:

1. **Compute** products in the Pauli group and determine commutation relations
2. **Identify** stabilizer states and their stabilizer groups
3. **Construct** stabilizer codes from generator sets
4. **Build** CSS codes from classical linear codes
5. **Analyze** the Steane code and compute its properties
6. **Apply** the Gottesman-Knill theorem to determine simulability
7. **Perform** stabilizer tableau updates under Clifford gates

---

## Key Formulas

| Concept | Formula |
|---------|---------|
| Pauli group size | $$\|\mathcal{G}_n\| = 4^{n+1}$$ |
| Code dimension | $$k = n - r$$ (r independent generators) |
| CSS X-stabilizers | From rows of $$H_2$$ (parity check of $$C_2$$) |
| CSS Z-stabilizers | From rows of $$H_1$$ (parity check of $$C_1$$) |
| Clifford condition | $$CPC^\dagger \in \mathcal{G}_n$$ for all $$P \in \mathcal{G}_n$$ |

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

1. Gottesman, D. "Stabilizer Codes and Quantum Error Correction" PhD Thesis (1997)
2. Calderbank, A.R. & Shor, P.W. "Good Quantum Error-Correcting Codes Exist" (1996)
3. Steane, A.M. "Error Correcting Codes in Quantum Theory" (1996)
4. Nielsen & Chuang, Chapter 10.5
5. Aaronson, S. & Gottesman, D. "Improved Simulation of Stabilizer Circuits" (2004)

---

**Week 170 Created:** February 10, 2026
