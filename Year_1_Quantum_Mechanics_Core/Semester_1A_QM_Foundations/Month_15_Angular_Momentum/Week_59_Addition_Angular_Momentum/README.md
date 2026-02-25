# Week 59: Addition of Angular Momenta

## Overview

**Days:** 407-413
**Duration:** 7 days (49 hours)
**Position:** Year 1, Semester 1A, Month 15, Week 3
**Theme:** Combining Angular Momentum Systems

This week addresses one of the most important topics in quantum mechanics: how to combine two angular momentum systems into a single composite system. The mathematical framework developed here underlies atomic physics (spin-orbit coupling), nuclear physics (coupling of nucleons), and quantum computing (multi-qubit entanglement).

---

## STATUS: IN PROGRESS

| Day | Date | Topic | Status | Hours |
|-----|------|-------|--------|-------|
| 407 | Monday | Two Angular Momenta | NOT STARTED | 0/7 |
| 408 | Tuesday | Coupled vs Uncoupled Basis | NOT STARTED | 0/7 |
| 409 | Wednesday | Clebsch-Gordan Coefficients | NOT STARTED | 0/7 |
| 410 | Thursday | Spin-Orbit Coupling | NOT STARTED | 0/7 |
| 411 | Friday | Two Spin-1/2 Addition | NOT STARTED | 0/7 |
| 412 | Saturday | Wigner 3j Symbols | NOT STARTED | 0/7 |
| 413 | Sunday | Week Review & Lab | NOT STARTED | 0/7 |
| **Total** | | | **0/49 hours** | |

---

## Learning Objectives

By the end of Week 59, you will be able to:

1. **Construct** the tensor product space for two angular momentum systems
2. **Transform** between uncoupled |j_1,m_1;j_2,m_2> and coupled |j,m;j_1,j_2> bases
3. **Calculate** Clebsch-Gordan coefficients using recursion relations
4. **Apply** the triangle rule and selection rules for angular momentum addition
5. **Derive** singlet and triplet states for two spin-1/2 particles
6. **Explain** spin-orbit coupling and atomic fine structure
7. **Use** Wigner 3j symbols and their symmetry properties
8. **Connect** angular momentum addition to two-qubit entanglement

---

## Daily Schedule

### Day 407 (Monday): Two Angular Momenta

| Block | Time | Duration | Content |
|-------|------|----------|---------|
| Morning | 9:00-12:30 | 3.5 hrs | Total angular momentum operator, tensor products |
| Afternoon | 2:00-4:30 | 2.5 hrs | Problem solving: Tensor product construction |
| Evening | 7:00-8:00 | 1 hr | Lab: Building tensor product spaces |

**Key Topics:**
- Total angular momentum: J = J_1 + J_2
- Tensor product Hilbert space: H = H_1 tensor H_2
- Dimension counting: (2j_1+1)(2j_2+1)
- Proof that J satisfies SU(2) algebra

---

### Day 408 (Tuesday): Coupled vs Uncoupled Basis

| Block | Time | Duration | Content |
|-------|------|----------|---------|
| Morning | 9:00-12:30 | 3.5 hrs | Two complete bases, good quantum numbers |
| Afternoon | 2:00-4:30 | 2.5 hrs | Problem solving: Basis transformations |
| Evening | 7:00-8:00 | 1 hr | Lab: Unitary transformation matrices |

**Key Topics:**
- Uncoupled basis: |j_1,m_1;j_2,m_2> = |j_1,m_1> tensor |j_2,m_2>
- Coupled basis: |j,m;j_1,j_2> eigenstates of J^2, J_z, J_1^2, J_2^2
- Completeness and orthonormality in both bases
- Unitary transformation connecting them

---

### Day 409 (Wednesday): Clebsch-Gordan Coefficients

| Block | Time | Duration | Content |
|-------|------|----------|---------|
| Morning | 9:00-12:30 | 3.5 hrs | Definition, recursion relations, properties |
| Afternoon | 2:00-4:30 | 2.5 hrs | Problem solving: CG calculations |
| Evening | 7:00-8:00 | 1 hr | Lab: CG coefficient calculator with SymPy |

**Key Topics:**
- Definition: <j_1,m_1;j_2,m_2|j,m>
- Triangle inequality: |j_1-j_2| <= j <= j_1+j_2
- Selection rule: m = m_1 + m_2
- Orthogonality and completeness relations
- Phase conventions (Condon-Shortley)

---

### Day 410 (Thursday): Spin-Orbit Coupling

| Block | Time | Duration | Content |
|-------|------|----------|---------|
| Morning | 9:00-12:30 | 3.5 hrs | L + S coupling, fine structure |
| Afternoon | 2:00-4:30 | 2.5 hrs | Problem solving: Hydrogen fine structure |
| Evening | 7:00-8:00 | 1 hr | Lab: Fine structure energy calculation |

**Key Topics:**
- Total angular momentum: J = L + S for an electron
- Allowed values: j = l +/- 1/2 for spin-1/2
- Spin-orbit Hamiltonian: H_SO = A(r) L.S
- Fine structure of hydrogen
- Lande g-factor for Zeeman effect

---

### Day 411 (Friday): Two Spin-1/2 Addition

| Block | Time | Duration | Content |
|-------|------|----------|---------|
| Morning | 9:00-12:30 | 3.5 hrs | Singlet and triplet states, exchange symmetry |
| Afternoon | 2:00-4:30 | 2.5 hrs | Problem solving: Spin states and entanglement |
| Evening | 7:00-8:00 | 1 hr | Lab: Building singlet and triplet in code |

**Key Topics:**
- Addition: 1/2 tensor 1/2 = 0 direct sum 1
- Singlet |0,0> = (|up,down> - |down,up>)/sqrt(2) (antisymmetric)
- Triplet |1,m> states (symmetric)
- Exchange symmetry and the Pauli principle
- Connection to Bell states and entanglement

---

### Day 412 (Saturday): Wigner 3j Symbols

| Block | Time | Duration | Content |
|-------|------|----------|---------|
| Morning | 9:00-12:30 | 3.5 hrs | Definition, symmetry properties |
| Afternoon | 2:00-4:30 | 2.5 hrs | Problem solving: 3j symbol calculations |
| Evening | 7:00-8:00 | 1 hr | Lab: 3j symbol calculator |

**Key Topics:**
- Definition and relation to CG coefficients
- Symmetry under column permutations
- Selection rules and triangular conditions
- Orthogonality relations
- Applications to matrix elements

---

### Day 413 (Sunday): Week Review & Comprehensive Lab

| Block | Time | Duration | Content |
|-------|------|----------|---------|
| Morning | 9:00-12:30 | 3.5 hrs | Week synthesis and applications |
| Afternoon | 2:00-4:30 | 2.5 hrs | Comprehensive problem set |
| Evening | 7:00-8:00 | 1 hr | Full computational lab |

**Key Activities:**
- Connect all concepts from the week
- Physical applications in atoms and nuclei
- Complete CG coefficient calculator
- Prepare for Week 60: Rotations and Wigner D-matrices

---

## Key Formulas

### Total Angular Momentum

$$\hat{\mathbf{J}} = \hat{\mathbf{J}}_1 + \hat{\mathbf{J}}_2$$

$$[\hat{J}_i, \hat{J}_j] = i\hbar\varepsilon_{ijk}\hat{J}_k$$

### Tensor Product Space

$$\mathcal{H} = \mathcal{H}_1 \otimes \mathcal{H}_2, \quad \dim(\mathcal{H}) = (2j_1+1)(2j_2+1)$$

### Clebsch-Gordan Expansion

$$|j,m\rangle = \sum_{m_1,m_2} \langle j_1,m_1;j_2,m_2|j,m\rangle |j_1,m_1;j_2,m_2\rangle$$

### Triangle Rule

$$|j_1 - j_2| \leq j \leq j_1 + j_2$$

### Selection Rule

$$m = m_1 + m_2$$

### CG Orthogonality

$$\sum_{m_1,m_2} \langle j_1,m_1;j_2,m_2|j,m\rangle \langle j_1,m_1;j_2,m_2|j',m'\rangle = \delta_{jj'}\delta_{mm'}$$

### Singlet and Triplet (Two Spin-1/2)

$$|0,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$$

$$|1,1\rangle = |\uparrow\uparrow\rangle, \quad |1,0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle), \quad |1,-1\rangle = |\downarrow\downarrow\rangle$$

### Wigner 3j Symbol

$$\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix} = \frac{(-1)^{j_1-j_2-m_3}}{\sqrt{2j_3+1}} \langle j_1,m_1;j_2,m_2|j_3,-m_3\rangle$$

### Spin-Orbit Coupling

$$\hat{\mathbf{L}} \cdot \hat{\mathbf{S}} = \frac{1}{2}(\hat{J}^2 - \hat{L}^2 - \hat{S}^2)$$

$$\langle \hat{\mathbf{L}} \cdot \hat{\mathbf{S}} \rangle = \frac{\hbar^2}{2}[j(j+1) - l(l+1) - s(s+1)]$$

---

## Primary References

### Textbooks

| Reference | Chapters | Topics |
|-----------|----------|--------|
| **Shankar** | Ch. 15 | Addition of Angular Momenta (Primary) |
| **Sakurai** | Ch. 3.7-3.8 | Angular Momentum Addition |
| **Griffiths** | Ch. 4.4 | Addition of Angular Momenta |
| **Cohen-Tannoudji** | Ch. X | Addition of Angular Momenta |

### Supplementary Resources

- MIT OCW 8.05 Lectures 20-22 (Zwiebach)
- Feynman Lectures Vol. III, Ch. 18
- Edmonds "Angular Momentum in Quantum Mechanics"
- Varshalovich "Quantum Theory of Angular Momentum"

---

## Quantum Computing Connections

| Angular Momentum Concept | Quantum Computing Application |
|-------------------------|------------------------------|
| Two spin-1/2 addition | Two-qubit Hilbert space |
| Tensor product space | Multi-qubit state space |
| Singlet state | Bell state \|psi^-\> |
| Triplet states | Bell states \|phi^+\>, \|psi^+\> |
| Clebsch-Gordan coefficients | Entanglement characterization |
| Exchange symmetry | Fermionic qubit encoding |

---

## Prerequisites from Weeks 57-58

| Previous Topic | Week 59 Application |
|----------------|---------------------|
| Angular momentum algebra | Foundation for J = J_1 + J_2 |
| Ladder operators | CG recursion relations |
| Spin-1/2 formalism | Two-spin addition |
| Pauli matrices | Singlet/triplet construction |
| Bloch sphere | Two-qubit visualization |

---

## Computational Tools

```python
# Key Python imports for Week 59
import numpy as np
from scipy.special import comb
import sympy as sp
from sympy.physics.quantum.spin import CG, Rotation
from sympy.physics.quantum.cg import cg_simp
import matplotlib.pyplot as plt
```

### SymPy CG Coefficient Example

```python
from sympy.physics.quantum.cg import CG
from sympy import S, sqrt

# Calculate <1/2, 1/2; 1/2, -1/2 | 1, 0>
j1, m1 = S(1)/2, S(1)/2
j2, m2 = S(1)/2, -S(1)/2
j, m = 1, 0

cg = CG(j1, m1, j2, m2, j, m).doit()
print(f"CG coefficient = {cg}")  # Output: sqrt(2)/2
```

---

## Assessment Checkpoints

### Day 407-408 Checkpoint

- [ ] Write tensor product of two states as |j_1,m_1> tensor |j_2,m_2>
- [ ] Calculate dimension of composite Hilbert space
- [ ] Identify good quantum numbers in each basis

### Day 409-410 Checkpoint

- [ ] Use triangle rule to find allowed j values
- [ ] Calculate at least 5 CG coefficients by hand
- [ ] Apply spin-orbit formula to hydrogen 2p state

### Day 411-413 Checkpoint

- [ ] Derive singlet and triplet states from scratch
- [ ] Verify orthonormality of all four two-spin states
- [ ] Implement CG coefficient calculator in Python

---

## Week 59 Files

```
Week_59_Addition_Angular_Momentum/
|-- README.md                    # This file
|-- Day_407_Monday.md           # Two Angular Momenta
|-- Day_408_Tuesday.md          # Coupled vs Uncoupled Basis
|-- Day_409_Wednesday.md        # Clebsch-Gordan Coefficients
|-- Day_410_Thursday.md         # Spin-Orbit Coupling
|-- Day_411_Friday.md           # Two Spin-1/2 Addition
|-- Day_412_Saturday.md         # Wigner 3j Symbols
|-- Day_413_Sunday.md           # Week Review & Lab
```

---

## Preview: Week 60

Next week covers **Rotations and Wigner D-Matrices**, where we:

- Study the rotation operator R(n,theta) = exp(-i theta n.J / hbar)
- Master Euler angle parameterization
- Derive Wigner D-matrices D^j_{m'm}(alpha, beta, gamma)
- Prove the Wigner-Eckart theorem
- Apply to selection rules in atomic transitions

---

*"The addition of angular momenta is where the abstract beauty of group theory meets the concrete reality of atomic physics."*
-- John Archibald Wheeler

---

**Created:** February 2, 2026
**Status:** In Progress
**Week:** 59 of 312
