# Month 11: Group Theory & Symmetries (Days 281-308)

## Overview

Month 11 introduces **group theory**, the mathematical language of symmetry that underlies all of modern physics. This is arguably the most important mathematical structure for quantum mechanics—every conservation law corresponds to a symmetry (Noether's theorem), and the classification of particles is fundamentally a group-theoretic problem.

**Why This Month is CRITICAL for Quantum Mechanics:**
- Angular momentum operators generate rotations via Lie groups
- Spin arises from representations of SU(2)
- Selection rules follow from symmetry constraints
- Particle classification uses representation theory
- Conservation laws connect to continuous symmetries

## Learning Objectives

By the end of Month 11, you will be able to:

1. **Define and analyze abstract groups** - Recognize group structures, classify groups, understand subgroups and quotient groups
2. **Work with representations** - Understand how groups act on vector spaces, apply Schur's lemma, decompose representations
3. **Navigate Lie groups and algebras** - Work with SO(3), SU(2), and their relationship, understand exponential map
4. **Master angular momentum theory** - Couple angular momenta using Clebsch-Gordan coefficients, understand selection rules
5. **Apply symmetry to physics** - Use group theory to simplify quantum mechanical problems

## Weekly Structure

### Week 41: Abstract Group Theory (Days 281-287)
| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| 281 | Monday | Introduction to Groups | Group axioms, examples, Cayley tables |
| 282 | Tuesday | Subgroups and Cosets | Lagrange's theorem, normal subgroups |
| 283 | Wednesday | Group Homomorphisms | Isomorphism theorems, kernels, images |
| 284 | Thursday | Quotient Groups | Factor groups, First Isomorphism Theorem |
| 285 | Friday | Cyclic and Abelian Groups | Generators, classification theorems |
| 286 | Saturday | Permutation Groups | Symmetric group, cycle notation, alternating group |
| 287 | Sunday | Week 41 Review | Synthesis, problem session, applications |

### Week 42: Representation Theory (Days 288-294)
| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| 288 | Monday | Group Representations | Definition, examples, matrix representations |
| 289 | Tuesday | Reducible and Irreducible | Invariant subspaces, complete reducibility |
| 290 | Wednesday | Schur's Lemma | Statement, proof, consequences |
| 291 | Thursday | Characters | Character tables, orthogonality relations |
| 292 | Friday | Tensor Products | Product representations, Clebsch-Gordan |
| 293 | Saturday | Representations of S_n | Young tableaux, partitions |
| 294 | Sunday | Week 42 Review | Physical applications, synthesis |

### Week 43: Lie Groups and Lie Algebras (Days 295-301)
| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| 295 | Monday | Introduction to Lie Groups | Continuous groups, matrix Lie groups |
| 296 | Tuesday | The Rotation Group SO(3) | Parameterization, Euler angles |
| 297 | Wednesday | Lie Algebras | Tangent space, commutators, structure constants |
| 298 | Thursday | The Lie Algebra so(3) | Angular momentum generators, commutation relations |
| 299 | Friday | SU(2) and Its Relationship to SO(3) | Double cover, spinors, topology |
| 300 | Saturday | Representations of SU(2) | Spin representations, ladder operators |
| 301 | Sunday | Week 43 Review | Connections to quantum mechanics |

### Week 44: Angular Momentum & Clebsch-Gordan (Days 302-308)
| Day | Date | Topic | Key Concepts |
|-----|------|-------|--------------|
| 302 | Monday | Angular Momentum in Quantum Mechanics | Operators, eigenvalues, spherical harmonics |
| 303 | Tuesday | Spin Angular Momentum | Pauli matrices, spin-1/2, spinors |
| 304 | Wednesday | Addition of Angular Momenta | Coupled vs uncoupled bases, total J |
| 305 | Thursday | Clebsch-Gordan Coefficients | Definition, computation, symmetries |
| 306 | Friday | Wigner-Eckart Theorem | Selection rules, reduced matrix elements |
| 307 | Saturday | Applications to Atomic Physics | LS coupling, term symbols, spectra |
| 308 | Sunday | Month 11 Capstone | Comprehensive review, integration project |

## Primary Resources

### Textbooks
1. **Tinkham, M.** - *Group Theory and Quantum Mechanics* (Primary for physics applications)
2. **Hall, Brian** - *Lie Groups, Lie Algebras, and Representations* (Mathematical rigor)
3. **Sakurai, J.J.** - *Modern Quantum Mechanics*, Chapter 3-4 (Angular momentum)
4. **Artin, Michael** - *Algebra* (Abstract algebra background)

### Online Resources
- MIT OCW 18.701 (Algebra I)
- Physics LibreTexts: Group Theory
- Brian Hall's lecture notes (University of Notre Dame)

## Mathematical Prerequisites

From previous months:
- Linear algebra (eigenvalues, diagonalization, inner products)
- Complex analysis (complex exponentials, matrices)
- Differential equations (for Lie algebra exponentials)
- Calculus on manifolds (for Lie group structure)

## Quantum Mechanics Connections

| Group Theory Concept | Quantum Mechanics Application |
|---------------------|------------------------------|
| Group axioms | Composition of transformations |
| Representations | How states transform |
| Irreducible representations | Particle types, multiplets |
| SO(3) | Spatial rotations |
| SU(2) | Spin rotations, spinors |
| Lie algebra | Infinitesimal generators = observables |
| Clebsch-Gordan | Angular momentum coupling |
| Characters | Selection rules |
| Schur's lemma | Degeneracy in quantum systems |

## Key Formulas Preview

### Group Axioms
$$G \text{ is a group if: } \forall a,b,c \in G$$
$$\text{Closure: } a \cdot b \in G$$
$$\text{Associativity: } (a \cdot b) \cdot c = a \cdot (b \cdot c)$$
$$\text{Identity: } \exists e : e \cdot a = a \cdot e = a$$
$$\text{Inverse: } \exists a^{-1} : a \cdot a^{-1} = a^{-1} \cdot a = e$$

### Angular Momentum Commutation Relations
$$[J_i, J_j] = i\hbar \epsilon_{ijk} J_k$$

### SU(2) Generators (Pauli Matrices)
$$\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### Clebsch-Gordan Decomposition
$$|j_1, m_1\rangle \otimes |j_2, m_2\rangle = \sum_{j=|j_1-j_2|}^{j_1+j_2} \sum_{m=-j}^{j} C_{j_1 m_1; j_2 m_2}^{j m} |j, m\rangle$$

## Progress Tracking

| Week | Topic | Days | Status |
|------|-------|------|--------|
| 41 | Abstract Group Theory | 281-287 | 🔄 In Progress |
| 42 | Representation Theory | 288-294 | ⬜ Not Started |
| 43 | Lie Groups and Algebras | 295-301 | ⬜ Not Started |
| 44 | Angular Momentum | 302-308 | ⬜ Not Started |

**Month Progress:** 0/28 days (0%)

## Assessment Milestones

1. **Week 41 Check:** Prove Lagrange's theorem, compute quotient groups
2. **Week 42 Check:** Build character table for S_3, verify orthogonality
3. **Week 43 Check:** Derive so(3) commutation relations from SO(3)
4. **Week 44 Check:** Compute Clebsch-Gordan coefficients for j_1 = j_2 = 1/2

## Connection to Year 1

Month 11 directly prepares you for:
- **Year 1, Month 1-3:** Quantum mechanics formalism (states as representations)
- **Year 1, Month 9-10:** Angular momentum and spin
- **Year 2:** Quantum field theory (Lorentz group representations)
- **Research:** Particle physics, condensed matter, quantum computing

---

*"The universe is an enormous direct product of representations of symmetry groups." — Steven Weinberg*
