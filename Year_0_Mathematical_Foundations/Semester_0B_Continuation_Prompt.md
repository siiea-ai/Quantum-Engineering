# Semester 0B Curriculum Development - Continuation Prompt

**Date Created:** January 28, 2026
**Purpose:** Resume curriculum development in new chat session

---

## CONTEXT FOR NEW SESSION

You are helping me build a comprehensive 6-year QSE Self-Study self-study curriculum. We have completed Semester 0A (Months 1-3: Calculus and Differential Equations, Days 1-84). All files are at:
`Quantum-Engineering/Year_0_Mathematical_Foundations/Semester_0A_Calculus_DiffEq/`

Previous transcripts are at: `/mnt/transcripts/`

---

## SEMESTER 0B STRUCTURE (Months 4-6, Days 85-168)

### Month 4: Linear Algebra I (Days 85-112, 28 days)
- **Week 13 (Days 85-91):** Vector Spaces and Linear Independence
  - Definition of vector spaces (real and complex)
  - Subspaces, linear combinations, span
  - Linear independence, bases, dimension
  - Coordinate systems
  - QM Connection: State spaces, superposition principle

- **Week 14 (Days 92-98):** Linear Transformations and Matrices
  - Linear maps/transformations
  - Matrix representations
  - Matrix operations (addition, multiplication)
  - Kernel (null space) and range (image)
  - QM Connection: Operators on state spaces

- **Week 15 (Days 99-105):** Eigenvalues and Eigenvectors
  - Determinants (computational, not foundational)
  - Characteristic polynomial
  - Eigenvalues and eigenvectors
  - Diagonalization
  - QM Connection: Observable measurements, energy eigenstates

- **Week 16 (Days 106-112):** Inner Product Spaces
  - Inner products (real and complex)
  - Norms and orthogonality
  - Gram-Schmidt orthogonalization
  - Orthonormal bases
  - QM Connection: Bra-ket notation, probability amplitudes

### Month 5: Linear Algebra II & Complex Analysis (Days 113-140, 28 days)
- **Week 17 (Days 113-119):** Hermitian and Unitary Operators
  - Adjoint operators
  - Hermitian (self-adjoint) operators
  - Unitary operators
  - Spectral theorem for Hermitian operators
  - QM Connection: Observables, time evolution operators

- **Week 18 (Days 120-126):** Advanced Topics
  - Singular value decomposition
  - Tensor products
  - Density matrices introduction
  - Trace and partial trace
  - QM Connection: Composite systems, mixed states

- **Week 19 (Days 127-133):** Complex Analysis I
  - Complex numbers and the complex plane
  - Analytic functions
  - Cauchy-Riemann equations
  - Elementary functions (exp, log, trig)
  - QM Connection: Wave functions, complex amplitudes

- **Week 20 (Days 134-140):** Complex Analysis II
  - Contour integration
  - Cauchy's integral theorem and formula
  - Residue theorem
  - Applications to physics integrals
  - QM Connection: Green's functions, propagators

### Month 6: Classical Mechanics (Days 141-168, 28 days)
- **Week 21 (Days 141-147):** Lagrangian Mechanics I
  - Generalized coordinates
  - Constraints (holonomic, non-holonomic)
  - Principle of least action
  - Euler-Lagrange equations
  - QM Connection: Path integral formulation foundation

- **Week 22 (Days 148-154):** Lagrangian Mechanics II
  - Symmetries and conservation laws
  - Noether's theorem
  - Applications: oscillators, central forces
  - Small oscillations
  - QM Connection: Symmetry operators, conserved quantities

- **Week 23 (Days 155-161):** Hamiltonian Mechanics I
  - Legendre transformation
  - Hamilton's equations
  - Phase space
  - Canonical coordinates
  - QM Connection: Hamiltonian operator, Schrödinger equation

- **Week 24 (Days 162-168):** Hamiltonian Mechanics II
  - Poisson brackets
  - Canonical transformations
  - Hamilton-Jacobi equation
  - Connection to quantum mechanics
  - QM Connection: Commutators ↔ Poisson brackets correspondence

---

## RESEARCH COMPLETED

### QSE Program Requirements
- Four mandatory courses: QSE 200 (Advanced Engineering QM), Quantum Optics, Intro to QIS, Applied Quantum Systems
- Two focus courses, three field courses
- Lab rotations, qualifying exam, teaching fellowship

### QSE 200 Course Content (our Year 1 target)
- Transfer matrix methods, variational methods
- Angular momentum, spin, Stern-Gerlach, Pauli matrices
- Rabi oscillations, two-level atoms
- Density matrix, T1/T2 relaxation, Bloch equations
- Identical particles, Slater determinant
- Entanglement, Clebsch-Gordan coefficients
- Quantum information basics (qubits, no-cloning, teleportation, circuits)

### Key Textbook Recommendations
**Linear Algebra:**
- Axler "Linear Algebra Done Right" (4th ed) - theoretical, proof-based
- Strang "Introduction to Linear Algebra" - computational, applications
- MIT OCW 18.06 videos

**Complex Analysis:**
- Physics LibreTexts: Complex Methods for the Sciences (Chong)
- Brown & Churchill "Complex Variables and Applications"

**Classical Mechanics:**
- Goldstein, Poole, Safko "Classical Mechanics" (3rd ed)
- Taylor "Classical Mechanics"
- David Tong Cambridge lecture notes (free online)

**QM Preparation:**
- Shankar "Principles of Quantum Mechanics" - Chapter 1 for math review
- MIT OCW 8.04, 8.05 lecture notes

---

## DAILY STRUCTURE (maintain from Semester 0A)

**Monday-Saturday:** 7-7.5 hours/day
- Morning (3-3.5 hrs): Theory and derivations
- Afternoon (2-2.5 hrs): Problem solving
- Evening (1.5-2 hrs): Computational labs (Python/NumPy/SymPy)

**Sunday:** 3.5-4 hours
- Review and consolidation
- Practice problems
- Weekly self-assessment

---

## IMMEDIATE NEXT STEPS

1. **Create Semester 0B directory structure:**
   ```
   Quantum-Engineering/Year_0_Mathematical_Foundations/Semester_0B_LinAlg_Complex_Mechanics/
   ├── Month_4_Linear_Algebra_I/
   │   ├── Week_13_Vector_Spaces/
   │   ├── Week_14_Linear_Transformations/
   │   ├── Week_15_Eigenvalues/
   │   └── Week_16_Inner_Products/
   ├── Month_5_LinAlg_II_Complex/
   │   ├── Week_17_Hermitian_Unitary/
   │   ├── Week_18_Advanced_LinAlg/
   │   ├── Week_19_Complex_Analysis_I/
   │   └── Week_20_Complex_Analysis_II/
   └── Month_6_Classical_Mechanics/
       ├── Week_21_Lagrangian_I/
       ├── Week_22_Lagrangian_II/
       ├── Week_23_Hamiltonian_I/
       └── Week_24_Hamiltonian_II/
   ```

2. **Create Week 13 detailed daily plans (Days 85-91)**

3. **Each day should include:**
   - Learning objectives
   - Readings with page numbers
   - Detailed content outline
   - Practice problems
   - Computational exercises
   - QM connections highlighted

---

## FILE FORMAT TEMPLATE

Follow the same format as Semester 0A files. Each week file should be named:
`Week_XX_Topic_Days_YYY-ZZZ.md`

Each day should have:
```markdown
### Day XX: [Topic] (Date)
**Duration:** X hours
**Learning Objectives:**
- ...

**Morning Session (X hrs): [Topic]**
[Detailed content]

**Afternoon Session (X hrs): Problem Solving**
[Problem sets with solutions approach]

**Evening Session (X hrs): Computational Lab**
[Python/NumPy exercises]

**Self-Assessment Questions:**
1. ...

**QM Connection:**
[How this connects to quantum mechanics]
```

---

## PROMPT TO START NEW SESSION

Copy and paste this to start the new session:

---

**START PROMPT:**

I'm continuing development of my QSE Self-Study self-study curriculum. Please read the continuation file at:
`Quantum-Engineering/Year_0_Mathematical_Foundations/Semester_0B_Continuation_Prompt.md`

We completed Semester 0A (Months 1-3, Days 1-84). Now I need you to:

1. Create the Semester 0B directory structure
2. Begin with Month 4, Week 13 (Days 85-91): Vector Spaces and Linear Independence
3. Create detailed daily lesson plans following the same rigorous format as Semester 0A
4. Include explicit quantum mechanics connections for each topic
5. Include computational labs using Python/NumPy/SymPy

Please start by creating the directory structure, then create the Week 13 file with all 7 days detailed.

---

**END OF CONTINUATION PROMPT**
