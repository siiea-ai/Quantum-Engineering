# Week 14: Linear Transformations and Matrices

## Overview
**Days:** 92-98 (7 days)
**Topics:** Linear maps, matrix representations, matrix operations, kernel, range, rank-nullity theorem

This week bridges abstract vector spaces with computational tools. Linear transformations are functions that preserve the vector space structure, and matrices provide their computational representation. In quantum mechanics, these become operators—the mathematical objects representing measurements, gates, and time evolution.

---

## Daily Schedule

| Day | Date | Topic | Hours |
|-----|------|-------|-------|
| 92 | Monday | Definition of Linear Transformations | 7 |
| 93 | Tuesday | Matrix Representation of Linear Maps | 7 |
| 94 | Wednesday | Matrix Operations and Composition | 7 |
| 95 | Thursday | Kernel and Range | 7 |
| 96 | Friday | Rank-Nullity Theorem | 7 |
| 97 | Saturday | Computational Lab | 7 |
| 98 | Sunday | Review and Assessment | 4 |

**Total Study Time:** ~46 hours

---

## Learning Objectives

By the end of this week, you will be able to:

1. Define linear transformations and verify linearity
2. Construct the matrix of a linear transformation
3. Perform matrix operations (addition, multiplication, transpose)
4. Find the kernel (null space) of a linear map
5. Find the range (image) of a linear map
6. Apply and prove the rank-nullity theorem
7. Represent quantum operators as matrices

---

## Key Concepts

### Linear Transformation
A function T: V → W is linear if:
- T(u + v) = T(u) + T(v) for all u, v ∈ V
- T(cv) = cT(v) for all c ∈ F, v ∈ V

Equivalently: T(au + bv) = aT(u) + bT(v)

### Matrix of a Linear Map
If T: V → W with bases B = {v₁,...,vₙ} and B' = {w₁,...,wₘ}:
- Column j of [T]_{B',B} contains coordinates of T(vⱼ) in basis B'
- T(v) ↔ [T]·[v]

### Kernel and Range
- ker(T) = {v ∈ V : T(v) = 0} (null space)
- range(T) = {T(v) : v ∈ V} (image)
- Both are subspaces

### Rank-Nullity Theorem
For T: V → W with V finite-dimensional:
**dim(V) = dim(ker(T)) + dim(range(T))**

Or: **dim(V) = nullity(T) + rank(T)**

---

## Quantum Mechanics Connections

| This Week | Quantum Mechanics |
|-----------|-------------------|
| Linear transformation | Quantum operator Ô |
| Matrix | Operator in matrix form |
| Kernel | States annihilated by operator |
| Range | Accessible output states |
| Composition T∘S | Sequential operations |
| Identity map | Identity operator I |

### Operators in QM
- **Observables:** Hermitian operators (measurement)
- **Evolution:** Unitary operators (time evolution, gates)
- **Matrix elements:** ⟨φ\|Ô\|ψ⟩

### Key Examples
1. **Pauli matrices:** σₓ, σᵧ, σᵤ (spin operators)
2. **Hadamard gate:** H = (1/√2)[1,1; 1,-1]
3. **Projection operators:** P = |ψ⟩⟨ψ|
4. **Lowering/raising operators:** a, a†

---

## Reading Assignments

### Primary: Axler, "Linear Algebra Done Right" (4th ed.)
- Chapter 3.A: The Vector Space of Linear Maps
- Chapter 3.B: Null Spaces and Ranges
- Chapter 3.C: Matrices

### Secondary: Strang, "Introduction to Linear Algebra"
- Chapter 2: Solving Linear Equations
- Chapter 3.3-3.4: Dimension and Rank

### Supplementary
- Shankar, Chapter 1.4-1.6: Linear Operators
- MIT 8.04 Lecture Notes: Operators and Observables

---

## Video Resources

1. **MIT OCW 18.06** (Gilbert Strang)
   - Lecture 3: Multiplication and Inverse Matrices
   - Lecture 6: Column Space and Nullspace
   - Lecture 7: Solving Ax = 0: Pivot Variables, Special Solutions

2. **3Blue1Brown: Essence of Linear Algebra**
   - Chapter 3: Linear transformations and matrices
   - Chapter 4: Matrix multiplication as composition

3. **MIT 18.065** (Matrix Methods)
   - Lecture 1: The Column Space of A Contains All Vectors Ax

---

## Problem Sets

### From Axler:
- Chapter 3.A: Exercises 1-16
- Chapter 3.B: Exercises 1-23
- Chapter 3.C: Exercises 1-15

### From Strang:
- Chapter 3.2: Problems 1-30

### Challenge Problems:
1. Prove that the composition of linear maps is linear
2. Prove dim(V) = dim(ker(T)) + dim(range(T)) from first principles
3. Show that T: ℝ² → ℝ² given by rotation by θ is linear
4. Find all linear maps T: ℝ² → ℝ such that T(1,0) = 2

---

## Week 14 Checklist

- [ ] Day 92: Linear transformation definition and examples
- [ ] Day 93: Matrix representations
- [ ] Day 94: Matrix operations and composition
- [ ] Day 95: Kernel and range
- [ ] Day 96: Rank-nullity theorem
- [ ] Day 97: Computational lab
- [ ] Day 98: Review and self-assessment
- [ ] All Axler Chapter 3 exercises attempted
- [ ] Can construct matrices from linear maps
- [ ] Understand quantum operator connection

---

*"In mathematics, you don't understand things. You just get used to them."*
— John von Neumann
