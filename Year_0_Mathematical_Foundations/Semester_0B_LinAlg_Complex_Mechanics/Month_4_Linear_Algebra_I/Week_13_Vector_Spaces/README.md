# Week 13: Vector Spaces and Linear Independence

## Overview
**Days:** 85-91 (7 days)
**Topics:** Definition of vector spaces, subspaces, span, linear independence, bases, dimension

This week introduces the fundamental structures of linear algebra. Vector spaces are the mathematical setting for quantum mechanics—quantum states live in vector spaces (specifically, Hilbert spaces), and understanding their abstract properties is essential.

---

## Daily Schedule

| Day | Date | Topic | Hours |
|-----|------|-------|-------|
| 85 | Monday | Definition of Vector Spaces | 7 |
| 86 | Tuesday | Subspaces and Linear Combinations | 7 |
| 87 | Wednesday | Span and Spanning Sets | 7 |
| 88 | Thursday | Linear Independence | 7 |
| 89 | Friday | Bases and Dimension | 7 |
| 90 | Saturday | Computational Lab | 7 |
| 91 | Sunday | Review and Assessment | 3.5 |

**Total Study Time:** ~45.5 hours

---

## Learning Objectives

By the end of this week, you will be able to:

1. Verify the vector space axioms for a given set
2. Identify and work with subspaces
3. Determine if a set of vectors spans a space
4. Test for linear independence
5. Find bases for vector spaces
6. Calculate dimensions of vector spaces
7. Connect these concepts to quantum state spaces

---

## Key Concepts

### Vector Space Axioms
A vector space V over a field F satisfies:
- Closure under addition and scalar multiplication
- Associativity and commutativity of addition
- Existence of zero vector and additive inverses
- Distributive properties of scalar multiplication
- Identity property of scalar multiplication

### Subspaces
A subset W ⊆ V is a subspace if:
- 0 ∈ W
- W is closed under addition
- W is closed under scalar multiplication

### Linear Independence
Vectors v₁, ..., vₙ are linearly independent if:
c₁v₁ + ... + cₙvₙ = 0 implies c₁ = ... = cₙ = 0

### Basis and Dimension
- A basis is a linearly independent spanning set
- Dimension = number of vectors in any basis

---

## Quantum Mechanics Connections

| This Week | Quantum Mechanics |
|-----------|-------------------|
| Vector space | Hilbert space of quantum states |
| Vector | Quantum state \|ψ⟩ |
| Complex scalars | Probability amplitudes |
| Linear combination | Superposition of states |
| Basis | Complete set of orthonormal states |
| Dimension | Number of distinguishable states |

### The Superposition Principle
If \|ψ₁⟩ and \|ψ₂⟩ are valid quantum states, then
α\|ψ₁⟩ + β\|ψ₂⟩ is also a valid quantum state.

This is precisely the statement that quantum states form a vector space!

---

## Reading Assignments

### Primary: Axler, "Linear Algebra Done Right" (4th ed.)
- Chapter 1: Vector Spaces (entire chapter)
- Focus on: Definitions, examples, subspaces

### Secondary: Strang, "Introduction to Linear Algebra"
- Chapter 1: Introduction to Vectors
- Chapter 3: Vector Spaces and Subspaces

### Supplementary: Shankar, "Principles of Quantum Mechanics"
- Chapter 1.1-1.3: Mathematical Introduction
- Note how Shankar motivates linear algebra for QM

---

## Video Resources

1. **MIT OCW 18.06** (Gilbert Strang)
   - Lecture 1: The Geometry of Linear Equations
   - Lecture 5: Transposes, Permutations, Spaces Rⁿ

2. **3Blue1Brown: Essence of Linear Algebra**
   - Chapter 1: Vectors, what even are they?
   - Chapter 2: Linear combinations, span, and basis vectors

3. **MIT OCW 8.04** (Quantum Physics)
   - Lecture 1: Introduction to Superposition (preview)

---

## Problem Sets

### From Axler:
- Chapter 1: Exercises 1-23 (select based on daily assignments)

### From Strang:
- Chapter 3.1: Problems 1-20

### Challenge Problems:
- Prove that the intersection of two subspaces is a subspace
- Prove that the union of two subspaces is generally NOT a subspace
- Show that polynomials of degree ≤ n form a vector space of dimension n+1

---

## Computational Lab Preview

Saturday's lab will include:
- Implementing vector space operations in NumPy
- Testing linear independence numerically
- Visualizing spans in ℝ² and ℝ³
- Introduction to complex vector spaces
- Simulating quantum superposition

---

## Self-Assessment Questions

At the end of the week, you should be able to answer:

1. What are the 8 axioms defining a vector space?
2. How do you verify something is a subspace?
3. What's the geometric meaning of span?
4. Why is linear independence important?
5. Why do all bases have the same size?
6. What's the dimension of the space of 2×2 matrices?
7. How does this connect to quantum mechanics?

---

## Week 13 Checklist

- [ ] Day 85: Vector space definition and examples
- [ ] Day 86: Subspaces and linear combinations
- [ ] Day 87: Span and spanning sets
- [ ] Day 88: Linear independence
- [ ] Day 89: Bases and dimension
- [ ] Day 90: Computational lab
- [ ] Day 91: Review and self-assessment
- [ ] All Axler Chapter 1 exercises attempted
- [ ] Flashcards created for key definitions
- [ ] QM connections understood

---

*"The introduction of the cipher 0 or the group concept was general nonsense too, and mathematics was more or less stagnating for thousands of years because nobody was around to take such childish steps."*
— Alexander Grothendieck, on abstract approaches to mathematics
