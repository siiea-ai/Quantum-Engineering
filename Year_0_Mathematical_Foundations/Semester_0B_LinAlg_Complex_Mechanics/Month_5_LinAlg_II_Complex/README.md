# Month 5: Linear Algebra II & Complex Analysis

## Overview

**Duration:** Days 113-140 (28 days, 4 weeks)
**Status:** ✅ **COMPLETE**
**Focus:** Advanced linear algebra and complex analysis for quantum mechanics

This month advances your linear algebra to quantum-ready level with Hermitian and unitary operators, then introduces complex analysis—the mathematics of wave functions and quantum amplitudes. These topics directly enable the Hilbert space formalism of quantum mechanics.

---

## Weekly Schedule

### Week 17 (Days 113-119): Hermitian & Unitary Operators ✅

- Self-adjoint (Hermitian) operators
- Spectral theorem for Hermitian matrices
- Unitary operators and transformations
- Normal operators
- **QM Connection:** Observables are Hermitian, time evolution is unitary

### Week 18 (Days 120-126): Advanced Linear Algebra ✅

- Singular Value Decomposition (SVD)
- Tensor products of vector spaces
- Trace and partial trace
- Positive operators
- **QM Connection:** Density matrices, composite quantum systems

### Week 19 (Days 127-133): Complex Analysis I ✅

- Complex numbers and the complex plane
- Analytic functions, Cauchy-Riemann equations
- Elementary complex functions (exp, log, trig)
- Conformal mappings
- **QM Connection:** Wave functions are complex-valued

### Week 20 (Days 134-140): Complex Analysis II ✅

- Complex integration, Cauchy's theorem
- Cauchy's integral formula
- Taylor and Laurent series
- Introduction to residues
- **QM Connection:** Propagators, Green's functions

---

## Primary Textbooks

### Linear Algebra

1. **Axler, "Linear Algebra Done Right" (4th edition)**
   - Chapters 7-10 (spectral theory, operators)
   - Rigorous treatment of Hermitian/unitary operators

2. **Strang, "Introduction to Linear Algebra" (5th edition)**
   - Chapters 7-8 (SVD, positive definiteness)
   - Computational perspective

### Complex Analysis

3. **Brown & Churchill, "Complex Variables and Applications" (9th edition)**
   - Chapters 1-6 (foundational complex analysis)
   - Standard physics-oriented treatment

4. **Needham, "Visual Complex Analysis"**
   - Geometric intuition for complex functions
   - Excellent visualizations

---

## Quantum Mechanics Connections

### Advanced Linear Algebra → QM

| Concept | Quantum Application |
|---------|---------------------|
| Hermitian operator | Physical observable (A = A†) |
| Real eigenvalues | Measurement outcomes are real |
| Orthogonal eigenvectors | Distinct outcomes are distinguishable |
| Unitary operator | Time evolution (U†U = I) |
| Spectral decomposition | A = Σλₙ\|n⟩⟨n\| |
| Tensor product | Composite systems H₁ ⊗ H₂ |
| Partial trace | Reduced density matrix |
| Positive operator | Density matrix (ρ ≥ 0, Tr(ρ) = 1) |

### Complex Analysis → QM

| Concept | Quantum Application |
|---------|---------------------|
| Complex amplitude | ψ(x) ∈ ℂ |
| Modulus squared | Probability \|ψ\|² |
| Analytic functions | Regularity of wave functions |
| Contour integration | Propagator calculations |
| Residue theorem | Evaluating physics integrals |
| Branch cuts | Multi-valued quantum phases |

---

## Key Formulas

### Spectral Theorem

$$A = A^\dagger \implies A = \sum_n \lambda_n |n\rangle\langle n|$$

where λₙ are real eigenvalues and |n⟩ are orthonormal eigenvectors.

### Unitary Evolution

$$U^\dagger U = U U^\dagger = I$$

$$|\langle\psi|U^\dagger U|\phi\rangle| = |\langle\psi|\phi\rangle|$$

### Tensor Product

$$(|a\rangle \otimes |b\rangle)(|c\rangle \otimes |d\rangle)^\dagger = \langle a|c\rangle \langle b|d\rangle$$

### Cauchy-Riemann Equations

$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

### Cauchy's Integral Formula

$$f(z_0) = \frac{1}{2\pi i} \oint_C \frac{f(z)}{z - z_0} dz$$

---

## Learning Path

```
Week 17: Hermitian & Unitary Operators
    ↓
Week 18: Tensor Products & Advanced Topics
    ↓
Week 19: Complex Numbers & Analytic Functions
    ↓
Week 20: Contour Integration & Residues
    ↓
Month 6: Classical Mechanics (Hamiltonian Formalism)
```

---

## Computational Focus

Each week includes Python labs covering:

- **Week 17:** Eigendecomposition, unitary matrix verification
- **Week 18:** SVD computation, tensor product construction
- **Week 19:** Complex function visualization, conformal maps
- **Week 20:** Numerical contour integration

### Required Packages

```python
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
```

---

## Assessment Goals

By the end of Month 5, you should be able to:

1. ✅ Prove that Hermitian operators have real eigenvalues
2. ✅ Verify unitary operators preserve inner products
3. ✅ Compute singular value decomposition
4. ✅ Construct tensor products of vector spaces
5. ✅ Determine if a function is analytic
6. ✅ Evaluate contour integrals using Cauchy's theorem
7. ✅ Apply the residue theorem to evaluate real integrals
8. ✅ Connect all concepts to quantum mechanics formalism

---

## Directory Structure

```
Month_5_LinAlg_II_Complex/
├── README.md                    # This file
├── Week_17_Hermitian_Unitary/   # Days 113-119 ✅
├── Week_18_Advanced_LinAlg/     # Days 120-126 ✅
├── Week_19_Complex_Analysis_I/  # Days 127-133 ✅
└── Week_20_Complex_Analysis_II/ # Days 134-140 ✅
```

---

## Completion Checklist

- [x] Week 17: Hermitian & Unitary Operators (Days 113-119)
- [x] Week 18: Advanced Linear Algebra (Days 120-126)
- [x] Week 19: Complex Analysis I (Days 127-133)
- [x] Week 20: Complex Analysis II (Days 134-140)

---

*"The shortest path between two truths in the real domain passes through the complex domain."*
— Jacques Hadamard

---

**Last Updated:** January 29, 2026
**Status:** ✅ COMPLETE (28/28 days)
