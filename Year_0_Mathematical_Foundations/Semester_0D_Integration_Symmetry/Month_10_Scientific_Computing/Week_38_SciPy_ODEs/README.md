# Week 38: SciPy & Numerical Methods

## Overview

This week provides comprehensive coverage of SciPy, the fundamental library for scientific computing in Python. Building on the NumPy foundations from previous weeks, we explore SciPy's powerful numerical algorithms for integration, differential equations, optimization, linear algebra, special functions, and sparse matrices. These tools form the computational backbone for solving quantum mechanical problems numerically.

## Weekly Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| **Day 260** | Numerical Integration | `scipy.integrate.quad`, `dblquad`, `tplquad`, Gaussian quadrature, adaptive integration |
| **Day 261** | ODE Solvers | `scipy.integrate.solve_ivp`, RK45, BDF, stiff equations, event detection |
| **Day 262** | Root Finding & Optimization | `scipy.optimize.root`, `minimize`, `curve_fit`, gradient descent, constrained optimization |
| **Day 263** | Advanced Linear Algebra | `scipy.linalg.eigh`, `expm`, `sqrtm`, `funm`, matrix decompositions |
| **Day 264** | Special Functions & FFT | Hermite, Legendre, spherical harmonics, Bessel functions, FFT/IFFT |
| **Day 265** | Sparse Matrices | `scipy.sparse`, CSR/CSC formats, sparse eigensolvers, large-scale quantum systems |
| **Day 266** | Week Review | Integration project, comprehensive exercises |

## Learning Objectives

By the end of this week, you will be able to:

1. **Numerical Integration**
   - Compute definite integrals using adaptive quadrature methods
   - Evaluate multi-dimensional integrals for expectation values
   - Understand error estimation and convergence criteria

2. **Differential Equations**
   - Solve initial value problems (IVPs) for ODEs
   - Choose appropriate solvers for stiff vs. non-stiff problems
   - Implement the time-dependent Schrödinger equation numerically

3. **Optimization**
   - Find roots of nonlinear equations (eigenvalue conditions)
   - Minimize energy functionals for variational methods
   - Fit quantum mechanical models to experimental data

4. **Matrix Functions**
   - Compute matrix exponentials for time evolution operators
   - Diagonalize Hermitian matrices for eigenvalue problems
   - Apply arbitrary functions to matrices

5. **Special Functions**
   - Evaluate quantum mechanical wavefunctions (Hermite, Legendre, spherical harmonics)
   - Use Bessel functions for cylindrical/spherical problems
   - Apply FFT for momentum-space representations

6. **Sparse Methods**
   - Represent large sparse Hamiltonians efficiently
   - Solve eigenvalue problems for systems with millions of basis states
   - Understand memory and computational complexity trade-offs

## Quantum Mechanics Connections

| SciPy Module | Quantum Application |
|--------------|---------------------|
| `integrate.quad` | Expectation values $$\langle\hat{O}\rangle = \int \psi^* \hat{O} \psi \, dx$$ |
| `integrate.solve_ivp` | Time-dependent Schrödinger equation $$i\hbar\frac{\partial\psi}{\partial t} = \hat{H}\psi$$ |
| `optimize.minimize` | Variational principle $$E[\psi] \geq E_0$$ |
| `linalg.eigh` | Stationary state eigenvalue problems $$\hat{H}\psi_n = E_n\psi_n$$ |
| `linalg.expm` | Time evolution operator $$\hat{U}(t) = e^{-i\hat{H}t/\hbar}$$ |
| `special.hermite` | Harmonic oscillator wavefunctions |
| `special.sph_harm` | Angular momentum eigenfunctions |
| `fft.fft` | Momentum representation $$\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int\psi(x)e^{-ipx/\hbar}dx$$ |
| `sparse.csr_matrix` | Large basis set Hamiltonians |

## Prerequisites

- Week 37: NumPy Essentials
- Linear Algebra (Months 4-5)
- Differential Equations (Month 3)
- Classical Mechanics (Month 6)

## Key Resources

### Documentation
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [SciPy Lecture Notes](https://scipy-lectures.org/)
- [NumPy/SciPy Cookbook](https://scipy-cookbook.readthedocs.io/)

### Textbooks
- Johansson, *Numerical Python: Scientific Computing and Data Science Applications with NumPy, SciPy and Matplotlib*
- Newman, *Computational Physics*
- Thijssen, *Computational Physics*

### Quantum Computing References
- Nielsen & Chuang, *Quantum Computation and Quantum Information*
- Griffiths, *Introduction to Quantum Mechanics*

## Required Software

```python
# Core packages
import numpy as np
import scipy as sp
from scipy import integrate, optimize, linalg, special, fft, sparse
import matplotlib.pyplot as plt

# Version requirements
# numpy >= 1.20
# scipy >= 1.7
# matplotlib >= 3.4
```

## Weekly Project: Quantum Harmonic Oscillator Complete Solver

Throughout this week, you will build a complete numerical solver for the quantum harmonic oscillator:

1. **Monday**: Compute normalization integrals and expectation values
2. **Tuesday**: Solve time-dependent dynamics
3. **Wednesday**: Find energy eigenvalues variationally
4. **Thursday**: Compute matrix exponentials for time evolution
5. **Friday**: Generate wavefunctions using special functions and analyze in momentum space
6. **Saturday**: Scale to large basis sets using sparse matrices
7. **Sunday**: Integrate all components into a complete package

## Assessment Criteria

| Component | Weight | Description |
|-----------|--------|-------------|
| Daily Labs | 40% | Working code for each day's exercises |
| Problem Sets | 30% | Correct solutions with clear explanations |
| Weekly Project | 20% | Complete harmonic oscillator solver |
| Conceptual Understanding | 10% | QM connections and physical interpretation |

## Tips for Success

1. **Run all code examples** - Modify parameters to build intuition
2. **Check dimensions** - Most errors come from shape mismatches
3. **Visualize results** - Plotting reveals computational issues quickly
4. **Compare with analytical** - Validate numerics against known solutions
5. **Start simple** - Build complexity incrementally

## Navigation

- **Previous**: [Week 37: NumPy Essentials](../Week_37_NumPy_Essentials/README.md)
- **Next**: [Week 39: SymPy & Symbolic Computing](../Week_39_SymPy_Symbolic/README.md)
- **Month Overview**: [Month 10: Scientific Computing](../README.md)
