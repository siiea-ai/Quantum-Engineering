# Month 10: Scientific Computing

## Overview

**Duration:** Days 253-280 (28 days)
**Status:** 🔄 IN PROGRESS
**Focus:** Computational tools for quantum physics

This month develops the programming and numerical skills essential for modern physics research. We master Python, NumPy, SciPy, and visualization tools, applying them to physics simulations that preview quantum mechanics.

---

## Weekly Structure

| Week | Days | Topic | Focus |
|------|------|-------|-------|
| 37 | 253-259 | Python & NumPy | Foundations, arrays, linear algebra |
| 38 | 260-266 | SciPy & ODEs | Numerical methods, differential equations |
| 39 | 267-273 | Visualization | Matplotlib, animations, publication figures |
| 40 | 274-280 | Physics Simulations | Classical to quantum transition |

---

## Learning Objectives

By the end of Month 10, you will be able to:

1. **Python Mastery:** Write clean, efficient Python code following best practices
2. **NumPy Arrays:** Leverage vectorized operations for numerical computing
3. **Linear Algebra:** Solve eigenvalue problems and matrix equations numerically
4. **ODE Solvers:** Apply Runge-Kutta and other methods to physics problems
5. **Optimization:** Use SciPy's optimization and root-finding algorithms
6. **Visualization:** Create publication-quality figures and animations
7. **Simulations:** Implement classical and quantum physics simulations
8. **Reproducibility:** Write documented, reproducible computational workflows

---

## Key Topics by Week

### Week 37: Python & NumPy Foundations
- Python refresher: functions, classes, modules
- NumPy arrays: creation, indexing, broadcasting
- Vectorization and avoiding loops
- Linear algebra with numpy.linalg
- Random number generation
- File I/O and data management
- Performance optimization basics

### Week 38: SciPy & Numerical Methods
- scipy.integrate: quadrature and ODE solvers
- scipy.optimize: root finding, minimization
- scipy.linalg: advanced linear algebra
- scipy.special: special functions (Bessel, Legendre, etc.)
- scipy.fft: Fourier transforms
- Numerical stability and error analysis
- scipy.sparse: sparse matrices

### Week 39: Visualization & Jupyter
- Matplotlib fundamentals
- 2D and 3D plotting
- Contour plots and colormaps
- Animations with FuncAnimation
- Interactive widgets in Jupyter
- Publication-quality figures
- SymPy for symbolic computation

### Week 40: Physics Simulations
- Classical mechanics simulations
- Electromagnetism field visualization
- Wave equation and diffusion
- Schrödinger equation (1D, time-dependent)
- Monte Carlo methods
- Eigenvalue problems in physics
- Capstone project: quantum harmonic oscillator

---

## Essential Tools

### Core Libraries
```python
import numpy as np
import scipy as sp
from scipy import linalg, integrate, optimize, special
import matplotlib.pyplot as plt
from matplotlib import animation
import sympy as sym
```

### Key Functions

| Task | NumPy/SciPy Function |
|------|---------------------|
| Eigenvalues | `np.linalg.eig`, `sp.linalg.eigh` |
| Matrix solve | `np.linalg.solve`, `sp.linalg.solve` |
| ODE solve | `sp.integrate.solve_ivp` |
| Integration | `sp.integrate.quad` |
| Optimization | `sp.optimize.minimize` |
| Root finding | `sp.optimize.root` |
| FFT | `sp.fft.fft`, `sp.fft.ifft` |
| Special functions | `sp.special.hermite`, `sp.special.sph_harm` |

---

## Quantum Mechanics Connections

| Computational Skill | QM Application |
|--------------------|----------------|
| Eigenvalue solvers | Energy levels, stationary states |
| Matrix exponentiation | Time evolution $e^{-iHt/\hbar}$ |
| ODE solvers | Time-dependent Schrödinger equation |
| FFT | Momentum space, spectral methods |
| Quadrature | Normalization, expectation values |
| Sparse matrices | Large Hilbert spaces |
| Monte Carlo | Quantum Monte Carlo, path integrals |
| Optimization | Variational methods |

---

## Textbook References

**Primary:**
- VanderPlas, *Python Data Science Handbook*
- Newman, *Computational Physics with Python*

**Supplementary:**
- Langtangen, *A Primer on Scientific Programming with Python*
- NumPy/SciPy documentation

---

## Directory Structure

```
Month_10_Scientific_Computing/
├── README.md                          # This file
├── Week_37_Python_NumPy/              # Days 253-259
│   ├── README.md
│   └── Day_253-259_*.md
├── Week_38_SciPy_ODEs/                # Days 260-266
│   ├── README.md
│   └── Day_260-266_*.md
├── Week_39_Visualization/             # Days 267-273
│   ├── README.md
│   └── Day_267-273_*.md
└── Week_40_Physics_Simulations/       # Days 274-280
    ├── README.md
    └── Day_274-280_*.md
```

---

## Prerequisites

**Required from earlier months:**
- All previous mathematical content (calculus, linear algebra, etc.)
- Basic programming experience
- Classical mechanics (Month 6)
- Electromagnetism (Month 8)

---

## Completion Checklist

- [ ] Week 37: Python & NumPy (0/7 days)
- [ ] Week 38: SciPy & ODEs (0/7 days)
- [ ] Week 39: Visualization (0/7 days)
- [ ] Week 40: Physics Simulations (0/7 days)

**Month 10 Status: IN PROGRESS (0/28 days)**

---

## Key Insights

1. **Vectorize everything** — NumPy's power comes from avoiding Python loops.

2. **Use the right solver** — Different ODE methods for different problems.

3. **Visualization communicates** — A good plot is worth a thousand equations.

4. **Numerical ≠ approximate** — Modern numerics can achieve machine precision.

5. **Code is documentation** — Write readable, reproducible science.

---

*"The purpose of computing is insight, not numbers."*
— Richard Hamming

---

**Previous:** Month 9 — Functional Analysis
**Next:** Month 11 — Group Theory & Symmetries
