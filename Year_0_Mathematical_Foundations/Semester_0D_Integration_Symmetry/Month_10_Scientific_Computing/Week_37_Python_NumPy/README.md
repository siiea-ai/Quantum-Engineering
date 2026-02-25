# Week 37: Python & NumPy Foundations

## Overview

Week 37 establishes the computational foundation essential for all subsequent scientific computing in this curriculum. We systematically develop proficiency in Python programming and NumPy array operations, emphasizing the patterns and techniques that dominate quantum physics simulations. By week's end, you will command the tools needed to discretize wave functions, solve eigenvalue problems numerically, and implement Monte Carlo methods.

**Week Duration:** Days 253-259 (7 days)
**Total Hours:** ~56 hours
**Prerequisites:** Basic programming experience, linear algebra (Month 4-5), calculus (Months 1-3)

---

## Learning Objectives for the Week

By the end of Week 37, you will be able to:

1. **Write Pythonic code** using functions, classes, decorators, and generators for scientific applications
2. **Create and manipulate NumPy arrays** with efficient indexing, slicing, and fancy indexing
3. **Apply broadcasting rules** to eliminate explicit loops and achieve vectorized computation
4. **Solve linear algebra problems** using `np.linalg` for eigenvalues, matrix decompositions, and linear systems
5. **Generate random numbers** and compute statistics for Monte Carlo simulations
6. **Optimize I/O and performance** through binary file formats and profiling techniques

---

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| **253 (Mon)** | Python Refresher: Functions and Classes | Functions, lambdas, decorators, OOP, generators |
| **254 (Tue)** | NumPy Arrays: Creation and Indexing | `np.array`, `zeros`, `linspace`, slicing, fancy indexing |
| **255 (Wed)** | Vectorization and Broadcasting | Universal functions, broadcasting rules, avoiding loops |
| **256 (Thu)** | Linear Algebra with NumPy | `np.linalg.eig`, `solve`, `svd`, `inv`, `det` |
| **257 (Fri)** | Random Numbers and Statistics | `np.random`, distributions, Monte Carlo basics |
| **258 (Sat)** | File I/O and Performance | `np.save`, `np.load`, pandas intro, profiling |
| **259 (Sun)** | Week Review | Integration project, comprehensive exercises |

---

## Quantum Mechanics Connections

This week's material directly supports quantum physics computation:

### Wave Function Discretization
NumPy arrays naturally represent discretized wave functions $\psi(x_i)$ on spatial grids. The operations we learn—element-wise multiplication, broadcasting, reduction—map directly to computing observables like $\langle x \rangle$ and $\langle p \rangle$.

### Eigenvalue Problems
The Schrödinger equation $\hat{H}\psi = E\psi$ becomes a matrix eigenvalue problem when discretized. `np.linalg.eig` and `np.linalg.eigh` are the workhorses for finding energy levels and stationary states.

### Monte Carlo Methods
Quantum Monte Carlo (QMC) methods rely on random sampling to estimate integrals and ground state properties. The `np.random` module provides the foundation for variational Monte Carlo and path integral methods.

### Performance Matters
Quantum simulations scale exponentially with system size. The optimization techniques in this week—vectorization, efficient memory access, compiled code—are essential for practical calculations.

---

## Required Software

```bash
# Create virtual environment (recommended)
python -m venv quantum_env
source quantum_env/bin/activate  # Linux/Mac
# quantum_env\Scripts\activate   # Windows

# Install required packages
pip install numpy scipy matplotlib sympy pandas
pip install jupyterlab  # For interactive work
```

**Versions used in this curriculum:**
- Python 3.10+
- NumPy 1.24+
- SciPy 1.11+
- Matplotlib 3.7+

---

## Key Resources

### Official Documentation
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy Reference](https://numpy.org/doc/stable/reference/index.html)
- [Python Tutorial](https://docs.python.org/3/tutorial/)

### Textbooks
- *Numerical Python* by Robert Johansson (Apress)
- *Python for Data Analysis* by Wes McKinney (O'Reilly)
- *Computational Physics* by Mark Newman (University of Michigan)

### Online Courses
- MIT 6.100L: Introduction to CS and Programming Using Python
- Stanford CS231n: NumPy Tutorial

---

## Weekly Project: Quantum Particle-in-a-Box Simulator

Throughout this week, we build components for a complete 1D particle-in-a-box simulation:

**Monday:** Class structure for `QuantumSystem`
**Tuesday:** Grid discretization and wave function storage
**Wednesday:** Vectorized Hamiltonian application
**Thursday:** Eigenvalue solution for energy levels
**Friday:** Monte Carlo position measurement simulation
**Saturday:** Save/load functionality and performance optimization
**Sunday:** Integration and analysis

---

## Assessment Criteria

### Daily Self-Assessment
Each day includes a checklist of competencies. You should be able to:
- [ ] Explain the day's concepts without notes
- [ ] Write code implementing key algorithms from memory
- [ ] Connect the material to quantum mechanics applications

### Weekly Mastery Indicators
- [ ] Complete all practice problems (Direct, Intermediate, Challenging)
- [ ] Run all computational labs successfully
- [ ] Finish the week integration project
- [ ] Pass the Sunday comprehensive review

---

## Common Pitfalls to Avoid

1. **Using Python loops instead of NumPy operations** — Orders of magnitude slower
2. **Forgetting NumPy arrays are mutable** — `b = a` creates a view, not a copy
3. **Ignoring broadcasting rules** — Leads to shape mismatch errors
4. **Using `np.matrix` instead of `np.ndarray`** — Deprecated, causes confusion
5. **Not setting random seeds** — Results become non-reproducible

---

## Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Week 36: Month 9 Review](../../Month_09_Methods_Math_Physics_II/Week_36_Month9_Review/) | **Week 37: Python & NumPy** | [Week 38: SciPy Fundamentals](../Week_38_SciPy_Fundamentals/) |

---

*This week transforms you from a mathematics student into a computational physicist. The patterns you learn here—vectorization, broadcasting, efficient linear algebra—will appear in every simulation you write for the rest of this curriculum and beyond.*
