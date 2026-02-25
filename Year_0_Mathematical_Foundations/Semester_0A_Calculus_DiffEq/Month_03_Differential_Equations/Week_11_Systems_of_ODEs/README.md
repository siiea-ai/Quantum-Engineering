# Week 11: Systems of Ordinary Differential Equations

## 📋 Overview

This week covers systems of first-order ODEs—the mathematical framework for analyzing coupled, interacting components. From coupled oscillators to population dynamics to quantum states, systems of ODEs describe how multiple quantities evolve together.

**Duration:** 7 days  
**Total Study Time:** ~45 hours  
**Difficulty:** Advanced

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Convert higher-order ODEs to first-order systems
2. ✅ Solve systems using the eigenvalue method
3. ✅ Handle real, complex, and repeated eigenvalues
4. ✅ Classify and sketch phase portraits
5. ✅ Analyze stability of equilibrium points
6. ✅ Apply systems to coupled oscillators and population dynamics

---

## 📚 Required Materials

### Primary Textbook
**Boyce & DiPrima: Elementary Differential Equations (11th Edition)**
- Chapter 7: Systems of First Order Linear Equations

### Prerequisites
- Matrix algebra
- Eigenvalues and eigenvectors
- Second-order ODEs

### Software
- Python with NumPy, SciPy, Matplotlib

---

## 📅 Daily Schedule

| Day | Topic | File |
|-----|-------|------|
| 71 (Mon) | Introduction to Systems | [Day_71_Monday.md](Day_71_Monday.md) |
| 72 (Tue) | The Eigenvalue Method | [Day_72_Tuesday.md](Day_72_Tuesday.md) |
| 73 (Wed) | Complex and Repeated Eigenvalues | [Day_73_Wednesday.md](Day_73_Wednesday.md) |
| 74 (Thu) | Phase Portraits and Stability | [Day_74_Thursday.md](Day_74_Thursday.md) |
| 75 (Fri) | Problem Set | [Day_75_Friday.md](Day_75_Friday.md) |
| 76 (Sat) | Computational Lab | [Day_76_Saturday.md](Day_76_Saturday.md) |
| 77 (Sun) | Review & Rest | [Day_77_Sunday.md](Day_77_Sunday.md) |

---

## 📊 Key Formulas

### System Form
$$\mathbf{x}' = A\mathbf{x}$$

### General Solution
$$\mathbf{x}(t) = c_1\mathbf{v}_1 e^{\lambda_1 t} + c_2\mathbf{v}_2 e^{\lambda_2 t}$$

### Stability Criterion
- All Re(λ) < 0: Asymptotically stable
- Any Re(λ) > 0: Unstable

---

## 📊 Phase Portrait Types

| Eigenvalues | Type | Stability |
|-------------|------|-----------|
| Both negative real | Stable node | Asymptotically stable |
| Both positive real | Unstable node | Unstable |
| Opposite signs | Saddle | Unstable |
| Complex, Re < 0 | Stable spiral | Asymptotically stable |
| Complex, Re > 0 | Unstable spiral | Unstable |
| Pure imaginary | Center | Stable |

---

## 📊 Assessment

### Problem Set (Day 75)
- 200 points total
- Part I: Matrix formulation (40 pts)
- Part II: Eigenvalue method (50 pts)
- Part III: Phase portraits (50 pts)
- Part IV: Applications (60 pts)

---

## 🔗 Connections to Quantum Mechanics

| System Concept | QM Application |
|----------------|----------------|
| Eigenvalues | Energy levels |
| Eigenvectors | Stationary states |
| Matrix exponential | Time evolution operator |
| Coupled systems | Interacting qubits |
| Phase portraits | Bloch sphere dynamics |

---

## ➡️ Next Week

**Week 12: Laplace Transforms**
- Transform methods for ODEs
- Step functions and impulses
- Applications to circuits and control systems

---

*"Systems of equations reveal the dance of interacting components—each eigenvalue a natural frequency of the whole."*
