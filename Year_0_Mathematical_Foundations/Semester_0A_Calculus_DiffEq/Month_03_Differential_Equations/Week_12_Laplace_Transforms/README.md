# Week 12: Laplace Transforms

## 📋 Overview

This week covers Laplace transforms—a powerful technique that converts differential equations into algebraic equations. Essential for control systems, signal processing, and any application involving discontinuous or impulsive forcing.

**Duration:** 7 days  
**Total Study Time:** ~45 hours  
**Difficulty:** Advanced

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Compute Laplace transforms using definition and tables
2. ✅ Apply linearity and shifting theorems
3. ✅ Find inverse transforms using partial fractions
4. ✅ Solve ODEs using the Laplace method
5. ✅ Handle step functions and impulses
6. ✅ Apply transforms to circuits and mechanical systems

---

## 📚 Required Materials

### Primary Textbook
**Boyce & DiPrima: Elementary Differential Equations (11th Edition)**
- Chapter 6: The Laplace Transform

### Prerequisites
- Partial fractions
- Second-order ODEs
- Complex numbers

### Software
- Python with SymPy, NumPy, SciPy, Matplotlib

---

## 📅 Daily Schedule

| Day | Topic | File |
|-----|-------|------|
| 78 (Mon) | Introduction to Laplace Transforms | [Day_78_Monday.md](Day_78_Monday.md) |
| 79 (Tue) | Inverse Laplace Transforms | [Day_79_Tuesday.md](Day_79_Tuesday.md) |
| 80 (Wed) | Solving ODEs with Laplace | [Day_80_Wednesday.md](Day_80_Wednesday.md) |
| 81 (Thu) | Step Functions and Impulses | [Day_81_Thursday.md](Day_81_Thursday.md) |
| 82 (Fri) | Problem Set | [Day_82_Friday.md](Day_82_Friday.md) |
| 83 (Sat) | Computational Lab | [Day_83_Saturday.md](Day_83_Saturday.md) |
| 84 (Sun) | Review & Rest | [Day_84_Sunday.md](Day_84_Sunday.md) |

---

## 📊 Key Formulas

### Definition
$$\mathcal{L}\{f(t)\} = \int_0^\infty f(t)e^{-st} dt$$

### Transform of Derivatives
$$\mathcal{L}\{f'\} = sF(s) - f(0)$$
$$\mathcal{L}\{f''\} = s^2F(s) - sf(0) - f'(0)$$

### Shifting Theorems
- **s-shifting:** $\mathcal{L}\{e^{at}f(t)\} = F(s-a)$
- **t-shifting:** $\mathcal{L}\{u(t-c)f(t-c)\} = e^{-cs}F(s)$

---

## 📊 Assessment

### Problem Set (Day 82)
- 200 points total
- Part I: Basic transforms (40 pts)
- Part II: Solving ODEs (50 pts)
- Part III: Step functions & impulses (50 pts)
- Part IV: Applications (60 pts)

---

## 🔗 Connections to Quantum Mechanics

| Laplace Concept | QM Application |
|-----------------|----------------|
| Transform pairs | Fourier analysis of wavefunctions |
| Poles | Energy eigenvalues |
| Impulse response | Green's functions |
| Transfer function | Scattering matrix |
| Convolution | Time evolution |

---

## ➡️ Completion

**This completes Month 3 and Semester 0A!**

Next: Semester 0B (Advanced Math/Physics)
- Month 4: Linear Algebra I
- Month 5: Linear Algebra II & Complex Analysis
- Month 6: Classical Mechanics

---

*"Laplace transforms turn differential equations into algebra—the final tool in your ODE toolkit."*
