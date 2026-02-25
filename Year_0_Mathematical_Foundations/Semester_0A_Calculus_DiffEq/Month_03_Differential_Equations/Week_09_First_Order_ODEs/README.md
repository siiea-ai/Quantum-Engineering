# Week 09: First-Order Ordinary Differential Equations

## 📋 Overview

This week introduces differential equations—the mathematical language of change. You'll master various techniques for solving first-order ODEs and apply them to real-world problems in physics, biology, and engineering.

**Duration:** 7 days  
**Total Study Time:** ~45 hours  
**Difficulty:** Intermediate

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Classify first-order ODEs by type
2. ✅ Solve separable equations
3. ✅ Apply the integrating factor method for linear equations
4. ✅ Recognize and solve exact equations
5. ✅ Use substitution methods (Bernoulli, homogeneous)
6. ✅ Model real-world phenomena with differential equations

---

## 📚 Required Materials

### Primary Textbook
**Boyce & DiPrima: Elementary Differential Equations (11th Edition)**
- Chapter 1: Introduction
- Chapter 2: First Order Differential Equations

### Alternative Textbooks
- Zill: A First Course in Differential Equations
- Edwards & Penney: Differential Equations

### Software
- Python 3.9+ with NumPy, SciPy, Matplotlib, SymPy
- scipy.integrate for numerical solutions

---

## 📅 Daily Schedule

| Day | Topic | File |
|-----|-------|------|
| 57 (Mon) | Introduction & Separable Equations | [Day_57_Monday.md](Day_57_Monday.md) |
| 58 (Tue) | Linear First-Order Equations | [Day_58_Tuesday.md](Day_58_Tuesday.md) |
| 59 (Wed) | Exact Equations & Special Methods | [Day_59_Wednesday.md](Day_59_Wednesday.md) |
| 60 (Thu) | Applications & Modeling | [Day_60_Thursday.md](Day_60_Thursday.md) |
| 61 (Fri) | Problem Set | [Day_61_Friday.md](Day_61_Friday.md) |
| 62 (Sat) | Computational Lab | [Day_62_Saturday.md](Day_62_Saturday.md) |
| 63 (Sun) | Review & Rest | [Day_63_Sunday.md](Day_63_Sunday.md) |

---

## 📊 Key Methods

### Separable Equations
$$\frac{dy}{dx} = f(x)g(y) \quad \Rightarrow \quad \int \frac{dy}{g(y)} = \int f(x) \, dx$$

### Linear Equations
$$y' + P(x)y = Q(x) \quad \Rightarrow \quad \mu = e^{\int P dx}$$

### Exact Equations
$$M dx + N dy = 0 \text{ is exact if } \frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$$

### Bernoulli Equations
$$y' + P(x)y = Q(x)y^n \quad \Rightarrow \quad v = y^{1-n}$$

---

## 📊 Key Applications

| Application | Model | Solution |
|-------------|-------|----------|
| Exponential growth | $y' = ky$ | $y = y_0 e^{kt}$ |
| Newton's cooling | $T' = -k(T-T_s)$ | $T = T_s + (T_0-T_s)e^{-kt}$ |
| Logistic growth | $P' = rP(1-P/K)$ | $P = K/(1+Ae^{-rt})$ |
| Radioactive decay | $N' = -\lambda N$ | $N = N_0 e^{-\lambda t}$ |

---

## 📊 Assessment

### Problem Set (Day 61)
- 200 points total
- Part I: Separable equations (40 pts)
- Part II: Linear equations (50 pts)
- Part III: Exact & special equations (50 pts)
- Part IV: Applications (60 pts)

---

## 🔗 Connections to Quantum Mechanics

| ODE Concept | QM Application |
|-------------|----------------|
| Exponential decay | Radioactive decay, state lifetimes |
| Linear first-order | Time evolution of amplitudes |
| Separable equations | Separation of variables in Schrödinger eq |
| Initial conditions | Initial state preparation |

---

## ➡️ Next Week

**Week 10: Second-Order Linear ODEs**
- Homogeneous equations
- Characteristic equation
- Complex roots and oscillations
- Method of undetermined coefficients

---

*"Differential equations are the language nature uses to express its laws of change."*
