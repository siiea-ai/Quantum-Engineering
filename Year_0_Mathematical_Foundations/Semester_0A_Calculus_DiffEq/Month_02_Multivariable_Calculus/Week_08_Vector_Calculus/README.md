# Week 08: Vector Calculus

## 📋 Overview

This week introduces the calculus of vector fields—the mathematical framework for electromagnetism, fluid dynamics, and quantum mechanics. You'll learn about divergence, curl, line integrals, surface integrals, and the three fundamental theorems.

**Duration:** 7 days  
**Total Study Time:** ~45 hours  
**Difficulty:** Advanced

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Work with vector fields in 2D and 3D
2. ✅ Compute divergence and curl
3. ✅ Evaluate line and surface integrals
4. ✅ Apply Green's Theorem
5. ✅ Apply Stokes' Theorem
6. ✅ Apply the Divergence Theorem

---

## 📚 Required Materials

### Primary Textbook
**Stewart's Calculus (8th Edition)**
- Section 16.1: Vector Fields
- Section 16.2: Line Integrals
- Section 16.3: Fundamental Theorem for Line Integrals
- Section 16.4: Green's Theorem
- Section 16.5: Curl and Divergence
- Section 16.6-16.7: Surface Integrals
- Section 16.8: Stokes' Theorem
- Section 16.9: Divergence Theorem

### Software
- Python 3.9+ with NumPy, SciPy, Matplotlib
- 3D visualization with mplot3d

---

## 📅 Daily Schedule

| Day | Topic | File |
|-----|-------|------|
| 50 (Mon) | Vector Fields | [Day_50_Monday.md](Day_50_Monday.md) |
| 51 (Tue) | Line Integrals | [Day_51_Tuesday.md](Day_51_Tuesday.md) |
| 52 (Wed) | Green's Theorem | [Day_52_Wednesday.md](Day_52_Wednesday.md) |
| 53 (Thu) | Surface Integrals & Big Theorems | [Day_53_Thursday.md](Day_53_Thursday.md) |
| 54 (Fri) | Problem Set | [Day_54_Friday.md](Day_54_Friday.md) |
| 55 (Sat) | Computational Lab | [Day_55_Saturday.md](Day_55_Saturday.md) |
| 56 (Sun) | Review & Month 2 Conclusion | [Day_56_Sunday.md](Day_56_Sunday.md) |

---

## 📊 Key Formulas

### Divergence
$$\nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$$

### Curl
$$\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \partial_x & \partial_y & \partial_z \\ P & Q & R \end{vmatrix}$$

### Line Integral
$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t) \, dt$$

### Green's Theorem
$$\oint_C P \, dx + Q \, dy = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$$

### Stokes' Theorem
$$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$$

### Divergence Theorem
$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_E \nabla \cdot \mathbf{F} \, dV$$

---

## 📊 Assessment

### Problem Set (Day 54)
- 179 points total
- Vector fields, divergence, curl
- Line and surface integrals
- Green's, Stokes', Divergence Theorems

---

## 🔗 Connections to Quantum Mechanics

| Calculus Concept | QM Application |
|------------------|----------------|
| Divergence | Continuity equation |
| Curl | Angular momentum |
| Line integrals | Berry phase |
| Stokes' Theorem | Aharonov-Bohm effect |
| Divergence Theorem | Gauss's law |

---

## ➡️ Next Month

**Month 03: Differential Equations**
- First-order ODEs
- Second-order linear ODEs
- Systems of ODEs
- Laplace transforms

---

*"Vector calculus is the mathematical foundation of field theory—the language of forces that act at every point in space."*
