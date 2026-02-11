# Week 07: Multiple Integrals

## 📋 Overview

This week extends integration to multiple dimensions. You'll learn to evaluate double and triple integrals, work in polar/cylindrical/spherical coordinates, and apply these techniques to compute areas, volumes, masses, and centers of mass.

**Duration:** 7 days  
**Total Study Time:** ~45 hours  
**Difficulty:** Intermediate-Advanced

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Evaluate double integrals over rectangles and general regions
2. ✅ Convert to polar coordinates for circular regions
3. ✅ Set up and evaluate triple integrals
4. ✅ Compute areas, volumes, and masses
5. ✅ Find centers of mass and moments of inertia

---

## 📚 Required Materials

### Primary Textbook
**Stewart's Calculus (8th Edition)**
- Section 15.1: Double Integrals over Rectangles
- Section 15.2: Double Integrals over General Regions
- Section 15.3: Double Integrals in Polar Coordinates
- Section 15.4: Applications of Double Integrals
- Section 15.6: Triple Integrals

### Software
- Python 3.9+ with NumPy, SciPy, Matplotlib
- scipy.integrate for numerical integration

---

## 📅 Daily Schedule

| Day | Topic | File |
|-----|-------|------|
| 43 (Mon) | Double Integrals over Rectangles | [Day_43_Monday.md](Day_43_Monday.md) |
| 44 (Tue) | Double Integrals over General Regions | [Day_44_Tuesday.md](Day_44_Tuesday.md) |
| 45 (Wed) | Polar Coordinates | [Day_45_Wednesday.md](Day_45_Wednesday.md) |
| 46 (Thu) | Triple Integrals & Applications | [Day_46_Thursday.md](Day_46_Thursday.md) |
| 47 (Fri) | Problem Set | [Day_47_Friday.md](Day_47_Friday.md) |
| 48 (Sat) | Computational Lab | [Day_48_Saturday.md](Day_48_Saturday.md) |
| 49 (Sun) | Review & Rest | [Day_49_Sunday.md](Day_49_Sunday.md) |

---

## 📊 Key Formulas

### Double Integral (Rectangle)
$$\iint_R f(x,y) \, dA = \int_a^b \int_c^d f(x,y) \, dy \, dx$$

### Double Integral (General Region - Type I)
$$\iint_D f(x,y) \, dA = \int_a^b \int_{g_1(x)}^{g_2(x)} f(x,y) \, dy \, dx$$

### Polar Coordinates
$$\iint_R f(x,y) \, dA = \iint f(r\cos\theta, r\sin\theta) \cdot r \, dr \, d\theta$$

### Triple Integral
$$\iiint_E f(x,y,z) \, dV = \int\int\int f \, dz \, dy \, dx$$

### Applications
| Quantity | Formula |
|----------|---------|
| Area | $\iint_D 1 \, dA$ |
| Volume | $\iiint_E 1 \, dV$ |
| Mass | $\iint_D \rho \, dA$ |
| Center of Mass | $\bar{x} = \frac{1}{m}\iint x\rho \, dA$ |

---

## 📊 Assessment

### Problem Set (Day 47)
- 153 points total
- Double integrals (rectangular and general regions)
- Polar coordinate integrals
- Triple integrals
- Applications

---

## 🔗 Connections to Quantum Mechanics

| Calculus Concept | QM Application |
|------------------|----------------|
| Double integrals | 2D probability calculations |
| Triple integrals | 3D wave function normalization |
| Polar coordinates | Angular momentum eigenstates |
| Mass/center of mass | Expectation values |

---

## ➡️ Next Week

**Week 08: Vector Calculus**
- Vector fields
- Line integrals
- Green's Theorem
- Surface integrals
- Divergence and Stokes' Theorems

---

*"Multiple integrals let us measure the unmeasurable—volumes, masses, and probabilities in higher dimensions."*
