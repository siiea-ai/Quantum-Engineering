# Week 10: Second-Order Linear ODEs

## 📋 Overview

This week covers second-order linear ordinary differential equations—the mathematical heart of oscillatory physics. From pendulums to quantum states, second-order ODEs describe systems where acceleration depends on position and velocity.

**Duration:** 7 days  
**Total Study Time:** ~45 hours  
**Difficulty:** Intermediate-Advanced

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Solve homogeneous 2nd-order ODEs via the characteristic equation
2. ✅ Handle all three root cases (real distinct, complex, repeated)
3. ✅ Apply undetermined coefficients for nonhomogeneous equations
4. ✅ Use variation of parameters for general forcing functions
5. ✅ Model mechanical and electrical oscillations
6. ✅ Understand damping regimes and resonance

---

## 📚 Required Materials

### Primary Textbook
**Boyce & DiPrima: Elementary Differential Equations (11th Edition)**
- Chapter 3: Second Order Linear Equations

### Alternative Textbooks
- Zill: A First Course in Differential Equations
- Edwards & Penney: Differential Equations

### Software
- Python with NumPy, SciPy, Matplotlib, SymPy

---

## 📅 Daily Schedule

| Day | Topic | File |
|-----|-------|------|
| 64 (Mon) | Homogeneous Equations & Characteristic Equation | [Day_64_Monday.md](Day_64_Monday.md) |
| 65 (Tue) | Nonhomogeneous: Undetermined Coefficients | [Day_65_Tuesday.md](Day_65_Tuesday.md) |
| 66 (Wed) | Variation of Parameters | [Day_66_Wednesday.md](Day_66_Wednesday.md) |
| 67 (Thu) | Mechanical & Electrical Oscillations | [Day_67_Thursday.md](Day_67_Thursday.md) |
| 68 (Fri) | Problem Set | [Day_68_Friday.md](Day_68_Friday.md) |
| 69 (Sat) | Computational Lab | [Day_69_Saturday.md](Day_69_Saturday.md) |
| 70 (Sun) | Review & Rest | [Day_70_Sunday.md](Day_70_Sunday.md) |

---

## 📊 Key Formulas

### Characteristic Equation
For $ay'' + by' + cy = 0$:
$$ar^2 + br + c = 0$$

### Three Cases

| Case | Condition | Solution |
|------|-----------|----------|
| Real distinct | $b^2 > 4ac$ | $y = c_1 e^{r_1 x} + c_2 e^{r_2 x}$ |
| Complex | $b^2 < 4ac$ | $y = e^{\alpha x}(c_1 \cos\beta x + c_2 \sin\beta x)$ |
| Repeated | $b^2 = 4ac$ | $y = (c_1 + c_2 x)e^{rx}$ |

### Nonhomogeneous Solution
$$y = y_h + y_p$$

### Variation of Parameters
$$y_p = -y_1 \int \frac{y_2 f}{W} dx + y_2 \int \frac{y_1 f}{W} dx$$

---

## 📊 Applications

| Physical System | Equation | Key Feature |
|-----------------|----------|-------------|
| Simple harmonic motion | $x'' + \omega_0^2 x = 0$ | Eternal oscillation |
| Damped oscillator | $x'' + 2\gamma x' + \omega_0^2 x = 0$ | Decaying oscillation |
| Forced oscillator | $x'' + 2\gamma x' + \omega_0^2 x = F_0\cos\omega t$ | Steady state + transient |
| RLC circuit | $LQ'' + RQ' + Q/C = E(t)$ | Electrical analog |

---

## 📊 Assessment

### Problem Set (Day 68)
- 200 points total
- Part I: Homogeneous equations (50 pts)
- Part II: Nonhomogeneous equations (50 pts)
- Part III: Applications (50 pts)
- Part IV: Comprehensive (50 pts)

---

## 🔗 Connections to Quantum Mechanics

| ODE Concept | QM Application |
|-------------|----------------|
| Complex roots | Wave functions, oscillatory behavior |
| Resonance | Transition frequencies, spectroscopy |
| Harmonic oscillator | QHO energy levels, phonons |
| Damping | Decoherence, dissipative systems |

---

## ➡️ Next Week

**Week 11: Systems of ODEs**
- Matrix formulation
- Eigenvalue methods
- Phase portraits
- Coupled oscillators

---

*"Second-order equations are the heartbeat of physics—every oscillation follows their rhythm."*
