# Month 3: Differential Equations

## 📋 Overview

Month 3 covers ordinary differential equations (ODEs)—the mathematical language of change and dynamics. From first-order equations modeling population growth to second-order equations describing oscillators, to systems modeling interacting components, to Laplace transforms providing powerful solution techniques.

**Duration:** 4 weeks (28 days)  
**Total Study Time:** ~180 hours  
**Days:** 57-84

---

## 🎯 Month Learning Objectives

By the end of this month, you will be able to:

1. ✅ Classify and solve first-order ODEs (separable, linear, exact, Bernoulli)
2. ✅ Solve second-order linear ODEs using characteristic equations
3. ✅ Apply undetermined coefficients and variation of parameters
4. ✅ Model mechanical and electrical oscillations
5. ✅ Solve systems of ODEs using eigenvalue methods
6. ✅ Analyze stability via phase portraits
7. ✅ Use Laplace transforms to solve ODEs with discontinuous forcing

---

## 📚 Required Materials

### Primary Textbook
**Boyce & DiPrima: Elementary Differential Equations (11th Edition)**
- Chapters 2-7

### Alternative
- Zill: A First Course in Differential Equations
- Edwards & Penney: Differential Equations and Boundary Value Problems

### Software
- Python with NumPy, SciPy, SymPy, Matplotlib

---

## 📅 Weekly Schedule

| Week | Days | Topic | Focus |
|------|------|-------|-------|
| **9** | 57-63 | First-Order ODEs | Classification, solution methods, applications |
| **10** | 64-70 | Second-Order ODEs | Oscillations, resonance, circuits |
| **11** | 71-77 | Systems of ODEs | Eigenvalues, phase portraits, stability |
| **12** | 78-84 | Laplace Transforms | Transform methods, discontinuous forcing |

---

## 📁 Directory Structure

```
Month_03_Differential_Equations/
├── README.md (this file)
├── Week_09_First_Order_ODEs/
│   ├── Day_57_Monday.md through Day_63_Sunday.md
│   └── README.md
├── Week_10_Second_Order_ODEs/
│   ├── Day_64_Monday.md through Day_70_Sunday.md
│   └── README.md
├── Week_11_Systems_of_ODEs/
│   ├── Day_71_Monday.md through Day_77_Sunday.md
│   └── README.md
└── Week_12_Laplace_Transforms/
    ├── Day_78_Monday.md through Day_84_Sunday.md
    └── README.md
```

---

## 📊 Key Concepts by Week

### Week 9: First-Order ODEs
- Separable: $\frac{dy}{dx} = g(x)h(y)$
- Linear: $y' + P(x)y = Q(x)$ with integrating factor
- Exact: $M dx + N dy = 0$ where $M_y = N_x$
- Bernoulli: $y' + Py = Qy^n$
- Applications: population, cooling, mixing

### Week 10: Second-Order ODEs
- Characteristic equation: $ar^2 + br + c = 0$
- Three cases: real distinct, complex, repeated roots
- Undetermined coefficients for polynomial/exponential/trig forcing
- Variation of parameters for general forcing
- Mechanical oscillations: underdamped, critical, overdamped
- Resonance: $\omega = \omega_0$

### Week 11: Systems of ODEs
- Matrix form: $\mathbf{x}' = A\mathbf{x}$
- Eigenvalue method: $\mathbf{x} = \mathbf{v}e^{\lambda t}$
- Phase portraits: nodes, saddles, spirals, centers
- Stability: Re(λ) < 0 ⟹ asymptotically stable

### Week 12: Laplace Transforms
- Definition: $\mathcal{L}\{f\} = \int_0^\infty f(t)e^{-st}dt$
- Shifting theorems (s and t)
- Step and impulse functions
- Solving IVPs: transform → algebra → invert

---

## 📊 Assessment Summary

| Week | Problem Set | Points | Target |
|------|-------------|--------|--------|
| 9 | First-Order ODEs | 200 | 160+ |
| 10 | Second-Order ODEs | 200 | 160+ |
| 11 | Systems of ODEs | 200 | 160+ |
| 12 | Laplace Transforms | 200 | 160+ |

---

## 🔗 Connections to Quantum Mechanics

| ODE Topic | QM Application |
|-----------|----------------|
| First-order decay | Radioactive decay, state transitions |
| Harmonic oscillator | Quantum harmonic oscillator, phonons |
| Damping | Decoherence, open quantum systems |
| Systems | Multi-level atoms, coupled qubits |
| Eigenvalues | Energy levels |
| Phase portraits | Bloch sphere dynamics |
| Laplace transforms | Green's functions, propagators |

---

## 📈 Month Progression

```
Week 9: First-Order
    ↓ (increase order)
Week 10: Second-Order  
    ↓ (add coupling)
Week 11: Systems
    ↓ (add tools)
Week 12: Laplace Transforms
```

---

## ✅ Completion Checklist

### Week 9
- [ ] All 7 daily lessons completed
- [ ] Problem set scored 160+/200
- [ ] Computational lab finished

### Week 10
- [ ] All 7 daily lessons completed
- [ ] Problem set scored 160+/200
- [ ] Computational lab finished

### Week 11
- [ ] All 7 daily lessons completed
- [ ] Problem set scored 160+/200
- [ ] Computational lab finished

### Week 12
- [ ] All 7 daily lessons completed
- [ ] Problem set scored 160+/200
- [ ] Computational lab finished

---

## ➡️ What's Next

**Month 3 completes Semester 0A: Calculus & Differential Equations!**

Next: **Semester 0B: Advanced Math/Physics**
- Month 4: Linear Algebra I
- Month 5: Linear Algebra II & Complex Analysis
- Month 6: Classical Mechanics

---

*"Differential equations are the language of change—master them, and you can describe any evolving system in nature."*
