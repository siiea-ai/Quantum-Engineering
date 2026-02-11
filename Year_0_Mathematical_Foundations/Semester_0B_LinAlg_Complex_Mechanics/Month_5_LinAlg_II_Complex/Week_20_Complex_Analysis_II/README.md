# Week 20: Complex Analysis II — Residues and Applications

## 📋 Overview

**Week 20** (Days 134-140) completes our study of complex analysis with the powerful **Residue Theorem** and its applications. This week transforms theoretical knowledge into practical computational tools for physics.

**Total Study Time:** ~49 hours (7 hours/day × 7 days)

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. Compute Laurent series in various annular regions
2. Classify singularities (removable, poles, essential)
3. Calculate residues using multiple methods
4. Apply the Residue Theorem to contour integrals
5. Evaluate real integrals using complex methods
6. Apply the Argument Principle and Rouché's Theorem
7. Connect these tools to physics applications

---

## 📚 Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| **134** (Mon) | Laurent Series | Expansions around singularities, principal part |
| **135** (Tue) | Singularity Classification | Removable, poles, essential; Casorati-Weierstrass |
| **136** (Wed) | Residue Theorem | Proof and applications |
| **137** (Thu) | Real Integrals | Semicircular contours, Jordan's lemma |
| **138** (Fri) | Argument Principle | Winding numbers, Rouché's theorem |
| **139** (Sat) | Computational Lab | Complete toolkit, physics applications |
| **140** (Sun) | Week Review | Problem sets, Month 6 preview |

---

## 🔑 Key Theorems

### Laurent Series
$$f(z) = \sum_{n=-\infty}^{\infty} a_n(z-z_0)^n$$

### Residue Theorem
$$\oint_C f(z)\,dz = 2\pi i \sum_{\text{inside}} \text{Res}[f, z_k]$$

### Argument Principle
$$\frac{1}{2\pi i}\oint_C \frac{f'(z)}{f(z)}\,dz = Z - P$$

### Rouché's Theorem
If |g| < |f| on C, then f and f+g have the same number of zeros inside C.

---

## 🔬 Physics Connections

| Math Concept | Physics Application |
|--------------|---------------------|
| Residues | Bound state energies, resonance positions |
| Laurent series | Multipole expansions |
| Contour integrals | Propagators, Green's functions |
| Argument principle | Nyquist stability, Levinson's theorem |
| Poles in complex plane | Scattering resonances, decay widths |

---

## 📖 Primary Resources

### Textbooks
- **Churchill & Brown**, Chapters 5-7
- **Ahlfors**, "Complex Analysis", Chapters 4-5
- **Arfken & Weber**, Chapter 11

### Physics Applications
- **Sakurai**, "Modern Quantum Mechanics" — Scattering theory
- **Fetter & Walecka**, "Quantum Theory" — Green's functions

---

## 💻 Computational Tools

Python libraries:
- `numpy` — Complex arithmetic
- `scipy.integrate` — Numerical integration
- `matplotlib` — Visualization

Key implementations:
- Automated residue calculation
- Contour integral evaluation
- Domain coloring visualization
- Zero-finding via argument principle

---

## ✅ Completion Checklist

- [ ] Day 134: Laurent series mastery
- [ ] Day 135: Singularity classification
- [ ] Day 136: Residue Theorem applications
- [ ] Day 137: Real integral evaluation
- [ ] Day 138: Argument Principle and Rouché
- [ ] Day 139: Computational lab completed
- [ ] Day 140: Review and self-assessment (score ≥ 30/40)

---

## 📊 Progress Tracking

**Week Status:** ✅ COMPLETE

| Metric | Value |
|--------|-------|
| Days completed | 7/7 |
| Total study hours | ~49 |
| Problem sets | 2 (A & B) |
| Computational labs | 1 (comprehensive) |

---

## 🎓 Month 5 Complete!

With Week 20, you've completed **Month 5: Linear Algebra II & Complex Analysis**.

### Month 5 Summary:
- Week 17: Hermitian & Unitary Operators
- Week 18: Advanced Linear Algebra (SVD, Tensors)
- Week 19: Complex Analysis I (Foundations)
- Week 20: Complex Analysis II (Applications)

### Next: Month 6 — Classical Mechanics
- Week 21: Lagrangian Mechanics I
- Week 22: Lagrangian Mechanics II
- Week 23: Hamiltonian Mechanics I
- Week 24: Hamiltonian Mechanics II

---

*Part of the QSE Self-Study Curriculum*
*Semester 0B: Linear Algebra II & Complex Analysis*
*Month 5, Week 4*
