# Day 133: Week 19 Review — Complex Analysis I Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Concept Review & Problem Set A |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Set B & Self-Assessment |
| Evening | 6:00 PM - 7:00 PM | 1 hour | Week 20 Preview & Planning |

**Total Study Time: 7 hours**

---

## 🎯 Week 19 Learning Objectives — Final Check

By the end of this review, confirm mastery of:

- [ ] Complex number arithmetic and geometry
- [ ] Topology of the complex plane
- [ ] Complex differentiation and the Cauchy-Riemann equations
- [ ] Analytic vs non-analytic functions
- [ ] Elementary complex functions (exp, log, trig, powers)
- [ ] Multi-valued functions and branch cuts
- [ ] Contour integration fundamentals
- [ ] Cauchy's Theorem and Integral Formula
- [ ] Applications to quantum mechanics

---

## 📊 Week 19 Concept Map

```
                    COMPLEX ANALYSIS I
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   FOUNDATIONS      ANALYTICITY       INTEGRATION
        │                 │                 │
   ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
   │         │      │         │      │         │
Complex   Topology  Cauchy-  Harmonic Contour  Cauchy's
Numbers   of ℂ     Riemann  Functions Integrals Theorem
   │         │      │         │      │         │
   ├─Polar   ├─Open/  ├─∂u/∂x=∂v/∂y  ├─∮f dz  ├─∮f dz=0
   │ form    │ Closed │         │    │        │ (analytic)
   ├─Euler's ├─Limits ├─∂u/∂y=-∂v/∂x ├─Path   ├─Integral
   │ formula │        │         │    │ param  │ Formula
   └─Roots   └─Riemann └─Entire  └─∇²φ=0     └─f(z₀)=...
     of unity  sphere   functions
        │                 │                 │
        └────────────┬────┴────────────────┘
                     │
              ELEMENTARY FUNCTIONS
                     │
         ┌───────────┼───────────┐
         │           │           │
       e^z        log z      Complex
         │           │        Powers
    ┌────┴────┐  ┌───┴───┐      │
    │         │  │       │   z^α = e^(α log z)
  Period   Maps Multi-  Branch    │
  2πi     strips valued  cuts   Multi-valued
                              (non-integer α)
                     │
              ┌──────┴──────┐
              │             │
         sin z, cos z   sinh z, cosh z
              │             │
          Unbounded     Related by
          in ℂ         cos(iz)=cosh(z)
```

---

## 📝 Key Formulas Summary

### Complex Numbers
| Formula | Description |
|---------|-------------|
| z = x + iy = re^{iθ} | Cartesian and polar forms |
| e^{iθ} = cos θ + i sin θ | Euler's formula |
| z̄ = x - iy | Complex conjugate |
| \|z\| = √(zz̄) = √(x² + y²) | Modulus |
| arg(z) = arctan(y/x) | Argument |

### Analyticity
| Formula | Description |
|---------|-------------|
| ∂u/∂x = ∂v/∂y, ∂u/∂y = -∂v/∂x | Cauchy-Riemann equations |
| f'(z₀) = ∂u/∂x + i∂v/∂x | Derivative formula |
| ∇²u = ∇²v = 0 | Harmonic functions |

### Elementary Functions
| Function | Definition | Key Property |
|----------|------------|--------------|
| e^z | e^x(cos y + i sin y) | Entire, period 2πi |
| log z | ln\|z\| + i(arg z + 2πn) | Multi-valued |
| z^α | e^{α log z} | Multi-valued (non-integer α) |
| sin z | (e^{iz} - e^{-iz})/2i | Entire, unbounded |
| cos z | (e^{iz} + e^{-iz})/2 | Entire, unbounded |

### Integration
| Formula | Description |
|---------|-------------|
| ∮ z^n dz = 2πi δ_{n,-1} | Fundamental integral |
| ∮_γ f dz = 0 | Cauchy's theorem (f analytic) |
| f(z₀) = (1/2πi) ∮ f(z)/(z-z₀) dz | Cauchy's integral formula |
| f^{(n)}(z₀) = (n!/2πi) ∮ f(z)/(z-z₀)^{n+1} dz | Derivative formula |

---

## 🔬 Problem Set A: Foundations & Analyticity

### Problem A1: Complex Arithmetic
Compute in both Cartesian and polar form:
a) (2 + 3i)(1 - 4i)
b) (1 + i)^6
c) All cube roots of -8

**Solutions:**
a) (2 + 3i)(1 - 4i) = 2 - 8i + 3i - 12i² = 2 - 5i + 12 = **14 - 5i**
   Polar: |14 - 5i| = √221 ≈ 14.87, arg = arctan(-5/14) ≈ -0.343
   
b) (1 + i)^6 = (√2 e^{iπ/4})^6 = 8 e^{3iπ/2} = 8(-i) = **-8i**

c) -8 = 8e^{iπ}, so roots are:
   - 2e^{iπ/3} = 2(1/2 + i√3/2) = **1 + i√3**
   - 2e^{iπ} = **-2**
   - 2e^{i5π/3} = 2(1/2 - i√3/2) = **1 - i√3**

---

### Problem A2: Cauchy-Riemann Verification
For each function, determine where it is analytic:

a) f(z) = z² + z
b) f(z) = Re(z)
c) f(z) = z · z̄
d) f(z) = z³ - 3z

**Solutions:**
a) f(z) = z² + z is a polynomial → **entire** (analytic everywhere)

b) f(z) = Re(z) = x, so u = x, v = 0
   ∂u/∂x = 1, ∂v/∂y = 0 → Not equal!
   **Nowhere analytic**

c) f(z) = zz̄ = |z|² = x² + y², so u = x² + y², v = 0
   ∂u/∂x = 2x, ∂v/∂y = 0 → Equal only at x = 0
   ∂u/∂y = 2y, -∂v/∂x = 0 → Equal only at y = 0
   Only satisfied at z = 0, not a neighborhood.
   **Nowhere analytic**

d) f(z) = z³ - 3z is a polynomial → **entire**

---

### Problem A3: Harmonic Conjugate
Given u(x,y) = x³ - 3xy², find its harmonic conjugate v and the corresponding analytic function f.

**Solution:**
First verify u is harmonic:
∂u/∂x = 3x² - 3y², ∂²u/∂x² = 6x
∂u/∂y = -6xy, ∂²u/∂y² = -6x
∇²u = 6x - 6x = 0 ✓

From C-R: ∂v/∂y = ∂u/∂x = 3x² - 3y²
Integrating: v = 3x²y - y³ + g(x)

From C-R: ∂v/∂x = 2·3xy + g'(x) = -∂u/∂y = 6xy
So g'(x) = 0, meaning g(x) = C

**v = 3x²y - y³ + C**

The analytic function: f = u + iv = (x³ - 3xy²) + i(3x²y - y³) = **z³**

---

### Problem A4: Topology
For each set, determine if it is: (i) open, (ii) closed, (iii) connected, (iv) simply connected:

a) {z : |z| < 1}
b) {z : |z| ≤ 1}
c) {z : 0 < |z| < 1}
d) {z : Re(z) > 0}

**Solutions:**
a) Open disk: (i) open ✓, (ii) not closed, (iii) connected ✓, (iv) simply connected ✓

b) Closed disk: (i) not open, (ii) closed ✓, (iii) connected ✓, (iv) simply connected ✓

c) Punctured disk: (i) open ✓, (ii) not closed, (iii) connected ✓, (iv) **not simply connected** (holes around origin)

d) Right half-plane: (i) open ✓, (ii) not closed, (iii) connected ✓, (iv) simply connected ✓

---

## 🔬 Problem Set B: Functions & Integration

### Problem B1: Elementary Functions
Compute:

a) All values of log(-e)
b) Principal value of (-1)^i
c) sin(i)
d) All solutions to e^z = 1 + i

**Solutions:**
a) -e = e · e^{iπ} = e^{1+iπ}
   log(-e) = 1 + i(π + 2πn) = **1 + i(2n+1)π** for n ∈ ℤ

b) (-1)^i = e^{i·log(-1)} = e^{i·iπ} = e^{-π} ≈ **0.0432** (principal value)

c) sin(i) = (e^{i·i} - e^{-i·i})/(2i) = (e^{-1} - e^1)/(2i) = -sinh(1)/i = **i·sinh(1) ≈ 1.175i**

d) e^z = 1 + i = √2 e^{iπ/4}
   z = ln(√2) + i(π/4 + 2πn) = **(1/2)ln 2 + i(π/4 + 2πn)**

---

### Problem B2: Multi-valued Functions
a) Find all values of (1+i)^{1/2}
b) Find all values of i^{2i}
c) On what domain is Log(z²) = 2 Log(z)?

**Solutions:**
a) 1+i = √2 e^{iπ/4}
   (1+i)^{1/2} = 2^{1/4} e^{i(π/8 + πn)} for n = 0, 1
   - n=0: **2^{1/4} e^{iπ/8} ≈ 1.099 + 0.455i**
   - n=1: **2^{1/4} e^{i9π/8} ≈ -1.099 - 0.455i**

b) i^{2i} = e^{2i·log(i)} = e^{2i·i(π/2 + 2πn)} = e^{-π - 4πn}
   Values: **e^{-π(1+4n)}** for n ∈ ℤ (infinitely many real values!)

c) Log(z²) = 2 Log(z) fails when:
   - z crosses negative real axis (branch cut of Log)
   - z² crosses negative real axis
   
   It holds on: **{z : -π/2 < Arg(z) < π/2}** (right half-plane excluding imaginary axis)

---

### Problem B3: Contour Integration
Evaluate each integral:

a) ∮_{|z|=1} (z + 1/z) dz
b) ∮_{|z|=2} dz/(z-1)
c) ∮_{|z|=2} z² e^z/(z-1) dz
d) ∮_{|z|=1} e^z/z³ dz

**Solutions:**
a) ∮(z + 1/z) dz = ∮ z dz + ∮ dz/z = 0 + 2πi = **2πi**
   (z is entire so first integral is 0; second is fundamental)

b) f(z) = 1 is entire, pole of 1/(z-1) is at z = 1 inside |z| = 2
   By Cauchy's formula: **2πi**

c) Let f(z) = z² e^z (entire). By Cauchy's formula:
   ∮ f(z)/(z-1) dz = 2πi · f(1) = 2πi · 1² · e = **2πie**

d) f(z) = e^z, need f''(0) by derivative formula.
   f''(z) = e^z, so f''(0) = 1.
   ∮ e^z/z³ dz = 2πi · f''(0)/2! = 2πi · 1/2 = **πi**

---

### Problem B4: Applications
a) Use contour methods to show ∫₀^∞ dx/(1+x⁴) = π/(2√2)

b) Prove: If f is entire and |f(z)| ≤ M|z|^n for large |z|, then f is a polynomial of degree ≤ n.

**Solutions:**
a) Consider ∮ dz/(1+z⁴) over contour: real axis [-R, R] plus upper semicircle.
   
   Poles of 1/(1+z⁴): z⁴ = -1 = e^{iπ}, so z = e^{i(π+2πk)/4}
   - z₁ = e^{iπ/4} = (1+i)/√2 (in upper half-plane)
   - z₂ = e^{i3π/4} = (-1+i)/√2 (in upper half-plane)
   
   Residues: At simple pole z_k, Res = 1/(4z_k³) = z_k/(4z_k⁴) = z_k/(-4)
   - Res(z₁) = -e^{iπ/4}/4
   - Res(z₂) = -e^{i3π/4}/4
   
   Sum of residues = -(1/4)[e^{iπ/4} + e^{i3π/4}] = -(1/4)[√2 i] = -i√2/4
   
   ∮ = 2πi · (-i√2/4) = π√2/2
   
   Semicircle contribution → 0 as R → ∞.
   Real integral = π√2/2, so **∫₀^∞ = (π√2/2)/2 = π/(2√2)** ✓

b) **Generalized Liouville Theorem:**
   By Cauchy's formula: f^{(n+1)}(z₀) = (n+1)!/(2πi) ∮_{|z-z₀|=R} f(z)/(z-z₀)^{n+2} dz
   
   |f^{(n+1)}(z₀)| ≤ (n+1)!/(2π) · (M(R + |z₀|)^n)/R^{n+2} · 2πR
                   = (n+1)! M (R + |z₀|)^n / R^{n+1}
   
   As R → ∞, this → 0.
   
   So f^{(n+1)} ≡ 0, meaning f is polynomial of degree ≤ n. □

---

## 📊 Self-Assessment Rubric

Rate yourself 1-5 on each topic:

| Topic | Score | Notes |
|-------|-------|-------|
| Complex arithmetic (polar, roots) | /5 | |
| Cauchy-Riemann equations | /5 | |
| Testing analyticity | /5 | |
| Complex exponential and log | /5 | |
| Multi-valued functions | /5 | |
| Contour parametrization | /5 | |
| Cauchy's theorem application | /5 | |
| Cauchy's integral formula | /5 | |
| QM connections | /5 | |

**Total: /45**

- 40-45: Excellent! Ready for Week 20
- 35-39: Good foundation, review weak areas
- 30-34: Need more practice before proceeding
- Below 30: Consider additional review time

---

## 🔮 Week 20 Preview: Complex Analysis II

### Topics Coming Up:
1. **Residue Theorem**: The general tool for evaluating contour integrals
2. **Laurent Series**: Expansion around singularities
3. **Classification of Singularities**: Removable, poles, essential
4. **Real Integral Evaluation**: Systematic methods
5. **Argument Principle & Rouché's Theorem**
6. **Applications to Physics**: Scattering, dispersion, propagators

### Preparation:
- Review partial fractions
- Practice identifying poles and their orders
- Think about how poles in Green's functions relate to eigenvalues

---

## ✅ Week 19 Completion Checklist

- [ ] Can fluently work with complex numbers in all forms
- [ ] Understand and apply Cauchy-Riemann equations
- [ ] Can identify analytic vs non-analytic functions
- [ ] Master elementary complex functions
- [ ] Handle multi-valued functions and branch cuts
- [ ] Compute contour integrals
- [ ] Apply Cauchy's theorem and integral formula
- [ ] Connect complex analysis to quantum mechanics
- [ ] Completed all problem sets
- [ ] Self-assessment score ≥ 35/45

---

## 📚 Resources for Further Study

### Books:
- Needham, "Visual Complex Analysis" — Geometric intuition
- Ahlfors, "Complex Analysis" — Rigorous treatment
- Arfken & Weber, Chapter 11 — Physics applications

### Online:
- 3Blue1Brown complex analysis videos
- MIT OCW 18.04 Complex Variables

---

## 🎉 Congratulations!

You've completed **Week 19: Complex Analysis I**!

This week established the foundation for one of the most powerful mathematical tools in physics. The results may seem abstract, but they're directly applicable to:

- Quantum scattering theory
- Response functions and causality
- Analytic continuation (imaginary time, Wick rotation)
- Spectral theory and Green's functions

**Next: Week 20 — Complex Analysis II (Residue Theorem & Applications)**
