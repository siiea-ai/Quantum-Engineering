# Day 140: Week 20 Review — Complex Analysis II Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Concept Review & Problem Set A |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Set B & Self-Assessment |
| Evening | 6:00 PM - 7:00 PM | 1 hour | Month 6 Preview & Planning |

**Total Study Time: 7 hours**

---

## 🎯 Week 20 Learning Objectives — Final Check

By the end of this review, confirm mastery of:

- [ ] Laurent series expansions
- [ ] Classification of singularities (removable, poles, essential)
- [ ] Residue computation methods
- [ ] The Residue Theorem
- [ ] Evaluation of real integrals via residues
- [ ] Argument Principle and winding numbers
- [ ] Rouché's Theorem and zero counting
- [ ] Physics applications (Green's functions, scattering)

---

## 📊 Week 20 Concept Map

```
                    COMPLEX ANALYSIS II
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   SINGULARITIES      RESIDUES           APPLICATIONS
        │                  │                  │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │         │       │         │       │         │
Laurent  Classification Residue    Real      Argument
Series        │       Theorem   Integrals  Principle
   │     ┌────┴────┐     │         │         │
   │     │    │    │     │    ┌────┴────┐    │
Principal Remov Poles Ess  ∮=2πi  Rational  Winding
Part     │         │   ΣRes    Trig    Numbers
   │     │         │     │    Fourier    │
   │   f bounded  |f|→∞  Wild    │     Rouché
   │     │         │   behavior  │    Theorem
Residue  Res=0   Res=   Infinite │      │
= a₋₁         formula  series   │   Zero
                              Jordan  Counting
                              Lemma
```

---

## 📝 Key Formulas Summary

### Laurent Series
$$f(z) = \sum_{n=-\infty}^{\infty} a_n(z-z_0)^n, \quad a_n = \frac{1}{2\pi i}\oint_C \frac{f(z)}{(z-z_0)^{n+1}}dz$$

### Singularity Classification
| Type | Principal Part | Behavior |
|------|---------------|----------|
| Removable | None | f bounded |
| Pole (order m) | Finite (m terms) | \|f\| → ∞ |
| Essential | Infinite | Casorati-Weierstrass |

### Residue Formulas
| Case | Formula |
|------|---------|
| Simple pole | Res = lim_{z→z₀} (z-z₀)f(z) |
| Quotient g/h | Res = g(z₀)/h'(z₀) |
| Pole order m | Res = (1/(m-1)!) lim d^{m-1}/dz^{m-1}[(z-z₀)^m f(z)] |

### Residue Theorem
$$\oint_C f(z)\,dz = 2\pi i \sum_{\text{inside}} \text{Res}[f, z_k]$$

### Real Integrals
| Type | Method |
|------|--------|
| ∫ p(x)/q(x) dx | Semicircle, upper poles |
| ∫ R(cos,sin) dθ | Unit circle, z = e^{iθ} |
| ∫ f(x)e^{iax} dx | Jordan's lemma |

### Argument Principle
$$\frac{1}{2\pi i}\oint_C \frac{f'(z)}{f(z)}dz = Z - P$$

### Rouché's Theorem
If \|g(z)\| < \|f(z)\| on C, then f and f+g have the same number of zeros inside C.

---

## 🔬 Problem Set A: Laurent Series & Singularities

### Problem A1: Laurent Series
Find the Laurent series for f(z) = 1/((z-1)(z-2)) in:
a) |z| < 1
b) 1 < |z| < 2
c) |z| > 2

**Solutions:**
a) Both poles outside, Taylor series:
$$f(z) = \frac{1}{(z-1)(z-2)} = \frac{-1}{z-1} + \frac{1}{z-2} = \frac{1}{1-z} - \frac{1}{2(1-z/2)}$$
$$= \sum_{n=0}^{\infty} z^n - \frac{1}{2}\sum_{n=0}^{\infty}\frac{z^n}{2^n} = \sum_{n=0}^{\infty}\left(1 - \frac{1}{2^{n+1}}\right)z^n$$

b) Pole at z=1 inside, pole at z=2 outside:
$$\frac{1}{z-1} = \frac{1}{z(1-1/z)} = \frac{1}{z}\sum_{n=0}^{\infty}\frac{1}{z^n}$$
$$f(z) = -\sum_{n=0}^{\infty}z^{-n-1} - \frac{1}{2}\sum_{n=0}^{\infty}\frac{z^n}{2^n}$$

c) Both poles inside relative to ∞:
$$f(z) = -\sum_{n=0}^{\infty}z^{-n-1} + \sum_{n=0}^{\infty}\frac{2^n}{z^{n+1}}$$

---

### Problem A2: Classify Singularities
Classify the singularity at z = 0:
a) f(z) = sin z/z
b) f(z) = (1 - cos z)/z²
c) f(z) = e^{1/z²}
d) f(z) = z²/(e^z - 1)

**Solutions:**
a) sin z/z = 1 - z²/6 + ... → **Removable** (define f(0) = 1)

b) (1 - cos z)/z² = (z²/2 - z⁴/24 + ...)/z² = 1/2 - z²/24 + ... → **Removable** (f(0) = 1/2)

c) e^{1/z²} = 1 + 1/z² + 1/(2z⁴) + ... → **Essential** (infinitely many negative powers)

d) e^z - 1 = z + z²/2 + ..., so z²/(e^z-1) = z/(1 + z/2 + ...) = z - z²/2 + ... → **Removable** at z=0

---

### Problem A3: Residue Computation
Find all residues of f(z) = z/((z²+1)(z-2)):

**Solution:**
Poles at z = i, -i, 2 (all simple).

At z = i: Res = i/((2i)(i-2)) = i/(2i(i-2)) = 1/(2(i-2)) = (i+2)/(2(-5)) = **(i+2)/(-10)**

At z = -i: Res = -i/((-2i)(-i-2)) = -i/(2i(i+2)) = -1/(2(i+2)) = **(-i+2)/(-10)**

At z = 2: Res = 2/((4+1)(1)) = **2/5**

Check: Sum = (i+2-i+2)/(-10) + 2/5 = 4/(-10) + 4/10 = 0 ✓

---

### Problem A4: Essential Singularity
Show that e^{1/z} takes every nonzero value infinitely often in any neighborhood of z = 0.

**Solution:**
For any w ≠ 0, solve e^{1/z} = w:
- 1/z = log w + 2πin for n ∈ ℤ
- z = 1/(log w + 2πin)

For large |n|, these z values become arbitrarily small (close to 0).
There are infinitely many solutions for each w ≠ 0. ✓

The exception w = 0 is never achieved since e^{1/z} ≠ 0 for all z ≠ 0.

---

## 🔬 Problem Set B: Residue Theorem & Applications

### Problem B1: Contour Integrals
Evaluate:
a) ∮_{|z|=3} dz/(z²-1)
b) ∮_{|z|=2} z·e^z/(z-1)² dz
c) ∮_{|z|=1} e^{1/z} dz

**Solutions:**
a) Poles at z = ±1, both inside |z| = 3.
Res at z=1: 1/(2·1) = 1/2
Res at z=-1: 1/(2·(-1)) = -1/2
∮ = 2πi(1/2 - 1/2) = **0**

b) Double pole at z = 1 inside |z| = 2.
Res = d/dz[z·e^z]|_{z=1} = (e^z + z·e^z)|_{z=1} = 2e
∮ = 2πi(2e) = **4πie**

c) Essential singularity at z = 0.
e^{1/z} = 1 + 1/z + 1/(2z²) + ...
Res = coefficient of 1/z = 1
∮ = 2πi(1) = **2πi**

---

### Problem B2: Real Integrals
Evaluate:
a) ∫_{-∞}^{∞} dx/(x²+4)²
b) ∫_0^{2π} dθ/(2+cos θ)
c) ∫_{-∞}^{∞} x·sin x/(x²+1) dx

**Solutions:**
a) f(z) = 1/(z²+4)² has double poles at z = ±2i.
Only z = 2i in upper half-plane.
Res = d/dz[(z-2i)²/(z²+4)²]|_{z=2i} = d/dz[1/(z+2i)²]|_{z=2i} = -2/(4i)³ = -2/(-64i) = 1/(32i)
∫ = 2πi · 1/(32i) = **π/16**

b) z = e^{iθ}, cos θ = (z+1/z)/2
1/(2+cos θ) = 2z/(z² + 4z + 1)
Poles: z = -2 ± √3, only z = -2+√3 inside |z|=1
Res = 2(-2+√3)/(2(-2+√3)+4) = (−2+√3)/(√3) = 1 - 2/√3
∫ = (2/i)·2πi·Res = **2π/√3**

c) Consider ∫ z·e^{iz}/(z²+1) dz.
Pole at z = i in upper half-plane.
Res = i·e^{-1}/(2i) = e^{-1}/2
∫_{-∞}^{∞} x·e^{ix}/(x²+1) dx = 2πi · e^{-1}/2 = πi/e
Taking imaginary part: **π/e**

---

### Problem B3: Argument Principle
a) How many zeros does z⁵ + 3z + 1 have in |z| < 1?
b) Prove z⁴ - 5z + 1 has exactly 3 zeros in 1 < |z| < 2.

**Solutions:**
a) Use Rouché: f(z) = 3z, g(z) = z⁵ + 1
On |z| = 1: |f| = 3, |g| ≤ 2 < 3
f has 1 zero in |z| < 1, so z⁵ + 3z + 1 has **1 zero**.

b) Count in |z| < 1: f = -5z, g = z⁴ + 1, |f| = 5, |g| ≤ 2 → **1 zero**
Count in |z| < 2: f = z⁴, g = -5z + 1, |f| = 16, |g| ≤ 11 → **4 zeros**
Therefore in 1 < |z| < 2: 4 - 1 = **3 zeros**

---

### Problem B4: Physics Application
The scattering amplitude is A(E) = 1/(E - E₀ + iΓ/2) with E₀ = 1, Γ = 0.2.
a) Where is the pole?
b) Compute the cross section σ ∝ |A|² at E = E₀.
c) What is the full width at half maximum?

**Solutions:**
a) Pole at E = E₀ - iΓ/2 = **1 - 0.1i**

b) |A(E₀)|² = |1/(iΓ/2)|² = 4/Γ² = 4/0.04 = **100**

c) |A(E)|² = 1/((E-E₀)² + Γ²/4)
Half max when (E-E₀)² = Γ²/4
E - E₀ = ±Γ/2
FWHM = Γ = **0.2**

---

## 📊 Self-Assessment Rubric

Rate yourself 1-5 on each topic:

| Topic | Score | Notes |
|-------|-------|-------|
| Laurent series computation | /5 | |
| Singularity classification | /5 | |
| Residue calculations | /5 | |
| Residue theorem application | /5 | |
| Real integral evaluation | /5 | |
| Argument principle | /5 | |
| Rouché's theorem | /5 | |
| Physics connections | /5 | |

**Total: /40**

- 35-40: Excellent! Ready for Classical Mechanics
- 30-34: Good foundation, review weak areas
- 25-29: Need more practice
- Below 25: Consider additional review

---

## 🎉 Complex Analysis Complete!

You've completed **Weeks 19-20: Complex Analysis I & II**!

### What You've Mastered:
- Complex numbers and the complex plane
- Analyticity and Cauchy-Riemann equations
- Elementary complex functions
- Contour integration and Cauchy's theorem
- Laurent series and singularities
- The Residue Theorem
- Real integral evaluation
- Argument Principle and Rouché's theorem

### Applications in Quantum Mechanics:
- Green's functions and propagators
- Scattering theory (poles = bound states/resonances)
- Kramers-Kronig relations (causality)
- Path integrals and analytic continuation
- Spectral theory

---

## 🔮 Month 6 Preview: Classical Mechanics

**Next:** Weeks 21-24 cover **Classical Mechanics** — the foundation for quantum mechanics!

### Week 21: Lagrangian Mechanics I
- Generalized coordinates
- Principle of least action
- Euler-Lagrange equations
- Constraints and Lagrange multipliers

### Week 22: Lagrangian Mechanics II
- Symmetries and conservation laws
- Noether's theorem
- Central force problems
- Small oscillations

### Week 23: Hamiltonian Mechanics I
- Legendre transformation
- Hamilton's equations
- Phase space
- Poisson brackets

### Week 24: Hamiltonian Mechanics II
- Canonical transformations
- Hamilton-Jacobi equation
- Action-angle variables
- Connection to quantum mechanics

---

## ✅ Week 20 Completion Checklist

- [ ] Master Laurent series in different annuli
- [ ] Classify all singularity types
- [ ] Compute residues by multiple methods
- [ ] Apply Residue Theorem confidently
- [ ] Evaluate real integrals via contours
- [ ] Use Argument Principle and Rouché
- [ ] Connect to physics applications
- [ ] Self-assessment score ≥ 30/40

---

*Congratulations on completing Complex Analysis!*
*Ready for the beautiful structure of Classical Mechanics!*
