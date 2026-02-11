# Day 182: Week 26 Review — Contour Integration Mastery

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 11:00 AM | 2 hours | Concept Review & Synthesis |
| Late Morning | 11:00 AM - 12:30 PM | 1.5 hours | Problem Set A |
| Afternoon | 2:00 PM - 4:00 PM | 2 hours | Problem Set B |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Self-Assessment & Preview |

**Total Study Time: 6.5 hours**

---

## Week 26 Concept Map

```
                        CONTOUR INTEGRATION
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   FOUNDATIONS           TECHNIQUES             APPLICATIONS
        │                     │                     │
   ┌────┴────┐          ┌─────┴─────┐         ┌────┴────┐
   │         │          │           │         │         │
 Complex   Cauchy's   Residues   Contour    Real     Physics
  Line     Theorem    (preview)   Types    Integrals   QM
Integrals     │           │         │         │         │
   │          │           │         │         │         │
 ∫f(z)dz   ∮f(z)dz=0   Simple   Semicircle  Rational  Green's
   │       (analytic)   Poles      │        Functions  Functions
   │          │           │     Keyhole     │         │
 Path      Cauchy's    Residue    │      Fourier   Propagators
 Dependence Formula    Theorem  Indented   │         │
   │          │           │        │    Trigono-  Kramers-
   └──────────┴───────────┴────────┴───metric───Kronig
                    UNIFIED BY
              THE RESIDUE THEOREM
```

---

## Key Formulas Summary

### Line Integrals (Day 176)

| Formula | Description |
|---------|-------------|
| $\int_C f(z) \, dz = \int_a^b f(z(t)) z'(t) \, dt$ | Definition via parametrization |
| $\left\|\int_C f(z) \, dz\right\| \leq ML$ | ML inequality |
| $\oint_{\|z-z_0\|=r} (z-z_0)^n \, dz = \begin{cases} 2\pi i & n=-1 \\ 0 & n \neq -1 \end{cases}$ | Fundamental integral |

### Cauchy's Theorem (Day 177)

| Formula | Description |
|---------|-------------|
| $\oint_C f(z) \, dz = 0$ | Analytic $f$, simply connected |
| $n(C, z_0) = \frac{1}{2\pi i}\oint_C \frac{dz}{z-z_0}$ | Winding number |
| Deformation: $\oint_{C_1} = \oint_{C_2}$ | If no singularities crossed |

### Cauchy's Integral Formula (Day 178)

| Formula | Description |
|---------|-------------|
| $f(z_0) = \frac{1}{2\pi i}\oint_C \frac{f(z)}{z-z_0} dz$ | Value from boundary |
| $f^{(n)}(z_0) = \frac{n!}{2\pi i}\oint_C \frac{f(z)}{(z-z_0)^{n+1}} dz$ | Derivatives |
| $\|f^{(n)}(z_0)\| \leq \frac{n!M}{R^n}$ | Cauchy's inequality |
| Bounded entire → constant | Liouville's theorem |

### Real Integrals (Days 179-180)

| Formula | Description |
|---------|-------------|
| $\int_{-\infty}^\infty \frac{P(x)}{Q(x)} dx = 2\pi i \sum_{\text{UHP}} \text{Res}$ | Semicircular contour |
| $\int_{-\infty}^\infty f(x)e^{iax} dx$ (a>0) | Close in UHP |
| $\int_0^\infty \frac{x^{\alpha-1}}{1+x} dx = \frac{\pi}{\sin(\pi\alpha)}$ | Keyhole contour |
| $\int_0^{2\pi} \frac{d\theta}{a+b\cos\theta} = \frac{2\pi}{\sqrt{a^2-b^2}}$ | $z = e^{i\theta}$ |
| $\int_0^\infty \frac{\sin x}{x} dx = \frac{\pi}{2}$ | Dirichlet integral |

---

## Problem Set A: Fundamental Techniques

### Problem A1: Line Integral Computation

Compute $\int_C z^2 \bar{z} \, dz$ where $C$ is the circle $|z| = 2$ traversed counterclockwise.

### Problem A2: Cauchy's Theorem Application

Show that $\oint_C \frac{z^2 + z + 1}{z^3 - 1} \, dz = 0$ if $C$ is any contour in the region $|z| > 2$.

### Problem A3: Cauchy's Integral Formula

Compute:
(a) $\oint_{|z|=2} \frac{\sin z}{z^3} \, dz$
(b) $\oint_{|z|=1} \frac{e^{z^2}}{z^4} \, dz$

### Problem A4: Winding Numbers

Find the winding number of the curve $z(t) = 2\cos t + i\sin t$ ($0 \leq t \leq 2\pi$) about:
(a) $z_0 = 0$
(b) $z_0 = 3$
(c) $z_0 = i$

### Problem A5: Maximum Modulus

Find the maximum of $|e^z + z|$ on the closed disk $|z| \leq 1$.

---

## Problem Set B: Applications

### Problem B1: Rational Function Integral

Evaluate $\int_{-\infty}^{\infty} \frac{x^2}{(x^2+1)(x^2+4)} \, dx$.

### Problem B2: Fourier-Type Integral

Compute $\int_{-\infty}^{\infty} \frac{x \sin(2x)}{x^2 + 1} \, dx$.

### Problem B3: Trigonometric Integral

Evaluate $\int_0^{2\pi} \frac{\cos^2\theta}{5 - 4\cos\theta} \, d\theta$.

### Problem B4: Keyhole Contour

Using a keyhole contour, show that:
$$\int_0^\infty \frac{x^{-1/2}}{1 + x} \, dx = \pi$$

### Problem B5: Dirichlet Variation

Prove that $\int_0^\infty \frac{\sin^2(ax)}{x^2} \, dx = \frac{\pi|a|}{2}$ for real $a$.

### Problem B6: Physics Application

The retarded Green's function for a harmonic oscillator is:
$$G(E) = \frac{1}{E^2 - \omega_0^2 + i\varepsilon}$$

(a) Find the poles and their residues.
(b) Compute $\frac{1}{2\pi}\int_{-\infty}^{\infty} G(E) e^{-iEt/\hbar} \, dE$ for $t > 0$.
(c) Interpret the result physically.

---

## Solutions to Selected Problems

### Solution A1

On $|z| = 2$: $z = 2e^{i\theta}$, $\bar{z} = 2e^{-i\theta}$, $dz = 2ie^{i\theta}d\theta$

$$z^2\bar{z} = 4e^{2i\theta} \cdot 2e^{-i\theta} = 8e^{i\theta}$$

$$\int_C z^2\bar{z} \, dz = \int_0^{2\pi} 8e^{i\theta} \cdot 2ie^{i\theta} \, d\theta = 16i \int_0^{2\pi} e^{2i\theta} \, d\theta$$

$$= 16i \left[\frac{e^{2i\theta}}{2i}\right]_0^{2\pi} = 8(e^{4\pi i} - 1) = 8(1 - 1) = 0$$

### Solution A3

**(a)** Using $f^{(n)}(0) = \frac{n!}{2\pi i}\oint \frac{f(z)}{z^{n+1}} dz$:

$$\oint_{|z|=2} \frac{\sin z}{z^3} dz = \frac{2\pi i}{2!} f''(0)$$

where $f(z) = \sin z$, so $f''(z) = -\sin z$, $f''(0) = 0$.

Answer: $\boxed{0}$

**(b)** Similarly with $f(z) = e^{z^2}$:
$$f'(z) = 2ze^{z^2}, \quad f''(z) = 2e^{z^2} + 4z^2e^{z^2}, \quad f'''(z) = 12ze^{z^2} + 8z^3e^{z^2}$$

$f'''(0) = 0$

$$\oint_{|z|=1} \frac{e^{z^2}}{z^4} dz = \frac{2\pi i}{3!} f'''(0) = 0$$

Answer: $\boxed{0}$

### Solution B1

$$\frac{x^2}{(x^2+1)(x^2+4)} = \frac{A}{x^2+1} + \frac{B}{x^2+4}$$

Multiply out: $x^2 = A(x^2+4) + B(x^2+1)$

Setting $x^2 = -1$: $-1 = 3A$, so $A = -1/3$
Setting $x^2 = -4$: $-4 = -3B$, so $B = 4/3$

$$\frac{x^2}{(x^2+1)(x^2+4)} = -\frac{1/3}{x^2+1} + \frac{4/3}{x^2+4}$$

$$\int_{-\infty}^\infty = -\frac{1}{3}\pi + \frac{4}{3} \cdot \frac{\pi}{2} = -\frac{\pi}{3} + \frac{2\pi}{3} = \boxed{\frac{\pi}{3}}$$

### Solution B2

Consider $\int_{-\infty}^\infty \frac{x e^{2ix}}{x^2+1} dx$.

Pole at $z = i$ in UHP. Near $z = i$:
$$\frac{ze^{2iz}}{(z-i)(z+i)} \to \frac{ie^{-2}}{2i} = \frac{e^{-2}}{2}$$

$$\int_{-\infty}^\infty \frac{xe^{2ix}}{x^2+1} dx = 2\pi i \cdot \frac{e^{-2}}{2} = \pi i e^{-2}$$

Taking the imaginary part:
$$\int_{-\infty}^\infty \frac{x\sin(2x)}{x^2+1} dx = \text{Im}(\pi i e^{-2}) = \boxed{\frac{\pi}{e^2}}$$

### Solution B4

Let $f(z) = \frac{z^{-1/2}}{1+z}$ with branch cut on positive real axis.

**Keyhole contour:**
- Upper edge: $z = x$, $z^{-1/2} = x^{-1/2}$
- Lower edge: $z = xe^{2\pi i}$, $z^{-1/2} = x^{-1/2}e^{-\pi i} = -x^{-1/2}$

**Residue at $z = -1 = e^{i\pi}$:**
$$\text{Res} = (-1)^{-1/2} = e^{-i\pi/2} = -i$$

**Contour integral:**
$$\oint = \int_0^\infty \frac{x^{-1/2}}{1+x}dx - \int_0^\infty \frac{-x^{-1/2}}{1+x}dx = 2\int_0^\infty \frac{x^{-1/2}}{1+x}dx$$

$$= 2\pi i \cdot (-i) = 2\pi$$

Therefore:
$$\int_0^\infty \frac{x^{-1/2}}{1+x}dx = \pi \checkmark$$

### Solution B6

**(a)** $G(E) = \frac{1}{E^2 - \omega_0^2 + i\varepsilon} = \frac{1}{(E-\omega_0+i\varepsilon')(E+\omega_0+i\varepsilon')}$

Poles at $E = \pm\omega_0 - i\varepsilon'$ (just below real axis)

For retarded Green's function (poles in lower half-plane):
- Pole at $E_+ = \omega_0 - i\varepsilon'$ with residue $\frac{1}{2\omega_0}$
- Pole at $E_- = -\omega_0 - i\varepsilon'$ with residue $\frac{1}{-2\omega_0}$

**(b)** For $t > 0$, close in lower half-plane:

$$\frac{1}{2\pi}\int G(E)e^{-iEt/\hbar}dE = -2\pi i \cdot \frac{1}{2\pi}\left[\frac{e^{-i\omega_0 t/\hbar}}{2\omega_0} - \frac{e^{i\omega_0 t/\hbar}}{2\omega_0}\right]$$

$$= -i\left[\frac{e^{-i\omega_0 t/\hbar} - e^{i\omega_0 t/\hbar}}{2\omega_0}\right] = \frac{\sin(\omega_0 t/\hbar)}{\omega_0}$$

**(c)** This is the **impulse response** of a harmonic oscillator — the response to a delta-function kick at $t = 0$. It oscillates at the natural frequency $\omega_0$.

---

## Self-Assessment Checklist

### Conceptual Understanding

- [ ] I understand why complex line integrals may be path-dependent
- [ ] I can explain Cauchy's theorem and its significance
- [ ] I understand simple connectivity and its role
- [ ] I can interpret the winding number geometrically
- [ ] I understand why analytic functions have path-independent integrals

### Computational Skills

- [ ] I can parametrize contours and compute integrals directly
- [ ] I can apply Cauchy's integral formula for values and derivatives
- [ ] I can identify which poles contribute to a contour integral
- [ ] I can evaluate real integrals using semicircular contours
- [ ] I can handle branch cuts with keyhole contours
- [ ] I can convert trigonometric integrals using $z = e^{i\theta}$

### Physics Applications

- [ ] I understand how Green's functions are computed
- [ ] I can apply Kramers-Kronig relations
- [ ] I see the connection between causality and analyticity

---

## Week 26 Key Insights

### 1. Contour Integration is Powerful

What seemed impossible in real analysis becomes tractable:
- $\int_{-\infty}^\infty \frac{dx}{1+x^4}$ — no elementary antiderivative, but equals $\frac{\pi}{\sqrt{2}}$
- $\int_0^\infty \frac{\sin x}{x} dx$ — not even absolutely convergent, but equals $\frac{\pi}{2}$

### 2. Topology Matters

The value of $\oint f(z)dz$ depends only on:
- The singularities enclosed
- The winding numbers around them

The exact shape of the contour is irrelevant!

### 3. Physics Applications are Direct

- Green's functions are computed via contour integrals
- Kramers-Kronig relations follow from Cauchy's theorem
- The $+i\varepsilon$ prescription encodes causality

### 4. Unified by Residues

All techniques reduce to finding and summing residues (Week 27 topic).

---

## Preview: Week 27 — Laurent Series and Residues

Next week we develop the **residue theorem** systematically:

**Laurent Series:**
$$f(z) = \sum_{n=-\infty}^{\infty} a_n (z - z_0)^n$$

**Residue:**
$$\text{Res}_{z=z_0} f(z) = a_{-1} = \frac{1}{2\pi i}\oint f(z) dz$$

**Residue Theorem:**
$$\oint_C f(z) dz = 2\pi i \sum_{k} \text{Res}_{z=z_k} f(z)$$

**Topics:**
- Day 183: Laurent series expansion
- Day 184: Classification of singularities
- Day 185: Residue computation techniques
- Day 186: Residue theorem proof
- Day 187: Advanced applications
- Day 188: Computational lab
- Day 189: Week review

This will complete our contour integration toolkit!

---

## Spaced Repetition: Key Facts

Review at increasing intervals (1 day, 3 days, 1 week, 2 weeks):

1. **Cauchy's Theorem:** $\oint_C f(z)dz = 0$ for analytic $f$ in simply connected domain

2. **Cauchy's Formula:** $f(z_0) = \frac{1}{2\pi i}\oint_C \frac{f(z)}{z-z_0}dz$

3. **Derivative Formula:** $f^{(n)}(z_0) = \frac{n!}{2\pi i}\oint_C \frac{f(z)}{(z-z_0)^{n+1}}dz$

4. **Key Integral:** $\oint \frac{dz}{z-z_0} = 2\pi i \cdot n$ where $n$ is winding number

5. **Liouville:** Bounded entire → constant

6. **Jordan's Lemma:** $\int_{\gamma_R} f(z)e^{iaz}dz \to 0$ for $a > 0$, $R \to \infty$

7. **Dirichlet:** $\int_0^\infty \frac{\sin x}{x}dx = \frac{\pi}{2}$

---

## Reflection Questions

1. Why does complex differentiability imply such strong properties (Cauchy's theorem, infinite differentiability)?

2. How does the residue at a pole "measure" the obstruction to Cauchy's theorem?

3. What is the physical interpretation of closing a contour in the upper vs. lower half-plane?

4. Why do branch cuts appear in physics (e.g., in propagators)?

5. How do Kramers-Kronig relations express causality mathematically?

---

## Computational Challenge

Write a program that:
1. Takes a rational function $P(x)/Q(x)$ as input
2. Automatically finds all poles
3. Computes residues at UHP poles
4. Returns $\int_{-\infty}^\infty P(x)/Q(x) dx$

Verify your program on the integrals from this week.

---

*"The theory of residues is perhaps the most delicate and at the same time the most useful tool in the whole field of analysis."*
— E.T. Whittaker

---

## Week 26 Complete! 🎉

You have now mastered contour integration:
- Complex line integrals and their properties
- Cauchy's theorem and integral formula
- Evaluation of real integrals via contours
- Branch cuts and keyhole contours
- Physics applications (Green's functions, dispersion)

**Next:** Week 27 — Laurent Series and the Residue Theorem
