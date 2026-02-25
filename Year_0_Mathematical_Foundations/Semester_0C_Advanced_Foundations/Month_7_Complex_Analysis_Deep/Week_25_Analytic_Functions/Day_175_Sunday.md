# Day 175: Week 25 Review — Analytic Functions and Foundations

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 11:00 AM | 2 hours | Concept Review & Synthesis |
| Late Morning | 11:00 AM - 12:30 PM | 1.5 hours | Problem Set A |
| Afternoon | 2:00 PM - 4:00 PM | 2 hours | Problem Set B |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Self-Assessment & Preview |

**Total Study Time: 6.5 hours**

---

## Week 25 Concept Map

```
                    COMPLEX ANALYSIS FOUNDATIONS
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   FUNCTIONS              ANALYTICITY           APPLICATIONS
        │                     │                     │
   ┌────┴────┐          ┌─────┴─────┐         ┌────┴────┐
   │         │          │           │         │         │
Complex   Multi-     Cauchy-    Harmonic   Conformal  Quantum
Numbers   valued     Riemann    Functions  Mappings   Mechanics
   │      Functions     │           │         │         │
   ├→ Polar form        │        ┌──┴──┐      │     Green's
   ├→ Euler's formula   │       Max   Mean    │     functions
   └→ Conjugate     ∂u/∂x=∂v/∂y  Principle Value  Möbius   │
                    ∂u/∂y=-∂v/∂x    │         │    Disk↔HP  Wave
                        │           │         │      │   functions
                    ∇²u=∇²v=0   Poisson   Schwarz   │
                        │       Formula   Christoffel│
                        └─────────┴─────────┴────────┘
                              UNIFIED BY
                         LAPLACE EQUATION
```

---

## Key Formulas Summary

### Complex Numbers (Day 169)

| Formula | Description |
|---------|-------------|
| $z = re^{i\theta}$ | Polar form |
| $e^{i\theta} = \cos\theta + i\sin\theta$ | Euler's formula |
| $\ln z = \ln\|z\| + i(\arg z + 2\pi k)$ | Multi-valued logarithm |
| $\|\psi_1 + \psi_2\|^2 = \|\psi_1\|^2 + \|\psi_2\|^2 + 2\text{Re}(\psi_1^*\psi_2)$ | Quantum interference |

### Analyticity (Days 170-171)

| Formula | Description |
|---------|-------------|
| $f'(z) = \lim_{h\to 0} \frac{f(z+h)-f(z)}{h}$ | Complex derivative |
| $\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}$, $\frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$ | Cauchy-Riemann |
| $\nabla^2 u = \nabla^2 v = 0$ | Harmonic property |
| $G(E) = (E - H + i\varepsilon)^{-1}$ | Green's function |

### Harmonic Functions (Day 172)

| Formula | Description |
|---------|-------------|
| $\nabla^2 u = 0$ | Laplace equation |
| $u(z_0) = \frac{1}{2\pi}\int_0^{2\pi} u(z_0 + re^{i\theta})d\theta$ | Mean value |
| $P_r(\psi) = \frac{1-r^2}{1-2r\cos\psi+r^2}$ | Poisson kernel |
| $\max_{\bar{D}} u = \max_{\partial D} u$ | Maximum principle |

### Conformal Mappings (Day 173)

| Formula | Description |
|---------|-------------|
| $T(z) = \frac{az+b}{cz+d}$ | Möbius transformation |
| $w = i\frac{1+z}{1-z}$ | Disk → half-plane |
| $w = \frac{z-a}{1-\bar{a}z}$ | Disk automorphism |
| $w = z + \frac{1}{z}$ | Joukowsky transform |

---

## Problem Set A: Fundamentals

### Problem A1: Complex Arithmetic

Compute in both algebraic and polar forms:
$$z = \frac{(2 + 2i)^4}{(1 - i\sqrt{3})^3}$$

### Problem A2: Analyticity Verification

For $f(z) = \frac{z}{z^2 + 1}$:

a) Find the singularities and classify them
b) Identify where $f$ is analytic
c) Compute $f'(z)$

### Problem A3: Cauchy-Riemann Application

Given $u(x,y) = e^x(x\cos y - y\sin y)$:

a) Verify that $u$ is harmonic
b) Find the harmonic conjugate $v$
c) Identify the corresponding analytic function $f(z)$

### Problem A4: Maximum Principle

Let $u$ be harmonic on the disk $\|z\| < 2$ with $u = \sin\theta + \cos 2\theta$ on the boundary.

a) What is the maximum value of $u$ on the closed disk?
b) Where on the boundary does this maximum occur?
c) Can $u$ have a local maximum inside the disk?

### Problem A5: Conformal Mapping

Find the Möbius transformation that maps:
- $0 \to i$
- $\infty \to 1$
- $-1 \to 0$

---

## Problem Set B: Applications

### Problem B1: Branch Cuts

For $f(z) = \sqrt{z^2 - 1}$:

a) Find all branch points
b) Draw a suitable branch cut configuration
c) Specify the principal branch

### Problem B2: Poisson Formula

Use the Poisson integral formula to solve:
- Domain: Unit disk
- Boundary condition: $u(1, \theta) = 1$ for $0 \leq \theta < \pi$, $u(1, \theta) = 0$ for $\pi \leq \theta < 2\pi$

Find $u$ at the center $z = 0$.

### Problem B3: Conformal Transformation

The mapping $w = z^2$ transforms the first quadrant $\{x > 0, y > 0\}$ to what region? Describe the image of:
- The positive real axis
- The positive imaginary axis
- The ray $\arg z = \pi/4$

### Problem B4: Quantum Connection

The Green's function for a 1D particle is:

$$G(x, x'; E) = \frac{m}{i\hbar^2 k}e^{ik|x-x'|}$$

where $k = \sqrt{2mE}/\hbar$.

a) For what values of $E$ is this expression well-defined?
b) How does the $+i\varepsilon$ prescription appear when $E$ is near a real eigenvalue?
c) Relate this to the analyticity structure we studied

### Problem B5: Harmonic Conjugates and Physics

In 2D electrostatics, the electric potential $\phi(x,y)$ satisfies $\nabla^2\phi = 0$ (in charge-free regions).

a) If the potential between two parallel plates (at $y = 0$ and $y = 1$) is $\phi = y$, verify this is harmonic
b) Find the electric field $\mathbf{E} = -\nabla\phi$
c) Identify the harmonic conjugate (stream function) and explain its physical meaning

---

## Solutions to Selected Problems

### Solution A1

$2 + 2i = 2\sqrt{2}e^{i\pi/4}$, so $(2 + 2i)^4 = (2\sqrt{2})^4 e^{i\pi} = 64 \cdot (-1) = -64$

$1 - i\sqrt{3} = 2e^{-i\pi/3}$, so $(1 - i\sqrt{3})^3 = 8e^{-i\pi} = -8$

Therefore: $z = \frac{-64}{-8} = 8$

### Solution A3

$\nabla^2 u = e^x[(1)\cos y - y(-\sin y)] + e^x[x(-\cos y) - (\sin y + y\cos y)]$

$= e^x[\cos y + y\sin y - x\cos y - \sin y - y\cos y]$

After careful calculation: $\nabla^2 u = 0$ ✓

Using Cauchy-Riemann and integrating: $v = e^x(x\sin y + y\cos y)$

The analytic function is $f(z) = ze^z$.

### Solution B2

By the mean value property (or direct Poisson calculation):

$$u(0) = \frac{1}{2\pi}\int_0^{2\pi} u(1,\theta)d\theta = \frac{1}{2\pi}\left[\int_0^{\pi} 1 \, d\theta + \int_\pi^{2\pi} 0 \, d\theta\right] = \frac{\pi}{2\pi} = \frac{1}{2}$$

---

## Self-Assessment Checklist

### Conceptual Understanding

- [ ] I can explain why complex differentiability is stronger than real differentiability
- [ ] I understand the physical meaning of harmonic functions
- [ ] I can describe what conformal mappings preserve
- [ ] I can connect Green's functions to analyticity

### Computational Skills

- [ ] I can verify the Cauchy-Riemann equations
- [ ] I can find harmonic conjugates
- [ ] I can classify singularities
- [ ] I can construct Möbius transformations for given conditions
- [ ] I can apply the maximum principle

### Applications

- [ ] I understand how conformal maps help solve PDEs
- [ ] I can visualize complex functions
- [ ] I see the quantum mechanics connections

---

## Week 25 Key Insights

### 1. The Power of Analyticity

Complex differentiability is remarkably strong:
- Once differentiable → infinitely differentiable
- Real and imaginary parts are coupled (Cauchy-Riemann)
- Both parts are harmonic

### 2. Harmonic Functions Everywhere

Laplace's equation $\nabla^2 u = 0$ appears in:
- Electrostatics (potential)
- Heat conduction (steady state)
- Fluid dynamics (potential flow)
- Quantum mechanics (certain regions)

### 3. Geometry and Analysis United

Conformal mappings connect:
- Complex analysis (analyticity)
- Geometry (angle preservation)
- Physics (boundary value problems)

### 4. Quantum Connections

- Wave functions are complex-valued
- Green's functions have specific analyticity properties
- Causality ↔ Upper half-plane analyticity

---

## Preview: Week 26 — Contour Integration

Next week we study the crown jewel of complex analysis:

**Cauchy's Integral Theorem and Formula:**

$$\oint_C f(z) dz = 0 \quad \text{(for analytic } f \text{)}$$

$$f(z_0) = \frac{1}{2\pi i}\oint_C \frac{f(z)}{z - z_0} dz$$

**Topics:**
- Day 176: Line integrals in ℂ
- Day 177: Cauchy's integral theorem
- Day 178: Cauchy's integral formula
- Day 179: Applications to real integrals
- Day 180: Computational lab
- Day 181: Week review

This will enable us to:
- Evaluate difficult real integrals
- Prove all derivatives exist (Taylor series)
- Understand residues (Week 28)
- Compute propagators in quantum mechanics

---

## Spaced Repetition: Key Facts

Review these at increasing intervals (1 day, 3 days, 1 week, 2 weeks):

1. **Euler's Formula:** $e^{i\theta} = \cos\theta + i\sin\theta$

2. **Cauchy-Riemann:** $u_x = v_y$, $u_y = -v_x$

3. **Analyticity implies harmonicity:** $\nabla^2 u = \nabla^2 v = 0$

4. **Maximum principle:** Harmonic functions have extrema on boundaries only

5. **Conformal:** Analytic with $f' \neq 0$ preserves angles

6. **Möbius:** Maps circles/lines to circles/lines

7. **Green's function:** $G = (E - H + i\varepsilon)^{-1}$ analytic in upper half-plane

---

## Reflection Questions

1. Why is complex analysis more powerful than real analysis?

2. How do the Cauchy-Riemann equations encode "two-dimensional information" in a complex derivative?

3. What physical intuition explains the maximum principle?

4. Why is conformal mapping useful for solving PDEs?

5. How does the $+i\varepsilon$ prescription ensure causality in quantum mechanics?

---

*"Mathematics is the art of giving the same name to different things."*
— Henri Poincaré

---

## Week 25 Complete! 🎉

You have now mastered the foundations of complex analysis:
- Complex functions and multi-valued behavior
- Analyticity and the Cauchy-Riemann equations
- Harmonic functions and their properties
- Conformal mappings and their applications

**Next:** Week 26 — Contour Integration (the most powerful tool in complex analysis!)
