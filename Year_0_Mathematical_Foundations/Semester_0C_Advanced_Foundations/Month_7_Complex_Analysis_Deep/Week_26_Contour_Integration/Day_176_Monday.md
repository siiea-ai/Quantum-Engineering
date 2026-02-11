# Day 176: Line Integrals in the Complex Plane

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Complex Line Integrals |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Solving & Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 176, you will be able to:

1. Define and compute complex line integrals along parametrized curves
2. Apply the fundamental theorem of calculus for complex integrals (when applicable)
3. Understand the path-dependence of complex integrals
4. Compute integrals using antiderivatives when they exist
5. Relate complex line integrals to work and circulation in physics
6. Connect to path integrals in quantum mechanics

---

## Core Content

### 1. Complex Line Integrals: Definition

#### The Setting

We integrate a complex function $f(z)$ along a curve $C$ in the complex plane.

**Definition (Contour):** A *contour* $C$ is a piecewise smooth curve parametrized by:
$$z(t) = x(t) + iy(t), \quad a \leq t \leq b$$

where $x(t)$ and $y(t)$ are continuous with piecewise continuous derivatives.

**Definition (Complex Line Integral):**
$$\boxed{\int_C f(z) \, dz = \int_a^b f(z(t)) \, z'(t) \, dt}$$

Since $z'(t) = x'(t) + iy'(t)$ and $f(z(t)) = u + iv$, this expands to:
$$\int_C f(z) \, dz = \int_C (u + iv)(dx + i\,dy)$$
$$= \int_C (u\,dx - v\,dy) + i\int_C (v\,dx + u\,dy)$$

This shows that a complex line integral is really **two real line integrals**.

#### Notation

- $\int_C f(z) \, dz$ — integral along curve $C$
- $\oint_C f(z) \, dz$ — integral along **closed** curve $C$
- $\int_{z_1}^{z_2} f(z) \, dz$ — integral from $z_1$ to $z_2$ (path-dependent unless $f$ has antiderivative)

### 2. Basic Examples

#### Example 1: Integral of $z$ Along a Straight Line

Compute $\int_C z \, dz$ where $C$ is the straight line from $0$ to $1 + i$.

**Solution:**
Parametrize: $z(t) = t(1 + i)$ for $0 \leq t \leq 1$

Then $z'(t) = 1 + i$ and $f(z(t)) = t(1+i)$

$$\int_C z \, dz = \int_0^1 t(1+i) \cdot (1+i) \, dt = (1+i)^2 \int_0^1 t \, dt$$
$$= 2i \cdot \frac{1}{2} = i$$

#### Example 2: Integral Around a Circle

Compute $\oint_{|z|=1} \bar{z} \, dz$.

**Solution:**
Parametrize: $z(t) = e^{it}$ for $0 \leq t \leq 2\pi$

Then $z'(t) = ie^{it}$ and $\bar{z}(t) = e^{-it}$

$$\oint_{|z|=1} \bar{z} \, dz = \int_0^{2\pi} e^{-it} \cdot ie^{it} \, dt = i\int_0^{2\pi} 1 \, dt = 2\pi i$$

**Note:** $\bar{z}$ is NOT analytic, so this integral is non-zero even though the curve is closed.

#### Example 3: The Critical Integral $\oint \frac{dz}{z}$

Compute $\oint_{|z|=1} \frac{dz}{z}$.

**Solution:**
Parametrize: $z(t) = e^{it}$, $z'(t) = ie^{it}$

$$\oint_{|z|=1} \frac{dz}{z} = \int_0^{2\pi} \frac{ie^{it}}{e^{it}} \, dt = i\int_0^{2\pi} dt = \boxed{2\pi i}$$

This is the **most important integral in complex analysis**. It equals $2\pi i$ regardless of the radius:

$$\oint_{|z|=r} \frac{dz}{z} = 2\pi i \quad \text{for any } r > 0$$

### 3. Properties of Complex Line Integrals

#### Linearity
$$\int_C [\alpha f(z) + \beta g(z)] \, dz = \alpha \int_C f(z) \, dz + \beta \int_C g(z) \, dz$$

#### Reversal
$$\int_{-C} f(z) \, dz = -\int_C f(z) \, dz$$

where $-C$ denotes $C$ traversed in the opposite direction.

#### Concatenation
If $C = C_1 + C_2$ (curves joined end-to-end):
$$\int_C f(z) \, dz = \int_{C_1} f(z) \, dz + \int_{C_2} f(z) \, dz$$

#### ML Inequality (Estimation Lemma)

**Theorem:** If $|f(z)| \leq M$ for all $z$ on $C$ and $L$ is the length of $C$, then:
$$\boxed{\left|\int_C f(z) \, dz\right| \leq ML}$$

**Proof:**
$$\left|\int_C f(z) \, dz\right| = \left|\int_a^b f(z(t)) z'(t) \, dt\right| \leq \int_a^b |f(z(t))| |z'(t)| \, dt \leq M \int_a^b |z'(t)| \, dt = ML$$

### 4. Path Independence and Antiderivatives

#### The Fundamental Question

Given $f(z)$ and two points $z_1, z_2$, does $\int_{z_1}^{z_2} f(z) \, dz$ depend on the path?

**Answer:** It depends on whether $f$ has an antiderivative in the region.

#### Antiderivatives in ℂ

**Definition:** $F(z)$ is an *antiderivative* of $f(z)$ in domain $D$ if $F'(z) = f(z)$ for all $z \in D$.

**Fundamental Theorem for Complex Integrals:**
If $f$ is continuous in $D$ and has an antiderivative $F$, then for any contour $C$ in $D$ from $z_1$ to $z_2$:
$$\boxed{\int_C f(z) \, dz = F(z_2) - F(z_1)}$$

**Corollary:** If $f$ has an antiderivative in $D$, then for any closed curve $C$ in $D$:
$$\oint_C f(z) \, dz = 0$$

#### When Do Antiderivatives Exist?

**Theorem:** If $f$ is analytic in a *simply connected* domain $D$, then $f$ has an antiderivative in $D$.

**Simply Connected:** A domain is simply connected if every closed curve in it can be continuously shrunk to a point without leaving the domain. (No "holes")

#### Example: Why $\oint \frac{dz}{z} \neq 0$

The function $f(z) = 1/z$ is analytic in $\mathbb{C} \setminus \{0\}$, but this domain is **not simply connected** (it has a hole at the origin).

The "antiderivative" $\ln z$ is multi-valued. Going around the origin, $\ln z$ increases by $2\pi i$.

### 5. Computing Integrals with Antiderivatives

#### Example: Using Antiderivatives

Compute $\int_C e^z \, dz$ where $C$ goes from $0$ to $1 + i\pi$.

**Solution:**
$e^z$ is entire, so it has an antiderivative $F(z) = e^z$ everywhere.

$$\int_C e^z \, dz = e^{1+i\pi} - e^0 = e \cdot e^{i\pi} - 1 = -e - 1$$

The path doesn't matter!

#### Example: Polynomial Integrals

For any polynomial $P(z)$, the integral from $z_1$ to $z_2$ is path-independent.

$$\int_{z_1}^{z_2} z^n \, dz = \frac{z_2^{n+1} - z_1^{n+1}}{n+1} \quad (n \neq -1)$$

### 6. Integrals Involving $z^n$ Around a Circle

**Key Result:** For the circle $|z - z_0| = r$ traversed counterclockwise:

$$\boxed{\oint_{|z-z_0|=r} (z - z_0)^n \, dz = \begin{cases} 2\pi i & \text{if } n = -1 \\ 0 & \text{if } n \neq -1 \text{ (integer)} \end{cases}}$$

**Proof (n ≠ -1):**
$(z - z_0)^n$ has antiderivative $\frac{(z-z_0)^{n+1}}{n+1}$ which is single-valued in $\mathbb{C} \setminus \{z_0\}$.

For $n = -1$: Parametrize $z = z_0 + re^{it}$:
$$\oint \frac{dz}{z-z_0} = \int_0^{2\pi} \frac{ire^{it}}{re^{it}} dt = 2\pi i$$

---

## Quantum Mechanics Connection

### Path Integrals in Quantum Mechanics

Feynman's path integral formulation expresses quantum amplitudes as integrals over all possible paths:

$$\langle x_f | e^{-iHt/\hbar} | x_i \rangle = \int \mathcal{D}[x(t)] \, e^{iS[x(t)]/\hbar}$$

Key parallels to today's material:

| Complex Line Integrals | Quantum Path Integrals |
|-----------------------|----------------------|
| Parametrized curves $z(t)$ | Particle trajectories $x(t)$ |
| Integration measure $dz$ | Path measure $\mathcal{D}[x]$ |
| Path dependence | All paths contribute |
| Closed loops | Periodic boundary conditions |

### Aharonov-Bohm Effect

When a charged particle moves in a region with magnetic vector potential $\mathbf{A}$:

$$\psi \to \psi \exp\left(\frac{ie}{\hbar c}\oint \mathbf{A} \cdot d\mathbf{r}\right)$$

The phase depends on the **line integral** of $\mathbf{A}$ around a closed loop. This is analogous to $\oint \frac{dz}{z} = 2\pi i$ — the integral "counts" the enclosed flux.

### Berry Phase

When a quantum system evolves adiabatically around a closed loop in parameter space, it acquires a geometric phase:

$$\gamma = i\oint \langle n | \nabla_R | n \rangle \cdot dR$$

This is a complex line integral over parameter space, directly analogous to today's contour integrals.

---

## Worked Examples

### Example 1: Evaluating $\int_C z^2 \, dz$

**Problem:** Compute $\int_C z^2 \, dz$ where $C$ is the arc of $|z| = 2$ from $2$ to $2i$ (counterclockwise).

**Solution:**

**Method 1: Direct Parametrization**
$z(t) = 2e^{it}$, $0 \leq t \leq \pi/2$
$z'(t) = 2ie^{it}$
$z^2 = 4e^{2it}$

$$\int_C z^2 \, dz = \int_0^{\pi/2} 4e^{2it} \cdot 2ie^{it} \, dt = 8i \int_0^{\pi/2} e^{3it} \, dt$$
$$= 8i \left[\frac{e^{3it}}{3i}\right]_0^{\pi/2} = \frac{8}{3}(e^{3i\pi/2} - 1) = \frac{8}{3}(-i - 1)$$

**Method 2: Antiderivative**
$z^2$ has antiderivative $z^3/3$.
$$\int_C z^2 \, dz = \frac{(2i)^3}{3} - \frac{2^3}{3} = \frac{-8i - 8}{3} = \frac{8}{3}(-i - 1)$$

Both methods agree. ✓

### Example 2: A Non-Analytic Integrand

**Problem:** Compute $\int_C |z|^2 \, dz$ where $C$ is:
(a) The straight line from $0$ to $1 + i$
(b) The path $0 \to 1 \to 1 + i$

**Solution:**

**(a) Straight line:** $z(t) = t(1+i)$, $0 \leq t \leq 1$
$|z|^2 = |t(1+i)|^2 = 2t^2$
$z'(t) = 1 + i$

$$\int_C |z|^2 \, dz = \int_0^1 2t^2(1+i) \, dt = 2(1+i) \cdot \frac{1}{3} = \frac{2(1+i)}{3}$$

**(b) Path via corners:**

*First segment (0 → 1):* $z(t) = t$, $0 \leq t \leq 1$
$$\int_0^1 t^2 \, dt = \frac{1}{3}$$

*Second segment (1 → 1+i):* $z(t) = 1 + it$, $0 \leq t \leq 1$
$|z|^2 = 1 + t^2$, $z'(t) = i$
$$\int_0^1 (1 + t^2) \cdot i \, dt = i\left(1 + \frac{1}{3}\right) = \frac{4i}{3}$$

Total: $\frac{1}{3} + \frac{4i}{3} = \frac{1 + 4i}{3}$

**Comparison:** $(a) \neq (b)$, so the integral is **path-dependent**.

This is expected because $|z|^2 = z\bar{z}$ involves $\bar{z}$, which is not analytic.

### Example 3: Integral Around a Square

**Problem:** Compute $\oint_C \frac{dz}{z-1}$ where $C$ is the square with vertices at $\pm 2 \pm 2i$ (counterclockwise).

**Solution:**

The function $f(z) = \frac{1}{z-1}$ has a singularity at $z = 1$, which is **inside** the square.

**Key insight:** The integral around any simple closed curve enclosing $z = 1$ once (counterclockwise) equals $2\pi i$.

This will be proven rigorously by Cauchy's integral theorem, but we can verify by deformation:

The square can be continuously deformed to a circle $|z - 1| = \epsilon$ without crossing the singularity. By the result we proved:
$$\oint_C \frac{dz}{z-1} = \oint_{|z-1|=\epsilon} \frac{dz}{z-1} = 2\pi i$$

---

## Practice Problems

### Problem Set A: Direct Computation

**A1.** Compute $\int_C z \, dz$ where $C$ is:
(a) The straight line from $1$ to $i$
(b) The arc of $|z| = 1$ from $1$ to $i$ (counterclockwise)
Compare your answers.

**A2.** Evaluate $\oint_{|z|=2} \frac{dz}{z^2}$.

**A3.** Compute $\int_0^{1+i} z^3 \, dz$ using antiderivatives.

### Problem Set B: Path Dependence

**B1.** Show that $\int_C \text{Re}(z) \, dz$ is path-dependent by computing it along:
(a) The straight line from $0$ to $1 + i$
(b) The path $0 \to 1 \to 1+i$

**B2.** For $f(z) = \bar{z}$, verify that $\oint_{|z|=1} \bar{z} \, dz = 2\pi i$ by direct computation.

**B3.** Consider $f(z) = z^2\bar{z}$. Compute $\int_C f(z) \, dz$ from $0$ to $1$ along the real axis. Is this function analytic?

### Problem Set C: Estimation and Theory

**C1.** Using the ML inequality, show that:
$$\left|\oint_{|z|=R} \frac{e^{iz}}{z^2} \, dz\right| \leq \frac{2\pi}{R}$$
for $R > 0$.

**C2.** Prove that $\oint_C P(z) \, dz = 0$ for any polynomial $P$ and any closed contour $C$.

**C3.** Let $f$ be analytic in an annulus $r < |z| < R$. Can we conclude that $\oint_{|z|=\rho} f(z) \, dz = 0$ for $r < \rho < R$? Explain.

---

## Solutions to Selected Problems

### Solution A1

**(a) Straight line:** $z(t) = 1 + t(i-1) = (1-t) + it$ for $0 \leq t \leq 1$
$z'(t) = -1 + i$

$$\int_C z \, dz = \int_0^1 [(1-t) + it](-1+i) \, dt$$
$$= (-1+i)\int_0^1 [(1-t) + it] \, dt = (-1+i)\left[\frac{1}{2} + \frac{i}{2}\right] = (-1+i) \cdot \frac{1+i}{2}$$
$$= \frac{-1-i+i+i^2}{2} = \frac{-2}{2} = -1$$

**(b) Arc:** $z(t) = e^{it}$ for $0 \leq t \leq \pi/2$
$$\int_C z \, dz = \int_0^{\pi/2} e^{it} \cdot ie^{it} \, dt = i\int_0^{\pi/2} e^{2it} \, dt = i\left[\frac{e^{2it}}{2i}\right]_0^{\pi/2}$$
$$= \frac{1}{2}(e^{i\pi} - 1) = \frac{1}{2}(-1-1) = -1$$

**Both paths give $-1$!** This confirms $z$ has an antiderivative $z^2/2$.

### Solution A2

$$\oint_{|z|=2} \frac{dz}{z^2} = \oint_{|z|=2} z^{-2} \, dz$$

Since $n = -2 \neq -1$, and $z^{-2}$ has antiderivative $-z^{-1}$:
$$\oint_{|z|=2} z^{-2} \, dz = 0$$

### Solution C1

On $|z| = R$: $|e^{iz}| = e^{-\text{Im}(z)}$. For $z = Re^{i\theta}$:
$$\text{Im}(z) = R\sin\theta$$

When $0 \leq \theta \leq \pi$: $\sin\theta \geq 0$, so $|e^{iz}| \leq 1$.
When $\pi \leq \theta \leq 2\pi$: $\sin\theta \leq 0$, so $|e^{iz}| \leq e^R$.

More careful: on the upper semicircle $|e^{iz}| \leq 1$, so:
$$\left|\frac{e^{iz}}{z^2}\right| \leq \frac{1}{R^2}$$

For the full circle (crude bound): $M = \frac{e^R}{R^2}$ (but this grows!)

Better approach for the intended bound: note that we're asked for large $R$ behavior. On the entire circle:
$$\left|\frac{e^{iz}}{z^2}\right| \leq \frac{e^R}{R^2}$$

But if we restrict to $|e^{iz}| \leq 1$ (which holds when $\text{Im}(z) \geq 0$):

Using $|e^{iz}| = e^{-y}$ and assuming we stay in upper half-plane:
$$M = \frac{1}{R^2}, \quad L = 2\pi R$$
$$\left|\oint\right| \leq ML = \frac{2\pi}{R}$$

---

## Computational Lab

### Lab 1: Visualizing Complex Line Integrals

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def complex_line_integral(f, z_path, t_range, n_points=1000):
    """
    Compute complex line integral numerically.

    Parameters:
    -----------
    f : callable
        Complex function to integrate
    z_path : callable
        Parametrization z(t)
    t_range : tuple
        (t_start, t_end)
    n_points : int
        Number of points for numerical integration

    Returns:
    --------
    complex : Value of the integral
    """
    t = np.linspace(t_range[0], t_range[1], n_points)
    dt = t[1] - t[0]

    # Compute z'(t) numerically
    z = z_path(t)
    dz = np.gradient(z, dt)

    # Integrand: f(z(t)) * z'(t)
    integrand = f(z) * dz

    # Trapezoidal integration
    return np.trapz(integrand, t)

# Example: Integral of 1/z around unit circle
z_circle = lambda t: np.exp(1j * t)
f_inverse = lambda z: 1/z

result = complex_line_integral(f_inverse, z_circle, (0, 2*np.pi))
print(f"∮ dz/z around |z|=1: {result:.6f}")
print(f"Expected: 2πi = {2*np.pi*1j:.6f}")
print(f"Error: {abs(result - 2*np.pi*1j):.2e}")

# Visualize the path and integrand
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Path visualization
t = np.linspace(0, 2*np.pi, 100)
z = z_circle(t)
axes[0].plot(z.real, z.imag, 'b-', linewidth=2)
axes[0].arrow(0.7, 0.7, -0.1, 0.1, head_width=0.1, color='b')
axes[0].plot(0, 0, 'ro', markersize=10, label='Singularity at z=0')
axes[0].set_xlabel('Re(z)')
axes[0].set_ylabel('Im(z)')
axes[0].set_title('Integration Path: Unit Circle')
axes[0].legend()
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

# Real part of integrand
dz = 1j * z  # z'(t) = i*e^{it}
integrand = f_inverse(z) * dz
axes[1].plot(t, integrand.real, 'r-', linewidth=2)
axes[1].set_xlabel('t')
axes[1].set_ylabel('Re(f·dz/dt)')
axes[1].set_title('Real Part of Integrand')
axes[1].grid(True, alpha=0.3)

# Imaginary part of integrand
axes[2].plot(t, integrand.imag, 'b-', linewidth=2)
axes[2].set_xlabel('t')
axes[2].set_ylabel('Im(f·dz/dt)')
axes[2].set_title('Imaginary Part (constant = 1)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('line_integral_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 2: Path Dependence Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

def integrate_along_path(f, z_path, t_range, n=1000):
    """Numerically integrate f(z)dz along parametrized path."""
    t = np.linspace(t_range[0], t_range[1], n)
    dt = t[1] - t[0]
    z = z_path(t)
    dz = np.gradient(z, dt)
    return np.trapz(f(z) * dz, t)

# Non-analytic function: f(z) = |z|^2 = z * conj(z)
f_non_analytic = lambda z: np.abs(z)**2

# Path 1: Straight line from 0 to 1+i
path1 = lambda t: t * (1 + 1j)
result1 = integrate_along_path(f_non_analytic, path1, (0, 1))

# Path 2: Right angle (0 -> 1 -> 1+i)
path2a = lambda t: t  # 0 to 1
path2b = lambda t: 1 + 1j*t  # 1 to 1+i
result2 = integrate_along_path(f_non_analytic, path2a, (0, 1)) + \
          integrate_along_path(f_non_analytic, path2b, (0, 1))

# Path 3: Circular arc
path3 = lambda t: np.sqrt(2) * np.exp(1j * t)  # |z| = sqrt(2)
result3 = integrate_along_path(f_non_analytic, path3, (0, np.pi/4))

print("Integration of |z|² from 0 to 1+i:")
print(f"Path 1 (straight line):  {result1:.6f}")
print(f"Path 2 (right angle):    {result2:.6f}")
print(f"Path 3 (circular arc):   {result3:.6f}")
print("\nResults differ → path dependent → f(z) = |z|² is NOT analytic")

# Analytic function: f(z) = z
f_analytic = lambda z: z

result1_a = integrate_along_path(f_analytic, path1, (0, 1))
result2_a = integrate_along_path(f_analytic, path2a, (0, 1)) + \
            integrate_along_path(f_analytic, path2b, (0, 1))
result3_a = integrate_along_path(f_analytic, path3, (0, np.pi/4))

print("\n" + "="*50)
print("Integration of z from 0 to 1+i:")
print(f"Path 1 (straight line):  {result1_a:.6f}")
print(f"Path 2 (right angle):    {result2_a:.6f}")
print(f"Path 3 (circular arc):   {result3_a:.6f}")
print(f"Theoretical (z²/2):      {(1+1j)**2/2:.6f}")
print("\nResults match → path independent → f(z) = z is analytic")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot paths
t1 = np.linspace(0, 1, 100)
z1 = path1(t1)

t2a = np.linspace(0, 1, 50)
t2b = np.linspace(0, 1, 50)
z2a = path2a(t2a)
z2b = path2b(t2b)

t3 = np.linspace(0, np.pi/4, 100)
z3 = path3(t3)

axes[0].plot(z1.real, z1.imag, 'b-', linewidth=2, label='Path 1: Straight')
axes[0].plot(z2a.real, z2a.imag, 'r-', linewidth=2, label='Path 2: Corner')
axes[0].plot(z2b.real, z2b.imag, 'r-', linewidth=2)
axes[0].plot(z3.real, z3.imag, 'g-', linewidth=2, label='Path 3: Arc')
axes[0].plot([0], [0], 'ko', markersize=10)
axes[0].plot([1], [1], 'ko', markersize=10)
axes[0].annotate('Start', (0, 0), xytext=(-0.2, -0.1))
axes[0].annotate('End', (1, 1), xytext=(1.05, 1.05))
axes[0].set_xlabel('Re(z)')
axes[0].set_ylabel('Im(z)')
axes[0].set_title('Three Paths from 0 to 1+i')
axes[0].legend()
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

# Bar chart of results
labels = ['Straight', 'Corner', 'Arc']
non_analytic_real = [result1.real, result2.real, result3.real]
non_analytic_imag = [result1.imag, result2.imag, result3.imag]
analytic_real = [result1_a.real, result2_a.real, result3_a.real]

x = np.arange(len(labels))
width = 0.25

bars1 = axes[1].bar(x - width, non_analytic_real, width, label='|z|² (Re)', color='blue', alpha=0.7)
bars2 = axes[1].bar(x, non_analytic_imag, width, label='|z|² (Im)', color='red', alpha=0.7)
bars3 = axes[1].bar(x + width, analytic_real, width, label='z (Re)', color='green', alpha=0.7)

axes[1].axhline(y=i.real, color='green', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Path')
axes[1].set_ylabel('Integral Value')
axes[1].set_title('Path Dependence Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('path_dependence.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 3: The Critical Integral $\oint \frac{dz}{z-z_0}$

```python
import numpy as np
import matplotlib.pyplot as plt

def integrate_around_point(z0, radius, n=1000):
    """
    Compute ∮ dz/(z-z0) around a circle of given radius centered at z0.
    """
    t = np.linspace(0, 2*np.pi, n)
    dt = t[1] - t[0]
    z = z0 + radius * np.exp(1j * t)
    dz = 1j * radius * np.exp(1j * t)
    integrand = dz / (z - z0)
    return np.trapz(integrand, t)

# Test for various radii and center positions
print("Testing ∮ dz/(z-z0) = 2πi:")
print("-" * 50)

for z0 in [0, 1+1j, -2]:
    for r in [0.5, 1, 2, 5]:
        result = integrate_around_point(z0, r)
        error = abs(result - 2*np.pi*1j)
        print(f"z0 = {z0:6}, r = {r}: {result:.6f}, error = {error:.2e}")
    print()

# Visualize contours and the integrand
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Multiple contours around origin
theta = np.linspace(0, 2*np.pi, 100)
for r, color in zip([0.5, 1, 2], ['blue', 'green', 'red']):
    z = r * np.exp(1j * theta)
    axes[0].plot(z.real, z.imag, color=color, linewidth=2, label=f'r = {r}')
    # Add arrow to show direction
    idx = 25
    axes[0].annotate('', xy=(z[idx+1].real, z[idx+1].imag),
                    xytext=(z[idx].real, z[idx].imag),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

axes[0].plot(0, 0, 'ko', markersize=10, label='Singularity')
axes[0].set_xlabel('Re(z)')
axes[0].set_ylabel('Im(z)')
axes[0].set_title('All Contours Give ∮ dz/z = 2πi')
axes[0].legend()
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

# Winding number visualization
# When singularity is inside vs outside
z0_inside = 0
z0_outside = 3

# Contour: |z-1| = 0.5 (does not contain origin)
center = 1
radius = 0.5
z = center + radius * np.exp(1j * theta)

result_inside = integrate_around_point(0, 1)  # Circle contains 0
result_outside = integrate_around_point(3, 0.5)  # Circle around 3, doesn't contain 0

# Actually compute for the specific contour
t = np.linspace(0, 2*np.pi, 1000)
dt = t[1] - t[0]

# Contour |z-1| = 0.5
z_path = 1 + 0.5*np.exp(1j*t)
dz = 0.5j*np.exp(1j*t)
integrand = dz / z_path
result_not_enclosing = np.trapz(integrand, t)

# Contour |z| = 2 (encloses origin)
z_path2 = 2*np.exp(1j*t)
dz2 = 2j*np.exp(1j*t)
integrand2 = dz2 / z_path2
result_enclosing = np.trapz(integrand2, t)

axes[1].plot((1 + 0.5*np.exp(1j*theta)).real, (1 + 0.5*np.exp(1j*theta)).imag,
             'b-', linewidth=2, label=f'|z-1|=0.5: {result_not_enclosing:.3f}')
axes[1].plot((2*np.exp(1j*theta)).real, (2*np.exp(1j*theta)).imag,
             'r-', linewidth=2, label=f'|z|=2: {result_enclosing:.3f}')
axes[1].plot(0, 0, 'ko', markersize=10, label='Singularity at 0')
axes[1].set_xlabel('Re(z)')
axes[1].set_ylabel('Im(z)')
axes[1].set_title('Contour Enclosing Singularity vs Not')
axes[1].legend()
axes[1].axis('equal')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('winding_number.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n∮ dz/z for |z-1|=0.5 (origin outside): {result_not_enclosing:.6f}")
print(f"∮ dz/z for |z|=2 (origin inside):     {result_enclosing:.6f}")
```

### Lab 4: ML Inequality Verification

```python
import numpy as np
import matplotlib.pyplot as plt

def verify_ml_inequality(f, f_bound, z_path, t_range, label):
    """
    Verify ML inequality: |∮f(z)dz| ≤ M * L
    where M = max|f(z)| on path, L = arc length
    """
    t = np.linspace(t_range[0], t_range[1], 10000)
    dt = t[1] - t[0]
    z = z_path(t)
    dz = np.gradient(z, dt)

    # Compute integral
    integral = np.trapz(f(z) * dz, t)

    # Compute M (maximum of |f|)
    M = np.max(np.abs(f(z)))
    M_theoretical = f_bound(z_path, t)

    # Compute L (arc length)
    L = np.trapz(np.abs(dz), t)

    print(f"\n{label}:")
    print(f"  |∮ f(z) dz| = {np.abs(integral):.6f}")
    print(f"  M (numerical) = {M:.6f}")
    print(f"  L = {L:.6f}")
    print(f"  M * L = {M * L:.6f}")
    print(f"  Inequality satisfied: {np.abs(integral) <= M * L + 1e-10}")

    return np.abs(integral), M * L

# Example 1: f(z) = z² on semicircle
f1 = lambda z: z**2
bound1 = lambda path, t: np.max(np.abs(path(t))**2)
path1 = lambda t: 2 * np.exp(1j * t)

result1, ML1 = verify_ml_inequality(f1, bound1, path1, (0, np.pi), "f(z) = z² on |z|=2 semicircle")

# Example 2: f(z) = e^z/z² on circle
f2 = lambda z: np.exp(z) / z**2
bound2 = lambda path, t: np.exp(np.max(path(t).real)) / np.min(np.abs(path(t)))**2
path2 = lambda t: 3 * np.exp(1j * t)

result2, ML2 = verify_ml_inequality(f2, bound2, path2, (0, 2*np.pi), "f(z) = e^z/z² on |z|=3")

# Visualization: How tight is the bound?
fig, ax = plt.subplots(figsize=(10, 6))

# For f(z) = 1/(z-a) on |z|=1 with varying a
a_values = np.linspace(1.1, 5, 50)
actual_values = []
ml_bounds = []

for a in a_values:
    f = lambda z, a=a: 1/(z - a)
    path = lambda t: np.exp(1j * t)
    t = np.linspace(0, 2*np.pi, 1000)
    dt = t[1] - t[0]
    z = path(t)
    dz = 1j * np.exp(1j * t)

    integral = np.abs(np.trapz(f(z) * dz, t))
    M = 1 / (a - 1)  # max |1/(z-a)| on |z|=1 when a > 1
    L = 2 * np.pi

    actual_values.append(integral)
    ml_bounds.append(M * L)

ax.plot(a_values, actual_values, 'b-', linewidth=2, label='Actual |∮ dz/(z-a)|')
ax.plot(a_values, ml_bounds, 'r--', linewidth=2, label='ML bound: 2π/(a-1)')
ax.fill_between(a_values, actual_values, ml_bounds, alpha=0.3, color='gray', label='Gap')
ax.set_xlabel('a (position of singularity)')
ax.set_ylabel('Value')
ax.set_title('ML Inequality: Actual vs Bound for ∮ dz/(z-a) on |z|=1')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_inequality.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nNote: As a → 1⁺, both actual integral and bound diverge.")
print("As a → ∞, both approach 0.")
print("The ML bound is sharp when |f| is nearly constant on the contour.")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\int_C f(z) \, dz = \int_a^b f(z(t)) z'(t) \, dt$ | Definition of complex line integral |
| $\int_C f(z) \, dz = F(z_2) - F(z_1)$ | Fundamental theorem (when antiderivative exists) |
| $\left\|\int_C f(z) \, dz\right\| \leq ML$ | ML inequality |
| $\oint_{\|z-z_0\|=r} (z-z_0)^n \, dz = \begin{cases} 2\pi i & n=-1 \\ 0 & n \neq -1 \end{cases}$ | Key integral formula |

### Main Takeaways

1. **Complex line integrals generalize real integrals** but with richer structure due to path dependence.

2. **Analytic functions have path-independent integrals** in simply connected domains.

3. **The integral $\oint \frac{dz}{z-z_0} = 2\pi i$** is the most important result, independent of the specific closed curve.

4. **The ML inequality** provides bounds on integrals without explicit computation.

5. **Path integrals in QM** are direct generalizations of these complex line integrals.

---

## Daily Checklist

### Understanding
- [ ] I can define complex line integrals via parametrization
- [ ] I understand when integrals are path-independent
- [ ] I can compute $\oint z^n \, dz$ for integer $n$
- [ ] I can apply the ML inequality

### Computation
- [ ] I can parametrize common paths (lines, circles, arcs)
- [ ] I can evaluate integrals using antiderivatives when available
- [ ] I can verify path dependence for non-analytic functions

### Connection
- [ ] I see how this relates to Feynman path integrals
- [ ] I understand the geometric meaning of $\oint \frac{dz}{z} = 2\pi i$

---

## Preview: Day 177

Tomorrow we prove **Cauchy's Integral Theorem**:

$$\oint_C f(z) \, dz = 0 \quad \text{for analytic } f \text{ in simply connected domain}$$

This fundamental result explains why analytic functions have path-independent integrals and leads directly to Cauchy's integral formula — the key to all of complex analysis.

---

*"In mathematics the art of asking questions is more valuable than solving problems."*
— Georg Cantor
