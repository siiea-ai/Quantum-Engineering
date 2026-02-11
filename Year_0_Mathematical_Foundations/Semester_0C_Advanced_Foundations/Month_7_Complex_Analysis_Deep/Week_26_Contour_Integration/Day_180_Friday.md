# Day 180: Applications to Real Integrals II — Branch Cuts and Special Techniques

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Branch Cuts & Keyhole Contours |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Special Integrals & Techniques |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 180, you will be able to:

1. Use keyhole contours for integrals with branch points
2. Evaluate integrals involving $x^\alpha$ for non-integer $\alpha$
3. Compute trigonometric integrals via complex methods
4. Derive the famous Dirichlet integral
5. Handle integrals with poles on the real axis
6. Connect to quantum mechanical propagators with branch cuts

---

## Core Content

### 1. Branch Cuts and Keyhole Contours

#### The Challenge

Integrals like $\int_0^\infty \frac{x^{\alpha-1}}{1+x} dx$ involve $x^{\alpha-1}$ which is multi-valued for non-integer $\alpha$.

**Solution:** Use a **keyhole contour** that avoids the branch cut.

#### The Keyhole Contour

For integrands with a branch point at the origin:

1. Draw a branch cut along the positive real axis
2. Create a "keyhole" contour:
   - Along the top of the cut: $x + i\varepsilon$
   - Large circle: $|z| = R$
   - Along the bottom of the cut: $x - i\varepsilon$
   - Small circle: $|z| = \varepsilon$

The function $z^{\alpha-1} = e^{(\alpha-1)\ln z}$ is single-valued on this contour with a specific branch of $\ln z$.

### 2. The Fundamental Beta Integral

**Problem:** Evaluate $\int_0^\infty \frac{x^{\alpha-1}}{1+x} dx$ for $0 < \alpha < 1$.

**Solution:**

Consider $f(z) = \frac{z^{\alpha-1}}{1+z}$ with branch cut on positive real axis.

**On top of cut:** $z = x$, $z^{\alpha-1} = x^{\alpha-1}$
**On bottom of cut:** $z = xe^{2\pi i}$, $z^{\alpha-1} = x^{\alpha-1}e^{2\pi i(\alpha-1)}$

The pole at $z = -1$ has residue:
$$\text{Res}_{z=-1} \frac{z^{\alpha-1}}{1+z} = (-1)^{\alpha-1} = e^{i\pi(\alpha-1)}$$

**Contour integral:**
$$\oint = \int_0^\infty \frac{x^{\alpha-1}}{1+x}dx - e^{2\pi i(\alpha-1)}\int_0^\infty \frac{x^{\alpha-1}}{1+x}dx + (\text{circles})$$

As $\varepsilon \to 0$ and $R \to \infty$, the circles contribute zero (for $0 < \alpha < 1$).

$$\oint = (1 - e^{2\pi i(\alpha-1)}) \int_0^\infty \frac{x^{\alpha-1}}{1+x}dx = 2\pi i \cdot e^{i\pi(\alpha-1)}$$

Solving:
$$\int_0^\infty \frac{x^{\alpha-1}}{1+x}dx = \frac{2\pi i \cdot e^{i\pi(\alpha-1)}}{1 - e^{2\pi i(\alpha-1)}}$$

Using $1 - e^{2i\theta} = -2i\sin\theta \cdot e^{i\theta}$:
$$= \frac{2\pi i \cdot e^{i\pi(\alpha-1)}}{-2i\sin(\pi(\alpha-1))e^{i\pi(\alpha-1)}} = \frac{\pi}{-\sin(\pi\alpha - \pi)} = \boxed{\frac{\pi}{\sin(\pi\alpha)}}$$

This is related to the **Beta function** and **Gamma function**.

### 3. Trigonometric Integrals

#### Type: $\int_0^{2\pi} R(\cos\theta, \sin\theta) d\theta$

**Method:** Substitute $z = e^{i\theta}$, $dz = ie^{i\theta}d\theta$

Then:
$$\cos\theta = \frac{z + z^{-1}}{2}, \quad \sin\theta = \frac{z - z^{-1}}{2i}, \quad d\theta = \frac{dz}{iz}$$

The integral becomes a contour integral around the unit circle.

#### Example: $\int_0^{2\pi} \frac{d\theta}{a + b\cos\theta}$ ($a > b > 0$)

With the substitution:
$$\int_0^{2\pi} \frac{d\theta}{a + b\cos\theta} = \oint_{|z|=1} \frac{1}{a + b\frac{z+z^{-1}}{2}} \cdot \frac{dz}{iz}$$
$$= \oint_{|z|=1} \frac{2}{2az + b(z^2 + 1)} \cdot \frac{dz}{iz} = \frac{2}{ib}\oint_{|z|=1} \frac{dz}{z^2 + \frac{2a}{b}z + 1}$$

The denominator factors as $(z - z_1)(z - z_2)$ where:
$$z_{1,2} = \frac{-a \pm \sqrt{a^2 - b^2}}{b}$$

Since $a > b > 0$: $|z_1| < 1$, $|z_2| > 1$.

Only $z_1$ is inside the unit circle:
$$\text{Res}_{z=z_1} = \frac{1}{z_1 - z_2} = \frac{b}{-2\sqrt{a^2-b^2}}$$

Result:
$$\int_0^{2\pi} \frac{d\theta}{a + b\cos\theta} = 2\pi i \cdot \frac{2}{ib} \cdot \frac{b}{-2\sqrt{a^2-b^2}} = \boxed{\frac{2\pi}{\sqrt{a^2-b^2}}}$$

### 4. The Dirichlet Integral

**Theorem:** $\displaystyle\int_0^\infty \frac{\sin x}{x} dx = \frac{\pi}{2}$

This is one of the most important integrals in mathematics and physics!

**Proof via Contour Integration:**

Consider $f(z) = \frac{e^{iz}}{z}$ around the indented semicircle:
- Real axis from $\varepsilon$ to $R$
- Semicircle $|z| = R$ in upper half-plane
- Real axis from $-R$ to $-\varepsilon$
- Small semicircle $|z| = \varepsilon$ around origin (above real axis)

Since $f$ is analytic inside (the origin is excluded):
$$\oint f(z) dz = 0$$

**Large semicircle:** By Jordan's lemma, $\int_{\gamma_R} \to 0$ as $R \to \infty$.

**Small semicircle:** Parametrize $z = \varepsilon e^{i\theta}$, $\theta: \pi \to 0$:
$$\int_{\gamma_\varepsilon} \frac{e^{iz}}{z} dz = \int_\pi^0 \frac{e^{i\varepsilon e^{i\theta}}}{\varepsilon e^{i\theta}} \cdot i\varepsilon e^{i\theta} d\theta = i\int_\pi^0 e^{i\varepsilon e^{i\theta}} d\theta$$

As $\varepsilon \to 0$: $\to i \cdot (-\pi) = -i\pi$

**Real axis contributions:**
$$\int_\varepsilon^R \frac{e^{ix}}{x}dx + \int_{-R}^{-\varepsilon} \frac{e^{ix}}{x}dx = \int_\varepsilon^R \frac{e^{ix} - e^{-ix}}{x}dx = 2i\int_\varepsilon^R \frac{\sin x}{x}dx$$

Setting total to zero:
$$2i\int_0^\infty \frac{\sin x}{x}dx - i\pi = 0$$

$$\boxed{\int_0^\infty \frac{\sin x}{x} dx = \frac{\pi}{2}}$$

### 5. Indented Contours and Principal Value

When poles lie on the real axis, we use indented contours.

**Key result:** For a simple pole at $x = x_0$ on the real axis, a small semicircular indentation (radius $\varepsilon$, in the upper half-plane) contributes:
$$-i\pi \cdot \text{Res}_{z=x_0} f(z)$$

as $\varepsilon \to 0$ (the minus sign for upper indentation, plus for lower).

**Relation to principal value:**
$$\mathcal{P}\int_{-\infty}^{\infty} f(x) dx = 2\pi i \sum_{\text{UHP}} \text{Res} + i\pi \sum_{\text{real axis}} \text{Res}$$

(for upper indentations)

### 6. Integrals with $\ln x$

**Example:** $\int_0^\infty \frac{\ln x}{1 + x^2} dx$

**Method:** Keyhole contour for $f(z) = \frac{(\ln z)^2}{1+z^2}$

On upper edge: $\ln z = \ln x$
On lower edge: $\ln z = \ln x + 2\pi i$

Working through the algebra:
$$\int_0^\infty \frac{\ln x}{1+x^2} dx = 0$$

(by symmetry arguments)

But $\int_0^\infty \frac{(\ln x)^2}{1+x^2} dx = \frac{\pi^3}{8}$

---

## Quantum Mechanics Connection

### Propagators and Branch Cuts

The free particle propagator in energy representation:
$$G_0(E) = \frac{1}{E - p^2/2m + i\varepsilon}$$

has a branch cut along the positive real axis (continuous spectrum) when summed over momenta:
$$G_0(r, r'; E) = \int \frac{d^3p}{(2\pi)^3} \frac{e^{i\mathbf{p}\cdot(\mathbf{r}-\mathbf{r}')}}{E - p^2/2m + i\varepsilon}$$

The $+i\varepsilon$ prescription:
- Places the branch cut just **below** the real axis
- Ensures **retarded** (causal) propagation
- Corresponds to **outgoing** spherical waves

### Density of States from Discontinuity

The **spectral function**:
$$A(E) = -\frac{1}{\pi}\text{Im}\, G(E + i\varepsilon) = -\frac{1}{\pi}\lim_{\varepsilon \to 0^+}[G(E + i\varepsilon) - G(E - i\varepsilon)]/(2i)$$

This is the **discontinuity across the branch cut** — exactly what keyhole contours compute!

### Sommerfeld Expansion

At finite temperature, Fermi integrals involve:
$$\int_0^\infty \frac{f(\varepsilon)}{\exp[(\varepsilon-\mu)/k_BT] + 1} d\varepsilon$$

The contour integration techniques, combined with the pole structure of the Fermi function, give the Sommerfeld expansion for low-temperature thermodynamics.

---

## Worked Examples

### Example 1: Gamma Function Integral

**Problem:** Show that $\Gamma(\alpha)\Gamma(1-\alpha) = \frac{\pi}{\sin(\pi\alpha)}$ using:
$$\int_0^\infty \frac{x^{\alpha-1}}{1+x} dx = \frac{\pi}{\sin(\pi\alpha)}$$

**Solution:**

The Beta function: $B(\alpha, 1-\alpha) = \int_0^1 t^{\alpha-1}(1-t)^{-\alpha} dt$

Substitute $t = x/(1+x)$, so $dt = dx/(1+x)^2$ and $1-t = 1/(1+x)$:
$$B(\alpha, 1-\alpha) = \int_0^\infty \frac{(x/(1+x))^{\alpha-1}}{(1+x)^{-\alpha}} \cdot \frac{dx}{(1+x)^2} = \int_0^\infty \frac{x^{\alpha-1}}{1+x} dx = \frac{\pi}{\sin(\pi\alpha)}$$

Since $B(\alpha, 1-\alpha) = \frac{\Gamma(\alpha)\Gamma(1-\alpha)}{\Gamma(1)} = \Gamma(\alpha)\Gamma(1-\alpha)$:
$$\boxed{\Gamma(\alpha)\Gamma(1-\alpha) = \frac{\pi}{\sin(\pi\alpha)}}$$

This is the **reflection formula** for the Gamma function!

### Example 2: Fresnel Integrals

**Problem:** Compute $\int_0^\infty \cos(x^2) dx$.

**Solution:**

Consider $\int_C e^{iz^2} dz$ around a **pie slice** contour:
- Real axis from $0$ to $R$
- Arc from $R$ to $Re^{i\pi/4}$
- Ray from $Re^{i\pi/4}$ to $0$

Since $e^{iz^2}$ is entire, the integral is zero.

**On the ray $z = te^{i\pi/4}$:**
$$z^2 = t^2 e^{i\pi/2} = it^2$$
$$e^{iz^2} = e^{i(it^2)} = e^{-t^2}$$
$$dz = e^{i\pi/4} dt$$

So: $\int_{\text{ray}} = e^{i\pi/4} \int_R^0 e^{-t^2} dt = -e^{i\pi/4} \int_0^R e^{-t^2} dt$

**On the arc:** As $R \to \infty$, this vanishes (argue using Jordan-like bounds).

Taking $R \to \infty$:
$$\int_0^\infty e^{ix^2} dx = e^{i\pi/4} \int_0^\infty e^{-t^2} dt = e^{i\pi/4} \cdot \frac{\sqrt{\pi}}{2}$$

Taking real and imaginary parts:
$$\int_0^\infty \cos(x^2) dx = \frac{\sqrt{\pi}}{2} \cos(\pi/4) = \frac{\sqrt{\pi}}{2} \cdot \frac{1}{\sqrt{2}} = \boxed{\frac{1}{2}\sqrt{\frac{\pi}{2}}}$$

$$\int_0^\infty \sin(x^2) dx = \frac{1}{2}\sqrt{\frac{\pi}{2}}$$

These are the **Fresnel integrals**, essential in optics and diffraction theory!

### Example 3: Trigonometric Integral with Powers

**Problem:** Evaluate $\int_0^{2\pi} \frac{d\theta}{2 + \cos\theta}$.

**Solution:**

Let $z = e^{i\theta}$, so $\cos\theta = (z + z^{-1})/2$, $d\theta = dz/(iz)$.

$$\int_0^{2\pi} \frac{d\theta}{2 + \cos\theta} = \oint_{|z|=1} \frac{1}{2 + (z+z^{-1})/2} \cdot \frac{dz}{iz}$$
$$= \oint_{|z|=1} \frac{2}{4 + z + z^{-1}} \cdot \frac{dz}{iz} = \frac{2}{i}\oint_{|z|=1} \frac{dz}{z^2 + 4z + 1}$$

Roots of $z^2 + 4z + 1 = 0$: $z = \frac{-4 \pm \sqrt{12}}{2} = -2 \pm \sqrt{3}$

$z_1 = -2 + \sqrt{3} \approx -0.27$ (inside unit circle)
$z_2 = -2 - \sqrt{3} \approx -3.73$ (outside)

$$\text{Res}_{z=z_1} = \frac{1}{z_1 - z_2} = \frac{1}{2\sqrt{3}}$$

$$\int_0^{2\pi} \frac{d\theta}{2+\cos\theta} = \frac{2}{i} \cdot 2\pi i \cdot \frac{1}{2\sqrt{3}} = \boxed{\frac{2\pi}{\sqrt{3}}}$$

### Example 4: Mixed Integral

**Problem:** Evaluate $\int_0^\infty \frac{x^{1/3}}{1+x^2} dx$.

**Solution:**

Use keyhole contour for $f(z) = \frac{z^{1/3}}{1+z^2}$ with branch cut on positive real axis.

Poles at $z = \pm i$, both in the keyhole region.

**Residue at $z = i = e^{i\pi/2}$:**
$$\text{Res} = \frac{e^{i\pi/6}}{2i}$$

**Residue at $z = -i = e^{3i\pi/2}$:**
$$\text{Res} = \frac{e^{i\pi/2}}{-2i} = \frac{i}{-2i} = -\frac{1}{2}$$

Wait, let me recalculate:
$(-i)^{1/3} = e^{i(3\pi/2)/3} = e^{i\pi/2} = i$ (using principal branch $0 < \arg z < 2\pi$)

Actually for the keyhole contour: $\arg z \in (0, 2\pi)$, so:
- At $z = i$: $\arg z = \pi/2$, $z^{1/3} = e^{i\pi/6}$, Res $= e^{i\pi/6}/(2i)$
- At $z = -i$: $\arg z = 3\pi/2$, $z^{1/3} = e^{i\pi/2} = i$, Res $= i/(-2i) = -1/2$

Sum: $\frac{e^{i\pi/6}}{2i} - \frac{1}{2} = \frac{e^{i\pi/6}}{2i} - \frac{1}{2}$

Keyhole contribution:
$$\oint = (1 - e^{2\pi i/3}) \int_0^\infty \frac{x^{1/3}}{1+x^2} dx = 2\pi i \left(\frac{e^{i\pi/6}}{2i} - \frac{1}{2}\right)$$

Solving (this gets algebraically intensive):
$$\int_0^\infty \frac{x^{1/3}}{1+x^2} dx = \boxed{\frac{\pi}{\sqrt{3}}}$$

---

## Practice Problems

### Problem Set A: Keyhole Contours

**A1.** Evaluate $\int_0^\infty \frac{x^{-1/2}}{1+x} dx$.

**A2.** Compute $\int_0^\infty \frac{\ln x}{(1+x)^2} dx$.

**A3.** Show that $\int_0^\infty \frac{x^{p-1}}{1+x^n} dx = \frac{\pi/n}{\sin(p\pi/n)}$ for $0 < p < n$.

### Problem Set B: Trigonometric Integrals

**B1.** Evaluate $\int_0^{2\pi} \frac{d\theta}{3 + 2\cos\theta}$.

**B2.** Compute $\int_0^{2\pi} \cos^4\theta \, d\theta$.

**B3.** Find $\int_0^\pi \frac{d\theta}{1 + \sin^2\theta}$.

### Problem Set C: Special Integrals

**C1.** Derive $\int_0^\infty \frac{\sin^2 x}{x^2} dx = \frac{\pi}{2}$ from the Dirichlet integral.

**C2.** Compute $\int_0^\infty e^{-x^2}\cos(2x) dx$.

**C3.** Evaluate $\int_0^\infty \frac{x}{e^x - 1} dx = \frac{\pi^2}{6}$.

---

## Solutions to Selected Problems

### Solution A1

Using the general result with $\alpha = 1/2$:
$$\int_0^\infty \frac{x^{-1/2}}{1+x} dx = \frac{\pi}{\sin(\pi/2)} = \pi$$

### Solution B1

Following the method of Example 3 with $a = 3$, $b = 2$:
$$\int_0^{2\pi} \frac{d\theta}{3 + 2\cos\theta} = \frac{2\pi}{\sqrt{9-4}} = \frac{2\pi}{\sqrt{5}}$$

### Solution C1

Integration by parts: Let $u = \sin^2 x$, $dv = dx/x^2$.

Or use: $\frac{d}{dx}\left(\frac{\sin x}{x}\right)^2$ and relate to Dirichlet.

Direct approach: $\sin^2 x = \frac{1-\cos(2x)}{2}$

$\int_0^\infty \frac{\sin^2 x}{x^2} dx = \frac{1}{2}\int_0^\infty \frac{1-\cos(2x)}{x^2} dx$

Using integration by parts or known results:
$$= \frac{1}{2} \cdot \pi = \frac{\pi}{2}$$

---

## Computational Lab

### Lab 1: Keyhole Contour Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def keyhole_contour(R, epsilon, n_points=1000):
    """Generate points on a keyhole contour."""
    # Upper edge of branch cut (x + i*delta)
    delta = epsilon/10
    x1 = np.linspace(epsilon, R, n_points//4)
    z1 = x1 + 1j*delta

    # Large circle (counterclockwise from angle delta to 2pi-delta)
    theta1 = np.linspace(np.arctan(delta/R), 2*np.pi - np.arctan(delta/R), n_points//4)
    z2 = R * np.exp(1j * theta1)

    # Lower edge of branch cut (x - i*delta), going from R to epsilon
    x3 = np.linspace(R, epsilon, n_points//4)
    z3 = x3 - 1j*delta

    # Small circle (clockwise from 2pi-small to small)
    theta2 = np.linspace(2*np.pi - np.arctan(delta/epsilon), np.arctan(delta/epsilon), n_points//4)
    z4 = epsilon * np.exp(1j * theta2)

    return np.concatenate([z1, z2, z3, z4])

# Create keyhole contour
R = 3
eps = 0.2
z = keyhole_contour(R, eps)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot contour
ax.plot(z.real, z.imag, 'b-', linewidth=2)

# Add arrows to show direction
for idx in [100, 350, 600, 850]:
    idx = min(idx, len(z)-2)
    ax.annotate('', xy=(z[idx+1].real, z[idx+1].imag),
               xytext=(z[idx].real, z[idx].imag),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

# Mark branch cut
ax.plot([0, R+0.5], [0, 0], 'r-', linewidth=3, label='Branch cut')

# Mark poles for 1/(1+z)
ax.plot([-1], [0], 'g*', markersize=15, label='Pole at z=-1')

# Shade regions
theta = np.linspace(0, 2*np.pi, 100)
ax.fill(R*np.cos(theta), R*np.sin(theta), alpha=0.1, color='blue')

ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_title('Keyhole Contour for $\\int_0^\\infty \\frac{x^{\\alpha-1}}{1+x} dx$')
ax.legend()
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

plt.savefig('keyhole_contour.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 2: Verifying the Beta Integral

```python
import numpy as np
from scipy import integrate
from scipy.special import gamma
import matplotlib.pyplot as plt

def beta_integral_numerical(alpha):
    """Compute ∫₀^∞ x^(α-1)/(1+x) dx numerically."""
    # Split integral at x=1 for better convergence
    result1, _ = integrate.quad(lambda x: x**(alpha-1)/(1+x), 0, 1)
    result2, _ = integrate.quad(lambda x: x**(alpha-1)/(1+x), 1, np.inf)
    return result1 + result2

def beta_integral_analytical(alpha):
    """Analytical result: π/sin(πα)."""
    return np.pi / np.sin(np.pi * alpha)

# Test for various alpha values
alpha_values = np.linspace(0.1, 0.9, 17)
numerical = [beta_integral_numerical(a) for a in alpha_values]
analytical = [beta_integral_analytical(a) for a in alpha_values]

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(alpha_values, numerical, 'bo-', label='Numerical', markersize=8)
axes[0].plot(alpha_values, analytical, 'r-', label='π/sin(πα)', linewidth=2)
axes[0].set_xlabel('α')
axes[0].set_ylabel('Integral value')
axes[0].set_title('$\\int_0^\\infty \\frac{x^{\\alpha-1}}{1+x} dx = \\frac{\\pi}{\\sin(\\pi\\alpha)}$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Error plot
errors = np.abs(np.array(numerical) - np.array(analytical))
axes[1].semilogy(alpha_values, errors, 'g-', linewidth=2)
axes[1].set_xlabel('α')
axes[1].set_ylabel('|Numerical - Analytical|')
axes[1].set_title('Verification Error')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('beta_integral_verification.png', dpi=150, bbox_inches='tight')
plt.show()

# Print verification
print("Verification of ∫₀^∞ x^(α-1)/(1+x) dx = π/sin(πα)")
print("-" * 55)
for a in [0.25, 0.5, 0.75]:
    num = beta_integral_numerical(a)
    ana = beta_integral_analytical(a)
    print(f"α = {a:.2f}: Numerical = {num:.8f}, Analytical = {ana:.8f}")
```

### Lab 3: Dirichlet Integral and Variations

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def dirichlet_numerical(upper_limit=1000):
    """Compute ∫₀^∞ sin(x)/x dx numerically."""
    result, _ = integrate.quad(lambda x: np.sinc(x/np.pi), 0, upper_limit)
    return result

def si_function(x):
    """Sine integral Si(x) = ∫₀^x sin(t)/t dt."""
    if x == 0:
        return 0
    result, _ = integrate.quad(lambda t: np.sinc(t/np.pi), 0, x)
    return result

# Verify Dirichlet integral
print("Dirichlet Integral Verification")
print("=" * 40)
print(f"∫₀^∞ sin(x)/x dx:")
print(f"  Numerical: {dirichlet_numerical():.10f}")
print(f"  Analytical: π/2 = {np.pi/2:.10f}")

# Plot Si(x) approaching π/2
x_values = np.linspace(0.01, 50, 500)
si_values = [si_function(x) for x in x_values]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(x_values, si_values, 'b-', linewidth=2)
axes[0].axhline(y=np.pi/2, color='r', linestyle='--', label='π/2')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Si(x)')
axes[0].set_title('Sine Integral Si(x) → π/2 as x → ∞')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Variations of Dirichlet integral
# ∫₀^∞ sin(ax)/x dx = π/2 * sign(a)
a_values = np.linspace(-3, 3, 100)
integrals = []
for a in a_values:
    if abs(a) < 0.01:
        integrals.append(0)
    else:
        result, _ = integrate.quad(lambda x: np.sin(a*x)/x, 0.001, 100)
        integrals.append(result)

axes[1].plot(a_values, integrals, 'b-', linewidth=2, label='Numerical')
axes[1].plot(a_values, np.pi/2 * np.sign(a_values), 'r--', linewidth=2, label='(π/2)sign(a)')
axes[1].set_xlabel('a')
axes[1].set_ylabel('∫₀^∞ sin(ax)/x dx')
axes[1].set_title('Generalized Dirichlet: dependence on parameter a')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dirichlet_integral.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 4: Fresnel Integrals

```python
import numpy as np
from scipy import integrate
from scipy.special import fresnel
import matplotlib.pyplot as plt

def fresnel_numerical(upper):
    """Compute Fresnel integrals numerically."""
    C, _ = integrate.quad(lambda t: np.cos(t**2), 0, upper)
    S, _ = integrate.quad(lambda t: np.sin(t**2), 0, upper)
    return C, S

# Compute Fresnel integrals for various upper limits
x = np.linspace(0, 10, 200)
C_values = []
S_values = []

for xi in x:
    C, S = fresnel_numerical(xi)
    C_values.append(C)
    S_values.append(S)

C_values = np.array(C_values)
S_values = np.array(S_values)

# Theoretical limit
limit = 0.5 * np.sqrt(np.pi/2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fresnel C(x)
axes[0, 0].plot(x, C_values, 'b-', linewidth=2)
axes[0, 0].axhline(y=limit, color='r', linestyle='--', label=f'Limit = √(π/8) ≈ {limit:.4f}')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('C(x)')
axes[0, 0].set_title('Fresnel Cosine Integral C(x) = ∫₀^x cos(t²) dt')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Fresnel S(x)
axes[0, 1].plot(x, S_values, 'g-', linewidth=2)
axes[0, 1].axhline(y=limit, color='r', linestyle='--', label=f'Limit = √(π/8) ≈ {limit:.4f}')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('S(x)')
axes[0, 1].set_title('Fresnel Sine Integral S(x) = ∫₀^x sin(t²) dt')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Cornu spiral (parametric plot)
axes[1, 0].plot(C_values, S_values, 'purple', linewidth=2)
axes[1, 0].plot([limit], [limit], 'r*', markersize=15, label='Spiral center')
axes[1, 0].plot([0], [0], 'go', markersize=10, label='Start')
axes[1, 0].set_xlabel('C(x)')
axes[1, 0].set_ylabel('S(x)')
axes[1, 0].set_title('Cornu Spiral (Fresnel Spiral)')
axes[1, 0].legend()
axes[1, 0].axis('equal')
axes[1, 0].grid(True, alpha=0.3)

# Convergence to limit
errors_C = np.abs(C_values - limit)
errors_S = np.abs(S_values - limit)
axes[1, 1].semilogy(x[10:], errors_C[10:], 'b-', linewidth=2, label='|C(x) - limit|')
axes[1, 1].semilogy(x[10:], errors_S[10:], 'g-', linewidth=2, label='|S(x) - limit|')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('Error')
axes[1, 1].set_title('Convergence of Fresnel Integrals to √(π/8)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fresnel_integrals.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFresnel Integral Limits:")
print(f"∫₀^∞ cos(x²) dx = {C_values[-1]:.6f} (theory: {limit:.6f})")
print(f"∫₀^∞ sin(x²) dx = {S_values[-1]:.6f} (theory: {limit:.6f})")
```

---

## Summary

### Key Formulas

| Integral | Method | Result |
|----------|--------|--------|
| $\int_0^\infty \frac{x^{\alpha-1}}{1+x} dx$ | Keyhole | $\frac{\pi}{\sin(\pi\alpha)}$ |
| $\int_0^{2\pi} \frac{d\theta}{a+b\cos\theta}$ | $z = e^{i\theta}$ | $\frac{2\pi}{\sqrt{a^2-b^2}}$ |
| $\int_0^\infty \frac{\sin x}{x} dx$ | Indented | $\frac{\pi}{2}$ |
| $\int_0^\infty \cos(x^2) dx$ | Pie slice | $\frac{1}{2}\sqrt{\frac{\pi}{2}}$ |

### Main Takeaways

1. **Keyhole contours** handle branch cuts by going around them on both sides.

2. **Trigonometric integrals** become algebraic via $z = e^{i\theta}$.

3. **The Dirichlet integral** $\int_0^\infty \frac{\sin x}{x} dx = \frac{\pi}{2}$ is fundamental.

4. **Fresnel integrals** appear in diffraction and converge to $\sqrt{\pi/8}$.

5. **Quantum propagators** with branch cuts are analyzed using these techniques.

---

## Daily Checklist

- [ ] I can set up keyhole contours for branch point integrands
- [ ] I understand how branch cuts affect contributions
- [ ] I can evaluate trigonometric integrals via $z = e^{i\theta}$
- [ ] I can derive the Dirichlet integral
- [ ] I understand indented contours for real axis poles

---

## Preview: Day 181

Tomorrow's computational lab brings together all contour integration techniques:
- Numerical verification of analytical results
- Visualization of various contours
- Physics applications: scattering amplitudes
- Integration challenges and competition problems

---

*"The power of mathematics lies in its ability to find the simple in the complicated."*
— Stanislaw Ulam
