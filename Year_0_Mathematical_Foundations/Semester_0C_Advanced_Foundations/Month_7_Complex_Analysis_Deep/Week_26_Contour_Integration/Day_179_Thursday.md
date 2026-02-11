# Day 179: Applications to Real Integrals I

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Contour Integration Techniques |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Solving: Rational Functions |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 179, you will be able to:

1. Evaluate improper integrals using semicircular contours
2. Apply the residue-like technique for simple poles
3. Handle integrals of rational functions
4. Understand Jordan's lemma for exponential integrands
5. Compute Fourier-type integrals
6. Connect to quantum scattering amplitudes

---

## Core Content

### 1. The Strategy

Complex contour integration transforms difficult real integrals into contour integrals that can be evaluated using Cauchy's theorem.

**General Strategy:**
1. Extend the real integrand to a complex function
2. Choose a closed contour that includes the real axis
3. Apply Cauchy's theorem (or residue theorem)
4. Show that contributions from non-real parts vanish
5. Extract the desired real integral

### 2. Integrals of Rational Functions

#### Type I: $\int_{-\infty}^{\infty} \frac{P(x)}{Q(x)} dx$

**Conditions:**
- $Q(x)$ has no real zeros
- $\deg(Q) \geq \deg(P) + 2$ (ensures convergence)

**Method:** Use a semicircular contour in the upper half-plane.

#### The Semicircular Contour

Let $C_R$ be the contour consisting of:
- $[-R, R]$ on the real axis
- Semicircle $\gamma_R: z = Re^{i\theta}$, $0 \leq \theta \leq \pi$

For large $R$, if $\deg(Q) \geq \deg(P) + 2$:
$$\left|\int_{\gamma_R} \frac{P(z)}{Q(z)} dz\right| \leq \frac{M}{R^2} \cdot \pi R = \frac{\pi M}{R} \to 0$$

So:
$$\oint_{C_R} \frac{P(z)}{Q(z)} dz = \int_{-R}^R \frac{P(x)}{Q(x)} dx + \int_{\gamma_R} \to \int_{-\infty}^\infty \frac{P(x)}{Q(x)} dx$$

By Cauchy's theorem, this equals $2\pi i$ times the sum of residues in the upper half-plane.

### 3. Computing Residues (Preview)

For a simple pole at $z_0$:
$$\text{Res}_{z=z_0} f(z) = \lim_{z \to z_0} (z - z_0) f(z)$$

For $f(z) = \frac{P(z)}{Q(z)}$ with simple zero of $Q$ at $z_0$:
$$\text{Res}_{z=z_0} \frac{P(z)}{Q(z)} = \frac{P(z_0)}{Q'(z_0)}$$

### 4. Example: $\int_{-\infty}^{\infty} \frac{dx}{1+x^2}$

**Step 1:** Extend to complex plane.
$$f(z) = \frac{1}{1+z^2} = \frac{1}{(z+i)(z-i)}$$

**Step 2:** Identify singularities.
Poles at $z = \pm i$. Only $z = i$ is in the upper half-plane.

**Step 3:** Compute the residue at $z = i$.
$$\text{Res}_{z=i} \frac{1}{(z+i)(z-i)} = \lim_{z \to i} \frac{1}{z+i} = \frac{1}{2i}$$

**Step 4:** Apply Cauchy's theorem.
$$\oint_{C_R} \frac{dz}{1+z^2} = 2\pi i \cdot \frac{1}{2i} = \pi$$

**Step 5:** Take $R \to \infty$.
The semicircular arc contribution vanishes, so:
$$\boxed{\int_{-\infty}^{\infty} \frac{dx}{1+x^2} = \pi}$$

**Verification:** This equals $[\arctan x]_{-\infty}^{\infty} = \frac{\pi}{2} - (-\frac{\pi}{2}) = \pi$ ✓

### 5. Example: $\int_{-\infty}^{\infty} \frac{dx}{(1+x^2)^2}$

**Poles:** $z = \pm i$ are poles of order 2.

**Residue at double pole $z = i$:**
$$\text{Res}_{z=i} \frac{1}{(z-i)^2(z+i)^2} = \lim_{z \to i} \frac{d}{dz}\left[\frac{1}{(z+i)^2}\right]$$
$$= \lim_{z \to i} \frac{-2}{(z+i)^3} = \frac{-2}{(2i)^3} = \frac{-2}{-8i} = \frac{1}{4i}$$

**Result:**
$$\int_{-\infty}^{\infty} \frac{dx}{(1+x^2)^2} = 2\pi i \cdot \frac{1}{4i} = \boxed{\frac{\pi}{2}}$$

### 6. Jordan's Lemma

**Lemma (Jordan):** Let $f(z) \to 0$ uniformly as $|z| \to \infty$ in the upper half-plane, and let $a > 0$. Then:
$$\lim_{R \to \infty} \int_{\gamma_R} f(z) e^{iaz} dz = 0$$

where $\gamma_R$ is the upper semicircle $z = Re^{i\theta}$, $0 \leq \theta \leq \pi$.

**Key insight:** $|e^{iaz}| = e^{-a\text{Im}(z)}$ decays in the upper half-plane when $a > 0$.

**Why this works:** On $\gamma_R$:
$$|e^{iaRe^{i\theta}}| = e^{-aR\sin\theta}$$

Since $\sin\theta \geq 0$ for $0 \leq \theta \leq \pi$, this is bounded by 1 and decays for $\theta$ away from 0 and $\pi$.

### 7. Fourier-Type Integrals

#### Type II: $\int_{-\infty}^{\infty} f(x) e^{iax} dx$ (a > 0)

**Method:** Close in the upper half-plane and use Jordan's lemma.

#### Example: $\int_{-\infty}^{\infty} \frac{e^{ix}}{1+x^2} dx$

**Solution:**

Consider $f(z) = \frac{e^{iz}}{1+z^2}$.

Pole at $z = i$ (upper half-plane) with residue:
$$\text{Res}_{z=i} \frac{e^{iz}}{(z-i)(z+i)} = \frac{e^{i \cdot i}}{2i} = \frac{e^{-1}}{2i}$$

By Jordan's lemma, the semicircular arc vanishes.

$$\int_{-\infty}^{\infty} \frac{e^{ix}}{1+x^2} dx = 2\pi i \cdot \frac{e^{-1}}{2i} = \boxed{\frac{\pi}{e}}$$

#### Extracting Real Parts

$$\int_{-\infty}^{\infty} \frac{\cos x}{1+x^2} dx = \text{Re}\left(\frac{\pi}{e}\right) = \frac{\pi}{e}$$

$$\int_{-\infty}^{\infty} \frac{\sin x}{1+x^2} dx = \text{Im}\left(\frac{\pi}{e}\right) = 0$$

(The sine integral is zero by symmetry since the integrand is odd.)

### 8. Integrals with Branch Points

For integrands like $x^\alpha$ where $\alpha$ is not an integer, we need branch cuts.

**Example:** $\int_0^\infty \frac{x^{\alpha-1}}{1+x} dx$ for $0 < \alpha < 1$

This requires a keyhole contour (covered tomorrow).

---

## Quantum Mechanics Connection

### Scattering Amplitudes

In quantum scattering theory, transition amplitudes often involve integrals of the form:
$$\mathcal{A} = \int_{-\infty}^{\infty} \frac{e^{ikr}}{k^2 - k_0^2 + i\varepsilon} dk$$

The $+i\varepsilon$ prescription places poles slightly into the complex plane:
- $k = \sqrt{k_0^2 - i\varepsilon} \approx k_0 - i\varepsilon/(2k_0)$
- $k = -\sqrt{k_0^2 - i\varepsilon} \approx -k_0 + i\varepsilon/(2k_0)$

Choosing the upper or lower half-plane for closure gives retarded or advanced Green's functions.

### Density of States

The density of states in a quantum system:
$$\rho(E) = -\frac{1}{\pi} \text{Im}\, \text{Tr}\, G(E + i\varepsilon)$$

This involves taking imaginary parts of contour integrals — exactly the techniques we're developing.

### Kramers-Kronig Relations

For a causal response function $\chi(\omega)$:
$$\text{Re}\,\chi(\omega) = \frac{1}{\pi} \mathcal{P}\int_{-\infty}^{\infty} \frac{\text{Im}\,\chi(\omega')}{\omega' - \omega} d\omega'$$

These dispersion relations follow from contour integration, connecting absorption (Im$\chi$) to dispersion (Re$\chi$).

---

## Worked Examples

### Example 1: Higher-Degree Denominator

**Problem:** Evaluate $\int_{-\infty}^{\infty} \frac{dx}{x^4 + 1}$.

**Solution:**

**Step 1:** Factor $z^4 + 1 = 0$ ⟹ $z^4 = -1 = e^{i\pi}$

The four roots are: $z_k = e^{i(\pi + 2\pi k)/4}$ for $k = 0, 1, 2, 3$

$$z_0 = e^{i\pi/4} = \frac{1+i}{\sqrt{2}}, \quad z_1 = e^{3i\pi/4} = \frac{-1+i}{\sqrt{2}}$$
$$z_2 = e^{5i\pi/4} = \frac{-1-i}{\sqrt{2}}, \quad z_3 = e^{7i\pi/4} = \frac{1-i}{\sqrt{2}}$$

**Step 2:** Poles in upper half-plane: $z_0$ and $z_1$.

**Step 3:** Compute residues.

For simple poles of $\frac{1}{z^4+1}$:
$$\text{Res}_{z=z_k} = \frac{1}{4z_k^3} = \frac{z_k}{4z_k^4} = \frac{z_k}{4(-1)} = -\frac{z_k}{4}$$

At $z_0$: Res $= -\frac{e^{i\pi/4}}{4} = -\frac{1+i}{4\sqrt{2}}$

At $z_1$: Res $= -\frac{e^{3i\pi/4}}{4} = -\frac{-1+i}{4\sqrt{2}}$

**Step 4:** Sum of residues:
$$\text{Res}_{z_0} + \text{Res}_{z_1} = -\frac{1}{4\sqrt{2}}[(1+i) + (-1+i)] = -\frac{2i}{4\sqrt{2}} = -\frac{i}{2\sqrt{2}}$$

**Step 5:** Result:
$$\int_{-\infty}^{\infty} \frac{dx}{x^4+1} = 2\pi i \cdot \left(-\frac{i}{2\sqrt{2}}\right) = \frac{2\pi}{2\sqrt{2}} = \boxed{\frac{\pi}{\sqrt{2}}}$$

### Example 2: Fourier Transform

**Problem:** Compute $\int_{-\infty}^{\infty} \frac{\cos(2x)}{x^2 + 4} dx$.

**Solution:**

Consider $\int_{-\infty}^{\infty} \frac{e^{2ix}}{x^2+4} dx$.

Pole at $z = 2i$ (upper half-plane) with residue:
$$\text{Res}_{z=2i} \frac{e^{2iz}}{(z-2i)(z+2i)} = \frac{e^{2i(2i)}}{4i} = \frac{e^{-4}}{4i}$$

By Jordan's lemma ($a = 2 > 0$), semicircle vanishes.

$$\int_{-\infty}^{\infty} \frac{e^{2ix}}{x^2+4} dx = 2\pi i \cdot \frac{e^{-4}}{4i} = \frac{\pi e^{-4}}{2}$$

Taking the real part:
$$\int_{-\infty}^{\infty} \frac{\cos(2x)}{x^2+4} dx = \boxed{\frac{\pi e^{-4}}{2}} \approx 0.0288$$

### Example 3: Negative Exponential Factor

**Problem:** Evaluate $\int_{-\infty}^{\infty} \frac{e^{-3ix}}{x^2 + 1} dx$.

**Solution:**

Here $a = -3 < 0$, so we close in the **lower** half-plane where $e^{-3iz}$ decays.

Pole at $z = -i$ (lower half-plane) with residue:
$$\text{Res}_{z=-i} \frac{e^{-3iz}}{(z-i)(z+i)} = \frac{e^{-3i(-i)}}{-2i} = \frac{e^{-3}}{-2i}$$

**Note:** Traversing the lower semicircle clockwise gives a factor of $-2\pi i$:
$$\int_{-\infty}^{\infty} \frac{e^{-3ix}}{x^2+1} dx = -2\pi i \cdot \frac{e^{-3}}{-2i} = \pi e^{-3}$$

### Example 4: Principal Value Integrals

**Problem:** Evaluate $\mathcal{P}\int_{-\infty}^{\infty} \frac{dx}{x^2 - 1}$ where $\mathcal{P}$ denotes principal value.

**Solution:**

The integrand has poles on the real axis at $x = \pm 1$.

**Principal value definition:**
$$\mathcal{P}\int_{-\infty}^{\infty} = \lim_{\varepsilon \to 0} \left[\int_{-\infty}^{-1-\varepsilon} + \int_{-1+\varepsilon}^{1-\varepsilon} + \int_{1+\varepsilon}^{\infty}\right]$$

**Complex method:** Indent the contour with small semicircles around the poles.

Each pole on the real axis contributes $-\pi i \cdot \text{Res}$ (for upper semicircle indentation).

$$\frac{1}{x^2-1} = \frac{1}{(x-1)(x+1)} = \frac{1}{2}\left(\frac{1}{x-1} - \frac{1}{x+1}\right)$$

Residue at $x = 1$: $\frac{1}{2}$
Residue at $x = -1$: $-\frac{1}{2}$

Since the integrand is even and the residues sum to zero:
$$\mathcal{P}\int_{-\infty}^{\infty} \frac{dx}{x^2-1} = 0$$

---

## Practice Problems

### Problem Set A: Basic Rational Functions

**A1.** Evaluate $\int_{-\infty}^{\infty} \frac{dx}{x^2 + 4}$.

**A2.** Compute $\int_{-\infty}^{\infty} \frac{x^2}{(x^2+1)(x^2+4)} dx$.

**A3.** Find $\int_0^{\infty} \frac{dx}{x^4 + 1}$ (use symmetry and your result from Example 1).

### Problem Set B: Fourier-Type Integrals

**B1.** Evaluate $\int_{-\infty}^{\infty} \frac{e^{i\omega x}}{x^2 + a^2} dx$ for $\omega > 0$, $a > 0$.

**B2.** Compute $\int_0^{\infty} \frac{\cos x}{x^2 + 1} dx$.

**B3.** Find $\int_{-\infty}^{\infty} \frac{x \sin x}{x^2 + 4} dx$.

### Problem Set C: Advanced

**C1.** Evaluate $\int_{-\infty}^{\infty} \frac{dx}{(x^2+1)^3}$.

**C2.** Compute $\int_{-\infty}^{\infty} \frac{e^{i\omega x}}{(x-i)^2} dx$ for $\omega > 0$.

**C3.** Show that $\int_0^{\infty} \frac{dx}{1+x^n} = \frac{\pi/n}{\sin(\pi/n)}$ for integer $n \geq 2$.

---

## Solutions to Selected Problems

### Solution A1

$f(z) = \frac{1}{z^2+4} = \frac{1}{(z-2i)(z+2i)}$

Pole at $z = 2i$ in upper half-plane.

Residue: $\frac{1}{4i}$

$$\int_{-\infty}^{\infty} \frac{dx}{x^2+4} = 2\pi i \cdot \frac{1}{4i} = \frac{\pi}{2}$$

### Solution A3

By Example 1: $\int_{-\infty}^{\infty} \frac{dx}{x^4+1} = \frac{\pi}{\sqrt{2}}$

Since the integrand is even:
$$\int_0^{\infty} \frac{dx}{x^4+1} = \frac{1}{2} \cdot \frac{\pi}{\sqrt{2}} = \frac{\pi}{2\sqrt{2}} = \frac{\pi\sqrt{2}}{4}$$

### Solution B1

Pole at $z = ia$ in upper half-plane.

$$\text{Res}_{z=ia} \frac{e^{i\omega z}}{(z-ia)(z+ia)} = \frac{e^{i\omega(ia)}}{2ia} = \frac{e^{-\omega a}}{2ia}$$

$$\int_{-\infty}^{\infty} \frac{e^{i\omega x}}{x^2+a^2} dx = 2\pi i \cdot \frac{e^{-\omega a}}{2ia} = \frac{\pi e^{-\omega a}}{a}$$

---

## Computational Lab

### Lab 1: Verifying Contour Integration Results

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def verify_integral(integrand, expected, name, limits=(-np.inf, np.inf)):
    """Verify contour integration result numerically."""
    result, error = integrate.quad(integrand, limits[0], limits[1])
    print(f"{name}:")
    print(f"  Numerical: {result:.10f}")
    print(f"  Expected:  {expected:.10f}")
    print(f"  Error:     {abs(result - expected):.2e}")
    print()
    return result

print("Verifying Contour Integration Results")
print("=" * 50)

# Example 1: ∫ dx/(1+x²) = π
verify_integral(lambda x: 1/(1+x**2), np.pi, "∫ dx/(1+x²)")

# Example 2: ∫ dx/(1+x²)² = π/2
verify_integral(lambda x: 1/(1+x**2)**2, np.pi/2, "∫ dx/(1+x²)²")

# Example 3: ∫ dx/(x⁴+1) = π/√2
verify_integral(lambda x: 1/(x**4+1), np.pi/np.sqrt(2), "∫ dx/(x⁴+1)")

# Example 4: ∫ cos(x)/(x²+1) dx = π/e
verify_integral(lambda x: np.cos(x)/(x**2+1), np.pi/np.e, "∫ cos(x)/(x²+1) dx")

# Example 5: ∫ cos(2x)/(x²+4) dx = πe⁻⁴/2
verify_integral(lambda x: np.cos(2*x)/(x**2+4), np.pi*np.exp(-4)/2,
               "∫ cos(2x)/(x²+4) dx")
```

### Lab 2: Visualizing the Contour

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_contour_and_poles(poles, title, R=3):
    """Visualize semicircular contour and poles."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Real axis segment
    x_real = np.linspace(-R, R, 100)
    ax.plot(x_real, np.zeros_like(x_real), 'b-', linewidth=2, label='Real axis')
    ax.arrow(0, 0, R-0.5, 0, head_width=0.1, head_length=0.1, fc='b', ec='b')

    # Semicircular arc
    theta = np.linspace(0, np.pi, 100)
    x_arc = R * np.cos(theta)
    y_arc = R * np.sin(theta)
    ax.plot(x_arc, y_arc, 'b-', linewidth=2, label='Semicircle γR')
    ax.arrow(R*np.cos(np.pi/2+0.1), R*np.sin(np.pi/2+0.1),
            -R*0.1*np.sin(np.pi/2+0.1), R*0.1*np.cos(np.pi/2+0.1),
            head_width=0.1, head_length=0.1, fc='b', ec='b')

    # Plot poles
    for i, pole in enumerate(poles):
        color = 'r' if pole.imag > 0 else 'g'
        marker = '*' if pole.imag > 0 else 'o'
        label = 'Upper HP pole' if i == 0 and pole.imag > 0 else \
                ('Lower HP pole' if i == 0 or (i > 0 and poles[i-1].imag > 0 and pole.imag < 0) else '')
        ax.plot(pole.real, pole.imag, marker, markersize=15, color=color, label=label if label else None)

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(title)
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-R-1, R+1)
    ax.set_ylim(-R-1, R+1)

    # Shade upper half-plane
    ax.fill_between(x_real, 0, R, alpha=0.1, color='blue')

    return fig, ax

# Example: 1/(z²+1) with poles at ±i
fig1, ax1 = plot_contour_and_poles([1j, -1j], "Contour for ∫ dz/(z²+1)")
plt.savefig('contour_1_over_z2plus1.png', dpi=150, bbox_inches='tight')

# Example: 1/(z⁴+1) with 4 poles
poles_z4 = [np.exp(1j*np.pi*(2*k+1)/4) for k in range(4)]
fig2, ax2 = plot_contour_and_poles(poles_z4, "Contour for ∫ dz/(z⁴+1)")
plt.savefig('contour_1_over_z4plus1.png', dpi=150, bbox_inches='tight')

plt.show()
```

### Lab 3: Jordan's Lemma Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

def jordan_lemma_demo(f, a, R_values):
    """
    Demonstrate Jordan's lemma: ∫_γR f(z)e^{iaz} dz → 0 as R → ∞
    for a > 0 and f(z) → 0 as |z| → ∞.
    """
    integrals = []

    for R in R_values:
        # Parametrize upper semicircle
        theta = np.linspace(0, np.pi, 1000)
        z = R * np.exp(1j * theta)
        dz = 1j * R * np.exp(1j * theta)

        # Integrand
        integrand = f(z) * np.exp(1j * a * z) * dz

        # Integrate
        integral = np.trapz(integrand, theta)
        integrals.append(np.abs(integral))

    return integrals

# Test function: f(z) = 1/(z² + 1)
f = lambda z: 1/(z**2 + 1)
a = 1

R_values = np.linspace(1, 50, 100)
integrals = jordan_lemma_demo(f, a, R_values)

plt.figure(figsize=(10, 6))
plt.semilogy(R_values, integrals, 'b-', linewidth=2)
plt.xlabel('Radius R')
plt.ylabel('|∫_γR f(z)e^{iz} dz|')
plt.title("Jordan's Lemma: Semicircular Integral → 0 as R → ∞")
plt.grid(True, alpha=0.3)
plt.savefig('jordan_lemma.png', dpi=150, bbox_inches='tight')
plt.show()

print("Jordan's Lemma Verification:")
print(f"R = {R_values[-1]:.0f}: |∫| = {integrals[-1]:.2e}")
```

### Lab 4: Fourier Transform via Contour Integration

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def fourier_transform_contour(a, omega_values):
    """
    Compute Fourier transform of 1/(x² + a²) using contour integration:
    F(ω) = ∫ e^{iωx}/(x² + a²) dx = (π/a)e^{-a|ω|}
    """
    results = []
    for omega in omega_values:
        if omega >= 0:
            # Close in upper HP, pole at z = ia
            result = np.pi * np.exp(-a * omega) / a
        else:
            # Close in lower HP, pole at z = -ia
            result = np.pi * np.exp(a * omega) / a
        results.append(result)
    return np.array(results)

def fourier_transform_numerical(a, omega_values):
    """Numerical Fourier transform for verification."""
    results = []
    for omega in omega_values:
        # Split into real and imaginary parts
        real_part, _ = integrate.quad(
            lambda x: np.cos(omega * x) / (x**2 + a**2), -np.inf, np.inf)
        imag_part, _ = integrate.quad(
            lambda x: np.sin(omega * x) / (x**2 + a**2), -np.inf, np.inf)
        results.append(real_part + 1j * imag_part)
    return np.array(results)

# Parameters
a = 2
omega = np.linspace(-5, 5, 100)

# Compute
analytical = fourier_transform_contour(a, omega)
numerical = fourier_transform_numerical(a, omega)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Real part comparison
axes[0, 0].plot(omega, analytical.real, 'b-', linewidth=2, label='Analytical')
axes[0, 0].plot(omega, numerical.real, 'r--', linewidth=2, label='Numerical')
axes[0, 0].set_xlabel('ω')
axes[0, 0].set_ylabel('Re[F(ω)]')
axes[0, 0].set_title(f'Real Part of Fourier Transform (a = {a})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Imaginary part (should be ~0)
axes[0, 1].plot(omega, numerical.imag, 'g-', linewidth=2)
axes[0, 1].set_xlabel('ω')
axes[0, 1].set_ylabel('Im[F(ω)]')
axes[0, 1].set_title('Imaginary Part (should be ~0)')
axes[0, 1].grid(True, alpha=0.3)

# Original function
x = np.linspace(-5, 5, 200)
axes[1, 0].plot(x, 1/(x**2 + a**2), 'b-', linewidth=2)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('f(x)')
axes[1, 0].set_title(f'Original function: f(x) = 1/(x² + {a}²)')
axes[1, 0].grid(True, alpha=0.3)

# Theoretical result
axes[1, 1].plot(omega, np.pi/a * np.exp(-a * np.abs(omega)), 'b-', linewidth=2)
axes[1, 1].set_xlabel('ω')
axes[1, 1].set_ylabel('F(ω)')
axes[1, 1].set_title(f'Analytical: F(ω) = (π/{a})exp(-{a}|ω|)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_contour.png', dpi=150, bbox_inches='tight')
plt.show()

# Print verification
print("\nNumerical Verification at Selected Points:")
print("-" * 50)
for w in [0, 1, 2, -1]:
    idx = np.argmin(np.abs(omega - w))
    print(f"ω = {w:3}: Analytical = {analytical[idx]:.6f}, "
          f"Numerical = {numerical[idx].real:.6f}")
```

---

## Summary

### Key Formulas

| Integral Type | Method | Result |
|---------------|--------|--------|
| $\int_{-\infty}^{\infty} \frac{P(x)}{Q(x)} dx$ | Semicircle, $\deg Q \geq \deg P + 2$ | $2\pi i \sum \text{Res}$ (UHP) |
| $\int_{-\infty}^{\infty} f(x) e^{iax} dx$, $a > 0$ | Jordan's lemma, close in UHP | $2\pi i \sum \text{Res}$ (UHP) |
| $\int_{-\infty}^{\infty} f(x) e^{iax} dx$, $a < 0$ | Close in LHP | $-2\pi i \sum \text{Res}$ (LHP) |

### Main Takeaways

1. **Semicircular contours** transform real integrals into contour integrals.

2. **Degree condition** $\deg Q \geq \deg P + 2$ ensures vanishing arc contribution.

3. **Jordan's lemma** handles exponential factors by exploiting decay in half-planes.

4. **Close in the appropriate half-plane** based on the sign of the exponential argument.

5. **Quantum scattering amplitudes** are computed using exactly these techniques.

---

## Daily Checklist

- [ ] I can set up semicircular contours for rational integrands
- [ ] I can compute residues at simple poles
- [ ] I understand Jordan's lemma and when to apply it
- [ ] I can evaluate Fourier-type integrals
- [ ] I see the connection to quantum Green's functions

---

## Preview: Day 180

Tomorrow we continue with more advanced applications:
- Integrals involving branch cuts
- Keyhole contours for $x^\alpha$ factors
- Trigonometric integrals
- The Dirichlet integral: $\int_0^\infty \frac{\sin x}{x} dx = \frac{\pi}{2}$

---

*"In the theory of analytic functions of a complex variable, Cauchy's formula is one of the most remarkable and useful theorems."*
— Lars Ahlfors
