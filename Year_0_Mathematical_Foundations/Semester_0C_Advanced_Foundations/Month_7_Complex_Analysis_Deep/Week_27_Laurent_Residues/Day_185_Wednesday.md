# Day 185: Residue Computation Techniques

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Residue Formulas |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Advanced Techniques & Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 185, you will be able to:

1. Compute residues at simple poles using the limit formula
2. Apply the derivative formula for higher-order poles
3. Use L'Hopital's technique for rational functions
4. Calculate residues at infinity
5. Handle logarithmic and algebraic branch points
6. Connect residue calculations to quantum scattering amplitudes

---

## Core Content

### 1. Definition Recap: The Residue

The **residue** of $f$ at $z_0$ is the coefficient $a_{-1}$ in the Laurent series:

$$\boxed{\text{Res}_{z=z_0} f(z) = a_{-1} = \frac{1}{2\pi i}\oint_C f(z)\, dz}$$

where $C$ is any small circle around $z_0$.

The residue is the **only coefficient that contributes to contour integrals** around $z_0$!

### 2. Simple Pole Formula

**Theorem:** If $f$ has a **simple pole** at $z_0$, then:

$$\boxed{\text{Res}_{z=z_0} f(z) = \lim_{z \to z_0} (z - z_0) f(z)}$$

**Proof:**
Near a simple pole: $f(z) = \frac{a_{-1}}{z - z_0} + a_0 + a_1(z - z_0) + \cdots$

$(z - z_0)f(z) = a_{-1} + a_0(z - z_0) + \cdots$

Taking $z \to z_0$: $(z - z_0)f(z) \to a_{-1}$ $\blacksquare$

**Example 1:** $\text{Res}_{z=1} \frac{e^z}{z-1}$

$$\text{Res} = \lim_{z \to 1}(z-1) \cdot \frac{e^z}{z-1} = \lim_{z \to 1} e^z = e$$

**Example 2:** $\text{Res}_{z=i} \frac{1}{z^2 + 1}$

Factor: $\frac{1}{z^2+1} = \frac{1}{(z-i)(z+i)}$

$$\text{Res}_{z=i} = \lim_{z \to i}(z-i) \cdot \frac{1}{(z-i)(z+i)} = \frac{1}{2i} = -\frac{i}{2}$$

### 3. L'Hopital Technique for Rational Functions

**Theorem:** If $f(z) = \frac{P(z)}{Q(z)}$ where $P(z_0) \neq 0$ and $Q$ has a simple zero at $z_0$:

$$\boxed{\text{Res}_{z=z_0} \frac{P(z)}{Q(z)} = \frac{P(z_0)}{Q'(z_0)}}$$

**Proof:**
$$\text{Res} = \lim_{z \to z_0}(z - z_0)\frac{P(z)}{Q(z)} = \lim_{z \to z_0}\frac{P(z)}{\frac{Q(z) - Q(z_0)}{z - z_0}} = \frac{P(z_0)}{Q'(z_0)}$$

This is essentially L'Hopital's rule! $\blacksquare$

**Example 3:** $\text{Res}_{z=\pi} \frac{z}{\sin z}$

$P(z) = z$, $Q(z) = \sin z$, $P(\pi) = \pi$, $Q'(\pi) = \cos\pi = -1$

$$\text{Res} = \frac{\pi}{-1} = -\pi$$

**Example 4:** $\text{Res}_{z=n\pi} \frac{1}{\sin z}$ for integer $n$

$$\text{Res} = \frac{1}{\cos(n\pi)} = \frac{1}{(-1)^n} = (-1)^n$$

### 4. Higher-Order Pole Formula

**Theorem:** If $f$ has a **pole of order $m$** at $z_0$, then:

$$\boxed{\text{Res}_{z=z_0} f(z) = \frac{1}{(m-1)!}\lim_{z \to z_0} \frac{d^{m-1}}{dz^{m-1}}\left[(z-z_0)^m f(z)\right]}$$

**Proof:**
Let $g(z) = (z - z_0)^m f(z)$, which is analytic at $z_0$.

Laurent series: $f(z) = \frac{a_{-m}}{(z-z_0)^m} + \cdots + \frac{a_{-1}}{z-z_0} + \cdots$

So: $g(z) = a_{-m} + a_{-m+1}(z-z_0) + \cdots + a_{-1}(z-z_0)^{m-1} + \cdots$

Taylor coefficient: $a_{-1} = \frac{g^{(m-1)}(z_0)}{(m-1)!}$ $\blacksquare$

**Example 5:** $\text{Res}_{z=0} \frac{e^z}{z^3}$

Order 3 pole, so $m = 3$:

$g(z) = z^3 \cdot \frac{e^z}{z^3} = e^z$

$$\text{Res} = \frac{1}{2!}\lim_{z \to 0}\frac{d^2}{dz^2}e^z = \frac{1}{2}e^0 = \frac{1}{2}$$

**Example 6:** $\text{Res}_{z=1} \frac{z^2}{(z-1)^2}$

Order 2 pole, $m = 2$:

$g(z) = (z-1)^2 \cdot \frac{z^2}{(z-1)^2} = z^2$

$$\text{Res} = \frac{1}{1!}\frac{d}{dz}z^2 \bigg|_{z=1} = 2z\bigg|_{z=1} = 2$$

### 5. Alternative: Partial Fractions / Laurent Expansion

For some problems, **direct Laurent expansion** is easier than the formula.

**Example 7:** $\text{Res}_{z=0} \frac{\sin z}{z^4}$

Expand: $\sin z = z - \frac{z^3}{6} + \frac{z^5}{120} - \cdots$

$$\frac{\sin z}{z^4} = \frac{1}{z^3} - \frac{1}{6z} + \frac{z}{120} - \cdots$$

Read off coefficient of $\frac{1}{z}$: $\text{Res} = -\frac{1}{6}$

**Example 8:** $\text{Res}_{z=0} \frac{1}{z(e^z - 1)}$

Need Laurent series of $\frac{1}{e^z - 1}$ at $z = 0$:

$e^z - 1 = z + \frac{z^2}{2} + \frac{z^3}{6} + \cdots = z(1 + \frac{z}{2} + \frac{z^2}{6} + \cdots)$

$\frac{1}{e^z - 1} = \frac{1}{z}\cdot\frac{1}{1 + z/2 + \cdots} = \frac{1}{z}(1 - \frac{z}{2} + \frac{z^2}{12} - \cdots)$

$= \frac{1}{z} - \frac{1}{2} + \frac{z}{12} - \cdots$

So: $\frac{1}{z(e^z - 1)} = \frac{1}{z^2} - \frac{1}{2z} + \frac{1}{12} - \cdots$

$$\text{Res}_{z=0} = -\frac{1}{2}$$

### 6. Residue at Infinity

**Definition:** The residue at infinity is defined as:

$$\boxed{\text{Res}_{z=\infty} f(z) = -\text{Res}_{w=0} \frac{1}{w^2}f\left(\frac{1}{w}\right)}$$

**Alternative formula:**

$$\text{Res}_{z=\infty} f(z) = -\frac{1}{2\pi i}\oint_{|z|=R} f(z)\, dz$$

where the integral is taken **clockwise** (negative orientation).

**Key Result:** For a function meromorphic in the extended plane:

$$\boxed{\sum_{\text{all finite poles}} \text{Res}_{z=z_k} f(z) + \text{Res}_{z=\infty} f(z) = 0}$$

**Example 9:** $f(z) = \frac{z}{z^2 + 1}$

Finite poles at $z = \pm i$:
- $\text{Res}_{z=i} = \frac{i}{2i} = \frac{1}{2}$
- $\text{Res}_{z=-i} = \frac{-i}{-2i} = \frac{1}{2}$

Sum = 1

For infinity: $g(w) = \frac{1}{w^2} \cdot \frac{1/w}{1/w^2 + 1} = \frac{1}{w^2} \cdot \frac{w}{1 + w^2} = \frac{1}{w(1+w^2)}$

$\text{Res}_{w=0} g(w) = \lim_{w \to 0} w \cdot \frac{1}{w(1+w^2)} = 1$

$\text{Res}_{z=\infty} f(z) = -1$

Check: $\frac{1}{2} + \frac{1}{2} + (-1) = 0$ $\checkmark$

### 7. Residues for Specific Function Types

**Logarithmic Functions:**

For $f(z) = \frac{\ln z}{g(z)}$ where $g$ has simple zeros, use the standard formula.

**Example 10:** $\text{Res}_{z=1} \frac{\ln z}{z - 1}$

This is actually a branch point, not a pole! The function has no residue in the usual sense.

**But:** $\text{Res}_{z=-1} \frac{\ln z}{z^2 - 1}$

$$= \frac{\ln(-1)}{(-1) - 1} = \frac{i\pi}{-2} = -\frac{i\pi}{2}$$

**Algebraic Branch Points:**

For $f(z) = z^{\alpha}g(z)$ with $\alpha$ non-integer, use contour methods directly.

### 8. Computational Strategy Summary

| Singularity Type | Method |
|-----------------|--------|
| Simple pole of $P/Q$ | $\text{Res} = P(z_0)/Q'(z_0)$ |
| Simple pole, general | $\text{Res} = \lim_{z \to z_0}(z-z_0)f(z)$ |
| Pole of order $m$ | Derivative formula |
| Complicated function | Laurent series expansion |
| Residue at infinity | Transform $w = 1/z$ |

---

## Quantum Mechanics Connection

### Computing Scattering Amplitudes

In quantum scattering, the T-matrix (related to the scattering amplitude) can be written:

$$T(E) = V + V G_0(E) T(E)$$

where $G_0(E) = (E - H_0 + i\varepsilon)^{-1}$.

**Near a Resonance:**

The T-matrix near a resonance at $E = E_R - i\Gamma/2$ has the form:

$$T(E) \approx \frac{\gamma}{E - E_R + i\Gamma/2} + T_{\text{bg}}$$

The **residue** at the pole gives:
- $|\gamma|^2$ related to **partial widths** (coupling to channels)
- The pole position gives energy and lifetime

### Propagator Residues

The Feynman propagator for a particle of mass $m$:

$$G(p) = \frac{1}{p^2 - m^2 + i\varepsilon}$$

**Poles at:** $p^0 = \pm\sqrt{|\mathbf{p}|^2 + m^2} \mp i\varepsilon$

**Residue at positive energy pole:**

$$\text{Res}_{p^0 = E_p} G(p) = \lim_{p^0 \to E_p}(p^0 - E_p)\frac{1}{(p^0)^2 - E_p^2} = \frac{1}{2E_p}$$

This gives the **normalization factor** in the propagator: $\frac{1}{2E_p}$.

### Sum Rules from Residues

In QM, **sum rules** often arise from closing contours:

$$\sum_n \frac{|\langle n|A|0\rangle|^2}{E_n - E_0} = \text{(contour integral at infinity)}$$

The residue at each pole $E_n$ is the matrix element squared!

---

## Worked Examples

### Example 1: Multiple Techniques

**Problem:** Find all residues of $f(z) = \frac{z^2}{(z^2 + 1)(z - 2)}$.

**Solution:**

Poles at $z = i, -i, 2$.

**At $z = i$ (simple):**
$$\text{Res}_{z=i} = \frac{i^2}{(i+i)(i-2)} = \frac{-1}{2i(i-2)} = \frac{-1}{2i \cdot i - 4i} = \frac{-1}{-2 - 4i}$$
$$= \frac{1}{2 + 4i} = \frac{2-4i}{(2+4i)(2-4i)} = \frac{2-4i}{20} = \frac{1-2i}{10}$$

**At $z = -i$ (simple):**
$$\text{Res}_{z=-i} = \frac{(-i)^2}{(-i-i)(-i-2)} = \frac{-1}{-2i(-i-2)} = \frac{-1}{-2i^2 - 4i}$$
$$= \frac{-1}{2 - 4i} = \frac{-1(2+4i)}{20} = \frac{-2-4i}{20} = \frac{-1-2i}{10}$$

**At $z = 2$ (simple):**
$$\text{Res}_{z=2} = \frac{4}{(4+1)(2-2)} \cdot \lim_{z\to 2}(z-2)$$

Using L'Hopital: $Q(z) = (z^2+1)(z-2)$, $Q'(z) = 2z(z-2) + (z^2+1)$

$Q'(2) = 0 + 5 = 5$

$$\text{Res}_{z=2} = \frac{4}{5}$$

**Check:** Sum should equal residue at $\infty$ (negated).

### Example 2: Higher-Order Pole

**Problem:** $\text{Res}_{z=0} \frac{\cosh z - 1}{z^4}$

**Solution:**

Expand $\cosh z = 1 + \frac{z^2}{2!} + \frac{z^4}{4!} + \cdots$

$$\cosh z - 1 = \frac{z^2}{2} + \frac{z^4}{24} + \cdots$$

$$\frac{\cosh z - 1}{z^4} = \frac{1}{2z^2} + \frac{1}{24} + \cdots$$

There's no $1/z$ term, so $\text{Res}_{z=0} = 0$.

**Alternatively:** This is a pole of order 2 (not 4!).

$g(z) = z^2 \cdot \frac{\cosh z - 1}{z^4} = \frac{\cosh z - 1}{z^2}$

$$\text{Res} = \lim_{z \to 0}\frac{d}{dz}\left(\frac{\cosh z - 1}{z^2}\right)$$

Using L'Hopital twice on $g(z)$ as $z \to 0$... this confirms $\text{Res} = 0$.

### Example 3: Essential Singularity Residue

**Problem:** $\text{Res}_{z=0} e^{1/z}$

**Solution:**

$$e^{1/z} = \sum_{n=0}^{\infty} \frac{1}{n! z^n} = 1 + \frac{1}{z} + \frac{1}{2!z^2} + \cdots$$

The coefficient of $1/z$ is $a_{-1} = 1$.

$$\text{Res}_{z=0} e^{1/z} = 1$$

### Example 4: Residue at Infinity

**Problem:** Find $\text{Res}_{z=\infty} \frac{z^3}{z^2 - 1}$.

**Solution:**

Let $w = 1/z$:

$$g(w) = \frac{1}{w^2} f(1/w) = \frac{1}{w^2} \cdot \frac{1/w^3}{1/w^2 - 1} = \frac{1}{w^2} \cdot \frac{1}{w^3(1/w^2 - 1)}$$
$$= \frac{1}{w^2} \cdot \frac{1}{w - w^3} = \frac{1}{w^3(1 - w^2)}$$

This has a pole of order 3 at $w = 0$.

$h(w) = w^3 g(w) = \frac{1}{1 - w^2} = 1 + w^2 + w^4 + \cdots$

$$\text{Res}_{w=0} g(w) = \frac{1}{2!}h''(0) = \frac{1}{2} \cdot 2 = 1$$

$$\text{Res}_{z=\infty} f(z) = -1$$

---

## Practice Problems

### Problem Set A: Simple Poles

**A1.** Compute the residue at each pole:
(a) $\frac{z}{z^2 - 4}$ at $z = 2$ and $z = -2$
(b) $\frac{e^z}{z^2 + \pi^2}$ at $z = i\pi$
(c) $\frac{z^3}{z^4 - 1}$ at each pole

**A2.** Find: $\text{Res}_{z=n\pi} \cot z$ for integer $n$.

**A3.** Compute: $\text{Res}_{z=0} \frac{z}{e^z - 1}$

### Problem Set B: Higher-Order Poles

**B1.** Find the residue:
(a) $\frac{e^z}{(z-1)^2}$ at $z = 1$
(b) $\frac{\sin z}{z^5}$ at $z = 0$
(c) $\frac{z}{(z^2+1)^2}$ at $z = i$

**B2.** Compute: $\text{Res}_{z=0} \frac{z - \sin z}{z^5}$

**B3.** Find: $\text{Res}_{z=1} \frac{z^n}{(z-1)^3}$ as a function of $n$.

### Problem Set C: Advanced Techniques

**C1.** Show: $\text{Res}_{z=0} \frac{1}{z(e^z - 1)} = -\frac{1}{2}$

**C2.** Find: $\text{Res}_{z=\infty} \frac{z^2}{(z-1)(z-2)}$

**C3.** For $f(z) = \frac{1}{z^4 + 1}$, find the sum of all residues at finite poles.

---

## Computational Lab

```python
import numpy as np
import sympy as sp
from scipy.misc import derivative
import matplotlib.pyplot as plt

class ResidueCalculator:
    """
    Comprehensive residue computation using multiple methods.
    """

    def __init__(self):
        self.z = sp.Symbol('z')

    def residue_symbolic(self, expr, point):
        """
        Compute residue symbolically using SymPy.
        """
        return sp.residue(expr, self.z, point)

    def residue_simple_pole_numeric(self, f, z0, epsilon=1e-8):
        """
        Numerical residue for simple pole: lim (z-z0)*f(z)
        """
        # Sample from multiple directions
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        values = []

        for theta in angles:
            z = z0 + epsilon * np.exp(1j * theta)
            val = (z - z0) * f(z)
            values.append(val)

        return np.mean(values)

    def residue_contour_numeric(self, f, z0, radius=0.1, n_points=10000):
        """
        Compute residue via contour integration.
        """
        theta = np.linspace(0, 2*np.pi, n_points)
        z = z0 + radius * np.exp(1j * theta)
        dz = 1j * radius * np.exp(1j * theta)

        integrand = f(z) * dz
        integral = np.trapz(integrand, theta)

        return integral / (2 * np.pi * 1j)

    def residue_higher_order_numeric(self, f, z0, order, epsilon=1e-6):
        """
        Numerical residue for pole of order m using derivative formula.
        """
        def g_real(x):
            z = complex(x, epsilon)
            return ((z - z0)**order * f(z)).real

        def g_imag(x):
            z = complex(x, epsilon)
            return ((z - z0)**order * f(z)).imag

        # Compute (m-1)th derivative at z0
        deriv_real = derivative(g_real, z0.real, n=order-1, dx=1e-5)
        deriv_imag = derivative(g_imag, z0.real, n=order-1, dx=1e-5)

        return complex(deriv_real, deriv_imag) / np.math.factorial(order - 1)

    def residue_at_infinity(self, f, R=100):
        """
        Compute residue at infinity via large contour (clockwise).
        """
        theta = np.linspace(0, 2*np.pi, 10000)
        z = R * np.exp(1j * theta)
        dz = 1j * R * np.exp(1j * theta)

        integrand = f(z) * dz
        # Negative because we want clockwise orientation
        integral = -np.trapz(integrand, theta)

        return integral / (2 * np.pi * 1j)


def demonstrate_residue_calculations():
    """
    Demonstrate various residue calculation techniques.
    """
    RC = ResidueCalculator()
    z = sp.Symbol('z')

    print("=" * 60)
    print("RESIDUE COMPUTATION DEMONSTRATIONS")
    print("=" * 60)

    # Example 1: Simple pole - e^z/(z-1)
    print("\n1. SIMPLE POLE: e^z/(z-1) at z=1")

    f1 = lambda z: np.exp(z)/(z - 1)
    expr1 = sp.exp(z)/(z - 1)

    res1_symbolic = RC.residue_symbolic(expr1, 1)
    res1_limit = RC.residue_simple_pole_numeric(f1, 1)
    res1_contour = RC.residue_contour_numeric(f1, 1, radius=0.5)

    print(f"   Symbolic:    {res1_symbolic} = {float(res1_symbolic):.6f}")
    print(f"   Limit:       {res1_limit:.6f}")
    print(f"   Contour:     {res1_contour:.6f}")
    print(f"   Expected:    e = {np.e:.6f}")

    # Example 2: L'Hopital technique - z/(z^2+1) at z=i
    print("\n2. L'HOPITAL: z/(z²+1) at z=i")

    f2 = lambda z: z/(z**2 + 1)
    expr2 = z/(z**2 + 1)

    res2_symbolic = RC.residue_symbolic(expr2, sp.I)
    res2_contour = RC.residue_contour_numeric(f2, 1j, radius=0.5)

    print(f"   Symbolic:    {res2_symbolic}")
    print(f"   Contour:     {res2_contour:.6f}")
    print(f"   P(i)/Q'(i) = i/(2i) = 1/2")

    # Example 3: Higher order pole - e^z/z^3 at z=0
    print("\n3. HIGHER ORDER: e^z/z³ at z=0 (order 3)")

    f3 = lambda z: np.exp(z)/z**3
    expr3 = sp.exp(z)/z**3

    res3_symbolic = RC.residue_symbolic(expr3, 0)
    res3_contour = RC.residue_contour_numeric(f3, 0, radius=0.1)

    print(f"   Symbolic:    {res3_symbolic}")
    print(f"   Contour:     {res3_contour:.6f}")
    print(f"   Expected:    1/2! = 0.5")

    # Example 4: sin(z)/z^4 at z=0
    print("\n4. LAURENT EXPANSION: sin(z)/z⁴ at z=0")

    f4 = lambda z: np.sin(z)/z**4
    expr4 = sp.sin(z)/z**4

    res4_symbolic = RC.residue_symbolic(expr4, 0)
    res4_contour = RC.residue_contour_numeric(f4, 0, radius=0.1)

    print(f"   Symbolic:    {res4_symbolic}")
    print(f"   Contour:     {res4_contour:.6f}")
    print(f"   Expected:    -1/6 = {-1/6:.6f}")

    # Example 5: Essential singularity - e^(1/z) at z=0
    print("\n5. ESSENTIAL: e^(1/z) at z=0")

    f5 = lambda z: np.exp(1/z)

    res5_contour = RC.residue_contour_numeric(f5, 0, radius=0.1)

    print(f"   Contour:     {res5_contour:.6f}")
    print(f"   Expected:    1 (coefficient of 1/z in expansion)")

    # Example 6: Residue at infinity
    print("\n6. RESIDUE AT INFINITY: z³/(z²-1)")

    f6 = lambda z: z**3/(z**2 - 1)

    # Finite poles at z = ±1
    res6_plus1 = RC.residue_contour_numeric(f6, 1, radius=0.5)
    res6_minus1 = RC.residue_contour_numeric(f6, -1, radius=0.5)
    res6_inf = RC.residue_at_infinity(f6, R=100)

    print(f"   Res at z=1:   {res6_plus1:.6f}")
    print(f"   Res at z=-1:  {res6_minus1:.6f}")
    print(f"   Res at z=∞:   {res6_inf:.6f}")
    print(f"   Sum:          {res6_plus1 + res6_minus1 + res6_inf:.6f}")
    print(f"   Expected sum: 0")


def visualize_residue_computation():
    """
    Visualize the contour integration method for residue computation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Function: 1/(z^2 + 1)
    f = lambda z: 1/(z**2 + 1)

    # Create complex plane grid
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Compute |f(z)|
    with np.errstate(divide='ignore', invalid='ignore'):
        F = 1/(Z**2 + 1)
        F_mag = np.abs(F)
        F_mag = np.clip(F_mag, 0, 5)

    # Plot 1: |f(z)| with poles marked
    ax1 = axes[0, 0]
    im1 = ax1.contourf(X, Y, F_mag, levels=50, cmap='viridis')
    ax1.plot([0], [1], 'r*', markersize=15, label='Pole at i')
    ax1.plot([0], [-1], 'r*', markersize=15, label='Pole at -i')
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(0.5*np.cos(theta), 1 + 0.5*np.sin(theta), 'w-', linewidth=2)
    ax1.set_xlabel('Re(z)')
    ax1.set_ylabel('Im(z)')
    ax1.set_title('|f(z)| = |1/(z²+1)| with contour around z=i')
    plt.colorbar(im1, ax=ax1)

    # Plot 2: Integrand along contour
    ax2 = axes[0, 1]
    t = np.linspace(0, 2*np.pi, 1000)
    z_contour = 1j + 0.5 * np.exp(1j * t)  # Circle around i
    dz = 1j * 0.5 * np.exp(1j * t)
    integrand = f(z_contour) * dz

    ax2.plot(t * 180/np.pi, integrand.real, 'b-', linewidth=2, label='Real part')
    ax2.plot(t * 180/np.pi, integrand.imag, 'r-', linewidth=2, label='Imag part')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('f(z) dz')
    ax2.set_title('Integrand f(z)dz along contour')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Convergence with radius
    ax3 = axes[1, 0]
    radii = np.linspace(0.05, 0.9, 50)  # Must not include pole at 0
    residues_real = []
    residues_imag = []

    for r in radii:
        t = np.linspace(0, 2*np.pi, 5000)
        z = 1j + r * np.exp(1j * t)
        dz = 1j * r * np.exp(1j * t)
        integral = np.trapz(f(z) * dz, t) / (2 * np.pi * 1j)
        residues_real.append(integral.real)
        residues_imag.append(integral.imag)

    ax3.plot(radii, residues_real, 'b-', linewidth=2, label='Real part')
    ax3.plot(radii, residues_imag, 'r-', linewidth=2, label='Imag part')
    ax3.axhline(y=-0.5, color='b', linestyle='--', alpha=0.5, label='Expected: -1/(2i) = -0.5i')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Contour radius')
    ax3.set_ylabel('Computed residue')
    ax3.set_title('Residue vs contour radius (should be constant)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Phase portrait
    ax4 = axes[1, 1]
    phase = np.angle(F)
    ax4.contourf(X, Y, phase, levels=50, cmap='hsv')
    ax4.plot([0], [1], 'w*', markersize=15)
    ax4.plot([0], [-1], 'w*', markersize=15)
    ax4.set_xlabel('Re(z)')
    ax4.set_ylabel('Im(z)')
    ax4.set_title('Phase of f(z) = 1/(z²+1)')

    plt.tight_layout()
    plt.savefig('residue_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_methods_accuracy():
    """
    Compare accuracy of different residue computation methods.
    """
    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON OF METHODS")
    print("=" * 60)

    RC = ResidueCalculator()
    z = sp.Symbol('z')

    # Test function: 1/(z^3 - 1) at z = 1
    f = lambda z: 1/(z**3 - 1)
    expr = 1/(z**3 - 1)

    # Exact residue using L'Hopital: 1/(3z^2)|_{z=1} = 1/3
    exact = 1/3

    print("\nFunction: 1/(z³-1) at z=1")
    print(f"Exact residue: 1/3 = {exact:.10f}")
    print("-" * 40)

    # Method 1: Symbolic
    res_symbolic = complex(RC.residue_symbolic(expr, 1))
    print(f"Symbolic:     {res_symbolic.real:.10f}")

    # Method 2: Limit formula with various epsilons
    for eps in [1e-4, 1e-6, 1e-8, 1e-10]:
        res_limit = RC.residue_simple_pole_numeric(f, 1, epsilon=eps)
        error = abs(res_limit - exact)
        print(f"Limit (ε={eps:.0e}): {res_limit.real:.10f}  error: {error:.2e}")

    # Method 3: Contour with various radii
    for r in [0.5, 0.1, 0.01]:
        res_contour = RC.residue_contour_numeric(f, 1, radius=r)
        error = abs(res_contour - exact)
        print(f"Contour (r={r}):  {res_contour.real:.10f}  error: {error:.2e}")


if __name__ == "__main__":
    demonstrate_residue_calculations()
    visualize_residue_computation()
    compare_methods_accuracy()
```

---

## Summary

### Key Formulas

| Situation | Formula |
|-----------|---------|
| Simple pole | $\text{Res} = \lim_{z \to z_0}(z-z_0)f(z)$ |
| Simple pole of $P/Q$ | $\text{Res} = P(z_0)/Q'(z_0)$ |
| Pole of order $m$ | $\text{Res} = \frac{1}{(m-1)!}\lim_{z \to z_0}\frac{d^{m-1}}{dz^{m-1}}[(z-z_0)^m f(z)]$ |
| From Laurent series | $\text{Res} = a_{-1}$ (coefficient of $1/(z-z_0)$) |
| At infinity | $\text{Res}_{z=\infty} = -\text{Res}_{w=0}\frac{1}{w^2}f(1/w)$ |

### Main Takeaways

1. **Simple pole residues** are computed using limit or L'Hopital formulas.

2. **Higher-order poles** require the derivative formula or Laurent expansion.

3. **Laurent series** method works for all cases, including essential singularities.

4. **Residue at infinity** completes the sum rule: total residue = 0.

5. **In QM**, residues give scattering amplitudes, partial widths, and normalization factors.

---

## Daily Checklist

- [ ] I can compute residues at simple poles using limit formula
- [ ] I can apply L'Hopital technique for rational functions
- [ ] I can use the derivative formula for higher-order poles
- [ ] I can find residues from Laurent series
- [ ] I understand residue at infinity
- [ ] I see how residues relate to scattering amplitudes

---

## Preview: Day 186

Tomorrow we prove the **Residue Theorem**:

$$\oint_C f(z)\, dz = 2\pi i \sum_k \text{Res}_{z=z_k} f(z)$$

This powerful result connects all our residue computations to contour integrals!

We'll also cover the **Argument Principle** and **Rouche's Theorem** for counting zeros.

---

*"Mathematics is the art of giving the same name to different things."*
— Henri Poincare
