# Day 131: Complex Integration & Cauchy's Theorem

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Contour Integrals & Cauchy |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Compute contour integrals of complex functions
2. State and understand Cauchy's Theorem
3. Apply Cauchy's Integral Formula
4. Evaluate integrals using deformation of contours
5. Understand why analyticity makes integration powerful
6. Connect contour integration to quantum mechanics

---

## 📚 Required Reading

### Primary Text: Churchill & Brown
- **Chapter 4**: Integrals (pp. 103-142)

### Alternative: Needham, "Visual Complex Analysis"
- **Chapter 8**: Winding Numbers and Cauchy's Theorem

### Physics Connection
- **Arfken & Weber, Chapter 11**: Complex Variable Theory

---

## 🎬 Video Resources

### 3Blue1Brown
- "But what is a contour integral?"

### MIT OpenCourseWare
- 18.04: Lectures on Cauchy's Theorem

---

## 📖 Core Content: Theory and Concepts

### 1. Contour Integrals

**Definition:** Let γ be a smooth curve parametrized by z(t) for a ≤ t ≤ b. The **contour integral** of f along γ is:

$$\boxed{\int_\gamma f(z)\,dz = \int_a^b f(z(t)) z'(t)\,dt}$$

**Intuition:** We're summing up f(z) · dz along the path, where both f and dz are complex numbers.

**Key Properties:**
- Linearity: ∫_γ (αf + βg) dz = α∫_γ f dz + β∫_γ g dz
- Reversal: ∫_{-γ} f dz = -∫_γ f dz
- Additivity: ∫_{γ₁+γ₂} f dz = ∫_{γ₁} f dz + ∫_{γ₂} f dz

**ML Inequality (Estimation):**
$$\left|\int_\gamma f(z)\,dz\right| \leq M \cdot L$$
where M = max|f(z)| on γ and L = length of γ.

---

### 2. Computing Contour Integrals

**Example 1: ∫_γ z dz where γ is the line from 0 to 1+i**

Parametrize: z(t) = t(1+i) = t + it for 0 ≤ t ≤ 1
Then z'(t) = 1 + i

$$\int_\gamma z\,dz = \int_0^1 (t+it)(1+i)\,dt = (1+i)^2 \int_0^1 t\,dt = 2i \cdot \frac{1}{2} = i$$

**Example 2: ∫_γ z̄ dz around the unit circle**

Parametrize: z(t) = e^{it} for 0 ≤ t ≤ 2π
Then z̄ = e^{-it} and z'(t) = ie^{it}

$$\int_\gamma \bar{z}\,dz = \int_0^{2\pi} e^{-it} \cdot ie^{it}\,dt = i\int_0^{2\pi} dt = 2\pi i$$

---

### 3. The Fundamental Example: ∫ z^n dz

**Theorem:** For any closed curve γ encircling the origin once (counterclockwise):

$$\oint_\gamma z^n\,dz = \begin{cases} 2\pi i & \text{if } n = -1 \\ 0 & \text{if } n \neq -1 \end{cases}$$

**Proof for unit circle γ:** z(t) = e^{it}, 0 ≤ t ≤ 2π

$$\oint_\gamma z^n\,dz = \int_0^{2\pi} e^{int} \cdot ie^{it}\,dt = i\int_0^{2\pi} e^{i(n+1)t}\,dt$$

If n ≠ -1:
$$= i \left[\frac{e^{i(n+1)t}}{i(n+1)}\right]_0^{2\pi} = \frac{e^{2\pi i(n+1)} - 1}{n+1} = 0$$

If n = -1:
$$= i\int_0^{2\pi} 1\,dt = 2\pi i$$

**This is the heart of residue theory!**

---

### 4. Cauchy's Theorem

**Theorem (Cauchy's Theorem / Cauchy-Goursat):**
If f is analytic on and inside a simple closed contour γ, then:
$$\boxed{\oint_\gamma f(z)\,dz = 0}$$

**Why is this remarkable?** The integral depends only on the endpoints (for simply connected domains), not the path! This is like the fundamental theorem of calculus, but for complex functions.

**Proof Sketch (using Green's Theorem):**
Write f = u + iv and dz = dx + idy:
$$\int_\gamma f\,dz = \int_\gamma (u+iv)(dx+idy)$$
$$= \int_\gamma (u\,dx - v\,dy) + i\int_\gamma (v\,dx + u\,dy)$$

By Green's theorem:
$$= \iint_D \left(-\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}\right)dA + i\iint_D \left(\frac{\partial u}{\partial x} - \frac{\partial v}{\partial y}\right)dA$$

By Cauchy-Riemann equations, both integrands are zero! □

---

### 5. Consequences of Cauchy's Theorem

**Path Independence:**
If f is analytic in a simply connected domain D, and γ₁, γ₂ are any two paths from z₁ to z₂ in D:
$$\int_{\gamma_1} f(z)\,dz = \int_{\gamma_2} f(z)\,dz$$

**Antiderivatives Exist:**
If f is analytic in simply connected D, then F(z) = ∫_{z_0}^z f(w)dw is well-defined and F'(z) = f(z).

**Deformation of Contours:**
If f is analytic between two closed contours γ₁ and γ₂:
$$\oint_{\gamma_1} f(z)\,dz = \oint_{\gamma_2} f(z)\,dz$$

---

### 6. Cauchy's Integral Formula

**Theorem:** If f is analytic on and inside a simple closed contour γ, and z₀ is inside γ:
$$\boxed{f(z_0) = \frac{1}{2\pi i}\oint_\gamma \frac{f(z)}{z-z_0}\,dz}$$

**Derivative Formula:**
$$f^{(n)}(z_0) = \frac{n!}{2\pi i}\oint_\gamma \frac{f(z)}{(z-z_0)^{n+1}}\,dz$$

**Stunning Implication:** The values of an analytic function on a curve completely determine its values (and all derivatives!) inside!

**Proof of Main Formula:**
By deformation, we can shrink γ to a small circle Cᵣ around z₀:
$$\oint_\gamma \frac{f(z)}{z-z_0}\,dz = \oint_{C_r} \frac{f(z)}{z-z_0}\,dz$$

Write f(z) = f(z₀) + [f(z) - f(z₀)]:
$$= f(z_0)\oint_{C_r} \frac{dz}{z-z_0} + \oint_{C_r} \frac{f(z)-f(z_0)}{z-z_0}\,dz$$

The first integral equals 2πi (our fundamental example with n = -1).

The second integral → 0 as r → 0 because f is continuous:
$$\left|\oint_{C_r} \frac{f(z)-f(z_0)}{z-z_0}\,dz\right| \leq \frac{\max|f(z)-f(z_0)|}{r} \cdot 2\pi r \to 0$$

Therefore: ∮_γ f(z)/(z-z₀) dz = 2πi f(z₀) □

---

### 7. Applications

**Evaluating Real Integrals:**
Many real integrals become trivial with contour methods!

**Example:** ∫_{-∞}^{∞} 1/(1+x²) dx

Consider f(z) = 1/(1+z²) = 1/((z+i)(z-i))

Use semicircular contour in upper half-plane. The only pole inside is z = i.

By Cauchy's formula:
$$\oint = 2\pi i \cdot \text{Res}_{z=i} = 2\pi i \cdot \frac{1}{2i} = \pi$$

The semicircle contribution → 0, leaving ∫_{-∞}^{∞} = π.

---

### 8. 🔬 Quantum Mechanics Connection

**Path Integrals:**
Feynman's formulation uses complex integrals over all paths:
$$\langle x_f|e^{-iHt/\hbar}|x_i\rangle = \int \mathcal{D}[x(t)] e^{iS[x]/\hbar}$$

**Green's Functions:**
The propagator G(E) = (E - H)⁻¹ has poles at energy eigenvalues. Contour integration around these poles extracts spectral information.

**Kramers-Kronig Relations:**
The analyticity of response functions leads to:
$$\text{Re}[\chi(\omega)] = \frac{1}{\pi}\mathcal{P}\int_{-\infty}^{\infty} \frac{\text{Im}[\chi(\omega')]}{\omega' - \omega}\,d\omega'$$

**Residue Theorem in Scattering:**
S-matrix poles in the complex energy plane:
- Real axis: bound states
- Lower half-plane: resonances (with decay width)

**Example: Hydrogen Bound States**
The resolvent G(E) = (E - H)⁻¹ has poles at Eₙ = -13.6/n² eV. Contour integration extracts:
$$\text{Tr}[e^{-\beta H}] = \sum_n e^{-\beta E_n}$$

---

## ✏️ Worked Examples

### Example 1: Evaluate ∮ z² dz around |z| = 2

**Solution:**
f(z) = z² is entire (analytic everywhere), so by Cauchy's theorem:
$$\oint_{|z|=2} z^2\,dz = 0$$

---

### Example 2: Evaluate ∮ 1/(z-1) dz around |z| = 2

**Solution:**
The function 1/(z-1) has a pole at z = 1, which is inside |z| = 2.

By our fundamental result (or Cauchy's integral formula with f(z) = 1):
$$\oint_{|z|=2} \frac{dz}{z-1} = 2\pi i$$

---

### Example 3: Evaluate ∮ e^z/(z-1) dz around |z| = 2

**Solution:**
f(z) = eᶻ is entire. By Cauchy's integral formula:
$$\oint_{|z|=2} \frac{e^z}{z-1}\,dz = 2\pi i \cdot e^1 = 2\pi i e$$

---

### Example 4: Evaluate ∮ z/(z²+1) dz around |z| = 2

**Solution:**
Factor: z/(z²+1) = z/((z+i)(z-i))

Both poles z = ±i are inside |z| = 2.

Using partial fractions: z/(z²+1) = ½·1/(z-i) + ½·1/(z+i)

$$\oint = \frac{1}{2}\oint \frac{dz}{z-i} + \frac{1}{2}\oint \frac{dz}{z+i} = \frac{1}{2}(2\pi i) + \frac{1}{2}(2\pi i) = 2\pi i$$

---

### Example 5: Use Cauchy's Formula to Find f''(0) for f(z) = eᶻ

**Solution:**
By the derivative formula:
$$f''(0) = \frac{2!}{2\pi i}\oint_{|z|=1} \frac{e^z}{z^3}\,dz$$

We know f''(z) = eᶻ, so f''(0) = 1.

Therefore:
$$\oint_{|z|=1} \frac{e^z}{z^3}\,dz = \frac{2\pi i \cdot f''(0)}{2!} = \pi i$$

---

## 🔧 Practice Problems

### Level 1: Direct Computation
1. Evaluate ∫_γ z dz where γ is the circle |z| = 1.
2. Compute ∫_γ z̄ dz where γ goes from 0 to 1+i along the real axis then vertically to 1+i.
3. Evaluate ∮_{|z|=1} 1/z² dz.

### Level 2: Cauchy's Theorem
4. Use Cauchy's theorem to show ∮_{|z|=1} sin z dz = 0.
5. Evaluate ∮_{|z|=3} 1/(z-2) dz.
6. Compute ∮_{|z|=1} 1/(z-2) dz.

### Level 3: Cauchy's Integral Formula
7. Evaluate ∮_{|z|=2} eᶻ/(z+1) dz.
8. Find ∮_{|z|=1} cos z/z³ dz using the derivative formula.
9. Compute ∮_{|z|=4} z²/(z-1)(z-2) dz.

### Level 4: Applications
10. Evaluate ∫₀^{2π} 1/(2+cos θ) dθ using z = e^{iθ}.
11. Show that ∫_{-∞}^{∞} cos x/(1+x²) dx = π/e.
12. Prove: If f is entire and bounded, then f is constant (Liouville's theorem).

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Contour integral visualization and computation
def contour_integral_demo():
    """Visualize and compute contour integrals."""
    print("=" * 60)
    print("CONTOUR INTEGRAL DEMONSTRATIONS")
    print("=" * 60)
    
    # Parametric integration along a contour
    def integrate_along_contour(f, z_of_t, dz_dt, t_range, n_points=1000):
        """Numerically integrate f(z) dz along parametrized contour."""
        t = np.linspace(t_range[0], t_range[1], n_points)
        z = z_of_t(t)
        integrand = f(z) * dz_dt(t)
        
        # Trapezoidal integration
        dt = t[1] - t[0]
        integral = np.sum(integrand[:-1] + integrand[1:]) * dt / 2
        return integral
    
    # Example 1: ∮ z dz around unit circle (should be 0)
    print("\n1. ∮ z dz around |z|=1")
    f = lambda z: z
    z_of_t = lambda t: np.exp(1j * t)
    dz_dt = lambda t: 1j * np.exp(1j * t)
    result = integrate_along_contour(f, z_of_t, dz_dt, (0, 2*np.pi))
    print(f"   Numerical result: {result:.6f}")
    print(f"   Exact (Cauchy): 0")
    
    # Example 2: ∮ 1/z dz around unit circle (should be 2πi)
    print("\n2. ∮ 1/z dz around |z|=1")
    f = lambda z: 1/z
    result = integrate_along_contour(f, z_of_t, dz_dt, (0, 2*np.pi))
    print(f"   Numerical result: {result:.6f}")
    print(f"   Exact: 2πi = {2*np.pi*1j:.6f}")
    
    # Example 3: ∮ e^z/(z-0.5) dz around |z|=1 (pole at 0.5)
    print("\n3. ∮ e^z/(z-0.5) dz around |z|=1")
    f = lambda z: np.exp(z)/(z - 0.5)
    result = integrate_along_contour(f, z_of_t, dz_dt, (0, 2*np.pi))
    exact = 2*np.pi*1j * np.exp(0.5)  # Cauchy's formula
    print(f"   Numerical result: {result:.6f}")
    print(f"   Exact (Cauchy): 2πi·e^0.5 = {exact:.6f}")
    
    # Example 4: Path independence
    print("\n4. Path independence: ∫ z² dz from 0 to 1+i")
    
    # Path 1: straight line
    z1_of_t = lambda t: t * (1+1j)
    dz1_dt = lambda t: (1+1j) * np.ones_like(t)
    result1 = integrate_along_contour(lambda z: z**2, z1_of_t, dz1_dt, (0, 1))
    
    # Path 2: along real axis then imaginary
    # First segment: 0 to 1
    z2a_of_t = lambda t: t
    dz2a_dt = lambda t: np.ones_like(t)
    result2a = integrate_along_contour(lambda z: z**2, z2a_of_t, dz2a_dt, (0, 1))
    
    # Second segment: 1 to 1+i
    z2b_of_t = lambda t: 1 + 1j*t
    dz2b_dt = lambda t: 1j * np.ones_like(t)
    result2b = integrate_along_contour(lambda z: z**2, z2b_of_t, dz2b_dt, (0, 1))
    result2 = result2a + result2b
    
    # Exact: antiderivative F(z) = z³/3, so F(1+i) - F(0)
    exact = (1+1j)**3 / 3
    
    print(f"   Path 1 (straight): {result1:.6f}")
    print(f"   Path 2 (L-shaped): {result2:.6f}")
    print(f"   Exact: (1+i)³/3 = {exact:.6f}")

contour_integral_demo()

# Visualize Cauchy's theorem
def visualize_cauchy_theorem():
    """Visualize why ∮ f dz = 0 for analytic functions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Create a grid
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # f(z) = z² (analytic)
    W1 = Z**2
    U1, V1 = np.real(W1), np.imag(W1)
    
    # Plot vector field (u, v) showing how f maps infinitesimal elements
    skip = 5
    axes[0, 0].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                      U1[::skip, ::skip], V1[::skip, ::skip], color='blue', alpha=0.7)
    theta = np.linspace(0, 2*np.pi, 100)
    axes[0, 0].plot(np.cos(theta), np.sin(theta), 'r-', lw=2, label='Contour')
    axes[0, 0].set_title('f(z) = z² (analytic)\n∮ z² dz = 0 by Cauchy')
    axes[0, 0].axis('equal')
    axes[0, 0].set_xlim(-2, 2); axes[0, 0].set_ylim(-2, 2)
    axes[0, 0].legend()
    
    # f(z) = 1/z (not analytic at 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        W2 = 1/Z
        W2[np.abs(Z) < 0.2] = np.nan
    U2, V2 = np.real(W2), np.imag(W2)
    
    axes[0, 1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                      U2[::skip, ::skip], V2[::skip, ::skip], color='blue', alpha=0.7)
    axes[0, 1].plot(np.cos(theta), np.sin(theta), 'r-', lw=2, label='Contour')
    axes[0, 1].scatter([0], [0], c='red', s=100, marker='x', label='Singularity')
    axes[0, 1].set_title('f(z) = 1/z (singular at 0)\n∮ dz/z = 2πi ≠ 0')
    axes[0, 1].axis('equal')
    axes[0, 1].set_xlim(-2, 2); axes[0, 1].set_ylim(-2, 2)
    axes[0, 1].legend()
    
    # Show deformation of contours
    axes[1, 0].set_title('Contour Deformation\n(for analytic functions)')
    for r in [0.5, 1.0, 1.5]:
        axes[1, 0].plot(r*np.cos(theta), r*np.sin(theta), '-', lw=1.5, 
                        label=f'r = {r}')
    axes[1, 0].annotate('', xy=(1.4, 0.4), xytext=(0.4, 0.1),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))
    axes[1, 0].set_xlim(-2, 2); axes[1, 0].set_ylim(-2, 2)
    axes[1, 0].axis('equal')
    axes[1, 0].legend()
    axes[1, 0].text(0.5, -1.8, 'All integrals are equal!', fontsize=12, ha='center')
    
    # Numerical verification of Cauchy
    print("\nNumerical verification of Cauchy's theorem:")
    
    def integrate_circle(f, radius, n=1000):
        t = np.linspace(0, 2*np.pi, n)
        z = radius * np.exp(1j * t)
        dz = 1j * radius * np.exp(1j * t)
        dt = t[1] - t[0]
        return np.sum((f(z[:-1]) + f(z[1:])) * (dz[:-1] + dz[1:])/2) * dt / 2
    
    # Test with z² (analytic)
    for r in [0.5, 1.0, 2.0]:
        result = integrate_circle(lambda z: z**2, r)
        print(f"  ∮ z² dz (r={r}): {result:.6f}")
    
    # Test with 1/z (pole at 0)
    print("\n  ∮ dz/z:")
    for r in [0.5, 1.0, 2.0]:
        result = integrate_circle(lambda z: 1/z, r)
        print(f"    r={r}: {result:.6f} (expect 2πi ≈ {2*np.pi*1j:.4f})")
    
    # Animate integrand along contour
    axes[1, 1].set_title('Integrand f(z)·dz along |z|=1')
    t = np.linspace(0, 2*np.pi, 500)
    z = np.exp(1j * t)
    dz = 1j * np.exp(1j * t)
    
    # For z²
    integrand1 = z**2 * dz
    axes[1, 1].plot(t, np.real(integrand1), 'b-', label='Re(z²·dz)')
    axes[1, 1].plot(t, np.imag(integrand1), 'b--', label='Im(z²·dz)')
    
    # For 1/z
    integrand2 = (1/z) * dz
    axes[1, 1].plot(t, np.real(integrand2), 'r-', label='Re(dz/z)')
    axes[1, 1].plot(t, np.imag(integrand2), 'r--', label='Im(dz/z)')
    
    axes[1, 1].axhline(y=0, color='k', lw=0.5)
    axes[1, 1].set_xlabel('t (parameter)')
    axes[1, 1].set_ylabel('Integrand value')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 2*np.pi)
    
    plt.tight_layout()
    plt.savefig('cauchy_theorem.png', dpi=150)
    plt.show()

visualize_cauchy_theorem()

# Cauchy's integral formula demonstration
def cauchy_integral_formula_demo():
    """Demonstrate Cauchy's integral formula."""
    print("\n" + "=" * 60)
    print("CAUCHY'S INTEGRAL FORMULA")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    def evaluate_at_point_via_contour(f, z0, contour_radius=2, n=1000):
        """
        Compute f(z₀) using Cauchy's formula:
        f(z₀) = (1/2πi) ∮ f(z)/(z-z₀) dz
        """
        t = np.linspace(0, 2*np.pi, n)
        z = contour_radius * np.exp(1j * t)
        dz = 1j * contour_radius * np.exp(1j * t)
        integrand = f(z) / (z - z0)
        dt = t[1] - t[0]
        integral = np.sum((integrand[:-1] + integrand[1:]) * (dz[:-1] + dz[1:])/2) * dt / 2
        return integral / (2 * np.pi * 1j)
    
    # Test function: f(z) = e^z
    f = lambda z: np.exp(z)
    
    print("\nf(z) = e^z")
    print("-" * 40)
    
    test_points = [0, 0.5, 1j, 1+1j, -0.5+0.5j]
    for z0 in test_points:
        computed = evaluate_at_point_via_contour(f, z0)
        exact = np.exp(z0)
        error = np.abs(computed - exact)
        print(f"  z₀ = {z0:>8}: Computed = {computed:.6f}, Exact = {exact:.6f}, Error = {error:.2e}")
    
    # Visualize: show function values are determined by boundary values
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Mask points outside contour
    W = f(Z)
    mask = np.abs(Z) <= 1.5
    
    axes[0].contourf(X, Y, np.real(W), levels=30, cmap='coolwarm')
    theta = np.linspace(0, 2*np.pi, 100)
    axes[0].plot(1.5*np.cos(theta), 1.5*np.sin(theta), 'k-', lw=3, label='Contour')
    axes[0].scatter([0, 0.5, 1], [0, 0.5, 0], c='white', s=100, 
                    edgecolors='black', zorder=5)
    axes[0].set_title('f(z) = e^z\nBoundary values determine interior!')
    axes[0].axis('equal')
    axes[0].set_xlim(-2, 2); axes[0].set_ylim(-2, 2)
    
    # Derivative formula demonstration
    print("\nDerivative formula: f^(n)(z₀) = n!/(2πi) ∮ f(z)/(z-z₀)^(n+1) dz")
    print("-" * 60)
    
    def evaluate_derivative_via_contour(f, z0, n, contour_radius=2, n_points=2000):
        """Compute f^(n)(z₀) using Cauchy's derivative formula."""
        t = np.linspace(0, 2*np.pi, n_points)
        z = z0 + contour_radius * np.exp(1j * t)
        dz = 1j * contour_radius * np.exp(1j * t)
        integrand = f(z) / (z - z0)**(n+1)
        dt = t[1] - t[0]
        integral = np.sum((integrand[:-1] + integrand[1:]) * (dz[:-1] + dz[1:])/2) * dt / 2
        return np.math.factorial(n) * integral / (2 * np.pi * 1j)
    
    z0 = 0.5
    print(f"\nAt z₀ = {z0} for f(z) = e^z (all derivatives = e^{z0}):")
    exact_deriv = np.exp(z0)
    for n in range(5):
        computed = evaluate_derivative_via_contour(f, z0, n)
        print(f"  f^({n})({z0}) = {computed:.6f} (exact: {exact_deriv:.6f})")
    
    # Show convergence with contour size
    radii = np.linspace(0.1, 3, 50)
    errors = []
    for r in radii:
        try:
            computed = evaluate_at_point_via_contour(f, 0.5, contour_radius=r)
            errors.append(np.abs(computed - np.exp(0.5)))
        except:
            errors.append(np.nan)
    
    axes[1].semilogy(radii, errors, 'b-', lw=2)
    axes[1].axhline(y=1e-10, color='g', linestyle='--', label='Machine precision')
    axes[1].set_xlabel('Contour radius')
    axes[1].set_ylabel('Error in f(z₀) computation')
    axes[1].set_title('Cauchy formula accuracy vs contour size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cauchy_formula.png', dpi=150)
    plt.show()

cauchy_integral_formula_demo()

# Real integral evaluation via contour methods
def evaluate_real_integral():
    """Use contour integration for real integrals."""
    print("\n" + "=" * 60)
    print("EVALUATING REAL INTEGRALS VIA CONTOURS")
    print("=" * 60)
    
    # Example: ∫₀^∞ 1/(1+x²) dx = π/2
    print("\n1. ∫₀^∞ 1/(1+x²) dx")
    
    # Numerical verification
    result, _ = integrate.quad(lambda x: 1/(1+x**2), 0, np.inf)
    print(f"   Numerical: {result:.6f}")
    print(f"   Exact (π/2): {np.pi/2:.6f}")
    
    # Example: ∫₀^{2π} 1/(2+cos θ) dθ
    print("\n2. ∫₀^{2π} 1/(2+cos θ) dθ")
    
    result2, _ = integrate.quad(lambda t: 1/(2+np.cos(t)), 0, 2*np.pi)
    exact2 = 2*np.pi/np.sqrt(3)  # Computed via residues
    print(f"   Numerical: {result2:.6f}")
    print(f"   Exact (2π/√3): {exact2:.6f}")
    
    # Visualize the contour method
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show semicircular contour for 1/(1+z²)
    t_real = np.linspace(-5, 5, 100)
    t_semi = np.linspace(0, np.pi, 100)
    R = 5
    
    axes[0].plot(t_real, np.zeros_like(t_real), 'b-', lw=2, label='Real axis')
    axes[0].plot(R*np.cos(t_semi), R*np.sin(t_semi), 'r-', lw=2, label='Semicircle')
    axes[0].scatter([0, 0], [1, -1], c='green', s=100, marker='x', 
                    label='Poles at ±i', zorder=5)
    axes[0].arrow(-5, 0, 9.8, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    axes[0].set_title('Contour for ∫ 1/(1+x²) dx\nClose in upper half-plane')
    axes[0].set_xlim(-6, 6); axes[0].set_ylim(-2, 6)
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Show unit circle contour for trig integral
    theta = np.linspace(0, 2*np.pi, 100)
    axes[1].plot(np.cos(theta), np.sin(theta), 'b-', lw=2)
    axes[1].scatter([2-np.sqrt(3)], [0], c='red', s=100, marker='x', 
                    label='Pole inside')
    axes[1].scatter([2+np.sqrt(3)], [0], c='green', s=100, marker='o', 
                    label='Pole outside')
    axes[1].set_title('Contour for ∫ 1/(2+cos θ) dθ\nz = e^{iθ}, cos θ = (z+1/z)/2')
    axes[1].axis('equal')
    axes[1].set_xlim(-2, 4); axes[1].set_ylim(-2, 2)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_integrals.png', dpi=150)
    plt.show()

evaluate_real_integral()
```

---

## 📝 Summary

### Key Results

| Theorem | Statement |
|---------|-----------|
| Cauchy's Theorem | ∮_γ f dz = 0 if f analytic inside γ |
| Integral Formula | f(z₀) = (1/2πi) ∮ f(z)/(z-z₀) dz |
| Derivative Formula | f^{(n)}(z₀) = (n!/2πi) ∮ f(z)/(z-z₀)^{n+1} dz |
| Fundamental Integral | ∮ z^n dz = 2πi if n=-1, else 0 |

### Deep Insights
- Analyticity makes integration path-independent
- Boundary values completely determine interior values
- All derivatives exist and can be computed from a single contour integral
- Complex integration simplifies many real integrals

---

## ✅ Daily Checklist

- [ ] Compute basic contour integrals by parametrization
- [ ] Apply Cauchy's theorem to show integrals vanish
- [ ] Use Cauchy's integral formula to evaluate f(z₀)
- [ ] Understand why ∮ dz/z = 2πi
- [ ] Compute derivatives using contour integrals
- [ ] Connect to quantum mechanics applications
- [ ] Complete computational demonstrations

---

## 🔮 Preview: Day 132

Tomorrow is our **comprehensive computational lab** where we'll build a complex analysis toolkit: domain coloring, contour integration, residue computation, and applications to physics problems!
