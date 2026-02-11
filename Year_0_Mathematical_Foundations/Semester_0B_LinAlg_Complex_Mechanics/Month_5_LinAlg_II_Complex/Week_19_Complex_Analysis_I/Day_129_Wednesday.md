# Day 129: Analytic Functions & The Cauchy-Riemann Equations

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Analyticity & Cauchy-Riemann |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Define and compute complex derivatives
2. State and apply the Cauchy-Riemann equations
3. Determine where functions are analytic
4. Understand the profound constraints analyticity imposes
5. Connect analytic functions to harmonic functions
6. Recognize why quantum wavefunctions often involve analytic structures

---

## 📚 Required Reading

### Primary Text: Churchill & Brown
- **Chapter 2, Sections 18-26**: Analytic Functions

### Alternative: Needham, "Visual Complex Analysis"
- **Chapter 4**: Differentiation

### Physics Connection
- **Shankar, Chapter 1**: Mathematical Introduction (analytic continuation)

---

## 🎬 Video Resources

### 3Blue1Brown
- "What makes complex analysis special?"

### MIT OpenCourseWare
- 18.04 Complex Variables: Lecture on Cauchy-Riemann

---

## 📖 Core Content: Theory and Concepts

### 1. Complex Differentiation

**Definition:** A function f(z) is **differentiable** at z₀ if the limit exists:
$$f'(z_0) = \lim_{z \to z_0} \frac{f(z) - f(z_0)}{z - z_0}$$

**Critical Insight:** Unlike real functions, this limit must exist regardless of the direction from which z approaches z₀ in the complex plane!

**Example: f(z) = z²**
$$f'(z_0) = \lim_{z \to z_0} \frac{z^2 - z_0^2}{z - z_0} = \lim_{z \to z_0} (z + z_0) = 2z_0$$
This limit is the same from every direction ✓

**Counterexample: f(z) = z̄ (complex conjugate)**
$$\frac{f(z) - f(z_0)}{z - z_0} = \frac{\bar{z} - \bar{z_0}}{z - z_0}$$

Approaching along real axis (z - z₀ = h ∈ ℝ):
$$\frac{\bar{z_0 + h} - \bar{z_0}}{h} = \frac{h}{h} = 1$$

Approaching along imaginary axis (z - z₀ = ik, k ∈ ℝ):
$$\frac{\bar{z_0 + ik} - \bar{z_0}}{ik} = \frac{-ik}{ik} = -1$$

Different limits! So f(z) = z̄ is **nowhere differentiable**.

---

### 2. The Cauchy-Riemann Equations

Write f(z) = f(x + iy) = u(x,y) + iv(x,y), where u and v are real-valued functions.

**Theorem (Cauchy-Riemann Equations):**
If f(z) = u + iv is differentiable at z₀ = x₀ + iy₀, then:
$$\boxed{\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y} \quad \text{and} \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}}$$

**Proof Sketch:**
The derivative f'(z₀) must be the same approaching horizontally (along x) and vertically (along y):

Horizontal approach (Δy = 0):
$$f'(z_0) = \lim_{\Delta x \to 0} \frac{[u(x_0+\Delta x, y_0) - u(x_0,y_0)] + i[v(x_0+\Delta x, y_0) - v(x_0,y_0)]}{\Delta x}$$
$$= \frac{\partial u}{\partial x} + i\frac{\partial v}{\partial x}$$

Vertical approach (Δx = 0):
$$f'(z_0) = \lim_{\Delta y \to 0} \frac{[u(x_0, y_0+\Delta y) - u(x_0,y_0)] + i[v(x_0, y_0+\Delta y) - v(x_0,y_0)]}{i\Delta y}$$
$$= \frac{1}{i}\frac{\partial u}{\partial y} + \frac{\partial v}{\partial y} = -i\frac{\partial u}{\partial y} + \frac{\partial v}{\partial y}$$

Setting these equal:
$$\frac{\partial u}{\partial x} + i\frac{\partial v}{\partial x} = \frac{\partial v}{\partial y} - i\frac{\partial u}{\partial y}$$

Matching real and imaginary parts gives the Cauchy-Riemann equations. □

**The Converse (Sufficient Conditions):**
If u and v have continuous first partial derivatives in a neighborhood of z₀, and satisfy the Cauchy-Riemann equations at z₀, then f is differentiable at z₀.

---

### 3. Analytic Functions

**Definition:** A function f is **analytic** (or **holomorphic**) at z₀ if it is differentiable in some neighborhood of z₀.

A function is **analytic on a domain D** if it is analytic at every point in D.

**Terminology:**
- **Entire function:** Analytic on all of ℂ (e.g., polynomials, eᶻ, sin z, cos z)
- **Singular point:** A point where f fails to be analytic

**Key Insight:** Analyticity is an incredibly strong condition! It implies:
- Infinitely differentiable (smooth)
- Equals its Taylor series locally
- Uniquely determined by values on any curve
- The real and imaginary parts are harmonic functions

---

### 4. Examples: Testing Analyticity

**Example 1: f(z) = z³**
Write z = x + iy:
$$z^3 = (x+iy)^3 = x^3 + 3x^2(iy) + 3x(iy)^2 + (iy)^3$$
$$= x^3 + 3ix^2y - 3xy^2 - iy^3$$
$$= (x^3 - 3xy^2) + i(3x^2y - y^3)$$

So u = x³ - 3xy² and v = 3x²y - y³.

Check Cauchy-Riemann:
- ∂u/∂x = 3x² - 3y²
- ∂v/∂y = 3x² - 3y² ✓
- ∂u/∂y = -6xy
- -∂v/∂x = -6xy ✓

C-R satisfied everywhere → f(z) = z³ is entire.

**Example 2: f(z) = |z|² = zz̄**
$$f(z) = x^2 + y^2$$
So u = x² + y², v = 0.

Check Cauchy-Riemann:
- ∂u/∂x = 2x, ∂v/∂y = 0 → Equal only when x = 0
- ∂u/∂y = 2y, -∂v/∂x = 0 → Equal only when y = 0

C-R satisfied only at z = 0, but a single point isn't a neighborhood.
→ f(z) = |z|² is **nowhere analytic**.

**Example 3: f(z) = eᶻ**
$$e^z = e^{x+iy} = e^x e^{iy} = e^x(\cos y + i\sin y)$$

So u = eˣ cos y, v = eˣ sin y.

Check Cauchy-Riemann:
- ∂u/∂x = eˣ cos y = ∂v/∂y ✓
- ∂u/∂y = -eˣ sin y = -∂v/∂x ✓

C-R satisfied everywhere → eᶻ is entire.

---

### 5. Harmonic Functions

**Definition:** A real-valued function φ(x,y) is **harmonic** if it satisfies Laplace's equation:
$$\nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} = 0$$

**Theorem:** If f = u + iv is analytic, then both u and v are harmonic.

**Proof:**
From C-R: ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x

Differentiate the first equation with respect to x:
$$\frac{\partial^2 u}{\partial x^2} = \frac{\partial^2 v}{\partial x \partial y}$$

Differentiate the second equation with respect to y:
$$\frac{\partial^2 u}{\partial y^2} = -\frac{\partial^2 v}{\partial y \partial x}$$

Adding (using equality of mixed partials):
$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$

Similarly for v. □

**Harmonic Conjugate:** If u is harmonic, v is called a **harmonic conjugate** of u if u + iv is analytic.

---

### 6. Cauchy-Riemann in Polar Form

For f(z) = f(re^{iθ}) = U(r,θ) + iV(r,θ):

$$\boxed{\frac{\partial U}{\partial r} = \frac{1}{r}\frac{\partial V}{\partial \theta} \quad \text{and} \quad \frac{\partial V}{\partial r} = -\frac{1}{r}\frac{\partial U}{\partial \theta}}$$

**Example: f(z) = log z = ln r + iθ**
Here U = ln r, V = θ.

- ∂U/∂r = 1/r, (1/r)∂V/∂θ = 1/r ✓
- ∂V/∂r = 0, -(1/r)∂U/∂θ = 0 ✓

The Cauchy-Riemann equations are satisfied for r > 0.

---

### 7. 🔬 Quantum Mechanics Connection: Analytic Wavefunctions

**Why Analyticity Matters in QM:**

1. **Analytic Continuation:** Wave functions and Green's functions are often analytic in parts of the complex energy plane. This allows extending solutions beyond their original domain.

2. **Scattering Theory:** The S-matrix has poles in the complex energy plane corresponding to bound states and resonances.

3. **Path Integrals:** Feynman's formulation extends classical action to complex values, requiring analytic continuation.

4. **Wick Rotation:** Transforming t → iτ (imaginary time) relies on analytic continuation to connect quantum mechanics with statistical mechanics.

**Example: Harmonic Oscillator Wave Functions**
The ground state ψ₀(x) = e^{-x²/2} extends to an entire function ψ₀(z) = e^{-z²/2} in the complex plane!

**Dispersion Relations:** In quantum field theory, the analyticity of scattering amplitudes leads to Kramers-Kronig relations connecting real and imaginary parts of response functions.

---

## ✏️ Worked Examples

### Example 1: Find Where f(z) = z̄z is Analytic

**Solution:**
$$f(z) = \bar{z} \cdot z = (x-iy)(x+iy) = x^2 + y^2$$

This is real, so v = 0 and u = x² + y².

Cauchy-Riemann equations:
- ∂u/∂x = 2x should equal ∂v/∂y = 0 → x = 0
- ∂u/∂y = 2y should equal -∂v/∂x = 0 → y = 0

Only satisfied at z = 0, which isn't a neighborhood.

**Answer:** f(z) = |z|² is nowhere analytic.

---

### Example 2: Verify f(z) = sin z is Entire

**Solution:**
Using sin z = (eⁱᶻ - e⁻ⁱᶻ)/(2i):

$$\sin z = \sin(x+iy) = \sin x \cosh y + i \cos x \sinh y$$

So:
- u = sin x cosh y
- v = cos x sinh y

Check Cauchy-Riemann:
- ∂u/∂x = cos x cosh y
- ∂v/∂y = cos x cosh y ✓
- ∂u/∂y = sin x sinh y
- -∂v/∂x = -(-sin x sinh y) = sin x sinh y ✓

C-R satisfied everywhere, partial derivatives are continuous.

**Answer:** sin z is entire.

---

### Example 3: Find the Harmonic Conjugate of u = x² - y²

**Solution:**
First verify u is harmonic:
$$\nabla^2 u = \frac{\partial^2}{\partial x^2}(x^2-y^2) + \frac{\partial^2}{\partial y^2}(x^2-y^2) = 2 + (-2) = 0 \checkmark$$

Use Cauchy-Riemann to find v:
$$\frac{\partial v}{\partial y} = \frac{\partial u}{\partial x} = 2x$$

Integrate with respect to y:
$$v = 2xy + g(x)$$

Use the second C-R equation:
$$\frac{\partial v}{\partial x} = 2y + g'(x) = -\frac{\partial u}{\partial y} = -(-2y) = 2y$$

So g'(x) = 0, meaning g(x) = C (constant).

**Answer:** v = 2xy + C, and f(z) = (x² - y²) + i(2xy) = z²

---

### Example 4: Show that u = eˣ cos y Satisfies Laplace's Equation

**Solution:**
$$\frac{\partial u}{\partial x} = e^x \cos y, \quad \frac{\partial^2 u}{\partial x^2} = e^x \cos y$$

$$\frac{\partial u}{\partial y} = -e^x \sin y, \quad \frac{\partial^2 u}{\partial y^2} = -e^x \cos y$$

$$\nabla^2 u = e^x \cos y + (-e^x \cos y) = 0 \checkmark$$

This is the real part of eᶻ, which is entire, so it must be harmonic.

---

## 🔧 Practice Problems

### Level 1: Basic Cauchy-Riemann
1. Verify that f(z) = z⁴ satisfies the Cauchy-Riemann equations everywhere.
2. Show that f(z) = Re(z) = x is nowhere analytic.
3. Find u and v for f(z) = z² + 2z and verify C-R.

### Level 2: Testing Analyticity
4. Determine all points where f(z) = xy + iy² is differentiable.
5. Show that f(z) = eˣ(cos y - i sin y) is nowhere analytic.
6. Prove that f(z) = z + 1/z is analytic except at z = 0.

### Level 3: Harmonic Conjugates
7. Find the harmonic conjugate of u = x³ - 3xy².
8. Given u = ln(x² + y²), find v such that f = u + iv is analytic.
9. Show that if v is a harmonic conjugate of u, then -u is a harmonic conjugate of v.

### Level 4: Theory and Proofs
10. Prove that if f and f̄ are both analytic on a domain, then f is constant.
11. Show that |f(z)|² cannot be harmonic for a non-constant analytic f.
12. If f = u + iv is analytic and u + v = 0, prove f is constant.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Visualize the Cauchy-Riemann equations
def plot_analytic_function(f, name, x_range=(-2, 2), y_range=(-2, 2), n=100):
    """Visualize an analytic function and verify C-R equations."""
    x = np.linspace(x_range[0], x_range[1], n)
    y = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    W = f(Z)
    U = np.real(W)
    V = np.imag(W)
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot real part (surface)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot_surface(X, Y, U, cmap='coolwarm', alpha=0.8)
    ax1.set_title(f'Re({name})')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    
    # Plot imaginary part (surface)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, V, cmap='viridis', alpha=0.8)
    ax2.set_title(f'Im({name})')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    
    # Level curves (both on same plot)
    ax3 = fig.add_subplot(2, 3, 3)
    levels_u = ax3.contour(X, Y, U, colors='blue', linestyles='solid', levels=15)
    levels_v = ax3.contour(X, Y, V, colors='red', linestyles='dashed', levels=15)
    ax3.clabel(levels_u, inline=True, fontsize=8)
    ax3.set_title(f'Level curves: u (blue), v (red)')
    ax3.set_xlabel('x'); ax3.set_ylabel('y')
    ax3.axis('equal')
    
    # Verify Cauchy-Riemann numerically
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Compute partial derivatives using central differences
    du_dx = (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2*dx)
    du_dy = (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2*dy)
    dv_dx = (np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)) / (2*dx)
    dv_dy = (np.roll(V, -1, axis=0) - np.roll(V, 1, axis=0)) / (2*dy)
    
    # C-R equations: du/dx = dv/dy and du/dy = -dv/dx
    cr1_error = np.abs(du_dx - dv_dy)[10:-10, 10:-10]
    cr2_error = np.abs(du_dy + dv_dx)[10:-10, 10:-10]
    
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(cr1_error, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                     origin='lower', cmap='hot')
    plt.colorbar(im4, ax=ax4)
    ax4.set_title('|∂u/∂x - ∂v/∂y| (should be ~0)')
    
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(cr2_error, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                     origin='lower', cmap='hot')
    plt.colorbar(im5, ax=ax5)
    ax5.set_title('|∂u/∂y + ∂v/∂x| (should be ~0)')
    
    # Magnitude and phase
    ax6 = fig.add_subplot(2, 3, 6)
    magnitude = np.abs(W)
    phase = np.angle(W)
    ax6.contourf(X, Y, magnitude, levels=20, cmap='plasma')
    ax6.set_title(f'|{name}|')
    plt.colorbar(ax6.contourf(X, Y, magnitude, levels=20, cmap='plasma'), ax=ax6)
    
    plt.tight_layout()
    plt.savefig(f'{name.replace(" ", "_")}_analysis.png', dpi=150)
    plt.show()
    
    print(f"\n{name} Cauchy-Riemann verification:")
    print(f"  Max |∂u/∂x - ∂v/∂y|: {np.max(cr1_error):.2e}")
    print(f"  Max |∂u/∂y + ∂v/∂x|: {np.max(cr2_error):.2e}")

# Test various functions
print("=" * 60)
print("ANALYTIC FUNCTION ANALYSIS")
print("=" * 60)

# 1. f(z) = z^2 (analytic everywhere)
plot_analytic_function(lambda z: z**2, "z²")

# 2. f(z) = e^z (analytic everywhere)
plot_analytic_function(lambda z: np.exp(z), "exp(z)")

# 3. f(z) = sin(z) (analytic everywhere)
plot_analytic_function(lambda z: np.sin(z), "sin(z)")

# 4. f(z) = 1/z (analytic except at 0)
def f_inv(z):
    result = np.zeros_like(z, dtype=complex)
    mask = np.abs(z) > 0.1
    result[mask] = 1/z[mask]
    result[~mask] = np.nan
    return result
plot_analytic_function(f_inv, "1/z")

# Compare analytic vs non-analytic functions
print("\n" + "=" * 60)
print("COMPARING ANALYTIC VS NON-ANALYTIC")
print("=" * 60)

def compare_analyticity():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Analytic: z^2
    f_analytic = Z**2
    
    # Non-analytic: |z|^2
    f_non_analytic = np.abs(Z)**2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Real parts
    axes[0, 0].contourf(X, Y, np.real(f_analytic), levels=20, cmap='coolwarm')
    axes[0, 0].set_title('Re(z²) - ANALYTIC')
    axes[0, 0].axis('equal')
    
    axes[0, 1].contourf(X, Y, f_non_analytic, levels=20, cmap='coolwarm')
    axes[0, 1].set_title('|z|² - NOT ANALYTIC')
    axes[0, 1].axis('equal')
    
    # Level curves comparison
    axes[1, 0].contour(X, Y, np.real(f_analytic), colors='blue', levels=15)
    axes[1, 0].contour(X, Y, np.imag(f_analytic), colors='red', levels=15)
    axes[1, 0].set_title('z²: u (blue), v (red) level curves\nNote: Orthogonal!')
    axes[1, 0].axis('equal')
    
    # For |z|^2, imaginary part is 0
    axes[1, 1].contour(X, Y, f_non_analytic, colors='blue', levels=15)
    axes[1, 1].set_title('|z|²: Only real part (v=0)\nNo orthogonal structure')
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('analytic_vs_nonanalytic.png', dpi=150)
    plt.show()

compare_analyticity()

# Harmonic conjugate finder
def find_harmonic_conjugate():
    """
    Given u = x^2 - y^2, find harmonic conjugate v
    using Cauchy-Riemann equations
    """
    print("\n" + "=" * 60)
    print("FINDING HARMONIC CONJUGATE")
    print("=" * 60)
    
    print("\nGiven: u(x,y) = x² - y²")
    print("\nStep 1: Verify u is harmonic")
    print("  ∂²u/∂x² = 2")
    print("  ∂²u/∂y² = -2")
    print("  ∇²u = 2 + (-2) = 0 ✓")
    
    print("\nStep 2: Use C-R to find v")
    print("  ∂v/∂y = ∂u/∂x = 2x")
    print("  Integrating: v = 2xy + g(x)")
    
    print("\nStep 3: Use second C-R equation")
    print("  ∂v/∂x = 2y + g'(x) = -∂u/∂y = 2y")
    print("  So g'(x) = 0, meaning g(x) = C")
    
    print("\nResult: v = 2xy + C")
    print("The analytic function is f(z) = (x² - y²) + i(2xy) = z²")
    
    # Visualize
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    u = X**2 - Y**2
    v = 2*X*Y
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Level curves of u and v
    axes[0].contour(X, Y, u, colors='blue', levels=15, linewidths=1)
    axes[0].contour(X, Y, v, colors='red', levels=15, linewidths=1)
    axes[0].set_title('u = x² - y² (blue) and v = 2xy (red)\nLevel curves are orthogonal')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Show they are orthogonal by plotting gradient vectors
    skip = 10
    Xs, Ys = X[::skip, ::skip], Y[::skip, ::skip]
    
    # Gradients of u
    du_dx = 2*Xs
    du_dy = -2*Ys
    
    # Gradients of v
    dv_dx = 2*Ys
    dv_dy = 2*Xs
    
    axes[1].quiver(Xs, Ys, du_dx, du_dy, color='blue', alpha=0.7, label='∇u')
    axes[1].quiver(Xs, Ys, dv_dx, dv_dy, color='red', alpha=0.7, label='∇v')
    axes[1].set_title('Gradient vectors: ∇u ⊥ ∇v')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    axes[1].axis('equal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('harmonic_conjugate.png', dpi=150)
    plt.show()

find_harmonic_conjugate()
```

---

## 📝 Summary

### Key Concepts

1. **Complex Differentiability**: f'(z₀) exists only if the limit is the same from ALL directions in ℂ

2. **Cauchy-Riemann Equations**: For f = u + iv:
   - ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x (necessary)
   - Plus continuity of partials (sufficient)

3. **Analytic (Holomorphic)**: Differentiable in a neighborhood

4. **Harmonic Functions**: u, v from an analytic function satisfy ∇²φ = 0

5. **Level Curve Property**: For analytic f, level curves of u and v are orthogonal

### The Profound Implications
- Only a tiny fraction of smooth functions ℝ² → ℝ² can be real and imaginary parts of analytic functions
- Knowing an analytic function on any small region determines it everywhere (unique continuation)
- Analyticity provides the mathematical structure underlying quantum amplitudes

---

## ✅ Daily Checklist

- [ ] Master the derivation of Cauchy-Riemann equations
- [ ] Verify C-R for z², eᶻ, sin z
- [ ] Show z̄ and |z|² are nowhere analytic
- [ ] Find harmonic conjugate for u = x² - y²
- [ ] Understand orthogonality of level curves
- [ ] Complete computational visualizations
- [ ] Connect to quantum mechanics applications

---

## 🔮 Preview: Day 130

Tomorrow we explore the **elementary complex functions** — exponential, logarithm, trigonometric, and hyperbolic functions extended to the complex plane. We'll discover remarkable connections like eⁱᶿ = cos θ + i sin θ and why the complex logarithm is multi-valued!
