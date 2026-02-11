# Day 130: Elementary Complex Functions

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: exp, log, trig in ℂ |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Extend exponential, logarithmic, and trigonometric functions to ℂ
2. Understand and work with Euler's formula
3. Handle the multi-valued nature of complex logarithm
4. Compute complex powers using principal values
5. Relate hyperbolic and trigonometric functions in ℂ
6. Apply these functions in quantum mechanics contexts

---

## 📚 Required Reading

### Primary Text: Churchill & Brown
- **Chapter 3**: Elementary Functions (pp. 69-102)

### Alternative: Needham, "Visual Complex Analysis"
- **Chapter 5**: Möbius Transformations and Exponentials

### Physics Connection
- **Sakurai, Chapter 2**: Dynamics — time evolution operator e^{-iHt/ℏ}

---

## 🎬 Video Resources

### 3Blue1Brown
- "e^(iπ) in 3.14 minutes"
- "Euler's formula with introductory group theory"

### Mathologer
- "e to the pi i for dummies"

---

## 📖 Core Content: Theory and Concepts

### 1. The Complex Exponential

**Definition:**
$$\boxed{e^z = e^{x+iy} = e^x(\cos y + i\sin y)}$$

This is derived by requiring:
1. eᶻ is entire (analytic everywhere)
2. d/dz(eᶻ) = eᶻ
3. eᶻ agrees with real exponential when z is real

**Euler's Formula:**
$$\boxed{e^{i\theta} = \cos\theta + i\sin\theta}$$

**Key Properties:**
| Property | Formula |
|----------|---------|
| Addition | e^{z₁+z₂} = e^{z₁}e^{z₂} |
| Never zero | e^z ≠ 0 for all z |
| Periodicity | e^{z+2πi} = e^z (period 2πi) |
| Modulus | \|e^z\| = e^x = e^{Re(z)} |
| Argument | arg(e^z) = y = Im(z) (mod 2π) |
| Derivative | d/dz(e^z) = e^z |

**Special Values:**
- e^{iπ} = -1 (Euler's identity)
- e^{iπ/2} = i
- e^{2πi} = 1

**Mapping Properties:**
The function w = eᶻ maps:
- Horizontal lines y = c → rays from origin at angle c
- Vertical lines x = c → circles |w| = eᶜ
- Horizontal strip 0 ≤ Im(z) < 2π → entire w-plane minus origin (one-to-one)

---

### 2. The Complex Logarithm

**The Inverse Problem:** Solve eʷ = z for w.

Let w = u + iv and z = re^{iθ}:
$$e^{u+iv} = e^u e^{iv} = re^{i\theta}$$

Matching modulus and argument:
- e^u = r → u = ln r
- v = θ + 2πn for any integer n

**Definition (Multi-valued):**
$$\boxed{\log z = \ln|z| + i\arg(z) = \ln|z| + i(\theta + 2\pi n), \quad n \in \mathbb{Z}}$$

**Principal Value:**
$$\boxed{\text{Log } z = \ln|z| + i\text{Arg}(z), \quad -\pi < \text{Arg}(z) \leq \pi}$$

**Examples:**
- log(1) = 0 + 2πin = 2πin
- log(-1) = ln(1) + i(π + 2πn) = i(2n+1)π
- log(i) = ln(1) + i(π/2 + 2πn) = i(π/2 + 2πn)
- Log(i) = iπ/2 (principal value)
- Log(-1) = iπ (principal value)
- log(-i) = i(-π/2 + 2πn)

**Branch Cuts:**
To make log z single-valued, we cut the plane (typically along negative real axis) and restrict to one branch.

**Properties (on a branch):**
- log(z₁z₂) = log z₁ + log z₂ (up to 2πi)
- log(z₁/z₂) = log z₁ - log z₂ (up to 2πi)
- d/dz(Log z) = 1/z (for z not on branch cut)

---

### 3. Complex Powers

**Definition:** For complex α:
$$\boxed{z^\alpha = e^{\alpha \log z}}$$

This is generally multi-valued because log z is multi-valued.

**Special Cases:**

**Integer powers (n ∈ ℤ):** Single-valued
$$z^n = e^{n \log z} = e^{n(\ln|z| + i\theta + 2\pi in)} = |z|^n e^{in\theta}$$
(The 2πin term gives e^{2πin²} = 1)

**Rational powers (p/q):** Finite-valued (q distinct values)
$$z^{p/q} = e^{(p/q)\log z} = |z|^{p/q} e^{i(p/q)(\theta + 2\pi n)}$$
Distinct values for n = 0, 1, ..., q-1

**Irrational/Complex powers:** Infinitely many values
$$z^\alpha = e^{\alpha(\ln|z| + i\theta + 2\pi in)} = |z|^\alpha e^{i\alpha\theta} e^{2\pi in\alpha}$$

**Examples:**

**i^i:**
$$i^i = e^{i \log i} = e^{i(i\pi/2 + 2\pi in)} = e^{-\pi/2 - 2\pi n}$$
Principal value: e^{-π/2} ≈ 0.2079 (real!)

**(-1)^i:**
$$(-1)^i = e^{i \log(-1)} = e^{i(i\pi + 2\pi in)} = e^{-\pi - 2\pi n}$$
Principal value: e^{-π} ≈ 0.0432

**2^{1+i}:**
$$2^{1+i} = e^{(1+i)\log 2} = e^{(1+i)(\ln 2 + 2\pi in)}$$
$$= e^{\ln 2 - 2\pi n + i(\ln 2 + 2\pi n)}$$
$$= 2e^{-2\pi n}(\cos(\ln 2 + 2\pi n) + i\sin(\ln 2 + 2\pi n))$$

---

### 4. Complex Trigonometric Functions

**Definitions via Euler's formula:**
$$\boxed{\cos z = \frac{e^{iz} + e^{-iz}}{2}, \quad \sin z = \frac{e^{iz} - e^{-iz}}{2i}}$$

**Properties:**
| Property | Formula |
|----------|---------|
| Entire | sin z, cos z are analytic on all ℂ |
| Pythagorean | sin²z + cos²z = 1 |
| Periodicity | sin(z + 2π) = sin z |
| Derivatives | d/dz(sin z) = cos z, d/dz(cos z) = -sin z |
| Zeros | sin z = 0 iff z = nπ; cos z = 0 iff z = (n+½)π |

**Unbounded!** Unlike real trig functions, complex sin and cos are **unbounded**:
$$|\sin(iy)| = |\frac{e^{-y} - e^y}{2i}| = \frac{e^y - e^{-y}}{2} = \sinh y \to \infty$$

**Explicit real/imaginary parts:**
$$\sin(x+iy) = \sin x \cosh y + i\cos x \sinh y$$
$$\cos(x+iy) = \cos x \cosh y - i\sin x \sinh y$$

---

### 5. Complex Hyperbolic Functions

**Definitions:**
$$\boxed{\cosh z = \frac{e^z + e^{-z}}{2}, \quad \sinh z = \frac{e^z - e^{-z}}{2}}$$

**Relation to trigonometric functions:**
$$\boxed{\cos(iz) = \cosh z, \quad \sin(iz) = i\sinh z}$$
$$\boxed{\cosh(iz) = \cos z, \quad \sinh(iz) = i\sin z}$$

**Identity:**
$$\cosh^2 z - \sinh^2 z = 1$$

**This unifies trig and hyperbolic functions in ℂ!**

---

### 6. Inverse Trigonometric Functions

**Solving sin w = z:**
$$\frac{e^{iw} - e^{-iw}}{2i} = z$$

Let u = e^{iw}:
$$u - 1/u = 2iz \implies u^2 - 2izu - 1 = 0$$
$$u = iz \pm \sqrt{1-z^2}$$
$$w = -i\log(iz + \sqrt{1-z^2})$$

**Definition:**
$$\boxed{\sin^{-1} z = -i\log(iz + \sqrt{1-z^2})}$$
$$\boxed{\cos^{-1} z = -i\log(z + \sqrt{z^2-1})}$$

These are multi-valued due to the logarithm and square root.

---

### 7. 🔬 Quantum Mechanics Connection

**Time Evolution Operator:**
$$U(t) = e^{-iHt/\hbar}$$

For energy eigenstate |E⟩:
$$U(t)|E\rangle = e^{-iEt/\hbar}|E\rangle$$

The phase rotates in complex plane with angular frequency E/ℏ!

**Rotation Operators:**
$$R_z(\theta) = e^{-i\theta S_z/\hbar} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

**Coherent States:**
$$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^\infty \frac{\alpha^n}{\sqrt{n!}}|n\rangle$$

where α ∈ ℂ parametrizes the coherent state.

**Analytic Continuation in Physics:**
- Wick rotation: t → -iτ connects quantum mechanics to statistical mechanics
- Imaginary time gives thermal states
- Matsubara frequencies in finite-temperature QFT

**Scattering Amplitudes:**
The S-matrix analytically continued to complex energy reveals:
- Bound state poles on real negative energy axis
- Resonance poles in lower half-plane

---

## ✏️ Worked Examples

### Example 1: Compute All Values of log(-1 - i)

**Solution:**
First, express -1 - i in polar form:
$$|-1-i| = \sqrt{1+1} = \sqrt{2}$$
$$\text{Arg}(-1-i) = -3\pi/4 \quad \text{(third quadrant)}$$

So -1 - i = √2 e^{-3πi/4}

Therefore:
$$\log(-1-i) = \ln\sqrt{2} + i(-3\pi/4 + 2\pi n)$$
$$= \frac{1}{2}\ln 2 + i(-3\pi/4 + 2\pi n), \quad n \in \mathbb{Z}$$

**Principal value:** Log(-1-i) = ½ln 2 - 3πi/4

---

### Example 2: Find All Values of (1+i)^{2i}

**Solution:**
$$\log(1+i) = \ln|1+i| + i(\text{arg}(1+i) + 2\pi n)$$
$$= \ln\sqrt{2} + i(\pi/4 + 2\pi n)$$
$$= \frac{1}{2}\ln 2 + i(\pi/4 + 2\pi n)$$

Therefore:
$$(1+i)^{2i} = e^{2i \cdot \log(1+i)}$$
$$= e^{2i[\frac{1}{2}\ln 2 + i(\pi/4 + 2\pi n)]}$$
$$= e^{i\ln 2 - (\pi/2 + 4\pi n)}$$
$$= e^{-\pi/2 - 4\pi n}(\cos(\ln 2) + i\sin(\ln 2))$$

**Principal value (n=0):**
$$(1+i)^{2i} = e^{-\pi/2}(\cos(\ln 2) + i\sin(\ln 2)) \approx 0.1556 + 0.1245i$$

---

### Example 3: Show sin z can have |sin z| > 1

**Solution:**
Consider z = iπ (purely imaginary):
$$\sin(i\pi) = \frac{e^{i(i\pi)} - e^{-i(i\pi)}}{2i} = \frac{e^{-\pi} - e^{\pi}}{2i}$$
$$= \frac{-(e^\pi - e^{-\pi})}{2i} = \frac{-2\sinh\pi}{2i} = i\sinh\pi$$

Since sinh π ≈ 11.5:
$$|\sin(i\pi)| = \sinh\pi \approx 11.5 > 1$$

Complex sine is unbounded!

---

### Example 4: Solve cos z = 3

**Solution:**
This has no real solution since |cos x| ≤ 1, but it has complex solutions!

Using the inverse:
$$z = \cos^{-1}(3) = -i\log(3 + \sqrt{9-1})$$
$$= -i\log(3 + 2\sqrt{2})$$
$$= -i[\ln(3+2\sqrt{2}) + 2\pi in]$$
$$= 2\pi n - i\ln(3+2\sqrt{2})$$

**Principal value:** z = -i ln(3 + 2√2) ≈ -1.763i

Verify: cos(-1.763i) = cosh(1.763) ≈ 3 ✓

---

## 🔧 Practice Problems

### Level 1: Basic Computations
1. Compute e^{2+iπ/4}.
2. Find all values of log(2i).
3. Express sin(2+i) in the form a + bi.
4. Compute cosh(iπ/3).

### Level 2: Powers and Logarithms
5. Find all values of (-i)^{1/3}.
6. Compute the principal value of (2+2i)^i.
7. Solve e^z = -2 for all z.
8. Find all z such that sin z = 2.

### Level 3: Identities and Mappings
9. Prove: |sin z|² = sin²x + sinh²y where z = x + iy.
10. Show that w = e^z maps the strip -π < Im(z) ≤ π onto ℂ\{0} bijectively.
11. Prove: cos(z₁ + z₂) = cos z₁ cos z₂ - sin z₁ sin z₂ using exponential definitions.
12. Find where Log z is not continuous.

### Level 4: Theory
13. Prove that z^α z^β = z^{α+β} fails for general complex powers. Give a counterexample.
14. Show that if f is entire and e^f is bounded, then f is constant.
15. Derive the formula for sin^{-1} z from scratch.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Complex exponential visualization
def visualize_exp():
    """Visualize how e^z maps the complex plane."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Domain: grid lines in z-plane
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-np.pi, np.pi, 400)
    
    # Horizontal lines (constant y = c)
    for c in np.linspace(-np.pi, np.pi, 13):
        z = x + 1j * c
        w = np.exp(z)
        axes[0].plot(np.real(z), np.imag(z), 'b-', lw=0.5)
        axes[1].plot(np.real(w), np.imag(w), 'b-', lw=0.5)
    
    # Vertical lines (constant x = c)
    for c in np.linspace(-2, 2, 17):
        z = c + 1j * y
        w = np.exp(z)
        axes[0].plot(np.real(z), np.imag(z), 'r-', lw=0.5)
        axes[1].plot(np.real(w), np.imag(w), 'r-', lw=0.5)
    
    axes[0].set_title('z-plane (domain)\nHorizontal: y = const, Vertical: x = const')
    axes[0].set_xlabel('Re(z)'); axes[0].set_ylabel('Im(z)')
    axes[0].set_xlim(-2.5, 2.5); axes[0].set_ylim(-4, 4)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    axes[1].set_title('w = e^z (image)\nHorizontal → rays, Vertical → circles')
    axes[1].set_xlabel('Re(w)'); axes[1].set_ylabel('Im(w)')
    axes[1].set_xlim(-10, 10); axes[1].set_ylim(-10, 10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('complex_exp_mapping.png', dpi=150)
    plt.show()

visualize_exp()

# Multi-valued logarithm visualization
def visualize_log_branches():
    """Show multiple branches of log z."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    r = np.linspace(0.1, 3, 100)
    theta = np.linspace(-np.pi, np.pi, 200)
    R, Theta = np.meshgrid(r, theta)
    Z = R * np.exp(1j * Theta)
    
    # Different branches of log z (n = -1, 0, 1, 2)
    for ax, n in zip(axes.flat, [-1, 0, 1, 2]):
        W = np.log(np.abs(Z)) + 1j * (Theta + 2*np.pi*n)
        
        # Plot real part of log z for this branch
        im = ax.contourf(np.real(Z), np.imag(Z), np.imag(W), 
                         levels=30, cmap='hsv')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Im(log z), branch n = {n}\n' + 
                     f'Range: [{-np.pi + 2*np.pi*n:.2f}, {np.pi + 2*np.pi*n:.2f}]')
        ax.set_xlabel('Re(z)'); ax.set_ylabel('Im(z)')
        ax.axis('equal')
    
    plt.suptitle('Multiple Branches of log z', fontsize=14)
    plt.tight_layout()
    plt.savefig('log_branches.png', dpi=150)
    plt.show()

visualize_log_branches()

# Complex powers: i^i
def explore_complex_powers():
    """Explore multi-valued nature of complex powers."""
    print("=" * 60)
    print("COMPLEX POWERS")
    print("=" * 60)
    
    # i^i
    print("\n1. i^i = e^(i * log i)")
    print("   log i = ln|i| + i(π/2 + 2πn) = i(π/2 + 2πn)")
    print("   i^i = e^(i * i(π/2 + 2πn)) = e^(-(π/2 + 2πn))")
    print("\n   Values for different n:")
    for n in range(-2, 4):
        val = np.exp(-(np.pi/2 + 2*np.pi*n))
        print(f"     n = {n:2d}: i^i = {val:.6f}")
    print(f"\n   Principal value (n=0): i^i = {np.exp(-np.pi/2):.6f}")
    print("   Note: i^i is REAL!")
    
    # 2^(1+i)
    print("\n2. 2^(1+i) = e^((1+i) * log 2)")
    print("   log 2 = ln 2 + 2πin")
    for n in range(3):
        exponent = (1 + 1j) * (np.log(2) + 2j*np.pi*n)
        val = np.exp(exponent)
        print(f"     n = {n}: 2^(1+i) = {val.real:.4f} + {val.imag:.4f}i")

explore_complex_powers()

# Visualize complex sine
def visualize_complex_sine():
    """Show how complex sine differs from real sine."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x = np.linspace(-2*np.pi, 2*np.pi, 300)
    y = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    W = np.sin(Z)
    
    # Real part
    im0 = axes[0, 0].contourf(X, Y, np.real(W), levels=30, cmap='RdBu_r')
    plt.colorbar(im0, ax=axes[0, 0])
    axes[0, 0].set_title('Re(sin z) = sin x cosh y')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')
    
    # Imaginary part
    im1 = axes[0, 1].contourf(X, Y, np.imag(W), levels=30, cmap='RdBu_r')
    plt.colorbar(im1, ax=axes[0, 1])
    axes[0, 1].set_title('Im(sin z) = cos x sinh y')
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')
    
    # Magnitude (shows it's unbounded)
    mag = np.abs(W)
    mag_clipped = np.clip(mag, 0, 10)  # Clip for visualization
    im2 = axes[1, 0].contourf(X, Y, mag_clipped, levels=30, cmap='hot')
    plt.colorbar(im2, ax=axes[1, 0])
    axes[1, 0].set_title('|sin z| (clipped at 10)\nNote: unbounded!')
    axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('y')
    
    # Compare along real and imaginary axes
    axes[1, 1].plot(x, np.sin(x), 'b-', lw=2, label='sin(x) on real axis')
    y_vals = np.linspace(-2, 2, 100)
    axes[1, 1].plot(np.sinh(y_vals) * 1j, y_vals, 'r-', lw=2, label='|sin(iy)| = sinh(y)')
    axes[1, 1].plot(y_vals, np.sinh(y_vals), 'r--', lw=2)
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Position')
    axes[1, 1].set_title('Real: bounded; Imaginary: unbounded')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complex_sine.png', dpi=150)
    plt.show()

visualize_complex_sine()

# Domain coloring for complex functions
def domain_coloring(f, title, x_range=(-3, 3), y_range=(-3, 3), n=500):
    """
    Create domain coloring plot of complex function.
    Hue = argument, Brightness = modulus (log scale)
    """
    x = np.linspace(x_range[0], x_range[1], n)
    y = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    W = f(Z)
    
    # Convert to HSV
    H = (np.angle(W) + np.pi) / (2 * np.pi)  # Hue from argument
    S = np.ones_like(H)  # Full saturation
    V = 2 * np.arctan(np.abs(W)) / np.pi  # Value from modulus (compressed)
    
    # Convert HSV to RGB
    HSV = np.dstack([H, S, V])
    RGB = hsv_to_rgb(HSV)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(RGB, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
               origin='lower')
    plt.title(title)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.colorbar(label='Argument (hue)')
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150)
    plt.show()

# Visualize various functions
domain_coloring(lambda z: np.exp(z), 'exp(z)', (-3, 3), (-np.pi, np.pi))
domain_coloring(lambda z: np.log(z + 0.001), 'log(z)')
domain_coloring(lambda z: np.sin(z), 'sin(z)')
domain_coloring(lambda z: z**0.5, 'z^(1/2) principal branch')

# Euler's formula animation preparation
def euler_formula_spiral():
    """Visualize e^(it) tracing the unit circle."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    t = np.linspace(0, 4*np.pi, 500)
    z = np.exp(1j * t)
    
    # Unit circle
    axes[0].plot(np.real(z), np.imag(z), 'b-', lw=2)
    axes[0].scatter([1, 0, -1, 0], [0, 1, 0, -1], c='red', s=100, zorder=5)
    axes[0].annotate('e^0 = 1', (1, 0), xytext=(1.2, 0.2), fontsize=12)
    axes[0].annotate('e^(iπ/2) = i', (0, 1), xytext=(0.2, 1.2), fontsize=12)
    axes[0].annotate('e^(iπ) = -1', (-1, 0), xytext=(-1.5, 0.2), fontsize=12)
    axes[0].annotate('e^(3iπ/2) = -i', (0, -1), xytext=(0.2, -1.3), fontsize=12)
    axes[0].set_xlim(-2, 2); axes[0].set_ylim(-2, 2)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', lw=0.5)
    axes[0].axvline(x=0, color='k', lw=0.5)
    axes[0].set_title('e^(it) traces the unit circle\nas t goes from 0 to 2π')
    axes[0].set_xlabel('Re'); axes[0].set_ylabel('Im')
    
    # cos(t) and sin(t) as projections
    axes[1].plot(t, np.cos(t), 'b-', lw=2, label='cos t = Re(e^(it))')
    axes[1].plot(t, np.sin(t), 'r-', lw=2, label='sin t = Im(e^(it))')
    axes[1].axhline(y=0, color='k', lw=0.5)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Trig functions from Euler\'s formula')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 4*np.pi)
    
    plt.tight_layout()
    plt.savefig('euler_formula.png', dpi=150)
    plt.show()

euler_formula_spiral()

# QM Application: Time evolution
def quantum_time_evolution():
    """Visualize quantum state evolution using e^(-iEt/ℏ)."""
    print("\n" + "=" * 60)
    print("QUANTUM TIME EVOLUTION")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    t = np.linspace(0, 10, 500)
    hbar = 1  # Natural units
    
    # Single energy state
    E = 2.0
    psi = np.exp(-1j * E * t / hbar)
    
    axes[0].plot(t, np.real(psi), 'b-', lw=2, label='Re(ψ)')
    axes[0].plot(t, np.imag(psi), 'r-', lw=2, label='Im(ψ)')
    axes[0].plot(t, np.abs(psi), 'k--', lw=2, label='|ψ|')
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Time evolution: ψ(t) = e^(-iEt/ℏ)\nE = {E}, ℏ = 1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Superposition of two states (creates beating)
    E1, E2 = 2.0, 2.5
    c1, c2 = 1/np.sqrt(2), 1/np.sqrt(2)
    psi_super = c1 * np.exp(-1j * E1 * t) + c2 * np.exp(-1j * E2 * t)
    
    axes[1].plot(t, np.abs(psi_super)**2, 'g-', lw=2)
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Probability |ψ|²')
    axes[1].set_title(f'Superposition beating\n|ψ|² oscillates at frequency (E2-E1)/ℏ')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_evolution.png', dpi=150)
    plt.show()
    
    print("\nKey insight: The complex exponential e^(-iEt/ℏ) creates")
    print("oscillating phases that, in superpositions, lead to")
    print("observable quantum interference effects!")

quantum_time_evolution()
```

---

## 📝 Summary

### Key Formulas

| Function | Definition | Key Property |
|----------|------------|--------------|
| e^z | e^x(cos y + i sin y) | Entire, period 2πi |
| log z | ln\|z\| + i arg z | Multi-valued! |
| z^α | e^(α log z) | Multi-valued for non-integer α |
| sin z | (e^{iz} - e^{-iz})/2i | Entire, unbounded |
| cos z | (e^{iz} + e^{-iz})/2 | Entire, unbounded |

### Critical Insights
- Complex exponential is periodic with period 2πi
- Complex logarithm has infinitely many values differing by 2πi
- Complex powers z^α are generally multi-valued
- Complex trig functions are unbounded (unlike real versions)
- Euler's formula unifies exponential and trigonometric functions

---

## ✅ Daily Checklist

- [ ] Master Euler's formula and its consequences
- [ ] Compute with multi-valued logarithm
- [ ] Evaluate complex powers (e.g., i^i)
- [ ] Work with complex trig functions
- [ ] Understand branch cuts for log z
- [ ] Connect to quantum time evolution
- [ ] Complete domain coloring visualizations

---

## 🔮 Preview: Day 131

Tomorrow we tackle **complex integration** — the heart of complex analysis. We'll develop line integrals in ℂ and discover the remarkable **Cauchy's Theorem**: the integral of an analytic function around any closed curve is zero!
