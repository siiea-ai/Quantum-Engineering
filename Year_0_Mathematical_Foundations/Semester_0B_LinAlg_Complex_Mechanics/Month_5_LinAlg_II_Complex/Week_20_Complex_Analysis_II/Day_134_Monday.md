# Day 134: Laurent Series — Expanding Around Singularities

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Laurent Series |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand why Taylor series fail at singularities
2. Derive and compute Laurent series expansions
3. Identify the principal part and analytic part
4. Determine the annulus of convergence
5. Extract residues from Laurent series
6. Connect to physics: multipole expansions

---

## 📚 Required Reading

### Primary Text: Churchill & Brown
- **Chapter 5**: Series (pp. 143-180)

### Alternative: Ahlfors
- **Chapter 4.3**: Laurent Series

### Physics Connection
- **Jackson, Classical Electrodynamics**: Multipole expansions

---

## 📖 Core Content: Theory and Concepts

### 1. Motivation: When Taylor Series Fail

**Taylor series** require analyticity at the expansion point:
$$f(z) = \sum_{n=0}^{\infty} \frac{f^{(n)}(z_0)}{n!}(z-z_0)^n$$

This fails if f has a singularity at z₀. But what if f is analytic in an **annulus** around z₀?

**Example:** f(z) = 1/z is not analytic at z = 0, but is analytic for 0 < |z| < ∞.

**Question:** Can we still expand f in powers of z?

---

### 2. Laurent Series Definition

**Theorem (Laurent Series):**
If f is analytic in the annulus R₁ < |z - z₀| < R₂, then f has a unique expansion:

$$\boxed{f(z) = \sum_{n=-\infty}^{\infty} a_n(z-z_0)^n = \sum_{n=0}^{\infty} a_n(z-z_0)^n + \sum_{n=1}^{\infty} a_{-n}(z-z_0)^{-n}}$$

where the coefficients are given by:
$$a_n = \frac{1}{2\pi i}\oint_C \frac{f(z)}{(z-z_0)^{n+1}}\,dz$$

for any simple closed contour C in the annulus encircling z₀.

**Key Parts:**
- **Analytic part:** $\sum_{n=0}^{\infty} a_n(z-z_0)^n$ (non-negative powers)
- **Principal part:** $\sum_{n=1}^{\infty} a_{-n}(z-z_0)^{-n}$ (negative powers)

---

### 3. Computing Laurent Series

**Method 1: Direct Calculation**
Use the coefficient formula or known expansions.

**Method 2: Manipulation of Known Series**
Use Taylor series of component functions.

**Method 3: Partial Fractions**
Decompose rational functions.

---

### 4. Examples

**Example 1: f(z) = 1/z around z₀ = 0**

Already in Laurent form:
$$\frac{1}{z} = z^{-1}$$

Coefficients: a₋₁ = 1, all others = 0.
Valid for 0 < |z| < ∞.

---

**Example 2: f(z) = 1/(z(z-1)) around z₀ = 0**

Use partial fractions:
$$\frac{1}{z(z-1)} = \frac{-1}{z} + \frac{1}{z-1}$$

For 0 < |z| < 1, expand 1/(z-1) = -1/(1-z) = -∑_{n=0}^∞ z^n:
$$\frac{1}{z(z-1)} = -\frac{1}{z} - \sum_{n=0}^{\infty} z^n = -z^{-1} - 1 - z - z^2 - \cdots$$

**Principal part:** -z⁻¹
**Analytic part:** -1 - z - z² - ...

---

**Example 3: f(z) = e^{1/z} around z₀ = 0**

Substitute w = 1/z into eʷ = ∑ wⁿ/n!:
$$e^{1/z} = \sum_{n=0}^{\infty} \frac{1}{n! z^n} = 1 + \frac{1}{z} + \frac{1}{2!z^2} + \frac{1}{3!z^3} + \cdots$$

**Infinitely many negative powers!** This is an essential singularity.

Valid for 0 < |z| < ∞.

---

**Example 4: f(z) = 1/(z-1)² around z₀ = 0**

For |z| < 1:
$$\frac{1}{(z-1)^2} = \frac{1}{(1-z)^2} = \frac{d}{dz}\left(\frac{1}{1-z}\right) = \frac{d}{dz}\sum_{n=0}^{\infty} z^n = \sum_{n=1}^{\infty} nz^{n-1}$$
$$= 1 + 2z + 3z^2 + 4z^3 + \cdots$$

This is actually a Taylor series (no principal part) — the singularity at z = 1 is outside |z| < 1.

---

**Example 5: f(z) = 1/(z²(z-2)) around z₀ = 0**

Partial fractions:
$$\frac{1}{z^2(z-2)} = \frac{A}{z} + \frac{B}{z^2} + \frac{C}{z-2}$$

Solving: A = 1/4, B = -1/2, C = -1/4

For 0 < |z| < 2:
$$\frac{1}{z-2} = -\frac{1}{2}\cdot\frac{1}{1-z/2} = -\frac{1}{2}\sum_{n=0}^{\infty}\left(\frac{z}{2}\right)^n$$

Laurent series:
$$f(z) = \frac{1/4}{z} - \frac{1/2}{z^2} - \frac{1}{4}\cdot\frac{1}{2}\sum_{n=0}^{\infty}\frac{z^n}{2^n}$$
$$= -\frac{1}{2z^2} + \frac{1}{4z} - \frac{1}{8} - \frac{z}{16} - \frac{z^2}{32} - \cdots$$

---

### 5. Different Annuli, Different Series

**Critical Point:** The same function can have different Laurent series in different annuli!

**Example: f(z) = 1/(z(z-1)) around z₀ = 0**

**Region I: 0 < |z| < 1**
$$f(z) = -\frac{1}{z} - 1 - z - z^2 - \cdots$$

**Region II: |z| > 1**
Expand 1/(z-1) = (1/z)·1/(1-1/z) = (1/z)∑(1/z)ⁿ:
$$\frac{1}{z-1} = \sum_{n=1}^{\infty} z^{-n}$$

So:
$$f(z) = -\frac{1}{z} + \frac{1}{z}\sum_{n=1}^{\infty} z^{-n} = -z^{-1} + z^{-2} + z^{-3} + \cdots$$

Different series in different regions!

---

### 6. The Residue

**Definition:** The **residue** of f at z₀ is the coefficient a₋₁ in the Laurent series:

$$\boxed{\text{Res}[f, z_0] = a_{-1} = \frac{1}{2\pi i}\oint_C f(z)\,dz}$$

**Why Important:** From the Laurent series, only the z⁻¹ term contributes to contour integrals:
$$\oint_C f(z)\,dz = \oint_C \sum_{n=-\infty}^{\infty} a_n(z-z_0)^n\,dz = a_{-1} \cdot 2\pi i$$

(All other terms integrate to zero around a closed contour.)

---

### 7. 🔬 Physics Connection: Multipole Expansions

**Electrostatic Potential:**
The potential due to a charge distribution can be expanded:
$$\phi(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\sum_{l=0}^{\infty} \frac{q_l}{r^{l+1}} P_l(\cos\theta)$$

This is a Laurent-like expansion in 1/r!

- l = 0: Monopole (total charge)
- l = 1: Dipole
- l = 2: Quadrupole

**Green's Functions:**
The propagator G(r) often has a Laurent expansion in momentum space:
$$G(k) = \frac{1}{k^2 - m^2} = -\frac{1}{m^2}\sum_{n=0}^{\infty}\left(\frac{k^2}{m^2}\right)^n$$

---

## ✏️ Worked Examples

### Example 1: Laurent Series of sin(z)/z³

**Solution:**
Start with the Taylor series of sin z:
$$\sin z = z - \frac{z^3}{3!} + \frac{z^5}{5!} - \frac{z^7}{7!} + \cdots$$

Divide by z³:
$$\frac{\sin z}{z^3} = \frac{1}{z^2} - \frac{1}{3!} + \frac{z^2}{5!} - \frac{z^4}{7!} + \cdots$$

**Residue:** a₋₁ = 0 (no z⁻¹ term)

---

### Example 2: Laurent Series of 1/(z² + 1) around z₀ = i

**Solution:**
Factor: 1/(z² + 1) = 1/((z-i)(z+i))

Near z = i, let w = z - i:
$$\frac{1}{z^2+1} = \frac{1}{(z-i)(z+i)} = \frac{1}{w(w+2i)}$$

$$= \frac{1}{w} \cdot \frac{1}{2i(1 + w/2i)} = \frac{1}{2iw}\sum_{n=0}^{\infty}\left(-\frac{w}{2i}\right)^n$$

$$= \frac{1}{2i(z-i)} - \frac{1}{(2i)^2} + \frac{z-i}{(2i)^3} - \cdots$$

**Residue at z = i:** a₋₁ = 1/(2i) = -i/2

---

### Example 3: Find the Residue of (z² + 1)/((z-1)²(z+2))

**Solution:**
This has a pole of order 2 at z = 1 and a simple pole at z = -2.

**At z = -2 (simple pole):**
$$\text{Res}_{z=-2} = \lim_{z \to -2} (z+2) \cdot \frac{z^2+1}{(z-1)^2(z+2)} = \frac{(-2)^2+1}{(-2-1)^2} = \frac{5}{9}$$

**At z = 1 (double pole):**
$$\text{Res}_{z=1} = \lim_{z \to 1} \frac{d}{dz}\left[(z-1)^2 \cdot \frac{z^2+1}{(z-1)^2(z+2)}\right]$$
$$= \lim_{z \to 1} \frac{d}{dz}\left[\frac{z^2+1}{z+2}\right] = \lim_{z \to 1} \frac{2z(z+2) - (z^2+1)}{(z+2)^2}$$
$$= \frac{2(3) - 2}{9} = \frac{4}{9}$$

---

## 🔧 Practice Problems

### Level 1: Basic Laurent Series
1. Find the Laurent series of 1/(z-2) around z₀ = 0 for |z| < 2.
2. Find the Laurent series of 1/z² around z₀ = 0.
3. Expand z/(z-1) in the region |z| > 1.

### Level 2: Computing Laurent Series
4. Find the Laurent series of 1/((z-1)(z-2)) around z₀ = 0 for 1 < |z| < 2.
5. Expand e^z/z² around z₀ = 0.
6. Find the Laurent series of cos(1/z) around z₀ = 0.

### Level 3: Residues
7. Find the residue of z²/(z-1)³ at z = 1.
8. Compute the residue of e^z/(z²+1) at z = i.
9. Find all residues of 1/(z⁴-1).

### Level 4: Theory
10. Prove that the Laurent series is unique.
11. Show that if f has a pole of order n at z₀, then the principal part has exactly n terms.
12. Find the Laurent expansion of 1/(eᶻ - 1) around z = 0.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def laurent_series_visualization():
    """Visualize Laurent series and convergence."""
    
    print("=" * 60)
    print("LAURENT SERIES EXPLORATION")
    print("=" * 60)
    
    # Example 1: 1/(z(z-1)) in different regions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Function definition
    f = lambda z: 1/(z*(z-1))
    
    # Region |z| < 1 (excluding 0)
    def laurent_region1(z, n_terms=20):
        """Laurent series for 0 < |z| < 1"""
        result = -1/z  # Principal part
        for n in range(n_terms):
            result -= z**n  # Analytic part
        return result
    
    # Region |z| > 1
    def laurent_region2(z, n_terms=20):
        """Laurent series for |z| > 1"""
        result = 0
        for n in range(1, n_terms+1):
            result += z**(-n) - z**(-n-1)
        return result
    
    # Visualize convergence
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Test on circle |z| = 0.5 (inside unit circle)
    r = 0.5
    z_test = r * np.exp(1j * theta)
    
    f_exact = f(z_test)
    f_laurent = laurent_region1(z_test, 10)
    
    axes[0, 0].plot(theta, np.real(f_exact), 'b-', lw=2, label='Exact')
    axes[0, 0].plot(theta, np.real(f_laurent), 'r--', lw=2, label='Laurent (10 terms)')
    axes[0, 0].set_title(f'1/(z(z-1)) on |z| = {r}\nRegion: 0 < |z| < 1')
    axes[0, 0].set_xlabel('θ')
    axes[0, 0].set_ylabel('Re[f]')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test on circle |z| = 2 (outside unit circle)
    r = 2
    z_test = r * np.exp(1j * theta)
    
    f_exact = f(z_test)
    f_laurent = laurent_region2(z_test, 10)
    
    axes[0, 1].plot(theta, np.real(f_exact), 'b-', lw=2, label='Exact')
    axes[0, 1].plot(theta, np.real(f_laurent), 'r--', lw=2, label='Laurent (10 terms)')
    axes[0, 1].set_title(f'1/(z(z-1)) on |z| = {r}\nRegion: |z| > 1')
    axes[0, 1].set_xlabel('θ')
    axes[0, 1].set_ylabel('Re[f]')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Example 2: e^(1/z) - essential singularity
    def exp_inv_z_laurent(z, n_terms=20):
        """Laurent series of e^(1/z)"""
        result = np.zeros_like(z, dtype=complex)
        for n in range(n_terms):
            result += z**(-n) / np.math.factorial(n)
        return result
    
    r = 0.5
    z_test = r * np.exp(1j * theta)
    
    f_exact = np.exp(1/z_test)
    
    axes[1, 0].set_title('e^(1/z) Laurent convergence')
    for n_terms in [5, 10, 20, 50]:
        f_laurent = exp_inv_z_laurent(z_test, n_terms)
        error = np.abs(f_laurent - f_exact)
        axes[1, 0].semilogy(theta, error, label=f'{n_terms} terms')
    axes[1, 0].set_xlabel('θ')
    axes[1, 0].set_ylabel('|Error|')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Visualize different annuli
    ax = axes[1, 1]
    
    # Draw singularities
    ax.scatter([0, 1], [0, 0], c='red', s=100, marker='x', zorder=5, label='Singularities')
    
    # Draw annulus boundaries
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(0.5*np.cos(theta), 0.5*np.sin(theta), 'b-', lw=2, label='|z| = 0.5')
    ax.plot(np.cos(theta), np.sin(theta), 'g--', lw=2, label='|z| = 1')
    ax.plot(2*np.cos(theta), 2*np.sin(theta), 'r-', lw=2, label='|z| = 2')
    
    ax.fill_between(np.cos(theta)*0.1, -np.sin(theta)*0.1, np.sin(theta)*0.1, alpha=0.3, color='blue')
    ax.annotate('Region I\n0 < |z| < 1', (0.3, 0.3), fontsize=10)
    ax.annotate('Region II\n|z| > 1', (1.5, 0.5), fontsize=10)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.set_title('Different Laurent series in different regions')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laurent_series.png', dpi=150)
    plt.show()

laurent_series_visualization()

# Residue calculation
def compute_residue_examples():
    """Demonstrate residue calculations."""
    
    print("\n" + "=" * 60)
    print("RESIDUE CALCULATIONS")
    print("=" * 60)
    
    # Method 1: From Laurent series coefficient
    print("\n1. Residue from Laurent series")
    print("   f(z) = sin(z)/z³")
    print("   sin(z) = z - z³/6 + z⁵/120 - ...")
    print("   sin(z)/z³ = 1/z² - 1/6 + z²/120 - ...")
    print("   Res[f, 0] = coefficient of 1/z = 0")
    
    # Method 2: Simple pole formula
    print("\n2. Simple pole formula: Res[f, z₀] = lim (z-z₀)f(z)")
    print("   f(z) = 1/(z² + 1) at z = i")
    print("   Res = lim_{z→i} (z-i)/(z²+1) = lim_{z→i} (z-i)/((z-i)(z+i))")
    print("       = 1/(2i) = -i/2")
    
    # Numerical verification
    def numerical_residue(f, z0, r=0.001, n_points=1000):
        """Compute residue numerically via contour integral."""
        theta = np.linspace(0, 2*np.pi, n_points)
        z = z0 + r * np.exp(1j * theta)
        dz = 1j * r * np.exp(1j * theta)
        integral = np.trapz(f(z) * dz, theta) / (2 * np.pi * 1j)
        return integral
    
    f = lambda z: 1/(z**2 + 1)
    res_numerical = numerical_residue(f, 1j)
    res_exact = -1j/2
    print(f"   Numerical: {res_numerical:.6f}")
    print(f"   Exact: {res_exact:.6f}")
    
    # Method 3: Higher order pole
    print("\n3. Double pole formula")
    print("   f(z) = e^z/(z-1)² at z = 1")
    print("   Res = lim_{z→1} d/dz[(z-1)² · e^z/(z-1)²] = lim_{z→1} d/dz[e^z] = e")
    
    f = lambda z: np.exp(z)/(z-1)**2
    res_numerical = numerical_residue(f, 1)
    print(f"   Numerical: {res_numerical:.6f}")
    print(f"   Exact: e = {np.e:.6f}")

compute_residue_examples()
```

---

## 📝 Summary

### Key Concepts

1. **Laurent Series**: f(z) = Σ aₙ(z-z₀)ⁿ for n from -∞ to +∞
2. **Principal Part**: Negative power terms (characterize singularity)
3. **Analytic Part**: Non-negative power terms
4. **Residue**: Coefficient a₋₁ of (z-z₀)⁻¹
5. **Different Regions**: Same function, different Laurent series

### Residue Formulas

| Singularity Type | Residue Formula |
|------------------|-----------------|
| Simple pole | lim_{z→z₀} (z-z₀)f(z) |
| Pole of order n | (1/(n-1)!) lim_{z→z₀} d^{n-1}/dz^{n-1}[(z-z₀)ⁿf(z)] |
| From series | Coefficient of (z-z₀)⁻¹ |

---

## ✅ Daily Checklist

- [ ] Understand why Laurent series generalize Taylor series
- [ ] Compute Laurent series by various methods
- [ ] Identify principal and analytic parts
- [ ] Recognize different series in different annuli
- [ ] Calculate residues
- [ ] Connect to multipole expansions in physics
- [ ] Complete computational exercises

---

## 🔮 Preview: Day 135

Tomorrow we study the **classification of singularities** — removable, poles, and essential — and discover how the Laurent series reveals the nature of each type!
