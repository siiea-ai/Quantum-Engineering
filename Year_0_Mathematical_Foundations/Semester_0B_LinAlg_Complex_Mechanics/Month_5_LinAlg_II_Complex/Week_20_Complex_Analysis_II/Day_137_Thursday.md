# Day 137: Evaluating Real Integrals via Residues

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Contour Methods for Real Integrals |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Evaluate improper integrals of rational functions
2. Handle trigonometric integrals over [0, 2π]
3. Compute integrals involving sin x/x and similar
4. Apply Jordan's lemma
5. Choose appropriate contours for different integral types
6. Connect to physics integrals (Fourier transforms, etc.)

---

## 📚 Required Reading

### Primary Text: Churchill & Brown
- **Chapter 7**: Applications of Residues (pp. 231-270)

### Alternative: Arfken & Weber
- **Section 11.8**: Evaluation of Definite Integrals

---

## 📖 Core Content: Theory and Concepts

### 1. Type I: Rational Functions on (-∞, ∞)

**Goal:** Evaluate ∫_{-∞}^{∞} p(x)/q(x) dx where p, q are polynomials.

**Conditions:**
1. q(x) ≠ 0 for all real x (no real poles)
2. deg(q) ≥ deg(p) + 2 (ensures convergence)

**Method:** Use semicircular contour in upper (or lower) half-plane.

**Contour:** C = [-R, R] ∪ Γ_R where Γ_R is the semicircle from R to -R in the upper half-plane.

**Key Result:** If deg(q) ≥ deg(p) + 2, then:
$$\int_{\Gamma_R} \frac{p(z)}{q(z)} dz \to 0 \text{ as } R \to \infty$$

**Formula:**
$$\boxed{\int_{-\infty}^{\infty} \frac{p(x)}{q(x)} dx = 2\pi i \sum_{\text{Im}(z_k) > 0} \text{Res}\left[\frac{p}{q}, z_k\right]}$$

(Sum over poles in upper half-plane only!)

---

### Example 1: ∫_{-∞}^{∞} dx/(1+x²)

**Solution:**
f(z) = 1/(1+z²) = 1/((z-i)(z+i))

Poles: z = ±i. Only z = i is in upper half-plane.

Residue at z = i:
$$\text{Res}[f, i] = \lim_{z \to i} (z-i) \cdot \frac{1}{(z-i)(z+i)} = \frac{1}{2i}$$

$$\int_{-\infty}^{\infty} \frac{dx}{1+x^2} = 2\pi i \cdot \frac{1}{2i} = \boxed{\pi}$$

---

### Example 2: ∫_{-∞}^{∞} dx/(1+x⁴)

**Solution:**
1 + z⁴ = 0 → z⁴ = -1 = e^{iπ}

Roots: z = e^{i(π+2πk)/4} for k = 0, 1, 2, 3
- k=0: z₁ = e^{iπ/4} = (1+i)/√2 (upper half-plane)
- k=1: z₂ = e^{i3π/4} = (-1+i)/√2 (upper half-plane)
- k=2: z₃ = e^{i5π/4} (lower half-plane)
- k=3: z₄ = e^{i7π/4} (lower half-plane)

For simple poles: Res[1/q, zₖ] = 1/q'(zₖ) = 1/(4zₖ³) = zₖ/(4zₖ⁴) = -zₖ/4

$$\text{Res}[f, z_1] = -\frac{e^{i\pi/4}}{4}, \quad \text{Res}[f, z_2] = -\frac{e^{i3\pi/4}}{4}$$

Sum: -(1/4)[e^{iπ/4} + e^{i3π/4}] = -(1/4)[(1+i)/√2 + (-1+i)/√2] = -(1/4)(2i/√2) = -i/(2√2)

$$\int_{-\infty}^{\infty} \frac{dx}{1+x^4} = 2\pi i \cdot \left(-\frac{i}{2\sqrt{2}}\right) = \frac{\pi}{\sqrt{2}} = \boxed{\frac{\pi\sqrt{2}}{2}}$$

---

### 2. Type II: Trigonometric Integrals over [0, 2π]

**Goal:** Evaluate ∫₀^{2π} R(cos θ, sin θ) dθ

**Method:** Substitute z = e^{iθ}:
- cos θ = (z + 1/z)/2
- sin θ = (z - 1/z)/(2i)
- dθ = dz/(iz)

The integral becomes a contour integral over |z| = 1.

---

### Example 3: ∫₀^{2π} dθ/(2 + cos θ)

**Solution:**
Substitute z = e^{iθ}:
$$\frac{1}{2 + \cos\theta} = \frac{1}{2 + (z+1/z)/2} = \frac{2}{4 + z + 1/z} = \frac{2z}{z^2 + 4z + 1}$$

$$d\theta = \frac{dz}{iz}$$

$$\int_0^{2\pi} \frac{d\theta}{2+\cos\theta} = \oint_{|z|=1} \frac{2z}{z^2+4z+1} \cdot \frac{dz}{iz} = \frac{2}{i}\oint_{|z|=1} \frac{dz}{z^2+4z+1}$$

Find poles: z² + 4z + 1 = 0 → z = (-4 ± √12)/2 = -2 ± √3

- z₁ = -2 + √3 ≈ -0.27 (inside |z| = 1)
- z₂ = -2 - √3 ≈ -3.73 (outside |z| = 1)

Residue at z₁:
$$\text{Res} = \frac{1}{2z_1 + 4} = \frac{1}{2(-2+\sqrt{3}) + 4} = \frac{1}{2\sqrt{3}}$$

$$\int_0^{2\pi} \frac{d\theta}{2+\cos\theta} = \frac{2}{i} \cdot 2\pi i \cdot \frac{1}{2\sqrt{3}} = \boxed{\frac{2\pi}{\sqrt{3}}}$$

---

### 3. Type III: Integrals with e^{iax} (Fourier-type)

**Goal:** Evaluate ∫_{-∞}^{∞} f(x) e^{iax} dx for a > 0

**Jordan's Lemma:**
If f(z) → 0 uniformly as |z| → ∞ in the upper half-plane, and a > 0, then:
$$\int_{\Gamma_R} f(z) e^{iaz} dz \to 0 \text{ as } R \to \infty$$

This works even if f(z) ~ 1/|z| (not just 1/|z|²)!

**Key:** For a > 0, use upper half-plane. For a < 0, use lower half-plane.

---

### Example 4: ∫_{-∞}^{∞} cos(x)/(1+x²) dx

**Solution:**
Write cos x = Re(e^{ix}). Consider:
$$I = \int_{-\infty}^{\infty} \frac{e^{ix}}{1+x^2} dx$$

The real part is what we want.

Poles of 1/(1+z²): z = ±i. Only i is in upper half-plane.

By Jordan's lemma (a = 1 > 0), semicircle contributes 0.

$$\text{Res}\left[\frac{e^{iz}}{1+z^2}, i\right] = \frac{e^{i \cdot i}}{2i} = \frac{e^{-1}}{2i}$$

$$I = 2\pi i \cdot \frac{e^{-1}}{2i} = \frac{\pi}{e}$$

Since I is real, we have:
$$\int_{-\infty}^{\infty} \frac{\cos x}{1+x^2} dx = \text{Re}(I) = \boxed{\frac{\pi}{e}}$$

---

### 4. Type IV: Integrals with sin(x)/x (Dirichlet-type)

**Goal:** Evaluate ∫_{-∞}^{∞} sin(x)/x dx

**Problem:** sin z/z has a removable singularity at z = 0 (on real axis), but it's still tricky.

**Method:** Use indented contour avoiding z = 0.

**The Integral:**
$$\int_{-\infty}^{\infty} \frac{\sin x}{x} dx = \pi$$

**Proof sketch:**
Consider ∮ e^{iz}/z dz over contour:
- [-R, -ε] on real axis
- Small semicircle C_ε around origin (avoiding it)
- [ε, R] on real axis  
- Large semicircle Γ_R

As R → ∞ and ε → 0:
- Γ_R contribution → 0 (Jordan's lemma)
- C_ε contribution → -πi (half the residue at a simple pole)

$$\int_{-\infty}^{\infty} \frac{e^{ix}}{x} dx = \pi i$$

Taking imaginary part:
$$\int_{-\infty}^{\infty} \frac{\sin x}{x} dx = \pi$$

---

### 5. Type V: Integrals with Branch Cuts

**Goal:** Handle integrals like ∫₀^∞ x^{a-1}/(1+x) dx for 0 < a < 1

**Method:** Use keyhole contour around the branch cut.

**Result:**
$$\int_0^{\infty} \frac{x^{a-1}}{1+x} dx = \frac{\pi}{\sin(\pi a)}$$

---

### 6. Summary Table of Contour Choices

| Integral Type | Contour | Key Condition |
|---------------|---------|---------------|
| ∫_{-∞}^{∞} p/q dx | Semicircle | deg(q) ≥ deg(p) + 2 |
| ∫₀^{2π} R(cos,sin) dθ | Unit circle | z = e^{iθ} substitution |
| ∫_{-∞}^{∞} f(x)e^{iax} dx | Semicircle (a>0: upper) | Jordan's lemma |
| ∫_{-∞}^{∞} sin(x)/x dx | Indented semicircle | Avoid pole on axis |
| ∫₀^{∞} x^{a-1}f(x) dx | Keyhole | Branch cut on [0,∞) |

---

### 7. 🔬 Quantum Mechanics Connection

**Fourier Transforms:**
$$\hat{\psi}(k) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \psi(x) e^{-ikx} dx$$

Often evaluated by residues when ψ(x) is rational.

**Green's Functions:**
The free particle propagator:
$$G_0(E) = \int_{-\infty}^{\infty} \frac{dk}{2\pi} \frac{e^{ikx}}{E - \hbar^2 k^2/(2m) + i\epsilon}$$

The poles are at k = ±√(2mE/ℏ²), and the contour + iε prescription determines causality.

**Kramers-Kronig Relations:**
$$\chi'(\omega) = \frac{1}{\pi}\mathcal{P}\int_{-\infty}^{\infty} \frac{\chi''(\omega')}{\omega' - \omega} d\omega'$$

Derived using contour integration and the analyticity of χ in the upper half-plane.

**Partition Function:**
$$Z = \int_0^{\infty} g(E) e^{-\beta E} dE$$

where g(E) is the density of states, often computed via residues.

---

## ✏️ Worked Examples

### Example 5: ∫_{-∞}^{∞} x²/(x⁴+1) dx

**Solution:**
Poles: x⁴ + 1 = 0 → x = e^{i(π+2πk)/4}

Upper half-plane poles: z₁ = e^{iπ/4}, z₂ = e^{i3π/4}

f(z) = z²/(z⁴+1), so Res[f, zₖ] = zₖ²/(4zₖ³) = 1/(4zₖ)

$$\text{Res}[f, z_1] = \frac{1}{4e^{i\pi/4}} = \frac{e^{-i\pi/4}}{4}$$
$$\text{Res}[f, z_2] = \frac{1}{4e^{i3\pi/4}} = \frac{e^{-i3\pi/4}}{4}$$

Sum: (1/4)[e^{-iπ/4} + e^{-i3π/4}] = (1/4)[(1-i)/√2 + (-1-i)/√2] = -i/(2√2)

$$\int_{-\infty}^{\infty} \frac{x^2}{x^4+1} dx = 2\pi i \cdot \left(-\frac{i}{2\sqrt{2}}\right) = \boxed{\frac{\pi}{\sqrt{2}}}$$

---

### Example 6: ∫₀^{2π} dθ/(5 - 4cos θ)

**Solution:**
z = e^{iθ}, cos θ = (z+1/z)/2

$$\frac{1}{5-4\cos\theta} = \frac{1}{5 - 2(z+1/z)} = \frac{z}{5z - 2z^2 - 2} = \frac{-z}{2z^2 - 5z + 2} = \frac{-z}{2(z-2)(z-1/2)}$$

$$\oint \frac{-z}{2(z-2)(z-1/2)} \cdot \frac{dz}{iz} = \frac{1}{2i} \oint \frac{dz}{(z-2)(z-1/2)}$$

Only z = 1/2 is inside |z| = 1.

$$\text{Res at } z=1/2: \frac{1}{1/2 - 2} = \frac{1}{-3/2} = -\frac{2}{3}$$

$$\int_0^{2\pi} \frac{d\theta}{5-4\cos\theta} = \frac{1}{2i} \cdot 2\pi i \cdot \left(-\frac{2}{3}\right) = -\frac{2\pi}{3}$$

Wait, this should be positive! Let me recheck...

Actually: 5 - 4cos θ ≥ 5 - 4 = 1 > 0, so integral is positive.

Going back: 5 - 2(z + 1/z) = 5 - 2z - 2/z = (5z - 2z² - 2)/z

So 1/(5-4cos θ) = z/(5z - 2z² - 2) = -z/(2z² - 5z + 2) = -z/(2(z-2)(z-1/2))

∫ = ∮ (-z)/(2(z-2)(z-1/2)) · dz/(iz) = (1/2i) ∮ 1/((z-2)(z-1/2)) dz · (-1) = (-1/2i) ∮ ...

Hmm, let me be more careful. We have:
$$\int_0^{2\pi} \frac{d\theta}{5-4\cos\theta} = \oint_{|z|=1} \frac{1}{5 - 2(z+z^{-1})} \cdot \frac{dz}{iz}$$

$$= \oint \frac{z}{5z - 2z^2 - 2} \cdot \frac{dz}{iz} = \oint \frac{1}{i(5z - 2z^2 - 2)} dz = \frac{-1}{2i} \oint \frac{dz}{z^2 - (5/2)z + 1}$$

Roots: z = (5/2 ± √(25/4 - 4))/2 = (5/2 ± 3/2)/2 → z = 2 or z = 1/2

Only z = 1/2 inside. Res = 1/(2·(1/2) - 5/2) = 1/(1 - 5/2) = -2/3

$$\int = \frac{-1}{2i} \cdot 2\pi i \cdot (-2/3) = \boxed{\frac{2\pi}{3}}$$

---

## 🔧 Practice Problems

### Level 1: Rational Functions
1. ∫_{-∞}^{∞} dx/(x²+4)
2. ∫_{-∞}^{∞} dx/(x²+1)²
3. ∫_{-∞}^{∞} x²dx/(x²+1)(x²+4)

### Level 2: Trigonometric
4. ∫₀^{2π} dθ/(3 + sin θ)
5. ∫₀^{2π} cos²θ/(5 - 4cos θ) dθ
6. ∫₀^{π} dθ/(1 + sin²θ)

### Level 3: Fourier-type
7. ∫_{-∞}^{∞} cos(2x)/(x²+1) dx
8. ∫_{-∞}^{∞} x sin x/(x²+4) dx
9. ∫₀^{∞} sin x/x dx (show = π/2)

### Level 4: Advanced
10. ∫₀^{∞} dx/(1+x³)
11. ∫₀^{∞} ln x/(1+x²) dx
12. ∫₀^{∞} x^{-1/2}/(1+x) dx

---

## 💻 Computational Lab

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def verify_residue_integral(f_real, f_complex, poles_upper, expected, name):
    """Compare numerical integration with residue theorem result."""
    # Numerical integration
    result, error = integrate.quad(f_real, -100, 100)
    
    # Residue calculation
    residue_sum = sum(poles_upper.values())
    theoretical = 2 * np.pi * 1j * residue_sum
    
    print(f"\n{name}")
    print(f"  Numerical: {result:.6f} ± {error:.2e}")
    print(f"  Theoretical: {np.real(theoretical):.6f}")
    print(f"  Expected: {expected:.6f}")

# Example 1: 1/(1+x^2)
print("=" * 60)
print("VERIFICATION OF RESIDUE INTEGRALS")
print("=" * 60)

verify_residue_integral(
    f_real=lambda x: 1/(1+x**2),
    f_complex=lambda z: 1/(1+z**2),
    poles_upper={1j: 1/(2j)},
    expected=np.pi,
    name="∫ dx/(1+x²)"
)

# Example 2: 1/(1+x^4)
verify_residue_integral(
    f_real=lambda x: 1/(1+x**4),
    f_complex=lambda z: 1/(1+z**4),
    poles_upper={
        np.exp(1j*np.pi/4): -np.exp(1j*np.pi/4)/4,
        np.exp(1j*3*np.pi/4): -np.exp(1j*3*np.pi/4)/4
    },
    expected=np.pi/np.sqrt(2),
    name="∫ dx/(1+x⁴)"
)

# Example 3: cos(x)/(1+x^2)
verify_residue_integral(
    f_real=lambda x: np.cos(x)/(1+x**2),
    f_complex=lambda z: np.exp(1j*z)/(1+z**2),
    poles_upper={1j: np.exp(-1)/(2j)},
    expected=np.pi/np.e,
    name="∫ cos(x)/(1+x²) dx"
)

# Visualize contours
def visualize_contour_integration():
    """Visualize the contour integration process."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Example 1: Semicircular contour for 1/(1+z^2)
    ax = axes[0, 0]
    
    # Plot poles
    ax.scatter([0, 0], [1, -1], c='red', s=200, marker='x', 
              linewidths=3, label='Poles at ±i')
    
    # Draw contour
    R = 3
    x_line = np.linspace(-R, R, 100)
    ax.plot(x_line, np.zeros_like(x_line), 'b-', lw=2, label='Real axis')
    
    theta = np.linspace(0, np.pi, 100)
    ax.plot(R*np.cos(theta), R*np.sin(theta), 'g-', lw=2, label='Semicircle')
    
    # Add arrows
    ax.annotate('', xy=(0.5, 0), xytext=(-0.5, 0),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=(R*np.cos(np.pi/2+0.1), R*np.sin(np.pi/2+0.1)), 
               xytext=(R*np.cos(np.pi/2-0.1), R*np.sin(np.pi/2-0.1)),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2, 4)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Contour for ∫ dx/(1+x²)\nOnly upper pole contributes')
    ax.grid(True, alpha=0.3)
    
    # Example 2: Unit circle for trig integral
    ax = axes[0, 1]
    
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'b-', lw=2)
    ax.scatter([0.5, 2], [0, 0], c='red', s=200, marker='x', linewidths=3)
    ax.annotate('z = 1/2\n(inside)', (0.5, 0.2), fontsize=10)
    ax.annotate('z = 2\n(outside)', (2, 0.2), fontsize=10)
    
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Contour for ∫ dθ/(5-4cos θ)\nOnly z=1/2 contributes')
    ax.grid(True, alpha=0.3)
    
    # Example 3: Jordan's lemma visualization
    ax = axes[1, 0]
    
    # Show e^{iz} decays in upper half-plane
    x = np.linspace(-3, 3, 100)
    y = np.linspace(0, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    decay = np.abs(np.exp(1j * Z))  # = e^{-y}
    ax.contourf(X, Y, decay, levels=20, cmap='Blues_r')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title('|e^{iz}| = e^{-Im(z)}\nDecays in upper half-plane')
    plt.colorbar(ax.contourf(X, Y, decay, levels=20, cmap='Blues_r'), ax=ax)
    
    # Example 4: Show convergence of integral
    ax = axes[1, 1]
    
    Rs = np.linspace(1, 20, 50)
    integrals = []
    
    for R in Rs:
        result, _ = integrate.quad(lambda x: 1/(1+x**2), -R, R)
        integrals.append(result)
    
    ax.plot(Rs, integrals, 'b-', lw=2, label='∫_{-R}^{R} dx/(1+x²)')
    ax.axhline(y=np.pi, color='r', linestyle='--', label=f'π ≈ {np.pi:.4f}')
    ax.set_xlabel('R')
    ax.set_ylabel('Integral value')
    ax.set_title('Convergence of improper integral')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_integrals_contours.png', dpi=150)
    plt.show()

visualize_contour_integration()

# Trigonometric integral example
def trig_integral_demo():
    """Demonstrate trigonometric integral via contour."""
    print("\n" + "=" * 60)
    print("TRIGONOMETRIC INTEGRAL DEMO")
    print("=" * 60)
    
    # ∫_0^{2π} dθ/(2 + cos θ)
    print("\n∫₀^{2π} dθ/(2 + cos θ)")
    
    # Direct numerical
    result, _ = integrate.quad(lambda t: 1/(2 + np.cos(t)), 0, 2*np.pi)
    print(f"Numerical: {result:.6f}")
    
    # Theoretical (from residue)
    # Poles at z = -2 ± √3, only -2+√3 inside |z|=1
    z_pole = -2 + np.sqrt(3)
    residue = 1 / (2 * z_pole + 4)  # derivative of z^2 + 4z + 1
    theoretical = (2/1j) * 2*np.pi*1j * residue  # = 4π × residue
    print(f"Theoretical: {np.real(theoretical):.6f}")
    print(f"Expected 2π/√3 = {2*np.pi/np.sqrt(3):.6f}")

trig_integral_demo()

# Fourier transform example
def fourier_demo():
    """Demonstrate Fourier-type integral."""
    print("\n" + "=" * 60)
    print("FOURIER-TYPE INTEGRAL")
    print("=" * 60)
    
    # ∫ e^{iax}/(x^2+1) dx for various a > 0
    print("\n∫ e^{iax}/(x²+1) dx = π·e^{-a} for a > 0")
    
    a_values = [0.5, 1, 2, 3]
    
    for a in a_values:
        # Numerical (integrate real and imaginary parts)
        real_part, _ = integrate.quad(lambda x: np.cos(a*x)/(1+x**2), -100, 100)
        imag_part, _ = integrate.quad(lambda x: np.sin(a*x)/(1+x**2), -100, 100)
        numerical = real_part + 1j * imag_part
        
        # Theoretical
        theoretical = np.pi * np.exp(-a)
        
        print(f"a = {a}: Numerical = {numerical:.4f}, Theoretical = {theoretical:.4f}")

fourier_demo()
```

---

## 📝 Summary

### Integration Methods by Integral Type

| Type | Example | Contour | Result |
|------|---------|---------|--------|
| Rational | ∫ p(x)/q(x) dx | Semicircle | 2πi × (upper residues) |
| Trig [0,2π] | ∫ R(cos,sin) dθ | Unit circle | 2πi × residues |
| Fourier | ∫ f(x)e^{iax} dx | Semicircle (Jordan) | 2πi × residues |
| sin x/x | ∫ sin x/x dx | Indented | π |

### Key Conditions
- Rational: deg(q) ≥ deg(p) + 2 for semicircle to vanish
- Fourier (a > 0): Use upper half-plane
- Fourier (a < 0): Use lower half-plane

---

## ✅ Daily Checklist

- [ ] Evaluate improper integrals of rational functions
- [ ] Handle trigonometric integrals via unit circle
- [ ] Apply Jordan's lemma for Fourier-type integrals
- [ ] Understand indented contours for sin x/x
- [ ] Connect to physics applications
- [ ] Complete numerical verifications

---

## 🔮 Preview: Day 138

Tomorrow we study the **Argument Principle and Rouché's Theorem** — powerful tools for counting zeros and poles of analytic functions!
