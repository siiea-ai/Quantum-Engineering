# Day 136: The Residue Theorem

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Residue Theorem & Proof |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State and prove the Residue Theorem
2. Apply the theorem to evaluate contour integrals
3. Choose appropriate contours for different problems
4. Handle multiple singularities
5. Understand the connection to Cauchy's integral formula
6. Apply to physics problems

---

## 📚 Required Reading

### Primary Text: Churchill & Brown
- **Chapter 6, Sections 69-75**: Residues and Poles

### Alternative: Ahlfors
- **Chapter 4.5**: The Residue Theorem

### Physics Connection
- **Arfken & Weber, Section 11.7**: The Residue Theorem

---

## 📖 Core Content: Theory and Concepts

### 1. The Residue Theorem: Statement

**Theorem (Residue Theorem):**
Let f be analytic inside and on a simple closed contour C, except for isolated singularities z₁, z₂, ..., zₙ inside C. Then:

$$\boxed{\oint_C f(z)\,dz = 2\pi i \sum_{k=1}^{n} \text{Res}[f, z_k]}$$

where the integral is taken counterclockwise.

**In words:** The contour integral equals 2πi times the sum of all residues inside the contour.

---

### 2. Proof of the Residue Theorem

**Setup:** Let C be a simple closed contour containing singularities z₁, ..., zₙ.

**Step 1:** Draw small circles Cₖ around each singularity zₖ, all inside C.

**Step 2:** By Cauchy's theorem applied to the multiply-connected region:
$$\oint_C f\,dz = \sum_{k=1}^{n} \oint_{C_k} f\,dz$$
(The integrals around the "holes" add up to the integral around the boundary)

**Step 3:** For each small circle Cₖ around zₖ, use the Laurent series:
$$f(z) = \sum_{n=-\infty}^{\infty} a_n^{(k)} (z-z_k)^n$$

**Step 4:** Integrate term by term:
$$\oint_{C_k} f\,dz = \sum_{n=-\infty}^{\infty} a_n^{(k)} \oint_{C_k} (z-z_k)^n\,dz$$

Only the n = -1 term contributes (giving 2πi), all others give 0:
$$\oint_{C_k} f\,dz = 2\pi i \cdot a_{-1}^{(k)} = 2\pi i \cdot \text{Res}[f, z_k]$$

**Step 5:** Sum over all singularities:
$$\oint_C f\,dz = \sum_{k=1}^{n} 2\pi i \cdot \text{Res}[f, z_k] = 2\pi i \sum_{k=1}^{n} \text{Res}[f, z_k]$$ □

---

### 3. Connection to Earlier Results

**Cauchy's Integral Formula is a Special Case!**

For f analytic everywhere inside C and z₀ inside C:
$$g(z) = \frac{f(z)}{z-z_0}$$
has a simple pole at z₀ with Res[g, z₀] = f(z₀).

By the Residue Theorem:
$$\oint_C \frac{f(z)}{z-z_0}\,dz = 2\pi i \cdot f(z_0)$$

This is exactly Cauchy's integral formula!

---

### 4. Strategy for Applying the Residue Theorem

**Step-by-step approach:**

1. **Identify all singularities** of f(z)
2. **Determine which are inside** the contour C
3. **Classify each interior singularity** (pole order, essential, etc.)
4. **Compute the residue** at each interior singularity
5. **Sum and multiply by 2πi**

**Common mistakes to avoid:**
- Forgetting a singularity
- Including singularities ON the contour (not allowed!)
- Wrong sign (counterclockwise = positive)
- Including singularities outside the contour

---

### 5. Examples: Direct Application

**Example 1:** Evaluate ∮_{|z|=2} z/(z²-1) dz

**Solution:**
Singularities: z = ±1 (both inside |z| = 2)

At z = 1 (simple pole):
$$\text{Res}[f, 1] = \lim_{z \to 1} (z-1) \cdot \frac{z}{(z-1)(z+1)} = \frac{1}{2}$$

At z = -1 (simple pole):
$$\text{Res}[f, -1] = \lim_{z \to -1} (z+1) \cdot \frac{z}{(z-1)(z+1)} = \frac{-1}{-2} = \frac{1}{2}$$

By Residue Theorem:
$$\oint_{|z|=2} \frac{z}{z^2-1}\,dz = 2\pi i \left(\frac{1}{2} + \frac{1}{2}\right) = \boxed{2\pi i}$$

---

**Example 2:** Evaluate ∮_{|z|=1} eᶻ/(z²+4) dz

**Solution:**
Singularities: z = ±2i (both OUTSIDE |z| = 1)

No singularities inside the contour → By Cauchy's theorem:
$$\oint_{|z|=1} \frac{e^z}{z^2+4}\,dz = \boxed{0}$$

---

**Example 3:** Evaluate ∮_{|z|=3} (z+1)/((z-1)(z-2)²) dz

**Solution:**
Singularities: z = 1 (simple pole), z = 2 (pole of order 2). Both inside |z| = 3.

At z = 1:
$$\text{Res}[f, 1] = \lim_{z \to 1} \frac{z+1}{(z-2)^2} = \frac{2}{1} = 2$$

At z = 2 (order 2):
$$\text{Res}[f, 2] = \lim_{z \to 2} \frac{d}{dz}\left[\frac{z+1}{z-1}\right] = \lim_{z \to 2} \frac{(z-1) - (z+1)}{(z-1)^2} = \frac{-2}{1} = -2$$

By Residue Theorem:
$$\oint_{|z|=3} f\,dz = 2\pi i(2 + (-2)) = \boxed{0}$$

---

**Example 4:** Evaluate ∮_{|z|=1} eᶻ/z³ dz

**Solution:**
Only singularity: z = 0 (pole of order 3) inside |z| = 1.

Find residue from Laurent series:
$$\frac{e^z}{z^3} = \frac{1}{z^3}\left(1 + z + \frac{z^2}{2} + \frac{z^3}{6} + ...\right) = \frac{1}{z^3} + \frac{1}{z^2} + \frac{1}{2z} + \frac{1}{6} + ...$$

Residue (coefficient of 1/z): a₋₁ = 1/2

$$\oint_{|z|=1} \frac{e^z}{z^3}\,dz = 2\pi i \cdot \frac{1}{2} = \boxed{\pi i}$$

---

### 6. The Extended Residue Theorem

**For clockwise contours:** Add a negative sign:
$$\oint_C f\,dz = -2\pi i \sum_{\text{inside}} \text{Res}[f, z_k]$$

**For multiple contours:** If C consists of outer contour C₀ (counterclockwise) and inner contours C₁, ..., Cₘ (clockwise):
$$\oint_{C_0} f\,dz - \sum_{j=1}^{m}\oint_{C_j} f\,dz = 2\pi i \sum_{\text{in region}} \text{Res}[f, z_k]$$

---

### 7. Residue at Infinity

**Definition:** If f is analytic for |z| > R, define:
$$\text{Res}[f, \infty] = -\frac{1}{2\pi i}\oint_{|z|=R'} f(z)\,dz$$
for any R' > R (integral counterclockwise).

**Alternative formula:**
$$\text{Res}[f, \infty] = -\text{Res}\left[\frac{1}{z^2}f\left(\frac{1}{z}\right), 0\right]$$

**Extended Residue Theorem:**
If f is meromorphic on the extended complex plane (ℂ ∪ {∞}):
$$\sum_{\text{all } z_k} \text{Res}[f, z_k] + \text{Res}[f, \infty] = 0$$

The sum of ALL residues (including at ∞) is zero!

---

### 8. 🔬 Quantum Mechanics Connection

**Partition Function via Residues:**
$$Z = \text{Tr}[e^{-\beta H}] = \sum_n e^{-\beta E_n}$$

Using the resolvent G(E) = (E - H)⁻¹:
$$Z = -\frac{1}{2\pi i}\oint_C e^{-\beta E} \text{Tr}[G(E)]\,dE$$

where C encloses all eigenvalues. The residue at each Eₙ is e^{-βEₙ}!

**Spectral Density:**
$$\rho(E) = -\frac{1}{\pi}\text{Im}\,\text{Tr}[G(E+i0^+)] = \sum_n \delta(E - E_n)$$

**Propagators in QFT:**
The Feynman propagator involves contour integrals:
$$G_F(x-y) = \int \frac{d^4p}{(2\pi)^4} \frac{e^{-ip\cdot(x-y)}}{p^2 - m^2 + i\epsilon}$$

The iε prescription determines which poles contribute (Feynman boundary conditions).

**Matsubara Sums:**
In finite-temperature QFT, sums over Matsubara frequencies become contour integrals:
$$\frac{1}{\beta}\sum_n f(i\omega_n) = \frac{1}{2\pi i}\oint f(z) n_B(z)\,dz$$
where n_B is the Bose-Einstein distribution.

---

## ✏️ Worked Examples

### Example 5: Multiple poles of same function

Evaluate ∮_{|z|=4} dz/(z⁴-1)

**Solution:**
z⁴ - 1 = (z-1)(z+1)(z-i)(z+i)

All four roots (±1, ±i) are inside |z| = 4.

For each simple pole zₖ:
$$\text{Res}[1/(z^4-1), z_k] = \frac{1}{4z_k^3}$$

- Res at z = 1: 1/4
- Res at z = -1: 1/(-4) = -1/4
- Res at z = i: 1/(4i³) = 1/(-4i) = i/4
- Res at z = -i: 1/(4(-i)³) = 1/(4i) = -i/4

Sum: 1/4 - 1/4 + i/4 - i/4 = 0

$$\oint_{|z|=4} \frac{dz}{z^4-1} = 2\pi i \cdot 0 = \boxed{0}$$

---

### Example 6: Essential singularity

Evaluate ∮_{|z|=1} e^{1/z} dz

**Solution:**
Essential singularity at z = 0.

Laurent series: e^{1/z} = 1 + 1/z + 1/(2!z²) + 1/(3!z³) + ...

Residue (coefficient of 1/z): a₋₁ = 1

$$\oint_{|z|=1} e^{1/z}\,dz = 2\pi i \cdot 1 = \boxed{2\pi i}$$

---

### Example 7: Choosing which singularities to include

Evaluate ∮_C dz/((z-1)(z-3)) where C is the circle |z-2| = 1.

**Solution:**
Singularities: z = 1 and z = 3

The circle |z-2| = 1 has center 2 and radius 1.
- z = 1: |1-2| = 1 → ON the contour (not allowed!)
- z = 3: |3-2| = 1 → ON the contour (not allowed!)

**This integral is not well-defined** as stated because singularities are on the contour.

If we use |z-2| = 0.9 instead:
- z = 1: outside
- z = 3: outside

Result: 0 (no singularities inside)

If we use |z-2| = 1.5:
- z = 1: inside (|1-2| = 1 < 1.5)
- z = 3: inside (|3-2| = 1 < 1.5)

Res at z = 1: 1/(1-3) = -1/2
Res at z = 3: 1/(3-1) = 1/2

Result: 2πi(-1/2 + 1/2) = 0

---

## 🔧 Practice Problems

### Level 1: Direct Application
1. ∮_{|z|=2} dz/(z-1)
2. ∮_{|z|=1} z²/(z-2) dz
3. ∮_{|z|=3} dz/(z²+1)

### Level 2: Multiple Singularities
4. ∮_{|z|=2} (3z+2)/(z(z-1)) dz
5. ∮_{|z|=4} z/((z-1)(z-2)(z-3)) dz
6. ∮_{|z|=1} dz/(z²(z-2))

### Level 3: Higher Order Poles
7. ∮_{|z|=2} eᶻ/(z-1)² dz
8. ∮_{|z|=1} sin z/z⁴ dz
9. ∮_{|z|=2} z²/((z-1)³) dz

### Level 4: Applications
10. Use residues to compute ∮_{|z|=2} tan z dz.
11. Evaluate ∮_C dz/(z⁶+1) where C is |z| = 2.
12. Show that ∮_{|z|=R} p(z)/q(z) dz → 0 as R → ∞ if deg(q) ≥ deg(p) + 2.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

class ResidueCalculator:
    """Tools for residue theorem calculations."""
    
    @staticmethod
    def find_poles(denominator_roots):
        """Given roots of denominator, return pole locations."""
        return np.array(denominator_roots)
    
    @staticmethod
    def residue_simple_pole(f, z0, eps=1e-8):
        """Compute residue at simple pole using limit."""
        return (z0 + eps - z0) * f(z0 + eps) / eps * eps
    
    @staticmethod
    def residue_quotient(g, h, h_prime, z0):
        """For f = g/h where h has simple zero at z0."""
        return g(z0) / h_prime(z0)
    
    @staticmethod
    def poles_inside_circle(poles, center, radius):
        """Find which poles are inside a circular contour."""
        inside = []
        for p in poles:
            if np.abs(p - center) < radius:
                inside.append(p)
        return np.array(inside)
    
    @staticmethod
    def contour_integral_numerical(f, center, radius, n_points=10000):
        """Numerically compute contour integral."""
        t = np.linspace(0, 2*np.pi, n_points)
        z = center + radius * np.exp(1j * t)
        dz = 1j * radius * np.exp(1j * t)
        integrand = f(z) * dz
        return np.trapz(integrand, t)


def demonstrate_residue_theorem():
    """Demonstrate the residue theorem with examples."""
    print("=" * 60)
    print("RESIDUE THEOREM DEMONSTRATIONS")
    print("=" * 60)
    
    calc = ResidueCalculator()
    
    # Example 1: f(z) = 1/(z² - 1)
    print("\n1. ∮_{|z|=2} dz/(z² - 1)")
    
    f1 = lambda z: 1/(z**2 - 1)
    poles = np.array([1, -1])
    center, radius = 0, 2
    
    inside = calc.poles_inside_circle(poles, center, radius)
    print(f"   Poles: {poles}")
    print(f"   Inside |z| = 2: {inside}")
    
    # Residues by quotient formula
    g = lambda z: 1
    h_prime = lambda z: 2*z
    residues = [g(p)/h_prime(p) for p in inside]
    print(f"   Residues: {residues}")
    print(f"   Sum of residues: {sum(residues)}")
    
    # Numerical verification
    numerical = calc.contour_integral_numerical(f1, center, radius)
    theoretical = 2 * np.pi * 1j * sum(residues)
    print(f"   Numerical: {numerical:.6f}")
    print(f"   Theoretical: {theoretical:.6f}")
    
    # Example 2: f(z) = z/((z-1)(z-2))
    print("\n2. ∮_{|z|=1.5} z/((z-1)(z-2)) dz")
    
    f2 = lambda z: z/((z-1)*(z-2))
    poles = np.array([1, 2])
    center, radius = 0, 1.5
    
    inside = calc.poles_inside_circle(poles, center, radius)
    print(f"   Poles: {poles}")
    print(f"   Inside |z| = 1.5: {inside}")
    
    # Only z = 1 is inside
    res_1 = 1 / (1 - 2)  # Using partial fractions: z/((z-1)(z-2)) near z=1
    res_1 = np.lim = 1 * 1 / (1 - 2)  # Actually: lim_{z→1} (z-1) * z/((z-1)(z-2)) = 1/(1-2) = -1
    
    # Correct calculation
    res_at_1 = 1 / (1 - 2)  # = -1
    print(f"   Residue at z=1: {res_at_1}")
    
    numerical = calc.contour_integral_numerical(f2, center, radius)
    theoretical = 2 * np.pi * 1j * res_at_1
    print(f"   Numerical: {numerical:.6f}")
    print(f"   Theoretical: {theoretical:.6f}")
    
    # Example 3: e^z / z^3 (higher order pole)
    print("\n3. ∮_{|z|=1} e^z/z³ dz")
    
    f3 = lambda z: np.exp(z) / z**3
    
    # Residue from Laurent series: e^z/z³ = 1/z³ + 1/z² + 1/(2z) + ...
    # Coefficient of 1/z is 1/2
    res = 0.5
    
    numerical = calc.contour_integral_numerical(f3, 0, 1)
    theoretical = 2 * np.pi * 1j * res
    print(f"   Residue at z=0: {res}")
    print(f"   Numerical: {numerical:.6f}")
    print(f"   Theoretical: {theoretical:.6f}")

demonstrate_residue_theorem()


def visualize_residue_theorem():
    """Visualize the residue theorem geometrically."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Example function: f(z) = 1/((z-0.5)(z+0.5-0.5i))
    poles = [0.5, -0.5 + 0.5j]
    
    # Plot domain coloring
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    f = lambda z: 1/((z - poles[0]) * (z - poles[1]))
    
    with np.errstate(all='ignore'):
        W = f(Z)
        W[~np.isfinite(W)] = np.nan
    
    # Phase plot
    phase = np.angle(W)
    axes[0].contourf(X, Y, phase, levels=30, cmap='hsv')
    
    # Mark poles
    for p in poles:
        axes[0].scatter([np.real(p)], [np.imag(p)], c='white', s=200, 
                       marker='x', linewidths=3)
    
    # Draw contours
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Large contour (contains both poles)
    r1 = 1.5
    axes[0].plot(r1*np.cos(theta), r1*np.sin(theta), 'g-', lw=3, 
                label=f'|z|={r1} (both inside)')
    
    # Small contour (contains only first pole)
    r2 = 0.3
    c2 = 0.5
    axes[0].plot(c2 + r2*np.cos(theta), r2*np.sin(theta), 'r-', lw=3,
                label=f'|z-0.5|={r2} (one inside)')
    
    axes[0].set_xlabel('Re(z)')
    axes[0].set_ylabel('Im(z)')
    axes[0].set_title('Contours and Poles')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Compute integrals for different contours
    radii = np.linspace(0.1, 2.0, 50)
    integrals_real = []
    integrals_imag = []
    
    for r in radii:
        integral = ResidueCalculator.contour_integral_numerical(f, 0, r)
        integrals_real.append(np.real(integral))
        integrals_imag.append(np.imag(integral))
    
    axes[1].plot(radii, integrals_real, 'b-', lw=2, label='Re(∮f dz)')
    axes[1].plot(radii, integrals_imag, 'r-', lw=2, label='Im(∮f dz)')
    
    # Mark where poles are crossed
    for p in poles:
        axes[1].axvline(x=np.abs(p), color='green', linestyle='--', alpha=0.5)
    
    # Theoretical jumps
    # When r crosses |pole|, integral jumps by 2πi × residue
    axes[1].axhline(y=0, color='k', lw=0.5)
    
    axes[1].set_xlabel('Contour radius')
    axes[1].set_ylabel('Integral value')
    axes[1].set_title('Integral vs Contour Size\n(jumps when poles enter)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('residue_theorem_visual.png', dpi=150)
    plt.show()

visualize_residue_theorem()


def sum_of_residues_zero():
    """Demonstrate that sum of all residues (including ∞) is zero."""
    print("\n" + "=" * 60)
    print("SUM OF ALL RESIDUES = 0")
    print("=" * 60)
    
    # For rational function p(z)/q(z)
    # Example: f(z) = (z+1)/((z-1)(z-2)(z-3))
    
    print("\nf(z) = (z+1)/((z-1)(z-2)(z-3))")
    
    # Residues at finite poles
    def res_at_pole(a, other_poles):
        """Residue of (z+1)/prod(z-p) at z=a."""
        numerator = a + 1
        denominator = np.prod([a - p for p in other_poles])
        return numerator / denominator
    
    poles = [1, 2, 3]
    residues = {}
    for p in poles:
        others = [x for x in poles if x != p]
        residues[p] = res_at_pole(p, others)
        print(f"  Res at z={p}: {residues[p]:.4f}")
    
    sum_finite = sum(residues.values())
    print(f"\n  Sum of finite residues: {sum_finite:.4f}")
    
    # Residue at infinity
    # For f(z) = (z+1)/((z-1)(z-2)(z-3))
    # As z → ∞, f(z) ~ 1/z² → 0
    # Res[f, ∞] = -Res[f(1/w)/w², 0]
    
    # f(1/w) = (1/w + 1)/((1/w - 1)(1/w - 2)(1/w - 3))
    #        = (1 + w)/(w) / ((1-w)(1-2w)(1-3w)/w³)
    #        = w²(1+w)/((1-w)(1-2w)(1-3w))
    
    # f(1/w)/w² = (1+w)/((1-w)(1-2w)(1-3w))
    # This is analytic at w=0! So Res = value at w=0 = 1/((1)(1)(1)) = 1
    # Res[f, ∞] = -1
    
    res_infinity = -1  # For this example
    print(f"  Res at z=∞: {res_infinity:.4f}")
    
    total = sum_finite + res_infinity
    print(f"\n  Total (should be 0): {total:.4f}")

sum_of_residues_zero()
```

---

## 📝 Summary

### The Residue Theorem

$$\oint_C f(z)\,dz = 2\pi i \sum_{\text{inside } C} \text{Res}[f, z_k]$$

### Application Strategy

1. Find all singularities
2. Identify which are inside the contour
3. Compute residues at interior singularities
4. Sum and multiply by 2πi

### Key Points

- Only singularities INSIDE the contour contribute
- Singularities ON the contour are not allowed
- Counterclockwise orientation gives positive sign
- Sum of ALL residues (including ∞) equals zero for meromorphic functions

---

## ✅ Daily Checklist

- [ ] State and understand the Residue Theorem
- [ ] Apply to integrals with multiple poles
- [ ] Handle higher-order poles
- [ ] Understand the connection to Cauchy's formula
- [ ] Work with residue at infinity
- [ ] Connect to physics applications
- [ ] Complete computational exercises

---

## 🔮 Preview: Day 137

Tomorrow we apply the Residue Theorem to **evaluate real integrals** — one of the most powerful applications of complex analysis! We'll develop systematic techniques for integrals of rational functions, trigonometric integrals, and more.
