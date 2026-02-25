# Day 138: Argument Principle & Rouché's Theorem

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Argument Principle & Rouché |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State and prove the Argument Principle
2. Count zeros and poles using contour integrals
3. Apply Rouché's theorem to locate zeros
4. Understand the winding number concept
5. Apply these theorems to prove the Fundamental Theorem of Algebra
6. Connect to physics applications (Nyquist stability criterion)

---

## 📚 Required Reading

### Primary Text: Churchill & Brown
- **Chapter 7, Sections 87-91**: Argument Principle and Rouché's Theorem

### Alternative: Ahlfors
- **Chapter 5.2**: The Argument Principle

### Physics Connection
- Control theory: Nyquist stability criterion

---

## 📖 Core Content: Theory and Concepts

### 1. The Argument Principle

**Theorem (Argument Principle):**
Let f be meromorphic inside and on a simple closed contour C (no zeros or poles on C). Let Z = number of zeros inside C (counting multiplicity) and P = number of poles inside C (counting multiplicity). Then:

$$\boxed{\frac{1}{2\pi i}\oint_C \frac{f'(z)}{f(z)} dz = Z - P}$$

**Equivalent form (change in argument):**
$$\frac{1}{2\pi}\Delta_C \arg f(z) = Z - P$$

The total change in arg f(z) as z traverses C equals 2π(Z - P).

---

### 2. Proof of the Argument Principle

**Key observation:** f'/f = (log f)' = d/dz[log f]

Near a zero z₀ of order m: f(z) = (z-z₀)^m g(z) where g(z₀) ≠ 0

$$\frac{f'(z)}{f(z)} = \frac{m(z-z_0)^{m-1}g(z) + (z-z_0)^m g'(z)}{(z-z_0)^m g(z)} = \frac{m}{z-z_0} + \frac{g'(z)}{g(z)}$$

So f'/f has a simple pole at z₀ with **residue m**.

Similarly, near a pole z₁ of order n: f'/f has **residue -n**.

By the Residue Theorem:
$$\frac{1}{2\pi i}\oint_C \frac{f'}{f} dz = \sum_{\text{zeros}} (\text{order}) - \sum_{\text{poles}} (\text{order}) = Z - P$$ □

---

### 3. The Winding Number

**Definition:** The **winding number** of a curve Γ around a point w₀ is:
$$n(\Gamma, w_0) = \frac{1}{2\pi i}\oint_\Gamma \frac{dw}{w - w_0}$$

This counts how many times Γ winds around w₀ (counterclockwise = positive).

**Connection to Argument Principle:**
If w = f(z) and Γ = f(C) is the image of C under f:
$$n(\Gamma, 0) = \frac{1}{2\pi i}\oint_\Gamma \frac{dw}{w} = \frac{1}{2\pi i}\oint_C \frac{f'(z)}{f(z)} dz = Z - P$$

The number of times f(C) winds around the origin equals Z - P!

---

### 4. Rouché's Theorem

**Theorem (Rouché):**
If f and g are analytic inside and on a simple closed contour C, and:
$$|g(z)| < |f(z)| \text{ for all } z \text{ on } C$$

Then f and f + g have the same number of zeros inside C.

**Intuition:** If g is "small" compared to f on the boundary, adding g doesn't change the zero count.

**Proof sketch:**
Consider h(t) = f + tg for t ∈ [0,1].
- At t = 0: h(0) = f
- At t = 1: h(1) = f + g

Since |g| < |f| on C, h(t) ≠ 0 on C for all t.

The number of zeros N(t) = (1/2πi) ∮ h'(t)/h(t) dz is continuous in t.
Since N(t) is integer-valued and continuous, it must be constant.
Therefore N(0) = N(1), i.e., f and f + g have the same number of zeros. □

---

### 5. Applications of Rouché's Theorem

**Example 1: Fundamental Theorem of Algebra**

**Claim:** Every polynomial p(z) = zⁿ + aₙ₋₁zⁿ⁻¹ + ... + a₀ has exactly n roots in ℂ.

**Proof:**
Take f(z) = zⁿ and g(z) = aₙ₋₁zⁿ⁻¹ + ... + a₀.

On |z| = R for large R:
$$|f(z)| = R^n$$
$$|g(z)| \leq |a_{n-1}|R^{n-1} + ... + |a_0| < R^n \text{ for } R \text{ large enough}$$

By Rouché: p(z) = f(z) + g(z) has the same number of zeros as f(z) = zⁿ inside |z| = R.

Since zⁿ has n zeros (all at z = 0), p(z) has n zeros inside |z| = R.

Taking R → ∞ covers all of ℂ. □

---

**Example 2: Zeros of z⁵ + 3z + 1**

**How many zeros are in |z| < 1?**

Try f(z) = 3z and g(z) = z⁵ + 1.

On |z| = 1:
- |f(z)| = 3
- |g(z)| ≤ |z|⁵ + 1 = 2 < 3

By Rouché: z⁵ + 3z + 1 has the same number of zeros as 3z in |z| < 1.

3z has exactly **1 zero** (at z = 0) in |z| < 1.

Therefore z⁵ + 3z + 1 has **1 zero** in |z| < 1.

---

**Example 3: Zeros of eᶻ - 4z**

**How many zeros are in |z| < 1?**

Try f(z) = -4z and g(z) = eᶻ.

On |z| = 1:
- |f(z)| = 4
- |g(z)| = |eᶻ| = e^{Re(z)} ≤ e < 3 < 4

By Rouché: eᶻ - 4z has the same number of zeros as -4z in |z| < 1.

-4z has **1 zero** in |z| < 1.

---

### 6. The Open Mapping Theorem

**Theorem:** If f is analytic and non-constant on a domain D, then f maps open sets to open sets.

**Consequence:** Analytic functions are "locally surjective" — near any point, they hit all nearby values.

**Application:** Maximum Modulus Principle
If f is analytic and non-constant on a bounded domain D, then |f| achieves its maximum on the boundary ∂D, never in the interior.

---

### 7. 🔬 Quantum Mechanics & Physics Connections

**Nyquist Stability Criterion:**
In control theory, a system is stable if its transfer function G(s) has no poles in the right half-plane (Re(s) > 0).

The Nyquist criterion uses the argument principle:
- Plot G(s) as s traces a large semicircle in the right half-plane
- Count encirclements of the origin → gives number of unstable poles

**Levinson's Theorem (Scattering Theory):**
The phase shift δ(k) at zero energy is related to the number of bound states:
$$\delta(0) = n_b \pi$$
where n_b = number of bound states.

This is essentially the argument principle applied to the Jost function!

**Index Theorems:**
In quantum field theory, the Atiyah-Singer index theorem generalizes the argument principle:
$$\text{Index}(D) = \text{dim ker}(D) - \text{dim ker}(D^\dagger)$$

The difference of zero counts is computable from topological data.

**Topological Phases:**
Winding numbers characterize topological insulators:
$$\nu = \frac{1}{2\pi i}\oint \text{Tr}\left[H^{-1} dH\right]$$

This topological invariant counts "zeros" of the Hamiltonian in a generalized sense.

---

## ✏️ Worked Examples

### Example 4: Count zeros of p(z) = z⁴ - 5z + 1 in |z| < 1

**Solution:**
Try f(z) = -5z, g(z) = z⁴ + 1.

On |z| = 1:
- |f(z)| = 5
- |g(z)| ≤ 1 + 1 = 2 < 5

By Rouché: p(z) has same zeros as -5z in |z| < 1, which is **1 zero**.

---

### Example 5: Show z⁷ - 5z³ + 12 = 0 has all roots in |z| < 2

**Solution:**
Let f(z) = 12, g(z) = z⁷ - 5z³.

On |z| = 2:
- |f(z)| = 12
- |g(z)| ≤ 2⁷ + 5·2³ = 128 + 40 = 168 > 12 ✗

This doesn't work. Try different split.

Let f(z) = -5z³, g(z) = z⁷ + 12... still need to check.

Actually, let's use f(z) = z⁷, g(z) = -5z³ + 12.

On |z| = 2:
- |f(z)| = 128
- |g(z)| ≤ 5·8 + 12 = 52 < 128 ✓

By Rouché: p(z) has 7 zeros in |z| < 2 (same as z⁷).

All 7 roots are in |z| < 2! ✓

---

### Example 6: Use the Argument Principle directly

Find Z - P for f(z) = (z² + 1)/(z - 1) inside |z| = 2.

**Solution:**
- Zeros: z² + 1 = 0 → z = ±i (both inside |z| = 2), each simple → Z = 2
- Poles: z = 1 (simple) → P = 1

By Argument Principle:
$$\frac{1}{2\pi i}\oint_{|z|=2} \frac{f'(z)}{f(z)} dz = Z - P = 2 - 1 = 1$$

Let's verify:
$$\frac{f'}{f} = \frac{d}{dz}\log\left(\frac{z^2+1}{z-1}\right) = \frac{2z}{z^2+1} - \frac{1}{z-1}$$

Residues inside |z| = 2:
- At z = i: Res[2z/(z²+1), i] = 2i/(2i) = 1
- At z = -i: Res[2z/(z²+1), -i] = -2i/(-2i) = 1
- At z = 1: Res[-1/(z-1), 1] = -1

Sum: 1 + 1 + (-1) = 1 ✓

---

### Example 7: Applying Rouché for perturbation analysis

If |ε| < 1, how many zeros does z³ - z + ε have in |z| < 1?

**Solution:**
Let f(z) = z³ - z = z(z-1)(z+1), g(z) = ε.

On |z| = 1:
- |f(z)| = |z||z-1||z+1| 

At z = e^{iθ}: |f| = |e^{iθ} - 1||e^{iθ} + 1| = 2|sin(θ/2)| · 2|cos(θ/2)| = 2|sin θ|

Minimum of |f| on |z| = 1 is 0 (at z = ±1). Rouché fails!

**Different approach:** Count directly.
f(z) = z³ - z has zeros at 0, 1, -1.
- z = 0 is inside |z| < 1
- z = ±1 are ON the boundary

For small ε, the zeros move slightly. The zero at z = 0 stays inside.

By continuity of zeros, z³ - z + ε has **1 zero** in |z| < 1 for small |ε|.

---

## 🔧 Practice Problems

### Level 1: Direct Application
1. Use the argument principle to find Z - P for f(z) = z²/(z-1) inside |z| = 2.
2. How many zeros does z⁴ + z + 1 have in |z| < 1?
3. Show that z⁶ + 4z² - 1 has exactly 2 zeros in |z| < 1.

### Level 2: Rouché Applications
4. Prove that z⁴ - 6z + 3 has exactly one zero in |z| < 1.
5. How many zeros does eᶻ - 2z have in |z| < 1?
6. Show all zeros of z⁵ + z² + 1 lie in |z| < 2.

### Level 3: Theory
7. Prove: If f is analytic and |f(z)| = 1 for all z on |z| = 1, and f has no zeros in |z| ≤ 1, then f is constant.
8. Use Rouché to prove the Fundamental Theorem of Algebra.
9. Show that for |a| > e, the equation eᶻ = azⁿ has exactly n roots in |z| < 1.

### Level 4: Physics Applications  
10. Apply the Nyquist criterion to determine stability of G(s) = 1/(s² + s + 1).
11. If the Jost function f(k) has n zeros in the upper half k-plane, what does this say about bound states?

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def winding_number(f, center, radius, n_points=1000):
    """
    Compute winding number of f(z) around origin 
    as z traces a circle.
    """
    t = np.linspace(0, 2*np.pi, n_points)
    z = center + radius * np.exp(1j * t)
    w = f(z)
    
    # Compute change in argument
    darg = np.angle(w[1:]) - np.angle(w[:-1])
    
    # Handle branch cut crossings
    darg = np.where(darg > np.pi, darg - 2*np.pi, darg)
    darg = np.where(darg < -np.pi, darg + 2*np.pi, darg)
    
    total_arg_change = np.sum(darg)
    return total_arg_change / (2 * np.pi)

def count_zeros_rouche(f, g, center, radius, n_points=1000):
    """
    Verify Rouché's theorem conditions and count zeros.
    """
    t = np.linspace(0, 2*np.pi, n_points)
    z = center + radius * np.exp(1j * t)
    
    f_vals = np.abs(f(z))
    g_vals = np.abs(g(z))
    
    if np.all(g_vals < f_vals):
        print("Rouché condition satisfied: |g| < |f| on contour")
        # Count zeros of f (which equals zeros of f+g)
        wn = winding_number(f, center, radius)
        return int(round(wn))
    else:
        print("Rouché condition NOT satisfied")
        return None

def visualize_argument_principle():
    """Visualize the argument principle."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Example: f(z) = z^2 - 1 = (z-1)(z+1)
    f = lambda z: z**2 - 1
    
    # Contour
    R = 2
    theta = np.linspace(0, 2*np.pi, 500)
    z_contour = R * np.exp(1j * theta)
    w_contour = f(z_contour)
    
    # z-plane with contour and zeros
    ax = axes[0, 0]
    ax.plot(np.real(z_contour), np.imag(z_contour), 'b-', lw=2)
    ax.scatter([1, -1], [0, 0], c='red', s=200, marker='x', 
              linewidths=3, label='Zeros at ±1')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title('z-plane: Contour and zeros of f(z) = z² - 1')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # w-plane with image
    ax = axes[0, 1]
    ax.plot(np.real(w_contour), np.imag(w_contour), 'g-', lw=2)
    ax.scatter([0], [0], c='black', s=100, marker='o', label='Origin')
    ax.set_xlabel('Re(w)')
    ax.set_ylabel('Im(w)')
    ax.set_title('w-plane: Image f(C)\nWinds around origin 2 times')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Argument change
    ax = axes[1, 0]
    arg_vals = np.unwrap(np.angle(w_contour))
    ax.plot(theta, arg_vals, 'b-', lw=2)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axhline(y=4*np.pi, color='r', linestyle='--', 
              label=f'Total change = 4π (Z=2)')
    ax.set_xlabel('Parameter θ')
    ax.set_ylabel('arg(f(z))')
    ax.set_title('Change in argument')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Winding number computation
    ax = axes[1, 1]
    
    # Test various functions
    test_functions = [
        (lambda z: z, "z", 1),
        (lambda z: z**2, "z²", 2),
        (lambda z: z**3 - z, "z³-z", 2),  # zeros at 0, ±1
        (lambda z: (z-0.5)*(z+0.5), "(z-0.5)(z+0.5)", 2),
    ]
    
    radii = np.linspace(0.1, 2, 50)
    
    for f, name, expected in test_functions:
        wns = [winding_number(f, 0, r) for r in radii]
        ax.plot(radii, wns, '-', lw=2, label=f'{name}')
    
    ax.set_xlabel('Contour radius R')
    ax.set_ylabel('Winding number (= Z - P)')
    ax.set_title('Winding number vs contour size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('argument_principle.png', dpi=150)
    plt.show()

visualize_argument_principle()

def demonstrate_rouche():
    """Demonstrate Rouché's theorem."""
    print("=" * 60)
    print("ROUCHÉ'S THEOREM DEMONSTRATIONS")
    print("=" * 60)
    
    # Example 1: z^5 + 3z + 1, count zeros in |z| < 1
    print("\n1. p(z) = z⁵ + 3z + 1")
    print("   f(z) = 3z, g(z) = z⁵ + 1")
    
    f1 = lambda z: 3*z
    g1 = lambda z: z**5 + 1
    p1 = lambda z: z**5 + 3*z + 1
    
    zeros = count_zeros_rouche(f1, g1, 0, 1)
    print(f"   f(z) = 3z has 1 zero in |z|<1")
    print(f"   Therefore p(z) has {zeros} zero(s) in |z|<1")
    
    # Verify by direct winding number
    wn = winding_number(p1, 0, 1)
    print(f"   Verification (winding number): {wn:.2f}")
    
    # Example 2: e^z - 4z
    print("\n2. p(z) = e^z - 4z")
    print("   f(z) = -4z, g(z) = e^z")
    
    f2 = lambda z: -4*z
    g2 = lambda z: np.exp(z)
    p2 = lambda z: np.exp(z) - 4*z
    
    zeros = count_zeros_rouche(f2, g2, 0, 1)
    print(f"   Therefore p(z) has {zeros} zero(s) in |z|<1")
    
    # Example 3: Fundamental Theorem of Algebra
    print("\n3. Fundamental Theorem of Algebra")
    print("   p(z) = z^4 + 2z^3 - z + 3")
    
    f3 = lambda z: z**4
    g3 = lambda z: 2*z**3 - z + 3
    p3 = lambda z: z**4 + 2*z**3 - z + 3
    
    # For R = 10
    R = 10
    zeros = count_zeros_rouche(f3, g3, 0, R)
    print(f"   Using R = {R}: p(z) has {zeros} zeros in |z|<{R}")

demonstrate_rouche()

def nyquist_stability():
    """Demonstrate Nyquist stability criterion."""
    print("\n" + "=" * 60)
    print("NYQUIST STABILITY CRITERION")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Transfer function G(s) = 1/(s^2 + s + 1)
    G = lambda s: 1/(s**2 + s + 1)
    
    # Nyquist contour: imaginary axis
    omega = np.linspace(-50, 50, 1000)
    s_vals = 1j * omega
    G_vals = G(s_vals)
    
    ax = axes[0]
    ax.plot(np.real(G_vals), np.imag(G_vals), 'b-', lw=2)
    ax.scatter([-1], [0], c='red', s=200, marker='x', linewidths=3, 
              label='Critical point (-1, 0)')
    ax.set_xlabel('Re(G)')
    ax.set_ylabel('Im(G)')
    ax.set_title('Nyquist Plot: G(s) = 1/(s² + s + 1)\nNo encirclement of -1 → STABLE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Show poles in s-plane
    ax = axes[1]
    
    # Poles of G(s): s^2 + s + 1 = 0 → s = (-1 ± i√3)/2
    poles = [(-1 + 1j*np.sqrt(3))/2, (-1 - 1j*np.sqrt(3))/2]
    
    ax.scatter([p.real for p in poles], [p.imag for p in poles], 
              c='red', s=200, marker='x', linewidths=3, label='Poles')
    ax.axvline(x=0, color='g', linestyle='--', label='Stability boundary')
    ax.fill_between([-2, 0], [-2, -2], [2, 2], alpha=0.2, color='green',
                   label='Stable region (Re < 0)')
    ax.set_xlabel('Re(s)')
    ax.set_ylabel('Im(s)')
    ax.set_title('s-plane: Pole locations\nBoth poles in left half-plane → STABLE')
    ax.legend()
    ax.set_xlim(-2, 1)
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nyquist_stability.png', dpi=150)
    plt.show()
    
    print("\nG(s) = 1/(s² + s + 1)")
    print(f"Poles at s = {poles[0]:.3f} and s = {poles[1]:.3f}")
    print("Both poles have Re(s) < 0 → System is STABLE")

nyquist_stability()
```

---

## 📝 Summary

### The Argument Principle
$$\frac{1}{2\pi i}\oint_C \frac{f'(z)}{f(z)} dz = Z - P$$

- Z = number of zeros inside C
- P = number of poles inside C
- Equals winding number of f(C) around origin

### Rouché's Theorem
If |g(z)| < |f(z)| on C, then f and f + g have the same number of zeros inside C.

### Key Applications
- Counting zeros of polynomials
- Proving Fundamental Theorem of Algebra
- Nyquist stability criterion
- Topological invariants in physics

---

## ✅ Daily Checklist

- [ ] State and understand the Argument Principle
- [ ] Compute winding numbers
- [ ] Apply Rouché's theorem to count zeros
- [ ] Prove the Fundamental Theorem of Algebra
- [ ] Understand Nyquist stability criterion
- [ ] Complete computational exercises

---

## 🔮 Preview: Day 139

Tomorrow is our **Computational Lab** where we build comprehensive tools for complex analysis and explore advanced applications!
