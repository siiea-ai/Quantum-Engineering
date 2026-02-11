# Day 135: Classification of Singularities

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Types of Singularities |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Classify singularities as removable, poles, or essential
2. Determine the order of a pole
3. Apply the Casorati-Weierstrass theorem for essential singularities
4. Compute residues efficiently based on singularity type
5. Understand singularities at infinity
6. Connect singularity types to physical behavior

---

## 📚 Required Reading

### Primary Text: Churchill & Brown
- **Chapter 5, Sections 63-68**: Isolated Singularities

### Alternative: Ahlfors
- **Chapter 4.3**: Isolated Singularities

### Physics Connection
- **Shankar, QM**: Poles in scattering amplitudes

---

## 📖 Core Content: Theory and Concepts

### 1. Isolated Singularities: Definition

**Definition:** A point z₀ is an **isolated singularity** of f if:
- f is not analytic at z₀
- f IS analytic in some punctured neighborhood 0 < |z - z₀| < δ

**Non-examples of isolated singularities:**
- Branch points (like z = 0 for √z)
- Accumulation points of singularities (like z = 0 for sin(1/z)/z)

**The Classification:** Based on the Laurent series, there are exactly THREE types:

---

### 2. Type 1: Removable Singularities

**Definition:** z₀ is a **removable singularity** if the Laurent series has NO negative powers:
$$f(z) = \sum_{n=0}^{\infty} a_n (z-z_0)^n$$

**Equivalent conditions:**
1. lim_{z→z₀} (z-z₀)f(z) = 0
2. f is bounded near z₀
3. lim_{z→z₀} f(z) exists and is finite

**Riemann's Removable Singularity Theorem:**
If f is bounded near an isolated singularity z₀, then z₀ is removable, and defining f(z₀) = lim_{z→z₀} f(z) makes f analytic at z₀.

**Example:** f(z) = sin(z)/z

At z = 0: 
$$\frac{\sin z}{z} = \frac{1}{z}\left(z - \frac{z^3}{6} + \frac{z^5}{120} - ...\right) = 1 - \frac{z^2}{6} + \frac{z^4}{120} - ...$$

No negative powers → removable singularity! Define f(0) = 1.

---

### 3. Type 2: Poles

**Definition:** z₀ is a **pole of order m** if the Laurent series has finitely many negative powers, with a₋ₘ ≠ 0:
$$f(z) = \frac{a_{-m}}{(z-z_0)^m} + ... + \frac{a_{-1}}{z-z_0} + a_0 + a_1(z-z_0) + ...$$

**Equivalent conditions:**
1. lim_{z→z₀} |f(z)| = ∞
2. (z-z₀)ᵐf(z) is analytic and nonzero at z₀
3. f = g/(z-z₀)ᵐ where g is analytic with g(z₀) ≠ 0

**Simple pole:** m = 1 (order 1)
**Double pole:** m = 2 (order 2)

**Key Property:** Near a pole, |f(z)| → ∞ as z → z₀.

**Examples:**

1. f(z) = 1/(z-1)² has a **pole of order 2** at z = 1

2. f(z) = eᶻ/(z-1) has a **simple pole** at z = 1

3. f(z) = (z²-1)/(z-1) = z+1 (for z ≠ 1) has a **removable singularity** at z = 1

4. f(z) = 1/sin(z) has **simple poles** at z = nπ (zeros of sin are simple)

---

### 4. Type 3: Essential Singularities

**Definition:** z₀ is an **essential singularity** if the Laurent series has infinitely many negative powers.

**Examples:**
1. e^{1/z} at z = 0: e^{1/z} = 1 + 1/z + 1/(2!z²) + 1/(3!z³) + ...

2. sin(1/z) at z = 0: sin(1/z) = 1/z - 1/(3!z³) + 1/(5!z⁵) - ...

3. z·e^{1/z} at z = 0

**Casorati-Weierstrass Theorem:**
If z₀ is an essential singularity, then for any complex number w and any ε > 0, there exists z arbitrarily close to z₀ with |f(z) - w| < ε.

In other words: f takes values arbitrarily close to ANY complex number in every neighborhood of an essential singularity!

**Great Picard Theorem (stronger):**
In any neighborhood of an essential singularity, f takes every complex value infinitely often, with at most one exception.

**Example:** e^{1/z} takes every nonzero value infinitely often near z = 0. (The exception is 0.)

---

### 5. Quick Classification Tests

| Test | Removable | Pole (order m) | Essential |
|------|-----------|----------------|-----------|
| lim_{z→z₀} f(z) | Finite | ∞ | DNE |
| lim_{z→z₀} (z-z₀)f(z) | 0 | 0 (if m>1) or finite≠0 (if m=1) | DNE |
| lim_{z→z₀} (z-z₀)ᵐf(z) | 0 (if m≥1) | finite ≠ 0 | DNE for all m |
| Bounded near z₀? | Yes | No | No |
| Principal part | None | Finite | Infinite |

**Practical Algorithm:**
1. Try to find lim_{z→z₀} f(z). If finite → removable.
2. If limit is ∞, find smallest m such that lim_{z→z₀} (z-z₀)ᵐf(z) is finite and nonzero → pole of order m.
3. If no such m exists → essential singularity.

---

### 6. Residue Computation Based on Type

**For removable singularity:** Res[f, z₀] = 0

**For simple pole (m = 1):**
$$\boxed{\text{Res}[f, z_0] = \lim_{z \to z_0} (z-z_0)f(z)}$$

**For pole of order m:**
$$\boxed{\text{Res}[f, z_0] = \frac{1}{(m-1)!} \lim_{z \to z_0} \frac{d^{m-1}}{dz^{m-1}}\left[(z-z_0)^m f(z)\right]}$$

**Special case - quotient of analytic functions:**
If f(z) = g(z)/h(z) where g(z₀) ≠ 0 and h has a simple zero at z₀:
$$\boxed{\text{Res}[f, z_0] = \frac{g(z_0)}{h'(z_0)}}$$

**For essential singularity:** Must find Laurent series explicitly or use other methods.

---

### 7. Singularities at Infinity

**Definition:** The behavior of f(z) at z = ∞ is determined by studying g(w) = f(1/w) at w = 0.

| f at ∞ | g(w) = f(1/w) at w = 0 |
|--------|------------------------|
| Removable | Removable |
| Pole of order m | Pole of order m |
| Essential | Essential |

**Examples:**

1. f(z) = z² at ∞: g(w) = 1/w² has pole of order 2 at 0 → f has **pole of order 2 at ∞**

2. f(z) = eᶻ at ∞: g(w) = e^{1/w} has essential singularity at 0 → f has **essential singularity at ∞**

3. f(z) = 1/(1+z²) at ∞: g(w) = w²/(w²+1) → g(0) = 0 → f has **removable singularity at ∞** (actually a zero!)

**Rational Functions:** 
- Always have poles or removable singularities (no essential singularities)
- Degree of numerator > degree of denominator → pole at ∞
- Degree of numerator ≤ degree of denominator → removable at ∞

---

### 8. Zeros and Their Relationship to Poles

**Definition:** z₀ is a **zero of order m** of f if:
$$f(z) = (z-z_0)^m g(z)$$ 
where g is analytic and g(z₀) ≠ 0.

**Key Relationship:** 
- If f has a zero of order m at z₀, then 1/f has a pole of order m at z₀
- If f has a pole of order m at z₀, then 1/f has a zero of order m at z₀

**Example:** sin z has simple zeros at z = nπ, so 1/sin z = csc z has simple poles at z = nπ.

---

### 9. 🔬 Quantum Mechanics Connection

**Poles in Green's Functions:**
The resolvent G(E) = (E - H)⁻¹ has:
- **Simple poles** at discrete eigenvalues Eₙ
- **Branch cuts** along continuous spectrum

**Resonances:**
Scattering amplitudes have poles at complex energies E = E₀ - iΓ/2:
- Real part E₀ = resonance energy
- Imaginary part Γ/2 = half-width (decay rate)
- These are poles of the analytically continued S-matrix

**The Pole-Zero Structure Determines Physics:**
- Poles → bound states and resonances
- Zeros → destructive interference, transmission zeros
- Essential singularities → non-perturbative effects

**Example: Coulomb Scattering**
The Coulomb scattering amplitude has:
- Poles at E = -13.6/n² eV (bound states of hydrogen)
- Essential singularity structure from the long-range nature of the potential

---

## ✏️ Worked Examples

### Example 1: Classify all singularities of f(z) = (eᶻ - 1)/z²

**Solution:**
At z = 0: 
$$\frac{e^z - 1}{z^2} = \frac{1}{z^2}\left(z + \frac{z^2}{2!} + \frac{z^3}{3!} + ...\right) = \frac{1}{z} + \frac{1}{2} + \frac{z}{6} + ...$$

Principal part has one term (1/z) → **simple pole at z = 0**

No other finite singularities (eᶻ - 1 and z² are entire).

At ∞: g(w) = (e^{1/w} - 1)w² has essential singularity at w = 0 → **essential singularity at ∞**

---

### Example 2: Find the order of the pole of cot z at z = 0

**Solution:**
cot z = cos z / sin z

At z = 0:
- cos 0 = 1 ≠ 0
- sin z has a simple zero (sin z = z - z³/6 + ...)

So cot z has a **simple pole** at z = 0.

Verify: lim_{z→0} z·cot z = lim_{z→0} z·cos z/sin z = lim_{z→0} cos z/(sin z/z) = 1/1 = 1 ≠ 0 ✓

---

### Example 3: Find residues of f(z) = z/((z-1)(z-2)²)

**Solution:**

**At z = 1 (simple pole):**
$$\text{Res}[f, 1] = \lim_{z \to 1} (z-1) \cdot \frac{z}{(z-1)(z-2)^2} = \frac{1}{(1-2)^2} = 1$$

**At z = 2 (pole of order 2):**
$$\text{Res}[f, 2] = \lim_{z \to 2} \frac{d}{dz}\left[(z-2)^2 \cdot \frac{z}{(z-1)(z-2)^2}\right]$$
$$= \lim_{z \to 2} \frac{d}{dz}\left[\frac{z}{z-1}\right] = \lim_{z \to 2} \frac{(z-1) - z}{(z-1)^2} = \frac{-1}{1} = -1$$

---

### Example 4: Show e^{1/z} takes all nonzero values near z = 0

**Solution:**
We want to solve e^{1/z} = w for any w ≠ 0.

Taking log: 1/z = log w + 2πin for n ∈ ℤ

So: z = 1/(log w + 2πin)

For any w ≠ 0 and sufficiently large |n|, these z values can be arbitrarily close to 0!

This demonstrates Casorati-Weierstrass for e^{1/z}. The only exception is w = 0, which e^{1/z} never equals.

---

### Example 5: Residue via the quotient rule

Find the residue of f(z) = eᶻ/sin z at z = 0.

**Solution:**
sin z has a simple zero at z = 0, and eᶻ is nonzero there.

Using the formula Res[g/h, z₀] = g(z₀)/h'(z₀):
$$\text{Res}\left[\frac{e^z}{\sin z}, 0\right] = \frac{e^0}{\cos 0} = \frac{1}{1} = 1$$

---

## 🔧 Practice Problems

### Level 1: Classification
Classify each singularity:
1. f(z) = (z² - 1)/(z - 1) at z = 1
2. f(z) = 1/(z⁴ + 1) at z = e^{iπ/4}
3. f(z) = sin z/z³ at z = 0
4. f(z) = e^{1/(z-1)} at z = 1

### Level 2: Residue Computation
Find the residue at each singularity:
5. f(z) = z²/(z² + 1) at z = i
6. f(z) = eᶻ/(z² - 1) at z = 1 and z = -1
7. f(z) = 1/(z(z-1)³) at z = 0 and z = 1
8. f(z) = cot z at z = π

### Level 3: Singularities at Infinity
9. Determine the nature of singularity at ∞ for f(z) = z³/(z² + 1).
10. Find all singularities (including ∞) for f(z) = tan z.

### Level 4: Theory
11. Prove: If f has a pole of order m at z₀ and g has a pole of order n at z₀, then fg has a pole of order m + n at z₀.
12. Prove: The sum of residues of a rational function (including at ∞) is zero.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

def classify_singularity(f, z0, eps=1e-8, max_m=10):
    """
    Classify singularity of f at z0.
    Returns: ('removable', 0), ('pole', m), or ('essential', None)
    """
    # Test if bounded (removable)
    test_points = z0 + eps * np.exp(1j * np.linspace(0, 2*np.pi, 100))
    try:
        values = f(test_points)
        if np.all(np.isfinite(values)) and np.max(np.abs(values)) < 1e10:
            # Might be removable - check if limit exists
            return ('removable', 0)
    except:
        pass
    
    # Test for pole of order m
    for m in range(1, max_m + 1):
        g = lambda z: (z - z0)**m * f(z)
        try:
            # Check if (z-z0)^m * f(z) is bounded and nonzero
            values = g(test_points)
            if np.all(np.isfinite(values)):
                center_val = np.mean(values)
                if np.abs(center_val) > eps and np.std(values) < np.abs(center_val):
                    return ('pole', m)
        except:
            pass
    
    return ('essential', None)

def visualize_singularity_types():
    """Visualize different types of singularities."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    examples = [
        (lambda z: np.sin(z)/z, "sin(z)/z at z=0\n(Removable)", 0),
        (lambda z: 1/z, "1/z at z=0\n(Simple pole)", 0),
        (lambda z: 1/z**2, "1/z² at z=0\n(Pole order 2)", 0),
        (lambda z: np.exp(1/z), "e^(1/z) at z=0\n(Essential)", 0),
        (lambda z: 1/((z-1)*(z+1)), "1/((z-1)(z+1))\n(Poles at ±1)", None),
        (lambda z: z*np.exp(1/z), "z·e^(1/z) at z=0\n(Essential)", 0),
    ]
    
    for ax, (f, title, z0) in zip(axes.flat, examples):
        with np.errstate(all='ignore'):
            W = f(Z)
            W[~np.isfinite(W)] = np.nan
        
        # Plot phase with brightness from modulus
        phase = np.angle(W)
        mag = np.abs(W)
        mag_normalized = 1 - 1/(1 + np.log1p(mag)/3)
        
        # Create HSV image
        H = (phase + np.pi) / (2 * np.pi)
        S = np.ones_like(H) * 0.8
        V = np.nan_to_num(mag_normalized, nan=0)
        
        from matplotlib.colors import hsv_to_rgb
        RGB = hsv_to_rgb(np.dstack([H, S, V]))
        
        ax.imshow(RGB, extent=[-2, 2, -2, 2], origin='lower')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        
        # Mark singularities
        if z0 is not None:
            ax.scatter([np.real(z0)], [np.imag(z0)], c='white', s=100, 
                      marker='x', linewidths=2)
    
    plt.tight_layout()
    plt.savefig('singularity_types.png', dpi=150)
    plt.show()

visualize_singularity_types()

# Demonstrate Casorati-Weierstrass for essential singularity
def casorati_weierstrass_demo():
    """Show that e^(1/z) comes arbitrarily close to any value near z=0."""
    print("=" * 60)
    print("CASORATI-WEIERSTRASS THEOREM DEMONSTRATION")
    print("=" * 60)
    
    target_values = [1+1j, -5, 100j, 0.01, -2-3j]
    
    print("\nFor f(z) = e^(1/z), finding z near 0 where f(z) ≈ w:")
    print("-" * 60)
    
    for w in target_values:
        # Solve e^(1/z) = w
        # 1/z = log(w) + 2πin
        # z = 1/(log(w) + 2πin)
        
        log_w = np.log(w)
        
        print(f"\nTarget w = {w}")
        z_values = []
        for n in range(-5, 6):
            z = 1 / (log_w + 2j * np.pi * n)
            if np.abs(z) < 1:  # Only show z near origin
                z_values.append((n, z, np.exp(1/z)))
        
        if z_values:
            closest = min(z_values, key=lambda x: np.abs(x[1]))
            n, z, fz = closest
            print(f"  z = {z:.6f} (|z| = {np.abs(z):.6f})")
            print(f"  f(z) = {fz:.6f}")
            print(f"  |f(z) - w| = {np.abs(fz - w):.2e}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot many values of e^(1/z) for z near 0
    r = np.linspace(0.01, 0.5, 100)
    theta = np.linspace(0, 2*np.pi, 500)
    R, Theta = np.meshgrid(r, theta)
    Z = R * np.exp(1j * Theta)
    
    W = np.exp(1/Z)
    
    ax.scatter(np.real(W.flatten()), np.imag(W.flatten()), 
              c=np.abs(Z.flatten()), cmap='viridis', s=1, alpha=0.3)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('Re(e^(1/z))')
    ax.set_ylabel('Im(e^(1/z))')
    ax.set_title('Values of e^(1/z) for |z| < 0.5\n(Fills dense region of ℂ)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], label='|z|')
    
    plt.savefig('casorati_weierstrass.png', dpi=150)
    plt.show()

casorati_weierstrass_demo()

# Residue computation comparison
def residue_methods():
    """Compare different methods of computing residues."""
    print("\n" + "=" * 60)
    print("RESIDUE COMPUTATION METHODS")
    print("=" * 60)
    
    # Example: f(z) = e^z / (z^2 - 1) = e^z / ((z-1)(z+1))
    f = lambda z: np.exp(z) / (z**2 - 1)
    
    print("\nf(z) = e^z / (z² - 1)")
    print("\n1. At z = 1 (simple pole):")
    
    # Method 1: Direct limit
    eps = 1e-8
    res_limit = (1 + eps - 1) * f(1 + eps)
    print(f"   Limit method: {res_limit:.6f}")
    
    # Method 2: Quotient formula
    g = lambda z: np.exp(z)
    h = lambda z: z**2 - 1
    h_prime = lambda z: 2*z
    res_quotient = g(1) / h_prime(1)
    print(f"   Quotient formula: {res_quotient:.6f}")
    
    # Method 3: Contour integration
    C = 1 + 0.1 * np.exp(1j * np.linspace(0, 2*np.pi, 1000))
    integrand = f(C) * 1j * 0.1 * np.exp(1j * np.linspace(0, 2*np.pi, 1000))
    res_contour = np.trapz(integrand) / (2 * np.pi * 1j)
    print(f"   Contour integral: {res_contour:.6f}")
    
    print(f"   Exact: e/2 = {np.e/2:.6f}")
    
    print("\n2. At z = -1 (simple pole):")
    res_quotient_neg1 = g(-1) / h_prime(-1)
    print(f"   Quotient formula: {res_quotient_neg1:.6f}")
    print(f"   Exact: -e^(-1)/2 = {-np.exp(-1)/2:.6f}")

residue_methods()
```

---

## 📝 Summary

### The Three Types of Isolated Singularities

| Type | Principal Part | Behavior | Residue |
|------|---------------|----------|---------|
| Removable | None | f bounded | 0 |
| Pole (order m) | Finite (m terms) | \|f\| → ∞ | Compute by formula |
| Essential | Infinite | Wild (C-W theorem) | From Laurent series |

### Quick Identification

| Condition | Singularity Type |
|-----------|-----------------|
| lim f(z) exists and finite | Removable |
| lim \|f(z)\| = ∞ | Pole |
| Neither | Essential |

### Residue Formulas

- **Simple pole:** Res = lim_{z→z₀} (z-z₀)f(z)
- **Quotient g/h:** Res = g(z₀)/h'(z₀)
- **Pole order m:** Res = (1/(m-1)!) · lim d^{m-1}/dz^{m-1}[(z-z₀)^m f(z)]

---

## ✅ Daily Checklist

- [ ] Classify singularities (removable, pole, essential)
- [ ] Determine pole orders
- [ ] Understand Casorati-Weierstrass theorem
- [ ] Compute residues efficiently
- [ ] Analyze singularities at infinity
- [ ] Connect to physics applications
- [ ] Complete computational lab

---

## 🔮 Preview: Day 136

Tomorrow we prove and apply the **Residue Theorem** — the most powerful tool in complex integration! It allows us to evaluate contour integrals by simply summing residues.
