# Day 170: Analytic Functions and Holomorphicity — Complex Differentiability

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Analyticity and Differentiability |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of this day, you will be able to:

1. Define analyticity (holomorphicity) for complex functions
2. Distinguish complex differentiability from real differentiability
3. Identify entire functions (analytic everywhere)
4. Classify singularities: removable, poles, and essential
5. Connect analyticity to Green's functions in quantum mechanics
6. Apply the concept to quantum scattering problems

---

## Core Content

### 1. Definition of Analyticity

**Definition:** A function $f(z)$ is **analytic** (or **holomorphic**) at a point $z_0$ if:

1. $f$ is defined in a neighborhood of $z_0$
2. $f'(z_0)$ exists: $$f'(z_0) = \lim_{h \to 0} \frac{f(z_0 + h) - f(z_0)}{h}$$

A function is **analytic on a domain $D$** if it is analytic at every point in $D$.

**Complex Derivative:**

$$\boxed{f'(z) = \lim_{h \to 0} \frac{f(z + h) - f(z)}{h}}$$

where $h \in \mathbb{C}$ can approach 0 from any direction in the complex plane.

### 2. Complex vs. Real Differentiability

This is a **fundamental distinction** that makes complex analysis so powerful:

**Real Analysis (1D):**
- Limit from left: $\lim_{h \to 0^-} \frac{f(x+h) - f(x)}{h}$
- Limit from right: $\lim_{h \to 0^+} \frac{f(x+h) - f(x)}{h}$
- Only 2 directions to check
- Many functions are differentiable (e.g., $|x|$ at $x \neq 0$)

**Complex Analysis (2D):**
- $h$ can approach 0 from **infinitely many directions**
- The limit must be **path-independent**
- This is an extremely strong constraint!
- Far fewer functions are complex differentiable

**Example: $f(z) = \bar{z}$ is NOT analytic**

For $f(z) = \bar{z} = x - iy$:

$$\frac{f(z + h) - f(z)}{h} = \frac{\bar{h}}{h}$$

- Along real axis ($h = \epsilon \in \mathbb{R}$): $\frac{\epsilon}{\epsilon} = 1$
- Along imaginary axis ($h = i\epsilon$): $\frac{-i\epsilon}{i\epsilon} = -1$

The limits differ! Therefore $\bar{z}$ is **nowhere analytic**.

### 3. Entire Functions

**Definition:** A function is **entire** (or **integral**) if it is analytic on all of $\mathbb{C}$.

**Examples of Entire Functions:**

| Function | Proof of Entireness |
|----------|---------------------|
| Polynomials $p(z)$ | Sum and product of analytic functions |
| $e^z$ | Power series converges everywhere |
| $\sin z$, $\cos z$ | Defined via $e^{iz}$, entire |
| $\sinh z$, $\cosh z$ | Defined via $e^z$, entire |

**Key Properties of Entire Functions:**

1. **Infinitely differentiable** — If $f$ is analytic, all derivatives exist and are analytic
2. **Equal to Taylor series** — In any disk of analyticity
3. **Liouville's Theorem:** A bounded entire function is constant

**Liouville's Theorem:**

$$\boxed{\text{If } f \text{ is entire and } |f(z)| \leq M \text{ for all } z, \text{ then } f \text{ is constant.}}$$

**Application:** Proves the Fundamental Theorem of Algebra — every non-constant polynomial has a root.

### 4. Singularities

A **singularity** is a point where $f$ fails to be analytic.

**Classification of Isolated Singularities:**

At an isolated singularity $z_0$, the Laurent series is:

$$f(z) = \sum_{n=-\infty}^{\infty} a_n (z - z_0)^n = \underbrace{\sum_{n=1}^{\infty} \frac{a_{-n}}{(z-z_0)^n}}_{\text{Principal part}} + \underbrace{\sum_{n=0}^{\infty} a_n (z-z_0)^n}_{\text{Analytic part}}$$

**1. Removable Singularity:**
- Principal part is zero (all $a_{-n} = 0$)
- $\lim_{z \to z_0} f(z)$ exists and is finite
- Can be "removed" by defining $f(z_0) = \lim_{z \to z_0} f(z)$

**Example:** $f(z) = \frac{\sin z}{z}$ at $z = 0$

$$\frac{\sin z}{z} = \frac{z - z^3/6 + z^5/120 - \cdots}{z} = 1 - \frac{z^2}{6} + \frac{z^4}{120} - \cdots$$

No negative powers! Define $f(0) = 1$ to remove singularity.

**2. Poles:**
- Principal part has finitely many nonzero terms
- **Order $m$:** If $a_{-m} \neq 0$ but $a_{-n} = 0$ for $n > m$
- $|f(z)| \to \infty$ as $z \to z_0$

**Example:** $f(z) = \frac{1}{(z-1)^2}$ has a pole of order 2 at $z = 1$

**Simple Pole (order 1):**
$$f(z) = \frac{a_{-1}}{z - z_0} + (\text{analytic})$$

**3. Essential Singularity:**
- Principal part has infinitely many nonzero terms
- Behavior is extremely wild

**Picard's Great Theorem:** In every neighborhood of an essential singularity, $f$ takes every complex value (with at most one exception) infinitely many times.

**Example:** $f(z) = e^{1/z}$ at $z = 0$

$$e^{1/z} = 1 + \frac{1}{z} + \frac{1}{2z^2} + \frac{1}{6z^3} + \cdots$$

Infinitely many negative powers — essential singularity.

### 5. Quantum Mechanics Connection: Green's Functions

**The Resolvent (Green's Function):**

In quantum mechanics, the Green's function is:

$$\boxed{G(E) = \frac{1}{E - H + i\varepsilon}}$$

where $H$ is the Hamiltonian and $\varepsilon \to 0^+$ is a regulator.

**Analyticity Properties:**

1. **Analytic in upper half-plane:** $G(E + i\varepsilon)$ is analytic for $\text{Im}(E) > 0$
2. **Poles on real axis:** Bound state energies $E_n$ where $H|\psi_n\rangle = E_n|\psi_n\rangle$
3. **Branch cuts:** Start at continuum threshold (scattering states)

**Physical Interpretation:**

| Singularity Type | Physical Meaning |
|------------------|------------------|
| Poles (real axis, $E < E_{threshold}$) | Bound states |
| Poles (second Riemann sheet) | Resonances/unstable states |
| Branch cut | Continuum of scattering states |

**Dispersion Relations:**

Analyticity leads to powerful constraints. For a causal response function:

$$\text{Re}[G(E)] = \frac{1}{\pi} \mathcal{P} \int_{-\infty}^{\infty} \frac{\text{Im}[G(E')]}{E' - E} dE'$$

This **Kramers-Kronig relation** connects real and imaginary parts — directly from analyticity!

**Causality and Analyticity:**

The $+i\varepsilon$ prescription ensures:
- **Retarded Green's function:** Effect follows cause
- **Analyticity in upper half-plane:** Mathematical encoding of causality

---

## Worked Examples

### Example 1: Classifying Singularities

**Problem:** Classify the singularities of $f(z) = \frac{z^2 - 1}{z^2 + 1}$.

**Solution:**

Step 1: Find singularities (where denominator = 0)
$$z^2 + 1 = 0 \implies z = \pm i$$

Step 2: Check numerator at these points
$$z^2 - 1 \big|_{z=i} = -1 - 1 = -2 \neq 0$$
$$z^2 - 1 \big|_{z=-i} = -1 - 1 = -2 \neq 0$$

Step 3: Since numerator $\neq 0$ and denominator has simple zeros, both are **simple poles** (order 1).

Step 4: Find residues
$$\text{Res}_{z=i} f(z) = \lim_{z \to i} (z - i) \frac{z^2 - 1}{z^2 + 1} = \lim_{z \to i} \frac{z^2 - 1}{z + i} = \frac{-2}{2i} = \frac{1}{i} = -i$$

**Answer:** Simple poles at $z = i$ and $z = -i$.

### Example 2: Removable Singularity

**Problem:** Show $f(z) = \frac{e^z - 1}{z}$ has a removable singularity at $z = 0$.

**Solution:**

Expand $e^z$ in Taylor series:
$$e^z = 1 + z + \frac{z^2}{2!} + \frac{z^3}{3!} + \cdots$$

Therefore:
$$\frac{e^z - 1}{z} = \frac{z + \frac{z^2}{2!} + \frac{z^3}{3!} + \cdots}{z} = 1 + \frac{z}{2!} + \frac{z^2}{3!} + \cdots$$

No negative powers! The limit exists:
$$\lim_{z \to 0} \frac{e^z - 1}{z} = 1$$

Define $f(0) = 1$ to extend analytically. ✓

### Example 3: Essential Singularity Behavior

**Problem:** Show that $e^{1/z}$ takes all nonzero complex values in any neighborhood of $z = 0$.

**Solution:**

Want to solve $e^{1/z} = w$ for any $w \neq 0$.

Taking logarithm: $\frac{1}{z} = \ln w + 2\pi i k$ for $k \in \mathbb{Z}$

So: $z = \frac{1}{\ln w + 2\pi i k}$

For any $w \neq 0$:
- As $|k| \to \infty$, $|z| \to 0$
- For each $k$, we get a different solution

Therefore, in any neighborhood of $z = 0$, there are solutions for every $w \neq 0$.

The exception is $w = 0$ (since $e^{1/z}$ never equals 0).

---

## Practice Problems

### Level 1: Direct Application

1. Show that $f(z) = z^3 - 2z + 1$ is entire.

2. Find all singular points of $f(z) = \frac{1}{z(z-2)^2}$ and classify each.

3. Determine if $f(z) = \frac{\cos z - 1}{z^2}$ has a removable singularity at $z = 0$.

### Level 2: Intermediate

4. Find the order of the pole of $f(z) = \frac{\sin z}{z^4}$ at $z = 0$.

5. Prove that $f(z) = e^{z^2}$ is entire but not constant. Does this contradict Liouville's theorem?

6. For $f(z) = \frac{1}{e^z - 1}$, find all singularities in $\mathbb{C}$ and classify them.

### Level 3: Challenging

7. The quantum S-matrix element is $S(k) = e^{2i\delta(k)}$ where $\delta(k)$ is the phase shift. If $\delta(k)$ has a pole at $k = k_0 - i\gamma$:
   - Where is this pole in the complex $k$-plane?
   - What is the physical interpretation?
   - How does this relate to resonance width?

8. Prove that if $f$ is entire and $|f(z)| \leq A + B|z|^n$ for some constants $A, B$ and integer $n$, then $f$ is a polynomial of degree $\leq n$.

9. Show that an analytic function with only removable singularities can be extended to an entire function.

---

## Computational Lab

```python
"""
Day 170: Analytic Functions and Singularities
Visualizing poles, essential singularities, and Green's functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm

fig, axes = plt.subplots(2, 3, figsize=(16, 11))

# ========================================
# 1. SIMPLE POLE: f(z) = 1/(z - 1)
# ========================================
ax = axes[0, 0]
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y

f1 = 1 / (Z - 1)
f1_mag = np.abs(f1)
f1_mag = np.clip(f1_mag, 0, 10)  # Clip for visualization

contour1 = ax.contourf(X, Y, f1_mag, levels=30, cmap='viridis')
ax.plot(1, 0, 'r*', markersize=15, label='Simple pole at z=1')
ax.set_title('|f(z)| = |1/(z-1)| — Simple Pole', fontsize=11)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.legend(loc='upper right')
plt.colorbar(contour1, ax=ax, label='Magnitude')

# ========================================
# 2. DOUBLE POLE: f(z) = 1/(z - 1)^2
# ========================================
ax = axes[0, 1]
f2 = 1 / ((Z - 1)**2)
f2_mag = np.clip(np.abs(f2), 0, 15)

contour2 = ax.contourf(X, Y, f2_mag, levels=30, cmap='plasma')
ax.plot(1, 0, 'r*', markersize=15, label='Double pole at z=1')
ax.set_title('|f(z)| = |1/(z-1)²| — Double Pole', fontsize=11)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.legend(loc='upper right')
plt.colorbar(contour2, ax=ax, label='Magnitude')

# ========================================
# 3. TWO POLES: f(z) = 1/(z² - 1)
# ========================================
ax = axes[0, 2]
f3 = 1 / (Z**2 - 1)
f3_mag = np.clip(np.abs(f3), 0, 10)

contour3 = ax.contourf(X, Y, f3_mag, levels=30, cmap='coolwarm')
ax.plot(1, 0, 'r*', markersize=12)
ax.plot(-1, 0, 'r*', markersize=12)
ax.annotate('Pole', (1, 0), xytext=(1.3, 0.5), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('Pole', (-1, 0), xytext=(-1.8, 0.5), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='black'))
ax.set_title('|f(z)| = |1/(z²-1)| — Two Simple Poles', fontsize=11)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
plt.colorbar(contour3, ax=ax, label='Magnitude')

# ========================================
# 4. ESSENTIAL SINGULARITY: e^(1/z)
# ========================================
ax = axes[1, 0]

# Mask near origin to avoid numerical issues
mask = np.abs(Z) < 0.1
f4 = np.exp(1 / Z)
f4[mask] = np.nan

# Use log scale for better visualization
f4_log = np.log10(np.abs(f4) + 1e-10)
f4_log = np.clip(f4_log, -2, 4)

contour4 = ax.contourf(X, Y, f4_log, levels=30, cmap='twilight')
circle = Circle((0, 0), 0.1, fill=False, edgecolor='red',
                linewidth=2, linestyle='--')
ax.add_patch(circle)
ax.plot(0, 0, 'r*', markersize=12)
ax.set_title('log|e^(1/z)| — Essential Singularity', fontsize=11)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.annotate('Essential\nsingularity', (0, 0), xytext=(0.5, 0.8),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))
plt.colorbar(contour4, ax=ax, label='log₁₀|f(z)|')

# ========================================
# 5. GREEN'S FUNCTION REAL PART
# ========================================
ax = axes[1, 1]

E_values = np.linspace(-3, 3, 500)
E0 = 0.0  # Bound state energy
epsilon_values = [0.01, 0.05, 0.1, 0.3]

for eps in epsilon_values:
    G = 1 / (E_values - E0 + 1j*eps)
    ax.plot(E_values, np.real(G), linewidth=2, label=f'ε = {eps}')

ax.axvline(x=E0, color='red', linestyle='--', alpha=0.5, label='E₀')
ax.set_xlabel('Energy E')
ax.set_ylabel('Re[G(E)]')
ax.set_title("Green's Function: Re[1/(E - E₀ + iε)]", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 1)

# ========================================
# 6. GREEN'S FUNCTION IMAGINARY PART
# ========================================
ax = axes[1, 2]

for eps in epsilon_values:
    G = 1 / (E_values - E0 + 1j*eps)
    ax.plot(E_values, np.imag(G), linewidth=2, label=f'ε = {eps}')

ax.axvline(x=E0, color='red', linestyle='--', alpha=0.5, label='E₀')
ax.set_xlabel('Energy E')
ax.set_ylabel('Im[G(E)]')
ax.set_title("Green's Function: Im[1/(E - E₀ + iε)]", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.1, 0.1)

# Annotation: Lorentzian approaches delta function
ax.annotate('→ -πδ(E-E₀)\nas ε→0', xy=(0, -1), xytext=(1.5, -0.5),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'))

plt.tight_layout()
plt.savefig('day_170_singularities.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# SINGULARITY COMPARISON TABLE
# ========================================
print("=" * 70)
print("SINGULARITY CLASSIFICATION SUMMARY")
print("=" * 70)
print()
print("| Singularity Type      | Laurent Series                | Behavior at z₀        |")
print("|" + "-"*22 + "|" + "-"*32 + "|" + "-"*22 + "|")
print("| Removable             | No principal part             | lim f(z) exists       |")
print("| Pole (order m)        | Finitely many negative powers | |f(z)| → ∞            |")
print("| Essential             | ∞ many negative powers        | Picard theorem        |")
print()
print("QUANTUM MECHANICS CONNECTIONS:")
print("-" * 70)
print("• Green's function G(E) = 1/(E - H + iε) is analytic in upper half-plane")
print("• Poles on real axis (below threshold) = bound states")
print("• Poles on second Riemann sheet = resonances/unstable states")
print("• Branch cuts starting at threshold = continuum of scattering states")
print("• The +iε prescription enforces causality (retarded propagator)")
print()
print("PHYSICAL INTERPRETATION OF Im[G(E)]:")
print("-" * 70)
print("• Im[G(E)] → -πδ(E - E₀) as ε → 0")
print("• Spectral density: ρ(E) = -Im[G(E)]/π")
print("• Peaks in ρ(E) indicate bound states or resonances")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Complex Derivative | $f'(z) = \lim_{h \to 0} \frac{f(z+h) - f(z)}{h}$ |
| Analytic at $z_0$ | $f'(z_0)$ exists (path-independent limit) |
| Entire Function | Analytic on all of $\mathbb{C}$ |
| Liouville's Theorem | Bounded entire ⟹ constant |
| Laurent Series | $f(z) = \sum_{n=-\infty}^{\infty} a_n(z-z_0)^n$ |
| Green's Function | $G(E) = (E - H + i\varepsilon)^{-1}$ |

### Singularity Classification

| Type | Principal Part | Behavior |
|------|----------------|----------|
| Removable | None | Limit exists |
| Pole (order $m$) | Finite | $\|f\| \to \infty$ |
| Essential | Infinite | Picard theorem |

### Main Takeaways

1. **Complex differentiability** requires path-independent limits — much stronger than real
2. **Entire functions** are analytic everywhere (polynomials, $e^z$, trig functions)
3. **Singularities** are classified by Laurent series principal part
4. **Green's functions** encode quantum dynamics through their analyticity structure
5. **Causality** is mathematically expressed through $+i\varepsilon$ and upper half-plane analyticity

---

## Daily Checklist

- [ ] I can define analyticity and explain why it's stronger than real differentiability
- [ ] I can identify entire functions and apply Liouville's theorem
- [ ] I can classify singularities as removable, poles, or essential
- [ ] I understand the Laurent series structure at singularities
- [ ] I can explain the physical meaning of poles in Green's functions
- [ ] I completed the computational visualizations

---

## Preview: Day 171

Tomorrow we derive the **Cauchy-Riemann Equations** — the conditions that real and imaginary parts must satisfy for a function to be analytic. This leads to:
- The connection between analyticity and harmonic functions
- Why $u$ and $v$ are not independent — they're related by PDEs
- The foundation for Cauchy's integral theorem

---

*"The study of holomorphic functions has an inner unity that originates from Cauchy."*
— Henri Poincaré
