# Day 194: Special Functions from Complex Analysis

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Gamma & Beta Functions |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Zeta Function & Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 194, you will be able to:

1. Define and analyze the Gamma function via complex analysis
2. Apply the reflection and duplication formulas
3. Understand the Riemann zeta function and analytic continuation
4. Compute special values of $\Gamma$ and $\zeta$
5. Connect to physics: partition functions, Casimir effect
6. Use special functions in quantum calculations

---

## Core Content

### 1. The Gamma Function

**Definition (Euler's integral):**
$$\boxed{\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt, \quad \text{Re}(z) > 0}$$

**Key properties:**
- $\Gamma(n+1) = n!$ for positive integers
- $\Gamma(z+1) = z\Gamma(z)$ (functional equation)
- $\Gamma(1) = 1$, $\Gamma(1/2) = \sqrt{\pi}$

### 2. Analytic Continuation of Gamma

Use $\Gamma(z+1) = z\Gamma(z)$ to extend:
$$\Gamma(z) = \frac{\Gamma(z+1)}{z} = \frac{\Gamma(z+2)}{z(z+1)} = \cdots$$

**Result:** $\Gamma(z)$ is meromorphic on $\mathbb{C}$ with simple poles at $z = 0, -1, -2, \ldots$

**Residue at $z = -n$:**
$$\text{Res}_{z=-n} \Gamma(z) = \frac{(-1)^n}{n!}$$

### 3. Reflection Formula

$$\boxed{\Gamma(z)\Gamma(1-z) = \frac{\pi}{\sin(\pi z)}}$$

**Proof sketch:** Use contour integration on $\int_0^\infty \frac{t^{z-1}}{1+t} dt = \frac{\pi}{\sin(\pi z)}$ combined with Beta function.

**Consequence:** $\Gamma(1/2) = \sqrt{\pi}$ (set $z = 1/2$).

### 4. The Beta Function

**Definition:**
$$B(a,b) = \int_0^1 t^{a-1}(1-t)^{b-1} dt$$

**Relation to Gamma:**
$$\boxed{B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}}$$

**Application:** Normalize quantum wave functions, compute integrals.

### 5. The Riemann Zeta Function

**Definition (for Re$(s) > 1$):**
$$\boxed{\zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s}}$$

**Euler product:**
$$\zeta(s) = \prod_p \frac{1}{1 - p^{-s}}$$

where product is over primes.

### 6. Analytic Continuation of Zeta

**Functional equation:**
$$\boxed{\zeta(s) = 2^s \pi^{s-1} \sin\left(\frac{\pi s}{2}\right) \Gamma(1-s) \zeta(1-s)}$$

This extends $\zeta$ to all $\mathbb{C}$ except $s = 1$ (simple pole).

**Trivial zeros:** $\zeta(-2n) = 0$ for $n = 1, 2, 3, \ldots$

**Special values:**
- $\zeta(2) = \pi^2/6$
- $\zeta(4) = \pi^4/90$
- $\zeta(-1) = -1/12$ (regularized sum of integers!)

### 7. Physics Applications

**Casimir Effect:**
The zero-point energy sum $\sum_{n=1}^\infty n$ is "regularized" using:
$$\zeta(-1) = -\frac{1}{12}$$

This gives the Casimir force between conducting plates.

**Partition Functions:**
For bosonic systems: $\ln Z \sim \zeta(3)$ terms appear.

**Black-body Radiation:**
Stefan-Boltzmann law involves $\zeta(4) = \pi^4/90$.

---

## Worked Examples

### Example 1: Computing $\Gamma(-3/2)$

**Solution:**
Using $\Gamma(z) = \Gamma(z+1)/z$:
$$\Gamma(-1/2) = \frac{\Gamma(1/2)}{-1/2} = -2\sqrt{\pi}$$
$$\Gamma(-3/2) = \frac{\Gamma(-1/2)}{-3/2} = \frac{-2\sqrt{\pi}}{-3/2} = \frac{4\sqrt{\pi}}{3}$$

### Example 2: Evaluating $\int_0^\infty x^3 e^{-x^2} dx$

**Solution:**
Substitute $u = x^2$: $du = 2x dx$
$$\int_0^\infty x^3 e^{-x^2} dx = \frac{1}{2}\int_0^\infty u e^{-u} du = \frac{1}{2}\Gamma(2) = \frac{1}{2}$$

### Example 3: $\zeta(2)$ via Contour Integration

**Problem:** Show $\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}$.

**Solution:**
Consider $f(z) = \pi\cot(\pi z)/z^2$ and integrate around a large rectangle.

Residues at $z = n \neq 0$: $\frac{1}{n^2}$
Residue at $z = 0$: $-\pi^2/3$

Sum of residues = 0 (by Cauchy), so:
$$2\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{3}$$
$$\zeta(2) = \frac{\pi^2}{6}$$

---

## Practice Problems

**P1.** Show that $\Gamma(n+1/2) = \frac{(2n)!}{4^n n!}\sqrt{\pi}$.

**P2.** Evaluate $\int_0^\infty \frac{x^{a-1}}{1+x} dx$ using Gamma functions.

**P3.** Use the reflection formula to show $|\Gamma(iy)|^2 = \frac{\pi}{y\sinh(\pi y)}$ for real $y$.

**P4.** Verify $\zeta(0) = -1/2$ from the functional equation.

---

## Computational Lab

```python
import numpy as np
from scipy.special import gamma, zeta
import matplotlib.pyplot as plt

# Gamma function visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Real axis
x = np.linspace(-4.5, 4.5, 1000)
# Avoid poles
x = x[np.abs(x - np.round(x)) > 0.05]
y = gamma(x)

axes[0, 0].plot(x, y.real, 'b-', linewidth=2)
axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 0].set_ylim(-10, 10)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Γ(x)')
axes[0, 0].set_title('Gamma Function on Real Axis')
axes[0, 0].grid(True, alpha=0.3)

# |Γ| in complex plane
x_c = np.linspace(-4, 4, 200)
y_c = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_c, y_c)
Z = X + 1j * Y

G = gamma(Z)
log_abs_G = np.log10(np.abs(G) + 1e-10)
log_abs_G = np.clip(log_abs_G, -2, 2)

im = axes[0, 1].contourf(X, Y, log_abs_G, levels=50, cmap='viridis')
plt.colorbar(im, ax=axes[0, 1], label='log₁₀|Γ(z)|')
axes[0, 1].set_xlabel('Re(z)')
axes[0, 1].set_ylabel('Im(z)')
axes[0, 1].set_title('|Γ(z)| in Complex Plane')

# Zeta function
s_real = np.linspace(-10, 10, 500)
s_real = s_real[np.abs(s_real - 1) > 0.1]
zeta_vals = zeta(s_real)

axes[1, 0].plot(s_real, zeta_vals, 'r-', linewidth=2)
axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].axvline(x=1, color='g', linestyle='--', label='Pole at s=1')
axes[1, 0].set_ylim(-5, 5)
axes[1, 0].set_xlabel('s')
axes[1, 0].set_ylabel('ζ(s)')
axes[1, 0].set_title('Riemann Zeta Function')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Special values
special_vals = {
    'ζ(2)': (zeta(2), np.pi**2/6),
    'ζ(4)': (zeta(4), np.pi**4/90),
    'ζ(-1)': (zeta(-1), -1/12),
    'ζ(0)': (zeta(0), -1/2),
    'Γ(1/2)': (gamma(0.5), np.sqrt(np.pi)),
    'Γ(3/2)': (gamma(1.5), np.sqrt(np.pi)/2),
}

table_text = "Special Values:\n" + "-"*35 + "\n"
for name, (computed, exact) in special_vals.items():
    table_text += f"{name:10} = {computed:10.6f} ({exact:.6f})\n"

axes[1, 1].text(0.1, 0.5, table_text, fontsize=12, family='monospace',
               verticalalignment='center', transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')
axes[1, 1].set_title('Verification of Special Values')

plt.tight_layout()
plt.savefig('special_functions.png', dpi=150, bbox_inches='tight')
plt.show()

print("Special Function Verifications:")
print("="*40)
print(f"ζ(2) = π²/6 = {np.pi**2/6:.10f}")
print(f"Computed:    {zeta(2):.10f}")
print(f"\nζ(-1) = -1/12 = {-1/12:.10f}")
print(f"Computed:      {zeta(-1):.10f}")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\Gamma(z+1) = z\Gamma(z)$ | Functional equation |
| $\Gamma(z)\Gamma(1-z) = \pi/\sin(\pi z)$ | Reflection formula |
| $B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)$ | Beta-Gamma relation |
| $\zeta(2) = \pi^2/6$ | Basel problem |
| $\zeta(-1) = -1/12$ | Regularized sum |

### Main Takeaways

1. **Gamma function** generalizes factorials to complex plane
2. **Analytic continuation** defines functions beyond original domain
3. **Zeta function** encodes prime number information
4. **Regularization** via zeta gives finite physics results
5. These functions are **ubiquitous** in quantum field theory

---

## Preview: Day 195

Tomorrow: **Comprehensive Computational Lab**
- Integration of all Month 7 techniques
- Physics problem solving
- Numerical methods

---

*"The Gamma function is perhaps the most fundamental of all special functions."*
— E.T. Whittaker
