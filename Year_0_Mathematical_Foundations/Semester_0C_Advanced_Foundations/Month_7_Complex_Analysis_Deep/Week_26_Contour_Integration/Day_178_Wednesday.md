# Day 178: Cauchy's Integral Formula

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Cauchy's Integral Formula |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Derivatives & Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 178, you will be able to:

1. State and prove Cauchy's integral formula
2. Derive the formula for derivatives of analytic functions
3. Apply the formulas to compute values and derivatives
4. Understand the maximum modulus principle
5. Prove Liouville's theorem and the fundamental theorem of algebra
6. Connect to quantum mechanical propagators

---

## Core Content

### 1. Cauchy's Integral Formula

#### The Central Result

**Theorem (Cauchy's Integral Formula):**
Let $f$ be analytic in a simply connected domain $D$ containing a simple closed contour $C$ and its interior. For any point $z_0$ inside $C$:

$$\boxed{f(z_0) = \frac{1}{2\pi i} \oint_C \frac{f(z)}{z - z_0} \, dz}$$

This is remarkable: **the value of $f$ at any interior point is completely determined by its values on the boundary!**

#### Proof

Consider the function $g(z) = \frac{f(z) - f(z_0)}{z - z_0}$.

Since $f$ is analytic at $z_0$, we have:
$$\lim_{z \to z_0} g(z) = \lim_{z \to z_0} \frac{f(z) - f(z_0)}{z - z_0} = f'(z_0)$$

So $g$ has a removable singularity at $z_0$ and can be extended to an analytic function in all of $D$.

By Cauchy's theorem:
$$0 = \oint_C g(z) \, dz = \oint_C \frac{f(z) - f(z_0)}{z - z_0} \, dz = \oint_C \frac{f(z)}{z - z_0} \, dz - f(z_0) \oint_C \frac{dz}{z - z_0}$$

Since $\oint_C \frac{dz}{z - z_0} = 2\pi i$ (winding number = 1):
$$\oint_C \frac{f(z)}{z - z_0} \, dz = 2\pi i \cdot f(z_0)$$

Dividing by $2\pi i$ gives the result. ∎

### 2. Formula for Derivatives

#### First Derivative

Differentiating Cauchy's formula with respect to $z_0$:

$$f'(z_0) = \frac{d}{dz_0}\left[\frac{1}{2\pi i} \oint_C \frac{f(z)}{z - z_0} \, dz\right]$$

We can differentiate under the integral sign:
$$\boxed{f'(z_0) = \frac{1}{2\pi i} \oint_C \frac{f(z)}{(z - z_0)^2} \, dz}$$

#### Higher Derivatives

Continuing this process:

**Theorem (Cauchy's Formula for Derivatives):**
$$\boxed{f^{(n)}(z_0) = \frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z - z_0)^{n+1}} \, dz}$$

**Consequence:** Every analytic function is infinitely differentiable!

This is a major difference from real analysis, where differentiability doesn't imply infinite differentiability.

### 3. Morera's Theorem (Converse of Cauchy)

**Theorem (Morera):**
If $f$ is continuous in a domain $D$ and $\oint_C f(z) \, dz = 0$ for every closed contour $C$ in $D$, then $f$ is analytic in $D$.

**Proof sketch:** Define $F(z) = \int_{z_0}^z f(\zeta) \, d\zeta$. This is well-defined (path-independent), and one can show $F' = f$. Since $F$ is analytic and $f = F'$, $f$ is also analytic.

### 4. Cauchy's Inequality

**Theorem (Cauchy's Inequality):**
If $f$ is analytic in $|z - z_0| \leq R$ and $|f(z)| \leq M$ on $|z - z_0| = R$, then:
$$\boxed{|f^{(n)}(z_0)| \leq \frac{n! M}{R^n}}$$

**Proof:** From the derivative formula on circle $C: |z - z_0| = R$:
$$|f^{(n)}(z_0)| = \left|\frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z-z_0)^{n+1}} \, dz\right| \leq \frac{n!}{2\pi} \cdot \frac{M}{R^{n+1}} \cdot 2\pi R = \frac{n!M}{R^n}$$

### 5. Liouville's Theorem

**Theorem (Liouville):**
A bounded entire function is constant.

**Proof:** Let $f$ be entire with $|f(z)| \leq M$ for all $z \in \mathbb{C}$.

By Cauchy's inequality for $n = 1$ with any $R$:
$$|f'(z_0)| \leq \frac{M}{R}$$

Since $R$ can be arbitrarily large, $|f'(z_0)| = 0$ for all $z_0$.

Therefore $f' \equiv 0$, so $f$ is constant. ∎

### 6. Fundamental Theorem of Algebra

**Theorem:** Every non-constant polynomial $P(z)$ has at least one root in $\mathbb{C}$.

**Proof (using Liouville):** Suppose $P(z) \neq 0$ for all $z$.

Then $f(z) = 1/P(z)$ is entire.

For large $|z|$: $|P(z)| \sim |a_n z^n|$ (leading term dominates), so:
$$|f(z)| = \frac{1}{|P(z)|} \to 0 \text{ as } |z| \to \infty$$

Thus $f$ is bounded. By Liouville, $f$ is constant, hence $P$ is constant.

Contradiction! Therefore $P$ has a root. ∎

### 7. Maximum Modulus Principle

**Theorem (Maximum Modulus Principle):**
If $f$ is analytic and non-constant in a domain $D$, then $|f|$ has no maximum in $D$.

Equivalently: If $D$ is bounded and $f$ is analytic on $\bar{D}$ and continuous on $\partial D$, then:
$$\max_{\bar{D}} |f(z)| = \max_{\partial D} |f(z)|$$

**Proof idea:** By Cauchy's formula:
$$f(z_0) = \frac{1}{2\pi} \int_0^{2\pi} f(z_0 + re^{i\theta}) \, d\theta$$

So $f(z_0)$ is the **average** of $f$ on any circle. The modulus of an average is at most the average of the moduli (by the triangle inequality). Equality holds only if $f$ is constant.

---

## Quantum Mechanics Connection

### Propagators and Green's Functions

The Cauchy integral formula is structurally identical to the expression for quantum propagators.

**Free particle propagator:**
$$K(x, t; x', 0) = \langle x | e^{-iHt/\hbar} | x' \rangle$$

Using the resolution of identity and energy representation:
$$K = \int \frac{dE}{2\pi\hbar} \, e^{-iEt/\hbar} G(E)$$

where $G(E)$ is the **Green's function** (resolvent).

**Analogy:**
| Cauchy Formula | Quantum Propagator |
|----------------|-------------------|
| $f(z_0) = \frac{1}{2\pi i}\oint \frac{f(z)}{z-z_0} dz$ | $K = \frac{1}{2\pi i}\oint \frac{G(E)}{E-E_0} dE$ |
| Contour around $z_0$ | Contour in energy plane |
| Analytic function $f$ | Analytic continuation of $G$ |

### Spectral Decomposition

For a Hamiltonian with discrete spectrum:
$$G(E) = \sum_n \frac{|n\rangle\langle n|}{E - E_n}$$

This has **poles** at the energy eigenvalues $E_n$.

Using Cauchy's formula:
$$\oint_{C_n} G(E) \, dE = 2\pi i |n\rangle\langle n|$$

The contour integral extracts the projection operator!

### Analyticity and Causality

The retarded Green's function $G^R(E)$ is analytic in the **upper half-plane** (Im$(E) > 0$).

This analyticity property is equivalent to **causality** — effects cannot precede causes.

Kramers-Kronig relations (which follow from Cauchy's theorem) connect:
- Real part of response function (dispersion)
- Imaginary part (absorption)

---

## Worked Examples

### Example 1: Basic Application of Cauchy's Formula

**Problem:** Compute $\oint_{|z|=2} \frac{e^z}{z - 1} \, dz$.

**Solution:**

The function $f(z) = e^z$ is entire. The singularity of the integrand is at $z = 1$, which is inside $|z| = 2$.

By Cauchy's integral formula with $f(z) = e^z$ and $z_0 = 1$:
$$\oint_{|z|=2} \frac{e^z}{z - 1} \, dz = 2\pi i \cdot f(1) = 2\pi i \cdot e^1 = \boxed{2\pi i e}$$

### Example 2: Computing a Derivative

**Problem:** Find $f''(0)$ where $f(z) = \frac{\sin z}{z^2 + 1}$.

**Solution:**

**Method 1: Direct differentiation** (complicated)

**Method 2: Cauchy's derivative formula**

By the formula:
$$f''(0) = \frac{2!}{2\pi i} \oint_C \frac{f(z)}{z^3} \, dz$$

where $C$ is a small circle around the origin not enclosing $\pm i$.

We need to evaluate $\oint_{|z|=1/2} \frac{\sin z}{z^3(z^2+1)} \, dz$.

Near $z = 0$: $\sin z = z - z^3/6 + \cdots$ and $\frac{1}{z^2+1} = 1 - z^2 + \cdots$

So:
$$\frac{\sin z}{z^3(z^2+1)} = \frac{(z - z^3/6 + \cdots)(1 - z^2 + \cdots)}{z^3} = \frac{z - z^3 - z^3/6 + \cdots}{z^3}$$
$$= \frac{1}{z^2} - \frac{7}{6} + O(z^2)$$

The coefficient of $1/z$ is 0, so $\oint = 0$, giving $f''(0) = 0$.

Wait, let me recalculate more carefully:
$$\frac{\sin z}{z^2+1} = \frac{z - z^3/6 + z^5/120 - \cdots}{1 + z^2}$$
$$= (z - z^3/6 + \cdots)(1 - z^2 + z^4 - \cdots)$$
$$= z - z^3 - z^3/6 + z^5 + \cdots = z - \frac{7z^3}{6} + \cdots$$

So $f(z) = z - \frac{7z^3}{6} + O(z^5)$

Therefore: $f'(z) = 1 - \frac{7z^2}{2} + \cdots$ and $f''(z) = -7z + \cdots$

Thus $f''(0) = 0$ ✓

### Example 3: Multiple Singularities

**Problem:** Evaluate $\oint_{|z|=3} \frac{z^2}{(z-1)(z-2)} \, dz$.

**Solution:**

Both $z = 1$ and $z = 2$ are inside $|z| = 3$.

Use partial fractions:
$$\frac{z^2}{(z-1)(z-2)} = 1 + \frac{A}{z-1} + \frac{B}{z-2}$$

Multiplying out: $z^2 = (z-1)(z-2) + A(z-2) + B(z-1)$

At $z = 1$: $1 = A(-1)$, so $A = -1$
At $z = 2$: $4 = B(1)$, so $B = 4$

Therefore:
$$\frac{z^2}{(z-1)(z-2)} = 1 - \frac{1}{z-1} + \frac{4}{z-2}$$

And:
$$\oint_{|z|=3} \frac{z^2}{(z-1)(z-2)} \, dz = \oint 1 \, dz - \oint \frac{dz}{z-1} + 4\oint \frac{dz}{z-2}$$
$$= 0 - 2\pi i + 4 \cdot 2\pi i = 6\pi i$$

### Example 4: Derivatives via Contour Integrals

**Problem:** Compute $\oint_{|z|=1} \frac{e^z}{z^4} \, dz$.

**Solution:**

This integral equals $\frac{2\pi i}{3!} f^{(3)}(0)$ where $f(z) = e^z$.

Since $f^{(n)}(z) = e^z$ for all $n$, we have $f^{(3)}(0) = 1$.

$$\oint_{|z|=1} \frac{e^z}{z^4} \, dz = \frac{2\pi i}{6} \cdot 1 = \frac{\pi i}{3}$$

---

## Practice Problems

### Problem Set A: Cauchy's Formula Applications

**A1.** Evaluate:
(a) $\oint_{|z|=1} \frac{\cos z}{z} \, dz$
(b) $\oint_{|z|=2} \frac{z^3}{z - i} \, dz$
(c) $\oint_{|z|=1} \frac{e^{2z}}{z^2} \, dz$

**A2.** Using Cauchy's derivative formula, find $f^{(4)}(0)$ for $f(z) = \cos z$.

**A3.** Compute $\oint_{|z|=3} \frac{\sin z}{z(z-\pi)} \, dz$.

### Problem Set B: Theoretical Applications

**B1.** Use Liouville's theorem to prove that $e^z$ is not bounded on $\mathbb{C}$.

**B2.** Show that if $f$ is entire and $|f(z)| \leq |z|^n$ for some $n$ and all large $|z|$, then $f$ is a polynomial of degree at most $n$.

**B3.** Prove that if $f$ is analytic and $|f(z)| = 1$ for $|z| = 1$, then $f(z) = cz^n$ for some $|c| = 1$ and integer $n \geq 0$.

### Problem Set C: Maximum Modulus

**C1.** Find the maximum of $|z^2 - 2z + 3|$ on the closed disk $|z| \leq 1$.

**C2.** If $f$ is analytic on $|z| \leq 1$ with $|f(z)| \leq 1$ on $|z| = 1$ and $f(0) = 0$, prove that $|f(z)| \leq |z|$ for $|z| \leq 1$. (Schwarz Lemma)

**C3.** Show that $\text{Re}(f)$ cannot have a local maximum inside a domain where $f$ is analytic and non-constant.

---

## Solutions to Selected Problems

### Solution A1

**(a)** $f(z) = \cos z$ is entire, singularity at $z = 0$.
$$\oint_{|z|=1} \frac{\cos z}{z} \, dz = 2\pi i \cdot \cos(0) = 2\pi i$$

**(b)** $f(z) = z^3$ is entire, singularity at $z = i$.
$$\oint_{|z|=2} \frac{z^3}{z-i} \, dz = 2\pi i \cdot i^3 = 2\pi i \cdot (-i) = 2\pi$$

**(c)** This is $\frac{2\pi i}{1!} f'(0)$ where $f(z) = e^{2z}$.
Since $f'(z) = 2e^{2z}$, $f'(0) = 2$.
$$\oint_{|z|=1} \frac{e^{2z}}{z^2} \, dz = 2\pi i \cdot 2 = 4\pi i$$

### Solution A2

$$f^{(4)}(0) = \frac{4!}{2\pi i} \oint_{|z|=1} \frac{\cos z}{z^5} \, dz$$

Alternatively: $\cos z = 1 - z^2/2! + z^4/4! - \cdots$

So $f^{(4)}(z) = \cos z$, and $f^{(4)}(0) = \cos(0) = 1$.

### Solution B1

If $e^z$ were bounded, say $|e^z| \leq M$ for all $z$, then by Liouville's theorem $e^z$ would be constant.

But $e^0 = 1$ and $e^1 = e \neq 1$, so $e^z$ is not constant.

Contradiction! Therefore $e^z$ is unbounded.

(In fact, $|e^z| = e^{\text{Re}(z)} \to \infty$ as $\text{Re}(z) \to \infty$.)

---

## Computational Lab

### Lab 1: Verifying Cauchy's Integral Formula

```python
import numpy as np
import matplotlib.pyplot as plt

def contour_integral(f, z_path, t_range, n=10000):
    """Compute contour integral numerically."""
    t = np.linspace(t_range[0], t_range[1], n)
    dt = t[1] - t[0]
    z = z_path(t)
    dz = np.gradient(z, dt)
    return np.trapz(f(z) * dz, t)

def cauchy_formula_value(f, z0, radius=1, n=10000):
    """Compute f(z0) using Cauchy's integral formula."""
    integrand = lambda z: f(z) / (z - z0)
    path = lambda t: z0 + radius * np.exp(1j * t)
    integral = contour_integral(integrand, path, (0, 2*np.pi), n)
    return integral / (2 * np.pi * 1j)

# Test with various functions
print("Verifying Cauchy's Integral Formula")
print("=" * 60)

test_cases = [
    (lambda z: np.exp(z), "e^z", 0, np.exp(0)),
    (lambda z: np.exp(z), "e^z", 1, np.exp(1)),
    (lambda z: np.sin(z), "sin(z)", 0.5, np.sin(0.5)),
    (lambda z: z**3 - 2*z, "z³-2z", 1j, (1j)**3 - 2*1j),
    (lambda z: 1/(z+2), "1/(z+2)", 0, 0.5),
]

for f, name, z0, expected in test_cases:
    computed = cauchy_formula_value(f, z0, radius=0.5)
    error = abs(computed - expected)
    print(f"f(z) = {name:10}, z₀ = {z0:5}")
    print(f"  Computed: {computed:20.10f}")
    print(f"  Expected: {expected:20.10f}")
    print(f"  Error:    {error:.2e}\n")
```

### Lab 2: Computing Derivatives via Contour Integrals

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial

def cauchy_derivative(f, z0, n, radius=0.5, num_points=50000):
    """
    Compute f^(n)(z0) using Cauchy's derivative formula:
    f^(n)(z0) = n! / (2πi) ∮ f(z)/(z-z0)^(n+1) dz
    """
    t = np.linspace(0, 2*np.pi, num_points)
    dt = t[1] - t[0]
    z = z0 + radius * np.exp(1j * t)
    dz = 1j * radius * np.exp(1j * t)

    integrand = f(z) / (z - z0)**(n+1) * dz
    integral = np.trapz(integrand, t)

    return factorial(n, exact=True) * integral / (2 * np.pi * 1j)

# Test: derivatives of e^z (all derivatives equal e^z)
print("Derivatives of e^z at z=0 using Cauchy's formula")
print("-" * 50)

f = lambda z: np.exp(z)
z0 = 0

for n in range(6):
    computed = cauchy_derivative(f, z0, n)
    expected = np.exp(z0)  # d^n/dz^n (e^z) = e^z
    print(f"f^({n})(0): Computed = {computed.real:10.6f}, Expected = {expected:.6f}")

# Test: derivatives of sin(z) at z=0
print("\nDerivatives of sin(z) at z=0")
print("-" * 50)

f = lambda z: np.sin(z)
# sin(z) at z=0: 0, 1, 0, -1, 0, 1, ...
expected_vals = [0, 1, 0, -1, 0, 1]

for n in range(6):
    computed = cauchy_derivative(f, z0, n)
    print(f"f^({n})(0): Computed = {computed.real:10.6f}, Expected = {expected_vals[n]}")

# Visualization: contour and integrand
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Show the contour
theta = np.linspace(0, 2*np.pi, 100)
z_contour = 0.5 * np.exp(1j * theta)
axes[0].plot(z_contour.real, z_contour.imag, 'b-', linewidth=2)
axes[0].plot([0], [0], 'r*', markersize=15, label='z₀ = 0')
axes[0].set_xlabel('Re(z)')
axes[0].set_ylabel('Im(z)')
axes[0].set_title('Integration Contour')
axes[0].legend()
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

# Show integrand for f''(0) of e^z
n = 2
z = 0.5 * np.exp(1j * theta)
integrand = np.exp(z) / z**(n+1)
axes[1].plot(theta, np.abs(integrand), 'g-', linewidth=2, label='|f(z)/z³|')
axes[1].plot(theta, integrand.real, 'b-', linewidth=1, alpha=0.7, label='Re')
axes[1].plot(theta, integrand.imag, 'r-', linewidth=1, alpha=0.7, label='Im')
axes[1].set_xlabel('θ (parametrization)')
axes[1].set_ylabel('Integrand value')
axes[1].set_title("Integrand for f''(0) where f(z) = e^z")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cauchy_derivatives.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 3: Maximum Modulus Principle Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_max_modulus(f, title, xlim=(-2, 2), ylim=(-2, 2)):
    """Visualize |f(z)| and show that max occurs on boundary."""
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Compute |f(z)|, handling potential infinities
    with np.errstate(divide='ignore', invalid='ignore'):
        W = np.abs(f(Z))
        W = np.clip(W, 0, 10)  # Clip for visualization

    fig = plt.figure(figsize=(15, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, W, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Re(z)')
    ax1.set_ylabel('Im(z)')
    ax1.set_zlabel('|f(z)|')
    ax1.set_title(f'|{title}|')

    # Contour plot with unit circle
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(X, Y, W, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='|f(z)|')

    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2, label='|z|=1')
    ax2.set_xlabel('Re(z)')
    ax2.set_ylabel('Im(z)')
    ax2.set_title(f'Contours of |{title}|')
    ax2.legend()
    ax2.axis('equal')

    # Values on unit circle
    ax3 = fig.add_subplot(133)
    z_circle = np.exp(1j * theta)
    f_circle = np.abs(f(z_circle))
    ax3.plot(theta * 180 / np.pi, f_circle, 'b-', linewidth=2)
    ax3.axhline(y=np.max(f_circle), color='r', linestyle='--',
                label=f'Max = {np.max(f_circle):.3f}')
    ax3.set_xlabel('θ (degrees)')
    ax3.set_ylabel('|f(e^{iθ})|')
    ax3.set_title('|f| on unit circle')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Example 1: f(z) = z^2 - 2z + 3
f1 = lambda z: z**2 - 2*z + 3
fig1 = visualize_max_modulus(f1, "z² - 2z + 3")
plt.savefig('max_modulus_poly.png', dpi=150, bbox_inches='tight')

# Example 2: f(z) = e^z
f2 = lambda z: np.exp(z)
fig2 = visualize_max_modulus(f2, "e^z")
plt.savefig('max_modulus_exp.png', dpi=150, bbox_inches='tight')

plt.show()

# Verify maximum on boundary
print("Maximum Modulus Verification")
print("=" * 50)

# Sample interior points and boundary
theta = np.linspace(0, 2*np.pi, 1000)
z_boundary = np.exp(1j * theta)

# Random interior points
np.random.seed(42)
r = np.random.uniform(0, 0.99, 1000)
phi = np.random.uniform(0, 2*np.pi, 1000)
z_interior = r * np.exp(1j * phi)

for f, name in [(f1, "z² - 2z + 3"), (f2, "e^z")]:
    max_boundary = np.max(np.abs(f(z_boundary)))
    max_interior = np.max(np.abs(f(z_interior)))
    print(f"\nf(z) = {name}:")
    print(f"  Max |f| on boundary:  {max_boundary:.6f}")
    print(f"  Max |f| in interior:  {max_interior:.6f}")
    print(f"  Max occurs on boundary: {max_boundary >= max_interior}")
```

### Lab 4: Liouville's Theorem Illustration

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_liouville():
    """
    Demonstrate why bounded entire functions must be constant.
    Show how the derivative bound from Cauchy's inequality
    forces f' = 0 as R → ∞.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # For a hypothetical bounded entire function with |f| ≤ M
    M = 1  # Bound

    # Cauchy inequality: |f'(z0)| ≤ M/R
    R_values = np.linspace(0.1, 100, 1000)
    derivative_bound = M / R_values

    axes[0].plot(R_values, derivative_bound, 'b-', linewidth=2)
    axes[0].fill_between(R_values, 0, derivative_bound, alpha=0.3)
    axes[0].set_xlabel('Radius R')
    axes[0].set_ylabel("|f'(z₀)| bound")
    axes[0].set_title("Cauchy's Inequality: |f'(z₀)| ≤ M/R")
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate('As R → ∞, bound → 0\nSo f\' ≡ 0',
                    xy=(50, 0.02), fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat'))

    # Show why e^z is unbounded
    # |e^z| = e^x where z = x + iy
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # |e^z| = e^x (independent of y!)
    modulus = np.exp(X)

    contour = axes[1].contourf(X, Y, modulus, levels=50, cmap='hot')
    plt.colorbar(contour, ax=axes[1], label='|e^z|')
    axes[1].set_xlabel('Re(z)')
    axes[1].set_ylabel('Im(z)')
    axes[1].set_title('|e^z| = e^{Re(z)} (unbounded as Re(z) → ∞)')

    # Draw circles of increasing radius
    for R in [1, 2, 3]:
        theta = np.linspace(0, 2*np.pi, 100)
        axes[1].plot(R*np.cos(theta), R*np.sin(theta), 'w--', linewidth=1,
                    label=f'|z|={R}' if R == 1 else '')

    axes[1].axis('equal')
    axes[1].set_xlim(-3.5, 3.5)
    axes[1].set_ylim(-3.5, 3.5)

    plt.tight_layout()
    plt.savefig('liouville_theorem.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Show that |e^z| is unbounded on any circle
    print("\nMaximum of |e^z| on circles of radius R:")
    print("-" * 40)
    for R in [1, 2, 5, 10, 100]:
        # Max occurs at z = R (on real axis)
        max_val = np.exp(R)
        print(f"R = {R:5}: max|e^z| = e^{R} = {max_val:.2e}")

demonstrate_liouville()
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $f(z_0) = \frac{1}{2\pi i}\oint_C \frac{f(z)}{z-z_0} dz$ | Cauchy's integral formula |
| $f^{(n)}(z_0) = \frac{n!}{2\pi i}\oint_C \frac{f(z)}{(z-z_0)^{n+1}} dz$ | Derivatives formula |
| $\|f^{(n)}(z_0)\| \leq \frac{n!M}{R^n}$ | Cauchy's inequality |
| Bounded entire $\Rightarrow$ constant | Liouville's theorem |

### Main Takeaways

1. **Cauchy's integral formula** expresses interior values in terms of boundary values.

2. **All derivatives exist** for analytic functions — computed via contour integrals.

3. **Cauchy's inequality** bounds derivatives in terms of function bounds.

4. **Liouville's theorem** proves bounded entire functions are constant.

5. **The fundamental theorem of algebra** follows from Liouville's theorem.

6. **In quantum mechanics**, Cauchy's formula appears in spectral decomposition and propagator theory.

---

## Daily Checklist

- [ ] I can state and prove Cauchy's integral formula
- [ ] I can compute function values via contour integrals
- [ ] I can use the derivative formula
- [ ] I understand Liouville's theorem and its proof
- [ ] I can apply the maximum modulus principle
- [ ] I see the connection to quantum propagators

---

## Preview: Day 179

Tomorrow we apply contour integration to **evaluate real integrals** — one of the most powerful applications of complex analysis:

$$\int_0^\infty \frac{\sin x}{x} \, dx = \frac{\pi}{2}$$

$$\int_{-\infty}^\infty \frac{dx}{1 + x^2} = \pi$$

These and many other "impossible" real integrals become straightforward with complex methods!

---

*"Between two truths of the real domain, the easiest path passes through the complex domain."*
— Paul Painlevé
