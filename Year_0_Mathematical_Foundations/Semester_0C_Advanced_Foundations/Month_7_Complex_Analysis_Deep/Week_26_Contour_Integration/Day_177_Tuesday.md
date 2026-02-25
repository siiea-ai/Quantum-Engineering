# Day 177: Cauchy's Integral Theorem

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Cauchy's Theorem & Proofs |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications & Problem Solving |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 177, you will be able to:

1. State and prove Cauchy's integral theorem for simply connected domains
2. Apply Cauchy's theorem to evaluate contour integrals
3. Understand the role of simple connectivity
4. Use deformation of contours to evaluate integrals
5. Connect Cauchy's theorem to Green's theorem
6. Relate to path independence in quantum mechanics

---

## Core Content

### 1. Statement of Cauchy's Integral Theorem

#### The Fundamental Result

**Theorem (Cauchy's Integral Theorem — Basic Version):**
Let $f$ be analytic in a simply connected domain $D$, and let $C$ be any closed contour in $D$. Then:
$$\boxed{\oint_C f(z) \, dz = 0}$$

This is one of the most important theorems in all of mathematics.

#### What Does It Mean?

1. **For any closed path**, the integral of an analytic function is zero
2. **Integrals are path-independent**: $\int_{C_1} f \, dz = \int_{C_2} f \, dz$ for paths with same endpoints
3. **Antiderivatives exist**: Every analytic function in a simply connected domain has an antiderivative

### 2. Simple Connectivity

**Definition:** A domain $D$ is *simply connected* if every closed curve in $D$ can be continuously shrunk to a point without leaving $D$.

**Examples:**

| Simply Connected | Not Simply Connected |
|-----------------|---------------------|
| Open disk $\|z\| < 1$ | Punctured disk $0 < \|z\| < 1$ |
| Half-plane $\text{Re}(z) > 0$ | Annulus $1 < \|z\| < 2$ |
| The entire plane $\mathbb{C}$ | $\mathbb{C} \setminus \{0\}$ |

**Key Point:** Simple connectivity ensures no "holes" that could make the integral non-zero.

### 3. Proof of Cauchy's Theorem

#### Proof via Green's Theorem (Classical Approach)

**Assumption:** $f'$ exists and is continuous (Goursat removed this requirement).

Let $f(z) = u(x,y) + iv(x,y)$ and $C$ enclose region $R$.

From Day 176:
$$\oint_C f(z) \, dz = \oint_C (u\,dx - v\,dy) + i\oint_C (v\,dx + u\,dy)$$

Apply Green's theorem to each integral:
$$\oint_C (P\,dx + Q\,dy) = \iint_R \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$$

**Real part:**
$$\oint_C (u\,dx - v\,dy) = \iint_R \left(-\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}\right) dA$$

By Cauchy-Riemann: $-v_x = u_y$, so:
$$= \iint_R (-u_y - u_y) dA = -2\iint_R u_y \, dA$$

Wait—let's be more careful. Using $u_x = v_y$ and $u_y = -v_x$:
$$-v_x - u_y = -(-u_y) - u_y = u_y - u_y = 0$$

**Imaginary part:**
$$\oint_C (v\,dx + u\,dy) = \iint_R \left(\frac{\partial u}{\partial x} - \frac{\partial v}{\partial y}\right) dA$$

By Cauchy-Riemann: $u_x = v_y$, so:
$$= \iint_R (v_y - v_y) dA = 0$$

Therefore: $\oint_C f(z) \, dz = 0 + i \cdot 0 = 0$ ∎

#### Goursat's Contribution

**Goursat's Theorem:** Cauchy's theorem holds if $f$ is merely *differentiable* in $D$, without assuming continuity of $f'$.

This is remarkable: complex differentiability alone implies the integral is zero.

The proof uses a subdivision argument (divide the region into smaller and smaller triangles).

### 4. Consequences of Cauchy's Theorem

#### Path Independence

**Corollary 1:** If $f$ is analytic in a simply connected domain $D$, and $C_1, C_2$ are two paths from $z_1$ to $z_2$ in $D$, then:
$$\int_{C_1} f(z) \, dz = \int_{C_2} f(z) \, dz$$

**Proof:** $C_1 - C_2$ (i.e., $C_1$ followed by $-C_2$) is a closed contour:
$$0 = \oint_{C_1 - C_2} f \, dz = \int_{C_1} f \, dz - \int_{C_2} f \, dz$$

#### Existence of Antiderivatives

**Corollary 2:** If $f$ is analytic in a simply connected domain $D$, then $f$ has an antiderivative $F$ in $D$.

**Proof:** Fix $z_0 \in D$ and define:
$$F(z) = \int_{z_0}^z f(\zeta) \, d\zeta$$

This is well-defined (path-independent). One can show $F'(z) = f(z)$.

#### Indefinite Integrals

We can write:
$$\int f(z) \, dz = F(z) + C$$

where $F' = f$, just like in real calculus!

### 5. Deformation of Contours

#### The Deformation Principle

**Theorem:** If $f$ is analytic in a region containing two closed contours $C_1$ and $C_2$, and $C_1$ can be continuously deformed to $C_2$ without crossing any singularities of $f$, then:
$$\oint_{C_1} f(z) \, dz = \oint_{C_2} f(z) \, dz$$

This is incredibly powerful: we can replace complicated contours with simpler ones!

#### Example: Deforming to a Circle

**Problem:** Compute $\oint_C \frac{dz}{z}$ where $C$ is the square with vertices $\pm 1 \pm i$.

**Solution:** The only singularity is at $z = 0$, which is inside the square. Deform $C$ to the circle $|z| = \epsilon$:

$$\oint_C \frac{dz}{z} = \oint_{|z|=\epsilon} \frac{dz}{z} = 2\pi i$$

The shape of $C$ doesn't matter—only that it winds once around the origin.

### 6. Multiply Connected Domains

#### Handling Holes

For domains with "holes" (not simply connected), we need a generalized theorem.

**Theorem (Cauchy's Theorem for Multiply Connected Domains):**

Let $D$ be a domain bounded by outer contour $C_0$ and inner contours $C_1, \ldots, C_n$. If $f$ is analytic in $D$ and on its boundary:

$$\oint_{C_0} f(z) \, dz = \sum_{k=1}^n \oint_{C_k} f(z) \, dz$$

where all contours are traversed counterclockwise.

**Intuition:** The integral around the outer boundary equals the sum of integrals around the holes.

#### Example: Annulus

In the annulus $1 < |z| < 2$, for $f(z) = 1/z$:

$$\oint_{|z|=2} \frac{dz}{z} - \oint_{|z|=1} \frac{dz}{z} = 2\pi i - 2\pi i = 0$$

This equals the integral over the annular region (which contains no singularities of $1/z$).

### 7. The Index (Winding Number)

**Definition:** The *index* or *winding number* of a closed curve $C$ with respect to point $z_0$ not on $C$ is:
$$\boxed{n(C, z_0) = \frac{1}{2\pi i} \oint_C \frac{dz}{z - z_0}}$$

**Properties:**
- $n(C, z_0)$ is always an integer
- It counts how many times $C$ winds around $z_0$
- Counterclockwise = positive, clockwise = negative

**Example:**
- Circle $|z| = 1$ around $z_0 = 0$: $n = 1$
- Same circle around $z_0 = 2$: $n = 0$ (outside)
- Figure-eight around $z_0 = 0$: $n = 0$ (winds cancel)

---

## Quantum Mechanics Connection

### Topological Phases

Cauchy's theorem explains why quantum phases depend on topology, not geometry.

**Berry Phase:** When a quantum system evolves adiabatically around a closed loop in parameter space:
$$\gamma = i\oint_C \langle n | \nabla_\lambda | n \rangle \cdot d\lambda$$

If the integrand is "analytic" (no singularities), the Berry phase is zero. **Non-zero Berry phases arise from singularities (degeneracy points) enclosed by the path.**

### Aharonov-Bohm Effect Redux

The phase acquired by an electron encircling a magnetic flux $\Phi$:
$$\phi = \frac{e}{\hbar c} \oint \mathbf{A} \cdot d\mathbf{r} = \frac{e\Phi}{\hbar c}$$

By Cauchy's theorem for multiply connected domains:
- The integral depends only on the enclosed flux
- The exact path shape doesn't matter
- This is a topological effect!

### Path Integrals

In Feynman's formulation, the propagator is:
$$K(x_f, t_f; x_i, t_i) = \int \mathcal{D}[x] \, e^{iS[x]/\hbar}$$

Cauchy's theorem applied to complex time (Wick rotation) connects:
- Quantum mechanics (real time)
- Statistical mechanics (imaginary time)

This is why $\oint e^{-H\tau/\hbar} = $ traces in partition functions!

---

## Worked Examples

### Example 1: Verifying Cauchy's Theorem

**Problem:** Verify that $\oint_C e^z \, dz = 0$ where $C$ is the circle $|z| = 2$.

**Solution:**

**Method 1: Direct Computation**

Parametrize: $z(t) = 2e^{it}$, $0 \leq t \leq 2\pi$, so $dz = 2ie^{it} dt$

$$\oint_C e^z \, dz = \int_0^{2\pi} e^{2e^{it}} \cdot 2ie^{it} \, dt$$

Using $e^{2e^{it}} = e^{2\cos t + 2i\sin t} = e^{2\cos t}(\cos(2\sin t) + i\sin(2\sin t))$:

This is complicated, but we can show the integral is zero by symmetry or numerical verification.

**Method 2: Cauchy's Theorem**

$e^z$ is entire (analytic everywhere). The circle $|z| = 2$ lies in the simply connected domain $\mathbb{C}$.

By Cauchy's theorem: $\oint_C e^z \, dz = 0$ ✓

### Example 2: Non-Simply Connected Domain

**Problem:** Compute $\oint_C \frac{z}{z^2 - 1} \, dz$ where $C$ is:
(a) $|z| = 1/2$
(b) $|z| = 2$
(c) $|z - 1| = 1/2$

**Solution:**

The function $f(z) = \frac{z}{z^2-1} = \frac{z}{(z-1)(z+1)}$ has singularities at $z = \pm 1$.

**(a) $|z| = 1/2$:** Neither singularity is inside. Domain is simply connected.
$$\oint_{|z|=1/2} f(z) \, dz = 0$$

**(b) $|z| = 2$:** Both singularities are inside. We'll compute this tomorrow using residues.

**(c) $|z-1| = 1/2$:** Only $z = 1$ is inside. Near $z = 1$:
$$f(z) = \frac{z}{(z-1)(z+1)} = \frac{1}{z-1} \cdot \frac{z}{z+1}$$

The factor $\frac{z}{z+1}$ is analytic at $z = 1$ with value $\frac{1}{2}$.

By deformation, we'll see tomorrow that this integral is $2\pi i \cdot \frac{1}{2} = \pi i$.

### Example 3: Using Path Independence

**Problem:** Compute $\int_C \cos z \, dz$ where $C$ is any path from $0$ to $i$.

**Solution:**

Since $\cos z$ is entire, the integral is path-independent.

Antiderivative: $F(z) = \sin z$

$$\int_C \cos z \, dz = \sin(i) - \sin(0) = \sin(i)$$

Now, $\sin(i) = \frac{e^{i \cdot i} - e^{-i \cdot i}}{2i} = \frac{e^{-1} - e^1}{2i} = \frac{-2\sinh(1)}{2i} = \frac{\sinh(1)}{i} = -i\sinh(1)$

Actually: $\sin(iz) = i\sinh(z)$, so $\sin(i) = i\sinh(1) \approx 1.175i$

$$\int_C \cos z \, dz = i\sinh(1) \approx 1.175i$$

### Example 4: Deformation Principle

**Problem:** Show that $\oint_C \frac{e^z}{z} \, dz = 2\pi i$ for any simple closed contour $C$ enclosing the origin once counterclockwise.

**Solution:**

The function $\frac{e^z}{z}$ has only one singularity at $z = 0$.

By the deformation principle, deform $C$ to a small circle $|z| = \epsilon$:

$$\oint_C \frac{e^z}{z} \, dz = \oint_{|z|=\epsilon} \frac{e^z}{z} \, dz$$

For small $\epsilon$, $e^z \approx 1 + z + z^2/2 + \cdots$, so:
$$\frac{e^z}{z} \approx \frac{1}{z} + 1 + \frac{z}{2} + \cdots$$

Only the $1/z$ term contributes to the contour integral:
$$\oint_{|z|=\epsilon} \frac{e^z}{z} \, dz = \oint_{|z|=\epsilon} \frac{1}{z} \, dz + \oint_{|z|=\epsilon} \left(1 + \frac{z}{2} + \cdots\right) dz$$
$$= 2\pi i + 0 = 2\pi i$$

This will be formalized by the residue theorem.

---

## Practice Problems

### Problem Set A: Direct Applications

**A1.** Verify Cauchy's theorem for $f(z) = z^2$ on the circle $|z| = 1$ by:
(a) Direct parametric computation
(b) Using the existence of an antiderivative

**A2.** Compute $\oint_C \frac{dz}{z^2 + 4}$ where $C$ is:
(a) $|z| = 1$
(b) $|z| = 3$
(c) $|z - 2i| = 1$

**A3.** Find $\int_0^{1+i} z^2 \, dz$ using path independence.

### Problem Set B: Deformation

**B1.** Show that for any simple closed curve $C$ not passing through $z = 1$:
$$\oint_C \frac{dz}{(z-1)^2} = 0$$
regardless of whether $z = 1$ is inside or outside $C$.

**B2.** Let $C$ be the boundary of the square with vertices $\pm 2 \pm 2i$. Compute:
$$\oint_C \frac{z^2 + 1}{z(z^2 + 4)} \, dz$$
by identifying singularities and using deformation.

**B3.** For $f(z) = \frac{1}{z(z-2)}$, compute $\oint_C f(z) \, dz$ where $C$ is:
(a) $|z| = 1$
(b) $|z - 2| = 1$
(c) $|z - 1| = 2$

### Problem Set C: Theory

**C1.** Prove that if $f$ is entire and $\oint_C f(z) \, dz = 0$ for all closed contours $C$, then $f$ has an antiderivative.

**C2.** Show that the winding number $n(C, z_0)$ is always an integer by writing:
$$n(C, z_0) = \frac{1}{2\pi i} \oint_C \frac{dz}{z-z_0} = \frac{1}{2\pi i} \int_0^{2\pi} \frac{z'(t)}{z(t) - z_0} dt$$
and interpreting geometrically.

**C3.** If $f$ is analytic in annulus $r < |z| < R$, show that $\oint_{|z|=\rho_1} f \, dz = \oint_{|z|=\rho_2} f \, dz$ for any $r < \rho_1, \rho_2 < R$.

---

## Solutions to Selected Problems

### Solution A1

**(a) Direct computation:**

$z(t) = e^{it}$, $dz = ie^{it} dt$, $z^2 = e^{2it}$

$$\oint_{|z|=1} z^2 \, dz = \int_0^{2\pi} e^{2it} \cdot ie^{it} \, dt = i\int_0^{2\pi} e^{3it} \, dt$$
$$= i \left[\frac{e^{3it}}{3i}\right]_0^{2\pi} = \frac{1}{3}(e^{6\pi i} - 1) = \frac{1}{3}(1 - 1) = 0 \checkmark$$

**(b) Antiderivative:**

$z^2$ has antiderivative $F(z) = z^3/3$. For any closed contour starting and ending at $z_0$:
$$\oint z^2 \, dz = F(z_0) - F(z_0) = 0 \checkmark$$

### Solution B1

The function $g(z) = \frac{1}{(z-1)^2}$ has antiderivative $G(z) = -\frac{1}{z-1}$ in any simply connected domain not containing $z = 1$.

**Case 1:** If $z = 1$ is outside $C$, then $C$ lies in a simply connected domain where $g$ is analytic.
$$\oint_C g(z) \, dz = 0$$

**Case 2:** If $z = 1$ is inside $C$, deform to a small circle $|z - 1| = \epsilon$:
$$\oint_C \frac{dz}{(z-1)^2} = \oint_{|z-1|=\epsilon} \frac{dz}{(z-1)^2}$$

Parametrize: $z - 1 = \epsilon e^{it}$, $dz = i\epsilon e^{it} dt$
$$= \int_0^{2\pi} \frac{i\epsilon e^{it}}{\epsilon^2 e^{2it}} \, dt = \frac{i}{\epsilon} \int_0^{2\pi} e^{-it} \, dt = \frac{i}{\epsilon} \left[\frac{e^{-it}}{-i}\right]_0^{2\pi} = 0$$

So $\oint_C \frac{dz}{(z-1)^2} = 0$ regardless! This is because $(z-1)^{-2}$ has a "second-order pole" which doesn't contribute to the integral.

---

## Computational Lab

### Lab 1: Verifying Cauchy's Theorem

```python
import numpy as np
import matplotlib.pyplot as plt

def contour_integral(f, z_path, t_range, n=10000):
    """Compute ∮_C f(z) dz numerically."""
    t = np.linspace(t_range[0], t_range[1], n)
    dt = t[1] - t[0]
    z = z_path(t)
    dz = np.gradient(z, dt)
    return np.trapz(f(z) * dz, t)

# Test Cauchy's theorem for various analytic functions
print("Verifying Cauchy's Theorem: ∮_C f(z) dz = 0 for analytic f")
print("=" * 60)

# Contour: unit circle
circle = lambda t: np.exp(1j * t)

# Analytic functions to test
test_functions = [
    (lambda z: z**2, "z²"),
    (lambda z: np.exp(z), "e^z"),
    (lambda z: np.sin(z), "sin(z)"),
    (lambda z: z**3 - 2*z + 1, "z³ - 2z + 1"),
    (lambda z: 1/(z**2 + 4), "1/(z² + 4)"),  # Analytic inside |z|=1
]

for f, name in test_functions:
    result = contour_integral(f, circle, (0, 2*np.pi))
    print(f"∮ {name:15} dz = {result:20.10f} (|•| = {abs(result):.2e})")

print("\nNon-analytic function (should be non-zero):")
f_non = lambda z: np.conj(z)
result = contour_integral(f_non, circle, (0, 2*np.pi))
print(f"∮ {'z̄':15} dz = {result:20.10f}")
print(f"Expected: 2πi = {2*np.pi*1j:.10f}")
```

### Lab 2: Winding Number Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def winding_number(z_path, z0, t_range, n=10000):
    """Compute winding number n(C, z0) = (1/2πi) ∮ dz/(z-z0)."""
    t = np.linspace(t_range[0], t_range[1], n)
    dt = t[1] - t[0]
    z = z_path(t)
    dz = np.gradient(z, dt)
    integrand = dz / (z - z0)
    return np.trapz(integrand, t) / (2 * np.pi * 1j)

# Create various contours
def circle(center, radius):
    return lambda t: center + radius * np.exp(1j * t)

def figure_eight():
    def path(t):
        if isinstance(t, np.ndarray):
            result = np.zeros_like(t, dtype=complex)
            mask = t < np.pi
            result[mask] = np.exp(1j * 2 * t[mask])  # First loop
            result[~mask] = -1 + np.exp(-1j * 2 * (t[~mask] - np.pi))  # Second loop
            return result
        else:
            if t < np.pi:
                return np.exp(1j * 2 * t)
            else:
                return -1 + np.exp(-1j * 2 * (t - np.pi))
    return path

# Simple circle test
print("Winding Numbers for Circle |z| = 2:")
print("-" * 40)
circle_path = circle(0, 2)
for z0 in [0, 1, 1j, 3, -3]:
    w = winding_number(circle_path, z0, (0, 2*np.pi))
    print(f"n(C, {z0:4}) = {w.real:6.3f} + {w.imag:.3f}i ≈ {round(w.real)}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Simple circle
t = np.linspace(0, 2*np.pi, 1000)
z = circle(0, 2)(t)
axes[0].plot(z.real, z.imag, 'b-', linewidth=2)
axes[0].plot([0], [0], 'ro', markersize=10, label='z₀=0 (inside, n=1)')
axes[0].plot([3], [0], 'go', markersize=10, label='z₀=3 (outside, n=0)')
axes[0].arrow(2, 0.1, -0.1, 0.3, head_width=0.2, color='b')
axes[0].set_xlabel('Re(z)')
axes[0].set_ylabel('Im(z)')
axes[0].set_title('Simple Circle')
axes[0].legend(loc='upper right')
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

# Panel 2: Doubly-wound circle (goes around twice)
z2 = circle(0, 2)(2*t)  # Parameter goes 0 to 4π
axes[1].plot(z2.real, z2.imag, 'b-', linewidth=2)
axes[1].plot([0], [0], 'ro', markersize=10)
w = winding_number(lambda s: circle(0, 2)(s), 0, (0, 4*np.pi))
axes[1].set_title(f'Double-wound Circle (n = {w.real:.0f})')
axes[1].set_xlabel('Re(z)')
axes[1].set_ylabel('Im(z)')
axes[1].axis('equal')
axes[1].grid(True, alpha=0.3)

# Panel 3: Two separate circles
theta = np.linspace(0, 2*np.pi, 100)
# Circle around z=1
z_c1 = 1 + 0.5*np.exp(1j*theta)
# Circle around z=-1
z_c2 = -1 + 0.5*np.exp(1j*theta)
axes[2].plot(z_c1.real, z_c1.imag, 'b-', linewidth=2, label='C₁ around z=1')
axes[2].plot(z_c2.real, z_c2.imag, 'r-', linewidth=2, label='C₂ around z=-1')
axes[2].plot([1], [0], 'bo', markersize=8)
axes[2].plot([-1], [0], 'ro', markersize=8)
axes[2].plot([0], [0], 'ko', markersize=8, label='Origin')
axes[2].set_xlabel('Re(z)')
axes[2].set_ylabel('Im(z)')
axes[2].set_title('Two Separate Contours')
axes[2].legend()
axes[2].axis('equal')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('winding_numbers.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 3: Deformation of Contours

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def contour_integral(f, z_path, t_range, n=5000):
    """Compute contour integral numerically."""
    t = np.linspace(t_range[0], t_range[1], n)
    dt = t[1] - t[0]
    z = z_path(t)
    dz = np.gradient(z, dt)
    return np.trapz(f(z) * dz, t)

# Function with singularity at origin
f = lambda z: 1/z

# Define family of contours interpolating between circle and square
def interpolated_contour(s):
    """s=0: circle, s=1: square"""
    def path(t):
        # Circle
        circle = 2 * np.exp(1j * t)

        # Square (parametrized)
        def square_point(theta):
            # Normalize theta to [0, 2π]
            theta = theta % (2*np.pi)
            # Four sides
            if theta < np.pi/2:
                x = 2
                y = 2 * (2*theta/np.pi - 0.5) * 2
                y = np.clip(y, -2, 2)
                return complex(2, 2*np.tan(theta)) if np.cos(theta) > 0.01 else 2+2j
            elif theta < np.pi:
                return complex(2/np.tan(theta-np.pi/2), 2) if np.sin(theta-np.pi/2) > 0.01 else -2+2j
            elif theta < 3*np.pi/2:
                return complex(-2, -2*np.tan(theta-np.pi)) if np.cos(theta-np.pi) < -0.01 else -2-2j
            else:
                return complex(-2/np.tan(theta-3*np.pi/2), -2) if np.sin(theta-3*np.pi/2) < -0.01 else 2-2j

        # For array input, handle element-wise
        if isinstance(t, np.ndarray):
            # Simpler square parametrization
            square = np.zeros_like(t, dtype=complex)
            n = len(t)
            quarter = n // 4
            # Right side
            square[:quarter] = 2 + 2j * np.linspace(-1, 1, quarter)
            # Top side
            square[quarter:2*quarter] = np.linspace(2, -2, quarter) + 2j
            # Left side
            square[2*quarter:3*quarter] = -2 + 2j * np.linspace(1, -1, quarter)
            # Bottom side
            square[3*quarter:] = np.linspace(-2, 2, n - 3*quarter) - 2j

            return (1-s) * circle + s * square
        else:
            return (1-s) * circle

    return path

# Compute integrals for family of contours
s_values = np.linspace(0, 1, 20)
integrals = []

print("Contour Deformation: Circle → Square")
print("Integral of 1/z (should remain 2πi throughout)")
print("-" * 50)

for s in s_values:
    path = interpolated_contour(s)
    integral = contour_integral(f, path, (0, 2*np.pi))
    integrals.append(integral)
    if s in [0, 0.5, 1]:
        print(f"s = {s:.1f}: ∮ dz/z = {integral:.6f}")

print(f"\nExpected: 2πi = {2*np.pi*1j:.6f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot several contours
t = np.linspace(0, 2*np.pi, 1000)
colors = plt.cm.viridis(np.linspace(0, 1, 5))
for i, s in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
    path = interpolated_contour(s)
    z = path(t)
    axes[0].plot(z.real, z.imag, color=colors[i], linewidth=2,
                label=f's={s:.2f}')

axes[0].plot([0], [0], 'r*', markersize=15, label='Singularity')
axes[0].set_xlabel('Re(z)')
axes[0].set_ylabel('Im(z)')
axes[0].set_title('Contour Deformation: Circle to Square')
axes[0].legend()
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

# Plot integral values
axes[1].plot(s_values, [i.real for i in integrals], 'b-', linewidth=2, label='Re(∮dz/z)')
axes[1].plot(s_values, [i.imag for i in integrals], 'r-', linewidth=2, label='Im(∮dz/z)')
axes[1].axhline(y=0, color='b', linestyle='--', alpha=0.5)
axes[1].axhline(y=2*np.pi, color='r', linestyle='--', alpha=0.5, label='2π')
axes[1].set_xlabel('Deformation parameter s')
axes[1].set_ylabel('Integral value')
axes[1].set_title('Integral Remains Constant During Deformation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('contour_deformation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 4: Multiply Connected Domains

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

# Function: 1/(z(z-2)) has singularities at z=0 and z=2
f = lambda z: 1 / (z * (z - 2))

# Partial fractions: 1/(z(z-2)) = A/z + B/(z-2)
# A(z-2) + Bz = 1 => A = -1/2, B = 1/2
# So: 1/(z(z-2)) = -1/(2z) + 1/(2(z-2))

print("Function: f(z) = 1/(z(z-2))")
print("Singularities at z = 0 and z = 2")
print("=" * 50)

# Various contours
contours = [
    (lambda t: np.exp(1j*t), "Unit circle |z|=1", "encloses 0 only"),
    (lambda t: 2 + 0.5*np.exp(1j*t), "|z-2|=0.5", "encloses 2 only"),
    (lambda t: 1 + 2*np.exp(1j*t), "|z-1|=2", "encloses both"),
    (lambda t: 5 + 0.5*np.exp(1j*t), "|z-5|=0.5", "encloses neither"),
]

# Theoretical values from partial fractions
# ∮ f dz = -1/2 * (2πi if 0 inside) + 1/2 * (2πi if 2 inside)

for path, name, description in contours:
    integral = contour_integral(f, path, (0, 2*np.pi))
    print(f"\n{name} ({description}):")
    print(f"  ∮ f(z) dz = {integral:.6f}")

print(f"\nTheoretical predictions:")
print(f"  Only z=0 inside: -1/2 × 2πi = {-0.5 * 2*np.pi*1j:.6f}")
print(f"  Only z=2 inside: +1/2 × 2πi = {0.5 * 2*np.pi*1j:.6f}")
print(f"  Both inside: -πi + πi = 0")
print(f"  Neither inside: 0")

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))

theta = np.linspace(0, 2*np.pi, 100)

# Plot contours
z1 = np.exp(1j*theta)
z2 = 2 + 0.5*np.exp(1j*theta)
z3 = 1 + 2*np.exp(1j*theta)

ax.plot(z1.real, z1.imag, 'b-', linewidth=2, label='|z|=1')
ax.plot(z2.real, z2.imag, 'g-', linewidth=2, label='|z-2|=0.5')
ax.plot(z3.real, z3.imag, 'r-', linewidth=2, label='|z-1|=2')

# Plot singularities
ax.plot([0], [0], 'k*', markersize=15, label='z=0')
ax.plot([2], [0], 'k*', markersize=15, label='z=2')

ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_title('Contours and Singularities of f(z) = 1/(z(z-2))')
ax.legend()
ax.axis('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiply_connected.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\oint_C f(z) \, dz = 0$ | Cauchy's theorem (analytic $f$, simply connected) |
| $n(C, z_0) = \frac{1}{2\pi i}\oint_C \frac{dz}{z-z_0}$ | Winding number (integer) |
| $\int_{C_1} f \, dz = \int_{C_2} f \, dz$ | Path independence |
| $\oint_{C_0} f \, dz = \sum_k \oint_{C_k} f \, dz$ | Multiply connected domains |

### Main Takeaways

1. **Cauchy's theorem** states that $\oint_C f(z) \, dz = 0$ for analytic $f$ in simply connected domains.

2. **Simple connectivity** is essential — holes in the domain can make integrals non-zero.

3. **Contours can be deformed** without changing the integral, as long as we don't cross singularities.

4. **The winding number** counts how many times a curve encircles a point.

5. **In physics**, Cauchy's theorem explains topological effects like the Aharonov-Bohm effect and Berry phase.

---

## Daily Checklist

- [ ] I can state Cauchy's integral theorem precisely
- [ ] I understand the role of simple connectivity
- [ ] I can apply the deformation principle
- [ ] I can compute winding numbers
- [ ] I understand the connection to Green's theorem
- [ ] I see the quantum mechanical significance

---

## Preview: Day 178

Tomorrow we develop **Cauchy's Integral Formula**:

$$f(z_0) = \frac{1}{2\pi i} \oint_C \frac{f(z)}{z - z_0} \, dz$$

This remarkable formula expresses the value of an analytic function inside a contour in terms of its values on the boundary — one of the most powerful results in complex analysis!

---

*"The theory of functions of a complex variable is one of the most beautiful branches of mathematics."*
— David Hilbert
