# Day 200: Boundary Value Problems

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Separation of Variables |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Spherical & Cylindrical Coordinates |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 200, you will be able to:

1. Apply separation of variables to Laplace's equation in Cartesian coordinates
2. Solve boundary value problems with Dirichlet and Neumann conditions
3. Apply separation of variables in spherical coordinates
4. Recognize Legendre polynomials as angular solutions
5. Connect to the angular part of the hydrogen atom wave function
6. Implement numerical solutions using finite differences

---

## Core Content

### 1. Separation of Variables: The Key Technique

**Laplace's equation in Cartesian coordinates:**
$$\frac{\partial^2\phi}{\partial x^2} + \frac{\partial^2\phi}{\partial y^2} + \frac{\partial^2\phi}{\partial z^2} = 0$$

**Separation ansatz:** Assume $\phi(x, y, z) = X(x)Y(y)Z(z)$

Substituting and dividing by $XYZ$:
$$\frac{1}{X}\frac{d^2X}{dx^2} + \frac{1}{Y}\frac{d^2Y}{dy^2} + \frac{1}{Z}\frac{d^2Z}{dz^2} = 0$$

Each term must be constant! Let:
$$\frac{1}{X}\frac{d^2X}{dx^2} = -k_x^2, \quad \frac{1}{Y}\frac{d^2Y}{dy^2} = -k_y^2, \quad \frac{1}{Z}\frac{d^2Z}{dz^2} = k_z^2$$

with $k_x^2 + k_y^2 = k_z^2$.

### 2. Solutions in a Rectangular Box

**Problem:** Find $\phi$ in a box $0 \leq x \leq a$, $0 \leq y \leq b$, $0 \leq z \leq c$ with:
- $\phi = 0$ on all faces except $z = c$
- $\phi = V_0$ on $z = c$

**Solution:**
From boundary conditions on $x$ and $y$:
$$X_n(x) = \sin\left(\frac{n\pi x}{a}\right), \quad Y_m(y) = \sin\left(\frac{m\pi y}{b}\right)$$

For $z$: $k_z^2 = \pi^2\left(\frac{n^2}{a^2} + \frac{m^2}{b^2}\right)$, so:
$$Z(z) = A\sinh(k_z z) + B\cosh(k_z z)$$

With $Z(0) = 0$: $Z(z) = A\sinh(k_z z)$

General solution:
$$\phi(x,y,z) = \sum_{n=1}^{\infty}\sum_{m=1}^{\infty} A_{nm}\sin\frac{n\pi x}{a}\sin\frac{m\pi y}{b}\sinh(k_{nm}z)$$

Coefficients from Fourier series at $z = c$:
$$A_{nm} = \frac{4V_0}{ab\sinh(k_{nm}c)}\int_0^a\int_0^b \sin\frac{n\pi x}{a}\sin\frac{m\pi y}{b}\,dx\,dy$$

### 3. Laplace's Equation in Spherical Coordinates

$$\boxed{\frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial\phi}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial\phi}{\partial\theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2\phi}{\partial\varphi^2} = 0}$$

**Separation:** $\phi(r,\theta,\varphi) = R(r)\Theta(\theta)\Phi(\varphi)$

**Azimuthal equation:**
$$\frac{d^2\Phi}{d\varphi^2} = -m^2\Phi \quad \Rightarrow \quad \Phi = e^{im\varphi}$$

where $m = 0, \pm 1, \pm 2, \ldots$ (single-valuedness).

**Polar equation:** (Associated Legendre equation)
$$\frac{1}{\sin\theta}\frac{d}{d\theta}\left(\sin\theta\frac{d\Theta}{d\theta}\right) + \left[l(l+1) - \frac{m^2}{\sin^2\theta}\right]\Theta = 0$$

Solutions: **Associated Legendre functions** $P_l^m(\cos\theta)$

**Radial equation:**
$$\frac{d}{dr}\left(r^2\frac{dR}{dr}\right) = l(l+1)R$$

Solutions: $R(r) = Ar^l + Br^{-(l+1)}$

### 4. Spherical Harmonics

The angular solutions combine to form **spherical harmonics**:

$$\boxed{Y_l^m(\theta,\varphi) = \sqrt{\frac{(2l+1)}{4\pi}\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\varphi}}$$

**First few Legendre polynomials:**
| $l$ | $P_l(x)$ |
|-----|----------|
| 0 | 1 |
| 1 | $x$ |
| 2 | $\frac{1}{2}(3x^2 - 1)$ |
| 3 | $\frac{1}{2}(5x^3 - 3x)$ |

**General solution in spherical coordinates:**
$$\phi(r,\theta,\varphi) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l}\left(A_{lm}r^l + B_{lm}r^{-(l+1)}\right)Y_l^m(\theta,\varphi)$$

### 5. Azimuthally Symmetric Problems

For $\phi$ independent of $\varphi$ (only $m = 0$):

$$\phi(r,\theta) = \sum_{l=0}^{\infty}\left(A_l r^l + B_l r^{-(l+1)}\right)P_l(\cos\theta)$$

**Interior problem** (no singularity at origin): $B_l = 0$
**Exterior problem** (finite at infinity): $A_l = 0$

### 6. Uniqueness Theorems

**First uniqueness theorem:** If $\phi$ is specified on all boundaries (Dirichlet), the solution to Laplace's equation is unique.

**Second uniqueness theorem:** If $\partial\phi/\partial n$ is specified on all boundaries (Neumann), $\phi$ is unique up to a constant.

---

## Quantum Mechanics Connection

### Hydrogen Atom: The Same Equation!

The angular part of the Schrödinger equation for any central potential:

$$\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial Y}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2 Y}{\partial\varphi^2} = -l(l+1)Y$$

This is **exactly** the angular Laplace equation!

**Key insight:** Spherical harmonics $Y_l^m$ are:
- Eigenfunctions of $\hat{L}^2$ with eigenvalue $\hbar^2 l(l+1)$
- Eigenfunctions of $\hat{L}_z$ with eigenvalue $\hbar m$

### The Quantum Numbers

| Quantum Number | Range | Physical Meaning |
|----------------|-------|------------------|
| $l$ | $0, 1, 2, \ldots$ | Total angular momentum |
| $m$ | $-l, \ldots, +l$ | $z$-component of angular momentum |

**Spectroscopic notation:** $l = 0$ (s), $l = 1$ (p), $l = 2$ (d), $l = 3$ (f), ...

### Radial Schrödinger vs. Laplace

For Laplace: $R(r) \sim r^l$ or $r^{-(l+1)}$

For Schrödinger with Coulomb: $R_{nl}(r) \sim r^l e^{-r/na_0} L_{n-l-1}^{2l+1}(2r/na_0)$

The $r^l$ behavior near the origin is the same!

---

## Worked Examples

### Example 1: Sphere in Uniform Field

**Problem:** A conducting sphere of radius $R$ at potential $\phi = 0$ is placed in a uniform field $\mathbf{E}_0 = E_0\hat{\mathbf{z}}$. Find the potential outside.

**Solution:**
Far from sphere: $\phi \to -E_0 z = -E_0 r\cos\theta$

This suggests $l = 1$ terms. General solution:
$$\phi(r,\theta) = -E_0 r\cos\theta + \frac{A}{r^2}\cos\theta$$

Boundary condition $\phi(R,\theta) = 0$:
$$-E_0 R + \frac{A}{R^2} = 0 \quad \Rightarrow \quad A = E_0 R^3$$

$$\boxed{\phi(r,\theta) = -E_0\left(r - \frac{R^3}{r^2}\right)\cos\theta}$$

The induced dipole moment is $p = 4\pi\varepsilon_0 R^3 E_0$.

### Example 2: Sphere with Surface Charge

**Problem:** A sphere of radius $R$ has surface charge $\sigma(\theta) = \sigma_0\cos\theta$. Find $\phi$ inside and outside.

**Solution:**
This charge distribution has $l = 1$ symmetry.

**Inside ($r < R$):**
$$\phi_{\text{in}} = A_1 r\cos\theta$$

**Outside ($r > R$):**
$$\phi_{\text{out}} = \frac{B_1}{r^2}\cos\theta$$

**Boundary conditions at $r = R$:**
1. Normal $E$ discontinuity: $-\varepsilon_0\left(\frac{\partial\phi_{\text{out}}}{\partial r} - \frac{\partial\phi_{\text{in}}}{\partial r}\right) = \sigma_0\cos\theta$
2. Tangential $E$ continuity: $\phi_{\text{in}}(R) = \phi_{\text{out}}(R)$

From continuity: $A_1 R = B_1/R^2 \Rightarrow B_1 = A_1 R^3$

From discontinuity:
$$-\varepsilon_0\left(-\frac{2B_1}{R^3} - A_1\right)\cos\theta = \sigma_0\cos\theta$$
$$\varepsilon_0(2A_1 + A_1) = \sigma_0 \Rightarrow A_1 = \frac{\sigma_0}{3\varepsilon_0}$$

$$\boxed{\phi_{\text{in}} = \frac{\sigma_0 r\cos\theta}{3\varepsilon_0}, \quad \phi_{\text{out}} = \frac{\sigma_0 R^3\cos\theta}{3\varepsilon_0 r^2}}$$

### Example 3: Rectangular Duct

**Problem:** A rectangular duct has conducting walls at $x = 0, a$ and $y = 0, b$ held at potential $0$, and extends from $z = 0$ to $z = L$. At $z = 0$, $\phi = V_0$. Find $\phi(x, y, z)$ for large $L$.

**Solution:**
$$\phi = \sum_{n,m=1,3,5,...}^{\infty} \frac{16V_0}{nm\pi^2}\sin\frac{n\pi x}{a}\sin\frac{m\pi y}{b}e^{-\pi z\sqrt{n^2/a^2 + m^2/b^2}}$$

The potential decays exponentially into the duct.

---

## Practice Problems

### Problem 1: Direct Application
Solve Laplace's equation in 2D for a semi-infinite strip: $0 < x < a$, $y > 0$ with $\phi(0,y) = \phi(a,y) = 0$ and $\phi(x,0) = V_0$.

**Hint:** Use separation of variables with exponential decay in $y$.

### Problem 2: Intermediate
A hemispherical shell of radius $R$ has its flat face at potential $0$ and its curved surface at potential $V_0$. Find an approximate expression for $\phi$ at the center.

### Problem 3: Challenging
Two concentric spherical shells (radii $a$ and $b > a$) are maintained at potentials $V_a$ and $V_b$. Find $\phi(r)$ between them and show it satisfies Laplace's equation.

**Answer:** $\phi(r) = \frac{V_b b - V_a a}{b-a} + \frac{(V_a - V_b)ab}{(b-a)r}$

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv, sph_harm
from mpl_toolkits.mplot3d import Axes3D

# ========== Legendre Polynomials ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Legendre polynomials
ax1 = axes[0, 0]
x = np.linspace(-1, 1, 200)

for l in range(5):
    P_l = np.polynomial.legendre.Legendre.basis(l)(x)
    ax1.plot(x, P_l, label=f'$P_{l}(x)$', linewidth=2)

ax1.set_xlabel('x = cos(θ)')
ax1.set_ylabel('$P_l(x)$')
ax1.set_title('Legendre Polynomials')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot 2: Spherical harmonics magnitude
ax2 = axes[0, 1]
theta = np.linspace(0, np.pi, 100)
phi_angle = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi_angle)

# Y_2^0
Y_20 = np.real(sph_harm(0, 2, PHI, THETA))
R = np.abs(Y_20)
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)

ax2.contourf(X[:, 50], Z[:, 50], Y_20[:, 50].reshape(-1, 1) * np.ones((100, 100)),
              levels=20, cmap='RdBu_r')
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_title('$Y_2^0(θ,φ)$ in xz-plane')
ax2.set_aspect('equal')

# Plot 3: Finite difference solution to Laplace equation
ax3 = axes[1, 0]

# Solve 2D Laplace in a square with boundary conditions
N = 50
phi = np.zeros((N, N))

# Boundary conditions
phi[0, :] = 0      # bottom
phi[-1, :] = 100   # top (V = 100)
phi[:, 0] = 0      # left
phi[:, -1] = 0     # right

# Jacobi iteration
for iteration in range(5000):
    phi_old = phi.copy()
    phi[1:-1, 1:-1] = 0.25 * (phi[2:, 1:-1] + phi[:-2, 1:-1] +
                               phi[1:-1, 2:] + phi[1:-1, :-2])
    if np.max(np.abs(phi - phi_old)) < 1e-6:
        break

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

contour = ax3.contourf(X, Y, phi, levels=30, cmap='hot')
plt.colorbar(contour, ax=ax3, label='φ (V)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title(f'Laplace Equation (Numerical, {iteration} iterations)')

# Plot 4: Conducting sphere in uniform field
ax4 = axes[1, 1]

# Create grid
r = np.linspace(0.01, 3, 100)
theta = np.linspace(0, np.pi, 100)
R_grid, Theta_grid = np.meshgrid(r, theta)

# Parameters
R_sphere = 1.0
E0 = 1.0

# Potential outside sphere
phi_outside = -E0 * (R_grid - R_sphere**3/R_grid**2) * np.cos(Theta_grid)
phi_outside = np.where(R_grid > R_sphere, phi_outside, 0)

# Convert to Cartesian for plotting
X = R_grid * np.sin(Theta_grid)
Z = R_grid * np.cos(Theta_grid)

contour = ax4.contourf(X, Z, phi_outside, levels=30, cmap='RdBu_r')
plt.colorbar(contour, ax=ax4, label='φ')

# Draw sphere
circle = plt.Circle((0, 0), R_sphere, fill=True, color='gray', alpha=0.5)
ax4.add_patch(circle)

ax4.set_xlabel('x')
ax4.set_ylabel('z')
ax4.set_title('Conducting Sphere in Uniform Field')
ax4.set_xlim(0, 3)
ax4.set_ylim(-3, 3)
ax4.set_aspect('equal')

plt.tight_layout()
plt.savefig('day_200_boundary_value.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 3D Spherical Harmonics ==========
fig = plt.figure(figsize=(15, 5))

for idx, (l, m) in enumerate([(1, 0), (2, 0), (2, 1)]):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')

    theta = np.linspace(0, np.pi, 50)
    phi_angle = np.linspace(0, 2*np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi_angle)

    Y_lm = sph_harm(m, l, PHI, THETA)
    R = np.abs(Y_lm)

    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # Color by real part
    colors = np.real(Y_lm)
    colors = (colors - colors.min()) / (colors.max() - colors.min())

    ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r(colors), alpha=0.8)
    ax.set_title(f'$|Y_{l}^{m}|$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

plt.tight_layout()
plt.savefig('day_200_spherical_harmonics.png', dpi=150, bbox_inches='tight')
plt.show()

print("Day 200: Boundary Value Problems Complete")
print("="*50)
print("\nKey results:")
print("• Separation of variables reduces PDEs to ODEs")
print("• Spherical harmonics Y_l^m are universal angular solutions")
print("• Same Y_l^m appear in quantum angular momentum")
print("• l = 0,1,2,... → s,p,d,... orbitals")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\phi = X(x)Y(y)Z(z)$ | Separation ansatz |
| $Y_l^m(\theta,\phi) = P_l^m(\cos\theta)e^{im\phi}$ | Spherical harmonics |
| $\phi = \sum_{l,m}(A_{lm}r^l + B_{lm}r^{-(l+1)})Y_l^m$ | General spherical solution |
| $P_0 = 1, P_1 = x, P_2 = \frac{1}{2}(3x^2-1)$ | Legendre polynomials |

### Main Takeaways

1. **Separation of variables** reduces Laplace's equation to ODEs
2. **Boundary conditions** determine coefficients via Fourier series
3. **Spherical harmonics** are the universal angular solutions
4. **Same math** appears in quantum angular momentum
5. **$l$ and $m$** become orbital and magnetic quantum numbers

---

## Daily Checklist

- [ ] I can apply separation of variables in Cartesian coordinates
- [ ] I understand the spherical Laplace equation
- [ ] I recognize Legendre polynomials and spherical harmonics
- [ ] I can set up boundary value problems
- [ ] I see the connection to quantum angular momentum

---

## Preview: Day 201

Tomorrow we study the **method of images** — a clever technique for finding potentials near conductors by replacing boundaries with fictitious charges.

---

*"The same mathematics that governs electrostatic potentials governs the angular part of quantum wave functions — this is no coincidence."*

---

**Next:** Day 201 — Method of Images
