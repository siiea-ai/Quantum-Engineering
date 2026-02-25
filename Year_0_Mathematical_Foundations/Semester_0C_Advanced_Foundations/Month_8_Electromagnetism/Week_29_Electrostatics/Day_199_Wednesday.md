# Day 199: Electrostatic Potential

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Potential & Poisson's Equation |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Boundary Value Problems |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 199, you will be able to:

1. Define electrostatic potential from work and energy considerations
2. Derive the potential from field and vice versa
3. State and derive Poisson's and Laplace's equations
4. Calculate potentials for various charge distributions
5. Understand the physical meaning of potential energy
6. Connect to quantum mechanical potential energy in the Hamiltonian

---

## Core Content

### 1. Work and Potential Energy

**Work done by the electric field** moving a charge from $\mathbf{a}$ to $\mathbf{b}$:
$$W = \int_{\mathbf{a}}^{\mathbf{b}} \mathbf{F} \cdot d\boldsymbol{\ell} = q\int_{\mathbf{a}}^{\mathbf{b}} \mathbf{E} \cdot d\boldsymbol{\ell}$$

**Key insight:** Because $\nabla \times \mathbf{E} = 0$ (electrostatics), the work is path-independent.

**Potential energy:**
$$U(\mathbf{r}) = -W_{\infty \to \mathbf{r}} = -q\int_{\infty}^{\mathbf{r}} \mathbf{E} \cdot d\boldsymbol{\ell}$$

### 2. Electrostatic Potential

**Definition:**
$$\boxed{\phi(\mathbf{r}) = -\int_{\infty}^{\mathbf{r}} \mathbf{E} \cdot d\boldsymbol{\ell}}$$

So $U = q\phi$ (potential energy = charge × potential).

**Units:** Volts (V) = Joules/Coulomb

**Fundamental relationship:**
$$\boxed{\mathbf{E} = -\nabla \phi}$$

The electric field is the negative gradient of the potential.

### 3. Potential from Charge Distributions

**Point charge:**
$$\phi(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0}\frac{q}{|\mathbf{r} - \mathbf{r}'|}$$

**Discrete charges:**
$$\phi(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0}\sum_i \frac{q_i}{|\mathbf{r} - \mathbf{r}_i|}$$

**Continuous distribution:**
$$\boxed{\phi(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0}\int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|}d^3r'}$$

### 4. Poisson's and Laplace's Equations

Taking divergence of $\mathbf{E} = -\nabla\phi$ and using Gauss's law:

$$\nabla \cdot \mathbf{E} = -\nabla^2\phi = \frac{\rho}{\varepsilon_0}$$

**Poisson's equation:**
$$\boxed{\nabla^2\phi = -\frac{\rho}{\varepsilon_0}}$$

**Laplace's equation** (in charge-free regions):
$$\boxed{\nabla^2\phi = 0}$$

In Cartesian coordinates:
$$\frac{\partial^2\phi}{\partial x^2} + \frac{\partial^2\phi}{\partial y^2} + \frac{\partial^2\phi}{\partial z^2} = 0$$

### 5. Properties of Harmonic Functions

Solutions to Laplace's equation (harmonic functions) have remarkable properties:

1. **Mean value theorem:** The value at any point equals the average over any sphere centered there
2. **Maximum principle:** No local maxima or minima in the interior
3. **Uniqueness:** Given boundary conditions, the solution is unique

**Earnshaw's theorem:** A charge cannot be held in stable equilibrium by electrostatic forces alone. (No potential wells!)

### 6. Potential Energy of Charge Distributions

**Energy to assemble discrete charges:**
$$U = \frac{1}{2}\sum_{i\neq j}\frac{q_i q_j}{4\pi\varepsilon_0|\mathbf{r}_i - \mathbf{r}_j|} = \frac{1}{2}\sum_i q_i \phi(\mathbf{r}_i)$$

**Energy in terms of field:**
$$\boxed{U = \frac{\varepsilon_0}{2}\int E^2\,d^3r}$$

Energy is stored in the electric field itself.

---

## Quantum Mechanics Connection

### The Quantum Potential Energy

In quantum mechanics, the potential energy operator is simply multiplication by $V(\mathbf{r}) = q\phi(\mathbf{r})$:

$$\hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r})$$

For the hydrogen atom with Coulomb potential:
$$V(r) = -\frac{e^2}{4\pi\varepsilon_0 r}$$

### Why Potential Matters

1. **Energy eigenvalues** depend directly on $V(\mathbf{r})$
2. **Bound states** exist only where $E < V(\infty)$
3. **Tunneling** occurs through potential barriers
4. **WKB approximation** depends on $\sqrt{E - V(x)}$

### Poisson's Equation and Charge Density

In quantum mechanics, the charge density from a wave function is:
$$\rho(\mathbf{r}) = -e|\psi(\mathbf{r})|^2$$

This creates a potential through Poisson's equation — the basis of the Hartree-Fock method for many-electron atoms.

---

## Worked Examples

### Example 1: Potential of a Dipole

**Problem:** Find the potential of an electric dipole (charges $\pm q$ separated by $d$) at large distances.

**Solution:**
Place $+q$ at $(0, 0, d/2)$ and $-q$ at $(0, 0, -d/2)$.

At point $(r, \theta, \phi)$ with $r \gg d$:
$$\phi = \frac{q}{4\pi\varepsilon_0}\left(\frac{1}{r_+} - \frac{1}{r_-}\right)$$

Using:
$$r_\pm = \sqrt{r^2 \mp rd\cos\theta + d^2/4} \approx r\left(1 \mp \frac{d\cos\theta}{2r}\right)$$

$$\frac{1}{r_\pm} \approx \frac{1}{r}\left(1 \pm \frac{d\cos\theta}{2r}\right)$$

$$\phi \approx \frac{q}{4\pi\varepsilon_0 r}\cdot\frac{d\cos\theta}{r}$$

$$\boxed{\phi = \frac{p\cos\theta}{4\pi\varepsilon_0 r^2}}$$

where $p = qd$ is the dipole moment.

### Example 2: Potential of Uniformly Charged Sphere

**Problem:** Find $\phi(r)$ for a uniformly charged sphere of radius $R$ and total charge $Q$.

**Solution:**

*Outside ($r > R$):*
From $E = Q/(4\pi\varepsilon_0 r^2)$:
$$\phi = -\int_\infty^r E\,dr = \frac{Q}{4\pi\varepsilon_0 r}$$

*Inside ($r < R$):*
From $E = Qr/(4\pi\varepsilon_0 R^3)$:
$$\phi(r) = \phi(R) - \int_R^r E\,dr = \frac{Q}{4\pi\varepsilon_0 R} - \frac{Q}{4\pi\varepsilon_0 R^3}\int_R^r r'\,dr'$$

$$\phi(r) = \frac{Q}{4\pi\varepsilon_0 R} - \frac{Q}{4\pi\varepsilon_0 R^3}\cdot\frac{r^2 - R^2}{2}$$

$$\boxed{\phi(r) = \frac{Q}{4\pi\varepsilon_0}\cdot\frac{3R^2 - r^2}{2R^3}} \quad (r < R)$$

At center: $\phi(0) = \frac{3Q}{8\pi\varepsilon_0 R} = \frac{3}{2}\phi(R)$

### Example 3: Parallel Plate Capacitor

**Problem:** Two infinite parallel conducting plates are at $z = 0$ (potential $0$) and $z = d$ (potential $V_0$). Find $\phi(z)$ and $\mathbf{E}$ between them.

**Solution:**
Laplace's equation in 1D: $\frac{d^2\phi}{dz^2} = 0$

General solution: $\phi(z) = Az + B$

Boundary conditions:
- $\phi(0) = 0 \Rightarrow B = 0$
- $\phi(d) = V_0 \Rightarrow A = V_0/d$

$$\boxed{\phi(z) = \frac{V_0 z}{d}}$$

$$\mathbf{E} = -\nabla\phi = -\frac{V_0}{d}\hat{\mathbf{z}}$$

The field is uniform between the plates.

---

## Practice Problems

### Problem 1: Direct Application
Calculate the potential at the center of a square of side $a$ with charges $+q$ at each corner.

**Answer:** $\phi = \frac{4q}{4\pi\varepsilon_0(a/\sqrt{2})} = \frac{4\sqrt{2}q}{4\pi\varepsilon_0 a}$

### Problem 2: Intermediate
A thin spherical shell of radius $R$ has surface charge density $\sigma = \sigma_0\cos\theta$. Show that the potential inside is $\phi = \frac{\sigma_0 r\cos\theta}{3\varepsilon_0}$.

**Hint:** The solution must satisfy Laplace's equation and match boundary conditions.

### Problem 3: Challenging
Two concentric spherical shells have radii $a$ and $b$ ($b > a$). The inner shell is at potential $V_0$, the outer at potential $0$. Find $\phi(r)$ for $a < r < b$.

**Answer:** $\phi(r) = V_0\frac{b}{b-a}\left(\frac{a}{r} - 1\right) + V_0$ (after simplification)

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import lpmv  # Legendre polynomials

# Constants
eps0 = 8.854e-12
k = 1/(4*np.pi*eps0)

def potential_point_charge(q, r_charge, r_field):
    """Potential from a point charge."""
    r_sep = np.linalg.norm(r_field - r_charge)
    if r_sep < 1e-10:
        return np.inf
    return k * q / r_sep

def potential_dipole(p, r, theta):
    """Dipole potential at (r, theta)."""
    return k * p * np.cos(theta) / r**2

def potential_charged_sphere(Q, R, r):
    """Potential of uniformly charged sphere."""
    if r >= R:
        return k * Q / r
    else:
        return k * Q * (3*R**2 - r**2) / (2*R**3)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Point charge potential ==========
ax1 = axes[0, 0]
r = np.linspace(0.1, 5, 200)
Q = 1e-9
phi = k * Q / r

ax1.plot(r, phi * 1e9, 'b-', linewidth=2)
ax1.set_xlabel('r (m)')
ax1.set_ylabel('φ (nV·m)')
ax1.set_title('Point Charge Potential (1/r)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 5)

# ========== Plot 2: Charged sphere potential ==========
ax2 = axes[0, 1]
R = 1.0
Q = 1e-9
r = np.linspace(0.01, 3, 300)
phi = np.array([potential_charged_sphere(Q, R, ri) for ri in r])

ax2.plot(r, phi * 1e9, 'b-', linewidth=2, label='Charged sphere')
ax2.axvline(x=R, color='r', linestyle='--', label=f'R = {R} m')
ax2.plot(r[r > R], k * Q / r[r > R] * 1e9, 'g--', linewidth=1.5, label='Point charge')
ax2.set_xlabel('r (m)')
ax2.set_ylabel('φ (nV·m)')
ax2.set_title('Uniformly Charged Sphere Potential')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ========== Plot 3: 2D potential contours (dipole) ==========
ax3 = axes[1, 0]
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Dipole: +q at (0, 0.2), -q at (0, -0.2)
q = 1e-9
d = 0.4
phi_dipole = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        r_field = np.array([X[i,j], Y[i,j], 0])
        phi_plus = potential_point_charge(q, np.array([0, d/2, 0]), r_field)
        phi_minus = potential_point_charge(-q, np.array([0, -d/2, 0]), r_field)
        phi_dipole[i,j] = phi_plus + phi_minus

# Clip for visualization
phi_dipole = np.clip(phi_dipole * 1e9, -10, 10)

contour = ax3.contourf(X, Y, phi_dipole, levels=30, cmap='RdBu_r')
plt.colorbar(contour, ax=ax3, label='φ (nV·m)')
ax3.plot(0, d/2, 'ro', markersize=10, label='+q')
ax3.plot(0, -d/2, 'bo', markersize=10, label='-q')
ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')
ax3.set_title('Electric Dipole Potential')
ax3.set_aspect('equal')
ax3.legend()

# ========== Plot 4: Parallel plate potential ==========
ax4 = axes[1, 1]
d = 1.0  # plate separation
V0 = 100  # voltage
z = np.linspace(0, d, 100)
phi_plates = V0 * z / d

ax4.plot(z, phi_plates, 'b-', linewidth=2)
ax4.axhline(y=0, color='r', linewidth=3, label='Plate at V=0')
ax4.axhline(y=V0, color='g', linewidth=3, label=f'Plate at V={V0}V')
ax4.fill_between([0, d], [0, 0], [V0, V0], alpha=0.2)
ax4.set_xlabel('z (m)')
ax4.set_ylabel('φ (V)')
ax4.set_title('Parallel Plate Capacitor')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_199_potential.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 3D Potential Surface ==========
fig = plt.figure(figsize=(12, 5))

# Point charge
ax1 = fig.add_subplot(121, projection='3d')
r = np.linspace(0.5, 3, 50)
theta = np.linspace(0, 2*np.pi, 50)
R, Theta = np.meshgrid(r, theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = k * 1e-9 / R  # Potential

ax1.plot_surface(X, Y, Z * 1e9, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('φ (nV·m)')
ax1.set_title('Point Charge Potential Surface')

# Dipole
ax2 = fig.add_subplot(122, projection='3d')
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
R = np.where(R < 0.3, 0.3, R)  # Avoid singularity
Theta = np.arctan2(Y, X)

# Dipole along x-axis
p = 1e-9 * 0.1  # dipole moment
Z_dipole = k * p * X / (R**3)
Z_dipole = np.clip(Z_dipole * 1e9, -5, 5)

ax2.plot_surface(X, Y, Z_dipole, cmap='RdBu_r', alpha=0.8)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('φ (nV·m)')
ax2.set_title('Dipole Potential Surface')

plt.tight_layout()
plt.savefig('day_199_potential_3d.png', dpi=150, bbox_inches='tight')
plt.show()

# Verify Laplace equation numerically
print("Numerical verification of Laplace's equation")
print("="*50)
print("\nFor the parallel plate solution φ = V₀z/d:")
print("∂²φ/∂z² = 0 ✓ (linear function)")

# Energy in electric field
E_field = V0 / d  # V/m
energy_density = 0.5 * eps0 * E_field**2
print(f"\nParallel plate capacitor:")
print(f"  E = {E_field:.0f} V/m")
print(f"  Energy density = ½ε₀E² = {energy_density:.3e} J/m³")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\phi = -\int_\infty^{\mathbf{r}} \mathbf{E}\cdot d\boldsymbol{\ell}$ | Potential definition |
| $\mathbf{E} = -\nabla\phi$ | Field from potential |
| $\phi = \frac{1}{4\pi\varepsilon_0}\int\frac{\rho}{|\mathbf{r}-\mathbf{r}'|}d^3r'$ | Potential from distribution |
| $\nabla^2\phi = -\rho/\varepsilon_0$ | Poisson's equation |
| $\nabla^2\phi = 0$ | Laplace's equation |
| $U = \frac{\varepsilon_0}{2}\int E^2\,d^3r$ | Field energy |

### Main Takeaways

1. **Potential** is work per unit charge to bring from infinity
2. **$\mathbf{E} = -\nabla\phi$** — scalar potential simplifies vector problems
3. **Poisson's equation** connects potential to charge distribution
4. **Laplace's equation** governs charge-free regions
5. **Energy is stored in the field**, not just in charges
6. **In QM**, the potential becomes the potential energy operator in $\hat{H}$

---

## Daily Checklist

- [ ] I can define electrostatic potential from work
- [ ] I can derive $\mathbf{E}$ from $\phi$ and vice versa
- [ ] I can state Poisson's and Laplace's equations
- [ ] I understand the energy stored in electric fields
- [ ] I see the connection to quantum potential energy

---

## Preview: Day 200

Tomorrow we tackle **boundary value problems** — solving Laplace's equation with separation of variables. This technique directly connects to solving the Schrödinger equation for hydrogen.

---

*"The introduction of the potential was one of the most important advances in the mathematical treatment of electromagnetism."*

---

**Next:** Day 200 — Boundary Value Problems
