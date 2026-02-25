# Day 198: Gauss's Law

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Gauss's Law & Flux |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications to Symmetric Problems |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 198, you will be able to:

1. Define electric flux and calculate it through various surfaces
2. State and prove Gauss's law from Coulomb's law
3. Apply Gauss's law to problems with spherical symmetry
4. Apply Gauss's law to problems with cylindrical symmetry
5. Apply Gauss's law to problems with planar symmetry
6. Recognize when Gauss's law is and isn't useful

---

## Core Content

### 1. Electric Flux

**Definition:**
The electric flux through a surface $S$ is:

$$\boxed{\Phi_E = \int_S \mathbf{E} \cdot d\mathbf{a}}$$

where $d\mathbf{a} = \hat{\mathbf{n}}\,da$ is the outward-pointing area element.

**Physical interpretation:**
- Flux measures "flow" of field lines through a surface
- Positive: field lines exit; Negative: field lines enter
- Units: N·m²/C (or equivalently V·m)

**For uniform field through flat surface:**
$$\Phi_E = E \cdot A \cdot \cos\theta$$

where $\theta$ is the angle between $\mathbf{E}$ and $\hat{\mathbf{n}}$.

### 2. Gauss's Law: Statement

$$\boxed{\oint_S \mathbf{E} \cdot d\mathbf{a} = \frac{Q_{\text{enc}}}{\varepsilon_0}}$$

**In words:** The electric flux through any closed surface equals the enclosed charge divided by $\varepsilon_0$.

**Differential form:**
$$\boxed{\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}}$$

This is the first of Maxwell's equations.

### 3. Proof from Coulomb's Law

**For a point charge at the origin:**

Consider a spherical Gaussian surface of radius $r$:
$$\Phi_E = \oint \mathbf{E} \cdot d\mathbf{a} = \oint \frac{q}{4\pi\varepsilon_0 r^2}\hat{\mathbf{r}} \cdot \hat{\mathbf{r}}\,da$$

Since $\mathbf{E} \parallel d\mathbf{a}$ everywhere on the sphere:
$$\Phi_E = \frac{q}{4\pi\varepsilon_0 r^2} \cdot 4\pi r^2 = \frac{q}{\varepsilon_0}$$

**Key insight:** The flux is independent of $r$ because the field falls off as $1/r^2$ but the area grows as $r^2$. This is the geometric essence of the inverse-square law.

**For a charge outside the surface:** The flux entering equals the flux leaving, giving net flux = 0.

### 4. Strategy for Applying Gauss's Law

1. **Identify symmetry:** Spherical, cylindrical, or planar
2. **Choose Gaussian surface:** Where $\mathbf{E} \cdot d\mathbf{a}$ is constant or zero
3. **Evaluate the flux integral:** Usually becomes $E \times (\text{area})$
4. **Calculate enclosed charge:** $Q_{\text{enc}} = \int \rho\,dV$
5. **Apply Gauss's law:** Solve for $E$

### 5. Spherical Symmetry

**Uniformly charged solid sphere** (radius $R$, total charge $Q$):

*Outside ($r > R$):*
Choose Gaussian sphere of radius $r$:
$$E \cdot 4\pi r^2 = \frac{Q}{\varepsilon_0}$$
$$\boxed{E = \frac{Q}{4\pi\varepsilon_0 r^2}} \quad (r > R)$$

Same as point charge — the sphere "looks like" a point from outside.

*Inside ($r < R$):*
Enclosed charge: $Q_{\text{enc}} = Q \cdot \frac{4\pi r^3/3}{4\pi R^3/3} = Q\frac{r^3}{R^3}$
$$E \cdot 4\pi r^2 = \frac{Q r^3}{\varepsilon_0 R^3}$$
$$\boxed{E = \frac{Qr}{4\pi\varepsilon_0 R^3}} \quad (r < R)$$

Field grows linearly with $r$ inside!

**Spherical shell** (inner radius $a$, outer radius $b$):
- $r < a$: $E = 0$ (no enclosed charge)
- $a < r < b$: $E = \frac{Q_{\text{enc}}(r)}{4\pi\varepsilon_0 r^2}$
- $r > b$: $E = \frac{Q}{4\pi\varepsilon_0 r^2}$

### 6. Cylindrical Symmetry

**Infinite line charge** (linear charge density $\lambda$):

Choose Gaussian cylinder of radius $s$ and length $L$:
- Curved surface: $\mathbf{E} \parallel d\mathbf{a}$
- End caps: $\mathbf{E} \perp d\mathbf{a}$ (no contribution)

$$E \cdot 2\pi s L = \frac{\lambda L}{\varepsilon_0}$$
$$\boxed{E = \frac{\lambda}{2\pi\varepsilon_0 s}}$$

**Infinite cylindrical conductor** (radius $R$, surface charge $\sigma$):
- Inside: $E = 0$ (conductor in equilibrium)
- Outside: $E = \frac{\lambda}{2\pi\varepsilon_0 s}$ where $\lambda = 2\pi R \sigma$

### 7. Planar Symmetry

**Infinite plane** (surface charge density $\sigma$):

Choose Gaussian pillbox straddling the plane:
$$2EA = \frac{\sigma A}{\varepsilon_0}$$
$$\boxed{E = \frac{\sigma}{2\varepsilon_0}}$$

The factor of 2 accounts for field lines emerging from both sides.

**Parallel plates** (capacitor):
Between plates: $E = \sigma/\varepsilon_0$ (fields add)
Outside: $E = 0$ (fields cancel)

---

## Quantum Mechanics Connection

### Field Inside Atoms

Gauss's law explains why electrons don't collapse into the nucleus:

For a uniformly charged sphere (crude nuclear model):
$$E(r) \propto r \quad \text{(inside)}$$

The Coulomb potential inside is:
$$V(r) = \frac{Q}{4\pi\varepsilon_0}\left(\frac{3R^2 - r^2}{2R^3}\right)$$

This is approximately **harmonic** near the center! The nuclear shell model uses this insight.

### Screening in Many-Electron Atoms

In multi-electron atoms, inner electrons "screen" the nuclear charge for outer electrons:

$$Z_{\text{eff}} = Z - \sigma_{\text{screen}}$$

This is essentially Gauss's law: outer electrons see only the net enclosed charge.

### Flux Quantization in Superconductors

The magnetic analog of Gauss's law leads to flux quantization:
$$\Phi_B = n\frac{h}{2e}$$

where $n$ is an integer. This is a macroscopic quantum effect.

---

## Worked Examples

### Example 1: Non-uniform Spherical Charge

**Problem:** A sphere of radius $R$ has charge density $\rho(r) = \rho_0(1 - r/R)$. Find $E(r)$ for all $r$.

**Solution:**

*Enclosed charge for $r < R$:*
$$Q_{\text{enc}} = \int_0^r \rho_0\left(1 - \frac{r'}{R}\right) 4\pi r'^2 dr'$$
$$= 4\pi\rho_0\left[\frac{r^3}{3} - \frac{r^4}{4R}\right]$$

Apply Gauss's law:
$$E = \frac{Q_{\text{enc}}}{4\pi\varepsilon_0 r^2} = \frac{\rho_0}{\varepsilon_0}\left(\frac{r}{3} - \frac{r^2}{4R}\right)$$

*For $r > R$:*
$$Q_{\text{total}} = 4\pi\rho_0\left[\frac{R^3}{3} - \frac{R^4}{4R}\right] = 4\pi\rho_0 \cdot \frac{R^3}{12} = \frac{\pi\rho_0 R^3}{3}$$

$$\boxed{E = \frac{\rho_0 R^3}{12\varepsilon_0 r^2}} \quad (r > R)$$

### Example 2: Coaxial Cable

**Problem:** A long coaxial cable has inner conductor (radius $a$) with charge $+\lambda$ per unit length and outer conductor (inner radius $b$) with charge $-\lambda$ per unit length. Find $E(r)$ everywhere.

**Solution:**

*For $r < a$:* Inside conductor, $E = 0$.

*For $a < r < b$:*
$$E \cdot 2\pi r L = \frac{\lambda L}{\varepsilon_0}$$
$$\boxed{E = \frac{\lambda}{2\pi\varepsilon_0 r}}$$

*For $r > b$:* Total enclosed charge = $+\lambda L - \lambda L = 0$
$$\boxed{E = 0}$$

The electric field is confined between the conductors — this is why coaxial cables don't radiate.

### Example 3: Two Concentric Shells

**Problem:** A spherical shell of radius $a$ carries charge $+Q$. A concentric shell of radius $b > a$ carries charge $-2Q$. Find the field in all regions.

**Solution:**

| Region | $Q_{\text{enc}}$ | $E$ |
|--------|------------------|-----|
| $r < a$ | 0 | 0 |
| $a < r < b$ | $+Q$ | $\frac{Q}{4\pi\varepsilon_0 r^2}$ (outward) |
| $r > b$ | $+Q - 2Q = -Q$ | $\frac{Q}{4\pi\varepsilon_0 r^2}$ (inward) |

---

## Practice Problems

### Problem 1: Direct Application
A uniformly charged sphere of radius $R = 0.1$ m has total charge $Q = 10$ nC. Find the electric field at:
(a) $r = 0.05$ m (inside)
(b) $r = 0.2$ m (outside)

**Answers:** (a) $E = 4.5 \times 10^3$ N/C; (b) $E = 2.25 \times 10^3$ N/C

### Problem 2: Intermediate
A long cylinder of radius $R$ has volume charge density $\rho = \rho_0(r/R)^2$. Find the electric field for $r < R$ and $r > R$.

**Hint:** Compute $Q_{\text{enc}}$ by integrating over cylindrical shells.

### Problem 3: Challenging
A hemisphere of radius $R$ with uniform surface charge $\sigma$ sits on the $xy$-plane. Use Gauss's law considerations to find the electric field at the center of the flat base.

**Hint:** Consider what a complete sphere would give, then use symmetry.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
eps0 = 8.854e-12
k = 1/(4*np.pi*eps0)

def E_solid_sphere(r, R, Q):
    """Electric field of uniformly charged solid sphere."""
    if r < R:
        return k * Q * r / R**3
    else:
        return k * Q / r**2

def E_spherical_shell(r, a, b, Q):
    """Electric field of uniformly charged spherical shell."""
    if r < a:
        return 0
    elif r < b:
        # Enclosed charge
        rho = Q / (4/3 * np.pi * (b**3 - a**3))
        Q_enc = rho * 4/3 * np.pi * (r**3 - a**3)
        return k * Q_enc / r**2
    else:
        return k * Q / r**2

def E_infinite_cylinder(s, R, lambda_):
    """Electric field of uniformly charged infinite cylinder."""
    if s < R:
        rho = lambda_ / (np.pi * R**2)
        lambda_enc = rho * np.pi * s**2
        return lambda_enc / (2 * np.pi * eps0 * s) if s > 0 else 0
    else:
        return lambda_ / (2 * np.pi * eps0 * s)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Solid sphere ==========
ax1 = axes[0, 0]
R = 1.0  # radius
Q = 1e-9  # charge

r = np.linspace(0.01, 3, 500)
E = np.array([E_solid_sphere(ri, R, Q) for ri in r])

ax1.plot(r, E * 1e6, 'b-', linewidth=2, label='E(r)')
ax1.axvline(x=R, color='r', linestyle='--', linewidth=1.5, label=f'R = {R} m')
ax1.fill_between([0, R], [0, 0], [E.max()*1.1e6, E.max()*1.1e6],
                  alpha=0.2, color='blue', label='Inside sphere')
ax1.set_xlabel('r (m)')
ax1.set_ylabel('E (μN/C)')
ax1.set_title('Uniformly Charged Solid Sphere')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 3)

# ========== Plot 2: Spherical shell ==========
ax2 = axes[0, 1]
a, b = 0.5, 1.0
Q = 1e-9

r = np.linspace(0.01, 2.5, 500)
E = np.array([E_spherical_shell(ri, a, b, Q) for ri in r])

ax2.plot(r, E * 1e6, 'b-', linewidth=2, label='E(r)')
ax2.axvline(x=a, color='g', linestyle='--', linewidth=1.5, label=f'a = {a} m')
ax2.axvline(x=b, color='r', linestyle='--', linewidth=1.5, label=f'b = {b} m')
ax2.set_xlabel('r (m)')
ax2.set_ylabel('E (μN/C)')
ax2.set_title('Charged Spherical Shell (a < r < b)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 2.5)

# ========== Plot 3: Infinite cylinder ==========
ax3 = axes[1, 0]
R = 0.5
lambda_ = 1e-9

s = np.linspace(0.01, 2, 500)
E = np.array([E_infinite_cylinder(si, R, lambda_) for si in s])

ax3.plot(s, E * 1e6, 'b-', linewidth=2, label='E(s)')
ax3.axvline(x=R, color='r', linestyle='--', linewidth=1.5, label=f'R = {R} m')
ax3.set_xlabel('s (m)')
ax3.set_ylabel('E (μN/C)')
ax3.set_title('Uniformly Charged Infinite Cylinder')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 2)

# ========== Plot 4: Field comparison ==========
ax4 = axes[1, 1]

# Compare point charge, solid sphere, and shell
r = np.linspace(0.01, 3, 500)
E_point = k * Q / r**2
E_solid = np.array([E_solid_sphere(ri, 1.0, Q) for ri in r])
E_shell = np.array([E_spherical_shell(ri, 0.8, 1.0, Q) for ri in r])

ax4.plot(r, E_point * 1e6, 'r-', linewidth=2, label='Point charge')
ax4.plot(r, E_solid * 1e6, 'b-', linewidth=2, label='Solid sphere (R=1)')
ax4.plot(r, E_shell * 1e6, 'g-', linewidth=2, label='Shell (0.8<r<1)')
ax4.axvline(x=1, color='k', linestyle=':', linewidth=1)
ax4.set_xlabel('r (m)')
ax4.set_ylabel('E (μN/C)')
ax4.set_title('Comparison of Field Profiles')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 3)
ax4.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('day_198_gauss_law.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 2D Visualization of Gaussian Surfaces ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Spherical Gaussian surface
ax1 = axes[0]
theta = np.linspace(0, 2*np.pi, 100)
circle = plt.Circle((0, 0), 1.5, fill=False, color='blue', linewidth=2,
                     linestyle='--', label='Gaussian surface')
ax1.add_patch(circle)
ax1.plot(0, 0, 'ro', markersize=20, label='Point charge')

# Draw field arrows
for t in np.linspace(0, 2*np.pi, 12, endpoint=False):
    x, y = 1.5*np.cos(t), 1.5*np.sin(t)
    dx, dy = 0.3*np.cos(t), 0.3*np.sin(t)
    ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.05, fc='green', ec='green')

ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_aspect('equal')
ax1.set_title('Spherical Symmetry')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cylindrical Gaussian surface
ax2 = axes[1]
rect = plt.Rectangle((-1, -1.5), 2, 3, fill=False, color='blue',
                       linewidth=2, linestyle='--')
ax2.add_patch(rect)
ax2.axvline(x=0, color='r', linewidth=3, label='Line charge')

# Field arrows (radial)
for y_pos in np.linspace(-1, 1, 5):
    ax2.arrow(0.1, y_pos, 0.7, 0, head_width=0.1, head_length=0.05, fc='green', ec='green')
    ax2.arrow(-0.1, y_pos, -0.7, 0, head_width=0.1, head_length=0.05, fc='green', ec='green')

ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_aspect('equal')
ax2.set_title('Cylindrical Symmetry')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Planar Gaussian surface (pillbox)
ax3 = axes[2]
# Plane
ax3.axhline(y=0, color='r', linewidth=3, label='Charged plane')
# Pillbox
rect = plt.Rectangle((-0.5, -0.8), 1, 1.6, fill=False, color='blue',
                       linewidth=2, linestyle='--')
ax3.add_patch(rect)

# Field arrows
ax3.arrow(0, 0.1, 0, 0.5, head_width=0.1, head_length=0.05, fc='green', ec='green')
ax3.arrow(0, -0.1, 0, -0.5, head_width=0.1, head_length=0.05, fc='green', ec='green')

ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.set_title('Planar Symmetry (Pillbox)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_198_gaussian_surfaces.png', dpi=150, bbox_inches='tight')
plt.show()

print("Day 198: Gauss's Law Visualizations Complete")
print("="*50)
print("Key results:")
print(f"• Sphere (r > R): E = kQ/r² (same as point charge)")
print(f"• Sphere (r < R): E = kQr/R³ (linear in r)")
print(f"• Cylinder: E = λ/(2πε₀s) (falls as 1/s)")
print(f"• Infinite plane: E = σ/(2ε₀) (constant!)")
```

---

## Summary

### Key Formulas

| Geometry | Field Expression |
|----------|------------------|
| Point charge | $E = \frac{Q}{4\pi\varepsilon_0 r^2}$ |
| Sphere (outside) | $E = \frac{Q}{4\pi\varepsilon_0 r^2}$ |
| Sphere (inside) | $E = \frac{Qr}{4\pi\varepsilon_0 R^3}$ |
| Infinite line | $E = \frac{\lambda}{2\pi\varepsilon_0 s}$ |
| Infinite plane | $E = \frac{\sigma}{2\varepsilon_0}$ |

### Main Takeaways

1. **Gauss's law** relates flux through closed surface to enclosed charge
2. **Choose Gaussian surfaces** where $E$ is constant or perpendicular
3. **Spherical symmetry:** $\oint E\,dA = E \cdot 4\pi r^2$
4. **Cylindrical symmetry:** $\oint E\,dA = E \cdot 2\pi s L$
5. **Planar symmetry:** $\oint E\,dA = 2EA$
6. **Inside conductors:** $E = 0$ in equilibrium

---

## Daily Checklist

- [ ] I can define electric flux through a surface
- [ ] I can state Gauss's law in integral and differential form
- [ ] I can apply Gauss's law to spherical problems
- [ ] I can apply Gauss's law to cylindrical problems
- [ ] I can apply Gauss's law to planar problems

---

## Preview: Day 199

Tomorrow we introduce the **electrostatic potential** $\phi$, which provides an even more powerful approach. We'll derive Poisson's equation and see how scalar potentials simplify vector problems.

---

*"Gauss's law is the most elegant formulation of electrostatics — it encapsulates Coulomb's law in the language of geometry."*

---

**Next:** Day 199 — Electrostatic Potential
