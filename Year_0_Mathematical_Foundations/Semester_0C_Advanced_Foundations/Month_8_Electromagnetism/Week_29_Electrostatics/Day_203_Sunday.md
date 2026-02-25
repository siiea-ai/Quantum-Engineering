# Day 203: Dielectrics & Week 29 Review

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 11:30 AM | 2.5 hours | Theory: Dielectrics & Polarization |
| Late Morning | 11:30 AM - 12:30 PM | 1 hour | Problem Set A |
| Afternoon | 2:00 PM - 4:00 PM | 2 hours | Problem Set B |
| Evening | 7:00 PM - 8:30 PM | 1.5 hours | Week 29 Review & Assessment |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of Day 203, you will be able to:

1. Explain polarization and bound charges in dielectrics
2. Apply boundary conditions at dielectric interfaces
3. Solve electrostatic problems with dielectrics
4. Understand the dielectric constant and permittivity
5. Synthesize all Week 29 topics
6. Connect dielectric theory to quantum polarizability

---

## Core Content: Dielectrics

### 1. Polarization

When an electric field is applied to a dielectric material, the bound charges shift slightly:
- Electrons shift opposite to $\mathbf{E}$
- Positive nuclei shift along $\mathbf{E}$

**Polarization** $\mathbf{P}$: Electric dipole moment per unit volume

$$\boxed{\mathbf{P} = \frac{\sum \mathbf{p}_i}{\Delta V}}$$

Units: C/m²

### 2. Bound Charges

Polarization creates **bound charges** where polarization varies:

**Volume bound charge:**
$$\boxed{\rho_b = -\nabla \cdot \mathbf{P}}$$

**Surface bound charge:**
$$\boxed{\sigma_b = \mathbf{P} \cdot \hat{\mathbf{n}}}$$

These are real charges, just bound to molecules rather than free to move.

### 3. The Electric Displacement Field

To handle both free and bound charges, introduce **displacement** $\mathbf{D}$:

$$\boxed{\mathbf{D} = \varepsilon_0\mathbf{E} + \mathbf{P}}$$

**Gauss's law for $\mathbf{D}$:**
$$\nabla \cdot \mathbf{D} = \rho_f$$

where $\rho_f$ is the free charge density.

### 4. Linear Dielectrics

For most materials, polarization is proportional to field:

$$\mathbf{P} = \varepsilon_0\chi_e\mathbf{E}$$

where $\chi_e$ is the **electric susceptibility**.

Then:
$$\mathbf{D} = \varepsilon_0(1 + \chi_e)\mathbf{E} = \varepsilon_0\varepsilon_r\mathbf{E} = \varepsilon\mathbf{E}$$

**Dielectric constant (relative permittivity):**
$$\boxed{\varepsilon_r = 1 + \chi_e = \frac{\varepsilon}{\varepsilon_0}}$$

| Material | $\varepsilon_r$ |
|----------|-----------------|
| Vacuum | 1 |
| Air | 1.0006 |
| Water | 80 |
| Glass | 4-10 |
| Silicon | 12 |

### 5. Boundary Conditions

At an interface between two dielectrics:

**Normal component:**
$$D_{1n} - D_{2n} = \sigma_f$$ (free surface charge)

If no free charge: $\varepsilon_1 E_{1n} = \varepsilon_2 E_{2n}$

**Tangential component:**
$$E_{1t} = E_{2t}$$

### 6. Energy in Dielectrics

$$\boxed{U = \frac{1}{2}\int \mathbf{D}\cdot\mathbf{E}\,d^3r = \frac{1}{2}\int \varepsilon E^2\,d^3r}$$

### 7. Capacitor with Dielectric

For a parallel plate capacitor filled with dielectric:

$$C = \varepsilon_r C_0 = \frac{\varepsilon_r\varepsilon_0 A}{d}$$

The capacitance increases by factor $\varepsilon_r$.

---

## Quantum Mechanics Connection

### Atomic Polarizability

At the quantum level, an external field mixes ground and excited states:

$$|\psi\rangle = |0\rangle + \sum_{n>0}\frac{\langle n|\hat{d}\cdot\mathbf{E}|0\rangle}{E_0 - E_n}|n\rangle$$

where $\hat{d} = -e\mathbf{r}$ is the dipole operator.

**Polarizability:**
$$\alpha = 2\sum_{n>0}\frac{|\langle n|\hat{d}|0\rangle|^2}{E_n - E_0}$$

This is second-order perturbation theory!

### Connection to Refractive Index

$$n^2 = \varepsilon_r$$

The speed of light in matter: $v = c/n$

Dispersion (frequency-dependent $n$) comes from quantum resonances.

### Kramers-Kronig Relations

The real and imaginary parts of $\varepsilon(\omega)$ are related by:

$$\varepsilon_1(\omega) - 1 = \frac{2}{\pi}\mathcal{P}\int_0^\infty\frac{\omega'\varepsilon_2(\omega')}{\omega'^2 - \omega^2}d\omega'$$

This connects absorption ($\varepsilon_2$) to dispersion ($\varepsilon_1$) — a result from complex analysis (Month 7)!

---

## Week 29 Review

### Concept Map

```
                        ELECTROSTATICS
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   FOUNDATIONS          TECHNIQUES           APPLICATIONS
        │                    │                    │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │         │          │         │          │         │
Coulomb's  Gauss's  Separation  Method   Multipole  Dielectrics
  Law       Law    of Variables of Images Expansion
   │         │          │         │          │         │
   ↓         ↓          ↓         ↓          ↓         ↓
Superposition Symmetry Spherical  Boundary  Dipole   Polarization
             Problems  Harmonics Conditions  Field
```

### Key Formulas Summary

| Topic | Key Formula |
|-------|-------------|
| Coulomb's law | $\mathbf{F} = \frac{q_1 q_2}{4\pi\varepsilon_0 r^2}\hat{\mathbf{r}}$ |
| Gauss's law | $\oint\mathbf{E}\cdot d\mathbf{a} = Q_{\text{enc}}/\varepsilon_0$ |
| Potential | $\phi = -\int_\infty^{\mathbf{r}}\mathbf{E}\cdot d\boldsymbol{\ell}$ |
| Poisson | $\nabla^2\phi = -\rho/\varepsilon_0$ |
| Image charge (plane) | $q' = -q$ at $z = -d$ |
| Image charge (sphere) | $q' = -qR/a$ at $r = R^2/a$ |
| Dipole potential | $\phi = \frac{p\cos\theta}{4\pi\varepsilon_0 r^2}$ |
| Dielectric | $\mathbf{D} = \varepsilon\mathbf{E}$ |

---

## Problem Set A: Core Techniques

### A1: Gauss's Law
A spherical shell of inner radius $a$ and outer radius $b$ has uniform volume charge density $\rho$. Find the electric field in all regions.

### A2: Potential
Calculate the potential along the axis of a uniformly charged disk of radius $R$ and surface charge density $\sigma$.

### A3: Separation of Variables
Solve Laplace's equation in 2D for a semi-infinite strip $0 < x < a$, $y > 0$ with boundary conditions $\phi(0,y) = \phi(a,y) = 0$ and $\phi(x,0) = V_0\sin(\pi x/a)$.

### A4: Method of Images
A charge $q$ is at distance $d$ from the center of a grounded conducting sphere of radius $R$. Find the force on the charge.

---

## Problem Set B: Applications

### B1: Multipole Expansion
A charge distribution has monopole moment $Q = 0$, dipole moment $p = 10^{-29}$ C·m, and quadrupole moment $Q_{zz} = 10^{-38}$ C·m². At what distance does the dipole term equal the quadrupole term (on axis)?

### B2: Dielectric Sphere
A dielectric sphere of radius $R$ and permittivity $\varepsilon$ is placed in a uniform external field $\mathbf{E}_0$. Find the field inside the sphere.

### B3: Energy
Calculate the energy stored in the electric field of a uniformly charged solid sphere of radius $R$ and total charge $Q$.

### B4: Comprehensive Problem
Two concentric conducting spherical shells have radii $a$ and $b$ ($b > a$). The region between them is filled with two dielectrics: $\varepsilon_1$ for $a < r < c$ and $\varepsilon_2$ for $c < r < b$. If a potential difference $V_0$ is applied, find:
(a) $\mathbf{E}$ in each region
(b) The bound surface charge at $r = c$
(c) The capacitance

---

## Solutions to Selected Problems

### Solution A2
On the axis at height $z$ above the disk:

$$\phi(z) = \frac{\sigma}{2\varepsilon_0}\left(\sqrt{R^2 + z^2} - |z|\right)$$

For $z \gg R$: $\phi \approx Q/(4\pi\varepsilon_0 z)$ (point charge)

### Solution A3
The solution is simply:
$$\phi(x,y) = V_0\sin\left(\frac{\pi x}{a}\right)e^{-\pi y/a}$$

Verification: Satisfies Laplace, and all boundary conditions.

### Solution B2
Inside a dielectric sphere in uniform field:

$$\mathbf{E}_{\text{in}} = \frac{3\varepsilon_0}{\varepsilon + 2\varepsilon_0}\mathbf{E}_0$$

The field is uniform inside and reduced compared to $\mathbf{E}_0$.

### Solution B3
For a uniformly charged sphere:

$$U = \frac{3Q^2}{20\pi\varepsilon_0 R}$$

This is 3/5 of the energy of a spherical shell (same $Q$, $R$).

---

## Self-Assessment Checklist

### Foundations (Days 197-198)
- [ ] I can apply Coulomb's law to discrete and continuous distributions
- [ ] I can identify symmetry and choose appropriate Gaussian surfaces
- [ ] I can calculate fields using Gauss's law

### Potentials (Days 199-200)
- [ ] I understand the relationship between $\mathbf{E}$ and $\phi$
- [ ] I can solve Laplace's equation using separation of variables
- [ ] I recognize spherical harmonics and Legendre polynomials

### Techniques (Days 201-202)
- [ ] I can apply the method of images to plane and sphere problems
- [ ] I can calculate multipole moments
- [ ] I understand the distance dependence of multipole potentials

### Materials (Day 203)
- [ ] I understand polarization and bound charges
- [ ] I can apply boundary conditions at dielectric interfaces
- [ ] I can connect dielectric response to quantum polarizability

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

# Week 29 comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Constants
eps0 = 8.854e-12
k = 1/(4*np.pi*eps0)

# ========== Plot 1: Gauss's Law - Sphere ==========
ax1 = axes[0, 0]
R = 1.0
Q = 1e-9
r = np.linspace(0.01, 3, 200)

E_inside = k * Q * r / R**3
E_outside = k * Q / r**2
E = np.where(r < R, E_inside, E_outside)

ax1.plot(r, E * 1e6, 'b-', linewidth=2)
ax1.axvline(x=R, color='r', linestyle='--', label='r = R')
ax1.set_xlabel('r (m)')
ax1.set_ylabel('E (μN/C)')
ax1.set_title('Gauss: Uniformly Charged Sphere')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ========== Plot 2: Separation of Variables ==========
ax2 = axes[0, 1]
a = 1.0
V0 = 100

x = np.linspace(0, a, 50)
y = np.linspace(0, 2, 50)
X, Y = np.meshgrid(x, y)

phi = V0 * np.sin(np.pi * X / a) * np.exp(-np.pi * Y / a)

contour = ax2.contourf(X, Y, phi, levels=20, cmap='hot')
plt.colorbar(contour, ax=ax2, label='φ (V)')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('Separation of Variables')

# ========== Plot 3: Method of Images ==========
ax3 = axes[0, 2]
q = 1e-9
d = 1.0

x = np.linspace(-2, 2, 100)
z = np.linspace(0.01, 3, 100)
X, Z = np.meshgrid(x, z)

r1 = np.sqrt(X**2 + (Z-d)**2)
r2 = np.sqrt(X**2 + (Z+d)**2)
phi = k * q * (1/r1 - 1/r2)
phi = np.clip(phi * 1e9, -5, 5)

contour = ax3.contourf(X, Z, phi, levels=30, cmap='RdBu_r')
plt.colorbar(contour, ax=ax3, label='φ (nV·m)')
ax3.fill_between([-2, 2], [0, 0], [-0.1, -0.1], color='gray')
ax3.plot(0, d, 'ro', markersize=10)
ax3.set_xlabel('x (m)')
ax3.set_ylabel('z (m)')
ax3.set_title('Method of Images')
ax3.set_ylim(-0.1, 3)

# ========== Plot 4: Multipole Comparison ==========
ax4 = axes[1, 0]
r = np.linspace(0.5, 5, 100)

ax4.loglog(r, 1/r, 'b-', linewidth=2, label='Monopole (1/r)')
ax4.loglog(r, 1/r**2, 'r-', linewidth=2, label='Dipole (1/r²)')
ax4.loglog(r, 1/r**3, 'g-', linewidth=2, label='Quadrupole (1/r³)')

ax4.set_xlabel('r (m)')
ax4.set_ylabel('φ (normalized)')
ax4.set_title('Multipole Distance Dependence')
ax4.legend()
ax4.grid(True, alpha=0.3, which='both')

# ========== Plot 5: Dielectric in Capacitor ==========
ax5 = axes[1, 1]
d_cap = 1.0
V0 = 100

# Without dielectric
z1 = np.linspace(0, d_cap, 50)
phi1 = V0 * z1 / d_cap

# With dielectric (half filled)
epsilon_r = 4
z2a = np.linspace(0, d_cap/2, 25)
z2b = np.linspace(d_cap/2, d_cap, 25)

# E1 = E2/epsilon_r at interface, V0 = E1*d/2 + E2*d/2
E2 = 2*V0 / (d_cap * (1 + 1/epsilon_r))
E1 = E2 / epsilon_r

phi2a = E1 * z2a
phi2b = E1 * d_cap/2 + E2 * (z2b - d_cap/2)

ax5.plot(z1, phi1, 'b-', linewidth=2, label='No dielectric')
ax5.plot(z2a, phi2a, 'r-', linewidth=2, label='With dielectric (ε_r=4)')
ax5.plot(z2b, phi2b, 'r-', linewidth=2)
ax5.axvline(x=d_cap/2, color='g', linestyle='--', alpha=0.5)
ax5.fill_between([0, d_cap/2], [0, 0], [V0, V0], alpha=0.2, color='yellow')
ax5.set_xlabel('z (m)')
ax5.set_ylabel('φ (V)')
ax5.set_title('Capacitor with Dielectric')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ========== Plot 6: Spherical harmonics (angular solutions) ==========
ax6 = axes[1, 2]
theta = np.linspace(0, np.pi, 100)

# Legendre polynomials
P0 = np.ones_like(theta)
P1 = np.cos(theta)
P2 = 0.5 * (3*np.cos(theta)**2 - 1)
P3 = 0.5 * (5*np.cos(theta)**3 - 3*np.cos(theta))

ax6.plot(theta*180/np.pi, P0, 'b-', linewidth=2, label='$P_0$')
ax6.plot(theta*180/np.pi, P1, 'r-', linewidth=2, label='$P_1$')
ax6.plot(theta*180/np.pi, P2, 'g-', linewidth=2, label='$P_2$')
ax6.plot(theta*180/np.pi, P3, 'm-', linewidth=2, label='$P_3$')

ax6.axhline(y=0, color='k', linewidth=0.5)
ax6.set_xlabel('θ (degrees)')
ax6.set_ylabel('$P_l(\\cos\\theta)$')
ax6.set_title('Legendre Polynomials (Angular Solutions)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_203_week29_review.png', dpi=150, bbox_inches='tight')
plt.show()

# Dielectric constants table
print("\nDielectric Constants (Relative Permittivity)")
print("="*50)
materials = {
    'Vacuum': 1.0,
    'Air': 1.0006,
    'Teflon': 2.1,
    'Polystyrene': 2.6,
    'Paper': 3.7,
    'Glass': 5.0,
    'Mica': 7.0,
    'Silicon': 11.7,
    'Water': 80.1,
    'Titanium dioxide': 100
}

for mat, eps in materials.items():
    print(f"{mat:20} ε_r = {eps:8.1f}")

print("\nWeek 29 Complete!")
```

---

## Summary

### Week 29 Key Takeaways

1. **Coulomb's law** is the foundation — inverse-square, central force
2. **Gauss's law** provides elegant solutions for symmetric problems
3. **Potential** reduces vector problems to scalar problems
4. **Separation of variables** solves boundary value problems systematically
5. **Method of images** handles conductor boundaries cleverly
6. **Multipole expansion** organizes potentials by distance dependence
7. **Dielectrics** modify fields through polarization

### Connection to Quantum Mechanics

| Classical | Quantum |
|-----------|---------|
| Coulomb potential | Hydrogen atom |
| Spherical harmonics | Angular momentum eigenstates |
| Boundary value problems | Particle in a box |
| Polarizability | Second-order perturbation |
| Dielectric function | Optical response |

---

## Preview: Week 30

Next week covers **Magnetostatics**:
- Day 204: Lorentz Force & Magnetic Field
- Day 205: Biot-Savart Law
- Day 206: Ampère's Law
- Day 207: Magnetic Vector Potential
- Day 208: Magnetic Dipoles
- Day 209: Magnetization & Materials
- Day 210: Week Review

We'll see how magnetic phenomena have a very different structure from electrostatics — no magnetic monopoles!

---

*"The study of electrostatics, from Coulomb to dielectrics, provides the complete toolkit for understanding electric phenomena in matter."*

---

**Week 29 Complete!**

**Next:** Week 30 — Magnetostatics
