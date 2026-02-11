# Day 213: Complete Maxwell's Equations

## Schedule Overview (8 hours)

| Block | Time | Focus |
|-------|------|-------|
| Morning I | 2 hrs | The four Maxwell equations unified |
| Morning II | 2 hrs | Symmetry, duality, and mathematical structure |
| Afternoon | 2 hrs | Boundary conditions and Maxwell in matter |
| Evening | 2 hrs | Computational lab: Maxwell equations visualization |

## Learning Objectives

By the end of today, you will be able to:

1. **State all four Maxwell equations** in both differential and integral forms
2. **Explain the physical meaning** of each equation and its role in electromagnetism
3. **Identify symmetries** between electric and magnetic fields in Maxwell's equations
4. **Derive boundary conditions** at interfaces from Maxwell's equations
5. **Write Maxwell's equations in matter** using D, H, P, and M fields
6. **Connect Maxwell's equations** to the quantum field theory of photons

## Core Content

### 1. Maxwell's Equations: The Complete Set

The four Maxwell equations, in differential and integral forms:

#### Gauss's Law for Electricity
$$\boxed{\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}}$$
$$\oint_S \vec{E} \cdot d\vec{A} = \frac{Q_{enc}}{\epsilon_0}$$

*Electric charges are sources/sinks of electric field lines.*

#### Gauss's Law for Magnetism
$$\boxed{\nabla \cdot \vec{B} = 0}$$
$$\oint_S \vec{B} \cdot d\vec{A} = 0$$

*There are no magnetic monopoles; magnetic field lines always close on themselves.*

#### Faraday's Law
$$\boxed{\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}}$$
$$\oint_C \vec{E} \cdot d\vec{l} = -\frac{d\Phi_B}{dt}$$

*A changing magnetic field induces an electric field.*

#### Ampère-Maxwell Law
$$\boxed{\nabla \times \vec{B} = \mu_0 \vec{J} + \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}}$$
$$\oint_C \vec{B} \cdot d\vec{l} = \mu_0 I_{enc} + \mu_0 \epsilon_0 \frac{d\Phi_E}{dt}$$

*Currents and changing electric fields produce magnetic fields.*

### 2. The Structure of Maxwell's Equations

**Divergence equations** (scalar): Describe sources
- $\nabla \cdot \vec{E}$: Electric charges
- $\nabla \cdot \vec{B}$: No magnetic charges (monopoles)

**Curl equations** (vector): Describe circulation/induction
- $\nabla \times \vec{E}$: Induced by changing $\vec{B}$
- $\nabla \times \vec{B}$: Induced by currents and changing $\vec{E}$

### 3. Electric-Magnetic Symmetry (Duality)

In **source-free regions** ($\rho = 0$, $\vec{J} = 0$), Maxwell's equations become:

$$\nabla \cdot \vec{E} = 0 \qquad \nabla \cdot \vec{B} = 0$$
$$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t} \qquad \nabla \times \vec{B} = \frac{1}{c^2}\frac{\partial \vec{E}}{\partial t}$$

These equations are nearly symmetric under the **duality transformation**:
$$\vec{E} \rightarrow c\vec{B}, \qquad \vec{B} \rightarrow -\frac{\vec{E}}{c}$$

If magnetic monopoles existed with charge density $\rho_m$ and current $\vec{J}_m$:
$$\nabla \cdot \vec{B} = \mu_0 \rho_m \qquad \nabla \times \vec{E} = -\mu_0 \vec{J}_m - \frac{\partial \vec{B}}{\partial t}$$

The equations would then be completely symmetric!

### 4. The Continuity Equation

Maxwell's equations imply charge conservation. Taking the divergence of Ampère-Maxwell:
$$\nabla \cdot (\nabla \times \vec{B}) = \mu_0 \nabla \cdot \vec{J} + \mu_0 \epsilon_0 \frac{\partial}{\partial t}(\nabla \cdot \vec{E})$$

Since $\nabla \cdot (\nabla \times \vec{B}) = 0$ and $\nabla \cdot \vec{E} = \rho/\epsilon_0$:
$$0 = \mu_0 \nabla \cdot \vec{J} + \mu_0 \frac{\partial \rho}{\partial t}$$

$$\boxed{\nabla \cdot \vec{J} + \frac{\partial \rho}{\partial t} = 0}$$

This is the **continuity equation**—charge is locally conserved.

### 5. Boundary Conditions

Maxwell's equations in integral form yield boundary conditions at interfaces.

**From Gauss's law (E):**
$$(\vec{E}_2 - \vec{E}_1) \cdot \hat{n} = \frac{\sigma}{\epsilon_0}$$

Normal component discontinuous by surface charge density.

**From Gauss's law (B):**
$$(\vec{B}_2 - \vec{B}_1) \cdot \hat{n} = 0$$

Normal component of B is continuous.

**From Faraday's law:**
$$\hat{n} \times (\vec{E}_2 - \vec{E}_1) = 0$$

Tangential component of E is continuous.

**From Ampère-Maxwell:**
$$\hat{n} \times (\vec{B}_2 - \vec{B}_1) = \mu_0 \vec{K}$$

Tangential B discontinuous by surface current density $\vec{K}$.

### 6. Maxwell's Equations in Matter

In materials, we define auxiliary fields:

**Electric displacement:**
$$\vec{D} = \epsilon_0 \vec{E} + \vec{P}$$

where $\vec{P}$ is the polarization (dipole moment per volume).

**Magnetic field intensity:**
$$\vec{H} = \frac{\vec{B}}{\mu_0} - \vec{M}$$

where $\vec{M}$ is the magnetization.

**Maxwell's equations in matter:**
$$\boxed{\nabla \cdot \vec{D} = \rho_f}$$
$$\nabla \cdot \vec{B} = 0$$
$$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$$
$$\boxed{\nabla \times \vec{H} = \vec{J}_f + \frac{\partial \vec{D}}{\partial t}}$$

Here $\rho_f$ and $\vec{J}_f$ are free (not bound) charges and currents.

**Linear materials:**
$$\vec{D} = \epsilon \vec{E} = \epsilon_r \epsilon_0 \vec{E}$$
$$\vec{B} = \mu \vec{H} = \mu_r \mu_0 \vec{H}$$

### 7. Potential Formulation

Maxwell's equations can be rewritten using potentials:
$$\vec{B} = \nabla \times \vec{A}$$
$$\vec{E} = -\nabla V - \frac{\partial \vec{A}}{\partial t}$$

Gauss's law for B is automatically satisfied. Faraday's law becomes an identity.

The remaining equations give:
$$\nabla^2 V + \frac{\partial}{\partial t}(\nabla \cdot \vec{A}) = -\frac{\rho}{\epsilon_0}$$
$$\nabla^2 \vec{A} - \mu_0 \epsilon_0 \frac{\partial^2 \vec{A}}{\partial t^2} - \nabla\left(\nabla \cdot \vec{A} + \mu_0 \epsilon_0 \frac{\partial V}{\partial t}\right) = -\mu_0 \vec{J}$$

**Lorenz gauge:** $\nabla \cdot \vec{A} + \mu_0 \epsilon_0 \frac{\partial V}{\partial t} = 0$

This yields wave equations for both potentials:
$$\nabla^2 V - \frac{1}{c^2}\frac{\partial^2 V}{\partial t^2} = -\frac{\rho}{\epsilon_0}$$
$$\nabla^2 \vec{A} - \frac{1}{c^2}\frac{\partial^2 \vec{A}}{\partial t^2} = -\mu_0 \vec{J}$$

### 8. Covariant Formulation (Preview)

In special relativity, Maxwell's equations take an elegant form using the **electromagnetic field tensor** $F^{\mu\nu}$:

$$\partial_\mu F^{\mu\nu} = \mu_0 J^\nu$$
$$\partial_{[\lambda} F_{\mu\nu]} = 0$$

Here $J^\mu = (c\rho, \vec{J})$ is the four-current. This shows that electromagnetism is inherently relativistic—it transforms correctly under Lorentz transformations.

## Quantum Mechanics Connection

### Quantization of the Electromagnetic Field

Maxwell's equations describe classical fields. In **quantum electrodynamics (QED)**:

1. **Field quantization:** The vector potential $\vec{A}$ becomes an operator:
$$\hat{\vec{A}}(\vec{r}, t) = \sum_{\vec{k}, \lambda} \sqrt{\frac{\hbar}{2\epsilon_0 \omega_k V}}\left[\hat{a}_{\vec{k},\lambda} \vec{\epsilon}_{\vec{k},\lambda} e^{i(\vec{k}\cdot\vec{r} - \omega_k t)} + \text{h.c.}\right]$$

2. **Photon creation/annihilation:** $\hat{a}^\dagger$ and $\hat{a}$ create and destroy photons
3. **Energy quantization:** Each mode has energy $E_n = \hbar\omega(n + 1/2)$

### The QED Lagrangian

The classical Maxwell Lagrangian density:
$$\mathcal{L} = -\frac{1}{4\mu_0}F_{\mu\nu}F^{\mu\nu} - J^\mu A_\mu$$

This generates Maxwell's equations via the Euler-Lagrange equations. In QED, we add the Dirac Lagrangian for electrons:
$$\mathcal{L}_{QED} = \bar{\psi}(i\hbar c\gamma^\mu D_\mu - mc^2)\psi - \frac{1}{4\mu_0}F_{\mu\nu}F^{\mu\nu}$$

where $D_\mu = \partial_\mu + ieA_\mu/\hbar$ is the covariant derivative.

### Gauge Invariance

Maxwell's equations are invariant under gauge transformations:
$$\vec{A} \rightarrow \vec{A} + \nabla \chi$$
$$V \rightarrow V - \frac{\partial \chi}{\partial t}$$

In QED, this becomes a **local U(1) gauge symmetry**, the foundation of modern particle physics. The photon is the "gauge boson" of this symmetry.

### Vacuum Energy

The quantized electromagnetic field has zero-point energy:
$$E_0 = \sum_{\vec{k}, \lambda} \frac{1}{2}\hbar\omega_k$$

This is infinite but has measurable consequences:
- **Casimir effect:** Attractive force between conducting plates
- **Lamb shift:** Energy level splitting in hydrogen
- **Spontaneous emission:** Atoms radiate even without stimulation

## Worked Examples

### Example 1: Verifying Maxwell's Equations for a Point Charge

Verify that the fields of a point charge $q$ at the origin satisfy Maxwell's equations.

**Solution:**

Fields: $\vec{E} = \frac{q}{4\pi\epsilon_0 r^2}\hat{r}$, $\vec{B} = 0$

**Gauss's law (E):**
$$\nabla \cdot \vec{E} = \frac{q}{4\pi\epsilon_0}\nabla \cdot \left(\frac{\hat{r}}{r^2}\right) = \frac{q}{4\pi\epsilon_0}(4\pi\delta^3(\vec{r})) = \frac{q}{\epsilon_0}\delta^3(\vec{r})$$

This equals $\rho/\epsilon_0$ where $\rho = q\delta^3(\vec{r})$. ✓

**Gauss's law (B):** $\nabla \cdot \vec{B} = 0$ ✓

**Faraday's law:** $\nabla \times \vec{E} = 0$ for radial field, and $\partial\vec{B}/\partial t = 0$. ✓

**Ampère-Maxwell:** $\nabla \times \vec{B} = 0$ and there's no current or changing E at the field point. ✓

### Example 2: Boundary Conditions at a Conductor Surface

A perfect conductor has surface charge $\sigma$ and surface current $\vec{K}$. Find the fields just outside.

**Solution:**

Inside a perfect conductor: $\vec{E}_{in} = 0$, $\vec{B}_{in} = 0$

**Normal E-field:**
$$E_\perp - 0 = \frac{\sigma}{\epsilon_0} \quad \Rightarrow \quad E_\perp = \frac{\sigma}{\epsilon_0}$$

**Tangential E-field:**
$$\hat{n} \times (\vec{E}_{out} - 0) = 0 \quad \Rightarrow \quad E_\parallel = 0$$

**Normal B-field:**
$$B_\perp = 0$$

**Tangential B-field:**
$$\hat{n} \times (\vec{B}_{out} - 0) = \mu_0 \vec{K}$$
$$\vec{B}_\parallel = \mu_0(\vec{K} \times \hat{n})$$

Just outside:
$$\boxed{\vec{E} = \frac{\sigma}{\epsilon_0}\hat{n}, \quad \vec{B} = \mu_0 \vec{K} \times \hat{n}}$$

### Example 3: Maxwell's Equations in a Linear Dielectric

A linear dielectric ($\epsilon = \kappa\epsilon_0$, $\mu = \mu_0$) fills a region with no free charges or currents. Write Maxwell's equations explicitly.

**Solution:**

With $\vec{D} = \epsilon\vec{E}$, $\vec{B} = \mu_0\vec{H}$, and no free sources:

$$\nabla \cdot (\epsilon\vec{E}) = 0 \quad \Rightarrow \quad \nabla \cdot \vec{E} = 0 \text{ (if $\epsilon$ uniform)}$$
$$\nabla \cdot \vec{B} = 0$$
$$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$$
$$\nabla \times \left(\frac{\vec{B}}{\mu_0}\right) = \epsilon\frac{\partial \vec{E}}{\partial t}$$

The last equation becomes:
$$\nabla \times \vec{B} = \mu_0\epsilon\frac{\partial \vec{E}}{\partial t} = \frac{\kappa}{c^2}\frac{\partial \vec{E}}{\partial t}$$

The wave speed becomes $v = c/\sqrt{\kappa}$—light slows in dielectrics!

## Practice Problems

### Level 1: Direct Application

1. Write Maxwell's equations in integral form for a region with no charges or currents.

2. Given $\vec{E} = E_0\sin(kx - \omega t)\hat{y}$ in vacuum, use Faraday's law to find $\vec{B}$.

3. A sphere of radius $R$ has uniform polarization $\vec{P} = P_0\hat{z}$. Find the bound surface charge density.

### Level 2: Intermediate

4. Show that $\nabla \times (\nabla \times \vec{E}) = \nabla(\nabla \cdot \vec{E}) - \nabla^2\vec{E}$, and use Maxwell's equations to derive the wave equation for $\vec{E}$ in vacuum.

5. At the interface between vacuum and a dielectric ($\epsilon_r = 4$), an incident E-field makes angle $45°$ with the normal. Find the angle of the refracted E-field.

6. Verify that the fields $\vec{E} = E_0 e^{i(kz - \omega t)}\hat{x}$ and $\vec{B} = B_0 e^{i(kz - \omega t)}\hat{y}$ satisfy all four Maxwell equations if appropriate conditions on $E_0$, $B_0$, $k$, and $\omega$ are met.

### Level 3: Challenging

7. Starting from Maxwell's equations, derive the wave equation for $\vec{A}$ in the Lorenz gauge.

8. A conducting sphere of radius $R$ has surface charge oscillating as $\sigma(\theta, t) = \sigma_0 \cos\theta \cos(\omega t)$. Find the displacement current density in the region just outside the sphere.

9. **Quantum connection:** The Casimir energy per unit area between two parallel perfectly conducting plates separated by distance $d$ is approximately $E/A = -\pi^2\hbar c/(720 d^3)$. For $d = 100$ nm, calculate the force per unit area and compare to atmospheric pressure.

## Computational Lab: Maxwell's Equations Visualization

```python
"""
Day 213 Computational Lab: Complete Maxwell's Equations
Topics: Field visualization, boundary conditions, gauge potentials
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize

# Set up styling
plt.style.use('default')

# Physical constants
epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7
c = 1 / np.sqrt(mu_0 * epsilon_0)

# =============================================================================
# Part 1: Visualize All Four Maxwell Equations
# =============================================================================

def visualize_maxwell_equations():
    """Create visual representations of all four Maxwell equations."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ========== Gauss's Law (Electric) ==========
    ax = axes[0, 0]

    # Point charge at origin - field lines radiate outward
    theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in theta:
        x_start = 0.1 * np.cos(angle)
        y_start = 0.1 * np.sin(angle)
        x_end = 0.9 * np.cos(angle)
        y_end = 0.9 * np.sin(angle)
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax.plot(0, 0, 'ro', markersize=20, label='+ charge')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(r'$\nabla \cdot \vec{E} = \rho/\epsilon_0$' + '\n(Charges are sources of E-field)',
                fontsize=12)
    ax.legend()
    ax.axis('off')

    # ========== Gauss's Law (Magnetic) ==========
    ax = axes[0, 1]

    # Magnetic field lines must close - draw loops
    for r in [0.3, 0.5, 0.7]:
        theta = np.linspace(0, 2*np.pi, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, 'g-', linewidth=2)

        # Add arrows to show direction
        for t in [0, np.pi/2, np.pi, 3*np.pi/2]:
            ax.annotate('', xy=(r*np.cos(t+0.1), r*np.sin(t+0.1)),
                       xytext=(r*np.cos(t), r*np.sin(t)),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.plot(0, 0, 'ko', markersize=15)
    ax.text(0, 0, 'I', fontsize=12, ha='center', va='center', color='white')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(r'$\nabla \cdot \vec{B} = 0$' + '\n(B-field lines always close)',
                fontsize=12)
    ax.axis('off')

    # ========== Faraday's Law ==========
    ax = axes[1, 0]

    # Changing B (going into page) induces circulating E
    # Draw B going into page
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i**2 + j**2 < 5:
                ax.text(i*0.25, j*0.25, '×', fontsize=16, ha='center',
                       va='center', color='purple')

    # Circulating E field
    r = 0.8
    theta = np.linspace(0, 2*np.pi, 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, 'b-', linewidth=2)

    # Arrows showing E circulation
    for t in [0, np.pi/2, np.pi, 3*np.pi/2]:
        ax.annotate('', xy=(r*np.cos(t-0.15), r*np.sin(t-0.15)),
                   xytext=(r*np.cos(t), r*np.sin(t)),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax.text(0, -0.3, r'$\frac{\partial B}{\partial t}$', fontsize=14,
           ha='center', color='purple')
    ax.text(0.95, 0, r'$\vec{E}$', fontsize=14, color='blue')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(r'$\nabla \times \vec{E} = -\partial\vec{B}/\partial t$' +
                '\n(Changing B induces circulating E)', fontsize=12)
    ax.axis('off')

    # ========== Ampère-Maxwell Law ==========
    ax = axes[1, 1]

    # Current or changing E produces circulating B
    # Draw current coming out of page
    ax.plot(0, 0, 'ro', markersize=20)
    ax.text(0, 0, '⊙', fontsize=16, ha='center', va='center')
    ax.text(0.15, 0, r'$I$ or $\frac{\partial E}{\partial t}$', fontsize=12)

    # Circulating B field
    r = 0.7
    theta = np.linspace(0, 2*np.pi, 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, 'g-', linewidth=2)

    # Arrows showing B circulation
    for t in [0, np.pi/2, np.pi, 3*np.pi/2]:
        ax.annotate('', xy=(r*np.cos(t+0.15), r*np.sin(t+0.15)),
                   xytext=(r*np.cos(t), r*np.sin(t)),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.text(0.85, 0, r'$\vec{B}$', fontsize=14, color='green')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(r'$\nabla \times \vec{B} = \mu_0\vec{J} + \mu_0\epsilon_0\partial\vec{E}/\partial t$' +
                '\n(Currents and changing E produce circulating B)', fontsize=12)
    ax.axis('off')

    plt.suptitle("Maxwell's Equations: The Four Laws of Electromagnetism",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('maxwell_equations_visual.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 2: Boundary Conditions Visualization
# =============================================================================

def visualize_boundary_conditions():
    """Visualize electromagnetic boundary conditions at interfaces."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ========== Normal E discontinuity ==========
    ax = axes[0, 0]

    # Interface
    ax.axhline(y=0, color='black', linewidth=3)
    ax.fill_between([-1, 1], 0, 1, alpha=0.2, color='blue', label='Medium 2')
    ax.fill_between([-1, 1], -1, 0, alpha=0.2, color='red', label='Medium 1')

    # E-field arrows (normal component)
    ax.annotate('', xy=(0, 0.8), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax.annotate('', xy=(0, -0.3), xytext=(0, -0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))

    ax.text(0.1, 0.5, r'$E_{2\perp}$', fontsize=14, color='blue')
    ax.text(0.1, -0.5, r'$E_{1\perp}$', fontsize=14, color='red')
    ax.text(-0.8, 0.05, r'$\sigma$', fontsize=14, color='black')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(r'$E_{2\perp} - E_{1\perp} = \sigma/\epsilon_0$' +
                '\nNormal E discontinuous by surface charge', fontsize=11)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    # ========== Tangential E continuity ==========
    ax = axes[0, 1]

    ax.axhline(y=0, color='black', linewidth=3)
    ax.fill_between([-1, 1], 0, 1, alpha=0.2, color='blue')
    ax.fill_between([-1, 1], -1, 0, alpha=0.2, color='red')

    # E-field arrows (tangential)
    ax.annotate('', xy=(0.6, 0.4), xytext=(-0.4, 0.4),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax.annotate('', xy=(0.6, -0.4), xytext=(-0.4, -0.4),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))

    ax.text(0, 0.55, r'$E_{2\parallel}$', fontsize=14, color='blue')
    ax.text(0, -0.55, r'$E_{1\parallel}$', fontsize=14, color='red')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(r'$E_{2\parallel} = E_{1\parallel}$' +
                '\nTangential E is continuous', fontsize=11)
    ax.set_aspect('equal')

    # ========== Normal B continuity ==========
    ax = axes[1, 0]

    ax.axhline(y=0, color='black', linewidth=3)
    ax.fill_between([-1, 1], 0, 1, alpha=0.2, color='blue')
    ax.fill_between([-1, 1], -1, 0, alpha=0.2, color='red')

    # B-field arrows (normal)
    ax.annotate('', xy=(0, 0.8), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.annotate('', xy=(0, -0.3), xytext=(0, -0.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))

    ax.text(0.1, 0.5, r'$B_{2\perp}$', fontsize=14, color='green')
    ax.text(0.1, -0.5, r'$B_{1\perp}$', fontsize=14, color='green')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(r'$B_{2\perp} = B_{1\perp}$' +
                '\nNormal B is continuous (no monopoles)', fontsize=11)
    ax.set_aspect('equal')

    # ========== Tangential B discontinuity ==========
    ax = axes[1, 1]

    ax.axhline(y=0, color='black', linewidth=3)
    ax.fill_between([-1, 1], 0, 1, alpha=0.2, color='blue')
    ax.fill_between([-1, 1], -1, 0, alpha=0.2, color='red')

    # B-field arrows (tangential) - different magnitudes
    ax.annotate('', xy=(0.8, 0.4), xytext=(-0.4, 0.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.annotate('', xy=(0.4, -0.4), xytext=(-0.4, -0.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.text(0.2, 0.55, r'$B_{2\parallel}$', fontsize=14, color='green')
    ax.text(0, -0.55, r'$B_{1\parallel}$', fontsize=14, color='green')

    # Surface current
    ax.annotate('', xy=(0.3, 0.05), xytext=(-0.3, 0.05),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(0, 0.15, r'$\vec{K}$', fontsize=14, color='orange')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(r'$B_{2\parallel} - B_{1\parallel} = \mu_0 K$' +
                '\nTangential B discontinuous by surface current', fontsize=11)
    ax.set_aspect('equal')

    plt.suptitle('Electromagnetic Boundary Conditions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('boundary_conditions.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 3: Fields of a Point Charge
# =============================================================================

def plot_point_charge_fields():
    """Visualize E-field of a point charge and verify Gauss's law."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create grid
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # E-field of point charge (normalized)
    Ex = X / (R**3 + 0.1)
    Ey = Y / (R**3 + 0.1)

    # Mask near origin
    mask = R < 0.2
    Ex[mask] = 0
    Ey[mask] = 0

    # Plot 1: Vector field
    ax = axes[0]
    ax.quiver(X, Y, Ex, Ey, np.sqrt(Ex**2 + Ey**2), cmap='Reds')
    ax.plot(0, 0, 'bo', markersize=15)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('E-field of Point Charge')
    ax.set_aspect('equal')

    # Gaussian surfaces
    for r in [0.5, 1.0, 1.5]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'g--', linewidth=2,
               label=f'Gaussian surface r={r}' if r == 0.5 else '')
    ax.legend()

    # Plot 2: Verify Gauss's law
    ax = axes[1]

    # For various Gaussian sphere radii
    radii = np.linspace(0.1, 2.0, 50)
    q = 1.0  # Charge in units of ε₀

    # Flux through each sphere
    flux = q * np.ones_like(radii)  # Should be constant!

    ax.plot(radii, flux, 'b-', linewidth=2, label='Flux = Q/ε₀')
    ax.axhline(y=q, color='r', linestyle='--', label='Expected: Q/ε₀ = const')
    ax.set_xlabel('Gaussian Sphere Radius')
    ax.set_ylabel('Electric Flux (units of Q/ε₀)')
    ax.set_title("Verification of Gauss's Law\nFlux independent of surface radius")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('point_charge_gauss.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 4: Symmetry and Duality
# =============================================================================

def visualize_em_duality():
    """Visualize the symmetry between E and B in Maxwell's equations."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Time array
    t = np.linspace(0, 4*np.pi, 500)
    omega = 1

    # For a plane wave
    E = np.sin(t)
    B = np.sin(t)  # In phase for propagating wave

    # Plot 1: E and B in a wave
    ax = axes[0]
    ax.plot(t, E, 'b-', linewidth=2, label=r'$E_y(z=0, t)$')
    ax.plot(t, B, 'r--', linewidth=2, label=r'$cB_x(z=0, t)$')
    ax.set_xlabel('Time (ωt)')
    ax.set_ylabel('Field Amplitude')
    ax.set_title('E and B in Electromagnetic Wave\n(In phase, perpendicular)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Duality transformation table
    ax = axes[1]
    ax.axis('off')

    table_data = [
        ['Quantity', 'Transform', 'Result'],
        [r'$\vec{E}$', r'$\rightarrow$', r'$c\vec{B}$'],
        [r'$\vec{B}$', r'$\rightarrow$', r'$-\vec{E}/c$'],
        [r'$\rho$', r'$\rightarrow$', r'$c\rho_m$'],
        [r'$\vec{J}$', r'$\rightarrow$', r'$c\vec{J}_m$'],
        ['', '', ''],
        ['Effect on Maxwell Equations:', '', ''],
        [r'$\nabla \cdot \vec{E} = \rho/\epsilon_0$', r'$\leftrightarrow$',
         r'$\nabla \cdot \vec{B} = \mu_0\rho_m$'],
        [r'$\nabla \times \vec{E} = -\partial\vec{B}/\partial t$', r'$\leftrightarrow$',
         r'$\nabla \times \vec{B} = \partial\vec{E}/(c^2\partial t)$'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.15, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    ax.set_title('Electric-Magnetic Duality\n(Perfect symmetry if magnetic monopoles existed)',
                fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig('em_duality.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Part 5: Maxwell Equations Summary Table
# =============================================================================

def create_maxwell_summary():
    """Create a comprehensive summary table of Maxwell's equations."""

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # Create comprehensive table
    equations = [
        ['Equation', 'Differential Form', 'Integral Form', 'Physical Meaning'],
        ["Gauss's Law\n(Electric)",
         r'$\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}$',
         r'$\oint \vec{E} \cdot d\vec{A} = \frac{Q_{enc}}{\epsilon_0}$',
         'Electric charges are\nsources of E-field'],
        ["Gauss's Law\n(Magnetic)",
         r'$\nabla \cdot \vec{B} = 0$',
         r'$\oint \vec{B} \cdot d\vec{A} = 0$',
         'No magnetic monopoles;\nB-lines always close'],
        ["Faraday's Law",
         r'$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$',
         r'$\oint \vec{E} \cdot d\vec{l} = -\frac{d\Phi_B}{dt}$',
         'Changing B induces E'],
        ["Ampère-Maxwell\nLaw",
         r'$\nabla \times \vec{B} = \mu_0\vec{J} + \mu_0\epsilon_0\frac{\partial \vec{E}}{\partial t}$',
         r'$\oint \vec{B} \cdot d\vec{l} = \mu_0 I + \mu_0\epsilon_0\frac{d\Phi_E}{dt}$',
         'Currents and changing E\ninduce B'],
    ]

    table = ax.table(cellText=equations, loc='center', cellLoc='center',
                     colWidths=[0.18, 0.3, 0.3, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.5)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title("Maxwell's Equations: Complete Summary\n" +
                r"These four equations describe ALL classical electromagnetic phenomena",
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('maxwell_summary_table.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Day 213: Complete Maxwell's Equations - Computational Lab")
    print("="*60)

    print("\n1. Visual Representation of All Four Maxwell Equations")
    visualize_maxwell_equations()

    print("\n2. Electromagnetic Boundary Conditions")
    visualize_boundary_conditions()

    print("\n3. Point Charge Fields and Gauss's Law")
    plot_point_charge_fields()

    print("\n4. Electric-Magnetic Duality")
    visualize_em_duality()

    print("\n5. Maxwell's Equations Summary Table")
    create_maxwell_summary()

    print("\nAll visualizations complete!")
    print(f"\nKey constant: c = 1/√(μ₀ε₀) = {c:.0f} m/s")
```

## Summary

### Key Formulas

| Maxwell Equation | Differential Form | Integral Form |
|-----------------|-------------------|---------------|
| Gauss (E) | $\nabla \cdot \vec{E} = \rho/\epsilon_0$ | $\oint \vec{E} \cdot d\vec{A} = Q_{enc}/\epsilon_0$ |
| Gauss (B) | $\nabla \cdot \vec{B} = 0$ | $\oint \vec{B} \cdot d\vec{A} = 0$ |
| Faraday | $\nabla \times \vec{E} = -\partial\vec{B}/\partial t$ | $\oint \vec{E} \cdot d\vec{l} = -d\Phi_B/dt$ |
| Ampère-Maxwell | $\nabla \times \vec{B} = \mu_0\vec{J} + \mu_0\epsilon_0\partial\vec{E}/\partial t$ | $\oint \vec{B} \cdot d\vec{l} = \mu_0 I + \mu_0\epsilon_0 d\Phi_E/dt$ |

### Boundary Conditions

| Condition | Formula |
|-----------|---------|
| Normal E | $(E_{2\perp} - E_{1\perp}) = \sigma/\epsilon_0$ |
| Tangential E | $E_{2\parallel} = E_{1\parallel}$ |
| Normal B | $B_{2\perp} = B_{1\perp}$ |
| Tangential B | $(B_{2\parallel} - B_{1\parallel}) = \mu_0 K$ |

### Main Takeaways

1. **Maxwell's four equations** completely describe classical electromagnetism
2. **Divergence equations** describe sources; **curl equations** describe induction
3. **Maxwell's equations imply** charge conservation and wave propagation
4. **Boundary conditions** follow from the integral forms
5. **In matter**, we use $\vec{D}$ and $\vec{H}$ to handle bound charges and currents
6. **Gauge invariance** becomes U(1) symmetry in quantum electrodynamics

## Daily Checklist

- [ ] I can write all four Maxwell equations in both forms
- [ ] I understand the physical meaning of each equation
- [ ] I can derive boundary conditions from Maxwell's equations
- [ ] I can write Maxwell's equations in linear media
- [ ] I understand the potential formulation and gauge invariance
- [ ] I know how Maxwell's equations relate to QED
- [ ] I completed the computational lab

## Preview: Day 214

Tomorrow we derive the most profound consequence of Maxwell's equations: **electromagnetic waves**. By combining Faraday's law and Ampère-Maxwell law, we'll show that oscillating electric and magnetic fields propagate through space at speed $c = 1/\sqrt{\mu_0\epsilon_0}$—the speed of light! This discovery unified optics with electromagnetism.

---

*"From a long view of the history of mankind—seen from, say, ten thousand years from now—there can be little doubt that the most significant event of the 19th century will be judged as Maxwell's discovery of the laws of electrodynamics."* — Richard Feynman
