# Day 219: Spacetime and 4-Vectors

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Minkowski Spacetime |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | 4-Vectors: Position, Velocity, Momentum |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 219, you will be able to:

1. Describe the geometry of Minkowski spacetime using the metric tensor
2. Calculate spacetime intervals and classify them as timelike, spacelike, or lightlike
3. Define and manipulate 4-vectors including position, velocity, and momentum
4. Work with contravariant and covariant indices
5. Apply Lorentz transformations in matrix form
6. Connect 4-vectors to conserved quantities in relativistic physics

---

## Core Content

### 1. Minkowski Spacetime

**The key insight:** Space and time are not separate entities but form a unified 4-dimensional continuum called **spacetime**.

**Coordinates:** An event is specified by four coordinates:
$$x^{\mu} = (x^0, x^1, x^2, x^3) = (ct, x, y, z)$$

where $\mu = 0, 1, 2, 3$ is the spacetime index.

### 2. The Minkowski Metric

The **spacetime interval** between two events is:

$$\boxed{ds^2 = \eta_{\mu\nu}dx^{\mu}dx^{\nu} = -c^2dt^2 + dx^2 + dy^2 + dz^2}$$

where the **Minkowski metric tensor** is:

$$\eta_{\mu\nu} = \begin{pmatrix} -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Note:** We use the "mostly plus" convention $(-,+,+,+)$. Some texts use $(+,-,-,-)$.

**Einstein summation convention:** Repeated indices are summed:
$$\eta_{\mu\nu}dx^{\mu}dx^{\nu} = \sum_{\mu=0}^{3}\sum_{\nu=0}^{3}\eta_{\mu\nu}dx^{\mu}dx^{\nu}$$

### 3. Classification of Intervals

For the interval $\Delta s^2 = -c^2\Delta t^2 + \Delta x^2 + \Delta y^2 + \Delta z^2$:

| Type | Condition | Physical Meaning |
|------|-----------|------------------|
| **Timelike** | $\Delta s^2 < 0$ | Events can be causally connected; a massive particle can travel between them |
| **Spacelike** | $\Delta s^2 > 0$ | Events cannot be causally connected; would require faster-than-light travel |
| **Lightlike (null)** | $\Delta s^2 = 0$ | Events connected by a light ray |

**Proper time** for a timelike interval:
$$\boxed{d\tau^2 = -\frac{ds^2}{c^2} = dt^2 - \frac{dx^2 + dy^2 + dz^2}{c^2}}$$

For a particle moving with velocity $\mathbf{v}$:
$$d\tau = dt\sqrt{1 - v^2/c^2} = \frac{dt}{\gamma}$$

### 4. Light Cones

At each event, spacetime divides into regions:

- **Future light cone:** Events that can be reached by light or slower ($\Delta s^2 \leq 0$, $\Delta t > 0$)
- **Past light cone:** Events from which light or matter could have arrived ($\Delta s^2 \leq 0$, $\Delta t < 0$)
- **Elsewhere:** Spacelike separated events ($\Delta s^2 > 0$); no causal connection possible

### 5. Contravariant and Covariant Vectors

**Contravariant 4-vector** (upper index): Transforms like coordinates
$$A'^{\mu} = \Lambda^{\mu}_{\ \nu}A^{\nu}$$

**Covariant 4-vector** (lower index): Defined using the metric
$$A_{\mu} = \eta_{\mu\nu}A^{\nu}$$

**Lowering indices:**
$$A_0 = -A^0, \quad A_i = A^i \quad (i = 1, 2, 3)$$

**Inner product (Lorentz scalar):**
$$\boxed{A \cdot B = A^{\mu}B_{\mu} = \eta_{\mu\nu}A^{\mu}B^{\nu} = -A^0B^0 + A^1B^1 + A^2B^2 + A^3B^3}$$

This is **invariant** under Lorentz transformations!

### 6. The Lorentz Transformation Matrix

For a boost along the $x$-axis with velocity $v$:

$$\Lambda^{\mu}_{\ \nu} = \begin{pmatrix} \gamma & -\beta\gamma & 0 & 0 \\ -\beta\gamma & \gamma & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

where $\beta = v/c$ and $\gamma = 1/\sqrt{1-\beta^2}$.

**Properties:**
- $\det(\Lambda) = 1$ (preserves orientation)
- $\Lambda^T\eta\Lambda = \eta$ (preserves metric)
- $\Lambda^{-1}(\beta) = \Lambda(-\beta)$

### 7. The 4-Position Vector

$$\boxed{x^{\mu} = (ct, x, y, z) = (ct, \mathbf{r})}$$

**Invariant "length" squared:**
$$x_{\mu}x^{\mu} = -c^2t^2 + r^2$$

### 8. The 4-Velocity

**Definition:** The rate of change of position with respect to proper time:

$$\boxed{u^{\mu} = \frac{dx^{\mu}}{d\tau} = \gamma(c, \mathbf{v}) = \gamma(c, v_x, v_y, v_z)}$$

where $\gamma = 1/\sqrt{1 - v^2/c^2}$ with $v = |\mathbf{v}|$.

**Key property:** The 4-velocity has constant magnitude:
$$u_{\mu}u^{\mu} = \gamma^2(-c^2 + v^2) = -c^2$$

$$\boxed{u_{\mu}u^{\mu} = -c^2}$$

This is always true, regardless of the particle's speed!

### 9. The 4-Momentum

**Definition:**
$$\boxed{p^{\mu} = mu^{\mu} = (E/c, \mathbf{p}) = (\gamma mc, \gamma m\mathbf{v})}$$

where $m$ is the **rest mass** (invariant mass).

**Components:**
- $p^0 = E/c = \gamma mc$ (relativistic energy divided by $c$)
- $\mathbf{p} = \gamma m\mathbf{v}$ (relativistic 3-momentum)

**Invariant:**
$$p_{\mu}p^{\mu} = -\frac{E^2}{c^2} + |\mathbf{p}|^2 = -m^2c^2$$

This gives the famous **energy-momentum relation:**
$$\boxed{E^2 = (pc)^2 + (mc^2)^2}$$

### 10. The 4-Acceleration

$$a^{\mu} = \frac{du^{\mu}}{d\tau}$$

**Property:** Since $u_{\mu}u^{\mu} = -c^2$ is constant:
$$\frac{d}{d\tau}(u_{\mu}u^{\mu}) = 2u_{\mu}a^{\mu} = 0$$

$$\boxed{u_{\mu}a^{\mu} = 0}$$

The 4-acceleration is always orthogonal to the 4-velocity!

### 11. The 4-Current Density

For a charge distribution with charge density $\rho$ and current density $\mathbf{J}$:

$$\boxed{J^{\mu} = (c\rho, \mathbf{J}) = \rho_0 u^{\mu}}$$

where $\rho_0$ is the proper charge density (charge density in the rest frame).

**Continuity equation** becomes:
$$\partial_{\mu}J^{\mu} = \frac{\partial(c\rho)}{\partial(ct)} + \nabla \cdot \mathbf{J} = \frac{\partial\rho}{\partial t} + \nabla \cdot \mathbf{J} = 0$$

---

## Quantum Mechanics Connection

### 4-Momentum and the de Broglie Relations

The quantum mechanical wave function $\psi \propto e^{i(k \cdot r - \omega t)}$ can be written covariantly:

$$\psi \propto e^{ip_{\mu}x^{\mu}/\hbar} = e^{i(p \cdot r - Et)/\hbar}$$

where we use the 4-momentum $p^{\mu} = (E/c, \mathbf{p})$.

The **de Broglie relations** become:
$$p^{\mu} = \hbar k^{\mu}$$

where $k^{\mu} = (\omega/c, \mathbf{k})$ is the 4-wave vector.

### The Klein-Gordon Equation (Covariant Form)

The relativistic wave equation in covariant notation:

$$\boxed{\left(\partial_{\mu}\partial^{\mu} + \frac{m^2c^2}{\hbar^2}\right)\psi = 0}$$

Or using the d'Alembertian operator $\Box = \partial_{\mu}\partial^{\mu} = -\frac{1}{c^2}\frac{\partial^2}{\partial t^2} + \nabla^2$:

$$\left(\Box + \frac{m^2c^2}{\hbar^2}\right)\psi = 0$$

### Path to the Dirac Equation

The Klein-Gordon equation is second-order in time. Dirac sought a first-order equation:

$$i\hbar\frac{\partial\psi}{\partial t} = H\psi$$

where $H$ must satisfy $H^2 = c^2\mathbf{p}^2 + m^2c^4$.

This requires $H = c\boldsymbol{\alpha} \cdot \mathbf{p} + \beta mc^2$ where $\boldsymbol{\alpha}$ and $\beta$ are **matrices** (not numbers).

The resulting **Dirac equation** predicts:
1. Electron spin (spin-1/2 automatically emerges)
2. Antimatter (positron)
3. Precise magnetic moment

---

## Worked Examples

### Example 1: Spacetime Interval Classification

**Problem:** Classify the following intervals:
(a) $\Delta t = 3$ s, $\Delta r = 4 \times 10^8$ m
(b) $\Delta t = 5$ s, $\Delta r = 15 \times 10^8$ m
(c) $\Delta t = 2$ s, $\Delta r = 6 \times 10^8$ m

**Solution:**

(a) $\Delta s^2 = -(3 \times 10^8)^2(3)^2 + (4 \times 10^8)^2$
$= -8.1 \times 10^{17} + 1.6 \times 10^{17} = -6.5 \times 10^{17}$ m²

$\Delta s^2 < 0$ → **Timelike** (events can be causally connected)

(b) $\Delta s^2 = -(3 \times 10^8)^2(5)^2 + (15 \times 10^8)^2$
$= -2.25 \times 10^{18} + 2.25 \times 10^{18} = 0$

$\Delta s^2 = 0$ → **Lightlike** (connected by light ray)

(c) $\Delta s^2 = -(3 \times 10^8)^2(2)^2 + (6 \times 10^8)^2$
$= -3.6 \times 10^{17} + 3.6 \times 10^{17} = 0$

Wait, let me recalculate: $c \cdot \Delta t = 3 \times 10^8 \times 2 = 6 \times 10^8$ m = $\Delta r$

$\Delta s^2 = 0$ → **Lightlike**

### Example 2: 4-Velocity Calculation

**Problem:** A particle moves with velocity $\mathbf{v} = 0.6c\,\hat{\mathbf{x}} + 0.8c\,\hat{\mathbf{y}}$. Wait, this exceeds $c$! Let me use $\mathbf{v} = 0.6c\,\hat{\mathbf{x}}$.

**Solution:**

Speed: $v = 0.6c$

Lorentz factor: $\gamma = \frac{1}{\sqrt{1 - 0.36}} = \frac{1}{0.8} = 1.25$

4-velocity:
$$u^{\mu} = \gamma(c, v_x, v_y, v_z) = 1.25(c, 0.6c, 0, 0) = (1.25c, 0.75c, 0, 0)$$

**Verification:**
$$u_{\mu}u^{\mu} = -(1.25c)^2 + (0.75c)^2 = c^2(-1.5625 + 0.5625) = -c^2 \checkmark$$

### Example 3: Energy-Momentum from 4-Momentum

**Problem:** An electron ($m_e = 9.11 \times 10^{-31}$ kg) has total energy $E = 1.5$ MeV. Find its momentum and velocity.

**Solution:**

Rest energy: $E_0 = m_ec^2 = 9.11 \times 10^{-31} \times (3 \times 10^8)^2 = 8.2 \times 10^{-14}$ J = 0.511 MeV

From $E^2 = (pc)^2 + (m_ec^2)^2$:
$$pc = \sqrt{E^2 - (m_ec^2)^2} = \sqrt{1.5^2 - 0.511^2} = \sqrt{2.25 - 0.261} = 1.41 \text{ MeV}$$

Momentum: $p = 1.41 \text{ MeV}/c = 7.5 \times 10^{-22}$ kg·m/s

Lorentz factor: $\gamma = E/(m_ec^2) = 1.5/0.511 = 2.94$

Velocity: $v = c\sqrt{1 - 1/\gamma^2} = c\sqrt{1 - 0.116} = 0.94c$

$$\boxed{p = 7.5 \times 10^{-22} \text{ kg·m/s}, \quad v = 0.94c}$$

---

## Practice Problems

### Problem 1: Direct Application
A proton has kinetic energy $K = 500$ MeV. Calculate:
(a) Its total energy
(b) Its momentum
(c) Its 4-momentum components

**Answers:** (a) $E = 1438$ MeV; (b) $p = 1089$ MeV/c; (c) $p^{\mu} = (1438 \text{ MeV}/c, 1089 \text{ MeV}/c, 0, 0)$ for motion along x

### Problem 2: Intermediate
Two particles collide: particle 1 has 4-momentum $p_1^{\mu} = (5, 3, 0, 0)$ GeV/c, particle 2 has $p_2^{\mu} = (4, -2, 1, 0)$ GeV/c.
(a) Find the total 4-momentum
(b) Calculate the invariant mass of the system
(c) What is the total energy in the center-of-mass frame?

**Answers:** (a) $P^{\mu} = (9, 1, 1, 0)$ GeV/c; (b) $M = 8.89$ GeV/c²; (c) $E_{CM} = Mc^2 = 8.89$ GeV

### Problem 3: Challenging
A photon with energy $E_\gamma$ scatters off an electron at rest (Compton scattering). Using 4-momentum conservation, derive the Compton formula:
$$\lambda' - \lambda = \frac{h}{m_ec}(1 - \cos\theta)$$

**Hint:** Use $p_{\gamma}^{\mu}p_{\gamma\mu} = 0$ for photons and conservation of 4-momentum.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
c = 1  # Natural units: c = 1
m_e = 0.511  # MeV/c^2
m_p = 938.3  # MeV/c^2

# Minkowski metric (signature -,+,+,+)
eta = np.diag([-1, 1, 1, 1])

def lorentz_boost_x(beta):
    """Lorentz boost matrix along x-axis"""
    gamma = 1 / np.sqrt(1 - beta**2)
    return np.array([
        [gamma, -beta*gamma, 0, 0],
        [-beta*gamma, gamma, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def four_velocity(v_x, v_y, v_z, c=1):
    """Calculate 4-velocity from 3-velocity components"""
    v_sq = v_x**2 + v_y**2 + v_z**2
    gamma = 1 / np.sqrt(1 - v_sq/c**2)
    return np.array([gamma*c, gamma*v_x, gamma*v_y, gamma*v_z])

def minkowski_inner(A, B, metric=eta):
    """Calculate Minkowski inner product A·B"""
    return np.einsum('i,ij,j', A, metric, B)

def spacetime_interval(x1, x2):
    """Calculate spacetime interval between two events"""
    dx = x2 - x1
    return minkowski_inner(dx, dx)

# Create visualization
fig = plt.figure(figsize=(16, 12))

# ========== Plot 1: Light cone structure ==========
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# Create light cone
theta = np.linspace(0, 2*np.pi, 50)
t = np.linspace(0, 2, 30)
T, Theta = np.meshgrid(t, theta)
X = T * np.cos(Theta)
Y = T * np.sin(Theta)

# Future light cone
ax1.plot_surface(X, Y, T, alpha=0.3, color='yellow', label='Future')
# Past light cone
ax1.plot_surface(X, Y, -T, alpha=0.3, color='orange', label='Past')

# World lines
t_line = np.linspace(-2, 2, 100)
ax1.plot([0]*100, [0]*100, t_line, 'b-', linewidth=3, label='At rest')
ax1.plot(t_line*0.5, [0]*100, t_line, 'r-', linewidth=2, label='v=0.5c')
ax1.plot(t_line*0.8, [0]*100, t_line, 'g-', linewidth=2, label='v=0.8c')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('ct')
ax1.set_title('Light Cone Structure in 2+1 Dimensions')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_zlim(-2, 2)

# ========== Plot 2: 4-velocity magnitude ==========
ax2 = fig.add_subplot(2, 2, 2)

v_values = np.linspace(0, 0.99, 100)
gamma_values = 1 / np.sqrt(1 - v_values**2)

# Components of 4-velocity (c=1)
u0 = gamma_values  # temporal component
u1 = gamma_values * v_values  # spatial component (1D)

# Magnitude squared
u_squared = -u0**2 + u1**2  # Should be -1 (since c=1)

ax2.plot(v_values, u0, 'b-', linewidth=2, label='$u^0 = \\gamma c$')
ax2.plot(v_values, u1, 'r-', linewidth=2, label='$u^1 = \\gamma v$')
ax2.plot(v_values, -u_squared, 'g--', linewidth=2, label='$-u_\\mu u^\\mu = c^2$')

ax2.axhline(y=1, color='k', linestyle=':', alpha=0.5)
ax2.set_xlabel('v/c', fontsize=12)
ax2.set_ylabel('Components (c=1)', fontsize=12)
ax2.set_title('4-Velocity Components and Invariant Magnitude', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 8)

# ========== Plot 3: Lorentz transformation of events ==========
ax3 = fig.add_subplot(2, 2, 3)

# Events in frame S
events_S = np.array([
    [0, 0],      # Event 1: origin
    [1, 0.5],    # Event 2
    [2, 1.5],    # Event 3
    [1.5, -0.5], # Event 4
])

# Light cone lines
t_lc = np.linspace(-1, 3, 100)
ax3.plot(t_lc, t_lc, 'y-', linewidth=2, alpha=0.7)
ax3.plot(t_lc, -t_lc, 'y-', linewidth=2, alpha=0.7)
ax3.fill_between(t_lc, t_lc, 3, alpha=0.1, color='yellow')

# Plot events in S
ax3.scatter(events_S[:, 1], events_S[:, 0], c='blue', s=100, zorder=5, label='Frame S')
for i, (t, x) in enumerate(events_S):
    ax3.annotate(f'E{i+1}', (x, t), textcoords='offset points', xytext=(5, 5))

# Transform to frame S' moving at v=0.5c
beta = 0.5
Lambda = lorentz_boost_x(beta)

events_S_prime = []
for t, x in events_S:
    event_4vec = np.array([t, x, 0, 0])
    transformed = Lambda @ event_4vec
    events_S_prime.append([transformed[0], transformed[1]])

events_S_prime = np.array(events_S_prime)
ax3.scatter(events_S_prime[:, 1], events_S_prime[:, 0], c='red', s=100,
            marker='s', zorder=5, label="Frame S' (v=0.5c)")

# Draw simultaneity lines
ax3.axhline(y=events_S[1, 0], color='blue', linestyle='--', alpha=0.5)
ax3.axhline(y=events_S_prime[1, 0], color='red', linestyle='--', alpha=0.5)

ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('ct', fontsize=12)
ax3.set_title('Events in Different Reference Frames', fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1, 3)
ax3.set_ylim(-1, 3)
ax3.set_aspect('equal')

# ========== Plot 4: Energy-momentum hyperbola ==========
ax4 = fig.add_subplot(2, 2, 4)

# For a particle with mass m, E^2 - (pc)^2 = (mc^2)^2
# This is a hyperbola in (pc, E) space

p_values = np.linspace(-3, 3, 200)  # in units of mc

# Different masses (in arbitrary units)
for m_label, m_val in [('Electron', 1), ('Proton', 2), ('Photon', 0)]:
    if m_val > 0:
        E_values = np.sqrt(p_values**2 + m_val**2)
        ax4.plot(p_values, E_values, linewidth=2, label=f'{m_label} (m={m_val})')
    else:
        # Photon: E = |p|c
        ax4.plot(p_values, np.abs(p_values), 'k--', linewidth=2, label='Photon (m=0)')

ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)
ax4.set_xlabel('pc (energy units)', fontsize=12)
ax4.set_ylabel('E (energy units)', fontsize=12)
ax4.set_title('Energy-Momentum Relation: $E^2 = (pc)^2 + (mc^2)^2$', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-3, 3)
ax4.set_ylim(0, 4)

plt.tight_layout()
plt.savefig('day_219_4vectors.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Numerical demonstrations ==========
print("=" * 60)
print("4-VECTOR CALCULATIONS")
print("=" * 60)

# Calculate 4-velocity for different speeds
print("\n4-Velocity for different speeds (c=1):")
print("-" * 40)
speeds = [0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99]
for v in speeds:
    u = four_velocity(v, 0, 0)
    u_sq = minkowski_inner(u, u)
    print(f"v = {v:.2f}c: u^μ = ({u[0]:.3f}, {u[1]:.3f}, 0, 0), u_μu^μ = {u_sq:.6f}")

# Verify spacetime interval invariance
print("\n" + "=" * 60)
print("SPACETIME INTERVAL INVARIANCE")
print("=" * 60)

event1 = np.array([0, 0, 0, 0])  # ct, x, y, z
event2 = np.array([3, 1, 0, 0])  # timelike separation

s2_original = spacetime_interval(event1, event2)
print(f"\nOriginal events:")
print(f"  Event 1: {event1}")
print(f"  Event 2: {event2}")
print(f"  Interval s² = {s2_original:.4f}")

# Transform to moving frame
beta = 0.6
Lambda = lorentz_boost_x(beta)

event1_prime = Lambda @ event1
event2_prime = Lambda @ event2

s2_transformed = spacetime_interval(event1_prime, event2_prime)
print(f"\nTransformed to frame with v = {beta}c:")
print(f"  Event 1': {event1_prime}")
print(f"  Event 2': {event2_prime}")
print(f"  Interval s² = {s2_transformed:.4f}")
print(f"\nDifference: {abs(s2_original - s2_transformed):.2e} (should be ~0)")

# 4-momentum conservation example
print("\n" + "=" * 60)
print("4-MOMENTUM CONSERVATION: PARTICLE DECAY")
print("=" * 60)

# Pion at rest decaying to muon + neutrino
m_pi = 139.6  # MeV/c^2
m_mu = 105.7  # MeV/c^2
m_nu = 0      # neutrino mass negligible

# Initial 4-momentum (pion at rest)
p_pi = np.array([m_pi, 0, 0, 0])  # (E/c, px, py, pz)

print(f"\nPion (at rest) 4-momentum: {p_pi} MeV/c")
print(f"Invariant mass: {np.sqrt(-minkowski_inner(p_pi, p_pi)):.1f} MeV/c²")

# Conservation: p_pi = p_mu + p_nu
# In rest frame of pion:
# E_mu + E_nu = m_pi * c^2
# p_mu = -p_nu (opposite directions)

# From energy-momentum relation:
# E_mu^2 = p_mu^2 * c^2 + m_mu^2 * c^4
# E_nu = p_nu * c = |p_mu| * c

# Solving: (m_pi - E_nu)^2 = p_mu^2 + m_mu^2 = E_nu^2 + m_mu^2
# m_pi^2 - 2*m_pi*E_nu = m_mu^2
# E_nu = (m_pi^2 - m_mu^2) / (2*m_pi)

E_nu = (m_pi**2 - m_mu**2) / (2 * m_pi)
p_nu = E_nu  # since m_nu = 0
E_mu = m_pi - E_nu
p_mu = np.sqrt(E_mu**2 - m_mu**2)

print(f"\nDecay products:")
print(f"  Muon: E = {E_mu:.1f} MeV, p = {p_mu:.1f} MeV/c")
print(f"  Neutrino: E = {E_nu:.1f} MeV, p = {p_nu:.1f} MeV/c")

# Verify 4-momentum conservation
p_mu_4 = np.array([E_mu, p_mu, 0, 0])
p_nu_4 = np.array([E_nu, -p_nu, 0, 0])
p_total = p_mu_4 + p_nu_4

print(f"\nVerification:")
print(f"  Sum of 4-momenta: {p_total}")
print(f"  Original pion: {p_pi}")
print(f"  Conservation satisfied: {np.allclose(p_total, p_pi)}")

print("\n" + "=" * 60)
print("Day 219: Spacetime and 4-Vectors Complete")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $ds^2 = -c^2dt^2 + d\mathbf{r}^2$ | Minkowski spacetime interval |
| $\eta_{\mu\nu} = \text{diag}(-1, 1, 1, 1)$ | Minkowski metric |
| $x^{\mu} = (ct, \mathbf{r})$ | 4-position |
| $u^{\mu} = \gamma(c, \mathbf{v})$ | 4-velocity |
| $p^{\mu} = (E/c, \mathbf{p})$ | 4-momentum |
| $u_{\mu}u^{\mu} = -c^2$ | 4-velocity invariant |
| $p_{\mu}p^{\mu} = -m^2c^2$ | 4-momentum invariant |
| $J^{\mu} = (c\rho, \mathbf{J})$ | 4-current density |

### Main Takeaways

1. **Minkowski spacetime** unifies space and time into a 4-dimensional continuum
2. **Spacetime interval** $ds^2$ is invariant under Lorentz transformations
3. **4-vectors** transform covariantly, preserving their inner products
4. **4-velocity** has constant magnitude $-c^2$, regardless of speed
5. **4-momentum** combines energy and momentum into a single object
6. **Covariant formulation** makes Lorentz invariance manifest

---

## Daily Checklist

- [ ] I understand the geometry of Minkowski spacetime
- [ ] I can classify intervals as timelike, spacelike, or lightlike
- [ ] I can construct and manipulate 4-vectors
- [ ] I understand contravariant and covariant indices
- [ ] I can verify the invariance of 4-vector inner products
- [ ] I understand the connection to relativistic quantum mechanics

---

## Preview: Day 220

Tomorrow we focus on **relativistic mechanics** - deriving the relativistic energy-momentum relation $E^2 = (pc)^2 + (mc^2)^2$, understanding mass-energy equivalence, and applying these concepts to particle physics.

---

*"Henceforth space by itself, and time by itself, are doomed to fade away into mere shadows, and only a kind of union of the two will preserve an independent reality."*
— Hermann Minkowski, 1908

---

**Next:** Day 220 — Relativistic Mechanics
