# Day 223: Relativistic Electrodynamics - Liénard-Wiechert Potentials and Radiation

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Liénard-Wiechert Potentials |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Radiation from Accelerating Charges |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 223, you will be able to:

1. Derive the Liénard-Wiechert potentials for a moving point charge
2. Calculate electric and magnetic fields from the potentials
3. Understand the concept of retarded time and light-cone constraints
4. Apply the Larmor formula for radiation from accelerating charges
5. Describe relativistic effects on radiation patterns
6. Connect to synchrotron radiation and particle accelerator physics

---

## Core Content

### 1. The Retarded Time Problem

When a charge moves, information about its position travels at speed $c$. The field at point $\mathbf{r}$ at time $t$ depends on where the charge was at an earlier **retarded time** $t_r$.

**Retarded time condition:**
$$\boxed{t_r = t - \frac{|\mathbf{r} - \mathbf{r}_s(t_r)|}{c}}$$

where $\mathbf{r}_s(t_r)$ is the source position at the retarded time.

**Define:**
$$\mathbf{R}(t_r) = \mathbf{r} - \mathbf{r}_s(t_r)$$

Then: $c(t - t_r) = |\mathbf{R}(t_r)| = R$

### 2. Liénard-Wiechert Potentials

For a point charge $q$ moving with velocity $\mathbf{v}(t)$, the potentials at field point $\mathbf{r}$ at time $t$ are:

$$\boxed{\phi(\mathbf{r}, t) = \frac{q}{4\pi\epsilon_0}\frac{1}{R - \boldsymbol{\beta} \cdot \mathbf{R}}\Bigg|_{t_r}}$$

$$\boxed{\mathbf{A}(\mathbf{r}, t) = \frac{\mu_0 q c}{4\pi}\frac{\boldsymbol{\beta}}{R - \boldsymbol{\beta} \cdot \mathbf{R}}\Bigg|_{t_r}}$$

where $\boldsymbol{\beta} = \mathbf{v}/c$ and all quantities are evaluated at the retarded time $t_r$.

**In covariant form:**
$$A^{\mu}(x) = \frac{q}{4\pi\epsilon_0 c}\frac{u^{\mu}}{u_{\nu}R^{\nu}}\Bigg|_{ret}$$

where $R^{\mu} = x^{\mu} - x_s^{\mu}(t_r)$ is the 4-vector from source to field point.

### 3. The Kappa Factor

Define the crucial factor:
$$\kappa = 1 - \hat{\mathbf{R}} \cdot \boldsymbol{\beta} = 1 - \beta\cos\theta$$

where $\theta$ is the angle between velocity and the direction from source to field point.

$$\boxed{\phi = \frac{q}{4\pi\epsilon_0}\frac{1}{\kappa R}\Bigg|_{t_r}}$$

**Physical meaning:** The $\kappa$ factor accounts for:
1. The Doppler effect (source moving toward/away from field point)
2. The "searchlight" effect for relativistic motion

### 4. Electric Field from Liénard-Wiechert Potentials

The electric field has two parts:

$$\boxed{\mathbf{E} = \mathbf{E}_{vel} + \mathbf{E}_{acc}}$$

**Velocity field** (Coulomb-like, no radiation):
$$\mathbf{E}_{vel} = \frac{q}{4\pi\epsilon_0}\frac{(1-\beta^2)(\hat{\mathbf{R}} - \boldsymbol{\beta})}{\kappa^3 R^2}\Bigg|_{t_r}$$

**Acceleration field** (radiation field):
$$\mathbf{E}_{acc} = \frac{q}{4\pi\epsilon_0 c}\frac{\hat{\mathbf{R}} \times [(\hat{\mathbf{R}} - \boldsymbol{\beta}) \times \dot{\boldsymbol{\beta}}]}{\kappa^3 R}\Bigg|_{t_r}$$

**Key differences:**
| Property | Velocity field | Acceleration field |
|----------|----------------|-------------------|
| Falls off as | $1/R^2$ | $1/R$ |
| Carries energy to infinity | No | Yes |
| Requires acceleration | No | Yes |
| Radiates | No | Yes |

### 5. Magnetic Field

The magnetic field is always:
$$\boxed{\mathbf{B} = \frac{1}{c}\hat{\mathbf{R}} \times \mathbf{E}}$$

In the radiation zone, $\mathbf{E}$, $\mathbf{B}$, and $\hat{\mathbf{R}}$ are mutually perpendicular.

### 6. The Larmor Formula

For a **non-relativistic** accelerating charge, the total radiated power is:

$$\boxed{P = \frac{q^2 a^2}{6\pi\epsilon_0 c^3} = \frac{\mu_0 q^2 a^2 c}{6\pi}}$$

where $a = |\dot{\mathbf{v}}|$ is the acceleration.

**Angular distribution:**
$$\frac{dP}{d\Omega} = \frac{q^2 a^2}{16\pi^2\epsilon_0 c^3}\sin^2\theta$$

where $\theta$ is measured from the acceleration direction.

The radiation pattern is a **donut** shape with maximum perpendicular to acceleration.

### 7. Relativistic Generalization: Liénard Formula

For relativistic motion, the total radiated power is:

$$\boxed{P = \frac{q^2 \gamma^6}{6\pi\epsilon_0 c^3}\left[|\dot{\mathbf{v}}|^2 - \frac{|\mathbf{v} \times \dot{\mathbf{v}}|^2}{c^2}\right]}$$

**Equivalent form using 4-acceleration:**
$$P = -\frac{q^2 c}{6\pi\epsilon_0}a_{\mu}a^{\mu}$$

where $a^{\mu} = du^{\mu}/d\tau$ is the 4-acceleration.

**Special cases:**

**Linear acceleration** ($\mathbf{v} \parallel \dot{\mathbf{v}}$):
$$P_{linear} = \frac{q^2 \gamma^6 a^2}{6\pi\epsilon_0 c^3}$$

**Circular motion** ($\mathbf{v} \perp \dot{\mathbf{v}}$):
$$P_{circular} = \frac{q^2 \gamma^4 a^2}{6\pi\epsilon_0 c^3}$$

Note: Circular motion radiates less by a factor of $\gamma^2$!

### 8. Radiation Pattern for Relativistic Motion

For ultrarelativistic charges ($\gamma \gg 1$), the radiation is strongly **beamed forward** into a cone of half-angle:

$$\boxed{\theta_{beam} \approx \frac{1}{\gamma}}$$

**Physical explanation:**
- The $\kappa^{-3}$ factor in the fields becomes enormous for small $\theta$ (particle moving toward observer)
- Radiation that would spread over $4\pi$ in the rest frame gets compressed into a narrow cone

### 9. Synchrotron Radiation

For a relativistic electron in a circular orbit (magnetic field $B$):

**Power radiated:**
$$P = \frac{q^2 c \gamma^4}{6\pi\epsilon_0}\left(\frac{qB}{\gamma m}\right)^2 = \frac{q^4 B^2 \gamma^2}{6\pi\epsilon_0 m^2 c}$$

**Energy loss per revolution:**
$$\Delta E = \frac{q^2 \gamma^4}{3\epsilon_0 R}$$

where $R$ is the orbit radius.

**Critical frequency:**
$$\omega_c = \frac{3\gamma^3 c}{2R}$$

This is the frequency at which the spectrum peaks.

**Applications:**
- Particle accelerator energy limits
- Synchrotron light sources
- Astrophysical radiation (pulsars, jets)

---

## Quantum Mechanics Connection

### Quantum Radiation Theory

In QED, radiation is the emission of photons. The classical Larmor formula corresponds to the limit of many photons with small individual energies.

**Classical-quantum correspondence:**
$$P_{classical} = \int \hbar\omega \frac{dN}{d\omega dt} d\omega$$

where $dN/d\omega dt$ is the photon emission rate.

### Bremsstrahlung

When a charged particle is decelerated (e.g., by atomic nuclei), it emits **bremsstrahlung** (braking radiation):

**Quantum formula:**
$$\frac{dN}{d\omega} = \frac{2\alpha}{\pi\omega}\left[\ln\left(\frac{2E_i E_f}{m_e c^2 \hbar\omega}\right) - \frac{1}{2}\right]$$

where $\alpha \approx 1/137$ is the fine-structure constant.

### Radiation Reaction

Accelerating charges lose energy to radiation. This must affect their motion, leading to the **Abraham-Lorentz force**:

$$\mathbf{F}_{rad} = \frac{q^2}{6\pi\epsilon_0 c^3}\dot{\mathbf{a}}$$

**Problem:** This leads to runaway solutions and pre-acceleration!

In QED, radiation reaction is handled through quantum corrections (vertex corrections, self-energy) and is well-defined.

### Unruh Effect

An accelerating observer sees a thermal bath of particles even in the vacuum! The temperature is:

$$T_{Unruh} = \frac{\hbar a}{2\pi k_B c}$$

This deep connection between acceleration and quantum fields underlies the relationship between radiation and the structure of spacetime.

### Hawking Radiation

Black holes emit thermal radiation at temperature:
$$T_{Hawking} = \frac{\hbar c^3}{8\pi G M k_B}$$

This is related to the Unruh effect through the equivalence principle - an observer hovering near a black hole experiences acceleration.

---

## Worked Examples

### Example 1: Field of a Charge with Constant Velocity

**Problem:** A charge $q$ moves with constant velocity $\mathbf{v} = v\hat{\mathbf{x}}$. Find the electric field at a field point perpendicular to the velocity.

**Solution:**

At $t = 0$, let the charge be at the origin. Field point: $\mathbf{r} = r\hat{\mathbf{y}}$

Retarded position: $\mathbf{r}_s(t_r) = v t_r \hat{\mathbf{x}}$

Retarded time condition:
$$c(t - t_r) = |\mathbf{r} - \mathbf{r}_s(t_r)| = \sqrt{v^2 t_r^2 + r^2}$$

At $t = 0$ for field point on y-axis, $t_r < 0$ (the relevant source position was in the past).

For constant velocity, $\dot{\boldsymbol{\beta}} = 0$, so only velocity field:

$$\mathbf{E} = \frac{q}{4\pi\epsilon_0}\frac{(1-\beta^2)(\hat{\mathbf{R}} - \boldsymbol{\beta})}{\kappa^3 R^2}$$

At the field point directly above the current position ($\theta = 90°$):
- $\hat{\mathbf{R}} \cdot \boldsymbol{\beta} = 0$ so $\kappa = 1$
- $\hat{\mathbf{R}} = \hat{\mathbf{y}}$ (pointing from retarded position toward field point)

$$\mathbf{E} = \frac{q(1-\beta^2)}{4\pi\epsilon_0 r^2}\hat{\mathbf{y}} = \frac{q}{4\pi\epsilon_0 \gamma^2 r^2}\hat{\mathbf{y}}$$

Wait, but this should actually be enhanced by $\gamma$, not reduced. Let me reconsider...

The field perpendicular to motion is actually **enhanced** due to field line compression. The correct result at $\theta = 90°$ is:

$$\boxed{\mathbf{E}_{\perp} = \frac{\gamma q}{4\pi\epsilon_0 r^2}\hat{\mathbf{y}}}$$

### Example 2: Synchrotron Radiation Power

**Problem:** An electron with energy $E = 3$ GeV circulates in a synchrotron of radius $R = 10$ m. Calculate the power radiated.

**Solution:**

Lorentz factor:
$$\gamma = \frac{E}{m_e c^2} = \frac{3000 \text{ MeV}}{0.511 \text{ MeV}} = 5870$$

Centripetal acceleration:
$$a = \frac{v^2}{R} \approx \frac{c^2}{R} = \frac{(3 \times 10^8)^2}{10} = 9 \times 10^{15} \text{ m/s}^2$$

Power (circular motion):
$$P = \frac{e^2 \gamma^4 a^2}{6\pi\epsilon_0 c^3}$$

$$P = \frac{(1.6 \times 10^{-19})^2 \times (5870)^4 \times (9 \times 10^{15})^2}{6\pi \times 8.85 \times 10^{-12} \times (3 \times 10^8)^3}$$

$$P = \frac{2.56 \times 10^{-38} \times 1.19 \times 10^{15} \times 8.1 \times 10^{31}}{4.5 \times 10^{14}}$$

$$P \approx 5.5 \times 10^{-6} \text{ W} = 5.5 \text{ μW}$$

Energy loss per turn:
$$\Delta E = P \cdot T = P \cdot \frac{2\pi R}{c} = 5.5 \times 10^{-6} \times \frac{2\pi \times 10}{3 \times 10^8}$$
$$\Delta E \approx 1.2 \times 10^{-12} \text{ J} = 7.3 \text{ keV}$$

$$\boxed{P = 5.5 \text{ μW}, \quad \Delta E = 7.3 \text{ keV/turn}}$$

### Example 3: Radiation Cone Angle

**Problem:** A 50 GeV electron emits synchrotron radiation. What is the opening angle of the radiation cone?

**Solution:**

$$\gamma = \frac{50000 \text{ MeV}}{0.511 \text{ MeV}} = 97,800$$

Opening angle:
$$\theta_{beam} \approx \frac{1}{\gamma} = \frac{1}{97800} = 1.02 \times 10^{-5} \text{ rad} = 0.59 \text{ millidegrees}$$

$$\boxed{\theta_{beam} = 0.6 \text{ millidegrees} = 2.1 \text{ arcseconds}}$$

The radiation is incredibly tightly beamed!

---

## Practice Problems

### Problem 1: Direct Application
A proton accelerates uniformly from rest at $a = 10^{15}$ m/s². Calculate the instantaneous radiated power when it reaches $v = 0.1c$.

**Answer:** $P = 3.2 \times 10^{-23}$ W

### Problem 2: Intermediate
Derive the angular distribution of radiation power for a non-relativistic oscillating dipole $\mathbf{p}(t) = p_0\cos(\omega t)\hat{\mathbf{z}}$.

**Answer:** $\frac{dP}{d\Omega} = \frac{p_0^2\omega^4}{32\pi^2\epsilon_0 c^3}\sin^2\theta$

### Problem 3: Challenging
A relativistic electron ($\gamma = 1000$) moves in a circle. Calculate the ratio of power radiated in the forward cone ($\theta < 1/\gamma$) to the total power. Show that most of the radiation is confined to this cone.

**Hint:** Integrate the angular distribution function, which involves $\kappa^{-5}$ factors.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Physical constants
c = 3e8  # m/s
epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7
e = 1.602e-19  # C
m_e = 9.109e-31  # kg

def larmor_power(q, a):
    """Non-relativistic Larmor formula"""
    return q**2 * a**2 / (6 * np.pi * epsilon_0 * c**3)

def relativistic_power_linear(q, a, gamma):
    """Power for linear acceleration"""
    return q**2 * gamma**6 * a**2 / (6 * np.pi * epsilon_0 * c**3)

def relativistic_power_circular(q, a, gamma):
    """Power for circular motion"""
    return q**2 * gamma**4 * a**2 / (6 * np.pi * epsilon_0 * c**3)

def radiation_pattern_nonrel(theta, a):
    """Angular distribution for non-relativistic acceleration along z"""
    # dP/dΩ ∝ sin²θ
    return np.sin(theta)**2

def radiation_pattern_rel(theta, beta, phi=0):
    """
    Angular distribution for relativistic motion.
    beta: velocity/c
    theta: angle from velocity direction
    phi: azimuthal angle (for acceleration perpendicular to velocity)
    Returns normalized angular distribution
    """
    gamma = 1 / np.sqrt(1 - beta**2)

    # For motion along z, acceleration perpendicular
    kappa = 1 - beta * np.cos(theta)

    # Simplified pattern for acceleration perpendicular to velocity
    # Full expression is complex - this captures the main features
    numerator = np.sin(theta)**2
    denominator = kappa**5

    return numerator / denominator

# Create visualization
fig = plt.figure(figsize=(16, 12))

# ========== Plot 1: Radiation pattern comparison ==========
ax1 = fig.add_subplot(2, 2, 1, projection='polar')

theta = np.linspace(0, 2*np.pi, 360)

# Non-relativistic
P_nonrel = radiation_pattern_nonrel(theta, 1)
P_nonrel /= np.max(P_nonrel)

# Various relativistic velocities
betas = [0.5, 0.8, 0.95, 0.99]
colors = ['blue', 'green', 'orange', 'red']

ax1.plot(theta, P_nonrel, 'k-', linewidth=2, label='Non-rel')

for beta, color in zip(betas, colors):
    gamma = 1 / np.sqrt(1 - beta**2)
    P_rel = radiation_pattern_rel(theta, beta)
    P_rel /= np.max(P_rel)  # Normalize
    ax1.plot(theta, P_rel, color=color, linewidth=2, label=f'β={beta}, γ={gamma:.1f}')

ax1.set_title('Radiation Pattern vs Velocity\n(velocity along 0°)', fontsize=12)
ax1.legend(loc='upper right', fontsize=8)

# ========== Plot 2: Beaming angle vs gamma ==========
ax2 = fig.add_subplot(2, 2, 2)

gammas = np.logspace(0, 4, 100)  # 1 to 10000
beam_angles_deg = np.degrees(1 / gammas)

ax2.loglog(gammas, beam_angles_deg, 'b-', linewidth=2)
ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='1 degree')
ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='0.1 degree')

# Mark typical accelerator values
accelerator_gammas = [100, 1000, 10000, 100000]
for g in accelerator_gammas:
    angle = np.degrees(1/g)
    ax2.plot(g, angle, 'ko', markersize=8)
    ax2.annotate(f'γ={g}\n{angle:.3f}°', (g, angle),
                 textcoords='offset points', xytext=(10, 10), fontsize=9)

ax2.set_xlabel('Lorentz factor γ', fontsize=12)
ax2.set_ylabel('Beam half-angle (degrees)', fontsize=12)
ax2.set_title('Radiation Beaming Angle', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(1, 1e5)

# ========== Plot 3: Synchrotron radiation power ==========
ax3 = fig.add_subplot(2, 2, 3)

# Power vs energy for fixed radius
R = 10  # m
E_GeV = np.linspace(0.1, 10, 100)
E_J = E_GeV * 1e9 * e  # Convert to Joules

gamma_vals = E_J / (m_e * c**2)
a = c**2 / R  # Centripetal acceleration

# Power (circular)
P_vals = e**2 * gamma_vals**4 * a**2 / (6 * np.pi * epsilon_0 * c**3)

ax3.semilogy(E_GeV, P_vals * 1e6, 'b-', linewidth=2)  # Convert to μW
ax3.set_xlabel('Electron Energy (GeV)', fontsize=12)
ax3.set_ylabel('Radiated Power (μW)', fontsize=12)
ax3.set_title(f'Synchrotron Radiation Power\n(R = {R} m)', fontsize=14)
ax3.grid(True, alpha=0.3)

# Add energy loss per turn
ax3_twin = ax3.twinx()
period = 2 * np.pi * R / c
E_loss_keV = P_vals * period / e * 1e-3  # keV
ax3_twin.semilogy(E_GeV, E_loss_keV, 'r--', linewidth=2)
ax3_twin.set_ylabel('Energy loss per turn (keV)', fontsize=12, color='red')
ax3_twin.tick_params(axis='y', labelcolor='red')

# ========== Plot 4: Spectrum of synchrotron radiation ==========
ax4 = fig.add_subplot(2, 2, 4)

# Synchrotron spectrum shape (universal function)
# S(x) where x = ω/ω_c
x = np.logspace(-2, 1, 200)

# Approximate synchrotron spectrum function
# S(x) ≈ x * K_{5/3}(x) where K is modified Bessel function
# Approximation: S(x) ~ x^(1/3) for x << 1, S(x) ~ sqrt(x) * exp(-x) for x >> 1
def synchrotron_spectrum(x):
    # Simple approximation
    low_x = 0.78 * x**(1/3) * np.exp(-x)
    high_x = 0.91 * np.sqrt(x) * np.exp(-x)
    # Blend
    return np.where(x < 1, low_x + 0.5*high_x, 0.2*low_x + high_x)

S = synchrotron_spectrum(x)
S /= np.max(S)

ax4.loglog(x, S, 'b-', linewidth=2)
ax4.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='$\\omega = \\omega_c$')
ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

ax4.set_xlabel('$\\omega/\\omega_c$', fontsize=12)
ax4.set_ylabel('Intensity (normalized)', fontsize=12)
ax4.set_title('Universal Synchrotron Spectrum', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')
ax4.set_xlim(0.01, 10)
ax4.set_ylim(0.001, 2)

plt.tight_layout()
plt.savefig('day_223_radiation.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 3D visualization of radiation pattern ==========
fig2 = plt.figure(figsize=(14, 6))

for idx, beta in enumerate([0, 0.9]):
    ax = fig2.add_subplot(1, 2, idx+1, projection='3d')

    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi)

    if beta == 0:
        # Non-relativistic: sin²θ pattern
        R = np.sin(THETA)**2
        title = 'Non-relativistic\n(dipole pattern)'
    else:
        # Relativistic: forward beaming
        gamma = 1 / np.sqrt(1 - beta**2)
        kappa = 1 - beta * np.cos(THETA)
        R = np.sin(THETA)**2 / kappa**5
        R /= np.max(R)
        title = f'Relativistic (β={beta}, γ={gamma:.1f})\n(forward beamed)'

    # Convert to Cartesian
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z (velocity)')
    ax.set_title(title, fontsize=12)

    # Equal aspect ratio
    max_range = np.max([np.max(np.abs(X)), np.max(np.abs(Y)), np.max(np.abs(Z))])
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

plt.tight_layout()
plt.savefig('day_223_3d_pattern.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Numerical calculations ==========
print("=" * 60)
print("LIÉNARD-WIECHERT POTENTIALS AND RADIATION")
print("=" * 60)

# Example 1: Larmor formula
print("\n--- Example 1: Larmor Power ---")
a = 1e15  # m/s² (typical lab acceleration)
P_larmor = larmor_power(e, a)
print(f"Acceleration: a = {a:.2e} m/s²")
print(f"Larmor power: P = {P_larmor:.4e} W")

# For comparison: electron at rest in 1 V/m field
# a = eE/m = 1.6e-19 * 1 / 9.1e-31 = 1.76e11 m/s²
a_1V = e * 1 / m_e
P_1V = larmor_power(e, a_1V)
print(f"\nElectron in 1 V/m field:")
print(f"  Acceleration: a = {a_1V:.2e} m/s²")
print(f"  Power: P = {P_1V:.4e} W")

# Example 2: Synchrotron radiation
print("\n" + "=" * 60)
print("--- Example 2: Synchrotron Radiation ---")

energies_GeV = [1, 3, 7, 50]  # Various accelerators
R = 10  # m

print(f"\nCircular orbit radius R = {R} m")
print(f"\n{'Energy (GeV)':<15} {'γ':<10} {'P (μW)':<12} {'ΔE/turn (keV)':<15} {'θ_beam (mrad)':<12}")
print("-" * 70)

for E_GeV in energies_GeV:
    gamma = E_GeV * 1e9 * e / (m_e * c**2)
    a = c**2 / R
    P = relativistic_power_circular(e, a, gamma)
    period = 2 * np.pi * R / c
    dE = P * period / e * 1e-3  # keV
    theta_beam = 1 / gamma * 1000  # mrad

    print(f"{E_GeV:<15.1f} {gamma:<10.0f} {P*1e6:<12.4f} {dE:<15.2f} {theta_beam:<12.4f}")

# Example 3: Critical frequency
print("\n" + "=" * 60)
print("--- Example 3: Critical Frequency ---")

E_GeV = 3
gamma = E_GeV * 1e9 * e / (m_e * c**2)
omega_c = 3 * gamma**3 * c / (2 * R)
f_c = omega_c / (2 * np.pi)
lambda_c = c / f_c
E_photon_eV = 6.626e-34 * f_c / e

print(f"\nElectron energy: {E_GeV} GeV (γ = {gamma:.0f})")
print(f"Orbit radius: R = {R} m")
print(f"Critical frequency: f_c = {f_c:.3e} Hz")
print(f"Critical wavelength: λ_c = {lambda_c*1e9:.3f} nm")
print(f"Critical photon energy: E_c = {E_photon_eV:.1f} eV")

# This is in the X-ray range for GeV electrons!

# Example 4: Radiation reaction
print("\n" + "=" * 60)
print("--- Example 4: Radiation Reaction Time Scale ---")

# Characteristic time for radiation reaction
tau_0 = e**2 / (6 * np.pi * epsilon_0 * m_e * c**3)
print(f"\nRadiation reaction time: τ₀ = {tau_0:.4e} s")
print(f"This is the time scale over which radiation reaction is significant.")
print(f"Classical electron radius: r_e = {tau_0 * c:.4e} m")

print("\n" + "=" * 60)
print("Day 223: Relativistic Electrodynamics Complete")
print("=" * 60)
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\phi = \frac{q}{4\pi\epsilon_0}\frac{1}{\kappa R}\big|_{t_r}$ | Liénard-Wiechert scalar potential |
| $\kappa = 1 - \hat{\mathbf{R}} \cdot \boldsymbol{\beta}$ | Doppler/beaming factor |
| $P = \frac{q^2 a^2}{6\pi\epsilon_0 c^3}$ | Larmor formula (non-relativistic) |
| $P = \frac{q^2 \gamma^4 a^2}{6\pi\epsilon_0 c^3}$ | Power (circular motion) |
| $P = \frac{q^2 \gamma^6 a^2}{6\pi\epsilon_0 c^3}$ | Power (linear acceleration) |
| $\theta_{beam} \approx 1/\gamma$ | Radiation cone half-angle |
| $\omega_c = \frac{3\gamma^3 c}{2R}$ | Critical frequency (synchrotron) |

### Main Takeaways

1. **Liénard-Wiechert potentials** give exact fields for any motion of a point charge
2. **Retarded time** accounts for the finite speed of light
3. **Acceleration field** ($1/R$) carries energy to infinity - this is radiation
4. **Relativistic beaming** concentrates radiation into cone of angle $\sim 1/\gamma$
5. **Synchrotron radiation** is a major consideration in accelerator design
6. **$\gamma^4$ vs $\gamma^6$** scaling explains why circular accelerators are limited by radiation loss

---

## Daily Checklist

- [ ] I can derive and interpret the Liénard-Wiechert potentials
- [ ] I understand the concept of retarded time
- [ ] I can apply the Larmor formula and its relativistic generalizations
- [ ] I understand relativistic beaming of radiation
- [ ] I can calculate synchrotron radiation parameters
- [ ] I understand the connection to quantum radiation theory

---

## Preview: Day 224

Tomorrow is the **Month 8 Review Day**, where we synthesize all four weeks of electromagnetism: electrostatics, magnetostatics, electromagnetic waves, and relativistic electrodynamics.

---

*"Light thinks it travels faster than anything but it is wrong. No matter how fast light travels, it finds the darkness has always got there first, and is waiting for it."*
— Terry Pratchett (on the philosophical side)

*"The electromagnetic field is real in the same sense that matter is real."*
— Richard Feynman (on the physics side)

---

**Next:** Day 224 — Month 8 Comprehensive Review
