# Day 208: Magnetic Dipoles

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Magnetic Dipole Field |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Torque, Energy, and Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 208, you will be able to:

1. Calculate the magnetic dipole moment of a current loop
2. Derive the magnetic field of a dipole at large distances
3. Compute torque and energy of a dipole in an external field
4. Relate orbital motion to orbital magnetic moments
5. Connect classical dipoles to electron spin magnetic moments
6. Understand the Zeeman effect and spin precession

---

## Core Content

### 1. Magnetic Dipole Moment

**For a current loop:**

A planar loop of area $A$ carrying current $I$ has magnetic dipole moment:

$$\boxed{\boldsymbol{\mu} = IA\hat{\mathbf{n}} = I\mathbf{A}}$$

where $\hat{\mathbf{n}}$ is the unit normal to the loop (right-hand rule with current direction).

**Units:** A$\cdot$m$^2$ = J/T

**For a general current distribution:**
$$\boldsymbol{\mu} = \frac{1}{2}\int\mathbf{r}' \times \mathbf{J}(\mathbf{r}')\,d^3r'$$

**For multiple turns:** $\mu = NIA$

### 2. The Magnetic Dipole Field

**Far from a magnetic dipole** (at distance $r \gg$ size of loop):

**Vector potential:**
$$\mathbf{A} = \frac{\mu_0}{4\pi}\frac{\boldsymbol{\mu} \times \hat{\mathbf{r}}}{r^2}$$

**Magnetic field:**
$$\boxed{\mathbf{B}_{\text{dip}} = \frac{\mu_0}{4\pi}\left[\frac{3(\boldsymbol{\mu}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \boldsymbol{\mu}}{r^3}\right]}$$

**In spherical coordinates** (dipole along $z$-axis):

$$B_r = \frac{\mu_0\mu}{4\pi}\frac{2\cos\theta}{r^3}$$

$$B_\theta = \frac{\mu_0\mu}{4\pi}\frac{\sin\theta}{r^3}$$

**Compare to electric dipole field:**
$$\mathbf{E}_{\text{dip}} = \frac{1}{4\pi\varepsilon_0}\left[\frac{3(\mathbf{p}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{p}}{r^3}\right]$$

Same angular structure! The factor $1/\varepsilon_0$ is replaced by $\mu_0$.

### 3. Field on the Axis and Equator

**On the axis** ($\theta = 0$):
$$\mathbf{B}_{\text{axis}} = \frac{\mu_0}{4\pi}\frac{2\mu}{z^3}\hat{\mathbf{z}} = \frac{\mu_0\mu}{2\pi z^3}\hat{\mathbf{z}}$$

**On the equator** ($\theta = \pi/2$):
$$\mathbf{B}_{\text{eq}} = -\frac{\mu_0}{4\pi}\frac{\mu}{r^3}\hat{\mathbf{z}} = -\frac{\mu_0\mu}{4\pi r^3}\hat{\mathbf{z}}$$

The field on the axis is twice as strong and in the opposite direction to the field on the equator.

### 4. Torque on a Dipole

**In a uniform external field $\mathbf{B}$:**

$$\boxed{\boldsymbol{\tau} = \boldsymbol{\mu} \times \mathbf{B}}$$

**Magnitude:** $\tau = \mu B\sin\theta$

The torque tends to align the dipole with the field.

**Derivation:** Consider a rectangular loop. Forces on opposite sides create a couple:
$$\tau = (BIL)w\sin\theta = BIA\sin\theta = \mu B\sin\theta$$

### 5. Potential Energy of a Dipole

The work done to rotate the dipole from alignment:

$$\boxed{U = -\boldsymbol{\mu}\cdot\mathbf{B} = -\mu B\cos\theta}$$

**Minimum energy:** $U = -\mu B$ when $\boldsymbol{\mu} \parallel \mathbf{B}$ (aligned)

**Maximum energy:** $U = +\mu B$ when $\boldsymbol{\mu}$ anti-parallel to $\mathbf{B}$

### 6. Force on a Dipole in Non-uniform Field

**In a non-uniform field:**

$$\boxed{\mathbf{F} = \nabla(\boldsymbol{\mu}\cdot\mathbf{B})}$$

For a dipole aligned with the field:
$$F = \mu\frac{\partial B}{\partial z}$$

**Important:** In a uniform field, there is torque but no net force.

### 7. Orbital Magnetic Moment

**An electron orbiting a nucleus** creates a current loop:

Current: $I = \frac{e}{T} = \frac{ev}{2\pi r}$

Area: $A = \pi r^2$

Magnetic moment:
$$\mu_{\text{orb}} = IA = \frac{evr}{2} = \frac{e}{2m_e}L$$

where $L = m_e vr$ is the orbital angular momentum.

**Gyromagnetic ratio for orbital motion:**
$$\boxed{\boldsymbol{\mu}_{\text{orb}} = -\frac{e}{2m_e}\mathbf{L} = -\gamma_L\mathbf{L}}$$

The minus sign indicates $\boldsymbol{\mu}$ is anti-parallel to $\mathbf{L}$ (electron negative charge).

**Bohr magneton:**
$$\boxed{\mu_B = \frac{e\hbar}{2m_e} = 9.274 \times 10^{-24}\text{ J/T}}$$

This is the natural unit of magnetic moment for electrons.

### 8. Classical Magnetic Moment Summary

| Source | Magnetic Moment |
|--------|-----------------|
| Current loop | $\mu = IA$ |
| Solenoid ($N$ turns) | $\mu = NIA$ |
| Electron orbit | $\mu = \frac{e}{2m_e}L = \mu_B(L/\hbar)$ |
| Earth | $\mu \approx 8 \times 10^{22}$ A$\cdot$m$^2$ |
| Bar magnet | $\mu \sim 1$ A$\cdot$m$^2$ |

---

## Quantum Mechanics Connection

### Electron Spin Magnetic Moment

**The electron has an intrinsic magnetic moment** from its spin:

$$\boxed{\boldsymbol{\mu}_s = -g_s\frac{e}{2m_e}\mathbf{S} = -g_s\mu_B\frac{\mathbf{S}}{\hbar}}$$

where $g_s \approx 2.002$ is the electron g-factor.

**Spin quantum number:** $s = 1/2$, so $S_z = \pm\hbar/2$

**Spin magnetic moment (z-component):**
$$\mu_{s,z} = \mp g_s\frac{\mu_B}{2} \approx \mp\mu_B$$

### The Anomalous g-factor

**Dirac equation predicts:** $g_s = 2$ exactly.

**QED corrections (Schwinger):** $g_s = 2\left(1 + \frac{\alpha}{2\pi} + \cdots\right)$

$$g_s = 2.002\,319\,304\,362\ldots$$

One of the most precisely verified predictions in physics!

### Zeeman Effect

**In an external magnetic field,** the energy of an atom depends on $m_l$ and $m_s$:

$$\boxed{E = E_0 + \mu_B B(m_l + g_s m_s)}$$

For hydrogen: Energy levels split into multiple sub-levels.

**Normal Zeeman effect** (spin ignored): Three lines (for $\Delta m_l = 0, \pm 1$)

**Anomalous Zeeman effect** (spin included): More complex splitting due to $g_s \neq 1$

### Larmor Precession

**A magnetic moment in a field precesses** around the field direction:

$$\frac{d\boldsymbol{\mu}}{dt} = \boldsymbol{\tau} = \boldsymbol{\mu} \times \mathbf{B}$$

**Precession frequency (Larmor frequency):**
$$\boxed{\omega_L = \gamma B}$$

where $\gamma = \frac{e}{2m}$ is the gyromagnetic ratio.

For electron spin: $\omega_L \approx g_s\frac{eB}{2m_e} \approx \frac{eB}{m_e}$ (same as cyclotron frequency!)

### Total Magnetic Moment

**For an electron in an atom:**
$$\boldsymbol{\mu}_{\text{total}} = \boldsymbol{\mu}_{\text{orb}} + \boldsymbol{\mu}_s = -\frac{\mu_B}{\hbar}(\mathbf{L} + g_s\mathbf{S})$$

The total angular momentum $\mathbf{J} = \mathbf{L} + \mathbf{S}$ has associated magnetic moment with effective g-factor given by the Lande g-factor.

### Stern-Gerlach Experiment

**Demonstrates space quantization of spin:**

A beam of silver atoms passes through an inhomogeneous magnetic field. The force:
$$F_z = \mu_z\frac{\partial B_z}{\partial z}$$

Since $\mu_z$ is quantized ($\pm\mu_B$), the beam splits into two discrete components.

This was the first direct evidence of electron spin (1922).

---

## Worked Examples

### Example 1: Dipole Moment of a Circular Loop

**Problem:** A circular loop of radius $R = 5$ cm carries current $I = 2$ A. Find:
(a) The magnetic dipole moment
(b) The field at a point 10 cm above the center
(c) The torque if placed in a field $B = 0.1$ T at 30 degrees

**Solution:**

(a) $\mu = IA = I\pi R^2 = 2 \times \pi \times (0.05)^2 = 1.57 \times 10^{-2}$ A$\cdot$m$^2$

(b) On axis at $z = 10$ cm (dipole approximation):
$$B = \frac{\mu_0\mu}{2\pi z^3} = \frac{(4\pi \times 10^{-7})(1.57 \times 10^{-2})}{2\pi \times (0.1)^3}$$
$$B = 3.14 \times 10^{-6}\text{ T} = 3.14\ \mu\text{T}$$

(c) $\tau = \mu B\sin\theta = (1.57 \times 10^{-2})(0.1)\sin(30°)$
$$\tau = 7.85 \times 10^{-4}\text{ N}\cdot\text{m}$$

### Example 2: Electron Orbital Moment in Hydrogen

**Problem:** An electron in the hydrogen atom ground state has orbital angular momentum $L = 0$. In the first excited state with $l = 1$, find:
(a) The orbital magnetic moment
(b) The energy splitting in a 1 T field

**Solution:**

(a) For $l = 1$: $L = \sqrt{l(l+1)}\hbar = \sqrt{2}\hbar$

$$\mu_{\text{orb}} = \frac{e}{2m_e}L = \mu_B\sqrt{l(l+1)} = \sqrt{2}\mu_B$$
$$\mu_{\text{orb}} = 1.31 \times 10^{-23}\text{ J/T}$$

(b) Energy splitting for $m_l = -1, 0, +1$:
$$\Delta E = \mu_B B \cdot \Delta m_l$$

Between adjacent levels:
$$\Delta E = \mu_B B = (9.27 \times 10^{-24})(1) = 9.27 \times 10^{-24}\text{ J}$$
$$\Delta E = 5.79 \times 10^{-5}\text{ eV}$$

Frequency: $f = \Delta E/h = 14.0$ GHz

### Example 3: Force on Magnetic Dipole

**Problem:** A small bar magnet ($\mu = 1$ A$\cdot$m$^2$) is near a wire carrying current $I = 10$ A. At distance $r = 5$ cm from the wire, with $\mu$ aligned radially, find the force on the magnet.

**Solution:**

Field of wire: $B = \frac{\mu_0 I}{2\pi r}$

Field gradient: $\frac{\partial B}{\partial r} = -\frac{\mu_0 I}{2\pi r^2}$

Force:
$$F = \mu\frac{\partial B}{\partial r} = -\frac{\mu\mu_0 I}{2\pi r^2}$$
$$F = -\frac{(1)(4\pi \times 10^{-7})(10)}{2\pi(0.05)^2} = -8 \times 10^{-4}\text{ N}$$

The force is attractive (toward the wire).

---

## Practice Problems

### Problem 1: Direct Application
A square loop of side $a = 10$ cm carries current $I = 5$ A. Calculate:
(a) The magnetic dipole moment
(b) The maximum torque in a field $B = 0.5$ T

**Answers:** (a) $\mu = 0.05$ A$\cdot$m$^2$; (b) $\tau_{\max} = 0.025$ N$\cdot$m

### Problem 2: Intermediate
The Earth has a magnetic dipole moment of approximately $8 \times 10^{22}$ A$\cdot$m$^2$. Calculate the magnetic field strength at the Earth's surface on the equator. (Earth radius = 6371 km)

**Answer:** $B \approx 31\ \mu$T (actual value varies from 25-65 $\mu$T)

### Problem 3: Challenging
An electron in a Bohr orbit of radius $r$ with orbital quantum number $l$ has angular momentum $L = \sqrt{l(l+1)}\hbar$.

(a) Show that the orbital magnetic moment is $\mu = \mu_B\sqrt{l(l+1)}$
(b) In a magnetic field $B$, the precession frequency is $\omega_L = \mu B/(L\sin\theta)$. Show this equals $eB/(2m_e)$.

### Problem 4: Spin Magnetic Moment
An electron with spin-up ($m_s = +1/2$) is placed in a magnetic field $B = 2$ T. Calculate:
(a) The spin magnetic moment component along the field
(b) The potential energy
(c) The energy difference to spin-down state

**Answers:** (a) $\mu_{s,z} = -\mu_B$; (b) $U = +\mu_B B = 1.85 \times 10^{-23}$ J; (c) $\Delta U = 2\mu_B B = 0.23$ meV

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
mu0 = 4 * np.pi * 1e-7
mu_B = 9.274e-24  # Bohr magneton (J/T)
hbar = 1.055e-34
e = 1.602e-19
m_e = 9.109e-31
g_s = 2.002  # Electron g-factor

# Create visualizations
fig = plt.figure(figsize=(16, 12))

# ========== Plot 1: Magnetic dipole field lines ==========
ax1 = fig.add_subplot(2, 2, 1)

# Create grid
x = np.linspace(-2, 2, 30)
z = np.linspace(-2, 2, 30)
X, Z = np.meshgrid(x, z)

# Avoid origin
r = np.sqrt(X**2 + Z**2)
r[r < 0.3] = np.nan

# Dipole field components (dipole along z-axis)
mu = 1.0  # arbitrary units
theta = np.arctan2(np.abs(X), Z)
cos_theta = Z / r
sin_theta = X / r

# B_r and B_theta to B_x and B_z
B_r = (mu0 * mu / (4 * np.pi)) * (2 * cos_theta / r**3)
B_theta = (mu0 * mu / (4 * np.pi)) * (sin_theta / r**3)

B_z = B_r * cos_theta - B_theta * sin_theta * np.sign(X)
B_x = B_r * sin_theta * np.sign(X) + B_theta * cos_theta

# Normalize for streamplot
B_mag = np.sqrt(B_x**2 + B_z**2)

# Streamplot
ax1.streamplot(X, Z, B_x, B_z, density=1.5, color=np.log10(B_mag + 1e-10),
               cmap='hot', linewidth=1)

# Draw dipole
ax1.annotate('', xy=(0, 0.15), xytext=(0, -0.15),
             arrowprops=dict(arrowstyle='->', color='blue', lw=3))
ax1.text(0.15, 0, '$\\boldsymbol{\\mu}$', fontsize=14, color='blue')

ax1.set_xlabel('x')
ax1.set_ylabel('z')
ax1.set_title('Magnetic Dipole Field Lines')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')

# ========== Plot 2: Field magnitude along axis and equator ==========
ax2 = fig.add_subplot(2, 2, 2)

r = np.linspace(0.5, 5, 100)
mu = 1.0  # A*m^2

# Field on axis (theta = 0)
B_axis = (mu0 / (4 * np.pi)) * (2 * mu / r**3)

# Field on equator (theta = pi/2)
B_eq = (mu0 / (4 * np.pi)) * (mu / r**3)

ax2.loglog(r, B_axis * 1e6, 'b-', linewidth=2, label='On axis')
ax2.loglog(r, B_eq * 1e6, 'r--', linewidth=2, label='On equator')
ax2.loglog(r, 1/(r**3) * B_axis[0] * r[0]**3 * 1e6, 'k:', alpha=0.5,
           label='$\\propto 1/r^3$')

ax2.set_xlabel('Distance r (m)')
ax2.set_ylabel('|B| ($\\mu$T)')
ax2.set_title('Dipole Field Magnitude ($\\mu = 1$ A$\\cdot$m$^2$)')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

# ========== Plot 3: Torque and energy vs angle ==========
ax3 = fig.add_subplot(2, 2, 3)

theta = np.linspace(0, 2*np.pi, 200)
mu_val = 1.0
B_val = 1.0

# Torque magnitude
tau = mu_val * B_val * np.abs(np.sin(theta))

# Potential energy
U = -mu_val * B_val * np.cos(theta)

ax3.plot(theta * 180 / np.pi, tau, 'b-', linewidth=2, label='$|\\tau|/\\mu B$')
ax3.plot(theta * 180 / np.pi, U, 'r-', linewidth=2, label='$U/\\mu B$')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='g', linestyle='--', alpha=0.5)
ax3.axvline(x=180, color='g', linestyle='--', alpha=0.5)
ax3.axvline(x=360, color='g', linestyle='--', alpha=0.5)

ax3.text(0, -1.2, 'stable', ha='center', fontsize=10, color='green')
ax3.text(180, -1.2, 'unstable', ha='center', fontsize=10, color='red')

ax3.set_xlabel('Angle $\\theta$ (degrees)')
ax3.set_ylabel('Normalized values')
ax3.set_title('Torque and Energy of Dipole in Field')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 360)

# ========== Plot 4: Zeeman splitting ==========
ax4 = fig.add_subplot(2, 2, 4)

B_vals = np.linspace(0, 5, 100)

# Energy levels for l=1 state (ignoring spin for simplicity)
E_0 = 0  # Reference energy

# m_l = -1, 0, +1
for m_l in [-1, 0, 1]:
    E = E_0 + mu_B * B_vals * m_l
    label = f'$m_l = {m_l:+d}$'
    ax4.plot(B_vals, E * 1e23, linewidth=2, label=label)

ax4.set_xlabel('Magnetic Field B (T)')
ax4.set_ylabel('Energy shift ($10^{-23}$ J)')
ax4.set_title('Zeeman Splitting (l = 1 state)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('day_208_magnetic_dipoles.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Larmor precession simulation ==========
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Time evolution of spin precessing in B field
ax5 = axes2[0]

B = 1.0  # Tesla
omega_L = e * B / (2 * m_e)  # Larmor frequency (classical orbital)
omega_L_spin = g_s * e * B / (2 * m_e)  # For spin

T = 2 * np.pi / omega_L_spin
t = np.linspace(0, 3 * T, 1000)

# Initial spin at angle theta_0 from B
theta_0 = np.pi / 4

mu_x = np.sin(theta_0) * np.cos(omega_L_spin * t)
mu_y = np.sin(theta_0) * np.sin(omega_L_spin * t)
mu_z = np.cos(theta_0) * np.ones_like(t)

ax5.plot(t * 1e12, mu_x, 'b-', linewidth=2, label='$\\mu_x$')
ax5.plot(t * 1e12, mu_y, 'r-', linewidth=2, label='$\\mu_y$')
ax5.plot(t * 1e12, mu_z, 'g-', linewidth=2, label='$\\mu_z$')
ax5.set_xlabel('Time (ps)')
ax5.set_ylabel('$\\mu$ component (normalized)')
ax5.set_title(f'Larmor Precession in B = {B} T\n$f_L = ${omega_L_spin/(2*np.pi)/1e9:.1f} GHz')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 3D trajectory
ax6 = fig2.add_subplot(1, 2, 2, projection='3d')
ax6.plot(mu_x, mu_y, mu_z, 'b-', linewidth=1.5)
ax6.plot([0], [0], [0], 'ko', markersize=5)  # Origin
ax6.quiver(0, 0, 0, 0, 0, 1.2, color='red', arrow_length_ratio=0.1, linewidth=2)
ax6.text(0, 0, 1.4, '$\\mathbf{B}$', fontsize=12, color='red')

ax6.set_xlabel('$\\mu_x$')
ax6.set_ylabel('$\\mu_y$')
ax6.set_zlabel('$\\mu_z$')
ax6.set_title('Precession Trajectory')

plt.tight_layout()
plt.savefig('day_208_larmor_precession.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Comparison: Orbital vs Spin magnetic moments ==========
print("\nDay 208: Magnetic Dipoles Complete")
print("="*55)

print(f"\nBohr magneton: μ_B = {mu_B:.4e} J/T")
print(f"                   = {mu_B/e*1e3:.4f} meV/T")

print(f"\nElectron spin g-factor: g_s = {g_s:.10f}")
print(f"(QED prediction accurate to 10^-12!)")

print("\nMagnetic moments comparison:")
print(f"  Electron spin:       μ_s = g_s × μ_B/2 = {g_s * mu_B / 2:.4e} J/T")
print(f"  Electron orbit (l=1): μ_orb = √2 × μ_B = {np.sqrt(2) * mu_B:.4e} J/T")
print(f"  Proton (nuclear):    μ_p = 2.79 × μ_N = {2.79 * 5.05e-27:.4e} J/T")

print("\nLarmor frequencies in 1 T field:")
omega_L_orb = e * 1 / (2 * m_e)
omega_L_spin = g_s * e * 1 / (2 * m_e)
print(f"  Orbital:  ω_L = {omega_L_orb:.4e} rad/s = {omega_L_orb/(2*np.pi)/1e9:.2f} GHz")
print(f"  Spin:     ω_L = {omega_L_spin:.4e} rad/s = {omega_L_spin/(2*np.pi)/1e9:.2f} GHz")

print("\nZeeman splitting in 1 T:")
Delta_E = mu_B * 1
print(f"  ΔE = μ_B × B = {Delta_E:.4e} J = {Delta_E/e*1e3:.4f} meV")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\boldsymbol{\mu} = IA\hat{\mathbf{n}}$ | Magnetic dipole moment of loop |
| $\mathbf{B}_{\text{dip}} = \frac{\mu_0}{4\pi r^3}[3(\boldsymbol{\mu}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \boldsymbol{\mu}]$ | Dipole field |
| $\boldsymbol{\tau} = \boldsymbol{\mu} \times \mathbf{B}$ | Torque on dipole |
| $U = -\boldsymbol{\mu}\cdot\mathbf{B}$ | Potential energy |
| $\mathbf{F} = \nabla(\boldsymbol{\mu}\cdot\mathbf{B})$ | Force in non-uniform field |
| $\boldsymbol{\mu}_{\text{orb}} = -\frac{e}{2m}\mathbf{L}$ | Orbital magnetic moment |
| $\boldsymbol{\mu}_s = -g_s\frac{e}{2m}\mathbf{S}$ | Spin magnetic moment |
| $\mu_B = \frac{e\hbar}{2m_e}$ | Bohr magneton |

### Main Takeaways

1. **Magnetic dipole moment** $\boldsymbol{\mu} = IA\hat{\mathbf{n}}$ characterizes current loops
2. **Dipole field** falls off as $1/r^3$, same structure as electric dipole
3. **Torque** aligns dipole with field; **energy** is minimized when aligned
4. **Orbital motion** creates magnetic moments proportional to angular momentum
5. **Electron spin** has intrinsic magnetic moment with $g_s \approx 2$
6. **Zeeman effect** splits energy levels in magnetic fields

---

## Daily Checklist

- [ ] I can calculate magnetic dipole moments of current loops
- [ ] I can derive and apply the dipole field formula
- [ ] I understand torque and energy of dipoles in fields
- [ ] I can relate orbital motion to magnetic moments
- [ ] I understand electron spin magnetic moment and g-factor
- [ ] I can explain the Zeeman effect

---

## Preview: Day 209

Tomorrow we study **magnetization and magnetic materials** — how bulk matter responds to magnetic fields. We'll classify materials as diamagnetic, paramagnetic, or ferromagnetic and understand the quantum origins of each behavior.

---

*"The magnetic dipole is to magnetism what the electric dipole is to electricity — the fundamental source of long-range fields. In quantum mechanics, the electron's spin magnetic moment makes every electron a tiny bar magnet."*

---

**Next:** Day 209 — Magnetization & Materials
