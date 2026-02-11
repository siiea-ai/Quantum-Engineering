# Day 204: Lorentz Force and Magnetic Field

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Lorentz Force |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Charged Particle Motion |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 204, you will be able to:

1. State the Lorentz force law and explain its physical meaning
2. Describe the motion of charged particles in uniform magnetic fields
3. Calculate cyclotron frequency and radius
4. Analyze motion in crossed electric and magnetic fields
5. Understand the magnetic force on current-carrying wires
6. Connect to quantum Landau levels and the Hall effect

---

## Core Content

### 1. The Magnetic Force

**Historical context:** Unlike electric forces, magnetic forces were known only through permanent magnets until Oersted (1820) discovered that currents produce magnetic fields.

**Lorentz force law:**
$$\boxed{\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})}$$

For the magnetic part:
$$\mathbf{F}_{\text{mag}} = q\mathbf{v} \times \mathbf{B}$$

**Key properties:**
1. **Perpendicular:** $\mathbf{F} \perp \mathbf{v}$ always
2. **No work done:** $W = \int \mathbf{F} \cdot d\boldsymbol{\ell} = 0$ (force perpendicular to displacement)
3. **Speed unchanged:** $|\mathbf{v}|$ is constant
4. **Only deflects:** Changes direction, not speed

### 2. Units of Magnetic Field

From $F = qvB\sin\theta$:
$$[B] = \frac{[F]}{[q][v]} = \frac{\text{N}}{\text{C}\cdot\text{m/s}} = \frac{\text{N}}{\text{A}\cdot\text{m}} = \text{T (Tesla)}$$

Also: 1 T = 10⁴ Gauss (CGS unit)

**Typical field strengths:**
| Source | Field Strength |
|--------|---------------|
| Earth's surface | $\sim 50\ \mu$T |
| Bar magnet | $\sim 10$ mT |
| MRI machine | $1-7$ T |
| Strongest lab magnets | $\sim 45$ T |
| Neutron star | $10^8 - 10^{11}$ T |

### 3. Motion in Uniform Magnetic Field

**Setup:** $\mathbf{B} = B\hat{\mathbf{z}}$, particle with charge $q$, mass $m$.

**Equation of motion:**
$$m\frac{d\mathbf{v}}{dt} = q\mathbf{v} \times \mathbf{B}$$

**Components:**
$$\dot{v}_x = \frac{qB}{m}v_y, \quad \dot{v}_y = -\frac{qB}{m}v_x, \quad \dot{v}_z = 0$$

**Solution:** Circular motion in $xy$-plane with:

**Cyclotron frequency:**
$$\boxed{\omega_c = \frac{|q|B}{m}}$$

**Cyclotron radius (gyroradius):**
$$\boxed{r_c = \frac{mv_\perp}{|q|B}}$$

**Period:**
$$T = \frac{2\pi}{\omega_c} = \frac{2\pi m}{|q|B}$$

Note: Period is independent of velocity!

### 4. Helical Motion

If $v_\parallel \neq 0$ (velocity component along $\mathbf{B}$):
- Motion perpendicular to $\mathbf{B}$: circular with radius $r_c$
- Motion parallel to $\mathbf{B}$: constant velocity $v_\parallel$

**Result:** Helical trajectory with:
- Pitch: $p = v_\parallel T = \frac{2\pi m v_\parallel}{|q|B}$

### 5. Crossed Electric and Magnetic Fields

**Setup:** $\mathbf{E} = E\hat{\mathbf{y}}$, $\mathbf{B} = B\hat{\mathbf{z}}$

**Steady drift:** The particle drifts with constant velocity when:
$$\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B}) = 0$$

$$\mathbf{v}_d = \frac{\mathbf{E} \times \mathbf{B}}{B^2}$$

**Drift velocity magnitude:**
$$v_d = \frac{E}{B}$$

This is the principle behind velocity selectors!

### 6. Force on Current-Carrying Wire

For a wire carrying current $I$:
$$d\mathbf{F} = I\,d\boldsymbol{\ell} \times \mathbf{B}$$

**Total force:**
$$\boxed{\mathbf{F} = I\int d\boldsymbol{\ell} \times \mathbf{B}}$$

For uniform field and straight wire:
$$\mathbf{F} = I\mathbf{L} \times \mathbf{B}$$

**Magnitude:** $F = BIL\sin\theta$

---

## Quantum Mechanics Connection

### Landau Levels

In quantum mechanics, a charged particle in a uniform magnetic field has quantized energy levels:

$$\boxed{E_n = \hbar\omega_c\left(n + \frac{1}{2}\right) = \frac{\hbar|q|B}{m}\left(n + \frac{1}{2}\right)}$$

where $n = 0, 1, 2, \ldots$

These are called **Landau levels**. The classical cyclotron motion becomes quantized circular orbits.

### Degeneracy

Each Landau level has enormous degeneracy — the number of states equals the number of magnetic flux quanta through the sample:
$$N = \frac{\Phi}{\Phi_0} = \frac{BA}{\hbar/e}$$

### Hall Effect

When current flows through a conductor in a magnetic field, a transverse voltage develops:

$$V_H = \frac{IB}{nqd}$$

where $n$ is carrier density and $d$ is thickness.

**Quantum Hall Effect:** At low temperatures and high fields, the Hall conductance is quantized:
$$\sigma_{xy} = \nu\frac{e^2}{h}$$

This is one of the most precise measurements in physics!

### Electron Spin Magnetic Moment

The electron has an intrinsic magnetic moment:
$$\boldsymbol{\mu}_s = -g_s\frac{e}{2m_e}\mathbf{S}$$

where $g_s \approx 2.002$ (from QED).

This gives rise to the Zeeman effect and spin-orbit coupling.

---

## Worked Examples

### Example 1: Proton in Earth's Field

**Problem:** A proton (mass $m_p = 1.67 \times 10^{-27}$ kg) moves at $v = 10^6$ m/s perpendicular to Earth's magnetic field ($B = 50$ μT). Find:
(a) Cyclotron frequency
(b) Cyclotron radius
(c) Period

**Solution:**

(a) $\omega_c = \frac{eB}{m_p} = \frac{1.6 \times 10^{-19} \times 50 \times 10^{-6}}{1.67 \times 10^{-27}} = 4.79 \times 10^3$ rad/s

$f_c = \omega_c/(2\pi) = 763$ Hz

(b) $r_c = \frac{m_p v}{eB} = \frac{1.67 \times 10^{-27} \times 10^6}{1.6 \times 10^{-19} \times 50 \times 10^{-6}} = 209$ km

(c) $T = 2\pi/\omega_c = 1.31$ ms

### Example 2: Velocity Selector

**Problem:** A velocity selector uses $E = 10^4$ V/m and $B = 0.1$ T. What velocity passes through undeflected?

**Solution:**
$$v = \frac{E}{B} = \frac{10^4}{0.1} = 10^5 \text{ m/s}$$

### Example 3: Mass Spectrometer

**Problem:** Ions of charge $q$ and mass $m$ are accelerated through potential $V$ and enter a region of uniform magnetic field $B$ perpendicular to their velocity. Find the radius of their circular path.

**Solution:**
After acceleration: $\frac{1}{2}mv^2 = qV \Rightarrow v = \sqrt{2qV/m}$

In magnetic field: $r = \frac{mv}{qB} = \frac{m\sqrt{2qV/m}}{qB} = \sqrt{\frac{2mV}{qB^2}}$

$$\boxed{r = \frac{1}{B}\sqrt{\frac{2mV}{q}}}$$

Different masses give different radii — the basis of mass spectrometry.

---

## Practice Problems

### Problem 1: Direct Application
An electron enters a uniform magnetic field $B = 0.5$ T with velocity $v = 2 \times 10^7$ m/s perpendicular to the field. Find:
(a) The cyclotron radius
(b) The kinetic energy in eV

**Answers:** (a) $r = 0.227$ mm; (b) $KE = 1.14$ keV

### Problem 2: Intermediate
A proton and an alpha particle ($q = 2e$, $m = 4m_p$) have the same kinetic energy. They enter the same magnetic field. What is the ratio of their cyclotron radii?

**Answer:** $r_\alpha/r_p = \sqrt{2}$

### Problem 3: Challenging
A charged particle enters a region with $\mathbf{E} = E_0\hat{\mathbf{x}}$ and $\mathbf{B} = B_0\hat{\mathbf{z}}$ with initial velocity $\mathbf{v}_0 = v_0\hat{\mathbf{y}}$. Find the trajectory.

**Hint:** Transform to the drift frame where $\mathbf{E}' = 0$.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

# Physical constants
e = 1.602e-19  # C
m_e = 9.109e-31  # kg
m_p = 1.673e-27  # kg

def lorentz_force(state, t, q, m, E, B):
    """
    Equations of motion for charged particle in E and B fields.
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state
    v = np.array([vx, vy, vz])

    # Lorentz force
    F = q * (E + np.cross(v, B))
    a = F / m

    return [vx, vy, vz, a[0], a[1], a[2]]

# Create visualization
fig = plt.figure(figsize=(15, 12))

# ========== Plot 1: Circular motion in B field ==========
ax1 = fig.add_subplot(2, 2, 1)

B = np.array([0, 0, 1.0])  # 1 T in z
E = np.array([0, 0, 0])
q = e
m = m_e
v0 = 1e6  # m/s

# Initial conditions
state0 = [0, 0, 0, v0, 0, 0]

# Time array
omega_c = np.abs(q) * np.linalg.norm(B) / m
T = 2 * np.pi / omega_c
t = np.linspace(0, 3*T, 1000)

# Solve
sol = odeint(lorentz_force, state0, t, args=(q, m, E, B))

ax1.plot(sol[:, 0] * 1e6, sol[:, 1] * 1e6, 'b-', linewidth=2)
ax1.set_xlabel('x (μm)')
ax1.set_ylabel('y (μm)')
ax1.set_title(f'Electron in B = 1 T\nCyclotron period T = {T*1e12:.2f} ps')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# ========== Plot 2: Helical motion ==========
ax2 = fig.add_subplot(2, 2, 2, projection='3d')

# Initial conditions with z velocity
state0_helix = [0, 0, 0, v0, 0, v0/3]
sol_helix = odeint(lorentz_force, state0_helix, t, args=(q, m, E, B))

ax2.plot(sol_helix[:, 0] * 1e6, sol_helix[:, 1] * 1e6, sol_helix[:, 2] * 1e6,
          'b-', linewidth=1.5)
ax2.set_xlabel('x (μm)')
ax2.set_ylabel('y (μm)')
ax2.set_zlabel('z (μm)')
ax2.set_title('Helical Motion\n$v_z \\neq 0$')

# ========== Plot 3: Crossed E and B fields ==========
ax3 = fig.add_subplot(2, 2, 3)

E_cross = np.array([0, 1e4, 0])  # 10^4 V/m in y
B_cross = np.array([0, 0, 1.0])  # 1 T in z

# Drift velocity
v_drift = np.cross(E_cross, B_cross) / np.linalg.norm(B_cross)**2

# Particle starting at rest
state0_cross = [0, 0, 0, 0, 0, 0]
t_cross = np.linspace(0, 5*T, 2000)

sol_cross = odeint(lorentz_force, state0_cross, t_cross, args=(q, m, E_cross, B_cross))

ax3.plot(sol_cross[:, 0] * 1e6, sol_cross[:, 1] * 1e6, 'b-', linewidth=1.5)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.set_xlabel('x (μm)')
ax3.set_ylabel('y (μm)')
ax3.set_title(f'Crossed E and B Fields\nDrift velocity = {v_drift[0]:.0f} m/s')
ax3.grid(True, alpha=0.3)

# ========== Plot 4: Mass spectrometer ==========
ax4 = fig.add_subplot(2, 2, 4)

B_spec = 0.5  # T
V_acc = 1000  # V

# Different isotopes
isotopes = [
    ('¹H⁺', 1, 1),
    ('²H⁺', 2, 1),
    ('⁴He²⁺', 4, 2),
    ('¹²C⁴⁺', 12, 4),
]

colors = ['red', 'blue', 'green', 'orange']

for (name, A, Z), color in zip(isotopes, colors):
    m = A * m_p
    q_ion = Z * e

    # Velocity after acceleration
    v = np.sqrt(2 * q_ion * V_acc / m)

    # Radius
    r = m * v / (q_ion * B_spec)

    # Draw semicircle
    theta = np.linspace(0, np.pi, 100)
    x = r * (1 - np.cos(theta))
    y = r * np.sin(theta)

    ax4.plot(x * 100, y * 100, color=color, linewidth=2, label=f'{name}: r={r*100:.2f} cm')

ax4.set_xlabel('x (cm)')
ax4.set_ylabel('y (cm)')
ax4.set_title(f'Mass Spectrometer\nB = {B_spec} T, V = {V_acc} V')
ax4.legend()
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 15)

plt.tight_layout()
plt.savefig('day_204_lorentz_force.png', dpi=150, bbox_inches='tight')
plt.show()

# Landau levels visualization
fig, ax = plt.subplots(figsize=(10, 6))

B_vals = np.linspace(0.1, 10, 100)
hbar = 1.055e-34

for n in range(5):
    E_n = hbar * e * B_vals / m_e * (n + 0.5) / e  # Convert to eV
    ax.plot(B_vals, E_n * 1000, label=f'n = {n}')

ax.set_xlabel('Magnetic Field B (T)')
ax.set_ylabel('Energy (meV)')
ax.set_title('Landau Levels for Electron')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('day_204_landau_levels.png', dpi=150, bbox_inches='tight')
plt.show()

print("Day 204: Lorentz Force Complete")
print("="*50)
print(f"Electron cyclotron frequency in 1 T: ω_c = {e*1/m_e:.3e} rad/s")
print(f"Electron cyclotron frequency: f_c = {e*1/(2*np.pi*m_e)/1e9:.2f} GHz")
print(f"Proton cyclotron frequency in 1 T: f_c = {e*1/(2*np.pi*m_p)/1e6:.2f} MHz")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\mathbf{F} = q\mathbf{v} \times \mathbf{B}$ | Magnetic force |
| $\omega_c = |q|B/m$ | Cyclotron frequency |
| $r_c = mv_\perp/(|q|B)$ | Cyclotron radius |
| $\mathbf{v}_d = \mathbf{E} \times \mathbf{B}/B^2$ | E×B drift velocity |
| $\mathbf{F} = I\mathbf{L} \times \mathbf{B}$ | Force on wire |

### Main Takeaways

1. **Magnetic force** is perpendicular to velocity — no work done
2. **Cyclotron motion** has frequency independent of speed
3. **E×B drift** is independent of charge and mass
4. **Landau quantization** gives discrete energy levels in B field
5. **Spin magnetic moment** leads to Zeeman effect

---

## Daily Checklist

- [ ] I can state and apply the Lorentz force law
- [ ] I can calculate cyclotron frequency and radius
- [ ] I understand E×B drift
- [ ] I can connect to quantum Landau levels
- [ ] I understand the force on current-carrying wires

---

## Preview: Day 205

Tomorrow we study the **Biot-Savart law** — how currents create magnetic fields. This is the magnetic analog of Coulomb's law.

---

*"The magnetic force never does work — it only changes the direction of motion, never the speed."*

---

**Next:** Day 205 — Biot-Savart Law
