# Day 210: Week 30 Review — Magnetostatics

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 11:30 AM | 2.5 hours | Concept Review & Synthesis |
| Late Morning | 11:30 AM - 12:30 PM | 1 hour | Problem Set A |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Set B |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Comprehensive Lab & Assessment |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 210, you will be able to:

1. Synthesize all Week 30 magnetostatics concepts
2. Solve problems combining multiple topics
3. Connect classical magnetostatics to quantum mechanics
4. Apply concepts to real-world electromagnetic systems
5. Identify gaps in understanding for further study
6. Prepare for electromagnetic waves (Week 31)

---

## Week 30 Concept Map

```
                           MAGNETOSTATICS
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   FOUNDATIONS              POTENTIALS              MATERIALS
        │                        │                        │
   ┌────┴────┐              ┌────┴────┐              ┌────┴────┐
   │         │              │         │              │         │
Lorentz   Biot-      Vector        Magnetic      Magnetization
 Force   Savart    Potential       Dipoles       & Materials
   │         │              │         │              │
   ↓         ↓              ↓         ↓              ↓
F=qv×B   dB=μ₀I dl×r̂   B=∇×A      μ=IA      M, H, B relations
         4π   r²       Gauge      τ=μ×B      Dia/Para/Ferro
                      freedom     U=-μ·B     Hysteresis
   │         │              │         │              │
   └────┬────┘              │         │              │
        │                   │         │              │
        ↓                   ↓         ↓              ↓
   Ampère's Law      Aharonov-    Electron      Quantum
   ∮B·dl=μ₀I          Bohm       Spin/Orbit   Exchange
```

---

## Core Concepts Summary

### 1. Lorentz Force (Day 204)

**The fundamental force law:**
$$\boxed{\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})}$$

**Key results:**
| Quantity | Formula |
|----------|---------|
| Cyclotron frequency | $\omega_c = \frac{\|q\|B}{m}$ |
| Cyclotron radius | $r_c = \frac{mv_\perp}{\|q\|B}$ |
| E×B drift | $\mathbf{v}_d = \frac{\mathbf{E} \times \mathbf{B}}{B^2}$ |
| Force on wire | $\mathbf{F} = I\mathbf{L} \times \mathbf{B}$ |

**QM connection:** Landau levels $E_n = \hbar\omega_c(n + \frac{1}{2})$

### 2. Biot-Savart Law (Day 205)

**Field from current element:**
$$\boxed{d\mathbf{B} = \frac{\mu_0}{4\pi}\frac{I\,d\boldsymbol{\ell} \times \hat{\mathbf{r}}}{r^2}}$$

**Key results:**
| Configuration | Field |
|---------------|-------|
| Infinite wire | $B = \frac{\mu_0 I}{2\pi s}$ |
| Circular loop (center) | $B = \frac{\mu_0 I}{2R}$ |
| Circular loop (axis) | $B = \frac{\mu_0 IR^2}{2(R^2+z^2)^{3/2}}$ |

### 3. Ampere's Law (Day 206)

**Integral form:**
$$\boxed{\oint_C \mathbf{B} \cdot d\boldsymbol{\ell} = \mu_0 I_{\text{enc}}}$$

**Differential form:**
$$\boxed{\nabla \times \mathbf{B} = \mu_0 \mathbf{J}}$$

**Key results:**
| Configuration | Field |
|---------------|-------|
| Solenoid | $B = \mu_0 nI$ |
| Toroid | $B = \frac{\mu_0 NI}{2\pi r}$ |
| Current sheet | $B = \frac{\mu_0 K}{2}$ |

### 4. Magnetic Vector Potential (Day 207)

**Definition:** $\mathbf{B} = \nabla \times \mathbf{A}$

$$\boxed{\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int\frac{\mathbf{J}(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}d^3r'}$$

**Gauge freedom:** $\mathbf{A}' = \mathbf{A} + \nabla\chi$ gives same $\mathbf{B}$

**Coulomb gauge:** $\nabla \cdot \mathbf{A} = 0$

**Flux from A:** $\Phi = \oint \mathbf{A} \cdot d\boldsymbol{\ell}$

**QM connection:** Minimal coupling $\hat{\mathbf{p}} \to \hat{\mathbf{p}} - q\mathbf{A}$, Aharonov-Bohm effect

### 5. Magnetic Dipoles (Day 208)

**Dipole moment:** $\boldsymbol{\mu} = IA\hat{\mathbf{n}}$

**Dipole field:**
$$\boxed{\mathbf{B}_{\text{dip}} = \frac{\mu_0}{4\pi r^3}[3(\boldsymbol{\mu}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \boldsymbol{\mu}]}$$

**Torque and energy:**
$$\boldsymbol{\tau} = \boldsymbol{\mu} \times \mathbf{B}, \quad U = -\boldsymbol{\mu}\cdot\mathbf{B}$$

**QM connection:**
- Orbital moment: $\boldsymbol{\mu}_{\text{orb}} = -\frac{e}{2m}\mathbf{L}$
- Spin moment: $\boldsymbol{\mu}_s = -g_s\frac{e}{2m}\mathbf{S}$
- Bohr magneton: $\mu_B = \frac{e\hbar}{2m_e} = 9.27 \times 10^{-24}$ J/T

### 6. Magnetization & Materials (Day 209)

**Constitutive relations:**
$$\mathbf{B} = \mu_0(\mathbf{H} + \mathbf{M}), \quad \mathbf{M} = \chi_m\mathbf{H}$$

**Material classification:**
| Type | $\chi_m$ | Origin |
|------|----------|--------|
| Diamagnetic | $-10^{-5}$ | Induced orbital currents |
| Paramagnetic | $+10^{-4}$ | Permanent moments align |
| Ferromagnetic | $+10^{4}$ | Exchange interaction |

**QM connection:** Exchange interaction, Landau diamagnetism, Pauli paramagnetism

---

## Quantum Mechanics Connections Summary

| Classical Concept | Quantum Extension |
|-------------------|-------------------|
| Cyclotron motion | Landau levels |
| Vector potential A | Minimal coupling, Aharonov-Bohm |
| Magnetic flux | Flux quantization $\Phi_0 = h/2e$ |
| Orbital moment | Orbital quantum number $\mu = \mu_B\sqrt{l(l+1)}$ |
| Spin moment | Electron spin $g_s \approx 2.002$ |
| Zeeman effect | Energy splitting $\Delta E = \mu_B B$ |
| Diamagnetism | Landau diamagnetism (quantum, not classical!) |
| Ferromagnetism | Exchange interaction (Heisenberg model) |
| Superconductivity | Meissner effect, Cooper pairs |

---

## Problem Set A: Core Concepts

### A1: Lorentz Force
A proton ($m = 1.67 \times 10^{-27}$ kg) enters a region with $\mathbf{B} = 0.5\hat{\mathbf{z}}$ T and $\mathbf{E} = 10^4\hat{\mathbf{y}}$ V/m with initial velocity $\mathbf{v} = 10^5\hat{\mathbf{x}}$ m/s.

(a) Find the E×B drift velocity
(b) Calculate the cyclotron frequency
(c) Describe the qualitative motion

### A2: Biot-Savart
Two parallel wires separated by distance $d = 10$ cm carry currents $I_1 = 5$ A and $I_2 = 3$ A in opposite directions.

(a) Find the magnetic field at the midpoint between the wires
(b) Find the location where $B = 0$
(c) Calculate the force per unit length between the wires

### A3: Ampere's Law
A coaxial cable has inner conductor (radius $a = 1$ mm) carrying current $I = 10$ A and outer conductor (inner radius $b = 3$ mm, outer radius $c = 4$ mm) carrying return current $-I$.

(a) Find $\mathbf{B}$ in all four regions
(b) Calculate the inductance per unit length

### A4: Vector Potential
An infinite solenoid of radius $R$ carries $n$ turns per meter with current $I$.

(a) Find $\mathbf{A}$ inside and outside the solenoid
(b) Verify that $\nabla \times \mathbf{A} = \mathbf{B}$
(c) Calculate the Aharonov-Bohm phase for an electron going around the solenoid

---

## Problem Set B: Advanced Applications

### B1: Magnetic Dipole Force
A small magnetic dipole $\boldsymbol{\mu} = \mu_0\hat{\mathbf{z}}$ (with $\mu_0 = 10^{-2}$ A$\cdot$m$^2$) is located at distance $z = 0.1$ m above the center of a circular current loop of radius $R = 5$ cm carrying current $I = 1$ A.

(a) Find the magnetic field of the loop at the dipole's location
(b) Calculate the force on the dipole
(c) Is the force attractive or repulsive?

### B2: Magnetized Cylinder
An infinitely long cylinder of radius $R$ has uniform magnetization $\mathbf{M} = M_0\hat{\mathbf{z}}$.

(a) Find the bound surface current
(b) Calculate $\mathbf{B}$ inside and outside the cylinder
(c) Find $\mathbf{H}$ inside and outside

### B3: Ferromagnetic Core
A toroidal core ($\mu_r = 2000$) has mean radius $r = 10$ cm and cross-sectional area $A = 4$ cm$^2$. It is wound with $N = 500$ turns carrying current $I = 0.2$ A.

(a) Find $H$, $B$, and $M$ in the core
(b) Calculate the magnetic flux
(c) Find the inductance
(d) What current would produce the same flux without the core?

### B4: Comprehensive Problem
An electron in a hydrogen atom has orbital angular momentum $\mathbf{L} = \sqrt{2}\hbar\hat{\mathbf{z}}$ (corresponding to $l = 1$).

(a) Calculate the orbital magnetic moment
(b) If a magnetic field $B = 1$ T is applied along $z$, find the interaction energy for $m_l = -1, 0, +1$
(c) Include spin: what are all possible energies for the $l = 1$ state?
(d) Calculate the Larmor precession frequency

---

## Solutions to Selected Problems

### Solution A1: Lorentz Force

(a) **E×B drift:**
$$\mathbf{v}_d = \frac{\mathbf{E} \times \mathbf{B}}{B^2} = \frac{(10^4\hat{\mathbf{y}}) \times (0.5\hat{\mathbf{z}})}{0.25}$$
$$\mathbf{v}_d = \frac{10^4 \times 0.5}{0.25}(-\hat{\mathbf{x}}) = -2 \times 10^4\hat{\mathbf{x}} \text{ m/s}$$

(b) **Cyclotron frequency:**
$$\omega_c = \frac{eB}{m_p} = \frac{(1.6 \times 10^{-19})(0.5)}{1.67 \times 10^{-27}} = 4.79 \times 10^7 \text{ rad/s}$$
$$f_c = 7.6 \text{ MHz}$$

(c) **Motion:** Cycloid — circular motion superposed with drift

### Solution A3: Coaxial Cable

**Region 1 ($r < a$):** Assuming uniform current distribution
$$B = \frac{\mu_0 Ir}{2\pi a^2}$$

**Region 2 ($a < r < b$):**
$$B = \frac{\mu_0 I}{2\pi r}$$

**Region 3 ($b < r < c$):**
$$I_{\text{enc}} = I - I\frac{r^2 - b^2}{c^2 - b^2}$$
$$B = \frac{\mu_0 I}{2\pi r}\left(1 - \frac{r^2 - b^2}{c^2 - b^2}\right)$$

**Region 4 ($r > c$):** $B = 0$

**Inductance per unit length:**
$$L = \frac{\mu_0}{2\pi}\ln\frac{b}{a} = \frac{4\pi \times 10^{-7}}{2\pi}\ln 3 = 2.2 \times 10^{-7} \text{ H/m}$$

### Solution B2: Magnetized Cylinder

(a) **Bound surface current:**
$$\mathbf{K}_b = \mathbf{M} \times \hat{\mathbf{n}} = M_0\hat{\mathbf{z}} \times \hat{\mathbf{s}} = M_0\hat{\boldsymbol{\phi}}$$

(b) **Inside:** Like a solenoid with $nI = K_b = M_0$
$$\mathbf{B}_{\text{in}} = \mu_0 M_0\hat{\mathbf{z}}$$

**Outside:** $\mathbf{B}_{\text{out}} = 0$

(c) **H field:**
Inside: $\mathbf{H} = \frac{\mathbf{B}}{\mu_0} - \mathbf{M} = M_0\hat{\mathbf{z}} - M_0\hat{\mathbf{z}} = 0$

Outside: $\mathbf{H} = 0$

### Solution B4: Electron in Hydrogen

(a) **Orbital magnetic moment:**
$$\mu_{\text{orb}} = \frac{e}{2m_e}L = \frac{e}{2m_e}\sqrt{2}\hbar = \sqrt{2}\mu_B$$
$$\mu_{\text{orb}} = 1.31 \times 10^{-23} \text{ J/T}$$

(b) **Interaction energy:**
$$E = -\mu_B B m_l$$

For $m_l = +1$: $E = -\mu_B B = -9.27 \times 10^{-24}$ J
For $m_l = 0$: $E = 0$
For $m_l = -1$: $E = +\mu_B B = +9.27 \times 10^{-24}$ J

(c) **Including spin** ($m_s = \pm\frac{1}{2}$):
$$E = \mu_B B(m_l + g_s m_s) \approx \mu_B B(m_l + 2m_s)$$

Total states: $m_l = -1, 0, +1$ and $m_s = \pm\frac{1}{2}$ gives 6 states.

Energies: $(-1 - 1), (-1 + 1), (0 - 1), (0 + 1), (+1 - 1), (+1 + 1)$
= $-2, 0, -1, +1, 0, +2$ in units of $\mu_B B$

(d) **Larmor frequency:**
$$\omega_L = \frac{eB}{2m_e} = \frac{\mu_B B}{\hbar} = 8.79 \times 10^{10} \text{ rad/s}$$
$$f_L = 14.0 \text{ GHz}$$

---

## Self-Assessment Checklist

### Fundamentals
- [ ] I can state and apply the Lorentz force law
- [ ] I can calculate magnetic fields using Biot-Savart and Ampere's laws
- [ ] I understand when to use each method

### Vector Potential
- [ ] I understand gauge freedom and the Coulomb gauge
- [ ] I can calculate $\mathbf{A}$ for simple configurations
- [ ] I can explain the Aharonov-Bohm effect

### Dipoles
- [ ] I can calculate dipole moments and fields
- [ ] I understand torque and energy in external fields
- [ ] I can relate classical to quantum magnetic moments

### Materials
- [ ] I can classify materials by magnetic response
- [ ] I understand the $\mathbf{B}$-$\mathbf{H}$-$\mathbf{M}$ relationships
- [ ] I can explain ferromagnetism qualitatively

### Quantum Connections
- [ ] I understand Landau levels
- [ ] I can calculate Zeeman splitting
- [ ] I understand why ferromagnetism is quantum mechanical

---

## Computational Lab: Week Synthesis

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

# Physical constants
mu0 = 4 * np.pi * 1e-7
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
hbar = 1.055e-34
mu_B = e * hbar / (2 * m_e)

print("="*65)
print("Week 30: Magnetostatics - Comprehensive Review")
print("="*65)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 14))

# ========== Plot 1: Lorentz force trajectories ==========
ax1 = fig.add_subplot(2, 3, 1, projection='3d')

def lorentz_eom(state, t, q, m, E, B):
    x, y, z, vx, vy, vz = state
    v = np.array([vx, vy, vz])
    F = q * (E + np.cross(v, B))
    a = F / m
    return [vx, vy, vz, a[0], a[1], a[2]]

# Electron in uniform B field
B = np.array([0, 0, 1.0])
E = np.array([0, 0, 0])
v0 = 1e6
state0 = [0, 0, 0, v0, 0, v0/5]

omega_c = e * 1.0 / m_e
T = 2 * np.pi / omega_c
t = np.linspace(0, 5*T, 1000)

sol = odeint(lorentz_eom, state0, t, args=(e, m_e, E, B))

ax1.plot(sol[:, 0]*1e6, sol[:, 1]*1e6, sol[:, 2]*1e6, 'b-', linewidth=1)
ax1.set_xlabel('x (μm)')
ax1.set_ylabel('y (μm)')
ax1.set_zlabel('z (μm)')
ax1.set_title('Helical Motion in B field')

# ========== Plot 2: Biot-Savart - Field of loop ==========
ax2 = fig.add_subplot(2, 3, 2)

R = 0.05  # Loop radius
I = 1.0   # Current

z = np.linspace(-0.2, 0.2, 100)
B_axis = mu0 * I * R**2 / (2 * (R**2 + z**2)**(3/2))

ax2.plot(z * 100, B_axis * 1e6, 'b-', linewidth=2)
ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Loop position')
ax2.set_xlabel('z (cm)')
ax2.set_ylabel('B (μT)')
ax2.set_title('Biot-Savart: Circular Loop on Axis')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ========== Plot 3: Ampere's Law - Coaxial cable ==========
ax3 = fig.add_subplot(2, 3, 3)

a, b, c = 1, 3, 4  # mm
I_cable = 10  # A

r = np.linspace(0.1, 6, 200)
B_coax = np.zeros_like(r)

for i, ri in enumerate(r):
    if ri < a:
        B_coax[i] = mu0 * I_cable * ri / (2 * np.pi * (a*1e-3)**2) * 1e-3
    elif ri < b:
        B_coax[i] = mu0 * I_cable / (2 * np.pi * ri * 1e-3)
    elif ri < c:
        I_enc = I_cable * (1 - (ri**2 - b**2)/(c**2 - b**2))
        B_coax[i] = mu0 * I_enc / (2 * np.pi * ri * 1e-3)
    else:
        B_coax[i] = 0

ax3.plot(r, B_coax * 1e3, 'b-', linewidth=2)
ax3.axvline(x=a, color='r', linestyle='--', alpha=0.5, label='a')
ax3.axvline(x=b, color='g', linestyle='--', alpha=0.5, label='b')
ax3.axvline(x=c, color='orange', linestyle='--', alpha=0.5, label='c')
ax3.set_xlabel('r (mm)')
ax3.set_ylabel('B (mT)')
ax3.set_title('Ampère: Coaxial Cable (I=10A)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ========== Plot 4: Vector potential of solenoid ==========
ax4 = fig.add_subplot(2, 3, 4)

R_sol = 1.0  # cm
n_sol = 1000  # turns/m
I_sol = 1.0  # A

s = np.linspace(0.01, 3, 100)
A_phi = np.zeros_like(s)
B_z = np.zeros_like(s)

for i, si in enumerate(s):
    if si < R_sol:
        A_phi[i] = mu0 * n_sol * I_sol * si * 1e-2 / 2
        B_z[i] = mu0 * n_sol * I_sol
    else:
        A_phi[i] = mu0 * n_sol * I_sol * (R_sol * 1e-2)**2 / (2 * si * 1e-2)
        B_z[i] = 0

ax4.plot(s, A_phi * 1e6, 'b-', linewidth=2, label='$A_\\phi$')
ax4.plot(s, B_z * 1e6, 'r--', linewidth=2, label='$B_z$')
ax4.axvline(x=R_sol, color='gray', linestyle=':', alpha=0.7)
ax4.set_xlabel('s (cm)')
ax4.set_ylabel('Field/Potential (μT or μT·m)')
ax4.set_title('Vector Potential: Solenoid')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ========== Plot 5: Magnetic dipole field ==========
ax5 = fig.add_subplot(2, 3, 5)

# Create streamplot of dipole field
x = np.linspace(-3, 3, 30)
z = np.linspace(-3, 3, 30)
X, Z = np.meshgrid(x, z)

r = np.sqrt(X**2 + Z**2)
r[r < 0.5] = np.nan

cos_theta = Z / r
sin_theta = X / r

mu_dip = 1.0
B_r = (mu0 * mu_dip / (4 * np.pi)) * (2 * cos_theta / r**3)
B_theta = (mu0 * mu_dip / (4 * np.pi)) * (sin_theta / r**3)

B_x = B_r * sin_theta + B_theta * cos_theta
B_z = B_r * cos_theta - B_theta * sin_theta

B_mag = np.sqrt(B_x**2 + B_z**2)

ax5.streamplot(X, Z, B_x, B_z, density=1.2, color=np.log10(B_mag + 1e-15),
               cmap='plasma', linewidth=1)
ax5.set_xlabel('x')
ax5.set_ylabel('z')
ax5.set_title('Magnetic Dipole Field Lines')
ax5.set_xlim(-3, 3)
ax5.set_ylim(-3, 3)
ax5.set_aspect('equal')

# ========== Plot 6: Zeeman splitting ==========
ax6 = fig.add_subplot(2, 3, 6)

B_vals = np.linspace(0, 2, 100)

# l=1 state with spin
colors = ['blue', 'cyan', 'green', 'lime', 'orange', 'red']
labels = [
    '$m_l=-1, m_s=-1/2$',
    '$m_l=-1, m_s=+1/2$',
    '$m_l=0, m_s=-1/2$',
    '$m_l=0, m_s=+1/2$',
    '$m_l=+1, m_s=-1/2$',
    '$m_l=+1, m_s=+1/2$'
]
m_vals = [(-1, -0.5), (-1, 0.5), (0, -0.5), (0, 0.5), (1, -0.5), (1, 0.5)]
g_s = 2.002

for (m_l, m_s), color, label in zip(m_vals, colors, labels):
    E = mu_B * B_vals * (m_l + g_s * m_s)
    ax6.plot(B_vals, E * 1e23, color=color, linewidth=2, label=label)

ax6.set_xlabel('B (T)')
ax6.set_ylabel('Energy shift ($10^{-23}$ J)')
ax6.set_title('Zeeman Splitting (l=1, s=1/2)')
ax6.legend(fontsize=8, loc='upper left')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_210_week_review.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Summary calculations ==========
print("\n" + "="*65)
print("Key Numerical Results")
print("="*65)

print("\n1. CYCLOTRON MOTION (B = 1 T)")
print("-"*40)
for name, q, m in [("Electron", e, m_e), ("Proton", e, m_p)]:
    omega_c = q * 1.0 / m
    f_c = omega_c / (2 * np.pi)
    r_c = m * 1e6 / (q * 1.0)  # for v = 10^6 m/s
    print(f"  {name}:")
    print(f"    ω_c = {omega_c:.3e} rad/s")
    print(f"    f_c = {f_c/1e9:.3f} GHz" if f_c > 1e9 else f"    f_c = {f_c/1e6:.3f} MHz")
    print(f"    r_c (v=10⁶ m/s) = {r_c*1e6:.3f} μm" if r_c < 1e-3 else f"    r_c (v=10⁶ m/s) = {r_c*1e2:.3f} cm")

print("\n2. MAGNETIC FIELDS")
print("-"*40)
print(f"  Infinite wire (I=10A, r=1cm): B = {mu0*10/(2*np.pi*0.01)*1e3:.3f} mT")
print(f"  Solenoid (n=1000/m, I=1A):    B = {mu0*1000*1:.3f} mT")
print(f"  Loop center (R=5cm, I=1A):    B = {mu0*1/(2*0.05)*1e6:.1f} μT")

print("\n3. MAGNETIC MOMENTS")
print("-"*40)
print(f"  Bohr magneton: μ_B = {mu_B:.4e} J/T")
print(f"  Electron spin: μ_s ≈ {2*mu_B:.4e} J/T")
print(f"  Loop (R=5cm, I=1A): μ = {1*np.pi*0.05**2:.4e} A·m²")

print("\n4. ZEEMAN EFFECT (B = 1 T)")
print("-"*40)
print(f"  Energy splitting: ΔE = μ_B·B = {mu_B*1:.4e} J")
print(f"                        = {mu_B*1/e*1e3:.4f} meV")
print(f"  Larmor frequency: f_L = {e*1/(2*m_e)/(2*np.pi)/1e9:.2f} GHz")

print("\n5. MATERIAL SUSCEPTIBILITIES")
print("-"*40)
materials = [
    ("Copper (dia)", -9.6e-6),
    ("Water (dia)", -9.0e-6),
    ("Aluminum (para)", 2.3e-5),
    ("Soft iron (ferro)", 5000),
]
for name, chi in materials:
    mu_r = 1 + chi
    print(f"  {name}: χ_m = {chi:.2e}, μ_r = {mu_r:.4f}")

print("\n" + "="*65)
print("Week 30 Review Complete!")
print("="*65)
```

---

## Summary

### Week 30 Key Takeaways

1. **Lorentz force** ($\mathbf{F} = q\mathbf{v} \times \mathbf{B}$) does no work; cyclotron motion is fundamental
2. **Biot-Savart law** gives fields from current elements; works for any geometry
3. **Ampere's law** is powerful for symmetric problems; $\nabla \times \mathbf{B} = \mu_0\mathbf{J}$
4. **Vector potential** $\mathbf{A}$ is physical in QM (Aharonov-Bohm effect)
5. **Magnetic dipoles** have same field structure as electric dipoles; connect to spin
6. **Magnetic materials** respond via $\mathbf{M}$; ferromagnetism is quantum mechanical

### Maxwell's Equations (Magnetostatics)

$$\nabla \cdot \mathbf{B} = 0 \quad \text{(no monopoles)}$$
$$\nabla \times \mathbf{B} = \mu_0\mathbf{J} \quad \text{(currents create B)}$$

### The Quantum Thread

Every topic this week connects to quantum mechanics:
- Cyclotron motion → Landau levels
- Vector potential → Minimal coupling, gauge invariance
- Magnetic dipoles → Electron spin ($g_s \approx 2$)
- Zeeman effect → Atomic spectroscopy
- Ferromagnetism → Exchange interaction

---

## Preview: Week 31

Next week we study **Faraday's Law and Electromagnetic Induction**:

- Day 211: Faraday's Law
- Day 212: Inductance
- Day 213: Magnetic Energy
- Day 214: Maxwell's Equations (Complete)
- Day 215: Electromagnetic Waves I
- Day 216: Electromagnetic Waves II
- Day 217: Week Review

We'll complete Maxwell's equations and see how changing magnetic fields create electric fields — leading to electromagnetic waves!

---

*"Magnetostatics reveals a beautiful parallel to electrostatics, but with one crucial difference: there are no magnetic monopoles. In quantum mechanics, the vector potential A becomes as fundamental as the fields themselves, and electron spin emerges as a purely quantum magnetic phenomenon."*

---

**Week 30 Complete!**

**Next:** Week 31 — Faraday's Law & Electromagnetic Induction
