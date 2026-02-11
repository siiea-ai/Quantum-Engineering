# Day 207: Magnetic Vector Potential

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Vector Potential |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications & Gauge Theory |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 207, you will be able to:

1. Define the magnetic vector potential and explain why it exists
2. Derive $\mathbf{B} = \nabla \times \mathbf{A}$ and verify Maxwell's equations
3. Understand gauge freedom and the Coulomb gauge
4. Calculate $\mathbf{A}$ for common current distributions
5. Explain the Aharonov-Bohm effect and minimal coupling
6. Connect to the quantum mechanical momentum operator

---

## Core Content

### 1. Why a Vector Potential?

**From Maxwell's equation:** Since $\nabla \cdot \mathbf{B} = 0$ (no magnetic monopoles), we can always write:

$$\boxed{\mathbf{B} = \nabla \times \mathbf{A}}$$

where $\mathbf{A}$ is the **magnetic vector potential**.

**Mathematical justification:** For any vector field, if its divergence is zero, it can be written as the curl of another vector field (Helmholtz theorem).

**Why "potential"?** Just as $\mathbf{E} = -\nabla\phi$ reduces three components to one scalar, $\mathbf{B} = \nabla \times \mathbf{A}$ provides a systematic framework for calculations.

### 2. The Vector Potential from Currents

**Biot-Savart law:**
$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int\frac{\mathbf{J}(\mathbf{r}')\times(\mathbf{r}-\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|^3}d^3r'$$

**Vector potential:**
$$\boxed{\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int\frac{\mathbf{J}(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}d^3r'}$$

Compare to the scalar potential in electrostatics:
$$\phi(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0}\int\frac{\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}d^3r'$$

**For a line current:**
$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0 I}{4\pi}\oint\frac{d\boldsymbol{\ell}'}{|\mathbf{r}-\mathbf{r}'|}$$

### 3. Gauge Freedom

**Critical insight:** $\mathbf{B} = \nabla \times \mathbf{A}$ does not uniquely determine $\mathbf{A}$.

If we add the gradient of any scalar function $\chi$:
$$\mathbf{A}' = \mathbf{A} + \nabla\chi$$

Then:
$$\nabla \times \mathbf{A}' = \nabla \times \mathbf{A} + \nabla \times (\nabla\chi) = \mathbf{B} + 0 = \mathbf{B}$$

This is called **gauge freedom** — different choices of $\mathbf{A}$ give the same physics.

### 4. Coulomb Gauge

**The Coulomb gauge** (or radiation gauge) chooses:
$$\boxed{\nabla \cdot \mathbf{A} = 0}$$

**Why this choice?** From $\nabla \times \mathbf{B} = \mu_0\mathbf{J}$:
$$\nabla \times (\nabla \times \mathbf{A}) = \mu_0\mathbf{J}$$
$$\nabla(\nabla \cdot \mathbf{A}) - \nabla^2\mathbf{A} = \mu_0\mathbf{J}$$

In Coulomb gauge ($\nabla \cdot \mathbf{A} = 0$):
$$\boxed{\nabla^2\mathbf{A} = -\mu_0\mathbf{J}}$$

This is Poisson's equation for each component — just like electrostatics!

### 5. Other Gauges

**Lorenz gauge:** $\nabla \cdot \mathbf{A} + \mu_0\varepsilon_0\frac{\partial\phi}{\partial t} = 0$

This is Lorentz covariant and used in relativistic electrodynamics.

**Temporal gauge:** $\phi = 0$

**Axial gauge:** $A_z = 0$

Each gauge has advantages for different problems.

### 6. Examples of Vector Potential

**Infinite straight wire (current $I$ along $z$-axis):**

In cylindrical coordinates:
$$\mathbf{A} = -\frac{\mu_0 I}{2\pi}\ln(s)\,\hat{\mathbf{z}} + \text{const.}$$

Verify: $\mathbf{B} = \nabla \times \mathbf{A} = \frac{\mu_0 I}{2\pi s}\hat{\boldsymbol{\phi}}$ (correct!)

**Uniform magnetic field $\mathbf{B} = B\hat{\mathbf{z}}$:**

Two common choices (both valid):

*Symmetric gauge:*
$$\mathbf{A} = \frac{1}{2}\mathbf{B} \times \mathbf{r} = \frac{B}{2}(-y\hat{\mathbf{x}} + x\hat{\mathbf{y}})$$

*Landau gauge:*
$$\mathbf{A} = Bx\hat{\mathbf{y}}$$

Both satisfy $\nabla \times \mathbf{A} = B\hat{\mathbf{z}}$ and differ by $\nabla\chi$.

**Magnetic dipole (far field):**
$$\mathbf{A} = \frac{\mu_0}{4\pi}\frac{\boldsymbol{\mu} \times \hat{\mathbf{r}}}{r^2}$$

### 7. Multipole Expansion of $\mathbf{A}$

For a localized current distribution, expand in powers of $1/r$:

$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi}\left[\frac{1}{r}\int\mathbf{J}\,dV' + \frac{1}{r^2}\int\mathbf{J}(\hat{\mathbf{r}}\cdot\mathbf{r}')\,dV' + \cdots\right]$$

**Monopole term:** $\int\mathbf{J}\,dV' = 0$ for steady currents (no magnetic monopoles!)

**Dipole term:** Leads to the magnetic dipole moment:
$$\boldsymbol{\mu} = \frac{1}{2}\int\mathbf{r}' \times \mathbf{J}(\mathbf{r}')\,dV'$$

### 8. Magnetic Flux and $\mathbf{A}$

The magnetic flux through a surface $S$:
$$\Phi = \int_S \mathbf{B} \cdot d\mathbf{a} = \int_S (\nabla \times \mathbf{A}) \cdot d\mathbf{a}$$

By Stokes' theorem:
$$\boxed{\Phi = \oint_C \mathbf{A} \cdot d\boldsymbol{\ell}}$$

The flux equals the line integral of $\mathbf{A}$ around the boundary!

---

## Quantum Mechanics Connection

### Minimal Coupling: The Central Role of $\mathbf{A}$

**In quantum mechanics, the momentum operator becomes:**
$$\boxed{\hat{\mathbf{p}} \to \hat{\mathbf{p}} - q\mathbf{A}}$$

This is called **minimal coupling** and is the prescription for including magnetic fields in QM.

**The Hamiltonian:**
$$\hat{H} = \frac{(\hat{\mathbf{p}} - q\mathbf{A})^2}{2m} + q\phi = \frac{1}{2m}\left(-i\hbar\nabla - q\mathbf{A}\right)^2 + q\phi$$

**Expanded:**
$$\hat{H} = \frac{\hat{p}^2}{2m} - \frac{q}{2m}(\hat{\mathbf{p}}\cdot\mathbf{A} + \mathbf{A}\cdot\hat{\mathbf{p}}) + \frac{q^2A^2}{2m} + q\phi$$

### The Aharonov-Bohm Effect

**A profound quantum phenomenon:** A charged particle can be affected by electromagnetic potentials even when $\mathbf{E} = 0$ and $\mathbf{B} = 0$ in all regions where the particle travels.

**Setup:** An electron travels around a solenoid. Outside the solenoid:
- $\mathbf{B} = 0$ (field confined inside)
- But $\mathbf{A} \neq 0$ (vector potential extends outside)

**Phase shift:**
$$\Delta\phi = \frac{q}{\hbar}\oint \mathbf{A} \cdot d\boldsymbol{\ell} = \frac{q\Phi_B}{\hbar}$$

where $\Phi_B$ is the magnetic flux through the solenoid.

**Implication:** The vector potential $\mathbf{A}$ is physically real in quantum mechanics, not just a mathematical convenience!

### Gauge Invariance in QM

Under a gauge transformation:
$$\mathbf{A} \to \mathbf{A}' = \mathbf{A} + \nabla\chi$$
$$\phi \to \phi' = \phi - \frac{\partial\chi}{\partial t}$$

The wave function transforms as:
$$\psi \to \psi' = e^{iq\chi/\hbar}\psi$$

Physical observables (probability $|\psi|^2$, expectation values) are unchanged.

### Landau Levels Revisited

Using the symmetric gauge $\mathbf{A} = \frac{B}{2}(-y,x,0)$, the Schrodinger equation gives:

$$E_n = \hbar\omega_c\left(n + \frac{1}{2}\right)$$

The Landau gauge $\mathbf{A} = (0, Bx, 0)$ gives the same energy levels but different wave functions — gauge invariance!

---

## Worked Examples

### Example 1: Vector Potential of a Solenoid

**Problem:** An ideal solenoid of radius $R$ carries $n$ turns per length with current $I$. Find $\mathbf{A}$ inside and outside.

**Solution:**

Inside: $\mathbf{B} = \mu_0 nI\hat{\mathbf{z}}$
Outside: $\mathbf{B} = 0$

Use the flux relation: $\Phi = \oint \mathbf{A} \cdot d\boldsymbol{\ell} = A_\phi \cdot 2\pi s$

**Inside ($s < R$):**
$$\Phi = B\pi s^2 = \mu_0 nI\pi s^2$$
$$A_\phi = \frac{\mu_0 nI s}{2}$$

$$\boxed{\mathbf{A} = \frac{\mu_0 nI s}{2}\hat{\boldsymbol{\phi}} \quad (s < R)}$$

**Outside ($s > R$):**
$$\Phi = B\pi R^2 = \mu_0 nI\pi R^2$$
$$A_\phi = \frac{\mu_0 nI R^2}{2s}$$

$$\boxed{\mathbf{A} = \frac{\mu_0 nI R^2}{2s}\hat{\boldsymbol{\phi}} \quad (s > R)}$$

Note: $\mathbf{A} \neq 0$ outside even though $\mathbf{B} = 0$!

### Example 2: Verifying the Curl

**Problem:** Verify that $\mathbf{A} = \frac{B}{2}(-y\hat{\mathbf{x}} + x\hat{\mathbf{y}})$ gives $\mathbf{B} = B\hat{\mathbf{z}}$.

**Solution:**
$$\nabla \times \mathbf{A} = \begin{vmatrix} \hat{\mathbf{x}} & \hat{\mathbf{y}} & \hat{\mathbf{z}} \\ \partial_x & \partial_y & \partial_z \\ -By/2 & Bx/2 & 0 \end{vmatrix}$$

$$= \hat{\mathbf{x}}(0 - 0) - \hat{\mathbf{y}}(0 - 0) + \hat{\mathbf{z}}\left(\frac{B}{2} - \left(-\frac{B}{2}\right)\right)$$

$$= B\hat{\mathbf{z}} \quad \checkmark$$

### Example 3: Aharonov-Bohm Phase

**Problem:** An electron travels around a solenoid containing flux $\Phi_B = 10^{-15}$ Wb. Calculate the phase shift.

**Solution:**
$$\Delta\phi = \frac{e\Phi_B}{\hbar} = \frac{(1.6 \times 10^{-19})(10^{-15})}{1.055 \times 10^{-34}}$$

$$\Delta\phi = 1.52 \text{ radians} \approx 87°$$

The flux quantum: $\Phi_0 = h/e = 4.14 \times 10^{-15}$ Wb

For $\Phi_B = \Phi_0$: $\Delta\phi = 2\pi$ (full cycle)

---

## Practice Problems

### Problem 1: Direct Application
Find the vector potential $\mathbf{A}$ at a distance $s$ from an infinite straight wire carrying current $I$, using the Coulomb gauge.

**Answer:** $\mathbf{A} = -\frac{\mu_0 I}{2\pi}\ln(s/s_0)\hat{\mathbf{z}}$ where $s_0$ is a reference distance.

### Problem 2: Intermediate
A circular loop of radius $R$ carries current $I$. Find $\mathbf{A}$ on the axis of the loop at distance $z$ from the center.

**Hint:** On axis, only $A_\phi$ component survives, and by symmetry around the axis, use the integral form.

### Problem 3: Challenging
Show that the symmetric gauge $\mathbf{A} = \frac{1}{2}\mathbf{B} \times \mathbf{r}$ satisfies the Coulomb gauge condition $\nabla \cdot \mathbf{A} = 0$ for a uniform field $\mathbf{B}$.

**Hint:** Use the vector identity for the divergence of a cross product.

### Problem 4: Gauge Transformation
Starting from the Landau gauge $\mathbf{A}_L = (0, Bx, 0)$, find the gauge function $\chi$ that transforms to the symmetric gauge $\mathbf{A}_S = \frac{B}{2}(-y, x, 0)$.

**Answer:** $\chi = -\frac{B}{2}xy$

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

# Physical constants
mu0 = 4 * np.pi * 1e-7
hbar = 1.055e-34
e = 1.602e-19

# Create figure with multiple visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Vector potential of a solenoid ==========
ax1 = axes[0, 0]

R = 1.0  # Solenoid radius
n = 1000  # turns per meter
I = 1.0  # current

# Create grid
s = np.linspace(0.01, 2.5, 100)
A_phi = np.zeros_like(s)

# Calculate A_phi
for i, si in enumerate(s):
    if si < R:
        A_phi[i] = mu0 * n * I * si / 2
    else:
        A_phi[i] = mu0 * n * I * R**2 / (2 * si)

# Also plot B
B = np.zeros_like(s)
B[s < R] = mu0 * n * I

ax1.plot(s, A_phi * 1e3, 'b-', linewidth=2, label='$A_\\phi$ (mT$\\cdot$m)')
ax1.plot(s, B * 1e3, 'r--', linewidth=2, label='$B$ (mT)')
ax1.axvline(x=R, color='gray', linestyle=':', alpha=0.7, label='Solenoid edge')
ax1.fill_between([0, R], [0, 0], [max(B)*1.2*1e3, max(B)*1.2*1e3],
                  alpha=0.1, color='blue')
ax1.set_xlabel('Distance from axis s (m)')
ax1.set_ylabel('Field / Potential (mT or mT$\\cdot$m)')
ax1.set_title('Solenoid: $\\mathbf{A}$ extends beyond $\\mathbf{B}$')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 2.5)

# ========== Plot 2: Vector field plot of A (symmetric gauge) ==========
ax2 = axes[0, 1]

B_uniform = 1.0  # 1 T

x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
X, Y = np.meshgrid(x, y)

# Symmetric gauge: A = (B/2)(-y, x, 0)
Ax = -B_uniform * Y / 2
Ay = B_uniform * X / 2

# Magnitude for coloring
A_mag = np.sqrt(Ax**2 + Ay**2)

ax2.quiver(X, Y, Ax, Ay, A_mag, cmap='viridis', alpha=0.8)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('Vector Potential: Symmetric Gauge\n$\\mathbf{A} = \\frac{B}{2}(-y\\hat{x} + x\\hat{y})$')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# ========== Plot 3: Comparison of gauge choices ==========
ax3 = axes[1, 0]

# Along y=0 line for different gauges
x = np.linspace(-2, 2, 100)
y_val = 0.5
B = 1.0

# Symmetric gauge at y = y_val
A_sym_x = -B * y_val / 2 * np.ones_like(x)
A_sym_y = B * x / 2

# Landau gauge: A = (0, Bx, 0)
A_lan_x = np.zeros_like(x)
A_lan_y = B * x

ax3.plot(x, A_sym_y, 'b-', linewidth=2, label='Symmetric: $A_y$')
ax3.plot(x, A_lan_y, 'r--', linewidth=2, label='Landau: $A_y$')
ax3.plot(x, A_sym_x, 'b:', linewidth=2, label='Symmetric: $A_x$')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.set_xlabel('x (m)')
ax3.set_ylabel('$A$ component (T$\\cdot$m)')
ax3.set_title(f'Gauge Comparison at y = {y_val} m\n(Same $\\mathbf{{B}} = B\\hat{{z}}$)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ========== Plot 4: Aharonov-Bohm setup ==========
ax4 = axes[1, 1]

# Draw solenoid cross-section
theta = np.linspace(0, 2*np.pi, 100)
R_sol = 0.5

# Draw solenoid
solenoid = Circle((0, 0), R_sol, fill=True, color='gray', alpha=0.5,
                   label='Solenoid (B inside)')
ax4.add_patch(solenoid)

# Draw electron paths
# Path 1 (above)
theta1 = np.linspace(0, np.pi, 50)
r1 = 1.5
x1 = r1 * np.cos(theta1)
y1 = r1 * np.sin(theta1) * 0.6 + 0.3
ax4.plot(x1, y1, 'b-', linewidth=2, label='Path 1')

# Path 2 (below)
theta2 = np.linspace(0, np.pi, 50)
x2 = r1 * np.cos(theta2)
y2 = -r1 * np.sin(theta2) * 0.6 - 0.3
ax4.plot(x2, y2, 'r-', linewidth=2, label='Path 2')

# Draw A field lines (circles around solenoid)
for r_a in [0.8, 1.2, 1.6]:
    theta_a = np.linspace(0, 2*np.pi, 100)
    ax4.plot(r_a * np.cos(theta_a), r_a * np.sin(theta_a),
             'g:', alpha=0.5, linewidth=1)

ax4.annotate('', xy=(-1.5, 0), xytext=(-2, 0),
             arrowprops=dict(arrowstyle='->', color='black'))
ax4.annotate('', xy=(2, 0), xytext=(1.5, 0),
             arrowprops=dict(arrowstyle='->', color='black'))
ax4.text(-2.3, 0, 'Source', ha='right', va='center')
ax4.text(2.3, 0, 'Detector', ha='left', va='center')
ax4.text(0, -1.5, '$\\mathbf{A} \\neq 0$\n$\\mathbf{B} = 0$\n(outside solenoid)',
         ha='center', fontsize=10)

ax4.set_xlim(-2.5, 2.5)
ax4.set_ylim(-2, 2)
ax4.set_aspect('equal')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Aharonov-Bohm Effect\nPhase shift: $\\Delta\\phi = q\\Phi_B/\\hbar$')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_207_vector_potential.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Aharonov-Bohm phase calculation ==========
fig2, ax = plt.subplots(figsize=(10, 6))

# Phase shift vs flux
Phi_0 = 2 * np.pi * hbar / e  # Flux quantum h/e
Phi_B = np.linspace(0, 5 * Phi_0, 200)
Delta_phi = e * Phi_B / hbar

ax.plot(Phi_B / Phi_0, Delta_phi / (2*np.pi), 'b-', linewidth=2)
ax.set_xlabel('Magnetic flux $\\Phi_B / \\Phi_0$')
ax.set_ylabel('Phase shift $\\Delta\\phi / 2\\pi$')
ax.set_title('Aharonov-Bohm Phase Shift')
ax.grid(True, alpha=0.3)

# Mark integer flux quanta
for n in range(6):
    ax.axvline(x=n, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=n, color='g', linestyle=':', alpha=0.3)

ax.text(0.5, 4.5, f'$\\Phi_0 = h/e = {Phi_0*1e15:.2f}$ fWb', fontsize=12)

plt.tight_layout()
plt.savefig('day_207_ab_phase.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== Verify curl calculation numerically ==========
print("\nDay 207: Magnetic Vector Potential Complete")
print("="*55)

# Check that curl A = B for symmetric gauge
dx = 0.001
x0, y0 = 1.0, 1.0
B_test = 2.0

# Symmetric gauge: A = (B/2)(-y, x, 0)
def A_symmetric(x, y, B):
    return np.array([-B*y/2, B*x/2, 0])

# Numerical curl
def numerical_curl_z(func, x, y, B, dx):
    dAy_dx = (func(x+dx, y, B)[1] - func(x-dx, y, B)[1]) / (2*dx)
    dAx_dy = (func(x, y+dx, B)[0] - func(x, y-dx, B)[0]) / (2*dx)
    return dAy_dx - dAx_dy

curl_z = numerical_curl_z(A_symmetric, x0, y0, B_test, dx)
print(f"\nNumerical verification of curl(A) = B:")
print(f"For B = {B_test} T at ({x0}, {y0}):")
print(f"Numerical curl_z(A) = {curl_z:.6f} T")
print(f"Expected B = {B_test:.6f} T")
print(f"Error: {abs(curl_z - B_test):.2e} T")

# Flux quantum
print(f"\nFlux quantum: Phi_0 = h/e = {2*np.pi*hbar/e*1e15:.3f} fWb")
print(f"              Phi_0 = h/2e = {np.pi*hbar/e*1e15:.3f} fWb (Cooper pairs)")
```

---

## Summary

### Key Formulas

| Formula | Description |
|---------|-------------|
| $\mathbf{B} = \nabla \times \mathbf{A}$ | Definition of vector potential |
| $\mathbf{A} = \frac{\mu_0}{4\pi}\int\frac{\mathbf{J}}{r}dV'$ | $\mathbf{A}$ from current distribution |
| $\nabla \cdot \mathbf{A} = 0$ | Coulomb gauge condition |
| $\nabla^2\mathbf{A} = -\mu_0\mathbf{J}$ | Poisson equation (Coulomb gauge) |
| $\Phi = \oint\mathbf{A}\cdot d\boldsymbol{\ell}$ | Flux from vector potential |
| $\hat{\mathbf{p}} \to \hat{\mathbf{p}} - q\mathbf{A}$ | Minimal coupling (QM) |

### Main Takeaways

1. **Vector potential** $\mathbf{A}$ exists because $\nabla \cdot \mathbf{B} = 0$
2. **Gauge freedom** allows multiple choices of $\mathbf{A}$ for the same $\mathbf{B}$
3. **Coulomb gauge** ($\nabla \cdot \mathbf{A} = 0$) simplifies to Poisson's equation
4. **$\mathbf{A}$ is physical** in quantum mechanics (Aharonov-Bohm effect)
5. **Minimal coupling** $\mathbf{p} \to \mathbf{p} - q\mathbf{A}$ is fundamental to QM

---

## Daily Checklist

- [ ] I can derive $\mathbf{A}$ from current distributions
- [ ] I understand gauge freedom and can work in Coulomb gauge
- [ ] I can calculate $\mathbf{A}$ for solenoids and uniform fields
- [ ] I can explain the Aharonov-Bohm effect
- [ ] I understand minimal coupling in quantum mechanics

---

## Preview: Day 208

Tomorrow we study **magnetic dipoles** — the magnetic analog of electric dipoles. We'll see how current loops create dipole fields and connect to the intrinsic magnetic moments of electrons (spin).

---

*"The vector potential A, once considered merely a mathematical convenience, reveals itself in quantum mechanics as physically real — the Aharonov-Bohm effect proves that electrons can sense A even where B vanishes."*

---

**Next:** Day 208 — Magnetic Dipoles
