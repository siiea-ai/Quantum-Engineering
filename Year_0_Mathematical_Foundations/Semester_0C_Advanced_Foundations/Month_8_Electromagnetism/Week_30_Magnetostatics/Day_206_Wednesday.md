# Day 206: Ampère's Law

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Theory: Ampère's Law |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Computational Lab |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 206, you will be able to:

1. State Ampère's law in integral and differential form
2. Apply Ampère's law to symmetric current configurations
3. Calculate magnetic fields inside and outside conductors
4. Understand the limitations of Ampère's law (displacement current)
5. Compare to Gauss's law for electrostatics
6. Connect to the quantum mechanics of current loops

---

## Core Content

### 1. Ampère's Law

**Integral form:**
$$\boxed{\oint_C \mathbf{B} \cdot d\boldsymbol{\ell} = \mu_0 I_{\text{enc}}}$$

The line integral of $\mathbf{B}$ around any closed loop equals $\mu_0$ times the current passing through the loop.

**Differential form:**
$$\boxed{\nabla \times \mathbf{B} = \mu_0 \mathbf{J}}$$

This is one of Maxwell's equations (in magnetostatics).

### 2. Comparison with Gauss's Law

| Gauss's Law | Ampère's Law |
|-------------|--------------|
| $\oint \mathbf{E} \cdot d\mathbf{a} = Q/\varepsilon_0$ | $\oint \mathbf{B} \cdot d\boldsymbol{\ell} = \mu_0 I$ |
| Surface integral | Line integral |
| Relates to charge | Relates to current |
| $\nabla \cdot \mathbf{E} = \rho/\varepsilon_0$ | $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$ |

### 3. Strategy for Using Ampère's Law

1. **Identify symmetry:** cylindrical, planar, or toroidal
2. **Choose Amperian loop:** where $\mathbf{B}$ is constant or perpendicular
3. **Evaluate $\oint \mathbf{B} \cdot d\boldsymbol{\ell}$:** usually $B \times (\text{length})$
4. **Calculate enclosed current:** $I_{\text{enc}} = \int \mathbf{J} \cdot d\mathbf{a}$
5. **Solve for $B$**

### 4. Infinite Straight Wire

**Amperian loop:** Circle of radius $s$ centered on wire

$$B \cdot 2\pi s = \mu_0 I$$

$$\boxed{B = \frac{\mu_0 I}{2\pi s}}$$

Same result as Biot-Savart, but much easier!

### 5. Cylindrical Conductor

**Solid cylinder of radius $R$** with uniform current density $J = I/(\pi R^2)$:

**Outside ($s > R$):**
$$B = \frac{\mu_0 I}{2\pi s}$$

**Inside ($s < R$):**
$$I_{\text{enc}} = J \cdot \pi s^2 = I\frac{s^2}{R^2}$$
$$\boxed{B = \frac{\mu_0 I s}{2\pi R^2}} \quad (s < R)$$

Field grows linearly inside, just like $\mathbf{E}$ inside a uniformly charged sphere.

### 6. Solenoid (Ideal)

**Amperian rectangular loop** with one side inside, one outside:

Inside length $L$ encloses $N = nL$ turns:
$$B \cdot L = \mu_0 n L I$$

$$\boxed{B = \mu_0 n I}$$

The field is uniform inside and zero outside.

### 7. Toroid

A toroid is a solenoid bent into a circle of radius $R$ with $N$ turns.

**Inside the toroid:**
$$B \cdot 2\pi r = \mu_0 N I$$
$$\boxed{B = \frac{\mu_0 N I}{2\pi r}}$$

**Outside:** $B = 0$ (no enclosed current)

### 8. Current Sheet

An infinite plane carries surface current $\mathbf{K}$ (A/m).

**Amperian rectangular loop** straddling the sheet:
$$2BL = \mu_0 K L$$
$$\boxed{B = \frac{\mu_0 K}{2}}$$

The field is uniform on each side, opposite directions.

---

## Quantum Mechanics Connection

### Magnetic Flux Quantization

In a superconductor, the magnetic flux through a loop is quantized:
$$\Phi = n\Phi_0 = n\frac{h}{2e}$$

where $\Phi_0 = 2.07 \times 10^{-15}$ Wb is the flux quantum.

This comes from requiring the superconducting wave function to be single-valued.

### Persistent Currents in Quantum Rings

In mesoscopic metal rings at low temperature, electrons can maintain persistent currents:
$$I = -\frac{\partial E}{\partial \Phi}$$

This is a purely quantum effect with no classical analog.

### Aharonov-Bohm Effect

A charged particle can be affected by a magnetic field even when $\mathbf{B} = 0$ at the particle's location!

The phase shift:
$$\Delta\phi = \frac{q}{\hbar}\oint \mathbf{A} \cdot d\boldsymbol{\ell} = \frac{q\Phi}{\hbar}$$

This demonstrates that the vector potential $\mathbf{A}$ is physically real in quantum mechanics.

---

## Worked Examples

### Example 1: Coaxial Cable

**Problem:** A coaxial cable has inner conductor (radius $a$) carrying current $I$ and outer conductor (inner radius $b$, outer radius $c$) carrying current $-I$. Find $\mathbf{B}$ everywhere.

**Solution:**

| Region | $I_{\text{enc}}$ | $B$ |
|--------|------------------|-----|
| $s < a$ | $I(s/a)^2$ | $\frac{\mu_0 I s}{2\pi a^2}$ |
| $a < s < b$ | $I$ | $\frac{\mu_0 I}{2\pi s}$ |
| $b < s < c$ | $I - I\frac{s^2-b^2}{c^2-b^2}$ | Varies |
| $s > c$ | 0 | 0 |

The field is confined inside the cable — no external magnetic field.

### Example 2: Two Solenoids

**Problem:** A solenoid of radius $R_1$ with $n_1$ turns/m and current $I_1$ is inside a larger solenoid of radius $R_2$ with $n_2$ turns/m and current $I_2$. Find $\mathbf{B}$ everywhere.

**Solution:**

| Region | $B$ |
|--------|-----|
| $r < R_1$ | $\mu_0(n_1 I_1 + n_2 I_2)$ |
| $R_1 < r < R_2$ | $\mu_0 n_2 I_2$ |
| $r > R_2$ | 0 |

Superposition applies!

### Example 3: Thick-Walled Pipe

**Problem:** A hollow cylindrical conductor (inner radius $a$, outer radius $b$) carries current $I$ uniformly distributed. Find $\mathbf{B}$ for $a < r < b$.

**Solution:**
$$I_{\text{enc}} = I\frac{r^2 - a^2}{b^2 - a^2}$$

$$B = \frac{\mu_0 I}{2\pi r}\cdot\frac{r^2 - a^2}{b^2 - a^2}$$

$$\boxed{B = \frac{\mu_0 I(r^2 - a^2)}{2\pi r(b^2 - a^2)}}$$

---

## Practice Problems

### Problem 1: Direct Application
A long solenoid has 2000 turns per meter and carries 5 A. Find the magnetic field inside.

**Answer:** $B = 12.6$ mT

### Problem 2: Intermediate
A wire of radius $R$ carries current $I$ with non-uniform density $J(r) = J_0(r/R)^2$. Find $\mathbf{B}$ inside and outside.

### Problem 3: Challenging
Two infinite current sheets at $z = \pm d$ carry surface currents $\mathbf{K} = K\hat{\mathbf{x}}$ and $\mathbf{K} = -K\hat{\mathbf{x}}$ respectively. Find $\mathbf{B}$ everywhere.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt

mu0 = 4 * np.pi * 1e-7

def B_wire(I, s):
    """Field of infinite wire."""
    return mu0 * I / (2 * np.pi * s)

def B_solid_cylinder(I, R, s):
    """Field inside/outside solid cylinder."""
    if isinstance(s, np.ndarray):
        B = np.zeros_like(s)
        inside = s < R
        outside = ~inside
        B[inside] = mu0 * I * s[inside] / (2 * np.pi * R**2)
        B[outside] = mu0 * I / (2 * np.pi * s[outside])
        return B
    else:
        if s < R:
            return mu0 * I * s / (2 * np.pi * R**2)
        else:
            return mu0 * I / (2 * np.pi * s)

def B_solenoid(n, I):
    """Field inside ideal solenoid."""
    return mu0 * n * I

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== Plot 1: Solid vs hollow cylinder ==========
ax1 = axes[0, 0]
R = 0.01  # 1 cm
I = 100  # A

s = np.linspace(0.001, 0.05, 200)
B_solid = np.array([B_solid_cylinder(I, R, si) for si in s])

# Hollow cylinder (inner radius a, outer radius b)
a, b = 0.005, 0.01
B_hollow = np.zeros_like(s)
for i, si in enumerate(s):
    if si < a:
        B_hollow[i] = 0
    elif si < b:
        B_hollow[i] = mu0 * I * (si**2 - a**2) / (2 * np.pi * si * (b**2 - a**2))
    else:
        B_hollow[i] = mu0 * I / (2 * np.pi * si)

ax1.plot(s * 100, B_solid * 1e3, 'b-', linewidth=2, label='Solid (R=1cm)')
ax1.plot(s * 100, B_hollow * 1e3, 'r-', linewidth=2, label='Hollow (a=0.5cm, b=1cm)')
ax1.axvline(x=R*100, color='b', linestyle='--', alpha=0.5)
ax1.axvline(x=a*100, color='r', linestyle=':', alpha=0.5)
ax1.axvline(x=b*100, color='r', linestyle='--', alpha=0.5)
ax1.set_xlabel('s (cm)')
ax1.set_ylabel('B (mT)')
ax1.set_title('Magnetic Field: Solid vs Hollow Cylinder')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ========== Plot 2: Coaxial cable ==========
ax2 = axes[0, 1]
a, b, c = 0.003, 0.008, 0.01  # inner, outer inner, outer outer
I = 50  # A

s = np.linspace(0.001, 0.015, 300)
B_coax = np.zeros_like(s)

for i, si in enumerate(s):
    if si < a:
        B_coax[i] = mu0 * I * si / (2 * np.pi * a**2)
    elif si < b:
        B_coax[i] = mu0 * I / (2 * np.pi * si)
    elif si < c:
        I_enc = I - I * (si**2 - b**2) / (c**2 - b**2)
        B_coax[i] = mu0 * I_enc / (2 * np.pi * si)
    else:
        B_coax[i] = 0

ax2.plot(s * 100, B_coax * 1e3, 'b-', linewidth=2)
ax2.axvline(x=a*100, color='r', linestyle='--', alpha=0.5, label='a')
ax2.axvline(x=b*100, color='g', linestyle='--', alpha=0.5, label='b')
ax2.axvline(x=c*100, color='orange', linestyle='--', alpha=0.5, label='c')
ax2.set_xlabel('s (cm)')
ax2.set_ylabel('B (mT)')
ax2.set_title('Coaxial Cable Magnetic Field')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ========== Plot 3: Solenoid field profile ==========
ax3 = axes[1, 0]

# Field along axis of finite solenoid
L = 0.2  # 20 cm length
R = 0.02  # 2 cm radius
n = 1000  # turns/m
I = 1  # A

z = np.linspace(-0.15, 0.15, 200)

# Approximate field (exact would need integration)
# Use formula for finite solenoid on axis
B_solenoid_finite = np.zeros_like(z)
for i, zi in enumerate(z):
    # Angles to ends
    cos1 = (L/2 - zi) / np.sqrt(R**2 + (L/2 - zi)**2)
    cos2 = (-L/2 - zi) / np.sqrt(R**2 + (-L/2 - zi)**2)
    B_solenoid_finite[i] = mu0 * n * I * (cos1 - cos2) / 2

B_ideal = mu0 * n * I

ax3.plot(z * 100, B_solenoid_finite * 1e3, 'b-', linewidth=2, label='Finite solenoid')
ax3.axhline(y=B_ideal * 1e3, color='r', linestyle='--', label='Ideal (infinite)')
ax3.axvline(x=-L/2*100, color='g', linestyle=':', alpha=0.5)
ax3.axvline(x=L/2*100, color='g', linestyle=':', alpha=0.5)
ax3.fill_between([-L/2*100, L/2*100],
                  [0, 0], [B_ideal*1e3*1.1, B_ideal*1e3*1.1],
                  alpha=0.1, color='blue')
ax3.set_xlabel('z (cm)')
ax3.set_ylabel('B (mT)')
ax3.set_title('Solenoid Field Along Axis')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ========== Plot 4: Toroid ==========
ax4 = axes[1, 1]

R_toroid = 0.1  # major radius
a_toroid = 0.02  # minor radius
N = 500  # turns
I = 2  # A

r = np.linspace(R_toroid - a_toroid + 0.001, R_toroid + a_toroid - 0.001, 100)
B_toroid = mu0 * N * I / (2 * np.pi * r)

ax4.plot(r * 100, B_toroid * 1e3, 'b-', linewidth=2)
ax4.fill_between([R_toroid - a_toroid, R_toroid + a_toroid],
                  [0, 0], [B_toroid.max() * 1.1 * 1e3, B_toroid.max() * 1.1 * 1e3],
                  alpha=0.2, color='blue')
ax4.axvline(x=R_toroid*100, color='r', linestyle='--', alpha=0.5, label='Center')
ax4.set_xlabel('r (cm)')
ax4.set_ylabel('B (mT)')
ax4.set_title('Toroid Magnetic Field')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day_206_ampere.png', dpi=150, bbox_inches='tight')
plt.show()

print("Day 206: Ampère's Law Complete")
print("="*50)
print(f"\nSolenoid (n=1000/m, I=1A): B = {B_ideal*1e3:.3f} mT")
print(f"Wire (I=100A, s=1cm): B = {B_wire(100, 0.01)*1e3:.3f} mT")
```

---

## Summary

### Key Formulas

| Configuration | Field |
|--------------|-------|
| Infinite wire | $B = \frac{\mu_0 I}{2\pi s}$ |
| Inside solid conductor | $B = \frac{\mu_0 I s}{2\pi R^2}$ |
| Solenoid | $B = \mu_0 n I$ |
| Toroid | $B = \frac{\mu_0 N I}{2\pi r}$ |
| Current sheet | $B = \frac{\mu_0 K}{2}$ |

### Main Takeaways

1. **Ampère's law** relates circulation of $\mathbf{B}$ to enclosed current
2. **Symmetric problems** are easy with Ampère's law
3. **Solenoid** has uniform field $B = \mu_0 n I$ inside
4. **Coaxial cable** confines field between conductors
5. **Aharonov-Bohm effect** shows $\mathbf{A}$ is physical in QM

---

## Daily Checklist

- [ ] I can state Ampère's law in integral and differential form
- [ ] I can apply it to symmetric current configurations
- [ ] I can calculate fields inside conductors
- [ ] I understand the solenoid and toroid
- [ ] I see the quantum connection (Aharonov-Bohm)

---

## Preview: Day 207

Tomorrow we introduce the **magnetic vector potential** $\mathbf{A}$, where $\mathbf{B} = \nabla \times \mathbf{A}$. This is essential for quantum mechanics!

---

*"Ampère's law is to magnetostatics what Gauss's law is to electrostatics — the key to symmetric problems."*

---

**Next:** Day 207 — Magnetic Vector Potential
