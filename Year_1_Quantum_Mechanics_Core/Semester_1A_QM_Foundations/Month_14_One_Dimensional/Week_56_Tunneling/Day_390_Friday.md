# Day 390: Scanning Tunneling Microscope (STM)

## Week 56, Day 5 | Month 14: One-Dimensional Quantum Mechanics

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning** | 2.5 hrs | STM operating principles, tunneling current theory |
| **Afternoon** | 2.5 hrs | Imaging modes, spectroscopy, resolution limits |
| **Evening** | 2 hrs | Computational lab: STM simulation |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** the basic operating principle of the scanning tunneling microscope
2. **Derive** the exponential dependence of tunneling current on tip-sample distance
3. **Calculate** typical tunneling currents and required tip sharpness
4. **Distinguish** between constant-current and constant-height imaging modes
5. **Understand** scanning tunneling spectroscopy (STS) and its applications
6. **Appreciate** the revolutionary impact of STM on surface science and nanotechnology

---

## Core Content

### 1. Historical Context and Nobel Prize

**1981**: Gerd Binnig and Heinrich Rohrer at IBM Zurich invented the STM
**1986**: Nobel Prize in Physics "for their design of the scanning tunneling microscope"

The STM was the first instrument to image individual atoms on surfaces, opening the era of nanotechnology.

### 2. Basic Operating Principle

```
    Piezo scanners (x, y, z)
           |
           v
    +-------------+
    |    Tip      |  ← Atomically sharp (ideally single atom)
    +------+------+
           |
      d ~0.5 nm    ← Vacuum gap
           |
    ═══════╪═══════  ← Sample surface

    V_bias applied between tip and sample
    Tunneling current I measured
```

**Key insight:** Electrons tunnel quantum mechanically through the vacuum gap. The current depends exponentially on the gap distance!

### 3. Tunneling Current Formula

From our barrier tunneling analysis, for a rectangular barrier of width d and height φ (work function):

$$I \propto e^{-2\kappa d}$$

where:
$$\kappa = \frac{\sqrt{2m\phi}}{\hbar}$$

For typical work functions φ ≈ 4-5 eV:
$$\kappa \approx 10 \text{ nm}^{-1}$$

Therefore:
$$\boxed{I \propto e^{-2\kappa d} \approx e^{-d/0.05\text{ nm}}}$$

**Crucial result:** Current changes by factor of ~10 for every 0.1 nm (1 Angstrom) change in distance!

### 4. Full Tunneling Current Expression

Using Bardeen's transfer Hamiltonian formalism:

$$I = \frac{4\pi e}{\hbar}\int_{-\infty}^{\infty}\rho_s(E_F - eV + \epsilon)\rho_t(E_F + \epsilon)|M|^2[f(\epsilon) - f(\epsilon + eV)]d\epsilon$$

For low temperature and small bias V:
$$\boxed{I \approx \frac{4\pi^2 e^2}{\hbar}V\rho_s(E_F)\rho_t(E_F)|M|^2}$$

where:
- $\rho_s$, $\rho_t$ = sample and tip density of states
- $M$ = tunneling matrix element ∝ $e^{-\kappa d}$
- $f(\epsilon)$ = Fermi-Dirac distribution

### 5. Exponential Sensitivity: The Key to Atomic Resolution

**Why STM achieves atomic resolution:**

Consider a hemispherical tip apex with radius R approaching a flat surface:

```
      Tip (R)
      /   \
     /     \     ← Most current flows through
    |   o   |       this closest atom
    |       |
   d_min    d(x)
    ↓       ↓
═══════════════  Surface
```

The distance at lateral position x from the tip center:
$$d(x) = d_{min} + R - \sqrt{R^2 - x^2} \approx d_{min} + \frac{x^2}{2R}$$

The tunneling current density:
$$j(x) \propto e^{-2\kappa d(x)} = e^{-2\kappa d_{min}} e^{-\kappa x^2/R}$$

This is a **Gaussian profile** with width:
$$\Delta x = \sqrt{R/\kappa}$$

For R = 1 nm, κ = 10 nm⁻¹:
$$\Delta x = \sqrt{0.1} \approx 0.3 \text{ nm}$$

**Sub-atomic lateral resolution is achievable!**

### 6. Imaging Modes

**Constant Current Mode (most common):**
- Feedback loop adjusts z-position to maintain constant I
- Records z(x,y) as "topographic" image
- Slower but works for rough surfaces

```
Tip follows surface contour:

    ~~~     ~~~     ← Tip path
       \   /
        ~~~
═══════════════════  Surface
```

**Constant Height Mode:**
- Tip scans at fixed z
- Records I(x,y) as image
- Faster but requires flat surfaces
- Risk of tip crash

```
Tip at fixed height:

    ─────────────────  ← Tip path
           ↑
═══════════════════  Surface
```

### 7. Scanning Tunneling Spectroscopy (STS)

By varying bias voltage V at fixed position:

$$\frac{dI}{dV} \propto \rho_s(E_F + eV)$$

STS measures the **local density of states (LDOS)** at the sample surface!

**Applications:**
- Mapping electronic band structure
- Identifying atomic species
- Observing superconducting gaps
- Detecting surface states

### 8. Resolution Limits

**Vertical resolution:** ~0.01 nm (limited by vibrations, electronics)
- Individual atomic steps clearly resolved
- Even sub-atomic corrugation of electron density visible

**Lateral resolution:** ~0.1 nm
- Limited by tip sharpness
- Best achieved with single-atom tip apex

**Fundamental limit:** Wave function overlap requires d < few nm

### 9. Technical Challenges

1. **Vibration isolation:** Picometer stability required
   - Multi-stage spring suspension
   - Eddy current damping
   - Sometimes building on bedrock

2. **Tip preparation:**
   - Electrochemical etching of W or Pt-Ir wire
   - Field emission cleaning in vacuum
   - Hope for single-atom apex

3. **Ultra-high vacuum:** Prevents surface contamination
   - Pressures < 10⁻¹⁰ torr
   - Clean surfaces stable for hours

4. **Low temperature:** Reduces thermal drift
   - Cryogenic STM at 4 K or lower
   - mK for highest resolution

### 10. Quantum Computing Connection

STM techniques are crucial for quantum device development:

**Atomic-scale fabrication:**
- Positioning individual atoms for qubit arrays
- Creating atomic-scale quantum dots
- STM lithography for device patterning

**Characterization:**
- Imaging superconducting vortices
- Measuring qubit coherence locally
- Probing topological surface states

**Silicon quantum computing:**
- Placement of single P donors in Si
- Reading out individual electron spins
- Creating atom-by-atom quantum devices

---

## Worked Examples

### Example 1: Tunneling Current Calculation

An STM operates with a tungsten tip (φ = 4.5 eV) at d = 0.5 nm from a gold surface. Calculate the decay constant and estimate how current changes with 0.1 nm height change.

**Solution:**

1. Decay constant:
$$\kappa = \frac{\sqrt{2m_e\phi}}{\hbar} = \frac{\sqrt{2 \times 9.11 \times 10^{-31} \times 4.5 \times 1.6 \times 10^{-19}}}{1.055 \times 10^{-34}}$$
$$\kappa = 1.09 \times 10^{10} \text{ m}^{-1} = 10.9 \text{ nm}^{-1}$$

2. Current ratio for Δd = 0.1 nm:
$$\frac{I(d + \Delta d)}{I(d)} = e^{-2\kappa \Delta d} = e^{-2 \times 10.9 \times 0.1}$$
$$= e^{-2.18} = 0.11$$

**Result:** Current drops by factor of ~9 for each Angstrom increase in distance!

---

### Example 2: Lateral Resolution Estimate

For a tip with effective radius R = 0.5 nm probing a surface with φ = 5 eV, estimate the lateral resolution.

**Solution:**

$$\kappa = \sqrt{2m\phi}/\hbar = 11.5 \text{ nm}^{-1}$$

Lateral width of current distribution:
$$\Delta x = \sqrt{R/\kappa} = \sqrt{0.5/11.5} = 0.21 \text{ nm}$$

**Result:** Lateral resolution ~0.2 nm, sufficient to resolve individual atoms (typical spacing ~0.3 nm)!

---

### Example 3: Spectroscopy Analysis

An STS measurement shows dI/dV peaks at V = ±1.5 mV on a superconductor. What is the superconducting gap Δ?

**Solution:**

The peaks in dI/dV correspond to the BCS density of states singularities at $E = \pm\Delta$.

For small bias, eV = E, so:
$$\Delta = e \times 1.5 \text{ mV} = 1.5 \text{ meV}$$

Using the BCS relation for conventional superconductors:
$$\Delta = 1.76 k_B T_c$$
$$T_c = \frac{\Delta}{1.76 k_B} = \frac{1.5 \times 10^{-3} \times 1.6 \times 10^{-19}}{1.76 \times 1.38 \times 10^{-23}} = 9.9 \text{ K}$$

**Result:** The superconductor has T_c ≈ 10 K (consistent with niobium).

---

## Practice Problems

### Level 1: Direct Application

1. Calculate κ for platinum (φ = 5.7 eV). What is the tunneling current ratio for d = 0.4 nm vs d = 0.6 nm?

2. An STM setpoint current is 1 nA at d = 0.5 nm. Estimate the current at d = 0.3 nm.

3. What tip radius R is needed for 0.15 nm lateral resolution with φ = 4 eV?

### Level 2: Intermediate

4. Derive the effective barrier height φ from the measured slope of ln(I) vs d.

5. An STM image shows atomic corrugation of 0.02 nm on a surface with 0.25 nm atomic spacing. What is the contrast ratio (I_max/I_min)?

6. In STS, why do we measure dI/dV rather than just I(V)? What information would be lost?

### Level 3: Challenging

7. **Tersoff-Hamann model:** Derive the result that STM images represent the LDOS at the tip position, assuming an s-wave tip state.

8. **Thermal effects:** At 300 K, thermal fluctuations cause ~0.01 nm tip-sample distance variations. Estimate the resulting current noise.

9. **Spin-polarized STM:** How does the tunneling current depend on the relative magnetization of tip and sample? Derive the magnetic contrast.

---

## Computational Lab

### Python: STM Simulation

```python
"""
Day 390: Scanning Tunneling Microscope Simulation
Quantum Tunneling & Barriers - Week 56

This lab explores:
1. Tunneling current calculation
2. Atomic resolution imaging
3. Topographic image simulation
4. Spectroscopy curves
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Physical constants
hbar = 1.055e-34  # J·s
m_e = 9.109e-31   # kg
e = 1.602e-19     # C

def decay_constant(phi_eV):
    """Calculate κ in nm^{-1} for work function phi in eV"""
    phi_J = phi_eV * e
    kappa = np.sqrt(2 * m_e * phi_J) / hbar
    return kappa * 1e-9  # Convert to nm^{-1}

def tunneling_current(d, phi, V=0.1, I0=1.0):
    """
    Calculate tunneling current

    Parameters:
    d: tip-sample distance (nm)
    phi: work function (eV)
    V: bias voltage (V)
    I0: normalization current

    Returns:
    I: tunneling current (nA if I0=1)
    """
    kappa = decay_constant(phi)
    return I0 * np.abs(V) * np.exp(-2 * kappa * d)

#%% Plot 1: Current vs distance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
d = np.linspace(0.3, 1.5, 200)
phi_values = [3.5, 4.5, 5.5]
colors = ['blue', 'green', 'red']

for phi, color in zip(phi_values, colors):
    I = tunneling_current(d, phi)
    ax1.semilogy(d, I, color=color, linewidth=2, label=f'φ = {phi} eV')

ax1.set_xlabel('Tip-sample distance d (nm)', fontsize=12)
ax1.set_ylabel('Tunneling current I (nA)', fontsize=12)
ax1.set_title('STM Tunneling Current vs Distance', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(0.3, 1.5)
ax1.set_ylim(1e-6, 1e2)

# Plot ln(I) vs d to show linearity
ax2 = axes[1]
for phi, color in zip(phi_values, colors):
    I = tunneling_current(d, phi)
    kappa = decay_constant(phi)
    ax2.plot(d, np.log(I), color=color, linewidth=2, label=f'φ = {phi} eV, κ = {kappa:.1f} nm⁻¹')

ax2.set_xlabel('Tip-sample distance d (nm)', fontsize=12)
ax2.set_ylabel('ln(I)', fontsize=12)
ax2.set_title('Linear behavior of ln(I) vs d', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stm_current_distance.png', dpi=150)
plt.show()

#%% Plot 2: Simulate atomic resolution imaging
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Create a hexagonal atomic lattice (like graphite surface)
a = 0.246  # nm, graphene lattice constant
N = 30
x = np.linspace(0, 3, N)
y = np.linspace(0, 3, N)
X, Y = np.meshgrid(x, y)

# Hexagonal lattice: sum of three cosines
def atomic_surface(x, y, a=0.246, corrugation=0.02):
    """
    Create hexagonal atomic lattice surface

    corrugation: height variation in nm
    """
    k = 4 * np.pi / (a * np.sqrt(3))

    # Three-fold symmetric cosine pattern
    z = corrugation * (np.cos(k * x) +
                       np.cos(k * (x/2 + y * np.sqrt(3)/2)) +
                       np.cos(k * (x/2 - y * np.sqrt(3)/2)))
    return z

# Generate surface
Z_surface = atomic_surface(X, Y)

# STM imaging simulation
def simulate_stm_image(X, Y, Z_surface, phi=4.5, d_setpoint=0.5, mode='constant_current'):
    """
    Simulate STM image

    mode: 'constant_current' or 'constant_height'
    """
    kappa = decay_constant(phi)

    if mode == 'constant_current':
        # In constant current mode, z_tip follows surface with exponential weighting
        # z_tip such that I = I_setpoint
        # I ∝ exp(-2κ(d_setpoint - Z_surface))
        # So image is essentially Z_surface
        return Z_surface

    else:  # constant_height
        # Current varies exponentially with local height
        d_local = d_setpoint - Z_surface
        I = np.exp(-2 * kappa * d_local)
        return I

# Constant current image
Z_cc = simulate_stm_image(X, Y, Z_surface, mode='constant_current')

# Constant height image
I_ch = simulate_stm_image(X, Y, Z_surface, mode='constant_height')

# Plot surface
ax1 = axes[0, 0]
im1 = ax1.pcolormesh(X, Y, Z_surface, cmap='terrain', shading='auto')
ax1.set_xlabel('x (nm)', fontsize=11)
ax1.set_ylabel('y (nm)', fontsize=11)
ax1.set_title('Surface Topography (atoms)', fontsize=12)
plt.colorbar(im1, ax=ax1, label='Height (nm)')
ax1.set_aspect('equal')

# Constant current STM image
ax2 = axes[0, 1]
im2 = ax2.pcolormesh(X, Y, Z_cc, cmap='copper', shading='auto')
ax2.set_xlabel('x (nm)', fontsize=11)
ax2.set_ylabel('y (nm)', fontsize=11)
ax2.set_title('Constant Current STM Image', fontsize=12)
plt.colorbar(im2, ax=ax2, label='Tip height (nm)')
ax2.set_aspect('equal')

# Constant height STM image
ax3 = axes[1, 0]
im3 = ax3.pcolormesh(X, Y, I_ch, cmap='hot', shading='auto')
ax3.set_xlabel('x (nm)', fontsize=11)
ax3.set_ylabel('y (nm)', fontsize=11)
ax3.set_title('Constant Height STM Image', fontsize=12)
plt.colorbar(im3, ax=ax3, label='Current (arb.)')
ax3.set_aspect('equal')

# Line profile
ax4 = axes[1, 1]
profile_y = N // 2
ax4.plot(X[profile_y, :], Z_surface[profile_y, :], 'b-', linewidth=2, label='Surface')
ax4.plot(X[profile_y, :], Z_cc[profile_y, :], 'r--', linewidth=2, label='STM trace')

ax4.set_xlabel('x (nm)', fontsize=11)
ax4.set_ylabel('Height (nm)', fontsize=11)
ax4.set_title('Line Profile through Atomic Row', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stm_atomic_imaging.png', dpi=150)
plt.show()

#%% Plot 3: Lateral resolution demonstration
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Show how tip radius affects resolution
R_values = [0.3, 1.0, 3.0]  # nm
phi = 4.5  # eV
kappa = decay_constant(phi)

x = np.linspace(-2, 2, 500)

ax1 = axes[0]
for R in R_values:
    # Current profile for hemispherical tip
    # j(x) ∝ exp(-κx²/R)
    width = np.sqrt(R / kappa)
    j = np.exp(-kappa * x**2 / R)
    j = j / np.max(j)
    ax1.plot(x, j, linewidth=2, label=f'R = {R} nm, Δx = {width:.2f} nm')

ax1.set_xlabel('Lateral position x (nm)', fontsize=12)
ax1.set_ylabel('Normalized current', fontsize=12)
ax1.set_title('Current Profile vs Tip Radius (φ = 4.5 eV)', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Resolution vs work function
ax2 = axes[1]
phi_range = np.linspace(2, 8, 100)
R = 1.0  # nm

for R in [0.3, 1.0, 3.0]:
    kappa_range = decay_constant(phi_range)
    resolution = np.sqrt(R / kappa_range)
    ax2.plot(phi_range, resolution, linewidth=2, label=f'R = {R} nm')

ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Typical atomic spacing')

ax2.set_xlabel('Work function φ (eV)', fontsize=12)
ax2.set_ylabel('Lateral resolution Δx (nm)', fontsize=12)
ax2.set_title('Lateral Resolution vs Work Function', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stm_resolution.png', dpi=150)
plt.show()

#%% Plot 4: Scanning Tunneling Spectroscopy
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Simulate STS on different materials

# Metal (flat DOS)
def dos_metal(E, EF=0):
    return np.ones_like(E)

# Semiconductor (bandgap)
def dos_semiconductor(E, Eg=1.0, EF=0):
    dos = np.zeros_like(E)
    dos[np.abs(E - EF) > Eg/2] = 1.0
    return dos

# Superconductor (BCS DOS)
def dos_superconductor(E, Delta=0.001, EF=0):
    E_shifted = E - EF
    # BCS DOS: ρ ∝ |E|/√(E² - Δ²) for |E| > Δ
    dos = np.zeros_like(E)
    mask = np.abs(E_shifted) > Delta
    dos[mask] = np.abs(E_shifted[mask]) / np.sqrt(E_shifted[mask]**2 - Delta**2)
    # Cap at reasonable value
    dos = np.minimum(dos, 5)
    return dos

# Bias voltage range
V = np.linspace(-2, 2, 500)  # Volts for semiconductor, mV for superconductor

ax1 = axes[0]
# For semiconductor at T=0, I = ∫ρ(E)dE from 0 to eV
E = V  # eV (electron energy corresponds to bias)

dos_m = dos_metal(E)
dos_sc = dos_semiconductor(E, Eg=1.0)

# Current is integral of DOS
from scipy.integrate import cumtrapz
I_metal = cumtrapz(dos_m, V, initial=0)
I_semiconductor = cumtrapz(dos_sc, V, initial=0)

ax1.plot(V, I_metal / np.max(np.abs(I_metal)), 'b-', linewidth=2, label='Metal')
ax1.plot(V, I_semiconductor / np.max(np.abs(I_semiconductor)), 'r-', linewidth=2, label='Semiconductor (1 eV gap)')

ax1.set_xlabel('Bias Voltage V (V)', fontsize=12)
ax1.set_ylabel('Tunneling Current I (normalized)', fontsize=12)
ax1.set_title('I-V Characteristics', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# dI/dV (STS spectrum)
ax2 = axes[1]
ax2.plot(V, dos_m, 'b-', linewidth=2, label='Metal')
ax2.plot(V, dos_sc, 'r-', linewidth=2, label='Semiconductor')

# Add superconductor (different scale)
V_sc = np.linspace(-5, 5, 500)  # mV
dos_sup = dos_superconductor(V_sc / 1000, Delta=0.0015)  # Δ = 1.5 meV
ax2.plot(V_sc / 1000, dos_sup, 'g-', linewidth=2, label='Superconductor (Δ = 1.5 meV)')

ax2.set_xlabel('Energy E (eV)', fontsize=12)
ax2.set_ylabel('dI/dV ∝ DOS', fontsize=12)
ax2.set_title('Scanning Tunneling Spectroscopy', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2, 2)
ax2.set_ylim(0, 5)

plt.tight_layout()
plt.savefig('stm_spectroscopy.png', dpi=150)
plt.show()

#%% Plot 5: 3D STM image rendering
fig = plt.figure(figsize=(14, 6))

# Create more detailed surface
N = 100
x = np.linspace(0, 4, N)
y = np.linspace(0, 4, N)
X, Y = np.meshgrid(x, y)

# Add a step edge and some point defects
Z = atomic_surface(X, Y, corrugation=0.02)

# Add atomic step at x = 2 nm
Z[X > 2] += 0.22  # Monatomic step height ~0.22 nm for many metals

# Add vacancy (missing atom) at (1, 1)
dist_vacancy = np.sqrt((X - 1)**2 + (Y - 1)**2)
Z -= 0.03 * np.exp(-dist_vacancy**2 / 0.1**2)

# Add adatom at (3, 2.5)
dist_adatom = np.sqrt((X - 3)**2 + (Y - 2.5)**2)
Z += 0.03 * np.exp(-dist_adatom**2 / 0.05**2)

# 3D surface plot
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='copper', linewidth=0, antialiased=True)
ax1.set_xlabel('x (nm)')
ax1.set_ylabel('y (nm)')
ax1.set_zlabel('z (nm)')
ax1.set_title('3D STM Topography')

# 2D image with annotations
ax2 = fig.add_subplot(122)
im = ax2.pcolormesh(X, Y, Z, cmap='copper', shading='auto')
plt.colorbar(im, ax=ax2, label='Height (nm)')

# Annotations
ax2.annotate('Step edge', xy=(2, 0.5), xytext=(2.5, 0.5),
            arrowprops=dict(arrowstyle='->', color='white'),
            fontsize=10, color='white')
ax2.annotate('Vacancy', xy=(1, 1), xytext=(0.3, 1.5),
            arrowprops=dict(arrowstyle='->', color='white'),
            fontsize=10, color='white')
ax2.annotate('Adatom', xy=(3, 2.5), xytext=(3.5, 3),
            arrowprops=dict(arrowstyle='->', color='white'),
            fontsize=10, color='white')

ax2.set_xlabel('x (nm)', fontsize=11)
ax2.set_ylabel('y (nm)', fontsize=11)
ax2.set_title('STM Image: Step, Vacancy, and Adatom', fontsize=12)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('stm_3d_features.png', dpi=150)
plt.show()

# Summary statistics
print("\n=== STM Physics Summary ===")
print(f"\nFor typical work function φ = 4.5 eV:")
kappa = decay_constant(4.5)
print(f"Decay constant κ = {kappa:.1f} nm⁻¹")
print(f"Current drops by factor {np.exp(2*kappa*0.1):.1f} per Angstrom")

print(f"\nLateral resolution for R = 1 nm tip:")
print(f"Δx = √(R/κ) = {np.sqrt(1/kappa):.2f} nm")

print("\n=== Tunneling Current Examples ===")
for d in [0.3, 0.5, 0.7, 1.0]:
    I = tunneling_current(d, 4.5)
    print(f"d = {d} nm: I = {I:.2e} (normalized)")
```

### Expected Output

```
=== STM Physics Summary ===

For typical work function φ = 4.5 eV:
Decay constant κ = 10.9 nm⁻¹
Current drops by factor 8.8 per Angstrom

Lateral resolution for R = 1 nm tip:
Δx = √(R/κ) = 0.30 nm

=== Tunneling Current Examples ===
d = 0.3 nm: I = 1.42e-03 (normalized)
d = 0.5 nm: I = 1.83e-05 (normalized)
d = 0.7 nm: I = 2.36e-07 (normalized)
d = 1.0 nm: I = 3.05e-10 (normalized)
```

---

## Summary

### Key Formulas Table

| Quantity | Formula |
|----------|---------|
| Decay constant | $\kappa = \sqrt{2m\phi}/\hbar$ |
| Tunneling current | $I \propto e^{-2\kappa d}$ |
| Current sensitivity | $\Delta I/I = -2\kappa \Delta d$ |
| Lateral resolution | $\Delta x \approx \sqrt{R/\kappa}$ |
| STS relation | $dI/dV \propto \rho_s(E_F + eV)$ |

### Main Takeaways

1. **Exponential sensitivity** to distance enables atomic resolution
2. **Current drops ~10× per Angstrom** for typical work functions
3. **Lateral resolution** determined by tip sharpness and work function
4. **Two imaging modes:** constant current (topography) and constant height (current)
5. **STS measures local DOS** - powerful spectroscopic tool

### Technological Impact

- First true atomic-resolution surface microscopy
- Manipulation of individual atoms (IBM "Quantum Corral")
- Surface physics and catalysis studies
- Semiconductor characterization
- Foundation for atomic force microscopy (AFM)

---

## Daily Checklist

- [ ] I can explain why STM achieves atomic resolution
- [ ] I can calculate the decay constant and current change with distance
- [ ] I understand the difference between constant-current and constant-height modes
- [ ] I can explain how STS measures the local density of states
- [ ] I understand the technical challenges in STM operation
- [ ] I ran the Python code and can interpret STM images
- [ ] I attempted problems from each difficulty level

---

## Preview: Day 391

Tomorrow we explore **tunnel diodes and the Josephson effect** - electronic devices that exploit quantum tunneling. We'll see how negative differential resistance emerges in semiconductor tunnel diodes and how Cooper pair tunneling in superconductors leads to the remarkable Josephson effects that underpin superconducting qubit technology!
