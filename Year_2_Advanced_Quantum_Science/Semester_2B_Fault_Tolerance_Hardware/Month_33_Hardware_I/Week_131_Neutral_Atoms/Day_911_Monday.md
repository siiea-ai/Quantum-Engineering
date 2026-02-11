# Day 911: Optical Tweezer Arrays

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Dipole trapping physics, AOD/SLM technology |
| **Afternoon** | 2 hours | Problem solving: trap design and optimization |
| **Evening** | 2 hours | Computational lab: tweezer array simulation |

## Learning Objectives

By the end of this day, you will be able to:

1. **Derive the optical dipole potential** from the AC Stark shift and atomic polarizability
2. **Calculate trap frequencies and depths** for alkali atoms in focused Gaussian beams
3. **Compare AOD and SLM approaches** for generating reconfigurable atom arrays
4. **Design tweezer array geometries** for specific quantum computing applications
5. **Analyze heating mechanisms** and atom lifetime limitations in optical traps
6. **Implement numerical simulations** of trap potentials and atomic motion

## Core Content

### 1. Optical Dipole Force Fundamentals

#### AC Stark Shift and Induced Dipole

When an atom interacts with an off-resonant light field, the oscillating electric field induces a dipole moment. The interaction energy between this induced dipole and the field creates a position-dependent potential.

The induced dipole moment is:
$$\mathbf{d} = \alpha(\omega)\mathbf{E}$$

where $\alpha(\omega)$ is the frequency-dependent atomic polarizability. The interaction potential becomes:
$$U(\mathbf{r}) = -\frac{1}{2}\langle\mathbf{d}\cdot\mathbf{E}\rangle = -\frac{1}{2\epsilon_0 c}\text{Re}[\alpha(\omega)]I(\mathbf{r})$$

#### Two-Level Atom Polarizability

For a two-level atom with resonance frequency $\omega_0$ and natural linewidth $\Gamma$:
$$\alpha(\omega) = 6\pi\epsilon_0 c^3 \frac{\Gamma/\omega_0^3}{\omega_0^2 - \omega^2 - i(\omega^3/\omega_0^3)\Gamma}$$

In the far-detuned limit $|\Delta| = |\omega - \omega_0| \gg \Gamma$:
$$\text{Re}[\alpha] \approx \frac{3\pi\epsilon_0 c^3\Gamma}{\omega_0^3\Delta}$$

This gives the **dipole potential**:
$$\boxed{U(\mathbf{r}) = \frac{3\pi c^2}{2\omega_0^3}\frac{\Gamma}{\Delta}I(\mathbf{r})}$$

Key observations:
- **Red detuning** ($\Delta < 0$): Atoms attracted to intensity maxima (trapping at focus)
- **Blue detuning** ($\Delta > 0$): Atoms repelled from intensity maxima (potential barriers)

#### Photon Scattering Rate

Spontaneous emission from the excited state causes heating. The scattering rate is:
$$\Gamma_{sc}(\mathbf{r}) = \frac{3\pi c^2}{2\hbar\omega_0^3}\left(\frac{\Gamma}{\Delta}\right)^2 I(\mathbf{r})$$

The ratio of trap depth to scattering rate scales as:
$$\frac{U}{\hbar\Gamma_{sc}} = \frac{\Delta}{\Gamma}$$

This motivates using **large detunings** for long trap lifetimes.

### 2. Gaussian Beam Optical Tweezers

#### Focused Gaussian Beam Intensity

A Gaussian beam focused by a lens with numerical aperture NA creates an intensity distribution:
$$I(r, z) = I_0\left(\frac{w_0}{w(z)}\right)^2 \exp\left(-\frac{2r^2}{w(z)^2}\right)$$

where:
- $w_0 = \frac{\lambda}{\pi \cdot \text{NA}}$ is the beam waist
- $w(z) = w_0\sqrt{1 + (z/z_R)^2}$ is the beam radius at position $z$
- $z_R = \pi w_0^2/\lambda$ is the Rayleigh range

#### Trap Frequencies

Near the trap center, the potential is approximately harmonic:
$$U(r, z) \approx -U_0 + \frac{1}{2}m\omega_r^2 r^2 + \frac{1}{2}m\omega_z^2 z^2$$

The trap frequencies are:
$$\boxed{\omega_r = \sqrt{\frac{4U_0}{mw_0^2}}, \quad \omega_z = \sqrt{\frac{2U_0}{mz_R^2}}}$$

The aspect ratio is:
$$\frac{\omega_r}{\omega_z} = \sqrt{2}\frac{z_R}{w_0} = \sqrt{2}\frac{\pi w_0}{\lambda}$$

For typical tweezers with $w_0 \sim 1\,\mu\text{m}$ and $\lambda \sim 1\,\mu\text{m}$: $\omega_r/\omega_z \sim 5$.

#### Numerical Example: Rb-87 in 1064 nm Tweezer

For rubidium-87 with:
- Wavelength: $\lambda = 1064\,\text{nm}$ (Nd:YAG)
- Detuning from D2 line: $\Delta \approx 2\pi \times 100\,\text{THz}$
- Power: $P = 10\,\text{mW}$
- Waist: $w_0 = 0.9\,\mu\text{m}$

Peak intensity:
$$I_0 = \frac{2P}{\pi w_0^2} = \frac{2 \times 10^{-2}}{\pi(0.9\times 10^{-6})^2} \approx 7.9 \times 10^9\,\text{W/m}^2$$

Trap depth:
$$U_0/k_B \approx 1\,\text{mK}$$

Trap frequencies:
$$\omega_r \approx 2\pi \times 100\,\text{kHz}, \quad \omega_z \approx 2\pi \times 20\,\text{kHz}$$

### 3. Array Generation Technologies

#### Acousto-Optic Deflectors (AODs)

AODs use acoustic waves in a crystal to diffract light at controllable angles.

**Operating principle:**
1. RF signal creates traveling acoustic wave in crystal (typically TeO₂)
2. Light diffracts from acoustic grating
3. Deflection angle: $\theta = \lambda f/v_s$ where $f$ is RF frequency, $v_s$ is sound velocity

**Multi-tone operation:**
- Multiple RF frequencies create multiple diffracted beams
- 1D AOD: linear array of spots
- Crossed AODs: 2D rectangular array

**Advantages:**
- Fast reconfiguration (microseconds)
- High efficiency (>80% per tone)
- Continuous position control

**Limitations:**
- Power variation across tones
- Intermodulation distortion
- Limited number of spots (~10-20 per dimension)

#### Spatial Light Modulators (SLMs)

SLMs use liquid crystal arrays to imprint arbitrary phase patterns on the beam.

**Holographic generation:**
The phase pattern $\phi(x, y)$ is computed to generate desired intensity distribution $I(\mathbf{r})$ at the focal plane via Gerchberg-Saxton algorithm:

1. Start with target intensity, random phase
2. Propagate to SLM plane (inverse FFT)
3. Apply SLM constraint (phase-only modulation)
4. Propagate to focal plane (FFT)
5. Apply target intensity constraint
6. Iterate until convergence

**Advantages:**
- Arbitrary array geometries
- Large atom numbers (1000+)
- Complex potential landscapes

**Limitations:**
- Slower update rate (60-1000 Hz)
- Speckle from phase discontinuities
- Lower efficiency (~50-70%)

### 4. Trap Loading and Atom Lifetime

#### Loading from MOT

Typical loading sequence:
1. Laser cool atoms in magneto-optical trap (MOT)
2. Compress MOT to increase density
3. Optical molasses for sub-Doppler cooling
4. Turn on tweezers to capture atoms

**Single-atom loading:**
- Collisional blockade ensures 0 or 1 atom per trap
- Loading probability ~50% (parity projection)
- Light-assisted collisions eject pairs

#### Atom Lifetime Limitations

**Photon scattering:**
$$\tau_{sc}^{-1} = \Gamma_{sc} = \frac{3\pi c^2}{2\hbar\omega_0^3}\left(\frac{\Gamma}{\Delta}\right)^2 I$$

**Background gas collisions:**
$$\tau_{bg} \propto \frac{1}{n_{bg}\sigma v}$$

At $10^{-11}$ Torr: $\tau_{bg} \sim 100$ s

**Parametric heating:**
Intensity fluctuations at $2\omega_r$ or $2\omega_z$ drive parametric heating:
$$\dot{E} = \frac{\pi^2 \omega_{trap}^2}{2}S_I(2\omega_{trap})E$$

where $S_I$ is the relative intensity noise spectral density.

### 5. Quantum Computing Applications

#### Array Architectures

**1D chains:**
- Simple connectivity
- Ideal for studying 1D physics
- Limited for error correction

**2D square lattice:**
- Natural for surface codes
- Typical spacing: 3-5 μm
- Local addressing possible

**2D triangular/honeycomb:**
- Frustrated systems
- Topological codes

**3D arrays:**
- Enhanced connectivity
- Recent demonstrations with SLMs
- Challenging for imaging

#### Scalability Considerations

Current demonstrations:
- QuEra: 256 atoms in 16×16 array
- Atom Computing: 1200+ sites
- Pasqal: 200+ atoms

Scaling challenges:
- Optical power (scales linearly with atom number)
- Aberration correction
- Imaging system complexity
- Vacuum system size

## Worked Examples

### Example 1: Designing a 1064 nm Tweezer for Cesium

**Problem:** Design an optical tweezer for Cs-133 atoms using a 1064 nm laser. The trap should have depth >0.5 mK and radial frequency >50 kHz. Calculate required power and beam waist.

**Solution:**

**Step 1: Relevant Cs-133 parameters**
- D2 transition: $\lambda_0 = 852$ nm, $\Gamma = 2\pi \times 5.2$ MHz
- Mass: $m = 133$ amu $= 2.21 \times 10^{-25}$ kg
- Detuning: $\Delta = 2\pi c(1/852 - 1/1064) \times 10^9 = 2\pi \times 70$ THz

**Step 2: Polarizability calculation**
The scalar polarizability at 1064 nm:
$$\alpha = \frac{3\pi\epsilon_0 c^3 \Gamma}{\omega_0^3}\frac{1}{\Delta} \approx 1.0 \times 10^{-38}\,\text{C}^2\text{m}^2/\text{J}$$

**Step 3: Trap depth relation**
$$U_0 = \frac{\alpha I_0}{2\epsilon_0 c} = \frac{\alpha P}{\pi\epsilon_0 c w_0^2}$$

For $U_0/k_B = 0.5$ mK:
$$U_0 = 0.5 \times 10^{-3} \times 1.38 \times 10^{-23} = 6.9 \times 10^{-27}\,\text{J}$$

**Step 4: Radial frequency constraint**
$$\omega_r = \sqrt{\frac{4U_0}{mw_0^2}} > 2\pi \times 50\,\text{kHz}$$

This gives:
$$w_0 < \sqrt{\frac{4U_0}{m(2\pi \times 50000)^2}} = \sqrt{\frac{4 \times 6.9 \times 10^{-27}}{2.21 \times 10^{-25} \times (3.14 \times 10^5)^2}} \approx 1.1\,\mu\text{m}$$

**Step 5: Required power**
Choosing $w_0 = 1.0\,\mu\text{m}$:
$$P = \frac{\pi\epsilon_0 c w_0^2 U_0}{\alpha} = \frac{\pi \times 8.85 \times 10^{-12} \times 3 \times 10^8 \times (10^{-6})^2 \times 6.9 \times 10^{-27}}{1.0 \times 10^{-38}}$$
$$P \approx 5.8\,\text{mW}$$

**Result:** Use $w_0 = 1.0\,\mu\text{m}$ and $P = 6$ mW to achieve $U_0/k_B = 0.52$ mK and $\omega_r = 2\pi \times 54$ kHz.

---

### Example 2: AOD Array Design

**Problem:** Design an AOD system to create a 10-site linear array with 4 μm spacing. The AOD crystal has sound velocity 4200 m/s, and the beam is focused by a 50 mm lens with 1064 nm light. Calculate the required RF frequencies.

**Solution:**

**Step 1: Deflection angle to position**
For small angles, the position in the focal plane is:
$$x = f\theta = f\frac{\lambda f_{RF}}{v_s}$$

**Step 2: Position spacing**
$$\Delta x = f\frac{\lambda \Delta f_{RF}}{v_s}$$

For $\Delta x = 4\,\mu\text{m}$:
$$\Delta f_{RF} = \frac{v_s \Delta x}{f\lambda} = \frac{4200 \times 4 \times 10^{-6}}{0.05 \times 1064 \times 10^{-9}} = 316\,\text{kHz}$$

**Step 3: RF frequency array**
Centering around 100 MHz typical operating point:
- Site 1: 98.58 MHz
- Site 2: 98.90 MHz
- ...
- Site 10: 101.42 MHz

**Step 4: Bandwidth check**
Total bandwidth: $9 \times 316$ kHz $= 2.84$ MHz

This is well within typical AOD bandwidth (>10 MHz), so the design is feasible.

---

### Example 3: Trap Lifetime Estimation

**Problem:** Calculate the atom lifetime limited by photon scattering for Rb-87 in a 1064 nm tweezer with 1 mK depth.

**Solution:**

**Step 1: Calculate intensity from trap depth**
$$U_0 = \frac{3\pi c^2}{2\omega_0^3}\frac{\Gamma}{\Delta}I_0$$

Solving for $I_0$:
$$I_0 = \frac{2\omega_0^3 U_0 \Delta}{3\pi c^2 \Gamma}$$

With $U_0 = k_B \times 1$ mK $= 1.38 \times 10^{-26}$ J, $\omega_0 = 2\pi c/780$ nm, $\Gamma = 2\pi \times 6$ MHz, $\Delta = 2\pi \times 100$ THz:

$$I_0 \approx 7.9 \times 10^9\,\text{W/m}^2$$

**Step 2: Scattering rate at trap center**
$$\Gamma_{sc} = \frac{3\pi c^2}{2\hbar\omega_0^3}\left(\frac{\Gamma}{\Delta}\right)^2 I_0$$
$$\Gamma_{sc} = \frac{U_0}{\hbar}\frac{\Gamma}{\Delta} = \frac{1.38 \times 10^{-26}}{1.05 \times 10^{-34}} \times \frac{6 \times 10^6}{10^{14}}$$
$$\Gamma_{sc} \approx 8\,\text{s}^{-1}$$

**Step 3: Scattering-limited lifetime**
$$\tau_{sc} = \frac{1}{\Gamma_{sc}} \approx 125\,\text{ms}$$

With improved vacuum (10⁻¹¹ Torr) and intensity stabilization, total lifetime can exceed 10 s.

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the trap depth and radial frequency for a Rb-87 atom in a 850 nm tweezer (blue-detuned from D1, red-detuned from D2) with 5 mW power and 1 μm waist.

**Problem 1.2:** An AOD has 5 MHz RF bandwidth centered at 80 MHz. What is the maximum array extent for 5 μm site spacing using a 100 mm focal length lens at 1064 nm?

**Problem 1.3:** Compare the scattering-limited trap lifetime for tweezers at 852 nm (near D2) vs 1064 nm for Cs atoms with 0.5 mK trap depth.

### Level 2: Intermediate Analysis

**Problem 2.1:** A 2D array uses crossed AODs with 10 tones each. If the total laser power is 1 W and the diffraction efficiency per tone is 70%, calculate the power per tweezer. What trap depth does this give for Rb-87 with 0.9 μm waist at 1064 nm?

**Problem 2.2:** An SLM-based system has 5% RMS intensity variation across a 100-site array. Calculate the variation in trap frequencies and estimate the impact on global single-qubit gate fidelity if the gate is calibrated for the mean trap depth.

**Problem 2.3:** Design a "magic wavelength" tweezer for Rb-87 where the trap potential is identical for the ground state and first excited Rydberg manifold. Research the approximate magic wavelength and calculate required power for 100 kHz trap frequency.

### Level 3: Challenging Problems

**Problem 3.1:** Derive the optimal detuning for a dipole trap that maximizes the ratio of coherence time to gate time, assuming the gate time scales as $1/\omega_r$ and coherence is limited by scattering.

**Problem 3.2:** A triangular lattice array is generated using three-beam interference from an SLM. Calculate the phase pattern required to generate a 20×20 triangular array with 5 μm spacing. Consider the Gerchberg-Saxton algorithm convergence.

**Problem 3.3:** Analyze the heating rate from relative intensity noise (RIN) for a trap with $\omega_r = 2\pi \times 100$ kHz. If the laser has RIN of -140 dBc/Hz at 200 kHz, calculate the time for the atom to heat from the ground state to the first excited motional state.

## Computational Lab: Tweezer Array Simulation

### Lab 1: Gaussian Tweezer Potential

```python
"""
Day 911 Lab: Optical Tweezer Array Simulation
Simulating dipole trap potentials and array generation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar, k, epsilon_0, atomic_mass
from mpl_toolkits.mplot3d import Axes3D

# Physical constants for Rb-87
class Rb87:
    mass = 87 * atomic_mass  # kg
    lambda_D2 = 780e-9  # m
    Gamma_D2 = 2 * np.pi * 6.07e6  # rad/s
    omega_D2 = 2 * np.pi * c / lambda_D2  # rad/s

def calculate_polarizability(wavelength, atom=Rb87):
    """
    Calculate the atomic polarizability at given wavelength.
    Uses simplified two-level model.
    """
    omega = 2 * np.pi * c / wavelength
    Delta = omega - atom.omega_D2

    # Real part of polarizability (far-detuned limit)
    alpha = 3 * np.pi * epsilon_0 * c**3 * atom.Gamma_D2 / (atom.omega_D2**3 * Delta)
    return alpha

def gaussian_beam_intensity(x, y, z, P, w0, wavelength):
    """
    Calculate intensity distribution of focused Gaussian beam.

    Parameters:
    -----------
    x, y, z : arrays
        Position coordinates (m)
    P : float
        Total beam power (W)
    w0 : float
        Beam waist at focus (m)
    wavelength : float
        Wavelength (m)

    Returns:
    --------
    I : array
        Intensity distribution (W/m^2)
    """
    z_R = np.pi * w0**2 / wavelength  # Rayleigh range
    w_z = w0 * np.sqrt(1 + (z/z_R)**2)  # Beam radius at z
    r_sq = x**2 + y**2

    I0 = 2 * P / (np.pi * w0**2)  # Peak intensity
    I = I0 * (w0/w_z)**2 * np.exp(-2 * r_sq / w_z**2)

    return I

def dipole_potential(x, y, z, P, w0, wavelength, atom=Rb87):
    """
    Calculate dipole trap potential.

    Returns potential in units of temperature (K) when divided by k_B.
    """
    alpha = calculate_polarizability(wavelength, atom)
    I = gaussian_beam_intensity(x, y, z, P, w0, wavelength)

    # Potential: U = -alpha * I / (2 * epsilon_0 * c)
    U = -alpha * I / (2 * epsilon_0 * c)

    return U

def trap_frequencies(U0, w0, z_R, mass):
    """
    Calculate radial and axial trap frequencies.

    Parameters:
    -----------
    U0 : float
        Trap depth (J, negative for attractive trap)
    w0 : float
        Beam waist (m)
    z_R : float
        Rayleigh range (m)
    mass : float
        Atom mass (kg)
    """
    omega_r = np.sqrt(4 * np.abs(U0) / (mass * w0**2))
    omega_z = np.sqrt(2 * np.abs(U0) / (mass * z_R**2))

    return omega_r, omega_z

# Simulation parameters
wavelength = 1064e-9  # m (Nd:YAG)
power = 10e-3  # W
waist = 0.9e-6  # m

# Calculate key parameters
alpha = calculate_polarizability(wavelength)
z_R = np.pi * waist**2 / wavelength
I0 = 2 * power / (np.pi * waist**2)
U0 = -alpha * I0 / (2 * epsilon_0 * c)

print("=== Single Tweezer Parameters ===")
print(f"Wavelength: {wavelength*1e9:.0f} nm")
print(f"Power: {power*1e3:.1f} mW")
print(f"Waist: {waist*1e6:.2f} μm")
print(f"Rayleigh range: {z_R*1e6:.2f} μm")
print(f"Peak intensity: {I0:.2e} W/m²")
print(f"Trap depth: {-U0/k:.3f} mK")

omega_r, omega_z = trap_frequencies(U0, waist, z_R, Rb87.mass)
print(f"Radial frequency: {omega_r/(2*np.pi)/1e3:.1f} kHz")
print(f"Axial frequency: {omega_z/(2*np.pi)/1e3:.1f} kHz")

# Scattering rate
Delta = 2 * np.pi * c / wavelength - Rb87.omega_D2
Gamma_sc = (Rb87.Gamma_D2 / Delta)**2 * np.abs(U0) / hbar
print(f"Scattering rate: {Gamma_sc:.1f} s⁻¹")
print(f"Scattering-limited lifetime: {1/Gamma_sc*1e3:.0f} ms")

# Plot 2D potential
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Radial cut (z=0)
r = np.linspace(-3*waist, 3*waist, 200)
z_vals = [0, z_R/2, z_R]
for z_val in z_vals:
    U_r = dipole_potential(r, 0, z_val, power, waist, wavelength)
    axes[0].plot(r*1e6, U_r/k*1e3, label=f'z = {z_val/z_R:.1f} z_R')

axes[0].set_xlabel('Radial position (μm)')
axes[0].set_ylabel('Potential (mK)')
axes[0].set_title('Radial Potential Profile')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Axial cut (r=0)
z = np.linspace(-5*z_R, 5*z_R, 200)
U_z = dipole_potential(0, 0, z, power, waist, wavelength)
axes[1].plot(z*1e6, U_z/k*1e3)
axes[1].set_xlabel('Axial position (μm)')
axes[1].set_ylabel('Potential (mK)')
axes[1].set_title('Axial Potential Profile')
axes[1].grid(True, alpha=0.3)

# 2D cross-section
r_2d = np.linspace(-3*waist, 3*waist, 100)
z_2d = np.linspace(-5*z_R, 5*z_R, 100)
R, Z = np.meshgrid(r_2d, z_2d)
U_2d = dipole_potential(R, 0, Z, power, waist, wavelength)

im = axes[2].pcolormesh(R*1e6, Z*1e6, U_2d/k*1e3, shading='auto', cmap='viridis')
axes[2].set_xlabel('Radial position (μm)')
axes[2].set_ylabel('Axial position (μm)')
axes[2].set_title('2D Potential Landscape')
plt.colorbar(im, ax=axes[2], label='Potential (mK)')

plt.tight_layout()
plt.savefig('single_tweezer_potential.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== Tweezer Array Generation ===")

# Simulate 2D tweezer array
def create_tweezer_array(positions, power_per_site, waist, wavelength, grid_x, grid_y, z=0):
    """
    Create potential from array of tweezers.

    Parameters:
    -----------
    positions : list of (x, y) tuples
        Tweezer center positions (m)
    power_per_site : float
        Power per tweezer (W)
    """
    X, Y = np.meshgrid(grid_x, grid_y)
    U_total = np.zeros_like(X)

    for x0, y0 in positions:
        U_total += dipole_potential(X - x0, Y - y0, z, power_per_site, waist, wavelength)

    return X, Y, U_total

# Create 5x5 square array with 4 μm spacing
spacing = 4e-6  # m
n_sites = 5
positions = []
for i in range(n_sites):
    for j in range(n_sites):
        x = (i - n_sites//2) * spacing
        y = (j - n_sites//2) * spacing
        positions.append((x, y))

# Grid for visualization
grid_extent = 15e-6
grid_points = 300
grid_x = np.linspace(-grid_extent, grid_extent, grid_points)
grid_y = np.linspace(-grid_extent, grid_extent, grid_points)

# Calculate total potential
power_per_site = 5e-3  # 5 mW per site
X, Y, U_array = create_tweezer_array(positions, power_per_site, waist, wavelength, grid_x, grid_y)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2D view
im = axes[0].pcolormesh(X*1e6, Y*1e6, U_array/k*1e3, shading='auto', cmap='viridis')
axes[0].set_xlabel('x position (μm)')
axes[0].set_ylabel('y position (μm)')
axes[0].set_title(f'{n_sites}×{n_sites} Tweezer Array Potential')
plt.colorbar(im, ax=axes[0], label='Potential (mK)')

# Mark tweezer centers
for x0, y0 in positions:
    axes[0].plot(x0*1e6, y0*1e6, 'r+', markersize=8)

# Line cut through center row
y_idx = grid_points // 2
axes[1].plot(grid_x*1e6, U_array[y_idx, :]/k*1e3)
axes[1].set_xlabel('x position (μm)')
axes[1].set_ylabel('Potential (mK)')
axes[1].set_title('Potential Cut Through Center Row')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tweezer_array_potential.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze crosstalk between neighboring sites
print("\n=== Crosstalk Analysis ===")
barrier_height = U_array[y_idx, grid_points//2 + int(spacing/2/(grid_x[1]-grid_x[0]))]
trap_minimum = np.min(U_array[y_idx, :])
print(f"Barrier between sites: {(barrier_height - trap_minimum)/k*1e3:.3f} mK")
print(f"Trap depth: {-trap_minimum/k*1e3:.3f} mK")
print(f"Barrier/Depth ratio: {(barrier_height - trap_minimum)/(-trap_minimum)*100:.1f}%")
```

### Lab 2: AOD vs SLM Comparison

```python
"""
Lab 2: Compare AOD and SLM array generation methods
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def aod_array_1d(n_sites, spacing, f_center, v_sound, focal_length, wavelength):
    """
    Calculate RF frequencies for 1D AOD array.

    Returns:
    --------
    freqs : array
        RF frequencies (Hz)
    positions : array
        Spot positions in focal plane (m)
    """
    # Position to frequency conversion
    # x = f * lambda * f_RF / v_s
    delta_f = v_sound * spacing / (focal_length * wavelength)

    freqs = f_center + np.arange(n_sites) * delta_f - (n_sites - 1) * delta_f / 2
    positions = (freqs - f_center) * focal_length * wavelength / v_sound

    return freqs, positions

def gerchberg_saxton(target_intensity, n_iterations=100):
    """
    Gerchberg-Saxton algorithm for phase retrieval.

    Parameters:
    -----------
    target_intensity : 2D array
        Desired intensity pattern in focal plane
    n_iterations : int
        Number of iterations

    Returns:
    --------
    phase : 2D array
        Phase pattern for SLM
    achieved_intensity : 2D array
        Achieved intensity after algorithm
    """
    # Initialize with random phase
    target_amplitude = np.sqrt(target_intensity)
    phase = np.random.uniform(0, 2*np.pi, target_intensity.shape)

    convergence = []

    for i in range(n_iterations):
        # Focal plane field
        focal_field = target_amplitude * np.exp(1j * phase)

        # Propagate to SLM plane (inverse FFT)
        slm_field = ifft2(ifftshift(focal_field))

        # Apply SLM constraint (phase-only, uniform amplitude)
        slm_phase = np.angle(slm_field)
        slm_field_constrained = np.exp(1j * slm_phase)

        # Propagate to focal plane (FFT)
        focal_field_new = fftshift(fft2(slm_field_constrained))

        # Apply target intensity constraint
        phase = np.angle(focal_field_new)

        # Calculate convergence metric
        achieved = np.abs(focal_field_new)**2
        achieved_norm = achieved / np.max(achieved) * np.max(target_intensity)
        error = np.sum((achieved_norm - target_intensity)**2) / np.sum(target_intensity**2)
        convergence.append(error)

    return slm_phase, np.abs(focal_field_new)**2, convergence

# AOD Analysis
print("=== AOD Array Design ===")
n_sites = 10
spacing = 4e-6  # m
f_center = 100e6  # Hz
v_sound = 4200  # m/s (TeO2)
focal_length = 50e-3  # m
wavelength = 1064e-9  # m

freqs, positions = aod_array_1d(n_sites, spacing, f_center, v_sound, focal_length, wavelength)

print(f"Number of sites: {n_sites}")
print(f"Spacing: {spacing*1e6:.1f} μm")
print(f"Total array extent: {(positions[-1]-positions[0])*1e6:.1f} μm")
print(f"RF frequency range: {freqs[0]/1e6:.3f} - {freqs[-1]/1e6:.3f} MHz")
print(f"RF bandwidth required: {(freqs[-1]-freqs[0])/1e6:.3f} MHz")

# Plot AOD array positions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].stem(freqs/1e6, np.ones(n_sites), basefmt=' ')
axes[0].set_xlabel('RF Frequency (MHz)')
axes[0].set_ylabel('Amplitude (a.u.)')
axes[0].set_title('AOD RF Tones')
axes[0].grid(True, alpha=0.3)

axes[0].axhspan(0.7, 1.0, alpha=0.2, color='green', label='Typical efficiency range')
axes[0].legend()

# Simulate intensity variation from acoustic attenuation
acoustic_attenuation = 0.02  # dB/MHz
freq_offset = freqs - f_center
attenuation = 10**(-acoustic_attenuation * np.abs(freq_offset)/1e6 / 20)

axes[1].bar(positions*1e6, attenuation, width=spacing*1e6*0.3)
axes[1].set_xlabel('Position (μm)')
axes[1].set_ylabel('Relative Intensity')
axes[1].set_title('Intensity Variation Across Array')
axes[1].set_ylim(0, 1.1)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aod_array_design.png', dpi=150, bbox_inches='tight')
plt.show()

# SLM Analysis with Gerchberg-Saxton
print("\n=== SLM Hologram Generation ===")

# Create target pattern: 5x5 array
n_pixels = 256
target = np.zeros((n_pixels, n_pixels))

# Place spots
spot_spacing = 20  # pixels
spot_size = 3  # pixels (FWHM)
n_array = 5

center = n_pixels // 2
for i in range(n_array):
    for j in range(n_array):
        x = center + (i - n_array//2) * spot_spacing
        y = center + (j - n_array//2) * spot_spacing

        # Gaussian spot
        xx, yy = np.meshgrid(np.arange(n_pixels), np.arange(n_pixels))
        target += np.exp(-((xx-x)**2 + (yy-y)**2) / (2*(spot_size/2.355)**2))

target /= np.max(target)

# Run Gerchberg-Saxton
print("Running Gerchberg-Saxton algorithm...")
slm_phase, achieved, convergence = gerchberg_saxton(target, n_iterations=200)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Target pattern
im1 = axes[0, 0].imshow(target, cmap='hot')
axes[0, 0].set_title('Target Intensity Pattern')
plt.colorbar(im1, ax=axes[0, 0])

# SLM phase pattern
im2 = axes[0, 1].imshow(slm_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
axes[0, 1].set_title('SLM Phase Pattern')
plt.colorbar(im2, ax=axes[0, 1], label='Phase (rad)')

# Achieved pattern
achieved_norm = achieved / np.max(achieved)
im3 = axes[1, 0].imshow(achieved_norm, cmap='hot')
axes[1, 0].set_title('Achieved Intensity Pattern')
plt.colorbar(im3, ax=axes[1, 0])

# Convergence
axes[1, 1].semilogy(convergence)
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Error')
axes[1, 1].set_title('Algorithm Convergence')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('slm_hologram_generation.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze achieved pattern quality
print(f"\nFinal error: {convergence[-1]:.4f}")

# Calculate efficiency (power in spots vs total)
spot_mask = target > 0.5
efficiency = np.sum(achieved[spot_mask]) / np.sum(achieved)
print(f"Diffraction efficiency: {efficiency*100:.1f}%")

# Calculate intensity uniformity
spot_intensities = []
for i in range(n_array):
    for j in range(n_array):
        x = center + (i - n_array//2) * spot_spacing
        y = center + (j - n_array//2) * spot_spacing
        spot_intensities.append(achieved[y, x])

spot_intensities = np.array(spot_intensities)
uniformity = np.std(spot_intensities) / np.mean(spot_intensities)
print(f"Intensity uniformity (std/mean): {uniformity*100:.1f}%")
```

### Lab 3: Trap Lifetime and Heating Analysis

```python
"""
Lab 3: Analyze trap lifetime limitations and heating mechanisms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar, k, epsilon_0, atomic_mass
from scipy.integrate import odeint

class AtomicSpecies:
    """Container for atomic species parameters."""
    pass

# Rb-87 parameters
Rb87 = AtomicSpecies()
Rb87.name = "Rb-87"
Rb87.mass = 87 * atomic_mass
Rb87.lambda_D2 = 780e-9
Rb87.Gamma_D2 = 2 * np.pi * 6.07e6
Rb87.omega_D2 = 2 * np.pi * c / Rb87.lambda_D2

# Cs-133 parameters
Cs133 = AtomicSpecies()
Cs133.name = "Cs-133"
Cs133.mass = 133 * atomic_mass
Cs133.lambda_D2 = 852e-9
Cs133.Gamma_D2 = 2 * np.pi * 5.22e6
Cs133.omega_D2 = 2 * np.pi * c / Cs133.lambda_D2

def scattering_rate(trap_depth_K, wavelength, atom):
    """
    Calculate photon scattering rate.

    Parameters:
    -----------
    trap_depth_K : float
        Trap depth in Kelvin
    wavelength : float
        Trap laser wavelength (m)
    atom : AtomicSpecies
        Atomic species parameters
    """
    omega = 2 * np.pi * c / wavelength
    Delta = omega - atom.omega_D2
    U0 = trap_depth_K * k  # Convert to Joules

    Gamma_sc = (atom.Gamma_D2 / Delta)**2 * np.abs(U0) / hbar
    return Gamma_sc

def heating_rate_rin(omega_trap, S_RIN, temperature):
    """
    Calculate heating rate from relative intensity noise.

    Parameters:
    -----------
    omega_trap : float
        Trap frequency (rad/s)
    S_RIN : float
        Relative intensity noise at 2*omega_trap (1/Hz)
    temperature : float
        Current temperature (K)

    Returns:
    --------
    dT_dt : float
        Heating rate (K/s)
    """
    # Parametric heating rate
    dE_dt = np.pi**2 * omega_trap**2 * S_RIN * k * temperature / 2
    dT_dt = dE_dt / k
    return dT_dt

def simulate_heating(omega_trap, S_RIN, T0, t_span):
    """
    Simulate temperature evolution due to parametric heating.
    """
    def dT_dt(T, t):
        return heating_rate_rin(omega_trap, S_RIN, T)

    t = np.linspace(0, t_span, 1000)
    T = odeint(dT_dt, T0, t)
    return t, T.flatten()

# Compare scattering rates for different wavelengths
print("=== Scattering Rate Analysis ===\n")

wavelengths = np.array([800, 850, 900, 1000, 1064, 1200]) * 1e-9
trap_depth = 1e-3  # 1 mK

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for atom in [Rb87, Cs133]:
    rates = []
    for wl in wavelengths:
        rate = scattering_rate(trap_depth, wl, atom)
        rates.append(rate)

    rates = np.array(rates)
    axes[0].semilogy(wavelengths*1e9, rates, 'o-', label=atom.name)

    print(f"{atom.name}:")
    for wl, rate in zip(wavelengths, rates):
        print(f"  {wl*1e9:.0f} nm: Γ_sc = {rate:.2f} s⁻¹, τ = {1/rate*1e3:.0f} ms")
    print()

axes[0].axhline(y=1, color='gray', linestyle='--', label='1 s⁻¹')
axes[0].set_xlabel('Wavelength (nm)')
axes[0].set_ylabel('Scattering Rate (s⁻¹)')
axes[0].set_title('Scattering Rate vs Trap Wavelength (1 mK depth)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Lifetime vs trap depth
trap_depths = np.logspace(-4, -2, 50)  # 0.1 mK to 10 mK
wavelength = 1064e-9

for atom in [Rb87, Cs133]:
    rates = [scattering_rate(U, wavelength, atom) for U in trap_depths]
    lifetimes = 1 / np.array(rates)
    axes[1].loglog(trap_depths*1e3, lifetimes*1e3, label=atom.name)

axes[1].set_xlabel('Trap Depth (mK)')
axes[1].set_ylabel('Scattering-limited Lifetime (ms)')
axes[1].set_title('Lifetime vs Trap Depth (1064 nm)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scattering_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Parametric heating analysis
print("=== Parametric Heating Analysis ===\n")

omega_r = 2 * np.pi * 100e3  # 100 kHz trap frequency
T0 = 10e-6  # Initial temperature: 10 μK (ground state occupancy ~0.1)

# Different RIN levels (in dBc/Hz)
RIN_dBc = np.array([-120, -130, -140, -150, -160])
RIN_linear = 10**(RIN_dBc/10)  # Convert to linear (1/Hz)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = plt.cm.viridis(np.linspace(0, 1, len(RIN_dBc)))

for rin_db, rin_lin, color in zip(RIN_dBc, RIN_linear, colors):
    # Simulate for 1 second
    t, T = simulate_heating(omega_r, rin_lin, T0, 1.0)
    axes[0].semilogy(t*1e3, T*1e6, color=color, label=f'{rin_db} dBc/Hz')

axes[0].axhline(y=hbar*omega_r/(2*k)*1e6, color='red', linestyle='--',
                label='Ground state energy')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Temperature (μK)')
axes[0].set_title('Temperature Evolution from RIN Heating')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(1, 1000)

# Time to reach 100 μK (roughly motional excitation)
T_threshold = 100e-6  # 100 μK
times_to_threshold = []

for rin_lin in RIN_linear:
    t, T = simulate_heating(omega_r, rin_lin, T0, 100.0)
    idx = np.where(T > T_threshold)[0]
    if len(idx) > 0:
        times_to_threshold.append(t[idx[0]])
    else:
        times_to_threshold.append(100.0)

axes[1].semilogy(-RIN_dBc, times_to_threshold, 'bo-')
axes[1].set_xlabel('RIN Suppression (-dBc/Hz)')
axes[1].set_ylabel('Time to 100 μK (s)')
axes[1].set_title('Heating Time vs Laser Noise')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heating_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("RIN Level | Heating Rate | Time to 100 μK")
print("-" * 45)
for rin_db, rin_lin, t_thresh in zip(RIN_dBc, RIN_linear, times_to_threshold):
    heat_rate = heating_rate_rin(omega_r, rin_lin, T0) * 1e6  # μK/s
    print(f"{rin_db:4d} dBc/Hz | {heat_rate:10.2f} μK/s | {t_thresh:8.2f} s")

# Total lifetime budget
print("\n=== Lifetime Budget Summary ===")
print("\nFor Rb-87 at 1064 nm, 1 mK trap, 100 kHz trap frequency:")

# Scattering
Gamma_sc = scattering_rate(1e-3, 1064e-9, Rb87)
tau_sc = 1/Gamma_sc

# Background gas (assume 10^-11 Torr)
tau_bg = 100  # s (typical for good UHV)

# Heating (assume -140 dBc/Hz RIN)
RIN = 10**(-140/10)
t_heat = 10  # s (time to significant heating)

# Total
tau_total = 1 / (1/tau_sc + 1/tau_bg + 1/t_heat)

print(f"Scattering-limited: {tau_sc*1e3:.0f} ms")
print(f"Background gas limited: {tau_bg:.0f} s")
print(f"Heating limited: ~{t_heat:.0f} s")
print(f"Total effective lifetime: ~{tau_total*1e3:.0f} ms")
print(f"\nThis allows ~{tau_total*omega_r/(2*np.pi)/1e3:.0f} trap oscillations")
print(f"Or ~{tau_total*1e6:.0f} single-qubit gates (assuming 1 μs gate)")
```

## Summary

### Key Formulas Table

| Quantity | Formula | Typical Value |
|----------|---------|---------------|
| Dipole potential | $U = -\frac{3\pi c^2\Gamma}{2\omega_0^3\Delta}I$ | ~1 mK depth |
| Radial trap frequency | $\omega_r = \sqrt{4U_0/mw_0^2}$ | ~100 kHz |
| Axial trap frequency | $\omega_z = \sqrt{2U_0/mz_R^2}$ | ~20 kHz |
| Scattering rate | $\Gamma_{sc} = (\Gamma/\Delta)^2 U_0/\hbar$ | ~10 s⁻¹ |
| AOD deflection | $\theta = \lambda f_{RF}/v_s$ | ~mrad |
| Rayleigh range | $z_R = \pi w_0^2/\lambda$ | ~3 μm |

### Main Takeaways

1. **Dipole trapping** exploits the AC Stark shift to create conservative potentials for neutral atoms; red detuning creates attractive traps at intensity maxima.

2. **Far detuning** is essential for long coherence times since scattering rate scales as $1/\Delta^2$ while trap depth scales as $1/\Delta$.

3. **AODs** provide fast, continuous reconfiguration ideal for atom sorting, while **SLMs** enable arbitrary geometries for large arrays.

4. **Single-atom loading** occurs naturally through collisional blockade, giving ~50% loading probability per site.

5. **Trap lifetime** is limited by photon scattering, background gas collisions, and parametric heating from intensity noise.

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain why red-detuned light creates attractive traps
- [ ] I understand the tradeoff between trap depth and scattering rate
- [ ] I can compare AOD and SLM approaches for array generation
- [ ] I understand how collisional blockade enables single-atom loading

### Mathematical Skills
- [ ] I can derive the dipole potential from atomic polarizability
- [ ] I can calculate trap frequencies from beam parameters
- [ ] I can design an AOD array for specified geometry

### Computational Skills
- [ ] I can simulate Gaussian beam intensity distributions
- [ ] I can implement the Gerchberg-Saxton algorithm
- [ ] I can analyze trap lifetime from various noise sources

## Preview: Day 912

Tomorrow we explore **Rydberg States and Interactions**, where we will:
- Calculate Rydberg state properties using quantum defect theory
- Derive van der Waals interaction strengths (C₆ coefficients)
- Understand the scaling of Rydberg properties with principal quantum number
- Analyze Rydberg excitation dynamics and coherence

The strong, long-range interactions between Rydberg atoms are what make neutral atom arrays powerful for quantum computing, enabling fast entangling gates across multiple qubits.
