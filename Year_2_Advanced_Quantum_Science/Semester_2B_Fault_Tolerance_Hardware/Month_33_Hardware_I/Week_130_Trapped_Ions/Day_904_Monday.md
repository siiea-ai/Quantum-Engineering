# Day 904: Ion Trapping Physics - Paul Traps and Secular Motion

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Paul trap theory, pseudopotential, Mathieu equation |
| Afternoon | 2 hours | Problem solving with trap parameters |
| Evening | 2 hours | Computational lab: trap potential simulation |

## Learning Objectives

By the end of today, you will be able to:

1. **Derive the pseudopotential** for a linear Paul trap from time-averaging
2. **Solve the Mathieu equation** and determine stability regions
3. **Calculate secular frequencies** from trap geometry and drive parameters
4. **Explain micromotion** and its effects on qubit operations
5. **Analyze ion crystal configurations** in multi-ion systems
6. **Design trap parameters** for specific experimental requirements

## Core Content

### 1. Introduction to Ion Trapping

Trapped ions represent one of the most successful platforms for quantum computing, achieving the highest gate fidelities (~99.9%) among all quantum hardware. The fundamental challenge is confining charged particles in free space—impossible with static electric fields alone due to Earnshaw's theorem.

**Earnshaw's Theorem:** No static configuration of electric charges can create a stable equilibrium for a test charge in free space.

$$\nabla^2 \Phi = 0 \implies \text{No local minimum for static potential}$$

The solution: use **oscillating fields** that create an effective confining potential through time-averaging.

### 2. The Linear Paul Trap

The linear (or RF) Paul trap confines ions using a combination of radiofrequency (RF) and static (DC) electric fields.

#### Electrode Configuration

A typical linear Paul trap consists of:
- **Four RF electrodes** in a quadrupole configuration (confinement in x-y plane)
- **Two DC endcap electrodes** (confinement along z-axis)

The electric potential near the trap center is:

$$\Phi(x, y, z, t) = \frac{V_{RF}\cos(\Omega_{RF}t)}{2r_0^2}(x^2 - y^2) + \frac{V_{DC}}{z_0^2}\left(z^2 - \frac{x^2 + y^2}{2}\right)$$

where:
- $V_{RF}$ = RF voltage amplitude (typically 100-1000 V)
- $\Omega_{RF}$ = RF drive frequency (typically 10-100 MHz)
- $r_0$ = characteristic radial distance
- $V_{DC}$ = DC endcap voltage
- $z_0$ = characteristic axial distance

### 3. Equations of Motion

The classical equations of motion for an ion of mass $m$ and charge $q$ are:

$$m\ddot{x} = -q\frac{\partial \Phi}{\partial x} = -\frac{qV_{RF}\cos(\Omega_{RF}t)}{r_0^2}x + \frac{qV_{DC}}{2z_0^2}x$$

This can be recast as the **Mathieu equation**:

$$\frac{d^2 x}{d\tau^2} + (a_x - 2q_x\cos(2\tau))x = 0$$

where $\tau = \Omega_{RF}t/2$ and the stability parameters are:

$$\boxed{a_x = \frac{4qV_{DC}}{m\Omega_{RF}^2 z_0^2}, \quad q_x = \frac{2qV_{RF}}{m\Omega_{RF}^2 r_0^2}}$$

### 4. Stability Regions

The Mathieu equation has stable solutions only for certain ranges of $(a, q)$ parameters. The first stability region, commonly used in ion traps, requires:

$$|q| < 0.908, \quad a < \frac{q^2}{2}$$

For typical trapped ion experiments: $|q| \sim 0.1-0.3$ and $|a| \ll 1$.

### 5. Pseudopotential Approximation

When $|q| \ll 1$, the ion motion separates into:
1. **Secular motion:** Slow oscillation at frequency $\omega_{sec}$
2. **Micromotion:** Fast oscillation at $\Omega_{RF}$

The **pseudopotential** (or ponderomotive potential) describes the time-averaged confining force:

$$\boxed{\Psi_{pseudo}(x,y) = \frac{q^2 V_{RF}^2}{4m\Omega_{RF}^2 r_0^4}(x^2 + y^2)}$$

This is a harmonic potential with secular frequencies:

$$\boxed{\omega_x = \omega_y = \omega_r = \frac{qV_{RF}}{\sqrt{2}m\Omega_{RF}r_0^2} = \frac{q_x \Omega_{RF}}{2\sqrt{2}}}$$

For the axial direction (DC confinement):

$$\boxed{\omega_z = \sqrt{\frac{2qV_{DC}}{mz_0^2}}}$$

### 6. Complete Solution and Micromotion

The complete ion trajectory includes micromotion:

$$x(t) = x_{sec}(t)\left[1 + \frac{q_x}{2}\cos(\Omega_{RF}t)\right]$$

where $x_{sec}(t) = x_0\cos(\omega_r t + \phi)$ is the secular motion.

**Micromotion amplitude:**
$$x_{micro} = \frac{q_x}{2}x_{sec}$$

**Critical insight:** Micromotion is **driven motion** that cannot be laser-cooled. It occurs when ions are displaced from the RF null (trap center).

### 7. Multi-Ion Crystals

When multiple ions are loaded, they form ordered structures due to Coulomb repulsion balanced by the confining potential.

For $N$ ions along the z-axis, the equilibrium positions $z_i$ minimize:

$$U = \sum_i \frac{1}{2}m\omega_z^2 z_i^2 + \sum_{i<j}\frac{q^2}{4\pi\epsilon_0|z_i - z_j|}$$

**Two-ion separation:**
$$d = \left(\frac{q^2}{4\pi\epsilon_0 m\omega_z^2}\right)^{1/3}$$

**Normal modes:** The collective motion splits into:
- **Center-of-mass (COM) mode:** $\omega_{COM} = \omega_z$
- **Stretch (breathing) mode:** $\omega_{stretch} = \sqrt{3}\omega_z$

### 8. Quantum Harmonic Oscillator Description

Each motional mode is quantized as a harmonic oscillator:

$$\hat{H}_{motion} = \hbar\omega_z\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right)$$

The ground state wavefunction has width:

$$x_0 = \sqrt{\frac{\hbar}{2m\omega}} \sim 5-15 \text{ nm for typical traps}$$

This defines the **Lamb-Dicke parameter**:

$$\boxed{\eta = k \cdot x_0 = k\sqrt{\frac{\hbar}{2m\omega}}}$$

where $k$ is the laser wavevector. For $\eta \ll 1$, the system is in the **Lamb-Dicke regime**, essential for high-fidelity gates.

## Quantum Computing Applications

### Trap Design for Quantum Computing

| Parameter | Typical Value | Design Consideration |
|-----------|---------------|---------------------|
| $\omega_r/2\pi$ | 1-5 MHz | Higher = stronger confinement |
| $\omega_z/2\pi$ | 0.5-2 MHz | Sets ion spacing, mode frequencies |
| $\eta$ | 0.05-0.2 | Must be in Lamb-Dicke regime |
| $r_0$ | 100-500 μm | Smaller = stronger fields, more heating |
| Ion spacing | 3-10 μm | Must resolve individual ions |

### Why Trapped Ions Excel

1. **Identical qubits:** All ions of same species are truly identical
2. **Long coherence:** Atomic transitions isolated from environment
3. **Individual addressing:** Focused lasers select specific ions
4. **All-to-all connectivity:** Collective modes couple all ions

## Worked Examples

### Example 1: Calculating Trap Parameters

**Problem:** Design a linear Paul trap for $^{171}$Yb$^+$ ions with $\omega_r/2\pi = 3$ MHz and $\omega_z/2\pi = 1$ MHz.

**Given:**
- Mass: $m = 171 \times 1.66 \times 10^{-27}$ kg
- Charge: $q = 1.6 \times 10^{-19}$ C
- $\Omega_{RF}/2\pi = 30$ MHz
- $r_0 = 300$ μm

**Solution:**

From $\omega_r = \frac{q_x \Omega_{RF}}{2\sqrt{2}}$:

$$q_x = \frac{2\sqrt{2}\omega_r}{\Omega_{RF}} = \frac{2\sqrt{2} \times 2\pi \times 3\text{ MHz}}{2\pi \times 30\text{ MHz}} = 0.283$$

From $q_x = \frac{2qV_{RF}}{m\Omega_{RF}^2 r_0^2}$:

$$V_{RF} = \frac{q_x m \Omega_{RF}^2 r_0^2}{2q}$$

$$V_{RF} = \frac{0.283 \times 2.84 \times 10^{-25} \times (2\pi \times 3 \times 10^7)^2 \times (3 \times 10^{-4})^2}{2 \times 1.6 \times 10^{-19}}$$

$$\boxed{V_{RF} \approx 640 \text{ V}}$$

For axial confinement, from $\omega_z = \sqrt{\frac{2qV_{DC}}{mz_0^2}}$:

With $z_0 = 2$ mm:

$$V_{DC} = \frac{m\omega_z^2 z_0^2}{2q} = \frac{2.84 \times 10^{-25} \times (2\pi \times 10^6)^2 \times (2 \times 10^{-3})^2}{2 \times 1.6 \times 10^{-19}}$$

$$\boxed{V_{DC} \approx 35 \text{ V}}$$

### Example 2: Lamb-Dicke Parameter

**Problem:** Calculate the Lamb-Dicke parameter for $^{40}$Ca$^+$ at 729 nm with $\omega_z/2\pi = 1.2$ MHz.

**Solution:**

$$x_0 = \sqrt{\frac{\hbar}{2m\omega_z}} = \sqrt{\frac{1.055 \times 10^{-34}}{2 \times 40 \times 1.66 \times 10^{-27} \times 2\pi \times 1.2 \times 10^6}}$$

$$x_0 = \sqrt{\frac{1.055 \times 10^{-34}}{5.01 \times 10^{-19}}} = 14.5 \text{ nm}$$

$$k = \frac{2\pi}{\lambda} = \frac{2\pi}{729 \times 10^{-9}} = 8.62 \times 10^6 \text{ m}^{-1}$$

$$\boxed{\eta = k \cdot x_0 = 8.62 \times 10^6 \times 14.5 \times 10^{-9} = 0.125}$$

This is well within the Lamb-Dicke regime ($\eta \ll 1$).

### Example 3: Ion Crystal Spacing

**Problem:** Find the equilibrium spacing of two $^{171}$Yb$^+$ ions with $\omega_z/2\pi = 800$ kHz.

**Solution:**

$$d = \left(\frac{q^2}{4\pi\epsilon_0 m\omega_z^2}\right)^{1/3}$$

$$d = \left(\frac{(1.6 \times 10^{-19})^2}{4\pi \times 8.85 \times 10^{-12} \times 2.84 \times 10^{-25} \times (2\pi \times 8 \times 10^5)^2}\right)^{1/3}$$

$$d = \left(\frac{2.56 \times 10^{-38}}{2.52 \times 10^{-30}}\right)^{1/3} = (1.02 \times 10^{-8})^{1/3}$$

$$\boxed{d \approx 4.6 \text{ μm}}$$

## Practice Problems

### Level 1: Direct Application

1. Calculate the stability parameter $q_x$ for $V_{RF} = 500$ V, $\Omega_{RF}/2\pi = 25$ MHz, $r_0 = 250$ μm, and $^{88}$Sr$^+$.

2. Find the secular frequency ratio $\omega_r/\omega_z$ for a trap with $q_x = 0.2$, $a_x = -0.002$.

3. Determine the micromotion amplitude for an ion displaced 1 μm from the RF null with $q_x = 0.25$.

### Level 2: Intermediate

4. A three-ion crystal has modes at $\omega_1$, $\omega_2$, $\omega_3$. Given $\omega_z/2\pi = 1$ MHz, calculate all mode frequencies and sketch the mode patterns.

5. Design a trap where the Lamb-Dicke parameter $\eta = 0.1$ for a 369 nm transition in $^{171}$Yb$^+$. What axial frequency is required?

6. Calculate the heating rate contribution to gate infidelity if the motional heating rate is $\dot{\bar{n}} = 100$ quanta/s and the gate time is 100 μs.

### Level 3: Challenging

7. Derive the pseudopotential from first principles by solving the Mathieu equation to first order in $q$ and time-averaging.

8. For a surface-electrode trap, the RF pseudopotential is modified. If the ion height is $h = 50$ μm and the RF electrodes are separated by $s = 100$ μm, estimate the relationship between secular frequency and ion height.

9. Analyze the effect of stray DC fields on micromotion. If a stray field $E_s = 10$ V/m exists, calculate the excess micromotion amplitude and the resulting Doppler shift for a 397 nm cooling laser.

## Computational Lab: Paul Trap Simulation

```python
"""
Day 904 Computational Lab: Paul Trap Physics Simulation
Simulating ion trajectories, pseudopotential, and multi-ion crystals
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
e = 1.602e-19  # Elementary charge (C)
amu = 1.661e-27  # Atomic mass unit (kg)
epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
hbar = 1.055e-34  # Reduced Planck constant (J·s)

class PaulTrap:
    """Linear Paul trap simulator"""

    def __init__(self, ion_mass_amu, V_RF, Omega_RF, r0, V_DC, z0):
        """
        Initialize trap parameters

        Parameters:
        -----------
        ion_mass_amu : float - Ion mass in atomic mass units
        V_RF : float - RF voltage amplitude (V)
        Omega_RF : float - RF drive frequency (rad/s)
        r0 : float - Radial electrode distance (m)
        V_DC : float - DC endcap voltage (V)
        z0 : float - Axial electrode distance (m)
        """
        self.m = ion_mass_amu * amu
        self.q = e
        self.V_RF = V_RF
        self.Omega_RF = Omega_RF
        self.r0 = r0
        self.V_DC = V_DC
        self.z0 = z0

        # Calculate stability parameters
        self.q_x = 2 * e * V_RF / (self.m * Omega_RF**2 * r0**2)
        self.a_x = -4 * e * V_DC / (self.m * Omega_RF**2 * z0**2)

        # Calculate secular frequencies
        self.omega_r = self.q_x * Omega_RF / (2 * np.sqrt(2))
        self.omega_z = np.sqrt(2 * e * V_DC / (self.m * z0**2))

        print(f"Trap Parameters:")
        print(f"  Stability: q_x = {self.q_x:.4f}, a_x = {self.a_x:.6f}")
        print(f"  Secular frequencies: ω_r/2π = {self.omega_r/(2*np.pi)/1e6:.3f} MHz")
        print(f"                       ω_z/2π = {self.omega_z/(2*np.pi)/1e6:.3f} MHz")

    def check_stability(self):
        """Check if trap parameters are in stable region"""
        stable = (abs(self.q_x) < 0.908) and (self.a_x < self.q_x**2 / 2)
        print(f"Stability check: {'STABLE' if stable else 'UNSTABLE'}")
        return stable

    def pseudopotential(self, x, y):
        """Calculate pseudopotential at position (x, y)"""
        return (self.q**2 * self.V_RF**2) / (4 * self.m * self.Omega_RF**2 * self.r0**4) * (x**2 + y**2)

    def equations_of_motion(self, state, t):
        """Full equations of motion including RF drive"""
        x, vx, y, vy, z, vz = state

        # RF potential contribution (time-dependent)
        RF_term = self.V_RF * np.cos(self.Omega_RF * t) / self.r0**2

        # DC potential contribution
        DC_term = self.V_DC / self.z0**2

        # Accelerations
        ax = -(self.q / self.m) * RF_term * x + (self.q / self.m) * DC_term * x / 2
        ay = (self.q / self.m) * RF_term * y + (self.q / self.m) * DC_term * y / 2
        az = -(self.q / self.m) * DC_term * z

        return [vx, ax, vy, ay, vz, az]

    def simulate_trajectory(self, x0, y0, z0, vx0, vy0, vz0, t_max, dt):
        """Simulate ion trajectory"""
        t = np.arange(0, t_max, dt)
        initial_state = [x0, vx0, y0, vy0, z0, vz0]

        solution = odeint(self.equations_of_motion, initial_state, t)

        return t, solution

    def lamb_dicke_parameter(self, wavelength, omega):
        """Calculate Lamb-Dicke parameter"""
        k = 2 * np.pi / wavelength
        x0 = np.sqrt(hbar / (2 * self.m * omega))
        return k * x0, x0

def simulate_two_ion_crystal(trap, omega_z):
    """Calculate equilibrium positions and normal modes for two ions"""
    m = trap.m

    # Equilibrium separation
    d = (e**2 / (4 * np.pi * epsilon_0 * m * omega_z**2))**(1/3)

    # Mode frequencies
    omega_COM = omega_z
    omega_stretch = np.sqrt(3) * omega_z

    return d, omega_COM, omega_stretch

def plot_stability_diagram():
    """Plot Mathieu equation stability diagram"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mesh for stability parameters
    a = np.linspace(-1, 1, 500)
    q = np.linspace(0, 1, 500)
    A, Q = np.meshgrid(a, q)

    # Approximate stability boundary (first region)
    # β² ≈ a + q²/2 for small a, q
    # Stable when 0 < β < 1
    beta_squared = A + Q**2 / 2

    # First stability region approximation
    stable = (beta_squared > 0) & (beta_squared < 1) & (Q < 0.908)

    ax.contourf(a, q, stable.astype(int), levels=[0.5, 1.5], colors=['lightblue'], alpha=0.7)
    ax.contour(a, q, stable.astype(int), levels=[0.5], colors=['blue'], linewidths=2)

    # Mark typical operating point
    ax.scatter([0], [0.2], color='red', s=100, zorder=5, label='Typical operating point')

    ax.set_xlabel('Stability parameter a', fontsize=12)
    ax.set_ylabel('Stability parameter q', fontsize=12)
    ax.set_title('Paul Trap Stability Diagram (First Region)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 0.95)

    plt.tight_layout()
    plt.savefig('stability_diagram.png', dpi=150)
    plt.show()

def main():
    """Main simulation routine"""
    print("=" * 60)
    print("Day 904: Paul Trap Physics Simulation")
    print("=" * 60)

    # Create trap for Yb-171
    trap = PaulTrap(
        ion_mass_amu=171,
        V_RF=600,  # V
        Omega_RF=2 * np.pi * 30e6,  # 30 MHz
        r0=300e-6,  # 300 μm
        V_DC=35,  # V
        z0=2e-3  # 2 mm
    )

    trap.check_stability()

    # Calculate Lamb-Dicke parameter for 369 nm transition
    eta, x0 = trap.lamb_dicke_parameter(369e-9, trap.omega_z)
    print(f"\nLamb-Dicke parameter (369 nm): η = {eta:.4f}")
    print(f"Ground state size: x₀ = {x0*1e9:.2f} nm")

    # Simulate ion trajectory
    print("\nSimulating ion trajectory...")
    t, traj = trap.simulate_trajectory(
        x0=1e-6, y0=0, z0=0,  # Initial position (1 μm off-center)
        vx0=0, vy0=0, vz0=0,  # Starting at rest
        t_max=50e-6,  # 50 μs
        dt=1e-9  # 1 ns resolution
    )

    # Plot trajectory
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # X motion showing secular + micromotion
    ax1 = axes[0, 0]
    ax1.plot(t * 1e6, traj[:, 0] * 1e6, 'b-', linewidth=0.5)
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('x position (μm)')
    ax1.set_title('Radial Motion (x): Secular + Micromotion')
    ax1.grid(True, alpha=0.3)

    # Zoom to show micromotion
    ax2 = axes[0, 1]
    mask = t < 2e-6
    ax2.plot(t[mask] * 1e9, traj[mask, 0] * 1e6, 'b-', linewidth=1)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('x position (μm)')
    ax2.set_title('Micromotion Detail (first 2 μs)')
    ax2.grid(True, alpha=0.3)

    # Phase space
    ax3 = axes[1, 0]
    ax3.plot(traj[:, 0] * 1e6, traj[:, 1], 'b-', linewidth=0.5, alpha=0.7)
    ax3.set_xlabel('x position (μm)')
    ax3.set_ylabel('x velocity (m/s)')
    ax3.set_title('Phase Space (x, vx)')
    ax3.grid(True, alpha=0.3)

    # Pseudopotential
    ax4 = axes[1, 1]
    x_range = np.linspace(-5e-6, 5e-6, 200)
    y_range = np.linspace(-5e-6, 5e-6, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Psi = trap.pseudopotential(X, Y)

    # Convert to meV for better visualization
    Psi_meV = Psi / e * 1000

    contour = ax4.contourf(X * 1e6, Y * 1e6, Psi_meV, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax4, label='Pseudopotential (meV)')
    ax4.set_xlabel('x (μm)')
    ax4.set_ylabel('y (μm)')
    ax4.set_title('Radial Pseudopotential')
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('ion_trajectory.png', dpi=150)
    plt.show()

    # Two-ion crystal analysis
    print("\n" + "=" * 60)
    print("Two-Ion Crystal Analysis")
    print("=" * 60)

    d, omega_COM, omega_stretch = simulate_two_ion_crystal(trap, trap.omega_z)
    print(f"Ion separation: d = {d*1e6:.2f} μm")
    print(f"COM mode frequency: ω_COM/2π = {omega_COM/(2*np.pi)/1e6:.3f} MHz")
    print(f"Stretch mode frequency: ω_stretch/2π = {omega_stretch/(2*np.pi)/1e6:.3f} MHz")

    # Plot stability diagram
    print("\nGenerating stability diagram...")
    plot_stability_diagram()

    # Normal mode visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # COM mode
    ax1 = axes[0]
    positions = [-d/2, d/2]
    for i, pos in enumerate(positions):
        ax1.scatter([pos * 1e6], [0], s=200, c='blue')
        ax1.arrow(pos * 1e6, 0, 0.3, 0, head_width=0.1, head_length=0.05, fc='red', ec='red')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel('Position (μm)')
    ax1.set_title(f'COM Mode: ω = ω_z = {trap.omega_z/(2*np.pi)/1e6:.2f} MHz')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Stretch mode
    ax2 = axes[1]
    for i, pos in enumerate(positions):
        ax2.scatter([pos * 1e6], [0], s=200, c='blue')
        direction = 1 if i == 1 else -1
        ax2.arrow(pos * 1e6, 0, 0.3 * direction, 0, head_width=0.1, head_length=0.05, fc='red', ec='red')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel('Position (μm)')
    ax2.set_title(f'Stretch Mode: ω = √3 ω_z = {omega_stretch/(2*np.pi)/1e6:.2f} MHz')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('normal_modes.png', dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Mathieu stability | $q_x = \frac{2qV_{RF}}{m\Omega_{RF}^2 r_0^2}$, $a_x = \frac{4qV_{DC}}{m\Omega_{RF}^2 z_0^2}$ |
| Radial secular frequency | $\omega_r = \frac{q_x \Omega_{RF}}{2\sqrt{2}}$ |
| Axial secular frequency | $\omega_z = \sqrt{\frac{2qV_{DC}}{mz_0^2}}$ |
| Pseudopotential | $\Psi = \frac{q^2V_{RF}^2}{4m\Omega_{RF}^2 r_0^4}(x^2 + y^2)$ |
| Lamb-Dicke parameter | $\eta = k\sqrt{\frac{\hbar}{2m\omega}}$ |
| Two-ion separation | $d = \left(\frac{q^2}{4\pi\epsilon_0 m\omega_z^2}\right)^{1/3}$ |
| Stretch mode | $\omega_{stretch} = \sqrt{3}\omega_z$ |

### Main Takeaways

1. **Paul traps** use oscillating RF fields to create effective confinement, bypassing Earnshaw's theorem
2. **Stability** requires $|q| < 0.908$ and specific $(a, q)$ combinations
3. **Secular motion** is the slow harmonic oscillation; **micromotion** is driven motion at RF frequency
4. The **Lamb-Dicke regime** ($\eta \ll 1$) is essential for high-fidelity quantum operations
5. **Multi-ion crystals** exhibit collective motional modes used for entangling gates
6. Trap design involves trade-offs between confinement strength, ion spacing, and heating rates

## Daily Checklist

- [ ] I can derive the pseudopotential from the time-dependent potential
- [ ] I understand the Mathieu equation and stability regions
- [ ] I can calculate secular frequencies from trap parameters
- [ ] I understand the origin and consequences of micromotion
- [ ] I can analyze normal modes of multi-ion crystals
- [ ] I can calculate the Lamb-Dicke parameter for a given transition
- [ ] I have run the computational lab and understand ion trajectories

## Preview of Day 905

Tomorrow we explore **Qubit Encoding Schemes** in trapped ions, including:
- Hyperfine qubits (nuclear spin states)
- Zeeman qubits (magnetic sublevels)
- Optical qubits (metastable electronic states)
- Comparison of coherence times, gate speeds, and experimental complexity

We will analyze the trade-offs between different encodings and understand why certain ion species are preferred for quantum computing.

---

*Day 904 of the QSE PhD Curriculum - Year 2, Month 33: Hardware Implementations I*
