# Day 914: Single-Qubit Gates

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| **Morning** | 3 hours | Microwave and Raman transitions, addressing techniques |
| **Afternoon** | 2 hours | Problem solving: gate pulse design |
| **Evening** | 2 hours | Computational lab: gate simulations |

## Learning Objectives

By the end of this day, you will be able to:

1. **Design microwave pulses** for hyperfine qubit transitions
2. **Analyze two-photon Raman transitions** for fast single-qubit gates
3. **Compare global vs local addressing** strategies and their trade-offs
4. **Calculate light shifts** and compensation techniques
5. **Optimize gate fidelity** against various error sources
6. **Implement numerical simulations** of single-qubit gate dynamics

## Core Content

### 1. Hyperfine Qubit Encoding

#### Ground State Qubit

For alkali atoms, the most common qubit encoding uses hyperfine ground states:

**Rubidium-87:**
$$|0\rangle = |5S_{1/2}, F=1, m_F=0\rangle$$
$$|1\rangle = |5S_{1/2}, F=2, m_F=0\rangle$$

Hyperfine splitting: $\Delta_{HFS} = 6.835$ GHz

**Cesium-133:**
$$|0\rangle = |6S_{1/2}, F=3, m_F=0\rangle$$
$$|1\rangle = |6S_{1/2}, F=4, m_F=0\rangle$$

Hyperfine splitting: $\Delta_{HFS} = 9.193$ GHz (defines the SI second)

#### Advantages of Hyperfine Encoding

1. **Long coherence times**: T₂ > 1 second possible with clock states (m_F = 0)
2. **Insensitivity to magnetic fields**: First-order Zeeman shift vanishes for clock states
3. **High-fidelity readout**: State-selective fluorescence detection
4. **Independent of Rydberg lifetime**: Rydberg used only during gates

#### Clock States and Magnetic Field Sensitivity

The energy levels in a magnetic field:
$$E_{F,m_F} = E_0 + g_F \mu_B m_F B + \beta B^2$$

For clock states ($m_F = 0$): linear term vanishes, leaving only quadratic shift:
$$\Delta E_{clock} = \beta B^2 \approx 575\,\text{Hz/G}^2 \times B^2$$ (for Rb-87)

At typical bias fields of 1-10 G, the shift is <10 kHz, easily compensated.

### 2. Microwave Single-Qubit Gates

#### Direct Microwave Coupling

The hyperfine transition can be driven directly with microwaves at ~6.8 GHz (Rb) or ~9.2 GHz (Cs).

**Hamiltonian:**
$$\hat{H} = \frac{\hbar\omega_0}{2}\hat{\sigma}_z + \frac{\hbar\Omega_{MW}}{2}(\hat{\sigma}_+ e^{-i\omega t} + \hat{\sigma}_- e^{i\omega t})$$

In the rotating frame with resonance ($\omega = \omega_0$):
$$\hat{H}_{rot} = \frac{\hbar\Omega_{MW}}{2}\hat{\sigma}_x$$

**Rabi frequency:**
$$\Omega_{MW} = \frac{\mu_B B_{MW}}{\hbar}\langle F', m_F'|\hat{\mu}|F, m_F\rangle$$

For typical horn antennas with B_MW ~ 10 mG:
$$\Omega_{MW} \approx 2\pi \times 10-100\,\text{kHz}$$

#### Gate Implementation

**X gate (π rotation about x):**
Apply microwave pulse for time $t_\pi = \pi/\Omega_{MW}$

For $\Omega_{MW} = 2\pi \times 50$ kHz: $t_\pi = 10$ μs

**Hadamard gate:**
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Implemented as $R_y(\pi/2)$ or combination of rotations.

**Arbitrary rotations:**
Phase-controlled microwave pulses:
$$R_\phi(\theta) = \exp\left(-i\frac{\theta}{2}(\cos\phi\,\hat{\sigma}_x + \sin\phi\,\hat{\sigma}_y)\right)$$

#### Microwave Limitations

1. **Slow gates**: Limited by B-field amplitude
2. **Global operation**: Microwave wavelength (~5 cm) >> array size, all atoms driven
3. **Crosstalk**: Difficult to address individual atoms
4. **Heating**: Near-field effects can heat the trap

### 3. Two-Photon Raman Transitions

#### Raman Coupling Mechanism

Two laser beams with frequencies $\omega_1$ and $\omega_2$ couple the qubit states via an intermediate excited state:

$$|0\rangle \xrightarrow{\omega_1} |e\rangle \xrightarrow{\omega_2} |1\rangle$$

When $\omega_1 - \omega_2 = \omega_{HFS}$ and both beams are detuned from $|e\rangle$ by $\Delta$:

**Effective Rabi frequency:**
$$\boxed{\Omega_{eff} = \frac{\Omega_1 \Omega_2}{2\Delta}}$$

where $\Omega_{1,2}$ are the single-photon Rabi frequencies.

**Effective Hamiltonian:**
$$\hat{H}_{eff} = \frac{\hbar\Omega_{eff}}{2}(|0\rangle\langle 1|e^{i\phi} + |1\rangle\langle 0|e^{-i\phi}) + \frac{\hbar\delta}{2}\hat{\sigma}_z$$

where $\phi = \phi_1 - \phi_2$ is the relative laser phase and $\delta = (\omega_1 - \omega_2) - \omega_{HFS}$ is the two-photon detuning.

#### Advantages of Raman Gates

1. **Fast gates**: MHz-scale Rabi frequencies achievable
2. **Local addressing**: Focused beams enable single-atom control
3. **Phase control**: Laser phase sets rotation axis
4. **Momentum transfer**: Can be made Doppler-insensitive

#### Raman Beam Configurations

**Co-propagating beams:**
- Minimal Doppler shift
- Cannot address individual atoms
- Used for global operations

**Counter-propagating beams:**
- Large Doppler shift: $\delta_D = (\mathbf{k}_1 - \mathbf{k}_2) \cdot \mathbf{v}$
- Velocity-selective
- Can address via frequency

**Crossed beams:**
- Intermediate Doppler sensitivity
- Spatial addressing possible
- Most common for local gates

### 4. Light Shifts and Compensation

#### AC Stark Shift from Raman Beams

The Raman beams cause differential light shifts:
$$\Delta_{LS} = \frac{|\Omega_1|^2 - |\Omega_2|^2}{4\Delta} + \frac{|\Omega_1|^2}{4(\Delta + \omega_{HFS})} - \frac{|\Omega_2|^2}{4(\Delta - \omega_{HFS})}$$

For $\Delta \gg \omega_{HFS}$:
$$\Delta_{LS} \approx \frac{|\Omega_1|^2 - |\Omega_2|^2}{4\Delta}$$

**Compensation strategies:**

1. **Balanced intensities**: $|\Omega_1|^2 = |\Omega_2|^2$ eliminates leading term
2. **Magic detuning**: Choose $\Delta$ where light shifts cancel
3. **Composite pulses**: Sequences that refocus light shift errors

#### Tweezer Light Shift

The optical tweezer creates a differential light shift between qubit states:
$$\Delta_{trap} = \frac{\alpha_1 - \alpha_0}{2\epsilon_0 c}I_{trap}$$

For 1064 nm tweezers, this is typically ~kHz, requiring:
- Trap-off gates (turn off tweezer during gates)
- Magic wavelength traps
- Calibration and compensation

### 5. Addressing Strategies

#### Global Addressing

All atoms experience the same gate operation.

**Implementation:**
- Microwave horns
- Large-area Raman beams

**Advantages:**
- Simple optical setup
- Uniform gates
- Fast parallel operations

**Disadvantages:**
- No individual control
- Limited algorithm flexibility

#### Local Addressing

Individual atoms can be controlled independently.

**Crossed-beam addressing:**
Two beams at angle θ create intensity pattern:
$$I(x) \propto \cos^2(k_\perp x)$$

where $k_\perp = 2k\sin(\theta/2)$.

Addressing resolution:
$$\sigma_x \approx \frac{\lambda}{2\sin(\theta/2)}$$

For $\theta = 90°$ and $\lambda = 780$ nm: $\sigma_x \approx 550$ nm

**Focused beam addressing:**
Tightly focused beam addresses single sites.

Spot size: $w_0 \approx \lambda/(2\text{NA})$

With NA = 0.5: $w_0 \approx 780$ nm

**Crosstalk:** Neighboring atoms at distance $a$ experience intensity:
$$\frac{I(a)}{I(0)} = \exp(-2a^2/w_0^2)$$

For $a = 4$ μm, $w_0 = 1$ μm: crosstalk = $10^{-14}$ (negligible)

#### Hybrid Strategies

Modern systems combine:
- Global microwave for rotations
- Local Raman for addressability
- Rydberg for entanglement

### 6. Gate Fidelity Optimization

#### Error Sources

| Error Source | Scaling | Mitigation |
|--------------|---------|------------|
| Laser phase noise | $\propto t_{gate}$ | Faster gates |
| Spontaneous emission | $\propto \Omega^2/\Delta^2$ | Larger detuning |
| Light shifts | $\propto \Omega^2/\Delta$ | Compensation |
| Doppler shifts | $\propto T/m$ | Cooling |
| Addressing crosstalk | $\propto \exp(-a^2/w_0^2)$ | Larger spacing |

#### Composite Pulse Sequences

**BB1 (Broadband 1):**
Corrects pulse area errors:
$$R(\theta)_{composite} = R_\phi(\pi) R_{3\phi}(2\pi) R_\phi(\pi) R_0(\theta)$$

where $\phi = \arccos(-\theta/4\pi)$.

**CORPSE (Compensating for Off-Resonance with a Pulse Sequence):**
Corrects detuning errors:
$$R(\theta)_{CORPSE} = R_\phi(\theta_1) R_{\phi+\pi}(\theta_2) R_\phi(\theta_3)$$

**Fidelity improvement:** Composite pulses can improve fidelity from 99% to 99.9%+.

## Worked Examples

### Example 1: Microwave π-Pulse Design

**Problem:** Design a microwave π-pulse for Rb-87 hyperfine transition. The available microwave power produces B_MW = 5 mG. Calculate pulse duration and estimate fidelity limited by magnetic field noise of 1 mG RMS.

**Solution:**

**Step 1: Calculate Rabi frequency**
The magnetic dipole matrix element for the clock transition is approximately:
$$|\langle F=2, m_F=0|\hat{\mu}|F=1, m_F=0\rangle| \approx \mu_B$$

Rabi frequency:
$$\Omega_{MW} = \frac{\mu_B B_{MW}}{\hbar} = \frac{9.27 \times 10^{-24} \times 5 \times 10^{-7}}{1.05 \times 10^{-34}}$$
$$\Omega_{MW} = 4.4 \times 10^4\,\text{rad/s} = 2\pi \times 7.0\,\text{kHz}$$

**Step 2: π-pulse duration**
$$t_\pi = \frac{\pi}{\Omega_{MW}} = \frac{\pi}{2\pi \times 7000} = 71\,\mu\text{s}$$

**Step 3: Estimate fidelity from B-field noise**
Magnetic field fluctuations cause frequency fluctuations via Zeeman shift:
$$\delta\omega = g_F \mu_B \delta B / \hbar$$

For clock states, this is second-order. The linear sensitivity for non-clock states:
$$\delta\omega \approx \mu_B \times 1\,\text{mG} / \hbar \approx 2\pi \times 1.4\,\text{kHz}$$

Phase accumulated during gate:
$$\delta\phi = \delta\omega \times t_\pi = 2\pi \times 1.4\,\text{kHz} \times 71\,\mu\text{s} = 0.63\,\text{rad}$$

For clock states (second-order sensitivity at 1 G bias):
$$\delta\omega_{clock} \approx 2 \times 575\,\text{Hz/G}^2 \times 1\,\text{G} \times 0.001\,\text{G} = 1.15\,\text{Hz}$$

This gives negligible phase error.

**Fidelity estimate:**
$$F \approx 1 - (\delta\phi)^2/2 \approx 1 - 0.2 = 0.8$$ (for non-clock states)

$$F \approx 0.9999$$ (for clock states)

This demonstrates the advantage of clock state encoding.

---

### Example 2: Raman Gate Optimization

**Problem:** Design a two-photon Raman gate for Rb-87 with the following constraints:
- Target Rabi frequency: $\Omega_{eff} = 2\pi \times 1$ MHz
- Scattering probability < 1%
- Available laser power: 50 mW per beam, beam waist 50 μm

Calculate the required detuning and verify the scattering constraint.

**Solution:**

**Step 1: Calculate single-photon Rabi frequencies**
For 780 nm light:
$$I = \frac{2P}{\pi w_0^2} = \frac{2 \times 0.05}{\pi \times (50 \times 10^{-6})^2} = 1.27 \times 10^7\,\text{W/m}^2$$

Saturation intensity for Rb D2: $I_{sat} = 16$ W/m²

$$\Omega_1 = \Omega_2 = \Gamma\sqrt{\frac{I}{2I_{sat}}} = 2\pi \times 6\,\text{MHz} \times \sqrt{\frac{1.27 \times 10^7}{32}}$$
$$\Omega_1 = 2\pi \times 3.8\,\text{GHz}$$

**Step 2: Required detuning**
$$\Omega_{eff} = \frac{\Omega_1 \Omega_2}{2\Delta}$$

$$\Delta = \frac{\Omega_1 \Omega_2}{2\Omega_{eff}} = \frac{(2\pi \times 3.8 \times 10^9)^2}{2 \times 2\pi \times 10^6}$$
$$\Delta = 2\pi \times 7.2\,\text{THz}$$

This is about 1000 linewidths from the D2 line.

**Step 3: Verify scattering**
Scattering probability during π-pulse ($t_\pi = 500$ ns):
$$P_{sc} = \frac{\Omega_1^2}{4\Delta^2}\Gamma t_\pi = \frac{(3.8 \times 10^9)^2}{4 \times (7.2 \times 10^{12})^2} \times 6 \times 10^6 \times 500 \times 10^{-9}$$
$$P_{sc} = 7 \times 10^{-8} \times 3 = 2.1 \times 10^{-7}$$

This is well below 1%, so the design is acceptable.

**Step 4: Check light shift**
Differential light shift:
$$\Delta_{LS} = \frac{|\Omega_1|^2 - |\Omega_2|^2}{4\Delta} = 0$$ (balanced beams)

The gate is well-optimized.

---

### Example 3: Addressing Crosstalk

**Problem:** A focused Raman beam has waist $w_0 = 1.5$ μm. Atoms are arranged in a square lattice with 4 μm spacing. Calculate:
a) The crosstalk to nearest neighbors
b) The crosstalk to diagonal neighbors
c) The maximum number of simultaneous local gates without exceeding 0.1% total crosstalk error

**Solution:**

**Step 1: Nearest-neighbor crosstalk**
Distance: $r_{NN} = 4$ μm

Intensity ratio:
$$\frac{I(r_{NN})}{I(0)} = \exp\left(-\frac{2 \times 4^2}{1.5^2}\right) = \exp(-35.6) = 3.2 \times 10^{-16}$$

This is completely negligible.

**Step 2: Diagonal crosstalk**
Distance: $r_{diag} = 4\sqrt{2} = 5.66$ μm

$$\frac{I(r_{diag})}{I(0)} = \exp\left(-\frac{2 \times 5.66^2}{1.5^2}\right) = \exp(-28.4) = 4.6 \times 10^{-13}$$

Also negligible.

**Step 3: Adjacent simultaneous gates**
For two adjacent targeted atoms (separated by 4 μm), each experiences intensity $I_0$ from its intended beam.

The error from crosstalk per gate: $\epsilon = I(4\,\mu\text{m})/I_0 \approx 0$.

Since crosstalk is exponentially suppressed, the practical limit is set by other factors (optical power, beam steering speed).

In principle, all atoms could be addressed simultaneously with negligible crosstalk at this spacing and beam waist.

**More realistic analysis:**
If we consider beam pointing stability of σ = 100 nm:
$$\frac{\delta I}{I} = \frac{4\sigma^2}{w_0^2} = \frac{4 \times 0.01}{2.25} = 1.8\%$$

This intensity fluctuation causes gate error:
$$\epsilon_{point} \approx (\delta\Omega/\Omega)^2 = (0.9\%)^2 \approx 10^{-4}$$

This is the dominant addressing error.

## Practice Problems

### Level 1: Direct Application

**Problem 1.1:** Calculate the Rabi frequency for a microwave transition in Cs-133 with B_MW = 20 mG.

**Problem 1.2:** Design a two-photon Raman system with $\Omega_{eff} = 2\pi \times 500$ kHz using 100 GHz detuning. What single-photon Rabi frequencies are required?

**Problem 1.3:** A Raman beam focused to $w_0 = 2$ μm addresses atoms with 5 μm spacing. What is the intensity crosstalk to the nearest neighbor?

### Level 2: Intermediate Analysis

**Problem 2.1:** Compare the fidelity of a 10 μs microwave π-pulse vs a 1 μs Raman π-pulse, assuming:
- Laser phase noise: 1 kHz linewidth
- Magnetic field noise: 0.5 mG RMS at 1 G bias (clock states)
- Spontaneous emission: detuning = 100 GHz

**Problem 2.2:** Design a compensation scheme for the differential light shift from a 1064 nm tweezer (depth 1 mK) during a Raman gate. Calculate the required frequency chirp.

**Problem 2.3:** A composite BB1 pulse sequence is used to correct for 5% Rabi frequency variation across the array. Calculate the residual error after the composite pulse.

### Level 3: Challenging Problems

**Problem 3.1:** Derive the optimal detuning for a Raman gate that minimizes the combined error from spontaneous emission and light shift fluctuations, assuming intensity noise of δI/I.

**Problem 3.2:** Design a STIRAP (Stimulated Raman Adiabatic Passage) pulse sequence for population transfer between hyperfine states. Analyze the adiabaticity condition and calculate the minimum pulse duration for 99.99% transfer efficiency.

**Problem 3.3:** Analyze the effect of motional excitation on Raman gate fidelity. For an atom in the n-th motional state, calculate the effective Rabi frequency and the gate error from the Lamb-Dicke factor.

## Computational Lab: Single-Qubit Gate Simulations

### Lab 1: Microwave Gate Dynamics

```python
"""
Day 914 Lab: Single-Qubit Gate Simulations
Microwave and Raman gates for neutral atoms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_p = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_m = np.array([[0, 0], [1, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def bloch_coordinates(psi):
    """Extract Bloch sphere coordinates from state vector."""
    rho = np.outer(psi, np.conj(psi))
    x = np.real(np.trace(sigma_x @ rho))
    y = np.real(np.trace(sigma_y @ rho))
    z = np.real(np.trace(sigma_z @ rho))
    return x, y, z

def microwave_hamiltonian(Omega, delta=0, phi=0):
    """
    Microwave-driven two-level Hamiltonian.

    H = (delta/2) * sigma_z + (Omega/2) * (cos(phi)*sigma_x + sin(phi)*sigma_y)
    """
    return (delta/2) * sigma_z + (Omega/2) * (np.cos(phi)*sigma_x + np.sin(phi)*sigma_y)

def simulate_gate(H, psi0, t_gate, n_steps=200):
    """Simulate unitary evolution under time-independent H."""
    t = np.linspace(0, t_gate, n_steps)
    psi_t = np.zeros((n_steps, 2), dtype=complex)

    for i, ti in enumerate(t):
        U = expm(-1j * H * ti)
        psi_t[i] = U @ psi0

    return t, psi_t

# Microwave pi-pulse simulation
print("=== Microwave π-Pulse Simulation ===\n")

Omega = 2 * np.pi * 50e3  # 50 kHz Rabi frequency
t_pi = np.pi / Omega

print(f"Rabi frequency: {Omega/(2*np.pi)/1e3:.1f} kHz")
print(f"π-pulse duration: {t_pi*1e6:.1f} μs")

# Simulate starting from |0⟩
psi0 = np.array([1, 0], dtype=complex)
H = microwave_hamiltonian(Omega)

t, psi_t = simulate_gate(H, psi0, 2*t_pi)

# Extract populations and Bloch coordinates
P0 = np.abs(psi_t[:, 0])**2
P1 = np.abs(psi_t[:, 1])**2
bloch = np.array([bloch_coordinates(psi) for psi in psi_t])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Population dynamics
axes[0].plot(t*1e6, P0, 'b-', label='|0⟩', linewidth=2)
axes[0].plot(t*1e6, P1, 'r-', label='|1⟩', linewidth=2)
axes[0].axvline(x=t_pi*1e6, color='gray', linestyle='--', label='π-pulse')
axes[0].set_xlabel('Time (μs)')
axes[0].set_ylabel('Population')
axes[0].set_title('Microwave Rabi Oscillations')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bloch sphere trajectory
from mpl_toolkits.mplot3d import Axes3D

ax3d = fig.add_subplot(132, projection='3d')
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax3d.plot_surface(x, y, z, alpha=0.1, color='gray')

ax3d.plot(bloch[:, 0], bloch[:, 1], bloch[:, 2], 'b-', linewidth=2)
ax3d.scatter([bloch[0, 0]], [bloch[0, 1]], [bloch[0, 2]], c='g', s=100, label='Start')
ax3d.scatter([bloch[-1, 0]], [bloch[-1, 1]], [bloch[-1, 2]], c='r', s=100, label='End')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_title('Bloch Sphere Trajectory')
axes[1].axis('off')  # Hide the 2D axis

# Effect of detuning
delta_values = [0, 0.5*Omega, Omega, 2*Omega]
axes[2].set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, len(delta_values))))

for delta in delta_values:
    H = microwave_hamiltonian(Omega, delta)
    t, psi_t = simulate_gate(H, psi0, 2*t_pi)
    P1 = np.abs(psi_t[:, 1])**2
    axes[2].plot(t*1e6, P1, label=f'δ/Ω = {delta/Omega:.1f}', linewidth=2)

axes[2].set_xlabel('Time (μs)')
axes[2].set_ylabel('P(|1⟩)')
axes[2].set_title('Effect of Detuning')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('microwave_gate.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 2: Raman Gate with Light Shifts

```python
"""
Lab 2: Two-photon Raman gate simulation with light shifts
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def raman_3level_dynamics(t, y, Omega1, Omega2, Delta, Gamma, omega_HFS, omega_12):
    """
    Three-level system dynamics for Raman transitions.

    States: |0⟩ (F=1), |e⟩ (excited), |1⟩ (F=2)

    y = [Re(c0), Im(c0), Re(ce), Im(ce), Re(c1), Im(c1)]
    """
    c0 = y[0] + 1j * y[1]
    ce = y[2] + 1j * y[3]
    c1 = y[4] + 1j * y[5]

    # Two-photon detuning
    delta_2ph = omega_12 - omega_HFS

    # Equations of motion (rotating frame)
    dc0 = -1j * Omega1/2 * ce
    dce = -1j * Omega1/2 * c0 - 1j * Omega2/2 * c1 + (1j*Delta - Gamma/2) * ce
    dc1 = -1j * Omega2/2 * ce + 1j * delta_2ph * c1

    return [dc0.real, dc0.imag, dce.real, dce.imag, dc1.real, dc1.imag]

def effective_2level_raman(t, y, Omega_eff, delta_LS, delta_2ph):
    """
    Effective two-level dynamics after adiabatic elimination.

    y = [Re(c0), Im(c0), Re(c1), Im(c1)]
    """
    c0 = y[0] + 1j * y[1]
    c1 = y[2] + 1j * y[3]

    # Include light shift
    dc0 = -1j * Omega_eff/2 * c1 - 1j * delta_LS/2 * c0
    dc1 = -1j * Omega_eff/2 * c0 + 1j * (delta_2ph - delta_LS/2) * c1

    return [dc0.real, dc0.imag, dc1.real, dc1.imag]

# Parameters
Omega1 = 2 * np.pi * 1e9  # 1 GHz single-photon Rabi
Omega2 = 2 * np.pi * 1e9  # 1 GHz single-photon Rabi
Delta = 2 * np.pi * 100e9  # 100 GHz detuning
Gamma = 2 * np.pi * 6e6  # 6 MHz linewidth
omega_HFS = 2 * np.pi * 6.835e9  # Rb-87 hyperfine

# Effective parameters
Omega_eff = Omega1 * Omega2 / (2 * Delta)
print(f"=== Raman Gate Parameters ===")
print(f"Single-photon Rabi: {Omega1/(2*np.pi)/1e9:.1f} GHz")
print(f"Detuning: {Delta/(2*np.pi)/1e9:.0f} GHz")
print(f"Effective Rabi: {Omega_eff/(2*np.pi)/1e6:.1f} MHz")
print(f"π-pulse time: {np.pi/Omega_eff*1e9:.1f} ns")

# Scattering rate
Gamma_sc = Gamma * (Omega1/(2*Delta))**2
print(f"Scattering rate: {Gamma_sc/(2*np.pi):.0f} Hz")
print(f"Scattering per π-pulse: {Gamma_sc * np.pi/Omega_eff:.2e}")

# Simulate with different light shift imbalances
delta_LS_values = [0, 0.1*Omega_eff, 0.3*Omega_eff, Omega_eff]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

t_max = 4 * np.pi / Omega_eff
t_eval = np.linspace(0, t_max, 500)

for delta_LS in delta_LS_values:
    y0 = [1, 0, 0, 0]  # Start in |0⟩

    sol = solve_ivp(effective_2level_raman, (0, t_max), y0,
                   args=(Omega_eff, delta_LS, 0),
                   t_eval=t_eval, method='RK45')

    P1 = sol.y[2]**2 + sol.y[3]**2
    axes[0].plot(sol.t * Omega_eff / (2*np.pi), P1,
                label=f'δ_LS/Ω = {delta_LS/Omega_eff:.1f}', linewidth=2)

axes[0].set_xlabel('Time (1/Ω_eff)')
axes[0].set_ylabel('P(|1⟩)')
axes[0].set_title('Effect of Light Shift Imbalance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Compare full 3-level vs effective 2-level
y0_3level = [1, 0, 0, 0, 0, 0]
y0_2level = [1, 0, 0, 0]

sol_3level = solve_ivp(raman_3level_dynamics, (0, t_max), y0_3level,
                       args=(Omega1, Omega2, Delta, Gamma, omega_HFS, omega_HFS),
                       t_eval=t_eval, method='RK45', max_step=1e-12)

sol_2level = solve_ivp(effective_2level_raman, (0, t_max), y0_2level,
                       args=(Omega_eff, 0, 0),
                       t_eval=t_eval, method='RK45')

P1_3level = sol_3level.y[4]**2 + sol_3level.y[5]**2
P1_2level = sol_2level.y[2]**2 + sol_2level.y[3]**2

axes[1].plot(sol_3level.t * Omega_eff / (2*np.pi), P1_3level, 'b-',
            label='Full 3-level', linewidth=2)
axes[1].plot(sol_2level.t * Omega_eff / (2*np.pi), P1_2level, 'r--',
            label='Effective 2-level', linewidth=2)
axes[1].set_xlabel('Time (1/Ω_eff)')
axes[1].set_ylabel('P(|1⟩)')
axes[1].set_title('3-Level vs Effective 2-Level Model')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('raman_gate_simulation.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze gate fidelity vs detuning
print("\n=== Fidelity vs Detuning Analysis ===")

Delta_values = np.logspace(10, 12, 30)  # 10 GHz to 1 THz
fidelities = []
gate_times = []
scattering_probs = []

Omega_target = 2 * np.pi * 1e6  # Target 1 MHz effective Rabi

for Delta_test in Delta_values:
    # Adjust single-photon Rabi to maintain target effective Rabi
    Omega_single = np.sqrt(2 * Delta_test * Omega_target)

    # Gate time
    t_pi = np.pi / Omega_target
    gate_times.append(t_pi)

    # Scattering probability
    Gamma_sc = Gamma * (Omega_single/(2*Delta_test))**2
    P_sc = Gamma_sc * t_pi
    scattering_probs.append(P_sc)

    # Fidelity (limited by scattering)
    F = 1 - P_sc
    fidelities.append(F)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].semilogx(Delta_values/(2*np.pi)/1e9, 1 - np.array(fidelities), 'b-', linewidth=2)
axes[0].set_xlabel('Detuning (GHz)')
axes[0].set_ylabel('Infidelity')
axes[0].set_title('Gate Error vs Detuning')
axes[0].grid(True, alpha=0.3)

axes[1].loglog(Delta_values/(2*np.pi)/1e9, scattering_probs, 'r-', linewidth=2)
axes[1].set_xlabel('Detuning (GHz)')
axes[1].set_ylabel('Scattering probability')
axes[1].set_title('Scattering per Gate')
axes[1].grid(True, alpha=0.3)

# Required single-photon intensity
I_required = (2 * Delta_values * Omega_target) / (2*np.pi*6e6)**2 * 16  # Relative to Isat
axes[2].loglog(Delta_values/(2*np.pi)/1e9, I_required, 'g-', linewidth=2)
axes[2].set_xlabel('Detuning (GHz)')
axes[2].set_ylabel('Required intensity (I/I_sat)')
axes[2].set_title('Intensity Requirement')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('raman_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Lab 3: Addressing and Crosstalk

```python
"""
Lab 3: Local addressing and crosstalk analysis
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian_intensity(x, y, w0, x0=0, y0=0):
    """2D Gaussian intensity profile."""
    r2 = (x - x0)**2 + (y - y0)**2
    return np.exp(-2 * r2 / w0**2)

def calculate_crosstalk(spacing, w0):
    """Calculate intensity crosstalk to neighbors."""
    return np.exp(-2 * spacing**2 / w0**2)

# Create 5x5 atom array
n_atoms = 5
spacing = 4.0  # μm
positions = []
for i in range(n_atoms):
    for j in range(n_atoms):
        positions.append([i * spacing, j * spacing])
positions = np.array(positions)

# Addressing beam
w0 = 1.5  # μm

# Target the center atom
target_idx = 12  # Center of 5x5 array
target_pos = positions[target_idx]

# Calculate intensity at all positions
intensities = gaussian_intensity(positions[:, 0], positions[:, 1], w0,
                                  target_pos[0], target_pos[1])

print("=== Addressing Crosstalk Analysis ===")
print(f"Beam waist: {w0} μm")
print(f"Atom spacing: {spacing} μm")
print(f"Target atom: {target_idx} at {target_pos}")
print(f"\nIntensity at each atom (relative to target):")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 2D intensity map
x = np.linspace(-2, (n_atoms+1)*spacing, 200)
y = np.linspace(-2, (n_atoms+1)*spacing, 200)
X, Y = np.meshgrid(x, y)
I_map = gaussian_intensity(X, Y, w0, target_pos[0], target_pos[1])

im = axes[0].pcolormesh(X, Y, I_map, cmap='hot', shading='auto')
axes[0].scatter(positions[:, 0], positions[:, 1], c='cyan', s=100, edgecolors='white')
axes[0].scatter([target_pos[0]], [target_pos[1]], c='lime', s=200, marker='*')
axes[0].set_xlabel('x (μm)')
axes[0].set_ylabel('y (μm)')
axes[0].set_title('Addressing Beam Intensity')
axes[0].set_aspect('equal')
plt.colorbar(im, ax=axes[0], label='Relative intensity')

# Crosstalk to neighbors
distances = np.sqrt(np.sum((positions - target_pos)**2, axis=1))
sorted_idx = np.argsort(distances)

axes[1].bar(range(len(intensities)), intensities[sorted_idx], color='steelblue')
axes[1].set_xlabel('Atom index (sorted by distance)')
axes[1].set_ylabel('Relative intensity')
axes[1].set_title('Crosstalk per Atom')
axes[1].set_yscale('log')
axes[1].axhline(y=1e-4, color='r', linestyle='--', label='0.01% threshold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Crosstalk vs beam waist
w0_values = np.linspace(0.5, 3.0, 50)
crosstalk_NN = calculate_crosstalk(spacing, w0_values)
crosstalk_diag = calculate_crosstalk(spacing*np.sqrt(2), w0_values)

axes[2].semilogy(w0_values, crosstalk_NN, 'b-', label='Nearest neighbor', linewidth=2)
axes[2].semilogy(w0_values, crosstalk_diag, 'r-', label='Diagonal', linewidth=2)
axes[2].axhline(y=1e-4, color='gray', linestyle='--', label='0.01% threshold')
axes[2].set_xlabel('Beam waist (μm)')
axes[2].set_ylabel('Crosstalk')
axes[2].set_title(f'Crosstalk vs Beam Waist (spacing = {spacing} μm)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('addressing_crosstalk.png', dpi=150, bbox_inches='tight')
plt.show()

# Multi-site addressing
print("\n=== Multi-Site Addressing ===")

# Address 4 atoms in a 2x2 pattern
target_indices = [6, 7, 11, 12]
target_positions = positions[target_indices]

# Calculate total intensity at each site
total_intensity = np.zeros(len(positions))
for tp in target_positions:
    total_intensity += gaussian_intensity(positions[:, 0], positions[:, 1], w0, tp[0], tp[1])

# Calculate total crosstalk
target_set = set(target_indices)
crosstalk_total = 0
for i, I in enumerate(total_intensity):
    if i not in target_set:
        crosstalk_total += I

print(f"Target atoms: {target_indices}")
print(f"Total crosstalk to non-target atoms: {crosstalk_total:.2e}")
print(f"Average crosstalk per target: {crosstalk_total/len(target_indices):.2e}")

fig, ax = plt.subplots(figsize=(8, 8))

# Plot intensity from all addressing beams
I_total = np.zeros_like(X)
for tp in target_positions:
    I_total += gaussian_intensity(X, Y, w0, tp[0], tp[1])

im = ax.pcolormesh(X, Y, I_total, cmap='hot', shading='auto')
ax.scatter(positions[:, 0], positions[:, 1], c='cyan', s=100, edgecolors='white')
ax.scatter(target_positions[:, 0], target_positions[:, 1], c='lime', s=200, marker='*')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title('Multi-Site Addressing')
ax.set_aspect('equal')
plt.colorbar(im, ax=ax, label='Total relative intensity')

plt.tight_layout()
plt.savefig('multi_site_addressing.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Summary

### Key Formulas Table

| Quantity | Formula | Typical Value |
|----------|---------|---------------|
| Microwave Rabi freq | $\Omega_{MW} = \mu_B B_{MW}/\hbar$ | 10-100 kHz |
| Raman effective Rabi | $\Omega_{eff} = \Omega_1\Omega_2/2\Delta$ | 0.1-10 MHz |
| Scattering rate | $\Gamma_{sc} = (\Omega/2\Delta)^2\Gamma$ | Hz-kHz |
| Light shift | $\Delta_{LS} = |\Omega|^2/4\Delta$ | kHz-MHz |
| Crosstalk | $I(r)/I(0) = e^{-2r^2/w_0^2}$ | <10⁻⁴ |

### Main Takeaways

1. **Hyperfine clock states** provide long coherence times (>1s) due to their insensitivity to first-order magnetic field fluctuations.

2. **Microwave gates** are simple but slow (~10-100 μs); they provide global operations on all atoms simultaneously.

3. **Raman gates** enable fast (~100 ns) local operations through focused beams, with scattering limited by the large detuning from excited states.

4. **Light shift compensation** is critical for high-fidelity Raman gates, achieved through intensity balancing or composite pulse sequences.

5. **Addressing crosstalk** is exponentially suppressed with beam waist, making individual addressing feasible with μm-scale focusing.

## Daily Checklist

### Conceptual Understanding
- [ ] I can explain clock state advantages
- [ ] I understand Raman gate mechanisms
- [ ] I can describe addressing strategies
- [ ] I know the main error sources

### Mathematical Skills
- [ ] I can calculate microwave Rabi frequencies
- [ ] I can derive effective Raman parameters
- [ ] I can estimate gate fidelities

### Computational Skills
- [ ] I can simulate microwave dynamics
- [ ] I can model Raman gates with light shifts
- [ ] I can analyze addressing crosstalk

## Preview: Day 915

Tomorrow we explore **Two-Qubit Rydberg Gates**, where we will:
- Implement CZ gates using blockade dynamics
- Analyze Rydberg-dressed interactions for continuous coupling
- Design native multi-qubit gates (CCZ)
- Optimize gate fidelity against Rydberg decay and motional errors

The combination of high-fidelity single-qubit gates from today with Rydberg-based entangling gates enables universal quantum computation with neutral atoms.
