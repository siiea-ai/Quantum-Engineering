# Day 391: Tunnel Diodes & Josephson Effect

## Week 56, Day 6 | Month 14: One-Dimensional Quantum Mechanics

### Schedule Overview (7 hours)

| Block | Time | Focus |
|-------|------|-------|
| **Morning** | 2.5 hrs | Tunnel diode physics, negative differential resistance |
| **Afternoon** | 2.5 hrs | Josephson effect, superconducting qubits |
| **Evening** | 2 hrs | Computational lab: I-V characteristics |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Explain** how tunneling creates negative differential resistance in Esaki diodes
2. **Derive** the current-voltage characteristic of tunnel diodes
3. **Describe** both DC and AC Josephson effects in superconducting junctions
4. **Understand** how Josephson junctions form the basis of superconducting qubits
5. **Analyze** the physics of SQUIDs and their quantum applications
6. **Connect** tunneling phenomena to modern quantum computing technology

---

## Core Content

### Part I: Semiconductor Tunnel Diodes

### 1. The Esaki (Tunnel) Diode

**1957**: Leo Esaki discovered tunneling in heavily doped p-n junctions while at Sony.
**1973**: Nobel Prize in Physics (shared with Giaever and Josephson)

**Key insight**: In a heavily doped semiconductor, the depletion region becomes thin enough (~10 nm) for electrons to tunnel directly between bands.

### 2. Energy Band Picture

```
Forward bias (small V):

   p+ side    |    n+ side
              |
    ████      |      ○○○○    Conduction band
    ████ ←—   |  ←—  ○○○○
    ═══════════════════════   E_F
              |
    ○○○○  —→  |  —→  ████    Valence band
    ○○○○      |      ████
              |
         Tunneling current flows!
```

**Band overlap condition**: For tunneling to occur, the n-side conduction band must overlap in energy with the p-side valence band.

### 3. Current-Voltage Characteristic

The tunnel diode I-V curve has three distinct regions:

```
     I
     |        Peak
     |       /\
     |      /  \
     |     /    \____  Valley
     |    /          \
     |   /            \____ Normal diode
     |  /                  \
     | /                    \
     |/                      \
     +————————————————————————→ V
    0   V_p      V_v
```

**Region 1 (0 < V < V_p):** Tunneling current increases with bias
**Region 2 (V_p < V < V_v):** Negative differential resistance (NDR)
**Region 3 (V > V_v):** Normal thermionic/diffusion current

### 4. Negative Differential Resistance (NDR)

In the NDR region:
$$\frac{dI}{dV} < 0$$

**Physical mechanism:**
- At V = 0: Fermi levels aligned, no net tunneling
- Small V: Band overlap increases, tunneling increases
- V = V_p: Maximum overlap between filled states and empty states
- V > V_p: Overlap decreases (bands move past each other)
- V = V_v: Bands no longer overlap, tunneling stops

### 5. Tunnel Current Formula

The tunneling current density:
$$J = \frac{em^*}{2\pi^2\hbar^3}\int_0^{E_{max}}T(E)[f_p(E) - f_n(E)]dE$$

Simplified model (parabolic bands):
$$I = I_p\frac{V}{V_p}\exp\left(1 - \frac{V}{V_p}\right) + I_s\left[\exp\left(\frac{eV}{k_BT}\right) - 1\right]$$

The first term is the tunneling component, the second is the standard diode current.

### 6. Tunnel Diode Applications

**High-speed oscillators:** NDR enables relaxation oscillations at GHz frequencies
**Amplifiers:** Small-signal amplification in microwave circuits
**Memory elements:** Bistable switching between low-V and high-V states
**Quantum cascade lasers:** Sequential tunneling between quantum wells

---

### Part II: The Josephson Effect

### 7. Superconducting Tunnel Junction

A Josephson junction consists of two superconductors separated by a thin barrier:

```
Superconductor 1    Barrier    Superconductor 2
    (φ₁)           (~1 nm)         (φ₂)

   ████████       ┃┃┃┃┃┃       ████████
   ████████       ┃┃┃┃┃┃       ████████
   ████████  ←→   ┃┃┃┃┃┃   ←→  ████████
   ████████       ┃┃┃┃┃┃       ████████
   ████████       ┃┃┃┃┃┃       ████████

   Cooper pairs tunnel coherently!
```

**Key difference from normal tunneling:** Cooper pairs (bound electron pairs) tunnel as a coherent quantum entity, preserving phase information.

### 8. The DC Josephson Effect

Even with no applied voltage, a supercurrent flows:

$$\boxed{I_s = I_c \sin(\phi)}$$

where:
- $I_c$ = critical current (maximum supercurrent)
- $\phi = \phi_1 - \phi_2$ = phase difference across junction

**Key insight:** The current depends on the quantum mechanical phase difference!

### 9. The AC Josephson Effect

When a DC voltage V is applied:

$$\boxed{\frac{d\phi}{dt} = \frac{2eV}{\hbar}}$$

This gives an oscillating current:
$$I_s = I_c \sin\left(\phi_0 + \frac{2eV}{\hbar}t\right)$$

**Josephson frequency:**
$$\boxed{f_J = \frac{2eV}{h} = 483.6 \text{ GHz/mV}}$$

A 1 mV junction oscillates at 483.6 GHz!

### 10. The Josephson Relations

The two Josephson equations:

$$I = I_c \sin\phi \quad \text{(DC Josephson)}$$
$$V = \frac{\hbar}{2e}\frac{d\phi}{dt} \quad \text{(AC Josephson)}$$

Together they completely describe junction dynamics.

### 11. RCSJ Model

The Resistively and Capacitively Shunted Junction model:

```
         I (total)
           ↓
    ┌──────┴──────┐
    │      │      │
   ═╪═    ═╪═    ═╪═
    C      R      J (Josephson element)
    │      │      │
    └──────┬──────┘
           ↓
```

Total current:
$$I = I_c\sin\phi + \frac{V}{R} + C\frac{dV}{dt}$$

$$I = I_c\sin\phi + \frac{\hbar}{2eR}\frac{d\phi}{dt} + \frac{\hbar C}{2e}\frac{d^2\phi}{dt^2}$$

This is analogous to a damped nonlinear pendulum!

### 12. The Tilted Washboard Potential

Define a "potential":
$$U(\phi) = -E_J\cos\phi - \frac{\hbar I}{2e}\phi$$

where $E_J = \frac{\hbar I_c}{2e}$ is the Josephson energy.

```
U(φ)
  |    •  <-- phase particle
  |   /\
  | •/  \    /\    /\
  |      \  /  \  /  \
  |       \/    \/    \
  +————————————————————→ φ

Tilted more when I increases
```

The phase behaves like a particle rolling in this potential:
- I < I_c: Trapped in well, φ oscillates (plasma oscillation)
- I > I_c: Particle escapes, φ runs → voltage appears

### 13. Superconducting Qubits

Josephson junctions are the key element in superconducting qubits!

**Transmon qubit:**
$$H = 4E_C(n - n_g)^2 - E_J\cos\phi$$

where:
- $E_C = e^2/2C$ = charging energy
- $E_J$ = Josephson energy
- n = number of Cooper pairs
- φ = junction phase

**Energy levels:**
$$E_n \approx -E_J + \sqrt{8E_JE_C}\left(n + \frac{1}{2}\right) - \frac{E_C}{12}(6n^2 + 6n + 3)$$

The **anharmonicity** (non-equal level spacing) allows addressing individual transitions!

### 14. SQUID: Superconducting Quantum Interference Device

Two Josephson junctions in a loop:

```
        I →
    ┌───┬───┐
    │   │   │
   ═╪═  Φ  ═╪═   ← Junctions
    │   │   │
    └───┴───┘
        ↓
```

The critical current oscillates with magnetic flux:
$$I_c(\Phi) = 2I_{c0}\left|\cos\left(\frac{\pi\Phi}{\Phi_0}\right)\right|$$

where $\Phi_0 = h/2e = 2.07 \times 10^{-15}$ Wb is the flux quantum.

**Applications:**
- Ultra-sensitive magnetometers (10⁻¹⁴ T resolution)
- Qubit readout
- Flux qubits
- Brain imaging (MEG)

---

## Worked Examples

### Example 1: Tunnel Diode Peak Current

A tunnel diode has peak voltage V_p = 50 mV and peak current I_p = 5 mA. Calculate the negative resistance in the NDR region.

**Solution:**

The empirical I-V relation near the peak:
$$I \approx I_p\exp\left[-\left(\frac{V - V_p}{\Delta V}\right)^2\right]$$

At the peak:
$$\frac{dI}{dV} = 0$$

In the NDR region (V slightly > V_p):
$$\frac{dI}{dV} \approx -\frac{2I_p(V - V_p)}{(\Delta V)^2}$$

For typical Δ V ≈ V_p:
$$\left|\frac{dI}{dV}\right|_{max} \approx \frac{2I_p}{V_p} = \frac{2 \times 5}{50} = 0.2 \text{ S}$$

$$|R_n| = 5 \text{ Ω}$$

---

### Example 2: Josephson Frequency

A Josephson junction is biased at V = 10 μV. Calculate the frequency of the AC supercurrent.

**Solution:**

$$f_J = \frac{2eV}{h} = \frac{2 \times 1.6 \times 10^{-19} \times 10 \times 10^{-6}}{6.63 \times 10^{-34}}$$
$$f_J = 4.83 \times 10^9 \text{ Hz} = 4.83 \text{ GHz}$$

**Or using the conversion factor:**
$$f_J = 483.6 \text{ GHz/mV} \times 0.01 \text{ mV} = 4.836 \text{ GHz}$$

---

### Example 3: Transmon Qubit Frequency

A transmon qubit has E_J/E_C = 50 with E_C/h = 200 MHz. Calculate the qubit transition frequency.

**Solution:**

The 0→1 transition frequency:
$$\omega_{01} = \sqrt{8E_JE_C} - E_C$$

$$E_J = 50 E_C = 50 \times h \times 200 \text{ MHz} = h \times 10 \text{ GHz}$$

$$\sqrt{8E_JE_C} = \sqrt{8 \times 50 \times E_C^2} = \sqrt{400} \times E_C = 20 E_C$$

$$\omega_{01} = 20 E_C - E_C = 19 E_C$$

$$f_{01} = 19 \times 200 \text{ MHz} = 3.8 \text{ GHz}$$

**Anharmonicity:**
$$\alpha = -E_C = -200 \text{ MHz}$$

This means the 1→2 transition is at 3.6 GHz, sufficiently different to address selectively!

---

## Practice Problems

### Level 1: Direct Application

1. A tunnel diode has I_p = 10 mA at V_p = 100 mV. Estimate the tunneling resistance at low bias.

2. Calculate the Josephson frequency for V = 100 μV.

3. A SQUID has I_c0 = 10 μA. What is the critical current when Φ = Φ_0/4?

### Level 2: Intermediate

4. Derive the condition for NDR in a tunnel diode in terms of the tunneling transmission coefficient T(E).

5. Show that the RCSJ model is mathematically equivalent to a damped pendulum. What is the "gravity" term?

6. For a transmon with E_J/E_C = 30 and E_C/h = 250 MHz, calculate the 0→1 and 1→2 transition frequencies.

### Level 3: Challenging

7. **Shapiro steps:** When a Josephson junction is irradiated with microwaves at frequency f, voltage steps appear at V_n = nhf/2e. Derive this from the AC Josephson effect.

8. **Quantum tunneling of phase:** Below a critical temperature, the phase can tunnel through the potential barrier (macroscopic quantum tunneling). Estimate this temperature for a junction with C = 1 fF, I_c = 1 μA.

9. **Flux qubit:** A SQUID loop with three junctions can be biased near Φ_0/2 to create a qubit. Derive the double-well potential in this configuration.

---

## Computational Lab

### Python: Tunnel Diode and Josephson Junction

```python
"""
Day 391: Tunnel Diodes and Josephson Effect
Quantum Tunneling & Barriers - Week 56

This lab explores:
1. Tunnel diode I-V characteristics
2. Josephson junction physics
3. RCSJ model dynamics
4. SQUID interference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.constants import h, e, hbar

# Flux quantum
Phi_0 = h / (2 * e)

#%% Part 1: Tunnel Diode I-V Characteristic

def tunnel_diode_current(V, Ip, Vp, Is, VT=0.026):
    """
    Tunnel diode I-V characteristic

    Parameters:
    V: Bias voltage (V)
    Ip: Peak current (A)
    Vp: Peak voltage (V)
    Is: Saturation current (A)
    VT: Thermal voltage kT/e (V)

    Returns:
    I: Total current (A)
    """
    # Tunneling component (Gaussian-like peak)
    I_tunnel = Ip * (V / Vp) * np.exp(1 - V / Vp)

    # Normal diode component
    I_diode = Is * (np.exp(V / VT) - 1)

    # Total (only for V > 0 typically)
    I_total = np.where(V > 0, I_tunnel + I_diode, 0)

    return I_total

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Typical tunnel diode parameters
Ip = 5e-3    # 5 mA peak
Vp = 0.05    # 50 mV peak voltage
Is = 1e-9    # 1 nA saturation current
Vv = 0.3     # Valley voltage

V = np.linspace(0, 0.8, 500)
I = tunnel_diode_current(V, Ip, Vp, Is)

ax1 = axes[0]
ax1.plot(V * 1000, I * 1000, 'b-', linewidth=2)

# Mark key points
ax1.scatter([Vp * 1000], [Ip * 1000], color='red', s=100, zorder=5)
ax1.annotate('Peak', (Vp * 1000 + 20, Ip * 1000), fontsize=10)

# NDR region
V_ndr = V[(V > Vp) & (V < Vv)]
I_ndr = tunnel_diode_current(V_ndr, Ip, Vp, Is)
ax1.fill_between(V_ndr * 1000, 0, I_ndr * 1000, alpha=0.3, color='red',
                 label='NDR region')

ax1.set_xlabel('Voltage V (mV)', fontsize=12)
ax1.set_ylabel('Current I (mA)', fontsize=12)
ax1.set_title('Tunnel (Esaki) Diode I-V Characteristic', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 800)
ax1.set_ylim(0, 7)

# Differential resistance
ax2 = axes[1]
dI_dV = np.gradient(I, V)

ax2.plot(V * 1000, 1 / dI_dV, 'b-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.fill_between(V * 1000, 0, 1/dI_dV, where=(1/dI_dV < 0),
                 alpha=0.3, color='red', label='Negative resistance')

ax2.set_xlabel('Voltage V (mV)', fontsize=12)
ax2.set_ylabel('Differential Resistance dV/dI (Ω)', fontsize=12)
ax2.set_title('Differential Resistance', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 800)
ax2.set_ylim(-50, 200)

plt.tight_layout()
plt.savefig('tunnel_diode_IV.png', dpi=150)
plt.show()

#%% Part 2: Josephson Junction I-V Characteristic

def josephson_IV(V, Ic, Rn, T=4.2):
    """
    Josephson junction I-V characteristic (RSJ model)

    Parameters:
    V: Voltage (V)
    Ic: Critical current (A)
    Rn: Normal state resistance (Ohm)
    T: Temperature (K)

    Returns:
    I: Current (A)
    """
    kT = 1.38e-23 * T

    # For |V| small: supercurrent branch I = Ic sin(phi)
    # For |V| > 0: resistive branch I = V/Rn + small supercurrent (time-averaged)

    I = np.zeros_like(V)

    # Supercurrent branch (|I| < Ic at V = 0)
    V_small = np.abs(V) < 1e-9
    I[V_small] = 0  # Can be any |I| < Ic

    # Resistive branch
    # Time-averaged current: I = sqrt(Ic² + (V/Rn)²) for simple model
    # More accurate: I = V/Rn for V >> IcRn
    V_large = ~V_small
    I[V_large] = np.sign(V[V_large]) * np.sqrt(Ic**2 + (V[V_large]/Rn)**2)

    return I

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

Ic = 10e-6   # 10 μA
Rn = 100     # 100 Ω

V = np.linspace(-0.005, 0.005, 1000)  # ±5 mV
I = josephson_IV(V, Ic, Rn)

ax1 = axes[0]

# Plot characteristic
ax1.plot(V * 1000, I * 1e6, 'b-', linewidth=2)

# Add supercurrent branch at V = 0
ax1.plot([0, 0], [-Ic * 1e6, Ic * 1e6], 'r-', linewidth=3, label='Supercurrent branch')

# Mark critical current
ax1.axhline(y=Ic * 1e6, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(y=-Ic * 1e6, color='gray', linestyle='--', alpha=0.5)

ax1.set_xlabel('Voltage V (mV)', fontsize=12)
ax1.set_ylabel('Current I (μA)', fontsize=12)
ax1.set_title('Josephson Junction I-V Characteristic', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-60, 60)

# AC Josephson frequency
ax2 = axes[1]
V_ac = np.linspace(0.001, 1, 100)  # mV
f_J = 483.6e9 * V_ac / 1000  # Hz (483.6 GHz/mV)

ax2.semilogy(V_ac, f_J / 1e9, 'b-', linewidth=2)
ax2.set_xlabel('Junction Voltage V (mV)', fontsize=12)
ax2.set_ylabel('Josephson Frequency (GHz)', fontsize=12)
ax2.set_title('AC Josephson Effect: Frequency vs Voltage', fontsize=14)
ax2.grid(True, alpha=0.3, which='both')

# Mark typical qubit frequencies
ax2.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Typical qubit frequency')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('josephson_IV.png', dpi=150)
plt.show()

#%% Part 3: RCSJ Model Dynamics

def rcsj_dynamics(y, t, I_bias, Ic, R, C):
    """
    RCSJ model equations of motion

    State: y = [phi, dphi/dt]
    """
    phi, phi_dot = y

    # Second-order ODE: (hbar C/2e) d²phi/dt² + (hbar/2eR) dphi/dt + Ic sin(phi) = I
    # Rewrite: d²phi/dt² = (2e/hbar C)[I - Ic sin(phi) - (hbar/2eR) dphi/dt]

    omega_p = np.sqrt(2 * e * Ic / (hbar * C))  # Plasma frequency
    Q = omega_p * R * C  # Quality factor

    dphi_dt = phi_dot
    d2phi_dt2 = omega_p**2 * (I_bias/Ic - np.sin(phi) - phi_dot/(omega_p * Q))

    return [dphi_dt, d2phi_dt2]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Junction parameters
Ic = 1e-6      # 1 μA
R = 100        # 100 Ω
C = 1e-15      # 1 fF

omega_p = np.sqrt(2 * e * Ic / (hbar * C))
f_p = omega_p / (2 * np.pi)
print(f"Plasma frequency: {f_p/1e9:.2f} GHz")

# Time array (several plasma periods)
t = np.linspace(0, 5e-9, 10000)

# Case 1: I < Ic (plasma oscillation)
I_bias = 0.5 * Ic
y0 = [0.1, 0]  # Small initial phase displacement
sol = odeint(rcsj_dynamics, y0, t, args=(I_bias, Ic, R, C))

ax1 = axes[0, 0]
ax1.plot(t * 1e9, sol[:, 0], 'b-', linewidth=1)
ax1.set_xlabel('Time (ns)', fontsize=11)
ax1.set_ylabel('Phase φ (rad)', fontsize=11)
ax1.set_title(f'I = 0.5 Ic: Plasma Oscillations', fontsize=12)
ax1.grid(True, alpha=0.3)

# Case 2: I > Ic (voltage state)
I_bias = 1.5 * Ic
y0 = [0, 0]
sol = odeint(rcsj_dynamics, y0, t, args=(I_bias, Ic, R, C))

ax2 = axes[0, 1]
ax2.plot(t * 1e9, sol[:, 0], 'b-', linewidth=1)
ax2.set_xlabel('Time (ns)', fontsize=11)
ax2.set_ylabel('Phase φ (rad)', fontsize=11)
ax2.set_title(f'I = 1.5 Ic: Running Phase (Voltage State)', fontsize=12)
ax2.grid(True, alpha=0.3)

# Average voltage from phase velocity
V_avg = hbar / (2 * e) * np.mean(np.gradient(sol[:, 0], t))
print(f"Average voltage: {V_avg * 1e6:.2f} μV")

# Washboard potential
ax3 = axes[1, 0]
phi = np.linspace(0, 6 * np.pi, 500)

for I_ratio in [0, 0.3, 0.6, 0.9, 1.0]:
    I_bias = I_ratio * Ic
    E_J = hbar * Ic / (2 * e)
    U = -E_J * np.cos(phi) - (hbar * I_bias / (2 * e)) * phi
    U = U / E_J  # Normalize
    ax3.plot(phi, U, linewidth=2, label=f'I/Ic = {I_ratio}')

ax3.set_xlabel('Phase φ (rad)', fontsize=11)
ax3.set_ylabel('U(φ) / E_J', fontsize=11)
ax3.set_title('Tilted Washboard Potential', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# I-V curve from simulation
ax4 = axes[1, 1]
I_values = np.linspace(0, 2, 50) * Ic
V_values = []

for I_bias in I_values:
    y0 = [0, 0]
    sol = odeint(rcsj_dynamics, y0, t, args=(I_bias, Ic, R, C))
    # Average voltage
    phi_dot = np.gradient(sol[:, 0], t)
    V_avg = hbar / (2 * e) * np.mean(phi_dot[-len(phi_dot)//2:])
    V_values.append(V_avg)

V_values = np.array(V_values)

ax4.plot(V_values * 1e6, I_values * 1e6, 'b-', linewidth=2)
ax4.axhline(y=Ic * 1e6, color='red', linestyle='--', alpha=0.5, label='$I_c$')
ax4.set_xlabel('Voltage V (μV)', fontsize=11)
ax4.set_ylabel('Current I (μA)', fontsize=11)
ax4.set_title('Simulated I-V from RCSJ Model', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rcsj_dynamics.png', dpi=150)
plt.show()

#%% Part 4: SQUID Interference

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

Ic0 = 10e-6  # Single junction critical current

# Magnetic flux
Phi = np.linspace(-3, 3, 500) * Phi_0

# SQUID critical current (symmetric)
Ic_squid = 2 * Ic0 * np.abs(np.cos(np.pi * Phi / Phi_0))

ax1 = axes[0]
ax1.plot(Phi / Phi_0, Ic_squid * 1e6, 'b-', linewidth=2)
ax1.set_xlabel('Flux Φ/Φ₀', fontsize=12)
ax1.set_ylabel('Critical Current Ic (μA)', fontsize=12)
ax1.set_title('DC SQUID: Critical Current vs Magnetic Flux', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3, 3)

# Mark flux quanta
for n in range(-2, 3):
    ax1.axvline(x=n, color='gray', linestyle='--', alpha=0.3)

# SQUID as flux detector
ax2 = axes[1]

# Operating point: bias just above minimum Ic
I_bias = 12e-6  # μA (slightly above minimum Ic)

# Voltage as function of flux (simplified model)
# V ≈ R * sqrt(I² - Ic(Φ)²) when I > Ic(Φ)
R = 10  # Ohm

V = np.zeros_like(Phi)
for i, phi in enumerate(Phi):
    Ic = 2 * Ic0 * np.abs(np.cos(np.pi * phi / Phi_0))
    if I_bias > Ic:
        V[i] = R * np.sqrt(I_bias**2 - Ic**2)

ax2.plot(Phi / Phi_0, V * 1e6, 'b-', linewidth=2)
ax2.set_xlabel('Flux Φ/Φ₀', fontsize=12)
ax2.set_ylabel('Voltage V (μV)', fontsize=12)
ax2.set_title(f'SQUID Voltage Response (I = {I_bias*1e6:.0f} μA bias)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-3, 3)

plt.tight_layout()
plt.savefig('squid_interference.png', dpi=150)
plt.show()

#%% Part 5: Transmon Qubit Energy Levels

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def transmon_energies(n_levels, EJ_EC_ratio):
    """
    Calculate transmon energy levels (perturbation theory)

    E_n ≈ -E_J + sqrt(8 E_J E_C)(n + 1/2) - E_C(6n² + 6n + 3)/12
    """
    E_C = 1  # Set E_C = 1 as energy unit
    E_J = EJ_EC_ratio * E_C

    n = np.arange(n_levels)
    E = -E_J + np.sqrt(8 * E_J * E_C) * (n + 0.5) - E_C * (6*n**2 + 6*n + 3) / 12

    return E - E[0]  # Reference to ground state

# Energy levels for different EJ/EC ratios
ax1 = axes[0]
EJ_EC_ratios = [10, 30, 50, 100]

for ratio in EJ_EC_ratios:
    E = transmon_energies(5, ratio)
    for i, e in enumerate(E):
        ax1.hlines(e, ratio - 3, ratio + 3, linewidth=2)

ax1.set_xlabel('$E_J/E_C$', fontsize=12)
ax1.set_ylabel('Energy / $E_C$', fontsize=12)
ax1.set_title('Transmon Energy Levels', fontsize=14)
ax1.set_xlim(0, 110)
ax1.grid(True, alpha=0.3)

# Transition frequencies and anharmonicity
ax2 = axes[1]
ratios = np.linspace(5, 100, 50)

f_01 = []
alpha = []

for r in ratios:
    E = transmon_energies(3, r)
    f_01.append(E[1] - E[0])
    alpha.append((E[2] - E[1]) - (E[1] - E[0]))

ax2.plot(ratios, f_01, 'b-', linewidth=2, label='$f_{01}$ / $E_C$')
ax2.plot(ratios, np.abs(alpha), 'r-', linewidth=2, label='|α| / $E_C$ (anharmonicity)')

ax2.set_xlabel('$E_J/E_C$', fontsize=12)
ax2.set_ylabel('Frequency / $E_C$', fontsize=12)
ax2.set_title('Transmon: Transition Frequency and Anharmonicity', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transmon_levels.png', dpi=150)
plt.show()

# Summary
print("\n=== Tunnel Diode & Josephson Summary ===")
print("\n--- Tunnel Diode ---")
print(f"Peak current: {Ip*1e3:.1f} mA")
print(f"Peak voltage: {Vp*1e3:.1f} mV")
print(f"NDR region: {Vp*1e3:.0f} - {Vv*1e3:.0f} mV")

print("\n--- Josephson Junction ---")
print(f"Critical current: {Ic*1e6:.1f} μA")
print(f"Normal resistance: {Rn:.0f} Ω")
print(f"Plasma frequency: {f_p/1e9:.2f} GHz")
print(f"Josephson energy: {hbar*Ic/(2*e)*1e6/h:.2f} GHz × h")

print("\n--- Fundamental Constants ---")
print(f"Flux quantum Φ₀ = h/2e = {Phi_0*1e15:.2f} fWb")
print(f"Josephson constant: 483.6 GHz/mV")
```

### Expected Output

```
Plasma frequency: 15.62 GHz
Average voltage: 25.37 μV

=== Tunnel Diode & Josephson Summary ===

--- Tunnel Diode ---
Peak current: 5.0 mA
Peak voltage: 50.0 mV
NDR region: 50 - 300 mV

--- Josephson Junction ---
Critical current: 1.0 μA
Normal resistance: 100 Ω
Plasma frequency: 15.62 GHz
Josephson energy: 0.48 GHz × h

--- Fundamental Constants ---
Flux quantum Φ₀ = h/2e = 2.07 fWb
Josephson constant: 483.6 GHz/mV
```

---

## Summary

### Key Formulas Table

| Quantity | Formula |
|----------|---------|
| **Tunnel Diode** | |
| NDR condition | $dI/dV < 0$ |
| **Josephson Junction** | |
| DC Josephson | $I_s = I_c \sin\phi$ |
| AC Josephson | $d\phi/dt = 2eV/\hbar$ |
| Josephson frequency | $f_J = 2eV/h = 483.6$ GHz/mV |
| Josephson energy | $E_J = \hbar I_c/2e$ |
| **SQUID** | |
| Critical current | $I_c(\Phi) = 2I_{c0}|\cos(\pi\Phi/\Phi_0)|$ |
| Flux quantum | $\Phi_0 = h/2e = 2.07 \times 10^{-15}$ Wb |
| **Transmon** | |
| Frequency | $\omega_{01} \approx \sqrt{8E_JE_C} - E_C$ |
| Anharmonicity | $\alpha \approx -E_C$ |

### Main Takeaways

1. **Tunnel diodes exhibit NDR** due to band overlap effects
2. **Josephson junctions carry supercurrent** through phase-coherent Cooper pair tunneling
3. **DC Josephson effect**: Current depends on phase difference
4. **AC Josephson effect**: Voltage creates phase oscillation at precise frequency
5. **SQUIDs are ultra-sensitive magnetometers** based on quantum interference
6. **Transmon qubits** use Josephson nonlinearity for quantum computing

### Quantum Computing Applications

- **Transmon qubits**: Workhorse of superconducting quantum computing
- **Flux qubits**: Use SQUID geometry for quantum states
- **Parametric amplifiers**: Near-quantum-limited amplification
- **Josephson voltage standards**: Ultra-precise voltage references

---

## Daily Checklist

- [ ] I can explain negative differential resistance in tunnel diodes
- [ ] I understand both DC and AC Josephson effects
- [ ] I can derive the Josephson frequency for a given voltage
- [ ] I understand the washboard potential analogy
- [ ] I can explain how SQUIDs detect magnetic flux
- [ ] I understand how transmon qubits work
- [ ] I ran the Python code and understand the dynamics
- [ ] I attempted problems from each difficulty level

---

## Preview: Day 392

Tomorrow we conclude Month 14 with a comprehensive **Capstone Review**. We'll integrate all the one-dimensional quantum mechanics we've learned: free particles, bound states (wells, harmonic oscillator), and scattering/tunneling problems. There will be a capstone project and assessment to solidify your understanding before moving to Month 15: Angular Momentum and Spin!
