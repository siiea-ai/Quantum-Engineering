# Day 369: Wave Packet Dynamics

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2.5 hours | Theory: Time evolution, group velocity, dispersion |
| Afternoon | 2.5 hours | Problem solving: Spreading calculations |
| Evening | 2 hours | Computational lab: Animated wave packet evolution |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Derive the time-dependent wave packet** from the TDSE
2. **Calculate phase velocity and group velocity** from the dispersion relation
3. **Explain why group velocity equals classical velocity**
4. **Compute wave packet spreading** and its time dependence
5. **Interpret dispersion** as a consequence of non-linear ω(k)
6. **Animate wave packet propagation** to visualize spreading

---

## Core Content

### 1. Time Evolution from the TDSE

The time-dependent Schrödinger equation:

$$i\hbar\frac{\partial\Psi}{\partial t} = \hat{H}\Psi$$

For the free particle (Ĥ = p̂²/2m), each plane wave evolves as:

$$e^{ikx} \to e^{ikx}e^{-i\omega(k)t}$$

where the **angular frequency** is determined by the dispersion relation:

$$\boxed{\omega(k) = \frac{E(k)}{\hbar} = \frac{\hbar k^2}{2m}}$$

### 2. Time-Dependent Wave Packet

The general wave packet evolves as:

$$\boxed{\Psi(x,t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\phi(k)e^{i(kx - \omega(k)t)}dk}$$

Each component k travels with its own frequency ω(k). Since ω(k) is non-linear in k, different components travel at different speeds.

### 3. Phase Velocity

The **phase velocity** is the speed of individual wavefronts (constant phase surfaces):

$$\boxed{v_p = \frac{\omega}{k}}$$

For the free particle:

$$v_p = \frac{\hbar k^2/2m}{k} = \frac{\hbar k}{2m} = \frac{p}{2m}$$

$$\boxed{v_p = \frac{v_{\text{classical}}}{2}}$$

The phase velocity is **half** the classical particle velocity!

### 4. Group Velocity

The **group velocity** is the speed at which the wave packet envelope (and thus the particle) moves:

$$\boxed{v_g = \frac{d\omega}{dk}}$$

For the free particle:

$$v_g = \frac{d}{dk}\left(\frac{\hbar k^2}{2m}\right) = \frac{\hbar k}{m}$$

Since p = ℏk:

$$\boxed{v_g = \frac{p}{m} = v_{\text{classical}}}$$

**The group velocity equals the classical particle velocity!** This is a manifestation of the correspondence principle.

### 5. Classical Correspondence

Consider a wave packet with average momentum p₀ = ℏk₀:

- **Wave packet center moves at:** v_g = p₀/m
- **Classical particle with momentum p₀ moves at:** v = p₀/m

The quantum wave packet follows the classical trajectory on average!

This is related to **Ehrenfest's theorem:**

$$\frac{d\langle x \rangle}{dt} = \frac{\langle p \rangle}{m}$$

### 6. Dispersion and Spreading

**Dispersion** occurs when ω(k) is not linear in k. For the free particle:

$$\omega(k) = \frac{\hbar k^2}{2m} \quad \text{(quadratic)}$$

Different k-components travel at different phase velocities, causing the wave packet to **spread**.

The rate of spreading depends on the second derivative:

$$\boxed{\frac{d^2\omega}{dk^2} = \frac{\hbar}{m}}$$

Larger ℏ/m → faster spreading.

### 7. Gaussian Wave Packet Evolution

For an initial Gaussian:

$$\Psi(x,0) = \left(\frac{1}{2\pi\sigma_0^2}\right)^{1/4}e^{ik_0 x}e^{-x^2/4\sigma_0^2}$$

The time-evolved wave function (after integration):

$$\boxed{\Psi(x,t) = \left(\frac{1}{2\pi\sigma(t)^2}\right)^{1/4}e^{i\phi(x,t)}\exp\left[-\frac{(x - v_g t)^2}{4\sigma_0\sigma(t)}\right]}$$

where the **time-dependent width** is:

$$\boxed{\sigma(t) = \sigma_0\sqrt{1 + \frac{\hbar^2 t^2}{4m^2\sigma_0^4}}}$$

### 8. Spreading Analysis

The width grows from σ₀ at t = 0:

$$\sigma(t) = \sigma_0\sqrt{1 + \left(\frac{t}{\tau}\right)^2}$$

where the **spreading time scale** is:

$$\boxed{\tau = \frac{2m\sigma_0^2}{\hbar}}$$

**Key observations:**

- At t = 0: σ = σ₀ (initial width)
- At t = τ: σ = √2 σ₀ (width increased by √2)
- At t >> τ: σ ≈ (ℏt)/(2mσ₀) (linear growth)

### 9. Uncertainty Evolution

**Position uncertainty:**

$$\Delta x(t) = \sigma(t) = \sigma_0\sqrt{1 + \frac{t^2}{\tau^2}}$$

**Momentum uncertainty:** Stays constant!

$$\Delta p = \frac{\hbar}{2\sigma_0} = \text{constant}$$

Why? The momentum distribution |φ(k)|² doesn't change — only phases change under free evolution.

**Uncertainty product:**

$$\Delta x(t) \cdot \Delta p = \frac{\hbar}{2}\sqrt{1 + \frac{t^2}{\tau^2}} \geq \frac{\hbar}{2}$$

Minimum uncertainty is **only at t = 0**.

### 10. Physical Interpretation of Spreading

Why does the wave packet spread?

1. **Superposition of momenta:** The packet contains a range of momenta Δp
2. **Different velocities:** Each component moves at v = p/m
3. **Velocity spread:** Δv = Δp/m
4. **Position spread growth:** After time t, Δx ~ Δx₀ + Δv·t = Δx₀ + (Δp/m)t

Using Δp = ℏ/(2σ₀):

$$\Delta x(t) \sim \sigma_0 + \frac{\hbar t}{2m\sigma_0} \sim \sigma_0\sqrt{1 + \frac{t^2}{\tau^2}}$$

This matches our exact result!

---

## Quantum Computing Connection

### Matter Wave Interferometry

Wave packet dynamics underpins **atom interferometry**:
- Matter waves are split, travel different paths, and recombine
- Phase differences reveal accelerations (inertial sensing)
- Wave packet spreading limits interrogation times

### Quantum Communication Channels

In quantum communication:
- Photon wave packets spread in dispersive media (optical fibers)
- **Dispersion compensation** is needed for long-distance quantum key distribution
- Group velocity dispersion limits data rates

### Ultrafast Dynamics

Femtosecond laser pulses (wave packets) control:
- Molecular reactions (femtochemistry)
- Coherent dynamics in qubits
- Entanglement generation timing

---

## Worked Examples

### Example 1: Spreading Time for an Electron

**Problem:** An electron is localized to 1 nm at t = 0. Find (a) the spreading time τ, and (b) the width after 1 femtosecond.

**Solution:**

Given: σ₀ = 1 nm = 10⁻⁹ m, m = 9.11 × 10⁻³¹ kg, ℏ = 1.055 × 10⁻³⁴ J·s

(a) Spreading time:
$$\tau = \frac{2m\sigma_0^2}{\hbar} = \frac{2 \times 9.11 \times 10^{-31} \times (10^{-9})^2}{1.055 \times 10^{-34}}$$

$$\tau = \frac{1.822 \times 10^{-48}}{1.055 \times 10^{-34}} = 1.73 \times 10^{-14} \text{ s} = 17.3 \text{ fs}$$

$$\boxed{\tau = 17.3 \text{ fs}}$$

(b) Width after t = 1 fs = 10⁻¹⁵ s:
$$\sigma(t) = \sigma_0\sqrt{1 + \frac{t^2}{\tau^2}} = 1\text{ nm}\sqrt{1 + \frac{(1)^2}{(17.3)^2}}$$

$$\sigma(1\text{ fs}) = 1\text{ nm} \times \sqrt{1.003} \approx 1.002 \text{ nm}$$

$$\boxed{\sigma(1\text{ fs}) \approx 1.00 \text{ nm}}$$

The packet barely spreads in 1 fs because τ >> 1 fs.

### Example 2: Group vs Phase Velocity

**Problem:** A free electron has momentum p = 10⁻²⁴ kg·m/s. Calculate the phase and group velocities. Compare to the classical velocity.

**Solution:**

Classical velocity:
$$v_{\text{classical}} = \frac{p}{m} = \frac{10^{-24}}{9.11 \times 10^{-31}} = 1.10 \times 10^6 \text{ m/s}$$

Group velocity:
$$v_g = \frac{p}{m} = v_{\text{classical}} = 1.10 \times 10^6 \text{ m/s}$$

Phase velocity:
$$v_p = \frac{p}{2m} = \frac{v_{\text{classical}}}{2} = 5.5 \times 10^5 \text{ m/s}$$

$$\boxed{v_g = 1.10 \times 10^6 \text{ m/s}, \quad v_p = 5.5 \times 10^5 \text{ m/s}}$$

Note: v_g > v_p. The wave crests move slower than the envelope!

### Example 3: Long-Time Spreading

**Problem:** After time t >> τ, show that the wave packet width grows as σ(t) ≈ ℏt/(2mσ₀).

**Solution:**

Starting from:
$$\sigma(t) = \sigma_0\sqrt{1 + \frac{\hbar^2 t^2}{4m^2\sigma_0^4}}$$

For t >> τ = 2mσ₀²/ℏ, we have t >> 2mσ₀²/ℏ, so:

$$\frac{\hbar^2 t^2}{4m^2\sigma_0^4} >> 1$$

Therefore:
$$\sigma(t) \approx \sigma_0 \cdot \frac{\hbar t}{2m\sigma_0^2} = \frac{\hbar t}{2m\sigma_0}$$

$$\boxed{\sigma(t) \approx \frac{\hbar t}{2m\sigma_0} \quad \text{for } t >> \tau}$$

**Physical interpretation:** At long times, the width grows linearly with time. The rate of growth is ℏ/(2mσ₀), which equals Δp/m = the velocity uncertainty.

---

## Practice Problems

### Level 1: Direct Application

1. **Group velocity:** A wave packet has central wave number k₀ = 5 nm⁻¹ for a particle of mass m = 10⁻²⁶ kg. Find v_g.

2. **Phase velocity ratio:** Show that v_p/v_g = 1/2 for a free particle.

3. **Spreading time:** Calculate τ for a proton localized to 0.1 nm.

### Level 2: Intermediate

4. **Photon dispersion:** For a massless particle (like a photon in vacuum), ω = ck. Show there is no dispersion (v_g = v_p = c).

5. **Width doubling time:** Find the time for a Gaussian wave packet to double its width (σ(t) = 2σ₀).

6. **Relativistic correction:** For a relativistic particle, E = √((pc)² + (mc²)²). Find v_g and v_p and show v_g v_p = c².

### Level 3: Challenging

7. **Exact Gaussian evolution:** Starting from the integral
   $$\Psi(x,t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\phi(k)e^{i(kx-\omega t)}dk$$
   with φ(k) Gaussian, perform the integral to derive σ(t).

8. **Energy-time uncertainty:** From the spreading formula, interpret the relation ΔE·Δt ~ ℏ where ΔE is the energy spread and Δt ~ τ.

9. **Minimal spreading packet:** Find the initial width σ₀ that minimizes σ(T) at a fixed future time T. Show this requires σ₀² = ℏT/(2m).

---

## Computational Lab: Animated Wave Packet Evolution

```python
"""
Day 369 Computational Lab: Wave Packet Dynamics
================================================
Animating free particle wave packet evolution and spreading
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.integrate import simps
from IPython.display import HTML

# Physical constants (using atomic units: ℏ = m = 1)
hbar = 1.0
m = 1.0

# =============================================================================
# Part 1: Gaussian Wave Packet Evolution
# =============================================================================

def sigma_t(sigma0, t, m=1.0, hbar=1.0):
    """Time-dependent width of Gaussian wave packet."""
    tau = 2 * m * sigma0**2 / hbar
    return sigma0 * np.sqrt(1 + (t/tau)**2)

def gaussian_evolved(x, t, sigma0, k0, m=1.0, hbar=1.0):
    """
    Time-evolved Gaussian wave packet (analytical formula).

    Uses the exact solution for free particle Gaussian evolution.
    """
    v_g = hbar * k0 / m  # Group velocity
    tau = 2 * m * sigma0**2 / hbar
    alpha = 1 + 1j * t / tau

    # Normalization factor
    norm = (2 * np.pi * sigma0**2)**(-0.25) * (1/alpha)**0.5

    # Position relative to moving center
    x_rel = x - v_g * t

    # Wave function
    psi = norm * np.exp(-(x_rel)**2 / (4 * sigma0**2 * alpha)) * np.exp(1j * k0 * x)

    # Additional phase
    phase = -hbar * k0**2 * t / (2 * m)
    psi *= np.exp(1j * phase)

    return psi

# Set up grid
x = np.linspace(-30, 50, 2000)
dx = x[1] - x[0]

# Wave packet parameters
sigma0 = 2.0   # Initial width
k0 = 3.0       # Central wave number

# Calculate key quantities
tau = 2 * m * sigma0**2 / hbar
v_g = hbar * k0 / m
v_p = hbar * k0 / (2 * m)

print("="*60)
print("WAVE PACKET PARAMETERS")
print("="*60)
print(f"Initial width σ₀ = {sigma0}")
print(f"Central wave number k₀ = {k0}")
print(f"Spreading time τ = {tau:.2f}")
print(f"Group velocity v_g = {v_g:.2f}")
print(f"Phase velocity v_p = {v_p:.2f}")
print(f"Ratio v_g/v_p = {v_g/v_p:.2f}")

# Plot evolution at several times
times = [0, tau/2, tau, 2*tau, 4*tau]

fig, axes = plt.subplots(2, len(times), figsize=(18, 8))

for i, t in enumerate(times):
    psi = gaussian_evolved(x, t, sigma0, k0)
    width = sigma_t(sigma0, t)

    # Probability density
    ax = axes[0, i]
    prob = np.abs(psi)**2
    ax.fill_between(x, 0, prob, alpha=0.4)
    ax.plot(x, prob, 'b-', linewidth=2)

    # Mark center and width
    center = v_g * t
    ax.axvline(center, color='r', linestyle='--', alpha=0.7)
    ax.axvline(center - width, color='g', linestyle=':', alpha=0.7)
    ax.axvline(center + width, color='g', linestyle=':', alpha=0.7)

    ax.set_xlim(-10, 50)
    ax.set_xlabel('x')
    ax.set_title(f't = {t:.1f} ({t/tau:.1f}τ)\nσ = {width:.2f}')
    if i == 0:
        ax.set_ylabel('|Ψ(x,t)|²')

    # Real part of wave function
    ax = axes[1, i]
    ax.plot(x, np.real(psi), 'b-', linewidth=1, label='Re(Ψ)')
    ax.plot(x, np.abs(psi), 'k--', linewidth=1.5, label='|Ψ|')
    ax.set_xlim(-10, 50)
    ax.set_xlabel('x')
    if i == 0:
        ax.set_ylabel('Ψ(x,t)')
        ax.legend()

plt.suptitle('Gaussian Wave Packet Evolution: Spreading and Translation', fontsize=14)
plt.tight_layout()
plt.savefig('wavepacket_evolution.png', dpi=150)
plt.show()

# =============================================================================
# Part 2: Width vs Time
# =============================================================================

t_array = np.linspace(0, 5*tau, 500)
width_array = sigma_t(sigma0, t_array)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Width vs time
ax = axes[0]
ax.plot(t_array/tau, width_array/sigma0, 'b-', linewidth=2)
ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='σ₀')
ax.axhline(y=np.sqrt(2), color='g', linestyle='--', alpha=0.5, label='√2 σ₀')

# Asymptotic behavior
asymptotic = t_array / tau
ax.plot(t_array/tau, asymptotic, 'k:', linewidth=1.5, label='Linear (t >> τ)')

ax.set_xlabel('t / τ')
ax.set_ylabel('σ(t) / σ₀')
ax.set_title('Wave Packet Width vs Time')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)

# Uncertainty product
delta_x = width_array
delta_p = hbar / (2 * sigma0) * np.ones_like(t_array)
product = delta_x * delta_p

ax = axes[1]
ax.plot(t_array/tau, product / (hbar/2), 'b-', linewidth=2)
ax.axhline(y=1, color='r', linestyle='--', label='Minimum (ℏ/2)')
ax.set_xlabel('t / τ')
ax.set_ylabel('Δx·Δp / (ℏ/2)')
ax.set_title('Uncertainty Product vs Time')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)

plt.tight_layout()
plt.savefig('spreading_analysis.png', dpi=150)
plt.show()

# =============================================================================
# Part 3: Animation of Wave Packet
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

line_prob, = ax.plot([], [], 'b-', linewidth=2, label='|Ψ|²')
line_real, = ax.plot([], [], 'r-', linewidth=1, alpha=0.5, label='Re(Ψ)')
vline_center = ax.axvline(0, color='k', linestyle='--', alpha=0.7)
fill = ax.fill_between([], [], [], alpha=0.3)

ax.set_xlim(-10, 60)
ax.set_ylim(-0.3, 0.35)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Ψ(x,t)', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
width_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=12)

def init():
    line_prob.set_data([], [])
    line_real.set_data([], [])
    time_text.set_text('')
    width_text.set_text('')
    return line_prob, line_real, time_text, width_text

def animate(frame):
    t = frame * 0.1
    psi = gaussian_evolved(x, t, sigma0, k0)
    prob = np.abs(psi)**2 * 5  # Scale for visibility
    width = sigma_t(sigma0, t)
    center = v_g * t

    line_prob.set_data(x, prob)
    line_real.set_data(x, np.real(psi))
    vline_center.set_xdata([center])
    time_text.set_text(f't = {t:.1f} ({t/tau:.2f}τ)')
    width_text.set_text(f'σ = {width:.2f} (ratio: {width/sigma0:.2f})')

    return line_prob, line_real, time_text, width_text

anim = FuncAnimation(fig, animate, init_func=init, frames=200,
                     interval=50, blit=True)

# To save: anim.save('wavepacket_animation.mp4', writer='ffmpeg', fps=20)
plt.show()

# =============================================================================
# Part 4: FFT-Based Evolution
# =============================================================================

def evolve_fft(psi0, x, t, m=1.0, hbar=1.0):
    """
    Evolve wave function using FFT.

    This is the numerical approach for arbitrary initial conditions.
    """
    N = len(x)
    dx = x[1] - x[0]

    # k-space grid
    k = fftfreq(N, dx) * 2 * np.pi

    # Fourier transform
    psi_k = fft(psi0)

    # Apply time evolution operator in k-space
    omega = hbar * k**2 / (2 * m)
    psi_k_evolved = psi_k * np.exp(-1j * omega * t)

    # Inverse Fourier transform
    psi_t = ifft(psi_k_evolved)

    return psi_t

# Initial Gaussian
psi0 = gaussian_evolved(x, 0, sigma0, k0)

# Compare analytical and FFT evolution
t_test = 2 * tau

psi_analytical = gaussian_evolved(x, t_test, sigma0, k0)
psi_fft = evolve_fft(psi0, x, t_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(x, np.abs(psi_analytical)**2, 'b-', linewidth=2, label='Analytical')
ax.plot(x, np.abs(psi_fft)**2, 'r--', linewidth=2, label='FFT')
ax.set_xlim(-5, 40)
ax.set_xlabel('x')
ax.set_ylabel('|Ψ|²')
ax.set_title(f'Probability Density at t = {t_test:.1f}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x, np.real(psi_analytical), 'b-', linewidth=2, label='Analytical')
ax.plot(x, np.real(psi_fft), 'r--', linewidth=2, label='FFT')
ax.set_xlim(-5, 40)
ax.set_xlabel('x')
ax.set_ylabel('Re(Ψ)')
ax.set_title('Real Part Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fft_vs_analytical.png', dpi=150)
plt.show()

# =============================================================================
# Part 5: Non-Gaussian Initial Condition
# =============================================================================

# Rectangular initial condition (localized, not Gaussian)
def rectangular_packet(x, width, k0):
    psi = np.where(np.abs(x) < width/2, 1.0, 0.0) * np.exp(1j * k0 * x)
    norm = np.sqrt(simps(np.abs(psi)**2, x))
    return psi / norm

# Double Gaussian (cat-like state)
def double_gaussian(x, sigma, d, k0):
    psi1 = gaussian_evolved(x, 0, sigma, k0, m, hbar) * np.exp(1j * k0 * d/2)
    psi2 = gaussian_evolved(x, 0, sigma, k0, m, hbar) * np.exp(-1j * k0 * d/2)
    psi = psi1 * np.exp(-d/2) + psi2 * np.exp(d/2)
    # Shift centers
    psi = np.roll(psi1, -int(d/(2*dx))) + np.roll(psi2, int(d/(2*dx)))
    norm = np.sqrt(simps(np.abs(psi)**2, x))
    return psi / norm

# Create non-Gaussian packets
psi_rect = rectangular_packet(x, 4.0, k0)
psi_double = double_gaussian(x, 1.5, 6.0, k0)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

initial_conditions = [
    ('Gaussian', psi0),
    ('Rectangular', psi_rect),
    ('Double Gaussian', psi_double)
]

times_show = [0, tau, 3*tau]

for i, (name, psi_init) in enumerate(initial_conditions):
    for j, t in enumerate(times_show):
        psi_t = evolve_fft(psi_init, x, t)

        ax = axes[i, j]
        ax.fill_between(x, 0, np.abs(psi_t)**2, alpha=0.4)
        ax.plot(x, np.abs(psi_t)**2, 'b-', linewidth=2)
        ax.set_xlim(-10, 50)
        ax.set_xlabel('x')

        if j == 0:
            ax.set_ylabel(f'{name}\n|Ψ|²')
        if i == 0:
            ax.set_title(f't = {t:.1f} = {t/tau:.1f}τ')

plt.suptitle('Different Initial Conditions: All Spread Due to Dispersion', fontsize=14)
plt.tight_layout()
plt.savefig('different_initials.png', dpi=150)
plt.show()

# =============================================================================
# Part 6: Phase and Group Velocity Visualization
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

# Create a narrow-bandwidth Gaussian (so we can see individual crests)
sigma_narrow = 4.0
k0_vis = 4.0

t_values = np.linspace(0, 3, 4)

for t in t_values:
    psi = gaussian_evolved(x, t, sigma_narrow, k0_vis)
    # Shift vertically for visualization
    offset = t * 0.4
    ax.plot(x, np.real(psi) + offset, linewidth=1.5, label=f't = {t:.1f}')

    # Mark a specific crest (phase velocity tracking)
    # Phase moves as kx - ωt = const, so x_phase = (ω/k)t = v_p t
    v_p_vis = hbar * k0_vis / (2 * m)
    x_crest = v_p_vis * t
    ax.plot(x_crest, offset, 'ko', markersize=8)

    # Mark envelope center (group velocity)
    v_g_vis = hbar * k0_vis / m
    x_center = v_g_vis * t
    ax.plot(x_center, offset, 'rs', markersize=10)

# Add velocity lines
t_line = np.linspace(0, 3, 100)
ax.plot(v_p_vis * t_line, t_line * 0.4, 'k--', linewidth=2, label=f'Phase velocity v_p = {v_p_vis:.1f}')
ax.plot(v_g_vis * t_line, t_line * 0.4, 'r--', linewidth=2, label=f'Group velocity v_g = {v_g_vis:.1f}')

ax.set_xlim(-5, 20)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Ψ(x,t) + offset', fontsize=12)
ax.set_title('Phase Velocity (●) vs Group Velocity (■)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase_vs_group.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS FROM THIS LAB")
print("="*60)
print("""
1. Wave packets MOVE at the group velocity v_g = ℏk₀/m = classical velocity.

2. Individual crests move at phase velocity v_p = v_g/2 (half as fast).

3. Wave packets SPREAD due to dispersion (ω ∝ k² is non-linear).

4. Spreading time τ = 2mσ₀²/ℏ sets the time scale.

5. Momentum distribution is PRESERVED (only phases change).

6. Uncertainty product Δx·Δp increases from ℏ/2 over time.

7. FFT provides efficient numerical evolution for any initial state.
""")
```

---

## Summary

### Key Formulas Table

| Quantity | Formula | Notes |
|----------|---------|-------|
| Wave packet evolution | $$\Psi(x,t) = \frac{1}{\sqrt{2\pi}}\int\phi(k)e^{i(kx-\omega t)}dk$$ | General form |
| Dispersion relation | $$\omega = \frac{\hbar k^2}{2m}$$ | Free particle |
| Phase velocity | $$v_p = \frac{\omega}{k} = \frac{\hbar k}{2m}$$ | Speed of crests |
| Group velocity | $$v_g = \frac{d\omega}{dk} = \frac{\hbar k}{m} = v_{\text{classical}}$$ | Speed of envelope |
| Width evolution | $$\sigma(t) = \sigma_0\sqrt{1 + t^2/\tau^2}$$ | Spreading |
| Spreading time | $$\tau = \frac{2m\sigma_0^2}{\hbar}$$ | Characteristic time |

### Main Takeaways

1. **Group velocity = classical velocity:** v_g = p/m (correspondence principle)
2. **Phase velocity = half group velocity:** v_p = v_g/2 for free particle
3. **Dispersion causes spreading:** Non-linear ω(k) → different k's travel differently
4. **Spreading time scale:** τ = 2mσ₀²/ℏ (smaller for light particles, narrow packets)
5. **Momentum preserved:** |φ(k)|² unchanged; only phases evolve
6. **Uncertainty grows:** Δx·Δp > ℏ/2 for t > 0

---

## Daily Checklist

- [ ] I understand the difference between phase and group velocity
- [ ] I can derive v_g = dω/dk from the wave packet integral
- [ ] I can calculate the spreading time τ
- [ ] I understand why spreading occurs (dispersion)
- [ ] I can connect group velocity to classical mechanics
- [ ] I successfully ran the computational lab
- [ ] I completed at least 4 practice problems

---

## Preview: Day 370

Tomorrow we complete the wave packet story with a deep dive into **position and momentum space representations**:
- Operators in both representations
- Completeness and resolution of identity
- Practical calculations using both pictures
- Connection to Dirac notation

---

*"The group velocity tells us where the particle goes; the phase velocity is the ghost that moves through the wave crests. Quantum mechanics beautifully resolves this apparent paradox."*
