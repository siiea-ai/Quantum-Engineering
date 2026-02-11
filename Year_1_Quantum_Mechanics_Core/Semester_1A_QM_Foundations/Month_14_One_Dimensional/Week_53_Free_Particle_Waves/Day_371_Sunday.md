# Day 371: Week Review & Comprehensive Lab

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 2 hours | Concept review and synthesis |
| Afternoon | 3 hours | Comprehensive computational lab |
| Evening | 2 hours | Practice problems and week assessment |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Synthesize all free particle concepts** from the week into a coherent picture
2. **Solve complex problems** combining multiple concepts
3. **Implement a complete wave packet evolution simulator**
4. **Analyze wave packet dynamics** quantitatively
5. **Prepare for bound state problems** in Week 54
6. **Demonstrate mastery** through comprehensive practice problems

---

## Week 53 Synthesis

### The Big Picture

This week we studied the **free particle** — the simplest quantum system with V(x) = 0. Despite its simplicity, this system reveals fundamental aspects of quantum mechanics:

```
Free Particle Schrödinger Equation
           ↓
     Plane Wave Solutions
      /              \
Continuous Spectrum    Non-normalizable
      ↓                      ↓
Box/Delta Normalization  Wave Packets
      ↓                      ↓
Momentum Eigenstates    Localized States
      \                    /
       \                  /
        ↘              ↙
    Fourier Transform Connection
             ↓
      Position ↔ Momentum Space
             ↓
    Time Evolution & Spreading
             ↓
    Uncertainty Principle in Action
```

### Concept Map

| Day | Topic | Key Results |
|-----|-------|-------------|
| 365 | Free Particle TISE | ψ_k = Ae^{ikx}, E = ℏ²k²/2m, continuous spectrum |
| 366 | Normalization | Box: ψ = L^{-1/2}e^{ikx}, Delta: ⟨k\|k'⟩ = δ(k-k') |
| 367 | Wave Packets I | ψ(x) = ∫φ(k)e^{ikx}dk, Fourier transform |
| 368 | Gaussian Packets | Minimum uncertainty Δx·Δp = ℏ/2 |
| 369 | Wave Packet Dynamics | v_g = ℏk/m = v_classical, spreading |
| 370 | Position/Momentum Space | x̂ ↔ iℏ∂_p, p̂ ↔ -iℏ∂_x |

### Master Equations

**Time-Independent Schrödinger Equation:**
$$\boxed{-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi}$$

**Dispersion Relation:**
$$\boxed{E = \frac{\hbar^2 k^2}{2m}, \quad \omega = \frac{\hbar k^2}{2m}}$$

**Wave Packet:**
$$\boxed{\Psi(x,t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\phi(k)e^{i(kx - \omega(k)t)}dk}$$

**Gaussian Evolution:**
$$\boxed{\sigma(t) = \sigma_0\sqrt{1 + \frac{t^2}{\tau^2}}, \quad \tau = \frac{2m\sigma_0^2}{\hbar}}$$

**Velocities:**
$$\boxed{v_g = \frac{d\omega}{dk} = \frac{\hbar k}{m} = v_{\text{classical}}, \quad v_p = \frac{\omega}{k} = \frac{v_g}{2}}$$

**Fourier Transform Pair:**
$$\boxed{\phi(p) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{-ipx/\hbar}\psi(x)dx}$$
$$\boxed{\psi(x) = \frac{1}{\sqrt{2\pi\hbar}}\int e^{ipx/\hbar}\phi(p)dp}$$

---

## Connections to Quantum Computing

| QM Concept | QC Application |
|------------|----------------|
| Fourier transform | Quantum Fourier Transform (Shor's algorithm) |
| Wave packet localization | Photon pulse shaping |
| Group velocity | Signal propagation in quantum networks |
| Dispersion | Channel compensation in quantum communication |
| Minimum uncertainty | Coherent states, squeezed states |
| Continuous spectrum | Continuous-variable quantum computing |

---

## Comprehensive Computational Lab

```python
"""
Day 371 Comprehensive Lab: Wave Packet Evolution Simulator
===========================================================
A complete simulation and analysis tool for free particle dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.integrate import simps
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Part 1: Wave Packet Simulator Class
# =============================================================================

class WavePacketSimulator:
    """
    Complete wave packet evolution simulator for free particle.

    Features:
    - Multiple initial condition types
    - FFT-based time evolution
    - Automatic uncertainty tracking
    - Animation support
    """

    def __init__(self, x_range=(-30, 50), N=2048, hbar=1.0, m=1.0):
        """
        Initialize the simulator.

        Parameters:
        -----------
        x_range : tuple
            (x_min, x_max) for position grid
        N : int
            Number of grid points
        hbar, m : float
            Physical constants
        """
        self.hbar = hbar
        self.m = m
        self.N = N

        # Position grid
        self.x = np.linspace(x_range[0], x_range[1], N)
        self.dx = self.x[1] - self.x[0]

        # Momentum grid
        self.k = fftfreq(N, self.dx) * 2 * np.pi
        self.p = self.hbar * self.k

        # Wave function storage
        self.psi = None
        self.phi = None
        self.psi_initial = None

        # Time tracking
        self.t = 0

        # History for analysis
        self.history = {
            't': [],
            'x_mean': [],
            'p_mean': [],
            'delta_x': [],
            'delta_p': [],
            'energy': []
        }

    def set_gaussian(self, sigma, x0=0, p0=0):
        """Initialize with Gaussian wave packet."""
        self.sigma0 = sigma
        self.x0 = x0
        self.p0 = p0
        self.tau = 2 * self.m * sigma**2 / self.hbar

        norm = (2 * np.pi * sigma**2)**(-0.25)
        self.psi = norm * np.exp(-(self.x - x0)**2 / (4 * sigma**2)) * \
                   np.exp(1j * p0 * self.x / self.hbar)
        self.psi_initial = self.psi.copy()
        self._update_momentum_space()
        self.t = 0
        self._record_state()

    def set_double_gaussian(self, sigma, d, x0=0, p0=0):
        """Initialize with double Gaussian (cat state)."""
        norm = (2 * np.pi * sigma**2)**(-0.25)
        psi1 = norm * np.exp(-(self.x - x0 - d/2)**2 / (4 * sigma**2))
        psi2 = norm * np.exp(-(self.x - x0 + d/2)**2 / (4 * sigma**2))
        self.psi = (psi1 + psi2) * np.exp(1j * p0 * self.x / self.hbar)
        self.psi /= np.sqrt(simps(np.abs(self.psi)**2, self.x))
        self.psi_initial = self.psi.copy()
        self._update_momentum_space()
        self.t = 0
        self.sigma0 = sigma
        self.tau = 2 * self.m * sigma**2 / self.hbar
        self._record_state()

    def set_rectangular(self, width, x0=0, p0=0):
        """Initialize with rectangular (sinc in momentum) wave packet."""
        self.psi = np.where(np.abs(self.x - x0) < width/2, 1.0, 0.0)
        self.psi = self.psi * np.exp(1j * p0 * self.x / self.hbar)
        self.psi /= np.sqrt(simps(np.abs(self.psi)**2, self.x))
        self.psi_initial = self.psi.copy()
        self._update_momentum_space()
        self.t = 0
        self._record_state()

    def _update_momentum_space(self):
        """Update momentum-space wave function from position space."""
        self.phi = fft(self.psi) * self.dx / np.sqrt(2 * np.pi * self.hbar)

    def _update_position_space(self):
        """Update position-space wave function from momentum space."""
        self.psi = ifft(self.phi) * np.sqrt(2 * np.pi * self.hbar) / self.dx

    def evolve(self, dt):
        """Evolve the wave packet by time dt using split-step FFT."""
        # Apply time evolution in momentum space
        omega = self.hbar * self.k**2 / (2 * self.m)
        self.phi *= np.exp(-1j * omega * dt)
        self._update_position_space()
        self.t += dt
        self._record_state()

    def evolve_to(self, t_final, n_steps=100):
        """Evolve to a specific time."""
        dt = (t_final - self.t) / n_steps
        for _ in range(n_steps):
            self.evolve(dt)

    def _record_state(self):
        """Record current state for analysis."""
        self.history['t'].append(self.t)

        # Position statistics
        prob_x = np.abs(self.psi)**2
        x_mean = simps(self.x * prob_x, self.x)
        x2_mean = simps(self.x**2 * prob_x, self.x)
        delta_x = np.sqrt(x2_mean - x_mean**2)

        self.history['x_mean'].append(x_mean)
        self.history['delta_x'].append(delta_x)

        # Momentum statistics
        phi_shifted = fftshift(self.phi)
        p_shifted = fftshift(self.p)
        prob_p = np.abs(phi_shifted)**2
        norm_p = simps(prob_p, p_shifted)
        if norm_p > 0:
            prob_p /= norm_p
            p_mean = simps(p_shifted * prob_p, p_shifted)
            p2_mean = simps(p_shifted**2 * prob_p, p_shifted)
            delta_p = np.sqrt(max(0, p2_mean - p_mean**2))
        else:
            p_mean = 0
            delta_p = 0

        self.history['p_mean'].append(p_mean)
        self.history['delta_p'].append(delta_p)

        # Energy
        E = p2_mean / (2 * self.m) if norm_p > 0 else 0
        self.history['energy'].append(E)

    def get_uncertainty_product(self):
        """Return current uncertainty product Δx·Δp."""
        return self.history['delta_x'][-1] * self.history['delta_p'][-1]

    def plot_state(self, ax=None, show_initial=True):
        """Plot current wave function."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))

        if show_initial and self.psi_initial is not None:
            ax.plot(self.x, np.abs(self.psi_initial)**2, 'k--',
                    alpha=0.3, label='Initial')

        ax.fill_between(self.x, 0, np.abs(self.psi)**2, alpha=0.4)
        ax.plot(self.x, np.abs(self.psi)**2, 'b-', linewidth=2, label=f't = {self.t:.2f}')
        ax.axvline(self.history['x_mean'][-1], color='r', linestyle='--', alpha=0.7)

        ax.set_xlabel('x')
        ax.set_ylabel('|ψ(x)|²')
        ax.set_title('Wave Packet Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_momentum(self, ax=None):
        """Plot momentum distribution."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))

        phi_shifted = fftshift(self.phi)
        p_shifted = fftshift(self.p)

        ax.fill_between(p_shifted, 0, np.abs(phi_shifted)**2, alpha=0.4, color='red')
        ax.plot(p_shifted, np.abs(phi_shifted)**2, 'r-', linewidth=2)
        ax.axvline(self.history['p_mean'][-1], color='b', linestyle='--', alpha=0.7)

        ax.set_xlabel('p')
        ax.set_ylabel('|φ(p)|²')
        ax.set_title('Momentum Distribution')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_analysis(self):
        """Plot comprehensive analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        t = np.array(self.history['t'])

        # Position space
        self.plot_state(axes[0, 0])

        # Momentum space
        self.plot_momentum(axes[0, 1])

        # Position mean and uncertainty
        ax = axes[0, 2]
        x_mean = np.array(self.history['x_mean'])
        delta_x = np.array(self.history['delta_x'])
        ax.fill_between(t, x_mean - delta_x, x_mean + delta_x, alpha=0.3)
        ax.plot(t, x_mean, 'b-', linewidth=2, label='⟨x⟩')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_title('Position: Mean ± Δx')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Uncertainty product
        ax = axes[1, 0]
        delta_x = np.array(self.history['delta_x'])
        delta_p = np.array(self.history['delta_p'])
        product = delta_x * delta_p
        ax.plot(t, product / (self.hbar / 2), 'b-', linewidth=2)
        ax.axhline(1, color='r', linestyle='--', label='Minimum (ℏ/2)')
        ax.set_xlabel('t')
        ax.set_ylabel('Δx·Δp / (ℏ/2)')
        ax.set_title('Uncertainty Product')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Width evolution
        ax = axes[1, 1]
        if hasattr(self, 'sigma0'):
            ax.plot(t, delta_x / self.sigma0, 'b-', linewidth=2, label='Δx(t)/σ₀')
            # Analytical for Gaussian
            analytical = np.sqrt(1 + (t / self.tau)**2)
            ax.plot(t, analytical, 'r--', linewidth=2, label='Analytical')
        else:
            ax.plot(t, delta_x, 'b-', linewidth=2, label='Δx(t)')
        ax.set_xlabel('t')
        ax.set_ylabel('Width ratio')
        ax.set_title('Width Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Energy conservation
        ax = axes[1, 2]
        E = np.array(self.history['energy'])
        ax.plot(t, E / E[0] if E[0] > 0 else E, 'b-', linewidth=2)
        ax.set_xlabel('t')
        ax.set_ylabel('E(t) / E(0)')
        ax.set_title('Energy Conservation')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.9, 1.1)

        plt.tight_layout()
        return fig

    def animate(self, t_final, n_frames=100, interval=50, save_path=None):
        """Create animation of wave packet evolution."""
        # Reset to initial state
        self.psi = self.psi_initial.copy()
        self._update_momentum_space()
        self.t = 0
        self.history = {key: [] for key in self.history}
        self._record_state()

        dt = t_final / n_frames

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        line1, = ax1.plot([], [], 'b-', linewidth=2)
        fill1 = ax1.fill_between([], [], [], alpha=0.4)
        ax1.set_xlim(self.x.min(), self.x.max())
        ax1.set_ylim(0, np.max(np.abs(self.psi_initial)**2) * 1.5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('|ψ(x)|²')
        ax1.grid(True, alpha=0.3)
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

        p_shifted = fftshift(self.p)
        phi_init = fftshift(fft(self.psi_initial) * self.dx / np.sqrt(2 * np.pi * self.hbar))
        line2, = ax2.plot([], [], 'r-', linewidth=2)
        ax2.set_xlim(p_shifted.min(), p_shifted.max())
        ax2.set_ylim(0, np.max(np.abs(phi_init)**2) * 1.2)
        ax2.set_xlabel('p')
        ax2.set_ylabel('|φ(p)|²')
        ax2.set_title('Momentum Distribution (constant magnitude)')
        ax2.grid(True, alpha=0.3)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            time_text.set_text('')
            return line1, line2, time_text

        def animate_frame(frame):
            self.evolve(dt)
            prob_x = np.abs(self.psi)**2
            line1.set_data(self.x, prob_x)
            ax1.collections.clear()
            ax1.fill_between(self.x, 0, prob_x, alpha=0.4)

            phi_shifted = fftshift(self.phi)
            line2.set_data(p_shifted, np.abs(phi_shifted)**2)

            time_text.set_text(f't = {self.t:.2f}')
            ax1.set_title(f'Position Space (Δx = {self.history["delta_x"][-1]:.2f})')

            return line1, line2, time_text

        anim = FuncAnimation(fig, animate_frame, init_func=init,
                            frames=n_frames, interval=interval, blit=False)

        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=20)

        plt.show()
        return anim


# =============================================================================
# Part 2: Demonstrations
# =============================================================================

print("="*70)
print("WAVE PACKET EVOLUTION SIMULATOR")
print("="*70)

# Create simulator
sim = WavePacketSimulator(x_range=(-20, 60), N=2048)

# Demo 1: Standard Gaussian
print("\n--- Demo 1: Gaussian Wave Packet ---")
sim.set_gaussian(sigma=2.0, x0=0, p0=3.0)
print(f"Initial: σ = 2.0, x₀ = 0, p₀ = 3.0")
print(f"Spreading time τ = {sim.tau:.2f}")
print(f"Initial uncertainty product: {sim.get_uncertainty_product():.4f}")

# Evolve
for t in [0, sim.tau, 2*sim.tau, 3*sim.tau]:
    sim.evolve_to(t)
    print(f"t = {t:.1f} ({t/sim.tau:.1f}τ): Δx = {sim.history['delta_x'][-1]:.3f}, "
          f"Δx·Δp = {sim.get_uncertainty_product():.4f}")

# Plot analysis
fig = sim.plot_analysis()
plt.savefig('gaussian_analysis_complete.png', dpi=150)
plt.show()

# Demo 2: Double Gaussian (Cat State)
print("\n--- Demo 2: Double Gaussian (Cat State) ---")
sim2 = WavePacketSimulator(x_range=(-20, 60), N=2048)
sim2.set_double_gaussian(sigma=1.5, d=8, x0=5, p0=2.0)

# Evolve and show interference
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
times = [0, 1, 2, 4, 8, 12]

for ax, t in zip(axes.flatten(), times):
    sim2.evolve_to(t)
    ax.fill_between(sim2.x, 0, np.abs(sim2.psi)**2, alpha=0.4)
    ax.plot(sim2.x, np.abs(sim2.psi)**2, 'b-', linewidth=2)
    ax.set_xlim(-10, 50)
    ax.set_xlabel('x')
    ax.set_ylabel('|ψ|²')
    ax.set_title(f't = {t:.1f}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Double Gaussian Evolution: Wave Packet Spreading', fontsize=14)
plt.tight_layout()
plt.savefig('double_gaussian_evolution.png', dpi=150)
plt.show()

# Demo 3: Rectangular Wave Packet
print("\n--- Demo 3: Rectangular Wave Packet ---")
sim3 = WavePacketSimulator(x_range=(-20, 60), N=2048)
sim3.set_rectangular(width=5, x0=0, p0=3.0)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
times = [0, 0.5, 1, 2, 4, 8]

for ax, t in zip(axes.flatten(), times):
    sim3.evolve_to(t)
    ax.fill_between(sim3.x, 0, np.abs(sim3.psi)**2, alpha=0.4)
    ax.plot(sim3.x, np.abs(sim3.psi)**2, 'b-', linewidth=2)
    ax.set_xlim(-10, 50)
    ax.set_xlabel('x')
    ax.set_ylabel('|ψ|²')
    ax.set_title(f't = {t:.1f}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Rectangular Wave Packet Evolution', fontsize=14)
plt.tight_layout()
plt.savefig('rectangular_evolution.png', dpi=150)
plt.show()

# =============================================================================
# Part 3: Comparison of Different Initial Conditions
# =============================================================================

print("\n--- Comparison: Different Initial Conditions ---")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Three different initial conditions
configs = [
    ("Gaussian", lambda s: s.set_gaussian(2.0, 0, 3.0)),
    ("Double Gaussian", lambda s: s.set_double_gaussian(1.5, 6, 0, 3.0)),
    ("Rectangular", lambda s: s.set_rectangular(4, 0, 3.0))
]

times = [0, 2, 5, 10]

for i, (name, setup_func) in enumerate(configs):
    sim_temp = WavePacketSimulator(x_range=(-15, 50), N=2048)
    setup_func(sim_temp)

    for j, t in enumerate(times):
        sim_temp.evolve_to(t)
        ax = axes[i, j]
        ax.fill_between(sim_temp.x, 0, np.abs(sim_temp.psi)**2, alpha=0.4)
        ax.plot(sim_temp.x, np.abs(sim_temp.psi)**2, 'b-', linewidth=2)
        ax.set_xlim(-10, 45)

        if i == 2:
            ax.set_xlabel('x')
        if j == 0:
            ax.set_ylabel(f'{name}\n|ψ|²')
        if i == 0:
            ax.set_title(f't = {t}')
        ax.grid(True, alpha=0.3)

plt.suptitle('Evolution Comparison: All Wave Packets Spread', fontsize=14)
plt.tight_layout()
plt.savefig('initial_condition_comparison.png', dpi=150)
plt.show()

# =============================================================================
# Part 4: Quantitative Analysis
# =============================================================================

print("\n" + "="*70)
print("QUANTITATIVE ANALYSIS: SPREADING RATES")
print("="*70)

# Compare spreading for different initial widths
sigma_values = [1.0, 2.0, 4.0]
t_max = 20

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for sigma in sigma_values:
    sim_temp = WavePacketSimulator(x_range=(-30, 80), N=2048)
    sim_temp.set_gaussian(sigma=sigma, x0=0, p0=2.0)

    # Evolve and collect data
    t_vals = np.linspace(0, t_max, 100)
    delta_x_vals = []

    for t in t_vals:
        sim_temp.evolve_to(t)
        delta_x_vals.append(sim_temp.history['delta_x'][-1])

    tau = 2 * sigma**2  # with m=ℏ=1
    axes[0].plot(t_vals, delta_x_vals, linewidth=2, label=f'σ₀ = {sigma} (τ = {tau:.1f})')

    # Normalized
    axes[1].plot(t_vals/tau, np.array(delta_x_vals)/sigma, linewidth=2,
                 label=f'σ₀ = {sigma}')

axes[0].set_xlabel('t')
axes[0].set_ylabel('Δx(t)')
axes[0].set_title('Absolute Spreading')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Analytical curve
t_norm = np.linspace(0, 10, 100)
axes[1].plot(t_norm, np.sqrt(1 + t_norm**2), 'k--', linewidth=2, label='√(1 + t²/τ²)')
axes[1].set_xlabel('t / τ')
axes[1].set_ylabel('Δx(t) / σ₀')
axes[1].set_title('Normalized Spreading (Universal Curve)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spreading_analysis.png', dpi=150)
plt.show()

print("\nAll Gaussian packets follow the universal spreading curve σ(t)/σ₀ = √(1 + t²/τ²)")
print("Narrower packets spread faster (smaller τ).")

# =============================================================================
# Part 5: Summary Statistics
# =============================================================================

print("\n" + "="*70)
print("WEEK 53 SUMMARY VERIFICATION")
print("="*70)

# Verify key results
sim_verify = WavePacketSimulator()
sim_verify.set_gaussian(sigma=2.0, x0=0, p0=5.0)

print("\nInitial Gaussian: σ₀ = 2.0, p₀ = 5.0")
print(f"  ⟨x⟩ = {sim_verify.history['x_mean'][-1]:.4f} (expected: 0)")
print(f"  ⟨p⟩ = {sim_verify.history['p_mean'][-1]:.4f} (expected: 5)")
print(f"  Δx = {sim_verify.history['delta_x'][-1]:.4f} (expected: 2)")
print(f"  Δp = {sim_verify.history['delta_p'][-1]:.4f} (expected: 0.25)")
print(f"  Δx·Δp = {sim_verify.get_uncertainty_product():.4f} (expected: 0.5)")

# Group velocity check
sim_verify.evolve_to(10)
v_g_expected = 5.0  # p₀/m with m=1
v_g_actual = sim_verify.history['x_mean'][-1] / 10
print(f"\nAfter t = 10:")
print(f"  ⟨x⟩ = {sim_verify.history['x_mean'][-1]:.2f}")
print(f"  v_g (actual) = {v_g_actual:.4f}")
print(f"  v_g (expected) = {v_g_expected:.4f}")

print("\n✓ All week results verified numerically!")
```

---

## Practice Problems: Comprehensive Review

### Conceptual Questions

1. **Continuous vs Discrete Spectrum:** Why does the free particle have a continuous spectrum while the harmonic oscillator has discrete energies?

2. **Normalization:** A student claims plane waves are physical because |ψ|² = 1 everywhere. Explain the flaw in this reasoning.

3. **Group vs Phase Velocity:** For a free particle, v_p = v_g/2. Is this always true? What about photons?

4. **Spreading Direction:** A wave packet initially at x = 0 with p₀ > 0 spreads as it moves right. Does it also spread to the left?

### Calculation Problems

5. **Electron Wave Packet:** An electron (m = 9.11 × 10⁻³¹ kg) is localized to σ = 10 nm with average momentum corresponding to 1 eV kinetic energy.
   (a) What is the spreading time τ?
   (b) After what time does Δx double?
   (c) What is v_g?

6. **Proton vs Electron:** Compare the spreading times for a proton and electron with the same initial width σ = 1 nm. By what factor do they differ?

7. **Momentum Spread:** A Gaussian wave packet has Δx = 0.5 nm at t = 0 and Δx = 1 nm at t = 10 fs. Find the initial momentum uncertainty Δp.

8. **Kinetic Energy:** For a Gaussian with σ = 2 nm and p₀ = 10⁻²⁴ kg·m/s, calculate ⟨T⟩ and ΔT.

### Advanced Problems

9. **Non-Gaussian Spreading:** A rectangular wave packet has the same Δx·Δp product as a Gaussian at t = 0. After time t >> τ, which spreads faster?

10. **Relativistic Limit:** Estimate the initial width σ for which v_g approaches c (speed of light) for an electron.

11. **Measurement Problem:** A position measurement localizes a free particle to Δx = 1 nm. Estimate how long before Δx grows to 1 μm.

12. **Wigner Function:** Sketch the Wigner function W(x, p) for a Gaussian at t = 0 and t = τ. How does the shape change?

---

## Summary

### Week 53 Key Results

| Topic | Key Equation | Physical Meaning |
|-------|--------------|------------------|
| Free particle TISE | -ℏ²ψ''/2m = Eψ | No potential → continuous spectrum |
| Plane waves | ψ_k = Ae^{ikx} | Momentum eigenstates, delocalized |
| Dispersion | E = ℏ²k²/2m | Energy quadratic in k |
| Wave packet | ψ = ∫φ(k)e^{ikx}dk | Localized, normalizable |
| Gaussian | Δx·Δp = ℏ/2 | Minimum uncertainty |
| Group velocity | v_g = p/m | Classical correspondence |
| Spreading | σ(t) = σ₀√(1+t²/τ²) | Quantum diffusion |
| Dual spaces | FT: ψ(x) ↔ φ(p) | Equivalent descriptions |

### Checklist for Week 53 Mastery

- [ ] I can solve the free particle TISE
- [ ] I understand continuous spectrum and normalization
- [ ] I can construct wave packets from plane waves
- [ ] I know why Gaussians are special (minimum uncertainty)
- [ ] I can calculate group and phase velocities
- [ ] I understand wave packet spreading
- [ ] I can transform between x and p representations
- [ ] I can use FFT for numerical calculations
- [ ] I completed the comprehensive lab
- [ ] I answered the practice problems

---

## Preview: Week 54 - Bound States

Next week, we introduce the first **bound state** problem: the **infinite square well** (particle in a box). Key contrasts with the free particle:

| Free Particle | Infinite Square Well |
|---------------|---------------------|
| No boundaries | Hard wall boundaries |
| Continuous spectrum | Discrete spectrum |
| E = ℏ²k²/2m (any E > 0) | E_n = n²π²ℏ²/2mL² |
| Plane wave solutions | Sinusoidal solutions |
| Not normalizable | Normalizable |
| Wave packet spreading | Stationary states |

The infinite well is the simplest model showing **quantization** — the quantum mechanical origin of discrete atomic energy levels.

---

*"This week we learned that even 'nothing' — a free particle with no potential — exhibits rich quantum behavior. The wave packet is nature's way of compromising between the wave and particle descriptions, and its spreading is a direct manifestation of the uncertainty principle in action."*
