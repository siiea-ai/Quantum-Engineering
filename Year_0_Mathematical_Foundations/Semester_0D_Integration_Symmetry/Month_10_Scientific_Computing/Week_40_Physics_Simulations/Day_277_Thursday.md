# Day 277: Schrödinger Equation Simulations (1D)

## Schedule Overview
**Date**: Week 40, Day 4 (Thursday)
**Duration**: 7 hours
**Theme**: Numerical Solution of the Time-Dependent Schrödinger Equation

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Split-step Fourier method, unitarity preservation |
| Afternoon | 2.5 hours | Quantum tunneling, wave packet dynamics |
| Evening | 1.5 hours | Computational lab: Complete TDSE solver |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Implement the split-step Fourier method for TDSE
2. Understand and preserve unitarity in numerical evolution
3. Simulate quantum tunneling through barriers
4. Visualize wave packet spreading and interference
5. Connect numerical solutions to analytical results

---

## Core Content

### 1. Time-Dependent Schrödinger Equation

The TDSE in one dimension:
$$i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2} + V(x)\psi = \hat{H}\psi$$

Formal solution:
$$\psi(x, t+\Delta t) = e^{-i\hat{H}\Delta t/\hbar}\psi(x, t)$$

### 2. Split-Step Fourier Method

The Hamiltonian splits: $$\hat{H} = \hat{T} + \hat{V}$$ where:
- $$\hat{T} = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}$$ (kinetic)
- $$\hat{V} = V(x)$$ (potential)

Using Trotter-Suzuki decomposition:
$$e^{-i\hat{H}\Delta t/\hbar} \approx e^{-i\hat{V}\Delta t/2\hbar} e^{-i\hat{T}\Delta t/\hbar} e^{-i\hat{V}\Delta t/2\hbar}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

class SchrodingerSolver1D:
    """
    Solve 1D time-dependent Schrödinger equation using split-step FFT.

    iℏ ∂ψ/∂t = [-ℏ²/(2m) ∂²/∂x² + V(x)] ψ
    """

    def __init__(self, x_min=-20, x_max=20, nx=1024, hbar=1, m=1):
        """
        Initialize solver.

        Parameters
        ----------
        x_min, x_max : float
            Spatial domain
        nx : int
            Number of grid points (should be power of 2 for FFT)
        hbar, m : float
            Physical constants
        """
        self.hbar = hbar
        self.m = m
        self.nx = nx

        # Spatial grid
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = self.x[1] - self.x[0]
        self.L = x_max - x_min

        # Momentum grid
        self.k = 2 * np.pi * fftfreq(nx, self.dx)

        # Kinetic energy in k-space
        self.T_k = (hbar * self.k)**2 / (2 * m)

        # Initialize
        self.psi = np.zeros(nx, dtype=complex)
        self.V = np.zeros(nx)

    def set_potential(self, V_func):
        """Set potential energy function V(x)."""
        self.V = V_func(self.x)

    def set_initial_state(self, psi0_func):
        """Set initial wave function."""
        self.psi = psi0_func(self.x).astype(complex)
        self._normalize()

    def _normalize(self):
        """Normalize wave function."""
        norm = np.sqrt(np.trapz(np.abs(self.psi)**2, self.x))
        self.psi /= norm

    def step(self, dt):
        """
        Advance one time step using split-step FFT.

        Second-order accurate Trotter decomposition:
        exp(-iH dt) ≈ exp(-iV dt/2) exp(-iT dt) exp(-iV dt/2)
        """
        # Half step in position space (potential)
        self.psi *= np.exp(-0.5j * self.V * dt / self.hbar)

        # Full step in momentum space (kinetic)
        psi_k = fft(self.psi)
        psi_k *= np.exp(-1j * self.T_k * dt / self.hbar)
        self.psi = ifft(psi_k)

        # Half step in position space (potential)
        self.psi *= np.exp(-0.5j * self.V * dt / self.hbar)

    def evolve(self, t_total, dt, save_every=10):
        """
        Evolve for total time t_total.

        Returns history of |ψ|² at saved times.
        """
        n_steps = int(t_total / dt)
        history = [np.abs(self.psi)**2]
        psi_history = [self.psi.copy()]
        times = [0]

        for step in range(n_steps):
            self.step(dt)
            if (step + 1) % save_every == 0:
                history.append(np.abs(self.psi)**2)
                psi_history.append(self.psi.copy())
                times.append((step + 1) * dt)

        return np.array(times), np.array(history), np.array(psi_history)

    def expectation_value(self, observable='x'):
        """Compute expectation value ⟨ψ|O|ψ⟩."""
        prob = np.abs(self.psi)**2

        if observable == 'x':
            return np.trapz(self.x * prob, self.x)
        elif observable == 'x2':
            return np.trapz(self.x**2 * prob, self.x)
        elif observable == 'p':
            # ⟨p⟩ = ∫ ψ* (-iℏ d/dx) ψ dx
            psi_deriv = np.gradient(self.psi, self.dx)
            return np.trapz(np.conj(self.psi) * (-1j * self.hbar) * psi_deriv, self.x).real
        elif observable == 'E':
            # ⟨E⟩ = ⟨T⟩ + ⟨V⟩
            psi_k = fft(self.psi) * self.dx / np.sqrt(2*np.pi)
            T_expect = np.sum(self.T_k * np.abs(psi_k)**2) * 2*np.pi/self.L
            V_expect = np.trapz(self.V * prob, self.x)
            return (T_expect + V_expect).real

    def uncertainty(self, var='x'):
        """Compute uncertainty Δx or Δp."""
        if var == 'x':
            x_mean = self.expectation_value('x')
            x2_mean = self.expectation_value('x2')
            return np.sqrt(x2_mean - x_mean**2)
        # Similar for p


def gaussian_wavepacket(x, x0=0, sigma=1, k0=0):
    """
    Gaussian wave packet.

    ψ(x) = (2πσ²)^(-1/4) exp(-(x-x0)²/4σ²) exp(ik₀x)
    """
    norm = (2 * np.pi * sigma**2)**(-0.25)
    return norm * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)


# Demonstration: Free particle spreading
solver = SchrodingerSolver1D(x_min=-30, x_max=30, nx=1024)
solver.set_potential(lambda x: np.zeros_like(x))  # Free particle
solver.set_initial_state(lambda x: gaussian_wavepacket(x, x0=-10, sigma=2, k0=3))

times, history, psi_history = solver.evolve(t_total=15, dt=0.01, save_every=50)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Space-time diagram
im = axes[0, 0].imshow(history, aspect='auto', origin='lower',
                       extent=[solver.x[0], solver.x[-1], 0, times[-1]],
                       cmap='viridis')
axes[0, 0].set_xlabel('Position x')
axes[0, 0].set_ylabel('Time t')
axes[0, 0].set_title('Free Particle Wave Packet Spreading')
plt.colorbar(im, ax=axes[0, 0], label=r'$|\psi|^2$')

# Snapshots
for i in range(0, len(history), len(history)//5):
    axes[0, 1].plot(solver.x, history[i], label=f't = {times[i]:.1f}')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel(r'$|\psi(x)|^2$')
axes[0, 1].legend()
axes[0, 1].set_title('Probability Density Snapshots')
axes[0, 1].set_xlim(-30, 30)

# Width spreading
widths = [np.sqrt(np.trapz(solver.x**2 * h, solver.x) -
                  np.trapz(solver.x * h, solver.x)**2) for h in history]
axes[1, 0].plot(times, widths, 'b-', linewidth=2)
# Analytical: σ(t) = σ₀√(1 + (ℏt/2mσ₀²)²)
sigma0 = 2
t_theory = np.linspace(0, times[-1], 100)
sigma_theory = sigma0 * np.sqrt(1 + (solver.hbar * t_theory / (2 * solver.m * sigma0**2))**2)
axes[1, 0].plot(t_theory, sigma_theory, 'r--', label='Theory')
axes[1, 0].set_xlabel('Time t')
axes[1, 0].set_ylabel('Width σ(t)')
axes[1, 0].set_title('Wave Packet Spreading')
axes[1, 0].legend()

# Probability conservation
norms = [np.trapz(h, solver.x) for h in history]
axes[1, 1].plot(times, norms)
axes[1, 1].set_xlabel('Time t')
axes[1, 1].set_ylabel('Total probability')
axes[1, 1].set_title('Unitarity Check')
axes[1, 1].set_ylim(0.99, 1.01)

plt.tight_layout()
plt.savefig('free_particle_tdse.png', dpi=150)
plt.show()
```

### 3. Quantum Tunneling

```python
def quantum_tunneling_simulation():
    """Simulate quantum tunneling through a rectangular barrier."""

    solver = SchrodingerSolver1D(x_min=-30, x_max=30, nx=2048)

    # Rectangular barrier
    barrier_width = 2
    barrier_height = 5

    def barrier(x):
        return np.where(np.abs(x) < barrier_width/2, barrier_height, 0)

    solver.set_potential(barrier)

    # Incident wave packet
    k0 = 3  # Incident momentum (E = k0²/2 = 4.5 < barrier)
    solver.set_initial_state(lambda x: gaussian_wavepacket(x, x0=-12, sigma=2, k0=k0))

    times, history, psi_history = solver.evolve(t_total=15, dt=0.005, save_every=100)

    # Compute transmission and reflection
    def compute_probs(psi, x, barrier_end=barrier_width/2):
        prob = np.abs(psi)**2
        mask_trans = x > barrier_end + 2
        mask_refl = x < -barrier_end - 2
        P_trans = np.trapz(prob[mask_trans], x[mask_trans])
        P_refl = np.trapz(prob[mask_refl], x[mask_refl])
        return P_trans, P_refl

    T_probs = []
    R_probs = []
    for psi in psi_history:
        T, R = compute_probs(psi, solver.x)
        T_probs.append(T)
        R_probs.append(R)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Space-time with barrier
    im = axes[0, 0].imshow(history, aspect='auto', origin='lower',
                           extent=[solver.x[0], solver.x[-1], 0, times[-1]],
                           cmap='viridis', vmax=0.3)
    # Mark barrier region
    axes[0, 0].axvline(-barrier_width/2, color='white', linestyle='--')
    axes[0, 0].axvline(barrier_width/2, color='white', linestyle='--')
    axes[0, 0].set_xlabel('Position x')
    axes[0, 0].set_ylabel('Time t')
    axes[0, 0].set_title(f'Quantum Tunneling (E = {k0**2/2:.1f}, V₀ = {barrier_height})')
    plt.colorbar(im, ax=axes[0, 0])

    # Snapshots with barrier
    indices = [0, len(history)//4, len(history)//2, 3*len(history)//4, -1]
    for idx in indices:
        axes[0, 1].plot(solver.x, history[idx], label=f't = {times[idx]:.1f}')
    axes[0, 1].fill_between(solver.x, 0, solver.V/solver.V.max()*0.15,
                           alpha=0.3, color='gray', label='Barrier')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel(r'$|\psi|^2$')
    axes[0, 1].set_xlim(-25, 25)
    axes[0, 1].legend()
    axes[0, 1].set_title('Wave Packet Evolution')

    # Transmission and reflection
    axes[1, 0].plot(times, T_probs, 'b-', label='Transmitted')
    axes[1, 0].plot(times, R_probs, 'r-', label='Reflected')
    axes[1, 0].plot(times, np.array(T_probs) + np.array(R_probs), 'k--',
                   label='T + R', alpha=0.5)
    axes[1, 0].set_xlabel('Time t')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].legend()
    axes[1, 0].set_title('Transmission and Reflection Probabilities')

    # Final state detail
    axes[1, 1].plot(solver.x, history[-1], 'b-', linewidth=2)
    axes[1, 1].fill_between(solver.x, 0, solver.V/solver.V.max()*0.1,
                           alpha=0.3, color='gray')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel(r'$|\psi|^2$')
    axes[1, 1].set_title('Final State')
    axes[1, 1].set_xlim(-25, 25)

    # Annotate
    axes[1, 1].annotate('Reflected', xy=(-15, 0.02), fontsize=12)
    axes[1, 1].annotate('Transmitted', xy=(10, 0.01), fontsize=12)

    plt.tight_layout()
    plt.savefig('quantum_tunneling.png', dpi=150)
    plt.show()

    return T_probs[-1], R_probs[-1]

T, R = quantum_tunneling_simulation()
print(f"Final transmission probability: T = {T:.4f}")
print(f"Final reflection probability: R = {R:.4f}")
```

### 4. Harmonic Oscillator Dynamics

```python
def harmonic_oscillator_dynamics():
    """Simulate coherent state in harmonic potential."""

    solver = SchrodingerSolver1D(x_min=-15, x_max=15, nx=1024)

    # Harmonic potential
    omega = 1.0
    solver.set_potential(lambda x: 0.5 * omega**2 * x**2)

    # Coherent state (displaced ground state)
    alpha = 3  # Displacement
    sigma = 1/np.sqrt(2*omega)  # Ground state width

    solver.set_initial_state(
        lambda x: gaussian_wavepacket(x, x0=alpha*np.sqrt(2), sigma=sigma, k0=0)
    )

    # Evolve for several periods
    T_period = 2*np.pi/omega
    times, history, psi_history = solver.evolve(t_total=3*T_period, dt=0.01, save_every=20)

    # Track expectation values
    x_expect = []
    for psi in psi_history:
        prob = np.abs(psi)**2
        x_mean = np.trapz(solver.x * prob, solver.x)
        x_expect.append(x_mean)

    # Classical trajectory for comparison
    x_classical = alpha * np.sqrt(2) * np.cos(omega * times)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Space-time
    im = axes[0, 0].imshow(history, aspect='auto', origin='lower',
                           extent=[solver.x[0], solver.x[-1], 0, times[-1]],
                           cmap='viridis')
    axes[0, 0].plot(x_classical, times, 'r--', linewidth=2, label='Classical')
    axes[0, 0].set_xlabel('Position x')
    axes[0, 0].set_ylabel('Time t')
    axes[0, 0].set_title('Coherent State in Harmonic Potential')
    axes[0, 0].legend()
    plt.colorbar(im, ax=axes[0, 0])

    # ⟨x⟩ vs classical
    axes[0, 1].plot(times, x_expect, 'b-', linewidth=2, label='Quantum ⟨x⟩')
    axes[0, 1].plot(times, x_classical, 'r--', linewidth=2, label='Classical')
    axes[0, 1].set_xlabel('Time t')
    axes[0, 1].set_ylabel('Position')
    axes[0, 1].legend()
    axes[0, 1].set_title('Ehrenfest Theorem: ⟨x⟩ follows classical')

    # Difference
    axes[1, 0].plot(times, np.array(x_expect) - x_classical)
    axes[1, 0].set_xlabel('Time t')
    axes[1, 0].set_ylabel('⟨x⟩ - x_classical')
    axes[1, 0].set_title('Deviation from Classical (numerical error)')

    # Width (should remain constant for coherent state)
    widths = [np.sqrt(np.trapz(solver.x**2 * h, solver.x) -
                      np.trapz(solver.x * h, solver.x)**2) for h in history]
    axes[1, 1].plot(times, widths)
    axes[1, 1].axhline(sigma, color='r', linestyle='--', label=f'Ground state width σ₀={sigma:.3f}')
    axes[1, 1].set_xlabel('Time t')
    axes[1, 1].set_ylabel('Width σ(t)')
    axes[1, 1].set_title('Coherent State: Width Preserved')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('harmonic_oscillator_coherent.png', dpi=150)
    plt.show()

harmonic_oscillator_dynamics()
```

---

## Practice Problems

### Level 1: Direct Application

1. **Plane Wave**: Simulate a plane wave $$\psi = e^{ikx}$$ in a periodic box. Verify it doesn't spread.

2. **Step Potential**: Scatter a wave packet off a step potential. Compute reflection coefficient.

3. **Double Barrier**: Set up a double barrier (resonant tunneling structure). Observe resonances.

### Level 2: Intermediate

4. **Anharmonic Oscillator**: Add a quartic term to the harmonic potential. Observe how dynamics change.

5. **Wave Packet Collision**: Initialize two wave packets moving toward each other. Observe interference.

6. **Time-Dependent Potential**: Implement a time-dependent potential V(x,t). Study parametric driving.

### Level 3: Challenging

7. **Landau-Zener**: Simulate Landau-Zener transitions with a linearly-swept avoided crossing.

8. **Rabi Oscillations**: Model a two-level system interacting with an oscillating field.

9. **Scattering Phase Shift**: Extract scattering phase shifts from numerically computed transmission amplitudes.

---

## Summary

### Key Equations

$$\boxed{i\hbar\frac{\partial\psi}{\partial t} = \hat{H}\psi}$$

$$\boxed{e^{-i\hat{H}dt/\hbar} \approx e^{-i\hat{V}dt/2\hbar} e^{-i\hat{T}dt/\hbar} e^{-i\hat{V}dt/2\hbar}}$$

$$\boxed{\int|\psi|^2dx = 1 \text{ (preserved by unitary evolution)}}$$

---

## Daily Checklist

- [ ] Implemented split-step FFT method
- [ ] Verified unitarity preservation
- [ ] Simulated free particle spreading
- [ ] Modeled quantum tunneling
- [ ] Studied harmonic oscillator dynamics
- [ ] Completed practice problems

---

## Preview of Day 278

Tomorrow: **Quantum Eigenvalue Problems**
- Shooting method for bound states
- Matrix diagonalization approach
- Finding eigenstates of arbitrary potentials
- Variational methods
