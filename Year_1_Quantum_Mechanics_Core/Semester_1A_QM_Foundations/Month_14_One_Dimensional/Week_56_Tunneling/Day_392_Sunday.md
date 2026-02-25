# Day 392: Month 14 Capstone - One-Dimensional Quantum Mechanics

## Week 56, Day 7 | Month 14 Review & Assessment

### Schedule Overview (8 hours - Extended Capstone)

| Block | Time | Focus |
|-------|------|-------|
| **Morning** | 3 hrs | Comprehensive review of Weeks 53-56 |
| **Afternoon** | 3 hrs | Capstone computational project |
| **Evening** | 2 hrs | Assessment problems and self-reflection |

---

## Learning Objectives

By the end of today, you will:

1. **Integrate** all one-dimensional quantum mechanics concepts from Month 14
2. **Connect** free particles, bound states, and scattering/tunneling phenomena
3. **Apply** mathematical techniques to solve novel problems
4. **Complete** a comprehensive computational capstone project
5. **Demonstrate** mastery through assessment problems
6. **Prepare** for Month 15: Angular Momentum and Spin

---

## Part I: Comprehensive Month 14 Review

### Week 53: Free Particle & Wave Packets

**Key Concepts:**
- Plane wave solutions: $\psi_k(x,t) = Ae^{i(kx - \omega t)}$
- Dispersion relation: $\omega = \hbar k^2/2m$
- Wave packet construction: $\Psi(x,t) = \int_{-\infty}^{\infty}A(k)e^{i(kx-\omega t)}dk$
- Group velocity: $v_g = d\omega/dk = \hbar k/m$ (equals classical velocity!)
- Spreading: $\Delta x(t) \propto \sqrt{1 + t^2/\tau^2}$
- Uncertainty relation: $\Delta x \cdot \Delta p \geq \hbar/2$

**Master Equations:**
$$\boxed{E = \frac{\hbar^2k^2}{2m} = \frac{p^2}{2m}}$$
$$\boxed{\Delta x(t) = \Delta x_0\sqrt{1 + \frac{\hbar^2 t^2}{4m^2\Delta x_0^4}}}$$

### Week 54: Bound States & Infinite/Finite Wells

**Key Concepts:**
- Infinite square well: $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$
- Discrete spectrum from boundary conditions
- Finite well: transcendental equations for energies
- Even/odd parity states
- Penetration into classically forbidden regions

**Master Equations:**
$$\boxed{\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)}$$
$$\boxed{\tan(ka) = \frac{\kappa}{k} \text{ (even)}, \quad \cot(ka) = -\frac{\kappa}{k} \text{ (odd)}}$$

### Week 55: Quantum Harmonic Oscillator

**Key Concepts:**
- Ladder operators: $\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}(\hat{x} + \frac{i\hat{p}}{m\omega})$
- Energy spectrum: $E_n = \hbar\omega(n + 1/2)$
- Zero-point energy: $E_0 = \frac{1}{2}\hbar\omega$
- Coherent states: eigenstates of $\hat{a}$
- Hermite polynomials: $\psi_n(x) \propto H_n(\xi)e^{-\xi^2/2}$

**Master Equations:**
$$\boxed{\hat{a}|n\rangle = \sqrt{n}|n-1\rangle, \quad \hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle}$$
$$\boxed{E_n = \hbar\omega\left(n + \frac{1}{2}\right)}$$

### Week 56: Tunneling & Barriers

**Key Concepts:**
- Step potential: reflection and transmission coefficients
- Rectangular barrier: $T = [1 + V_0^2\sinh^2(\kappa L)/(4E(V_0-E))]^{-1}$
- WKB approximation: $T \approx e^{-2\gamma}$
- Gamow factor: $\gamma = \frac{1}{\hbar}\int\sqrt{2m(V-E)}dx$
- Alpha decay: Gamow model, Geiger-Nuttall law
- STM: exponential sensitivity, atomic resolution
- Josephson effect: supercurrent, AC oscillation

**Master Equations:**
$$\boxed{T \approx e^{-2\gamma}, \quad \gamma = \frac{1}{\hbar}\int_{x_1}^{x_2}\sqrt{2m(V(x)-E)}dx}$$
$$\boxed{I_s = I_c\sin\phi, \quad \frac{d\phi}{dt} = \frac{2eV}{\hbar}}$$

---

## Part II: Unified Framework

### The Schrodinger Equation as Central Theme

All Month 14 topics stem from the time-independent Schrodinger equation:

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$$

| System | V(x) | Spectrum | Key Physics |
|--------|------|----------|-------------|
| Free particle | 0 | Continuous | Wave packets, dispersion |
| Infinite well | 0 inside, ∞ outside | Discrete | Quantization from boundaries |
| Finite well | -V_0 inside, 0 outside | Discrete (bound) + continuous | Penetration, finite states |
| Harmonic oscillator | $\frac{1}{2}m\omega^2x^2$ | Discrete, equally spaced | Ladder operators, coherent states |
| Step potential | 0 to V_0 | Continuous | Reflection, evanescent waves |
| Rectangular barrier | V_0 for 0<x<L | Continuous | Tunneling, resonances |

### Mathematical Techniques Summary

1. **Boundary conditions:** Continuity of ψ and dψ/dx
2. **Normalization:** $\int|\psi|^2dx = 1$
3. **Operator methods:** Ladder operators for harmonic oscillator
4. **Perturbation theory:** Small corrections to known solutions
5. **WKB approximation:** Semi-classical tunneling
6. **Transfer matrices:** Multiple interfaces

### Physical Insight Summary

1. **Quantization arises from boundary conditions** - confinement discretizes energy
2. **Wave functions penetrate forbidden regions** - evanescent waves, tunneling
3. **Uncertainty limits localization** - wave packets spread
4. **Phase coherence enables interference** - resonances, Josephson effect
5. **Classical limit emerges for large quantum numbers** - correspondence principle

---

## Part III: Capstone Computational Project

### Project: Universal 1D Quantum Solver

Build a comprehensive tool that solves any 1D quantum mechanics problem numerically.

```python
"""
Day 392: Month 14 Capstone Project
Universal 1D Quantum Mechanics Solver

This comprehensive project integrates all Month 14 concepts:
1. Arbitrary potential V(x) input
2. Eigenvalue finder for bound states
3. Scattering solver for transmission/reflection
4. Time evolution of wave packets
5. Visualization suite
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad, solve_ivp
from scipy.optimize import brentq, fsolve
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.animation as animation

# Physical constants
hbar = 1.055e-34
m_e = 9.109e-31
eV = 1.602e-19

class QuantumSystem1D:
    """
    Universal 1D Quantum Mechanics Solver

    Handles:
    - Bound state eigenvalue problems
    - Scattering and tunneling
    - Time evolution
    """

    def __init__(self, V_func, x_min, x_max, N_points=1000, mass=m_e):
        """
        Initialize quantum system

        Parameters:
        V_func: Potential function V(x) in Joules, takes x in meters
        x_min, x_max: Domain boundaries in meters
        N_points: Number of grid points
        mass: Particle mass
        """
        self.V_func = V_func
        self.x_min = x_min
        self.x_max = x_max
        self.N = N_points
        self.m = mass

        # Spatial grid
        self.x = np.linspace(x_min, x_max, N_points)
        self.dx = self.x[1] - self.x[0]
        self.V = np.array([V_func(xi) for xi in self.x])

    def solve_bound_states(self, n_states=10, E_max=None):
        """
        Find bound state energies and wave functions using finite differences

        Returns:
        energies: Array of eigenvalues
        wavefunctions: 2D array, wavefunctions[i] is the i-th eigenstate
        """
        # Kinetic energy coefficient
        T_coeff = -hbar**2 / (2 * self.m * self.dx**2)

        # Tridiagonal Hamiltonian matrix
        # H_ii = -2*T_coeff + V_i
        # H_i,i+1 = H_i,i-1 = T_coeff

        diagonal = -2 * T_coeff * np.ones(self.N) + self.V
        off_diagonal = T_coeff * np.ones(self.N - 1)

        # Solve eigenvalue problem
        energies, wavefunctions = eigh_tridiagonal(diagonal, off_diagonal)

        # Normalize wavefunctions
        for i in range(len(energies)):
            norm = np.sqrt(np.trapz(wavefunctions[:, i]**2, self.x))
            wavefunctions[:, i] /= norm

        # Filter bound states (E < 0 or E < V_max at boundaries)
        V_boundary = max(self.V[0], self.V[-1])
        bound_mask = energies < V_boundary

        if E_max is not None:
            bound_mask &= (energies < E_max)

        bound_energies = energies[bound_mask][:n_states]
        bound_wfs = wavefunctions[:, bound_mask][:, :n_states]

        return bound_energies, bound_wfs.T

    def transmission_coefficient(self, E):
        """
        Calculate transmission coefficient for energy E using transfer matrix

        Assumes V = 0 at boundaries (scattering setup)
        """
        if E <= 0:
            return 0.0

        # Numerov method for shooting
        k_left = np.sqrt(2 * self.m * E) / hbar
        k_right = k_left  # Assuming same V at both boundaries

        # Integrate Schrodinger equation from left
        # Initial conditions: plane wave from left
        psi = np.zeros(self.N, dtype=complex)
        psi[0] = 1.0
        psi[1] = np.exp(1j * k_left * self.dx)

        # Numerov integration
        for i in range(1, self.N - 1):
            k2 = 2 * self.m * (E - self.V[i]) / hbar**2
            k2_prev = 2 * self.m * (E - self.V[i-1]) / hbar**2
            k2_next = 2 * self.m * (E - self.V[i+1]) / hbar**2

            psi[i+1] = (2 * (1 - 5*self.dx**2*k2/12) * psi[i] -
                       (1 + self.dx**2*k2_prev/12) * psi[i-1]) / (1 + self.dx**2*k2_next/12)

        # Extract transmission amplitude from asymptotic behavior
        # At right boundary: psi = A*exp(ikx) + B*exp(-ikx)
        # For pure transmission: B = 0, T = |A|^2 * k_right/k_left

        # Simplified: use ratio of amplitudes
        T = np.abs(psi[-1])**2 / np.abs(psi[0])**2

        return min(T, 1.0)  # Cap at 1

    def wkb_transmission(self, E):
        """
        Calculate transmission using WKB approximation
        """
        # Find classical turning points
        turning_points = []
        for i in range(len(self.x) - 1):
            if (self.V[i] - E) * (self.V[i+1] - E) < 0:
                # Linear interpolation
                x_turn = self.x[i] + (E - self.V[i]) / (self.V[i+1] - self.V[i]) * self.dx
                turning_points.append(x_turn)

        if len(turning_points) < 2:
            return 1.0 if E > np.max(self.V) else 0.0

        # Gamow integral between turning points
        x1, x2 = turning_points[0], turning_points[1]

        def integrand(x):
            V = self.V_func(x)
            if V > E:
                return np.sqrt(2 * self.m * (V - E))
            return 0

        gamma, _ = quad(integrand, x1, x2)
        gamma /= hbar

        return np.exp(-2 * gamma)

    def time_evolution(self, psi0, t_final, n_steps=100):
        """
        Evolve wave function in time using split-operator method

        Parameters:
        psi0: Initial wave function array
        t_final: Final time
        n_steps: Number of time steps

        Returns:
        t_array: Time points
        psi_t: Wave function at each time (n_steps x N array)
        """
        dt = t_final / n_steps
        t_array = np.linspace(0, t_final, n_steps)

        # Momentum space grid
        k = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi

        # Evolution operators
        exp_V = np.exp(-1j * self.V * dt / (2 * hbar))
        exp_T = np.exp(-1j * hbar * k**2 * dt / (2 * self.m))

        psi = psi0.astype(complex)
        psi_t = [psi.copy()]

        for _ in range(n_steps - 1):
            # Split-operator: exp(-iHt) ≈ exp(-iVt/2) exp(-iTt) exp(-iVt/2)
            psi = exp_V * psi
            psi = np.fft.ifft(exp_T * np.fft.fft(psi))
            psi = exp_V * psi
            psi_t.append(psi.copy())

        return t_array, np.array(psi_t)

    def plot_potential_and_states(self, n_states=5, scale=1.0):
        """
        Visualize potential and bound states
        """
        energies, wavefunctions = self.solve_bound_states(n_states)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot potential
        ax.plot(self.x * 1e9, self.V / eV, 'k-', linewidth=2, label='V(x)')

        # Plot each bound state
        colors = plt.cm.viridis(np.linspace(0, 1, len(energies)))

        for i, (E, psi) in enumerate(zip(energies, wavefunctions)):
            # Scale wave function and offset by energy
            psi_scaled = scale * psi / np.max(np.abs(psi)) + E / eV
            ax.plot(self.x * 1e9, psi_scaled, color=colors[i], linewidth=1.5,
                   label=f'n={i}, E={E/eV:.3f} eV')
            ax.axhline(y=E/eV, color=colors[i], linestyle='--', alpha=0.5)

        ax.set_xlabel('Position x (nm)', fontsize=12)
        ax.set_ylabel('Energy (eV) / Wave function', fontsize=12)
        ax.set_title('Bound States in Potential', fontsize=14)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        return fig, ax

#%% Example 1: Finite Square Well
print("="*60)
print("Example 1: Finite Square Well")
print("="*60)

def finite_well(x, V0=2*eV, L=1e-9):
    """Finite square well potential"""
    if abs(x) < L/2:
        return -V0
    return 0

system_well = QuantumSystem1D(finite_well, -3e-9, 3e-9, N_points=1000)
energies_well, wfs_well = system_well.solve_bound_states(5)

print(f"\nBound state energies:")
for i, E in enumerate(energies_well):
    print(f"  n={i}: E = {E/eV:.4f} eV")

fig1, ax1 = system_well.plot_potential_and_states(5, scale=0.3)
plt.savefig('capstone_finite_well.png', dpi=150)
plt.show()

#%% Example 2: Harmonic Oscillator
print("\n" + "="*60)
print("Example 2: Harmonic Oscillator")
print("="*60)

omega = 1e14  # rad/s (typical for molecules)

def harmonic(x):
    """Harmonic oscillator potential"""
    return 0.5 * m_e * omega**2 * x**2

system_ho = QuantumSystem1D(harmonic, -5e-9, 5e-9, N_points=1000)
energies_ho, wfs_ho = system_ho.solve_bound_states(8)

# Compare with analytical
E_analytical = hbar * omega * (np.arange(8) + 0.5)

print(f"\nEnergy comparison (ℏω = {hbar*omega/eV:.4f} eV):")
print(f"{'n':>3} {'Numerical':>12} {'Analytical':>12} {'Error (%)':>12}")
for i, (E_num, E_ana) in enumerate(zip(energies_ho, E_analytical)):
    error = 100 * (E_num - E_ana) / E_ana
    print(f"{i:>3} {E_num/eV:>12.6f} {E_ana/eV:>12.6f} {error:>12.4f}")

fig2, ax2 = system_ho.plot_potential_and_states(5, scale=0.02)
ax2.set_xlim(-3, 3)
plt.savefig('capstone_harmonic.png', dpi=150)
plt.show()

#%% Example 3: Double Well (Quantum Tunneling)
print("\n" + "="*60)
print("Example 3: Double Well (Tunnel Splitting)")
print("="*60)

def double_well(x, V0=0.5*eV, a=1e-9, b=0.5e-9):
    """Double well potential"""
    # Barrier in middle, wells on either side
    if abs(x) < b/2:
        return V0  # Barrier
    elif abs(x) < a:
        return 0   # Wells
    return 2*V0   # Walls

system_dw = QuantumSystem1D(double_well, -2e-9, 2e-9, N_points=1000)
energies_dw, wfs_dw = system_dw.solve_bound_states(4)

# Tunnel splitting
if len(energies_dw) >= 2:
    splitting = energies_dw[1] - energies_dw[0]
    print(f"\nTunnel splitting: ΔE = {splitting/eV*1000:.4f} meV")
    print(f"Tunnel frequency: f = {splitting/(2*np.pi*hbar)/1e9:.4f} GHz")

fig3, ax3 = system_dw.plot_potential_and_states(4, scale=0.15)
plt.savefig('capstone_double_well.png', dpi=150)
plt.show()

#%% Example 4: Tunneling Through Barrier
print("\n" + "="*60)
print("Example 4: Barrier Transmission")
print("="*60)

def barrier(x, V0=1*eV, L=0.5e-9):
    """Rectangular barrier"""
    if 0 < x < L:
        return V0
    return 0

system_barrier = QuantumSystem1D(barrier, -2e-9, 3e-9, N_points=2000)

E_values = np.linspace(0.01, 2, 100) * eV
T_numerical = [system_barrier.transmission_coefficient(E) for E in E_values]
T_wkb = [system_barrier.wkb_transmission(E) for E in E_values]

fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.semilogy(E_values/eV, T_numerical, 'b-', linewidth=2, label='Numerical')
ax4.semilogy(E_values/eV, T_wkb, 'r--', linewidth=2, label='WKB')
ax4.axvline(x=1, color='gray', linestyle=':', label='$V_0$')
ax4.set_xlabel('Energy E (eV)', fontsize=12)
ax4.set_ylabel('Transmission T', fontsize=12)
ax4.set_title('Barrier Transmission Coefficient', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')
plt.savefig('capstone_transmission.png', dpi=150)
plt.show()

#%% Example 5: Wave Packet Dynamics
print("\n" + "="*60)
print("Example 5: Wave Packet Time Evolution")
print("="*60)

# Free particle wave packet
def free(x):
    return 0

system_free = QuantumSystem1D(free, -50e-9, 50e-9, N_points=2000)

# Initial Gaussian wave packet
x0 = -20e-9  # Initial position
sigma = 5e-9  # Width
k0 = 5e9     # Initial momentum (wave number)

psi0 = np.exp(-(system_free.x - x0)**2 / (2*sigma**2)) * np.exp(1j * k0 * system_free.x)
psi0 /= np.sqrt(np.trapz(np.abs(psi0)**2, system_free.x))  # Normalize

# Time evolution
t_final = 5e-13  # 500 fs
t_array, psi_t = system_free.time_evolution(psi0, t_final, n_steps=100)

# Plot snapshots
fig5, axes = plt.subplots(2, 3, figsize=(15, 8))
times_to_plot = [0, 20, 40, 60, 80, 99]

for ax, t_idx in zip(axes.flatten(), times_to_plot):
    prob = np.abs(psi_t[t_idx])**2
    ax.plot(system_free.x * 1e9, prob * 1e-9, 'b-', linewidth=2)
    ax.set_xlabel('x (nm)', fontsize=10)
    ax.set_ylabel('$|\\psi|^2$ (nm$^{-1}$)', fontsize=10)
    ax.set_title(f't = {t_array[t_idx]*1e15:.0f} fs', fontsize=11)
    ax.set_xlim(-50, 50)
    ax.grid(True, alpha=0.3)

plt.suptitle('Wave Packet Propagation and Spreading', fontsize=14)
plt.tight_layout()
plt.savefig('capstone_wavepacket.png', dpi=150)
plt.show()

# Calculate spreading
widths = []
centers = []
for psi in psi_t:
    prob = np.abs(psi)**2
    center = np.trapz(system_free.x * prob, system_free.x)
    width = np.sqrt(np.trapz((system_free.x - center)**2 * prob, system_free.x))
    centers.append(center)
    widths.append(width)

fig6, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(t_array * 1e15, np.array(centers) * 1e9, 'b-', linewidth=2)
axes[0].set_xlabel('Time (fs)', fontsize=12)
axes[0].set_ylabel('Center position (nm)', fontsize=12)
axes[0].set_title('Wave Packet Center Motion', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Theoretical velocity
v_theory = hbar * k0 / m_e
axes[0].plot(t_array * 1e15, (x0 + v_theory * t_array) * 1e9, 'r--',
             linewidth=2, label=f'$v_g = \\hbar k_0/m$ = {v_theory/1000:.0f} km/s')
axes[0].legend(fontsize=10)

axes[1].plot(t_array * 1e15, np.array(widths) * 1e9, 'b-', linewidth=2)
axes[1].set_xlabel('Time (fs)', fontsize=12)
axes[1].set_ylabel('Width $\\Delta x$ (nm)', fontsize=12)
axes[1].set_title('Wave Packet Spreading', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Theoretical spreading
tau = 2 * m_e * sigma**2 / hbar
width_theory = sigma * np.sqrt(1 + (t_array / tau)**2)
axes[1].plot(t_array * 1e15, width_theory * 1e9, 'r--', linewidth=2,
             label=f'$\\sigma\\sqrt{{1+(t/\\tau)^2}}$, τ = {tau*1e15:.0f} fs')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('capstone_spreading.png', dpi=150)
plt.show()

#%% Summary visualization
print("\n" + "="*60)
print("Month 14 Capstone Complete!")
print("="*60)

fig_summary = plt.figure(figsize=(16, 12))

# Create 2x3 grid of final results
ax1 = fig_summary.add_subplot(231)
ax1.plot(system_well.x * 1e9, system_well.V / eV, 'k-', linewidth=2)
ax1.set_title('Finite Well', fontsize=12)
ax1.set_xlabel('x (nm)')
ax1.set_ylabel('V (eV)')

ax2 = fig_summary.add_subplot(232)
ax2.plot(system_ho.x * 1e9, system_ho.V / eV, 'k-', linewidth=2)
ax2.set_title('Harmonic Oscillator', fontsize=12)
ax2.set_xlabel('x (nm)')
ax2.set_ylabel('V (eV)')
ax2.set_xlim(-3, 3)
ax2.set_ylim(0, 0.5)

ax3 = fig_summary.add_subplot(233)
ax3.plot(system_dw.x * 1e9, np.array([double_well(xi) for xi in system_dw.x]) / eV, 'k-', linewidth=2)
ax3.set_title('Double Well', fontsize=12)
ax3.set_xlabel('x (nm)')
ax3.set_ylabel('V (eV)')

ax4 = fig_summary.add_subplot(234)
ax4.bar(range(len(energies_well)), energies_well / eV)
ax4.set_title('Well Energies', fontsize=12)
ax4.set_xlabel('State n')
ax4.set_ylabel('E (eV)')

ax5 = fig_summary.add_subplot(235)
ax5.semilogy(E_values/eV, T_numerical, 'b-', linewidth=2)
ax5.axvline(x=1, color='r', linestyle='--')
ax5.set_title('Barrier Transmission', fontsize=12)
ax5.set_xlabel('E (eV)')
ax5.set_ylabel('T')

ax6 = fig_summary.add_subplot(236)
prob_final = np.abs(psi_t[-1])**2
ax6.plot(system_free.x * 1e9, prob_final * 1e-9, 'b-', linewidth=2)
ax6.set_title('Final Wave Packet', fontsize=12)
ax6.set_xlabel('x (nm)')
ax6.set_ylabel('$|\\psi|^2$')

plt.suptitle('Month 14: One-Dimensional Quantum Mechanics - Capstone Summary', fontsize=14)
plt.tight_layout()
plt.savefig('capstone_summary.png', dpi=150)
plt.show()

print("\nAll capstone visualizations saved!")
```

---

## Part IV: Assessment Problems

### Section A: Conceptual Questions (15 points)

**A1.** (3 pts) Explain why the ground state energy of any bound system is always greater than the minimum of the potential. Connect this to the uncertainty principle.

**A2.** (3 pts) Why does the transmission coefficient show resonances (T = 1) for energies above a rectangular barrier but not below?

**A3.** (3 pts) Compare the energy spectra of the infinite square well, finite square well, and harmonic oscillator. Which has the largest spacing between ground and first excited state for the same characteristic length?

**A4.** (3 pts) Explain physically why heavier particles tunnel less efficiently than lighter ones, using the Gamow factor.

**A5.** (3 pts) How does the Josephson effect differ fundamentally from normal electron tunneling? What makes it "quantum coherent"?

### Section B: Calculations (35 points)

**B1.** (7 pts) An electron in a 2 nm infinite square well absorbs a photon and transitions from n=1 to n=3. Calculate:
(a) The energy of the absorbed photon
(b) The wavelength of the photon
(c) The new expectation value of momentum ⟨p²⟩

**B2.** (7 pts) A particle of mass m is in the ground state of a harmonic oscillator with frequency ω. At t=0, the frequency is suddenly doubled to 2ω.
(a) What is the probability that the particle remains in the ground state?
(b) What is the expectation value of energy immediately after the change?

**B3.** (7 pts) An electron with E = 3 eV encounters a step potential with V₀ = 5 eV.
(a) Calculate the penetration depth
(b) Find the phase shift of the reflected wave
(c) At what distance into the barrier has the probability density decreased to 1% of its value at x=0?

**B4.** (7 pts) For the rectangular barrier (V₀ = 4 eV, L = 0.8 nm), calculate:
(a) The transmission coefficient at E = 2 eV
(b) The transmission coefficient at E = 6 eV (above barrier)
(c) The first resonance energy above the barrier

**B5.** (7 pts) A transmon qubit has $E_J/h$ = 15 GHz and $E_C/h$ = 300 MHz.
(a) Calculate the qubit transition frequency f₀₁
(b) Calculate the anharmonicity α
(c) If a π-pulse takes 20 ns, estimate the leakage to the |2⟩ state

### Section C: Extended Problem (20 points)

**C1.** (20 pts) **Complete Scattering Analysis**

Consider a potential consisting of two delta function barriers:
$$V(x) = \alpha[\delta(x) + \delta(x-a)]$$

(a) Set up the wave function in each region (x < 0, 0 < x < a, x > a)
(b) Apply boundary conditions (ψ continuous, dψ/dx has discontinuity α ψ(0)/(\hbar²/2m) at each delta)
(c) Derive the transmission coefficient T(E)
(d) Find the condition for perfect transmission (T = 1)
(e) Interpret the resonance condition physically in terms of standing waves between the barriers

---

## Part V: Self-Assessment & Reflection

### Month 14 Mastery Checklist

Rate your confidence (1-5) for each topic:

**Week 53: Free Particle**
- [ ] I can construct and evolve wave packets (  /5)
- [ ] I understand dispersion and group velocity (  /5)
- [ ] I can derive the spreading rate (  /5)

**Week 54: Bound States**
- [ ] I can solve the infinite well problem completely (  /5)
- [ ] I can set up transcendental equations for finite wells (  /5)
- [ ] I understand even/odd parity classification (  /5)

**Week 55: Harmonic Oscillator**
- [ ] I can use ladder operators fluently (  /5)
- [ ] I know the energy spectrum and wave functions (  /5)
- [ ] I understand coherent states (  /5)

**Week 56: Tunneling**
- [ ] I can calculate R and T for step and barrier (  /5)
- [ ] I can apply WKB approximation (  /5)
- [ ] I understand Josephson physics (  /5)

**Total Score:** ___/60

**Interpretation:**
- 50-60: Excellent mastery, ready for Month 15
- 40-49: Good understanding, review weak areas
- 30-39: Adequate, focus practice on key topics
- <30: Review Month 14 material before proceeding

### Reflection Questions

1. What concept from Month 14 was most surprising or counterintuitive?

2. How has your physical intuition about quantum mechanics changed?

3. Which computational technique will be most useful for future work?

4. What connections do you see between Month 14 topics and quantum computing?

---

## Preview: Month 15 - Angular Momentum and Spin

Month 15 will extend our quantum mechanics foundation to three dimensions and introduce one of the most important concepts in quantum physics: **spin**.

**Week 57-58:** Angular Momentum
- Orbital angular momentum operators L̂x, L̂y, L̂z
- Commutation relations [L̂i, L̂j] = iℏε_{ijk}L̂k
- Spherical harmonics Y_l^m(θ,φ)
- Ladder operators L̂± = L̂x ± iL̂y

**Week 59-60:** Spin and Addition
- Intrinsic spin angular momentum
- Spin-1/2 systems and Pauli matrices
- Addition of angular momenta
- Clebsch-Gordan coefficients

**Key equation preview:**
$$[\hat{L}^2, \hat{L}_z] = 0, \quad \hat{L}^2|l,m\rangle = \hbar^2 l(l+1)|l,m\rangle, \quad \hat{L}_z|l,m\rangle = \hbar m|l,m\rangle$$

---

## Congratulations!

You have completed **Month 14: One-Dimensional Quantum Mechanics**!

You now have a solid foundation in:
- Time evolution and wave packet dynamics
- Bound state quantization
- Operator methods (especially ladder operators)
- Scattering and tunneling phenomena
- Applications from alpha decay to superconducting qubits

These tools will serve you throughout your quantum mechanics journey and are directly applicable to quantum computing and quantum engineering!

---

*Month 14 Complete. Next: Month 15 - Angular Momentum and Spin*
