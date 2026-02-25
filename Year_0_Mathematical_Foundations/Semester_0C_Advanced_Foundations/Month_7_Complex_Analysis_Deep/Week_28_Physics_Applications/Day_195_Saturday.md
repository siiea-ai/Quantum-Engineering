# Day 195: Comprehensive Computational Lab

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Labs 1-3: Core Techniques |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Labs 4-6: Physics Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Integration Project |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 195, you will be able to:

1. Implement all complex analysis techniques computationally
2. Visualize singularities, residues, and contours
3. Numerically verify analytical results
4. Apply complex methods to physics simulations
5. Build a comprehensive complex analysis toolkit
6. Solve research-level problems

---

## Lab 1: Complete Contour Integration Toolkit

```python
"""
Comprehensive Complex Analysis Toolkit
Integrates all techniques from Month 7
"""

import numpy as np
from scipy import integrate, special
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional

class ComplexAnalysisToolkit:
    """Complete toolkit for complex analysis computations."""

    def __init__(self):
        self.precision = 1e-10

    # ============= CONTOUR INTEGRATION =============

    def contour_integral(self, f: Callable, path: Callable,
                        t_range: Tuple[float, float], n: int = 10000) -> complex:
        """Compute ∮_C f(z) dz numerically."""
        t = np.linspace(t_range[0], t_range[1], n)
        dt = t[1] - t[0]
        z = path(t)
        dz = np.gradient(z, dt, edge_order=2)
        return np.trapz(f(z) * dz, t)

    def residue_at_point(self, f: Callable, z0: complex,
                         radius: float = 0.1) -> complex:
        """Compute residue via contour integration."""
        path = lambda t: z0 + radius * np.exp(1j * t)
        integral = self.contour_integral(f, path, (0, 2*np.pi))
        return integral / (2 * np.pi * 1j)

    def find_poles(self, f: Callable, region: Tuple[complex, complex],
                   grid_size: int = 50, threshold: float = 1e6) -> List[complex]:
        """Find poles in a rectangular region."""
        x = np.linspace(region[0].real, region[1].real, grid_size)
        y = np.linspace(region[0].imag, region[1].imag, grid_size)
        poles = []

        for xi in x:
            for yi in y:
                z = xi + 1j * yi
                try:
                    val = np.abs(f(z))
                    if val > threshold:
                        # Refine pole location
                        poles.append(z)
                except:
                    pass
        return poles

    # ============= REAL INTEGRALS =============

    def rational_integral(self, P: np.poly1d, Q: np.poly1d) -> float:
        """
        Compute ∫_{-∞}^{∞} P(x)/Q(x) dx via residues.
        Requires deg(Q) >= deg(P) + 2.
        """
        poles = np.roots(Q.coeffs)
        uhp_poles = poles[poles.imag > self.precision]

        total_res = 0
        Q_deriv = np.polyder(Q)
        for pole in uhp_poles:
            total_res += P(pole) / Q_deriv(pole)

        return (2 * np.pi * 1j * total_res).real

    def fourier_type_integral(self, f: Callable, omega: float,
                              poles_residues: List[Tuple[complex, complex]]) -> complex:
        """
        Compute ∫_{-∞}^{∞} f(x) e^{iωx} dx via Jordan's lemma.
        poles_residues: list of (pole, residue) in upper half-plane (for ω > 0).
        """
        if omega > 0:
            total = sum(res * np.exp(1j * omega * pole)
                       for pole, res in poles_residues if pole.imag > 0)
            return 2 * np.pi * 1j * total
        else:
            total = sum(res * np.exp(1j * omega * pole)
                       for pole, res in poles_residues if pole.imag < 0)
            return -2 * np.pi * 1j * total

    # ============= LAURENT SERIES =============

    def laurent_coefficients(self, f: Callable, z0: complex,
                             n_terms: int = 10, radius: float = 1.0) -> dict:
        """Compute Laurent series coefficients."""
        coeffs = {}
        for n in range(-n_terms, n_terms + 1):
            g = lambda z: f(z) / (z - z0)**(n + 1)
            coeffs[n] = self.residue_at_point(g, z0, radius)
        return coeffs

    # ============= SPECIAL FUNCTIONS =============

    def gamma_via_integral(self, z: complex) -> complex:
        """Compute Γ(z) via integral (for Re(z) > 0)."""
        if z.real <= 0:
            # Use reflection formula
            return np.pi / (np.sin(np.pi * z) * self.gamma_via_integral(1 - z))

        result, _ = integrate.quad(
            lambda t: t**(z.real - 1) * np.exp(-t) * np.exp(1j * z.imag * np.log(t + 1e-100)),
            0, 100, limit=200)
        return result

    def zeta_via_series(self, s: complex, terms: int = 1000) -> complex:
        """Compute ζ(s) via series (for Re(s) > 1)."""
        return sum(1/n**s for n in range(1, terms + 1))


# Demonstration
toolkit = ComplexAnalysisToolkit()

print("=" * 60)
print("COMPLEX ANALYSIS TOOLKIT DEMONSTRATION")
print("=" * 60)

# Test 1: Basic contour integral
print("\n1. Contour Integral ∮ dz/(z² + 1) around |z| = 2")
f1 = lambda z: 1 / (z**2 + 1)
path1 = lambda t: 2 * np.exp(1j * t)
result1 = toolkit.contour_integral(f1, path1, (0, 2*np.pi))
print(f"   Result: {result1:.6f}")
print(f"   Expected: 2πi·(1/2i + 1/(-2i)) = π·i·0 = 0... wait")
print(f"   Actually: 2πi·(Res at i + Res at -i) = 2πi·(1/2i - 1/2i) = 0")

# Test 2: Residue computation
print("\n2. Residue of e^z/z² at z = 0")
f2 = lambda z: np.exp(z) / z**2
res2 = toolkit.residue_at_point(f2, 0, radius=0.5)
print(f"   Residue: {res2:.6f}")
print(f"   Expected: 1 (from e^z = 1 + z + z²/2 + ...)")

# Test 3: Real integral
print("\n3. ∫_{-∞}^{∞} dx/(x⁴ + 1)")
P3 = np.poly1d([1])
Q3 = np.poly1d([1, 0, 0, 0, 1])
result3 = toolkit.rational_integral(P3, Q3)
print(f"   Result: {result3:.6f}")
print(f"   Expected: π/√2 = {np.pi/np.sqrt(2):.6f}")
```

---

## Lab 2: Green's Function Simulator

```python
"""
Quantum Green's Function Simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class QuantumGreensFunctions:
    """Simulate quantum Green's functions and propagators."""

    def __init__(self, m=1, hbar=1):
        self.m = m
        self.hbar = hbar

    def free_particle_G(self, x, xp, E, epsilon=1e-6):
        """1D free particle retarded Green's function."""
        E_c = E + 1j * epsilon
        k = np.sqrt(2 * self.m * E_c) / self.hbar
        return -1j * self.m / (self.hbar**2 * k) * np.exp(1j * k * np.abs(x - xp))

    def harmonic_oscillator_G(self, E, n_states=50, omega=1, epsilon=1e-6):
        """Harmonic oscillator Green's function (diagonal)."""
        G = 0
        for n in range(n_states):
            E_n = self.hbar * omega * (n + 0.5)
            G += 1 / (E - E_n + 1j * epsilon)
        return G

    def spectral_function(self, G_func, E_values):
        """A(E) = -Im G(E) / π"""
        return -np.array([G_func(E).imag for E in E_values]) / np.pi

    def propagator(self, G_func, x, xp, t_values, E_range=(-5, 20), n_E=500):
        """Time-dependent propagator from Green's function."""
        E = np.linspace(E_range[0], E_range[1], n_E)
        K = []
        for t in t_values:
            integrand = np.array([G_func(x, xp, Ei) * np.exp(-1j * Ei * t / self.hbar)
                                 for Ei in E])
            K.append(np.trapz(integrand, E) / (2 * np.pi * self.hbar))
        return np.array(K)


# Demonstration
QGF = QuantumGreensFunctions()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Free particle spectral function
E = np.linspace(0.01, 5, 200)
A_free = QGF.spectral_function(lambda E: QGF.free_particle_G(0, 0, E), E)

axes[0, 0].plot(E, A_free, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Energy E')
axes[0, 0].set_ylabel('A(E)')
axes[0, 0].set_title('Free Particle Spectral Function')
axes[0, 0].grid(True, alpha=0.3)

# Harmonic oscillator spectral function
E_ho = np.linspace(0, 10, 500)
A_ho = QGF.spectral_function(lambda E: QGF.harmonic_oscillator_G(E, omega=1), E_ho)

axes[0, 1].plot(E_ho, A_ho, 'r-', linewidth=2)
for n in range(10):
    E_n = (n + 0.5)
    axes[0, 1].axvline(x=E_n, color='g', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Energy E')
axes[0, 1].set_ylabel('A(E)')
axes[0, 1].set_title('Harmonic Oscillator Spectral Function')
axes[0, 1].set_xlim(0, 10)
axes[0, 1].grid(True, alpha=0.3)

# Spatial dependence of free particle G
x = np.linspace(-5, 5, 200)
for E in [0.5, 1, 2, 4]:
    G = QGF.free_particle_G(x, 0, E)
    axes[1, 0].plot(x, np.abs(G), label=f'E={E}')

axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('|G(x, 0; E)|')
axes[1, 0].set_title('Spatial Dependence of Green\'s Function')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Complex E plane structure
E_real = np.linspace(-1, 5, 100)
E_imag = np.linspace(-1, 1, 100)
E_R, E_I = np.meshgrid(E_real, E_imag)
E_complex = E_R + 1j * E_I

G_complex = QGF.free_particle_G(0, 0, E_complex)
axes[1, 1].contourf(E_R, E_I, np.log10(np.abs(G_complex) + 1e-10),
                   levels=50, cmap='viridis')
axes[1, 1].axhline(y=0, color='r', linewidth=2, label='Branch cut')
axes[1, 1].set_xlabel('Re(E)')
axes[1, 1].set_ylabel('Im(E)')
axes[1, 1].set_title('Analytic Structure in Complex E Plane')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('greens_function_lab.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Lab 3: Scattering Theory Simulator

```python
"""
S-Matrix and Scattering Simulations
"""

import numpy as np
import matplotlib.pyplot as plt

class ScatteringSimulator:
    """Simulate quantum scattering using complex analysis."""

    def __init__(self, m=1, hbar=1):
        self.m = m
        self.hbar = hbar

    def S_delta_potential(self, k, strength):
        """S-matrix for delta function potential V = λδ(x)."""
        return (1 + 1j * self.m * strength / (self.hbar**2 * k)) / \
               (1 - 1j * self.m * strength / (self.hbar**2 * k))

    def phase_shift(self, S):
        """Extract phase shift from S-matrix."""
        return np.angle(S) / 2

    def cross_section(self, k, S):
        """Total cross section from S-matrix."""
        return 4 * np.pi / k**2 * np.sin(self.phase_shift(S))**2

    def find_bound_state(self, strength):
        """Find bound state energy for attractive delta potential."""
        if strength >= 0:
            return None  # No bound state
        kappa = self.m * np.abs(strength) / self.hbar**2
        E_bound = -self.hbar**2 * kappa**2 / (2 * self.m)
        return E_bound

    def breit_wigner(self, E, E_R, Gamma):
        """Breit-Wigner resonance cross section."""
        return Gamma**2 / 4 / ((E - E_R)**2 + Gamma**2 / 4)


# Demonstration
SS = ScatteringSimulator()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# S-matrix in complex k-plane
k_real = np.linspace(-3, 3, 100)
k_imag = np.linspace(-1, 2, 100)
K_R, K_I = np.meshgrid(k_real, k_imag)
K = K_R + 1j * K_I

# Attractive delta potential
lambda_val = -2
S = SS.S_delta_potential(K, lambda_val)

axes[0, 0].contourf(K_R, K_I, np.log10(np.abs(S) + 1e-10), levels=50, cmap='viridis')
kappa = np.abs(lambda_val) / 2
axes[0, 0].plot([0], [kappa], 'r*', markersize=15, label=f'Bound state at k=i·{kappa:.2f}')
axes[0, 0].axhline(y=0, color='w', linestyle='-', linewidth=0.5)
axes[0, 0].set_xlabel('Re(k)')
axes[0, 0].set_ylabel('Im(k)')
axes[0, 0].set_title('|S(k)| for Attractive Delta Potential')
axes[0, 0].legend()

# Phase shift
k_real_pos = np.linspace(0.01, 5, 200)
S_real = SS.S_delta_potential(k_real_pos, lambda_val)
delta = SS.phase_shift(S_real)

axes[0, 1].plot(k_real_pos, delta / np.pi, 'b-', linewidth=2)
axes[0, 1].axhline(y=1, color='r', linestyle='--', label='δ/π = 1 (Levinson)')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('δ/π')
axes[0, 1].set_title('Phase Shift (Levinson\'s Theorem Verification)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Breit-Wigner resonance
E = np.linspace(0, 20, 500)
E_R, Gamma = 10, 2
sigma = SS.breit_wigner(E, E_R, Gamma)

axes[1, 0].plot(E, sigma, 'g-', linewidth=2)
axes[1, 0].axvline(x=E_R, color='r', linestyle='--', label=f'E_R = {E_R}')
axes[1, 0].axvline(x=E_R - Gamma/2, color='b', linestyle=':', alpha=0.5)
axes[1, 0].axvline(x=E_R + Gamma/2, color='b', linestyle=':', alpha=0.5, label=f'Γ = {Gamma}')
axes[1, 0].fill_between(E, 0, sigma, where=(np.abs(E-E_R) < Gamma/2), alpha=0.3)
axes[1, 0].set_xlabel('Energy')
axes[1, 0].set_ylabel('Cross section')
axes[1, 0].set_title('Breit-Wigner Resonance')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Argand diagram
theta = np.linspace(0, 2*np.pi, 100)
axes[1, 1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
axes[1, 1].plot(S_real.real, S_real.imag, 'b-', linewidth=2, label='S(k) trajectory')
axes[1, 1].plot([S_real[0].real], [S_real[0].imag], 'go', markersize=10, label='k→0')
axes[1, 1].plot([S_real[-1].real], [S_real[-1].imag], 'ro', markersize=10, label='k→∞')
axes[1, 1].set_xlabel('Re(S)')
axes[1, 1].set_ylabel('Im(S)')
axes[1, 1].set_title('S-matrix Argand Diagram')
axes[1, 1].legend()
axes[1, 1].axis('equal')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scattering_lab.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Bound state energy: {SS.find_bound_state(lambda_val):.4f}")
print(f"Phase shift at k→0: {delta[0]/np.pi:.4f}π")
print(f"Phase shift at k→∞: {delta[-1]/np.pi:.4f}π")
print(f"Levinson: Δδ/π = {(delta[0] - delta[-1])/np.pi:.4f} (expect 1 for 1 bound state)")
```

---

## Summary

Today's computational lab integrates all techniques from Month 7:

### Implemented Tools

| Tool | Purpose |
|------|---------|
| `ComplexAnalysisToolkit` | General contour integration |
| `QuantumGreensFunctions` | Green's functions and propagators |
| `ScatteringSimulator` | S-matrix and resonances |
| Visualization routines | Domain coloring, pole plots |

### Key Applications

1. **Contour integration** for any path
2. **Residue computation** numerically
3. **Real integrals** via semicircular contours
4. **Green's functions** with correct analyticity
5. **Scattering theory** with Levinson verification

---

## Preview: Day 196

Tomorrow: **Month 7 Comprehensive Review**
- Complete concept synthesis
- Comprehensive problem sets
- Self-assessment
- Preparation for Month 8

---

*"Good code is its own best documentation."*
— Steve McConnell
