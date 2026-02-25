# Day 139: Computational Lab — Advanced Complex Analysis Tools

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Part 1: Complete Analysis Toolkit |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Part 2: Physics Applications |
| Evening | 6:00 PM - 7:30 PM | 1.5 hours | Part 3: Advanced Projects |

**Total Study Time: 8 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Build a comprehensive complex analysis library
2. Automate residue calculations and contour integration
3. Implement zero-finding using the argument principle
4. Apply complex analysis to physics problems
5. Create publication-quality visualizations

---

## 💻 Part 1: Complete Analysis Toolkit (3.5 hours)

```python
"""
Complex Analysis Toolkit
========================
A comprehensive library for complex analysis computations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy import integrate, optimize
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

class ComplexFunction:
    """Represents a complex function with analysis methods."""
    
    def __init__(self, f, name="f"):
        self.f = f
        self.name = name
    
    def __call__(self, z):
        return self.f(z)
    
    def derivative(self, z, h=1e-8):
        """Numerical derivative using central difference."""
        return (self.f(z + h) - self.f(z - h)) / (2 * h)
    
    def find_zeros(self, x_range, y_range, resolution=50, tol=1e-6):
        """Find zeros in a rectangular region."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        
        zeros = []
        for xi in x:
            for yi in y:
                z0 = xi + 1j * yi
                try:
                    # Use Newton's method
                    z = z0
                    for _ in range(50):
                        fz = self.f(z)
                        fpz = self.derivative(z)
                        if abs(fpz) < 1e-12:
                            break
                        z_new = z - fz / fpz
                        if abs(z_new - z) < tol:
                            # Check if this is a new zero
                            is_new = True
                            for z_found in zeros:
                                if abs(z_new - z_found) < 10 * tol:
                                    is_new = False
                                    break
                            if is_new and abs(self.f(z_new)) < tol:
                                zeros.append(z_new)
                            break
                        z = z_new
                except:
                    pass
        return np.array(zeros)
    
    def find_poles(self, x_range, y_range, resolution=50, tol=1e-6):
        """Find poles by looking for zeros of 1/f."""
        g = ComplexFunction(lambda z: 1/self.f(z) if abs(self.f(z)) > tol else 1e10)
        return g.find_zeros(x_range, y_range, resolution, tol)
    
    def residue(self, z0, order=1, eps=1e-6):
        """Compute residue at z0."""
        if order == 1:
            # Simple pole: limit of (z-z0)*f(z)
            return (z0 + eps - z0) * self.f(z0 + eps)
        else:
            # Higher order: use contour integral
            return self.contour_integral(z0, eps) / (2 * np.pi * 1j)
    
    def contour_integral(self, center, radius, n_points=1000):
        """Compute ∮ f(z) dz around a circle."""
        t = np.linspace(0, 2*np.pi, n_points)
        z = center + radius * np.exp(1j * t)
        dz = 1j * radius * np.exp(1j * t)
        integrand = self.f(z) * dz
        return np.trapz(integrand, t)
    
    def winding_number(self, center, radius, n_points=1000):
        """Compute winding number around a point."""
        t = np.linspace(0, 2*np.pi, n_points)
        z = center + radius * np.exp(1j * t)
        w = self.f(z)
        
        darg = np.diff(np.unwrap(np.angle(w)))
        return np.sum(darg) / (2 * np.pi)
    
    def count_zeros_minus_poles(self, center, radius):
        """Use argument principle to count Z - P."""
        g = ComplexFunction(lambda z: self.derivative(z) / self.f(z))
        integral = g.contour_integral(center, radius)
        return int(round(np.real(integral / (2 * np.pi * 1j))))


class ContourIntegration:
    """Tools for contour integration."""
    
    @staticmethod
    def semicircle_upper(f, R, n_points=1000):
        """Integrate over upper semicircle."""
        t = np.linspace(0, np.pi, n_points)
        z = R * np.exp(1j * t)
        dz = 1j * R * np.exp(1j * t)
        integrand = f(z) * dz
        return np.trapz(integrand, t)
    
    @staticmethod
    def real_line_integral(f, R, n_points=2000):
        """Integrate along real line from -R to R."""
        x = np.linspace(-R, R, n_points)
        return np.trapz(f(x), x)
    
    @staticmethod
    def real_improper_integral(f, n_points=10000):
        """Compute ∫_{-∞}^{∞} f(x) dx numerically."""
        result, error = integrate.quad(lambda x: np.real(f(x)), -np.inf, np.inf)
        return result, error
    
    @staticmethod
    def trig_integral(R_func, n_points=1000):
        """
        Compute ∫_0^{2π} R(cos θ, sin θ) dθ
        via unit circle contour.
        """
        t = np.linspace(0, 2*np.pi, n_points)
        z = np.exp(1j * t)
        cos_t = np.real(z)
        sin_t = np.imag(z)
        integrand = R_func(cos_t, sin_t)
        return np.trapz(integrand, t)


class RealIntegrals:
    """Evaluate real integrals using residue methods."""
    
    @staticmethod
    def rational(p_coeffs, q_coeffs):
        """
        Compute ∫_{-∞}^{∞} p(x)/q(x) dx
        where p, q are polynomials given by coefficient lists.
        """
        p = np.poly1d(p_coeffs)
        q = np.poly1d(q_coeffs)
        
        if len(q_coeffs) < len(p_coeffs) + 2:
            raise ValueError("Need deg(q) >= deg(p) + 2 for convergence")
        
        # Find poles (roots of q)
        poles = np.roots(q_coeffs)
        
        # Select poles in upper half-plane
        upper_poles = [pole for pole in poles if np.imag(pole) > 1e-10]
        
        # Compute residues
        total_residue = 0
        for pole in upper_poles:
            # Residue at simple pole of p/q
            residue = p(pole) / np.polyval(np.polyder(q_coeffs), pole)
            total_residue += residue
        
        return 2 * np.pi * 1j * total_residue
    
    @staticmethod
    def fourier_type(f, a):
        """
        Compute ∫_{-∞}^{∞} f(x) e^{iax} dx for a > 0
        using Jordan's lemma.
        """
        if a <= 0:
            raise ValueError("Need a > 0 for upper half-plane closure")
        
        # Numerical integration
        def integrand_real(x):
            return np.real(f(x) * np.exp(1j * a * x))
        def integrand_imag(x):
            return np.imag(f(x) * np.exp(1j * a * x))
        
        real_part, _ = integrate.quad(integrand_real, -100, 100)
        imag_part, _ = integrate.quad(integrand_imag, -100, 100)
        
        return real_part + 1j * imag_part


class Visualization:
    """Visualization tools for complex functions."""
    
    @staticmethod
    def domain_coloring(f, x_range=(-2, 2), y_range=(-2, 2), 
                        resolution=500, title="f(z)"):
        """Create domain coloring plot."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        with np.errstate(all='ignore'):
            W = f(Z)
            W = np.where(np.isfinite(W), W, np.nan)
        
        H = (np.angle(W) + np.pi) / (2 * np.pi)
        S = np.ones_like(H) * 0.9
        V = 1 - 1/(1 + np.log1p(np.abs(W))/3)
        
        H = np.nan_to_num(H, nan=0)
        S = np.nan_to_num(S, nan=0)
        V = np.nan_to_num(V, nan=0)
        
        RGB = hsv_to_rgb(np.dstack([H, S, V]))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(RGB, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                  origin='lower')
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title(title)
        
        return fig, ax
    
    @staticmethod  
    def contour_plot(f, contour_func, n_points=500, title="Contour"):
        """Plot a contour and its image under f."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        t = np.linspace(0, 1, n_points)
        z = contour_func(t)
        w = f(z)
        
        axes[0].plot(np.real(z), np.imag(z), 'b-', lw=2)
        axes[0].set_xlabel('Re(z)')
        axes[0].set_ylabel('Im(z)')
        axes[0].set_title('z-plane (domain)')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(np.real(w), np.imag(w), 'r-', lw=2)
        axes[1].set_xlabel('Re(w)')
        axes[1].set_ylabel('Im(w)')
        axes[1].set_title('w-plane (image)')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        return fig, axes


# Demonstration
print("=" * 70)
print("COMPLEX ANALYSIS TOOLKIT DEMONSTRATION")
print("=" * 70)

# Example 1: Find zeros and poles
print("\n1. Finding zeros and poles of f(z) = (z²-1)/(z²+1)")
f1 = ComplexFunction(lambda z: (z**2 - 1)/(z**2 + 1), "f(z)")
zeros = f1.find_zeros((-2, 2), (-2, 2))
poles = f1.find_poles((-2, 2), (-2, 2))
print(f"   Zeros: {[f'{z:.4f}' for z in zeros]}")
print(f"   Poles: {[f'{p:.4f}' for p in poles]}")

# Example 2: Contour integral
print("\n2. Contour integral: ∮_{|z|=2} dz/(z-1)")
f2 = ComplexFunction(lambda z: 1/(z-1))
integral = f2.contour_integral(0, 2)
print(f"   Result: {integral:.6f}")
print(f"   Expected: 2πi = {2*np.pi*1j:.6f}")

# Example 3: Real integral via residues
print("\n3. Real integral: ∫_{-∞}^{∞} dx/(1+x²)")
result = RealIntegrals.rational([1], [1, 0, 1])  # 1/(x²+1)
print(f"   Result: {np.real(result):.6f}")
print(f"   Expected: π = {np.pi:.6f}")

# Example 4: Winding number
print("\n4. Winding number of z² around origin (|z|=1)")
f4 = ComplexFunction(lambda z: z**2)
wn = f4.winding_number(0, 1)
print(f"   Winding number: {wn:.2f}")
print(f"   Expected: 2")

# Visualization
fig, ax = Visualization.domain_coloring(
    lambda z: (z**2 - 1)/(z**2 + 1),
    title="f(z) = (z²-1)/(z²+1)\nZeros at ±1 (dark), Poles at ±i (bright)"
)
plt.savefig('toolkit_demo.png', dpi=150)
plt.show()
```

---

## 💻 Part 2: Physics Applications (3 hours)

```python
"""
Physics Applications of Complex Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import jv  # Bessel functions

class QuantumMechanics:
    """Complex analysis applications in quantum mechanics."""
    
    @staticmethod
    def free_particle_propagator(x, t, m=1, hbar=1):
        """
        Free particle propagator K(x,t) = ⟨x|e^{-iHt/ℏ}|0⟩
        Computed via contour integration.
        """
        if t == 0:
            return np.inf if x == 0 else 0
        
        # K(x,t) = √(m/(2πiℏt)) exp(imx²/(2ℏt))
        prefactor = np.sqrt(m / (2 * np.pi * 1j * hbar * t))
        phase = 1j * m * x**2 / (2 * hbar * t)
        return prefactor * np.exp(phase)
    
    @staticmethod
    def harmonic_oscillator_propagator(x, x0, t, omega=1, m=1, hbar=1):
        """
        Harmonic oscillator propagator via Mehler's formula.
        """
        prefactor = np.sqrt(m * omega / (2 * np.pi * 1j * hbar * np.sin(omega * t)))
        exponent = (1j * m * omega / (2 * hbar * np.sin(omega * t))) * \
                   ((x**2 + x0**2) * np.cos(omega * t) - 2 * x * x0)
        return prefactor * np.exp(exponent)
    
    @staticmethod
    def green_function_1d(E, V, x_range, n_points=100):
        """
        Compute Green's function G(x,x';E) for 1D potential.
        Uses spectral representation with regularization.
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        dx = x[1] - x[0]
        
        # Discretize Hamiltonian
        H = np.diag(-2 * np.ones(n_points)) + \
            np.diag(np.ones(n_points-1), 1) + \
            np.diag(np.ones(n_points-1), -1)
        H *= -1 / (2 * dx**2)
        H += np.diag(V(x))
        
        # G(E) = (E - H + iε)^{-1}
        eps = 0.01
        G = np.linalg.inv((E + 1j*eps) * np.eye(n_points) - H)
        
        return x, G
    
    @staticmethod
    def spectral_density(H, E_range, eta=0.1):
        """
        Compute spectral density ρ(E) = -Im[Tr G(E+iη)]/π
        """
        E_vals = np.linspace(E_range[0], E_range[1], 500)
        rho = np.zeros_like(E_vals)
        
        n = H.shape[0]
        for i, E in enumerate(E_vals):
            G = np.linalg.inv((E + 1j*eta) * np.eye(n) - H)
            rho[i] = -np.imag(np.trace(G)) / np.pi
        
        return E_vals, rho


class ScatteringTheory:
    """Scattering theory applications."""
    
    @staticmethod
    def breit_wigner(E, E0, Gamma):
        """Breit-Wigner resonance amplitude."""
        return Gamma / 2 / (E - E0 + 1j * Gamma / 2)
    
    @staticmethod
    def s_matrix_pole_analysis(poles, energies):
        """
        Analyze S-matrix pole structure.
        poles: list of (E_pole, residue) tuples
        """
        S = np.ones_like(energies, dtype=complex)
        
        for E_pole, res in poles:
            S *= (energies - np.conj(E_pole)) / (energies - E_pole)
        
        return S
    
    @staticmethod
    def phase_shift(S):
        """Extract phase shift from S-matrix: S = e^{2iδ}."""
        return np.angle(S) / 2
    
    @staticmethod
    def cross_section(S, k):
        """Compute cross section σ = (4π/k²)|f|² where S = 1 + 2ikf."""
        f = (S - 1) / (2j * k)
        return 4 * np.pi * np.abs(f)**2


class StatisticalMechanics:
    """Statistical mechanics applications."""
    
    @staticmethod
    def partition_function_contour(eigenvalues, beta):
        """
        Compute Z = Σ e^{-βE_n} via contour integral
        Z = -1/(2πi) ∮ e^{-βE} Tr[G(E)] dE
        """
        return np.sum(np.exp(-beta * eigenvalues))
    
    @staticmethod
    def matsubara_sum(f, beta, n_max=100):
        """
        Compute (1/β) Σ_n f(iω_n) for bosonic Matsubara frequencies.
        ω_n = 2πn/β
        """
        total = 0
        for n in range(-n_max, n_max + 1):
            omega_n = 2 * np.pi * n / beta
            total += f(1j * omega_n)
        return total / beta


# Demonstrations
print("\n" + "=" * 70)
print("PHYSICS APPLICATIONS")
print("=" * 70)

# 1. Free particle propagator
print("\n1. Free Particle Propagator")
x = np.linspace(-5, 5, 200)
t_values = [0.1, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for t in t_values:
    K = QuantumMechanics.free_particle_propagator(x, t)
    axes[0].plot(x, np.abs(K)**2, label=f't = {t}')
    axes[1].plot(x, np.real(K), label=f't = {t}')

axes[0].set_xlabel('x')
axes[0].set_ylabel('|K(x,t)|²')
axes[0].set_title('Free Particle: Probability Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('x')
axes[1].set_ylabel('Re[K(x,t)]')
axes[1].set_title('Free Particle: Real Part')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('free_particle_propagator.png', dpi=150)
plt.show()

# 2. Spectral density from Green's function
print("\n2. Spectral Density from Green's Function")

# Create a simple Hamiltonian (particle in a box)
n = 50
dx = 1.0 / n
H = np.diag(-2 * np.ones(n)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
H *= -1 / (2 * dx**2)

# Exact eigenvalues
eigenvalues = np.linalg.eigvalsh(H)

# Spectral density
E_range = (0, eigenvalues[-1] * 1.2)
E_vals, rho = QuantumMechanics.spectral_density(H, E_range, eta=0.5)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(E_vals, rho, 'b-', lw=2, label='Spectral density ρ(E)')
for ev in eigenvalues[:10]:
    ax.axvline(x=ev, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Energy E')
ax.set_ylabel('ρ(E)')
ax.set_title('Spectral Density: Peaks at Eigenvalues')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('spectral_density.png', dpi=150)
plt.show()

# 3. Breit-Wigner resonance
print("\n3. Breit-Wigner Resonance")

E = np.linspace(0, 5, 500)
E0, Gamma = 2.5, 0.3

A = ScatteringTheory.breit_wigner(E, E0, Gamma)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(E, np.abs(A)**2, 'b-', lw=2)
axes[0].axvline(x=E0, color='r', linestyle='--', label=f'E₀ = {E0}')
axes[0].axvline(x=E0-Gamma/2, color='g', linestyle=':', alpha=0.7)
axes[0].axvline(x=E0+Gamma/2, color='g', linestyle=':', alpha=0.7, 
               label=f'FWHM = Γ = {Gamma}')
axes[0].set_xlabel('Energy E')
axes[0].set_ylabel('|A|²')
axes[0].set_title('Breit-Wigner Cross Section')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Phase shift
S = 1 + 2j * A
delta = ScatteringTheory.phase_shift(S)
axes[1].plot(E, delta, 'r-', lw=2)
axes[1].axhline(y=np.pi/2, color='k', linestyle='--', alpha=0.5)
axes[1].axvline(x=E0, color='b', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Energy E')
axes[1].set_ylabel('Phase shift δ')
axes[1].set_title('Phase Shift Through Resonance')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('breit_wigner.png', dpi=150)
plt.show()

# 4. Kramers-Kronig relations
print("\n4. Kramers-Kronig Relations")

def susceptibility(omega, omega0=2, gamma=0.3):
    """Model susceptibility (damped oscillator)."""
    return 1 / (omega0**2 - omega**2 - 1j * gamma * omega)

omega = np.linspace(0.1, 5, 500)
chi = susceptibility(omega)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(omega, np.real(chi), 'b-', lw=2, label="χ'(ω) - real part")
ax.plot(omega, np.imag(chi), 'r-', lw=2, label="χ''(ω) - imaginary part")
ax.axvline(x=2, color='g', linestyle='--', alpha=0.5, label='Resonance ω₀')
ax.set_xlabel('Frequency ω')
ax.set_ylabel('χ(ω)')
ax.set_title('Susceptibility: Real and Imaginary Parts\n(Related by Kramers-Kronig)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('kramers_kronig_susceptibility.png', dpi=150)
plt.show()
```

---

## 💻 Part 3: Advanced Projects (1.5 hours)

```python
"""
Advanced Complex Analysis Projects
"""

# Project 1: Riemann Zeta Function Visualization
print("\n" + "=" * 70)
print("ADVANCED PROJECT: RIEMANN ZETA FUNCTION")
print("=" * 70)

def zeta_approx(s, n_terms=1000):
    """Approximate ζ(s) for Re(s) > 1."""
    return sum(1/n**s for n in range(1, n_terms+1))

def zeta_continued(s, n_terms=100):
    """
    Analytic continuation using Dirichlet eta function:
    ζ(s) = η(s) / (1 - 2^{1-s})
    where η(s) = Σ (-1)^{n-1}/n^s
    """
    eta = sum((-1)**(n-1) / n**s for n in range(1, n_terms+1))
    return eta / (1 - 2**(1-s))

# Visualize zeta in the critical strip
x = np.linspace(-2, 4, 200)
y = np.linspace(-30, 30, 300)
X, Y = np.meshgrid(x, y)
S = X + 1j * Y

Z = np.zeros_like(S, dtype=complex)
for i in range(len(y)):
    for j in range(len(x)):
        s = S[i, j]
        if np.real(s) > 1:
            Z[i, j] = zeta_approx(s, 500)
        else:
            Z[i, j] = zeta_continued(s, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Phase plot
phase = np.angle(Z)
im0 = axes[0].imshow(phase, extent=[x[0], x[-1], y[0], y[-1]], 
                     origin='lower', cmap='hsv', aspect='auto')
axes[0].axvline(x=0.5, color='white', linestyle='--', label='Critical line')
axes[0].set_xlabel('Re(s)')
axes[0].set_ylabel('Im(s)')
axes[0].set_title('Phase of ζ(s)')
axes[0].legend()
plt.colorbar(im0, ax=axes[0])

# Magnitude (log scale)
mag = np.log10(np.abs(Z) + 1e-10)
im1 = axes[1].imshow(mag, extent=[x[0], x[-1], y[0], y[-1]], 
                     origin='lower', cmap='hot', aspect='auto',
                     vmin=-2, vmax=2)
axes[1].axvline(x=0.5, color='cyan', linestyle='--', label='Critical line')
axes[1].set_xlabel('Re(s)')
axes[1].set_ylabel('Im(s)')
axes[1].set_title('log₁₀|ζ(s)|')
axes[1].legend()
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig('riemann_zeta.png', dpi=150)
plt.show()

# Project 2: Conformal Mapping Gallery
print("\n" + "=" * 70)
print("CONFORMAL MAPPING GALLERY")
print("=" * 70)

def create_mapping_gallery():
    """Create gallery of conformal mappings."""
    
    mappings = [
        (lambda z: z**2, "w = z²"),
        (lambda z: np.exp(z), "w = eᶻ"),
        (lambda z: np.sin(z), "w = sin(z)"),
        (lambda z: (z-1)/(z+1), "w = (z-1)/(z+1)"),
        (lambda z: z + 1/z, "w = z + 1/z (Joukowski)"),
        (lambda z: np.log(z), "w = log(z)"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Create grid in z-plane
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    
    for ax, (f, title) in zip(axes.flat, mappings):
        # Draw grid lines and their images
        for xi in np.linspace(-2, 2, 11):
            z = xi + 1j * y
            with np.errstate(all='ignore'):
                w = f(z)
            ax.plot(np.real(w), np.imag(w), 'b-', lw=0.5, alpha=0.7)
        
        for yi in np.linspace(-2, 2, 11):
            z = x + 1j * yi
            with np.errstate(all='ignore'):
                w = f(z)
            ax.plot(np.real(w), np.imag(w), 'r-', lw=0.5, alpha=0.7)
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Conformal Mapping Gallery', fontsize=14)
    plt.tight_layout()
    plt.savefig('conformal_gallery.png', dpi=150)
    plt.show()

create_mapping_gallery()

print("\n" + "=" * 70)
print("COMPUTATIONAL LAB COMPLETE!")
print("=" * 70)
```

---

## 📝 Summary

### Tools Built Today

| Tool | Purpose |
|------|---------|
| ComplexFunction | Analysis of complex functions |
| ContourIntegration | Numerical contour integrals |
| RealIntegrals | Real integrals via residues |
| Visualization | Domain coloring, mappings |
| QuantumMechanics | Propagators, Green's functions |
| ScatteringTheory | S-matrix, phase shifts |

### Physics Applications Explored

1. Free particle propagator
2. Spectral density from Green's functions
3. Breit-Wigner resonances
4. Kramers-Kronig relations
5. Riemann zeta function visualization

---

## ✅ Daily Checklist

- [ ] Build comprehensive analysis toolkit
- [ ] Implement automated residue calculation
- [ ] Visualize the Riemann zeta function
- [ ] Compute physics propagators
- [ ] Explore scattering theory applications
- [ ] Create conformal mapping gallery
- [ ] Save all code for future use

---

## 🔮 Preview: Day 140

Tomorrow we consolidate Week 20 with a comprehensive **review and problem set**, preparing for the transition to Classical Mechanics (Month 6)!
