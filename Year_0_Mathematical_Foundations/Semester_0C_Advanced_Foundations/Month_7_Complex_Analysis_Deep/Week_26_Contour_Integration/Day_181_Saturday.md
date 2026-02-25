# Day 181: Computational Lab — Contour Integration Mastery

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Lab 1-3: Fundamental Techniques |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Lab 4-6: Physics Applications |
| Evening | 7:00 PM - 9:00 PM | 2 hours | Challenge Problems & Synthesis |

**Total Study Time: 8 hours**

---

## Learning Objectives

By the end of Day 181, you will be able to:

1. Implement contour integration numerically with high precision
2. Visualize contours and integrand behavior
3. Verify analytical results computationally
4. Apply contour methods to physics problems
5. Solve challenging integration problems
6. Build intuition for choosing optimal contours

---

## Lab 1: Comprehensive Contour Integral Calculator

```python
"""
Comprehensive Contour Integration Library
Supports various contour types and integration methods
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

class ContourIntegrator:
    """A class for computing and visualizing complex contour integrals."""

    def __init__(self, n_points: int = 10000):
        self.n_points = n_points

    def integrate(self, f: Callable, path: Callable,
                  t_range: Tuple[float, float]) -> complex:
        """
        Compute ∮_C f(z) dz along parametrized path z(t).

        Parameters:
        -----------
        f : function z -> complex
        path : function t -> z(t)
        t_range : (t_start, t_end)

        Returns:
        --------
        Complex value of the integral
        """
        t = np.linspace(t_range[0], t_range[1], self.n_points)
        dt = t[1] - t[0]
        z = path(t)
        dz = np.gradient(z, dt, edge_order=2)

        integrand = f(z) * dz
        return np.trapz(integrand, t)

    def integrate_adaptive(self, f: Callable, path: Callable,
                          t_range: Tuple[float, float]) -> Tuple[complex, float]:
        """
        Adaptive integration using scipy for higher accuracy.
        """
        def real_part(t):
            z = path(t)
            dz_dt = (path(t + 1e-8) - path(t - 1e-8)) / (2e-8)
            return (f(z) * dz_dt).real

        def imag_part(t):
            z = path(t)
            dz_dt = (path(t + 1e-8) - path(t - 1e-8)) / (2e-8)
            return (f(z) * dz_dt).imag

        real_result, real_error = integrate.quad(real_part, t_range[0], t_range[1])
        imag_result, imag_error = integrate.quad(imag_part, t_range[0], t_range[1])

        return complex(real_result, imag_result), max(real_error, imag_error)

    @staticmethod
    def circle(center: complex = 0, radius: float = 1):
        """Return a circular path centered at `center` with given `radius`."""
        return lambda t: center + radius * np.exp(1j * t)

    @staticmethod
    def semicircle_upper(radius: float = 1):
        """Upper semicircle from -R to R through upper half-plane."""
        def path(t):
            if isinstance(t, np.ndarray):
                result = np.zeros_like(t, dtype=complex)
                # Real axis: t in [0, 1] maps to [-R, R]
                mask1 = t <= 1
                result[mask1] = -radius + 2*radius*t[mask1]
                # Semicircle: t in [1, 2] maps to angle 0 to π
                mask2 = t > 1
                theta = np.pi * (t[mask2] - 1)
                result[mask2] = radius * np.exp(1j * theta)
                return result
            else:
                if t <= 1:
                    return -radius + 2*radius*t
                else:
                    return radius * np.exp(1j * np.pi * (t - 1))
        return path

    @staticmethod
    def keyhole(R: float = 5, epsilon: float = 0.1):
        """Keyhole contour avoiding branch cut on positive real axis."""
        def path(t):
            # t in [0, 1]: upper edge from epsilon to R
            # t in [1, 2]: large circle from 0 to 2π
            # t in [2, 3]: lower edge from R to epsilon
            # t in [3, 4]: small circle from 2π to 0
            delta = epsilon / 10

            if isinstance(t, np.ndarray):
                result = np.zeros_like(t, dtype=complex)

                mask1 = (t >= 0) & (t < 1)
                x = epsilon + (R - epsilon) * t[mask1]
                result[mask1] = x + 1j * delta

                mask2 = (t >= 1) & (t < 2)
                theta = 2 * np.pi * (t[mask2] - 1)
                result[mask2] = R * np.exp(1j * theta)

                mask3 = (t >= 2) & (t < 3)
                x = R - (R - epsilon) * (t[mask3] - 2)
                result[mask3] = x - 1j * delta

                mask4 = (t >= 3) & (t <= 4)
                theta = 2 * np.pi * (1 - (t[mask4] - 3))
                result[mask4] = epsilon * np.exp(1j * theta)

                return result
            else:
                if t < 1:
                    return epsilon + (R - epsilon) * t + 1j * delta
                elif t < 2:
                    return R * np.exp(1j * 2 * np.pi * (t - 1))
                elif t < 3:
                    return R - (R - epsilon) * (t - 2) - 1j * delta
                else:
                    return epsilon * np.exp(1j * 2 * np.pi * (1 - (t - 3)))
        return path

    def visualize_contour(self, path: Callable, t_range: Tuple[float, float],
                         poles: List[complex] = None, title: str = ""):
        """Visualize the integration contour."""
        t = np.linspace(t_range[0], t_range[1], 1000)
        z = path(t)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(z.real, z.imag, 'b-', linewidth=2, label='Contour')

        # Add direction arrows
        for i in range(0, len(t)-1, len(t)//10):
            dx = z[i+1].real - z[i].real
            dy = z[i+1].imag - z[i].imag
            ax.arrow(z[i].real, z[i].imag, dx*0.5, dy*0.5,
                    head_width=0.1, head_length=0.05, fc='blue', ec='blue')

        if poles:
            for pole in poles:
                ax.plot(pole.real, pole.imag, 'r*', markersize=15)
            ax.plot([], [], 'r*', markersize=15, label='Poles')

        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title(title if title else 'Integration Contour')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        return fig, ax


# Demonstration
if __name__ == "__main__":
    CI = ContourIntegrator(n_points=50000)

    print("=" * 60)
    print("CONTOUR INTEGRATION DEMONSTRATIONS")
    print("=" * 60)

    # Test 1: ∮ dz/z = 2πi
    f1 = lambda z: 1/z
    path1 = CI.circle(0, 1)
    result1 = CI.integrate(f1, path1, (0, 2*np.pi))
    print(f"\n1. ∮ dz/z around unit circle:")
    print(f"   Computed: {result1:.6f}")
    print(f"   Expected: {2*np.pi*1j:.6f}")

    # Test 2: ∮ e^z/(z-1) dz = 2πi·e
    f2 = lambda z: np.exp(z)/(z-1)
    path2 = CI.circle(0, 2)
    result2 = CI.integrate(f2, path2, (0, 2*np.pi))
    print(f"\n2. ∮ e^z/(z-1) dz around |z|=2:")
    print(f"   Computed: {result2:.6f}")
    print(f"   Expected: {2*np.pi*1j*np.e:.6f}")

    # Test 3: Visualize semicircular contour
    fig, ax = CI.visualize_contour(
        CI.semicircle_upper(3), (0, 2),
        poles=[1j, -1j],
        title="Semicircular Contour for Rational Function Integrals"
    )
    plt.savefig('semicircular_contour.png', dpi=150, bbox_inches='tight')

    plt.show()
```

---

## Lab 2: Residue Calculator and Verification

```python
"""
Automatic residue computation and verification.
"""

import numpy as np
from scipy.misc import derivative
import sympy as sp

class ResidueCalculator:
    """Compute residues numerically and symbolically."""

    @staticmethod
    def residue_numerical(f, z0, order=1, epsilon=1e-6):
        """
        Compute residue at z0 numerically.

        For simple pole: Res = lim_{z→z0} (z-z0)f(z)
        For order n pole: Res = lim_{z→z0} (1/(n-1)!) d^{n-1}/dz^{n-1}[(z-z0)^n f(z)]
        """
        if order == 1:
            # Simple pole
            g = lambda z: (z - z0) * f(z)
            return g(z0 + epsilon)
        else:
            # Higher order pole
            g = lambda z: (z - z0)**order * f(z)
            # Compute (order-1)th derivative
            result = g(z0 + epsilon)
            for _ in range(order - 1):
                result = derivative(lambda x: g(complex(x, epsilon)),
                                   z0.real, dx=1e-6)
            return result / np.math.factorial(order - 1)

    @staticmethod
    def residue_contour(f, z0, radius=0.1, n_points=10000):
        """Compute residue via contour integral: Res = (1/2πi)∮f(z)dz."""
        t = np.linspace(0, 2*np.pi, n_points)
        z = z0 + radius * np.exp(1j * t)
        dz = 1j * radius * np.exp(1j * t)
        integrand = f(z) * dz
        integral = np.trapz(integrand, t)
        return integral / (2 * np.pi * 1j)

    @staticmethod
    def residue_symbolic(expr_str, var='z', point_str='0'):
        """Compute residue symbolically using SymPy."""
        z = sp.Symbol(var)
        expr = sp.sympify(expr_str)
        point = sp.sympify(point_str)
        return sp.residue(expr, z, point)


# Demonstration
print("=" * 60)
print("RESIDUE CALCULATIONS")
print("=" * 60)

RC = ResidueCalculator()

# Example 1: Simple pole
print("\n1. f(z) = 1/(z-i) at z = i")
f1 = lambda z: 1/(z - 1j)
res1_contour = RC.residue_contour(f1, 1j)
print(f"   Contour method: {res1_contour:.6f}")
print(f"   Expected: 1")

# Example 2: Double pole
print("\n2. f(z) = z/(z-1)² at z = 1")
f2 = lambda z: z/(z - 1)**2
res2_contour = RC.residue_contour(f2, 1, radius=0.5)
print(f"   Contour method: {res2_contour:.6f}")
print(f"   Expected: 1 (by L'Hôpital or formula)")

# Example 3: e^z/z² at z = 0 (pole of order 2)
print("\n3. f(z) = e^z/z² at z = 0")
f3 = lambda z: np.exp(z)/z**2
res3_contour = RC.residue_contour(f3, 0, radius=0.1)
print(f"   Contour method: {res3_contour:.6f}")
print(f"   Expected: 1 (coefficient of 1/z in Laurent series)")

# Symbolic computation
print("\n4. Symbolic residue of 1/(z²+1) at z = i")
res4_symbolic = RC.residue_symbolic('1/(z**2+1)', 'z', 'I')
print(f"   SymPy result: {res4_symbolic}")
print(f"   Simplified: {sp.simplify(res4_symbolic)}")

# Verification: sum of residues
print("\n5. Verification: ∮ dz/(z²+1) around |z|=2")
f5 = lambda z: 1/(z**2 + 1)
path5 = lambda t: 2 * np.exp(1j * t)

CI = ContourIntegrator()
integral = CI.integrate(f5, path5, (0, 2*np.pi))

res_i = RC.residue_contour(f5, 1j)
res_minus_i = RC.residue_contour(f5, -1j)

print(f"   Direct integral: {integral:.6f}")
print(f"   2πi × (Res at i + Res at -i): {2*np.pi*1j*(res_i + res_minus_i):.6f}")
```

---

## Lab 3: Real Integral Evaluation Suite

```python
"""
Comprehensive evaluation of real integrals via contour integration.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

class RealIntegralEvaluator:
    """Evaluate real integrals using contour integration."""

    @staticmethod
    def rational_integral(P_coeffs, Q_coeffs, numerical_check=True):
        """
        Evaluate ∫_{-∞}^{∞} P(x)/Q(x) dx using contour integration.

        P_coeffs, Q_coeffs: coefficients [a_n, ..., a_1, a_0] for polynomial

        Returns analytical result using residue theorem.
        """
        # Build polynomials
        P = np.poly1d(P_coeffs)
        Q = np.poly1d(Q_coeffs)

        # Check convergence condition
        if len(Q_coeffs) < len(P_coeffs) + 2:
            raise ValueError("Need deg(Q) >= deg(P) + 2 for convergence")

        # Find poles (roots of Q)
        poles = np.roots(Q_coeffs)

        # Keep only poles in upper half-plane
        uhp_poles = poles[poles.imag > 1e-10]

        # Compute residues at UHP poles
        total_residue = 0
        for pole in uhp_poles:
            # Simple pole residue: P(pole)/Q'(pole)
            Q_deriv = np.polyder(Q)
            residue = P(pole) / Q_deriv(pole)
            total_residue += residue

        analytical = 2 * np.pi * 1j * total_residue

        if numerical_check:
            numerical, _ = integrate.quad(lambda x: P(x)/Q(x), -np.inf, np.inf)
            return analytical.real, numerical

        return analytical.real

    @staticmethod
    def fourier_integral(f, omega, a_poles, numerical_check=True):
        """
        Evaluate ∫_{-∞}^{∞} f(x) e^{iωx} dx for ω > 0.

        f: function of z
        omega: frequency (positive)
        a_poles: list of (pole_location, residue) in upper half-plane

        Uses Jordan's lemma.
        """
        if omega <= 0:
            raise ValueError("Use lower half-plane for ω < 0")

        # Sum residues with exponential factors
        total = sum(res * np.exp(1j * omega * pole) for pole, res in a_poles)
        analytical = 2 * np.pi * 1j * total

        if numerical_check:
            # Numerical check (careful with oscillatory integral)
            real_part, _ = integrate.quad(
                lambda x: f(x) * np.cos(omega * x), -100, 100, limit=1000)
            imag_part, _ = integrate.quad(
                lambda x: f(x) * np.sin(omega * x), -100, 100, limit=1000)
            numerical = real_part + 1j * imag_part
            return analytical, numerical

        return analytical


# Demonstrations
print("=" * 60)
print("REAL INTEGRAL EVALUATION")
print("=" * 60)

RIE = RealIntegralEvaluator()

# Test 1: ∫ dx/(x²+1) = π
print("\n1. ∫_{-∞}^{∞} dx/(x²+1)")
ana1, num1 = RIE.rational_integral([1], [1, 0, 1])
print(f"   Analytical: {ana1:.10f}")
print(f"   Numerical:  {num1:.10f}")
print(f"   Expected:   {np.pi:.10f}")

# Test 2: ∫ dx/(x²+1)² = π/2
print("\n2. ∫_{-∞}^{∞} dx/(x²+1)²")
ana2, num2 = RIE.rational_integral([1], [1, 0, 2, 0, 1])
print(f"   Analytical: {ana2:.10f}")
print(f"   Numerical:  {num2:.10f}")
print(f"   Expected:   {np.pi/2:.10f}")

# Test 3: ∫ dx/(x⁴+1) = π/√2
print("\n3. ∫_{-∞}^{∞} dx/(x⁴+1)")
ana3, num3 = RIE.rational_integral([1], [1, 0, 0, 0, 1])
print(f"   Analytical: {ana3:.10f}")
print(f"   Numerical:  {num3:.10f}")
print(f"   Expected:   {np.pi/np.sqrt(2):.10f}")

# Test 4: Fourier-type ∫ e^{ix}/(x²+1) dx = π/e
print("\n4. ∫_{-∞}^{∞} e^{ix}/(x²+1) dx")
f4 = lambda x: 1/(x**2 + 1)
# Pole at z=i with residue 1/(2i)
ana4, num4 = RIE.fourier_integral(f4, 1, [(1j, 1/(2*1j))])
print(f"   Analytical: {ana4:.10f}")
print(f"   Numerical:  {num4:.10f}")
print(f"   Expected:   {np.pi/np.e:.10f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot integrands
x = np.linspace(-5, 5, 500)

# 1/(x²+1)
axes[0, 0].plot(x, 1/(x**2+1), 'b-', linewidth=2)
axes[0, 0].fill_between(x, 0, 1/(x**2+1), alpha=0.3)
axes[0, 0].set_title('$\\frac{1}{x^2+1}$ (integral = π)')
axes[0, 0].set_xlabel('x')
axes[0, 0].grid(True, alpha=0.3)

# 1/(x⁴+1)
axes[0, 1].plot(x, 1/(x**4+1), 'r-', linewidth=2)
axes[0, 1].fill_between(x, 0, 1/(x**4+1), alpha=0.3)
axes[0, 1].set_title('$\\frac{1}{x^4+1}$ (integral = π/√2)')
axes[0, 1].set_xlabel('x')
axes[0, 1].grid(True, alpha=0.3)

# cos(x)/(x²+1)
axes[1, 0].plot(x, np.cos(x)/(x**2+1), 'g-', linewidth=2)
axes[1, 0].fill_between(x, 0, np.cos(x)/(x**2+1), alpha=0.3)
axes[1, 0].set_title('$\\frac{\\cos x}{x^2+1}$ (integral = π/e)')
axes[1, 0].set_xlabel('x')
axes[1, 0].grid(True, alpha=0.3)

# Pole locations
theta = np.linspace(0, 2*np.pi, 100)
for ax in [axes[1, 1]]:
    # z²+1 = 0 poles
    ax.plot([0, 0], [1, -1], 'ro', markersize=10, label='Poles of 1/(z²+1)')
    # Unit circle
    ax.plot(np.cos(theta), np.sin(theta), 'b--', alpha=0.5)
    # Semicircle contour
    ax.plot(np.linspace(-3, 3, 100), np.zeros(100), 'g-', linewidth=2)
    ax.plot(3*np.cos(theta[:51]), 3*np.sin(theta[:51]), 'g-', linewidth=2)
    ax.arrow(0, 0, 1, 0, head_width=0.1, color='g')
axes[1, 1].set_title('Pole Locations and Semicircular Contour')
axes[1, 1].set_xlabel('Re(z)')
axes[1, 1].set_ylabel('Im(z)')
axes[1, 1].legend()
axes[1, 1].axis('equal')
axes[1, 1].set_xlim(-4, 4)
axes[1, 1].set_ylim(-2, 4)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('real_integral_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Lab 4: Physics Applications — Green's Functions

```python
"""
Quantum mechanics applications: Green's functions and propagators.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class QuantumGreensFunction:
    """Compute and visualize quantum Green's functions."""

    @staticmethod
    def free_particle_1d(x, xp, E, m=1, hbar=1):
        """
        1D free particle Green's function G(x,x';E).

        G(E) = (E - H + iε)^{-1} in energy representation.

        For free particle: G(x,x';E) = -im/(ℏ²k) exp(ik|x-x'|)
        where k = √(2mE)/ℏ
        """
        epsilon = 1e-6
        E_complex = E + 1j * epsilon

        k = np.sqrt(2 * m * E_complex) / hbar
        G = -1j * m / (hbar**2 * k) * np.exp(1j * k * np.abs(x - xp))
        return G

    @staticmethod
    def propagator_from_green(G_func, x, xp, t, E_range=(-10, 10), n_E=1000):
        """
        Compute propagator K(x,t;x',0) from Green's function via:
        K = (1/2πℏ) ∫ G(E) e^{-iEt/ℏ} dE

        Uses contour integration closing in appropriate half-plane.
        """
        hbar = 1
        E_values = np.linspace(E_range[0], E_range[1], n_E)
        dE = E_values[1] - E_values[0]

        integrand = G_func(x, xp, E_values) * np.exp(-1j * E_values * t / hbar)
        K = np.trapz(integrand, E_values) / (2 * np.pi * hbar)

        return K

    @staticmethod
    def spectral_function(G_func, x, xp, E_values):
        """
        Spectral function A(E) = -Im G(E+iε) / π

        This gives the density of states.
        """
        G_values = G_func(x, xp, E_values)
        A = -G_values.imag / np.pi
        return A


# Demonstrations
print("=" * 60)
print("QUANTUM GREEN'S FUNCTIONS")
print("=" * 60)

QGF = QuantumGreensFunction()

# Parameters
m = 1
hbar = 1
x = 0
xp = 0

# Energy dependence of Green's function
E_values = np.linspace(0.1, 10, 200)
G_values = QGF.free_particle_1d(x, xp, E_values, m, hbar)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Real and imaginary parts of G
axes[0, 0].plot(E_values, G_values.real, 'b-', linewidth=2, label='Re G')
axes[0, 0].plot(E_values, G_values.imag, 'r-', linewidth=2, label='Im G')
axes[0, 0].set_xlabel('E')
axes[0, 0].set_ylabel('G(x=0, x\'=0; E)')
axes[0, 0].set_title('Free Particle Green\'s Function')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Spectral function
A_values = QGF.spectral_function(QGF.free_particle_1d, x, xp, E_values)
axes[0, 1].plot(E_values, A_values, 'g-', linewidth=2)
axes[0, 1].set_xlabel('E')
axes[0, 1].set_ylabel('A(E)')
axes[0, 1].set_title('Spectral Function (∝ Density of States)')
axes[0, 1].grid(True, alpha=0.3)

# Spatial dependence
x_values = np.linspace(-5, 5, 200)
G_spatial = QGF.free_particle_1d(x_values, 0, E=1, m=1, hbar=1)

axes[1, 0].plot(x_values, G_spatial.real, 'b-', linewidth=2, label='Re G')
axes[1, 0].plot(x_values, G_spatial.imag, 'r-', linewidth=2, label='Im G')
axes[1, 0].plot(x_values, np.abs(G_spatial), 'k--', linewidth=1, label='|G|')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('G(x, 0; E=1)')
axes[1, 0].set_title('Spatial Dependence of Green\'s Function')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Pole structure in complex E plane
E_real = np.linspace(-2, 5, 100)
E_imag = np.linspace(-1, 1, 100)
E_R, E_I = np.meshgrid(E_real, E_imag)
E_complex = E_R + 1j * E_I

# Avoid branch cut
k = np.sqrt(2 * m * E_complex) / hbar
G_complex = -1j * m / (hbar**2 * k)

axes[1, 1].contourf(E_R, E_I, np.log10(np.abs(G_complex) + 1e-10),
                    levels=50, cmap='viridis')
axes[1, 1].axhline(y=0, color='r', linewidth=2, label='Branch cut (E > 0)')
axes[1, 1].plot([0], [0], 'r*', markersize=15, label='Branch point at E=0')
axes[1, 1].set_xlabel('Re(E)')
axes[1, 1].set_ylabel('Im(E)')
axes[1, 1].set_title('Analytic Structure in Complex E Plane')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('quantum_greens_function.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGreen's function at x=x'=0, E=1:")
G_sample = QGF.free_particle_1d(0, 0, 1)
print(f"G = {G_sample:.6f}")
print(f"|G| = {np.abs(G_sample):.6f}")
```

---

## Lab 5: Dispersion Relations and Kramers-Kronig

```python
"""
Kramers-Kronig relations via contour integration.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def kramers_kronig_real(chi_imag_func, omega, omega_range=(-100, 100)):
    """
    Compute Re χ(ω) from Im χ(ω) using Kramers-Kronig relation:

    Re χ(ω) = (1/π) P ∫_{-∞}^{∞} Im χ(ω') / (ω' - ω) dω'
    """
    def integrand(wp):
        if abs(wp - omega) < 1e-10:
            return 0
        return chi_imag_func(wp) / (wp - omega)

    # Principal value integral
    result, _ = integrate.quad(integrand, omega_range[0], omega - 0.01, limit=100)
    result2, _ = integrate.quad(integrand, omega + 0.01, omega_range[1], limit=100)

    return (result + result2) / np.pi

def kramers_kronig_imag(chi_real_func, omega, omega_range=(-100, 100)):
    """
    Compute Im χ(ω) from Re χ(ω) using Kramers-Kronig relation:

    Im χ(ω) = -(1/π) P ∫_{-∞}^{∞} Re χ(ω') / (ω' - ω) dω'
    """
    def integrand(wp):
        if abs(wp - omega) < 1e-10:
            return 0
        return chi_real_func(wp) / (wp - omega)

    result, _ = integrate.quad(integrand, omega_range[0], omega - 0.01, limit=100)
    result2, _ = integrate.quad(integrand, omega + 0.01, omega_range[1], limit=100)

    return -(result + result2) / np.pi

# Example: Lorentzian response function
# χ(ω) = 1/(ω_0² - ω² - iγω)

omega_0 = 5  # Resonance frequency
gamma = 0.5  # Damping

def chi_exact(omega):
    """Exact susceptibility."""
    return 1 / (omega_0**2 - omega**2 - 1j * gamma * omega)

def chi_imag_lorentzian(omega):
    """Imaginary part of Lorentzian."""
    return (chi_exact(omega)).imag

def chi_real_lorentzian(omega):
    """Real part of Lorentzian."""
    return (chi_exact(omega)).real

# Compute Kramers-Kronig transformation
omega_values = np.linspace(0.1, 10, 50)
chi_real_exact = np.array([chi_real_lorentzian(w) for w in omega_values])
chi_imag_exact = np.array([chi_imag_lorentzian(w) for w in omega_values])

chi_real_kk = np.array([kramers_kronig_real(chi_imag_lorentzian, w) for w in omega_values])
chi_imag_kk = np.array([kramers_kronig_imag(chi_real_lorentzian, w) for w in omega_values])

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Real part comparison
axes[0, 0].plot(omega_values, chi_real_exact, 'b-', linewidth=2, label='Exact Re χ')
axes[0, 0].plot(omega_values, chi_real_kk, 'r--', linewidth=2, label='KK from Im χ')
axes[0, 0].axvline(x=omega_0, color='g', linestyle=':', label=f'ω₀ = {omega_0}')
axes[0, 0].set_xlabel('ω')
axes[0, 0].set_ylabel('Re χ(ω)')
axes[0, 0].set_title('Real Part: Exact vs Kramers-Kronig')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Imaginary part comparison
axes[0, 1].plot(omega_values, chi_imag_exact, 'b-', linewidth=2, label='Exact Im χ')
axes[0, 1].plot(omega_values, chi_imag_kk, 'r--', linewidth=2, label='KK from Re χ')
axes[0, 1].axvline(x=omega_0, color='g', linestyle=':', label=f'ω₀ = {omega_0}')
axes[0, 1].set_xlabel('ω')
axes[0, 1].set_ylabel('Im χ(ω)')
axes[0, 1].set_title('Imaginary Part: Exact vs Kramers-Kronig')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# |χ| and phase
chi_values = chi_exact(omega_values)
axes[1, 0].plot(omega_values, np.abs(chi_values), 'purple', linewidth=2)
axes[1, 0].axvline(x=omega_0, color='g', linestyle=':', label=f'Resonance')
axes[1, 0].set_xlabel('ω')
axes[1, 0].set_ylabel('|χ(ω)|')
axes[1, 0].set_title('Magnitude of Response')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(omega_values, np.angle(chi_values) * 180 / np.pi, 'orange', linewidth=2)
axes[1, 1].axvline(x=omega_0, color='g', linestyle=':', label=f'Resonance')
axes[1, 1].axhline(y=-90, color='k', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('ω')
axes[1, 1].set_ylabel('Phase (degrees)')
axes[1, 1].set_title('Phase of Response')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kramers_kronig.png', dpi=150, bbox_inches='tight')
plt.show()

print("Kramers-Kronig Verification:")
print("-" * 40)
print("Lorentzian response χ(ω) = 1/(ω₀² - ω² - iγω)")
print(f"Parameters: ω₀ = {omega_0}, γ = {gamma}")
print(f"\nAt ω = {omega_0/2}:")
w_test = omega_0 / 2
print(f"  Exact Re χ:    {chi_real_lorentzian(w_test):.6f}")
print(f"  KK from Im χ:  {kramers_kronig_real(chi_imag_lorentzian, w_test):.6f}")
```

---

## Lab 6: Challenge Problem Set

```python
"""
Challenge problems in contour integration.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

print("=" * 60)
print("CHALLENGE PROBLEMS")
print("=" * 60)

# Challenge 1: Compute ∫_0^∞ ln(x)/(1+x²) dx = 0
print("\nChallenge 1: ∫_0^∞ ln(x)/(1+x²) dx")
result1, _ = integrate.quad(lambda x: np.log(x)/(1+x**2), 0.001, 1000)
print(f"  Numerical: {result1:.10f}")
print(f"  Expected:  0")

# Challenge 2: Compute ∫_0^∞ x^a/(1+x)^2 dx for a = 1/3
print("\nChallenge 2: ∫_0^∞ x^(1/3)/(1+x)² dx")
a = 1/3
result2, _ = integrate.quad(lambda x: x**a/(1+x)**2, 0, np.inf)
expected2 = np.pi * a / np.sin(np.pi * a)
print(f"  Numerical: {result2:.10f}")
print(f"  Expected:  πa/sin(πa) = {expected2:.10f}")

# Challenge 3: ∫_0^∞ sin²(x)/x² dx = π/2
print("\nChallenge 3: ∫_0^∞ sin²(x)/x² dx")
result3, _ = integrate.quad(lambda x: np.sin(x)**2/x**2 if x > 0.001 else 1, 0, 1000)
print(f"  Numerical: {result3:.10f}")
print(f"  Expected:  π/2 = {np.pi/2:.10f}")

# Challenge 4: ∫_0^∞ e^{-x²}cos(2bx) dx = √π/2 e^{-b²}
print("\nChallenge 4: ∫_0^∞ e^{-x²}cos(2x) dx")
b = 1
result4, _ = integrate.quad(lambda x: np.exp(-x**2)*np.cos(2*b*x), 0, np.inf)
expected4 = np.sqrt(np.pi)/2 * np.exp(-b**2)
print(f"  Numerical: {result4:.10f}")
print(f"  Expected:  (√π/2)e^{-1} = {expected4:.10f}")

# Challenge 5: ∫_0^{2π} dθ/(a + sin θ)² for a > 1
print("\nChallenge 5: ∫_0^{2π} dθ/(2 + sin θ)²")
a = 2
result5, _ = integrate.quad(lambda t: 1/(a + np.sin(t))**2, 0, 2*np.pi)
expected5 = 2*np.pi*a / (a**2 - 1)**1.5
print(f"  Numerical: {result5:.10f}")
print(f"  Expected:  2πa/(a²-1)^{3/2} = {expected5:.10f}")

# Visualization of challenge integrands
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

x = np.linspace(0.01, 10, 500)
theta = np.linspace(0, 2*np.pi, 500)

# Challenge 1
axes[0, 0].plot(x, np.log(x)/(1+x**2), 'b-', linewidth=2)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[0, 0].fill_between(x, 0, np.log(x)/(1+x**2), alpha=0.3)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_title('Challenge 1: ln(x)/(1+x²)')
axes[0, 0].grid(True, alpha=0.3)

# Challenge 2
axes[0, 1].plot(x, x**(1/3)/(1+x)**2, 'r-', linewidth=2)
axes[0, 1].fill_between(x, 0, x**(1/3)/(1+x)**2, alpha=0.3)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_title('Challenge 2: x^{1/3}/(1+x)²')
axes[0, 1].grid(True, alpha=0.3)

# Challenge 3
x3 = np.linspace(0.01, 20, 500)
y3 = np.sin(x3)**2/x3**2
axes[0, 2].plot(x3, y3, 'g-', linewidth=2)
axes[0, 2].fill_between(x3, 0, y3, alpha=0.3)
axes[0, 2].set_xlabel('x')
axes[0, 2].set_title('Challenge 3: sin²(x)/x²')
axes[0, 2].grid(True, alpha=0.3)

# Challenge 4
x4 = np.linspace(0, 5, 500)
y4 = np.exp(-x4**2)*np.cos(2*x4)
axes[1, 0].plot(x4, y4, 'purple', linewidth=2)
axes[1, 0].fill_between(x4, 0, y4, alpha=0.3)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_title('Challenge 4: e^{-x²}cos(2x)')
axes[1, 0].grid(True, alpha=0.3)

# Challenge 5
y5 = 1/(2 + np.sin(theta))**2
axes[1, 1].plot(theta, y5, 'orange', linewidth=2)
axes[1, 1].fill_between(theta, 0, y5, alpha=0.3)
axes[1, 1].set_xlabel('θ')
axes[1, 1].set_title('Challenge 5: 1/(2 + sin θ)²')
axes[1, 1].grid(True, alpha=0.3)

# Summary
axes[1, 2].axis('off')
summary_text = """
CHALLENGE RESULTS SUMMARY
─────────────────────────
1. ∫₀^∞ ln(x)/(1+x²) dx = 0

2. ∫₀^∞ x^a/(1+x)² dx = πa/sin(πa)

3. ∫₀^∞ sin²x/x² dx = π/2

4. ∫₀^∞ e^{-x²}cos(2bx) dx = (√π/2)e^{-b²}

5. ∫₀^{2π} dθ/(a+sin θ)² = 2πa/(a²-1)^{3/2}

All verified by contour integration!
"""
axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
               verticalalignment='center')

plt.tight_layout()
plt.savefig('challenge_problems.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Summary

### Key Computational Tools

| Tool | Purpose |
|------|---------|
| `ContourIntegrator` | General contour integration |
| `ResidueCalculator` | Numerical/symbolic residues |
| `RealIntegralEvaluator` | Real integrals via contours |
| `QuantumGreensFunction` | Physics applications |
| Kramers-Kronig | Dispersion relations |

### Main Takeaways

1. **Numerical verification** confirms analytical results from contour integration.

2. **Visualization** builds intuition for contour choice and pole locations.

3. **Physics applications** directly use contour methods (Green's functions, dispersion).

4. **Challenge problems** test mastery of techniques.

---

## Daily Checklist

- [ ] I can implement contour integrals numerically
- [ ] I can compute residues both ways (formula and contour)
- [ ] I can verify analytical results computationally
- [ ] I understand Green's function pole structure
- [ ] I can apply Kramers-Kronig relations

---

## Preview: Day 182

Tomorrow's week review synthesizes all contour integration techniques with:
- Comprehensive concept map
- Problem sets covering all methods
- Connections to quantum mechanics
- Preparation for residue calculus (Week 27)

---

*"The purpose of computing is insight, not numbers."*
— Richard Hamming
