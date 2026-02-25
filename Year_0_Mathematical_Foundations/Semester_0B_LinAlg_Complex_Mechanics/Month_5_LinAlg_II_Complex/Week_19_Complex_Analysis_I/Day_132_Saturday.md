# Day 132: Computational Lab — Complex Analysis Toolkit

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Part 1: Visualization & Domain Coloring |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Part 2: Contour Integration Library |
| Evening | 6:00 PM - 7:30 PM | 1.5 hours | Part 3: Physics Applications |

**Total Study Time: 8 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Create domain coloring visualizations for any complex function
2. Implement numerical contour integration
3. Visualize poles, zeros, and branch cuts
4. Apply complex analysis to physics problems
5. Build reusable tools for future quantum mechanics work

---

## 💻 Part 1: Complex Function Visualization (3.5 hours)

### 1.1 Domain Coloring Implementation

```python
"""
Complex Analysis Visualization Toolkit
======================================
A comprehensive library for visualizing complex functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

class ComplexVisualizer:
    """Visualization tools for complex functions."""
    
    def __init__(self, x_range=(-2, 2), y_range=(-2, 2), resolution=500):
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self._setup_grid()
    
    def _setup_grid(self):
        """Create the complex plane grid."""
        x = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = self.X + 1j * self.Y
    
    def domain_coloring(self, f, title='f(z)', 
                        show_grid=True, 
                        brightness_mode='log',
                        save_path=None):
        """
        Create domain coloring plot.
        
        Parameters:
        -----------
        f : callable
            Complex function to visualize
        title : str
            Plot title
        show_grid : bool
            Whether to overlay conformal grid
        brightness_mode : 'log', 'arctan', 'constant'
            How to map modulus to brightness
        """
        # Evaluate function
        with np.errstate(all='ignore'):
            W = f(self.Z)
        
        # Handle infinities and NaN
        W = np.where(np.isfinite(W), W, np.nan)
        
        # Create HSV image
        # Hue: argument (angle)
        H = (np.angle(W) + np.pi) / (2 * np.pi)
        
        # Saturation: constant
        S = np.ones_like(H) * 0.9
        
        # Value (brightness): from modulus
        abs_W = np.abs(W)
        if brightness_mode == 'log':
            # Log scale, normalized
            V = 1 - 1/(1 + np.log1p(abs_W)/3)
        elif brightness_mode == 'arctan':
            V = 2 * np.arctan(abs_W) / np.pi
        else:
            V = np.ones_like(H) * 0.8
        
        # Handle NaN values
        H = np.nan_to_num(H, nan=0)
        S = np.nan_to_num(S, nan=0)
        V = np.nan_to_num(V, nan=0)
        
        # Convert to RGB
        HSV = np.dstack([H, S, V])
        RGB = hsv_to_rgb(HSV)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(RGB, extent=[self.x_range[0], self.x_range[1],
                               self.y_range[0], self.y_range[1]],
                  origin='lower', aspect='equal')
        
        # Add conformal grid (level curves of |f| and arg(f))
        if show_grid:
            # Level curves of modulus
            levels_mod = [0.5, 1, 2, 4, 8]
            ax.contour(self.X, self.Y, abs_W, levels=levels_mod,
                      colors='white', linewidths=0.3, alpha=0.5)
            
            # Level curves of argument
            levels_arg = np.linspace(-np.pi, np.pi, 13)[1:-1]
            ax.contour(self.X, self.Y, np.angle(W), levels=levels_arg,
                      colors='white', linewidths=0.3, alpha=0.5)
        
        ax.set_xlabel('Re(z)', fontsize=12)
        ax.set_ylabel('Im(z)', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Add colorbar for phase
        sm = plt.cm.ScalarMappable(cmap='hsv', 
                                    norm=plt.Normalize(-np.pi, np.pi))
        cbar = plt.colorbar(sm, ax=ax, label='arg(f(z))')
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        
        return fig, ax
    
    def riemann_surface(self, f, title='Riemann Surface', 
                        n_sheets=2, save_path=None):
        """
        Visualize Riemann surface for multi-valued function.
        """
        fig = plt.figure(figsize=(14, 6))
        
        # Create polar coordinates
        r = np.linspace(0.1, 2, 100)
        theta = np.linspace(0, n_sheets * 2 * np.pi, 400)
        R, Theta = np.meshgrid(r, theta)
        
        # Complex coordinates
        Z = R * np.exp(1j * Theta / n_sheets)  # Spread sheets
        
        # Evaluate function
        W = f(R, Theta)
        
        # Plot real part
        ax1 = fig.add_subplot(121, projection='3d')
        X_plot = R * np.cos(Theta)
        Y_plot = R * np.sin(Theta)
        ax1.plot_surface(X_plot, Y_plot, np.real(W), 
                        cmap='coolwarm', alpha=0.8)
        ax1.set_title(f'Re({title})')
        ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('Re(f)')
        
        # Plot imaginary part
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X_plot, Y_plot, np.imag(W), 
                        cmap='viridis', alpha=0.8)
        ax2.set_title(f'Im({title})')
        ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('Im(f)')
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_mapping(self, f, title='Conformal Map', save_path=None):
        """Visualize how f maps the complex plane."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Domain grid
        x_lines = np.linspace(self.x_range[0], self.x_range[1], 21)
        y_lines = np.linspace(self.y_range[0], self.y_range[1], 21)
        
        # Plot domain
        ax = axes[0]
        for x in x_lines:
            y_vals = np.linspace(self.y_range[0], self.y_range[1], 200)
            ax.plot(np.full_like(y_vals, x), y_vals, 'b-', lw=0.5)
        for y in y_lines:
            x_vals = np.linspace(self.x_range[0], self.x_range[1], 200)
            ax.plot(x_vals, np.full_like(x_vals, y), 'r-', lw=0.5)
        ax.set_title('Domain (z-plane)')
        ax.set_xlabel('Re(z)'); ax.set_ylabel('Im(z)')
        ax.axis('equal')
        ax.set_xlim(self.x_range); ax.set_ylim(self.y_range)
        
        # Plot image
        ax = axes[1]
        t = np.linspace(self.y_range[0], self.y_range[1], 200)
        for x in x_lines:
            z = x + 1j * t
            w = f(z)
            ax.plot(np.real(w), np.imag(w), 'b-', lw=0.5)
        
        t = np.linspace(self.x_range[0], self.x_range[1], 200)
        for y in y_lines:
            z = t + 1j * y
            w = f(z)
            ax.plot(np.real(w), np.imag(w), 'r-', lw=0.5)
        
        ax.set_title('Image (w-plane)')
        ax.set_xlabel('Re(w)'); ax.set_ylabel('Im(w)')
        ax.axis('equal')
        
        plt.suptitle(title, fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        
        return fig

# Demonstration
print("=" * 70)
print("COMPLEX FUNCTION VISUALIZATION")
print("=" * 70)

viz = ComplexVisualizer(x_range=(-3, 3), y_range=(-3, 3), resolution=600)

# Example 1: Polynomial
print("\n1. f(z) = z³ - 1")
viz.domain_coloring(lambda z: z**3 - 1, 
                    title='f(z) = z³ - 1\nZeros at cube roots of unity',
                    save_path='polynomial.png')

# Example 2: Rational function with poles
print("\n2. f(z) = 1/((z-1)(z+1))")
viz.domain_coloring(lambda z: 1/((z-1)*(z+1)), 
                    title='f(z) = 1/((z-1)(z+1))\nPoles at z = ±1',
                    save_path='rational.png')

# Example 3: Essential singularity
print("\n3. f(z) = e^(1/z)")
viz.domain_coloring(lambda z: np.exp(1/z), 
                    title='f(z) = e^(1/z)\nEssential singularity at z = 0',
                    x_range=(-1, 1), y_range=(-1, 1),
                    save_path='essential.png')

# Example 4: Branch cut (square root)
print("\n4. f(z) = √z (principal branch)")
viz.domain_coloring(lambda z: np.sqrt(z), 
                    title='f(z) = √z\nBranch cut on negative real axis',
                    save_path='sqrt.png')

# Example 5: Conformal mapping
print("\n5. Conformal map: w = (z-1)/(z+1)")
viz.plot_mapping(lambda z: (z-1)/(z+1),
                title='w = (z-1)/(z+1) [Möbius transformation]',
                save_path='mobius.png')
```

### 1.2 Zeros and Poles Detection

```python
def find_zeros_poles(f, x_range, y_range, resolution=100, threshold=0.1):
    """
    Find approximate locations of zeros and poles.
    
    Zeros: where |f(z)| is small
    Poles: where |f(z)| is large (or |1/f(z)| is small)
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    with np.errstate(all='ignore'):
        W = f(Z)
        abs_W = np.abs(W)
    
    # Find local minima (zeros)
    from scipy.ndimage import minimum_filter
    zeros = []
    min_filtered = minimum_filter(abs_W, size=5)
    zero_mask = (abs_W == min_filtered) & (abs_W < threshold)
    zero_indices = np.where(zero_mask)
    for i, j in zip(zero_indices[0], zero_indices[1]):
        zeros.append(Z[i, j])
    
    # Find local maxima (poles)
    poles = []
    abs_inv_W = np.abs(1/W)
    abs_inv_W = np.nan_to_num(abs_inv_W, nan=0, posinf=0)
    min_inv_filtered = minimum_filter(abs_inv_W, size=5)
    pole_mask = (abs_inv_W == min_inv_filtered) & (abs_inv_W < threshold)
    pole_indices = np.where(pole_mask)
    for i, j in zip(pole_indices[0], pole_indices[1]):
        poles.append(Z[i, j])
    
    return zeros, poles

# Example: Find zeros of z³ - 1
print("\nFinding zeros and poles:")
print("-" * 40)

f = lambda z: z**3 - 1
zeros, poles = find_zeros_poles(f, (-2, 2), (-2, 2))
print("f(z) = z³ - 1")
print(f"Zeros found: {[f'{z:.4f}' for z in zeros]}")
print(f"Expected: 1, e^(2πi/3), e^(4πi/3)")

f = lambda z: 1/(z**2 + 1)
zeros, poles = find_zeros_poles(f, (-2, 2), (-2, 2))
print("\nf(z) = 1/(z² + 1)")
print(f"Poles found: {[f'{z:.4f}' for z in poles]}")
print(f"Expected: ±i")
```

---

## 💻 Part 2: Contour Integration Library (3 hours)

### 2.1 Contour Class and Integration

```python
class Contour:
    """Represents a parametric contour in the complex plane."""
    
    def __init__(self, z_func, dz_func, t_range, name='γ'):
        """
        Parameters:
        -----------
        z_func : callable
            z(t) parametrization
        dz_func : callable
            dz/dt derivative
        t_range : tuple
            (t_start, t_end)
        name : str
            Contour name for display
        """
        self.z = z_func
        self.dz = dz_func
        self.t_range = t_range
        self.name = name
    
    @classmethod
    def circle(cls, center=0, radius=1, direction=1):
        """Create circular contour."""
        name = f'C({center}, {radius})'
        return cls(
            z_func=lambda t: center + radius * np.exp(1j * direction * t),
            dz_func=lambda t: 1j * direction * radius * np.exp(1j * direction * t),
            t_range=(0, 2*np.pi),
            name=name
        )
    
    @classmethod
    def line(cls, z_start, z_end):
        """Create straight line contour."""
        name = f'[{z_start} → {z_end}]'
        return cls(
            z_func=lambda t: z_start + t * (z_end - z_start),
            dz_func=lambda t: (z_end - z_start) * np.ones_like(t),
            t_range=(0, 1),
            name=name
        )
    
    @classmethod
    def semicircle(cls, radius=1, half='upper'):
        """Create semicircular contour."""
        if half == 'upper':
            return cls(
                z_func=lambda t: radius * np.exp(1j * t),
                dz_func=lambda t: 1j * radius * np.exp(1j * t),
                t_range=(0, np.pi),
                name=f'Semicircle(r={radius}, upper)'
            )
        else:
            return cls(
                z_func=lambda t: radius * np.exp(-1j * t),
                dz_func=lambda t: -1j * radius * np.exp(-1j * t),
                t_range=(0, np.pi),
                name=f'Semicircle(r={radius}, lower)'
            )
    
    @classmethod
    def rectangle(cls, x_range, y_range):
        """Create rectangular contour."""
        x0, x1 = x_range
        y0, y1 = y_range
        
        # Four sides
        contours = [
            cls.line(x0 + 1j*y0, x1 + 1j*y0),  # Bottom
            cls.line(x1 + 1j*y0, x1 + 1j*y1),  # Right
            cls.line(x1 + 1j*y1, x0 + 1j*y1),  # Top
            cls.line(x0 + 1j*y1, x0 + 1j*y0),  # Left
        ]
        return CompositeContour(contours, name=f'Rectangle({x_range}, {y_range})')
    
    def plot(self, ax=None, n_points=500, **kwargs):
        """Plot the contour."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        t = np.linspace(self.t_range[0], self.t_range[1], n_points)
        z_vals = self.z(t)
        
        ax.plot(np.real(z_vals), np.imag(z_vals), **kwargs)
        
        # Add arrow to show direction
        mid_idx = len(t) // 2
        ax.annotate('', xy=(np.real(z_vals[mid_idx+5]), np.imag(z_vals[mid_idx+5])),
                   xytext=(np.real(z_vals[mid_idx]), np.imag(z_vals[mid_idx])),
                   arrowprops=dict(arrowstyle='->', color=kwargs.get('color', 'blue')))
        
        return ax


class CompositeContour:
    """Contour made of multiple segments."""
    
    def __init__(self, contours, name='Composite'):
        self.contours = contours
        self.name = name
    
    def plot(self, ax=None, n_points=200, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        for c in self.contours:
            c.plot(ax, n_points, **kwargs)
        return ax


class ContourIntegrator:
    """Numerical contour integration."""
    
    def __init__(self, n_points=2000):
        self.n_points = n_points
    
    def integrate(self, f, contour, return_path=False):
        """
        Compute ∮_γ f(z) dz numerically.
        
        Uses trapezoidal rule on parametric form.
        """
        if isinstance(contour, CompositeContour):
            total = 0
            for c in contour.contours:
                total += self.integrate(f, c)
            return total
        
        t = np.linspace(contour.t_range[0], contour.t_range[1], self.n_points)
        dt = t[1] - t[0]
        
        z = contour.z(t)
        dz = contour.dz(t)
        
        with np.errstate(all='ignore'):
            integrand = f(z) * dz
        
        # Handle any infinities
        integrand = np.where(np.isfinite(integrand), integrand, 0)
        
        # Trapezoidal rule
        integral = np.trapz(integrand, t)
        
        if return_path:
            return integral, z, integrand
        return integral
    
    def verify_cauchy(self, f, contours, names=None):
        """Verify Cauchy's theorem for multiple contours."""
        print("Verifying Cauchy's Theorem")
        print("-" * 50)
        
        results = []
        for i, contour in enumerate(contours):
            result = self.integrate(f, contour)
            name = names[i] if names else f"Contour {i+1}"
            results.append(result)
            print(f"{name}: ∮ f dz = {result:.6f}")
        
        return results


# Demonstration
print("\n" + "=" * 70)
print("CONTOUR INTEGRATION")
print("=" * 70)

integrator = ContourIntegrator(n_points=5000)

# Example 1: ∮ z² dz around unit circle (should be 0)
print("\n1. ∮ z² dz around |z| = 1")
C = Contour.circle(center=0, radius=1)
result = integrator.integrate(lambda z: z**2, C)
print(f"   Result: {result:.10f}")
print(f"   Expected (Cauchy): 0")

# Example 2: ∮ 1/z dz around unit circle (should be 2πi)
print("\n2. ∮ dz/z around |z| = 1")
result = integrator.integrate(lambda z: 1/z, C)
print(f"   Result: {result:.10f}")
print(f"   Expected: 2πi = {2*np.pi*1j:.10f}")

# Example 3: ∮ e^z/(z-0.5) dz (Cauchy's formula)
print("\n3. ∮ e^z/(z-0.5) dz around |z| = 1")
result = integrator.integrate(lambda z: np.exp(z)/(z-0.5), C)
expected = 2*np.pi*1j * np.exp(0.5)
print(f"   Result: {result:.6f}")
print(f"   Expected (2πi·e^0.5): {expected:.6f}")

# Example 4: Path independence
print("\n4. Path independence: ∫ z dz from 0 to 1+i")
path1 = Contour.line(0, 1+1j)
path2_a = Contour.line(0, 1)
path2_b = Contour.line(1, 1+1j)

result1 = integrator.integrate(lambda z: z, path1)
result2 = integrator.integrate(lambda z: z, path2_a) + integrator.integrate(lambda z: z, path2_b)
exact = (1+1j)**2 / 2

print(f"   Straight path: {result1:.6f}")
print(f"   L-shaped path: {result2:.6f}")
print(f"   Exact: (1+i)²/2 = {exact:.6f}")

# Visualize contours
fig, ax = plt.subplots(figsize=(10, 8))
Contour.circle(0, 1).plot(ax, color='blue', lw=2, label='Unit circle')
Contour.circle(0.5, 0.3).plot(ax, color='red', lw=2, label='Small circle around 0.5')
Contour.line(0, 1+1j).plot(ax, color='green', lw=2, label='Line to 1+i')

ax.scatter([0, 0.5], [0, 0], c='red', s=100, zorder=5)
ax.annotate('0', (0.05, 0.1), fontsize=12)
ax.annotate('0.5', (0.55, 0.1), fontsize=12)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Various Contours')
plt.savefig('contours.png', dpi=150)
plt.show()
```

### 2.2 Residue Computation

```python
def compute_residue(f, z0, method='limit', order=1):
    """
    Compute the residue of f at z0.
    
    For simple pole: Res[f, z0] = lim_{z→z0} (z-z0)f(z)
    For pole of order n: Res[f, z0] = (1/(n-1)!) lim_{z→z0} d^{n-1}/dz^{n-1}[(z-z0)^n f(z)]
    """
    if method == 'limit':
        # Use small circle around z0
        eps = 1e-8
        C = Contour.circle(center=z0, radius=eps)
        integrator = ContourIntegrator(n_points=1000)
        integral = integrator.integrate(f, C)
        return integral / (2 * np.pi * 1j)
    
    elif method == 'numerical':
        # Numerical differentiation approach
        h = 1e-6
        if order == 1:
            # Simple pole: lim (z-z0)f(z)
            z = z0 + h
            return (z - z0) * f(z)
        else:
            # Higher order pole
            raise NotImplementedError("Higher order poles not yet implemented")

# Example: Residues
print("\n" + "=" * 70)
print("RESIDUE COMPUTATION")
print("=" * 70)

# f(z) = 1/(z² + 1) = 1/((z-i)(z+i))
print("\n1. f(z) = 1/(z² + 1)")
f = lambda z: 1/(z**2 + 1)

res_i = compute_residue(f, 1j)
res_neg_i = compute_residue(f, -1j)

print(f"   Res[f, i] = {res_i:.6f}")
print(f"   Expected: 1/(2i) = {1/(2j):.6f}")
print(f"   Res[f, -i] = {res_neg_i:.6f}")
print(f"   Expected: -1/(2i) = {-1/(2j):.6f}")

# f(z) = e^z / (z-1)²
print("\n2. f(z) = e^z / (z-1)")
f = lambda z: np.exp(z) / (z-1)
res = compute_residue(f, 1)
print(f"   Res[f, 1] = {res:.6f}")
print(f"   Expected: e = {np.e:.6f}")
```

---

## 💻 Part 3: Physics Applications (1.5 hours)

### 3.1 Real Integrals via Residues

```python
def evaluate_real_integral_residue():
    """
    Evaluate real integrals using contour methods.
    """
    print("\n" + "=" * 70)
    print("REAL INTEGRALS VIA CONTOURS")
    print("=" * 70)
    
    # 1. ∫_{-∞}^{∞} 1/(1+x²) dx = π
    print("\n1. ∫_{-∞}^{∞} 1/(1+x²) dx")
    
    # Direct numerical integration
    from scipy.integrate import quad
    result_direct, _ = quad(lambda x: 1/(1+x**2), -100, 100)
    print(f"   Direct numerical: {result_direct:.6f}")
    print(f"   Exact (π): {np.pi:.6f}")
    
    # Contour method: close in upper half plane
    # Only pole at z = i contributes
    # Res[1/(z²+1), i] = 1/(2i)
    # Integral = 2πi × (1/2i) = π
    res = 1/(2j)
    result_contour = 2 * np.pi * 1j * res
    print(f"   Contour method: {result_contour:.6f}")
    
    # 2. ∫_{0}^{2π} 1/(2 + cos θ) dθ
    print("\n2. ∫_{0}^{2π} 1/(2 + cos θ) dθ")
    
    result_direct, _ = quad(lambda t: 1/(2+np.cos(t)), 0, 2*np.pi)
    print(f"   Direct numerical: {result_direct:.6f}")
    
    # Substitution z = e^{iθ}, cos θ = (z + 1/z)/2
    # dθ = dz/(iz)
    # Integral becomes ∮ 1/(2 + (z+1/z)/2) × dz/(iz)
    #                = ∮ 2/(z² + 4z + 1) × dz/i
    # Poles at z = -2 ± √3
    # Only -2 + √3 ≈ -0.27 is inside |z| = 1
    
    z_pole = -2 + np.sqrt(3)
    # Residue at this pole
    residue = 2 / (2*z_pole + 4)  # derivative of z² + 4z + 1
    result_contour = np.real(2 * np.pi * 1j * residue / (1j))
    exact = 2*np.pi/np.sqrt(3)
    
    print(f"   Contour method: {result_contour:.6f}")
    print(f"   Exact (2π/√3): {exact:.6f}")
    
    # 3. ∫_{-∞}^{∞} cos(x)/(1+x²) dx = π/e
    print("\n3. ∫_{-∞}^{∞} cos(x)/(1+x²) dx")
    
    result_direct, _ = quad(lambda x: np.cos(x)/(1+x**2), -100, 100)
    print(f"   Direct numerical: {result_direct:.6f}")
    print(f"   Exact (π/e): {np.pi/np.e:.6f}")
    
    # Use e^{iz}/(1+z²), take real part
    # Residue at i: e^{i·i}/(2i) = e^{-1}/(2i)
    # Integral of e^{iz}/(1+z²) = 2πi × e^{-1}/(2i) = π/e
    # Real part (cosine integral) = π/e

evaluate_real_integral_residue()
```

### 3.2 Quantum Green's Function

```python
def quantum_greens_function():
    """
    Compute quantum mechanical Green's function using contours.
    
    G(E) = 1/(E - H) has poles at eigenvalues of H
    """
    print("\n" + "=" * 70)
    print("QUANTUM GREEN'S FUNCTION")
    print("=" * 70)
    
    # Simple model: 1D harmonic oscillator-like spectrum
    # Eigenvalues at E_n = n + 1/2 (ℏω = 1)
    E_n = lambda n: n + 0.5
    
    # Green's function in spectral representation
    # G(E) = Σ_n |n⟩⟨n| / (E - E_n)
    
    # For visualization, use scalar version
    def G(E, n_max=10, eta=0.1):
        """Green's function with small imaginary part for regularization."""
        E_complex = E + 1j * eta
        return sum(1/(E_complex - E_n(n)) for n in range(n_max))
    
    # Plot spectral function A(E) = -2 Im[G(E+i0⁺)]
    E_vals = np.linspace(-0.5, 10, 1000)
    A_vals = -2 * np.imag([G(E, eta=0.05) for E in E_vals])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Spectral function
    axes[0, 0].plot(E_vals, A_vals, 'b-', lw=2)
    for n in range(10):
        axes[0, 0].axvline(x=E_n(n), color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Energy E')
    axes[0, 0].set_ylabel('A(E)')
    axes[0, 0].set_title('Spectral Function A(E) = -2 Im[G(E+iη)]\nPeaks at eigenvalues')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Poles in complex plane
    axes[0, 1].scatter([E_n(n) for n in range(10)], np.zeros(10), 
                       c='red', s=100, marker='x', label='Poles (eigenvalues)')
    axes[0, 1].axhline(y=0, color='k', lw=0.5)
    axes[0, 1].set_xlabel('Re(E)')
    axes[0, 1].set_ylabel('Im(E)')
    axes[0, 1].set_title('Poles of G(E) in Complex Energy Plane')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(-1, 11)
    axes[0, 1].set_ylim(-1, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Contour integration to extract density of states
    # ρ(E) = -1/π Im[G(E+i0⁺)] = Σ_n δ(E - E_n)
    axes[1, 0].plot(E_vals, A_vals/(2*np.pi), 'b-', lw=2)
    axes[1, 0].set_xlabel('Energy E')
    axes[1, 0].set_ylabel('ρ(E)')
    axes[1, 0].set_title('Density of States ρ(E) = -Im[G]/π')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time evolution via contour integral
    # ⟨n|e^{-iHt}|n⟩ = e^{-iE_n t}
    # Can be obtained from (1/2πi) ∮ G(E) e^{-iEt} dE
    
    t = np.linspace(0, 10, 200)
    # Ground state evolution
    psi_0 = np.exp(-1j * E_n(0) * t)
    
    axes[1, 1].plot(t, np.real(psi_0), 'b-', lw=2, label='Re⟨0|ψ(t)⟩')
    axes[1, 1].plot(t, np.imag(psi_0), 'r-', lw=2, label='Im⟨0|ψ(t)⟩')
    axes[1, 1].plot(t, np.abs(psi_0), 'k--', lw=2, label='|⟨0|ψ(t)⟩|')
    axes[1, 1].set_xlabel('Time t')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Time Evolution of Ground State\nObtained from contour integral of G(E)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('greens_function.png', dpi=150)
    plt.show()
    
    print("\nKey insight: The poles of G(E) = (E-H)⁻¹ are exactly the")
    print("eigenvalues of H. Contour integration extracts spectral information!")

quantum_greens_function()
```

### 3.3 Kramers-Kronig Relations

```python
def kramers_kronig_demo():
    """
    Demonstrate Kramers-Kronig relations from analyticity.
    """
    print("\n" + "=" * 70)
    print("KRAMERS-KRONIG RELATIONS")
    print("=" * 70)
    
    # Response function: Lorentzian (like a damped harmonic oscillator)
    # χ(ω) = 1/(ω₀² - ω² - iγω)
    
    omega_0 = 2.0  # Resonance frequency
    gamma = 0.3    # Damping
    
    def chi(omega):
        return 1/(omega_0**2 - omega**2 - 1j*gamma*omega)
    
    omega = np.linspace(-5, 5, 1000)
    chi_vals = chi(omega)
    
    chi_real = np.real(chi_vals)
    chi_imag = np.imag(chi_vals)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Real and imaginary parts
    axes[0, 0].plot(omega, chi_real, 'b-', lw=2, label="Re[χ(ω)]")
    axes[0, 0].plot(omega, chi_imag, 'r-', lw=2, label="Im[χ(ω)]")
    axes[0, 0].axhline(y=0, color='k', lw=0.5)
    axes[0, 0].axvline(x=omega_0, color='g', linestyle='--', alpha=0.5, label=f'ω₀ = {omega_0}')
    axes[0, 0].set_xlabel('ω')
    axes[0, 0].set_ylabel('χ(ω)')
    axes[0, 0].set_title('Response Function χ(ω)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Verify Kramers-Kronig
    # Re[χ(ω)] = (1/π) P ∫ Im[χ(ω')]/(ω'-ω) dω'
    # Im[χ(ω)] = -(1/π) P ∫ Re[χ(ω')]/(ω'-ω) dω'
    
    def kramers_kronig_transform(func_vals, omega_vals, omega_point):
        """Compute Hilbert transform (principal value integral)."""
        # Numerical principal value
        mask = np.abs(omega_vals - omega_point) > 0.01
        integrand = func_vals[mask] / (omega_vals[mask] - omega_point)
        d_omega = omega_vals[1] - omega_vals[0]
        return np.sum(integrand) * d_omega / np.pi
    
    # Reconstruct real part from imaginary part
    omega_test = np.linspace(-4, 4, 50)
    chi_real_reconstructed = []
    for w in omega_test:
        chi_real_reconstructed.append(kramers_kronig_transform(chi_imag, omega, w))
    
    axes[0, 1].plot(omega, chi_real, 'b-', lw=2, label="Re[χ] (direct)")
    axes[0, 1].plot(omega_test, chi_real_reconstructed, 'ro', ms=4, 
                    label="Re[χ] (from KK)")
    axes[0, 1].set_xlabel('ω')
    axes[0, 1].set_ylabel("Re[χ(ω)]")
    axes[0, 1].set_title('Kramers-Kronig: Re from Im')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Analyticity in upper half plane
    omega_complex = np.linspace(-5, 5, 100) + 1j * np.linspace(0, 2, 100)[:, np.newaxis]
    chi_complex = chi(omega_complex)
    
    axes[1, 0].contourf(np.real(omega_complex), np.imag(omega_complex), 
                        np.log(np.abs(chi_complex)+1), levels=30, cmap='viridis')
    axes[1, 0].scatter([omega_0 + gamma/2, -omega_0 + gamma/2], 
                       [-gamma/2 * np.sqrt(4*omega_0**2/gamma**2 - 1), 
                        gamma/2 * np.sqrt(4*omega_0**2/gamma**2 - 1)],
                       c='red', s=100, marker='x', label='Poles')
    axes[1, 0].axhline(y=0, color='white', lw=1)
    axes[1, 0].set_xlabel('Re(ω)')
    axes[1, 0].set_ylabel('Im(ω)')
    axes[1, 0].set_title('|χ(ω)| in Complex ω Plane\nPoles in lower half-plane → causal!')
    
    # Sum rule
    # ∫ Im[χ(ω)] dω = π × (coefficient of 1/ω at high freq)
    integral_imag = np.trapz(chi_imag, omega)
    axes[1, 1].fill_between(omega, 0, chi_imag, alpha=0.3)
    axes[1, 1].plot(omega, chi_imag, 'r-', lw=2)
    axes[1, 1].axhline(y=0, color='k', lw=0.5)
    axes[1, 1].set_xlabel('ω')
    axes[1, 1].set_ylabel("Im[χ(ω)]")
    axes[1, 1].set_title(f'Sum Rule: ∫Im[χ]dω = {integral_imag:.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kramers_kronig.png', dpi=150)
    plt.show()
    
    print("\nKramers-Kronig relations:")
    print("  Re[χ(ω)] = (1/π) P ∫ Im[χ(ω')]/(ω'-ω) dω'")
    print("  Im[χ(ω)] = -(1/π) P ∫ Re[χ(ω')]/(ω'-ω) dω'")
    print("\nThese follow directly from the analyticity of χ(ω) in the upper half-plane,")
    print("which is a consequence of causality!")

kramers_kronig_demo()

print("\n" + "=" * 70)
print("LAB COMPLETE!")
print("=" * 70)
```

---

## 📝 Summary

### Tools Built Today

| Tool | Purpose |
|------|---------|
| ComplexVisualizer | Domain coloring, Riemann surfaces, conformal maps |
| Contour | Parametric contour representation |
| ContourIntegrator | Numerical contour integration |
| compute_residue | Residue calculation |
| Kramers-Kronig | Response function analysis |

### Key Applications

1. **Visualization**: Domain coloring reveals zeros, poles, branch cuts
2. **Integration**: Cauchy's theorem makes many integrals trivial
3. **Real Integrals**: Contour methods evaluate difficult real integrals
4. **Green's Functions**: Poles encode eigenvalue information
5. **Kramers-Kronig**: Analyticity implies relations between real/imaginary parts

---

## ✅ Daily Checklist

- [ ] Build domain coloring visualizer
- [ ] Implement contour integration
- [ ] Verify Cauchy's theorem numerically
- [ ] Compute residues
- [ ] Evaluate real integrals via residues
- [ ] Explore quantum Green's functions
- [ ] Understand Kramers-Kronig relations
- [ ] Save all code for future use

---

## 🔮 Preview: Day 133

Tomorrow we consolidate everything in our **Week 19 Review**, with a comprehensive problem set and preparation for Week 20 (Complex Analysis II: Residue Theorem and Applications)!
