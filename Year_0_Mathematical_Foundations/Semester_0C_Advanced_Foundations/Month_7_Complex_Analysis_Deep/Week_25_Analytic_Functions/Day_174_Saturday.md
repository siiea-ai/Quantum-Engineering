# Day 174: Computational Lab — Complex Analysis in Practice

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Lab 1-3: Fundamentals |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Lab 4-5: Applications |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Lab 6: Integration Project |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of this lab, you will be able to:

1. Visualize complex functions using domain coloring and magnitude/phase plots
2. Numerically verify the Cauchy-Riemann equations
3. Implement conformal mappings interactively
4. Solve the Poisson equation numerically
5. Simulate quantum wave functions using complex analysis tools
6. Create publication-quality visualizations

---

## Lab 1: Domain Coloring Visualization

Domain coloring represents complex functions by mapping $w = f(z)$ to colors:
- **Hue:** encodes $\arg(w)$ (phase)
- **Brightness:** encodes $|w|$ (magnitude)

```python
"""
Lab 1: Domain Coloring for Complex Functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def domain_coloring(f, xlim=(-2, 2), ylim=(-2, 2), N=500, title="Domain Coloring"):
    """
    Create domain coloring visualization of complex function f.

    Hue: arg(f(z)) ∈ [0, 2π] → [0, 1]
    Saturation: 1 (full)
    Value: based on |f(z)| with logarithmic scaling
    """
    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    # Evaluate function
    W = f(Z)

    # Handle infinities/NaN
    W = np.where(np.isfinite(W), W, np.nan)

    # Compute HSV values
    H = (np.angle(W) + np.pi) / (2*np.pi)  # Hue from argument
    S = np.ones_like(H)  # Full saturation

    # Value from magnitude with sigmoid-like compression
    magnitude = np.abs(W)
    V = 1 - 1/(1 + magnitude**0.2)  # Compress infinite values
    V = np.clip(V, 0, 1)

    # Handle NaN
    H = np.nan_to_num(H, nan=0.5)
    V = np.nan_to_num(V, nan=0.0)

    # Convert HSV to RGB
    HSV = np.stack([H, S, V], axis=-1)
    RGB = hsv_to_rgb(HSV)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(RGB, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              origin='lower', aspect='auto')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_title(title)

    return fig, ax

# Example functions
fig1, ax1 = domain_coloring(lambda z: z**2, title="f(z) = z²")
plt.savefig('lab1_z_squared.png', dpi=150, bbox_inches='tight')

fig2, ax2 = domain_coloring(lambda z: (z**3 - 1)/(z**3 + 1),
                             title="f(z) = (z³-1)/(z³+1)")
plt.savefig('lab1_rational.png', dpi=150, bbox_inches='tight')

fig3, ax3 = domain_coloring(lambda z: np.exp(1/z), xlim=(-1, 1), ylim=(-1, 1),
                             title="f(z) = e^(1/z) - Essential Singularity")
plt.savefig('lab1_essential.png', dpi=150, bbox_inches='tight')

plt.show()

print("Lab 1 Complete: Domain coloring reveals phase and magnitude structure")
print("- Red/Yellow: positive real direction")
print("- Cyan/Blue: negative real direction")
print("- Poles appear as color wheels (all colors meeting)")
```

---

## Lab 2: Cauchy-Riemann Verification

Numerically verify that analytic functions satisfy Cauchy-Riemann equations.

```python
"""
Lab 2: Numerical Verification of Cauchy-Riemann Equations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def verify_cauchy_riemann(f, name, xlim=(-2, 2), ylim=(-2, 2), N=200):
    """
    Numerically verify Cauchy-Riemann equations:
    ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x
    """
    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    # Evaluate function
    W = f(Z)
    u = np.real(W)
    v = np.imag(W)

    # Compute numerical derivatives
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    # Check CR equations
    cr1_error = np.abs(du_dx - dv_dy)  # Should be ~0
    cr2_error = np.abs(du_dy + dv_dx)  # Should be ~0
    total_error = np.sqrt(cr1_error**2 + cr2_error**2)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].contourf(X, Y, u, levels=30, cmap='RdBu')
    axes[0, 0].set_title(f'u = Re[{name}]')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')

    axes[0, 1].contourf(X, Y, v, levels=30, cmap='RdBu')
    axes[0, 1].set_title(f'v = Im[{name}]')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')

    im = axes[0, 2].imshow(total_error, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                          origin='lower', cmap='hot', aspect='auto')
    axes[0, 2].set_title(f'CR Error (max={np.nanmax(total_error):.2e})')
    plt.colorbar(im, ax=axes[0, 2])

    # Plot derivatives
    axes[1, 0].contourf(X, Y, du_dx, levels=30, cmap='viridis')
    axes[1, 0].set_title('∂u/∂x')

    axes[1, 1].contourf(X, Y, dv_dy, levels=30, cmap='viridis')
    axes[1, 1].set_title('∂v/∂y (should match ∂u/∂x)')

    # Histogram of errors
    axes[1, 2].hist(total_error.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('CR Error')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Error Distribution')
    axes[1, 2].axvline(x=np.nanmean(total_error), color='r',
                       linestyle='--', label=f'Mean: {np.nanmean(total_error):.2e}')
    axes[1, 2].legend()

    plt.suptitle(f'Cauchy-Riemann Verification: {name}', fontsize=14)
    plt.tight_layout()

    return fig, np.nanmax(total_error)

# Test analytic functions
fig1, err1 = verify_cauchy_riemann(lambda z: z**2, "z²")
plt.savefig('lab2_cr_z2.png', dpi=150)
print(f"z²: Max CR error = {err1:.2e}")

fig2, err2 = verify_cauchy_riemann(lambda z: np.exp(z), "e^z")
plt.savefig('lab2_cr_exp.png', dpi=150)
print(f"e^z: Max CR error = {err2:.2e}")

# Test non-analytic function
fig3, err3 = verify_cauchy_riemann(lambda z: np.conj(z), "z̄ (conjugate)")
plt.savefig('lab2_cr_conjugate.png', dpi=150)
print(f"z̄: Max CR error = {err3:.2e} (should be ~√2 ≈ 1.41)")

plt.show()
```

---

## Lab 3: Interactive Conformal Mapping

Explore how conformal maps transform the complex plane.

```python
"""
Lab 3: Interactive Conformal Mapping Exploration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def conformal_map_explorer():
    """
    Interactive conformal mapping visualization with grid
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)

    # Initial grid
    x = np.linspace(-2, 2, 25)
    y = np.linspace(-2, 2, 25)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    def update_plot(a_real, a_imag, b_real, b_imag):
        """Update the conformal map w = (az + b)/(z + 1)"""
        a = a_real + 1j*a_imag
        b = b_real + 1j*b_imag

        W = (a*Z + b)/(Z + 1 + 0.01j)  # Avoid singularity

        axes[0].clear()
        axes[1].clear()

        # Original grid
        for i in range(len(x)):
            axes[0].plot(Z[i, :].real, Z[i, :].imag, 'b-', linewidth=0.5)
            axes[0].plot(Z[:, i].real, Z[:, i].imag, 'r-', linewidth=0.5)
        axes[0].set_title('z-plane (input)')
        axes[0].set_xlabel('Re(z)')
        axes[0].set_ylabel('Im(z)')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(-3, 3)
        axes[0].set_ylim(-3, 3)

        # Transformed grid
        for i in range(len(x)):
            w_row = W[i, :]
            w_col = W[:, i]
            # Filter out extreme values
            mask_row = np.abs(w_row) < 10
            mask_col = np.abs(w_col) < 10
            axes[1].plot(w_row.real[mask_row], w_row.imag[mask_row], 'b-', linewidth=0.5)
            axes[1].plot(w_col.real[mask_col], w_col.imag[mask_col], 'r-', linewidth=0.5)
        axes[1].set_title(f'w-plane: w = ({a:.1f}z + {b:.1f})/(z+1)')
        axes[1].set_xlabel('Re(w)')
        axes[1].set_ylabel('Im(w)')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(-5, 5)
        axes[1].set_ylim(-5, 5)

        fig.canvas.draw_idle()

    # Create sliders
    ax_a_real = plt.axes([0.1, 0.15, 0.35, 0.03])
    ax_a_imag = plt.axes([0.1, 0.10, 0.35, 0.03])
    ax_b_real = plt.axes([0.55, 0.15, 0.35, 0.03])
    ax_b_imag = plt.axes([0.55, 0.10, 0.35, 0.03])

    slider_a_real = Slider(ax_a_real, 'Re(a)', -3, 3, valinit=1)
    slider_a_imag = Slider(ax_a_imag, 'Im(a)', -3, 3, valinit=0)
    slider_b_real = Slider(ax_b_real, 'Re(b)', -3, 3, valinit=0)
    slider_b_imag = Slider(ax_b_imag, 'Im(b)', -3, 3, valinit=0)

    def update(val):
        update_plot(slider_a_real.val, slider_a_imag.val,
                   slider_b_real.val, slider_b_imag.val)

    slider_a_real.on_changed(update)
    slider_a_imag.on_changed(update)
    slider_b_real.on_changed(update)
    slider_b_imag.on_changed(update)

    # Initial plot
    update_plot(1, 0, 0, 0)

    plt.savefig('lab3_conformal_explorer.png', dpi=150, bbox_inches='tight')
    plt.show()

conformal_map_explorer()

print("\nLab 3 Complete: Adjust sliders to explore Möbius transformations")
print("Notice how circles/lines map to circles/lines!")
```

---

## Lab 4: Poisson Equation Solver

Solve the Dirichlet problem using finite differences.

```python
"""
Lab 4: Numerical Solution of Laplace's Equation (Dirichlet Problem)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_laplace_2d(N=50, max_iter=10000, tol=1e-6):
    """
    Solve ∇²u = 0 on unit disk with boundary condition u = cos(θ)
    Uses iterative Jacobi method
    """
    # Grid
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    # Mask for interior points
    interior = R < 1 - dx
    boundary = (R >= 1 - dx) & (R <= 1 + dx)

    # Initialize solution
    u = np.zeros((N, N))

    # Apply boundary condition: u = cos(θ) on unit circle
    u[boundary] = np.cos(Theta[boundary])

    # Iterative solution (Jacobi method)
    for iteration in range(max_iter):
        u_old = u.copy()

        # Update interior points: average of neighbors
        for i in range(1, N-1):
            for j in range(1, N-1):
                if interior[i, j]:
                    u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] +
                                      u_old[i, j+1] + u_old[i, j-1])

        # Check convergence
        change = np.max(np.abs(u - u_old))
        if change < tol:
            print(f"Converged after {iteration} iterations")
            break

    return X, Y, u, R, interior

# Solve and visualize
X, Y, u_numerical, R, interior = solve_laplace_2d(N=100)

# Exact solution: u = r cos(θ) = x (from Poisson formula)
u_exact = X.copy()
u_exact[R > 1] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Numerical solution
ax = axes[0]
mask = R > 1
u_plot = u_numerical.copy()
u_plot[mask] = np.nan
contour1 = ax.contourf(X, Y, u_plot, levels=30, cmap='RdBu')
ax.plot(np.cos(np.linspace(0, 2*np.pi, 100)),
        np.sin(np.linspace(0, 2*np.pi, 100)), 'k-', linewidth=2)
ax.set_title('Numerical Solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.colorbar(contour1, ax=ax)

# Exact solution
ax = axes[1]
contour2 = ax.contourf(X, Y, u_exact, levels=30, cmap='RdBu')
ax.plot(np.cos(np.linspace(0, 2*np.pi, 100)),
        np.sin(np.linspace(0, 2*np.pi, 100)), 'k-', linewidth=2)
ax.set_title('Exact Solution: u = x')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.colorbar(contour2, ax=ax)

# Error
ax = axes[2]
error = np.abs(u_numerical - X)
error[mask] = np.nan
contour3 = ax.contourf(X, Y, error, levels=30, cmap='hot')
ax.plot(np.cos(np.linspace(0, 2*np.pi, 100)),
        np.sin(np.linspace(0, 2*np.pi, 100)), 'k-', linewidth=2)
ax.set_title(f'Error (max = {np.nanmax(error):.2e})')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.colorbar(contour3, ax=ax)

plt.tight_layout()
plt.savefig('lab4_laplace_solver.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLab 4 Complete: Solved Dirichlet problem on unit disk")
print("Boundary: u = cos(θ), Interior: Laplace equation ∇²u = 0")
print("Exact solution: u = r cos(θ) = x (Poisson formula)")
```

---

## Lab 5: Quantum Wave Function Visualization

Visualize complex wave functions and their probability densities.

```python
"""
Lab 5: Quantum Wave Functions as Complex Functions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_2d_wavefunction():
    """
    Visualize 2D quantum harmonic oscillator wave functions
    """
    # Grid
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)

    # Harmonic oscillator ground state (n=0, m=0)
    psi_00 = np.exp(-(X**2 + Y**2)/2) / np.sqrt(np.pi)

    # First excited state (n=1, m=0)
    psi_10 = X * np.exp(-(X**2 + Y**2)/2) * np.sqrt(2/np.pi)

    # Angular momentum state (n=1, m=1) - complex!
    psi_11 = (X + 1j*Y) * np.exp(-(X**2 + Y**2)/2) / np.sqrt(np.pi)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Ground state
    ax = axes[0, 0]
    contour1 = ax.contourf(X, Y, np.abs(psi_00)**2, levels=30, cmap='viridis')
    ax.set_title('|ψ₀₀|² (Ground State)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(contour1, ax=ax)

    # First excited (real)
    ax = axes[0, 1]
    contour2 = ax.contourf(X, Y, np.abs(psi_10)**2, levels=30, cmap='viridis')
    ax.set_title('|ψ₁₀|² (First Excited)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(contour2, ax=ax)

    # Angular momentum state (probability)
    ax = axes[0, 2]
    contour3 = ax.contourf(X, Y, np.abs(psi_11)**2, levels=30, cmap='viridis')
    ax.set_title('|ψ₁₁|² (m=1 Angular Momentum)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(contour3, ax=ax)

    # Phase of complex wave function
    ax = axes[1, 0]
    phase = np.angle(psi_11)
    contour4 = ax.contourf(X, Y, phase, levels=30, cmap='hsv')
    ax.set_title('Phase of ψ₁₁ = arg(ψ)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(contour4, ax=ax, label='Phase (rad)')

    # Real part
    ax = axes[1, 1]
    contour5 = ax.contourf(X, Y, np.real(psi_11), levels=30, cmap='RdBu')
    ax.set_title('Re(ψ₁₁)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(contour5, ax=ax)

    # Imaginary part
    ax = axes[1, 2]
    contour6 = ax.contourf(X, Y, np.imag(psi_11), levels=30, cmap='RdBu')
    ax.set_title('Im(ψ₁₁)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(contour6, ax=ax)

    plt.suptitle('2D Quantum Harmonic Oscillator Wave Functions', fontsize=14)
    plt.tight_layout()
    plt.savefig('lab5_quantum_wavefunctions.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_2d_wavefunction()

# Time evolution
def time_evolution():
    """
    Animate time evolution of superposition state
    """
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)

    # Ground and first excited states
    psi_0 = np.exp(-(X**2 + Y**2)/2) / np.sqrt(np.pi)
    psi_1 = X * np.exp(-(X**2 + Y**2)/2) * np.sqrt(2/np.pi)

    # Energies (in units of ℏω)
    E0, E1 = 1, 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    times = [0, 0.25, 0.5, 0.75, 1.0, 1.25]  # in units of 2π/ω

    for idx, t in enumerate(times):
        # Time-evolved superposition
        psi_t = (psi_0 * np.exp(-1j*E0*2*np.pi*t) +
                 psi_1 * np.exp(-1j*E1*2*np.pi*t)) / np.sqrt(2)

        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        prob = np.abs(psi_t)**2
        contour = ax.contourf(X, Y, prob, levels=30, cmap='viridis')
        ax.set_title(f't = {t:.2f} × 2π/ω')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.suptitle('Time Evolution of Superposition: ψ = (ψ₀ + ψ₁)/√2', fontsize=14)
    plt.tight_layout()
    plt.savefig('lab5_time_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()

time_evolution()

print("\nLab 5 Complete: Quantum wave functions visualized")
print("Key insights:")
print("- Complex wave functions encode both amplitude and phase")
print("- Angular momentum states have circulating phase")
print("- Time evolution creates oscillating probability density")
```

---

## Lab 6: Integration Project — Complex Potential Flow

Combine all concepts to solve a fluid dynamics problem.

```python
"""
Lab 6: Complex Potential Flow Around Objects
Integrating conformal mapping, harmonic functions, and visualization
"""

import numpy as np
import matplotlib.pyplot as plt

def complex_potential_flow():
    """
    Simulate potential flow around a cylinder using complex analysis

    Complex potential: Ω(z) = U(z + a²/z) for uniform flow around cylinder radius a
    Velocity: dΩ/dz = U(1 - a²/z²)
    """
    # Parameters
    U = 1.0  # Free stream velocity
    a = 1.0  # Cylinder radius

    # Grid
    x = np.linspace(-4, 4, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    # Complex potential (uniform flow + flow around cylinder)
    Omega = U * (Z + a**2/Z)

    # Velocity potential and stream function
    phi = np.real(Omega)  # Velocity potential
    psi = np.imag(Omega)  # Stream function

    # Velocity field (dΩ/dz)
    dOmega_dz = U * (1 - a**2/Z**2)
    u = np.real(dOmega_dz)  # x-component of velocity
    v = -np.imag(dOmega_dz)  # y-component (note sign for flow)

    # Mask inside cylinder
    mask = np.abs(Z) < a
    phi[mask] = np.nan
    psi[mask] = np.nan
    u[mask] = 0
    v[mask] = 0

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Stream function (streamlines)
    ax = axes[0, 0]
    levels_psi = np.linspace(-3, 3, 31)
    contour1 = ax.contour(X, Y, psi, levels=levels_psi, colors='blue', linewidths=0.8)
    circle = plt.Circle((0, 0), a, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_title('Streamlines (ψ = const)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)

    # Velocity potential (equipotentials)
    ax = axes[0, 1]
    levels_phi = np.linspace(-4, 4, 31)
    contour2 = ax.contour(X, Y, phi, levels=levels_phi, colors='red', linewidths=0.8)
    circle = plt.Circle((0, 0), a, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_title('Equipotentials (φ = const)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)

    # Both overlaid (orthogonality)
    ax = axes[1, 0]
    ax.contour(X, Y, psi, levels=levels_psi[::2], colors='blue', linewidths=0.6)
    ax.contour(X, Y, phi, levels=levels_phi[::2], colors='red', linewidths=0.6)
    circle = plt.Circle((0, 0), a, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_title('Streamlines (blue) ⟂ Equipotentials (red)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)

    # Velocity magnitude
    ax = axes[1, 1]
    speed = np.sqrt(u**2 + v**2)
    speed[mask] = np.nan
    im = ax.contourf(X, Y, speed, levels=30, cmap='hot')
    circle = plt.Circle((0, 0), a, color='white', fill=True)
    ax.add_patch(circle)

    # Add velocity vectors (subsampled)
    skip = 15
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
              color='cyan', alpha=0.7, scale=20)

    ax.set_title('Velocity Magnitude with Flow Vectors')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    plt.colorbar(im, ax=ax, label='|v|')

    plt.suptitle('Complex Potential Flow Around Cylinder: Ω(z) = U(z + a²/z)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('lab6_cylinder_flow.png', dpi=150, bbox_inches='tight')
    plt.show()

    return phi, psi, u, v

phi, psi, u, v = complex_potential_flow()

print("\n" + "="*60)
print("DAY 174 COMPUTATIONAL LAB - COMPLETE")
print("="*60)
print("\nLabs Completed:")
print("1. Domain Coloring: Visualize magnitude and phase simultaneously")
print("2. Cauchy-Riemann: Numerical verification of analyticity")
print("3. Conformal Maps: Interactive exploration with sliders")
print("4. Laplace Solver: Finite difference solution of Dirichlet problem")
print("5. Quantum Waves: Complex wave functions and time evolution")
print("6. Potential Flow: Integration of all concepts in fluid dynamics")
print("\nKey Takeaways:")
print("• Complex analysis provides powerful computational tools")
print("• Conformal mapping transforms problems to simpler geometries")
print("• Harmonic functions arise naturally in physics")
print("• Complex potential unifies velocity potential and stream function")
```

---

## Summary

### Labs Completed

| Lab | Topic | Key Skill |
|-----|-------|-----------|
| 1 | Domain Coloring | Visualize complex functions |
| 2 | Cauchy-Riemann | Verify analyticity numerically |
| 3 | Conformal Maps | Interactive transformation exploration |
| 4 | Poisson Equation | Solve Laplace equation numerically |
| 5 | Quantum Waves | Visualize complex wave functions |
| 6 | Potential Flow | Integrate all concepts |

### Key Python Libraries

- `numpy`: Array operations, complex numbers
- `matplotlib`: Visualization
- `scipy`: Numerical methods, solvers
- `mpl_toolkits.mplot3d`: 3D plots

---

## Daily Checklist

- [ ] I can create domain coloring visualizations
- [ ] I can numerically verify Cauchy-Riemann equations
- [ ] I understand how conformal maps transform grids
- [ ] I can solve Laplace's equation numerically
- [ ] I can visualize quantum wave functions as complex fields
- [ ] I understand how complex potential describes fluid flow

---

## Preview: Day 175

Tomorrow: **Week Review** — Synthesizing the week's material!
- Concept map of Week 25 topics
- Problem sets covering all material
- Self-assessment
- Preparation for Week 26: Contour Integration

---

*"The computer is incredibly fast, accurate, and stupid. Man is incredibly slow, inaccurate, and brilliant. The marriage of the two is a force beyond calculation."*
— Leo Cherne
