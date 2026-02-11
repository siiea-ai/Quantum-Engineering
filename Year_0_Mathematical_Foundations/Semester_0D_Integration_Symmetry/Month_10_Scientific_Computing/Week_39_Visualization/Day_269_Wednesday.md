# Day 269: 3D Visualization

## Schedule Overview
**Date**: Week 39, Day 3 (Wednesday)
**Duration**: 7 hours
**Theme**: Visualizing Quantum Mechanical Systems in Three Dimensions

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | 3D plot fundamentals, surface plots, wire frames |
| Afternoon | 2.5 hours | Quantum orbital visualization, isosurfaces |
| Evening | 1.5 hours | Computational lab: Complete orbital visualization toolkit |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Create 3D axes and configure viewing angles
2. Generate surface plots for potential energy landscapes
3. Visualize 3D probability densities using color mapping
4. Create wire frame and contour plots in 3D
5. Render hydrogen atom orbitals with spherical harmonics
6. Combine multiple 3D visualizations in single figures

---

## Core Content

### 1. Introduction to mplot3d

Matplotlib's `mplot3d` toolkit provides 3D plotting capabilities built on the 2D infrastructure.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Alternative syntax
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 8))
```

#### Viewing Angle Control

```python
# Set viewing angle
ax.view_init(elev=30, azim=45)  # elevation and azimuthal angles

# Interactive rotation is enabled by default in notebooks
# In scripts, use plt.show() for interactive window
```

### 2. Surface Plots

Surface plots display a function of two variables z = f(x, y).

#### Basic Surface Plot

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create mesh grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# 2D harmonic oscillator potential
V = 0.5 * (X**2 + Y**2)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot with colormap
surf = ax.plot_surface(X, Y, V, cmap='viridis', alpha=0.8,
                       rstride=2, cstride=2, antialiased=True)

# Labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('V(x, y)')
ax.set_title('2D Harmonic Oscillator Potential')

# Colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Energy')

plt.show()
```

#### Quantum Potential Landscapes

```python
def plot_potential_surface(V_func, x_range, y_range, title="Potential Energy Surface"):
    """
    Generic 3D potential energy surface plotter.

    Parameters
    ----------
    V_func : callable
        Function V(X, Y) returning potential values
    x_range, y_range : tuple
        (min, max, n_points) for each dimension
    """
    x = np.linspace(*x_range)
    y = np.linspace(*y_range)
    X, Y = np.meshgrid(x, y)
    V = V_func(X, Y)

    fig = plt.figure(figsize=(14, 10))

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, V, cmap='plasma', alpha=0.9,
                           rstride=2, cstride=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('V(x, y)')
    ax1.set_title(title + ' (3D)')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # Contour projection
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, V, levels=30, cmap='plasma')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(title + ' (Contours)')
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2)

    plt.tight_layout()
    return fig

# Example: Double well potential
def double_well(X, Y):
    return (X**2 - 1)**2 + 0.5 * Y**2

fig = plot_potential_surface(double_well, (-2, 2, 100), (-2, 2, 100),
                            "Double Well Potential")
plt.savefig('double_well_3d.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3. Wire Frames and Contour Plots in 3D

#### Wire Frame Plots

Wire frames show the surface structure without filling:

```python
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                         subplot_kw={'projection': '3d'})

# Surface plot
axes[0].plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
axes[0].set_title('Surface')

# Wire frame
axes[1].plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='blue', alpha=0.8)
axes[1].set_title('Wire Frame')

# Combined
axes[2].plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
axes[2].plot_wireframe(X, Y, Z, rstride=5, cstride=5, color='black', linewidth=0.5)
axes[2].set_title('Combined')

for ax in axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

plt.tight_layout()
plt.show()
```

#### 3D Contour Plots

```python
fig = plt.figure(figsize=(12, 5))

# Contour at base
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax1.contour(X, Y, Z, zdir='z', offset=-1.5, cmap='viridis')
ax1.set_zlim(-1.5, 1)
ax1.set_title('Contour at Base')

# Multiple contour levels
ax2 = fig.add_subplot(122, projection='3d')
ax2.contour3D(X, Y, Z, 50, cmap='viridis')
ax2.set_title('3D Contours')

plt.tight_layout()
plt.show()
```

### 4. 3D Probability Densities

Quantum wave functions in 3D require special visualization techniques.

#### Isosurface Approach

For 3D probability densities $$|\psi(x,y,z)|^2$$, isosurfaces show surfaces of constant probability:

```python
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure  # For marching cubes algorithm

def plot_3d_probability_isosurface(x, y, z, psi, iso_value=0.1, ax=None):
    """
    Plot isosurface of probability density |ψ|².

    Uses marching cubes algorithm to find surface.
    """
    prob = np.abs(psi)**2
    prob_normalized = prob / prob.max()

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    try:
        # Extract isosurface using marching cubes
        verts, faces, _, _ = measure.marching_cubes(prob_normalized, iso_value)

        # Scale vertices to actual coordinates
        verts_scaled = np.zeros_like(verts)
        verts_scaled[:, 0] = x[0] + verts[:, 0] * (x[-1] - x[0]) / len(x)
        verts_scaled[:, 1] = y[0] + verts[:, 1] * (y[-1] - y[0]) / len(y)
        verts_scaled[:, 2] = z[0] + verts[:, 2] * (z[-1] - z[0]) / len(z)

        # Create mesh
        mesh = Poly3DCollection(verts_scaled[faces], alpha=0.5)
        mesh.set_facecolor('cyan')
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_zlim(z[0], z[-1])

    except Exception as e:
        print(f"Isosurface extraction failed: {e}")
        # Fallback: scatter plot
        prob_flat = prob_normalized.flatten()
        mask = prob_flat > iso_value
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        ax.scatter(X.flatten()[mask], Y.flatten()[mask], Z.flatten()[mask],
                  c=prob_flat[mask], cmap='viridis', alpha=0.3, s=1)

    return ax
```

#### Slice Visualization

An alternative is to show 2D slices through the 3D density:

```python
def plot_3d_slices(x, y, z, psi, n_slices=5):
    """
    Show 2D slices through a 3D probability density.
    """
    prob = np.abs(psi)**2

    fig = plt.figure(figsize=(15, 5))

    # XY slices (varying z)
    ax1 = fig.add_subplot(131, projection='3d')
    z_indices = np.linspace(0, len(z)-1, n_slices, dtype=int)
    for iz in z_indices:
        X_slice, Y_slice = np.meshgrid(x, y)
        Z_slice = np.full_like(X_slice, z[iz])
        colors = plt.cm.viridis(prob[:, :, iz] / prob.max())
        ax1.plot_surface(X_slice, Y_slice, Z_slice, facecolors=colors, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('XY Slices')

    # XZ slices
    ax2 = fig.add_subplot(132, projection='3d')
    y_indices = np.linspace(0, len(y)-1, n_slices, dtype=int)
    for iy in y_indices:
        X_slice, Z_slice = np.meshgrid(x, z)
        Y_slice = np.full_like(X_slice, y[iy])
        colors = plt.cm.viridis(prob[:, iy, :].T / prob.max())
        ax2.plot_surface(X_slice, Y_slice, Z_slice, facecolors=colors, alpha=0.7)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('XZ Slices')

    # YZ slices
    ax3 = fig.add_subplot(133, projection='3d')
    x_indices = np.linspace(0, len(x)-1, n_slices, dtype=int)
    for ix in x_indices:
        Y_slice, Z_slice = np.meshgrid(y, z)
        X_slice = np.full_like(Y_slice, x[ix])
        colors = plt.cm.viridis(prob[ix, :, :].T / prob.max())
        ax3.plot_surface(X_slice, Y_slice, Z_slice, facecolors=colors, alpha=0.7)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('YZ Slices')

    plt.tight_layout()
    return fig
```

### 5. Hydrogen Atom Orbitals

The hydrogen atom wave functions are the product of radial and angular parts:
$$\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)$$

```python
from scipy.special import sph_harm, assoc_laguerre
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def hydrogen_radial(n, l, r, a0=1):
    """
    Hydrogen atom radial wave function R_nl(r).

    Parameters
    ----------
    n : int
        Principal quantum number
    l : int
        Orbital angular momentum quantum number
    r : array
        Radial distance in units of Bohr radius
    a0 : float
        Bohr radius (default 1 for atomic units)
    """
    rho = 2 * r / (n * a0)

    # Normalization
    from math import factorial
    norm = np.sqrt((2/(n*a0))**3 * factorial(n-l-1) / (2*n*factorial(n+l)))

    # Associated Laguerre polynomial
    L = assoc_laguerre(rho, n-l-1, 2*l+1)

    return norm * np.exp(-rho/2) * rho**l * L

def hydrogen_angular(l, m, theta, phi):
    """
    Spherical harmonics Y_l^m(θ, φ).

    Returns real form for visualization.
    """
    # scipy's sph_harm uses (m, l, phi, theta) convention
    Y = sph_harm(m, l, phi, theta)

    # Return real part for m >= 0, imaginary for m < 0
    if m >= 0:
        return Y.real * np.sqrt(2) if m != 0 else Y.real
    else:
        return Y.imag * np.sqrt(2)

def hydrogen_wavefunction(n, l, m, r, theta, phi, a0=1):
    """Complete hydrogen wave function."""
    R = hydrogen_radial(n, l, r, a0)
    Y = sph_harm(m, l, phi, theta)
    return R * Y

def plot_orbital(n, l, m, grid_size=50, plot_type='surface'):
    """
    Visualize hydrogen atom orbital.

    Parameters
    ----------
    n, l, m : int
        Quantum numbers
    grid_size : int
        Number of points in each dimension
    plot_type : str
        'surface' for angular distribution, 'density' for probability cloud
    """
    # Angular grid
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2*np.pi, grid_size)
    THETA, PHI = np.meshgrid(theta, phi)

    # Compute spherical harmonic
    Y = sph_harm(m, l, PHI, THETA)

    # Use |Y|² for surface plot
    R_angular = np.abs(Y)**2

    # Convert to Cartesian for plotting
    X = R_angular * np.sin(THETA) * np.cos(PHI)
    Y_cart = R_angular * np.sin(THETA) * np.sin(PHI)
    Z = R_angular * np.cos(THETA)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color by sign of real part
    Y_real = sph_harm(m, l, PHI, THETA).real
    colors = np.where(Y_real >= 0, 'blue', 'red')

    # Create custom colormap
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=-np.abs(Y_real).max(), vmax=np.abs(Y_real).max())
    facecolors = plt.cm.RdBu(norm(Y_real))

    surf = ax.plot_surface(X, Y_cart, Z, facecolors=facecolors,
                          alpha=0.8, rstride=1, cstride=1, antialiased=True)

    # Set equal aspect ratio
    max_range = max(np.abs(X).max(), np.abs(Y_cart).max(), np.abs(Z).max())
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Orbital name
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orbital_letter = orbital_names.get(l, str(l))
    ax.set_title(f'Hydrogen Orbital: {n}{orbital_letter} (l={l}, m={m})')

    return fig, ax

# Plot first few orbitals
fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                         subplot_kw={'projection': '3d'})

orbitals = [(1, 0, 0), (2, 0, 0), (2, 1, 0),
            (2, 1, 1), (3, 2, 0), (3, 2, 2)]

for ax, (n, l, m) in zip(axes.flatten(), orbitals):
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi)

    Y = sph_harm(m, l, PHI, THETA)
    R = np.abs(Y)**2

    X = R * np.sin(THETA) * np.cos(PHI)
    Y_c = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    Y_real = Y.real
    norm = plt.Normalize(vmin=-np.abs(Y_real).max(), vmax=np.abs(Y_real).max())
    colors = plt.cm.RdBu(norm(Y_real))

    ax.plot_surface(X, Y_c, Z, facecolors=colors, alpha=0.8,
                   rstride=1, cstride=1)

    max_r = max(np.abs(X).max(), np.abs(Y_c).max(), np.abs(Z).max()) * 1.1
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-max_r, max_r)

    orbital_names = {0: 's', 1: 'p', 2: 'd'}
    ax.set_title(f'{n}{orbital_names[l]} (m={m})')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

plt.suptitle('Hydrogen Atom Angular Wave Functions $|Y_l^m|^2$', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('hydrogen_orbitals.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6. 3D Scatter Plots for Probability Clouds

Another approach is Monte Carlo sampling from the probability distribution:

```python
def sample_orbital_positions(n, l, m, n_samples=10000, r_max=None):
    """
    Sample positions from hydrogen orbital probability distribution.

    Uses rejection sampling.
    """
    if r_max is None:
        r_max = 4 * n**2  # Rough estimate of orbital extent

    samples = []
    while len(samples) < n_samples:
        # Generate random positions in sphere
        r = r_max * np.random.rand(n_samples)
        theta = np.arccos(2*np.random.rand(n_samples) - 1)
        phi = 2*np.pi * np.random.rand(n_samples)

        # Compute probability density
        psi = hydrogen_wavefunction(n, l, m, r, theta, phi)
        prob = np.abs(psi)**2 * r**2  # Include Jacobian

        # Rejection sampling
        accept = np.random.rand(n_samples) < prob / prob.max()
        r_accept = r[accept]
        theta_accept = theta[accept]
        phi_accept = phi[accept]

        # Convert to Cartesian
        x = r_accept * np.sin(theta_accept) * np.cos(phi_accept)
        y = r_accept * np.sin(theta_accept) * np.sin(phi_accept)
        z = r_accept * np.cos(theta_accept)

        samples.extend(list(zip(x, y, z)))

    samples = np.array(samples[:n_samples])
    return samples[:, 0], samples[:, 1], samples[:, 2]

# Probability cloud visualization
fig = plt.figure(figsize=(15, 5))

for i, (n, l, m) in enumerate([(2, 1, 0), (3, 2, 0), (3, 2, 1)]):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')

    try:
        x, y, z = sample_orbital_positions(n, l, m, n_samples=5000)
        r = np.sqrt(x**2 + y**2 + z**2)
        ax.scatter(x, y, z, c=r, cmap='viridis', s=1, alpha=0.5)
    except:
        # Fallback to simple grid visualization
        pass

    orbital_names = {0: 's', 1: 'p', 2: 'd'}
    ax.set_title(f'{n}{orbital_names[l]} (m={m}) Probability Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

plt.tight_layout()
plt.show()
```

---

## Quantum Mechanics Connection

### Angular Momentum Visualization

The spherical harmonics $$Y_l^m(\theta, \phi)$$ are eigenfunctions of both $$\hat{L}^2$$ and $$\hat{L}_z$$:

$$\hat{L}^2 Y_l^m = \hbar^2 l(l+1) Y_l^m$$
$$\hat{L}_z Y_l^m = \hbar m Y_l^m$$

The shapes of orbitals directly reflect these symmetries:
- **s orbitals (l=0)**: Spherically symmetric, no angular nodes
- **p orbitals (l=1)**: One nodal plane, directional
- **d orbitals (l=2)**: Two nodal planes, complex shapes

```python
def visualize_angular_momentum_states(l_max=2):
    """
    Show all spherical harmonics up to l_max.
    """
    n_plots = sum(2*l + 1 for l in range(l_max + 1))
    n_cols = 5
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4*n_cols, 4*n_rows))

    plot_idx = 1
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')

            # Angular grid
            theta = np.linspace(0, np.pi, 40)
            phi = np.linspace(0, 2*np.pi, 40)
            THETA, PHI = np.meshgrid(theta, phi)

            Y = sph_harm(m, l, PHI, THETA)
            R = np.abs(Y)

            X = R * np.sin(THETA) * np.cos(PHI)
            Y_c = R * np.sin(THETA) * np.sin(PHI)
            Z = R * np.cos(THETA)

            # Color by phase
            phase = np.angle(Y)
            norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
            colors = plt.cm.hsv(norm(phase))

            ax.plot_surface(X, Y_c, Z, facecolors=colors, alpha=0.8,
                           rstride=1, cstride=1)

            max_r = 1.2
            ax.set_xlim(-max_r, max_r)
            ax.set_ylim(-max_r, max_r)
            ax.set_zlim(-max_r, max_r)
            ax.set_title(f'$Y_{l}^{{{m}}}$')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            plot_idx += 1

    plt.suptitle('Spherical Harmonics (color = phase)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

fig = visualize_angular_momentum_states(l_max=2)
plt.savefig('spherical_harmonics.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Orbital Hybridization

In chemistry, atomic orbitals combine to form hybrid orbitals:

```python
def sp3_hybrid_visualization():
    """Visualize sp³ hybridization in 3D."""
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi)

    # s orbital (l=0)
    Y_s = sph_harm(0, 0, PHI, THETA)

    # p orbitals (l=1)
    Y_px = (sph_harm(-1, 1, PHI, THETA) - sph_harm(1, 1, PHI, THETA)) / np.sqrt(2)
    Y_py = 1j * (sph_harm(-1, 1, PHI, THETA) + sph_harm(1, 1, PHI, THETA)) / np.sqrt(2)
    Y_pz = sph_harm(0, 1, PHI, THETA)

    # sp³ hybrids (tetrahedral directions)
    hybrids = [
        (Y_s + Y_px + Y_py + Y_pz) / 2,
        (Y_s + Y_px - Y_py - Y_pz) / 2,
        (Y_s - Y_px + Y_py - Y_pz) / 2,
        (Y_s - Y_px - Y_py + Y_pz) / 2
    ]

    colors_list = ['red', 'blue', 'green', 'orange']

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for hybrid, color in zip(hybrids, colors_list):
        R = np.abs(hybrid.real)

        X = R * np.sin(THETA) * np.cos(PHI)
        Y_c = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        ax.plot_surface(X, Y_c, Z, color=color, alpha=0.6,
                       rstride=2, cstride=2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('sp³ Hybrid Orbitals (Tetrahedral Geometry)')

    # Equal aspect ratio
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    return fig

fig = sp3_hybrid_visualization()
plt.savefig('sp3_hybrids.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Worked Examples

### Example 1: 3D Harmonic Oscillator

**Problem**: Visualize the 3D isotropic harmonic oscillator ground state probability density.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D harmonic oscillator ground state: ψ ∝ exp(-(x²+y²+z²)/2)
def ho_3d_ground_state(x, y, z, omega=1, m=1, hbar=1):
    """3D harmonic oscillator ground state."""
    alpha = m * omega / hbar
    norm = (alpha / np.pi)**(3/4)
    return norm * np.exp(-alpha * (x**2 + y**2 + z**2) / 2)

# Create 3D grid
n_points = 30
x = np.linspace(-3, 3, n_points)
y = np.linspace(-3, 3, n_points)
z = np.linspace(-3, 3, n_points)
X, Y, Z = np.meshgrid(x, y, z)

psi = ho_3d_ground_state(X, Y, Z)
prob = np.abs(psi)**2

# Visualization 1: Slices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# XY plane (z=0)
im1 = axes[0].contourf(x, y, prob[:, :, n_points//2], 20, cmap='viridis')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('XY Plane (z=0)')
axes[0].set_aspect('equal')
plt.colorbar(im1, ax=axes[0])

# XZ plane (y=0)
im2 = axes[1].contourf(x, z, prob[:, n_points//2, :].T, 20, cmap='viridis')
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[1].set_title('XZ Plane (y=0)')
axes[1].set_aspect('equal')
plt.colorbar(im2, ax=axes[1])

# YZ plane (x=0)
im3 = axes[2].contourf(y, z, prob[n_points//2, :, :].T, 20, cmap='viridis')
axes[2].set_xlabel('y')
axes[2].set_ylabel('z')
axes[2].set_title('YZ Plane (x=0)')
axes[2].set_aspect('equal')
plt.colorbar(im3, ax=axes[2])

plt.suptitle('3D Harmonic Oscillator Ground State $|\\psi_{000}|^2$', fontsize=14)
plt.tight_layout()
plt.savefig('ho_3d_slices.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 2: Radial distribution
r = np.sqrt(X**2 + Y**2 + Z**2)
r_flat = r.flatten()
prob_flat = prob.flatten()

# Bin by radius
r_bins = np.linspace(0, 3, 50)
prob_radial = np.zeros(len(r_bins) - 1)
for i in range(len(r_bins) - 1):
    mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
    if mask.sum() > 0:
        prob_radial[i] = prob_flat[mask].mean()

r_centers = (r_bins[:-1] + r_bins[1:]) / 2

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(r_centers, prob_radial, 'b-', linewidth=2, label='Numerical')
# Analytical: P(r) ∝ r² |ψ(r)|² for 3D
ax.plot(r_centers, (1/np.pi)**(3/2) * np.exp(-r_centers**2), 'r--',
       linewidth=2, label='Analytical')
ax.set_xlabel('r')
ax.set_ylabel(r'$|\psi(r)|^2$')
ax.set_title('Radial Probability Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('ho_3d_radial.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Example 2: Particle in a 3D Box

**Problem**: Visualize the (1,1,2) excited state of a particle in a cubic box.

**Solution**:
```python
def box_3d_state(nx, ny, nz, x, y, z, L=1):
    """
    Particle in a cubic box wave function.

    ψ_{n_x, n_y, n_z} = (2/L)^(3/2) sin(n_x πx/L) sin(n_y πy/L) sin(n_z πz/L)
    """
    norm = (2/L)**(3/2)
    return norm * (np.sin(nx * np.pi * x / L) *
                   np.sin(ny * np.pi * y / L) *
                   np.sin(nz * np.pi * z / L))

# Create grid
L = 1
n_points = 40
x = np.linspace(0, L, n_points)
y = np.linspace(0, L, n_points)
z = np.linspace(0, L, n_points)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# (1,1,2) state
psi = box_3d_state(1, 1, 2, X, Y, Z, L)
prob = np.abs(psi)**2

# Create isosurface-like visualization using scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Threshold for high probability regions
threshold = 0.5 * prob.max()
mask = prob > threshold

ax.scatter(X[mask], Y[mask], Z[mask], c=prob[mask],
          cmap='plasma', s=20, alpha=0.6)

# Mark nodal plane (z = L/2 for nz=2)
xx, yy = np.meshgrid(np.linspace(0, L, 20), np.linspace(0, L, 20))
zz = np.full_like(xx, L/2)
ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Particle in Box: (1,1,2) State\nGray plane = nodal surface')

# Draw box edges
for i in [0, L]:
    for j in [0, L]:
        ax.plot([0, L], [i, i], [j, j], 'k-', linewidth=1)
        ax.plot([i, i], [0, L], [j, j], 'k-', linewidth=1)
        ax.plot([i, i], [j, j], [0, L], 'k-', linewidth=1)

plt.savefig('box_3d_state.png', dpi=150, bbox_inches='tight')
plt.show()

# Energy calculation
def energy_box(nx, ny, nz, L=1, m=1, hbar=1):
    return (hbar**2 * np.pi**2 / (2 * m * L**2)) * (nx**2 + ny**2 + nz**2)

E_112 = energy_box(1, 1, 2)
print(f"Energy of (1,1,2) state: E = {E_112:.4f} (units of ℏ²/2mL²)")
```

### Example 3: Hydrogen 2p Orbital Cloud

**Problem**: Create a probability cloud visualization of the hydrogen 2p orbital.

**Solution**:
```python
from scipy.special import sph_harm

def hydrogen_2p_z(r, theta, phi, a0=1):
    """
    Hydrogen 2p_z orbital (n=2, l=1, m=0).

    ψ_{2,1,0} = (1/4√(2π)) (1/a₀)^(3/2) (r/a₀) exp(-r/2a₀) cos(θ)
    """
    rho = r / a0
    R = (1/(2*np.sqrt(6))) * (1/a0)**(3/2) * rho * np.exp(-rho/2)
    Y = np.sqrt(3/(4*np.pi)) * np.cos(theta)
    return R * Y

# Monte Carlo sampling
n_samples = 50000
r_max = 15

# Sample in spherical coordinates
r = r_max * np.random.rand(n_samples*10)
theta = np.arccos(2*np.random.rand(n_samples*10) - 1)
phi = 2*np.pi * np.random.rand(n_samples*10)

psi = hydrogen_2p_z(r, theta, phi)
prob = np.abs(psi)**2 * r**2  # r² from Jacobian

# Rejection sampling
accept = np.random.rand(len(prob)) < prob / prob.max()
r_acc = r[accept][:n_samples]
theta_acc = theta[accept][:n_samples]
phi_acc = phi[accept][:n_samples]

# Convert to Cartesian
x = r_acc * np.sin(theta_acc) * np.cos(phi_acc)
y = r_acc * np.sin(theta_acc) * np.sin(phi_acc)
z = r_acc * np.cos(theta_acc)

# Sign of wave function for coloring
psi_sign = np.sign(hydrogen_2p_z(r_acc, theta_acc, phi_acc))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Color by sign (positive = blue, negative = red)
colors = np.where(psi_sign > 0, 'blue', 'red')
ax.scatter(x, y, z, c=colors, s=1, alpha=0.3)

ax.set_xlabel('x (Bohr radii)')
ax.set_ylabel('y (Bohr radii)')
ax.set_zlabel('z (Bohr radii)')
ax.set_title('Hydrogen 2p_z Orbital Probability Cloud\n(Blue = +, Red = -)')

# Equal aspect ratio
max_range = 10
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

# Mark nodal plane (z=0)
xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 20),
                     np.linspace(-max_range, max_range, 20))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

plt.savefig('hydrogen_2pz_cloud.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Practice Problems

### Level 1: Direct Application

1. **Basic Surface Plot**: Create a 3D surface plot of $$z = \sin(x)\cos(y)$$ for $$x, y \in [-\pi, \pi]$$.

2. **Wire Frame**: Visualize the paraboloid $$z = x^2 + y^2$$ as a wire frame with colored edges based on z-value.

3. **Viewing Angles**: Create a 2×2 grid showing the same surface from four different viewing angles: (30°, 0°), (30°, 90°), (60°, 45°), (0°, 0°).

### Level 2: Intermediate

4. **Double Slit Potential**: Visualize a potential with two Gaussian wells centered at (±1, 0) with depth 1 and width 0.5.

5. **3D Probability Slices**: For the hydrogen 2s orbital (n=2, l=0), create XY, XZ, and YZ slice plots through the origin showing the radial node.

6. **Orbital Comparison**: Create a figure comparing the 3s, 3p, and 3d orbitals side by side.

### Level 3: Challenging

7. **Wannier-Stark States**: Visualize a particle in a box with a linear potential (tilted box). Show how the wave function shifts toward one wall.

8. **Molecular Orbital**: Create a σ bonding molecular orbital by combining two 1s hydrogen orbitals centered at different positions.

9. **Time Evolution**: Create a series of 3D plots showing a Gaussian wave packet evolving in a harmonic potential (use 4-6 time snapshots).

---

## Computational Lab

### Project: Complete 3D Orbital Visualization Toolkit

```python
"""
3D Quantum Orbital Visualization Toolkit
Day 269: 3D Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm, assoc_laguerre
from math import factorial


class OrbitalVisualizer:
    """
    Comprehensive toolkit for visualizing atomic orbitals.

    Supports hydrogen-like atoms with arbitrary quantum numbers.
    """

    def __init__(self, a0=1):
        """
        Initialize visualizer.

        Parameters
        ----------
        a0 : float
            Bohr radius in chosen units
        """
        self.a0 = a0

    def radial_wavefunction(self, n, l, r):
        """
        Compute hydrogen radial wave function R_{nl}(r).

        Parameters
        ----------
        n : int
            Principal quantum number (n >= 1)
        l : int
            Orbital quantum number (0 <= l < n)
        r : array
            Radial distances
        """
        rho = 2 * r / (n * self.a0)

        # Normalization factor
        norm = np.sqrt(
            (2 / (n * self.a0))**3 *
            factorial(n - l - 1) / (2 * n * factorial(n + l))
        )

        # Associated Laguerre polynomial
        L = assoc_laguerre(rho, n - l - 1, 2 * l + 1)

        return norm * np.exp(-rho / 2) * rho**l * L

    def angular_wavefunction(self, l, m, theta, phi):
        """
        Compute spherical harmonic Y_l^m(θ, φ).

        Returns complex spherical harmonic.
        """
        return sph_harm(m, l, phi, theta)

    def full_wavefunction(self, n, l, m, r, theta, phi):
        """Compute full hydrogen wave function ψ_{nlm}."""
        R = self.radial_wavefunction(n, l, r)
        Y = self.angular_wavefunction(l, m, theta, phi)
        return R * Y

    def plot_angular_part(self, l, m, ax=None, n_points=50, colormap='RdBu'):
        """
        Plot angular part |Y_l^m|² as a 3D surface.

        Parameters
        ----------
        l, m : int
            Angular quantum numbers
        ax : Axes3D, optional
            Existing 3D axes
        n_points : int
            Grid resolution
        colormap : str
            Colormap for phase visualization
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2 * np.pi, n_points)
        THETA, PHI = np.meshgrid(theta, phi)

        Y = sph_harm(m, l, PHI, THETA)
        R = np.abs(Y)

        # Cartesian coordinates
        X = R * np.sin(THETA) * np.cos(PHI)
        Y_cart = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        # Color by real part (shows sign/phase)
        Y_real = Y.real
        vmax = np.abs(Y_real).max()
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        colors = plt.cm.get_cmap(colormap)(norm(Y_real))

        ax.plot_surface(X, Y_cart, Z, facecolors=colors, alpha=0.9,
                       rstride=1, cstride=1, antialiased=True)

        # Equal aspect ratio
        max_r = 1.2
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_zlim(-max_r, max_r)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        ax.set_title(f'$Y_{l}^{{{m}}}$ ({orbital_names.get(l, str(l))} orbital)')

        return fig, ax

    def plot_probability_cloud(self, n, l, m, ax=None, n_samples=5000, r_max=None):
        """
        Create Monte Carlo probability cloud visualization.

        Parameters
        ----------
        n, l, m : int
            Quantum numbers
        n_samples : int
            Number of points to sample
        r_max : float, optional
            Maximum radius for sampling
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        if r_max is None:
            r_max = 4 * n**2 * self.a0

        # Oversample for rejection sampling
        n_attempt = n_samples * 20

        r = r_max * np.random.rand(n_attempt)
        theta = np.arccos(2 * np.random.rand(n_attempt) - 1)
        phi = 2 * np.pi * np.random.rand(n_attempt)

        psi = self.full_wavefunction(n, l, m, r, theta, phi)
        prob = np.abs(psi)**2 * r**2

        # Rejection sampling
        accept = np.random.rand(n_attempt) < prob / prob.max()

        r_acc = r[accept][:n_samples]
        theta_acc = theta[accept][:n_samples]
        phi_acc = phi[accept][:n_samples]

        # Convert to Cartesian
        x = r_acc * np.sin(theta_acc) * np.cos(phi_acc)
        y = r_acc * np.sin(theta_acc) * np.sin(phi_acc)
        z = r_acc * np.cos(theta_acc)

        # Color by distance from origin
        ax.scatter(x, y, z, c=r_acc, cmap='viridis', s=2, alpha=0.5)

        ax.set_xlabel(f'x (${self.a0}$)')
        ax.set_ylabel(f'y (${self.a0}$)')
        ax.set_zlabel(f'z (${self.a0}$)')

        orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        ax.set_title(f'{n}{orbital_names.get(l, str(l))} Orbital (m={m})')

        return fig, ax

    def plot_radial_probability(self, n, l, ax=None, r_max=None):
        """
        Plot radial probability distribution P(r) = r²|R_{nl}|².
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if r_max is None:
            r_max = 4 * n**2 * self.a0

        r = np.linspace(0, r_max, 500)
        R = self.radial_wavefunction(n, l, r)
        P = r**2 * np.abs(R)**2

        ax.plot(r, P, 'b-', linewidth=2)
        ax.fill_between(r, P, alpha=0.3)

        # Mark most probable radius
        r_max_prob = r[np.argmax(P)]
        ax.axvline(r_max_prob, color='r', linestyle='--',
                  label=f'$r_{{max}} = {r_max_prob:.2f}$')

        ax.set_xlabel(f'r / $a_0$', fontsize=12)
        ax.set_ylabel('$P(r) = r^2|R_{nl}|^2$', fontsize=12)

        orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        ax.set_title(f'{n}{orbital_names.get(l, str(l))} Radial Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

    def orbital_gallery(self, max_n=3):
        """
        Create gallery of all orbitals up to principal quantum number max_n.
        """
        # Count total orbitals
        n_orbitals = sum(n**2 for n in range(1, max_n + 1))

        n_cols = max_n + 1
        n_rows = max_n

        fig = plt.figure(figsize=(4*n_cols, 4*n_rows))

        plot_idx = 1
        for n in range(1, max_n + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
                    self.plot_angular_part(l, m, ax=ax, n_points=30)

                    orbital_names = {0: 's', 1: 'p', 2: 'd'}
                    ax.set_title(f'{n}{orbital_names.get(l, str(l))} m={m}', fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])

                    plot_idx += 1

                    if plot_idx > n_cols * n_rows:
                        break
                if plot_idx > n_cols * n_rows:
                    break
            if plot_idx > n_cols * n_rows:
                break

        plt.tight_layout()
        return fig


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("3D Quantum Orbital Visualization Toolkit")
    print("Day 269: 3D Visualization")
    print("=" * 60)

    viz = OrbitalVisualizer(a0=1)

    # 1. Angular part visualization
    print("\n1. Visualizing spherical harmonics...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                            subplot_kw={'projection': '3d'})

    orbitals = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    for ax, (l, m) in zip(axes.flatten(), orbitals):
        viz.plot_angular_part(l, m, ax=ax)

    plt.suptitle('Spherical Harmonics $Y_l^m$', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('spherical_harmonics_gallery.png', dpi=150, bbox_inches='tight')
    print("   Saved: spherical_harmonics_gallery.png")

    # 2. Probability cloud
    print("\n2. Creating probability clouds...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                            subplot_kw={'projection': '3d'})

    clouds = [(2, 1, 0), (3, 1, 0), (3, 2, 0)]
    for ax, (n, l, m) in zip(axes, clouds):
        viz.plot_probability_cloud(n, l, m, ax=ax, n_samples=3000)

    plt.suptitle('Orbital Probability Clouds', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('probability_clouds.png', dpi=150, bbox_inches='tight')
    print("   Saved: probability_clouds.png")

    # 3. Radial probability distributions
    print("\n3. Plotting radial distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, n in zip(axes, [1, 2, 3]):
        for l in range(n):
            r = np.linspace(0, 20, 500)
            R = viz.radial_wavefunction(n, l, r)
            P = r**2 * np.abs(R)**2

            orbital_names = {0: 's', 1: 'p', 2: 'd'}
            ax.plot(r, P, label=f'{n}{orbital_names[l]}', linewidth=2)

        ax.set_xlabel('r / $a_0$')
        ax.set_ylabel('$P(r)$')
        ax.set_title(f'n = {n}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Radial Probability Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig('radial_distributions.png', dpi=150, bbox_inches='tight')
    print("   Saved: radial_distributions.png")

    # 4. 3D potential surface
    print("\n4. Visualizing 3D potential...")
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    V = 0.5 * (X**2 + Y**2) + 0.1 * (X**4 + Y**4)  # Anharmonic oscillator

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, V, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('V(x,y)')
    ax1.set_title('Anharmonic Potential Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(122)
    cont = ax2.contourf(X, Y, V, levels=30, cmap='viridis')
    ax2.contour(X, Y, V, levels=10, colors='white', linewidths=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour View')
    ax2.set_aspect('equal')
    fig.colorbar(cont, ax=ax2)

    plt.tight_layout()
    plt.savefig('potential_surface.png', dpi=150, bbox_inches='tight')
    print("   Saved: potential_surface.png")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  Figures generated: 4")
    print("  3D visualization techniques covered:")
    print("    - Spherical harmonic surfaces")
    print("    - Monte Carlo probability clouds")
    print("    - Radial probability distributions")
    print("    - Potential energy surfaces")
    print("=" * 60)

    plt.show()
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| `projection='3d'` | Create 3D axes in matplotlib |
| `plot_surface` | Render 3D surface from meshgrid data |
| `plot_wireframe` | Wire mesh representation |
| `contour3D` | 3D contour lines |
| `view_init(elev, azim)` | Set camera viewing angle |
| Spherical harmonics | Angular part of hydrogen orbitals |
| Monte Carlo sampling | Probability cloud visualization |

### Key Formulas

$$\boxed{\psi_{nlm}(r, \theta, \phi) = R_{nl}(r) Y_l^m(\theta, \phi)}$$

$$\boxed{R_{nl}(r) = \sqrt{\left(\frac{2}{na_0}\right)^3 \frac{(n-l-1)!}{2n(n+l)!}} e^{-\rho/2} \rho^l L_{n-l-1}^{2l+1}(\rho)}$$

$$\boxed{Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}} P_l^m(\cos\theta) e^{im\phi}}$$

---

## Daily Checklist

- [ ] Created 3D axes and configured viewing angles
- [ ] Generated surface plots with colormaps
- [ ] Created wire frame visualizations
- [ ] Plotted 3D contours
- [ ] Visualized hydrogen orbitals (angular parts)
- [ ] Created probability cloud representations
- [ ] Completed computational lab exercises

---

## Preview of Day 270

Tomorrow we explore **Interactive Visualization** with Plotly:
- Creating interactive 3D plots
- Plotly's Python interface
- Building widgets for parameter exploration
- Linked views and hover information
- Publishing interactive figures

Interactive visualization enables exploration of quantum systems with real-time parameter adjustment.
