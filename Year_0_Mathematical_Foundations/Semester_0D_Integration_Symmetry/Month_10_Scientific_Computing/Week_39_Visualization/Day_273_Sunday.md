# Day 273: Week 39 Review — Comprehensive Quantum Visualization

## Schedule Overview
**Date**: Week 39, Day 7 (Sunday)
**Duration**: 7 hours
**Theme**: Integration and Mastery of Scientific Visualization

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Review core concepts, integrated visualization package |
| Afternoon | 2.5 hours | Capstone project: Complete quantum state visualizer |
| Evening | 1.5 hours | Self-assessment and Week 40 preparation |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Synthesize all visualization techniques learned this week
2. Build a comprehensive quantum state visualization package
3. Create publication-quality animated figures
4. Choose appropriate visualization methods for different quantum systems
5. Develop a personal visualization style guide

---

## Week 39 Summary

### Topics Covered

| Day | Topic | Key Skills |
|-----|-------|------------|
| **267** | Matplotlib Fundamentals | Figure/Axes hierarchy, basic plots, styling |
| **268** | Scientific Plotting | Error bars, log scales, colormaps, contours |
| **269** | 3D Visualization | Surface plots, orbitals, probability clouds |
| **270** | Interactive (Plotly) | Hover info, animations, dashboards |
| **271** | Animation | FuncAnimation, wave dynamics, Bloch sphere |
| **272** | Publication Quality | LaTeX, journal specs, vector exports |

### Core Visualization Toolkit

```python
# Essential imports for quantum visualization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.special import hermite, sph_harm
```

---

## Comprehensive Review

### 1. Matplotlib Foundation Review

```python
import numpy as np
import matplotlib.pyplot as plt

# Object-oriented interface (preferred)
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

x = np.linspace(-5, 5, 200)
psi = np.exp(-x**2/2) / np.pi**0.25

ax.plot(x, psi, 'b-', linewidth=2, label=r'$\psi(x)$')
ax.fill_between(x, psi, alpha=0.3)

ax.set_xlabel(r'Position $x$', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title('Wave Function', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('review_basic.pdf', bbox_inches='tight')
plt.show()
```

### 2. Advanced Plotting Review

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(-5, 5, 200)

# Panel 1: Error bars
ax1 = axes[0, 0]
x_data = np.linspace(-3, 3, 15)
y_data = np.exp(-x_data**2/2) + np.random.normal(0, 0.05, len(x_data))
y_err = 0.05 * np.ones_like(y_data)
ax1.errorbar(x_data, y_data, yerr=y_err, fmt='o', capsize=3)
ax1.plot(x, np.exp(-x**2/2), 'r-', label='Theory')
ax1.set_title('Error Bars')
ax1.legend()

# Panel 2: Log scale
ax2 = axes[0, 1]
t = np.linspace(0, 5, 100)
decay = np.exp(-t) + 0.01 * np.random.rand(len(t))
ax2.semilogy(t, decay)
ax2.set_title('Semi-log Plot')
ax2.set_xlabel('Time')
ax2.set_ylabel('Probability (log)')

# Panel 3: Colormap
ax3 = axes[1, 0]
X, Y = np.meshgrid(x, x)
Z = np.exp(-(X**2 + Y**2)/2)
im = ax3.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
ax3.set_title('2D Density')
ax3.set_aspect('equal')
plt.colorbar(im, ax=ax3)

# Panel 4: Contour
ax4 = axes[1, 1]
levels = np.linspace(0, 1, 11)
cs = ax4.contourf(X, Y, Z, levels=levels, cmap='plasma')
ax4.contour(X, Y, Z, levels=levels[::2], colors='white', linewidths=0.5)
ax4.set_title('Contour Plot')
ax4.set_aspect('equal')
plt.colorbar(cs, ax=ax4)

plt.tight_layout()
plt.savefig('review_advanced.pdf', bbox_inches='tight')
plt.show()
```

### 3. 3D Visualization Review

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

fig = plt.figure(figsize=(14, 5))

# 3D Surface
ax1 = fig.add_subplot(131, projection='3d')
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)/2)
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('3D Surface')

# Orbital
ax2 = fig.add_subplot(132, projection='3d')
theta = np.linspace(0, np.pi, 40)
phi = np.linspace(0, 2*np.pi, 40)
THETA, PHI = np.meshgrid(theta, phi)
Y_lm = sph_harm(0, 1, PHI, THETA)
R = np.abs(Y_lm)
Xs = R * np.sin(THETA) * np.cos(PHI)
Ys = R * np.sin(THETA) * np.sin(PHI)
Zs = R * np.cos(THETA)
ax2.plot_surface(Xs, Ys, Zs, cmap='RdBu', alpha=0.8)
ax2.set_title('p Orbital')

# Scatter cloud
ax3 = fig.add_subplot(133, projection='3d')
np.random.seed(42)
n = 1000
r = np.random.exponential(1, n)
theta = np.arccos(2*np.random.rand(n) - 1)
phi = 2*np.pi*np.random.rand(n)
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
ax3.scatter(x, y, z, c=r, cmap='plasma', s=1, alpha=0.5)
ax3.set_title('Probability Cloud')

plt.tight_layout()
plt.savefig('review_3d.pdf', bbox_inches='tight')
plt.show()
```

---

## Capstone Project: Complete Quantum Visualization Package

### Project Overview

Build a comprehensive Python package that provides unified visualization capabilities for quantum mechanical systems.

```python
"""
QuantumViz: Complete Quantum Visualization Package
Week 39 Capstone Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import hermite, sph_harm, assoc_laguerre
from scipy.linalg import expm
from math import factorial
import warnings
warnings.filterwarnings('ignore')


class QuantumViz:
    """
    Comprehensive visualization toolkit for quantum mechanical systems.

    Combines all visualization techniques from Week 39:
    - Static plots (1D, 2D, 3D)
    - Interactive features
    - Animations
    - Publication-quality output
    """

    # Default styling
    STYLE = {
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    }

    # Colorblind-safe palette
    COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7',
              '#56B4E9', '#D55E00', '#F0E442']

    def __init__(self, use_latex=False):
        """Initialize visualization toolkit."""
        self.use_latex = use_latex
        self._apply_style()

    def _apply_style(self):
        """Apply default style settings."""
        plt.rcParams.update(self.STYLE)
        if self.use_latex:
            plt.rcParams.update({
                'text.usetex': True,
                'text.latex.preamble': r'\usepackage{amsmath}',
            })

    # ==================== WAVE FUNCTION GENERATION ====================

    def harmonic_oscillator(self, n, x, omega=1, m=1, hbar=1):
        """
        Generate harmonic oscillator eigenstate.

        Parameters
        ----------
        n : int
            Quantum number
        x : array
            Position array
        omega, m, hbar : float
            Physical parameters
        """
        xi = np.sqrt(m * omega / hbar) * x
        prefactor = (m * omega / (np.pi * hbar))**0.25
        norm = 1 / np.sqrt(2**n * factorial(n))
        H_n = hermite(n)
        return prefactor * norm * H_n(xi) * np.exp(-xi**2 / 2)

    def particle_in_box(self, n, x, L=1):
        """Generate particle in a box eigenstate."""
        psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
        psi[(x < 0) | (x > L)] = 0
        return psi

    def hydrogen_radial(self, n, l, r, a0=1):
        """Generate hydrogen radial wave function."""
        rho = 2 * r / (n * a0)
        norm = np.sqrt((2/(n*a0))**3 * factorial(n-l-1) / (2*n*factorial(n+l)))
        L = assoc_laguerre(rho, n-l-1, 2*l+1)
        return norm * np.exp(-rho/2) * rho**l * L

    def gaussian_packet(self, x, x0=0, sigma=1, k0=0):
        """Generate Gaussian wave packet."""
        norm = (2 * np.pi * sigma**2)**(-0.25)
        return norm * np.exp(-(x-x0)**2/(4*sigma**2)) * np.exp(1j*k0*x)

    # ==================== 1D VISUALIZATION ====================

    def plot_wavefunction(self, x, psi, ax=None, show_prob=True,
                         show_phase=False, title=None, **kwargs):
        """
        Plot 1D wave function with various representations.

        Parameters
        ----------
        x : array
            Position array
        psi : array
            Wave function (can be complex)
        ax : Axes, optional
            Matplotlib axes
        show_prob : bool
            Show probability density
        show_phase : bool
            Show phase information for complex wave functions
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        is_complex = np.iscomplexobj(psi)

        if is_complex and not show_phase:
            # Show real and imaginary parts
            ax.plot(x, psi.real, 'b-', linewidth=1.5,
                   label=r'Re[$\psi$]', alpha=0.8)
            ax.plot(x, psi.imag, 'r--', linewidth=1.5,
                   label=r'Im[$\psi$]', alpha=0.8)
        elif is_complex and show_phase:
            # Color by phase
            mag = np.abs(psi)
            phase = np.angle(psi)
            points = ax.scatter(x, mag, c=phase, cmap='hsv',
                              s=2, vmin=-np.pi, vmax=np.pi)
            plt.colorbar(points, ax=ax, label='Phase')
        else:
            ax.plot(x, psi, 'b-', linewidth=2, label=r'$\psi(x)$')
            ax.fill_between(x, psi, alpha=0.3)

        if show_prob:
            prob = np.abs(psi)**2
            ax.plot(x, prob, 'purple', linewidth=2,
                   linestyle='--', label=r'$|\psi|^2$')

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel(r'Position $x$')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title)

        return fig, ax

    def plot_in_potential(self, x, psi, V, E, ax=None, scale=0.3, **kwargs):
        """Plot wave function overlaid on potential."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Plot potential
        ax.fill_between(x, V, alpha=0.2, color='gray')
        ax.plot(x, V, 'k-', linewidth=1, label=r'$V(x)$')

        # Energy level
        ax.axhline(E, color='gray', linestyle='--', alpha=0.7)

        # Scaled probability density at energy level
        prob = np.abs(psi)**2
        scale_factor = scale * (V.max() - V.min()) / prob.max()
        ax.fill_between(x, E, E + scale_factor * prob,
                       alpha=0.5, color='blue', label=r'$|\psi|^2$')

        ax.set_xlabel(r'Position $x$')
        ax.set_ylabel('Energy')
        ax.legend()

        return fig, ax

    def plot_energy_spectrum(self, energies, labels=None, ax=None, **kwargs):
        """Create energy level diagram."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 8))
        else:
            fig = ax.figure

        n = len(energies)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

        for i, (E, color) in enumerate(zip(energies, colors)):
            ax.hlines(E, 0.2, 0.8, color=color, linewidth=3)
            if labels:
                ax.text(0.85, E, labels[i], va='center', fontsize=10)

        ax.set_xlim(0, 1.2)
        ax.set_ylabel('Energy')
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        return fig, ax

    # ==================== 2D VISUALIZATION ====================

    def plot_2d_density(self, X, Y, psi, ax=None, cmap='viridis',
                       show_contours=True, **kwargs):
        """Plot 2D probability density."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        prob = np.abs(psi)**2

        im = ax.pcolormesh(X, Y, prob, cmap=cmap, shading='auto')
        plt.colorbar(im, ax=ax, label=r'$|\psi|^2$')

        if show_contours:
            levels = np.linspace(0, prob.max(), 10)
            ax.contour(X, Y, prob, levels=levels, colors='white',
                      linewidths=0.5, alpha=0.7)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        return fig, ax

    def plot_density_matrix(self, rho, basis_labels=None, ax=None, **kwargs):
        """Visualize density matrix."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        n = rho.shape[0]
        im = ax.imshow(np.abs(rho), cmap='Blues', aspect='equal')

        # Annotations
        for i in range(n):
            for j in range(n):
                val = rho[i, j]
                if np.abs(val) > 0.01:
                    color = 'white' if np.abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{np.abs(val):.2f}',
                           ha='center', va='center', color=color, fontsize=9)

        if basis_labels:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(basis_labels)
            ax.set_yticklabels(basis_labels)

        plt.colorbar(im, ax=ax, label=r'$|\rho_{ij}|$')

        return fig, ax

    # ==================== 3D VISUALIZATION ====================

    def plot_3d_surface(self, X, Y, Z, ax=None, cmap='viridis', **kwargs):
        """Create 3D surface plot."""
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8,
                              rstride=2, cstride=2)
        fig.colorbar(surf, ax=ax, shrink=0.5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        return fig, ax

    def plot_orbital(self, l, m, ax=None, n_points=50, **kwargs):
        """Visualize spherical harmonic orbital."""
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2*np.pi, n_points)
        THETA, PHI = np.meshgrid(theta, phi)

        Y_lm = sph_harm(m, l, PHI, THETA)
        R = np.abs(Y_lm)

        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        # Color by real part
        Y_real = Y_lm.real
        norm = Normalize(vmin=-np.abs(Y_real).max(), vmax=np.abs(Y_real).max())
        colors = plt.cm.RdBu(norm(Y_real))

        ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.9,
                       rstride=1, cstride=1)

        max_r = 1.2
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_zlim(-max_r, max_r)

        orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        ax.set_title(f'{orbital_names.get(l, str(l))} orbital (l={l}, m={m})')

        return fig, ax

    # ==================== ANIMATION ====================

    def animate_wavepacket(self, x, psi0, H, dt=0.02, n_frames=200,
                          interval=30, **kwargs):
        """
        Animate wave packet time evolution.

        Parameters
        ----------
        x : array
            Position grid
        psi0 : array
            Initial wave function
        H : array
            Hamiltonian matrix
        dt : float
            Time step
        n_frames : int
            Number of frames
        interval : int
            Milliseconds between frames
        """
        U = expm(-1j * H * dt)
        psi = psi0.copy()

        fig, ax = plt.subplots(figsize=(10, 6))
        line_prob, = ax.plot([], [], 'b-', linewidth=2)
        line_real, = ax.plot([], [], 'gray', linewidth=1, alpha=0.5)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 1.2 * np.abs(psi0).max()**2)
        ax.set_xlabel('Position x')
        ax.set_ylabel(r'$|\psi|^2$')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            line_prob.set_data([], [])
            line_real.set_data([], [])
            time_text.set_text('')
            return line_prob, line_real, time_text

        def update(frame):
            nonlocal psi
            t = frame * dt

            prob = np.abs(psi)**2
            line_prob.set_data(x, prob)
            line_real.set_data(x, psi.real * 0.5 + 0.5)

            time_text.set_text(f't = {t:.2f}')
            psi = U @ psi

            return line_prob, line_real, time_text

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                           blit=True, interval=interval)

        return fig, ani

    def animate_bloch(self, trajectory_func, n_frames=200, interval=30, **kwargs):
        """
        Animate state evolution on Bloch sphere.

        Parameters
        ----------
        trajectory_func : callable
            Function that takes time t and returns (x, y, z) Bloch vector
        n_frames : int
            Number of frames
        interval : int
            Milliseconds between frames
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Draw sphere wireframe
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.3)

        # Axes
        ax.plot([-1.3, 1.3], [0, 0], [0, 0], 'r-', linewidth=1)
        ax.plot([0, 0], [-1.3, 1.3], [0, 0], 'g-', linewidth=1)
        ax.plot([0, 0], [0, 0], [-1.3, 1.3], 'b-', linewidth=1)

        state_line, = ax.plot([], [], [], 'purple', linewidth=3)
        state_point, = ax.plot([], [], [], 'o', color='purple', markersize=10)
        trail, = ax.plot([], [], [], '-', color='purple', alpha=0.3)

        trajectory = {'x': [], 'y': [], 'z': []}

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

        def init():
            state_line.set_data([], [])
            state_line.set_3d_properties([])
            state_point.set_data([], [])
            state_point.set_3d_properties([])
            trail.set_data([], [])
            trail.set_3d_properties([])
            return state_line, state_point, trail

        def update(frame):
            t = frame * 0.02
            x, y, z = trajectory_func(t)

            trajectory['x'].append(x)
            trajectory['y'].append(y)
            trajectory['z'].append(z)

            for key in trajectory:
                if len(trajectory[key]) > 100:
                    trajectory[key].pop(0)

            state_line.set_data([0, x], [0, y])
            state_line.set_3d_properties([0, z])
            state_point.set_data([x], [y])
            state_point.set_3d_properties([z])
            trail.set_data(trajectory['x'], trajectory['y'])
            trail.set_3d_properties(trajectory['z'])

            return state_line, state_point, trail

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                           blit=False, interval=interval)

        return fig, ani

    # ==================== PUBLICATION FIGURES ====================

    def create_multipanel(self, n_rows, n_cols, figsize=None, **kwargs):
        """Create labeled multi-panel figure."""
        if figsize is None:
            figsize = (3.4 * n_cols, 2.5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, **kwargs)
        axes_flat = np.array(axes).flatten()

        import string
        for i, ax in enumerate(axes_flat):
            if i < 26:
                ax.text(-0.12, 1.05, f'({string.ascii_lowercase[i]})',
                       transform=ax.transAxes, fontsize=11, fontweight='bold')

        return fig, axes

    def save_figure(self, fig, filename, formats=['pdf', 'png'], **kwargs):
        """Save figure in multiple formats."""
        save_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.02, 'dpi': 300}
        save_kwargs.update(kwargs)

        for fmt in formats:
            fig.savefig(f'{filename}.{fmt}', format=fmt, **save_kwargs)
            print(f'Saved: {filename}.{fmt}')


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QuantumViz: Complete Quantum Visualization Package")
    print("Week 39 Capstone Project")
    print("=" * 60)

    viz = QuantumViz(use_latex=False)

    # Demo 1: Harmonic oscillator states
    print("\n1. Harmonic oscillator visualization...")
    x = np.linspace(-6, 6, 500)
    V = 0.5 * x**2

    fig, axes = viz.create_multipanel(2, 2, figsize=(12, 10))

    # First few eigenstates
    for n in range(4):
        psi = viz.harmonic_oscillator(n, x)
        E_n = n + 0.5
        viz.plot_in_potential(x, psi, V, E_n, ax=axes[0, 0], scale=0.35)

    axes[0, 0].set_title('Eigenstates in Potential')
    axes[0, 0].set_ylim(-0.5, 5)

    # Energy spectrum
    energies = [n + 0.5 for n in range(8)]
    labels = [f'n={n}' for n in range(8)]
    viz.plot_energy_spectrum(energies, labels, ax=axes[0, 1])
    axes[0, 1].set_title('Energy Levels')

    # Ground state detail
    psi_0 = viz.harmonic_oscillator(0, x)
    viz.plot_wavefunction(x, psi_0, ax=axes[1, 0], title='Ground State')

    # Wave packet
    psi_wp = viz.gaussian_packet(x, x0=-2, sigma=0.5, k0=3)
    viz.plot_wavefunction(x, psi_wp, ax=axes[1, 1], title='Wave Packet')

    plt.tight_layout()
    viz.save_figure(fig, 'capstone_harmonic')

    # Demo 2: 3D orbital
    print("\n2. Orbital visualization...")
    fig = plt.figure(figsize=(15, 5))

    orbitals = [(0, 0), (1, 0), (2, 0)]
    for i, (l, m) in enumerate(orbitals):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        viz.plot_orbital(l, m, ax=ax)

    plt.tight_layout()
    viz.save_figure(fig, 'capstone_orbitals')

    # Demo 3: 2D density
    print("\n3. 2D density visualization...")
    x_2d = np.linspace(-4, 4, 100)
    y_2d = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_2d, y_2d)
    psi_2d = X * np.exp(-(X**2 + Y**2)/2)  # 2D excited state

    fig, ax = plt.subplots(figsize=(8, 6))
    viz.plot_2d_density(X, Y, psi_2d, ax=ax)
    ax.set_title('2D Harmonic Oscillator (1,0) State')
    viz.save_figure(fig, 'capstone_2d')

    # Demo 4: Density matrix
    print("\n4. Density matrix visualization...")
    psi = np.array([1, 1, 0]) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())

    fig, ax = plt.subplots(figsize=(6, 5))
    viz.plot_density_matrix(rho, basis_labels=['|0⟩', '|1⟩', '|2⟩'], ax=ax)
    ax.set_title('Superposition State Density Matrix')
    viz.save_figure(fig, 'capstone_density_matrix')

    print("\n" + "=" * 60)
    print("Capstone project complete!")
    print("All figures saved to current directory.")
    print("=" * 60)

    plt.show()
```

---

## Week 39 Assessment

### Concept Checklist

- [ ] **Matplotlib Basics**: Figure/Axes hierarchy, plot types, styling
- [ ] **Scientific Plotting**: Error bars, log scales, colormaps
- [ ] **3D Visualization**: Surfaces, orbitals, probability clouds
- [ ] **Interactive Plots**: Plotly, hover info, animations
- [ ] **Animation**: FuncAnimation, wave dynamics, Bloch sphere
- [ ] **Publication Quality**: LaTeX, journal specs, vector export

### Skills Demonstration

Create a figure that demonstrates:
1. A multi-panel layout with proper labels
2. At least one 3D visualization
3. Proper use of colormaps
4. LaTeX-formatted labels
5. Publication-quality styling

### Self-Assessment Questions

1. When should you use `pcolormesh` vs `contourf`?
2. What is the difference between `FuncAnimation` and `ArtistAnimation`?
3. How do you ensure figures are accessible to colorblind readers?
4. What file formats are best for journal submission?
5. How do you add inset plots to a figure?

---

## Key Formulas Reference

### Quantum Wave Functions

$$\boxed{\psi_n^{HO}(x) = \frac{1}{\sqrt{2^n n!}} \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} H_n(\xi) e^{-\xi^2/2}}$$

$$\boxed{\psi_n^{box}(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right)}$$

$$\boxed{Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}} P_l^m(\cos\theta) e^{im\phi}}$$

### Visualization Formulas

$$\boxed{\text{Probability density: } P(x) = |\psi(x)|^2}$$

$$\boxed{\text{Normalization: } \int_{-\infty}^{\infty} |\psi(x)|^2 dx = 1}$$

---

## Navigation

- **Previous**: [Day 272: Publication-Quality Figures](Day_272_Saturday.md)
- **Next**: [Week 40: Physics Simulations](../Week_40_Physics_Simulations/README.md)
- **Week Overview**: [Week 39 README](README.md)

---

## Preview of Week 40

Next week focuses on **Physics Simulations**:
- Numerical solution of the Schrödinger equation
- Molecular dynamics basics
- Monte Carlo methods for quantum systems
- Integration of visualization with simulation
- Complete quantum simulation projects

The visualization skills from this week will be essential for presenting simulation results.
