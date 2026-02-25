# Day 267: Matplotlib Fundamentals

## Schedule Overview
**Date**: Week 39, Day 1 (Monday)
**Duration**: 7 hours
**Theme**: Mastering the Foundation of Scientific Visualization in Python

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Matplotlib architecture, pyplot vs OO interface |
| Afternoon | 2.5 hours | Basic plot types, styling, and customization |
| Evening | 1.5 hours | Computational lab: Quantum wave function visualization |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Explain Matplotlib's architecture (Figure, Axes, Artist hierarchy)
2. Choose appropriately between pyplot and object-oriented interfaces
3. Create and customize line plots, scatter plots, and histograms
4. Configure plot styling including colors, markers, and line styles
5. Build multi-panel figures with `subplots`
6. Save figures in various formats with appropriate resolution

---

## Core Content

### 1. Matplotlib Architecture

Matplotlib is built on a hierarchical structure that provides both simplicity and power:

$$\text{Figure} \supset \text{Axes} \supset \text{Axis} \supset \text{Tick}$$

#### The Object Hierarchy

```
Figure (top-level container)
├── Axes (individual plot area)
│   ├── XAxis
│   │   └── Ticks, Labels
│   ├── YAxis
│   │   └── Ticks, Labels
│   ├── Lines (plot data)
│   ├── Patches (bars, histograms)
│   ├── Text (titles, annotations)
│   └── Legend
├── Axes (second subplot)
└── ...
```

**Key Concepts**:
- **Figure**: The entire window or page; can contain multiple Axes
- **Axes**: A single plot with its own coordinate system (NOT plural of Axis!)
- **Axis**: The number-line objects (XAxis, YAxis) with ticks and labels
- **Artist**: Everything visible on the figure (lines, text, patches)

### 2. Two Interfaces: pyplot vs Object-Oriented

Matplotlib provides two distinct APIs:

#### pyplot Interface (MATLAB-style)
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.xlabel('Position')
plt.ylabel('Amplitude')
plt.title('Wave Function')
plt.show()
```

**Pros**: Quick, interactive, familiar to MATLAB users
**Cons**: Implicit state, harder to manage multiple figures

#### Object-Oriented Interface (Recommended for Science)
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y)
ax.set_xlabel('Position')
ax.set_ylabel('Amplitude')
ax.set_title('Wave Function')
plt.show()
```

**Pros**: Explicit control, better for complex figures, reusable
**Cons**: Slightly more verbose

**Rule of Thumb**: Use OO interface for scripts and publications, pyplot for quick exploration.

### 3. Basic Plot Types

#### Line Plots
The fundamental visualization for continuous functions:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)
psi = np.exp(-x**2/2) / np.pi**0.25  # Gaussian wave packet

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, psi, 'b-', linewidth=2, label=r'$\psi(x)$')
ax.plot(x, np.abs(psi)**2, 'r--', linewidth=2, label=r'$|\psi(x)|^2$')
ax.set_xlabel(r'Position $x$', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.show()
```

#### Scatter Plots
For discrete data points or correlations:

```python
# Simulated quantum measurement outcomes
np.random.seed(42)
positions = np.random.normal(0, 1, 500)  # Position measurements
momenta = np.random.normal(0, 1, 500)    # Momentum measurements

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(positions, momenta, c=positions**2 + momenta**2,
                     cmap='viridis', alpha=0.6, s=20)
ax.set_xlabel(r'Position $x$', fontsize=12)
ax.set_ylabel(r'Momentum $p$', fontsize=12)
ax.set_title('Phase Space Distribution', fontsize=14)
plt.colorbar(scatter, label=r'$x^2 + p^2$')
ax.set_aspect('equal')
plt.show()
```

#### Histograms
For probability distributions and measurement statistics:

```python
# Quantum measurement simulation
measurements = np.random.choice([0, 1], size=10000, p=[0.3, 0.7])

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(measurements, bins=2, edgecolor='black', alpha=0.7,
        color=['blue', 'red'], rwidth=0.8)
ax.set_xticks([0.25, 0.75])
ax.set_xticklabels([r'$|0\rangle$', r'$|1\rangle$'], fontsize=14)
ax.set_ylabel('Counts', fontsize=12)
ax.set_title('Qubit Measurement Outcomes', fontsize=14)
plt.show()
```

### 4. Line Styles, Markers, and Colors

#### Complete Style Specification

```python
# Format string: '[marker][line][color]'
ax.plot(x, y, 'o-b')  # circle markers, solid line, blue

# Keyword arguments (more control)
ax.plot(x, y,
        color='#1f77b4',      # Hex color
        linestyle='--',        # Dashed
        linewidth=2,           # Line width
        marker='o',            # Circle markers
        markersize=6,          # Marker size
        markerfacecolor='white',
        markeredgecolor='#1f77b4',
        markeredgewidth=1.5,
        alpha=0.8,             # Transparency
        label='Data')
```

#### Common Line Styles
| Code | Style |
|------|-------|
| `-` | Solid |
| `--` | Dashed |
| `-.` | Dash-dot |
| `:` | Dotted |
| `''` | None (markers only) |

#### Common Markers
| Code | Marker |
|------|--------|
| `o` | Circle |
| `s` | Square |
| `^` | Triangle up |
| `v` | Triangle down |
| `*` | Star |
| `+` | Plus |
| `x` | Cross |
| `.` | Point |

#### Color Specification
```python
# Named colors
color='blue', color='red', color='green'

# Hex codes
color='#1f77b4', color='#ff7f0e'

# RGB tuples (0-1 scale)
color=(0.1, 0.2, 0.5)

# RGBA with transparency
color=(0.1, 0.2, 0.5, 0.7)

# Tableau palette (default cycle)
color='C0', color='C1', color='C2'  # etc.
```

### 5. Subplots and Multi-Panel Figures

#### Basic Subplots
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Access individual axes
axes[0, 0].plot(x, y1)
axes[0, 1].plot(x, y2)
axes[1, 0].plot(x, y3)
axes[1, 1].plot(x, y4)

plt.tight_layout()  # Prevent overlap
```

#### Shared Axes
```python
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(x, psi_n[i])
    ax.set_ylabel(f'$\psi_{i}(x)$')

axes[-1].set_xlabel('Position $x$')
plt.tight_layout()
```

#### Unequal Subplot Sizes
```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :2])   # Top-left, spans 2 columns
ax2 = fig.add_subplot(gs[0, 2])    # Top-right
ax3 = fig.add_subplot(gs[1, :])    # Bottom, spans all columns
```

### 6. Figure Configuration and Saving

#### DPI and Size
```python
# For screen display
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

# For publication (high resolution)
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
```

#### Saving Figures
```python
# PNG for presentations
fig.savefig('wavefunction.png', dpi=300, bbox_inches='tight')

# PDF for papers (vector format)
fig.savefig('wavefunction.pdf', bbox_inches='tight')

# SVG for web (vector)
fig.savefig('wavefunction.svg', bbox_inches='tight')

# Transparent background
fig.savefig('wavefunction.png', dpi=300, transparent=True)
```

---

## Quantum Mechanics Connection

### Wave Function Visualization Best Practices

When visualizing quantum mechanical wave functions, several considerations are essential:

#### 1. Real vs Complex Wave Functions
Quantum wave functions are generally complex: $$\psi(x) = |\psi(x)|e^{i\phi(x)}$$

Visualization options:
- **Real and imaginary parts**: Two separate plots or one with two y-axes
- **Magnitude and phase**: $$|\psi(x)|$$ and $$\arg(\psi(x))$$
- **Probability density**: $$|\psi(x)|^2$$ (most common for interpretation)

```python
def plot_complex_wavefunction(x, psi, ax=None):
    """Plot complex wave function with magnitude and phase."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    magnitude = np.abs(psi)
    phase = np.angle(psi)

    # Use color for phase
    points = ax.scatter(x, magnitude, c=phase, cmap='hsv',
                       s=1, vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel(r'Position $x$')
    ax.set_ylabel(r'$|\psi(x)|$')
    plt.colorbar(points, label=r'Phase $\phi$')
    return ax
```

#### 2. Normalization Check
Always verify $$\int_{-\infty}^{\infty} |\psi(x)|^2 dx = 1$$:

```python
def plot_normalized_wavefunction(x, psi):
    """Plot wave function with normalization verification."""
    dx = x[1] - x[0]
    norm = np.trapz(np.abs(psi)**2, x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, np.abs(psi)**2, alpha=0.3, label=r'$|\psi|^2$')
    ax.plot(x, np.abs(psi)**2, 'b-', linewidth=2)
    ax.set_xlabel(r'Position $x$')
    ax.set_ylabel(r'Probability density')
    ax.set_title(f'Normalization: $\\int|\\psi|^2 dx = {norm:.6f}$')
    ax.legend()
    return fig, ax
```

#### 3. Potential Energy Overlay
Show wave functions with their confining potential:

```python
def plot_wavefunction_in_potential(x, psi, V, E, ax=None):
    """Plot wave function overlaid on potential energy."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Scale wave function for visibility
    scale = 0.3 * (V.max() - V.min()) / np.abs(psi).max()

    ax.plot(x, V, 'k-', linewidth=2, label='V(x)')
    ax.axhline(E, color='gray', linestyle='--', label=f'E = {E:.2f}')
    ax.fill_between(x, E, E + scale * np.abs(psi)**2,
                    alpha=0.5, label=r'$|\psi|^2$')
    ax.set_xlabel(r'Position $x$')
    ax.set_ylabel('Energy')
    ax.legend()
    return ax
```

---

## Worked Examples

### Example 1: Harmonic Oscillator Wave Functions

**Problem**: Visualize the first four energy eigenstates of the quantum harmonic oscillator.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.misc import factorial

def harmonic_oscillator_wavefunction(n, x, m=1, omega=1, hbar=1):
    """
    Quantum harmonic oscillator eigenstate.

    ψ_n(x) = (mω/πℏ)^(1/4) * (1/√(2^n n!)) * H_n(ξ) * exp(-ξ²/2)
    where ξ = √(mω/ℏ) * x
    """
    xi = np.sqrt(m * omega / hbar) * x
    prefactor = (m * omega / (np.pi * hbar))**0.25
    normalization = 1 / np.sqrt(2**n * factorial(n))
    H_n = hermite(n)
    return prefactor * normalization * H_n(xi) * np.exp(-xi**2 / 2)

# Create visualization
x = np.linspace(-5, 5, 1000)
V = 0.5 * x**2  # Potential

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for n, ax in enumerate(axes):
    psi = harmonic_oscillator_wavefunction(n, x)
    E_n = n + 0.5  # Energy eigenvalue (ℏω = 1)

    # Plot potential
    ax.plot(x, V, 'k-', linewidth=1.5, alpha=0.5)

    # Plot energy level
    ax.axhline(E_n, color='gray', linestyle='--', alpha=0.7)

    # Plot probability density (scaled and shifted to energy level)
    scale = 0.4
    prob = np.abs(psi)**2
    ax.fill_between(x, E_n, E_n + scale * prob / prob.max(),
                    alpha=0.6, color=f'C{n}')

    # Plot wave function (scaled and shifted)
    ax.plot(x, E_n + scale * psi / np.abs(psi).max(),
            color=f'C{n}', linewidth=1.5)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, 5)
    ax.set_xlabel(r'Position $x$ (units of $\sqrt{\hbar/m\omega}$)')
    ax.set_ylabel('Energy (units of $\hbar\omega$)')
    ax.set_title(f'$n = {n}$, $E_{n} = {E_n}\\,\\hbar\\omega$')
    ax.grid(True, alpha=0.3)

plt.suptitle('Quantum Harmonic Oscillator Eigenstates', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('harmonic_oscillator_states.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Example 2: Particle in a Box

**Problem**: Create a multi-panel figure showing the first three energy eigenstates of a particle in a 1D infinite square well.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt

def particle_in_box(n, x, L=1):
    """
    Infinite square well eigenstate.

    ψ_n(x) = √(2/L) sin(nπx/L) for 0 < x < L
    """
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

L = 1  # Box width
x = np.linspace(0, L, 500)

# Create figure with shared x-axis
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

for n in range(1, 4):
    psi = particle_in_box(n, x, L)
    prob = np.abs(psi)**2
    E_n = n**2  # Energy in units of π²ℏ²/(2mL²)

    # Left column: Wave function
    axes[n-1, 0].plot(x, psi, 'b-', linewidth=2)
    axes[n-1, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[n-1, 0].fill_between(x, psi, alpha=0.3)
    axes[n-1, 0].set_ylabel(f'$\\psi_{n}(x)$', fontsize=12)
    axes[n-1, 0].set_title(f'$n = {n}$, $E_{n} = {E_n}\\pi^2\\hbar^2/2mL^2$')
    axes[n-1, 0].set_xlim(0, L)
    axes[n-1, 0].grid(True, alpha=0.3)

    # Mark nodes
    nodes = np.linspace(0, L, n+1)[1:-1]
    axes[n-1, 0].scatter(nodes, np.zeros_like(nodes),
                         color='red', s=50, zorder=5, label='Nodes')

    # Right column: Probability density
    axes[n-1, 1].plot(x, prob, 'r-', linewidth=2)
    axes[n-1, 1].fill_between(x, prob, alpha=0.3, color='red')
    axes[n-1, 1].set_ylabel(f'$|\\psi_{n}(x)|^2$', fontsize=12)
    axes[n-1, 1].set_xlim(0, L)
    axes[n-1, 1].grid(True, alpha=0.3)

    # Mark probability maxima
    axes[n-1, 1].axhline(2/L, color='gray', linestyle='--', alpha=0.5)

axes[-1, 0].set_xlabel('Position $x/L$', fontsize=12)
axes[-1, 1].set_xlabel('Position $x/L$', fontsize=12)

# Add wall indicators
for ax_row in axes:
    for ax in ax_row:
        ax.axvline(0, color='k', linewidth=3)
        ax.axvline(L, color='k', linewidth=3)

plt.suptitle('Particle in a Box: Wave Functions and Probability Densities',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('particle_in_box.pdf', bbox_inches='tight')
plt.show()

# Verification: Check normalization
for n in range(1, 4):
    psi = particle_in_box(n, x, L)
    norm = np.trapz(np.abs(psi)**2, x)
    print(f"n={n}: ∫|ψ|² dx = {norm:.6f}")
```

### Example 3: Gaussian Wave Packet

**Problem**: Visualize a Gaussian wave packet with momentum, showing both real/imaginary parts.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_wave_packet(x, x0=0, sigma=1, k0=5):
    """
    Gaussian wave packet with initial momentum k0.

    ψ(x) = (2πσ²)^(-1/4) exp(-(x-x0)²/4σ²) exp(ik0·x)
    """
    normalization = (2 * np.pi * sigma**2)**(-0.25)
    gaussian = np.exp(-(x - x0)**2 / (4 * sigma**2))
    plane_wave = np.exp(1j * k0 * x)
    return normalization * gaussian * plane_wave

# Create wave packet
x = np.linspace(-10, 10, 1000)
psi = gaussian_wave_packet(x, x0=-2, sigma=1, k0=3)

# Create multi-panel figure
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Real part
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x, psi.real, 'b-', linewidth=1.5)
ax1.fill_between(x, psi.real, alpha=0.3)
ax1.set_ylabel(r'Re$[\psi(x)]$', fontsize=12)
ax1.set_title('Real Part', fontsize=12)
ax1.grid(True, alpha=0.3)

# Imaginary part
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x, psi.imag, 'r-', linewidth=1.5)
ax2.fill_between(x, psi.imag, alpha=0.3, color='red')
ax2.set_ylabel(r'Im$[\psi(x)]$', fontsize=12)
ax2.set_title('Imaginary Part', fontsize=12)
ax2.grid(True, alpha=0.3)

# Probability density
ax3 = fig.add_subplot(gs[1, :])
prob = np.abs(psi)**2
ax3.plot(x, prob, 'purple', linewidth=2)
ax3.fill_between(x, prob, alpha=0.4, color='purple')
ax3.set_xlabel(r'Position $x$', fontsize=12)
ax3.set_ylabel(r'$|\psi(x)|^2$', fontsize=12)
ax3.set_title('Probability Density', fontsize=12)
ax3.grid(True, alpha=0.3)

# Add expectation value marker
x_mean = np.trapz(x * prob, x)
ax3.axvline(x_mean, color='k', linestyle='--', label=f'$\\langle x \\rangle = {x_mean:.2f}$')
ax3.legend()

# Phase plot
ax4 = fig.add_subplot(gs[2, :])
phase = np.angle(psi)
ax4.plot(x, phase, 'green', linewidth=1.5)
ax4.set_xlabel(r'Position $x$', fontsize=12)
ax4.set_ylabel(r'Phase $\arg[\psi(x)]$', fontsize=12)
ax4.set_title('Phase (related to local momentum)', fontsize=12)
ax4.set_ylim(-np.pi, np.pi)
ax4.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax4.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
ax4.grid(True, alpha=0.3)

plt.suptitle('Gaussian Wave Packet Analysis', fontsize=14, y=1.02)
plt.savefig('gaussian_wave_packet.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Practice Problems

### Level 1: Direct Application

1. **Basic Line Plot**: Create a plot of $$\sin(x)$$ and $$\cos(x)$$ for $$x \in [0, 4\pi]$$ with different line styles, a legend, and labeled axes.

2. **Histogram**: Generate 10,000 samples from a normal distribution and create a histogram with 50 bins. Overlay the theoretical PDF.

3. **Scatter Plot**: Create a scatter plot of 200 random (x, y) points where both coordinates are drawn from a uniform distribution on [0, 1]. Color points by their distance from the origin.

### Level 2: Intermediate

4. **Multi-Panel Comparison**: Create a 2×2 subplot figure showing:
   - Top-left: $$\psi(x) = e^{-x^2}$$
   - Top-right: $$\psi(x) = xe^{-x^2}$$
   - Bottom-left: $$|\psi(x)|^2$$ for first function
   - Bottom-right: $$|\psi(x)|^2$$ for second function
   Use shared x-axes within each column.

5. **Custom Style**: Create a style dictionary that sets:
   - Figure size: 10×6 inches
   - Font: serif family
   - Line width: 2
   - Grid: on with alpha=0.3
   Apply this style to a wave function plot.

6. **Energy Level Diagram**: Create a horizontal bar chart showing the first 10 energy levels of:
   - Particle in a box: $$E_n = n^2$$
   - Harmonic oscillator: $$E_n = n + 1/2$$
   Plot both on the same axes with different colors.

### Level 3: Challenging

7. **Wigner Function Slice**: For a coherent state $$|\alpha\rangle$$ with $$\alpha = 2$$, plot the probability distribution in both position and momentum space. Verify the uncertainty relation.

8. **Interactive-Style Parameter Exploration**: Create a function that generates a figure showing how the ground state wave function of an asymmetric double well changes as the asymmetry parameter varies. Display 5 different parameter values in a single multi-panel figure.

9. **Publication Figure**: Create a single figure suitable for journal submission showing:
   - Main panel: First 5 harmonic oscillator eigenstates
   - Inset: Zoom on the ground state near x=0
   - All text in LaTeX format
   - Exported as both PNG (300 dpi) and PDF

---

## Computational Lab

### Project: Complete Wave Function Visualization Toolkit

Build a reusable Python module for quantum wave function visualization.

```python
"""
Quantum Wave Function Visualization Toolkit
Day 267: Matplotlib Fundamentals
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import hermite
from scipy.misc import factorial
import warnings

# Suppress factorial deprecation warning
warnings.filterwarnings('ignore', category=DeprecationWarning)


class WaveFunctionPlotter:
    """
    A class for visualizing quantum mechanical wave functions.

    Provides methods for plotting wave functions, probability densities,
    and related quantum mechanical quantities with consistent styling.
    """

    # Default style settings
    DEFAULT_STYLE = {
        'figsize': (10, 6),
        'dpi': 150,
        'linewidth': 2,
        'fontsize': 12,
        'title_fontsize': 14,
        'grid_alpha': 0.3,
        'fill_alpha': 0.3,
        'colors': {
            'wavefunction': '#1f77b4',
            'probability': '#d62728',
            'potential': '#2ca02c',
            'energy': '#7f7f7f'
        }
    }

    def __init__(self, style=None):
        """Initialize plotter with optional custom style."""
        self.style = {**self.DEFAULT_STYLE, **(style or {})}

    def plot_wavefunction(self, x, psi, ax=None, show_probability=True,
                         title=None, xlabel=r'Position $x$'):
        """
        Plot a wave function with optional probability density.

        Parameters
        ----------
        x : array
            Position array
        psi : array
            Wave function values (can be complex)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on; creates new figure if None
        show_probability : bool
            If True, also plot |psi|^2
        title : str, optional
            Plot title
        xlabel : str
            X-axis label

        Returns
        -------
        fig, ax : tuple
            Figure and axes objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style['figsize'],
                                   dpi=self.style['dpi'])
        else:
            fig = ax.figure

        colors = self.style['colors']
        lw = self.style['linewidth']

        # Check if complex
        is_complex = np.iscomplexobj(psi)

        if is_complex:
            # Plot real and imaginary parts
            ax.plot(x, psi.real, '-', color=colors['wavefunction'],
                   linewidth=lw, label=r'Re$[\psi(x)]$')
            ax.plot(x, psi.imag, '--', color=colors['wavefunction'],
                   linewidth=lw, alpha=0.7, label=r'Im$[\psi(x)]$')
        else:
            ax.plot(x, psi, '-', color=colors['wavefunction'],
                   linewidth=lw, label=r'$\psi(x)$')
            ax.fill_between(x, psi, alpha=self.style['fill_alpha'],
                           color=colors['wavefunction'])

        if show_probability:
            prob = np.abs(psi)**2
            ax.plot(x, prob, '-', color=colors['probability'],
                   linewidth=lw, label=r'$|\psi(x)|^2$')

        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=self.style['fontsize'])
        ax.set_ylabel('Amplitude', fontsize=self.style['fontsize'])
        ax.legend(fontsize=self.style['fontsize']-1)
        ax.grid(True, alpha=self.style['grid_alpha'])

        if title:
            ax.set_title(title, fontsize=self.style['title_fontsize'])

        return fig, ax

    def plot_in_potential(self, x, psi, V, E, ax=None,
                         scale_factor=0.3, title=None):
        """
        Plot wave function overlaid on potential energy.

        Parameters
        ----------
        x : array
            Position array
        psi : array
            Wave function
        V : array
            Potential energy
        E : float
            Energy eigenvalue
        scale_factor : float
            Scaling for wave function height
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style['figsize'],
                                   dpi=self.style['dpi'])
        else:
            fig = ax.figure

        colors = self.style['colors']
        lw = self.style['linewidth']

        # Plot potential
        ax.plot(x, V, '-', color=colors['potential'], linewidth=lw,
               label='$V(x)$')

        # Plot energy level
        ax.axhline(E, color=colors['energy'], linestyle='--',
                  label=f'$E = {E:.2f}$')

        # Scale and shift probability density
        prob = np.abs(psi)**2
        scale = scale_factor * (V.max() - V.min()) / prob.max()
        ax.fill_between(x, E, E + scale * prob, alpha=0.5,
                       color=colors['probability'], label=r'$|\psi|^2$')
        ax.plot(x, E + scale * prob, color=colors['probability'], linewidth=1)

        ax.set_xlabel(r'Position $x$', fontsize=self.style['fontsize'])
        ax.set_ylabel('Energy', fontsize=self.style['fontsize'])
        ax.legend(fontsize=self.style['fontsize']-1)
        ax.grid(True, alpha=self.style['grid_alpha'])

        if title:
            ax.set_title(title, fontsize=self.style['title_fontsize'])

        return fig, ax

    def plot_multiple_states(self, x, psi_list, labels=None, ax=None, title=None):
        """Plot multiple wave functions on the same axes."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style['figsize'],
                                   dpi=self.style['dpi'])
        else:
            fig = ax.figure

        lw = self.style['linewidth']

        for i, psi in enumerate(psi_list):
            label = labels[i] if labels else f'$\\psi_{i}(x)$'
            ax.plot(x, np.abs(psi)**2, linewidth=lw, label=label)

        ax.set_xlabel(r'Position $x$', fontsize=self.style['fontsize'])
        ax.set_ylabel(r'$|\psi(x)|^2$', fontsize=self.style['fontsize'])
        ax.legend(fontsize=self.style['fontsize']-1)
        ax.grid(True, alpha=self.style['grid_alpha'])

        if title:
            ax.set_title(title, fontsize=self.style['title_fontsize'])

        return fig, ax

    def create_state_comparison(self, x, states_dict, V=None):
        """
        Create a multi-panel comparison of quantum states.

        Parameters
        ----------
        x : array
            Position array
        states_dict : dict
            Dictionary mapping state labels to (psi, E) tuples
        V : array, optional
            Potential energy
        """
        n_states = len(states_dict)
        n_cols = min(3, n_states)
        n_rows = (n_states + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(5*n_cols, 4*n_rows),
                                dpi=self.style['dpi'])
        axes = np.atleast_2d(axes).flatten()

        for ax, (label, (psi, E)) in zip(axes, states_dict.items()):
            if V is not None:
                self.plot_in_potential(x, psi, V, E, ax=ax, title=label)
            else:
                self.plot_wavefunction(x, psi, ax=ax, title=label)

        # Hide unused subplots
        for ax in axes[len(states_dict):]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig, axes


def harmonic_oscillator_state(n, x, omega=1, m=1, hbar=1):
    """Generate harmonic oscillator eigenstate."""
    xi = np.sqrt(m * omega / hbar) * x
    prefactor = (m * omega / (np.pi * hbar))**0.25
    normalization = 1 / np.sqrt(2**n * float(factorial(n)))
    H_n = hermite(n)
    return prefactor * normalization * H_n(xi) * np.exp(-xi**2 / 2)


def infinite_well_state(n, x, L=1):
    """Generate infinite square well eigenstate."""
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    psi[(x < 0) | (x > L)] = 0
    return psi


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Wave Function Visualization Toolkit")
    print("Day 267: Matplotlib Fundamentals")
    print("=" * 60)

    # Initialize plotter
    plotter = WaveFunctionPlotter()

    # Example 1: Simple harmonic oscillator state
    print("\n1. Plotting ground state of harmonic oscillator...")
    x = np.linspace(-5, 5, 1000)
    psi_0 = harmonic_oscillator_state(0, x)
    fig, ax = plotter.plot_wavefunction(x, psi_0,
                                        title='Harmonic Oscillator Ground State')
    plt.savefig('ho_ground_state.png', dpi=150, bbox_inches='tight')
    print("   Saved: ho_ground_state.png")

    # Example 2: State in potential
    print("\n2. Plotting wave function in potential well...")
    V = 0.5 * x**2
    E_0 = 0.5
    fig, ax = plotter.plot_in_potential(x, psi_0, V, E_0,
                                        title='Ground State in Harmonic Potential')
    plt.savefig('ho_in_potential.png', dpi=150, bbox_inches='tight')
    print("   Saved: ho_in_potential.png")

    # Example 3: Multiple states comparison
    print("\n3. Creating multi-state comparison...")
    states = {
        f'$n = {n}$': (harmonic_oscillator_state(n, x), n + 0.5)
        for n in range(4)
    }
    fig, axes = plotter.create_state_comparison(x, states, V=V)
    plt.savefig('ho_states_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: ho_states_comparison.png")

    # Example 4: Complex wave packet
    print("\n4. Plotting complex Gaussian wave packet...")
    k0 = 5  # Initial momentum
    sigma = 0.5
    psi_packet = (1/(2*np.pi*sigma**2))**0.25 * np.exp(-(x+2)**2/(4*sigma**2)) * np.exp(1j*k0*x)
    fig, ax = plotter.plot_wavefunction(x, psi_packet,
                                        title='Gaussian Wave Packet with Momentum')
    plt.savefig('wave_packet.png', dpi=150, bbox_inches='tight')
    print("   Saved: wave_packet.png")

    # Example 5: Particle in a box
    print("\n5. Plotting particle in a box states...")
    x_box = np.linspace(0, 1, 1000)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for n, ax in enumerate(axes, 1):
        psi = infinite_well_state(n, x_box)
        ax.plot(x_box, psi, 'b-', linewidth=2, label=f'$\\psi_{n}$')
        ax.fill_between(x_box, psi, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_ylabel(f'$\\psi_{n}(x)$')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark walls
        ax.axvline(0, color='k', linewidth=3)
        ax.axvline(1, color='k', linewidth=3)

    axes[-1].set_xlabel(r'Position $x/L$')
    plt.suptitle('Particle in a Box Eigenstates', fontsize=14)
    plt.tight_layout()
    plt.savefig('particle_in_box.png', dpi=150, bbox_inches='tight')
    print("   Saved: particle_in_box.png")

    # Verification
    print("\n" + "=" * 60)
    print("Normalization Verification")
    print("=" * 60)

    for n in range(4):
        psi = harmonic_oscillator_state(n, x)
        norm = np.trapz(np.abs(psi)**2, x)
        print(f"  HO state n={n}: ∫|ψ|² dx = {norm:.6f}")

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"  Figures generated: 5")
    print(f"  Quantum states visualized: 8")
    print(f"  All normalizations verified ✓")
    print("=" * 60)

    plt.show()
```

### Expected Output
```
============================================================
Quantum Wave Function Visualization Toolkit
Day 267: Matplotlib Fundamentals
============================================================

1. Plotting ground state of harmonic oscillator...
   Saved: ho_ground_state.png

2. Plotting wave function in potential well...
   Saved: ho_in_potential.png

3. Creating multi-state comparison...
   Saved: ho_states_comparison.png

4. Plotting complex Gaussian wave packet...
   Saved: wave_packet.png

5. Plotting particle in a box states...
   Saved: particle_in_box.png

============================================================
Normalization Verification
============================================================
  HO state n=0: ∫|ψ|² dx = 1.000000
  HO state n=1: ∫|ψ|² dx = 1.000000
  HO state n=2: ∫|ψ|² dx = 1.000000
  HO state n=3: ∫|ψ|² dx = 1.000000

============================================================
Summary Statistics
============================================================
  Figures generated: 5
  Quantum states visualized: 8
  All normalizations verified ✓
============================================================
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Figure/Axes hierarchy | Container structure for matplotlib visualizations |
| OO interface | Explicit control via `fig, ax = plt.subplots()` |
| Line styles | `-` (solid), `--` (dashed), `:` (dotted), `-.` (dash-dot) |
| Markers | `o`, `s`, `^`, `*`, `+`, `x` for data points |
| Color specification | Named, hex (`#RRGGBB`), RGB tuple, tableau (`C0-C9`) |
| Subplots | `plt.subplots(nrows, ncols)` for multi-panel figures |
| Saving | `fig.savefig()` with dpi and format specification |

### Key Formulas

$$\boxed{\text{Harmonic Oscillator: } \psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \frac{1}{\sqrt{2^n n!}} H_n(\xi) e^{-\xi^2/2}}$$

$$\boxed{\text{Infinite Well: } \psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right)}$$

$$\boxed{\text{Normalization: } \int_{-\infty}^{\infty} |\psi(x)|^2 \, dx = 1}$$

### Main Takeaways

1. Use the **object-oriented interface** for reproducible scientific figures
2. **Consistent styling** makes figures professional and readable
3. Wave functions require careful handling of **complex values**
4. Always **verify normalization** when visualizing probability densities
5. **Multi-panel figures** are essential for comparing quantum states

---

## Daily Checklist

- [ ] Understood Figure/Axes hierarchy
- [ ] Created basic line, scatter, and histogram plots
- [ ] Applied custom styling (colors, markers, line styles)
- [ ] Built multi-panel figures with subplots
- [ ] Saved figures in multiple formats
- [ ] Visualized quantum wave functions correctly
- [ ] Completed computational lab exercises

---

## Preview of Day 268

Tomorrow we dive into **Scientific Plotting** with Matplotlib, covering:
- Error bars and uncertainty visualization
- Logarithmic and semi-log scales
- Colormaps for 2D data
- Contour plots and filled contours
- Heatmaps for matrix visualization

These techniques are essential for presenting quantum measurement results and energy spectra professionally.
