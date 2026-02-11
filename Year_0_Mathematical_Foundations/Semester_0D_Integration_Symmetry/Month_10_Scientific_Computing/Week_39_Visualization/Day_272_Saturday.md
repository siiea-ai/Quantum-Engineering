# Day 272: Publication-Quality Figures

## Schedule Overview
**Date**: Week 39, Day 6 (Saturday)
**Duration**: 7 hours
**Theme**: Creating Journal-Ready Scientific Visualizations

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | LaTeX integration, typography, journal requirements |
| Afternoon | 2.5 hours | Vector exports, multi-panel layouts, style templates |
| Evening | 1.5 hours | Computational lab: Complete publication figure toolkit |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Configure matplotlib for LaTeX rendering
2. Meet journal-specific figure requirements (Nature, APS, AIP)
3. Export figures in vector formats (PDF, SVG, EPS)
4. Create professional multi-panel figures with proper labeling
5. Build reusable style templates for consistent formatting
6. Optimize figures for both print and screen display

---

## Core Content

### 1. LaTeX Integration

LaTeX rendering enables publication-quality mathematical typesetting.

#### Enabling LaTeX

```python
import matplotlib.pyplot as plt
import numpy as np

# Method 1: Use text.usetex (requires LaTeX installation)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
})

# Method 2: Use mathtext (built-in, no LaTeX needed)
plt.rcParams.update({
    'mathtext.fontset': 'cm',  # Computer Modern
    'font.family': 'serif',
})
```

#### LaTeX Formatting Examples

```python
import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(8, 6))

x = np.linspace(-3, 3, 200)
psi = np.exp(-x**2/2) / np.pi**0.25

ax.plot(x, psi, 'b-', linewidth=2)
ax.plot(x, np.abs(psi)**2, 'r--', linewidth=2)

# LaTeX labels
ax.set_xlabel(r'Position $x$ (units of $\sqrt{\hbar/m\omega}$)', fontsize=14)
ax.set_ylabel(r'Amplitude', fontsize=14)
ax.set_title(r'Ground State: $\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} e^{-\frac{m\omega x^2}{2\hbar}}$',
            fontsize=14)

# Legend with LaTeX
ax.legend([r'$\psi_0(x)$', r'$|\psi_0(x)|^2$'], fontsize=12)

# Annotations
ax.annotate(r'$\langle x \rangle = 0$', xy=(0, 0.75), xytext=(1.5, 0.6),
           fontsize=12, arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('latex_example.pdf', bbox_inches='tight')
plt.show()
```

#### Common LaTeX Patterns for Physics

```python
# Quantum mechanics notation
title_qm = r'$\hat{H}\psi = E\psi$'
xlabel_qm = r'$\langle\hat{x}\rangle$ (nm)'
ylabel_qm = r'$|\langle n|\psi\rangle|^2$'

# Bra-ket notation
bra_ket = r'$\langle\psi|\hat{A}|\phi\rangle$'
state = r'$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$'

# Integrals
integral = r'$\int_{-\infty}^{\infty} |\psi(x)|^2\, dx = 1$'

# Fractions and roots
fraction = r'$E_n = \left(n + \frac{1}{2}\right)\hbar\omega$'
root = r'$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$'

# Matrices
matrix = r'$\hat{\sigma}_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$'

# Greek letters
greek = r'$\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \eta, \theta$'
greek2 = r'$\Gamma, \Delta, \Theta, \Lambda, \Xi, \Pi, \Sigma, \Omega$'
```

### 2. Journal Requirements

Different journals have specific figure requirements:

#### Nature/Science Requirements
- Single column: 89 mm (3.5 in)
- Double column: 183 mm (7.2 in)
- Full page: 247 mm maximum height
- Font: 5-7 pt for labels, sans-serif preferred
- Resolution: 300 dpi minimum for raster, vector preferred

#### APS (Physical Review) Requirements
- Single column: 8.6 cm (3.4 in)
- Double column: 17.8 cm (7 in)
- Font: Times or Computer Modern
- File formats: EPS, PDF preferred

#### AIP Requirements
- Column width: 8.5 cm
- Two-column width: 17 cm
- Font: Matching journal typeface

```python
def create_journal_figure(journal='aps'):
    """Create figure with journal-specific settings."""

    settings = {
        'nature': {
            'single_col': 3.5,
            'double_col': 7.2,
            'font_size': 7,
            'font_family': 'sans-serif',
            'dpi': 300,
        },
        'aps': {
            'single_col': 3.4,
            'double_col': 7.0,
            'font_size': 10,
            'font_family': 'serif',
            'dpi': 300,
        },
        'aip': {
            'single_col': 3.35,
            'double_col': 6.69,
            'font_size': 10,
            'font_family': 'serif',
            'dpi': 300,
        }
    }

    s = settings.get(journal, settings['aps'])

    plt.rcParams.update({
        'font.size': s['font_size'],
        'font.family': s['font_family'],
        'figure.dpi': s['dpi'],
        'savefig.dpi': s['dpi'],
        'axes.labelsize': s['font_size'],
        'axes.titlesize': s['font_size'] + 1,
        'legend.fontsize': s['font_size'] - 1,
        'xtick.labelsize': s['font_size'] - 1,
        'ytick.labelsize': s['font_size'] - 1,
    })

    return s

# Example usage
settings = create_journal_figure('aps')
fig, ax = plt.subplots(figsize=(settings['single_col'], settings['single_col']*0.8))
```

### 3. Vector Graphics Export

Vector formats (PDF, SVG, EPS) maintain quality at any scale.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), 'b-', linewidth=1.5)
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')

# PDF - Best for LaTeX documents
fig.savefig('figure.pdf', bbox_inches='tight', pad_inches=0.02)

# SVG - Best for web
fig.savefig('figure.svg', bbox_inches='tight')

# EPS - For legacy workflows
fig.savefig('figure.eps', bbox_inches='tight', format='eps')

# High-resolution PNG for presentations
fig.savefig('figure.png', dpi=300, bbox_inches='tight', transparent=False)

# Transparent background
fig.savefig('figure_transparent.png', dpi=300, bbox_inches='tight', transparent=True)
```

### 4. Multi-Panel Figures

Professional multi-panel figures with consistent labeling:

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import string

def create_multipanel_figure(n_rows, n_cols, figsize=None, wspace=0.3, hspace=0.3):
    """
    Create a multi-panel figure with automatic panel labels.

    Parameters
    ----------
    n_rows, n_cols : int
        Number of rows and columns
    figsize : tuple, optional
        Figure size in inches
    wspace, hspace : float
        Spacing between panels
    """
    if figsize is None:
        figsize = (3.4 * n_cols, 2.5 * n_rows)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=wspace, hspace=hspace)

    axes = []
    labels = list(string.ascii_lowercase)  # a, b, c, ...

    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            ax = fig.add_subplot(gs[i, j])
            idx = i * n_cols + j
            if idx < len(labels):
                # Add panel label
                ax.text(-0.15, 1.05, f'({labels[idx]})',
                       transform=ax.transAxes, fontsize=12, fontweight='bold')
            row.append(ax)
        axes.append(row)

    return fig, np.array(axes)

# Example: 2x2 panel figure
fig, axes = create_multipanel_figure(2, 2, figsize=(7, 5.5))

x = np.linspace(-5, 5, 200)

# Panel (a): Wave function
psi = np.exp(-x**2/2) / np.pi**0.25
axes[0, 0].plot(x, psi, 'b-', linewidth=1.5)
axes[0, 0].set_xlabel(r'$x$')
axes[0, 0].set_ylabel(r'$\psi(x)$')
axes[0, 0].set_title('Wave Function')

# Panel (b): Probability density
axes[0, 1].plot(x, np.abs(psi)**2, 'r-', linewidth=1.5)
axes[0, 1].fill_between(x, np.abs(psi)**2, alpha=0.3, color='red')
axes[0, 1].set_xlabel(r'$x$')
axes[0, 1].set_ylabel(r'$|\psi(x)|^2$')
axes[0, 1].set_title('Probability Density')

# Panel (c): Potential
V = 0.5 * x**2
axes[1, 0].plot(x, V, 'k-', linewidth=1.5)
axes[1, 0].set_xlabel(r'$x$')
axes[1, 0].set_ylabel(r'$V(x)$')
axes[1, 0].set_title('Potential')

# Panel (d): Energy levels
for n in range(5):
    E_n = n + 0.5
    axes[1, 1].axhline(E_n, color=f'C{n}', linewidth=2, label=f'$n={n}$')
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 5)
axes[1, 1].set_ylabel('Energy')
axes[1, 1].legend(loc='right', fontsize=8)
axes[1, 1].set_title('Energy Levels')
axes[1, 1].set_xticks([])

plt.savefig('multipanel_figure.pdf', bbox_inches='tight')
plt.show()
```

### 5. Style Templates

Create reusable style files for consistent formatting:

```python
# Create a custom style dictionary
PUBLICATION_STYLE = {
    # Figure
    'figure.figsize': (3.4, 2.5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,

    # Fonts
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'mathtext.fontset': 'cm',

    # Axes
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'axes.spines.top': True,
    'axes.spines.right': True,

    # Ticks
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    # Lines
    'lines.linewidth': 1.0,
    'lines.markersize': 4,

    # Legend
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'gray',

    # Grid
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
}

def apply_publication_style():
    """Apply publication-quality style settings."""
    plt.rcParams.update(PUBLICATION_STYLE)

def reset_style():
    """Reset to matplotlib defaults."""
    plt.rcdefaults()

# Save as .mplstyle file
def save_style_file(filename='publication.mplstyle'):
    """Save style as matplotlib style file."""
    with open(filename, 'w') as f:
        for key, value in PUBLICATION_STYLE.items():
            if isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            elif isinstance(value, bool):
                value = str(value).lower()
            f.write(f'{key}: {value}\n')
    print(f"Style saved to {filename}")
    print("Use with: plt.style.use('publication.mplstyle')")

# Usage
apply_publication_style()
fig, ax = plt.subplots()
# ... create your figure ...
```

### 6. Color and Accessibility

Publication figures must be accessible to colorblind readers.

```python
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe palettes
CB_PALETTE = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'pink': '#CC79A7',
    'sky': '#56B4E9',
    'yellow': '#F0E442',
    'vermillion': '#D55E00',
    'black': '#000000'
}

# Palette from ColorBrewer (colorblind safe)
COLORBREWER_QUALITATIVE = [
    '#1b9e77',  # Teal
    '#d95f02',  # Orange
    '#7570b3',  # Purple
    '#e7298a',  # Pink
    '#66a61e',  # Green
    '#e6ab02',  # Gold
    '#a6761d',  # Brown
    '#666666'   # Gray
]

def set_colorblind_palette():
    """Set colorblind-friendly default colors."""
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=list(CB_PALETTE.values()))

# Example with accessibility features
fig, ax = plt.subplots(figsize=(6, 4))

x = np.linspace(0, 10, 100)

# Use different line styles AND colors for redundant encoding
styles = ['-', '--', '-.', ':']
colors = list(CB_PALETTE.values())[:4]
labels = ['State 1', 'State 2', 'State 3', 'State 4']

for i, (style, color, label) in enumerate(zip(styles, colors, labels)):
    ax.plot(x, np.sin(x + i*np.pi/4), linestyle=style, color=color,
           linewidth=1.5, label=label)

ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.legend()
ax.set_title('Colorblind-Accessible Figure')

plt.tight_layout()
plt.savefig('accessible_figure.pdf', bbox_inches='tight')
plt.show()
```

---

## Quantum Mechanics Connection

### Publication-Ready Quantum Figures

Example figures following journal standards:

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.special import hermite
from math import factorial

def create_qho_publication_figure():
    """Create publication-ready quantum harmonic oscillator figure."""

    # Apply style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
    })

    # Single column figure for APS journal
    fig = plt.figure(figsize=(3.4, 4.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)

    x = np.linspace(-5, 5, 500)
    V = 0.5 * x**2

    # Panel (a): Wave functions in potential
    ax1 = fig.add_subplot(gs[0])
    ax1.text(-0.12, 1.02, r'\textbf{(a)}', transform=ax1.transAxes, fontsize=10)

    # Plot potential
    ax1.fill_between(x, V, alpha=0.1, color='gray')
    ax1.plot(x, V, 'k-', linewidth=0.5, alpha=0.5)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))

    for n in range(4):
        E_n = n + 0.5
        prefactor = (1/np.pi)**0.25
        norm = 1 / np.sqrt(2**n * factorial(n))
        H_n = hermite(n)
        psi_n = prefactor * norm * H_n(x) * np.exp(-x**2/2)

        # Scale and shift
        scale = 0.35
        ax1.fill_between(x, E_n, E_n + scale * np.abs(psi_n)**2,
                        alpha=0.5, color=colors[n])
        ax1.plot(x, E_n + scale * np.abs(psi_n)**2, color=colors[n], linewidth=0.8)
        ax1.axhline(E_n, color=colors[n], linestyle='--', linewidth=0.5, alpha=0.7)

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(0, 4.5)
    ax1.set_xlabel(r'Position $x$ ($\sqrt{\hbar/m\omega}$)')
    ax1.set_ylabel(r'Energy ($\hbar\omega$)')
    ax1.set_title(r'Quantum Harmonic Oscillator Eigenstates')

    # Panel (b): Energy level diagram
    ax2 = fig.add_subplot(gs[1])
    ax2.text(-0.12, 1.05, r'\textbf{(b)}', transform=ax2.transAxes, fontsize=10)

    for n in range(6):
        E_n = n + 0.5
        ax2.hlines(E_n, 0.2, 0.8, colors=plt.cm.viridis(n/6), linewidth=2)
        ax2.text(0.85, E_n, f'$n={n}$', va='center', fontsize=8)

    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(0, 6)
    ax2.set_ylabel(r'$E_n/\hbar\omega$')
    ax2.set_xticks([])
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel (c): Transition frequencies
    ax3 = fig.add_subplot(gs[2])
    ax3.text(-0.12, 1.05, r'\textbf{(c)}', transform=ax3.transAxes, fontsize=10)

    n_vals = np.arange(0, 10)
    delta_n = [1, 2, 3]
    markers = ['o', 's', '^']

    for dn, marker in zip(delta_n, markers):
        n_from = n_vals[:-dn]
        omega_trans = np.ones_like(n_from) * dn  # ΔE = ℏω × Δn
        ax3.scatter(n_from, omega_trans, marker=marker, s=30,
                   label=rf'$\Delta n = {dn}$')

    ax3.set_xlabel(r'Initial state $n$')
    ax3.set_ylabel(r'$\Delta E/\hbar\omega$')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.set_xlim(-0.5, 9.5)
    ax3.set_ylim(0, 4)

    plt.savefig('qho_publication.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig('qho_publication.png', dpi=300, bbox_inches='tight')

    return fig

fig = create_qho_publication_figure()
plt.show()
```

### Density Matrix Visualization for Publications

```python
def create_density_matrix_figure():
    """Publication-quality density matrix visualization."""

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'text.usetex': True,
    })

    # Create sample density matrices
    # Pure state: |+⟩ = (|0⟩ + |1⟩)/√2
    psi_plus = np.array([1, 1]) / np.sqrt(2)
    rho_pure = np.outer(psi_plus, psi_plus.conj())

    # Mixed state: 50% |0⟩⟨0| + 50% |1⟩⟨1|
    rho_mixed = 0.5 * np.eye(2)

    # Partially mixed
    rho_partial = 0.7 * np.outer(psi_plus, psi_plus.conj()) + 0.3 * 0.5 * np.eye(2)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    matrices = [rho_pure, rho_partial, rho_mixed]
    titles = [
        r'Pure state $|+\rangle$',
        r'Partially mixed',
        r'Maximally mixed'
    ]
    purities = [np.trace(rho @ rho).real for rho in matrices]

    for ax, rho, title, purity, label in zip(axes, matrices, titles, purities, 'abc'):
        im = ax.imshow(np.abs(rho), cmap='Blues', vmin=0, vmax=1)

        # Add text annotations
        for i in range(2):
            for j in range(2):
                val = rho[i, j]
                text = f'{val.real:.2f}'
                if np.abs(val.imag) > 0.01:
                    text += f'\n{val.imag:+.2f}i'
                color = 'white' if np.abs(val) > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([r'$|0\rangle$', r'$|1\rangle$'])
        ax.set_yticklabels([r'$\langle 0|$', r'$\langle 1|$'])
        ax.set_title(f'{title}\n$\\gamma = {purity:.2f}$', fontsize=9)

        ax.text(-0.15, 1.1, rf'\textbf{{({label})}}', transform=ax.transAxes, fontsize=10)

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label=r'$|\rho_{ij}|$')

    plt.savefig('density_matrices.pdf', bbox_inches='tight')
    return fig

fig = create_density_matrix_figure()
plt.show()
```

---

## Worked Examples

### Example 1: Complete Journal Figure

**Problem**: Create a figure suitable for Physical Review Letters showing tunneling probability vs. barrier parameters.

**Solution**:
```python
import matplotlib.pyplot as plt
import numpy as np

def transmission_coefficient(E, V0, a, m=1, hbar=1):
    """Calculate transmission coefficient for rectangular barrier."""
    if E >= V0:
        # Above barrier
        k1 = np.sqrt(2*m*E) / hbar
        k2 = np.sqrt(2*m*(E-V0)) / hbar
        T = 4*k1*k2 / ((k1+k2)**2 * np.sin(k2*a)**2 + 4*k1*k2*np.cos(k2*a)**2)
    else:
        # Below barrier
        k = np.sqrt(2*m*E) / hbar
        kappa = np.sqrt(2*m*(V0-E)) / hbar
        denom = ((k**2 + kappa**2)**2 / (4*k**2*kappa**2)) * np.sinh(kappa*a)**2 + np.cosh(kappa*a)**2
        T = 1 / denom
    return T

# Vectorize
transmission_vec = np.vectorize(transmission_coefficient)

# PRL style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'text.usetex': True,
    'figure.figsize': (3.4, 2.8),
})

fig, ax = plt.subplots()

# Energy range
E = np.linspace(0.01, 2, 200)
V0 = 1.0  # Barrier height

# Different barrier widths
widths = [0.5, 1.0, 2.0, 3.0]
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
styles = ['-', '--', '-.', ':']

for a, color, style in zip(widths, colors, styles):
    T = transmission_vec(E, V0, a)
    ax.plot(E/V0, T, color=color, linestyle=style, linewidth=1.2,
           label=f'$a = {a}$')

# Mark barrier height
ax.axvline(1, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.text(1.02, 0.5, r'$E = V_0$', fontsize=8, rotation=90, va='center')

ax.set_xlabel(r'Energy $E/V_0$')
ax.set_ylabel(r'Transmission $T$')
ax.set_xlim(0, 2)
ax.set_ylim(0, 1.05)
ax.legend(title=r'Width $a$', fontsize=8, title_fontsize=8, loc='lower right')

# Add inset showing barrier
axins = ax.inset_axes([0.15, 0.55, 0.35, 0.35])
x_barrier = np.linspace(-2, 2, 100)
V_barrier = np.where(np.abs(x_barrier) < 0.5, 1, 0)
axins.fill_between(x_barrier, V_barrier, alpha=0.3, color='gray')
axins.plot(x_barrier, V_barrier, 'k-', linewidth=1)
axins.set_xlim(-2, 2)
axins.set_ylim(-0.1, 1.3)
axins.set_xlabel(r'$x$', fontsize=7)
axins.set_ylabel(r'$V$', fontsize=7)
axins.tick_params(labelsize=6)
axins.set_title('Barrier', fontsize=7)

plt.savefig('tunneling_prl.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('tunneling_prl.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Example 2: Supplementary Figure with Insets

**Problem**: Create a detailed supplementary figure with multiple insets.

**Solution**:
```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.special import hermite
from math import factorial

def create_supplementary_figure():
    """Create detailed supplementary figure."""

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'text.usetex': True,
    })

    fig = plt.figure(figsize=(7, 8))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    x = np.linspace(-6, 6, 500)

    # Panel A: All eigenstates
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.text(-0.05, 1.02, r'\textbf{(a)}', transform=ax_a.transAxes, fontsize=10)

    for n in range(8):
        prefactor = (1/np.pi)**0.25
        norm = 1 / np.sqrt(2**n * factorial(n))
        H_n = hermite(n)
        psi_n = prefactor * norm * H_n(x) * np.exp(-x**2/2)

        offset = n * 0.4
        ax_a.plot(x, offset + psi_n * 0.3, linewidth=0.8, label=f'$n={n}$')

    ax_a.set_xlabel(r'Position $x$')
    ax_a.set_ylabel(r'$\psi_n(x)$ (offset for clarity)')
    ax_a.set_title('Harmonic Oscillator Eigenstates')
    ax_a.legend(ncol=4, fontsize=7, loc='upper right')

    # Panel B: Ground state detail
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.text(-0.15, 1.02, r'\textbf{(b)}', transform=ax_b.transAxes, fontsize=10)

    psi_0 = (1/np.pi)**0.25 * np.exp(-x**2/2)
    ax_b.plot(x, psi_0, 'b-', linewidth=1.2, label=r'$\psi_0$')
    ax_b.plot(x, np.abs(psi_0)**2, 'r--', linewidth=1.2, label=r'$|\psi_0|^2$')
    ax_b.axhline(0, color='gray', linewidth=0.5)
    ax_b.fill_between(x, np.abs(psi_0)**2, alpha=0.2, color='red')
    ax_b.set_xlabel(r'$x$')
    ax_b.set_ylabel('Amplitude')
    ax_b.set_title('Ground State')
    ax_b.legend(fontsize=7)

    # Inset: zoom near peak
    axins_b = ax_b.inset_axes([0.6, 0.5, 0.35, 0.35])
    mask = np.abs(x) < 1
    axins_b.plot(x[mask], psi_0[mask], 'b-', linewidth=1)
    axins_b.plot(x[mask], np.abs(psi_0[mask])**2, 'r--', linewidth=1)
    axins_b.set_xlim(-1, 1)
    axins_b.tick_params(labelsize=6)

    # Panel C: First excited state
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.text(-0.15, 1.02, r'\textbf{(c)}', transform=ax_c.transAxes, fontsize=10)

    psi_1 = (1/np.pi)**0.25 * np.sqrt(2) * x * np.exp(-x**2/2)
    ax_c.plot(x, psi_1, 'b-', linewidth=1.2, label=r'$\psi_1$')
    ax_c.plot(x, np.abs(psi_1)**2, 'r--', linewidth=1.2, label=r'$|\psi_1|^2$')
    ax_c.axhline(0, color='gray', linewidth=0.5)
    ax_c.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax_c.fill_between(x, np.abs(psi_1)**2, alpha=0.2, color='red')
    ax_c.set_xlabel(r'$x$')
    ax_c.set_ylabel('Amplitude')
    ax_c.set_title('First Excited State (node at $x=0$)')
    ax_c.legend(fontsize=7)

    # Panel D: Position uncertainty
    ax_d = fig.add_subplot(gs[2, 0])
    ax_d.text(-0.15, 1.02, r'\textbf{(d)}', transform=ax_d.transAxes, fontsize=10)

    n_states = np.arange(0, 20)
    delta_x = np.sqrt(n_states + 0.5)
    delta_p = np.sqrt(n_states + 0.5)

    ax_d.plot(n_states, delta_x, 'o-', markersize=4, linewidth=1, label=r'$\Delta x$')
    ax_d.plot(n_states, delta_p, 's--', markersize=4, linewidth=1, label=r'$\Delta p$')
    ax_d.set_xlabel('Quantum number $n$')
    ax_d.set_ylabel('Uncertainty')
    ax_d.set_title('Uncertainty vs. Quantum Number')
    ax_d.legend(fontsize=7)
    ax_d.grid(True, alpha=0.3)

    # Panel E: Uncertainty product
    ax_e = fig.add_subplot(gs[2, 1])
    ax_e.text(-0.15, 1.02, r'\textbf{(e)}', transform=ax_e.transAxes, fontsize=10)

    product = delta_x * delta_p
    ax_e.plot(n_states, product, 'o-', markersize=4, linewidth=1, color='purple')
    ax_e.axhline(0.5, color='red', linestyle='--', linewidth=1, label=r'$\hbar/2$ minimum')
    ax_e.set_xlabel('Quantum number $n$')
    ax_e.set_ylabel(r'$\Delta x \cdot \Delta p$')
    ax_e.set_title('Uncertainty Product')
    ax_e.legend(fontsize=7)
    ax_e.grid(True, alpha=0.3)

    plt.savefig('supplementary_figure.pdf', bbox_inches='tight')
    plt.savefig('supplementary_figure.png', dpi=300, bbox_inches='tight')

    return fig

fig = create_supplementary_figure()
plt.show()
```

---

## Practice Problems

### Level 1: Direct Application

1. **LaTeX Labels**: Create a plot with the Schrödinger equation as the title and properly formatted axis labels using LaTeX.

2. **Journal Size**: Create a single-column figure (3.4 inches wide) suitable for Physical Review with proper font sizes.

3. **Vector Export**: Save the same figure as PDF, SVG, and high-resolution PNG. Compare file sizes.

### Level 2: Intermediate

4. **Multi-Panel Figure**: Create a 2×2 figure with panels labeled (a)-(d), each showing different aspects of a quantum system.

5. **Style Template**: Create a complete matplotlib style file for your preferred journal and apply it to a figure.

6. **Colorblind Accessibility**: Redesign a figure to be fully accessible using both color and line style differentiation.

### Level 3: Challenging

7. **Complete Paper Figure**: Create a figure suitable as Figure 1 of a research paper, including main panel and insets, following Nature guidelines.

8. **Data Comparison Figure**: Create a figure comparing theoretical predictions (lines) with simulated experimental data (points with error bars) for quantum tunneling.

9. **Graphical Abstract**: Create a single, visually striking figure that could serve as a graphical abstract for a paper on quantum harmonic oscillators.

---

## Computational Lab

### Project: Publication Figure Toolkit

```python
"""
Publication Figure Toolkit
Day 272: Publication-Quality Figures
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import string


class PublicationFigure:
    """
    Toolkit for creating publication-quality scientific figures.
    """

    # Journal presets
    JOURNALS = {
        'nature': {
            'single_col': 3.5,
            'double_col': 7.2,
            'full_page': 9.75,
            'font_size': 7,
            'font_family': 'sans-serif',
            'dpi': 300,
        },
        'science': {
            'single_col': 2.25,
            'double_col': 4.75,
            'full_page': 6.93,
            'font_size': 7,
            'font_family': 'sans-serif',
            'dpi': 300,
        },
        'aps': {
            'single_col': 3.4,
            'double_col': 7.0,
            'full_page': 9.5,
            'font_size': 10,
            'font_family': 'serif',
            'dpi': 300,
        },
        'aip': {
            'single_col': 3.35,
            'double_col': 6.69,
            'full_page': 9.5,
            'font_size': 10,
            'font_family': 'serif',
            'dpi': 300,
        },
    }

    # Colorblind-safe palette
    COLORS = [
        '#0072B2',  # Blue
        '#E69F00',  # Orange
        '#009E73',  # Green
        '#CC79A7',  # Pink
        '#56B4E9',  # Sky blue
        '#D55E00',  # Vermillion
        '#F0E442',  # Yellow
        '#000000',  # Black
    ]

    def __init__(self, journal='aps', use_latex=True):
        """
        Initialize publication figure toolkit.

        Parameters
        ----------
        journal : str
            Journal preset ('nature', 'science', 'aps', 'aip')
        use_latex : bool
            Whether to use LaTeX rendering
        """
        self.journal = journal
        self.settings = self.JOURNALS.get(journal, self.JOURNALS['aps'])
        self.use_latex = use_latex

        self._apply_style()

    def _apply_style(self):
        """Apply journal-specific style settings."""
        s = self.settings

        style = {
            'font.size': s['font_size'],
            'font.family': s['font_family'],
            'figure.dpi': s['dpi'],
            'savefig.dpi': s['dpi'],
            'axes.labelsize': s['font_size'],
            'axes.titlesize': s['font_size'] + 1,
            'legend.fontsize': s['font_size'] - 1,
            'xtick.labelsize': s['font_size'] - 1,
            'ytick.labelsize': s['font_size'] - 1,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'axes.linewidth': 0.8,
            'axes.prop_cycle': plt.cycler(color=self.COLORS),
        }

        if self.use_latex:
            style.update({
                'text.usetex': True,
                'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
            })

        plt.rcParams.update(style)

    def create_figure(self, width='single', height_ratio=0.75, n_panels=None):
        """
        Create a figure with journal-appropriate dimensions.

        Parameters
        ----------
        width : str or float
            'single', 'double', 'full', or width in inches
        height_ratio : float
            Height as fraction of width
        n_panels : tuple, optional
            (n_rows, n_cols) for multi-panel figure
        """
        if isinstance(width, str):
            width_inch = self.settings[f'{width}_col']
        else:
            width_inch = width

        height_inch = width_inch * height_ratio

        if n_panels:
            n_rows, n_cols = n_panels
            fig, axes = plt.subplots(n_rows, n_cols,
                                    figsize=(width_inch, height_inch))

            # Add panel labels
            axes_flat = np.array(axes).flatten()
            for i, ax in enumerate(axes_flat):
                if i < 26:
                    label = f'({string.ascii_lowercase[i]})'
                    if self.use_latex:
                        label = r'\textbf{' + label + '}'
                    ax.text(-0.12, 1.05, label,
                           transform=ax.transAxes,
                           fontsize=self.settings['font_size'] + 1)

            return fig, axes
        else:
            return plt.subplots(figsize=(width_inch, height_inch))

    def save(self, fig, filename, formats=['pdf', 'png'], **kwargs):
        """
        Save figure in multiple formats.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to save
        filename : str
            Base filename (without extension)
        formats : list
            List of formats to save
        """
        save_kwargs = {
            'bbox_inches': 'tight',
            'pad_inches': 0.02,
            'dpi': self.settings['dpi'],
        }
        save_kwargs.update(kwargs)

        for fmt in formats:
            output_file = f'{filename}.{fmt}'
            fig.savefig(output_file, format=fmt, **save_kwargs)
            print(f"Saved: {output_file}")

    def add_inset(self, ax, position, **kwargs):
        """
        Add an inset axes.

        Parameters
        ----------
        ax : Axes
            Parent axes
        position : list
            [x, y, width, height] in axes coordinates
        """
        return ax.inset_axes(position, **kwargs)

    def format_axis(self, ax, xlabel=None, ylabel=None, title=None,
                   xlim=None, ylim=None, legend=True, grid=False):
        """
        Apply consistent formatting to an axis.
        """
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend()
        if grid:
            ax.grid(True, alpha=0.3)

    @staticmethod
    def get_color(index):
        """Get colorblind-safe color by index."""
        return PublicationFigure.COLORS[index % len(PublicationFigure.COLORS)]


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Publication Figure Toolkit")
    print("Day 272: Publication-Quality Figures")
    print("=" * 60)

    # Create toolkit
    pub = PublicationFigure(journal='aps', use_latex=False)

    # Demo 1: Single panel figure
    print("\n1. Creating single-panel figure...")
    fig, ax = pub.create_figure(width='single', height_ratio=0.8)

    x = np.linspace(0, 10, 100)
    for i in range(4):
        ax.plot(x, np.sin(x + i*np.pi/4), label=f'Phase {i}',
               color=pub.get_color(i))

    pub.format_axis(ax, xlabel='Time (a.u.)', ylabel='Amplitude',
                   title='Quantum Oscillations', legend=True, grid=True)

    pub.save(fig, 'demo_single', formats=['pdf', 'png'])

    # Demo 2: Multi-panel figure
    print("\n2. Creating multi-panel figure...")
    fig, axes = pub.create_figure(width='double', height_ratio=0.6, n_panels=(1, 3))

    x = np.linspace(-5, 5, 200)
    psi_0 = np.exp(-x**2/2) / np.pi**0.25
    psi_1 = np.sqrt(2) * x * np.exp(-x**2/2) / np.pi**0.25
    psi_2 = (2*x**2 - 1) * np.exp(-x**2/2) / (np.sqrt(2) * np.pi**0.25)

    for ax, psi, n in zip(axes, [psi_0, psi_1, psi_2], range(3)):
        ax.plot(x, psi, color=pub.get_color(0), linewidth=1.2)
        ax.fill_between(x, psi, alpha=0.3, color=pub.get_color(0))
        ax.axhline(0, color='gray', linewidth=0.5)
        pub.format_axis(ax, xlabel='$x$', ylabel=f'$\\psi_{n}(x)$',
                       xlim=(-4, 4), ylim=(-0.8, 0.8))

    plt.tight_layout()
    pub.save(fig, 'demo_multipanel', formats=['pdf', 'png'])

    # Demo 3: Figure with inset
    print("\n3. Creating figure with inset...")
    fig, ax = pub.create_figure(width='single', height_ratio=0.9)

    # Main plot
    x = np.linspace(0, 10, 200)
    y = np.exp(-x/3) * np.sin(5*x)
    ax.plot(x, y, color=pub.get_color(0), linewidth=1.2)
    pub.format_axis(ax, xlabel='Time', ylabel='Signal',
                   title='Damped Oscillation')

    # Inset
    axins = pub.add_inset(ax, [0.55, 0.55, 0.4, 0.35])
    mask = x < 2
    axins.plot(x[mask], y[mask], color=pub.get_color(0), linewidth=1)
    axins.set_xlim(0, 2)
    axins.tick_params(labelsize=6)
    axins.set_title('Early time', fontsize=7)

    pub.save(fig, 'demo_inset', formats=['pdf', 'png'])

    print("\n" + "=" * 60)
    print("All demonstration figures saved!")
    print("=" * 60)

    plt.show()
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| `text.usetex` | Enable LaTeX rendering |
| Vector formats | PDF, SVG, EPS for scalable graphics |
| Journal specs | Width, font size, DPI requirements |
| Panel labels | (a), (b), etc. for multi-panel figures |
| Colorblind safety | Use distinguishable colors + line styles |
| Style files | `.mplstyle` for consistent formatting |

### Journal Quick Reference

| Journal | Single Column | Font Size | Format |
|---------|--------------|-----------|--------|
| Nature | 3.5 in | 7 pt | PDF/TIFF |
| Science | 2.25 in | 7 pt | PDF/EPS |
| Phys. Rev. | 3.4 in | 10 pt | PDF/EPS |
| AIP | 3.35 in | 10 pt | PDF/EPS |

---

## Daily Checklist

- [ ] Configured matplotlib for LaTeX rendering
- [ ] Created figures meeting journal specifications
- [ ] Exported figures in vector formats
- [ ] Built multi-panel figures with proper labels
- [ ] Created reusable style template
- [ ] Ensured colorblind accessibility
- [ ] Completed computational lab exercises

---

## Preview of Day 273

Tomorrow is **Week Review Day** where we:
- Integrate all visualization techniques from the week
- Build a comprehensive quantum visualization package
- Create a complete style guide
- Practice creating publication-ready figures
- Review key concepts and best practices
