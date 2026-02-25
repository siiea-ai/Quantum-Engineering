"""
Matplotlib Configuration for Publication-Quality Figures
=========================================================

This module provides utilities for creating publication-ready
figures that integrate seamlessly with LaTeX documents.

Author: Quantum Engineering PhD Program
Week 212: Scientific Writing Tools

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Tuple, List, Optional, Dict, Any
import os


# =============================================================================
# Style Configuration
# =============================================================================

# Physical Review / APS style
APS_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.375, 2.5),  # APS single column width
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'legend.frameon': False,
    'legend.borderpad': 0.3,
    'legend.labelspacing': 0.3,
    'text.usetex': False,  # Set True if LaTeX available
    'mathtext.fontset': 'cm',
}

# Nature style
NATURE_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.figsize': (3.5, 2.625),  # Nature single column
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.8,
    'lines.markersize': 3,
    'legend.frameon': False,
    'text.usetex': False,
    'mathtext.fontset': 'stixsans',
}

# IEEE style
IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.5, 2.625),  # IEEE single column
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
}


def set_style(style: str = 'aps'):
    """
    Set matplotlib style for specific journal.

    Parameters
    ----------
    style : str
        Journal style: 'aps', 'nature', or 'ieee'
    """
    style_dict = {
        'aps': APS_STYLE,
        'nature': NATURE_STYLE,
        'ieee': IEEE_STYLE
    }

    if style not in style_dict:
        raise ValueError(f"Unknown style: {style}. Use 'aps', 'nature', or 'ieee'")

    rcParams.update(style_dict[style])
    print(f"Set {style.upper()} style")


def enable_latex():
    """Enable LaTeX rendering (requires LaTeX installation)."""
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{physics}
    '''
    print("LaTeX rendering enabled")


# =============================================================================
# Color Schemes
# =============================================================================

# Colorblind-friendly palette
COLORBLIND_COLORS = [
    '#0077BB',  # Blue
    '#EE7733',  # Orange
    '#009988',  # Teal
    '#CC3311',  # Red
    '#33BBEE',  # Cyan
    '#EE3377',  # Magenta
    '#BBBBBB',  # Grey
]

# APS-style colors
APS_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
]


def get_colors(n: int, palette: str = 'colorblind') -> List[str]:
    """
    Get n colors from specified palette.

    Parameters
    ----------
    n : int
        Number of colors needed
    palette : str
        'colorblind' or 'aps'

    Returns
    -------
    list
        List of color hex codes
    """
    palettes = {
        'colorblind': COLORBLIND_COLORS,
        'aps': APS_COLORS
    }

    colors = palettes.get(palette, COLORBLIND_COLORS)

    if n <= len(colors):
        return colors[:n]
    else:
        # Cycle colors if more needed
        return [colors[i % len(colors)] for i in range(n)]


# =============================================================================
# Figure Creation Utilities
# =============================================================================

def create_figure(nrows: int = 1, ncols: int = 1,
                  width: str = 'single',
                  height_ratio: float = 0.75,
                  style: str = 'aps') -> Tuple[plt.Figure, Any]:
    """
    Create figure with appropriate size for publication.

    Parameters
    ----------
    nrows : int
        Number of subplot rows
    ncols : int
        Number of subplot columns
    width : str
        'single' or 'double' column
    height_ratio : float
        Height as fraction of width
    style : str
        Journal style

    Returns
    -------
    tuple
        (figure, axes)
    """
    set_style(style)

    # Column widths in inches
    widths = {
        'aps': {'single': 3.375, 'double': 7.0},
        'nature': {'single': 3.5, 'double': 7.2},
        'ieee': {'single': 3.5, 'double': 7.16}
    }

    fig_width = widths[style][width] * ncols
    fig_height = fig_width * height_ratio * nrows / ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    return fig, axes


def add_panel_labels(axes, labels: List[str] = None,
                     loc: str = 'upper left',
                     fontweight: str = 'bold'):
    """
    Add panel labels (a), (b), (c) to subplots.

    Parameters
    ----------
    axes : array-like
        Matplotlib axes
    labels : list, optional
        Custom labels (default: (a), (b), ...)
    loc : str
        Label location
    fontweight : str
        Font weight for labels
    """
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    else:
        axes = axes.flatten()

    if labels is None:
        labels = [f'({chr(97+i)})' for i in range(len(axes))]

    locs = {
        'upper left': (0.02, 0.98),
        'upper right': (0.98, 0.98),
        'lower left': (0.02, 0.02),
        'lower right': (0.98, 0.02)
    }

    ha = 'left' if 'left' in loc else 'right'
    va = 'top' if 'upper' in loc else 'bottom'
    x, y = locs[loc]

    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes,
                fontweight=fontweight, ha=ha, va=va)


# =============================================================================
# Common Plot Types
# =============================================================================

def plot_convergence(iterations: np.ndarray, values: np.ndarray,
                     exact_value: float = None,
                     xlabel: str = 'Iteration',
                     ylabel: str = 'Energy (Ha)',
                     label: str = 'VQE',
                     save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create convergence plot for optimization.

    Parameters
    ----------
    iterations : np.ndarray
        Iteration numbers
    values : np.ndarray
        Objective values at each iteration
    exact_value : float, optional
        Exact solution for reference line
    xlabel, ylabel : str
        Axis labels
    label : str
        Legend label
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    fig, ax = create_figure(style='aps')

    colors = get_colors(3)

    ax.plot(iterations, values, '-', color=colors[0],
            linewidth=1.5, label=label)

    if exact_value is not None:
        ax.axhline(y=exact_value, color=colors[1], linestyle='--',
                   linewidth=1, label=f'Exact = {exact_value:.4f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_error_comparison(methods: List[str], errors: List[np.ndarray],
                          xlabel: str = 'Method',
                          ylabel: str = 'Error',
                          save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create bar chart comparing errors across methods.

    Parameters
    ----------
    methods : list
        Method names
    errors : list
        Arrays of error values for each method
    xlabel, ylabel : str
        Axis labels
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    fig, ax = create_figure(style='aps')

    colors = get_colors(len(methods))
    x = np.arange(len(methods))

    means = [np.mean(e) for e in errors]
    stds = [np.std(e) for e in errors]

    bars = ax.bar(x, means, yerr=stds, capsize=3,
                  color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_heatmap(data: np.ndarray,
                 xlabel: str = 'X', ylabel: str = 'Y',
                 cbar_label: str = 'Value',
                 x_ticks: np.ndarray = None,
                 y_ticks: np.ndarray = None,
                 cmap: str = 'viridis',
                 save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create heatmap visualization.

    Parameters
    ----------
    data : np.ndarray
        2D data array
    xlabel, ylabel : str
        Axis labels
    cbar_label : str
        Colorbar label
    x_ticks, y_ticks : np.ndarray, optional
        Tick values
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    fig, ax = create_figure(style='aps', height_ratio=0.9)

    im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if x_ticks is not None:
        n_xticks = min(5, len(x_ticks))
        idx = np.linspace(0, len(x_ticks)-1, n_xticks, dtype=int)
        ax.set_xticks(idx)
        ax.set_xticklabels([f'{x_ticks[i]:.2f}' for i in idx])

    if y_ticks is not None:
        n_yticks = min(5, len(y_ticks))
        idx = np.linspace(0, len(y_ticks)-1, n_yticks, dtype=int)
        ax.set_yticks(idx)
        ax.set_yticklabels([f'{y_ticks[i]:.2f}' for i in idx])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_with_inset(main_data: Tuple[np.ndarray, np.ndarray],
                    inset_data: Tuple[np.ndarray, np.ndarray],
                    inset_loc: List[float] = [0.55, 0.55, 0.4, 0.4],
                    main_label: str = None,
                    inset_label: str = 'Inset',
                    save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create plot with inset.

    Parameters
    ----------
    main_data : tuple
        (x, y) data for main plot
    inset_data : tuple
        (x, y) data for inset
    inset_loc : list
        [x, y, width, height] in figure coordinates
    main_label, inset_label : str
        Labels
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, main_axes, inset_axes)
    """
    fig, ax = create_figure(style='aps')

    colors = get_colors(2)

    # Main plot
    ax.plot(main_data[0], main_data[1], '-', color=colors[0],
            linewidth=1.5, label=main_label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if main_label:
        ax.legend(loc='upper right')

    # Inset
    ax_inset = fig.add_axes(inset_loc)
    ax_inset.plot(inset_data[0], inset_data[1], '-', color=colors[1],
                  linewidth=1)
    ax_inset.set_title(inset_label, fontsize=8)
    ax_inset.tick_params(labelsize=7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax, ax_inset


# =============================================================================
# Quantum-Specific Plots
# =============================================================================

def plot_energy_levels(energies: List[float],
                       labels: List[str] = None,
                       transitions: List[Tuple[int, int]] = None,
                       save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create energy level diagram.

    Parameters
    ----------
    energies : list
        Energy values
    labels : list, optional
        Labels for each level
    transitions : list of tuples, optional
        Pairs of indices for transition arrows
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    fig, ax = create_figure(style='aps', height_ratio=1.2)

    colors = get_colors(len(energies))

    # Draw energy levels
    for i, E in enumerate(energies):
        ax.hlines(E, 0.2, 0.8, colors=colors[i], linewidth=2)
        label = labels[i] if labels else f'E_{i}'
        ax.text(0.85, E, f'{label}: {E:.3f}', va='center', fontsize=9)

    # Draw transitions
    if transitions:
        for i, j in transitions:
            mid_x = 0.5
            ax.annotate('', xy=(mid_x, energies[j]), xytext=(mid_x, energies[i]),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlim(0, 1.5)
    ax.set_ylim(min(energies) - 0.5, max(energies) + 0.5)
    ax.set_ylabel('Energy')
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_probability_distribution(probs: Dict[str, float],
                                   xlabel: str = 'State',
                                   ylabel: str = 'Probability',
                                   highlight: List[str] = None,
                                   save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create probability distribution bar chart.

    Parameters
    ----------
    probs : dict
        State labels to probabilities
    xlabel, ylabel : str
        Axis labels
    highlight : list, optional
        States to highlight
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    fig, ax = create_figure(style='aps')

    states = list(probs.keys())
    values = list(probs.values())

    colors = []
    default_color = get_colors(1)[0]
    highlight_color = '#d62728'

    for s in states:
        if highlight and s in highlight:
            colors.append(highlight_color)
        else:
            colors.append(default_color)

    x = np.arange(len(states))
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=45, ha='right')
    ax.set_ylim(0, max(values) * 1.1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


# =============================================================================
# Demonstration
# =============================================================================

def demo():
    """Demonstrate figure creation for publications."""
    print("=" * 60)
    print("Publication Figure Demonstration")
    print("=" * 60)

    # Create output directory
    output_dir = "demo_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Demo 1: Convergence plot
    print("\n1. VQE Convergence Plot")
    print("-" * 40)

    set_style('aps')
    iterations = np.arange(100)
    values = -1.0 + 0.5 * np.exp(-iterations / 20) + 0.02 * np.random.randn(100)
    fig1, ax1 = plot_convergence(
        iterations, values,
        exact_value=-1.0,
        save_path=f"{output_dir}/convergence.pdf"
    )
    plt.close(fig1)

    # Demo 2: Heatmap
    print("\n2. Energy Landscape Heatmap")
    print("-" * 40)

    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi)
    energy = np.cos(THETA) * np.sin(PHI)

    fig2, ax2 = plot_heatmap(
        energy,
        xlabel=r'$\theta$',
        ylabel=r'$\phi$',
        cbar_label='Energy',
        x_ticks=theta,
        y_ticks=phi,
        save_path=f"{output_dir}/heatmap.pdf"
    )
    plt.close(fig2)

    # Demo 3: Multi-panel figure
    print("\n3. Multi-Panel Figure")
    print("-" * 40)

    fig3, axes3 = create_figure(nrows=1, ncols=2, width='double')

    x = np.linspace(0, 2*np.pi, 100)
    axes3[0].plot(x, np.sin(x), label=r'$\sin(x)$')
    axes3[0].plot(x, np.cos(x), label=r'$\cos(x)$')
    axes3[0].set_xlabel('x')
    axes3[0].set_ylabel('y')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    axes3[1].plot(x, np.sin(x)**2, label=r'$\sin^2(x)$')
    axes3[1].set_xlabel('x')
    axes3[1].set_ylabel(r'$y$')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    add_panel_labels(axes3)

    plt.tight_layout()
    fig3.savefig(f"{output_dir}/multipanel.pdf", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/multipanel.pdf")
    plt.close(fig3)

    # Demo 4: Probability distribution
    print("\n4. Probability Distribution")
    print("-" * 40)

    probs = {'|00>': 0.48, '|01>': 0.02, '|10>': 0.02, '|11>': 0.48}
    fig4, ax4 = plot_probability_distribution(
        probs,
        highlight=['|00>', '|11>'],
        save_path=f"{output_dir}/probability.pdf"
    )
    plt.close(fig4)

    # Demo 5: Energy levels
    print("\n5. Energy Level Diagram")
    print("-" * 40)

    energies = [0, 1.5, 2.8, 4.0]
    labels = [r'$E_0$', r'$E_1$', r'$E_2$', r'$E_3$']
    fig5, ax5 = plot_energy_levels(
        energies, labels,
        transitions=[(0, 1), (1, 2)],
        save_path=f"{output_dir}/energy_levels.pdf"
    )
    plt.close(fig5)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    demo()
