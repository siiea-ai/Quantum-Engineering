"""
Publication-Quality Figure Generation for Quantum Computing Papers
===================================================================

This module provides utilities for creating figures suitable for
scientific publications in quantum computing.

Author: Quantum Engineering PhD Program
Week 212: Scientific Writing Tools

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


# =============================================================================
# Publication Style Configuration
# =============================================================================

def configure_latex_style(use_latex: bool = False):
    """
    Configure matplotlib for publication-quality figures.

    Parameters
    ----------
    use_latex : bool
        Whether to use LaTeX for text rendering (requires LaTeX installation)
    """
    # Font settings
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    rcParams['font.size'] = 10

    # Figure settings
    rcParams['figure.figsize'] = (3.5, 2.5)  # Single column width
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.05

    # Axes settings
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['axes.linewidth'] = 0.8
    rcParams['axes.grid'] = False

    # Tick settings
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['xtick.minor.width'] = 0.5
    rcParams['ytick.minor.width'] = 0.5
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'

    # Line settings
    rcParams['lines.linewidth'] = 1.0
    rcParams['lines.markersize'] = 4

    # Legend settings
    rcParams['legend.fontsize'] = 9
    rcParams['legend.frameon'] = False
    rcParams['legend.handlelength'] = 1.5

    # LaTeX settings
    rcParams['text.usetex'] = use_latex
    if use_latex:
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'


def get_publication_colors():
    """
    Return a colorblind-friendly color palette for publications.

    Returns
    -------
    dict
        Dictionary of named colors
    """
    return {
        'blue': '#0077BB',
        'red': '#CC3311',
        'green': '#009988',
        'orange': '#EE7733',
        'purple': '#AA3377',
        'cyan': '#33BBEE',
        'grey': '#BBBBBB',
        'black': '#000000',
    }


# =============================================================================
# Common Plot Types
# =============================================================================

def plot_vqe_convergence(history: List[float],
                          exact_energy: float = None,
                          xlabel: str = 'Iteration',
                          ylabel: str = 'Energy (Ha)',
                          title: str = None,
                          figsize: Tuple[float, float] = (3.5, 2.5),
                          save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create publication-quality VQE convergence plot.

    Parameters
    ----------
    history : list
        Energy values at each iteration
    exact_energy : float, optional
        Exact ground state energy for reference line
    xlabel, ylabel : str
        Axis labels
    title : str, optional
        Plot title
    figsize : tuple
        Figure size in inches
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    configure_latex_style()
    colors = get_publication_colors()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot convergence
    iterations = np.arange(len(history))
    ax.plot(iterations, history, color=colors['blue'], linewidth=1.5,
            label='VQE')

    # Add exact energy line
    if exact_energy is not None:
        ax.axhline(y=exact_energy, color=colors['red'], linestyle='--',
                  linewidth=1.0, label=f'Exact = {exact_energy:.4f}')

        # Add chemical accuracy threshold
        chem_acc = exact_energy + 0.0016  # 1 kcal/mol
        ax.axhline(y=chem_acc, color=colors['green'], linestyle=':',
                  linewidth=0.8, label='Chemical accuracy')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    return fig, ax


def plot_comparison_bars(data: Dict[str, float],
                          ylabel: str = 'Value',
                          title: str = None,
                          reference_value: float = None,
                          reference_label: str = 'Reference',
                          figsize: Tuple[float, float] = (4, 3),
                          save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create bar chart comparing different methods.

    Parameters
    ----------
    data : dict
        Dictionary mapping method names to values
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    reference_value : float, optional
        Reference line value
    reference_label : str
        Label for reference line
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    configure_latex_style()
    colors = get_publication_colors()

    fig, ax = plt.subplots(figsize=figsize)

    methods = list(data.keys())
    values = list(data.values())
    x_pos = np.arange(len(methods))

    bars = ax.bar(x_pos, values, color=colors['blue'], width=0.6,
                 edgecolor='black', linewidth=0.5)

    if reference_value is not None:
        ax.axhline(y=reference_value, color=colors['red'], linestyle='--',
                  linewidth=1.0, label=reference_label)
        ax.legend()

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_heatmap(data: np.ndarray,
                 x_labels: List[str] = None,
                 y_labels: List[str] = None,
                 xlabel: str = None,
                 ylabel: str = None,
                 cbar_label: str = None,
                 title: str = None,
                 cmap: str = 'viridis',
                 vmin: float = None,
                 vmax: float = None,
                 figsize: Tuple[float, float] = (4, 3),
                 save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create publication-quality heatmap.

    Parameters
    ----------
    data : np.ndarray
        2D array of values
    x_labels, y_labels : list, optional
        Tick labels
    xlabel, ylabel : str, optional
        Axis labels
    cbar_label : str, optional
        Colorbar label
    title : str, optional
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    configure_latex_style()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='lower')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)

    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_multi_panel(data_list: List[Dict[str, Any]],
                      n_cols: int = 2,
                      figsize: Tuple[float, float] = None,
                      save_path: str = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create multi-panel figure.

    Parameters
    ----------
    data_list : list of dict
        Each dict contains:
        - 'x': x data
        - 'y': y data or list of y data
        - 'labels': list of labels (optional)
        - 'xlabel': x-axis label
        - 'ylabel': y-axis label
        - 'title': panel title
        - 'type': 'line', 'scatter', 'bar'
    n_cols : int
        Number of columns
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes array)
    """
    configure_latex_style()
    colors = get_publication_colors()
    color_list = list(colors.values())

    n_panels = len(data_list)
    n_rows = int(np.ceil(n_panels / n_cols))

    if figsize is None:
        figsize = (3.5 * n_cols, 2.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_panels == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    panel_labels = 'abcdefghijklmnopqrstuvwxyz'

    for idx, data in enumerate(data_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        plot_type = data.get('type', 'line')
        x = data.get('x', np.arange(len(data['y'])))
        y_data = data.get('y', [])
        labels = data.get('labels', [])

        # Handle single or multiple y datasets
        if isinstance(y_data[0], (list, np.ndarray)):
            y_list = y_data
        else:
            y_list = [y_data]

        for i, y in enumerate(y_list):
            color = color_list[i % len(color_list)]
            label = labels[i] if i < len(labels) else None

            if plot_type == 'line':
                ax.plot(x, y, color=color, linewidth=1.0, label=label)
            elif plot_type == 'scatter':
                ax.scatter(x, y, color=color, s=20, label=label)
            elif plot_type == 'bar':
                width = 0.8 / len(y_list)
                offset = (i - len(y_list)/2 + 0.5) * width
                ax.bar(np.array(x) + offset, y, width=width, color=color, label=label)

        ax.set_xlabel(data.get('xlabel', ''))
        ax.set_ylabel(data.get('ylabel', ''))

        # Add panel label
        panel_title = data.get('title', '')
        ax.set_title(f'({panel_labels[idx]}) {panel_title}')

        if labels:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # Remove empty axes
    for idx in range(n_panels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


# =============================================================================
# Quantum-Specific Plots
# =============================================================================

def plot_energy_landscape(energy_fn,
                           param1_range: np.ndarray,
                           param2_range: np.ndarray,
                           param1_label: str = r'$\theta_1$',
                           param2_label: str = r'$\theta_2$',
                           title: str = 'Energy Landscape',
                           figsize: Tuple[float, float] = (4, 3),
                           save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create 2D energy landscape contour plot.

    Parameters
    ----------
    energy_fn : callable
        Function that takes (param1, param2) and returns energy
    param1_range, param2_range : np.ndarray
        Parameter ranges
    param1_label, param2_label : str
        Axis labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    configure_latex_style()

    # Compute landscape
    P1, P2 = np.meshgrid(param1_range, param2_range)
    Z = np.zeros_like(P1)

    for i in range(len(param2_range)):
        for j in range(len(param1_range)):
            Z[i, j] = energy_fn(P1[i, j], P2[i, j])

    fig, ax = plt.subplots(figsize=figsize)

    # Contour plot
    levels = 30
    cs = ax.contourf(P1, P2, Z, levels=levels, cmap='viridis')
    ax.contour(P1, P2, Z, levels=10, colors='white', alpha=0.5, linewidths=0.3)

    cbar = plt.colorbar(cs, ax=ax, shrink=0.9)
    cbar.set_label('Energy')

    # Mark minimum
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    ax.plot(P1[min_idx], P2[min_idx], 'r*', markersize=10)

    ax.set_xlabel(param1_label)
    ax.set_ylabel(param2_label)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_measurement_histogram(counts: Dict[str, int],
                                ideal_probs: Dict[str, float] = None,
                                xlabel: str = 'Measurement Outcome',
                                ylabel: str = 'Counts',
                                title: str = None,
                                figsize: Tuple[float, float] = (4, 3),
                                save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot measurement histogram with optional ideal probabilities.

    Parameters
    ----------
    counts : dict
        Measurement counts {bitstring: count}
    ideal_probs : dict, optional
        Ideal probabilities for comparison
    xlabel, ylabel : str
        Axis labels
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    configure_latex_style()
    colors = get_publication_colors()

    fig, ax = plt.subplots(figsize=figsize)

    # Sort bitstrings
    bitstrings = sorted(counts.keys())
    measured_counts = [counts[b] for b in bitstrings]
    total_counts = sum(measured_counts)

    x = np.arange(len(bitstrings))
    width = 0.35 if ideal_probs else 0.7

    # Measured counts
    bars = ax.bar(x if not ideal_probs else x - width/2,
                 measured_counts, width, color=colors['blue'],
                 label='Measured', edgecolor='black', linewidth=0.5)

    # Ideal probabilities (scaled to counts)
    if ideal_probs:
        ideal_counts = [ideal_probs.get(b, 0) * total_counts for b in bitstrings]
        ax.bar(x + width/2, ideal_counts, width, color=colors['red'],
               label='Ideal', edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.legend()

    ax.set_xticks(x)
    ax.set_xticklabels(bitstrings, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


# =============================================================================
# Demonstration
# =============================================================================

def demo():
    """Demonstrate figure generation utilities."""
    print("=" * 60)
    print("Publication Figure Generation Demo")
    print("=" * 60)

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    # Demo 1: VQE convergence
    print("\n1. VQE Convergence Plot")
    print("-" * 40)

    np.random.seed(42)
    exact_energy = -1.137
    history = [0.5]
    for i in range(49):
        next_val = history[-1] - 0.03 + 0.02 * np.random.randn()
        history.append(max(next_val, exact_energy + 0.001))

    fig1, _ = plot_vqe_convergence(
        history,
        exact_energy=exact_energy,
        title='VQE Convergence for H$_2$',
        save_path=str(output_dir / 'vqe_convergence.pdf')
    )
    plt.close(fig1)
    print("Created: figures/vqe_convergence.pdf")

    # Demo 2: Method comparison
    print("\n2. Method Comparison Bar Chart")
    print("-" * 40)

    methods_data = {
        'VQE': -1.132,
        'ADAPT': -1.135,
        'UCCSD': -1.136,
        'Exact': -1.137
    }

    fig2, _ = plot_comparison_bars(
        methods_data,
        ylabel='Energy (Ha)',
        title='Ground State Energy Comparison',
        save_path=str(output_dir / 'method_comparison.pdf')
    )
    plt.close(fig2)
    print("Created: figures/method_comparison.pdf")

    # Demo 3: Heatmap
    print("\n3. Parameter Heatmap")
    print("-" * 40)

    data = np.random.rand(10, 10)
    fig3, _ = plot_heatmap(
        data,
        xlabel='Parameter 1',
        ylabel='Parameter 2',
        cbar_label='Energy',
        title='Energy Landscape',
        save_path=str(output_dir / 'heatmap.pdf')
    )
    plt.close(fig3)
    print("Created: figures/heatmap.pdf")

    # Demo 4: Multi-panel figure
    print("\n4. Multi-Panel Figure")
    print("-" * 40)

    x = np.linspace(0, 2*np.pi, 50)
    panel_data = [
        {
            'x': x,
            'y': [np.sin(x), np.cos(x)],
            'labels': ['sin', 'cos'],
            'xlabel': 'x',
            'ylabel': 'y',
            'title': 'Trigonometric',
            'type': 'line'
        },
        {
            'x': x,
            'y': np.exp(-x/3) * np.sin(3*x),
            'xlabel': 'x',
            'ylabel': 'y',
            'title': 'Damped oscillation',
            'type': 'line'
        },
        {
            'x': ['A', 'B', 'C', 'D'],
            'y': [0.3, 0.5, 0.2, 0.4],
            'xlabel': 'State',
            'ylabel': 'Probability',
            'title': 'Distribution',
            'type': 'bar'
        },
        {
            'x': np.random.randn(30),
            'y': np.random.randn(30),
            'xlabel': 'x',
            'ylabel': 'y',
            'title': 'Scatter',
            'type': 'scatter'
        }
    ]

    fig4, _ = plot_multi_panel(
        panel_data,
        n_cols=2,
        save_path=str(output_dir / 'multi_panel.pdf')
    )
    plt.close(fig4)
    print("Created: figures/multi_panel.pdf")

    # Demo 5: Measurement histogram
    print("\n5. Measurement Histogram")
    print("-" * 40)

    counts = {'00': 450, '01': 52, '10': 48, '11': 450}
    ideal_probs = {'00': 0.5, '01': 0.0, '10': 0.0, '11': 0.5}

    fig5, _ = plot_measurement_histogram(
        counts,
        ideal_probs=ideal_probs,
        title='Bell State Measurement',
        save_path=str(output_dir / 'histogram.pdf')
    )
    plt.close(fig5)
    print("Created: figures/histogram.pdf")

    print("\n" + "=" * 60)
    print("Demo Complete! Figures saved to 'figures/' directory")
    print("=" * 60)


if __name__ == "__main__":
    demo()
