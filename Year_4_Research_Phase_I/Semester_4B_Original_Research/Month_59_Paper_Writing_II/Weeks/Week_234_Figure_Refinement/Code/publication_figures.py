#!/usr/bin/env python3
"""
Publication-Quality Figure Templates for Quantum Science Papers
================================================================

This module provides templates and utilities for creating publication-ready
figures suitable for journals like Physical Review Letters, Nature Physics,
and Science.

Author: Quantum Engineering PhD Curriculum
Week 234: Figure Refinement | Month 59 | Year 4

Requirements:
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy import stats
from typing import Tuple, List, Optional, Dict
import warnings

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

def set_publication_style(journal: str = 'prl') -> None:
    """
    Configure matplotlib for publication-quality output.

    Parameters
    ----------
    journal : str
        Target journal style: 'prl', 'nature', or 'science'

    Examples
    --------
    >>> set_publication_style('prl')
    >>> fig, ax = plt.subplots()
    >>> # Now all subsequent plots use PRL styling
    """
    # Common settings for all journals
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    rcParams['mathtext.fontset'] = 'dejavusans'

    # High-quality output
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 600
    rcParams['savefig.format'] = 'pdf'
    rcParams['pdf.fonttype'] = 42  # TrueType fonts (better compatibility)

    # Clean axis styling
    rcParams['axes.linewidth'] = 0.8
    rcParams['axes.labelpad'] = 4
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

    # Tick styling
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['xtick.major.size'] = 4
    rcParams['ytick.major.size'] = 4
    rcParams['xtick.minor.visible'] = True
    rcParams['ytick.minor.visible'] = True
    rcParams['xtick.minor.size'] = 2
    rcParams['ytick.minor.size'] = 2

    # Legend styling
    rcParams['legend.frameon'] = False
    rcParams['legend.fontsize'] = 8

    # Line styling
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 5

    # Journal-specific sizes
    if journal.lower() == 'prl':
        # Physical Review Letters: single column = 3.4 in, double = 7.0 in
        rcParams['figure.figsize'] = (3.4, 2.5)
        rcParams['font.size'] = 8
        rcParams['axes.labelsize'] = 9
        rcParams['axes.titlesize'] = 9
    elif journal.lower() == 'nature':
        # Nature: single column = 89 mm = 3.5 in
        rcParams['figure.figsize'] = (3.5, 2.5)
        rcParams['font.size'] = 7
        rcParams['axes.labelsize'] = 8
        rcParams['axes.titlesize'] = 8
    elif journal.lower() == 'science':
        # Science: single column = 55 mm = 2.17 in
        rcParams['figure.figsize'] = (2.17, 2.0)
        rcParams['font.size'] = 6
        rcParams['axes.labelsize'] = 7
        rcParams['axes.titlesize'] = 7


# =============================================================================
# COLORBLIND-SAFE PALETTES
# =============================================================================

# Wong color palette - optimized for colorblindness
WONG_COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'yellow': '#F0E442',
    'sky_blue': '#56B4E9',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000',
}

# Ordered list for cycling
WONG_CYCLE = [
    WONG_COLORS['blue'],
    WONG_COLORS['orange'],
    WONG_COLORS['green'],
    WONG_COLORS['vermillion'],
    WONG_COLORS['purple'],
    WONG_COLORS['sky_blue'],
]

# High-contrast palette for 3 categories
CONTRAST_3 = ['#0072B2', '#E69F00', '#CC79A7']

# For diverging colormaps (positive/negative)
DIVERGING = {
    'positive': '#D55E00',  # Vermillion
    'negative': '#0072B2',  # Blue
    'neutral': '#999999',   # Gray
}


def get_colors(n: int) -> List[str]:
    """
    Get n colorblind-safe colors from the Wong palette.

    Parameters
    ----------
    n : int
        Number of colors needed

    Returns
    -------
    List[str]
        List of hex color codes
    """
    if n <= len(WONG_CYCLE):
        return WONG_CYCLE[:n]
    else:
        # Cycle through if more needed
        return [WONG_CYCLE[i % len(WONG_CYCLE)] for i in range(n)]


# =============================================================================
# FIGURE TEMPLATES
# =============================================================================

def create_single_panel_figure(
    width: float = 3.4,
    height: float = 2.5,
    journal: str = 'prl'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a single-panel figure with publication styling.

    Parameters
    ----------
    width : float
        Figure width in inches
    height : float
        Figure height in inches
    journal : str
        Target journal for styling

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects
    """
    set_publication_style(journal)
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def create_multi_panel_figure(
    nrows: int = 1,
    ncols: int = 2,
    width: float = 7.0,
    height: float = 2.5,
    journal: str = 'prl',
    share_x: bool = False,
    share_y: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a multi-panel figure with consistent styling.

    Parameters
    ----------
    nrows, ncols : int
        Number of rows and columns
    width, height : float
        Figure dimensions in inches
    journal : str
        Target journal styling
    share_x, share_y : bool
        Whether to share axes

    Returns
    -------
    fig, axes : tuple
        Figure and array of axes
    """
    set_publication_style(journal)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width, height),
        sharex=share_x,
        sharey=share_y,
        constrained_layout=True
    )
    return fig, axes


def add_panel_labels(
    axes: np.ndarray,
    labels: Optional[List[str]] = None,
    x: float = -0.15,
    y: float = 1.1,
    fontweight: str = 'bold',
    fontsize: int = 10,
) -> None:
    """
    Add (a), (b), (c) labels to multi-panel figure.

    Parameters
    ----------
    axes : np.ndarray
        Array of axes objects
    labels : List[str], optional
        Custom labels; defaults to (a), (b), ...
    x, y : float
        Position in axes coordinates
    fontweight : str
        Font weight for labels
    fontsize : int
        Font size for labels
    """
    if labels is None:
        labels = [f'({chr(97 + i)})' for i in range(axes.size)]

    flat_axes = np.array(axes).flatten()
    for ax, label in zip(flat_axes, labels):
        ax.text(x, y, label, transform=ax.transAxes,
                fontweight=fontweight, fontsize=fontsize,
                verticalalignment='top')


# =============================================================================
# DATA VISUALIZATION FUNCTIONS
# =============================================================================

def plot_with_error_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    color: str = WONG_COLORS['blue'],
    label: Optional[str] = None,
    alpha: float = 0.3,
    line_kwargs: Optional[Dict] = None,
) -> None:
    """
    Plot data with shaded error band (good for dense data).

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    x, y, yerr : np.ndarray
        Data and error values
    color : str
        Color for line and fill
    label : str, optional
        Legend label
    alpha : float
        Transparency for error band
    line_kwargs : dict, optional
        Additional keyword arguments for the line
    """
    if line_kwargs is None:
        line_kwargs = {}

    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)
    ax.plot(x, y, color=color, label=label, **line_kwargs)


def plot_with_error_bars(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    xerr: Optional[np.ndarray] = None,
    color: str = WONG_COLORS['blue'],
    marker: str = 'o',
    label: Optional[str] = None,
    capsize: float = 2,
    markersize: float = 5,
) -> None:
    """
    Plot data with error bars (good for sparse data).

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    x, y : np.ndarray
        Data values
    yerr : np.ndarray
        Y-direction error
    xerr : np.ndarray, optional
        X-direction error
    color : str
        Color for points and bars
    marker : str
        Marker style
    label : str, optional
        Legend label
    capsize : float
        Error bar cap size
    markersize : float
        Marker size
    """
    ax.errorbar(
        x, y, yerr=yerr, xerr=xerr,
        fmt=marker, color=color, label=label,
        capsize=capsize, markersize=markersize,
        markeredgecolor='white', markeredgewidth=0.5,
        elinewidth=0.8, capthick=0.8
    )


def plot_theory_comparison(
    ax: plt.Axes,
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    yerr_exp: np.ndarray,
    x_theory: np.ndarray,
    y_theory: np.ndarray,
    exp_color: str = WONG_COLORS['blue'],
    theory_color: str = WONG_COLORS['black'],
    exp_label: str = 'Experiment',
    theory_label: str = 'Theory',
) -> None:
    """
    Plot experimental data with theoretical curve overlay.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    x_exp, y_exp, yerr_exp : np.ndarray
        Experimental data with errors
    x_theory, y_theory : np.ndarray
        Theoretical curve
    exp_color, theory_color : str
        Colors for experiment and theory
    exp_label, theory_label : str
        Legend labels
    """
    # Plot theory first (behind data)
    ax.plot(x_theory, y_theory, color=theory_color,
            linestyle='-', linewidth=1.0, label=theory_label)

    # Plot experimental data on top
    plot_with_error_bars(
        ax, x_exp, y_exp, yerr_exp,
        color=exp_color, label=exp_label
    )


# =============================================================================
# QUANTUM-SPECIFIC VISUALIZATIONS
# =============================================================================

def plot_bloch_vector_trajectory(
    ax: plt.Axes,
    theta: np.ndarray,
    phi: np.ndarray,
    color: str = WONG_COLORS['blue'],
    show_sphere: bool = True,
) -> None:
    """
    Plot trajectory on Bloch sphere (2D projection).

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    theta, phi : np.ndarray
        Bloch angles in radians
    color : str
        Color for trajectory
    show_sphere : bool
        Whether to show sphere outline
    """
    # Convert to Cartesian (x-z projection)
    x = np.sin(theta) * np.cos(phi)
    z = np.cos(theta)

    if show_sphere:
        # Draw sphere outline
        circle = plt.Circle((0, 0), 1, fill=False,
                            color='gray', linestyle='--', linewidth=0.5)
        ax.add_patch(circle)

        # Draw axes
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')

    # Plot trajectory
    ax.plot(x, z, color=color, linewidth=1.5)

    # Mark start and end
    ax.scatter([x[0]], [z[0]], color=color, s=50, marker='o',
               zorder=5, edgecolor='white')
    ax.scatter([x[-1]], [z[-1]], color=color, s=50, marker='s',
               zorder=5, edgecolor='white')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\langle X \rangle$')
    ax.set_ylabel(r'$\langle Z \rangle$')


def plot_energy_levels(
    ax: plt.Axes,
    energies: List[float],
    labels: Optional[List[str]] = None,
    width: float = 0.3,
    transitions: Optional[List[Tuple[int, int]]] = None,
    transition_colors: Optional[List[str]] = None,
) -> None:
    """
    Plot quantum energy level diagram.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    energies : List[float]
        List of energy values
    labels : List[str], optional
        State labels
    width : float
        Width of energy level lines
    transitions : List[Tuple[int, int]], optional
        Pairs of (lower, upper) state indices for transitions
    transition_colors : List[str], optional
        Colors for transition arrows
    """
    for i, E in enumerate(energies):
        ax.hlines(E, 0.5 - width/2, 0.5 + width/2,
                 color='black', linewidth=2)
        if labels is not None:
            ax.text(0.5 + width/2 + 0.05, E, labels[i],
                   verticalalignment='center')

    if transitions is not None:
        if transition_colors is None:
            transition_colors = get_colors(len(transitions))

        for (lower, upper), color in zip(transitions, transition_colors):
            ax.annotate(
                '', xy=(0.5, energies[upper]),
                xytext=(0.5, energies[lower]),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5)
            )

    ax.set_xlim(0, 1)
    ax.set_ylabel('Energy (arb. units)')
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_pulse_sequence(
    ax: plt.Axes,
    times: np.ndarray,
    pulses: List[Dict],
    colors: Optional[List[str]] = None,
) -> None:
    """
    Plot a quantum control pulse sequence.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    times : np.ndarray
        Time array
    pulses : List[Dict]
        List of pulse dictionaries with keys:
        - 'start': start time
        - 'duration': pulse duration
        - 'amplitude': pulse amplitude
        - 'label': optional label
    colors : List[str], optional
        Colors for different pulse types
    """
    if colors is None:
        colors = get_colors(len(pulses))

    for pulse, color in zip(pulses, colors):
        t_start = pulse['start']
        t_end = pulse['start'] + pulse['duration']
        amp = pulse['amplitude']

        # Draw pulse
        ax.fill_between(
            [t_start, t_end], [0, 0], [amp, amp],
            color=color, alpha=0.7
        )
        ax.plot([t_start, t_start, t_end, t_end],
               [0, amp, amp, 0], color=color, linewidth=1)

        # Add label if present
        if 'label' in pulse:
            ax.text((t_start + t_end) / 2, amp * 1.1, pulse['label'],
                   ha='center', fontsize=7)

    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')


# =============================================================================
# HEATMAP AND 2D VISUALIZATIONS
# =============================================================================

def plot_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    x_values: Optional[np.ndarray] = None,
    y_values: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xlabel: str = '',
    ylabel: str = '',
    cbar_label: str = '',
    add_colorbar: bool = True,
) -> plt.cm.ScalarMappable:
    """
    Create a publication-quality heatmap.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    data : np.ndarray
        2D data array
    x_values, y_values : np.ndarray, optional
        Axis values
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    xlabel, ylabel, cbar_label : str
        Axis labels
    add_colorbar : bool
        Whether to add colorbar

    Returns
    -------
    im : ScalarMappable
        The image object (for colorbar customization)
    """
    if x_values is None:
        x_values = np.arange(data.shape[1])
    if y_values is None:
        y_values = np.arange(data.shape[0])

    im = ax.pcolormesh(x_values, y_values, data,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, label=cbar_label)
        cbar.ax.tick_params(labelsize=7)

    return im


def plot_density_matrix(
    fig: plt.Figure,
    rho: np.ndarray,
    state_labels: Optional[List[str]] = None,
    show_phase: bool = True,
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plot density matrix magnitude and phase.

    Parameters
    ----------
    fig : plt.Figure
        Figure to plot on
    rho : np.ndarray
        Complex density matrix
    state_labels : List[str], optional
        Labels for basis states
    show_phase : bool
        Whether to show phase subplot

    Returns
    -------
    ax_mag, ax_phase : tuple
        Axes for magnitude and phase plots
    """
    n = rho.shape[0]

    if state_labels is None:
        state_labels = [f'$|{i}\\rangle$' for i in range(n)]

    if show_phase:
        ax_mag = fig.add_subplot(121)
        ax_phase = fig.add_subplot(122)
    else:
        ax_mag = fig.add_subplot(111)
        ax_phase = None

    # Magnitude
    im_mag = ax_mag.imshow(np.abs(rho), cmap='Blues', vmin=0)
    ax_mag.set_xticks(range(n))
    ax_mag.set_yticks(range(n))
    ax_mag.set_xticklabels(state_labels)
    ax_mag.set_yticklabels(state_labels)
    ax_mag.set_title('$|\\rho_{ij}|$')
    plt.colorbar(im_mag, ax=ax_mag, shrink=0.8)

    # Add value annotations
    for i in range(n):
        for j in range(n):
            val = np.abs(rho[i, j])
            if val > 0.05:  # Only show significant values
                ax_mag.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color='white' if val > 0.5 else 'black', fontsize=7)

    if show_phase:
        # Phase (only where magnitude is significant)
        phase = np.angle(rho)
        phase_masked = np.ma.masked_where(np.abs(rho) < 0.05, phase)

        im_phase = ax_phase.imshow(phase_masked, cmap='twilight',
                                   vmin=-np.pi, vmax=np.pi)
        ax_phase.set_xticks(range(n))
        ax_phase.set_yticks(range(n))
        ax_phase.set_xticklabels(state_labels)
        ax_phase.set_yticklabels(state_labels)
        ax_phase.set_title(r'$\arg(\rho_{ij})$')
        cbar = plt.colorbar(im_phase, ax=ax_phase, shrink=0.8)
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])

    return ax_mag, ax_phase


# =============================================================================
# SCHEMATIC DRAWING UTILITIES
# =============================================================================

def draw_qubit_symbol(
    ax: plt.Axes,
    x: float,
    y: float,
    size: float = 0.1,
    color: str = WONG_COLORS['blue'],
    label: Optional[str] = None,
) -> None:
    """
    Draw a qubit symbol (circle with ground state indicator).

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on
    x, y : float
        Center position
    size : float
        Size of the symbol
    color : str
        Color for the symbol
    label : str, optional
        Label to display
    """
    # Main circle
    circle = Circle((x, y), size, fill=False,
                   edgecolor=color, linewidth=1.5)
    ax.add_patch(circle)

    # Ground state indicator (horizontal line)
    ax.plot([x - size*0.6, x + size*0.6], [y - size*0.5, y - size*0.5],
           color=color, linewidth=1.5)

    # Excited state indicator (dot)
    ax.scatter([x], [y + size*0.4], color=color, s=10, zorder=5)

    if label is not None:
        ax.text(x, y - size*1.4, label, ha='center', fontsize=8)


def draw_coupling(
    ax: plt.Axes,
    x1: float, y1: float,
    x2: float, y2: float,
    style: str = 'capacitive',
    color: str = 'gray',
) -> None:
    """
    Draw coupling between qubits.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on
    x1, y1, x2, y2 : float
        Start and end positions
    style : str
        'capacitive' or 'inductive'
    color : str
        Line color
    """
    if style == 'capacitive':
        # Two parallel lines for capacitor
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.plot([x1, mid_x - 0.03], [y1, mid_y], color=color, linewidth=1)
        ax.plot([mid_x + 0.03, x2], [mid_y, y2], color=color, linewidth=1)
        ax.plot([mid_x - 0.03, mid_x - 0.03],
               [mid_y - 0.05, mid_y + 0.05], color=color, linewidth=1.5)
        ax.plot([mid_x + 0.03, mid_x + 0.03],
               [mid_y - 0.05, mid_y + 0.05], color=color, linewidth=1.5)
    else:
        # Wavy line for inductor
        t = np.linspace(0, 1, 100)
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t + 0.02 * np.sin(20 * np.pi * t)
        ax.plot(x, y, color=color, linewidth=1)


# =============================================================================
# EXAMPLE FIGURES
# =============================================================================

def example_gate_fidelity_figure() -> plt.Figure:
    """
    Example: Gate fidelity vs drive amplitude with theory comparison.

    This demonstrates typical quantum computing experimental results.
    """
    set_publication_style('prl')

    # Simulated experimental data
    np.random.seed(42)
    drive_amp = np.linspace(0.1, 1.0, 15)

    # Model: Fidelity peaks then decreases due to leakage
    fidelity_theory = 0.999 * np.exp(-((drive_amp - 0.6) / 0.3)**2)
    fidelity_exp = fidelity_theory + 0.005 * np.random.randn(len(drive_amp))
    fidelity_err = 0.002 + 0.003 * np.random.rand(len(drive_amp))

    # Dense theory curve
    drive_theory = np.linspace(0.05, 1.05, 100)
    fidelity_theory_smooth = 0.999 * np.exp(-((drive_theory - 0.6) / 0.3)**2)

    # Create figure
    fig, ax = create_single_panel_figure(width=3.4, height=2.5)

    # Plot theory comparison
    plot_theory_comparison(
        ax, drive_amp, fidelity_exp, fidelity_err,
        drive_theory, fidelity_theory_smooth,
        exp_label='Data', theory_label='Simulation'
    )

    # Mark optimal point
    ax.axvline(0.6, color='gray', linestyle=':', linewidth=0.8)
    ax.text(0.62, 0.85, 'Optimal', fontsize=7, color='gray')

    # Labels
    ax.set_xlabel(r'Drive amplitude, $\Omega/2\pi$ (MHz)')
    ax.set_ylabel('Gate fidelity')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.75, 1.02)

    # Legend
    ax.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    return fig


def example_multi_panel_figure() -> plt.Figure:
    """
    Example: Multi-panel figure with coherence measurements.

    This demonstrates proper multi-panel layout and labeling.
    """
    set_publication_style('prl')

    # Create 2x2 figure
    fig, axes = create_multi_panel_figure(nrows=2, ncols=2,
                                          width=7.0, height=5.0)

    # Simulated data
    np.random.seed(42)

    # Panel (a): T1 decay
    t = np.linspace(0, 100, 50)
    T1 = 45  # us
    p_excited = np.exp(-t / T1)
    p_data = p_excited + 0.02 * np.random.randn(len(t))
    p_err = 0.02 * np.ones_like(t)

    ax = axes[0, 0]
    plot_with_error_bars(ax, t, p_data, p_err, label='Data')
    ax.plot(t, p_excited, color=WONG_COLORS['black'],
            linestyle='--', label=f'$T_1 = {T1}$ μs')
    ax.set_xlabel('Delay time (μs)')
    ax.set_ylabel(r'$P_{|1\rangle}$')
    ax.legend(fontsize=6, loc='upper right')

    # Panel (b): T2 Ramsey
    t2 = np.linspace(0, 50, 40)
    T2 = 35  # us
    detuning = 0.5  # MHz
    ramsey = 0.5 * (1 + np.exp(-t2 / T2) * np.cos(2 * np.pi * detuning * t2))
    ramsey_data = ramsey + 0.03 * np.random.randn(len(t2))

    ax = axes[0, 1]
    plot_with_error_bars(ax, t2, ramsey_data, 0.03 * np.ones_like(t2))
    ax.plot(t2, ramsey, color=WONG_COLORS['black'], linestyle='--',
            label=f'$T_2^* = {T2}$ μs')
    ax.set_xlabel('Delay time (μs)')
    ax.set_ylabel(r'$\langle X \rangle$')
    ax.legend(fontsize=6, loc='upper right')

    # Panel (c): Spectroscopy
    freq = np.linspace(4.8, 5.2, 100)
    f0 = 5.0  # GHz
    linewidth = 0.05  # GHz
    spectrum = 0.8 / (1 + ((freq - f0) / (linewidth/2))**2)
    spectrum_data = spectrum + 0.05 * np.random.randn(len(freq))

    ax = axes[1, 0]
    ax.plot(freq, spectrum_data, color=WONG_COLORS['blue'], linewidth=1)
    ax.plot(freq, spectrum, color=WONG_COLORS['black'], linestyle='--')
    ax.axvline(f0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Response (arb. u.)')

    # Panel (d): Histogram of repeated measurements
    ax = axes[1, 1]
    measurements_0 = np.random.normal(0.1, 0.05, 1000)
    measurements_1 = np.random.normal(0.8, 0.05, 1000)

    ax.hist(measurements_0, bins=30, alpha=0.7, color=WONG_COLORS['blue'],
            label='$|0\\rangle$', density=True)
    ax.hist(measurements_1, bins=30, alpha=0.7, color=WONG_COLORS['orange'],
            label='$|1\\rangle$', density=True)
    ax.set_xlabel('Measurement signal (V)')
    ax.set_ylabel('Probability density')
    ax.legend(fontsize=6)

    # Add panel labels
    add_panel_labels(axes)

    plt.tight_layout()
    return fig


def example_schematic_figure() -> plt.Figure:
    """
    Example: Experimental setup schematic.

    This demonstrates schematic drawing for methodology figures.
    """
    set_publication_style('prl')

    fig, ax = create_single_panel_figure(width=3.4, height=2.0)

    # Draw qubit chain
    qubit_positions = [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5)]

    for i, (x, y) in enumerate(qubit_positions):
        draw_qubit_symbol(ax, x, y, size=0.06,
                         color=WONG_COLORS['blue'],
                         label=f'$Q_{i+1}$')

    # Draw couplings
    for i in range(len(qubit_positions) - 1):
        x1, y1 = qubit_positions[i]
        x2, y2 = qubit_positions[i + 1]
        draw_coupling(ax, x1 + 0.08, y1, x2 - 0.08, y2,
                     style='capacitive', color='gray')

    # Add readout resonators
    for i, (x, y) in enumerate(qubit_positions):
        ax.annotate('', xy=(x, y + 0.15), xytext=(x, y + 0.08),
                   arrowprops=dict(arrowstyle='-', color='gray'))
        ax.text(x, y + 0.2, 'R', fontsize=7, ha='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add title
    ax.text(0.5, 0.75, 'Four-qubit linear chain', ha='center', fontsize=9)

    return fig


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def save_figure(
    fig: plt.Figure,
    filename: str,
    formats: List[str] = ['pdf', 'png'],
    dpi: int = 600,
    transparent: bool = False,
) -> None:
    """
    Save figure in multiple formats for publication.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    formats : List[str]
        List of formats to save
    dpi : int
        Resolution for raster formats
    transparent : bool
        Whether to use transparent background
    """
    for fmt in formats:
        fig.savefig(
            f'{filename}.{fmt}',
            format=fmt,
            dpi=dpi if fmt in ['png', 'tiff'] else None,
            bbox_inches='tight',
            transparent=transparent,
            facecolor='white' if not transparent else 'none',
        )
        print(f'Saved: {filename}.{fmt}')


def test_colorblind_safety(fig: plt.Figure) -> None:
    """
    Print reminder to test colorblind safety.

    Parameters
    ----------
    fig : plt.Figure
        Figure to test
    """
    print("\n=== COLORBLIND SAFETY REMINDER ===")
    print("Please verify your figure using:")
    print("1. Coblis: https://www.color-blindness.com/coblis-color-blindness-simulator/")
    print("2. Viz Palette: https://projects.susielu.com/viz-palette")
    print("3. Convert to grayscale to verify readability")
    print("===================================\n")


# =============================================================================
# MAIN: DEMO EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("Publication Figure Templates - Demo")
    print("=" * 50)

    # Generate example figures
    print("\n1. Generating gate fidelity figure...")
    fig1 = example_gate_fidelity_figure()
    save_figure(fig1, 'example_gate_fidelity', formats=['pdf', 'png'])

    print("\n2. Generating multi-panel figure...")
    fig2 = example_multi_panel_figure()
    save_figure(fig2, 'example_multi_panel', formats=['pdf', 'png'])

    print("\n3. Generating schematic figure...")
    fig3 = example_schematic_figure()
    save_figure(fig3, 'example_schematic', formats=['pdf', 'png'])

    # Colorblind reminder
    test_colorblind_safety(fig1)

    print("\nAll example figures generated successfully!")
    print("Review the output files and adapt templates for your data.")

    plt.show()
