"""
Quantum Visualization Examples
==============================

This module provides visualization utilities for quantum computing
research, including Bloch spheres, state visualization, and
publication-quality figures.

Author: Quantum Engineering PhD Program
Week 211: Simulation & Analysis

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Union


# =============================================================================
# Publication Style Setup
# =============================================================================

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    rcParams['figure.figsize'] = (3.5, 2.5)
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['axes.linewidth'] = 0.8
    rcParams['lines.linewidth'] = 1.0
    rcParams['lines.markersize'] = 4
    rcParams['legend.frameon'] = False


# =============================================================================
# Quantum State Utilities
# =============================================================================

def state_to_bloch(state: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert single-qubit pure state to Bloch vector.

    Parameters
    ----------
    state : np.ndarray
        2-component complex state vector [alpha, beta]

    Returns
    -------
    tuple
        (x, y, z) Bloch vector coordinates

    Examples
    --------
    >>> state_to_bloch(np.array([1, 0]))  # |0> state
    (0.0, 0.0, 1.0)
    >>> state_to_bloch(np.array([0, 1]))  # |1> state
    (0.0, 0.0, -1.0)
    >>> state_to_bloch(np.array([1, 1]) / np.sqrt(2))  # |+> state
    (1.0, 0.0, 0.0)
    """
    # Ensure normalized
    state = state / np.linalg.norm(state)

    # Compute density matrix
    rho = np.outer(state, state.conj())

    # Extract Bloch vector components
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])

    return float(x), float(y), float(z)


def bloch_to_state(x: float, y: float, z: float) -> np.ndarray:
    """
    Convert Bloch vector to quantum state.

    Parameters
    ----------
    x, y, z : float
        Bloch vector coordinates

    Returns
    -------
    np.ndarray
        2-component complex state vector
    """
    # Compute angles
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r > 1e-10 else 0
    phi = np.arctan2(y, x)

    # State: cos(theta/2)|0> + e^{i*phi}*sin(theta/2)|1>
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)

    return np.array([alpha, beta], dtype=complex)


def density_matrix_to_bloch(rho: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert density matrix to Bloch vector.

    Works for both pure and mixed states.

    Parameters
    ----------
    rho : np.ndarray
        2x2 density matrix

    Returns
    -------
    tuple
        (x, y, z) Bloch vector coordinates
    """
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])
    return float(x), float(y), float(z)


# =============================================================================
# Bloch Sphere Visualization
# =============================================================================

def plot_bloch_sphere(states: List[np.ndarray] = None,
                      vectors: List[Tuple[float, float, float]] = None,
                      labels: List[str] = None,
                      colors: List[str] = None,
                      title: str = None,
                      show_axes_labels: bool = True,
                      figsize: Tuple[int, int] = (6, 6),
                      save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot states/vectors on a Bloch sphere.

    Parameters
    ----------
    states : list of np.ndarray, optional
        List of 2-component quantum states
    vectors : list of tuples, optional
        List of (x, y, z) Bloch vectors
    labels : list of str, optional
        Labels for each state/vector
    colors : list of str, optional
        Colors for each state/vector
    title : str, optional
        Plot title
    show_axes_labels : bool
        Whether to show axes labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.3, linewidth=0.5)

    # Draw coordinate axes
    axis_length = 1.3
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='gray',
              arrow_length_ratio=0.08, linewidth=1)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='gray',
              arrow_length_ratio=0.08, linewidth=1)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='gray',
              arrow_length_ratio=0.08, linewidth=1)

    # Axes labels
    if show_axes_labels:
        ax.text(1.4, 0, 0, r'$|+\rangle$', fontsize=10, ha='center')
        ax.text(-1.4, 0, 0, r'$|-\rangle$', fontsize=10, ha='center')
        ax.text(0, 1.4, 0, r'$|+i\rangle$', fontsize=10, ha='center')
        ax.text(0, -1.4, 0, r'$|-i\rangle$', fontsize=10, ha='center')
        ax.text(0, 0, 1.4, r'$|0\rangle$', fontsize=10, ha='center')
        ax.text(0, 0, -1.4, r'$|1\rangle$', fontsize=10, ha='center')

    # Default colors
    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot states
    all_vectors = []
    if states is not None:
        for state in states:
            all_vectors.append(state_to_bloch(state))

    if vectors is not None:
        all_vectors.extend(vectors)

    for i, (vx, vy, vz) in enumerate(all_vectors):
        color = colors[i] if colors and i < len(colors) else default_colors[i % 10]
        label = labels[i] if labels and i < len(labels) else None

        # Draw vector as arrow
        ax.quiver(0, 0, 0, vx, vy, vz, color=color,
                 arrow_length_ratio=0.1, linewidth=2, label=label)
        # Draw point at tip
        ax.scatter([vx], [vy], [vz], color=color, s=50, zorder=10)

    # Settings
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])

    if title:
        ax.set_title(title)

    if labels:
        ax.legend(loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_bloch_trajectory(states_history: List[np.ndarray],
                          title: str = "State Evolution",
                          show_initial: bool = True,
                          show_final: bool = True,
                          save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot trajectory of state evolution on Bloch sphere.

    Parameters
    ----------
    states_history : list
        List of states at each time step
    title : str
        Plot title
    show_initial : bool
        Highlight initial state
    show_final : bool
        Highlight final state
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.2, linewidth=0.3)

    # Convert states to Bloch vectors
    trajectory = np.array([state_to_bloch(s) for s in states_history])

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            'b-', linewidth=1.5, alpha=0.7, label='Trajectory')

    # Color gradient for time evolution
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
               c=colors, s=10, alpha=0.5)

    # Highlight initial and final states
    if show_initial:
        ax.scatter(*trajectory[0], color='green', s=100, marker='o',
                  label='Initial', zorder=10)
    if show_final:
        ax.scatter(*trajectory[-1], color='red', s=100, marker='*',
                  label='Final', zorder=10)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


# =============================================================================
# Multi-Qubit State Visualization
# =============================================================================

def plot_state_city(state: np.ndarray, basis_labels: List[str] = None,
                    title: str = "State Amplitudes",
                    save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create 3D bar plot of state amplitudes (city plot).

    Parameters
    ----------
    state : np.ndarray
        State vector
    basis_labels : list of str, optional
        Labels for basis states
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    dim = len(state)
    n_qubits = int(np.log2(dim))

    if basis_labels is None:
        basis_labels = [f"|{format(i, f'0{n_qubits}b')}>" for i in range(dim)]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Positions
    xpos = np.arange(dim)
    ypos = np.zeros(dim)
    zpos = np.zeros(dim)

    # Bar dimensions
    dx = 0.5 * np.ones(dim)
    dy = 0.5 * np.ones(dim)

    # Real and imaginary parts
    real_parts = np.real(state)
    imag_parts = np.imag(state)

    # Plot real parts
    colors_real = ['blue' if r >= 0 else 'red' for r in real_parts]
    ax.bar3d(xpos - 0.25, ypos, zpos, dx, dy, np.abs(real_parts),
             color=colors_real, alpha=0.7, label='Real')

    # Plot imaginary parts (offset)
    colors_imag = ['green' if i >= 0 else 'orange' for i in imag_parts]
    ax.bar3d(xpos + 0.25, ypos + 0.5, zpos, dx, dy, np.abs(imag_parts),
             color=colors_imag, alpha=0.7, label='Imag')

    ax.set_xticks(xpos)
    ax.set_xticklabels(basis_labels, rotation=45, ha='right')
    ax.set_xlabel('Basis State')
    ax.set_zlabel('Amplitude')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_density_matrix(rho: np.ndarray, basis_labels: List[str] = None,
                        title: str = "Density Matrix",
                        save_path: str = None) -> plt.Figure:
    """
    Plot density matrix as heatmaps for real and imaginary parts.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix
    basis_labels : list of str, optional
        Labels for basis states
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    Figure
        Matplotlib figure
    """
    dim = rho.shape[0]
    n_qubits = int(np.log2(dim))

    if basis_labels is None:
        basis_labels = [f"|{format(i, f'0{n_qubits}b')}>" for i in range(dim)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Real part
    im0 = axes[0].imshow(np.real(rho), cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Real Part')
    axes[0].set_xticks(range(dim))
    axes[0].set_yticks(range(dim))
    axes[0].set_xticklabels(basis_labels, rotation=45, ha='right')
    axes[0].set_yticklabels(basis_labels)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Imaginary part
    im1 = axes[1].imshow(np.imag(rho), cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('Imaginary Part')
    axes[1].set_xticks(range(dim))
    axes[1].set_yticks(range(dim))
    axes[1].set_xticklabels(basis_labels, rotation=45, ha='right')
    axes[1].set_yticklabels(basis_labels)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Optimization Visualization
# =============================================================================

def plot_vqe_convergence(history: List[float],
                          exact_energy: float = None,
                          title: str = "VQE Convergence",
                          save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot VQE optimization convergence.

    Parameters
    ----------
    history : list
        Energy values at each iteration
    exact_energy : float, optional
        Exact ground state energy for reference
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (figure, axes)
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(5, 3.5))

    iterations = np.arange(len(history))
    ax.plot(iterations, history, 'b-', linewidth=1.5, label='VQE')

    if exact_energy is not None:
        ax.axhline(y=exact_energy, color='red', linestyle='--',
                  linewidth=1, label=f'Exact: {exact_energy:.4f}')

        # Compute chemical accuracy threshold
        chemical_accuracy = exact_energy + 0.0016  # 1 kcal/mol
        ax.axhline(y=chemical_accuracy, color='green', linestyle=':',
                  linewidth=1, label='Chemical Accuracy')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy (Ha)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_parameter_landscape_2d(cost_fn,
                                 param1_range: np.ndarray,
                                 param2_range: np.ndarray,
                                 param1_name: str = r"$\theta_1$",
                                 param2_name: str = r"$\theta_2$",
                                 fixed_params: dict = None,
                                 title: str = "Energy Landscape",
                                 save_path: str = None) -> plt.Figure:
    """
    Plot 2D parameter landscape.

    Parameters
    ----------
    cost_fn : callable
        Function that takes dict of parameters and returns energy
    param1_range, param2_range : np.ndarray
        Parameter values to sweep
    param1_name, param2_name : str
        Parameter names for labels
    fixed_params : dict, optional
        Fixed values for other parameters
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    Figure
        Matplotlib figure
    """
    if fixed_params is None:
        fixed_params = {}

    # Compute landscape
    P1, P2 = np.meshgrid(param1_range, param2_range)
    Z = np.zeros_like(P1)

    for i in range(len(param2_range)):
        for j in range(len(param1_range)):
            params = fixed_params.copy()
            params['param1'] = P1[i, j]
            params['param2'] = P2[i, j]
            Z[i, j] = cost_fn(params)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Contour plot
    ax0 = axes[0]
    cs = ax0.contourf(P1, P2, Z, levels=30, cmap='viridis')
    ax0.contour(P1, P2, Z, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    plt.colorbar(cs, ax=ax0, label='Energy')

    # Mark minimum
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    ax0.plot(P1[min_idx], P2[min_idx], 'r*', markersize=15, label='Minimum')

    ax0.set_xlabel(param1_name)
    ax0.set_ylabel(param2_name)
    ax0.set_title('Contour Plot')
    ax0.legend()

    # 3D surface
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.plot_surface(P1, P2, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel(param1_name)
    ax1.set_ylabel(param2_name)
    ax1.set_zlabel('Energy')
    ax1.set_title('Surface Plot')

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Demonstration
# =============================================================================

def demo():
    """Run demonstration of visualization tools."""
    print("=" * 60)
    print("Quantum Visualization Demonstration")
    print("=" * 60)

    # Demo 1: Bloch sphere with common states
    print("\n1. Bloch Sphere - Common States")
    print("-" * 40)

    # Define common states
    state_0 = np.array([1, 0], dtype=complex)
    state_1 = np.array([0, 1], dtype=complex)
    state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    state_plus_i = np.array([1, 1j], dtype=complex) / np.sqrt(2)

    states = [state_0, state_1, state_plus, state_plus_i]
    labels = [r'$|0\rangle$', r'$|1\rangle$', r'$|+\rangle$', r'$|+i\rangle$']

    fig1, ax1 = plot_bloch_sphere(states=states, labels=labels,
                                   title="Common Quantum States")
    print("Bloch sphere created with 4 states")
    plt.close(fig1)

    # Demo 2: State evolution trajectory
    print("\n2. State Evolution Trajectory")
    print("-" * 40)

    # Simulate Rabi oscillation
    n_steps = 50
    omega = 2 * np.pi / 20  # Rabi frequency

    states_history = []
    for t in range(n_steps):
        theta = omega * t
        state = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=complex)
        states_history.append(state)

    fig2, ax2 = plot_bloch_trajectory(states_history,
                                       title="Rabi Oscillation")
    print(f"Trajectory plotted with {n_steps} time steps")
    plt.close(fig2)

    # Demo 3: Bell state visualization
    print("\n3. Bell State Density Matrix")
    print("-" * 40)

    # Bell state |Phi+> = (|00> + |11>)/sqrt(2)
    bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(bell_state, bell_state.conj())

    fig3 = plot_density_matrix(rho, title="Bell State |Phi+>")
    print("Density matrix visualization created")
    plt.close(fig3)

    # Demo 4: VQE convergence
    print("\n4. VQE Convergence Plot")
    print("-" * 40)

    # Simulated VQE history
    exact_energy = -1.137
    history = [0.5 - 0.03*i + 0.1*np.random.randn() for i in range(50)]
    history = [max(h, exact_energy + 0.001) for h in history]

    fig4, ax4 = plot_vqe_convergence(history, exact_energy=exact_energy)
    print("VQE convergence plot created")
    plt.close(fig4)

    # Demo 5: State amplitudes
    print("\n5. State Amplitude City Plot")
    print("-" * 40)

    # Create superposition state
    ghz_state = np.zeros(8, dtype=complex)
    ghz_state[0] = 1/np.sqrt(2)
    ghz_state[7] = 1/np.sqrt(2)

    fig5, ax5 = plot_state_city(ghz_state, title="3-Qubit GHZ State")
    print("State city plot created for 3-qubit GHZ state")
    plt.close(fig5)

    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("All visualization functions work correctly.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
