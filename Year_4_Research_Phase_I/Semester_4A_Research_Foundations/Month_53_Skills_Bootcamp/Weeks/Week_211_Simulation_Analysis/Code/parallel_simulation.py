"""
Parallel Quantum Simulation Examples
====================================

This module demonstrates parallel computing techniques for
quantum simulation and optimization.

Author: Quantum Engineering PhD Program
Week 211: Simulation & Analysis

Requirements:
    pip install numpy matplotlib joblib
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any
import time
from pathlib import Path


# =============================================================================
# Parallel Parameter Sweeps
# =============================================================================

def parameter_sweep_sequential(cost_fn: Callable,
                                param_grid: List[Tuple],
                                verbose: bool = False) -> np.ndarray:
    """
    Sequential parameter sweep (baseline).

    Parameters
    ----------
    cost_fn : callable
        Function that takes parameters and returns cost
    param_grid : list of tuples
        List of parameter combinations
    verbose : bool
        Print progress

    Returns
    -------
    np.ndarray
        Array of cost values
    """
    results = []
    for i, params in enumerate(param_grid):
        result = cost_fn(params)
        results.append(result)
        if verbose and (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(param_grid)}")

    return np.array(results)


def parameter_sweep_parallel(cost_fn: Callable,
                              param_grid: List[Tuple],
                              n_jobs: int = -1,
                              backend: str = 'loky') -> np.ndarray:
    """
    Parallel parameter sweep using joblib.

    Parameters
    ----------
    cost_fn : callable
        Function that takes parameters and returns cost
    param_grid : list of tuples
        List of parameter combinations
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    backend : str
        Joblib backend ('loky', 'threading', 'multiprocessing')

    Returns
    -------
    np.ndarray
        Array of cost values
    """
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=5)(
        delayed(cost_fn)(params) for params in param_grid
    )

    return np.array(results)


def create_parameter_grid(param_ranges: List[np.ndarray]) -> List[Tuple]:
    """
    Create grid of all parameter combinations.

    Parameters
    ----------
    param_ranges : list of arrays
        Range of values for each parameter

    Returns
    -------
    list of tuples
        All parameter combinations
    """
    from itertools import product
    return list(product(*param_ranges))


# =============================================================================
# VQE Example
# =============================================================================

def create_vqe_cost_function(n_qubits: int = 2):
    """
    Create a VQE cost function for parallel evaluation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits

    Returns
    -------
    callable
        Cost function that takes parameter tuple
    """
    # Define H2-like Hamiltonian matrix
    H = np.array([
        [-1.0523, 0, 0, 0.1809],
        [0, 0.3979, 0, 0],
        [0, 0, -0.3979, 0],
        [0.1809, 0, 0, -0.0112]
    ], dtype=complex)

    def cost_function(params: Tuple[float, ...]) -> float:
        """Compute VQE cost for given parameters."""
        theta1, theta2, theta3, theta4 = params

        # Simple hardware-efficient ansatz
        # |00> -> RY(theta1) x RY(theta2) -> CNOT -> RY(theta3) x RY(theta4)

        # Initial state
        state = np.array([1, 0, 0, 0], dtype=complex)

        # RY gates on each qubit
        def ry(theta):
            return np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ], dtype=complex)

        # Apply RY(theta1) x RY(theta2)
        ry_layer1 = np.kron(ry(theta1), ry(theta2))
        state = ry_layer1 @ state

        # CNOT
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        state = cnot @ state

        # Apply RY(theta3) x RY(theta4)
        ry_layer2 = np.kron(ry(theta3), ry(theta4))
        state = ry_layer2 @ state

        # Compute expectation value
        energy = np.real(state.conj() @ H @ state)
        return energy

    return cost_function


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_parallel_speedup(cost_fn: Callable,
                                param_ranges: List[np.ndarray],
                                max_workers: int = 8) -> Dict[str, Any]:
    """
    Benchmark parallel vs sequential execution.

    Parameters
    ----------
    cost_fn : callable
        Cost function to benchmark
    param_ranges : list of arrays
        Parameter ranges for grid
    max_workers : int
        Maximum number of workers to test

    Returns
    -------
    dict
        Benchmark results
    """
    param_grid = create_parameter_grid(param_ranges)
    n_points = len(param_grid)

    print(f"Benchmarking with {n_points} parameter combinations")
    print("=" * 50)

    results = {
        'n_points': n_points,
        'sequential_time': None,
        'parallel_times': {},
        'speedups': {}
    }

    # Sequential baseline
    print("\nSequential execution...")
    start = time.perf_counter()
    _ = parameter_sweep_sequential(cost_fn, param_grid)
    seq_time = time.perf_counter() - start
    results['sequential_time'] = seq_time
    print(f"Sequential time: {seq_time:.2f}s")

    # Parallel with different worker counts
    worker_counts = [2, 4, 6, 8]
    worker_counts = [w for w in worker_counts if w <= max_workers]

    for n_workers in worker_counts:
        print(f"\nParallel with {n_workers} workers...")
        start = time.perf_counter()
        _ = parameter_sweep_parallel(cost_fn, param_grid, n_jobs=n_workers)
        par_time = time.perf_counter() - start
        results['parallel_times'][n_workers] = par_time
        results['speedups'][n_workers] = seq_time / par_time
        print(f"Time: {par_time:.2f}s, Speedup: {results['speedups'][n_workers]:.2f}x")

    return results


def plot_speedup(benchmark_results: Dict[str, Any], save_path: str = None):
    """
    Plot speedup results.

    Parameters
    ----------
    benchmark_results : dict
        Results from benchmark_parallel_speedup
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt

    workers = list(benchmark_results['speedups'].keys())
    speedups = list(benchmark_results['speedups'].values())

    fig, ax = plt.subplots(figsize=(6, 4))

    # Actual speedup
    ax.plot(workers, speedups, 'bo-', linewidth=2, markersize=8, label='Actual')

    # Ideal speedup
    ax.plot(workers, workers, 'r--', linewidth=1.5, label='Ideal (linear)')

    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Speedup')
    ax.set_title('Parallel Speedup Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.set_xticks(workers)
    ax.set_xlim([min(workers) - 0.5, max(workers) + 0.5])
    ax.set_ylim([0, max(max(speedups), max(workers)) + 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig, ax


# =============================================================================
# Energy Landscape with Parallel Computation
# =============================================================================

def compute_energy_landscape_parallel(cost_fn: Callable,
                                       theta_range: np.ndarray,
                                       phi_range: np.ndarray,
                                       n_jobs: int = -1) -> np.ndarray:
    """
    Compute 2D energy landscape in parallel.

    Parameters
    ----------
    cost_fn : callable
        Cost function taking (theta, phi) as first two parameters
    theta_range : np.ndarray
        Values for first parameter
    phi_range : np.ndarray
        Values for second parameter
    n_jobs : int
        Number of parallel jobs

    Returns
    -------
    np.ndarray
        2D energy landscape
    """
    from joblib import Parallel, delayed

    # Create grid
    THETA, PHI = np.meshgrid(theta_range, phi_range)
    points = [(THETA[i, j], PHI[i, j], 0, 0)
              for i in range(len(phi_range))
              for j in range(len(theta_range))]

    # Parallel computation
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(cost_fn)(p) for p in points
    )

    # Reshape to 2D
    landscape = np.array(results).reshape(len(phi_range), len(theta_range))

    return landscape


def plot_energy_landscape(landscape: np.ndarray,
                          theta_range: np.ndarray,
                          phi_range: np.ndarray,
                          save_path: str = None):
    """
    Plot energy landscape.

    Parameters
    ----------
    landscape : np.ndarray
        2D energy values
    theta_range : np.ndarray
        Theta parameter values
    phi_range : np.ndarray
        Phi parameter values
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 5))

    # 2D contour plot
    ax1 = fig.add_subplot(121)
    THETA, PHI = np.meshgrid(theta_range, phi_range)
    cs = ax1.contourf(THETA, PHI, landscape, levels=30, cmap='viridis')
    ax1.contour(THETA, PHI, landscape, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    plt.colorbar(cs, ax=ax1, label='Energy')

    # Mark minimum
    min_idx = np.unravel_index(np.argmin(landscape), landscape.shape)
    ax1.plot(theta_range[min_idx[1]], phi_range[min_idx[0]], 'r*', markersize=15)

    ax1.set_xlabel(r'$\theta_1$')
    ax1.set_ylabel(r'$\theta_2$')
    ax1.set_title('Energy Landscape (2D)')

    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(THETA, PHI, landscape, cmap='viridis', alpha=0.8,
                    linewidth=0, antialiased=True)
    ax2.set_xlabel(r'$\theta_1$')
    ax2.set_ylabel(r'$\theta_2$')
    ax2.set_zlabel('Energy')
    ax2.set_title('Energy Landscape (3D)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


# =============================================================================
# Parallel VQE Optimization
# =============================================================================

def parallel_vqe_random_starts(cost_fn: Callable,
                                n_params: int,
                                n_starts: int = 10,
                                n_jobs: int = -1) -> Dict[str, Any]:
    """
    Run VQE from multiple random starting points in parallel.

    Parameters
    ----------
    cost_fn : callable
        VQE cost function
    n_params : int
        Number of variational parameters
    n_starts : int
        Number of random starting points
    n_jobs : int
        Number of parallel jobs

    Returns
    -------
    dict
        Best result and all results
    """
    from joblib import Parallel, delayed
    from scipy.optimize import minimize

    def optimize_from_start(seed: int) -> Dict[str, Any]:
        """Run single optimization from random start."""
        np.random.seed(seed)
        x0 = np.random.randn(n_params) * 0.1

        result = minimize(
            lambda x: cost_fn(tuple(x)),
            x0,
            method='COBYLA',
            options={'maxiter': 200}
        )

        return {
            'energy': result.fun,
            'params': result.x,
            'success': result.success,
            'seed': seed
        }

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(optimize_from_start)(seed) for seed in range(n_starts)
    )

    # Find best result
    best_idx = np.argmin([r['energy'] for r in results])
    best_result = results[best_idx]

    return {
        'best': best_result,
        'all_results': results,
        'n_converged': sum(r['success'] for r in results),
        'energies': [r['energy'] for r in results]
    }


# =============================================================================
# Demonstration
# =============================================================================

def demo():
    """Run demonstration of parallel simulation techniques."""
    print("=" * 60)
    print("Parallel Quantum Simulation Demonstration")
    print("=" * 60)

    # Create VQE cost function
    cost_fn = create_vqe_cost_function(n_qubits=2)

    # Demo 1: Benchmark parallel speedup
    print("\n" + "-" * 40)
    print("1. Parallel Speedup Benchmark")
    print("-" * 40)

    # Small grid for quick demo
    theta_range = np.linspace(-np.pi, np.pi, 15)
    param_ranges = [theta_range, theta_range, theta_range, theta_range]

    # Reduce grid for faster demo
    small_ranges = [np.linspace(-np.pi, np.pi, 8) for _ in range(4)]

    benchmark_results = benchmark_parallel_speedup(
        cost_fn,
        small_ranges,
        max_workers=4
    )

    # Demo 2: Energy landscape
    print("\n" + "-" * 40)
    print("2. Energy Landscape Computation")
    print("-" * 40)

    theta1_range = np.linspace(-np.pi, np.pi, 30)
    theta2_range = np.linspace(-np.pi, np.pi, 30)

    print(f"Computing {len(theta1_range) * len(theta2_range)} points...")
    start = time.perf_counter()
    landscape = compute_energy_landscape_parallel(
        cost_fn, theta1_range, theta2_range, n_jobs=4
    )
    elapsed = time.perf_counter() - start
    print(f"Computation time: {elapsed:.2f}s")

    # Find minimum
    min_idx = np.unravel_index(np.argmin(landscape), landscape.shape)
    min_energy = landscape[min_idx]
    print(f"Minimum energy: {min_energy:.6f}")
    print(f"At theta1={theta1_range[min_idx[1]]:.3f}, theta2={theta2_range[min_idx[0]]:.3f}")

    # Demo 3: Parallel multi-start optimization
    print("\n" + "-" * 40)
    print("3. Parallel Multi-Start VQE")
    print("-" * 40)

    vqe_results = parallel_vqe_random_starts(
        cost_fn, n_params=4, n_starts=8, n_jobs=4
    )

    print(f"Best energy: {vqe_results['best']['energy']:.6f}")
    print(f"Converged: {vqe_results['n_converged']}/{len(vqe_results['all_results'])}")
    print(f"Energy range: [{min(vqe_results['energies']):.6f}, {max(vqe_results['energies']):.6f}]")

    # Plot results
    print("\n" + "-" * 40)
    print("4. Generating Plots")
    print("-" * 40)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Plot speedup
        fig1, _ = plot_speedup(benchmark_results)
        plt.close(fig1)
        print("Speedup plot generated")

        # Plot energy landscape
        fig2 = plot_energy_landscape(landscape, theta1_range, theta2_range)
        plt.close(fig2)
        print("Energy landscape plot generated")

        # Plot VQE results histogram
        fig3, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vqe_results['energies'], bins=10, edgecolor='black')
        ax.axvline(vqe_results['best']['energy'], color='red', linestyle='--',
                  label=f"Best: {vqe_results['best']['energy']:.4f}")
        ax.set_xlabel('Energy')
        ax.set_ylabel('Count')
        ax.set_title('VQE Optimization Results Distribution')
        ax.legend()
        plt.close(fig3)
        print("VQE histogram generated")

    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
