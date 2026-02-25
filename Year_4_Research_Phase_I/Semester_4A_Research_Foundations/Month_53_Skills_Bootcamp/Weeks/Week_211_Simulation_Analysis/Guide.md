# Simulation & Analysis Guide

## High-Performance Computing and Visualization for Quantum Research

---

## Table of Contents

1. [High-Performance Simulation](#1-high-performance-simulation)
2. [Parallel Computing](#2-parallel-computing)
3. [Profiling and Optimization](#3-profiling-and-optimization)
4. [Data Management](#4-data-management)
5. [Scientific Visualization](#5-scientific-visualization)
6. [Quantum-Specific Visualization](#6-quantum-specific-visualization)

---

## 1. High-Performance Simulation

### 1.1 Choosing the Right Simulator

```python
import numpy as np
from typing import Callable
import time

def benchmark_simulator(simulator_func: Callable, n_qubits: int, n_runs: int = 5):
    """
    Benchmark a quantum simulator.

    Parameters
    ----------
    simulator_func : callable
        Function that creates and runs a circuit
    n_qubits : int
        Number of qubits to simulate
    n_runs : int
        Number of benchmark runs

    Returns
    -------
    dict
        Timing statistics
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        simulator_func(n_qubits)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

# Example: Compare PennyLane backends
def pennylane_default(n_qubits):
    import pennylane as qml
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        return qml.state()

    return circuit()

def pennylane_lightning(n_qubits):
    import pennylane as qml
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        return qml.state()

    return circuit()
```

### 1.2 Memory-Efficient Simulation

```python
import numpy as np
from scipy.sparse import csr_matrix, kron, eye

def sparse_pauli_z(n_qubits: int, qubit: int) -> csr_matrix:
    """
    Create sparse Pauli Z operator for specified qubit.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits
    qubit : int
        Qubit index (0-indexed)

    Returns
    -------
    csr_matrix
        Sparse Pauli Z operator
    """
    Z = csr_matrix([[1, 0], [0, -1]], dtype=complex)
    I = eye(2, format='csr', dtype=complex)

    result = eye(1, format='csr', dtype=complex)
    for i in range(n_qubits):
        if i == qubit:
            result = kron(result, Z, format='csr')
        else:
            result = kron(result, I, format='csr')

    return result

def sparse_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 0.5) -> csr_matrix:
    """
    Create sparse transverse-field Ising Hamiltonian.

    H = -J * sum(Z_i Z_{i+1}) - h * sum(X_i)

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    J : float
        Coupling strength
    h : float
        Transverse field strength

    Returns
    -------
    csr_matrix
        Sparse Hamiltonian
    """
    dim = 2 ** n_qubits
    H = csr_matrix((dim, dim), dtype=complex)

    X = csr_matrix([[0, 1], [1, 0]], dtype=complex)
    Z = csr_matrix([[1, 0], [0, -1]], dtype=complex)
    I = eye(2, format='csr', dtype=complex)

    # ZZ terms
    for i in range(n_qubits - 1):
        ZZ = eye(1, format='csr', dtype=complex)
        for j in range(n_qubits):
            if j == i or j == i + 1:
                ZZ = kron(ZZ, Z, format='csr')
            else:
                ZZ = kron(ZZ, I, format='csr')
        H -= J * ZZ

    # X terms
    for i in range(n_qubits):
        XI = eye(1, format='csr', dtype=complex)
        for j in range(n_qubits):
            if j == i:
                XI = kron(XI, X, format='csr')
            else:
                XI = kron(XI, I, format='csr')
        H -= h * XI

    return H

def ground_state_sparse(H: csr_matrix, k: int = 1):
    """
    Find ground state using sparse eigensolver.

    Parameters
    ----------
    H : csr_matrix
        Sparse Hamiltonian
    k : int
        Number of eigenvalues to compute

    Returns
    -------
    tuple
        (eigenvalues, eigenvectors)
    """
    from scipy.sparse.linalg import eigsh

    # Use shift-invert for ground state
    eigenvalues, eigenvectors = eigsh(H, k=k, which='SA')
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]
```

### 1.3 Vectorized Operations

```python
import numpy as np

def vectorized_expectation_values(states: np.ndarray,
                                   operators: list) -> np.ndarray:
    """
    Compute expectation values for multiple states and operators.

    Parameters
    ----------
    states : np.ndarray
        Array of state vectors, shape (n_states, dim)
    operators : list
        List of operator matrices

    Returns
    -------
    np.ndarray
        Expectation values, shape (n_states, n_operators)
    """
    n_states = states.shape[0]
    n_ops = len(operators)
    results = np.zeros((n_states, n_ops))

    for j, op in enumerate(operators):
        # Vectorized: compute op @ states.T, then element-wise multiply and sum
        op_states = op @ states.T  # Shape: (dim, n_states)
        results[:, j] = np.real(np.sum(states.conj() * op_states.T, axis=1))

    return results

def batch_unitary_evolution(states: np.ndarray,
                            unitary: np.ndarray) -> np.ndarray:
    """
    Apply unitary to batch of states.

    Parameters
    ----------
    states : np.ndarray
        Batch of state vectors, shape (batch_size, dim)
    unitary : np.ndarray
        Unitary matrix, shape (dim, dim)

    Returns
    -------
    np.ndarray
        Evolved states, shape (batch_size, dim)
    """
    # Efficient batch matrix multiplication
    return (unitary @ states.T).T
```

---

## 2. Parallel Computing

### 2.1 Using joblib for Parallel Loops

```python
from joblib import Parallel, delayed
import numpy as np

def parameter_sweep_parallel(circuit_fn, param_ranges, n_jobs=-1):
    """
    Parallel parameter sweep for quantum circuits.

    Parameters
    ----------
    circuit_fn : callable
        Function that takes parameters and returns result
    param_ranges : list of arrays
        Parameter values for each parameter
    n_jobs : int
        Number of parallel jobs (-1 for all cores)

    Returns
    -------
    np.ndarray
        Results array
    """
    from itertools import product

    # Generate all parameter combinations
    param_grid = list(product(*param_ranges))

    # Parallel execution
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(circuit_fn)(params) for params in param_grid
    )

    return np.array(results)

# Example usage
def vqe_energy(params):
    """Compute VQE energy for given parameters."""
    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return circuit(params)

# Run parallel sweep
# theta_range = np.linspace(0, np.pi, 20)
# results = parameter_sweep_parallel(
#     lambda params: vqe_energy(np.array(params)),
#     [theta_range, theta_range],
#     n_jobs=4
# )
```

### 2.2 Using concurrent.futures

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

def parallel_vqe_optimization(hamiltonians: list, n_workers: int = 4):
    """
    Run VQE optimization for multiple Hamiltonians in parallel.

    Parameters
    ----------
    hamiltonians : list
        List of Hamiltonian matrices or specifications
    n_workers : int
        Number of parallel workers

    Returns
    -------
    list
        List of optimization results
    """
    def optimize_single(H_spec):
        """Optimize single Hamiltonian."""
        # Import inside function for process isolation
        import pennylane as qml
        from pennylane import numpy as np
        from scipy.optimize import minimize

        dev = qml.device('default.qubit', wires=2)

        coeffs, obs = H_spec
        H = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(dev)
        def cost(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(H)

        def cost_numpy(params):
            return float(cost(np.array(params, requires_grad=False)))

        result = minimize(cost_numpy, np.random.randn(2) * 0.1, method='COBYLA')
        return {'energy': result.fun, 'params': result.x}

    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(optimize_single, hamiltonians))

    return results

def parallel_circuit_sampling(circuit_fn, n_samples: int,
                              n_threads: int = 4) -> np.ndarray:
    """
    Sample quantum circuit in parallel using threads.

    Useful for I/O-bound operations like hardware access.

    Parameters
    ----------
    circuit_fn : callable
        Function that executes circuit and returns measurement
    n_samples : int
        Number of samples to collect
    n_threads : int
        Number of threads

    Returns
    -------
    np.ndarray
        Array of measurement results
    """
    # Use ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(circuit_fn) for _ in range(n_samples)]
        results = [f.result() for f in futures]

    return np.array(results)
```

### 2.3 Dask for Large-Scale Computing

```python
import numpy as np

def dask_parameter_sweep():
    """
    Example of using Dask for distributed parameter sweeps.
    """
    import dask
    from dask import delayed
    import dask.array as da

    @delayed
    def compute_energy(theta, phi):
        """Delayed computation of energy."""
        # Simulate circuit computation
        return np.cos(theta) * np.cos(phi)

    # Create delayed computation graph
    theta_vals = np.linspace(0, np.pi, 50)
    phi_vals = np.linspace(0, np.pi, 50)

    results = []
    for theta in theta_vals:
        row = []
        for phi in phi_vals:
            row.append(compute_energy(theta, phi))
        results.append(row)

    # Compute in parallel
    with dask.config.set(scheduler='threads', num_workers=4):
        computed = dask.compute(*[dask.compute(*row) for row in results])

    return np.array(computed)
```

---

## 3. Profiling and Optimization

### 3.1 Profiling Code

```python
import cProfile
import pstats
from io import StringIO
import time

def profile_function(func, *args, **kwargs):
    """
    Profile a function and return statistics.

    Parameters
    ----------
    func : callable
        Function to profile
    *args, **kwargs
        Arguments to pass to function

    Returns
    -------
    tuple
        (result, stats_string)
    """
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()

    # Get statistics
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    return result, stream.getvalue()

def line_profile_decorator(func):
    """
    Decorator for line-by-line profiling.

    Requires line_profiler: pip install line_profiler
    """
    try:
        from line_profiler import LineProfiler

        def wrapper(*args, **kwargs):
            lp = LineProfiler()
            lp.add_function(func)
            lp.enable()
            result = func(*args, **kwargs)
            lp.disable()
            lp.print_stats()
            return result

        return wrapper
    except ImportError:
        print("line_profiler not installed. Using regular execution.")
        return func

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Block"):
        self.name = name
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name}: {self.elapsed:.4f} seconds")

# Usage:
# with Timer("Matrix multiplication"):
#     result = A @ B
```

### 3.2 Numba JIT Compilation

```python
import numpy as np
from numba import jit, prange, complex128

@jit(nopython=True, cache=True)
def fast_state_evolution(state: np.ndarray,
                         hamiltonian: np.ndarray,
                         dt: float,
                         n_steps: int) -> np.ndarray:
    """
    Fast time evolution using Numba JIT.

    Parameters
    ----------
    state : np.ndarray
        Initial state vector
    hamiltonian : np.ndarray
        Hamiltonian matrix
    dt : float
        Time step
    n_steps : int
        Number of time steps

    Returns
    -------
    np.ndarray
        Final state vector
    """
    # Compute propagator U = exp(-i H dt)
    # For small dt, use first-order approximation
    dim = len(state)
    identity = np.eye(dim, dtype=np.complex128)
    propagator = identity - 1j * dt * hamiltonian

    # Normalize propagator (approximate unitarity)
    for i in range(dim):
        norm = 0.0
        for j in range(dim):
            norm += np.abs(propagator[i, j])**2
        norm = np.sqrt(norm)
        for j in range(dim):
            propagator[i, j] /= norm

    # Evolve
    current_state = state.copy()
    for _ in range(n_steps):
        new_state = np.zeros(dim, dtype=np.complex128)
        for i in range(dim):
            for j in range(dim):
                new_state[i] += propagator[i, j] * current_state[j]
        current_state = new_state

    # Normalize
    norm = 0.0
    for i in range(dim):
        norm += np.abs(current_state[i])**2
    norm = np.sqrt(norm)
    for i in range(dim):
        current_state[i] /= norm

    return current_state

@jit(nopython=True, parallel=True)
def parallel_expectation_values(states: np.ndarray,
                                 operator: np.ndarray) -> np.ndarray:
    """
    Compute expectation values in parallel.

    Parameters
    ----------
    states : np.ndarray
        Array of states, shape (n_states, dim)
    operator : np.ndarray
        Operator matrix, shape (dim, dim)

    Returns
    -------
    np.ndarray
        Expectation values, shape (n_states,)
    """
    n_states = states.shape[0]
    dim = states.shape[1]
    results = np.zeros(n_states, dtype=np.float64)

    for i in prange(n_states):
        exp_val = 0.0 + 0.0j
        for j in range(dim):
            for k in range(dim):
                exp_val += np.conj(states[i, j]) * operator[j, k] * states[i, k]
        results[i] = np.real(exp_val)

    return results
```

---

## 4. Data Management

### 4.1 HDF5 for Large Datasets

```python
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

def save_experiment_results(filepath: Path, results: dict, metadata: dict):
    """
    Save experiment results to HDF5 file.

    Parameters
    ----------
    filepath : Path
        Output file path
    results : dict
        Dictionary of numpy arrays to save
    metadata : dict
        Experiment metadata
    """
    with h5py.File(filepath, 'w') as f:
        # Save metadata as attributes
        for key, value in metadata.items():
            if isinstance(value, str):
                f.attrs[key] = value
            elif isinstance(value, (int, float)):
                f.attrs[key] = value
            else:
                f.attrs[key] = str(value)

        # Add timestamp
        f.attrs['created_at'] = datetime.now().isoformat()

        # Save arrays
        for name, data in results.items():
            if isinstance(data, np.ndarray):
                f.create_dataset(name, data=data, compression='gzip')
            elif isinstance(data, list):
                f.create_dataset(name, data=np.array(data), compression='gzip')

def load_experiment_results(filepath: Path) -> tuple:
    """
    Load experiment results from HDF5 file.

    Parameters
    ----------
    filepath : Path
        Input file path

    Returns
    -------
    tuple
        (results dict, metadata dict)
    """
    results = {}
    metadata = {}

    with h5py.File(filepath, 'r') as f:
        # Load metadata
        for key in f.attrs:
            metadata[key] = f.attrs[key]

        # Load datasets
        for name in f.keys():
            results[name] = f[name][:]

    return results, metadata

def append_to_experiment(filepath: Path, new_results: dict):
    """
    Append new results to existing HDF5 file.

    Parameters
    ----------
    filepath : Path
        HDF5 file path
    new_results : dict
        New data to append
    """
    with h5py.File(filepath, 'a') as f:
        for name, data in new_results.items():
            if name in f:
                # Resize and append
                dset = f[name]
                old_shape = dset.shape
                new_shape = (old_shape[0] + len(data),) + old_shape[1:]
                dset.resize(new_shape)
                dset[old_shape[0]:] = data
            else:
                # Create new dataset with unlimited first dimension
                maxshape = (None,) + data.shape[1:] if data.ndim > 1 else (None,)
                f.create_dataset(name, data=data, maxshape=maxshape,
                               compression='gzip', chunks=True)
```

### 4.2 Experiment Tracking

```python
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import hashlib

@dataclass
class ExperimentRun:
    """Container for experiment run data."""
    name: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    artifacts: Dict[str, Path]
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"
    run_id: Optional[str] = None

    def __post_init__(self):
        if self.run_id is None:
            # Generate unique run ID
            hash_input = f"{self.name}{self.start_time}{json.dumps(self.config, sort_keys=True)}"
            self.run_id = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

class ExperimentTracker:
    """Simple experiment tracking without external dependencies."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.current_run: Optional[ExperimentRun] = None

    def start_run(self, name: str, config: Dict[str, Any]) -> ExperimentRun:
        """Start a new experiment run."""
        self.current_run = ExperimentRun(
            name=name,
            config=config,
            metrics={},
            artifacts={},
            start_time=datetime.now().isoformat()
        )

        # Create run directory
        run_dir = self.experiment_dir / self.current_run.run_id
        run_dir.mkdir(exist_ok=True)

        # Save config
        with open(run_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        return self.current_run

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if self.current_run is None:
            raise ValueError("No active run")

        if name not in self.current_run.metrics:
            self.current_run.metrics[name] = []

        entry = {'value': value, 'timestamp': datetime.now().isoformat()}
        if step is not None:
            entry['step'] = step

        self.current_run.metrics[name].append(entry)

    def log_artifact(self, name: str, data: Any, artifact_type: str = "numpy"):
        """Save an artifact."""
        if self.current_run is None:
            raise ValueError("No active run")

        run_dir = self.experiment_dir / self.current_run.run_id

        if artifact_type == "numpy":
            filepath = run_dir / f"{name}.npy"
            np.save(filepath, data)
        elif artifact_type == "json":
            filepath = run_dir / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif artifact_type == "figure":
            filepath = run_dir / f"{name}.png"
            data.savefig(filepath, dpi=150, bbox_inches='tight')
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

        self.current_run.artifacts[name] = filepath

    def end_run(self, status: str = "completed"):
        """End the current run."""
        if self.current_run is None:
            raise ValueError("No active run")

        self.current_run.end_time = datetime.now().isoformat()
        self.current_run.status = status

        # Save run summary
        run_dir = self.experiment_dir / self.current_run.run_id
        summary = asdict(self.current_run)
        summary['artifacts'] = {k: str(v) for k, v in summary['artifacts'].items()}

        with open(run_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        run = self.current_run
        self.current_run = None
        return run
```

---

## 5. Scientific Visualization

### 5.1 Publication-Quality Matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    rcParams['figure.figsize'] = (3.5, 2.5)  # Single column width
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['axes.linewidth'] = 0.8
    rcParams['lines.linewidth'] = 1.0
    rcParams['lines.markersize'] = 4
    rcParams['legend.frameon'] = False
    rcParams['text.usetex'] = False  # Set True if LaTeX is available

def plot_convergence(history: list, exact_value: float = None,
                    title: str = None, save_path: str = None):
    """
    Plot optimization convergence.

    Parameters
    ----------
    history : list
        List of objective values during optimization
    exact_value : float, optional
        Exact solution for comparison
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(4, 3))

    iterations = np.arange(len(history))
    ax.plot(iterations, history, 'b-', linewidth=1.5, label='VQE')

    if exact_value is not None:
        ax.axhline(y=exact_value, color='r', linestyle='--',
                  linewidth=1, label=f'Exact = {exact_value:.4f}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy (Ha)')
    ax.legend(loc='upper right')

    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_heatmap(data: np.ndarray, x_label: str, y_label: str,
                title: str = None, cmap: str = 'viridis',
                save_path: str = None):
    """
    Create publication-quality heatmap.

    Parameters
    ----------
    data : np.ndarray
        2D array of values
    x_label, y_label : str
        Axis labels
    title : str, optional
        Plot title
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(4, 3.5))

    im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_with_error_bars(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                         label: str = None, save_path: str = None):
    """
    Plot data with error bars.

    Parameters
    ----------
    x : np.ndarray
        X values
    y : np.ndarray
        Y values (mean)
    yerr : np.ndarray
        Y error (std or confidence interval)
    label : str, optional
        Data label
    save_path : str, optional
        Path to save figure
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=3,
                capthick=1, markersize=4, label=label)

    ax.set_xlabel('Parameter')
    ax.set_ylabel('Value')

    if label:
        ax.legend()

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax
```

---

## 6. Quantum-Specific Visualization

### 6.1 Bloch Sphere

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def state_to_bloch(state: np.ndarray) -> tuple:
    """
    Convert single-qubit state to Bloch vector.

    Parameters
    ----------
    state : np.ndarray
        2-component complex state vector

    Returns
    -------
    tuple
        (x, y, z) Bloch vector coordinates
    """
    rho = np.outer(state, state.conj())

    # Bloch vector components
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])

    return x, y, z

def plot_bloch_sphere(states: list = None, vectors: list = None,
                      labels: list = None, save_path: str = None):
    """
    Plot states on Bloch sphere.

    Parameters
    ----------
    states : list of np.ndarray, optional
        List of quantum states to plot
    vectors : list of tuples, optional
        List of (x, y, z) Bloch vectors
    labels : list of str, optional
        Labels for each state/vector
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1, linewidth=0.5)

    # Draw axes
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='k', arrow_length_ratio=0.1, linewidth=1)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='k', arrow_length_ratio=0.1, linewidth=1)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='k', arrow_length_ratio=0.1, linewidth=1)

    ax.text(1.4, 0, 0, r'$|+\rangle$', fontsize=10)
    ax.text(0, 1.4, 0, r'$|+i\rangle$', fontsize=10)
    ax.text(0, 0, 1.4, r'$|0\rangle$', fontsize=10)
    ax.text(0, 0, -1.4, r'$|1\rangle$', fontsize=10)

    # Plot states
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    if states is not None:
        for i, state in enumerate(states):
            x, y, z = state_to_bloch(state)
            color = colors[i % len(colors)]
            label = labels[i] if labels and i < len(labels) else None
            ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.1,
                     linewidth=2, label=label)
            ax.scatter([x], [y], [z], color=color, s=50)

    if vectors is not None:
        for i, (x, y, z) in enumerate(vectors):
            color = colors[(i + len(states) if states else i) % len(colors)]
            label = labels[i + len(states) if states else i] if labels else None
            ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.1,
                     linewidth=2, label=label)
            ax.scatter([x], [y], [z], color=color, s=50)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if labels:
        ax.legend(loc='upper left')

    ax.set_box_aspect([1, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax

def plot_state_evolution(states_history: list, times: np.ndarray = None,
                        save_path: str = None):
    """
    Animate state evolution on Bloch sphere.

    Parameters
    ----------
    states_history : list
        List of states at each time step
    times : np.ndarray, optional
        Time values
    save_path : str, optional
        Path to save animation
    """
    from matplotlib.animation import FuncAnimation

    # Convert all states to Bloch vectors
    bloch_vectors = [state_to_bloch(s) for s in states_history]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)

    # Initialize arrow and point
    arrow = ax.quiver(0, 0, 0, 0, 0, 1, color='blue', arrow_length_ratio=0.1)
    point, = ax.plot([], [], [], 'ro', markersize=8)
    trajectory, = ax.plot([], [], [], 'b-', alpha=0.5, linewidth=1)

    trajectory_x, trajectory_y, trajectory_z = [], [], []

    def init():
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        return arrow, point, trajectory

    def update(frame):
        nonlocal arrow
        x, y, z = bloch_vectors[frame]

        # Update trajectory
        trajectory_x.append(x)
        trajectory_y.append(y)
        trajectory_z.append(z)

        # Remove old arrow and create new
        arrow.remove()
        arrow = ax.quiver(0, 0, 0, x, y, z, color='blue', arrow_length_ratio=0.1)

        point.set_data([x], [y])
        point.set_3d_properties([z])

        trajectory.set_data(trajectory_x, trajectory_y)
        trajectory.set_3d_properties(trajectory_z)

        if times is not None:
            ax.set_title(f't = {times[frame]:.2f}')

        return arrow, point, trajectory

    anim = FuncAnimation(fig, update, frames=len(states_history),
                        init_func=init, blit=False, interval=50)

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)

    return anim
```

### 6.2 Energy Landscape Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_energy_landscape(energy_fn, param_ranges: tuple,
                          n_points: int = 50, save_path: str = None):
    """
    Plot 2D energy landscape for variational optimization.

    Parameters
    ----------
    energy_fn : callable
        Function that takes (theta, phi) and returns energy
    param_ranges : tuple
        ((theta_min, theta_max), (phi_min, phi_max))
    n_points : int
        Number of grid points per dimension
    save_path : str, optional
        Path to save figure
    """
    theta_range, phi_range = param_ranges

    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    phi = np.linspace(phi_range[0], phi_range[1], n_points)

    THETA, PHI = np.meshgrid(theta, phi)
    ENERGY = np.zeros_like(THETA)

    for i in range(n_points):
        for j in range(n_points):
            ENERGY[i, j] = energy_fn(THETA[i, j], PHI[i, j])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Contour plot
    ax1 = axes[0]
    cs = ax1.contourf(THETA, PHI, ENERGY, levels=30, cmap='viridis')
    ax1.contour(THETA, PHI, ENERGY, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    plt.colorbar(cs, ax=ax1, label='Energy')
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$\phi$')
    ax1.set_title('Energy Landscape')

    # Mark global minimum
    min_idx = np.unravel_index(np.argmin(ENERGY), ENERGY.shape)
    ax1.plot(THETA[min_idx], PHI[min_idx], 'r*', markersize=15, label='Global minimum')
    ax1.legend()

    # 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(THETA, PHI, ENERGY, cmap='viridis', alpha=0.8)
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$\phi$')
    ax2.set_zlabel('Energy')
    ax2.set_title('3D Energy Surface')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes
```

---

## Quick Reference

### Performance Optimization Checklist

1. **Profile first** - Don't optimize without data
2. **Vectorize** - Use NumPy broadcasting
3. **Parallelize** - joblib for CPU-bound, threads for I/O
4. **JIT compile** - Numba for hot loops
5. **Cache results** - Avoid redundant computation
6. **Choose right backend** - Lightning, GPU, tensor networks

### Visualization Checklist

1. **Font sizes** - Readable at final size
2. **Colors** - Colorblind-friendly
3. **Labels** - Clear axis labels with units
4. **Legend** - Not overlapping data
5. **DPI** - 300+ for print
6. **File format** - PDF/SVG for vectors, PNG for raster

---

*Week 211: Simulation & Analysis - Complete methodology guide for high-performance quantum simulation and visualization.*
