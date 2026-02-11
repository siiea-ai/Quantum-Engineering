# Day 258: File I/O and Performance

## Overview

**Day 258** | **Week 37** | **Month 10: Scientific Computing**

Today we master data persistence and code optimization. Real physics computations often run for hours—you need to save results reliably. And when simulations become bottlenecks, you need profiling tools to find and fix performance issues. By day's end, you'll manage large datasets efficiently and write code that makes the most of your hardware.

**Prerequisites:** Days 253-257 (Python, NumPy, linear algebra, random numbers)
**Outcome:** Save/load arrays, profile code, optimize performance-critical sections

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Binary formats, HDF5, pandas basics |
| Afternoon | 3 hours | Practice: Profiling and optimization |
| Evening | 2 hours | Lab: Complete data pipeline for quantum system |

---

## Learning Objectives

By the end of Day 258, you will be able to:

1. **Save and load NumPy arrays** in binary and text formats
2. **Work with HDF5 files** for hierarchical data storage
3. **Use pandas** for tabular data and analysis
4. **Profile code** to identify performance bottlenecks
5. **Apply optimization techniques** (vectorization, memory layout, numba)
6. **Design efficient data pipelines** for computational physics
7. **Benchmark code** to verify optimizations work

---

## Core Content

### 1. NumPy Binary I/O

```python
import numpy as np

# Create test data
psi = np.random.randn(1000) + 1j * np.random.randn(1000)
energies = np.linspace(0, 10, 100)
hamiltonian = np.random.randn(50, 50)

# Save single array
np.save('wavefunction.npy', psi)

# Load single array
psi_loaded = np.load('wavefunction.npy')
print(f"Loaded shape: {psi_loaded.shape}")

# Save multiple arrays (compressed archive)
np.savez('quantum_data.npz', psi=psi, energies=energies, H=hamiltonian)

# Save compressed (slower save, smaller file)
np.savez_compressed('quantum_data_compressed.npz',
                     psi=psi, energies=energies, H=hamiltonian)

# Load archive (lazy loading)
data = np.load('quantum_data.npz')
print(f"Keys: {list(data.keys())}")
psi_loaded = data['psi']
energies_loaded = data['energies']
data.close()  # Close file when done

# Context manager (recommended)
with np.load('quantum_data.npz') as data:
    psi_loaded = data['psi']
    H_loaded = data['H']
```

### 2. Text File I/O

For human-readable output or compatibility:

```python
# Save as text (human-readable but larger)
np.savetxt('energies.txt', energies, header='Energy levels')

# Save with formatting
np.savetxt('energies_formatted.txt', energies,
           fmt='%.8f', header='Energy (atomic units)')

# Load text file
energies_loaded = np.loadtxt('energies.txt')

# For complex data, save real and imag separately
np.savetxt('psi_real.txt', psi.real)
np.savetxt('psi_imag.txt', psi.imag)

# Save 2D array with headers
data_table = np.column_stack([energies[:10], np.random.randn(10)])
np.savetxt('table.txt', data_table,
           fmt=['%.4f', '%.6e'],
           header='Energy    Value',
           delimiter='\t')
```

### 3. HDF5 for Large Datasets

HDF5 (Hierarchical Data Format) handles complex, large datasets:

```python
import h5py

# Create HDF5 file
with h5py.File('quantum_simulation.h5', 'w') as f:
    # Create groups (like folders)
    grp = f.create_group('harmonic_oscillator')

    # Save arrays as datasets
    grp.create_dataset('energies', data=energies)
    grp.create_dataset('wavefunctions', data=np.random.randn(10, 1000))

    # Add metadata as attributes
    grp.attrs['omega'] = 1.0
    grp.attrs['mass'] = 1.0
    grp.attrs['n_states'] = 10
    grp.attrs['description'] = 'Harmonic oscillator eigenstates'

    # Create another group
    params = f.create_group('parameters')
    params.create_dataset('grid', data=np.linspace(-10, 10, 1000))

# Read HDF5 file
with h5py.File('quantum_simulation.h5', 'r') as f:
    # List contents
    print("Groups:", list(f.keys()))

    # Access group
    ho = f['harmonic_oscillator']
    print("Datasets:", list(ho.keys()))

    # Read data
    E = ho['energies'][:]
    psi_0 = ho['wavefunctions'][0, :]  # First eigenstate

    # Read attributes
    omega = ho.attrs['omega']
    print(f"ω = {omega}")
```

### 4. Pandas for Tabular Data

```python
import pandas as pd
import numpy as np

# Create DataFrame from dict
data = {
    'n': range(10),
    'energy': [n + 0.5 for n in range(10)],
    'degeneracy': [1] * 10,
    'parity': ['even' if n % 2 == 0 else 'odd' for n in range(10)]
}
df = pd.DataFrame(data)
print(df)

# Save to CSV
df.to_csv('energy_levels.csv', index=False)

# Load from CSV
df_loaded = pd.read_csv('energy_levels.csv')

# Basic operations
print(f"Mean energy: {df['energy'].mean()}")
print(f"Even parity states:\n{df[df['parity'] == 'even']}")

# Save to HDF5 (fast binary)
df.to_hdf('energy_data.h5', key='levels', mode='w')

# Load from HDF5
df_from_h5 = pd.read_hdf('energy_data.h5', key='levels')
```

### 5. Profiling Code

Find bottlenecks before optimizing:

```python
import cProfile
import pstats
from io import StringIO

def expensive_function():
    """Function to profile."""
    result = 0
    for i in range(10000):
        arr = np.random.randn(100)
        result += np.sum(arr**2)
    return result

# Basic timing
import time
start = time.perf_counter()
result = expensive_function()
elapsed = time.perf_counter() - start
print(f"Time: {elapsed:.4f} s")

# Detailed profiling
profiler = cProfile.Profile()
profiler.enable()
result = expensive_function()
profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

# Line-by-line profiling (requires line_profiler)
# @profile  # Decorator for line_profiler
# def function_to_profile():
#     ...
```

### 6. Memory Profiling

```python
import sys

# Check array memory usage
arr = np.random.randn(1000, 1000)
print(f"Array size: {arr.nbytes / 1e6:.2f} MB")
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")

# Memory of Python objects
print(f"sys.getsizeof(arr): {sys.getsizeof(arr)} bytes (just metadata)")

# Total memory including data
total_memory = arr.nbytes + sys.getsizeof(arr)
print(f"Total: {total_memory / 1e6:.2f} MB")

# Choose appropriate dtype
arr_float32 = arr.astype(np.float32)
print(f"float64: {arr.nbytes / 1e6:.2f} MB")
print(f"float32: {arr_float32.nbytes / 1e6:.2f} MB")

# Views don't copy memory
view = arr[::2, ::2]  # Every other element
print(f"View shares memory: {np.shares_memory(arr, view)}")
```

### 7. Optimization Techniques

#### Vectorization (Most Important)

```python
# SLOW: Python loops
def slow_norm(psi):
    total = 0
    for x in psi:
        total += abs(x)**2
    return np.sqrt(total)

# FAST: Vectorized
def fast_norm(psi):
    return np.sqrt(np.sum(np.abs(psi)**2))

# Benchmark
psi = np.random.randn(100000) + 1j * np.random.randn(100000)

%timeit slow_norm(psi)  # ~500 ms
%timeit fast_norm(psi)  # ~0.5 ms
```

#### Memory Layout

```python
# C-order (row-major) vs F-order (column-major)
arr_c = np.ascontiguousarray(np.random.randn(1000, 1000))
arr_f = np.asfortranarray(np.random.randn(1000, 1000))

# Row-wise operations faster in C-order
%timeit np.sum(arr_c, axis=1)  # Faster
%timeit np.sum(arr_f, axis=1)  # Slower

# Column-wise operations faster in F-order
%timeit np.sum(arr_c, axis=0)  # Slower
%timeit np.sum(arr_f, axis=0)  # Faster
```

#### Numba JIT Compilation

```python
from numba import jit

# JIT compile the slow function
@jit(nopython=True)
def numba_norm(psi):
    total = 0.0
    for i in range(len(psi)):
        total += psi[i].real**2 + psi[i].imag**2
    return np.sqrt(total)

# First call compiles (slow), subsequent calls fast
result = numba_norm(psi)  # Compilation
%timeit numba_norm(psi)   # ~0.3 ms (even faster than vectorized!)
```

#### Avoiding Unnecessary Copies

```python
# BAD: Creates temporary arrays
result = a + b + c + d  # Creates 3 temporary arrays

# BETTER: In-place operations
result = a.copy()
result += b
result += c
result += d

# Or use np.add with out parameter
np.add(a, b, out=result)
np.add(result, c, out=result)
np.add(result, d, out=result)
```

---

## Quantum Mechanics Connection

### Saving Quantum Simulation Results

```python
def save_quantum_simulation(filename, energies, wavefunctions,
                            potential, grid, metadata=None):
    """
    Save complete quantum simulation to HDF5.

    Parameters
    ----------
    filename : str
        Output file path
    energies : ndarray
        Energy eigenvalues
    wavefunctions : ndarray
        Eigenstates (columns)
    potential : ndarray
        Potential function on grid
    grid : ndarray
        Spatial grid points
    metadata : dict, optional
        Additional parameters
    """
    import h5py
    from datetime import datetime

    with h5py.File(filename, 'w') as f:
        # Main datasets
        f.create_dataset('energies', data=energies,
                         compression='gzip', compression_opts=4)
        f.create_dataset('wavefunctions', data=wavefunctions,
                         compression='gzip', compression_opts=4)
        f.create_dataset('potential', data=potential)
        f.create_dataset('grid', data=grid)

        # Metadata
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['n_states'] = len(energies)
        f.attrs['n_grid'] = len(grid)
        f.attrs['dx'] = float(grid[1] - grid[0])

        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value

def load_quantum_simulation(filename):
    """Load quantum simulation from HDF5."""
    import h5py

    with h5py.File(filename, 'r') as f:
        data = {
            'energies': f['energies'][:],
            'wavefunctions': f['wavefunctions'][:],
            'potential': f['potential'][:],
            'grid': f['grid'][:],
            'metadata': dict(f.attrs)
        }
    return data
```

### Checkpointing Long Simulations

```python
def time_evolution_with_checkpoints(psi_0, H, t_final, dt,
                                    checkpoint_interval=100,
                                    checkpoint_file='evolution_checkpoint.npz'):
    """
    Time evolve with periodic checkpointing.

    Allows resuming interrupted calculations.
    """
    import os
    from scipy.linalg import expm

    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        checkpoint = np.load(checkpoint_file)
        psi = checkpoint['psi']
        t_start = checkpoint['t']
        step_start = checkpoint['step']
        print(f"Resuming from t = {t_start}")
    else:
        psi = psi_0.copy()
        t_start = 0.0
        step_start = 0

    # Evolution operator
    U = expm(-1j * H * dt)

    n_steps = int((t_final - t_start) / dt)

    for step in range(step_start, step_start + n_steps):
        psi = U @ psi
        t = t_start + (step - step_start + 1) * dt

        # Checkpoint periodically
        if (step + 1) % checkpoint_interval == 0:
            np.savez(checkpoint_file, psi=psi, t=t, step=step+1)
            print(f"Checkpoint at t = {t:.4f}")

    # Remove checkpoint file on completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return psi
```

---

## Worked Examples

### Example 1: Efficient Eigenstate Storage

```python
def save_eigenstates_efficient(filename, eigenstates, energies,
                               threshold=1e-10):
    """
    Save eigenstates efficiently by storing only significant components.

    For sparse states (like tight-binding models), this can save
    orders of magnitude in storage.
    """
    import h5py

    with h5py.File(filename, 'w') as f:
        f.create_dataset('energies', data=energies)

        for n, psi in enumerate(eigenstates.T):
            # Find significant components
            mask = np.abs(psi) > threshold
            indices = np.where(mask)[0]
            values = psi[mask]

            # Store sparse representation
            grp = f.create_group(f'state_{n}')
            grp.create_dataset('indices', data=indices)
            grp.create_dataset('values', data=values)
            grp.attrs['n_nonzero'] = len(indices)
            grp.attrs['total_dim'] = len(psi)

def load_eigenstate_sparse(filename, n, full_dim=None):
    """Load sparse eigenstate and reconstruct."""
    import h5py

    with h5py.File(filename, 'r') as f:
        grp = f[f'state_{n}']
        indices = grp['indices'][:]
        values = grp['values'][:]

        if full_dim is None:
            full_dim = grp.attrs['total_dim']

        psi = np.zeros(full_dim, dtype=values.dtype)
        psi[indices] = values

    return psi
```

### Example 2: Profiling a Quantum Simulation

```python
def profile_quantum_solver():
    """
    Profile eigenvalue computation for different methods.
    """
    import time

    sizes = [100, 200, 500, 1000]
    results = []

    for N in sizes:
        # Build Hamiltonian
        x = np.linspace(-10, 10, N)
        dx = x[1] - x[0]
        V = 0.5 * x**2

        T_coeff = 1 / (2 * dx**2)
        H = (np.diag(2*T_coeff + V) +
             np.diag(np.full(N-1, -T_coeff), 1) +
             np.diag(np.full(N-1, -T_coeff), -1))

        # Time eigenvalue computation
        start = time.perf_counter()
        energies, states = np.linalg.eigh(H)
        elapsed = time.perf_counter() - start

        results.append({
            'N': N,
            'time': elapsed,
            'memory_H': H.nbytes / 1e6,
            'memory_psi': states.nbytes / 1e6
        })

        print(f"N={N:4d}: {elapsed:.4f}s, "
              f"H={H.nbytes/1e6:.1f}MB, "
              f"ψ={states.nbytes/1e6:.1f}MB")

    return results

# Run profiling
print("Profiling eigenvalue solver:")
results = profile_quantum_solver()
```

### Example 3: Memory-Efficient Large Simulation

```python
def large_scale_simulation(n_grid=10000, n_states=100):
    """
    Memory-efficient simulation for large systems.

    Instead of storing full wavefunctions, compute and save
    observables on the fly.
    """
    # Grid
    x = np.linspace(-20, 20, n_grid)
    dx = x[1] - x[0]

    # Build Hamiltonian (sparse would be better for large N)
    V = 0.5 * x**2
    T_coeff = 1 / (2 * dx**2)

    # For very large systems, use sparse matrices
    from scipy.sparse import diags
    diagonals = [2*T_coeff + V, np.full(n_grid-1, -T_coeff), np.full(n_grid-1, -T_coeff)]
    H_sparse = diags(diagonals, [0, 1, -1], format='csr')

    # Use sparse eigenvalue solver for first k states
    from scipy.sparse.linalg import eigsh
    energies, states = eigsh(H_sparse, k=n_states, which='SA')

    # Sort by energy
    idx = np.argsort(energies)
    energies = energies[idx]
    states = states[:, idx]

    # Compute observables without storing all states
    observables = {
        'energies': energies,
        'x_expectation': np.zeros(n_states),
        'x2_expectation': np.zeros(n_states),
        'nodes': np.zeros(n_states, dtype=int)
    }

    for n in range(n_states):
        psi = states[:, n]
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        psi /= norm

        observables['x_expectation'][n] = np.sum(x * np.abs(psi)**2) * dx
        observables['x2_expectation'][n] = np.sum(x**2 * np.abs(psi)**2) * dx
        observables['nodes'][n] = np.sum(np.diff(np.sign(psi.real)) != 0)

    return observables
```

---

## Practice Problems

### Direct Application

**Problem 1:** Save a complex wave function array to both `.npy` and `.txt` formats. Compare file sizes.

**Problem 2:** Create an HDF5 file with groups for "ground_state", "first_excited", and "second_excited", each containing the wave function and energy.

**Problem 3:** Profile a matrix multiplication `A @ B` for matrices of size 100×100, 500×500, and 1000×1000. How does time scale?

### Intermediate

**Problem 4:** Implement a checkpoint system for a Monte Carlo simulation that saves the random state, accumulated statistics, and can resume from interruption.

**Problem 5:** Compare the performance of row-wise vs column-wise iteration over a 2D array. Explain the difference.

**Problem 6:** Write a function that streams large datasets from HDF5 without loading them entirely into memory.

### Challenging

**Problem 7:** Implement a caching decorator that saves function results to disk and reloads them if called with the same arguments.

**Problem 8:** Optimize a nested loop that computes pairwise interactions $V_{ij} = 1/|r_i - r_j|$ using vectorization, numba, and compare performance.

**Problem 9:** Design a data format for storing time-dependent quantum trajectories efficiently, including metadata for reproducibility.

---

## Computational Lab

### Complete Quantum Simulation Pipeline

```python
"""
Day 258 Computational Lab: File I/O and Performance
===================================================

Build a complete simulation pipeline with data persistence.
"""

import numpy as np
import h5py
import time
import os
from typing import Dict, Any, Callable
from dataclasses import dataclass, asdict

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SimulationParameters:
    """Container for simulation parameters."""
    x_min: float = -10.0
    x_max: float = 10.0
    n_grid: int = 500
    n_states: int = 20
    potential_type: str = 'harmonic'
    omega: float = 1.0
    mass: float = 1.0
    hbar: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SimulationResults:
    """Container for simulation results."""
    energies: np.ndarray
    wavefunctions: np.ndarray
    grid: np.ndarray
    potential: np.ndarray
    parameters: SimulationParameters

# ============================================================
# FILE I/O UTILITIES
# ============================================================

def save_simulation(filename: str, results: SimulationResults):
    """Save simulation results to HDF5."""
    with h5py.File(filename, 'w') as f:
        # Datasets with compression
        f.create_dataset('energies', data=results.energies,
                         compression='gzip', compression_opts=4)
        f.create_dataset('wavefunctions', data=results.wavefunctions,
                         compression='gzip', compression_opts=4)
        f.create_dataset('grid', data=results.grid)
        f.create_dataset('potential', data=results.potential)

        # Parameters as attributes
        params = f.create_group('parameters')
        for key, value in results.parameters.to_dict().items():
            params.attrs[key] = value

        # Metadata
        f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['numpy_version'] = np.__version__

def load_simulation(filename: str) -> SimulationResults:
    """Load simulation results from HDF5."""
    with h5py.File(filename, 'r') as f:
        params_dict = {key: f['parameters'].attrs[key]
                       for key in f['parameters'].attrs}
        parameters = SimulationParameters(**params_dict)

        results = SimulationResults(
            energies=f['energies'][:],
            wavefunctions=f['wavefunctions'][:],
            grid=f['grid'][:],
            potential=f['potential'][:],
            parameters=parameters
        )
    return results

# ============================================================
# PROFILING UTILITIES
# ============================================================

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            print(f"[{self.name}] {self.elapsed:.4f} s")

def benchmark(func: Callable, *args, n_runs: int = 10, **kwargs) -> Dict[str, float]:
    """Benchmark a function."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

# ============================================================
# SIMULATION CODE
# ============================================================

def build_hamiltonian(params: SimulationParameters) -> np.ndarray:
    """Build Hamiltonian matrix."""
    x = np.linspace(params.x_min, params.x_max, params.n_grid)
    dx = x[1] - x[0]

    # Potential
    if params.potential_type == 'harmonic':
        V = 0.5 * params.mass * params.omega**2 * x**2
    elif params.potential_type == 'infinite_well':
        V = np.zeros_like(x)
        V[(x < 0) | (x > 1)] = 1e10
    else:
        raise ValueError(f"Unknown potential: {params.potential_type}")

    # Kinetic energy (tridiagonal)
    T_coeff = params.hbar**2 / (2 * params.mass * dx**2)
    H = (np.diag(2*T_coeff + V) +
         np.diag(np.full(params.n_grid-1, -T_coeff), 1) +
         np.diag(np.full(params.n_grid-1, -T_coeff), -1))

    return H, x, V

def run_simulation(params: SimulationParameters) -> SimulationResults:
    """Run complete quantum simulation."""
    with Timer("Building Hamiltonian"):
        H, x, V = build_hamiltonian(params)

    with Timer("Solving eigenvalue problem"):
        energies, wavefunctions = np.linalg.eigh(H)

    # Keep only requested number of states
    energies = energies[:params.n_states]
    wavefunctions = wavefunctions[:, :params.n_states]

    with Timer("Normalizing wavefunctions"):
        dx = x[1] - x[0]
        for n in range(params.n_states):
            norm = np.sqrt(np.sum(np.abs(wavefunctions[:, n])**2) * dx)
            wavefunctions[:, n] /= norm

    return SimulationResults(
        energies=energies,
        wavefunctions=wavefunctions,
        grid=x,
        potential=V,
        parameters=params
    )

# ============================================================
# OPTIMIZATION COMPARISONS
# ============================================================

def compare_inner_product_methods(psi: np.ndarray, phi: np.ndarray,
                                  dx: float, n_trials: int = 100):
    """Compare different inner product implementations."""
    results = {}

    # Method 1: Loop (slow)
    def loop_inner(psi, phi, dx):
        total = 0j
        for i in range(len(psi)):
            total += np.conj(psi[i]) * phi[i]
        return total * dx

    # Method 2: Vectorized
    def vec_inner(psi, phi, dx):
        return np.sum(np.conj(psi) * phi) * dx

    # Method 3: np.vdot (optimized)
    def vdot_inner(psi, phi, dx):
        return np.vdot(psi, phi) * dx

    # Method 4: einsum
    def einsum_inner(psi, phi, dx):
        return np.einsum('i,i->', np.conj(psi), phi) * dx

    methods = {
        'loop': loop_inner,
        'vectorized': vec_inner,
        'vdot': vdot_inner,
        'einsum': einsum_inner
    }

    for name, func in methods.items():
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            result = func(psi, phi, dx)
            times.append(time.perf_counter() - start)
        results[name] = {
            'mean_time': np.mean(times),
            'result': result
        }

    return results

# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Day 258: File I/O and Performance")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Run and Save Simulation
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. QUANTUM SIMULATION PIPELINE")
    print("=" * 70)

    params = SimulationParameters(
        x_min=-10.0,
        x_max=10.0,
        n_grid=500,
        n_states=10,
        potential_type='harmonic',
        omega=1.0
    )

    print(f"\nParameters: {params}")

    with Timer("\nTotal simulation"):
        results = run_simulation(params)

    print(f"\nFirst 5 energies:")
    for n in range(5):
        exact = n + 0.5
        print(f"  E_{n} = {results.energies[n]:.6f} (exact: {exact})")

    # --------------------------------------------------------
    # 2. Save to Various Formats
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. FILE I/O COMPARISON")
    print("=" * 70)

    # Save to HDF5
    with Timer("\nSave to HDF5"):
        save_simulation('simulation.h5', results)

    # Save to NPZ
    with Timer("Save to NPZ"):
        np.savez('simulation.npz',
                 energies=results.energies,
                 wavefunctions=results.wavefunctions,
                 grid=results.grid,
                 potential=results.potential)

    # Save compressed
    with Timer("Save to NPZ (compressed)"):
        np.savez_compressed('simulation_compressed.npz',
                            energies=results.energies,
                            wavefunctions=results.wavefunctions,
                            grid=results.grid,
                            potential=results.potential)

    # Compare file sizes
    print("\nFile sizes:")
    for fname in ['simulation.h5', 'simulation.npz', 'simulation_compressed.npz']:
        size = os.path.getsize(fname)
        print(f"  {fname}: {size/1024:.1f} KB")

    # Load and verify
    with Timer("\nLoad from HDF5"):
        loaded = load_simulation('simulation.h5')

    print(f"Loaded energies match: {np.allclose(results.energies, loaded.energies)}")

    # --------------------------------------------------------
    # 3. Performance Comparison
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. INNER PRODUCT PERFORMANCE")
    print("=" * 70)

    # Create test wavefunctions
    N = 10000
    psi = np.random.randn(N) + 1j * np.random.randn(N)
    phi = np.random.randn(N) + 1j * np.random.randn(N)
    dx = 0.01

    print(f"\nArray size: {N}")

    comparison = compare_inner_product_methods(psi, phi, dx, n_trials=100)

    print("\nMethod comparison:")
    print(f"{'Method':>12} {'Time (μs)':>12} {'Result':>20}")
    print("-" * 50)

    baseline = comparison['loop']['mean_time']
    for name, data in comparison.items():
        t = data['mean_time'] * 1e6
        r = data['result']
        speedup = baseline / data['mean_time']
        print(f"{name:>12} {t:>12.2f} {r.real:>10.4f}+{r.imag:.4f}j  ({speedup:.1f}x)")

    # --------------------------------------------------------
    # 4. Memory Usage Analysis
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. MEMORY ANALYSIS")
    print("=" * 70)

    print("\nResults object memory:")
    print(f"  Energies: {results.energies.nbytes / 1024:.1f} KB")
    print(f"  Wavefunctions: {results.wavefunctions.nbytes / 1024:.1f} KB")
    print(f"  Grid: {results.grid.nbytes / 1024:.1f} KB")
    print(f"  Potential: {results.potential.nbytes / 1024:.1f} KB")
    total = (results.energies.nbytes + results.wavefunctions.nbytes +
             results.grid.nbytes + results.potential.nbytes)
    print(f"  Total: {total / 1024:.1f} KB")

    # dtype comparison
    print("\nDtype comparison (wavefunctions):")
    wf = results.wavefunctions
    for dtype in [np.float64, np.float32, np.complex128, np.complex64]:
        wf_cast = wf.real.astype(dtype) if np.issubdtype(dtype, np.floating) else wf.astype(dtype)
        print(f"  {str(dtype):>20}: {wf_cast.nbytes / 1024:.1f} KB")

    # --------------------------------------------------------
    # 5. Scaling Analysis
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. SCALING ANALYSIS")
    print("=" * 70)

    print("\nEigenvalue solver scaling:")
    print(f"{'N':>6} {'Time (s)':>10} {'Memory (MB)':>12}")
    print("-" * 32)

    for N in [100, 200, 500, 1000]:
        params_test = SimulationParameters(n_grid=N, n_states=10)

        start = time.perf_counter()
        H, _, _ = build_hamiltonian(params_test)
        energies, _ = np.linalg.eigh(H)
        elapsed = time.perf_counter() - start

        memory = H.nbytes / 1e6

        print(f"{N:>6} {elapsed:>10.4f} {memory:>12.2f}")

    # Cleanup test files
    for fname in ['simulation.h5', 'simulation.npz', 'simulation_compressed.npz']:
        if os.path.exists(fname):
            os.remove(fname)

    print("\n" + "=" * 70)
    print("Lab complete! Week review on Day 259.")
    print("=" * 70)
```

---

## Summary

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `np.save(f, a)` | Save single array | `np.save('data.npy', arr)` |
| `np.savez(f, **kw)` | Save multiple arrays | `np.savez('data.npz', x=x, y=y)` |
| `np.load(f)` | Load array(s) | `data = np.load('data.npz')` |
| `h5py.File(f)` | HDF5 file handle | `with h5py.File('data.h5') as f:` |
| `time.perf_counter()` | High-precision timer | `start = time.perf_counter()` |

### File Format Comparison

| Format | Pros | Cons |
|--------|------|------|
| `.npy` | Fast, simple | Single array only |
| `.npz` | Multiple arrays | No hierarchy |
| `.npz` (compressed) | Smaller files | Slower I/O |
| HDF5 | Hierarchy, metadata, partial I/O | More complex API |
| Text | Human-readable | Large, slow |

### Main Takeaways

1. **Use binary formats for speed** — `.npy` and HDF5 are much faster than text
2. **HDF5 for complex data** — hierarchical organization, metadata, compression
3. **Profile before optimizing** — find the actual bottlenecks
4. **Vectorization is usually enough** — 100x speedup is common
5. **Memory layout matters** — C-order for row operations, F-order for columns
6. **Checkpoint long simulations** — allows recovery from interruptions

---

## Daily Checklist

- [ ] Can save/load NumPy arrays in binary and text formats
- [ ] Comfortable with HDF5 for complex data
- [ ] Know basic pandas for tabular data
- [ ] Can profile code and identify bottlenecks
- [ ] Understand memory layout implications
- [ ] Completed performance optimization exercises
- [ ] Ran lab successfully
- [ ] Built a complete simulation pipeline

---

## Preview: Day 259

Tomorrow is the **Week 37 Review** where we integrate everything:
- Build a complete particle-in-a-box simulator
- Apply all NumPy skills: arrays, vectorization, linear algebra
- Save results and create visualizations
- Comprehensive exercises testing the week's material

---

*"Premature optimization is the root of all evil, but mature optimization is the fruit of profiling."*
