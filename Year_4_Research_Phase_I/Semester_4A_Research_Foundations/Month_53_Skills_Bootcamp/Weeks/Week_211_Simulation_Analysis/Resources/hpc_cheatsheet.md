# High-Performance Computing Cheatsheet

## Quick Reference for Quantum Simulation

---

## Memory Requirements

| Qubits | States | Statevector Size | RAM (complex128) |
|--------|--------|------------------|------------------|
| 10 | 1,024 | 16 KB | ~20 KB |
| 15 | 32,768 | 512 KB | ~600 KB |
| 20 | 1,048,576 | 16 MB | ~20 MB |
| 25 | 33,554,432 | 512 MB | ~600 MB |
| 30 | 1,073,741,824 | 16 GB | ~20 GB |
| 35 | 34,359,738,368 | 512 GB | ~600 GB |
| 40 | 1,099,511,627,776 | 16 TB | ~20 TB |

**Formula:** Memory = 2^n * 16 bytes (complex128)

---

## Parallel Computing

### joblib (Easiest)

```python
from joblib import Parallel, delayed

# Parallel loop
results = Parallel(n_jobs=-1)(
    delayed(function)(arg) for arg in args
)

# With progress bar
results = Parallel(n_jobs=4, verbose=10)(
    delayed(function)(arg) for arg in args
)
```

### concurrent.futures

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# CPU-bound tasks (use processes)
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(function, args))

# I/O-bound tasks (use threads)
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(function, args))
```

### multiprocessing

```python
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(function, args)
```

---

## Profiling

### Time Profiling

```python
# Simple timing
import time
start = time.perf_counter()
# ... code ...
print(f"Elapsed: {time.perf_counter() - start:.4f}s")

# Context manager
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - start:.4f}s")

with timer("Matrix multiply"):
    result = A @ B
```

### cProfile

```bash
# From command line
python -m cProfile -s cumtime script.py

# In code
import cProfile
cProfile.run('function()', sort='cumtime')
```

### line_profiler

```python
# Install: pip install line_profiler

@profile
def function_to_profile():
    pass

# Run: kernprof -l -v script.py
```

### memory_profiler

```python
# Install: pip install memory_profiler

from memory_profiler import profile

@profile
def memory_intensive():
    pass

# Run: python -m memory_profiler script.py
```

---

## NumPy Optimization

### Vectorization

```python
# Bad: Python loop
result = []
for x in data:
    result.append(x ** 2)

# Good: Vectorized
result = data ** 2

# Bad: Element-wise condition
for i in range(len(data)):
    if data[i] > 0:
        data[i] = 0

# Good: Boolean indexing
data[data > 0] = 0
```

### Broadcasting

```python
# Outer product without explicit loop
A = np.arange(3)[:, np.newaxis]  # (3, 1)
B = np.arange(4)[np.newaxis, :]  # (1, 4)
C = A + B  # (3, 4)
```

### In-place Operations

```python
# Creates new array
A = A + B

# In-place (saves memory)
A += B
np.add(A, B, out=A)
```

### Pre-allocation

```python
# Bad: Growing array
results = []
for i in range(1000):
    results.append(compute(i))
results = np.array(results)

# Good: Pre-allocate
results = np.empty(1000)
for i in range(1000):
    results[i] = compute(i)
```

---

## Numba JIT

### Basic Usage

```python
from numba import jit

@jit(nopython=True)
def fast_function(x):
    result = 0.0
    for i in range(len(x)):
        result += x[i] ** 2
    return result
```

### Parallel Loops

```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def parallel_function(x):
    result = np.zeros_like(x)
    for i in prange(len(x)):
        result[i] = x[i] ** 2
    return result
```

### Type Hints

```python
from numba import jit, float64, int32

@jit(float64[:](float64[:], int32), nopython=True)
def typed_function(x, n):
    return x[:n] ** 2
```

---

## Sparse Matrices

```python
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

# Create sparse matrix
sparse = csr_matrix((data, (rows, cols)), shape=(m, n))

# From dense
sparse = csr_matrix(dense_matrix)

# Operations
result = sparse @ vector  # Matrix-vector product
result = sparse.dot(sparse2)  # Sparse-sparse product

# Sparse eigenvalues
from scipy.sparse.linalg import eigsh
vals, vecs = eigsh(sparse, k=5, which='SA')
```

---

## HDF5 Storage

```python
import h5py

# Write
with h5py.File('data.h5', 'w') as f:
    f.create_dataset('array', data=array, compression='gzip')
    f.attrs['description'] = 'My data'

# Read
with h5py.File('data.h5', 'r') as f:
    data = f['array'][:]
    desc = f.attrs['description']

# Partial read (memory efficient)
with h5py.File('data.h5', 'r') as f:
    chunk = f['array'][1000:2000]
```

---

## Matplotlib Performance

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (faster)
import matplotlib.pyplot as plt

# Faster for many points
plt.plot(x, y, rasterized=True)

# Reduce figure complexity
fig.set_dpi(100)  # Lower DPI for screen
fig.savefig('output.png', dpi=300)  # Higher DPI for print

# Clear figures
plt.close('all')  # Free memory
```

---

## Simulation Backend Selection

### PennyLane

```python
# Fast C++ simulator
dev = qml.device('lightning.qubit', wires=20)

# Default Python
dev = qml.device('default.qubit', wires=10)

# With noise
dev = qml.device('default.mixed', wires=10)
```

### Qiskit

```python
from qiskit_aer import AerSimulator

# Automatic method selection
sim = AerSimulator(method='automatic')

# Specific methods
sim = AerSimulator(method='statevector')  # Exact
sim = AerSimulator(method='matrix_product_state')  # Tensor network
```

---

## Quick Benchmarking Template

```python
import numpy as np
import time

def benchmark(func, *args, n_runs=5, warmup=1, **kwargs):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'result': result
    }

# Usage
stats = benchmark(my_function, arg1, arg2, n_runs=10)
print(f"Time: {stats['mean']:.4f} +/- {stats['std']:.4f}s")
```

---

## Environment Variables

```bash
# OpenMP threads
export OMP_NUM_THREADS=4

# MKL threads
export MKL_NUM_THREADS=4

# NumPy threads
export OPENBLAS_NUM_THREADS=4

# Disable GPU
export CUDA_VISIBLE_DEVICES=""
```

---

*Week 211: Simulation & Analysis - HPC Quick Reference*
