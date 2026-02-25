# Advanced Python Best Practices Guide

## Research Software Engineering Principles

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Management](#2-environment-management)
3. [Version Control](#3-version-control)
4. [Testing Strategies](#4-testing-strategies)
5. [Documentation Standards](#5-documentation-standards)
6. [Code Quality](#6-code-quality)
7. [Reproducibility](#7-reproducibility)
8. [Debugging & Profiling](#8-debugging--profiling)

---

## 1. Project Structure

### Recommended Layout for Quantum Research Projects

```
my_quantum_project/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Continuous integration
│       └── docs.yml            # Documentation build
├── docs/
│   ├── conf.py                 # Sphinx configuration
│   ├── index.rst               # Documentation home
│   ├── api/                    # Auto-generated API docs
│   └── tutorials/              # User guides
├── notebooks/
│   ├── 01_exploration.ipynb    # Numbered for order
│   └── 02_analysis.ipynb
├── src/
│   └── my_quantum_project/
│       ├── __init__.py
│       ├── circuits/           # Quantum circuit modules
│       │   ├── __init__.py
│       │   └── variational.py
│       ├── simulation/         # Simulation backends
│       │   ├── __init__.py
│       │   └── state_vector.py
│       └── utils/              # Helper functions
│           ├── __init__.py
│           └── math_helpers.py
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_circuits/
│   └── test_simulation/
├── data/
│   ├── raw/                    # Immutable raw data
│   ├── processed/              # Cleaned data
│   └── results/                # Experiment outputs
├── scripts/
│   ├── run_experiment.py       # Entry points
│   └── analyze_results.py
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── LICENSE
└── Makefile
```

### Modern `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-quantum-project"
version = "0.1.0"
description = "Quantum research project for [your topic]"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "you@university.edu"}
]
keywords = ["quantum computing", "research"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Physics",
]

dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "qiskit>=1.0",
    "pennylane>=0.35",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "hypothesis>=6.90",
    "mypy>=1.8",
    "ruff>=0.2",
    "black>=24.0",
    "pre-commit>=3.6",
]
docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=2.0",
    "sphinx-autodoc-typehints>=2.0",
    "myst-parser>=2.0",
]

[project.urls]
Homepage = "https://github.com/username/my-quantum-project"
Documentation = "https://my-quantum-project.readthedocs.io"
Repository = "https://github.com/username/my-quantum-project"

[project.scripts]
run-experiment = "my_quantum_project.scripts:main"

[tool.hatch.build.targets.wheel]
packages = ["src/my_quantum_project"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=my_quantum_project --cov-report=term-missing"
filterwarnings = ["ignore::DeprecationWarning"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
plugins = ["numpy.typing.mypy_plugin"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "D", "UP", "B", "C4", "SIM"]
ignore = ["D100", "D104"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.black]
line-length = 88
target-version = ["py311"]
```

---

## 2. Environment Management

### Virtual Environments

**Using venv (standard library):**
```bash
# Create environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install project in editable mode
pip install -e ".[dev,docs]"
```

**Using conda:**
```yaml
# environment.yml
name: quantum-research
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.24
  - scipy>=1.10
  - matplotlib>=3.7
  - pip
  - pip:
    - qiskit>=1.0
    - pennylane>=0.35
    - -e ".[dev]"
```

**Using Poetry (recommended for packages):**
```bash
# Initialize
poetry init

# Add dependencies
poetry add numpy scipy qiskit
poetry add --group dev pytest mypy ruff

# Install
poetry install

# Run commands
poetry run pytest
```

### Dependency Locking

Always lock your dependencies for reproducibility:

```bash
# pip
pip freeze > requirements.lock

# pip-tools (recommended)
pip-compile pyproject.toml -o requirements.lock

# conda
conda env export > environment.lock.yml

# poetry
poetry lock
```

---

## 3. Version Control

### Git Configuration for Research

```bash
# Essential settings
git config --global user.name "Your Name"
git config --global user.email "you@university.edu"
git config --global core.autocrlf input  # For cross-platform
git config --global pull.rebase true     # Clean history

# Useful aliases
git config --global alias.lg "log --oneline --graph --decorate"
git config --global alias.st "status -sb"
git config --global alias.co "checkout"
git config --global alias.br "branch"
```

### Research-Oriented `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
.eggs/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Data (version with DVC instead)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Results (may want to version specific outputs)
data/results/*
!data/results/.gitkeep

# Large files
*.h5
*.hdf5
*.pkl
*.npy
*.npz

# Secrets
.env
*.pem
credentials.json

# OS
.DS_Store
Thumbs.db

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# Documentation build
docs/_build/
```

### Branching Strategy for Research

```
main
  │
  ├── develop (integration branch)
  │     │
  │     ├── feature/new-algorithm
  │     ├── feature/error-mitigation
  │     └── experiment/vqe-benchmark
  │
  ├── paper/arxiv-2024    (paper-specific code)
  └── release/v1.0.0      (stable releases)
```

### Commit Message Convention

```
<type>(<scope>): <short summary>

<body - what and why, not how>

<footer - references, breaking changes>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code restructuring
- `test`: Adding tests
- `perf`: Performance improvement
- `exp`: Experiment/research exploration

**Example:**
```
feat(vqe): implement ADAPT-VQE optimizer

Add adaptive derivative-assembled pseudo-Trotter ansatz for VQE.
This method dynamically grows the ansatz based on gradient information,
typically achieving chemical accuracy with fewer parameters.

Refs: arXiv:1812.11173
```

---

## 4. Testing Strategies

### Test Organization

```python
# tests/conftest.py - Shared fixtures
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

@pytest.fixture
def random_seed():
    """Provide reproducible random state."""
    return 42

@pytest.fixture
def simulator():
    """Provide a statevector simulator."""
    return AerSimulator(method='statevector')

@pytest.fixture
def bell_circuit():
    """Create a Bell state circuit."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

@pytest.fixture
def random_state(random_seed):
    """Provide reproducible random state vector."""
    np.random.seed(random_seed)
    state = np.random.randn(4) + 1j * np.random.randn(4)
    return state / np.linalg.norm(state)
```

### Testing Numerical Code

```python
# tests/test_simulation/test_state_vector.py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from my_quantum_project.simulation import evolve_state, compute_fidelity

class TestStateEvolution:
    """Tests for state vector evolution."""

    def test_unitary_preserves_norm(self, random_state):
        """Evolution should preserve state norm."""
        # Arrange
        hamiltonian = np.array([[1, 0], [0, -1]])  # Pauli Z
        time = 1.0

        # Act
        evolved = evolve_state(random_state[:2], hamiltonian, time)

        # Assert
        assert_allclose(np.linalg.norm(evolved), 1.0, rtol=1e-10)

    def test_identity_evolution(self):
        """Zero Hamiltonian should not change state."""
        state = np.array([1, 0], dtype=complex)
        hamiltonian = np.zeros((2, 2))

        evolved = evolve_state(state, hamiltonian, time=1.0)

        assert_allclose(evolved, state, atol=1e-14)

    @pytest.mark.parametrize("time", [0.1, 0.5, 1.0, np.pi])
    def test_pauli_z_evolution(self, time):
        """Test evolution under Pauli Z."""
        state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        hamiltonian = np.array([[1, 0], [0, -1]])

        evolved = evolve_state(state, hamiltonian, time)

        # Expected: |+> -> (e^{-it}|0> + e^{it}|1>) / sqrt(2)
        expected = np.array([
            np.exp(-1j * time),
            np.exp(1j * time)
        ]) / np.sqrt(2)

        assert_allclose(evolved, expected, atol=1e-12)


class TestFidelity:
    """Tests for fidelity computation."""

    def test_identical_states(self):
        """Fidelity of identical states should be 1."""
        state = np.array([1, 0], dtype=complex)
        assert_allclose(compute_fidelity(state, state), 1.0)

    def test_orthogonal_states(self):
        """Fidelity of orthogonal states should be 0."""
        state1 = np.array([1, 0], dtype=complex)
        state2 = np.array([0, 1], dtype=complex)
        assert_allclose(compute_fidelity(state1, state2), 0.0, atol=1e-14)

    def test_fidelity_symmetry(self, random_state):
        """Fidelity should be symmetric."""
        state1 = random_state[:2] / np.linalg.norm(random_state[:2])
        state2 = np.array([1, 1j], dtype=complex) / np.sqrt(2)

        f12 = compute_fidelity(state1, state2)
        f21 = compute_fidelity(state2, state1)

        assert_allclose(f12, f21)
```

### Property-Based Testing with Hypothesis

```python
# tests/test_circuits/test_properties.py
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

from my_quantum_project.circuits import apply_unitary

# Strategy for generating valid quantum states
def quantum_state(n_qubits: int):
    """Generate normalized quantum states."""
    dim = 2 ** n_qubits
    return arrays(
        dtype=np.complex128,
        shape=(dim,),
        elements=st.complex_numbers(
            min_magnitude=0, max_magnitude=1,
            allow_nan=False, allow_infinity=False
        )
    ).map(lambda x: x / (np.linalg.norm(x) + 1e-10))

# Strategy for generating unitary matrices
def unitary_matrix(dim: int):
    """Generate random unitary matrices via QR decomposition."""
    return arrays(
        dtype=np.complex128,
        shape=(dim, dim),
        elements=st.complex_numbers(
            min_magnitude=0, max_magnitude=1,
            allow_nan=False, allow_infinity=False
        )
    ).map(lambda x: np.linalg.qr(x)[0])

class TestUnitaryProperties:
    """Property-based tests for unitary operations."""

    @given(state=quantum_state(2), unitary=unitary_matrix(4))
    @settings(max_examples=100)
    def test_unitary_preserves_norm(self, state, unitary):
        """Unitary evolution preserves norm."""
        result = apply_unitary(state, unitary)
        assert np.abs(np.linalg.norm(result) - 1.0) < 1e-10

    @given(state=quantum_state(2))
    def test_identity_unchanged(self, state):
        """Identity operation leaves state unchanged."""
        identity = np.eye(4, dtype=complex)
        result = apply_unitary(state, identity)
        np.testing.assert_allclose(result, state, atol=1e-12)
```

---

## 5. Documentation Standards

### NumPy-Style Docstrings

```python
def optimize_variational_circuit(
    circuit: QuantumCircuit,
    hamiltonian: np.ndarray,
    initial_params: np.ndarray,
    optimizer: str = "COBYLA",
    maxiter: int = 1000,
    tol: float = 1e-6,
) -> OptimizationResult:
    """
    Optimize a variational quantum circuit to minimize energy expectation.

    This function implements the Variational Quantum Eigensolver (VQE) algorithm
    to find the ground state energy of a given Hamiltonian using a parameterized
    quantum circuit ansatz.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized quantum circuit serving as the ansatz.
        Must have bound parameters accessible via `circuit.parameters`.
    hamiltonian : np.ndarray
        Hermitian matrix representing the Hamiltonian, shape (2^n, 2^n)
        where n is the number of qubits.
    initial_params : np.ndarray
        Initial parameter values for the circuit, shape (n_params,).
    optimizer : str, optional
        Classical optimizer to use. Supported: "COBYLA", "SPSA", "L-BFGS-B".
        Default is "COBYLA".
    maxiter : int, optional
        Maximum number of optimizer iterations. Default is 1000.
    tol : float, optional
        Convergence tolerance for the optimizer. Default is 1e-6.

    Returns
    -------
    OptimizationResult
        Named tuple containing:
        - optimal_params : np.ndarray, optimized circuit parameters
        - optimal_energy : float, minimum energy found
        - convergence_history : list, energy values during optimization
        - n_function_evals : int, number of circuit evaluations

    Raises
    ------
    ValueError
        If hamiltonian is not Hermitian or dimensions don't match circuit.
    OptimizationError
        If optimizer fails to converge within maxiter iterations.

    See Also
    --------
    create_ansatz : Create parameterized ansatz circuits
    compute_expectation : Compute expectation values

    Notes
    -----
    The VQE algorithm was introduced in [1]_. This implementation uses
    statevector simulation for exact expectation value computation.

    For noisy simulations or hardware execution, consider using
    shot-based estimation with error mitigation.

    References
    ----------
    .. [1] Peruzzo, A. et al. "A variational eigenvalue solver on a photonic
           quantum processor." Nature Communications 5, 4213 (2014).

    Examples
    --------
    Basic usage with a 2-qubit circuit:

    >>> from qiskit import QuantumCircuit
    >>> from qiskit.circuit import Parameter
    >>> import numpy as np
    >>>
    >>> # Create parameterized ansatz
    >>> theta = Parameter('theta')
    >>> qc = QuantumCircuit(2)
    >>> qc.ry(theta, 0)
    >>> qc.cx(0, 1)
    >>>
    >>> # Define Hamiltonian (Heisenberg model)
    >>> H = np.array([[1, 0, 0, 0],
    ...               [0, -1, 2, 0],
    ...               [0, 2, -1, 0],
    ...               [0, 0, 0, 1]])
    >>>
    >>> result = optimize_variational_circuit(qc, H, np.array([0.1]))
    >>> print(f"Ground state energy: {result.optimal_energy:.6f}")
    Ground state energy: -3.000000
    """
    pass
```

### Sphinx Configuration

```python
# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'My Quantum Project'
copyright = '2024, Your Name'
author = 'Your Name'
version = '0.1.0'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

# Napoleon settings for NumPy docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'qiskit': ('https://docs.quantum.ibm.com/api/qiskit', None),
}

# Theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}

# Source file types
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
```

---

## 6. Code Quality

### Ruff Configuration (Modern Linter)

```toml
# In pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"
exclude = [
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".venv",
]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "D",   # pydocstyle
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "NPY", # NumPy-specific rules
]
ignore = [
    "D100",  # Missing docstring in public module
    "D104",  # Missing docstring in public package
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D"]  # No docstrings required in tests

[tool.ruff.lint.pydocstyle]
convention = "numpy"
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies:
          - numpy
          - types-requests

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/ -x --tb=short
        language: system
        pass_filenames: false
        always_run: true
```

---

## 7. Reproducibility

### Random Seed Management

```python
# src/my_quantum_project/utils/reproducibility.py
"""Utilities for reproducible experiments."""
import os
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np

def set_global_seed(seed: int) -> None:
    """
    Set random seeds for all relevant libraries.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Framework-specific seeds
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

@contextmanager
def reproducible_context(seed: int):
    """
    Context manager for reproducible code blocks.

    Saves and restores random states after the block.

    Examples
    --------
    >>> with reproducible_context(42):
    ...     result = run_stochastic_algorithm()
    """
    # Save states
    py_state = random.getstate()
    np_state = np.random.get_state()

    # Set seeds
    set_global_seed(seed)

    try:
        yield
    finally:
        # Restore states
        random.setstate(py_state)
        np.random.set_state(np_state)

def get_random_generator(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a numpy random generator with optional seed.

    Using Generator is preferred over global np.random for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed for the generator.

    Returns
    -------
    np.random.Generator
        Seeded random number generator.
    """
    return np.random.default_rng(seed)
```

### Experiment Configuration with Hydra

```python
# config/config.yaml
defaults:
  - _self_
  - optimizer: cobyla
  - ansatz: hardware_efficient

experiment:
  name: vqe_h2
  seed: 42
  n_shots: 1000

circuit:
  n_qubits: 4
  n_layers: 3

# config/optimizer/cobyla.yaml
name: COBYLA
maxiter: 1000
tol: 1e-6

# config/optimizer/spsa.yaml
name: SPSA
maxiter: 500
learning_rate: 0.1
perturbation: 0.01
```

```python
# scripts/run_experiment.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run VQE experiment with Hydra configuration."""
    from my_quantum_project.utils.reproducibility import set_global_seed
    from my_quantum_project.experiments import run_vqe

    # Set seed for reproducibility
    set_global_seed(cfg.experiment.seed)

    # Run experiment
    result = run_vqe(
        n_qubits=cfg.circuit.n_qubits,
        n_layers=cfg.circuit.n_layers,
        optimizer=cfg.optimizer.name,
        maxiter=cfg.optimizer.maxiter,
        n_shots=cfg.experiment.n_shots,
    )

    print(f"Optimal energy: {result.optimal_energy}")

if __name__ == "__main__":
    main()
```

---

## 8. Debugging & Profiling

### Debugging Techniques

```python
# Using breakpoint() for interactive debugging
def complex_calculation(data):
    intermediate = process_step_1(data)

    # Insert breakpoint for inspection
    breakpoint()  # or: import pdb; pdb.set_trace()

    result = process_step_2(intermediate)
    return result

# Using logging for production debugging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_vqe(circuit, hamiltonian):
    logger.info("Starting VQE optimization")
    logger.debug(f"Circuit depth: {circuit.depth()}")

    for iteration in range(max_iterations):
        energy = compute_energy(circuit, hamiltonian)
        logger.debug(f"Iteration {iteration}: energy = {energy:.6f}")

        if converged(energy):
            logger.info(f"Converged at iteration {iteration}")
            break

    return energy
```

### Profiling

```python
# Line-by-line profiling with line_profiler
# Install: pip install line_profiler

@profile  # Decorator for line_profiler
def expensive_function(n):
    result = []
    for i in range(n):
        result.append(np.random.randn(1000, 1000))
    return np.sum([np.sum(r) for r in result])

# Memory profiling
# Install: pip install memory_profiler

from memory_profiler import profile

@profile
def memory_intensive_function():
    large_array = np.zeros((10000, 10000))
    processed = np.fft.fft2(large_array)
    return np.abs(processed).sum()
```

### Performance Timing

```python
import time
from contextlib import contextmanager
from functools import wraps

@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f} seconds")

# Usage
with timer("Matrix multiplication"):
    result = A @ B

def timed(func):
    """Decorator for timing function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.4f} seconds")
        return result
    return wrapper

@timed
def expensive_computation():
    # ...
    pass
```

---

## Quick Reference Commands

```bash
# Environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"

# Testing
pytest                          # Run all tests
pytest -x                       # Stop on first failure
pytest -k "test_fidelity"       # Run matching tests
pytest --cov --cov-report=html  # Coverage report

# Code quality
ruff check .                    # Lint
ruff format .                   # Format
mypy src/                       # Type check
pre-commit run --all-files      # All checks

# Documentation
cd docs && make html            # Build docs
python -m http.server -d docs/_build/html  # Serve locally

# Git
git status -sb                  # Short status
git log --oneline -10           # Recent history
git diff --staged               # Staged changes
```

---

*This guide provides the foundation for professional research software development. Adapt these practices to your specific research needs while maintaining the core principles of reproducibility, testability, and documentation.*
