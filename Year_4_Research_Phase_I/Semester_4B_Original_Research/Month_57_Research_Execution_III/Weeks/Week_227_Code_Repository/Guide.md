# Code Repository Preparation Guide

## Introduction

This guide provides comprehensive instructions for transforming research code into a professional, open-source ready repository. Following these practices ensures your code is reproducible, maintainable, and valuable to the scientific community.

---

## Part 1: Repository Organization

### 1.1 Standard Directory Structure

A well-organized repository follows predictable conventions:

```
quantum_research_project/
│
├── .github/                          # GitHub-specific files
│   ├── workflows/
│   │   ├── ci.yml                   # Continuous integration
│   │   ├── docs.yml                 # Documentation deployment
│   │   └── release.yml              # Release automation
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── docs/                             # Documentation source
│   ├── source/
│   │   ├── conf.py                  # Sphinx configuration
│   │   ├── index.rst
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   └── api/
│   ├── tutorials/
│   │   ├── 01_getting_started.ipynb
│   │   ├── 02_basic_simulation.ipynb
│   │   └── 03_advanced_analysis.ipynb
│   └── Makefile
│
├── examples/                         # Example scripts
│   ├── basic_simulation.py
│   ├── parameter_sweep.py
│   └── notebooks/
│       └── example_workflow.ipynb
│
├── src/                              # Source code (src layout)
│   └── quantum_project/
│       ├── __init__.py
│       ├── __version__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── hamiltonian.py
│       │   ├── evolution.py
│       │   └── measurement.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── statistics.py
│       │   └── visualization.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── io.py
│       │   └── validation.py
│       └── data/
│           └── default_params.yaml
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── unit/
│   │   ├── test_hamiltonian.py
│   │   ├── test_evolution.py
│   │   └── test_measurement.py
│   ├── integration/
│   │   └── test_full_simulation.py
│   └── data/
│       └── test_fixtures.json
│
├── scripts/                          # Utility scripts
│   ├── download_data.py
│   └── generate_figures.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

### 1.2 File Organization Principles

**Separation of Concerns:**
- Source code in `src/` (protects against import issues)
- Tests separate from source
- Documentation in dedicated folder
- Examples clearly distinguished

**Naming Conventions:**
```python
# File names: lowercase with underscores
quantum_simulation.py  # Good
QuantumSimulation.py   # Avoid
quantum-simulation.py  # Avoid

# Module names: short, lowercase
import quantum_project.core as qp_core  # Good
import quantum_project.core_simulation_module  # Too long

# Class names: CamelCase
class QuantumSimulator:  # Good
class quantum_simulator:  # Wrong

# Function names: lowercase with underscores
def run_simulation():  # Good
def RunSimulation():   # Wrong
```

### 1.3 .gitignore Best Practices

```gitignore
# Byte-compiled / optimized files
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
dist/
build/
*.egg-info/
.eggs/

# Virtual environments
venv/
.venv/
env/

# IDE settings
.idea/
.vscode/
*.swp
*.swo

# Jupyter notebooks
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation builds
docs/_build/
docs/source/api/_autosummary/

# Project-specific
data/raw/
data/processed/
*.h5
*.hdf5
results/
figures/

# Secrets (NEVER commit these)
.env
*.key
credentials.json
secrets.yaml
```

---

## Part 2: Code Quality

### 2.1 Code Formatting

**Using Black for Consistent Formatting:**

```bash
# Install
pip install black isort

# Format all files
black src/ tests/
isort src/ tests/

# Check without modifying
black --check src/
```

**pyproject.toml configuration:**

```toml
[tool.black]
line-length = 88
target-version = ['py39', 'py310', 'py311']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 88
known_first_party = ["quantum_project"]
```

### 2.2 Type Hints

Add type hints to all public functions:

```python
from typing import Optional, Union, List, Tuple, Dict, Callable
import numpy as np
from numpy.typing import NDArray

def simulate_evolution(
    hamiltonian: NDArray[np.complex128],
    initial_state: NDArray[np.complex128],
    times: NDArray[np.float64],
    *,
    method: str = "expm",
    rtol: float = 1e-8,
    callback: Optional[Callable[[float, NDArray], None]] = None,
) -> Tuple[NDArray[np.complex128], Dict[str, float]]:
    """
    Simulate quantum state evolution under a Hamiltonian.

    Parameters
    ----------
    hamiltonian : ndarray of complex128
        The system Hamiltonian, shape (n, n)
    initial_state : ndarray of complex128
        Initial quantum state, shape (n,) or (n, 1)
    times : ndarray of float64
        Time points for evolution
    method : str, optional
        Evolution method: 'expm' or 'ode'. Default 'expm'
    rtol : float, optional
        Relative tolerance for ODE solver. Default 1e-8
    callback : callable, optional
        Function called at each time step

    Returns
    -------
    states : ndarray of complex128
        Evolved states at each time, shape (len(times), n)
    info : dict
        Computation metadata including runtime and method used

    Raises
    ------
    ValueError
        If hamiltonian is not square or dimensions don't match
    RuntimeError
        If evolution fails to converge

    Examples
    --------
    >>> H = np.array([[0, 1], [1, 0]], dtype=complex)
    >>> psi0 = np.array([1, 0], dtype=complex)
    >>> times = np.linspace(0, 1, 100)
    >>> states, info = simulate_evolution(H, psi0, times)
    """
    # Implementation here
    pass
```

### 2.3 Linting Configuration

**.flake8 configuration:**

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude =
    .git,
    __pycache__,
    docs/source/conf.py,
    build,
    dist
per-file-ignores =
    __init__.py:F401
```

**Pre-commit hooks (.pre-commit-config.yaml):**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [numpy]
```

---

## Part 3: Documentation

### 3.1 README.md Structure

A comprehensive README includes:

```markdown
# Project Name

[![CI](https://github.com/username/project/workflows/CI/badge.svg)](...)
[![codecov](https://codecov.io/gh/username/project/branch/main/graph/badge.svg)](...)
[![docs](https://readthedocs.org/projects/project/badge/?version=latest)](...)
[![PyPI](https://img.shields.io/pypi/v/project)](...)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](...)

Brief description (1-2 sentences).

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

### From PyPI
pip install project-name

### From source
git clone https://github.com/username/project.git
cd project
pip install -e .

## Quick Start

import project_name

# Minimal working example
result = project_name.do_something(...)

## Documentation

Full documentation: https://project.readthedocs.io

## Citation

If you use this software, please cite:

@software{project2024,
  author = {Author Name},
  title = {Project Name},
  year = {2024},
  url = {https://github.com/username/project}
}

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE)
```

### 3.2 Docstring Standards

Follow NumPy docstring convention:

```python
def calculate_fidelity(
    state1: NDArray[np.complex128],
    state2: NDArray[np.complex128],
    *,
    normalize: bool = True,
) -> float:
    """
    Calculate the fidelity between two quantum states.

    The fidelity is defined as |<psi1|psi2>|^2 for pure states.

    Parameters
    ----------
    state1 : ndarray
        First quantum state vector, shape (n,) or (n, 1)
    state2 : ndarray
        Second quantum state vector, shape (n,) or (n, 1)
    normalize : bool, optional
        If True, normalize states before calculation. Default True.

    Returns
    -------
    float
        Fidelity between 0 and 1, where 1 indicates identical states

    Raises
    ------
    ValueError
        If states have incompatible dimensions
    TypeError
        If inputs are not numpy arrays

    See Also
    --------
    trace_distance : Alternative distance measure
    process_fidelity : Fidelity for quantum channels

    Notes
    -----
    The fidelity is symmetric: F(psi1, psi2) = F(psi2, psi1)

    For mixed states (density matrices), use :func:`mixed_state_fidelity`.

    References
    ----------
    .. [1] Nielsen & Chuang, "Quantum Computation and Quantum Information"

    Examples
    --------
    >>> psi1 = np.array([1, 0], dtype=complex)
    >>> psi2 = np.array([1, 0], dtype=complex)
    >>> calculate_fidelity(psi1, psi2)
    1.0

    >>> psi3 = np.array([1, 1], dtype=complex) / np.sqrt(2)
    >>> calculate_fidelity(psi1, psi3)
    0.5
    """
    if normalize:
        state1 = state1 / np.linalg.norm(state1)
        state2 = state2 / np.linalg.norm(state2)

    overlap = np.vdot(state1, state2)
    return float(np.abs(overlap) ** 2)
```

### 3.3 Sphinx Documentation Setup

**docs/source/conf.py:**

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Quantum Project'
copyright = '2024, Author Name'
author = 'Author Name'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'nbsphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
```

---

## Part 4: Testing

### 4.1 Test Structure

**tests/conftest.py (shared fixtures):**

```python
import pytest
import numpy as np

@pytest.fixture
def random_state():
    """Provide reproducible random state for tests."""
    return np.random.RandomState(42)

@pytest.fixture
def sample_hamiltonian():
    """Create a sample 2-qubit Hamiltonian."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    # XX + ZZ interaction
    H = (np.kron(sigma_x, sigma_x) + np.kron(sigma_z, sigma_z))
    return H

@pytest.fixture
def ground_state(sample_hamiltonian):
    """Compute ground state of sample Hamiltonian."""
    eigenvalues, eigenvectors = np.linalg.eigh(sample_hamiltonian)
    return eigenvectors[:, 0]

@pytest.fixture(params=[2, 4, 8])
def system_size(request):
    """Parametrized fixture for different system sizes."""
    return request.param
```

**tests/unit/test_hamiltonian.py:**

```python
import pytest
import numpy as np
from quantum_project.core import hamiltonian

class TestHamiltonianConstruction:
    """Tests for Hamiltonian construction functions."""

    def test_pauli_x_hermitian(self):
        """Pauli X should be Hermitian."""
        sigma_x = hamiltonian.pauli_x()
        assert np.allclose(sigma_x, sigma_x.conj().T)

    def test_pauli_x_eigenvalues(self):
        """Pauli X should have eigenvalues +1 and -1."""
        sigma_x = hamiltonian.pauli_x()
        eigenvalues = np.linalg.eigvalsh(sigma_x)
        assert np.allclose(sorted(eigenvalues), [-1, 1])

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_identity_dimension(self, n_qubits):
        """Identity should have correct dimension."""
        I = hamiltonian.identity(n_qubits)
        expected_dim = 2 ** n_qubits
        assert I.shape == (expected_dim, expected_dim)

    def test_tensor_product_dimensions(self):
        """Tensor product should have correct dimensions."""
        A = np.random.randn(2, 2)
        B = np.random.randn(3, 3)
        C = hamiltonian.tensor_product(A, B)
        assert C.shape == (6, 6)

    def test_invalid_hamiltonian_raises(self):
        """Non-square matrix should raise ValueError."""
        with pytest.raises(ValueError, match="must be square"):
            hamiltonian.validate_hamiltonian(np.random.randn(2, 3))

    def test_non_hermitian_raises(self):
        """Non-Hermitian matrix should raise ValueError."""
        H = np.array([[1, 2], [3, 4]], dtype=complex)
        with pytest.raises(ValueError, match="must be Hermitian"):
            hamiltonian.validate_hamiltonian(H)


class TestHamiltonianEvolution:
    """Tests for time evolution under Hamiltonians."""

    def test_evolution_preserves_norm(self, sample_hamiltonian, ground_state):
        """Evolution should preserve state norm."""
        from quantum_project.core import evolution

        times = np.linspace(0, 1, 10)
        states = evolution.evolve(sample_hamiltonian, ground_state, times)

        for state in states:
            assert np.isclose(np.linalg.norm(state), 1.0, rtol=1e-10)

    def test_ground_state_stationary(self, sample_hamiltonian, ground_state):
        """Ground state should remain stationary (up to phase)."""
        from quantum_project.core import evolution

        final_state = evolution.evolve_to(
            sample_hamiltonian, ground_state, t=10.0
        )

        # Check fidelity, not direct equality (allows for global phase)
        fidelity = np.abs(np.vdot(ground_state, final_state)) ** 2
        assert np.isclose(fidelity, 1.0, rtol=1e-8)
```

### 4.2 pytest Configuration

**pyproject.toml:**

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--strict-config",
]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src/quantum_project"]
branch = true
omit = ["*/__init__.py", "*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
fail_under = 80
```

### 4.3 Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quantum_project --cov-report=html

# Run specific test file
pytest tests/unit/test_hamiltonian.py

# Run tests matching pattern
pytest -k "evolution"

# Run excluding slow tests
pytest -m "not slow"

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

---

## Part 5: Continuous Integration

### 5.1 GitHub Actions Workflow

**.github/workflows/ci.yml:**

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run linting
      run: |
        flake8 src tests
        black --check src tests
        isort --check-only src tests

    - name: Run type checking
      run: mypy src

    - name: Run tests
      run: pytest --cov=quantum_project --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e ".[docs]"

    - name: Build docs
      run: |
        cd docs
        make html

    - name: Check links
      run: |
        cd docs
        make linkcheck
```

### 5.2 Documentation Deployment

**.github/workflows/docs.yml:**

```yaml
name: Deploy Documentation

on:
  push:
    branches: [main]
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install -e ".[docs]"

    - name: Build documentation
      run: |
        cd docs
        make html

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

---

## Part 6: Open Source Preparation

### 6.1 License Selection

**Common licenses for scientific software:**

| License | Permissions | Conditions | When to Use |
|---------|-------------|------------|-------------|
| MIT | Very permissive | Attribution | Maximum adoption |
| Apache 2.0 | Permissive + patents | Attribution, notice | Patent protection |
| BSD 3-Clause | Permissive | Attribution | Academic tradition |
| GPL 3.0 | Copyleft | Source disclosure | Force open source |
| LGPL 3.0 | Weak copyleft | Library linking OK | Libraries |

**MIT License Template:**

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 6.2 Citation File

**CITATION.cff:**

```yaml
cff-version: 1.2.0
title: "Quantum Project: Efficient Simulation of Quantum Systems"
message: "If you use this software, please cite it as below."
type: software
authors:
  - given-names: "Your"
    family-names: "Name"
    email: "your.email@institution.edu"
    affiliation: "Your Institution"
    orcid: "https://orcid.org/0000-0000-0000-0000"
repository-code: "https://github.com/username/quantum-project"
url: "https://quantum-project.readthedocs.io"
license: MIT
version: 0.1.0
date-released: "2024-01-15"
keywords:
  - quantum computing
  - simulation
  - physics
abstract: >
  Quantum Project provides efficient tools for simulating
  quantum systems, including time evolution, measurement,
  and analysis utilities.
```

### 6.3 Contributing Guidelines

**CONTRIBUTING.md:**

```markdown
# Contributing to Quantum Project

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/project.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Make your changes
6. Run tests: `pytest`
7. Submit a pull request

## Development Setup

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
flake8 src tests
black --check src tests

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (NumPy format)
- Keep functions focused and small

## Pull Request Process

1. Ensure tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG.md
4. Request review from maintainers

## Reporting Bugs

Please include:
- Python version
- Package version
- Minimal reproducible example
- Expected vs actual behavior

## Questions?

Open an issue or email maintainers@project.org
```

---

## Part 7: Final Checklist

### 7.1 Repository Structure
- [ ] Standard directory layout implemented
- [ ] .gitignore configured properly
- [ ] No sensitive files committed
- [ ] No large binary files

### 7.2 Code Quality
- [ ] Consistent formatting (black/isort)
- [ ] No linting errors (flake8)
- [ ] Type hints on public APIs
- [ ] No dead code or unused imports

### 7.3 Documentation
- [ ] README.md complete and accurate
- [ ] Installation instructions tested
- [ ] All functions have docstrings
- [ ] API documentation generated
- [ ] Tutorials available

### 7.4 Testing
- [ ] Unit tests for core functionality
- [ ] Integration tests for workflows
- [ ] Coverage > 80%
- [ ] Tests pass on multiple Python versions

### 7.5 CI/CD
- [ ] GitHub Actions configured
- [ ] Tests run on push/PR
- [ ] Coverage reported
- [ ] Documentation builds

### 7.6 Open Source
- [ ] License file present
- [ ] CITATION.cff created
- [ ] CONTRIBUTING.md written
- [ ] CODE_OF_CONDUCT.md added
- [ ] CHANGELOG.md initialized

---

## Navigation

| Previous | Current | Next |
|----------|---------|------|
| Week 227 README | Code Repository Guide | [README Template](./Templates/README_Template.md) |
