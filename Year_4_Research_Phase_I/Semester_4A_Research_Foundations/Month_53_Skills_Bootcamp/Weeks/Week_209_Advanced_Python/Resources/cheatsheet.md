# Advanced Python Quick Reference

## Research Software Engineering Cheatsheet

---

## Project Setup Commands

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install in editable mode with extras
pip install -e ".[dev,docs]"

# Create requirements lock file
pip freeze > requirements.lock

# Using conda
conda env create -f environment.yml
conda activate quantum-research

# Using poetry
poetry init
poetry add numpy scipy qiskit
poetry install
```

---

## Testing with pytest

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific test file
pytest tests/test_module.py

# Run specific test class/function
pytest tests/test_module.py::TestClass::test_method

# Run tests matching pattern
pytest -k "fidelity or unitary"

# Show coverage
pytest --cov=src --cov-report=html

# Generate JUnit XML for CI
pytest --junitxml=results.xml
```

### pytest Fixtures

```python
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Fixture providing test data."""
    return np.array([1, 2, 3])

@pytest.fixture(scope="module")
def expensive_resource():
    """Module-scoped fixture (created once per module)."""
    return load_expensive_data()

@pytest.fixture(params=[1, 2, 3])
def parameterized_fixture(request):
    """Parameterized fixture."""
    return request.param * 10
```

### Parameterized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (0, 0),
    (1, 1),
    (4, 2),
])
def test_sqrt(input, expected):
    assert math.isqrt(input) == expected
```

### Numerical Testing

```python
from numpy.testing import assert_allclose, assert_array_equal

# For floating point comparisons
assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)

# For exact comparisons
assert_array_equal(actual, expected)
```

---

## Code Quality Tools

```bash
# Ruff (linting and formatting)
ruff check .              # Lint
ruff check . --fix        # Lint with auto-fix
ruff format .             # Format code

# Black (formatting only)
black .

# mypy (type checking)
mypy src/

# pre-commit
pre-commit install              # Install hooks
pre-commit run --all-files      # Run all checks
```

---

## Git Commands for Research

```bash
# Status
git status -sb

# Stage specific files
git add path/to/file.py

# Commit with message
git commit -m "feat(vqe): add ADAPT-VQE optimizer"

# View history
git log --oneline -10
git log --graph --oneline --all

# Create and switch branch
git checkout -b feature/new-algorithm

# Stash changes
git stash
git stash pop

# Interactive rebase (clean history)
git rebase -i HEAD~3

# Undo last commit (keep changes)
git reset --soft HEAD~1
```

### Commit Message Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `refactor` | Code restructuring |
| `test` | Adding/updating tests |
| `perf` | Performance improvement |
| `exp` | Experimental research code |

---

## Documentation with Sphinx

```bash
# Initialize Sphinx
sphinx-quickstart docs

# Build HTML documentation
cd docs && make html

# Serve locally
python -m http.server -d docs/_build/html 8000
```

### NumPy Docstring Template

```python
def function_name(param1: Type, param2: Type = default) -> ReturnType:
    """
    Short description of function.

    Longer description if needed.

    Parameters
    ----------
    param1 : Type
        Description of param1.
    param2 : Type, optional
        Description of param2. Default is `default`.

    Returns
    -------
    ReturnType
        Description of return value.

    Raises
    ------
    ValueError
        When param1 is invalid.

    Examples
    --------
    >>> function_name(1, 2)
    3
    """
    pass
```

---

## Type Hints

```python
from typing import Optional, List, Dict, Tuple, Callable, Union
import numpy as np
from numpy.typing import NDArray

# Basic types
def func(x: int, y: float = 1.0) -> str:
    pass

# Optional (can be None)
def func(x: Optional[int] = None) -> Optional[str]:
    pass

# Collections
def func(items: List[int]) -> Dict[str, float]:
    pass

# NumPy arrays
def func(arr: NDArray[np.float64]) -> NDArray[np.complex128]:
    pass

# Callable
def func(callback: Callable[[int, int], float]) -> None:
    pass

# Union types
def func(x: Union[int, float]) -> Union[str, None]:
    pass

# Python 3.10+ syntax
def func(x: int | float) -> str | None:
    pass
```

---

## pyproject.toml Sections

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-project"
version = "0.1.0"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.2"]
docs = ["sphinx>=7.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov"

[tool.ruff]
line-length = 88

[tool.mypy]
python_version = "3.11"
```

---

## Reproducibility Patterns

```python
import random
import numpy as np

# Set global seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# Use Generator for better reproducibility
rng = np.random.default_rng(42)
values = rng.random(10)

# Context manager for reproducible blocks
from contextlib import contextmanager

@contextmanager
def reproducible(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
```

---

## Debugging

```python
# Insert breakpoint
breakpoint()  # Python 3.7+

# Or explicitly
import pdb; pdb.set_trace()

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

---

## Profiling

```python
# Simple timing
import time

start = time.perf_counter()
result = expensive_function()
elapsed = time.perf_counter() - start
print(f"Elapsed: {elapsed:.4f}s")

# cProfile
python -m cProfile -s cumtime script.py

# line_profiler (pip install line_profiler)
@profile
def function_to_profile():
    pass
# Run: kernprof -l -v script.py

# memory_profiler (pip install memory_profiler)
from memory_profiler import profile

@profile
def memory_intensive():
    pass
```

---

## GitHub Actions CI Template

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: mypy src/
      - run: pytest --cov
```

---

## Useful Patterns

### Context Manager for Timing

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - start:.4f}s")

with timer("Matrix multiply"):
    result = A @ B
```

### Decorator for Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    return sum(range(n))
```

### Dataclass for Configurations

```python
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class Config:
    n_qubits: int = 4
    n_shots: int = 1024
    seed: Optional[int] = None

    def to_dict(self):
        return asdict(self)
```

---

## Quick Links

| Resource | URL |
|----------|-----|
| pytest docs | https://docs.pytest.org |
| Sphinx docs | https://sphinx-doc.org |
| Ruff docs | https://docs.astral.sh/ruff |
| NumPy docstring guide | https://numpydoc.readthedocs.io |
| Type hints cheatsheet | https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html |
| Good Research Code | https://goodresearch.dev |

---

*Week 209: Advanced Python for Reproducible Research*
