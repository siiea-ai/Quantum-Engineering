"""
Reproducible Research Utilities for Quantum Computing
=====================================================

This module provides essential utilities for ensuring reproducibility
in quantum computing research experiments.

Author: Quantum Engineering PhD Program
Week 209: Advanced Python for Reproducible Research
"""

import os
import random
import json
import hashlib
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from functools import wraps

import numpy as np


# =============================================================================
# Random Seed Management
# =============================================================================

def set_global_seed(seed: int) -> None:
    """
    Set random seeds for all commonly used libraries.

    Parameters
    ----------
    seed : int
        The seed value to use for random number generation.

    Examples
    --------
    >>> set_global_seed(42)
    >>> np.random.rand()  # Will always produce the same value
    0.3745401188473625
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Try to set seeds for optional dependencies
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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

    Saves the current random state, sets new seeds, and restores
    the original state after the block completes.

    Parameters
    ----------
    seed : int
        Seed for random number generation within the context.

    Yields
    ------
    None

    Examples
    --------
    >>> with reproducible_context(42):
    ...     result = np.random.rand(3)
    >>> print(result)
    [0.37454012 0.95071431 0.73199394]
    """
    # Save current states
    py_state = random.getstate()
    np_state = np.random.get_state()

    # Set seeds
    set_global_seed(seed)

    try:
        yield
    finally:
        # Restore original states
        random.setstate(py_state)
        np.random.set_state(np_state)


def create_seeded_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a numpy random Generator with optional seed.

    Using Generator objects is preferred over global np.random
    for better reproducibility and thread safety.

    Parameters
    ----------
    seed : int, optional
        Seed for the generator. If None, uses entropy from OS.

    Returns
    -------
    np.random.Generator
        A seeded random number generator.

    Examples
    --------
    >>> rng = create_seeded_rng(42)
    >>> rng.random(3)
    array([0.77395605, 0.43887844, 0.85859792])
    """
    return np.random.default_rng(seed)


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration container for reproducible experiments.

    Attributes
    ----------
    name : str
        Human-readable experiment name.
    seed : int
        Random seed for reproducibility.
    n_qubits : int
        Number of qubits in the system.
    n_shots : int
        Number of measurement shots.
    optimizer : str
        Classical optimizer to use.
    max_iterations : int
        Maximum optimization iterations.
    tolerance : float
        Convergence tolerance.
    created_at : str
        ISO timestamp of configuration creation.
    metadata : dict
        Additional experiment metadata.
    """
    name: str
    seed: int = 42
    n_qubits: int = 4
    n_shots: int = 1024
    optimizer: str = "COBYLA"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    created_at: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def compute_hash(self) -> str:
        """Compute a hash of the configuration for tracking."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


# =============================================================================
# Results Tracking
# =============================================================================

@dataclass
class ExperimentResult:
    """
    Container for experiment results with metadata.

    Attributes
    ----------
    config : ExperimentConfig
        The configuration used for this experiment.
    optimal_value : float
        The optimal value found (e.g., ground state energy).
    optimal_params : np.ndarray
        The optimal parameters found.
    convergence_history : list
        History of objective values during optimization.
    runtime_seconds : float
        Total runtime in seconds.
    n_function_evals : int
        Number of function evaluations.
    success : bool
        Whether the optimization converged successfully.
    """
    config: ExperimentConfig
    optimal_value: float
    optimal_params: np.ndarray
    convergence_history: list
    runtime_seconds: float
    n_function_evals: int
    success: bool

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'config': self.config.to_dict(),
            'optimal_value': self.optimal_value,
            'optimal_params': self.optimal_params.tolist(),
            'convergence_history': self.convergence_history,
            'runtime_seconds': self.runtime_seconds,
            'n_function_evals': self.n_function_evals,
            'success': self.success,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Timing and Profiling Utilities
# =============================================================================

@contextmanager
def timer(name: str = "Block"):
    """
    Context manager for timing code blocks.

    Parameters
    ----------
    name : str
        Description of the code block being timed.

    Yields
    ------
    dict
        Dictionary that will contain 'elapsed' key after block completes.

    Examples
    --------
    >>> with timer("Matrix multiplication") as t:
    ...     result = np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)
    Matrix multiplication: 0.1234 seconds
    >>> print(t['elapsed'])
    0.1234
    """
    import time
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result['elapsed'] = elapsed
        print(f"{name}: {elapsed:.4f} seconds")


def timed(func: Callable) -> Callable:
    """
    Decorator for timing function calls.

    Parameters
    ----------
    func : callable
        Function to time.

    Returns
    -------
    callable
        Wrapped function that prints timing information.

    Examples
    --------
    >>> @timed
    ... def slow_function(n):
    ...     return sum(range(n))
    >>> slow_function(1000000)
    slow_function: 0.0234 seconds
    499999500000
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.4f} seconds")
        return result
    return wrapper


# =============================================================================
# Caching for Expensive Computations
# =============================================================================

def cache_result(cache_dir: Path = Path(".cache")):
    """
    Decorator for caching function results to disk.

    Parameters
    ----------
    cache_dir : Path
        Directory to store cached results.

    Returns
    -------
    callable
        Decorator function.

    Examples
    --------
    >>> @cache_result(Path(".cache"))
    ... def expensive_computation(n):
    ...     return np.random.rand(n, n)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create hash of arguments
            key_data = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            key_hash = hashlib.sha256(
                json.dumps(key_data).encode()
            ).hexdigest()[:16]

            cache_file = cache_dir / f"{func.__name__}_{key_hash}.npy"

            if cache_file.exists():
                print(f"Loading cached result for {func.__name__}")
                return np.load(cache_file, allow_pickle=True)

            result = func(*args, **kwargs)

            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, result, allow_pickle=True)
            print(f"Cached result for {func.__name__}")

            return result
        return wrapper
    return decorator


# =============================================================================
# Demo and Testing
# =============================================================================

def demo_reproducibility():
    """Demonstrate reproducibility features."""
    print("=" * 60)
    print("Reproducibility Demonstration")
    print("=" * 60)

    # Demo 1: Global seed
    print("\n1. Global Seed Management")
    print("-" * 40)

    set_global_seed(42)
    vals1 = np.random.rand(3)
    print(f"First run with seed 42: {vals1}")

    set_global_seed(42)
    vals2 = np.random.rand(3)
    print(f"Second run with seed 42: {vals2}")
    print(f"Values match: {np.allclose(vals1, vals2)}")

    # Demo 2: Reproducible context
    print("\n2. Reproducible Context Manager")
    print("-" * 40)

    # Generate some random numbers first
    np.random.seed(0)
    before = np.random.rand()
    print(f"Before context: {before}")

    with reproducible_context(42):
        inside = np.random.rand()
        print(f"Inside context (seed 42): {inside}")

    # State should be restored
    after = np.random.rand()
    print(f"After context (state restored): {after}")

    # Demo 3: Seeded RNG
    print("\n3. Seeded Random Generator")
    print("-" * 40)

    rng1 = create_seeded_rng(42)
    rng2 = create_seeded_rng(42)

    vals1 = rng1.random(3)
    vals2 = rng2.random(3)
    print(f"RNG 1: {vals1}")
    print(f"RNG 2: {vals2}")
    print(f"Values match: {np.allclose(vals1, vals2)}")

    # Demo 4: Experiment configuration
    print("\n4. Experiment Configuration")
    print("-" * 40)

    config = ExperimentConfig(
        name="VQE_H2_Ground_State",
        seed=42,
        n_qubits=4,
        n_shots=1024,
        metadata={"molecule": "H2", "basis": "STO-3G"}
    )

    print(f"Config name: {config.name}")
    print(f"Config hash: {config.compute_hash()}")
    print(f"Config dict: {config.to_dict()}")

    # Demo 5: Timing
    print("\n5. Timing Utilities")
    print("-" * 40)

    with timer("Matrix operation"):
        _ = np.random.rand(500, 500) @ np.random.rand(500, 500)

    @timed
    def sample_function(n):
        return sum(range(n))

    result = sample_function(1_000_000)
    print(f"Result: {result}")

    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_reproducibility()
