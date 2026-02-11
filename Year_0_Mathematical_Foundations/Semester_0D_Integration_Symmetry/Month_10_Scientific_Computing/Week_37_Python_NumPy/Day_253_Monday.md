# Day 253: Python Refresher — Functions and Classes

## Overview

**Day 253** | **Week 37** | **Month 10: Scientific Computing**

Today we establish Python fluency at the level required for scientific computing. While many students have programming experience, scientific Python demands specific patterns: functional programming with higher-order functions, object-oriented design for physical systems, decorators for code organization, and generators for memory-efficient data processing. These patterns appear throughout quantum simulations and must become second nature.

**Prerequisites:** Basic Python syntax, experience with any programming language
**Outcome:** Write Pythonic code for scientific applications with proper OOP design

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Functions, closures, decorators, generators |
| Afternoon | 3 hours | Practice: OOP design patterns for physics |
| Evening | 2 hours | Lab: Building a QuantumSystem class |

---

## Learning Objectives

By the end of Day 253, you will be able to:

1. **Write functions** with default arguments, *args, **kwargs, and type hints
2. **Use closures and higher-order functions** for flexible function factories
3. **Apply decorators** to add functionality without modifying function code
4. **Design classes** with inheritance, properties, and special methods
5. **Implement generators** for memory-efficient iteration over large datasets
6. **Recognize Pythonic patterns** used in NumPy, SciPy, and physics libraries

---

## Core Content

### 1. Functions: First-Class Objects in Python

In Python, functions are first-class objects—they can be assigned to variables, passed as arguments, and returned from other functions. This enables powerful abstractions.

#### Basic Function Anatomy

```python
def gaussian(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Normalized Gaussian probability density function.

    Parameters
    ----------
    x : float
        Evaluation point
    mu : float, optional
        Mean (default 0.0)
    sigma : float, optional
        Standard deviation (default 1.0)

    Returns
    -------
    float
        Probability density at x
    """
    from math import pi, exp
    normalization = 1 / (sigma * (2 * pi) ** 0.5)
    return normalization * exp(-0.5 * ((x - mu) / sigma) ** 2)
```

**Key features:**
- **Type hints** (`x: float`, `-> float`) document expected types (not enforced at runtime)
- **Default arguments** allow flexible calling: `gaussian(0)` or `gaussian(0, mu=1, sigma=2)`
- **Docstrings** follow NumPy style for scientific code

#### Lambda Functions (Anonymous Functions)

```python
# Short, single-expression functions
square = lambda x: x ** 2
harmonic_potential = lambda x, omega: 0.5 * omega**2 * x**2

# Common in sorting and functional programming
wavefunctions = [(1, 0.5), (2, 0.3), (0, 0.2)]  # (n, amplitude)
sorted_by_n = sorted(wavefunctions, key=lambda state: state[0])
```

#### Variable-Length Arguments

```python
def superposition(*coefficients, normalize=True):
    """
    Create a superposition state from arbitrary coefficients.

    Args:
        *coefficients: Variable number of complex amplitudes
        normalize: Whether to normalize (default True)
    """
    import cmath
    total = sum(abs(c)**2 for c in coefficients)
    if normalize and total > 0:
        norm = total ** 0.5
        return tuple(c / norm for c in coefficients)
    return coefficients

# Usage
state1 = superposition(1, 0)      # |0⟩
state2 = superposition(1, 1)      # (|0⟩ + |1⟩)/√2
state3 = superposition(1, 1, 1)   # Equal superposition of 3 states
```

```python
def hamiltonian_params(**kwargs):
    """Accept arbitrary named parameters for Hamiltonian construction."""
    defaults = {'mass': 1.0, 'omega': 1.0, 'hbar': 1.0}
    defaults.update(kwargs)
    return defaults

params = hamiltonian_params(mass=9.109e-31, omega=1e15)
```

---

### 2. Closures and Function Factories

A **closure** is a function that captures variables from its enclosing scope. This enables powerful **function factories**.

```python
def potential_factory(potential_type: str):
    """
    Factory that returns a potential function based on type.

    The returned function 'closes over' parameters from this scope.
    """
    if potential_type == "harmonic":
        omega = 1.0
        def V(x):
            return 0.5 * omega**2 * x**2
        return V

    elif potential_type == "infinite_well":
        L = 1.0
        def V(x):
            if 0 <= x <= L:
                return 0.0
            return float('inf')
        return V

    elif potential_type == "hydrogen":
        # Coulomb potential (in atomic units)
        def V(r):
            if r == 0:
                return float('-inf')
            return -1.0 / r
        return V

    else:
        raise ValueError(f"Unknown potential: {potential_type}")

# Create specific potential functions
V_harmonic = potential_factory("harmonic")
V_well = potential_factory("infinite_well")

print(V_harmonic(2.0))  # 2.0 (= 0.5 * 1^2 * 2^2)
```

#### Parameterized Function Factory

```python
def make_gaussian_wavepacket(x0: float, p0: float, sigma: float):
    """
    Create a Gaussian wavepacket function with specified parameters.

    Returns a function ψ(x) = A exp(-(x-x0)²/4σ²) exp(ipx/ℏ)
    """
    import cmath
    from math import pi

    # Normalization (assuming ℏ = 1)
    A = (2 * pi * sigma**2) ** (-0.25)

    def psi(x):
        spatial = cmath.exp(-(x - x0)**2 / (4 * sigma**2))
        momentum = cmath.exp(1j * p0 * x)
        return A * spatial * momentum

    return psi

# Create wavepacket centered at x=5, moving right with momentum p=2
wavepacket = make_gaussian_wavepacket(x0=5.0, p0=2.0, sigma=1.0)
print(wavepacket(5.0))  # Maximum amplitude at center
```

---

### 3. Decorators: Modifying Function Behavior

A **decorator** wraps a function to extend its behavior without modifying its source code. This is a key pattern in scientific Python.

#### Basic Decorator Pattern

```python
import functools
import time

def timer(func):
    """Decorator that times function execution."""
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def expensive_computation(n):
    """Simulate expensive calculation."""
    total = 0
    for i in range(n):
        total += i ** 0.5
    return total

result = expensive_computation(1_000_000)
# Prints: expensive_computation took 0.1234 seconds
```

#### Decorator with Arguments

```python
def validate_positive(param_names):
    """
    Decorator factory that validates specified parameters are positive.

    Usage: @validate_positive(['mass', 'omega'])
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map args to names
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name in param_names:
                value = bound.arguments.get(name)
                if value is not None and value <= 0:
                    raise ValueError(f"{name} must be positive, got {value}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_positive(['mass', 'omega'])
def harmonic_energy(n: int, mass: float, omega: float, hbar: float = 1.0) -> float:
    """Compute energy level n of harmonic oscillator."""
    return hbar * omega * (n + 0.5)

print(harmonic_energy(0, mass=1.0, omega=2.0))  # 1.0
# harmonic_energy(0, mass=-1.0, omega=2.0)  # Raises ValueError
```

#### Common Scientific Decorators

```python
def memoize(func):
    """Cache function results for expensive computations."""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    """Compute Fibonacci number (memoized for efficiency)."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# From functools (Python 3.9+)
from functools import cache

@cache
def factorial(n):
    """Cached factorial computation."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

---

### 4. Object-Oriented Programming for Physics

OOP provides natural abstractions for physical systems. A quantum state, an operator, a measurement—each maps cleanly to a class.

#### Designing a Physics Class Hierarchy

```python
from abc import ABC, abstractmethod
from typing import Callable
import cmath

class Potential(ABC):
    """Abstract base class for quantum mechanical potentials."""

    @abstractmethod
    def __call__(self, x: float) -> float:
        """Evaluate potential at position x."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable description."""
        pass

class HarmonicPotential(Potential):
    """Harmonic oscillator potential V(x) = ½mω²x²."""

    def __init__(self, mass: float = 1.0, omega: float = 1.0):
        self.mass = mass
        self.omega = omega
        self.k = mass * omega**2  # Spring constant

    def __call__(self, x: float) -> float:
        return 0.5 * self.k * x**2

    def __str__(self) -> str:
        return f"Harmonic: V(x) = ½({self.k:.3f})x²"

    def classical_turning_point(self, energy: float) -> float:
        """Compute classical turning point for given energy."""
        return (2 * energy / self.k) ** 0.5

class MorsePotential(Potential):
    """Morse potential for diatomic molecules: V(r) = D[1 - e^(-a(r-r_e))]²."""

    def __init__(self, D: float, a: float, r_e: float):
        self.D = D      # Dissociation energy
        self.a = a      # Width parameter
        self.r_e = r_e  # Equilibrium distance

    def __call__(self, r: float) -> float:
        from math import exp
        return self.D * (1 - exp(-self.a * (r - self.r_e)))**2

    def __str__(self) -> str:
        return f"Morse: D={self.D:.3f}, a={self.a:.3f}, r_e={self.r_e:.3f}"
```

#### Special Methods (Dunder Methods)

```python
class ComplexAmplitude:
    """
    Represents a quantum amplitude with magnitude and phase.

    Demonstrates Python special methods for operator overloading.
    """

    def __init__(self, value: complex):
        self.value = complex(value)

    # Representation
    def __repr__(self) -> str:
        return f"ComplexAmplitude({self.value})"

    def __str__(self) -> str:
        r, phi = cmath.polar(self.value)
        return f"|{r:.4f}| ∠ {phi:.4f} rad"

    # Arithmetic operations
    def __add__(self, other: 'ComplexAmplitude') -> 'ComplexAmplitude':
        if isinstance(other, ComplexAmplitude):
            return ComplexAmplitude(self.value + other.value)
        return ComplexAmplitude(self.value + other)

    def __mul__(self, other) -> 'ComplexAmplitude':
        if isinstance(other, ComplexAmplitude):
            return ComplexAmplitude(self.value * other.value)
        return ComplexAmplitude(self.value * other)

    def __abs__(self) -> float:
        return abs(self.value)

    # Comparison
    def __eq__(self, other: 'ComplexAmplitude') -> bool:
        if isinstance(other, ComplexAmplitude):
            return self.value == other.value
        return self.value == other

    # Properties
    @property
    def probability(self) -> float:
        """Born rule: probability = |amplitude|²."""
        return abs(self.value)**2

    @property
    def phase(self) -> float:
        """Extract phase angle."""
        return cmath.phase(self.value)

    def conjugate(self) -> 'ComplexAmplitude':
        """Return complex conjugate."""
        return ComplexAmplitude(self.value.conjugate())

# Usage
a = ComplexAmplitude(1 + 1j)
b = ComplexAmplitude(1 - 1j)
print(f"a = {a}")                 # |1.4142| ∠ 0.7854 rad
print(f"P(a) = {a.probability}")  # 2.0
print(f"a + b = {a + b}")         # |2.0000| ∠ 0.0000 rad
```

---

### 5. Properties and Descriptors

Properties provide computed attributes with getter/setter control.

```python
class QuantumState:
    """
    Represents a normalized quantum state.

    Uses properties to enforce normalization automatically.
    """

    def __init__(self, coefficients: list):
        self._coefficients = None
        self.coefficients = coefficients  # Uses setter

    @property
    def coefficients(self) -> list:
        return self._coefficients.copy()  # Return copy to prevent mutation

    @coefficients.setter
    def coefficients(self, values: list):
        # Automatically normalize on assignment
        norm = sum(abs(c)**2 for c in values) ** 0.5
        if norm == 0:
            raise ValueError("Cannot create zero state")
        self._coefficients = [c / norm for c in values]

    @property
    def dimension(self) -> int:
        return len(self._coefficients)

    @property
    def probabilities(self) -> list:
        """Born rule probabilities for each basis state."""
        return [abs(c)**2 for c in self._coefficients]

    def inner_product(self, other: 'QuantumState') -> complex:
        """Compute ⟨self|other⟩."""
        if self.dimension != other.dimension:
            raise ValueError("Dimension mismatch")
        return sum(a.conjugate() * b for a, b in
                   zip(self._coefficients, other._coefficients))

# Usage
state = QuantumState([1, 1])  # Automatically normalized to [1/√2, 1/√2]
print(state.probabilities)    # [0.5, 0.5]
```

---

### 6. Generators for Memory Efficiency

Generators produce values on-demand, crucial for large datasets.

```python
def eigenstate_generator(n_max: int, x_values):
    """
    Generator yielding harmonic oscillator eigenstates.

    Yields (n, ψ_n(x)) pairs lazily to save memory.
    """
    from math import factorial, pi, exp

    # Precompute Hermite polynomials using recurrence
    def hermite(n, x):
        if n == 0:
            return 1
        elif n == 1:
            return 2 * x
        else:
            H_prev, H_curr = 1, 2 * x
            for k in range(2, n + 1):
                H_prev, H_curr = H_curr, 2 * x * H_curr - 2 * (k - 1) * H_prev
            return H_curr

    for n in range(n_max):
        # Normalization constant
        norm = (1 / (2**n * factorial(n))) ** 0.5 * (1 / pi) ** 0.25

        # Compute eigenstate at all x values
        psi_n = []
        for x in x_values:
            psi_n.append(norm * hermite(n, x) * exp(-x**2 / 2))

        yield n, psi_n

# Memory-efficient iteration
x = [i * 0.1 - 5 for i in range(101)]  # x from -5 to 5
for n, psi in eigenstate_generator(5, x):
    print(f"ψ_{n}: max = {max(psi):.4f}")
```

#### Generator Expressions

```python
# Generator expression (memory-efficient)
squares_gen = (x**2 for x in range(1000000))

# List comprehension (creates full list in memory)
squares_list = [x**2 for x in range(1000000)]

# Use generator for large datasets
def compute_expectation(psi_generator, operator_func):
    """Compute ⟨ψ|Ô|ψ⟩ using generators for memory efficiency."""
    total = sum(psi.conjugate() * operator_func(x) * psi
                for x, psi in psi_generator)
    return total.real
```

---

## Quantum Mechanics Connection

### The QuantumSystem Class Pattern

Scientific Python code organizing quantum simulations typically follows this pattern:

```python
class QuantumSystem:
    """
    Base class for quantum mechanical systems.

    This pattern appears throughout computational physics:
    - QuTiP's Qobj
    - PennyLane's quantum circuits
    - Qiskit's QuantumCircuit
    """

    def __init__(self, dimension: int, hbar: float = 1.0):
        self.dimension = dimension
        self.hbar = hbar
        self._hamiltonian = None
        self._eigenstates = None
        self._energies = None

    @property
    def hamiltonian(self):
        """Lazy evaluation of Hamiltonian matrix."""
        if self._hamiltonian is None:
            self._hamiltonian = self._build_hamiltonian()
        return self._hamiltonian

    @abstractmethod
    def _build_hamiltonian(self):
        """Subclasses implement specific Hamiltonian construction."""
        pass

    def solve(self):
        """Solve eigenvalue problem (to be implemented with NumPy)."""
        # Will use np.linalg.eigh in Day 256
        pass

    def evolve(self, state, t):
        """Time evolution (to be implemented with SciPy)."""
        # Will use scipy.linalg.expm in Week 38
        pass
```

This design pattern:
1. Encapsulates physical parameters
2. Uses lazy evaluation for expensive computations
3. Provides clear interface for eigenvalue solving and time evolution
4. Enables subclassing for specific systems (harmonic oscillator, particle in box, etc.)

---

## Worked Examples

### Example 1: Function Factory for Quantum Operators

**Problem:** Create a factory that returns position or momentum operator functions.

```python
def operator_factory(operator_type: str, hbar: float = 1.0):
    """
    Factory for quantum operator functions.

    Parameters
    ----------
    operator_type : str
        'position' or 'momentum'
    hbar : float
        Reduced Planck constant

    Returns
    -------
    Callable
        Function that applies the operator
    """
    if operator_type == 'position':
        # x̂ψ(x) = xψ(x)
        def x_operator(psi, x):
            return x * psi
        return x_operator

    elif operator_type == 'momentum':
        # p̂ψ(x) = -iℏ dψ/dx (approximate with finite difference)
        def p_operator(psi_values, x_values, dx):
            import cmath
            result = []
            for i in range(1, len(psi_values) - 1):
                dpsi_dx = (psi_values[i+1] - psi_values[i-1]) / (2 * dx)
                result.append(-1j * hbar * dpsi_dx)
            return result
        return p_operator

    else:
        raise ValueError(f"Unknown operator: {operator_type}")

# Usage
x_hat = operator_factory('position')
p_hat = operator_factory('momentum', hbar=1.054e-34)
```

### Example 2: Decorator for Physical Units

```python
def atomic_units(func):
    """
    Decorator that converts SI inputs to atomic units for computation
    and converts results back to SI.

    Atomic units: ℏ = mₑ = e = 4πε₀ = 1
    """
    # Conversion factors
    hbar_SI = 1.054571817e-34  # J·s
    m_e_SI = 9.1093837015e-31  # kg
    e_SI = 1.602176634e-19     # C
    a0_SI = 5.29177210903e-11  # m (Bohr radius)
    E_h_SI = 4.3597447222071e-18  # J (Hartree)

    @functools.wraps(func)
    def wrapper(energy_SI=None, length_SI=None, **kwargs):
        # Convert inputs to atomic units
        au_kwargs = kwargs.copy()
        if energy_SI is not None:
            au_kwargs['energy'] = energy_SI / E_h_SI
        if length_SI is not None:
            au_kwargs['length'] = length_SI / a0_SI

        # Compute in atomic units
        result_au = func(**au_kwargs)

        # Convert back to SI (assuming energy output)
        return result_au * E_h_SI

    return wrapper

@atomic_units
def hydrogen_ground_state_energy(Z: float = 1.0, **kwargs) -> float:
    """Compute hydrogen-like atom ground state energy in atomic units."""
    return -0.5 * Z**2

# Returns energy in Joules
E_ground = hydrogen_ground_state_energy(Z=1.0)
print(f"H ground state: {E_ground:.6e} J = {E_ground/1.602e-19:.4f} eV")
```

### Example 3: Generator for Schrödinger Equation Snapshots

```python
def schrodinger_evolution(psi_initial, V, x_grid, dt, n_steps, hbar=1.0, m=1.0):
    """
    Generator yielding time evolution snapshots.

    Uses split-operator method (simplified for illustration).
    Proper implementation in Week 38.
    """
    import cmath

    dx = x_grid[1] - x_grid[0]
    psi = psi_initial.copy()

    for step in range(n_steps):
        # Yield current state
        yield step * dt, psi.copy()

        # Simplified evolution (proper method in Week 38)
        # This is a placeholder showing the generator pattern
        for i in range(len(psi)):
            # Apply potential phase
            psi[i] *= cmath.exp(-1j * V(x_grid[i]) * dt / (2 * hbar))

        # Note: Momentum space evolution would go here

        for i in range(len(psi)):
            psi[i] *= cmath.exp(-1j * V(x_grid[i]) * dt / (2 * hbar))

# Usage pattern
x = [i * 0.1 - 5 for i in range(101)]
psi_0 = [cmath.exp(-(xi**2)/2) for xi in x]  # Gaussian initial state
V = lambda x: 0.5 * x**2  # Harmonic potential

for t, psi_t in schrodinger_evolution(psi_0, V, x, dt=0.01, n_steps=100):
    if t % 0.1 < 0.01:  # Print every 0.1 time units
        print(f"t = {t:.2f}: max|ψ| = {max(abs(p) for p in psi_t):.4f}")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Write a function `commutator(A_func, B_func, psi, x)` that computes [A, B]ψ = ABψ - BAψ numerically for two operator functions.

**Problem 2:** Create a decorator `@check_normalization` that verifies a returned quantum state is normalized (sum of |coefficients|² = 1) and raises a warning if not.

**Problem 3:** Implement a `Particle` class with position, momentum, and mass attributes, including a method to compute kinetic energy.

### Intermediate

**Problem 4:** Design a class hierarchy for quantum gates: base class `Gate` with subclasses `PauliX`, `PauliY`, `PauliZ`, `Hadamard`. Each should have a `matrix` property and an `apply(state)` method.

**Problem 5:** Write a generator `prime_quantum_numbers(n_max)` that yields all valid (n, l, m) tuples for a hydrogen atom up to principal quantum number n_max.

**Problem 6:** Create a function factory `make_spherical_harmonic(l, m)` that returns a function Y_l^m(θ, φ) using the formula for spherical harmonics.

### Challenging

**Problem 7:** Implement a caching decorator `@quantum_memoize` that works with complex-valued function arguments by converting them to a hashable form.

**Problem 8:** Design a `QuantumRegister` class for multi-qubit systems with methods: `tensor_product(other)`, `partial_trace(qubit_indices)`, and `measure(qubit)`.

**Problem 9:** Write a context manager class `AtomicUnits` that temporarily sets global unit conventions and restores them on exit.

---

## Computational Lab

### Building the QuantumSystem Base Class

Today's lab implements the foundation we'll build on all week.

```python
"""
Day 253 Computational Lab: QuantumSystem Base Class
====================================================

This module establishes the class hierarchy for quantum simulations.
We'll add numerical methods in subsequent days.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Tuple
import cmath
from math import pi, factorial
import functools

# ============================================================
# DECORATORS FOR QUANTUM COMPUTING
# ============================================================

def timer(func):
    """Time function execution - useful for benchmarking."""
    import time
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__}: {elapsed:.4f} s")
        return result
    return wrapper

def validate_normalized(func):
    """Verify returned state is normalized."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if hasattr(result, '__iter__'):
            norm_sq = sum(abs(c)**2 for c in result)
            if abs(norm_sq - 1.0) > 1e-10:
                print(f"[WARNING] State not normalized: |ψ|² = {norm_sq:.10f}")
        return result
    return wrapper

# ============================================================
# ABSTRACT BASE CLASSES
# ============================================================

class Potential(ABC):
    """Abstract base class for potentials."""

    @abstractmethod
    def __call__(self, x: float) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

class HarmonicPotential(Potential):
    """V(x) = ½mω²x²"""

    def __init__(self, m: float = 1.0, omega: float = 1.0):
        self.m = m
        self.omega = omega
        self.k = m * omega**2

    def __call__(self, x: float) -> float:
        return 0.5 * self.k * x**2

    def name(self) -> str:
        return f"Harmonic (ω={self.omega})"

    def ground_state_energy(self, hbar: float = 1.0) -> float:
        return 0.5 * hbar * self.omega

    def classical_amplitude(self, E: float) -> float:
        """Classical turning point for energy E."""
        return (2 * E / self.k) ** 0.5

class InfiniteWell(Potential):
    """Infinite square well: V=0 for 0<x<L, V=∞ otherwise."""

    def __init__(self, L: float = 1.0):
        self.L = L

    def __call__(self, x: float) -> float:
        if 0 < x < self.L:
            return 0.0
        return float('inf')

    def name(self) -> str:
        return f"Infinite Well (L={self.L})"

    def energy_level(self, n: int, m: float = 1.0, hbar: float = 1.0) -> float:
        """Energy of level n (n = 1, 2, 3, ...)"""
        return (n * pi * hbar)**2 / (2 * m * self.L**2)

# ============================================================
# QUANTUM STATE CLASS
# ============================================================

class QuantumState:
    """
    Represents a quantum state as a vector of amplitudes.

    In the computational basis, |ψ⟩ = Σᵢ cᵢ|i⟩
    """

    def __init__(self, coefficients: List[complex], normalize: bool = True):
        """
        Initialize quantum state.

        Parameters
        ----------
        coefficients : list of complex
            Amplitudes in computational basis
        normalize : bool
            Whether to normalize automatically
        """
        self._coefficients = [complex(c) for c in coefficients]
        if normalize:
            self._normalize()

    def _normalize(self):
        """Normalize state to unit length."""
        norm = sum(abs(c)**2 for c in self._coefficients) ** 0.5
        if norm > 0:
            self._coefficients = [c / norm for c in self._coefficients]

    @property
    def dim(self) -> int:
        return len(self._coefficients)

    @property
    def coefficients(self) -> List[complex]:
        return self._coefficients.copy()

    @property
    def probabilities(self) -> List[float]:
        """Born rule: Pᵢ = |cᵢ|²"""
        return [abs(c)**2 for c in self._coefficients]

    def inner_product(self, other: 'QuantumState') -> complex:
        """Compute ⟨self|other⟩."""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return sum(a.conjugate() * b
                   for a, b in zip(self._coefficients, other._coefficients))

    def expectation(self, operator_matrix: List[List[complex]]) -> float:
        """
        Compute ⟨ψ|Ô|ψ⟩ for operator given as matrix.

        This is a placeholder - proper NumPy implementation in Day 256.
        """
        # Apply operator: Ô|ψ⟩
        result = []
        for i in range(self.dim):
            val = sum(operator_matrix[i][j] * self._coefficients[j]
                      for j in range(self.dim))
            result.append(val)

        # Compute ⟨ψ|result⟩
        expectation = sum(self._coefficients[i].conjugate() * result[i]
                         for i in range(self.dim))
        return expectation.real

    def __repr__(self) -> str:
        return f"QuantumState({self._coefficients})"

    def __str__(self) -> str:
        terms = []
        for i, c in enumerate(self._coefficients):
            if abs(c) > 1e-10:
                r, phi = cmath.polar(c)
                if abs(phi) < 1e-10:
                    terms.append(f"{r:.4f}|{i}⟩")
                else:
                    terms.append(f"{c:.4f}|{i}⟩")
        return " + ".join(terms) if terms else "0"

# ============================================================
# QUANTUM SYSTEM BASE CLASS
# ============================================================

class QuantumSystem(ABC):
    """
    Abstract base class for quantum mechanical systems.

    Subclasses implement specific Hamiltonians.
    Numerical methods (eigensolve, evolve) added in Days 254-256.
    """

    def __init__(self,
                 potential: Potential,
                 x_min: float = -10.0,
                 x_max: float = 10.0,
                 n_points: int = 100,
                 mass: float = 1.0,
                 hbar: float = 1.0):
        """
        Initialize quantum system.

        Parameters
        ----------
        potential : Potential
            Potential energy function
        x_min, x_max : float
            Spatial domain boundaries
        n_points : int
            Number of grid points for discretization
        mass : float
            Particle mass
        hbar : float
            Reduced Planck constant
        """
        self.potential = potential
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points
        self.mass = mass
        self.hbar = hbar

        # Derived quantities
        self.dx = (x_max - x_min) / (n_points - 1)
        self.x_grid = [x_min + i * self.dx for i in range(n_points)]

        # Lazy-evaluated properties
        self._hamiltonian = None
        self._energies = None
        self._eigenstates = None

    @property
    def hamiltonian(self) -> List[List[complex]]:
        """Return Hamiltonian matrix (lazy evaluation)."""
        if self._hamiltonian is None:
            self._hamiltonian = self._build_hamiltonian()
        return self._hamiltonian

    @abstractmethod
    def _build_hamiltonian(self) -> List[List[complex]]:
        """Build the Hamiltonian matrix. Subclasses implement this."""
        pass

    def potential_at_grid(self) -> List[float]:
        """Evaluate potential at all grid points."""
        return [self.potential(x) for x in self.x_grid]

    def __str__(self) -> str:
        return (f"QuantumSystem(\n"
                f"  potential: {self.potential.name()}\n"
                f"  domain: [{self.x_min}, {self.x_max}]\n"
                f"  grid points: {self.n_points}\n"
                f"  mass: {self.mass}, ℏ: {self.hbar}\n"
                f")")

class ParticleInBox(QuantumSystem):
    """
    Particle in 1D infinite square well.

    Analytic solutions known: Eₙ = n²π²ℏ²/(2mL²)
    """

    def __init__(self, L: float = 1.0, n_points: int = 100,
                 mass: float = 1.0, hbar: float = 1.0):
        well = InfiniteWell(L)
        super().__init__(well, 0, L, n_points, mass, hbar)
        self.L = L

    def _build_hamiltonian(self) -> List[List[complex]]:
        """
        Build Hamiltonian using finite difference.

        H = -ℏ²/(2m) d²/dx² + V(x)

        Finite difference: d²ψ/dx² ≈ (ψᵢ₊₁ - 2ψᵢ + ψᵢ₋₁)/dx²
        """
        n = self.n_points
        H = [[0.0 + 0j for _ in range(n)] for _ in range(n)]

        kinetic_coeff = -self.hbar**2 / (2 * self.mass * self.dx**2)

        for i in range(n):
            # Diagonal: kinetic (-2) + potential
            H[i][i] = -2 * kinetic_coeff + self.potential(self.x_grid[i])

            # Off-diagonal: kinetic (1)
            if i > 0:
                H[i][i-1] = kinetic_coeff
            if i < n - 1:
                H[i][i+1] = kinetic_coeff

        return H

    def analytic_energy(self, n: int) -> float:
        """Exact energy level (n = 1, 2, 3, ...)"""
        return (n * pi * self.hbar)**2 / (2 * self.mass * self.L**2)

    def analytic_wavefunction(self, n: int, x: float) -> float:
        """Exact wavefunction ψₙ(x) = √(2/L) sin(nπx/L)"""
        from math import sin
        return (2 / self.L)**0.5 * sin(n * pi * x / self.L)

# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 253: Python Refresher - QuantumSystem Foundation")
    print("=" * 60)

    # Test potentials
    print("\n--- Potential Classes ---")
    V_harm = HarmonicPotential(omega=2.0)
    print(f"{V_harm.name()}: V(1.0) = {V_harm(1.0)}")
    print(f"Ground state energy: {V_harm.ground_state_energy()}")

    V_well = InfiniteWell(L=2.0)
    print(f"\n{V_well.name()}: E_1 = {V_well.energy_level(1):.6f}")
    print(f"E_2 = {V_well.energy_level(2):.6f}")

    # Test quantum state
    print("\n--- Quantum State ---")
    psi = QuantumState([1, 1j])  # (|0⟩ + i|1⟩)/√2
    print(f"State: {psi}")
    print(f"Probabilities: {psi.probabilities}")

    phi = QuantumState([1, 0])  # |0⟩
    print(f"⟨ψ|φ⟩ = {psi.inner_product(phi)}")

    # Test quantum system
    print("\n--- Particle in Box ---")
    system = ParticleInBox(L=1.0, n_points=50)
    print(system)

    print("\nAnalytic energies (n=1,2,3):")
    for n in [1, 2, 3]:
        print(f"  E_{n} = {system.analytic_energy(n):.6f}")

    # Hamiltonian structure (just show size)
    H = system.hamiltonian
    print(f"\nHamiltonian: {len(H)}×{len(H[0])} matrix")
    print(f"H[25,25] (diagonal) = {H[25][25]:.6f}")
    print(f"H[25,26] (off-diag) = {H[25][26]:.6f}")

    print("\n" + "=" * 60)
    print("Lab complete! NumPy implementation continues on Day 254.")
    print("=" * 60)
```

**Expected Output:**
```
============================================================
Day 253: Python Refresher - QuantumSystem Foundation
============================================================

--- Potential Classes ---
Harmonic (ω=2.0): V(1.0) = 2.0
Ground state energy: 1.0

Infinite Well (L=2.0): E_1 = 1.233701
E_2 = 4.934802

--- Quantum State ---
State: 0.7071|0⟩ + 0.7071j|1⟩
Probabilities: [0.5, 0.5]
⟨ψ|φ⟩ = (0.7071067811865476+0j)

--- Particle in Box ---
QuantumSystem(
  potential: Infinite Well (L=1.0)
  domain: [0, 1.0]
  grid points: 50
  mass: 1.0, ℏ: 1.0
)

Analytic energies (n=1,2,3):
  E_1 = 4.934802
  E_2 = 19.739209
  E_3 = 44.413220

Hamiltonian: 50×50 matrix
H[25,25] (diagonal) = 4802.000000
H[25,26] (off-diag) = -2401.000000

============================================================
Lab complete! NumPy implementation continues on Day 254.
============================================================
```

---

## Summary

### Key Formulas and Concepts

| Concept | Python Pattern | QM Application |
|---------|---------------|----------------|
| First-class functions | `f = lambda x: x**2` | Operator definitions |
| Closures | `def factory(param): return lambda x: ...` | Parameterized potentials |
| Decorators | `@decorator` above function | Validation, timing, caching |
| Classes | `class System:` with `__init__`, methods | Physical systems |
| Properties | `@property` for computed attributes | Lazy Hamiltonian building |
| Generators | `yield` for lazy iteration | Large state spaces |

### Main Takeaways

1. **Functions are objects** — pass them, return them, compose them
2. **Closures capture state** — function factories are powerful for physics
3. **Decorators modify behavior** — validation, timing, caching without code changes
4. **OOP models physics naturally** — systems, states, operators as classes
5. **Generators save memory** — essential for large Hilbert spaces
6. **Type hints document intent** — not enforced but clarify usage

---

## Daily Checklist

- [ ] Can write functions with `*args`, `**kwargs`, and type hints
- [ ] Understand closures and can create function factories
- [ ] Can write and apply decorators (with and without arguments)
- [ ] Can design classes with inheritance, properties, and special methods
- [ ] Understand generators and when to use them over lists
- [ ] Completed all practice problems
- [ ] Ran the computational lab successfully
- [ ] Can explain how these patterns apply to quantum simulations

---

## Preview: Day 254

Tomorrow we dive into **NumPy arrays** — the foundation of all numerical Python. We'll learn:
- Creating arrays efficiently with `np.zeros`, `np.linspace`, `np.arange`
- Powerful indexing and slicing syntax
- Memory layout and views vs copies
- Converting our pure-Python classes to use NumPy arrays

The transition from Python lists to NumPy arrays typically provides **100x speedup** for numerical operations. This is why NumPy is non-negotiable for computational physics.

---

*"Programs must be written for people to read, and only incidentally for machines to execute."*
— Harold Abelson, Structure and Interpretation of Computer Programs
