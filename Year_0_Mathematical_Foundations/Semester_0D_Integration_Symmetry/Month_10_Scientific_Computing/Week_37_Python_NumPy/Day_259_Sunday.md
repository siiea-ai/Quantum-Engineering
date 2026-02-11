# Day 259: Week 37 Review — Python & NumPy Integration

## Overview

**Day 259** | **Week 37** | **Month 10: Scientific Computing**

Today we synthesize everything from Week 37 into a comprehensive project and review. We'll build a complete quantum mechanics simulation from scratch, integrating Python classes, NumPy arrays, vectorized operations, linear algebra, random sampling, and file I/O. This capstone demonstrates mastery of computational foundations.

**Prerequisites:** Days 253-258 (complete Week 37)
**Outcome:** Integrate all week's skills into production-quality physics code

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Capstone project: Complete quantum simulator |
| Afternoon | 3 hours | Review exercises and problem solving |
| Evening | 2 hours | Self-assessment and Week 38 preparation |

---

## Week 37 Summary

### Key Concepts Covered

| Day | Topic | Key Skills |
|-----|-------|------------|
| 253 | Python Refresher | Functions, classes, decorators, generators |
| 254 | NumPy Arrays | Creation, indexing, slicing, dtypes |
| 255 | Vectorization | Broadcasting, ufuncs, eliminating loops |
| 256 | Linear Algebra | Eigenvalues, solve, SVD, QR |
| 257 | Random Numbers | Distributions, Monte Carlo, measurement |
| 258 | File I/O | NPY, HDF5, profiling, optimization |

### Essential NumPy Operations

```python
import numpy as np

# Array creation
x = np.linspace(-10, 10, 1000)
zeros = np.zeros((N, M))
identity = np.eye(N)
diagonal = np.diag(values)

# Indexing
a[0], a[-1], a[2:5], a[::2], a[::-1]
matrix[i, j], matrix[i, :], matrix[:, j]
a[mask], a[indices]  # Boolean and fancy indexing

# Vectorization
np.sin(x), np.exp(x), np.abs(x)
a * b, a @ b  # Element-wise vs matrix

# Broadcasting
row[np.newaxis, :] * col[:, np.newaxis]

# Linear algebra
np.linalg.eigh(H)
np.linalg.solve(A, b)
np.linalg.norm(v)

# Random
rng = np.random.default_rng(seed)
rng.random(N), rng.normal(0, 1, N)
rng.choice(N, size=M, p=probabilities)

# I/O
np.save('file.npy', array)
np.savez('file.npz', a=a, b=b)
np.load('file.npz')
```

---

## Capstone Project: Quantum Harmonic Oscillator Simulator

Build a complete, modular quantum mechanics simulation package.

```python
"""
Week 37 Capstone: Complete Quantum Harmonic Oscillator Simulator
================================================================

This module demonstrates all NumPy skills from Week 37:
- OOP design (Day 253)
- Array operations (Day 254)
- Vectorization (Day 255)
- Eigenvalue solving (Day 256)
- Monte Carlo sampling (Day 257)
- File I/O (Day 258)
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass, field
import h5py
import time

# ============================================================
# CONFIGURATION AND DATA CLASSES
# ============================================================

@dataclass
class QuantumSystemConfig:
    """Configuration for quantum system simulation."""
    x_min: float = -10.0
    x_max: float = 10.0
    n_grid: int = 500
    n_states: int = 20
    mass: float = 1.0
    hbar: float = 1.0

    @property
    def grid(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n_grid)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.n_grid - 1)

@dataclass
class SimulationResults:
    """Container for simulation results."""
    energies: np.ndarray
    wavefunctions: np.ndarray
    grid: np.ndarray
    potential: np.ndarray
    config: QuantumSystemConfig
    computation_time: float = 0.0
    metadata: Dict = field(default_factory=dict)

# ============================================================
# POTENTIAL FUNCTIONS
# ============================================================

def harmonic_potential(x: np.ndarray, omega: float = 1.0,
                       mass: float = 1.0) -> np.ndarray:
    """Harmonic oscillator: V(x) = ½mω²x²"""
    return 0.5 * mass * omega**2 * x**2

def double_well_potential(x: np.ndarray, barrier: float = 2.0,
                          separation: float = 3.0) -> np.ndarray:
    """Double well: V(x) = ¼(x²-a²)²"""
    a = separation
    return 0.25 * (x**2 - a**2)**2 / a**4 * barrier

def morse_potential(x: np.ndarray, D: float = 10.0,
                    a: float = 1.0) -> np.ndarray:
    """Morse potential for molecules."""
    return D * (1 - np.exp(-a * x))**2 - D

def anharmonic_potential(x: np.ndarray, omega: float = 1.0,
                         lambda_4: float = 0.1) -> np.ndarray:
    """Anharmonic: V(x) = ½ω²x² + λx⁴"""
    return 0.5 * omega**2 * x**2 + lambda_4 * x**4

# ============================================================
# QUANTUM SYSTEM CLASS
# ============================================================

class QuantumSystem:
    """
    Complete 1D quantum system simulator.

    Integrates all Week 37 skills:
    - Class design and properties
    - Vectorized array operations
    - Eigenvalue computation
    - Statistical analysis
    - File persistence
    """

    def __init__(self, config: QuantumSystemConfig,
                 potential: Callable[[np.ndarray], np.ndarray]):
        """
        Initialize quantum system.

        Parameters
        ----------
        config : QuantumSystemConfig
            System configuration
        potential : callable
            Potential function V(x)
        """
        self.config = config
        self.potential_func = potential

        # Computed properties (lazy initialization)
        self._hamiltonian: Optional[np.ndarray] = None
        self._energies: Optional[np.ndarray] = None
        self._wavefunctions: Optional[np.ndarray] = None
        self._solved: bool = False

    # -------------------- Properties --------------------

    @property
    def x(self) -> np.ndarray:
        """Spatial grid."""
        return self.config.grid

    @property
    def dx(self) -> float:
        """Grid spacing."""
        return self.config.dx

    @property
    def potential(self) -> np.ndarray:
        """Potential on grid."""
        return self.potential_func(self.x)

    @property
    def hamiltonian(self) -> np.ndarray:
        """Hamiltonian matrix (lazy evaluation)."""
        if self._hamiltonian is None:
            self._hamiltonian = self._build_hamiltonian()
        return self._hamiltonian

    @property
    def energies(self) -> np.ndarray:
        """Energy eigenvalues."""
        if not self._solved:
            self.solve()
        return self._energies

    @property
    def wavefunctions(self) -> np.ndarray:
        """Normalized eigenstates (columns)."""
        if not self._solved:
            self.solve()
        return self._wavefunctions

    # -------------------- Core Methods --------------------

    def _build_hamiltonian(self) -> np.ndarray:
        """
        Build Hamiltonian matrix using finite difference.

        H = -ℏ²/(2m) d²/dx² + V(x)

        Uses second-order central difference for kinetic energy.
        """
        N = self.config.n_grid
        dx = self.dx
        m = self.config.mass
        hbar = self.config.hbar

        # Kinetic energy coefficient
        T_coeff = hbar**2 / (2 * m * dx**2)

        # Build tridiagonal matrix (vectorized)
        V = self.potential
        diag = 2 * T_coeff + V
        off_diag = np.full(N - 1, -T_coeff)

        H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

        return H

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the time-independent Schrödinger equation.

        Returns
        -------
        energies : ndarray
            Energy eigenvalues (sorted)
        wavefunctions : ndarray
            Normalized eigenstates (columns)
        """
        # Solve eigenvalue problem
        energies, wavefunctions = np.linalg.eigh(self.hamiltonian)

        # Keep requested number of states
        n_states = self.config.n_states
        self._energies = energies[:n_states]
        self._wavefunctions = wavefunctions[:, :n_states]

        # Normalize wavefunctions
        for n in range(n_states):
            norm = np.sqrt(np.sum(np.abs(self._wavefunctions[:, n])**2) * self.dx)
            self._wavefunctions[:, n] /= norm

        self._solved = True
        return self._energies, self._wavefunctions

    # -------------------- Quantum Observables --------------------

    def expectation_value(self, operator: np.ndarray, state: int = 0) -> float:
        """
        Compute ⟨n|Ô|n⟩ for diagonal operators.

        Parameters
        ----------
        operator : ndarray
            Operator values on grid (diagonal representation)
        state : int
            State index

        Returns
        -------
        float
            Expectation value
        """
        psi = self.wavefunctions[:, state]
        return np.real(np.sum(np.conj(psi) * operator * psi) * self.dx)

    def position_expectation(self, state: int = 0) -> float:
        """Compute ⟨x⟩ for state n."""
        return self.expectation_value(self.x, state)

    def position_uncertainty(self, state: int = 0) -> float:
        """Compute Δx = √(⟨x²⟩ - ⟨x⟩²)."""
        x_avg = self.position_expectation(state)
        x2_avg = self.expectation_value(self.x**2, state)
        return np.sqrt(x2_avg - x_avg**2)

    def momentum_expectation(self, state: int = 0) -> float:
        """Compute ⟨p⟩ using finite difference."""
        psi = self.wavefunctions[:, state]
        hbar = self.config.hbar

        # p̂ψ = -iℏ dψ/dx
        dpsi = np.zeros_like(psi)
        dpsi[1:-1] = (psi[2:] - psi[:-2]) / (2 * self.dx)
        p_psi = -1j * hbar * dpsi

        return np.real(np.sum(np.conj(psi) * p_psi) * self.dx)

    def kinetic_energy(self, state: int = 0) -> float:
        """Compute ⟨T⟩ = ⟨p²⟩/(2m)."""
        psi = self.wavefunctions[:, state]
        hbar = self.config.hbar
        m = self.config.mass

        # T̂ψ = -ℏ²/(2m) d²ψ/dx²
        d2psi = np.zeros_like(psi)
        d2psi[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / self.dx**2
        T_psi = -hbar**2 / (2 * m) * d2psi

        return np.real(np.sum(np.conj(psi) * T_psi) * self.dx)

    def potential_energy(self, state: int = 0) -> float:
        """Compute ⟨V⟩."""
        return self.expectation_value(self.potential, state)

    def inner_product(self, state1: int, state2: int) -> complex:
        """Compute ⟨ψ₁|ψ₂⟩."""
        psi1 = self.wavefunctions[:, state1]
        psi2 = self.wavefunctions[:, state2]
        return np.sum(np.conj(psi1) * psi2) * self.dx

    def transition_dipole(self, state1: int, state2: int) -> complex:
        """Compute ⟨ψ₁|x|ψ₂⟩ (dipole matrix element)."""
        psi1 = self.wavefunctions[:, state1]
        psi2 = self.wavefunctions[:, state2]
        return np.sum(np.conj(psi1) * self.x * psi2) * self.dx

    # -------------------- Monte Carlo Measurement --------------------

    def simulate_measurements(self, state: int, observable: str,
                              n_measurements: int = 10000,
                              seed: int = None) -> Dict:
        """
        Simulate quantum measurements using Monte Carlo.

        Parameters
        ----------
        state : int
            State to measure
        observable : str
            'energy' or 'position'
        n_measurements : int
            Number of measurements
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Measurement statistics
        """
        rng = np.random.default_rng(seed)

        if observable == 'energy':
            # Energy measurement collapses to energy eigenstate
            psi = self.wavefunctions[:, state]

            # Probabilities in energy basis
            probs = np.abs(np.sum(np.conj(self.wavefunctions) * psi[:, np.newaxis],
                                  axis=0) * self.dx)**2

            # Sample outcomes
            indices = rng.choice(len(self.energies), size=n_measurements, p=probs)
            outcomes = self.energies[indices]

        elif observable == 'position':
            # Position measurement - sample from |ψ(x)|²
            psi = self.wavefunctions[:, state]
            prob_density = np.abs(psi)**2 * self.dx
            prob_density /= np.sum(prob_density)

            indices = rng.choice(len(self.x), size=n_measurements, p=prob_density)
            outcomes = self.x[indices]

        else:
            raise ValueError(f"Unknown observable: {observable}")

        return {
            'outcomes': outcomes,
            'mean': np.mean(outcomes),
            'std': np.std(outcomes),
            'variance': np.var(outcomes),
            'n_measurements': n_measurements
        }

    # -------------------- File I/O --------------------

    def save(self, filename: str):
        """Save system to HDF5 file."""
        with h5py.File(filename, 'w') as f:
            # Ensure system is solved
            if not self._solved:
                self.solve()

            # Save data
            f.create_dataset('energies', data=self.energies,
                             compression='gzip')
            f.create_dataset('wavefunctions', data=self.wavefunctions,
                             compression='gzip')
            f.create_dataset('grid', data=self.x)
            f.create_dataset('potential', data=self.potential)

            # Save configuration
            config = f.create_group('config')
            config.attrs['x_min'] = self.config.x_min
            config.attrs['x_max'] = self.config.x_max
            config.attrs['n_grid'] = self.config.n_grid
            config.attrs['n_states'] = self.config.n_states
            config.attrs['mass'] = self.config.mass
            config.attrs['hbar'] = self.config.hbar

            # Metadata
            f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')

    @classmethod
    def load(cls, filename: str,
             potential: Callable[[np.ndarray], np.ndarray]) -> 'QuantumSystem':
        """Load system from HDF5 file."""
        with h5py.File(filename, 'r') as f:
            config = QuantumSystemConfig(
                x_min=f['config'].attrs['x_min'],
                x_max=f['config'].attrs['x_max'],
                n_grid=f['config'].attrs['n_grid'],
                n_states=f['config'].attrs['n_states'],
                mass=f['config'].attrs['mass'],
                hbar=f['config'].attrs['hbar']
            )

            system = cls(config, potential)
            system._energies = f['energies'][:]
            system._wavefunctions = f['wavefunctions'][:]
            system._solved = True

        return system

    # -------------------- Analysis --------------------

    def summary(self) -> str:
        """Generate summary of quantum system."""
        lines = [
            "=" * 60,
            "Quantum System Summary",
            "=" * 60,
            f"Grid: [{self.config.x_min}, {self.config.x_max}], N={self.config.n_grid}",
            f"States computed: {self.config.n_states}",
            "",
            "Energy Levels:",
            "-" * 40,
        ]

        for n in range(min(10, len(self.energies))):
            E = self.energies[n]
            T = self.kinetic_energy(n)
            V = self.potential_energy(n)
            lines.append(f"  E_{n} = {E:10.6f}  (T={T:.4f}, V={V:.4f})")

        lines.extend([
            "",
            "Observables for ground state:",
            "-" * 40,
            f"  ⟨x⟩ = {self.position_expectation(0):.6f}",
            f"  Δx = {self.position_uncertainty(0):.6f}",
            f"  ⟨p⟩ = {self.momentum_expectation(0):.6f}",
            "=" * 60
        ])

        return "\n".join(lines)


# ============================================================
# DEMONSTRATION AND TESTS
# ============================================================

def run_capstone():
    """Run complete capstone demonstration."""

    print("=" * 70)
    print("WEEK 37 CAPSTONE: Quantum Harmonic Oscillator Simulator")
    print("=" * 70)

    # -------------------- Create System --------------------
    print("\n1. Creating quantum system...")

    config = QuantumSystemConfig(
        x_min=-10.0,
        x_max=10.0,
        n_grid=500,
        n_states=15
    )

    # Using omega = 1 (natural units)
    system = QuantumSystem(config, lambda x: harmonic_potential(x, omega=1.0))

    # -------------------- Solve --------------------
    print("\n2. Solving Schrödinger equation...")
    start = time.perf_counter()
    energies, wavefunctions = system.solve()
    solve_time = time.perf_counter() - start
    print(f"   Solved in {solve_time*1000:.2f} ms")

    # -------------------- Verify Results --------------------
    print("\n3. Verification:")

    print("\n   Energy levels (ω=1, expect E_n = n + 0.5):")
    print("   " + "-" * 45)
    for n in range(8):
        E = energies[n]
        exact = n + 0.5
        error = abs(E - exact)
        print(f"   n={n}: E={E:.6f}, exact={exact}, error={error:.2e}")

    print("\n   Orthonormality check:")
    for m in range(3):
        for n in range(m, 3):
            overlap = system.inner_product(m, n)
            expected = 1.0 if m == n else 0.0
            status = "✓" if abs(overlap - expected) < 1e-6 else "✗"
            print(f"   ⟨{m}|{n}⟩ = {overlap.real:.6f} {status}")

    print("\n   Selection rules (⟨m|x|n⟩ ≠ 0 only for |m-n|=1):")
    for m in range(4):
        for n in range(4):
            dipole = system.transition_dipole(m, n)
            if abs(dipole) > 0.01:
                print(f"   ⟨{m}|x|{n}⟩ = {dipole.real:.4f}")

    # -------------------- Observables --------------------
    print("\n4. Ground state observables:")
    print(f"   ⟨x⟩ = {system.position_expectation(0):.6f} (expect 0)")
    print(f"   Δx = {system.position_uncertainty(0):.6f} (expect 1/√2 ≈ 0.707)")
    print(f"   ⟨T⟩ = {system.kinetic_energy(0):.6f} (expect 0.25)")
    print(f"   ⟨V⟩ = {system.potential_energy(0):.6f} (expect 0.25)")

    # -------------------- Monte Carlo --------------------
    print("\n5. Monte Carlo measurement simulation:")

    # Superposition state measurement
    print("\n   Measuring position on ground state (10000 times):")
    pos_results = system.simulate_measurements(0, 'position', 10000, seed=42)
    print(f"   Measured ⟨x⟩ = {pos_results['mean']:.4f}")
    print(f"   Measured Δx = {pos_results['std']:.4f}")

    print("\n   Measuring energy on first excited state:")
    E_results = system.simulate_measurements(1, 'energy', 10000, seed=42)
    print(f"   Measured ⟨E⟩ = {E_results['mean']:.4f}")
    print(f"   (State 1 is eigenstate, so variance should be ~0)")
    print(f"   Measured ΔE = {E_results['std']:.6f}")

    # -------------------- File I/O --------------------
    print("\n6. File I/O:")

    filename = 'harmonic_oscillator.h5'
    system.save(filename)
    print(f"   Saved to {filename}")

    # Reload and verify
    system_loaded = QuantumSystem.load(filename, lambda x: harmonic_potential(x))
    energies_match = np.allclose(system.energies, system_loaded.energies)
    print(f"   Loaded and verified: {energies_match}")

    # Cleanup
    import os
    os.remove(filename)

    # -------------------- Summary --------------------
    print("\n" + system.summary())

    return system


# ============================================================
# COMPREHENSIVE REVIEW EXERCISES
# ============================================================

def review_exercises():
    """Week 37 review exercises."""

    print("\n" + "=" * 70)
    print("WEEK 37 REVIEW EXERCISES")
    print("=" * 70)

    # Exercise 1: Array Creation
    print("\n--- Exercise 1: Array Creation ---")
    x = np.linspace(-5, 5, 11)
    y = np.arange(-5, 6)
    print(f"linspace: {x[:5]}...")
    print(f"arange: {y[:5]}...")
    print(f"Equal spacing: {np.allclose(x, y)}")

    # Exercise 2: Vectorization
    print("\n--- Exercise 2: Vectorization ---")
    N = 100000
    x = np.random.randn(N)

    # Loop version (slow)
    start = time.perf_counter()
    result_loop = sum(xi**2 for xi in x)
    loop_time = time.perf_counter() - start

    # Vectorized (fast)
    start = time.perf_counter()
    result_vec = np.sum(x**2)
    vec_time = time.perf_counter() - start

    print(f"Loop: {loop_time*1000:.2f} ms, Vec: {vec_time*1000:.4f} ms")
    print(f"Speedup: {loop_time/vec_time:.0f}x")

    # Exercise 3: Broadcasting
    print("\n--- Exercise 3: Broadcasting ---")
    a = np.array([[1], [2], [3]])    # (3, 1)
    b = np.array([10, 20, 30, 40])   # (4,)
    c = a * b                         # (3, 4)
    print(f"Shape: {a.shape} × {b.shape} → {c.shape}")
    print(f"Result:\n{c}")

    # Exercise 4: Eigenvalues
    print("\n--- Exercise 4: Eigenvalues ---")
    # Pauli Z matrix
    sigma_z = np.array([[1, 0], [0, -1]])
    eigenvalues, eigenvectors = np.linalg.eigh(sigma_z)
    print(f"σ_z eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

    # Exercise 5: Random Sampling
    print("\n--- Exercise 5: Random Sampling ---")
    rng = np.random.default_rng(42)
    # Simulate coin flips
    flips = rng.choice([0, 1], size=1000, p=[0.5, 0.5])
    print(f"1000 coin flips: {np.sum(flips)} heads, {1000-np.sum(flips)} tails")

    # Exercise 6: File I/O
    print("\n--- Exercise 6: File I/O ---")
    data = {'x': x[:100], 'sigma_z': sigma_z}
    np.savez('test_data.npz', **data)
    loaded = np.load('test_data.npz')
    print(f"Saved keys: {list(loaded.keys())}")
    print(f"Data matches: {np.allclose(x[:100], loaded['x'])}")
    loaded.close()
    import os
    os.remove('test_data.npz')

    print("\n--- All exercises complete! ---")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Run capstone project
    system = run_capstone()

    # Run review exercises
    review_exercises()

    print("\n" + "=" * 70)
    print("WEEK 37 COMPLETE!")
    print("=" * 70)
    print("""
    Skills Mastered:
    ✓ Python: functions, classes, decorators, generators
    ✓ NumPy arrays: creation, indexing, dtypes, views
    ✓ Vectorization: ufuncs, broadcasting, eliminating loops
    ✓ Linear algebra: eigenvalues, solve, decompositions
    ✓ Random numbers: distributions, Monte Carlo, sampling
    ✓ File I/O: NPY, HDF5, profiling, optimization

    Ready for Week 38: SciPy & Numerical Methods
    """)
```

---

## Practice Problems

### Comprehensive Review

**Problem 1:** Create a class `WavePacket` that represents a Gaussian wave packet. Include methods for computing the probability density, expectation values, and time evolution.

**Problem 2:** Implement a function that computes the first `n` terms of the harmonic oscillator creation and annihilation operator matrices, and verify the commutator $[a, a^\dagger] = I$.

**Problem 3:** Write a Monte Carlo program to estimate the tunneling probability through a rectangular barrier.

**Problem 4:** Compare the performance of computing $\sum_i\sum_j A_{ij}B_{ji}$ using:
- Nested loops
- `np.einsum('ij,ji->', A, B)`
- `np.trace(A @ B.T)`

**Problem 5:** Build a data pipeline that:
1. Generates random quantum states
2. Computes expectation values for multiple observables
3. Saves results to HDF5 with metadata
4. Loads and analyzes results

---

## Week 37 Mastery Checklist

### Python Foundations (Day 253)
- [ ] Can write functions with *args, **kwargs, type hints
- [ ] Understand closures and function factories
- [ ] Can create and use decorators
- [ ] Can design classes with inheritance and special methods
- [ ] Understand generators for memory efficiency

### NumPy Arrays (Day 254)
- [ ] Can create arrays with linspace, arange, zeros, eye, diag
- [ ] Understand dtypes and can choose appropriate precision
- [ ] Master slicing including negative indices
- [ ] Can use boolean and fancy indexing
- [ ] Distinguish views from copies

### Vectorization (Day 255)
- [ ] Consistently write loop-free numerical code
- [ ] Understand and apply broadcasting rules
- [ ] Use ufuncs for element-wise operations
- [ ] Can use einsum for complex operations

### Linear Algebra (Day 256)
- [ ] Solve eigenvalue problems with eigh and eig
- [ ] Use solve for linear systems
- [ ] Understand SVD, QR, and other decompositions
- [ ] Can verify eigenvector orthonormality

### Random Numbers (Day 257)
- [ ] Use Generator instead of global random state
- [ ] Generate from common distributions
- [ ] Implement Monte Carlo integration
- [ ] Simulate quantum measurements

### File I/O & Performance (Day 258)
- [ ] Save/load arrays in binary formats
- [ ] Use HDF5 for complex data
- [ ] Profile code to find bottlenecks
- [ ] Apply optimization techniques

---

## Preview: Week 38

Next week we dive into **SciPy and Numerical Methods**:

- **Day 260:** SciPy Overview and Integration (`scipy.integrate`)
- **Day 261:** Ordinary Differential Equations (`solve_ivp`)
- **Day 262:** Root Finding and Optimization (`scipy.optimize`)
- **Day 263:** Advanced Linear Algebra (`scipy.linalg`)
- **Day 264:** Special Functions (`scipy.special`)
- **Day 265:** Fast Fourier Transforms (`scipy.fft`)
- **Day 266:** Week Review and Integration

These tools extend NumPy for real physics simulations.

---

## Summary

Week 37 established the computational foundation for all future work:

| Skill | Impact |
|-------|--------|
| NumPy arrays | 100x faster than Python lists |
| Vectorization | Eliminates slow Python loops |
| Broadcasting | Memory-efficient operations |
| Linear algebra | Solves quantum eigenvalue problems |
| Monte Carlo | Simulates measurement and integration |
| File I/O | Enables large-scale computation |

You now have the tools to implement any numerical algorithm from the physics literature.

---

## Key Insights from Week 37

1. **NumPy is not optional** — it's the foundation of scientific Python
2. **Loops are almost always wrong** — vectorize instead
3. **Broadcasting is powerful** — learn the rules deeply
4. **eigh for Hamiltonians** — guarantees real eigenvalues
5. **Set random seeds** — reproducibility is essential
6. **Profile before optimizing** — find the real bottlenecks
7. **HDF5 for complex data** — better than NPZ for large projects

---

*"The goal of computing is insight, not numbers."* — Richard Hamming

*Week 37 gave us the numbers; Week 38 will sharpen the insight.*
