# Day 266: Week 38 Review — SciPy Integration

## Overview

**Day 266** | **Week 38** | **Month 10: Scientific Computing**

Today we synthesize all SciPy skills into a comprehensive capstone project. We'll build a complete quantum simulation framework that combines numerical integration, ODE solving, optimization, matrix functions, special functions, and sparse methods. This integration demonstrates mastery of scientific computing for quantum physics.

**Prerequisites:** Days 260-265 (complete Week 38)
**Outcome:** Integrate all SciPy tools into production-quality physics simulations

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Capstone: Complete quantum simulation package |
| Afternoon | 3 hours | Review exercises and problem solving |
| Evening | 2 hours | Self-assessment and Week 39 preparation |

---

## Week 38 Summary

### Key Modules Covered

| Day | Module | Key Functions |
|-----|--------|---------------|
| 260 | `scipy.integrate` | `quad`, `dblquad`, `tplquad` |
| 261 | `scipy.integrate` | `solve_ivp`, RK45, BDF |
| 262 | `scipy.optimize` | `minimize`, `root`, `curve_fit` |
| 263 | `scipy.linalg` | `expm`, `sqrtm`, `schur` |
| 264 | `scipy.special`, `scipy.fft` | `hermite`, `sph_harm`, `fft` |
| 265 | `scipy.sparse` | `csr_matrix`, `eigsh`, `diags` |

### Essential Imports

```python
import numpy as np
from scipy import integrate, optimize, linalg, special, fft, sparse
from scipy.sparse.linalg import eigsh, eigs
```

---

## Capstone Project: Complete Quantum Simulator

```python
"""
Week 38 Capstone: Comprehensive Quantum Simulation Framework
============================================================

Integrates all SciPy modules for quantum physics:
- Integration for expectation values
- ODEs for time evolution
- Optimization for variational methods
- Matrix functions for propagators
- Special functions for wave functions
- Sparse matrices for large systems
"""

import numpy as np
from scipy import integrate, optimize, linalg, special, fft, sparse
from scipy.sparse.linalg import eigsh
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass
import time

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class QuantumState:
    """Container for quantum state information."""
    amplitudes: np.ndarray
    grid: np.ndarray
    is_momentum_space: bool = False

    @property
    def probability_density(self) -> np.ndarray:
        return np.abs(self.amplitudes)**2

    @property
    def norm(self) -> float:
        dx = self.grid[1] - self.grid[0]
        return np.sqrt(np.sum(self.probability_density) * dx)

    def normalize(self) -> 'QuantumState':
        return QuantumState(
            self.amplitudes / self.norm,
            self.grid,
            self.is_momentum_space
        )

@dataclass
class QuantumSystem:
    """Complete quantum system specification."""
    potential: Callable
    grid: np.ndarray
    mass: float = 1.0
    hbar: float = 1.0

    @property
    def dx(self) -> float:
        return self.grid[1] - self.grid[0]

    @property
    def N(self) -> int:
        return len(self.grid)

# ============================================================
# HAMILTONIAN CONSTRUCTION (Sparse)
# ============================================================

def build_hamiltonian_sparse(system: QuantumSystem) -> sparse.csr_matrix:
    """
    Build sparse Hamiltonian matrix.

    H = -ℏ²/(2m) d²/dx² + V(x)
    """
    N = system.N
    dx = system.dx
    hbar = system.hbar
    m = system.mass

    # Kinetic energy coefficient
    T_coeff = hbar**2 / (2 * m * dx**2)

    # Potential on grid
    V = system.potential(system.grid)

    # Build tridiagonal Hamiltonian
    H = sparse.diags(
        [-T_coeff * np.ones(N-1), 2*T_coeff + V, -T_coeff * np.ones(N-1)],
        [-1, 0, 1],
        format='csr'
    )

    return H

def build_hamiltonian_dense(system: QuantumSystem) -> np.ndarray:
    """Build dense Hamiltonian matrix."""
    return build_hamiltonian_sparse(system).toarray()

# ============================================================
# EIGENVALUE SOLVER
# ============================================================

def solve_eigenstates(system: QuantumSystem, n_states: int = 10,
                      use_sparse: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve time-independent Schrödinger equation.

    Returns (energies, wavefunctions).
    """
    if use_sparse:
        H = build_hamiltonian_sparse(system)
        energies, states = eigsh(H, k=n_states, which='SA')
    else:
        H = build_hamiltonian_dense(system)
        energies, states = np.linalg.eigh(H)
        energies = energies[:n_states]
        states = states[:, :n_states]

    # Sort by energy
    idx = np.argsort(energies)
    energies = energies[idx]
    states = states[:, idx]

    # Normalize
    dx = system.dx
    for i in range(states.shape[1]):
        norm = np.sqrt(np.sum(np.abs(states[:, i])**2) * dx)
        states[:, i] /= norm

    return energies, states

# ============================================================
# TIME EVOLUTION
# ============================================================

def time_evolve_exact(system: QuantumSystem, psi_0: np.ndarray,
                      t: float) -> np.ndarray:
    """
    Exact time evolution using eigendecomposition.

    ψ(t) = Σ c_n e^(-iE_n t/ℏ) ψ_n
    """
    # Get all eigenstates
    H = build_hamiltonian_dense(system)
    energies, states = np.linalg.eigh(H)

    # Expand initial state
    dx = system.dx
    c_n = np.array([np.sum(np.conj(states[:, n]) * psi_0) * dx
                    for n in range(len(energies))])

    # Evolve
    phase = np.exp(-1j * energies * t / system.hbar)
    psi_t = np.sum(c_n * phase * states, axis=1)

    return psi_t

def time_evolve_split_operator(system: QuantumSystem, psi_0: np.ndarray,
                               t_final: float, dt: float) -> np.ndarray:
    """
    Time evolution using split-operator FFT method.

    e^(-iHdt) ≈ e^(-iVdt/2) e^(-iTdt) e^(-iVdt/2)
    """
    N = system.N
    dx = system.dx
    hbar = system.hbar
    m = system.mass

    # Momentum grid
    p = fft.fftfreq(N, d=dx) * 2 * np.pi * hbar

    # Operators in respective spaces
    V = system.potential(system.grid)
    T_p = p**2 / (2 * m)

    # Half potential phase
    exp_V_half = np.exp(-0.5j * V * dt / hbar)

    # Full kinetic phase
    exp_T = np.exp(-1j * T_p * dt / hbar)

    psi = psi_0.copy()
    n_steps = int(t_final / dt)

    for _ in range(n_steps):
        # Half V step
        psi = exp_V_half * psi

        # Full T step in momentum space
        psi_p = fft.fft(psi)
        psi_p = exp_T * psi_p
        psi = fft.ifft(psi_p)

        # Half V step
        psi = exp_V_half * psi

    return psi

def time_evolve_ode(system: QuantumSystem, psi_0: np.ndarray,
                    t_span: Tuple[float, float],
                    t_eval: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time evolution by solving ODE directly.
    """
    H = build_hamiltonian_dense(system)
    hbar = system.hbar

    def schrodinger_rhs(t, psi_flat):
        psi = psi_flat[:len(psi_flat)//2] + 1j * psi_flat[len(psi_flat)//2:]
        dpsi = -1j/hbar * (H @ psi)
        return np.concatenate([dpsi.real, dpsi.imag])

    psi0_flat = np.concatenate([psi_0.real, psi_0.imag])

    sol = integrate.solve_ivp(
        schrodinger_rhs, t_span, psi0_flat,
        t_eval=t_eval, method='DOP853',
        rtol=1e-8, atol=1e-10
    )

    psi = sol.y[:len(psi_0)] + 1j * sol.y[len(psi_0):]

    return sol.t, psi

# ============================================================
# EXPECTATION VALUES (Integration)
# ============================================================

def expectation_value(psi: np.ndarray, operator: np.ndarray,
                      dx: float) -> float:
    """Compute ⟨ψ|Ô|ψ⟩."""
    return np.real(np.sum(np.conj(psi) * operator * psi) * dx)

def position_expectation(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    return expectation_value(psi, x, dx)

def position_uncertainty(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    x_avg = position_expectation(psi, x, dx)
    x2_avg = expectation_value(psi, x**2, dx)
    return np.sqrt(x2_avg - x_avg**2)

def energy_expectation(system: QuantumSystem, psi: np.ndarray) -> float:
    """Compute ⟨H⟩."""
    H = build_hamiltonian_dense(system)
    H_psi = H @ psi
    return np.real(np.sum(np.conj(psi) * H_psi) * system.dx)

# ============================================================
# VARIATIONAL METHOD (Optimization)
# ============================================================

def variational_energy(params: np.ndarray, system: QuantumSystem,
                       trial_func: Callable) -> float:
    """
    Compute variational energy for trial wave function.
    """
    psi = trial_func(system.grid, params)

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * system.dx)
    if norm < 1e-10:
        return 1e10
    psi = psi / norm

    return energy_expectation(system, psi)

def variational_minimize(system: QuantumSystem, trial_func: Callable,
                         params0: np.ndarray, method: str = 'Nelder-Mead') -> dict:
    """
    Find optimal variational parameters.
    """
    result = optimize.minimize(
        variational_energy, params0,
        args=(system, trial_func),
        method=method
    )

    return {
        'params': result.x,
        'energy': result.fun,
        'success': result.success,
        'psi': trial_func(system.grid, result.x)
    }

# ============================================================
# SPECIAL FUNCTION WAVE FUNCTIONS
# ============================================================

def harmonic_oscillator_eigenstate(x: np.ndarray, n: int,
                                   omega: float = 1.0) -> np.ndarray:
    """Generate harmonic oscillator eigenstate using Hermite polynomials."""
    from math import factorial, pi

    Hn = special.hermite(n)
    norm = (omega/pi)**0.25 / np.sqrt(2**n * factorial(n))
    xi = np.sqrt(omega) * x

    return norm * Hn(xi) * np.exp(-xi**2 / 2)

def gaussian_wavepacket(x: np.ndarray, x0: float, sigma: float,
                        k0: float) -> np.ndarray:
    """Generate Gaussian wave packet."""
    norm = (2 * np.pi * sigma**2)**(-0.25)
    return norm * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)

# ============================================================
# MOMENTUM SPACE (FFT)
# ============================================================

def to_momentum_space(psi: np.ndarray, x: np.ndarray,
                      hbar: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Transform to momentum space."""
    N = len(x)
    dx = x[1] - x[0]

    p = fft.fftfreq(N, d=dx) * 2 * np.pi * hbar
    psi_p = fft.fft(psi) * dx / np.sqrt(2 * np.pi * hbar)

    p = fft.fftshift(p)
    psi_p = fft.fftshift(psi_p)

    return p, psi_p

def to_position_space(psi_p: np.ndarray, p: np.ndarray,
                      hbar: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Transform to position space."""
    N = len(p)
    dp = p[1] - p[0]

    x = fft.fftfreq(N, d=dp/(2*np.pi*hbar)) * 2 * np.pi * hbar
    psi_x = fft.ifft(fft.ifftshift(psi_p)) * np.sqrt(2 * np.pi * hbar) * N / (x[-1] - x[0])

    return fft.fftshift(x), fft.fftshift(psi_x)

# ============================================================
# DEMONSTRATION
# ============================================================

def run_capstone():
    """Run complete demonstration of all SciPy features."""

    print("=" * 70)
    print("WEEK 38 CAPSTONE: Complete Quantum Simulator")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. System Setup
    # --------------------------------------------------------
    print("\n1. SYSTEM SETUP")
    print("-" * 40)

    x = np.linspace(-15, 15, 512)
    V = lambda x: 0.5 * x**2  # Harmonic oscillator

    system = QuantumSystem(potential=V, grid=x)
    print(f"Grid: N={system.N}, dx={system.dx:.4f}")

    # --------------------------------------------------------
    # 2. Eigenvalue Problem (Sparse)
    # --------------------------------------------------------
    print("\n2. EIGENVALUE PROBLEM (sparse eigsh)")
    print("-" * 40)

    start = time.perf_counter()
    energies, states = solve_eigenstates(system, n_states=10, use_sparse=True)
    elapsed = time.perf_counter() - start

    print(f"Solved in {elapsed*1000:.2f} ms")
    print("\nEnergies (expect n + 0.5):")
    for n in range(5):
        print(f"  E_{n} = {energies[n]:.6f}")

    # --------------------------------------------------------
    # 3. Expectation Values (Integration)
    # --------------------------------------------------------
    print("\n3. EXPECTATION VALUES (integration concepts)")
    print("-" * 40)

    psi_0 = states[:, 0]
    x_avg = position_expectation(psi_0, x, system.dx)
    delta_x = position_uncertainty(psi_0, x, system.dx)

    print(f"Ground state: ⟨x⟩ = {x_avg:.6f}, Δx = {delta_x:.6f}")
    print(f"Theory: ⟨x⟩ = 0, Δx = 1/√2 = {1/np.sqrt(2):.6f}")

    # --------------------------------------------------------
    # 4. Time Evolution (ODE vs Split-Operator)
    # --------------------------------------------------------
    print("\n4. TIME EVOLUTION")
    print("-" * 40)

    # Initial: displaced Gaussian
    psi_init = gaussian_wavepacket(x, x0=3.0, sigma=1.0, k0=0.0)

    # Method 1: Split-operator
    t_final = 2 * np.pi  # One classical period
    start = time.perf_counter()
    psi_split = time_evolve_split_operator(system, psi_init, t_final, dt=0.01)
    time_split = time.perf_counter() - start

    # Method 2: Exact (eigendecomposition)
    start = time.perf_counter()
    psi_exact = time_evolve_exact(system, psi_init, t_final)
    time_exact = time.perf_counter() - start

    print(f"Split-operator: {time_split*1000:.2f} ms")
    print(f"Exact:          {time_exact*1000:.2f} ms")
    print(f"Difference: {np.linalg.norm(psi_split - psi_exact):.2e}")

    # Check: should return to initial position
    x_final = position_expectation(psi_split, x, system.dx)
    print(f"Initial ⟨x⟩ = 3.0, Final ⟨x⟩ = {x_final:.4f}")

    # --------------------------------------------------------
    # 5. Variational Method (Optimization)
    # --------------------------------------------------------
    print("\n5. VARIATIONAL METHOD (optimization)")
    print("-" * 40)

    def trial_gaussian(x, params):
        alpha = params[0]
        if alpha <= 0:
            return np.zeros_like(x)
        return np.exp(-alpha * x**2)

    result = variational_minimize(system, trial_gaussian, params0=[0.3])
    print(f"Optimal α = {result['params'][0]:.6f} (exact: 0.5)")
    print(f"E_var = {result['energy']:.6f} (exact: 0.5)")

    # --------------------------------------------------------
    # 6. Special Functions
    # --------------------------------------------------------
    print("\n6. SPECIAL FUNCTIONS")
    print("-" * 40)

    for n in range(4):
        psi_analytic = harmonic_oscillator_eigenstate(x, n)
        psi_numeric = states[:, n]

        # Sign convention may differ
        overlap = np.abs(np.sum(psi_analytic * psi_numeric) * system.dx)
        print(f"n={n}: |⟨ψ_analytic|ψ_numeric⟩| = {overlap:.6f}")

    # --------------------------------------------------------
    # 7. Momentum Space (FFT)
    # --------------------------------------------------------
    print("\n7. MOMENTUM SPACE (FFT)")
    print("-" * 40)

    p, psi_p = to_momentum_space(psi_0, x)

    # For HO ground state: Δp = √(ω)/√2 = 1/√2
    p2_avg = np.sum(np.abs(psi_p)**2 * p**2) * (p[1] - p[0])
    p_avg = np.sum(np.abs(psi_p)**2 * p) * (p[1] - p[0])
    delta_p = np.sqrt(p2_avg - p_avg**2)

    print(f"Ground state in momentum space:")
    print(f"  ⟨p⟩ = {p_avg:.6f}, Δp = {delta_p:.6f}")
    print(f"  Δx·Δp = {delta_x * delta_p:.6f} (≥ 0.5)")

    # --------------------------------------------------------
    # 8. Large-Scale (Sparse)
    # --------------------------------------------------------
    print("\n8. LARGE-SCALE SPARSE")
    print("-" * 40)

    x_large = np.linspace(-20, 20, 10000)
    system_large = QuantumSystem(potential=V, grid=x_large)

    start = time.perf_counter()
    H_sparse = build_hamiltonian_sparse(system_large)
    E_large, _ = eigsh(H_sparse, k=5, which='SA')
    elapsed = time.perf_counter() - start

    print(f"N = 10000: solved in {elapsed*1000:.2f} ms")
    print(f"Ground state: E_0 = {E_large[0]:.8f}")

    print("\n" + "=" * 70)
    print("CAPSTONE COMPLETE!")
    print("=" * 70)


# ============================================================
# REVIEW EXERCISES
# ============================================================

def review_exercises():
    """Week 38 review exercises."""

    print("\n" + "=" * 70)
    print("WEEK 38 REVIEW EXERCISES")
    print("=" * 70)

    # Exercise 1: Integration
    print("\n--- Exercise 1: Integration ---")
    result, _ = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
    print(f"∫exp(-x²)dx = {result:.6f} (exact: √π = {np.sqrt(np.pi):.6f})")

    # Exercise 2: ODE
    print("\n--- Exercise 2: ODE ---")
    sol = integrate.solve_ivp(lambda t, y: -y, [0, 5], [1.0])
    print(f"dy/dt = -y, y(0)=1: y(5) = {sol.y[0,-1]:.6f} (exact: {np.exp(-5):.6f})")

    # Exercise 3: Optimization
    print("\n--- Exercise 3: Optimization ---")
    result = optimize.minimize(lambda x: (x[0]-1)**2 + (x[1]-2)**2, [0, 0])
    print(f"Min of (x-1)² + (y-2)²: ({result.x[0]:.4f}, {result.x[1]:.4f})")

    # Exercise 4: Matrix exponential
    print("\n--- Exercise 4: Matrix Exponential ---")
    A = np.array([[0, 1], [-1, 0]])
    expA = linalg.expm(np.pi/2 * A)
    print(f"exp(π/2 * [[0,1],[-1,0]]):")
    print(np.round(expA, 4))
    print("(Should be 90° rotation)")

    # Exercise 5: Special functions
    print("\n--- Exercise 5: Special Functions ---")
    H3 = special.hermite(3)
    print(f"H_3(0) = {H3(0):.0f}, H_3(1) = {H3(1):.0f}")

    # Exercise 6: FFT
    print("\n--- Exercise 6: FFT ---")
    x = np.linspace(0, 2*np.pi, 64, endpoint=False)
    f = np.sin(3*x) + 0.5*np.sin(7*x)
    F = fft.fft(f)
    freqs = fft.fftfreq(64, x[1]-x[0])
    peaks = np.abs(F) > 10
    print(f"Frequencies detected: {freqs[peaks][:4]}")

    # Exercise 7: Sparse
    print("\n--- Exercise 7: Sparse ---")
    H = sparse.diags([np.ones(99), 2*np.ones(100), np.ones(99)], [-1, 0, 1])
    E, _ = eigsh(H, k=3, which='SA')
    print(f"Tridiagonal eigenvalues: {E}")

    print("\n--- All exercises complete! ---")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_capstone()
    review_exercises()

    print("\n" + "=" * 70)
    print("WEEK 38 COMPLETE!")
    print("=" * 70)
    print("""
    Skills Mastered:
    ✓ Numerical integration (quad, dblquad)
    ✓ ODE solving (solve_ivp, RK45, BDF)
    ✓ Optimization (minimize, root, curve_fit)
    ✓ Matrix functions (expm, sqrtm, logm)
    ✓ Special functions (hermite, sph_harm, jv)
    ✓ FFT (fft, ifft, fftfreq)
    ✓ Sparse matrices (csr_matrix, eigsh)

    Ready for Week 39: Visualization with Matplotlib
    """)
```

---

## Week 38 Mastery Checklist

### Integration (Day 260)
- [ ] Compute definite integrals with `quad`
- [ ] Handle infinite limits and singularities
- [ ] Perform multi-dimensional integration

### ODEs (Day 261)
- [ ] Solve initial value problems with `solve_ivp`
- [ ] Choose appropriate solvers for stiff/non-stiff
- [ ] Implement Schrödinger equation evolution

### Optimization (Day 262)
- [ ] Find roots with `brentq`, `fsolve`
- [ ] Minimize functions with `minimize`
- [ ] Fit data with `curve_fit`

### Linear Algebra (Day 263)
- [ ] Compute matrix exponentials with `expm`
- [ ] Apply matrix functions (sqrt, log)
- [ ] Use specialized decompositions

### Special Functions (Day 264)
- [ ] Generate Hermite polynomials
- [ ] Compute spherical harmonics
- [ ] Use FFT for spectral analysis

### Sparse Matrices (Day 265)
- [ ] Choose appropriate sparse formats
- [ ] Build large sparse Hamiltonians
- [ ] Solve sparse eigenvalue problems

---

## Preview: Week 39

Next week we master **Visualization with Matplotlib**:

- **Day 267:** Matplotlib Fundamentals
- **Day 268:** 2D Plotting for Physics
- **Day 269:** 3D Visualization
- **Day 270:** Animations
- **Day 271:** Interactive Widgets
- **Day 272:** Publication-Quality Figures
- **Day 273:** Week Review

Visualization turns numbers into physical insight!

---

## Summary

Week 38 provided the computational toolkit for serious quantum mechanics:

| Tool | Quantum Application |
|------|---------------------|
| `integrate.quad` | Expectation values, normalization |
| `solve_ivp` | Time-dependent Schrödinger |
| `minimize` | Variational methods |
| `expm` | Time evolution operator |
| `hermite` | HO wave functions |
| `fft` | Momentum space |
| `eigsh` | Large-scale eigenproblems |

---

## Key Insights from Week 38

1. **quad handles difficult integrals** — adaptive methods, infinite limits
2. **Choose ODE solvers wisely** — RK45 for smooth, BDF for stiff
3. **Variational methods need good optimization** — derivatives help
4. **Matrix exponentials enable exact propagation** — eigendecomposition for efficiency
5. **Special functions are exact** — use them instead of numerical approximations
6. **FFT connects position and momentum** — O(N log N) complexity
7. **Sparse methods unlock large systems** — O(N) storage, O(N) eigensolvers

---

*"SciPy provides the numerical engine; understanding provides the physics."*

*Week 38 gave us the engine; Week 39 will show us how to see what it produces.*
