# Day 257: Random Numbers and Statistics

## Overview

**Day 257** | **Week 37** | **Month 10: Scientific Computing**

Today we harness NumPy's random number generation for physics simulations. Randomness is fundamental to quantum mechanics—measurement outcomes are probabilistic—and Monte Carlo methods leverage this to solve otherwise intractable problems. By day's end, you'll generate random samples from any distribution, compute statistical properties, and implement basic Monte Carlo integration.

**Prerequisites:** Days 253-256 (NumPy fundamentals, linear algebra)
**Outcome:** Generate random samples, compute statistics, implement Monte Carlo methods

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Random generators, distributions, statistics |
| Afternoon | 3 hours | Practice: Monte Carlo integration and sampling |
| Evening | 2 hours | Lab: Quantum measurement simulation |

---

## Learning Objectives

By the end of Day 257, you will be able to:

1. **Generate random numbers** from uniform, normal, and other distributions
2. **Set seeds for reproducibility** in scientific computations
3. **Compute statistical quantities** (mean, variance, correlation)
4. **Implement Monte Carlo integration** for multidimensional integrals
5. **Sample from custom distributions** using various techniques
6. **Simulate quantum measurements** using random sampling
7. **Apply bootstrap methods** for error estimation

---

## Core Content

### 1. NumPy Random Generator

NumPy's modern random API uses `numpy.random.Generator`:

```python
import numpy as np

# Create a Generator (recommended method)
rng = np.random.default_rng(seed=42)  # Reproducible

# Alternative: legacy interface (still works but not recommended)
# np.random.seed(42)  # Global state - avoid

# Basic random numbers
uniform = rng.random(10)          # 10 uniform [0, 1)
integers = rng.integers(0, 100, 10)  # 10 integers [0, 100)
normal = rng.standard_normal(10)   # 10 standard normal
```

### 2. Common Distributions

```python
rng = np.random.default_rng(42)

# Uniform distribution
uniform = rng.uniform(low=-1, high=1, size=1000)

# Normal (Gaussian) distribution
normal = rng.normal(loc=0, scale=1, size=1000)  # μ=0, σ=1

# Exponential distribution (radioactive decay, waiting times)
exponential = rng.exponential(scale=1.0, size=1000)  # λ=1

# Poisson distribution (photon counting)
poisson = rng.poisson(lam=5.0, size=1000)  # λ=5

# Binomial distribution (N independent trials)
binomial = rng.binomial(n=10, p=0.5, size=1000)

# Chi-squared distribution (sum of squared normals)
chi2 = rng.chisquare(df=3, size=1000)

# Multivariate normal (correlated variables)
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix
multivar = rng.multivariate_normal(mean, cov, size=1000)
```

### 3. Shuffling and Sampling

```python
rng = np.random.default_rng(42)

# Shuffle array in place
arr = np.arange(10)
rng.shuffle(arr)
print(arr)  # [2, 6, 1, ...]

# Random permutation (returns copy)
perm = rng.permutation(10)
print(perm)

# Random choice from array
elements = np.array(['a', 'b', 'c', 'd'])
choices = rng.choice(elements, size=10, replace=True)

# Weighted random choice (for quantum measurement!)
probs = np.array([0.1, 0.2, 0.3, 0.4])  # Must sum to 1
weighted_choices = rng.choice(4, size=1000, p=probs)
```

### 4. Statistical Functions

```python
data = np.random.randn(1000)

# Central tendency
mean = np.mean(data)
median = np.median(data)

# Spread
variance = np.var(data)
std = np.std(data)
iqr = np.percentile(data, 75) - np.percentile(data, 25)

# Extrema
minimum, maximum = np.min(data), np.max(data)

# Histogram
counts, bin_edges = np.histogram(data, bins=50)

# Correlation
x, y = np.random.randn(2, 1000)
correlation = np.corrcoef(x, y)
print(f"Correlation coefficient: {correlation[0, 1]}")
```

### 5. Monte Carlo Integration

The key insight: $\int f(x) dx \approx \frac{V}{N} \sum_{i=1}^{N} f(x_i)$ where $x_i$ are random samples.

```python
def monte_carlo_integrate(f, bounds, n_samples=100000):
    """
    Monte Carlo integration in arbitrary dimensions.

    Parameters
    ----------
    f : callable
        Function to integrate
    bounds : list of tuples
        [(x_min, x_max), (y_min, y_max), ...]
    n_samples : int
        Number of random samples

    Returns
    -------
    integral : float
        Estimated integral
    error : float
        Statistical error estimate
    """
    rng = np.random.default_rng()
    n_dims = len(bounds)

    # Generate random points in the domain
    samples = np.zeros((n_samples, n_dims))
    volume = 1.0
    for i, (low, high) in enumerate(bounds):
        samples[:, i] = rng.uniform(low, high, n_samples)
        volume *= (high - low)

    # Evaluate function at sample points
    if n_dims == 1:
        values = f(samples[:, 0])
    else:
        values = np.array([f(*point) for point in samples])

    # Estimate integral and error
    integral = volume * np.mean(values)
    error = volume * np.std(values) / np.sqrt(n_samples)

    return integral, error

# Test: ∫₀¹ x² dx = 1/3
f = lambda x: x**2
result, error = monte_carlo_integrate(f, [(0, 1)], n_samples=100000)
print(f"∫₀¹ x² dx = {result:.6f} ± {error:.6f} (exact: 0.333333)")

# Test: ∫∫ exp(-(x²+y²)) dxdy over [-5,5]² ≈ π
f_2d = lambda x, y: np.exp(-(x**2 + y**2))
result_2d, error_2d = monte_carlo_integrate(f_2d, [(-5, 5), (-5, 5)])
print(f"∫∫ exp(-(x²+y²)) dxdy = {result_2d:.6f} ± {error_2d:.6f} (exact: {np.pi:.6f})")
```

### 6. Importance Sampling

For integrals with peaked integrands, sample from a distribution close to the integrand shape:

```python
def importance_sampling(f, g, g_sample, g_pdf, n_samples=100000):
    """
    Monte Carlo with importance sampling.

    ∫f(x)dx = ∫[f(x)/g(x)]g(x)dx ≈ (1/N)Σf(xᵢ)/g(xᵢ)

    Parameters
    ----------
    f : callable
        Function to integrate
    g_sample : callable
        Function to generate samples from g(x)
    g_pdf : callable
        Probability density function g(x)
    """
    rng = np.random.default_rng()

    # Sample from importance distribution
    samples = g_sample(n_samples)

    # Compute weighted values
    weights = f(samples) / g_pdf(samples)

    integral = np.mean(weights)
    error = np.std(weights) / np.sqrt(n_samples)

    return integral, error

# Example: ∫₀^∞ x² e^(-x) dx = 2 (Gamma function)
f = lambda x: x**2 * np.exp(-x)

# Use exponential distribution as importance function
g_sample = lambda n: np.random.exponential(1.0, n)
g_pdf = lambda x: np.exp(-x)

result, error = importance_sampling(f, None, g_sample, g_pdf)
print(f"∫₀^∞ x² e^(-x) dx = {result:.6f} ± {error:.6f} (exact: 2.000000)")
```

---

## Quantum Mechanics Connection

### Simulating Quantum Measurements

When we measure an observable $\hat{A}$ on state $|\psi\rangle$:
- Outcome $a_n$ occurs with probability $P_n = |\langle a_n|\psi\rangle|^2$
- After measurement, state collapses to $|a_n\rangle$

```python
def simulate_measurement(psi, eigenstates, eigenvalues, n_measurements=1000):
    """
    Simulate quantum measurements using Born rule.

    Parameters
    ----------
    psi : ndarray
        State vector (normalized)
    eigenstates : ndarray
        Columns are eigenstates of observable
    eigenvalues : ndarray
        Eigenvalues (measurement outcomes)
    n_measurements : int
        Number of measurements to simulate

    Returns
    -------
    outcomes : ndarray
        Array of measurement outcomes
    statistics : dict
        Mean, variance, and distribution
    """
    rng = np.random.default_rng()

    # Compute Born probabilities
    probabilities = np.abs(eigenstates.T.conj() @ psi)**2

    # Normalize (handle numerical errors)
    probabilities /= np.sum(probabilities)

    # Sample measurement outcomes
    outcome_indices = rng.choice(len(eigenvalues), size=n_measurements, p=probabilities)
    outcomes = eigenvalues[outcome_indices]

    # Compute statistics
    stats = {
        'mean': np.mean(outcomes),
        'variance': np.var(outcomes),
        'std': np.std(outcomes),
        'histogram': np.bincount(outcome_indices, minlength=len(eigenvalues))
    }

    return outcomes, stats

# Example: Measure energy of superposition state
# |ψ⟩ = (|0⟩ + |1⟩)/√2 in harmonic oscillator
psi = np.array([1, 1, 0, 0, 0]) / np.sqrt(2)
eigenstates = np.eye(5)  # Energy eigenstates
energies = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # E_n = n + 0.5

outcomes, stats = simulate_measurement(psi, eigenstates, energies, 10000)
print(f"⟨E⟩ measured: {stats['mean']:.4f} (expected: 1.0)")
print(f"ΔE measured: {stats['std']:.4f} (expected: 0.5)")
```

### Quantum Monte Carlo Preview

Variational Monte Carlo estimates ground state energy:

$$E[\psi] = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle} = \int |\psi(x)|^2 E_L(x) dx$$

where $E_L(x) = H\psi(x)/\psi(x)$ is the local energy.

```python
def variational_monte_carlo(psi_trial, local_energy, x_range, n_samples=100000):
    """
    Variational Monte Carlo for ground state energy.

    Samples from |ψ|² using Metropolis algorithm.
    """
    rng = np.random.default_rng()

    # Metropolis sampling from |ψ|²
    x = 0.0  # Initial position
    step_size = 0.5
    samples = []
    local_energies = []

    for _ in range(n_samples + 1000):  # Include burn-in
        # Propose new position
        x_new = x + rng.uniform(-step_size, step_size)

        # Metropolis acceptance
        if abs(psi_trial(x_new))**2 / abs(psi_trial(x))**2 > rng.random():
            x = x_new

        if len(samples) >= 1000:  # After burn-in
            samples.append(x)
            local_energies.append(local_energy(x))

    samples = np.array(samples[:n_samples])
    local_energies = np.array(local_energies[:n_samples])

    E_mean = np.mean(local_energies)
    E_error = np.std(local_energies) / np.sqrt(n_samples)

    return E_mean, E_error, samples

# Example: Harmonic oscillator ground state
# Trial function: ψ(x) = exp(-αx²)
alpha = 0.5  # Variational parameter (optimal is 0.5)

psi_trial = lambda x: np.exp(-alpha * x**2)

def local_energy(x):
    # E_L = Hψ/ψ = -½ψ''/ψ + ½x²
    # For ψ = exp(-αx²): ψ'' = (4α²x² - 2α)ψ
    return alpha - 2*alpha**2*x**2 + 0.5*x**2

E, E_err, samples = variational_monte_carlo(psi_trial, local_energy, (-10, 10))
print(f"Ground state energy: {E:.6f} ± {E_err:.6f} (exact: 0.5)")
```

---

## Worked Examples

### Example 1: Central Limit Theorem Demonstration

```python
def demonstrate_clt(n_samples_per_mean=30, n_means=10000):
    """
    Demonstrate Central Limit Theorem.

    Average of n_samples from ANY distribution → Normal as n→∞
    """
    rng = np.random.default_rng(42)

    # Start with uniform distribution [0, 1]
    # True mean = 0.5, true variance = 1/12
    true_mean = 0.5
    true_var = 1/12

    # Generate many sample means
    sample_means = np.zeros(n_means)
    for i in range(n_means):
        samples = rng.uniform(0, 1, n_samples_per_mean)
        sample_means[i] = np.mean(samples)

    # CLT predicts: sample means ~ N(μ, σ²/n)
    expected_mean = true_mean
    expected_std = np.sqrt(true_var / n_samples_per_mean)

    actual_mean = np.mean(sample_means)
    actual_std = np.std(sample_means)

    print("Central Limit Theorem Demonstration")
    print("-" * 40)
    print(f"Samples per mean: {n_samples_per_mean}")
    print(f"Number of means: {n_means}")
    print(f"\nExpected mean: {expected_mean:.6f}")
    print(f"Actual mean:   {actual_mean:.6f}")
    print(f"\nExpected std:  {expected_std:.6f}")
    print(f"Actual std:    {actual_std:.6f}")

    return sample_means

means = demonstrate_clt()
```

### Example 2: Bootstrap Error Estimation

```python
def bootstrap_error(data, statistic_func, n_bootstrap=10000, confidence=0.95):
    """
    Estimate error using bootstrap resampling.

    Parameters
    ----------
    data : ndarray
        Original data
    statistic_func : callable
        Function to compute statistic of interest
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level for interval

    Returns
    -------
    estimate : float
        Point estimate
    ci_low, ci_high : float
        Confidence interval bounds
    """
    rng = np.random.default_rng()
    n = len(data)

    # Generate bootstrap statistics
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample with replacement
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(resample)

    # Point estimate
    estimate = statistic_func(data)

    # Confidence interval
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return estimate, ci_low, ci_high

# Example: Estimate mean and confidence interval
data = np.random.normal(5, 2, 100)  # 100 samples from N(5, 4)
mean_est, ci_low, ci_high = bootstrap_error(data, np.mean)
print(f"Mean: {mean_est:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
```

### Example 3: Random Walk and Diffusion

```python
def random_walk_1d(n_steps, n_walkers=1000, step_size=1.0):
    """
    Simulate 1D random walk (models quantum diffusion).

    Returns positions of walkers at each time step.
    """
    rng = np.random.default_rng()

    # Each step: +step_size or -step_size with equal probability
    steps = rng.choice([-step_size, step_size], size=(n_steps, n_walkers))

    # Cumulative sum gives positions
    positions = np.cumsum(steps, axis=0)

    return positions

# Simulate
n_steps = 1000
positions = random_walk_1d(n_steps, n_walkers=10000)

# Analyze diffusion
times = np.arange(1, n_steps + 1)
mean_sq_displacement = np.mean(positions**2, axis=1)

# Should scale as ⟨x²⟩ = Dt (diffusion equation)
# For random walk: D = step_size²/dt = 1
print("Random Walk Diffusion Analysis")
print("-" * 40)
print(f"Time    ⟨x²⟩     √(⟨x²⟩)   Theory")
for t in [10, 100, 500, 1000]:
    msd = mean_sq_displacement[t-1]
    print(f"{t:>4}    {msd:>7.2f}  {np.sqrt(msd):>7.2f}   {np.sqrt(t):>7.2f}")
```

---

## Practice Problems

### Direct Application

**Problem 1:** Generate 1000 samples from a Maxwell-Boltzmann speed distribution $f(v) \propto v^2 e^{-v^2/2}$ using rejection sampling.

**Problem 2:** Use Monte Carlo to estimate $\pi$ by computing the ratio of points falling inside a unit circle to points in a unit square.

**Problem 3:** Compute the mean and standard deviation of the energy for a thermal state $\rho = e^{-\beta H}/Z$ using random sampling.

### Intermediate

**Problem 4:** Implement the Box-Muller transform to generate normal random numbers from uniform randoms.

**Problem 5:** Use importance sampling to compute $\langle x^4 \rangle$ for a Gaussian wave function, sampling from the Gaussian.

**Problem 6:** Simulate 1000 measurements of the spin-z operator on the state $(|↑⟩ + |↓⟩)/\sqrt{2}$ and verify the variance.

### Challenging

**Problem 7:** Implement the Metropolis-Hastings algorithm to sample from the probability density $|\psi(x)|^2$ for the harmonic oscillator ground state.

**Problem 8:** Estimate the ground state energy of a quartic oscillator $V(x) = x^4$ using variational Monte Carlo with a Gaussian trial function.

**Problem 9:** Simulate the quantum Zeno effect: show that frequent measurements slow down the evolution of a quantum state.

---

## Computational Lab

### Quantum Measurement and Monte Carlo

```python
"""
Day 257 Computational Lab: Random Numbers and Quantum Simulation
================================================================

This lab demonstrates Monte Carlo methods for quantum physics.
"""

import numpy as np
from typing import Tuple, Callable, List
import time

# ============================================================
# RANDOM NUMBER GENERATION
# ============================================================

class QuantumRNG:
    """
    Random number generator with quantum-relevant methods.
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def uniform(self, low: float, high: float, size: int = None) -> np.ndarray:
        return self.rng.uniform(low, high, size)

    def normal(self, mean: float = 0, std: float = 1, size: int = None) -> np.ndarray:
        return self.rng.normal(mean, std, size)

    def complex_normal(self, size: int = None) -> np.ndarray:
        """Generate complex numbers with real and imag parts from N(0,1)."""
        return self.rng.standard_normal(size) + 1j * self.rng.standard_normal(size)

    def measure(self, probabilities: np.ndarray, size: int = 1) -> np.ndarray:
        """Sample measurement outcome according to Born rule."""
        return self.rng.choice(len(probabilities), size=size, p=probabilities)

    def haar_random_state(self, dim: int) -> np.ndarray:
        """Generate a random state uniformly distributed on Bloch sphere."""
        # For higher dimensions, use Haar measure
        z = self.complex_normal(dim)
        return z / np.linalg.norm(z)

# ============================================================
# QUANTUM MEASUREMENT SIMULATION
# ============================================================

def born_probabilities(psi: np.ndarray, eigenstates: np.ndarray) -> np.ndarray:
    """Compute Born rule probabilities."""
    overlaps = eigenstates.T.conj() @ psi
    probs = np.abs(overlaps)**2
    return probs / np.sum(probs)  # Normalize

def simulate_measurements(psi: np.ndarray,
                          eigenstates: np.ndarray,
                          eigenvalues: np.ndarray,
                          n_measurements: int,
                          rng: QuantumRNG = None) -> dict:
    """
    Simulate repeated quantum measurements.

    Returns statistics and raw outcomes.
    """
    if rng is None:
        rng = QuantumRNG()

    # Compute probabilities
    probs = born_probabilities(psi, eigenstates)

    # Sample outcomes
    indices = rng.measure(probs, n_measurements)
    outcomes = eigenvalues[indices]

    # Compute statistics
    results = {
        'outcomes': outcomes,
        'indices': indices,
        'probabilities': probs,
        'mean': np.mean(outcomes),
        'variance': np.var(outcomes),
        'std': np.std(outcomes),
        'n_measurements': n_measurements
    }

    # Compare with quantum prediction
    results['quantum_mean'] = np.sum(probs * eigenvalues)
    results['quantum_variance'] = np.sum(probs * eigenvalues**2) - results['quantum_mean']**2

    return results

def measurement_statistics_convergence(psi: np.ndarray,
                                       eigenstates: np.ndarray,
                                       eigenvalues: np.ndarray,
                                       n_max: int = 10000) -> dict:
    """
    Study convergence of measurement statistics with sample size.
    """
    rng = QuantumRNG(42)
    probs = born_probabilities(psi, eigenstates)

    # True values
    true_mean = np.sum(probs * eigenvalues)
    true_var = np.sum(probs * eigenvalues**2) - true_mean**2

    # Generate all measurements at once
    all_indices = rng.measure(probs, n_max)
    all_outcomes = eigenvalues[all_indices]

    # Compute running statistics
    n_values = np.logspace(1, np.log10(n_max), 50, dtype=int)
    n_values = np.unique(n_values)

    means = []
    stds = []
    errors = []

    for n in n_values:
        sample = all_outcomes[:n]
        means.append(np.mean(sample))
        stds.append(np.std(sample))
        errors.append(np.abs(np.mean(sample) - true_mean))

    return {
        'n_values': n_values,
        'means': np.array(means),
        'stds': np.array(stds),
        'errors': np.array(errors),
        'true_mean': true_mean,
        'true_std': np.sqrt(true_var)
    }

# ============================================================
# MONTE CARLO INTEGRATION
# ============================================================

def monte_carlo_expectation(psi_func: Callable,
                            operator_func: Callable,
                            x_range: Tuple[float, float],
                            n_samples: int = 100000) -> Tuple[float, float]:
    """
    Compute ⟨ψ|Ô|ψ⟩ using Monte Carlo integration.

    Assumes psi is normalized.
    """
    rng = QuantumRNG()
    x_min, x_max = x_range

    # Sample uniformly
    x = rng.uniform(x_min, x_max, n_samples)

    # Compute integrand: ψ*(x) Ô ψ(x)
    psi = psi_func(x)
    O_psi = operator_func(psi, x)
    integrand = np.conj(psi) * O_psi

    # Estimate integral
    volume = x_max - x_min
    integral = volume * np.mean(integrand)
    error = volume * np.std(integrand) / np.sqrt(n_samples)

    return np.real(integral), np.real(error)

def importance_sampling_expectation(psi_func: Callable,
                                    operator_func: Callable,
                                    x_range: Tuple[float, float],
                                    n_samples: int = 100000) -> Tuple[float, float]:
    """
    Compute ⟨ψ|Ô|ψ⟩ using importance sampling.

    Samples from |ψ|² using rejection sampling.
    """
    rng = QuantumRNG()
    x_min, x_max = x_range

    # Find maximum of |ψ|² for rejection sampling
    x_test = np.linspace(x_min, x_max, 1000)
    psi_test = psi_func(x_test)
    max_prob = np.max(np.abs(psi_test)**2) * 1.1  # Add margin

    # Rejection sampling from |ψ|²
    samples = []
    while len(samples) < n_samples:
        x = rng.uniform(x_min, x_max)
        u = rng.uniform(0, max_prob)
        if u < np.abs(psi_func(x))**2:
            samples.append(x)

    x_samples = np.array(samples)

    # Compute expectation (integrand is O(x) since we sample from |ψ|²)
    psi_samples = psi_func(x_samples)
    # ⟨O⟩ = ∫|ψ|² O dx when sampling from |ψ|², need O·ψ/|ψ| = O·sign(ψ)
    # For real ψ, this simplifies
    O_samples = operator_func(psi_samples, x_samples) / psi_samples

    expectation = np.mean(np.real(O_samples))
    error = np.std(np.real(O_samples)) / np.sqrt(n_samples)

    return expectation, error

# ============================================================
# METROPOLIS ALGORITHM
# ============================================================

def metropolis_sample(target_pdf: Callable,
                      x_init: float,
                      step_size: float,
                      n_samples: int,
                      burn_in: int = 1000) -> np.ndarray:
    """
    Metropolis algorithm to sample from target distribution.

    Parameters
    ----------
    target_pdf : callable
        Unnormalized probability density (e.g., |ψ|²)
    x_init : float
        Initial position
    step_size : float
        Proposal step size
    n_samples : int
        Number of samples to generate
    burn_in : int
        Number of initial samples to discard

    Returns
    -------
    samples : ndarray
        Array of samples from target distribution
    """
    rng = QuantumRNG()

    x = x_init
    samples = []
    accepted = 0

    for i in range(n_samples + burn_in):
        # Propose new position
        x_new = x + rng.uniform(-step_size, step_size)

        # Acceptance probability
        p_accept = min(1, target_pdf(x_new) / target_pdf(x))

        if rng.uniform(0, 1) < p_accept:
            x = x_new
            if i >= burn_in:
                accepted += 1

        if i >= burn_in:
            samples.append(x)

    acceptance_rate = accepted / n_samples
    return np.array(samples), acceptance_rate

# ============================================================
# VARIATIONAL MONTE CARLO
# ============================================================

def vmc_energy(trial_wf: Callable,
               local_energy: Callable,
               x_range: Tuple[float, float],
               n_samples: int = 50000) -> Tuple[float, float]:
    """
    Variational Monte Carlo energy estimate.

    E[ψ] = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ = ∫|ψ|² E_L dx

    where E_L = Hψ/ψ is the local energy.
    """
    # Target: |ψ|²
    target = lambda x: np.abs(trial_wf(x))**2

    # Metropolis sampling
    samples, acc_rate = metropolis_sample(
        target, x_init=0.0, step_size=1.0,
        n_samples=n_samples, burn_in=5000
    )

    # Compute local energies
    E_local = local_energy(samples)

    # Statistics
    E_mean = np.mean(E_local)
    E_error = np.std(E_local) / np.sqrt(n_samples)

    return E_mean, E_error, acc_rate

# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Day 257: Random Numbers and Quantum Simulation")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Basic Random Number Generation
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. RANDOM NUMBER GENERATION")
    print("=" * 70)

    rng = QuantumRNG(42)

    print("\nUniform [0, 1):", rng.uniform(0, 1, 5))
    print("Normal (μ=0, σ=1):", rng.normal(0, 1, 5))
    print("Complex normal:", rng.complex_normal(3))
    print("Haar random state (d=3):", rng.haar_random_state(3))

    # --------------------------------------------------------
    # 2. Quantum Measurement Simulation
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. QUANTUM MEASUREMENT SIMULATION")
    print("=" * 70)

    # Two-level system: |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
    theta = np.pi / 3  # 60 degrees
    psi = np.array([np.cos(theta/2), np.sin(theta/2)])
    eigenstates = np.eye(2)
    eigenvalues = np.array([0, 1])  # Measure |1⟩ vs |0⟩

    print(f"\nState: cos({np.degrees(theta)/2:.1f}°)|0⟩ + sin({np.degrees(theta)/2:.1f}°)|1⟩")

    results = simulate_measurements(psi, eigenstates, eigenvalues, 10000)

    print(f"\nSimulated (N=10000):")
    print(f"  P(0) = {np.sum(results['indices']==0)/10000:.4f}")
    print(f"  P(1) = {np.sum(results['indices']==1)/10000:.4f}")
    print(f"  ⟨A⟩ = {results['mean']:.4f}")
    print(f"  ΔA = {results['std']:.4f}")

    print(f"\nQuantum prediction:")
    print(f"  P(0) = {np.cos(theta/2)**2:.4f}")
    print(f"  P(1) = {np.sin(theta/2)**2:.4f}")
    print(f"  ⟨A⟩ = {results['quantum_mean']:.4f}")
    print(f"  ΔA = {np.sqrt(results['quantum_variance']):.4f}")

    # --------------------------------------------------------
    # 3. Convergence of Measurement Statistics
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. STATISTICAL CONVERGENCE")
    print("=" * 70)

    conv = measurement_statistics_convergence(psi, eigenstates, eigenvalues, 100000)

    print("\nConvergence of ⟨A⟩ to true value:")
    print(f"{'N':>8} {'Measured':>12} {'Error':>12}")
    print("-" * 36)
    for i in range(0, len(conv['n_values']), 10):
        n = conv['n_values'][i]
        mean = conv['means'][i]
        err = conv['errors'][i]
        print(f"{n:>8} {mean:>12.6f} {err:>12.6f}")

    print(f"\nTrue value: {conv['true_mean']:.6f}")

    # --------------------------------------------------------
    # 4. Monte Carlo Integration
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. MONTE CARLO INTEGRATION")
    print("=" * 70)

    # Harmonic oscillator ground state
    def psi_ho(x):
        return (1/np.pi)**0.25 * np.exp(-x**2/2)

    def x_operator(psi, x):
        return x * psi

    def x2_operator(psi, x):
        return x**2 * psi

    # ⟨x⟩ (should be 0)
    x_mean, x_err = monte_carlo_expectation(psi_ho, x_operator, (-10, 10))
    print(f"\n⟨x⟩ = {x_mean:.6f} ± {x_err:.6f} (exact: 0)")

    # ⟨x²⟩ (should be 0.5)
    x2_mean, x2_err = monte_carlo_expectation(psi_ho, x2_operator, (-10, 10))
    print(f"⟨x²⟩ = {x2_mean:.6f} ± {x2_err:.6f} (exact: 0.5)")

    # --------------------------------------------------------
    # 5. Variational Monte Carlo
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. VARIATIONAL MONTE CARLO")
    print("=" * 70)

    # Harmonic oscillator with trial function ψ(x) = e^(-αx²)
    alpha = 0.5  # Variational parameter

    trial_wf = lambda x: np.exp(-alpha * x**2)

    def local_energy_ho(x):
        # E_L = Hψ/ψ = -½ψ''/ψ + ½x²
        # For ψ = e^(-αx²): E_L = α - 2α²x² + ½x²
        return alpha - 2*alpha**2*x**2 + 0.5*x**2

    print(f"\nVariational parameter α = {alpha}")

    E_vmc, E_err, acc = vmc_energy(trial_wf, local_energy_ho, (-10, 10))
    print(f"VMC energy: {E_vmc:.6f} ± {E_err:.6f}")
    print(f"Exact ground state: 0.500000")
    print(f"Acceptance rate: {acc:.2%}")

    # Scan over α values
    print("\nVariational scan over α:")
    print(f"{'α':>6} {'E(α)':>12} {'Error':>12}")
    print("-" * 32)
    for alpha_test in [0.3, 0.4, 0.5, 0.6, 0.7]:
        trial_test = lambda x, a=alpha_test: np.exp(-a * x**2)
        local_test = lambda x, a=alpha_test: a - 2*a**2*x**2 + 0.5*x**2
        E, err, _ = vmc_energy(trial_test, local_test, (-10, 10), n_samples=20000)
        print(f"{alpha_test:>6.2f} {E:>12.6f} {err:>12.6f}")

    # --------------------------------------------------------
    # 6. Metropolis Sampling Demo
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. METROPOLIS SAMPLING")
    print("=" * 70)

    # Sample from |ψ_0(x)|² = π^(-1/2) e^(-x²)
    target = lambda x: np.exp(-x**2) / np.sqrt(np.pi)

    samples, acc_rate = metropolis_sample(
        target, x_init=0.0, step_size=1.5,
        n_samples=50000, burn_in=5000
    )

    print(f"\nSampling from Gaussian |ψ|²:")
    print(f"  Acceptance rate: {acc_rate:.2%}")
    print(f"  Sample mean: {np.mean(samples):.6f} (expected: 0)")
    print(f"  Sample std: {np.std(samples):.6f} (expected: {1/np.sqrt(2):.6f})")

    print("\n" + "=" * 70)
    print("Lab complete! File I/O and performance on Day 258.")
    print("=" * 70)
```

---

## Summary

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `rng.random(n)` | Uniform [0,1) | `rng.random(100)` |
| `rng.normal(μ, σ, n)` | Gaussian | `rng.normal(0, 1, 100)` |
| `rng.choice(a, n, p)` | Weighted sampling | `rng.choice(3, 100, p=[0.2,0.3,0.5])` |
| `np.mean(a)` | Mean | `np.mean(data)` |
| `np.std(a)` | Standard deviation | `np.std(data)` |
| `np.histogram(a)` | Histogram | `np.histogram(data, bins=50)` |

### Main Takeaways

1. **Use Generator, not global state** — `rng = np.random.default_rng(seed)`
2. **Set seeds for reproducibility** — essential in scientific computing
3. **Monte Carlo = random sampling + averaging** — fundamental technique
4. **Importance sampling reduces variance** — sample where integrand is large
5. **Metropolis algorithm samples arbitrary distributions** — key for QMC
6. **Quantum measurement is inherently random** — Born rule gives probabilities

---

## Daily Checklist

- [ ] Can generate random numbers from various distributions
- [ ] Understand and use seeding for reproducibility
- [ ] Can compute statistical quantities efficiently
- [ ] Implemented Monte Carlo integration
- [ ] Understand importance sampling concept
- [ ] Simulated quantum measurements correctly
- [ ] Completed practice problems
- [ ] Ran lab successfully

---

## Preview: Day 258

Tomorrow we cover **File I/O and Performance** — essential for managing large computations. We'll learn:
- Saving and loading NumPy arrays efficiently
- Working with HDF5 for large datasets
- Profiling code to find bottlenecks
- Optimization strategies

This enables workflows where computation results persist across sessions.

---

*"God does not play dice with the universe, but we certainly can simulate it with Monte Carlo."*
