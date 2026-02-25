# Day 279: Monte Carlo Methods in Physics

## Schedule Overview
**Date**: Week 40, Day 6 (Saturday)
**Duration**: 7 hours
**Theme**: Stochastic Methods for Classical and Quantum Simulations

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Monte Carlo integration, Metropolis algorithm |
| Afternoon | 2.5 hours | Variational Monte Carlo, quantum applications |
| Evening | 1.5 hours | Computational lab: VMC for helium atom |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Implement Monte Carlo integration for multi-dimensional integrals
2. Understand and apply the Metropolis-Hastings algorithm
3. Simulate thermal equilibrium in classical systems
4. Apply Variational Monte Carlo to quantum problems
5. Estimate ground state energies stochastically

---

## Core Content

### 1. Monte Carlo Integration

For integrals over high-dimensional spaces:
$$I = \int_\Omega f(\mathbf{x}) d\mathbf{x} \approx \frac{V}{N}\sum_{i=1}^N f(\mathbf{x}_i)$$

where $$\mathbf{x}_i$$ are uniformly sampled points.

```python
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_integrate(func, bounds, n_samples=10000):
    """
    Monte Carlo integration over rectangular domain.

    Parameters
    ----------
    func : callable
        Function to integrate
    bounds : list of tuples
        [(x_min, x_max), (y_min, y_max), ...]
    n_samples : int
        Number of random samples
    """
    dim = len(bounds)

    # Generate random points
    samples = np.zeros((n_samples, dim))
    volume = 1.0
    for i, (lo, hi) in enumerate(bounds):
        samples[:, i] = np.random.uniform(lo, hi, n_samples)
        volume *= (hi - lo)

    # Evaluate function
    values = func(samples)

    # Estimate integral and error
    mean = np.mean(values)
    var = np.var(values)
    integral = volume * mean
    error = volume * np.sqrt(var / n_samples)

    return integral, error


# Example: Volume of unit sphere in d dimensions
def sphere_indicator(points):
    """Returns 1 if inside unit sphere, 0 otherwise."""
    r2 = np.sum(points**2, axis=1)
    return (r2 <= 1).astype(float)

# Test in different dimensions
print("Monte Carlo: Volume of Unit Sphere")
print("d   MC Estimate    Exact         Error")
for d in range(2, 7):
    bounds = [(-1, 1)] * d
    vol_mc, err = monte_carlo_integrate(sphere_indicator, bounds, n_samples=100000)

    # Exact: V_d = π^(d/2) / Γ(d/2 + 1)
    from scipy.special import gamma
    vol_exact = np.pi**(d/2) / gamma(d/2 + 1)

    print(f"{d}   {vol_mc:.4f} ± {err:.4f}   {vol_exact:.4f}    {abs(vol_mc-vol_exact)/vol_exact*100:.2f}%")
```

### 2. Importance Sampling

For integrals of the form $$I = \int f(x) p(x) dx$$ where $$p(x)$$ is a probability density:

$$I \approx \frac{1}{N}\sum_{i=1}^N f(x_i), \quad x_i \sim p(x)$$

```python
def importance_sampling_demo():
    """
    Demonstrate importance sampling vs uniform sampling.

    Compute ⟨x²⟩ for Gaussian distribution.
    """
    n_samples = 10000

    # Uniform sampling (inefficient)
    x_uniform = np.random.uniform(-10, 10, n_samples)
    gaussian = np.exp(-x_uniform**2 / 2) / np.sqrt(2*np.pi)
    integrand = x_uniform**2 * gaussian
    I_uniform = 20 * np.mean(integrand)  # Volume factor
    err_uniform = 20 * np.std(integrand) / np.sqrt(n_samples)

    # Importance sampling (sample from Gaussian directly)
    x_importance = np.random.normal(0, 1, n_samples)
    I_importance = np.mean(x_importance**2)
    err_importance = np.std(x_importance**2) / np.sqrt(n_samples)

    print("\nImportance Sampling Demo: ⟨x²⟩ for Gaussian")
    print(f"Uniform:     {I_uniform:.4f} ± {err_uniform:.4f}")
    print(f"Importance:  {I_importance:.4f} ± {err_importance:.4f}")
    print(f"Exact:       1.0000")
    print(f"Efficiency gain: {(err_uniform/err_importance)**2:.1f}x")

importance_sampling_demo()
```

### 3. Metropolis-Hastings Algorithm

For sampling from complex probability distributions:

```python
class MetropolisHastings:
    """
    Metropolis-Hastings MCMC sampler.
    """

    def __init__(self, target_log_prob, dim, step_size=1.0):
        """
        Parameters
        ----------
        target_log_prob : callable
            Log of target probability density
        dim : int
            Dimensionality
        step_size : float
            Proposal step size
        """
        self.log_prob = target_log_prob
        self.dim = dim
        self.step_size = step_size

    def sample(self, n_samples, x0=None, burn_in=1000):
        """Generate samples using Metropolis algorithm."""
        if x0 is None:
            x0 = np.zeros(self.dim)

        samples = []
        x = x0.copy()
        log_p = self.log_prob(x)
        n_accept = 0

        for i in range(n_samples + burn_in):
            # Propose new state
            x_new = x + self.step_size * np.random.randn(self.dim)
            log_p_new = self.log_prob(x_new)

            # Accept/reject
            log_alpha = log_p_new - log_p
            if np.log(np.random.rand()) < log_alpha:
                x = x_new
                log_p = log_p_new
                n_accept += 1

            if i >= burn_in:
                samples.append(x.copy())

        acceptance_rate = n_accept / (n_samples + burn_in)
        return np.array(samples), acceptance_rate


# Example: 2D Gaussian mixture
def gaussian_mixture_log_prob(x):
    """Log probability of 2D Gaussian mixture."""
    # Two Gaussians centered at (±2, 0)
    sigma = 0.5
    p1 = np.exp(-np.sum((x - np.array([2, 0]))**2) / (2*sigma**2))
    p2 = np.exp(-np.sum((x - np.array([-2, 0]))**2) / (2*sigma**2))
    return np.log(p1 + p2 + 1e-10)

sampler = MetropolisHastings(gaussian_mixture_log_prob, dim=2, step_size=0.5)
samples, acc_rate = sampler.sample(10000, burn_in=2000)

print(f"\nMetropolis-Hastings: Gaussian Mixture")
print(f"Acceptance rate: {acc_rate:.2%}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Samples
axes[0].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('MCMC Samples')

# Marginal x
axes[1].hist(samples[:, 0], bins=50, density=True, alpha=0.7)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Density')
axes[1].set_title('Marginal Distribution (x)')

# Trace plot
axes[2].plot(samples[:1000, 0])
axes[2].set_xlabel('Step')
axes[2].set_ylabel('x')
axes[2].set_title('Trace Plot')

plt.tight_layout()
plt.savefig('metropolis_hastings.png', dpi=150)
plt.show()
```

### 4. Variational Monte Carlo (VMC)

For quantum systems, VMC combines the variational principle with Monte Carlo:

$$E[\psi_T] = \frac{\langle\psi_T|\hat{H}|\psi_T\rangle}{\langle\psi_T|\psi_T\rangle} = \int |\psi_T|^2 E_L(\mathbf{r}) d\mathbf{r}$$

where the local energy is:
$$E_L(\mathbf{r}) = \frac{\hat{H}\psi_T(\mathbf{r})}{\psi_T(\mathbf{r})}$$

```python
class VariationalMonteCarlo:
    """
    Variational Monte Carlo for 1D quantum systems.
    """

    def __init__(self, V_func, trial_wf, trial_wf_deriv2, hbar=1, m=1):
        """
        Parameters
        ----------
        V_func : callable
            Potential energy V(x)
        trial_wf : callable
            Trial wave function ψ(x, params)
        trial_wf_deriv2 : callable
            Second derivative d²ψ/dx² (x, params)
        """
        self.V = V_func
        self.psi = trial_wf
        self.d2psi = trial_wf_deriv2
        self.hbar = hbar
        self.m = m

    def local_energy(self, x, params):
        """Compute local energy E_L = Hψ/ψ."""
        psi_val = self.psi(x, params)
        d2psi_val = self.d2psi(x, params)

        kinetic = -self.hbar**2 / (2 * self.m) * d2psi_val / psi_val
        potential = self.V(x)

        return kinetic + potential

    def sample_position(self, params, n_samples, step_size=1.0, burn_in=1000):
        """Sample positions from |ψ|² using Metropolis."""
        x = 0.0
        psi2 = np.abs(self.psi(x, params))**2
        samples = []
        n_accept = 0

        for i in range(n_samples + burn_in):
            # Propose
            x_new = x + step_size * np.random.randn()
            psi2_new = np.abs(self.psi(x_new, params))**2

            # Accept/reject
            if np.random.rand() < psi2_new / psi2:
                x = x_new
                psi2 = psi2_new
                n_accept += 1

            if i >= burn_in:
                samples.append(x)

        return np.array(samples), n_accept / (n_samples + burn_in)

    def compute_energy(self, params, n_samples=10000, step_size=1.0):
        """Compute variational energy estimate."""
        positions, acc_rate = self.sample_position(params, n_samples, step_size)
        local_energies = self.local_energy(positions, params)

        E_mean = np.mean(local_energies)
        E_var = np.var(local_energies)
        E_err = np.sqrt(E_var / n_samples)

        return E_mean, E_err, acc_rate

    def optimize(self, params_init, n_iter=50, n_samples=5000):
        """Optimize variational parameters."""
        params = params_init.copy()
        energies = []

        for i in range(n_iter):
            E, E_err, acc = self.compute_energy(params, n_samples)
            energies.append(E)

            # Simple gradient-free optimization
            # Try small perturbations
            delta = 0.1 * np.random.randn(len(params))
            params_new = params + delta
            E_new, _, _ = self.compute_energy(params_new, n_samples//2)

            if E_new < E:
                params = params_new

            if i % 10 == 0:
                print(f"Iter {i}: E = {E:.4f} ± {E_err:.4f}, params = {params}")

        return params, energies


# VMC for harmonic oscillator
def harmonic_V(x):
    return 0.5 * x**2

def gaussian_trial(x, params):
    alpha = params[0]
    return np.exp(-alpha * x**2)

def gaussian_trial_d2(x, params):
    alpha = params[0]
    return (4 * alpha**2 * x**2 - 2 * alpha) * np.exp(-alpha * x**2)

vmc = VariationalMonteCarlo(harmonic_V, gaussian_trial, gaussian_trial_d2)

# Test at optimal value (α = 0.5 for ground state)
E, E_err, acc = vmc.compute_energy([0.5], n_samples=50000)
print(f"\nVMC for Harmonic Oscillator:")
print(f"E(α=0.5) = {E:.4f} ± {E_err:.4f}")
print(f"Exact E₀ = 0.5000")
print(f"Acceptance rate: {acc:.2%}")

# Test at suboptimal value
E2, E_err2, _ = vmc.compute_energy([0.3], n_samples=50000)
print(f"E(α=0.3) = {E2:.4f} ± {E_err2:.4f} (should be > 0.5)")
```

### 5. Ising Model Simulation

```python
class IsingModel:
    """
    2D Ising model Monte Carlo simulation.
    """

    def __init__(self, L, J=1.0, h=0.0):
        """
        Parameters
        ----------
        L : int
            Lattice size (L x L)
        J : float
            Coupling constant
        h : float
            External field
        """
        self.L = L
        self.J = J
        self.h = h
        self.spins = np.random.choice([-1, 1], size=(L, L))

    def energy(self):
        """Compute total energy."""
        # Nearest neighbor interaction
        E_nn = -self.J * np.sum(
            self.spins * np.roll(self.spins, 1, axis=0) +
            self.spins * np.roll(self.spins, 1, axis=1)
        )
        # External field
        E_h = -self.h * np.sum(self.spins)
        return E_nn + E_h

    def magnetization(self):
        """Compute magnetization per spin."""
        return np.mean(self.spins)

    def metropolis_step(self, T):
        """Single Metropolis sweep."""
        for _ in range(self.L**2):
            # Random spin
            i, j = np.random.randint(0, self.L, 2)

            # Energy change if flipped
            neighbors = (
                self.spins[(i+1) % self.L, j] +
                self.spins[(i-1) % self.L, j] +
                self.spins[i, (j+1) % self.L] +
                self.spins[i, (j-1) % self.L]
            )
            dE = 2 * self.J * self.spins[i, j] * neighbors
            dE += 2 * self.h * self.spins[i, j]

            # Accept/reject
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                self.spins[i, j] *= -1

    def simulate(self, T, n_steps, n_equil=1000):
        """Run simulation at temperature T."""
        energies = []
        magnetizations = []

        # Equilibration
        for _ in range(n_equil):
            self.metropolis_step(T)

        # Measurement
        for _ in range(n_steps):
            self.metropolis_step(T)
            energies.append(self.energy() / self.L**2)
            magnetizations.append(np.abs(self.magnetization()))

        return np.array(energies), np.array(magnetizations)


# Phase transition study
L = 20
temperatures = np.linspace(1.5, 3.5, 15)
T_c = 2.269  # Critical temperature

avg_magnetizations = []
avg_energies = []

print("\nIsing Model Phase Transition:")
for T in temperatures:
    ising = IsingModel(L)
    E, M = ising.simulate(T, n_steps=5000, n_equil=2000)
    avg_magnetizations.append(np.mean(M))
    avg_energies.append(np.mean(E))
    print(f"T = {T:.2f}: ⟨|M|⟩ = {np.mean(M):.3f}, ⟨E⟩ = {np.mean(E):.3f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(temperatures, avg_magnetizations, 'o-')
axes[0].axvline(T_c, color='r', linestyle='--', label=f'$T_c$ = {T_c:.3f}')
axes[0].set_xlabel('Temperature T')
axes[0].set_ylabel('⟨|M|⟩')
axes[0].set_title('Magnetization')
axes[0].legend()

axes[1].plot(temperatures, avg_energies, 'o-')
axes[1].axvline(T_c, color='r', linestyle='--')
axes[1].set_xlabel('Temperature T')
axes[1].set_ylabel('⟨E⟩ per spin')
axes[1].set_title('Energy')

# Show configuration
ising_cold = IsingModel(50)
ising_cold.simulate(1.5, n_steps=5000)
axes[2].imshow(ising_cold.spins, cmap='binary')
axes[2].set_title('Configuration at T = 1.5 (ordered)')

plt.tight_layout()
plt.savefig('ising_model.png', dpi=150)
plt.show()
```

---

## Summary

### Key Monte Carlo Methods

| Method | Application |
|--------|-------------|
| Direct MC | Multi-dimensional integrals |
| Importance sampling | Efficient sampling |
| Metropolis-Hastings | Complex distributions |
| VMC | Quantum ground states |
| MCMC | Statistical mechanics |

### Key Equations

$$\boxed{I = \int f(x)p(x)dx \approx \frac{1}{N}\sum_i f(x_i), \quad x_i \sim p(x)}$$

$$\boxed{P(\text{accept}) = \min\left(1, \frac{p(x')}{p(x)}\right)}$$

$$\boxed{E_L = \frac{\hat{H}\psi_T}{\psi_T}}$$

---

## Daily Checklist

- [ ] Implemented Monte Carlo integration
- [ ] Applied importance sampling
- [ ] Built Metropolis-Hastings sampler
- [ ] Applied VMC to quantum systems
- [ ] Simulated Ising model phase transition
- [ ] Completed practice problems
