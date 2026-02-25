# Day 262: Root Finding and Optimization with SciPy

## Overview

**Day 262** | **Week 38** | **Month 10: Scientific Computing**

Today we master `scipy.optimize` for finding roots and minimizing functions. These tools are fundamental for quantum mechanics: finding eigenvalues from secular equations, minimizing energy in variational methods, and fitting models to experimental data. Optimization underlies everything from variational quantum eigensolvers to machine learning of quantum systems.

**Prerequisites:** Days 260-261 (integration, ODEs), calculus
**Outcome:** Find roots, minimize functions, fit data for quantum applications

---

## Schedule

| Time | Duration | Activity |
|------|----------|----------|
| Morning | 3 hours | Theory: Root finding, optimization algorithms |
| Afternoon | 3 hours | Practice: Variational methods, curve fitting |
| Evening | 2 hours | Lab: Variational ground state energy |

---

## Learning Objectives

By the end of Day 262, you will be able to:

1. **Find roots of nonlinear equations** using `fsolve`, `root`, `brentq`
2. **Minimize scalar functions** with `minimize_scalar`
3. **Minimize multivariate functions** using `minimize` with various methods
4. **Fit models to data** with `curve_fit` and least squares
5. **Apply variational principle** to find ground state energies
6. **Handle constraints** in optimization problems
7. **Choose appropriate algorithms** for different problem types

---

## Core Content

### 1. Root Finding: Single Variable

```python
import numpy as np
from scipy import optimize

# Find root of f(x) = x² - 2 (i.e., find √2)
def f(x):
    return x**2 - 2

# Brent's method (bracketing, guaranteed convergence)
root = optimize.brentq(f, 0, 2)
print(f"√2 = {root:.15f}")
print(f"Exact: {np.sqrt(2):.15f}")

# Newton's method (faster but needs good initial guess)
root_newton = optimize.newton(f, x0=1.5)
print(f"Newton: {root_newton:.15f}")

# With derivative (faster convergence)
def fprime(x):
    return 2*x

root_newton2 = optimize.newton(f, x0=1.5, fprime=fprime)
print(f"Newton with derivative: {root_newton2:.15f}")
```

### 2. Root Finding: Systems of Equations

```python
# Solve system: x² + y² = 1, x - y = 0.5
def equations(vars):
    x, y = vars
    return [x**2 + y**2 - 1, x - y - 0.5]

# fsolve: general-purpose solver
solution = optimize.fsolve(equations, x0=[0.5, 0.5])
print(f"Solution: x={solution[0]:.6f}, y={solution[1]:.6f}")

# Verify
residual = equations(solution)
print(f"Residual: {residual}")

# root: more options and information
result = optimize.root(equations, x0=[0.5, 0.5], method='hybr')
print(f"Success: {result.success}")
print(f"Solution: {result.x}")
```

### 3. Scalar Minimization

```python
# Minimize f(x) = (x-3)² + 1
def f(x):
    return (x - 3)**2 + 1

# Bounded minimization (golden section / Brent)
result = optimize.minimize_scalar(f, bounds=(0, 10), method='bounded')
print(f"Minimum at x = {result.x:.6f}")
print(f"Minimum value = {result.fun:.6f}")

# Unbounded (Brent's method)
result2 = optimize.minimize_scalar(f, bracket=(0, 2, 5))
print(f"Brent: x = {result2.x:.6f}")

# With complex function
def double_well(x):
    return x**4 - 2*x**2 + 0.5*x

# Find local minima
result_left = optimize.minimize_scalar(double_well, bounds=(-2, 0), method='bounded')
result_right = optimize.minimize_scalar(double_well, bounds=(0, 2), method='bounded')
print(f"Left minimum: x = {result_left.x:.4f}, f = {result_left.fun:.4f}")
print(f"Right minimum: x = {result_right.x:.4f}, f = {result_right.fun:.4f}")
```

### 4. Multivariate Minimization

```python
# Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# Nelder-Mead (derivative-free, robust)
result = optimize.minimize(rosenbrock, x0=[0, 0], method='Nelder-Mead')
print(f"Nelder-Mead: {result.x}, iterations: {result.nit}")

# BFGS (quasi-Newton, uses gradient)
result_bfgs = optimize.minimize(rosenbrock, x0=[0, 0], method='BFGS')
print(f"BFGS: {result_bfgs.x}, iterations: {result_bfgs.nit}")

# With analytical gradient
def rosenbrock_grad(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

result_grad = optimize.minimize(rosenbrock, x0=[0, 0], method='BFGS', jac=rosenbrock_grad)
print(f"BFGS+grad: {result_grad.x}, iterations: {result_grad.nit}")

# Comparison of methods
methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B']
for method in methods:
    res = optimize.minimize(rosenbrock, x0=[-1, 1], method=method)
    print(f"{method:12s}: f={res.fun:.2e}, nfev={res.nfev:4d}")
```

### 5. Constrained Optimization

```python
# Minimize f(x,y) = x² + y² subject to x + y = 1
def objective(x):
    return x[0]**2 + x[1]**2

# Equality constraint: x + y - 1 = 0
def constraint_eq(x):
    return x[0] + x[1] - 1

constraints = {'type': 'eq', 'fun': constraint_eq}

result = optimize.minimize(objective, x0=[0.5, 0.5], constraints=constraints)
print(f"Constrained minimum: {result.x}")
print(f"Minimum value: {result.fun:.6f}")
print(f"Constraint satisfied: {abs(sum(result.x) - 1) < 1e-6}")

# Inequality constraints: x ≥ 0, y ≥ 0, x + y ≤ 1
def ineq1(x): return x[0]       # x ≥ 0
def ineq2(x): return x[1]       # y ≥ 0
def ineq3(x): return 1 - x[0] - x[1]  # x + y ≤ 1

constraints = [
    {'type': 'ineq', 'fun': ineq1},
    {'type': 'ineq', 'fun': ineq2},
    {'type': 'ineq', 'fun': ineq3}
]

# Different objective
def objective2(x):
    return -(x[0] + 2*x[1])  # Maximize x + 2y → minimize -(x + 2y)

result = optimize.minimize(objective2, x0=[0.1, 0.1], method='SLSQP', constraints=constraints)
print(f"LP solution: {result.x}")
```

### 6. Curve Fitting

```python
from scipy.optimize import curve_fit

# Generate noisy data
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_true = 2.5 * np.exp(-0.3 * x_data) + 0.5
y_data = y_true + 0.1 * np.random.randn(len(x_data))

# Model: y = A * exp(-k * x) + c
def model(x, A, k, c):
    return A * np.exp(-k * x) + c

# Fit
popt, pcov = curve_fit(model, x_data, y_data, p0=[1, 0.1, 0])
A_fit, k_fit, c_fit = popt
errors = np.sqrt(np.diag(pcov))

print(f"Fitted parameters:")
print(f"  A = {A_fit:.4f} ± {errors[0]:.4f} (true: 2.5)")
print(f"  k = {k_fit:.4f} ± {errors[1]:.4f} (true: 0.3)")
print(f"  c = {c_fit:.4f} ± {errors[2]:.4f} (true: 0.5)")

# With bounds
popt_bounded, _ = curve_fit(model, x_data, y_data, p0=[1, 0.1, 0],
                            bounds=([0, 0, -1], [10, 1, 1]))
```

### 7. Least Squares

```python
from scipy.optimize import least_squares

# More control than curve_fit
def residuals(params, x, y):
    A, k, c = params
    return y - A * np.exp(-k * x) - c

result = least_squares(residuals, x0=[1, 0.1, 0], args=(x_data, y_data))
print(f"Least squares: {result.x}")

# Robust fitting (handles outliers)
result_robust = least_squares(residuals, x0=[1, 0.1, 0], args=(x_data, y_data),
                              loss='soft_l1')  # Robust loss function
```

---

## Quantum Mechanics Connection

### Variational Principle

The variational principle states:
$$E[\psi] = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle} \geq E_0$$

with equality when $\psi = \psi_0$ (ground state).

```python
from scipy import integrate, optimize

def variational_energy(params, potential, x_range=(-10, 10)):
    """
    Compute variational energy for Gaussian trial function.

    ψ(x) = (2α/π)^(1/4) exp(-αx²)
    """
    alpha = params[0]
    if alpha <= 0:
        return 1e10  # Return large value for invalid α

    # Normalized Gaussian
    def psi(x):
        return (2*alpha/np.pi)**0.25 * np.exp(-alpha * x**2)

    # Kinetic energy: ⟨T⟩ = ∫ψ*(-ℏ²/2m d²/dx²)ψ dx = ℏ²α/(2m) for Gaussian
    # In natural units (ℏ=m=1): ⟨T⟩ = α/2
    T = alpha / 2

    # Potential energy: ⟨V⟩ = ∫|ψ|² V(x) dx
    integrand = lambda x: psi(x)**2 * potential(x)
    V, _ = integrate.quad(integrand, *x_range)

    return T + V

# Harmonic oscillator: V(x) = ½x²
V_harmonic = lambda x: 0.5 * x**2

result = optimize.minimize(variational_energy, x0=[0.3],
                           args=(V_harmonic,), method='Nelder-Mead')
alpha_opt = result.x[0]
E_var = result.fun

print(f"Variational ground state (harmonic oscillator):")
print(f"  Optimal α = {alpha_opt:.6f} (exact: 0.5)")
print(f"  E_var = {E_var:.6f} (exact: 0.5)")
```

### Finding Eigenvalues from Secular Equation

The secular equation $\det(H - EI) = 0$ gives eigenvalues:

```python
def secular_equation(E, H):
    """Secular equation: det(H - E*I) = 0"""
    return np.linalg.det(H - E * np.eye(len(H)))

# 3x3 Hamiltonian
H = np.array([
    [1.0, 0.5, 0.0],
    [0.5, 2.0, 0.5],
    [0.0, 0.5, 3.0]
])

# Find eigenvalues by root finding
# (In practice, use np.linalg.eigh, but this demonstrates the concept)
E_roots = []
for E_guess in [0.5, 2.0, 3.5]:
    E = optimize.brentq(secular_equation, E_guess - 0.5, E_guess + 0.5, args=(H,))
    E_roots.append(E)

E_numpy = np.linalg.eigvalsh(H)

print("Eigenvalues:")
print(f"  Root finding: {sorted(E_roots)}")
print(f"  NumPy eigh:   {list(E_numpy)}")
```

### Variational Method with Multiple Parameters

```python
def variational_hydrogen_like(params, Z=1):
    """
    Variational calculation for hydrogen-like atom.

    Trial function: ψ(r) = (β³/π)^(1/2) exp(-βr)

    ⟨T⟩ = β²/2
    ⟨V⟩ = -Zβ
    E(β) = β²/2 - Zβ

    Optimal: β = Z, E = -Z²/2
    """
    beta = params[0]
    if beta <= 0:
        return 1e10

    T = beta**2 / 2
    V = -Z * beta

    return T + V

# Optimize for Z = 1 (hydrogen)
result = optimize.minimize(variational_hydrogen_like, x0=[0.5], args=(1,))
print(f"Hydrogen ground state:")
print(f"  Optimal β = {result.x[0]:.6f} (exact: 1.0)")
print(f"  E_var = {result.fun:.6f} (exact: -0.5)")

# Helium with two parameters
def variational_helium(params):
    """
    Variational calculation for helium (simplified).

    Trial: ψ = ψ₁(r₁)ψ₂(r₂) with effective Z
    """
    Z_eff = params[0]
    if Z_eff <= 0:
        return 1e10

    # One-electron energies
    E_1e = -Z_eff**2 / 2

    # Electron-electron repulsion (approximate)
    E_ee = 5 * Z_eff / 8  # Variational estimate

    # Total (2 electrons)
    return 2 * E_1e + E_ee + 2 * (2 - Z_eff) * Z_eff  # Nuclear attraction correction

result_He = optimize.minimize(variational_helium, x0=[1.5])
print(f"\nHelium ground state:")
print(f"  Optimal Z_eff = {result_He.x[0]:.4f} (≈ 1.69)")
print(f"  E_var = {result_He.fun:.4f} (exact: -2.904)")
```

### Fitting Quantum Oscillations

```python
def fit_rabi_oscillations():
    """Fit Rabi oscillation data to extract frequency."""

    # Generate synthetic data
    np.random.seed(42)
    t_data = np.linspace(0, 10, 100)
    omega_true = 2.5
    P1_true = np.sin(omega_true * t_data / 2)**2
    P1_data = P1_true + 0.05 * np.random.randn(len(t_data))
    P1_data = np.clip(P1_data, 0, 1)  # Probabilities in [0, 1]

    # Model
    def rabi_model(t, omega, offset, amplitude):
        return offset + amplitude * np.sin(omega * t / 2)**2

    # Fit
    popt, pcov = curve_fit(rabi_model, t_data, P1_data,
                           p0=[2.0, 0.0, 1.0],
                           bounds=([0, -0.5, 0], [10, 0.5, 2]))

    omega_fit, offset_fit, amp_fit = popt
    errors = np.sqrt(np.diag(pcov))

    print("Rabi oscillation fit:")
    print(f"  ω = {omega_fit:.4f} ± {errors[0]:.4f} (true: {omega_true})")
    print(f"  offset = {offset_fit:.4f} ± {errors[1]:.4f}")
    print(f"  amplitude = {amp_fit:.4f} ± {errors[2]:.4f}")

    return t_data, P1_data, popt

fit_rabi_oscillations()
```

---

## Worked Examples

### Example 1: Shooting Method for Eigenvalues

```python
def shooting_method_eigenvalue(V_func, E_guess, x_range, bc='zero'):
    """
    Find eigenvalue using shooting method.

    Integrate Schrödinger equation from left boundary and find E
    such that boundary condition at right is satisfied.
    """
    from scipy.integrate import solve_ivp

    x_min, x_max = x_range

    def schrodinger(x, y, E):
        psi, dpsi = y
        d2psi = 2 * (V_func(x) - E) * psi  # ℏ = m = 1
        return [dpsi, d2psi]

    def bc_residual(E):
        # Initial conditions at left boundary
        if bc == 'zero':
            y0 = [0, 1]  # ψ(x_min) = 0, ψ'(x_min) = 1
        elif bc == 'even':
            y0 = [1, 0]  # ψ(0) = 1, ψ'(0) = 0 (even parity)
        elif bc == 'odd':
            y0 = [0, 1]  # ψ(0) = 0, ψ'(0) = 1 (odd parity)

        sol = solve_ivp(schrodinger, [x_min, x_max], y0, args=(E,),
                        dense_output=True)
        psi_right = sol.y[0, -1]
        return psi_right

    # Find E where psi(x_max) = 0
    E = optimize.brentq(bc_residual, E_guess - 1, E_guess + 1)
    return E

# Harmonic oscillator
V = lambda x: 0.5 * x**2

print("Shooting method eigenvalues (harmonic oscillator):")
for n, E_guess in enumerate([0.5, 1.5, 2.5]):
    try:
        if n % 2 == 0:  # Even states
            E = shooting_method_eigenvalue(V, E_guess, (-5, 5), bc='even')
        else:  # Odd states
            E = shooting_method_eigenvalue(V, E_guess, (-5, 5), bc='odd')
        print(f"  n={n}: E = {E:.6f} (exact: {n + 0.5})")
    except ValueError:
        print(f"  n={n}: Failed to converge")
```

### Example 2: Hartree-Fock Self-Consistent Field

```python
def simple_hf_iteration():
    """
    Simplified Hartree-Fock for two electrons in 1D.

    Solve iteratively: h_eff[n] φ = ε φ
    where h_eff depends on φ through mean-field potential.
    """
    N = 100
    x = np.linspace(-5, 5, N)
    dx = x[1] - x[0]

    # External potential (harmonic)
    V_ext = 0.5 * x**2

    # Initial guess: Gaussian
    phi = np.exp(-x**2/2)
    phi /= np.sqrt(np.sum(phi**2) * dx)

    def build_hamiltonian(phi, V_ext, x, dx):
        """Build effective Hamiltonian with mean-field."""
        N = len(x)

        # Kinetic energy
        T = np.diag(np.full(N, 1/dx**2)) - \
            np.diag(np.full(N-1, 0.5/dx**2), 1) - \
            np.diag(np.full(N-1, 0.5/dx**2), -1)

        # Hartree potential (electron-electron, simplified)
        rho = phi**2  # Density
        V_hartree = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    V_hartree[i] += rho[j] / np.abs(x[i] - x[j]) * dx

        V_total = V_ext + V_hartree

        return T + np.diag(V_total)

    # Self-consistent iteration
    for iteration in range(20):
        H = build_hamiltonian(phi, V_ext, x, dx)
        energies, states = np.linalg.eigh(H)

        phi_new = states[:, 0]
        phi_new /= np.sqrt(np.sum(phi_new**2) * dx)

        # Check convergence
        diff = np.max(np.abs(np.abs(phi_new) - np.abs(phi)))
        phi = phi_new

        if diff < 1e-8:
            print(f"Converged in {iteration+1} iterations")
            break

    print(f"Ground state energy: {energies[0]:.6f}")
    return energies[0], phi

E_hf, phi_hf = simple_hf_iteration()
```

### Example 3: Global Optimization

```python
def global_optimization_demo():
    """
    Find global minimum of multi-modal function.
    """
    # Rastrigin function (many local minima)
    def rastrigin(x):
        A = 10
        return A * len(x) + sum(xi**2 - A*np.cos(2*np.pi*xi) for xi in x)

    # Local optimization (gets stuck)
    result_local = optimize.minimize(rastrigin, x0=[5, 5], method='BFGS')
    print(f"Local (BFGS): {result_local.x}, f = {result_local.fun:.4f}")

    # Basin hopping (global)
    result_bh = optimize.basinhopping(rastrigin, x0=[5, 5], niter=100)
    print(f"Basin hopping: {result_bh.x}, f = {result_bh.fun:.4f}")

    # Differential evolution (global, no initial guess)
    bounds = [(-10, 10), (-10, 10)]
    result_de = optimize.differential_evolution(rastrigin, bounds)
    print(f"Diff evolution: {result_de.x}, f = {result_de.fun:.4f}")

    print(f"Global minimum: [0, 0], f = 0")

global_optimization_demo()
```

---

## Practice Problems

### Direct Application

**Problem 1:** Find the roots of $x^3 - x - 1 = 0$ using `brentq` and `newton`.

**Problem 2:** Minimize the Himmelblau function $f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2$ and find all four local minima.

**Problem 3:** Fit an exponential decay model to radioactive decay data and extract the half-life with uncertainty.

### Intermediate

**Problem 4:** Use the variational method with a two-parameter trial function $\psi(x) = (1 + ax^2)e^{-bx^2}$ for the harmonic oscillator.

**Problem 5:** Find the ground state energy of the Morse potential using the shooting method.

**Problem 6:** Implement steepest descent optimization from scratch and compare with `minimize`.

### Challenging

**Problem 7:** Perform a variational calculation for the helium atom with electron correlation using a Hylleraas-type trial function.

**Problem 8:** Solve the Gross-Pitaevskii equation self-consistently for a 1D BEC in a harmonic trap.

**Problem 9:** Implement simulated annealing for finding the global minimum of a potential energy surface.

---

## Computational Lab

```python
"""
Day 262 Lab: Optimization for Quantum Mechanics
===============================================
"""

import numpy as np
from scipy import optimize, integrate
from typing import Callable, Tuple

# ============================================================
# VARIATIONAL METHOD FRAMEWORK
# ============================================================

class VariationalSolver:
    """
    Framework for variational calculations.
    """

    def __init__(self, potential: Callable, x_range: Tuple[float, float] = (-10, 10)):
        self.potential = potential
        self.x_range = x_range

    def trial_wavefunction(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Override in subclass."""
        raise NotImplementedError

    def kinetic_energy(self, params: np.ndarray) -> float:
        """Compute ⟨T⟩ numerically or analytically."""
        raise NotImplementedError

    def potential_energy(self, params: np.ndarray) -> float:
        """Compute ⟨V⟩."""
        def integrand(x):
            psi = self.trial_wavefunction(x, params)
            return np.abs(psi)**2 * self.potential(x)

        result, _ = integrate.quad(integrand, *self.x_range)
        return result

    def energy(self, params: np.ndarray) -> float:
        """Total variational energy."""
        return self.kinetic_energy(params) + self.potential_energy(params)

    def optimize(self, params0: np.ndarray, method: str = 'Nelder-Mead') -> dict:
        """Find optimal parameters."""
        result = optimize.minimize(self.energy, params0, method=method)
        return {
            'params': result.x,
            'energy': result.fun,
            'success': result.success,
            'iterations': result.nit
        }


class GaussianVariational(VariationalSolver):
    """Gaussian trial function: ψ(x) = N exp(-αx²)"""

    def trial_wavefunction(self, x, params):
        alpha = params[0]
        norm = (2*alpha/np.pi)**0.25
        return norm * np.exp(-alpha * x**2)

    def kinetic_energy(self, params):
        alpha = params[0]
        return alpha / 2  # Analytical result for Gaussian


class TwoParamVariational(VariationalSolver):
    """Two-parameter trial: ψ(x) = N(1 + ax²)exp(-bx²)"""

    def trial_wavefunction(self, x, params):
        a, b = params
        if b <= 0:
            return np.zeros_like(x)
        unnorm = (1 + a*x**2) * np.exp(-b * x**2)
        # Numerical normalization
        norm_sq, _ = integrate.quad(lambda x: ((1 + a*x**2) * np.exp(-b * x**2))**2,
                                    *self.x_range)
        return unnorm / np.sqrt(norm_sq)

    def kinetic_energy(self, params):
        a, b = params
        if b <= 0:
            return 1e10

        def integrand(x):
            # ψ'' for (1+ax²)e^(-bx²)
            psi = (1 + a*x**2) * np.exp(-b * x**2)
            dpsi = (2*a*x - 2*b*x*(1 + a*x**2)) * np.exp(-b * x**2)
            d2psi = (2*a - 2*b*(1 + a*x**2) - 2*b*x*(2*a*x - 2*b*x*(1 + a*x**2)) +
                     (2*a*x - 2*b*x*(1 + a*x**2))*(-2*b*x)) * np.exp(-b * x**2)
            return psi * (-0.5 * d2psi)  # ⟨ψ|T|ψ⟩ integrand (unnormalized)

        num, _ = integrate.quad(integrand, *self.x_range)
        norm_sq, _ = integrate.quad(
            lambda x: ((1 + a*x**2) * np.exp(-b * x**2))**2,
            *self.x_range
        )
        return num / norm_sq


# ============================================================
# DEMONSTRATIONS
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Day 262: Optimization for Quantum Mechanics")
    print("=" * 70)

    # 1. Harmonic oscillator variational
    print("\n1. VARIATIONAL: HARMONIC OSCILLATOR")
    print("-" * 40)

    V_ho = lambda x: 0.5 * x**2
    solver_gaussian = GaussianVariational(V_ho)

    result = solver_gaussian.optimize([0.3])
    print(f"Gaussian trial function:")
    print(f"  Optimal α = {result['params'][0]:.6f}")
    print(f"  E_var = {result['energy']:.6f}")
    print(f"  Exact E₀ = 0.500000")

    # Two-parameter
    solver_2param = TwoParamVariational(V_ho)
    result_2p = solver_2param.optimize([0.0, 0.5])
    print(f"\nTwo-parameter trial:")
    print(f"  Optimal (a, b) = {result_2p['params']}")
    print(f"  E_var = {result_2p['energy']:.6f}")

    # 2. Anharmonic oscillator
    print("\n2. VARIATIONAL: ANHARMONIC OSCILLATOR")
    print("-" * 40)

    V_anharmonic = lambda x: 0.5 * x**2 + 0.1 * x**4
    solver_anharmonic = GaussianVariational(V_anharmonic)

    result_anh = solver_anharmonic.optimize([0.5])
    print(f"V(x) = ½x² + 0.1x⁴")
    print(f"  Optimal α = {result_anh['params'][0]:.6f}")
    print(f"  E_var = {result_anh['energy']:.6f}")

    # 3. Curve fitting demo
    print("\n3. CURVE FITTING: DECAY DATA")
    print("-" * 40)

    np.random.seed(42)
    t = np.linspace(0, 5, 50)
    y_true = 10 * np.exp(-t / 2)
    y_data = y_true + 0.5 * np.random.randn(len(t))

    def decay_model(t, A, tau):
        return A * np.exp(-t / tau)

    popt, pcov = optimize.curve_fit(decay_model, t, y_data, p0=[8, 1])
    errors = np.sqrt(np.diag(pcov))

    print(f"Fitted: A = {popt[0]:.3f} ± {errors[0]:.3f} (true: 10)")
    print(f"        τ = {popt[1]:.3f} ± {errors[1]:.3f} (true: 2)")
    print(f"Half-life: {popt[1] * np.log(2):.3f} ± {errors[1] * np.log(2):.3f}")

    # 4. Root finding for eigenvalues
    print("\n4. ROOT FINDING: TRANSCENDENTAL EIGENVALUE")
    print("-" * 40)

    # Finite square well: tan(ka) = √((V₀-E)/E) for even states
    V0 = 10.0  # Well depth
    a = 1.0    # Half-width

    def eigenvalue_condition(E):
        if E <= 0 or E >= V0:
            return 1e10
        k = np.sqrt(2 * E)
        kappa = np.sqrt(2 * (V0 - E))
        return np.tan(k * a) - kappa / k

    # Find bound states
    print("Finite square well eigenvalues (V₀=10, a=1):")
    for E_guess in [1, 4, 8]:
        try:
            E = optimize.brentq(eigenvalue_condition, E_guess - 0.5, E_guess + 0.5)
            print(f"  E = {E:.6f}")
        except ValueError:
            pass

    print("\n" + "=" * 70)
    print("Lab complete! Linear algebra continues on Day 263.")
    print("=" * 70)
```

---

## Summary

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `brentq(f, a, b)` | Bracketed root | `optimize.brentq(f, 0, 1)` |
| `fsolve(f, x0)` | System roots | `optimize.fsolve(f, [1,1])` |
| `minimize_scalar(f)` | 1D minimization | `optimize.minimize_scalar(f)` |
| `minimize(f, x0)` | N-D minimization | `optimize.minimize(f, [0,0])` |
| `curve_fit(f, x, y)` | Fit model | `curve_fit(model, x, y)` |

### Optimization Methods

| Method | Type | Use Case |
|--------|------|----------|
| Nelder-Mead | Derivative-free | Robust, noisy functions |
| BFGS | Quasi-Newton | Smooth functions |
| L-BFGS-B | Bounded BFGS | Large-scale, bounded |
| SLSQP | Sequential QP | Constrained |
| Basin-hopping | Global | Multi-modal |

---

## Daily Checklist

- [ ] Can find roots with brentq, newton, fsolve
- [ ] Minimize scalar and multivariate functions
- [ ] Handle constraints in optimization
- [ ] Fit models to data with curve_fit
- [ ] Applied variational principle numerically
- [ ] Understand different optimization methods
- [ ] Completed practice problems
- [ ] Ran lab successfully

---

## Preview: Day 263

Tomorrow we explore **Advanced Linear Algebra** with `scipy.linalg`. We'll learn:
- Matrix exponentials for time evolution
- Matrix functions (sqrt, log, arbitrary)
- Specialized decompositions
- Solving structured linear systems

Essential for quantum dynamics and propagators!

---

*"Optimization is the art of making the best choice among infinitely many possibilities."*
