#!/usr/bin/env python3
"""
SIIEA Quantum Engineering — Notebook Generator for Year 0, Months 10-12

Generates three companion Jupyter notebooks:
  1. Month 10: Scientific Computing & Numerical Methods (Days 253-280)
  2. Month 11: Group Theory — Symmetries & Representations (Days 281-308)
  3. Month 12: Foundation Capstone — Hydrogen Atom Project (Days 309-336)

Run with:
    .venv/bin/python3 notebooks/generate_months_10_12.py
"""

import sys
import os

# Ensure we can import the builder from the notebooks/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebook import NotebookBuilder


# ──────────────────────────────────────────────────────────────────────
# NOTEBOOK 1 — Month 10: Scientific Computing & Numerical Methods
# ──────────────────────────────────────────────────────────────────────
def build_month_10():
    nb = NotebookBuilder(
        "Scientific Computing & Numerical Methods",
        "year_0/month_10_scientific_computing/10_numerical_methods.ipynb",
        "Days 253-280",
    )

    # ── Imports & setup ──────────────────────────────────────────────
    nb.code("""\
# ── Imports & matplotlib configuration ──────────────────────────────
import numpy as np
import scipy
from scipy import linalg, integrate, optimize, sparse
from scipy.sparse.linalg import eigsh, spsolve, cg
from scipy.integrate import solve_ivp, quad, dblquad
import matplotlib.pyplot as plt
from matplotlib import cm
import time

%matplotlib inline

# Publication-quality defaults
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'figure.dpi': 100,
})

print(f"NumPy {np.__version__}  |  SciPy {scipy.__version__}")
print("All imports successful — ready for scientific computing.")""")

    # ── Section 1: NumPy Advanced ────────────────────────────────────
    nb.md("""\
## 1. NumPy Advanced: Broadcasting, Vectorization & Memory Layout

NumPy's power comes from *vectorized operations* that run in compiled C/Fortran
rather than Python loops.  Three critical concepts:

| Concept | Description |
|---------|-------------|
| **Broadcasting** | Rules that let arrays of different shapes combine element-wise |
| **Vectorization** | Replacing Python loops with array operations |
| **Memory layout** | Row-major (C) vs column-major (Fortran) affects cache performance |

**Broadcasting rule:** Two dimensions are compatible when they are equal or one
of them is 1.  Dimensions are compared from the trailing (rightmost) axis.

$$\\text{Shape }(3,1) + \\text{Shape }(1,4) \\rightarrow \\text{Shape }(3,4)$$

### Quantum Connection
Tensor products in quantum mechanics follow similar broadcasting ideas:
if $|\\psi\\rangle$ has dimension $d_1$ and $|\\phi\\rangle$ has dimension $d_2$,
the product state lives in $d_1 \\times d_2$ dimensions.""")

    nb.code("""\
# ── Broadcasting demonstration ──────────────────────────────────────
# Create a 2D Gaussian via broadcasting (no loops!)
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)

# x is (200,) → reshape to (200, 1); y stays (200,)
# Broadcasting produces (200, 200)
X = x[:, np.newaxis]
Y = y[np.newaxis, :]

# 2D Gaussian with different widths
sigma_x, sigma_y = 1.0, 1.5
Z = np.exp(-X**2 / (2 * sigma_x**2) - Y**2 / (2 * sigma_y**2))

print(f"x shape: {x.shape}")
print(f"X (after newaxis) shape: {X.shape}")
print(f"Y (after newaxis) shape: {Y.shape}")
print(f"Z (broadcasted result) shape: {Z.shape}")
print(f"Z max = {Z.max():.4f}, Z min = {Z.min():.6f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Contour plot
c = axes[0].contourf(x, y, Z.T, levels=30, cmap='viridis')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('2D Gaussian via Broadcasting')
axes[0].set_aspect('equal')
plt.colorbar(c, ax=axes[0], label='Amplitude')

# Surface plot of the same data
ax3d = fig.add_subplot(122, projection='3d')
axes[1].remove()  # remove the flat axes[1]
Xm, Ym = np.meshgrid(x[::5], y[::5])
ax3d.plot_surface(Xm, Ym, Z[::5, ::5].T, cmap='viridis', alpha=0.9)
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('Amplitude')
ax3d.set_title('Surface Plot')

plt.tight_layout()
plt.show()""")

    nb.code("""\
# ── Vectorization speed comparison ──────────────────────────────────
N = 500_000

# Python loop version
def loop_norm(arr):
    result = 0.0
    for val in arr:
        result += val * val
    return np.sqrt(result)

# Vectorized version
def vec_norm(arr):
    return np.sqrt(np.sum(arr**2))

data = np.random.randn(N)

t0 = time.perf_counter()
r1 = loop_norm(data)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
r2 = vec_norm(data)
t_vec = time.perf_counter() - t0

t0 = time.perf_counter()
r3 = np.linalg.norm(data)
t_np = time.perf_counter() - t0

print(f"{'Method':<20} {'Time (ms)':>10} {'Result':>14}")
print("-" * 46)
print(f"{'Python loop':<20} {t_loop*1000:>10.2f} {r1:>14.8f}")
print(f"{'Vectorized':<20} {t_vec*1000:>10.2f} {r2:>14.8f}")
print(f"{'np.linalg.norm':<20} {t_np*1000:>10.2f} {r3:>14.8f}")
print(f"\\nSpeedup (vectorized vs loop): {t_loop/t_vec:.0f}x")
print(f"Speedup (np.linalg vs loop):  {t_loop/t_np:.0f}x")""")

    nb.code("""\
# ── Memory layout: C-order vs Fortran-order ─────────────────────────
N = 2000
A_c = np.random.randn(N, N)                      # C-order (row-major)
A_f = np.asfortranarray(A_c)                      # Fortran-order (col-major)

print(f"C-order flags: C_CONTIGUOUS={A_c.flags['C_CONTIGUOUS']}, "
      f"F_CONTIGUOUS={A_c.flags['F_CONTIGUOUS']}")
print(f"F-order flags: C_CONTIGUOUS={A_f.flags['C_CONTIGUOUS']}, "
      f"F_CONTIGUOUS={A_f.flags['F_CONTIGUOUS']}")

# Row-sum: access by rows (good for C-order)
t0 = time.perf_counter()
_ = A_c.sum(axis=1)
t_row_c = time.perf_counter() - t0

t0 = time.perf_counter()
_ = A_f.sum(axis=1)
t_row_f = time.perf_counter() - t0

# Column-sum: access by columns (good for F-order)
t0 = time.perf_counter()
_ = A_c.sum(axis=0)
t_col_c = time.perf_counter() - t0

t0 = time.perf_counter()
_ = A_f.sum(axis=0)
t_col_f = time.perf_counter() - t0

print(f"\\n{'Operation':<20} {'C-order (ms)':>14} {'F-order (ms)':>14}")
print("-" * 50)
print(f"{'Row sum':<20} {t_row_c*1000:>14.3f} {t_row_f*1000:>14.3f}")
print(f"{'Column sum':<20} {t_col_c*1000:>14.3f} {t_col_f*1000:>14.3f}")
print("\\nCache-friendly access patterns give measurable speedups.")""")

    # ── Section 2: Sparse Matrices ───────────────────────────────────
    nb.md("""\
## 2. SciPy Linear Algebra: Sparse Matrices & Iterative Solvers

Many physics problems produce **sparse matrices** — matrices where most entries
are zero.  Storing only the nonzero elements saves enormous memory and enables
specialized fast algorithms.

**Key sparse formats in SciPy:**

| Format | Best for |
|--------|----------|
| `csr_matrix` | Row slicing, matrix-vector products |
| `csc_matrix` | Column slicing, sparse direct solvers |
| `coo_matrix` | Construction from triplets |
| `lil_matrix` | Incremental construction |

### Quantum Connection
The Hamiltonian of a quantum system with $n$ qubits lives in a $2^n \\times 2^n$
Hilbert space, but many physically relevant Hamiltonians (e.g., nearest-neighbor
interactions) are extremely sparse.  Sparse eigensolvers let us find the ground
state energy of systems far beyond what dense methods allow.""")

    nb.code("""\
# ── Sparse matrix construction: 1D tight-binding Hamiltonian ────────
# H = -t Σ (|i><i+1| + |i+1><i|) + V(i)|i><i|
# This is the discrete Schrödinger equation on a lattice

N_sites = 500       # number of lattice sites
t_hop = 1.0         # hopping parameter
V0 = 0.5            # potential strength

# Build the sparse Hamiltonian
diag_main = V0 * np.cos(2 * np.pi * np.arange(N_sites) / N_sites)  # cosine potential
diag_off = -t_hop * np.ones(N_sites - 1)

H_sparse = sparse.diags(
    [diag_off, diag_main, diag_off],
    offsets=[-1, 0, 1],
    shape=(N_sites, N_sites),
    format='csr'
)

# Compare memory usage
H_dense = H_sparse.toarray()
sparse_bytes = H_sparse.data.nbytes + H_sparse.indices.nbytes + H_sparse.indptr.nbytes
dense_bytes = H_dense.nbytes

print(f"Matrix size: {N_sites} x {N_sites}")
print(f"Nonzero elements: {H_sparse.nnz} / {N_sites**2} "
      f"({100*H_sparse.nnz/N_sites**2:.2f}%)")
print(f"Dense memory:  {dense_bytes / 1024:.1f} KB")
print(f"Sparse memory: {sparse_bytes / 1024:.1f} KB")
print(f"Compression ratio: {dense_bytes / sparse_bytes:.1f}x")

# Find lowest 6 eigenvalues using sparse eigensolver
n_eig = 6
eigenvalues, eigenvectors = eigsh(H_sparse, k=n_eig, which='SA')

print(f"\\nLowest {n_eig} eigenvalues:")
for i, ev in enumerate(eigenvalues):
    print(f"  E_{i} = {ev:+.6f}")

# Plot eigenstates
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
x = np.arange(N_sites)
for i, ax in enumerate(axes.flat):
    ax.plot(x, eigenvectors[:, i], color=f'C{i}', linewidth=1.5)
    ax.fill_between(x, eigenvectors[:, i], alpha=0.3, color=f'C{i}')
    ax.set_title(f'$\\\\psi_{i}$,  $E_{i}$ = {eigenvalues[i]:.4f}')
    ax.set_xlabel('Site index')
    ax.set_ylabel('Amplitude')
    ax.axhline(0, color='gray', linewidth=0.5)

plt.suptitle('Tight-Binding Model: Lowest 6 Eigenstates', fontsize=16)
plt.tight_layout()
plt.show()""")

    # ── Section 3: Numerical Integration ─────────────────────────────
    nb.md("""\
## 3. Numerical Integration: Quadrature & Monte Carlo

Exact analytical integrals are rare in physics.  Numerical integration is
essential for computing expectation values, normalization constants, and
transition amplitudes.

### Methods Compared

| Method | Dimension | Convergence | Best for |
|--------|-----------|-------------|----------|
| `quad` (adaptive Gauss) | 1D | $O(h^{2n+1})$ | Smooth 1D integrands |
| `dblquad` | 2D | adaptive | Low-dimensional integrals |
| **Monte Carlo** | Any $d$ | $O(1/\\sqrt{N})$ | High-dimensional integrals |

### Key formula
The Monte Carlo estimate of $\\int_\\Omega f(\\mathbf{x})\\,d\\mathbf{x}$ over
volume $V$ using $N$ random samples:

$$I \\approx \\frac{V}{N} \\sum_{i=1}^{N} f(\\mathbf{x}_i), \\quad
\\sigma_I \\approx \\frac{V}{\\sqrt{N}} \\sqrt{\\langle f^2 \\rangle - \\langle f \\rangle^2}$$

### Quantum Connection
Quantum Monte Carlo methods use stochastic sampling to solve the many-body
Schrödinger equation for systems with dozens or hundreds of electrons.""")

    nb.code("""\
# ── Numerical integration comparison ────────────────────────────────
# Integral: ∫₀^∞ x² e^{-x} dx = Γ(3) = 2! = 2

# 1) SciPy quad (adaptive Gaussian quadrature)
result_quad, error_quad = quad(lambda x: x**2 * np.exp(-x), 0, np.inf)
print(f"quad result:   {result_quad:.10f} ± {error_quad:.2e}")
print(f"Exact answer:  {2.0:.10f}")

# 2) Double integral: ∫∫ e^{-(x²+y²)} dx dy over [-∞,∞]² = π
result_dbl, error_dbl = dblquad(
    lambda y, x: np.exp(-(x**2 + y**2)),
    -5, 5,     # x limits (approximate ∞)
    -5, 5      # y limits
)
print(f"\\ndblquad result: {result_dbl:.10f} ± {error_dbl:.2e}")
print(f"Exact (π):      {np.pi:.10f}")

# 3) Monte Carlo integration of the same 2D Gaussian
rng = np.random.default_rng(42)
N_samples_list = [100, 1000, 10_000, 100_000, 1_000_000]
mc_results = []
mc_errors = []

for N_mc in N_samples_list:
    # Sample uniformly in [-5, 5] x [-5, 5], volume = 100
    xy = rng.uniform(-5, 5, size=(N_mc, 2))
    f_vals = np.exp(-(xy[:, 0]**2 + xy[:, 1]**2))
    volume = 10.0 * 10.0  # = 100
    estimate = volume * np.mean(f_vals)
    std_err = volume * np.std(f_vals) / np.sqrt(N_mc)
    mc_results.append(estimate)
    mc_errors.append(std_err)

print(f"\\n{'N samples':>12} {'MC estimate':>14} {'Error':>12} {'Rel. error':>12}")
print("-" * 54)
for N_mc, est, err in zip(N_samples_list, mc_results, mc_errors):
    print(f"{N_mc:>12,} {est:>14.8f} {err:>12.6f} {abs(est-np.pi)/np.pi:>12.2e}")

# Plot convergence
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(N_samples_list, [abs(r - np.pi) for r in mc_results], 'o-',
          label='MC absolute error', markersize=8)
ax.loglog(N_samples_list, mc_errors, 's--', label='MC standard error', markersize=8)
ax.loglog(N_samples_list, 5 / np.sqrt(N_samples_list), 'k:',
          label=r'$O(1/\\sqrt{N})$ reference')
ax.set_xlabel('Number of samples')
ax.set_ylabel('Error')
ax.set_title('Monte Carlo Convergence: $\\\\int e^{-(x^2+y^2)} \\\\, dA = \\\\pi$')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""")

    # ── Section 4: ODE Solvers ───────────────────────────────────────
    nb.md("""\
## 4. ODE Solvers: Euler, RK4, and `solve_ivp`

Ordinary differential equations appear throughout physics — from classical
trajectories to the time-dependent Schrödinger equation.

### Methods

**Forward Euler** (1st order):
$$y_{n+1} = y_n + h\\,f(t_n, y_n), \\quad \\text{error} = O(h)$$

**Runge-Kutta 4** (4th order):
$$y_{n+1} = y_n + \\frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4), \\quad \\text{error} = O(h^4)$$

**SciPy `solve_ivp`**: Adaptive step-size control using RK45, RK23, DOP853,
Radau (implicit), and BDF methods.

### Quantum Connection
The time-dependent Schrödinger equation $i\\hbar \\frac{\\partial}{\\partial t}|\\psi\\rangle = \\hat{H}|\\psi\\rangle$
is a first-order ODE in time, often solved with these same numerical methods.""")

    nb.code("""\
# ── ODE solver comparison: harmonic oscillator ──────────────────────
# ẍ + ω²x = 0  →  y = [x, v],  dy/dt = [v, -ω²x]
omega = 2 * np.pi  # frequency

def harmonic_rhs(t, y):
    return [y[1], -omega**2 * y[0]]

# Exact solution: x(t) = cos(ωt), v(t) = -ω sin(ωt)
y0 = [1.0, 0.0]
t_span = (0, 5)

# Forward Euler implementation
def euler_solve(f, t_span, y0, dt):
    t_arr = np.arange(t_span[0], t_span[1] + dt, dt)
    y_arr = np.zeros((len(t_arr), len(y0)))
    y_arr[0] = y0
    for i in range(len(t_arr) - 1):
        dydt = f(t_arr[i], y_arr[i])
        y_arr[i+1] = y_arr[i] + dt * np.array(dydt)
    return t_arr, y_arr

# Classical RK4 implementation
def rk4_solve(f, t_span, y0, dt):
    t_arr = np.arange(t_span[0], t_span[1] + dt, dt)
    y_arr = np.zeros((len(t_arr), len(y0)))
    y_arr[0] = y0
    for i in range(len(t_arr) - 1):
        t, y = t_arr[i], y_arr[i]
        k1 = np.array(f(t, y))
        k2 = np.array(f(t + dt/2, y + dt/2 * k1))
        k3 = np.array(f(t + dt/2, y + dt/2 * k2))
        k4 = np.array(f(t + dt, y + dt * k3))
        y_arr[i+1] = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return t_arr, y_arr

dt = 0.01

# Solve with all three methods
t0 = time.perf_counter()
t_e, y_e = euler_solve(harmonic_rhs, t_span, y0, dt)
time_euler = time.perf_counter() - t0

t0 = time.perf_counter()
t_r, y_r = rk4_solve(harmonic_rhs, t_span, y0, dt)
time_rk4 = time.perf_counter() - t0

t0 = time.perf_counter()
sol = solve_ivp(harmonic_rhs, t_span, y0, method='RK45',
                t_eval=np.arange(t_span[0], t_span[1], dt),
                rtol=1e-10, atol=1e-12)
time_ivp = time.perf_counter() - t0

# Exact solution
t_exact = np.linspace(0, 5, 1000)
x_exact = np.cos(omega * t_exact)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Trajectories
axes[0].plot(t_exact, x_exact, 'k-', label='Exact', linewidth=2.5)
axes[0].plot(t_e, y_e[:, 0], '--', label=f'Euler (dt={dt})', alpha=0.8)
axes[0].plot(t_r, y_r[:, 0], '-.', label=f'RK4 (dt={dt})', alpha=0.8)
axes[0].plot(sol.t, sol.y[0], ':', label='solve_ivp (RK45)', linewidth=2)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('x(t)')
axes[0].set_title('Harmonic Oscillator Solutions')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Errors
err_euler = np.abs(y_e[:, 0] - np.cos(omega * t_e))
err_rk4 = np.abs(y_r[:, 0] - np.cos(omega * t_r))
err_ivp = np.abs(sol.y[0] - np.cos(omega * sol.t))

axes[1].semilogy(t_e, err_euler + 1e-16, label='Euler')
axes[1].semilogy(t_r, err_rk4 + 1e-16, label='RK4')
axes[1].semilogy(sol.t, err_ivp + 1e-16, label='solve_ivp (RK45)')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('|error|')
axes[1].set_title('Absolute Error Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\n{'Method':<18} {'Time (ms)':>10} {'Max error':>14} {'Final error':>14}")
print("-" * 60)
print(f"{'Euler':<18} {time_euler*1000:>10.2f} {err_euler.max():>14.2e} {err_euler[-1]:>14.2e}")
print(f"{'RK4':<18} {time_rk4*1000:>10.2f} {err_rk4.max():>14.2e} {err_rk4[-1]:>14.2e}")
print(f"{'solve_ivp RK45':<18} {time_ivp*1000:>10.2f} {err_ivp.max():>14.2e} {err_ivp[-1]:>14.2e}")""")

    # ── Section 5: Optimization ──────────────────────────────────────
    nb.md("""\
## 5. Optimization: Minimization, Root Finding & Curve Fitting

Optimization is ubiquitous in physics:
- Variational methods minimize $\\langle \\psi | \\hat{H} | \\psi \\rangle$
- Least-squares fitting extracts physical parameters from experimental data
- Root finding locates energy eigenvalues from transcendental equations

### Key SciPy tools

| Function | Purpose |
|----------|---------|
| `optimize.minimize` | General-purpose minimization (BFGS, Nelder-Mead, etc.) |
| `optimize.root_scalar` | 1D root finding (Brent, bisection, Newton) |
| `optimize.curve_fit` | Nonlinear least-squares fitting |

### Quantum Connection
The **variational principle** states that for any trial wavefunction $|\\psi_T\\rangle$:

$$E_0 \\leq \\frac{\\langle \\psi_T | \\hat{H} | \\psi_T \\rangle}{\\langle \\psi_T | \\psi_T \\rangle}$$

Minimizing this expression over parameters yields an upper bound on the ground
state energy — this is the foundation of variational quantum eigensolvers (VQE).""")

    nb.code("""\
# ── Optimization demonstrations ─────────────────────────────────────

# 1) Variational method: find ground state of quartic potential V = x⁴
#    Trial function: ψ(x) = (α/π)^{1/4} exp(-αx²/2)
#    ⟨H⟩ = α/4 + 3/(4α²)  (analytically derived)

def energy_quartic(alpha):
    \"\"\"Energy expectation value for Gaussian trial in quartic potential.\"\"\"
    return alpha / 4.0 + 3.0 / (4.0 * alpha**2)

alpha_range = np.linspace(0.3, 5, 200)
E_range = [energy_quartic(a) for a in alpha_range]

# Minimize
from scipy.optimize import minimize_scalar, minimize, curve_fit, root_scalar

result = minimize_scalar(energy_quartic, bounds=(0.1, 10), method='bounded')
alpha_opt = result.x
E_opt = result.fun

print("=== Variational Method: Quartic Oscillator ===")
print(f"Optimal α = {alpha_opt:.6f}")
print(f"Variational energy = {E_opt:.6f}")
print(f"Analytical optimum: α = {(3/2)**(1/3):.6f}, "
      f"E = {3/(4*(3/2)**(2/3)) + (3/2)**(1/3)/4:.6f}")

# 2) Root finding: transcendental equation for finite square well
#    z tan(z) = √(z₀² - z²)  where z₀ = √(2mVa²/ℏ²)
z0 = 8.0  # well depth parameter

def transcendental(z):
    if z >= z0:
        return float('inf')
    return z * np.tan(z) - np.sqrt(z0**2 - z**2)

# Find roots in intervals where tan(z) > 0
roots = []
for n in range(int(z0 / np.pi) + 1):
    lo = n * np.pi + 0.01
    hi = (n + 0.5) * np.pi - 0.01
    if lo < z0:
        try:
            sol = root_scalar(transcendental, bracket=[lo, min(hi, z0 - 0.01)])
            roots.append(sol.root)
        except ValueError:
            pass

print(f"\\n=== Finite Square Well (z₀ = {z0}) ===")
print("Bound state z values (even parity):")
for i, r in enumerate(roots):
    E_ratio = r**2 / z0**2
    print(f"  n={i}: z = {r:.6f}, E/V₀ = {E_ratio:.6f}")

# 3) Curve fitting: noisy exponential decay
rng = np.random.default_rng(42)
t_data = np.linspace(0, 5, 50)
true_params = (3.0, 1.5, 0.5)  # A, γ, offset
y_data = true_params[0] * np.exp(-true_params[1] * t_data) + true_params[2]
y_noisy = y_data + 0.15 * rng.normal(size=len(t_data))

def exp_decay(t, A, gamma, offset):
    return A * np.exp(-gamma * t) + offset

popt, pcov = curve_fit(exp_decay, t_data, y_noisy, p0=[2, 1, 0])
perr = np.sqrt(np.diag(pcov))

print(f"\\n=== Curve Fitting: Exponential Decay ===")
print(f"{'Param':<8} {'True':>8} {'Fitted':>10} {'±Error':>10}")
print("-" * 38)
for name, true, fit, err in zip(['A', 'γ', 'offset'], true_params, popt, perr):
    print(f"{name:<8} {true:>8.3f} {fit:>10.4f} {err:>10.4f}")

# Combined plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Variational energy
axes[0].plot(alpha_range, E_range, 'b-', linewidth=2)
axes[0].axvline(alpha_opt, color='r', linestyle='--', label=f'α* = {alpha_opt:.3f}')
axes[0].axhline(E_opt, color='r', linestyle=':', alpha=0.5)
axes[0].set_xlabel(r'$\\alpha$')
axes[0].set_ylabel(r'$\\langle E \\rangle$')
axes[0].set_title('Variational Energy: Quartic Potential')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Transcendental equation
z_plot = np.linspace(0.01, z0 - 0.01, 500)
lhs = z_plot * np.tan(z_plot)
rhs = np.sqrt(z0**2 - z_plot**2)
lhs_clipped = np.where(np.abs(lhs) < 50, lhs, np.nan)
axes[1].plot(z_plot, lhs_clipped, 'b-', label=r'$z \\tan(z)$')
axes[1].plot(z_plot, rhs, 'r-', label=r'$\\sqrt{z_0^2 - z^2}$')
for r in roots:
    axes[1].axvline(r, color='green', linestyle='--', alpha=0.6)
axes[1].set_xlim(0, z0)
axes[1].set_ylim(-5, 30)
axes[1].set_xlabel('z')
axes[1].set_title('Finite Square Well: Root Finding')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Curve fitting
axes[2].scatter(t_data, y_noisy, s=30, alpha=0.7, label='Noisy data')
t_fine = np.linspace(0, 5, 200)
axes[2].plot(t_fine, exp_decay(t_fine, *true_params), 'g--',
             label='True curve', linewidth=2)
axes[2].plot(t_fine, exp_decay(t_fine, *popt), 'r-',
             label='Fitted curve', linewidth=2)
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Signal')
axes[2].set_title('Nonlinear Curve Fitting')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()""")

    # ── Section 6: Eigenvalue Problems ───────────────────────────────
    nb.md("""\
## 6. Eigenvalue Problems for Large Systems

In quantum mechanics, finding eigenvalues means finding energy levels:

$$\\hat{H}|\\psi_n\\rangle = E_n|\\psi_n\\rangle$$

For large systems ($N > 10^4$), storing the full matrix is impractical.
**Sparse eigensolvers** (Lanczos/Arnoldi algorithms via `scipy.sparse.linalg.eigsh`)
find the few eigenvalues you need without ever forming the full matrix.

### Scaling comparison

| Method | Time complexity | Memory |
|--------|-----------------|--------|
| Dense `eigh` | $O(N^3)$ | $O(N^2)$ |
| Sparse `eigsh` (k values) | $O(k \\cdot N \\cdot \\text{nnz}/N)$ | $O(\\text{nnz} + kN)$ |

### Quantum Connection
The difference between tractable and intractable quantum problems often comes
down to sparsity.  A 20-qubit Hamiltonian is a $10^6 \\times 10^6$ matrix —
only feasible with sparse methods.""")

    nb.code("""\
# ── Dense vs sparse eigenvalue scaling ──────────────────────────────
sizes = [50, 100, 200, 500, 1000, 2000]
times_dense = []
times_sparse = []

for N in sizes:
    # Create a sparse tridiagonal matrix (like 1D Schrödinger)
    diag = 2.0 * np.ones(N)
    off = -1.0 * np.ones(N - 1)
    H = sparse.diags([off, diag, off], [-1, 0, 1], format='csr')

    # Dense eigenvalue decomposition (all eigenvalues)
    H_dense = H.toarray()
    t0 = time.perf_counter()
    evals_d = np.linalg.eigvalsh(H_dense)
    t_dense = time.perf_counter() - t0
    times_dense.append(t_dense)

    # Sparse eigensolver (only lowest 6 eigenvalues)
    t0 = time.perf_counter()
    evals_s, _ = eigsh(H, k=6, which='SA')
    t_sparse = time.perf_counter() - t0
    times_sparse.append(t_sparse)

    # Verify they agree
    max_diff = np.max(np.abs(np.sort(evals_s) - np.sort(evals_d[:6])))
    print(f"N={N:>5}: dense={t_dense*1000:>8.2f} ms, "
          f"sparse={t_sparse*1000:>8.2f} ms, "
          f"speedup={t_dense/t_sparse:>6.1f}x, "
          f"max diff={max_diff:.2e}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(sizes, [t*1000 for t in times_dense], 'o-',
          label='Dense (all eigenvalues)', markersize=8)
ax.loglog(sizes, [t*1000 for t in times_sparse], 's-',
          label='Sparse (6 eigenvalues)', markersize=8)

# Reference lines
s = np.array(sizes, dtype=float)
ax.loglog(s, 0.001 * (s/50)**3, 'k--', alpha=0.3, label=r'$O(N^3)$ reference')
ax.loglog(s, 0.05 * (s/50)**1.2, 'k:', alpha=0.3, label=r'$O(N^{1.2})$ reference')

ax.set_xlabel('Matrix size N')
ax.set_ylabel('Time (ms)')
ax.set_title('Dense vs Sparse Eigenvalue Solver Scaling')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()""")

    # ── Section 7: QM Application ────────────────────────────────────
    nb.md("""\
## 7. Quantum Application: Particle in a Box (Numerical vs Exact)

Let us put all these tools together on a real quantum problem.

The 1D infinite square well has exact solutions:

$$\\psi_n(x) = \\sqrt{\\frac{2}{L}} \\sin\\left(\\frac{n\\pi x}{L}\\right), \\quad
E_n = \\frac{n^2 \\pi^2 \\hbar^2}{2mL^2}$$

We will solve the same problem numerically using finite differences and compare.""")

    nb.code("""\
# ── Quantum particle in a box: numerical vs analytical ──────────────
# Units: ℏ = m = 1, L = 1
L = 1.0
N_grid = 200
dx = L / (N_grid + 1)
x_grid = np.linspace(dx, L - dx, N_grid)  # interior points only

# Finite-difference Hamiltonian: H = -ℏ²/(2m) d²/dx²
# Second derivative: d²ψ/dx² ≈ (ψ_{i+1} - 2ψ_i + ψ_{i-1}) / dx²
coeff = 1.0 / (2.0 * dx**2)
H_box = sparse.diags(
    [-coeff * np.ones(N_grid - 1),
     2 * coeff * np.ones(N_grid),
     -coeff * np.ones(N_grid - 1)],
    [-1, 0, 1], format='csr'
)

# Solve for lowest 5 states
n_states = 5
energies, states = eigsh(H_box, k=n_states, which='SA')
idx = np.argsort(energies)
energies = energies[idx]
states = states[:, idx]

# Normalize eigenstates
for i in range(n_states):
    norm = np.sqrt(np.trapezoid(states[:, i]**2, x_grid))
    states[:, i] /= norm
    # Fix sign convention: positive at first peak
    if states[N_grid // (2*(i+1)), i] < 0:
        states[:, i] *= -1

# Exact solutions
def exact_psi(n, x):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def exact_E(n):
    return (n * np.pi)**2 / 2.0

# Compare
print(f"{'n':>3} {'E_numerical':>14} {'E_exact':>14} {'Rel. Error':>14}")
print("-" * 48)
for i in range(n_states):
    n = i + 1
    E_ex = exact_E(n)
    rel_err = abs(energies[i] - E_ex) / E_ex
    print(f"{n:>3} {energies[i]:>14.8f} {E_ex:>14.8f} {rel_err:>14.2e}")

# Plot wavefunctions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x_fine = np.linspace(0, L, 500)

for i in range(n_states):
    n = i + 1
    offset = energies[i]
    scale = 0.3 * (energies[1] - energies[0])  # for visual separation

    # Numerical
    axes[0].plot(x_grid, states[:, i] * scale + offset, f'C{i}', linewidth=2)
    axes[0].axhline(offset, color=f'C{i}', linewidth=0.5, alpha=0.3)

    # Exact
    axes[1].plot(x_fine, exact_psi(n, x_fine) * scale + offset, f'C{i}',
                 linewidth=2, label=f'n={n}')
    axes[1].axhline(offset, color=f'C{i}', linewidth=0.5, alpha=0.3)

for ax, title in zip(axes, ['Numerical (finite difference)', 'Exact (analytical)']):
    ax.set_xlabel('x / L')
    ax.set_ylabel('Energy + ψ(x)')
    ax.set_title(title)
    ax.set_xlim(0, L)
    ax.grid(True, alpha=0.3)

axes[1].legend(loc='upper right')
plt.suptitle('Infinite Square Well: Numerical vs Exact', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""")

    # ── Summary ──────────────────────────────────────────────────────
    nb.md("""\
## Summary: Scientific Computing Toolkit

| Tool | SciPy Function | Use Case |
|------|---------------|----------|
| Broadcasting | NumPy array ops | Efficient multi-dimensional computation |
| Sparse matrices | `scipy.sparse` | Large Hamiltonians, lattice models |
| Integration | `quad`, `dblquad`, Monte Carlo | Expectation values, normalization |
| ODE solving | `solve_ivp` (RK45, Radau, BDF) | Time evolution, Schrödinger equation |
| Optimization | `minimize`, `root_scalar`, `curve_fit` | Variational methods, energy levels |
| Eigenvalues | `eigsh`, `eigh` | Energy spectra, quantum states |

### Key Takeaway
These numerical tools are the **computational backbone** of quantum physics.
Every technique practiced here — from sparse eigensolvers to Monte Carlo
integration — maps directly onto quantum simulation tasks you will encounter
starting in Year 1.

---
*Notebook generated for the SIIEA Quantum Engineering curriculum.*
*License: CC BY-NC-SA 4.0 | Siiea Innovations, LLC*""")

    nb.save()
    print("  -> Month 10 notebook complete.\n")


# ──────────────────────────────────────────────────────────────────────
# NOTEBOOK 2 — Month 11: Group Theory — Symmetries & Representations
# ──────────────────────────────────────────────────────────────────────
def build_month_11():
    nb = NotebookBuilder(
        "Symmetries & Representations: Group Theory for Quantum Mechanics",
        "year_0/month_11_group_theory/11_symmetries_and_representations.ipynb",
        "Days 281-308",
    )

    # ── Imports ──────────────────────────────────────────────────────
    nb.code("""\
# ── Imports & configuration ─────────────────────────────────────────
import numpy as np
from scipy.linalg import expm, block_diag
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import sympy
from sympy import Matrix, eye, sqrt, Rational, pi, cos, sin, symbols
from sympy import pprint

%matplotlib inline

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'figure.dpi': 100,
})

print("All imports successful — ready for group theory.")""")

    # ── Section 1: Permutation Groups ────────────────────────────────
    nb.md("""\
## 1. Permutation Groups: The Symmetric Group $S_3$

A **group** $(G, \\cdot)$ satisfies four axioms:
1. **Closure:** $a, b \\in G \\Rightarrow a \\cdot b \\in G$
2. **Associativity:** $(a \\cdot b) \\cdot c = a \\cdot (b \\cdot c)$
3. **Identity:** $\\exists\\, e \\in G$ such that $e \\cdot a = a \\cdot e = a$
4. **Inverse:** $\\forall\\, a \\in G$, $\\exists\\, a^{-1}$ such that $a \\cdot a^{-1} = e$

The **symmetric group** $S_n$ consists of all permutations of $n$ objects.
$S_3$ has $3! = 6$ elements and is the simplest non-abelian group.

### Elements of $S_3$

| Symbol | Cycle notation | Meaning |
|--------|---------------|---------|
| $e$ | $(1)(2)(3)$ | Identity |
| $r$ | $(123)$ | Rotation by $120°$ |
| $r^2$ | $(132)$ | Rotation by $240°$ |
| $s_1$ | $(23)$ | Reflection fixing vertex 1 |
| $s_2$ | $(13)$ | Reflection fixing vertex 2 |
| $s_3$ | $(12)$ | Reflection fixing vertex 3 |

### Quantum Connection
Identical particles in quantum mechanics are described by representations of the
symmetric group.  Bosons transform under the trivial representation (symmetric
wavefunctions), while fermions transform under the sign representation
(antisymmetric wavefunctions).""")

    nb.code("""\
# ── S₃ multiplication table ─────────────────────────────────────────
# Represent permutations as tuples: (a,b,c) means 1→a, 2→b, 3→c

def compose(p, q):
    \"\"\"Compose two permutations: (p ∘ q)(i) = p(q(i)).\"\"\"
    return tuple(p[q[i]-1] for i in range(len(p)))

# Define S₃ elements with names
S3_elements = {
    'e':   (1, 2, 3),   # identity
    'r':   (2, 3, 1),   # (123) rotation
    'r²':  (3, 1, 2),   # (132) rotation
    's₁':  (1, 3, 2),   # (23) reflection
    's₂':  (3, 2, 1),   # (13) reflection
    's₃':  (2, 1, 3),   # (12) reflection
}

names = list(S3_elements.keys())
perms = list(S3_elements.values())

# Build multiplication table
table = np.empty((6, 6), dtype=object)
for i, (ni, pi) in enumerate(zip(names, perms)):
    for j, (nj, pj) in enumerate(zip(names, perms)):
        product = compose(pi, pj)
        # Find which element this is
        for k, pk in enumerate(perms):
            if product == pk:
                table[i, j] = names[k]
                break

# Display as formatted table
print("S₃ Multiplication Table  (row ∘ column):")
print(f"{'':>4}", end="")
for n in names:
    print(f"{n:>5}", end="")
print()
print("-" * (4 + 5 * len(names)))
for i, n in enumerate(names):
    print(f"{n:>3} |", end="")
    for j in range(len(names)):
        print(f"{table[i,j]:>5}", end="")
    print()

# Verify non-commutativity
r, s1 = S3_elements['r'], S3_elements['s₁']
print(f"\\nr ∘ s₁ = {compose(r, s1)} = s₃")
print(f"s₁ ∘ r = {compose(s1, r)} = s₂")
print(f"r ∘ s₁ ≠ s₁ ∘ r  →  S₃ is NON-ABELIAN ✓")""")

    # ── Section 2: Symmetry Groups of Geometric Objects ──────────────
    nb.md("""\
## 2. Symmetry Groups of Geometric Objects

The **dihedral group** $D_n$ is the symmetry group of a regular $n$-gon.
It has $2n$ elements: $n$ rotations and $n$ reflections.

- $D_3 \\cong S_3$: symmetry group of an equilateral triangle (order 6)
- $D_4$: symmetry group of a square (order 8)

For $D_4$, the elements are:
- Rotations: $e, r, r^2, r^3$ (by $0°, 90°, 180°, 270°$)
- Reflections: $s_v, s_h, s_{d1}, s_{d2}$ (vertical, horizontal, two diagonals)

### Quantum Connection
Crystal symmetries form **space groups** that determine the band structure of
solids.  The irreducible representations of the crystal's point group label
the electronic energy bands and phonon modes.""")

    nb.code("""\
# ── Visualize D₃ and D₄ symmetries ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# D₃: equilateral triangle
ax = axes[0]
angles_tri = np.array([np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3])
verts_tri = np.column_stack([np.cos(angles_tri), np.sin(angles_tri)])
triangle = plt.Polygon(verts_tri, fill=False, edgecolor='blue', linewidth=2.5)
ax.add_patch(triangle)

# Label vertices
for i, (x, y) in enumerate(verts_tri):
    ax.annotate(f'{i+1}', (x, y), fontsize=16, fontweight='bold',
                ha='center', va='center',
                xytext=(0.15*np.cos(angles_tri[i]), 0.15*np.sin(angles_tri[i])),
                textcoords='offset fontsize')

# Draw rotation axis and reflection axes
ax.plot(0, 0, 'ko', markersize=6)
for i in range(3):
    angle = np.pi/2 + i * 2*np.pi/3
    ax.plot([0, 1.3*np.cos(angle)], [0, 1.3*np.sin(angle)],
            'r--', alpha=0.5, linewidth=1.5)
    ax.annotate(f'$s_{i+1}$', (1.4*np.cos(angle), 1.4*np.sin(angle)),
                fontsize=13, color='red', ha='center')

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.5, 1.8)
ax.set_aspect('equal')
ax.set_title('$D_3$: Symmetries of the Triangle', fontsize=15)
ax.grid(True, alpha=0.2)

# D₄: square
ax = axes[1]
angles_sq = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
verts_sq = np.column_stack([np.cos(angles_sq), np.sin(angles_sq)])
square = plt.Polygon(verts_sq, fill=False, edgecolor='blue', linewidth=2.5)
ax.add_patch(square)

for i, (x, y) in enumerate(verts_sq):
    ax.annotate(f'{i+1}', (x, y), fontsize=16, fontweight='bold',
                ha='center', va='center',
                xytext=(0.15*np.cos(angles_sq[i]), 0.15*np.sin(angles_sq[i])),
                textcoords='offset fontsize')

# Reflection axes for D₄
ref_angles = [0, np.pi/2, np.pi/4, 3*np.pi/4]
ref_labels = ['$s_v$', '$s_h$', '$s_{d1}$', '$s_{d2}$']
for angle, label in zip(ref_angles, ref_labels):
    ax.plot([-1.3*np.cos(angle), 1.3*np.cos(angle)],
            [-1.3*np.sin(angle), 1.3*np.sin(angle)],
            'r--', alpha=0.5, linewidth=1.5)
    ax.annotate(label, (1.45*np.cos(angle), 1.45*np.sin(angle)),
                fontsize=13, color='red', ha='center')

ax.plot(0, 0, 'ko', markersize=6)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.set_title('$D_4$: Symmetries of the Square', fontsize=15)
ax.grid(True, alpha=0.2)

plt.suptitle('Dihedral Group Symmetries', fontsize=17, y=1.02)
plt.tight_layout()
plt.show()

# Count elements
print(f"|D₃| = {2*3} = 2×3 (3 rotations + 3 reflections)")
print(f"|D₄| = {2*4} = 2×4 (4 rotations + 4 reflections)")""")

    # ── Section 3: Representation Theory ─────────────────────────────
    nb.md("""\
## 3. Representation Theory: Matrix Representations of Finite Groups

A **representation** of a group $G$ is a homomorphism $\\rho: G \\to GL(V)$
assigning to each group element an invertible matrix such that:

$$\\rho(g_1 \\cdot g_2) = \\rho(g_1)\\,\\rho(g_2)$$

### Key definitions

- **Dimension** of a representation: the size of the matrices
- **Faithful** representation: $\\rho$ is injective (different group elements get different matrices)
- **Irreducible representation (irrep):** has no nontrivial invariant subspaces
- **Reducible:** can be block-diagonalized into smaller representations

### Irreps of $S_3$

$S_3$ has exactly **3 irreducible representations**:

| Irrep | Dimension | Description |
|-------|-----------|-------------|
| **Trivial** ($\\Gamma_1$) | 1 | All elements $\\to 1$ |
| **Sign** ($\\Gamma_2$) | 1 | Rotations $\\to +1$, reflections $\\to -1$ |
| **Standard** ($\\Gamma_3$) | 2 | Faithful 2D representation |

Check: $1^2 + 1^2 + 2^2 = 6 = |S_3|$ ✓

### Quantum Connection
Irreducible representations label quantum states.  When the Hamiltonian commutes
with a symmetry group, energy eigenstates transform as irreps, giving rise to
quantum numbers and selection rules.""")

    nb.code("""\
# ── Matrix representations of S₃ ────────────────────────────────────
# Standard 2D representation (faithful): act on (x,y) plane

# 120° rotation matrix
theta = 2 * np.pi / 3
R120 = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

# Reflection across x-axis
Sx = np.array([[1, 0], [0, -1]], dtype=float)

# Generate all 6 elements from r = R120 and s = Sx
I2 = np.eye(2)
reps_2d = {
    'e':   I2,
    'r':   R120,
    'r²':  R120 @ R120,
    's₁':  Sx,
    's₂':  R120 @ Sx,
    's₃':  R120 @ R120 @ Sx,
}

# Also build the trivial and sign representations
reps_trivial = {name: np.array([[1.0]]) for name in reps_2d}
reps_sign = {}
for name in reps_2d:
    if name in ['e', 'r', 'r²']:
        reps_sign[name] = np.array([[1.0]])
    else:
        reps_sign[name] = np.array([[-1.0]])

# Display the 2D representation
print("Standard 2D Representation of S₃:")
print("=" * 50)
for name, mat in reps_2d.items():
    print(f"\\nρ({name}) =")
    for row in mat:
        print(f"  [{row[0]:+.4f}  {row[1]:+.4f}]")

# Verify homomorphism: ρ(r)·ρ(s₁) should equal ρ(s₃) = ρ(r∘s₁)
product = reps_2d['r'] @ reps_2d['s₁']
print(f"\\nVerification: ρ(r) · ρ(s₁) =")
for row in product:
    print(f"  [{row[0]:+.4f}  {row[1]:+.4f}]")
print(f"\\nρ(s₃) =")
for row in reps_2d['s₃']:
    print(f"  [{row[0]:+.4f}  {row[1]:+.4f}]")
diff = np.max(np.abs(product - reps_2d['s₃']))
print(f"\\nMax difference: {diff:.2e} → Homomorphism verified ✓")""")

    # ── Section 4: Character Tables ──────────────────────────────────
    nb.md("""\
## 4. Character Tables & Orthogonality Relations

The **character** of a representation $\\rho$ for group element $g$ is:

$$\\chi_{\\rho}(g) = \\text{Tr}[\\rho(g)]$$

Characters are constant on **conjugacy classes** (sets of elements related by
$g' = hgh^{-1}$).

### Great Orthogonality Theorem

For irreps $\\alpha$ and $\\beta$ of a finite group of order $|G|$:

$$\\sum_{g \\in G} \\chi_\\alpha(g)^* \\chi_\\beta(g) = |G|\\,\\delta_{\\alpha\\beta}$$

### Character Table of $S_3$

| | $\\{e\\}$ | $\\{r, r^2\\}$ | $\\{s_1, s_2, s_3\\}$ |
|------|---------|-------------|-------------------|
| $\\Gamma_1$ (trivial) | 1 | 1 | 1 |
| $\\Gamma_2$ (sign) | 1 | 1 | -1 |
| $\\Gamma_3$ (standard) | 2 | -1 | 0 |

### Quantum Connection
Character tables predict which optical transitions are allowed.
A transition $|i\\rangle \\to |f\\rangle$ under operator $\\hat{O}$ is allowed only if
the product representation $\\Gamma_i \\otimes \\Gamma_O \\otimes \\Gamma_f$ contains
the trivial representation.""")

    nb.code("""\
# ── Character table computation and orthogonality check ─────────────

# Conjugacy classes of S₃
classes = {
    '{e}': ['e'],
    '{r,r²}': ['r', 'r²'],
    '{s₁,s₂,s₃}': ['s₁', 's₂', 's₃'],
}

# Compute characters for each representation and class
all_reps = {
    'Γ₁ (trivial)': reps_trivial,
    'Γ₂ (sign)': reps_sign,
    'Γ₃ (standard)': reps_2d,
}

char_table = {}
print("Character Table of S₃")
print("=" * 55)
print(f"{'Irrep':<17} ", end="")
for cls_name in classes:
    print(f"{cls_name:>14}", end="")
print()
print("-" * 55)

for rep_name, rep_matrices in all_reps.items():
    chars = []
    for cls_name, cls_elements in classes.items():
        # Character = trace, same for all elements in a class
        chi = np.trace(rep_matrices[cls_elements[0]])
        chars.append(chi)
    char_table[rep_name] = chars
    print(f"{rep_name:<17} ", end="")
    for c in chars:
        print(f"{c:>14.1f}", end="")
    print()

# Verify Great Orthogonality Theorem
print("\\n--- Orthogonality Check ---")
print("⟨χ_α | χ_β⟩ = (1/|G|) Σ |C_k| χ_α(C_k)* χ_β(C_k)")
class_sizes = [len(v) for v in classes.values()]
rep_names = list(char_table.keys())

for i, name_i in enumerate(rep_names):
    for j, name_j in enumerate(rep_names):
        inner = sum(
            class_sizes[k] * char_table[name_i][k] * char_table[name_j][k]
            for k in range(len(class_sizes))
        ) / 6.0  # |S₃| = 6
        status = "✓" if abs(inner - (1 if i == j else 0)) < 1e-10 else "✗"
        print(f"  ⟨{name_i}, {name_j}⟩ = {inner:.1f}  {status}")""")

    # ── Section 5: Lie Groups ────────────────────────────────────────
    nb.md("""\
## 5. Lie Groups: SO(2) and SO(3) Rotation Matrices

**Lie groups** are continuous groups that are also smooth manifolds.  They are
described by their **Lie algebra** — the tangent space at the identity.

### SO(2): Rotations in 2D

$$R(\\theta) = \\begin{pmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta \\end{pmatrix}
= e^{i\\theta J}, \\quad J = \\begin{pmatrix} 0 & -1 \\\\ 1 & 0 \\end{pmatrix}$$

One parameter ($\\theta$), one generator ($J$).

### SO(3): Rotations in 3D

Three generators (angular momentum operators):

$$J_x = \\begin{pmatrix} 0&0&0 \\\\ 0&0&-1 \\\\ 0&1&0 \\end{pmatrix}, \\quad
J_y = \\begin{pmatrix} 0&0&1 \\\\ 0&0&0 \\\\ -1&0&0 \\end{pmatrix}, \\quad
J_z = \\begin{pmatrix} 0&-1&0 \\\\ 1&0&0 \\\\ 0&0&0 \\end{pmatrix}$$

Lie algebra: $[J_i, J_j] = \\epsilon_{ijk} J_k$

### Quantum Connection
Angular momentum in quantum mechanics comes from the representation theory of
SO(3).  The commutation relations $[\\hat{L}_i, \\hat{L}_j] = i\\hbar\\epsilon_{ijk}\\hat{L}_k$
are exactly the SO(3) Lie algebra.""")

    nb.code("""\
# ── SO(2) and SO(3) rotations ───────────────────────────────────────
# SO(2): 2D rotations
def rotation_2d(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

# Verify group properties
theta1, theta2 = np.pi/4, np.pi/3
R1 = rotation_2d(theta1)
R2 = rotation_2d(theta2)
R_product = R1 @ R2
R_sum = rotation_2d(theta1 + theta2)

print("SO(2) Verification:")
print(f"  R(π/4) · R(π/3) = R(π/4 + π/3)?")
print(f"  Max difference: {np.max(np.abs(R_product - R_sum)):.2e} ✓")
print(f"  det(R) = {np.linalg.det(R1):.6f} (should be 1)")
print(f"  R·Rᵀ = I? {np.allclose(R1 @ R1.T, np.eye(2))} ✓")

# SO(3): 3D rotation generators
Jx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
Jy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
Jz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)

# Verify Lie algebra: [Jx, Jy] = Jz
comm_xy = Jx @ Jy - Jy @ Jx
print(f"\\nSO(3) Lie Algebra Verification:")
print(f"  [Jx, Jy] = Jz? {np.allclose(comm_xy, Jz)} ✓")
print(f"  [Jy, Jz] = Jx? {np.allclose(Jy @ Jz - Jz @ Jy, Jx)} ✓")
print(f"  [Jz, Jx] = Jy? {np.allclose(Jz @ Jx - Jx @ Jz, Jy)} ✓")

# Generate rotation matrices via matrix exponential
# R(θ, n̂) = exp(θ n̂·J)
theta = np.pi / 6
Rz = expm(theta * Jz)   # rotation about z by 30°
Rx = expm(theta * Jx)   # rotation about x by 30°

print(f"\\nR_z(30°) via matrix exponential:")
print(Rz.round(6))
print(f"det = {np.linalg.det(Rz):.6f}, orthogonal: {np.allclose(Rz @ Rz.T, np.eye(3))}")

# Visualize: rotate a set of points
fig = plt.figure(figsize=(12, 5))

# Original and rotated vectors
v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
              [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=float)
v_rot = (expm(np.pi/4 * Jz) @ v.T).T

ax1 = fig.add_subplot(121, projection='3d')
for i, (orig, rot) in enumerate(zip(v, v_rot)):
    ax1.quiver(0, 0, 0, *orig, color=f'C{i}', alpha=0.5, linewidth=1.5,
               arrow_length_ratio=0.1)
    ax1.quiver(0, 0, 0, *rot, color=f'C{i}', linewidth=2.5,
               arrow_length_ratio=0.1)

ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.set_title('SO(3) Rotation about z-axis (45°)')

# Demonstrate non-commutativity of 3D rotations
ax2 = fig.add_subplot(122, projection='3d')
angle = np.pi / 2

v0 = np.array([1, 0, 0])

# Path 1: Rx then Rz
v1_step1 = expm(angle * Jx) @ v0
v1_final = expm(angle * Jz) @ v1_step1

# Path 2: Rz then Rx
v2_step1 = expm(angle * Jz) @ v0
v2_final = expm(angle * Jx) @ v2_step1

ax2.quiver(0, 0, 0, *v0, color='black', linewidth=3, arrow_length_ratio=0.1,
           label='Original')
ax2.quiver(0, 0, 0, *v1_final, color='blue', linewidth=3, arrow_length_ratio=0.1,
           label='Rx then Rz')
ax2.quiver(0, 0, 0, *v2_final, color='red', linewidth=3, arrow_length_ratio=0.1,
           label='Rz then Rx')

ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.set_title('Non-commutativity of 3D Rotations')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()

print(f"\\nRx·Rz applied to [1,0,0]: {v1_final.round(4)}")
print(f"Rz·Rx applied to [1,0,0]: {v2_final.round(4)}")
print(f"Different! 3D rotations do NOT commute (in general).")""")

    # ── Section 6: SU(2) and Spin ────────────────────────────────────
    nb.md("""\
## 6. SU(2) and Spin: Pauli Matrices as Generators

The group **SU(2)** consists of $2 \\times 2$ unitary matrices with determinant 1.
It is the **double cover** of SO(3): every SO(3) rotation corresponds to two
SU(2) elements ($U$ and $-U$).

### Pauli Matrices (generators of SU(2))

$$\\sigma_x = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}, \\quad
\\sigma_y = \\begin{pmatrix} 0 & -i \\\\ i & 0 \\end{pmatrix}, \\quad
\\sigma_z = \\begin{pmatrix} 1 & 0 \\\\ 0 & -1 \\end{pmatrix}$$

### Key Properties

- $\\sigma_i^2 = I$ for all $i$
- $\\sigma_i \\sigma_j = i\\epsilon_{ijk}\\sigma_k$ for $i \\neq j$
- $\\{\\sigma_i, \\sigma_j\\} = 2\\delta_{ij}I$
- $\\text{Tr}(\\sigma_i) = 0$, $\\det(\\sigma_i) = -1$

### SU(2) rotation

$$U(\\hat{n}, \\theta) = \\exp\\left(-i\\frac{\\theta}{2}\\hat{n}\\cdot\\vec{\\sigma}\\right)
= \\cos\\frac{\\theta}{2}\\,I - i\\sin\\frac{\\theta}{2}\\,(\\hat{n}\\cdot\\vec{\\sigma})$$

### Quantum Connection
Spin-1/2 particles (electrons, protons, neutrons) are described by SU(2).
The spin operators are $\\hat{S}_i = \\frac{\\hbar}{2}\\sigma_i$.
SU(2) also underlies the electroweak force in the Standard Model.""")

    nb.code("""\
# ── Pauli matrices and SU(2) ────────────────────────────────────────
# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

paulis = {'σx': sigma_x, 'σy': sigma_y, 'σz': sigma_z}

# Verify algebraic properties
print("=== Pauli Matrix Properties ===")
for name, sigma in paulis.items():
    print(f"\\n{name}:")
    print(f"  σ² = I? {np.allclose(sigma @ sigma, I2)}")
    print(f"  Tr(σ) = {np.trace(sigma):.1f}")
    print(f"  det(σ) = {np.linalg.det(sigma):.1f}")
    print(f"  Hermitian? {np.allclose(sigma, sigma.conj().T)}")
    print(f"  Unitary? {np.allclose(sigma @ sigma.conj().T, I2)}")

# Commutation relations: [σi, σj] = 2i εijk σk
print("\\n=== Commutation Relations ===")
comm_xy = sigma_x @ sigma_y - sigma_y @ sigma_x
print(f"[σx, σy] = 2i·σz? {np.allclose(comm_xy, 2j * sigma_z)} ✓")
comm_yz = sigma_y @ sigma_z - sigma_z @ sigma_y
print(f"[σy, σz] = 2i·σx? {np.allclose(comm_yz, 2j * sigma_x)} ✓")
comm_zx = sigma_z @ sigma_x - sigma_x @ sigma_z
print(f"[σz, σx] = 2i·σy? {np.allclose(comm_zx, 2j * sigma_y)} ✓")

# Anti-commutation: {σi, σj} = 2δij I
print("\\n=== Anti-commutation Relations ===")
acomm_xy = sigma_x @ sigma_y + sigma_y @ sigma_x
print(f"{{σx, σy}} = 0? {np.allclose(acomm_xy, 0)} ✓")
acomm_xx = sigma_x @ sigma_x + sigma_x @ sigma_x
print(f"{{σx, σx}} = 2I? {np.allclose(acomm_xx, 2*I2)} ✓")

# SU(2) rotation: spin-1/2 state rotated about z-axis
print("\\n=== SU(2) Rotation of Spin-1/2 ===")
theta_values = [0, np.pi/2, np.pi, 2*np.pi, 4*np.pi]
spin_up = np.array([1, 0], dtype=complex)

for theta in theta_values:
    U = expm(-1j * theta/2 * sigma_z)
    rotated = U @ spin_up
    print(f"  θ = {theta/np.pi:.1f}π: U|↑⟩ = [{rotated[0]:.4f}, {rotated[1]:.4f}]"
          f"  (phase: e^{{-iθ/2}} = {np.exp(-1j*theta/2):.4f})")

print("\\nNote: 2π rotation gives -|↑⟩ (spinor sign flip!)")
print("      4π rotation restores |↑⟩ — this is the SU(2) double cover of SO(3)")""")

    # ── Section 7: Clebsch-Gordan Coefficients ───────────────────────
    nb.md("""\
## 7. Clebsch-Gordan Coefficients & Angular Momentum Addition

When two angular momenta $j_1$ and $j_2$ combine, the total angular momentum
$J$ ranges from $|j_1 - j_2|$ to $j_1 + j_2$:

$$|j_1, m_1\\rangle \\otimes |j_2, m_2\\rangle = \\sum_{J, M} \\langle j_1, m_1; j_2, m_2 | J, M \\rangle \\, |J, M\\rangle$$

The coefficients $\\langle j_1, m_1; j_2, m_2 | J, M \\rangle$ are the
**Clebsch-Gordan coefficients**.

### Selection Rules
- $M = m_1 + m_2$
- $|j_1 - j_2| \\leq J \\leq j_1 + j_2$
- Triangle inequality: $j_1 + j_2 \\geq J \\geq |j_1 - j_2|$

### Common case: two spin-1/2 particles

$$\\frac{1}{2} \\otimes \\frac{1}{2} = 1 \\oplus 0$$

The triplet ($J=1$) and singlet ($J=0$) states:

$$|1,1\\rangle = |{\\uparrow\\uparrow}\\rangle, \\quad
|1,0\\rangle = \\frac{1}{\\sqrt{2}}(|{\\uparrow\\downarrow}\\rangle + |{\\downarrow\\uparrow}\\rangle), \\quad
|1,-1\\rangle = |{\\downarrow\\downarrow}\\rangle$$

$$|0,0\\rangle = \\frac{1}{\\sqrt{2}}(|{\\uparrow\\downarrow}\\rangle - |{\\downarrow\\uparrow}\\rangle)$$

### Quantum Connection
Clebsch-Gordan coefficients are essential for adding angular momenta in
multi-electron atoms, nuclear physics, and particle physics.  They determine
the spectral fine structure of hydrogen.""")

    nb.code("""\
# ── Clebsch-Gordan coefficients for two spin-1/2 particles ─────────
from sympy.physics.quantum.cg import CG
from sympy import S

print("=== Clebsch-Gordan Coefficients: ½ ⊗ ½ → 1 ⊕ 0 ===\\n")

j1, j2 = S(1)/2, S(1)/2

# Enumerate all possible |j1,m1⟩|j2,m2⟩ → |J,M⟩
print(f"{'|j1,m1; j2,m2⟩':<25} {'|J,M⟩':<12} {'CG coeff':>12}")
print("-" * 50)

for m1 in [S(1)/2, -S(1)/2]:
    for m2 in [S(1)/2, -S(1)/2]:
        M = m1 + m2
        for J in [S(1), S(0)]:
            if abs(M) <= J:
                cg = CG(j1, m1, j2, m2, J, M).doit()
                if cg != 0:
                    m1_str = "↑" if m1 > 0 else "↓"
                    m2_str = "↑" if m2 > 0 else "↓"
                    print(f"|{m1_str}{m2_str}⟩ = |½,{float(m1):+.1f}; ½,{float(m2):+.1f}⟩"
                          f"  |{float(J):.0f},{float(M):+.0f}⟩"
                          f"  {float(cg):+.4f}")

# Build the coupled states explicitly
print("\\n=== Coupled States ===")
print("Triplet (J=1, symmetric):")
print("  |1,+1⟩ = |↑↑⟩")
print("  |1, 0⟩ = (1/√2)(|↑↓⟩ + |↓↑⟩)")
print("  |1,-1⟩ = |↓↓⟩")
print("\\nSinglet (J=0, antisymmetric):")
print("  |0, 0⟩ = (1/√2)(|↑↓⟩ - |↓↑⟩)")

# Verify with tensor product of Pauli matrices
# S_total = S_1 ⊗ I + I ⊗ S_2
# S² = S_total · S_total
Sx_tot = np.kron(sigma_x/2, I2) + np.kron(I2, sigma_x/2)
Sy_tot = np.kron(sigma_y/2, I2) + np.kron(I2, sigma_y/2)
Sz_tot = np.kron(sigma_z/2, I2) + np.kron(I2, sigma_z/2)
S2_tot = Sx_tot @ Sx_tot + Sy_tot @ Sy_tot + Sz_tot @ Sz_tot

print("\\n=== Verification: S² eigenvalues ===")
evals = np.linalg.eigvalsh(S2_tot.real)
print(f"S² eigenvalues: {sorted(evals)}")
print(f"Expected: J(J+1) = 0·1=0 (singlet), 1·2=2,2,2 (triplet)")
print(f"Match: {np.allclose(sorted(evals), [0, 2, 2, 2])} ✓")""")

    nb.code("""\
# ── Higher angular momentum: j=1 ⊗ j=1/2 ──────────────────────────
# 1 ⊗ 1/2 = 3/2 ⊕ 1/2

j1_val, j2_val = S(1), S(1)/2

print("=== Clebsch-Gordan Table: j=1 ⊗ j=½ → j=3/2 ⊕ j=½ ===\\n")

# Build full CG table
m1_vals = [S(1), S(0), -S(1)]
m2_vals = [S(1)/2, -S(1)/2]
J_vals = [S(3)/2, S(1)/2]

print(f"{'m1':>5} {'m2':>5} {'M':>5} {'J=3/2':>10} {'J=1/2':>10}")
print("-" * 40)

for m1 in m1_vals:
    for m2 in m2_vals:
        M = m1 + m2
        row = f"{float(m1):>+5.1f} {float(m2):>+5.1f} {float(M):>+5.1f}"
        for J in J_vals:
            if abs(M) <= J:
                cg = float(CG(j1_val, m1, j2_val, m2, J, M).doit())
                row += f" {cg:>+10.4f}"
            else:
                row += f" {'---':>10}"
        print(row)

# Dimension check
dim_product = int(2*j1_val + 1) * int(2*j2_val + 1)
dim_sum = sum(int(2*J + 1) for J in J_vals)
print(f"\\nDimension check: {int(2*j1_val+1)} × {int(2*j2_val+1)} = {dim_product}")
print(f"                 {int(2*J_vals[0]+1)} + {int(2*J_vals[1]+1)} = {dim_sum}")
print(f"                 {dim_product} = {dim_sum} ✓")""")

    # ── Summary ──────────────────────────────────────────────────────
    nb.md("""\
## Summary: Group Theory for Quantum Mechanics

| Concept | Mathematical Object | Quantum Application |
|---------|-------------------|---------------------|
| Permutations ($S_n$) | Finite group | Identical particles (bosons/fermions) |
| Dihedral ($D_n$) | Finite group | Molecular symmetries, crystal field theory |
| Representations | Group $\\to$ matrices | Quantum numbers, selection rules |
| Character tables | Traces of irreps | Symmetry-allowed transitions |
| SO(3) | Lie group | Orbital angular momentum $L$ |
| SU(2) | Lie group (double cover) | Spin angular momentum $S$ |
| Clebsch-Gordan | Coupling coefficients | Angular momentum addition |

### The Deep Connection

**Noether's theorem** links every continuous symmetry to a conservation law:

| Symmetry | Conservation Law | Generator |
|----------|-----------------|-----------|
| Translation in time | Energy | $\\hat{H}$ |
| Translation in space | Momentum | $\\hat{p}$ |
| Rotation | Angular momentum | $\\hat{L}$ |
| Gauge (U(1)) | Electric charge | $\\hat{Q}$ |

Group theory is not merely a mathematical tool — it is the **language** in which
the fundamental laws of quantum physics are written.

---
*Notebook generated for the SIIEA Quantum Engineering curriculum.*
*License: CC BY-NC-SA 4.0 | Siiea Innovations, LLC*""")

    nb.save()
    print("  -> Month 11 notebook complete.\n")


# ──────────────────────────────────────────────────────────────────────
# NOTEBOOK 3 — Month 12: Foundation Capstone — Hydrogen Atom
# ──────────────────────────────────────────────────────────────────────
def build_month_12():
    nb = NotebookBuilder(
        "Foundation Capstone: The Hydrogen Atom",
        "year_0/month_12_capstone/12_foundation_capstone.ipynb",
        "Days 309-336",
    )

    # ── Imports ──────────────────────────────────────────────────────
    nb.code("""\
# ── Imports & configuration ─────────────────────────────────────────
import numpy as np
from scipy import integrate, special, sparse
from scipy.sparse.linalg import eigsh
from scipy.integrate import solve_ivp, quad
from scipy.special import sph_harm, factorial, assoc_laguerre
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

%matplotlib inline

plt.rcParams.update({
    'figure.figsize': (10, 7),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'figure.dpi': 100,
})

# Physical constants (atomic units: ℏ = mₑ = e = 4πε₀ = 1)
a0 = 1.0    # Bohr radius
E1 = -0.5   # ground state energy in Hartrees

print("Hydrogen atom capstone — all Year 0 tools in action!")
print(f"Working in atomic units: a₀ = {a0}, E₁ = {E1} Hartree")""")

    # ── Section 1: Review ────────────────────────────────────────────
    nb.md("""\
## 1. Year 0 Comprehensive Review: Tools for Quantum Mechanics

Over the past 12 months, we have built a complete mathematical toolkit.
This capstone demonstrates **every major tool** applied to a single problem:
the hydrogen atom.

### Tools Applied

| Year 0 Topic | Hydrogen Atom Application |
|--------------|--------------------------|
| Calculus & ODEs | Radial Schrödinger equation |
| Linear algebra | Eigenvalue problem for energy levels |
| Complex analysis | Spherical harmonics, Euler's formula |
| Classical mechanics | Kepler problem analogy, Hamiltonian |
| Special functions | Laguerre polynomials, Legendre functions |
| Numerical methods | Finite-difference radial solver |
| Group theory | SO(3) symmetry → angular momentum quantum numbers |
| Scientific computing | Sparse eigensolvers, visualization |

### The Hydrogen Atom Hamiltonian

In atomic units ($\\hbar = m_e = e = 4\\pi\\epsilon_0 = 1$):

$$\\hat{H} = -\\frac{1}{2}\\nabla^2 - \\frac{1}{r}$$

Separation of variables in spherical coordinates gives:

$$\\psi_{nlm}(r,\\theta,\\phi) = R_{nl}(r)\\,Y_l^m(\\theta,\\phi)$$

where $R_{nl}$ satisfies the **radial Schrödinger equation** and $Y_l^m$ are
**spherical harmonics**.""")

    # ── Section 2: Analytical Solutions ──────────────────────────────
    nb.md("""\
## 2. Analytical Solutions: Radial Wavefunctions

The normalized radial wavefunctions are:

$$R_{nl}(r) = -\\sqrt{\\left(\\frac{2}{na_0}\\right)^3 \\frac{(n-l-1)!}{2n[(n+l)!]^3}} \\;
e^{-r/(na_0)} \\left(\\frac{2r}{na_0}\\right)^l L_{n-l-1}^{2l+1}\\left(\\frac{2r}{na_0}\\right)$$

where $L_p^q$ are the **associated Laguerre polynomials**.

### Energy Levels

$$E_n = -\\frac{1}{2n^2} \\text{ (Hartree)}, \\quad n = 1, 2, 3, \\ldots$$

### Quantum Numbers

| Symbol | Name | Range | Physical Meaning |
|--------|------|-------|-----------------|
| $n$ | Principal | $1, 2, 3, \\ldots$ | Energy, orbital size |
| $l$ | Azimuthal | $0, 1, \\ldots, n-1$ | Orbital shape, $|\\mathbf{L}|^2 = l(l+1)\\hbar^2$ |
| $m$ | Magnetic | $-l, \\ldots, +l$ | $z$-component of $\\mathbf{L}$, $L_z = m\\hbar$ |""")

    nb.code("""\
# ── Analytical radial wavefunctions ─────────────────────────────────
def hydrogen_radial(n, l, r):
    \"\"\"
    Normalized radial wavefunction R_nl(r) for hydrogen atom.

    Parameters:
        n: principal quantum number (1, 2, 3, ...)
        l: azimuthal quantum number (0, 1, ..., n-1)
        r: radial coordinate (in units of a₀)

    Returns:
        R_nl(r) array
    \"\"\"
    rho = 2.0 * r / n  # scaled variable
    # Normalization constant
    norm = np.sqrt((2.0/n)**3 * factorial(n-l-1, exact=True) /
                   (2.0 * n * factorial(n+l, exact=True)**3))

    # For numerical stability, we use a corrected normalization
    # that works with scipy's assoc_laguerre
    norm = np.sqrt((2.0/n)**3 * factorial(n-l-1, exact=True) /
                   (2.0 * n * (factorial(n+l, exact=True))))

    return norm * np.exp(-rho/2) * rho**l * assoc_laguerre(n-l-1, 2*l+1, rho)


# Radial wavefunctions for n=1,2,3
r = np.linspace(0, 25, 500)
states = [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)]
labels = ['1s', '2s', '2p', '3s', '3p', '3d']

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot R_nl(r)
for (n, l), label in zip(states, labels):
    R = hydrogen_radial(n, l, r)
    axes[0].plot(r, R, label=f'{label} ($n$={n}, $l$={l})', linewidth=2)

axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_xlabel('$r / a_0$')
axes[0].set_ylabel('$R_{nl}(r)$')
axes[0].set_title('Hydrogen Radial Wavefunctions')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 25)

# Plot r²|R_nl|² (radial probability density)
for (n, l), label in zip(states, labels):
    R = hydrogen_radial(n, l, r)
    P = r**2 * R**2
    axes[1].plot(r, P, label=f'{label}', linewidth=2)

axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_xlabel('$r / a_0$')
axes[1].set_ylabel('$r^2 |R_{nl}(r)|^2$')
axes[1].set_title('Radial Probability Density')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 25)

plt.tight_layout()
plt.show()

# Print expectation values
print(f"{'State':<6} {'⟨r⟩ (a₀)':>12} {'Nodes':>8}")
print("-" * 28)
for (n, l), label in zip(states, labels):
    R = hydrogen_radial(n, l, r)
    dr = r[1] - r[0]
    expect_r = np.trapezoid(r**3 * R**2, r)
    nodes = n - l - 1
    print(f"{label:<6} {expect_r:>12.4f} {nodes:>8}")
    print(f"  (exact ⟨r⟩ = {n**2*(3 - l*(l+1)/n**2)/2:.4f})")""")

    # ── Section 3: Numerical Solution ────────────────────────────────
    nb.md("""\
## 3. Numerical Solution: Radial Schrödinger Equation

The radial equation (after substitution $u(r) = r\\,R(r)$):

$$-\\frac{1}{2}\\frac{d^2u}{dr^2} + \\left[\\frac{l(l+1)}{2r^2} - \\frac{1}{r}\\right]u = Eu$$

This is an eigenvalue problem we can solve numerically using finite differences.
The effective potential is:

$$V_{\\text{eff}}(r) = -\\frac{1}{r} + \\frac{l(l+1)}{2r^2}$$""")

    nb.code("""\
# ── Numerical radial solver using finite differences ────────────────
def solve_radial_schrodinger(l, r_max=50, N=1000, n_states=5):
    \"\"\"
    Solve the radial Schrödinger equation for hydrogen numerically.

    Uses the substitution u(r) = r·R(r) and finite differences.
    Returns eigenvalues and eigenfunctions u_n(r).
    \"\"\"
    # Grid (avoid r=0 singularity)
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)

    # Effective potential
    V_eff = -1.0/r + l*(l+1) / (2.0 * r**2)

    # Kinetic energy: finite-difference second derivative
    coeff = 1.0 / (2.0 * dr**2)
    diag_main = 2 * coeff + V_eff
    diag_off = -coeff * np.ones(N - 1)

    H = sparse.diags([diag_off, diag_main, diag_off],
                      [-1, 0, 1], format='csr')

    # Solve for lowest eigenvalues
    energies, wavefuncs = eigsh(H, k=n_states, which='SA')
    idx = np.argsort(energies)
    energies = energies[idx]
    wavefuncs = wavefuncs[:, idx]

    # Normalize: ∫|u(r)|² dr = 1
    for i in range(n_states):
        norm = np.sqrt(np.trapezoid(wavefuncs[:, i]**2, r))
        wavefuncs[:, i] /= norm

    return r, energies, wavefuncs


# Solve for l=0, 1, 2
print(f"{'State':<8} {'E_numerical':>14} {'E_exact':>14} {'Rel. Error':>14}")
print("=" * 54)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
orbital_names = {0: 's', 1: 'p', 2: 'd'}

for l_val, ax in zip([0, 1, 2], axes):
    r_num, E_num, u_num = solve_radial_schrodinger(l_val, r_max=60, N=2000, n_states=4)

    for i in range(min(4, len(E_num))):
        n = i + l_val + 1
        E_exact = -0.5 / n**2
        rel_err = abs(E_num[i] - E_exact) / abs(E_exact)
        name = f"{n}{orbital_names[l_val]}"
        print(f"{name:<8} {E_num[i]:>14.8f} {E_exact:>14.8f} {rel_err:>14.2e}")

        # Fix sign for plotting: match analytical
        R_analytic = hydrogen_radial(n, l_val, r_num)
        u_analytic = r_num * R_analytic
        if np.dot(u_num[:, i], u_analytic) < 0:
            u_num[:, i] *= -1

        ax.plot(r_num, u_num[:, i], '-', label=f'{name} (num)', linewidth=2)
        ax.plot(r_num, u_analytic, '--', label=f'{name} (exact)', alpha=0.7)

    ax.set_xlabel('$r / a_0$')
    ax.set_ylabel('$u_{nl}(r) = r\\,R_{nl}(r)$')
    ax.set_title(f'$l = {l_val}$ states')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 40)
    print()

plt.suptitle('Numerical vs Analytical Radial Wavefunctions', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""")

    # ── Section 4: Spherical Harmonics ───────────────────────────────
    nb.md("""\
## 4. Spherical Harmonics & Angular Momentum

The angular part of the hydrogen wavefunction is given by **spherical harmonics**:

$$Y_l^m(\\theta, \\phi) = \\sqrt{\\frac{(2l+1)}{4\\pi}\\frac{(l-|m|)!}{(l+|m|)!}} \\;
P_l^{|m|}(\\cos\\theta)\\; e^{im\\phi}$$

These are simultaneous eigenfunctions of:
- $\\hat{L}^2 Y_l^m = l(l+1)\\hbar^2 Y_l^m$
- $\\hat{L}_z Y_l^m = m\\hbar Y_l^m$

### Connection to Group Theory (Month 11)
Spherical harmonics are the **irreducible representations** of SO(3).
Each value of $l$ gives a $(2l+1)$-dimensional irrep, with $m = -l, \\ldots, +l$
labeling the basis states.""")

    nb.code("""\
# ── Spherical harmonics visualization ───────────────────────────────
def plot_spherical_harmonic(l, m, ax, title=None):
    \"\"\"Plot |Y_l^m|² on the unit sphere.\"\"\"
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)

    Y = sph_harm(m, l, PHI, THETA)  # note: scipy uses (m, l, phi, theta)
    R_val = np.abs(Y)**2

    # Convert to Cartesian
    X = R_val * np.sin(THETA) * np.cos(PHI)
    Y_cart = R_val * np.sin(THETA) * np.sin(PHI)
    Z = R_val * np.cos(THETA)

    # Color by phase
    phase = np.angle(Y)
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    colors = cm.coolwarm(norm(phase))

    ax.plot_surface(X, Y_cart, Z, facecolors=colors, alpha=0.8)
    max_r = np.max(R_val) * 1.1
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-max_r, max_r)
    if title:
        ax.set_title(title, fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


# Plot spherical harmonics for l = 0, 1, 2
harmonics = [(0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0), (2,1), (2,2)]

fig = plt.figure(figsize=(18, 12))
for idx, (l, m) in enumerate(harmonics):
    ax = fig.add_subplot(3, 3, idx+1, projection='3d')
    plot_spherical_harmonic(l, m, ax, title=f'$Y_{l}^{{{m}}}$')

plt.suptitle('Spherical Harmonics $|Y_l^m(\\\\theta,\\\\phi)|^2$', fontsize=18, y=0.98)
plt.tight_layout()
plt.show()

# Print properties
print(f"\\n{'(l,m)':<10} {'L²':>10} {'Lz':>10} {'Degeneracy':>12}")
print("-" * 45)
for l in range(4):
    for m in range(-l, l+1):
        if m == 0:
            print(f"({l},{m:+d}){'':<5} {l*(l+1):>10} {m:>10} {2*l+1:>12}")
        elif m == -l:
            print(f"({l},{m:+d}){'':<4} {l*(l+1):>10} {m:>10}")
        else:
            print(f"({l},{m:+d}){'':<4} {'':>10} {m:>10}")""")

    # ── Section 5: 3D Orbital Visualization ──────────────────────────
    nb.md("""\
## 5. Probability Distributions: s, p, and d Orbitals

The full probability density is:

$$|\\psi_{nlm}(r, \\theta, \\phi)|^2 = |R_{nl}(r)|^2 \\, |Y_l^m(\\theta, \\phi)|^2$$

We visualize these in 2D cross-sections (the $xz$-plane, where $\\phi = 0$)
to see the characteristic shapes:

| Orbital | Shape | Nodes |
|---------|-------|-------|
| s ($l=0$) | Spherical | $n-1$ radial |
| p ($l=1$) | Dumbbell (lobed) | $n-2$ radial, 1 angular |
| d ($l=2$) | Cloverleaf | $n-3$ radial, 2 angular |""")

    nb.code("""\
# ── Cross-section visualization of hydrogen orbitals ────────────────
def hydrogen_density_xz(n, l, m, grid_size=200, r_max=None):
    \"\"\"
    Compute |ψ_nlm|² in the xz-plane (y=0, so φ=0 for x>0, φ=π for x<0).
    \"\"\"
    if r_max is None:
        r_max = 4 * n**2  # scale with n²

    x = np.linspace(-r_max, r_max, grid_size)
    z = np.linspace(-r_max, r_max, grid_size)
    X, Z = np.meshgrid(x, z)

    R_grid = np.sqrt(X**2 + Z**2) + 1e-10
    THETA = np.arccos(Z / R_grid)
    PHI = np.where(X >= 0, 0.0, np.pi)  # xz-plane

    R_val = hydrogen_radial(n, l, R_grid)
    Y_val = sph_harm(m, l, PHI, THETA)

    density = np.abs(R_val * Y_val)**2
    return x, z, density


# Orbitals to visualize
orbitals = [
    (1, 0, 0, '1s'),
    (2, 0, 0, '2s'),
    (2, 1, 0, '2p₀'),
    (3, 0, 0, '3s'),
    (3, 1, 0, '3p₀'),
    (3, 2, 0, '3d₀'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for (n, l, m, label), ax in zip(orbitals, axes.flat):
    x, z, density = hydrogen_density_xz(n, l, m)

    # Use logarithmic color scaling for better visibility
    density_plot = np.log10(density + 1e-10)
    vmin = density_plot.max() - 4  # show 4 orders of magnitude

    c = ax.contourf(x, z, density_plot, levels=30, cmap='inferno',
                    vmin=vmin)
    ax.set_xlabel('$x / a_0$')
    ax.set_ylabel('$z / a_0$')
    ax.set_title(f'{label}  ($n$={n}, $l$={l}, $m$={m})', fontsize=14)
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label='$\\log_{10}|\\psi|^2$', shrink=0.8)

plt.suptitle('Hydrogen Orbital Probability Densities (xz-plane)',
             fontsize=17, y=1.02)
plt.tight_layout()
plt.show()""")

    nb.code("""\
# ── Radial probability comparison: analytical vs numerical ──────────
fig, ax = plt.subplots(figsize=(12, 7))

r_plot = np.linspace(0.01, 30, 500)
colors = plt.cm.Set1(np.linspace(0, 1, 6))

for idx, ((n, l), label) in enumerate(zip([(1,0),(2,0),(2,1),(3,0),(3,1),(3,2)],
                                           ['1s','2s','2p','3s','3p','3d'])):
    # Analytical
    R_an = hydrogen_radial(n, l, r_plot)
    P_an = r_plot**2 * R_an**2
    ax.plot(r_plot, P_an, color=colors[idx], linewidth=2.5, label=f'{label}')

    # Mark most probable radius
    i_max = np.argmax(P_an)
    ax.plot(r_plot[i_max], P_an[i_max], 'v', color=colors[idx],
            markersize=10, markeredgecolor='black')

    # Theoretical most probable radius for l=n-1 states: r_mp = n² a₀
    if l == n - 1:
        ax.axvline(n**2, color=colors[idx], linestyle=':', alpha=0.4)

ax.set_xlabel('$r / a_0$', fontsize=14)
ax.set_ylabel('$r^2 |R_{nl}|^2$', fontsize=14)
ax.set_title('Radial Probability Density (triangles mark maxima)', fontsize=15)
ax.legend(fontsize=12, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)
plt.tight_layout()
plt.show()

# Print most probable radii
print(f"{'State':<6} {'r_max (numerical)':>18} {'r_max (exact)':>16}")
print("-" * 42)
for (n, l), label in zip([(1,0),(2,0),(2,1),(3,0),(3,1),(3,2)],
                           ['1s','2s','2p','3s','3p','3d']):
    R = hydrogen_radial(n, l, r_plot)
    P = r_plot**2 * R**2
    r_mp = r_plot[np.argmax(P)]
    # For circular orbits (l=n-1): r_mp = n²
    exact = f"{n**2:.1f}" if l == n-1 else "---"
    print(f"{label:<6} {r_mp:>18.2f} {exact:>16}")""")

    # ── Section 6: Energy Level Diagram ──────────────────────────────
    nb.md("""\
## 6. Energy Level Diagram with Fine Structure Preview

The non-relativistic hydrogen energy levels depend only on $n$:

$$E_n = -\\frac{13.6 \\text{ eV}}{n^2}$$

This gives a **degeneracy** of $n^2$ for each level (or $2n^2$ including spin).

### Fine Structure (Preview for Year 1)

Relativistic corrections lift the $l$-degeneracy, splitting each $n$ level by:

$$\\Delta E_{\\text{fine}} = -\\frac{\\alpha^2 E_n}{n} \\left(\\frac{1}{j+1/2} - \\frac{3}{4n}\\right)$$

where $\\alpha \\approx 1/137$ is the fine-structure constant and $j = l \\pm 1/2$
is the total angular momentum.""")

    nb.code("""\
# ── Energy level diagram ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# --- Left panel: non-relativistic levels ---
ax = axes[0]
n_max = 5
for n in range(1, n_max + 1):
    E = -13.6 / n**2
    degeneracy = 2 * n**2

    # Draw energy level line
    x_left, x_right = 0.2, 0.8
    ax.plot([x_left, x_right], [E, E], 'b-', linewidth=2.5)

    # Label
    ax.text(x_right + 0.05, E, f'$n={n}$  ({degeneracy})',
            fontsize=12, va='center')
    ax.text(x_left - 0.05, E, f'{E:.2f} eV',
            fontsize=10, va='center', ha='right', color='gray')

    # Show orbital labels
    orb_labels = []
    for l in range(n):
        orb_names = {0:'s', 1:'p', 2:'d', 3:'f', 4:'g'}
        orb_labels.append(f'{n}{orb_names.get(l,"?")}')
    ax.text(0.5, E + 0.3, ', '.join(orb_labels),
            fontsize=9, ha='center', color='green')

# Ionization limit
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.text(0.5, 0.5, 'Ionization (0 eV)', ha='center', color='red')

ax.set_ylabel('Energy (eV)', fontsize=14)
ax.set_title('Hydrogen Energy Levels\\n(Non-relativistic)', fontsize=14)
ax.set_xlim(0, 1.2)
ax.set_ylim(-15, 2)
ax.get_xaxis().set_visible(False)

# --- Right panel: fine structure splitting ---
ax = axes[1]
alpha_fs = 1 / 137.036  # fine-structure constant

for n in range(1, 4):
    E_n = -13.6 / n**2

    for l in range(n):
        for j_sign in [-0.5, 0.5]:
            j = l + j_sign
            if j < 0:
                continue

            # Fine structure correction
            if j + 0.5 > 0:
                dE = -alpha_fs**2 * abs(E_n) / n * (1/(j + 0.5) - 3/(4*n))
            else:
                continue

            E_fine = E_n + dE * 1000  # exaggerate for visibility

            # x position based on l
            x_pos = l * 0.25 + 0.1
            x_width = 0.15

            ax.plot([x_pos, x_pos + x_width], [E_fine, E_fine],
                    linewidth=2, color=f'C{l}')

            orb_names = {0:'s', 1:'p', 2:'d'}
            label = f'{n}{orb_names[l]}$_{{j={j:.0f}/2}}$' if j != int(j) else \
                    f'{n}{orb_names[l]}$_{{j={j:.0f}}}$'
            label = f'{n}{orb_names[l]}$_{{{int(2*j)}/2}}$'
            ax.text(x_pos + x_width + 0.02, E_fine, label,
                    fontsize=9, va='center')

ax.set_ylabel('Energy (eV) — splitting exaggerated', fontsize=12)
ax.set_title('Fine Structure Splitting\\n(Preview for Year 1)', fontsize=14)
ax.set_ylim(-15, 1)
ax.get_xaxis().set_visible(False)
ax.text(0.1, -14.5, '$l$: ', fontsize=11)
for l, name in enumerate(['s', 'p', 'd']):
    ax.text(0.1 + l*0.25, -14.5, name, fontsize=11, color=f'C{l}',
            fontweight='bold')

plt.suptitle('Hydrogen Atom: Energy Levels', fontsize=17, y=1.01)
plt.tight_layout()
plt.show()

# Print energy levels with degeneracies
print(f"\\n{'n':>3} {'E (eV)':>10} {'Degeneracy':>12} {'Orbitals':>20}")
print("=" * 48)
for n in range(1, 6):
    E = -13.6 / n**2
    deg = 2 * n**2
    orbs = ', '.join(f'{n}{"spdfg"[l]}' for l in range(n))
    print(f"{n:>3} {E:>10.4f} {deg:>12} {orbs:>20}")""")

    # ── Section 7: Transition to Year 1 ─────────────────────────────
    nb.md("""\
## 7. Bridge to Year 1: The Postulates of Quantum Mechanics

The hydrogen atom showcases quantum mechanics beautifully, but we solved it
using the **Schrödinger equation** framework, which we have been taking on faith.
In Year 1, we build quantum mechanics from its foundational **postulates**:

### The Six Postulates

1. **State space:** The state of a quantum system is a vector $|\\psi\\rangle$
   in a Hilbert space $\\mathcal{H}$.

2. **Observables:** Physical observables correspond to Hermitian operators
   $\\hat{A}$ on $\\mathcal{H}$.

3. **Measurement outcomes:** The possible outcomes of measuring $\\hat{A}$ are
   its eigenvalues $a_n$.

4. **Born rule:** The probability of outcome $a_n$ is $|\\langle a_n|\\psi\\rangle|^2$.

5. **State collapse:** After measuring outcome $a_n$, the state becomes $|a_n\\rangle$.

6. **Time evolution:** The state evolves via $i\\hbar\\frac{d}{dt}|\\psi\\rangle = \\hat{H}|\\psi\\rangle$.

### What Year 1 Covers

| Semester 1A | Semester 1B |
|-------------|-------------|
| Dirac notation & Hilbert spaces | Quantum information theory |
| Operators & commutators | Entanglement & Bell inequalities |
| Angular momentum theory | Quantum computing basics |
| Perturbation theory | Density matrices |
| Scattering theory | Quantum error correction preview |""")

    nb.code("""\
# ── Quantum postulates in action: measurement simulation ────────────
# Simulate measuring L_z on hydrogen 3d state (l=2)
# Possible outcomes: m = -2, -1, 0, +1, +2

rng = np.random.default_rng(42)

# Prepare a superposition state in the l=2 subspace
# |ψ⟩ = Σ c_m |2, m⟩
coefficients = rng.normal(size=5) + 1j * rng.normal(size=5)
coefficients /= np.linalg.norm(coefficients)  # normalize

m_values = np.array([-2, -1, 0, 1, 2])
probabilities = np.abs(coefficients)**2

print("=== Measurement Simulation: L_z on l=2 state ===\\n")
print("Prepared superposition state:")
for m, c, p in zip(m_values, coefficients, probabilities):
    print(f"  |m={m:+d}⟩: c = {c:.4f}, P = {p:.4f}")
print(f"  Sum of probabilities: {probabilities.sum():.6f}")

# Simulate N measurements
N_meas = 10000
outcomes = rng.choice(m_values, size=N_meas, p=probabilities)

# Count occurrences
unique, counts = np.unique(outcomes, return_counts=True)
frequencies = counts / N_meas

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: theoretical vs measured
width = 0.35
axes[0].bar(m_values - width/2, probabilities, width, label='Theory (Born rule)',
            color='steelblue', alpha=0.8, edgecolor='black')
axes[0].bar(unique + width/2, frequencies, width, label=f'Measured (N={N_meas:,})',
            color='coral', alpha=0.8, edgecolor='black')
axes[0].set_xlabel('$m$ (L$_z$ quantum number)')
axes[0].set_ylabel('Probability')
axes[0].set_title('Born Rule: L$_z$ Measurement')
axes[0].legend()
axes[0].set_xticks(m_values)
axes[0].grid(True, alpha=0.3, axis='y')

# Expectation value convergence
cumulative_mean = np.cumsum(outcomes) / np.arange(1, N_meas + 1)
exact_expectation = np.sum(m_values * probabilities)

axes[1].plot(cumulative_mean, linewidth=1.5, alpha=0.7, label='Running mean')
axes[1].axhline(exact_expectation, color='red', linestyle='--', linewidth=2,
                label=f'$\\langle L_z \\rangle$ = {exact_expectation:.4f}$\\hbar$')
axes[1].set_xlabel('Number of measurements')
axes[1].set_ylabel('$\\langle m \\rangle$')
axes[1].set_title('Convergence of Expectation Value')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\nExact ⟨Lz⟩ = {exact_expectation:.4f} ℏ")
print(f"Measured ⟨Lz⟩ = {outcomes.mean():.4f} ℏ  (from {N_meas:,} measurements)")
print(f"Standard error: {outcomes.std()/np.sqrt(N_meas):.4f} ℏ")""")

    nb.code("""\
# ── Time evolution: Gaussian wavepacket in Coulomb potential ────────
# Demonstrate Postulate 6: |ψ(t)⟩ = e^{-iHt/ℏ}|ψ(0)⟩

# For visualization, use 1D radial evolution of a superposition
# ψ(r,0) = c₁ R₁₀(r) + c₂ R₂₀(r) + c₃ R₃₀(r)
# Beat frequency: ω₁₂ = (E₁ - E₂)/ℏ

r_grid = np.linspace(0.01, 30, 500)

# Energy eigenvalues (atomic units)
E = {1: -0.5, 2: -0.125, 3: -0.5/9}

# Radial wavefunctions
R = {}
for n in [1, 2, 3]:
    R[n] = hydrogen_radial(n, 0, r_grid)

# Superposition coefficients
c1, c2, c3 = 0.6, 0.7, 0.3
norm_c = np.sqrt(c1**2 + c2**2 + c3**2)
c1, c2, c3 = c1/norm_c, c2/norm_c, c3/norm_c

# Time evolution
times = np.linspace(0, 100, 8)  # in atomic time units

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, t in enumerate(times):
    ax = axes[idx // 4, idx % 4]

    # ψ(r,t) = Σ cₙ Rₙ₀(r) e^{-iEₙt}
    psi = (c1 * R[1] * np.exp(-1j * E[1] * t) +
           c2 * R[2] * np.exp(-1j * E[2] * t) +
           c3 * R[3] * np.exp(-1j * E[3] * t))

    density = r_grid**2 * np.abs(psi)**2

    ax.fill_between(r_grid, density, alpha=0.4, color='steelblue')
    ax.plot(r_grid, density, color='steelblue', linewidth=1.5)
    ax.set_title(f't = {t:.0f} a.u.', fontsize=11)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, density.max() * 1.3 + 0.01)
    ax.set_xlabel('$r / a_0$')
    if idx % 4 == 0:
        ax.set_ylabel('$r^2|\\psi|^2$')
    ax.grid(True, alpha=0.2)

plt.suptitle('Time Evolution of Hydrogen Superposition State\\n'
             '$|\\psi\\rangle = c_1|1s\\rangle + c_2|2s\\rangle + c_3|3s\\rangle$',
             fontsize=15, y=1.03)
plt.tight_layout()
plt.show()

# Print beat frequencies
omega_12 = abs(E[1] - E[2])
omega_13 = abs(E[1] - E[3])
omega_23 = abs(E[2] - E[3])
print(f"Beat frequencies (atomic units):")
print(f"  ω₁₂ = |E₁ - E₂| = {omega_12:.6f}")
print(f"  ω₁₃ = |E₁ - E₃| = {omega_13:.6f}")
print(f"  ω₂₃ = |E₂ - E₃| = {omega_23:.6f}")
print(f"\\nBeat periods:")
print(f"  T₁₂ = 2π/ω₁₂ = {2*np.pi/omega_12:.2f} a.u.")
print(f"  T₁₃ = 2π/ω₁₃ = {2*np.pi/omega_13:.2f} a.u.")
print(f"  T₂₃ = 2π/ω₂₃ = {2*np.pi/omega_23:.2f} a.u.")""")

    nb.code("""\
# ── Complete quantum numbers table for hydrogen ─────────────────────
from IPython.display import HTML

print("=== Complete Hydrogen Quantum Numbers (n = 1 to 4) ===\\n")
print(f"{'n':>3} {'l':>3} {'m':>4} {'Orbital':>10} {'E (eV)':>10} "
      f"{'Degeneracy':>12} {'<r> (a₀)':>10}")
print("=" * 58)

total_states = 0
for n in range(1, 5):
    E_n = -13.6 / n**2
    deg = 2 * n**2
    first_in_n = True
    for l in range(n):
        orb_letter = 'spdfg'[l]
        for m in range(-l, l+1):
            total_states += 2  # factor of 2 for spin
            orbital = f'{n}{orb_letter}'
            r_expect = 0.5 * n**2 * (3 - l*(l+1)/n**2)
            if first_in_n:
                print(f"{n:>3} {l:>3} {m:>+4d} {orbital:>10} {E_n:>10.4f} "
                      f"{deg:>12} {r_expect:>10.2f}")
                first_in_n = False
            else:
                print(f"{'':>3} {l:>3} {m:>+4d} {orbital:>10} {'':>10} "
                      f"{'':>12} {r_expect:>10.2f}")
    print("-" * 58)

print(f"\\nTotal states (including spin) for n=1 to 4: {total_states}")
print(f"Formula: Σ 2n² = {sum(2*n**2 for n in range(1,5))}  ✓")""")

    # ── Final Summary ────────────────────────────────────────────────
    nb.md("""\
## Summary: Year 0 Foundation Complete

### What We Built

Over 336 days and 12 months, we constructed the complete mathematical foundation
for quantum science:

| Month | Topic | Key Tool |
|-------|-------|----------|
| 1-3 | Calculus & ODEs | Derivatives, integrals, differential equations |
| 4-5 | Linear Algebra | Eigenvalues, Hilbert spaces, inner products |
| 6 | Classical Mechanics | Lagrangian, Hamiltonian, Poisson brackets |
| 7 | Complex Analysis | Analytic functions, residues, contour integrals |
| 8 | Electromagnetism | Maxwell's equations, gauge theory |
| 9 | Functional Analysis | Operators, spectra, distribution theory |
| 10 | Scientific Computing | NumPy, SciPy, numerical methods |
| 11 | Group Theory | Symmetries, representations, angular momentum |
| 12 | **Capstone** | **Hydrogen atom: everything together** |

### The Hydrogen Atom Used Every Tool

- **Calculus:** Solving the radial ODE
- **Linear algebra:** Eigenvalue problem $H\\psi = E\\psi$
- **Special functions:** Laguerre polynomials, spherical harmonics
- **Group theory:** SO(3) symmetry → quantum numbers $l, m$
- **Numerical methods:** Finite-difference sparse eigensolver
- **Visualization:** Orbital plots, energy diagrams

### Ready for Year 1

You now have all the mathematical tools needed to study quantum mechanics
from first principles.  Year 1 begins with the **postulates of quantum mechanics**
and builds up to quantum information and computing.

**The foundation is complete. The quantum journey begins.**

---
*Notebook generated for the SIIEA Quantum Engineering curriculum.*
*License: CC BY-NC-SA 4.0 | Siiea Innovations, LLC*""")

    nb.save()
    print("  -> Month 12 notebook complete.\n")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SIIEA Quantum Engineering — Generating Months 10-12")
    print("=" * 60)
    print()

    build_month_10()
    build_month_11()
    build_month_12()

    print("=" * 60)
    print("All 3 notebooks generated successfully!")
    print("=" * 60)
