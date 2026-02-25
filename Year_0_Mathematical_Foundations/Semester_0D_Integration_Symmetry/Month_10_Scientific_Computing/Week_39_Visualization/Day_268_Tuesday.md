# Day 268: Scientific Plotting & Data Visualization

## Schedule Overview
**Date**: Week 39, Day 2 (Tuesday)
**Duration**: 7 hours
**Theme**: Advanced Visualization Techniques for Experimental and Theoretical Physics

| Block | Duration | Activity |
|-------|----------|----------|
| Morning | 3 hours | Error bars, log scales, and uncertainty visualization |
| Afternoon | 2.5 hours | Colormaps, contour plots, and 2D data representation |
| Evening | 1.5 hours | Computational lab: Quantum measurement visualization |

---

## Learning Objectives

By the end of this day, you will be able to:

1. Add error bars to plots representing measurement uncertainty
2. Choose appropriate axis scales (linear, log, symlog) for different data types
3. Select and apply colormaps for 2D data visualization
4. Create contour plots for wave function probability densities
5. Generate heatmaps for correlation matrices and density matrices
6. Visualize quantum energy spectra with proper formatting

---

## Core Content

### 1. Error Bars and Uncertainty Visualization

In experimental physics and numerical simulations, measurements always have associated uncertainties. Proper visualization requires showing both the central value and its error.

#### Basic Error Bars

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulated experimental data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = 2.5 * x + np.random.normal(0, 0.5, len(x))  # Noisy linear relationship
y_err = 0.3 + 0.1 * np.random.rand(len(x))  # Measurement uncertainties

fig, ax = plt.subplots(figsize=(10, 6))

# Plot with error bars
ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=5, capthick=1.5,
           color='blue', ecolor='gray', label='Data ± uncertainty')

# Fit line for comparison
coeffs = np.polyfit(x, y, 1)
ax.plot(x, np.polyval(coeffs, x), 'r--', label=f'Linear fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')

ax.set_xlabel('Measurement index', fontsize=12)
ax.set_ylabel('Observable value', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

#### Asymmetric Error Bars

For quantities like lifetimes or rates, errors may be asymmetric:

```python
# Asymmetric errors: [lower_error, upper_error]
y_err_asym = np.array([[0.2, 0.3, 0.25, 0.35, 0.2, 0.3, 0.25, 0.35, 0.3, 0.25],
                       [0.4, 0.5, 0.45, 0.6, 0.4, 0.5, 0.45, 0.6, 0.5, 0.45]])

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(x, y, yerr=y_err_asym, fmt='s', capsize=5,
           color='green', ecolor='green', alpha=0.7)
ax.set_xlabel('Measurement')
ax.set_ylabel('Value')
ax.set_title('Asymmetric Error Bars')
plt.show()
```

#### Error Bands for Continuous Data

```python
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/5)
y_err = 0.1 + 0.05 * np.abs(np.cos(x))

fig, ax = plt.subplots(figsize=(10, 6))

# Shaded error band
ax.fill_between(x, y - y_err, y + y_err, alpha=0.3, color='blue', label='Uncertainty')
ax.plot(x, y, 'b-', linewidth=2, label='Mean value')

ax.set_xlabel('Time (a.u.)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### 2. Logarithmic and Semi-Logarithmic Scales

Many physical quantities span multiple orders of magnitude, requiring logarithmic scales.

#### Log-Log Plots

Power law relationships $$y = ax^n$$ appear as straight lines on log-log plots:

```python
# Power spectrum example
frequencies = np.logspace(-2, 2, 100)  # 0.01 to 100 Hz
# 1/f noise spectrum
power = 1e-3 / frequencies + 1e-5 * np.random.rand(len(frequencies))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale - hard to see structure
axes[0].plot(frequencies, power)
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Power Spectral Density')
axes[0].set_title('Linear Scale')

# Log-log scale - reveals power law
axes[1].loglog(frequencies, power, 'b-')
axes[1].loglog(frequencies, 1e-3/frequencies, 'r--', label='$1/f$ reference')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power Spectral Density')
axes[1].set_title('Log-Log Scale')
axes[1].legend()
axes[1].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Semi-Log Plots

Exponential decay appears linear on semi-log (log y, linear x):

```python
t = np.linspace(0, 5, 100)
tau = 1.2  # Decay constant
N = 1000 * np.exp(-t/tau) + np.random.normal(0, 20, len(t))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
axes[0].plot(t, N, 'o-', markersize=3)
axes[0].set_xlabel('Time (μs)')
axes[0].set_ylabel('Counts')
axes[0].set_title('Linear Scale')

# Semilog scale - exponential becomes linear
axes[1].semilogy(t, N, 'o-', markersize=3)
axes[1].set_xlabel('Time (μs)')
axes[1].set_ylabel('Counts (log scale)')
axes[1].set_title(f'Semi-Log Scale: τ = {tau} μs visible as slope')
axes[1].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Symmetric Log Scale (symlog)

For data that spans negative and positive values across many orders of magnitude:

```python
x = np.linspace(-100, 100, 1000)
y = x**3 / 1000  # Cubic function

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x, y)
axes[0].set_title('Linear')

axes[1].plot(x, y)
axes[1].set_yscale('symlog', linthresh=1)  # Linear within |y| < 1
axes[1].set_title('Symlog (linthresh=1)')

axes[2].plot(x, y)
axes[2].set_yscale('symlog', linthresh=10)
axes[2].set_title('Symlog (linthresh=10)')

for ax in axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3. Colormaps for 2D Data

Colormaps are essential for visualizing functions of two variables, like probability densities.

#### Colormap Selection

```python
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm, SymLogNorm

# 2D Gaussian probability density
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

# Show different colormaps
cmaps = ['viridis', 'plasma', 'cividis', 'RdBu', 'coolwarm', 'hot']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, cmap_name in zip(axes, cmaps):
    im = ax.pcolormesh(X, Y, Z, cmap=cmap_name, shading='auto')
    ax.set_title(cmap_name, fontsize=14)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Colormap Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
```

#### Colormap Guidelines for Physics

| Data Type | Recommended Colormap | Reason |
|-----------|---------------------|--------|
| Sequential (0 to max) | `viridis`, `plasma`, `cividis` | Perceptually uniform, colorblind-safe |
| Diverging (- to +) | `RdBu`, `coolwarm`, `seismic` | Symmetric around zero |
| Phase (0 to 2π) | `hsv`, `twilight` | Cyclic, wraps smoothly |
| Probability density | `viridis`, `inferno` | Emphasizes peaks |

#### Normalizations

```python
# Data with wide dynamic range
Z_wide = np.exp(-(X**2 + Y**2)) + 1e-4 * np.random.rand(*X.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Linear normalization
im1 = axes[0].pcolormesh(X, Y, Z_wide, norm=Normalize(vmin=0, vmax=1),
                         cmap='viridis', shading='auto')
axes[0].set_title('Linear Normalization')
plt.colorbar(im1, ax=axes[0])

# Log normalization (for positive data spanning decades)
im2 = axes[1].pcolormesh(X, Y, Z_wide + 1e-6, norm=LogNorm(vmin=1e-4, vmax=1),
                         cmap='viridis', shading='auto')
axes[1].set_title('Log Normalization')
plt.colorbar(im2, ax=axes[1])

# Custom vmin/vmax
im3 = axes[2].pcolormesh(X, Y, Z_wide, vmin=0.1, vmax=0.9,
                         cmap='viridis', shading='auto')
axes[2].set_title('Custom Range [0.1, 0.9]')
plt.colorbar(im3, ax=axes[2])

for ax in axes:
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

### 4. Contour Plots

Contour plots show level curves of 2D functions—essential for wave functions and potential energy surfaces.

#### Basic Contour Plots

```python
# 2D harmonic oscillator potential
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
V = 0.5 * (X**2 + Y**2)  # Potential

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Contour lines only
cs1 = axes[0].contour(X, Y, V, levels=10)
axes[0].clabel(cs1, inline=True, fontsize=10)
axes[0].set_title('Contour Lines')

# Filled contours
cs2 = axes[1].contourf(X, Y, V, levels=20, cmap='viridis')
plt.colorbar(cs2, ax=axes[1])
axes[1].set_title('Filled Contours')

# Combined
cs3 = axes[2].contourf(X, Y, V, levels=20, cmap='viridis', alpha=0.8)
axes[2].contour(X, Y, V, levels=10, colors='white', linewidths=0.5)
plt.colorbar(cs3, ax=axes[2])
axes[2].set_title('Combined')

for ax in axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

#### Quantum Probability Density Contours

```python
# 2D quantum harmonic oscillator first excited state
# ψ_10 ∝ x·exp(-(x²+y²)/2)
psi_10 = X * np.exp(-(X**2 + Y**2)/2)
prob_10 = np.abs(psi_10)**2

# Normalize for visualization
prob_10 /= prob_10.max()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Wave function (can be negative)
levels_wf = np.linspace(-1, 1, 21) * np.abs(psi_10).max()
cs1 = axes[0].contourf(X, Y, psi_10, levels=levels_wf, cmap='RdBu')
axes[0].contour(X, Y, psi_10, levels=[0], colors='black', linewidths=2)
plt.colorbar(cs1, ax=axes[0], label=r'$\psi_{10}(x,y)$')
axes[0].set_title(r'Wave Function $\psi_{10}$')

# Probability density (always positive)
cs2 = axes[1].contourf(X, Y, prob_10, levels=20, cmap='viridis')
axes[1].contour(X, Y, prob_10, levels=[0.1, 0.5, 0.9],
                colors='white', linewidths=1)
plt.colorbar(cs2, ax=axes[1], label=r'$|\psi_{10}|^2$')
axes[1].set_title(r'Probability Density $|\psi_{10}|^2$')

for ax in axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

### 5. Heatmaps

Heatmaps are ideal for visualizing matrices, including density matrices in quantum mechanics.

#### Basic Heatmap

```python
# Correlation matrix example
np.random.seed(42)
data = np.random.randn(5, 100)
corr_matrix = np.corrcoef(data)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(im, label='Correlation')

# Add text annotations
for i in range(5):
    for j in range(5):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                      ha='center', va='center', color='black')

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([f'Var {i}' for i in range(5)])
ax.set_yticklabels([f'Var {i}' for i in range(5)])
ax.set_title('Correlation Matrix')
plt.show()
```

#### Density Matrix Visualization

```python
def plot_density_matrix(rho, title="Density Matrix"):
    """
    Plot a quantum density matrix showing both magnitude and phase.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Magnitude
    im1 = axes[0].imshow(np.abs(rho), cmap='viridis')
    axes[0].set_title(r'$|\rho_{ij}|$')
    plt.colorbar(im1, ax=axes[0], label='Magnitude')

    # Phase (only where magnitude is significant)
    phase = np.angle(rho)
    phase_masked = np.ma.masked_where(np.abs(rho) < 0.01, phase)
    im2 = axes[1].imshow(phase_masked, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(r'$\arg(\rho_{ij})$')
    cbar = plt.colorbar(im2, ax=axes[1], label='Phase')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    for ax in axes:
        ax.set_xlabel('Column index')
        ax.set_ylabel('Row index')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig, axes

# Example: Coherent superposition density matrix
# |ψ⟩ = (|0⟩ + |1⟩)/√2
psi = np.array([1, 1]) / np.sqrt(2)
rho_coherent = np.outer(psi, psi.conj())

# Mixed state for comparison
rho_mixed = 0.5 * np.eye(2)

fig1, _ = plot_density_matrix(rho_coherent, "Pure State: Coherent Superposition")
fig2, _ = plot_density_matrix(rho_mixed, "Mixed State: Maximally Mixed")
plt.show()
```

### 6. Energy Level Diagrams

Visualizing energy spectra is fundamental to quantum mechanics.

```python
def plot_energy_spectrum(energies, labels=None, ax=None, width=0.8):
    """Plot horizontal energy level diagram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    n_levels = len(energies)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_levels))

    for i, (E, color) in enumerate(zip(energies, colors)):
        ax.hlines(E, 0.5 - width/2, 0.5 + width/2, color=color, linewidth=3)
        if labels:
            ax.text(0.5 + width/2 + 0.1, E, labels[i], va='center', fontsize=11)

    ax.set_xlim(0, 1.5)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return fig, ax

# Compare different quantum systems
fig, axes = plt.subplots(1, 3, figsize=(14, 8))

# Harmonic oscillator: E_n = (n + 1/2)ℏω
n = np.arange(6)
E_ho = n + 0.5
labels_ho = [f'$n={i}$' for i in n]
plot_energy_spectrum(E_ho, labels_ho, axes[0])
axes[0].set_title('Harmonic Oscillator\n$E_n = (n+\\frac{1}{2})\\hbar\\omega$')

# Particle in a box: E_n = n²E₁
n = np.arange(1, 7)
E_box = n**2
labels_box = [f'$n={i}$' for i in n]
plot_energy_spectrum(E_box, labels_box, axes[1])
axes[1].set_title('Particle in Box\n$E_n = n^2 E_1$')

# Hydrogen atom: E_n = -13.6/n² eV
n = np.arange(1, 7)
E_H = -13.6 / n**2
labels_H = [f'$n={i}$' for i in n]
plot_energy_spectrum(E_H, labels_H, axes[2])
axes[2].set_title('Hydrogen Atom\n$E_n = -13.6/n^2$ eV')
axes[2].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[2].text(0.6, 0.5, 'Continuum', fontsize=10)

plt.tight_layout()
plt.show()
```

---

## Quantum Mechanics Connection

### Visualizing Quantum Measurement Results

Quantum measurements yield probabilistic outcomes. Proper visualization must convey:
1. **Probability distribution** of possible outcomes
2. **Statistical uncertainty** from finite sampling
3. **Comparison with theoretical predictions**

```python
def quantum_measurement_visualization(psi, basis_labels, n_measurements=1000):
    """
    Simulate and visualize quantum measurements.

    Parameters
    ----------
    psi : array
        Quantum state vector (normalized)
    basis_labels : list
        Labels for basis states
    n_measurements : int
        Number of measurements to simulate
    """
    # Compute probabilities
    probs = np.abs(psi)**2

    # Simulate measurements
    outcomes = np.random.choice(len(psi), size=n_measurements, p=probs)

    # Count occurrences
    unique, counts = np.unique(outcomes, return_counts=True)
    measured_probs = counts / n_measurements
    measured_err = np.sqrt(measured_probs * (1 - measured_probs) / n_measurements)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart with theoretical overlay
    x = np.arange(len(psi))
    width = 0.35

    axes[0].bar(x - width/2, probs, width, label='Theory', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, measured_probs, width, yerr=measured_err,
               label=f'Experiment (n={n_measurements})', color='red', alpha=0.7,
               capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(basis_labels)
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Measurement Outcome Distribution')
    axes[0].legend()

    # Histogram of individual measurements
    axes[1].hist(outcomes, bins=np.arange(len(psi)+1)-0.5,
                 edgecolor='black', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(basis_labels)
    axes[1].set_ylabel('Counts')
    axes[1].set_title('Raw Measurement Counts')

    plt.tight_layout()
    return fig

# Example: Three-level system
psi_3level = np.array([0.5, 0.7, 0.5])  # Unnormalized
psi_3level = psi_3level / np.linalg.norm(psi_3level)
labels_3level = [r'$|0\rangle$', r'$|1\rangle$', r'$|2\rangle$']

fig = quantum_measurement_visualization(psi_3level, labels_3level, n_measurements=1000)
plt.show()
```

### Uncertainty Relation Visualization

The Heisenberg uncertainty principle states:
$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

```python
def uncertainty_visualization():
    """Visualize position-momentum uncertainty relation."""
    sigmas = np.linspace(0.3, 3, 10)
    hbar = 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Position space wave function
    x = np.linspace(-10, 10, 1000)
    for sigma in sigmas[::2]:
        psi_x = (2*np.pi*sigma**2)**(-0.25) * np.exp(-x**2/(4*sigma**2))
        axes[0].plot(x, np.abs(psi_x)**2, label=f'σ={sigma:.1f}')
    axes[0].set_xlabel('Position x')
    axes[0].set_ylabel(r'$|\psi(x)|^2$')
    axes[0].set_title('Position Space')
    axes[0].legend()

    # Momentum space wave function (Fourier transform of Gaussian)
    p = np.linspace(-10, 10, 1000)
    for sigma in sigmas[::2]:
        sigma_p = hbar / (2 * sigma)  # Momentum spread
        psi_p = (2*np.pi*sigma_p**2)**(-0.25) * np.exp(-p**2/(4*sigma_p**2))
        axes[1].plot(p, np.abs(psi_p)**2, label=f'σ={sigma:.1f}')
    axes[1].set_xlabel('Momentum p')
    axes[1].set_ylabel(r'$|\phi(p)|^2$')
    axes[1].set_title('Momentum Space')
    axes[1].legend()

    # Uncertainty product
    delta_x = sigmas
    delta_p = hbar / (2 * sigmas)
    product = delta_x * delta_p

    axes[2].plot(sigmas, product, 'bo-', markersize=8)
    axes[2].axhline(hbar/2, color='r', linestyle='--', label=r'$\hbar/2$')
    axes[2].fill_between(sigmas, 0, hbar/2, alpha=0.2, color='red', label='Forbidden region')
    axes[2].set_xlabel(r'$\Delta x$')
    axes[2].set_ylabel(r'$\Delta x \cdot \Delta p$')
    axes[2].set_title('Uncertainty Product')
    axes[2].legend()
    axes[2].set_ylim(0, max(product)*1.1)

    plt.tight_layout()
    return fig

fig = uncertainty_visualization()
plt.show()
```

---

## Worked Examples

### Example 1: Decay Curve with Uncertainties

**Problem**: Visualize radioactive decay data with Poisson counting statistics.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Simulated decay data
np.random.seed(42)
t = np.linspace(0, 10, 20)
tau = 2.5  # True lifetime
N0 = 1000  # Initial counts

# True curve with Poisson noise
N_true = N0 * np.exp(-t / tau)
N_measured = np.random.poisson(N_true)

# Poisson uncertainty: sqrt(N)
N_err = np.sqrt(N_measured)
N_err[N_err == 0] = 1  # Avoid zero error

# Fit exponential
def exponential(t, N0, tau):
    return N0 * np.exp(-t / tau)

popt, pcov = curve_fit(exponential, t, N_measured, sigma=N_err,
                       p0=[1000, 2], absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

# Main plot
axes[0].errorbar(t, N_measured, yerr=N_err, fmt='ko', capsize=3,
                label='Data', markersize=5)
t_fine = np.linspace(0, 10, 200)
axes[0].plot(t_fine, exponential(t_fine, *popt), 'r-', linewidth=2,
            label=f'Fit: τ = {popt[1]:.2f} ± {perr[1]:.2f}')
axes[0].plot(t_fine, exponential(t_fine, N0, tau), 'b--', alpha=0.5,
            label=f'True: τ = {tau:.2f}')

axes[0].set_ylabel('Counts', fontsize=12)
axes[0].set_yscale('log')
axes[0].legend(fontsize=11)
axes[0].set_title('Radioactive Decay Measurement', fontsize=14)
axes[0].grid(True, alpha=0.3, which='both')

# Residuals
residuals = (N_measured - exponential(t, *popt)) / N_err
axes[1].errorbar(t, residuals, yerr=1, fmt='ko', capsize=3, markersize=5)
axes[1].axhline(0, color='r', linestyle='-')
axes[1].fill_between(t, -2, 2, alpha=0.2, color='gray', label='±2σ')
axes[1].set_xlabel('Time (half-lives)', fontsize=12)
axes[1].set_ylabel('Residual (σ)', fontsize=12)
axes[1].set_ylim(-4, 4)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decay_measurement.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Example 2: 2D Probability Density Map

**Problem**: Create a publication-quality contour plot of a 2D coherent state.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt

# 2D coherent state displaced from origin
alpha = 2 + 1j  # Coherent state parameter
sigma = 1 / np.sqrt(2)

x = np.linspace(-2, 6, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)

# Gaussian probability distribution centered at (Re(α), Im(α))
x0, y0 = alpha.real, alpha.imag
prob = (1 / (2 * np.pi * sigma**2)) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Filled contours
levels = np.linspace(0, prob.max(), 20)
cs = ax.contourf(X, Y, prob, levels=levels, cmap='plasma')

# Contour lines at specific probability levels
contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
contour_levels = [l * prob.max() for l in contour_levels]
ax.contour(X, Y, prob, levels=contour_levels, colors='white', linewidths=1, alpha=0.7)

# Mark the center
ax.plot(x0, y0, 'w+', markersize=15, markeredgewidth=2, label=rf'$\alpha = {alpha.real} + {alpha.imag}i$')
ax.axhline(0, color='white', linestyle='--', alpha=0.5)
ax.axvline(0, color='white', linestyle='--', alpha=0.5)

# Labels
ax.set_xlabel(r'$\mathrm{Re}(\alpha)$ (position quadrature)', fontsize=12)
ax.set_ylabel(r'$\mathrm{Im}(\alpha)$ (momentum quadrature)', fontsize=12)
ax.set_title('Coherent State Phase Space Distribution', fontsize=14)

# Colorbar
cbar = plt.colorbar(cs, ax=ax)
cbar.set_label('Probability Density', fontsize=12)

ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('coherent_state_phase_space.pdf', bbox_inches='tight')
plt.show()
```

### Example 3: Multi-Well Potential Energy Surface

**Problem**: Visualize a double-well potential with energy level indicators.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Double well potential: V(x) = (x² - a²)²
def double_well(x, a=1, V0=1):
    return V0 * (x**2 - a**2)**2

x = np.linspace(-2.5, 2.5, 500)
V = double_well(x, a=1.2, V0=0.5)

# Energy levels (approximate)
E_levels = [0.05, 0.15, 0.6, 0.9, 1.2]

fig, ax = plt.subplots(figsize=(12, 7))

# Plot potential
ax.fill_between(x, V, alpha=0.3, color='blue')
ax.plot(x, V, 'b-', linewidth=2.5, label='V(x)')

# Plot energy levels with classical turning points
for i, E in enumerate(E_levels):
    # Find turning points where V(x) = E
    color = plt.cm.viridis(i / len(E_levels))

    # Draw energy level
    x_allowed = x[V <= E]
    if len(x_allowed) > 0:
        x_min, x_max = x_allowed.min(), x_allowed.max()
        ax.hlines(E, x_min, x_max, colors=color, linewidths=2,
                 label=f'$E_{i} = {E:.2f}$' if i < 3 else None)

        # Mark turning points
        ax.plot([x_min, x_max], [E, E], 'o', color=color, markersize=6)

# Annotations
ax.annotate('Left well', xy=(-1.2, 0.02), fontsize=12, ha='center')
ax.annotate('Right well', xy=(1.2, 0.02), fontsize=12, ha='center')
ax.annotate('Barrier', xy=(0, 0.5), fontsize=12, ha='center', va='bottom')

# Styling
ax.set_xlabel('Position x', fontsize=12)
ax.set_ylabel('Energy', fontsize=12)
ax.set_title('Double Well Potential with Energy Levels', fontsize=14)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.05, 1.5)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Mark barrier height
ax.annotate('', xy=(0, 0), xytext=(0, 0.5),
           arrowprops=dict(arrowstyle='<->', color='red'))
ax.text(0.15, 0.25, r'$V_b$', fontsize=12, color='red')

plt.tight_layout()
plt.savefig('double_well_potential.pdf', bbox_inches='tight')
plt.show()
```

---

## Practice Problems

### Level 1: Direct Application

1. **Error Bars**: Create a plot of 10 data points with y-errors of 10%. Add a linear fit and display the fit parameters in a legend.

2. **Log Scale**: Generate an exponential function $$y = 100 e^{-x/3}$$ for $$x \in [0, 15]$$ and plot it on both linear and semi-log scales.

3. **Colormap**: Create a 2D Gaussian centered at (1, -1) with σ=0.5 and display it using the 'plasma' colormap.

### Level 2: Intermediate

4. **Energy Diagram with Transitions**: Create an energy level diagram for a 4-level system. Draw arrows between levels to indicate allowed transitions (Δn = ±1). Color the arrows by transition frequency.

5. **Uncertainty Visualization**: For a Gaussian wave packet, create a figure showing:
   - Position probability density with ⟨x⟩ marked
   - Shaded region indicating ±σ_x
   - Numerical value of ⟨x⟩ and σ_x displayed

6. **2D Wave Function**: Plot the 2D harmonic oscillator state ψ_11(x,y) = xy·exp(-(x²+y²)/2) showing both the wave function (with node lines) and probability density.

### Level 3: Challenging

7. **Publication Figure**: Create a 4-panel figure showing a particle in a box:
   - Panel A: First 5 energy levels
   - Panel B: Ground state probability density
   - Panel C: n=5 state with nodes marked
   - Panel D: Comparison of classical vs quantum probability

8. **Density Matrix Evolution**: Visualize the time evolution of a qubit density matrix starting from |+⟩ under dephasing. Show 5 time snapshots as heatmaps in a single figure.

9. **Wigner Function**: Compute and visualize the Wigner quasi-probability distribution for a coherent state, including the characteristic negative regions for a cat state.

---

## Computational Lab

### Project: Complete Scientific Plotting Toolkit

```python
"""
Scientific Plotting Toolkit for Quantum Physics
Day 268: Advanced Visualization Techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.gridspec import GridSpec
from scipy.stats import poisson


class QuantumPlotter:
    """
    Professional plotting utilities for quantum physics.

    Includes methods for error visualization, energy diagrams,
    density matrices, and publication-quality formatting.
    """

    # Publication-quality settings
    PUBLICATION_STYLE = {
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
    }

    def __init__(self, use_publication_style=False):
        """Initialize plotter with optional publication styling."""
        if use_publication_style:
            plt.rcParams.update(self.PUBLICATION_STYLE)
            # Enable LaTeX
            try:
                plt.rcParams['text.usetex'] = True
                plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
            except:
                pass  # LaTeX not available

    def plot_with_errors(self, x, y, y_err, fit_func=None, ax=None,
                        data_label='Data', fit_label='Fit'):
        """
        Plot data with error bars and optional fit curve.

        Parameters
        ----------
        x, y : arrays
            Data points
        y_err : array or tuple
            Symmetric or asymmetric errors
        fit_func : callable, optional
            Function to evaluate fit curve
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Plot data with errors
        ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=4, capthick=1.5,
                   markersize=6, label=data_label, color='C0')

        # Plot fit if provided
        if fit_func is not None:
            x_fine = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_fine, fit_func(x_fine), '-', color='C1',
                   linewidth=2, label=fit_label)

        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig, ax

    def plot_energy_levels(self, systems_dict, ax=None):
        """
        Compare energy level diagrams of multiple quantum systems.

        Parameters
        ----------
        systems_dict : dict
            {system_name: (energies, labels)}
        """
        n_systems = len(systems_dict)
        if ax is None:
            fig, ax = plt.subplots(figsize=(4*n_systems, 8))
        else:
            fig = ax.figure

        width = 0.6
        positions = np.arange(n_systems)

        for pos, (name, (energies, labels)) in zip(positions, systems_dict.items()):
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(energies)))

            for E, label, color in zip(energies, labels, colors):
                ax.hlines(E, pos - width/2, pos + width/2, color=color, linewidth=2.5)
                ax.text(pos + width/2 + 0.05, E, label, va='center', fontsize=9)

        ax.set_xticks(positions)
        ax.set_xticklabels(systems_dict.keys(), fontsize=11)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_xlim(-0.5, n_systems - 0.5 + 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return fig, ax

    def plot_density_matrix(self, rho, basis_labels=None, ax=None, cmap='viridis'):
        """Plot quantum density matrix with magnitude."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        n = rho.shape[0]
        im = ax.imshow(np.abs(rho), cmap=cmap, aspect='equal')

        # Add value annotations
        for i in range(n):
            for j in range(n):
                val = rho[i, j]
                if np.abs(val) > 0.01:
                    color = 'white' if np.abs(val) > 0.5 * np.abs(rho).max() else 'black'
                    ax.text(j, i, f'{np.abs(val):.2f}', ha='center', va='center',
                           color=color, fontsize=10)

        if basis_labels is None:
            basis_labels = [f'{i}' for i in range(n)]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(basis_labels)
        ax.set_yticklabels(basis_labels)

        plt.colorbar(im, ax=ax, label=r'$|\rho_{ij}|$')
        return fig, ax

    def plot_2d_wavefunction(self, X, Y, psi, ax=None, levels=20, show_nodal=True):
        """
        Plot 2D wave function with filled contours.

        Shows wave function with optional nodal lines marked.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Determine if wave function has sign changes
        has_negative = psi.min() < 0

        if has_negative:
            # Use diverging colormap for signed wave function
            vmax = np.abs(psi).max()
            cs = ax.contourf(X, Y, psi, levels=levels, cmap='RdBu',
                           vmin=-vmax, vmax=vmax)
            if show_nodal:
                ax.contour(X, Y, psi, levels=[0], colors='black',
                          linewidths=2, linestyles='-')
        else:
            # Use sequential colormap for |ψ|²
            cs = ax.contourf(X, Y, psi, levels=levels, cmap='viridis')

        ax.set_aspect('equal')
        plt.colorbar(cs, ax=ax)
        return fig, ax

    def create_measurement_figure(self, probs_theory, counts_experiment, labels):
        """
        Create publication-quality measurement comparison figure.
        """
        n_states = len(probs_theory)
        n_total = sum(counts_experiment)

        probs_exp = np.array(counts_experiment) / n_total
        probs_err = np.sqrt(probs_exp * (1 - probs_exp) / n_total)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(n_states)
        width = 0.35

        # Theory bars
        bars1 = ax.bar(x - width/2, probs_theory, width, label='Theory',
                       color='steelblue', alpha=0.8)

        # Experiment bars with error bars
        bars2 = ax.bar(x + width/2, probs_exp, width, yerr=probs_err,
                       label=f'Experiment (N={n_total})', color='coral',
                       alpha=0.8, capsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Quantum Measurement Statistics', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_ylim(0, max(max(probs_theory), max(probs_exp)) * 1.2)

        return fig, ax


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Scientific Plotting Toolkit for Quantum Physics")
    print("Day 268: Advanced Visualization")
    print("=" * 60)

    plotter = QuantumPlotter(use_publication_style=False)

    # Example 1: Error bar plot with exponential fit
    print("\n1. Decay measurement with uncertainties...")
    np.random.seed(42)
    t = np.linspace(0, 10, 15)
    tau_true = 2.5
    N0 = 500
    counts = np.random.poisson(N0 * np.exp(-t / tau_true))
    counts_err = np.sqrt(counts + 1)

    from scipy.optimize import curve_fit
    def exp_decay(t, N0, tau):
        return N0 * np.exp(-t / tau)

    popt, _ = curve_fit(exp_decay, t, counts, p0=[500, 2], sigma=counts_err)

    fig, ax = plotter.plot_with_errors(
        t, counts, counts_err,
        fit_func=lambda x: exp_decay(x, *popt),
        data_label='Counts',
        fit_label=f'Fit: τ = {popt[1]:.2f}'
    )
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Counts')
    ax.set_title('Exponential Decay Measurement')
    ax.set_yscale('log')
    plt.savefig('error_bar_plot.png', dpi=150, bbox_inches='tight')
    print("   Saved: error_bar_plot.png")

    # Example 2: Energy level comparison
    print("\n2. Comparing energy level structures...")
    systems = {
        'Harmonic\nOscillator': (
            [n + 0.5 for n in range(5)],
            [f'n={n}' for n in range(5)]
        ),
        'Infinite\nWell': (
            [n**2 for n in range(1, 6)],
            [f'n={n}' for n in range(1, 6)]
        ),
        'Hydrogen\nAtom': (
            [-13.6/n**2 for n in range(1, 6)],
            [f'n={n}' for n in range(1, 6)]
        )
    }
    fig, ax = plotter.plot_energy_levels(systems)
    ax.set_title('Quantum System Energy Level Comparison')
    plt.tight_layout()
    plt.savefig('energy_levels.png', dpi=150, bbox_inches='tight')
    print("   Saved: energy_levels.png")

    # Example 3: 2D wave function
    print("\n3. 2D quantum harmonic oscillator state...")
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)

    # ψ₁₁ = x*y*exp(-(x²+y²)/2)
    psi_11 = X * Y * np.exp(-(X**2 + Y**2)/2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plotter.plot_2d_wavefunction(X, Y, psi_11, ax=axes[0], show_nodal=True)
    axes[0].set_title(r'Wave Function $\psi_{11}(x,y)$')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    plotter.plot_2d_wavefunction(X, Y, np.abs(psi_11)**2, ax=axes[1], show_nodal=False)
    axes[1].set_title(r'Probability $|\psi_{11}|^2$')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    plt.tight_layout()
    plt.savefig('2d_wavefunction.png', dpi=150, bbox_inches='tight')
    print("   Saved: 2d_wavefunction.png")

    # Example 4: Density matrix
    print("\n4. Density matrix visualization...")
    # Coherent superposition: (|0⟩ + |1⟩ + |2⟩)/√3
    psi = np.array([1, 1, 1]) / np.sqrt(3)
    rho = np.outer(psi, psi.conj())

    labels = [r'$|0\rangle$', r'$|1\rangle$', r'$|2\rangle$']
    fig, ax = plotter.plot_density_matrix(rho, basis_labels=labels)
    ax.set_title('Coherent Superposition Density Matrix')
    plt.tight_layout()
    plt.savefig('density_matrix.png', dpi=150, bbox_inches='tight')
    print("   Saved: density_matrix.png")

    # Example 5: Measurement statistics
    print("\n5. Quantum measurement comparison...")
    probs = [0.2, 0.5, 0.3]
    np.random.seed(123)
    outcomes = np.random.choice(3, size=1000, p=probs)
    counts = [np.sum(outcomes == i) for i in range(3)]

    labels = [r'$|g\rangle$', r'$|e\rangle$', r'$|f\rangle$']
    fig, ax = plotter.create_measurement_figure(probs, counts, labels)
    plt.tight_layout()
    plt.savefig('measurement_stats.png', dpi=150, bbox_inches='tight')
    print("   Saved: measurement_stats.png")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  Figures generated: 5")
    print("  Visualization types covered:")
    print("    - Error bars with fits")
    print("    - Energy level diagrams")
    print("    - 2D wave function contours")
    print("    - Density matrix heatmaps")
    print("    - Measurement statistics")
    print("=" * 60)

    plt.show()
```

---

## Summary

### Key Concepts

| Concept | Application |
|---------|-------------|
| Error bars | Display measurement uncertainty (statistical, systematic) |
| Log scales | Reveal power laws (log-log) and exponentials (semi-log) |
| Symlog | Handle data spanning positive and negative over decades |
| Colormaps | Map scalar values to colors; choose appropriately for data type |
| Contour plots | Show level curves of 2D functions (wave functions, potentials) |
| Heatmaps | Visualize matrices (density matrices, correlation) |

### Key Formulas

$$\boxed{\text{Poisson uncertainty: } \sigma_N = \sqrt{N}}$$

$$\boxed{\text{Binomial uncertainty: } \sigma_p = \sqrt{\frac{p(1-p)}{N}}}$$

$$\boxed{\text{Power law: } y = ax^n \Rightarrow \log y = \log a + n \log x}$$

### Colormap Selection Guide

| Data Type | Recommended | Avoid |
|-----------|-------------|-------|
| Sequential positive | viridis, plasma, cividis | jet, rainbow |
| Diverging (±) | RdBu, coolwarm, seismic | Hot, cool |
| Cyclic (phase) | hsv, twilight | Linear colormaps |
| Categorical | tab10, Set1 | Continuous |

---

## Daily Checklist

- [ ] Created plots with symmetric and asymmetric error bars
- [ ] Used log and semi-log scales appropriately
- [ ] Applied different colormaps and normalizations
- [ ] Generated contour plots with labeled contours
- [ ] Created heatmaps for matrix visualization
- [ ] Built energy level diagrams
- [ ] Completed computational lab exercises

---

## Preview of Day 269

Tomorrow we explore **3D Visualization** with Matplotlib:
- Surface plots for potential energy landscapes
- Wire frames and contour plots in 3D
- 3D wave function probability densities
- Orbital visualization for hydrogen atom
- Interactive 3D rotation

These techniques enable visualization of the full spatial structure of quantum systems.
