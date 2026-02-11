# Day 835: Below-Threshold Operation Analysis

## Week 120, Day 2 | Month 30: Surface Codes | Semester 2A: Quantum Error Correction

### Overview

Today we analyze Google's landmark demonstration of below-threshold surface code operation. The Willow processor achieved an error suppression factor of λ = 2.14 ± 0.02, meaning each increase in code distance by 2 reduces the logical error rate by more than half. This validates the fundamental promise of quantum error correction: scaling up truly suppresses errors exponentially.

---

## Daily Schedule

| Time Block | Duration | Activity |
|------------|----------|----------|
| **Morning** | 3 hours | Below-threshold theory and measurements |
| **Afternoon** | 2.5 hours | Statistical analysis and implications |
| **Evening** | 1.5 hours | Computational lab: Error rate analysis |

---

## Learning Objectives

By the end of today, you will be able to:

1. **Interpret the error suppression factor** - Calculate and explain λ = 2.14
2. **Analyze logical error rates** - Compare d=3, 5, 7 performance quantitatively
3. **Verify below-threshold operation** - Apply statistical tests for threshold crossing
4. **Evaluate logical qubit lifetime** - Compare logical vs. physical coherence
5. **Project scaling behavior** - Extrapolate to higher code distances
6. **Assess experimental methodology** - Understand memory experiment protocol

---

## Core Content

### 1. The Threshold Theorem Revisited

#### 1.1 Fundamental Scaling

The surface code threshold theorem predicts that below threshold, logical error rate scales as:

$$\boxed{p_L(d) \approx A \left(\frac{p}{p_{\text{th}}}\right)^{(d+1)/2}}$$

where:
- $p_L(d)$ = logical error rate at distance $d$
- $p$ = physical error rate per cycle
- $p_{\text{th}}$ = threshold error rate (~1% for surface codes)
- $A$ = constant prefactor
- $d$ = code distance (odd integer)

#### 1.2 Error Suppression Factor

The error suppression factor λ measures how much the logical error rate decreases when distance increases by 2:

$$\boxed{\lambda = \frac{p_L(d)}{p_L(d+2)}}$$

From the threshold scaling:
$$\lambda = \left(\frac{p}{p_{\text{th}}}\right)^{-1} = \frac{p_{\text{th}}}{p}$$

**Critical Insight:** Below threshold ($p < p_{\text{th}}$), we have $\lambda > 1$, meaning larger codes have lower error rates. Above threshold ($p > p_{\text{th}}$), we have $\lambda < 1$, meaning larger codes perform worse.

### 2. Willow Experimental Results

#### 2.1 Memory Experiment Protocol

Google's experiment measured logical qubit lifetime in a memory configuration:

1. **Initialize** logical $|0\rangle$ or $|+\rangle$ state
2. **Run** $N$ error correction cycles (syndrome extraction)
3. **Decode** syndromes using MWPM decoder
4. **Measure** logical qubit in Z or X basis
5. **Compare** to expected value

**Logical error detection:**
- Z memory: Prepare $|0_L\rangle$, measure $Z_L$, check for bit flip
- X memory: Prepare $|+_L\rangle$, measure $X_L$, check for phase flip

#### 2.2 Measured Logical Error Rates

After 25 error correction cycles (25 μs):

| Distance | Logical Error Rate | Uncertainty |
|----------|-------------------|-------------|
| d = 3 | 3.028% | ± 0.023% |
| d = 5 | 0.306% | ± 0.005% |
| d = 7 | 0.143% | ± 0.003% |

#### 2.3 Error Suppression Factor

$$\lambda_{3\rightarrow 5} = \frac{3.028\%}{0.306\%} = 9.9 \pm 0.2$$

$$\lambda_{5\rightarrow 7} = \frac{0.306\%}{0.143\%} = 2.14 \pm 0.04$$

The combined fit gives:

$$\boxed{\lambda = 2.14 \pm 0.02}$$

**Why is λ higher for d=3→5?**
The d=3 code has fewer qubits and higher edge effects. The asymptotic suppression factor λ is best measured from larger codes (d=5→7).

### 3. Extracting Physical Error Rate

#### 3.1 Inverting the Scaling Relation

From $\lambda = p_{\text{th}}/p$, we can extract:

$$p = \frac{p_{\text{th}}}{\lambda} = \frac{1\%}{2.14} \approx 0.47\%$$

This effective physical error rate is lower than the 2.5% per-qubit per-cycle error budget because:
1. Not all errors lead to logical failures
2. Many errors are correctable by the decoder
3. The effective threshold includes decoder efficiency

#### 3.2 Comparison with Numerical Simulations

Google validated their results against circuit-level noise simulations:

| Metric | Experiment | Simulation |
|--------|------------|------------|
| λ | 2.14 ± 0.02 | 2.10 ± 0.05 |
| d=7 error rate | 0.143% | 0.140% |

The close agreement confirms the accuracy of the noise model.

### 4. Logical Qubit Lifetime

#### 4.1 Definition

The logical qubit lifetime $T_{L}$ is defined as the time for logical error probability to reach $1/e$:

$$p_L(t) = 1 - e^{-t/T_L}$$

For small errors: $p_L \approx t/T_L$, giving:

$$T_L = \frac{T_{\text{cycle}}}{p_L(\text{per cycle})}$$

#### 4.2 Willow Results

For d=7 with $p_L = 0.143\%$ per 1 μs cycle:

$$T_L^{(d=7)} = \frac{1 \text{ μs}}{0.00143} = 700 \text{ μs}$$

**Comparison with physical qubits:**

| Qubit Type | Lifetime |
|------------|----------|
| Best physical qubit | $T_1 = 68$ μs |
| Median physical qubit | $T_1 = 68$ μs |
| d=7 logical qubit | $T_L = 700$ μs |

$$\boxed{\frac{T_L^{(d=7)}}{T_1^{\text{best}}} = \frac{700}{68} \approx 10.3}$$

The reported factor of 2.4× in the Nature paper uses a different metric (comparing to median physical error rate). Either way, **the logical qubit outlives any physical qubit**.

### 5. Statistical Analysis Methods

#### 5.1 Confidence Intervals

The uncertainty in λ comes from:

1. **Shot noise**: $\delta p_L \propto \sqrt{p_L(1-p_L)/N_{\text{shots}}}$
2. **Systematic errors**: Calibration drift, decoder imperfections
3. **Finite cycle number**: Edge effects in initialization/measurement

With $N_{\text{shots}} \approx 10^5$ shots per configuration:

$$\delta p_L^{(d=7)} = \sqrt{\frac{0.00143 \times 0.99857}{10^5}} \approx 0.00004 = 0.004\%$$

#### 5.2 Hypothesis Testing

**Null hypothesis:** The system is at threshold ($\lambda = 1$)

**Test statistic:**
$$z = \frac{\lambda - 1}{\sigma_\lambda} = \frac{2.14 - 1}{0.02} = 57$$

This corresponds to a p-value of essentially zero—the below-threshold result is statistically unambiguous.

### 6. Error Rate Per Cycle Analysis

#### 6.1 Extracting Per-Cycle Error Rate

The total logical error after $N$ cycles:

$$p_L^{\text{total}}(N) = 1 - (1 - p_L^{\text{cycle}})^N \approx N \cdot p_L^{\text{cycle}}$$

For d=7 after 25 cycles:
$$p_L^{\text{cycle}} = \frac{0.143\%}{25} = 0.00572\%$$

Wait—this seems too low. Let's reconsider.

**Correction:** The 0.143% is already the per-cycle equivalent error rate extracted from the exponential fit. The actual measurement shows:

$$p_L(N) = \frac{1 - e^{-N/N_L}}{2}$$

where $N_L$ is the characteristic number of cycles before logical error.

#### 6.2 Detailed Error Model

The error probability after N cycles follows:

$$\boxed{p_L(N) = \frac{1}{2}\left(1 - e^{-N \cdot \epsilon_L}\right)}$$

where $\epsilon_L$ is the logical error rate per cycle.

From the data:
- d=7: $\epsilon_L = 2.86 \times 10^{-3}$ per cycle → 0.143% per cycle when converted to failure probability at 25 cycles

### 7. Implications for Scaling

#### 7.1 Projections to Higher Distance

Using λ = 2.14:

| Distance | Projected Error Rate | Qubits Needed |
|----------|---------------------|---------------|
| d = 7 | 0.143% | 97 |
| d = 9 | 0.067% | 161 |
| d = 11 | 0.031% | 241 |
| d = 13 | 0.015% | 337 |
| d = 15 | 0.007% | 449 |
| d = 17 | 0.003% | 577 |
| d = 21 | 0.0007% | 881 |

#### 7.2 Target Error Rates for Algorithms

For useful quantum computation:
- **Variational algorithms**: $p_L \sim 10^{-3}$ (d ~ 7-9)
- **Shor's algorithm (2048-bit)**: $p_L \sim 10^{-10}$ (d ~ 25-30)
- **Quantum chemistry (industrial)**: $p_L \sim 10^{-8}$ (d ~ 17-21)

---

## Worked Examples

### Example 1: Calculating Error Suppression Factor

**Problem:** A quantum processor measures the following logical error rates: d=5 gives 0.45% and d=7 gives 0.19%. Calculate λ and determine if the system is below threshold.

**Solution:**

$$\lambda = \frac{p_L(d=5)}{p_L(d=7)} = \frac{0.45\%}{0.19\%} = \frac{0.0045}{0.0019} = 2.37$$

Since $\lambda > 1$, the system is operating **below threshold**.

The effective ratio $p/p_{\text{th}}$:
$$\frac{p}{p_{\text{th}}} = \frac{1}{\lambda} = \frac{1}{2.37} = 0.42$$

This means the physical error rate is 42% of the threshold value.

$$\boxed{\lambda = 2.37 \text{ (below threshold)}}$$

### Example 2: Logical Qubit Lifetime Calculation

**Problem:** A d=9 surface code has logical error rate 0.067% per 1.2 μs cycle. Calculate:
a) The logical qubit lifetime
b) The number of cycles before 1% total error probability

**Solution:**

a) **Logical lifetime:**
$$T_L = \frac{T_{\text{cycle}}}{p_L^{\text{cycle}}} = \frac{1.2 \text{ μs}}{0.00067} = 1790 \text{ μs} \approx 1.8 \text{ ms}$$

b) **Cycles to 1% error:**
Using $p_L^{\text{total}} \approx N \cdot p_L^{\text{cycle}}$:
$$0.01 = N \times 0.00067$$
$$N = \frac{0.01}{0.00067} = 15 \text{ cycles}$$

More precisely, using the exponential model:
$$p_L(N) = \frac{1}{2}(1 - e^{-N \cdot \epsilon_L})$$
$$0.01 = 0.5(1 - e^{-N \times 0.00134})$$
$$0.02 = 1 - e^{-N \times 0.00134}$$
$$e^{-N \times 0.00134} = 0.98$$
$$N = \frac{-\ln(0.98)}{0.00134} = \frac{0.0202}{0.00134} \approx 15 \text{ cycles}$$

$$\boxed{T_L = 1.8 \text{ ms}, \quad N_{1\%} = 15 \text{ cycles}}$$

### Example 3: Projecting to Target Error Rate

**Problem:** An algorithm requires logical error rate below $10^{-6}$. Given λ = 2.14 and $p_L(d=7) = 0.143\%$, what code distance is needed?

**Solution:**

From the scaling relation:
$$p_L(d) = p_L(d=7) \cdot \lambda^{-(d-7)/2}$$

We need:
$$10^{-6} = 0.00143 \cdot (2.14)^{-(d-7)/2}$$

$$(2.14)^{(d-7)/2} = \frac{0.00143}{10^{-6}} = 1430$$

$$\frac{d-7}{2} = \log_{2.14}(1430) = \frac{\ln(1430)}{\ln(2.14)} = \frac{7.27}{0.76} = 9.6$$

$$d - 7 = 19.2$$
$$d = 26.2$$

Rounding up to the next odd integer:
$$\boxed{d = 27}$$

This requires $27^2 + (27^2 - 1) = 729 + 728 = 1457$ qubits for one logical qubit.

---

## Practice Problems

### Direct Application

**Problem 1:** A system measures $p_L(d=5) = 0.5\%$ and $p_L(d=7) = 0.25\%$.
a) Calculate the error suppression factor λ
b) Is the system below threshold?
c) Estimate the effective $p/p_{\text{th}}$ ratio

**Problem 2:** The Willow d=7 code achieves 0.143% error per cycle with 1 μs cycles.
a) How many cycles can run before the logical error exceeds 1%?
b) What is the logical qubit lifetime?
c) How does this compare to a 68 μs physical $T_1$?

**Problem 3:** Convert the following to consistent units: A code has 0.03% error rate per 0.8 μs cycle. Express this as:
a) Errors per microsecond
b) Errors per millisecond
c) Logical qubit lifetime

### Intermediate

**Problem 4:** An experiment runs 10,000 shots at each code distance, finding:
- d=3: 312 logical errors
- d=5: 31 logical errors
- d=7: 14 logical errors

Calculate λ with uncertainty estimates using binomial statistics.

**Problem 5:** The error suppression factor from d=3→5 is typically larger than d=5→7. Explain why this happens and what it implies about the approach to asymptotic scaling.

**Problem 6:** A next-generation processor aims for λ = 3 by reducing physical error rates. If the current threshold is 1%, what physical error rate is needed? What 2-qubit gate fidelity does this require (assuming 4 gates per cycle dominate)?

### Challenging

**Problem 7:** Derive the relationship between λ and the number of logical qubits achievable:
$$N_{\text{logical}}(Q, p_L^{\text{target}}) = f(Q, \lambda, p_L^{(d=7)})$$
where Q is the total number of physical qubits available.

**Problem 8:** The crossing point where logical error equals physical error defines a "break-even" distance. For a physical qubit with 1% error rate per cycle and λ = 2.14 with the Willow baseline, find this distance.

---

## Computational Lab: Below-Threshold Analysis

```python
"""
Day 835 Computational Lab: Below-Threshold Operation Analysis
Analyzes Google Willow's landmark below-threshold results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# =============================================================================
# Part 1: Willow Experimental Data
# =============================================================================

# Measured logical error rates (per cycle equivalent)
distances = np.array([3, 5, 7])
error_rates = np.array([0.03028, 0.00306, 0.00143])  # As fractions
error_uncertainties = np.array([0.00023, 0.00005, 0.00003])

print("=" * 60)
print("WILLOW BELOW-THRESHOLD ANALYSIS")
print("=" * 60)
print("\nExperimental Data:")
print("-" * 40)
for d, p, dp in zip(distances, error_rates, error_uncertainties):
    print(f"d = {d}: p_L = {p*100:.3f}% ± {dp*100:.3f}%")

# =============================================================================
# Part 2: Error Suppression Factor Calculation
# =============================================================================

def calculate_lambda(p1, p2, dp1, dp2):
    """
    Calculate error suppression factor and its uncertainty.

    Parameters:
    -----------
    p1, p2 : float
        Error rates at distances d and d+2
    dp1, dp2 : float
        Uncertainties in error rates

    Returns:
    --------
    lam : float
        Error suppression factor
    dlam : float
        Uncertainty in lambda
    """
    lam = p1 / p2

    # Error propagation: d(p1/p2) = sqrt((dp1/p2)^2 + (p1*dp2/p2^2)^2)
    dlam = lam * np.sqrt((dp1/p1)**2 + (dp2/p2)**2)

    return lam, dlam

# Calculate lambda for each pair
lambda_3_5, dlambda_3_5 = calculate_lambda(error_rates[0], error_rates[1],
                                            error_uncertainties[0], error_uncertainties[1])
lambda_5_7, dlambda_5_7 = calculate_lambda(error_rates[1], error_rates[2],
                                            error_uncertainties[1], error_uncertainties[2])

print("\n" + "=" * 60)
print("ERROR SUPPRESSION FACTORS")
print("=" * 60)
print(f"λ (d=3→5): {lambda_3_5:.2f} ± {dlambda_3_5:.2f}")
print(f"λ (d=5→7): {lambda_5_7:.2f} ± {dlambda_5_7:.2f}")

# =============================================================================
# Part 3: Threshold Model Fitting
# =============================================================================

def threshold_model(d, A, p_over_pth):
    """
    Threshold scaling model for logical error rate.

    p_L = A * (p/p_th)^((d+1)/2)
    """
    return A * (p_over_pth) ** ((d + 1) / 2)

def log_threshold_model(d, log_A, log_p_ratio):
    """
    Log-linear version for fitting.
    log(p_L) = log(A) + ((d+1)/2) * log(p/p_th)
    """
    return log_A + ((d + 1) / 2) * log_p_ratio

# Fit in log space
log_errors = np.log(error_rates)
log_uncertainties = error_uncertainties / error_rates  # Relative errors

# Linear fit: log(p_L) = a + b * (d+1)/2
X = (distances + 1) / 2
popt, pcov = np.polyfit(X, log_errors, 1, cov=True)
slope, intercept = popt
slope_err, intercept_err = np.sqrt(np.diag(pcov))

# Extract parameters
log_p_ratio = slope
p_over_pth = np.exp(log_p_ratio)
lambda_fit = 1 / p_over_pth

print("\n" + "=" * 60)
print("THRESHOLD MODEL FIT")
print("=" * 60)
print(f"Fitted p/p_th: {p_over_pth:.4f}")
print(f"Fitted λ: {lambda_fit:.2f}")
print(f"Assuming p_th = 1%, effective p = {p_over_pth * 100:.2f}%")

# =============================================================================
# Part 4: Projections to Higher Distance
# =============================================================================

def project_error_rate(d_target, d_ref, p_ref, lambda_val):
    """
    Project error rate to a target distance.

    Parameters:
    -----------
    d_target : int
        Target code distance
    d_ref : int
        Reference code distance
    p_ref : float
        Error rate at reference distance
    lambda_val : float
        Error suppression factor

    Returns:
    --------
    p_target : float
        Projected error rate
    """
    n_steps = (d_target - d_ref) / 2
    return p_ref / (lambda_val ** n_steps)

# Project to higher distances
d_ref = 7
p_ref = error_rates[2]
lambda_val = lambda_5_7

print("\n" + "=" * 60)
print("PROJECTIONS TO HIGHER DISTANCE")
print("=" * 60)
print(f"Using λ = {lambda_val:.2f}, baseline d={d_ref}, p_L = {p_ref*100:.3f}%")
print("-" * 50)
print(f"{'Distance':>10} {'Error Rate':>15} {'Qubits Needed':>15}")
print("-" * 50)

projected_distances = np.arange(7, 26, 2)
projected_errors = []
qubit_counts = []

for d in projected_distances:
    p_proj = project_error_rate(d, d_ref, p_ref, lambda_val)
    n_qubits = 2 * d**2 - 1
    projected_errors.append(p_proj)
    qubit_counts.append(n_qubits)
    print(f"{d:>10} {p_proj*100:>14.4f}% {n_qubits:>15}")

# =============================================================================
# Part 5: Logical Qubit Lifetime
# =============================================================================

def logical_lifetime(p_cycle, T_cycle):
    """
    Calculate logical qubit lifetime.

    T_L = T_cycle / p_cycle
    """
    return T_cycle / p_cycle

T_cycle = 1.0  # μs

print("\n" + "=" * 60)
print("LOGICAL QUBIT LIFETIME")
print("=" * 60)
print(f"Cycle time: {T_cycle} μs")
print("-" * 50)

T1_physical = 68  # μs (best physical qubit)

for d, p in zip(distances, error_rates):
    T_L = logical_lifetime(p, T_cycle)
    ratio = T_L / T1_physical
    print(f"d = {d}: T_L = {T_L:.1f} μs = {T_L/1000:.2f} ms (×{ratio:.1f} vs T1)")

# =============================================================================
# Part 6: Statistical Significance
# =============================================================================

def below_threshold_significance(lambda_val, lambda_err):
    """
    Calculate statistical significance of below-threshold operation.
    """
    # Z-score for lambda > 1
    z_score = (lambda_val - 1) / lambda_err

    # Two-sided p-value (but we care about one-sided)
    from scipy.stats import norm
    p_value = 1 - norm.cdf(z_score)

    return z_score, p_value

z_5_7, p_val_5_7 = below_threshold_significance(lambda_5_7, dlambda_5_7)

print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE")
print("=" * 60)
print(f"λ (d=5→7) = {lambda_5_7:.2f} ± {dlambda_5_7:.2f}")
print(f"Z-score (λ > 1): {z_5_7:.1f}")
print(f"P-value: {p_val_5_7:.2e}")
print(f"Confidence level: {(1-p_val_5_7)*100:.10f}%")

# =============================================================================
# Part 7: Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Error rate vs distance (log scale)
ax1 = axes[0, 0]
ax1.errorbar(distances, error_rates * 100, yerr=error_uncertainties * 100,
             fmt='o', markersize=10, capsize=5, capthick=2, color='blue',
             label='Willow data')

# Fit line
d_fine = np.linspace(2.5, 25, 100)
fit_line = np.exp(intercept + slope * (d_fine + 1) / 2) * 100
ax1.plot(d_fine, fit_line, 'r--', linewidth=2, label=f'Fit: λ = {lambda_fit:.2f}')

# Projections
ax1.scatter(projected_distances, np.array(projected_errors) * 100,
            marker='s', s=80, facecolors='none', edgecolors='green',
            linewidths=2, label='Projections')

ax1.set_yscale('log')
ax1.set_xlabel('Code Distance d', fontsize=12)
ax1.set_ylabel('Logical Error Rate (%)', fontsize=12)
ax1.set_title('Logical Error Rate vs Code Distance', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(2, 26)

# Add horizontal lines for target error rates
ax1.axhline(y=0.1, color='orange', linestyle=':', alpha=0.7, label='0.1% target')
ax1.axhline(y=0.01, color='purple', linestyle=':', alpha=0.7, label='0.01% target')

# Plot 2: Lambda values
ax2 = axes[0, 1]
pairs = ['d=3→5', 'd=5→7']
lambdas = [lambda_3_5, lambda_5_7]
lambda_errs = [dlambda_3_5, dlambda_5_7]

bars = ax2.bar(pairs, lambdas, yerr=lambda_errs, capsize=10, color=['steelblue', 'coral'],
               edgecolor='black', linewidth=2)
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Threshold (λ=1)')
ax2.set_ylabel('Error Suppression Factor λ', fontsize=12)
ax2.set_title('Error Suppression Factor by Distance Pair', fontsize=14)
ax2.legend()
ax2.set_ylim(0, 12)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val, err in zip(bars, lambdas, lambda_errs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.3,
             f'{val:.2f}±{err:.2f}', ha='center', va='bottom', fontsize=11)

# Plot 3: Logical vs Physical Lifetime
ax3 = axes[1, 0]
T_L_values = [logical_lifetime(p, T_cycle) for p in error_rates]

x_pos = np.arange(len(distances) + 1)
lifetimes = T_L_values + [T1_physical]
labels = [f'd={d}' for d in distances] + ['Physical\n(best T1)']
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

bars = ax3.bar(x_pos, lifetimes, color=colors, edgecolor='black', linewidth=2)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels)
ax3.set_ylabel('Lifetime (μs)', fontsize=12)
ax3.set_title('Logical vs Physical Qubit Lifetime', fontsize=14)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, lifetimes):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
             f'{val:.0f} μs', ha='center', va='bottom', fontsize=10)

# Plot 4: Qubits needed vs target error rate
ax4 = axes[1, 1]
target_errors = np.logspace(-2, -7, 50)  # 1% to 10^-7
qubits_needed = []

for p_target in target_errors:
    # Find distance needed
    d_needed = 7
    while project_error_rate(d_needed, 7, error_rates[2], lambda_5_7) > p_target and d_needed < 101:
        d_needed += 2
    qubits = 2 * d_needed**2 - 1
    qubits_needed.append(qubits)

ax4.loglog(target_errors * 100, qubits_needed, 'b-', linewidth=2)
ax4.set_xlabel('Target Logical Error Rate (%)', fontsize=12)
ax4.set_ylabel('Physical Qubits Needed', fontsize=12)
ax4.set_title('Scaling: Qubits Required per Logical Qubit', fontsize=14)
ax4.grid(True, alpha=0.3, which='both')
ax4.invert_xaxis()

# Mark key points
key_targets = [1e-2, 1e-4, 1e-6]
for p_t in key_targets:
    d_needed = 7
    while project_error_rate(d_needed, 7, error_rates[2], lambda_5_7) > p_t and d_needed < 101:
        d_needed += 2
    q = 2 * d_needed**2 - 1
    ax4.scatter([p_t * 100], [q], s=100, c='red', zorder=5)
    ax4.annotate(f'd={d_needed}\n{q} qubits', (p_t * 100, q),
                textcoords="offset points", xytext=(10, 10), fontsize=9)

plt.tight_layout()
plt.savefig('day_835_below_threshold.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Visualization saved to: day_835_below_threshold.png")
print("=" * 60)

# =============================================================================
# Part 8: Target Distance Calculator
# =============================================================================

def find_distance_for_target(p_target, p_ref, d_ref, lambda_val):
    """
    Find the minimum code distance to achieve a target error rate.
    """
    d = d_ref
    while d < 201:  # Reasonable upper limit
        p_d = project_error_rate(d, d_ref, p_ref, lambda_val)
        if p_d <= p_target:
            return d, p_d
        d += 2
    return None, None

print("\n" + "=" * 60)
print("TARGET ERROR RATE CALCULATOR")
print("=" * 60)
targets = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
print(f"{'Target Error':>15} {'Distance Needed':>18} {'Qubits':>10}")
print("-" * 45)

for target in targets:
    d_needed, p_achieved = find_distance_for_target(target, error_rates[2], 7, lambda_5_7)
    if d_needed:
        n_qubits = 2 * d_needed**2 - 1
        print(f"{target:>15.0e} {d_needed:>18} {n_qubits:>10}")

# =============================================================================
# Part 9: Improvement Trajectory
# =============================================================================

print("\n" + "=" * 60)
print("IMPROVEMENT TRAJECTORY (History)")
print("=" * 60)

# Historical data (approximate)
years = [2019, 2021, 2023, 2024]
systems = ['Sycamore', 'Weber', 'Sycamore++', 'Willow']
two_q_errors = [0.0036, 0.0050, 0.0030, 0.0025]  # Best CZ errors
t1_values = [15, 25, 40, 68]  # μs

print(f"{'Year':>6} {'System':>12} {'CZ Error':>12} {'T1 (μs)':>10}")
print("-" * 45)
for y, s, e, t in zip(years, systems, two_q_errors, t1_values):
    print(f"{y:>6} {s:>12} {e*100:>11.2f}% {t:>10}")

# Project future improvements needed
print("\n" + "=" * 60)
print("FUTURE TARGETS FOR λ = 3")
print("=" * 60)
# For λ = 3, need p/p_th = 1/3 = 0.33
# If p_th = 1%, need p = 0.33%
target_p = 0.0033
print(f"Required effective error rate: {target_p*100:.2f}%")
print(f"Required 2Q gate error (if 4 gates dominate): {target_p/4 * 100:.3f}%")
print(f"This is approximately {0.25/0.08:.1f}× improvement over current Willow")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Threshold scaling | $p_L(d) \approx A(p/p_{\text{th}})^{(d+1)/2}$ |
| Error suppression factor | $\lambda = p_L(d)/p_L(d+2)$ |
| Below threshold condition | $\lambda > 1$ (equivalently $p < p_{\text{th}}$) |
| Logical lifetime | $T_L = T_{\text{cycle}}/p_L^{\text{cycle}}$ |
| Distance for target error | $d = d_{\text{ref}} + 2\lceil\log_\lambda(p_{\text{ref}}/p_{\text{target}})\rceil$ |

### Main Takeaways

1. **λ = 2.14 confirms below-threshold operation** - Error rates decrease exponentially with distance
2. **Willow achieves 0.143% logical error per cycle at d=7** - A 10× improvement over d=3
3. **Logical qubit lifetime exceeds physical** - 700 μs vs. 68 μs best physical
4. **Scaling to practical algorithms requires d~25-30** - About 1500 qubits per logical qubit
5. **Statistical significance is overwhelming** - Z-score > 50 rules out at-threshold operation

### Daily Checklist

- [ ] I can calculate and interpret the error suppression factor λ
- [ ] I understand why λ > 1 means below-threshold operation
- [ ] I can project logical error rates to higher code distances
- [ ] I can calculate logical qubit lifetime from per-cycle error rates
- [ ] I understand the statistical methods for confirming below-threshold operation
- [ ] I completed the computational lab and analyzed scaling projections

---

## Preview: Day 836

Tomorrow we examine IBM's heavy-hex architecture, a fundamentally different approach to implementing surface codes. The 3-connectivity constraint requires flag qubit protocols for fault tolerance, presenting both challenges and advantages compared to Google's 4-connectivity grid.

**Key topics:**
- Heavy-hex lattice geometry
- Flag qubit fault tolerance protocols
- IBM Heron and future processor designs
- Comparison with Google's approach
