# Week 247: Expanded Analysis

## Days 1723-1729 | Systematic Parameter Space Exploration

---

## Overview

Week 247 expands the investigation from initial data collection to comprehensive parameter space exploration. With baseline results established in Week 246, you now systematically vary parameters, perform sensitivity analysis, and build a complete understanding of your system's behavior across relevant conditions.

### Learning Objectives

By the end of this week, you will be able to:

1. Design and execute systematic parameter sweeps for quantum systems
2. Perform sensitivity analysis to identify critical parameters
3. Construct and interpret comparative analysis matrices
4. Identify trends, patterns, and phase transitions in parameter space
5. Apply statistical methods to accumulated datasets
6. Extract actionable insights to guide further investigation

---

## 1. Principles of Parameter Space Exploration

### 1.1 Why Explore Parameter Space?

**Beyond Single-Point Measurements**

Initial investigations provide snapshots at specific conditions. Parameter space exploration reveals:

- **Robustness**: How sensitive are results to parameter variations?
- **Optimality**: What parameter values maximize performance?
- **Universality**: Do behaviors persist across conditions?
- **Transitions**: Where do qualitative changes occur?
- **Limits**: What are the boundaries of operation?

### 1.2 The Parameter Landscape

Visualize your research as a landscape in parameter space:

$$
f: \mathbb{R}^n \rightarrow \mathbb{R}^m
$$

where $n$ is the number of input parameters and $m$ is the number of output metrics.

**Landscape Features**

- **Peaks**: Optimal performance regions
- **Valleys**: Poor performance or failure regions
- **Ridges**: Constrained optima
- **Saddles**: Transition points
- **Plateaus**: Insensitive regions
- **Cliffs**: Sharp transitions

### 1.3 Exploration Strategies

**Grid Sampling**

Systematic coverage with uniform spacing:
$$
\{(x_i, y_j) : x_i = x_{min} + i \cdot \Delta x, y_j = y_{min} + j \cdot \Delta y\}
$$

Pros: Complete coverage, easy visualization
Cons: Exponential scaling with dimensions ($N^d$ points)

**Random Sampling**

Uniform or weighted random selection:
$$
(x, y) \sim \mathcal{U}([x_{min}, x_{max}] \times [y_{min}, y_{max}])
$$

Pros: Scales better, unbiased
Cons: May miss features, uneven coverage

**Adaptive Sampling**

Focus resources on interesting regions:
$$
P(\mathbf{x}_{next}) \propto \sigma(\mathbf{x}) + \alpha \cdot |f(\mathbf{x})|
$$

Pros: Efficient, finds features
Cons: May miss isolated features

**Latin Hypercube Sampling (LHS)**

Stratified random sampling ensuring coverage:

```python
from scipy.stats import qmc

def latin_hypercube_sample(n_samples: int, n_dims: int, bounds: np.ndarray) -> np.ndarray:
    """
    Generate Latin Hypercube samples within bounds.

    Args:
        n_samples: Number of samples
        n_dims: Number of dimensions
        bounds: Array of shape (n_dims, 2) with [min, max] for each dimension

    Returns:
        Samples of shape (n_samples, n_dims)
    """
    sampler = qmc.LatinHypercube(d=n_dims)
    samples = sampler.random(n=n_samples)

    # Scale to bounds
    for i in range(n_dims):
        samples[:, i] = bounds[i, 0] + samples[:, i] * (bounds[i, 1] - bounds[i, 0])

    return samples
```

---

## 2. Designing Parameter Sweeps

### 2.1 Parameter Classification

**Primary Parameters**: Directly relevant to research question
- Must explore thoroughly
- Fine resolution near expected transitions
- Full range coverage

**Secondary Parameters**: Affect results but not central
- Coarse exploration to verify insensitivity
- Fix at typical values for primary sweeps

**Nuisance Parameters**: Should be held constant
- Document fixed values
- Verify independence if possible

### 2.2 Parameter Sweep Design

**Sweep Design Template**

```yaml
sweep_id: SWEEP_001
objective: "Characterize gate fidelity vs. pulse amplitude and duration"

parameters:
  primary:
    - name: pulse_amplitude
      type: continuous
      range: [0.1, 1.0]
      unit: "V"
      resolution: 0.05
      points: 19
      rationale: "Covers sublinear to saturation regime"

    - name: pulse_duration
      type: continuous
      range: [10, 100]
      unit: "ns"
      resolution: 5
      points: 19
      rationale: "From fast gates to decoherence-limited"

  secondary:
    - name: temperature
      fixed_value: 15
      unit: "mK"
      justification: "Nominal operating temperature"

  nuisance:
    - name: lab_temperature
      nominal: 21
      unit: "C"
      tolerance: 1

metrics:
  - name: gate_fidelity
    type: continuous
    range: [0.9, 1.0]
    precision: 0.001

  - name: gate_time
    type: continuous
    unit: "ns"

design:
  type: full_factorial
  total_points: 361  # 19 x 19
  repetitions: 5
  total_experiments: 1805
  estimated_time: "18 hours"

contingency:
  if_time_limited: "Use LHS with 100 samples"
  if_drift_detected: "Intersperse baseline measurements"
```

### 2.3 Quantum-Specific Considerations

**Calibration Drift**

Quantum systems drift over time. Interleave reference measurements:

```python
def interleaved_sweep(parameters: list, reference_interval: int = 10):
    """
    Execute parameter sweep with periodic reference measurements.

    Args:
        parameters: List of parameter sets to measure
        reference_interval: Number of measurements between references

    Yields:
        (parameter_set, is_reference) tuples
    """
    reference_params = get_baseline_parameters()

    for i, params in enumerate(parameters):
        # Regular measurement
        yield params, False

        # Periodic reference
        if (i + 1) % reference_interval == 0:
            yield reference_params, True
```

**Coherence Constraints**

Some parameter combinations may exceed coherence limits:

$$
T_{operation} < T_2 \cdot \ln(1/\epsilon)
$$

where $\epsilon$ is the acceptable decoherence error.

**State Preparation Fidelity**

Ensure state preparation is consistent across parameter space:
- Verify preparation fidelity at extremes
- Flag combinations where preparation degrades

---

## 3. Sensitivity Analysis

### 3.1 Types of Sensitivity Analysis

**Local Sensitivity**

Derivatives at a specific point:
$$
S_i = \frac{\partial f}{\partial x_i}\bigg|_{\mathbf{x}_0}
$$

**Normalized Sensitivity**
$$
S_i^* = \frac{x_i}{f} \cdot \frac{\partial f}{\partial x_i}
$$

**Global Sensitivity**

Variance decomposition over entire parameter space:
$$
V(Y) = \sum_i V_i + \sum_{i<j} V_{ij} + ... + V_{1,2,...,n}
$$

### 3.2 Sobol Sensitivity Indices

**First-Order Indices**

Fraction of variance due to parameter $x_i$ alone:
$$
S_i = \frac{V_i}{V(Y)} = \frac{V[E(Y|X_i)]}{V(Y)}
$$

**Total-Order Indices**

All variance involving parameter $x_i$:
$$
S_{Ti} = \frac{E[V(Y|X_{\sim i})]}{V(Y)}
$$

**Implementation**

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

def sobol_sensitivity_analysis(
    model_function,
    problem: dict,
    n_samples: int = 1024
) -> dict:
    """
    Perform Sobol sensitivity analysis.

    Args:
        model_function: Function that maps parameters to output
        problem: SALib problem definition
        n_samples: Base sample count (total = n_samples * (2D + 2))

    Returns:
        Dictionary with Sobol indices
    """
    # Generate samples
    param_values = saltelli.sample(problem, n_samples)

    # Evaluate model
    Y = np.array([model_function(X) for X in param_values])

    # Analyze
    Si = sobol.analyze(problem, Y)

    return {
        'S1': Si['S1'],           # First-order indices
        'S1_conf': Si['S1_conf'], # Confidence intervals
        'ST': Si['ST'],           # Total-order indices
        'ST_conf': Si['ST_conf'],
        'parameter_names': problem['names']
    }

# Example problem definition
problem = {
    'num_vars': 3,
    'names': ['pulse_amp', 'pulse_duration', 'detuning'],
    'bounds': [[0.1, 1.0], [10, 100], [-10, 10]]
}
```

### 3.3 Interpreting Sensitivity Results

| Scenario | Interpretation | Action |
|----------|---------------|--------|
| $S_i$ high, $S_{Ti}$ high | Parameter important, little interaction | Focus optimization here |
| $S_i$ low, $S_{Ti}$ high | Parameter important through interactions | Study interactions |
| $S_i$ ≈ $S_{Ti}$ | Purely additive effect | Simple response |
| $\sum S_i \ll 1$ | Strong interactions dominate | Need full exploration |

---

## 4. Comparative Analysis

### 4.1 Building Comparison Matrices

**Pairwise Comparison Matrix**

```python
def build_comparison_matrix(
    results: dict,
    metric: str,
    comparison: str = 'ratio'
) -> pd.DataFrame:
    """
    Build pairwise comparison matrix for different conditions.

    Args:
        results: Dict mapping condition name to metric values
        metric: Name of metric to compare
        comparison: 'ratio', 'difference', or 'percent_change'

    Returns:
        DataFrame with pairwise comparisons
    """
    conditions = list(results.keys())
    n = len(conditions)

    matrix = pd.DataFrame(index=conditions, columns=conditions)

    for i, cond_i in enumerate(conditions):
        for j, cond_j in enumerate(conditions):
            val_i = results[cond_i][metric]
            val_j = results[cond_j][metric]

            if comparison == 'ratio':
                matrix.loc[cond_i, cond_j] = val_i / val_j
            elif comparison == 'difference':
                matrix.loc[cond_i, cond_j] = val_i - val_j
            elif comparison == 'percent_change':
                matrix.loc[cond_i, cond_j] = 100 * (val_i - val_j) / val_j

    return matrix
```

### 4.2 Statistical Comparison

**Multiple Comparison Tests**

When comparing multiple conditions, control for multiple testing:

```python
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def statistical_comparison(
    groups: dict,
    alpha: float = 0.05
) -> dict:
    """
    Perform statistical comparison across multiple groups.

    Args:
        groups: Dict mapping group name to array of values
        alpha: Significance level

    Returns:
        Statistical comparison results
    """
    results = {}

    # ANOVA for overall effect
    f_stat, p_anova = stats.f_oneway(*groups.values())
    results['anova'] = {'F': f_stat, 'p': p_anova}

    # Tukey HSD for pairwise comparisons
    all_values = np.concatenate(list(groups.values()))
    group_labels = np.concatenate([
        [name] * len(values) for name, values in groups.items()
    ])

    tukey = pairwise_tukeyhsd(all_values, group_labels, alpha=alpha)
    results['tukey'] = tukey

    # Effect sizes (Cohen's d)
    names = list(groups.keys())
    results['effect_sizes'] = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            d = cohens_d(groups[names[i]], groups[names[j]])
            results['effect_sizes'][(names[i], names[j])] = d

    return results

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

### 4.3 Visualization for Comparison

**Heatmap Visualization**

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_parameter_sweep_heatmap(
    results: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    metric_label: str,
    save_path: str = None
):
    """
    Create heatmap visualization of 2D parameter sweep.

    Args:
        results: 2D array of metric values
        x_values: Array of x-axis parameter values
        y_values: Array of y-axis parameter values
        x_label, y_label, metric_label: Axis labels
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(
        results,
        origin='lower',
        aspect='auto',
        extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
        cmap='viridis'
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label, fontsize=12)

    # Labels
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{metric_label} vs {x_label} and {y_label}', fontsize=14)

    # Contour lines
    contours = ax.contour(
        results,
        levels=5,
        colors='white',
        alpha=0.5,
        extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]]
    )
    ax.clabel(contours, inline=True, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
```

---

## 5. Pattern Recognition and Trend Identification

### 5.1 Identifying Functional Relationships

**Regression Analysis**

```python
from scipy.optimize import curve_fit

def identify_functional_form(
    x_data: np.ndarray,
    y_data: np.ndarray,
    y_err: np.ndarray = None
) -> dict:
    """
    Test multiple functional forms and identify best fit.

    Returns:
        Best fit model and parameters
    """
    models = {
        'linear': lambda x, a, b: a * x + b,
        'quadratic': lambda x, a, b, c: a * x**2 + b * x + c,
        'exponential': lambda x, a, b, c: a * np.exp(-b * x) + c,
        'power': lambda x, a, b, c: a * x**b + c,
        'logarithmic': lambda x, a, b: a * np.log(x) + b
    }

    results = {}

    for name, func in models.items():
        try:
            # Fit model
            popt, pcov = curve_fit(
                func, x_data, y_data,
                sigma=y_err,
                maxfev=5000
            )

            # Calculate residuals
            y_pred = func(x_data, *popt)
            residuals = y_data - y_pred

            # Metrics
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - ss_res / ss_tot

            # AIC for model comparison
            n = len(y_data)
            k = len(popt)
            aic = n * np.log(ss_res / n) + 2 * k

            results[name] = {
                'parameters': popt,
                'covariance': pcov,
                'r_squared': r_squared,
                'aic': aic,
                'residuals': residuals
            }

        except Exception as e:
            results[name] = {'error': str(e)}

    # Find best model by AIC
    valid_models = {k: v for k, v in results.items() if 'aic' in v}
    if valid_models:
        best_model = min(valid_models, key=lambda k: valid_models[k]['aic'])
        results['best_model'] = best_model

    return results
```

### 5.2 Phase Transition Detection

In quantum systems, look for:

**First-Order Transitions**
- Discontinuous jumps in observables
- Hysteresis under parameter sweep
- Bimodal distributions at transition

**Second-Order (Continuous) Transitions**
- Power-law scaling near critical point: $|O| \sim |g - g_c|^\beta$
- Diverging susceptibility: $\chi \sim |g - g_c|^{-\gamma}$
- Correlation length divergence: $\xi \sim |g - g_c|^{-\nu}$

```python
def detect_transition(
    parameter: np.ndarray,
    observable: np.ndarray,
    observable_err: np.ndarray = None
) -> dict:
    """
    Detect and characterize phase transitions.

    Returns:
        Transition characteristics if found
    """
    # Compute derivative
    d_obs = np.gradient(observable, parameter)

    # Find maximum derivative (potential transition)
    peak_idx = np.argmax(np.abs(d_obs))
    peak_param = parameter[peak_idx]

    # Check if transition-like
    # Criteria: derivative significantly larger than mean
    mean_derivative = np.mean(np.abs(d_obs))
    peak_derivative = np.abs(d_obs[peak_idx])

    if peak_derivative > 3 * mean_derivative:
        # Potential transition found
        # Fit near-critical behavior

        # Select data near transition
        window = len(parameter) // 10
        idx_range = slice(max(0, peak_idx - window), min(len(parameter), peak_idx + window))

        near_critical = parameter[idx_range] - peak_param
        near_obs = observable[idx_range]

        # Attempt power-law fit
        try:
            # Only use one side of transition
            right = near_critical > 0
            if np.sum(right) > 3:
                popt, _ = curve_fit(
                    lambda x, a, beta: a * np.abs(x)**beta,
                    near_critical[right],
                    np.abs(near_obs[right] - observable[peak_idx])
                )
                critical_exponent = popt[1]
            else:
                critical_exponent = None
        except:
            critical_exponent = None

        return {
            'transition_found': True,
            'critical_parameter': peak_param,
            'derivative_ratio': peak_derivative / mean_derivative,
            'critical_exponent': critical_exponent
        }

    return {'transition_found': False}
```

### 5.3 Anomaly Detection in Parameter Space

```python
from sklearn.ensemble import IsolationForest

def detect_anomalies(
    parameters: np.ndarray,
    observables: np.ndarray,
    contamination: float = 0.05
) -> np.ndarray:
    """
    Detect anomalous points in parameter-observable space.

    Args:
        parameters: Parameter values (n_samples, n_params)
        observables: Observable values (n_samples, n_observables)
        contamination: Expected fraction of anomalies

    Returns:
        Boolean array indicating anomalies
    """
    # Combine parameters and observables
    X = np.hstack([parameters, observables])

    # Fit isolation forest
    clf = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    predictions = clf.fit_predict(X)

    # -1 indicates anomaly
    return predictions == -1
```

---

## 6. Statistical Analysis of Accumulated Data

### 6.1 Aggregating Multiple Measurements

```python
def aggregate_measurements(
    measurements: List[dict],
    groupby: str,
    metric: str
) -> pd.DataFrame:
    """
    Aggregate measurements by parameter value.

    Args:
        measurements: List of measurement dictionaries
        groupby: Parameter to group by
        metric: Metric to aggregate

    Returns:
        DataFrame with aggregated statistics
    """
    df = pd.DataFrame(measurements)

    aggregated = df.groupby(groupby)[metric].agg([
        'count',
        'mean',
        'std',
        'sem',
        ('ci_95_low', lambda x: x.mean() - 1.96 * x.sem()),
        ('ci_95_high', lambda x: x.mean() + 1.96 * x.sem())
    ])

    return aggregated
```

### 6.2 Uncertainty Propagation in Parameter Space

```python
def propagate_uncertainty_interpolation(
    known_params: np.ndarray,
    known_values: np.ndarray,
    known_errors: np.ndarray,
    query_params: np.ndarray,
    method: str = 'gaussian_process'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate uncertainty when interpolating in parameter space.

    Returns:
        (predicted_values, predicted_uncertainties)
    """
    if method == 'gaussian_process':
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=known_errors**2,
            normalize_y=True
        )

        gp.fit(known_params, known_values)
        pred_mean, pred_std = gp.predict(query_params, return_std=True)

        return pred_mean, pred_std

    else:
        raise ValueError(f"Unknown method: {method}")
```

### 6.3 Model Selection for Parameter Dependence

```python
def model_selection_bic(
    parameter: np.ndarray,
    observable: np.ndarray,
    max_polynomial_order: int = 5
) -> dict:
    """
    Select best polynomial model using Bayesian Information Criterion.

    Returns:
        Best model order and fit statistics
    """
    n = len(parameter)
    results = {}

    for order in range(1, max_polynomial_order + 1):
        # Fit polynomial
        coeffs = np.polyfit(parameter, observable, order)
        pred = np.polyval(coeffs, parameter)

        # Compute BIC
        residuals = observable - pred
        ss_res = np.sum(residuals**2)
        sigma2 = ss_res / n

        k = order + 1  # Number of parameters
        bic = n * np.log(sigma2) + k * np.log(n)

        results[order] = {
            'coefficients': coeffs,
            'bic': bic,
            'ss_residual': ss_res,
            'r_squared': 1 - ss_res / np.sum((observable - np.mean(observable))**2)
        }

    # Best model
    best_order = min(results, key=lambda k: results[k]['bic'])
    results['best_order'] = best_order

    return results
```

---

## 7. Practical Implementation

### 7.1 Day-by-Day Schedule

**Day 1723 (Monday): Sweep Design**
- Review Week 246 results to inform sweep design
- Identify parameters for exploration
- Design sweep strategy (grid, LHS, adaptive)
- Set up execution infrastructure
- Begin first parameter sweep

**Day 1724 (Tuesday): Primary Sweeps**
- Execute main parameter sweeps
- Monitor for drift (interleave references)
- Real-time visualization of emerging patterns
- Document observations

**Day 1725 (Wednesday): Extended Sweeps**
- Continue parameter exploration
- Focus on regions of interest identified Tuesday
- Increase resolution near transitions
- Begin preliminary sensitivity analysis

**Day 1726 (Thursday): Sensitivity Analysis**
- Complete Sobol sensitivity analysis
- Identify dominant parameters
- Quantify interaction effects
- Document surprising sensitivities

**Day 1727 (Friday): Comparative Analysis**
- Build comparison matrices
- Statistical testing across conditions
- Identify significant differences
- Synthesize patterns across sweeps

**Day 1728 (Saturday): Integration and Visualization**
- Create comprehensive visualizations
- Integrate findings from all sweeps
- Identify gaps requiring additional data
- Prepare weekly summary

**Day 1729 (Sunday): Reflection and Planning**
- Complete weekly reflection
- Synthesize key findings
- Plan Week 248 focus areas
- Rest and recover

### 7.2 Common Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Too many parameters | Use sensitivity analysis to prioritize |
| Computational limits | Apply adaptive sampling |
| Noisy data | Increase repetitions at key points |
| Drift during sweeps | Interleave reference measurements |
| Unexpected transitions | Increase resolution locally |
| Conflicting results | Check for systematic errors |

---

## 8. Connection to Quantum Computing Research

### 8.1 Algorithm Performance Landscapes

For variational algorithms:

**Parameter Landscapes**
- Visualize cost function vs. variational parameters
- Identify barren plateaus: $\text{Var}(\partial_\theta E) \sim 2^{-n}$
- Find trainable regions

**Hyperparameter Sensitivity**
- Optimizer learning rate
- Ansatz depth
- Number of shots

### 8.2 Device Characterization Sweeps

**Cross-Talk Characterization**
- Sweep drive amplitude on neighbor qubits
- Measure target qubit response
- Build cross-talk matrix

**Frequency Sweeps**
- Characterize resonator/qubit spectra
- Identify avoided crossings
- Map energy levels

### 8.3 Error Rate Landscapes

**Gate Error vs. Parameters**
- Pulse amplitude, duration, shape
- Identify optimal operating points
- Map error budget contributions

---

## Summary

Week 247 transforms point measurements into comprehensive understanding. Key takeaways:

1. **Systematic exploration** reveals robustness and optimality
2. **Sensitivity analysis** identifies what matters
3. **Comparative analysis** quantifies differences
4. **Pattern recognition** extracts underlying physics
5. **Statistical rigor** ensures reliable conclusions

The goal is not just more data, but actionable insight about your system's behavior across the relevant parameter space.

---

## References

1. Saltelli, A. et al. "Global Sensitivity Analysis: The Primer"
2. Montgomery, D. "Design and Analysis of Experiments"
3. Rasmussen & Williams, "Gaussian Processes for Machine Learning"
4. Quantum Process Tomography and Characterization

---

*Next Week: Week 248 assesses Month 1 progress and adjusts plans for Month 2.*
