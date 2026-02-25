# Analysis Methodologies for Quantum Research

## Overview

This resource provides comprehensive coverage of analysis methodologies relevant to quantum research, from basic statistical techniques to advanced machine learning approaches. These methods enable extraction of meaningful insights from experimental and computational data.

---

## 1. Foundations of Quantitative Analysis

### 1.1 The Analysis Pipeline

```
Raw Data → Preprocessing → Exploratory Analysis → Formal Analysis → Interpretation
    ↓           ↓               ↓                     ↓               ↓
 Validate    Clean/         Visualize/            Statistical      Conclusions
 & Archive   Transform      Summarize             Inference         & Actions
```

### 1.2 Analysis Principles

**Objectivity**
- Pre-specify analysis plans where possible
- Apply consistent standards across conditions
- Report all results, not just favorable ones

**Transparency**
- Document all analysis steps
- Share code and data where possible
- Report uncertainty appropriately

**Rigor**
- Use appropriate statistical methods
- Validate assumptions
- Account for multiple comparisons

---

## 2. Descriptive Statistics

### 2.1 Measures of Central Tendency

**Mean**
$$\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$$

**Median**
The middle value when sorted; robust to outliers.

**Mode**
Most frequently occurring value; useful for discrete distributions.

### 2.2 Measures of Dispersion

**Variance**
$$s^2 = \frac{1}{N-1}\sum_{i=1}^{N}(x_i - \bar{x})^2$$

**Standard Deviation**
$$s = \sqrt{s^2}$$

**Standard Error of the Mean**
$$SE = \frac{s}{\sqrt{N}}$$

**Interquartile Range (IQR)**
$$IQR = Q_3 - Q_1$$

### 2.3 Distribution Shape

**Skewness**
$$\gamma_1 = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{x_i - \bar{x}}{s}\right)^3$$

**Kurtosis**
$$\gamma_2 = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{x_i - \bar{x}}{s}\right)^4 - 3$$

### 2.4 Implementation

```python
import numpy as np
from scipy import stats

def comprehensive_descriptive_stats(data: np.ndarray) -> dict:
    """
    Compute comprehensive descriptive statistics.

    Args:
        data: 1D array of observations

    Returns:
        Dictionary of descriptive statistics
    """
    return {
        # Central tendency
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': stats.mode(data, keepdims=False).mode,

        # Dispersion
        'std': np.std(data, ddof=1),
        'var': np.var(data, ddof=1),
        'sem': stats.sem(data),
        'iqr': stats.iqr(data),
        'range': np.ptp(data),
        'cv': np.std(data, ddof=1) / np.mean(data),  # Coefficient of variation

        # Shape
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),

        # Extremes
        'min': np.min(data),
        'max': np.max(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),

        # Size
        'n': len(data)
    }
```

---

## 3. Inferential Statistics

### 3.1 Hypothesis Testing Framework

**Null Hypothesis (H₀)**: Default assumption (usually no effect)
**Alternative Hypothesis (H₁)**: What we're testing for

**Type I Error (α)**: Rejecting H₀ when it's true (false positive)
**Type II Error (β)**: Failing to reject H₀ when it's false (false negative)
**Power (1-β)**: Probability of correctly rejecting false H₀

### 3.2 Common Statistical Tests

**Comparing Two Groups**

| Scenario | Parametric Test | Non-Parametric Alternative |
|----------|-----------------|---------------------------|
| Independent samples | t-test | Mann-Whitney U |
| Paired samples | Paired t-test | Wilcoxon signed-rank |
| Proportions | Z-test | Fisher's exact |

**Comparing Multiple Groups**

| Scenario | Parametric Test | Non-Parametric Alternative |
|----------|-----------------|---------------------------|
| Independent groups | ANOVA | Kruskal-Wallis |
| Repeated measures | Repeated measures ANOVA | Friedman |
| Two factors | Two-way ANOVA | -- |

### 3.3 Implementation

```python
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def compare_groups(
    groups: dict,
    test_type: str = 'auto',
    alpha: float = 0.05
) -> dict:
    """
    Compare multiple groups statistically.

    Args:
        groups: Dict mapping group name to array of values
        test_type: 'parametric', 'nonparametric', or 'auto'
        alpha: Significance level

    Returns:
        Comprehensive comparison results
    """
    results = {}

    # Check normality for auto-selection
    if test_type == 'auto':
        all_normal = True
        for name, values in groups.items():
            if len(values) >= 8:  # Need enough samples
                _, p = stats.shapiro(values)
                if p < 0.05:
                    all_normal = False
                    break
        test_type = 'parametric' if all_normal else 'nonparametric'

    results['test_type'] = test_type

    # Overall test
    values_list = list(groups.values())

    if test_type == 'parametric':
        f_stat, p_overall = stats.f_oneway(*values_list)
        results['overall_test'] = {
            'test': 'ANOVA',
            'F': f_stat,
            'p': p_overall
        }
    else:
        h_stat, p_overall = stats.kruskal(*values_list)
        results['overall_test'] = {
            'test': 'Kruskal-Wallis',
            'H': h_stat,
            'p': p_overall
        }

    # Pairwise comparisons if overall is significant
    if p_overall < alpha:
        all_values = np.concatenate(values_list)
        all_labels = np.concatenate([
            [name] * len(values) for name, values in groups.items()
        ])

        if test_type == 'parametric':
            tukey = pairwise_tukeyhsd(all_values, all_labels, alpha=alpha)
            results['pairwise'] = {
                'test': 'Tukey HSD',
                'results': tukey
            }
        else:
            # Manual pairwise Mann-Whitney with Bonferroni
            names = list(groups.keys())
            n_comparisons = len(names) * (len(names) - 1) // 2
            pairwise = {}

            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    u_stat, p_pair = stats.mannwhitneyu(
                        groups[names[i]],
                        groups[names[j]],
                        alternative='two-sided'
                    )
                    pairwise[f'{names[i]} vs {names[j]}'] = {
                        'U': u_stat,
                        'p': p_pair,
                        'p_adjusted': min(p_pair * n_comparisons, 1.0)
                    }

            results['pairwise'] = {
                'test': 'Mann-Whitney with Bonferroni',
                'results': pairwise
            }

    return results
```

### 3.4 Effect Sizes

**Cohen's d** (standardized mean difference)
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

where $s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$

**Eta-squared** (proportion of variance explained)
$$\eta^2 = \frac{SS_{between}}{SS_{total}}$$

**Omega-squared** (less biased)
$$\omega^2 = \frac{SS_{between} - df_{between} \cdot MS_{within}}{SS_{total} + MS_{within}}$$

```python
def effect_sizes(group1: np.ndarray, group2: np.ndarray) -> dict:
    """Calculate effect sizes for two-group comparison."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Cohen's d
    pooled_std = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / pooled_std

    # Glass's delta (using control group std)
    glass_delta = (m1 - m2) / np.sqrt(v2)

    # Hedges' g (bias-corrected d)
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    hedges_g = cohens_d * correction

    # Common language effect size
    # P(X1 > X2) for random samples
    from scipy.stats import norm
    cles = norm.cdf(cohens_d / np.sqrt(2))

    return {
        'cohens_d': cohens_d,
        'glass_delta': glass_delta,
        'hedges_g': hedges_g,
        'cles': cles,
        'interpretation': interpret_effect_size(abs(cohens_d))
    }

def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'
```

---

## 4. Regression and Curve Fitting

### 4.1 Linear Regression

**Simple Linear Regression**
$$y = \beta_0 + \beta_1 x + \epsilon$$

**Multiple Linear Regression**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \epsilon$$

### 4.2 Nonlinear Regression

Common models in quantum research:

**Exponential Decay**
$$y = A e^{-t/T} + B$$

**Oscillatory Decay** (Ramsey, Rabi)
$$y = A e^{-t/T_2} \cos(2\pi f t + \phi) + B$$

**Power Law**
$$y = A x^{\alpha} + B$$

### 4.3 Implementation

```python
from scipy.optimize import curve_fit
from scipy import stats as sp_stats

def fit_and_analyze(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray = None,
    model: str = 'linear'
) -> dict:
    """
    Fit data to specified model and compute statistics.

    Args:
        x: Independent variable
        y: Dependent variable
        y_err: Measurement uncertainties
        model: 'linear', 'exponential', 'power', 'oscillation'

    Returns:
        Fit results with statistics
    """
    models = {
        'linear': (lambda x, a, b: a * x + b, ['slope', 'intercept']),
        'exponential': (lambda x, A, T, B: A * np.exp(-x/T) + B, ['amplitude', 'decay_time', 'offset']),
        'power': (lambda x, A, alpha, B: A * x**alpha + B, ['amplitude', 'exponent', 'offset']),
        'oscillation': (
            lambda x, A, T, f, phi, B: A * np.exp(-x/T) * np.cos(2*np.pi*f*x + phi) + B,
            ['amplitude', 'decay_time', 'frequency', 'phase', 'offset']
        )
    }

    func, param_names = models[model]

    # Fit
    try:
        popt, pcov = curve_fit(
            func, x, y,
            sigma=y_err,
            absolute_sigma=True if y_err is not None else False,
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as e:
        return {'success': False, 'error': str(e)}

    # Predictions and residuals
    y_pred = func(x, *popt)
    residuals = y - y_pred

    # Statistics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot

    # Adjusted R-squared
    n, p = len(y), len(popt)
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    # Reduced chi-squared
    if y_err is not None:
        chi_squared = np.sum((residuals / y_err)**2)
        dof = n - p
        chi_squared_reduced = chi_squared / dof
    else:
        chi_squared_reduced = None

    # AIC and BIC
    aic = n * np.log(ss_res / n) + 2 * p
    bic = n * np.log(ss_res / n) + p * np.log(n)

    return {
        'success': True,
        'parameters': dict(zip(param_names, popt)),
        'uncertainties': dict(zip(param_names, perr)),
        'covariance': pcov,
        'r_squared': r_squared,
        'r_squared_adj': r_squared_adj,
        'chi_squared_reduced': chi_squared_reduced,
        'aic': aic,
        'bic': bic,
        'residuals': residuals,
        'predictions': y_pred
    }
```

---

## 5. Time Series Analysis

### 5.1 Stationarity and Trends

**Tests for Stationarity**
- Augmented Dickey-Fuller (ADF) test
- KPSS test

**Detrending Methods**
- Differencing
- Moving average subtraction
- Polynomial fitting

### 5.2 Spectral Analysis

**Power Spectral Density**
$$S(f) = \left|\int_{-\infty}^{\infty} x(t) e^{-2\pi i f t} dt\right|^2$$

```python
from scipy import signal

def spectral_analysis(data: np.ndarray, sampling_rate: float) -> dict:
    """
    Perform spectral analysis on time series.

    Args:
        data: Time series data
        sampling_rate: Sampling frequency in Hz

    Returns:
        Spectral analysis results
    """
    # Power spectral density (Welch's method)
    freqs, psd = signal.welch(data, fs=sampling_rate, nperseg=min(256, len(data)//4))

    # Find dominant frequency
    peak_idx = np.argmax(psd)
    dominant_freq = freqs[peak_idx]

    # Total power
    total_power = np.trapz(psd, freqs)

    return {
        'frequencies': freqs,
        'psd': psd,
        'dominant_frequency': dominant_freq,
        'total_power': total_power,
        'peak_power': psd[peak_idx]
    }
```

### 5.3 Correlation Analysis

**Autocorrelation**
$$R(\tau) = \frac{1}{(N-\tau)\sigma^2}\sum_{t=1}^{N-\tau}(x_t - \mu)(x_{t+\tau} - \mu)$$

**Cross-correlation**
$$R_{xy}(\tau) = \frac{1}{N\sigma_x\sigma_y}\sum_{t=1}^{N-\tau}(x_t - \mu_x)(y_{t+\tau} - \mu_y)$$

---

## 6. Quantum-Specific Analysis Methods

### 6.1 Quantum State Tomography Analysis

```python
def analyze_tomography_results(
    measurement_results: dict,
    n_qubits: int
) -> dict:
    """
    Analyze quantum state tomography results.

    Args:
        measurement_results: Dict mapping basis to counts
        n_qubits: Number of qubits

    Returns:
        Analysis including density matrix and quality metrics
    """
    from qiskit.ignis.verification import StateTomographyFitter

    # Reconstruct density matrix
    rho = reconstruct_density_matrix(measurement_results, n_qubits)

    # Compute quality metrics
    # Purity
    purity = np.real(np.trace(rho @ rho))

    # Von Neumann entropy
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter numerical zeros
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    # Fidelity to ideal (if known)
    # fidelity = compute_fidelity(rho, ideal_rho)

    return {
        'density_matrix': rho,
        'purity': purity,
        'entropy': entropy,
        'eigenvalues': np.linalg.eigvalsh(rho)
    }
```

### 6.2 Randomized Benchmarking Analysis

```python
def analyze_rb_data(
    sequence_lengths: np.ndarray,
    survival_probabilities: np.ndarray,
    survival_errors: np.ndarray = None
) -> dict:
    """
    Analyze randomized benchmarking data.

    Args:
        sequence_lengths: Array of Clifford sequence lengths
        survival_probabilities: Measured survival probabilities
        survival_errors: Uncertainties in probabilities

    Returns:
        RB analysis results including error per Clifford
    """
    # Model: p = A * p^m + B
    # where p = 1 - 2 * r (r = error per Clifford)

    def rb_model(m, A, p, B):
        return A * p**m + B

    # Fit
    popt, pcov = curve_fit(
        rb_model,
        sequence_lengths,
        survival_probabilities,
        p0=[0.5, 0.99, 0.5],
        sigma=survival_errors,
        bounds=([0, 0, 0], [1, 1, 1])
    )

    A, p, B = popt
    A_err, p_err, B_err = np.sqrt(np.diag(pcov))

    # Error per Clifford
    r = (1 - p) / 2  # For single qubit
    r_err = p_err / 2

    return {
        'amplitude': (A, A_err),
        'depolarizing_parameter': (p, p_err),
        'offset': (B, B_err),
        'error_per_clifford': (r, r_err),
        'average_gate_fidelity': (1 - r, r_err)
    }
```

### 6.3 Process Tomography Analysis

```python
def analyze_process_tomography(chi_matrix: np.ndarray) -> dict:
    """
    Analyze quantum process from chi matrix.

    Args:
        chi_matrix: Process chi matrix

    Returns:
        Process analysis results
    """
    d = int(np.sqrt(chi_matrix.shape[0]))

    # Process fidelity
    # Assumes ideal process is identity
    process_fidelity = np.real(chi_matrix[0, 0])  # χ_II for identity

    # Unitarity
    # U(E) = (d/(d-1)) * (Tr(χ²) - 1/d)
    unitarity = (d / (d - 1)) * (np.real(np.trace(chi_matrix @ chi_matrix)) - 1/d)

    return {
        'chi_matrix': chi_matrix,
        'process_fidelity': process_fidelity,
        'unitarity': unitarity,
        'dimension': d
    }
```

---

## 7. Bootstrap and Resampling Methods

### 7.1 Bootstrap Confidence Intervals

```python
def bootstrap_ci(
    data: np.ndarray,
    statistic: callable,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95
) -> dict:
    """
    Compute bootstrap confidence interval for any statistic.

    Args:
        data: Original data
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level

    Returns:
        Bootstrap results including CI
    """
    n = len(data)
    boot_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats[i] = statistic(boot_sample)

    # Original statistic
    original = statistic(data)

    # Confidence interval (percentile method)
    alpha = 1 - ci_level
    ci_low = np.percentile(boot_stats, 100 * alpha / 2)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    # BCa method (bias-corrected and accelerated) is more accurate
    # but more complex to implement

    return {
        'statistic': original,
        'bootstrap_mean': np.mean(boot_stats),
        'bootstrap_std': np.std(boot_stats),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'ci_level': ci_level
    }
```

### 7.2 Permutation Tests

```python
def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic: callable = lambda x, y: np.mean(x) - np.mean(y),
    n_permutations: int = 10000
) -> dict:
    """
    Perform permutation test for two-group comparison.

    Args:
        group1, group2: Two groups to compare
        statistic: Test statistic function
        n_permutations: Number of permutations

    Returns:
        Permutation test results
    """
    # Observed statistic
    observed = statistic(group1, group2)

    # Combined data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    # Permutation distribution
    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(combined)
        perm_stats[i] = statistic(combined[:n1], combined[n1:])

    # p-value (two-sided)
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))

    return {
        'observed_statistic': observed,
        'p_value': p_value,
        'permutation_distribution': perm_stats
    }
```

---

## 8. Machine Learning for Analysis

### 8.1 Dimensionality Reduction

**Principal Component Analysis (PCA)**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_analysis(data: np.ndarray, n_components: int = None) -> dict:
    """
    Perform PCA on multivariate data.

    Args:
        data: Data matrix (samples x features)
        n_components: Number of components to keep

    Returns:
        PCA analysis results
    """
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data_scaled)

    return {
        'transformed_data': transformed,
        'components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    }
```

### 8.2 Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster_analysis(data: np.ndarray, max_clusters: int = 10) -> dict:
    """
    Perform cluster analysis with automatic cluster selection.

    Args:
        data: Data matrix
        max_clusters: Maximum clusters to try

    Returns:
        Clustering results
    """
    scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append({'k': k, 'silhouette': score})

    # Best k
    best_k = max(scores, key=lambda x: x['silhouette'])['k']

    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(data)

    return {
        'optimal_k': best_k,
        'labels': labels,
        'centroids': kmeans.cluster_centers_,
        'silhouette_scores': scores
    }
```

---

## 9. Visualization for Analysis

### 9.1 Essential Plots

**Distribution Plots**
- Histogram with KDE
- Box plot / Violin plot
- Q-Q plot (for normality)

**Relationship Plots**
- Scatter plot with regression line
- Correlation heatmap
- Pair plot

**Time Series Plots**
- Line plot with confidence band
- Autocorrelation plot
- Spectral density plot

### 9.2 Publication-Quality Figures

```python
import matplotlib.pyplot as plt

def create_publication_figure(
    data: dict,
    plot_type: str,
    save_path: str = None
) -> None:
    """
    Create publication-quality figure.

    Args:
        data: Data to plot
        plot_type: Type of plot
        save_path: Path to save figure
    """
    # Set publication style
    plt.style.use('seaborn-paper')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (3.5, 2.5),  # Single column width
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf'
    })

    # Create figure
    fig, ax = plt.subplots()

    # Plot based on type
    # [Implementation details...]

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
```

---

## 10. Best Practices Summary

### Analysis Workflow

1. **Explore** data visually before formal analysis
2. **Check** assumptions of statistical tests
3. **Document** all analysis decisions
4. **Report** effect sizes alongside p-values
5. **Visualize** results for interpretation
6. **Validate** findings with multiple methods

### Common Pitfalls to Avoid

| Pitfall | Solution |
|---------|----------|
| P-hacking | Pre-register analyses |
| Ignoring assumptions | Always check |
| Multiple comparisons | Apply corrections |
| Overfitting | Cross-validate |
| Cherry-picking | Report all results |
| Misinterpreting significance | Focus on effect sizes |

---

## Resources

### Statistical Software
- Python: scipy.stats, statsmodels, scikit-learn
- R: Comprehensive statistical ecosystem

### References
- Field, A. "Discovering Statistics"
- Gelman & Hill, "Data Analysis Using Regression"
- Nielsen & Chuang, "Quantum Computation" (for quantum metrics)

---

*Rigorous analysis transforms data into knowledge.*
