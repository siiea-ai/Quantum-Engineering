# Data Analysis and Interpretation Guide

## Introduction

This guide provides comprehensive methodology for analyzing quantum engineering research data. It covers statistical analysis, visualization, physical interpretation, and documentation practices that enable reproducible, publication-quality research outputs.

---

## Part 1: Analysis Philosophy

### 1.1 Goals of Data Analysis

Data analysis serves multiple purposes:

1. **Characterization:** Understand what the data shows
2. **Quantification:** Measure effects precisely
3. **Validation:** Test hypotheses and predictions
4. **Discovery:** Identify unexpected phenomena
5. **Communication:** Present findings clearly

### 1.2 The Analysis Mindset

**Be Systematic:** Follow a structured analysis pipeline
**Be Skeptical:** Question results, especially surprising ones
**Be Thorough:** Don't cherry-pick; analyze everything
**Be Honest:** Report what you find, not what you want
**Be Clear:** Enable others to understand and reproduce

### 1.3 Reproducibility Requirements

Every analysis should be:

- **Documented:** Clear description of methods
- **Scripted:** Code for all computations
- **Versioned:** History of changes tracked
- **Portable:** Can run on different systems
- **Complete:** From raw data to final figures

---

## Part 2: Data Preparation

### 2.1 Data Import and Verification

```python
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

class DataLoader:
    """
    Robust data loading with verification.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.manifest = None

    def verify_integrity(self, expected_checksums=None):
        """
        Verify data file integrity.

        Parameters:
        -----------
        expected_checksums : dict, optional
            {filename: expected_sha256}

        Returns:
        --------
        dict with verification results
        """
        results = {}
        for filepath in self.data_dir.glob('**/*'):
            if filepath.is_file():
                with open(filepath, 'rb') as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                results[str(filepath)] = checksum

                if expected_checksums:
                    expected = expected_checksums.get(filepath.name)
                    if expected and checksum != expected:
                        print(f"WARNING: Checksum mismatch for {filepath}")

        return results

    def load_experiment(self, exp_id):
        """
        Load data and metadata for single experiment.
        """
        exp_dir = self.data_dir / exp_id

        # Load data
        data_file = exp_dir / 'data.npz'
        data = np.load(data_file)

        # Load metadata
        import yaml
        metadata_file = exp_dir / 'metadata.yaml'
        with open(metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)

        return {
            'data': {k: data[k] for k in data.files},
            'metadata': metadata
        }

    def load_all(self, filter_func=None):
        """
        Load all experiments, optionally filtered.

        Parameters:
        -----------
        filter_func : callable, optional
            Function(metadata) -> bool for filtering

        Returns:
        --------
        dict of {exp_id: experiment_data}
        """
        experiments = {}
        for exp_dir in self.data_dir.glob('EXP-*'):
            exp_id = exp_dir.name
            exp_data = self.load_experiment(exp_id)

            if filter_func is None or filter_func(exp_data['metadata']):
                experiments[exp_id] = exp_data

        return experiments
```

### 2.2 Data Cleaning

```python
def clean_dataset(data, metadata):
    """
    Clean and prepare dataset for analysis.

    Steps:
    1. Remove invalid entries
    2. Handle missing values
    3. Apply calibrations
    4. Flag anomalies
    """
    cleaned = data.copy()

    # Step 1: Remove invalid entries
    valid_mask = ~np.isnan(cleaned) & ~np.isinf(cleaned)
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        print(f"Removed {n_invalid} invalid entries")

    # Step 2: Handle missing values
    # Option A: Remove
    # Option B: Interpolate
    # Option C: Flag and keep

    # Step 3: Apply calibrations
    if 'calibration' in metadata:
        cal = metadata['calibration']
        cleaned = cal['slope'] * cleaned + cal['offset']

    # Step 4: Flag anomalies
    mean = np.nanmean(cleaned)
    std = np.nanstd(cleaned)
    anomaly_mask = np.abs(cleaned - mean) > 5 * std

    return {
        'data': cleaned,
        'valid_mask': valid_mask,
        'anomaly_mask': anomaly_mask,
        'n_invalid': n_invalid,
        'n_anomalies': np.sum(anomaly_mask)
    }
```

### 2.3 Data Aggregation

```python
def aggregate_experiments(experiments, group_by, agg_func='mean'):
    """
    Aggregate experiments by parameter values.

    Parameters:
    -----------
    experiments : dict
        {exp_id: experiment_data}
    group_by : str or list
        Parameter name(s) to group by
    agg_func : str or callable
        Aggregation function ('mean', 'median', 'std', or callable)

    Returns:
    --------
    DataFrame with aggregated results
    """
    if isinstance(group_by, str):
        group_by = [group_by]

    # Extract data for aggregation
    records = []
    for exp_id, exp in experiments.items():
        record = {
            'exp_id': exp_id,
            'result': np.mean(exp['data']['observable'])  # or appropriate extraction
        }
        for param in group_by:
            record[param] = exp['metadata']['parameters'][param]
        records.append(record)

    df = pd.DataFrame(records)

    # Aggregate
    if agg_func == 'mean':
        agg = df.groupby(group_by)['result'].agg(['mean', 'std', 'count'])
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])
    elif agg_func == 'median':
        agg = df.groupby(group_by)['result'].agg(['median', 'mad', 'count'])
    else:
        agg = df.groupby(group_by)['result'].agg(agg_func)

    return agg.reset_index()
```

---

## Part 3: Statistical Analysis

### 3.1 Descriptive Statistics

```python
import numpy as np
from scipy import stats

class DescriptiveAnalysis:
    """
    Comprehensive descriptive statistics.
    """

    def __init__(self, data):
        self.data = np.asarray(data).flatten()
        self.n = len(self.data)

    def central_tendency(self):
        """Measures of central tendency."""
        return {
            'mean': np.mean(self.data),
            'median': np.median(self.data),
            'mode': float(stats.mode(self.data, keepdims=True).mode[0]),
            'geometric_mean': stats.gmean(self.data[self.data > 0]) if np.all(self.data > 0) else None,
            'harmonic_mean': stats.hmean(self.data[self.data > 0]) if np.all(self.data > 0) else None,
            'trimmed_mean_10': stats.trim_mean(self.data, 0.1)
        }

    def dispersion(self):
        """Measures of dispersion."""
        return {
            'variance': np.var(self.data, ddof=1),
            'std': np.std(self.data, ddof=1),
            'sem': stats.sem(self.data),
            'range': np.ptp(self.data),
            'iqr': stats.iqr(self.data),
            'mad': stats.median_abs_deviation(self.data),
            'cv': np.std(self.data, ddof=1) / np.mean(self.data) if np.mean(self.data) != 0 else None
        }

    def shape(self):
        """Shape statistics."""
        return {
            'skewness': stats.skew(self.data),
            'kurtosis': stats.kurtosis(self.data),
            'normality_test': stats.normaltest(self.data) if self.n >= 20 else None
        }

    def percentiles(self, ps=[5, 10, 25, 50, 75, 90, 95]):
        """Percentile values."""
        return {f'p{p}': np.percentile(self.data, p) for p in ps}

    def full_summary(self):
        """Complete statistical summary."""
        return {
            'n': self.n,
            **self.central_tendency(),
            **self.dispersion(),
            **self.shape(),
            **self.percentiles()
        }
```

### 3.2 Hypothesis Testing

```python
from scipy import stats
from typing import Tuple, Dict, Any

class HypothesisTester:
    """
    Statistical hypothesis testing framework.
    """

    @staticmethod
    def one_sample_t(data, expected_mean, alpha=0.05):
        """
        One-sample t-test.

        H0: population mean = expected_mean
        H1: population mean != expected_mean
        """
        stat, pvalue = stats.ttest_1samp(data, expected_mean)

        return {
            'test': 'one-sample t-test',
            'statistic': stat,
            'pvalue': pvalue,
            'significant': pvalue < alpha,
            'alpha': alpha,
            'sample_mean': np.mean(data),
            'expected_mean': expected_mean,
            'effect_size': (np.mean(data) - expected_mean) / np.std(data, ddof=1)
        }

    @staticmethod
    def two_sample_t(data1, data2, equal_var=True, alpha=0.05):
        """
        Two-sample t-test.

        H0: mean(data1) = mean(data2)
        H1: mean(data1) != mean(data2)
        """
        stat, pvalue = stats.ttest_ind(data1, data2, equal_var=equal_var)

        # Effect size (Cohen's d)
        n1, n2 = len(data1), len(data2)
        pooled_std = np.sqrt(((n1-1)*np.var(data1, ddof=1) + (n2-1)*np.var(data2, ddof=1)) / (n1+n2-2))
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std

        return {
            'test': 'two-sample t-test',
            'statistic': stat,
            'pvalue': pvalue,
            'significant': pvalue < alpha,
            'alpha': alpha,
            'mean1': np.mean(data1),
            'mean2': np.mean(data2),
            'cohens_d': cohens_d,
            'equal_var_assumed': equal_var
        }

    @staticmethod
    def paired_t(data1, data2, alpha=0.05):
        """
        Paired t-test.

        H0: mean(data1 - data2) = 0
        H1: mean(data1 - data2) != 0
        """
        stat, pvalue = stats.ttest_rel(data1, data2)
        differences = np.array(data1) - np.array(data2)

        return {
            'test': 'paired t-test',
            'statistic': stat,
            'pvalue': pvalue,
            'significant': pvalue < alpha,
            'alpha': alpha,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences, ddof=1),
            'effect_size': np.mean(differences) / np.std(differences, ddof=1)
        }

    @staticmethod
    def anova_one_way(*groups, alpha=0.05):
        """
        One-way ANOVA.

        H0: all group means are equal
        H1: at least one group mean is different
        """
        stat, pvalue = stats.f_oneway(*groups)

        # Effect size (eta-squared)
        all_data = np.concatenate(groups)
        ss_total = np.sum((all_data - np.mean(all_data))**2)
        ss_between = sum(len(g) * (np.mean(g) - np.mean(all_data))**2 for g in groups)
        eta_squared = ss_between / ss_total

        return {
            'test': 'one-way ANOVA',
            'statistic': stat,
            'pvalue': pvalue,
            'significant': pvalue < alpha,
            'alpha': alpha,
            'n_groups': len(groups),
            'eta_squared': eta_squared
        }

    @staticmethod
    def chi_square(observed, expected=None, alpha=0.05):
        """
        Chi-square test.

        If expected is None, tests for uniform distribution.
        """
        if expected is None:
            expected = np.ones_like(observed) * np.mean(observed)

        stat, pvalue = stats.chisquare(observed, expected)

        return {
            'test': 'chi-square',
            'statistic': stat,
            'pvalue': pvalue,
            'significant': pvalue < alpha,
            'alpha': alpha,
            'dof': len(observed) - 1
        }
```

### 3.3 Correlation Analysis

```python
class CorrelationAnalysis:
    """
    Comprehensive correlation analysis.
    """

    @staticmethod
    def pearson(x, y):
        """Pearson correlation coefficient."""
        r, pvalue = stats.pearsonr(x, y)
        return {
            'method': 'pearson',
            'r': r,
            'r_squared': r**2,
            'pvalue': pvalue,
            'significant_05': pvalue < 0.05
        }

    @staticmethod
    def spearman(x, y):
        """Spearman rank correlation."""
        rho, pvalue = stats.spearmanr(x, y)
        return {
            'method': 'spearman',
            'rho': rho,
            'pvalue': pvalue,
            'significant_05': pvalue < 0.05
        }

    @staticmethod
    def kendall(x, y):
        """Kendall tau correlation."""
        tau, pvalue = stats.kendalltau(x, y)
        return {
            'method': 'kendall',
            'tau': tau,
            'pvalue': pvalue,
            'significant_05': pvalue < 0.05
        }

    @staticmethod
    def correlation_matrix(df, method='pearson'):
        """
        Compute correlation matrix for DataFrame.
        """
        if method == 'pearson':
            corr = df.corr(method='pearson')
        elif method == 'spearman':
            corr = df.corr(method='spearman')
        else:
            raise ValueError(f"Unknown method: {method}")

        return corr

    @staticmethod
    def partial_correlation(x, y, z):
        """
        Partial correlation of x and y controlling for z.

        r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
        """
        r_xy = stats.pearsonr(x, y)[0]
        r_xz = stats.pearsonr(x, z)[0]
        r_yz = stats.pearsonr(y, z)[0]

        numerator = r_xy - r_xz * r_yz
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

        if denominator == 0:
            return {'partial_r': np.nan, 'note': 'Perfect correlation with control'}

        partial_r = numerator / denominator

        return {
            'partial_r': partial_r,
            'r_xy': r_xy,
            'r_xz': r_xz,
            'r_yz': r_yz
        }
```

### 3.4 Regression Analysis

```python
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

class RegressionAnalysis:
    """
    Regression and curve fitting analysis.
    """

    @staticmethod
    def linear_regression(x, y, confidence=0.95):
        """
        Ordinary least squares linear regression.
        """
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Predictions
        y_pred = slope * x + intercept
        residuals = y - y_pred

        # Confidence intervals
        n = len(x)
        t_val = stats.t.ppf((1 + confidence) / 2, df=n-2)

        se_slope = std_err
        se_intercept = std_err * np.sqrt(np.sum(x**2) / n)

        ci_slope = (slope - t_val * se_slope, slope + t_val * se_slope)
        ci_intercept = (intercept - t_val * se_intercept, intercept + t_val * se_intercept)

        return {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'ci_slope': ci_slope,
            'ci_intercept': ci_intercept,
            'residuals': residuals,
            'rmse': np.sqrt(np.mean(residuals**2))
        }

    @staticmethod
    def polynomial_regression(x, y, degree):
        """
        Polynomial regression.
        """
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)

        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot

        # Adjusted R-squared
        n = len(y)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - degree - 1)

        return {
            'coefficients': coeffs,
            'degree': degree,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'rmse': np.sqrt(ss_res / n)
        }

    @staticmethod
    def nonlinear_fit(x, y, model_func, p0, bounds=(-np.inf, np.inf)):
        """
        Nonlinear least squares fit.

        Parameters:
        -----------
        x, y : array-like
            Data points
        model_func : callable
            Function f(x, *params) to fit
        p0 : array-like
            Initial parameter guess
        bounds : tuple
            Parameter bounds

        Returns:
        --------
        dict with fit results
        """
        try:
            popt, pcov = curve_fit(model_func, x, y, p0=p0, bounds=bounds,
                                   maxfev=10000)
            perr = np.sqrt(np.diag(pcov))

            y_pred = model_func(x, *popt)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - ss_res / ss_tot

            return {
                'parameters': popt,
                'uncertainties': perr,
                'covariance': pcov,
                'r_squared': r_squared,
                'rmse': np.sqrt(ss_res / len(y)),
                'converged': True
            }
        except RuntimeError as e:
            return {
                'converged': False,
                'error': str(e)
            }
```

---

## Part 4: Pattern Recognition

### 4.1 Trend Detection

```python
def detect_trends(x, y, window_sizes=[5, 10, 20]):
    """
    Detect trends using multiple methods.
    """
    results = {}

    # Linear trend
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(x, y)
    results['linear'] = {
        'slope': slope,
        'r_squared': r**2,
        'significant': p < 0.05
    }

    # Mann-Kendall trend test
    from scipy.stats import kendalltau
    tau, p_mk = kendalltau(np.arange(len(y)), y)
    results['mann_kendall'] = {
        'tau': tau,
        'pvalue': p_mk,
        'trend': 'increasing' if tau > 0 else 'decreasing',
        'significant': p_mk < 0.05
    }

    # Moving average trends
    for w in window_sizes:
        if len(y) >= w:
            ma = np.convolve(y, np.ones(w)/w, mode='valid')
            results[f'moving_avg_{w}'] = {
                'values': ma,
                'trend': np.mean(np.diff(ma))
            }

    return results
```

### 4.2 Scaling Analysis

```python
def analyze_scaling(x, y, log_scale=True):
    """
    Analyze scaling behavior (power law, exponential, etc.).
    """
    results = {}

    # Power law: y = A * x^alpha
    if log_scale and np.all(x > 0) and np.all(y > 0):
        log_x = np.log(x)
        log_y = np.log(y)
        slope, intercept, r, p, se = stats.linregress(log_x, log_y)
        results['power_law'] = {
            'exponent': slope,
            'amplitude': np.exp(intercept),
            'r_squared': r**2,
            'model': f'y = {np.exp(intercept):.4f} * x^{slope:.4f}'
        }

    # Exponential: y = A * exp(x/xi)
    if np.all(y > 0):
        log_y = np.log(y)
        slope, intercept, r, p, se = stats.linregress(x, log_y)
        results['exponential'] = {
            'rate': slope,
            'amplitude': np.exp(intercept),
            'characteristic_scale': -1/slope if slope != 0 else np.inf,
            'r_squared': r**2,
            'model': f'y = {np.exp(intercept):.4f} * exp({slope:.4f} * x)'
        }

    # Linear: y = m*x + b
    slope, intercept, r, p, se = stats.linregress(x, y)
    results['linear'] = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r**2,
        'model': f'y = {slope:.4f} * x + {intercept:.4f}'
    }

    # Determine best model
    r2_values = {k: v['r_squared'] for k, v in results.items()}
    best_model = max(r2_values, key=r2_values.get)
    results['best_model'] = best_model

    return results
```

### 4.3 Phase Transition Detection

```python
def detect_phase_transition(x, y, methods=['derivative', 'variance', 'cumulant']):
    """
    Detect phase transitions or crossovers.
    """
    results = {}

    # Derivative method: find maximum of |dy/dx|
    if 'derivative' in methods:
        dy = np.gradient(y, x)
        idx_max = np.argmax(np.abs(dy))
        results['derivative'] = {
            'transition_x': x[idx_max],
            'max_derivative': dy[idx_max]
        }

    # Variance method: find maximum variance in sliding window
    if 'variance' in methods:
        window = len(y) // 5
        variances = []
        positions = []
        for i in range(len(y) - window):
            variances.append(np.var(y[i:i+window]))
            positions.append(x[i + window//2])

        idx_max = np.argmax(variances)
        results['variance'] = {
            'transition_x': positions[idx_max],
            'max_variance': variances[idx_max]
        }

    # Binder cumulant (for order parameter like data)
    if 'cumulant' in methods:
        # Fourth-order cumulant
        mean_y2 = np.mean(y**2)
        mean_y4 = np.mean(y**4)
        binder = 1 - mean_y4 / (3 * mean_y2**2)
        results['binder_cumulant'] = {
            'value': binder,
            'interpretation': 'ordered' if binder > 0.5 else 'disordered'
        }

    return results
```

---

## Part 5: Physical Interpretation Framework

### 5.1 From Numbers to Physics

The interpretation process:

1. **Identify observable:** What physical quantity was measured?
2. **Extract parameters:** What values characterize the observation?
3. **Connect to theory:** What does theory predict?
4. **Compare:** Agreement or discrepancy?
5. **Explain:** What physical mechanism is responsible?
6. **Predict:** What would happen in new conditions?
7. **Validate:** How can we test this interpretation?

### 5.2 Interpretation Template

```markdown
## Physical Interpretation: [Observable/Finding]

### Observation
**What was measured:** [Description]
**Key result:** [Value with uncertainty]
**Conditions:** [Parameter values]

### Theoretical Context
**Relevant theory:** [Reference]
**Predicted behavior:** [Mathematical expression or description]
**Expected value:** [Theoretical prediction]

### Comparison
**Agreement:** [Quantitative comparison]
**Discrepancy:** [If any, magnitude and sign]

### Physical Mechanism
**Proposed explanation:** [Physical reasoning]
**Supporting evidence:** [From data or literature]
**Alternative explanations:** [Other possibilities]

### Predictions
**Testable prediction 1:** [If this interpretation is correct, then...]
**Testable prediction 2:** [Additional prediction]

### Validation Plan
**Experiment to verify:** [Description]
**Expected outcome:** [What would confirm/refute interpretation]
```

### 5.3 Quantum-Specific Interpretations

**For Coherence/Decoherence:**
- Connect T1, T2 to physical noise sources
- Compare with theoretical noise models
- Identify dominant decoherence channel

**For Entanglement:**
- Quantify entanglement (concurrence, negativity, entropy)
- Compare with theoretical maximum
- Identify entanglement generation mechanism

**For Gate Fidelity:**
- Decompose error sources
- Compare with error budgets
- Identify limiting factors

**For Quantum Dynamics:**
- Compare with Schrodinger evolution
- Identify deviations from unitary dynamics
- Connect to open system effects

---

## Part 6: Visualization Best Practices

### 6.1 Figure Design Principles

```python
import matplotlib.pyplot as plt
import numpy as np

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,

        # Figure
        'figure.figsize': (3.5, 2.5),  # Single column width
        'figure.dpi': 300,

        # Lines and markers
        'lines.linewidth': 1.5,
        'lines.markersize': 6,

        # Axes
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.3,

        # Saving
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })


def parameter_sweep_plot(x, y, yerr=None, xlabel='Parameter', ylabel='Observable',
                          title=None, theory_func=None, save_path=None):
    """
    Create publication-quality parameter sweep plot.
    """
    setup_publication_style()

    fig, ax = plt.subplots()

    # Data with error bars
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=3,
                   label='Experiment', color='C0')
    else:
        ax.plot(x, y, 'o', label='Experiment', color='C0')

    # Theory comparison
    if theory_func is not None:
        x_fine = np.linspace(x.min(), x.max(), 100)
        y_theory = theory_func(x_fine)
        ax.plot(x_fine, y_theory, '-', label='Theory', color='C1')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path)

    return fig, ax


def heatmap_plot(data, x_values, y_values, xlabel='X', ylabel='Y',
                 cbar_label='Value', title=None, save_path=None):
    """
    Create publication-quality heatmap.
    """
    setup_publication_style()

    fig, ax = plt.subplots()

    im = ax.imshow(data, aspect='auto', origin='lower',
                   extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
                   cmap='viridis')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if save_path:
        fig.savefig(save_path)

    return fig, ax
```

### 6.2 Color Accessibility

```python
# Colorblind-friendly palettes
COLORBLIND_PALETTE = [
    '#0077BB',  # Blue
    '#EE7733',  # Orange
    '#009988',  # Teal
    '#CC3311',  # Red
    '#33BBEE',  # Cyan
    '#EE3377',  # Magenta
    '#BBBBBB',  # Gray
]

def use_colorblind_palette():
    """Set colorblind-friendly color cycle."""
    import cycler
    plt.rcParams['axes.prop_cycle'] = cycler.cycler(color=COLORBLIND_PALETTE)
```

---

## Part 7: Uncertainty and Error Analysis

### 7.1 Uncertainty Propagation

```python
import numpy as np

def propagate_uncertainty(func, params, uncertainties, correlations=None, n_samples=10000):
    """
    Monte Carlo uncertainty propagation.

    Parameters:
    -----------
    func : callable
        Function of parameters
    params : array-like
        Central values of parameters
    uncertainties : array-like
        Standard uncertainties of parameters
    correlations : array-like, optional
        Correlation matrix (default: independent)
    n_samples : int
        Number of Monte Carlo samples

    Returns:
    --------
    dict with central value and uncertainty
    """
    params = np.asarray(params)
    uncertainties = np.asarray(uncertainties)

    # Build covariance matrix
    if correlations is None:
        cov = np.diag(uncertainties**2)
    else:
        cov = np.outer(uncertainties, uncertainties) * correlations

    # Sample from multivariate normal
    samples = np.random.multivariate_normal(params, cov, size=n_samples)

    # Evaluate function for each sample
    results = np.array([func(*s) for s in samples])

    return {
        'value': np.mean(results),
        'uncertainty': np.std(results),
        'median': np.median(results),
        'ci_68': np.percentile(results, [16, 84]),
        'ci_95': np.percentile(results, [2.5, 97.5])
    }


def linear_error_propagation(func, params, uncertainties, dx=1e-8):
    """
    Linear (first-order) error propagation using numerical derivatives.
    """
    from scipy.misc import derivative

    params = np.asarray(params)
    uncertainties = np.asarray(uncertainties)

    # Compute partial derivatives
    partials = []
    for i in range(len(params)):
        def partial_func(x):
            p = params.copy()
            p[i] = x
            return func(*p)
        partials.append(derivative(partial_func, params[i], dx=dx))

    partials = np.array(partials)

    # Propagate uncertainty
    variance = np.sum(partials**2 * uncertainties**2)

    return {
        'value': func(*params),
        'uncertainty': np.sqrt(variance),
        'partial_derivatives': partials
    }
```

### 7.2 Uncertainty Budget

```markdown
## Uncertainty Budget: [Measurement Name]

### Statistical Uncertainty

| Source | Magnitude | Method | Notes |
|--------|-----------|--------|-------|
| Measurement noise | | Standard deviation of repeats | |
| Fitting uncertainty | | Covariance matrix | |
| Sampling | | Bootstrap / Monte Carlo | |
| **Total statistical** | | Quadrature sum | |

### Systematic Uncertainty

| Source | Magnitude | Method | Notes |
|--------|-----------|--------|-------|
| Calibration | | Calibration uncertainty | |
| Model approximation | | Comparison with full model | |
| Environmental | | Control experiments | |
| Equipment | | Specifications | |
| **Total systematic** | | Quadrature sum | |

### Combined Uncertainty

| Component | Value |
|-----------|-------|
| Statistical | |
| Systematic | |
| **Combined (k=1)** | |
| **Expanded (k=2, 95%)** | |
```

---

## Part 8: Documentation and Reproducibility

### 8.1 Analysis Report Structure

```markdown
# Analysis Report: [Title]

## Executive Summary
[Key findings in 2-3 sentences]

## Data Overview
- Dataset: [Description]
- Date range: [When collected]
- N measurements: [Count]
- Parameter ranges: [Summary]

## Methods
### Data Preprocessing
[Cleaning, calibration, etc.]

### Statistical Analysis
[Tests applied, methods used]

### Model Fitting
[Models, fitting procedures]

## Results
### Finding 1: [Title]
[Description with figures and statistics]

### Finding 2: [Title]
[Description with figures and statistics]

## Interpretation
[Physical meaning of results]

## Conclusions
[Summary of main findings]

## Appendices
### A. Complete Statistical Results
### B. Additional Figures
### C. Code References
```

### 8.2 Reproducibility Checklist

```markdown
## Reproducibility Checklist

### Environment
- [ ] Python version recorded
- [ ] Package versions in requirements.txt
- [ ] Random seeds documented
- [ ] Hardware/OS noted (if relevant)

### Data
- [ ] Raw data archived
- [ ] Data transformations documented
- [ ] Intermediate files saved (or can be regenerated)
- [ ] Data checksums computed

### Code
- [ ] All analysis in scripts (not just notebooks)
- [ ] Scripts version controlled
- [ ] Clear directory structure
- [ ] README with instructions

### Output
- [ ] Figures generated from scripts
- [ ] Tables generated from scripts
- [ ] All numbers traceable to data
- [ ] No manual adjustments
```

---

## Conclusion

Rigorous data analysis transforms raw measurements into scientific knowledge. The key principles are:

1. **Systematic approach:** Follow a structured analysis pipeline
2. **Statistical rigor:** Apply appropriate methods correctly
3. **Physical interpretation:** Connect numbers to mechanisms
4. **Honest reporting:** Present what the data shows
5. **Complete documentation:** Enable reproduction

Remember that analysis is iterative. Initial findings often lead to new questions, refined methods, and deeper understanding.

---

*Data Analysis and Interpretation Guide - Week 219*
*Month 55: Research Execution I*
