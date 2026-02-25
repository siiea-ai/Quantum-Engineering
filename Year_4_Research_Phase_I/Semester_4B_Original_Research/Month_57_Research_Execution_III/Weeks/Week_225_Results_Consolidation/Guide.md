# Results Organization Guide

## Introduction

This guide provides comprehensive instructions for organizing, analyzing, and presenting your research results in a clear, reproducible, and publication-ready format. Proper results organization is critical for both internal clarity and external communication of your research findings.

---

## Part 1: Completing Your Experiments

### 1.1 Experiment Inventory

Before consolidating results, create a complete inventory of all experiments:

```markdown
| Exp ID | Description | Status | Priority | Completion Date |
|--------|-------------|--------|----------|-----------------|
| E001 | Baseline characterization | Complete | - | 2024-01-15 |
| E002 | Parameter sweep | Complete | - | 2024-01-20 |
| E003 | Edge case testing | Incomplete | High | Pending |
| E004 | Robustness check | Not started | Medium | Pending |
```

### 1.2 Prioritization Framework

Use the **ICE** method to prioritize remaining experiments:

- **I**mpact: How important is this for your main claims?
- **C**onfidence: How likely is it to succeed?
- **E**ffort: How much time/resources required?

Score each factor 1-10 and multiply for total priority score.

### 1.3 Completion Checklist

For each experiment, verify:

- [ ] All planned runs executed
- [ ] Sufficient repetitions for statistical power
- [ ] Raw data properly stored and backed up
- [ ] Metadata (parameters, conditions) documented
- [ ] Initial quality checks passed
- [ ] Anomalies noted and explained

---

## Part 2: Data Organization

### 2.1 Directory Structure

Adopt a consistent, self-documenting directory structure:

```
project_root/
├── data/
│   ├── raw/
│   │   ├── experiment_001/
│   │   │   ├── run_001.csv
│   │   │   ├── run_002.csv
│   │   │   └── metadata.json
│   │   └── experiment_002/
│   ├── processed/
│   │   ├── cleaned_dataset.parquet
│   │   └── feature_matrix.npy
│   └── external/
│       └── reference_data.csv
├── results/
│   ├── figures/
│   │   ├── main/
│   │   └── supplementary/
│   ├── tables/
│   └── statistics/
└── analysis/
    ├── scripts/
    └── notebooks/
```

### 2.2 File Naming Conventions

Use consistent, informative file names:

```
[date]_[experiment]_[description]_[version].[ext]

Examples:
2024-03-15_exp001_baseline_v2.csv
2024-03-16_exp001_analysis_final.ipynb
fig_01_performance_comparison.pdf
```

### 2.3 Metadata Standards

Every dataset should have accompanying metadata:

```json
{
  "experiment_id": "E001",
  "description": "Baseline fidelity characterization",
  "date": "2024-03-15",
  "researcher": "Your Name",
  "parameters": {
    "temperature_K": 20e-3,
    "drive_power_dBm": -30,
    "measurement_time_us": 100
  },
  "equipment": {
    "dilution_refrigerator": "BlueFors LD400",
    "signal_generator": "R&S SMW200A"
  },
  "files": ["run_001.csv", "run_002.csv"],
  "notes": "Stable conditions throughout"
}
```

---

## Part 3: Statistical Analysis

### 3.1 Descriptive Statistics

Always start with basic descriptive statistics:

```python
import pandas as pd
import numpy as np
from scipy import stats

def compute_descriptive_stats(data):
    """Compute comprehensive descriptive statistics."""
    return {
        'n': len(data),
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'sem': stats.sem(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'min': np.min(data),
        'max': np.max(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
```

### 3.2 Hypothesis Testing

Choose appropriate tests based on your data:

| Scenario | Test | Assumptions |
|----------|------|-------------|
| Two groups, normal | t-test | Normality, equal variance |
| Two groups, non-normal | Mann-Whitney U | Ordinal data |
| Paired samples | Paired t-test | Normality of differences |
| Multiple groups | ANOVA | Normality, homoscedasticity |
| Correlation | Pearson/Spearman | Linearity (Pearson) |

### 3.3 Effect Size Calculation

Always report effect sizes alongside p-values:

```python
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

| Effect Size (d) | Interpretation |
|-----------------|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### 3.4 Confidence Intervals

Report confidence intervals for all estimates:

```python
def bootstrap_ci(data, statistic=np.mean, n_boot=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    boot_stats = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))

    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return lower, upper
```

---

## Part 4: Uncertainty Quantification

### 4.1 Measurement Uncertainty

Follow the GUM (Guide to the Expression of Uncertainty in Measurement):

1. **Identify sources** of uncertainty
2. **Quantify** each component
3. **Combine** using uncertainty propagation
4. **Report** with appropriate significant figures

### 4.2 Uncertainty Budget

Create a comprehensive uncertainty budget:

```markdown
| Source | Type | Distribution | Value | Contribution |
|--------|------|--------------|-------|--------------|
| Readout noise | A | Normal | 0.02 | 15% |
| Temperature drift | B | Rectangular | 0.01 | 5% |
| Timing jitter | B | Normal | 0.03 | 20% |
| Calibration | B | Normal | 0.05 | 40% |
| Model error | B | Uniform | 0.02 | 20% |
| **Combined** | - | - | **0.07** | **100%** |
```

### 4.3 Error Propagation

For derived quantities:

```python
from uncertainties import ufloat
from uncertainties.umath import *

# Define measurements with uncertainties
x = ufloat(10.0, 0.1)  # 10.0 +/- 0.1
y = ufloat(5.0, 0.2)   # 5.0 +/- 0.2

# Automatic error propagation
z = x * y / (x + y)
print(f"z = {z}")  # Outputs value with propagated uncertainty
```

---

## Part 5: Visualization

### 5.1 Figure Design Principles

1. **Clarity**: Message should be immediately apparent
2. **Accuracy**: Data representation must be faithful
3. **Efficiency**: Maximize data-to-ink ratio
4. **Aesthetics**: Consistent, professional appearance

### 5.2 Publication Standards

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication-quality settings
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.5),  # Single column
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
})
```

### 5.3 Figure Types and Uses

| Figure Type | Best For | Key Elements |
|-------------|----------|--------------|
| Line plot | Trends, time series | Legend, error bands |
| Scatter plot | Correlations | Trendline, R-squared |
| Bar chart | Comparisons | Error bars, significance |
| Heatmap | Matrices, 2D data | Colorbar, annotations |
| Histogram | Distributions | Bin edges, density |
| Box plot | Distribution comparison | Outliers, quartiles |

### 5.4 Color Guidelines

Use colorblind-friendly palettes:

```python
# Recommended palettes
colorblind_palette = ['#0173B2', '#DE8F05', '#029E73',
                      '#D55E00', '#CC78BC', '#CA9161']

# Or use seaborn's colorblind palette
import seaborn as sns
palette = sns.color_palette("colorblind")
```

### 5.5 Error Representation

Always show uncertainty:

```python
# Error bars for discrete data
plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=3)

# Confidence bands for continuous data
plt.fill_between(x, y_lower, y_upper, alpha=0.3)
plt.plot(x, y_mean, '-')
```

---

## Part 6: Results Narrative

### 6.1 Narrative Structure

Organize results following this structure:

1. **Overview**: High-level summary of main findings
2. **Primary Results**: Core experimental outcomes
3. **Secondary Results**: Supporting evidence
4. **Comparative Analysis**: Comparison with baselines/literature
5. **Limitations**: Acknowledged constraints

### 6.2 Writing Style

- Use past tense for completed experiments
- Be precise and quantitative
- State observations before interpretations
- Reference figures and tables explicitly

### 6.3 Quantitative Language

| Instead of | Write |
|------------|-------|
| "significantly improved" | "improved by 23% (p < 0.001)" |
| "much better" | "3.5x higher (95% CI: 2.8-4.2)" |
| "approximately" | "within 5% of the expected value" |

---

## Part 7: Quality Assurance

### 7.1 Data Validation Checklist

- [ ] No missing values in critical columns
- [ ] Values within expected physical ranges
- [ ] No duplicate entries
- [ ] Consistent units across datasets
- [ ] Timestamps are valid and sequential

### 7.2 Analysis Validation

- [ ] Scripts run without errors
- [ ] Results reproducible from raw data
- [ ] Statistical tests appropriate for data type
- [ ] Effect sizes and confidence intervals reported
- [ ] Multiple comparison correction applied

### 7.3 Figure Validation

- [ ] Axes labeled with units
- [ ] Legend complete and clear
- [ ] Error bars/bands included
- [ ] Font sizes readable when printed
- [ ] Colors accessible to colorblind readers

---

## Part 8: Common Pitfalls

### 8.1 Statistical Errors

1. **P-hacking**: Running multiple tests until finding significance
2. **Cherry-picking**: Reporting only favorable results
3. **Overfitting**: Complex models on limited data
4. **Ignoring assumptions**: Using tests without checking requirements

### 8.2 Visualization Errors

1. **Truncated axes**: Exaggerating differences
2. **Missing error bars**: Hiding uncertainty
3. **3D charts**: Distorting proportions
4. **Dual y-axes**: Confusing correlations

### 8.3 Documentation Errors

1. **Missing metadata**: Can't reproduce setup
2. **Unclear units**: Ambiguous quantities
3. **Version confusion**: Wrong code/data version
4. **Lost provenance**: Can't trace data origin

---

## Part 9: Templates and Examples

### 9.1 Results Section Template

```markdown
## Results

### Primary Finding

We observed [main result] across [n] independent experiments
(Figure X). The measured [quantity] was [value ± uncertainty]
[units], which represents a [X%] improvement over the baseline
(p < [value], Cohen's d = [value]).

### Supporting Evidence

Additional experiments confirmed [secondary findings]
(Supplementary Figure Y). The correlation between [A] and [B]
was statistically significant (r = [value], 95% CI: [lower, upper]).

### Comparison with Literature

Our results are consistent with previous reports of [topic]
[citations], while extending the range of [parameter] from
[old value] to [new value].
```

### 9.2 Figure Caption Template

```markdown
**Figure X. [Brief descriptive title].**
(a) [Panel a description]. (b) [Panel b description].
Data points show mean ± SEM (n = [number]). Solid lines
indicate [model/fit]. Shaded regions represent 95% confidence
intervals. Statistical significance: *p < 0.05, **p < 0.01,
***p < 0.001.
```

---

## Part 10: Checklist for Week 225

### Experiment Completion
- [ ] All experiments inventoried
- [ ] Missing experiments identified and prioritized
- [ ] Remaining experiments executed
- [ ] Raw data backed up in multiple locations

### Data Organization
- [ ] Directory structure established
- [ ] Consistent naming conventions applied
- [ ] Metadata files created for each experiment
- [ ] Data integrity verified

### Statistical Analysis
- [ ] Descriptive statistics computed
- [ ] Appropriate hypothesis tests applied
- [ ] Effect sizes calculated
- [ ] Confidence intervals reported

### Uncertainty Quantification
- [ ] All uncertainty sources identified
- [ ] Type A and Type B uncertainties quantified
- [ ] Uncertainty budget created
- [ ] Combined uncertainties calculated

### Visualization
- [ ] Publication-quality figures created
- [ ] Color accessibility verified
- [ ] Error bars/bands included
- [ ] Figure captions drafted

### Results Summary
- [ ] Results narrative written
- [ ] Quantitative language used
- [ ] Figures and tables referenced
- [ ] Template completed

---

## Navigation

| Previous | Current | Next |
|----------|---------|------|
| Week 225 README | Results Organization Guide | [Results Summary Template](./Templates/Results_Summary.md) |
