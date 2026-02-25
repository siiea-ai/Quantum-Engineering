# Additional Investigations Guide: Gap Analysis and Robustness Testing

## Introduction

This guide provides comprehensive methodology for conducting additional investigations that strengthen your research before publication. The goal is to identify and address gaps, anticipate reviewer concerns, and ensure your conclusions are robust to scrutiny.

---

## Part 1: Systematic Gap Analysis

### 1.1 The Gap Analysis Process

Effective gap analysis follows a structured approach:

```
Current Results → Ideal Results → Gap Identification → Prioritization → Resolution
```

**Step 1: Define "Complete" Research**

Before identifying gaps, articulate what a complete, publication-ready study would include:

- All primary hypotheses tested
- Sufficient statistical power for all claims
- All major confounds controlled
- Baseline comparisons completed
- Edge cases characterized
- Uncertainties fully quantified

**Step 2: Assess Current State**

Evaluate your current results against this ideal:

| Dimension | Ideal State | Current State | Gap Severity |
|-----------|-------------|---------------|--------------|
| Coverage | [Description] | [Status] | High/Med/Low |
| Controls | [Description] | [Status] | High/Med/Low |
| Statistics | [Description] | [Status] | High/Med/Low |
| Baselines | [Description] | [Status] | High/Med/Low |

### 1.2 Gap Categories and Resolution Strategies

#### Coverage Gaps

**Identification Questions:**
- Are there parameter combinations not tested?
- Are there conditions not explored?
- Are sample sizes adequate for claimed precision?
- Are there missing controls?

**Resolution Strategies:**
```python
# Example: Identifying coverage gaps in parameter space
import numpy as np
from scipy.spatial import ConvexHull

def identify_coverage_gaps(tested_params, param_bounds, grid_resolution=20):
    """
    Identify regions of parameter space not adequately covered.

    Parameters:
    -----------
    tested_params : ndarray
        Array of shape (n_experiments, n_params) with tested parameter values
    param_bounds : list of tuples
        [(min1, max1), (min2, max2), ...] bounds for each parameter
    grid_resolution : int
        Number of grid points per dimension

    Returns:
    --------
    coverage_map : ndarray
        Boolean array indicating covered regions
    gaps : list
        List of uncovered regions
    """
    n_params = len(param_bounds)

    # Create grid
    grids = [np.linspace(low, high, grid_resolution)
             for low, high in param_bounds]
    mesh = np.meshgrid(*grids, indexing='ij')
    grid_points = np.stack([m.ravel() for m in mesh], axis=1)

    # Calculate distances to nearest tested point
    from scipy.spatial import cKDTree
    tree = cKDTree(tested_params)
    distances, _ = tree.query(grid_points)

    # Identify gaps (points far from any tested point)
    threshold = np.median(distances) * 2
    gaps = grid_points[distances > threshold]

    return distances.reshape([grid_resolution]*n_params), gaps
```

#### Methodological Gaps

**Identification Questions:**
- Are all experimental procedures documented?
- Are calibration procedures included?
- Are there missing validation steps?
- Is the data processing pipeline complete?

**Resolution Approach:**

Create a methodology completeness checklist:

```markdown
## Methodology Completeness Checklist

### Experimental Setup
- [ ] Equipment specifications documented
- [ ] Calibration procedures described
- [ ] Environmental conditions recorded
- [ ] Control experiments included

### Data Collection
- [ ] Sampling procedures documented
- [ ] Data formats specified
- [ ] Quality control procedures in place
- [ ] Metadata standards followed

### Data Processing
- [ ] Pre-processing steps documented
- [ ] Outlier handling justified
- [ ] Missing data treatment described
- [ ] Software versions recorded

### Analysis
- [ ] Statistical methods justified
- [ ] Assumptions verified
- [ ] Multiple comparison corrections applied
- [ ] Effect sizes reported
```

#### Analytical Gaps

**Identification Questions:**
- Have all appropriate statistical tests been applied?
- Are assumptions of statistical tests verified?
- Are confidence intervals reported?
- Is the uncertainty analysis complete?

**Resolution Framework:**

```python
def statistical_gap_analysis(results_df, alpha=0.05):
    """
    Check for common statistical gaps in analysis.

    Returns dict of identified gaps and recommendations.
    """
    gaps = {}

    # Check for normality tests
    from scipy import stats
    for column in results_df.select_dtypes(include=[np.number]).columns:
        _, p_value = stats.shapiro(results_df[column].dropna())
        if p_value < alpha:
            gaps[f'normality_{column}'] = {
                'issue': f'{column} may not be normally distributed (p={p_value:.4f})',
                'recommendation': 'Consider non-parametric tests or bootstrap methods'
            }

    # Check for adequate sample sizes
    for column in results_df.columns:
        n = results_df[column].count()
        if n < 30:
            gaps[f'sample_size_{column}'] = {
                'issue': f'{column} has small sample size (n={n})',
                'recommendation': 'Report exact confidence intervals, consider non-parametric methods'
            }

    # Check for multiple comparisons
    n_comparisons = len(results_df.columns) * (len(results_df.columns) - 1) // 2
    if n_comparisons > 3:
        gaps['multiple_comparisons'] = {
            'issue': f'{n_comparisons} pairwise comparisons without correction',
            'recommendation': 'Apply Bonferroni or FDR correction'
        }

    return gaps
```

#### Interpretive Gaps

**Identification Questions:**
- Have alternative explanations been considered?
- Are limitations clearly stated?
- Is generalizability addressed?
- Are implications appropriately scoped?

**Alternative Explanation Framework:**

| Your Claim | Alternative Explanation | Discriminating Test | Result |
|------------|------------------------|---------------------|--------|
| [Claim 1] | [Alt. explanation] | [Experiment design] | [Evidence] |
| [Claim 2] | [Alt. explanation] | [Experiment design] | [Evidence] |

### 1.3 Gap Prioritization

Use the **RICE** framework to prioritize gaps:

- **R**each: How much of your research does this gap affect?
- **I**mpact: How severely does it weaken your claims?
- **C**onfidence: How sure are you that addressing it helps?
- **E**ffort: How much work is required?

$$\text{Priority Score} = \frac{\text{Reach} \times \text{Impact} \times \text{Confidence}}{\text{Effort}}$$

Score each factor 1-10:

| Gap | Reach | Impact | Confidence | Effort | Score | Priority |
|-----|-------|--------|------------|--------|-------|----------|
| [Gap 1] | 8 | 9 | 7 | 3 | 168 | 1 |
| [Gap 2] | 6 | 7 | 8 | 5 | 67 | 2 |
| [Gap 3] | 4 | 5 | 6 | 8 | 15 | 3 |

---

## Part 2: Anticipating Reviewer Concerns

### 2.1 Common Reviewer Objections

Based on analysis of peer review literature, reviewers most commonly raise these concerns:

#### Technical Concerns

1. **Insufficient Controls**
   - "The authors did not include a control for..."
   - "How do we know the effect is not due to..."

2. **Statistical Issues**
   - "The sample size is too small for the claimed precision"
   - "Multiple comparisons were not corrected"
   - "Effect size is not reported"

3. **Reproducibility Concerns**
   - "Methods are insufficiently detailed"
   - "Code/data are not available"
   - "Key parameters are not specified"

#### Conceptual Concerns

1. **Novelty Questions**
   - "How does this differ from [prior work]?"
   - "The contribution is incremental"

2. **Significance Doubts**
   - "The practical implications are unclear"
   - "This applies only to a narrow case"

3. **Alternative Explanations**
   - "The authors do not consider..."
   - "A simpler explanation would be..."

### 2.2 The Reviewer Simulation Exercise

Put yourself in a critical reviewer's position:

**Exercise 1: Read your abstract as a skeptic**

For each claim in your abstract, ask:
- Is this supported by quantitative evidence?
- Could this be explained otherwise?
- Is this overstated?

**Exercise 2: Weakness identification**

Complete this table honestly:

| Claim | Supporting Evidence | Weakness | Severity |
|-------|-------------------|----------|----------|
| [Claim 1] | [Evidence] | [Weakness] | High/Med/Low |
| [Claim 2] | [Evidence] | [Weakness] | High/Med/Low |

**Exercise 3: Alternative hypothesis brainstorming**

For your main finding, list at least three alternative explanations:

1. [Alternative 1 and why you can rule it out]
2. [Alternative 2 and why you can rule it out]
3. [Alternative 3 and why you can rule it out]

### 2.3 Building the Anticipated Q&A Document

Structure your Q&A document as follows:

```markdown
# Anticipated Reviewer Questions and Responses

## Category 1: Technical Questions

### Q1.1: [Anticipated question]
**Why they might ask:** [Reasoning]
**Our response:** [Detailed answer]
**Supporting evidence:** [Figure/Table/Data reference]

### Q1.2: [Anticipated question]
...

## Category 2: Significance Questions
...

## Category 3: Methodology Questions
...

## Category 4: Comparison Questions
...
```

---

## Part 3: Robustness Analysis

### 3.1 Types of Robustness Tests

#### Parameter Sensitivity Analysis

Test how results change when parameters vary within their uncertainty ranges:

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

def parameter_sensitivity_analysis(model_func, param_bounds, n_samples=1024):
    """
    Perform Sobol sensitivity analysis on model parameters.

    Parameters:
    -----------
    model_func : callable
        Function that takes parameter dict and returns metric
    param_bounds : dict
        {'param_name': [min, max], ...}
    n_samples : int
        Number of samples for analysis

    Returns:
    --------
    sensitivity_results : dict
        First-order and total sensitivity indices for each parameter
    """
    problem = {
        'num_vars': len(param_bounds),
        'names': list(param_bounds.keys()),
        'bounds': list(param_bounds.values())
    }

    # Generate samples
    param_values = saltelli.sample(problem, n_samples)

    # Evaluate model
    Y = np.array([
        model_func({name: val for name, val in zip(problem['names'], params)})
        for params in param_values
    ])

    # Analyze
    Si = sobol.analyze(problem, Y)

    return {
        'first_order': dict(zip(problem['names'], Si['S1'])),
        'total': dict(zip(problem['names'], Si['ST'])),
        'interactions': Si['S2'] if 'S2' in Si else None
    }
```

#### Bootstrap Robustness

Test stability of conclusions across resampled datasets:

```python
def bootstrap_robustness_test(data, analysis_func, n_bootstrap=1000, ci=0.95):
    """
    Test robustness of analysis conclusions via bootstrap.

    Parameters:
    -----------
    data : array-like
        Original dataset
    analysis_func : callable
        Function that takes data and returns metric of interest
    n_bootstrap : int
        Number of bootstrap iterations
    ci : float
        Confidence level for intervals

    Returns:
    --------
    results : dict
        Contains point estimate, bootstrap distribution, CI
    """
    # Original estimate
    original_estimate = analysis_func(data)

    # Bootstrap
    boot_estimates = []
    n = len(data)
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_estimates.append(analysis_func(boot_sample))

    boot_estimates = np.array(boot_estimates)

    # Confidence interval
    alpha = 1 - ci
    ci_lower = np.percentile(boot_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2))

    # Stability metrics
    stability = {
        'cv': np.std(boot_estimates) / np.mean(boot_estimates),
        'sign_stable': np.mean(np.sign(boot_estimates) == np.sign(original_estimate))
    }

    return {
        'original': original_estimate,
        'bootstrap_mean': np.mean(boot_estimates),
        'bootstrap_std': np.std(boot_estimates),
        'ci': (ci_lower, ci_upper),
        'stability': stability
    }
```

#### Leave-One-Out Analysis

Test sensitivity to individual data points:

```python
def leave_one_out_sensitivity(data, analysis_func):
    """
    Assess sensitivity of analysis to individual data points.

    Returns:
    --------
    results : dict
        Influence scores and outlier flags for each data point
    """
    n = len(data)
    full_result = analysis_func(data)

    influences = []
    for i in range(n):
        loo_data = np.delete(data, i, axis=0) if data.ndim > 1 else np.delete(data, i)
        loo_result = analysis_func(loo_data)
        influences.append(full_result - loo_result)

    influences = np.array(influences)

    # Identify high-influence points
    threshold = np.mean(np.abs(influences)) + 2 * np.std(np.abs(influences))
    high_influence = np.abs(influences) > threshold

    return {
        'influences': influences,
        'high_influence_indices': np.where(high_influence)[0],
        'max_influence': np.max(np.abs(influences)),
        'mean_influence': np.mean(np.abs(influences))
    }
```

### 3.2 Model Robustness

Test whether conclusions depend on specific modeling choices:

```python
def model_robustness_analysis(data, models, metric_func):
    """
    Compare conclusions across different model specifications.

    Parameters:
    -----------
    data : dict
        Data for model fitting
    models : list of callables
        Different model implementations
    metric_func : callable
        Function to extract metric of interest from fitted model

    Returns:
    --------
    comparison : DataFrame
        Metrics across all models
    """
    results = []
    for i, model in enumerate(models):
        try:
            fitted = model(data)
            metric = metric_func(fitted)
            results.append({
                'model': i,
                'model_name': getattr(model, '__name__', f'Model_{i}'),
                'metric': metric,
                'converged': True
            })
        except Exception as e:
            results.append({
                'model': i,
                'model_name': getattr(model, '__name__', f'Model_{i}'),
                'metric': np.nan,
                'converged': False,
                'error': str(e)
            })

    return pd.DataFrame(results)
```

### 3.3 Assumption Violation Testing

Systematically test what happens when assumptions are violated:

| Assumption | Violation Tested | Impact on Conclusions | Action Required |
|------------|------------------|----------------------|-----------------|
| Normality | Non-normal residuals | [Minor/Major] | [Action] |
| Independence | Correlated samples | [Minor/Major] | [Action] |
| Homoscedasticity | Heterogeneous variance | [Minor/Major] | [Action] |
| Linearity | Non-linear relationships | [Minor/Major] | [Action] |

---

## Part 4: Baseline Comparisons

### 4.1 Selecting Appropriate Baselines

Baselines should include:

1. **Trivial Baseline** - Simplest possible approach (e.g., random guessing, mean prediction)
2. **Standard Method** - Established technique in the field
3. **State-of-the-Art** - Best published method for comparison
4. **Ablations** - Your method with components removed

### 4.2 Fair Comparison Protocol

```markdown
## Fair Comparison Checklist

### Setup Fairness
- [ ] Same training/test data splits
- [ ] Same computational resources (or normalized)
- [ ] Same hyperparameter tuning budget
- [ ] Same random seeds where applicable

### Implementation Fairness
- [ ] Official implementations used where available
- [ ] Hyperparameters tuned on validation set
- [ ] Multiple runs for statistical comparison
- [ ] Same evaluation metrics

### Reporting Fairness
- [ ] Uncertainty estimates for all methods
- [ ] Statistical significance tests
- [ ] Discussion of where baseline excels
- [ ] Acknowledgment of limitations
```

### 4.3 Comparison Visualization

```python
def create_comparison_figure(methods, metrics, uncertainties,
                            metric_name='Performance', save_path=None):
    """
    Create publication-quality comparison figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(methods))
    colors = sns.color_palette("colorblind", len(methods))

    bars = ax.bar(x, metrics, yerr=uncertainties,
                  capsize=5, color=colors, edgecolor='black', linewidth=0.5)

    # Highlight your method
    bars[-1].set_edgecolor('red')
    bars[-1].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison')

    # Add significance markers
    # (implement based on your statistical test results)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

---

## Part 5: Integrating Additional Results

### 5.1 Updating the Results Summary

After completing additional investigations:

1. **Integrate new data** into existing results tables
2. **Update uncertainty estimates** based on robustness analysis
3. **Add new figures** showing sensitivity/robustness results
4. **Revise conclusions** if sensitivity analysis reveals issues
5. **Strengthen claims** with additional supporting evidence

### 5.2 Documentation Standards

For each additional investigation, document:

```markdown
## Additional Investigation: [Name]

### Motivation
[Why this investigation was needed]

### Method
[How it was conducted]

### Results
[What was found]

### Impact on Conclusions
[How this changes or supports main conclusions]

### Evidence
[Figures, tables, data references]
```

### 5.3 Version Control for Results

Track changes to your results documentation:

```bash
# After completing additional investigations
git add results/
git add analysis/robustness/
git commit -m "Week 226: Additional investigations complete

- Gap analysis addressing [X, Y, Z]
- Robustness tests showing [stability/sensitivity]
- Baseline comparisons demonstrating [improvement]
- Updated uncertainty estimates"

git tag -a "v0.2-additional-investigations" -m "Results with robustness analysis"
```

---

## Part 6: Quality Assurance Checklist

### 6.1 Gap Analysis Complete

- [ ] All major gaps identified
- [ ] Gaps prioritized using RICE framework
- [ ] High-priority gaps addressed
- [ ] Remaining gaps documented as limitations

### 6.2 Reviewer Preparation Complete

- [ ] Alternative explanations addressed
- [ ] Anticipated Q&A document drafted
- [ ] Weakest claims strengthened or qualified
- [ ] Supporting experiments completed

### 6.3 Robustness Analysis Complete

- [ ] Parameter sensitivity analyzed
- [ ] Bootstrap stability confirmed
- [ ] Leave-one-out analysis performed
- [ ] Model robustness verified
- [ ] Assumption violations tested

### 6.4 Baseline Comparisons Complete

- [ ] Appropriate baselines selected
- [ ] Fair comparison protocol followed
- [ ] Statistical significance established
- [ ] Comparison figures created

---

## Navigation

| Previous | Current | Next |
|----------|---------|------|
| Week 226 README | Additional Investigations Guide | [Gap Analysis Template](./Templates/Gap_Analysis.md) |
