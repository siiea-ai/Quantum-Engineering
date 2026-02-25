# Parameter Space Analysis Template

## Analysis Information

| Field | Entry |
|-------|-------|
| Analysis ID | PSA-YYYY-MM-DD-### |
| Date | |
| Researcher | |
| Related Experiment IDs | |
| Analysis Version | 1.0 |

---

## 1. Parameter Space Definition

### 1.1 Parameters Under Investigation

**Primary Parameters**

| Parameter | Symbol | Range | Unit | Resolution | # Points | Rationale |
|-----------|--------|-------|------|------------|----------|-----------|
| | | [ , ] | | | | |
| | | [ , ] | | | | |
| | | [ , ] | | | | |

**Secondary Parameters (Held Constant)**

| Parameter | Symbol | Fixed Value | Unit | Justification |
|-----------|--------|-------------|------|---------------|
| | | | | |
| | | | | |

**Nuisance Parameters**

| Parameter | Nominal | Tolerance | Monitoring |
|-----------|---------|-----------|------------|
| | | ± | |
| | | ± | |

### 1.2 Output Metrics

| Metric | Symbol | Expected Range | Unit | Measurement Method |
|--------|--------|----------------|------|-------------------|
| | | [ , ] | | |
| | | [ , ] | | |
| | | [ , ] | | |

### 1.3 Parameter Space Dimensions

- **Total dimensions**: ___
- **Total parameter combinations**: ___
- **Sampling strategy**: ☐ Grid ☐ Random ☐ LHS ☐ Adaptive
- **Samples per point**: ___
- **Total measurements**: ___

---

## 2. Sampling Strategy

### 2.1 Sampling Design

**Grid Sampling (if used)**

| Parameter | Start | End | Step | Points |
|-----------|-------|-----|------|--------|
| | | | | |
| | | | | |

**Latin Hypercube / Random Sampling (if used)**

- Number of samples: ___
- Random seed: ___
- Coverage verification: ☐ Complete

**Adaptive Sampling (if used)**

- Initial samples: ___
- Acquisition function: ___
- Stopping criterion: ___

### 2.2 Reference Measurements

- Reference interval: ___ measurements
- Reference parameters: _______________
- Drift detection threshold: ___

---

## 3. Data Collection Summary

### 3.1 Collection Statistics

| Statistic | Value |
|-----------|-------|
| Total samples collected | |
| Valid samples | |
| Rejected samples | |
| Reference samples | |
| Collection duration | |

### 3.2 Quality Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| | | | ☐ Pass ☐ Fail |
| | | | ☐ Pass ☐ Fail |
| | | | ☐ Pass ☐ Fail |

### 3.3 Data Files

| Description | Filename | Size | Checksum |
|-------------|----------|------|----------|
| Raw data | | | |
| Processed data | | | |
| Metadata | | | |

---

## 4. Single-Parameter Analysis

### 4.1 Parameter 1: ________________

**Marginal Distribution**

| Value | Mean (Metric 1) | Std Dev | N |
|-------|-----------------|---------|---|
| | | | |
| | | | |
| | | | |

**Trend Analysis**

- Best fit function: _______________
- Fit parameters: _______________
- R-squared: ___
- Trend significance (p-value): ___

**Plot Reference:** [Filename]

### 4.2 Parameter 2: ________________

**Marginal Distribution**

| Value | Mean (Metric 1) | Std Dev | N |
|-------|-----------------|---------|---|
| | | | |
| | | | |
| | | | |

**Trend Analysis**

- Best fit function: _______________
- Fit parameters: _______________
- R-squared: ___
- Trend significance (p-value): ___

**Plot Reference:** [Filename]

### 4.3 Parameter 3: ________________

[Repeat structure as above]

---

## 5. Two-Parameter Analysis

### 5.1 Parameters: ____________ vs ____________

**Interaction Heatmap**

| Param 1 \ Param 2 | Value 1 | Value 2 | Value 3 | ... |
|-------------------|---------|---------|---------|-----|
| Value A | | | | |
| Value B | | | | |
| Value C | | | | |

**Interaction Analysis**

- Interaction detected: ☐ Yes ☐ No
- Interaction type: ☐ Synergistic ☐ Antagonistic ☐ None
- Interaction strength: ___
- p-value: ___

**Contour/Surface Features**

| Feature | Location | Description |
|---------|----------|-------------|
| Maximum | ( , ) | |
| Minimum | ( , ) | |
| Ridge/Valley | | |
| Transition | | |

**Plot Reference:** [Filename]

### 5.2 Parameters: ____________ vs ____________

[Repeat structure as above for other parameter pairs]

---

## 6. Sensitivity Analysis

### 6.1 Local Sensitivity

**At Reference Point:** (______, ______, ______)

| Parameter | Sensitivity | Normalized Sensitivity | Rank |
|-----------|-------------|------------------------|------|
| | | | |
| | | | |
| | | | |

### 6.2 Global Sensitivity (Sobol Indices)

**First-Order Indices (S1)**

| Parameter | S1 | 95% CI | Interpretation |
|-----------|----|----|----------------|
| | | [ , ] | |
| | | [ , ] | |
| | | [ , ] | |

**Total-Order Indices (ST)**

| Parameter | ST | 95% CI | Interpretation |
|-----------|----|----|----------------|
| | | [ , ] | |
| | | [ , ] | |
| | | [ , ] | |

**Interaction Assessment**

- Sum of S1: ___
- If Σ S1 << 1, strong interactions present
- Key interactions: _______________

### 6.3 Sensitivity Visualization

**Tornado Plot Reference:** [Filename]
**Scatter Plots Reference:** [Filename]

---

## 7. Transition and Threshold Analysis

### 7.1 Identified Transitions

| Transition ID | Parameter | Critical Value | Type | Confidence |
|---------------|-----------|----------------|------|------------|
| T1 | | | ☐ Sharp ☐ Gradual | |
| T2 | | | ☐ Sharp ☐ Gradual | |

### 7.2 Threshold Boundaries

| Metric | Threshold | Parameter Boundary | Uncertainty |
|--------|-----------|-------------------|-------------|
| | > | = | ± |
| | < | = | ± |

### 7.3 Critical Behavior (if applicable)

- Critical point: _______________
- Critical exponent (β): ___ ± ___
- Scaling regime: _______________

---

## 8. Optimal Regions

### 8.1 Single Objective Optimization

| Objective | Optimal Parameters | Optimal Value | Uncertainty |
|-----------|-------------------|---------------|-------------|
| Maximize Metric 1 | ( , , ) | | ± |
| Minimize Metric 2 | ( , , ) | | ± |

### 8.2 Multi-Objective Trade-offs

| Trade-off | Pareto Front | Recommended Point |
|-----------|--------------|-------------------|
| Metric 1 vs Metric 2 | [Plot reference] | ( , , ) |

### 8.3 Robustness Analysis

| Operating Point | Volume of Acceptable Region | Sensitivity to Perturbation |
|-----------------|-----------------------------|-----------------------------|
| Optimal | | |
| Robust | | |

---

## 9. Anomalies and Outliers

### 9.1 Detected Anomalies

| Point ID | Parameters | Observed Value | Expected Value | Deviation | Action |
|----------|------------|----------------|----------------|-----------|--------|
| | ( , , ) | | | σ | Kept/Removed |
| | ( , , ) | | | σ | Kept/Removed |

### 9.2 Investigation Notes

```
[Notes on investigation of anomalies]
```

---

## 10. Model Fitting

### 10.1 Empirical Models Tested

| Model | Formula | Parameters | R² | AIC | BIC |
|-------|---------|------------|----|----|-----|
| | | | | | |
| | | | | | |

### 10.2 Best Model

**Selected Model:** _______________

**Formula:**
$$
f(x_1, x_2, ...) =
$$

**Fitted Parameters:**

| Parameter | Value | Std Error | 95% CI |
|-----------|-------|-----------|--------|
| | | | [ , ] |
| | | | [ , ] |

**Residual Analysis:**

- Mean residual: ___
- Std residual: ___
- Pattern detected: ☐ Yes ☐ No

**Model Predictions Plot Reference:** [Filename]

---

## 11. Interpretation and Conclusions

### 11.1 Key Findings

**Finding 1:**
```
[Description of finding and its significance]
```

**Finding 2:**
```
[Description of finding and its significance]
```

**Finding 3:**
```
[Description of finding and its significance]
```

### 11.2 Parameter Importance Ranking

| Rank | Parameter | Evidence | Recommended Action |
|------|-----------|----------|-------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

### 11.3 Unexplained Observations

```
[Observations that require further investigation]
```

### 11.4 Implications for Research

```
[How these findings affect the research direction]
```

---

## 12. Recommendations

### 12.1 Additional Data Needed

| Region | Parameters | Justification |
|--------|------------|---------------|
| | | |
| | | |

### 12.2 Parameter Adjustments

| Current Setting | Recommended Setting | Reason |
|-----------------|---------------------|--------|
| | | |
| | | |

### 12.3 Next Steps

1.
2.
3.

---

## Appendices

### A. Complete Data Tables

[Reference to supplementary data files]

### B. All Visualization Files

| Plot Type | Filename | Description |
|-----------|----------|-------------|
| | | |
| | | |

### C. Statistical Analysis Details

[Reference to detailed statistical output]

### D. Code/Scripts Used

| Purpose | Filename | Version |
|---------|----------|---------|
| | | |
| | | |

---

## Sign-Off

**Analysis completed by:** ____________________

**Date:** ____________________

**Reviewed by:** ____________________

**Date:** ____________________

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | | | Initial analysis |
| | | | |
