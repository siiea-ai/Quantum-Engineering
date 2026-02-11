# Parameter Study Template

## Study Information

**Project Title:** _______________________________________________

**Principal Investigator:** _______________________________________

**Study ID:** PARAM-[YYYY]-[NNN]

**Dates:** _______ to _______

**Status:** [ ] Planning [ ] In Progress [ ] Complete [ ] Under Review

---

## 1. Study Objectives

### 1.1 Primary Research Question
_What specific question does this parameter study address?_

```
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________
```

### 1.2 Success Criteria
_What outcomes would constitute a successful study?_

| Criterion | Target Value | Acceptable Range |
|-----------|--------------|------------------|
|           |              |                  |
|           |              |                  |
|           |              |                  |

### 1.3 Connection to Thesis/Project
_How does this study contribute to the broader research goals?_

```
___________________________________________________________________
___________________________________________________________________
```

---

## 2. Parameter Space Definition

### 2.1 Parameter Classification

#### Tier 1: Primary Control Parameters
_Parameters that will be systematically varied_

| Parameter | Symbol | Units | Range | Levels | Justification |
|-----------|--------|-------|-------|--------|---------------|
|           |        |       |       |        |               |
|           |        |       |       |        |               |
|           |        |       |       |        |               |

#### Tier 2: Secondary Control Parameters
_Parameters held constant but may be varied in follow-up studies_

| Parameter | Symbol | Units | Fixed Value | Rationale |
|-----------|--------|-------|-------------|-----------|
|           |        |       |             |           |
|           |        |       |             |           |

#### Tier 3: Nuisance Parameters
_Parameters that must be monitored but cannot be controlled_

| Parameter | Symbol | Units | Expected Range | Monitoring Method |
|-----------|--------|-------|----------------|-------------------|
|           |        |       |                |                   |
|           |        |       |                |                   |

### 2.2 Observable Quantities

| Observable | Symbol | Units | Measurement Method | Expected Range | Uncertainty |
|------------|--------|-------|-------------------|----------------|-------------|
|            |        |       |                   |                |             |
|            |        |       |                   |                |             |
|            |        |       |                   |                |             |

### 2.3 Dimensionless Groups
_Identify relevant dimensionless combinations_

$$\Pi_1 = \frac{[\text{numerator}]}{[\text{denominator}]} = $$

$$\Pi_2 = $$

**Physical interpretation:**
- $\Pi_1$: ___________
- $\Pi_2$: ___________

---

## 3. Experimental/Computational Design

### 3.1 Design Type
- [ ] Full Factorial
- [ ] Fractional Factorial (Resolution: ___)
- [ ] Central Composite Design
- [ ] Latin Hypercube Sampling
- [ ] Adaptive/Sequential
- [ ] Other: ___________

### 3.2 Design Rationale
_Why was this design chosen?_

```
___________________________________________________________________
___________________________________________________________________
```

### 3.3 Design Matrix

**Number of runs:** _______
**Replicates per condition:** _______
**Total measurements:** _______

| Run | $\theta_1$ | $\theta_2$ | $\theta_3$ | Block | Notes |
|-----|------------|------------|------------|-------|-------|
| 1   |            |            |            |       |       |
| 2   |            |            |            |       |       |
| 3   |            |            |            |       |       |
| ... |            |            |            |       |       |

### 3.4 Randomization and Blocking
_Describe how runs are randomized and blocked_

```
Randomization scheme: _______________________________________________
Blocking structure: ________________________________________________
Control/reference runs: ____________________________________________
```

### 3.5 Resource Requirements

| Resource | Quantity | Availability | Cost/Time |
|----------|----------|--------------|-----------|
| Equipment time |  |              |           |
| Computation |      |              |           |
| Materials |        |              |           |
| Personnel |        |              |           |

---

## 4. Measurement Protocol

### 4.1 Equipment/Software Configuration

```
System: _____________________________________________________________
Configuration: ______________________________________________________
Calibration date: __________________
Reference settings: ________________________________________________
```

### 4.2 Measurement Procedure

**Pre-measurement checklist:**
- [ ] Equipment warmed up and stable
- [ ] Calibration verified
- [ ] Data storage configured
- [ ] Backup systems active
- [ ] Safety protocols reviewed

**Step-by-step procedure:**

1. ________________________________________________________________
2. ________________________________________________________________
3. ________________________________________________________________
4. ________________________________________________________________
5. ________________________________________________________________

**Post-measurement checklist:**
- [ ] Data backed up
- [ ] Equipment returned to safe state
- [ ] Anomalies logged
- [ ] Next run prepared

### 4.3 Data Recording

**Data format:** ___________________
**File naming convention:** `YYYYMMDD_Run[NNN]_[description].[ext]`
**Storage location:** ______________________________________________
**Backup location:** _______________________________________________

---

## 5. Results

### 5.1 Raw Data Summary

| Run | $\theta_1$ | $\theta_2$ | Observable 1 | Observable 2 | Quality Flag |
|-----|------------|------------|--------------|--------------|--------------|
| 1   |            |            |              |              |              |
| 2   |            |            |              |              |              |
| ... |            |            |              |              |              |

**Quality flags:** G = Good, S = Suspect, R = Rejected

### 5.2 Data Quality Assessment

**Total runs attempted:** _______
**Successful runs:** _______
**Rejected runs:** _______ (reasons: _________________________________)

**Missing data:**
| Run | Parameter(s) | Reason | Mitigation |
|-----|--------------|--------|------------|
|     |              |        |            |

### 5.3 Primary Findings

#### Main Effects

| Parameter | Effect Size | 95% CI | p-value | Significance |
|-----------|-------------|--------|---------|--------------|
|           |             |        |         |              |
|           |             |        |         |              |

#### Interaction Effects

| Interaction | Effect Size | 95% CI | p-value | Significance |
|-------------|-------------|--------|---------|--------------|
|             |             |        |         |              |
|             |             |        |         |              |

### 5.4 Response Surface (if applicable)

**Fitted Model:**

$$y = \beta_0 + \sum_i \beta_i x_i + \sum_{i \leq j} \beta_{ij} x_i x_j + \epsilon$$

**Coefficient Estimates:**

| Term | Estimate | Std. Error | t-value | p-value |
|------|----------|------------|---------|---------|
| $\beta_0$ |    |            |         |         |
| $\beta_1$ |    |            |         |         |
| ...  |          |            |         |         |

**Model Quality:**
- $R^2 = $ _______
- Adjusted $R^2 = $ _______
- RMSE = _______

### 5.5 Figures

_Insert or reference key figures_

**Figure 1:** [Description]
- File: `figures/fig01_[description].pdf`
- Key observation: __________________________________________________

**Figure 2:** [Description]
- File: `figures/fig02_[description].pdf`
- Key observation: __________________________________________________

---

## 6. Scaling Analysis

### 6.1 Scaling Relationships Identified

| Relationship | Functional Form | Exponent(s) | Validity Range |
|--------------|-----------------|-------------|----------------|
|              |                 |             |                |
|              |                 |             |                |

### 6.2 Scaling Plots

**Log-log analysis:**
- Variable: ___________ vs. ___________
- Slope: ___________ ± ___________
- $R^2$: ___________

**Semi-log analysis:**
- Variable: ___________ vs. ___________
- Decay constant: ___________ ± ___________
- $R^2$: ___________

### 6.3 Asymptotic Behavior

| Limit | Expected Behavior | Observed Behavior | Agreement |
|-------|-------------------|-------------------|-----------|
| $x \to 0$ |               |                   |           |
| $x \to \infty$ |          |                   |           |

---

## 7. Boundary Conditions

### 7.1 Boundaries Identified

| Parameter | Boundary Type | Value | Failure Mode | Safety Margin |
|-----------|--------------|-------|--------------|---------------|
|           | Hard/Soft    |       |              |               |
|           | Stability    |       |              |               |

### 7.2 Operating Envelope

**Recommended operating range:**

| Parameter | Minimum | Maximum | Optimal |
|-----------|---------|---------|---------|
|           |         |         |         |
|           |         |         |         |

### 7.3 Warning Indicators

| Indicator | Warning Level | Action Required |
|-----------|--------------|-----------------|
|           |              |                 |
|           |              |                 |

---

## 8. Anomalies and Unexpected Findings

### 8.1 Anomalies Log

| ID | Run(s) | Description | Potential Cause | Follow-up Needed |
|----|--------|-------------|-----------------|------------------|
| A1 |        |             |                 | Yes/No           |
| A2 |        |             |                 | Yes/No           |

### 8.2 Unexpected Findings

**Finding 1:**
```
Description: ________________________________________________________
Significance: _______________________________________________________
Recommended action: _________________________________________________
```

**Finding 2:**
```
Description: ________________________________________________________
Significance: _______________________________________________________
Recommended action: _________________________________________________
```

---

## 9. Validation Requirements

### 9.1 Claims Requiring Validation

| Claim | Validation Method | Priority | Week 222 Task |
|-------|------------------|----------|---------------|
|       |                  | H/M/L    |               |
|       |                  | H/M/L    |               |

### 9.2 Reproducibility Tests Needed

- [ ] Repeat key measurements
- [ ] Independent implementation
- [ ] Cross-validation with theory
- [ ] Comparison with literature

---

## 10. Conclusions and Next Steps

### 10.1 Summary of Findings

1. ________________________________________________________________
2. ________________________________________________________________
3. ________________________________________________________________

### 10.2 Research Questions Answered

| Question | Answer | Confidence |
|----------|--------|------------|
|          |        | High/Med/Low |
|          |        |            |

### 10.3 New Questions Raised

1. ________________________________________________________________
2. ________________________________________________________________

### 10.4 Recommended Next Steps

| Priority | Action | Timeline | Resources Needed |
|----------|--------|----------|------------------|
| 1        |        |          |                  |
| 2        |        |          |                  |
| 3        |        |          |                  |

---

## 11. Documentation Checklist

- [ ] All raw data archived with metadata
- [ ] Analysis scripts committed to version control
- [ ] Figures publication-ready
- [ ] Lab notebook entries complete
- [ ] Parameter log up to date
- [ ] Anomalies documented
- [ ] Validation requirements identified
- [ ] Report reviewed by collaborator/advisor

---

## Appendices

### Appendix A: Detailed Procedures
_Include detailed step-by-step procedures here_

### Appendix B: Calibration Records
_Include calibration data and certificates_

### Appendix C: Analysis Code Reference
_List analysis scripts and their locations_

| Script | Location | Purpose |
|--------|----------|---------|
|        |          |         |
|        |          |         |

### Appendix D: Supplementary Figures
_Additional figures not included in main report_

---

**Report completed by:** _____________________ **Date:** ___________

**Reviewed by:** _____________________ **Date:** ___________

**Approved by:** _____________________ **Date:** ___________
