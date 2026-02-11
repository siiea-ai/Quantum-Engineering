# Comparative Analysis Template

## Multi-Project Thesis Comparison Framework

---

## Purpose

This template provides a structured approach to comparing results across your research projects (R1 and R2) as well as with theoretical predictions and published literature. Effective comparative analysis demonstrates your deep understanding of the field and highlights the significance of your contributions.

---

## Part I: Research Project 1 vs. Research Project 2 Comparison

### 1.1 Direct Metric Comparison

**Quantitative Metrics:**

| Metric | Symbol | R1 Value | R2 Value | R2/R1 Ratio | Significance Level |
|--------|--------|----------|----------|-------------|-------------------|
| [Metric 1] | $M_1$ | $v_1 \pm \sigma_1$ | $v_2 \pm \sigma_2$ | $r \pm \delta r$ | $p < $ |
| [Metric 2] | $M_2$ | $v_1 \pm \sigma_1$ | $v_2 \pm \sigma_2$ | $r \pm \delta r$ | $p < $ |
| [Metric 3] | $M_3$ | $v_1 \pm \sigma_1$ | $v_2 \pm \sigma_2$ | $r \pm \delta r$ | $p < $ |
| [Metric 4] | $M_4$ | $v_1 \pm \sigma_1$ | $v_2 \pm \sigma_2$ | $r \pm \delta r$ | $p < $ |
| [Metric 5] | $M_5$ | $v_1 \pm \sigma_1$ | $v_2 \pm \sigma_2$ | $r \pm \delta r$ | $p < $ |

**Ratio Calculation:**
$$r = \frac{v_2}{v_1}, \quad \delta r = r \sqrt{\left(\frac{\sigma_1}{v_1}\right)^2 + \left(\frac{\sigma_2}{v_2}\right)^2}$$

**Significance Testing:**
- Null hypothesis: $H_0: v_1 = v_2$
- Test statistic: $t = \frac{v_2 - v_1}{\sqrt{\sigma_1^2 + \sigma_2^2}}$
- p-value calculation: [Method]

### 1.2 Qualitative Comparison

**Approach Comparison:**

| Aspect | R1 Approach | R2 Approach | Evolution |
|--------|-------------|-------------|-----------|
| Research Question | [R1 question] | [R2 question] | [How refined] |
| Methodology | [R1 method] | [R2 method] | [Improvements] |
| Sample/System | [R1 system] | [R2 system] | [Differences] |
| Analysis | [R1 analysis] | [R2 analysis] | [Enhancements] |
| Key Assumption | [R1 assumption] | [R2 assumption] | [Relaxations] |

**Capability Comparison:**

| Capability | R1 | R2 | Notes |
|------------|:--:|:--:|-------|
| [Capability A] | Yes/No/Partial | Yes/No/Partial | [Explanation] |
| [Capability B] | Yes/No/Partial | Yes/No/Partial | [Explanation] |
| [Capability C] | Yes/No/Partial | Yes/No/Partial | [Explanation] |
| [Capability D] | Yes/No/Partial | Yes/No/Partial | [Explanation] |

### 1.3 Source of Improvement Analysis

When R2 shows improvement over R1, decompose the contributing factors:

**Improvement Attribution:**

| Factor | Estimated Contribution | Evidence | Uncertainty |
|--------|----------------------|----------|-------------|
| [Factor 1] | [X]% | [Data/Analysis reference] | ±[Y]% |
| [Factor 2] | [X]% | [Data/Analysis reference] | ±[Y]% |
| [Factor 3] | [X]% | [Data/Analysis reference] | ±[Y]% |
| Other/Unexplained | [X]% | Residual | ±[Y]% |
| **Total** | **100%** | | |

**Verification:**
- Cross-check: Does sum of individual contributions match total improvement?
- Interaction effects: Are there synergistic effects between factors?

### 1.4 Visualization Templates

**Side-by-Side Comparison Plot:**

```python
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# R1 Data (left panel)
ax1.errorbar(x_r1, y_r1, yerr=err_r1, fmt='o', color='#0072B2',
             label='R1 Data', capsize=3)
ax1.plot(x_theory, y_theory_r1, '--', color='#999999', label='Theory')
ax1.set_xlabel('Parameter')
ax1.set_ylabel('Observable')
ax1.set_title('Research Project 1')
ax1.legend()

# R2 Data (right panel)
ax2.errorbar(x_r2, y_r2, yerr=err_r2, fmt='s', color='#D55E00',
             label='R2 Data', capsize=3)
ax2.plot(x_theory, y_theory_r2, '--', color='#999999', label='Theory')
ax2.set_xlabel('Parameter')
ax2.set_ylabel('Observable')
ax2.set_title('Research Project 2')
ax2.legend()

# Same scales for comparison
y_max = max(max(y_r1 + err_r1), max(y_r2 + err_r2)) * 1.1
ax1.set_ylim(0, y_max)
ax2.set_ylim(0, y_max)

plt.tight_layout()
plt.savefig('r1_r2_comparison.pdf', dpi=300)
```

**Improvement Factor Plot:**

```python
fig, ax = plt.subplots(figsize=(8, 6))

metrics = ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4']
ratios = [r1, r2, r3, r4]  # R2/R1 ratios
errors = [e1, e2, e3, e4]

x = np.arange(len(metrics))
colors = ['#D55E00' if r > 1 else '#0072B2' for r in ratios]

ax.bar(x, ratios, yerr=errors, capsize=5, color=colors, edgecolor='black')
ax.axhline(y=1, color='black', linestyle='--', label='R1 baseline')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.set_ylabel('R2 / R1 Ratio')
ax.set_title('Improvement Factors: R2 vs R1')
ax.legend()

plt.tight_layout()
plt.savefig('improvement_factors.pdf', dpi=300)
```

---

## Part II: Theory vs. Experiment Comparison

### 2.1 Model Validation Matrix

| Observable | Theory Prediction | Measured Value | Agreement | Notes |
|------------|------------------|----------------|-----------|-------|
| [Obs 1] | $f(\theta) = $ [expression] | [value ± error] | Excellent/Good/Fair/Poor | [Comment] |
| [Obs 2] | $g(\theta) = $ [expression] | [value ± error] | Excellent/Good/Fair/Poor | [Comment] |
| [Obs 3] | $h(\theta) = $ [expression] | [value ± error] | Excellent/Good/Fair/Poor | [Comment] |

**Agreement Criteria:**
- Excellent: |Experiment - Theory| < 1σ
- Good: |Experiment - Theory| < 2σ
- Fair: |Experiment - Theory| < 3σ
- Poor: |Experiment - Theory| > 3σ

### 2.2 Scaling Behavior Comparison

| Scaling Relation | Theoretical Exponent | Measured Exponent | Agreement |
|------------------|---------------------|-------------------|-----------|
| $M_1 \propto x^{\alpha}$ | $\alpha_{\text{th}} = $ | $\alpha_{\text{exp}} = \pm$ | [χ²/dof] |
| $M_2 \propto y^{\beta}$ | $\beta_{\text{th}} = $ | $\beta_{\text{exp}} = \pm$ | [χ²/dof] |
| $M_3 \propto z^{\gamma}$ | $\gamma_{\text{th}} = $ | $\gamma_{\text{exp}} = \pm$ | [χ²/dof] |

### 2.3 Residual Analysis

**Systematic Deviations:**

| Parameter Regime | Observed Deviation | Magnitude | Possible Explanation |
|------------------|-------------------|-----------|---------------------|
| [Regime 1] | [Description] | [Value] | [Theory refinement needed] |
| [Regime 2] | [Description] | [Value] | [Experimental artifact] |
| [Regime 3] | [Description] | [Value] | [Unknown origin] |

### 2.4 Model Selection

If comparing multiple theoretical models:

| Model | Parameters | χ²/dof | AIC | BIC | Physical Basis |
|-------|------------|--------|-----|-----|----------------|
| Model A | $n$ | [value] | [value] | [value] | [Brief description] |
| Model B | $m$ | [value] | [value] | [value] | [Brief description] |
| Model C | $p$ | [value] | [value] | [value] | [Brief description] |

**Selection Criteria:**
- Primary: Physical plausibility
- Secondary: Goodness of fit (χ²/dof ≈ 1)
- Tertiary: Parsimony (prefer simpler models)

---

## Part III: Literature Comparison

### 3.1 State-of-the-Art Comparison

**Performance Benchmarks:**

| Metric | This Work | Best Prior Result | Reference | Conditions |
|--------|-----------|-------------------|-----------|------------|
| [Metric 1] | [Value] | [Value] | [Cite] | [Comparable?] |
| [Metric 2] | [Value] | [Value] | [Cite] | [Comparable?] |
| [Metric 3] | [Value] | [Value] | [Cite] | [Comparable?] |

**Comparability Assessment:**

For each comparison, assess whether conditions are truly comparable:

| Aspect | This Work | Reference [X] | Impact on Comparison |
|--------|-----------|---------------|---------------------|
| System | [Description] | [Description] | [Effect on metrics] |
| Temperature | [Value] | [Value] | [Effect on metrics] |
| Method | [Description] | [Description] | [Effect on metrics] |
| Calibration | [Standard] | [Standard] | [Effect on metrics] |

### 3.2 Research Group Comparison

**Comparison with Leading Groups:**

| Group | Institution | Approach | Best Result | Our Comparison |
|-------|-------------|----------|-------------|----------------|
| [Group A] | [University] | [Method] | [Result] | [Our value, context] |
| [Group B] | [Lab] | [Method] | [Result] | [Our value, context] |
| [Group C] | [University] | [Method] | [Result] | [Our value, context] |

### 3.3 Timeline Analysis

**Historical Progression:**

| Year | Best Reported Value | Reference | This Work Position |
|------|--------------------|-----------|--------------------|
| [Year-5] | [Value] | [Cite] | — |
| [Year-4] | [Value] | [Cite] | — |
| [Year-3] | [Value] | [Cite] | — |
| [Year-2] | [Value] | [Cite] | — |
| [Year-1] | [Value] | [Cite] | — |
| [Current] | [Best literature] | [Cite] | **[Our value]** |

**Trend Analysis:**
- Historical improvement rate: [X]% per year
- Our result vs. trend: [Above/At/Below] trend line by [factor]
- Projected timeline to reach [target]: [Years] at current rate

---

## Part IV: Synthesis Tables

### 4.1 Multi-Way Comparison Summary

| Aspect | R1 | R2 | Theory | Literature Best |
|--------|:--:|:--:|:------:|:--------------:|
| [Metric 1] | ○ | ● | ◐ | ○ |
| [Metric 2] | ○ | ● | ● | ◐ |
| [Metric 3] | ◐ | ● | ● | ● |

**Legend:**
- ● Best
- ◐ Intermediate
- ○ Baseline/Lower

### 4.2 SWOT Analysis of Your Results

**Strengths (vs. alternatives):**
1. [Strength 1]: [Quantitative advantage]
2. [Strength 2]: [Quantitative advantage]
3. [Strength 3]: [Quantitative advantage]

**Weaknesses (areas for improvement):**
1. [Weakness 1]: [Quantitative gap]
2. [Weakness 2]: [Quantitative gap]
3. [Weakness 3]: [Quantitative gap]

**Opportunities (potential applications):**
1. [Opportunity 1]: [Expected impact]
2. [Opportunity 2]: [Expected impact]
3. [Opportunity 3]: [Expected impact]

**Threats (challenges to validity/adoption):**
1. [Threat 1]: [Mitigation strategy]
2. [Threat 2]: [Mitigation strategy]
3. [Threat 3]: [Mitigation strategy]

---

## Part V: Narrative Integration

### 5.1 Comparison Paragraph Templates

**R1 vs R2 Narrative:**

> The [metric] achieved in Research Project 2 ($v_2 \pm \sigma_2$) represents a [ratio]-fold improvement over our earlier result in Research Project 1 ($v_1 \pm \sigma_1$). This enhancement arises from [primary factor], which [mechanism]. Additionally, [secondary factor] contributes approximately [percentage] of the improvement, as demonstrated by [control experiment/analysis].

**Theory vs Experiment Narrative:**

> The measured [observable] ($v_{\text{exp}} \pm \sigma_{\text{exp}}$) agrees with the theoretical prediction ($v_{\text{th}}$) to within [N]σ across the parameter range studied (Figure [X]). This agreement validates the [key assumption] underlying our theoretical model. At [specific condition], we observe a systematic deviation of [magnitude], which we attribute to [explanation]. This discrepancy suggests that [refinement needed].

**Literature Comparison Narrative:**

> Our result of [metric] = [value] compares favorably with the current state of the art, reported by [Group] as [value] in [system] [citation]. While our measurement is [higher/lower] by [factor], this comparison should be contextualized by noting that [difference in conditions]. When normalized for [factor], our result exceeds the prior benchmark by [amount].

### 5.2 Figures-Narrative Coordination

| Figure Number | Comparison Type | Key Message | Supporting Text Location |
|---------------|-----------------|-------------|-------------------------|
| Fig. [X] | R1 vs R2 | [Message] | Section [Y.Z], para [N] |
| Fig. [X] | Theory vs Exp | [Message] | Section [Y.Z], para [N] |
| Fig. [X] | Literature | [Message] | Section [Y.Z], para [N] |

---

## Usage Instructions

1. **Complete all applicable tables** with your actual data
2. **Calculate uncertainties** using proper error propagation
3. **Perform significance tests** where quantitative comparisons are made
4. **Create comparison figures** using provided templates
5. **Write narrative paragraphs** integrating quantitative results
6. **Review for balance** - acknowledge both strengths and limitations

---

*Comparative Analysis Template | Week 274 | Thesis Writing III*
