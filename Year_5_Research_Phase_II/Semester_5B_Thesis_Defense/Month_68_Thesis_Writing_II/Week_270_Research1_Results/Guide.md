# Week 270: Research 1 - Results

## Overview

This week focuses on transforming your paper's compressed results section into a comprehensive thesis chapter component. In journal publications, results are highly curated—you present only the key findings that support your narrative. In your thesis, you present the complete picture: all relevant findings, additional analyses, negative results, and the full statistical foundation of your work.

The results section is often where the paper-to-thesis transformation is most dramatic. A 3-4 page results section may expand to 15-22 pages. This expansion includes not just more text, but also larger figures, more detailed captions, additional data presentations, and thorough error analysis.

## Learning Objectives

By the end of this week, you will be able to:

1. Design a comprehensive results architecture that presents findings logically
2. Enhance paper figures for thesis format with improved size and detail
3. Present complete results including secondary and exploratory analyses
4. Document negative results and failed approaches appropriately
5. Provide rigorous statistical analysis and uncertainty quantification
6. Write self-contained figure captions that fully explain each visualization

## Day-by-Day Schedule

### Day 1884 (Monday): Results Architecture and Narrative Planning

**Morning (3 hours): Results Inventory**

Begin by cataloging all results from your research project—not just those in the paper:

*Complete Results Inventory:*

```
Results Inventory for Research Project 1

Category 1: Paper Results (Published)
- [Result 1]: [Brief description, paper figure reference]
- [Result 2]: [Brief description, paper figure reference]
- [Result 3]: [Brief description, paper figure reference]

Category 2: Supplementary Results (Published supplementary)
- [Result S1]: [Brief description]
- [Result S2]: [Brief description]

Category 3: Additional Results (Not published)
- [Result A1]: [Brief description, reason not included in paper]
- [Result A2]: [Brief description, reason not included in paper]

Category 4: Negative Results
- [Result N1]: [What was tried, why it didn't work]
- [Result N2]: [What was tried, why it didn't work]

Category 5: Exploratory Analyses
- [Analysis E1]: [Brief description, what was learned]
- [Analysis E2]: [Brief description, what was learned]

Category 6: Control Experiments
- [Control C1]: [Brief description]
- [Control C2]: [Brief description]
```

**Afternoon (3 hours): Results Narrative Design**

Organize your results into a coherent narrative structure:

*Narrative Architecture Framework:*

```
N.4 Results

N.4.1 Overview of Experimental/Computational Campaign
    - Summary of what was done
    - Organization of results section
    - Key findings preview

N.4.2 [Primary Result Category 1]
    N.4.2.1 [Sub-result 1a]
    N.4.2.2 [Sub-result 1b]
    N.4.2.3 [Sub-result 1c]

N.4.3 [Primary Result Category 2]
    N.4.3.1 [Sub-result 2a]
    N.4.3.2 [Sub-result 2b]

N.4.4 [Additional Analyses]
    N.4.4.1 [Secondary analysis 1]
    N.4.4.2 [Secondary analysis 2]

N.4.5 Negative Results and Failed Approaches
    N.4.5.1 [Failed approach 1]
    N.4.5.2 [Failed approach 2]

N.4.6 Summary of Key Findings
```

**Narrative Principles:**
1. **Logical Flow**: Results should build on each other
2. **Complete Story**: Include all relevant findings
3. **Honest Reporting**: Present failures alongside successes
4. **Statistical Rigor**: Every claim backed by analysis
5. **Clear Signposting**: Reader always knows where they are

**Evening (1 hour): Data Organization**

Organize your data files, analysis scripts, and figures for efficient access during writing.

### Day 1885 (Tuesday): Primary Results Presentation

**Morning (3 hours): Core Results Writing**

Transform your paper's key results into thesis format:

*Expansion Template for Each Result:*

Paper version (~100-200 words + small figure):
> "Figure 2 shows the gate fidelity as a function of gate duration. We observe optimal fidelity at τ = 340 ns, achieving F = 99.2 ± 0.1%."

Thesis version (~500-1000 words + enhanced figure):
> "N.4.2 Two-Qubit Gate Fidelity Optimization
>
> N.4.2.1 Gate Duration Dependence
>
> We investigated the dependence of CZ gate fidelity on gate duration τ to identify the optimal operating point. Figure N.3 presents the measured gate fidelity as a function of τ for the range 100-600 ns, encompassing the theoretically expected optimal range based on our system parameters (Section N.3.2).
>
> The data reveal several key features:
>
> **Fidelity-Duration Relationship**: Gate fidelity exhibits a clear maximum at τ = (340 ± 5) ns, where F = (99.2 ± 0.1)%. For shorter durations, fidelity is limited by [physical mechanism], while for longer durations, [different physical mechanism] dominates. This behavior is consistent with the theoretical model presented in Section N.2.3.
>
> **Quantitative Comparison with Theory**: The dashed line in Figure N.3 shows the predicted fidelity from our theoretical model (Equation N.7) with no free parameters. The model captures the qualitative behavior but slightly overestimates fidelity by ~0.2% on average, likely due to [systematic effect not captured in model].
>
> **Uncertainty Analysis**: Error bars represent 1σ statistical uncertainty from [N] repetitions per point. Systematic uncertainties from [sources] contribute an additional [X]% to the total uncertainty (see Table N.4).
>
> **Reproducibility**: This measurement was repeated on [N] different calibrations over [time period], with results consistent within statistical uncertainty (see Appendix N.C for full dataset).
>
> **Stability**: At the optimal duration, we monitored fidelity continuously for [time], observing a drift rate of [value] per hour (Figure N.4)."

**Afternoon (3 hours): Figure Enhancement**

Transform each paper figure into a thesis-quality figure:

*Figure Enhancement Checklist:*

For each figure, address:

1. **Size**
   - Paper: 3" × 3" typical
   - Thesis: 5" × 5" or larger as appropriate

2. **Resolution**
   - Ensure vector graphics or 300+ dpi
   - Font sizes readable at thesis size

3. **Content**
   - Add data points that may have been omitted
   - Include error bars if not present
   - Add theory comparison if available

4. **Annotation**
   - Larger, clearer axis labels
   - More tick marks and labels
   - Annotations pointing to key features
   - Legend with complete descriptions

5. **Color and Style**
   - Accessible color scheme (colorblind-friendly)
   - Consistent style with other thesis figures
   - Clear distinction between data series

*Enhanced Caption Writing:*

Paper caption:
> "Fig. 2. Gate fidelity vs. duration."

Thesis caption:
> "Figure N.3: Two-qubit controlled-Z (CZ) gate fidelity as a function of gate duration τ. Blue circles: measured fidelity from interleaved randomized benchmarking with K = 50 random sequences and N = 1000 shots per sequence. Error bars represent 1σ statistical uncertainty from bootstrap resampling. Dashed black line: theoretical prediction from Equation N.7 using independently measured system parameters (Table N.2) with no free fitting parameters. Gray shaded region: ±1σ theoretical uncertainty from parameter uncertainties. Vertical dashed line: optimal operating point at τ = 340 ns where maximum fidelity F = (99.2 ± 0.1)% is achieved. Data collected over calibration cycle [date], sample [ID], measurement conditions in Table N.3. The deviation between experiment and theory at short durations is attributed to finite rise-time effects not captured in the model (discussed in Section N.5.3). Inset: Expanded view of the region near optimal duration showing fine structure due to [physical effect]."

**Evening (1 hour): Caption Review**

Review all enhanced captions for completeness and consistency.

### Day 1886 (Wednesday): Secondary and Exploratory Results

**Morning (3 hours): Secondary Results Inclusion**

Present results that supported your paper but weren't featured:

*Categories of Secondary Results:*

1. **Validation Measurements**
   - Results confirming methodology works
   - Cross-checks between different measurement approaches
   - Calibration verification data

2. **Parameter Explorations**
   - Sensitivity analyses
   - Parameter sweeps beyond the optimal operating point
   - Edge cases and limiting behaviors

3. **Control Experiments**
   - Experiments confirming the effect is real
   - Ruling out alternative explanations
   - Baseline measurements

*Example Secondary Result Section:*

> "N.4.3.2 Validation Through Independent Measurement Techniques
>
> To validate the gate fidelity reported in Section N.4.2, we performed independent characterization using quantum process tomography (QPT). While less statistically robust than randomized benchmarking (RB), QPT provides complementary information about the nature of errors.
>
> Figure N.6 presents the reconstructed χ matrix for the CZ gate. The process fidelity Fp = Tr(χ_ideal · χ_meas) = (98.5 ± 0.3)% is consistent with the RB fidelity when accounting for state preparation and measurement (SPAM) errors:
>
> $$F_{RB} = F_p + (1 - F_p) \cdot F_{SPAM}$$
>
> Using independently measured SPAM fidelity F_SPAM = 0.995, we predict F_RB = 99.2%, in excellent agreement with the direct measurement.
>
> The χ matrix reveals the dominant error channel is [type], contributing approximately [X]% to the total error (Table N.6). This error channel is expected based on [physical mechanism] and is targeted for improvement in the next-generation device design (Chapter [N+1])."

**Afternoon (3 hours): Exploratory Analyses**

Document exploratory analyses not included in the paper:

*Exploratory Analysis Documentation Template:*

```
Exploratory Analysis: [Title]

Motivation:
[Why this analysis was performed]

Approach:
[What was done]

Results:
[What was found, with figures/data]

Interpretation:
[What this means, even if inconclusive]

Why Not in Paper:
[Reason for exclusion: space, tangential, inconclusive, etc.]

Value for Thesis:
[Why this is included in the thesis: completeness, insight, future work basis]
```

**Evening (1 hour): Exploratory Section Drafting**

Draft the exploratory results section with appropriate framing.

### Day 1887 (Thursday): Statistical Analysis and Uncertainty

**Morning (3 hours): Comprehensive Statistical Presentation**

Provide complete statistical underpinning for all results:

*Statistical Reporting Framework:*

1. **For Each Measurement:**
   - Sample size (N)
   - Central value (mean, median, MLE)
   - Uncertainty (standard error, confidence interval, credible interval)
   - Distribution characteristics if relevant

2. **For Each Comparison:**
   - Effect size
   - Statistical test used and justification
   - p-value or Bayes factor
   - Confidence in conclusion

3. **For Each Fit:**
   - Model equation
   - Fitted parameters with uncertainties
   - Goodness of fit metric
   - Residual analysis

*Example Statistical Table:*

| Quantity | Value | Stat. Unc. | Sys. Unc. | Combined | N | Method |
|----------|-------|------------|-----------|----------|---|--------|
| CZ fidelity | 99.2% | ±0.08% | ±0.05% | ±0.1% | 50K | IRB |
| Gate time | 340 ns | ±5 ns | ±2 ns | ±5 ns | 20 | Rabi |
| T₁ | 85 μs | ±3 μs | ±5 μs | ±6 μs | 100 | Exp fit |
| T₂ | 45 μs | ±2 μs | ±3 μs | ±4 μs | 100 | CPMG |

**Afternoon (3 hours): Uncertainty Quantification**

Document all sources of uncertainty:

*Uncertainty Budget Template:*

```
Uncertainty Budget for [Key Result]

Statistical Uncertainty:
- Measurement noise: [value], from [analysis]
- Fitting uncertainty: [value], from [analysis]
- Sampling uncertainty: [value], from [analysis]
Combined statistical: [value] by [combination method]

Systematic Uncertainty:
- Source 1 ([description]): [value], from [estimation method]
- Source 2 ([description]): [value], from [estimation method]
- Source 3 ([description]): [value], from [estimation method]
Combined systematic: [value] by [combination method]

Total Uncertainty:
Combined: [value] by [combination method]
Coverage factor: [k]
Reported uncertainty: [value] ([confidence level] confidence)

Discussion:
[Which uncertainties dominate, what could reduce them, how they compare
to similar measurements in the literature]
```

*Error Propagation Documentation:*

For derived quantities, document the error propagation:

$$\sigma_f^2 = \sum_i \left(\frac{\partial f}{\partial x_i}\right)^2 \sigma_{x_i}^2 + 2\sum_{i<j}\frac{\partial f}{\partial x_i}\frac{\partial f}{\partial x_j}\text{cov}(x_i, x_j)$$

**Evening (1 hour): Statistical Methods Review**

Ensure all statistical methods are correctly applied and documented.

### Day 1888 (Friday): Negative Results and Failed Approaches

**Morning (3 hours): Documenting Failures**

Papers rarely include negative results; theses should:

*Negative Results Framework:*

```
N.4.5 Negative Results and Failed Approaches

This section documents approaches that were investigated but did not yield
the expected results. These are included for completeness and to benefit
future researchers who may consider similar approaches.

N.4.5.1 [Failed Approach 1]: [Title]

Hypothesis:
[What we expected to work and why]

Approach:
[What we tried, with enough detail for replication]

Results:
[What we observed, with data if available]
[Figure N.X shows the unsuccessful result]

Analysis:
[Why we believe this approach failed]

Lessons Learned:
[What we learned that informed subsequent work]

N.4.5.2 [Failed Approach 2]: [Title]
[Same structure]
```

*Types of Negative Results to Include:*

1. **Failed Approaches**: Methods that didn't work as expected
2. **Null Results**: Experiments showing no effect where one was expected
3. **Unexpected Outcomes**: Results contrary to hypothesis
4. **Abandoned Directions**: Promising approaches not fully pursued

*Why Include Negative Results:*

| Reason | Benefit |
|--------|---------|
| Scientific honesty | Complete record of research |
| Help future researchers | Others won't waste time |
| Demonstrate thoroughness | Shows doctoral-level rigor |
| Inform interpretation | Context for positive results |
| Personal archive | Future reference for yourself |

**Afternoon (3 hours): Framing Negative Results**

Write about negative results constructively:

*Effective Negative Result Framing:*

Poor framing:
> "We tried approach X but it failed. We then moved on to approach Y."

Better framing:
> "Initial investigations explored approach X, motivated by [reasoning]. The results, presented in Figure N.12, did not show the expected [outcome]. Analysis of the data suggests [explanation for failure]. This insight informed our subsequent investigation of approach Y, which addresses the identified limitation by [modification]. The negative result from approach X also constrains the physical picture: [what we learned about the system]."

**Evening (1 hour): Negative Results Review**

Review negative results section for appropriate tone—honest but not apologetic.

### Day 1889 (Saturday): Figure Optimization and Caption Writing

**Morning (3 hours): Systematic Figure Enhancement**

Review all figures for consistency and quality:

*Figure Optimization Checklist:*

**Technical Quality:**
- [ ] Vector graphics or 300+ dpi raster
- [ ] Appropriate file format (PDF, PNG, EPS)
- [ ] No compression artifacts
- [ ] Fonts embedded (for vector)

**Visual Design:**
- [ ] Consistent color scheme across thesis
- [ ] Colorblind-accessible palette
- [ ] Clear contrast between elements
- [ ] Appropriate whitespace

**Scientific Content:**
- [ ] All data points shown
- [ ] Error bars included
- [ ] Theory/model comparison if applicable
- [ ] Scale bars where needed
- [ ] Units on all axes

**Labeling:**
- [ ] Axis labels complete with units
- [ ] Font size ≥ 10 pt at final size
- [ ] Legend clear and complete
- [ ] Panel labels (a), (b), etc. if applicable

**Consistency:**
- [ ] Same style as other thesis figures
- [ ] Consistent axis ranges where sensible
- [ ] Consistent symbol usage for same quantities

**Afternoon (3 hours): Comprehensive Caption Writing**

Write detailed, self-contained captions:

*Caption Components:*

1. **Figure title**: What the figure shows
2. **Visual description**: What each element represents
3. **Data source**: How the data was obtained
4. **Methods summary**: Key methodological details
5. **Error description**: What error bars/bands represent
6. **Sample/conditions**: Which sample, under what conditions
7. **Key observations**: What to notice (optional)
8. **Reference to text**: Where discussed in text (optional)

*Caption Length Guide:*

| Figure Type | Typical Caption Length |
|-------------|----------------------|
| Simple data plot | 3-5 lines |
| Multi-panel figure | 8-15 lines |
| Schematic/diagram | 5-10 lines |
| Complex methodology | 10-15 lines |

**Evening (1 hour): Figure Cross-Referencing**

Ensure all figures are referenced in text and numbering is consistent.

### Day 1890 (Sunday): Results Section Completion and Review

**Morning (2 hours): Section Assembly**

Compile all results into a coherent section:

1. Write section introduction (results overview)
2. Organize subsections in logical order
3. Write transitions between subsections
4. Write section summary

*Results Introduction Template:*

> "This section presents the experimental results from [brief description of research campaign]. Data were collected over [time period] using [brief methodology summary]. The results are organized as follows: Section N.4.2 presents [primary results]. Section N.4.3 addresses [secondary analyses]. Section N.4.4 documents [additional investigations]. Section N.4.5 presents negative results and failed approaches. The main findings are summarized in Section N.4.6 before detailed discussion in Section N.5."

**Afternoon (2 hours): Comprehensive Review**

Review the complete results section:

*Review Checklist:*

**Completeness:**
- [ ] All paper results included and expanded
- [ ] Secondary/exploratory results included
- [ ] Negative results documented
- [ ] All figures enhanced
- [ ] All tables complete

**Quality:**
- [ ] Every result has uncertainty
- [ ] Statistical methods documented
- [ ] Figures publication-quality
- [ ] Captions self-contained

**Flow:**
- [ ] Logical organization
- [ ] Clear transitions
- [ ] Consistent terminology
- [ ] No gaps in narrative

**Integration:**
- [ ] Connects to methods section
- [ ] Foreshadows discussion
- [ ] Consistent with thesis themes

**Evening (1 hour): Planning Next Week**

Prepare for discussion writing:
- Identify key interpretation points
- Gather literature for comparison
- List limitations to discuss
- Outline future directions

## Key Concepts

### Results Section Philosophy

The thesis results section serves multiple purposes:

1. **Complete Record**: Document all findings, not just highlights
2. **Reproducibility Support**: Others should be able to verify results
3. **Archival Function**: Future researchers can find detailed data
4. **Context for Discussion**: Provide foundation for interpretation
5. **Demonstration of Rigor**: Show doctoral-level thoroughness

### The Completeness Standard

Every result in your thesis should include:

| Element | Purpose |
|---------|---------|
| Data | What was measured/calculated |
| Uncertainty | How confident we are |
| Method | How it was obtained |
| Conditions | Under what circumstances |
| Context | How it relates to other results |
| Interpretation preview | What it means (briefly) |

### Figure Enhancement Philosophy

Thesis figures should be:
- **Self-contained**: Understandable without reading main text
- **Complete**: All relevant data shown
- **Clear**: Easy to read and interpret
- **Professional**: Publication quality
- **Accessible**: Usable by readers with visual impairments

## Common Challenges

### Challenge 1: "I don't have the original data/scripts"

**Solution**: Document what you have, note what's missing. If data can be reconstructed, do so. If not, acknowledge the limitation and present what's available with appropriate caveats.

### Challenge 2: "There's too much data to present"

**Solution**: Use hierarchical presentation. Main text has key results; appendices have complete datasets; supplementary materials (if allowed) have raw data.

### Challenge 3: "I'm not sure how to present negative results"

**Solution**: Frame constructively. Focus on what was learned, not on failure. Explain why the approach seemed reasonable, why it didn't work, and what this taught you.

### Challenge 4: "My statistics weren't rigorous enough"

**Solution**: If possible, redo analysis with proper methods. If not, acknowledge limitations explicitly. Better to be honest about limitations than to overclaim.

## Deliverables Checklist

### Results Section (~15-22 pages)
- [ ] Results overview and organization
- [ ] Primary results with complete presentation
- [ ] Secondary analyses and validations
- [ ] Exploratory results
- [ ] Negative results and failed approaches
- [ ] Results summary

### Figures
- [ ] All paper figures enhanced for thesis
- [ ] Additional figures for new content
- [ ] All captions complete and self-contained
- [ ] Consistent style throughout
- [ ] All figures referenced in text

### Tables
- [ ] Summary statistics tables
- [ ] Uncertainty budgets
- [ ] Parameter tables
- [ ] Comparison tables

### Quality Indicators
- [ ] Every result has uncertainty quantification
- [ ] Statistical methods appropriate and documented
- [ ] Figures meet publication quality
- [ ] Narrative is complete and logical

## Resources

### Scientific Figure Design
- Tufte, E. "The Visual Display of Quantitative Information"
- Rougier, N. et al. "Ten Simple Rules for Better Figures"
- Weissgerber, T. et al. "Beyond Bar and Line Graphs"

### Statistical Reporting
- Wasserstein, R. "The ASA Statement on p-Values"
- Cumming, G. "The New Statistics: Why and How"
- Morey, R. et al. "The Fallacy of Placing Confidence in Confidence Intervals"

### Uncertainty Quantification
- JCGM 100:2008 "Evaluation of Measurement Data — Guide to the Expression of Uncertainty in Measurement"
- Taylor, J. "An Introduction to Error Analysis"

## Looking Ahead

Next week (Week 271), you will write the discussion section. Prepare by:
- Reviewing literature for comparison
- Identifying key interpretation points
- Listing all limitations
- Outlining future directions
- Considering connections to thesis themes
