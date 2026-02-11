# Methodology Expansion Template

## Purpose

This template guides the systematic expansion of abbreviated paper methods into comprehensive thesis methodology documentation. The goal is to achieve the reproducibility standard: documentation sufficient for an independent researcher to replicate your work.

## Master Methodology Structure

```
Chapter N.3 Methodology

N.3.1 Methodological Overview
N.3.2 Theoretical Framework
N.3.3 Experimental Design / Computational Approach
N.3.4 Detailed Protocols
N.3.5 Equipment and Materials
N.3.6 Calibration and Validation
N.3.7 Data Collection
N.3.8 Data Processing and Analysis
N.3.9 Statistical Methods
N.3.10 Quality Control and Error Handling
```

---

## Section N.3.1: Methodological Overview

### Template

**Research Design Philosophy**

[Describe the overall methodological approach and why it was chosen]

This research employed a [quantitative/qualitative/mixed] methodology based on [experimental/computational/theoretical] approaches. The choice of methodology was driven by [research questions, practical constraints, state of the art].

**Key Methodological Decisions**

| Decision | Options Considered | Selection | Rationale |
|----------|-------------------|-----------|-----------|
| [Decision 1] | [Options] | [Choice] | [Why] |
| [Decision 2] | [Options] | [Choice] | [Why] |
| [Decision 3] | [Options] | [Choice] | [Why] |

**Methodological Workflow**

[Include workflow diagram showing major phases]

```
Phase 1: [Description]
    ↓
Phase 2: [Description]
    ↓
Phase 3: [Description]
    ↓
Phase 4: [Description]
```

**Relationship to Established Approaches**

[How your methodology relates to and differs from prior work]

---

## Section N.3.2: Theoretical Framework

### Template

**Underlying Theory**

[Present the theoretical basis for your experimental/computational approach]

The methodology is grounded in [theoretical framework], which describes [relevant physics/mathematics].

**Key Equations and Derivations**

[Provide full derivations, not just final results]

*Starting Point:*
$$
[Initial equation]
$$

*Derivation:*
[Step-by-step derivation with explanations]

*Result:*
$$
\boxed{[Final equation]}
$$

**Assumptions and Approximations**

| Assumption | Description | Validity Range | Impact if Violated |
|------------|-------------|----------------|-------------------|
| [Assumption 1] | [Description] | [When valid] | [What happens] |
| [Assumption 2] | [Description] | [When valid] | [What happens] |

**Theoretical Predictions**

Based on this framework, we expect:
- [Prediction 1]
- [Prediction 2]
- [Prediction 3]

---

## Section N.3.3: Experimental Design / Computational Approach

### Template (Experimental)

**Experimental Overview**

[Describe the overall experimental approach]

**Control Strategy**

| Variable | Type | How Controlled | Range/Value |
|----------|------|----------------|-------------|
| [Variable 1] | Independent | [Control method] | [Range] |
| [Variable 2] | Dependent | [Measurement] | [Expected range] |
| [Variable 3] | Controlled | [Control method] | [Fixed value] |

**Experimental Conditions**

- Temperature: [Value ± uncertainty]
- Pressure: [Value ± uncertainty]
- Magnetic field: [Value ± uncertainty]
- Other relevant conditions: [...]

**Measurement Strategy**

[Describe what is measured and how measurements relate to quantities of interest]

### Template (Computational)

**Computational Approach**

[Describe the overall computational methodology]

**Algorithm Selection**

| Task | Algorithm | Implementation | Justification |
|------|-----------|----------------|---------------|
| [Task 1] | [Algorithm] | [Library/custom] | [Why chosen] |
| [Task 2] | [Algorithm] | [Library/custom] | [Why chosen] |

**Computational Resources**

- Hardware: [Specifications]
- Software: [Languages, libraries, versions]
- Computational time: [Estimated for key tasks]

**Convergence and Accuracy**

[Describe convergence criteria and accuracy validation]

---

## Section N.3.4: Detailed Protocols

### Protocol Template

```
Protocol N.3.4.[X]: [Protocol Name]

Purpose:
[Clear statement of what this protocol accomplishes]

Prerequisites:
- [Required prior steps completed]
- [Equipment ready/configured]
- [Consumables available]

Materials Required:
- [Material 1]: [Specification, quantity]
- [Material 2]: [Specification, quantity]

Equipment Required:
- [Equipment 1]: [Model, configuration]
- [Equipment 2]: [Model, configuration]

Procedure:
1. [Step 1]
   - Substep 1a: [Details]
   - Substep 1b: [Details]
   Note: [Important considerations]

2. [Step 2]
   - [Details]
   Critical: [Safety or quality-critical information]

3. [Step 3]
   [...]

Expected Outcomes:
- [What success looks like]
- [Key parameters to verify]

Timing:
- Total duration: [Time]
- Time-critical steps: [Which steps have timing requirements]

Troubleshooting:
| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| [Issue 1] | [Cause] | [Fix] |
| [Issue 2] | [Cause] | [Fix] |

Notes and Tips:
- [Lesson learned 1]
- [Lesson learned 2]

Version History:
- v1.0 (Date): Initial protocol
- v1.1 (Date): [Changes and why]
```

---

## Section N.3.5: Equipment and Materials

### Equipment Documentation Template

```
Equipment: [Name]

Basic Information:
- Manufacturer: [Company]
- Model: [Model number]
- Serial number: [If tracked]
- Location: [Where housed]
- Acquisition date: [When acquired]

Specifications:
| Parameter | Specification | Our Measured Value |
|-----------|--------------|-------------------|
| [Param 1] | [Spec] | [Actual] |
| [Param 2] | [Spec] | [Actual] |

Configuration:
[Describe the specific configuration used in this research]

Role in Experiments:
[How this equipment is used in your protocols]

Calibration:
- Method: [How calibrated]
- Frequency: [How often]
- Standard: [Calibration standard used]
- Last calibration: [Date]
- Calibration records: [Where stored]

Maintenance:
- Regular maintenance: [What and when]
- Maintenance during research: [Any relevant events]

Known Issues and Limitations:
- [Issue 1 and how addressed]
- [Issue 2 and how addressed]

References:
- Manual: [Location]
- Technical notes: [Location]
```

### Materials Documentation Template

```
Material: [Name]

Identification:
- Chemical/physical description: [Description]
- Supplier: [Company]
- Product number: [Number]
- Lot/batch number: [Number(s) used]

Specifications:
- Purity: [Specification]
- Form: [Powder, liquid, etc.]
- Grade: [Research, semiconductor, etc.]

Handling:
- Storage conditions: [How stored]
- Safety considerations: [Relevant hazards]
- Preparation: [Any preparation required before use]

Purpose in Research:
[How this material is used]

Quality Verification:
[How purity/quality was verified]

Quantity Used:
[How much used in the research]
```

---

## Section N.3.6: Calibration and Validation

### Calibration Documentation Template

```
Calibration: [What is being calibrated]

Purpose:
[Why this calibration is necessary]

Standard:
- Type: [Primary/secondary/transfer standard]
- Traceability: [Traceability to national/international standards]
- Uncertainty: [Uncertainty of standard]

Procedure:
1. [Calibration step 1]
2. [Calibration step 2]
[...]

Acceptance Criteria:
- [Criterion 1]: [Threshold]
- [Criterion 2]: [Threshold]

Results:
| Parameter | Standard Value | Measured Value | Deviation |
|-----------|---------------|----------------|-----------|
| [Param] | [Value] | [Value] | [Dev] |

Calibration Curve/Correction:
[If applicable, include calibration curve or correction factors]

Calibration Interval:
- Nominal interval: [Frequency]
- Basis for interval: [Why this frequency]
- Events triggering recalibration: [What would require early recalibration]

Records:
- Storage location: [Where records kept]
- Record retention: [How long kept]
```

---

## Section N.3.7: Data Collection

### Data Collection Protocol Template

```
Data Collection Protocol: [Experiment Name]

Measurement Configuration:
[Describe the measurement setup]

Measurement Sequence:
1. [Initialize system]
2. [Prepare state/sample]
3. [Apply operation/wait]
4. [Measure observable]
5. [Record data]
6. [Repeat for statistics]

Parameters:
| Parameter | Value | Units | Notes |
|-----------|-------|-------|-------|
| Repetitions per point | [N] | - | [Statistical basis] |
| Points per sweep | [M] | - | [Resolution basis] |
| Sweep parameters | [List] | [Units] | [Ranges] |
| Total measurements | [Total] | - | - |

Timing:
- Time per measurement: [Duration]
- Time per sweep: [Duration]
- Total data collection time: [Duration]

Automation:
- Control software: [Name, version]
- Control scripts: [Location in repository]
- Human oversight: [What is monitored manually]

Real-time Quality Checks:
- [Check 1]: [What is checked, acceptance criterion]
- [Check 2]: [What is checked, acceptance criterion]

Data Recording:
- Format: [File format]
- Naming convention: [Convention explained]
- Storage location: [Where saved]
- Metadata recorded: [What metadata captured]
- Backup: [How and when backed up]
```

---

## Section N.3.8: Data Processing and Analysis

### Data Processing Pipeline Template

```
Data Processing Pipeline

Stage 0: Raw Data Ingestion
- Input: [Raw data format and source]
- Processing: [How data is loaded and organized]
- Output: [Organized data structure]

Stage 1: Preprocessing
- Input: [Data from Stage 0]
- Processing:
  - [Step 1]: [Description, parameters]
  - [Step 2]: [Description, parameters]
- Output: [Preprocessed data]
- Validation: [How preprocessing verified]

Stage 2: Primary Analysis
- Input: [Preprocessed data]
- Processing:
  - [Analysis method 1]: [Description, parameters]
  - [Analysis method 2]: [Description, parameters]
- Output: [Primary results]
- Validation: [How analysis verified]

Stage 3: Error Analysis
- Input: [Primary results]
- Processing:
  - [Error propagation method]
  - [Uncertainty quantification]
- Output: [Results with uncertainties]

Stage 4: Derived Quantities
- Input: [Primary results with uncertainties]
- Processing: [Calculations of derived quantities]
- Output: [Final results]

Software Implementation:
- Language: [Python, Julia, etc.]
- Key libraries: [List with versions]
- Repository: [Location]
- Key scripts: [List of main analysis scripts]

Reproducibility:
- Random seeds: [How handled]
- Version control: [How versions tracked]
- Environment: [How environment specified]
```

---

## Section N.3.9: Statistical Methods

### Statistical Analysis Template

```
Statistical Methods

Descriptive Statistics:
- Central tendency: [Mean, median, mode as appropriate]
- Dispersion: [Standard deviation, IQR, etc.]
- Distribution characterization: [How distributions assessed]

Inferential Statistics:
- Hypothesis testing: [Tests used and why]
- Confidence intervals: [How computed, confidence level]
- Multiple comparison correction: [If applicable, method used]

Estimation Methods:
| Quantity | Estimation Method | Justification |
|----------|------------------|---------------|
| [Param 1] | [Method] | [Why appropriate] |
| [Param 2] | [Method] | [Why appropriate] |

Uncertainty Quantification:
- Type A (statistical): [How computed]
- Type B (systematic): [How estimated]
- Combined uncertainty: [How combined]
- Reporting convention: [Standard deviation, confidence interval, etc.]

Fitting and Modeling:
- Fitting method: [Least squares, MLE, Bayesian, etc.]
- Goodness of fit: [Metrics used]
- Model selection: [If multiple models, how selected]

Software:
- Statistical software: [Packages used]
- Validation: [How statistical calculations verified]
```

---

## Section N.3.10: Quality Control and Error Handling

### Quality Control Template

```
Quality Control Procedures

Pre-experiment Verification:
- [ ] [Check 1]: [Criterion]
- [ ] [Check 2]: [Criterion]
- [ ] [Check 3]: [Criterion]

During Experiment Monitoring:
| Parameter | Acceptable Range | Action if Outside |
|-----------|-----------------|-------------------|
| [Param 1] | [Range] | [Action] |
| [Param 2] | [Range] | [Action] |

Post-experiment Validation:
- [ ] [Validation 1]: [Criterion]
- [ ] [Validation 2]: [Criterion]

Error Handling:
| Error Type | Detection | Response |
|------------|-----------|----------|
| [Error 1] | [How detected] | [What to do] |
| [Error 2] | [How detected] | [What to do] |

Data Exclusion Criteria:
[Criteria for excluding data points or runs, with justification]

Documentation:
- Anomaly log: [How anomalies recorded]
- Deviation reports: [How deviations documented]
```

---

## Expansion Workflow

### Step 1: Inventory Paper Methods

List every method mentioned in your paper:
- [ ] [Method 1]
- [ ] [Method 2]
- [ ] [Method 3]

### Step 2: Identify Expansion Needs

For each method:
| Method | Paper Coverage | Needs Expansion | Priority |
|--------|----------------|-----------------|----------|
| [Method 1] | [Brief/Medium/Good] | [Yes/No] | [High/Med/Low] |

### Step 3: Gather Source Materials

For expanded documentation:
- [ ] Lab notebooks
- [ ] Protocol documents
- [ ] Equipment manuals
- [ ] Calibration records
- [ ] Software documentation
- [ ] Email/meeting notes

### Step 4: Write Section Drafts

Following templates above, draft each section

### Step 5: Review for Reproducibility

Ask: Could someone replicate this from my thesis alone?

### Step 6: Advisor Review

Submit methodology section for advisor feedback

---

## Quality Checklist

### Completeness
- [ ] All methods from paper documented
- [ ] All parameters listed with values
- [ ] All equipment specified
- [ ] All materials documented
- [ ] All calibrations described
- [ ] All analysis steps explained

### Reproducibility
- [ ] Protocols detailed enough for replication
- [ ] Software versions specified
- [ ] Code accessible
- [ ] Data formats documented
- [ ] Analysis reproducible

### Clarity
- [ ] Methods organized logically
- [ ] Technical terms defined
- [ ] Figures/diagrams aid understanding
- [ ] Cross-references included

### Integration
- [ ] Connects to theoretical framework
- [ ] Links to results section
- [ ] Consistent terminology
- [ ] Consistent notation
