# Week 218: Initial Investigation

## Overview

**Days:** 1520-1526
**Theme:** First Experiments and Simulations
**Goal:** Execute systematic initial investigations to generate first real data

---

## Week Purpose

Week 218 transitions from implementation to active investigation. With your core methodology implemented and validated, you now conduct your first systematic experiments or simulations. This week generates your initial dataset and provides crucial feedback on methodology effectiveness, identifying both strengths and areas requiring refinement.

### Learning Objectives

By the end of this week, you will:

1. Execute systematic initial experiments/simulations with proper controls
2. Collect high-quality data with complete documentation
3. Identify methodology strengths and limitations through practical application
4. Develop intuition for system behavior through hands-on investigation
5. Establish data management and quality control practices

---

## Daily Structure

### Day 1520 (Monday): Investigation Planning

**Morning (3 hours):**
- Review implementation from Week 217
- Define specific investigation questions
- Plan experiment/simulation sequence
- Identify key parameters to vary

**Afternoon (4 hours):**
- Prepare experimental protocols
- Set up data collection infrastructure
- Configure analysis pipelines
- Run pilot experiments

**Evening (2 hours):**
- Review pilot results
- Adjust protocols if needed
- Finalize investigation plan
- Begin experiment log

### Day 1521 (Tuesday): First Systematic Investigation

**Morning (3 hours):**
- Execute first experiment set
- Monitor for unexpected behavior
- Record all observations
- Preliminary data quality checks

**Afternoon (4 hours):**
- Continue systematic data collection
- Maintain real-time documentation
- Address any technical issues
- Save data with complete metadata

**Evening (2 hours):**
- Quick-look analysis of day's data
- Update experiment log
- Plan adjustments for Day 2

### Day 1522 (Wednesday): Extended Data Collection

**Morning (3 hours):**
- Continue systematic investigation
- Expand parameter range
- Test boundary conditions
- Collect replication data

**Afternoon (4 hours):**
- Complete planned experiment sequence
- Additional measurements for interesting features
- Data backup and organization
- Initial pattern recognition

**Evening (2 hours):**
- Compile day's results
- Preliminary trend identification
- Document unexpected observations
- Literature check for similar findings

### Day 1523 (Thursday): Control Experiments

**Morning (3 hours):**
- Design control experiments
- Identify potential confounding factors
- Execute control measurements
- Compare with primary results

**Afternoon (4 hours):**
- Complete control experiment set
- Assess systematic uncertainties
- Evaluate noise characteristics
- Test reproducibility

**Evening (2 hours):**
- Analyze control experiment results
- Update uncertainty estimates
- Document control methodology
- Assess data reliability

### Day 1524 (Friday): Investigation Completion

**Morning (3 hours):**
- Final data collection runs
- Fill gaps in parameter space
- High-priority repeat measurements
- Complete primary dataset

**Afternoon (4 hours):**
- Data quality assessment
- Complete metadata documentation
- Organize raw data archive
- Begin preliminary analysis

**Evening (2 hours):**
- Week progress review
- Compile all observations
- Identify key findings
- Advisor check-in preparation

### Day 1525 (Saturday): Data Organization

**Morning (3 hours):**
- Comprehensive data organization
- Create analysis-ready datasets
- Document data processing steps
- Verify data integrity

**Afternoon (3 hours):**
- Preliminary visualization
- Identify analysis priorities
- Document initial observations
- Prepare for Week 219 analysis

**Evening (1 hour):**
- Rest and reflection
- Light reading on analysis methods

### Day 1526 (Sunday): Review and Planning

**Morning (2 hours):**
- Review week's accomplishments
- Assess data quality and completeness
- Identify gaps requiring additional investigation
- Document lessons learned

**Afternoon (2 hours):**
- Plan Week 219 analysis approach
- Prioritize analysis tasks
- Prepare analysis environment
- Light preparation work

**Evening (1 hour):**
- Rest
- Informal thinking about results

---

## Investigation Framework

### Pre-Investigation Checklist

Before beginning experiments:

- [ ] Implementation validated against benchmarks
- [ ] Data collection infrastructure tested
- [ ] Storage capacity confirmed
- [ ] Backup procedures established
- [ ] Experiment log template ready
- [ ] Analysis scripts prepared
- [ ] Advisor consulted on investigation plan

### Parameter Space Definition

**Primary Parameters (Tier 1):**
| Parameter | Range | Resolution | Units | Rationale |
|-----------|-------|------------|-------|-----------|
| [Param 1] | [min-max] | [step] | | |
| [Param 2] | [min-max] | [step] | | |

**Secondary Parameters (Tier 2):**
| Parameter | Nominal Value | Variation Range | Purpose |
|-----------|--------------|-----------------|---------|
| | | | |

**Fixed Parameters (Tier 3):**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| | | |

### Investigation Sequence

**Phase 1: Baseline Establishment (Days 1520-1521)**
- Nominal parameter configuration
- Multiple replications
- Establish measurement precision
- Identify systematic variations

**Phase 2: Parameter Sweeps (Days 1521-1522)**
- Vary primary parameters
- Coarse resolution first
- Identify regions of interest
- Document transitions and anomalies

**Phase 3: Control Experiments (Day 1523)**
- Null experiments
- Known reference cases
- Systematic error characterization
- Noise floor measurements

**Phase 4: Refinement (Days 1524-1525)**
- Fine resolution in interesting regions
- Replications for key measurements
- Gap filling
- Final dataset completion

---

## Data Collection Standards

### File Naming Convention

```
YYYYMMDD_HHMMSS_[experiment_type]_[parameter_set]_[run_number].ext
```

**Example:**
```
20260215_143022_gate_fidelity_amp_sweep_001.npz
```

### Metadata Requirements

Every data file must include:

```yaml
metadata:
  timestamp: "YYYY-MM-DD HH:MM:SS"
  researcher: "[Name]"
  experiment_type: "[Type]"

  parameters:
    [param1]: [value]
    [param2]: [value]
    # ... all parameters

  equipment:
    [device1]: [serial/ID]
    [device2]: [serial/ID]

  conditions:
    temperature: [value]
    [other]: [value]

  notes: "[Any relevant observations]"
```

### Data Quality Checks

**Immediate Checks (during collection):**
- [ ] File saved successfully
- [ ] Expected data shape/size
- [ ] Values within physical range
- [ ] No obvious artifacts

**End-of-Day Checks:**
- [ ] All files readable
- [ ] Metadata complete
- [ ] Backup completed
- [ ] Log entries complete

**End-of-Week Checks:**
- [ ] Full dataset inventory
- [ ] Cross-reference with experiment log
- [ ] Missing data identified
- [ ] Quality assessment documented

---

## Experiment Types

### For Computational Projects

#### Simulation Sweep
```python
# Example simulation sweep structure
sweep_parameters = {
    'coupling_strength': np.linspace(0.01, 1.0, 20),
    'disorder_strength': np.linspace(0.0, 0.5, 10)
}

results = {}
for g in sweep_parameters['coupling_strength']:
    for W in sweep_parameters['disorder_strength']:
        config = {'g': g, 'W': W, 'seed': 42}
        result = run_simulation(config)
        results[(g, W)] = result
        save_result(result, config)
```

#### Convergence Study
- Vary discretization/resolution parameter
- Assess convergence of observables
- Estimate systematic errors
- Document convergence criteria

#### Statistical Sampling
- Multiple random realizations
- Disorder averaging
- Bootstrap uncertainty estimation
- Sample size sufficiency tests

### For Experimental Projects

#### Baseline Measurements
- Reference samples/signals
- Calibration verification
- Noise floor characterization
- Stability assessment

#### Parameter Sweeps
- Single parameter variations
- Hysteresis tests (up vs. down)
- Rate-dependent effects
- Equilibration time verification

#### Reproducibility Tests
- Repeat measurements
- Day-to-day variations
- Sample-to-sample variations (if applicable)
- Operator-to-operator variations

### For Theoretical Projects

#### Analytical Verification
- Compare derivations with known limits
- Verify special cases
- Test against numerical results
- Consistency checks across methods

#### Parameter Regime Exploration
- Perturbative regime
- Strong coupling regime
- Transition regions
- Asymptotic behavior

---

## Troubleshooting Guide

### Issue: Unexpected Results

**Assessment:**
1. Are results reproducible?
2. Are parameters set correctly?
3. Are there equipment/code issues?
4. Is this a genuine new finding?

**Response:**
1. Document thoroughly before changing anything
2. Check against known benchmarks
3. Consult with advisor before concluding it's wrong
4. Consider if it could be correct but unexpected

### Issue: High Noise/Variability

**Assessment:**
1. Is noise statistical or systematic?
2. What is the noise spectrum?
3. Are there environmental factors?
4. Is averaging sufficient?

**Response:**
1. Characterize noise properties
2. Increase sample size if statistical
3. Identify and control if systematic
4. Consider noise reduction techniques

### Issue: Data Collection Failures

**Assessment:**
1. Is it hardware or software?
2. Is it intermittent or persistent?
3. What was the last successful run?
4. What changed?

**Response:**
1. Document the failure mode
2. Check logs and error messages
3. Restart from known good state
4. Escalate if not resolved quickly

### Issue: Slower Than Expected

**Assessment:**
1. Is the slowdown acceptable?
2. What's the bottleneck?
3. Can parameters be adjusted?
4. Is the timeline still feasible?

**Response:**
1. Profile to find bottleneck
2. Prioritize critical measurements
3. Consider approximations
4. Adjust scope if necessary

---

## Documentation Standards

### Experiment Log Format

Each experiment should have:

1. **Header:**
   - Date, time, experiment ID
   - Purpose and hypothesis
   - Connection to research questions

2. **Setup:**
   - Complete parameter list
   - Equipment/software configuration
   - Deviations from protocol

3. **Execution:**
   - Time log of activities
   - Real-time observations
   - Issues encountered

4. **Results:**
   - Data file references
   - Quick-look analysis
   - Immediate conclusions

5. **Follow-up:**
   - Questions raised
   - Next steps
   - Modifications for future

### Observation Categories

**Expected observations:** Results matching predictions
**Anomalies:** Unexpected but potentially explicable
**Surprises:** Genuinely unexpected findings
**Failures:** Equipment/method failures
**Insights:** New understanding or ideas

---

## Deliverables Checklist

### Required Deliverables

- [ ] Complete raw dataset with metadata
- [ ] Comprehensive experiment log (daily entries)
- [ ] Data quality assessment report
- [ ] Preliminary observations document
- [ ] Data backup in 2+ locations

### Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Experiments completed | 5-10 distinct sets | Count |
| Data completeness | >95% planned points | Audit |
| Metadata coverage | 100% files | Audit |
| Documentation currency | <1 day delay | Log review |
| Backup frequency | Daily | Log review |

---

## Success Indicators

### Strong Progress Signs
- Data collection proceeding smoothly
- Reproducible measurements
- Interesting patterns emerging
- Clear understanding of system behavior
- Documentation current and complete

### Warning Signs
- Frequent unexpected failures
- Non-reproducible results
- No clear patterns visible
- Documentation falling behind
- Growing list of unexplained issues

---

## Resources

### Data Management
- "Best Practices for Scientific Data Management"
- FAIR Principles documentation
- Your institution's data policies

### Experimental Methods
- Domain-specific protocols and guides
- Equipment manuals and documentation
- Relevant literature methods sections

### Analysis Preparation
- Python scientific stack documentation
- Statistical analysis references
- Visualization best practices

---

## Notes

This week is about generating high-quality data, not analyzing it deeply. Focus on:
1. **Completeness:** Cover the planned parameter space
2. **Quality:** Every measurement should be reliable
3. **Documentation:** Everything should be traceable
4. **Observation:** Notice and record unexpected findings

Analysis will come in Week 219. Resist the temptation to over-analyze during data collection - it can bias your measurements and distract from thorough data collection.

**Week Mantra:** "Good data now enables good science later."

---

*Week 218 of the QSE Self-Study Curriculum*
*Month 55: Research Execution I*
