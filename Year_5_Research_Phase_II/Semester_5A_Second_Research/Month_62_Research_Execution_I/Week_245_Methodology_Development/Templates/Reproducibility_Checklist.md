# Reproducibility Checklist for Quantum Research

## Project Information

| Field | Entry |
|-------|-------|
| Project Title | |
| Researcher | |
| Date Completed | |
| Checklist Version | 1.0 |

---

## Instructions

Complete this checklist before beginning primary data collection. Each item should be checked only when fully implemented. Items marked with (*) are critical and must be completed before proceeding.

---

## 1. Documentation (*)

### 1.1 Methodology Documentation

- [ ] (*) Complete methodology document written and version-controlled
- [ ] (*) All assumptions explicitly stated
- [ ] (*) Research questions and hypotheses documented
- [ ] Theoretical background summarized with key references
- [ ] Methodology approved by advisor

**Notes:**
```
[Document location and version]
```

### 1.2 Protocol Documentation

- [ ] (*) All experimental/computational protocols written
- [ ] (*) Decision trees for conditional procedures documented
- [ ] Troubleshooting guides prepared
- [ ] Protocols peer-reviewed
- [ ] Protocol versions tracked

**Protocol Registry:**

| Protocol ID | Name | Version | Status |
|-------------|------|---------|--------|
| | | | |

### 1.3 Parameter Documentation

- [ ] (*) All parameters with values and units documented
- [ ] Parameter ranges and justifications provided
- [ ] Default values distinguished from experimental values
- [ ] Parameter sensitivity documented

---

## 2. Environment Specification (*)

### 2.1 Hardware Environment

- [ ] (*) Computing hardware fully specified
- [ ] (*) Experimental apparatus documented
- [ ] Manufacturer, model, and serial numbers recorded
- [ ] Calibration certificates archived

**Hardware Inventory:**

| Item | Specification | Serial/ID | Calibration Date |
|------|---------------|-----------|------------------|
| | | | |

### 2.2 Software Environment

- [ ] (*) Operating system and version documented
- [ ] (*) All software packages with exact versions listed
- [ ] (*) Environment file created (conda/pip/docker)
- [ ] Custom code version-controlled
- [ ] Build/installation instructions provided

**Environment Specification:**

```yaml
# environment.yml
name:
channels:
  -
dependencies:
  -
```

### 2.3 Quantum-Specific Environment

- [ ] Quantum hardware specifications documented (if applicable)
- [ ] Quantum software framework versions recorded
- [ ] Backend configuration parameters saved
- [ ] Noise model specifications documented (if simulated)

---

## 3. Data Management (*)

### 3.1 Data Organization

- [ ] (*) Directory structure established
- [ ] (*) Naming conventions defined and documented
- [ ] (*) Raw data designated as immutable
- [ ] Processed data separated from raw data
- [ ] Results directory prepared

**Directory Structure:**

```
project/
├── data/
│   ├── raw/          # Immutable
│   ├── processed/
│   └── results/
├── code/
├── docs/
└── configs/
```

### 3.2 Data Integrity

- [ ] (*) Checksums/hashes for raw data files
- [ ] Data validation procedures defined
- [ ] Corruption detection mechanisms in place
- [ ] Data format specifications documented

### 3.3 Backup and Archive

- [ ] (*) Primary backup location established
- [ ] (*) Backup frequency defined
- [ ] Secondary backup available
- [ ] Long-term archive plan defined
- [ ] Backup restoration tested

**Backup Schedule:**

| Data Type | Backup Frequency | Location | Last Tested |
|-----------|------------------|----------|-------------|
| | | | |

### 3.4 Data Access

- [ ] Access permissions documented
- [ ] Data sharing policy defined
- [ ] Sensitive data protection measures in place
- [ ] Data availability plan for publication

---

## 4. Version Control (*)

### 4.1 Code Version Control

- [ ] (*) Git repository initialized
- [ ] (*) .gitignore properly configured
- [ ] Branching strategy defined
- [ ] Commit message conventions established
- [ ] Tags used for major versions/releases

### 4.2 Document Version Control

- [ ] Methodology documents version-controlled
- [ ] Protocol versions tracked
- [ ] Change log maintained
- [ ] Major revisions documented

### 4.3 Data Version Control

- [ ] Data versioning strategy defined
- [ ] Links between code versions and data versions maintained
- [ ] Derived data versions tracked
- [ ] Analysis versions linked to data versions

---

## 5. Computational Reproducibility (*)

### 5.1 Random Number Generation

- [ ] (*) Random seeds recorded for all stochastic processes
- [ ] Random number generator specified
- [ ] Seed management documented in code
- [ ] Reproducibility of random processes verified

**Seed Registry:**

| Process | Seed Value | Location in Code |
|---------|------------|------------------|
| | | |

### 5.2 Numerical Precision

- [ ] Floating-point precision documented
- [ ] Numerical tolerances specified
- [ ] Convergence criteria defined
- [ ] Platform-specific numerical behavior documented

### 5.3 Parallel Computation

- [ ] Parallelization strategy documented
- [ ] Non-determinism from parallelism addressed
- [ ] Number of threads/processes recorded
- [ ] Load balancing approach documented

---

## 6. Experimental Reproducibility (if applicable)

### 6.1 Calibration

- [ ] (*) Calibration procedures documented
- [ ] (*) Calibration data archived
- [ ] Calibration frequency defined
- [ ] Calibration drift monitoring in place
- [ ] Re-calibration triggers defined

### 6.2 Environmental Conditions

- [ ] (*) Critical environmental parameters identified
- [ ] Environmental monitoring in place
- [ ] Acceptable ranges defined
- [ ] Environmental data logged

**Environmental Parameters:**

| Parameter | Target | Tolerance | Monitoring |
|-----------|--------|-----------|------------|
| | | | |

### 6.3 Sample/System Preparation

- [ ] Preparation procedures documented
- [ ] Sample/system provenance recorded
- [ ] Preparation variations documented
- [ ] Quality assurance for preparation defined

---

## 7. Quality Assurance (*)

### 7.1 Validation

- [ ] (*) Validation experiments/computations designed
- [ ] Known results/benchmarks identified
- [ ] Validation acceptance criteria defined
- [ ] Validation results documented

### 7.2 Quality Metrics

- [ ] (*) Quality metrics defined
- [ ] Acceptance thresholds established
- [ ] Quality monitoring procedures in place
- [ ] Quality deviations documented

### 7.3 Error Handling

- [ ] Error detection mechanisms implemented
- [ ] Error logging active
- [ ] Error recovery procedures defined
- [ ] Anomaly detection in place

---

## 8. Provenance Tracking (*)

### 8.1 Data Provenance

- [ ] (*) All data transformations logged
- [ ] Input-output relationships documented
- [ ] Transformation parameters recorded
- [ ] Intermediate results preserved (as needed)

### 8.2 Analysis Provenance

- [ ] (*) Analysis scripts version-controlled
- [ ] Analysis parameters recorded
- [ ] Analysis execution logged
- [ ] Results linked to analysis versions

### 8.3 Result Provenance

- [ ] (*) Each result traceable to raw data
- [ ] Complete provenance chain documented
- [ ] Provenance verification possible
- [ ] Provenance metadata with results

---

## 9. Reporting Standards

### 9.1 Uncertainty Reporting

- [ ] Uncertainty quantification methods documented
- [ ] Error bars/intervals defined
- [ ] Statistical significance thresholds set
- [ ] Uncertainty propagation documented

### 9.2 Figure Standards

- [ ] Figure generation scripts version-controlled
- [ ] Figure parameters recorded
- [ ] Publication-quality format defined
- [ ] Figure data archived

### 9.3 Statistical Reporting

- [ ] Statistical tests documented
- [ ] Effect sizes reported
- [ ] Multiple comparison corrections documented
- [ ] Null results reported

---

## 10. Pre-Registration (Optional but Recommended)

### 10.1 Hypothesis Pre-Registration

- [ ] Hypotheses registered before data collection
- [ ] Analysis plan pre-registered
- [ ] Registration timestamped
- [ ] Deviations from pre-registration documented

### 10.2 Protocol Pre-Registration

- [ ] Protocols registered
- [ ] Stopping criteria pre-defined
- [ ] Sample size justified in advance
- [ ] Protocol modifications documented

---

## Completion Summary

| Section | Items Complete | Items Total | Status |
|---------|---------------|-------------|--------|
| 1. Documentation | | | |
| 2. Environment | | | |
| 3. Data Management | | | |
| 4. Version Control | | | |
| 5. Computational | | | |
| 6. Experimental | | | |
| 7. Quality Assurance | | | |
| 8. Provenance | | | |
| 9. Reporting | | | |
| 10. Pre-Registration | | | |
| **TOTAL** | | | |

---

## Sign-Off

**Researcher Declaration:**

I confirm that all items marked with (*) have been completed and that I have made reasonable efforts to complete all other applicable items.

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Researcher | | | |
| Advisor Review | | | |

---

## Notes and Comments

```
[Additional notes on reproducibility measures]
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | | Initial completion |
| | | |
