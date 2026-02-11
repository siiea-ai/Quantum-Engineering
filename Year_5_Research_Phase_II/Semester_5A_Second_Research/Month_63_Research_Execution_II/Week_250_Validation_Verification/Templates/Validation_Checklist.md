# Validation Checklist Template

## Research Validation Protocol

---

## Project Information

| Field | Details |
|-------|---------|
| **Project Title** | |
| **Validation Period** | Week 250 (Days 1744-1750) |
| **Principal Investigator** | |
| **Date Started** | |
| **Date Completed** | |

---

## Claim Inventory

### Claim Registry

List all claims requiring validation:

| ID | Claim Statement | Type | Priority | Status |
|----|-----------------|------|----------|--------|
| C1 | | Theorem / Bound / Algorithm / Numerical | Critical / High / Medium | Not Started / In Progress / Complete |
| C2 | | | | |
| C3 | | | | |
| C4 | | | | |
| C5 | | | | |

---

## Validation Methods Checklist

### Method 1: Independent Re-Derivation

| Claim | Re-Derivation Attempted | Matches Original | Notes |
|-------|------------------------|------------------|-------|
| C1 | [ ] Yes [ ] No [ ] N/A | [ ] Yes [ ] No [ ] Partial | |
| C2 | [ ] | [ ] | |
| C3 | [ ] | [ ] | |
| C4 | [ ] | [ ] | |
| C5 | [ ] | [ ] | |

**Re-Derivation Details:**

**Claim C1:**
- Alternative approach used:
- Key differences from original:
- Result:

**Claim C2:**
- Alternative approach used:
- Key differences from original:
- Result:

---

### Method 2: Numerical Verification

| Claim | Tests Run | Tests Passed | Pass Rate | Notes |
|-------|-----------|--------------|-----------|-------|
| C1 | | | % | |
| C2 | | | % | |
| C3 | | | % | |
| C4 | | | % | |
| C5 | | | % | |

**Test Case Categories:**

| Category | Number of Tests | Purpose |
|----------|-----------------|---------|
| Random | | General verification |
| Structured | | Known cases |
| Edge cases | | Boundary conditions |
| Adversarial | | Stress testing |

**Numerical Test Configuration:**

- Tolerance: _______________
- Random seed: _______________
- Number of dimensions tested: _______________
- Parameter ranges: _______________

---

### Method 3: Edge Case Testing

**Edge Cases Identified:**

| ID | Edge Case Description | Expected Behavior | Actual Behavior | Pass? |
|----|----------------------|-------------------|-----------------|-------|
| E1 | | | | [ ] |
| E2 | | | | [ ] |
| E3 | | | | [ ] |
| E4 | | | | [ ] |
| E5 | | | | [ ] |

**Edge Case Coverage by Claim:**

| Claim | Edge Cases Tested | All Passed? | Notes |
|-------|-------------------|-------------|-------|
| C1 | E__, E__, E__ | [ ] | |
| C2 | | [ ] | |
| C3 | | [ ] | |

---

### Method 4: Adversarial Testing

| Claim | Adversarial Method | Attempts | Counterexamples Found | Notes |
|-------|-------------------|----------|----------------------|-------|
| C1 | | | | |
| C2 | | | | |
| C3 | | | | |

**Adversarial Methods Used:**
- [ ] Optimization-based search
- [ ] Genetic algorithm search
- [ ] Random adversarial generation
- [ ] Manual construction
- [ ] Other: _______________

**Counterexamples Analysis:**

| Counterexample | Claim Affected | Resolution |
|----------------|----------------|------------|
| | | [ ] Claim modified [ ] Edge case documented [ ] False positive |

---

### Method 5: Limiting Case Verification

| Claim | Limiting Case | Expected Limit | Computed Limit | Match? |
|-------|---------------|----------------|----------------|--------|
| C1 | d → ∞ | | | [ ] |
| C1 | ε → 0 | | | [ ] |
| C2 | | | | [ ] |
| C3 | | | | [ ] |

---

### Method 6: Literature Comparison

| Claim | Related Result | Reference | Comparison | Consistent? |
|-------|---------------|-----------|------------|-------------|
| C1 | | [Author, Year] | | [ ] |
| C2 | | | | [ ] |
| C3 | | | | [ ] |

**Discrepancy Analysis:**

| Discrepancy | Explanation | Resolution |
|-------------|-------------|------------|
| | | |

---

### Method 7: Peer Review

| Claim | Reviewer | Date | Issues Found | Resolution |
|-------|----------|------|--------------|------------|
| C1 | | | | |
| C2 | | | | |

---

## Precision Analysis

### Numerical Precision Requirements

| Computation | Minimum Precision | Recommended Precision | Justification |
|-------------|-------------------|----------------------|---------------|
| | | | |
| | | | |

### Condition Number Analysis

| Matrix/Operator | Condition Number | Precision Impact |
|-----------------|------------------|------------------|
| | | |
| | | |

### Precision Verification

- [ ] Tested with float32
- [ ] Tested with float64
- [ ] Tested with extended precision (if applicable)
- [ ] Precision requirements documented

---

## Reproducibility Checklist

### Code

- [ ] All code version controlled
- [ ] Dependencies documented (requirements.txt / environment.yml)
- [ ] Entry point script provided (run_validation.py)
- [ ] Random seeds fixed and documented
- [ ] Comments explain non-obvious code

### Data

- [ ] Test data saved
- [ ] Data format documented
- [ ] Data generation code provided
- [ ] Data provenance recorded

### Environment

- [ ] Python version documented: _______________
- [ ] Key package versions documented:
  - NumPy: _______________
  - SciPy: _______________
  - Other: _______________
- [ ] Hardware requirements documented
- [ ] OS tested on: _______________

### Cross-Platform Verification

| Platform | Tested? | Results Match? | Notes |
|----------|---------|----------------|-------|
| Linux | [ ] | [ ] | |
| macOS | [ ] | [ ] | |
| Windows | [ ] | [ ] | |

---

## Validation Summary

### Per-Claim Summary

| Claim | Methods Used | All Passed? | Confidence | Notes |
|-------|--------------|-------------|------------|-------|
| C1 | | [ ] | High / Medium / Low | |
| C2 | | [ ] | | |
| C3 | | [ ] | | |
| C4 | | [ ] | | |
| C5 | | [ ] | | |

### Overall Validation Status

- Total claims: _______________
- Fully validated: _______________
- Partially validated: _______________
- Failed validation: _______________

### Known Limitations

1.
2.
3.

### Caveats and Notes

>

---

## Sign-Off

### Validation Completion

- [ ] All critical claims validated
- [ ] Validation report written
- [ ] Reproducibility package created
- [ ] Peer review completed
- [ ] Documentation complete

**Validated by:** _________________________ **Date:** _____________

**Reviewed by:** _________________________ **Date:** _____________

---

## Appendix: Detailed Test Logs

### Test Run 1

- Date/Time:
- Configuration:
- Results:

### Test Run 2

- Date/Time:
- Configuration:
- Results:

---

*Validation checklist version 1.0*
