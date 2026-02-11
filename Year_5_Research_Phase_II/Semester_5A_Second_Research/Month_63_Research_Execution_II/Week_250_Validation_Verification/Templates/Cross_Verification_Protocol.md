# Cross-Verification Protocol Template

## Multi-Method Validation Framework

---

## Protocol Overview

This protocol ensures research claims are verified through multiple independent methods. Cross-verification increases confidence by demonstrating that results are robust to different approaches.

---

## Claim Under Verification

| Field | Details |
|-------|---------|
| **Claim ID** | |
| **Claim Statement** | |
| **Claim Type** | Theorem / Inequality / Algorithm / Numerical Result |
| **Verification Date** | |
| **Verifier** | |

---

## Verification Method Matrix

### Required Verification Depth

| Claim Importance | Minimum Methods | Required Methods |
|------------------|-----------------|------------------|
| Critical (main result) | 4 | Re-derivation, Numerical, Edge cases, Peer review |
| High (key lemma) | 3 | Numerical, Edge cases, One of: Re-derivation or Peer |
| Medium (supporting) | 2 | Numerical, Edge cases |
| Low (remark) | 1 | Any appropriate method |

### This Claim

**Importance Level:** _____________

**Required Methods:**
- [ ]
- [ ]
- [ ]
- [ ]

---

## Method 1: Independent Re-Derivation

### Approach

**Original derivation approach:**
>

**Alternative approach for re-derivation:**
>

**Why this alternative is truly independent:**
>

### Execution

**Starting point:**
>

**Key steps:**

1.
2.
3.
4.

**Final result:**
>

### Comparison

| Aspect | Original | Re-Derivation | Match? |
|--------|----------|---------------|--------|
| Final statement | | | [ ] |
| Conditions/hypotheses | | | [ ] |
| Proof technique | | | [ ] Different (good) |
| Intermediate results | | | [ ] |

### Outcome

- [ ] **VERIFIED**: Re-derivation matches original
- [ ] **PARTIAL**: Minor differences, explained by:
- [ ] **FAILED**: Significant discrepancy, requires investigation

**Notes:**
>

---

## Method 2: Numerical Verification

### Test Design

**Test categories planned:**

| Category | Number | Purpose | Status |
|----------|--------|---------|--------|
| Random inputs | | General verification | [ ] |
| Structured inputs | | Known cases | [ ] |
| Edge cases | | Boundaries | [ ] |
| Adversarial | | Stress testing | [ ] |

**Parameter space:**

| Parameter | Range | Resolution |
|-----------|-------|------------|
| | | |
| | | |

**Tolerance settings:**
- Absolute tolerance: _______________
- Relative tolerance: _______________

### Execution Log

**Test Run 1:**
- Date/Time:
- Configuration:
- Seed:
- Results: ___ / ___ passed

**Test Run 2:**
- Date/Time:
- Configuration:
- Seed:
- Results: ___ / ___ passed

### Failure Analysis

| Test ID | Input Summary | Expected | Actual | Gap | Explanation |
|---------|---------------|----------|--------|-----|-------------|
| | | | | | |
| | | | | | |

### Outcome

- [ ] **VERIFIED**: All tests pass within tolerance
- [ ] **PARTIAL**: Minor failures, explained by:
- [ ] **FAILED**: Significant failures require investigation

**Summary Statistics:**
- Total tests: _______________
- Passed: _______________
- Failed: _______________
- Pass rate: _______________%

---

## Method 3: Edge Case Analysis

### Edge Case Inventory

| ID | Edge Case | Why It's Important | Claim Should Behave |
|----|-----------|-------------------|---------------------|
| E1 | | | |
| E2 | | | |
| E3 | | | |
| E4 | | | |
| E5 | | | |

### Edge Case Testing

| ID | Test Method | Expected | Actual | Pass? | Notes |
|----|-------------|----------|--------|-------|-------|
| E1 | | | | [ ] | |
| E2 | | | | [ ] | |
| E3 | | | | [ ] | |
| E4 | | | | [ ] | |
| E5 | | | | [ ] | |

### Edge Case Failures

For any failures:

**Edge Case ID:** _______________

**Description of failure:**
>

**Investigation:**
>

**Resolution:**
- [ ] Claim modified to exclude edge case
- [ ] Edge case behavior documented as limitation
- [ ] Test was incorrect (false failure)
- [ ] Other: _______________

### Outcome

- [ ] **VERIFIED**: All edge cases handled correctly
- [ ] **PARTIAL**: Some edge cases require documentation
- [ ] **FAILED**: Edge case reveals fundamental issue

---

## Method 4: Limiting Case Verification

### Limiting Cases Identified

| Limit | Mathematical Expression | Physical Meaning | Expected Behavior |
|-------|------------------------|------------------|-------------------|
| | | | |
| | | | |
| | | | |

### Limiting Case Derivation

**Limit 1:** _______________

**Taking the limit:**
>

**Result:**
>

**Known result it should match:**
>

**Match?** [ ] Yes [ ] No

---

**Limit 2:** _______________

**Taking the limit:**
>

**Result:**
>

**Known result it should match:**
>

**Match?** [ ] Yes [ ] No

### Numerical Limit Verification

| Limit | Approach Value | Limiting Behavior | Matches Theory? |
|-------|----------------|-------------------|-----------------|
| | | | [ ] |
| | | | [ ] |

### Outcome

- [ ] **VERIFIED**: All limiting cases match expectations
- [ ] **PARTIAL**: Some limits not analytically tractable
- [ ] **FAILED**: Limiting behavior contradicts expectations

---

## Method 5: Comparison with Literature

### Related Results

| Reference | Related Claim | Relationship to Our Claim |
|-----------|---------------|---------------------------|
| [Author, Year] | | Generalizes / Specializes / Analogous |
| | | |
| | | |

### Detailed Comparison

**Reference 1:** _______________

**Their claim:**
>

**Our claim:**
>

**How they relate:**
>

**Consistency check:**

| Aspect | Their Result | Our Result | Consistent? |
|--------|-------------|------------|-------------|
| | | | [ ] |
| | | | [ ] |

**Numerical comparison (if applicable):**

| Test Case | Literature Value | Our Value | Difference |
|-----------|------------------|-----------|------------|
| | | | |
| | | | |

### Outcome

- [ ] **VERIFIED**: Consistent with all relevant literature
- [ ] **PARTIAL**: Minor differences, explained by:
- [ ] **FAILED**: Contradiction with literature, requires resolution

---

## Method 6: Peer Verification

### Reviewer Information

| Field | Details |
|-------|---------|
| Reviewer name | |
| Expertise area | |
| Review date | |
| Time spent | |

### Review Scope

- [ ] Proof verification
- [ ] Numerical code review
- [ ] Conceptual review
- [ ] Full review

### Reviewer Feedback

**Overall assessment:**
>

**Specific issues raised:**

| Issue | Severity | Our Response | Status |
|-------|----------|--------------|--------|
| | Critical / Major / Minor | | Resolved / Open |
| | | | |
| | | | |

### Outcome

- [ ] **VERIFIED**: Reviewer approves
- [ ] **PARTIAL**: Minor issues, addressed
- [ ] **FAILED**: Major issues remain

---

## Cross-Verification Summary

### Method Outcomes

| Method | Outcome | Confidence Contribution |
|--------|---------|------------------------|
| Re-derivation | Verified / Partial / Failed / N/A | High / Medium / Low |
| Numerical | | |
| Edge cases | | |
| Limiting cases | | |
| Literature | | |
| Peer review | | |

### Overall Verification Status

Based on the above:

- [ ] **FULLY VERIFIED**: All methods pass, high confidence
- [ ] **VERIFIED WITH CAVEATS**: Passes with documented limitations
- [ ] **PARTIALLY VERIFIED**: Some methods incomplete
- [ ] **VERIFICATION FAILED**: Fundamental issues discovered
- [ ] **REQUIRES FURTHER WORK**: Additional verification needed

### Confidence Assessment

**Numerical confidence score:** ___ / 100

**Qualitative confidence:** High / Medium / Low

**Justification:**
>

### Documented Limitations

Based on cross-verification, the following limitations are documented:

1.
2.
3.

### Open Issues

| Issue | Priority | Planned Resolution |
|-------|----------|-------------------|
| | | |
| | | |

---

## Verification Record

**Verification completed by:** _________________________

**Date:** _____________

**Time spent on verification:** _________ hours

**Reviewer (if different):** _________________________

**Final status:** VERIFIED / PARTIAL / FAILED

---

## Appendix: Supporting Evidence

### A. Re-Derivation Notes

[Attach or reference detailed derivation]

### B. Numerical Test Output

[Attach or reference test logs]

### C. Code Used

[Attach or reference verification scripts]

### D. Correspondence

[Any relevant discussions or communications]

---

*Cross-Verification Protocol v1.0*

*Remember: Multiple independent verification methods are essential for research credibility.*
