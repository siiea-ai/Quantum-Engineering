# Documentation Checklist Template

## Comprehensive Research Documentation Verification

---

## Project Information

| Field | Details |
|-------|---------|
| **Project Title** | |
| **Documentation Period** | Month 63 |
| **Principal Investigator** | |
| **Date Completed** | |

---

## Part I: Theoretical Documentation

### Theoretical Framework Document

**Document Status:**
- [ ] Complete
- [ ] Partial
- [ ] Not started

**Location:** _________________________________

### Content Checklist

| Section | Status | Quality (1-5) | Notes |
|---------|--------|---------------|-------|
| Introduction and Motivation | | | |
| Notation and Preliminaries | | | |
| Main Results (statements) | | | |
| Main Results (proofs) | | | |
| Technical Lemmas | | | |
| Applications/Examples | | | |
| Discussion of Limitations | | | |
| Open Questions | | | |

### Proof Documentation

| Theorem/Lemma | Stated | Proved | Verified | Notes |
|---------------|--------|--------|----------|-------|
| | [ ] | [ ] | [ ] | |
| | [ ] | [ ] | [ ] | |
| | [ ] | [ ] | [ ] | |
| | [ ] | [ ] | [ ] | |
| | [ ] | [ ] | [ ] | |

### Notation Documentation

- [ ] All notation defined before first use
- [ ] Notation table provided
- [ ] Notation consistent throughout
- [ ] Notation matches standard conventions where possible

### Citation Documentation

- [ ] All prior results properly cited
- [ ] All borrowed techniques acknowledged
- [ ] Full bibliography compiled
- [ ] Citation format matches target venue

---

## Part II: Code Documentation

### Repository Structure

**Repository Location:** _________________________________

**Structure Verification:**

| Directory/File | Exists? | Complete? | Notes |
|----------------|---------|-----------|-------|
| README.md | [ ] | [ ] | |
| requirements.txt | [ ] | [ ] | |
| setup.py or pyproject.toml | [ ] | [ ] | |
| src/ or main code | [ ] | [ ] | |
| tests/ | [ ] | [ ] | |
| docs/ | [ ] | [ ] | |
| examples/ or notebooks/ | [ ] | [ ] | |
| .gitignore | [ ] | [ ] | |
| LICENSE | [ ] | [ ] | |

### README Completeness

- [ ] Project overview
- [ ] Installation instructions
- [ ] Usage examples
- [ ] Project structure description
- [ ] Citation information
- [ ] License information
- [ ] Contact information

### Code Quality

| Module | Docstrings | Type Hints | Tests | Comments |
|--------|------------|------------|-------|----------|
| | [ ] | [ ] | [ ] | [ ] |
| | [ ] | [ ] | [ ] | [ ] |
| | [ ] | [ ] | [ ] | [ ] |
| | [ ] | [ ] | [ ] | [ ] |

### Function Documentation

For critical functions:

| Function | Purpose Documented | Args Documented | Returns Documented | Example Provided |
|----------|-------------------|-----------------|-------------------|------------------|
| | [ ] | [ ] | [ ] | [ ] |
| | [ ] | [ ] | [ ] | [ ] |
| | [ ] | [ ] | [ ] | [ ] |
| | [ ] | [ ] | [ ] | [ ] |

### Testing Documentation

- [ ] Unit tests exist
- [ ] Integration tests exist
- [ ] Tests are documented
- [ ] How to run tests is documented
- [ ] Test coverage is acceptable

---

## Part III: Data Documentation

### Data Inventory

| Dataset | Source | Format | Size | Location | Documented? |
|---------|--------|--------|------|----------|-------------|
| | | | | | [ ] |
| | | | | | [ ] |
| | | | | | [ ] |

### Data Provenance

For each dataset:

**Dataset 1: _______________**
- [ ] Source documented
- [ ] Collection method documented
- [ ] Preprocessing documented
- [ ] Format documented
- [ ] Schema/structure documented

**Dataset 2: _______________**
- [ ] Source documented
- [ ] Collection method documented
- [ ] Preprocessing documented
- [ ] Format documented
- [ ] Schema/structure documented

### Generated Data

| Output | Script | Parameters | Location | Reproducible? |
|--------|--------|------------|----------|---------------|
| | | | | [ ] |
| | | | | [ ] |
| | | | | [ ] |

---

## Part IV: Reproducibility Package

### Package Location

**Package Directory:** _________________________________

### Package Contents

| Item | Present? | Tested? | Notes |
|------|----------|---------|-------|
| README.md | [ ] | [ ] | |
| requirements.txt | [ ] | [ ] | |
| environment.yml | [ ] | [ ] | |
| reproduce_all.py/sh | [ ] | [ ] | |
| Source code | [ ] | [ ] | |
| Test suite | [ ] | [ ] | |
| Expected outputs | [ ] | [ ] | |

### Environment Specification

- [ ] Python version specified
- [ ] Package versions frozen
- [ ] OS compatibility noted
- [ ] Hardware requirements noted (if applicable)

### Verification

- [ ] Fresh environment installation tested
- [ ] All scripts run without error
- [ ] Outputs match expected
- [ ] Tested on different machine (if possible)

### Random Seed Reproducibility

- [ ] All random seeds documented
- [ ] Seeds are settable via configuration
- [ ] Results are deterministic when seeds are fixed

---

## Part V: Paper Materials

### Paper Outline

**Status:** [ ] Complete [ ] Partial [ ] Not Started

**Location:** _________________________________

| Section | Outlined? | Notes |
|---------|-----------|-------|
| Abstract | [ ] | |
| Introduction | [ ] | |
| Background | [ ] | |
| Main Results | [ ] | |
| Methods/Proofs | [ ] | |
| Experiments | [ ] | |
| Discussion | [ ] | |
| Conclusion | [ ] | |
| Appendix | [ ] | |

### Draft Sections

| Section | Status | Word Count | Quality (1-5) |
|---------|--------|------------|---------------|
| Abstract | Draft / Final / None | | |
| Introduction | | | |
| Other sections | | | |

### Figures

| Figure | Title | File Location | Paper Ready? |
|--------|-------|---------------|--------------|
| 1 | | | [ ] |
| 2 | | | [ ] |
| 3 | | | [ ] |
| 4 | | | [ ] |
| 5 | | | [ ] |

### Tables

| Table | Title | Content Complete? | Formatted? |
|-------|-------|-------------------|------------|
| 1 | | [ ] | [ ] |
| 2 | | [ ] | [ ] |
| 3 | | [ ] | [ ] |

### References

- [ ] All references collected
- [ ] BibTeX file complete
- [ ] Citation style matches venue
- [ ] DOIs included where available

---

## Part VI: Validation Documentation

### Validation Report

**Status:** [ ] Complete [ ] Partial [ ] Not Started

**Location:** _________________________________

### Validation Record

| Claim | Methods Used | Outcome | Documented? |
|-------|--------------|---------|-------------|
| | | Pass / Partial / Fail | [ ] |
| | | | [ ] |
| | | | [ ] |
| | | | [ ] |

### Numerical Precision Documentation

- [ ] Tolerances documented
- [ ] Precision analysis included
- [ ] Platform dependencies noted

### Edge Case Documentation

- [ ] Edge cases enumerated
- [ ] Behavior at edges documented
- [ ] Limitations stated

---

## Part VII: Completeness Verification

### Documentation Coverage

| Category | Coverage | Quality |
|----------|----------|---------|
| Theoretical work | % | /10 |
| Code | % | /10 |
| Data | % | /10 |
| Validation | % | /10 |
| Paper materials | % | /10 |
| **Overall** | % | /10 |

### Missing Items

| Item | Priority | Plan to Complete |
|------|----------|------------------|
| | High / Medium / Low | |
| | | |
| | | |

### Quality Issues

| Issue | Location | Severity | Fix Plan |
|-------|----------|----------|----------|
| | | High / Medium / Low | |
| | | | |
| | | | |

---

## Part VIII: Sign-Off

### Documentation Complete

I verify that the following are complete and accurate:

- [ ] Theoretical framework document
- [ ] Code documentation
- [ ] Data documentation
- [ ] Reproducibility package
- [ ] Paper materials
- [ ] Validation records

### Ready for Paper Writing

- [ ] All results are documented
- [ ] All materials are organized
- [ ] Paper outline is detailed enough to write from
- [ ] No blocking issues remain

**Completed by:** _________________________

**Date:** _____________

**Reviewed by:** _________________________

**Date:** _____________

---

## Notes

>

---

*Documentation Checklist v1.0*

*Complete documentation enables reproducible science and efficient paper writing.*
