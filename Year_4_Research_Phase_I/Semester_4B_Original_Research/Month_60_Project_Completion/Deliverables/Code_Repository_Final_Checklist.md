# Code Repository Final Checklist

## Overview

This checklist ensures your code repository is complete, well-documented, and ready for public release. Use this as the final verification before creating a release and obtaining a DOI.

---

## Repository Information

**Repository Name:** ________________________________________

**Repository URL:** ________________________________________

**Primary Language:** ________________________________________

**License:** ________________________________________

**Release Version:** ________________________________________

**Release Date:** ______________

---

## Section 1: Core Files

### 1.1 Essential Files Present

| File | Present | Complete | Last Updated |
|------|---------|----------|--------------|
| README.md | [ ] | [ ] | |
| LICENSE | [ ] | [ ] | |
| CITATION.cff | [ ] | [ ] | |
| requirements.txt | [ ] | [ ] | |
| setup.py / pyproject.toml | [ ] | [ ] | |
| .gitignore | [ ] | [ ] | |
| CONTRIBUTING.md | [ ] | [ ] | |
| CHANGELOG.md | [ ] | [ ] | |
| CODE_OF_CONDUCT.md | [ ] | [ ] | |

### 1.2 README Quality

**README Contains:**
- [ ] Project title and description
- [ ] Badges (DOI, license, CI status)
- [ ] Installation instructions
- [ ] Quick start example
- [ ] Usage documentation
- [ ] Link to full documentation
- [ ] Citation information
- [ ] License information
- [ ] Acknowledgments
- [ ] Contact information

**README Quality Assessment:**
| Criterion | Rating (1-5) |
|-----------|--------------|
| Clarity | |
| Completeness | |
| Accuracy | |
| Usability | |

### 1.3 License

- [ ] Open source license selected
- [ ] LICENSE file in root directory
- [ ] License compatible with dependencies
- [ ] Headers in source files (if required)

**License Type:** ________________________________________

---

## Section 2: Code Quality

### 2.1 Code Cleanliness

- [ ] No debug print statements
- [ ] No commented-out code blocks
- [ ] No TODO/FIXME for critical issues
- [ ] No unused imports
- [ ] No dead code
- [ ] No temporary files
- [ ] No personal information
- [ ] No hardcoded paths

### 2.2 Code Style

- [ ] Consistent indentation
- [ ] Consistent naming conventions
- [ ] PEP 8 compliant (Python)
- [ ] Line length limits respected
- [ ] Imports organized
- [ ] Linting passes

**Linting Report:**
```
Tool: ______________
Errors: ______
Warnings: ______
```

### 2.3 Security Check

- [ ] No API keys or passwords
- [ ] No hardcoded credentials
- [ ] No sensitive data in history
- [ ] Configuration uses environment variables
- [ ] .gitignore excludes sensitive files

**Security Scan:**
| Item | Status |
|------|--------|
| Secrets in code | [ ] Clean |
| Secrets in history | [ ] Clean |
| Dependency vulnerabilities | [ ] Clean |

### 2.4 Documentation in Code

- [ ] All modules have docstrings
- [ ] All public functions have docstrings
- [ ] All classes have docstrings
- [ ] Complex logic has comments
- [ ] Type hints included (Python 3.5+)

**Documentation Coverage:**
- Modules documented: _______ / _______
- Functions documented: _______ / _______
- Classes documented: _______ / _______

---

## Section 3: Directory Structure

### 3.1 Organization

- [ ] Clear top-level structure
- [ ] Source code in dedicated directory
- [ ] Tests in dedicated directory
- [ ] Documentation in dedicated directory
- [ ] Examples/notebooks separate
- [ ] Data files organized

**Directory Tree:**
```
project/
├── src/             [ ] Present
├── tests/           [ ] Present
├── docs/            [ ] Present
├── examples/        [ ] Present
├── notebooks/       [ ] Present
├── data/            [ ] Present
└── scripts/         [ ] Present
```

### 3.2 File Naming

- [ ] Consistent naming convention
- [ ] Descriptive names
- [ ] No spaces in filenames
- [ ] Lowercase (or consistent case)

---

## Section 4: Dependencies

### 4.1 Dependency Specification

- [ ] All dependencies listed
- [ ] Version constraints specified
- [ ] Optional dependencies marked
- [ ] Dev dependencies separated

**requirements.txt Check:**
```
[ ] All runtime dependencies
[ ] Version bounds (e.g., >=1.0,<2.0)
[ ] No unnecessary dependencies
```

### 4.2 Environment Reproducibility

- [ ] Virtual environment tested
- [ ] Conda environment tested (if applicable)
- [ ] Docker tested (if applicable)
- [ ] Specific Python version noted

**Tested Environments:**
| Environment | Version | Status |
|-------------|---------|--------|
| Python | | [ ] Works |
| pip | | [ ] Works |
| conda | | [ ] Works |
| Docker | | [ ] Works |

### 4.3 Fresh Installation Test

- [ ] Cloned to new location
- [ ] Installed in clean environment
- [ ] All dependencies resolved
- [ ] Basic functionality works

**Test Date:** ______________

**Test Environment:**
```
OS: ______________
Python: ______________
pip: ______________
```

---

## Section 5: Testing

### 5.1 Test Suite

- [ ] Tests exist
- [ ] Tests are runnable
- [ ] Tests pass
- [ ] Tests documented
- [ ] Test requirements specified

**Test Command:** ________________________________________

**Test Results:**
```
Tests run: ______
Passed: ______
Failed: ______
Skipped: ______
```

### 5.2 Test Coverage

- [ ] Coverage measured
- [ ] Critical paths covered
- [ ] Edge cases tested
- [ ] Error handling tested

**Coverage:** _______%

### 5.3 Reproducibility Tests

- [ ] Results can be reproduced
- [ ] Random seeds set
- [ ] Output matches expected
- [ ] Paper results reproducible

---

## Section 6: Documentation

### 6.1 User Documentation

| Document | Present | Complete |
|----------|---------|----------|
| Installation guide | [ ] | [ ] |
| Quick start | [ ] | [ ] |
| Usage guide | [ ] | [ ] |
| API reference | [ ] | [ ] |
| Examples | [ ] | [ ] |
| FAQ/Troubleshooting | [ ] | [ ] |

### 6.2 Example Quality

- [ ] Examples are runnable
- [ ] Examples are well-commented
- [ ] Examples cover main use cases
- [ ] Output shown

**Examples Tested:**
| Example | Works |
|---------|-------|
| | [ ] |
| | [ ] |
| | [ ] |

### 6.3 API Documentation

- [ ] All public APIs documented
- [ ] Parameters described
- [ ] Return values described
- [ ] Examples in docstrings
- [ ] Exceptions documented

---

## Section 7: Version Control

### 7.1 Git History

- [ ] Meaningful commit messages
- [ ] No large binary files
- [ ] No sensitive data
- [ ] Clean history

### 7.2 Branching

- [ ] Main branch stable
- [ ] Feature branches merged
- [ ] Tags for releases
- [ ] Development branch clean

### 7.3 Release Preparation

- [ ] Version number updated
- [ ] CHANGELOG updated
- [ ] Tag created
- [ ] Release notes written

**Release Tag:** ________________________________________

---

## Section 8: CI/CD

### 8.1 Continuous Integration

- [ ] CI configured
- [ ] Tests run automatically
- [ ] Linting runs automatically
- [ ] Builds pass

**CI Platform:** ________________________________________

**CI Status:** [ ] Passing [ ] Failing

### 8.2 Badges

- [ ] CI status badge
- [ ] Coverage badge
- [ ] License badge
- [ ] DOI badge (after Zenodo)

---

## Section 9: Data Handling

### 9.1 Data in Repository

- [ ] Only necessary data included
- [ ] Large files use Git LFS
- [ ] Data formats documented
- [ ] No sensitive data

### 9.2 External Data

- [ ] Download scripts provided
- [ ] Data sources documented
- [ ] Instructions clear
- [ ] Data versioned

### 9.3 Data Documentation

- [ ] README in data directory
- [ ] File formats explained
- [ ] Column/field descriptions
- [ ] Units specified

---

## Section 10: Reproducibility

### 10.1 Paper Results

For each key result in the paper:

| Result | Script | Verified |
|--------|--------|----------|
| Figure 1 | | [ ] |
| Figure 2 | | [ ] |
| Figure 3 | | [ ] |
| Table 1 | | [ ] |
| Key value 1 | | [ ] |
| Key value 2 | | [ ] |

### 10.2 Reproduction Steps

- [ ] Steps documented in README
- [ ] Scripts for reproduction
- [ ] Expected outputs described
- [ ] Compute time estimated

### 10.3 Environment Specification

- [ ] Python version specified
- [ ] Package versions pinned
- [ ] OS requirements noted
- [ ] Hardware requirements noted

---

## Section 11: Final Verification

### 11.1 Fresh Clone Test

```bash
# Test commands
git clone [URL]
cd [repo]
pip install -e .
pytest
python scripts/reproduce_paper.py
```

- [ ] Clone successful
- [ ] Install successful
- [ ] Tests pass
- [ ] Paper results reproduce

### 11.2 External Review

- [ ] Colleague tested installation
- [ ] Colleague ran examples
- [ ] Feedback incorporated

**Reviewer:** ________________________________________

**Date:** ______________

### 11.3 Pre-Release Checklist

- [ ] All tests pass
- [ ] Documentation complete
- [ ] Examples work
- [ ] Version numbers correct
- [ ] CHANGELOG updated
- [ ] No critical issues

---

## Section 12: Release

### 12.1 GitHub Release

- [ ] Release created on GitHub
- [ ] Release notes complete
- [ ] Assets uploaded (if any)
- [ ] Tag correct

**GitHub Release URL:** ________________________________________

### 12.2 Zenodo Integration

- [ ] Zenodo account connected
- [ ] Repository enabled in Zenodo
- [ ] Release triggered archive
- [ ] DOI assigned

**Zenodo DOI:** 10.5281/zenodo.________________________________________

### 12.3 Post-Release Updates

- [ ] README updated with DOI badge
- [ ] Paper updated with DOI
- [ ] CV updated
- [ ] Website updated

---

## Summary

| Section | Items | Complete |
|---------|-------|----------|
| Core Files | /15 | % |
| Code Quality | /20 | % |
| Structure | /12 | % |
| Dependencies | /10 | % |
| Testing | /10 | % |
| Documentation | /12 | % |
| Version Control | /10 | % |
| CI/CD | /6 | % |
| Data | /10 | % |
| Reproducibility | /10 | % |
| Verification | /10 | % |
| Release | /8 | % |

**Total Completion:** _______%

**Repository Release Ready:** [ ] Yes [ ] No

---

## Issues Log

### Critical (Must Fix)

1. ________________________________________
2. ________________________________________
3. ________________________________________

### Important (Should Fix)

1. ________________________________________
2. ________________________________________
3. ________________________________________

### Minor (Can Fix Later)

1. ________________________________________
2. ________________________________________
3. ________________________________________

---

**Completed By:** ________________________________________

**Date:** ______________

**DOI Assigned:** ________________________________________
