# Template: Code Repository Release Checklist

## Instructions

Use this comprehensive checklist to ensure your code repository is ready for public release. Complete each section systematically before making your repository public or creating a release.

---

## Repository Information

**Repository Name:** ________________________________________

**Repository URL:** ________________________________________

**Primary Language:** ________________________________________

**Target Release Version:** ________________________________________

**Release Date:** ________________________________________

---

## Section 1: Repository Structure

### 1.1 Essential Files

| File | Present | Complete | Notes |
|------|---------|----------|-------|
| README.md | [ ] | [ ] | |
| LICENSE | [ ] | [ ] | |
| .gitignore | [ ] | [ ] | |
| requirements.txt / environment.yml | [ ] | [ ] | |
| setup.py / pyproject.toml | [ ] | [ ] | |
| CITATION.cff | [ ] | [ ] | |
| CONTRIBUTING.md | [ ] | [ ] | |
| CHANGELOG.md | [ ] | [ ] | |
| CODE_OF_CONDUCT.md | [ ] | [ ] | |

### 1.2 Directory Structure

```
[ ] Project follows clear organizational structure
[ ] Source code in dedicated directory (src/, lib/, etc.)
[ ] Tests in dedicated directory (tests/)
[ ] Documentation in dedicated directory (docs/)
[ ] Examples/notebooks in dedicated directory
[ ] Data files organized appropriately
```

**Current Structure (fill in):**
```
project/
├──
├──
├──
├──
└──
```

---

## Section 2: Code Quality

### 2.1 Code Cleanliness

- [ ] No debug print statements
- [ ] No commented-out code blocks (or minimal, justified)
- [ ] No TODO/FIXME comments for critical issues
- [ ] No unused imports
- [ ] No unused variables or functions
- [ ] No dead code paths
- [ ] No temporary/test files

### 2.2 Code Style

- [ ] Consistent indentation (spaces vs tabs)
- [ ] Consistent naming conventions
- [ ] Line length limits respected
- [ ] Imports organized (standard library, third-party, local)
- [ ] Style follows language conventions (PEP 8, etc.)

**Linting Tool Used:** ________________________________________

**Linting Warnings/Errors Remaining:** _______

### 2.3 Code Security

- [ ] No hardcoded credentials (API keys, passwords)
- [ ] No hardcoded absolute paths
- [ ] No sensitive information in comments
- [ ] No personal information in code
- [ ] Configuration uses environment variables or config files
- [ ] .gitignore excludes sensitive files

**Security Scan Tool (if used):** ________________________________________

### 2.4 Documentation in Code

- [ ] All modules have docstrings
- [ ] All public functions have docstrings
- [ ] All classes have docstrings
- [ ] Complex logic has inline comments
- [ ] Type hints included (for Python 3.5+)

---

## Section 3: README Quality

### 3.1 README Sections

| Section | Present | Quality |
|---------|---------|---------|
| Project Title/Description | [ ] | [ ] Good |
| Badges (CI, license, DOI) | [ ] | [ ] Good |
| Installation Instructions | [ ] | [ ] Good |
| Quick Start / Usage | [ ] | [ ] Good |
| Examples | [ ] | [ ] Good |
| Documentation Link | [ ] | [ ] Good |
| Contributing Guidelines | [ ] | [ ] Good |
| License Information | [ ] | [ ] Good |
| Citation Information | [ ] | [ ] Good |
| Acknowledgments | [ ] | [ ] Good |

### 3.2 README Content

- [ ] Clear, concise project description
- [ ] Prerequisites clearly stated
- [ ] Installation steps tested and working
- [ ] At least one runnable example
- [ ] Links to detailed documentation
- [ ] Contact information provided
- [ ] No broken links

---

## Section 4: Dependencies

### 4.1 Dependency Documentation

- [ ] All dependencies listed in requirements file
- [ ] Version constraints specified
- [ ] Optional dependencies marked
- [ ] Development dependencies separated

### 4.2 Dependency Verification

- [ ] All listed dependencies are necessary
- [ ] No missing dependencies
- [ ] Compatible version ranges specified
- [ ] Dependencies available on standard package managers

### 4.3 Fresh Installation Test

- [ ] Tested on clean virtual environment
- [ ] Installation completes without errors
- [ ] Basic functionality works after fresh install

**Test Environment:**
- Python Version: _______
- OS: _______
- Date Tested: _______

---

## Section 5: Testing

### 5.1 Test Suite

- [ ] Test suite exists
- [ ] Tests cover core functionality
- [ ] Tests are documented
- [ ] Tests are easy to run
- [ ] Test requirements documented

### 5.2 Test Results

**Test Command:** ________________________________________

```
[ ] All tests pass
[ ] Test coverage measured
```

**Test Coverage:** _______%

**Failing Tests (if any):**
```
________________________________________
```

### 5.3 Continuous Integration

- [ ] CI configured (GitHub Actions, Travis, etc.)
- [ ] CI builds pass
- [ ] CI runs tests
- [ ] CI badge in README

**CI Platform:** ________________________________________

---

## Section 6: Documentation

### 6.1 User Documentation

- [ ] Installation guide complete
- [ ] Usage guide complete
- [ ] Examples/tutorials provided
- [ ] FAQ or troubleshooting section

### 6.2 API Documentation

- [ ] API reference generated
- [ ] All public functions documented
- [ ] Parameter descriptions complete
- [ ] Return value descriptions complete
- [ ] Examples in docstrings

### 6.3 Developer Documentation

- [ ] Architecture overview
- [ ] Contributing guidelines
- [ ] Development setup instructions
- [ ] Code style guidelines

**Documentation URL:** ________________________________________

---

## Section 7: Version Control

### 7.1 Git History

- [ ] Commit history is clean
- [ ] No large binary files in history
- [ ] No sensitive information in history
- [ ] Meaningful commit messages

### 7.2 Branches

- [ ] Main/master branch is stable
- [ ] Development branch cleaned up
- [ ] Feature branches merged or deleted
- [ ] Tags used for releases

### 7.3 Release

- [ ] Version number updated in code
- [ ] CHANGELOG updated
- [ ] Release notes prepared
- [ ] Tag created for release

**Release Tag:** ________________________________________

---

## Section 8: Licensing

### 8.1 License File

- [ ] LICENSE file present in root
- [ ] License text is complete and correct
- [ ] License compatible with all dependencies

**License Type:** ________________________________________

### 8.2 License Headers

- [ ] Source files have license headers (if required)
- [ ] Third-party code attributed
- [ ] License compatibility verified

### 8.3 Third-Party Attributions

| Component | License | Attribution |
|-----------|---------|-------------|
| | | [ ] Done |
| | | [ ] Done |
| | | [ ] Done |

---

## Section 9: Reproducibility

### 9.1 Environment Reproducibility

- [ ] Environment can be recreated from spec file
- [ ] Random seeds are set for reproducibility
- [ ] Output is deterministic (or stochasticity documented)

### 9.2 Result Reproducibility

- [ ] Paper results can be regenerated
- [ ] Figures can be reproduced
- [ ] Numerical values match expected

### 9.3 Reproducibility Documentation

- [ ] Steps to reproduce results documented
- [ ] Expected outputs described
- [ ] Compute requirements documented

---

## Section 10: Data Handling

### 10.1 Data Included

- [ ] Only necessary data included
- [ ] Large files handled appropriately (Git LFS, external)
- [ ] Data formats are open standards
- [ ] Data documentation included

### 10.2 Data Download

- [ ] Download scripts provided (if external data)
- [ ] Data sources documented
- [ ] Data versioned or dated

### 10.3 Data Privacy

- [ ] No personally identifiable information
- [ ] No proprietary data without permission
- [ ] Data sharing agreements complied with

---

## Section 11: Pre-Release Checklist

### 11.1 Final Verification

- [ ] Fresh clone and install tested
- [ ] All tests pass
- [ ] README is accurate
- [ ] Documentation is current
- [ ] Version numbers consistent
- [ ] No debug/development artifacts

### 11.2 Team Approval

| Reviewer | Date | Approved |
|----------|------|----------|
| | | [ ] |
| | | [ ] |
| | | [ ] |

### 11.3 Release Preparation

- [ ] CHANGELOG entry written
- [ ] Release notes drafted
- [ ] GitHub release created
- [ ] Zenodo integration triggered

---

## Section 12: Post-Release

### 12.1 Verification

- [ ] Release appears correctly on GitHub
- [ ] Zenodo DOI assigned
- [ ] Documentation site updated
- [ ] Package manager updated (if applicable)

### 12.2 Announcement

- [ ] Paper updated with DOI
- [ ] README updated with DOI badge
- [ ] Co-authors notified
- [ ] Social media announcement (if desired)

### 12.3 Documentation

- [ ] Release recorded in project log
- [ ] Lessons learned documented
- [ ] Future improvements noted

---

## Issues Identified

### Critical (Must Fix Before Release)

1. ________________________________________
2. ________________________________________
3. ________________________________________

### Important (Should Fix Before Release)

1. ________________________________________
2. ________________________________________
3. ________________________________________

### Minor (Can Fix After Release)

1. ________________________________________
2. ________________________________________
3. ________________________________________

---

## Summary

| Category | Items | Completed | Percentage |
|----------|-------|-----------|------------|
| Structure | 15 | /15 | % |
| Code Quality | 20 | /20 | % |
| README | 12 | /12 | % |
| Dependencies | 8 | /8 | % |
| Testing | 10 | /10 | % |
| Documentation | 10 | /10 | % |
| Version Control | 10 | /10 | % |
| Licensing | 8 | /8 | % |
| Reproducibility | 8 | /8 | % |
| Data | 10 | /10 | % |

**Total Completion:** _______%

**Ready for Release:** [ ] Yes [ ] No

---

*Date Completed:* ______________

*Reviewed By:* ______________

*Version:* 1.0
