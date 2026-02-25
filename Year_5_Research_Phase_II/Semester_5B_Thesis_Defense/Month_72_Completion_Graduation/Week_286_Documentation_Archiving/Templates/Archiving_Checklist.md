# Research Archiving Checklist

## Project Information

| Field | Value |
|-------|-------|
| **Project Title** | |
| **Principal Investigator** | |
| **Date Range** | to |
| **Archive Date** | |
| **Repository Location(s)** | |

---

## Code Archiving

### Repository Preparation

- [ ] All code committed to version control
- [ ] No sensitive information (API keys, passwords) in code
- [ ] `.gitignore` excludes appropriate files
- [ ] Commit history cleaned (no accidental large file commits)
- [ ] Branch structure documented

### Documentation Completeness

- [ ] README.md complete and accurate
- [ ] Installation instructions tested
- [ ] Usage examples provided
- [ ] API documentation generated
- [ ] Changelog maintained
- [ ] Contributing guidelines (if applicable)

### License and Citation

- [ ] LICENSE file present
- [ ] CITATION.cff file present
- [ ] Copyright notices in source files
- [ ] Third-party licenses documented

### Quality Assurance

- [ ] All tests passing
- [ ] Test coverage documented
- [ ] Linting passes
- [ ] Type hints present (Python)
- [ ] Code review completed

### Archive Submission

| Platform | Status | URL/DOI |
|----------|--------|---------|
| GitHub | ☐ | |
| Zenodo | ☐ | |
| Institutional | ☐ | |

---

## Data Archiving

### Data Inventory

| Dataset | Size | Format | Location | DOI |
|---------|------|--------|----------|-----|
| | | | | |
| | | | | |
| | | | | |

### Metadata Requirements

For each dataset:
- [ ] Description/abstract
- [ ] Creator information (name, ORCID)
- [ ] Creation date
- [ ] Keywords
- [ ] License specified
- [ ] Related publications linked
- [ ] Methodology documented

### Data Quality

- [ ] Data validated for completeness
- [ ] Missing values documented
- [ ] Outliers documented
- [ ] File integrity verified (checksums)
- [ ] File formats are open/standard

### Privacy and Ethics

- [ ] No personally identifiable information (or properly anonymized)
- [ ] IRB/ethics approval documented (if applicable)
- [ ] Consent forms on file (if applicable)
- [ ] Sensitive data appropriately restricted

### Archive Submission

| Repository | Status | DOI |
|------------|--------|-----|
| Zenodo | ☐ | |
| Institutional | ☐ | |
| Domain-specific | ☐ | |

---

## Environment Archiving

### Dependency Documentation

- [ ] requirements.txt (Python packages)
- [ ] environment.yml (Conda)
- [ ] Dockerfile
- [ ] docker-compose.yml (if applicable)
- [ ] System dependencies documented

### Version Pinning

- [ ] All package versions pinned
- [ ] Python/language version specified
- [ ] OS requirements documented
- [ ] Hardware requirements documented

### Containerization

- [ ] Docker image builds successfully
- [ ] Docker image tested
- [ ] Docker image pushed to registry (DockerHub/GitHub Container Registry)
- [ ] Image tagged with version

---

## Reproducibility Verification

### Fresh Install Test

- [ ] Cloned repository to new machine
- [ ] Installed dependencies from scratch
- [ ] All tests pass in new environment
- [ ] Example analyses run successfully
- [ ] Outputs match expected values

### Documentation Test

- [ ] Someone else followed instructions successfully
- [ ] Missing steps identified and added
- [ ] Ambiguous instructions clarified

---

## Publication Linkage

### Cross-References

| Output | DOI | Links To | Linked From |
|--------|-----|----------|-------------|
| Thesis | | Code, Data | |
| Code | | Thesis, Data | |
| Data | | Thesis, Code | |
| Paper 1 | | | |
| Paper 2 | | | |

### Metadata Connections

For each DOI:
- [ ] relatedIdentifiers populated
- [ ] Relationship types correct (isSupplementTo, etc.)
- [ ] All links bidirectional where possible

---

## Long-Term Preservation

### File Formats

| Current Format | Preservation Format | Converted |
|----------------|---------------------|-----------|
| .docx | .pdf/a | ☐ |
| .xlsx | .csv | ☐ |
| .pptx | .pdf | ☐ |
| proprietary | open format | ☐ |

### Multiple Copies

- [ ] Primary archive (institutional repository)
- [ ] Secondary archive (Zenodo/domain repository)
- [ ] Personal backup (cloud storage)
- [ ] Physical backup (external drive)

### Access Verification

- [ ] All DOIs resolve correctly
- [ ] All download links work
- [ ] Access permissions correct
- [ ] Embargo dates set (if applicable)

---

## Final Verification

### Completeness Check

- [ ] All code archived with DOI
- [ ] All data archived with DOI
- [ ] All publications linked
- [ ] Environment fully documented
- [ ] Reproducibility verified

### Future Access

- [ ] Contact information current
- [ ] Institutional affiliation documented
- [ ] Personal email backup listed
- [ ] ORCID profile updated

---

## Notes

| Issue | Resolution | Date |
|-------|------------|------|
| | | |
| | | |

---

## Signatures

**Researcher:**
- Name: _____________________
- Date: _____________________

**Supervisor (optional):**
- Name: _____________________
- Date: _____________________

---

*Research archiving ensures your work remains accessible and reproducible for future researchers.*
