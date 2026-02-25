# Research Data Management Best Practices

## Overview

Effective research data management (RDM) ensures your data remains findable, accessible, interoperable, and reusable (FAIR principles). This guide covers best practices for managing research data throughout its lifecycle.

---

## FAIR Principles

### Findable

- **Persistent identifiers:** Assign DOIs to datasets
- **Rich metadata:** Describe data comprehensively
- **Indexed:** Register in searchable repositories
- **Keywords:** Use standardized vocabularies

### Accessible

- **Open access:** Make data publicly available when possible
- **Standard protocols:** Use HTTP/HTTPS for access
- **Authentication:** Clear access procedures if restricted
- **Long-term preservation:** Use trusted repositories

### Interoperable

- **Standard formats:** Use open, non-proprietary formats
- **Vocabularies:** Use community standards
- **References:** Link to related data and publications
- **Machine-readable:** Structure data for automated processing

### Reusable

- **Clear licensing:** Specify usage terms (CC-BY, CC0, etc.)
- **Provenance:** Document data origin and processing
- **Community standards:** Follow discipline-specific practices
- **Quality assurance:** Validate and document data quality

---

## Data Lifecycle Management

### 1. Planning

**Before data collection:**

- [ ] Define data types to be collected
- [ ] Estimate data volumes
- [ ] Plan storage and backup strategy
- [ ] Identify security/privacy requirements
- [ ] Determine ownership and access rights
- [ ] Create data management plan

**Data Management Plan (DMP) Elements:**

| Section | Content |
|---------|---------|
| Data description | Types, formats, volumes |
| Standards | File formats, metadata standards |
| Storage | Where data will be stored |
| Backup | Backup frequency and locations |
| Access | Who can access, how |
| Sharing | How and when data will be shared |
| Preservation | Long-term archiving plan |
| Responsibilities | Who manages each aspect |

### 2. Collection

**During data collection:**

- Use consistent naming conventions
- Record metadata immediately
- Validate data quality in real-time
- Document collection procedures
- Track data provenance

**File Naming Conventions:**

```
[project]_[experiment]_[date]_[version].[ext]
qec_surface_code_20260115_v01.h5

# Date format: YYYYMMDD or YYYY-MM-DD
# Version: v01, v02, etc.
# Avoid: spaces, special characters (!@#$%^&*)
```

### 3. Processing

**During analysis:**

- Document all processing steps
- Preserve raw data (never modify originals)
- Version control analysis scripts
- Track software versions
- Record parameters and settings

**Processing Documentation:**

```markdown
## Processing Log

### Step 1: Raw Data Import
- Date: 2026-01-15
- Script: scripts/import_data.py
- Input: raw/experiment_001.h5
- Output: processed/cleaned_001.h5
- Parameters: threshold=0.5, normalize=True

### Step 2: Analysis
- Date: 2026-01-16
- Script: scripts/analyze.py (commit: abc123)
- Input: processed/cleaned_001.h5
- Output: results/analysis_001.json
- Runtime: 2.5 hours
```

### 4. Storage

**Storage best practices:**

| Practice | Implementation |
|----------|----------------|
| 3-2-1 Rule | 3 copies, 2 media types, 1 offsite |
| Redundancy | RAID storage or cloud replication |
| Versioning | Git for code, version numbers for data |
| Access control | Appropriate permissions |
| Encryption | For sensitive data |

**Storage Locations:**

| Type | Examples | Use For |
|------|----------|---------|
| Local | Lab server, workstation | Active work |
| Cloud | Google Drive, Dropbox | Collaboration |
| HPC | Cluster storage | Large computations |
| Archive | Zenodo, institutional | Long-term preservation |

### 5. Sharing

**When to share:**

- Upon publication (common requirement)
- Upon request (reasonable timeline)
- Immediately (for some funded research)
- Never (sensitive/proprietary data)

**How to share:**

1. Choose appropriate repository
2. Apply open license
3. Write comprehensive documentation
4. Obtain DOI
5. Link from publications

**Repository Selection:**

| Repository | Type | Best For |
|------------|------|----------|
| Zenodo | General | Any research data |
| Dryad | General | Data underlying publications |
| Figshare | General | Small to medium datasets |
| OSF | General | Project-level sharing |
| GitHub | Code | Software and scripts |
| Domain-specific | Specialized | Field-specific data |

### 6. Preservation

**Long-term preservation considerations:**

- Format obsolescence
- Media degradation
- Organizational changes
- Funding continuity

**Preservation Strategies:**

| Strategy | Implementation |
|----------|----------------|
| Format migration | Convert to current standards periodically |
| Redundant storage | Multiple copies in different locations |
| Institutional repository | Leverage university infrastructure |
| Trusted archives | Use certified repositories (CoreTrustSeal) |

---

## File Formats

### Recommended Formats

| Data Type | Preferred Format | Alternative |
|-----------|-----------------|-------------|
| Tabular | CSV, TSV | HDF5, Parquet |
| Hierarchical | HDF5, NetCDF | JSON |
| Images | TIFF, PNG | JPEG (lossy) |
| Documents | PDF/A | Plain text |
| Code | Plain text (.py, .r) | |
| Structured | JSON, XML | YAML |

### Format Selection Criteria

- **Open:** Non-proprietary specifications
- **Documented:** Well-defined format specification
- **Widely supported:** Multiple software can read/write
- **Stable:** Unlikely to become obsolete
- **Appropriate:** Suited to data characteristics

---

## Metadata

### Minimum Metadata Elements

| Element | Description | Example |
|---------|-------------|---------|
| Title | Descriptive name | "Surface Code Error Rates" |
| Creator | Author(s) with identifiers | "Jane Smith (ORCID: ...)" |
| Date | Creation/collection date | "2026-01-15" |
| Description | What data contains | "Logical error rates..." |
| Subject | Keywords | "quantum computing, ..." |
| Format | File format | "HDF5" |
| Identifier | Persistent ID | "doi:10.5281/..." |
| Rights | License | "CC-BY 4.0" |

### Discipline-Specific Metadata

Follow community standards when available:

- **Physical sciences:** Dublin Core, DataCite
- **Biology:** MIBBI standards
- **Geosciences:** ISO 19115
- **Social sciences:** DDI

### Machine-Readable Metadata

```json
{
  "@context": "https://schema.org",
  "@type": "Dataset",
  "name": "Surface Code Error Correction Measurements",
  "description": "Experimental measurements of logical error rates...",
  "creator": {
    "@type": "Person",
    "name": "Jane Smith",
    "identifier": "https://orcid.org/0000-0000-0000-0000"
  },
  "dateCreated": "2026-01-15",
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "distribution": {
    "@type": "DataDownload",
    "contentUrl": "https://doi.org/10.5281/zenodo.XXXXXXX",
    "encodingFormat": "application/x-hdf5"
  }
}
```

---

## Data Security

### Classification Levels

| Level | Description | Handling |
|-------|-------------|----------|
| Public | No restrictions | Open sharing |
| Internal | University only | Controlled access |
| Confidential | Need-to-know | Encrypted storage |
| Restricted | Legal requirements | Special protocols |

### Security Measures

| Measure | Implementation |
|---------|----------------|
| Access control | User authentication, role-based access |
| Encryption | At rest and in transit |
| Audit logging | Track who accessed what |
| Secure deletion | Proper data disposal |
| Network security | VPN, firewalls |

### Sensitive Data Handling

For data with privacy concerns:

1. Minimize collection (only what's needed)
2. Anonymize/pseudonymize when possible
3. Obtain proper consent
4. Store securely with encryption
5. Limit access to authorized personnel
6. Dispose properly when no longer needed

---

## Version Control

### Data Versioning

| Approach | Best For |
|----------|----------|
| File naming (v01, v02) | Simple datasets |
| Directory structure | Periodic snapshots |
| Git LFS | Code and small data |
| DVC | Large datasets |
| Database versioning | Structured data |

### Version Documentation

```markdown
## Version History

### v1.0.0 (2026-01-15)
- Initial release
- Contains measurements from experiments 1-50

### v1.1.0 (2026-02-01)
- Added experiments 51-75
- Corrected calibration error in experiments 20-25

### v2.0.0 (2026-03-15)
- Major restructuring of data format
- Added new derived quantities
- Breaking changes from v1.x
```

---

## Quality Assurance

### Data Validation

```python
import pandas as pd
import numpy as np

def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate dataset and return quality report."""
    report = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }

    # Check for out-of-range values
    for col in df.select_dtypes(include=[np.number]).columns:
        report[f'{col}_range'] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }

    return report
```

### Quality Checklist

- [ ] No missing required fields
- [ ] Values within expected ranges
- [ ] No duplicate records (unless expected)
- [ ] Consistent formatting
- [ ] File integrity verified (checksums)
- [ ] Documentation complete
- [ ] Metadata accurate

---

## Tools and Resources

### Data Management Tools

| Tool | Purpose | URL |
|------|---------|-----|
| DVC | Data version control | dvc.org |
| DBeaver | Database management | dbeaver.io |
| OpenRefine | Data cleaning | openrefine.org |
| Frictionless | Data validation | frictionlessdata.io |

### Repositories

| Repository | Type | URL |
|------------|------|-----|
| Zenodo | General | zenodo.org |
| Dryad | General | datadryad.org |
| Figshare | General | figshare.com |
| OSF | Projects | osf.io |

### Standards and Guidelines

| Resource | Description |
|----------|-------------|
| FAIR Principles | fairsharing.org |
| DataCite | Metadata schema |
| Creative Commons | Licenses |
| CoreTrustSeal | Repository certification |

---

## Institutional Resources

### University Services

- **Library data services:** Consultation, training, repository
- **Research computing:** Storage, HPC, software
- **Research office:** Compliance, DMP review
- **IT security:** Data classification, encryption

### Training Opportunities

- Library RDM workshops
- Research computing training
- Online courses (Coursera, DataCamp)
- Discipline-specific training

---

## References

1. Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data*, 3, 160018.

2. Borgman, C. L. (2015). *Big Data, Little Data, No Data: Scholarship in the Networked World*. MIT Press.

3. Inter-university Consortium for Political and Social Research (ICPSR). *Guide to Social Science Data Preparation and Archiving*.

---

*This guide provides general best practices. Consult your institution's specific policies and requirements.*
