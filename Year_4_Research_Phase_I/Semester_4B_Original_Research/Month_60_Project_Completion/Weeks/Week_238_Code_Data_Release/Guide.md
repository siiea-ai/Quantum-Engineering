# Code and Data Release Guide

## Introduction

This comprehensive guide covers the process of preparing research code and data for public release. Reproducibility is a cornerstone of scientific research, and proper code/data release ensures that others can verify, build upon, and extend your work.

---

## Part 1: Code Repository Preparation

### 1.1 Repository Structure

A well-organized repository follows a clear structure:

```
project-name/
├── README.md              # Project overview and quick start
├── LICENSE                # Open source license
├── CITATION.cff           # Citation information
├── setup.py               # Installation script (Python)
├── requirements.txt       # Dependencies
├── environment.yml        # Conda environment (alternative)
├── .gitignore             # Files to exclude from version control
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── core/              # Core functionality
│   ├── utils/             # Utility functions
│   └── visualization/     # Plotting and visualization
│
├── tests/                 # Test suite
│   ├── test_core.py
│   └── test_utils.py
│
├── data/                  # Small data files (or links to external)
│   ├── raw/               # Original, immutable data
│   └── processed/         # Processed data ready for analysis
│
├── notebooks/             # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_analysis.ipynb
│   └── 03_figures.ipynb
│
├── scripts/               # Standalone scripts
│   ├── run_simulation.py
│   └── generate_figures.py
│
├── docs/                  # Documentation
│   ├── installation.md
│   ├── usage.md
│   └── api/
│
├── results/               # Output files
│   ├── figures/
│   └── tables/
│
└── paper/                 # Manuscript files (optional)
    ├── main.tex
    └── figures/
```

### 1.2 Code Cleanup Checklist

Before release, systematically clean your code:

**Remove:**
- [ ] Debug print statements
- [ ] Commented-out code blocks
- [ ] Unused imports
- [ ] Dead code (unreachable functions)
- [ ] Temporary files
- [ ] Personal notes in comments
- [ ] Hardcoded absolute paths
- [ ] Sensitive information (API keys, passwords)

**Standardize:**
- [ ] Variable naming conventions
- [ ] Function naming conventions
- [ ] Indentation and formatting
- [ ] Import organization
- [ ] Comment style

**Add:**
- [ ] Module docstrings
- [ ] Function docstrings
- [ ] Inline comments for complex logic
- [ ] Type hints (Python 3.5+)
- [ ] Error handling

### 1.3 Code Style Guidelines

For Python (most quantum computing code), follow PEP 8:

```python
# Good example
def calculate_fidelity(rho_target: np.ndarray,
                       rho_actual: np.ndarray) -> float:
    """
    Calculate quantum state fidelity between two density matrices.

    Parameters
    ----------
    rho_target : np.ndarray
        Target density matrix (n x n complex array).
    rho_actual : np.ndarray
        Actual density matrix (n x n complex array).

    Returns
    -------
    float
        Fidelity value between 0 and 1.

    Examples
    --------
    >>> rho_target = np.array([[1, 0], [0, 0]])
    >>> rho_actual = np.array([[0.9, 0.1], [0.1, 0.1]])
    >>> fidelity = calculate_fidelity(rho_target, rho_actual)
    >>> print(f"Fidelity: {fidelity:.4f}")
    Fidelity: 0.9000
    """
    sqrt_rho_target = scipy.linalg.sqrtm(rho_target)
    product = sqrt_rho_target @ rho_actual @ sqrt_rho_target
    return np.real(np.trace(scipy.linalg.sqrtm(product))) ** 2
```

### 1.4 Dependency Management

#### requirements.txt

List all dependencies with version numbers:

```
numpy>=1.20.0,<2.0.0
scipy>=1.7.0
matplotlib>=3.4.0
qutip>=4.6.0
```

#### environment.yml (Conda)

```yaml
name: quantum-project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=1.21
  - scipy=1.7
  - matplotlib=3.5
  - pip:
    - qutip>=4.6.0
```

#### pyproject.toml (Modern Python)

```toml
[project]
name = "quantum-project"
version = "1.0.0"
description = "Quantum state transfer simulation"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-cov",
    "black",
    "flake8",
]
```

---

## Part 2: Documentation

### 2.1 README.md Template

A comprehensive README should include:

```markdown
# Project Title

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Brief description of the project (1-2 sentences).

## Overview

More detailed description of what this project does, what problem it
solves, and its main features.

## Publication

If you use this code, please cite:

> Author, A., Author, B. (2026). Paper Title. *Journal Name*,
> Volume, Pages. DOI: 10.XXXX/XXXXX

BibTeX:
```bibtex
@article{author2026title,
  author = {Author, A. and Author, B.},
  title = {Paper Title},
  journal = {Journal Name},
  year = {2026},
  doi = {10.XXXX/XXXXX}
}
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Using pip

```bash
git clone https://github.com/username/project.git
cd project
pip install -e .
```

### Using conda

```bash
git clone https://github.com/username/project.git
cd project
conda env create -f environment.yml
conda activate quantum-project
```

## Quick Start

```python
from project import simulate

# Run a basic simulation
result = simulate(parameters={'coupling': 0.1})
result.plot()
```

## Usage

### Basic Example

[Provide a complete working example]

### Advanced Usage

[Link to detailed documentation]

## Reproducing Paper Results

To reproduce the figures in the paper:

```bash
python scripts/generate_figures.py
```

Results will be saved to `results/figures/`.

## Repository Structure

```
project/
├── src/          # Source code
├── tests/        # Test suite
├── data/         # Data files
├── notebooks/    # Jupyter notebooks
├── scripts/      # Standalone scripts
├── docs/         # Documentation
└── results/      # Output files
```

## Testing

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE).

## Acknowledgments

- Funding: [Grant information]
- Computational resources: [HPC acknowledgment]
- Contributors: [Names]
```

### 2.2 API Documentation

For each module, document:
- Purpose
- Classes and functions
- Parameters and return values
- Examples

Use tools like Sphinx or MkDocs to generate HTML documentation.

### 2.3 Installation Testing

Test installation on a clean environment:

```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# test_env\Scripts\activate   # Windows

# Install from source
pip install -e .

# Run tests
pytest tests/

# Run a minimal example
python -c "from project import main; main.run_example()"

# Clean up
deactivate
rm -rf test_env
```

---

## Part 3: Data Organization

### 3.1 Data Structure

Organize data clearly:

```
data/
├── raw/                    # Original, unmodified data
│   ├── experiment_001/
│   │   ├── parameters.json
│   │   └── measurements.csv
│   └── experiment_002/
│
├── processed/              # Cleaned/transformed data
│   ├── combined_results.h5
│   └── analysis_ready.csv
│
├── external/               # Third-party data
│   └── reference_data.csv
│
└── README.md               # Data documentation
```

### 3.2 Data Documentation

Create a data README:

```markdown
# Data Description

## Overview

This directory contains data for [project name].

## Files

### raw/experiment_001/

Raw data from first experimental run.

- **parameters.json**: Simulation parameters
  - `coupling_strength`: float, range [0, 1]
  - `decoherence_rate`: float, units 1/ns
  - `num_trajectories`: int

- **measurements.csv**: Measurement results
  - Columns: time, fidelity, entropy, purity
  - 10,000 rows, one per trajectory

### processed/combined_results.h5

HDF5 file containing processed results.

**Structure:**
```
/
├── parameters/
│   ├── coupling (array, shape: 50)
│   └── temperature (array, shape: 50)
├── results/
│   ├── fidelity (array, shape: 50, 50, 1000)
│   └── coherence_time (array, shape: 50, 50)
└── metadata/
    ├── creation_date (string)
    └── software_version (string)
```

## Data Collection

Describe how data was collected/generated.

## Processing Pipeline

Describe processing steps from raw to processed.

## Access and Licensing

[License information]
```

### 3.3 Data Formats

Use standard, open formats:

| Type | Recommended Formats |
|------|---------------------|
| Tabular | CSV, Parquet |
| Arrays | HDF5, NumPy (.npy, .npz) |
| Structured | JSON, YAML |
| Images | PNG, TIFF |
| Time series | CSV, HDF5 |

Avoid proprietary formats when possible.

---

## Part 4: Zenodo Submission

### 4.1 Account Setup

1. Go to https://zenodo.org
2. Click "Sign Up" or use GitHub/ORCID login
3. Complete profile with ORCID linkage
4. Verify email address

### 4.2 GitHub Integration

For automatic archiving of releases:

1. Go to https://zenodo.org/account/settings/github
2. Connect GitHub account
3. Enable the repository you want to archive
4. Create a release on GitHub
5. Zenodo automatically creates archive and DOI

### 4.3 Manual Upload

For data packages or non-GitHub content:

1. Click "New Upload" on Zenodo
2. Choose upload type (Dataset, Software, etc.)
3. Upload files (drag-and-drop or select)
4. Complete metadata form
5. Choose access rights and license
6. Submit

### 4.4 Metadata Best Practices

**Title:** Clear, descriptive title
```
Quantum State Transfer Simulation Code and Data
```

**Authors:** All contributors with ORCID
```
Smith, John (ORCID: 0000-0001-2345-6789)
Jones, Mary (ORCID: 0000-0002-3456-7890)
```

**Description:** Comprehensive abstract
```
This dataset contains the simulation code and data for the paper
"High-Fidelity Quantum State Transfer in Coupled Resonators"
published in Physical Review Letters. The repository includes:

1. Python simulation code (src/)
2. Raw simulation data (data/raw/)
3. Processed results (data/processed/)
4. Jupyter notebooks for analysis (notebooks/)
5. Scripts to reproduce figures (scripts/)

System requirements:
- Python 3.8+
- Dependencies listed in requirements.txt
- ~2GB disk space for data files

To reproduce paper results, see README.md.
```

**Keywords:** Help discoverability
```
quantum computing, quantum state transfer, quantum simulation,
coupled resonators, quantum optics, reproducibility
```

**Related Identifiers:**
```
DOI of paper: 10.1103/PhysRevLett.XXX.XXXXXX (isSupplementTo)
arXiv ID: 2026.XXXXX (isSupplementTo)
GitHub URL: https://github.com/user/repo (isVersionOf)
```

### 4.5 Version Management

Zenodo supports versioning:
- Each upload gets a version-specific DOI
- A concept DOI points to all versions
- Always cite the concept DOI for general reference

Example:
```
Version-specific: 10.5281/zenodo.1234567
Concept DOI: 10.5281/zenodo.1234566
```

### 4.6 License Selection

Choose appropriate license:

**For Code:**
- MIT: Permissive, simple
- Apache 2.0: Permissive with patent protection
- GPL 3.0: Copyleft, derivative work must be open

**For Data:**
- CC0: Public domain dedication
- CC-BY 4.0: Attribution required
- CC-BY-SA 4.0: Attribution + share-alike

---

## Part 5: FAIR Principles in Practice

### 5.1 Findable

- Assign persistent identifier (DOI via Zenodo)
- Use rich, standardized metadata
- Register in searchable repositories
- Include descriptive keywords

### 5.2 Accessible

- Use open, standard protocols
- Provide clear access conditions
- Ensure long-term availability
- Document any access restrictions

### 5.3 Interoperable

- Use standard file formats
- Use community vocabularies
- Include references to related resources
- Document data structure

### 5.4 Reusable

- Provide clear license
- Include provenance information
- Document collection/processing methods
- Follow community standards

---

## Part 6: Reproducibility Testing

### 6.1 End-to-End Verification

Test complete reproducibility:

```bash
# Clone repository (as external user would)
git clone https://github.com/user/repo.git
cd repo

# Create clean environment
conda env create -f environment.yml
conda activate project

# Download data (if external)
python scripts/download_data.py

# Run analysis
python scripts/run_analysis.py

# Generate figures
python scripts/generate_figures.py

# Compare outputs
python scripts/verify_outputs.py
```

### 6.2 Containerization (Optional)

For maximum reproducibility, use Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/run_analysis.py"]
```

### 6.3 Reproducibility Checklist

- [ ] Fresh environment installation works
- [ ] All tests pass
- [ ] Examples run correctly
- [ ] Main results can be regenerated
- [ ] Figures match paper figures
- [ ] Numerical values match paper values
- [ ] Compute time is reasonable/documented

---

## Part 7: Integration with Paper

### 7.1 Data Availability Statement

Add to paper:

```
Data Availability Statement:

The data that support the findings of this study are openly
available in Zenodo at https://doi.org/10.5281/zenodo.XXXXXXX.

Code Availability Statement:

The code used to generate these results is available at
https://github.com/user/repo and archived at
https://doi.org/10.5281/zenodo.XXXXXXX.
```

### 7.2 Citation Information

Create CITATION.cff file:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "Smith"
    given-names: "John"
    orcid: "https://orcid.org/0000-0001-2345-6789"
  - family-names: "Jones"
    given-names: "Mary"
    orcid: "https://orcid.org/0000-0002-3456-7890"
title: "Quantum State Transfer Simulation"
version: 1.0.0
doi: 10.5281/zenodo.XXXXXXX
date-released: 2026-01-15
url: "https://github.com/user/repo"
preferred-citation:
  type: article
  authors:
    - family-names: "Smith"
      given-names: "John"
    - family-names: "Jones"
      given-names: "Mary"
  doi: "10.1103/PhysRevLett.XXX.XXXXXX"
  journal: "Physical Review Letters"
  title: "High-Fidelity Quantum State Transfer"
  year: 2026
```

---

## Summary

Releasing code and data requires:

1. **Clean, organized code** with consistent style
2. **Comprehensive documentation** for installation and usage
3. **FAIR-compliant data** with clear provenance
4. **Persistent identifiers** via Zenodo DOI
5. **Appropriate licensing** for reuse
6. **Verification** that results are reproducible

Time invested in release preparation:
- Makes your research more impactful
- Increases citations
- Enables collaboration
- Builds professional reputation
- Contributes to open science

---

*"Reproducibility is the foundation upon which the credibility of science is built."*
