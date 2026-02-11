# Reproducibility Guide Template

## Project: [Project Name]

**Version:** [version]
**Date:** [date]
**Author:** [name] (ORCID: [0000-0000-0000-0000])

---

## Purpose

This document provides complete instructions to reproduce all computational results presented in [thesis/paper title].

---

## Quick Reproduction

For those who want to reproduce results quickly:

```bash
# Clone and setup
git clone [repository-url]
cd [repository-name]

# Option A: Docker (recommended)
docker-compose up reproduce

# Option B: Local installation
pip install -e .
python scripts/reproduce_all.py
```

**Expected runtime:** [X hours/days]
**Expected output:** [location of results]

---

## System Requirements

### Hardware

| Component | Minimum | Recommended | Used in Publication |
|-----------|---------|-------------|---------------------|
| CPU | 4 cores | 16 cores | Intel Xeon E5-2680 |
| RAM | 16 GB | 64 GB | 128 GB |
| Storage | 50 GB | 200 GB | 1 TB SSD |
| GPU | None | NVIDIA RTX 3080 | NVIDIA A100 |

### Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10.x | Runtime |
| CUDA | 11.8 | GPU acceleration |
| Docker | 20.10+ | Containerization |

---

## Environment Setup

### Method 1: Docker (Recommended)

Docker provides the most reliable reproduction environment.

```bash
# Pull pre-built image
docker pull [dockerhub-username]/[image-name]:[tag]

# Or build locally
docker build -t [image-name] .

# Verify installation
docker run --rm [image-name] python -c "import package; print('OK')"
```

### Method 2: Conda

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate [env-name]

# Verify
python -c "import package; print('OK')"
```

### Method 3: pip + venv

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify
python -c "import package; print('OK')"
```

---

## Data Preparation

### Obtaining Data

| Dataset | Size | Location | DOI |
|---------|------|----------|-----|
| Dataset 1 | X GB | [URL] | 10.5281/... |
| Dataset 2 | X GB | [URL] | 10.5281/... |

```bash
# Download data
python scripts/download_data.py

# Or manually
wget [data-url] -O data/raw/dataset.zip
unzip data/raw/dataset.zip -d data/raw/
```

### Data Verification

```bash
# Verify data integrity
python scripts/verify_data.py

# Expected output:
# Checking dataset1... OK (SHA256: abc123...)
# Checking dataset2... OK (SHA256: def456...)
# All datasets verified
```

---

## Reproduction Steps

### Complete Reproduction

To reproduce all results:

```bash
python scripts/reproduce_all.py

# This will:
# 1. Verify environment
# 2. Run all analyses
# 3. Generate all figures
# 4. Compare with expected outputs
# 5. Generate reproduction report
```

**Expected runtime:** [X hours]

### Step-by-Step Reproduction

#### Step 1: [Name]

**Description:** [What this step does]

**Command:**
```bash
python scripts/step_1.py --param value
```

**Input:** [Required inputs]
**Output:** [Generated outputs]
**Runtime:** [Expected time]

#### Step 2: [Name]

[Continue for each step]

---

## Reproducing Specific Results

### Figure X.Y: [Figure Title]

**Thesis/Paper Location:** Figure X.Y, page XX

**Reproduction Command:**
```bash
python scripts/figures/figure_X_Y.py
```

**Input Data:** `data/processed/[filename]`
**Output:** `results/figures/figure_X_Y.pdf`
**Runtime:** [time]

**Verification:**
The generated figure should match the published version. Minor differences may occur due to:
- Random seed variations (if applicable)
- Floating-point precision differences
- Font rendering differences

### Table X.Y: [Table Title]

**Reproduction Command:**
```bash
python scripts/tables/table_X_Y.py
```

**Output:** `results/tables/table_X_Y.csv`

**Expected Values:**

| Column | Expected | Tolerance |
|--------|----------|-----------|
| Value 1 | 0.95 | ± 0.01 |
| Value 2 | 1.23 | ± 0.05 |

---

## Random Seeds and Reproducibility

### Controlling Randomness

All random processes use explicit seeds:

```python
# In config.py
RANDOM_SEED = 42

# Usage in code
import numpy as np
import random

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

### Known Sources of Non-Determinism

| Source | Mitigation | Impact |
|--------|------------|--------|
| GPU operations | Set CUDA seeds | Minor (< 0.1%) |
| Parallel execution | Fixed worker seeds | Minor |
| External APIs | Cached responses | None |

---

## Computational Resources

### Resource Requirements by Analysis

| Analysis | CPU Hours | GPU Hours | Memory | Storage |
|----------|-----------|-----------|--------|---------|
| Analysis 1 | 10 | 0 | 16 GB | 10 GB |
| Analysis 2 | 5 | 20 | 32 GB | 50 GB |
| Full reproduction | 50 | 40 | 64 GB | 100 GB |

### Reducing Resource Requirements

For limited resources:

```bash
# Use smaller dataset
python scripts/reproduce_all.py --dataset-size small

# Skip GPU-intensive analyses
python scripts/reproduce_all.py --skip-gpu

# Run subset of analyses
python scripts/reproduce_all.py --analyses "fig_3_1,fig_3_2"
```

---

## Verification

### Automated Verification

```bash
python scripts/verify_reproduction.py

# Output:
# Checking Figure 3.1... PASS
# Checking Figure 3.2... PASS
# Checking Table 4.1... PASS (within tolerance)
# ...
# Summary: 15/15 checks passed
```

### Manual Verification

Compare your outputs with reference outputs:

```bash
# Visual comparison of figures
python scripts/compare_figures.py results/figures/ reference/figures/

# Numerical comparison of tables
python scripts/compare_tables.py results/tables/ reference/tables/ --tolerance 0.01
```

---

## Troubleshooting

### Common Issues

#### Issue: Out of Memory

**Symptom:** `MemoryError` or process killed

**Solution:**
```bash
# Reduce batch size
python script.py --batch-size 16

# Use memory-mapped arrays
python script.py --mmap
```

#### Issue: CUDA Errors

**Symptom:** `CUDA out of memory` or `CUDA driver version insufficient`

**Solution:**
```bash
# Check CUDA version
nvidia-smi

# Use CPU fallback
python script.py --device cpu
```

#### Issue: Package Version Conflicts

**Symptom:** `ImportError` or `AttributeError`

**Solution:**
```bash
# Use Docker for guaranteed compatibility
docker run [image] python script.py

# Or recreate environment
pip install -r requirements.txt --force-reinstall
```

### Getting Help

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/[username]/[repo]/issues)
2. Search closed issues for solutions
3. Open new issue with:
   - System information (`python scripts/system_info.py`)
   - Full error message
   - Steps to reproduce

---

## Version Information

### Software Versions Used

```
python==3.10.8
numpy==1.24.3
scipy==1.10.1
matplotlib==3.7.1
[other packages...]
```

### Hardware Used

```
CPU: [model]
GPU: [model]
OS: [version]
```

To capture your system info:
```bash
python scripts/system_info.py > system_info.txt
```

---

## Acknowledgments

This research was supported by:
- [Funding source 1]
- [Funding source 2]

Computational resources were provided by:
- [Computing facility]

---

## Contact

For questions about reproduction:
- **Email:** [email]
- **GitHub:** [username]
- **ORCID:** [0000-0000-0000-0000]

---

*Last updated: [date]*
