# Results Documentation Guide

## Introduction

This guide provides comprehensive instructions for documenting your research results in preparation for publication. Effective documentation transforms your research findings into a coherent, reproducible, and impactful scientific contribution.

---

## Part 1: Technical Report Writing

### 1.1 Structure and Flow

A well-organized paper tells a story:

```
Introduction
    "Here's an important problem..."
    "Previous work has done X, but gap Y remains..."
    "We contribute Z..."
        ↓
Methods
    "Here's how we did it..."
    (Enough detail to reproduce)
        ↓
Results
    "Here's what we found..."
    (Facts, not interpretation)
        ↓
Discussion
    "Here's what it means..."
    (Interpretation, context, limitations)
        ↓
Conclusion
    "Here are the key takeaways..."
```

### 1.2 Writing the Abstract

The abstract is often written last but read first. It should standalone and include:

**Structure (IMRaD):**
1. **Introduction** (1-2 sentences): Context and gap
2. **Methods** (1-2 sentences): Approach
3. **Results** (2-3 sentences): Key findings with numbers
4. **Discussion** (1-2 sentences): Implications

**Example Abstract Template:**
```
[Background context - why does this matter?]. However, [gap or problem].
Here we [approach/contribution]. Using [method], we [main action].
We find that [main result 1 with quantitative detail] and [main result 2].
Furthermore, [secondary finding]. Our results demonstrate [implication]
and suggest [future direction or application].
```

**Word Counts by Journal Type:**
| Journal Type | Abstract Length |
|--------------|-----------------|
| Physical Review Letters | 150 words |
| Nature/Science | 150-200 words |
| PRX Quantum | 200 words |
| Full research article | 200-300 words |

### 1.3 Writing the Introduction

**Paragraph Structure:**

**Paragraph 1: Big Picture**
- Start broad (field importance)
- Establish context
- Engage the reader

**Paragraph 2-3: Literature Review**
- What has been done?
- What approaches exist?
- Build toward the gap

**Paragraph 4: The Gap**
- What's missing?
- Why is it important?
- Clear statement of problem

**Paragraph 5: Your Contribution**
- What do you do?
- Brief preview of approach
- Key results teaser

**Paragraph 6: Paper Organization (optional)**
- Section overview
- Guide the reader

### 1.4 Writing the Methods Section

**Guiding Principle:** Someone with appropriate expertise should be able to reproduce your work.

**Structure:**
```markdown
## Methods

### Experimental Setup
[Physical description of apparatus/computational environment]

### Sample/System Preparation
[How the system was prepared]

### Measurement Procedure
[Step-by-step experimental protocol]

### Data Analysis
[Processing and statistical methods]

### Uncertainty Analysis
[How errors were quantified]
```

**Level of Detail:**
- Include all parameters that affect results
- Reference standard methods, detail novel ones
- Specify software versions and settings
- Provide access to code/data

### 1.5 Writing the Results Section

**Guidelines:**
- Present findings objectively (facts, not interpretation)
- Follow logical order (not necessarily chronological)
- Reference figures and tables explicitly
- Report statistical results properly

**Result Statement Template:**
```
[What we measured/computed] revealed [main finding] (Figure X).
Specifically, [quantitative detail with uncertainty].
This represents [comparison/context, e.g., "a 3x improvement over..."].
```

**Statistical Reporting:**
| Statistic | Reporting Format |
|-----------|-----------------|
| Mean ± SE | $\bar{x} = 0.95 \pm 0.02$ |
| Confidence interval | 95% CI: [0.91, 0.99] |
| p-value | $p < 0.001$ or $p = 0.003$ |
| Effect size | Cohen's $d = 0.85$ (large) |
| Correlation | $r = 0.78$, $p < 0.001$ |

### 1.6 Writing the Discussion

**Structure:**

**Opening:** Restate main finding in context
```
Our results demonstrate that [main contribution],
which [addresses the gap stated in introduction].
```

**Interpretation:** What do results mean?
- Connect findings to existing knowledge
- Explain unexpected results
- Discuss mechanisms

**Comparison:** How does this relate to prior work?
- Agreements with literature
- Disagreements (and why)
- Advances over previous methods

**Limitations:** What are the caveats?
- Be honest and specific
- Explain impact on conclusions
- Distinguish fundamental vs. practical limitations

**Implications:** Why does this matter?
- Applications
- Future research directions
- Broader significance

### 1.7 Writing the Conclusion

**Keep it concise (1-2 paragraphs):**
- Summarize key contributions (not all results)
- Emphasize significance
- Suggest future directions
- End with impact statement

**Avoid:**
- Introducing new information
- Repeating the abstract verbatim
- Vague statements
- Overclaiming

---

## Part 2: Publication-Quality Figures

### 2.1 Figure Design Principles

**The ACCENT Principles:**
- **A**pprehension: Quick grasp of message
- **C**larity: Unambiguous presentation
- **C**onsistency: Uniform style throughout
- **E**fficiency: High data-to-ink ratio
- **N**ecessity: Every element has purpose
- **T**ruthfulness: Accurate representation

### 2.2 Technical Specifications

**Resolution and Size:**
| Output | Resolution | Figure Width |
|--------|------------|--------------|
| Screen/Web | 72-150 dpi | Variable |
| Print (single column) | 300-600 dpi | 3.5 inches (8.9 cm) |
| Print (double column) | 300-600 dpi | 7 inches (17.8 cm) |
| Vector (PDF, SVG) | N/A | Set by points |

**File Formats:**
| Format | Use Case | Notes |
|--------|----------|-------|
| PDF | Vector graphics | Best for line plots |
| SVG | Web, editing | Scalable |
| PNG | Raster, screenshots | Use high DPI |
| TIFF | Archival | Large files |
| EPS | Legacy journals | Vector |

### 2.3 Figure Templates

**matplotlib Publication Settings:**

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

def set_publication_style():
    """Configure matplotlib for publication-quality figures."""

    # Use LaTeX for text rendering
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
    })

    # Figure and axes properties
    mpl.rcParams.update({
        # Figure
        'figure.figsize': (3.5, 2.5),  # Single column
        'figure.dpi': 300,
        'figure.facecolor': 'white',

        # Font sizes
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,

        # Lines
        'lines.linewidth': 1.0,
        'lines.markersize': 4,

        # Axes
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',

        # Legend
        'legend.frameon': False,
        'legend.loc': 'best',

        # Saving
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

# Colorblind-friendly palette
COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#D55E00',
    'purple': '#CC78BC',
    'brown': '#CA9161',
    'pink': '#FBAFE4',
    'gray': '#949494',
}
```

### 2.4 Common Figure Types

**Data with Error Bars:**
```python
def plot_with_errors(x, y, yerr, label=None, save_path=None):
    """Create publication-quality plot with error bars."""
    set_publication_style()

    fig, ax = plt.subplots()

    ax.errorbar(x, y, yerr=yerr, fmt='o-',
                capsize=2, capthick=0.8,
                markersize=4, linewidth=1,
                color=COLORS['blue'],
                label=label)

    ax.set_xlabel(r'Parameter $\theta$ (units)')
    ax.set_ylabel(r'Measurement $M$ (units)')

    if label:
        ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax
```

**Multi-panel Figure:**
```python
def create_multipanel_figure(data_list, save_path=None):
    """Create multi-panel figure with consistent styling."""
    set_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.flatten()

    panel_labels = ['a', 'b', 'c', 'd']

    for ax, data, label in zip(axes, data_list, panel_labels):
        ax.plot(data['x'], data['y'], '-', color=COLORS['blue'])

        # Panel label
        ax.text(-0.15, 1.05, f'({label})',
                transform=ax.transAxes,
                fontsize=10, fontweight='bold')

        ax.set_xlabel(data.get('xlabel', ''))
        ax.set_ylabel(data.get('ylabel', ''))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes
```

### 2.5 Figure Captions

**Structure:**
1. **Title** (bold): Brief description of what the figure shows
2. **Description**: What is being plotted
3. **Methods note**: Key experimental/computational details
4. **Legend explanation**: What symbols/colors mean
5. **Panel descriptions**: If multi-panel, describe each

**Example:**
```
**Figure 1. Quantum gate fidelity as a function of pulse duration.**
(a) Single-qubit X gate fidelity measured by randomized benchmarking
(blue circles) compared with numerical simulation (solid line). Error bars
represent standard error of the mean (n = 100 randomization sequences).
(b) Two-qubit CNOT gate fidelity showing optimal performance at
τ = 45 ns. Shaded region indicates 95% confidence interval from
bootstrap resampling. Dashed line shows theoretical limit from T₂
decoherence.
```

---

## Part 3: Supplementary Materials

### 3.1 What to Include

**Supplementary Figures:**
- Additional data supporting main findings
- Alternative visualizations
- Extended parameter sweeps
- Intermediate processing steps

**Supplementary Tables:**
- Full parameter lists
- Extended statistical results
- Raw numerical data
- Equipment specifications

**Supplementary Methods:**
- Detailed derivations
- Algorithm pseudocode
- Extended protocols
- Calibration procedures

**Supplementary Data:**
- Raw datasets
- Processed data files
- Configuration files
- Log files

### 3.2 Organization

**Directory Structure:**
```
supplementary_materials/
├── supplementary_information.pdf
│   ├── Supplementary Methods
│   ├── Supplementary Figures S1-S10
│   ├── Supplementary Tables S1-S5
│   └── Supplementary References
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── [raw data files]
│   └── processed/
│       └── [processed data files]
├── code/
│   ├── README.md
│   └── analysis_scripts/
└── media/
    └── [additional multimedia]
```

### 3.3 Data Documentation

**README Template for Data:**
```markdown
# Data Description

## Overview
Brief description of what this data represents.

## Files
| File | Description | Format | Size |
|------|-------------|--------|------|
| data_001.csv | Main experimental results | CSV | 1.2 MB |
| data_002.h5 | Time series measurements | HDF5 | 50 MB |

## Data Structure
Describe columns, units, and any codes used.

## Collection
When and how was data collected?

## Processing
What processing steps were applied?

## Usage
How to load and use this data.

## Citation
How to cite this dataset.
```

---

## Part 4: Reproducibility Package

### 4.1 Components

A complete reproducibility package includes:

1. **Code Repository**
   - All analysis code
   - Documentation
   - Tests
   - Environment specification

2. **Data Archive**
   - Raw data (if shareable)
   - Processed data
   - Metadata

3. **Documentation**
   - Step-by-step instructions
   - Expected outputs
   - Troubleshooting guide

4. **Computational Environment**
   - requirements.txt
   - conda environment.yml
   - Docker container (optional)

### 4.2 Creating the Package

**Step 1: Organize Files**
```bash
mkdir reproducibility_package
cd reproducibility_package

# Copy essential files
cp -r ../code ./
cp -r ../data/processed ./data
cp ../manuscript/figures/*.pdf ./figures
```

**Step 2: Create Environment File**
```yaml
# environment.yml
name: research_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy=1.24
  - scipy=1.10
  - matplotlib=3.7
  - pandas=2.0
  - jupyter=1.0
  - pip:
    - custom-package==1.0.0
```

**Step 3: Create Reproduction Script**
```python
#!/usr/bin/env python
"""
reproduce_results.py

This script reproduces all main results from the paper.
Expected runtime: ~30 minutes on a standard laptop.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Verify environment
    check_dependencies()

    # Run analyses
    print("Step 1: Processing raw data...")
    run_script("scripts/01_process_data.py")

    print("Step 2: Running main analysis...")
    run_script("scripts/02_analyze.py")

    print("Step 3: Generating figures...")
    run_script("scripts/03_generate_figures.py")

    print("Step 4: Computing statistics...")
    run_script("scripts/04_statistics.py")

    print("\nReproduction complete!")
    print("Results saved to: ./outputs/")
    print("Figures saved to: ./figures/")

def check_dependencies():
    """Verify all required packages are installed."""
    required = ['numpy', 'scipy', 'matplotlib', 'pandas']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {missing}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)

def run_script(script_path):
    """Run a Python script and check for errors."""
    result = subprocess.run([sys.executable, script_path],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}:")
        print(result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 4.3 Archiving to Repository

**Zenodo Deposit Checklist:**
- [ ] Create Zenodo account (link to ORCID)
- [ ] Connect GitHub repository (optional)
- [ ] Upload files or link repository
- [ ] Fill in metadata (title, authors, description)
- [ ] Add keywords and subject areas
- [ ] Select license
- [ ] Link to related publications
- [ ] Reserve DOI before publication
- [ ] Publish after paper acceptance

**Metadata for Archive:**
```json
{
  "title": "Data and Code for: [Paper Title]",
  "upload_type": "dataset",
  "description": "Complete data and code to reproduce...",
  "creators": [
    {
      "name": "Last, First",
      "affiliation": "Institution",
      "orcid": "0000-0000-0000-0000"
    }
  ],
  "keywords": ["quantum computing", "simulation", "python"],
  "license": "CC-BY-4.0",
  "related_identifiers": [
    {
      "identifier": "10.1234/journal.12345",
      "relation": "isSupplementTo",
      "resource_type": "publication-article"
    }
  ]
}
```

---

## Part 5: Quality Checklist

### 5.1 Technical Report
- [ ] All sections complete
- [ ] Abstract accurately summarizes paper
- [ ] Introduction clearly states contribution
- [ ] Methods enable reproduction
- [ ] Results presented objectively
- [ ] Discussion interprets findings
- [ ] References complete and formatted
- [ ] Writing is clear and concise

### 5.2 Figures
- [ ] Publication-quality resolution
- [ ] Consistent styling throughout
- [ ] All elements labeled
- [ ] Error bars/uncertainty shown
- [ ] Colorblind accessible
- [ ] Captions complete and informative

### 5.3 Supplementary Materials
- [ ] Well organized
- [ ] Properly referenced from main text
- [ ] Self-contained descriptions
- [ ] Data properly documented

### 5.4 Reproducibility
- [ ] All code available
- [ ] Data available or access described
- [ ] Environment specified
- [ ] Instructions tested
- [ ] Expected outputs documented

---

## Navigation

| Previous | Current | Next |
|----------|---------|------|
| Week 228 README | Results Documentation Guide | [Technical Report Template](./Templates/Technical_Report_Template.md) |
