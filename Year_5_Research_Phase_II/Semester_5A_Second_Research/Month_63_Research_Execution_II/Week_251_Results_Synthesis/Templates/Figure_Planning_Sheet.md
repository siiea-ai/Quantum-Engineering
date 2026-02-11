# Figure Planning Sheet

## Publication-Quality Figure Development

---

## Project Information

| Field | Details |
|-------|---------|
| **Project Title** | |
| **Target Venue** | |
| **Figure Limit** | (if applicable) |
| **Date** | |

---

## Part I: Figure Suite Overview

### Figure List

| # | Working Title | Type | Priority | Status |
|---|---------------|------|----------|--------|
| 1 | | | Must have | |
| 2 | | | Must have | |
| 3 | | | Should have | |
| 4 | | | Should have | |
| 5 | | | Nice to have | |
| 6 | | | Nice to have | |

### Figure Types

- [ ] **Conceptual**: Explains the main idea visually
- [ ] **Results**: Presents data/outcomes
- [ ] **Comparison**: Compares with baselines/prior work
- [ ] **Method**: Illustrates the approach
- [ ] **Schematic**: Shows system architecture
- [ ] **Workflow**: Depicts process/algorithm

### Key Question

**If readers could only see one figure, which should it be and why?**
>

---

## Part II: Individual Figure Planning

### Figure 1

**Working Title:** _________________________________

**Type:** Conceptual / Results / Comparison / Method / Schematic / Workflow

**Purpose:**
>

**Key Message (one sentence):**
>

**Content Description:**
>

**Data/Input Required:**
- [ ]
- [ ]
- [ ]

**Sketch:**
```
[Draw rough sketch here or attach image]




```

**Design Notes:**
- Color scheme:
- Special formatting:
- Size (column width / full page):

**Caption Draft:**
>

**Status:** [ ] Concept [ ] Sketch [ ] Draft [ ] Revised [ ] Final

---

### Figure 2

**Working Title:** _________________________________

**Type:** Conceptual / Results / Comparison / Method / Schematic / Workflow

**Purpose:**
>

**Key Message (one sentence):**
>

**Content Description:**
>

**Data/Input Required:**
- [ ]
- [ ]
- [ ]

**Sketch:**
```
[Draw rough sketch here or attach image]




```

**Design Notes:**
- Color scheme:
- Special formatting:
- Size (column width / full page):

**Caption Draft:**
>

**Status:** [ ] Concept [ ] Sketch [ ] Draft [ ] Revised [ ] Final

---

### Figure 3

**Working Title:** _________________________________

**Type:** Conceptual / Results / Comparison / Method / Schematic / Workflow

**Purpose:**
>

**Key Message (one sentence):**
>

**Content Description:**
>

**Data/Input Required:**
- [ ]
- [ ]
- [ ]

**Sketch:**
```
[Draw rough sketch here or attach image]




```

**Design Notes:**
- Color scheme:
- Special formatting:
- Size (column width / full page):

**Caption Draft:**
>

**Status:** [ ] Concept [ ] Sketch [ ] Draft [ ] Revised [ ] Final

---

### Figure 4

**Working Title:** _________________________________

**Type:** Conceptual / Results / Comparison / Method / Schematic / Workflow

**Purpose:**
>

**Key Message (one sentence):**
>

**Content Description:**
>

**Data/Input Required:**
- [ ]
- [ ]
- [ ]

**Sketch:**
```
[Draw rough sketch here or attach image]




```

**Design Notes:**
- Color scheme:
- Special formatting:
- Size (column width / full page):

**Caption Draft:**
>

**Status:** [ ] Concept [ ] Sketch [ ] Draft [ ] Revised [ ] Final

---

## Part III: Style Guide

### Color Palette

**Primary colors:**
| Use | Color | Hex | Meaning |
|-----|-------|-----|---------|
| Main data | | #______ | |
| Comparison | | #______ | |
| Tertiary | | #______ | |
| Background | | #______ | |
| Emphasis | | #______ | |

**Colorblind considerations:**
- [ ] Tested with colorblind simulation
- [ ] Uses shapes/patterns as backup

### Typography

| Element | Font | Size |
|---------|------|------|
| Axis labels | | |
| Tick labels | | |
| Legend | | |
| Annotations | | |
| Title (if any) | | |

### Consistent Elements

| Element | Style |
|---------|-------|
| Line width | |
| Marker size | |
| Error bar style | |
| Grid style | |
| Legend position | |

---

## Part IV: Quantum Computing Specific Elements

### Quantum Circuits

**Tool:** Qiskit / Cirq / Quantikz / TikZ

**Style decisions:**
- Wire color:
- Gate fill:
- Control style:
- Measurement symbol:

**Circuit template:**
```
[Include standard circuit template code]
```

### Bloch Sphere

**Tool:** Qiskit / Custom matplotlib

**Style decisions:**
- Sphere color:
- State vector color:
- Axis labels:

### Entanglement Diagrams

**Style decisions:**
- Node representation:
- Edge representation:
- Labels:

---

## Part V: Data Visualization Guidelines

### For Performance Plots

**Best practices:**
- [ ] Log-log for scaling
- [ ] Clear axis labels with units
- [ ] Reference lines (e.g., theoretical scaling)
- [ ] Error bars when applicable
- [ ] Legend in consistent position

**Template:**
```python
# Standard performance plot template
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(x, y, 'o-', label='This work')
ax.plot(x, baseline, 's--', label='Baseline')
ax.set_xlabel('Problem size n')
ax.set_ylabel('Time (s)')
ax.legend(loc='upper left')
ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
```

### For Comparison Bar Charts

**Best practices:**
- [ ] Consistent ordering
- [ ] Highlight our method
- [ ] Include error bars
- [ ] Clear labels

### For Heatmaps/2D Plots

**Best practices:**
- [ ] Appropriate colormap
- [ ] Colorbar with label
- [ ] Clear axis labels
- [ ] Contours if helpful

---

## Part VI: Caption Templates

### Results Figure Caption

"**Figure X: [Descriptive title].** (a) [Description of panel a]. (b) [Description of panel b]. [Key observation from the figure]. [Connection to main result]. Parameters: [list any relevant parameters]."

### Comparison Figure Caption

"**Figure X: Comparison with prior methods.** [What is being compared]. [Main finding from comparison]. [Any caveats]. Data for [baseline method] from [reference]."

### Method Figure Caption

"**Figure X: Overview of our approach.** (a) [Step 1 description]. (b) [Step 2 description]. (c) [Final step]. [Key advantage of this approach]."

### Conceptual Figure Caption

"**Figure X: [Concept name].** [Description of what the figure shows]. [How this relates to the main contribution]. [Any additional context needed]."

---

## Part VII: Quality Checklist

### Before Finalization

For each figure:

**Content:**
- [ ] Message is clear to target audience
- [ ] All data is accurate and validated
- [ ] Labels are complete and correct
- [ ] Units are specified where needed

**Design:**
- [ ] Readable at intended size
- [ ] Colors are consistent with style guide
- [ ] Colors are colorblind accessible
- [ ] Text is large enough
- [ ] Layout is balanced

**Technical:**
- [ ] Resolution is sufficient (300+ DPI)
- [ ] Format is appropriate (PDF/EPS for vector, PNG for raster)
- [ ] File size is reasonable
- [ ] Fonts are embedded

**Caption:**
- [ ] Caption is self-contained
- [ ] All panels are described
- [ ] Key observation is stated
- [ ] Parameters are documented

---

## Part VIII: Revision Tracking

### Figure Revision Log

| Figure | Version | Date | Changes | Reviewer |
|--------|---------|------|---------|----------|
| 1 | v1 | | Initial draft | |
| 1 | v2 | | | |
| 2 | v1 | | | |
| | | | | |

### Feedback Received

| Figure | Feedback | Source | Action | Status |
|--------|----------|--------|--------|--------|
| | | | | Done/Pending |
| | | | | |

---

## Part IX: Export Specifications

### File Formats

| Venue Type | Preferred Format | Backup |
|------------|------------------|--------|
| Journal | PDF/EPS (vector) | TIFF 300 DPI |
| Conference | PDF | PNG 300 DPI |
| arXiv | PDF | PNG |

### Export Settings

```python
# Standard export settings
plt.savefig('figure.pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1)

# For high-resolution raster
plt.savefig('figure.png',
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=False)
```

### File Organization

```
figures/
├── raw/          # Raw output files
├── drafts/       # Draft versions
├── final/        # Final versions
├── source/       # Source code/scripts
└── data/         # Data used in figures
```

---

## Part X: Figure Summary

### Status Summary

| Figure | Concept | Data Ready | Draft | Reviewed | Final |
|--------|---------|------------|-------|----------|-------|
| 1 | [x] | [ ] | [ ] | [ ] | [ ] |
| 2 | [ ] | [ ] | [ ] | [ ] | [ ] |
| 3 | [ ] | [ ] | [ ] | [ ] | [ ] |
| 4 | [ ] | [ ] | [ ] | [ ] | [ ] |

### Priority for Completion

1. Figure ___: (most critical)
2. Figure ___:
3. Figure ___:
4. Figure ___:

### Outstanding Tasks

| Task | Figure | Due |
|------|--------|-----|
| | | |
| | | |
| | | |

---

*Figure Planning Sheet v1.0*

*Good figures are worth a thousand words. Plan carefully.*
