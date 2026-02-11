# Publication Figure Specifications

## Overview

This document provides comprehensive specifications for creating publication-quality figures. Following these guidelines ensures your figures meet journal requirements, communicate effectively, and are accessible to all readers.

---

## 1. Technical Requirements

### 1.1 Resolution

| Output Type | Minimum Resolution | Recommended |
|-------------|-------------------|-------------|
| Line art (plots) | 600 dpi | 1200 dpi |
| Halftone (photos) | 300 dpi | 600 dpi |
| Combination | 600 dpi | 600 dpi |
| Web display | 72-150 dpi | 150 dpi |

### 1.2 Figure Dimensions

**Single Column:**
- Width: 3.5 inches (8.9 cm, 21 picas)
- Common journals: PRL, Nature, Science

**1.5 Column:**
- Width: 5.5 inches (14 cm, 33 picas)
- Intermediate size

**Double Column:**
- Width: 7.0-7.5 inches (17.8-19 cm, 42-45 picas)
- Full page width

**Height:**
- Maximum: 9.0 inches (22.9 cm)
- Typical: Maintain aspect ratio ~1:1 to 4:3

### 1.3 File Formats

| Format | Use Case | Advantages | Disadvantages |
|--------|----------|------------|---------------|
| PDF | Vector graphics | Scalable, small size | Some viewers struggle |
| EPS | Legacy journals | Universal, vector | Outdated |
| SVG | Web, editing | Scalable, editable | Not all journals accept |
| TIFF | Raster images | Lossless, universal | Large file size |
| PNG | Raster, web | Lossless, small | Not for print journals |
| JPEG | Photos only | Small size | Lossy compression |

**Recommendation:** Use PDF or EPS for plots, TIFF for photographs.

### 1.4 Color Specifications

**Color Modes:**
- RGB: For web and most digital submissions
- CMYK: For print-only (check journal requirements)

**Color Profiles:**
- sRGB for RGB images
- Verify colors display correctly across systems

---

## 2. Design Principles

### 2.1 Clarity

**Typography:**
- Minimum font size: 6 pt (after scaling)
- Preferred: 8-10 pt
- Use sans-serif for labels (Helvetica, Arial)
- Use serif for mathematical text (if needed)
- Consistent fonts throughout all figures

**Labels:**
- All axes must be labeled
- Include units: "Time (ms)" not just "Time"
- Use consistent notation with main text
- Panel labels: (a), (b), (c) in consistent position

**Legends:**
- Keep inside figure when possible
- Use clear, distinguishable symbols
- Order matches data appearance

### 2.2 Data Representation

**Error Bars:**
- Always include when representing uncertainty
- State what they represent (SEM, SD, CI)
- Use caps appropriate to data density
- For dense data, use error bands/shading

**Line Types:**
- Solid: Primary/experimental data
- Dashed: Theory/model
- Dotted: Reference lines
- Minimum width: 0.5 pt (after scaling)

**Markers:**
- Use distinguishable shapes
- Minimum size: 3-4 pt
- Don't overcrowd with markers

### 2.3 Accessibility

**Colorblind-Safe Palettes:**

```python
# Recommended palette (colorblind-friendly)
COLORS = {
    'blue':   '#0173B2',  # Primary
    'orange': '#DE8F05',  # Secondary
    'green':  '#029E73',  # Tertiary
    'red':    '#D55E00',  # Alert/emphasis
    'purple': '#CC78BC',  # Alternative
    'brown':  '#CA9161',  # Alternative
    'gray':   '#949494',  # Neutral
}

# Avoid: pure red/green combinations
# Avoid: rainbow colormaps (use viridis, plasma, cividis)
```

**Additional Accessibility:**
- Use shape AND color to distinguish data
- Ensure sufficient contrast
- Test with colorblindness simulators
- Provide alt-text for digital versions

---

## 3. Figure Types

### 3.1 Data Plots

**Line Plots:**
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Data with error
x = np.linspace(0, 10, 50)
y = np.sin(x)
yerr = 0.1 * np.random.randn(len(x))

ax.plot(x, y, '-', color='#0173B2', linewidth=1, label='Experiment')
ax.fill_between(x, y-yerr, y+yerr, alpha=0.3, color='#0173B2')
ax.plot(x, np.sin(x), '--', color='#DE8F05', linewidth=1, label='Theory')

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Signal (a.u.)')
ax.legend(loc='upper right', frameon=False)

# Clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figure_1.pdf', dpi=300, bbox_inches='tight')
```

**Scatter Plots:**
```python
fig, ax = plt.subplots(figsize=(3.5, 3.5))

# Data with different groups
groups = ['Control', 'Treatment A', 'Treatment B']
colors = ['#949494', '#0173B2', '#DE8F05']
markers = ['o', 's', '^']

for i, (group, color, marker) in enumerate(zip(groups, colors, markers)):
    x = np.random.randn(20) + i
    y = np.random.randn(20) + i
    ax.scatter(x, y, c=color, marker=marker, s=30, label=group, alpha=0.7)

ax.set_xlabel('Metric 1')
ax.set_ylabel('Metric 2')
ax.legend(loc='upper left', frameon=False)

plt.tight_layout()
plt.savefig('figure_2.pdf', dpi=300, bbox_inches='tight')
```

**Bar Charts:**
```python
fig, ax = plt.subplots(figsize=(3.5, 2.5))

categories = ['Method A', 'Method B', 'Method C', 'Ours']
values = [0.82, 0.87, 0.91, 0.96]
errors = [0.03, 0.04, 0.02, 0.02]
colors = ['#949494', '#949494', '#949494', '#0173B2']

bars = ax.bar(categories, values, yerr=errors, capsize=3,
              color=colors, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Fidelity')
ax.set_ylim([0.7, 1.0])

# Add significance markers
ax.plot([2, 3], [0.97, 0.97], 'k-', linewidth=0.5)
ax.text(2.5, 0.975, '***', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('figure_3.pdf', dpi=300, bbox_inches='tight')
```

### 3.2 Heatmaps

```python
import seaborn as sns

fig, ax = plt.subplots(figsize=(4, 3.5))

# Sample data
data = np.random.randn(8, 8)

# Use colorblind-friendly colormap
sns.heatmap(data, cmap='viridis', center=0,
            square=True, linewidths=0.5,
            cbar_kws={'label': 'Value (units)', 'shrink': 0.8},
            ax=ax)

ax.set_xlabel('Parameter A')
ax.set_ylabel('Parameter B')

plt.tight_layout()
plt.savefig('figure_4.pdf', dpi=300, bbox_inches='tight')
```

### 3.3 Multi-Panel Figures

```python
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

# Panel labels
panel_labels = ['a', 'b', 'c', 'd']

for ax, label in zip(axes.flat, panel_labels):
    # Add panel label
    ax.text(-0.15, 1.05, f'({label})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

    # Placeholder content
    ax.plot(np.random.randn(100).cumsum())
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

plt.tight_layout()
plt.savefig('figure_5.pdf', dpi=300, bbox_inches='tight')
```

### 3.4 Schematics

**Best Practices:**
- Use vector graphics software (Inkscape, Illustrator)
- Keep consistent line weights
- Align elements on grid
- Use muted colors for backgrounds
- Emphasize important elements
- Export as PDF or SVG

---

## 4. Figure Captions

### 4.1 Structure

**Components:**
1. **Title:** Brief description (bold, ends with period)
2. **Description:** What the figure shows
3. **Methods note:** Key experimental details
4. **Legend:** What symbols/colors represent
5. **Statistics:** Error bar definition, sample sizes
6. **Panel descriptions:** For multi-panel figures

### 4.2 Templates

**Single Panel:**
```
**Figure X. [Descriptive title].** [Description of what is plotted and the main observation]. [Brief method note if needed]. Error bars represent [SEM/SD/95% CI] (n = [number]). [Any additional notes].
```

**Multi-Panel:**
```
**Figure X. [Overall figure title].** **(a)** [Description of panel a]. **(b)** [Description of panel b]. **(c)** [Description of panel c]. [Shared methods notes]. Data points show mean ± SEM (n = [number]). Solid lines indicate [model fits/theory]. Shaded regions represent 95% confidence intervals. Statistical significance: *p < 0.05, **p < 0.01, ***p < 0.001.
```

### 4.3 Examples

**Example 1 (Data Plot):**
```
**Figure 1. Quantum gate fidelity improves with optimized pulse shaping.**
Gate fidelity measured by randomized benchmarking (blue circles) compared
with numerical simulation (solid orange line). The optimized DRAG pulse
(filled circles) achieves F = 0.9965 ± 0.0008, a 3.5× improvement over
the unoptimized Gaussian pulse (open circles). Error bars represent
standard error of the mean from 100 randomization sequences. Dashed
line indicates the coherence-limited fidelity of 0.9990.
```

**Example 2 (Multi-Panel):**
```
**Figure 3. Scaling of logical error rate with code distance.**
**(a)** Logical error probability versus physical error rate for surface
codes with distances d = 3, 5, 7, 9, 11 (light to dark blue).
**(b)** Threshold extraction showing pseudo-threshold at p_th = 0.0103 ± 0.0005.
**(c)** Error rate scaling exponent γ versus physical error rate, confirming
exponential suppression below threshold. **(d)** Comparison with prior
experimental results (gray symbols, refs. [1-3]) and theory (dashed lines).
All simulations use 10^6 syndrome measurement rounds per data point.
Error bars in (b-c) represent 95% bootstrap confidence intervals.
```

---

## 5. Quality Checklist

### 5.1 Technical
- [ ] Resolution meets journal requirements
- [ ] Figure dimensions appropriate
- [ ] File format correct (PDF/EPS for vector)
- [ ] No compression artifacts
- [ ] Colors display correctly

### 5.2 Content
- [ ] All axes labeled with units
- [ ] Legend present and clear
- [ ] Error bars/uncertainty shown
- [ ] Data points distinguishable
- [ ] Consistent with text descriptions

### 5.3 Style
- [ ] Font sizes readable (≥6 pt after scaling)
- [ ] Line weights visible (≥0.5 pt)
- [ ] Consistent styling across all figures
- [ ] Panel labels in consistent position
- [ ] Minimal chart junk

### 5.4 Accessibility
- [ ] Colorblind-safe palette used
- [ ] Shape AND color distinguish data
- [ ] Sufficient contrast
- [ ] Tested with colorblindness simulator

### 5.5 Caption
- [ ] Title clearly describes figure
- [ ] All panels described
- [ ] Error bar definition stated
- [ ] Sample sizes included
- [ ] Methods notes included

---

## 6. Common Mistakes to Avoid

### 6.1 Technical Errors
- Low resolution raster images
- Incorrect color mode (RGB vs CMYK)
- Font embedding issues
- Mismatched dimensions

### 6.2 Design Errors
- Unlabeled or mislabeled axes
- Missing or unclear legends
- 3D effects that distort data
- Dual y-axes that mislead
- Truncated axes that exaggerate effects
- Rainbow colormaps

### 6.3 Accessibility Errors
- Red-green color schemes only
- Low contrast text
- No shape differentiation
- Tiny fonts

### 6.4 Caption Errors
- Vague descriptions
- Missing error bar definitions
- Undefined abbreviations
- No sample sizes

---

## 7. Software Settings

### 7.1 matplotlib Configuration

```python
# Save this as publication_style.py

import matplotlib as mpl
import matplotlib.pyplot as plt

def set_publication_defaults():
    """Set matplotlib defaults for publication figures."""

    # General
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 300

    # Fonts
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 7

    # Figure size (single column)
    mpl.rcParams['figure.figsize'] = (3.5, 2.5)

    # Lines
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['lines.markersize'] = 4

    # Axes
    mpl.rcParams['axes.linewidth'] = 0.6
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

    # Ticks
    mpl.rcParams['xtick.major.width'] = 0.6
    mpl.rcParams['ytick.major.width'] = 0.6
    mpl.rcParams['xtick.major.size'] = 3
    mpl.rcParams['ytick.major.size'] = 3
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'

    # Legend
    mpl.rcParams['legend.frameon'] = False

    # Save
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05
    mpl.rcParams['pdf.fonttype'] = 42  # Embed fonts
```

### 7.2 Color Palette Reference

```python
# Colorblind-safe palette
COLORS = {
    'blue':   '#0173B2',
    'orange': '#DE8F05',
    'green':  '#029E73',
    'red':    '#D55E00',
    'purple': '#CC78BC',
    'brown':  '#CA9161',
    'pink':   '#FBAFE4',
    'gray':   '#949494',
    'yellow': '#ECE133',
    'cyan':   '#56B4E9',
}

# Sequential colormaps (colorblind-safe)
# Use: 'viridis', 'plasma', 'cividis', 'magma'

# Diverging colormaps
# Use: 'coolwarm', 'RdBu_r' with center=0
```

---

## Navigation

| Previous | Current | Next |
|----------|---------|------|
| Week 228 Guide | Figure Specifications | [Week 228 README](../README.md) |
