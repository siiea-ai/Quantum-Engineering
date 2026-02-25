# Figure Checklist Template

## Publication-Quality Figure Standards for Quantum Physics Papers

---

## Figure Inventory

| Fig. # | Title | Type | Status | Priority |
|--------|-------|------|--------|----------|
| 1 | | [ ] Schematic [ ] Data [ ] Theory | [ ] Draft [ ] Final | [ ] Essential [ ] Supporting |
| 2 | | [ ] Schematic [ ] Data [ ] Theory | [ ] Draft [ ] Final | [ ] Essential [ ] Supporting |
| 3 | | [ ] Schematic [ ] Data [ ] Theory | [ ] Draft [ ] Final | [ ] Essential [ ] Supporting |
| 4 | | [ ] Schematic [ ] Data [ ] Theory | [ ] Draft [ ] Final | [ ] Essential [ ] Supporting |
| S1 | | [ ] Schematic [ ] Data [ ] Theory | [ ] Draft [ ] Final | [ ] Essential [ ] Supporting |
| S2 | | [ ] Schematic [ ] Data [ ] Theory | [ ] Draft [ ] Final | [ ] Essential [ ] Supporting |

---

## Per-Figure Checklist

### Figure ___: ________________________________

#### Content Quality
- [ ] Conveys key message clearly
- [ ] All necessary information included
- [ ] No extraneous elements
- [ ] Tells story without caption

#### Technical Specifications
- [ ] Resolution: 300 DPI minimum (print), 150 DPI (web)
- [ ] Format: [ ] PDF [ ] EPS [ ] PNG [ ] TIFF
- [ ] Color mode: [ ] RGB (web) [ ] CMYK (print)
- [ ] File size within limits

#### Dimensions
- [ ] Single column (3.4 in / 86 mm)
- [ ] 1.5 column (5.0 in / 127 mm)
- [ ] Double column (7.0 in / 178 mm)
- [ ] Aspect ratio appropriate

#### Typography
- [ ] Font: Serif (Times, Computer Modern) or Sans-serif (Helvetica, Arial)
- [ ] Font size: 8-10 pt (readable when scaled)
- [ ] Consistent fonts across all panels
- [ ] Axis labels: Clear and complete
- [ ] Units: Included on all axes
- [ ] Panel labels: (a), (b), (c) consistent size and position

#### Data Presentation
- [ ] Error bars: Included where appropriate
- [ ] Error bar type defined (std dev, std error, CI)
- [ ] Symbols distinguishable
- [ ] Line weights appropriate (0.5-1.5 pt)
- [ ] Data points visible (not obscured)

#### Colors
- [ ] Colorblind-accessible palette used
- [ ] Works in grayscale (if possible)
- [ ] Colors consistent across figures
- [ ] Legend included if needed
- [ ] Color meaning explained in caption

#### Axes and Scales
- [ ] Axis ranges appropriate
- [ ] Tick marks: Major and minor
- [ ] Log scales: Clearly indicated
- [ ] Broken axes: Clearly marked
- [ ] Zero included if meaningful

#### Legend and Labels
- [ ] Legend: Placed optimally (no data obscured)
- [ ] Legend: Entries in logical order
- [ ] Annotations: Minimal and necessary
- [ ] Arrows/callouts: Used sparingly

#### Caption Elements Covered
- [ ] Opening statement (figure title)
- [ ] Description of each panel
- [ ] Symbol definitions
- [ ] Method summary (if applicable)
- [ ] Key finding highlighted
- [ ] Parameter values
- [ ] Error bar/uncertainty definition

---

## Color Palette Recommendations

### Colorblind-Accessible Palettes

**Wong Palette** (recommended for up to 8 colors):
- #000000 (Black)
- #E69F00 (Orange)
- #56B4E9 (Sky Blue)
- #009E73 (Bluish Green)
- #F0E442 (Yellow)
- #0072B2 (Blue)
- #D55E00 (Vermillion)
- #CC79A7 (Reddish Purple)

**Qualitative Data** (distinct categories):
- Use Wong palette
- Ensure sufficient contrast
- Consider colorblind viewers

**Sequential Data** (one variable):
- Single hue gradient (light to dark)
- Examples: viridis, plasma, inferno (matplotlib)

**Diverging Data** (positive/negative from center):
- Two hues with neutral midpoint
- Examples: coolwarm, RdBu_r (matplotlib)

---

## Quantum-Specific Figure Guidelines

### Quantum Circuit Diagrams
- [ ] Wire lines clear and straight
- [ ] Gate symbols standard or defined
- [ ] Measurement symbols correct
- [ ] Time flows left to right
- [ ] Multi-qubit gates aligned

### Bloch Sphere Representations
- [ ] Sphere boundary visible
- [ ] Axes labeled (x, y, z)
- [ ] State vector clearly indicated
- [ ] Trajectory shown if relevant

### Energy Level Diagrams
- [ ] Energy increases upward
- [ ] Level spacing represents energy difference
- [ ] Transitions labeled
- [ ] Selection rules indicated

### Pulse Sequences
- [ ] Time axis labeled
- [ ] Pulse amplitudes to scale
- [ ] Pulse shapes accurate
- [ ] Channels clearly distinguished

### Noise/Error Visualizations
- [ ] Error bars on all data points
- [ ] Confidence intervals shaded
- [ ] Noise spectra on log-log
- [ ] Threshold lines labeled

---

## Multi-Panel Figure Template

```
+------------------+------------------+
|                  |                  |
|       (a)        |       (b)        |
|                  |                  |
+------------------+------------------+
|                  |                  |
|       (c)        |       (d)        |
|                  |                  |
+------------------+------------------+
```

### Panel Arrangement Checklist
- [ ] Logical reading order (left-to-right, top-to-bottom)
- [ ] Related panels grouped
- [ ] Consistent panel sizes
- [ ] Adequate spacing
- [ ] Shared legends/colorbars placed efficiently

---

## Python Figure Code Template

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Publication settings
plt.rcParams.update({
    # Figure
    'figure.figsize': (3.4, 2.5),
    'figure.dpi': 300,
    'figure.facecolor': 'white',

    # Font
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman'],
    'font.size': 9,
    'mathtext.fontset': 'cm',

    # Axes
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'axes.linewidth': 0.8,

    # Ticks
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',

    # Legend
    'legend.fontsize': 8,
    'legend.frameon': False,

    # Lines
    'lines.linewidth': 1.2,
    'lines.markersize': 4,

    # Saving
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Wong colorblind palette
wong_colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
               '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

# Panel (a)
ax = axes[0]
# ... plotting code ...
ax.set_xlabel(r'Time ($\mu$s)')
ax.set_ylabel(r'Fidelity')
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
        fontsize=10, fontweight='bold', va='top')

# Panel (b)
ax = axes[1]
# ... plotting code ...
ax.text(0.02, 0.98, '(b)', transform=ax.transAxes,
        fontsize=10, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('figure.pdf')
```

---

## Caption Template

### Standard Format

> **Figure X. [Descriptive title].** (a) [Description of panel a, including what is plotted and key features]. (b) [Description of panel b]. [Statement of key finding or observation]. [Definition of symbols, abbreviations, or colors if not in legend]. Parameters: [relevant parameter values]. Error bars represent [one standard deviation / standard error / 95% confidence interval].

### Example Caption

> **Figure 2. Logical error suppression in the surface code.** (a) Measured logical error rate $p_L$ as a function of physical error rate $p$ for code distances $d = 3$ (blue circles), $d = 5$ (orange squares), and $d = 7$ (green diamonds). Solid lines show fits to $p_L = A(p/p_{\text{th}})^{(d+1)/2}$ with threshold $p_{\text{th}} = 0.95 \pm 0.05\%$. (b) Extracted threshold as a function of measurement repetitions, showing convergence to the asymptotic value. The dashed line indicates the theoretical threshold of 1.1%. Error bars represent one standard deviation from 1000 bootstrap samples.

---

## File Naming Convention

```
figure_[number]_[description]_[version].[format]

Examples:
figure_01_schematic_v3.pdf
figure_02_main_result_v2.pdf
figure_03_comparison_final.pdf
figure_S1_extended_data_v1.pdf
```

---

## Pre-Submission Figure Review

### Self-Check Questions
1. Can the figure be understood without reading the caption?
2. Would the message survive grayscale printing?
3. Are all elements large enough to read in final size?
4. Is the figure referenced in the text?
5. Does the caption fully explain the figure?

### Common Issues
| Issue | Solution |
|-------|----------|
| Text too small | Increase font size, reduce figure complexity |
| Colors confusing | Use colorblind palette, add patterns |
| Too cluttered | Remove non-essential elements, use supplement |
| Inconsistent style | Create and follow style template |
| Low resolution | Regenerate at higher DPI, use vector format |

---

## Sign-Off

**Figure Review Completed**: [ ] Yes [ ] No

**Reviewer**: ________________________

**Date**: ________________________

**Notes**: ________________________________________________________________
