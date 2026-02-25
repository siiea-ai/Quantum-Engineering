# Figure Quality Standards

## Publication-Ready Figure Specifications for Quantum Physics Journals

---

## Journal-Specific Requirements

### Physical Review Journals (APS)

**File Formats (in order of preference)**:
1. PDF (vector)
2. EPS (vector)
3. PNG (raster, minimum 300 DPI)
4. TIFF (raster, minimum 300 DPI)

**Dimensions**:
| Type | Width | Max Height |
|------|-------|------------|
| Single column | 3.4 in (8.6 cm) | 9.5 in (24 cm) |
| Double column | 7.0 in (17.8 cm) | 9.5 in (24 cm) |

**Typography**:
- Minimum font size: 6 pt
- Recommended font: Helvetica, Arial, or Times
- LaTeX: Use consistent math font

**Colors**:
- RGB for online
- CMYK for print (automatic conversion)
- Avoid yellow on white

**Line Weights**:
- Minimum line weight: 0.5 pt
- Recommended: 0.75-1.5 pt

### Nature Family Journals

**File Formats**:
1. PDF or EPS (vector)
2. TIFF or PNG (minimum 300 DPI)

**Dimensions**:
| Type | Width |
|------|-------|
| Single column | 89 mm |
| 1.5 column | 120-136 mm |
| Double column | 183 mm |

**Typography**:
- Font: Helvetica or Arial
- Minimum size: 5 pt
- Maximum size: 7 pt for axis labels

**Specific Requirements**:
- No title in figure (title goes in caption)
- Panel labels: lowercase bold letters (a, b, c)
- Scale bars instead of axis for images

### npj Quantum Information

**Follows Nature Portfolio Standards**:
- Same dimensions and format requirements as Nature
- Open access: figures will be freely distributed
- Ensure you own or have permission for all elements

---

## Technical Specifications Checklist

### Resolution and Size

| Element | Minimum | Recommended |
|---------|---------|-------------|
| Line art | 300 DPI | 600 DPI or vector |
| Photographs | 300 DPI | 300 DPI |
| Combination | 600 DPI | 600 DPI |
| File size | - | Under 10 MB per figure |

### Vector vs. Raster Decision

**Use Vector (PDF, EPS, SVG) for**:
- Graphs and plots
- Diagrams and schematics
- Quantum circuits
- Any figure with text or lines

**Use Raster (PNG, TIFF) for**:
- Photographs
- Microscopy images
- Screenshots
- Heatmaps with many data points

### Color Mode

| Destination | Color Mode |
|-------------|------------|
| Screen/online | RGB |
| Print | CMYK or RGB (journals convert) |
| Grayscale OK | Check by converting |

---

## Design Standards

### Color Accessibility

**Colorblind-Accessible Palette (Wong 2011)**:

| Color | Hex Code | RGB | Use |
|-------|----------|-----|-----|
| Black | #000000 | (0, 0, 0) | Default text, lines |
| Orange | #E69F00 | (230, 159, 0) | Primary data |
| Sky Blue | #56B4E9 | (86, 180, 233) | Secondary data |
| Bluish Green | #009E73 | (0, 158, 115) | Tertiary data |
| Yellow | #F0E442 | (240, 228, 66) | Highlights (use carefully) |
| Blue | #0072B2 | (0, 114, 178) | Alternative primary |
| Vermillion | #D55E00 | (213, 94, 0) | Warnings, important |
| Reddish Purple | #CC79A7 | (204, 121, 167) | Additional category |

**Colorblind Testing**:
- Use Coblis simulator: https://www.color-blindness.com/coblis-color-blindness-simulator/
- Test for deuteranopia, protanopia, tritanopia

**Fallback Strategies**:
- Use different line styles (solid, dashed, dotted)
- Use different markers (circle, square, triangle)
- Add direct labels instead of legends

### Visual Hierarchy

**Emphasis Levels**:
1. **Primary emphasis**: Thickest lines, strongest colors, largest markers
2. **Secondary emphasis**: Medium line weight, moderate colors
3. **Background/reference**: Thin lines, light colors, gray

**Example: Highlighting Key Data**:
```python
# Primary result - thick, colored
ax.plot(x, y_main, 'o-', color='#E69F00', linewidth=2, markersize=6,
        label='This work')

# Comparison - thinner, less saturated
ax.plot(x, y_compare, 's--', color='#56B4E9', linewidth=1, markersize=4,
        label='Prior work')

# Theory/reference - lightest
ax.plot(x, y_theory, '-', color='gray', linewidth=0.8,
        label='Theory')
```

### Typography in Figures

**Font Hierarchy**:
| Element | Size | Weight |
|---------|------|--------|
| Panel labels | 10-12 pt | Bold |
| Axis labels | 8-10 pt | Normal |
| Tick labels | 8-9 pt | Normal |
| Legend | 8-9 pt | Normal |
| Annotations | 7-9 pt | Normal or Italic |

**Text Recommendations**:
- Use sans-serif (Helvetica, Arial) for labels
- Match main document math font in equations
- Avoid ALL CAPS except for standard abbreviations
- Right-align units: "Time (μs)" not "Time μs"

### Layout and Spacing

**Panel Arrangement**:
```
+-------+-------+    Reading order: a → b
|  (a)  |  (b)  |                   ↓
+-------+-------+                  c → d
|  (c)  |  (d)  |
+-------+-------+
```

**Spacing Guidelines**:
- Panel labels: 0.02-0.05 normalized figure coordinates from corner
- Axis labels: Sufficient space for tick labels
- Between panels: 0.1-0.2 inches minimum
- Margins: Minimal (let tight_layout handle)

---

## Quality Assessment Rubric

### Scoring Each Figure (1-5 scale)

| Criterion | 1 (Poor) | 3 (Adequate) | 5 (Excellent) |
|-----------|----------|--------------|---------------|
| **Message clarity** | Confusing, unclear purpose | Message present but could be clearer | Instantly communicates key point |
| **Data presentation** | Missing error bars, poor markers | Standard presentation | Optimal data-ink ratio, clear patterns |
| **Typography** | Unreadable, inconsistent | Readable, mostly consistent | Perfect readability, elegant |
| **Colors** | Poor choices, inaccessible | Acceptable palette | Accessible, purposeful color use |
| **Overall design** | Cluttered, distracting | Functional | Clean, professional, memorable |

**Minimum for publication**: Average score of 4 or higher

### Self-Check Questions

For each figure, answer:

1. **The Glance Test**: Can a reader understand the main point in 5 seconds?
   [ ] Yes [ ] No → Simplify or add annotation

2. **The Phone Test**: Is the figure readable on a phone screen?
   [ ] Yes [ ] No → Increase text/marker sizes

3. **The Grayscale Test**: Does figure work in black and white?
   [ ] Yes [ ] No → Add patterns or labels

4. **The Caption Test**: Can you understand figure with only the caption?
   [ ] Yes [ ] No → Improve caption completeness

5. **The Competition Test**: Would this impress reviewers/competitors?
   [ ] Yes [ ] No → Polish further

---

## Common Figure Types in Quantum Papers

### Quantum Circuit Diagrams

**Standards**:
- Horizontal wire for each qubit
- Time flows left to right
- Standard gate symbols (H, X, Z, CNOT, etc.)
- Measurements as meter symbols
- Multi-qubit gates aligned vertically

**Tools**:
- Quantikz (LaTeX/TikZ)
- Qiskit (Python)
- Quirk (online)
- Custom TikZ

**Example (Quantikz)**:
```latex
\begin{quantikz}
\lstick{$|0\rangle$} & \gate{H} & \ctrl{1} & \meter{} \\
\lstick{$|0\rangle$} & \qw & \targ{} & \qw
\end{quantikz}
```

### Bloch Sphere Representations

**Requirements**:
- Clear sphere outline
- Axis labels (x, y, z) or (|0⟩, |1⟩, |+⟩, etc.)
- State vector clearly visible
- Trajectory shown if relevant
- Color gradient for probability/phase

**Tools**:
- QuTiP (Python)
- Custom Matplotlib 3D
- Mathematica

### Data Plots with Error Bars

**Error Bar Standards**:
- Clearly visible but not overwhelming
- Cap ends optional (depends on style)
- Specify in caption: "Error bars represent [one standard deviation / standard error / 95% CI]"
- Consider violin plots or shaded regions for distributions

**Example Code**:
```python
ax.errorbar(x, y, yerr=y_err,
            fmt='o',           # Marker style
            capsize=3,         # Cap size
            capthick=1,        # Cap thickness
            elinewidth=1,      # Error line thickness
            color='#E69F00',
            markersize=6,
            label='Data')
```

### Schematics and Diagrams

**Best Practices**:
- Use vector graphics program (Illustrator, Inkscape)
- Consistent line weights
- Align elements to grid
- Use color purposefully
- Include scale bars for physical diagrams

---

## Figure Production Workflow

### Recommended Process

```
1. SKETCH
   - Rough layout on paper
   - Identify key elements
   - Determine panel arrangement

2. DRAFT
   - Create with code/software
   - Focus on content, not polish
   - Get feedback on concept

3. REFINE
   - Apply consistent styling
   - Optimize colors and typography
   - Add annotations

4. POLISH
   - Final resolution/format
   - Match journal requirements
   - Caption finalization

5. REVIEW
   - Self-assessment rubric
   - Colleague feedback
   - Colorblind simulation check
```

### File Organization

```
figures/
├── source/
│   ├── fig1_schematic.ai
│   ├── fig2_data.py
│   └── fig3_comparison.py
├── draft/
│   ├── fig1_v1.pdf
│   └── fig1_v2.pdf
├── final/
│   ├── fig1_schematic.pdf
│   ├── fig2_main_result.pdf
│   └── fig3_comparison.pdf
└── README.md  # Notes on how to regenerate
```

---

## Python Template for Publication Figures

```python
"""
Publication-quality figure template for quantum physics papers
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ============================================================
# Publication Settings
# ============================================================

# Journal-specific dimensions (inches)
SINGLE_COL = 3.4
DOUBLE_COL = 7.0
MAX_HEIGHT = 9.5

# Color palette (Wong colorblind-accessible)
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000'
}

# Style settings
plt.rcParams.update({
    # Figure
    'figure.figsize': (SINGLE_COL, 2.5),
    'figure.dpi': 300,
    'figure.facecolor': 'white',

    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 8,
    'mathtext.fontset': 'dejavusans',

    # Axes
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'axes.labelpad': 4,

    # Ticks
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',

    # Legend
    'legend.fontsize': 8,
    'legend.frameon': False,
    'legend.handlelength': 1.5,

    # Lines
    'lines.linewidth': 1.2,
    'lines.markersize': 5,

    # Saving
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,

    # Use LaTeX for text
    'text.usetex': False,  # Set True if LaTeX installed
})

# ============================================================
# Example Figure Creation
# ============================================================

def create_example_figure():
    """Create a two-panel example figure."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    # --- Panel (a): Line plot with error bars ---
    x = np.linspace(0, 10, 20)
    y = np.exp(-x/5) + 0.1 * np.random.randn(len(x))
    y_err = 0.05 * np.ones_like(y)

    ax1.errorbar(x, y, yerr=y_err,
                 fmt='o', color=COLORS['orange'],
                 capsize=2, capthick=0.8, elinewidth=0.8,
                 markersize=4, label='Measured')
    ax1.plot(x, np.exp(-x/5), '-', color=COLORS['blue'],
             linewidth=1, label='Theory')

    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Fidelity')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 1.2)
    ax1.legend(loc='upper right')

    # Panel label
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
             fontsize=10, fontweight='bold', va='top')

    # --- Panel (b): Scatter plot ---
    n_points = 50
    x2 = np.random.randn(n_points)
    y2 = 0.5 * x2 + 0.3 * np.random.randn(n_points)

    ax2.scatter(x2, y2, c=COLORS['green'], s=20, alpha=0.7)
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax2.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    ax2.set_xlabel('X Observable')
    ax2.set_ylabel('Z Observable')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-2, 2)

    # Panel label
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
             fontsize=10, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig('example_figure.pdf')
    plt.close()

if __name__ == '__main__':
    create_example_figure()
```

---

## Final Quality Control

### Pre-Submission Figure Checklist

- [ ] All figures at correct resolution (300+ DPI)
- [ ] All figures in acceptable format (PDF/EPS/PNG/TIFF)
- [ ] All figures at correct dimensions
- [ ] All text readable at publication size
- [ ] All colors accessible to colorblind readers
- [ ] All figures referenced in manuscript
- [ ] All captions complete and accurate
- [ ] Consistent style across all figures
- [ ] No copyright issues (permissions obtained)
- [ ] Supplementary figures also meet standards

---

*Standards based on APS, Nature, and IEEE guidelines. Always verify current journal requirements before submission.*
