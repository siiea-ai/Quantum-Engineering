# Figure Guidelines for Physics Publications

## Introduction

Figures are often the first (and sometimes only) part of a paper that readers examine closely. High-quality figures communicate your results effectively, enhance credibility, and increase the impact of your work. This guide provides comprehensive standards for creating publication-quality figures in physics research.

## Part I: Figure Planning

### The Role of Figures

Figures serve multiple purposes:
1. **Communicate data** more effectively than text
2. **Demonstrate results** visually
3. **Explain concepts** that are hard to describe verbally
4. **Attract readers** scanning the journal

### Figure Types

| Type | Purpose | Examples |
|------|---------|----------|
| Schematic | Explain setup/concept | Device diagram, pulse sequence |
| Data Plot | Present measurements | Scatter plot, line graph |
| Comparison | Show agreement | Theory vs. experiment |
| Phase Diagram | Map parameter space | Phase boundaries, regimes |
| Workflow | Explain process | Algorithm flowchart |
| Composite | Combine multiple aspects | Multi-panel summary |

### How Many Figures?

**Journal Guidelines:**
| Journal Type | Typical Figure Count |
|--------------|---------------------|
| Letters (PRL) | 3-4 |
| Regular articles (PRA) | 6-10 |
| PRX/PRX Quantum | 6-12 |
| Review articles | 10-20 |

**Planning Questions:**
- What are my key results? (1 figure each)
- What concepts need visual explanation? (schematic)
- What comparisons support my claims? (comparison figure)
- What supplementary visualizations support main figures?

## Part II: Design Principles

### Clarity First

**The 10-Second Rule:** A reader should understand your figure's main point within 10 seconds of looking at it.

**Clarity Checklist:**
- [ ] Main point is immediately apparent
- [ ] Labels are readable at print size
- [ ] Color scheme is intuitive
- [ ] Legend explains all elements
- [ ] Axes are properly labeled with units

### Visual Hierarchy

Guide the reader's eye to the most important information:

1. **Size:** Larger elements draw attention first
2. **Color:** Bright colors stand out from neutral backgrounds
3. **Position:** Upper-left is typically scanned first
4. **Contrast:** High contrast elements dominate

### Consistency

Maintain consistency across all figures:
- Same fonts throughout
- Consistent color scheme
- Uniform line weights
- Matching axis styles
- Similar panel layouts

## Part III: Technical Specifications

### Resolution and Format

**Print Resolution:**
- Minimum 300 DPI for halftone (photo) images
- Minimum 600 DPI for line art
- Vector formats (PDF, EPS, SVG) preferred

**File Formats:**
| Format | Use Case | Notes |
|--------|----------|-------|
| PDF | Vector graphics | Best for line plots |
| EPS | Vector graphics | LaTeX compatible |
| PNG | Raster with transparency | Good for schematics |
| TIFF | High-quality raster | Large file sizes |
| JPEG | Avoid | Lossy compression artifacts |

**Figure Dimensions:**
| Layout | Single Column | Double Column |
|--------|---------------|---------------|
| PRL/PRA | 3.4 inches | 7.0 inches |
| Nature | 89 mm | 183 mm |
| Science | 3.5 inches | 7.2 inches |

### Typography

**Font Sizes:**
- Axis labels: 8-10 pt at final size
- Tick labels: 7-9 pt at final size
- Legend: 7-9 pt at final size
- Panel labels: 10-12 pt bold

**Font Selection:**
- Sans-serif fonts (Helvetica, Arial) for clarity
- Match document font if possible
- Avoid decorative fonts
- Consistent font family throughout

### Color Guidelines

**Color Palettes:**

*Sequential (single variable):*
- Blue gradient: light blue → dark blue
- Viridis: yellow → green → blue (colorblind-safe)

*Diverging (centered data):*
- Blue-white-red
- Purple-white-orange

*Categorical (discrete groups):*
- Use 3-5 distinct colors maximum
- Colorblind-friendly options: blue, orange, green, red, purple

**Colorblind Considerations:**
- ~8% of males have color vision deficiency
- Avoid red-green distinctions alone
- Use shape/pattern in addition to color
- Test with colorblind simulators

**Recommended Colorblind-Safe Palette:**
```
Blue:    #0077BB
Orange:  #EE7733
Green:   #009988
Magenta: #CC3311
Cyan:    #33BBEE
Gray:    #BBBBBB
```

### Axis and Grid Styling

**Axes:**
- Clear, labeled axes with units
- Appropriate tick intervals
- No excessive tick marks
- Consider log scale for large ranges

**Grids:**
- Light gray for visibility without distraction
- Major grid only (avoid minor grid clutter)
- Dashed lines if needed

**Example Axis Label:**
```
"Gate fidelity, F (%)"
"Frequency, ω/2π (GHz)"
"Time, t (μs)"
```

## Part IV: Specific Figure Types

### Data Plots

**Line Plots:**
- Line weight: 1-2 pt for data, 0.5-1 pt for theory
- Markers: 4-6 pt diameter
- Error bars: visible but not overwhelming
- Connect points only if continuous variable

**Scatter Plots:**
- Marker size sufficient to see individual points
- Different markers for different data series
- Avoid overlapping when possible

**Error Bars:**
- Show error bars unless prohibitively small
- Specify meaning (1σ, 2σ, confidence interval) in caption
- Use shaded regions for densely packed data

**Example matplotlib settings:**
```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.figsize': (3.4, 2.5),  # single column
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
})
```

### Schematics

**Device Diagrams:**
- Accurate representation of key features
- Appropriate level of abstraction
- Labels for all components
- Scale bar if relevant

**Pulse Sequences:**
- Time axis clearly labeled
- Pulse amplitudes/phases shown
- Color coding for different controls
- Annotations for key timing

**Circuit Diagrams:**
- Standard symbol conventions
- Clear signal flow direction
- Labeled nodes and components

### Colormaps/Heatmaps

**Colorbar:**
- Always include colorbar
- Label with variable and units
- Appropriate range (not too saturated)

**Colormap Selection:**
- 'viridis': Good default, colorblind-safe
- 'RdBu': Diverging, centered on zero
- 'plasma': Perceptually uniform
- Avoid 'jet': Poor perceptual properties

### Multi-Panel Figures

**Panel Layout:**
- Label panels (a), (b), (c), ... consistently
- Align related panels
- Share axes where appropriate
- Consistent sizing

**Panel Labels:**
- Bold, 10-12 pt
- Upper-left corner of each panel
- Same position across all figures

## Part V: Caption Writing

### Caption Structure

```
Figure N. [Title: One-sentence summary]

(a) [Panel a description]
(b) [Panel b description]
...

[Symbol/color definitions]
[Parameter values]
[Source information if applicable]
```

### Caption Principles

**Self-Contained:** Reader should understand the figure from caption alone.

**Specific:** Mention specific values, conditions, and observations.

**Comprehensive:** Define all symbols, colors, and abbreviations.

### Caption Examples

**Good Caption:**
```
Figure 3. Two-qubit gate characterization.

(a) Gate fidelity versus pulse duration for the CZ gate. Red circles:
experimental randomized benchmarking data (N = 100 sequences per
point, error bars show 1σ statistical uncertainty). Blue line:
numerical simulation using measured system parameters.

(b) Gate leakage probability to |02⟩ state versus duration, extracted
from the same calibration data.

Optimal gate duration is 35 ns (dashed vertical line), achieving
fidelity F = 99.7 ± 0.1% with leakage L = 0.15 ± 0.05%. Qubit
parameters: ωq/2π = 5.12 GHz, α/2π = -340 MHz.
```

**Insufficient Caption:**
```
Figure 3. Gate fidelity (a) and leakage (b) versus duration.
The optimal point is marked.
```

## Part VI: Production Workflow

### Software Recommendations

**Data Plotting:**
- matplotlib (Python): Most flexible, publication-quality
- Origin: GUI-based, good for non-programmers
- MATLAB: Good integration with analysis
- gnuplot: Lightweight, scriptable

**Schematics:**
- Adobe Illustrator: Industry standard
- Inkscape: Free, open-source
- TikZ (LaTeX): Programmatic, vector
- Draw.io: Web-based, simple diagrams

**Image Editing:**
- ImageJ/Fiji: Scientific image processing
- GIMP: Free Photoshop alternative
- Photoshop: Professional editing

### Workflow Steps

1. **Create raw plots** from data analysis
2. **Export as vector** (PDF/SVG) if possible
3. **Refine in editing software** if needed
4. **Standardize styling** across figures
5. **Export at final resolution**
6. **Test at print size**

### Quality Control Checklist

Before submission, verify each figure:

- [ ] Readable at final print size
- [ ] Correct file format and resolution
- [ ] All text legible (8+ pt)
- [ ] Axes labeled with units
- [ ] Error bars included or explained
- [ ] Colors distinguishable (including grayscale)
- [ ] Panel labels present and consistent
- [ ] Caption is comprehensive

### Common Fixes

| Problem | Solution |
|---------|----------|
| Blurry at print size | Increase DPI or use vector format |
| Text too small | Increase font size, simplify if needed |
| Colors indistinguishable | Change to colorblind-safe palette |
| Cluttered | Remove non-essential elements |
| Inconsistent style | Apply uniform settings across figures |

## Part VII: Journal-Specific Requirements

### Physical Review (APS) Guidelines

- Width: 3.4" (single column), 7.0" (double column)
- Formats: EPS, PDF, PS (preferred); PNG, TIFF (acceptable)
- Resolution: 300+ DPI for halftones
- Color: Free in online version
- Caption: Separate from figure file

### Nature/Science Guidelines

- Width: 89 mm (single), 183 mm (double)
- Format: PDF, EPS, TIFF preferred
- Resolution: 300+ DPI
- Color: RGB preferred for online
- Extended Data figures for supplementary

### arXiv Guidelines

- Embed figures in PDF
- No file size limits (reasonable)
- PDF/X or standard PDF acceptable
- Ensure fonts embedded

## Summary

Creating publication-quality figures requires attention to clarity, consistency, and technical standards. Invest time in figure design—it pays dividends in paper impact.

**Key Principles:**
1. Clarity: One main point per figure
2. Consistency: Uniform style throughout
3. Quality: Professional technical standards
4. Accessibility: Colorblind-safe, readable at size
5. Self-contained: Comprehensive captions

---

*Use this guide alongside `Templates/Figure_Caption.md` to create effective figures and captions.*
