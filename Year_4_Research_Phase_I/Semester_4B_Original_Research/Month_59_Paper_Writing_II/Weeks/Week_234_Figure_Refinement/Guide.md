# Publication Figure Design Guide

## Introduction

This guide provides comprehensive methodology for creating publication-quality scientific figures. Effective figures are not merely decorative—they are essential communication tools that can determine whether readers engage with your work.

## Part I: Principles of Scientific Visualization

### The Data-Ink Ratio

Edward Tufte introduced the concept of **data-ink ratio**: the proportion of ink devoted to displaying data versus non-data elements.

$$\text{Data-Ink Ratio} = \frac{\text{Data Ink}}{\text{Total Ink}}$$

**Maximize this ratio by:**
- Removing unnecessary gridlines
- Eliminating decorative elements
- Simplifying legends
- Using whitespace effectively

**Before (low ratio):**
```
Heavy borders, 3D effects, textured backgrounds,
extensive gridlines, decorative elements
```

**After (high ratio):**
```
Clean axes, minimal gridlines, direct labels,
no decorative elements, focus on data
```

### The Gestalt Principles

Human visual perception follows predictable patterns. Use these for effective figure design:

| Principle | Description | Application |
|-----------|-------------|-------------|
| **Proximity** | Close items seem related | Group related data points |
| **Similarity** | Similar items seem related | Use consistent markers for same condition |
| **Continuity** | Eyes follow lines | Use smooth curves for continuous data |
| **Closure** | Mind completes shapes | Don't need complete borders |
| **Figure-Ground** | Distinguish foreground | Ensure data stands out from background |

### Information Hierarchy

Guide the viewer's eye through careful use of visual weight:

```
Primary (heaviest):   Data points, main curves
Secondary:            Axes, labels, error bars
Tertiary:             Legends, annotations
Minimal:              Gridlines, borders
```

**Visual weight factors:**
- Size (larger = heavier)
- Color saturation (more saturated = heavier)
- Line thickness (thicker = heavier)
- Position (center = heavier)

## Part II: Color in Scientific Figures

### Color Fundamentals

**Color spaces:**
- **RGB:** Screen display (additive)
- **CMYK:** Print (subtractive)
- **Perceptually uniform:** Lab, CIELUV (equal steps look equal)

**Key insight:** Not all color differences are perceived equally. Use perceptually uniform color maps for quantitative data.

### Sequential Color Maps

For ordered data (low → high), use:

**Recommended:**
- `viridis` (matplotlib default) - perceptually uniform
- `plasma` - good contrast
- `inferno` - dark backgrounds

**Avoid:**
- `jet` / `rainbow` - not perceptually uniform, misleading
- `hot` - poor in grayscale

### Diverging Color Maps

For data with meaningful center (e.g., correlation matrices):

**Recommended:**
- `RdBu` (Red-Blue) - clear positive/negative
- `coolwarm` - lower saturation, less harsh
- `BrBG` (Brown-Blue-Green) - colorblind-safe

### Qualitative Color Palettes

For categorical data (discrete groups):

**Wong Palette (colorblind-optimized):**

| Name | Hex Code | Use For |
|------|----------|---------|
| Blue | #0072B2 | Primary data |
| Orange | #E69F00 | Secondary data |
| Green | #009E73 | Tertiary data |
| Yellow | #F0E442 | Highlights |
| Sky Blue | #56B4E9 | Background data |
| Vermillion | #D55E00 | Important emphasis |
| Purple | #CC79A7 | Additional category |
| Black | #000000 | Reference, theory |

### Colorblind Considerations

**Types of color vision deficiency:**

| Type | Prevalence | Confusion Pairs |
|------|------------|-----------------|
| Deuteranomaly (green-weak) | 5% of males | Red/green |
| Protanomaly (red-weak) | 1% of males | Red/green |
| Tritanomaly (blue-weak) | 0.01% | Blue/yellow |
| Monochromacy | Rare | All colors |

**Safe strategies:**
1. Use both color AND shape
2. Use both color AND pattern
3. Label directly when possible
4. Test with simulators

## Part III: Figure Types and Design Patterns

### Line Plots

**Best for:** Time series, continuous functions, trends

**Design guidelines:**
- Line thickness: 1-2 pt for data, 0.5-1 pt for theory
- Markers: Use for discrete measurements
- Connect only continuous data
- Limit to 4-5 lines before confusion

**Example structure:**
```python
# Recommended line styles for up to 4 series
styles = [
    {'color': '#0072B2', 'linestyle': '-', 'marker': 'o'},
    {'color': '#E69F00', 'linestyle': '--', 'marker': 's'},
    {'color': '#009E73', 'linestyle': '-.', 'marker': '^'},
    {'color': '#D55E00', 'linestyle': ':', 'marker': 'd'},
]
```

### Scatter Plots

**Best for:** Correlation, distribution, individual measurements

**Design guidelines:**
- Marker size: Large enough to see, small enough for density
- Alpha/transparency for overlapping data
- Error bars when appropriate
- Consider 2D histograms for dense data

### Bar Charts

**Best for:** Categorical comparisons, discrete counts

**Design guidelines:**
- Bars should start at zero
- Use horizontal bars for many categories
- Grouped bars for comparison across conditions
- Consider alternatives (dot plots, lollipop charts)

### Heatmaps

**Best for:** Matrices, 2D parameter spaces, correlations

**Design guidelines:**
- Use perceptually uniform colormaps
- Include colorbar with units
- Consider log scale for wide ranges
- Annotate key values if readable

### Schematics

**Best for:** Experimental setups, conceptual illustrations

**Design guidelines:**
- Consistent style (line weights, colors)
- Clear labeling of components
- Scale indicators if spatial
- Flow direction indicators if process

## Part IV: Multi-Panel Figures

### When to Use Multi-Panel Figures

**Combine panels when:**
- Showing related aspects of same experiment
- Comparing conditions side-by-side
- Building a narrative sequence
- Space efficiency (fewer figure slots)

**Use separate figures when:**
- Panels tell independent stories
- Complexity requires full attention
- Different aspects need detailed captions

### Panel Layout Strategies

**Horizontal layout (a) (b) (c):**
- Natural left-to-right reading
- Good for sequence or comparison
- Best for wide data (time series)

**Vertical layout (a) over (b):**
- Good for showing before/after
- Shared x-axis possible
- Best for comparing same quantity

**Grid layout:**
- Matrix of conditions
- Parameter sweeps
- Comprehensive comparison

### Visual Consistency

**All panels should share:**
- Font family and size
- Color palette
- Line weights
- Axis style
- Scale bars (if applicable)

**Exception:** When contrast is the point (e.g., before/after style change)

### Panel Labels

**Positioning:**
- Top-left corner (most common)
- Outside plot area if space allows
- Consistent position across all panels

**Formatting:**
- Bold lowercase letters: **(a)**, **(b)**, **(c)**
- Some journals prefer uppercase or Roman numerals
- Check target journal style

## Part V: Typography in Figures

### Font Selection

**Primary choice:** Match document font when possible

**Sans-serif fonts for figures:**
- Helvetica (Mac default, professional)
- Arial (Windows, widely available)
- DejaVu Sans (open source, complete symbols)

**Avoid:**
- Serif fonts (less readable at small sizes)
- Decorative fonts
- Mixed font families

### Font Sizes

**At publication size:**

| Element | Minimum Size | Recommended |
|---------|-------------|-------------|
| Axis labels | 6 pt | 8-10 pt |
| Tick labels | 5 pt | 6-8 pt |
| Annotations | 5 pt | 6-8 pt |
| Panel labels | 8 pt | 10-12 pt (bold) |

**Rule of thumb:** If you can't read it in the print journal, it's too small.

### Mathematical Typography

**Use proper symbols:**
- Greek letters: α, β, γ, not alpha, beta, gamma
- Operators: ×, ±, ≈, not x, +/-, ~
- Units with proper spacing: 10 mK, not 10mK

**LaTeX rendering in matplotlib:**
```python
plt.rcParams['text.usetex'] = True  # Requires LaTeX installation
# or use mathtext:
plt.xlabel(r'$\Omega / 2\pi$ (MHz)')
```

## Part VI: Technical Requirements

### Resolution and DPI

**Understanding DPI:**
- DPI = Dots Per Inch
- Screen: 72-96 DPI typical
- Print: 300+ DPI required

**Guidelines by content:**

| Content Type | Minimum DPI | Recommended |
|--------------|-------------|-------------|
| Line art (schematics) | 600 | 1200 |
| Halftones (photos) | 300 | 300-600 |
| Combinations | 600 | 600 |

### Vector vs. Raster

**Vector graphics (PDF, EPS, SVG):**
- Infinitely scalable
- Small file size for simple graphics
- Editable
- Best for: schematics, plots, line art

**Raster graphics (PNG, TIFF, JPG):**
- Fixed resolution
- Better for complex images (photos)
- PNG: lossless, supports transparency
- TIFF: publication standard, no compression
- JPG: lossy, avoid for figures

**Best practice:** Create in vector, export to raster only if required.

### Color Modes

**RGB:** Use for screen display, web
**CMYK:** Use for print
**Grayscale:** Verify figures work in grayscale

**Converting for print:**
- Some colors shift RGB→CMYK
- Test print before final submission
- Avoid pure RGB colors (saturated, no CMYK equivalent)

## Part VII: Caption Writing

### Caption Structure

```
Figure N. [Title describing the figure].

[Panel descriptions: what each panel shows, key observations.]

[Methodology: measurement conditions, sample parameters.]

[Definitions: symbols, abbreviations, error bar meaning.]
```

### Caption Examples

**Example 1: Data Figure**

```
Figure 3. Gate fidelity as a function of drive amplitude.
(a) Single-qubit fidelity (blue circles) measured via randomized
benchmarking showing optimum at Ω/2π = 25 MHz. Solid line shows
numerical simulation with no free parameters. (b) Two-qubit controlled-Z
fidelity reaching 99.3 ± 0.1% at optimal amplitude (dashed line).
Error bars represent standard deviation from 10 independent measurements.
Measurements performed at T = 15 mK with τ = 40 ns gate duration.
```

**Example 2: Schematic**

```
Figure 1. Experimental setup for dispersive qubit readout.
(a) Simplified circuit diagram showing transmon qubit (orange)
capacitively coupled to readout resonator (blue). (b) Full measurement
chain including cryogenic amplification stages. HEMT: high electron
mobility transistor; JPA: Josephson parametric amplifier.
Not to scale; see Methods for component specifications.
```

### Caption Dos and Don'ts

**Do:**
- Begin with what the figure shows
- Describe each panel explicitly
- Define all symbols and abbreviations
- Include quantitative information
- Specify error bar meaning

**Don't:**
- Begin with "This figure shows..."
- Repeat the axis labels
- Include interpretation (save for text)
- Use vague language ("significant improvement")
- Leave symbols undefined

## Part VIII: Workflow and Tools

### Recommended Workflow

```
1. Data preparation
   ├── Clean and validate data
   ├── Calculate derived quantities
   └── Prepare error estimates

2. Draft visualization
   ├── Choose appropriate plot type
   ├── Create initial figure
   └── Assess message clarity

3. Style application
   ├── Apply journal template
   ├── Adjust colors, fonts
   └── Add labels, legends

4. Quality check
   ├── Review at publication size
   ├── Test colorblind accessibility
   └── Verify grayscale readability

5. Export
   ├── Vector format (PDF/EPS)
   ├── High-res raster backup
   └── Web-resolution preview

6. Integration
   ├── Place in manuscript
   ├── Write caption
   └── Verify references in text
```

### Tool Recommendations

**Python/matplotlib:** Primary recommendation for quantitative figures

**Inkscape:** Vector editing, schematic creation (free, cross-platform)

**Adobe Illustrator:** Industry standard vector editing (if available)

**Affinity Designer:** Illustrator alternative (lower cost)

**Blender/POV-Ray:** 3D renderings if needed

### Version Control for Figures

**Naming convention:**
```
fig1_gate_fidelity_v01.pdf
fig1_gate_fidelity_v02.pdf  # After colorblind fixes
fig1_gate_fidelity_v03_final.pdf
```

**Or use Git with descriptive commits:**
```bash
git add figures/fig1_gate_fidelity.pdf
git commit -m "Fig 1: Updated color palette for accessibility"
```

## Part IX: Common Mistakes and Fixes

### Mistake 1: The Rainbow Colormap

**Problem:** `jet` and `rainbow` are perceptually non-uniform. Equal data differences don't produce equal visual differences.

**Fix:** Use `viridis`, `plasma`, or `inferno` for sequential data.

### Mistake 2: Unreadable Text

**Problem:** Labels too small at publication size.

**Fix:** Create figures at target size, verify readability before export.

### Mistake 3: Missing Error Bars

**Problem:** Data shown without uncertainty quantification.

**Fix:** Always include error bars with defined meaning (SD, SEM, CI).

### Mistake 4: Inconsistent Styling

**Problem:** Each figure looks different—different fonts, colors, line weights.

**Fix:** Create a style template and apply to all figures.

### Mistake 5: 3D When 2D Suffices

**Problem:** 3D effects add no information, reduce clarity.

**Fix:** Only use 3D when data is genuinely three-dimensional.

### Mistake 6: Legend Overload

**Problem:** Legends with many entries, complex symbols.

**Fix:** Direct label where possible, simplify legend entries.

## Summary

### Key Principles

1. **Clarity first:** Every element should aid understanding
2. **Consistency:** All figures should look like a unified set
3. **Accessibility:** Design for all viewers including colorblind
4. **Appropriate resolution:** Vector when possible, high DPI when not
5. **Informative captions:** Figures should be understandable standalone

### Pre-Submission Checklist

- [ ] All figures serve a clear purpose
- [ ] Consistent style across all figures
- [ ] Colorblind-accessible palettes
- [ ] Readable at publication size
- [ ] High resolution (600+ DPI or vector)
- [ ] All labels include units
- [ ] Error bars present with defined meaning
- [ ] Captions complete and informative
- [ ] Matches journal style guide
- [ ] Works in grayscale

---

*Figure Design Guide | Week 234 | Month 59 | Year 4*
