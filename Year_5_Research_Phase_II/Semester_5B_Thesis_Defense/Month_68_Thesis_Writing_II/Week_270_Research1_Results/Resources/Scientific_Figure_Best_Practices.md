# Scientific Figure Best Practices

## Introduction

Figures are often the first (and sometimes only) part of your thesis that readers examine closely. High-quality figures communicate your research more effectively than any amount of text. This resource provides comprehensive guidance on creating publication-quality scientific figures for your thesis.

## Fundamental Principles

### 1. Data-to-Ink Ratio

Maximize the proportion of ink devoted to displaying data. Every element should serve a purpose.

**Before (Low data-to-ink):**
- Heavy grid lines
- Decorative borders
- 3D effects
- Gradient fills
- Background colors

**After (High data-to-ink):**
- Minimal or no grid
- Clean axes
- 2D representation
- Solid fills where needed
- White background

### 2. Clarity Over Aesthetics

Choose clarity over visual appeal when they conflict.

| Aesthetic Choice | Clarity Choice | Winner |
|-----------------|----------------|--------|
| Thin, elegant lines | Thick, visible lines | Clarity |
| Small, elegant fonts | Large, readable fonts | Clarity |
| Subtle color differences | Distinct colors | Clarity |
| Dense information | Spaced layout | Clarity |

### 3. Self-Containment

A figure with its caption should be understandable without reading the main text. Include:
- What is being shown
- How it was measured/calculated
- What the symbols represent
- What the uncertainty is
- Key conditions

### 4. Consistency

All figures in your thesis should look like they belong together:
- Same color scheme
- Same fonts
- Same axis label format
- Same symbol conventions
- Same caption style

## Data Visualization Types

### Line Plots

**Best for**: Continuous data, time series, theoretical curves

**Best practices:**
- Use distinct line styles and colors for different series
- Include data markers for measured points
- Reserve solid lines for data, dashed for theory
- Show error bars or bands for uncertainty

```
Example code (Python/matplotlib):
plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=3, label='Data')
plt.plot(x_theory, y_theory, '--', label='Theory')
plt.legend()
```

### Scatter Plots

**Best for**: Discrete measurements, correlations between variables

**Best practices:**
- Use different symbols for different categories
- Ensure symbols don't overlap excessively
- Add transparency (alpha) if many points
- Consider 2D histograms for very dense data

### Bar Charts

**Best for**: Categorical comparisons

**Best practices:**
- Always include error bars
- Start y-axis at zero for count/amount data
- Use horizontal bars for long category labels
- Avoid 3D effects
- Order bars logically (by value, time, or category logic)

**Avoid**: Pie charts (bar charts almost always work better)

### Heat Maps / 2D Color Plots

**Best for**: 2D parameter sweeps, correlation matrices, images

**Best practices:**
- Use perceptually uniform colormaps (viridis, plasma, inferno)
- Use diverging colormaps for data with meaningful midpoint
- Always include colorbar with label and units
- Consider contour overlays for key values

### Error Representation

**Types of error display:**

1. **Error bars**: Standard for discrete data points
   - Cap ends for visual clarity
   - State in caption what error bars represent (1σ, 2σ, SEM, 95% CI)

2. **Shaded bands**: For continuous theory uncertainty or dense data
   - Use transparency
   - State confidence level

3. **Box plots**: For distribution comparison
   - Show median, quartiles, and outliers
   - More informative than bar charts for distributions

4. **Violin plots**: Distribution shape comparison
   - Shows full distribution shape
   - Good for multimodal data

## Color Usage

### Colorblind-Safe Palettes

Approximately 8% of males and 0.5% of females have some form of color vision deficiency. Design figures accessible to all readers.

**Safe palettes:**
- ColorBrewer qualitative schemes (Set2, Dark2)
- Viridis family (viridis, plasma, magma, inferno)
- IBM Design palette
- Wong palette

**Avoid:**
- Red-green distinctions as only differentiator
- Rainbow colormaps (jet)
- Unlabeled color gradients

### Color Coding Principles

1. **Semantic meaning**: Use intuitive colors
   - Red for hot/high, blue for cold/low
   - Green for good, red for bad (but be careful with red-green)

2. **Consistent meaning**: Same color = same thing throughout thesis
   - If blue = experimental data in Figure 1, keep it throughout

3. **Sufficient contrast**: Colors should be distinguishable
   - Check by converting to grayscale
   - Use online colorblind simulators

### Grayscale Compatibility

Some readers may print in black and white. Test your figures:
1. Convert to grayscale
2. Check if all elements are distinguishable
3. Add patterns or line styles if colors alone aren't sufficient

## Typography in Figures

### Font Selection

**Recommended fonts:**
- Sans-serif for figures: Helvetica, Arial, Calibri
- Match thesis body if possible
- Avoid decorative fonts

### Font Sizes

At the final printed size:

| Element | Minimum | Recommended |
|---------|---------|-------------|
| Axis labels | 10 pt | 11-12 pt |
| Tick labels | 8 pt | 9-10 pt |
| Legend | 8 pt | 9-10 pt |
| Annotations | 8 pt | 9-10 pt |
| Panel labels | 12 pt | 12-14 pt bold |

### Text Formatting

- Axis labels: Title case or sentence case, consistent throughout
- Units: In parentheses after quantity, e.g., "Time (μs)"
- Scientific notation: Use × symbol, e.g., "2 × 10⁻³"
- Variables: Italic, matching equation formatting

## Layout and Composition

### Figure Sizing

**Standard sizes:**
- Single column: 3.25-3.5 inches wide (journal), 5-6 inches (thesis)
- Full width: 7 inches (journal), 6-7 inches (thesis)
- Aspect ratio: Typically 4:3 or 1:1; 16:9 for presentations

### White Space

- Leave breathing room around data
- Don't pack elements too tightly
- Use margins consistently

### Alignment

- Align related elements
- Use grid for multi-panel layouts
- Consistent spacing between panels

### Panel Organization

For multi-panel figures:
- Read left-to-right, top-to-bottom
- Label panels (a), (b), (c)... or (A), (B), (C)...
- Group related panels
- Shared axes where sensible

## Multi-Panel Figures

### When to Combine

Combine into one figure when:
- Panels show related aspects of same data
- Comparison between panels is important
- Panels share axes or legends
- Together they tell a complete story

Keep separate when:
- Each could stand alone
- Different methods/samples
- Reference from different text sections
- Page layout is cleaner

### Design Considerations

```
+------------------+------------------+
|       (a)        |       (b)        |
|                  |                  |
+------------------+------------------+
|       (c)        |       (d)        |
|                  |                  |
+------------------+------------------+

- Consistent panel sizes
- Aligned axes
- Shared colorbar on right if needed
- Panel labels in consistent position
```

### Shared Elements

- Share legend if same series across panels
- Share colorbar if same scale
- Use shared axis labels if same quantity
- Consider panel labels as in-figure text

## Schematics and Diagrams

### Purpose

Schematics explain experimental setups, theoretical concepts, or workflows that are hard to describe in text.

### Best Practices

1. **Simplify**: Show only what's necessary
2. **Label clearly**: All components identified
3. **Use consistent symbols**: Standard symbols for standard components
4. **Scale appropriately**: Not to scale? Say so
5. **Use color meaningfully**: Distinguish functional groups

### Elements to Include

- Component labels (with or without leader lines)
- Flow direction (arrows)
- Scale bar (if applicable)
- Legend for symbols/colors
- Key dimensions (if relevant)

## Image Figures

### Microscopy and Photography

**Required elements:**
- Scale bar (not just magnification)
- Annotation of key features
- Consistent exposure/contrast within figure
- Description of imaging conditions

**Enhancement guidelines:**
- Apply same processing to all images being compared
- Document any processing in methods
- Don't obscure data with heavy processing
- False color should be labeled as such

### Data Images (e.g., spectrograms)

- Include colorbar with units
- Label axes with quantities and units
- Annotate key features
- State measurement conditions

## Caption Writing

### Structure

1. **Title sentence**: What the figure shows (bold or italic optional)
2. **Panel descriptions**: What each panel/element shows
3. **Symbol/color legend**: What symbols and colors represent
4. **Method brief**: Key methodological details
5. **Conditions**: Sample, temperature, etc.
6. **Error description**: What uncertainty is shown

### Example Caption

> **Figure 3.4: Temperature dependence of qubit coherence times.** (a) Relaxation time T₁ (blue circles) and dephasing time T₂* (orange squares) as a function of sample stage temperature. Solid lines show fits to [equation reference]. (b) T₂ measured with CPMG dynamical decoupling using n = 1, 2, 4, 8 pulses (colors from light to dark). Error bars represent 1σ statistical uncertainty from exponential fits to decay curves. Each data point averages N = 50 measurements. Data collected on sample QC-17, operating point parameters in Table 3.2. The crossover temperature (~35 mK) corresponds to the regime where thermal photons begin to dominate, consistent with the analysis in Section 3.4.

### Common Caption Mistakes

❌ "Figure shows data."
✓ "Figure shows gate fidelity as a function of drive amplitude."

❌ "Error bars shown."
✓ "Error bars represent 1σ statistical uncertainty from N = 100 repetitions."

❌ "Data and theory."
✓ "Circles: experimental data. Dashed line: theoretical prediction from Eq. 3.5."

## File Formats and Export

### Vector vs. Raster

| Type | Use For | Formats | Benefits |
|------|---------|---------|----------|
| Vector | Line art, plots, diagrams | PDF, EPS, SVG | Infinite scalability |
| Raster | Photos, microscopy | PNG, TIFF | Faithful to original |

### Resolution Guidelines

- Minimum 300 dpi for raster images
- 600 dpi preferred for fine details
- Vector preferred when possible

### Export Checklist

- [ ] Correct dimensions
- [ ] Fonts embedded (for vector)
- [ ] Resolution sufficient (for raster)
- [ ] No compression artifacts
- [ ] Colors preserved correctly
- [ ] Transparency handled correctly

## Software Recommendations

### Python (matplotlib, seaborn)

**Pros**: Flexible, reproducible, scriptable
**Thesis settings:**
```python
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['pdf.fonttype'] = 42  # Embedded fonts
```

### MATLAB

**Pros**: Common in engineering, powerful built-in plotting
**Tips**: Use exportgraphics() for thesis-quality output

### Julia (Plots.jl, Makie.jl)

**Pros**: Fast, modern, flexible
**Tips**: Makie.jl offers publication quality with CairoMakie backend

### Origin, Igor Pro

**Pros**: GUI-based, powerful for experimental data
**Tips**: Use vector export, adjust font sizes before export

### Illustrator, Inkscape

**Pros**: Post-processing, complex diagrams
**Tips**: Finalize data plots in plotting software first

## Quality Assurance

### Review Checklist

Before finalizing each figure:

1. **View at 100%**: Does it look good at actual size?
2. **Print test**: Print and check readability
3. **Grayscale test**: Convert to grayscale
4. **Colorblind test**: Use online simulators
5. **Peer review**: Get fresh eyes on it
6. **Caption check**: Is it self-contained?

### Common Issues to Fix

| Issue | Solution |
|-------|----------|
| Text too small | Increase font sizes |
| Low contrast | Adjust colors, thicken lines |
| Cluttered | Remove non-essential elements |
| Inconsistent | Apply thesis style guide |
| Missing elements | Add error bars, labels, legend |

## Additional Resources

### Books
- Tufte, E. "The Visual Display of Quantitative Information"
- Cairo, A. "The Functional Art"
- Wilke, C. "Fundamentals of Data Visualization" (free online)

### Online
- ColorBrewer: colorbrewer2.org
- Colorblind simulator: color-blindness.com/coblis-color-blindness-simulator/
- matplotlib gallery: matplotlib.org/gallery/

### Papers
- Rougier et al. "Ten Simple Rules for Better Figures" (PLOS Comp Bio)
- Weissgerber et al. "Beyond Bar and Line Graphs" (PLOS Biology)
