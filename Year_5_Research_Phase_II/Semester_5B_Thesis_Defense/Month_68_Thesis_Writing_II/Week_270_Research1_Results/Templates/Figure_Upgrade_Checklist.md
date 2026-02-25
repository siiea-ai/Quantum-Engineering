# Figure Upgrade Checklist

## Purpose

This checklist guides the systematic upgrade of figures from paper quality to thesis quality. Each paper figure should be reviewed against these criteria and enhanced accordingly.

---

## Pre-Upgrade Assessment

For each figure from your paper, complete this assessment:

### Figure Identification

| Field | Value |
|-------|-------|
| Paper figure number | Fig. ___ |
| Thesis figure number | Figure N.___ |
| Figure type | ☐ Data plot ☐ Schematic ☐ Multi-panel ☐ Image ☐ Other |
| Current file format | ☐ PDF ☐ PNG ☐ EPS ☐ SVG ☐ Other: ___ |
| Original software | _______________ |
| Source data location | _______________ |

### Current State Assessment

| Criterion | Current | Target | Action Needed |
|-----------|---------|--------|---------------|
| Width | ___ in | 5-6 in | ☐ Resize |
| Font size | ___ pt | ≥10 pt | ☐ Increase |
| Resolution | ___ dpi | 300+ dpi | ☐ Regenerate |
| Error bars | ☐ Yes ☐ No | Yes | ☐ Add |
| Caption length | ___ lines | 5-15 lines | ☐ Expand |

---

## Technical Quality Checklist

### Resolution and Format

- [ ] **Vector format preferred** (PDF, EPS, SVG) for line art and plots
- [ ] **Raster at 300+ dpi** if vector not possible
- [ ] **No compression artifacts** visible at thesis print size
- [ ] **Fonts embedded** in vector files
- [ ] **Color space appropriate** (CMYK for print, RGB for screen)

### Size and Dimensions

- [ ] **Width: 5-6 inches** for single-column figures
- [ ] **Width: 4-5 inches** for multi-column or inset figures
- [ ] **Aspect ratio** appropriate for content (typically 4:3 or 1:1)
- [ ] **Panel sizes** consistent in multi-panel figures
- [ ] **Whitespace** balanced (not cramped, not excessive)

### Fonts and Text

- [ ] **Font size ≥ 10 pt** at final printed size
- [ ] **Font family** consistent with thesis body text (or standard sans-serif)
- [ ] **Font weight** readable (not too thin)
- [ ] **No text overlap** or crowding
- [ ] **Labels complete** with units

---

## Scientific Content Checklist

### Data Presentation

- [ ] **All relevant data points** shown (not just representative subset)
- [ ] **Error bars/bands** included for all data with uncertainty
- [ ] **Error type specified** (1σ, 2σ, 95% CI, SEM, etc.)
- [ ] **Outliers** shown if present (with explanation)
- [ ] **Connecting lines** appropriate (solid for continuous data, none for discrete)

### Theory/Model Comparison

- [ ] **Theoretical prediction** included if available
- [ ] **Theory uncertainty** shown if applicable
- [ ] **Legend distinguishes** data from theory clearly
- [ ] **Residuals** shown if fit quality important

### Statistical Information

- [ ] **Sample size** (N) indicated in caption or legend
- [ ] **Significance indicators** if comparing groups
- [ ] **Fit parameters** shown or referenced
- [ ] **Goodness of fit** (R², χ², etc.) if applicable

### Axes and Labels

- [ ] **Axis labels** include quantity name and units
- [ ] **Axis range** appropriate (data fills space, key features visible)
- [ ] **Tick marks** at sensible intervals
- [ ] **Tick labels** readable and not crowded
- [ ] **Scale type** (linear/log) appropriate for data
- [ ] **Zero included** if meaningful
- [ ] **Axis breaks** used appropriately if needed

---

## Visual Design Checklist

### Color Usage

- [ ] **Colorblind-accessible** palette (avoid red-green distinction alone)
- [ ] **Meaningful color mapping** (consistent colors for same quantities)
- [ ] **Sufficient contrast** between colors
- [ ] **Works in grayscale** if thesis may be printed B&W
- [ ] **Color legend** complete

### Symbol and Line Styles

- [ ] **Distinct symbols** for different data series
- [ ] **Symbol size** visible but not overwhelming
- [ ] **Line thickness** appropriate (visible at print size)
- [ ] **Line style** (solid, dashed, dotted) used for additional distinction
- [ ] **Legend** complete with all symbols/lines explained

### Layout and Composition

- [ ] **Data-to-ink ratio** optimized (minimize chartjunk)
- [ ] **Grid lines** minimal or absent unless helpful
- [ ] **Annotations** highlight key features without cluttering
- [ ] **Arrows/callouts** used sparingly and meaningfully
- [ ] **White space** used effectively

### Multi-Panel Figures

- [ ] **Panel labels** (a), (b), (c) clearly visible
- [ ] **Consistent axes** across panels where sensible
- [ ] **Aligned panels** for easy comparison
- [ ] **Shared legends** where appropriate
- [ ] **Logical panel arrangement** (reading order matches narrative)

---

## Caption Enhancement Checklist

### Required Caption Elements

- [ ] **Descriptive title** (what the figure shows)
- [ ] **Data description** (what symbols/lines represent)
- [ ] **Method summary** (how data was obtained)
- [ ] **Error bar definition** (what uncertainty is shown)
- [ ] **Sample/conditions** (which sample, what conditions)

### Optional Caption Elements

- [ ] **Theory description** (what model is shown, parameters)
- [ ] **Key observations** (what to notice)
- [ ] **Scale information** (for images)
- [ ] **Reference to text** (where discussed)
- [ ] **Panel descriptions** (for multi-panel figures)

### Caption Quality

- [ ] **Self-contained** (understandable without reading main text)
- [ ] **Complete** (all elements explained)
- [ ] **Concise** (no unnecessary words)
- [ ] **Consistent terminology** with main text
- [ ] **Correct numbering** matching text references

---

## Accessibility Checklist

### Visual Accessibility

- [ ] **Colorblind safe** (use ColorBrewer or similar palettes)
- [ ] **Pattern/texture** used in addition to color where possible
- [ ] **High contrast** between data and background
- [ ] **Large enough text** for readers with vision impairments

### Information Accessibility

- [ ] **Alt text ready** (can describe figure for screen readers)
- [ ] **No information only in color** (redundant encoding)
- [ ] **Clear without color** (works if printed B&W)

---

## Consistency Checklist

### Within-Thesis Consistency

- [ ] **Same font family** across all figures
- [ ] **Consistent color scheme** for same quantities throughout thesis
- [ ] **Same symbol usage** for recurring quantities
- [ ] **Consistent axis label format** (e.g., "Time (μs)" vs "Time / μs")
- [ ] **Same error bar style** throughout

### Between-Chapters Consistency

- [ ] **Numbering scheme** follows thesis convention
- [ ] **Caption style** consistent with other chapters
- [ ] **Cross-references** to related figures in other chapters

---

## Figure-Specific Checklists

### Data Plots (Line/Scatter)

- [ ] Data points clearly visible
- [ ] Error bars on all points with uncertainty
- [ ] Clear legend if multiple series
- [ ] Appropriate axis ranges
- [ ] Theory comparison if available

### Bar Charts

- [ ] Error bars shown
- [ ] Clear category labels
- [ ] Baseline at zero (unless log scale)
- [ ] Not 3D (avoid visual distortion)
- [ ] Color-coded legend if needed

### Histograms

- [ ] Bin width appropriate
- [ ] Y-axis labeled (count, frequency, or probability)
- [ ] Overlay (normal, fit) if relevant
- [ ] N specified

### Heat Maps/2D Plots

- [ ] Color scale clearly labeled
- [ ] Colorbar with units
- [ ] Appropriate color scale (diverging for signed data, sequential for unsigned)
- [ ] Key contours labeled if needed

### Schematics/Diagrams

- [ ] All components labeled
- [ ] Scale bar if applicable
- [ ] Arrow conventions explained
- [ ] Color coding explained
- [ ] Clean lines and text

### Photographs/Images

- [ ] Scale bar included
- [ ] Key features annotated
- [ ] Sufficient resolution
- [ ] Consistent exposure/contrast
- [ ] Artifact-free

---

## Upgrade Workflow

### Step 1: Gather Materials
- [ ] Locate original data files
- [ ] Locate plotting scripts
- [ ] Check software availability
- [ ] Identify target specifications

### Step 2: Regenerate Figure
- [ ] Open original script/file
- [ ] Adjust dimensions to thesis size
- [ ] Increase font sizes
- [ ] Add missing elements (error bars, theory, etc.)
- [ ] Apply consistent color scheme

### Step 3: Enhance Content
- [ ] Add additional data if available
- [ ] Include all error bars
- [ ] Add theory comparison
- [ ] Annotate key features
- [ ] Optimize axes

### Step 4: Write Caption
- [ ] Write draft caption
- [ ] Include all required elements
- [ ] Review for completeness
- [ ] Check length (5-15 lines typical)
- [ ] Verify terminology consistency

### Step 5: Review and Iterate
- [ ] Check against this checklist
- [ ] Get feedback from advisor/peers
- [ ] Iterate as needed
- [ ] Export final version

### Step 6: Integration
- [ ] Insert in thesis document
- [ ] Verify rendering quality
- [ ] Check cross-references
- [ ] Update figure list

---

## Quick Reference Tables

### Recommended Color Palettes

| Palette | Use Case | Colors |
|---------|----------|--------|
| ColorBrewer "Set2" | Qualitative data | 8 distinct colors |
| Viridis | Sequential data | Perceptually uniform |
| RdBu | Diverging data | Red to blue through white |
| Grays | B&W compatible | Multiple gray levels |

### Font Size Guidelines

| Element | Minimum Size | Recommended |
|---------|--------------|-------------|
| Axis labels | 10 pt | 12 pt |
| Tick labels | 8 pt | 10 pt |
| Legend | 8 pt | 10 pt |
| Annotations | 8 pt | 10 pt |
| Panel labels | 12 pt | 14 pt |

### Line Width Guidelines

| Element | Minimum | Recommended |
|---------|---------|-------------|
| Data lines | 1 pt | 1.5-2 pt |
| Theory lines | 0.75 pt | 1-1.5 pt |
| Axis lines | 0.5 pt | 0.75 pt |
| Grid lines | 0.25 pt | 0.5 pt |

---

## Sign-Off

### Figure: N.___

| Check | Initials | Date |
|-------|----------|------|
| Technical quality verified | | |
| Scientific content complete | | |
| Visual design optimized | | |
| Caption comprehensive | | |
| Consistency confirmed | | |
| Ready for thesis | | |
