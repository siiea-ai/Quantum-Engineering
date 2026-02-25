# Week 234: Figure Refinement

## Days 1632-1638 | Publication-Quality Visualization

### Overview

Figures are often the most impactful elements of a scientific paper. Reviewers and readers frequently examine figures before reading text, and many readers will only look at your figures. This week transforms working figures into publication-quality graphics that effectively communicate your research.

### Learning Objectives

By the end of this week, you will be able to:

1. Apply journal-specific style guidelines (APS, Nature, Science formats)
2. Design colorblind-accessible visualizations
3. Create vector graphics for scalability and print quality
4. Compose effective multi-panel figures
5. Write informative figure captions
6. Optimize figures for different publication contexts (print, web, presentations)

### Daily Schedule

| Day | Date | Focus | Key Activity |
|-----|------|-------|--------------|
| 1632 | Monday | Figure Audit | Inventory and assess current figures |
| 1633 | Tuesday | Style Guidelines | Apply journal-specific formatting |
| 1634 | Wednesday | Color & Accessibility | Implement colorblind-safe palettes |
| 1635 | Thursday | Multi-Panel Design | Create composite figures |
| 1636 | Friday | Caption Writing | Draft comprehensive captions |
| 1637 | Saturday | Technical Refinement | High-resolution output, vector formats |
| 1638 | Sunday | Integration | Finalize figures, update manuscript |

### The Role of Figures in Scientific Communication

#### Why Figures Matter

**Cognitive load:** Visual processing is faster and more efficient than text parsing. A well-designed figure can communicate in seconds what text requires minutes to convey.

**First impressions:** Many readers scan figures first to determine if a paper warrants detailed reading.

**Memorability:** Readers remember visual information better than text. Your figure may be how readers recall your work.

**Universal language:** Figures transcend language barriers in ways text cannot.

#### Figure Philosophy

Think of each figure as a **mini-paper** with:
- A central message (the "finding")
- Supporting evidence (the data shown)
- Context (labels, legends, captions)
- Quality standards (clarity, accuracy)

**The One-Sentence Test:** You should be able to describe what each figure shows in one clear sentence. If you cannot, the figure may be trying to do too much.

### Key Concepts

#### 1. Figure Types and Their Purposes

| Type | Purpose | Best For | Example |
|------|---------|----------|---------|
| **Schematic** | Explain concept or setup | Apparatus, workflows, concepts | Fig. 1: Experimental setup |
| **Data plot** | Present quantitative results | Measurements, comparisons | Fig. 2: Gate fidelity vs. drive amplitude |
| **Comparison** | Show agreement/disagreement | Theory vs. experiment | Fig. 3: Simulation overlay on data |
| **Phase diagram** | Map parameter space | Regime identification | Fig. 4: Coherence as function of T, B |
| **Process flow** | Show sequence | Protocols, algorithms | Fig. 5: Measurement sequence |

#### 2. Publication Standards by Journal

**American Physical Society (PRL, PRA, PRB):**
- Single column: 8.6 cm (3.4 in)
- Double column: 17.8 cm (7.0 in)
- Resolution: 600 DPI minimum
- Format: EPS, PDF (vector preferred)
- Font: 8-10 pt for labels

**Nature Publishing Group:**
- Single column: 89 mm
- Double column: 183 mm
- Resolution: 300 DPI (halftone), 1000 DPI (line art)
- Format: EPS, PDF, TIFF
- Font: 5-7 pt minimum

**Science/AAAS:**
- Single column: 55 mm
- Double column: 115 mm
- Full page: 178 mm
- Resolution: 300+ DPI
- Format: EPS, PDF preferred

#### 3. Color Theory for Scientific Visualization

**Primary Principles:**

1. **Use color meaningfully:** Color should encode information, not decoration
2. **Consider colorblind viewers:** 8% of males have color vision deficiency
3. **Print compatibility:** Colors must work in grayscale
4. **Consistency:** Same meaning for same color throughout paper

**Colorblind-Safe Palettes:**

```python
# Wong color palette (optimized for colorblindness)
wong_colors = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'yellow': '#F0E442',
    'sky_blue': '#56B4E9',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000'
}
```

**Problematic combinations to avoid:**
- Red/green (most common deficiency)
- Blue/purple
- Green/brown
- Light colors on white

#### 4. Typography in Figures

**Font Requirements:**
- Sans-serif fonts preferred (Helvetica, Arial)
- Consistent font throughout all figures
- Match document body font when possible
- Minimum readable size: 6 pt at final publication size

**Label Hierarchy:**
- Axis titles: Larger, bold acceptable
- Tick labels: Standard size
- Annotations: Smaller, but readable
- Panel labels: (a), (b), (c) in bold

**Mathematical Notation:**
- Use proper symbols (not text approximations)
- Match manuscript equation formatting
- LaTeX rendering preferred

### Figure Quality Checklist

**Before you begin refinement, assess each figure:**

```markdown
## Figure Assessment Form

**Figure Number:** ___
**Purpose:** ___________________
**Current State:** ☐ Draft ☐ Working ☐ Near-final

### Content Assessment
- [ ] Message is clear without caption
- [ ] Data is accurately represented
- [ ] Appropriate plot type for data
- [ ] No unnecessary elements
- [ ] Complete (all needed data shown)

### Technical Assessment
- [ ] Resolution sufficient for publication
- [ ] Vector format available
- [ ] Colors print well in grayscale
- [ ] Labels readable at publication size
- [ ] Axes properly labeled with units

### Accessibility Assessment
- [ ] Colorblind-safe palette
- [ ] Patterns supplement colors if needed
- [ ] Sufficient contrast
- [ ] Clear without color

### Style Assessment
- [ ] Matches journal guidelines
- [ ] Consistent with other figures
- [ ] Professional appearance
- [ ] No distracting elements
```

### Common Figure Problems and Solutions

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Cluttered** | Too much information | Split into panels or separate figures |
| **Low resolution** | Pixelated when zoomed | Re-export at higher DPI, use vector |
| **Poor contrast** | Hard to distinguish elements | Increase line weights, adjust colors |
| **Inconsistent style** | Figures look different | Apply unified style template |
| **Redundant** | Multiple figures show same thing | Combine or eliminate |
| **Wrong type** | Bar chart for continuous data | Choose appropriate plot type |
| **Missing context** | Readers can't interpret | Add labels, legends, reference lines |

### Practical Exercises

#### Exercise 1: Figure Audit (Day 1632)

Create inventory of all figures:

| Fig | Type | Purpose | Panels | Resolution | Format | Issues | Priority |
|-----|------|---------|--------|------------|--------|--------|----------|
| 1 | | | | | | | |
| 2 | | | | | | | |
| ... | | | | | | | |

#### Exercise 2: Color Accessibility Check (Day 1634)

For each figure with color:
1. Convert to grayscale - is information preserved?
2. Run through colorblindness simulator
3. Add patterns/markers if needed

**Online tools:**
- Coblis (Color Blindness Simulator)
- Viz Palette
- ColorBrewer 2.0

#### Exercise 3: Caption Drafting (Day 1636)

Caption structure template:

```markdown
**Figure N. [Title: what the figure shows]**
(a) [Description of panel a]. [Key observation in panel a].
(b) [Description of panel b]. [Key observation in panel b].
[Overall interpretation if needed].
[Technical details: sample parameters, measurement conditions].
[Symbol/color definitions if not in legend].
Error bars represent [standard deviation/standard error/95% CI].
```

### Resources

- [Guide.md](Guide.md) - Detailed figure design methodology
- [Code/publication_figures.py](Code/publication_figures.py) - Python templates
- [Templates/Figure_Checklist.md](Templates/Figure_Checklist.md) - Quality assurance template

### Checklist for Week 234

- [ ] Completed figure audit
- [ ] Applied journal style guidelines to all figures
- [ ] Verified colorblind accessibility
- [ ] Created high-resolution vector outputs
- [ ] Designed effective multi-panel compositions
- [ ] Drafted comprehensive captions
- [ ] Tested figures at publication size
- [ ] Updated manuscript with refined figures

### Transition to Week 235

With publication-quality figures complete, Week 235 focuses on polishing the prose at the sentence and paragraph level. The visual narrative established this week provides the foundation for textual narrative refinement.

---

*Week 234 of 260 | Month 59 | Year 4*
