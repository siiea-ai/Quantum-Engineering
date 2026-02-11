# Slide Design Guide for Physics Presentations

## Visual Communication Principles for Quantum Research

---

## Core Design Philosophy

### The Purpose of Slides

Slides are **visual aids**, not documents. They should:
- Support your spoken words, not replace them
- Focus attention on key concepts
- Provide visual representations of complex ideas
- Serve as memory anchors for your message

**If your slides make sense without you speaking, they have too much text.**

---

## The 5-Second Rule

**Test:** Can the audience understand the main point of a slide within 5 seconds?

### Applying the Rule

```
Before:                              After:
┌─────────────────────┐              ┌─────────────────────┐
│ Results of Our      │              │ 10x Resource        │
│ Numerical           │              │ Reduction Achieved  │
│ Simulations         │              │                     │
│                     │              │    ┌───────────┐    │
│ We performed        │              │    │ [Graph]   │    │
│ extensive numerical │              │    │           │    │
│ simulations using   │              │    └───────────┘    │
│ our novel algorithm │              │                     │
│ and found that the  │              │ Our protocol vs.    │
│ resource overhead   │              │ prior best: 100 vs  │
│ is reduced by a     │              │ 1000 T-gates        │
│ factor of...        │              │                     │
└─────────────────────┘              └─────────────────────┘

Time to understand: 30+ seconds      Time to understand: 5 seconds
```

---

## Layout and Visual Hierarchy

### Standard Slide Layout

```
┌─────────────────────────────────────────────────────────────┐
│ TITLE ANNOUNCES THE POINT (32-44 pt, bold)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌────────────────────────┐                               │
│    │                        │    ┌───────────────────┐      │
│    │    MAIN VISUAL         │    │ Supporting        │      │
│    │    (Figure, Diagram,   │    │ Information       │      │
│    │    or Equation)        │    │                   │      │
│    │                        │    │ - Key point       │      │
│    │                        │    │ - Key point       │      │
│    └────────────────────────┘    └───────────────────┘      │
│                                                              │
│ Brief explanatory text if absolutely necessary (24 pt)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Visual Hierarchy Principles

1. **Title (Most Prominent)**: States the slide's message
2. **Main Visual**: Largest element, draws eye first
3. **Key Points**: Supporting information, secondary focus
4. **Fine Print**: Axis labels, source citations, smallest

---

## Typography Standards

### Font Sizes

| Element | Minimum Size | Recommended |
|---------|-------------|-------------|
| Title | 32 pt | 36-44 pt |
| Body text | 24 pt | 26-28 pt |
| Equations | 24 pt | 28-32 pt |
| Bullet sub-items | 20 pt | 22-24 pt |
| Axis labels | 18 pt | 20-22 pt |
| Source citations | 14 pt | 16 pt |

### Font Selection

**Recommended:**
- Sans-serif for titles and body: Helvetica, Arial, Calibri, Open Sans
- Serif for equations: Computer Modern (LaTeX default), Times New Roman

**Avoid:**
- Decorative fonts
- Narrow/compressed fonts
- Low-contrast fonts (light gray on white)

### Text Guidelines

- **Maximum 6 lines** of text per slide
- **Maximum 6 words** per bullet point
- Use **fragments**, not complete sentences
- Parallel structure for bullet points

```
Good:                              Bad:
• Reduces resource overhead        • We have shown that resource
• Maintains fidelity above 99%       overhead can be reduced
• Scales to 100+ qubits           • The fidelity is maintained
                                   • Our approach scales well
```

---

## Color Design

### Color Palette Construction

**Base Colors:**
- Background: White or very light gray (#FFFFFF or #F5F5F5)
- Primary text: Black or very dark gray (#000000 or #333333)
- Accent: One primary color for emphasis

**Extended Palette:**
- Secondary accent: For second-level highlighting
- Complementary colors: For data visualization (max 5-6 distinct)

### Colorblind-Safe Palettes

**Avoid:** Red-green combinations

**Recommended Palettes:**
```
Blue-Orange:     #0077BB, #EE7733
Blue-Yellow:     #0077BB, #CCBB44
Blue-Red-Gray:   #0077BB, #CC3311, #888888

For multiple series:
#0077BB (Blue)
#33BBEE (Cyan)
#009988 (Teal)
#EE7733 (Orange)
#CC3311 (Red)
#EE3377 (Magenta)
#BBBBBB (Gray)
```

### Using Color Purposefully

| Use Color For | Example |
|---------------|---------|
| Emphasis | Key terms in a contrasting color |
| Grouping | Related items share a color |
| Data encoding | Different conditions/methods |
| Progress | Highlighting current section |

**Never use color as the only differentiator** - always combine with shape, pattern, or label.

---

## Equations and Mathematics

### When to Show Equations

| Show | Skip |
|------|------|
| Your main result | Standard definitions |
| Novel formulations | Well-known equations |
| Key relationships | Intermediate derivation steps |
| What you reference later | Equations you won't discuss |

### Equation Slide Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Our Protocol Achieves Optimal Scaling                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    Theorem (Main Result)                                     │
│    For any stabilizer code with distance d, our             │
│    protocol requires:                                        │
│                                                              │
│              $$T = O(d^2 \log d)$$                          │
│                                                              │
│    Key insight: By exploiting [technique], we               │
│    reduce the standard O(d³) scaling.                       │
│                                                              │
│    Compare to:                                               │
│    • Standard approach: O(d³)                               │
│    • Prior best: O(d² log² d)                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Equation Formatting Tips

1. **Box important results:**
   $$\boxed{F \geq 1 - O(\epsilon^2)}$$

2. **Use color for emphasis:**
   $$T = O(\textcolor{blue}{d^2} \log d)$$

3. **Break long equations across lines:**
   ```
   $$\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$$
   $$\text{where } \sum_k E_k^\dagger E_k = I$$
   ```

4. **Define notation on the same slide:**
   Include a small legend for non-obvious symbols

---

## Figure Design

### Types of Figures in Physics Talks

1. **Conceptual Diagrams**: Illustrate ideas, protocols, systems
2. **Data Plots**: Show experimental or numerical results
3. **Quantum Circuits**: Display algorithms and operations
4. **System Schematics**: Hardware, experimental setup

### Conceptual Diagram Guidelines

- **Simplify ruthlessly**: Remove all non-essential elements
- **Use consistent icons**: Same representation throughout talk
- **Add labels**: Don't assume interpretation
- **Build step-by-step**: Use animation for complex diagrams

```
Bad:  Everything at once, overwhelming
Good: Step 1 → Step 2 → Step 3 (builds)
```

### Data Plot Guidelines

**Essential Elements:**
- Clear, readable axis labels with units
- Legend (if multiple series)
- Title or caption stating the takeaway
- Error bars or confidence intervals

**Formatting:**
```python
# Example matplotlib settings for presentations
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.figsize': (10, 7),
    'lines.linewidth': 2.5,
    'lines.markersize': 10
})
```

**Common Mistakes:**
- Axis labels too small
- Legend obscuring data
- Too many series on one plot
- Missing units

### Quantum Circuit Guidelines

- Use standard gate symbols (IBM/Google conventions)
- Maintain consistent spacing
- Annotate key operations
- Color-code qubit types if relevant

```
Standard gates:
┌───┐  ┌───┐  ┌───┐
│ H │  │ X │  │ T │
└───┘  └───┘  └───┘

Control:    Measurement:
  │           ┌─┐
  ●           │M│
              └─┘
```

---

## Animation and Builds

### When to Use Animation

**Good Uses:**
- Revealing complex diagrams step-by-step
- Showing process/protocol progression
- Highlighting current topic in a list
- Before/after comparisons

**Bad Uses:**
- Decorative transitions (slide fly-ins)
- Every bullet point appearing separately (slows pace)
- Moving elements that distract from content

### Build Techniques

**Progressive Reveal:**
```
State 1:     State 2:     State 3:
[Step 1]     [Step 1]     [Step 1]
             [Step 2]     [Step 2]
                          [Step 3]
```

**Highlight Current:**
```
• Previous point (gray)
• CURRENT POINT (black, bold)
• Next point (gray or hidden)
```

**Before/After:**
```
┌───────────────┐         ┌───────────────┐
│ Before        │   →     │ After         │
│ [Old method]  │         │ [Our method]  │
└───────────────┘         └───────────────┘
```

---

## Slide Templates by Type

### Title Slide

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│                                                              │
│           PRESENTATION TITLE                                 │
│           Subtitle if needed                                 │
│                                                              │
│           Your Name                                          │
│           Institution                                        │
│           Collaborators                                      │
│                                                              │
│           Conference Name, Date                              │
│                                                              │
│           [Optional: Institution logos]                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Theorem/Result Slide

```
┌─────────────────────────────────────────────────────────────┐
│ Descriptive Title Stating the Result                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌───────────────────────────────────────────────────┐    │
│    │ Theorem [Name]                                     │    │
│    │                                                    │    │
│    │ [Statement in plain language]                      │    │
│    │                                                    │    │
│    │        $$\text{Formal equation}$$                  │    │
│    │                                                    │    │
│    └───────────────────────────────────────────────────┘    │
│                                                              │
│    Key insight: [Why this works / Why it's surprising]      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Slide

```
┌─────────────────────────────────────────────────────────────┐
│ Takeaway Message (Not "Results" or "Data")                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌─────────────────────────────────────────────────┐      │
│    │                                                   │      │
│    │             [FIGURE/PLOT]                        │      │
│    │                                                   │      │
│    │                                                   │      │
│    └─────────────────────────────────────────────────┘      │
│                                                              │
│    • Key observation 1                                       │
│    • Key observation 2                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Comparison Slide

```
┌─────────────────────────────────────────────────────────────┐
│ Our Approach Outperforms Prior Methods                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌─────────────────┬─────────────────┬─────────────────┐  │
│    │ Method          │ Metric 1        │ Metric 2        │  │
│    ├─────────────────┼─────────────────┼─────────────────┤  │
│    │ Prior Best      │ 1000            │ 95%             │  │
│    │ Alternative     │ 800             │ 97%             │  │
│    │ OUR WORK        │ 100 ✓           │ 99.5% ✓         │  │
│    └─────────────────┴─────────────────┴─────────────────┘  │
│                                                              │
│    10× improvement in resources while maintaining fidelity   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Summary Slide

```
┌─────────────────────────────────────────────────────────────┐
│ Summary                                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    What We Did:                                              │
│    → [One sentence summary of approach]                      │
│                                                              │
│    What We Found:                                            │
│    → [Key result 1]                                          │
│    → [Key result 2]                                          │
│                                                              │
│    What It Means:                                            │
│    → [Significance for the field]                            │
│                                                              │
│    Paper: arXiv:XXXX.XXXXX                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Considerations

### File Formats

**For Figures:**
- Vector (PDF, SVG): Scales perfectly, preferred for diagrams
- Raster (PNG): Use 300+ DPI for photos/complex plots
- Avoid: JPG (artifacts), low-resolution images

**For Presentation Files:**
- Always bring PDF backup
- Test on conference computers beforehand
- Embed fonts if using PowerPoint

### Aspect Ratios

- **16:9**: Modern standard, most projectors
- **4:3**: Older projectors, some venues
- **Check with conference** if uncertain

### Compatibility Checklist

- [ ] Fonts display correctly on different computers
- [ ] Equations render properly (not as images if using LaTeX)
- [ ] Animations work in PDF (or have static backup)
- [ ] Colors print acceptably in grayscale (for handouts)
- [ ] All figures are high resolution

---

## Quick Reference: Design Dos and Don'ts

### Do

- Use the title to state your message
- Limit text to essential points
- Choose readable fonts at large sizes
- Use color purposefully
- Create high-resolution figures
- Test on the actual presentation system

### Don't

- Read your slides aloud
- Use complete sentences
- Choose decorative fonts
- Use red-green color combinations
- Stretch low-resolution images
- Assume your computer will be available
