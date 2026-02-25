# Week 230: Methods & Results Sections

## Overview

**Days:** 1604-1610 (7 days)
**Theme:** Writing the Technical Core of Your Paper
**Deliverable:** Complete Methods and Results sections with figures

This week focuses on drafting the technical heart of your paper. Methods and Results are often the most straightforward sections to write because they describe concrete work. Starting here builds momentum and establishes the foundation for Introduction and Discussion.

## Learning Objectives

By the end of this week, you will be able to:

1. **Write Methods** with sufficient detail for reproducibility
2. **Present Results** objectively with appropriate statistical context
3. **Design effective figures** that communicate clearly
4. **Write figure captions** that stand alone
5. **Balance detail** between main text and supplementary material

## Daily Schedule

### Day 1604 (Monday): Methods Section Structure

**Morning (3 hours):** Methods fundamentals
- Read `Guide.md` on Methods writing
- Review Methods sections from your 3 model papers
- Outline your Methods subsections

**Afternoon (3 hours):** Draft Methods outline
- List all methods/procedures to describe
- Organize into logical subsections
- Identify what belongs in supplementary

**Evening (2 hours):** Begin Methods draft
- Write first subsection (system/model description)
- Focus on completeness, not polish
- Note gaps requiring additional detail

### Day 1605 (Tuesday): Methods Writing

**Morning (3 hours):** Continue Methods draft
- Write experimental/computational procedure subsection
- Include all relevant parameters
- Reference established methods appropriately

**Afternoon (3 hours):** Complete Methods draft
- Write data analysis/statistical methods subsection
- Document software and algorithms used
- Draft equations as needed

**Evening (2 hours):** Methods review
- Check for reproducibility
- Verify technical accuracy
- Identify missing details

### Day 1606 (Wednesday): Results Organization

**Morning (3 hours):** Results planning
- Read `Guide.md` Results section
- Organize your results logically
- Match results to outline from Week 229

**Afternoon (3 hours):** Begin Results draft
- Write first major result subsection
- Present data with quantitative specifics
- Reference figures as you write

**Evening (2 hours):** Assess progress
- Review what you've written
- Adjust organization if needed
- Plan remaining results content

### Day 1607 (Thursday): Results Writing

**Morning (3 hours):** Continue Results draft
- Write second major result subsection
- Maintain objective presentation
- Include error analysis

**Afternoon (3 hours):** Complete Results draft
- Write remaining result subsections
- Ensure all claims are supported
- Check logical flow

**Evening (2 hours):** Results self-review
- Verify all figures are referenced
- Check for unsupported claims
- Note areas needing revision

### Day 1608 (Friday): Figure Design

**Morning (3 hours):** Figure assessment
- Review `Resources/Figure_Guidelines.md`
- Evaluate existing figures against standards
- Identify figures needing improvement

**Afternoon (3 hours):** Figure refinement
- Redesign figures for clarity
- Standardize style across all figures
- Create any missing figures

**Evening (2 hours):** Figure finalization
- Export figures in correct format
- Check resolution and sizing
- Organize figure files

### Day 1609 (Saturday): Figure Captions

**Morning (3 hours):** Caption writing
- Review `Templates/Figure_Caption.md`
- Write comprehensive captions
- Ensure captions stand alone

**Afternoon (3 hours):** Caption refinement
- Check for completeness
- Define all symbols and abbreviations
- Add quantitative context

**Evening (2 hours):** Integration check
- Verify figure-text consistency
- Check numbering and references
- Review overall figure strategy

### Day 1610 (Sunday): Week Integration & Review

**Morning (3 hours):** Section integration
- Read Methods and Results together
- Check for consistency
- Smooth transitions

**Afternoon (3 hours):** Revision pass
- Apply feedback from self-review
- Polish prose
- Fill any remaining gaps

**Evening (2 hours):** Self-assessment
- Complete week checklist
- Document remaining issues
- Plan advisor meeting

## Key Concepts

### Methods Writing Principles

**The Reproducibility Standard:**
A competent researcher in your field should be able to reproduce your work from your Methods description. This requires:

- Complete parameter specifications
- Clear procedural descriptions
- Identification of critical steps
- Acknowledgment of approximations

**What to Include:**
| Category | Include | Example |
|----------|---------|---------|
| Equipment | Manufacturer, model, key specs | "Dilution refrigerator (Bluefors, LD400, base temperature 10 mK)" |
| Software | Name, version, key settings | "Numerical simulations used QuTiP v4.6 [ref]" |
| Parameters | All relevant values | "Qubit frequency: 5.123 GHz, anharmonicity: -340 MHz" |
| Procedures | Step-by-step when novel | "Calibration sequence: ..." |

### Results Writing Principles

**Objective Presentation:**
Results should present findings without interpretation. Reserve interpretation for Discussion.

**Good:** "The measured fidelity was 99.7% ± 0.1%"
**Avoid:** "The fidelity was excellent at 99.7%"

**Structure Options:**

1. **Chronological:** Present results in order performed
2. **Logical:** Build from simple to complex
3. **Figure-organized:** Structure around key figures

### Figure Principles

**The 10-Second Test:**
A reader should understand your figure's main point within 10 seconds.

**Figure Types:**
| Type | Purpose | Example |
|------|---------|---------|
| Schematic | Explain concept/setup | Device diagram |
| Data | Present measurements | Scatter plot |
| Comparison | Show agreement | Theory vs. experiment |
| Summary | Synthesize results | Phase diagram |

## Resources for This Week

- `Guide.md` - Comprehensive Methods and Results writing guide
- `Resources/Figure_Guidelines.md` - Figure design standards
- `Templates/Methods_Section.md` - Methods section template
- `Templates/Results_Section.md` - Results section template
- `Templates/Figure_Caption.md` - Caption writing template

## Self-Assessment Checklist

### Methods Section

- [ ] All procedures described with sufficient detail
- [ ] Parameters specified with appropriate precision
- [ ] Equipment and software properly documented
- [ ] Established methods referenced appropriately
- [ ] Approximations and assumptions stated
- [ ] Data analysis methods explained

### Results Section

- [ ] All major findings presented
- [ ] Quantitative values with uncertainties throughout
- [ ] Each claim supported by data
- [ ] Presentation is objective (no interpretation)
- [ ] Figures referenced in logical order
- [ ] Negative results included if relevant

### Figures

- [ ] Each figure has clear purpose
- [ ] Visual design is professional
- [ ] All figures referenced in text
- [ ] Captions are comprehensive
- [ ] Symbols and abbreviations defined
- [ ] Resolution meets journal requirements

### Deliverables

- [ ] Complete Methods section draft
- [ ] Complete Results section draft
- [ ] All figures created/refined
- [ ] All figure captions written
- [ ] Supplementary material organized

## Common Challenges

### "I don't know how much detail to include"

Apply the "competent researcher" test: Could someone in your subfield reproduce this? For standard methods, a reference suffices. For novel methods, provide complete detail. When in doubt, put extra detail in supplementary.

### "My results don't tell a clear story"

Reorganize around your main finding. Lead with your strongest result. Build supporting results around it. Remove tangential findings to supplementary or future work.

### "My figures look unprofessional"

Focus on clarity first, aesthetics second. Use consistent colors, fonts, and styles. Follow journal guidelines. Consider software like matplotlib with publication-quality settings.

## Writing Tips for This Week

### Methods

1. **Use past tense:** "We measured..." not "We measure..."
2. **Be specific:** "100 averages" not "multiple averages"
3. **Active voice where possible:** "We calibrated..." not "Calibration was performed..."
4. **Define notation on first use**

### Results

1. **Lead with data:** "Figure 2 shows..." not "We can see in Figure 2..."
2. **Quantify everything:** Include values, uncertainties, sample sizes
3. **Compare with theory:** Note agreement or disagreement
4. **Acknowledge anomalies:** Don't hide unexpected results

### Figures

1. **One main point per figure:** Split complex figures
2. **Visible at print size:** Test at final dimensions
3. **Colorblind-friendly:** Use distinguishable colors
4. **Label axes completely:** Units, scale, key values

## Connection to Other Sections

Methods and Results form the technical core that supports all other sections:

- **Introduction** previews the approach and key results
- **Discussion** interprets the Results in context
- **Conclusions** summarize the Results' significance
- **Abstract** condenses the key findings

Strong Methods and Results make other sections easier to write.

## Advisor Interaction

Share your Methods and Results draft with your advisor for feedback on:

1. Technical accuracy
2. Level of detail
3. Figure effectiveness
4. Missing content
5. Organizational clarity

Request specific feedback rather than general comments.

---

*"Data are just summaries of thousands of stories—tell a few of those stories to help make the data meaningful." - Dan Heath*

But in Results, let the data tell the story objectively. You'll interpret in Discussion.
