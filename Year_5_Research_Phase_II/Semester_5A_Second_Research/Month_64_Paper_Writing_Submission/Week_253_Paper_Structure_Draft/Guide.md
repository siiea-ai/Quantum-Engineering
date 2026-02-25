# Week 253: Paper Structure and Draft

## Days 1765-1771 | Transforming Research into Manuscript

---

## Overview

This week marks the transition from researcher to author. You will transform your experimental results, theoretical developments, and computational analyses into a cohesive scientific narrative. The goal is a complete first draft—imperfect but comprehensive—that captures your research story from motivation to implications.

### Week Objectives

By the end of Week 253, you will have:

1. Finalized the paper structure appropriate for your research type
2. Written a complete first draft of all major sections
3. Created preliminary figures with captions
4. Integrated references using proper citation management
5. Established the narrative arc that guides readers through your work

---

## The Scientific Paper: Anatomy and Purpose

### The IMRAD Structure

Most quantum physics papers follow the IMRAD framework, adapted for theoretical and computational work:

| Section | Purpose | Typical Length |
|---------|---------|----------------|
| **Title** | Capture the main finding; enable discovery | 10-15 words |
| **Abstract** | Summarize the entire paper | 150-250 words |
| **Introduction** | Motivate and contextualize | 1-2 pages |
| **Methods/Theory** | Enable reproduction/verification | 1-3 pages |
| **Results** | Present findings objectively | 2-4 pages |
| **Discussion** | Interpret and contextualize | 1-2 pages |
| **Conclusion** | Summarize and look forward | 0.5-1 page |

### Quantum Computing Paper Variations

Different research types require modified structures:

**Theoretical Papers**:
- Methods → Theoretical Framework
- Results → Analytical Results + Numerical Validation
- Extended appendices with derivations

**Experimental Papers**:
- Detailed experimental setup section
- Systematic uncertainty analysis
- Device characterization in supplementary

**Algorithm Papers**:
- Problem statement section
- Algorithm description (pseudocode)
- Complexity analysis
- Benchmarking results

**Hybrid Theory-Experiment**:
- Parallel presentation of theory predictions and experimental tests
- Comparison figures central to results

---

## Day-by-Day Structure

### Day 1765 (Monday): Outline and Narrative Arc

**Morning (3 hours): Structural Planning**

Before writing prose, establish your paper's skeleton:

1. **Identify Your Central Claim**

   Every successful paper makes one main claim clearly. Ask yourself:
   - What is the single most important thing I discovered?
   - Why should the quantum physics community care?
   - What changes because of this work?

   *Example claims in quantum computing*:
   - "We demonstrate fault-tolerant operation of a logical qubit below the threshold"
   - "Our protocol achieves exponential speedup for quantum simulation of fermions"
   - "This noise model explains previously anomalous decoherence in transmon qubits"

2. **Construct the Narrative Arc**

   Your paper should answer these questions in order:
   - Why is this problem important? (Introduction)
   - What did you do? (Methods)
   - What did you find? (Results)
   - What does it mean? (Discussion)
   - What comes next? (Conclusion)

3. **Create Section Outline**

```markdown
# Paper Outline Template

## Title Options
1. [Primary option]
2. [Alternative 1]
3. [Alternative 2]

## Abstract (write last, outline now)
- Context: [1 sentence]
- Problem: [1 sentence]
- Approach: [1 sentence]
- Key results: [2-3 sentences]
- Significance: [1 sentence]

## I. Introduction
### A. Opening hook / Broad context
### B. Specific problem and why it matters
### C. Previous approaches and their limitations
### D. Our contribution (preview)
### E. Paper structure (optional)

## II. Background/Theory
### A. Essential formalism
### B. Key prior results we build on
### C. Notation and conventions

## III. Methods/Approach
### A. Theoretical framework
### B. Computational methods
### C. Experimental setup (if applicable)

## IV. Results
### A. Main result 1 (with figure)
### B. Main result 2 (with figure)
### C. Supporting results
### D. Comparison with prior work

## V. Discussion
### A. Interpretation of results
### B. Limitations and assumptions
### C. Broader implications
### D. Future directions

## VI. Conclusion
### A. Summary of contributions
### B. Open questions
### C. Outlook

## Figures List
1. [Schematic of system/protocol]
2. [Main result visualization]
3. [Comparison/benchmarking]
4. [Additional results]

## Supplementary Material
- Extended derivations
- Additional data
- Code availability
```

**Afternoon (3 hours): Figure Planning**

Figures drive papers. Plan them before writing:

1. **List All Potential Figures** (aim for 4-6 in main text)
2. **Prioritize by Impact**: Which figures tell the story?
3. **Create Rough Sketches**: Hand-drawn layouts are fine
4. **Write Preliminary Captions**: Forces you to articulate what each shows

**Figure Types in Quantum Papers**:
- **Schematic**: System diagram, protocol flowchart, circuit diagram
- **Data Plot**: Experimental or simulation results
- **Comparison**: Your method vs. alternatives
- **Theoretical**: Phase diagrams, parameter landscapes
- **Combined**: Multi-panel figures telling a story

**Evening (1 hour): Research Documentation Review**

Review all research notes, code, and data from your project:
- Ensure you can reproduce all results
- Identify any gaps needing additional runs
- Organize references you'll cite

---

### Day 1766 (Tuesday): Introduction Drafting

**Morning (3 hours): The Art of the Introduction**

The introduction is your paper's sales pitch. It must:
1. Hook readers within the first paragraph
2. Establish the problem's importance
3. Position your work relative to the field
4. Clearly state your contribution

**Opening Paragraph Strategy**:

*Broad Context → Specific Problem → Your Work*

**Example for quantum error correction paper**:

> "Realizing fault-tolerant quantum computation requires protecting quantum information from environmental decoherence and operational errors [1-3]. The surface code has emerged as a leading candidate due to its high error threshold and local stabilizer measurements [4,5]. However, achieving logical error rates below physical error rates—the hallmark of fault tolerance—has remained experimentally elusive. Here, we demonstrate..."

**The Literature Review Component**:

Structure your literature discussion to create a "gap" your paper fills:

1. **Establish the Field**: Cite foundational and recent review papers
2. **Highlight Relevant Work**: Discuss papers most related to yours
3. **Identify the Gap**: What hasn't been done? What questions remain?
4. **Fill the Gap**: Your contribution addresses this

**Avoid Common Introduction Errors**:
- ❌ Starting too broadly ("Quantum mechanics has revolutionized...")
- ❌ Excessive literature review that buries your contribution
- ❌ Claiming too much ("for the first time ever...")
- ❌ Being vague about your actual contribution
- ✅ Specific, honest, positioned relative to recent work

**Afternoon (3 hours): Draft the Full Introduction**

Write a complete introduction draft (aim for 1-2 pages, 600-1000 words):

```markdown
# Introduction Drafting Checklist

Paragraph 1: The Hook
- [ ] Opens with specific, compelling context
- [ ] Establishes why quantum [X] matters
- [ ] Accessible to general physics audience

Paragraph 2-3: The Problem
- [ ] Defines specific problem you address
- [ ] Explains why it's challenging
- [ ] Discusses practical implications

Paragraph 4-5: Prior Work
- [ ] Reviews relevant approaches
- [ ] Honestly discusses their strengths
- [ ] Identifies remaining challenges

Paragraph 6: Your Contribution
- [ ] States what you do in this paper
- [ ] Highlights key results
- [ ] Explains significance

Paragraph 7 (optional): Paper Structure
- [ ] Brief roadmap of remaining sections
```

**Evening (1 hour): Introduction Refinement**

Read your introduction aloud. Mark passages that:
- Feel awkward or unclear
- Make claims without support
- Require citation
- Could be more specific

---

### Day 1767 (Wednesday): Methods/Theory Section

**Morning (3 hours): Methods Writing Philosophy**

The Methods section answers: "How did you do it?"

For theoretical papers, this becomes the **Theoretical Framework**.
For computational papers, include **Numerical Methods**.
For experimental papers, detail the **Experimental Setup**.

**Key Principle**: Enable a qualified reader to reproduce your work.

**Structure for Quantum Computing Methods**:

```markdown
## Methods

### A. System Model / Theoretical Framework

Define the physical system and mathematical model:
- Hamiltonian(s) used
- Approximations made
- Key parameters

Example:
"We consider a system of $n$ transmon qubits with Hamiltonian
$$H = \sum_i \omega_i a_i^\dagger a_i - \frac{\alpha_i}{2} a_i^\dagger a_i^\dagger a_i a_i + \sum_{ij} g_{ij}(a_i^\dagger a_j + a_i a_j^\dagger)$$
where $\omega_i$ are the qubit frequencies, $\alpha_i$ the anharmonicities,
and $g_{ij}$ the coupling strengths."

### B. Protocol / Algorithm

Describe your approach step-by-step:
- Main algorithmic steps
- Key innovations
- Pseudocode if helpful

### C. Numerical Methods

For simulations, specify:
- Simulation technique (exact diagonalization, tensor networks, Monte Carlo, etc.)
- Convergence criteria
- Computational resources used

### D. Experimental Details (if applicable)

Include:
- Device parameters
- Measurement protocols
- Calibration procedures
```

**Afternoon (3 hours): Draft Methods Section**

Write complete methods (aim for 1-3 pages depending on paper type):

**Best Practices**:
- Define all notation before using it
- State assumptions explicitly
- Reference well-known techniques rather than re-deriving
- Move lengthy derivations to supplementary material
- Include parameter tables

**Example Parameter Table**:

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Qubit frequency | $\omega_q/2\pi$ | 5.0 | GHz |
| Anharmonicity | $\alpha/2\pi$ | -200 | MHz |
| T1 | $T_1$ | 50 | $\mu$s |
| T2 | $T_2$ | 30 | $\mu$s |

**Evening (1 hour): Methods Clarity Check**

Review with these questions:
- Could a graduate student reproduce this?
- Are all symbols defined?
- Are approximations justified?
- Is anything missing?

---

### Day 1768 (Thursday): Results Section

**Morning (3 hours): Presenting Results Effectively**

The Results section presents findings objectively, saving interpretation for Discussion.

**Structure Strategy**: Let figures drive the organization

```markdown
## Results

### A. [First Major Finding - Corresponds to Fig. 2]

Present your primary result:
- State the finding clearly
- Reference the figure
- Provide quantitative details

"Figure 2 shows the measured logical error rate as a function of physical
error rate. We observe that for $p < 0.8\%$, the logical error rate scales
as $p_L \propto p^{(d+1)/2}$ where $d=3$ is the code distance, indicating
below-threshold operation."

### B. [Second Major Finding - Corresponds to Fig. 3]

Continue with secondary results:
- Maintain logical flow
- Connect to first result
- Include uncertainty/error analysis

### C. Supporting Results

Present additional results that:
- Validate main findings
- Address potential concerns
- Provide comprehensive picture

### D. Comparison with Prior Work

If applicable, compare directly:
- Same metrics, different methods
- Improvements quantified
- Fair comparison conditions stated
```

**Afternoon (3 hours): Draft Results Section**

Write complete results (aim for 2-4 pages):

**Results Writing Guidelines**:

1. **Lead with Findings**: Start each paragraph with the key result

   ✅ "The fidelity reached 99.5% after optimization."
   ❌ "After running the optimization algorithm, we measured the fidelity..."

2. **Quantify Everything**: Include numbers with uncertainties

   ✅ "We achieve a gate fidelity of $99.5 \pm 0.1\%$"
   ❌ "We achieve high gate fidelity"

3. **Reference Figures Strategically**: Guide readers through data

   "As shown in Fig. 2(a), the coherence time increases linearly with..."

4. **Present Negative Results**: If something didn't work, explain briefly

5. **Organize Hierarchically**: Most important results first

**Evening (1 hour): Figure-Text Alignment**

Verify each figure is:
- Referenced in the text
- Explained sufficiently
- Placed near relevant discussion
- Captioned appropriately

---

### Day 1769 (Friday): Discussion and Conclusion

**Morning (3 hours): Crafting the Discussion**

The Discussion interprets your results and places them in context.

**Discussion Structure**:

```markdown
## Discussion

### Interpretation of Main Results

What do your findings mean?
- Connect to theoretical expectations
- Explain surprises or deviations
- Propose mechanisms for observations

### Comparison to Other Work

How do your results compare?
- Quantitative comparison when possible
- Explain differences
- Acknowledge advantages of other approaches

### Limitations

Be honest about limitations:
- Experimental constraints
- Model assumptions
- Finite-size effects
- Generalizability concerns

### Implications

What are the broader impacts?
- For quantum computing development
- For fundamental understanding
- For practical applications

### Future Directions

What comes next?
- Natural extensions
- Remaining challenges
- Open questions raised by this work
```

**Afternoon (2 hours): Writing the Conclusion**

The Conclusion is not a summary—it's a synthesis.

**Conclusion Structure** (0.5-1 page):

1. **Restate Main Contribution** (1-2 sentences)
2. **Key Findings Summary** (2-3 sentences)
3. **Significance** (1-2 sentences)
4. **Outlook** (2-3 sentences)

**Example Conclusion Structure**:

> "In this work, we demonstrated [main contribution]. Our key findings—[result 1], [result 2], and [result 3]—establish [significance]. Looking forward, [future direction 1] and [future direction 2] represent natural extensions. This work provides [broader impact statement]."

**Evening (2 hours): Draft Discussion and Conclusion**

Complete drafts of both sections, even if rough.

---

### Day 1770 (Saturday): Abstract and Title Refinement

**Morning (3 hours): Crafting the Abstract**

The abstract is the most-read part of your paper. It must stand alone.

**Abstract Structure** (150-250 words):

```markdown
## Abstract Template

[CONTEXT - 1 sentence]
"Quantum error correction is essential for fault-tolerant quantum computation."

[PROBLEM - 1 sentence]
"However, demonstrating below-threshold logical operation remains challenging."

[APPROACH - 1-2 sentences]
"Here, we implement the surface code on a 17-qubit superconducting processor
using a novel stabilizer measurement protocol."

[RESULTS - 2-3 sentences]
"We achieve a logical error rate of X%, Y% below the physical error rate,
demonstrating below-threshold operation. Furthermore, we show that..."

[SIGNIFICANCE - 1 sentence]
"These results mark a significant step toward scalable quantum computation."
```

**Abstract Dos and Don'ts**:
- ✅ Specific quantitative results
- ✅ Clear statement of what you did
- ✅ Accessible first sentence
- ❌ Citations (usually not allowed)
- ❌ Undefined acronyms
- ❌ "In this paper, we will..."
- ❌ Vague claims without support

**Title Refinement**:

Your title should:
- Capture the main finding
- Be searchable (include key terms)
- Be concise (10-15 words)
- Avoid jargon when possible

**Title Formulas**:
- "[Achievement] of [System/Method]"
- "[Finding] in [Context]"
- "[Technique] for [Application]"

**Examples**:
- "Fault-tolerant operation of a logical qubit in a surface code"
- "Exponential quantum speedup in simulating coupled oscillators"
- "Coherence-limited quantum gate operations using trapped ions"

**Afternoon (3 hours): First Draft Integration**

Compile all sections into a single document:
1. Assemble in order
2. Check section transitions
3. Ensure notation consistency
4. Verify all references exist

**Evening (1 hour): Complete Draft Review**

Read the entire draft once through:
- Mark major issues
- Note missing content
- Identify strongest/weakest sections

---

### Day 1771 (Sunday): Figure Finalization and Draft Completion

**Morning (3 hours): Figure Creation**

Transform preliminary figures into draft-quality versions:

**Figure Standards**:
- Resolution: 300 DPI minimum for raster images
- Format: Vector (PDF, EPS) when possible
- Fonts: Consistent, readable at intended size
- Colors: Colorblind-accessible palette
- Labels: Clear axis labels with units

**Multi-Panel Figure Guidelines**:
- Label panels (a), (b), (c)...
- Maintain consistent style across panels
- Include shared legends when possible

**Python Figure Template**:

```python
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.4, 2.5),  # Single column width
    'figure.dpi': 300,
})

fig, ax = plt.subplots()

# Your data and plotting here
x = np.linspace(0, 10, 100)
y = np.exp(-x/5) * np.cos(2*np.pi*x)

ax.plot(x, y, 'b-', linewidth=1.5, label='Data')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('Signal (a.u.)')
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig('figure1.pdf', bbox_inches='tight', dpi=300)
```

**Afternoon (3 hours): Caption Writing**

Write complete figure captions that:
- Start with a brief title/statement
- Describe what is shown
- Define all symbols and abbreviations
- Note key features
- Are self-contained

**Caption Template**:
> **Figure X. [Brief title].** (a) [Description of panel a]. (b) [Description of panel b]. [Key findings highlighted]. Parameters: [relevant values]. Error bars represent [statistical uncertainty type].

**Evening (1 hour): First Draft Completion**

Finalize your Week 253 deliverable:
- Complete draft of all sections
- All figures placed with captions
- References compiled (even if incomplete)
- Document formatted consistently

**First Draft Checklist**:
- [ ] Title present
- [ ] Abstract complete
- [ ] All major sections written
- [ ] Figures created and captioned
- [ ] References in place
- [ ] No major gaps in content
- [ ] Single continuous document

---

## Writing Resources

### Style Guides
- APS Style Manual: https://journals.aps.org/authors/axis-information-initiative-text-style
- Nature Physics Guide: https://www.nature.com/nphys/for-authors
- Elements of Style (Strunk & White)
- "The Science of Scientific Writing" (Gopen & Swan)

### LaTeX Templates
- REVTeX 4.2 for Physical Review journals
- Nature template for Nature family journals
- General article class with custom preamble

### Reference Management
- Zotero (free, open-source)
- Mendeley (free)
- BibTeX/BibLaTeX for LaTeX documents

---

## Common First Draft Issues (and Solutions)

| Issue | Solution |
|-------|----------|
| "I don't know where to start" | Start with Results—it's the most concrete |
| "My intro is too long" | Move background to a separate section |
| "Methods are unclear" | Have a colleague read and identify confusion |
| "Results feel scattered" | Organize around figures, not chronology |
| "Discussion is thin" | Address: meaning, limitations, implications, future |
| "It's not perfect" | That's okay—this is a first draft! |

---

## Reflection Questions

At the end of Week 253, consider:

1. What is the single most important message of your paper?
2. Which section was hardest to write? Why?
3. What feedback would be most valuable at this stage?
4. What are you proudest of in this draft?
5. What areas need the most revision?

Complete the Week 253 Reflection template to document your progress.
