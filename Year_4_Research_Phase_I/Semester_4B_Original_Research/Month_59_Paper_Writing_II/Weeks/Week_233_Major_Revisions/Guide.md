# Major Revision Methodology Guide

## Introduction

This guide provides systematic strategies for transforming a first draft into a structurally sound manuscript. Major revision is the most impactful phase of the writing process—addressing fundamental issues here prevents wasted effort on polishing content that will ultimately be cut or reorganized.

## Part I: The Philosophy of Major Revision

### Why Structure Matters Most

Consider the physics analogy: just as a quantum system's behavior depends on its Hamiltonian structure rather than its basis representation, a paper's effectiveness depends on its logical architecture rather than its stylistic surface.

**The Revision Hierarchy:**

```
Level 1: Conceptual (Does the argument make sense?)
    ↓
Level 2: Structural (Are sections in optimal order?)
    ↓
Level 3: Paragraph (Does each paragraph contribute?)
    ↓
Level 4: Sentence (Is each sentence clear?)
    ↓
Level 5: Word (Is language precise?)
```

**Major revision addresses Levels 1-3.** Tackling lower levels first is inefficient—you may perfect sentences that belong in different sections or paragraphs that shouldn't exist at all.

### The Reader's Mental Model

Your paper must build a mental model in the reader's mind. Consider:

$$\text{Reader Understanding} = f(\text{Prior Knowledge}, \text{Information Sequence}, \text{Connections Made})$$

Readers process information sequentially. They cannot "randomly access" later sections. Your structure must respect this constraint.

**Mental Model Building Stages:**

| Stage | Reader State | Your Task |
|-------|-------------|-----------|
| 1. Context | "Why should I care?" | Establish relevance |
| 2. Problem | "What's missing?" | Create intellectual tension |
| 3. Approach | "How will you address this?" | Outline your method |
| 4. Execution | "What did you do/find?" | Present evidence |
| 5. Resolution | "What does it mean?" | Provide interpretation |
| 6. Synthesis | "What's next?" | Connect to broader context |

## Part II: Structural Audit Techniques

### Technique 1: The Reverse Outline

The reverse outline reveals your paper's actual structure (which may differ from your intended structure).

**Procedure:**

1. Print your manuscript or view in read-only mode
2. For each paragraph, write ONE sentence capturing its main point
3. List all sentences in order
4. Evaluate the resulting outline

**Diagnostic Questions:**

- Does this outline tell a coherent story?
- Are there logical gaps between consecutive points?
- Are any points redundant?
- Does the order make sense?
- Could a reader follow this progression?

**Example Reverse Outline (with problems):**

```
1. Quantum computing is important
2. Superconducting qubits are leading platform
3. Our lab uses transmon architecture
4. Gate fidelity is limited by decoherence  ← Gap: jumps to problem
5. We present results of dynamical decoupling
6. Previous work by Smith et al. showed...  ← Order: background too late
7. Our method improves on Smith
8. Figure 3 shows fidelity data
9. The data confirms our hypothesis
10. Future work will explore...  ← Missing: interpretation
```

**Revised Outline:**

```
1. Quantum computing is important
2. Superconducting qubits are leading platform
3. Gate fidelity is limited by decoherence
4. Previous work by Smith et al. showed... (early context)
5. Smith's approach had limitations
6. We present dynamical decoupling method
7. Our method addresses Smith's limitations
8. Experimental implementation uses transmon
9. Figure 3 shows fidelity data
10. Data interpretation and comparison
11. Implications for fault-tolerant QC
12. Future work will explore...
```

### Technique 2: The Section Purpose Test

Every section must have a clear, unique purpose. If you cannot articulate a section's purpose in one sentence, the section may be unfocused.

**Section Purpose Template:**

```markdown
## Section: [Name]

**Purpose (one sentence):** _______________

**What readers learn:** _______________

**Why this belongs here (not elsewhere):** _______________

**Prerequisite knowledge (what must come before):** _______________

**Enables (what this makes possible later):** _______________
```

**Apply to every section of your paper.**

### Technique 3: The Claim-Evidence Map

Scientific papers make claims supported by evidence. Map these explicitly.

**For each claim in your paper:**

| Claim | Location | Type of Evidence | Specific Reference | Strength |
|-------|----------|-----------------|-------------------|----------|
| "Fidelity exceeds 99%" | Results §3.2 | Experimental data | Fig. 3, Table II | Strong |
| "Method is scalable" | Discussion §4.1 | Theoretical argument | Eq. 12 | Moderate |
| "Outperforms prior work" | Discussion §4.2 | Comparison | Table III | Strong |

**Evidence Types:**

1. **Experimental data**: Direct measurements with uncertainties
2. **Theoretical derivation**: Mathematical proof or analysis
3. **Numerical simulation**: Computational validation
4. **Literature reference**: Established prior results
5. **Logical argument**: Reasoning from accepted premises

**Claim Strength Assessment:**

- **Strong**: Multiple independent evidence types, quantitative, reproducible
- **Moderate**: Single evidence type, some caveats
- **Weak**: Speculative, qualitative, limited support
- **Unsupported**: No evidence provided (fix immediately)

### Technique 4: The Transition Audit

Transitions between sections and paragraphs must guide readers. Audit every transition point.

**Transition Types:**

| Type | Signal Words | Purpose |
|------|-------------|---------|
| Addition | Furthermore, Additionally | Add related information |
| Contrast | However, Nevertheless | Introduce opposing view |
| Cause/Effect | Therefore, Consequently | Show logical relationship |
| Sequence | First, Subsequently, Finally | Indicate order |
| Summary | In summary, To conclude | Wrap up section |

**Transition Diagnostic:**

At each section boundary, answer:
1. What question should readers have at this point?
2. Does the next section answer that question?
3. Is the connection explicit or must readers infer it?

**Example Problem:**

```
[End of Methods section]
...The samples were measured at 10 mK in a dilution refrigerator.

[Start of Results section]
Figure 2 shows the measured coherence times.
```

**Problem:** No transition. Readers don't know what to expect.

**Improved:**

```
[End of Methods section]
...The samples were measured at 10 mK in a dilution refrigerator.

[Start of Results section]
Using this experimental setup, we systematically characterized the qubit
coherence properties. Figure 2 shows the measured coherence times...
```

## Part III: Common Structural Problems

### Problem 1: Buried Lead

**Symptoms:**
- Key results appear late in paper
- Abstract mentions findings not in introduction
- Readers surprised by conclusions

**Solution:** Front-load importance

$$\text{Information Value} \propto \frac{1}{\text{Position in Document}}$$

Put your most important findings in:
1. Title (if possible)
2. Abstract (definitely)
3. Introduction end (contribution statement)
4. Results beginning

### Problem 2: Scope Creep

**Symptoms:**
- Paper tries to do too much
- Disconnected sections
- Reviewers ask "What is this paper about?"

**Solution:** Define core contribution and stick to it

Ask: "If I could only tell readers ONE thing, what would it be?"

Everything else either supports this core or belongs elsewhere (supplement, future paper).

### Problem 3: Missing Motivation

**Symptoms:**
- Readers ask "So what?"
- Introduction jumps to approach without establishing need
- Conclusions seem anticlimactic

**Solution:** Explicitly state the gap your work fills

**The Gap Statement Formula:**

```
Despite [what's been achieved], [specific limitation] remains.
This limits [important application]. Here we address this by...
```

### Problem 4: Claim-Data Mismatch

**Symptoms:**
- Claims stronger than evidence supports
- Reviewers challenge conclusions
- Caveats buried or absent

**Solution:** Audit every claim against evidence

**Claim Calibration Scale:**

| Evidence Level | Appropriate Claim Language |
|---------------|---------------------------|
| Definitive proof | "We demonstrate that..." |
| Strong evidence | "Our results show..." |
| Suggestive evidence | "Our data suggest..." |
| Preliminary | "These results are consistent with..." |
| Speculation | "We speculate that..." |

### Problem 5: Circular Structure

**Symptoms:**
- Introduction and Conclusions are nearly identical
- Paper seems to "go nowhere"
- No sense of intellectual progress

**Solution:** Ensure each section advances understanding

**The Knowledge Progression:**

```
Introduction: "Here's what was known and what we'll show"
                            ↓
Results: "Here's what we found"
                            ↓
Discussion: "Here's what it means"
                            ↓
Conclusions: "Here's how this changes the field"
```

Each section should add NEW information or perspective.

## Part IV: Incorporating Feedback

### Processing Collaborator Comments

When receiving feedback, avoid two traps:
1. **Defensive rejection**: Dismissing valid criticism
2. **Uncritical acceptance**: Accepting all suggestions without evaluation

**Feedback Triage Matrix:**

|  | Agree with Substance | Disagree with Substance |
|--|---------------------|------------------------|
| **Easy to implement** | Do immediately | Discuss with commenter |
| **Hard to implement** | Plan carefully | Evaluate cost/benefit |

### Categorizing Feedback

**Category 1: Must Fix**
- Factual errors
- Missing essential content
- Logical fallacies
- Unsupported claims

**Category 2: Should Address**
- Clarity issues
- Structural suggestions
- Missing context

**Category 3: Consider**
- Style preferences
- Alternative interpretations
- Scope suggestions

**Category 4: Document but Decline**
- Out of scope
- Conflicts with other feedback
- Would compromise paper quality

### Creating a Response Log

Track your responses to all feedback:

```markdown
| Comment | Source | Category | Decision | Implementation | Status |
|---------|--------|----------|----------|----------------|--------|
| "Fig 2 unclear" | Advisor | Must fix | Revise | New color scheme | Done |
| "Add section on X" | Collab A | Consider | Decline | Out of scope | Noted |
```

## Part V: Revision Workflow

### Day-by-Day Structure

**Day 1: Structure Audit**
- Create reverse outline
- Identify structural problems
- Prioritize issues

**Day 2: Logic Flow**
- Map argument progression
- Fill logical gaps
- Verify claim-evidence alignment

**Day 3: Evidence Alignment**
- Audit every claim
- Strengthen weak claims or qualify language
- Remove unsupported claims

**Day 4: Reorganization**
- Implement structural changes
- Move sections as needed
- Create new transitions

**Day 5: Feedback Integration**
- Review all collaborator comments
- Implement required changes
- Document decisions

**Day 6: Coherence Check**
- Read manuscript straight through
- Check transitions
- Verify unified voice

**Day 7: Consolidation**
- Create clean new draft
- Update version control
- Prepare for next phase

### Version Control Best Practices

**Branch Strategy:**

```
main
  └── revision-v2
        └── revision-v2-structure
        └── revision-v2-methods
```

**Commit Message Convention:**

```
[STRUCTURE] Reorganize Results into chronological order
[CONTENT] Add missing error analysis to Section 3
[FEEDBACK] Address advisor comments on introduction
[FIX] Correct equation numbering
```

**LaTeX Diff Workflow:**

```bash
# Generate visual diff between versions
latexdiff manuscript-v1.tex manuscript-v2.tex > diff.tex
pdflatex diff.tex

# Review changes in PDF with additions/deletions highlighted
```

## Part VI: Quality Checks

### Pre-Polish Checklist

Before moving to line editing (Week 235), verify:

**Structure**
- [ ] Clear logical progression from start to end
- [ ] Each section has unique, identifiable purpose
- [ ] Transitions connect all sections
- [ ] No orphaned paragraphs

**Argumentation**
- [ ] Central claim clearly stated
- [ ] All claims supported by evidence
- [ ] No overclaiming beyond data
- [ ] Limitations acknowledged

**Content**
- [ ] No significant gaps in narrative
- [ ] No redundant sections
- [ ] Scope is focused and consistent
- [ ] All necessary background included

**Feedback**
- [ ] All must-fix items addressed
- [ ] Decline decisions documented
- [ ] Collaborators notified of major changes

### Self-Test Questions

Ask yourself:
1. Can I summarize this paper in 30 seconds?
2. Would a reviewer understand the contribution from the abstract?
3. Does the introduction make readers want to continue?
4. Does each figure earn its place?
5. Are conclusions justified by results?
6. Would I cite this paper?

## Summary

Major revision transforms a rough draft into a structurally sound manuscript. By addressing big-picture issues first, you ensure that subsequent polishing efforts are not wasted.

### Key Principles

1. **Structure before style**: Fix organization before prose
2. **Reader-centered thinking**: Build their mental model systematically
3. **Evidence-based claims**: Match language to support level
4. **Explicit connections**: Never leave readers to infer transitions
5. **Iterative feedback**: Incorporate input systematically

### Next Steps

With structural revision complete, proceed to Week 234 for figure refinement. Your solid structural foundation will make figure placement and caption writing more straightforward.

---

*Major Revision Guide | Week 233 | Month 59 | Year 4*
