# Guide: Gap Analysis and Problem Identification Methodology

## Introduction

Gap analysis is the art and science of finding the spaces between what is known and what needs to be known. For a researcher, these gaps represent opportunities—places where your work can make a difference. This guide provides a comprehensive methodology for identifying, evaluating, and prioritizing research gaps.

The goal is not just to find gaps, but to find *good* gaps: problems that are important, tractable, and matched to your capabilities.

---

## Part 1: Understanding Research Gaps

### 1.1 What is a Research Gap?

A research gap is a deficiency in existing knowledge that:
- Limits our understanding of a phenomenon
- Prevents progress on practical problems
- Creates uncertainty in decision-making
- Leaves interesting questions unanswered

**Not every gap is worth filling.** Some gaps exist because:
- The question isn't interesting
- The answer doesn't matter
- The problem is too hard
- Resources don't exist to pursue it

Your task is to distinguish valuable gaps from dead ends.

### 1.2 Types of Research Gaps

| Gap Type | Definition | Example in QC | How to Identify |
|----------|------------|---------------|-----------------|
| **Knowledge Gap** | Unknown fact or relationship | Optimal threshold for code X | "It is unknown whether..." |
| **Method Gap** | No technique exists | Efficient decoder for code Y | "No method currently..." |
| **Theory Gap** | No framework to understand | Theory of realistic noise | "There is no theory of..." |
| **Empirical Gap** | Hasn't been measured/tested | QEC on hardware Z | "No experiment has..." |
| **Application Gap** | Not applied to domain | Algorithm for problem W | "Has not been applied to..." |
| **Integration Gap** | Components not combined | End-to-end quantum stack | "Has not been demonstrated together..." |
| **Scale Gap** | Works small, not large | 100+ qubit QEC | "Scaling to larger systems..." |
| **Contradiction Gap** | Results conflict | Disagreement on metric M | Papers A and B disagree on... |

### 1.3 The Gap Quality Spectrum

```
Poor Gaps                                              Good Gaps
|------------------------------------------------------>

Vague          Obvious next step       Important missing piece
Too hard       Incremental advance     Breakthrough opportunity
No one cares   Community interest      Field-changing potential
No resources   Resources available     Ideal resource match
```

---

## Part 2: Systematic Gap Identification

### 2.1 The Literature Mining Approach

**Step 1: Identify Target Papers**

Select 15-20 papers from your area:
- 5 recent review articles
- 5 highly-cited foundational papers
- 5-10 recent (6-12 month) papers from leading groups

**Step 2: Extract Gap Signals**

Read each paper looking specifically for:

*Explicit Signals:*
- "Future work" sections
- "Limitations" sections
- "Open problems" sections
- "Challenges" subsections
- "Outlook" or "Perspective" sections

*Language Patterns:*
- "It remains unknown..."
- "Further study is needed..."
- "An open question is..."
- "It would be interesting to..."
- "We leave for future work..."
- "A limitation of our approach..."
- "Ideally, one would..."
- "To the best of our knowledge, no one has..."

**Step 3: Document Each Gap**

For each identified gap, record:
- Source paper(s)
- Exact quote or paraphrase
- Your interpretation
- Preliminary importance assessment
- Category/type

### 2.2 The Cross-Paper Analysis Approach

**Step 1: Create a Result Matrix**

For key results in your area, track across papers:

| Result/Metric | Paper A | Paper B | Paper C | Paper D | Discrepancy? |
|---------------|---------|---------|---------|---------|--------------|
| | | | | | |

**Step 2: Identify Contradictions**

Look for:
- Different values for same metric
- Conflicting conclusions
- Incompatible assumptions
- Unreconciled perspectives

**Step 3: Investigate Contradictions**

For each contradiction:
- Is it real or apparent?
- What might explain the discrepancy?
- Would resolving it advance the field?
- Could this be a research contribution?

### 2.3 The Systematic Coverage Approach

**Step 1: Map the Conceptual Space**

Create a matrix of important dimensions in your area:

Example for Quantum Error Correction:

| | Surface Code | LDPC | Bosonic | Color |
|---|-------------|------|---------|-------|
| Theory | X | X | X | X |
| Simulation | X | X | O | X |
| Experiment | X | O | X | O |
| Decoder | X | ? | O | O |
| Resource Est. | X | O | O | O |

X = Well covered, O = Gap exists, ? = Partially addressed

**Step 2: Identify Sparse Cells**

Gaps appear as empty or sparse cells in your matrix.

**Step 3: Validate Gaps**

For each potential gap:
- Search explicitly for coverage
- Confirm it's genuinely underexplored
- Assess why (not interesting? too hard? requires missing resources?)

### 2.4 The Expert Consultation Approach

**Step 1: Prepare Questions**

Based on your initial gap identification, prepare questions:
- "What do you see as the biggest open problems in X?"
- "Why hasn't anyone solved Y?"
- "What would be needed to make progress on Z?"
- "Where do you think the field is going?"

**Step 2: Consult Multiple Experts**

Talk to:
- Potential advisors
- Postdocs in relevant groups
- Industry researchers
- Attendees at relevant conferences/seminars

**Step 3: Synthesize Perspectives**

Look for:
- Consensus on important problems
- Disagreements (potential contradiction gaps)
- Problems only experts know about
- Outdated gaps that have been solved

---

## Part 3: Evaluating Gap Quality

### 3.1 The Importance Assessment

**Questions to Ask:**

1. **Who cares?**
   - Would solving this interest theorists? Experimentalists? Industry?
   - How many papers cite work in this area?
   - Is this mentioned at conferences?

2. **What does it enable?**
   - Does this unblock other research?
   - Does it have practical applications?
   - Does it answer a foundational question?

3. **Is the field ready?**
   - Are the prerequisites in place?
   - Is there growing interest in this direction?
   - Has related work been done recently?

**Importance Scoring:**

| Score | Description |
|-------|-------------|
| 5 | Field-changing if solved; many papers would cite |
| 4 | Important advance; multiple groups would care |
| 3 | Useful contribution; good paper potential |
| 2 | Incremental advance; limited interest |
| 1 | Minimal impact; few would notice |

### 3.2 The Tractability Assessment

**Questions to Ask:**

1. **Is it well-defined?**
   - Can you write a clear problem statement?
   - Would you know if you solved it?
   - Are success criteria objective?

2. **What approaches might work?**
   - Are there relevant existing methods?
   - What tools would be needed?
   - Can you imagine a path to solution?

3. **Why is it still open?**
   - Is it hard, or just unexplored?
   - What have others tried?
   - What would be different about your approach?

4. **Is the scope appropriate?**
   - Can meaningful progress be made in 2-4 years?
   - Can it be decomposed into subproblems?
   - Are there intermediate results?

**Tractability Scoring:**

| Score | Description |
|-------|-------------|
| 5 | Clear path to solution; high confidence in progress |
| 4 | Likely tractable; several viable approaches |
| 3 | Potentially tractable; some ideas for approach |
| 2 | Unclear; may be too hard or vague |
| 1 | Probably intractable; no clear approach |

### 3.3 The Resource Match Assessment

**Questions to Ask:**

1. **Skills match?**
   - Do I have the background?
   - What would I need to learn?
   - How long to become proficient?

2. **Computational resources?**
   - Do I need HPC? Quantum access?
   - Are these available?
   - What are the costs?

3. **Human resources?**
   - Is there advisor expertise?
   - Are collaborators available?
   - Is there a research community?

4. **Time resources?**
   - How long would this take?
   - Does it fit PhD timeline?
   - Are there publishable intermediate results?

**Resource Match Scoring:**

| Score | Description |
|-------|-------------|
| 5 | Perfect match; have or can easily acquire all resources |
| 4 | Good match; minor gaps to fill |
| 3 | Adequate match; some development needed |
| 2 | Weak match; significant gaps |
| 1 | Poor match; major barriers to entry |

### 3.4 Composite Scoring

Combine scores with weights:

```
Overall Score =
    (Importance × 0.30) +
    (Tractability × 0.30) +
    (Resource Match × 0.25) +
    (Personal Excitement × 0.15)
```

Adjust weights based on your priorities.

---

## Part 4: Creating Problem Profiles

### 4.1 Problem Profile Structure

For each top-ranked gap, create a detailed profile:

```markdown
## Problem: [Name]

### One-Sentence Description
[Clear statement of what needs to be solved]

### Background
- What is known:
- What is not known:
- Why this matters:

### Specific Questions/Goals
1.
2.
3.

### Potential Approaches
- Approach A:
- Approach B:
- Approach C:

### Prerequisites
- Knowledge:
- Tools:
- Resources:

### Risk Assessment
- Main risks:
- Mitigation strategies:
- Fallback positions:

### Success Criteria
- Minimum viable result:
- Strong result:
- Best case:

### Timeline Estimate
- Ramp-up: X months
- Main work: X months
- Writing/polishing: X months

### Competition Assessment
- Who else might work on this:
- Why I might succeed:
```

### 4.2 Quality Checks for Problem Profiles

Before finalizing a problem profile, verify:

- [ ] Problem statement is specific enough to be falsifiable
- [ ] At least two potential approaches identified
- [ ] Success criteria are concrete and measurable
- [ ] Timeline is realistic for PhD scope
- [ ] Resources are actually available (not just theoretically)
- [ ] Personal excitement is genuine
- [ ] Advisor would support this direction

---

## Part 5: Prioritization and Selection

### 5.1 Creating the Shortlist

From all identified and profiled gaps:

1. **Score all candidates** using the composite scoring system
2. **Rank by score** to create initial ordering
3. **Apply reality checks:**
   - Would advisor support this?
   - Does it fit program requirements?
   - Is the competition level acceptable?
4. **Diversify if possible:**
   - Include problems from different areas
   - Include different types (theory, experimental, applied)
   - Include different risk levels

### 5.2 The Final Shortlist

Aim for 5-7 problems on your shortlist:

| Rank | Problem | Area | Score | Risk Level | Backup? |
|------|---------|------|-------|------------|---------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

Include at least:
- 1-2 "safe" problems (higher tractability, lower risk)
- 1-2 "ambitious" problems (higher impact, higher risk)
- 1 backup option (definitely achievable)

### 5.3 Validation Steps

Before finalizing your shortlist:

1. **Sleep on it** - Does it still feel right the next day?
2. **Explain to someone** - Can you articulate why each problem matters?
3. **Imagine success** - Would you be proud of solving these?
4. **Imagine struggle** - Would you stay motivated when it's hard?
5. **Check with advisor** - Is there support for your top choices?

---

## Part 6: Common Pitfalls and How to Avoid Them

### 6.1 Gap Identification Pitfalls

**Pitfall 1: Finding gaps that aren't gaps**
- Problem: Literature search was incomplete
- Solution: Search extensively; ask experts; don't assume you've found everything

**Pitfall 2: Missing obvious gaps**
- Problem: Reading too narrowly
- Solution: Read adjacent areas; consider cross-disciplinary connections

**Pitfall 3: Over-valuing novelty**
- Problem: Chasing "new" over "important"
- Solution: Focus on impact, not just originality

### 6.2 Evaluation Pitfalls

**Pitfall 4: Overestimating tractability**
- Problem: Assuming problems are easier than they are
- Solution: Talk to people who've tried; be skeptical of your optimism

**Pitfall 5: Underestimating resource requirements**
- Problem: Not accounting for learning curves, failed attempts
- Solution: Add 50-100% buffer to time estimates; identify fallbacks

**Pitfall 6: Ignoring competition**
- Problem: Not considering who else might solve this first
- Solution: Assess competitive landscape; consider your unique advantages

### 6.3 Selection Pitfalls

**Pitfall 7: Choosing based on advisability rather than interest**
- Problem: Picking "should" over "want"
- Solution: Genuine excitement is essential; prioritize it

**Pitfall 8: Avoiding risk entirely**
- Problem: Choosing only safe problems limits potential impact
- Solution: Include at least one ambitious project in your portfolio

**Pitfall 9: Analysis paralysis**
- Problem: Can't decide; keep researching
- Solution: Set deadline; recognize any good problem is fine; action beats perfect choice

---

## Part 7: Integrating with Week 196

### 7.1 What Feeds Forward

Your Week 195 work provides:

1. **Ranked problem shortlist** - Candidates for your research direction
2. **Problem profiles** - Detailed understanding of top options
3. **Advisor feedback** - Validation and refinement from mentors
4. **Resource clarity** - Understanding of what's available

### 7.2 Preparing for Direction Selection

To maximize Week 196 effectiveness:

1. Complete all problem profiles thoroughly
2. Document advisor conversations carefully
3. Note any remaining uncertainties
4. Identify what additional information would help
5. Start drafting how you'd describe each problem to others

---

## Conclusion

Gap analysis transforms passive knowledge into active opportunity identification. The key principles are:

1. **Be systematic** - Use structured methods, not intuition alone
2. **Be thorough** - Look in multiple places using multiple approaches
3. **Be critical** - Not all gaps are worth filling; evaluate carefully
4. **Be realistic** - Match problems to available resources
5. **Be decisive** - Create a clear shortlist you can act on

By week's end, you should have a ranked list of 5-7 potential research problems, with clear profiles and preliminary advisor validation. This sets the stage for definitive direction selection in Week 196.

---

## Appendix: Checklists

### Gap Identification Checklist

- [ ] Reviewed 5+ review articles for explicit gaps
- [ ] Checked "future work" sections of 10+ papers
- [ ] Identified contradictions between papers
- [ ] Mapped conceptual space and found sparse areas
- [ ] Consulted at least one expert/advisor
- [ ] Searched to confirm gaps are actually open
- [ ] Categorized gaps by type

### Evaluation Checklist

- [ ] Scored importance (1-5) for all gaps
- [ ] Scored tractability (1-5) for promising gaps
- [ ] Scored resource match (1-5) for top candidates
- [ ] Created composite scores
- [ ] Validated with advisor/mentor

### Shortlist Checklist

- [ ] Selected 5-7 top problems
- [ ] Created detailed profile for each
- [ ] Included mix of safe and ambitious problems
- [ ] Verified advisor support for top choices
- [ ] Documented remaining uncertainties

---

**Next:** [Gap Sources Resource](./Resources/Gap_Sources.md) | [Gap Map Template](./Templates/Gap_Map.md)
