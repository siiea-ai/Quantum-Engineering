# Guide: The Anatomy of a Research Proposal

## Introduction

This guide provides a comprehensive overview of research proposal structure, focusing on NSF and DOE formats commonly used in quantum science and engineering. Understanding proposal anatomy is essential for crafting competitive applications.

---

## Part 1: Universal Proposal Elements

### The Proposal Ecosystem

Every research proposal, regardless of agency, contains these core elements:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RESEARCH PROPOSAL                           │
├─────────────────────────────────────────────────────────────────┤
│  1. WHAT you propose (Specific Aims/Objectives)                 │
│  2. WHY it matters (Significance/Background)                    │
│  3. HOW you'll do it (Methodology/Technical Approach)           │
│  4. WHEN you'll do it (Timeline/Milestones)                     │
│  5. WHO will do it (Personnel/Qualifications)                   │
│  6. HOW MUCH it costs (Budget/Justification)                    │
│  7. IMPACT beyond science (Broader Impacts/Societal Benefits)   │
└─────────────────────────────────────────────────────────────────┘
```

### The Inverted Pyramid of Importance

Reviewers read proposals with decreasing attention:

```
    ┌────────────────────┐
    │      TITLE         │  ← 5 seconds of attention
    ├────────────────────┤
    │     ABSTRACT       │  ← 1 minute
    ├────────────────────┤
    │   SPECIFIC AIMS    │  ← 5 minutes (CRITICAL)
    ├────────────────────┤
    │    BACKGROUND      │  ← 10 minutes
    ├────────────────────┤
    │   METHODOLOGY      │  ← 15 minutes
    ├────────────────────┤
    │   TIMELINE/BUDGET  │  ← 5 minutes
    └────────────────────┘
```

**Key insight:** Your Specific Aims page determines whether reviewers engage with the rest.

---

## Part 2: The Specific Aims Page

### The Most Important Page in Science

The Specific Aims page (1 page maximum) is where proposals are won or lost. It must accomplish:

1. **Hook the reader** - Establish urgency and importance
2. **Identify the gap** - What problem remains unsolved?
3. **Present your solution** - High-level approach
4. **State specific aims** - Concrete, measurable objectives
5. **Promise impact** - Why success matters

### Specific Aims Formula

```
PARAGRAPH 1: THE HOOK (3-4 sentences)
├── Sentence 1: Big picture importance
├── Sentence 2: Current state of the field
├── Sentence 3: The critical problem/opportunity
└── Sentence 4: Transition to your approach

PARAGRAPH 2: THE GAP (3-4 sentences)
├── What we don't know
├── Why previous approaches fall short
├── What's needed to move forward
└── Your unique insight/approach

PARAGRAPH 3: THE SOLUTION (3-4 sentences)
├── Your central hypothesis
├── Brief overview of approach
├── Why you're positioned to succeed
└── Transition to aims

SPECIFIC AIMS (1-2 sentences each):
├── Aim 1: [Action verb] [specific objective]
├── Aim 2: [Action verb] [specific objective]
├── Aim 3: [Action verb] [specific objective]
└── (Optional) Aim 4: [Action verb] [specific objective]

PARAGRAPH 4: IMPACT (2-3 sentences)
├── Expected outcomes
├── How this advances the field
└── Broader implications
```

### Example: Quantum Error Correction Proposal

**Title:** Novel Surface Code Architectures for Fault-Tolerant Quantum Computing

**Specific Aims Page:**

> Quantum computers promise exponential speedups for cryptography, optimization, and materials simulation, yet current devices are fundamentally limited by errors. Despite remarkable progress in qubit coherence and gate fidelity, no quantum computer has achieved the error rates needed for useful computation. The path to fault tolerance requires not just better qubits, but fundamentally new error correction architectures that can tolerate realistic noise.
>
> Current surface code implementations require millions of physical qubits to encode thousands of logical qubits, making near-term fault tolerance impractical. Standard approaches assume symmetric noise models that poorly match real superconducting and trapped-ion hardware. Furthermore, syndrome extraction circuits themselves introduce errors that propagate through the code. A new approach is needed that exploits hardware-specific noise structures and minimizes syndrome extraction overhead.
>
> We hypothesize that tailored surface code variants optimized for hardware-specific noise can reduce the physical-to-logical qubit overhead by 10x while maintaining equivalent logical error rates. Our approach combines machine learning-driven code optimization with novel syndrome extraction protocols that exploit biased noise. Preliminary simulations suggest 5x overhead reduction is achievable with current superconducting hardware parameters.
>
> **Aim 1:** Develop asymmetric surface code variants optimized for biased noise, targeting 3x reduction in qubit overhead for bias ratios >10.
>
> **Aim 2:** Design and simulate novel syndrome extraction circuits that reduce error propagation by 50% compared to standard approaches.
>
> **Aim 3:** Validate code performance on cloud-accessible quantum hardware (IBM, IonQ) and characterize practical overhead requirements.
>
> This research will establish a practical pathway to fault-tolerant quantum computing achievable within the next decade. Success will directly impact the quantum computing roadmaps of major technology companies and national laboratories, accelerating the timeline for quantum advantage in chemistry and optimization.

---

## Part 3: Background and Significance

### Purpose

This section establishes:
- Your understanding of the field
- The importance of the problem
- Why current solutions are insufficient
- How your research will advance knowledge

### Structure

```
SECTION 1: FIELD OVERVIEW (1-2 paragraphs)
├── Broad importance of the research area
├── Key advances in the past 5-10 years
└── Current state of the art

SECTION 2: THE PROBLEM (2-3 paragraphs)
├── Specific challenge your research addresses
├── Why this problem is important
├── What has been tried before
└── Why those approaches are insufficient

SECTION 3: YOUR APPROACH (1-2 paragraphs)
├── How your approach differs
├── Preliminary results (if any)
└── Why this approach will succeed

SECTION 4: SIGNIFICANCE (1 paragraph)
├── What will change if you succeed
├── Impact on field/society
└── Connection to broader goals
```

### Writing Tips

**Do:**
- Cite extensively (30-50 references typical)
- Be generous to prior work while identifying gaps
- Use figures to illustrate key concepts
- Connect to reviewer's likely expertise

**Don't:**
- Simply review the field without critique
- Dismiss prior work unfairly
- Assume reviewers are specialists
- Bury your innovation

---

## Part 4: Research Plan / Methodology

### The Heart of the Proposal

This section demonstrates you can actually do the research. It must be:
- **Specific** - Exact methods, not vague statements
- **Justified** - Why this approach over alternatives
- **Feasible** - Achievable within time and budget
- **Rigorous** - Controls, validation, statistical power

### Structure by Aim

```
AIM 1: [Restate aim title]
├── Rationale: Why this aim is necessary
├── Approach:
│   ├── Method A: Detailed protocol
│   ├── Method B: Detailed protocol
│   └── Method C: Detailed protocol
├── Expected Outcomes: What you'll learn
├── Potential Pitfalls: What could go wrong
└── Alternative Approaches: Backup plans

AIM 2: [Repeat structure]
...

AIM 3: [Repeat structure]
...
```

### Key Elements

#### Experimental Design
- Clear hypotheses with predictions
- Controls (positive, negative)
- Sample sizes and power calculations
- Blinding and randomization (where applicable)

#### Computational Methods
- Algorithms and software
- Hardware requirements
- Validation against known results
- Error/uncertainty quantification

#### Theoretical Framework
- Mathematical foundations
- Assumptions and limitations
- Connection to experiments
- Falsifiable predictions

### Pitfall/Alternative Section

This is CRITICAL. Reviewers want to know you've thought through what could go wrong:

```
POTENTIAL PITFALL: [Specific problem]
├── Likelihood: Low/Medium/High
├── Impact: How this would affect the project
└── Mitigation: How you'll address it

ALTERNATIVE APPROACH: [Backup plan]
├── When to pivot: Decision criteria
├── How it addresses the pitfall
└── Tradeoffs involved
```

---

## Part 5: Timeline and Milestones

### Demonstrating Feasibility

A clear timeline shows:
- You understand the project scope
- Tasks are appropriately sequenced
- Milestones are realistic
- There's slack for unexpected delays

### Standard Format

```
YEAR 1:
├── Q1: Task 1.1, Task 1.2
├── Q2: Task 1.3, Task 2.1
├── Q3: Task 2.2, Task 2.3
├── Q4: Task 2.4, Milestone: Aim 1 complete
└── Deliverable: Publication A

YEAR 2:
├── Q1: Task 3.1, Task 3.2
...
```

### Gantt Chart Elements

| Task | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
|------|----|----|----|----|----|----|----|----|
| Literature review | ██ | | | | | | | |
| Method development | ██ | ██ | ██ | | | | | |
| Data collection | | ██ | ██ | ██ | ██ | | | |
| Analysis | | | ██ | ██ | ██ | ██ | | |
| Publication prep | | | | | ██ | ██ | ██ | |
| Thesis writing | | | | | | ██ | ██ | ██ |

### Milestone Definition

Good milestones are:
- **Measurable** - Clear success criteria
- **Meaningful** - Represent real progress
- **Feasible** - Achievable in stated time
- **Sequential** - Build on each other

Example milestones for quantum proposal:
- M1 (Month 6): Complete simulation framework validated against 5-qubit codes
- M2 (Month 12): Demonstrate 3x overhead reduction in simulation
- M3 (Month 18): First hardware demonstration on 27-qubit device
- M4 (Month 24): Publication of validated code architecture

---

## Part 6: Broader Impacts

### NSF's Unique Requirement

NSF explicitly requires demonstration of societal benefit beyond scientific merit. This is reviewed with equal weight to intellectual merit.

### Categories of Broader Impact

1. **Educational Integration**
   - Undergraduate research involvement
   - Curriculum development
   - K-12 outreach
   - Public lectures

2. **Workforce Development**
   - Graduate student training
   - Postdoc mentoring
   - Industry partnerships
   - Skills for national needs

3. **Broadening Participation**
   - Underrepresented group involvement
   - First-generation student support
   - Accessible design
   - Community engagement

4. **Infrastructure Development**
   - Shared facilities
   - Open-source software
   - Databases and resources
   - Equipment availability

5. **Societal Benefit**
   - Economic development
   - National security
   - Health/environment
   - Policy implications

### Writing Effective Broader Impacts

**Do:**
- Be specific (not "we will involve undergraduates")
- Connect to your research naturally
- Show institutional support
- Describe metrics for success

**Don't:**
- Treat as an afterthought
- Make vague promises
- Ignore your institution's resources
- Underestimate reviewer expertise

### Example: Quantum Computing Broader Impacts

> **Educational Integration:** This project directly supports two PhD students who will receive training in quantum error correction, circuit optimization, and experimental validation. Both students will spend summer internships at partner national laboratories (confirmed letters attached). We will develop a new graduate course module on practical quantum error correction, with materials released publicly.
>
> **Broadening Participation:** We actively recruit from minority-serving institutions through the established QISE-NET partnership. The PI's lab currently includes 40% women and 30% underrepresented minorities. We will host two undergraduate summer researchers through the NSF REU program, targeting students from primarily undergraduate institutions.
>
> **Public Engagement:** All code developed will be released open-source on GitHub. We will create a public-facing website explaining quantum error correction for non-experts, and the PI will give at least two public lectures per year at local libraries and science museums.

---

## Part 7: Agency-Specific Considerations

### NSF vs. DOE: Key Differences

| Aspect | NSF | DOE Office of Science |
|--------|-----|----------------------|
| Primary mission | Fundamental science | National lab partnership |
| Page limit (project desc.) | 15 pages | Varies (often 15-25) |
| Broader Impacts | Explicit section | Less emphasized |
| Budget detail | Separate document | Often more detailed |
| Review process | Panel + ad hoc | Panel-centric |
| Collaboration | Encouraged | Often required |
| National labs | Optional | Often required |

### NSF-Specific Tips
- Broader Impacts equally weighted with Intellectual Merit
- Include data management plan
- Strong letters of collaboration if relevant
- Consider solicitation-specific requirements

### DOE-Specific Tips
- Statement of Work with explicit deliverables
- National lab letters often expected
- Technical volumes require more detail
- Mission relevance (energy, security) important

---

## Part 8: The Review Process

### Understanding Reviewers

**Who reviews proposals:**
- Active researchers in related fields
- 3-5 reviewers per proposal (typical)
- Mix of specialists and non-specialists
- Volunteer service (limited time)

**What reviewers want:**
- Clear, compelling narrative
- Evidence of feasibility
- Appropriate scope
- Novelty and significance
- Well-prepared team

### The Panel Process

```
INDIVIDUAL REVIEW (2-3 hours per proposal)
↓
WRITTEN CRITIQUES (submitted before meeting)
↓
PANEL DISCUSSION (15-30 minutes per proposal)
↓
RANKING/SCORING
↓
PROGRAM OFFICER DECISION
↓
AWARD/DECLINE NOTIFICATION
```

### Common Reviewer Concerns

1. **"Not novel enough"** - Need clearer differentiation from prior work
2. **"Too ambitious"** - Reduce scope or extend timeline
3. **"Feasibility concerns"** - Add more preliminary data or method detail
4. **"Weak broader impacts"** - Strengthen societal benefit section
5. **"Team lacks expertise"** - Add collaborator or training plan

---

## Part 9: Final Checklist

### Before Submission

**Format:**
- [ ] Page limits met (with margin)
- [ ] Correct font and margins
- [ ] All required sections present
- [ ] Figures legible in print
- [ ] References complete

**Content:**
- [ ] Aims are specific and measurable
- [ ] Methods detailed enough to replicate
- [ ] Timeline realistic
- [ ] Broader impacts meaningful
- [ ] No hyperbole or overclaiming

**Strategy:**
- [ ] Right program/solicitation chosen
- [ ] Aligns with agency priorities
- [ ] Team appropriate for scope
- [ ] Budget justified

---

## Summary

The best proposals tell a compelling story:

1. **Hook** - Why should anyone care?
2. **Gap** - What don't we know?
3. **Solution** - What's your approach?
4. **Plan** - How will you do it?
5. **Team** - Why are you the right people?
6. **Impact** - What changes if you succeed?

Master these elements, and your proposals will stand out.

---

*"A good proposal is simultaneously a sales pitch, a technical document, a project plan, and a story. Learn to write all four at once."*
