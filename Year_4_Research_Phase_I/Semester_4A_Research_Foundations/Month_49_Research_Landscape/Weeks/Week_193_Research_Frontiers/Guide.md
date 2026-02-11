# Guide: How to Survey Research Literature

## Introduction

Surveying research literature is a fundamental skill that distinguishes productive researchers from those who struggle. A systematic approach allows you to efficiently map a field, identify opportunities, and stay current without being overwhelmed. This guide provides a comprehensive methodology for literature surveying, specifically tailored to quantum computing research.

---

## Part 1: Principles of Effective Literature Review

### 1.1 The Purpose Hierarchy

Literature review serves multiple purposes, and your approach should match your goal:

| Purpose | Depth Required | Time Investment | Typical Outcome |
|---------|---------------|-----------------|-----------------|
| **Awareness** | Surface | Minutes per paper | Know it exists |
| **Understanding** | Moderate | 30-60 min per paper | Can explain to others |
| **Mastery** | Deep | Hours to days | Can reproduce/extend |
| **Integration** | Synthesis | Ongoing | Original insights |

**Key Insight:** Most papers only need awareness-level reading. Save deep reading for truly relevant work.

### 1.2 The Funnel Model

Think of literature survey as a funnel:

```
     ╭─────────────────────────────╮
     │   AWARENESS (thousands)     │  arXiv alerts, Google Scholar, Twitter
     ╰─────────────┬───────────────╯
                   │
           ╭───────▼───────╮
           │ TRIAGE (100s) │  Read abstracts, scan figures
           ╰───────┬───────╯
                   │
              ╭────▼────╮
              │READ (50)│  Understand main contribution
              ╰────┬────╯
                   │
               ╭───▼───╮
               │STUDY  │  Deep engagement, reproduce
               │(10-20)│
               ╰───────╯
```

### 1.3 Active vs. Passive Reading

**Passive Reading (avoid):**
- Reading papers like novels
- Highlighting without notes
- Reading without questions

**Active Reading (practice):**
- Reading with specific questions in mind
- Taking structured notes
- Connecting to existing knowledge
- Critically evaluating claims

---

## Part 2: Setting Up Your Literature Infrastructure

### 2.1 Reference Management System

**Recommended: Zotero**
- Free and open source
- Browser extension for one-click capture
- PDF annotation
- Citation generation
- Syncs across devices

**Setup Steps:**
1. Install Zotero desktop app
2. Install browser connector
3. Set up cloud sync (optional, consider privacy)
4. Create folder structure:
   ```
   My Library/
   ├── Quantum Algorithms/
   │   ├── VQE and Variational
   │   ├── Quantum Simulation
   │   ├── Quantum ML
   │   └── Complexity and Speedups
   ├── Quantum Hardware/
   │   ├── Superconducting
   │   ├── Trapped Ion
   │   ├── Neutral Atom
   │   ├── Photonic
   │   └── Other Platforms
   ├── Error Correction/
   │   ├── Surface Codes
   │   ├── LDPC Codes
   │   ├── Bosonic Codes
   │   └── Decoders
   ├── Applications/
   │   └── [subfolders by domain]
   └── Meta/
       ├── Reviews
       ├── Methodology
       └── To Read
   ```

### 2.2 Note-Taking System

**Recommended: Obsidian or Notion**

Create a system that connects:
- Paper notes
- Concept explanations
- Research ideas
- Questions and gaps

**Linking is Key:** The power comes from connections between notes, not individual notes.

### 2.3 Alert Systems

**arXiv Alerts:**
1. Go to arxiv.org → User Account → Email Alerts
2. Subscribe to categories: quant-ph, cond-mat.mes-hall
3. Set daily or weekly digests

**Google Scholar Alerts:**
1. Search for key topics or author names
2. Click "Create alert" at bottom of results
3. Set up alerts for:
   - Your specific research interests
   - Key researchers in your field
   - Important papers (for citations)

**ResearchRabbit:**
1. Create account at researchrabbit.ai
2. Add "seed" papers to a collection
3. Get recommendations for related papers

---

## Part 3: The Paper Processing Pipeline

### 3.1 Capture Phase

**Goal:** Collect potentially relevant papers without judgment

**Sources:**
- arXiv daily alerts (quant-ph, cond-mat)
- Google Scholar recommendations
- References from papers you're reading
- Conference proceedings (QIP, APS March Meeting)
- Social media/community channels

**Capture Workflow:**
1. See potentially interesting paper
2. One-click save to Zotero (captures metadata + PDF)
3. Add tag "unread" or "to-triage"
4. Move on (don't read yet!)

**Daily Time:** 15-20 minutes

### 3.2 Triage Phase

**Goal:** Quickly decide if paper deserves deeper reading

**Triage Protocol (5 minutes per paper max):**
1. Read title carefully
2. Read abstract completely
3. Look at figures/tables
4. Read conclusion
5. **Decision:** Discard / Queue for Reading / Read Now

**Triage Questions:**
- Is this relevant to my interests?
- Is this from a credible group?
- Does this claim something significant?
- Would understanding this help me?

**Efficiency Tip:** Be aggressive about discarding. You can always find papers again if needed.

### 3.3 Reading Phase

**Goal:** Understand the paper's contribution and evaluate its significance

**Reading Protocol (30-60 minutes per paper):**

1. **Context (5 min)**
   - What problem are they addressing?
   - Why is this problem important?
   - What did others do before?

2. **Contribution (10 min)**
   - What is the main claim/result?
   - What is the key insight or technique?
   - How does this advance the field?

3. **Evaluation (10 min)**
   - Are the claims well-supported?
   - What are the assumptions and limitations?
   - What questions remain?

4. **Implications (5 min)**
   - How does this connect to other work?
   - What does this enable next?
   - How does this affect my research?

5. **Notes (10 min)**
   - Complete paper annotation template
   - Extract key equations/figures
   - Record questions and ideas

### 3.4 Deep Study Phase (Selected Papers Only)

**Goal:** Master the paper well enough to reproduce or extend

**Deep Study Protocol (hours to days):**

1. **Reproduce derivations** - Work through all mathematical details
2. **Implement code** - If computational, implement key algorithms
3. **Verify claims** - Check if results make sense, spot-check calculations
4. **Find connections** - How does this connect to everything else?
5. **Identify extensions** - What natural next steps exist?

---

## Part 4: Mapping the Research Landscape

### 4.1 Identifying Major Areas

For quantum computing, the major research areas are:

**Theoretical:**
- Quantum Algorithms and Complexity
- Quantum Error Correction Theory
- Quantum Information Theory
- Quantum Foundations

**Experimental:**
- Quantum Hardware Development
- Quantum Control and Calibration
- Experimental QEC
- Quantum Sensing/Metrology

**Applied:**
- Quantum Software and Compilation
- Quantum Applications (chemistry, optimization, ML)
- Hybrid Quantum-Classical Systems

### 4.2 Finding Key Players

**Methods to Identify Leaders:**

1. **Citation Analysis**
   - Who has the most-cited papers in the area?
   - Whose work is referenced in every review?

2. **Institutional Affiliations**
   - Which universities have strong programs?
   - Which companies are publishing?

3. **Conference Presence**
   - Who gives invited talks at QIP?
   - Who organizes workshops and tutorials?

4. **Recent Impact**
   - Who has breakthrough papers in past 2-3 years?
   - Who is setting the agenda?

**Create a People Database:**
- Name and affiliation
- Research focus areas
- Key papers
- Connections to other researchers

### 4.3 Understanding Research Trajectories

**Questions to Answer:**
- Where has this field been?
- Where is it going?
- What drives progress?
- What are the barriers?

**Methods:**
1. Read historical reviews (understand origins)
2. Read recent reviews (understand current state)
3. Compare to identify trajectory
4. Look for inflection points (paradigm shifts)

### 4.4 Creating the Landscape Map

**Visual Representation Options:**

1. **Mind Map** - Central topic with branching subtopics
2. **Network Diagram** - Nodes for topics, edges for connections
3. **Matrix** - Areas vs. dimensions (theory/experiment, near/far term)
4. **Timeline** - Historical development with projections

**Recommended Approach: Hybrid**
- Create a hierarchical outline (text)
- Augment with a visual map (diagram)
- Include a key players table
- Add open problems list

---

## Part 5: Identifying Open Problems and Gaps

### 5.1 Where to Look for Open Problems

**Explicit Sources:**
- "Future work" sections of papers
- Review article conclusions
- Workshop reports and roadmaps
- Grant proposals (sometimes public)
- Researcher interviews/talks

**Implicit Sources:**
- Contradictions between papers
- Gaps in the literature
- Questions your reading raises
- Problems industry needs solved

### 5.2 Types of Research Gaps

| Gap Type | Description | Example |
|----------|-------------|---------|
| **Knowledge Gap** | Something unknown | Optimal threshold for code X |
| **Method Gap** | Approach doesn't exist | Decoder for code Y |
| **Application Gap** | Theory not applied | Code Z on hardware W |
| **Integration Gap** | Components not combined | End-to-end system |
| **Contradiction Gap** | Conflicting results | Papers disagree on metric |

### 5.3 Evaluating Problems

**Tractability:**
- Is the problem well-defined?
- Are the tools available?
- Is progress measurable?
- Can it be done in a PhD timeline?

**Impact:**
- Does solving this unblock other work?
- Do people care about the answer?
- Is the field moving this direction?

**Fit:**
- Does this match your skills?
- Are resources available?
- Is there advising expertise?
- Does it interest you?

---

## Part 6: Practical Tips and Common Pitfalls

### 6.1 Time Management

**Weekly Literature Time Budget:**
- Capture: 15 min/day = 1.5 hours/week
- Triage: 30 min/day = 3 hours/week
- Reading: 2-3 papers @ 45 min = 2 hours/week
- Deep study: As needed = 2-4 hours/week
- **Total:** 8-10 hours/week (during survey phase)

### 6.2 Common Pitfalls

**Pitfall 1: Reading Too Deeply Too Early**
- Symptom: Spending days on single papers
- Fix: Use triage ruthlessly, return later if needed

**Pitfall 2: Unfocused Collection**
- Symptom: 500 papers saved, none read
- Fix: Regular triage, aggressive deletion

**Pitfall 3: Missing the Forest for Trees**
- Symptom: Know details, miss big picture
- Fix: Start with reviews, build context first

**Pitfall 4: Ignoring Non-Traditional Sources**
- Symptom: Only read published papers
- Fix: Attend talks, follow blogs, engage online

**Pitfall 5: Not Taking Notes**
- Symptom: Re-reading same papers
- Fix: Systematic annotation, even if brief

### 6.3 Staying Current vs. Going Deep

**The Balance:**
- During survey month: 70% breadth, 30% depth
- During research: 30% breadth, 70% depth

**Sustainable Habits:**
- Daily: 15 min arXiv scan
- Weekly: 2-3 papers read properly
- Monthly: 1-2 papers deep study
- Quarterly: Survey new developments

---

## Part 7: Tools and Resources

### 7.1 Search and Discovery

| Tool | Best For | Tips |
|------|----------|------|
| **arXiv** | Latest preprints | Set daily alerts |
| **Google Scholar** | Citation tracking | Use "Cited by" feature |
| **Semantic Scholar** | AI-powered discovery | Good for related papers |
| **Connected Papers** | Visual exploration | Start from key paper |
| **ResearchRabbit** | Recommendations | Add papers to collections |

### 7.2 Reading and Annotation

| Tool | Features | Integration |
|------|----------|-------------|
| **Zotero** | Reference management | Browser, Word, LaTeX |
| **PDF Expert** | PDF annotation | Mac, iPad |
| **Notability** | Handwritten notes | iPad |
| **Hypothes.is** | Web annotation | Browser extension |

### 7.3 Note-Taking and Knowledge Management

| Tool | Style | Learning Curve |
|------|-------|----------------|
| **Obsidian** | Local markdown | Medium |
| **Notion** | All-in-one | Low |
| **Roam** | Outlining | High |
| **Apple Notes** | Simple | Very low |

### 7.4 Communication and Community

- **Twitter/X Academic** - Following researchers
- **Quantum Computing Stack Exchange** - Q&A
- **Reddit r/QuantumComputing** - Community discussion
- **Discord/Slack channels** - Real-time chat

---

## Conclusion

Effective literature surveying is a skill developed through practice. The key principles are:

1. **Be systematic** - Use consistent processes and tools
2. **Be selective** - You cannot read everything; filter aggressively
3. **Be active** - Read with questions, take notes, make connections
4. **Be persistent** - Build habits that sustain over years

The goal is not to read every paper but to build a mental model of the field that allows you to:
- Find relevant information quickly
- Identify where you can contribute
- Stay current without being overwhelmed
- Connect your work to the broader context

---

## Exercises

### Exercise 1: Triage Practice
Take 10 papers from today's arXiv quant-ph. Spend exactly 5 minutes per paper triaging. Note your decision and reasoning. Reflect on your efficiency.

### Exercise 2: Review Article Analysis
Select one major review article. Create a detailed annotation including:
- Main thesis
- Structure of argument
- Key citations (follow up 3)
- Open problems mentioned
- Your questions

### Exercise 3: Citation Network Exploration
Pick one influential paper. Use Connected Papers or citation tracking to map:
- 5 key predecessors
- 5 key papers citing it
- What trajectory does this reveal?

### Exercise 4: Area Comparison
Select two different research areas. Create a comparison table:
- Current state of knowledge
- Open problems
- Leading groups
- Your interest level

---

**Next:** [Key Reviews Resource](./Resources/Key_Reviews.md) | [Paper Annotation Template](./Templates/Paper_Annotation.md)
