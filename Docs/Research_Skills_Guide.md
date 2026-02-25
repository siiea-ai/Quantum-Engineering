# Research Skills Guide for Quantum Computing/Physics

A comprehensive guide for developing academic writing, research methodology, and professional communication skills for the research phase of quantum science education.

---

## Table of Contents

1. [Academic Writing Fundamentals](#1-academic-writing-fundamentals)
2. [Literature Review Methodology](#2-literature-review-methodology)
3. [Research Proposal Writing](#3-research-proposal-writing)
4. [Conference Presentations](#4-conference-presentations)
5. [Collaboration and Communication](#5-collaboration-and-communication)
6. [Recommended Tools and Resources](#6-recommended-tools-and-resources)
7. [Skill Development Timeline](#7-skill-development-timeline)

---

## 1. Academic Writing Fundamentals

### 1.1 Physics/Quantum Paper Structure

A typical physics or quantum computing research paper follows this structure:

#### Standard Paper Sections

| Section | Purpose | Typical Length |
|---------|---------|----------------|
| **Abstract** | Concise summary of motivation, methods, results, conclusions | 150-300 words |
| **Introduction** | Background, motivation, research question, paper overview | 1-2 pages |
| **Theory/Background** | Mathematical framework, relevant physics | 2-4 pages |
| **Methods** | Experimental/computational approach | 1-3 pages |
| **Results** | Data presentation, figures, analysis | 3-6 pages |
| **Discussion** | Interpretation, comparison with prior work | 1-2 pages |
| **Conclusion** | Summary, implications, future directions | 0.5-1 page |
| **Acknowledgments** | Funding, collaborators, facilities | Brief |
| **References** | Citations to prior work | Variable |
| **Appendices** | Supplementary derivations, data | As needed |

#### Key Principles for Quantum Papers

1. **Mathematical Rigor**: Present derivations clearly with proper notation
2. **Physical Intuition**: Connect formalism to physical understanding
3. **Reproducibility**: Provide sufficient detail for replication
4. **Context**: Situate work within the broader quantum computing landscape

### 1.2 LaTeX and Scientific Writing Tools

LaTeX is the standard document preparation system for physics and quantum computing papers.

#### Essential LaTeX Tools

**REVTeX**
- Developed by the American Physical Society (APS)
- Standard for Physical Review journals (PRA, PRB, PRX Quantum)
- Includes macros for common physics formatting

**Overleaf**
- Cloud-based LaTeX editor with real-time collaboration
- Over 17 million users globally
- Integrates with citation managers (Zotero, Mendeley)
- Provides templates for major journals

#### LaTeX Best Practices

```latex
% Good practices for physics papers

% 1. One sentence per line (easier version control)
This is the first sentence about quantum entanglement.
The second sentence continues the discussion.

% 2. Use proper equation environments (not $$...$$)
\begin{equation}
    |\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)
    \label{eq:superposition}
\end{equation}

% 3. Reference equations properly
As shown in Eq.~\eqref{eq:superposition}, the qubit exists in superposition.

% 4. Use BibTeX for references
\cite{nielsen2010quantum}

% 5. Organize large documents
\input{sections/introduction}
\input{sections/theory}
\input{sections/methods}
```

#### Version Control with Git

- Track all changes to your manuscript
- Collaborate with co-authors via GitHub/GitLab
- Never lose work; maintain complete history
- Enable easy comparison of manuscript versions

### 1.3 Citation Practices

#### Citation Management Tools

| Tool | Strengths | Best For |
|------|-----------|----------|
| **Zotero** | Free, accurate browser add-on, best bibliography generation | Individual researchers |
| **Mendeley** | Social networking features, PDF annotation | Collaborative teams |
| **BibTeX** | Native LaTeX integration, plain text format | LaTeX users |
| **EndNote** | Institutional support, comprehensive features | Large institutions |

#### Best Practices

1. **Cite primary sources**: Reference original papers, not review articles citing them
2. **Be comprehensive**: Include all relevant prior work
3. **Be current**: Include recent developments (past 2-3 years)
4. **Be accurate**: Verify citations match your claims
5. **Use DOIs**: Include permanent identifiers when available

#### Common Citation Styles in Physics

- **APS Style**: Used by Physical Review journals
- **AIP Style**: Used by American Institute of Physics journals
- **Nature Style**: Numbered references in order of appearance

### 1.4 arXiv Submission Process

arXiv is the primary preprint server for physics and quantum computing research.

#### Submission Requirements

**Format Requirements**
- Preferred format: TeX/LaTeX source files
- Does NOT accept: DVI, PS, or PDF generated from TeX
- Does NOT accept: Scanned documents
- Figures: PDF/JPG/PNG for PDFLaTeX; PS/EPS for standard LaTeX

**File Organization**
```
submission/
├── main.tex           # Main LaTeX file
├── references.bib     # BibTeX bibliography
├── figures/
│   ├── figure1.pdf
│   ├── figure2.pdf
│   └── figure3.pdf
└── supplementary.tex  # Optional supplementary material
```

#### Submission Timeline

- **14:00 ET deadline**: Submissions by this time appear that evening (20:00 ET)
- **Moderation**: All submissions reviewed for appropriateness
- **Endorsement**: First-time submitters to a category need endorsement

#### Category Selection for Quantum Computing

| arXiv Category | Focus |
|----------------|-------|
| `quant-ph` | Quantum physics, quantum information theory |
| `cond-mat` | Condensed matter implementations |
| `cs.ET` | Emerging technologies, quantum algorithms |
| `physics.comp-ph` | Computational physics aspects |

#### Licensing Options

- **CC BY 4.0**: Recommended; allows broad reuse with attribution
- **CC BY-SA 4.0**: Share-alike requirement
- **CC BY-NC-SA 4.0**: Non-commercial restriction
- **arXiv license**: Non-exclusive distribution rights

---

## 2. Literature Review Methodology

### 2.1 Systematic Literature Review Process

Based on established methodology (Kitchenham et al.), systematic reviews follow three phases:

#### Phase 1: Planning

1. **Define Research Questions**
   - What aspects of quantum computing are you investigating?
   - What time period will you cover?
   - What types of papers (theoretical, experimental, review)?

2. **Develop Search Strategy**
   - Select databases: arXiv, Web of Science, IEEE Xplore, ACM Digital Library
   - Define Boolean search strings
   - Determine inclusion/exclusion criteria

3. **Create Review Protocol**
   - Document methodology for reproducibility
   - Define data extraction forms
   - Plan quality assessment criteria

#### Phase 2: Conducting

1. **Execute Searches**
   ```
   Example search string:
   ("quantum computing" OR "quantum information")
   AND ("error correction" OR "fault tolerance")
   AND (2020-2025)
   ```

2. **Screen Results**
   - Title/abstract screening
   - Full-text review
   - Apply inclusion/exclusion criteria

3. **Extract Data**
   - Use standardized forms
   - Record: methods, results, quality indicators
   - Note connections between papers

4. **Synthesize Findings**
   - Identify themes and trends
   - Map the research landscape
   - Locate gaps and opportunities

#### Phase 3: Reporting

1. **Document Process**
   - PRISMA flow diagram
   - Search strategy details
   - Screening decisions

2. **Present Findings**
   - Taxonomy of research areas
   - Key results summary
   - Research gap identification

### 2.2 Organizing Papers and Notes

#### Recommended Workflow

1. **Initial Collection**
   - Save PDFs with consistent naming: `AuthorYear_ShortTitle.pdf`
   - Import to citation manager immediately
   - Tag with relevant categories

2. **Active Reading**
   - Annotate PDFs directly (Mendeley, Zotero PDF reader)
   - Create summary notes for each paper
   - Link related papers

3. **Knowledge Synthesis**
   - Use tools like Obsidian, Notion, or Roam for connected notes
   - Create concept maps
   - Maintain running bibliography

#### Note-Taking Template

```markdown
# Paper: [Title]
**Authors**: [Names]
**Year**: [YYYY]
**Venue**: [Journal/Conference]
**DOI/arXiv**: [Link]

## Summary (2-3 sentences)
[Main contribution and findings]

## Key Methods
- [Method 1]
- [Method 2]

## Key Results
- [Result 1]
- [Result 2]

## Relevance to My Research
[How this connects to your work]

## Questions/Gaps
[Unanswered questions, potential extensions]

## Key Quotes
> "[Important quote]" (p. X)

## Related Papers
- [[Paper A]]
- [[Paper B]]
```

### 2.3 Identifying Research Gaps

#### Common Gap Categories in Quantum Computing

Based on systematic reviews, persistent gaps include:

1. **Theory-Practice Gap**
   - Asymptotic speedups lack end-to-end resource analysis
   - Theoretical proposals without experimental validation

2. **Verification and Benchmarking**
   - Inconsistent reporting of quantum resources
   - Limited experimental validation methods

3. **Hardware-Software Co-design**
   - Insufficient integration of algorithm development with hardware constraints
   - Compiler optimization gaps

4. **Interdisciplinary Integration**
   - Lack of collaboration between physics, CS, and engineering
   - Insufficient economic analysis of quantum adoption

#### Gap Identification Techniques

1. **Taxonomy Analysis**: Map existing work to identify uncovered areas
2. **Trend Analysis**: Identify emerging topics with limited coverage
3. **Method Comparison**: Find techniques not yet applied to certain problems
4. **Scale Analysis**: Identify size/complexity regimes unexplored
5. **Application Domains**: Find fields where quantum approaches are untested

---

## 3. Research Proposal Writing

### 3.1 NSF Proposal Structure

The National Science Foundation (NSF) supports quantum information science through multiple programs.

#### Key NSF QIS Programs

- **Quantum Information Science (QIS)**: Theory and experiment
- **Quantum Leap Big Idea**: Large-scale quantum research
- **Connections in QIS (CQIS)**: Interdisciplinary connections

#### Standard NSF Proposal Components

| Section | Page Limit | Content |
|---------|------------|---------|
| Project Summary | 1 page | Overview, intellectual merit, broader impacts |
| Project Description | 15 pages | Technical proposal |
| References Cited | No limit | Bibliography |
| Budget | As needed | Personnel, equipment, travel |
| Budget Justification | 5 pages | Explanation of costs |
| Biographical Sketches | 3 pages/person | CV summary |
| Data Management Plan | 2 pages | Data sharing approach |

#### Project Description Structure

1. **Introduction and Background** (2-3 pages)
   - Research context and motivation
   - Prior work in the field
   - Preliminary results (if any)

2. **Research Objectives and Plan** (8-10 pages)
   - Specific aims and research questions
   - Technical approach and methodology
   - Timeline and milestones
   - Risk mitigation strategies

3. **Intellectual Merit** (integrated throughout)
   - Advancement of knowledge
   - Novel approaches
   - Qualifications of team

4. **Broader Impacts** (1-2 pages)
   - Societal benefits
   - Education and outreach
   - Broadening participation

### 3.2 DOE Proposal Writing

The Department of Energy Office of Science funds quantum research through multiple programs.

#### DOE QIS Funding Areas

- **Advanced Scientific Computing Research (ASCR)**
- **Basic Energy Sciences (BES)**
- **High Energy Physics (HEP)**
- **Nuclear Physics (NP)**
- **Fusion Energy Sciences (FES)**

#### National QIS Research Centers

DOE announced $625 million for National Quantum Information Science Research Centers:
- Awards of ~$125 million each over 5 years
- Must be led by DOE national laboratories
- Focus areas: quantum computing, communication, sensing, materials

#### DOE Proposal Tips

1. **Align with DOE mission**: Connect to energy, national security, science
2. **Team building**: Include national lab partners when possible
3. **Facilities**: Leverage DOE user facilities
4. **Milestones**: Clear, measurable progress indicators

### 3.3 Research Statement Writing

For academic job applications and internal reviews.

#### Structure

1. **Opening** (1 paragraph)
   - Research identity and vision
   - Major contributions

2. **Past Research** (1-2 pages)
   - Key accomplishments
   - Publications and impact
   - Technical skills demonstrated

3. **Current Research** (1 page)
   - Ongoing projects
   - Preliminary results

4. **Future Directions** (1-2 pages)
   - 5-year research plan
   - Potential funding sources
   - Connection to host institution

5. **Broader Vision** (1 paragraph)
   - Long-term goals
   - Field impact

### 3.4 Timeline and Milestone Planning

#### Example 3-Year Research Timeline

```
Year 1: Foundation
├── Q1: Literature review, preliminary calculations
├── Q2: Method development, initial simulations
├── Q3: First experimental/computational results
└── Q4: Analysis, first manuscript draft

Year 2: Development
├── Q1: Manuscript revision and submission
├── Q2: Extended studies, new directions
├── Q3: Conference presentation, collaboration building
└── Q4: Second manuscript preparation

Year 3: Maturation
├── Q1: Advanced results, synthesis
├── Q2: Major publication submission
├── Q3: Proposal writing for continuation
└── Q4: Documentation, mentoring, planning
```

#### Milestone Characteristics (SMART)

- **S**pecific: Clear deliverable
- **M**easurable: Quantifiable outcome
- **A**chievable: Realistic given resources
- **R**elevant: Connected to project goals
- **T**ime-bound: Specific deadline

---

## 4. Conference Presentations

### 4.1 Abstract Writing

#### Structure (200-300 words)

1. **Background/Motivation** (2-3 sentences)
   - Context and importance
   - Gap being addressed

2. **Methods** (2-3 sentences)
   - Approach taken
   - Key techniques

3. **Results** (2-4 sentences)
   - Main findings
   - Quantitative outcomes

4. **Conclusions** (1-2 sentences)
   - Significance
   - Implications

#### Best Practices

- Write standalone text (no references to figures/tables)
- Define acronyms
- Avoid jargon when possible
- Check word limits carefully
- Get feedback from non-specialists
- Review accepted abstracts from previous years

### 4.2 Poster Design

#### Layout Principles

```
┌──────────────────────────────────────────────────┐
│                    TITLE                          │
│         Authors, Affiliations, Contact            │
├──────────────────────────────────────────────────┤
│   Introduction  │   Methods    │   Results        │
│                 │              │                  │
│   • Background  │   • Approach │   • Finding 1    │
│   • Motivation  │   • Setup    │   • Finding 2    │
│   • Question    │   • Analysis │   • Key figure   │
├──────────────────────────────────────────────────┤
│       Conclusions        │    References/QR      │
│   • Key takeaways        │    [QR code to paper] │
│   • Future directions    │    Acknowledgments    │
└──────────────────────────────────────────────────┘
```

#### Design Guidelines (APS Recommendations)

- **Font size**: Title 85+ pt, section headers 48+ pt, body 28+ pt
- **Color**: High contrast, accessible color schemes
- **White space**: Don't overcrowd; aim for 40% empty space
- **Figures**: Large, clear, properly labeled
- **Flow**: Guide viewer through logical progression

#### Presenting Your Poster

1. Prepare 2-minute and 5-minute versions of your talk
2. Practice explaining figures without pointing
3. Have business cards or QR codes to your paper
4. Engage visitors with questions about their work

### 4.3 Oral Presentations

#### APS Talk Preparation

**Technical Requirements**
- Upload slides before the meeting or 3+ hours before session
- Use PowerPoint or PDF format
- Test audio/video equipment
- Arrive 15 minutes early

**Structure for a 12-minute Talk**

| Section | Time | Slides |
|---------|------|--------|
| Title/Motivation | 1 min | 1-2 |
| Background | 2 min | 2-3 |
| Methods | 2 min | 2-3 |
| Results | 5 min | 4-6 |
| Conclusions | 2 min | 1-2 |

**Slide Design**

- **Font**: 18+ pt minimum
- **Content**: One main idea per slide
- **Figures**: Prefer graphs/diagrams over text
- **Animations**: Use sparingly and purposefully

#### Delivery Tips

1. **Practice**: Time yourself repeatedly
2. **Pace**: Speak slowly; 1 slide per minute maximum
3. **Eye contact**: Look at audience, not screen
4. **Microphone**: Position 6 inches below chin
5. **Questions**: Repeat questions before answering

### 4.4 Q&A Handling

#### Preparation

- Anticipate likely questions
- Prepare backup slides for technical details
- Know limitations of your work
- Practice with colleagues

#### Response Strategies

1. **Listen completely**: Don't interrupt the question
2. **Clarify if needed**: "Just to make sure I understand..."
3. **Be honest**: "That's a great question we're still investigating"
4. **Stay calm**: Hostile questions happen; remain professional
5. **Time management**: Keep answers concise; offer to discuss afterward

---

## 5. Collaboration and Communication

### 5.1 Working with Advisors

#### Choosing an Advisor

Key factors for successful advisor-student relationships:
- Alignment of research interests
- Compatible working styles
- Clear communication patterns
- Support for professional development

#### Setting Expectations

Discuss early and document:
- Meeting frequency and format
- Response time for feedback
- Authorship policies
- Conference attendance
- Graduation timeline

#### Effective Communication

1. **Regular updates**: Brief weekly written summaries
2. **Meeting preparation**: Bring agenda and specific questions
3. **Progress documentation**: Keep research notebook current
4. **Proactive problem-solving**: Present solutions, not just problems
5. **Feedback integration**: Show how you've addressed comments

### 5.2 Research Group Dynamics

#### Building Productive Relationships

- **Peer collaboration**: Share knowledge, code, techniques
- **Lab citizenship**: Contribute to group infrastructure
- **Mentoring**: Help junior members as you advance
- **Social connection**: Participate in group activities

#### Navigating Challenges

- **Credit attribution**: Discuss authorship early in projects
- **Resource sharing**: Establish clear protocols
- **Conflicts**: Address early; involve advisor if needed
- **Diverse perspectives**: Value different approaches and backgrounds

### 5.3 Open Source Contribution

#### Major Quantum Computing Open Source Projects

| Project | Focus | Organization |
|---------|-------|--------------|
| **Qiskit** | Full-stack quantum SDK | IBM |
| **QuTiP** | Quantum dynamics simulation | Community |
| **ProjectQ** | Compilation framework | ETH Zurich |
| **Cirq** | NISQ algorithms | Google |
| **PennyLane** | Quantum ML | Xanadu |
| **ARTIQ** | Hardware control | NIST |

#### Contribution Best Practices

1. **Start small**: Documentation, bug fixes, tests
2. **Read guidelines**: Every project has contribution standards
3. **Engage community**: Join Discord/Slack, attend office hours
4. **Code quality**: Follow style guides, write tests
5. **Document**: Write clear commit messages and docstrings

#### Benefits of Open Source Contribution

- Build reputation in the community
- Learn from expert code review
- Network with researchers and engineers
- Gain practical software skills
- Contribute to the field's infrastructure

### 5.4 Scientific Communication

#### Written Communication

- **Email etiquette**: Clear subject lines, concise messages
- **Manuscript collaboration**: Use track changes, comment constructively
- **Documentation**: Write for future readers (including future you)

#### Oral Communication

- **Lab presentations**: Practice regularly
- **Journal clubs**: Discuss papers critically
- **Outreach**: Explain your work to non-experts

#### Building Your Presence

- **Personal website**: Research summary, publications, CV
- **Social media**: Twitter/X, LinkedIn for academic networking
- **ORCID**: Unique researcher identifier
- **Google Scholar**: Track citations, build profile

---

## 6. Recommended Tools and Resources

### 6.1 Writing and Documentation

| Tool | Purpose | Cost |
|------|---------|------|
| **Overleaf** | Collaborative LaTeX | Free/Premium |
| **VS Code + LaTeX Workshop** | Local LaTeX editing | Free |
| **Grammarly** | Grammar/style checking | Free/Premium |
| **Writefull** | Academic writing suggestions | Free/Premium |

### 6.2 Reference Management

| Tool | Best For | Cost |
|------|----------|------|
| **Zotero** | General use, browser integration | Free |
| **Mendeley** | PDF annotation, networking | Free |
| **Paperpile** | Google Docs integration | Paid |
| **JabRef** | BibTeX management | Free |

### 6.3 Knowledge Management

| Tool | Purpose | Cost |
|------|---------|------|
| **Obsidian** | Connected note-taking | Free |
| **Notion** | Project management + notes | Free/Paid |
| **Logseq** | Outline-based notes | Free |
| **Parsifal** | Systematic review management | Free |

### 6.4 Collaboration

| Tool | Purpose | Cost |
|------|---------|------|
| **GitHub** | Code collaboration | Free |
| **Slack** | Team communication | Free/Paid |
| **Zoom** | Video conferencing | Free/Paid |
| **Miro** | Visual collaboration | Free/Paid |

### 6.5 Key Databases and Resources

#### Literature Search
- **arXiv** (arxiv.org): Preprints
- **Web of Science**: Citation database
- **Google Scholar**: Broad search
- **Semantic Scholar**: AI-powered search
- **INSPIRE-HEP**: High-energy physics

#### Journals for Quantum Research
- **Physical Review A/X/Letters**: APS journals
- **PRX Quantum**: Dedicated quantum journal
- **Nature Physics/Communications Physics**: Nature portfolio
- **npj Quantum Information**: Nature partner journal
- **Quantum**: Open access quantum journal
- **New Journal of Physics**: IOP open access

---

## 7. Skill Development Timeline

### Phase 1: Foundation (Months 1-6)

**Goals**:
- Master LaTeX for physics papers
- Establish citation management workflow
- Complete first systematic literature review
- Present at group meetings

**Milestones**:
- [ ] Write 10-page document in LaTeX with figures
- [ ] Build bibliography of 50+ relevant papers
- [ ] Give 2-3 practice talks in group
- [ ] Create annotated bibliography

### Phase 2: Development (Months 7-12)

**Goals**:
- Submit first arXiv preprint
- Write research proposal (internal or small grant)
- Present poster at regional conference
- Begin open source contribution

**Milestones**:
- [ ] Complete manuscript draft with advisor feedback
- [ ] Submit to arXiv
- [ ] Write 5-page research proposal
- [ ] Design and present conference poster
- [ ] Make first open source contribution

### Phase 3: Advancement (Months 13-24)

**Goals**:
- Submit to peer-reviewed journal
- Give talk at major conference (APS March Meeting)
- Write NSF/DOE-style proposal
- Establish collaboration outside home institution

**Milestones**:
- [ ] Navigate peer review process
- [ ] Deliver 12-minute conference talk
- [ ] Complete full proposal with budget
- [ ] Collaborate on paper with external researcher
- [ ] Contribute significantly to open source project

### Phase 4: Independence (Months 25-36)

**Goals**:
- Lead multi-author paper
- Mentor junior researchers
- Write competitive fellowship application
- Build recognizable research identity

**Milestones**:
- [ ] First-author publication in top venue
- [ ] Mentor undergraduate or junior graduate student
- [ ] Submit fellowship application
- [ ] Establish personal research website
- [ ] Regular conference presentations

---

## References and Further Reading

### Academic Writing
- [Harvard Physics - How to Write a Scientific Paper](https://hoffman.physics.harvard.edu/Hoffman-Example-Paper.pdf)
- [GitHub - Paper Tips and Tricks](https://github.com/Wookai/paper-tips-and-tricks)
- [Overleaf Physics Templates](https://www.overleaf.com/latex/templates/tagged/physics)

### arXiv and Publishing
- [arXiv Submission Guidelines](https://info.arxiv.org/help/submit/index.html)
- [arXiv Format Requirements](https://info.arxiv.org/help/policies/format_requirements.html)
- [Communications Physics Submission Guidelines](https://www.nature.com/commsphys/submit/submission-guidelines)

### Literature Review
- [Systematic Literature Review: Quantum Machine Learning](https://arxiv.org/abs/2201.04093)
- [Quantum Computing: A Taxonomy and Systematic Review](https://arxiv.org/abs/2010.15559)
- [University of Nevada - Systematic Reviews in Physics](https://guides.library.unr.edu/c.php?g=51142&p=7853785)

### Research Funding
- [NSF Quantum Information Science](https://www.nsf.gov/funding/opportunities/quantum-information-science)
- [DOE Quantum Information Science](https://science.osti.gov/Initiatives/QIS)
- [National Quantum Initiative](https://www.quantum.gov/)

### Conference Presentations
- [APS Speaker Tips & Guidelines](https://www.aps.org/meetings/policies/speaker.cfm)
- [APS Tips for Poster Presentations](https://www.aps.org/meetings/policies/tips-poster-pre.cfm)
- [APS Abstract Writing Tips](https://www.aps.org/meetings/policies/abstract-tips.cfm)

### Collaboration and Tools
- [UC Berkeley Citation Management](https://guides.lib.berkeley.edu/physics/citations)
- [QOSF Quantum Software List](https://github.com/qosf/awesome-quantum-software)
- [QuTiP - Quantum Toolbox in Python](https://qutip.org/)
- [Qiskit on GitHub](https://github.com/Qiskit/qiskit)

### Peer Review and Publishing
- [UIUC - Writing and Responding to Referee Reports](https://courses.physics.illinois.edu/PHYS595/sp2023/lectures/WritingRefereeReports_SP23.pdf)
- [Communications Physics - Guide to Referees](https://www.nature.com/commsphys/referees/guide-to-referees)

### Research Group Dynamics
- [Physics Today - PhD Student-Adviser Pairing](https://pubs.aip.org/physicstoday/article/73/10/22/853198/PhD-student-adviser-pairing-is-critical-but-in-US)
- [arXiv - Physics PhD Student Perspectives on Finding Research Groups](https://arxiv.org/html/2311.04176v2)

---

*This guide was compiled from authoritative sources including the American Physical Society, National Science Foundation, Department of Energy, Nature Publishing Group, arXiv, and leading research universities. Last updated: February 2026.*
