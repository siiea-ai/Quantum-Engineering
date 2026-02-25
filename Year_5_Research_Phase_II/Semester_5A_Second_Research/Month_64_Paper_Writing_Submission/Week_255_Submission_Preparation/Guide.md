# Week 255: Submission Preparation

## Days 1779-1785 | Preparing the Complete Submission Package

---

## Overview

Week 255 transforms your polished manuscript into a complete submission package. This includes journal-specific formatting, crafting a compelling cover letter, preparing supplementary materials, and selecting reviewers. The goal is a submission-ready package that maximizes your paper's chance of fair and favorable review.

### Week Objectives

By the end of Week 255, you will have:

1. Formatted the manuscript to target journal specifications
2. Written a compelling cover letter
3. Prepared complete supplementary material
4. Created author contributions and acknowledgments sections
5. Identified suggested (and excluded) reviewers
6. Assembled all submission components

---

## Target Journal Selection

### Confirming Your Target

Before final formatting, confirm your journal choice:

**Journal Selection Criteria**:

| Factor | Weight | Your Assessment |
|--------|--------|-----------------|
| Scope match | High | Does journal publish this type of work? |
| Impact/prestige | Medium | Appropriate for contribution level? |
| Audience | High | Who needs to see this work? |
| Timeline | Medium | Acceptable review/publication time? |
| Open access | Varies | Required by funder? |
| Cost | Varies | APCs acceptable? |

### Quantum Computing/Information Journal Hierarchy

**Highest Impact** (very selective, broad audience):
- Nature (IF ~50)
- Science (IF ~45)
- Nature Physics (IF ~25)
- Physical Review X (IF ~12)

**High Impact, Field-Specific**:
- PRX Quantum (IF ~8)
- npj Quantum Information (IF ~7)
- Physical Review Letters (IF ~8)
- Nature Communications (IF ~17)

**Solid Impact, Specialized**:
- Physical Review A (IF ~3)
- Quantum (IF ~7)
- New Journal of Physics (IF ~3)
- Quantum Science and Technology (IF ~6)

**Rapid Publication**:
- arXiv (preprint, no review)
- Physical Review Research (IF ~4)

### Matching Paper to Journal

**Consider PRX Quantum/npj QI if**:
- Significant advance in quantum technology
- Broad interest within QIS community
- Strong experimental + theoretical components

**Consider PRL if**:
- Brief but significant result
- Broad physics interest
- Result can be presented in 4 pages

**Consider PRA if**:
- Solid technical contribution
- Specialized audience
- Full technical details needed

---

## Day-by-Day Structure

### Day 1779 (Monday): Journal Requirements and Formatting

**Morning (3 hours): Requirement Analysis**

Each journal has specific requirements. Find and document them.

**APS Journals (PRX Quantum, PRL, PRA)**:

Requirements to verify:
- [ ] REVTeX 4.2 template
- [ ] Word/character limits (PRL: 3500 words)
- [ ] Figure file formats (PDF, EPS, PNG, TIFF)
- [ ] Reference style (APS format)
- [ ] Supplementary material format
- [ ] Abstract word limit

**Nature Family (npj QI, Nature Physics)**:

Requirements to verify:
- [ ] Word limit (~3000-4000 main text)
- [ ] Reference limit (~50 in main text)
- [ ] Methods section location (end or separate)
- [ ] Extended Data vs. Supplementary Information
- [ ] Figure requirements
- [ ] Article type (Article, Letter, etc.)

**Format Comparison**:

| Element | APS Style | Nature Style |
|---------|-----------|--------------|
| Template | REVTeX 4.2 | Word/LaTeX template |
| Abstract | 500 words | 150-200 words |
| References | Numbered [1] | Superscript^1 |
| Methods | Section in main text | End of paper or separate |
| Equations | Numbered | Numbered |
| Figures | Numbered, separate files | Numbered, specific dimensions |

**Afternoon (3 hours): Manuscript Formatting**

Apply journal-specific formatting:

**REVTeX Template Setup** (APS):
```latex
\documentclass[prx,reprint,amsmath,amssymb,aps,showpacs]{revtex4-2}

\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{braket}

\begin{document}

\title{Your Paper Title}

\author{First A. Author}
\email{first.author@university.edu}
\affiliation{Department of Physics, University Name, City, Country}

\author{Second B. Author}
\affiliation{Department of Physics, Another University, City, Country}

\date{\today}

\begin{abstract}
Your abstract here...
\end{abstract}

\maketitle

% Main text

\bibliography{references}

\end{document}
```

**Formatting Checklist**:
- [ ] Correct document class/template
- [ ] Author information complete
- [ ] Affiliations correct
- [ ] Abstract within word limit
- [ ] Sections properly formatted
- [ ] Equations numbered
- [ ] Figures referenced correctly
- [ ] References in correct style

**Evening (1 hour): Reference Formatting**

Ensure references match journal style:
- Verify BibTeX entries are complete
- Check journal abbreviations
- Include DOIs where available
- Verify arXiv citations are current

---

### Day 1780 (Tuesday): Cover Letter Drafting

**Morning (3 hours): Cover Letter Strategy**

The cover letter is your paper's advocate. It should:
1. Introduce the paper appropriately
2. Highlight significance without overselling
3. Suggest suitable reviewers
4. Address any special considerations

**Cover Letter Structure**:

```
[Your Information]
[Date]

[Editor's Name]
[Journal Name]
[Address]

Dear [Editor Name / "Editor"],

[PARAGRAPH 1: Introduction]
We submit our manuscript entitled "[Title]" for consideration
as a [Article/Letter/Rapid Communication] in [Journal Name].

[PARAGRAPH 2: Summary]
[2-3 sentences describing what the paper does and its main findings]

[PARAGRAPH 3: Significance]
[Why this work is important and suitable for this journal]

[PARAGRAPH 4: Novelty (optional)]
[What's new compared to prior work]

[PARAGRAPH 5: Reviewer Suggestions]
We suggest the following reviewers:
[Names and affiliations]

We request exclusion of:
[Names and brief reason if appropriate]

[PARAGRAPH 6: Closing]
[Any special considerations: arXiv posting, related submissions, etc.]
We confirm that this work is original, not under consideration
elsewhere, and all authors have approved the submission.

Thank you for considering our work.

Sincerely,

[Corresponding Author]
[Title]
[Affiliation]
[Email]
[Phone]
```

**What to Include**:
- Brief summary of contribution
- Why suitable for this journal
- Any noteworthy aspects (novel technique, timeliness)
- Suggested and excluded reviewers
- Compliance statements

**What to Avoid**:
- Hyperbole ("revolutionary," "groundbreaking")
- Excessive length (1 page maximum)
- Repeating the abstract verbatim
- Making promises the paper doesn't deliver
- Criticizing competitors

**Afternoon (3 hours): Draft Complete Cover Letter**

Write your cover letter draft:

**Example Cover Letter** (Quantum Error Correction paper to PRX Quantum):

---

Dear Editor,

We submit our manuscript entitled "Below-Threshold Logical Error Rates in a Surface Code Processor" for consideration as an Article in PRX Quantum.

In this work, we demonstrate fault-tolerant operation of a distance-3 surface code on a 17-qubit superconducting processor. By implementing a novel stabilizer measurement protocol that reduces crosstalk errors, we achieve a logical error rate of 2.8%, which is below the physical error rate of 3.2% per operation. This represents the first experimental demonstration of below-threshold logical performance in a topological error-correcting code.

This result addresses a central challenge in quantum error correction: demonstrating that logical qubits can outperform their physical constituents. Our work is timely given recent advances in superconducting qubit coherence and our results provide a clear path toward larger code distances. We believe PRX Quantum readers, who are broadly interested in practical quantum computing advances, will find this work of significant interest.

We suggest the following researchers as potential reviewers:
- Prof. Jane Smith (University A) - expert in surface codes
- Prof. John Doe (University B) - expert in superconducting qubits
- Dr. Alice Johnson (Lab C) - expert in quantum error correction experiments

We request that Dr. [Name] ([Affiliation]) be excluded from the review process due to an ongoing collaboration.

This manuscript has been posted to arXiv (arXiv:XXXX.XXXXX) and is not under consideration at any other journal. All authors have read and approved the final manuscript.

Thank you for considering our work.

Sincerely,

[Corresponding Author]

---

**Evening (1 hour): Cover Letter Refinement**

Review and polish:
- Check for typos
- Verify all claims are accurate
- Ensure appropriate tone (confident but not arrogant)
- Confirm all author information is correct

---

### Day 1781 (Wednesday): Supplementary Material Preparation

**Morning (3 hours): Supplement Organization**

The supplement should contain:
- Extended derivations
- Additional data and figures
- Detailed methods
- Code and data availability

**Supplement Structure**:

```latex
\documentclass[aps,prl,twocolumn,superscriptaddress]{revtex4-2}
\usepackage{graphicx}

\begin{document}

\title{Supplementary Material for: [Paper Title]}

\maketitle

\tableofcontents

\section{Extended Derivations}
\subsection{Derivation of Threshold Formula}
...

\section{Additional Experimental Details}
\subsection{Device Parameters}
...
\subsection{Calibration Procedures}
...

\section{Supplementary Figures}
\subsection{Extended Data Analysis}
...

\section{Data and Code Availability}
...

\bibliography{references}

\end{document}
```

**Supplement Best Practices**:
- Number figures as S1, S2, etc. (or FigS1, following journal style)
- Cross-reference from main text ("see Supplementary Section II")
- Make supplement self-contained but not redundant
- Include equations that support but aren't central to main text

**Afternoon (3 hours): Complete Supplement**

Finalize all supplementary materials:

**Derivations Checklist**:
- [ ] All claimed derivations included
- [ ] Steps logical and complete
- [ ] Notation consistent with main text
- [ ] Key assumptions stated

**Additional Data Checklist**:
- [ ] Supplementary figures captioned
- [ ] Extended data tables complete
- [ ] Error analysis details included
- [ ] Control experiments shown

**Methods Details Checklist**:
- [ ] Device/sample information complete
- [ ] Experimental procedures detailed
- [ ] Numerical methods specified
- [ ] Analysis pipelines described

**Evening (1 hour): Supplement Review**

- Read through for coherence
- Check all references to main text
- Verify figure/table numbering
- Ensure formatting matches main paper

---

### Day 1782 (Thursday): Author Contributions and Acknowledgments

**Morning (2 hours): Author Contributions**

Clearly document each author's role using CRediT (Contributor Roles Taxonomy):

**CRediT Roles**:
- Conceptualization
- Methodology
- Software
- Validation
- Formal analysis
- Investigation
- Resources
- Data curation
- Writing – original draft
- Writing – review & editing
- Visualization
- Supervision
- Project administration
- Funding acquisition

**Example Statement**:
> **Author Contributions**: A.B.C. and D.E.F. conceived the project. A.B.C. designed and performed the experiments. G.H.I. developed the theoretical model. A.B.C. and G.H.I. analyzed the data. J.K.L. provided experimental resources. A.B.C. wrote the original draft with input from all authors. D.E.F. supervised the project. All authors reviewed and approved the manuscript.

**Best Practices**:
- Be specific about who did what
- Ensure all authors approve their listed contributions
- Include all substantial contributors as authors
- Thank non-author contributors in acknowledgments

**Morning (1 hour): Acknowledgments**

Acknowledge:
- Funding sources (with grant numbers)
- Facilities and resources used
- Helpful discussions
- Technical assistance
- Data/code providers

**Example Acknowledgments**:
> We thank J. Doe for helpful discussions and M. Smith for technical assistance with cryogenic systems. This work was supported by the National Science Foundation under Grant No. PHY-XXXXXXX, the Army Research Office under Grant No. W911NF-XX-X-XXXX, and the DOE Office of Science National Quantum Information Science Research Centers. Devices were fabricated at the University Nanofabrication Facility.

**Afternoon (3 hours): Data and Code Availability**

Address reproducibility requirements:

**Data Availability Statement Options**:

1. **Data in repository**:
> Data supporting this study are available in [Repository Name] at [DOI/URL].

2. **Data available on request**:
> Data are available from the corresponding author upon reasonable request.

3. **Data in supplement**:
> All data generated during this study are included in the Supplementary Information.

**Code Availability Statement Options**:

1. **Code in repository**:
> Code used for data analysis is available at [GitHub/Zenodo URL].

2. **Code available on request**:
> Custom code is available from the corresponding author upon reasonable request.

3. **Standard software**:
> Analysis was performed using [Software Name] version X.X.

**Repository Best Practices**:
- Use Zenodo for DOI-registered archives
- Include README with usage instructions
- Specify licenses
- Version control your releases

**Evening (1 hour): Ethics and Conflict Statements**

Prepare required statements:

**Conflict of Interest**:
> The authors declare no competing interests.

OR

> Author X.Y.Z. holds equity in Company Name, which develops quantum computing hardware.

**Ethics Statement** (if applicable):
> This research did not involve human subjects or animal experiments.

---

### Day 1783 (Friday): Reviewer Selection

**Morning (3 hours): Identifying Reviewers**

Suggesting appropriate reviewers increases chance of fair review.

**Reviewer Selection Criteria**:

| Criterion | Why Important |
|-----------|---------------|
| Expertise | Can evaluate technical claims |
| Objectivity | No conflicts with authors |
| Fairness | Known for constructive reviews |
| Availability | Likely to accept/respond promptly |
| Diversity | Different perspectives |

**Finding Suitable Reviewers**:

1. **Citation analysis**: Who have you cited extensively?
2. **Conference contacts**: Who asked good questions about this work?
3. **Related papers**: Who published similar work recently?
4. **Advisor input**: Who does your advisor suggest?

**Reviewer Categories**:

- **Method experts**: Can evaluate technical approach
- **Application experts**: Can assess significance for applications
- **Theory experts**: Can verify theoretical claims
- **Experimentalists**: Can evaluate experimental design (if applicable)

**Afternoon (2 hours): Reviewer Documentation**

Prepare reviewer information:

**For Each Suggested Reviewer**:
- Full name and title
- Institutional affiliation
- Email address
- Brief justification (not always required, but useful)

**Suggested Reviewer Template**:

| Name | Affiliation | Email | Expertise |
|------|-------------|-------|-----------|
| Prof. Jane Smith | MIT | jsmith@mit.edu | Surface codes, experimental QEC |
| Dr. John Doe | Google Quantum AI | johndoe@google.com | Superconducting qubits |
| Prof. Alice Johnson | ETH Zurich | ajohnson@ethz.ch | Fault-tolerant quantum computing |

**Reviewers to Exclude**:
- Direct collaborators (within ~3 years)
- Competitors with known bias
- People with personal conflicts
- Anyone who might not be objective

**Document exclusions with reasons** (some journals ask):
> We request exclusion of Dr. [Name] due to an ongoing collaboration on a related project.

**Evening (2 hours): Final Package Assembly**

Gather all submission components:

**Submission Package Checklist**:
- [ ] Main manuscript (formatted for journal)
- [ ] Figures (separate files, correct format)
- [ ] Supplementary material
- [ ] Cover letter
- [ ] Author contributions statement
- [ ] Acknowledgments
- [ ] Data/code availability statement
- [ ] Conflict of interest statement
- [ ] Suggested reviewers list
- [ ] Excluded reviewers list
- [ ] Author agreement (all authors approved)

---

### Day 1784 (Saturday): Graphical Abstract and Highlights

**Morning (3 hours): Graphical Abstract**

Some journals require or encourage graphical abstracts.

**Graphical Abstract Purpose**:
- Visual summary of paper
- Attracts readers browsing journal
- Shared on social media

**Design Principles**:
- One key figure or diagram
- Minimal text
- Clear at small size
- Self-explanatory

**Dimensions** (varies by journal):
- Nature/npj: 530 x 1328 pixels (1:2.5 aspect ratio)
- ACS: 5.0 x 2.75 inches (300 DPI)
- Elsevier: 531 x 1328 pixels

**Graphical Abstract Elements**:
1. Central figure (from paper or custom)
2. Key finding/message
3. Title or brief text (optional)
4. Author/journal info (optional)

**Afternoon (2 hours): Highlights/Key Points**

Some journals request bullet-point highlights.

**Highlights Format** (typically 3-5 points):
- Each point: one complete sentence
- Maximum ~85 characters each
- Active voice
- Quantitative when possible

**Example Highlights**:
1. We demonstrate below-threshold logical error rates in a surface code.
2. A novel stabilizer measurement protocol reduces crosstalk errors by 60%.
3. Logical error rates scale favorably with increasing code distance.
4. Results provide a path toward fault-tolerant quantum computing.

**Evening (1 hour): Social Media Materials**

Prepare materials for paper promotion:

**Twitter/X Thread Draft**:
```
1/n: New paper! We demonstrate below-threshold logical error rates
in a surface code for the first time. [Link]

2/n: The key innovation: a new stabilizer measurement protocol
that reduces crosstalk. [Figure]

3/n: Result: logical error rate < physical error rate,
crossing the threshold. [Graph]

4/n: This is a critical step toward fault-tolerant quantum
computers. Paper and data: [links]
```

**LinkedIn Post Draft**:
```
Excited to share our new paper: [Title]

Key finding: [One sentence summary]

Why it matters: [One sentence significance]

Link: [arXiv/journal]
```

---

### Day 1785 (Sunday): Final Review and Package Completion

**Morning (3 hours): Complete Package Review**

Go through entire submission package one final time.

**Manuscript Final Check**:
- [ ] Title accurate and compelling
- [ ] Abstract within word limit
- [ ] All sections complete
- [ ] Figures numbered correctly
- [ ] References complete and consistent
- [ ] No tracked changes or comments remaining
- [ ] Formatting matches journal requirements

**Cover Letter Final Check**:
- [ ] Addressed to correct editor/journal
- [ ] No typos or errors
- [ ] All claims accurate
- [ ] Reviewer suggestions complete
- [ ] Contact information correct

**Supplementary Material Final Check**:
- [ ] All referenced sections exist
- [ ] Figures captioned and numbered
- [ ] References complete
- [ ] Self-contained and coherent

**Afternoon (3 hours): Co-Author Approval**

Before submission, obtain approval from all co-authors.

**Author Approval Checklist**:
- [ ] All authors received final manuscript
- [ ] All authors approved submission
- [ ] Author order confirmed
- [ ] Affiliations verified by each author
- [ ] Author contributions approved
- [ ] Conflict statements approved

**Email Template for Co-Author Approval**:

```
Subject: [Paper Title] - Final Review Before Submission

Dear co-authors,

The manuscript "[Title]" is ready for submission to [Journal].

Please review the attached final version and confirm:
1. You approve submission
2. Your affiliation is correct
3. Your author contributions are accurate
4. You have disclosed any conflicts of interest

Please reply by [Date] with your approval or any final comments.

Best regards,
[Corresponding Author]

Attachments:
- Final manuscript
- Supplementary material
- Cover letter
```

**Evening (1 hour): Submission Preparation**

Prepare for Day 1786 submission:

- [ ] Journal account created/verified
- [ ] ORCID linked to account
- [ ] All files ready to upload
- [ ] Co-author emails confirmed
- [ ] Submission portal bookmarked
- [ ] arXiv account ready (if posting)

**Week 255 Deliverable**:
Complete submission package ready for upload

---

## Submission Portal Navigation

### Common Journal Systems

**Editorial Manager (APS journals)**:
1. Register/login at journal site
2. Select "Submit New Manuscript"
3. Choose article type
4. Enter title, abstract, authors
5. Upload manuscript and figures
6. Add cover letter and metadata
7. Suggest/exclude reviewers
8. Review and submit

**Nature Portfolio Manuscript Tracking System**:
1. Login to manuscript.nature.com
2. Start new submission
3. Select journal and article type
4. Enter metadata
5. Upload files (manuscript, figures, etc.)
6. Enter author information
7. Add additional information (ethics, data availability)
8. Submit

**ScholarOne (Various journals)**:
1. Create account/login
2. Start submission
3. Enter manuscript information
4. Upload files
5. Enter author details
6. Answer submission questions
7. Designate reviewers
8. Review and submit

---

## Common Submission Issues and Solutions

| Issue | Solution |
|-------|----------|
| File format not accepted | Convert to required format (PDF, EPS) |
| Figure resolution too low | Regenerate at 300+ DPI |
| Abstract too long | Edit to meet word limit |
| Reference format wrong | Use journal-specific BibTeX style |
| Author ORCID missing | Register at orcid.org |
| Affiliation unclear | Use official institution name |
| PDF compilation error | Debug LaTeX locally first |
| Reviewer email bounces | Find updated contact info |

---

## Reflection Questions

At the end of Week 255, consider:

1. Is the submission package complete and correct?
2. Does the cover letter effectively represent the paper?
3. Are suggested reviewers appropriate and unbiased?
4. Have all co-authors approved submission?
5. Is everything ready for Day 1786?

Complete the Week 255 Reflection template to document your progress.
