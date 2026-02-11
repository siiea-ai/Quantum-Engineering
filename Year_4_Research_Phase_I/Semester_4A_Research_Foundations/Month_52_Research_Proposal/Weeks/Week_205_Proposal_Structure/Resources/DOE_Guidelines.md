# DOE Office of Science Proposal Guidelines

## Overview

The Department of Energy (DOE) Office of Science is the nation's largest supporter of basic research in the physical sciences, funding approximately $8 billion annually. For quantum science, DOE supports research through its National Quantum Initiative centers, Basic Energy Sciences (BES), Advanced Scientific Computing Research (ASCR), and High Energy Physics (HEP) programs.

---

## Part 1: DOE Funding Landscape

### Office of Science Programs

| Program | Focus Area | Quantum Relevance |
|---------|------------|-------------------|
| ASCR | Computing research | Quantum algorithms, simulation |
| BES | Materials and chemistry | Quantum materials, sensing |
| HEP | Particle physics | Quantum sensing, computing |
| NP | Nuclear physics | Quantum simulation |
| FES | Fusion energy | Quantum diagnostics |

### Quantum Information Science (QIS) at DOE

**National QIS Research Centers:**
- Established under the National Quantum Initiative (2018)
- Five centers funded at $115M each over 5 years
- Focus: quantum computing, networking, sensing

**QIS Research Programs:**
- Quantum Algorithms for Scientific Applications
- Quantum Networks for Open Science
- Quantum Sensors for Discovery Science
- Quantum Materials for Energy

### Funding Mechanisms

1. **FOA (Funding Opportunity Announcement)**
   - Targeted solicitations for specific topics
   - Typically $200K-2M per year
   - 3-5 year duration

2. **Open Calls**
   - Core research funding
   - Broader scope
   - Annual submission windows

3. **Lab-University Partnerships**
   - Joint proposals with national laboratories
   - Access to unique facilities
   - Often required for experimental work

---

## Part 2: DOE Proposal Structure

### Standard DOE Proposal Package

```
DOE OFFICE OF SCIENCE PROPOSAL
├── Technical Volume (15-25 pages, varies by FOA)
│   ├── Project Narrative
│   │   ├── Background and Significance
│   │   ├── Research Objectives
│   │   ├── Technical Approach
│   │   ├── Preliminary Results
│   │   └── Expected Outcomes
│   ├── Statement of Project Objectives (SOPO)
│   ├── Schedule/Milestones
│   └── Deliverables
├── Budget Volume
│   ├── SF-424A (Budget)
│   ├── Budget Justification
│   └── Indirect Cost Agreement
├── Business Volume
│   ├── SF-424 (Application Form)
│   ├── NEPA Questionnaire
│   └── Representations and Certifications
├── Key Personnel
│   ├── Biographical Sketches
│   ├── Current/Pending Support
│   └── Collaboration Letters
└── Appendices (as allowed)
    ├── Equipment Quotes
    ├── Facility Descriptions
    └── Letters of Support
```

### Key Differences from NSF

| Aspect | NSF | DOE |
|--------|-----|-----|
| Page limit (narrative) | 15 pages fixed | 15-25 pages (varies) |
| Broader Impacts | Required section | Less emphasized |
| Statement of Work | Not required | Required (SOPO) |
| National lab connection | Optional | Often required |
| Milestone specificity | Moderate | High (with deliverables) |
| Budget detail | Moderate | Very detailed |
| Review process | Panel + mail | Panel-centric |

---

## Part 3: Technical Volume

### Project Narrative

The Project Narrative is the core scientific document. DOE reviewers expect more technical depth than NSF:

**Recommended Structure:**

```
1. BACKGROUND AND SIGNIFICANCE (3-4 pages)
   ├── Scientific context and importance
   ├── Current state of the art
   ├── Gap or opportunity
   └── Why DOE should care (mission relevance)

2. RESEARCH OBJECTIVES (1-2 pages)
   ├── Overall goal
   ├── Specific objectives (3-5)
   └── Success criteria

3. TECHNICAL APPROACH (6-10 pages)
   ├── Objective 1: Approach and methods
   ├── Objective 2: Approach and methods
   ├── Objective 3: Approach and methods
   ├── Risk assessment
   └── Mitigation strategies

4. PRELIMINARY RESULTS (1-2 pages)
   ├── Relevant prior work
   ├── Preliminary data
   └── Evidence of feasibility

5. EXPECTED OUTCOMES (1-2 pages)
   ├── Scientific deliverables
   ├── Publications and datasets
   ├── Technology development
   └── Broader implications
```

### Technical Approach Depth

DOE expects exceptional technical detail. For each objective:

```
OBJECTIVE 1: Develop quantum error correction codes for biased noise

1.1 Rationale
    - Why this objective is critical to overall success
    - Connection to DOE mission priorities

1.2 Technical Approach
    - Method 1: Stabilizer code optimization
      • Mathematical framework (specific equations)
      • Algorithms to be developed
      • Software tools and languages
      • Computational resources required

    - Method 2: Decoder implementation
      • Machine learning architecture
      • Training data generation
      • Performance benchmarks

    - Method 3: Hardware validation
      • Target systems (IBM, IonQ, etc.)
      • Experimental protocols
      • Data analysis procedures

1.3 Milestones
    - M1.1 (Month 6): Complete code design, demonstrate in simulation
    - M1.2 (Month 12): Decoder trained, benchmark complete
    - M1.3 (Month 18): Hardware validation on 27+ qubit system

1.4 Risks and Mitigation
    - Risk: Noise model mismatch
      Mitigation: Adaptive calibration protocol
    - Risk: Hardware access limitations
      Mitigation: Cloud platform alternatives identified

1.5 Deliverables
    - D1.1: Open-source code library
    - D1.2: Publication in Physical Review Letters
    - D1.3: Dataset of noise characterization
```

---

## Part 4: Statement of Project Objectives (SOPO)

### Critical DOE Requirement

The SOPO is a contractual document that defines what you will deliver. It becomes part of your award terms.

### SOPO Structure

```
STATEMENT OF PROJECT OBJECTIVES

1. OBJECTIVES
   Clear statement of what will be accomplished

2. SCOPE
   Boundaries of the work (what's included and excluded)

3. TASKS
   Task 1: [Title]
     Subtask 1.1: [Specific activity]
     Subtask 1.2: [Specific activity]
     Deliverable: [Specific output]
     Milestone: [Specific achievement with date]

   Task 2: [Title]
     ...

4. MILESTONES AND DELIVERABLES
   Table summarizing all milestones and deliverables with dates

5. REPORTING REQUIREMENTS
   Quarterly progress reports
   Annual review presentations
   Final report
```

### SOPO Example for Quantum Research

```
STATEMENT OF PROJECT OBJECTIVES

Project Title: Tailored Quantum Error Correction for Superconducting Systems

1. OBJECTIVES
The objective of this project is to develop and validate quantum error
correction codes optimized for the biased noise characteristics of
superconducting quantum processors, achieving a 5x reduction in physical
qubit overhead compared to standard surface codes.

2. SCOPE
This project includes:
- Theoretical design of asymmetric stabilizer codes
- Development of ML-based decoders
- Numerical simulation and benchmarking
- Experimental validation on cloud quantum hardware

This project excludes:
- Hardware development or fabrication
- Codes for non-superconducting systems
- Real-time decoder implementation in hardware

3. TASKS

Task 1: Code Design and Optimization (Months 1-12)
  Subtask 1.1: Survey asymmetric code families
  Subtask 1.2: Develop optimization framework
  Subtask 1.3: Design tailored stabilizer measurements
  Deliverable D1: Code design toolkit (open-source)
  Milestone M1: Demonstrate 3x overhead reduction in simulation

Task 2: Decoder Development (Months 6-18)
  Subtask 2.1: Design neural network decoder architecture
  Subtask 2.2: Generate training data from noise models
  Subtask 2.3: Train and validate decoder performance
  Deliverable D2: Trained decoder models
  Milestone M2: Achieve 99% logical fidelity in simulation

Task 3: Experimental Validation (Months 12-24)
  Subtask 3.1: Characterize noise on target hardware
  Subtask 3.2: Implement codes on cloud platforms
  Subtask 3.3: Measure logical error rates
  Deliverable D3: Experimental dataset and analysis
  Milestone M3: Demonstrate overhead reduction on hardware

Task 4: Dissemination and Reporting (Months 1-24)
  Subtask 4.1: Publish peer-reviewed articles
  Subtask 4.2: Present at conferences
  Subtask 4.3: Submit quarterly reports
  Deliverable D4: Final project report
  Milestone M4: Two publications submitted

4. MILESTONES AND DELIVERABLES

| ID | Description | Due Date |
|----|-------------|----------|
| M1 | 3x overhead reduction (simulation) | Month 12 |
| M2 | 99% logical fidelity (decoder) | Month 18 |
| M3 | Hardware validation complete | Month 24 |
| M4 | Two publications submitted | Month 24 |
| D1 | Code design toolkit | Month 12 |
| D2 | Trained decoder models | Month 18 |
| D3 | Experimental dataset | Month 24 |
| D4 | Final project report | Month 24 |

5. REPORTING REQUIREMENTS
- Quarterly progress reports (10 pages max)
- Annual review presentation (30 minutes)
- Final report within 90 days of project end
```

---

## Part 5: National Laboratory Partnerships

### Why Partner with National Labs?

DOE strongly encourages (and sometimes requires) partnerships with national laboratories:

1. **Unique Facilities**
   - Quantum testbeds (Sandia, ORNL, ANL)
   - Characterization tools
   - Computational resources

2. **Expert Collaborators**
   - Staff scientists with specialized expertise
   - Continuity beyond student turnover

3. **Review Credibility**
   - Demonstrates feasibility
   - Shows integration with DOE mission

### Labs with Quantum Programs

| Laboratory | Quantum Focus |
|------------|---------------|
| Argonne (ANL) | Quantum networks, sensing |
| Brookhaven (BNL) | Quantum computing, materials |
| Fermilab (FNAL) | Quantum sensing, algorithms |
| Lawrence Berkeley (LBNL) | Quantum simulation, AQT |
| Los Alamos (LANL) | Quantum computing theory |
| Oak Ridge (ORNL) | Quantum computing, networks |
| Pacific Northwest (PNNL) | Quantum materials |
| Sandia (SNL) | Quantum hardware, integration |

### Structuring Lab Partnerships

```
PARTNERSHIP STRUCTURE

University PI (Lead):
- Overall project coordination
- Graduate student training
- Theoretical framework development

National Lab Co-I:
- Facility access
- Specialized measurements
- Technical consultation

Collaboration Mechanism:
- Subcontract from university to lab, OR
- Separate funding to lab (coordinated proposal)

Personnel Exchange:
- Student internships at lab (DOE SCGSR)
- Lab staff visits to university
- Joint publications
```

### Letter of Support Template

```
[Lab Letterhead]

Dear [FOA Manager],

I am pleased to confirm the participation of [National Lab] in the
proposed project "[Project Title]" led by [PI Name] at [University].

Our laboratory will contribute:
1. Access to [specific facility/equipment]
2. Technical expertise in [specific area]
3. [X] hours of staff scientist time for [specific activities]

This collaboration aligns with our laboratory's mission in [area]
and will leverage our unique capabilities in [specific capability].

[Lab scientist name] will serve as the laboratory point of contact
and co-investigator on this project.

Sincerely,
[Lab Leadership Signature]
```

---

## Part 6: Budget Requirements

### DOE Budget Detail

DOE requires exceptional budget detail. The SF-424A must be accompanied by extensive justification.

### Budget Categories

```
A. PERSONNEL
   - Name (or TBD with qualifications)
   - % effort
   - Months of salary
   - Rate calculation

B. FRINGE BENEFITS
   - Rate and basis
   - Components included

C. TRAVEL
   - Number of trips
   - Purpose of each trip
   - Estimated costs (airfare, hotel, per diem)

D. EQUIPMENT (>$5,000)
   - Specific items
   - Quotes required
   - Necessity justification

E. SUPPLIES
   - Categories
   - Quantities
   - Unit costs

F. CONTRACTUAL/SUBAWARDS
   - Collaborator budgets
   - Statement of work for each

G. OTHER DIRECT COSTS
   - Publication fees
   - Computing costs
   - User facility charges

H. INDIRECT COSTS
   - Federally negotiated rate
   - Documentation required

I. TOTAL COSTS
```

### Budget Justification Example

```
BUDGET JUSTIFICATION - YEAR 1

A. PERSONNEL ($187,500)

1. Principal Investigator - Dr. Jane Smith (1.0 summer month)
   $12,500/month x 1 month = $12,500
   Dr. Smith will direct all aspects of the project, including
   theoretical framework development and graduate student mentoring.

2. Graduate Research Assistant - TBD (12 months)
   $35,000/year x 1 student = $35,000
   One PhD student will be supported to conduct theoretical code
   design and numerical simulation under Aim 1.

3. Postdoctoral Researcher - Dr. [TBD] (12 months)
   $60,000/year x 1 researcher = $60,000
   One postdoc will lead experimental validation activities under
   Aim 3, requiring expertise in superconducting systems.

Total Personnel: $107,500

B. FRINGE BENEFITS ($37,625)

   Faculty: 35% of $12,500 = $4,375
   Graduate: 25% of $35,000 = $8,750
   Postdoc: 35% of $60,000 = $21,000
   [Benefits include health insurance, retirement, FICA]

C. TRAVEL ($8,000)

   1. Domestic Conferences ($4,000)
      - APS March Meeting: $1,500 (registration, airfare, hotel)
      - QIP Conference: $1,500
      - DOE PI Meeting: $1,000

   2. Collaboration Travel ($4,000)
      - 2 trips to Argonne National Lab: $2,000 each
        Purpose: Experimental coordination and facility access

D. EQUIPMENT ($0)
   No equipment requested in Year 1.

E. SUPPLIES ($5,000)
   - Computing supplies: $3,000 (storage, minor hardware)
   - Publication costs: $2,000 (open access fees)

F. CONTRACTUAL ($50,000)
   Subaward to Argonne National Laboratory
   [Detailed ANL budget attached]
   ANL will provide: noise characterization, facility access

G. OTHER DIRECT COSTS ($15,000)
   - Cloud quantum computing: $10,000
     (IBM Quantum, IonQ Aria access)
   - HPC allocation supplement: $5,000

H. INDIRECT COSTS ($92,437)
   MTDC base: $193,625
   Rate: 47.7% (federally negotiated, agreement attached)

I. TOTAL YEAR 1: $395,562
```

---

## Part 7: Review Process

### DOE Review Structure

```
PROPOSAL SUBMISSION
    ↓
ADMINISTRATIVE REVIEW (compliance check)
    ↓
TECHNICAL MERIT REVIEW (peer review panel)
    ↓
PROGRAM RELEVANCE REVIEW (DOE staff)
    ↓
SELECTION DECISION
    ↓
AWARD NEGOTIATION
    ↓
AWARD ISSUANCE
```

### Merit Review Criteria

DOE evaluates proposals on:

1. **Scientific and/or Technical Merit (50%)**
   - Quality and innovation of approach
   - Appropriateness of methods
   - Awareness of prior work
   - Feasibility

2. **Appropriateness of Proposed Project (30%)**
   - Relevance to DOE mission
   - Potential for scientific impact
   - Reasonableness of proposed resources
   - Qualifications of personnel

3. **Reasonableness and Appropriateness of Budget (20%)**
   - Cost/benefit relationship
   - Budget justification quality
   - Appropriateness of collaboration structure

### Program Relevance

After merit review, DOE program managers consider:

- Alignment with program priorities
- Portfolio balance
- Budget availability
- Policy considerations
- Geographic/institutional diversity

---

## Part 8: Quantum-Specific Considerations

### DOE Quantum Priorities

The Office of Science prioritizes quantum research that:

1. **Advances fundamental science**
   - New algorithms for scientific simulation
   - Quantum advantage in materials discovery
   - Precision measurement capabilities

2. **Develops enabling technologies**
   - Error correction and fault tolerance
   - Quantum networking infrastructure
   - Quantum sensor systems

3. **Builds workforce**
   - Graduate student training
   - Postdoctoral development
   - Laboratory internships

### Writing for DOE Quantum Reviewers

**Emphasize:**
- Connection to DOE mission (energy, environment, security)
- Scalability and practical impact
- National lab partnerships
- Specific deliverables
- Risk mitigation

**Include:**
- Quantitative performance targets
- Benchmark comparisons
- Hardware pathway considerations
- Open science (data, code sharing)

### Example Framing for DOE

**NSF framing:**
"This research advances fundamental understanding of quantum error correction."

**DOE framing:**
"This research directly supports DOE's goal of achieving practical quantum advantage for materials simulation, enabling breakthroughs in energy storage and catalysis design."

---

## Part 9: Common Mistakes

### Technical Errors

1. **Insufficient detail**
   - DOE expects more depth than NSF
   - Vague methodology is fatal

2. **Missing SOPO**
   - Contractual requirement
   - Must be specific and measurable

3. **Weak lab partnership**
   - Letters must be substantive
   - Clear role for lab collaborators

### Budget Errors

1. **Inadequate justification**
   - Every line needs explanation
   - No round numbers without basis

2. **Missing documentation**
   - Indirect cost agreement
   - Equipment quotes
   - Subcontract budgets

3. **Unrealistic costs**
   - Cloud computing underestimated
   - Travel insufficient for collaborations

### Strategic Errors

1. **Wrong program**
   - ASCR vs. BES vs. HEP
   - Must match program priorities

2. **Ignoring FOA requirements**
   - Topic restrictions
   - Partnership requirements
   - Page limits

3. **Poor mission connection**
   - Must relate to energy/environment/security
   - Not just fundamental science

---

## Part 10: Tips for Success

### Before Writing

1. **Read FOA 5+ times**
2. **Contact program manager**
3. **Establish lab partnerships early**
4. **Study funded projects (DOE OSTI)**
5. **Plan timeline (6+ months)**

### During Writing

1. **Lead with DOE mission relevance**
2. **Exceed technical depth expectations**
3. **Make SOPO concrete**
4. **Quantify everything**
5. **Get lab partner review**

### Before Submission

1. **Verify all required documents**
2. **Check budget calculations**
3. **Confirm lab letters finalized**
4. **Submit 48+ hours early**
5. **Have backup submission plan**

---

## Quick Reference

```
DOE PROPOSAL CHECKLIST

□ Technical Volume
  □ Project Narrative (15-25 pages)
    □ Background and Significance
    □ Research Objectives
    □ Technical Approach (detailed)
    □ Preliminary Results
    □ Expected Outcomes
  □ Statement of Project Objectives
  □ Milestones and Deliverables
  □ Schedule (Gantt chart)

□ Budget Volume
  □ SF-424A Budget
  □ Detailed Budget Justification
  □ Subcontract budgets (if applicable)
  □ Indirect Cost Agreement

□ Personnel Documents
  □ Biographical Sketches
  □ Current/Pending Support
  □ Collaboration Letters (substantive)

□ Compliance Documents
  □ SF-424 Application Form
  □ NEPA Questionnaire
  □ Representations and Certifications
```

---

*"DOE funds research that advances its mission. Make that connection explicit, specific, and compelling."*
