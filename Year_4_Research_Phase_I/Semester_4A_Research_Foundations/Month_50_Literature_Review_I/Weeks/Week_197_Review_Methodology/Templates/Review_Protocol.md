# Systematic Literature Review Protocol Template

---

## 1. Protocol Information

### 1.1 Title

**Working Title:**
```
[Descriptive title of your systematic literature review]
Example: "Neural Network Decoders for Quantum Error Correction: A Systematic Review"
```

### 1.2 Authors

| Role | Name | Affiliation | Contact |
|------|------|-------------|---------|
| Lead Reviewer | [Your name] | [Institution] | [Email] |
| Co-Reviewer (if any) | | | |
| Advisor | | | |

### 1.3 Protocol Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | [Date] | Initial protocol | [Name] |
| | | | |

### 1.4 Registration

**Registration (Optional but Recommended):**
- PROSPERO: [If applicable]
- OSF Preregistration: [If applicable]
- Protocol DOI: [If applicable]

---

## 2. Background and Rationale

### 2.1 Context

**Research Domain:**
```
[Describe the broader research area]

Example: Quantum error correction (QEC) is essential for fault-tolerant quantum
computing. Efficient decoders that can quickly and accurately identify errors
are critical for practical implementation. Machine learning approaches have
emerged as promising alternatives to traditional algorithmic decoders.
```

### 2.2 Motivation

**Why is this review needed?**
```
[Explain the need for a systematic review in this area]

Example: While numerous studies have applied neural networks to QEC decoding,
there is no comprehensive systematic review that:
1. Catalogues all ML approaches applied to QEC decoding
2. Compares performance across different methods
3. Identifies gaps and future research directions
```

### 2.3 Existing Reviews

**Prior Reviews:**
| Authors | Year | Title | Gap This Review Addresses |
|---------|------|-------|---------------------------|
| [Citation] | [Year] | [Title] | [How your review differs] |

```
Example: Torlai & Melko (2020) reviewed neural network quantum states but
did not systematically cover the decoder literature. No SLR following
Kitchenham guidelines exists for ML-based QEC decoders.
```

---

## 3. Research Questions

### 3.1 Primary Research Questions

**RQ1:**
```
[Main research question]
Example: What machine learning techniques have been applied to quantum error
correction decoding?
```

**RQ2:**
```
[Comparative question]
Example: How do ML-based decoders compare to traditional decoders (e.g., MWPM)
in terms of logical error rate and computational efficiency?
```

**RQ3:**
```
[Gap/limitation question]
Example: What are the current limitations and open challenges for ML-based
QEC decoders?
```

### 3.2 Secondary Research Questions

**RQ4:**
```
[Additional question]
Example: What QEC codes have been studied with ML decoders, and which remain
unexplored?
```

**RQ5:**
```
[Additional question]
Example: What training data and simulation methodologies are used to develop
and evaluate ML decoders?
```

### 3.3 PICOC Framework

| Element | Definition | Your Review |
|---------|------------|-------------|
| **P**opulation | What is being studied | [e.g., QEC decoding systems] |
| **I**ntervention | Treatment/approach | [e.g., Machine learning decoders] |
| **C**omparison | Compared against | [e.g., Traditional algorithmic decoders] |
| **O**utcome | What is measured | [e.g., Logical error rate, speed, threshold] |
| **C**ontext | Setting/constraints | [e.g., Surface codes, circuit-level noise] |

---

## 4. Search Strategy

### 4.1 Databases

**Primary Databases:**
| Database | Reason for Inclusion |
|----------|---------------------|
| arXiv (quant-ph) | Preprints, most current research |
| Google Scholar | Broad coverage |
| IEEE Xplore | Engineering implementations |
| Web of Science | Citation analysis |
| Semantic Scholar | AI-powered recommendations |

**Secondary Sources:**
- Conference proceedings (QIP, QEC workshops)
- Key research group publications
- Reference lists of included papers

### 4.2 Search Terms

**Concept Groups:**

```
Group A (Population): QEC
- "quantum error correction"
- QEC
- "error correcting code"
- "surface code"
- "topological code"
- "stabilizer code"

Group B (Intervention): Machine Learning
- "machine learning"
- "neural network"
- "deep learning"
- "reinforcement learning"
- "ML"
- "neural decoder"

Group C (Outcome): Decoding
- decoder
- decoding
- decode
- "error correction"
- threshold
- "logical error"
```

### 4.3 Search Strings

**Combined Query (Generic):**
```
("quantum error correction" OR QEC OR "surface code" OR "stabilizer code")
AND
("machine learning" OR "neural network" OR "deep learning" OR "reinforcement learning")
AND
(decoder OR decoding OR threshold OR "logical error")
```

**Database-Specific Queries:**

**arXiv:**
```
(ti:"quantum error correction" OR ti:"surface code") AND
(abs:"neural network" OR abs:"machine learning") AND
abs:decoder
```

**Google Scholar:**
```
"quantum error correction" OR "surface code" "neural network" OR
"machine learning" decoder
```

**Web of Science:**
```
TS=("quantum error correction" OR "surface code") AND
TS=("neural network" OR "machine learning") AND
TS=(decoder OR decoding)
```

### 4.4 Search Execution Plan

| Step | Database | Query | Date | Executed By |
|------|----------|-------|------|-------------|
| 1 | arXiv | [Query A1] | [Planned date] | [Name] |
| 2 | Google Scholar | [Query G1] | [Planned date] | [Name] |
| 3 | IEEE Xplore | [Query I1] | [Planned date] | [Name] |
| 4 | Web of Science | [Query W1] | [Planned date] | [Name] |
| 5 | Forward snowballing | From included | [Planned date] | [Name] |
| 6 | Backward snowballing | References | [Planned date] | [Name] |

---

## 5. Selection Criteria

### 5.1 Inclusion Criteria

| ID | Criterion | Rationale |
|----|-----------|-----------|
| I1 | Published between [Year] and [Year] | [Recent relevant work] |
| I2 | Peer-reviewed or arXiv preprint | [Quality threshold] |
| I3 | Written in English | [Language capability] |
| I4 | Applies ML to QEC decoding | [Directly relevant] |
| I5 | Reports quantitative results | [Comparable outcomes] |
| I6 | Full text accessible | [Ability to assess] |

### 5.2 Exclusion Criteria

| ID | Criterion | Rationale |
|----|-----------|-----------|
| E1 | Duplicate publication | [Avoid double-counting] |
| E2 | Abstract/poster only | [Insufficient detail] |
| E3 | Non-QEC ML applications | [Out of scope] |
| E4 | Purely theoretical (no results) | [Need empirical data] |
| E5 | Theses/dissertations | [Prefer peer-reviewed] |
| E6 | Survey/review papers | [Not primary research] |

### 5.3 Screening Process

**Stage 1: Title/Abstract Screening**
```
1. Apply I1-I6 and E1-E6 to title/abstract
2. Mark as: Include / Exclude / Uncertain
3. Uncertain papers proceed to Stage 2
4. Document exclusion reasons
```

**Stage 2: Full-Text Screening**
```
1. Obtain full text for remaining papers
2. Apply all criteria to full text
3. Make final include/exclude decision
4. Document reasons for exclusion
```

**Disagreement Resolution:**
```
If multiple reviewers: Discuss disagreements, consult third party if needed
If single reviewer: Re-review uncertain papers after initial pass
```

---

## 6. Quality Assessment

### 6.1 Quality Criteria

| ID | Criterion | Score Range |
|----|-----------|-------------|
| Q1 | Clear research objectives | 0-3 |
| Q2 | Appropriate methodology | 0-3 |
| Q3 | Sound experimental design | 0-3 |
| Q4 | Sufficient data/evaluation | 0-3 |
| Q5 | Valid conclusions | 0-3 |
| Q6 | Reproducibility information | 0-3 |

**Scoring:**
- 3 = Fully satisfies criterion
- 2 = Partially satisfies criterion
- 1 = Minimally satisfies criterion
- 0 = Does not satisfy criterion

**Quality Threshold:**
```
Minimum total score: [X] out of 18
Papers below threshold: Flag for sensitivity analysis
```

### 6.2 Quality Assessment Form

| Paper ID | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Total | Include |
|----------|----|----|----|----|----|----|-------|---------|
| | | | | | | | | |

---

## 7. Data Extraction

### 7.1 Data Items

**Bibliographic:**
- Paper ID, Title, Authors, Year, Venue, DOI/arXiv

**Methodology:**
- ML technique (CNN, RNN, RL, etc.)
- QEC code (surface, color, etc.)
- Noise model
- Training approach
- Dataset size/generation

**Results:**
- Logical error rate
- Threshold value
- Comparison baseline
- Computational speed
- Scalability analysis

**Context:**
- Code distance studied
- Physical error rates
- Hardware considerations

### 7.2 Data Extraction Form

```
Paper ID: ____________
Title: ____________
Authors: ____________
Year: ____________
Venue: ____________

ML Technique: [ ] CNN [ ] RNN [ ] GNN [ ] RL [ ] Other: ____
QEC Code: [ ] Surface [ ] Color [ ] LDPC [ ] Other: ____
Noise Model: [ ] Depolarizing [ ] Circuit [ ] Realistic [ ] Other: ____

Baseline Comparison: ____________
Key Metrics Reported:
- Logical error rate: ____________
- Threshold: ____________
- Speed (inference time): ____________

Strengths: ____________
Limitations: ____________
Future Work Suggested: ____________

Notes: ____________
```

---

## 8. Data Synthesis

### 8.1 Synthesis Approach

**Narrative Synthesis:**
```
1. Group papers by ML technique
2. Group papers by QEC code type
3. Identify common findings
4. Note disagreements/contradictions
5. Identify research gaps
```

**Quantitative Synthesis (if applicable):**
```
1. Extract comparable metrics
2. Create comparison tables
3. Statistical analysis if sufficient data
4. Sensitivity analysis for quality
```

### 8.2 Presentation of Results

**Planned Tables:**
- Table 1: Summary of included studies
- Table 2: Comparison of ML techniques
- Table 3: Performance metrics summary
- Table 4: Quality assessment results

**Planned Figures:**
- PRISMA flow diagram
- Performance comparison chart
- Timeline of publications
- Concept map of research themes

---

## 9. Timeline

### 9.1 Milestones

| Phase | Activity | Start | End | Status |
|-------|----------|-------|-----|--------|
| 1 | Protocol development | [Date] | [Date] | [ ] |
| 2 | Database searches | [Date] | [Date] | [ ] |
| 3 | Title/abstract screening | [Date] | [Date] | [ ] |
| 4 | Full-text screening | [Date] | [Date] | [ ] |
| 5 | Quality assessment | [Date] | [Date] | [ ] |
| 6 | Data extraction | [Date] | [Date] | [ ] |
| 7 | Data synthesis | [Date] | [Date] | [ ] |
| 8 | Writing | [Date] | [Date] | [ ] |
| 9 | Review/revision | [Date] | [Date] | [ ] |

### 9.2 Progress Tracking

**Week 197 (Protocol):**
- [ ] Define research questions
- [ ] Develop search strategy
- [ ] Set up reference manager
- [ ] Complete protocol

**Week 198 (Search):**
- [ ] Execute all searches
- [ ] Deduplicate results
- [ ] Complete screening
- [ ] Document search log

**Week 199 (Reading):**
- [ ] Quality assessment
- [ ] Data extraction
- [ ] Detailed reading
- [ ] Paper summaries

**Week 200 (Synthesis):**
- [ ] Thematic analysis
- [ ] Concept mapping
- [ ] Annotated bibliography
- [ ] Review outline

---

## 10. Limitations and Bias

### 10.1 Potential Limitations

| Limitation | Mitigation Strategy |
|------------|---------------------|
| Language bias (English only) | Acknowledge in limitations |
| Database coverage | Use multiple databases |
| Publication bias | Include preprints |
| Single reviewer bias | Document all decisions transparently |
| Time constraints | Prioritize by relevance |

### 10.2 Threats to Validity

**Internal Validity:**
- Selection bias: Mitigated by explicit criteria
- Data extraction errors: Use standardized forms

**External Validity:**
- Generalizability: Limited to defined scope
- Temporal: Results reflect current state

**Construct Validity:**
- Definitions: Clearly defined terms
- Measurement: Standardized quality assessment

---

## 11. Reporting

### 11.1 Intended Output

**Primary Output:**
- [ ] Thesis chapter
- [ ] Standalone paper for publication
- [ ] Internal review document
- [ ] Other: ____________

**Target Venue (if publication):**
- Journal: ____________
- Conference: ____________

### 11.2 Reporting Standards

Following PRISMA 2020 guidelines:
- [ ] PRISMA checklist completed
- [ ] Flow diagram included
- [ ] All items reported

---

## 12. Appendices

### Appendix A: Search Log Template

| Date | Database | Query String | Results | Notes |
|------|----------|--------------|---------|-------|
| | | | | |

### Appendix B: Screening Decision Log

| Paper ID | Title | Stage 1 Decision | Stage 2 Decision | Reason |
|----------|-------|------------------|------------------|--------|
| | | | | |

### Appendix C: Quality Assessment Detailed Rubric

**Q1: Clear Research Objectives**
- 3: Objectives explicitly stated and well-defined
- 2: Objectives stated but could be clearer
- 1: Objectives implied but not explicit
- 0: No clear objectives

[Continue for each criterion...]

---

## Protocol Approval

**Prepared by:** ____________ **Date:** ____________

**Reviewed by:** ____________ **Date:** ____________

**Approved by:** ____________ **Date:** ____________

---

*This protocol is a living document. Any changes after initial approval should be documented in the version history with justification.*
