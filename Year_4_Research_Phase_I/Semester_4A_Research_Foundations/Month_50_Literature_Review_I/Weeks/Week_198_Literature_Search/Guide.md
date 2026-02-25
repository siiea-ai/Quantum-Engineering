# Comprehensive Literature Search: Complete Execution Guide

## Introduction

This guide provides detailed instructions for executing a comprehensive literature search as part of your systematic review. It covers practical techniques for searching, screening, and organizing academic literature in quantum computing research.

---

## Part 1: Search Execution Framework

### The Search Process

```
Protocol Queries → Database Execution → Result Collection → Screening → Organization
```

### Pre-Search Checklist

Before beginning your search:

- [ ] Review protocol research questions
- [ ] Finalize search strings
- [ ] Set up Zotero folder structure
- [ ] Create search log template
- [ ] Clear schedule for focused work

### Search Order Rationale

**Recommended Order:**
1. **arXiv first:** Most current, field-specific
2. **Google Scholar second:** Broadest coverage
3. **IEEE/ACM third:** Engineering perspective
4. **Web of Science fourth:** Citation metrics
5. **Snowballing last:** Fills gaps

---

## Part 2: arXiv Search Mastery

### Understanding arXiv Structure

**Primary Categories for Quantum Computing:**
- `quant-ph`: Quantum Physics (primary)
- `cond-mat.mes-hall`: Mesoscopic Systems (hardware)
- `cs.ET`: Emerging Technologies
- `cs.IT`: Information Theory
- `cs.LG`: Machine Learning (cross-listed)

### arXiv Search Interface

**Basic Search:**
```
1. Go to arxiv.org/search
2. Enter search terms
3. Select "Advanced Search" for more control
```

**Advanced Search Fields:**
```
Title: ti:"surface code"
Author: au:Preskill
Abstract: abs:"neural network"
All fields: all:"quantum error correction"
Category: cat:quant-ph
```

### Constructing arXiv Queries

**Example: ML-based QEC Decoders**

```
Query 1 (Narrow):
ti:"surface code" AND abs:"neural network" AND abs:decoder

Query 2 (Broader):
(ti:"quantum error correction" OR ti:QEC) AND
(abs:"machine learning" OR abs:"neural") AND
abs:decod*

Query 3 (By category):
cat:quant-ph AND abs:"decoder" AND abs:"learning"
```

### arXiv Search Tips

1. **Use multiple queries:** Start narrow, broaden if needed
2. **Check cross-listings:** Papers may be in cs.LG but cross-listed to quant-ph
3. **Note versions:** v1, v2, etc. - check for latest
4. **Export properly:** Use "Export BibTeX" for clean import

### arXiv to Zotero Workflow

```
1. Search arXiv
2. Open paper page (not PDF)
3. Click Zotero connector
4. Paper + PDF auto-imported
5. Add to appropriate folder
6. Tag with "source:arxiv"
```

### Tracking arXiv Papers

| arXiv ID | Title | Authors | Date | Category | Screened | Decision |
|----------|-------|---------|------|----------|----------|----------|
| 2301.12345 | Example | Smith | 2023-01 | quant-ph | Yes | Include |

---

## Part 3: Google Scholar Strategies

### Scholar Search Syntax

**Phrase Search:**
```
"quantum error correction"    # Exact phrase
```

**Exclusion:**
```
quantum computing -chemistry  # Exclude term
```

**Site-Specific:**
```
site:arxiv.org "surface code" # Only arXiv
```

**Author Search:**
```
author:"John Preskill"
```

**Date Range:**
```
"quantum computing" 2020..2024
```

### Advanced Scholar Techniques

**Finding Highly-Cited Work:**
1. Search topic
2. Sort by citations (default)
3. Check "Cited by X" for recent followers

**Building from Seed Paper:**
1. Find key paper
2. Click "Cited by"
3. Review citing papers
4. Click "Related articles"

**Using Scholar Library:**
1. Save papers to library
2. Organize into labels
3. Export for Zotero import

### Scholar Limitations and Workarounds

**No Direct Export:**
- Use "Import to Zotero" link (if configured)
- Copy citations and import manually
- Use Zotero connector on paper landing pages

**Quality Filtering:**
- Scholar includes everything (theses, presentations)
- Apply stricter screening criteria
- Verify venue quality manually

### Scholar to Zotero Workflow

```
1. Search Google Scholar
2. Click paper title to go to source
3. Use Zotero connector on source page
4. Better metadata than from Scholar page
5. Tag with "source:scholar"
```

---

## Part 4: IEEE Xplore Navigation

### IEEE Search Interface

**Basic Search:**
```
1. Go to ieeexplore.ieee.org
2. Enter search terms
3. Use filters for content type
```

**Command Search:**
```
("Document Title":"quantum error correction")
AND
("Abstract":"neural network")
```

### IEEE Filters

- **Content Type:** Journals, Conferences, Standards
- **Publisher:** IEEE, IET
- **Year:** Date range
- **Topic:** IEEE taxonomy

### Key IEEE Venues for Quantum Computing

**Journals:**
- IEEE Transactions on Quantum Engineering (TQE)
- IEEE Access (quantum computing section)

**Conferences:**
- IEEE International Conference on Quantum Computing and Engineering (QCE)
- Design Automation Conference (DAC) - quantum track
- International Symposium on Computer Architecture (ISCA)

### IEEE to Zotero Workflow

```
1. Search IEEE Xplore
2. Select relevant papers (checkbox)
3. Click "Download Citations"
4. Choose BibTeX format
5. Import BibTeX to Zotero
6. Tag with "source:ieee"
```

---

## Part 5: Web of Science and Citation Analysis

### WoS Search Syntax

```
Topic Search:
TS="quantum error correction"

Title Search:
TI="surface code"

Author Search:
AU=Gottesman D

Combined:
TS="quantum computing" AND TS="machine learning" AND PY=2020-2024
```

### Citation Analysis Features

**Citation Report:**
1. Run search
2. Click "Create Citation Report"
3. View: total citations, h-index, citing articles

**Analyze Results:**
- By author
- By institution
- By year
- By source journal

### Using WoS for Quality Assessment

**Identifying High-Impact Papers:**
1. Sort by "Times Cited"
2. Check Journal Impact Factor
3. Note author h-index

**Finding Emerging Papers:**
1. Sort by "Usage Count"
2. Recent papers with high downloads
3. May indicate important new work

### WoS to Zotero Workflow

```
1. Search Web of Science
2. Select papers (checkbox)
3. Export → RIS format
4. Include "Full Record + Cited References"
5. Import RIS to Zotero
6. Tag with "source:wos"
```

---

## Part 6: Supplementary Databases

### Semantic Scholar

**Unique Features:**
- AI-generated TLDR summaries
- Citation intent classification
- Influence scores

**Search:**
```
1. semanticscholar.org
2. Enter search terms
3. Use filters (year, venue)
4. Check "Highly Influential Citations"
```

### ACM Digital Library

**Focus:** Computing theory and systems

**Search:**
```
1. dl.acm.org
2. Advanced search
3. Filter by publication type
4. Export BibTeX
```

**Key Venues:**
- STOC, FOCS (theory)
- SIGCOMM, POPL (systems)
- ACM Computing Surveys (reviews)

### Connected Papers

**Visual Discovery:**
```
1. connectedpapers.com
2. Enter paper title or DOI
3. View citation graph
4. Find related papers not found by keywords
```

### Research Rabbit

**AI Recommendations:**
```
1. researchrabbit.ai
2. Create collection
3. Add seed papers
4. Get personalized recommendations
```

---

## Part 7: Snowballing Techniques

### Forward Snowballing

**Definition:** Finding papers that cite your seed papers

**Process:**
```
1. Select seed papers (most relevant from search)
2. For each seed paper:
   a. Go to Google Scholar
   b. Click "Cited by X"
   c. Review citing papers
   d. Add relevant ones to collection
3. Iterate with new inclusions
4. Stop when no new relevant papers found
```

**Documenting Forward Snowballing:**

| Seed Paper | Citing Paper | Decision | Reason |
|------------|--------------|----------|--------|
| Smith 2023 | Jones 2024 | Include | Extends method |
| Smith 2023 | Lee 2023 | Exclude | Different application |

### Backward Snowballing

**Definition:** Checking references of included papers

**Process:**
```
1. For each included paper:
   a. Review reference list
   b. Identify relevant citations
   c. Check if already in collection
   d. Add new relevant papers
2. Iterate with new inclusions
3. Stop when saturation reached
```

**Documenting Backward Snowballing:**

| Source Paper | Referenced Paper | Decision | Reason |
|--------------|------------------|----------|--------|
| Smith 2023 | Brown 2019 | Include | Foundational |
| Smith 2023 | Wilson 2018 | Exclude | Out of scope |

### Snowballing Saturation

**Signs of Saturation:**
- Same papers appearing repeatedly
- No new relevant papers discovered
- References leading to excluded papers
- Coverage feels complete

**Typical Iterations:**
- Initial seed: 10-15 papers
- Round 1: +20-30 papers
- Round 2: +10-15 papers
- Round 3: +5-10 papers
- Saturation usually by round 3-4

---

## Part 8: Screening and Selection

### Two-Stage Screening Process

**Stage 1: Title/Abstract Screening**

```
For each paper:
1. Read title
2. Skim abstract
3. Quick decision: Include / Exclude / Uncertain
4. Uncertain → Stage 2

Time per paper: 1-2 minutes
```

**Stage 1 Decision Criteria:**
- Include: Clearly relevant to RQs
- Exclude: Clearly out of scope
- Uncertain: Cannot decide from title/abstract

**Stage 2: Full-Text Screening**

```
For uncertain papers:
1. Obtain full text
2. Read introduction and conclusion
3. Skim methods and results
4. Apply full inclusion/exclusion criteria
5. Final decision: Include / Exclude

Time per paper: 10-15 minutes
```

### Documenting Screening Decisions

**Screening Log Template:**

| Paper ID | Title | Stage 1 | Reason | Stage 2 | Final | Notes |
|----------|-------|---------|--------|---------|-------|-------|
| P001 | "Surface code..." | Include | Relevant | - | Include | Core paper |
| P002 | "Quantum algo..." | Exclude | Wrong topic | - | Exclude | - |
| P003 | "ML decoder..." | Uncertain | Need full text | Include | Include | Methods relevant |

### Common Screening Mistakes

**Avoid:**
1. Reading full paper at Stage 1 (inefficient)
2. Excluding based on unfamiliar terminology
3. Including everything "just in case"
4. Not documenting exclusion reasons
5. Inconsistent criteria application

---

## Part 9: Deduplication

### Sources of Duplicates

- Same paper in multiple databases
- arXiv preprint + published version
- Conference paper + journal extension
- Multiple exports of same search

### Deduplication Process

**Step 1: Automatic Detection**
```
Zotero:
1. Library → Duplicate Items
2. Review each duplicate set
3. Merge or separate as appropriate
```

**Step 2: Manual Verification**
```
For potential duplicates:
1. Check DOI/arXiv ID
2. Compare titles exactly
3. Check author lists
4. Verify same paper or different versions
```

**Step 3: Handling Versions**
```
If preprint + published:
- Keep published version as primary
- Note arXiv ID in "Extra" field
- Don't double-count in statistics
```

### Deduplication Best Practices

1. **Deduplicate regularly:** Don't wait until end
2. **Use DOI matching:** Most reliable identifier
3. **Keep better metadata:** When merging, keep more complete record
4. **Document process:** Note how many duplicates removed

---

## Part 10: Organization and Prioritization

### Folder Structure for Collected Papers

```
Literature_Search/
├── All_Papers/                    # Complete deduplicated set
├── By_Source/
│   ├── arXiv/
│   ├── Google_Scholar/
│   ├── IEEE/
│   └── Snowballing/
├── By_Priority/
│   ├── Tier_1_Core/
│   ├── Tier_2_Important/
│   └── Tier_3_Supporting/
├── Excluded/
│   ├── Out_of_Scope/
│   ├── Duplicate/
│   └── No_Access/
└── To_Screen/
```

### Priority Assignment Criteria

**Tier 1: Core Papers (Read in Week 199)**
- Score 5 points:
  - [ ] Directly answers a research question (+2)
  - [ ] Highly cited (>50 citations) (+1)
  - [ ] From leading research group (+1)
  - [ ] Introduces novel method relevant to your work (+1)

**Tier 2: Important Papers**
- Score 3-4 points

**Tier 3: Supporting Papers**
- Score 1-2 points

### Tagging System for Organization

```
Priority Tags:
- tier:1, tier:2, tier:3

Source Tags:
- source:arxiv, source:scholar, source:ieee, source:wos, source:snowball

Status Tags:
- status:to_screen, status:screened, status:included, status:excluded

Content Tags:
- topic:surface_code, topic:ml_decoder, topic:threshold
```

---

## Part 11: Documentation and Reproducibility

### Complete Search Log

| Field | Description |
|-------|-------------|
| Date | When search was executed |
| Database | Which database |
| Query | Exact query string used |
| Filters | Any filters applied |
| Results | Number of results |
| Screened | Number screened |
| Included | Number included |
| Excluded | Number excluded |
| Notes | Any observations |

### PRISMA Flow Data Collection

Track these numbers throughout the week:

```
Identification:
- Records from databases: ____
  - arXiv: ____
  - Scholar: ____
  - IEEE: ____
  - WoS: ____
  - Other: ____
- Records from other sources: ____
  - Forward snowballing: ____
  - Backward snowballing: ____
  - Recommendations: ____

Screening:
- Records after duplicates removed: ____
- Records screened: ____
- Records excluded at screening: ____

Eligibility:
- Full-text articles assessed: ____
- Full-text articles excluded: ____
  - Reason 1: ____ (n=__)
  - Reason 2: ____ (n=__)
  - Reason 3: ____ (n=__)

Included:
- Studies included in review: ____
```

---

## Part 12: Troubleshooting

### Search Issues

**Too many results (>1000):**
```
Solutions:
1. Add more specific terms
2. Use title-only search
3. Narrow date range
4. Add mandatory terms (AND)
5. Focus on specific venues
```

**Too few results (<20):**
```
Solutions:
1. Add synonyms (OR)
2. Use broader terms
3. Remove restrictive filters
4. Search more databases
5. Increase snowballing
```

**Missing expected papers:**
```
Solutions:
1. Search by author name
2. Search by exact title
3. Check spelling variants
4. Try different databases
5. Check paper access restrictions
```

### Organization Issues

**Zotero not syncing:**
```
Solutions:
1. Check internet connection
2. Verify account credentials
3. Check storage quota
4. Restart Zotero
5. Sync manually
```

**Duplicate entries appearing:**
```
Solutions:
1. Check "Duplicate Items" folder
2. Merge duplicates properly
3. Verify import settings
4. Use DOI-based deduplication
```

---

## Week Summary Checklist

### By End of Day 1380 (arXiv)
- [ ] All arXiv queries executed
- [ ] Results imported to Zotero
- [ ] Initial screening complete
- [ ] Search log updated

### By End of Day 1381 (Scholar)
- [ ] Scholar searches complete
- [ ] Results imported
- [ ] Alerts set up
- [ ] Cross-referenced with arXiv

### By End of Day 1382 (IEEE/ACM)
- [ ] IEEE searches complete
- [ ] ACM searches complete
- [ ] Results imported
- [ ] Engineering perspective captured

### By End of Day 1383 (WoS)
- [ ] WoS searches complete
- [ ] Citation analysis done
- [ ] Key papers identified
- [ ] Metrics documented

### By End of Day 1384 (Snowballing)
- [ ] Forward snowballing complete
- [ ] Backward snowballing complete
- [ ] Saturation reached
- [ ] New papers added

### By End of Day 1385 (Deduplication)
- [ ] All sources merged
- [ ] Duplicates removed
- [ ] Metadata cleaned
- [ ] Statistics compiled

### By End of Day 1386 (Organization)
- [ ] Final screening complete
- [ ] Priority tiers assigned
- [ ] Week documentation complete
- [ ] Ready for Week 199

---

## Key Metrics for Week 198

| Metric | Target | Actual |
|--------|--------|--------|
| Total papers identified | 300-500 | |
| Papers after deduplication | 200-400 | |
| Papers after screening | 100-150 | |
| Tier 1 (Core) papers | 10-15 | |
| Tier 2 (Important) papers | 30-40 | |
| Tier 3 (Supporting) papers | 50-80 | |
| Databases searched | 5+ | |
| Snowballing iterations | 2-3 | |

---

*"Comprehensive search is the foundation of credible synthesis."*
