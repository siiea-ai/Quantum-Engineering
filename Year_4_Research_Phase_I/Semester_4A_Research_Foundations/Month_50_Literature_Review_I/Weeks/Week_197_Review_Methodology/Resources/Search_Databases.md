# Academic Database Navigation Guide

## Overview

Comprehensive literature search requires strategic use of multiple academic databases. Each database has unique strengths, coverage, and search syntax. This guide provides detailed navigation instructions for the key databases used in quantum computing research.

---

## Part 1: Database Landscape

### Coverage by Database

| Database | Coverage | Strengths | Access |
|----------|----------|-----------|--------|
| **arXiv** | Physics, CS, Math preprints | Most current, free | Open |
| **Google Scholar** | Broad academic | Comprehensive, citations | Open |
| **Web of Science** | Curated journals | Quality metrics, citation network | Institutional |
| **IEEE Xplore** | Engineering, CS | Conference proceedings | Institutional |
| **ACM Digital Library** | Computing | SIGCOMM, theory | Institutional |
| **Semantic Scholar** | Broad + AI features | AI-powered recommendations | Open |
| **PubMed** | Biomedical | Medical applications | Open |
| **INSPIRE-HEP** | High-energy physics | Particle physics QC | Open |

### Recommended Search Order for Quantum Computing

```
1. arXiv quant-ph       # Latest preprints
2. Google Scholar       # Broad coverage
3. IEEE Xplore          # Engineering/implementation
4. Semantic Scholar     # AI recommendations
5. Web of Science       # Citation analysis
6. ACM DL               # Theory/algorithms
```

---

## Part 2: arXiv (arxiv.org)

### Overview

arXiv is the primary preprint server for physics and computer science. For quantum computing, it's essential for accessing the most current research before formal publication.

### Relevant Categories

| Category | Focus | URL |
|----------|-------|-----|
| **quant-ph** | Quantum physics | arxiv.org/list/quant-ph |
| **cond-mat** | Condensed matter | arxiv.org/list/cond-mat |
| **cs.ET** | Emerging tech | arxiv.org/list/cs.ET |
| **cs.IT** | Information theory | arxiv.org/list/cs.IT |
| **cs.LG** | Machine learning | arxiv.org/list/cs.LG |

### Search Syntax

**Basic Search:**
```
quantum error correction decoder
```

**Field-Specific Search:**
```
ti:"surface code"              # Title contains
au:Preskill                    # Author
abs:"fault tolerant"           # Abstract contains
all:"neural network decoder"   # All fields
```

**Combined Search:**
```
ti:"quantum error correction" AND au:Terhal AND abs:threshold
```

**Date Filters:**
```
# In advanced search interface
Date: from 2020-01 to 2024-12
```

### Advanced Search Interface

Navigate to: [arxiv.org/search/advanced](https://arxiv.org/search/advanced)

**Fields:**
- Title, Author, Abstract, Comments, Journal reference
- Subject category (e.g., quant-ph)
- Date range

### API Access (for automation)

```python
# arXiv API Example
import arxiv

search = arxiv.Search(
    query="quantum error correction neural",
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for paper in search.results():
    print(f"{paper.title} - {paper.published}")
```

### Best Practices

1. **Check daily:** New quant-ph papers every day
2. **Use arXiv ID:** Cite as arXiv:XXXX.XXXXX
3. **Check for updates:** Papers may have multiple versions
4. **Link to published:** Add DOI when paper is published

---

## Part 3: Google Scholar (scholar.google.com)

### Overview

Google Scholar provides the broadest coverage of academic literature, including preprints, theses, books, and gray literature.

### Search Syntax

**Basic Search:**
```
"quantum error correction" decoder performance
```

**Phrase Search:**
```
"surface code decoder"         # Exact phrase
```

**Exclusion:**
```
quantum computing -classical   # Exclude term
```

**Author Search:**
```
author:"John Preskill"         # Specific author
author:preskill                # Last name
```

**Publication Search:**
```
source:"Physical Review"       # Journal/conference
```

**Date Range:**
```
# Use sidebar filters or custom range
quantum computing 2020..2024
```

### Advanced Search

Navigate to: [scholar.google.com/advanced_search](https://scholar.google.com/)

**Options:**
- With all of the words
- With the exact phrase
- With at least one of the words
- Without the words
- Where my words occur (title/article)
- Return articles authored by
- Return articles published in
- Return articles dated between

### Citation Features

**Finding Related Work:**
1. **Cited by:** Find papers citing this work
2. **Related articles:** Similar papers by content
3. **Versions:** Find all versions (preprint, published)

**Setting Up Alerts:**
```
1. Search for topic
2. Click "Create alert" (envelope icon)
3. Set email frequency
4. Receive updates when new papers match
```

### Library Integration

**Link to Zotero:**
```
1. Settings → Library links
2. Search for Zotero
3. Add to library links
4. See "Import into Zotero" links
```

### Limitations

- No structured metadata export
- Includes low-quality sources
- Citation counts can be manipulated
- No controlled vocabulary

---

## Part 4: Web of Science (webofscience.com)

### Overview

Web of Science (Clarivate) is a curated citation database covering high-quality peer-reviewed journals. Excellent for citation analysis and impact metrics.

### Access

Requires institutional subscription. Check your university library.

### Search Syntax

**Topic Search:**
```
TS="quantum error correction"  # Topic (title, abstract, keywords)
```

**Title Search:**
```
TI="surface code"              # Title only
```

**Author Search:**
```
AU=Preskill J                  # Author
```

**Organization:**
```
OG=MIT                         # Organization
```

**Combined:**
```
TS="quantum computing" AND AU=Kitaev AND PY=2020-2024
```

### Boolean Operators

```
AND: Both terms
OR: Either term
NOT: Exclude term
NEAR/n: Terms within n words
SAME: Terms in same sentence
```

**Wildcards:**
```
* : Zero or more characters (comput* = computing, computer)
? : Single character (wom?n = woman, women)
$ : Zero or one character (behavior$ = behavior, behaviour)
```

### Advanced Features

**Citation Report:**
1. Run search
2. Click "Create Citation Report"
3. View h-index, citation trends, citing articles

**Analyze Results:**
- By author
- By institution
- By country
- By publication year
- By source title

### Export Options

```
Export → Select format:
- Plain text
- BibTeX
- RIS (for Zotero)
- Excel
Records: Full Record + Cited References
```

---

## Part 5: IEEE Xplore (ieeexplore.ieee.org)

### Overview

IEEE Xplore covers engineering and computer science literature, particularly strong for conference proceedings and standards.

### Relevant Content for Quantum Computing

- **Journals:** IEEE TQE, IEEE Access
- **Conferences:** ISCA, MICRO, QCE, DAC
- **Standards:** IEEE quantum computing standards

### Search Syntax

**Basic:**
```
"quantum error correction"
```

**Controlled Vocabulary (IEEE Thesaurus):**
```
"Quantum computing"            # Controlled term
```

**Field Search:**
```
("Document Title":"surface code")
("Author":Gambetta)
("Abstract":threshold)
```

### Command Search

```
("quantum error correction" OR QEC)
AND ("neural network" OR "machine learning")
AND (decoder OR decoding)
```

### Filters

- **Content Type:** Journals, Conferences, Standards, Books
- **Year:** Date range
- **Publisher:** IEEE, IET, etc.
- **Topic:** Browse IEEE taxonomy

### Export

```
1. Check papers to export
2. Click "Download Citations"
3. Format: BibTeX, RIS, CSV
4. Citation + Abstract
```

---

## Part 6: Semantic Scholar (semanticscholar.org)

### Overview

AI-powered academic search engine with features like TLDR summaries, citation intent, and influence scores.

### Unique Features

**TLDR Summaries:**
- AI-generated one-sentence summaries
- Quick paper scanning

**Citation Intent:**
- Background, Method, Result citations
- Understand how papers cite each other

**Influence Score:**
- Paper impact beyond raw citations
- Considers citation context

### Search Syntax

**Basic:**
```
quantum error correction decoder
```

**Year Filter:**
```
quantum computing year:2020-2024
```

**Author:**
```
author:Preskill quantum
```

**Venue:**
```
venue:Nature quantum
```

### Research Feeds

```
1. Create account
2. Add papers to library
3. Receive personalized recommendations
4. Set up topic alerts
```

### API Access

```python
# Semantic Scholar API
import requests

query = "quantum error correction"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}"
response = requests.get(url)
papers = response.json()
```

---

## Part 7: ACM Digital Library (dl.acm.org)

### Overview

ACM DL covers computing literature comprehensively, particularly theory and systems.

### Relevant Content

- **Journals:** JACM, TALG, TQC
- **Conferences:** STOC, FOCS, QIP, POPL
- **SIGs:** Quantum computing proceedings

### Search Syntax

**Basic:**
```
"quantum algorithm"
```

**Field Search:**
```
Title:("quantum computing")
Author:(Aaronson)
Abstract:(complexity)
```

**ACM Classification:**
```
CCS:("Quantum computation")    # ACM Computing Classification System
```

### Filters

- Publication type
- Publication year
- ACM publication
- Conference/journal

---

## Part 8: Specialized Databases

### INSPIRE-HEP (inspirehep.net)

For quantum computing related to high-energy physics:

```
find t "quantum simulation" and date > 2020
find a Preskill
```

### PubMed (pubmed.ncbi.nlm.nih.gov)

For quantum computing in biomedical applications:

```
"quantum computing"[Title/Abstract] AND "drug discovery"
```

### Scopus (scopus.com)

Alternative to Web of Science with different coverage:

```
TITLE-ABS-KEY("quantum error correction")
```

---

## Part 9: Search Strategy Construction

### Step 1: Identify Key Concepts

```
Research Question:
"How do neural network decoders compare to MWPM for surface codes?"

Key Concepts:
1. Quantum error correction / QEC
2. Surface codes / topological codes
3. Neural network / machine learning
4. Decoder / decoding
5. MWPM / minimum weight
6. Performance / threshold
```

### Step 2: Find Synonyms and Related Terms

```
Concept 1: QEC
- "quantum error correction"
- "quantum error-correcting"
- QEC
- "fault tolerant"
- "fault-tolerant"

Concept 2: Surface codes
- "surface code"
- "surface codes"
- "topological code"
- "toric code"

Concept 3: Machine learning
- "machine learning"
- "neural network"
- "deep learning"
- "reinforcement learning"

Concept 4: Decoder
- decoder
- decoding
- decode
- "error correction algorithm"
```

### Step 3: Construct Boolean Query

```
("quantum error correction" OR QEC OR "fault tolerant")
AND
("surface code" OR "topological code" OR "toric code")
AND
("machine learning" OR "neural network" OR "deep learning")
AND
(decoder OR decoding)
```

### Step 4: Adapt for Each Database

**arXiv:**
```
ti:"surface code" AND abs:"neural network" AND abs:decoder
```

**Google Scholar:**
```
"surface code" "neural network" decoder
```

**Web of Science:**
```
TS=("surface code") AND TS=("neural network" OR "machine learning")
AND TS=(decoder)
```

### Step 5: Document Everything

Create search log:

| Date | Database | Query | Results | Notes |
|------|----------|-------|---------|-------|
| 2026-02-10 | arXiv | ti:"surface code" decoder | 234 | Good coverage |
| 2026-02-10 | Google Scholar | "surface code" decoder | 12,400 | Too broad |

---

## Part 10: Citation Tracking

### Forward Snowballing

Find papers that cite your seed papers:

```
Google Scholar: Click "Cited by X"
Web of Science: Click "Cited by" count
Semantic Scholar: View "Citations" tab
```

### Backward Snowballing

Check references of included papers:

```
1. Review reference list
2. Identify relevant citations
3. Add to search results
4. Iterate until saturation
```

### Citation Networks

**Connected Papers (connectedpapers.com):**
1. Enter paper DOI or title
2. View visual citation graph
3. Find related work not caught by keyword search

**Research Rabbit (researchrabbit.ai):**
1. Create collection
2. Add seed papers
3. Get AI recommendations
4. Track new publications

---

## Part 11: Alerts and Updates

### Set Up Alerts

**arXiv:**
- Subscribe to RSS feeds by category
- Use arXiv-sanity or similar tools

**Google Scholar:**
- Create alerts for queries
- Follow specific authors

**Semantic Scholar:**
- Follow research feeds
- Set topic alerts

**Web of Science:**
- Save searches with alerts
- Citation alerts for key papers

### Regular Monitoring Schedule

```
Daily: arXiv new submissions
Weekly: Google Scholar alerts
Monthly: Comprehensive search refresh
Quarterly: Snowballing from new key papers
```

---

## Quick Reference: Database Search Templates

### arXiv
```
(ti:"[concept1]" OR abs:"[concept1]")
AND (ti:"[concept2]" OR abs:"[concept2]")
Category: quant-ph
```

### Google Scholar
```
"[concept1]" "[concept2]" [additional terms]
```

### Web of Science
```
TS=("[concept1]" OR "[synonym1]")
AND TS=("[concept2]" OR "[synonym2]")
AND PY=2020-2024
```

### IEEE Xplore
```
(("[concept1]") AND ("[concept2]"))
Filters: Conference + Journals, Date range
```

### Semantic Scholar
```
[concept1] [concept2] year:2020-2024
```

---

## Troubleshooting

### Too Many Results
1. Add more specific terms
2. Narrow date range
3. Add field restrictions (title only)
4. Focus on high-quality venues

### Too Few Results
1. Remove restrictive terms
2. Add synonyms with OR
3. Search additional databases
4. Try snowballing from known papers

### Missing Key Papers
1. Verify search includes author terms
2. Check for variant spellings
3. Search by arXiv ID or DOI directly
4. Use citation tracking

---

*"The quality of your search determines the quality of your review."*
