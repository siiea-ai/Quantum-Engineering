# Bibliography Database Template

## Reference Management System for PhD Thesis

---

## Overview

This template provides a structured approach to organizing your thesis bibliography. A well-organized reference database enables efficient citation during writing and ensures comprehensive coverage of the relevant literature.

---

## Reference Manager Configuration

### Recommended Folder Structure

```
Thesis_Bibliography/
├── All_References/
│   └── thesis_master.bib           # Complete merged bibliography
├── By_Chapter/
│   ├── ch1_introduction.bib
│   ├── ch2_background.bib
│   ├── ch3_research_one.bib
│   ├── ch4_research_two.bib
│   └── ch5_discussion.bib
├── By_Topic/
│   ├── quantum_mechanics.bib
│   ├── quantum_computing.bib
│   ├── quantum_error_correction.bib
│   ├── quantum_algorithms.bib
│   ├── [your_specific_topic].bib
│   └── methods_and_tools.bib
├── PDFs/
│   └── [organized PDF library]
└── Notes/
    └── [reading notes and summaries]
```

### Reference Manager Settings

| Setting | Recommended Value |
|---------|-------------------|
| Citation key format | [AuthorYear] or [Author:Year] |
| PDF auto-rename | Author_Year_Title.pdf |
| Duplicate detection | Enabled (by DOI/Title) |
| Auto-sync | Enabled with cloud backup |
| BibTeX export | UTF-8 encoding |

---

## Reference Categories

### Category Definitions

| Code | Category | Description | Target Count |
|------|----------|-------------|--------------|
| FND | Foundational | Classic texts, seminal papers | 15-25 |
| BGD | Background | General field context | 30-50 |
| R1 | Research 1 | Specific to first contribution | 40-60 |
| R2 | Research 2 | Specific to second contribution | 40-60 |
| REC | Recent | Papers from past 2-3 years | 20-30 |
| MTH | Methods | Tools, techniques, software | 15-25 |
| REV | Reviews | Review articles, surveys | 10-20 |

---

## BibTeX Entry Templates

### Journal Article

```bibtex
@article{AuthorYear,
    author    = {Last1, First1 and Last2, First2 and Last3, First3},
    title     = {Full Title of the Article},
    journal   = {Journal Name},
    volume    = {XX},
    number    = {X},
    pages     = {XXX--XXX},
    year      = {YYYY},
    doi       = {10.XXXX/XXXXXXX},
    url       = {https://doi.org/10.XXXX/XXXXXXX},
    note      = {},
    keywords  = {category:BGD, chapter:ch2},
}
```

### arXiv Preprint

```bibtex
@article{AuthorYearArxiv,
    author        = {Last1, First1 and Last2, First2},
    title         = {Full Title of the Preprint},
    year          = {YYYY},
    eprint        = {YYYY.XXXXX},
    archiveprefix = {arXiv},
    primaryclass  = {quant-ph},
    url           = {https://arxiv.org/abs/YYYY.XXXXX},
    keywords      = {category:REC, chapter:ch3},
}
```

### Conference Paper

```bibtex
@inproceedings{AuthorYearConf,
    author    = {Last1, First1 and Last2, First2},
    title     = {Full Title of the Conference Paper},
    booktitle = {Proceedings of the Conference Name (Abbrev)},
    pages     = {XXX--XXX},
    year      = {YYYY},
    publisher = {Publisher},
    address   = {City, Country},
    doi       = {10.XXXX/XXXXXXX},
    keywords  = {category:R1, chapter:ch3},
}
```

### Book

```bibtex
@book{AuthorYear,
    author    = {Last, First},
    title     = {Book Title},
    publisher = {Publisher Name},
    address   = {City},
    year      = {YYYY},
    edition   = {Xth},
    isbn      = {XXX-X-XXX-XXXXX-X},
    keywords  = {category:FND, chapter:ch2},
}
```

### Book Chapter

```bibtex
@incollection{AuthorYear,
    author    = {Last1, First1 and Last2, First2},
    title     = {Chapter Title},
    booktitle = {Book Title},
    editor    = {Editor Last, Editor First},
    publisher = {Publisher Name},
    address   = {City},
    year      = {YYYY},
    pages     = {XX--XX},
    chapter   = {X},
    keywords  = {category:BGD, chapter:ch2},
}
```

### PhD Thesis

```bibtex
@phdthesis{AuthorYear,
    author  = {Last, First},
    title   = {Thesis Title},
    school  = {University Name},
    year    = {YYYY},
    address = {City, State/Country},
    type    = {{PhD} Thesis},
    url     = {URL if available},
    keywords = {category:BGD, chapter:ch2},
}
```

### Technical Report

```bibtex
@techreport{AuthorYear,
    author      = {Last1, First1 and Last2, First2},
    title       = {Report Title},
    institution = {Institution Name},
    year        = {YYYY},
    number      = {Report Number},
    type        = {Technical Report},
    url         = {URL if available},
    keywords    = {category:MTH, chapter:appB},
}
```

### Software/Code

```bibtex
@software{AuthorYear,
    author    = {Last1, First1 and {Organization Name}},
    title     = {Software Name},
    version   = {X.X.X},
    year      = {YYYY},
    url       = {https://github.com/...},
    doi       = {10.XXXX/zenodo.XXXXXXX},
    keywords  = {category:MTH, chapter:ch3},
}
```

---

## Core References by Topic

### Quantum Mechanics Foundations

| Citation Key | Reference | Category |
|--------------|-----------|----------|
| Nielsen2010 | Nielsen & Chuang, "Quantum Computation and Quantum Information" | FND |
| Sakurai2017 | Sakurai & Napolitano, "Modern Quantum Mechanics" | FND |
| Preskill1998 | Preskill, "Lecture Notes on Quantum Computation" | FND |
| Wilde2017 | Wilde, "Quantum Information Theory" | FND |
| Watrous2018 | Watrous, "The Theory of Quantum Information" | FND |

### Quantum Computing

| Citation Key | Reference | Category |
|--------------|-----------|----------|
| Kitaev2002 | Kitaev, Shen, Vyalyi, "Classical and Quantum Computation" | FND |
| Mermin2007 | Mermin, "Quantum Computer Science" | FND |
| Kaye2007 | Kaye, Laflamme, Mosca, "Introduction to Quantum Computing" | FND |
| Rieffel2011 | Rieffel & Polak, "Quantum Computing: A Gentle Introduction" | BGD |
| Lidar2013 | Lidar & Brun, "Quantum Error Correction" | FND |

### [Your Research Area 1]

| Citation Key | Reference | Category |
|--------------|-----------|----------|
| | | R1 |
| | | R1 |
| | | R1 |

### [Your Research Area 2]

| Citation Key | Reference | Category |
|--------------|-----------|----------|
| | | R2 |
| | | R2 |
| | | R2 |

---

## Bibliography Workflow

### Adding New References

1. **Find the paper**
   - Search DOI, arXiv ID, or title

2. **Import metadata**
   - Use DOI import feature
   - Or export BibTeX from publisher

3. **Verify entry**
   - Check all fields are complete
   - Verify author name formatting
   - Add missing DOI/URL

4. **Categorize**
   - Add category keyword
   - Add chapter keyword
   - Add topic tags

5. **Attach PDF**
   - Download and link PDF
   - Rename per convention

6. **Add notes**
   - Brief summary if useful
   - Key takeaways for thesis

### Quality Control Checklist

For each reference:
- [ ] All required fields present
- [ ] Author names: "Last, First and Last, First" format
- [ ] Title: Proper capitalization (protect with braces if needed)
- [ ] Journal: Consistent abbreviation style
- [ ] Year: Four digits
- [ ] Pages: En-dash (--) between page numbers
- [ ] DOI: Present and valid
- [ ] URL: Working link
- [ ] No duplicate entries
- [ ] PDF attached and accessible
- [ ] Category and chapter keywords added

---

## Chapter Reference Mapping

### Chapter 1: Introduction

**Purpose**: Motivational references, field overview, high-level context

| Citation Key | Purpose in Chapter | Section |
|--------------|-------------------|---------|
| | Opening context | 1.1 |
| | Problem motivation | 1.2 |
| | Approach justification | 1.3 |

**Target count**: 15-25 references

### Chapter 2: Background

**Purpose**: Foundational material, technical background, related work

| Citation Key | Purpose in Chapter | Section |
|--------------|-------------------|---------|
| | QM foundations | 2.1 |
| | QC fundamentals | 2.2 |
| | Topic 1 background | 2.3 |
| | Topic 2 background | 2.4 |
| | Related work | 2.5 |

**Target count**: 50-80 references

### Chapter 3: [Research Contribution 1]

**Purpose**: Specific prior work, methodology sources, comparison baselines

| Citation Key | Purpose in Chapter | Section |
|--------------|-------------------|---------|
| | Problem context | 3.1 |
| | Method foundations | 3.3 |
| | Comparison work | 3.5 |

**Target count**: 40-60 references

### Chapter 4: [Research Contribution 2]

**Purpose**: Specific prior work, methodology sources, comparison baselines

| Citation Key | Purpose in Chapter | Section |
|--------------|-------------------|---------|
| | | |

**Target count**: 40-60 references

### Chapter 5: Discussion

**Purpose**: Contextualizing results, future directions, broader impact

| Citation Key | Purpose in Chapter | Section |
|--------------|-------------------|---------|
| | | |

**Target count**: 10-20 references

---

## Bibliography Maintenance

### Regular Tasks

| Task | Frequency | Last Done |
|------|-----------|-----------|
| Check for duplicates | Weekly | |
| Update preprints → published | Monthly | |
| Add recent papers | Monthly | |
| Verify URLs/DOIs | Before submission | |
| Export backup | Weekly | |

### Before Thesis Submission

- [ ] All entries complete and verified
- [ ] Consistent formatting throughout
- [ ] All DOIs valid and accessible
- [ ] No orphan citations (cited but not in bib)
- [ ] No unused references (in bib but not cited)
- [ ] BibTeX file compiles without errors
- [ ] Backup in multiple locations

---

## Statistics Tracking

| Metric | Current | Target |
|--------|---------|--------|
| Total references | | 150-300 |
| Foundational (FND) | | 15-25 |
| Background (BGD) | | 30-50 |
| Research 1 (R1) | | 40-60 |
| Research 2 (R2) | | 40-60 |
| Recent (REC) | | 20-30 |
| Methods (MTH) | | 15-25 |
| Reviews (REV) | | 10-20 |
| With PDFs | | 100% |
| Verified entries | | 100% |

---

## Notes

[Additional notes about your bibliography organization]

---

*Keep this database current throughout the thesis writing process. Regular maintenance prevents last-minute bibliography crises.*
