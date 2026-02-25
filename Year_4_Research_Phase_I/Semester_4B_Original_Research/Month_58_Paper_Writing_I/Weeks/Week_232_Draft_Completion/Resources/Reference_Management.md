# Reference Management Guide

## Introduction

Proper reference management is essential for academic credibility, avoiding plagiarism, and enabling readers to find source materials. This guide covers reference collection, organization, formatting, and verification for physics publications.

## Part I: Citation Management Tools

### Recommended Software

**Zotero (Recommended)**
- Free, open-source
- Browser integration for easy capture
- BibTeX export for LaTeX
- Collaborative libraries
- Syncs across devices

**Mendeley**
- Free (with storage limits)
- PDF annotation features
- Social features for discovery
- Good mobile app

**EndNote**
- Institutional license often available
- Mature software, wide format support
- Better for large libraries
- Steeper learning curve

**BibDesk (Mac Only)**
- Lightweight BibTeX manager
- Good for LaTeX-focused workflows
- Free, open-source

### Setting Up Your Library

1. **Create project-specific library** for each paper
2. **Import references** from databases (Google Scholar, PubMed, arXiv)
3. **Attach PDFs** for easy access
4. **Add notes** with key points from each paper
5. **Tag by topic** for organization

### Key Fields to Complete

For each reference, ensure these fields are filled:

| Field | Importance | Source |
|-------|------------|--------|
| Authors | Essential | Paper |
| Title | Essential | Paper |
| Journal | Essential | Paper |
| Volume | Essential | Paper |
| Pages | Essential | Paper/DOI |
| Year | Essential | Paper |
| DOI | Highly useful | Paper/DOI.org |
| arXiv ID | For preprints | arXiv |
| URL | For access | Publisher |

## Part II: Finding and Evaluating References

### Search Strategies

**Google Scholar**
- Broad coverage
- Citation counts
- "Cited by" for finding related work
- Easy BibTeX export

**arXiv**
- Physics preprints
- Often first publication venue
- Track preprints to published versions

**Web of Science/Scopus**
- Curated, high-quality
- Better citation tracking
- Usually requires institutional access

**Publisher Sites**
- Physical Review: journals.aps.org
- Nature: nature.com
- Science: science.org

### Evaluating References

**Use This Reference If:**
- Primary source for method/result you cite
- Recent, peer-reviewed publication
- Highly cited (for foundational claims)
- Directly relevant to your work

**Be Cautious If:**
- Only source for critical claim
- Non-peer-reviewed (preprints for significant claims)
- Very old (for current state-of-art claims)
- From predatory journals

### Preprint vs. Published

**Citing Preprints:**
- Acceptable for recent work not yet published
- Update citations when paper is published before submission
- Indicate arXiv explicitly: "arXiv:XXXX.XXXXX"

**Tracking Publications:**
- Set alerts for preprints you've cited
- Check before final submission
- Update reference with published details

## Part III: Citation Practices

### When to Cite

| Situation | Citation Needed |
|-----------|-----------------|
| Specific claim from literature | Yes |
| Established method | Yes (original paper) |
| General knowledge in field | Not always |
| Your own prior work | Yes (if relevant) |
| Software/databases | Yes (check citation format) |
| Personal communication | Acknowledge, not reference |

### How Many References

| Paper Type | Typical Range |
|------------|---------------|
| Letter (PRL) | 25-40 |
| Regular article (PRA) | 40-60 |
| Long article (PRX) | 50-80 |
| Review | 100-300 |

### Citation Placement

**After Statement:**
```
"High-fidelity gates have been achieved in multiple systems [1-5]."
```

**After Specific Claim:**
```
"The highest reported fidelity of 99.9% [3] approaches the
fault-tolerant threshold."
```

**Author Mentioned:**
```
"The framework developed by Smith and Jones [7] provides..."
```

### Citation Ethics

**Do:**
- Cite sources you have read
- Cite primary sources when possible
- Include diverse sources (not just your group)
- Update preprint citations
- Credit competing approaches fairly

**Don't:**
- Cite papers you haven't read
- Cite only your advisor's work
- Omit directly competing work
- Inflate citations for impact
- Copy citation lists from other papers

## Part IV: Formatting References

### APS Style (Physical Review)

**Journal Article:**
```
A. B. Author, C. D. Author, and E. F. Author, Journal Name Vol, Page (Year).
```

**Examples:**
```
J. Koch, T. M. Yu, J. Gambetta, A. A. Houck, D. I. Schuster, J. Majer,
A. Blais, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf,
Phys. Rev. A 76, 042319 (2007).

A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland,
Phys. Rev. A 86, 032324 (2012).
```

**arXiv:**
```
A. B. Author and C. D. Author, arXiv:XXXX.XXXXX (Year).
```

**Book:**
```
A. B. Author, Book Title (Publisher, City, Year).
```

**Book Chapter:**
```
A. B. Author, in Book Title, edited by E. F. Editor (Publisher, City, Year), p. XXX.
```

### BibTeX for LaTeX

**Journal Article:**
```bibtex
@article{Koch2007,
  author = {Koch, J. and Yu, T. M. and Gambetta, J. and others},
  title = {Charge-insensitive qubit design derived from the Cooper pair box},
  journal = {Phys. Rev. A},
  volume = {76},
  pages = {042319},
  year = {2007},
  doi = {10.1103/PhysRevA.76.042319}
}
```

**arXiv Preprint:**
```bibtex
@misc{Author2024,
  author = {Author, A. B. and Author, C. D.},
  title = {Paper Title},
  year = {2024},
  eprint = {2401.12345},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph}
}
```

### Common Formatting Issues

| Issue | Solution |
|-------|----------|
| Inconsistent author format | Use "First M. Last" consistently |
| Missing page numbers | Check DOI; use article number if needed |
| Wrong journal abbreviation | Use journal's official abbreviation |
| Preprint cited as published | Update from arXiv to journal |
| Broken DOI links | Verify at doi.org |

## Part V: Verification Process

### Before Submission Checklist

**For Each Reference:**
- [ ] Author names correct (check accents, initials)
- [ ] Title accurate
- [ ] Journal name correctly abbreviated
- [ ] Volume and pages correct
- [ ] Year correct
- [ ] DOI works (if included)

**Overall:**
- [ ] All citations in text have reference entries
- [ ] No unused references in list
- [ ] Numbering is correct (no gaps)
- [ ] Format consistent throughout

### Common Errors to Catch

| Error | How to Check |
|-------|--------------|
| Duplicate references | Sort alphabetically, check |
| Missing references | Search document for uncited claims |
| Wrong page numbers | Cross-check with DOI |
| Outdated preprints | Check if published |
| Incomplete author lists | Verify against paper |

### Final Verification

1. Export reference list from management software
2. Compile LaTeX document
3. Check each reference in compiled PDF
4. Click each DOI link
5. Verify formatting matches journal requirements

## Part VI: Special Cases

### Software Citations

Many journals now expect software citations:

```bibtex
@article{QuTiP2012,
  author = {Johansson, J. R. and Nation, P. D. and Nori, Franco},
  title = {QuTiP: An open-source Python framework for the dynamics
           of open quantum systems},
  journal = {Computer Physics Communications},
  volume = {183},
  pages = {1760-1772},
  year = {2012}
}
```

### Data Citations

For shared datasets:

```bibtex
@misc{Dataset2024,
  author = {Author, A. B.},
  title = {Dataset Title},
  year = {2024},
  publisher = {Repository Name},
  doi = {10.XXXX/XXXXX}
}
```

### Conference Proceedings

```bibtex
@inproceedings{Author2023,
  author = {Author, A. B. and Author, C. D.},
  title = {Paper Title},
  booktitle = {Conference Name},
  year = {2023},
  pages = {123-456},
  doi = {10.XXXX/XXXXX}
}
```

### Theses

```bibtex
@phdthesis{Author2022,
  author = {Author, A. B.},
  title = {Thesis Title},
  school = {University Name},
  year = {2022}
}
```

## Summary

Effective reference management requires:

1. **Organization:** Use citation management software
2. **Completeness:** Fill all required fields
3. **Accuracy:** Verify each reference
4. **Ethics:** Cite fairly and honestly
5. **Formatting:** Follow journal style exactly

---

*Use this guide throughout your paper writing process, with final verification before submission.*
