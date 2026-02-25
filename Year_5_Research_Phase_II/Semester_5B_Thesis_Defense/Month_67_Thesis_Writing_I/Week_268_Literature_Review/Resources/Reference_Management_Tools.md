# Reference Management Tools and Best Practices

## Purpose

This guide covers reference management tools and best practices for maintaining a well-organized thesis bibliography. Proper reference management saves time, prevents errors, and ensures scholarly rigor.

---

## Recommended Reference Managers

### Option 1: Zotero (Recommended)

**Pros:**
- Free and open source
- Excellent browser integration (one-click save)
- Syncs across devices
- Strong collaboration features
- Large user community
- Good BibTeX export

**Cons:**
- 300MB free cloud storage (can pay for more)
- Learning curve for advanced features

**Setup for Thesis Writing:**
1. Install Zotero desktop: zotero.org
2. Install browser connector (Chrome/Firefox)
3. Install Better BibTeX plugin for export
4. Create collection for thesis
5. Set up auto-export to `.bib` file

**Key Features:**
- Automatic metadata extraction from DOI/arXiv
- PDF attachment and annotation
- Citation style editor
- Word/LaTeX integration

---

### Option 2: Mendeley

**Pros:**
- Free (owned by Elsevier)
- Good PDF organization
- Built-in PDF reader with annotation
- Web and desktop versions
- Social features for discovery

**Cons:**
- Owned by Elsevier (privacy concerns)
- BibTeX export less flexible
- Storage limits on free tier

**Setup for Thesis Writing:**
1. Install Mendeley Reference Manager
2. Create folder for thesis
3. Enable watched folder for PDFs
4. Set up BibTeX export

---

### Option 3: BibDesk (Mac only)

**Pros:**
- Native BibTeX editor
- Lightweight and fast
- Direct file editing (no database)
- Excellent for LaTeX users

**Cons:**
- Mac only
- No cloud sync (use Git instead)
- Manual metadata entry

**Setup for Thesis Writing:**
1. Install from bibdesk.sourceforge.io
2. Create single `.bib` file for thesis
3. Set up auto-file for PDFs
4. Use with Git for version control

---

### Option 4: JabRef

**Pros:**
- Free and open source
- Cross-platform (Java-based)
- Native BibTeX format
- Good for manual editing

**Cons:**
- Interface less polished
- Slower than native apps
- Manual metadata entry common

**Setup for Thesis Writing:**
1. Install from jabref.org
2. Create thesis bibliography file
3. Configure entry types for physics
4. Set up linked PDF folder

---

## BibTeX Best Practices

### File Organization

```
thesis/
├── main.tex
├── bibliography/
│   ├── thesis.bib          # Main bibliography
│   ├── foundational.bib    # Core QEC references (optional)
│   └── specialized.bib     # Your area (optional)
└── pdfs/
    ├── shor1995.pdf
    ├── gottesman1997.pdf
    └── ...
```

### Citation Key Convention

Use a consistent naming scheme:

**Format:** `lastname####keyword`

**Examples:**
- `shor1995scheme` - Shor's 1995 scheme paper
- `gottesman1997stabilizer` - Gottesman's 1997 stabilizer thesis
- `google2023suppressing` - Google's 2023 QEC paper
- `smith2024biased` - Your paper with Smith (if applicable)

**Benefits:**
- Memorable and predictable
- Sorts chronologically by author
- Easy to type

### Entry Formatting

**Standard article:**
```bibtex
@article{lastname2024keyword,
  author  = {LastName, FirstName and SecondAuthor, First and ThirdAuthor, First},
  title   = {Full Title in Title Case or Sentence case},
  journal = {Physical Review X},
  volume  = {14},
  number  = {2},
  pages   = {021001},
  year    = {2024},
  doi     = {10.1103/PhysRevX.14.021001},
  note    = {arXiv:2309.12345}
}
```

**arXiv preprint:**
```bibtex
@misc{lastname2024keyword,
  author       = {LastName, FirstName and SecondAuthor, First},
  title        = {Full Title},
  year         = {2024},
  eprint       = {2401.12345},
  archiveprefix= {arXiv},
  primaryclass = {quant-ph}
}
```

**Conference paper:**
```bibtex
@inproceedings{lastname2024keyword,
  author    = {LastName, FirstName},
  title     = {Paper Title},
  booktitle = {Proceedings of the 27th Annual Conference on Quantum Information Processing},
  year      = {2024},
  pages     = {1--10},
  doi       = {10.XXXX/...}
}
```

**PhD thesis:**
```bibtex
@phdthesis{lastname2024keyword,
  author = {LastName, FirstName},
  title  = {Thesis Title},
  school = {University Name},
  year   = {2024},
  note   = {arXiv:XXXX.XXXXX}
}
```

**Book:**
```bibtex
@book{lastname2024keyword,
  author    = {LastName, FirstName and SecondAuthor, First},
  title     = {Book Title},
  publisher = {Publisher Name},
  year      = {2024},
  edition   = {2nd},
  isbn      = {978-X-XXXX-XXXX-X}
}
```

---

## Workflow for Adding References

### Step 1: Capture
When you find a relevant paper:
1. Save to reference manager (browser button or DOI)
2. Verify metadata is complete
3. Download PDF
4. Add to thesis collection/folder

### Step 2: Categorize
Organize by topic:
- QEC foundations
- Topological codes
- Fault tolerance
- Your specialized area
- Recent papers (2024-2026)

### Step 3: Annotate (Optional but Recommended)
For key papers, add notes:
- Main results
- Relevance to your thesis
- Key figures/equations to cite
- Connection to your work

### Step 4: Export
Regular export to BibTeX:
- Set up automatic export if available
- Version control your `.bib` file
- Verify export is complete

### Step 5: Cite
When writing:
- Use citation key
- Verify citation is in bibliography
- Check PDF is accessible for verification

---

## Common BibTeX Errors and Fixes

### Error 1: Missing Braces in Titles
**Problem:** BibTeX lowercases titles
```bibtex
title = {Quantum Error Correction with the Toric Code}
% Becomes: "Quantum error correction with the toric code"
```

**Fix:** Protect capitalization:
```bibtex
title = {Quantum Error Correction with the {T}oric {C}ode}
% Or protect entire title:
title = {{Quantum Error Correction with the Toric Code}}
```

### Error 2: Missing Required Fields
**Problem:** Incomplete entries cause warnings

**Fix:** Include all required fields:
- `@article`: author, title, journal, year
- `@book`: author/editor, title, publisher, year
- `@inproceedings`: author, title, booktitle, year

### Error 3: Author Formatting
**Problem:** Incorrect author parsing
```bibtex
author = {John Smith and Jane Doe}  % Parsed as "John Smith" and "Jane Doe"
```

**Fix:** Use "LastName, FirstName" format:
```bibtex
author = {Smith, John and Doe, Jane}
```

### Error 4: Special Characters
**Problem:** LaTeX special characters in titles
```bibtex
title = {The $\alpha$-approximation}  % May cause errors
```

**Fix:** Use proper encoding:
```bibtex
title = {The $\alpha$-approximation}  % OK in most setups
% Or use text mode:
title = {The {\textalpha}-approximation}
```

### Error 5: Duplicate Keys
**Problem:** Same citation key used twice

**Fix:** Use unique, descriptive keys:
```bibtex
% Instead of smith2024, use:
smith2024biased   % Smith's biased noise paper
smith2024decoder  % Smith's decoder paper
```

---

## Version Control for Bibliography

### Using Git

```bash
# Initialize Git for thesis directory
cd thesis
git init

# Add bibliography to tracking
git add bibliography/thesis.bib

# Commit regularly
git commit -m "Add 10 references for QEC background"

# Before major changes
git checkout -b bibliography-update

# After verification, merge back
git checkout main
git merge bibliography-update
```

### Benefits
- Track changes over time
- Recover from errors
- Collaborate with co-authors
- Sync across machines

---

## Quick Reference: Finding DOIs

### From Journal Websites
Most journal pages show DOI prominently.

### From DOI.org
Enter title at doi.org to search.

### From CrossRef
Use crossref.org/guestquery for bulk lookup.

### From Google Scholar
Look for "Cited by" link, often includes DOI.

### From arXiv
If published, arXiv shows journal reference with DOI.

---

## Automation Tips

### Auto-Generate from DOI

Many tools can generate BibTeX from DOI:

**doi2bib.org:**
1. Paste DOI
2. Copy generated BibTeX
3. Add to your file

**Zotero/Mendeley:**
1. Add item by identifier (paste DOI)
2. Automatic BibTeX generation

### Batch Processing

For many DOIs:
```bash
# Using curl and doi2bib API
for doi in 10.1103/PhysRevX.14.021001 10.1038/nature23460; do
  curl "https://doi.org/$doi" -LH "Accept: application/x-bibtex"
done
```

---

## Final Checklist

Before submitting your thesis:

- [ ] All references have complete metadata
- [ ] No BibTeX warnings on compilation
- [ ] DOIs included for all published papers
- [ ] arXiv IDs included for preprints
- [ ] Citation keys are consistent and predictable
- [ ] No duplicate entries
- [ ] All PDFs accessible for verification
- [ ] Bibliography backed up in version control
- [ ] Export tested with clean compile

---

## Resources

### Reference Manager Documentation
- Zotero: zotero.org/support
- Mendeley: mendeley.com/guides
- BibDesk: bibdesk.sourceforge.io/manual
- JabRef: docs.jabref.org

### BibTeX References
- BibTeX documentation: bibtex.org
- BibLaTeX manual (for advanced usage)
- Your thesis template documentation

### Physics-Specific Resources
- APS journal BibTeX export
- arXiv BibTeX download
- NASA ADS for older papers
