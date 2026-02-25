# arXiv Submission Checklist

## Complete Guide to arXiv Preprint Submission

---

## Document Information

**Paper Title**: ___________________________________________________________

**Primary Category**: ___________________________________________________________

**Planned Submission Date**: ___________________________________________________________

**Announcement Target Date**: ___________________________________________________________

---

## 1. Pre-Submission Preparation

### Account Status

- [ ] arXiv account exists and accessible
- [ ] Email address current
- [ ] Username: _____________
- [ ] Password: [ ] Remembered [ ] Reset needed

### Endorsement Status

**For first-time submissions to a category, endorsement is required**:

- [ ] Endorsement not needed (previous submission in category)
- [ ] Endorsement obtained
  - Endorser name: _____________
  - Endorsement date: _____________
- [ ] Endorsement needed
  - Potential endorsers: _____________
  - Request sent: [ ] Yes [ ] No

### Category Selection

**Primary Category** (most specific match):
- [ ] quant-ph (Quantum Physics)
- [ ] cond-mat.mes-hall (Mesoscopic Systems)
- [ ] cond-mat.str-el (Strongly Correlated Electrons)
- [ ] physics.atom-ph (Atomic Physics)
- [ ] physics.optics (Optics)
- [ ] cs.IT (Information Theory)
- [ ] cs.ET (Emerging Technologies)
- [ ] Other: _____________

**Cross-List Categories** (additional relevant categories):
1. _____________
2. _____________
3. _____________

---

## 2. File Preparation

### LaTeX Source Files

**Main document**:
- [ ] main.tex compiles without errors
- [ ] Uses standard document class (revtex, article, etc.)
- [ ] No proprietary packages required
- [ ] No absolute file paths

**Bibliography**:
- [ ] references.bib included
- [ ] All entries complete
- [ ] No missing citations
- [ ] BibTeX or BibLaTeX format

### Figure Files

| Figure | File Name | Format | Size | arXiv Compatible |
|--------|-----------|--------|------|------------------|
| 1 | | [ ] PDF [ ] PNG [ ] EPS | _____ KB | [ ] |
| 2 | | [ ] PDF [ ] PNG [ ] EPS | _____ KB | [ ] |
| 3 | | [ ] PDF [ ] PNG [ ] EPS | _____ KB | [ ] |
| 4 | | [ ] PDF [ ] PNG [ ] EPS | _____ KB | [ ] |
| S1 | | [ ] PDF [ ] PNG [ ] EPS | _____ KB | [ ] |
| S2 | | [ ] PDF [ ] PNG [ ] EPS | _____ KB | [ ] |

**Figure Requirements**:
- [ ] All figures in acceptable format (PDF, PNG, EPS, PS, JPEG)
- [ ] No TIFF files (convert to PNG)
- [ ] Total size under limit (typically 50 MB)
- [ ] All referenced in main.tex

### Supplementary Files (if any)

- [ ] Supplementary .tex files included
- [ ] Supplementary figures included
- [ ] No standalone PDF supplements (if submitting source)

### File Organization

```
submission_folder/
├── main.tex
├── references.bib
├── figure1.pdf
├── figure2.pdf
├── figure3.pdf
├── figure4.pdf
├── supplementary.tex (optional)
└── [other files]
```

- [ ] All files in single directory (no subdirectories)
- [ ] File names: no spaces, simple characters
- [ ] All files referenced with correct names (case-sensitive)

---

## 3. Local Compilation Test

### Verify Compilation

Before uploading, ensure the document compiles:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

- [ ] Compiles without errors
- [ ] No missing figures
- [ ] No undefined references
- [ ] No missing citations
- [ ] Output PDF looks correct

### Common Issues to Fix Before Upload

| Issue | Solution |
|-------|----------|
| Missing font | Use standard LaTeX fonts |
| Missing package | Include necessary .sty files or use standard packages |
| Figure not found | Check file name and path |
| Bibliography error | Verify .bib file format |
| Overfull hbox | Adjust line breaks or figures |

---

## 4. Metadata Preparation

### Title

**Exact title for arXiv**:
> _____________________________________________________________________________

- [ ] Matches manuscript exactly
- [ ] No LaTeX formatting (unless necessary for equations)
- [ ] Appropriate capitalization

### Authors

**Author string format**: First1 Last1, First2 Last2, First3 Last3

**Author list for arXiv**:
> _____________________________________________________________________________

- [ ] All authors included
- [ ] Order matches manuscript
- [ ] Names formatted consistently
- [ ] No affiliations in author list (arXiv doesn't use them)

### Abstract

**Abstract for arXiv** (will be extracted from source if not provided):

Copy abstract here for verification:
> _____________________________________________________________________________
> _____________________________________________________________________________
> _____________________________________________________________________________

- [ ] Complete abstract ready
- [ ] No undefined abbreviations
- [ ] No citations (not supported in arXiv abstract display)
- [ ] LaTeX math formatting works
- [ ] Within reasonable length

### Comments (Optional)

**Comments field** (appears on abstract page):
> _____________________________________________________________________________

Examples:
- "15 pages, 4 figures"
- "Submitted to Physical Review Letters"
- "Extended version of conference paper"
- "v2: Corrected typo in Eq. 3"

### Report Number (Optional)

**Report number** (if your institution assigns one):
> _____________________________________________________________________________

---

## 5. License Selection

### Available Licenses

- [ ] **CC BY 4.0** - Most permissive, recommended for open access journals
- [ ] **CC BY-SA 4.0** - Share-alike requirement
- [ ] **CC BY-NC-SA 4.0** - Non-commercial only
- [ ] **arXiv.org perpetual, non-exclusive license** - Default, broad rights

**Selected License**: _____________

**Considerations**:
- CC BY 4.0 compatible with most journals
- Some journals require specific licenses
- Check target journal's preprint policy

---

## 6. Submission Execution

### Step-by-Step Submission

**Step 1: Start Submission**
- [ ] Logged into arXiv
- [ ] Clicked "Submit" or "Start New Submission"
- [ ] Selected license

**Step 2: Upload Files**
- [ ] Selected upload method: [ ] Individual files [ ] Archive (.tar.gz/.zip)
- [ ] Uploaded all files
- [ ] arXiv processing initiated
- [ ] Waited for compilation (may take several minutes)

**Step 3: Check Compilation**
- [ ] Viewed generated PDF
- [ ] All pages correct
- [ ] All figures appear
- [ ] All equations render
- [ ] References complete

**If Compilation Issues**:
- Note error messages: _____________
- Possible fix: _____________
- Reupload if needed

**Step 4: Enter Metadata**
- [ ] Selected primary category
- [ ] Selected cross-list categories
- [ ] Entered/verified title
- [ ] Entered/verified authors
- [ ] Entered/verified abstract
- [ ] Added comments (optional)
- [ ] Added report number (optional)

**Step 5: Preview**
- [ ] Reviewed abstract page preview
- [ ] All information correct
- [ ] PDF accessible and correct

**Step 6: Submit**
- [ ] Agreed to terms
- [ ] Submitted
- [ ] Noted submission ID: _____________

---

## 7. Post-Submission

### Confirmation

- [ ] Received confirmation email
- [ ] Submission ID recorded: _____________
- [ ] Status: [ ] Processing [ ] Submitted [ ] On hold

### Moderation Timeline

arXiv submissions undergo moderation:
- Weekday submissions: Typically announced next weekday at 8pm ET
- Weekend submissions: Announced Monday at 8pm ET
- Moderation may delay by 1-2 days

**Expected announcement date**: _____________

### After Announcement

- [ ] Paper announced
- [ ] arXiv ID: _____________
- [ ] URL: https://arxiv.org/abs/_____________
- [ ] Updated journal submission with arXiv ID
- [ ] Shared announcement

---

## 8. Common arXiv Issues and Solutions

### Compilation Failures

| Error | Solution |
|-------|----------|
| "Unknown package" | Include .sty file or remove package |
| "Missing figure" | Check file name (case-sensitive) |
| "Runaway argument" | Check for unclosed braces |
| "Missing \begin{document}" | Check preamble syntax |
| "Undefined control sequence" | Define command or check spelling |

### Moderation Issues

| Issue | Response |
|-------|----------|
| On hold for review | Wait (usually resolved in 1-2 days) |
| Reclassification suggested | Accept if appropriate or respond |
| Submission rejected | Review reason, resubmit to appropriate category |

### Version Updates (After Initial Posting)

To submit updated version:
1. Go to "Submissions" in your arXiv account
2. Select the paper
3. Click "Add a new version"
4. Upload new files
5. Explain changes in comments

---

## 9. arXiv Timing Strategy

### Optimal Submission Timing

**For maximum visibility**:
- Submit early in the week (Monday-Tuesday)
- Submit before 2pm ET for same-day processing
- Avoid submission right before major holidays

**For coordination with journal submission**:
- arXiv on same day or before journal submission
- Include arXiv ID in cover letter if possible
- Note preprint status in journal submission

### Announcement Schedule

arXiv announces new submissions:
- Monday-Friday at 8:00pm ET
- Cutoff for next-day announcement: 2:00pm ET (14:00 ET)
- Weekends: No announcements (queue for Monday)

---

## 10. Checklist Summary

### Before Starting

- [ ] Account ready
- [ ] Endorsement obtained (if needed)
- [ ] Category selected
- [ ] Files prepared
- [ ] Local compilation verified

### During Submission

- [ ] Files uploaded successfully
- [ ] Compilation succeeded
- [ ] Metadata entered correctly
- [ ] Preview verified
- [ ] Submitted

### After Submission

- [ ] Confirmation received
- [ ] Submission ID recorded
- [ ] Waiting for announcement
- [ ] Announcement materials prepared

---

## Notes

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

**Submission completed**: [ ] Yes [ ] No

**arXiv ID**: _____________

**Announcement date**: _____________
