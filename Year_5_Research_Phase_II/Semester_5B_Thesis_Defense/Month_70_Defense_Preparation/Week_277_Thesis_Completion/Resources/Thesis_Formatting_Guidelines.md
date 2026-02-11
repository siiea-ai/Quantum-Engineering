# Thesis Formatting Guidelines

## Overview

This document provides comprehensive formatting specifications for doctoral theses in Quantum Science and Engineering. These guidelines ensure professional presentation and compliance with typical institutional requirements. Always verify against your specific institution's requirements.

---

## Document Specifications

### Page Setup

| Element | Specification |
|---------|---------------|
| Paper Size | Letter (8.5" x 11") or A4 (210mm x 297mm) |
| Left Margin | 1.5" (for binding) or 1.25" |
| Right Margin | 1" |
| Top Margin | 1" |
| Bottom Margin | 1" |
| Orientation | Portrait (landscape for wide figures/tables allowed) |

### Typography

| Element | Specification |
|---------|---------------|
| Body Font | Times New Roman, Computer Modern, or Palatino |
| Font Size | 11pt or 12pt |
| Line Spacing | Double-spaced (2.0) or 1.5 |
| Paragraph Indent | 0.5" first line indent |
| Justification | Left-justified or full justification |
| Widow/Orphan Control | Enabled |

### Fonts for Special Content

| Content Type | Font/Style |
|--------------|------------|
| Chapter Titles | Bold, 14-16pt |
| Section Headings | Bold, 12-14pt |
| Subsection Headings | Bold or Italic, 11-12pt |
| Figure Captions | Regular or Italic, 10-11pt |
| Table Captions | Regular or Italic, 10-11pt |
| Equations | Standard LaTeX math fonts |
| Code Listings | Monospace (Courier, Consolas) |

---

## Page Numbering

### Front Matter
- Roman numerals (i, ii, iii, iv, ...)
- Title page counts as page i but number is not displayed
- Page numbers centered at bottom or upper right

### Main Body and Back Matter
- Arabic numerals (1, 2, 3, ...)
- Chapter 1 begins on page 1
- Page numbers upper right or bottom center
- Consistent placement throughout

### Chapter Openings
- Chapter starts on new page (odd-numbered if double-sided)
- Page number may be suppressed or placed at bottom center

---

## Heading Hierarchy

### Formatting Levels

```
CHAPTER 1
INTRODUCTION
[Centered, Bold, All Caps, 14-16pt]

1.1 Section Heading
[Left-aligned, Bold, Title Case, 12-14pt]

1.1.1 Subsection Heading
[Left-aligned, Bold or Italic, Title Case, 11-12pt]

1.1.1.1 Sub-subsection Heading
[Left-aligned, Italic, Sentence case, 11-12pt]

Paragraph heading.  Inline, bold or italic, followed by period,
then text continues on same line.
```

### Numbering Convention

- Chapters: Arabic numerals (Chapter 1, Chapter 2, ...)
- Sections: Chapter.Section (1.1, 1.2, ...)
- Subsections: Chapter.Section.Subsection (1.1.1, 1.1.2, ...)
- Appendices: Letters (Appendix A, Appendix B, ...)
- Appendix sections: A.1, A.2, ...

---

## Figures

### Figure Placement
- As close as possible to first reference
- Top or bottom of page preferred
- Full-page figures acceptable for complex content
- Landscape orientation for wide figures (rotate page, not figure)

### Figure Formatting

```
[FIGURE IMAGE]

Figure 2.3: Caption text begins with uppercase letter and provides
a complete description of the figure content. Captions should be
self-contained so readers can understand the figure without reading
the main text. Reproduced from [Source] with permission.
```

### Figure Specifications

| Element | Specification |
|---------|---------------|
| Resolution | 300 dpi minimum for print |
| File Format | PDF, EPS (vector), PNG, TIFF (raster) |
| Line Width | 0.5-1.0 pt minimum for visibility |
| Font Size | 8-10pt minimum for axis labels |
| Colors | Consider colorblind-friendly palettes |
| Caption Position | Below figure |
| Caption Width | Same as figure or full text width |

### Color Considerations

- Ensure figures are readable in grayscale (for print)
- Use colorblind-friendly palettes (avoid red-green only)
- Consider high contrast for projector presentations
- Recommended palettes: ColorBrewer, Viridis, Plasma

---

## Tables

### Table Placement
- As close as possible to first reference
- Top of page preferred
- Multi-page tables with continued headers

### Table Formatting

```
Table 3.2: Caption text appears above the table and describes
the content and context of the data presented.

-----------------------------------------------------------------
| Column 1 Header | Column 2 Header | Column 3 Header | Units   |
|-----------------|-----------------|-----------------|---------|
| Data entry 1    | 0.123 ± 0.005   | 45.6            | MHz     |
| Data entry 2    | 0.456 ± 0.012   | 78.9            | MHz     |
| Data entry 3    | 0.789 ± 0.023   | 12.3            | MHz     |
-----------------------------------------------------------------

Note: Additional information about the table can be included in
footnotes below the table.
```

### Table Specifications

| Element | Specification |
|---------|---------------|
| Caption Position | Above table |
| Alignment | Left or center |
| Horizontal Lines | Top, bottom, below header |
| Vertical Lines | Minimal or none (modern style) |
| Font Size | Same as body or slightly smaller (10-11pt) |
| Column Spacing | Consistent throughout |

---

## Equations

### Display Equations

For numbered equations (recommended for equations referenced in text):
```latex
The Schrödinger equation in the position representation is
\begin{equation}
    i\hbar \frac{\partial}{\partial t} \Psi(\mathbf{r},t) =
    \hat{H} \Psi(\mathbf{r},t)
    \label{eq:schrodinger}
\end{equation}
where $\hat{H}$ is the Hamiltonian operator.
```

Renders as:

$$i\hbar \frac{\partial}{\partial t} \Psi(\mathbf{r},t) = \hat{H} \Psi(\mathbf{r},t) \quad (2.1)$$

### Equation Formatting

| Element | Specification |
|---------|---------------|
| Numbering | Right-aligned: (Chapter.Number) |
| Placement | Centered on page |
| Spacing | Extra space above and below |
| Multi-line | Aligned at equals sign or operators |
| Variables | Italic ($E$, $\psi$, $t$) |
| Operators | Roman ($\sin$, $\cos$, $\exp$, $\mathrm{Tr}$) |
| Units | Roman with space (5 $\mu$s, 100 MHz) |

### Inline Equations
- Use for simple expressions: $E = \hbar\omega$
- Avoid line-breaking inline equations
- Consider display format for complex expressions

---

## Citations and Bibliography

### Citation Styles for Physics

**Numeric Style (APS/AIP):**
- In text: [1], [2,3], [4-7]
- Bibliography: Numbered list in citation order

**Author-Year Style (Harvard):**
- In text: (Smith, 2023), (Smith and Jones, 2024)
- Bibliography: Alphabetical by author

### Bibliography Entry Formats

**Journal Article:**
```
[1] A. B. Author and C. D. Author, "Article title," Journal Name
    Vol, Page (Year), doi:10.xxxx/xxxxx.
```

**Book:**
```
[2] A. B. Author, Book Title (Publisher, City, Year).
```

**Book Chapter:**
```
[3] A. B. Author, "Chapter title," in Book Title, edited by
    C. D. Editor (Publisher, City, Year), pp. xx-xx.
```

**Thesis:**
```
[4] A. B. Author, "Thesis title," Ph.D. thesis, University Name,
    Year.
```

**arXiv Preprint:**
```
[5] A. B. Author, "Article title," arXiv:xxxx.xxxxx (Year).
```

### BibTeX Example

```bibtex
@article{nielsen2000quantum,
  author = {Nielsen, Michael A. and Chuang, Isaac L.},
  title = {Quantum Computation and Quantum Information},
  journal = {Cambridge University Press},
  year = {2000},
  doi = {10.1017/CBO9780511976667}
}

@article{shor1995scheme,
  author = {Shor, Peter W.},
  title = {Scheme for reducing decoherence in quantum computer memory},
  journal = {Physical Review A},
  volume = {52},
  pages = {R2493--R2496},
  year = {1995},
  doi = {10.1103/PhysRevA.52.R2493}
}
```

---

## Special Content

### Code Listings

```latex
\begin{lstlisting}[language=Python, caption=Example simulation code]
import numpy as np
from qutip import *

# Define Hamiltonian
H = 2*np.pi * sigmax()

# Time evolution
times = np.linspace(0, 1, 100)
result = mesolve(H, basis(2,0), times, [], [sigmax(), sigmay(), sigmaz()])
\end{lstlisting}
```

### Algorithms

```latex
\begin{algorithm}
\caption{Quantum Phase Estimation}
\begin{algorithmic}[1]
\Require Unitary $U$, eigenstate $|\psi\rangle$, precision $n$
\Ensure Phase estimate $\tilde{\phi}$
\State Initialize register to $|0\rangle^{\otimes n}|\psi\rangle$
\State Apply Hadamard gates to first $n$ qubits
\For{$j = 0$ to $n-1$}
    \State Apply controlled-$U^{2^j}$ from qubit $j$
\EndFor
\State Apply inverse QFT to first $n$ qubits
\State Measure first $n$ qubits to obtain $\tilde{\phi}$
\end{algorithmic}
\end{algorithm}
```

### Theorems and Proofs

```latex
\begin{theorem}[No-Cloning Theorem]
\label{thm:nocloning}
It is impossible to create an identical copy of an arbitrary
unknown quantum state.
\end{theorem}

\begin{proof}
Assume a cloning operation $U$ exists such that...
[proof content]
\end{proof}
```

---

## LaTeX Packages

### Recommended Packages

```latex
% Document class
\documentclass[12pt, oneside]{report}

% Page setup
\usepackage[margin=1in, left=1.5in]{geometry}

% Typography
\usepackage[T1]{fontenc}
\usepackage{mathpazo}  % or \usepackage{times}
\usepackage{microtype}

% Math
\usepackage{amsmath, amssymb, amsthm}
\usepackage{physics}  % For bra-ket notation
\usepackage{siunitx}  % For units

% Graphics
\usepackage{graphicx}
\usepackage{float}
\usepackage{subcaption}

% Tables
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{multirow}

% Code listings
\usepackage{listings}
\usepackage{algorithm}
\usepackage{algorithmic}

% References
\usepackage[numbers,sort&compress]{natbib}
\usepackage{hyperref}
\usepackage{cleveref}

% Other
\usepackage{appendix}
\usepackage{setspace}
\doublespacing
```

---

## Quality Assurance Checklist

### Before Final Submission

- [ ] All margins meet requirements
- [ ] Page numbers correct and consistent
- [ ] Heading hierarchy consistent
- [ ] All figures high resolution (300+ dpi)
- [ ] All figure captions complete
- [ ] All table captions complete
- [ ] All equations numbered if referenced
- [ ] All citations present in bibliography
- [ ] No orphan bibliography entries
- [ ] No widows or orphans
- [ ] Consistent spacing throughout
- [ ] Spell check complete (including technical terms)
- [ ] PDF bookmarks/hyperlinks functional

### Common LaTeX Errors to Check

- Overfull hboxes (text extending into margin)
- Underfull hboxes (poor spacing)
- Missing references
- Missing citations
- Duplicate labels
- Figure/table placement issues

---

## Accessibility Considerations

### For Document Accessibility

1. **Figure Alt Text:** Provide descriptive alt text for all figures
2. **Table Headers:** Properly define table headers for screen readers
3. **Color Contrast:** Ensure sufficient contrast ratios
4. **Font Choice:** Use readable fonts at appropriate sizes
5. **PDF Tagging:** Generate tagged PDFs for accessibility

### For Colorblind Accessibility

- Use patterns in addition to colors in graphs
- Avoid red-green only color schemes
- Test figures with colorblind simulators
- Include grayscale-readable versions

---

## Institutional Variations

Always check your specific institution's requirements for:
- Exact margin specifications
- Required font families
- Specific page numbering conventions
- Required front matter elements
- Signature page format
- Binding requirements
- Electronic submission specifications

---

*Guidelines Version 1.0 | Month 70: Defense Preparation*
