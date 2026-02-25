# University Thesis Format Guide

## Overview

This guide provides comprehensive formatting requirements for doctoral thesis submission. Specific requirements vary by institution---always verify with your graduate school's official guidelines.

---

## Page Setup

### Paper Size and Margins

| Element | Standard US | Standard A4 |
|---------|-------------|-------------|
| Paper Size | 8.5" x 11" | 210mm x 297mm |
| Top Margin | 1 inch | 25mm |
| Bottom Margin | 1 inch | 25mm |
| Left Margin | 1.5 inch | 40mm |
| Right Margin | 1 inch | 25mm |

**Note:** Left margin is wider to accommodate binding.

### LaTeX Implementation

```latex
\documentclass[12pt, oneside]{report}

% US Letter size
\usepackage[
    letterpaper,
    top=1in,
    bottom=1in,
    left=1.5in,
    right=1in,
    includehead,
    includefoot
]{geometry}

% A4 size alternative
% \usepackage[
%     a4paper,
%     top=25mm,
%     bottom=25mm,
%     left=40mm,
%     right=25mm
% ]{geometry}
```

---

## Typography

### Font Requirements

**Acceptable Fonts:**
- Times New Roman
- Computer Modern (LaTeX default)
- Palatino
- Garamond
- Arial (sans-serif, some institutions)

**Font Sizes:**
| Element | Size |
|---------|------|
| Body text | 12pt |
| Chapter titles | 14-18pt |
| Section headings | 12-14pt |
| Footnotes | 10pt |
| Captions | 10-11pt |

### LaTeX Font Setup

```latex
% Times New Roman equivalent
\usepackage{mathptmx}

% Or Palatino
% \usepackage{palatino}

% For proper font encoding
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
```

### Line Spacing

| Element | Spacing |
|---------|---------|
| Body text | Double-spaced |
| Block quotes | Single-spaced |
| Footnotes | Single-spaced |
| Figure captions | Single-spaced |
| Table captions | Single-spaced |
| Bibliography entries | Single-spaced |
| Between bibliography entries | Double-spaced |

```latex
\usepackage{setspace}
\doublespacing

% For single-spacing where needed
\begin{singlespace}
% Content here
\end{singlespace}
```

---

## Document Structure

### Required Order of Elements

#### Front Matter (Roman numerals)

1. **Title Page** (no page number, counts as i)
2. **Copyright Page** (optional, no number or ii)
3. **Abstract** (page ii or iii)
4. **Dedication** (optional)
5. **Acknowledgments**
6. **Table of Contents**
7. **List of Figures** (if applicable)
8. **List of Tables** (if applicable)
9. **List of Abbreviations/Symbols** (if applicable)
10. **Preface** (optional)

#### Body (Arabic numerals, starting at 1)

1. **Chapter 1: Introduction**
2. **Chapter 2-N: Main Chapters**
3. **Chapter N+1: Conclusion**

#### Back Matter (continues Arabic numerals)

1. **Bibliography/References**
2. **Appendices** (lettered A, B, C...)
3. **Vita/CV** (if required)

---

## Title Page Format

```
[1.5" from top]

                    [TITLE IN ALL CAPITALS]
                    [SUBTITLE IF APPLICABLE]

                              by

                    [Your Full Legal Name]


         A dissertation submitted in partial fulfillment
              of the requirements for the degree of

                    Doctor of Philosophy

                   ([Field of Study])


                    [University Name]
                    [City, State/Country]
                         [Year]


Doctoral Committee:
    [Advisor Name], Chair
    [Committee Member Name]
    [Committee Member Name]
    [Committee Member Name]
```

### LaTeX Title Page

```latex
\begin{titlepage}
\centering
\vspace*{1in}

{\LARGE\bfseries QUANTUM ERROR CORRECTION IN\\
SUPERCONDUCTING QUBIT SYSTEMS\par}

\vspace{1in}

by\\[0.5in]
{\Large Your Full Name\par}

\vspace{1in}

A dissertation submitted in partial fulfillment\\
of the requirements for the degree of\\[0.3in]
{\large Doctor of Philosophy\\
(Physics)\par}

\vspace{0.5in}

{\large University Name\\
Year\par}

\vfill

Doctoral Committee:\\[0.2in]
\begin{tabular}{l}
Professor Advisor Name, Chair\\
Professor Member Name\\
Professor Member Name\\
Professor Member Name
\end{tabular}

\end{titlepage}
```

---

## Chapter Formatting

### Chapter Title Format

```latex
\usepackage{titlesec}

% Centered chapter titles
\titleformat{\chapter}[display]
    {\normalfont\Large\bfseries\centering}
    {\chaptertitlename\ \thechapter}
    {20pt}
    {\LARGE}

% Start chapters on new page
\newcommand{\chapterbreak}{\clearpage}

% Example chapter
\chapter{Introduction}
```

### Section Hierarchy

```latex
\titleformat{\section}
    {\normalfont\large\bfseries}
    {\thesection}
    {1em}
    {}

\titleformat{\subsection}
    {\normalfont\normalsize\bfseries}
    {\thesubsection}
    {1em}
    {}

\titleformat{\subsubsection}
    {\normalfont\normalsize\itshape}
    {\thesubsubsection}
    {1em}
    {}
```

---

## Page Numbers

### Placement Options

| Style | Position |
|-------|----------|
| Standard | Centered, bottom |
| Alternative | Right-aligned, top or bottom |
| Chapter pages | Centered, bottom (even if others are top) |

### LaTeX Page Numbering

```latex
\usepackage{fancyhdr}

% Front matter: Roman numerals
\frontmatter
\pagenumbering{roman}
\setcounter{page}{2}  % Title page is i

% Main matter: Arabic numerals
\mainmatter
\pagenumbering{arabic}

% Page style
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}

% Plain style for chapter pages
\fancypagestyle{plain}{
    \fancyhf{}
    \fancyfoot[C]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
}
```

---

## Figures and Tables

### Figure Formatting

```latex
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}

% Caption formatting
\captionsetup{
    font=small,
    labelfont=bf,
    labelsep=period,
    justification=justified,
    singlelinecheck=false
}

% Figure example
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figure_name}
    \caption{Description of the figure. This should be a complete
    sentence explaining what the figure shows.}
    \label{fig:label}
\end{figure}
```

### Table Formatting

```latex
\usepackage{booktabs}
\usepackage{array}

% Table example
\begin{table}[htbp]
    \centering
    \caption{Description of the table contents.}
    \label{tab:label}
    \begin{tabular}{lcc}
        \toprule
        Parameter & Value & Unit \\
        \midrule
        Frequency & 5.0 & GHz \\
        Temperature & 15 & mK \\
        Fidelity & 99.5 & \% \\
        \bottomrule
    \end{tabular}
\end{table}
```

### Figure/Table Numbering

- Number by chapter: Figure 2.1, Figure 2.2, Table 3.1
- Or continuous: Figure 1, Figure 2, Table 1

```latex
% Chapter-based numbering
\usepackage{chngcntr}
\counterwithin{figure}{chapter}
\counterwithin{table}{chapter}
\counterwithin{equation}{chapter}
```

---

## Equations

### Formatting Standards

```latex
\usepackage{amsmath}
\usepackage{amssymb}

% Numbered equation
\begin{equation}
    H = \sum_{i} \omega_i a_i^\dagger a_i + \sum_{i<j} g_{ij}(a_i^\dagger a_j + a_j^\dagger a_i)
    \label{eq:hamiltonian}
\end{equation}

% Multi-line equation
\begin{align}
    \ket{\psi} &= \alpha\ket{0} + \beta\ket{1} \\
    |\alpha|^2 + |\beta|^2 &= 1
\end{align}

% Boxed important equation
\begin{equation}
    \boxed{F = 1 - \frac{1}{2}\text{Tr}(\rho - \sigma)^2}
\end{equation}
```

---

## Bibliography

### Citation Styles

| Field | Common Style |
|-------|--------------|
| Physics | Physical Review (numeric) |
| Engineering | IEEE |
| Sciences | APA or Nature |
| Humanities | Chicago/MLA |

### BibLaTeX Setup

```latex
\usepackage[
    backend=biber,
    style=phys,
    articletitle=true,
    biblabel=brackets
]{biblatex}

\addbibresource{thesis.bib}

% In document
\cite{author2023}
\textcite{author2023}

% Print bibliography
\printbibliography[title={References}]
```

---

## Appendices

```latex
\appendix

\chapter{Derivation of the Master Equation}
\label{app:master_equation}
% Content here

\chapter{Supplementary Data}
\label{app:data}
% Content here
```

---

## Common Requirements Checklist

### Text Formatting
- [ ] 12pt font throughout body
- [ ] Double-spaced body text
- [ ] 1.5" left margin, 1" others
- [ ] Consistent heading styles
- [ ] No widows/orphans (single lines)

### Figures/Tables
- [ ] All figures minimum 300 DPI
- [ ] Captions in correct position
- [ ] All referenced in text
- [ ] Numbered correctly

### References
- [ ] Consistent citation style
- [ ] All citations have entries
- [ ] Complete bibliographic information
- [ ] DOIs where available

### PDF
- [ ] All fonts embedded
- [ ] No security restrictions
- [ ] Bookmarks present
- [ ] PDF/A compliant (if required)

---

## Institution-Specific Notes

*This section should be customized with your institution's specific requirements.*

### [University Name] Requirements

**Thesis Office Contact:**
- Email:
- Phone:
- Office:

**Submission Portal:**
- URL:

**Specific Requirements:**
-

**Deadlines:**
-

---

*Last Updated: [Date]*
*Always verify with official university guidelines*
