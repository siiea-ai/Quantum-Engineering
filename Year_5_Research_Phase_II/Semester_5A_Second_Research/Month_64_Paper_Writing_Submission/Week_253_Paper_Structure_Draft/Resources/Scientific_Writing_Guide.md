# Scientific Writing Resources for Quantum Physics Papers

## Journal-Specific Guidelines and Formats

---

## Physical Review Journals (APS)

### Physical Review X (PRX)
**Scope**: Broad-interest physics research of exceptional quality
**Format**: No strict length limit, typically 10-20 pages
**Impact Factor**: ~12

**Key Requirements**:
- Significant advances of broad interest
- Clear presentation accessible to non-specialists
- REVTeX 4.2 with `prx` option
- Figures in PDF, EPS, or high-resolution PNG

**Template Header**:
```latex
\documentclass[prx,reprint,amsmath,amssymb]{revtex4-2}
```

### Physical Review Letters (PRL)
**Scope**: Short communications of major significance
**Format**: 4 pages maximum (3500 words)
**Impact Factor**: ~8

**Key Requirements**:
- Strictly 4 pages in two-column format
- Broad appeal required
- Supplementary material accepted
- First paragraph crucial (editors read this first)

**Template Header**:
```latex
\documentclass[prl,reprint,amsmath,amssymb]{revtex4-2}
```

### Physical Review A (PRA) / PRX Quantum
**Scope**: AMO physics, quantum information, quantum optics
**Format**: No strict limit, typically 8-15 pages
**Impact Factor**: ~3 (PRA), ~8 (PRX Quantum)

**PRX Quantum Specific**:
- Dedicated to quantum science and technology
- Broader accessibility expected
- Open access journal

---

## Nature Family Journals

### Nature Physics
**Scope**: All areas of pure and applied physics
**Format**: ~3000 words main text, up to 6 figures
**Impact Factor**: ~25

**Key Requirements**:
- Accessible abstract (no jargon)
- Limited references (typically ~50)
- Extended Data figures for additional content
- Methods section separate from main text

**Formatting**:
```latex
\documentclass[12pt]{article}
% Nature template available from Overleaf
```

### Nature Communications
**Scope**: Significant advances across natural sciences
**Format**: ~4500 words, typically 8-10 figures (with extended data)
**Impact Factor**: ~17

### npj Quantum Information
**Scope**: Quantum information science and technology
**Format**: ~4000 words, typically 6-8 figures
**Impact Factor**: ~7

**Key Requirements**:
- Open access (APC required)
- FAIR data principles
- Reproducibility statement

---

## Writing Style Guides

### General Principles for Physics Writing

**Clarity Over Cleverness**
- Use simple, direct language
- Avoid unnecessarily complex sentences
- One idea per paragraph

**Active vs. Passive Voice**

Use active voice when possible:
- Active: "We measured the coherence time..."
- Passive: "The coherence time was measured..."

Both are acceptable; mix for variety, but active is often clearer.

**Tense Usage**
| Section | Tense | Example |
|---------|-------|---------|
| Abstract | Present (findings), Past (methods) | "We demonstrate... We measured..." |
| Introduction | Present (general), Past (prior work) | "Quantum computing requires... Smith et al. showed..." |
| Methods | Past | "We prepared... The sample was cooled..." |
| Results | Past | "We observed... The data revealed..." |
| Discussion | Present (implications), Past (results) | "This demonstrates... We found..." |

### Technical Writing Conventions

**Numbers**
- Write out one through nine
- Use numerals for 10 and above
- Always use numerals with units: "5 ms", not "five ms"
- Use SI units consistently

**Equations**
- Number equations referenced later: Eq. (1)
- Inline equations for simple expressions: $E = \hbar\omega$
- Display equations for key results:
  $$H = \sum_i \omega_i \sigma_i^z + \sum_{ij} J_{ij} \sigma_i^x \sigma_j^x$$

**Abbreviations**
- Define on first use: "quantum error correction (QEC)"
- Use abbreviation consistently thereafter
- Avoid abbreviations in titles and abstracts

**Citations**
- Use citation as noun: "As shown by Smith et al. [1]..."
- Or as reference: "Prior work [1,2] demonstrates..."
- Cite primary sources, not only reviews
- Include arXiv papers (now accepted in most journals)

---

## Reference Management

### BibTeX Best Practices

**Standard Entry Format**:
```bibtex
@article{author2023,
  title = {Title of the Paper},
  author = {Last1, First1 and Last2, First2 and Last3, First3},
  journal = {Physical Review X},
  volume = {13},
  pages = {021015},
  year = {2023},
  doi = {10.1103/PhysRevX.13.021015},
  publisher = {American Physical Society}
}
```

**arXiv Preprints**:
```bibtex
@article{author2023arxiv,
  title = {Title of the Preprint},
  author = {Last, First},
  journal = {arXiv preprint arXiv:2301.12345},
  year = {2023}
}
```

**Consistency Checks**:
- [ ] All journal names use same format
- [ ] Author names formatted consistently
- [ ] DOIs included where available
- [ ] Page numbers/article IDs correct

### Recommended Reference Managers

**Zotero** (Recommended)
- Free, open-source
- Browser integration
- Group libraries for collaboration
- BibTeX export

**Mendeley**
- Free, cloud-based
- PDF annotation
- Citation recommendations
- BibTeX export

---

## LaTeX Resources

### Essential Packages for Physics Papers

```latex
\usepackage{amsmath,amssymb}    % Mathematical symbols
\usepackage{graphicx}           % Figure inclusion
\usepackage{hyperref}           % Clickable links
\usepackage{xcolor}             % Color support
\usepackage{physics}            % Physics notation
\usepackage{braket}             % Dirac notation
\usepackage{siunitx}            % SI units
\usepackage{booktabs}           % Professional tables
\usepackage{algorithm2e}        % Pseudocode
```

### Quantum-Specific LaTeX

**Dirac Notation**:
```latex
\usepackage{braket}
$\ket{\psi}$, $\bra{\phi}$, $\braket{\phi|\psi}$
$\ketbra{\psi}{\phi}$ % outer product
```

**Quantum Circuits** (with quantikz):
```latex
\usepackage{tikz}
\usetikzlibrary{quantikz}

\begin{quantikz}
\lstick{$\ket{0}$} & \gate{H} & \ctrl{1} & \meter{} \\
\lstick{$\ket{0}$} & \qw & \targ{} & \qw
\end{quantikz}
```

### Common Math Notation

| Concept | LaTeX | Renders |
|---------|-------|---------|
| Hamiltonian | `\hat{H}` | $\hat{H}$ |
| Commutator | `[A, B]` | $[A, B]$ |
| Density matrix | `\rho` | $\rho$ |
| Trace | `\text{Tr}` or `\mathrm{Tr}` | Tr |
| Fidelity | `\mathcal{F}` | $\mathcal{F}$ |
| Pauli matrices | `\sigma_x, \sigma_y, \sigma_z` | $\sigma_x, \sigma_y, \sigma_z$ |
| Tensor product | `\otimes` | $\otimes$ |
| Partial trace | `\text{Tr}_B` | $\text{Tr}_B$ |

---

## Online Writing Resources

### Style Guides
- **APS Style Guide**: https://journals.aps.org/authors/axis-information-initiative-text-style
- **Nature Style Guide**: https://www.nature.com/nature-portfolio/for-authors/formatting-guide
- **Springer Physics Style**: https://www.springer.com/gp/authors-editors/book-authors-editors/manuscript-preparation

### Writing Tutorials
- **"The Science of Scientific Writing"** - Gopen & Swan (American Scientist)
- **MIT Physics Writing Guidelines**: https://physics.mit.edu/guides/
- **Stanford Scientific Writing**: https://www.nature.com/scitable/ebooks/english-communication-for-scientists-14053993

### Editing Tools
- **Grammarly** (academic style)
- **Writefull** (AI for academic writing)
- **LanguageTool** (open-source grammar checker)

---

## Video Resources

### Scientific Writing Lectures
- MIT OpenCourseWare: Scientific Communication
- CERN School of Computing: Scientific Writing
- YouTube: "How to Write a Scientific Paper" (various universities)

### Physics Presentation Skills
- APS March Meeting presentation guidelines
- QIP conference talk guidelines

---

## Books on Scientific Writing

### Essential Reading
1. **"The Elements of Style"** - Strunk & White
   Classic principles of clear writing

2. **"On Writing Well"** - William Zinsser
   Non-fiction writing craft

3. **"The Craft of Scientific Writing"** - Michael Alley
   Specific to scientific communication

4. **"Writing Science"** - Joshua Schimel
   Story structure in scientific papers

### Advanced
5. **"Style: Lessons in Clarity and Grace"** - Joseph Williams
   Deep dive into prose style

6. **"How to Write and Publish a Scientific Paper"** - Barbara Gastel
   Comprehensive guide to publication

---

## Quick Reference Card

### Paragraph Structure
1. Topic sentence (main point)
2. Supporting evidence/explanation
3. Connection to next paragraph

### Sentence Clarity Checklist
- [ ] Subject and verb close together
- [ ] Key information at sentence end
- [ ] Minimal words between subject and verb
- [ ] Active constructions preferred

### Word Choice
| Avoid | Use Instead |
|-------|-------------|
| utilize | use |
| methodology | method |
| in order to | to |
| the fact that | that |
| prior to | before |
| subsequent to | after |
| a number of | several, many |
| at this point in time | now |

---

*Last Updated: Month 64, Week 253*
