# LaTeX Thesis Resources

## Essential Resources for Academic Document Preparation

---

## Thesis Templates

### University-Specific Templates

Many universities provide official LaTeX thesis templates. Always check your graduate school website first.

| University | Template Location |
|------------|------------------|
| MIT | https://libraries.mit.edu/theses-specs/ |
| Stanford | https://library.stanford.edu/research/bibliography-management/latex-and-bibtex |
| Harvard | https://gsas.harvard.edu/degree-requirements |
| Caltech | https://thesis.library.caltech.edu/ |
| Cambridge | https://www.repository.cam.ac.uk/ |
| Oxford | https://www.ox.ac.uk/students/academic/thesis |

### General Templates

| Template | Source | Features |
|----------|--------|----------|
| Clean Thesis | https://github.com/derric/cleanthesis | Modern, minimal design |
| Classic Thesis | https://ctan.org/pkg/classicthesis | Elegant typography |
| Overleaf Gallery | https://www.overleaf.com/gallery/tagged/thesis | Many options |
| LaTeX Templates | https://www.latextemplates.com/cat/thesis | Curated collection |

---

## Essential LaTeX Packages

### Mathematics

| Package | Purpose | Example |
|---------|---------|---------|
| `amsmath` | Core math environments | `\begin{align}...\end{align}` |
| `amssymb` | Math symbols | `\mathbb{R}`, `\forall` |
| `amsthm` | Theorem environments | `\begin{theorem}...\end{theorem}` |
| `mathtools` | Math extensions | `\coloneqq`, aligned equations |
| `bm` | Bold math | `\bm{x}` for bold vectors |

### Quantum Computing

| Package | Purpose | Example |
|---------|---------|---------|
| `braket` | Dirac notation | `\bra{0}`, `\ket{1}`, `\braket{0|1}` |
| `quantikz` | Quantum circuits | Modern, TikZ-based |
| `qcircuit` | Quantum circuits | Classic, Xy-pic based |
| `physics` | Physics notation | `\dv{f}{x}`, `\abs{x}` |

### Figures and Graphics

| Package | Purpose | Example |
|---------|---------|---------|
| `graphicx` | Include images | `\includegraphics[width=\textwidth]{fig.pdf}` |
| `tikz` | Vector graphics | Create diagrams programmatically |
| `pgfplots` | Data plots | Publication-quality graphs |
| `subcaption` | Subfigures | `\begin{subfigure}...\end{subfigure}` |
| `float` | Float placement | `[H]` for "here" placement |

### Tables

| Package | Purpose | Example |
|---------|---------|---------|
| `booktabs` | Professional tables | `\toprule`, `\midrule`, `\bottomrule` |
| `tabularx` | Flexible columns | Auto-width columns |
| `longtable` | Multi-page tables | Tables spanning pages |
| `multirow` | Multi-row cells | Merged cells |

### Bibliography

| Package | Purpose | Notes |
|---------|---------|-------|
| `biblatex` | Modern bibliography | Recommended for new documents |
| `natbib` | Author-year citations | Classic, widely used |
| `biber` | Backend for biblatex | Modern Unicode support |

### Cross-References

| Package | Purpose | Example |
|---------|---------|---------|
| `hyperref` | Hyperlinks | Clickable refs, URLs |
| `cleveref` | Smart references | `\cref{fig:example}` → "Figure 1" |
| `nameref` | Reference names | `\nameref{sec:intro}` → "Introduction" |

### Code Listings

| Package | Purpose | Notes |
|---------|---------|-------|
| `listings` | Code listing | Syntax highlighting |
| `minted` | Code with Pygments | Better highlighting (requires Python) |
| `algorithm2e` | Pseudocode | Algorithm environments |
| `algorithmicx` | Pseudocode | Alternative to algorithm2e |

### Formatting

| Package | Purpose | Example |
|---------|---------|---------|
| `geometry` | Page layout | Margins, paper size |
| `setspace` | Line spacing | `\doublespacing` |
| `fancyhdr` | Headers/footers | Custom page styles |
| `enumitem` | List formatting | Customize itemize/enumerate |
| `siunitx` | Units | `\SI{300}{\kelvin}` |

---

## Quantum Circuit Drawing

### Using quantikz

```latex
\usepackage{quantikz}

\begin{figure}
\centering
\begin{quantikz}
\lstick{$\ket{0}$} & \gate{H} & \ctrl{1} & \meter{} \\
\lstick{$\ket{0}$} & \qw & \targ{} & \meter{}
\end{quantikz}
\caption{Bell state preparation circuit.}
\end{figure}
```

**Common quantikz elements**:
- `\gate{U}` - Single qubit gate
- `\ctrl{n}` - Control (n wires away)
- `\targ{}` - Target (X gate)
- `\meter{}` - Measurement
- `\qw` - Quantum wire
- `\cw` - Classical wire
- `\lstick{text}` - Left label
- `\rstick{text}` - Right label

### Using qcircuit

```latex
\usepackage{qcircuit}

\begin{figure}
\centering
\Qcircuit @C=1em @R=.7em {
\lstick{\ket{0}} & \gate{H} & \ctrl{1} & \meter \\
\lstick{\ket{0}} & \qw & \targ & \meter
}
\caption{Bell state preparation circuit.}
\end{figure}
```

---

## Overleaf Integration

### Getting Started with Overleaf

1. Create account: https://www.overleaf.com/
2. Start new project from template
3. Upload existing files if needed
4. Enable collaboration for advisor access

### Overleaf Features

| Feature | How to Access |
|---------|---------------|
| Rich text mode | Toggle in editor |
| Word count | Menu → Word Count |
| Track changes | Review menu |
| History | History button |
| Git sync | Menu → Git |
| GitHub sync | Menu → GitHub |
| Comments | Select text → Add comment |

### Overleaf + Git Integration

```bash
# Clone Overleaf project
git clone https://git.overleaf.com/PROJECT_ID

# Work locally, then push
git add .
git commit -m "Local changes"
git push
```

---

## Common LaTeX Issues and Solutions

### Compilation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Missing $ inserted` | Math mode error | Check for unescaped `_` or `^` |
| `Undefined control sequence` | Unknown command | Check package included |
| `File not found` | Image path wrong | Check path and extension |
| `Missing \begin{document}` | Preamble error | Check for syntax errors before `\begin{document}` |

### Float Placement

```latex
% Force figure placement
\begin{figure}[H]  % Requires float package
\begin{figure}[!htbp]  % Try here, top, bottom, page

% For problematic floats
\clearpage  % Force all pending floats
```

### Long Tables

```latex
% Table spanning multiple pages
\begin{longtable}{lcc}
\caption{Long table caption} \\
\toprule
Header 1 & Header 2 & Header 3 \\
\midrule
\endfirsthead

\multicolumn{3}{c}{{\tablename\ \thetable{} -- continued}} \\
\toprule
Header 1 & Header 2 & Header 3 \\
\midrule
\endhead

\midrule
\multicolumn{3}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot

Data 1 & Data 2 & Data 3 \\
% ... more rows ...
\end{longtable}
```

### Bibliography Issues

```latex
% If bibliography not appearing
% 1. Run: pdflatex → biber → pdflatex → pdflatex
% 2. Check .bib file path in \addbibresource{}
% 3. Verify entry keys match citations

% Common biber errors
% - Check for special characters in entries
% - Ensure proper UTF-8 encoding
```

---

## Typography Best Practices

### Math Typesetting

```latex
% Good practices
\[ E = mc^2 \]  % Display math

$E = mc^2$  % Inline math

% Use align for multi-line equations
\begin{align}
    a &= b + c \\
    d &= e + f
\end{align}

% Use proper operators
\sin, \cos, \log, \exp  % Not sin, cos, etc.

% Custom operators
\DeclareMathOperator{\Tr}{Tr}  % Then use \Tr
```

### Text Formatting

```latex
% Emphasis
\emph{emphasized text}  % Preferred
\textit{italic text}    % Direct italic

% Strong emphasis
\textbf{bold text}

% Avoid
% Underline (\underline{}) - looks unprofessional
% ALL CAPS - hard to read
```

### Spacing

```latex
% Non-breaking space (between number and unit)
Figure~\ref{fig:example}  % Prevents bad line breaks
5~km

% Thin space in math
$f(x) \, dx$  % Small space
$f(x)\,dx$    % Also acceptable

% Em-dash
---  % For parenthetical remarks—like this
--   % For number ranges: pages 5--10
```

---

## Useful Resources

### Documentation

| Resource | URL |
|----------|-----|
| Overleaf Documentation | https://www.overleaf.com/learn |
| TeX StackExchange | https://tex.stackexchange.com/ |
| CTAN Package List | https://ctan.org/ |
| LaTeX Wikibook | https://en.wikibooks.org/wiki/LaTeX |
| ShareLaTeX Tutorial | https://www.overleaf.com/learn/latex/Tutorials |

### Books

| Title | Authors | Notes |
|-------|---------|-------|
| *LaTeX Companion* | Mittelbach & Goossens | Comprehensive reference |
| *More Math Into LaTeX* | Grätzer | Math typesetting focus |
| *LaTeX Beginner's Guide* | Kottwitz | Good for beginners |

### Tools

| Tool | Purpose | URL |
|------|---------|-----|
| Detexify | Draw symbol to find command | https://detexify.kirelabs.org/ |
| TableGenerator | Create LaTeX tables | https://www.tablesgenerator.com/ |
| Mathpix | Image to LaTeX | https://mathpix.com/ |
| latexdiff | Track changes between versions | CTAN package |

---

## Quick Reference Card

### Essential Commands

```latex
% Document structure
\chapter{}, \section{}, \subsection{}

% Cross-references
\label{key}, \ref{key}, \pageref{key}

% Citations
\cite{key}, \citep{key}, \citet{key}

% Figures
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{file.pdf}
\caption{Caption}\label{fig:label}
\end{figure}

% Tables
\begin{table}[htbp]
\centering
\caption{Caption}\label{tab:label}
\begin{tabular}{lcc}
\toprule
... & ... & ... \\
\bottomrule
\end{tabular}
\end{table}

% Equations
\begin{equation}\label{eq:label}
...
\end{equation}
```

---

*"LaTeX is a document preparation system that is very good at setting type. It is very bad at everything else." — Overheard at academic conference*
