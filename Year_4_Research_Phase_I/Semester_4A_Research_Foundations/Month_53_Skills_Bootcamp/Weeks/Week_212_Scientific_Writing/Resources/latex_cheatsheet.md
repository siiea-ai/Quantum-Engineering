# LaTeX Quick Reference for Quantum Computing

## Essential Commands

---

## Document Classes

```latex
% Physics journals
\documentclass[aps,prl,reprint]{revtex4-2}  % Physical Review Letters
\documentclass[aps,pra,reprint]{revtex4-2}  % Physical Review A
\documentclass[aps,prx,reprint]{revtex4-2}  % Physical Review X

% General
\documentclass[12pt]{article}               % Standard article
\documentclass[twocolumn]{article}          % Two-column
```

---

## Essential Packages

```latex
\usepackage{amsmath}      % Math environments
\usepackage{amssymb}      % Math symbols
\usepackage{physics}      % Physics notation (bra-ket)
\usepackage{graphicx}     % Include figures
\usepackage{hyperref}     % Links
\usepackage{cleveref}     % Smart references
\usepackage{siunitx}      % Units
\usepackage{booktabs}     % Professional tables
\usepackage{quantikz}     % Quantum circuits
```

---

## Math Mode

```latex
% Inline
$E = mc^2$

% Display (numbered)
\begin{equation}
    E = mc^2 \label{eq:einstein}
\end{equation}

% Display (unnumbered)
\begin{equation*}
    E = mc^2
\end{equation*}

% Or
\[ E = mc^2 \]
```

---

## Dirac Notation (with physics package)

| Notation | Command | Output |
|----------|---------|--------|
| Ket | `\ket{\psi}` | $$\vert\psi\rangle$$ |
| Bra | `\bra{\phi}` | $$\langle\phi\vert$$ |
| Bracket | `\braket{\phi}{\psi}` | $$\langle\phi\vert\psi\rangle$$ |
| Outer product | `\ketbra{\psi}{\phi}` | $$\vert\psi\rangle\langle\phi\vert$$ |
| Matrix element | `\mel{\phi}{A}{\psi}` | $$\langle\phi\vert A\vert\psi\rangle$$ |
| Expectation | `\ev{A}{\psi}` | $$\langle\psi\vert A\vert\psi\rangle$$ |

---

## Common Math Symbols

| Symbol | Command | Symbol | Command |
|--------|---------|--------|---------|
| $$\alpha$$ | `\alpha` | $$\hbar$$ | `\hbar` |
| $$\beta$$ | `\beta` | $$\nabla$$ | `\nabla` |
| $$\psi$$ | `\psi` | $$\partial$$ | `\partial` |
| $$\otimes$$ | `\otimes` | $$\dagger$$ | `\dagger` |
| $$\sum$$ | `\sum` | $$\prod$$ | `\prod` |
| $$\int$$ | `\int` | $$\infty$$ | `\infty` |

---

## Matrices

```latex
% Parentheses
\begin{pmatrix} a & b \\ c & d \end{pmatrix}

% Square brackets
\begin{bmatrix} a & b \\ c & d \end{bmatrix}

% Pauli matrices
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
```

---

## Aligned Equations

```latex
\begin{align}
    \hat{H} &= \hat{T} + \hat{V} \\
            &= -\frac{\hbar^2}{2m}\nabla^2 + V(r)
\end{align}
```

---

## Quantum Circuits (quantikz)

```latex
\begin{quantikz}
    \lstick{$\ket{0}$} & \gate{H} & \ctrl{1} & \meter{} \\
    \lstick{$\ket{0}$} & \qw      & \targ{}  & \meter{}
\end{quantikz}
```

### Circuit Elements

| Element | Command |
|---------|---------|
| Hadamard | `\gate{H}` |
| Rotation | `\gate{R_y(\theta)}` |
| Control | `\ctrl{1}` (target 1 below) |
| Target | `\targ{}` |
| Measurement | `\meter{}` |
| Wire | `\qw` |
| Classical | `\cw` |

---

## Figures

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figure.pdf}
    \caption{Caption text.}
    \label{fig:example}
\end{figure}
```

---

## Tables

```latex
\begin{table}[htbp]
\caption{Table caption.}
\label{tab:example}
\begin{ruledtabular}
\begin{tabular}{lcc}
    Method & Energy (Ha) & Error \\
    \midrule
    VQE & $-1.136$ & $10^{-3}$ \\
    Exact & $-1.137$ & --- \\
\end{tabular}
\end{ruledtabular}
\end{table}
```

---

## References

```latex
% In text
See Ref.~\cite{author2020}.
As shown in \cref{eq:main,fig:circuit}.

% Bibliography
\bibliography{references}
```

---

## Units (siunitx)

```latex
\SI{1.5}{\mega\hertz}       % 1.5 MHz
\SI{300}{\kelvin}           % 300 K
\SI{1.6e-3}{\hartree}       % 1.6×10⁻³ Ha
\num{1.23e-4}               % 1.23×10⁻⁴
```

---

## Cross-References (cleveref)

```latex
\cref{eq:main}       % Eq. (1)
\cref{fig:circuit}   % Fig. 1
\cref{tab:results}   % Table I
\Cref{sec:methods}   % Section II (start of sentence)
```

---

## Custom Commands

```latex
% Define in preamble
\newcommand{\ket}[1]{\left|#1\right\rangle}
\newcommand{\Ham}{\hat{H}}
\DeclareMathOperator{\Tr}{Tr}

% Use in document
$\ket{\psi}$, $\Ham$, $\Tr{\rho}$
```

---

## Compilation

```bash
# Basic
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Using latexmk (recommended)
latexmk -pdf main.tex

# Clean
latexmk -c
```

---

## BibTeX Entry Types

```bibtex
@article{key,
    author = {Last, First and Other, Author},
    title = {Article Title},
    journal = {Journal Name},
    volume = {1},
    pages = {100--110},
    year = {2024},
    doi = {10.1234/xxxxx}
}

@book{key,
    author = {Author, Name},
    title = {Book Title},
    publisher = {Publisher},
    year = {2024}
}

@misc{key,
    author = {Author},
    title = {Title},
    howpublished = {\url{https://example.com}},
    year = {2024}
}
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Missing $ | Math mode not started | Add `$...$` |
| Undefined control sequence | Typo or missing package | Check spelling, add package |
| Missing } | Unbalanced braces | Count braces |
| Overfull hbox | Line too long | Rephrase or use `\allowbreak` |

---

*Week 212: Scientific Writing Tools - LaTeX Quick Reference*
