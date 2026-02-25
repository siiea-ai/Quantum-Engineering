# Scientific Writing Tools Guide

## LaTeX, REVTeX, and Professional Publishing

---

## Table of Contents

1. [LaTeX Fundamentals](#1-latex-fundamentals)
2. [Mathematical Typesetting](#2-mathematical-typesetting)
3. [Quantum Mechanics Notation](#3-quantum-mechanics-notation)
4. [REVTeX for Physics Journals](#4-revtex-for-physics-journals)
5. [Figures and Graphics](#5-figures-and-graphics)
6. [Bibliography Management](#6-bibliography-management)
7. [Collaborative Writing](#7-collaborative-writing)

---

## 1. LaTeX Fundamentals

### 1.1 Document Structure

```latex
\documentclass[12pt]{article}

% Preamble - packages and settings
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}

% Custom commands
\newcommand{\ket}[1]{\left|#1\right\rangle}
\newcommand{\bra}[1]{\left\langle#1\right|}

% Document metadata
\title{Your Paper Title}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
Your abstract text here.
\end{abstract}

\section{Introduction}
Your content here.

\section{Methods}
More content.

\section{Results}
Results section.

\section{Conclusion}
Conclusions.

\bibliography{references}

\end{document}
```

### 1.2 Essential Packages

```latex
% Mathematics
\usepackage{amsmath}      % Core math environments
\usepackage{amssymb}      % Math symbols
\usepackage{amsthm}       % Theorem environments
\usepackage{mathtools}    % Extended math tools

% Physics
\usepackage{physics}      % Physics notation
\usepackage{braket}       % Dirac notation
\usepackage{siunitx}      % SI units

% Graphics
\usepackage{graphicx}     % Include images
\usepackage{tikz}         % Programmatic graphics
\usepackage{quantikz}     % Quantum circuits

% Tables
\usepackage{booktabs}     % Professional tables
\usepackage{array}        % Extended columns
\usepackage{multirow}     % Spanning rows

% References
\usepackage{hyperref}     % Hyperlinks
\usepackage{cleveref}     % Smart references
\usepackage[style=phys]{biblatex}  % Bibliography

% Layout
\usepackage{geometry}     % Page margins
\usepackage{setspace}     % Line spacing
\usepackage{fancyhdr}     % Headers/footers
```

### 1.3 Text Formatting

```latex
% Emphasis
\emph{emphasized text}
\textbf{bold text}
\textit{italic text}
\texttt{typewriter text}

% Lists
\begin{itemize}
    \item First item
    \item Second item
\end{itemize}

\begin{enumerate}
    \item Numbered item
    \item Another item
\end{enumerate}

\begin{description}
    \item[Term] Definition
    \item[Another] Description
\end{description}

% Quotes
\begin{quote}
    Quoted text here.
\end{quote}

% Verbatim (code)
\begin{verbatim}
Code exactly as typed
\end{verbatim}
```

---

## 2. Mathematical Typesetting

### 2.1 Basic Math Modes

```latex
% Inline math
The energy is $E = mc^2$.

% Display math (numbered)
\begin{equation}
    E = mc^2
    \label{eq:einstein}
\end{equation}

% Display math (unnumbered)
\begin{equation*}
    E = mc^2
\end{equation*}

% Or using brackets
\[
    E = mc^2
\]
```

### 2.2 Multi-line Equations

```latex
% Aligned equations
\begin{align}
    \hat{H} &= \hat{T} + \hat{V} \\
            &= -\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r}) \\
            &= \sum_i \epsilon_i \ket{i}\bra{i}
\end{align}

% Split long equations
\begin{equation}
\begin{split}
    \langle\psi|\hat{H}|\psi\rangle &= \int \psi^*(\mathbf{r})
        \hat{H} \psi(\mathbf{r}) \, d^3r \\
    &= \sum_n c_n^* c_n E_n
\end{split}
\end{equation}

% Cases
\begin{equation}
    |n\rangle = \begin{cases}
        \ket{0} & \text{if } n = 0 \\
        \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}\ket{0} & \text{if } n > 0
    \end{cases}
\end{equation}
```

### 2.3 Matrices and Arrays

```latex
% Matrix
\begin{equation}
    \sigma_x = \begin{pmatrix}
        0 & 1 \\
        1 & 0
    \end{pmatrix}
\end{equation}

% Different bracket styles
\begin{bmatrix} a & b \\ c & d \end{bmatrix}  % Square brackets
\begin{vmatrix} a & b \\ c & d \end{vmatrix}  % Vertical bars (determinant)
\begin{Bmatrix} a & b \\ c & d \end{Bmatrix}  % Curly braces

% General array
\begin{equation}
    \begin{array}{c|c}
        \text{State} & \text{Energy} \\
        \hline
        \ket{0} & E_0 \\
        \ket{1} & E_1
    \end{array}
\end{equation}

% Small inline matrix
$\bigl(\begin{smallmatrix} a & b \\ c & d \end{smallmatrix}\bigr)$
```

### 2.4 Operators and Symbols

```latex
% Common operators
\hat{H}     % Hamiltonian
\hat{a}     % Annihilation operator
\hat{a}^\dagger  % Creation operator
\nabla      % Gradient
\partial    % Partial derivative

% Fractions
\frac{a}{b}
\dfrac{a}{b}  % Display style in inline
\tfrac{a}{b}  % Text style in display

% Sums and integrals
\sum_{i=1}^{N}
\prod_{j=1}^{M}
\int_0^\infty
\oint
\iint

% Limits
\lim_{x \to 0}
\sup_{x \in S}

% Brackets that scale
\left( \frac{a}{b} \right)
\left[ \sum_i x_i \right]
\left\{ \text{set} \right\}
\left\langle \phi | \psi \right\rangle
```

---

## 3. Quantum Mechanics Notation

### 3.1 Dirac Notation

```latex
% Using physics package
\usepackage{physics}

% Bra-ket notation
\ket{\psi}              % |ψ⟩
\bra{\phi}              % ⟨φ|
\braket{\phi}{\psi}     % ⟨φ|ψ⟩
\ketbra{\psi}{\phi}     % |ψ⟩⟨φ|
\mel{\phi}{\hat{O}}{\psi}  % ⟨φ|Ô|ψ⟩
\ev{\hat{O}}{\psi}      % ⟨ψ|Ô|ψ⟩

% Custom commands if not using physics package
\newcommand{\ket}[1]{\left|#1\right\rangle}
\newcommand{\bra}[1]{\left\langle#1\right|}
\newcommand{\braket}[2]{\left\langle#1|#2\right\rangle}
\newcommand{\ketbra}[2]{\left|#1\right\rangle\left\langle#2\right|}
\newcommand{\mel}[3]{\left\langle#1\right|#2\left|#3\right\rangle}
```

### 3.2 Quantum States

```latex
% Common states
\ket{0}, \ket{1}                  % Computational basis
\ket{+} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1})
\ket{-} = \frac{1}{\sqrt{2}}(\ket{0} - \ket{1})

% Multi-qubit states
\ket{00}, \ket{01}, \ket{10}, \ket{11}
\ket{\psi} = \sum_{i} c_i \ket{i}

% Tensor products
\ket{\psi} \otimes \ket{\phi}
\ket{\psi}^{\otimes n}

% Bell states
\ket{\Phi^+} = \frac{1}{\sqrt{2}}(\ket{00} + \ket{11})
\ket{\Phi^-} = \frac{1}{\sqrt{2}}(\ket{00} - \ket{11})
\ket{\Psi^+} = \frac{1}{\sqrt{2}}(\ket{01} + \ket{10})
\ket{\Psi^-} = \frac{1}{\sqrt{2}}(\ket{01} - \ket{10})
```

### 3.3 Operators and Commutators

```latex
% Using physics package
\comm{\hat{A}}{\hat{B}}     % [A, B]
\acomm{\hat{A}}{\hat{B}}    % {A, B}

% Pauli matrices
\sigma_x, \sigma_y, \sigma_z
\hat{\sigma}_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}

% Identity
\mathbb{I}, \mathbb{1}

% Trace
\Tr{\hat{\rho}}
\Tr_B\left[\hat{\rho}_{AB}\right]

% Tensor product
\otimes
\bigotimes_{i=1}^{n}
```

### 3.4 Density Matrices

```latex
% Pure state
\hat{\rho} = \ketbra{\psi}{\psi}

% Mixed state
\hat{\rho} = \sum_i p_i \ketbra{\psi_i}{\psi_i}

% Partial trace
\hat{\rho}_A = \Tr_B\left[\hat{\rho}_{AB}\right]

% Entropy
S(\hat{\rho}) = -\Tr{\hat{\rho} \log \hat{\rho}}
```

---

## 4. REVTeX for Physics Journals

### 4.1 Basic REVTeX Document

```latex
\documentclass[
    aps,              % American Physical Society
    prl,              % Physical Review Letters style
    reprint,          % Two-column format
    superscriptaddress,
    showpacs,
    showkeys
]{revtex4-2}

\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{physics}

\begin{document}

\preprint{APS/123-QED}

\title{Variational Quantum Eigensolver for Molecular Hamiltonians}

\author{First Author}
\email{first@university.edu}
\affiliation{Department of Physics, University Name, City, Country}

\author{Second Author}
\affiliation{Department of Physics, University Name, City, Country}
\affiliation{Quantum Computing Center, Lab Name, City, Country}

\date{\today}

\begin{abstract}
We present a novel approach to computing molecular ground state energies
using variational quantum algorithms. Our method achieves chemical accuracy
for small molecules using hardware-efficient ansatze on near-term quantum
devices.
\end{abstract}

\pacs{03.67.Ac, 31.15.xp, 03.67.Lx}
\keywords{quantum computing, VQE, molecular simulation}

\maketitle

\section{Introduction}
\label{sec:intro}

The simulation of quantum systems...

\section{Methods}
\label{sec:methods}

We employ the variational quantum eigensolver (VQE)...

\section{Results}
\label{sec:results}

Our numerical experiments demonstrate...

\section{Conclusion}
\label{sec:conclusion}

We have shown that...

\begin{acknowledgments}
This work was supported by...
\end{acknowledgments}

\bibliography{references}

\end{document}
```

### 4.2 REVTeX Options

```latex
% Journal styles
\documentclass[aps,prl,...]{revtex4-2}   % PRL
\documentclass[aps,pra,...]{revtex4-2}   % PRA
\documentclass[aps,prb,...]{revtex4-2}   % PRB
\documentclass[aps,prx,...]{revtex4-2}   % PRX

% Layout options
reprint           % Two-column final format
preprint          % One-column draft format
twocolumn         % Explicit two-column
onecolumn         % Explicit one-column
showpacs          % Show PACS numbers
showkeys          % Show keywords
superscriptaddress % Compact author format
groupedaddress    % Authors grouped by affiliation

% Other options
draft             % Show overfull boxes
linenumbers       % Add line numbers (for review)
longbibliography  % Full bibliography formatting
```

### 4.3 Supplementary Materials

```latex
% In main document
\appendix

\section{Derivation Details}
\label{app:derivation}

Full derivation of Eq.~\eqref{eq:main}...

% Separate supplementary file
\documentclass[aps,prl,superscriptaddress]{revtex4-2}
\begin{document}
\title{Supplementary Materials for: Main Title}
\maketitle

\section{Additional Data}
...
\end{document}
```

---

## 5. Figures and Graphics

### 5.1 Including Figures

```latex
\usepackage{graphicx}

% Single figure
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figure.pdf}
    \caption{Description of figure. (a) First panel description.
             (b) Second panel description.}
    \label{fig:example}
\end{figure}

% Wide figure (two-column)
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{wide_figure.pdf}
    \caption{Wide figure spanning both columns.}
    \label{fig:wide}
\end{figure*}

% Subfigures
\usepackage{subcaption}

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\columnwidth}
        \includegraphics[width=\linewidth]{fig_a.pdf}
        \caption{}
        \label{fig:sub_a}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\columnwidth}
        \includegraphics[width=\linewidth]{fig_b.pdf}
        \caption{}
        \label{fig:sub_b}
    \end{subfigure}
    \caption{(a) First subfigure. (b) Second subfigure.}
    \label{fig:subfigs}
\end{figure}
```

### 5.2 Quantum Circuits with quantikz

```latex
\usepackage{quantikz}

% Basic circuit
\begin{figure}[htbp]
\centering
\begin{quantikz}
    \lstick{$\ket{0}$} & \gate{H} & \ctrl{1} & \meter{} \\
    \lstick{$\ket{0}$} & \qw      & \targ{}  & \meter{}
\end{quantikz}
\caption{Bell state preparation circuit.}
\end{figure}

% More complex circuit
\begin{quantikz}
    \lstick{$\ket{0}$} & \gate{R_y(\theta_1)} & \ctrl{1} & \gate{R_z(\phi_1)} & \qw \\
    \lstick{$\ket{0}$} & \gate{R_y(\theta_2)} & \targ{}  & \gate{R_z(\phi_2)} & \qw
\end{quantikz}

% Gates
\gate{H}           % Hadamard
\gate{X}           % Pauli X
\gate{R_y(\theta)} % Rotation
\ctrl{1}           % Control (down 1)
\targ{}            % Target (CNOT)
\meter{}           % Measurement
\qw                % Wire
\cw                % Classical wire
```

### 5.3 TikZ Diagrams

```latex
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric}

% Bloch sphere
\begin{tikzpicture}
    % Sphere
    \shade[ball color=blue!10!white,opacity=0.5] (0,0) circle (1.5cm);
    \draw (0,0) circle (1.5cm);

    % Axes
    \draw[->] (0,0) -- (1.8,0) node[right] {$x$};
    \draw[->] (0,0) -- (0,1.8) node[above] {$z$};
    \draw[->] (0,0) -- (-0.9,-0.9) node[below left] {$y$};

    % State vector
    \draw[->,thick,red] (0,0) -- (0.7,1.2) node[above right] {$|\psi\rangle$};

    % Labels
    \node at (0,1.7) {$|0\rangle$};
    \node at (0,-1.7) {$|1\rangle$};
\end{tikzpicture}

% Energy diagram
\begin{tikzpicture}
    \draw (0,0) -- (2,0) node[right] {$E_0$};
    \draw (0,1) -- (2,1) node[right] {$E_1$};
    \draw (0,2.5) -- (2,2.5) node[right] {$E_2$};
    \draw[->,thick] (1,0.1) -- (1,0.9);
    \node at (1.3,0.5) {$\hbar\omega$};
\end{tikzpicture}
```

### 5.4 Matplotlib Integration

```python
import matplotlib.pyplot as plt
import numpy as np

# Configure for LaTeX
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.figsize': (3.5, 2.5),  # Single column
    'figure.dpi': 300,
})

# Create figure
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
ax.plot(x, np.sin(x), label=r'$\sin(x)$')
ax.plot(x, np.cos(x), label=r'$\cos(x)$')
ax.set_xlabel(r'$x$ (rad)')
ax.set_ylabel(r'$y$')
ax.legend()
ax.grid(True, alpha=0.3)

# Save for LaTeX
fig.savefig('figure.pdf', bbox_inches='tight')
```

---

## 6. Bibliography Management

### 6.1 BibTeX Entries

```bibtex
% references.bib

@article{peruzzo2014,
    author = {Peruzzo, Alberto and others},
    title = {A variational eigenvalue solver on a photonic quantum processor},
    journal = {Nature Communications},
    volume = {5},
    pages = {4213},
    year = {2014},
    doi = {10.1038/ncomms5213}
}

@article{preskill2018,
    author = {Preskill, John},
    title = {Quantum Computing in the {NISQ} era and beyond},
    journal = {Quantum},
    volume = {2},
    pages = {79},
    year = {2018},
    doi = {10.22331/q-2018-08-06-79}
}

@book{nielsen2010,
    author = {Nielsen, Michael A. and Chuang, Isaac L.},
    title = {Quantum Computation and Quantum Information},
    publisher = {Cambridge University Press},
    year = {2010},
    edition = {10th Anniversary},
    isbn = {978-1-107-00217-3}
}

@inproceedings{kitaev1995,
    author = {Kitaev, A. Yu.},
    title = {Quantum measurements and the Abelian Stabilizer Problem},
    booktitle = {Proceedings of the 35th Annual Symposium on
                 Foundations of Computer Science},
    year = {1995},
    pages = {866--875}
}

@misc{qiskit2024,
    author = {{Qiskit contributors}},
    title = {Qiskit: An Open-source Framework for Quantum Computing},
    year = {2024},
    doi = {10.5281/zenodo.2573505},
    howpublished = {\url{https://github.com/Qiskit/qiskit}}
}

@phdthesis{smith2020,
    author = {Smith, John},
    title = {Variational Quantum Algorithms for Near-Term Devices},
    school = {University of Example},
    year = {2020}
}
```

### 6.2 Citation Commands

```latex
% With natbib (REVTeX default)
\cite{peruzzo2014}              % [1]
\cite{peruzzo2014,preskill2018} % [1,2]
\onlinecite{nielsen2010}        % Nielsen and Chuang [3]

% With biblatex
\cite{peruzzo2014}
\textcite{peruzzo2014}     % Peruzzo et al. [1]
\parencite{peruzzo2014}    % [1]
\fullcite{peruzzo2014}     % Full reference inline
```

### 6.3 Bibliography Style

```latex
% REVTeX (automatic)
\bibliography{references}

% biblatex
\usepackage[
    style=phys,        % Physics style
    articletitle=true,
    biblabel=brackets,
    chaptertitle=false,
    pageranges=false
]{biblatex}
\addbibresource{references.bib}

% In document
\printbibliography
```

---

## 7. Collaborative Writing

### 7.1 Overleaf Setup

```latex
% Project structure
main.tex           % Main document
preamble.tex       % Package and command definitions
sections/
    introduction.tex
    methods.tex
    results.tex
    conclusion.tex
figures/
    fig1.pdf
    fig2.pdf
references.bib

% main.tex
\documentclass[aps,prl,reprint]{revtex4-2}
\input{preamble}

\begin{document}
\title{Paper Title}
\author{...}
\maketitle

\input{sections/introduction}
\input{sections/methods}
\input{sections/results}
\input{sections/conclusion}

\bibliography{references}
\end{document}
```

### 7.2 Comments and Track Changes

```latex
% Simple comments
% TODO: Add more data

% Highlighted comments
\usepackage{todonotes}
\todo{Fix this equation}
\todo[inline]{Add reference here}

% Track changes
\usepackage{changes}
\added[id=JD]{new text}
\deleted[id=JD]{removed text}
\replaced[id=JD]{new text}{old text}
\comment[id=JD]{Comment here}
```

### 7.3 Git Integration

```bash
# Clone from Overleaf
git clone https://git.overleaf.com/your_project_id

# Regular workflow
git pull origin master
# Make changes
git add .
git commit -m "Updated results section"
git push origin master
```

---

## Quick Reference

### Math Symbols Cheat Sheet

| Symbol | LaTeX | Symbol | LaTeX |
|--------|-------|--------|-------|
| $$\alpha$$ | `\alpha` | $$\hbar$$ | `\hbar` |
| $$\beta$$ | `\beta` | $$\nabla$$ | `\nabla` |
| $$\psi$$ | `\psi` | $$\partial$$ | `\partial` |
| $$\Psi$$ | `\Psi` | $$\infty$$ | `\infty` |
| $$\otimes$$ | `\otimes` | $$\dagger$$ | `\dagger` |
| $$\sum$$ | `\sum` | $$\prod$$ | `\prod` |
| $$\int$$ | `\int` | $$\oint$$ | `\oint` |

### Common Mistakes

1. **Missing $** - Math mode not enabled
2. **Underfull hbox** - Paragraph too short
3. **Overfull hbox** - Line too long
4. **Missing }** - Unbalanced braces
5. **Undefined control sequence** - Typo or missing package

### Compilation

```bash
# Basic compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# With latexmk (recommended)
latexmk -pdf main.tex

# Clean auxiliary files
latexmk -c
```

---

*Week 212: Scientific Writing Tools - Complete guide to professional physics publishing with LaTeX and REVTeX.*
