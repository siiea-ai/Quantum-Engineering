# LaTeX Thesis Template

## Complete Template for Quantum Computing PhD Thesis

---

## Overview

This document provides a complete, ready-to-use LaTeX thesis template optimized for quantum computing and physics dissertations. Copy and adapt for your thesis.

---

## Main Document (thesis.tex)

```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PhD Thesis Template for Quantum Science and Engineering
%
% Customize this template for your institution's requirements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass[12pt, letterpaper, oneside]{book}

% Page geometry
\usepackage[
    top=1in,
    bottom=1in,
    left=1.5in,
    right=1in
]{geometry}

% Line spacing
\usepackage{setspace}
\doublespacing

% Include preamble (packages and settings)
\input{preamble}

% Bibliography
\addbibresource{references.bib}

% Document metadata
\title{Your Thesis Title Here: A Study of Quantum Something}
\author{Your Full Name}
\date{Month Year}
\newcommand{\degree}{Doctor of Philosophy}
\newcommand{\field}{Quantum Science and Engineering}
\newcommand{\university}{University Name}
\newcommand{\department}{Department of Physics}
\newcommand{\advisor}{Professor Advisor Name}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{document}

%--- Front Matter ---
\frontmatter
\pagestyle{plain}

% Title Page
\include{chapters/titlepage}

% Copyright
\include{chapters/copyright}

% Abstract
\include{chapters/abstract}

% Dedication (optional)
% \include{chapters/dedication}

% Acknowledgments
\include{chapters/acknowledgments}

% Table of Contents
\tableofcontents
\clearpage

% List of Figures
\listoffigures
\clearpage

% List of Tables
\listoftables
\clearpage

% List of Abbreviations
\include{chapters/abbreviations}

%--- Main Matter ---
\mainmatter
\pagestyle{headings}

\include{chapters/chapter1_introduction}
\include{chapters/chapter2_background}
\include{chapters/chapter3_research1}
\include{chapters/chapter4_research2}
\include{chapters/chapter5_discussion}

%--- Back Matter ---
\backmatter

% Appendices
\appendix
\include{chapters/appendixA}
\include{chapters/appendixB}
\include{chapters/appendixC}

% Bibliography
\printbibliography[heading=bibintoc, title={Bibliography}]

\end{document}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

---

## Preamble (preamble.tex)

```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preamble: Packages and Custom Commands
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--- Core Packages ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}  % Improved Computer Modern fonts

%--- Mathematics ---
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{mathtools}
\usepackage{bm}  % Bold math symbols

%--- Quantum Computing Specific ---
\usepackage{braket}  % Dirac notation
\usepackage{quantikz}  % Quantum circuits
% Alternative: \usepackage{qcircuit}

%--- Figures and Graphics ---
\usepackage{graphicx}
\usepackage{subcaption}  % Subfigures
\usepackage{float}  % Better float placement
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, calc}

%--- Tables ---
\usepackage{booktabs}  % Professional tables
\usepackage{array}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{longtable}

%--- Algorithms ---
\usepackage{algorithm}
\usepackage{algpseudocode}

%--- Code Listings ---
\usepackage{listings}
\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{red},
    numbers=left,
    numberstyle=\tiny,
    frame=single,
    breaklines=true
}

%--- Bibliography ---
\usepackage[
    backend=biber,
    style=numeric-comp,
    sorting=none,
    maxbibnames=99,
    giveninits=true
]{biblatex}

%--- Cross-References ---
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue,
    pdfauthor={Your Name},
    pdftitle={Your Thesis Title}
}
\usepackage{cleveref}  % Smart cross-references

%--- Units ---
\usepackage{siunitx}

%--- Miscellaneous ---
\usepackage{enumitem}  % List customization
\usepackage{xcolor}  % Colors
\usepackage{epigraph}  % Chapter quotes


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Theorem Environments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}

\theoremstyle{definition}
\newtheorem{definition}{Definition}[chapter]
\newtheorem{example}{Example}[chapter]

\theoremstyle{remark}
\newtheorem{remark}{Remark}[chapter]
\newtheorem{note}{Note}[chapter]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Custom Commands - Quantum Mechanics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% State vectors (if not using braket package)
% \newcommand{\ket}[1]{\left| #1 \right\rangle}
% \newcommand{\bra}[1]{\left\langle #1 \right|}
% \newcommand{\braket}[2]{\left\langle #1 | #2 \right\rangle}
% \newcommand{\ketbra}[2]{\left| #1 \right\rangle\left\langle #2 \right|}
% \newcommand{\expval}[1]{\left\langle #1 \right\rangle}

% Common states
\newcommand{\zero}{\ket{0}}
\newcommand{\one}{\ket{1}}
\newcommand{\plus}{\ket{+}}
\newcommand{\minus}{\ket{-}}

% Density matrix
\newcommand{\rhostate}{\hat{\rho}}

% Operators
\newcommand{\id}{\mathbb{I}}  % Identity
\newcommand{\ham}{\hat{H}}  % Hamiltonian

% Trace and partial trace
\newcommand{\tr}{\operatorname{Tr}}
\newcommand{\ptr}[1]{\operatorname{Tr}_{#1}}

% Pauli matrices
\newcommand{\paulix}{\sigma_x}
\newcommand{\pauliy}{\sigma_y}
\newcommand{\pauliz}{\sigma_z}

% Commutator and anticommutator
\newcommand{\comm}[2]{\left[ #1, #2 \right]}
\newcommand{\acomm}[2]{\left\{ #1, #2 \right\}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Custom Commands - General
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Math operators
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator{\sgn}{sgn}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\poly}{poly}

% Complexity classes
\newcommand{\BQP}{\textbf{BQP}}
\newcommand{\NP}{\textbf{NP}}
\newcommand{\Ppoly}{\textbf{P}}
\newcommand{\QMA}{\textbf{QMA}}

% Common sets
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}

% Probability
\newcommand{\prob}[1]{\Pr\left[ #1 \right]}
\newcommand{\E}{\mathbb{E}}

% Big-O notation
\newcommand{\bigO}[1]{O\left( #1 \right)}
\newcommand{\smallo}[1]{o\left( #1 \right)}

% Vectors and matrices
\newcommand{\vect}[1]{\mathbf{#1}}
\newcommand{\mat}[1]{\mathbf{#1}}

% Absolute value and norm
\newcommand{\abs}[1]{\left| #1 \right|}
\newcommand{\norm}[1]{\left\| #1 \right\|}

% Notes and todos (remove for final version)
\newcommand{\todo}[1]{\textcolor{red}{\textbf{TODO:} #1}}
\newcommand{\note}[1]{\textcolor{blue}{\textbf{Note:} #1}}
```

---

## Title Page (chapters/titlepage.tex)

```latex
\begin{titlepage}
\centering

\vspace*{1in}

{\LARGE\bfseries \thetitle}

\vspace{1in}

{\Large A dissertation presented}\\[0.3em]
{\Large by}\\[0.3em]
{\Large\bfseries \theauthor}\\[0.3em]
{\Large to}\\[0.3em]
{\Large \thedepartment}

\vspace{1in}

{\Large in partial fulfillment of the requirements}\\[0.3em]
{\Large for the degree of}\\[0.3em]
{\Large\bfseries \degree}\\[0.3em]
{\Large in the subject of}\\[0.3em]
{\Large \field}

\vspace{1in}

{\Large \theuniversity}\\[0.3em]
{\Large Cambridge, Massachusetts}\\[0.5em]
{\Large \thedate}

\end{titlepage}
```

---

## Abstract (chapters/abstract.tex)

```latex
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}

\begin{center}
{\large\bfseries \thetitle}\\[1em]
{\large \theauthor}
\end{center}

\vspace{1em}

% Your abstract here (250-350 words)
[Write your abstract here. The abstract should concisely state:
(1) the motivation and context for your research,
(2) the specific problem addressed,
(3) your methodology and approach,
(4) key results and findings,
(5) conclusions and implications.
Avoid citations, abbreviations, and references to figures/tables.]

\vspace{1em}

\noindent\textbf{Thesis Supervisor:} \advisor\\
\textbf{Title:} Professor of Physics
```

---

## Abbreviations (chapters/abbreviations.tex)

```latex
\chapter*{List of Abbreviations}
\addcontentsline{toc}{chapter}{List of Abbreviations}

\begin{tabular}{ll}
CNOT & Controlled-NOT gate \\
CZ & Controlled-Z gate \\
NISQ & Noisy Intermediate-Scale Quantum \\
QC & Quantum Computing \\
QEC & Quantum Error Correction \\
QFT & Quantum Fourier Transform \\
QAOA & Quantum Approximate Optimization Algorithm \\
QPE & Quantum Phase Estimation \\
VQE & Variational Quantum Eigensolver \\
% Add your abbreviations here
\end{tabular}
```

---

## Chapter Template (chapters/chapterX.tex)

```latex
\chapter{Chapter Title}
\label{ch:chapterX}

% Optional chapter epigraph
\epigraph{A fitting quote about the chapter topic.}{--- Author Name}

\section{Introduction}
\label{sec:chX-intro}

[Chapter introduction text]


\section{Section Title}
\label{sec:chX-section}

[Section content]


\subsection{Subsection Title}
\label{subsec:chX-subsection}

[Subsection content]

% Example equation
\begin{equation}
\label{eq:example}
H = \sum_{i} h_i \pauliz_i + \sum_{i<j} J_{ij} \pauliz_i \pauliz_j
\end{equation}

As shown in \cref{eq:example}, the Hamiltonian...

% Example figure
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/chapterX/figure_name.pdf}
\caption{Figure caption describing the content.}
\label{fig:chX-figure}
\end{figure}

Referring to \cref{fig:chX-figure}, we observe...

% Example table
\begin{table}[htbp]
\centering
\caption{Table caption describing the content.}
\label{tab:chX-table}
\begin{tabular}{lcc}
\toprule
Column 1 & Column 2 & Column 3 \\
\midrule
Data 1 & Value 1 & Result 1 \\
Data 2 & Value 2 & Result 2 \\
\bottomrule
\end{tabular}
\end{table}

% Example theorem
\begin{theorem}[Theorem Name]
\label{thm:chX-theorem}
Statement of the theorem.
\end{theorem}

\begin{proof}
Proof of the theorem.
\end{proof}

% Example quantum circuit
\begin{figure}[htbp]
\centering
\begin{quantikz}
\lstick{$\ket{0}$} & \gate{H} & \ctrl{1} & \meter{} \\
\lstick{$\ket{0}$} & \qw & \targ{} & \meter{}
\end{quantikz}
\caption{Example quantum circuit creating a Bell state.}
\label{fig:chX-circuit}
\end{figure}


\section{Chapter Summary}
\label{sec:chX-summary}

[Summary of key points from this chapter]
```

---

## Bibliography Style Notes

### Common BibTeX Entry Types for Quantum Computing

```bibtex
% Journal article
@article{Nielsen2000,
    author = {Nielsen, Michael A. and Chuang, Isaac L.},
    title = {Quantum Computation and Quantum Information},
    journal = {Cambridge University Press},
    year = {2000},
}

% arXiv preprint
@article{Author2024arxiv,
    author = {Author, A. and Coauthor, B.},
    title = {Paper Title},
    year = {2024},
    eprint = {2401.12345},
    archiveprefix = {arXiv},
    primaryclass = {quant-ph},
}

% Conference paper
@inproceedings{Author2024conf,
    author = {Author, A.},
    title = {Paper Title},
    booktitle = {Proceedings of Conference},
    year = {2024},
    pages = {1--10},
}
```

---

## Compilation Instructions

### Using latexmk (Recommended)

```bash
# Compile with automatic bibliography
latexmk -pdf thesis.tex

# Clean auxiliary files
latexmk -c

# Full clean including PDF
latexmk -C
```

### Manual Compilation

```bash
pdflatex thesis.tex
biber thesis
pdflatex thesis.tex
pdflatex thesis.tex
```

### Overleaf Settings

- Compiler: pdfLaTeX or LuaLaTeX
- Main document: thesis.tex
- Bibliography: Biber

---

## Notes

- Customize page margins and spacing to match your institution's requirements
- Add/remove packages as needed for your specific content
- Test compilation regularly as you add content
- Keep backup copies of working versions
