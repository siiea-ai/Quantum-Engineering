# Introduction Chapter Template

## LaTeX Template

```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHAPTER 1: INTRODUCTION
% PhD Thesis - [Your Name]
% [University Name]
% [Year]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\chapter{Introduction}
\label{ch:introduction}

%------------------------------------------------------------------------------
% SECTION 1.1: MOTIVATION AND CONTEXT
%------------------------------------------------------------------------------
\section{Motivation and Context}
\label{sec:intro:motivation}

% OPENING HOOK (1-2 paragraphs)
% - Start with a compelling statement that captures the essence of your work
% - Can be a provocative claim, an important question, or a vision of the future
% - Should be accessible to a general scientific audience

[OPENING HOOK PARAGRAPH - REPLACE THIS]
% Example: "A large-scale quantum computer would revolutionize our ability to
% simulate quantum systems, breaking a computational barrier that has limited
% scientific progress for decades."

% BROAD CONTEXT (2-3 paragraphs)
% - Establish the field of quantum computing/information
% - Discuss why it matters (scientific and practical importance)
% - Reference landmark results (Shor, Grover, quantum simulation, etc.)

[BROAD CONTEXT PARAGRAPHS - REPLACE THIS]

% NARROWING TO YOUR AREA (2-3 paragraphs)
% - Transition from broad quantum computing to your specific subfield
% - Introduce quantum error correction if relevant
% - Discuss the specific challenge your thesis addresses

[NARROWING PARAGRAPHS - REPLACE THIS]

% THE PROBLEM (1-2 paragraphs)
% - State the specific problem or gap that motivates your work
% - Explain why solving this problem is important
% - Preview how your thesis addresses it

[PROBLEM STATEMENT PARAGRAPHS - REPLACE THIS]

%------------------------------------------------------------------------------
% SECTION 1.2: RESEARCH QUESTIONS
%------------------------------------------------------------------------------
\section{Research Questions}
\label{sec:intro:questions}

% INTRODUCTION TO QUESTIONS (1 paragraph)
% - Transition from motivation to specific questions
% - Explain how you will structure the questions

This thesis addresses the following research questions:

% RESEARCH QUESTION 1
\subsection{Research Question 1: [Title]}
\label{sec:intro:rq1}

\textbf{Question:} [State the question precisely and formally]

[1-2 paragraphs explaining:]
% - Why this question is important
% - What was known before your work
% - What your hypothesis is
% - How this question relates to the motivation

% RESEARCH QUESTION 2
\subsection{Research Question 2: [Title]}
\label{sec:intro:rq2}

\textbf{Question:} [State the question precisely and formally]

[1-2 paragraphs explaining the question]

% RESEARCH QUESTION 3
\subsection{Research Question 3: [Title]}
\label{sec:intro:rq3}

\textbf{Question:} [State the question precisely and formally]

[1-2 paragraphs explaining the question]

% OPTIONAL: Additional research questions
% Add RQ4, RQ5 as needed, typically 3-5 total

% CONNECTING THE QUESTIONS (1 paragraph)
% - Explain how the questions relate to each other
% - Show the logical progression

[PARAGRAPH CONNECTING THE QUESTIONS - REPLACE THIS]

%------------------------------------------------------------------------------
% SECTION 1.3: ORIGINAL CONTRIBUTIONS
%------------------------------------------------------------------------------
\section{Original Contributions}
\label{sec:intro:contributions}

% INTRODUCTION (1 paragraph)
This thesis makes the following original contributions to the field:

% CONTRIBUTION 1
\subsection*{Contribution 1: [Title]}
\label{contrib:1}

[2-3 paragraphs describing:]
% - What the contribution is (precisely stated)
% - Why it is novel (how it differs from prior work)
% - What impact it has (theoretical and/or practical)
% - Where it appears in the thesis (chapter reference)
% - Publication reference if applicable

% CONTRIBUTION 2
\subsection*{Contribution 2: [Title]}
\label{contrib:2}

[2-3 paragraphs describing the contribution]

% CONTRIBUTION 3
\subsection*{Contribution 3: [Title]}
\label{contrib:3}

[2-3 paragraphs describing the contribution]

% OPTIONAL: Additional contributions
% Add Contributions 4-7 as needed

% PUBLICATIONS LIST
\subsection*{Publications}
\label{sec:intro:publications}

The work in this thesis has been published or submitted as follows:

\begin{enumerate}
    \item [Author list], ``[Paper title],'' \textit{[Journal]}, [Volume], [Pages] ([Year]). [DOI or arXiv link]

    \item [Author list], ``[Paper title],'' \textit{[Journal]}, [Volume], [Pages] ([Year]). [DOI or arXiv link]

    \item [Author list], ``[Paper title],'' submitted to \textit{[Journal]} ([Year]). [arXiv link]
\end{enumerate}

%------------------------------------------------------------------------------
% SECTION 1.4: THESIS ORGANIZATION
%------------------------------------------------------------------------------
\section{Thesis Organization}
\label{sec:intro:organization}

% OVERVIEW (1 paragraph)
% - Briefly describe the structure
% - Mention how many chapters and their general purpose

This thesis is organized into [N] chapters, progressing from foundational
background through original research to conclusions and future directions.

% CHAPTER SUMMARIES
\subsection*{Chapter 2: Background}
[1 paragraph summary of background chapter]

\subsection*{Chapter 3: [First Research Chapter Title]}
[1 paragraph summary]

\subsection*{Chapter 4: [Second Research Chapter Title]}
[1 paragraph summary]

\subsection*{Chapter 5: [Third Research Chapter Title]}
[1 paragraph summary]

\subsection*{Chapter 6: Conclusions and Future Work}
[1 paragraph summary]

% THESIS STRUCTURE FIGURE
\begin{figure}[htbp]
    \centering
    % \includegraphics[width=0.9\textwidth]{figures/thesis_structure.pdf}
    \caption{Thesis structure and chapter relationships. Arrows indicate
    dependencies between chapters. [REPLACE WITH ACTUAL FIGURE]}
    \label{fig:thesis:structure}
\end{figure}

% READING GUIDE (optional, 1 paragraph)
% - Suggest how different readers might approach the thesis
% - Which chapters to read/skip depending on background

\paragraph{Reading Guide.} Readers familiar with quantum error correction may
wish to skim Chapter 2 and proceed directly to the research chapters.
Those seeking an introduction to the field should read Chapter 2 carefully
before continuing.

%------------------------------------------------------------------------------
% SECTION 1.5: NOTATION AND CONVENTIONS (Optional)
%------------------------------------------------------------------------------
\section{Notation and Conventions}
\label{sec:intro:notation}

% Often this section appears in the Background chapter instead
% Include here if notation is used extensively in the introduction

Throughout this thesis, we adopt the following notation and conventions:

\begin{itemize}
    \item We use Dirac notation for quantum states: $\ket{\psi}$ for state vectors.

    \item Operators are denoted by capital letters (e.g., $H$ for Hamiltonians,
    $U$ for unitaries).

    \item The Pauli operators are $X$, $Y$, and $Z$, with
    $X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, etc.

    \item We work in units where $\hbar = 1$ unless otherwise stated.

    \item Logarithms are base-2 ($\log$) unless otherwise indicated.
\end{itemize}

A complete notation glossary appears in Appendix~\ref{app:notation}.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF INTRODUCTION CHAPTER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

---

## Section Length Guidelines

| Section | Pages | Words | Purpose |
|---------|-------|-------|---------|
| 1.1 Motivation | 3-4 | 900-1200 | Hook and contextualize |
| 1.2 Research Questions | 3-4 | 900-1200 | Articulate what you studied |
| 1.3 Contributions | 4-5 | 1200-1500 | State your novel results |
| 1.4 Organization | 2-3 | 600-900 | Guide the reader |
| 1.5 Notation | 1-2 | 300-600 | Set conventions |
| **Total** | **15-20** | **4500-6000** | |

---

## Checklist Before Submission

### Content Checklist
- [ ] Opening hook is compelling and accessible
- [ ] Motivation covers both theoretical and practical aspects
- [ ] Research questions are specific and answerable
- [ ] Each contribution is clearly stated and significant
- [ ] Chapter summaries are accurate and complete
- [ ] Thesis structure figure is included and clear

### Technical Checklist
- [ ] All equations are correct and properly formatted
- [ ] All references are accurate and consistently formatted
- [ ] Figure labels match text references
- [ ] Cross-references use `\ref{}` or `\cref{}`
- [ ] Section labels are logical and consistent

### Style Checklist
- [ ] Voice is consistent (first person plural or singular)
- [ ] Active voice is used where appropriate
- [ ] Paragraphs have clear topic sentences
- [ ] Transitions between sections are smooth
- [ ] No jargon without explanation for first use
- [ ] Length is appropriate for each section
