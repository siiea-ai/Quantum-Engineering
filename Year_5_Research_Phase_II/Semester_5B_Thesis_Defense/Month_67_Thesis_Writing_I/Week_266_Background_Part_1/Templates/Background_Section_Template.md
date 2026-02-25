# Background Section Template

## LaTeX Template for QM/QIT Background

```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHAPTER 2: BACKGROUND (Part 1: QM & QIT)
% PhD Thesis - [Your Name]
% [University Name]
% [Year]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\chapter{Background}
\label{ch:background}

% CHAPTER INTRODUCTION
This chapter establishes the theoretical foundations necessary to understand
the contributions of this thesis. We begin with quantum mechanics fundamentals
(\cref{sec:bg:qm}), proceed to quantum information theory (\cref{sec:bg:qit}),
and continue with quantum error correction in \cref{sec:bg:qec}.

Readers familiar with quantum information may wish to skim this chapter,
referring back as needed. Those new to the field should read carefully,
as the notation established here is used throughout the thesis.

%==============================================================================
% SECTION 2.1: QUANTUM MECHANICS FOUNDATIONS
%==============================================================================
\section{Quantum Mechanics Foundations}
\label{sec:bg:qm}

% SECTION INTRODUCTION
We begin with the mathematical formalism of quantum mechanics, focusing on
the aspects most relevant to quantum information processing.

%------------------------------------------------------------------------------
% SUBSECTION: Quantum States
%------------------------------------------------------------------------------
\subsection{Quantum States}
\label{sec:bg:qm:states}

% PURE STATES
\subsubsection{Pure States and Hilbert Spaces}

The state of an isolated quantum system is described by a unit vector in a
complex Hilbert space $\mathcal{H}$. For a single qubit—the quantum analogue
of a classical bit—the Hilbert space is $\mathcal{H} = \mathbb{C}^2$, with
orthonormal basis states $\ket{0}$ and $\ket{1}$. An arbitrary pure qubit
state is
\begin{equation}
    \ket{\psi} = \alpha\ket{0} + \beta\ket{1},
    \label{eq:qubit}
\end{equation}
where $\alpha, \beta \in \mathbb{C}$ satisfy $|\alpha|^2 + |\beta|^2 = 1$.

[CONTINUE WITH: multi-qubit states, tensor products, computational basis...]

% MIXED STATES
\subsubsection{Density Operators}

When a quantum system is in a probabilistic mixture of pure states—either
due to incomplete knowledge or entanglement with other systems—we describe
it using a density operator $\rho$. For a system in pure state $\ket{\psi_i}$
with probability $p_i$, the density operator is
\begin{equation}
    \rho = \sum_i p_i \ketbra{\psi_i}{\psi_i},
    \label{eq:density}
\end{equation}
satisfying $\rho \geq 0$, $\tr(\rho) = 1$.

[CONTINUE WITH: purity, Bloch sphere representation, reduced density matrices...]

%------------------------------------------------------------------------------
% SUBSECTION: Quantum Evolution
%------------------------------------------------------------------------------
\subsection{Quantum Evolution}
\label{sec:bg:qm:evolution}

% UNITARY EVOLUTION
\subsubsection{Unitary Dynamics}

The evolution of an isolated quantum system is governed by the Schrödinger
equation. For time-independent Hamiltonians $H$, the state at time $t$ is
\begin{equation}
    \ket{\psi(t)} = U(t)\ket{\psi(0)}, \quad U(t) = e^{-iHt/\hbar}.
    \label{eq:schrodinger}
\end{equation}
The evolution operator $U$ is unitary: $U^\dagger U = UU^\dagger = I$.

[CONTINUE WITH: common gates, gate decomposition, universality...]

% QUANTUM CHANNELS
\subsubsection{Quantum Channels}

Open quantum systems—those interacting with an environment—undergo more
general evolutions described by quantum channels. A quantum channel
$\mathcal{E}: \mathcal{B}(\mathcal{H}_A) \to \mathcal{B}(\mathcal{H}_B)$ is
a completely positive, trace-preserving (CPTP) linear map.

Every quantum channel admits a Kraus representation:
\begin{equation}
    \mathcal{E}(\rho) = \sum_{k=1}^{K} E_k \rho E_k^\dagger,
    \label{eq:kraus}
\end{equation}
where the Kraus operators satisfy $\sum_k E_k^\dagger E_k = I$.

[CONTINUE WITH: common noise channels, Choi representation if needed...]

%------------------------------------------------------------------------------
% SUBSECTION: Quantum Measurement
%------------------------------------------------------------------------------
\subsection{Quantum Measurement}
\label{sec:bg:qm:measurement}

% PROJECTIVE MEASUREMENT
\subsubsection{Projective Measurements}

A projective measurement is described by an observable $M = \sum_m m P_m$,
where $P_m$ are orthogonal projectors corresponding to eigenvalue $m$. The
probability of outcome $m$ is
\begin{equation}
    p(m) = \tr(P_m \rho P_m) = \tr(P_m \rho),
    \label{eq:born}
\end{equation}
and the post-measurement state is $\rho' = P_m \rho P_m / p(m)$.

[CONTINUE WITH: measurement in different bases, non-demolition measurement...]

% GENERAL MEASUREMENT
\subsubsection{Generalized Measurements (POVMs)}

More generally, a measurement is described by a positive operator-valued
measure (POVM): a set of positive operators $\{E_m\}$ satisfying
$\sum_m E_m = I$. The probability of outcome $m$ is $p(m) = \tr(E_m \rho)$.

[CONTINUE WITH: relationship to projective, Neumark extension...]

%==============================================================================
% SECTION 2.2: QUANTUM INFORMATION THEORY
%==============================================================================
\section{Quantum Information Theory}
\label{sec:bg:qit}

% SECTION INTRODUCTION
Quantum information theory provides the mathematical framework for
understanding how quantum systems process and transmit information.

%------------------------------------------------------------------------------
% SUBSECTION: Entanglement
%------------------------------------------------------------------------------
\subsection{Entanglement}
\label{sec:bg:qit:entanglement}

% DEFINITION
\subsubsection{Separable and Entangled States}

A bipartite pure state $\ket{\psi}_{AB}$ is \emph{separable} (or product) if
it can be written as $\ket{\psi}_{AB} = \ket{\phi}_A \otimes \ket{\chi}_B$;
otherwise, it is \emph{entangled}.

[CONTINUE WITH: Bell states, Schmidt decomposition, entanglement measures...]

% BELL STATES
\subsubsection{Bell States}

The four Bell states form a maximally entangled basis for two qubits:
\begin{align}
    \ket{\Phi^\pm} &= \frac{1}{\sqrt{2}}(\ket{00} \pm \ket{11}), \\
    \ket{\Psi^\pm} &= \frac{1}{\sqrt{2}}(\ket{01} \pm \ket{10}).
    \label{eq:bell}
\end{align}

[CONTINUE WITH: applications, teleportation, superdense coding...]

%------------------------------------------------------------------------------
% SUBSECTION: Quantum Entropy
%------------------------------------------------------------------------------
\subsection{Quantum Entropy}
\label{sec:bg:qit:entropy}

% VON NEUMANN ENTROPY
\subsubsection{Von Neumann Entropy}

The von Neumann entropy of a quantum state $\rho$ is
\begin{equation}
    S(\rho) = -\tr(\rho \log \rho) = -\sum_i \lambda_i \log \lambda_i,
    \label{eq:vn-entropy}
\end{equation}
where $\{\lambda_i\}$ are the eigenvalues of $\rho$ and we use $\log = \log_2$.

[CONTINUE WITH: properties, conditional entropy, mutual information...]

%------------------------------------------------------------------------------
% SUBSECTION: Notation Summary
%------------------------------------------------------------------------------
\subsection{Notation and Conventions}
\label{sec:bg:notation}

For reference, we summarize the notation used throughout this thesis in
\cref{tab:notation}.

\begin{table}[htbp]
\centering
\caption{Notation conventions used in this thesis.}
\label{tab:notation}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Symbol} & \textbf{Meaning} \\
\midrule
$\ket{\psi}$ & Pure quantum state (ket) \\
$\bra{\psi}$ & Dual state (bra) \\
$\rho$ & Density operator \\
$\mathcal{H}$ & Hilbert space \\
$\mathcal{B}(\mathcal{H})$ & Bounded operators on $\mathcal{H}$ \\
$X, Y, Z$ & Pauli operators \\
$H$ & Hadamard gate \\
$S, T$ & Phase and $\pi/8$ gates \\
$\text{CNOT}$ & Controlled-NOT gate \\
$\mathcal{E}$ & Quantum channel \\
$S(\rho)$ & Von Neumann entropy \\
$\log$ & Logarithm base 2 \\
\bottomrule
\end{tabular}
\end{table}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF BACKGROUND PART 1
% Continue to QEC Background in next file/week
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

---

## Section-by-Section Guidelines

### Quantum States Section

**What to include:**
- Hilbert space basics (dimension, inner product)
- Dirac notation ($\ket{\cdot}$, $\bra{\cdot}$, $\braket{\cdot}{\cdot}$)
- Qubits and the computational basis
- Bloch sphere representation (with figure)
- Multi-qubit systems and tensor products
- Density matrix formalism
- Partial trace and reduced states
- Distinguishing pure vs. mixed states

**Target length:** 4-5 pages

**Key figure:** Bloch sphere diagram

### Quantum Evolution Section

**What to include:**
- Schrödinger equation (brief)
- Unitary operators and their properties
- Common single-qubit gates (table with matrices)
- Common two-qubit gates (CNOT, CZ, iSWAP)
- Circuit model basics
- Quantum channels and CPTP maps
- Kraus representation
- Common noise models (table)

**Target length:** 4-5 pages

**Key figures:** Gate symbols, circuit examples

### Quantum Measurement Section

**What to include:**
- Measurement postulate
- Born rule
- Post-measurement states
- Measurement in different bases
- POVM formalism
- Connection to syndrome measurement

**Target length:** 2-3 pages

**Key figure:** Measurement circuit diagram

### Entanglement Section

**What to include:**
- Separable vs. entangled states
- Bell states (equations and properties)
- Schmidt decomposition
- Entanglement as a resource
- Entanglement entropy
- Brief mention of LOCC

**Target length:** 3-4 pages

**Key figure:** Entanglement diagram or circuit

### Entropy Section

**What to include:**
- Von Neumann entropy definition and properties
- Conditional entropy
- Mutual information
- Relative entropy (if used in thesis)
- Connection to error correction capacity

**Target length:** 2-3 pages

---

## Common Noise Models Table

Include this table or similar in your quantum channels section:

```latex
\begin{table}[htbp]
\centering
\caption{Common quantum noise channels and their Kraus operators.}
\label{tab:noise-channels}
\begin{tabular}{@{}llc@{}}
\toprule
\textbf{Channel} & \textbf{Kraus Operators} & \textbf{Parameter} \\
\midrule
Bit-flip & $E_0 = \sqrt{1-p}\,I$, $E_1 = \sqrt{p}\,X$ & $p$ \\
Phase-flip & $E_0 = \sqrt{1-p}\,I$, $E_1 = \sqrt{p}\,Z$ & $p$ \\
Depolarizing & $E_0 = \sqrt{1-p}\,I$, $E_k = \sqrt{p/3}\,\sigma_k$ & $p$ \\
Amplitude damping & $E_0 = \begin{pmatrix}1 & 0 \\ 0 & \sqrt{1-\gamma}\end{pmatrix}$,
                    $E_1 = \begin{pmatrix}0 & \sqrt{\gamma} \\ 0 & 0\end{pmatrix}$ & $\gamma$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Common Gates Table

Include this table or similar:

```latex
\begin{table}[htbp]
\centering
\caption{Common single-qubit gates.}
\label{tab:single-qubit-gates}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Gate} & \textbf{Symbol} & \textbf{Matrix} \\
\midrule
Pauli-X & $X$ & $\begin{pmatrix}0 & 1 \\ 1 & 0\end{pmatrix}$ \\[1ex]
Pauli-Y & $Y$ & $\begin{pmatrix}0 & -i \\ i & 0\end{pmatrix}$ \\[1ex]
Pauli-Z & $Z$ & $\begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix}$ \\[1ex]
Hadamard & $H$ & $\frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1 \\ 1 & -1\end{pmatrix}$ \\[1ex]
Phase & $S$ & $\begin{pmatrix}1 & 0 \\ 0 & i\end{pmatrix}$ \\[1ex]
$\pi/8$ & $T$ & $\begin{pmatrix}1 & 0 \\ 0 & e^{i\pi/4}\end{pmatrix}$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Checklist Before Proceeding

- [ ] All key concepts from your thesis are covered
- [ ] Notation is defined before first use
- [ ] Equations are numbered for reference
- [ ] Tables summarize key information
- [ ] Figures illustrate important concepts
- [ ] References to Nielsen & Chuang and other standards included
- [ ] Transitions connect sections smoothly
- [ ] No concepts are introduced that aren't used later
