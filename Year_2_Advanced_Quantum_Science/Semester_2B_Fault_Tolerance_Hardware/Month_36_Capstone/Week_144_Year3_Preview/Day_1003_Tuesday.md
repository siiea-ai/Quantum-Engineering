# Day 1003: Qualifying Exam Format & Preparation

## Schedule Overview (7 hours)

| Session | Duration | Focus |
|---------|----------|-------|
| Morning | 3 hours | Exam format details, sample problems from each section |
| Afternoon | 2 hours | Problem-solving strategies, timed practice techniques |
| Evening | 2 hours | Oral exam preparation, presentation and Q&A skills |

## Learning Objectives

By the end of today, you will be able to:

1. **Analyze** sample qualifying exam problems from all four sections
2. **Apply** effective problem-solving strategies under time pressure
3. **Develop** a systematic approach to written exam preparation
4. **Prepare** for oral examination presentation and questioning
5. **Identify** common mistakes and how to avoid them
6. **Create** a personalized practice exam schedule

## Core Content

### 1. Written Examination: Detailed Format

The written qualifying exam tests breadth and depth of knowledge across all core quantum science areas.

**Exam Specifications:**

$$\boxed{\text{Duration: 4 hours | 12 problems | 4 sections | Closed book}}$$

| Section | Problems | Points | Time Target |
|---------|----------|--------|-------------|
| A: Quantum Mechanics | 3 | 30 | 60 min |
| B: Quantum Information | 3 | 30 | 60 min |
| C: Error Correction | 3 | 30 | 60 min |
| D: Algorithms | 3 | 30 | 60 min |
| **Total** | **12** | **120** | **240 min** |

**Scoring Criteria:**

Each problem is scored 0-10 based on:
- Correctness of approach (3 points)
- Mathematical accuracy (3 points)
- Completeness of solution (2 points)
- Clarity of presentation (2 points)

**Passing Requirements:**
- Overall score: 70% (84/120 points)
- No section below 50% (15/30 points)
- All problems attempted

### 2. Sample Problems: Section A (Quantum Mechanics)

#### Sample Problem A1: Density Matrix Evolution

**Problem Statement:**
A spin-1/2 particle is prepared in the state $|\psi\rangle = \frac{1}{\sqrt{3}}|+\rangle + \sqrt{\frac{2}{3}}|-\rangle$ where $|+\rangle$ and $|-\rangle$ are eigenstates of $\sigma_z$.

(a) Calculate the density matrix $\rho = |\psi\rangle\langle\psi|$.

(b) The particle is subjected to a magnetic field causing Hamiltonian $H = \frac{\omega}{2}\sigma_z$. Find $\rho(t)$.

(c) Calculate $\langle\sigma_x\rangle(t)$ and $\langle\sigma_y\rangle(t)$.

(d) Describe the motion on the Bloch sphere.

**Solution Approach:**

(a) Density matrix:
$$|\psi\rangle = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 \\ 0 \end{pmatrix} + \sqrt{\frac{2}{3}}\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1/\sqrt{3} \\ \sqrt{2/3} \end{pmatrix}$$

$$\rho = |\psi\rangle\langle\psi| = \begin{pmatrix} 1/3 & \sqrt{2}/3 \\ \sqrt{2}/3 & 2/3 \end{pmatrix}$$

(b) Time evolution under $H = \frac{\omega}{2}\sigma_z$:
$$U(t) = e^{-iHt/\hbar} = \begin{pmatrix} e^{-i\omega t/2} & 0 \\ 0 & e^{i\omega t/2} \end{pmatrix}$$

$$\rho(t) = U(t)\rho U^\dagger(t) = \begin{pmatrix} 1/3 & \frac{\sqrt{2}}{3}e^{-i\omega t} \\ \frac{\sqrt{2}}{3}e^{i\omega t} & 2/3 \end{pmatrix}$$

(c) Expectation values:
$$\langle\sigma_x\rangle = \text{Tr}(\rho\sigma_x) = \frac{2\sqrt{2}}{3}\cos(\omega t)$$
$$\langle\sigma_y\rangle = \text{Tr}(\rho\sigma_y) = \frac{2\sqrt{2}}{3}\sin(\omega t)$$

(d) The Bloch vector precesses around the z-axis with frequency $\omega$.

#### Sample Problem A2: Perturbation Theory

**Problem Statement:**
A particle in a 1D infinite square well (width $a$) is perturbed by $V(x) = V_0 x/a$.

(a) Calculate the first-order energy correction $E_n^{(1)}$ for all states.

(b) Calculate the second-order correction $E_1^{(2)}$ for the ground state.

(c) Does this perturbation lift any degeneracies?

**Key Results:**
- $E_n^{(1)} = V_0/2$ (same for all states)
- $E_1^{(2)} = -\frac{V_0^2 a^2}{m} \sum_{k\neq 1} \frac{|\langle k | x | 1\rangle|^2}{E_k - E_1}$
- No degeneracy in 1D infinite well to lift

### 3. Sample Problems: Section B (Quantum Information)

#### Sample Problem B1: Entanglement and CHSH

**Problem Statement:**
Consider the state $|\psi\rangle = \cos\theta|00\rangle + \sin\theta|11\rangle$.

(a) For what values of $\theta$ is this state entangled?

(b) Calculate the reduced density matrix $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$.

(c) Compute the von Neumann entropy $S(\rho_A)$.

(d) Find the value of $\theta$ that maximizes the CHSH inequality violation.

**Solution Approach:**

(a) Entangled for all $\theta \neq 0, \pi/2$ (product states only at those points).

(b) Reduced density matrix:
$$\rho_A = \begin{pmatrix} \cos^2\theta & 0 \\ 0 & \sin^2\theta \end{pmatrix}$$

(c) Von Neumann entropy:
$$S(\rho_A) = -\cos^2\theta \log_2(\cos^2\theta) - \sin^2\theta \log_2(\sin^2\theta)$$

Maximum at $\theta = \pi/4$: $S = 1$ ebit.

(d) CHSH maximally violated at $\theta = \pi/4$ (Bell state), giving $2\sqrt{2}$.

#### Sample Problem B2: Quantum Channels

**Problem Statement:**
The amplitude damping channel has Kraus operators:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

(a) Verify that $\sum_i K_i^\dagger K_i = I$.

(b) Find the output state for input $\rho = |+\rangle\langle+|$ where $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$.

(c) Calculate the channel fidelity for the $|1\rangle$ state.

(d) Find the fixed point(s) of this channel.

### 4. Sample Problems: Section C (Error Correction)

#### Sample Problem C1: Stabilizer Analysis

**Problem Statement:**
Consider the stabilizer group generated by $g_1 = XZZXI$, $g_2 = IXZZX$, $g_3 = XIXZZ$, $g_4 = ZXIXZ$.

(a) Verify these generators commute.

(b) How many logical qubits does this code encode?

(c) Find the logical $\bar{X}$ and $\bar{Z}$ operators.

(d) What is the code distance?

**Solution Approach:**

(a) Check commutation: count overlapping positions where operators anticommute (must be even).

(b) $k = n - \text{rank}(S) = 5 - 4 = 1$ logical qubit.

(c) Logical operators: Find Paulis that commute with all stabilizers but are not in the stabilizer group.

(d) Distance $d$ = minimum weight of logical operators = 3 for this code.

This is the $[[5,1,3]]$ perfect code.

#### Sample Problem C2: Surface Code Threshold

**Problem Statement:**
A surface code of distance $d$ has logical error rate:
$$p_L \approx 0.03 \left(\frac{p}{p_{th}}\right)^{(d+1)/2}$$

(a) If $p = 10^{-3}$ and $p_{th} = 10^{-2}$, find $d$ such that $p_L < 10^{-12}$.

(b) How many physical qubits are needed for this code distance?

(c) If physical error rates improve to $p = 10^{-4}$, how does the required distance change?

### 5. Sample Problems: Section D (Algorithms)

#### Sample Problem D1: Phase Estimation

**Problem Statement:**
You wish to estimate the phase $\phi$ of unitary $U$ with eigenvalue $e^{2\pi i\phi}$.

(a) How many ancilla qubits are needed to estimate $\phi$ to $n$ bits of precision?

(b) Describe the quantum circuit for phase estimation.

(c) If $\phi = 0.375$, what measurement outcome do you expect with $n=3$ ancilla qubits?

(d) Analyze the algorithm's query complexity.

#### Sample Problem D2: Grover Optimality

**Problem Statement:**
Prove that Grover's algorithm is optimal for unstructured search.

(a) Define the problem formally.

(b) Show that any quantum algorithm requires $\Omega(\sqrt{N})$ queries.

(c) Explain the adversary argument.

### 6. Problem-Solving Strategies

#### Time Management

**The 20-10-10 Rule:**
For each problem (20 min allocation):
- **2 minutes**: Read and understand
- **15 minutes**: Solve and write
- **3 minutes**: Review and check

**Section Strategy:**
1. Skim all problems first (5 min)
2. Start with strongest section
3. Do easier problems first within each section
4. Leave 10 minutes for final review

#### Common Mistakes to Avoid

| Mistake | Prevention |
|---------|------------|
| Rushing through reading | Read problem twice before starting |
| Calculation errors | Show all steps, check units |
| Incomplete answers | Address all parts explicitly |
| Poor notation | Define symbols, use standard notation |
| Time mismanagement | Wear watch, set section checkpoints |

#### Formula Memorization Essentials

**Must memorize for exam:**

*Quantum Mechanics:*
- Pauli matrices
- Commutator $[\hat{x}, \hat{p}] = i\hbar$
- Harmonic oscillator energies
- Angular momentum algebra

*Quantum Information:*
- Von Neumann entropy
- Fidelity definitions
- Teleportation protocol
- Bell states

*Error Correction:*
- Stabilizer formalism
- Code parameters $[[n,k,d]]$
- Surface code structure
- Threshold theorem statement

*Algorithms:*
- QFT circuit
- Phase estimation
- Grover iteration
- Complexity classes

### 7. Oral Examination Preparation

The oral exam assesses communication and deep understanding.

#### Presentation Guidelines

**30-Minute Presentation Structure:**

| Time | Content |
|------|---------|
| 0-3 min | Introduction and motivation |
| 3-12 min | Background and context |
| 12-22 min | Your contribution/proposal |
| 22-27 min | Results and implications |
| 27-30 min | Conclusions and future work |

**Slide Design:**
- Maximum 15-20 slides
- One main idea per slide
- Equations when necessary, not excessive
- Clear figures and diagrams

#### Handling Questions

**Question Categories:**

1. **Clarification**: "Can you explain what you mean by...?"
   - Answer directly and concisely
   - Use examples if helpful

2. **Extension**: "What would happen if...?"
   - Think aloud, show reasoning
   - It's okay to say "That's an interesting question, let me think..."

3. **Challenge**: "But doesn't this contradict...?"
   - Stay calm
   - Acknowledge valid points
   - Defend your position with evidence

4. **Knowledge probe**: "What is the relationship between X and Y?"
   - Draw on Year 2 knowledge
   - Connect to your research area

**Response Framework:**
1. Pause briefly (shows thoughtfulness)
2. Clarify if needed
3. Answer the question asked
4. Provide supporting reasoning
5. Check if the answer was satisfactory

### 8. Practice Exam Schedule

**Recommended Year 3 Practice Schedule:**

| Month | Practice Activity | Frequency |
|-------|------------------|-----------|
| 37-39 | Section A practice (QM) | 2/week |
| 40-42 | Section B practice (QI) | 2/week |
| 43-45 | Section C+D practice | 2/week |
| 46 | Full mock written | 1/week |
| 47 | Mock oral exams | 2 sessions |
| 48 | Final mocks | As needed |

## Connections to Year 2 Knowledge

### Error Correction Section Preparation

Your Year 2 Semester 2A knowledge directly applies:
- Stabilizer codes (Weeks 101-104)
- Surface codes (Weeks 117-120)
- Fault tolerance (Weeks 121-128)

**Key topics to review:**
- Stabilizer generators and code parameters
- CSS code construction
- Threshold calculations
- Decoding algorithms

### Algorithm Section Preparation

Your Year 2 Semester 2B algorithm knowledge covers:
- VQE and QAOA (Month 35)
- Quantum machine learning (Month 35)
- Complexity considerations

**Connection to Section D problems:**
- Variational algorithm analysis
- Query complexity
- Algorithm comparison

## Practical Exercises

### Exercise 1: Timed Problem Set

Complete the following under exam conditions (20 minutes each):

**Problem 1 (QM):** A hydrogen atom is in the state $|\psi\rangle = \frac{1}{\sqrt{2}}(|2,0,0\rangle + |2,1,0\rangle)$. Calculate $\langle L^2 \rangle$ and $\langle L_z \rangle$.

**Problem 2 (QI):** Two qubits are in state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$. If the first qubit undergoes depolarizing noise with probability $p$, find the final two-qubit density matrix.

**Problem 3 (QEC):** For the 7-qubit Steane code, how many syndrome bits are needed? What types of errors can be corrected?

### Exercise 2: Presentation Outline

Create a 30-minute presentation outline on a topic from Year 2:

1. **Title**: _______________________
2. **Main thesis**: _______________________
3. **Key background** (3 points):
   - _______________________
   - _______________________
   - _______________________
4. **Core content** (3 slides):
   - _______________________
   - _______________________
   - _______________________
5. **Conclusion**: _______________________

### Exercise 3: Question Preparation

For each qualifying exam topic, prepare answers to:
- What is the most important result?
- What is the key equation?
- What is an open problem?
- How does it connect to other areas?

## Computational Lab: Practice Problem Generator

```python
"""
Day 1003 Computational Lab: Qualifying Exam Practice Tools
Generating and scoring practice problems
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import random

# =============================================================================
# Part 1: Problem Bank Structure
# =============================================================================

@dataclass
class QualProblem:
    """Structure for a qualifying exam problem."""
    section: str  # 'QM', 'QI', 'QEC', 'Algorithms'
    difficulty: int  # 1-3
    topic: str
    statement: str
    solution_outline: str
    points: int = 10
    time_minutes: int = 20

class ProblemBank:
    """Bank of qualifying exam practice problems."""

    def __init__(self):
        self.problems = {
            'QM': [],
            'QI': [],
            'QEC': [],
            'Algorithms': []
        }
        self._populate_bank()

    def _populate_bank(self):
        """Add sample problems to the bank."""

        # QM Problems
        self.problems['QM'].append(QualProblem(
            section='QM',
            difficulty=2,
            topic='Density matrices',
            statement="""A spin-1/2 system is in a thermal state at temperature T.
(a) Write the density matrix in terms of beta = 1/kT and energy splitting Delta E.
(b) Calculate the purity Tr(rho^2).
(c) Find the limit as T -> 0 and T -> infinity.""",
            solution_outline="""(a) rho = exp(-beta H)/Z with Z = 2 cosh(beta Delta E/2)
(b) Purity = (1 + tanh^2(beta Delta E/2))/2
(c) T->0: pure ground state; T->infinity: maximally mixed"""
        ))

        self.problems['QM'].append(QualProblem(
            section='QM',
            difficulty=3,
            topic='Perturbation theory',
            statement="""A 2D harmonic oscillator with omega_x = omega_y = omega is
perturbed by V = lambda * x * y.
(a) Find the first-order energy shift for the first excited states.
(b) What is the new degeneracy structure?
(c) Find the correct zeroth-order states.""",
            solution_outline="""(a) Degenerate perturbation theory needed for |1,0> and |0,1>
(b) Degeneracy is lifted: E+ and E- split by ~ lambda
(c) New states are (|1,0> +/- |0,1>)/sqrt(2)"""
        ))

        # QI Problems
        self.problems['QI'].append(QualProblem(
            section='QI',
            difficulty=2,
            topic='Quantum channels',
            statement="""The dephasing channel has Kraus operators K_0 = sqrt(1-p)I,
K_1 = sqrt(p)Z.
(a) Show this is a valid quantum channel.
(b) Apply it to state |+><+| and find the output.
(c) What is the channel capacity for this noise?""",
            solution_outline="""(a) K_0^dag K_0 + K_1^dag K_1 = I (verify)
(b) Output: (1-p)|+><+| + p|-><-| = I/2 for p=1/2
(c) Capacity = 1 - H_2(p) bits"""
        ))

        self.problems['QI'].append(QualProblem(
            section='QI',
            difficulty=2,
            topic='Entanglement',
            statement="""Consider the Werner state rho_W = (1-p)|Phi+><Phi+| + p*I/4.
(a) For what values of p is this state separable?
(b) Calculate the concurrence as a function of p.
(c) Find the entanglement of formation.""",
            solution_outline="""(a) Separable for p >= 2/3 (Peres-Horodecki criterion)
(b) Concurrence C = max(0, 1 - 3p/2)
(c) E_F = H_2((1 + sqrt(1-C^2))/2)"""
        ))

        # QEC Problems
        self.problems['QEC'].append(QualProblem(
            section='QEC',
            difficulty=2,
            topic='Stabilizer codes',
            statement="""For a [[7,1,3]] Steane code with generators:
g1 = IIIXXXX, g2 = IXXIIXX, g3 = XIXIXIX,
g4 = IIIZZZZ, g5 = IZZIIZZ, g6 = ZIZIZIZ
(a) Verify the generators commute.
(b) What error can be detected by syndrome 101 for X errors?
(c) Find logical X-bar and Z-bar operators.""",
            solution_outline="""(a) Count anticommuting positions (must be even)
(b) Syndrome 101 -> error on qubit 5 (binary decoding)
(c) X-bar = XXXXXXX, Z-bar = ZZZZZZZ"""
        ))

        self.problems['QEC'].append(QualProblem(
            section='QEC',
            difficulty=3,
            topic='Fault tolerance',
            statement="""A fault-tolerant protocol must satisfy:
(a) Define what "fault-tolerant" means for a syndrome measurement.
(b) How does flag-qubit syndrome measurement work?
(c) For a distance-3 code, how many faults can occur while
maintaining correctability?""",
            solution_outline="""(a) A single fault in syndrome circuit cannot cause
uncorrectable error on data.
(b) Flag qubit detects if a single fault spreads to
multiple data qubits.
(c) One fault can be tolerated (t = floor((d-1)/2) = 1)"""
        ))

        # Algorithm Problems
        self.problems['Algorithms'].append(QualProblem(
            section='Algorithms',
            difficulty=2,
            topic='Grover',
            statement="""For Grover's algorithm searching N=2^n items with k solutions:
(a) What is the optimal number of iterations?
(b) What is the success probability after optimal iterations?
(c) How does the algorithm change if k is unknown?""",
            solution_outline="""(a) Optimal iterations ~ (pi/4)sqrt(N/k)
(b) Success probability ~ 1 - O(k/N)
(c) Use quantum counting or exponential search"""
        ))

        self.problems['Algorithms'].append(QualProblem(
            section='Algorithms',
            difficulty=3,
            topic='VQE',
            statement="""For the Variational Quantum Eigensolver:
(a) Write the cost function being minimized.
(b) Explain the parameter-shift rule for gradient estimation.
(c) What is barren plateau and when does it occur?""",
            solution_outline="""(a) C(theta) = <psi(theta)|H|psi(theta)>
(b) Gradient: (C(theta+pi/2) - C(theta-pi/2))/2
(c) Barren plateau: exponentially vanishing gradients,
occurs with deep random circuits or global observables"""
        ))

    def get_random_problem(self, section: str = None,
                           difficulty: int = None) -> QualProblem:
        """Get a random problem, optionally filtered."""
        if section:
            pool = self.problems[section]
        else:
            pool = [p for probs in self.problems.values() for p in probs]

        if difficulty:
            pool = [p for p in pool if p.difficulty == difficulty]

        return random.choice(pool) if pool else None

    def generate_mock_exam(self) -> List[QualProblem]:
        """Generate a full 12-problem mock exam."""
        exam = []
        for section in ['QM', 'QI', 'QEC', 'Algorithms']:
            # 3 problems per section
            section_problems = self.problems[section].copy()
            random.shuffle(section_problems)
            exam.extend(section_problems[:3] if len(section_problems) >= 3
                       else section_problems)
        return exam

# =============================================================================
# Part 2: Exam Timer and Scorer
# =============================================================================

class ExamSession:
    """Manage a timed practice exam session."""

    def __init__(self, problems: List[QualProblem]):
        self.problems = problems
        self.scores = [0] * len(problems)
        self.time_spent = [0] * len(problems)
        self.total_time = 240  # 4 hours in minutes

    def score_problem(self, index: int, score: int, time_min: int):
        """Record score and time for a problem."""
        self.scores[index] = min(score, 10)
        self.time_spent[index] = time_min

    def get_section_score(self, section: str) -> Tuple[int, int]:
        """Get score and max for a section."""
        section_indices = [i for i, p in enumerate(self.problems)
                          if p.section == section]
        earned = sum(self.scores[i] for i in section_indices)
        possible = len(section_indices) * 10
        return earned, possible

    def is_passing(self) -> Tuple[bool, dict]:
        """Check if exam is passing."""
        sections = ['QM', 'QI', 'QEC', 'Algorithms']
        section_results = {}
        all_passing = True

        for section in sections:
            earned, possible = self.get_section_score(section)
            percentage = earned / possible * 100 if possible > 0 else 0
            passing = percentage >= 50
            section_results[section] = {
                'earned': earned,
                'possible': possible,
                'percentage': percentage,
                'passing': passing
            }
            if not passing:
                all_passing = False

        total = sum(self.scores)
        overall_percentage = total / (len(self.problems) * 10) * 100

        return (all_passing and overall_percentage >= 70), {
            'sections': section_results,
            'total': total,
            'overall_percentage': overall_percentage
        }

    def visualize_results(self):
        """Create visualization of exam results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Scores by section
        ax1 = axes[0]
        sections = ['QM', 'QI', 'QEC', 'Algorithms']
        section_scores = []
        for section in sections:
            earned, possible = self.get_section_score(section)
            section_scores.append(earned / possible * 100)

        colors = ['green' if s >= 50 else 'red' for s in section_scores]
        ax1.bar(sections, section_scores, color=colors, alpha=0.7)
        ax1.axhline(50, color='red', linestyle='--', label='Minimum (50%)')
        ax1.axhline(70, color='orange', linestyle='--', label='Target (70%)')
        ax1.set_ylabel('Score (%)')
        ax1.set_title('Section Scores')
        ax1.legend()
        ax1.set_ylim(0, 100)

        # Time distribution
        ax2 = axes[1]
        section_times = []
        for section in sections:
            section_indices = [i for i, p in enumerate(self.problems)
                              if p.section == section]
            section_times.append(sum(self.time_spent[i] for i in section_indices))

        ax2.bar(sections, section_times, color='steelblue', alpha=0.7)
        ax2.axhline(60, color='green', linestyle='--', label='Target (60 min)')
        ax2.set_ylabel('Time (minutes)')
        ax2.set_title('Time Spent by Section')
        ax2.legend()

        # Individual problem scores
        ax3 = axes[2]
        problem_labels = [f"{p.section[0]}{i+1}" for i, p in enumerate(self.problems)]
        colors = ['green' if s >= 7 else 'orange' if s >= 5 else 'red'
                 for s in self.scores]
        ax3.bar(problem_labels, self.scores, color=colors, alpha=0.7)
        ax3.axhline(7, color='green', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Problem')
        ax3.set_ylabel('Score (out of 10)')
        ax3.set_title('Individual Problem Scores')

        plt.tight_layout()
        plt.savefig('mock_exam_results.png', dpi=150, bbox_inches='tight')
        plt.show()

        return fig

# =============================================================================
# Part 3: Study Progress Tracker
# =============================================================================

class StudyTracker:
    """Track practice problem performance over time."""

    def __init__(self):
        self.history = {
            'QM': [],
            'QI': [],
            'QEC': [],
            'Algorithms': []
        }

    def log_practice(self, section: str, score: float, date: str = None):
        """Log a practice session score."""
        from datetime import datetime
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        self.history[section].append({
            'date': date,
            'score': score
        })

    def get_trend(self, section: str) -> Tuple[List, List]:
        """Get score trend for a section."""
        if not self.history[section]:
            return [], []
        dates = [entry['date'] for entry in self.history[section]]
        scores = [entry['score'] for entry in self.history[section]]
        return dates, scores

    def visualize_progress(self):
        """Plot progress over time."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        sections = ['QM', 'QI', 'QEC', 'Algorithms']
        colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']

        for ax, section, color in zip(axes.flat, sections, colors):
            dates, scores = self.get_trend(section)
            if dates:
                x = range(len(dates))
                ax.plot(x, scores, 'o-', color=color, linewidth=2, markersize=8)
                ax.axhline(70, color='green', linestyle='--', alpha=0.5,
                          label='Passing (70%)')
                ax.axhline(50, color='red', linestyle='--', alpha=0.5,
                          label='Minimum (50%)')

                # Trend line
                if len(scores) > 2:
                    z = np.polyfit(x, scores, 1)
                    p = np.poly1d(z)
                    ax.plot(x, p(x), '--', color=color, alpha=0.5)
            else:
                ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)

            ax.set_title(f'{section} Progress', fontsize=12)
            ax.set_ylabel('Score (%)')
            ax.set_ylim(0, 100)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Qualifying Exam Preparation Progress', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('study_progress.png', dpi=150, bbox_inches='tight')
        plt.show()

        return fig

# =============================================================================
# Part 4: Demo
# =============================================================================

print("=" * 70)
print("Qualifying Exam Practice Tools")
print("=" * 70)

# Create problem bank
bank = ProblemBank()

print("\nProblem Bank Contents:")
for section, problems in bank.problems.items():
    print(f"  {section}: {len(problems)} problems")

# Get a sample problem
print("\n" + "=" * 70)
print("Sample Practice Problem:")
print("=" * 70)

sample = bank.get_random_problem(section='QEC')
if sample:
    print(f"\nSection: {sample.section}")
    print(f"Topic: {sample.topic}")
    print(f"Difficulty: {'*' * sample.difficulty}")
    print(f"\nProblem:\n{sample.statement}")
    print(f"\nSolution Outline:\n{sample.solution_outline}")

# Generate mock exam
print("\n" + "=" * 70)
print("Mock Exam Generation")
print("=" * 70)

mock_exam = bank.generate_mock_exam()
print(f"\nGenerated {len(mock_exam)}-problem mock exam:")
for i, prob in enumerate(mock_exam):
    print(f"  {i+1}. [{prob.section}] {prob.topic} (Difficulty: {prob.difficulty})")

# Demo exam session
print("\n" + "=" * 70)
print("Exam Session Demo")
print("=" * 70)

session = ExamSession(mock_exam)

# Simulate some scores
sample_scores = [8, 7, 6, 9, 7, 8, 7, 8, 9, 6, 7, 8]
sample_times = [18, 22, 25, 15, 20, 22, 20, 18, 15, 25, 20, 18]

for i, (score, time) in enumerate(zip(sample_scores, sample_times)):
    session.score_problem(i, score, time)

passing, results = session.is_passing()

print(f"\nExam Results:")
print(f"  Overall: {results['overall_percentage']:.1f}%")
print(f"  Passing: {'YES' if passing else 'NO'}")
print("\n  Section breakdown:")
for section, data in results['sections'].items():
    status = 'PASS' if data['passing'] else 'FAIL'
    print(f"    {section}: {data['percentage']:.1f}% [{status}]")

# Study tracker demo
print("\n" + "=" * 70)
print("Study Progress Tracking")
print("=" * 70)

tracker = StudyTracker()

# Simulate practice history
import random
for section in ['QM', 'QI', 'QEC', 'Algorithms']:
    for i in range(8):
        # Simulating improvement over time
        base_score = 50 + i * 3 + random.uniform(-5, 5)
        date = f"2026-02-{15+i:02d}"
        tracker.log_practice(section, min(base_score, 95), date)

print("\nSample practice history logged.")
print("Uncomment visualization calls to see plots.")

# Uncomment to generate visualizations:
# session.visualize_results()
# tracker.visualize_progress()

print("\n" + "=" * 70)
print("Practice tools ready for qualifying exam preparation!")
print("=" * 70)
```

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Written Exam | 4 hours, 12 problems, 4 sections, 70% passing |
| Section Requirement | Each section must be 50%+ |
| Oral Exam | 2 hours: presentation (30 min) + Q&A (90 min) |
| Time per Problem | ~20 minutes target |
| Key to Success | Practice + time management + breadth |

### Problem-Solving Framework

```
Read Problem (2 min)
    ↓
Plan Approach (2 min)
    ↓
Execute Solution (13 min)
    ↓
Check and Format (3 min)
```

### Main Takeaways

1. **The written exam tests breadth** across all four major areas equally

2. **Time management is critical** with only 20 minutes per problem

3. **Section minimums matter** - strong performance in one area cannot compensate for failure in another

4. **The oral exam tests depth** and communication ability

5. **Regular practice is essential** - use the practice tools consistently throughout Year 3

## Daily Checklist

- [ ] I understand the written exam format and scoring
- [ ] I have reviewed sample problems from each section
- [ ] I have practiced at least one problem under timed conditions
- [ ] I understand the oral exam structure
- [ ] I have drafted my practice exam schedule
- [ ] I know common mistakes and how to avoid them
- [ ] I have identified my strongest and weakest sections
- [ ] I have run the practice problem tools

## Preview: Day 1004

Tomorrow we focus on **Research Proposal Development**. We will learn:
- The structure and components of a research proposal
- How to formulate research questions
- Methods section writing
- Timeline and milestone planning
- Examples from quantum computing research

Your qualifying exam includes defending a research proposal - tomorrow prepares you for that crucial component.

---

*"The qualifying exam is not meant to trick you. It tests whether you can think like a researcher across the breadth of quantum science."*

---

| Navigation | Link |
|------------|------|
| Previous Day | [Day 1002: Year 3 Overview](./Day_1002_Monday.md) |
| Next Day | [Day 1004: Research Proposal](./Day_1004_Wednesday.md) |
| Week Overview | [Week 144 README](./README.md) |
