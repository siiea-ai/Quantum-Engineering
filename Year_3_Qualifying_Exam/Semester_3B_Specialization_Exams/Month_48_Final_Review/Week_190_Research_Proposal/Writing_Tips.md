# Scientific Writing Tips for Research Proposals

## Overview

Good scientific writing is clear, precise, and compelling. This guide covers essential writing skills for research proposals in quantum science and engineering.

---

## Part 1: Principles of Scientific Writing

### The Cardinal Rules

1. **Clarity over elegance** - Being understood is more important than sounding smart
2. **Precision over vagueness** - Specific claims are more credible
3. **Active over passive** - Active voice is usually clearer and more engaging
4. **Concise over wordy** - Respect your reader's time
5. **Evidence over assertion** - Back up claims with citations or data

### The Reader's Perspective

Always consider:
- What does my reader already know?
- What do they need to know to understand this?
- What is the minimum necessary background?
- What questions will they have?

---

## Part 2: Sentence-Level Writing

### Active vs. Passive Voice

**Passive (often unclear):**
*"The algorithm was implemented and the results were analyzed."*

**Active (clearer):**
*"We implemented the algorithm and analyzed the results."*

**When passive is acceptable:**
- When the actor is unknown or unimportant
- When the action is more important than the actor
- For variety in a paragraph of active sentences

### Strong Verbs

**Weak:**
*"There is a need for better decoders."*

**Strong:**
*"Practical implementation requires better decoders."*

**Weak:**
*"It was found that the error rate was reduced."*

**Strong:**
*"The protocol reduced the error rate by 40%."*

### Eliminating Wordiness

| Wordy | Concise |
|-------|---------|
| in order to | to |
| due to the fact that | because |
| a large number of | many |
| at the present time | now |
| in the event that | if |
| it is important to note that | Note: (or just state it) |
| the majority of | most |
| in spite of the fact that | although |
| has the ability to | can |

### Sentence Length and Structure

- Vary sentence length (average 15-25 words)
- Use short sentences for important points
- Complex ideas may need longer sentences
- Break up long sentences when possible

**Too long:**
*"The surface code, which was first proposed by Kitaev in 1997 based on ideas from topological quantum field theory and which has since become the leading candidate for near-term fault-tolerant quantum computing due to its high threshold and local stabilizer measurements, requires O(d^2) physical qubits for distance d encoding."*

**Better:**
*"The surface code, first proposed by Kitaev in 1997, has become the leading candidate for near-term fault-tolerant quantum computing. Its advantages include high threshold (~1%) and local stabilizer measurements. However, it requires O(d^2) physical qubits for distance d encoding."*

---

## Part 3: Paragraph-Level Organization

### Paragraph Structure

1. **Topic sentence** - Main point of the paragraph
2. **Supporting sentences** - Evidence, examples, explanation
3. **Concluding/transition sentence** - Wrap up and connect to next paragraph

### Example Paragraph Analysis

**Topic sentence:**
*"Quantum LDPC codes offer significant advantages over the surface code."*

**Supporting sentences:**
*"First, good qLDPC codes achieve constant rate, meaning the ratio k/n remains bounded away from zero. Second, they can have linear distance d = O(n), providing strong protection. Third, some families exhibit single-shot error correction, reducing syndrome measurement overhead."*

**Transition:**
*"Despite these advantages, significant challenges remain in developing practical decoders."*

### Paragraph Flow

Use transition words to show relationships:

**Addition:** furthermore, moreover, additionally, also
**Contrast:** however, nevertheless, in contrast, while
**Cause/effect:** therefore, consequently, as a result, thus
**Example:** for example, specifically, in particular, such as
**Sequence:** first, second, finally, subsequently
**Summary:** in summary, overall, in conclusion

---

## Part 4: Document-Level Structure

### Logical Flow

Each section should:
1. Connect to the previous section
2. Accomplish a specific purpose
3. Set up the next section

### The "Funnel" Structure

**Introduction:**
- Start broad (field importance)
- Narrow to specific problem
- End with your approach

**Body:**
- Build from background to proposed work
- Each section adds necessary context
- Lead reader step-by-step

**Conclusion:**
- Summarize key points
- Restate significance
- End with broader implications

### Section Transitions

**Weak:**
*"[End of background section] ...this concludes the background."*
*"[Start of methods section] In this section, we describe our methods."*

**Strong:**
*"[End of background section] ...this gap in current decoding approaches motivates the present research."*
*"[Start of methods section] To address this challenge, we will develop neural network-enhanced decoders using the following approach."*

---

## Part 5: Technical Writing Specifics

### Equations

**Introducing equations:**
*"The logical error rate scales as:"*
$$p_L \sim \left(\frac{p}{p_{th}}\right)^{(d+1)/2}$$
*"where p is the physical error rate, p_th is the threshold, and d is the code distance."*

**Key points:**
- Number important equations
- Define all variables
- Explain significance
- Don't let equations stand alone

### Figures and Tables

**Figure captions should be self-contained:**
*"Figure 1: Logical error rate vs. physical error rate for the surface code at distances d = 3, 5, 7, 9. Error bars represent 95% confidence intervals from 10,000 Monte Carlo samples. The threshold occurs where curves cross at approximately p_th = 0.8%."*

**Tables for comparison:**
- Include units in headers
- Align numbers by decimal point
- Bold best results
- Note data sources

### Technical Terms

**First use:** Define the term
*"Quantum low-density parity-check (qLDPC) codes are quantum error correcting codes characterized by sparse stabilizer generators."*

**Subsequent uses:** Use the abbreviation
*"The main advantage of qLDPC codes is their potential for constant overhead."*

### Numbers and Units

- Spell out numbers below 10 in text
- Use numerals for measurements
- Include units: "10 qubits", "100 $\mu$s"
- Be consistent with significant figures

---

## Part 6: Common Mistakes to Avoid

### Grammar and Style

| Mistake | Problem | Fix |
|---------|---------|-----|
| "This" without referent | Unclear what "this" refers to | "This result" or "This approach" |
| "Very" overuse | Weak intensifier | Remove or use precise term |
| "Interesting" | Vague | Explain why it's interesting |
| Starting with "It is..." | Weak construction | Rewrite with clear subject |
| Nominalizations | Verbs turned to nouns | Use verbs |

### Technical Issues

| Mistake | Example | Fix |
|---------|---------|-----|
| Undefined acronyms | "Using QEC" (first use) | "quantum error correction (QEC)" |
| Missing citations | "Recent work has shown..." | "Recent work (Author, 2024) has shown..." |
| Vague claims | "significantly better" | "40% higher fidelity" |
| Overstating | "proves that" | "provides evidence that" or "demonstrates that" |
| Wrong tense | Mixing past and present | Consistent tense within sections |

### Structural Issues

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Burying the lead | Main point in middle/end | State main point first |
| Missing transitions | Abrupt topic changes | Add transition sentences |
| Repetition | Same point made twice | Consolidate or eliminate |
| Tangents | Paragraph off-topic | Delete or move |
| Imbalance | Sections too long/short | Reorganize content |

---

## Part 7: The Writing Process

### Stage 1: Pre-Writing

1. **Outline** - Create detailed structure before writing
2. **Collect materials** - Gather references, data, equations
3. **Know your audience** - Adjust level accordingly
4. **Set goals** - What must reader understand?

### Stage 2: Drafting

1. **Write quickly** - Don't edit while drafting
2. **Start anywhere** - Don't force linear writing
3. **Leave placeholders** - [cite], [add equation], [expand]
4. **Focus on content** - Style comes in revision

### Stage 3: Revision

#### First pass: Structure
- Does the organization make sense?
- Is the flow logical?
- Are sections balanced?

#### Second pass: Content
- Are claims supported?
- Is everything necessary?
- Is anything missing?

#### Third pass: Clarity
- Are sentences clear?
- Are paragraphs focused?
- Are transitions smooth?

#### Fourth pass: Polish
- Grammar and spelling
- Consistent formatting
- Citation accuracy

### Stage 4: Feedback

1. **Self-review** - Read aloud, catch awkward phrasing
2. **Peer review** - Fresh eyes catch blind spots
3. **Expert review** - Technical accuracy check
4. **Rest and return** - Time away improves perspective

---

## Part 8: Specific Sections Guidance

### Writing the Abstract

**Template:**
```
[Context - 1 sentence on field importance]
[Problem - 1-2 sentences on specific challenge]
[Approach - 1-2 sentences on what you'll do]
[Methods - 1 sentence on how]
[Expected outcomes - 1-2 sentences]
[Impact - 1 sentence on significance]
```

**Tips:**
- Write last (after you know what you're summarizing)
- No citations in abstract
- No undefined acronyms
- Standalone and self-contained

### Writing the Introduction

**Paragraph 1:** Hook - why should reader care?
- Start with big picture importance
- Make it compelling

**Paragraph 2:** Context - what's the current state?
- Brief overview of the field
- Key accomplishments

**Paragraph 3:** Problem - what's missing?
- Specific challenge you address
- Why it matters

**Paragraph 4:** Approach - what will you do?
- Your solution in brief
- Preview of proposal structure

### Writing the Literature Review

**Organize thematically, not chronologically**

**For each theme:**
1. Introduce the theme
2. Summarize key works (grouped, not listed)
3. Compare and contrast approaches
4. Identify limitations and gaps

**Transition to your work:**
*"These approaches, while valuable, do not address [specific gap]. This gap motivates the present research."*

### Writing Methods

**Be specific enough to be reproduced:**
- "We will use belief propagation" (too vague)
- "We will implement belief propagation using log-likelihood ratio messages, with damping factor 0.5 and maximum 50 iterations" (specific)

**Justify choices:**
*"We use PyTorch for neural network implementation due to its flexibility and GPU support."*

---

## Part 9: LaTeX Tips

### Equation Formatting

```latex
% Inline math
The error rate is $p = 0.01$.

% Display math (numbered)
\begin{equation}
p_L \sim \left(\frac{p}{p_{th}}\right)^{(d+1)/2}
\label{eq:threshold}
\end{equation}

% Display math (unnumbered)
\[
H = -\sum_{\langle i,j \rangle} J_{ij} Z_i Z_j
\]

% Aligned equations
\begin{align}
S(\rho) &= -\text{Tr}(\rho \log \rho) \\
&= -\sum_i \lambda_i \log \lambda_i
\end{align}
```

### Common Symbols

```latex
% Quantum mechanics
|\psi\rangle, \langle\phi|, \langle\psi|\phi\rangle
\hat{H}, \hat{a}^\dagger, \hat{\sigma}_z
\otimes, \oplus

% Error correction
[[n,k,d]], O(\sqrt{n}), \Omega(n)
```

### References with BibTeX

```latex
% In text
Recent work~\cite{Author2024} demonstrated...
Several approaches exist~\cite{A2023,B2024,C2024}.

% Bibliography style
\bibliographystyle{apsrev4-2}  % For APS journals
\bibliography{references}
```

---

## Part 10: Editing Checklist

### Before Submission

#### Content
- [ ] All claims supported by evidence or citation
- [ ] All necessary background included
- [ ] No unnecessary tangents
- [ ] Research questions clearly stated
- [ ] Methods sufficiently detailed

#### Structure
- [ ] Logical flow throughout
- [ ] Smooth transitions between sections
- [ ] Balanced section lengths
- [ ] Clear paragraph structure

#### Style
- [ ] Active voice predominates
- [ ] Sentences clear and concise
- [ ] Technical terms defined
- [ ] Consistent terminology

#### Technical
- [ ] All equations explained
- [ ] All figures captioned
- [ ] All acronyms defined
- [ ] All citations accurate
- [ ] Consistent number formatting

#### Polish
- [ ] Spell-checked
- [ ] Grammar-checked
- [ ] Read aloud for flow
- [ ] Formatting consistent
- [ ] Page limits respected

---

## Quick Reference: Power Words

### For Introducing Research

| Instead of | Use |
|------------|-----|
| look at | investigate, examine, analyze |
| show | demonstrate, reveal, establish |
| use | employ, utilize, apply |
| make | develop, construct, design |
| find | discover, identify, determine |

### For Describing Results

| Instead of | Use |
|------------|-----|
| good | effective, efficient, robust |
| bad | limited, suboptimal, insufficient |
| big | significant, substantial, considerable |
| new | novel, original, innovative |
| important | critical, essential, fundamental |

### For Hedging Appropriately

| Too Strong | Appropriately Hedged |
|------------|---------------------|
| proves | provides evidence for |
| always | typically, generally |
| never | rarely, seldom |
| will | may, could, is expected to |
| shows | suggests, indicates |

---

**Remember:** Good writing is rewriting. Budget significant time for revision, and don't be afraid to delete text that isn't working. Clear communication is your most important skill as a researcher.
