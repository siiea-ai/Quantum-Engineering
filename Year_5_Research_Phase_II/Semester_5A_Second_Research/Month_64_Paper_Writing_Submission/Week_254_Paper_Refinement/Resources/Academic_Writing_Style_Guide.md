# Academic Writing Style Guide for Quantum Physics

## Principles, Patterns, and Best Practices

---

## Core Principles

### 1. Clarity Over Cleverness

Scientific writing exists to communicate ideas, not to impress. Every stylistic choice should serve clarity.

**The Fog Test**: If a qualified reader must re-read a sentence to understand it, revise.

### 2. Precision Over Generality

Vague language wastes reader attention and invites misinterpretation.

| Vague | Precise |
|-------|---------|
| "significantly improved" | "improved by a factor of 3.2" |
| "high fidelity" | "fidelity of 99.4%" |
| "fast gate" | "25 ns gate" |
| "recently" | "in 2024" |
| "various parameters" | "temperature, magnetic field, and RF power" |

### 3. Conciseness Over Verbosity

Remove words that add no meaning.

**Word Count Principle**: If you can say it in fewer words without losing meaning, do so.

---

## Sentence-Level Style

### Subject-Verb Proximity

Keep subjects close to their verbs. Long separations confuse readers.

**Poor**:
> The fidelity, which we measured using randomized benchmarking with 50 random Clifford sequences applied to both the computational and non-computational subspaces over 1000 experimental repetitions, was 99.2%.

**Better**:
> We measured the fidelity using randomized benchmarking with 50 random Clifford sequences over 1000 repetitions. The measured fidelity was 99.2%.

### Sentence Length Variation

Mix sentence lengths for rhythm. Technical prose often becomes monotonous with uniform long sentences.

**Pattern to follow**: Short sentence. Medium sentence with one or two clauses. Longer sentence that develops a complex idea with multiple connected elements. Short conclusion.

### Active vs. Passive Voice

Both are acceptable in physics writing. Use each purposefully.

**Active voice** emphasizes the actor:
> "We measured the coherence time." (Emphasizes researchers' action)

**Passive voice** emphasizes the object or result:
> "The sample was cooled to 20 mK." (Emphasizes what happened to sample)
> "A threshold of 1% was observed." (Emphasizes the finding)

**Default to active** unless:
- The actor is unknown or unimportant
- You want to emphasize the object
- Convention in your subfield prefers passive

### Parallel Structure

Keep parallel grammatical structures for parallel ideas.

**Poor**:
> We characterized the qubit coherence, measured gate fidelities, and the crosstalk was quantified.

**Better**:
> We characterized qubit coherence, measured gate fidelities, and quantified crosstalk.

---

## Paragraph-Level Style

### Topic Sentences

Begin each paragraph with its main point. Readers should grasp the paragraph's purpose from the first sentence.

**Topic Sentence Types**:

1. **Statement**: States the paragraph's claim
   > "The error rate decreases exponentially with code distance."

2. **Transition**: Connects to previous paragraph
   > "Beyond the threshold, a different scaling regime emerges."

3. **Question**: Sets up what the paragraph answers
   > "How does the error rate depend on measurement frequency?"

### Paragraph Unity

Each paragraph should develop one idea. If you find yourself making a new point, start a new paragraph.

**Signs of paragraph disunity**:
- Multiple topic sentences
- Unrelated points
- Length over 10-12 sentences
- "Also" appearing frequently

### Transitions Between Paragraphs

Connect paragraphs to maintain narrative flow.

**Transition Strategies**:

1. **Repeat a key term**:
   > Paragraph 1 ends: "...reaching a fidelity of 99%."
   > Paragraph 2 begins: "This fidelity exceeds the fault-tolerant threshold..."

2. **Use transition words**:
   - Continuation: furthermore, moreover, additionally
   - Contrast: however, nevertheless, in contrast
   - Cause/effect: therefore, consequently, as a result
   - Sequence: first, next, finally

3. **Echo the structure**:
   > Paragraph 1: "The first evidence for X comes from..."
   > Paragraph 2: "Additional support comes from..."

---

## Word Choice

### Words to Avoid or Limit

| Word/Phrase | Problem | Alternative |
|-------------|---------|-------------|
| very | Weak intensifier | Use specific adjective or quantify |
| really | Filler | Delete |
| basically | Suggests oversimplification | Delete or be more precise |
| utilize | Pretentious | use |
| methodology | Often unnecessary | method |
| in order to | Wordy | to |
| the fact that | Wordy | that |
| a number of | Vague | several, many, or specific number |
| in the case of | Wordy | for, in |
| due to the fact that | Very wordy | because |
| it is interesting to note that | Filler | [delete and just state the interesting thing] |
| it can be seen that | Passive filler | [delete and just show] |
| represents | Often overused | is, shows, indicates |

### Words That Signal Weak Arguments

**Flag these for revision**:
- "clearly" (if it's clear, you don't need to say so)
- "obviously" (same)
- "undoubtedly" (same)
- "proves" (rarely justified in physics)
- "perfect" (rarely true)
- "novel" (let readers judge)
- "first ever" (requires thorough literature check)

### Technical Term Consistency

Define terms on first use and use them consistently thereafter.

**Poor**:
> "We measured the coherence time, also known as T2. The dephasing time was 30 μs. After optimizing, the T2* improved to 45 μs."

**Better**:
> "We measured the coherence time T2 = 30 μs. After optimization, T2 improved to 45 μs."

(Note: T2 and T2* are actually different quantities; this example shows why consistency and precision matter.)

---

## Section-Specific Style

### Abstract Style

- Present tense for general statements and significance
- Past tense for methods and specific results
- No citations
- No undefined abbreviations
- One paragraph
- Standalone (understandable without reading paper)

**Abstract Template**:
> [Context - present tense]. [Problem - present tense]. Here, we [approach - present tense]. We [specific result - past tense]. [Additional result - past tense]. [Significance - present tense].

### Introduction Style

- Present tense for general truths and ongoing importance
- Past tense for prior work ("Smith et al. showed...")
- Third person for literature ("Previous work demonstrated...")
- First person for contributions ("We demonstrate...", "Here we show...")

**Opening Strategies**:

1. **The Grand Challenge**: Start with big-picture importance
   > "Fault-tolerant quantum computing promises computational capabilities beyond classical reach."

2. **The Specific Problem**: Start with the concrete challenge
   > "Reducing logical error rates below physical error rates remains a key milestone for quantum error correction."

3. **The Recent Development**: Start with what's new
   > "Recent demonstrations of quantum error correction have achieved repetition codes with up to 11 qubits."

### Methods Style

- Past tense for what you did
- Present tense for general descriptions of techniques
- Be specific and complete
- Avoid first person when possible (passive acceptable here)

**Example**:
> "Qubits were fabricated using standard lithographic techniques. The coupling between qubits is capacitive, with coupling strength g/2π = 5 MHz."

### Results Style

- Past tense for observations ("We measured...", "The data showed...")
- Present tense for what figures show ("Figure 2 shows...")
- Lead with findings, not methodology
- Quantify everything

**Results Sentence Patterns**:

1. **Finding-first**: "The fidelity reached 99.4 ± 0.1% (Fig. 2a)."
2. **Figure reference**: "As shown in Fig. 2, the error rate decreases exponentially."
3. **Comparison**: "Our measured threshold of 0.95% agrees with theoretical predictions of 1%."

### Discussion Style

- Present tense for implications and interpretations
- Past tense when referring back to results
- Balance confidence with appropriate hedging
- Connect to broader context

**Hedging Language** (use appropriately, not excessively):
- "suggests" (rather than "proves")
- "consistent with" (rather than "confirms")
- "may indicate" (for speculation)
- "likely" (with support)

---

## Mathematical Writing

### Introducing Equations

Always introduce equations before presenting them.

**Poor**:
> $$H = \sum_i \omega_i \sigma_i^z + \sum_{ij} J_{ij} \sigma_i^x \sigma_j^x$$
> This is the Hamiltonian.

**Better**:
> The system is described by the Ising Hamiltonian:
> $$H = \sum_i \omega_i \sigma_i^z + \sum_{ij} J_{ij} \sigma_i^x \sigma_j^x$$
> where $\omega_i$ are the qubit frequencies and $J_{ij}$ are the coupling strengths.

### Equation Grammar

Equations are part of sentences. They should flow grammatically.

**Equations as clauses**:
> The fidelity is given by
> $$\mathcal{F} = |\langle \psi_{target} | \psi_{final} \rangle|^2,$$
> which reduces to unity for perfect gates.

Note the comma after the equation and lowercase continuation.

### Defining Variables

Define variables immediately after first use.

**Pattern**: "...where [variable] is [definition], [variable2] is [definition2], and [variable3] is [definition3]."

**For many variables**: Use a definition list or table.

### Mathematical Conventions

- Use italic for variables: $x$, $H$, $\psi$
- Use roman for functions: sin, cos, Tr, det
- Use bold for vectors: **r**, or arrow notation: $\vec{r}$
- Distinguish vectors from operators: $\hat{H}$ vs $H$
- Number equations you reference: Eq. (1)

---

## Citations and References

### When to Cite

- Specific claims from other work
- Methods developed elsewhere
- Prior results you build on
- Foundational concepts
- Controversial statements

### When NOT to Cite

- Common knowledge in the field
- Your own statements
- General textbook material

### Citation Placement

Place citations at the end of the relevant clause or sentence.

**Poor**:
> [1] showed that quantum error correction can suppress errors.

**Better**:
> Quantum error correction can suppress errors [1].
> OR
> Previous work has shown that error correction suppresses errors [1].

### Citation Style

**Narrative citations** (author as subject):
> "Smith et al. [1] demonstrated fault-tolerant gates."

**Parenthetical citations** (citation as reference):
> "Fault-tolerant gates have been demonstrated [1]."

### Self-Citation

- Cite your relevant prior work
- Don't over-cite yourself
- Frame appropriately: "We previously showed..." or "Building on prior work [ref]..."

---

## Common Errors in Physics Writing

### Error 1: Dangling Modifiers

**Poor**:
> "After optimizing the pulse sequence, the fidelity improved."

(The fidelity didn't optimize the pulse sequence.)

**Better**:
> "After we optimized the pulse sequence, the fidelity improved."
> OR
> "After optimization of the pulse sequence, the fidelity improved."

### Error 2: Misplaced Modifiers

**Poor**:
> "We only measured three qubits." (Did you only measure, not prepare?)

**Better**:
> "We measured only three qubits."

### Error 3: That vs. Which

**That**: Restrictive clause (essential information)
> "The qubits that had the longest coherence times were selected."

**Which**: Non-restrictive clause (parenthetical information)
> "The qubits, which had coherence times exceeding 100 μs, were selected."

### Error 4: Data is/are

"Data" is technically plural (singular: datum), but usage varies.

**Traditional**: "The data are consistent with theory."
**Modern (acceptable)**: "The data is consistent with theory."

Be consistent within your paper.

### Error 5: Effect vs. Affect

**Effect** (noun): "The effect was significant."
**Affect** (verb): "Temperature affects coherence."

Rarely: effect as verb (to effect change), affect as noun (psychology).

---

## Style Checklist

### Before Submitting, Verify:

**Clarity**:
- [ ] Every sentence can be understood in one reading
- [ ] Technical terms are defined
- [ ] Jargon is minimized

**Precision**:
- [ ] Numbers have appropriate significant figures
- [ ] Uncertainties are included
- [ ] Claims are quantified

**Conciseness**:
- [ ] No unnecessary words
- [ ] No redundant phrases
- [ ] Active voice predominates

**Flow**:
- [ ] Paragraphs have topic sentences
- [ ] Transitions connect sections
- [ ] Logical order throughout

**Mechanics**:
- [ ] Consistent tense usage
- [ ] Subject-verb agreement
- [ ] Parallel structure

---

## Resources for Continued Learning

### Essential Reading

1. **"The Elements of Style"** by Strunk & White
   - Classic principles of clear writing

2. **"Style: Lessons in Clarity and Grace"** by Joseph Williams
   - Advanced prose style techniques

3. **"The Science of Scientific Writing"** by Gopen & Swan
   - Physics-specific writing principles

4. **"Writing Science"** by Joshua Schimel
   - Story structure in scientific papers

### Online Resources

- APS Style Guide: https://journals.aps.org/authors
- Nature author guidelines: https://www.nature.com/nature/for-authors
- Duke Scientific Writing Resource: https://sciwrite.duke.edu

---

*Good writing is rewriting. Every physicist can learn to write clearly. It takes practice and revision.*
