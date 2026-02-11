# QIP Conference Presentation Standards

## Guidelines for Quantum Information Processing Talks

---

## About QIP

QIP (Quantum Information Processing) is the flagship annual conference for theoretical quantum information science. Established in 1998 (originally as QIP Workshop), it has grown into the premier venue for presenting foundational advances in quantum computing, quantum communication, quantum cryptography, and related areas.

### Conference Characteristics

- **Audience**: Primarily theoretical researchers, strong mathematical sophistication
- **Focus**: Foundational advances, new algorithms, complexity results, protocols
- **Format**: Contributed talks (20-25 min), invited talks (45-60 min), tutorials
- **Selection**: Highly competitive (~15-20% acceptance for contributed talks)
- **Proceedings**: No formal proceedings; talks often correspond to arXiv preprints

---

## QIP Talk Expectations

### What the Audience Expects

1. **Novel theoretical contribution**: New algorithm, proof, protocol, or complexity result
2. **Rigorous foundations**: Claims should be precisely stated and justified
3. **Clear significance**: Why does this advance matter for quantum information?
4. **Technical depth**: Audience can handle mathematics; don't oversimplify
5. **Honest scope**: Acknowledge limitations and open questions

### What Distinguishes QIP Talks

Compared to broader physics venues, QIP talks can:
- Assume more quantum information background
- Present more mathematical detail
- Focus on asymptotic results and complexity
- Engage with theoretical subtleties

But should still:
- Motivate the problem clearly
- Explain why it was hard
- Give insight into the solution approach
- Be accessible to the full QIP audience (not just your subfield)

---

## Standard Talk Formats

### Contributed Talk (20-25 minutes)

**Structure:**

| Section | Time | Content |
|---------|------|---------|
| Motivation | 3 min | Problem importance, prior work, your contribution |
| Background | 4 min | Essential definitions, notation, prior results |
| Main Result | 10-12 min | Statement, proof idea, key techniques |
| Extensions/Implications | 3 min | Corollaries, applications, open questions |
| Summary | 2 min | Takeaways |
| Q&A | 3-5 min | After or during (venue dependent) |

**Slide Count**: 15-20 main slides + 5-10 backup slides

### Invited Talk (45-60 minutes)

Invited talks have more flexibility:
- Extended motivation connecting to broader context
- More detailed technical exposition
- Multiple related results
- Deeper open questions discussion

---

## Content Guidelines for Theory Talks

### Stating Theorems

**Slide Structure:**
```
Theorem [Name/Number] (Main Result)

[Plain language summary in 1-2 sentences]

Formal Statement:
For all [conditions], [conclusion]:
$$\text{Precise mathematical statement}$$

Significance: [Why this is interesting/important/surprising]
```

**Example:**
```
Theorem (Main Result)

We can simulate any quantum circuit with magic states using
exponentially fewer resources than previously known.

For any circuit C with T gates:
$$\text{Resources}(C) = O(2^{0.5t}) \text{ vs. prior } O(2^t)$$

This brings classically-hard circuits into simulable range.
```

### Proof Sketches

Don't try to present full proofs. Instead:

1. **State the key insight**: What's the main idea that makes it work?
2. **Sketch the structure**: "We first show X, then use X to prove Y"
3. **Highlight novelty**: What technique is new vs. standard?
4. **Point to paper**: "Details in arXiv:XXXX.XXXXX"

**Effective phrasing:**
- "The key observation is..."
- "The main technical lemma shows..."
- "The novel ingredient is..."
- "Standard techniques then give..."

### Complexity-Theoretic Results

For computational complexity results:

1. **Define complexity classes clearly** (even familiar ones, briefly)
2. **State the relationship precisely** (inclusion, separation, equivalence)
3. **Explain the proof strategy** (reduction, simulation, oracle)
4. **Discuss implications** (for quantum computing, for understanding complexity)

**Example slide structure:**
```
MIP* = RE

Multiprover interactive proofs with entanglement can
decide all recursively enumerable languages.

Implies:
• Connes' Embedding Problem is false
• NEEXP ⊆ MIP*
• Entanglement is computationally powerful beyond expectations

Proof approach: [2-3 sentence overview]
```

### Algorithm Presentations

For new quantum algorithms:

1. **Problem statement**: What does the algorithm solve?
2. **Complexity comparison**: Classical vs. quantum vs. your algorithm
3. **High-level approach**: Key quantum techniques used
4. **Circuit/protocol sketch**: Visual representation
5. **Resource requirements**: Qubits, gates, depth
6. **Noise considerations**: How robust to errors?

---

## Visual Standards

### Equations

QIP audiences are comfortable with mathematical notation. Use equations, but:
- Define all notation
- Limit to 2-3 equations per slide
- Box or highlight the main result
- Use color to draw attention to key terms

### Diagrams

Common diagram types at QIP:
- Quantum circuits
- Communication protocols (Alice-Bob diagrams)
- Complexity class containment diagrams
- Proof structure flowcharts

### Tables

Useful for:
- Complexity comparisons
- Resource requirement comparisons
- Prior work summary

---

## Specific Topics at QIP

### Quantum Algorithms

**Expected content:**
- Problem definition and classical complexity
- Algorithm overview (circuit-level or high-level)
- Complexity analysis (query, gate, space)
- Comparison to prior quantum algorithms
- Near-term feasibility discussion (increasingly expected)

### Quantum Error Correction

**Expected content:**
- Code construction or new technique
- Distance, rate, or threshold results
- Comparison to surface code or other standards
- Decoding complexity
- Physical implementation considerations

### Quantum Complexity Theory

**Expected content:**
- Class definitions involved
- Precise statement of relationship
- Proof technique overview
- Barriers or difficulty explanation
- Implications for quantum computing

### Quantum Cryptography

**Expected content:**
- Security model and assumptions
- Protocol description
- Security proof overview
- Comparison to prior protocols
- Practical considerations (noise tolerance, rate)

---

## Q&A Preparation

### Common Question Types at QIP

1. **Clarification**: "Can you clarify what you mean by X?"
2. **Comparison**: "How does this relate to the work of Y?"
3. **Extension**: "Does this work if you change assumption Z?"
4. **Limitation**: "What happens in the case of W?"
5. **Connection**: "Have you considered the connection to V?"

### Preparing Backup Slides

Create backup slides for:
- Full proof details (by lemma/theorem)
- Comparison to specific prior work
- Extensions you've considered
- Limitations you're aware of
- Numerical examples or special cases

### Handling Difficult Questions

- **If you don't know**: "That's a great question. I haven't considered that case, but my intuition is..." or "I don't know, but I'd be happy to discuss after."
- **If it's outside scope**: "That's beyond what we address in this work, but it's an interesting direction for future research."
- **If it's long/complex**: "That's a rich question. Perhaps we can discuss after the session."

---

## QIP-Specific Advice

### First-Time QIP Presenters

1. **Attend first, present second**: If possible, attend QIP before presenting to understand the culture
2. **Practice with theory colleagues**: Get feedback from people comfortable with QIP level
3. **Prepare for depth**: Expect probing questions from experts
4. **Know the literature**: Be prepared to discuss related work

### Common Mistakes at QIP

| Mistake | Why It Happens | Solution |
|---------|----------------|----------|
| Too basic | Underestimating audience | Calibrate to QIP level |
| Too detailed | Trying to prove everything | Sketch proofs, point to paper |
| Missing context | Assumes everyone knows your subfield | Brief background for other areas |
| No takeaway | Got lost in technicalities | State significance explicitly |

---

## Resources

### Past QIP Talks

- QIP YouTube channel
- Conference websites with slides (check acceptance emails for repositories)
- PIRSA (Perimeter Institute archive) for related theory talks

### QIP Community Norms

- Strong emphasis on priority and attribution
- Open problem discussions valued
- arXiv preprints expected for all presented work
- Follow-up conversations in hallways are crucial

### Timeline for QIP Preparation

| When | Activity |
|------|----------|
| -3 months | Abstract submission |
| -2 months | Notification of acceptance |
| -6 weeks | Draft slides complete |
| -4 weeks | Practice with research group |
| -2 weeks | Practice with external colleagues |
| -1 week | Final refinements |
| -1 day | Venue tech check if possible |
