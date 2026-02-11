# Articulating Your Research Gap

## Introduction

The gap statement is the pivotal moment in your Introduction. It creates the tension that your contribution resolves. A clear, well-articulated gap shows readers why your work matters and what makes it novel. This guide helps you identify, articulate, and validate your research gap.

## Part I: What is a Research Gap?

### Definition

A research gap is a specific problem, question, or limitation in the current state of knowledge that your work addresses. It represents:

- Something unknown that should be known
- Something impossible that should be possible
- Something unclear that should be understood
- A limitation that restricts progress

### The Gap-Contribution Connection

Your gap and contribution must connect directly:

```
GAP:          The problem/limitation that exists
              ↓
YOUR WORK:    What you did to address it
              ↓
CONTRIBUTION: How the gap is now (partially) closed
```

If these don't align, either revise your gap statement or reconsider your contribution framing.

## Part II: Types of Research Gaps

### Performance Gap

**Pattern:** "Current methods achieve X, but Y is needed."

**Suitable For:** Incremental improvements in established metrics

**Example:**
```
"State-of-the-art two-qubit gates achieve fidelities of 99.5%,
but fault-tolerant quantum computation requires >99.9% fidelity.
Closing this gap has proven challenging because..."
```

**Key Elements:**
- Quantify current state
- Quantify required/desired state
- Explain why closing gap is hard

### Capability Gap

**Pattern:** "X has never been demonstrated in/for Y."

**Suitable For:** First demonstrations, new systems, novel capabilities

**Example:**
```
"While high-fidelity gates have been demonstrated in fixed-frequency
transmon systems, extending these techniques to tunable couplers—
which offer additional control—remains unexplored."
```

**Key Elements:**
- Acknowledge what exists
- Clearly state what doesn't exist
- Indicate why extension is non-trivial

### Understanding Gap

**Pattern:** "X is observed, but the mechanism/cause/reason is unknown."

**Suitable For:** Fundamental studies, theoretical explanations

**Example:**
```
"Anomalous decoherence at specific flux operating points has been
observed in multiple experiments [refs], but the microscopic
origin of this effect remains unclear."
```

**Key Elements:**
- Cite observations of phenomenon
- State what is unknown
- Indicate importance of understanding

### Scalability Gap

**Pattern:** "Approach X works for Y, but fails/becomes impractical at Z."

**Suitable For:** Scaling studies, efficiency improvements

**Example:**
```
"Current calibration protocols require manual tuning of O(n²)
parameters, making the approach impractical beyond ~20 qubits.
Automating this process while maintaining calibration quality
is essential for larger systems."
```

**Key Elements:**
- State current scaling limitation
- Quantify size at which it fails
- Explain consequence of limitation

### Generality Gap

**Pattern:** "X has been studied only in Y; behavior in Z is unknown."

**Suitable For:** Generalization studies, new parameter regimes

**Example:**
```
"Prior work on dynamical decoupling has focused on the weak
driving regime. The behavior under strong driving, relevant
to fast gates, has not been systematically studied."
```

**Key Elements:**
- Define restricted scope of prior work
- Identify unexplored domain
- Explain relevance of unexplored domain

### Methodology Gap

**Pattern:** "Current methods for X require Y, limiting their applicability."

**Suitable For:** New methods, techniques, approaches

**Example:**
```
"Existing approaches to randomized benchmarking require the
ability to implement a complete Clifford group, which is
not native to all qubit architectures. A more general
benchmarking protocol is needed."
```

**Key Elements:**
- State limitation of current methods
- Explain consequence of limitation
- Indicate what improvement would enable

## Part III: Finding Your Gap

### Starting Questions

Answer these questions to identify your gap:

1. **Before your work:** What was the state of the art?
2. **Your contribution:** What changed because of your work?
3. **The difference:** Why couldn't someone else have done this?
4. **The benefit:** Who cares that this is now possible/known?

### The "Before/After" Exercise

| Before Your Work | After Your Work |
|-----------------|-----------------|
| [State of knowledge/capability] | [New state] |
| [Limitation that existed] | [How limitation is addressed] |
| [Question that was open] | [Answer you provide] |

The difference between "before" and "after" is your contribution.
The "before" column describes your gap.

### Validating Your Gap

**The Specificity Test:**
- Is the gap specific enough that your work addresses it?
- Is it narrow enough that progress is measurable?

**The Significance Test:**
- Does the gap matter to others in your field?
- Can you explain consequences of the gap?

**The Novelty Test:**
- Are you sure no one has addressed this?
- Check recent literature and preprints

**The Honesty Test:**
- Is this gap real, or manufactured to justify your work?
- Would experts agree this gap exists?

## Part IV: Writing Gap Statements

### Strong Gap Statement Patterns

**Pattern 1: Despite Progress**
```
"Despite significant progress in X [refs], achieving Y remains
challenging because Z."
```

**Pattern 2: However/Nevertheless**
```
"Previous work has demonstrated X [refs]. However, extending
this to Y has proven difficult due to Z."
```

**Pattern 3: While...Not**
```
"While X is now well-understood [refs], the behavior under
condition Y has not been systematically studied."
```

**Pattern 4: Limitation Statement**
```
"Current approaches to X require Y, which limits their
applicability to Z."
```

**Pattern 5: Open Question**
```
"A key open question is whether X is possible under condition Y.
Previous attempts have been limited by Z [refs]."
```

### Examples by Field

**Quantum Computing:**
```
"Two-qubit gate fidelities have steadily improved [1-5],
but the fastest high-fidelity gates remain sensitive to
parameter drift, requiring frequent recalibration. A robust
gate that maintains high fidelity despite experimental
variations would significantly ease scaling to larger systems."
```

**Condensed Matter:**
```
"Theoretical predictions of topological superconductivity in
semiconductor-superconductor heterostructures [6-8] have
motivated extensive experimental searches [9-12]. However,
definitive signatures distinguishing topological from trivial
states remain elusive, leaving the existence of Majorana
zero modes unconfirmed."
```

**Quantum Optics:**
```
"Photon-mediated entanglement between remote atoms has been
demonstrated using probabilistic protocols [13-15]. Scaling
these approaches to networks of many nodes requires near-
deterministic entanglement generation, which has not yet
been achieved."
```

### What Makes Gap Statements Weak

**Too Vague:**
```
WEAK: "There are still many open questions in this field."
BETTER: "The mechanism causing X under condition Y is unknown."
```

**Too Broad:**
```
WEAK: "Quantum computers are not yet practical."
BETTER: "Achieving the <0.1% error rates required for fault
tolerance remains challenging in current systems."
```

**Self-Serving:**
```
WEAK: "No one has used our specific approach before."
BETTER: Focus on what the approach enables, not its novelty per se.
```

**Manufactured:**
```
WEAK: "The color of quantum dots has not been studied at 3:47 AM."
BETTER: Identify a gap that matters to the field.
```

## Part V: Common Mistakes

### Mistake 1: Gap Doesn't Match Contribution

**Problem:** Gap statement describes problem A, but your work addresses problem B.

**Solution:** Revise gap to match what you actually did, or revise contribution framing.

### Mistake 2: Gap Too Broad

**Problem:** Gap is a major open problem that your work doesn't fully solve.

**Solution:** Narrow the gap to what you specifically address.

### Mistake 3: Gap Too Narrow

**Problem:** Gap is so specific it seems manufactured.

**Solution:** Frame within broader context, explain why specific gap matters.

### Mistake 4: Gap Already Closed

**Problem:** Someone else already addressed this gap.

**Solution:** Do thorough literature review. If gap exists, differentiate your work.

### Mistake 5: Significance Not Explained

**Problem:** Reader doesn't understand why gap matters.

**Solution:** Explain consequences: "This limits X, preventing Y, which matters because Z."

## Part VI: Exercises

### Exercise 1: Gap Identification

For your research, complete:

1. **Field context:** What is the broad area?
2. **Recent progress:** What advances have been made?
3. **Current limitation:** What can't we do yet?
4. **Your advance:** What can we do now?
5. **Significance:** Why does this matter?

### Exercise 2: Gap Statement Drafting

Write three versions of your gap statement:

**Version 1 (Performance frame):**
```
"Despite X, Y has not been achieved because Z."
[Your text here]
```

**Version 2 (Understanding frame):**
```
"While X is known, the Y remains unclear."
[Your text here]
```

**Version 3 (Capability frame):**
```
"X has been demonstrated, but extending to Y requires..."
[Your text here]
```

Select the version that best fits your work.

### Exercise 3: Validation

Test your gap statement:

- [ ] I can quantify the current state
- [ ] I can quantify what's needed
- [ ] I can explain why the gap exists
- [ ] I can explain why closing it matters
- [ ] My work addresses this specific gap
- [ ] Experts would agree this gap is real

## Summary

A strong gap statement:

1. **Is specific** - Narrow enough that your work addresses it
2. **Is significant** - Matters to people in your field
3. **Is honest** - Represents a real limitation
4. **Connects to contribution** - Your work addresses it
5. **Is well-supported** - Citations establish current state

---

*Use this guide alongside the main Introduction Guide and Template to craft your gap statement.*
