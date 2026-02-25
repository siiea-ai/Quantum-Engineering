# 20-Minute Presentation Guide

## Structuring a Research-Level Oral Exam Presentation

---

## Introduction

The 20-minute presentation is your opportunity to demonstrate mastery of a specific topic. Unlike a lecture (where you teach) or a seminar (where you report new results), this presentation should show that you understand something deeply enough to do research in the area.

This guide provides detailed structure, timing, and techniques for an excellent presentation.

---

## Part 1: The Architecture of 20 Minutes

### Overview Structure

| Section | Time | Purpose |
|---------|------|---------|
| **Hook & Motivation** | 2-3 min | Why should anyone care? |
| **Background** | 4-5 min | What do we need to know? |
| **Core Content** | 8-10 min | The heart of the presentation |
| **Implications** | 2-3 min | So what? What's next? |
| **Summary** | 1-2 min | Key takeaways |

**Total:** 17-23 minutes (target: 20)

### Timing Discipline

**Critical Rule:** Know exactly where you should be at:
- 5 minutes: Finishing background
- 10 minutes: Deep into core content
- 15 minutes: Starting implications
- 18 minutes: Beginning summary

**Practice with a timer until your pacing is consistent.**

---

## Part 2: Section-by-Section Guide

### Section 1: Hook and Motivation (2-3 minutes)

**Purpose:** Capture attention and establish why this topic matters.

**Elements:**
1. **Opening statement** (15 sec): A compelling fact or question
2. **Context setting** (30 sec): Where does this fit in physics?
3. **The problem/question** (45 sec): What are we trying to understand?
4. **Roadmap** (30 sec): What will you cover?

**Example Opening Approaches:**

*The Big Question:*
> "Can we build a computer that solves problems classical computers never could? Today I'll explain one reason why the answer might be yes: quantum error correction."

*The Surprising Fact:*
> "A single physical qubit can be worse than useless for computation. Its error rate is too high. Yet somehow, by combining many bad qubits, we can make one good qubit. Let me explain how."

*The Historical Entry:*
> "In 1995, Peter Shor showed something remarkable: quantum information can be protected against noise. This discovery transformed quantum computing from a mathematical curiosity to a realistic goal."

**What NOT to Do:**
- Don't start with definitions
- Don't apologize or equivocate
- Don't start with "Today I'll talk about..."
- Don't be boring

**Board During This Section:**
```
+----------------------------------------------------------+
|                                                          |
|           [Title of Your Talk]                           |
|                                                          |
|   Key Question: [Your central question]                  |
|                                                          |
|   Roadmap:                                               |
|   1. Background                                          |
|   2. Core Result                                         |
|   3. Implications                                        |
|                                                          |
+----------------------------------------------------------+
```

---

### Section 2: Background (4-5 minutes)

**Purpose:** Give the audience what they need to understand your core content.

**Principle:** Include only what's necessary. Every piece of background should connect to something you'll use later.

**Elements:**
1. **Prerequisites** (1.5 min): Key concepts the audience needs
2. **Prior work** (1.5 min): What was known before your main topic?
3. **Gap or question** (1 min): What was missing or open?

**The "Just Enough" Rule:**
- Ask: "Will I use this later?"
- If yes: Include it
- If no: Cut it

**Example Background for "Surface Codes":**
```
Prerequisites (1.5 min):
- Quick review of Pauli operators and their properties
- Stabilizer formalism: code space as joint +1 eigenspace
- Mention that we can measure stabilizers without destroying encoded info

Prior Work (1.5 min):
- Classical error correction uses redundancy
- Quantum: no-cloning prevents naive copying
- Shor's breakthrough: can still use redundancy in a subspace
- CSS codes showed nice structure

Gap (1 min):
- But CSS codes need non-local operations
- Question: Can we correct errors with only local measurements?
- This leads to topological codes...
```

**Board During This Section:**
```
+----------------------------+----------------------------+
|     PREREQUISITES          |      PRIOR WORK            |
|                            |                            |
|  Pauli group: X, Y, Z      |  Classical: copy & vote    |
|  Stabilizers: g|ψ⟩ = |ψ⟩   |  Quantum: no-cloning!      |
|  Syndrome measurement      |  Shor: encode in subspace  |
|                            |  CSS: nice X/Z structure   |
+----------------------------+----------------------------+
|                                                         |
|  QUESTION: Can we do this with only local operations?   |
|                                                         |
+---------------------------------------------------------+
```

---

### Section 3: Core Content (8-10 minutes)

**Purpose:** Present your main topic with depth and precision.

**This is the heart of your presentation.** Spend the most time preparing this section.

**Structure Options:**

**Option A: Build-Up Structure**
1. Simple case or example (2 min)
2. General principle or definition (2 min)
3. Key derivation or proof (3 min)
4. Important properties or implications (2 min)

**Option B: Problem-Solution Structure**
1. State the problem precisely (1 min)
2. Approach to solution (2 min)
3. Solution execution (4 min)
4. Verification/interpretation (2 min)

**Option C: Concept-Example-Concept**
1. Introduce main concept (2 min)
2. Detailed worked example (3 min)
3. Generalize from example (2 min)
4. Key results and implications (2 min)

### Example: Core Content for Surface Codes

**Simple Case (2 min):**
> "Let me start with the toric code on a torus. Data qubits on edges, Z stabilizers around faces, X stabilizers around vertices..."
> [Draw the lattice, label stabilizers]

**Key Properties (2 min):**
> "What makes this special? First, all stabilizers are local - 4-body. Second, logical operators must span the torus - they're topologically protected..."
> [Show logical X and Z as non-contractible loops]

**Surface Code Modification (2 min):**
> "For practical systems, we use a planar version - the surface code. Now logical operators span boundary to boundary..."
> [Draw surface code with boundaries]

**Error Correction Mechanism (3 min):**
> "How do we correct errors? An error anticommutes with some stabilizers, creating syndrome defects. We pair them up and correct the path between..."
> [Show example error, syndromes, correction]

### Derivation Guidelines

If including a derivation:

1. **State what you're deriving** before starting
2. **Highlight the key step** - don't treat all steps equally
3. **Check limiting cases** if time permits
4. **Box the final result**

**Example:**
> "Let me derive the threshold condition. We need P(failure) → 0 as code distance → ∞. Starting from..."
> [Write derivation]
> "The key insight is here - when p < p_th, this sum converges..."
> [Circle the key step]
> "So we get the threshold equation..."
> [Box result: p_th ≈ 1%]

---

### Section 4: Implications and Extensions (2-3 minutes)

**Purpose:** Show that you understand the broader significance.

**Elements:**
1. **Immediate applications** (1 min): What can we do with this?
2. **Connections** (1 min): How does this relate to other areas?
3. **Open questions** (30 sec): What don't we know yet?

**Example:**
> "What does this mean for building quantum computers? Surface codes are now implemented in labs at Google, IBM, and others. The recent Google result demonstrated error suppression as code distance increased."
> [Mention specific experiments if relevant]

> "This connects to topological phases of matter - the surface code ground space is a topological phase. And to complexity theory - decoding is related to matching problems."

> "Open questions remain: Can we reduce the overhead? How do we implement logical gates fault-tolerantly? What's the ultimate limit?"

---

### Section 5: Summary (1-2 minutes)

**Purpose:** Reinforce key points and transition to questions.

**Elements:**
1. **Recap main points** (45 sec): 3-4 key takeaways
2. **The big picture** (30 sec): One sentence summary
3. **Opening for questions** (15 sec): Invite discussion

**Example:**
> "Let me summarize the key points:
> - Surface codes protect quantum information using topological properties
> - They require only local measurements on a 2D grid
> - Errors create syndromes that can be paired and corrected
> - The threshold is around 1%, achievable with current technology
>
> The bottom line: topological protection makes fault-tolerant quantum computing architecturally feasible.
>
> I'd be happy to discuss any of these points further."

**Final Board State:**
```
+----------------------------------------------------------+
|                                                          |
|   KEY RESULTS:                                           |
|                                                          |
|   1. Surface code: local stabilizers on 2D grid          |
|   2. Logical operators span the lattice                  |
|   3. Threshold ≈ 1% (achievable!)                        |
|   4. Decoding via minimum-weight matching                |
|                                                          |
+----------------------------------------------------------+
```

---

## Part 3: Board Planning

### Pre-Planning Your Board

Before presenting, create a "board script":

| Time | Board Section | Content |
|------|---------------|---------|
| 0-3 min | Top left | Title, roadmap |
| 3-8 min | Top center | Background, prerequisites |
| 8-12 min | Center | Main derivation/content |
| 12-16 min | Center right | Examples, properties |
| 16-18 min | Bottom | Applications, open questions |
| 18-20 min | Right side | Summary (keep visible) |

### What to Keep Visible

Throughout presentation, try to keep visible:
- Your title/topic
- Key definitions used repeatedly
- Important diagrams
- Final results

### Erasing Strategy

Erase in this order of priority (keep most important):
1. Routine calculations (erase freely)
2. Background material (erase once used)
3. Intermediate results (erase when superseded)
4. Key results (keep if possible)
5. Main diagrams (never erase if still relevant)

---

## Part 4: Common Mistakes

### Timing Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Slow start | Still on background at 8 min | Time your intro strictly |
| Rushed core | Skipping key steps | Cut background instead |
| Running over | "Let me just quickly..." | Practice with timer |
| Too short | Done at 15 min | Add more depth to core |

### Content Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Too much background | "You probably know this..." | Cut ruthlessly |
| No derivation | Just stating results | Include at least one |
| Lost in details | Committee looks confused | Step back, give intuition |
| No big picture | Just technical details | Add motivation and implications |

### Delivery Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Reading notes | Looking down, monotone | Practice until fluent |
| Talking to board | Committee sees your back | Write-stop-turn |
| No eye contact | Never looking at audience | Deliberately look up |
| Mumbling | "Sorry, what was that?" | Project and enunciate |

---

## Part 5: Practice Protocol

### Day 1: Rough Draft
- Write complete outline
- Identify all equations and diagrams
- Time yourself (don't worry about quality yet)

### Day 2: First Refinement
- Adjust for timing
- Practice derivations on board
- Record and watch

### Day 3: Second Refinement
- Practice with board plan
- Refine transitions
- Time each section

### Day 4: Polish
- Full practice runs (record all)
- Fix remaining issues
- Practice recoveries

### Day 5: Final
- One or two full runs
- Mental rehearsal
- Rest

### Recording Review Checklist

When reviewing recordings, check:
- [ ] Time in each section
- [ ] Board organization
- [ ] Body position and eye contact
- [ ] Speaking pace and clarity
- [ ] Handling of equations
- [ ] Quality of explanations
- [ ] Overall flow

---

## Part 6: Advanced Techniques

### Signposting

Guide your audience through the talk:

> "So far we've seen [X]. Now I want to show you [Y], which is the key to understanding [Z]."

> "This is the central result. Let me break down what it means."

> "Before I move on, let me check: the key insight here is [summary]."

### Handling Committee Reactions

**If they look confused:**
> "Let me approach this differently..."
> "The key intuition is..."

**If they nod encouragingly:**
> Continue, maybe add a bit more depth

**If they ask a clarifying question:**
> Answer briefly, return to flow

**If they ask a derailing question:**
> "That's a great question - let me address it briefly now and we can discuss more after. The key point is..."

### Building to Key Results

Create anticipation:

> "We've now set up all the pieces. Here's what we get when we put them together..."

> "This calculation is a bit involved, but the result is beautiful..."

> "You might wonder why we went through all that. Here's the payoff..."

---

## Part 7: Specific Examples

### Example 1: Quantum Teleportation Presentation Outline

**Hook (2 min):**
"Can you send a quantum state using only classical communication? The answer is yes - if you share entanglement."

**Background (4 min):**
- Bell states and entanglement (1 min)
- No-cloning theorem (1 min)
- Classical communication limitations (1 min)
- The puzzle: How to transmit without copying? (1 min)

**Core Content (10 min):**
- The protocol statement (2 min)
- Mathematical derivation (4 min)
- Why it works - key insights (2 min)
- Resource accounting (2 min)

**Implications (2 min):**
- Gate teleportation
- Quantum networks
- Experimental realizations

**Summary (2 min):**
Key points, big picture, open for questions

---

### Example 2: Grover's Algorithm Presentation Outline

**Hook (2 min):**
"Searching an unstructured database seems to require checking every entry. Quantum mechanics says otherwise."

**Background (4 min):**
- Query complexity model (1 min)
- Classical lower bound Ω(N) (1 min)
- Quantum superposition and interference (2 min)

**Core Content (10 min):**
- The oracle and marked state (2 min)
- Grover iteration: reflection and diffusion (3 min)
- Geometric interpretation (2 min)
- Optimality of √N (3 min)

**Implications (2 min):**
- Amplitude amplification generalization
- Applications to other algorithms
- Optimality and limitations

**Summary (2 min):**
Key points, significance, open for questions

---

## Part 8: Final Checklist

Before your presentation is ready:

**Content:**
- [ ] Clear motivation that captures attention
- [ ] Background includes only necessary material
- [ ] Core content has genuine depth
- [ ] At least one real derivation
- [ ] Clear implications and connections
- [ ] Strong summary with key takeaways

**Timing:**
- [ ] Total time: 18-22 minutes
- [ ] Background complete by 5 minutes
- [ ] Core content gets 8-10 minutes
- [ ] Consistent across practice runs

**Delivery:**
- [ ] Fluent without notes
- [ ] Good board organization
- [ ] Eye contact with "audience"
- [ ] Clear speaking voice
- [ ] Smooth transitions

**Preparation:**
- [ ] At least 3 full practice runs
- [ ] Recorded and reviewed
- [ ] Timing refined
- [ ] Recovery strategies practiced

---

*"The goal of your presentation is not to show everything you know - it's to demonstrate that you understand something deeply enough to explain it clearly and answer questions about it."*

---

**Week 186 | Day 1298 Primary Material**
