# Research Questions Framework

## Purpose

This framework helps you develop precise, answerable, and significant research questions for your thesis introduction. Each research question should meet the criteria below and follow the structured format provided.

---

## Quality Criteria for Research Questions

### The FINER Framework

| Criterion | Description | Self-Check Question |
|-----------|-------------|---------------------|
| **F**easible | Answerable with available resources | Can I actually answer this with my methods? |
| **I**nteresting | Engages the scientific community | Will experts care about the answer? |
| **N**ovel | Not already fully answered | What's new about this question? |
| **E**thical | Meets ethical standards | Are there any ethical concerns? |
| **R**elevant | Advances knowledge significantly | Why does this answer matter? |

### Specificity Check

Your research question should be:

**Too Broad:** "How can we improve quantum error correction?"
- This could fill an entire textbook

**Too Narrow:** "What is the threshold of the [[4,2,2]] code under pure dephasing?"
- This is a calculation, not a research question

**Just Right:** "How does the threshold of surface-code-based architectures depend on the ratio of bit-flip to phase-flip error rates, and what code deformations optimize performance for biased noise?"
- Specific, answerable, and significant

---

## Research Question Template

For each research question, complete the following template:

```
===========================================================================
RESEARCH QUESTION [NUMBER]: [DESCRIPTIVE TITLE]
===========================================================================

FORMAL STATEMENT:
-----------------
[Write the question as a precise, formal statement. Use mathematical
notation if it adds clarity.]

Example: "Given a biased noise model with phase-flip probability p_Z and
bit-flip probability p_X where η = p_Z/p_X >> 1, what is the optimal
geometry for a topological code, and how does the threshold scale with η?"

CONTEXT AND MOTIVATION:
-----------------------
[Explain why this question arose and why it matters. Reference prior work.]

Prior to this work, researchers understood that [PRIOR KNOWLEDGE]. However,
the question of [GAP IN KNOWLEDGE] remained open because [REASON].

This question is important because [SIGNIFICANCE - theoretical and practical].

HYPOTHESIS:
-----------
[State your hypothesis or conjecture. What did you expect to find?]

We hypothesize that [YOUR CONJECTURE] because [REASONING].

APPROACH:
---------
[Briefly describe how you addressed this question.]

We address this question by [METHODOLOGY - computational, analytical,
experimental]. Specifically, we [SPECIFIC APPROACH].

KEY FINDINGS:
-------------
[Summarize what you found. This connects to your contributions.]

We find that [MAIN RESULT]. This [CONFIRMS/REFUTES/EXTENDS] the hypothesis
because [EXPLANATION].

CHAPTER REFERENCE:
------------------
This question is addressed primarily in Chapter [X], with supporting
analysis in Chapters [Y] and [Z].

PUBLICATION REFERENCE (if applicable):
--------------------------------------
Results related to this question appear in: [CITATION]
===========================================================================
```

---

## Example Research Questions

### Example 1: Theoretical/Analytical

```
===========================================================================
RESEARCH QUESTION 1: Noise-Adapted Topological Codes
===========================================================================

FORMAL STATEMENT:
-----------------
Given a biased Pauli noise channel with error probabilities p_X, p_Y, p_Z
satisfying p_Z >> p_X, p_Y, how should the geometry of a topological
stabilizer code be modified to maximize the error threshold, and what
is the optimal threshold as a function of the bias ratio η = p_Z/p_X?

CONTEXT AND MOTIVATION:
-----------------------
Prior to this work, it was understood that the standard surface code has a
threshold of approximately 1% under depolarizing noise [Dennis et al. 2002].
Researchers had observed that biased noise, common in superconducting and
spin qubits, might allow for improved thresholds [Tuckett et al. 2018].
However, the optimal code geometry for biased noise and the precise
threshold scaling remained open questions.

This question is important because experimental platforms exhibit
significant noise bias (η ~ 100-1000 in superconducting qubits), and
exploiting this bias could dramatically reduce the overhead for
fault-tolerant quantum computing.

HYPOTHESIS:
-----------
We hypothesize that by elongating the surface code lattice in the direction
that protects against the dominant error type, the threshold can be
improved proportionally to the noise bias. Specifically, we conjecture
that p_th(η) ~ √η × p_th(η=1) for large η.

APPROACH:
---------
We address this question using a combination of analytical bounds from
statistical mechanics mappings and large-scale Monte Carlo simulations
of error correction. Specifically, we develop a family of "rectangular
surface codes" with aspect ratio determined by the noise bias and
simulate error correction for system sizes up to 10^6 qubits.

KEY FINDINGS:
-------------
We find that the optimal aspect ratio scales as √η and the threshold
improves as p_th(η) ≈ 0.5(1 - 1/η) for large η, exceeding our initial
hypothesis. In the infinite bias limit (η → ∞), the threshold approaches
50%, matching the repetition code bound.

CHAPTER REFERENCE:
------------------
This question is addressed primarily in Chapter 3, with supporting
analytical derivations in Chapter 2 and numerical methods in Chapter 4.

PUBLICATION REFERENCE:
--------------------------------------
[Your Name] et al., "Optimal topological codes for biased noise,"
Physical Review X, 2024.
===========================================================================
```

### Example 2: Computational/Algorithmic

```
===========================================================================
RESEARCH QUESTION 2: Efficient Decoding for Noise-Adapted Codes
===========================================================================

FORMAL STATEMENT:
-----------------
Can the noise-adapted surface codes from RQ1 be decoded efficiently (in
polynomial time) while achieving near-optimal error correction performance,
and what is the complexity-performance tradeoff for practical decoder
implementations?

CONTEXT AND MOTIVATION:
-----------------------
Optimal maximum-likelihood decoding of the surface code is #P-hard in the
worst case [Iyer & Poulin 2015]. However, efficient approximate decoders
exist that achieve near-threshold performance, including MWPM and UF.
For standard surface codes, these decoders have been extensively studied
[Higgott et al. 2023]. However, the modified geometry of noise-adapted
codes may require modified decoding strategies.

This question is important because practical fault-tolerant quantum
computing requires real-time decoding at rates matching qubit operation
times (~1 μs). Even optimal codes are useless without efficient decoders.

HYPOTHESIS:
-----------
We hypothesize that belief propagation combined with ordered statistics
decoding (BP-OSD) can be adapted to noise-adapted codes while maintaining
O(n log n) complexity, where n is the number of physical qubits.

APPROACH:
---------
We develop a modified BP-OSD decoder that accounts for the anisotropic
error structure inherent to noise-adapted codes. We benchmark against
optimal decoding (for small systems) and MWPM, measuring both logical
error rates and decoder runtime.

KEY FINDINGS:
-------------
We find that our adapted BP-OSD decoder achieves within 5% of optimal
performance while running in O(n^1.5) time—slightly worse than hypothesized
but practical for current system sizes. We identify decoder bottlenecks
and propose hardware acceleration strategies.

CHAPTER REFERENCE:
------------------
This question is addressed primarily in Chapter 5.

PUBLICATION REFERENCE:
--------------------------------------
[Your Name] et al., "Efficient decoding of noise-adapted topological
codes," Quantum, 2025.
===========================================================================
```

### Example 3: Experimental/Practical

```
===========================================================================
RESEARCH QUESTION 3: Practical Implementation Constraints
===========================================================================

FORMAL STATEMENT:
-----------------
How do practical constraints—including finite connectivity, gate
compilation overhead, and calibration requirements—affect the performance
advantage of noise-adapted codes compared to standard surface codes in
realistic superconducting qubit architectures?

CONTEXT AND MOTIVATION:
-----------------------
Theoretical analyses of noise-adapted codes typically assume ideal
conditions: perfect knowledge of noise parameters, arbitrary connectivity,
and negligible gate overhead. Real devices have limited connectivity
(typically nearest-neighbor on a 2D grid), require additional SWAP gates
for non-local operations, and have imperfectly characterized noise.
These practical constraints could erode the theoretical advantages.

This question is important because bridging the theory-experiment gap is
essential for translating research into practical quantum computing
improvements.

HYPOTHESIS:
-----------
We hypothesize that noise-adapted codes retain a significant advantage
(≥50% threshold improvement) over standard surface codes even when
accounting for practical constraints, provided the noise bias η > 10.

APPROACH:
---------
We develop a complete simulation pipeline incorporating realistic device
models for IBM and Google architectures. We include gate compilation,
SWAP overhead, measurement errors, and noise calibration uncertainty.
We compare noise-adapted and standard codes under identical conditions.

KEY FINDINGS:
-------------
We find that noise-adapted codes retain ~30-40% of their theoretical
advantage under realistic conditions for η > 20. Below η = 10, the
overhead from code adaptation exceeds the benefit. We provide practical
guidelines for when noise-adapted codes are advantageous.

CHAPTER REFERENCE:
------------------
This question is addressed primarily in Chapter 4.

PUBLICATION REFERENCE:
--------------------------------------
In preparation for Physical Review Applied.
===========================================================================
```

---

## Connecting Research Questions

After developing your individual questions, write a paragraph explaining how they connect:

```
RESEARCH QUESTION CONNECTIONS:

These three research questions form a coherent investigation of noise-adapted
quantum error correction, progressing from theory to practice.

RQ1 establishes the theoretical foundation: what is the optimal code structure?
Without answering RQ1, we cannot know what codes to implement.

RQ2 addresses a prerequisite for practical implementation: can these optimal
codes be decoded efficiently? An optimal code with exponential decoding
complexity would be useless in practice.

RQ3 bridges theory and experiment by asking whether the theoretical advantages
survive practical constraints. This question directly informs near-term
experimental efforts.

Together, these questions provide a complete picture of whether and how
noise-adapted codes can advance fault-tolerant quantum computing.

[INSERT FIGURE: Research question relationship diagram]
```

---

## Research Question Development Worksheet

Use this worksheet to develop each of your research questions:

### Question Development

| Step | Your Notes |
|------|------------|
| 1. Initial question (rough) | |
| 2. What's already known? | |
| 3. What specifically is unknown? | |
| 4. Why does answering this matter? | |
| 5. How did you address it? | |
| 6. What did you find? | |
| 7. Refined question (precise) | |

### Quality Check

| Criterion | ✓ | Notes |
|-----------|---|-------|
| Is it specific enough? | | |
| Is it answerable? | | |
| Is it novel? | | |
| Is it significant? | | |
| Does it connect to your work? | | |
| Is it clearly stated? | | |

---

## Common Pitfalls

1. **Questions that are too vague**
   - Fix: Add mathematical precision, specific system, clear scope

2. **Questions already answered by others**
   - Fix: Identify the specific novel aspect your work addresses

3. **Questions disconnected from your actual research**
   - Fix: Start from your results and work backward to the question

4. **Too many questions (>5)**
   - Fix: Consolidate related questions, identify the core themes

5. **Questions in passive voice**
   - Fix: Use active construction: "What is...?" not "It was investigated whether..."
