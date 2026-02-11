# QEC Integration Exam: Oral Examination Questions

## Format

**Duration:** 45 minutes
- Opening/setup: 5 minutes
- Student presentation: 15 minutes
- Deep dive questions: 15 minutes
- Breadth questions: 10 minutes

---

## Presentation Topics

The student will be assigned one of these topics for a 15-minute whiteboard presentation.

### Topic 1: Threshold Theorem

**Prompt:** Present the threshold theorem for fault-tolerant quantum computation. Include the statement, key definitions, and a proof outline.

**Expected content:**
- Precise theorem statement
- Definition of fault-tolerant operation
- Concatenated code construction
- Recursion relation and threshold derivation
- Resource overhead analysis

---

### Topic 2: Eastin-Knill and Magic States

**Prompt:** Explain why transversal gates are insufficient for universal computation and how magic states provide a solution.

**Expected content:**
- Statement of Eastin-Knill theorem
- Proof sketch (discreteness argument)
- Definition of magic states
- Gate injection protocol
- 15-to-1 distillation overview

---

### Topic 3: Surface Code Decoding

**Prompt:** Describe the surface code structure and explain how MWPM decoding works.

**Expected content:**
- Surface code stabilizers and logical operators
- Syndrome interpretation
- Matching graph construction
- MWPM algorithm overview
- Threshold values and comparison

---

### Topic 4: QLDPC Breakthrough

**Prompt:** Explain the QLDPC conjecture and how Panteleev-Kalachev resolved it.

**Expected content:**
- QLDPC definition and conjecture
- Why the problem was hard
- Hypergraph product and its limitations
- Panteleev-Kalachev construction overview
- Implications for fault tolerance

---

## Deep Dive Questions

After the presentation, the examiner will probe understanding with targeted questions.

### For Topic 1 (Threshold Theorem):

1. What happens to the threshold if we use higher-distance codes?
2. Why is the threshold for surface codes higher than concatenated codes?
3. How do measurement errors affect the threshold?
4. What is the resource scaling for target error $$\epsilon = 10^{-15}$$?
5. Can you have a threshold without fault-tolerant syndrome extraction?

---

### For Topic 2 (Eastin-Knill and Magic States):

1. Why can't transversal gates form a continuous group?
2. What is the Clifford hierarchy and where does T sit?
3. Derive the output error formula for 15-to-1 distillation.
4. What is the distillation exponent and how is it calculated?
5. How does QLDPC achieve $$\gamma = 0$$?

---

### For Topic 3 (Surface Code Decoding):

1. Why can X and Z errors be decoded independently?
2. How does the matching graph change for measurement errors?
3. What is the complexity of MWPM and why?
4. Compare MWPM to union-find in detail.
5. How would a neural network decoder differ?

---

### For Topic 4 (QLDPC Breakthrough):

1. What is the role of expander graphs?
2. Why do non-Abelian groups give better distance than Abelian?
3. What are the practical challenges for QLDPC implementation?
4. How is single-shot error correction related to QLDPC?
5. When would you prefer QLDPC over surface codes?

---

## Breadth Questions

These questions test broad knowledge across all QEC topics.

### Stabilizer Formalism

1. What is the relationship between the Clifford group and stabilizer states?
2. How do you verify that stabilizer generators commute?
3. What distinguishes a degenerate quantum code?

---

### Surface Codes

1. What is the difference between rough and smooth boundaries?
2. How does code distance relate to minimum-weight logical operator?
3. Why is the surface code threshold so much higher than concatenated codes?

---

### Fault Tolerance

1. Define fault-tolerant and explain why it matters.
2. What are the main assumptions of the threshold theorem?
3. List three methods to implement T gates fault-tolerantly.

---

### Decoding

1. What makes the decoding problem computationally hard in general?
2. How does belief propagation fail for quantum codes?
3. What advantage do neural decoders have over classical algorithms?

---

### QLDPC

1. Why was the QLDPC conjecture important?
2. What parameters does the hypergraph product achieve?
3. What is constant-overhead fault tolerance?

---

## Rapid-Fire Questions

Answer in 1-2 sentences:

1. What is the MWPM threshold for the surface code?
2. What is the distillation exponent for 15-to-1?
3. How many logical qubits does the surface code encode?
4. What is the stabilizer weight for surface codes?
5. Who proved asymptotically good QLDPC exist?
6. What is single-shot error correction?
7. What is the union-find decoder complexity?
8. What constraint must CSS codes satisfy?
9. What is a magic state?
10. What is the Eastin-Knill theorem?

---

## Expected Answers (Rapid-Fire)

1. ~10.3% (code capacity), ~1% (circuit-level)
2. $$\gamma = \log_3(15) \approx 2.46$$
3. $$k = 1$$ (for standard surface code)
4. Weight 4 (bulk), weight 2-3 (boundary)
5. Panteleev and Kalachev (2021-2022)
6. Error correction with single round of noisy syndrome measurement
7. $$O(n \cdot \alpha(n))$$ where $$\alpha$$ is inverse Ackermann
8. $$H_X H_Z^T = 0$$
9. Non-stabilizer state enabling non-Clifford gates via injection
10. No QEC code has universal transversal gate set

---

## Scoring Rubric

### Presentation (40 points)

| Category | Excellent (10) | Good (7) | Fair (4) | Poor (1) |
|----------|---------------|----------|----------|----------|
| Content accuracy | All correct | Minor errors | Some gaps | Major errors |
| Completeness | All key points | Most points | Some points | Missing key points |
| Clarity | Crystal clear | Mostly clear | Some confusion | Hard to follow |
| Time management | Perfect timing | Slight over/under | Significant issues | Very off |

---

### Deep Dive (30 points)

| Category | Excellent (10) | Good (7) | Fair (4) | Poor (1) |
|----------|---------------|----------|----------|----------|
| Depth of understanding | Expert level | Good grasp | Basic understanding | Superficial |
| Handling probes | Addresses all | Most addressed | Struggles with some | Cannot answer |
| Connecting concepts | Makes connections | Some connections | Few connections | Isolated knowledge |

---

### Breadth (20 points)

| Category | Excellent (10) | Good (7) | Fair (4) | Poor (1) |
|----------|---------------|----------|----------|----------|
| Topic coverage | All topics | Most topics | Some topics | Few topics |
| Quick recall | Immediate | Some hesitation | Significant delay | Cannot recall |

---

### Communication (10 points)

| Category | Excellent (5) | Good (4) | Fair (2) | Poor (1) |
|----------|--------------|----------|----------|----------|
| Whiteboard use | Excellent | Good | Fair | Poor |
| Verbal clarity | Excellent | Good | Fair | Poor |

---

## Pass/Fail Criteria

**Pass:** ≥ 70 points overall, with no category below 50%
**Conditional:** 60-69 points, or one category below 50%
**Fail:** < 60 points, or multiple categories below 50%
