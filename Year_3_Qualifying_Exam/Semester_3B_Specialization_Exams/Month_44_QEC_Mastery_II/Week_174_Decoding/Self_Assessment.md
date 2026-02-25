# Week 174: Decoding Algorithms - Self-Assessment

## Overview

Use this self-assessment to evaluate your understanding of decoding algorithms for quantum error correction. Rate yourself honestly to identify areas for improvement.

**Scoring:**
- **3** - Can explain to others and solve novel problems
- **2** - Understand well, can apply to standard problems
- **1** - Basic familiarity, need review
- **0** - Cannot recall or explain

---

## Section 1: The Decoding Problem

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Define the decoding problem | | |
| Explain syndrome measurement | | |
| Define degeneracy in quantum codes | | |
| Explain maximum likelihood decoding | | |
| State complexity of optimal decoding | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Calculate syndrome from error | | |
| Identify degenerate errors | | |
| Formulate ML decoding problem | | |
| Convert ML to minimum weight | | |

### Self-Check Questions

1. What is the syndrome for an $$X_3$$ error on the 5-qubit code?
   - [ ] Can calculate immediately
   - [ ] Know the method, need to work it out
   - [ ] Unsure how to calculate

2. Why doesn't the decoder need to identify the exact error?
   - [ ] Can explain degeneracy argument
   - [ ] Know it involves stabilizers
   - [ ] Need review

---

## Section 2: MWPM Decoder

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Describe MWPM algorithm | | |
| Explain graph construction | | |
| State surface code threshold | | |
| Explain role of boundary vertices | | |
| Describe measurement error handling | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Construct matching graph from syndrome | | |
| Find minimum weight matching | | |
| Calculate edge weights for biased noise | | |
| Estimate logical error rate | | |
| Use PyMatching library | | |

### Self-Check Questions

1. What is the MWPM threshold for the surface code?
   - [ ] ~10.3% for code capacity
   - [ ] Know it's around 10% but not exact
   - [ ] Don't remember

2. How does the matching graph change for measurement errors?
   - [ ] Can describe 3D extension
   - [ ] Know time dimension is added
   - [ ] Unclear

---

## Section 3: Union-Find Decoder

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Describe union-find data structure | | |
| Explain cluster growth algorithm | | |
| State complexity | | |
| Compare threshold to MWPM | | |
| Explain real-time decoding advantage | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Implement find with path compression | | |
| Implement union by rank | | |
| Trace cluster growth example | | |
| Analyze when to use union-find | | |

### Self-Check Questions

1. What is the complexity of union-find decoding?
   - [ ] $$O(n \cdot \alpha(n))$$ where $$\alpha$$ is inverse Ackermann
   - [ ] Know it's almost linear
   - [ ] Unsure

2. Why is union-find threshold lower than MWPM?
   - [ ] Can explain local vs global optimality
   - [ ] Know it's a trade-off but not why
   - [ ] Don't know

---

## Section 4: Belief Propagation

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Describe BP algorithm | | |
| Explain factor graphs | | |
| State why BP fails for quantum codes | | |
| Describe OSD post-processing | | |
| Know when BP is appropriate | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Draw factor graph from check matrix | | |
| Compute BP message update | | |
| Identify short cycles | | |
| Apply BP to QLDPC codes | | |

### Self-Check Questions

1. Why does BP fail for most quantum codes?
   - [ ] Short cycles (4-cycles from CSS)
   - [ ] Know it fails but not why
   - [ ] Unclear

2. What is the girth requirement for BP success?
   - [ ] $$O(\log n)$$
   - [ ] Know larger girth is better
   - [ ] Don't remember

---

## Section 5: Neural Network Decoders

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Explain advantage of neural decoders | | |
| Describe training methodology | | |
| State AlphaQubit results | | |
| Explain transfer learning approach | | |
| Identify when neural decoders help | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Design input/output for neural decoder | | |
| Choose appropriate architecture | | |
| Generate training data | | |
| Evaluate decoder performance | | |

### Self-Check Questions

1. What improvement did AlphaQubit achieve?
   - [ ] 6% lower logical error than MWPM
   - [ ] Know it improved but not by how much
   - [ ] Don't remember

2. What types of noise can neural decoders capture?
   - [ ] Correlated, non-Markovian, leakage, drift
   - [ ] Know they handle correlated noise
   - [ ] Unsure

---

## Section 6: Decoder Comparison

### Decision-Making Skills

| Scenario | Can Choose Decoder? | Notes |
|----------|---------------------|-------|
| Research benchmarking | [ ] Yes [ ] No | |
| Real-time on FPGA | [ ] Yes [ ] No | |
| Correlated noise | [ ] Yes [ ] No | |
| QLDPC codes | [ ] Yes [ ] No | |
| Large-scale FT | [ ] Yes [ ] No | |

### Key Comparisons

| Comparison | Can Explain? | Notes |
|------------|--------------|-------|
| MWPM vs union-find threshold | [ ] Yes [ ] No | |
| BP vs MWPM for surface codes | [ ] Yes [ ] No | |
| Neural vs classical for real hardware | [ ] Yes [ ] No | |
| Complexity: MWPM vs UF vs BP | [ ] Yes [ ] No | |

---

## Comprehensive Assessment

### Oral Exam Readiness

Can you give a coherent 5-minute explanation of:

| Topic | Ready? |
|-------|--------|
| Why decoding is challenging | [ ] Yes [ ] Mostly [ ] No |
| MWPM for surface codes | [ ] Yes [ ] Mostly [ ] No |
| Trade-offs between decoders | [ ] Yes [ ] Mostly [ ] No |
| Future of decoding | [ ] Yes [ ] Mostly [ ] No |

### Problem-Solving Readiness

Can you solve problems involving:

| Type | Ready? |
|------|--------|
| Graph construction | [ ] Yes [ ] Mostly [ ] No |
| Threshold estimation | [ ] Yes [ ] Mostly [ ] No |
| Algorithm complexity | [ ] Yes [ ] Mostly [ ] No |
| Decoder selection | [ ] Yes [ ] Mostly [ ] No |

---

## Quick Reference Values

Test yourself by covering the right column:

| Concept | Value |
|---------|-------|
| MWPM threshold (code capacity) | 10.3% |
| Union-find threshold | 9.9% |
| Phenomenological threshold | ~3% |
| Circuit-level threshold | ~0.5-1% |
| MWPM complexity | $$O(n^3)$$ |
| Union-find complexity | $$O(n \cdot \alpha(n))$$ |
| BP complexity per iteration | $$O(n)$$ |
| AlphaQubit improvement | 6% lower error |

---

## Action Items

### Top 3 Strengths
1.
2.
3.

### Top 3 Areas Needing Improvement
1.
2.
3.

### Study Plan

| Area | Resources | Time | Deadline |
|------|-----------|------|----------|
| | | | |
| | | | |
| | | | |

---

## Progress Tracking

### Pre-Week
- Date: ___________
- Score: ___ / 100
- Main gaps: _________________________

### Post-Week
- Date: ___________
- Score: ___ / 100
- Improvements: _________________________

### Pre-Exam
- Date: ___________
- Score: ___ / 100
- Remaining concerns: _________________________

---

## Reflection Questions

1. **What decoder would you use for your research and why?**

2. **What is the most surprising thing you learned about decoding?**

3. **How has your understanding of the threshold theorem deepened?**

4. **What question about decoding do you still have?**

5. **How would you explain decoding to a non-expert?**
