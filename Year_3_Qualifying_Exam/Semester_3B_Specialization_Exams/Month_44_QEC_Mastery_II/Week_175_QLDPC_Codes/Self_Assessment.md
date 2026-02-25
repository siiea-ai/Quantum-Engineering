# Week 175: QLDPC Codes - Self-Assessment

## Overview

Evaluate your understanding of quantum LDPC codes using this self-assessment. Be honest to identify areas needing review.

**Scoring:**
- **3** - Can explain and solve novel problems
- **2** - Understand well, apply to standard problems
- **1** - Basic familiarity, need review
- **0** - Cannot recall or explain

---

## Section 1: Classical LDPC Foundations

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Define classical LDPC code | | |
| Draw Tanner graph | | |
| Explain belief propagation | | |
| State asymptotic goodness | | |
| Connect expansion to distance | | |

### Self-Check

1. What makes a classical code "LDPC"?
   - [ ] Sparse parity-check matrix with $$O(1)$$ weights
   - [ ] Know it involves sparsity
   - [ ] Unsure

2. What parameters do good classical LDPC achieve?
   - [ ] $$[n, \Theta(n), \Theta(n)]$$
   - [ ] Know they're good but not exact scaling
   - [ ] Don't remember

---

## Section 2: CSS Construction

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| State commutativity constraint | | |
| Explain why constraint is hard | | |
| Define QLDPC code | | |
| Explain early construction limitations | | |

### Self-Check

1. What is the CSS constraint?
   - [ ] $$H_X H_Z^T = 0 \mod 2$$
   - [ ] Know X and Z must commute
   - [ ] Unsure

2. Why is the surface code QLDPC but not "good"?
   - [ ] $$w = O(1)$$ but $$k = O(1)$$, $$d = O(\sqrt{n})$$
   - [ ] Know it has limitations
   - [ ] Don't understand

---

## Section 3: Hypergraph Product

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| State construction | | |
| Calculate output parameters | | |
| Explain distance limitation | | |
| State significance | | |

### Self-Check

1. What is the quantum distance from hypergraph product?
   - [ ] $$d = \min(d_1, d_2) = \Theta(\sqrt{n})$$
   - [ ] Know it's sublinear
   - [ ] Unsure

2. Why was hypergraph product important despite limitations?
   - [ ] First constant-rate QLDPC
   - [ ] Know it was a step forward
   - [ ] Don't remember

---

## Section 4: Panteleev-Kalachev

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| State main theorem | | |
| Explain construction overview | | |
| Explain role of expansion | | |
| Explain non-Abelian advantage | | |

### Self-Check

1. What parameters do P-K codes achieve?
   - [ ] $$[[n, \Theta(n), \Theta(n)]]$$ with $$O(1)$$ stabilizer weight
   - [ ] Know they're asymptotically good
   - [ ] Unsure

2. Why do non-Abelian groups help?
   - [ ] Break averaging that limits Abelian to $$d = O(n/\log n)$$
   - [ ] Know it's about group structure
   - [ ] Don't understand

---

## Section 5: Constant-Overhead FT

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Define distillation exponent $$\gamma$$ | | |
| State QLDPC achievement | | |
| Explain mechanism | | |
| Compare to surface codes | | |

### Self-Check

1. What is $$\gamma$$ for QLDPC-based distillation?
   - [ ] $$\gamma = 0$$ (constant overhead)
   - [ ] Know it's improved
   - [ ] Don't remember

2. How does linear distance enable constant overhead?
   - [ ] $$\epsilon \to \epsilon^{\Theta(n)}$$ in one round
   - [ ] Know it's related to distance
   - [ ] Unsure

---

## Section 6: Practical Considerations

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Identify connectivity challenges | | |
| Describe decoding approaches | | |
| Estimate thresholds | | |
| Compare to surface codes | | |

### Self-Check

1. Main practical challenge for QLDPC?
   - [ ] Non-local connectivity requirements
   - [ ] Know there are challenges
   - [ ] Unsure

2. What decoder is used for QLDPC?
   - [ ] BP+OSD
   - [ ] Know it's not MWPM
   - [ ] Don't know

---

## Quick Reference Values

| Concept | Value |
|---------|-------|
| Surface code: $$w$$ | 4 |
| Surface code: $$d$$ scaling | $$\Theta(\sqrt{n})$$ |
| Hypergraph product: rate | $$\Theta(1)$$ |
| Hypergraph product: distance | $$\Theta(\sqrt{n})$$ |
| P-K: rate | $$\Theta(1)$$ |
| P-K: distance | $$\Theta(n)$$ |
| Standard $$\gamma$$ | ~2.5 |
| QLDPC $$\gamma$$ | 0 |
| QLDPC conjecture duration | ~20 years |

---

## Comprehensive Assessment

### Oral Exam Readiness

| Topic | Ready? |
|-------|--------|
| Why QLDPC matters | [ ] Yes [ ] Mostly [ ] No |
| Hypergraph product | [ ] Yes [ ] Mostly [ ] No |
| P-K achievement | [ ] Yes [ ] Mostly [ ] No |
| Constant overhead FT | [ ] Yes [ ] Mostly [ ] No |

### Problem-Solving Readiness

| Type | Ready? |
|------|--------|
| Parameter calculations | [ ] Yes [ ] Mostly [ ] No |
| Construction analysis | [ ] Yes [ ] Mostly [ ] No |
| Comparison problems | [ ] Yes [ ] Mostly [ ] No |
| Proof sketches | [ ] Yes [ ] Mostly [ ] No |

---

## Action Items

### Strengths
1.
2.
3.

### Areas for Improvement
1.
2.
3.

### Study Plan

| Area | Resources | Time | Deadline |
|------|-----------|------|----------|
| | | | |
| | | | |

---

## Progress Tracking

### Pre-Week
- Date: ___________
- Score: ___ / 100

### Post-Week
- Date: ___________
- Score: ___ / 100

### Pre-Exam
- Date: ___________
- Score: ___ / 100

---

## Reflection

1. **What is the most important thing you learned about QLDPC codes?**

2. **How does this change your view of fault-tolerant QC?**

3. **What would you want to learn more about?**

4. **What question about QLDPC do you still have?**
