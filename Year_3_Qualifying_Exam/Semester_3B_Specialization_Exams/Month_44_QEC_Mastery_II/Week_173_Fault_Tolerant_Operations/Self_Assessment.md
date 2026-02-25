# Week 173: Fault-Tolerant Operations - Self-Assessment

## Overview

Use this self-assessment to evaluate your mastery of fault-tolerant quantum computation concepts. Be honest with yourself—identifying gaps is the first step to filling them.

**Scoring Guide:**
- **3** - Can explain to others and solve novel problems
- **2** - Understand well, can apply to standard problems
- **1** - Basic familiarity, need review
- **0** - Cannot recall or explain

---

## Section 1: Threshold Theorem

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| State the threshold theorem precisely | | |
| Explain why the theorem was historically significant | | |
| Describe the concatenated code construction | | |
| Define "fault-tolerant operation" | | |
| Explain why fault tolerance enables the threshold | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Calculate logical error rate for concatenation level $$L$$ | | |
| Determine required concatenation level for target error | | |
| Count malignant error sets | | |
| Derive threshold from gadget complexity | | |
| Compare thresholds for different codes/constructions | | |

### Self-Check Questions

1. Can you write down the recursion $$p^{(L+1)} = f(p^{(L)})$$?
   - [ ] Yes, immediately
   - [ ] Yes, with some thought
   - [ ] Need to look it up

2. Can you explain why $$p^{(L)} \propto (p/p_{\text{th}})^{2^L}$$ for distance-3?
   - [ ] Yes, and derive it
   - [ ] Yes, conceptually
   - [ ] No, need review

3. Can you compare surface code vs concatenated code thresholds?
   - [ ] Yes, with quantitative reasoning
   - [ ] Yes, qualitatively
   - [ ] No, uncertain

---

## Section 2: Transversal Gates

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Define transversal gate | | |
| Explain why transversal gates are fault-tolerant | | |
| List transversal gates for Steane code | | |
| Explain why CSS codes have transversal CNOT | | |
| Describe relationship between code symmetry and transversal gates | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Verify a gate is transversal for a given code | | |
| Determine logical gate from transversal physical gate | | |
| Identify when $$H^{\otimes n}$$ is a valid logical gate | | |
| Analyze stabilizer transformation under transversal gates | | |

### Self-Check Questions

1. For the Steane code, what does $$S^{\otimes 7}$$ implement?
   - [ ] Logical $$S$$
   - [ ] Logical $$S^\dagger$$
   - [ ] Something else / need to check

2. Why is $$\text{CNOT}^{\otimes n}$$ transversal for any CSS code?
   - [ ] Can explain with stabilizer argument
   - [ ] Know it's true but can't prove
   - [ ] Uncertain

---

## Section 3: Eastin-Knill Theorem

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| State the Eastin-Knill theorem | | |
| Explain the key proof idea | | |
| Describe why continuous transversal gates are impossible | | |
| Connect to Knill-Laflamme conditions | | |
| Explain implications for universal FTQC | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Outline the proof structure | | |
| Identify the role of discreteness | | |
| Explain why finite groups can't be universal | | |

### Self-Check Questions

1. Can you explain the "continuous symmetry" argument?
   - [ ] Yes, in detail
   - [ ] General idea only
   - [ ] Need review

2. What are the three main workarounds for Eastin-Knill?
   - [ ] Magic states, code switching, gauge fixing
   - [ ] Can name some but not all
   - [ ] Uncertain

---

## Section 4: Magic States

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Define magic state | | |
| Write down the T-magic state | | |
| Explain gate injection protocol | | |
| Describe why magic states enable universality | | |
| Connect to Clifford hierarchy | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Draw gate injection circuit | | |
| Prove gate injection implements T gate | | |
| Analyze error propagation from noisy magic states | | |

### Self-Check Questions

1. What is $$|T\rangle$$ explicitly?
   - [ ] $$\frac{1}{\sqrt{2}}(|0\rangle + e^{i\pi/4}|1\rangle)$$
   - [ ] Something similar but need to verify
   - [ ] Cannot recall

2. If magic state has error $$\epsilon$$, what is output error after injection?
   - [ ] Also $$O(\epsilon)$$
   - [ ] Depends on details
   - [ ] Don't know

---

## Section 5: Magic State Distillation

### Conceptual Understanding

| Topic | Score (0-3) | Notes |
|-------|-------------|-------|
| Explain the purpose of distillation | | |
| Describe the 15-to-1 protocol | | |
| Explain why cubic error suppression occurs | | |
| Define distillation exponent $$\gamma$$ | | |
| Describe constant-overhead distillation breakthrough | | |

### Technical Skills

| Skill | Score (0-3) | Notes |
|-------|-------------|-------|
| Calculate output error from input error | | |
| Determine rounds needed for target error | | |
| Compute total input states required | | |
| Derive $$\gamma$$ for 15-to-1 protocol | | |
| Account for rejection probability | | |

### Self-Check Questions

1. What is the output error formula for 15-to-1?
   - [ ] $$\epsilon_{\text{out}} = 35\epsilon_{\text{in}}^3$$
   - [ ] Something with $$\epsilon^3$$ but wrong coefficient
   - [ ] Don't remember

2. What is $$\gamma$$ for standard distillation vs QLDPC-based?
   - [ ] $$\sim 2.5$$ vs $$0$$
   - [ ] Know one but not the other
   - [ ] Uncertain

---

## Comprehensive Assessment

### Oral Exam Readiness

Can you give a coherent 5-minute explanation of:

| Topic | Ready? | Notes |
|-------|--------|-------|
| Why fault-tolerant QC is possible | [ ] Yes [ ] Mostly [ ] No | |
| What limits transversal gates | [ ] Yes [ ] Mostly [ ] No | |
| How magic states enable universality | [ ] Yes [ ] Mostly [ ] No | |
| Complete FTQC picture | [ ] Yes [ ] Mostly [ ] No | |

### Problem-Solving Readiness

Can you solve problems involving:

| Type | Ready? | Notes |
|------|--------|-------|
| Threshold calculations | [ ] Yes [ ] Mostly [ ] No | |
| Transversal gate analysis | [ ] Yes [ ] Mostly [ ] No | |
| Distillation overhead | [ ] Yes [ ] Mostly [ ] No | |
| Proof-based questions | [ ] Yes [ ] Mostly [ ] No | |

---

## Action Items

Based on your self-assessment, identify:

### Top 3 Strengths
1.
2.
3.

### Top 3 Areas Needing Improvement
1.
2.
3.

### Specific Study Plan

| Area | Resources to Review | Time Needed | Deadline |
|------|---------------------|-------------|----------|
| | | | |
| | | | |
| | | | |

---

## Progress Tracking

### Pre-Week Assessment
- Date: ___________
- Overall score: ___ / 100
- Major gaps identified: _________________________

### Post-Week Assessment
- Date: ___________
- Overall score: ___ / 100
- Improvement areas: _________________________

### Final Assessment (Pre-Exam)
- Date: ___________
- Overall score: ___ / 100
- Remaining concerns: _________________________

---

## Quick Reference Formulas

Use this section to test yourself. Cover the right column and try to recall each formula.

| Concept | Formula |
|---------|---------|
| Concatenated error rate | $$p^{(L)} = p_{\text{th}}\left(\frac{p}{p_{\text{th}}}\right)^{2^L}$$ |
| Threshold definition | $$p_{\text{th}} \approx 1/A$$ where $$A$$ = malignant pair count |
| T-magic state | $$\|T\rangle = \frac{1}{\sqrt{2}}(\|0\rangle + e^{i\pi/4}\|1\rangle)$$ |
| 15-to-1 distillation | $$\epsilon_{\text{out}} = 35\epsilon_{\text{in}}^3$$ |
| Distillation overhead | $$N = O(\log^{\gamma}(1/\epsilon))$$ |
| 15-to-1 $$\gamma$$ | $$\gamma = \log_3(15) \approx 2.46$$ |

---

## Reflection Questions

After completing this week's material, reflect on:

1. **What was the most surprising result you learned?**

2. **What connection between topics became clearer?**

3. **What would you explain differently now vs before this week?**

4. **What question would you most fear being asked on an oral exam?**

5. **What aspect of fault tolerance do you find most elegant?**
