# Week 169: Classical to Quantum Codes - Self Assessment

## Overview

Use this self-assessment to gauge your mastery of Week 169 material before the qualifying examination. Be honest with yourself—identifying gaps now allows time for remediation.

**Rating Scale:**
- **4 - Mastery:** Can explain to others, derive from first principles, apply to novel problems
- **3 - Proficiency:** Understand well, can solve standard problems, minor gaps
- **2 - Developing:** Basic understanding, struggle with applications, need review
- **1 - Beginning:** Significant gaps, need substantial study

---

## Core Concept Checklist

### Classical Error Correction

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Linear codes and vector space structure | | |
| Generator matrix construction | | |
| Parity-check matrix and syndrome | | |
| Hamming distance and error-correcting capability | | |
| Singleton bound proof | | |
| Hamming code example | | |

**Minimum competency questions:**
- [ ] Can you find the generator matrix from a parity-check matrix?
- [ ] Can you decode a received word using syndrome calculation?
- [ ] Can you state and prove the Singleton bound?

---

### Quantum Error Types

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Pauli group structure and multiplication | | |
| Bit-flip (X) errors | | |
| Phase-flip (Z) errors | | |
| Combined (Y) errors | | |
| Pauli expansion of general operators | | |
| Error discretization theorem | | |

**Minimum competency questions:**
- [ ] Can you compute products of Pauli operators (e.g., $$X_1Z_2 \cdot Y_1X_2$$)?
- [ ] Can you expand a general $$2 \times 2$$ matrix in the Pauli basis?
- [ ] Can you explain why correcting Pauli errors suffices for general errors?

---

### Knill-Laflamme Conditions

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Statement of KL conditions | | |
| Physical interpretation | | |
| Necessity proof | | |
| Sufficiency proof | | |
| Construction of recovery map | | |
| Degenerate vs non-degenerate codes | | |

**Minimum competency questions:**
- [ ] Can you state the KL conditions precisely without notes?
- [ ] Can you verify KL for a simple code (e.g., 2-qubit code)?
- [ ] Can you explain why $$C_{ab}$$ must be independent of $$i$$?

---

### Three-Qubit Codes

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Bit-flip code construction | | |
| Bit-flip syndrome measurement | | |
| Phase-flip code construction | | |
| Relationship via Hadamard | | |
| Limitations of each code | | |
| Logical operators | | |

**Minimum competency questions:**
- [ ] Can you write the encoding circuit for the bit-flip code?
- [ ] Can you explain why the bit-flip code fails for phase errors?
- [ ] Can you find $$\overline{X}$$ and $$\overline{Z}$$ for each code?

---

### Shor Code

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Concatenation concept | | |
| Explicit codeword construction | | |
| Syndrome identification | | |
| Error correction procedure | | |
| Distance calculation | | |
| Stabilizer generators (preview) | | |

**Minimum competency questions:**
- [ ] Can you write $$|0_L\rangle$$ and $$|1_L\rangle$$ for the Shor code?
- [ ] Can you determine the syndrome for a given single-qubit error?
- [ ] Can you prove the Shor code has distance 3?

---

### Bounds and Existence

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Quantum Hamming bound | | |
| Quantum Singleton bound | | |
| [[5,1,3]] code existence | | |
| Comparison with classical bounds | | |

**Minimum competency questions:**
- [ ] Can you state both bounds?
- [ ] Can you prove the quantum Singleton bound?
- [ ] Can you verify whether a code saturates the Hamming bound?

---

## Problem-Solving Skills Assessment

### Calculation Skills

For each skill, attempt a problem and record your result:

| Skill | Problem Attempted | Correct? | Time |
|-------|-------------------|----------|------|
| Syndrome calculation (classical) | Problem 1 | | |
| Pauli operator multiplication | Problem 6 | | |
| Knill-Laflamme verification | Problem 11 | | |
| Codeword construction | Problem 23 | | |
| Syndrome identification (quantum) | Problem 24 | | |

**Target:** 80%+ accuracy, reasonable time (10-15 min per problem)

---

### Proof Skills

Attempt these proofs without notes:

| Proof | Completed? | Gaps Identified |
|-------|------------|-----------------|
| Singleton bound (classical) | | |
| Error discretization theorem | | |
| Knill-Laflamme (necessity) | | |
| Knill-Laflamme (sufficiency) | | |
| Quantum Singleton bound | | |
| Shor code distance | | |

**Target:** Can complete each proof in 10-15 minutes with clear logical structure

---

## Oral Examination Readiness

### Can You Explain...

Rate your ability to give a clear 3-minute oral explanation:

| Topic | Rating (1-4) | Practice Needed? |
|-------|--------------|------------------|
| Why quantum error correction is harder than classical | | |
| The no-cloning connection to QEC | | |
| What the Knill-Laflamme conditions mean physically | | |
| How the Shor code is constructed | | |
| The difference between detection and correction | | |
| What makes a code degenerate | | |

---

### Common Oral Questions

Practice answering these aloud (record yourself):

1. "State the Knill-Laflamme conditions."
   - [ ] Can state precisely
   - [ ] Can explain the meaning of each term
   - [ ] Can give the physical interpretation

2. "Why can't we just copy quantum information for error protection?"
   - [ ] Can explain no-cloning
   - [ ] Can explain entanglement-based encoding
   - [ ] Can give specific example

3. "Walk me through the Shor code."
   - [ ] Can explain concatenation idea
   - [ ] Can write codewords
   - [ ] Can explain error correction procedure

4. "Prove the quantum Singleton bound."
   - [ ] Can set up the argument
   - [ ] Can use no-cloning correctly
   - [ ] Can complete the proof

5. "Compare classical and quantum error correction."
   - [ ] Can list key similarities
   - [ ] Can list key differences
   - [ ] Can explain why differences exist

---

## Study Plan Based on Assessment

### If mostly 4s and 3s:
- Focus on problem speed and proof elegance
- Practice oral explanations
- Move to advanced topics (degenerate codes, subsystem codes)

### If mostly 3s and 2s:
- Review lecture notes and textbook sections
- Work through more problems with solutions
- Form study group for discussion

### If mostly 2s and 1s:
- Start from scratch with basic concepts
- Watch video lectures (Preskill, IBM Quantum)
- Seek help from instructor or TA

---

## Remediation Resources

### For Classical Codes:
- MacWilliams & Sloane, "The Theory of Error-Correcting Codes"
- MIT 6.451 (Error-Correcting Codes) on OCW

### For Knill-Laflamme:
- Original paper: arXiv:quant-ph/9604034
- Preskill notes Chapter 7

### For Shor Code:
- Nielsen & Chuang Section 10.3
- IBM Quantum Learning tutorials

### For Proofs:
- Practice with study group
- Write out proofs by hand repeatedly
- Teach the material to someone else

---

## Weekly Progress Tracker

| Day | Material Reviewed | Problems Completed | Questions/Gaps |
|-----|-------------------|-------------------|----------------|
| Monday | | | |
| Tuesday | | | |
| Wednesday | | | |
| Thursday | | | |
| Friday | | | |
| Saturday | | | |
| Sunday | | | |

---

## Final Self-Assessment

Before moving to Week 170, verify:

### Essential Knowledge (Must have all)
- [ ] Can state Knill-Laflamme conditions from memory
- [ ] Can verify KL for the Shor code
- [ ] Can explain error discretization
- [ ] Can prove quantum Singleton bound
- [ ] Can construct syndrome measurement circuits

### Important Skills (Should have most)
- [ ] Can design simple quantum codes
- [ ] Can identify degenerate codes
- [ ] Can compare code efficiency
- [ ] Can explain material orally

### Advanced Understanding (Aim for some)
- [ ] Can construct recovery operations explicitly
- [ ] Can analyze approximate error correction
- [ ] Can connect to fault-tolerance

---

## Reflection Questions

Answer these in your study journal:

1. What is the most important concept from this week?

2. What concept was most difficult? Why?

3. How does this week's material connect to previous learning?

4. What questions remain unanswered?

5. How confident do you feel about this material on an oral exam?

---

**Self-Assessment Created:** February 9, 2026
**Next Review:** After completing Week 170
