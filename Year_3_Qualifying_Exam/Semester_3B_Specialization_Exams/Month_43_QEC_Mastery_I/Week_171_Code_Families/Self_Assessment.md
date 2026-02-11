# Week 171: Code Families - Self Assessment

## Overview

This self-assessment evaluates your understanding of the major quantum error-correcting code families, their properties, and how to select codes for specific applications.

**Rating Scale:**
- **4 - Mastery:** Can explain trade-offs, derive properties, and make informed selections
- **3 - Proficiency:** Understand properties, can compare codes, minor gaps
- **2 - Developing:** Know basics, struggle with comparisons and applications
- **1 - Beginning:** Significant gaps, need review

---

## Core Concept Checklist

### Quantum Reed-Muller Codes

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Classical RM code parameters | | |
| Quantum RM construction | | |
| Transversal gate levels | | |
| Clifford hierarchy connection | | |
| [[15,1,3]] code properties | | |

**Competency check:**
- [ ] Can calculate RM(r,m) parameters?
- [ ] Can explain why QRM codes have transversal T?
- [ ] Can state the Clifford hierarchy definition?

---

### Color Codes

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Lattice structure | | |
| Stabilizer generators | | |
| Transversal Clifford proof | | |
| 2D vs 3D color codes | | |
| Comparison to surface codes | | |

**Competency check:**
- [ ] Can draw the [[7,1,3]] color code lattice?
- [ ] Can explain why transversal H works?
- [ ] Can compare threshold to surface code?

---

### Quantum Reed-Solomon Codes

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Classical RS codes | | |
| MDS property | | |
| Quantum RS construction | | |
| Singleton bound achievement | | |
| Qudit vs qubit considerations | | |

**Competency check:**
- [ ] Can construct a quantum RS code?
- [ ] Can explain why RS codes achieve Singleton bound?
- [ ] Can discuss qudit practicality issues?

---

### Concatenated Codes

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Concatenation construction | | |
| Distance multiplication | | |
| Threshold theorem | | |
| Overhead calculation | | |
| Comparison to topological codes | | |

**Competency check:**
- [ ] Can calculate concatenated code parameters?
- [ ] Can derive the threshold condition?
- [ ] Can estimate overhead for target error rate?

---

### Fundamental Bounds

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Quantum Hamming bound | | |
| Quantum Singleton bound | | |
| Gilbert-Varshamov bound | | |
| Linear programming bounds | | |

**Competency check:**
- [ ] Can state and apply all major bounds?
- [ ] Can determine if a code is optimal?
- [ ] Can prove the Singleton bound?

---

### Code Comparison and Selection

| Concept | Self-Rating (1-4) | Notes |
|---------|-------------------|-------|
| Parameter trade-offs | | |
| Threshold comparison | | |
| Gate set requirements | | |
| Connectivity constraints | | |
| Application-specific selection | | |

**Competency check:**
- [ ] Can recommend a code given constraints?
- [ ] Can justify the recommendation?
- [ ] Can identify when no good option exists?

---

## Quick Reference Test

Fill in from memory:

### Code Parameters
| Code | n | k | d |
|------|---|---|---|
| [[5,1,3]] | | | |
| Steane | | | |
| Shor | | | |
| [[15,1,3]] RM | | | |

### Transversal Gates
| Code | Gates |
|------|-------|
| General CSS | |
| Steane | |
| 2D Color | |
| [[15,1,3]] RM | |

### Thresholds
| Code Family | Approximate Threshold |
|-------------|----------------------|
| Concatenated | |
| Surface | |
| 2D Color | |

---

## Comparison Exercises

### Exercise 1: Side-by-Side
Create a comparison table for Steane vs Surface codes covering:
- Parameters
- Threshold
- Transversal gates
- Overhead
- Best use case

### Exercise 2: Selection Scenario
For each scenario, choose a code family and justify:

1. Physical error rate 0.1%, need full Clifford gates, 2D connectivity:
   - Choice: _______
   - Justification: _______

2. Physical error rate 0.001%, need T gates, all-to-all connectivity:
   - Choice: _______
   - Justification: _______

3. Physical error rate 1%, only need storage (no gates):
   - Choice: _______
   - Justification: _______

---

## Proof Practice

Can you prove these without notes?

| Statement | Can Prove? | Time |
|-----------|------------|------|
| Quantum Singleton bound | | |
| Concatenated distance multiplication | | |
| Threshold theorem outline | | |
| Transversal gates preserve stabilizers | | |

---

## Oral Exam Readiness

Rate your ability to give clear explanations:

| Topic | Rating (1-4) |
|-------|--------------|
| Overview of code families | |
| Reed-Muller transversal gates | |
| Color code structure | |
| Threshold theorem | |
| Code selection methodology | |
| Trade-off analysis | |

---

## Gap Analysis

### Strong Areas
List topics where you feel confident:
1. _______
2. _______
3. _______

### Weak Areas
List topics needing more work:
1. _______
2. _______
3. _______

### Action Plan
For each weak area, identify:
- Specific resource to study
- Problems to practice
- Target completion date

---

## Weekly Progress

| Day | Topic Reviewed | Problems Done | Remaining Gaps |
|-----|---------------|---------------|----------------|
| Mon | | | |
| Tue | | | |
| Wed | | | |
| Thu | | | |
| Fri | | | |
| Sat | | | |
| Sun | | | |

---

## Final Checklist

Before Week 172, verify:

### Essential (Must Have)
- [ ] Know parameters of major codes
- [ ] Understand threshold theorem
- [ ] Can compare code families
- [ ] Can apply fundamental bounds

### Important (Should Have)
- [ ] Can construct codes from each family
- [ ] Understand transversal gate constraints
- [ ] Can do overhead calculations
- [ ] Can justify code selections

### Advanced (Aim For)
- [ ] Understand Eastin-Knill theorem
- [ ] Can analyze novel codes
- [ ] Can design hybrid schemes

---

## Reflection

1. Which code family do you find most elegant? Why?

2. Which would you use for a real quantum computer? Why?

3. What surprised you most in this week's material?

4. What remains confusing?

5. How does this connect to previous weeks?

---

**Self-Assessment Created:** February 10, 2026
