# QSE Self-Study Curriculum — Agent Handoff Document

## PROJECT PURPOSE

Create a **complete, day-by-day self-study curriculum** covering graduate-level Quantum Science & Engineering. Target audience: motivated self-learners going from basic calculus to research-level quantum science over 6 years (~10,000+ hours).

**Philosophy:** Quality over speed. Deep research incorporating latest, modern, and futuristic knowledge. Each day file should be comprehensive (~10-15KB, 400-600 lines).

---

## CURRENT STATUS

### Year 0: ✅ COMPLETE (336 days)

| Semester | Months | Days | Status | Grade |
|----------|--------|------|--------|-------|
| **Semester 0A** | 1-3 | 1-84 | ✅ COMPLETE | A- (93/100) |
| **Semester 0B** | 4-6 | 85-168 | ✅ COMPLETE | A- (92/100) |
| **Semester 0C** | 7-9 | 169-252 | ✅ COMPLETE | A- (93/100) |
| **Semester 0D** | 10-12 | 253-336 | ✅ COMPLETE | A+ (98/100) |
| **TOTAL** | 1-12 | 1-336 | ✅ **100%** | **A (95/100)** |

### Year 1: ✅ COMPLETE (336 days)

| Semester | Months | Days | Status | Files |
|----------|--------|------|--------|-------|
| **Semester 1A** | 13-18 | 337-504 | ✅ COMPLETE | 198 files |
| **Semester 1B** | 19-24 | 505-672 | ✅ COMPLETE | 198 files |
| **TOTAL** | 13-24 | 337-672 | ✅ **100%** | **396 files** |

**Completion Date:** February 3, 2026

---

## YEAR 1 COMPLETION DETAILS

### Semester 1A: Foundations of Quantum Mechanics

| Month | Topic | Days | Status |
|-------|-------|------|--------|
| 13 | Postulates & Mathematical Framework | 337-364 | ✅ Complete |
| 14 | One-Dimensional Systems | 365-392 | ✅ Complete |
| 15 | Angular Momentum & Spin | 393-420 | ✅ Complete |
| 16 | Three-Dimensional Problems | 421-448 | ✅ Complete |
| 17 | Approximation Methods | 449-476 | ✅ Complete |
| 18 | Identical Particles & Many-Body | 477-504 | ✅ Complete |

### Semester 1B: Quantum Information Foundations

| Month | Topic | Days | Status |
|-------|-------|------|--------|
| 19 | Density Matrices & Mixed States | 505-532 | ✅ Complete |
| 20 | Entanglement Theory | 533-560 | ✅ Complete |
| 21 | Quantum Gates & Circuits | 561-588 | ✅ Complete |
| 22 | Quantum Algorithms I | 589-616 | ✅ Complete |
| 23 | Quantum Algorithms II | 617-644 | ✅ Complete |
| 24 | Quantum Channels & Error Introduction | 645-672 | ✅ Complete |

---

## NEXT: YEAR 2 — ADVANCED QUANTUM SCIENCE

### Year 2 Overview (Days 673-1008, Months 25-36)

**Primary Texts:**
- Nielsen & Chuang, Ch. 10 (Quantum Error Correction)
- Gottesman, "Stabilizer Codes and Quantum Error Correction"
- Preskill, Ph219 Lecture Notes Ch. 7
- Fowler et al., "Surface Codes" review papers

**Topics from Master Curriculum:**

| Month | Weeks | Days | Topic | Primary Reference |
|-------|-------|------|-------|-------------------|
| 25-26 | 97-104 | 673-728 | Quantum Error Correction Fundamentals | N&C Ch. 10, Gottesman |
| 27-28 | 105-112 | 729-784 | Stabilizer Formalism and CSS Codes | N&C Ch. 10, Preskill Ch. 7 |
| 29-30 | 113-120 | 785-840 | Topological Codes and Surface Codes | Fowler et al. reviews |
| 31-32 | 121-128 | 841-896 | Fault-Tolerant Quantum Computing | Preskill, Kitaev papers |
| 33-34 | 129-136 | 897-952 | Quantum Hardware Platforms | Platform-specific reviews |
| 35-36 | 137-144 | 953-1008 | Advanced Algorithms (HHL, QAOA deep) | N&C Ch. 5-6, recent papers |

---

## DIRECTORY STRUCTURE

### Current (Years 0-1 Complete)

```
Quantum-Engineering/
├── Harvard_QSE_PhD_Complete_Curriculum.md   # MASTER CURRICULUM
├── CLAUDE.md                                 # Project instructions
├── README.md
│
├── Year_0_Mathematical_Foundations/          # ✅ COMPLETE (336 days)
│   ├── AGENT_HANDOFF_DOCUMENT.md            # THIS FILE
│   ├── README.md
│   ├── Semester_0A_Calculus_DiffEq/         # ✅ 84 days
│   ├── Semester_0B_LinAlg_Complex_Mechanics/ # ✅ 84 days
│   ├── Semester_0C_Advanced_Foundations/     # ✅ 84 days
│   └── Semester_0D_Integration_Symmetry/     # ✅ 84 days
│
└── Year_1_Quantum_Mechanics_Core/            # ✅ COMPLETE (336 days)
    ├── README.md
    ├── YEAR_1_MASTER_PLAN.md
    ├── Semester_1A_QM_Foundations/           # ✅ 168 days
    │   ├── Month_13_Postulates_Framework/
    │   ├── Month_14_One_Dimensional/
    │   ├── Month_15_Angular_Momentum/
    │   ├── Month_16_Three_Dimensional/
    │   ├── Month_17_Approximation_Methods/
    │   └── Month_18_Identical_Particles_Many_Body/
    │
    └── Semester_1B_Quantum_Information/      # ✅ 168 days
        ├── Month_19_Density_Matrices/
        ├── Month_20_Entanglement_Theory/
        ├── Month_21_Quantum_Gates_Circuits/
        ├── Month_22_Quantum_Algorithms_I/
        ├── Month_23_Quantum_Algorithms_II/
        └── Month_24_Quantum_Channels_Error/
```

### To Create (Year 2)

```
Year_2_Advanced_Quantum_Science/              # TO CREATE (336 days)
├── README.md
├── AGENT_HANDOFF_DOCUMENT.md
│
├── Semester_2A_Error_Correction/             # Days 673-840
│   ├── Month_25_QEC_Fundamentals_I/          # Days 673-700
│   ├── Month_26_QEC_Fundamentals_II/         # Days 701-728
│   ├── Month_27_Stabilizer_Formalism/        # Days 729-756
│   ├── Month_28_CSS_Codes/                   # Days 757-784
│   ├── Month_29_Topological_Codes_I/         # Days 785-812
│   └── Month_30_Surface_Codes/               # Days 813-840
│
└── Semester_2B_Advanced_Topics/              # Days 841-1008
    ├── Month_31_Fault_Tolerance_I/           # Days 841-868
    ├── Month_32_Fault_Tolerance_II/          # Days 869-896
    ├── Month_33_Hardware_Platforms_I/        # Days 897-924
    ├── Month_34_Hardware_Platforms_II/       # Days 925-952
    ├── Month_35_Advanced_Algorithms_I/       # Days 953-980
    └── Month_36_Year2_Capstone/              # Days 981-1008
```

---

## DAY FILE TEMPLATE (MANDATORY STRUCTURE)

Each day file (`Day_XXX_[Weekday].md`) **MUST** follow this structure:

```markdown
# Day XXX: [Topic Title]

## Schedule Overview

| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: [Topic] |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives
(5-6 specific, measurable objectives)

---

## Core Content
(Main theory with LaTeX math, derivations, explanations)
- Use $$...$$ for display math
- Use $$\boxed{...}$$ for key equations
- Include proofs and derivations

---

## Physical Interpretation
(What the math means physically - CRITICAL!)

---

## Worked Examples
(2-3 detailed, step-by-step examples)

---

## Practice Problems
### Level 1: Direct Application
### Level 2: Intermediate
### Level 3: Challenging

---

## Computational Lab
(Python code with numpy, scipy, matplotlib, qiskit)
(Must be runnable, well-commented, with visualizations)

---

## Summary
(Key formulas table, main takeaways)

---

## Daily Checklist
- [ ] Checklist items for self-assessment

---

## Preview: Day XXX+1
(Brief teaser of next day's topic)
```

---

## RESEARCH REQUIREMENTS

**Before writing each day file, agents MUST:**

1. **Web search** 3-5 authoritative sources:
   - Wikipedia (overview)
   - Physics LibreTexts
   - MIT OCW / Stanford / Caltech notes
   - Academic PDFs (arXiv, university course notes)
   - Recent review papers for modern perspectives

2. **Validate** all formulas against multiple sources

3. **Include modern/futuristic connections:**
   - Quantum computing applications
   - Recent experimental advances
   - Cutting-edge research directions
   - Connections to fault-tolerant QC, quantum simulation

4. **Cross-reference** with master curriculum topics

---

## QUALITY STANDARDS

1. Each file follows template structure **exactly**
2. File size 10-15KB each (~400-600 lines)
3. LaTeX math renders correctly ($$...$$ format)
4. Python code runs without errors
5. Physical interpretation in **every** day
6. Week READMEs created for each week
7. Problems have 3 difficulty levels (2-3 each)
8. Modern/futuristic knowledge incorporated
9. Cross-referenced with master curriculum

---

## FULL PROGRAM ROADMAP

| Year | Focus | Days | Status |
|------|-------|------|--------|
| 0 | Mathematical & Physical Foundations | 1-336 | ✅ COMPLETE |
| 1 | Quantum Mechanics Core | 337-672 | ✅ COMPLETE |
| 2 | Advanced Quantum Science | 673-1008 | 🔜 NEXT |
| 3 | Qualifying Exam Preparation | 1009-1344 | Planned |
| 4-5 | Research Phase | 1345-2016 | Planned |

---

## IVY LEAGUE ALIGNMENT

### Year 0 Review (Completed)

| University | Prerequisite Match | Status |
|------------|-------------------|--------|
| Harvard QSE | Physics 143a/b prerequisites | ✅ Comparable |
| MIT 8.04/8.05 | Mathematical preparation | ✅ Meets |
| Caltech Ph125 | Group theory requirement | ✅ Comparable |
| Princeton PHY521 | Functional analysis level | ✅ Meets |
| Stanford | Applied math foundations | ✅ Comparable |

### Year 1 Coverage

| University | Course Equivalent | Coverage |
|------------|-------------------|----------|
| Harvard | QSE 200/201 | Comprehensive |
| MIT | 8.04/8.05/8.06, 8.370x | Comprehensive |
| Caltech | Ph125abc, Ph219 | Substantial |
| Princeton | PHY 521/522 | Substantial |
| Stanford | PHYSICS 130/131 | Comprehensive |

---

*Document created: January 28, 2026*
*Last updated: February 3, 2026*
*Status: Years 0-1 COMPLETE (672 days), Year 2 NEXT (336 days)*
