# Day 54: Week 8 Problem Set — Vector Calculus Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Parts I-II |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Parts III-IV |
| Evening | 7:00 PM - 8:30 PM | 1.5 hours | Self-Assessment |

**Total Study Time: 7.5 hours**

---

## 📋 Instructions

This problem set tests your mastery of vector calculus. Show all work.

---

# 📝 PART I: VECTOR FIELDS

## Section A: Divergence and Curl (5 points each)

**A1.** Find div **F** and curl **F** for **F** = ⟨xy, yz, xz⟩

**A2.** Find div **F** and curl **F** for **F** = ⟨eˣʸ, sin(yz), cos(xz)⟩

**A3.** Find div **F** for **F** = ⟨x², y², z²⟩/r³ where r = √(x²+y²+z²)

**A4.** Verify that curl(∇f) = **0** for f(x,y,z) = x²y + yz² + zx

## Section B: Conservative Fields (6 points each)

**B1.** Is **F** = ⟨2xy + z, x² + 1, x⟩ conservative? If yes, find the potential.

**B2.** Is **F** = ⟨y cos(xy), x cos(xy)⟩ conservative? If yes, find the potential.

**B3.** Is **F** = ⟨y, -x⟩ conservative?

---

# 📝 PART II: LINE INTEGRALS

## Section C: Scalar Line Integrals (6 points each)

**C1.** ∫_C xy ds where C is the line from (0,0) to (4,3)

**C2.** ∫_C (x² + y²) ds where C is the circle x² + y² = 4

**C3.** ∫_C z ds where C is the helix **r**(t) = ⟨cos t, sin t, t⟩, 0 ≤ t ≤ 2π

## Section D: Vector Line Integrals (6 points each)

**D1.** ∫_C **F** · d**r** where **F** = ⟨y², x⟩ and C is the parabola y = x² from (0,0) to (1,1)

**D2.** ∫_C y dx + x dy along the line from (1,0) to (0,1)

**D3.** ∫_C **F** · d**r** where **F** = ∇(x²y + z³) from (0,0,0) to (1,1,1)

**D4.** ∮_C (x + y) dx + (x - y) dy where C is the unit circle, counterclockwise

---

# 📝 PART III: GREEN'S, STOKES', AND DIVERGENCE THEOREMS

## Section E: Green's Theorem (7 points each)

**E1.** Use Green's Theorem to evaluate ∮_C (x² - y) dx + (y² + x) dy where C is the square [0,2] × [0,2]

**E2.** Use Green's Theorem to find the area enclosed by x = cos t, y = sin 2t, 0 ≤ t ≤ π

**E3.** Verify Green's Theorem for **F** = ⟨xy, x²⟩ and D the triangle (0,0), (1,0), (0,1)

## Section F: Stokes' Theorem (8 points each)

**F1.** Use Stokes' Theorem to evaluate ∮_C **F** · d**r** where **F** = ⟨y, z, x⟩ and C is the boundary of the triangle (1,0,0), (0,1,0), (0,0,1)

**F2.** Use Stokes' Theorem: ∮_C ⟨-y², x, z²⟩ · d**r** where C is the circle x² + y² = 4 in the plane z = 3

**F3.** Verify Stokes' Theorem for **F** = ⟨z, x, y⟩ where S is the hemisphere z = √(1-x²-y²), z ≥ 0

## Section G: Divergence Theorem (8 points each)

**G1.** Use Divergence Theorem: ∬_S ⟨x, y, z⟩ · d**S** where S is the sphere x² + y² + z² = 4

**G2.** Use Divergence Theorem: ∬_S ⟨x³, y³, z³⟩ · d**S** where S is the unit sphere

**G3.** Use Divergence Theorem to find the flux of **F** = ⟨x², y², z²⟩ through the boundary of [0,1]³

---

# 📝 PART IV: COMPREHENSIVE PROBLEMS

## Section H: Mixed Applications (10 points each)

**H1.** A fluid flows with velocity **v** = ⟨x, y, -2z⟩.
(a) Is the flow incompressible? (Is div **v** = 0?)
(b) Find the flux through the hemisphere z = √(1-x²-y²), z ≥ 0

**H2.** Let **F** = ⟨-y, x, 0⟩/(x² + y²).
(a) Show that curl **F** = **0** for (x,y) ≠ (0,0)
(b) Compute ∮_C **F** · d**r** around the unit circle
(c) Explain why this doesn't contradict Stokes' Theorem

**H3.** Prove that for any closed surface S enclosing volume V:
$$\iint_S \mathbf{r} \cdot d\mathbf{S} = 3V$$
where **r** = ⟨x, y, z⟩

---

# ✅ ANSWER KEY

## Section A
- A1: div = y + z + x; curl = ⟨-y, -z, -x⟩
- A2: div = yeˣʸ + z cos(yz) - x sin(xz); curl = ⟨-y cos(yz), z sin(xz), -xeˣʸ⟩
- A3: 0 (for r ≠ 0)
- A4: Direct verification

## Section B
- B1: Yes; f = x²y + xz + y + C
- B2: Yes; f = sin(xy) + C
- B3: No (curl = -2 ≠ 0)

## Section C
- C1: 30
- C2: 16π
- C3: √2 · 2π²

## Section D
- D1: 1/3
- D2: 0
- D3: f(1,1,1) - f(0,0,0) = 2
- D4: -2π

## Section E
- E1: 8
- E2: 4/3
- E3: Both give 1/6

## Section F
- F1: -1/2
- F2: 8π
- F3: Both give -π

## Section G
- G1: 32π
- G2: 12π/5
- G3: 3

## Section H
- H1: (a) Yes, div = 0; (b) 0
- H2: (a) Direct; (b) 2π; (c) Surface doesn't exist (hole at origin)
- H3: div(**r**) = 3, apply Divergence Theorem

---

## 📊 Scoring Guide

| Section | Points | Your Score |
|---------|--------|------------|
| A (4 × 5) | 20 | |
| B (3 × 6) | 18 | |
| C (3 × 6) | 18 | |
| D (4 × 6) | 24 | |
| E (3 × 7) | 21 | |
| F (3 × 8) | 24 | |
| G (3 × 8) | 24 | |
| H (3 × 10) | 30 | |
| **TOTAL** | **179** | |

### Grade Scale
- 160-179: Excellent (A)
- 140-159: Good (B)
- 120-139: Satisfactory (C)
- Below 120: Review needed

---

## 🔜 Tomorrow: Computational Lab

---

*"Vector calculus unifies the geometry of curves, surfaces, and solids with the calculus of differentiation and integration."*
