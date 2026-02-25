# Day 154: Week 22 Review — Lagrangian Mechanics II Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Concept Review & Problem Set A |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Set B & Self-Assessment |
| Evening | 6:00 PM - 7:00 PM | 1 hour | Week 23 Preview |

**Total Study Time: 7 hours**

---

## 🎯 Week 22 Learning Objectives — Final Check

- [ ] State and apply Noether's theorem
- [ ] Solve central force problems using effective potential
- [ ] Derive Kepler's laws
- [ ] Use reduced mass for two-body problems
- [ ] Find normal modes of coupled systems
- [ ] Compute and diagonalize inertia tensors
- [ ] Understand rigid body kinematics

---

## 📊 Week 22 Concept Map

```
                    LAGRANGIAN MECHANICS II
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
NOETHER'S THEOREM     APPLICATIONS           OSCILLATIONS
    │                       │                   & ROTATIONS
    │               ┌───────┴───────┐               │
Symmetry →      Central       Two-Body      ┌───────┴───────┐
Conservation     Force        Problem       │               │
    │              │              │       Small          Rigid
    │          ┌───┴───┐      ┌───┴───┐  Oscillations    Bodies
Time →      Effective  Orbit  Reduced  CM    │               │
Energy      Potential Equation Mass  Separation Normal    Inertia
    │           │         │      μ      │    Modes     Tensor
Space →     V_eff(r)   r=p/(1+ecosθ)  ψ=φ·ψ   │        │
Momentum      │         │      │       │    ω_n      Euler
    │      Circular  Kepler's         2T→3DOF  a_n    Angles
Rotation → Orbits    Laws              │       │        │
L           │         │               │    Coupled  Principal
            └────E<0: bound────────────┘    Systems    Axes
                 E>0: unbound
```

---

## 📝 Key Formulas Summary

### Noether's Theorem
| Symmetry | Transformation | Conserved Quantity |
|----------|----------------|-------------------|
| Time translation | t → t + ε | Energy h |
| Space translation | x → x + ε | Momentum pₓ |
| Rotation | θ → θ + ε | Angular momentum L |

$$I = \sum_i p_i Q_i - h\tau = \text{const}$$

### Central Force & Two-Body
| Formula | Description |
|---------|-------------|
| V_eff = V(r) + L²/(2μr²) | Effective potential |
| r = p/(1 + e cos θ) | Orbit equation |
| μ = m₁m₂/(m₁+m₂) | Reduced mass |
| T² = 4π²a³/(GM) | Kepler's 3rd law |

### Normal Modes
| Formula | Description |
|---------|-------------|
| det(**K** - ω²**M**) = 0 | Secular equation |
| **K****a**ₙ = ωₙ²**M****a**ₙ | Eigenvalue problem |
| **a**ₘᵀ**M****a**ₙ = δₘₙ | Orthogonality |

### Rigid Bodies
| Formula | Description |
|---------|-------------|
| T_rot = ½**ω**ᵀ**I****ω** | Rotational kinetic energy |
| Iᵢⱼ = Σm(r²δᵢⱼ - rᵢrⱼ) | Inertia tensor |
| **L** = **I**·**ω** | Angular momentum |

---

## 🔬 Problem Set A: Theory

### Problem A1: Noether's Theorem
A particle moves in a potential V(x, y) = V(x² + y²).
a) What continuous symmetry does this system have?
b) Use Noether's theorem to find the conserved quantity.
c) Verify by direct calculation that this quantity is conserved.

**Solution:**
a) Rotational symmetry about z-axis

b) Infinitesimal rotation: δx = -εy, δy = εx, τ = 0
   
   I = pₓQₓ + pᵧQᵧ = pₓ(-y) + pᵧ(x) = xpᵧ - ypₓ = Lz

c) dLz/dt = xṗᵧ - yṗₓ = x(-∂V/∂y) - y(-∂V/∂x)
   
   For V = V(r²): ∂V/∂x = 2x·V'(r²), ∂V/∂y = 2y·V'(r²)
   
   dLz/dt = -2xyV' + 2xyV' = 0 ✓

---

### Problem A2: Kepler Problem
A comet has perihelion distance r_p = 0.5 AU and aphelion r_a = 50 AU.
a) Find the eccentricity e and semi-major axis a.
b) Find the orbital period.
c) Calculate the speed at perihelion and aphelion.

**Solution:**
a) r_p = a(1-e), r_a = a(1+e)
   
   a = (r_p + r_a)/2 = 25.25 AU
   
   e = (r_a - r_p)/(r_a + r_p) = 49.5/50.5 ≈ 0.98

b) T² = a³ (with a in AU, T in years)
   
   T = a^(3/2) = 25.25^1.5 ≈ 127 years

c) Conservation of angular momentum: r_p v_p = r_a v_a
   
   Conservation of energy: ½v² - GM/r = const
   
   v_p = √(GM(2/r_p - 1/a)) ≈ 2.8 AU/year ≈ 44 km/s
   
   v_a = v_p · r_p/r_a ≈ 0.44 km/s

---

### Problem A3: Normal Modes
Two identical masses m are connected by three springs (k-k-k) to walls.
a) Write the mass and stiffness matrices.
b) Find the normal mode frequencies.
c) Find and interpret the mode shapes.

**Solution:**
a) **M** = m·I₂, **K** = k·[[2, -1], [-1, 2]]

b) det(**K** - ω²**M**) = (2k - mω²)² - k² = 0
   
   ω²₁ = k/m, ω²₂ = 3k/m
   
   ω₁ = √(k/m), ω₂ = √(3k/m)

c) Mode 1: **a**₁ = (1, 1)/√2 — in-phase
   
   Mode 2: **a**₂ = (1, -1)/√2 — out-of-phase

---

### Problem A4: Rigid Body
A uniform rectangular plate has dimensions a × b × c (c ≪ a, b).
a) Find the principal moments of inertia.
b) About which axis is rotation most stable? Least stable?

**Solution:**
a) With axes at center, parallel to edges:
   
   I₁ = M(b² + c²)/12 ≈ Mb²/12 (about a-axis)
   
   I₂ = M(a² + c²)/12 ≈ Ma²/12 (about b-axis)
   
   I₃ = M(a² + b²)/12 (about c-axis, perpendicular to plate)

b) If a > b: I₃ > I₂ > I₁
   
   Most stable: I₃ (largest) and I₁ (smallest)
   
   Least stable: I₂ (intermediate) — tennis racket theorem!

---

## 🔬 Problem Set B: Applications

### Problem B1: Two-Body Reduction
A binary star system has m₁ = 3M☉, m₂ = M☉, separation a = 10 AU.
a) Find the reduced mass and period.
b) Find the semi-major axes of each star's orbit.
c) If m₂ suddenly loses half its mass, what happens?

**Solution:**
a) μ = 3M☉ · M☉/(4M☉) = 0.75M☉
   
   T² = 4π²a³/(G·4M☉) → T = 5 years

b) a₁ = a · m₂/M = 2.5 AU
   
   a₂ = a · m₁/M = 7.5 AU

c) Total mass halves, orbit becomes unbound (E > 0) with new M_total.
   Stars fly apart on hyperbolic trajectories!

---

### Problem B2: Molecular Vibrations
For a linear H-C≡C-H molecule (acetylene):
a) How many vibrational degrees of freedom?
b) List the types of normal modes expected.

**Solution:**
a) 4 atoms × 3 = 12 total DOF
   
   Linear molecule: 3 translation + 2 rotation = 5
   
   Vibrations: 12 - 5 = **7 vibrational modes**

b) Stretching modes (3):
   - C-H symmetric stretch
   - C-H antisymmetric stretch
   - C≡C stretch
   
   Bending modes (4, doubly degenerate pairs):
   - H-C-C bend (2 modes, perpendicular planes)
   - C-C-H bend (2 modes, perpendicular planes)

---

### Problem B3: Euler's Equations
A symmetric top has I₁ = I₂ = I, I₃ = 2I.
a) Write Euler's equations.
b) Show that ω₃ = const.
c) Find the precession frequency of **ω** about the symmetry axis.

**Solution:**
a) Iω̇₁ = (I - 2I)ω₂ω₃ = -Iω₂ω₃
   
   Iω̇₂ = (2I - I)ω₃ω₁ = Iω₃ω₁
   
   2Iω̇₃ = (I - I)ω₁ω₂ = 0

b) From the third equation: ω̇₃ = 0, so ω₃ = const ✓

c) From first two equations:
   
   ω̇₁ = -ω₃ω₂, ω̇₂ = ω₃ω₁
   
   d/dt(ω₁ + iω₂) = -ω₃(ω₂ - iω₁) = iω₃(ω₁ + iω₂)
   
   Solution: ω₁ + iω₂ = A·e^{iω₃t}
   
   Precession frequency: **Ω = ω₃**

---

## 📊 Self-Assessment Rubric

| Topic | Score (1-5) | Notes |
|-------|-------------|-------|
| Noether's theorem | /5 | |
| Central force problem | /5 | |
| Kepler's laws derivation | /5 | |
| Two-body reduction | /5 | |
| Normal mode analysis | /5 | |
| Inertia tensor | /5 | |
| Euler's equations | /5 | |
| Problem solving | /5 | |

**Total: /40**

- 35-40: Excellent! Ready for Hamiltonian Mechanics
- 28-34: Good, review weak areas
- 20-27: Need more practice
- <20: Review week before continuing

---

## 🔮 Week 23 Preview: Hamiltonian Mechanics I

### Topics Coming:
1. **Legendre Transformation** — From L to H
2. **Hamilton's Equations** — First-order form
3. **Phase Space** — The arena of dynamics
4. **Poisson Brackets** — Algebraic structure
5. **Liouville's Theorem** — Phase space preservation

### Key Preparations:
- Review Legendre transformation from thermodynamics
- Practice with partial derivatives
- Think about (q, p) as independent variables

---

## ✅ Week 22 Completion Checklist

- [ ] Mastered Noether's theorem
- [ ] Solved central force problems
- [ ] Applied reduced mass to two-body systems
- [ ] Found normal modes of coupled oscillators
- [ ] Computed inertia tensors
- [ ] Understood rigid body rotation
- [ ] Completed both problem sets
- [ ] Self-assessment score ≥ 28/40

---

## 🎉 Congratulations!

You've completed **Week 22: Lagrangian Mechanics II**!

### Key Achievements:
- Noether's theorem: deepest connection in physics
- Central force → Kepler problem → planetary motion
- Two-body reduction → molecular and stellar systems
- Normal modes → vibrational spectroscopy
- Rigid body → rotating machinery and molecules

### The Big Picture:
Lagrangian mechanics provides:
1. Systematic approach to complex systems
2. Direct connection to symmetries and conservation
3. Foundation for field theory and quantum mechanics
4. Computational framework for simulations

**Next: Week 23 — Hamiltonian Mechanics!**
