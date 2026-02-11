# Day 147: Week 21 Review — Lagrangian Mechanics I Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Concept Review & Problem Set A |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Set B & Self-Assessment |
| Evening | 6:00 PM - 7:00 PM | 1 hour | Week 22 Preview |

**Total Study Time: 7 hours**

---

## 🎯 Week 21 Learning Objectives — Final Check

- [ ] Define and use generalized coordinates
- [ ] Count degrees of freedom correctly
- [ ] Distinguish holonomic from non-holonomic constraints
- [ ] State the Principle of Least Action
- [ ] Derive Euler-Lagrange equations from δS = 0
- [ ] Apply E-L equations to physical systems
- [ ] Use Lagrange multipliers for constraints
- [ ] Connect symmetries to conservation laws

---

## 📊 Week 21 Concept Map

```
                    LAGRANGIAN MECHANICS I
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   FOUNDATIONS         VARIATIONAL          APPLICATIONS
        │               PRINCIPLE               │
   ┌────┴────┐        ┌────┴────┐        ┌────┴────┐
   │         │        │         │        │         │
Generalized  DOF    Action   Euler-   Constraints Conservation
Coordinates   │      S=∫Ldt  Lagrange     │         │
   │        n=3N-k    │        │      ┌───┴───┐  ┌───┴───┐
Config     Holonomic  L=T-V   d/dt   Eliminate Cyclic  Energy
Space      vs Non-H    │    (∂L/∂q̇) Coords   Coords  h=Σpq̇-L
   │          │        │    -∂L/∂q  Lagrange   │       │
Mass      f(q,t)=0   δS=0    =0    Multiplier  pᵢ   ∂L/∂t=0
Matrix        │                        λ      const   →h=const
Mᵢⱼ          │                        │         
             └────────────────────────┘
```

---

## 📝 Key Formulas Summary

### Configuration and Constraints
| Formula | Description |
|---------|-------------|
| n = 3N - k | Degrees of freedom |
| f(q, t) = 0 | Holonomic constraint |
| rᵢ = rᵢ(q₁,...,qₙ,t) | Position in generalized coords |
| T = ½Σᵢⱼ Mᵢⱼ q̇ᵢq̇ⱼ | Kinetic energy (quadratic form) |

### Lagrangian Formalism
| Formula | Description |
|---------|-------------|
| L = T - V | Lagrangian |
| S = ∫L dt | Action |
| δS = 0 | Principle of least action |
| d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ = 0 | Euler-Lagrange equation |

### Generalized Quantities
| Formula | Description |
|---------|-------------|
| pᵢ = ∂L/∂q̇ᵢ | Generalized momentum |
| Qᵢ = ∂L/∂qᵢ | Generalized force |
| h = Σᵢ pᵢq̇ᵢ - L | Energy function |

### Conservation Laws
| Condition | Conserved Quantity |
|-----------|-------------------|
| ∂L/∂qᵢ = 0 | pᵢ (momentum) |
| ∂L/∂t = 0 | h (energy) |

---

## 🔬 Problem Set A: Foundations

### Problem A1: Degrees of Freedom
Determine the DOF and choose generalized coordinates:
a) Particle on a sphere of radius R
b) Rigid rod in a plane (two endpoints)
c) Three particles connected by rigid rods (triangle)
d) Disk rolling on a line

**Solutions:**
a) N=1, k=1 (r=R), n=2. Coords: (θ, φ)
b) N=2, k=3 (rigid rod in plane), n=1. Coord: angle θ
c) N=3, k=3 (rigid triangle), n=3. Coords: (x_cm, y_cm, θ)
d) N=1, k=1 (rolling), n=1. Coord: x (or θ with x = Rθ)

---

### Problem A2: Lagrangian Construction
Write the Lagrangian for:
a) Particle in gravitational field near Earth's surface
b) Simple harmonic oscillator
c) Particle on inclined plane (angle α)
d) Projectile in 2D

**Solutions:**
a) L = ½m(ẋ² + ẏ² + ż²) - mgz

b) L = ½mẋ² - ½kx²

c) L = ½mṡ² - mgs sin α (s = position along plane)

d) L = ½m(ẋ² + ẏ²) - mgy

---

### Problem A3: Euler-Lagrange Application
For L = ½m(ṙ² + r²θ̇²) - V(r):
a) Find ∂L/∂r, ∂L/∂ṙ, ∂L/∂θ, ∂L/∂θ̇
b) Write the E-L equations
c) Identify any conserved quantities

**Solutions:**
a) ∂L/∂r = mrθ̇² - dV/dr, ∂L/∂ṙ = mṙ
   ∂L/∂θ = 0, ∂L/∂θ̇ = mr²θ̇

b) mṙ̈ - mrθ̇² + dV/dr = 0
   d/dt(mr²θ̇) = 0

c) θ is cyclic → L_z = mr²θ̇ = const (angular momentum)
   If V = V(r) only and ∂L/∂t = 0 → E = T + V = const

---

### Problem A4: Constraint Forces
A bead slides on a frictionless circular wire of radius R in a vertical plane.
a) Write the Lagrangian using the angle θ
b) Find the equation of motion
c) Use Lagrange multiplier to find the normal force

**Solutions:**
a) L = ½mR²θ̇² + mgR cos θ

b) mR²θ̈ = -mgR sin θ → θ̈ = -(g/R) sin θ

c) In Cartesian: L' = ½m(ẋ²+ẏ²) + mgy + λ(x²+y²-R²)
   
   E-L: mẍ = 2λx, mÿ = mg + 2λy
   
   Constraint acceleration: ẍx + ÿy = -(ẋ² + ẏ²)
   
   Solving: λ = -m(v² + gy)/R² where v² = ẋ² + ẏ²
   
   Normal force: N = |2λ(x,y)| = m(v²/R + g cos θ)

---

## 🔬 Problem Set B: Applications

### Problem B1: Atwood Machine
Two masses m₁ and m₂ connected by string over pulley (massless, frictionless).
a) Choose generalized coordinate
b) Write Lagrangian
c) Find acceleration

**Solutions:**
a) x = position of m₁ (m₂ at constant - x)

b) L = ½(m₁ + m₂)ẋ² + m₁gx - m₂g(const - x)
   = ½(m₁ + m₂)ẋ² + (m₁ - m₂)gx + const

c) E-L: (m₁ + m₂)ẍ = (m₁ - m₂)g
   a = (m₁ - m₂)g/(m₁ + m₂)

---

### Problem B2: Bead on Rotating Hoop
A bead slides on a vertical circular hoop of radius R rotating about vertical axis with angular velocity Ω.
a) Write Lagrangian (use angle θ from bottom)
b) Find equilibrium positions
c) Analyze stability

**Solutions:**
a) Position: (R sin θ cos Ωt, R sin θ sin Ωt, R(1-cos θ))
   
   T = ½m(R²θ̇² + R²Ω² sin²θ)
   V = mgR(1 - cos θ)
   
   L = ½mR²θ̇² + ½mR²Ω² sin²θ - mgR(1 - cos θ)

b) E-L: mR²θ̈ = mR²Ω² sin θ cos θ - mgR sin θ
   
   Equilibrium: sin θ(Ω² cos θ - g/R) = 0
   
   Solutions: θ = 0 or cos θ = g/(RΩ²)
   
   Second solution exists only if Ω² > g/R

c) θ = 0 stable if Ω² < g/R, unstable if Ω² > g/R
   θ = arccos(g/RΩ²) stable when it exists

---

### Problem B3: Double Pendulum
For double pendulum with m₁ = m₂ = m, L₁ = L₂ = L:
a) Write kinetic and potential energies
b) Show the Lagrangian is:
   L = mL²θ̇₁² + ½mL²θ̇₂² + mL²cos(θ₁-θ₂)θ̇₁θ̇₂ + 2mgL cos θ₁ + mgL cos θ₂

**Solutions:**
a) T₁ = ½m(L²θ̇₁²)
   
   v₂² = L²θ̇₁² + L²θ̇₂² + 2L²θ̇₁θ̇₂ cos(θ₁-θ₂)
   T₂ = ½mv₂²
   
   V = -mgL cos θ₁ - mgL(cos θ₁ + cos θ₂)
   = -2mgL cos θ₁ - mgL cos θ₂

b) L = T₁ + T₂ - V (combine terms)

---

### Problem B4: Conservation Laws
For a particle in a central force F = -f(r)r̂:
a) Write the Lagrangian in spherical coordinates
b) Identify all cyclic coordinates
c) List all conserved quantities

**Solutions:**
a) L = ½m(ṙ² + r²θ̇² + r²sin²θ φ̇²) - V(r)
   where V'(r) = f(r)

b) φ is cyclic (∂L/∂φ = 0)
   If we work in a plane (θ = π/2), effectively θ is also cyclic

c) Conserved:
   - L_z = mr²sin²θ φ̇ (angular momentum, z-component)
   - L² = mr²(θ̇² + sin²θ φ̇²) (total angular momentum squared)
   - E = ½mṙ² + L²/(2mr²) + V(r) (energy)
   
   For Kepler: Also Runge-Lenz vector!

---

## 📊 Self-Assessment Rubric

| Topic | Score (1-5) | Notes |
|-------|-------------|-------|
| Generalized coordinates | /5 | |
| Degrees of freedom | /5 | |
| Lagrangian construction | /5 | |
| Euler-Lagrange equations | /5 | |
| Constraints & multipliers | /5 | |
| Conservation laws | /5 | |
| Symmetry connections | /5 | |
| Problem solving | /5 | |

**Total: /40**

- 35-40: Excellent! Ready for Lagrangian II
- 28-34: Good, review weak areas
- 20-27: Need more practice
- <20: Review week before continuing

---

## 🔮 Week 22 Preview: Lagrangian Mechanics II

### Topics:
1. **Noether's Theorem** — Symmetries and conservation laws
2. **Central Force Problem** — Kepler orbits
3. **Two-Body Problem** — Reduced mass
4. **Small Oscillations** — Normal modes
5. **Rigid Body Motion** — Introduction

### Key Preparations:
- Review matrix eigenvalue problems
- Practice with coupled differential equations
- Think about symmetry transformations

---

## ✅ Week 21 Completion Checklist

- [ ] Mastered generalized coordinates
- [ ] Can count DOF and identify constraints
- [ ] Derived Euler-Lagrange from variational principle
- [ ] Applied to pendulum, oscillator, central force
- [ ] Used Lagrange multipliers for constraint forces
- [ ] Connected symmetries to conservation
- [ ] Completed both problem sets
- [ ] Self-assessment score ≥ 28/40

---

## 🎉 Congratulations!

You've completed **Week 21: Lagrangian Mechanics I**!

This week laid the foundation for analytical mechanics:
- The Principle of Least Action is the deepest principle in physics
- Euler-Lagrange equations provide systematic equations of motion
- Symmetries reveal conservation laws
- This formalism leads directly to quantum mechanics!

**Next: Week 22 — Noether's Theorem and Advanced Applications**
