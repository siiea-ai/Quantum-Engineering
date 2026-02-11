# Day 40: Week 6 Problem Set — Partial Derivatives Mastery

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:00 PM | 3 hours | Problem Set Part I |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Problem Set Part II |
| Evening | 7:00 PM - 8:30 PM | 1.5 hours | Self-Assessment |

**Total Study Time: 7.5 hours**

---

## 📋 Instructions

This problem set tests your mastery of:
- Functions of several variables
- Partial derivatives and the chain rule
- Gradient and directional derivatives
- Tangent planes and linear approximation

Work without notes for first attempt. Show all work clearly.

---

# 📝 PART I: PARTIAL DERIVATIVES

## Section A: Computing Partial Derivatives (4 points each)

Find all first-order partial derivatives:

**A1.** $f(x, y) = x^3y^2 - 4xy^3 + 5x$

**A2.** $f(x, y) = e^{xy} \sin(x + y)$

**A3.** $f(x, y) = \ln(x^2 + y^2)$

**A4.** $f(x, y, z) = xyz + x^2z^3$

**A5.** $f(x, y) = \frac{x - y}{x + y}$

**A6.** $f(x, y) = \arctan\left(\frac{y}{x}\right)$

---

## Section B: Second-Order Partial Derivatives (5 points each)

Find all second-order partial derivatives and verify that $f_{xy} = f_{yx}$:

**B1.** $f(x, y) = x^4 - 3x^2y^2 + y^4$

**B2.** $f(x, y) = e^x \cos(y)$

**B3.** $f(x, y) = x\ln(xy)$

---

## Section C: Chain Rule (6 points each)

**C1.** If $z = x^2y + xy^2$, $x = s + t$, $y = st$, find $\frac{\partial z}{\partial s}$ and $\frac{\partial z}{\partial t}$.

**C2.** If $w = \sin(xyz)$, $x = t$, $y = t^2$, $z = t^3$, find $\frac{dw}{dt}$.

**C3.** If $z = f(x^2 - y^2)$ where f is differentiable, show that $y\frac{\partial z}{\partial x} + x\frac{\partial z}{\partial y} = 0$.

**C4.** The radius of a cone increases at 2 cm/s while height decreases at 3 cm/s. When r = 10 cm and h = 15 cm, find the rate of change of volume.

---

# 📝 PART II: GRADIENT AND DIRECTIONAL DERIVATIVES

## Section D: Gradient Computation (4 points each)

Find the gradient vector:

**D1.** $f(x, y) = x^2e^y + y^2e^x$ at $(0, 0)$

**D2.** $f(x, y) = \sqrt{x^2 + y^2}$ at $(3, 4)$

**D3.** $f(x, y, z) = x\sin(y) + y\sin(z) + z\sin(x)$ at $(\pi, \pi, \pi)$

---

## Section E: Directional Derivatives (6 points each)

**E1.** Find $D_\mathbf{u}f$ at $(2, 1)$ for $f(x, y) = x^2 - 3xy + y^2$ in direction $\mathbf{v} = \langle 3, 4 \rangle$.

**E2.** Find $D_\mathbf{u}f$ at $(1, 1, 1)$ for $f(x, y, z) = xyz$ in direction from $(1, 1, 1)$ toward $(2, 3, 3)$.

**E3.** Find the direction of maximum increase of $f(x, y) = xe^y$ at $(2, 0)$ and the maximum rate.

**E4.** At $(1, 2)$, in what direction(s) is the directional derivative of $f(x, y) = x^2 + y^2$ equal to 1?

---

## Section F: Gradient Properties (7 points each)

**F1.** Show that $\nabla f$ is perpendicular to the level curve $f(x, y) = x^2 + 4y^2 = 8$ at the point $(2, 1)$.

**F2.** Temperature is given by $T(x, y) = 100 - x^2 - 2y^2$. A bug at $(1, 1)$ wants to warm up as fast as possible. In what direction should it move? At what rate will the temperature increase?

**F3.** Find the equation of the tangent line to the curve $x^2 + y^2 = 25$ at $(3, 4)$ using the gradient.

---

# 📝 PART III: TANGENT PLANES AND LINEAR APPROXIMATION

## Section G: Tangent Planes (6 points each)

**G1.** Find the equation of the tangent plane to $z = x^2 + y^2$ at $(1, 2, 5)$.

**G2.** Find the equation of the tangent plane to $z = \ln(x + 2y)$ at $(1, 0, 0)$.

**G3.** Find the equation of the tangent plane to $z = e^{x-y}$ at $(1, 1, 1)$.

**G4.** At what point on $z = x^2 + y^2$ is the tangent plane parallel to $2x + 4y - z = 5$?

---

## Section H: Linear Approximation (6 points each)

**H1.** Use linear approximation to estimate $(2.01)^3(1.98)^2$.

**H2.** Use linear approximation to estimate $\sqrt{(3.02)^2 + (3.97)^2}$.

**H3.** Find the linearization of $f(x, y) = \sqrt{x + e^y}$ at $(3, 0)$ and use it to approximate $f(3.1, 0.1)$.

---

## Section I: Differentials and Error (6 points each)

**I1.** Find the differential $dz$ for $z = x^2\sin(y)$.

**I2.** Use differentials to estimate the change in $z = xy^2$ when $(x, y)$ changes from $(3, 2)$ to $(3.02, 1.97)$.

**I3.** The area of a triangle is $A = \frac{1}{2}ab\sin(C)$. If $a = 10 \pm 0.1$, $b = 8 \pm 0.1$, and $C = 30° \pm 1°$, estimate the error in the area.

---

# 📝 PART IV: COMPREHENSIVE PROBLEMS

## Section J: Mixed Applications (10 points each)

**J1.** The pressure, volume, and temperature of a gas satisfy $PV = nRT$.
(a) Find $\frac{\partial P}{\partial T}$, $\frac{\partial V}{\partial P}$, $\frac{\partial T}{\partial V}$.
(b) Show that $\frac{\partial P}{\partial T} \cdot \frac{\partial T}{\partial V} \cdot \frac{\partial V}{\partial P} = -1$.

**J2.** The surface $z = xy$ and the plane $x + y + z = 3$ intersect in a curve.
(a) Find the point(s) on this curve where the tangent plane to the surface is parallel to the given plane.
(b) Write the equation of that tangent plane.

**J3.** A hiker is on a mountain whose elevation is $h(x, y) = 1000 - 0.01x^2 - 0.02y^2$ meters.
(a) At position $(50, 30)$, in what direction should the hiker walk to ascend most steeply?
(b) What is the rate of ascent in that direction?
(c) In what direction should the hiker walk to stay at the same elevation?

---

# ✅ ANSWER KEY

## Section A
- A1: $f_x = 3x^2y^2 - 4y^3 + 5$, $f_y = 2x^3y - 12xy^2$
- A2: $f_x = ye^{xy}\sin(x+y) + e^{xy}\cos(x+y)$, $f_y = xe^{xy}\sin(x+y) + e^{xy}\cos(x+y)$
- A3: $f_x = \frac{2x}{x^2+y^2}$, $f_y = \frac{2y}{x^2+y^2}$
- A4: $f_x = yz + 2xz^3$, $f_y = xz$, $f_z = xy + 3x^2z^2$
- A5: $f_x = \frac{2y}{(x+y)^2}$, $f_y = \frac{-2x}{(x+y)^2}$
- A6: $f_x = \frac{-y}{x^2+y^2}$, $f_y = \frac{x}{x^2+y^2}$

## Section B
- B1: $f_{xx} = 12x^2 - 6y^2$, $f_{yy} = -6x^2 + 12y^2$, $f_{xy} = f_{yx} = -12xy$
- B2: $f_{xx} = e^x\cos y$, $f_{yy} = -e^x\cos y$, $f_{xy} = f_{yx} = -e^x\sin y$
- B3: $f_{xx} = -1/x$, $f_{yy} = -x/y^2$, $f_{xy} = f_{yx} = 1/y$

## Section C
- C1: $\frac{\partial z}{\partial s} = (2xy + y^2) + (x^2 + 2xy)t$, $\frac{\partial z}{\partial t} = (2xy + y^2) + (x^2 + 2xy)s$
- C2: $\frac{dw}{dt} = \cos(t^6) \cdot 6t^5$
- C3: Use chain rule with $u = x^2 - y^2$
- C4: $\frac{dV}{dt} = \pi(2rh\frac{dr}{dt} + r^2\frac{dh}{dt})/3 = \pi(600 - 300)/3 = 100\pi$ cm³/s

## Section D
- D1: $\nabla f(0,0) = \langle 0, 0 \rangle$
- D2: $\nabla f(3,4) = \langle 3/5, 4/5 \rangle$
- D3: $\nabla f(\pi,\pi,\pi) = \langle -\pi - 1, -\pi - 1, -\pi - 1 \rangle$

## Section E
- E1: $D_\mathbf{u}f = \langle 1, -4 \rangle \cdot \langle 3/5, 4/5 \rangle = -13/5$
- E2: $\mathbf{u} = \langle 1, 2, 2 \rangle/3$, $D_\mathbf{u}f = \langle 1, 1, 1 \rangle \cdot \mathbf{u} = 5/3$
- E3: Direction $\langle 1, 2 \rangle/\sqrt{5}$, rate $= \sqrt{5}$
- E4: $\mathbf{u} = \langle \cos\theta, \sin\theta \rangle$ where $2\cos\theta + 4\sin\theta = 1$

## Section F
- F1: $\nabla f(2,1) = \langle 4, 8 \rangle$, tangent to level curve is $\langle -8, 4 \rangle$, dot product = 0 ✓
- F2: Direction $\langle 2, 4 \rangle/\sqrt{20}$, rate $= 2\sqrt{5}$ degrees per unit distance
- F3: $3x + 4y = 25$

## Section G
- G1: $z = 2x + 4y - 5$
- G2: $z = x - 1 + 2y$
- G3: $z = x - y + 1$
- G4: $(1, 2, 5)$

## Section H
- H1: $\approx 31.52$
- H2: $\approx 5.008$
- H3: $L(x,y) = 2 + \frac{1}{4}(x-3) + \frac{1}{2}y$; $f(3.1, 0.1) \approx 2.075$

## Section I
- I1: $dz = 2x\sin(y)dx + x^2\cos(y)dy$
- I2: $dz \approx 0.04$
- I3: $\Delta A \approx 1.13$ square units

## Section J
- J1: (a) $\frac{\partial P}{\partial T} = \frac{nR}{V}$, etc. (b) Product = -1
- J2: (a) $(1, 1, 1)$; (b) $x + y - z = 1$
- J3: (a) $\langle 1, 1.2 \rangle/|\langle 1, 1.2 \rangle|$; (b) $\approx 1.56$ m per unit; (c) perpendicular to gradient

---

## 📊 Scoring Guide

| Section | Points | Your Score |
|---------|--------|------------|
| A (6 × 4) | 24 | |
| B (3 × 5) | 15 | |
| C (4 × 6) | 24 | |
| D (3 × 4) | 12 | |
| E (4 × 6) | 24 | |
| F (3 × 7) | 21 | |
| G (4 × 6) | 24 | |
| H (3 × 6) | 18 | |
| I (3 × 6) | 18 | |
| J (3 × 10) | 30 | |
| **TOTAL** | **210** | |

### Grade Scale
- 190-210: Excellent (A)
- 165-189: Good (B)
- 140-164: Satisfactory (C)
- Below 140: Review needed

---

## 🔜 Tomorrow: Computational Lab

Day 41 applies partial derivatives in Python with visualization.

---

*"Mathematics is not about numbers, equations, computations, or algorithms: it is about understanding."* — William Thurston
