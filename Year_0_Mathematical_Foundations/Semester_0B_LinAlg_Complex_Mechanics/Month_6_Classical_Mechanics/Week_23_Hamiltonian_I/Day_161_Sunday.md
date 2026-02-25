# Day 161: Week 23 Review — Hamiltonian Mechanics I

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Comprehensive Review |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Set |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Self-Assessment & Week 24 Preview |

**Total Study Time: 7 hours**

---

## Week 23 Summary

This week we established the foundations of **Hamiltonian Mechanics** — the geometric and algebraic framework that serves as the bridge between classical and quantum physics.

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 155 | Legendre Transform | H = Σpᵢq̇ᵢ - L, convex duality, Hamiltonian construction |
| 156 | Hamilton's Equations | q̇ = ∂H/∂p, ṗ = -∂H/∂q, first-order form, cyclic coordinates |
| 157 | Phase Space | 2n-dimensional manifold, symplectic structure, non-crossing theorem |
| 158 | Poisson Brackets | {f,g} definition, Jacobi identity, df/dt = {f,H} |
| 159 | Constants of Motion | {f,H} = 0, Noether's theorem, integrable systems |
| 160 | Computational Lab | Symplectic integrators, Poincaré sections |

---

## Core Concepts Review

### 1. The Legendre Transform

**Definition:** The Hamiltonian is obtained from the Lagrangian via Legendre transform:

$$\boxed{H(q, p, t) = \sum_i p_i \dot{q}_i - L(q, \dot{q}, t)}$$

where the canonical momentum is:
$$p_i = \frac{\partial L}{\partial \dot{q}_i}$$

**Key Property:** The Legendre transform maps velocity-space (q, q̇) to momentum-space (q, p).

**Physical Interpretation:** For scleronomic systems with natural Lagrangian L = T - V:
$$H = T + V = E \quad \text{(total energy)}$$

---

### 2. Hamilton's Equations

**The Canonical Equations of Motion:**

$$\boxed{\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}}$$

**Matrix Form:**
$$\dot{\mathbf{z}} = \mathbf{J}\nabla H, \quad \mathbf{J} = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$$

**Advantages over Lagrangian:**
- 2n first-order equations (vs n second-order)
- Symmetric treatment of q and p
- Natural framework for quantum mechanics
- Conservation laws from cyclic coordinates

---

### 3. Phase Space

**Definition:** The 2n-dimensional manifold with coordinates (q₁, ..., qₙ, p₁, ..., pₙ).

**Symplectic Structure:**
$$\omega = \sum_i dp_i \wedge dq_i$$

**Key Theorems:**
- **Non-Crossing:** Trajectories cannot intersect (determinism)
- **Liouville:** Phase space volume is preserved (incompressible flow)

**Fixed Points:**
- **Centers:** Pure imaginary eigenvalues (±iω), stable
- **Saddles:** Real eigenvalues (±λ), unstable

---

### 4. Poisson Brackets

**Definition:**
$$\boxed{\{f, g\} = \sum_i \left(\frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i}\right)}$$

**Fundamental Brackets:**
$$\{q_i, q_j\} = 0, \quad \{p_i, p_j\} = 0, \quad \{q_i, p_j\} = \delta_{ij}$$

**Properties:**
- Antisymmetry: {f, g} = -{g, f}
- Bilinearity
- Leibniz rule: {fg, h} = f{g,h} + {f,h}g
- Jacobi identity: {f,{g,h}} + cyclic = 0

**Time Evolution:**
$$\frac{df}{dt} = \{f, H\} + \frac{\partial f}{\partial t}$$

---

### 5. Constants of Motion

**Conservation Criterion:**
$$\boxed{\{f, H\} = 0 \quad \Leftrightarrow \quad f \text{ conserved}}$$

**Noether's Theorem:**
$$\text{Continuous Symmetry} \quad \Longleftrightarrow \quad \text{Conservation Law}$$

| Symmetry | Conserved Quantity |
|----------|-------------------|
| Time translation | Energy H |
| Space translation | Momentum p |
| Rotation | Angular momentum L |

**Integrability:** n constants in involution → Liouville integrable

---

### 6. The Classical-Quantum Bridge

**The Dirac Correspondence:**
$$\boxed{\{A, B\}_{\text{classical}} \quad \longleftrightarrow \quad \frac{1}{i\hbar}[\hat{A}, \hat{B}]_{\text{quantum}}}$$

**Canonical Commutation Relations:**
$$[q̂, p̂] = i\hbar \quad \leftarrow \quad \{q, p\} = 1$$

**Heisenberg Equation:**
$$\frac{d\hat{A}}{dt} = \frac{1}{i\hbar}[\hat{A}, \hat{H}] + \frac{\partial \hat{A}}{\partial t}$$

---

## Key Formulas Reference

### Hamiltonian Mechanics

| Formula | Meaning |
|---------|---------|
| H = Σpᵢq̇ᵢ - L | Legendre transform |
| q̇ = ∂H/∂p, ṗ = -∂H/∂q | Hamilton's equations |
| ż = J∇H | Symplectic form |
| {f, g} = Σ(∂f/∂q ∂g/∂p - ∂f/∂p ∂g/∂q) | Poisson bracket |
| df/dt = {f, H} + ∂f/∂t | Time evolution |
| {f, H} = 0 | Conservation criterion |

### Phase Space

| Concept | Formula |
|---------|---------|
| Dimension | 2n |
| Symplectic form | ω = Σdpᵢ∧dqᵢ |
| Volume preservation | det(Dφᵗ) = 1 |
| Uncertainty cell | ΔqΔp ≥ ℏ/2 |

### Angular Momentum

| Bracket | Value |
|---------|-------|
| {Lₓ, Lᵧ} | Lᵤ |
| {Lᵧ, Lᵤ} | Lₓ |
| {Lᵤ, Lₓ} | Lᵧ |
| {L², Lᵢ} | 0 |

---

## Problem Set A: Conceptual Questions

1. **Legendre Transform:** Explain why the Legendre transform is needed to go from Lagrangian to Hamiltonian mechanics. What information is preserved?

2. **Phase Space Dimension:** A system of N particles in 3D has what phase space dimension? Why is this relevant for statistical mechanics?

3. **Non-Crossing:** Why can't phase space trajectories cross for autonomous systems? What does this imply about predictability?

4. **Poisson Bracket Interpretation:** What is the physical meaning of {f, g} ≠ 0?

5. **Quantum Bridge:** Explain why {q, p} = 1 becoming [q̂, p̂] = iℏ leads to the uncertainty principle.

---

## Problem Set B: Calculations

### Problem 1: Harmonic Oscillator
For H = p²/(2m) + mω²x²/2:
a) Write Hamilton's equations
b) Verify {H, H} = 0
c) Calculate {x², H} and interpret the result
d) Find the action variable I

### Problem 2: Central Force
For H = pᵣ²/(2m) + L²/(2mr²) - k/r:
a) Identify all cyclic coordinates
b) Which momenta are conserved?
c) Show {L, H} = 0 using Poisson brackets

### Problem 3: Angular Momentum Algebra
a) Calculate {Lₓ, Lᵧ} explicitly from the definition
b) Verify the Jacobi identity for Lₓ, Lᵧ, Lᵤ
c) Show {L², Lᵤ} = 0

### Problem 4: Phase Portrait
For the simple pendulum H = p²/(2mL²) - mgL cos(θ):
a) Find all fixed points
b) Classify each as center or saddle
c) Find the separatrix energy and equation
d) Sketch the phase portrait

### Problem 5: Symplectic Integrator
Implement the Störmer-Verlet method for the harmonic oscillator:
a) Write the algorithm
b) Show it preserves area in phase space
c) Compare energy conservation with Euler method

---

## Problem Set B: Solutions

### Solution 1: Harmonic Oscillator

a) **Hamilton's equations:**
$$\dot{x} = \frac{\partial H}{\partial p} = \frac{p}{m}, \quad \dot{p} = -\frac{\partial H}{\partial x} = -m\omega^2 x$$

b) **Verify {H, H} = 0:**
By antisymmetry, {H, H} = -{H, H}, so {H, H} = 0 ✓

c) **Calculate {x², H}:**
$$\{x^2, H\} = \frac{\partial(x^2)}{\partial x}\frac{\partial H}{\partial p} - \frac{\partial(x^2)}{\partial p}\frac{\partial H}{\partial x} = 2x \cdot \frac{p}{m} - 0 = \frac{2xp}{m}$$

**Interpretation:** This is the rate of change of x²: d(x²)/dt = 2xẋ = 2x(p/m) = 2xp/m ✓

d) **Action variable:**
$$I = \frac{1}{2\pi}\oint p\,dx = \frac{\text{Area of ellipse}}{2\pi} = \frac{\pi \cdot x_{\max} \cdot p_{\max}}{2\pi} = \frac{E}{\omega}$$

---

### Solution 2: Central Force

a) **Cyclic coordinates:** θ (azimuthal angle) doesn't appear in H

b) **Conserved momenta:** pθ = L (angular momentum) is conserved since ∂H/∂θ = 0

c) **Show {L, H} = 0:** This was proven in Day 159. The key is that both kinetic and potential energy terms give zero bracket with L for central forces.

---

### Solution 3: Angular Momentum

a) **Calculate {Lₓ, Lᵧ}:**

Lₓ = ypᵤ - zpᵧ, Lᵧ = zpₓ - xpᵤ

$$\{L_x, L_y\} = \sum_i \left(\frac{\partial L_x}{\partial q_i}\frac{\partial L_y}{\partial p_i} - \frac{\partial L_x}{\partial p_i}\frac{\partial L_y}{\partial q_i}\right)$$

Computing term by term and collecting: {Lₓ, Lᵧ} = xpᵧ - ypₓ = Lᵤ ✓

b) **Jacobi identity:**
$$\{L_x, \{L_y, L_z\}\} + \{L_y, \{L_z, L_x\}\} + \{L_z, \{L_x, L_y\}\}$$
$$= \{L_x, L_x\} + \{L_y, L_y\} + \{L_z, L_z\} = 0 + 0 + 0 = 0 \quad ✓$$

c) **{L², Lᵤ} = 0:**
$$\{L^2, L_z\} = 2L_x\{L_x, L_z\} + 2L_y\{L_y, L_z\} + 2L_z\{L_z, L_z\}$$
$$= 2L_x(-L_y) + 2L_y(L_x) + 0 = 0 \quad ✓$$

---

### Solution 4: Phase Portrait

a) **Fixed points:** ∇H = 0
- ∂H/∂p = p/(mL²) = 0 → p = 0
- ∂H/∂θ = mgL sin(θ) = 0 → θ = 0, π

Fixed points: (θ, p) = (0, 0) and (π, 0)

b) **Classification:**
- (0, 0): Center (stable, hanging equilibrium)
- (π, 0): Saddle (unstable, inverted equilibrium)

c) **Separatrix:** E_sep = H(π, 0) = mgL
$$\frac{p^2}{2mL^2} - mgL\cos\theta = mgL \implies p = \pm 2mL\sqrt{gL}\cos(\theta/2)$$

d) Phase portrait shows closed curves (libration) around center, open curves (rotation) above separatrix.

---

## Self-Assessment Checklist

### Conceptual Understanding

- [ ] I can explain why Hamiltonian mechanics uses (q, p) instead of (q, q̇)
- [ ] I understand phase space as a geometric arena for dynamics
- [ ] I can interpret Poisson brackets as measuring "incompatibility"
- [ ] I understand why {f, H} = 0 implies f is conserved
- [ ] I can explain the Noether theorem symmetry-conservation duality
- [ ] I understand the classical-quantum correspondence

### Technical Skills

- [ ] I can perform Legendre transforms
- [ ] I can write and solve Hamilton's equations
- [ ] I can compute Poisson brackets for arbitrary functions
- [ ] I can identify cyclic coordinates and conserved quantities
- [ ] I can construct phase portraits
- [ ] I can implement symplectic integrators

### Quantum Connections

- [ ] I can state the Dirac correspondence
- [ ] I understand why {q, p} = 1 becomes [q̂, p̂] = iℏ
- [ ] I can relate classical conservation to quantum good numbers
- [ ] I understand the Wigner function as quantum phase space

---

## Week 24 Preview: Hamiltonian Mechanics II

Next week we complete classical mechanics with advanced Hamiltonian topics:

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 162 | Canonical Transformations | Generating functions, symplectomorphisms |
| 163 | Liouville's Theorem | Phase space volume conservation, statistical mechanics |
| 164 | Action-Angle Variables | Integrable systems, tori, adiabatic invariants |
| 165 | Hamilton-Jacobi Equation | S(q,t), complete integrals, separation of variables |
| 166 | Introduction to Chaos | Sensitivity, Lyapunov exponents, KAM theorem |
| 167 | Computational Lab | Advanced simulations, chaos visualization |
| 168 | Year 0 Final Review | Comprehensive assessment, Year 1 preview |

**Key Themes:**
- Canonical transformations preserve Hamiltonian structure
- Hamilton-Jacobi theory connects to quantum mechanics (WKB, path integrals)
- Chaos emerges from non-integrability
- Action variables become quantum numbers in semiclassical limit

---

## Reflection Questions

1. What concept from this week surprised you the most?

2. How does the Poisson bracket formalism simplify the statement of conservation laws?

3. Why is the classical-quantum correspondence {,} → [,]/(iℏ) considered one of the most profound results in physics?

4. What aspects of Hamiltonian mechanics seem most relevant for quantum mechanics?

5. What questions do you still have about this week's material?

---

## Daily Checklist

- [ ] Complete Problem Set A (conceptual)
- [ ] Complete Problem Set B (calculations)
- [ ] Review all key formulas
- [ ] Complete self-assessment checklist honestly
- [ ] Identify weak areas for review
- [ ] Preview Week 24 topics
- [ ] Rest and consolidate!
