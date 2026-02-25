# Day 159: Constants of Motion and Integrable Systems

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Conservation Laws and Integrability |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Define constants of motion and apply the criterion {f, H} = 0
2. Apply Poisson's theorem to generate new conserved quantities
3. State and prove Noether's theorem connecting symmetries to conservation laws
4. Define Liouville integrability and explain the Liouville-Arnold theorem
5. Identify cyclic coordinates and perform reduction of degrees of freedom
6. Connect classical conservation laws to quantum good quantum numbers

---

## Core Content

### 1. Constants of Motion: Definitions

A **constant of motion** (or **first integral**) is a function f(q, p, t) whose value remains constant along trajectories of the system.

**Mathematical Definition:**
$$\frac{df}{dt} = 0 \quad \text{along solutions}$$

Using the result from Day 158:

$$\boxed{\frac{df}{dt} = \{f, H\} + \frac{\partial f}{\partial t} = 0}$$

**Time-Independent Case:** If f has no explicit time dependence (∂f/∂t = 0), the conservation criterion simplifies to:

$$\boxed{\{f, H\} = 0 \quad \Leftrightarrow \quad f \text{ is conserved}}$$

**Physical Interpretation:** The Poisson bracket {f, H} measures the rate of change of f along the Hamiltonian flow. Zero bracket means f is invariant under time evolution.

---

### 2. Functionally Independent Constants

**Definition:** Constants F₁, ..., Fₖ are **functionally independent** if their gradients ∇F₁, ..., ∇Fₖ are linearly independent almost everywhere.

**Why Independence Matters:** If F₂ = g(F₁) for some function g, then F₂ provides no new information beyond F₁.

**Maximum Number:** A system with n degrees of freedom (2n-dimensional phase space) can have at most **2n - 1** functionally independent constants. The Hamiltonian flow uses one dimension, leaving 2n - 1 constraints possible.

**Example:** For a 1D system (n = 1), at most one independent constant exists — the energy H itself.

---

### 3. Poisson's Theorem

**Theorem (Poisson, 1809):** If f and g are both constants of motion, then {f, g} is also a constant of motion.

$$\boxed{\{f, H\} = 0 \text{ and } \{g, H\} = 0 \quad \Rightarrow \quad \{\{f, g\}, H\} = 0}$$

**Proof:** Using the Jacobi identity with h = H:

$$\{f, \{g, H\}\} + \{g, \{H, f\}\} + \{H, \{f, g\}\} = 0$$

Since {g, H} = 0 and {f, H} = 0:
$$0 + 0 + \{H, \{f, g\}\} = 0$$

By antisymmetry: {{f, g}, H} = 0. ∎

**Application:** This provides a systematic method to discover **new** constants from known ones!

**Example — Angular Momentum:** For a central force:
- {Lₓ, H} = 0 and {Lᵧ, H} = 0
- By Poisson's theorem: {Lₓ, Lᵧ} = Lᵤ is also conserved
- The algebra closes: the so(3) Lie algebra

**Limitation:** The result may be:
1. Trivial (a constant number like {q, p} = 1)
2. Functionally dependent on existing constants

---

### 4. Noether's Theorem

**Theorem (Emmy Noether, 1918):** Every continuous symmetry of the action has a corresponding conservation law. Conversely, every conservation law corresponds to a continuous symmetry.

$$\boxed{\text{Continuous Symmetry} \quad \Longleftrightarrow \quad \text{Conservation Law}}$$

**Mathematical Statement:** If the Lagrangian is invariant (up to a total derivative) under:
$$q_i \to q_i + \epsilon \delta q_i$$

Then the **Noether charge** is conserved:
$$Q = \sum_i \frac{\partial L}{\partial \dot{q}_i} \delta q_i - \Lambda$$

where Λ accounts for any total derivative change in L.

---

### 5. Fundamental Symmetry-Conservation Pairs

| Symmetry | Transformation | Conserved Quantity |
|----------|----------------|-------------------|
| **Time translation** | t → t + ε | Energy H |
| **Space translation** | x → x + ε | Linear momentum p |
| **Rotation** | θ → θ + ε | Angular momentum L |
| **Galilean boost** | v → v + εu | Center of mass motion |
| **U(1) gauge** | ψ → e^{iα}ψ | Electric charge |

---

### 6. Generator Viewpoint

**Profound Insight:** In Hamiltonian mechanics, a conserved quantity G **generates** its own symmetry transformation!

For any observable f, the infinitesimal change under the symmetry is:
$$\delta f = \epsilon \{f, G\}$$

**Examples:**
- **Momentum p generates translation:** {x, p} = 1 → δx = ε
- **Angular momentum Lᵤ generates rotation:** {x, Lᵤ} = -y, {y, Lᵤ} = x
- **Hamiltonian H generates time evolution:** {f, H} = df/dt

**The Duality:**
$$\boxed{\{G, H\} = 0 \quad \Longleftrightarrow \quad G \text{ generates a symmetry of } H}$$

---

### 7. Integrable Systems

**Definition (Liouville Integrability):** A Hamiltonian system with n degrees of freedom is **completely integrable** if there exist n functionally independent constants F₁ = H, F₂, ..., Fₙ that are **in involution**:

$$\boxed{\{F_i, F_j\} = 0 \quad \text{for all } i, j = 1, \ldots, n}$$

**Constants in Involution:** Constants that pairwise Poisson-commute. This ensures their level sets intersect properly.

**Why n Constants?**
- Phase space has 2n dimensions
- Each independent constant in involution removes 2 dimensions (the constant's level set plus its conjugate direction)
- n such constraints reduce motion to an n-dimensional surface

---

### 8. The Liouville-Arnold Theorem

**Theorem:** For an integrable system with n constants in involution F₁, ..., Fₙ:

1. **Topology:** The level set Mₐ = {z : Fᵢ(z) = cᵢ for all i}, if compact and connected, is diffeomorphic to an **n-torus**:
$$M_c \cong T^n = S^1 \times S^1 \times \cdots \times S^1$$

2. **Action-Angle Variables:** There exist canonical coordinates (φ₁, ..., φₙ, I₁, ..., Iₙ) such that:
   - φᵢ ∈ [0, 2π) are **angle variables**
   - Iᵢ are **action variables** (constant on each torus)
   - H = H(I₁, ..., Iₙ) depends only on actions

3. **Linear Motion:** The equations of motion become trivial:
$$\dot{\phi}_i = \frac{\partial H}{\partial I_i} = \omega_i(I), \quad \dot{I}_i = 0$$

Motion on the torus is **quasi-periodic** with constant frequencies ω₁, ..., ωₙ.

---

### 9. Action-Angle Variables

**Action Variable:** For periodic motion:
$$\boxed{I = \frac{1}{2\pi} \oint p \, dq}$$

This equals 1/(2π) times the area enclosed by the orbit in phase space.

**Angle Variable:**
$$\phi = \omega t + \phi_0, \quad \omega = \frac{\partial H}{\partial I}$$

**Properties:**
- {φ, I} = 1 (canonical pair)
- I is an **adiabatic invariant** (preserved under slow parameter changes)
- Motion is linear in φ, constant in I

---

### 10. Examples of Integrable Systems

**Simple Harmonic Oscillator (n = 1):**
- Single constant: H = p²/(2m) + mω²q²/2
- Action: I = H/ω = E/ω
- Frequency: ω = ∂H/∂I = constant (isochronous)

**Kepler Problem (n = 3):**
- Three constants in involution: H, L², Lᵤ
- Additional "hidden" constant: Laplace-Runge-Lenz vector A
- Actually **superintegrable** (more constants than DOF)
- All bounded orbits are closed ellipses

**Two-Body Problem:**
- Reduce to center of mass + relative motion
- Relative motion is Kepler problem
- Completely integrable

---

### 11. Non-Integrable Systems

**Generic Behavior:** Most Hamiltonian systems are **not** integrable. They exhibit:
- Chaotic trajectories
- Sensitive dependence on initial conditions
- Positive Lyapunov exponents
- Mixing in phase space

**Famous Non-Integrable Systems:**
- **Three-body problem** (proved by Poincaré, 1889)
- **Double pendulum** (chaotic for high energies)
- **Hénon-Heiles potential**

**KAM Theorem:** Under small perturbations, **most** invariant tori survive (with "sufficiently irrational" frequencies). The transition to chaos is gradual, not sudden.

---

### 12. Cyclic Coordinates and Reduction

**Definition:** A coordinate qᵢ is **cyclic** (or **ignorable**) if:
$$\frac{\partial H}{\partial q_i} = 0$$

**Consequence:** By Hamilton's equation:
$$\dot{p}_i = -\frac{\partial H}{\partial q_i} = 0$$

So pᵢ = constant! The conjugate momentum to a cyclic coordinate is always conserved.

**Reduction:** If qᵢ is cyclic with pᵢ = α (constant):
1. Replace pᵢ by α everywhere in H
2. Solve the reduced (2n-2)-dimensional system
3. Recover qᵢ by quadrature at the end

**Example — Central Force:** Azimuthal angle φ is cyclic (rotation symmetry), so Lᵤ = mr²φ̇ is conserved. Problem reduces from 6D → 4D → effectively 2D.

---

### 13. The Laplace-Runge-Lenz Vector

For the Kepler problem (V = -k/r), there's a **hidden conserved quantity**:

$$\boxed{\mathbf{A} = \mathbf{p} \times \mathbf{L} - mk\hat{\mathbf{r}}}$$

**Properties:**
- {A, H} = 0 (conserved)
- A points toward perihelion (closest approach)
- |A| = mke, where e is orbital eccentricity
- A · L = 0 (A lies in orbital plane)

**SO(4) Hidden Symmetry:** The six quantities (L₁, L₂, L₃, A₁, A₂, A₃) generate the **so(4) Lie algebra** (for bound states). This explains the "accidental" n² degeneracy of hydrogen atom energy levels!

---

## Quantum Mechanics Connection

### Conservation Laws and Good Quantum Numbers

**Classical:** {f, H} = 0 means f is conserved

**Quantum:** [f̂, Ĥ] = 0 means:
1. f̂ and Ĥ have **simultaneous eigenstates**
2. Eigenvalues of f̂ are **good quantum numbers**
3. f̂ is constant in time (Heisenberg picture)

$$\boxed{\{f, H\} = 0 \quad \longleftrightarrow \quad [\hat{f}, \hat{H}] = 0}$$

---

### Complete Sets of Commuting Observables

**Definition:** Operators {Â, B̂, Ĉ, ...} form a **CSCO** if:
1. All mutually commute: [Â, B̂] = [B̂, Ĉ] = ... = 0
2. Simultaneous eigenstates |a, b, c, ...⟩ are uniquely labeled

**Hydrogen Atom CSCO:**
- {Ĥ, L̂², L̂ᵤ, Ŝᵤ}
- Good quantum numbers: (n, ℓ, mₗ, mₛ)
- States: |n, ℓ, mₗ, mₛ⟩

---

### Symmetry and Degeneracy

**Theorem:** If [Ĥ, Ŝ] = 0 for symmetry operator Ŝ, and |ψ⟩ is an eigenstate of Ĥ, then Ŝ|ψ⟩ is also an eigenstate with the **same energy**.

**Consequence:** Symmetries lead to **degeneracy**.

**Hydrogen SO(4) Symmetry:**
- [Ĥ, L̂] = 0 (angular momentum conserved)
- [Ĥ, Â] = 0 (quantum Runge-Lenz conserved)
- Together: SO(4) symmetry for bound states
- **Explains:** All ℓ ∈ {0, 1, ..., n-1} have same energy
- **Degeneracy:** n² instead of just (2ℓ+1)

---

### Selection Rules from Conservation Laws

If [Q̂, Ĥ] = 0 (Q conserved), transitions |i⟩ → |f⟩ require:
$$\langle f | \hat{V} | i \rangle \neq 0 \quad \Rightarrow \quad \text{consistent with conservation}$$

**Examples:**
- Angular momentum: Δmₗ = 0, ±1 (dipole transitions)
- Parity: even ↔ odd only (for parity-conserving interactions)
- Charge: Q_initial = Q_final

---

## Worked Examples

### Example 1: Verify Energy Conservation for SHO

**Problem:** Show {H, H} = 0 for the harmonic oscillator.

**Solution:**

H = p²/(2m) + mω²q²/2

By antisymmetry of the Poisson bracket:
$$\{H, H\} = -\{H, H\}$$

The only number equal to its own negative is zero:
$$\{H, H\} = 0 \quad ✓$$

**Alternative:** Direct calculation shows all terms cancel pairwise due to the symmetry of H in its own derivatives.

---

### Example 2: Central Force Angular Momentum Conservation

**Problem:** Prove {Lᵤ, H} = 0 for H = p²/(2m) + V(r).

**Solution:**

**Lᵤ = xpᵧ - ypₓ**, H = (pₓ² + pᵧ² + pᵤ²)/(2m) + V(r)

**Kinetic energy contribution:**
$$\{L_z, T\} = \{xp_y - yp_x, \frac{p_x^2 + p_y^2 + p_z^2}{2m}\}$$

Using {Lᵤ, pₓ} = pᵧ and {Lᵤ, pᵧ} = -pₓ:
$$\{L_z, T\} = \frac{1}{m}(p_x p_y - p_y p_x) = 0$$

**Potential energy contribution:**
For V = V(r) with r = √(x² + y² + z²):
$$\{L_z, V\} = \frac{\partial L_z}{\partial x}\frac{\partial V}{\partial p_x} - \frac{\partial L_z}{\partial p_x}\frac{\partial V}{\partial x} + (y \text{ terms})$$

Since ∂V/∂pᵢ = 0 and using ∂V/∂x = V'(r)·x/r:
$$\{L_z, V\} = -(-y)\frac{x V'(r)}{r} - (-x)\frac{y V'(r)}{r} = \frac{V'(r)}{r}(xy - xy) = 0$$

**Therefore:** {Lᵤ, H} = {Lᵤ, T} + {Lᵤ, V} = 0 + 0 = 0 ✓

---

### Example 3: Action Variable for SHO

**Problem:** Calculate the action variable I for the harmonic oscillator with energy E.

**Solution:**

The phase space orbit is an ellipse: p²/(2mE) + mω²q²/(2E) = 1

Semi-axes: q_max = √(2E/(mω²)), p_max = √(2mE)

**Action integral:**
$$I = \frac{1}{2\pi} \oint p \, dq$$

Parameterize: q = q_max cos(θ), so dq = -q_max sin(θ) dθ

From energy conservation: p = √(2m(E - mω²q²/2)) = p_max |sin(θ)|

$$I = \frac{1}{2\pi} \int_0^{2\pi} p_{\max}|\sin\theta| \cdot q_{\max}|\sin\theta| \, d\theta$$

$$= \frac{p_{\max} q_{\max}}{2\pi} \int_0^{2\pi} \sin^2\theta \, d\theta = \frac{p_{\max} q_{\max}}{2\pi} \cdot \pi$$

$$= \frac{1}{2}\sqrt{2mE} \cdot \sqrt{\frac{2E}{m\omega^2}} = \frac{E}{\omega}$$

$$\boxed{I = \frac{E}{\omega}}$$

**Verify:** H = ωI, so ∂H/∂I = ω, confirming the frequency.

---

## Practice Problems

### Level 1: Direct Application

1. **Energy Conservation:** For H = p²/(2m) - mgy, verify {H, H} = 0 and identify what this implies.

2. **Cyclic Coordinate:** For H = pᵣ²/(2m) + L²/(2mr²) + V(r), identify the cyclic coordinate and its conserved conjugate momentum.

3. **Linear Momentum:** Show that {pₓ, H} = 0 for a free particle H = p²/(2m).

### Level 2: Intermediate

4. **Two Constants:** Given {E, H} = 0 and {L, H} = 0, use Poisson's theorem to show that any f(E, L) is also conserved.

5. **Reduction:** For a particle in a central force, reduce the 3D problem to an effective 1D problem by using conservation of angular momentum.

6. **Action Calculation:** Calculate the action variable for the simple pendulum in the small-angle approximation.

### Level 3: Challenging

7. **Laplace-Runge-Lenz:** Verify that A = p × L - mk r̂ satisfies {A, H} = 0 for the Kepler problem.

8. **Integrability Count:** A system has n = 2 degrees of freedom and two constants H and F with {H, F} = 0. Is it integrable? What if {H, F} ≠ 0?

9. **Quantum Connection:** The hydrogen atom has quantum numbers (n, ℓ, mₗ). Identify the corresponding classical conserved quantities and verify they form a CSCO.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy import symbols, diff, simplify, sqrt, cos, sin

def constants_of_motion_demo():
    """
    Demonstrate conservation laws and integrability.
    """

    print("=" * 70)
    print("CONSTANTS OF MOTION AND INTEGRABLE SYSTEMS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # =========================================
    # 1. Noether's Theorem: Symmetry → Conservation
    # =========================================

    # Time translation → Energy (SHO)
    ax = axes[0, 0]

    def sho_eom(state, t, omega=1.0):
        q, p = state
        return [p, -omega**2 * q]

    t = np.linspace(0, 6*np.pi, 1000)
    omega = 1.0

    for E0 in [0.5, 1.0, 2.0]:
        q0 = np.sqrt(2*E0/omega**2)
        sol = odeint(sho_eom, [q0, 0], t)
        q, p = sol[:, 0], sol[:, 1]
        E = 0.5*p**2 + 0.5*omega**2*q**2
        ax.plot(t, E, lw=2, label=f'E₀ = {E0}')

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Energy H', fontsize=11)
    ax.set_title('Energy Conservation (SHO)\nTime Translation Symmetry', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Space translation → Momentum (free particles)
    ax = axes[0, 1]

    def free_particles(state, t):
        x1, x2, p1, p2 = state
        return [p1, p2, 0, 0]

    sol = odeint(free_particles, [0, 3, 1.5, -0.8], t)
    p_total = sol[:, 2] + sol[:, 3]

    ax.plot(t, sol[:, 2], 'b-', lw=2, label='p₁')
    ax.plot(t, sol[:, 3], 'r-', lw=2, label='p₂')
    ax.plot(t, p_total, 'k--', lw=3, label='P_total')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Momentum', fontsize=11)
    ax.set_title('Momentum Conservation\nSpace Translation Symmetry', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotation → Angular Momentum (central force)
    ax = axes[0, 2]

    def central_force(state, t, k=1.0):
        x, y, px, py = state
        r = np.sqrt(x**2 + y**2)
        return [px, py, -k*x/r**3, -k*y/r**3]

    t_kepler = np.linspace(0, 30, 3000)
    sol_kepler = odeint(central_force, [1.0, 0, 0, 0.8], t_kepler)
    x, y = sol_kepler[:, 0], sol_kepler[:, 1]
    px, py = sol_kepler[:, 2], sol_kepler[:, 3]
    L = x*py - y*px

    ax.plot(t_kepler, L, 'b-', lw=2)
    ax.axhline(L[0], color='r', linestyle='--', label=f'L = {L[0]:.4f}')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Angular Momentum L', fontsize=11)
    ax.set_title('Angular Momentum Conservation\nRotation Symmetry', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    print(f"Angular momentum variation: ΔL/L = {(L.max()-L.min())/abs(L.mean()):.2e}")

    # =========================================
    # 2. Integrable vs Non-Integrable
    # =========================================

    # Integrable: 2D Harmonic Oscillator (regular orbits)
    ax = axes[1, 0]

    def sho_2d(state, t):
        x, y, px, py = state
        return [px, py, -x, -y]

    # Multiple initial conditions - all regular
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 6))
    for i, phase in enumerate(np.linspace(0, np.pi, 6)):
        E = 1.5
        x0 = np.sqrt(2*E) * np.cos(phase)
        px0 = np.sqrt(2*E) * np.sin(phase)
        sol = odeint(sho_2d, [x0, 0.3, px0, 0.2], np.linspace(0, 20, 2000))
        ax.plot(sol[:, 0], sol[:, 2], '-', color=colors[i], lw=0.7, alpha=0.8)

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('pₓ', fontsize=11)
    ax.set_title('Integrable: 2D SHO\nRegular orbits on tori', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Non-integrable: Hénon-Heiles (chaotic)
    ax = axes[1, 1]

    def henon_heiles(state, t, lambda_param=1.0):
        x, y, px, py = state
        dxdt = px
        dydt = py
        dpxdt = -x - 2*lambda_param*x*y
        dpydt = -y - lambda_param*(x**2 - y**2)
        return [dxdt, dydt, dpxdt, dpydt]

    # Higher energy → more chaos
    t_hh = np.linspace(0, 200, 20000)

    # Two nearby initial conditions
    sol1 = odeint(henon_heiles, [0.0, 0.3, 0.3, 0.0], t_hh)
    sol2 = odeint(henon_heiles, [0.0, 0.301, 0.3, 0.0], t_hh)  # Tiny perturbation

    ax.plot(sol1[:, 0], sol1[:, 2], 'b-', lw=0.3, alpha=0.6, label='IC 1')
    ax.plot(sol2[:, 0], sol2[:, 2], 'r-', lw=0.3, alpha=0.6, label='IC 2')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('pₓ', fontsize=11)
    ax.set_title('Non-Integrable: Hénon-Heiles\nChaotic trajectories diverge', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # =========================================
    # 3. Action-Angle Variables
    # =========================================
    ax = axes[1, 2]

    # SHO: Action I = E/ω, Angle φ increases linearly
    omega = 1.0
    t_aa = np.linspace(0, 4*np.pi, 500)

    energies = [0.5, 1.0, 1.5, 2.0]
    colors_aa = plt.cm.plasma(np.linspace(0.2, 0.8, len(energies)))

    for E, c in zip(energies, colors_aa):
        I = E / omega  # Action
        phi = omega * t_aa  # Angle (linear increase)
        phi_wrapped = np.mod(phi, 2*np.pi)

        # Plot in action-angle space (cylinder projection)
        ax.scatter(phi_wrapped, np.full_like(phi_wrapped, I), s=1, c=[c], alpha=0.5)

    ax.set_xlabel('φ (angle)', fontsize=11)
    ax.set_ylabel('I (action)', fontsize=11)
    ax.set_title('Action-Angle Variables\nMotion linear in φ, I = const', fontsize=11)
    ax.set_xlim(0, 2*np.pi)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('constants_of_motion.png', dpi=150)
    plt.show()


def laplace_runge_lenz_demo():
    """
    Demonstrate the Laplace-Runge-Lenz vector conservation.
    """

    print("\n" + "=" * 70)
    print("LAPLACE-RUNGE-LENZ VECTOR: HIDDEN SYMMETRY IN KEPLER PROBLEM")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def kepler_2d(state, t, k=1.0, m=1.0):
        x, y, px, py = state
        r = np.sqrt(x**2 + y**2)
        return [px/m, py/m, -k*x/r**3, -k*y/r**3]

    # Elliptical orbit
    k, m = 1.0, 1.0
    state0 = [1.0, 0, 0, 0.7]  # Elliptical

    t = np.linspace(0, 50, 5000)
    sol = odeint(kepler_2d, state0, t)

    x, y = sol[:, 0], sol[:, 1]
    px, py = sol[:, 2], sol[:, 3]
    r = np.sqrt(x**2 + y**2)

    # Angular momentum
    L = x*py - y*px

    # Laplace-Runge-Lenz vector: A = p × L - mk r̂
    # In 2D: Ax = py*L/m - k*x/r, Ay = -px*L/m - k*y/r
    Ax = py*L/m - k*x/r
    Ay = -px*L/m - k*y/r
    A_mag = np.sqrt(Ax**2 + Ay**2)

    # Energy
    E = (px**2 + py**2)/(2*m) - k/r

    # Plot orbit
    ax = axes[0]
    ax.plot(x, y, 'b-', lw=1)
    ax.scatter([0], [0], c='orange', s=300, marker='*', zorder=5, label='Sun')

    # Plot A vector at several points
    stride = len(t) // 8
    for i in range(0, len(t), stride):
        scale = 0.3
        ax.arrow(x[i], y[i], Ax[i]*scale, Ay[i]*scale,
                head_width=0.03, head_length=0.015, fc='red', ec='red')

    # Draw A direction (constant!)
    A_dir = np.array([Ax.mean(), Ay.mean()])
    A_dir = A_dir / np.linalg.norm(A_dir) * 0.8
    ax.arrow(0, 0, A_dir[0], A_dir[1], head_width=0.05, head_length=0.03,
             fc='darkred', ec='darkred', lw=2, label='A direction')

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title('Kepler Orbit with Runge-Lenz Vector\nA points toward perihelion (constant!)', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Conservation verification
    ax = axes[1]
    ax.plot(t, L, 'b-', lw=2, label='L (angular momentum)')
    ax.plot(t, A_mag, 'r-', lw=2, label='|A| (Runge-Lenz)')
    ax.plot(t, E, 'g-', lw=2, label='E (energy)')

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Conserved Quantities', fontsize=11)
    ax.set_title('All Three Constants Conserved\nL, |A|, and E are constant', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    print(f"L variation: ΔL/L = {(L.max()-L.min())/abs(L.mean()):.2e}")
    print(f"|A| variation: Δ|A|/|A| = {(A_mag.max()-A_mag.min())/abs(A_mag.mean()):.2e}")
    print(f"E variation: ΔE/|E| = {(E.max()-E.min())/abs(E.mean()):.2e}")

    # A direction constancy
    ax = axes[2]
    A_angle = np.arctan2(Ay, Ax)

    ax.plot(t, np.degrees(A_angle), 'purple', lw=2)
    ax.axhline(np.degrees(A_angle.mean()), color='k', linestyle='--',
               label=f'Mean = {np.degrees(A_angle.mean()):.2f}°')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('A direction (degrees)', fontsize=11)
    ax.set_title('A Vector Direction (Constant!)\nPoints toward perihelion', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('runge_lenz_vector.png', dpi=150)
    plt.show()


def noether_symmetry_visual():
    """
    Visualize the symmetry-conservation correspondence.
    """

    print("\n" + "=" * 70)
    print("NOETHER'S THEOREM: SYMMETRY ↔ CONSERVATION")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create visual diagram
    ax.text(0.5, 0.95, "NOETHER'S THEOREM (1918)", fontsize=18,
            ha='center', fontweight='bold', transform=ax.transAxes)

    ax.text(0.5, 0.85, "Every continuous symmetry ↔ Conservation law",
            fontsize=14, ha='center', transform=ax.transAxes,
            style='italic', color='darkblue')

    # Draw table
    headers = ["SYMMETRY", "TRANSFORMATION", "CONSERVED QUANTITY", "GENERATOR"]
    data = [
        ["Time translation", "t → t + ε", "Energy H", "H generates time evolution"],
        ["Space translation", "x → x + ε", "Momentum p", "p generates translations"],
        ["Rotation", "θ → θ + ε", "Angular momentum L", "L generates rotations"],
        ["Galilean boost", "v → v + εu", "CoM motion", "Boost generator"],
        ["U(1) gauge", "ψ → e^{iα}ψ", "Electric charge Q", "Q generates phase"],
    ]

    y_start = 0.72
    y_step = 0.10

    # Headers
    x_positions = [0.08, 0.30, 0.55, 0.78]
    for x, header in zip(x_positions, headers):
        ax.text(x, y_start + 0.05, header, fontsize=11, fontweight='bold',
               transform=ax.transAxes, ha='left')

    # Horizontal line
    ax.axhline(y=y_start + 0.02, xmin=0.05, xmax=0.95, color='black', lw=1,
              transform=ax.transAxes)

    # Data rows
    colors = ['#e6f2ff', '#fff2e6', '#e6ffe6', '#ffe6e6', '#f2e6ff']
    for i, (row, color) in enumerate(zip(data, colors)):
        y = y_start - i * y_step
        # Background rectangle
        rect = plt.Rectangle((0.05, y - 0.04), 0.9, 0.08, transform=ax.transAxes,
                             facecolor=color, edgecolor='gray', alpha=0.5)
        ax.add_patch(rect)
        # Data
        for x, cell in zip(x_positions, row):
            ax.text(x, y, cell, fontsize=10, transform=ax.transAxes, ha='left', va='center')

    # Key insight box
    ax.text(0.5, 0.18, "THE DEEP CONNECTION", fontsize=14,
            ha='center', fontweight='bold', transform=ax.transAxes)

    insight = """In Hamiltonian mechanics, a conserved quantity G:
    1. Generates its own symmetry: δf = ε{f, G}
    2. Commutes with H: {G, H} = 0

    Conservation ← {G, H} = 0 → Symmetry"""

    ax.text(0.5, 0.08, insight, fontsize=11, ha='center', transform=ax.transAxes,
           family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow',
                                         edgecolor='orange', alpha=0.8))

    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('noether_theorem.png', dpi=150)
    plt.show()


def quantum_connection_demo():
    """
    Demonstrate classical → quantum conservation correspondence.
    """

    print("\n" + "=" * 70)
    print("CLASSICAL → QUANTUM: CONSERVATION AND GOOD QUANTUM NUMBERS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Correspondence table
    ax = axes[0]

    ax.text(0.5, 0.95, "CLASSICAL → QUANTUM CORRESPONDENCE", fontsize=14,
            ha='center', fontweight='bold', transform=ax.transAxes)

    classical = [
        "{f, H} = 0",
        "f is constant of motion",
        "Poisson bracket",
        "Phase space point",
        "Constants in involution"
    ]

    quantum = [
        "[f̂, Ĥ] = 0",
        "f̂ has good quantum numbers",
        "Commutator / iℏ",
        "State vector |ψ⟩",
        "Complete set (CSCO)"
    ]

    y_pos = 0.80
    for c, q in zip(classical, quantum):
        ax.text(0.15, y_pos, c, fontsize=11, transform=ax.transAxes, ha='left',
               family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue'))
        ax.text(0.50, y_pos, "→", fontsize=16, transform=ax.transAxes, ha='center')
        ax.text(0.60, y_pos, q, fontsize=11, transform=ax.transAxes, ha='left',
               family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen'))
        y_pos -= 0.15

    ax.text(0.5, 0.08, "Conservation laws become quantum numbers!",
            fontsize=12, ha='center', transform=ax.transAxes,
            style='italic', color='purple')

    ax.axis('off')

    # Right: Hydrogen atom example
    ax = axes[1]

    ax.text(0.5, 0.95, "EXAMPLE: HYDROGEN ATOM", fontsize=14,
            ha='center', fontweight='bold', transform=ax.transAxes)

    ax.text(0.5, 0.82, "Classical Conserved Quantities:", fontsize=12,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.72, "H (energy), L² (total angular momentum), Lᵤ (z-component)",
            fontsize=11, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue'))

    ax.text(0.5, 0.55, "↓", fontsize=20, ha='center', transform=ax.transAxes)

    ax.text(0.5, 0.45, "Quantum Good Numbers:", fontsize=12,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.35, "n (principal), ℓ (orbital), mₗ (magnetic)",
            fontsize=11, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.text(0.5, 0.20, "CSCO: {Ĥ, L̂², L̂ᵤ}\nStates: |n, ℓ, mₗ⟩",
            fontsize=11, ha='center', transform=ax.transAxes,
            family='monospace')

    ax.text(0.5, 0.05, "Hidden SO(4) symmetry → n² degeneracy!",
            fontsize=11, ha='center', transform=ax.transAxes,
            style='italic', color='red')

    ax.axis('off')

    plt.tight_layout()
    plt.savefig('quantum_connection.png', dpi=150)
    plt.show()


# Run all demonstrations
if __name__ == "__main__":
    constants_of_motion_demo()
    laplace_runge_lenz_demo()
    noether_symmetry_visual()
    quantum_connection_demo()

    print("\n" + "=" * 70)
    print("CONSTANTS OF MOTION LAB COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. {f, H} = 0 ↔ f is conserved")
    print("  2. Noether: Symmetry ↔ Conservation (one-to-one)")
    print("  3. n constants in involution → integrable system")
    print("  4. Cyclic coordinate qᵢ → pᵢ conserved")
    print("  5. Quantum: {f, H} = 0 → [f̂, Ĥ] = 0 → good quantum numbers")
    print("  6. Runge-Lenz vector: hidden symmetry in Kepler problem")
```

---

## Summary

### Conservation Criterion

$$\boxed{\frac{df}{dt} = \{f, H\} + \frac{\partial f}{\partial t} = 0}$$

For time-independent f: **{f, H} = 0 ↔ f conserved**

### Noether's Theorem

$$\boxed{\text{Continuous Symmetry} \quad \Longleftrightarrow \quad \text{Conservation Law}}$$

| Symmetry | Conservation |
|----------|--------------|
| Time translation | Energy |
| Space translation | Momentum |
| Rotation | Angular momentum |

### Integrability

$$\boxed{n \text{ DOF} + n \text{ constants in involution} \quad \Rightarrow \quad \text{Liouville integrable}}$$

### Action Variable

$$\boxed{I = \frac{1}{2\pi} \oint p \, dq}$$

### Classical-Quantum Bridge

$$\boxed{\{f, H\} = 0 \quad \longleftrightarrow \quad [\hat{f}, \hat{H}] = 0 \quad \text{(good quantum numbers)}}$$

---

## Daily Checklist

- [ ] State the conservation criterion {f, H} = 0
- [ ] Apply Poisson's theorem to generate new constants
- [ ] Explain Noether's theorem with examples
- [ ] Define Liouville integrability
- [ ] Calculate action variables for simple systems
- [ ] Identify cyclic coordinates and perform reduction
- [ ] Connect classical conservation to quantum numbers
- [ ] Run computational lab demonstrations

---

## Preview: Day 160

Tomorrow is our **Computational Lab Day** — a deep dive into implementing all the Hamiltonian mechanics concepts from this week in Python! We'll build phase space visualizers, Poisson bracket calculators, symplectic integrators, and explore the transition from integrable to chaotic systems.
