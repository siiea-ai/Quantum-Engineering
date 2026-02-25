# Day 163: Liouville's Theorem — Conservation of Phase Space Volume

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Liouville's Theorem |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. State and prove Liouville's theorem using multiple approaches (divergence, Jacobian, geometric)
2. Explain why Hamiltonian flow is analogous to incompressible fluid flow
3. Write and interpret the Liouville equation for phase space density evolution
4. Connect Liouville's theorem to the von Neumann equation in quantum mechanics
5. Apply Liouville's theorem to statistical mechanics and plasma physics
6. Understand why symplectic integrators preserve phase space structure

---

## Core Content

### 1. Statement of Liouville's Theorem

Liouville's theorem is one of the most profound results in classical mechanics. It states that **Hamiltonian dynamics preserves phase space volume**.

**Theorem (Liouville, 1838):** For a Hamiltonian system, the phase space volume occupied by any ensemble of systems is constant in time:

$$\boxed{\frac{dV}{dt} = 0}$$

Equivalently, the phase space probability density ρ(q, p, t) is constant along trajectories:

$$\boxed{\frac{d\rho}{dt} = \frac{\partial \rho}{\partial t} + \sum_{i=1}^{n}\left(\frac{\partial \rho}{\partial q_i}\dot{q}_i + \frac{\partial \rho}{\partial p_i}\dot{p}_i\right) = 0}$$

**Physical Picture:** Imagine an ensemble of identical classical systems, each starting from slightly different initial conditions. As time evolves, the cloud of representative points in phase space may change shape dramatically—stretching, folding, forming filaments—but its total volume remains constant.

---

### 2. Proof I: Divergence-Free Flow

The most elegant proof recognizes that Hamiltonian dynamics defines a vector field in phase space.

**Phase Space Velocity Field:**

The "velocity" of a point in phase space is:

$$\mathbf{v} = \left(\dot{q}_1, \ldots, \dot{q}_n, \dot{p}_1, \ldots, \dot{p}_n\right) = \left(\frac{\partial H}{\partial p_1}, \ldots, \frac{\partial H}{\partial p_n}, -\frac{\partial H}{\partial q_1}, \ldots, -\frac{\partial H}{\partial q_n}\right)$$

**Divergence Calculation:**

$$\nabla \cdot \mathbf{v} = \sum_{i=1}^{n}\left(\frac{\partial \dot{q}_i}{\partial q_i} + \frac{\partial \dot{p}_i}{\partial p_i}\right)$$

Substituting Hamilton's equations:

$$\nabla \cdot \mathbf{v} = \sum_{i=1}^{n}\left(\frac{\partial}{\partial q_i}\frac{\partial H}{\partial p_i} + \frac{\partial}{\partial p_i}\left(-\frac{\partial H}{\partial q_i}\right)\right)$$

$$= \sum_{i=1}^{n}\left(\frac{\partial^2 H}{\partial q_i \partial p_i} - \frac{\partial^2 H}{\partial p_i \partial q_i}\right) = 0$$

**Key Result:**

$$\boxed{\nabla \cdot \mathbf{v} = 0}$$

Hamiltonian flow is **divergence-free**—it is like an **incompressible fluid** in phase space!

**From Divergence to Volume Conservation:**

The continuity equation for phase space density is:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

Expanding:

$$\frac{\partial \rho}{\partial t} + \mathbf{v} \cdot \nabla \rho + \rho(\nabla \cdot \mathbf{v}) = 0$$

Since ∇ · **v** = 0:

$$\frac{\partial \rho}{\partial t} + \mathbf{v} \cdot \nabla \rho = \frac{d\rho}{dt} = 0$$

---

### 3. Proof II: The Jacobian Approach

Time evolution is a canonical transformation. We proved yesterday that canonical transformations have unit Jacobian.

**The Flow Map:**

Let Φ_t : (q₀, p₀) → (q(t), p(t)) be the time-t flow map. Then:

$$D_t = \det\left(\frac{\partial(q(t), p(t))}{\partial(q_0, p_0)}\right)$$

**Infinitesimal Analysis:**

For small δt:

$$q_i(t + \delta t) = q_i(t) + \dot{q}_i \delta t + O(\delta t^2)$$
$$p_i(t + \delta t) = p_i(t) + \dot{p}_i \delta t + O(\delta t^2)$$

The Jacobian matrix is:

$$\mathbf{M} = \mathbf{I} + \delta t \begin{pmatrix} \frac{\partial \dot{q}}{\partial q} & \frac{\partial \dot{q}}{\partial p} \\ \frac{\partial \dot{p}}{\partial q} & \frac{\partial \dot{p}}{\partial p} \end{pmatrix} + O(\delta t^2)$$

**Determinant:**

$$\det(\mathbf{M}) = 1 + \delta t \cdot \text{Tr}\begin{pmatrix} \frac{\partial \dot{q}}{\partial q} & \frac{\partial \dot{q}}{\partial p} \\ \frac{\partial \dot{p}}{\partial q} & \frac{\partial \dot{p}}{\partial p} \end{pmatrix} + O(\delta t^2)$$

$$= 1 + \delta t \cdot (\nabla \cdot \mathbf{v}) + O(\delta t^2) = 1$$

Therefore det(M) = 1 for all time, and phase space volume is preserved.

---

### 4. Proof III: Symplectic Structure

The most fundamental proof uses the symplectic form.

**Symplectic 2-Form:**

$$\omega = \sum_{i=1}^{n} dp_i \wedge dq_i$$

**Volume Element:**

The phase space volume element is:

$$d\Gamma = \frac{\omega^n}{n!} = dq_1 \wedge dp_1 \wedge \cdots \wedge dq_n \wedge dp_n$$

**Canonical Transformations Preserve ω:**

We showed yesterday that canonical transformations preserve the symplectic form:

$$\omega' = \omega$$

Therefore they also preserve the volume element d Γ, which is the nth exterior power of ω.

**Conclusion:** Hamiltonian time evolution is a canonical transformation, so it preserves phase space volume.

---

### 5. The Liouville Equation

Using the Poisson bracket, Liouville's theorem takes an elegant form.

**The Liouville Equation:**

$$\boxed{\frac{\partial \rho}{\partial t} + \{\rho, H\} = 0}$$

or equivalently:

$$\frac{\partial \rho}{\partial t} = \{H, \rho\}$$

**Verification:**

$$\{\rho, H\} = \sum_{i=1}^{n}\left(\frac{\partial \rho}{\partial q_i}\frac{\partial H}{\partial p_i} - \frac{\partial \rho}{\partial p_i}\frac{\partial H}{\partial q_i}\right) = \sum_{i=1}^{n}\left(\frac{\partial \rho}{\partial q_i}\dot{q}_i + \frac{\partial \rho}{\partial p_i}\dot{p}_i\right)$$

So:

$$\frac{\partial \rho}{\partial t} + \{\rho, H\} = \frac{\partial \rho}{\partial t} + \mathbf{v} \cdot \nabla \rho = \frac{d\rho}{dt} = 0 \quad \checkmark$$

---

### 6. The Liouville Operator

For any phase space function f(q, p, t), we can define the **Liouville operator**:

$$\boxed{iL = \{H, \cdot\}}$$

The factor of i is conventional, making L Hermitian with respect to the phase space inner product.

**Properties:**

| Property | Statement |
|----------|-----------|
| Definition | $iLf = \{H, f\} = -\{f, H\}$ |
| Equation of motion | $\frac{df}{dt} = -iLf + \frac{\partial f}{\partial t}$ |
| Time evolution operator | $f(t) = e^{-iLt}f(0)$ for time-independent H |
| Hermiticity | $\int (Lf)^* g \, d\Gamma = \int f^* (Lg) \, d\Gamma$ |

**Formal Solution:**

The phase space density evolves as:

$$\rho(t) = e^{-iLt}\rho(0)$$

This is the classical analog of unitary time evolution in quantum mechanics!

---

### 7. Geometric Interpretation: Incompressible Flow

**The Water Analogy:**

Imagine phase space filled with an incompressible fluid (like water). The "velocity field" is determined by Hamilton's equations. Because ∇ · **v** = 0:

- The fluid cannot compress or expand
- What it loses in one dimension, it gains in another
- "Packets" of fluid stretch, fold, and deform, but never change volume

**Consequences:**

1. **No Attractors:** Phase space trajectories cannot converge to a point or lower-dimensional manifold (this would require volume shrinkage)

2. **No Sources:** Trajectories cannot diverge from a point (this would require volume expansion)

3. **Shape Change Allowed:** While volume is preserved, the shape can change dramatically:
   - Stretching along unstable directions
   - Compression along stable directions
   - Folding and filamentation (chaos!)

**The Mixing Paradox:**

If volume is conserved, how can systems approach equilibrium?

**Resolution:** The phase space region may develop an incredibly complex, fractal-like structure while maintaining constant volume. Any finite-resolution measurement will see apparent mixing and equilibration, even though the exact microscopic state preserves its "volume."

---

### 8. Poincaré Recurrence Theorem

Liouville's theorem has a remarkable consequence discovered by Henri Poincaré in 1890.

**Theorem (Poincaré Recurrence):** For a Hamiltonian system with bounded phase space, almost every trajectory returns arbitrarily close to its initial point infinitely often.

**Proof Sketch:**

1. Consider a small region A in phase space with volume V(A) > 0
2. Let A, T(A), T²(A), ... be the successive time-evolved images of A
3. If these never intersected, they would be disjoint sets of equal positive volume
4. Infinite disjoint sets of positive volume cannot fit in a finite total volume
5. Therefore, some iterate Tⁿ(A) must intersect A
6. A point in Tⁿ(A) ∩ A has returned close to its starting point

**Implications:**

- Every isolated system will eventually return close to any previous state
- The recurrence time can be astronomically long (longer than the age of the universe for macroscopic systems)
- Resolves the apparent paradox between microscopic reversibility and macroscopic irreversibility

**Recurrence Time Estimate:**

For a gas of N molecules in a box, a rough estimate gives:

$$T_{\text{recurrence}} \sim e^{N}$$

For Avogadro's number of particles, this is unfathomably large—much greater than 10^(10²³) seconds!

---

## Quantum Mechanics Connection

### The von Neumann Equation

The quantum analog of the Liouville equation is the **von Neumann equation** (also called the quantum Liouville equation):

$$\boxed{\frac{\partial \hat{\rho}}{\partial t} = -\frac{i}{\hbar}[\hat{H}, \hat{\rho}]}$$

where $\hat{\rho}$ is the density operator (density matrix).

### The Classical-Quantum Correspondence

| Classical | Quantum |
|-----------|---------|
| Phase space density ρ(q, p, t) | Density operator $\hat{\rho}$ |
| Liouville equation: $\frac{\partial \rho}{\partial t} = \{H, \rho\}$ | von Neumann: $\frac{\partial \hat{\rho}}{\partial t} = -\frac{i}{\hbar}[\hat{H}, \hat{\rho}]$ |
| Poisson bracket {·,·} | Commutator [·,·]/(iℏ) |
| Liouville operator $iL = \{H, \cdot\}$ | Superoperator $\mathcal{L} = -\frac{i}{\hbar}[\hat{H}, \cdot]$ |
| Volume preservation | Unitarity: $\hat{\rho}(t) = \hat{U}(t)\hat{\rho}(0)\hat{U}^\dagger(t)$ |
| $\int \rho \, d\Gamma = 1$ | $\text{Tr}(\hat{\rho}) = 1$ |

### Unitarity as Quantum Liouville

For a pure state $|\psi(t)\rangle = \hat{U}(t)|\psi(0)\rangle$:

$$\hat{\rho}(t) = |\psi(t)\rangle\langle\psi(t)| = \hat{U}(t)\hat{\rho}(0)\hat{U}^\dagger(t)$$

**Properties preserved under unitary evolution:**

| Property | Classical | Quantum |
|----------|-----------|---------|
| Normalization | $\int \rho \, d\Gamma = 1$ | $\text{Tr}(\hat{\rho}) = 1$ |
| Positivity | ρ ≥ 0 | $\hat{\rho} \geq 0$ |
| Purity | N/A | $\text{Tr}(\hat{\rho}^2) = \text{const}$ |

**Unitarity IS quantum Liouville!** Both express the same fundamental principle: closed system evolution preserves the "size" of the state space.

### The Wigner Function

The **Wigner quasiprobability distribution** W(q, p, t) provides a phase space representation of quantum mechanics:

$$W(q, p) = \frac{1}{\pi\hbar}\int_{-\infty}^{\infty} \psi^*(q+y)\psi(q-y)e^{2ipy/\hbar} dy$$

**Evolution equation:**

$$\frac{\partial W}{\partial t} = \{H, W\}_M$$

where {·,·}_M is the **Moyal bracket**:

$$\{f, g\}_M = \frac{2}{\hbar}\sin\left(\frac{\hbar}{2}\left(\overleftarrow{\partial_q}\overrightarrow{\partial_p} - \overleftarrow{\partial_p}\overrightarrow{\partial_q}\right)\right)f \cdot g$$

**Classical Limit:** As ℏ → 0, the Moyal bracket reduces to the Poisson bracket, and quantum mechanics approaches classical mechanics.

### Decoherence: Apparent Liouville Violation

For **open quantum systems** (system + environment), the reduced density matrix obeys the **Lindblad equation**:

$$\frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar}[\hat{H}, \hat{\rho}] + \sum_k \left(L_k \hat{\rho} L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \hat{\rho}\}\right)$$

The additional "Lindblad" terms cause:
- Decoherence (off-diagonal decay)
- Entropy increase
- Approach to mixed states

**Not a true violation:** The full system + environment still evolves unitarily. Only the reduced description of the subsystem appears non-unitary.

---

## Applications

### 1. Statistical Mechanics Foundations

Liouville's theorem justifies the fundamental postulate of statistical mechanics.

**Equal A Priori Probability:**

For an isolated system with energy E, the microcanonical distribution is:

$$\rho_{\text{micro}} = \frac{1}{\Omega(E)}\delta(H - E)$$

where Ω(E) is the number of microstates with energy E.

**Why This Works:** Since ρ depends only on H (a constant of motion), we have:

$$\{\rho_{\text{micro}}, H\} = 0$$

The microcanonical distribution is stationary—it satisfies the Liouville equation with ∂ρ/∂t = 0.

**Gibbs Entropy:**

$$S = -k_B \int \rho \ln \rho \, d\Gamma$$

Liouville's theorem implies that this entropy is constant for isolated Hamiltonian systems—yet we observe entropy increase! This is the origin of the deep connection between statistical mechanics and information theory.

### 2. Plasma Physics: The Vlasov Equation

For a collisionless plasma, the distribution function f(x, v, t) satisfies:

$$\boxed{\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla_x f + \frac{q}{m}(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \nabla_v f = 0}$$

This is the **Vlasov equation**—the Liouville equation for charged particles in electromagnetic fields.

**Applications:**
- Landau damping of plasma waves
- Instabilities in fusion plasmas
- Solar wind dynamics
- Charged particle beams

### 3. Accelerator Physics: Beam Emittance

In particle accelerators, the **emittance** ε measures the phase space area occupied by a particle beam:

$$\epsilon = \frac{1}{\pi}\int\int dx \, dp_x$$

**Liouville's Theorem for Beams:**

For conservative (Hamiltonian) beam transport, emittance is conserved. This has profound implications:

- Position spread can only be reduced at the cost of momentum spread
- Beam brightness (particles per phase space volume) is limited
- Similar to the uncertainty principle!

**When Emittance Changes:**

Emittance can increase due to:
- Scattering (collisions)
- Non-Hamiltonian forces
- Radiation damping (synchrotron radiation)

Emittance can decrease due to:
- Cooling techniques (stochastic cooling, electron cooling)
- Synchrotron radiation damping (equilibrium between damping and quantum fluctuations)

### 4. Symplectic Integrators

**Key Insight:** Numerical integrators that preserve the symplectic structure automatically preserve phase space volume and qualitative dynamics.

**Standard Euler (Non-Symplectic):**

$$q_{n+1} = q_n + h \cdot p_n$$
$$p_{n+1} = p_n - h \cdot \nabla V(q_n)$$

This **does not** preserve phase space volume—trajectories spiral outward!

**Symplectic Euler:**

$$p_{n+1} = p_n - h \cdot \nabla V(q_n)$$
$$q_{n+1} = q_n + h \cdot p_{n+1}$$

Note: momentum is updated **first**, then position uses the **new** momentum.

**Störmer-Verlet (2nd Order Symplectic):**

$$p_{n+1/2} = p_n - \frac{h}{2}\nabla V(q_n)$$
$$q_{n+1} = q_n + h \cdot p_{n+1/2}$$
$$p_{n+1} = p_{n+1/2} - \frac{h}{2}\nabla V(q_{n+1})$$

**Why Symplectic Integrators Are Better:**

| Property | Non-Symplectic | Symplectic |
|----------|----------------|------------|
| Energy | Drifts (grows or shrinks) | Bounded oscillation |
| Phase space | Volume changes | Volume preserved |
| Long-time behavior | Qualitatively wrong | Qualitatively correct |
| Orbits | Spiral in/out | Remain bounded |

---

## Worked Examples

### Example 1: Harmonic Oscillator Ensemble

**Problem:** An ensemble of 1D harmonic oscillators has initial distribution:

$$\rho_0(q, p) = \frac{1}{2\pi\sigma_q\sigma_p}\exp\left(-\frac{q^2}{2\sigma_q^2} - \frac{p^2}{2\sigma_p^2}\right)$$

Find ρ(q, p, t) and verify Liouville's theorem.

**Solution:**

**Step 1:** Solve the dynamics.

For H = (p² + ω²q²)/2:

$$q(t) = q_0\cos(\omega t) + \frac{p_0}{\omega}\sin(\omega t)$$
$$p(t) = p_0\cos(\omega t) - \omega q_0\sin(\omega t)$$

**Step 2:** Invert to find initial conditions.

$$q_0 = q\cos(\omega t) - \frac{p}{\omega}\sin(\omega t)$$
$$p_0 = p\cos(\omega t) + \omega q\sin(\omega t)$$

**Step 3:** Apply Liouville's theorem.

Since dρ/dt = 0 along trajectories:

$$\rho(q, p, t) = \rho_0(q_0(q, p, t), p_0(q, p, t))$$

**Step 4:** Compute ρ(q, p, t).

$$q_0^2 = q^2\cos^2(\omega t) - \frac{qp}{\omega}\sin(2\omega t) + \frac{p^2}{\omega^2}\sin^2(\omega t)$$

$$p_0^2 = p^2\cos^2(\omega t) + \omega qp\sin(2\omega t) + \omega^2 q^2\sin^2(\omega t)$$

After simplification:

$$\frac{q_0^2}{\sigma_q^2} + \frac{p_0^2}{\sigma_p^2} = \frac{q^2}{\sigma_q^2(t)} + \frac{p^2}{\sigma_p^2(t)} + \text{correlation terms}$$

The distribution rotates in phase space while preserving area!

**Step 5:** Verify area preservation.

Initial area: $A_0 = \pi \sigma_q \sigma_p$ (within one standard deviation)

At time t: The ellipse has rotated, but det(Jacobian) = 1, so A(t) = A₀ ✓

---

### Example 2: Free Particle Spreading

**Problem:** Show that for free particles, position uncertainty grows while momentum uncertainty remains constant, but phase space area is preserved.

**Solution:**

**Hamiltonian:** H = p²/(2m)

**Dynamics:** q(t) = q₀ + (p₀/m)t, p(t) = p₀

**Initial distribution:** Rectangle with sides Δq₀ and Δp₀

**Time evolution:**

At time t, the rectangle shears into a parallelogram:
- Corners: (q₀, p₀) → (q₀ + p₀t/m, p₀)

**Area calculation:**

The Jacobian is:

$$\mathbf{M} = \begin{pmatrix} 1 & t/m \\ 0 & 1 \end{pmatrix}$$

$$\det(\mathbf{M}) = 1$$

So area = Δq₀ · Δp₀ = constant ✓

**Position spread:**

$$\Delta q(t) = \Delta q_0 + \frac{\Delta p_0}{m}t$$

grows linearly with time.

**Momentum spread:**

$$\Delta p(t) = \Delta p_0$$

remains constant.

**Correlation:**

A correlation develops: particles with larger p have moved further in q.

---

### Example 3: Damped Oscillator (Liouville Violation)

**Problem:** For the damped harmonic oscillator, show that phase space volume contracts.

**Equations of motion:**

$$\dot{q} = \frac{p}{m}, \quad \dot{p} = -kq - \gamma p$$

**Solution:**

**Step 1:** Compute divergence.

$$\nabla \cdot \mathbf{v} = \frac{\partial \dot{q}}{\partial q} + \frac{\partial \dot{p}}{\partial p} = 0 + (-\gamma) = -\gamma$$

**Step 2:** Analyze volume evolution.

$$\frac{dV}{dt} = V \cdot (\nabla \cdot \mathbf{v}) = -\gamma V$$

**Step 3:** Solve.

$$V(t) = V_0 e^{-\gamma t}$$

Phase space volume contracts exponentially!

**Physical Interpretation:**

- All trajectories spiral toward the origin (an attractor)
- Information is "lost" (entropy increases)
- The system is not Hamiltonian—friction is a non-conservative force

**Liouville's theorem only applies to Hamiltonian systems!**

---

## Practice Problems

### Level 1: Direct Application

1. **Divergence-free:** Verify that ∇·**v** = 0 for the Hamiltonian H = p²/(2m) + mω²q²/2.

2. **Phase space area:** An ensemble of particles fills a square region 0 ≤ q ≤ 1, 0 ≤ p ≤ 1 at t = 0. For H = p²/2, find the shape at t = 2 and verify the area is unchanged.

3. **Liouville equation:** Show that ρ(q, p) = f(H(q, p)) satisfies ∂ρ/∂t + {ρ, H} = 0 for any function f.

### Level 2: Intermediate

4. **Forced oscillator:** For H(q, p, t) = p²/2 + q²/2 + qF₀cos(ωt), is phase space volume still conserved? (Hint: Does the system remain Hamiltonian?)

5. **Rotating frame:** In a rotating reference frame with angular velocity Ω, the effective Hamiltonian includes -Ω·L. Show that Liouville's theorem still holds.

6. **Entropy paradox:** If the Gibbs entropy S = -k∫ρ ln ρ dΓ is constant under Hamiltonian evolution (Liouville), how can the thermodynamic entropy increase?

### Level 3: Challenging

7. **Poincaré recurrence:** For a particle in a 1D box (H = p²/2m, 0 < q < L with reflecting walls), estimate the recurrence time for a phase space region of area h.

8. **Liouville operator spectrum:** For H = ωJ (action-angle Hamiltonian), find the eigenvalues and eigenfunctions of the Liouville operator L = -i{H, ·}.

9. **Quantum-classical correspondence:** Starting from the von Neumann equation, derive the Wigner function evolution equation and show it reduces to Liouville in the classical limit ℏ → 0.

---

## Computational Lab

### Lab 1: Visualizing Liouville's Theorem

```python
"""
Day 163 Lab: Liouville's Theorem Visualization
Demonstrates phase space volume conservation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull

class Pendulum:
    """Simple pendulum Hamiltonian system."""

    def __init__(self, g=9.8, L=1.0, m=1.0):
        self.g = g
        self.L = L
        self.m = m

    def hamiltonian(self, theta, p):
        """H = p²/(2mL²) - mgL cos(θ)"""
        return p**2 / (2*self.m*self.L**2) - self.m*self.g*self.L*np.cos(theta)

    def equations(self, t, y):
        """Hamilton's equations."""
        theta, p = y
        dtheta = p / (self.m * self.L**2)
        dp = -self.m * self.g * self.L * np.sin(theta)
        return [dtheta, dp]


def compute_area(points):
    """Compute area using convex hull."""
    if len(points) < 3:
        return 0
    try:
        hull = ConvexHull(points)
        return hull.volume  # 2D 'volume' = area
    except:
        return 0


def liouville_theorem_demonstration():
    """Demonstrate Liouville's theorem for pendulum ensemble."""

    pendulum = Pendulum()

    # Create ensemble of initial conditions
    n_points = 300
    np.random.seed(42)

    # Initial region: small rectangle near stable equilibrium
    theta_center, p_center = 0.5, 0.3
    theta_spread, p_spread = 0.2, 0.2

    theta_init = theta_center + theta_spread * (np.random.rand(n_points) - 0.5)
    p_init = p_center + p_spread * (np.random.rand(n_points) - 0.5)

    # Time snapshots
    t_max = 8.0
    t_snapshots = [0, 1, 2, 4, 6, 8]

    # Evolve all trajectories
    trajectories = []
    for th0, p0 in zip(theta_init, p_init):
        sol = solve_ivp(
            pendulum.equations,
            [0, t_max],
            [th0, p0],
            t_eval=t_snapshots,
            method='DOP853',
            rtol=1e-10
        )
        trajectories.append(sol.y)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    areas = []

    for idx, t in enumerate(t_snapshots):
        ax = axes[idx // 3, idx % 3]

        # Extract points at this time
        points = np.array([[traj[0, idx], traj[1, idx]] for traj in trajectories])

        # Plot points
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.6, c='blue')

        # Compute area
        area = compute_area(points)
        areas.append(area)

        # Energy contours
        theta_grid = np.linspace(-np.pi, np.pi, 100)
        p_grid = np.linspace(-3, 3, 100)
        THETA, P = np.meshgrid(theta_grid, p_grid)
        H = pendulum.hamiltonian(THETA, P)
        ax.contour(THETA, P, H, levels=15, colors='gray', alpha=0.3, linewidths=0.5)

        ax.set_xlim(-1.5, 2.0)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('θ', fontsize=12)
        ax.set_ylabel('p', fontsize=12)
        ax.set_title(f't = {t:.1f}s, Area = {area:.4f}', fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Liouville's Theorem: Phase Space Volume Conservation\n"
                 "Shape changes dramatically but area remains constant",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('liouville_pendulum.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print("Phase Space Areas over Time:")
    print("=" * 40)
    for t, area in zip(t_snapshots, areas):
        print(f"t = {t:4.1f}s: Area = {area:.6f}")
    print(f"\nRelative variation: {(max(areas)-min(areas))/areas[0]*100:.2f}%")

liouville_theorem_demonstration()
```

### Lab 2: Hamiltonian vs Dissipative Comparison

```python
"""
Comparison of phase space evolution for Hamiltonian vs dissipative systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def harmonic_undamped(t, y, omega=1.0):
    """Hamiltonian system."""
    q, p = y
    return [p, -omega**2 * q]

def harmonic_damped(t, y, omega=1.0, gamma=0.3):
    """Non-Hamiltonian (dissipative) system."""
    q, p = y
    return [p, -omega**2 * q - gamma * p]

def evolve_ensemble(eqns, n_points=200, t_max=10.0, **kwargs):
    """Evolve an ensemble of initial conditions."""

    np.random.seed(42)
    q_init = 1.0 + 0.3 * (np.random.rand(n_points) - 0.5)
    p_init = 0.0 + 0.3 * (np.random.rand(n_points) - 0.5)

    t_eval = np.linspace(0, t_max, 100)

    all_q, all_p = [], []
    for q0, p0 in zip(q_init, p_init):
        sol = solve_ivp(
            lambda t, y: eqns(t, y, **kwargs),
            [0, t_max],
            [q0, p0],
            t_eval=t_eval
        )
        all_q.append(sol.y[0])
        all_p.append(sol.y[1])

    return np.array(all_q), np.array(all_p), t_eval

def compare_systems():
    """Compare Hamiltonian vs dissipative evolution."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Undamped (Hamiltonian)
    all_q, all_p, t_eval = evolve_ensemble(harmonic_undamped)

    time_indices = [0, 30, 60, 90]
    for idx, t_idx in enumerate(time_indices):
        ax = axes[0, idx]
        ax.scatter(all_q[:, t_idx], all_p[:, t_idx], s=5, alpha=0.6, c='blue')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f'Undamped, t={t_eval[t_idx]:.1f}')
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, lw=2)

    # Damped (Non-Hamiltonian)
    all_q, all_p, t_eval = evolve_ensemble(harmonic_damped, gamma=0.3)

    for idx, t_idx in enumerate(time_indices):
        ax = axes[1, idx]
        ax.scatter(all_q[:, t_idx], all_p[:, t_idx], s=5, alpha=0.6, c='red')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f'Damped (γ=0.3), t={t_eval[t_idx]:.1f}')
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.plot(0, 0, 'k*', markersize=15)  # Attractor

    fig.suptitle("Liouville's Theorem: Hamiltonian vs Dissipative\n"
                 "Top: Area preserved (rotates) | Bottom: Area shrinks (spirals to attractor)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('hamiltonian_vs_dissipative.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_systems()
```

### Lab 3: Symplectic vs Non-Symplectic Integrators

```python
"""
Compare symplectic and non-symplectic integrators for long-time simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

def euler(q0, p0, omega, dt, n_steps):
    """Standard Euler (non-symplectic)."""
    q, p = np.zeros(n_steps+1), np.zeros(n_steps+1)
    q[0], p[0] = q0, p0
    for i in range(n_steps):
        q[i+1] = q[i] + dt * p[i]
        p[i+1] = p[i] - dt * omega**2 * q[i]
    return q, p

def symplectic_euler(q0, p0, omega, dt, n_steps):
    """Symplectic Euler."""
    q, p = np.zeros(n_steps+1), np.zeros(n_steps+1)
    q[0], p[0] = q0, p0
    for i in range(n_steps):
        p[i+1] = p[i] - dt * omega**2 * q[i]   # p first
        q[i+1] = q[i] + dt * p[i+1]            # use new p
    return q, p

def verlet(q0, p0, omega, dt, n_steps):
    """Störmer-Verlet (2nd order symplectic)."""
    q, p = np.zeros(n_steps+1), np.zeros(n_steps+1)
    q[0], p[0] = q0, p0
    for i in range(n_steps):
        p_half = p[i] - 0.5 * dt * omega**2 * q[i]
        q[i+1] = q[i] + dt * p_half
        p[i+1] = p_half - 0.5 * dt * omega**2 * q[i+1]
    return q, p

def integrator_comparison():
    """Compare integrators over long time."""

    omega = 1.0
    dt = 0.1
    n_steps = 2000  # Long simulation
    q0, p0 = 1.0, 0.0
    E0 = 0.5 * (p0**2 + omega**2 * q0**2)

    # Run integrators
    q_eu, p_eu = euler(q0, p0, omega, dt, n_steps)
    q_se, p_se = symplectic_euler(q0, p0, omega, dt, n_steps)
    q_vl, p_vl = verlet(q0, p0, omega, dt, n_steps)

    # Energies
    E_eu = 0.5 * (p_eu**2 + omega**2 * q_eu**2)
    E_se = 0.5 * (p_se**2 + omega**2 * q_se**2)
    E_vl = 0.5 * (p_vl**2 + omega**2 * q_vl**2)

    t = np.arange(n_steps + 1) * dt

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Phase space
    ax1 = axes[0, 0]
    ax1.plot(q_eu, p_eu, 'r-', alpha=0.5, lw=0.5, label='Euler')
    ax1.plot(q_se, p_se, 'b-', alpha=0.5, lw=0.5, label='Symplectic Euler')
    ax1.plot(q_vl, p_vl, 'g-', alpha=0.5, lw=0.5, label='Verlet')
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', lw=2, label='Exact')
    ax1.set_xlabel('q')
    ax1.set_ylabel('p')
    ax1.set_title('Phase Space (Long Time)')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Energy
    ax2 = axes[0, 1]
    ax2.plot(t, E_eu, 'r-', lw=0.5, label='Euler')
    ax2.plot(t, E_se, 'b-', lw=0.5, label='Symplectic Euler')
    ax2.plot(t, E_vl, 'g-', lw=0.5, label='Verlet')
    ax2.axhline(E0, color='k', ls='--', lw=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Conservation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Energy error
    ax3 = axes[1, 0]
    ax3.semilogy(t, np.abs(E_eu - E0) + 1e-16, 'r-', lw=0.5, label='Euler')
    ax3.semilogy(t, np.abs(E_se - E0) + 1e-16, 'b-', lw=0.5, label='Symplectic Euler')
    ax3.semilogy(t, np.abs(E_vl - E0) + 1e-16, 'g-', lw=0.5, label='Verlet')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('|E - E₀|')
    ax3.set_title('Energy Error (log scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary = """
    SYMPLECTIC INTEGRATORS AND LIOUVILLE'S THEOREM

    Non-symplectic methods (standard Euler):
    • Do NOT preserve phase space volume
    • Energy drifts systematically
    • Trajectories spiral outward/inward
    • Qualitatively wrong long-time behavior

    Symplectic methods (Symplectic Euler, Verlet):
    • PRESERVE phase space volume
    • Energy oscillates but stays bounded
    • Trajectories remain on invariant curves
    • Qualitatively correct for any time

    This is Liouville's theorem in action!
    Symplectic = volume-preserving = physical.
    """
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Integrator Comparison: Liouville's Theorem Preservation", fontsize=14)
    plt.tight_layout()
    plt.savefig('integrator_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print final values
    print("\nFinal Energy Ratios:")
    print(f"Euler:           E_final/E_0 = {E_eu[-1]/E0:.4f}")
    print(f"Symplectic Euler: E_final/E_0 = {E_se[-1]/E0:.4f}")
    print(f"Verlet:          E_final/E_0 = {E_vl[-1]/E0:.4f}")

integrator_comparison()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Liouville's theorem | $\frac{dV}{dt} = 0$ for phase space volume |
| Liouville equation | $\frac{\partial \rho}{\partial t} + \{\rho, H\} = 0$ |
| Divergence-free flow | $\nabla \cdot \mathbf{v} = 0$ |
| Liouville operator | $iL = \{H, \cdot\}$ |
| Time evolution | $\rho(t) = e^{-iLt}\rho(0)$ |
| von Neumann equation | $\frac{\partial \hat{\rho}}{\partial t} = -\frac{i}{\hbar}[\hat{H}, \hat{\rho}]$ |
| Dissipative volume change | $\frac{dV}{dt} = -\gamma V$ for damping |

### Main Takeaways

1. **Liouville's Theorem:** Phase space volume is conserved under Hamiltonian evolution
   - Hamiltonian flow is like an incompressible fluid
   - Shape can change, but volume cannot

2. **Multiple Proofs:**
   - Divergence-free: ∇·**v** = 0 because mixed partials of H cancel
   - Jacobian: det(M) = 1 for canonical transformations
   - Symplectic: Canonical transformations preserve ω

3. **The Liouville Equation:** ∂ρ/∂t + {ρ, H} = 0
   - Density is constant along trajectories
   - Equilibrium distributions: {ρ, H} = 0 (e.g., ρ = f(H))

4. **Quantum Connection:**
   - Liouville equation → von Neumann equation
   - Volume preservation → Unitarity
   - Poisson bracket → Commutator/(iℏ)

5. **Applications:**
   - Statistical mechanics: justifies equal a priori probability
   - Plasma physics: Vlasov equation
   - Accelerators: emittance conservation
   - Numerics: symplectic integrators

6. **Poincaré Recurrence:** Almost all states eventually return close to their initial state
   - But recurrence times are astronomically long

---

## Daily Checklist

### Understanding
- [ ] I can prove Liouville's theorem using divergence-free flow
- [ ] I understand why Hamiltonian flow preserves phase space volume
- [ ] I can write and interpret the Liouville equation
- [ ] I see the connection to quantum unitarity

### Computation
- [ ] I can compute ∇·**v** for a given Hamiltonian
- [ ] I can verify that a distribution satisfies the Liouville equation
- [ ] I can implement symplectic integrators

### Connections
- [ ] I understand why statistical mechanics uses equal a priori probability
- [ ] I can explain why non-symplectic integrators fail for long-time simulations
- [ ] I appreciate the depth of the classical-quantum correspondence

---

## Preview: Day 164

Tomorrow we study **Action-Angle Variables**, the culmination of Hamiltonian mechanics. For integrable systems, these special coordinates make the Hamiltonian depend only on "actions" J, while "angles" θ evolve linearly in time.

Key concepts:
- Action: $J = \frac{1}{2\pi}\oint p \, dq$
- Angle: conjugate to J, period 2π
- The Liouville-Arnold theorem: phase space is foliated into invariant tori
- **Bohr-Sommerfeld quantization:** J = nℏ connects to quantum mechanics!

This is where classical mechanics reaches its most elegant form—and directly touches quantum theory.

---

*"One of the principal objects of theoretical research is to find the point of view from which the subject appears in the greatest simplicity."*
— Josiah Willard Gibbs

---

**Day 163 Complete. Next: Action-Angle Variables**
