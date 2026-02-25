# Day 157: Phase Space — The Geometric Arena of Hamiltonian Mechanics

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Phase Space Structure |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Define phase space as a 2n-dimensional manifold and understand its geometric structure
2. Explain why phase space trajectories cannot cross and derive the existence/uniqueness theorem
3. Construct and interpret phase portraits for standard mechanical systems
4. Classify fixed points (centers, saddles) and understand their stability
5. Identify separatrices and explain their physical significance
6. Connect phase space concepts to quantum mechanics via the Wigner function and uncertainty principle

---

## Core Content

### 1. Definition and Structure of Phase Space

**Phase space** is the geometric arena where Hamiltonian dynamics unfolds. For a mechanical system with n degrees of freedom, phase space is a **2n-dimensional manifold** whose points represent all possible states of the system.

**Definition:** Each point in phase space is specified by n generalized coordinates and n conjugate momenta:

$$\mathbf{z} = (q_1, q_2, \ldots, q_n, p_1, p_2, \ldots, p_n) \in \mathbb{R}^{2n}$$

**Mathematical Structure:** Phase space is the **cotangent bundle** T*M of configuration space M. For a particle in 3D space:
- Configuration space: M = R³ (positions only)
- Phase space: T*R³ = R⁶ (positions and momenta)

For N particles in 3D, phase space has dimension 6N — an enormous space for macroscopic systems!

| System | Degrees of Freedom (n) | Phase Space Dimension (2n) |
|--------|------------------------|---------------------------|
| 1D harmonic oscillator | 1 | 2 |
| Planar pendulum | 1 | 2 |
| 3D particle | 3 | 6 |
| Double pendulum | 2 | 4 |
| N particles in 3D | 3N | 6N |

**Key Insight:** A single point in phase space contains **complete information** about the system's state. Unlike configuration space (which only specifies positions), phase space includes momenta, enabling deterministic prediction of future (and past) evolution.

---

### 2. The Symplectic Structure

Phase space possesses a fundamental geometric structure called the **symplectic 2-form**:

$$\boxed{\omega = \sum_{i=1}^{n} dp_i \wedge dq_i}$$

where ∧ denotes the exterior (wedge) product of differential forms.

**Properties of the Symplectic Form:**
1. **Closed:** dω = 0 (exterior derivative vanishes)
2. **Non-degenerate:** If ω(X, Y) = 0 for all vectors Y, then X = 0
3. **Alternating:** ω(X, Y) = -ω(Y, X)

**The Symplectic Matrix:** Hamilton's equations take the elegant form:

$$\boxed{\dot{\mathbf{z}} = \mathbf{J}\nabla H}$$

where the symplectic matrix is:

$$\mathbf{J} = \begin{pmatrix} \mathbf{0}_n & \mathbf{I}_n \\ -\mathbf{I}_n & \mathbf{0}_n \end{pmatrix}$$

**Properties of J:**
- J² = -I₂ₙ (acts like imaginary unit!)
- Jᵀ = -J (antisymmetric)
- det(J) = 1

**Etymology of "Symplectic":** The term was coined by Hermann Weyl in 1939. Previously called the "complex group," Weyl replaced it with the Greek equivalent *symplektikos* (meaning "twining together") to avoid confusion with complex numbers. Both words derive from the Indo-European root *plek-* meaning "to fold or weave."

---

### 3. Phase Space Trajectories

**Existence and Uniqueness Theorem (Picard-Lindelöf):**

For the initial value problem:
$$\dot{\mathbf{z}} = \mathbf{f}(\mathbf{z}, t), \quad \mathbf{z}(t_0) = \mathbf{z}_0$$

If **f** is continuous in t and **Lipschitz continuous** in **z**, then there exists a **unique** solution on some interval containing t₀.

**Physical Interpretation:** Once position and momentum are specified at any instant, the entire future (and past) evolution is uniquely determined. This is the mathematical foundation of **Laplacian determinism**.

**The Non-Crossing Theorem:**

$$\boxed{\text{Phase space trajectories of autonomous systems cannot cross}}$$

**Proof:** If two trajectories crossed at point z₀, then the initial value problem with z(t₀) = z₀ would have two distinct solutions, contradicting uniqueness.

**Exception:** Trajectories can meet at **fixed points** where ż = J∇H = 0.

---

### 4. Hamiltonian Flow

**Definition:** The **Hamiltonian flow** φᵗ: P → P is the map taking each initial condition z₀ to its state z(t) at time t:

$$\phi^t(\mathbf{z}_0) = \mathbf{z}(t)$$

**One-Parameter Group Structure:** For autonomous systems:
1. φ⁰ = identity
2. φᵗ ∘ φˢ = φᵗ⁺ˢ (group composition)
3. (φᵗ)⁻¹ = φ⁻ᵗ (time reversal)

**Symplectomorphisms:** Each flow map φᵗ preserves the symplectic structure:
$$(\phi^t)^* \omega = \omega$$

This means the Jacobian matrix Dφᵗ satisfies:

$$\boxed{(D\phi^t)^T \mathbf{J} (D\phi^t) = \mathbf{J}}$$

A transformation satisfying this condition is called a **canonical transformation** or **symplectomorphism**.

---

### 5. Phase Portraits

A **phase portrait** is the collection of all trajectories in phase space, revealing the qualitative behavior of the system.

**Construction Method:**
1. Find fixed points: solve ∇H = 0
2. Classify fixed points by linearization
3. Plot level curves H(q, p) = E (energy surfaces)
4. Add arrows indicating direction of flow
5. Identify separatrices

**Key Principle:** Since dH/dt = 0 for autonomous systems, trajectories lie on **level sets** of H. The phase portrait is the family of curves H(q, p) = constant.

---

### 6. Classification of Fixed Points

**Definition:** A **fixed point** (equilibrium) is where ż = 0, i.e., ∇H(z*) = 0.

**Linearization:** Near a fixed point, the dynamics are approximated by:
$$\dot{\boldsymbol{\xi}} = \mathbf{A}\boldsymbol{\xi}, \quad \mathbf{A} = \mathbf{J} \cdot \text{Hess}(H)|_{\mathbf{z}^*}$$

where Hess(H) is the Hessian matrix of second derivatives.

**Classification for 2D Systems:**

| Eigenvalues | Type | Stability | Phase Portrait |
|-------------|------|-----------|----------------|
| Pure imaginary (±iω) | **Center** | Stable (Lyapunov) | Closed orbits |
| Real, opposite signs (±λ) | **Saddle** | Unstable | Hyperbolic curves |
| Real, same sign | Node | Stable/Unstable | Not Hamiltonian! |
| Complex with Re ≠ 0 | Spiral | Stable/Unstable | Not Hamiltonian! |

**Critical Result for Hamiltonian Systems:** Due to the symplectic structure, eigenvalues come in pairs: if λ is an eigenvalue, so are -λ, λ*, and -λ*. This **restricts fixed points to centers and saddles only** — no spirals or asymptotically stable nodes!

---

### 7. Separatrices and Special Orbits

**Definition:** A **separatrix** is a trajectory that divides phase space into regions with qualitatively different dynamics.

**Homoclinic Orbit:** A trajectory connecting a saddle point to **itself** — it lies in the intersection of the stable and unstable manifolds of the same equilibrium.

**Heteroclinic Orbit:** A trajectory connecting **two different** saddle points.

**Example — Simple Pendulum:**

The Hamiltonian is:
$$H = \frac{p_\theta^2}{2mL^2} - mgL\cos\theta$$

**Separatrix Energy:** At the unstable equilibrium (θ = π), E_sep = mgL.

**Separatrix Equation:**
$$\boxed{p_\theta = \pm 2mL\sqrt{gL}\cos\left(\frac{\theta}{2}\right)}$$

The separatrix divides:
- **Libration** (E < mgL): oscillation, closed curves
- **Rotation** (E > mgL): complete revolutions, wavy curves spanning all θ

---

### 8. Area Preservation (Preview of Liouville)

**Fundamental Result:** Hamiltonian flow preserves phase space volume!

**For 2D:** The area element dq ∧ dp is preserved. If a region has area A₀ at t = 0, its area at time t is still A₀, though its **shape** may change dramatically.

**Proof Preview:** The divergence of the Hamiltonian vector field vanishes:
$$\nabla \cdot \dot{\mathbf{z}} = \frac{\partial \dot{q}}{\partial q} + \frac{\partial \dot{p}}{\partial p} = \frac{\partial^2 H}{\partial q \partial p} - \frac{\partial^2 H}{\partial p \partial q} = 0$$

The flow is **incompressible** because mixed partial derivatives commute.

---

## Quantum Mechanics Connection

### The Uncertainty Principle and Phase Space Cells

**Heisenberg's Uncertainty Principle:**
$$\boxed{\Delta q \cdot \Delta p \geq \frac{\hbar}{2}}$$

**Profound Implication:** A quantum state cannot be localized to a phase space region smaller than ~ℏ per degree of freedom. For n degrees of freedom:

$$\Delta V_{\text{min}} \sim h^n = (2\pi\hbar)^n$$

**Counting Quantum States:** The number of quantum states in phase space volume V is approximately:
$$N \approx \frac{V}{h^n}$$

This is why Planck's constant has units of action (energy × time = momentum × position).

---

### The Wigner Quasi-probability Distribution

**Definition:** The Wigner function represents a quantum state ψ(x) in phase space:

$$\boxed{W(x,p) = \frac{1}{\pi\hbar} \int_{-\infty}^{\infty} \psi^*(x+y)\psi(x-y) e^{2ipy/\hbar} \, dy}$$

**Key Properties:**
1. **Real-valued:** W(x,p) ∈ ℝ
2. **Normalized:** ∫∫ W dx dp = 1
3. **Correct marginals:** ∫ W dp = |ψ(x)|² and ∫ W dx = |φ̃(p)|²
4. **Can be negative!** This is the signature of quantum behavior

**Physical Interpretation:** The Wigner function is the closest quantum analog to a classical phase space distribution. Negative regions (confined to areas ~ℏ) indicate quantum interference — there is no classical probability distribution that reproduces quantum predictions!

**Time Evolution:** The Wigner function evolves according to:
$$\frac{\partial W}{\partial t} = -\{H, W\}_{\text{Moyal}}$$

where {,}_{Moyal} is the **Moyal bracket** — a quantum deformation of the Poisson bracket that reduces to it as ℏ → 0.

---

### Classical-Quantum Correspondence

| Classical | Quantum |
|-----------|---------|
| Point in phase space (q, p) | State vector \|ψ⟩ in Hilbert space |
| Phase space distribution ρ(q,p) | Wigner function W(q,p) |
| Liouville equation | von Neumann equation |
| Poisson bracket {f, g} | Commutator [f̂, ĝ]/(iℏ) |
| Deterministic trajectories | Probability amplitudes |

**The Correspondence Principle:** As quantum numbers become large or ℏ → 0, quantum mechanics reproduces classical behavior. In phase space, the Wigner function becomes sharply peaked around the classical trajectory.

---

## Worked Examples

### Example 1: Simple Harmonic Oscillator Phase Portrait

**Problem:** Construct the complete phase portrait for H = p²/(2m) + ½mω²q².

**Solution:**

**Step 1: Find fixed points**
$$\nabla H = (m\omega^2 q, \, p/m) = 0$$

Only solution: q = 0, p = 0. Single fixed point at origin.

**Step 2: Classify the fixed point**

Hessian:
$$\text{Hess}(H) = \begin{pmatrix} m\omega^2 & 0 \\ 0 & 1/m \end{pmatrix}$$

Linearization matrix:
$$\mathbf{A} = \mathbf{J} \cdot \text{Hess}(H) = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}\begin{pmatrix} m\omega^2 & 0 \\ 0 & 1/m \end{pmatrix} = \begin{pmatrix} 0 & 1/m \\ -m\omega^2 & 0 \end{pmatrix}$$

Eigenvalues: det(A - λI) = λ² + ω² = 0, so λ = ±iω.

**Pure imaginary eigenvalues → Center (stable)**

**Step 3: Draw energy contours**

Setting H = E:
$$\frac{p^2}{2m} + \frac{1}{2}m\omega^2 q^2 = E$$

This is an **ellipse** with semi-axes:
$$q_{\max} = \sqrt{\frac{2E}{m\omega^2}}, \quad p_{\max} = \sqrt{2mE}$$

**Step 4: Determine flow direction**

At (q > 0, p = 0): q̇ = p/m = 0, ṗ = -mω²q < 0

Flow is **clockwise**.

**Step 5: Calculate enclosed area**

$$A = \pi \cdot q_{\max} \cdot p_{\max} = \pi \sqrt{\frac{2E}{m\omega^2}} \cdot \sqrt{2mE} = \frac{2\pi E}{\omega}$$

This equals the action variable J = E/ω (times 2π).

---

### Example 2: Simple Pendulum Separatrix

**Problem:** Find the equation of the separatrix for a simple pendulum and interpret its physical meaning.

**Solution:**

**Hamiltonian:**
$$H = \frac{p_\theta^2}{2mL^2} - mgL\cos\theta$$

**Fixed Points:**
- θ = 0, p = 0: stable equilibrium (hanging) → **Center**
- θ = ±π, p = 0: unstable equilibrium (inverted) → **Saddle**

**Separatrix Energy:**

At the saddle point (θ = π, p = 0):
$$E_{\text{sep}} = H(\pi, 0) = -mgL\cos(\pi) = mgL$$

**Separatrix Equation:**

Setting H = E_sep:
$$\frac{p_\theta^2}{2mL^2} - mgL\cos\theta = mgL$$
$$p_\theta^2 = 2m^2L^2g(1 + \cos\theta) = 4m^2L^2g\cos^2\left(\frac{\theta}{2}\right)$$
$$\boxed{p_\theta = \pm 2mL\sqrt{gL}\cos\left(\frac{\theta}{2}\right)}$$

**Physical Interpretation:**

The separatrix represents motion where the pendulum arrives at the top (θ = π) with **exactly zero velocity**. The motion takes **infinite time** (the pendulum asymptotically approaches the inverted position).

Inside the separatrix: **libration** (back-and-forth oscillation)
Outside the separatrix: **rotation** (complete revolutions)

---

### Example 3: Area Preservation Under Harmonic Oscillator Flow

**Problem:** Verify directly that a rectangular region in phase space maintains constant area under SHO evolution.

**Solution:**

Consider initial rectangle with corners at (q₀, p₀), (q₀ + δq, p₀), (q₀, p₀ + δp), (q₀ + δq, p₀ + δp).

**Initial area:** A₀ = δq · δp

**The SHO solution:**
$$q(t) = q_0\cos(\omega t) + \frac{p_0}{m\omega}\sin(\omega t)$$
$$p(t) = p_0\cos(\omega t) - m\omega q_0\sin(\omega t)$$

**The Jacobian of the transformation:**
$$\frac{\partial(q(t), p(t))}{\partial(q_0, p_0)} = \begin{pmatrix} \cos\omega t & \frac{\sin\omega t}{m\omega} \\ -m\omega\sin\omega t & \cos\omega t \end{pmatrix}$$

**Determinant:**
$$\det = \cos^2(\omega t) + \sin^2(\omega t) = 1$$

**Therefore:** A(t) = A₀ · |det J| = A₀

Area is exactly preserved, regardless of how distorted the shape becomes!

---

## Practice Problems

### Level 1: Direct Application

1. **Phase Space Dimension:** A system of 5 particles moving in 2D has how many phase space dimensions?

2. **Fixed Point Classification:** For H = p²/2 - q²/2, find the fixed point and classify it.

3. **Energy Contours:** Sketch the phase portrait for a free particle H = p²/(2m). What are the trajectories?

### Level 2: Intermediate

4. **Double Well Potential:** For H = p²/2 + q⁴/4 - q²/2:
   a) Find all fixed points
   b) Classify each as center or saddle
   c) Sketch the phase portrait
   d) Find the separatrix energy

5. **Verification:** Show that the transformation q' = p, p' = -q preserves the symplectic structure (i.e., dq' ∧ dp' = dq ∧ dp).

6. **Area Calculation:** For a harmonic oscillator with m = 1, ω = 2, find the area enclosed by the trajectory with energy E = 4.

### Level 3: Challenging

7. **Coupled Oscillators:** Two coupled harmonic oscillators have H = (p₁² + p₂²)/2 + (q₁² + q₂²)/2 + λq₁q₂. Find the normal mode frequencies and describe the phase space structure.

8. **Wigner Function:** Calculate the Wigner function for the ground state of the harmonic oscillator ψ₀(x) = (mω/πℏ)^(1/4) exp(-mωx²/2ℏ). Show it is everywhere positive (Gaussian).

9. **Non-Crossing Proof:** A system has two conservation laws H₁ and H₂ with {H₁, H₂} = 0. Explain geometrically why trajectories lie on the intersection of level surfaces.

---

## Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

def phase_portrait_gallery():
    """
    Generate comprehensive phase portraits for fundamental systems.
    Demonstrates fixed points, separatrices, and flow structure.
    """

    print("=" * 70)
    print("PHASE SPACE: THE GEOMETRIC ARENA OF HAMILTONIAN MECHANICS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    # =========================================
    # System 1: Simple Harmonic Oscillator
    # =========================================
    print("\n1. Simple Harmonic Oscillator: H = p²/2 + q²/2")
    print("-" * 50)

    ax = axes[0, 0]

    q = np.linspace(-2.5, 2.5, 25)
    p = np.linspace(-2.5, 2.5, 25)
    Q, P = np.meshgrid(q, p)

    # Hamilton's equations: dq/dt = ∂H/∂p = p, dp/dt = -∂H/∂q = -q
    dQ = P
    dP = -Q

    # Streamplot
    ax.streamplot(Q, P, dQ, dP, density=1.5, color='steelblue',
                  linewidth=0.8, arrowsize=0.8)

    # Energy contours
    for E in [0.5, 1.0, 2.0, 3.0]:
        theta = np.linspace(0, 2*np.pi, 100)
        q_E = np.sqrt(2*E) * np.cos(theta)
        p_E = np.sqrt(2*E) * np.sin(theta)
        ax.plot(q_E, p_E, 'darkred', lw=1.5, alpha=0.7)

    # Mark center
    ax.plot(0, 0, 'go', markersize=10, markeredgecolor='black',
            label='Center (stable)', zorder=5)

    ax.set_xlabel('q (position)', fontsize=11)
    ax.set_ylabel('p (momentum)', fontsize=11)
    ax.set_title('Simple Harmonic Oscillator\nH = p²/2 + q²/2', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    print(f"   Fixed point: (0, 0) - Center")
    print(f"   All trajectories are ellipses (circles for m=ω=1)")
    print(f"   Motion is clockwise (energy exchanges between T and V)")

    # =========================================
    # System 2: Simple Pendulum (with separatrix)
    # =========================================
    print("\n2. Simple Pendulum: H = p²/2 - cos(θ)")
    print("-" * 50)

    ax = axes[0, 1]

    theta = np.linspace(-2*np.pi, 2*np.pi, 40)
    p_theta = np.linspace(-3.5, 3.5, 40)
    THETA, P_TH = np.meshgrid(theta, p_theta)

    # Hamilton's equations (normalized: m=L=g=1)
    dTHETA = P_TH
    dP_TH = -np.sin(THETA)

    ax.streamplot(THETA, P_TH, dTHETA, dP_TH, density=2, color='steelblue',
                  linewidth=0.6, arrowsize=0.7)

    # Separatrix: E = 1 (i.e., H = 1, since minimum is H = -1)
    theta_sep = np.linspace(-np.pi + 0.001, np.pi - 0.001, 300)
    # H = p²/2 - cos(θ) = 1 → p² = 2(1 + cos(θ)) = 4cos²(θ/2)
    p_sep_upper = 2 * np.cos(theta_sep / 2)
    p_sep_lower = -2 * np.cos(theta_sep / 2)

    # Plot separatrix (with periodic copies)
    for shift in [-2*np.pi, 0, 2*np.pi]:
        ax.plot(theta_sep + shift, p_sep_upper, 'r-', lw=2.5,
                label='Separatrix' if shift == 0 else '')
        ax.plot(theta_sep + shift, p_sep_lower, 'r-', lw=2.5)

    # Fixed points
    ax.plot(0, 0, 'go', markersize=10, markeredgecolor='black',
            label='Center', zorder=5)
    ax.plot([-np.pi, np.pi], [0, 0], 'rs', markersize=10,
            markeredgecolor='black', label='Saddle', zorder=5)

    ax.set_xlabel('θ (angle)', fontsize=11)
    ax.set_ylabel('p_θ (angular momentum)', fontsize=11)
    ax.set_title('Simple Pendulum\nH = p²/2 - cos(θ)', fontsize=12)
    ax.set_xlim(-2*np.pi, 2*np.pi)
    ax.set_ylim(-3.5, 3.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    print(f"   Centers at θ = 0, ±2π, ... (hanging equilibrium)")
    print(f"   Saddles at θ = ±π (inverted equilibrium)")
    print(f"   Separatrix divides libration from rotation")

    # =========================================
    # System 3: Double Well Potential
    # =========================================
    print("\n3. Double Well: H = p²/2 + q⁴/4 - q²/2")
    print("-" * 50)

    ax = axes[0, 2]

    q = np.linspace(-2, 2, 35)
    p = np.linspace(-1.5, 1.5, 35)
    Q, P = np.meshgrid(q, p)

    # Hamilton's equations: dq/dt = p, dp/dt = -dV/dq = q - q³
    dQ = P
    dP = Q - Q**3

    ax.streamplot(Q, P, dQ, dP, density=2, color='steelblue',
                  linewidth=0.7, arrowsize=0.8)

    # Energy contours
    def H_double_well(q, p):
        return p**2/2 + q**4/4 - q**2/2

    q_fine = np.linspace(-2, 2, 300)
    p_fine = np.linspace(-1.5, 1.5, 300)
    Q_f, P_f = np.meshgrid(q_fine, p_fine)
    H_vals = H_double_well(Q_f, P_f)

    # Separatrix at E = 0 (energy at saddle point q=0)
    ax.contour(Q_f, P_f, H_vals, levels=[0], colors='red', linewidths=2.5)

    # Other energy levels
    ax.contour(Q_f, P_f, H_vals, levels=[-0.2, -0.1, 0.1, 0.3, 0.6],
               colors='darkred', linewidths=1, alpha=0.7)

    # Fixed points
    ax.plot(0, 0, 'rs', markersize=10, markeredgecolor='black',
            label='Saddle', zorder=5)
    ax.plot([-1, 1], [0, 0], 'go', markersize=10, markeredgecolor='black',
            label='Centers', zorder=5)

    ax.set_xlabel('q', fontsize=11)
    ax.set_ylabel('p', fontsize=11)
    ax.set_title('Double Well Potential\nH = p²/2 + q⁴/4 - q²/2', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    print(f"   Saddle at (0, 0) with E = 0")
    print(f"   Centers at (±1, 0) with E = -1/4")
    print(f"   Separatrix (figure-8) divides left/right oscillations from full motion")

    # =========================================
    # System 4: Inverted Oscillator (Saddle)
    # =========================================
    print("\n4. Inverted Oscillator: H = p²/2 - q²/2")
    print("-" * 50)

    ax = axes[1, 0]

    q = np.linspace(-2, 2, 25)
    p = np.linspace(-2, 2, 25)
    Q, P = np.meshgrid(q, p)

    dQ = P
    dP = Q  # Repulsive force!

    ax.streamplot(Q, P, dQ, dP, density=1.5, color='steelblue',
                  linewidth=0.8, arrowsize=0.8)

    # Energy contours (hyperbolas)
    for E in [-1, -0.5, 0.5, 1]:
        if E > 0:
            q_pos = np.linspace(np.sqrt(2*E), 2, 100)
            q_neg = np.linspace(-2, -np.sqrt(2*E), 100)
            p_pos = np.sqrt(2*E + q_pos**2)
            p_neg_branch = np.sqrt(2*E + q_neg**2)
            ax.plot(q_pos, p_pos, 'darkred', lw=1.2, alpha=0.7)
            ax.plot(q_pos, -p_pos, 'darkred', lw=1.2, alpha=0.7)
            ax.plot(q_neg, p_neg_branch, 'darkred', lw=1.2, alpha=0.7)
            ax.plot(q_neg, -p_neg_branch, 'darkred', lw=1.2, alpha=0.7)
        else:
            p_range = np.linspace(np.sqrt(-2*E), 2, 100)
            q_hyp = np.sqrt(p_range**2 + 2*E)
            ax.plot(q_hyp, p_range, 'darkred', lw=1.2, alpha=0.7)
            ax.plot(-q_hyp, p_range, 'darkred', lw=1.2, alpha=0.7)
            ax.plot(q_hyp, -p_range, 'darkred', lw=1.2, alpha=0.7)
            ax.plot(-q_hyp, -p_range, 'darkred', lw=1.2, alpha=0.7)

    # Separatrices (asymptotes): E = 0 → p = ±q
    q_line = np.linspace(-2, 2, 100)
    ax.plot(q_line, q_line, 'r-', lw=2.5, label='Separatrix (p = q)')
    ax.plot(q_line, -q_line, 'r-', lw=2.5, label='Separatrix (p = -q)')

    ax.plot(0, 0, 'rs', markersize=10, markeredgecolor='black',
            label='Saddle', zorder=5)

    ax.set_xlabel('q', fontsize=11)
    ax.set_ylabel('p', fontsize=11)
    ax.set_title('Inverted Oscillator (Saddle)\nH = p²/2 - q²/2', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    print(f"   Single saddle point at origin")
    print(f"   Trajectories are hyperbolas")
    print(f"   Separatrices are the lines p = ±q")

    # =========================================
    # System 5: Liouville's Theorem Demonstration
    # =========================================
    print("\n5. Liouville's Theorem: Area Preservation")
    print("-" * 50)

    ax = axes[1, 1]

    # Initial cloud of points (circular region)
    n_points = 800
    theta_init = np.random.uniform(0, 2*np.pi, n_points)
    r_init = np.sqrt(np.random.uniform(0, 1, n_points)) * 0.4

    q0_cloud = 1.0 + r_init * np.cos(theta_init)
    p0_cloud = 0.5 + r_init * np.sin(theta_init)

    # Evolve under SHO
    omega = 1.0
    times = [0, np.pi/3, 2*np.pi/3, np.pi]
    colors = ['blue', 'green', 'orange', 'red']

    for t, color in zip(times, colors):
        q_t = q0_cloud * np.cos(omega * t) + p0_cloud * np.sin(omega * t)
        p_t = -q0_cloud * np.sin(omega * t) + p0_cloud * np.cos(omega * t)
        ax.scatter(q_t, p_t, s=2, alpha=0.6, c=color, label=f't = {t/np.pi:.1f}π')

    ax.set_xlabel('q', fontsize=11)
    ax.set_ylabel('p', fontsize=11)
    ax.set_title("Liouville's Theorem\nArea preserved, shape changes", fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    print(f"   Initial circular cloud evolves under SHO flow")
    print(f"   Area remains constant (≈ π × 0.4² = 0.503)")
    print(f"   Shape rotates but total area preserved")

    # =========================================
    # System 6: Wigner Function for Ground State
    # =========================================
    print("\n6. Wigner Function: Quantum Phase Space")
    print("-" * 50)

    ax = axes[1, 2]

    # Wigner function for SHO ground state (in natural units ℏ=m=ω=1)
    # W(x,p) = (1/π) exp(-x² - p²)
    x = np.linspace(-3, 3, 200)
    p = np.linspace(-3, 3, 200)
    X, P_w = np.meshgrid(x, p)

    W_ground = (1/np.pi) * np.exp(-X**2 - P_w**2)

    # Plot
    contour = ax.contourf(X, P_w, W_ground, levels=50, cmap='RdYlBu_r')
    plt.colorbar(contour, ax=ax, label='W(x,p)')

    # Add uncertainty circle (Δx·Δp = ℏ/2 = 1/2 in natural units)
    circle = plt.Circle((0, 0), np.sqrt(0.5), fill=False, color='white',
                        linestyle='--', linewidth=2, label='Δx·Δp = ℏ/2')
    ax.add_patch(circle)

    ax.set_xlabel('x (position)', fontsize=11)
    ax.set_ylabel('p (momentum)', fontsize=11)
    ax.set_title('Wigner Function: SHO Ground State\nW(x,p) = (1/π)exp(-x²-p²)', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    print(f"   Wigner function for quantum ground state")
    print(f"   Gaussian distribution centered at origin")
    print(f"   Width determined by uncertainty principle")
    print(f"   This state is 'as classical as possible' (minimum uncertainty)")

    plt.tight_layout()
    plt.savefig('phase_space_gallery.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 70)
    print("Figure saved as 'phase_space_gallery.png'")


def trajectory_animation_demo():
    """
    Demonstrate trajectory evolution and non-crossing property.
    """

    print("\n" + "=" * 70)
    print("TRAJECTORY EVOLUTION AND NON-CROSSING PROPERTY")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Multiple trajectories for pendulum
    ax = axes[0]

    def pendulum_eom(state, t):
        theta, p = state
        return [p, -np.sin(theta)]

    t = np.linspace(0, 20, 2000)

    # Various initial conditions
    initial_conditions = [
        (0.5, 0), (1.0, 0), (1.5, 0), (2.0, 0), (2.5, 0),  # Libration
        (0.1, 2.5), (0.1, 3.0),  # Rotation
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))

    for (theta0, p0), color in zip(initial_conditions, colors):
        sol = odeint(pendulum_eom, [theta0, p0], t)
        ax.plot(sol[:, 0], sol[:, 1], '-', color=color, lw=0.8, alpha=0.8)
        ax.plot(theta0, p0, 'o', color=color, markersize=6)

    # Add separatrix
    theta_sep = np.linspace(-np.pi + 0.01, np.pi - 0.01, 300)
    p_sep = 2 * np.cos(theta_sep / 2)
    ax.plot(theta_sep, p_sep, 'r--', lw=2, label='Separatrix')
    ax.plot(theta_sep, -p_sep, 'r--', lw=2)

    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('p_θ', fontsize=12)
    ax.set_title('Pendulum: Trajectories Never Cross\n(except at fixed points)', fontsize=12)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Time evolution showing determinism
    ax = axes[1]

    def sho_eom(state, t):
        q, p = state
        return [p, -q]

    # Two very close initial conditions
    ic1 = [1.0, 0.0]
    ic2 = [1.001, 0.0]  # Tiny perturbation

    t = np.linspace(0, 4*np.pi, 1000)

    sol1 = odeint(sho_eom, ic1, t)
    sol2 = odeint(sho_eom, ic2, t)

    ax.plot(sol1[:, 0], sol1[:, 1], 'b-', lw=2, label='IC: (1.000, 0)')
    ax.plot(sol2[:, 0], sol2[:, 1], 'r--', lw=2, label='IC: (1.001, 0)')

    ax.plot(ic1[0], ic1[1], 'bo', markersize=10)
    ax.plot(ic2[0], ic2[1], 'ro', markersize=10)

    ax.set_xlabel('q', fontsize=12)
    ax.set_ylabel('p', fontsize=12)
    ax.set_title('SHO: Nearby Initial Conditions\nStay on distinct trajectories', fontsize=12)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_non_crossing.png', dpi=150)
    plt.show()


def wigner_function_states():
    """
    Compare Wigner functions for different quantum states.
    """

    print("\n" + "=" * 70)
    print("WIGNER FUNCTIONS: QUANTUM STATES IN PHASE SPACE")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.linspace(-4, 4, 200)
    p = np.linspace(-4, 4, 200)
    X, P = np.meshgrid(x, p)

    # Ground state |0⟩: W = (1/π) exp(-x² - p²)
    W0 = (1/np.pi) * np.exp(-X**2 - P**2)

    # First excited state |1⟩: W = (1/π)(2(x² + p²) - 1) exp(-x² - p²)
    W1 = (1/np.pi) * (2*(X**2 + P**2) - 1) * np.exp(-X**2 - P**2)

    # Coherent state |α⟩ with α = 2 (displaced Gaussian)
    x0, p0 = 2, 0
    W_coh = (1/np.pi) * np.exp(-(X-x0)**2 - (P-p0)**2)

    # Plot ground state
    ax = axes[0]
    im = ax.contourf(X, P, W0, levels=50, cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title('Ground State |0⟩\nW ≥ 0 everywhere (Gaussian)')
    ax.set_aspect('equal')

    # Plot first excited state (has negative regions!)
    ax = axes[1]
    vmax = np.abs(W1).max()
    im = ax.contourf(X, P, W1, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.contour(X, P, W1, levels=[0], colors='black', linewidths=2)
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title('First Excited |1⟩\nW < 0 near origin (non-classical!)')
    ax.set_aspect('equal')

    # Plot coherent state
    ax = axes[2]
    im = ax.contourf(X, P, W_coh, levels=50, cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.plot(x0, p0, 'k*', markersize=15, label=f'α = {x0}')
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title('Coherent State |α⟩\nDisplaced Gaussian (most classical)')
    ax.set_aspect('equal')
    ax.legend()

    plt.tight_layout()
    plt.savefig('wigner_functions.png', dpi=150)
    plt.show()

    print("\nKey observations:")
    print("  • Ground state: W ≥ 0, minimum uncertainty state")
    print("  • Excited states: W < 0 in some regions → non-classical")
    print("  • Coherent states: W ≥ 0, closest to classical behavior")
    print("  • Negative W indicates quantum interference (no classical analog)")


# Run all demonstrations
if __name__ == "__main__":
    phase_portrait_gallery()
    trajectory_animation_demo()
    wigner_function_states()

    print("\n" + "=" * 70)
    print("PHASE SPACE LAB COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Phase space is 2n-dimensional (positions + momenta)")
    print("  2. Trajectories never cross → determinism")
    print("  3. Area/volume is preserved (Liouville's theorem)")
    print("  4. Fixed points: centers (stable) and saddles (unstable)")
    print("  5. Separatrices divide qualitatively different motions")
    print("  6. Quantum mechanics: Wigner function, uncertainty cells ~ ℏ")
```

---

## Summary

### Phase Space Fundamentals

| Concept | Definition |
|---------|------------|
| Phase space | 2n-dimensional manifold of (q₁,...,qₙ,p₁,...,pₙ) |
| Symplectic form | ω = Σ dpᵢ ∧ dqᵢ |
| Hamilton's equations | ż = J∇H |
| Hamiltonian flow | φᵗ: maps initial to final state |

### Key Theorems

$$\boxed{\text{Existence \& Uniqueness: Unique trajectory through each point}}$$

$$\boxed{\text{Non-Crossing: Trajectories cannot intersect (except at fixed points)}}$$

$$\boxed{\text{Area Preservation: } \det(D\phi^t) = 1}$$

### Fixed Point Classification

| Type | Eigenvalues | Stability |
|------|-------------|-----------|
| Center | ±iω | Stable |
| Saddle | ±λ (real) | Unstable |

### Quantum Connection

$$\boxed{\Delta q \cdot \Delta p \geq \frac{\hbar}{2} \quad \text{(Phase space cell)}}$$

$$\boxed{W(x,p) = \frac{1}{\pi\hbar}\int \psi^*(x+y)\psi(x-y)e^{2ipy/\hbar}dy \quad \text{(Wigner function)}}$$

---

## Daily Checklist

- [ ] Define phase space dimension for multi-particle systems
- [ ] Explain why trajectories cannot cross
- [ ] Construct phase portrait for SHO
- [ ] Find and classify fixed points
- [ ] Identify separatrices for pendulum
- [ ] Calculate enclosed phase space area
- [ ] Explain uncertainty principle in phase space language
- [ ] Describe Wigner function properties
- [ ] Run computational lab visualizations

---

## Preview: Day 158

Tomorrow we introduce the **Poisson Bracket** — the fundamental algebraic operation on phase space that encodes the structure of Hamiltonian mechanics. You'll discover how {q, p} = 1 becomes the quantum commutator [q̂, p̂] = iℏ, completing the bridge between classical and quantum mechanics!
