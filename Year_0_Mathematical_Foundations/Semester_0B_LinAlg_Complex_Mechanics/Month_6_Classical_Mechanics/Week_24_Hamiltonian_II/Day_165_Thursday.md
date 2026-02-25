# Day 165: The Hamilton-Jacobi Equation — Where Classical Meets Quantum

## Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Hamilton-Jacobi Equation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## Learning Objectives

By the end of today, you should be able to:

1. Derive the Hamilton-Jacobi equation from canonical transformation theory
2. Distinguish between Hamilton's principal function S and characteristic function W
3. Apply separation of variables to solve the HJ equation for standard systems
4. Extract trajectories from a complete integral
5. Connect the HJ equation to the Schrödinger equation via S → iℏ ln ψ
6. Understand Hamilton's optical-mechanical analogy

---

## Core Content

### 1. Motivation: The Ultimate Simplification

We've seen that canonical transformations can simplify the Hamiltonian. What is the **most extreme** simplification possible?

**The Dream:** Find coordinates (Q, P) where the new Hamiltonian K = 0!

If K = 0, then Hamilton's equations become:

$$\dot{Q}_i = \frac{\partial K}{\partial P_i} = 0, \quad \dot{P}_i = -\frac{\partial K}{\partial Q_i} = 0$$

Both Q and P are **constants**! The problem is completely solved once we find this transformation.

**The Key Question:** What generating function produces K = 0?

---

### 2. Derivation of the Hamilton-Jacobi Equation

**Starting Point:** For a Type-2 generating function F₂(q, P, t):

$$p_i = \frac{\partial F_2}{\partial q_i}, \quad Q_i = \frac{\partial F_2}{\partial P_i}, \quad K = H + \frac{\partial F_2}{\partial t}$$

**The Requirement K = 0:**

$$H + \frac{\partial F_2}{\partial t} = 0$$

$$H\left(q_1, \ldots, q_n, \frac{\partial F_2}{\partial q_1}, \ldots, \frac{\partial F_2}{\partial q_n}, t\right) + \frac{\partial F_2}{\partial t} = 0$$

**Notation:** Call the generating function S (for "action"):

$$\boxed{\frac{\partial S}{\partial t} + H\left(q, \frac{\partial S}{\partial q}, t\right) = 0}$$

This is the **Hamilton-Jacobi equation** — a first-order partial differential equation for S(q, t).

**Explicit Form:**

For H = T + V with T = p²/(2m):

$$\frac{\partial S}{\partial t} + \frac{1}{2m}\left(\frac{\partial S}{\partial q}\right)^2 + V(q) = 0$$

---

### 3. Hamilton's Principal Function S

**Definition:** The function S(q, P, t) that solves the HJ equation is called **Hamilton's principal function**.

**Physical Meaning:**

Along an actual trajectory:

$$\frac{dS}{dt} = \frac{\partial S}{\partial t} + \sum_i \frac{\partial S}{\partial q_i}\dot{q}_i = -H + \sum_i p_i \dot{q}_i = L$$

Therefore:

$$\boxed{S = \int L \, dt}$$

**Hamilton's principal function equals the classical action!**

**Properties:**

| Property | Statement |
|----------|-----------|
| Relation to momentum | $p_i = \partial S / \partial q_i$ |
| Relation to energy | $-E = \partial S / \partial t$ (for conservative systems) |
| Action integral | S = ∫L dt along the trajectory |

---

### 4. Complete Integrals and Extracting Trajectories

**Definition:** A **complete integral** of the HJ equation is a solution containing n independent constants of integration α₁, ..., αₙ (plus an additive constant):

$$S = S(q_1, \ldots, q_n, \alpha_1, \ldots, \alpha_n, t)$$

**Jacobi's Theorem:** The trajectories are found by setting:

$$\beta_i = \frac{\partial S}{\partial \alpha_i} = \text{const}$$

This gives n equations that can be solved for q_i(t).

**Why This Works:**

The new momenta are P_i = α_i (constants by construction). The new coordinates are:

$$Q_i = \frac{\partial S}{\partial P_i} = \frac{\partial S}{\partial \alpha_i} = \beta_i$$

Since K = 0, both Q_i and P_i are constants. The equations Q_i = β_i implicitly define the motion.

---

### 5. Hamilton's Characteristic Function W

For **time-independent** Hamiltonians, we can separate the time dependence:

$$S(q, \alpha, t) = W(q, \alpha) - Et$$

where E = α₁ is the energy.

**The Reduced HJ Equation:**

$$\boxed{H\left(q, \frac{\partial W}{\partial q}\right) = E}$$

This is an equation for **Hamilton's characteristic function** W(q, E, α₂, ..., αₙ).

**Comparison:**

| Function | Equation | Time Dependence |
|----------|----------|-----------------|
| S (principal) | ∂S/∂t + H = 0 | Contains t explicitly |
| W (characteristic) | H = E | Time-independent; S = W - Et |

---

### 6. Separation of Variables

The HJ equation is a PDE, but for many important systems it can be reduced to ODEs through **separation of variables**.

**Separability Condition:**

The HJ equation is separable in coordinates (q₁, ..., qₙ) if we can write:

$$W(q_1, \ldots, q_n) = W_1(q_1) + W_2(q_2) + \cdots + W_n(q_n)$$

This is possible when each term in H involves only one coordinate and its conjugate momentum.

**The Stäckel Conditions:**

A Hamiltonian is separable in orthogonal coordinates if the kinetic energy takes the form:

$$T = \frac{1}{2}\sum_{i=1}^n \frac{p_i^2}{f_i(q_i)} \cdot g(q_1, \ldots, q_n)$$

and the potential satisfies certain conditions related to the coordinate system.

**Common Separable Coordinate Systems:**

| System | Coordinates | Applications |
|--------|-------------|--------------|
| Cartesian | (x, y, z) | Free particle, constant fields |
| Spherical | (r, θ, φ) | Central forces, hydrogen atom |
| Cylindrical | (ρ, φ, z) | Axial symmetry |
| Parabolic | (ξ, η, φ) | Stark effect |
| Elliptic | (u, v, φ) | Two-center problem |

---

### 7. Example 1: Free Particle

**Hamiltonian:**

$$H = \frac{p_x^2 + p_y^2 + p_z^2}{2m}$$

**HJ Equation:**

$$\frac{\partial S}{\partial t} + \frac{1}{2m}\left[\left(\frac{\partial S}{\partial x}\right)^2 + \left(\frac{\partial S}{\partial y}\right)^2 + \left(\frac{\partial S}{\partial z}\right)^2\right] = 0$$

**Separation:** Try S = S_x(x) + S_y(y) + S_z(z) - Et:

$$\frac{1}{2m}\left[(S_x')^2 + (S_y')^2 + (S_z')^2\right] = E$$

This separates into:

$$\frac{(S_x')^2}{2m} = \alpha_x, \quad \frac{(S_y')^2}{2m} = \alpha_y, \quad \frac{(S_z')^2}{2m} = \alpha_z$$

with E = α_x + α_y + α_z.

**Solution:**

$$S_x = \pm\sqrt{2m\alpha_x} \cdot x, \quad \text{etc.}$$

$$\boxed{S = \mathbf{p} \cdot \mathbf{r} - Et}$$

where **p** = (p_x, p_y, p_z) with p_i = √(2mα_i).

**Trajectories:**

$$\beta_x = \frac{\partial S}{\partial p_x} = x - \frac{p_x}{m}t$$

So x = β_x + (p_x/m)t — uniform motion!

---

### 8. Example 2: Harmonic Oscillator

**Hamiltonian:**

$$H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 q^2$$

**HJ Equation (characteristic function):**

$$\frac{1}{2m}\left(\frac{dW}{dq}\right)^2 + \frac{1}{2}m\omega^2 q^2 = E$$

**Solve for W:**

$$\frac{dW}{dq} = \sqrt{2mE - m^2\omega^2 q^2} = m\omega\sqrt{\frac{2E}{m\omega^2} - q^2}$$

$$W = \int m\omega\sqrt{a^2 - q^2} \, dq$$

where a² = 2E/(mω²).

$$W = \frac{m\omega}{2}\left[q\sqrt{a^2 - q^2} + a^2 \arcsin\left(\frac{q}{a}\right)\right]$$

**Extract Trajectory:**

$$\beta = \frac{\partial W}{\partial E} = \frac{\partial W}{\partial a^2} \cdot \frac{\partial a^2}{\partial E} = \frac{m\omega}{2} \cdot \arcsin\left(\frac{q}{a}\right) \cdot \frac{2}{m\omega^2}$$

$$\beta = \frac{1}{\omega}\arcsin\left(\frac{q}{a}\right)$$

From S = W - Et:

$$\beta + t_0 = \frac{1}{\omega}\arcsin\left(\frac{q}{a}\right) - Et/\omega + t_0$$

Taking β = t₀/ω and rearranging:

$$q = a\sin(\omega t + \phi_0) = \sqrt{\frac{2E}{m\omega^2}}\sin(\omega t + \phi_0)$$

**We've derived the solution without solving a differential equation!**

---

### 9. Example 3: The Kepler Problem

**Hamiltonian in Spherical Coordinates:**

$$H = \frac{p_r^2}{2m} + \frac{p_\theta^2}{2mr^2} + \frac{p_\phi^2}{2mr^2\sin^2\theta} - \frac{k}{r}$$

**Separation:**

Try W = W_r(r) + W_θ(θ) + W_φ(φ):

**Step 1:** φ is cyclic, so:

$$\frac{\partial W}{\partial \phi} = p_\phi = \alpha_\phi = L_z \quad (\text{const})$$

$$W_\phi = L_z \phi$$

**Step 2:** The θ equation:

$$\left(\frac{dW_\theta}{d\theta}\right)^2 + \frac{L_z^2}{\sin^2\theta} = \alpha_\theta^2$$

where α_θ = L (total angular momentum).

**Step 3:** The radial equation:

$$\frac{1}{2m}\left(\frac{dW_r}{dr}\right)^2 + \frac{L^2}{2mr^2} - \frac{k}{r} = E$$

$$W_r = \int \sqrt{2m\left(E + \frac{k}{r}\right) - \frac{L^2}{r^2}} \, dr$$

**Complete Integral:**

$$W = L_z \phi + \int \sqrt{L^2 - \frac{L_z^2}{\sin^2\theta}} \, d\theta + \int \sqrt{2mE + \frac{2mk}{r} - \frac{L^2}{r^2}} \, dr$$

**From this, all Kepler orbits can be extracted!**

---

## Quantum Mechanics Connection

### The Hamilton-Jacobi → Schrödinger Connection

This is perhaps the most profound connection in all of physics.

**The Ansatz:**

In quantum mechanics, write the wave function as:

$$\psi = Ae^{iS/\hbar}$$

where A and S are real functions.

**Substituting into Schrödinger's equation:**

$$i\hbar\frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi + V\psi$$

**Expanding:**

The time derivative:
$$i\hbar\frac{\partial \psi}{\partial t} = \left(i\hbar\frac{\partial A}{\partial t} - A\frac{\partial S}{\partial t}\right)e^{iS/\hbar}$$

The Laplacian:
$$\nabla^2\psi = \left[\nabla^2 A + \frac{2i}{\hbar}(\nabla A)\cdot(\nabla S) + \frac{i}{\hbar}A\nabla^2 S - \frac{A}{\hbar^2}|\nabla S|^2\right]e^{iS/\hbar}$$

**Separating real and imaginary parts:**

**Real part (O(ℏ⁰)):**

$$\frac{\partial S}{\partial t} + \frac{|\nabla S|^2}{2m} + V = 0$$

**This is exactly the Hamilton-Jacobi equation!**

**Imaginary part (O(ℏ¹)):**

$$\frac{\partial A^2}{\partial t} + \nabla \cdot \left(\frac{A^2 \nabla S}{m}\right) = 0$$

This is a **continuity equation** for probability density ρ = A²!

### The Classical Limit

As ℏ → 0, the quantum wave function oscillates infinitely rapidly except where S is stationary. The leading-order dynamics is governed by the HJ equation.

**The Correspondence:**

| Quantum | Classical |
|---------|-----------|
| ψ = Ae^{iS/ℏ} | S = Hamilton's principal function |
| |ψ|² = probability | Ensemble of trajectories |
| Schrödinger equation | Hamilton-Jacobi equation |
| ℏ → 0 | Classical limit |

### The WKB Approximation

For slowly-varying potentials, the WKB (Wentzel-Kramers-Brillouin) approximation gives:

$$\psi(x) \approx \frac{C}{\sqrt{p(x)}} \exp\left(\pm\frac{i}{\hbar}\int p(x) \, dx\right)$$

where p(x) = √(2m(E - V(x))) is the classical momentum.

**Connection to Action:**

$$\int p \, dx = W = \text{Hamilton's characteristic function}$$

The quantum phase is directly related to the classical action!

---

### Hamilton's Optical-Mechanical Analogy

Hamilton discovered the HJ equation by studying **optics**. The analogy runs deep:

| Mechanics | Optics |
|-----------|--------|
| Action S | Optical path length (eikonal) |
| Momentum p = ∂S/∂q | Wave vector k = ∂φ/∂x |
| Hamilton-Jacobi eq. | Eikonal equation |
| Particle trajectories | Light rays |
| Fermat's principle | Principle of least action |
| m (mass) | n (refractive index) |

**The Eikonal Equation (Geometric Optics):**

$$|\nabla \phi|^2 = n^2(x)$$

where φ is the phase and n is the refractive index.

**Compare with HJ:**

$$\frac{1}{2m}|\nabla S|^2 + V = E$$

$$|\nabla S|^2 = 2m(E - V)$$

**The analogy:** √(2m(E - V)) plays the role of refractive index!

**Historical Significance:** This analogy inspired Schrödinger. Just as geometric optics emerges from wave optics in the short-wavelength limit, classical mechanics emerges from quantum mechanics in the small-ℏ limit.

---

## Worked Examples

### Example 1: Particle in a Uniform Gravitational Field

**Problem:** Solve the HJ equation for a particle falling under gravity: H = p²/(2m) + mgz.

**Solution:**

**HJ Equation:**

$$\frac{\partial S}{\partial t} + \frac{1}{2m}\left(\frac{\partial S}{\partial z}\right)^2 + mgz = 0$$

**Separation:** Try S = W(z) - Et:

$$\frac{1}{2m}\left(\frac{dW}{dz}\right)^2 + mgz = E$$

$$\frac{dW}{dz} = \sqrt{2m(E - mgz)} = \sqrt{2mE - 2m^2gz}$$

**Integrate:**

$$W = -\frac{1}{3m^2g}\left(2mE - 2m^2gz\right)^{3/2} + C$$

$$W = -\frac{(2m(E-mgz))^{3/2}}{3m^2g}$$

**Extract trajectory:**

$$\beta = \frac{\partial S}{\partial E} = \frac{\partial W}{\partial E} - t = -\frac{\sqrt{2m(E-mgz)}}{mg} - t$$

Let β = -t₀:

$$t - t_0 = -\frac{\sqrt{2m(E-mgz)}}{mg}$$

$$\sqrt{2m(E-mgz)} = -mg(t-t_0)$$

This only works for t < t₀. Let's be more careful with signs. If the particle starts at z₀ with velocity v₀ downward:

$$z = z_0 + v_0 t - \frac{1}{2}gt^2$$

The HJ method confirms free-fall motion.

---

### Example 2: Time-Dependent Harmonic Oscillator

**Problem:** For H = p²/(2m) + (1/2)m ω(t)² q², find S when ω changes slowly.

**Solution:**

For slowly-varying ω, use the **adiabatic approximation**. The action J = E/ω is approximately constant.

**Ansatz:**

$$S \approx \int p \, dq - \int E(t) \, dt = J \cdot \theta(t) - \int \omega(t) J \, dt$$

where θ = ωt is the angle variable.

**Result:**

$$S = J\theta - J\int_0^t \omega(t') \, dt'$$

The momentum is:

$$p = \frac{\partial S}{\partial q} = J\frac{\partial \theta}{\partial q}$$

This shows how the classical action-angle structure persists under slow parameter changes.

---

### Example 3: Central Force in 2D

**Problem:** For H = (p_r² + p_θ²/r²)/(2m) + V(r), find W.

**Solution:**

**Separation:** W = W_r(r) + W_θ(θ)

Since θ is cyclic:
$$\frac{\partial W}{\partial \theta} = p_\theta = L = \text{const}$$
$$W_\theta = L\theta$$

**Radial equation:**

$$\frac{1}{2m}\left(\frac{dW_r}{dr}\right)^2 + \frac{L^2}{2mr^2} + V(r) = E$$

$$W_r = \int \sqrt{2m(E - V(r)) - \frac{L^2}{r^2}} \, dr$$

**Complete integral:**

$$W = L\theta + \int \sqrt{2m(E - V(r)) - \frac{L^2}{r^2}} \, dr$$

**Trajectory from ∂W/∂L = β:**

$$\theta + \frac{\partial}{\partial L}\int \sqrt{2m(E-V) - L^2/r^2} \, dr = \beta$$

This gives the orbit equation θ(r).

---

## Practice Problems

### Level 1: Direct Application

1. **Free particle in 1D:** Verify that S = px - Et solves the HJ equation for H = p²/(2m) and extract the trajectory.

2. **Constant force:** For H = p²/(2m) - Fq, find W and show the motion is uniformly accelerated.

3. **Harmonic oscillator energy:** From the complete integral W for the HO, verify that ∂W/∂E gives the period T = 2π/ω.

### Level 2: Intermediate

4. **Particle on a sphere:** For a free particle on a sphere of radius R, write the HJ equation in spherical coordinates and solve it.

5. **Isotropic oscillator:** For H = (p_x² + p_y²)/(2m) + (1/2)mω²(x² + y²), find W in Cartesian and polar coordinates.

6. **Relativistic free particle:** The relativistic Hamiltonian is H = √(p²c² + m²c⁴). Find S and show the trajectory is x = vt with v = pc/E.

### Level 3: Challenging

7. **Kepler orbit equation:** From the complete integral for the Kepler problem, derive the orbit equation r = p/(1 + e cos θ).

8. **Stark effect setup:** For hydrogen in an electric field, H = p²/(2m) - e²/r - eεz, write the HJ equation in parabolic coordinates (ξ, η, φ) where x + iy = √(ξη)e^{iφ}, z = (ξ - η)/2.

9. **WKB connection:** Starting from ψ = Ae^{iS/ℏ}, derive the next-order quantum correction to the HJ equation.

---

## Computational Lab

### Lab 1: Solving the HJ Equation Numerically

```python
"""
Day 165 Lab: Hamilton-Jacobi Equation
Numerical computation of Hamilton's characteristic function
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp

def characteristic_function_harmonic(q, E, m=1.0, omega=1.0):
    """
    Compute W(q, E) for harmonic oscillator.
    W = ∫ p dq where p = sqrt(2m(E - V))
    """
    V = 0.5 * m * omega**2 * q**2

    if E <= V:
        return 0

    p = np.sqrt(2*m*(E - V))
    return p  # This is dW/dq


def compute_W_harmonic(q_final, E, m=1.0, omega=1.0, n_points=1000):
    """Integrate to get W from q=0 to q=q_final."""
    q_max = np.sqrt(2*E / (m*omega**2))  # Turning point

    if abs(q_final) > q_max:
        return None

    q_vals = np.linspace(0, q_final, n_points)
    p_vals = np.array([characteristic_function_harmonic(q, E, m, omega)
                       for q in q_vals])

    W = np.trapz(p_vals, q_vals)
    return W


def plot_W_surface():
    """Plot W(q, E) as a surface."""

    m, omega = 1.0, 1.0

    E_vals = np.linspace(0.1, 2.0, 50)
    q_vals = np.linspace(-2, 2, 100)

    W_grid = np.zeros((len(E_vals), len(q_vals)))

    for i, E in enumerate(E_vals):
        q_max = np.sqrt(2*E / (m*omega**2))
        for j, q in enumerate(q_vals):
            if abs(q) < q_max:
                # Integrate from 0 to q
                result, _ = quad(
                    lambda qp: np.sqrt(max(0, 2*m*(E - 0.5*m*omega**2*qp**2))),
                    0, q
                )
                W_grid[i, j] = result
            else:
                W_grid[i, j] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Contour plot
    ax = axes[0]
    Q, En = np.meshgrid(q_vals, E_vals)
    cs = ax.contourf(Q, En, W_grid, levels=30, cmap='viridis')
    plt.colorbar(cs, ax=ax, label='W(q, E)')
    ax.set_xlabel('q')
    ax.set_ylabel('E')
    ax.set_title("Hamilton's Characteristic Function W(q, E)\nHarmonic Oscillator")

    # Plot turning points
    q_turn = np.sqrt(2*E_vals / (m*omega**2))
    ax.plot(q_turn, E_vals, 'r--', lw=2, label='Turning points')
    ax.plot(-q_turn, E_vals, 'r--', lw=2)
    ax.legend()

    # W vs q for fixed E
    ax = axes[1]
    E_fixed = [0.5, 1.0, 1.5]
    colors = ['blue', 'green', 'red']

    for E, color in zip(E_fixed, colors):
        q_max = np.sqrt(2*E / (m*omega**2))
        q_range = np.linspace(-q_max*0.99, q_max*0.99, 100)
        W_vals = []
        for q in q_range:
            result, _ = quad(
                lambda qp: np.sqrt(max(0, 2*m*(E - 0.5*m*omega**2*qp**2))),
                0, q
            )
            W_vals.append(result)

        ax.plot(q_range, W_vals, color=color, lw=2, label=f'E = {E}')

    ax.set_xlabel('q')
    ax.set_ylabel('W(q)')
    ax.set_title('W(q) for Different Energies')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hamilton_characteristic_function.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_W_surface()
```

### Lab 2: Trajectories from the HJ Solution

```python
"""
Extract trajectories from Hamilton-Jacobi solution.
"""

def extract_trajectory_harmonic(E, m=1.0, omega=1.0, t_max=10, n_points=500):
    """
    Use the HJ solution to extract q(t) for harmonic oscillator.

    β = ∂W/∂E = const
    S = W - Et
    β = ∂S/∂E + t = ∂W/∂E - t
    """
    t_vals = np.linspace(0, t_max, n_points)
    q_vals = []

    A = np.sqrt(2*E / (m*omega**2))  # Amplitude

    for t in t_vals:
        # From HJ: t - t0 = (1/ω) arcsin(q/A)
        # So: q = A sin(ω(t - t0))
        # Taking t0 = 0 and initial condition q(0) = 0:
        q = A * np.sin(omega * t)
        q_vals.append(q)

    return t_vals, np.array(q_vals)


def compare_hj_with_direct():
    """Compare HJ-derived trajectory with direct solution."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    m, omega = 1.0, 1.0
    E = 1.0
    t_max = 4 * np.pi / omega

    # HJ solution
    t_hj, q_hj = extract_trajectory_harmonic(E, m, omega, t_max)

    # Direct solution from Hamilton's equations
    def hamilton_eqs(t, y):
        q, p = y
        return [p/m, -m*omega**2*q]

    q0, p0 = 0, np.sqrt(2*m*E)  # Start at q=0 with max momentum
    sol = solve_ivp(hamilton_eqs, [0, t_max], [q0, p0],
                    t_eval=t_hj, method='RK45')

    # Plot comparison
    ax = axes[0]
    ax.plot(t_hj, q_hj, 'b-', lw=2, label='From HJ equation')
    ax.plot(sol.t, sol.y[0], 'r--', lw=2, label='Direct integration')
    ax.set_xlabel('t')
    ax.set_ylabel('q')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase space
    ax = axes[1]
    p_hj = m * omega * np.sqrt(2*E/(m*omega**2)) * np.cos(omega * t_hj)
    ax.plot(q_hj, p_hj, 'b-', lw=2, label='From HJ')
    ax.plot(sol.y[0], sol.y[1], 'r--', lw=2, label='Direct')
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title('Phase Space')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hj_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_hj_with_direct()
```

### Lab 3: The WKB Approximation

```python
"""
Demonstrate the WKB approximation and its connection to HJ.
"""

def wkb_wavefunction(x, E, V_func, m=1.0, hbar=1.0):
    """
    Compute WKB wavefunction: ψ ~ (1/√p) exp(±i∫p dx / ℏ)

    For classically allowed regions where E > V(x).
    """
    V = V_func(x)

    if E > V:
        p = np.sqrt(2*m*(E - V))
        return 1.0 / np.sqrt(p), p
    else:
        # Classically forbidden
        kappa = np.sqrt(2*m*(V - E))
        return np.exp(-kappa * abs(x) / hbar), 0


def plot_wkb_vs_exact():
    """Compare WKB with exact quantum solution for harmonic oscillator."""

    m, omega, hbar = 1.0, 1.0, 1.0

    # Exact quantum energies
    n_levels = 5
    E_exact = hbar * omega * (np.arange(n_levels) + 0.5)

    # For n=4 state
    n = 4
    E = E_exact[n]

    x = np.linspace(-4, 4, 1000)
    V = 0.5 * m * omega**2 * x**2

    # Classical turning points
    x_turn = np.sqrt(2*E / (m*omega**2))

    # WKB approximation (simplified)
    def wkb_amplitude(x_val):
        if abs(x_val) < x_turn:
            p = np.sqrt(2*m*(E - 0.5*m*omega**2*x_val**2))
            return 1.0 / np.sqrt(p)
        else:
            return 0

    # Phase integral
    def phase_integral(x_val):
        if x_val < -x_turn or x_val > x_turn:
            return 0
        # Integrate from -x_turn to x_val
        from scipy.integrate import quad
        result, _ = quad(
            lambda xp: np.sqrt(max(0, 2*m*(E - 0.5*m*omega**2*xp**2))),
            -x_turn, x_val
        )
        return result / hbar

    # Compute WKB wavefunction
    psi_wkb = []
    for xi in x:
        if abs(xi) < x_turn * 0.99:
            amp = wkb_amplitude(xi)
            phase = phase_integral(xi)
            psi_wkb.append(amp * np.sin(phase + np.pi/4))  # Connection formula
        else:
            psi_wkb.append(0)

    psi_wkb = np.array(psi_wkb)
    # Normalize
    norm = np.sqrt(np.trapz(psi_wkb**2, x))
    if norm > 0:
        psi_wkb = psi_wkb / norm

    # Exact solution (Hermite polynomials)
    from scipy.special import hermite
    from math import factorial

    alpha = m * omega / hbar
    Hn = hermite(n)
    psi_exact = (alpha/np.pi)**0.25 / np.sqrt(2**n * factorial(n)) * \
                Hn(np.sqrt(alpha) * x) * np.exp(-alpha * x**2 / 2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(x, psi_exact**2, 'b-', lw=2, label='Exact |ψ|²')
    ax.plot(x, psi_wkb**2, 'r--', lw=2, label='WKB |ψ|²')
    ax.fill_between(x, 0, V/E_exact[n], alpha=0.2, color='gray', label='V(x)/E')
    ax.axvline(x=x_turn, color='k', ls=':', alpha=0.5)
    ax.axvline(x=-x_turn, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('|ψ|²')
    ax.set_title(f'WKB vs Exact for n={n} Harmonic Oscillator')
    ax.legend()
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.3)

    # Energy level diagram
    ax = axes[1]
    for i, E_i in enumerate(E_exact):
        ax.axhline(y=E_i, color='blue', lw=2)
        ax.text(2.5, E_i, f'n={i}, E={(i+0.5):.1f}ℏω', fontsize=10)

    ax.plot(x, V, 'k-', lw=2, label='V(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Levels\nEBK: E = ℏω(n + 1/2)')
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wkb_approximation.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_wkb_vs_exact()
```

### Lab 4: Hamilton's Optical Analogy

```python
"""
Visualize Hamilton's optical-mechanical analogy.
"""

def optical_mechanical_analogy():
    """Show parallel between light rays and particle trajectories."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mechanics: particle in potential well
    ax = axes[0]

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Potential: V = x² + y²
    V = X**2 + Y**2

    # Effective "refractive index" n² = 2m(E - V)
    E = 5
    n_squared = np.maximum(0, 2*(E - V))
    n = np.sqrt(n_squared)

    ax.contourf(X, Y, V, levels=20, cmap='Blues', alpha=0.5)
    ax.contour(X, Y, V, levels=[E], colors='red', linewidths=2)

    # Plot some trajectories (circles for isotropic oscillator)
    for r0 in [0.5, 1.0, 1.5, 2.0]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r0*np.cos(theta), r0*np.sin(theta), 'b-', lw=1.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mechanics: Particle Trajectories\nin Potential V = x² + y²')
    ax.set_aspect('equal')

    # Optics: light rays in graded-index medium
    ax = axes[1]

    # Refractive index profile (like a GRIN lens)
    n_profile = np.sqrt(np.maximum(0.1, 5 - X**2 - Y**2))

    ax.contourf(X, Y, n_profile, levels=20, cmap='Oranges', alpha=0.7)
    ax.contour(X, Y, n_profile, levels=[1], colors='red', linewidths=2)

    # Light rays curve toward higher n (like trajectories in lower V)
    for y0 in [-1.5, -0.75, 0, 0.75, 1.5]:
        # Simplified ray path
        x_ray = np.linspace(-2, 2, 100)
        # Rays curve toward center (high n)
        y_ray = y0 * np.cos(0.5 * x_ray)
        ax.plot(x_ray, y_ray, 'b-', lw=1.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Optics: Light Rays\nin Graded-Index Medium n(x,y)')
    ax.set_aspect('equal')

    plt.suptitle("Hamilton's Optical-Mechanical Analogy\n" +
                 "Particle trajectories ↔ Light rays | Action S ↔ Optical path",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('optical_mechanical_analogy.png', dpi=150, bbox_inches='tight')
    plt.show()

optical_mechanical_analogy()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Hamilton-Jacobi equation | $\frac{\partial S}{\partial t} + H(q, \frac{\partial S}{\partial q}, t) = 0$ |
| Principal function | $S = \int L \, dt$ |
| Characteristic function | $W = S + Et$ (time-independent H) |
| Reduced HJ equation | $H(q, \frac{\partial W}{\partial q}) = E$ |
| Momentum from S | $p_i = \frac{\partial S}{\partial q_i}$ |
| Trajectory extraction | $\beta_i = \frac{\partial S}{\partial \alpha_i} = \text{const}$ |
| Quantum connection | $\psi = Ae^{iS/\hbar}$ |

### Main Takeaways

1. **The HJ Equation:** A single PDE that contains all of classical mechanics
   - Comes from requiring K = 0 (the simplest possible Hamiltonian)
   - The generating function is Hamilton's principal function S

2. **S = Classical Action:** Hamilton's principal function equals ∫L dt along the true path
   - Momentum: p = ∂S/∂q
   - Energy: E = -∂S/∂t

3. **Solving the HJ Equation:**
   - Separation of variables reduces PDE to ODEs
   - Complete integral has n constants (plus additive)
   - Trajectories from ∂S/∂α = const

4. **The Quantum Connection:**
   - ψ = Ae^{iS/ℏ} links quantum and classical
   - HJ equation is the classical limit of Schrödinger
   - WKB approximation: phase = ∫p dx/ℏ

5. **Hamilton's Analogy:**
   - Mechanics ↔ Optics
   - Action S ↔ Optical path (eikonal)
   - This analogy inspired Schrödinger!

---

## Daily Checklist

### Understanding
- [ ] I can derive the HJ equation from canonical transformation theory
- [ ] I understand why S equals the classical action
- [ ] I can separate variables for standard coordinate systems
- [ ] I see the connection to quantum mechanics

### Computation
- [ ] I can solve the HJ equation for free particles
- [ ] I can solve the HJ equation for the harmonic oscillator
- [ ] I can extract trajectories from a complete integral

### Connections
- [ ] I understand ψ = Ae^{iS/ℏ} and its implications
- [ ] I can explain Hamilton's optical-mechanical analogy
- [ ] I appreciate why Schrödinger called his equation after Hamilton

---

## Preview: Day 166

Tomorrow we study **Introduction to Chaos**, the final major theoretical topic. We'll see what happens when the beautiful structure of integrable systems breaks down:

- **Sensitive dependence on initial conditions**
- **Lyapunov exponents:** λ > 0 means chaos
- **The standard map:** A paradigm for studying chaos
- **KAM theorem:** What survives perturbations
- **Poincaré sections:** Visualizing chaos

This is where classical mechanics confronts its limits—and connects to the modern understanding of complex systems.

---

*"The Hamilton-Jacobi theory is the royal road to quantum mechanics."*
— Erwin Schrödinger

---

**Day 165 Complete. Next: Introduction to Chaos**
