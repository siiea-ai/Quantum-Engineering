# Day 156: Hamilton's Equations of Motion

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Hamilton's Equations |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Derive Hamilton's equations from the Legendre transform
2. Derive Hamilton's equations from Hamilton's principle
3. Solve Hamilton's equations for standard systems
4. Understand the first-order structure of Hamiltonian dynamics
5. Recognize cyclic coordinates and conservation laws
6. Compare Lagrangian and Hamiltonian formulations

---

## 📖 Core Content

### 1. Derivation from the Legendre Transform

From the differential of H(q, p, t):
$$dH = \sum_i \frac{\partial H}{\partial q_i}dq_i + \sum_i \frac{\partial H}{\partial p_i}dp_i + \frac{\partial H}{\partial t}dt$$

From H = Σpᵢq̇ᵢ - L and the Euler-Lagrange equations:
$$dH = \sum_i \dot{q}_i dp_i - \sum_i \dot{p}_i dq_i - \frac{\partial L}{\partial t}dt$$

**Comparing coefficients:**

$$\boxed{\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}}$$

These are **Hamilton's canonical equations of motion**.

---

### 2. Derivation from Hamilton's Principle

**Modified Hamilton's Principle:** The action in phase space is
$$S = \int_{t_1}^{t_2} \left(\sum_i p_i\dot{q}_i - H(q, p, t)\right) dt$$

**Variation:** Treating q and p as independent:
$$\delta S = \int_{t_1}^{t_2} \left[\sum_i \left(p_i\delta\dot{q}_i + \dot{q}_i\delta p_i - \frac{\partial H}{\partial q_i}\delta q_i - \frac{\partial H}{\partial p_i}\delta p_i\right)\right] dt = 0$$

**Integration by parts on the first term:**
$$\int p_i\delta\dot{q}_i\,dt = [p_i\delta q_i]_{t_1}^{t_2} - \int \dot{p}_i\delta q_i\,dt$$

**With boundary conditions** δq(t₁) = δq(t₂) = 0:
$$\delta S = \int_{t_1}^{t_2} \sum_i \left[\left(\dot{q}_i - \frac{\partial H}{\partial p_i}\right)\delta p_i - \left(\dot{p}_i + \frac{\partial H}{\partial q_i}\right)\delta q_i\right] dt = 0$$

Since δqᵢ and δpᵢ are arbitrary, Hamilton's equations follow.

---

### 3. Structure of Hamilton's Equations

**2n first-order ODEs** (vs n second-order Euler-Lagrange):

For n degrees of freedom:
$$\dot{q}_1 = \frac{\partial H}{\partial p_1}, \quad \dot{p}_1 = -\frac{\partial H}{\partial q_1}$$
$$\vdots$$
$$\dot{q}_n = \frac{\partial H}{\partial p_n}, \quad \dot{p}_n = -\frac{\partial H}{\partial q_n}$$

**Matrix form:** Define z = (q₁, ..., qₙ, p₁, ..., pₙ)ᵀ and the symplectic matrix:
$$\mathbf{J} = \begin{pmatrix} \mathbf{0} & \mathbf{I}_n \\ -\mathbf{I}_n & \mathbf{0} \end{pmatrix}$$

Then:
$$\dot{\mathbf{z}} = \mathbf{J}\nabla H$$

This is the **symplectic form** of Hamilton's equations.

---

### 4. Conservation of the Hamiltonian

**Time derivative of H along a trajectory:**
$$\frac{dH}{dt} = \sum_i \left(\frac{\partial H}{\partial q_i}\dot{q}_i + \frac{\partial H}{\partial p_i}\dot{p}_i\right) + \frac{\partial H}{\partial t}$$

Using Hamilton's equations:
$$\frac{dH}{dt} = \sum_i \left(\frac{\partial H}{\partial q_i}\frac{\partial H}{\partial p_i} - \frac{\partial H}{\partial p_i}\frac{\partial H}{\partial q_i}\right) + \frac{\partial H}{\partial t} = \frac{\partial H}{\partial t}$$

**Result:** If H has no explicit time dependence (∂H/∂t = 0), then:
$$\boxed{\frac{dH}{dt} = 0 \quad \text{(H is conserved)}}$$

---

### 5. Cyclic Coordinates and Conservation

**Definition:** A coordinate qⱼ is **cyclic** (or ignorable) if it doesn't appear in H:
$$\frac{\partial H}{\partial q_j} = 0$$

**Consequence:** From Hamilton's equations:
$$\dot{p}_j = -\frac{\partial H}{\partial q_j} = 0$$

**Therefore: pⱼ = constant (conserved momentum)**

| Cyclic Coordinate | Conserved Momentum |
|-------------------|-------------------|
| Linear position x | Linear momentum pₓ |
| Angle θ | Angular momentum L |
| Phase (gauge) | Charge |

---

### 6. Comparison: Lagrangian vs Hamiltonian

| Aspect | Lagrangian | Hamiltonian |
|--------|------------|-------------|
| Variables | (q, q̇) | (q, p) |
| Function | L = T - V | H (often T + V) |
| Equations | n 2nd-order | 2n 1st-order |
| Symmetry | Less symmetric | q and p nearly symmetric |
| Conservation | From ∂L/∂qᵢ = 0 | From ∂H/∂qᵢ = 0 |
| Path to QM | Less direct | Direct (commutators) |

---

### 7. 🔬 Quantum Mechanics Connection

**Classical Hamilton's equations:**
$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$

**Quantum Heisenberg equations:**
$$\frac{d\hat{q}}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{q}], \quad \frac{d\hat{p}}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{p}]$$

**The correspondence:**
$$\{A, B\}_{\text{Poisson}} \to \frac{1}{i\hbar}[\hat{A}, \hat{B}]$$

Hamilton's equations become the Heisenberg equations of motion!

---

## ✏️ Worked Examples

### Example 1: Simple Harmonic Oscillator

**Hamiltonian:** H = p²/(2m) + ½kx²

**Hamilton's equations:**
$$\dot{x} = \frac{\partial H}{\partial p} = \frac{p}{m}$$
$$\dot{p} = -\frac{\partial H}{\partial x} = -kx$$

**Combine:** ẍ = ṗ/m = -kx/m = -ω²x (where ω² = k/m)

**Solution:** x(t) = A cos(ωt + φ), p(t) = -mωA sin(ωt + φ)

**Phase space:** Trajectories are ellipses: p²/(2mE) + kx²/(2E) = 1

---

### Example 2: Central Force in 2D

**Hamiltonian:** H = p_r²/(2m) + p_θ²/(2mr²) + V(r)

**Hamilton's equations:**
$$\dot{r} = \frac{p_r}{m}, \quad \dot{p}_r = \frac{p_\theta^2}{mr^3} - \frac{dV}{dr}$$
$$\dot{\theta} = \frac{p_\theta}{mr^2}, \quad \dot{p}_\theta = 0$$

**Conservation:** θ is cyclic → p_θ = L (angular momentum) is conserved!

**Radial equation:** mṙ̈ = L²/(mr³) - dV/dr (effective 1D problem)

---

### Example 3: Charged Particle in B-field

**Hamiltonian:** H = (p - eA)²/(2m) with A = ½B(-y, x, 0)

**Canonical momenta:**
$$p_x = m\dot{x} - \frac{eB}{2}y, \quad p_y = m\dot{y} + \frac{eB}{2}x$$

**Hamilton's equations yield:** The cyclotron motion with ω_c = eB/m

---

## 🔧 Practice Problems

### Level 1: Direct Application
1. Write Hamilton's equations for H = p²/(2m) + mgy.
2. For H = p²/(2m) + ½kx², solve the equations and verify energy conservation.

### Level 2: Conservation Laws
3. For H = p_r²/(2m) + p_θ²/(2mr²) - k/r, identify cyclic coordinates and conserved quantities.
4. Show that for any H(p², q), the quantity p is conserved.

### Level 3: Advanced
5. For a particle on a sphere, use spherical coordinates to write H and Hamilton's equations.
6. Verify that Hamilton's equations are equivalent to Euler-Lagrange for H = p²/(2m) + V(x).

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def hamiltons_equations_demo():
    """Solve and visualize Hamilton's equations for various systems."""
    
    print("=" * 70)
    print("HAMILTON'S EQUATIONS: NUMERICAL SOLUTIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # System 1: Simple Harmonic Oscillator
    print("\n1. Simple Harmonic Oscillator")
    print("-" * 40)
    
    def sho_hamilton(state, t, m=1, k=1):
        x, p = state
        # Hamilton's equations: dx/dt = dH/dp, dp/dt = -dH/dx
        x_dot = p / m
        p_dot = -k * x
        return [x_dot, p_dot]
    
    m, k = 1.0, 1.0
    omega = np.sqrt(k/m)
    
    t = np.linspace(0, 4*np.pi, 500)
    
    # Multiple initial conditions
    ax = axes[0, 0]
    for E in [0.5, 1.0, 2.0]:
        x0 = np.sqrt(2*E/k)
        state0 = [x0, 0]
        sol = odeint(sho_hamilton, state0, t, args=(m, k))
        ax.plot(sol[:, 0], sol[:, 1], lw=1.5, label=f'E = {E}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title('SHO Phase Space\n(Ellipses of constant energy)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Time evolution
    ax = axes[0, 1]
    state0 = [1, 0]
    sol = odeint(sho_hamilton, state0, t, args=(m, k))
    
    ax.plot(t, sol[:, 0], 'b-', lw=2, label='x(t)')
    ax.plot(t, sol[:, 1], 'r-', lw=2, label='p(t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x, p')
    ax.set_title('SHO Time Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy conservation
    ax = axes[0, 2]
    E = sol[:, 1]**2/(2*m) + k*sol[:, 0]**2/2
    ax.plot(t, E, 'g-', lw=2)
    ax.set_xlabel('t')
    ax.set_ylabel('H = E')
    ax.set_title(f'Energy Conservation\nstd = {np.std(E):.2e}')
    ax.grid(True, alpha=0.3)
    
    # System 2: Pendulum (nonlinear)
    print("\n2. Nonlinear Pendulum")
    print("-" * 40)
    
    def pendulum_hamilton(state, t, m=1, L=1, g=10):
        theta, p_theta = state
        # H = p_theta²/(2mL²) - mgL*cos(theta)
        theta_dot = p_theta / (m * L**2)
        p_dot = -m * g * L * np.sin(theta)
        return [theta_dot, p_dot]
    
    m, L, g = 1.0, 1.0, 10.0
    
    ax = axes[1, 0]
    t = np.linspace(0, 10, 1000)
    
    # Different energies (libration vs rotation)
    for theta0 in [0.5, 1.5, 2.5, 3.14]:
        state0 = [theta0, 0]
        sol = odeint(pendulum_hamilton, state0, t, args=(m, L, g))
        ax.plot(sol[:, 0], sol[:, 1], lw=1, label=f'θ₀ = {theta0:.1f}')
    
    # Add separatrix
    theta_sep = np.linspace(-np.pi, np.pi, 200)
    # At separatrix: E = mgL, so p² = 2mL²mgL(1 + cos(theta))
    p_sep_pos = np.sqrt(2*m*L**2*m*g*L*(1 + np.cos(theta_sep)))
    p_sep_neg = -p_sep_pos
    ax.plot(theta_sep, p_sep_pos, 'k--', lw=2, label='Separatrix')
    ax.plot(theta_sep, p_sep_neg, 'k--', lw=2)
    
    ax.set_xlabel('θ')
    ax.set_ylabel('p_θ')
    ax.set_title('Pendulum Phase Space\n(Libration, Rotation, Separatrix)')
    ax.set_xlim(-4, 4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # System 3: Kepler Problem
    print("\n3. Kepler Problem (Central Force)")
    print("-" * 40)
    
    def kepler_hamilton(state, t, m=1, k=1):
        r, theta, p_r, p_theta = state
        # H = p_r²/(2m) + p_theta²/(2mr²) - k/r
        r_dot = p_r / m
        theta_dot = p_theta / (m * r**2)
        p_r_dot = p_theta**2 / (m * r**3) - k / r**2
        p_theta_dot = 0  # θ is cyclic!
        return [r_dot, theta_dot, p_r_dot, p_theta_dot]
    
    ax = axes[1, 1]
    
    # Elliptical orbit
    r0 = 1.0
    L = 0.8  # angular momentum
    E = -0.3  # negative for bound orbit
    p_r0 = 0
    p_theta0 = L
    
    state0 = [r0, 0, p_r0, p_theta0]
    t = np.linspace(0, 30, 2000)
    sol = odeint(kepler_hamilton, state0, t)
    
    r, theta = sol[:, 0], sol[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax.plot(x, y, 'b-', lw=1)
    ax.scatter([0], [0], c='orange', s=200, marker='*', zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Kepler Orbit\n(θ cyclic → L conserved)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Angular momentum conservation
    ax = axes[1, 2]
    L_values = sol[:, 3]  # p_theta is conserved
    
    ax.plot(t, L_values, 'b-', lw=2, label='p_θ (angular momentum)')
    ax.axhline(y=L, color='r', linestyle='--', label=f'L = {L}')
    ax.set_xlabel('t')
    ax.set_ylabel('p_θ')
    ax.set_title(f'Angular Momentum Conservation\nstd = {np.std(L_values):.2e}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hamiltons_equations.png', dpi=150)
    plt.show()

hamiltons_equations_demo()


def phase_flow_visualization():
    """Visualize the phase flow for different Hamiltonians."""
    
    print("\n" + "=" * 70)
    print("PHASE FLOW VISUALIZATION")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Grid for vector field
    x = np.linspace(-2, 2, 20)
    p = np.linspace(-2, 2, 20)
    X, P = np.meshgrid(x, p)
    
    # System 1: SHO - H = p²/2 + x²/2
    ax = axes[0]
    dX = P  # dx/dt = dH/dp = p
    dP = -X  # dp/dt = -dH/dx = -x
    
    ax.streamplot(X, P, dX, dP, density=1.5, color='blue', linewidth=1)
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title('SHO: H = p²/2 + x²/2\nCircular flow')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # System 2: Free particle - H = p²/2
    ax = axes[1]
    dX = P  # dx/dt = p
    dP = np.zeros_like(X)  # dp/dt = 0
    
    ax.streamplot(X, P, dX, dP, density=1.5, color='green', linewidth=1)
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title('Free Particle: H = p²/2\nHorizontal flow (p conserved)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # System 3: Unstable equilibrium - H = p²/2 - x²/2
    ax = axes[2]
    dX = P  # dx/dt = p
    dP = X  # dp/dt = x (repulsive!)
    
    ax.streamplot(X, P, dX, dP, density=1.5, color='red', linewidth=1)
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title('Unstable: H = p²/2 - x²/2\nHyperbolic flow')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_flow.png', dpi=150)
    plt.show()

phase_flow_visualization()
```

---

## 📝 Summary

### Hamilton's Equations

$$\boxed{\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}}$$

### Key Properties

| Property | Statement |
|----------|-----------|
| First-order | 2n equations instead of n second-order |
| Conservation | If ∂H/∂t = 0, then H = const |
| Cyclic coordinates | If ∂H/∂qⱼ = 0, then pⱼ = const |
| Symplectic form | ż = J∇H |

### Quantum Connection
$$\{A, B\} \to \frac{1}{i\hbar}[\hat{A}, \hat{B}]$$

---

## ✅ Daily Checklist

- [ ] Derive Hamilton's equations both ways
- [ ] Solve equations for SHO
- [ ] Identify cyclic coordinates
- [ ] Understand phase space flow
- [ ] Connect to quantum mechanics
- [ ] Complete computational exercises

---

## 🔮 Preview: Day 157

Tomorrow we explore **Phase Space** — the geometric arena where Hamiltonian dynamics unfolds!
