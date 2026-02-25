# Day 143: The Euler-Lagrange Equations

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Euler-Lagrange Derivation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Rigorously derive Euler-Lagrange equations
2. Handle multiple degrees of freedom
3. Understand generalized momenta and forces
4. Work with velocity-dependent potentials
5. Apply to electromagnetic forces
6. Recognize equivalent Lagrangians

---

## 📖 Core Content

### 1. The Euler-Lagrange Equations (General Form)

For a system with n generalized coordinates q = (q₁, ..., qₙ):

$$\boxed{\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0, \quad i = 1, ..., n}$$

These are n second-order ODEs determining qᵢ(t).

---

### 2. Generalized Momentum and Force

**Generalized (canonical) momentum:**
$$\boxed{p_i = \frac{\partial L}{\partial \dot{q}_i}}$$

**Generalized force:**
$$\boxed{Q_i = \frac{\partial L}{\partial q_i}}$$

**Euler-Lagrange in this notation:**
$$\dot{p}_i = Q_i$$

This generalizes Newton's ṗ = F!

---

### 3. Cyclic Coordinates and Conservation

**Definition:** A coordinate qᵢ is **cyclic** (or **ignorable**) if L does not depend on it:
$$\frac{\partial L}{\partial q_i} = 0$$

**Consequence:** The corresponding momentum is conserved!
$$\frac{dp_i}{dt} = \frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} = \frac{\partial L}{\partial q_i} = 0$$

**Example:** Free particle in Cartesian coordinates
- L = ½m(ẋ² + ẏ² + ż²)
- x, y, z are all cyclic
- pₓ = mẋ, pᵧ = mẏ, p_z = mż all conserved

---

### 4. Velocity-Dependent Potentials

For forces not derivable from V(q), we can sometimes use a **generalized potential** U(q, q̇, t):

$$Q_i = -\frac{\partial U}{\partial q_i} + \frac{d}{dt}\frac{\partial U}{\partial \dot{q}_i}$$

The Lagrangian becomes:
$$L = T - U$$

**Key Example:** Electromagnetic force on a charged particle

$$U = q\phi - q\mathbf{A} \cdot \mathbf{v}$$

where φ is scalar potential, **A** is vector potential.

---

### 5. Electromagnetic Lagrangian

For a particle of charge e in electromagnetic field:

$$\boxed{L = \frac{1}{2}m\mathbf{v}^2 - e\phi + e\mathbf{A} \cdot \mathbf{v}}$$

**Canonical momentum:**
$$\mathbf{p} = \frac{\partial L}{\partial \mathbf{v}} = m\mathbf{v} + e\mathbf{A}$$

Note: p ≠ mv! The canonical momentum includes the field contribution.

**Euler-Lagrange equation gives the Lorentz force:**
$$m\ddot{\mathbf{r}} = e(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

where **E** = -∇φ - ∂**A**/∂t and **B** = ∇ × **A**.

---

### 6. Equivalent Lagrangians

**Theorem:** If L' = L + dF/dt for some F(q, t), then L and L' give the same equations of motion.

**Proof:**
$$S' = \int_{t_1}^{t_2} L'\,dt = \int_{t_1}^{t_2} L\,dt + F(q_2, t_2) - F(q_1, t_1)$$

The added term is constant (depends only on fixed endpoints), so δS' = δS.

**Consequence:** The Lagrangian is not unique! We can add any total time derivative.

---

### 7. Examples

**Example 1: Particle in Central Force**

L = ½m(ṙ² + r²θ̇²) - V(r)

Euler-Lagrange for r:
$$m\ddot{r} - mr\dot{\theta}^2 = -\frac{dV}{dr}$$

Euler-Lagrange for θ:
$$\frac{d}{dt}(mr^2\dot{\theta}) = 0 \quad \Rightarrow \quad mr^2\dot{\theta} = \ell = \text{const}$$

Angular momentum is conserved! (θ is cyclic)

**Example 2: Double Pendulum**

L = ½(m₁+m₂)L₁²θ̇₁² + ½m₂L₂²θ̇₂² + m₂L₁L₂cos(θ₁-θ₂)θ̇₁θ̇₂ + (m₁+m₂)gL₁cos θ₁ + m₂gL₂cos θ₂

Two coupled nonlinear equations (leads to chaos!).

---

## 🔧 Practice Problems

### Level 1: Direct Application
1. For L = ½m(ẋ² + ẏ²) - mgy, find the equations of motion.
2. A bead slides on a rotating wire. L = ½m(ṙ² + r²ω²) - V(r). Find the equation of motion.
3. Verify that p_θ = mr²θ̇ for a free particle in polar coordinates.

### Level 2: Electromagnetic
4. Starting from L = ½mv² - eφ + e**A**·**v**, derive the x-component of the Lorentz force.
5. For a uniform magnetic field **B** = Bẑ, choose **A** and write the Lagrangian.

### Level 3: Equivalent Lagrangians
6. Show that L' = ½mẋ² - ½kx² and L'' = ½mẋ² - ½kx² + d(αxt)/dt give the same EOM.
7. Find F(q,t) such that L' = L + dF/dt simplifies a given problem.

### Level 4: Theory
8. Prove that if L is homogeneous of degree 2 in the velocities, then T = L.
9. Show that for time-independent L, the energy E = Σᵢ pᵢq̇ᵢ - L is conserved.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp

def symbolic_euler_lagrange():
    """Use SymPy to derive Euler-Lagrange equations symbolically."""
    
    print("=" * 60)
    print("SYMBOLIC EULER-LAGRANGE DERIVATION")
    print("=" * 60)
    
    # Define symbols
    t = sp.Symbol('t')
    m, g, L_val, k = sp.symbols('m g L k', positive=True)
    
    # Example 1: Simple pendulum
    print("\n1. Simple Pendulum")
    theta = sp.Function('theta')(t)
    theta_dot = sp.diff(theta, t)
    
    T = sp.Rational(1,2) * m * L_val**2 * theta_dot**2
    V = -m * g * L_val * sp.cos(theta)
    Lag = T - V
    
    # Euler-Lagrange: d/dt(∂L/∂θ̇) - ∂L/∂θ = 0
    dL_dtheta_dot = sp.diff(Lag, theta_dot)
    dL_dtheta = sp.diff(Lag, theta)
    EL = sp.diff(dL_dtheta_dot, t) - dL_dtheta
    
    print(f"   L = {Lag}")
    print(f"   ∂L/∂θ̇ = {dL_dtheta_dot}")
    print(f"   EL equation: {sp.simplify(EL)} = 0")
    
    # Example 2: Harmonic oscillator
    print("\n2. Harmonic Oscillator")
    x = sp.Function('x')(t)
    x_dot = sp.diff(x, t)
    
    T = sp.Rational(1,2) * m * x_dot**2
    V = sp.Rational(1,2) * k * x**2
    Lag = T - V
    
    dL_dx_dot = sp.diff(Lag, x_dot)
    dL_dx = sp.diff(Lag, x)
    EL = sp.diff(dL_dx_dot, t) - dL_dx
    
    print(f"   L = {Lag}")
    print(f"   EL equation: {sp.simplify(EL)} = 0")
    
    # Example 3: Central force
    print("\n3. Central Force Problem")
    r = sp.Function('r')(t)
    phi = sp.Function('phi')(t)
    r_dot = sp.diff(r, t)
    phi_dot = sp.diff(phi, t)
    
    T = sp.Rational(1,2) * m * (r_dot**2 + r**2 * phi_dot**2)
    V_r = sp.Function('V')(r)
    Lag = T - V_r
    
    # r equation
    dL_dr_dot = sp.diff(Lag, r_dot)
    dL_dr = sp.diff(Lag, r)
    EL_r = sp.diff(dL_dr_dot, t) - dL_dr
    
    # φ equation  
    dL_dphi_dot = sp.diff(Lag, phi_dot)
    dL_dphi = sp.diff(Lag, phi)
    EL_phi = sp.diff(dL_dphi_dot, t) - dL_dphi
    
    print(f"   L = T - V(r) where T = {T}")
    print(f"   EL(r): {sp.simplify(EL_r)} = 0")
    print(f"   EL(φ): {sp.simplify(EL_phi)} = 0")
    print(f"   Note: φ is cyclic → p_φ = mr²φ̇ = const (angular momentum)")

symbolic_euler_lagrange()


def electromagnetic_lagrangian():
    """Verify the Lorentz force from the Lagrangian."""
    
    print("\n" + "=" * 60)
    print("ELECTROMAGNETIC LAGRANGIAN")
    print("=" * 60)
    
    # Charged particle in uniform B field
    # B = B ẑ, so A = (1/2) B × r = (-By/2, Bx/2, 0) or A = (0, Bx, 0)
    
    # Using A = (0, Bx, 0):
    # L = (1/2)m(ẋ² + ẏ² + ż²) + eA·v = (1/2)m(ẋ² + ẏ² + ż²) + eBxẏ
    
    print("\nFor B = Bẑ, choose A = (0, Bx, 0)")
    print("L = (1/2)m(ẋ² + ẏ² + ż²) + eBxẏ")
    print("\nEuler-Lagrange equations:")
    print("  x: mẍ = eB·ẏ  (force in x from B)")
    print("  y: mÿ = -eB·ẋ (force in y from B)")
    print("  z: mz̈ = 0")
    print("\nThis gives circular motion in the xy-plane!")
    
    # Numerical simulation
    def lorentz_eom(state, t, e, m, B):
        x, y, z, vx, vy, vz = state
        ax = e * B * vy / m
        ay = -e * B * vx / m
        az = 0
        return [vx, vy, vz, ax, ay, az]
    
    # Parameters
    e, m, B = 1.0, 1.0, 1.0
    omega_c = e * B / m  # Cyclotron frequency
    
    # Initial conditions
    state0 = [0, 0, 0, 1, 0, 0]  # Start at origin, moving in x
    
    t = np.linspace(0, 4*np.pi/omega_c, 500)
    solution = odeint(lorentz_eom, state0, t, args=(e, m, B))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # xy trajectory
    axes[0].plot(solution[:, 0], solution[:, 1], 'b-', lw=2)
    axes[0].scatter([0], [0], c='red', s=100, zorder=5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Charged Particle in Magnetic Field\n(Cyclotron motion)')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Velocity components
    axes[1].plot(t, solution[:, 3], 'b-', lw=2, label='vₓ')
    axes[1].plot(t, solution[:, 4], 'r-', lw=2, label='vᵧ')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('velocity')
    axes[1].set_title('Velocity Components\n(Circular motion in velocity space)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('electromagnetic_motion.png', dpi=150)
    plt.show()

electromagnetic_lagrangian()
```

---

## 📝 Summary

### Euler-Lagrange Equations

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0$$

### Key Definitions

| Quantity | Formula |
|----------|---------|
| Generalized momentum | pᵢ = ∂L/∂q̇ᵢ |
| Generalized force | Qᵢ = ∂L/∂qᵢ |
| Cyclic coordinate | ∂L/∂qᵢ = 0 |

### Conservation Laws
- Cyclic coordinate → conserved momentum
- Time-independent L → conserved energy

---

## ✅ Daily Checklist

- [ ] Derive Euler-Lagrange rigorously
- [ ] Understand generalized momenta
- [ ] Identify cyclic coordinates
- [ ] Handle electromagnetic Lagrangian
- [ ] Recognize equivalent Lagrangians
- [ ] Complete computational exercises

---

## 🔮 Preview: Day 144

Tomorrow we study **Constraints and Lagrange Multipliers** — how to systematically handle constraints using the variational formulation!
