# Day 155: The Legendre Transformation — From L to H

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Legendre Transformation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Understand the geometric meaning of Legendre transformation
2. Transform between Lagrangian and Hamiltonian formulations
3. Define conjugate/canonical momentum
4. Construct Hamiltonians from Lagrangians
5. Connect to thermodynamic potentials
6. Recognize when the Legendre transform is well-defined

---

## 📖 Core Content

### 1. Motivation: Why a New Formulation?

**Lagrangian mechanics:** L(q, q̇, t) — uses (q, q̇) as independent variables

**Hamiltonian mechanics:** H(q, p, t) — uses (q, p) as independent variables

**Why switch?**
- First-order differential equations (2n equations vs n second-order)
- Phase space geometry (symplectic structure)
- Direct path to quantum mechanics
- Conservation laws more transparent
- Statistical mechanics foundations

---

### 2. The Legendre Transformation: Geometric Intuition

**Key Insight:** A convex function can be specified either by:
1. Its value f(x) at each point x
2. The y-intercept g(p) of the tangent line with slope p

**Definition:** For a convex function f(x), define:
$$p = \frac{df}{dx} \quad \text{(slope of tangent)}$$

The **Legendre transform** is:
$$\boxed{g(p) = px - f(x)}$$

where x = x(p) is found by inverting p = df/dx.

**Geometric meaning:** g(p) is the negative y-intercept of the tangent line with slope p.

---

### 3. Properties of Legendre Transformation

**Involution:** The Legendre transform of g returns f:
$$(f^*)^* = f$$

**Derivative exchange:**
$$\frac{dg}{dp} = x, \quad \frac{df}{dx} = p$$

**Symmetry:**
$$f(x) + g(p) = px$$

**Convexity requirement:** f must be convex (f'' > 0) for the transform to be well-defined.

---

### 4. From Lagrangian to Hamiltonian

**Canonical momentum:** Define the conjugate momentum:
$$\boxed{p_i = \frac{\partial L}{\partial \dot{q}_i}}$$

This is the "slope" with respect to q̇ᵢ.

**Hamiltonian:** The Legendre transform of L with respect to all q̇ᵢ:
$$\boxed{H(q, p, t) = \sum_i p_i \dot{q}_i - L(q, \dot{q}, t)}$$

where q̇ᵢ = q̇ᵢ(q, p, t) from inverting pᵢ = ∂L/∂q̇ᵢ.

---

### 5. Differential Relations

Taking the differential of H:
$$dH = \sum_i \dot{q}_i dp_i + \sum_i p_i d\dot{q}_i - \sum_i \frac{\partial L}{\partial q_i}dq_i - \sum_i \frac{\partial L}{\partial \dot{q}_i}d\dot{q}_i - \frac{\partial L}{\partial t}dt$$

Using p = ∂L/∂q̇, terms cancel:
$$dH = \sum_i \dot{q}_i dp_i - \sum_i \frac{\partial L}{\partial q_i}dq_i - \frac{\partial L}{\partial t}dt$$

**Therefore:**
$$\frac{\partial H}{\partial p_i} = \dot{q}_i, \quad \frac{\partial H}{\partial q_i} = -\frac{\partial L}{\partial q_i}, \quad \frac{\partial H}{\partial t} = -\frac{\partial L}{\partial t}$$

---

### 6. When H = T + V (Total Energy)

For **natural systems** where:
- L = T - V with T quadratic in velocities
- T = ½ Σᵢⱼ Mᵢⱼ(q) q̇ᵢ q̇ⱼ (kinetic energy homogeneous degree 2 in q̇)
- V = V(q) (velocity-independent potential)

**Euler's theorem:** For homogeneous degree n: Σᵢ xᵢ ∂f/∂xᵢ = nf

**Apply to T (degree 2):**
$$\sum_i \dot{q}_i \frac{\partial T}{\partial \dot{q}_i} = 2T$$

**Therefore:**
$$H = \sum_i p_i\dot{q}_i - L = \sum_i \dot{q}_i\frac{\partial L}{\partial \dot{q}_i} - (T-V) = 2T - T + V = T + V$$

**H equals total energy for natural, scleronomic systems!**

---

### 7. When H ≠ Total Energy

**Velocity-dependent potentials:** e.g., electromagnetic
$$L = \frac{1}{2}m\mathbf{v}^2 - e\phi + e\mathbf{A}\cdot\mathbf{v}$$

Here p = mv + eA ≠ mv, so H = T + eφ (not T + V in usual sense)

**Time-dependent constraints (rheonomic):** 
When L depends explicitly on t through constraints, H ≠ E

**Moving coordinate systems:**
In rotating frames, H includes fictitious potential terms

---

### 8. Thermodynamic Analogy

The Legendre transform appears throughout thermodynamics:

| Thermodynamic | Mechanics |
|---------------|-----------|
| Internal energy U(S, V) | Lagrangian L(q, q̇) |
| Enthalpy H(S, P) | — |
| Helmholtz F(T, V) | Hamiltonian H(q, p) |
| Gibbs G(T, P) | — |

**Pattern:** Replace a variable with its conjugate
- (S, V) → (T, V): U → F = U - TS
- (q̇) → (p): L → H = pq̇ - L

---

### 9. 🔬 Quantum Mechanics Connection

**Classical:** H(q, p) is a function on phase space

**Quantum:** Ĥ(q̂, p̂) is an operator on Hilbert space

The Legendre transform structure survives quantization:
- Canonical momentum p → p̂ = -iℏ∂/∂q
- Hamiltonian generates time evolution: iℏ∂ψ/∂t = Ĥψ

---

## ✏️ Worked Examples

### Example 1: Simple Harmonic Oscillator

**Lagrangian:** L = ½mẋ² - ½kx²

**Canonical momentum:**
$$p = \frac{\partial L}{\partial \dot{x}} = m\dot{x}$$

**Invert:** ẋ = p/m

**Hamiltonian:**
$$H = p\dot{x} - L = p \cdot \frac{p}{m} - \frac{1}{2}m\left(\frac{p}{m}\right)^2 + \frac{1}{2}kx^2$$

$$\boxed{H = \frac{p^2}{2m} + \frac{1}{2}kx^2}$$

**Check:** H = T + V ✓

---

### Example 2: Charged Particle in EM Field

**Lagrangian:** L = ½m|**v**|² - eφ + e**A**·**v**

**Canonical momentum:**
$$\mathbf{p} = \frac{\partial L}{\partial \mathbf{v}} = m\mathbf{v} + e\mathbf{A}$$

**Note:** p ≠ mv (kinetic momentum)! The canonical momentum includes the vector potential.

**Invert:** **v** = (**p** - e**A**)/m

**Hamiltonian:**
$$H = \mathbf{p} \cdot \mathbf{v} - L = \mathbf{p} \cdot \frac{\mathbf{p} - e\mathbf{A}}{m} - \frac{1}{2m}|\mathbf{p} - e\mathbf{A}|^2 + e\phi$$

$$\boxed{H = \frac{|\mathbf{p} - e\mathbf{A}|^2}{2m} + e\phi}$$

This is the form used in quantum mechanics for minimal coupling!

---

### Example 3: Relativistic Free Particle

**Lagrangian:** L = -mc²√(1 - v²/c²)

**Canonical momentum:**
$$p = \frac{\partial L}{\partial v} = \frac{mv}{\sqrt{1 - v^2/c^2}} = \gamma mv$$

**Invert:** v = pc/√(p² + m²c²)

**Hamiltonian:**
$$H = pv - L = \sqrt{p^2c^2 + m^2c^4}$$

This is the relativistic energy-momentum relation E² = p²c² + m²c⁴!

---

## 🔧 Practice Problems

### Level 1: Direct Computation
1. Find H for a free particle: L = ½mẋ²
2. Find H for a particle in gravity: L = ½m(ẋ² + ẏ²) - mgy
3. Find H for a pendulum: L = ½mL²θ̇² + mgL cos θ

### Level 2: Canonical Momentum
4. For L = ½m(ṙ² + r²θ̇²) - V(r), find p_r and p_θ, then H.
5. Show that for a bead on a rotating hoop, H ≠ T + V.

### Level 3: Challenging
6. For a charged particle in a magnetic field B = B₀ẑ with A = ½B₀(-y, x, 0), find H.
7. Prove that if L doesn't depend on q̇ᵢ explicitly, the Legendre transform fails. What does this mean physically?

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

def legendre_transform_demo():
    """Demonstrate Legendre transformation visually and algebraically."""
    
    print("=" * 70)
    print("LEGENDRE TRANSFORMATION DEMONSTRATION")
    print("=" * 70)
    
    # Part 1: Geometric visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original function: f(x) = x²/2 (like kinetic energy T = mv²/2)
    x = np.linspace(-3, 3, 100)
    f = x**2 / 2
    
    ax = axes[0]
    ax.plot(x, f, 'b-', lw=2, label='f(x) = x²/2')
    
    # Draw tangent lines at different points
    for x0, color in [(-2, 'red'), (0, 'green'), (1.5, 'orange')]:
        # Slope p = df/dx = x
        p = x0
        # y-intercept b = f(x0) - p*x0 = x0²/2 - x0² = -x0²/2
        b = f[np.argmin(np.abs(x - x0))] - p * x0
        tangent = p * x + b
        ax.plot(x, tangent, '--', color=color, lw=1.5, 
                label=f'Tangent at x={x0}, slope p={p}')
        ax.scatter([x0], [x0**2/2], c=color, s=100, zorder=5)
        ax.scatter([0], [b], c=color, s=100, marker='x', zorder=5)
    
    ax.set_xlabel('x (velocity)')
    ax.set_ylabel('f(x) (Lagrangian ≈ T)')
    ax.set_title('Original Function f(x) = x²/2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 5)
    
    # Legendre transform: g(p) = px - f(x) where x = x(p)
    # For f = x²/2: p = df/dx = x, so x = p
    # g(p) = p*p - p²/2 = p²/2
    ax = axes[1]
    p = np.linspace(-3, 3, 100)
    g = p**2 / 2
    
    ax.plot(p, g, 'r-', lw=2, label='g(p) = p²/2')
    ax.set_xlabel('p (momentum)')
    ax.set_ylabel('g(p) (Hamiltonian ≈ T)')
    ax.set_title('Legendre Transform g(p) = p²/2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Verify: f + g = px
    ax = axes[2]
    x_vals = np.linspace(-2, 2, 50)
    for x0 in x_vals:
        p0 = x0  # p = df/dx
        f0 = x0**2 / 2
        g0 = p0**2 / 2
        ax.scatter([x0], [f0 + g0], c='blue', s=20, alpha=0.5)
        ax.scatter([x0], [p0 * x0], c='red', s=20, alpha=0.5, marker='x')
    
    ax.plot(x_vals, x_vals**2, 'g-', lw=2, label='px = x² (since p=x)')
    ax.set_xlabel('x')
    ax.set_ylabel('Value')
    ax.set_title('Verification: f(x) + g(p) = px')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('legendre_transform.png', dpi=150)
    plt.show()
    
    # Part 2: Symbolic computation
    print("\n" + "=" * 50)
    print("SYMBOLIC HAMILTONIAN CONSTRUCTION")
    print("=" * 50)
    
    # Example: Simple Harmonic Oscillator
    m, k = symbols('m k', positive=True, real=True)
    x, x_dot, p = symbols('x x_dot p', real=True)
    
    L = Rational(1,2)*m*x_dot**2 - Rational(1,2)*k*x**2
    print(f"\nLagrangian: L = {L}")
    
    # Canonical momentum
    p_expr = diff(L, x_dot)
    print(f"Canonical momentum: p = ∂L/∂ẋ = {p_expr}")
    
    # Solve for x_dot
    x_dot_of_p = solve(p_expr - p, x_dot)[0]
    print(f"Velocity: ẋ = {x_dot_of_p}")
    
    # Hamiltonian
    H = p * x_dot_of_p - L.subs(x_dot, x_dot_of_p)
    H = simplify(H)
    print(f"Hamiltonian: H = pẋ - L = {H}")
    
    # Verify it equals T + V
    T = Rational(1,2)*m*x_dot**2
    V = Rational(1,2)*k*x**2
    T_in_p = T.subs(x_dot, x_dot_of_p)
    print(f"\nVerification: T + V = {simplify(T_in_p + V)}")
    print(f"H = T + V? {simplify(H - T_in_p - V) == 0}")

legendre_transform_demo()


def hamiltonian_examples():
    """Compute Hamiltonians for various systems."""
    
    print("\n" + "=" * 70)
    print("HAMILTONIAN EXAMPLES")
    print("=" * 70)
    
    # Pendulum
    print("\n1. Simple Pendulum")
    print("-" * 40)
    
    m, L_pend, g = symbols('m L g', positive=True)
    theta, theta_dot, p_theta = symbols('theta theta_dot p_theta', real=True)
    
    T = Rational(1,2) * m * L_pend**2 * theta_dot**2
    V = -m * g * L_pend * cos(theta)
    L = T - V
    
    print(f"T = {T}")
    print(f"V = {V}")
    print(f"L = {L}")
    
    p_theta_expr = diff(L, theta_dot)
    print(f"p_θ = {p_theta_expr}")
    
    theta_dot_of_p = solve(p_theta_expr - p_theta, theta_dot)[0]
    
    H = p_theta * theta_dot_of_p - L.subs(theta_dot, theta_dot_of_p)
    H = simplify(H)
    print(f"H = {H}")
    
    # Central force
    print("\n2. Central Force (2D)")
    print("-" * 40)
    
    r, phi = symbols('r phi', real=True, positive=True)
    r_dot, phi_dot = symbols('r_dot phi_dot', real=True)
    p_r, p_phi = symbols('p_r p_phi', real=True)
    V_r = Function('V')(r)
    
    T = Rational(1,2) * m * (r_dot**2 + r**2 * phi_dot**2)
    L = T - V_r
    
    print(f"T = {T}")
    print(f"L = T - V(r)")
    
    p_r_expr = diff(L, r_dot)
    p_phi_expr = diff(L, phi_dot)
    print(f"p_r = {p_r_expr}")
    print(f"p_φ = {p_phi_expr}")
    
    # Note: p_phi = m*r²*phi_dot is angular momentum!
    print("\nNote: p_φ = mr²φ̇ is the angular momentum L!")

hamiltonian_examples()
```

---

## 📝 Summary

### Legendre Transformation

$$g(p) = px - f(x), \quad \text{where } p = \frac{df}{dx}$$

### Lagrangian → Hamiltonian

| Step | Formula |
|------|---------|
| Define momentum | pᵢ = ∂L/∂q̇ᵢ |
| Invert for velocities | q̇ᵢ = q̇ᵢ(q, p) |
| Compute Hamiltonian | H = Σpᵢq̇ᵢ - L |

### Key Results

| Condition | Hamiltonian |
|-----------|-------------|
| Natural system (T quadratic, V(q) only) | H = T + V |
| Velocity-dependent potential | H ≠ simple T + V |
| Time-dependent constraints | H ≠ E (total energy) |

---

## ✅ Daily Checklist

- [ ] Understand Legendre transform geometrically
- [ ] Compute canonical momenta from L
- [ ] Construct H from L for standard systems
- [ ] Recognize when H = T + V
- [ ] Handle velocity-dependent potentials
- [ ] Connect to thermodynamics

---

## 🔮 Preview: Day 156

Tomorrow we derive **Hamilton's Equations of Motion** — the fundamental equations that replace Euler-Lagrange!
