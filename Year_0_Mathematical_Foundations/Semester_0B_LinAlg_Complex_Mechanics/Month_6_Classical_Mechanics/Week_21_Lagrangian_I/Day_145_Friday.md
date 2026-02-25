# Day 145: Symmetries & Conservation Laws (Noether Preview)

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Symmetries & Conservation |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Connect cyclic coordinates to symmetries
2. Understand energy conservation from time translation symmetry
3. Derive momentum conservation from space translation symmetry
4. Derive angular momentum conservation from rotational symmetry
5. Preview Noether's theorem
6. Apply to physical systems

---

## 📖 Core Content

### 1. The Symmetry-Conservation Connection

**Profound insight:** Every continuous symmetry corresponds to a conserved quantity!

| Symmetry | Conserved Quantity |
|----------|-------------------|
| Time translation | Energy |
| Space translation | Linear momentum |
| Rotation | Angular momentum |
| Gauge transformation | Charge |

This is **Noether's theorem** (1918) — one of the deepest results in physics!

---

### 2. Cyclic Coordinates and Symmetries

If L doesn't depend on qᵢ (cyclic coordinate):
$$\frac{\partial L}{\partial q_i} = 0 \quad \Rightarrow \quad \frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} = 0$$

The corresponding momentum pᵢ = ∂L/∂q̇ᵢ is conserved!

**Physical interpretation:** 
- Cyclic coordinate = symmetry direction
- Conserved momentum = generator of that symmetry

---

### 3. Energy Conservation

**Condition:** L does not depend explicitly on time (∂L/∂t = 0)

**Define the energy function:**
$$h = \sum_i \dot{q}_i \frac{\partial L}{\partial \dot{q}_i} - L$$

**Theorem:** If ∂L/∂t = 0, then dh/dt = 0.

**Proof:**
$$\frac{dh}{dt} = \sum_i \left[\ddot{q}_i \frac{\partial L}{\partial \dot{q}_i} + \dot{q}_i \frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i}\right] - \frac{dL}{dt}$$

Using E-L equations and chain rule:
$$\frac{dL}{dt} = \sum_i \left[\frac{\partial L}{\partial q_i}\dot{q}_i + \frac{\partial L}{\partial \dot{q}_i}\ddot{q}_i\right] + \frac{\partial L}{\partial t}$$

After substitution: dh/dt = -∂L/∂t = 0. ∎

**For natural systems (T quadratic in velocities, V independent of velocities):**
$$h = T + V = E \quad \text{(total energy)}$$

---

### 4. Momentum Conservation

**Condition:** L invariant under translation in direction n̂

For translation q → q + εn̂:
$$\delta L = \frac{\partial L}{\partial \mathbf{q}} \cdot \hat{\mathbf{n}} \, \epsilon = 0$$

This means ∂L/∂qₙ = 0 where qₙ is coordinate along n̂.

**Conserved quantity:** pₙ = ∂L/∂q̇ₙ = linear momentum in direction n̂

**For isolated system:** All translations are symmetries → **p** = Σᵢ mᵢ**v**ᵢ is conserved.

---

### 5. Angular Momentum Conservation

**Condition:** L invariant under rotation about axis n̂

For rotation by angle δφ about n̂:
$$\delta \mathbf{r}_i = \delta\phi \, \hat{\mathbf{n}} \times \mathbf{r}_i$$

**Conserved quantity:** Component of angular momentum about n̂:
$$L_n = \hat{\mathbf{n}} \cdot \sum_i \mathbf{r}_i \times \mathbf{p}_i$$

**For central force:** Rotational symmetry → **L** = **r** × **p** is conserved.

---

### 6. Noether's Theorem (Preview)

**General Statement:**
For every continuous symmetry of the action, there exists a conserved current (in field theory) or conserved quantity (in mechanics).

**Mathematical form:**
If L is invariant under q → q + εQ(q, t), the conserved quantity is:
$$\boxed{I = \sum_i \frac{\partial L}{\partial \dot{q}_i} Q_i}$$

We'll explore this in depth in Week 22!

---

### 7. 🔬 Quantum Mechanics Connection

**Classical → Quantum:**
| Classical | Quantum |
|-----------|---------|
| Conserved quantity I | [Ĥ, Î] = 0 |
| Symmetry generator | Unitary transformation |
| Poisson bracket | Commutator (×iℏ) |

**Examples:**
- Energy conservation ↔ Time evolution operator
- Momentum conservation ↔ Translation operator
- Angular momentum conservation ↔ Rotation operator

---

## ✏️ Worked Examples

### Example 1: Central Force Problem

L = ½m(ṙ² + r²θ̇² + r²sin²θ φ̇²) - V(r)

**Symmetries:**
- θ, φ don't appear → rotational symmetry
- pθ = mr²θ̇ conserved (if we fix the plane)
- pφ = mr²sin²θ φ̇ conserved (angular momentum about z)

**Energy:** E = ½m(ṙ² + r²θ̇² + r²sin²θ φ̇²) + V(r) = const

---

### Example 2: Particle in Homogeneous Field

L = ½m(ẋ² + ẏ² + ż²) - mgz

**Symmetries:**
- x, y cyclic → pₓ = mẋ, pᵧ = mẏ conserved
- z not cyclic (gravity breaks vertical symmetry)
- Time translation → E = ½mv² + mgz conserved

---

## 🔧 Practice Problems

### Level 1
1. For a free particle, identify all cyclic coordinates and conserved quantities.
2. Show that angular momentum is conserved for any central force V(r).

### Level 2
3. A particle moves on a cone z = αr. What symmetries does it have? What's conserved?
4. Two particles interact via V(|r₁ - r₂|). Show total momentum is conserved.

### Level 3
5. Prove that for L = T - V with T homogeneous degree 2 in velocities and V independent of velocities, h = T + V.
6. For the Kepler problem, find all conserved quantities (hint: there's a hidden one!).

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def conservation_laws_demo():
    """Demonstrate conservation laws in central force problem."""
    
    print("=" * 60)
    print("CONSERVATION LAWS IN CENTRAL FORCE")
    print("=" * 60)
    
    # Central force: F = -k/r^2 (Kepler problem)
    k = 1.0
    m = 1.0
    
    def equations(state, t):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        ax = -k * x / (m * r**3)
        ay = -k * y / (m * r**3)
        return [vx, vy, ax, ay]
    
    # Initial conditions (elliptical orbit)
    r0 = 1.0
    v0 = 1.2  # Less than escape velocity
    state0 = [r0, 0, 0, v0]
    
    t = np.linspace(0, 20, 2000)
    solution = odeint(equations, state0, t)
    
    x, y = solution[:, 0], solution[:, 1]
    vx, vy = solution[:, 2], solution[:, 3]
    
    # Compute conserved quantities
    r = np.sqrt(x**2 + y**2)
    v2 = vx**2 + vy**2
    
    E = 0.5 * m * v2 - k / r  # Energy
    Lz = m * (x * vy - y * vx)  # Angular momentum (z-component)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Orbit
    axes[0, 0].plot(x, y, 'b-', lw=1)
    axes[0, 0].scatter([0], [0], c='yellow', s=200, marker='*', zorder=5)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Orbit (Kepler Problem)')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy conservation
    axes[0, 1].plot(t, E, 'r-', lw=1)
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('E')
    axes[0, 1].set_title(f'Energy Conservation\nE = {np.mean(E):.4f} ± {np.std(E):.2e}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Angular momentum conservation
    axes[1, 0].plot(t, Lz, 'g-', lw=1)
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('Lz')
    axes[1, 0].set_title(f'Angular Momentum Conservation\nLz = {np.mean(Lz):.4f} ± {np.std(Lz):.2e}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Phase space (r, ṙ)
    r_dot = (x * vx + y * vy) / r
    axes[1, 1].plot(r, r_dot, 'b-', lw=1)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('ṙ')
    axes[1, 1].set_title('Phase Space (Effective 1D Problem)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('conservation_laws.png', dpi=150)
    plt.show()
    
    print(f"\nEnergy: E = {np.mean(E):.6f} (std: {np.std(E):.2e})")
    print(f"Angular momentum: Lz = {np.mean(Lz):.6f} (std: {np.std(Lz):.2e})")
    print("\nBoth are conserved to numerical precision!")

conservation_laws_demo()
```

---

## 📝 Summary

### Symmetry-Conservation Correspondence

| Symmetry | Conserved Quantity | Generator |
|----------|-------------------|-----------|
| Time translation (∂L/∂t = 0) | Energy h | Hamiltonian |
| Space translation (∂L/∂x = 0) | Momentum pₓ | Translation |
| Rotation (∂L/∂φ = 0) | Angular momentum Lφ | Rotation |

### Key Formula (Noether)

If L invariant under q → q + εQ:
$$I = \sum_i \frac{\partial L}{\partial \dot{q}_i} Q_i = \text{const}$$

---

## ✅ Daily Checklist

- [ ] Connect cyclic coordinates to symmetries
- [ ] Derive energy conservation
- [ ] Derive momentum conservation
- [ ] Derive angular momentum conservation
- [ ] Preview Noether's theorem
- [ ] Complete computational exercises

---

## 🔮 Preview: Day 146

Tomorrow is our **Computational Lab** where we simulate various mechanical systems using Lagrangian methods!
