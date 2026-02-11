# Day 144: Constraints & Lagrange Multipliers

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Constrained Systems |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Incorporate holonomic constraints via coordinate elimination
2. Use Lagrange multipliers for constraints
3. Find constraint forces from multipliers
4. Handle non-holonomic constraints
5. Apply to physical systems (pendulum, rolling, etc.)

---

## 📖 Core Content

### 1. Methods for Handling Constraints

**Two approaches:**

**Method 1: Eliminate constraints**
- Use n - k independent coordinates
- Constraints absorbed into coordinate choice
- No access to constraint forces

**Method 2: Lagrange multipliers**
- Keep all n coordinates + constraints
- Add constraint terms to Lagrangian
- Constraint forces emerge automatically

---

### 2. Lagrange Multipliers for Holonomic Constraints

For constraints fⱼ(q, t) = 0, j = 1, ..., k:

**Modified Lagrangian:**
$$L' = L + \sum_{j=1}^{k} \lambda_j f_j(q, t)$$

**Euler-Lagrange equations:**
$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = \sum_j \lambda_j \frac{\partial f_j}{\partial q_i}$$

plus the k constraint equations fⱼ = 0.

**The multipliers λⱼ are the generalized constraint forces!**

---

### 3. Example: Simple Pendulum with Multiplier

**Setup:** Mass m, position (x, y), constraint x² + y² - L² = 0.

**Lagrangian:** L = ½m(ẋ² + ẏ²) - mgy

**Constraint:** f = x² + y² - L² = 0

**Modified Lagrangian:**
L' = ½m(ẋ² + ẏ²) - mgy + λ(x² + y² - L²)

**Euler-Lagrange:**
- x: mẍ = 2λx → λ = mẍ/(2x)
- y: mÿ = -mg + 2λy

**Constraint force:** F_c = 2λ(x, y) = tension along the rod!

---

### 4. Advantages of Lagrange Multipliers

1. **Constraint forces computed automatically**
2. **Systematic for complex systems**
3. **Works when elimination is difficult**
4. **Natural for numerical methods**

---

### 5. Non-holonomic Constraints

Constraints involving velocities that can't be integrated:
$$g_j(q, \dot{q}, t) = 0$$

**Example:** Rolling without slipping

For a disk: v = Rω (velocity = radius × angular velocity)

This relates ẋ and θ̇ but cannot be integrated to a constraint on x and θ alone.

**Treatment:** Use d'Alembert's principle with virtual displacements compatible with constraints.

---

## 🔧 Practice Problems

### Level 1
1. Pendulum with Lagrange multiplier: Find tension as function of θ and θ̇.
2. Particle on sphere: Use multiplier to find normal force.

### Level 2
3. Atwood machine with multiplier: Find string tension.
4. Bead on rotating hoop: Find the normal force from the hoop.

### Level 3
5. Rolling disk on plane: Set up equations with non-holonomic constraint.
6. Spherical pendulum with constraint.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendulum_with_multiplier():
    """Solve pendulum using Cartesian coords + Lagrange multiplier."""
    
    print("=" * 60)
    print("PENDULUM WITH LAGRANGE MULTIPLIER")
    print("=" * 60)
    
    # Parameters
    m, g, L = 1.0, 10.0, 1.0
    
    def equations(state, t):
        x, y, vx, vy = state
        
        # Constraint: x² + y² = L²
        # Differentiate twice: ẍx + ẏy + ẋ² + ẏ² = 0
        # From EL: mẍ = 2λx, mÿ = -mg + 2λy
        # Substitute: 2λ(x² + y²)/m + (vx² + vy²) = 0
        # λ = -m(vx² + vy²)/(2L²)
        
        # But we need to enforce constraint more carefully
        # Use constraint acceleration: ẍx + ÿy = -(ẋ² + ẏ²)
        
        # Combined with EL: ẍ = 2λx/m, ÿ = -g + 2λy/m
        # (2λx/m)x + (-g + 2λy/m)y = -(vx² + vy²)
        # 2λL²/m = gy - (vx² + vy²)
        
        lambda_val = m * (g*y - vx**2 - vy**2) / (2 * L**2)
        
        ax = 2 * lambda_val * x / m
        ay = -g + 2 * lambda_val * y / m
        
        return [vx, vy, ax, ay]
    
    # Initial conditions (start at angle, release from rest)
    theta0 = np.pi/4
    x0, y0 = L * np.sin(theta0), -L * np.cos(theta0)
    state0 = [x0, y0, 0, 0]
    
    t = np.linspace(0, 5, 500)
    solution = odeint(equations, state0, t)
    
    x, y = solution[:, 0], solution[:, 1]
    vx, vy = solution[:, 2], solution[:, 3]
    
    # Compute tension (constraint force magnitude)
    tension = np.zeros_like(t)
    for i in range(len(t)):
        lambda_val = m * (g*y[i] - vx[i]**2 - vy[i]**2) / (2 * L**2)
        F_constraint = 2 * lambda_val * np.array([x[i], y[i]])
        tension[i] = np.linalg.norm(F_constraint)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory
    axes[0, 0].plot(x, y, 'b-', lw=2)
    axes[0, 0].scatter([0], [0], c='black', s=100, zorder=5)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Pendulum Trajectory')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Angle vs time
    theta = np.arctan2(x, -y)
    axes[0, 1].plot(t, theta, 'b-', lw=2)
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('θ')
    axes[0, 1].set_title('Angle vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Tension vs time
    axes[1, 0].plot(t, tension, 'r-', lw=2)
    axes[1, 0].axhline(y=m*g, color='k', linestyle='--', label='mg')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('Tension')
    axes[1, 0].set_title('String Tension (from Lagrange multiplier)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Constraint verification
    constraint = x**2 + y**2 - L**2
    axes[1, 1].plot(t, constraint, 'g-', lw=2)
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('x² + y² - L²')
    axes[1, 1].set_title('Constraint Verification (should be ≈ 0)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lagrange_multiplier_pendulum.png', dpi=150)
    plt.show()
    
    print(f"\nMax tension: {np.max(tension):.4f}")
    print(f"Min tension: {np.min(tension):.4f}")
    print(f"At lowest point: T = m(g + v²/L)")

pendulum_with_multiplier()
```

---

## 📝 Summary

### Lagrange Multiplier Method

For constraints fⱼ(q, t) = 0:

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = \sum_j \lambda_j \frac{\partial f_j}{\partial q_i}$$

**Key insight:** λⱼ represents the generalized constraint force!

---

## ✅ Daily Checklist

- [ ] Understand coordinate elimination method
- [ ] Apply Lagrange multipliers correctly
- [ ] Extract constraint forces from λ
- [ ] Handle non-holonomic constraints
- [ ] Complete computational exercises

---

## 🔮 Preview: Day 145

Tomorrow we explore **Symmetries and Conservation Laws** — the deep connection between continuous symmetries and conserved quantities (Noether's theorem preview)!
