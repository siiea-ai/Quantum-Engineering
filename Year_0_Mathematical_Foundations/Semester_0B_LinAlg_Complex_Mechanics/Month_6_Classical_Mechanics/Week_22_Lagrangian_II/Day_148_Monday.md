# Day 148: Noether's Theorem — Symmetries and Conservation Laws

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Noether's Theorem |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. State and prove Noether's theorem
2. Derive conservation laws from symmetries
3. Apply to translations, rotations, and time translations
4. Understand the deep structure of physics laws
5. Connect to quantum mechanics

---

## 📚 Required Reading

### Primary Text: Goldstein
- **Chapter 13, Section 13.7**: Noether's Theorem

### Alternative: Landau & Lifshitz
- **Chapter 2**: Conservation Laws

---

## 📖 Core Content: Theory and Concepts

### 1. The Profound Connection

**Emmy Noether (1918):** Every continuous symmetry of the action corresponds to a conserved quantity.

This is arguably the most beautiful theorem in physics!

| Symmetry | Conserved Quantity | Generator |
|----------|-------------------|-----------|
| Time translation | Energy | Hamiltonian |
| Space translation | Momentum | Translation operator |
| Rotation | Angular momentum | Rotation operator |
| Gauge transformation | Charge | Phase operator |
| Lorentz boost | Center of mass motion | Boost generator |

---

### 2. Continuous Symmetries

**Definition:** A continuous symmetry is a transformation that:
1. Depends on a continuous parameter ε
2. Leaves the action invariant (or changes it by a constant)
3. Reduces to identity when ε = 0

**Infinitesimal transformation:**
$$q_i \to q_i + \epsilon Q_i(q, \dot{q}, t)$$
$$t \to t + \epsilon \tau(q, \dot{q}, t)$$

---

### 3. Statement of Noether's Theorem

**Theorem:** If the action S = ∫L dt is invariant under the infinitesimal transformation:
$$q_i \to q_i + \epsilon Q_i, \quad t \to t + \epsilon \tau$$

then the quantity:
$$\boxed{I = \sum_i \frac{\partial L}{\partial \dot{q}_i} Q_i - h\tau}$$

is conserved along solutions of the Euler-Lagrange equations.

Here h = Σᵢ pᵢq̇ᵢ - L is the energy function.

---

### 4. Proof of Noether's Theorem

**Setup:** Under the transformation, the action becomes:
$$S' = \int_{t_1'}^{t_2'} L(q + \epsilon Q, \dot{q} + \epsilon \dot{Q}, t + \epsilon\tau)\,dt'$$

**Invariance condition:** δS = S' - S = 0 (to first order in ε)

**Detailed calculation:**

The change in Lagrangian:
$$\delta L = \sum_i \left(\frac{\partial L}{\partial q_i}Q_i + \frac{\partial L}{\partial \dot{q}_i}\dot{Q}_i\right) + \frac{\partial L}{\partial t}\tau$$

Using the Euler-Lagrange equations:
$$\frac{\partial L}{\partial q_i} = \frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i}$$

We can write:
$$\delta L = \sum_i \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}Q_i\right) + \frac{\partial L}{\partial t}\tau$$

The full variation of the action includes the change in integration limits:
$$\delta S = \int_{t_1}^{t_2}\left[\delta L + L\frac{d\tau}{dt}\right]dt = 0$$

After careful manipulation:
$$\frac{d}{dt}\left[\sum_i \frac{\partial L}{\partial \dot{q}_i}Q_i - h\tau\right] = 0$$

Therefore I = Σᵢ pᵢQᵢ - hτ is conserved. ∎

---

### 5. Special Cases

**Case 1: Time Translation Symmetry**
- Q_i = 0, τ = 1 (shift time by constant)
- Conserved: I = -h = L - Σ pᵢq̇ᵢ
- For natural systems: **h = E = T + V** (energy)

**Case 2: Space Translation Symmetry**
- Q_i = δᵢⱼ (shift coordinate j), τ = 0
- Conserved: I = pⱼ = ∂L/∂q̇ⱼ
- **Momentum in direction j**

**Case 3: Rotation Symmetry**
- For rotation about z-axis: Q_x = -y, Q_y = x, τ = 0
- Conserved: I = xp_y - yp_x = **L_z** (angular momentum)

---

### 6. General Rotation

For rotation by angle ε about axis **n̂**:
$$\delta\mathbf{r} = \epsilon\,\hat{\mathbf{n}} \times \mathbf{r}$$

The conserved quantity is:
$$\mathbf{L} \cdot \hat{\mathbf{n}} = (\mathbf{r} \times \mathbf{p}) \cdot \hat{\mathbf{n}}$$

For full rotational symmetry: entire angular momentum vector **L** is conserved.

---

### 7. 🔬 Quantum Mechanics Connection

**Noether's theorem in QM:**

| Classical | Quantum |
|-----------|---------|
| Conserved quantity I | [Ĥ, Î] = 0 |
| Symmetry generator | Unitary transformation e^{-iεÎ/ℏ} |
| Infinitesimal transformation | δψ = -iεÎψ/ℏ |

**Examples:**
- Energy conservation ↔ Time translation: Û(t) = e^{-iĤt/ℏ}
- Momentum conservation ↔ Space translation: T̂(a) = e^{-ip̂a/ℏ}
- Angular momentum ↔ Rotation: R̂(θ) = e^{-iL̂θ/ℏ}

**The conserved quantity generates its own symmetry!**

---

### 8. Beyond Mechanics: Field Theory

In field theory, Noether's theorem gives **conserved currents**:
$$\partial_\mu j^\mu = 0$$

Examples:
- Energy-momentum tensor T^μν from spacetime translations
- Electric current j^μ from gauge symmetry
- Chiral currents from chiral symmetry

---

## ✏️ Worked Examples

### Example 1: Free Particle

**Lagrangian:** L = ½m(ẋ² + ẏ² + ż²)

**Symmetries:**
1. Time translation (t → t + ε): E = ½mv² conserved
2. x-translation (x → x + ε): pₓ = mẋ conserved
3. Rotation about z (x → x - εy, y → y + εx): L_z = m(xẏ - yẋ) conserved

All 10 Galilean symmetries → 10 conserved quantities!

---

### Example 2: Central Force

**Lagrangian:** L = ½m(ṙ² + r²θ̇² + r²sin²θ φ̇²) - V(r)

**Symmetries:**
- Rotation about any axis (spherical symmetry)
- Time translation

**Conserved:**
- **L** = m**r** × **v** (angular momentum vector)
- E = T + V (energy)

---

### Example 3: Charged Particle in Uniform B Field

**Lagrangian:** L = ½mv² + e**A**·**v** with **A** = ½**B** × **r**

**Symmetry:** Translation along **B** direction

**Conserved:** Component of canonical momentum along **B**:
$$p_z = m\dot{z} + eA_z$$

---

## 🔧 Practice Problems

### Level 1: Direct Application
1. For a particle in gravity L = ½m(ẋ² + ẏ²) - mgy, find all conserved quantities using Noether's theorem.

2. Show explicitly that Lz = xpy - ypx is conserved for any central force.

### Level 2: Noether Derivations
3. For L = ½m(ẋ² + ẏ²) - V(x² + y²), derive the conserved angular momentum using Noether's theorem.

4. Consider L = ½m(ẋ² + ẏ²) - V(x - vt). What symmetry exists? What quantity is conserved?

### Level 3: Advanced
5. For a system with L = ½mṙ² - V(r) + f(θ)θ̇, under what conditions is angular momentum conserved?

6. Prove that for the Kepler problem, the Runge-Lenz vector is associated with a "hidden" symmetry.

### Level 4: Field Theory Preview
7. For a scalar field φ with L = ½(∂ₜφ)² - ½(∇φ)² - V(φ), derive the energy-momentum conservation from spacetime translation invariance.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def noether_verification():
    """Verify Noether's theorem for various systems."""
    
    print("=" * 70)
    print("NOETHER'S THEOREM VERIFICATION")
    print("=" * 70)
    
    # System 1: Central force (angular momentum conservation)
    print("\n1. Central Force: Angular Momentum Conservation")
    print("-" * 50)
    
    def central_force_eom(state, t, k=1):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        ax = -k * x / r**3
        ay = -k * y / r**3
        return [vx, vy, ax, ay]
    
    # Initial conditions
    state0 = [1, 0, 0, 1]  # Circular-ish orbit
    t = np.linspace(0, 20, 2000)
    sol = odeint(central_force_eom, state0, t)
    
    x, y, vx, vy = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    
    # Angular momentum Lz = x*vy - y*vx
    Lz = x * vy - y * vx
    
    # Energy E = T + V
    r = np.sqrt(x**2 + y**2)
    E = 0.5 * (vx**2 + vy**2) - 1/r
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(x, y, 'b-', lw=1)
    axes[0, 0].scatter([0], [0], c='orange', s=200, marker='*')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Central Force Orbit')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t, Lz, 'g-', lw=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Lz')
    axes[0, 1].set_title(f'Angular Momentum (Noether: rotation)\nstd = {np.std(Lz):.2e}')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(t, E, 'r-', lw=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('E')
    axes[1, 0].set_title(f'Energy (Noether: time translation)\nstd = {np.std(E):.2e}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # System 2: Particle in uniform gravity (px conserved, py not)
    print("\n2. Uniform Gravity: Linear Momentum")
    print("-" * 50)
    
    def gravity_eom(state, t, g=10):
        x, y, vx, vy = state
        return [vx, vy, 0, -g]
    
    state0 = [0, 0, 5, 10]  # Projectile
    t2 = np.linspace(0, 2, 200)
    sol2 = odeint(gravity_eom, state0, t2)
    
    px = sol2[:, 2]  # Should be conserved
    py = sol2[:, 3]  # Not conserved
    
    axes[1, 1].plot(t2, px, 'b-', lw=2, label=f'px (std={np.std(px):.2e})')
    axes[1, 1].plot(t2, py, 'r-', lw=2, label='py (not conserved)')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Momentum')
    axes[1, 1].set_title('Projectile: px conserved (x-translation)\npy not conserved (no y-translation symmetry)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noether_verification.png', dpi=150)
    plt.show()
    
    print(f"Central force: Lz = {np.mean(Lz):.6f} ± {np.std(Lz):.2e}")
    print(f"Central force: E = {np.mean(E):.6f} ± {np.std(E):.2e}")
    print(f"Gravity: px = {np.mean(px):.6f} ± {np.std(px):.2e}")

noether_verification()


def symmetry_transformations():
    """Visualize symmetry transformations and their generators."""
    
    print("\n" + "=" * 70)
    print("SYMMETRY TRANSFORMATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original trajectory
    t = np.linspace(0, 2*np.pi, 100)
    x = 2*np.cos(t) + 1
    y = np.sin(t)
    
    # 1. Translation
    ax = axes[0]
    ax.plot(x, y, 'b-', lw=2, label='Original')
    for dx in [0.5, 1.0, 1.5]:
        ax.plot(x + dx, y, '--', alpha=0.5, label=f'Translated by {dx}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Translation Symmetry\n→ Momentum Conservation')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Rotation
    ax = axes[1]
    ax.plot(x, y, 'b-', lw=2, label='Original')
    for theta in [np.pi/6, np.pi/3, np.pi/2]:
        x_rot = x*np.cos(theta) - y*np.sin(theta)
        y_rot = x*np.sin(theta) + y*np.cos(theta)
        ax.plot(x_rot, y_rot, '--', alpha=0.5, label=f'Rotated by {np.degrees(theta):.0f}°')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Rotation Symmetry\n→ Angular Momentum Conservation')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Time translation (show same orbit at different times)
    ax = axes[2]
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, t0 in enumerate([0, 0.5, 1.0, 1.5, 2.0]):
        idx = int(t0 * len(t) / (2*np.pi)) % len(t)
        ax.scatter([x[idx]], [y[idx]], c=[colors[i]], s=100, label=f't = {t0:.1f}')
    ax.plot(x, y, 'k-', lw=1, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Time Translation Symmetry\n→ Energy Conservation')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('symmetry_transformations.png', dpi=150)
    plt.show()

symmetry_transformations()
```

---

## 📝 Summary

### Noether's Theorem

**Statement:** Every continuous symmetry → conserved quantity

$$I = \sum_i p_i Q_i - h\tau = \text{constant}$$

### Key Correspondences

| Transformation | Q | τ | Conserved I |
|---------------|---|---|-------------|
| Time translation | 0 | 1 | -h (energy) |
| Space translation | δᵢⱼ | 0 | pⱼ (momentum) |
| Rotation about n̂ | n̂ × r | 0 | L·n̂ (ang. mom.) |

### Deep Structure

$$\text{Symmetry} \xleftrightarrow{\text{Noether}} \text{Conservation Law}$$

---

## ✅ Daily Checklist

- [ ] Understand continuous symmetries
- [ ] State Noether's theorem correctly
- [ ] Derive energy from time translation
- [ ] Derive momentum from space translation
- [ ] Derive angular momentum from rotation
- [ ] Connect to quantum mechanics
- [ ] Complete computational verification

---

## 🔮 Preview: Day 149

Tomorrow we study the **Central Force Problem** — applying our Lagrangian tools to planetary motion and the Kepler problem!
