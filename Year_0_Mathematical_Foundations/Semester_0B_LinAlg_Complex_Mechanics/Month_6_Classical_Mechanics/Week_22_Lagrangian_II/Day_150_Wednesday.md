# Day 150: The Two-Body Problem — Reduced Mass

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Two-Body Reduction |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Separate center of mass and relative motion
2. Define and use reduced mass
3. Transform the two-body problem to an equivalent one-body problem
4. Apply to binary stars and molecular systems
5. Connect to quantum mechanics (hydrogen atom, molecular spectra)

---

## 📖 Core Content

### 1. The General Two-Body Problem

**Setup:** Two particles with masses m₁, m₂ at positions **r**₁, **r**₂, interacting via V(|**r**₁ - **r**₂|).

**Lagrangian:**
$$L = \frac{1}{2}m_1|\dot{\mathbf{r}}_1|^2 + \frac{1}{2}m_2|\dot{\mathbf{r}}_2|^2 - V(|\mathbf{r}_1 - \mathbf{r}_2|)$$

This is a 6-DOF problem. Can we simplify?

---

### 2. Center of Mass Transformation

**Define:**

**Center of mass:**
$$\mathbf{R} = \frac{m_1\mathbf{r}_1 + m_2\mathbf{r}_2}{m_1 + m_2} = \frac{m_1\mathbf{r}_1 + m_2\mathbf{r}_2}{M}$$

**Relative position:**
$$\mathbf{r} = \mathbf{r}_1 - \mathbf{r}_2$$

**Inverse relations:**
$$\mathbf{r}_1 = \mathbf{R} + \frac{m_2}{M}\mathbf{r}, \quad \mathbf{r}_2 = \mathbf{R} - \frac{m_1}{M}\mathbf{r}$$

---

### 3. Reduced Mass

**Definition:** The **reduced mass** is:
$$\boxed{\mu = \frac{m_1 m_2}{m_1 + m_2}}$$

**Properties:**
- μ < min(m₁, m₂)
- If m₁ ≫ m₂: μ ≈ m₂
- If m₁ = m₂: μ = m/2
- 1/μ = 1/m₁ + 1/m₂

---

### 4. Separated Lagrangian

**Kinetic energy transforms:**
$$T = \frac{1}{2}m_1|\dot{\mathbf{r}}_1|^2 + \frac{1}{2}m_2|\dot{\mathbf{r}}_2|^2 = \frac{1}{2}M|\dot{\mathbf{R}}|^2 + \frac{1}{2}\mu|\dot{\mathbf{r}}|^2$$

**Full Lagrangian:**
$$\boxed{L = \frac{1}{2}M|\dot{\mathbf{R}}|^2 + \frac{1}{2}\mu|\dot{\mathbf{r}}|^2 - V(r)}$$

**Separation achieved!**
- CM motion: Free particle of mass M
- Relative motion: Particle of mass μ in potential V(r)

---

### 5. Equations of Motion

**Center of mass:**
$$M\ddot{\mathbf{R}} = 0 \quad \Rightarrow \quad \mathbf{R} = \mathbf{R}_0 + \mathbf{V}_{CM}t$$

**Relative motion:**
$$\mu\ddot{\mathbf{r}} = -\nabla V(r) = f(r)\hat{\mathbf{r}}$$

The relative motion is the central force problem we solved yesterday!

---

### 6. Conservation Laws

**Total momentum:**
$$\mathbf{P} = m_1\dot{\mathbf{r}}_1 + m_2\dot{\mathbf{r}}_2 = M\dot{\mathbf{R}} = \text{const}$$

**Angular momentum (relative):**
$$\mathbf{L} = \mu\mathbf{r} \times \dot{\mathbf{r}} = \text{const}$$

**Total energy:**
$$E = \frac{1}{2}M|\dot{\mathbf{R}}|^2 + \frac{1}{2}\mu|\dot{\mathbf{r}}|^2 + V(r) = E_{CM} + E_{rel}$$

---

### 7. Applications

**Binary Stars:**
- Two stars orbit their common CM
- Period: T² = 4π²a³/(G(m₁+m₂))
- Individual orbits: a₁ = a·m₂/M, a₂ = a·m₁/M

**Molecules:**
- Vibrational frequency: ω = √(k/μ)
- For H-Cl: μ ≈ m_H (since m_Cl ≫ m_H)
- For H₂: μ = m_H/2

**Hydrogen atom:**
- Electron-proton system
- μ = m_e·m_p/(m_e + m_p) ≈ m_e(1 - m_e/m_p)
- Small correction to energy levels

---

### 8. 🔬 Quantum Connection

**Schrödinger equation for two particles:**
$$\left[-\frac{\hbar^2}{2m_1}\nabla_1^2 - \frac{\hbar^2}{2m_2}\nabla_2^2 + V(|\mathbf{r}_1-\mathbf{r}_2|)\right]\Psi = E\Psi$$

**After separation:**
$$\Psi(\mathbf{r}_1, \mathbf{r}_2) = \phi_{CM}(\mathbf{R})\psi_{rel}(\mathbf{r})$$

- CM: Free particle wave function
- Relative: Hydrogen-like with μ instead of m_e

**Rydberg constant correction:**
$$R_\infty \to R_M = R_\infty \cdot \frac{\mu}{m_e} = R_\infty \cdot \frac{1}{1 + m_e/M_{nucleus}}$$

---

## ✏️ Worked Examples

### Example 1: Earth-Moon System

**Given:** M_E = 6×10²⁴ kg, M_M = 7.4×10²² kg, separation a = 384,000 km

**Reduced mass:**
$$\mu = \frac{M_E \cdot M_M}{M_E + M_M} = \frac{6 \times 10^{24} \times 7.4 \times 10^{22}}{6.074 \times 10^{24}} = 7.3 \times 10^{22} \text{ kg} \approx M_M$$

**Center of mass location:**
$$R_{E} = \frac{M_M}{M_E + M_M} \cdot a = \frac{7.4 \times 10^{22}}{6.074 \times 10^{24}} \times 384,000 \approx 4,670 \text{ km}$$

This is inside Earth (radius 6,371 km)!

---

### Example 2: Hydrogen Atom Correction

**Exact reduced mass:**
$$\mu = \frac{m_e m_p}{m_e + m_p} = m_e \cdot \frac{1}{1 + m_e/m_p}$$

**Ratio:** m_e/m_p ≈ 1/1836

**Energy correction:**
$$E_n = -\frac{13.6 \text{ eV}}{n^2} \times \frac{\mu}{m_e} = -\frac{13.6 \text{ eV}}{n^2} \times \frac{1}{1 + 1/1836}$$

**Shift:** About 0.05% — measurable in precision spectroscopy!

---

## 🔧 Practice Problems

### Level 1
1. Calculate the reduced mass of the H₂ molecule.
2. For a binary star system with m₁ = 2M☉, m₂ = M☉, find μ and the CM location.

### Level 2
3. Derive the separated Lagrangian from the original.
4. For a diatomic molecule with spring constant k and reduced mass μ, find the vibrational frequency.

### Level 3
5. Calculate the reduced mass correction for deuterium (D = p + n + e).
6. Two stars of equal mass orbit with period T. One explodes, losing half its mass instantly. Describe the subsequent motion.

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def two_body_simulation():
    """Simulate two-body problem showing CM and relative motion."""
    
    print("=" * 70)
    print("TWO-BODY PROBLEM SIMULATION")
    print("=" * 70)
    
    # Binary star parameters
    m1, m2 = 2.0, 1.0  # Solar masses (arbitrary units)
    M = m1 + m2
    mu = m1 * m2 / M
    G = 1.0  # Gravitational constant (scaled)
    
    print(f"\nMasses: m1 = {m1}, m2 = {m2}")
    print(f"Total mass: M = {M}")
    print(f"Reduced mass: μ = {mu:.4f}")
    
    def two_body_eom(state, t):
        """Full two-body equations of motion."""
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = state
        
        # Relative position
        rx, ry = x1 - x2, y1 - y2
        r = np.sqrt(rx**2 + ry**2)
        
        # Gravitational force
        F = G * m1 * m2 / r**2
        Fx = -F * rx / r
        Fy = -F * ry / r
        
        # Accelerations
        ax1, ay1 = Fx / m1, Fy / m1
        ax2, ay2 = -Fx / m2, -Fy / m2
        
        return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]
    
    # Initial conditions: circular orbit
    a = 1.0  # Semi-major axis
    v_orb = np.sqrt(G * M / a)  # Orbital velocity
    
    # Position particle 1 at (a * m2/M, 0), particle 2 at (-a * m1/M, 0)
    r1 = a * m2 / M
    r2 = a * m1 / M
    v1 = v_orb * m2 / M
    v2 = v_orb * m1 / M
    
    state0 = [r1, 0, -r2, 0, 0, v1, 0, -v2]
    
    # Simulate
    T_orbit = 2 * np.pi * np.sqrt(a**3 / (G * M))
    t = np.linspace(0, 2*T_orbit, 1000)
    sol = odeint(two_body_eom, state0, t)
    
    x1, y1 = sol[:, 0], sol[:, 1]
    x2, y2 = sol[:, 2], sol[:, 3]
    
    # Center of mass
    X_cm = (m1 * x1 + m2 * x2) / M
    Y_cm = (m1 * y1 + m2 * y2) / M
    
    # Relative position
    rx = x1 - x2
    ry = y1 - y2
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Lab frame
    ax = axes[0, 0]
    ax.plot(x1, y1, 'b-', lw=1.5, label=f'm₁ = {m1}')
    ax.plot(x2, y2, 'r-', lw=1.5, label=f'm₂ = {m2}')
    ax.scatter([X_cm[0]], [Y_cm[0]], c='green', s=100, marker='x', label='CM')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Two-Body Orbits (Lab Frame)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CM frame
    ax = axes[0, 1]
    ax.plot(x1 - X_cm, y1 - Y_cm, 'b-', lw=1.5, label='m₁')
    ax.plot(x2 - X_cm, y2 - Y_cm, 'r-', lw=1.5, label='m₂')
    ax.scatter([0], [0], c='green', s=100, marker='x', label='CM')
    ax.set_xlabel('x - X_cm')
    ax.set_ylabel('y - Y_cm')
    ax.set_title('Orbits in CM Frame')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Relative motion
    ax = axes[1, 0]
    ax.plot(rx, ry, 'purple', lw=2)
    ax.scatter([0], [0], c='black', s=100, marker='o', label='Origin')
    ax.set_xlabel('rₓ = x₁ - x₂')
    ax.set_ylabel('rᵧ = y₁ - y₂')
    ax.set_title(f'Relative Motion (reduced mass μ = {mu:.3f})')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CM drift check
    ax = axes[1, 1]
    ax.plot(t/T_orbit, X_cm, 'b-', lw=2, label='X_cm')
    ax.plot(t/T_orbit, Y_cm, 'r-', lw=2, label='Y_cm')
    ax.set_xlabel('t / T_orbit')
    ax.set_ylabel('CM Position')
    ax.set_title(f'CM Position (should be constant)\nstd = {np.std(X_cm):.2e}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('two_body_problem.png', dpi=150)
    plt.show()

two_body_simulation()
```

---

## 📝 Summary

### Two-Body Reduction

$$\text{6 DOF} \to \text{CM (3 DOF)} + \text{Relative (3 DOF)}$$

### Key Formulas

| Quantity | Formula |
|----------|---------|
| Reduced mass | μ = m₁m₂/(m₁+m₂) |
| CM position | **R** = (m₁**r**₁ + m₂**r**₂)/M |
| Relative position | **r** = **r**₁ - **r**₂ |
| Kinetic energy | T = ½MṘ² + ½μṙ² |

---

## ✅ Daily Checklist

- [ ] Define CM and relative coordinates
- [ ] Derive reduced mass
- [ ] Separate Lagrangian
- [ ] Apply to binary systems
- [ ] Connect to quantum mechanics
- [ ] Complete computational exercises

---

## 🔮 Preview: Day 151

Tomorrow we study **Small Oscillations** — linearizing equations of motion near equilibrium to find normal modes!
