# Day 149: Central Force Problem — Kepler and Beyond

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Theory: Central Force Dynamics |
| Afternoon | 2:00 PM - 4:30 PM | 2.5 hours | Problem Solving |
| Evening | 7:00 PM - 8:00 PM | 1 hour | Computational Lab |

**Total Study Time: 7 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Reduce the central force problem to one dimension
2. Use angular momentum conservation effectively
3. Analyze effective potential and orbit shapes
4. Derive Kepler's laws from Newton's gravity
5. Classify orbits by energy and angular momentum
6. Understand the connection to quantum hydrogen atom

---

## 📖 Core Content

### 1. Setup: Central Force

A **central force** depends only on distance and points radially:
$$\mathbf{F} = f(r)\hat{\mathbf{r}} = -\frac{dV}{dr}\hat{\mathbf{r}}$$

**Examples:**
- Gravity: f(r) = -GMm/r²
- Coulomb: f(r) = kq₁q₂/r²
- Harmonic: f(r) = -kr

---

### 2. Conservation Laws

**Angular momentum** (rotation symmetry):
$$\mathbf{L} = \mathbf{r} \times \mathbf{p} = \text{constant vector}$$

**Consequences:**
1. Motion confined to plane ⊥ **L**
2. Magnitude L = mr²θ̇ = const

**Energy** (time translation):
$$E = \frac{1}{2}m(\dot{r}^2 + r^2\dot{\theta}^2) + V(r) = \text{const}$$

---

### 3. Reduction to 1D: Effective Potential

Using L = mr²θ̇:
$$E = \frac{1}{2}m\dot{r}^2 + \frac{L^2}{2mr^2} + V(r)$$

Define the **effective potential**:
$$\boxed{V_{\text{eff}}(r) = V(r) + \frac{L^2}{2mr^2}}$$

The radial motion is equivalent to 1D motion in V_eff:
$$\frac{1}{2}m\dot{r}^2 + V_{\text{eff}}(r) = E$$

---

### 4. Effective Potential Analysis

**Centrifugal barrier:** L²/(2mr²) prevents r → 0 (if L ≠ 0)

**For gravity V(r) = -GMm/r:**
$$V_{\text{eff}} = -\frac{GMm}{r} + \frac{L^2}{2mr^2}$$

**Critical points:** dV_eff/dr = 0 gives circular orbit radius:
$$r_c = \frac{L^2}{GMm^2}$$

---

### 5. Orbit Classification

| Energy E | Orbit Type | Eccentricity |
|----------|------------|--------------|
| E < V_eff(min) | Impossible | - |
| E = V_eff(min) | Circular | e = 0 |
| V_eff(min) < E < 0 | Ellipse | 0 < e < 1 |
| E = 0 | Parabola | e = 1 |
| E > 0 | Hyperbola | e > 1 |

---

### 6. Orbit Equation (Binet's Method)

Using u = 1/r and θ as independent variable:
$$\frac{d^2u}{d\theta^2} + u = -\frac{m}{L^2 u^2}f(1/u)$$

**For gravity f = -GMm/r² = -GMmu²:**
$$\frac{d^2u}{d\theta^2} + u = \frac{GMm^2}{L^2}$$

**Solution:**
$$u = \frac{GMm^2}{L^2}(1 + e\cos(\theta - \theta_0))$$

Or in standard form:
$$\boxed{r = \frac{p}{1 + e\cos\theta}}$$

where p = L²/(GMm²) is the **semi-latus rectum**.

---

### 7. Kepler's Laws (Derived!)

**First Law:** Planets move in ellipses with Sun at one focus.
→ Follows from orbit equation with 0 < e < 1

**Second Law:** Equal areas in equal times.
→ Direct consequence of L = mr²θ̇ = const:
$$\frac{dA}{dt} = \frac{1}{2}r^2\dot{\theta} = \frac{L}{2m} = \text{const}$$

**Third Law:** T² ∝ a³
→ Period T = Area/rate = πab/(L/2m) = 2πa^{3/2}/√(GM)
$$T^2 = \frac{4\pi^2}{GM}a^3$$

---

### 8. 🔬 Quantum Connection: Hydrogen Atom

The classical Kepler problem maps to the quantum hydrogen atom!

| Classical | Quantum |
|-----------|---------|
| Orbit with E, L | State with n, ℓ |
| r_c = L²/(GMm²) | Bohr radius a₀ = ℏ²/(me²) |
| E = -GMm/(2a) | Eₙ = -13.6 eV/n² |
| Closed ellipse | Stationary state |
| L conservation | [Ĥ, L̂²] = 0 |

---

## ✏️ Worked Examples

### Example 1: Circular Orbit Conditions

**Find:** Conditions for circular orbit in gravity.

**Solution:**
Circular means ṙ = 0 and r̈ = 0.

From E = ½mṙ² + V_eff(r):
- ṙ = 0: E = V_eff(r)

From radial equation mr̈ = -dV_eff/dr:
- r̈ = 0: dV_eff/dr = 0

For gravity:
$$\frac{dV_{\text{eff}}}{dr} = \frac{GMm}{r^2} - \frac{L^2}{mr^3} = 0$$

$$r_c = \frac{L^2}{GMm^2}$$

Orbital velocity: v = L/(mr_c) = √(GM/r_c)

Energy: E = V_eff(r_c) = -GMm/(2r_c)

---

### Example 2: Escape Velocity

**Find:** Minimum velocity to escape from surface of Earth.

**Solution:**
Escape means E = 0 (parabolic orbit):
$$\frac{1}{2}mv^2 - \frac{GMm}{R} = 0$$

$$v_{\text{escape}} = \sqrt{\frac{2GM}{R}} \approx 11.2 \text{ km/s}$$

---

## 🔧 Practice Problems

### Level 1
1. For circular orbit at radius r₀, find the angular velocity ω.
2. Calculate the escape velocity from the Moon.

### Level 2
3. A comet has perihelion r_p = 1 AU and aphelion r_a = 100 AU. Find e and a.
4. Derive the virial theorem ⟨T⟩ = -½⟨V⟩ for bound orbits.

### Level 3
5. Find the orbit equation for a repulsive inverse-square force.
6. Analyze orbits for V(r) = -k/r + λ/r² (relativistic correction).

---

## 💻 Computational Lab

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def central_force_analysis():
    """Analyze central force problem: orbits and effective potential."""
    
    print("=" * 70)
    print("CENTRAL FORCE PROBLEM: KEPLER ORBITS")
    print("=" * 70)
    
    # Units: G*M = 1
    GM = 1.0
    m = 1.0
    
    def V_eff(r, L):
        """Effective potential for gravity."""
        return -GM*m/r + L**2/(2*m*r**2)
    
    # Plot effective potential for different L
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    r = np.linspace(0.1, 5, 500)
    
    ax = axes[0, 0]
    for L in [0.3, 0.5, 0.7, 1.0]:
        V = V_eff(r, L)
        ax.plot(r, V, lw=2, label=f'L = {L}')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 5)
    ax.set_ylim(-2, 1)
    ax.set_xlabel('r')
    ax.set_ylabel('V_eff(r)')
    ax.set_title('Effective Potential for Different L')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Simulate orbits
    def orbit_eom(state, t, L):
        r, phi, r_dot, phi_dot = state
        r_ddot = L**2/(m*r**3) - GM*m/r**2
        phi_ddot = -2*r_dot*phi_dot/r
        return [r_dot, phi_dot, r_ddot, phi_ddot]
    
    ax = axes[0, 1]
    
    # Different orbit types
    orbits = [
        {'L': 0.5, 'r0': 1.0, 'rdot0': 0, 'label': 'Ellipse (e<1)', 'color': 'blue'},
        {'L': 0.5, 'r0': 0.25, 'rdot0': 0, 'label': 'Circular', 'color': 'green'},
        {'L': 0.5, 'r0': 1.5, 'rdot0': 0.8, 'label': 'Hyperbola', 'color': 'red'},
    ]
    
    for orb in orbits:
        L = orb['L']
        phi_dot0 = L / (m * orb['r0']**2)
        state0 = [orb['r0'], 0, orb['rdot0'], phi_dot0]
        
        t = np.linspace(0, 20, 2000)
        sol = odeint(orbit_eom, state0, t, args=(L,))
        
        r_sol, phi_sol = sol[:, 0], sol[:, 1]
        x = r_sol * np.cos(phi_sol)
        y = r_sol * np.sin(phi_sol)
        
        ax.plot(x, y, color=orb['color'], lw=1.5, label=orb['label'])
    
    ax.scatter([0], [0], c='orange', s=200, marker='*', zorder=5, label='Center')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Different Orbit Types')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # Verify Kepler's second law (equal areas)
    ax = axes[1, 0]
    
    L = 0.6
    r0 = 1.0
    phi_dot0 = L / (m * r0**2)
    state0 = [r0, 0, 0, phi_dot0]
    
    t = np.linspace(0, 15, 1500)
    sol = odeint(orbit_eom, state0, t, args=(L,))
    r_sol, phi_sol = sol[:, 0], sol[:, 1]
    
    # Area swept: dA/dt = (1/2)r²φ̇ = L/(2m)
    dA_dt = 0.5 * r_sol**2 * sol[:, 3]
    
    ax.plot(t, dA_dt, 'b-', lw=2)
    ax.axhline(y=L/(2*m), color='r', linestyle='--', label=f'L/(2m) = {L/(2*m):.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('dA/dt')
    ax.set_title('Kepler\'s 2nd Law: dA/dt = const')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy conservation
    ax = axes[1, 1]
    
    T = 0.5 * m * (sol[:, 2]**2 + (r_sol * sol[:, 3])**2)
    V = -GM * m / r_sol
    E = T + V
    
    ax.plot(t, T, 'b-', lw=1, label='Kinetic T')
    ax.plot(t, V, 'r-', lw=1, label='Potential V')
    ax.plot(t, E, 'k-', lw=2, label=f'Total E = {np.mean(E):.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title(f'Energy Conservation (std = {np.std(E):.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kepler_orbits.png', dpi=150)
    plt.show()

central_force_analysis()
```

---

## 📝 Summary

### Key Results

| Quantity | Formula |
|----------|---------|
| Effective potential | V_eff = V(r) + L²/(2mr²) |
| Orbit equation | r = p/(1 + e cos θ) |
| Semi-latus rectum | p = L²/(GMm²) |
| Circular radius | r_c = L²/(GMm²) |
| Kepler's 3rd law | T² = 4π²a³/(GM) |

### Orbit Classification

| e Value | Orbit | Energy |
|---------|-------|--------|
| e = 0 | Circle | E = V_eff(min) |
| 0 < e < 1 | Ellipse | E < 0 |
| e = 1 | Parabola | E = 0 |
| e > 1 | Hyperbola | E > 0 |

---

## ✅ Daily Checklist

- [ ] Derive effective potential
- [ ] Analyze orbit types from energy
- [ ] Derive orbit equation
- [ ] Prove Kepler's laws
- [ ] Connect to quantum hydrogen
- [ ] Complete computational exercises

---

## 🔮 Preview: Day 150

Tomorrow we study the **Two-Body Problem** and learn how to reduce it to an equivalent one-body problem using reduced mass!
