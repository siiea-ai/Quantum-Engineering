# Day 153: Computational Lab — Lagrangian II Applications

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Part 1: Kepler & Two-Body Simulations |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Part 2: Normal Modes & Rigid Bodies |
| Evening | 6:00 PM - 7:30 PM | 1.5 hours | Part 3: Advanced Visualizations |

**Total Study Time: 8 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Simulate planetary orbits with perturbations
2. Visualize normal modes of complex systems
3. Compute and diagonalize inertia tensors
4. Simulate rigid body rotation (Euler's equations)
5. Create publication-quality visualizations

---

## 💻 Part 1: Kepler & Two-Body (3.5 hours)

```python
"""
Lagrangian Mechanics II: Computational Lab
==========================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D

print("=" * 70)
print("LAGRANGIAN MECHANICS II - COMPUTATIONAL LAB")
print("=" * 70)

# ============================================
# PART 1: KEPLER PROBLEM WITH PERTURBATIONS
# ============================================

def kepler_with_perturbation():
    """Simulate Kepler problem with relativistic correction."""
    
    print("\n" + "=" * 50)
    print("1. KEPLER PROBLEM WITH RELATIVISTIC CORRECTION")
    print("=" * 50)
    
    # Parameters (scaled units)
    GM = 1.0
    c = 100.0  # Speed of light (scaled)
    
    def kepler_eom(state, t, include_correction=False):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        
        # Newtonian force
        ax = -GM * x / r**3
        ay = -GM * y / r**3
        
        if include_correction:
            # Relativistic correction (perihelion precession)
            # F_corr = 3GM*L^2/(c^2*r^4) * r_hat
            L = x*vy - y*vx  # Angular momentum per unit mass
            correction = 3 * GM * L**2 / (c**2 * r**4)
            ax += correction * x / r
            ay += correction * y / r
        
        return [vx, vy, ax, ay]
    
    # Initial conditions for elliptical orbit
    r0 = 1.0
    e = 0.5
    v0 = np.sqrt(GM/r0 * (1+e))  # For given eccentricity
    
    state0 = [r0, 0, 0, v0]
    t = np.linspace(0, 100, 5000)
    
    # Simulate with and without correction
    sol_newton = odeint(kepler_eom, state0, t, args=(False,))
    sol_rel = odeint(kepler_eom, state0, t, args=(True,))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Orbits
    ax = axes[0]
    ax.plot(sol_newton[:, 0], sol_newton[:, 1], 'b-', alpha=0.5, lw=0.5, label='Newtonian')
    ax.plot(sol_rel[:, 0], sol_rel[:, 1], 'r-', alpha=0.5, lw=0.5, label='With GR correction')
    ax.scatter([0], [0], c='orange', s=200, marker='*', zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Orbit Comparison\n(GR causes perihelion precession)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Perihelion angle vs time
    ax = axes[1]
    
    # Find perihelion points (local minima of r)
    r_newton = np.sqrt(sol_newton[:, 0]**2 + sol_newton[:, 1]**2)
    r_rel = np.sqrt(sol_rel[:, 0]**2 + sol_rel[:, 1]**2)
    
    # Simple perihelion detection
    peri_newton = []
    peri_rel = []
    
    for i in range(1, len(r_newton)-1):
        if r_newton[i] < r_newton[i-1] and r_newton[i] < r_newton[i+1]:
            angle = np.arctan2(sol_newton[i, 1], sol_newton[i, 0])
            peri_newton.append((t[i], angle))
        if r_rel[i] < r_rel[i-1] and r_rel[i] < r_rel[i+1]:
            angle = np.arctan2(sol_rel[i, 1], sol_rel[i, 0])
            peri_rel.append((t[i], angle))
    
    if peri_newton and peri_rel:
        t_n, theta_n = zip(*peri_newton)
        t_r, theta_r = zip(*peri_rel)
        
        # Unwrap angles
        theta_n = np.unwrap(theta_n)
        theta_r = np.unwrap(theta_r)
        
        ax.plot(t_n, np.degrees(theta_n), 'b.-', label='Newtonian')
        ax.plot(t_r, np.degrees(theta_r), 'r.-', label='Relativistic')
        ax.set_xlabel('Time')
        ax.set_ylabel('Perihelion angle (degrees)')
        ax.set_title('Perihelion Precession')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kepler_precession.png', dpi=150)
    plt.show()
    
    print("\nRelativistic correction causes perihelion to precess!")
    print("This effect for Mercury was one of the first tests of GR.")

kepler_with_perturbation()


def three_body_restricted():
    """Simulate restricted three-body problem (circular)."""
    
    print("\n" + "=" * 50)
    print("2. RESTRICTED THREE-BODY PROBLEM")
    print("=" * 50)
    
    # Sun-Jupiter-asteroid system (restricted problem)
    mu = 0.001  # Mass ratio m2/(m1+m2), Jupiter/Sun ~ 0.001
    
    def cr3bp_eom(t, state):
        """Circular Restricted 3-Body Problem in rotating frame."""
        x, y, vx, vy = state
        
        # Distances to primaries
        r1 = np.sqrt((x + mu)**2 + y**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2)
        
        # Equations of motion in rotating frame
        ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
        
        return [vx, vy, ax, ay]
    
    # Find Lagrange points approximately
    # L1: between Sun and Jupiter
    x_L1 = 1 - mu - (mu/3)**(1/3)
    
    # Initial conditions near L1
    state0 = [x_L1 + 0.01, 0.01, 0, 0.1]
    
    t_span = (0, 50)
    t_eval = np.linspace(*t_span, 5000)
    
    sol = solve_ivp(cr3bp_eom, t_span, state0, t_eval=t_eval, method='RK45')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot orbit
    ax.plot(sol.y[0], sol.y[1], 'b-', lw=0.5, alpha=0.7)
    
    # Plot primaries
    ax.scatter([-mu], [0], c='orange', s=200, marker='*', label='Sun', zorder=5)
    ax.scatter([1-mu], [0], c='brown', s=100, marker='o', label='Jupiter', zorder=5)
    
    # Mark Lagrange points
    ax.scatter([x_L1], [0], c='red', s=50, marker='x', label='L1', zorder=5)
    
    ax.set_xlabel('x (rotating frame)')
    ax.set_ylabel('y (rotating frame)')
    ax.set_title('Restricted 3-Body Problem\n(Particle near L1 point)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('three_body.png', dpi=150)
    plt.show()

three_body_restricted()
```

---

## 💻 Part 2: Normal Modes & Rigid Bodies (3 hours)

```python
# ============================================
# PART 2: NORMAL MODES OF MOLECULAR SYSTEMS
# ============================================

def molecular_vibrations():
    """Compute normal modes of simple molecules."""
    
    print("\n" + "=" * 50)
    print("3. MOLECULAR VIBRATIONS: CO2")
    print("=" * 50)
    
    # CO2: O=C=O (linear triatomic)
    # Masses (atomic units)
    m_O = 16.0
    m_C = 12.0
    
    # Spring constant (same for both C=O bonds)
    k = 1.0
    
    # Mass matrix (for displacements along molecular axis)
    M = np.diag([m_O, m_C, m_O])
    
    # Stiffness matrix
    # Potential: V = (1/2)k[(x2-x1)^2 + (x3-x2)^2]
    K = k * np.array([
        [1, -1, 0],
        [-1, 2, -1],
        [0, -1, 1]
    ])
    
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(K, M)
    frequencies = np.sqrt(np.abs(eigenvalues))
    
    print("\nNormal mode frequencies (arbitrary units):")
    for i, (freq, mode) in enumerate(zip(frequencies, eigenvectors.T)):
        print(f"  Mode {i+1}: ω = {freq:.4f}")
        print(f"         Shape: {mode}")
    
    # Identify modes
    print("\nMode interpretation:")
    print("  Mode 1 (ω=0): Translation (CM motion)")
    print("  Mode 2: Symmetric stretch (C stationary)")
    print("  Mode 3: Antisymmetric stretch (C moves opposite to O's)")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x_eq = np.array([-1, 0, 1])  # Equilibrium positions
    
    for i, ax in enumerate(axes):
        mode = eigenvectors[:, i]
        mode = mode / np.max(np.abs(mode)) * 0.3
        
        # Draw molecule at equilibrium
        ax.scatter(x_eq, [0, 0, 0], c=['red', 'black', 'red'], s=200, zorder=5)
        ax.plot(x_eq, [0, 0, 0], 'k-', lw=2)
        
        # Draw displacement arrows
        for j, (x, d) in enumerate(zip(x_eq, mode)):
            if abs(d) > 0.01:
                ax.annotate('', xy=(x + d, 0.3), xytext=(x, 0.3),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 1)
        ax.set_title(f'Mode {i+1}: ω = {frequencies[i]:.3f}')
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.suptitle('CO2 Normal Modes (Longitudinal)', fontsize=14)
    plt.tight_layout()
    plt.savefig('co2_modes.png', dpi=150)
    plt.show()

molecular_vibrations()


def rigid_body_rotation():
    """Simulate free rigid body rotation (Euler's equations)."""
    
    print("\n" + "=" * 50)
    print("4. RIGID BODY ROTATION: EULER'S EQUATIONS")
    print("=" * 50)
    
    # Principal moments of inertia (asymmetric top)
    I1, I2, I3 = 1.0, 2.0, 3.0
    
    def euler_equations(state, t):
        """Euler's equations for torque-free rotation."""
        w1, w2, w3 = state
        
        dw1 = (I2 - I3) / I1 * w2 * w3
        dw2 = (I3 - I1) / I2 * w3 * w1
        dw3 = (I1 - I2) / I3 * w1 * w2
        
        return [dw1, dw2, dw3]
    
    # Initial conditions
    state0 = [1.0, 0.1, 0.1]  # Nearly aligned with principal axis 1
    
    t = np.linspace(0, 50, 2000)
    sol = odeint(euler_equations, state0, t)
    
    w1, w2, w3 = sol[:, 0], sol[:, 1], sol[:, 2]
    
    # Conserved quantities
    E = 0.5 * (I1*w1**2 + I2*w2**2 + I3*w3**2)  # Energy
    L2 = (I1*w1)**2 + (I2*w2)**2 + (I3*w3)**2   # L² 
    
    fig = plt.figure(figsize=(15, 10))
    
    # Angular velocity components
    ax1 = fig.add_subplot(221)
    ax1.plot(t, w1, 'r-', lw=1.5, label='ω₁')
    ax1.plot(t, w2, 'g-', lw=1.5, label='ω₂')
    ax1.plot(t, w3, 'b-', lw=1.5, label='ω₃')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Angular velocity')
    ax1.set_title('Euler Equations: ω(t)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Poinsot construction (ω trajectory in body frame)
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(w1, w2, w3, 'b-', lw=1)
    ax2.scatter([w1[0]], [w2[0]], [w3[0]], c='red', s=100, label='Start')
    
    # Draw energy ellipsoid
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    E0 = E[0]
    
    x_ell = np.sqrt(2*E0/I1) * np.outer(np.cos(u), np.sin(v))
    y_ell = np.sqrt(2*E0/I2) * np.outer(np.sin(u), np.sin(v))
    z_ell = np.sqrt(2*E0/I3) * np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x_ell, y_ell, z_ell, alpha=0.2, color='green')
    
    ax2.set_xlabel('ω₁')
    ax2.set_ylabel('ω₂')
    ax2.set_zlabel('ω₃')
    ax2.set_title('Poinsot Construction\n(ω traces curve on energy ellipsoid)')
    ax2.legend()
    
    # Conservation check
    ax3 = fig.add_subplot(223)
    ax3.plot(t, E/E[0], 'r-', lw=2, label='E/E₀')
    ax3.plot(t, L2/L2[0], 'b-', lw=2, label='L²/L²₀')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Normalized value')
    ax3.set_title(f'Conservation (E std: {np.std(E):.2e}, L² std: {np.std(L2):.2e})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.99, 1.01)
    
    # Phase portrait (w1 vs w2)
    ax4 = fig.add_subplot(224)
    ax4.plot(w1, w2, 'b-', lw=1)
    ax4.scatter([w1[0]], [w2[0]], c='red', s=100, zorder=5)
    ax4.set_xlabel('ω₁')
    ax4.set_ylabel('ω₂')
    ax4.set_title('Phase Portrait')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('euler_equations.png', dpi=150)
    plt.show()
    
    print("\nFor asymmetric top (I1 < I2 < I3):")
    print("- Rotation about axes 1 and 3 is stable")
    print("- Rotation about axis 2 (intermediate) is unstable!")
    print("- This is the 'tennis racket theorem'")

rigid_body_rotation()


# ============================================
# PART 3: ADVANCED VISUALIZATIONS
# ============================================

def tennis_racket_theorem():
    """Demonstrate the tennis racket (intermediate axis) theorem."""
    
    print("\n" + "=" * 50)
    print("5. TENNIS RACKET THEOREM")
    print("=" * 50)
    
    I1, I2, I3 = 1.0, 2.0, 3.0
    
    def euler_equations(state, t):
        w1, w2, w3 = state
        dw1 = (I2 - I3) / I1 * w2 * w3
        dw2 = (I3 - I1) / I2 * w3 * w1
        dw3 = (I1 - I2) / I3 * w1 * w2
        return [dw1, dw2, dw3]
    
    t = np.linspace(0, 30, 1500)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Rotation near each principal axis
    initial_conditions = [
        ([1.0, 0.05, 0.05], 'Axis 1 (smallest I): STABLE'),
        ([0.05, 1.0, 0.05], 'Axis 2 (intermediate I): UNSTABLE'),
        ([0.05, 0.05, 1.0], 'Axis 3 (largest I): STABLE'),
    ]
    
    for ax, (ic, title) in zip(axes, initial_conditions):
        sol = odeint(euler_equations, ic, t)
        
        ax.plot(t, sol[:, 0], 'r-', lw=1.5, label='ω₁')
        ax.plot(t, sol[:, 1], 'g-', lw=1.5, label='ω₂')
        ax.plot(t, sol[:, 2], 'b-', lw=1.5, label='ω₃')
        ax.set_xlabel('Time')
        ax.set_ylabel('ω')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('tennis_racket.png', dpi=150)
    plt.show()
    
    print("\nThe tennis racket theorem:")
    print("- Rotation about the intermediate principal axis is unstable")
    print("- Small perturbations grow, causing flipping motion")
    print("- Try it with a tennis racket or book!")

tennis_racket_theorem()

print("\n" + "=" * 70)
print("COMPUTATIONAL LAB COMPLETE!")
print("=" * 70)
```

---

## 📝 Summary

### Topics Covered

| Topic | Key Concepts |
|-------|--------------|
| Kepler with perturbations | Perihelion precession, GR correction |
| Three-body problem | Lagrange points, chaos |
| Molecular vibrations | Normal modes, IR spectroscopy |
| Rigid body rotation | Euler's equations, stability |
| Tennis racket theorem | Intermediate axis instability |

---

## ✅ Daily Checklist

- [ ] Simulate Kepler orbits with corrections
- [ ] Explore three-body dynamics
- [ ] Compute molecular normal modes
- [ ] Solve Euler's equations
- [ ] Demonstrate tennis racket theorem
- [ ] Create publication-quality figures

---

## 🔮 Preview: Day 154

Tomorrow we review Week 22 and consolidate our mastery of advanced Lagrangian mechanics!
