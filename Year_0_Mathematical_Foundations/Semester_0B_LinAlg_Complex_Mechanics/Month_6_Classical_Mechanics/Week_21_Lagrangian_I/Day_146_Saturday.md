# Day 146: Computational Lab — Lagrangian Mechanics Simulations

## 📅 Schedule Overview
| Block | Time | Duration | Activity |
|-------|------|----------|----------|
| Morning | 9:00 AM - 12:30 PM | 3.5 hours | Part 1: Basic Systems |
| Afternoon | 2:00 PM - 5:00 PM | 3 hours | Part 2: Coupled Systems |
| Evening | 6:00 PM - 7:30 PM | 1.5 hours | Part 3: Advanced Topics |

**Total Study Time: 8 hours**

---

## 🎯 Learning Objectives

By the end of today, you should be able to:

1. Implement Lagrangian mechanics numerically
2. Simulate pendulums, oscillators, and coupled systems
3. Verify conservation laws computationally
4. Animate mechanical systems
5. Explore chaotic dynamics (double pendulum)

---

## 💻 Part 1: Basic Systems (3.5 hours)

```python
"""
Lagrangian Mechanics Simulation Toolkit
=======================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class LagrangianSystem:
    """Base class for Lagrangian mechanical systems."""
    
    def __init__(self, params):
        self.params = params
    
    def lagrangian(self, q, q_dot, t):
        """Compute Lagrangian L = T - V."""
        raise NotImplementedError
    
    def equations_of_motion(self, state, t):
        """Return dq/dt, dq_dot/dt."""
        raise NotImplementedError
    
    def energy(self, state):
        """Compute total energy."""
        raise NotImplementedError
    
    def simulate(self, initial_state, t_span, n_points=1000):
        """Simulate the system."""
        t = np.linspace(t_span[0], t_span[1], n_points)
        solution = odeint(self.equations_of_motion, initial_state, t)
        return t, solution


class SimplePendulum(LagrangianSystem):
    """Simple pendulum: L = (1/2)mL²θ̇² + mgL cos θ"""
    
    def __init__(self, m=1.0, L=1.0, g=9.81):
        super().__init__({'m': m, 'L': L, 'g': g})
    
    def equations_of_motion(self, state, t):
        theta, theta_dot = state
        m, L, g = self.params['m'], self.params['L'], self.params['g']
        theta_ddot = -(g/L) * np.sin(theta)
        return [theta_dot, theta_ddot]
    
    def energy(self, state):
        theta, theta_dot = state[:, 0], state[:, 1]
        m, L, g = self.params['m'], self.params['L'], self.params['g']
        T = 0.5 * m * L**2 * theta_dot**2
        V = -m * g * L * np.cos(theta)
        return T + V
    
    def get_xy(self, theta):
        L = self.params['L']
        return L * np.sin(theta), -L * np.cos(theta)


class DoublePendulum(LagrangianSystem):
    """Double pendulum - exhibits chaos!"""
    
    def __init__(self, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
        super().__init__({'m1': m1, 'm2': m2, 'L1': L1, 'L2': L2, 'g': g})
    
    def equations_of_motion(self, state, t):
        theta1, theta2, omega1, omega2 = state
        m1, m2 = self.params['m1'], self.params['m2']
        L1, L2, g = self.params['L1'], self.params['L2'], self.params['g']
        
        delta = theta2 - theta1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
        den2 = (L2/L1) * den1
        
        omega1_dot = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                      m2 * g * np.sin(theta2) * np.cos(delta) +
                      m2 * L2 * omega2**2 * np.sin(delta) -
                      (m1 + m2) * g * np.sin(theta1)) / den1
        
        omega2_dot = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                      (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                      (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
                      (m1 + m2) * g * np.sin(theta2)) / den2
        
        return [omega1, omega2, omega1_dot, omega2_dot]
    
    def energy(self, state):
        theta1, theta2 = state[:, 0], state[:, 1]
        omega1, omega2 = state[:, 2], state[:, 3]
        m1, m2 = self.params['m1'], self.params['m2']
        L1, L2, g = self.params['L1'], self.params['L2'], self.params['g']
        
        T = (0.5 * (m1 + m2) * L1**2 * omega1**2 +
             0.5 * m2 * L2**2 * omega2**2 +
             m2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))
        V = (-(m1 + m2) * g * L1 * np.cos(theta1) -
             m2 * g * L2 * np.cos(theta2))
        return T + V
    
    def get_xy(self, theta1, theta2):
        L1, L2 = self.params['L1'], self.params['L2']
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)
        return x1, y1, x2, y2


class HarmonicOscillator(LagrangianSystem):
    """Simple harmonic oscillator: L = (1/2)mẋ² - (1/2)kx²"""
    
    def __init__(self, m=1.0, k=1.0):
        super().__init__({'m': m, 'k': k})
        self.omega = np.sqrt(k/m)
    
    def equations_of_motion(self, state, t):
        x, v = state
        m, k = self.params['m'], self.params['k']
        return [v, -k*x/m]
    
    def energy(self, state):
        x, v = state[:, 0], state[:, 1]
        m, k = self.params['m'], self.params['k']
        return 0.5 * m * v**2 + 0.5 * k * x**2


# Demonstrations
print("=" * 70)
print("LAGRANGIAN MECHANICS SIMULATIONS")
print("=" * 70)

# 1. Simple Pendulum
print("\n1. SIMPLE PENDULUM")
pendulum = SimplePendulum(m=1, L=1, g=10)
t, sol = pendulum.simulate([np.pi/3, 0], [0, 10])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(t, sol[:, 0], 'b-', lw=2)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('θ')
axes[0, 0].set_title('Simple Pendulum: Angle')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(sol[:, 0], sol[:, 1], 'b-', lw=1)
axes[0, 1].set_xlabel('θ')
axes[0, 1].set_ylabel('θ̇')
axes[0, 1].set_title('Phase Space')
axes[0, 1].grid(True, alpha=0.3)

E = pendulum.energy(sol)
axes[1, 0].plot(t, E, 'r-', lw=2)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Energy')
axes[1, 0].set_title(f'Energy Conservation (std: {np.std(E):.2e})')
axes[1, 0].grid(True, alpha=0.3)

# Trajectory
x, y = pendulum.get_xy(sol[:, 0])
axes[1, 1].plot(x, y, 'b-', lw=1, alpha=0.5)
axes[1, 1].scatter([0], [0], c='black', s=100, zorder=5)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title('Pendulum Trajectory')
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_pendulum_sim.png', dpi=150)
plt.show()

# 2. Double Pendulum (Chaos!)
print("\n2. DOUBLE PENDULUM (CHAOS)")
dp = DoublePendulum(m1=1, m2=1, L1=1, L2=1, g=10)

# Two nearly identical initial conditions
t, sol1 = dp.simulate([np.pi/2, np.pi/2, 0, 0], [0, 20], n_points=2000)
_, sol2 = dp.simulate([np.pi/2 + 0.001, np.pi/2, 0, 0], [0, 20], n_points=2000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trajectories
x1_1, y1_1, x2_1, y2_1 = dp.get_xy(sol1[:, 0], sol1[:, 1])
x1_2, y1_2, x2_2, y2_2 = dp.get_xy(sol2[:, 0], sol2[:, 1])

axes[0, 0].plot(x2_1, y2_1, 'b-', lw=0.5, alpha=0.7, label='IC 1')
axes[0, 0].plot(x2_2, y2_2, 'r-', lw=0.5, alpha=0.7, label='IC 2 (Δθ=0.001)')
axes[0, 0].scatter([0], [0], c='black', s=100, zorder=5)
axes[0, 0].set_xlabel('x₂')
axes[0, 0].set_ylabel('y₂')
axes[0, 0].set_title('Double Pendulum: Chaotic Trajectories')
axes[0, 0].legend()
axes[0, 0].set_aspect('equal')

# Divergence of trajectories
diff = np.sqrt((sol1[:, 0] - sol2[:, 0])**2 + (sol1[:, 1] - sol2[:, 1])**2)
axes[0, 1].semilogy(t, diff, 'g-', lw=2)
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('|Δθ|')
axes[0, 1].set_title('Exponential Divergence (Chaos!)')
axes[0, 1].grid(True, alpha=0.3)

# Energy conservation
E1 = dp.energy(sol1)
axes[1, 0].plot(t, E1, 'r-', lw=1)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Energy')
axes[1, 0].set_title(f'Energy Conservation (std: {np.std(E1):.2e})')
axes[1, 0].grid(True, alpha=0.3)

# Phase space (θ₁, θ₂)
axes[1, 1].plot(sol1[:, 0], sol1[:, 1], 'b-', lw=0.3, alpha=0.5)
axes[1, 1].set_xlabel('θ₁')
axes[1, 1].set_ylabel('θ₂')
axes[1, 1].set_title('Phase Space (Configuration)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_pendulum_chaos.png', dpi=150)
plt.show()

print("\nKey observation: Tiny difference in initial conditions")
print("leads to completely different trajectories!")
print("This is the hallmark of CHAOS.")
```

---

## 💻 Part 2: Coupled Systems (3 hours)

```python
class CoupledOscillators(LagrangianSystem):
    """Two masses coupled by springs."""
    
    def __init__(self, m1=1.0, m2=1.0, k1=1.0, k2=1.0, k_c=0.5):
        super().__init__({'m1': m1, 'm2': m2, 'k1': k1, 'k2': k2, 'k_c': k_c})
    
    def equations_of_motion(self, state, t):
        x1, x2, v1, v2 = state
        m1, m2 = self.params['m1'], self.params['m2']
        k1, k2, k_c = self.params['k1'], self.params['k2'], self.params['k_c']
        
        a1 = (-k1*x1 - k_c*(x1 - x2)) / m1
        a2 = (-k2*x2 - k_c*(x2 - x1)) / m2
        return [v1, v2, a1, a2]
    
    def normal_modes(self):
        """Find normal mode frequencies."""
        m1, m2 = self.params['m1'], self.params['m2']
        k1, k2, k_c = self.params['k1'], self.params['k2'], self.params['k_c']
        
        # For equal masses and springs
        if m1 == m2 and k1 == k2:
            m, k = m1, k1
            omega1 = np.sqrt(k/m)  # In-phase mode
            omega2 = np.sqrt((k + 2*k_c)/m)  # Out-of-phase mode
            return omega1, omega2
        return None


# Coupled oscillators demonstration
print("\n" + "=" * 70)
print("COUPLED OSCILLATORS")
print("=" * 70)

coupled = CoupledOscillators(m1=1, m2=1, k1=1, k2=1, k_c=0.3)
omega1, omega2 = coupled.normal_modes()
print(f"\nNormal mode frequencies: ω₁ = {omega1:.4f}, ω₂ = {omega2:.4f}")

# Simulate with different initial conditions
t, sol_inphase = coupled.simulate([1, 1, 0, 0], [0, 30], n_points=1000)
_, sol_outphase = coupled.simulate([1, -1, 0, 0], [0, 30], n_points=1000)
_, sol_general = coupled.simulate([1, 0, 0, 0], [0, 30], n_points=1000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# In-phase mode
axes[0, 0].plot(t, sol_inphase[:, 0], 'b-', lw=2, label='x₁')
axes[0, 0].plot(t, sol_inphase[:, 1], 'r--', lw=2, label='x₂')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Position')
axes[0, 0].set_title(f'In-Phase Mode (ω = {omega1:.3f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Out-of-phase mode
axes[0, 1].plot(t, sol_outphase[:, 0], 'b-', lw=2, label='x₁')
axes[0, 1].plot(t, sol_outphase[:, 1], 'r--', lw=2, label='x₂')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Position')
axes[0, 1].set_title(f'Out-of-Phase Mode (ω = {omega2:.3f})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# General motion (beats)
axes[1, 0].plot(t, sol_general[:, 0], 'b-', lw=1, label='x₁')
axes[1, 0].plot(t, sol_general[:, 1], 'r-', lw=1, label='x₂')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Position')
axes[1, 0].set_title('General Motion: Energy Transfer (Beats)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Energy in each oscillator
m, k, k_c = 1, 1, 0.3
E1 = 0.5 * sol_general[:, 2]**2 + 0.5 * k * sol_general[:, 0]**2
E2 = 0.5 * sol_general[:, 3]**2 + 0.5 * k * sol_general[:, 1]**2

axes[1, 1].plot(t, E1, 'b-', lw=1, label='E₁')
axes[1, 1].plot(t, E2, 'r-', lw=1, label='E₂')
axes[1, 1].plot(t, E1 + E2, 'k--', lw=2, label='E_total')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Energy')
axes[1, 1].set_title('Energy Exchange Between Oscillators')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coupled_oscillators.png', dpi=150)
plt.show()
```

---

## 💻 Part 3: Animation and Advanced Topics (1.5 hours)

```python
def create_pendulum_animation():
    """Create animation of double pendulum."""
    
    dp = DoublePendulum(m1=1, m2=1, L1=1, L2=1, g=10)
    t, sol = dp.simulate([np.pi/2, np.pi/2, 0, 0], [0, 15], n_points=750)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Double Pendulum Animation')
    
    line, = ax.plot([], [], 'o-', lw=2, markersize=10)
    trace, = ax.plot([], [], 'b-', lw=0.5, alpha=0.5)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    x1, y1, x2, y2 = dp.get_xy(sol[:, 0], sol[:, 1])
    
    trace_x, trace_y = [], []
    
    def init():
        line.set_data([], [])
        trace.set_data([], [])
        time_text.set_text('')
        return line, trace, time_text
    
    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        
        trace_x.append(x2[i])
        trace_y.append(y2[i])
        
        line.set_data(thisx, thisy)
        trace.set_data(trace_x, trace_y)
        time_text.set_text(f't = {t[i]:.2f}')
        return line, trace, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(t), interval=20, blit=True)
    
    # Save animation
    anim.save('double_pendulum.gif', writer='pillow', fps=30)
    plt.close()
    print("Animation saved as 'double_pendulum.gif'")

# Uncomment to create animation (takes time)
# create_pendulum_animation()

print("\n" + "=" * 70)
print("LAB COMPLETE!")
print("=" * 70)
print("\nSystems simulated:")
print("  1. Simple pendulum (energy conservation verified)")
print("  2. Double pendulum (chaos demonstrated)")
print("  3. Coupled oscillators (normal modes, beats)")
```

---

## 📝 Summary

### Systems Implemented

| System | Key Features |
|--------|--------------|
| Simple Pendulum | Nonlinear, energy conserved |
| Double Pendulum | Chaos, sensitive dependence |
| Coupled Oscillators | Normal modes, energy transfer |

### Key Observations

1. **Energy conservation** verified numerically for all systems
2. **Chaos** in double pendulum: tiny Δθ → completely different trajectories
3. **Normal modes** in coupled systems: simple motion patterns
4. **Beats** when modes mix: energy oscillates between oscillators

---

## ✅ Daily Checklist

- [ ] Implement Lagrangian systems numerically
- [ ] Simulate simple pendulum
- [ ] Observe chaos in double pendulum
- [ ] Study coupled oscillator normal modes
- [ ] Verify conservation laws
- [ ] Create animations (optional)

---

## 🔮 Preview: Day 147

Tomorrow we review Week 21 and consolidate our understanding of Lagrangian mechanics!
