# Day 317: Lagrangian Mechanics Review

## Overview

**Month 12, Week 46, Day 2 — Tuesday**

Today we review Lagrangian mechanics: the action principle, Euler-Lagrange equations, constraints, and the power of generalized coordinates. This formulation is the bridge to both Hamiltonian mechanics and quantum mechanics.

## Learning Objectives

1. Master the principle of least action
2. Derive equations of motion from Lagrangians
3. Handle constraints with Lagrange multipliers
4. Connect symmetries to conservation laws

---

## 1. The Lagrangian

### Definition

$$L(q_i, \dot{q}_i, t) = T - V$$

### The Action

$$S = \int_{t_1}^{t_2} L \, dt$$

### Principle of Least Action

The path taken extremizes the action: $\delta S = 0$.

---

## 2. Euler-Lagrange Equations

$$\boxed{\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0}$$

### Derivation

From $\delta S = 0$ with fixed endpoints.

### Generalized Momentum

$$p_i = \frac{\partial L}{\partial \dot{q}_i}$$

---

## 3. Constraints and Lagrange Multipliers

### Holonomic Constraints

$$f_k(q_1, ..., q_n, t) = 0$$

### Modified Equations

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = \sum_k \lambda_k \frac{\partial f_k}{\partial q_i}$$

---

## 4. Symmetry and Conservation (Noether's Theorem)

### Cyclic Coordinates

If $\partial L/\partial q_i = 0$, then $p_i$ is conserved.

### Noether's Theorem

Every continuous symmetry implies a conservation law:
- Time translation → Energy
- Space translation → Momentum
- Rotation → Angular momentum

---

## 5. Examples

### Simple Harmonic Oscillator

$$L = \frac{1}{2}m\dot{x}^2 - \frac{1}{2}kx^2$$

Euler-Lagrange: $m\ddot{x} + kx = 0$

### Pendulum

$$L = \frac{1}{2}m\ell^2\dot{\theta}^2 + mg\ell\cos\theta$$

### Central Force

$$L = \frac{1}{2}m(\dot{r}^2 + r^2\dot{\theta}^2) - V(r)$$

$\theta$ cyclic → $L_z = mr^2\dot{\theta}$ conserved.

---

## 6. Computational Lab

```python
"""
Day 317: Lagrangian Mechanics
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def double_pendulum():
    """Simulate double pendulum using Lagrangian mechanics."""
    m1, m2 = 1, 1
    l1, l2 = 1, 1
    g = 9.8

    def equations(y, t):
        theta1, omega1, theta2, omega2 = y

        delta = theta2 - theta1
        denom1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
        denom2 = (l2 / l1) * denom1

        domega1 = (m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                   m2 * g * np.sin(theta2) * np.cos(delta) +
                   m2 * l2 * omega2**2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta1)) / denom1

        domega2 = (-m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                   (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                   (m1 + m2) * l1 * omega1**2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta2)) / denom2

        return [omega1, domega1, omega2, domega2]

    y0 = [np.pi/2, 0, np.pi/2, 0]
    t = np.linspace(0, 20, 2000)
    sol = odeint(equations, y0, t)

    # Convert to Cartesian
    x1 = l1 * np.sin(sol[:, 0])
    y1 = -l1 * np.cos(sol[:, 0])
    x2 = x1 + l2 * np.sin(sol[:, 2])
    y2 = y1 - l2 * np.cos(sol[:, 2])

    plt.figure(figsize=(8, 8))
    plt.plot(x2, y2, 'b-', alpha=0.5, linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Double Pendulum Trajectory')
    plt.axis('equal')
    plt.savefig('double_pendulum.png', dpi=150)
    plt.close()
    print("Saved: double_pendulum.png")

if __name__ == "__main__":
    double_pendulum()
```

---

## Summary

### The Lagrangian Framework

$$\boxed{S = \int L \, dt, \quad \delta S = 0}$$

$$\boxed{\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} = \frac{\partial L}{\partial q}}$$

---

## Preview: Day 318

Tomorrow: **Hamiltonian Mechanics Review** — phase space and canonical formulation.
