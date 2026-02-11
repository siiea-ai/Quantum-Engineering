# Day 316: Newtonian Mechanics Review

## Overview

**Month 12, Week 46, Day 1 — Monday**

Today we review Newtonian mechanics: forces, Newton's laws, energy, momentum, and angular momentum. This is the foundation upon which the more powerful Lagrangian and Hamiltonian formulations are built.

## Learning Objectives

1. Review Newton's three laws
2. Master conservation laws
3. Solve classical mechanics problems
4. Connect to advanced formulations

---

## 1. Newton's Laws

### First Law (Inertia)

A body remains at rest or in uniform motion unless acted upon by a force.

### Second Law (F = ma)

$$\mathbf{F} = m\mathbf{a} = m\frac{d\mathbf{v}}{dt} = \frac{d\mathbf{p}}{dt}$$

### Third Law (Action-Reaction)

$$\mathbf{F}_{12} = -\mathbf{F}_{21}$$

---

## 2. Energy Conservation

### Kinetic Energy

$$T = \frac{1}{2}mv^2 = \frac{p^2}{2m}$$

### Potential Energy

$$F = -\nabla V$$

### Work-Energy Theorem

$$W = \Delta T$$

### Conservation

For conservative forces: $E = T + V = \text{constant}$

---

## 3. Momentum Conservation

### Linear Momentum

$$\mathbf{p} = m\mathbf{v}$$

$$\frac{d\mathbf{p}}{dt} = \mathbf{F}_{ext}$$

If $\mathbf{F}_{ext} = 0$: $\mathbf{p}$ conserved.

### Angular Momentum

$$\mathbf{L} = \mathbf{r} \times \mathbf{p}$$

$$\frac{d\mathbf{L}}{dt} = \boldsymbol{\tau}$$

If $\boldsymbol{\tau} = 0$: $\mathbf{L}$ conserved.

---

## 4. Central Force Problem

### Effective Potential

$$V_{eff}(r) = V(r) + \frac{L^2}{2mr^2}$$

### Kepler Problem

$$V(r) = -\frac{k}{r}$$

Orbits: ellipses, parabolas, hyperbolas

---

## 5. Computational Lab

```python
"""
Day 316: Newtonian Mechanics
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def projectile():
    """Solve projectile motion."""
    g = 9.8
    v0, theta = 20, 45

    vx0 = v0 * np.cos(np.radians(theta))
    vy0 = v0 * np.sin(np.radians(theta))

    t = np.linspace(0, 2*vy0/g, 100)
    x = vx0 * t
    y = vy0 * t - 0.5 * g * t**2

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Projectile Motion')
    plt.grid(True)
    plt.savefig('projectile.png', dpi=150)
    plt.close()
    print("Saved: projectile.png")

def kepler_orbit():
    """Solve Kepler problem."""
    def equations(state, t, k, m):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        ax = -k * x / (m * r**3)
        ay = -k * y / (m * r**3)
        return [vx, vy, ax, ay]

    # Initial conditions for elliptical orbit
    k, m = 1, 1
    r0, v0 = 1, 0.8
    state0 = [r0, 0, 0, v0]

    t = np.linspace(0, 20, 1000)
    solution = odeint(equations, state0, t, args=(k, m))

    plt.figure(figsize=(8, 8))
    plt.plot(solution[:, 0], solution[:, 1])
    plt.plot(0, 0, 'yo', markersize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Kepler Orbit')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('kepler.png', dpi=150)
    plt.close()
    print("Saved: kepler.png")

if __name__ == "__main__":
    projectile()
    kepler_orbit()
```

---

## Summary

### Newton's Framework

$$\boxed{\mathbf{F} = m\mathbf{a}}$$

### Conservation Laws

| Quantity | Condition | Formula |
|----------|-----------|---------|
| Energy | Conservative forces | $E = T + V$ |
| Momentum | No external force | $\mathbf{p} = m\mathbf{v}$ |
| Angular momentum | No torque | $\mathbf{L} = \mathbf{r} \times \mathbf{p}$ |

---

## Preview: Day 317

Tomorrow: **Lagrangian Mechanics Review** — the principle of least action.
