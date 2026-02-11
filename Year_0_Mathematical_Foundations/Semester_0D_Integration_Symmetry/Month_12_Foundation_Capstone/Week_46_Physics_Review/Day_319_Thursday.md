# Day 319: Symmetry and Conservation

## Overview

**Month 12, Week 46, Day 4 — Thursday**

Today we explore the deep connection between symmetry and conservation laws through Noether's theorem. This is the heart of modern physics.

## Learning Objectives

1. State and prove Noether's theorem
2. Apply to fundamental symmetries
3. Connect to quantum mechanics
4. Understand gauge symmetries

---

## 1. Noether's Theorem

### Statement

Every continuous symmetry of the action implies a conserved quantity.

### Mathematical Form

If $L$ is invariant under $q \to q + \epsilon\delta q$:
$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\delta q\right) = 0$$

---

## 2. Fundamental Symmetries

| Symmetry | Transformation | Conservation |
|----------|---------------|--------------|
| Time translation | $t \to t + a$ | Energy $H$ |
| Space translation | $\mathbf{r} \to \mathbf{r} + \mathbf{a}$ | Momentum $\mathbf{p}$ |
| Rotation | $\mathbf{r} \to R\mathbf{r}$ | Angular momentum $\mathbf{L}$ |
| Boost | $\mathbf{v} \to \mathbf{v} + \mathbf{u}$ | Center of mass |

---

## 3. Examples

### Free Particle

$L = \frac{1}{2}m\dot{x}^2$

- Translation invariant → momentum conserved
- Time independent → energy conserved

### Central Force

$L = \frac{1}{2}m(\dot{r}^2 + r^2\dot{\theta}^2) - V(r)$

- Rotation invariant → angular momentum conserved

---

## 4. QM Connection

### Symmetry Generators

In QM, symmetry generators are conserved observables:
$$[\hat{H}, \hat{G}] = 0 \implies \frac{d\langle G \rangle}{dt} = 0$$

| Classical | Quantum |
|-----------|---------|
| $\{H, G\} = 0$ | $[\hat{H}, \hat{G}] = 0$ |

---

## 5. Computational Lab

```python
"""
Day 319: Symmetry and Conservation
"""

import numpy as np
import matplotlib.pyplot as plt

def conservation_demo():
    """Demonstrate conservation laws in central force."""
    # Central force: F = -k/r²
    k = 1
    r0, v0 = 1, 0.8

    def simulate(dt=0.01, steps=1000):
        x, y = r0, 0
        vx, vy = 0, v0
        trajectory = []
        E_list, L_list = [], []

        for _ in range(steps):
            r = np.sqrt(x**2 + y**2)
            ax = -k * x / r**3
            ay = -k * y / r**3

            x += vx * dt
            y += vy * dt
            vx += ax * dt
            vy += ay * dt

            E = 0.5*(vx**2 + vy**2) - k/r
            L = x*vy - y*vx
            trajectory.append([x, y])
            E_list.append(E)
            L_list.append(L)

        return np.array(trajectory), E_list, L_list

    traj, E, L = simulate()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(traj[:, 0], traj[:, 1])
    axes[0].set_title('Orbit')
    axes[0].axis('equal')

    axes[1].plot(E)
    axes[1].set_title('Energy (conserved)')
    axes[1].set_ylim([min(E)-0.01, max(E)+0.01])

    axes[2].plot(L)
    axes[2].set_title('Angular Momentum (conserved)')
    axes[2].set_ylim([min(L)-0.01, max(L)+0.01])

    plt.savefig('conservation_laws.png', dpi=150)
    plt.close()
    print("Saved: conservation_laws.png")

if __name__ == "__main__":
    conservation_demo()
```

---

## Summary

### Noether's Theorem

$$\boxed{\text{Continuous Symmetry} \leftrightarrow \text{Conservation Law}}$$

---

## Preview: Day 320

Tomorrow: **Phase Space and Poisson Brackets** — the geometric structure of mechanics.
