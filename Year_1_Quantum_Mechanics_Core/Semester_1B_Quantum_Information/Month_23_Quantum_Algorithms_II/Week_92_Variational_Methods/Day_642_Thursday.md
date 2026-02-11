# Day 642: Optimization Landscapes

## Overview
**Day 642** | Week 92, Day 5 | Year 1, Month 23 | Variational Methods

Today we study how to navigate the optimization landscape of variational quantum algorithms, including gradient computation and the parameter shift rule.

---

## Learning Objectives

1. Understand VQA optimization landscapes
2. Compute gradients using parameter shift rule
3. Compare gradient-free vs gradient-based methods
4. Analyze local minima and convergence
5. Implement gradient descent for VQAs
6. Understand the role of initialization

---

## Core Content

### The Optimization Landscape

For cost function $C(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$:

**Challenges:**
- Non-convex landscape
- Many local minima
- Barren plateaus (flat regions)
- High dimensionality

### The Parameter Shift Rule

For gates of the form $U(\theta) = e^{-i\theta G/2}$ where $G^2 = I$:

$$\boxed{\frac{\partial C}{\partial \theta} = \frac{C(\theta + \pi/2) - C(\theta - \pi/2)}{2}}$$

**Derivation:** Uses that $e^{-i(\theta \pm \pi/2)G/2} = e^{\mp i\pi G/4}e^{-i\theta G/2}$

### Gradient-Based Optimization

**Gradient descent:**
$$\theta_{t+1} = \theta_t - \eta \nabla C(\theta_t)$$

**Variants:**
- Adam: Adaptive learning rate
- SPSA: Simultaneous perturbation (2 evaluations)
- Natural gradient: Uses quantum Fisher information

### Gradient-Free Methods

**COBYLA:** Constrained optimization
**Nelder-Mead:** Simplex method
**Powell:** Direction set method

Often preferred for noisy landscapes.

### Quantum Natural Gradient

$$\theta_{t+1} = \theta_t - \eta F^{-1}(\theta_t) \nabla C(\theta_t)$$

where $F$ is the quantum Fisher information matrix:

$$F_{ij} = \text{Re}[\langle\partial_i\psi|\partial_j\psi\rangle - \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle]$$

More efficient convergence but higher cost.

### Initialization Strategies

1. **Random:** Simple but may land in barren plateau
2. **Identity:** Start close to identity circuit
3. **Layer-wise:** Train one layer at a time
4. **Classical pre-training:** Use classical approximation

---

## Computational Lab

```python
"""Day 642: Optimization Landscapes"""
import numpy as np
import matplotlib.pyplot as plt

def parameter_shift_gradient(cost_func, params, idx, shift=np.pi/2):
    """Compute gradient using parameter shift rule."""
    params_plus = params.copy()
    params_minus = params.copy()
    params_plus[idx] += shift
    params_minus[idx] -= shift

    return (cost_func(params_plus) - cost_func(params_minus)) / 2

def full_gradient(cost_func, params):
    """Compute full gradient vector."""
    return np.array([parameter_shift_gradient(cost_func, params, i)
                     for i in range(len(params))])

def gradient_descent_vqa(cost_func, initial_params, lr=0.1, max_iter=100):
    """Gradient descent optimization for VQA."""
    params = initial_params.copy()
    history = [cost_func(params)]

    for _ in range(max_iter):
        grad = full_gradient(cost_func, params)
        params = params - lr * grad
        history.append(cost_func(params))

    return params, history

# Example: 2-parameter cost function
def example_cost(params):
    """Example VQA cost function."""
    theta1, theta2 = params
    return np.sin(theta1)**2 + 0.5 * np.sin(theta2)**2 + 0.3 * np.cos(theta1 - theta2)

# Landscape visualization
theta1_range = np.linspace(0, 2*np.pi, 100)
theta2_range = np.linspace(0, 2*np.pi, 100)
T1, T2 = np.meshgrid(theta1_range, theta2_range)
Z = np.zeros_like(T1)
for i in range(100):
    for j in range(100):
        Z[i, j] = example_cost([T1[i,j], T2[i,j]])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(T1, T2, Z, levels=30, cmap='viridis')
plt.colorbar(label='Cost')
plt.xlabel('θ₁', fontsize=12)
plt.ylabel('θ₂', fontsize=12)
plt.title('VQA Cost Landscape', fontsize=14)

# Run optimization
initial = np.array([0.5, 1.5])
final_params, history = gradient_descent_vqa(example_cost, initial, lr=0.2)

plt.subplot(1, 2, 2)
plt.plot(history, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.title('Gradient Descent Convergence', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_landscape.png', dpi=150)
plt.show()

print(f"Initial cost: {history[0]:.4f}")
print(f"Final cost: {history[-1]:.4f}")
print(f"Optimal params: {final_params}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Parameter shift | $\partial C/\partial\theta = [C(\theta+\pi/2) - C(\theta-\pi/2)]/2$ |
| Gradient descent | $\theta_{t+1} = \theta_t - \eta\nabla C$ |
| Natural gradient | Uses Fisher information matrix |

---

## Daily Checklist

- [ ] I understand VQA optimization landscapes
- [ ] I can compute gradients with parameter shift
- [ ] I know gradient-free alternatives
- [ ] I understand initialization strategies
- [ ] I ran the computational lab

---

*Next: Day 643 — Barren Plateaus*
