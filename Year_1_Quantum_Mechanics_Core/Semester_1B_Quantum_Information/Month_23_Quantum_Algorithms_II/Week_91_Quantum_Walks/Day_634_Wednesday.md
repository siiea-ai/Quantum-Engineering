# Day 634: Continuous-Time Quantum Walks

## Overview
**Day 634** | Week 91, Day 4 | Year 1, Month 23 | Quantum Walks

Today we study continuous-time quantum walks, which evolve according to the Schrodinger equation with the graph Laplacian or adjacency matrix as Hamiltonian.

---

## Learning Objectives

1. Define continuous-time quantum walks
2. Use adjacency matrix as Hamiltonian
3. Analyze walk dynamics on simple graphs
4. Compare to discrete-time walks
5. Study perfect state transfer
6. Connect to quantum simulation

---

## Core Content

### Continuous-Time Quantum Walk

**Definition:** Evolution under graph Hamiltonian:
$$|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle$$

**Hamiltonian choices:**
1. Adjacency matrix: $H = \gamma A$
2. Graph Laplacian: $H = \gamma L = \gamma(D - A)$

where $D$ is the degree matrix.

### Adjacency Matrix Walk

$$U(t) = e^{-i\gamma A t}$$

For simple graphs, can compute via eigendecomposition:
$$A = \sum_k \lambda_k |v_k\rangle\langle v_k|$$
$$U(t) = \sum_k e^{-i\gamma\lambda_k t}|v_k\rangle\langle v_k|$$

### Walk on Complete Graph

For $K_n$: $A = J - I$ where $J$ is all-ones matrix.

Eigenvalues: $\lambda_1 = n-1$ (uniform vector), $\lambda_{2...n} = -1$

$$U(t) = e^{i\gamma t}\left[\frac{1}{n}J + (1 - \frac{1}{n})e^{-in\gamma t}(I - \frac{1}{n}J)\right]$$

### Perfect State Transfer

**Definition:** $|U(t)_{jk}| = 1$ for some $t$.

Complete transfer from vertex $j$ to vertex $k$ at time $t$.

**Example:** Path graph $P_2$:
$$A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$
$$U(t) = \cos t \cdot I - i\sin t \cdot A$$

At $t = \pi/2$: $U = -iA$, giving perfect transfer!

### Mixing Properties

Unlike discrete walks, continuous walks on cycles:
- Don't mix well (small spectral gaps)
- But hypercubes give good mixing

---

## Worked Examples

### Example 1: Walk on Path $P_3$
Compute $U(t)$ for path of length 3.

**Solution:**
$$A = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

Eigenvalues: $\lambda = 0, \pm\sqrt{2}$

$U(t) = \sum_k e^{-i\lambda_k t}|v_k\rangle\langle v_k|$

---

## Computational Lab

```python
"""Day 634: Continuous-Time Quantum Walks"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def adjacency_path(n):
    """Adjacency matrix for path graph."""
    A = np.zeros((n, n))
    for i in range(n-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    return A

def adjacency_complete(n):
    """Adjacency matrix for complete graph."""
    return np.ones((n, n)) - np.eye(n)

def continuous_walk(A, t, gamma=1):
    """Compute U(t) = exp(-i*gamma*A*t)."""
    return expm(-1j * gamma * A * t)

def walk_evolution(A, initial, times, gamma=1):
    """Simulate walk evolution."""
    n = A.shape[0]
    probs = []

    for t in times:
        U = continuous_walk(A, t, gamma)
        state = U @ initial
        probs.append(np.abs(state)**2)

    return np.array(probs)

# Perfect state transfer on P_2
print("Perfect State Transfer on P_2:")
A = adjacency_path(2)
for t in [0, np.pi/4, np.pi/2]:
    U = continuous_walk(A, t)
    print(f"t = {t:.4f}: |U[0,1]|² = {abs(U[0,1])**2:.4f}")

# Visualize walk on complete graph
n = 5
A = adjacency_complete(n)
initial = np.zeros(n)
initial[0] = 1

times = np.linspace(0, 2*np.pi, 100)
probs = walk_evolution(A, initial, times)

plt.figure(figsize=(10, 6))
for v in range(n):
    plt.plot(times, probs[:, v], label=f'Vertex {v}')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title(f'Continuous Walk on Complete Graph K_{n}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('continuous_walk.png', dpi=150)
plt.show()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Evolution | $U(t) = e^{-i\gamma At}$ |
| Eigendecomposition | $U(t) = \sum_k e^{-i\gamma\lambda_k t}\|v_k\rangle\langle v_k\|$ |
| Perfect transfer | $\|U(t)_{jk}\| = 1$ |

---

## Daily Checklist

- [ ] I understand continuous-time walks
- [ ] I can use adjacency matrix as Hamiltonian
- [ ] I know about perfect state transfer
- [ ] I can compare to discrete walks
- [ ] I ran the computational lab

---

*Next: Day 635 — Quantum Walk Search*
