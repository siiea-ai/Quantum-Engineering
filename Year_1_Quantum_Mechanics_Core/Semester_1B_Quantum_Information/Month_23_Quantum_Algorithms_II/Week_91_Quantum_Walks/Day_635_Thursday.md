# Day 635: Quantum Walk Search

## Overview
**Day 635** | Week 91, Day 5 | Year 1, Month 23 | Quantum Walks

Today we apply quantum walks to the search problem, achieving Grover-like speedups through a different algorithmic paradigm.

---

## Learning Objectives

1. Formulate search as quantum walk problem
2. Understand the marked vertex Hamiltonian
3. Analyze search on complete graphs
4. Study search on grids and other structures
5. Compare to Grover's algorithm
6. Understand when quantum walk search is advantageous

---

## Core Content

### Quantum Walk Search Setup

**Problem:** Find marked vertex $w$ on graph $G$ with $N$ vertices.

**Hamiltonian approach:**
$$H = -\gamma L - |w\rangle\langle w|$$

where $L$ is graph Laplacian and the second term marks the target.

### Search on Complete Graph

For $K_N$:
$$H = -\gamma(|s\rangle\langle s| \cdot N - I) - |w\rangle\langle w|$$

where $|s\rangle = \frac{1}{\sqrt{N}}\sum_v|v\rangle$.

**Search time:** $t^* = O(\sqrt{N})$ — same as Grover!

**Success probability:** Approaches 1 at optimal time.

### Childs-Goldstone Analysis

**Key insight:** The walk effectively occurs in a 2D subspace:
- $|w\rangle$: marked vertex
- $|s'\rangle$: uniform over unmarked vertices

The dynamics reduce to 2-level system, similar to Grover geometry.

### Search on 2D Grid

For $\sqrt{N} \times \sqrt{N}$ grid:

**Classical random walk:** $O(N \log N)$ hitting time

**Quantum walk search:** $O(\sqrt{N \log N})$

**Speedup:** $\sqrt{N / \log N}$ — nearly quadratic!

### Discrete-Time Walk Search

Using coined walk with modified coin at marked vertex:

**Walk operator:**
$$W = S \cdot (C' \otimes |w\rangle\langle w| + C \otimes (I - |w\rangle\langle w|))$$

where $C' \neq C$ breaks symmetry at marked vertex.

Common choice: $C' = -I$ (flip all phases at target).

### Algorithm Summary

1. Initialize in uniform superposition
2. Evolve with search Hamiltonian for time $t^*$
3. Measure position
4. Verify if marked (repeat if needed)

---

## Worked Examples

### Example 1: Complete Graph Search
Analyze quantum walk search on $K_{16}$.

**Solution:**
$N = 16$, optimal time $t^* \approx \pi\sqrt{N}/2 \approx 6.3$

Hamiltonian in 2D subspace gives Rabi-like oscillation.

Success probability at $t^*$: $P \approx 0.5$ (can be boosted with amplitude amplification).

---

## Computational Lab

```python
"""Day 635: Quantum Walk Search"""
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def search_hamiltonian_complete(N, marked, gamma=1):
    """Search Hamiltonian for complete graph."""
    # Graph Laplacian for complete graph
    L = N * np.eye(N) - np.ones((N, N))

    # Marking term
    M = np.zeros((N, N))
    M[marked, marked] = 1

    return -gamma * L / N - M

def quantum_walk_search(N, marked, t):
    """Simulate quantum walk search."""
    H = search_hamiltonian_complete(N, marked)
    U = expm(-1j * H * t)

    # Initial uniform state
    psi_0 = np.ones(N) / np.sqrt(N)
    psi_t = U @ psi_0

    return np.abs(psi_t[marked])**2

# Search on complete graph
N = 64
marked = 0
times = np.linspace(0, 2*np.pi*np.sqrt(N), 200)
probs = [quantum_walk_search(N, marked, t) for t in times]

plt.figure(figsize=(10, 6))
plt.plot(times, probs, 'b-', linewidth=2)
plt.axhline(y=1/N, color='red', linestyle='--', label='Initial prob')
plt.xlabel('Time')
plt.ylabel('P(marked)')
plt.title(f'Quantum Walk Search on K_{N}')
plt.legend()
plt.grid(True, alpha=0.3)

# Find optimal time
t_opt = times[np.argmax(probs)]
p_max = max(probs)
print(f"Optimal time: {t_opt:.2f}, P_max = {p_max:.4f}")
print(f"Theoretical: t* ≈ π√N/2 = {np.pi*np.sqrt(N)/2:.2f}")

plt.savefig('walk_search.png', dpi=150)
plt.show()
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Search Hamiltonian | $H = -\gamma L - \|w\rangle\langle w\|$ |
| Complete graph time | $t^* = O(\sqrt{N})$ |
| Grid search time | $O(\sqrt{N \log N})$ |

---

## Daily Checklist

- [ ] I understand quantum walk search formulation
- [ ] I know the Hamiltonian construction
- [ ] I can analyze search on complete graphs
- [ ] I understand the 2D subspace reduction
- [ ] I ran the computational lab

---

*Next: Day 636 — Graph Problems*
