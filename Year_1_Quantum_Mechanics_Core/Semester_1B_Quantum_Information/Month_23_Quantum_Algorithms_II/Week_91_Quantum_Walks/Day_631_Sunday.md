# Day 631: Classical Random Walks Review

## Overview
**Day 631** | Week 91, Day 1 | Year 1, Month 23 | Quantum Walks

Today we review classical random walks on graphs, establishing the foundation for understanding quantum walks and appreciating the quantum speedups we'll see later this week.

---

## Learning Objectives

1. Define random walks on graphs mathematically
2. Understand Markov chain formulation
3. Calculate stationary distributions
4. Analyze hitting times and mixing times
5. Connect to classical algorithms
6. Prepare for quantum generalizations

---

## Core Content

### Random Walks on Graphs

**Definition:** A random walk on graph $G = (V, E)$ is a sequence of vertices where each step moves to a random neighbor.

**Transition probability:** For regular graph (degree $d$):
$$P(v \to u) = \frac{1}{d} \text{ if } (v,u) \in E$$

### Markov Chain Formulation

The random walk is described by a **stochastic matrix** $P$ where:
$$P_{uv} = P(\text{move from } v \text{ to } u)$$

**Properties:**
- $P_{uv} \geq 0$ for all $u, v$
- $\sum_u P_{uv} = 1$ for all $v$

**Probability evolution:**
$$|\pi_{t+1}\rangle = P|\pi_t\rangle$$

### Examples of Random Walks

**1. Walk on a line (1D):**
$$P = \frac{1}{2}\begin{pmatrix} 0 & 1 & 0 & \cdots \\ 1 & 0 & 1 & \cdots \\ 0 & 1 & 0 & \cdots \\ \vdots & & & \ddots \end{pmatrix}$$

**2. Walk on a complete graph $K_n$:**
$$P_{uv} = \frac{1}{n-1} \text{ for } u \neq v$$

**3. Walk on a cycle $C_n$:**
Move left or right with probability 1/2 each.

### Stationary Distribution

A distribution $|\pi^*\rangle$ is **stationary** if:
$$P|\pi^*\rangle = |\pi^*\rangle$$

For connected, non-bipartite graphs, the stationary distribution exists and is unique.

**For regular graphs:** $\pi^*_v = 1/|V|$ (uniform distribution)

**For general graphs:** $\pi^*_v \propto \deg(v)$

### Hitting Time

**Definition:** The hitting time $h_{uv}$ is the expected number of steps to reach $v$ starting from $u$.

For random walk on complete graph $K_n$:
$$h_{uv} = n - 1$$

**Relationship to cover time:** Expected time to visit all vertices.

### Mixing Time

**Definition:** The mixing time $t_{mix}$ is the number of steps until the distribution is close to stationary:
$$t_{mix} = \min\{t : \|P^t|\pi_0\rangle - |\pi^*\rangle\|_{TV} \leq \epsilon\}$$

where $\|\cdot\|_{TV}$ is total variation distance.

**Spectral gap:** If $P$ has eigenvalues $1 = \lambda_1 \geq \lambda_2 \geq ... \geq \lambda_n \geq -1$:
$$t_{mix} = O\left(\frac{1}{1 - \lambda_2}\right)$$

### Classical Walk for Search

**Problem:** Find a marked vertex in graph with $N$ vertices.

**Random walk search:**
1. Start at random vertex
2. Walk randomly until hitting marked vertex
3. Expected time: $O(\text{hitting time})$

For complete graph: $O(N)$ steps (just random sampling!)
For 2D grid: $O(N \log N)$ steps

---

## Worked Examples

### Example 1: Random Walk on a Triangle
Analyze random walk on $K_3$ (triangle graph).

**Solution:**
Transition matrix:
$$P = \frac{1}{2}\begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}$$

Eigenvalues: $\lambda_1 = 1$, $\lambda_2 = \lambda_3 = -1/2$

Stationary distribution: $\pi^* = (1/3, 1/3, 1/3)$

Spectral gap: $1 - (-1/2) = 3/2$? No, we use $1 - |\lambda_2| = 1/2$.

### Example 2: Walk on a Line
For a line of $N$ nodes, starting at one end, expected steps to reach other end?

**Solution:**
This is the "gambler's ruin" problem.

Expected hitting time: $O(N^2)$

The variance of position grows as $\sqrt{t}$, so reaching distance $N$ requires $t \sim N^2$.

### Example 3: Random Walk on 2D Grid
For $\sqrt{N} \times \sqrt{N}$ grid, analyze hitting time to a corner.

**Solution:**
Random walk on 2D grid is recurrent (returns to origin infinitely often).

Hitting time from center to corner: $O(N \log N)$ expected steps.

---

## Practice Problems

### Problem 1: Transition Matrix
Write the transition matrix for random walk on a 4-cycle.

### Problem 2: Stationary Distribution
For a graph where vertex $i$ has degree $d_i$, prove that $\pi^*_i = d_i / (2|E|)$ is stationary.

### Problem 3: Eigenvalue Analysis
For the complete graph $K_n$, find all eigenvalues of $P$ and verify the mixing time bound.

---

## Computational Lab

```python
"""Day 631: Classical Random Walks Review"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def transition_matrix_complete_graph(n):
    """Transition matrix for complete graph K_n."""
    P = np.ones((n, n)) / (n - 1)
    np.fill_diagonal(P, 0)
    return P

def transition_matrix_cycle(n):
    """Transition matrix for cycle C_n."""
    P = np.zeros((n, n))
    for i in range(n):
        P[(i-1) % n, i] = 0.5
        P[(i+1) % n, i] = 0.5
    return P

def transition_matrix_line(n):
    """Transition matrix for path graph P_n."""
    P = np.zeros((n, n))
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n - 1:
            neighbors.append(i + 1)
        for j in neighbors:
            P[j, i] = 1 / len(neighbors)
    return P

def analyze_walk(P, name):
    """Analyze random walk given transition matrix."""
    n = P.shape[0]

    # Eigenvalues
    eigenvalues, eigenvectors = eig(P)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Stationary distribution
    pi_star = np.ones(n) / n  # Assuming regular graph

    print(f"\n{name}:")
    print(f"  Size: {n} vertices")
    print(f"  Largest eigenvalues: {eigenvalues[:4]}")
    print(f"  Spectral gap: {1 - abs(eigenvalues[1]):.4f}")

    return eigenvalues

def simulate_random_walk(P, start, steps):
    """Simulate a random walk."""
    n = P.shape[0]
    path = [start]
    current = start

    for _ in range(steps):
        probs = P[:, current]
        current = np.random.choice(n, p=probs)
        path.append(current)

    return path

def hitting_time_simulation(P, start, target, trials=1000):
    """Estimate hitting time from start to target."""
    n = P.shape[0]
    hitting_times = []

    for _ in range(trials):
        current = start
        steps = 0
        while current != target and steps < 10 * n**2:
            probs = P[:, current]
            current = np.random.choice(n, p=probs)
            steps += 1

        if current == target:
            hitting_times.append(steps)

    return np.mean(hitting_times), np.std(hitting_times)

def mixing_time_experiment(P, epsilon=0.1):
    """Estimate mixing time."""
    n = P.shape[0]
    pi_star = np.ones(n) / n  # Uniform stationary

    # Start from corner (vertex 0)
    pi = np.zeros(n)
    pi[0] = 1

    t = 0
    max_t = 1000 * n

    while t < max_t:
        pi = P @ pi
        t += 1

        # Total variation distance
        tv_distance = 0.5 * np.sum(np.abs(pi - pi_star))

        if tv_distance < epsilon:
            return t

    return max_t  # Didn't mix

def visualize_walk_distribution(P, steps_list, start=0):
    """Visualize probability distribution evolution."""
    n = P.shape[0]

    fig, axes = plt.subplots(1, len(steps_list), figsize=(15, 4))

    for idx, steps in enumerate(steps_list):
        P_t = np.linalg.matrix_power(P, steps)
        pi = P_t[:, start]

        axes[idx].bar(range(n), pi, color='blue', alpha=0.7)
        axes[idx].axhline(y=1/n, color='red', linestyle='--', alpha=0.5)
        axes[idx].set_xlabel('Vertex')
        axes[idx].set_ylabel('Probability')
        axes[idx].set_title(f't = {steps}')
        axes[idx].set_ylim(0, max(0.5, 1.5/n))

    plt.suptitle('Random Walk Distribution Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig('classical_walk_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
print("="*60)
print("Classical Random Walks Review")
print("="*60)

# Analyze different graphs
print("\n1. EIGENVALUE ANALYSIS")
print("-"*50)

n = 10
graphs = [
    (transition_matrix_complete_graph(n), f"Complete Graph K_{n}"),
    (transition_matrix_cycle(n), f"Cycle C_{n}"),
    (transition_matrix_line(n), f"Path P_{n}"),
]

for P, name in graphs:
    analyze_walk(P, name)

# Hitting time simulation
print("\n2. HITTING TIME SIMULATION")
print("-"*50)

for n in [10, 20, 50]:
    P_complete = transition_matrix_complete_graph(n)
    P_cycle = transition_matrix_cycle(n)

    ht_complete, std_complete = hitting_time_simulation(P_complete, 0, n-1, 500)
    ht_cycle, std_cycle = hitting_time_simulation(P_cycle, 0, n//2, 500)

    print(f"n = {n}:")
    print(f"  Complete graph: {ht_complete:.1f} +/- {std_complete:.1f} (theory: {n-1})")
    print(f"  Cycle: {ht_cycle:.1f} +/- {std_cycle:.1f} (theory: O({n**2 // 4}))")

# Mixing time
print("\n3. MIXING TIME")
print("-"*50)

for n in [10, 20, 50]:
    P_complete = transition_matrix_complete_graph(n)
    t_mix = mixing_time_experiment(P_complete)
    print(f"Complete graph K_{n}: t_mix ≈ {t_mix}")

# Visualization
print("\n4. DISTRIBUTION EVOLUTION")
print("-"*50)
P_cycle = transition_matrix_cycle(20)
visualize_walk_distribution(P_cycle, [0, 5, 20, 100], start=0)

# Compare classical walk search
print("\n5. CLASSICAL SEARCH COMPARISON")
print("-"*60)
print(f"{'Graph':^20} | {'Search Time':^20} | {'Speedup Potential'}")
print("-"*60)
print(f"{'Complete K_N':^20} | {'O(N)':^20} | {'O(√N) quantum'}")
print(f"{'2D Grid':^20} | {'O(N log N)':^20} | {'O(√N log N) quantum'}")
print(f"{'Line P_N':^20} | {'O(N²)':^20} | {'O(N) quantum'}")
```

---

## Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Probability evolution | $\|\pi_{t+1}\rangle = P\|\pi_t\rangle$ |
| Stationary condition | $P\|\pi^*\rangle = \|\pi^*\rangle$ |
| Mixing time | $t_{mix} = O(1/(1-\|\lambda_2\|))$ |
| Line hitting time | $O(N^2)$ |
| Grid hitting time | $O(N \log N)$ |

### Key Takeaways

1. **Random walks** are described by Markov chains
2. **Stationary distribution** depends on graph structure
3. **Mixing time** determined by spectral gap
4. **Hitting times** can be very long (quadratic for line)
5. **Classical search** efficiency depends on graph
6. **Quantum walks** will improve these bounds

---

## Daily Checklist

- [ ] I can construct transition matrices for graphs
- [ ] I understand stationary distributions
- [ ] I can calculate hitting times
- [ ] I know how spectral gap affects mixing
- [ ] I see where quantum speedups will apply
- [ ] I ran the computational lab and analyzed walks

---

*Next: Day 632 — Discrete-Time Quantum Walks*
